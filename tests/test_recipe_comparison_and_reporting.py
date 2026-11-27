"""End-to-end tests for Multi-Recipe Comparison and Multi-Format Report Export (Issue 07)."""

from __future__ import annotations

import csv
import io
import json
import unittest

import plotly.graph_objects as go

from corrosion_simulator.database import HarmonizedDatabase
from corrosion_simulator.ml_predictor import HybridMLPredictor
from corrosion_simulator.platform_controller import (
    CorrosionPlatformController,
    SimulationRequest,
)
from corrosion_simulator.reporting import (
    CleaningRecipeSpec,
    RecipeComparisonPayload,
    compare_cleaning_recipes,
    export_report_csv,
    export_report_html,
    export_report_json,
    export_report_pdf,
    export_report_png,
    export_report_svg,
    generate_all_report_exports,
)


class TestRecipeComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db = HarmonizedDatabase.from_seed()
        cls.predictor = HybridMLPredictor(model_family="random_forest", random_state=42)
        cls.predictor.fit(cls.db)
        cls.controller = CorrosionPlatformController(cls.db, cls.predictor)

    def _sample_recipes(self) -> list[CleaningRecipeSpec]:
        return [
            CleaningRecipeSpec(
                name="Aggressive High-H2O2 5min",
                oxidizer_type="H2O2",
                oxidizer_concentration=5.0,
                acid_type="Citric acid",
                acid_concentration=1.0,
                pH=3.0,
                temperature_C=45.0,
                immersion_time_s=600.0,
            ),
            CleaningRecipeSpec(
                name="Mild Short Rinse 30s",
                oxidizer_type="H2O2",
                oxidizer_concentration=0.5,
                acid_type="Citric acid",
                acid_concentration=0.5,
                inhibitor_type="BTA",
                inhibitor_concentration=1.0,
                pH=5.0,
                temperature_C=25.0,
                immersion_time_s=30.0,
            ),
            CleaningRecipeSpec(
                name="Baseline Process 180s",
                oxidizer_type="H2O2",
                oxidizer_concentration=1.0,
                acid_type="Citric acid",
                acid_concentration=1.0,
                pH=3.5,
                temperature_C=25.0,
                immersion_time_s=180.0,
            ),
        ]

    def test_compare_recipes_enforces_2_to_5_bounds(self) -> None:
        base_req = SimulationRequest(primary_metal="Cu", parameter_source="ml", mc_draws=100)
        recipes = self._sample_recipes()
        with self.assertRaises(ValueError):
            compare_cleaning_recipes(self.controller, base_req, recipes[:1])
        too_many = [
            CleaningRecipeSpec(name=f"R{i}", pH=3.0 + i * 0.5, temperature_C=25.0, immersion_time_s=60.0)
            for i in range(6)
        ]
        with self.assertRaises(ValueError):
            compare_cleaning_recipes(self.controller, base_req, too_many)

    def test_single_metal_recipe_comparison_ranks_lowest_risk_first(self) -> None:
        base_req = SimulationRequest(
            primary_metal="Cu",
            secondary_metal=None,
            d_crit_nm=1.0,
            kinetics_model="linear",
            parameter_source="ml",
            mc_draws=100,
        )
        result = compare_cleaning_recipes(self.controller, base_req, self._sample_recipes())
        self.assertIsInstance(result, RecipeComparisonPayload)
        self.assertEqual(len(result.entries), 3)
        self.assertEqual(result.mode, "single_metal")

        # Entries must be strictly ordered by ascending risk_score (tie-broken by depth, CR0)
        risk_scores = [e.risk_score for e in result.entries]
        self.assertEqual(risk_scores, sorted(risk_scores))
        self.assertEqual(result.recommended_recipe, result.entries[0].recipe.name)
        self.assertIn(result.recommended_recipe, result.recommended_rationale)

        # Table and Plotly bar (includes Time to Threshold) + radar figures
        self.assertEqual(len(result.comparison_table), 3)
        self.assertIn("Thickness Loss Risk Score", result.comparison_table.columns)
        self.assertIsInstance(result.bar_figure, go.Figure)
        self.assertGreaterEqual(len(result.bar_figure.data), 4)
        self.assertIn("Time to Threshold (s)", [t.name for t in result.bar_figure.data])
        self.assertIsInstance(result.radar_figure, go.Figure)
        self.assertEqual(len(result.radar_figure.data), 3)

    def test_galvanic_recipe_comparison_includes_acceleration_factor(self) -> None:
        base_req = SimulationRequest(
            primary_metal="Cu",
            secondary_metal="Co",
            cathode_to_anode_area_ratio=2.0,
            d_crit_nm=1.0,
            kinetics_model="linear",
            parameter_source="ml",
            mc_draws=100,
        )
        result = compare_cleaning_recipes(self.controller, base_req, self._sample_recipes()[:2])
        self.assertEqual(result.mode, "galvanic")
        self.assertIn("Acceleration Factor", result.comparison_table.columns)
        for entry in result.entries:
            self.assertIsNotNone(entry.acceleration_factor)
            self.assertGreaterEqual(entry.acceleration_factor, 1.0)
        self.assertIsInstance(result.bar_figure, go.Figure)
        self.assertIsInstance(result.radar_figure, go.Figure)

    def test_cross_regime_recipe_comparison_includes_solvent_metadata(self) -> None:
        base_req = SimulationRequest(
            primary_metal="Cu",
            d_crit_nm=1.0,
            kinetics_model="linear",
            parameter_source="ml",
            mc_draws=80,
        )
        cross_recipes = [
            CleaningRecipeSpec(
                name="Aqueous Citric Clean",
                oxidizer_type="H2O2",
                oxidizer_concentration=1.0,
                acid_type="Citric acid",
                acid_concentration=1.0,
                pH=3.5,
                temperature_C=25.0,
                immersion_time_s=180.0,
                solvent_regime="aqueous",
                organic_solvent_type="None",
                water_content_wt_pct=100.0,
            ),
            CleaningRecipeSpec(
                name="Semi-Aqueous EG Clean",
                oxidizer_type="H2O2",
                oxidizer_concentration=0.5,
                pH=5.5,
                temperature_C=35.0,
                immersion_time_s=180.0,
                solvent_regime="semi_aqueous",
                organic_solvent_type="EG",
                water_content_wt_pct=40.0,
            ),
            CleaningRecipeSpec(
                name="Non-Aqueous DMSO Strip",
                oxidizer_type="None",
                oxidizer_concentration=0.0,
                pH=7.0,
                temperature_C=45.0,
                immersion_time_s=180.0,
                solvent_regime="non_aqueous",
                organic_solvent_type="DMSO",
                water_content_wt_pct=0.2,
            ),
        ]
        result = compare_cleaning_recipes(self.controller, base_req, cross_recipes)
        for col in ("Solvent Regime", "Solvent Subclass", "Co-Solvent", "Water (wt%)"):
            self.assertIn(col, result.comparison_table.columns)

        by_name = {item["recipe_name"]: item for item in result.to_summary_dict()["rankings"]}
        self.assertEqual(by_name["Aqueous Citric Clean"]["solvent_regime"], "aqueous")
        self.assertEqual(by_name["Aqueous Citric Clean"]["solvent_subclass"], "acidic")
        self.assertEqual(by_name["Semi-Aqueous EG Clean"]["solvent_regime"], "semi_aqueous")
        self.assertEqual(by_name["Semi-Aqueous EG Clean"]["solvent_subclass"], "glycol")
        self.assertEqual(by_name["Semi-Aqueous EG Clean"]["organic_solvent_type"], "EG")
        self.assertAlmostEqual(by_name["Semi-Aqueous EG Clean"]["water_content_wt_pct"], 40.0)
        self.assertEqual(by_name["Non-Aqueous DMSO Strip"]["solvent_regime"], "non_aqueous")
        self.assertEqual(by_name["Non-Aqueous DMSO Strip"]["solvent_subclass"], "polar_aprotic")
        self.assertTrue(
            any("aqueous Pourbaix" in w for w in by_name["Non-Aqueous DMSO Strip"]["warnings"])
        )


class TestMultiFormatReportExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db = HarmonizedDatabase.from_seed()
        cls.predictor = HybridMLPredictor(model_family="random_forest", random_state=42)
        cls.predictor.fit(cls.db)
        cls.controller = CorrosionPlatformController(cls.db, cls.predictor)

        cls.req = SimulationRequest(
            primary_metal="Cu",
            secondary_metal="Co",
            cathode_to_anode_area_ratio=1.5,
            oxidizer_type="H2O2",
            oxidizer_concentration=1.0,
            pH=3.0,
            temperature_C=25.0,
            immersion_time_s=300.0,
            d_crit_nm=1.0,
            kinetics_model="auto",
            parameter_source="ml",
            mc_draws=100,
        )
        cls.single_payload = cls.controller.run_single_metal_analysis(cls.req)
        cls.galvanic_payload = cls.controller.run_galvanic_analysis(cls.req)
        cls.comparison_payload = compare_cleaning_recipes(
            cls.controller,
            cls.req,
            [
                CleaningRecipeSpec(name="Recipe A", oxidizer_type="H2O2", oxidizer_concentration=1.0, pH=3.0, temperature_C=25.0, immersion_time_s=60.0),
                CleaningRecipeSpec(name="Recipe B", oxidizer_type="H2O2", oxidizer_concentration=3.0, pH=2.5, temperature_C=40.0, immersion_time_s=300.0),
            ],
        )

    def test_all_six_export_formats_are_valid_and_contain_required_metadata(self) -> None:
        exports = generate_all_report_exports(
            self.single_payload,
            galvanic_payload=self.galvanic_payload,
            comparison_payload=self.comparison_payload,
        )
        self.assertEqual(set(exports.keys()), {"html", "pdf", "csv", "json", "png", "svg"})

        # HTML: self-contained with provenance, warnings/confidence, MC uncertainty intervals, galvanic confidence, embedded image, comparison
        html_str = exports["html"].decode("utf-8")
        self.assertIn("<!DOCTYPE html>", html_str)
        self.assertIn("Confidence Tier", html_str)
        self.assertIn("Galvanic Confidence Tier", html_str)
        self.assertIn("Monte Carlo Uncertainty Intervals", html_str)
        self.assertIn("data:image/png;base64,", html_str)
        self.assertIn("Recipe Comparison", html_str)
        self.assertIn("ML", html_str)

        # JSON: valid JSON with inputs, single_metal, galvanic, recipe_comparison, uncertainty_intervals, warnings
        json_obj = json.loads(exports["json"].decode("utf-8"))
        self.assertIn("inputs", json_obj)
        self.assertEqual(json_obj["inputs"]["oxidizer_type"], "H2O2")
        self.assertIn("single_metal", json_obj)
        self.assertIn("galvanic", json_obj)
        self.assertIn("recipe_comparison", json_obj)
        self.assertIn("uncertainty_intervals", json_obj)
        self.assertIn("single_cr0_nm_min", json_obj["uncertainty_intervals"])
        self.assertEqual(json_obj["single_metal"]["parameter_source"], "ml")
        self.assertIn("warnings", json_obj)
        self.assertIn("recommended_recipe", json_obj["recipe_comparison"])

        # Composite Technical Report figure: unified dual-axis Depth & Risk legend + polarization V-bands
        from corrosion_simulator.reporting import _render_composite_figure

        fig = _render_composite_figure(
            self.single_payload,
            galvanic_payload=self.galvanic_payload,
            comparison_payload=self.comparison_payload,
        )
        ax_tafel = fig.axes[0]
        tafel_labels = [t.get_text() for t in ax_tafel.get_legend().get_texts()]
        self.assertIn("Net |i_net| 50% Band (p25–p75)", tafel_labels)
        self.assertIn("Net |i_net| 90% Band (p05–p95)", tafel_labels)
        ax_kin = fig.axes[1]
        kin_labels = [t.get_text() for t in ax_kin.get_legend().get_texts()]
        self.assertTrue(any("(left)" in lbl for lbl in kin_labels))
        self.assertIn("Risk R(t) (right)", kin_labels)

        # CSV: parseable CSV with inputs, uncertainty intervals, structured recipe comparison metrics, and provenance
        csv_text = exports["csv"].decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        self.assertGreaterEqual(len(rows), 2)
        sections = {r["section"] for r in rows}
        self.assertIn("Inputs", sections)
        self.assertIn("Single Metal", sections)
        self.assertIn("Uncertainty (90% CI)", sections)
        self.assertIn("Recipe Comparison", sections)
        cmp_metrics = {r["metric"] for r in rows if r["section"] == "Recipe Comparison"}
        self.assertIn("Time to Threshold", cmp_metrics)
        self.assertIn("Acceleration Factor", cmp_metrics)
        self.assertIn("Confidence Tier", cmp_metrics)

        # PDF: valid PDF header and non-trivial size
        pdf_bytes = exports["pdf"]
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))
        self.assertGreater(len(pdf_bytes), 2000)

        # PNG: valid PNG magic bytes
        png_bytes = exports["png"]
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(png_bytes), 2000)

        # SVG: valid SVG markup
        svg_str = exports["svg"].decode("utf-8")
        self.assertIn("<svg", svg_str)
        self.assertIn("</svg>", svg_str)

    def test_exported_reports_include_solvent_regime_and_water_metadata(self) -> None:
        import re
        import zlib

        semi_req = SimulationRequest(
            primary_metal="Cu",
            secondary_metal="Co",
            cathode_to_anode_area_ratio=1.0,
            oxidizer_type="H2O2",
            oxidizer_concentration=1.0,
            pH=4.5,
            temperature_C=30.0,
            solvent_regime="semi_aqueous",
            organic_solvent_type="EG",
            water_content_wt_pct=45.0,
            immersion_time_s=180.0,
            parameter_source="ml",
            mc_draws=80,
        )
        single_payload = self.controller.run_single_metal_analysis(semi_req)
        galvanic_payload = self.controller.run_galvanic_analysis(semi_req)
        comparison_payload = compare_cleaning_recipes(
            self.controller,
            semi_req,
            [
                CleaningRecipeSpec(
                    name="Semi-Aqueous EG",
                    solvent_regime="semi_aqueous",
                    organic_solvent_type="EG",
                    water_content_wt_pct=45.0,
                    pH=4.5,
                    temperature_C=30.0,
                    immersion_time_s=180.0,
                ),
                CleaningRecipeSpec(
                    name="Non-Aqueous DMSO",
                    solvent_regime="non_aqueous",
                    organic_solvent_type="DMSO",
                    water_content_wt_pct=0.5,
                    pH=7.0,
                    temperature_C=30.0,
                    immersion_time_s=180.0,
                ),
            ],
        )

        exports = generate_all_report_exports(
            single_payload,
            galvanic_payload=galvanic_payload,
            comparison_payload=comparison_payload,
        )

        # JSON checks
        json_obj = json.loads(exports["json"].decode("utf-8"))
        for section in (json_obj["inputs"], json_obj["single_metal"], json_obj["galvanic"]):
            self.assertEqual(section["solvent_regime"], "semi_aqueous")
            self.assertEqual(section["solvent_subclass"], "glycol")
            self.assertEqual(section["organic_solvent_type"], "EG")
            self.assertAlmostEqual(section["water_content_wt_pct"], 45.0)

        # HTML checks
        html_str = exports["html"].decode("utf-8")
        for token in (
            "solvent_regime=semi_aqueous",
            "solvent_subclass=glycol",
            "organic_solvent_type=EG",
            "water_content_wt_pct=45.00 wt%",
        ):
            self.assertIn(token, html_str)

        # CSV checks
        rows = list(csv.DictReader(io.StringIO(exports["csv"].decode("utf-8"))))
        input_metrics = {
            r["metric"]: r["value"] for r in rows if r["section"] == "Inputs"
        }
        self.assertEqual(input_metrics["solvent_regime"], "semi_aqueous")
        self.assertEqual(input_metrics["solvent_subclass"], "glycol")
        self.assertEqual(input_metrics["organic_solvent_type"], "EG")
        self.assertEqual(float(input_metrics["water_content_wt_pct"]), 45.0)

        # PDF checks: inspect both PDF metadata dictionary and decompressed page streams
        pdf_bytes = exports["pdf"]
        decompressed_parts: list[str] = [pdf_bytes.decode("latin-1", errors="ignore")]
        for match in re.finditer(rb"stream[\r\n]+(.*?)[\r\n]+endstream", pdf_bytes, re.DOTALL):
            try:
                decompressed_parts.append(
                    zlib.decompress(match.group(1)).decode("latin-1", errors="ignore")
                )
            except zlib.error:
                pass
        pdf_text = "\n".join(decompressed_parts)
        for token in ("semi_aqueous", "glycol", "EG", "45.00"):
            self.assertIn(token, pdf_text)

    def test_reports_record_calibration_mode_lab_project_and_residual_breakdown(self) -> None:
        from pathlib import Path
        import re
        import tempfile
        import zlib
        import pandas as pd

        from corrosion_simulator.build_artifacts import build_processed_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            raw_dir = data_dir / "raw"
            proc_dir = data_dir / "processed"
            lab_dir = data_dir / "lab" / "pilot_eg"
            raw_dir.mkdir(parents=True)
            lab_dir.mkdir(parents=True)

            pd.DataFrame(
                [
                    {
                        "record_id": f"R_{i}",
                        "metal": "Cu" if i % 2 == 0 else "Co",
                        "solution_id": f"SOL_{i % 2}",
                        "pH": 3.0 + (i % 4),
                        "temperature_C": 25.0 + (i % 2) * 10.0,
                        "oxidizer_type": "H2O2",
                        "oxidizer_concentration": 1.0,
                        "Ecorr_V_SHE": 0.25 if i % 2 == 0 else -0.10,
                        "icorr_uA_cm2": 2.5 + 0.3 * i,
                        "beta_a_mV_dec": 60.0,
                        "beta_c_mV_dec": 120.0,
                    }
                    for i in range(12)
                ]
            ).to_csv(raw_dir / "lit.csv", index=False)
            build_processed_artifacts(
                raw_dir=raw_dir,
                output_dir=proc_dir,
                models=["random_forest"],
                export_csv=False,
            )
            lab_rows = []
            for idx in range(5):
                for metal, ecorr, icorr in (("Cu", 0.38, 12.0), ("Co", -0.18, 22.0)):
                    lab_rows.append(
                        {
                            "record_id": f"L_{metal}_{idx}",
                            "metal": metal,
                            "solution_id": f"EG_{idx}",
                            "pH": 4.0 + idx * 0.5,
                            "temperature_C": 25.0 + idx * 5.0,
                            "oxidizer_type": "H2O2",
                            "oxidizer_concentration": 1.0,
                            "organic_solvent_type": "EG",
                            "water_content_wt_pct": 50.0,
                            "Ecorr_V_SHE": ecorr + 0.01 * idx,
                            "icorr_uA_cm2": icorr + idx,
                            "beta_a_mV_dec": 75.0,
                            "beta_c_mV_dec": 135.0,
                        }
                    )
            pd.DataFrame(lab_rows).to_csv(lab_dir / "pilot.csv", index=False)

            ctrl = CorrosionPlatformController.from_default_data(
                data_dir=data_dir,
                model_family="random_forest",
                auto_rebuild=False,
            )
            req = SimulationRequest(
                primary_metal="Cu",
                secondary_metal="Co",
                oxidizer_type="H2O2",
                oxidizer_concentration=1.0,
                pH=5.0,
                temperature_C=30.0,
                solvent_regime="semi_aqueous",
                organic_solvent_type="EG",
                water_content_wt_pct=50.0,
                parameter_source="ml",
                calibration_mode="delta_learning",
                lab_project="pilot_eg",
                mc_draws=80,
            )
            single_p = ctrl.run_single_metal_analysis(req)
            galv_p = ctrl.run_galvanic_analysis(req)
            exports = generate_all_report_exports(single_p, galvanic_payload=galv_p)

            json_obj = json.loads(exports["json"].decode("utf-8"))
            self.assertEqual(json_obj["inputs"]["calibration_mode"], "delta_learning")
            self.assertEqual(json_obj["inputs"]["lab_project"], "pilot_eg")
            self.assertIn("lab_calibration", json_obj)
            self.assertEqual(json_obj["lab_calibration"]["lab_project"], "pilot_eg")
            self.assertIn("single_metal_residual_breakdown", json_obj["lab_calibration"])
            self.assertIn("galvanic_residual_breakdown", json_obj["lab_calibration"])
            self.assertIn("applicability_domain", json_obj["single_metal"])
            self.assertEqual(
                json_obj["single_metal"]["applicability_domain"]["lab_anchoring_status"],
                "Lab-Anchored",
            )

            html_str = exports["html"].decode("utf-8")
            self.assertIn("calibration_mode=delta_learning", html_str)
            self.assertIn("pilot_eg", html_str)
            self.assertIn("Lab Calibration — Residual Breakdown", html_str)
            self.assertIn("GP Residual", html_str)

            csv_rows = list(csv.DictReader(io.StringIO(exports["csv"].decode("utf-8"))))
            input_map = {r["metric"]: r["value"] for r in csv_rows if r["section"] == "Inputs"}
            self.assertEqual(input_map["calibration_mode"], "delta_learning")
            self.assertEqual(input_map["lab_project"], "pilot_eg")
            rb_rows = [r for r in csv_rows if r["section"] == "Residual Breakdown"]
            self.assertGreaterEqual(len(rb_rows), 8)
            self.assertTrue(any("GP Residual" in r["metric"] for r in rb_rows))

            pdf_bytes = exports["pdf"]
            pdf_parts = [pdf_bytes.decode("latin-1", errors="ignore")]
            for match in re.finditer(rb"stream[\r\n]+(.*?)[\r\n]+endstream", pdf_bytes, re.DOTALL):
                try:
                    pdf_parts.append(zlib.decompress(match.group(1)).decode("latin-1", errors="ignore"))
                except zlib.error:
                    pass
            pdf_text = "\n".join(pdf_parts)
            for token in ("delta_learning", "pilot_eg", "RESIDUAL BREAKDOWN", "GP Residual"):
                self.assertIn(token, pdf_text)


if __name__ == "__main__":
    unittest.main()
