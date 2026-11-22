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

        # Table and Plotly bar + radar figures
        self.assertEqual(len(result.comparison_table), 3)
        self.assertIn("Thickness Loss Risk Score", result.comparison_table.columns)
        self.assertIsInstance(result.bar_figure, go.Figure)
        self.assertGreaterEqual(len(result.bar_figure.data), 2)
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

        # HTML: self-contained with provenance, warnings/confidence, embedded image, comparison
        html_str = exports["html"].decode("utf-8")
        self.assertIn("<!DOCTYPE html>", html_str)
        self.assertIn("Confidence Tier", html_str)
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

        # CSV: parseable CSV with inputs, uncertainty intervals, and provenance
        csv_text = exports["csv"].decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        self.assertGreaterEqual(len(rows), 2)
        sections = {r["section"] for r in rows}
        self.assertIn("Inputs", sections)
        self.assertIn("Single Metal", sections)
        self.assertIn("Uncertainty (90% CI)", sections)
        self.assertIn("Recipe Comparison", sections)

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


if __name__ == "__main__":
    unittest.main()
