"""Tests for data-driven material extensibility, graceful degradation, and offline artifact builds."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from corrosion_simulator.build_artifacts import (
    build_processed_artifacts,
    check_artifacts_freshness,
    load_manifest,
)
from corrosion_simulator.database import CORE_METALS, HarmonizedDatabase
from corrosion_simulator.ml_predictor import (
    ApplicabilityDomainChecker,
    ElectrochemicalFeatureExtractor,
    HybridMLPredictor,
    MLFeatureQuery,
)
from corrosion_simulator.physics import (
    calculate_faraday_corrosion_rate,
    has_faraday_properties,
)
from corrosion_simulator.platform_controller import (
    CorrosionPlatformController,
    SimulationRequest,
)
from corrosion_simulator.reporting import (
    CleaningRecipeSpec,
    compare_cleaning_recipes,
    generate_all_report_exports,
)


class TestDataDrivenMaterialCatalog(unittest.TestCase):
    def test_workspace_raw_ingests_extended_materials_and_quarantines_invalid_rows(self) -> None:
        db = HarmonizedDatabase.from_directory("data/raw")
        metals = db.available_metals
        # Core metals appear first in canonical order, followed by extended materials
        self.assertEqual(metals[: len(CORE_METALS)], list(CORE_METALS))
        self.assertIn("Fe", metals)
        self.assertIn("TiN", metals)
        self.assertIn("Ni", metals)
        self.assertGreater(len(db.rejected_records), 0)
        self.assertIn("rejection_reason", db.rejected_records.columns)

        # Opt-in core-only filter remains supported
        core_db = HarmonizedDatabase.from_directory("data/raw", filter_to_core_metals=True)
        self.assertEqual(set(core_db.available_metals), set(CORE_METALS))

    def test_default_does_not_inject_seed_records_into_subset_raw_data(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "record_id": "CUSTOM_1",
                    "metal": "Fe",
                    "solution_id": "SOL_FE",
                    "pH": 3.0,
                    "temperature_C": 25.0,
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 1.0,
                    "Ecorr_V_SHE": -0.42,
                    "icorr_uA_cm2": 15.0,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                }
            ]
        )
        db = HarmonizedDatabase.from_raw_tables(raw)
        self.assertEqual(db.available_metals, ["Fe"])
        # Pure-element fallback supplies Fe Faraday constants automatically
        self.assertTrue(has_faraday_properties("Fe", db.material_properties))
        rate = calculate_faraday_corrosion_rate("Fe", 15.0, material_properties=db.material_properties)
        self.assertGreater(rate.corrosion_rate_nm_min, 0.0)

    def test_alloy_and_compound_equivalent_weight_and_graceful_degradation(self) -> None:
        raw_echem = pd.DataFrame(
            [
                {
                    "record_id": "ALLOY_1",
                    "metal": "CoWP",
                    "solution_id": "SOL_1",
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 1.0,
                    "Ecorr_V_SHE": -0.15,
                    "icorr_uA_cm2": 8.0,
                    "beta_a_mV_dec": 65.0,
                    "beta_c_mV_dec": 115.0,
                },
                {
                    "record_id": "MYSTERY_1",
                    "metal": "MysteryAlloyX",
                    "solution_id": "SOL_1",
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 1.0,
                    "Ecorr_V_SHE": 0.12,
                    "icorr_uA_cm2": 5.0,
                    "beta_a_mV_dec": 70.0,
                    "beta_c_mV_dec": 120.0,
                },
            ]
        )
        raw_mat = pd.DataFrame(
            [
                {
                    "material_id": "CoWP",
                    "material_name": "Cobalt Tungsten Phosphide",
                    "category": "barrier",
                    "density_g_cm3": 9.5,
                    "equivalent_weight_g_eq": 28.5,
                }
            ]
        )
        db = HarmonizedDatabase.from_raw_tables(raw_echem, material_df=raw_mat)
        self.assertTrue(has_faraday_properties("CoWP", db.material_properties))
        self.assertFalse(has_faraday_properties("MysteryAlloyX", db.material_properties))

        cowp_rate = calculate_faraday_corrosion_rate("CoWP", 8.0, material_properties=db.material_properties)
        self.assertGreater(cowp_rate.corrosion_rate_nm_min, 0.0)

        # Controller degrades gracefully for MysteryAlloyX (returns polarization curve, NaN rate, warning)
        controller = CorrosionPlatformController(db)
        req = SimulationRequest(
            primary_metal="MysteryAlloyX",
            pH=4.0,
            temperature_C=25.0,
            oxidizer_type="H2O2",
            oxidizer_concentration=1.0,
            parameter_source="database",
        )
        payload = controller.run_single_metal_analysis(req)
        self.assertIsNone(payload.faraday_result)
        self.assertTrue(math.isnan(payload.initial_corrosion_rate_nm_min))
        self.assertEqual(payload.thickness_loss_risk_tier, "N/A")
        self.assertTrue(any("Missing physical properties" in w for w in payload.warnings))

        # Galvanic analysis also degrades gracefully when Active Metal lacks Faraday properties
        galv_req = SimulationRequest(
            primary_metal="MysteryAlloyX",
            secondary_metal="CoWP",
            pH=4.0,
            temperature_C=25.0,
            oxidizer_type="H2O2",
            oxidizer_concentration=1.0,
            parameter_source="database",
        )
        # Force MysteryAlloyX to be the active metal via manual override with lower Ecorr
        galv_req_manual = SimulationRequest(
            primary_metal="MysteryAlloyX",
            secondary_metal="CoWP",
            pH=4.0,
            temperature_C=25.0,
            oxidizer_type="H2O2",
            oxidizer_concentration=1.0,
            parameter_source="manual",
        )
        # Use database lookup directly where CoWP (-0.15) is active and MysteryAlloyX (+0.12) is noble,
        # and also test when MysteryAlloyX is active (Ecorr = -0.30 vs CoWP -0.15)
        from corrosion_simulator.platform_controller import ManualElectrochemicalInput

        galv_degraded = controller.run_galvanic_analysis(
            SimulationRequest(
                primary_metal="MysteryAlloyX",
                secondary_metal="CoWP",
                pH=4.0,
                temperature_C=25.0,
                parameter_source="manual",
                manual_primary=ManualElectrochemicalInput(-0.30, 5.0, 60.0, 120.0),
                manual_secondary=ManualElectrochemicalInput(0.10, 2.0, 60.0, 120.0),
            )
        )
        self.assertEqual(galv_degraded.active_metal, "MysteryAlloyX")
        self.assertIsNone(galv_degraded.coupled_kinetics)
        self.assertTrue(math.isfinite(galv_degraded.mixed_potential_V_SHE))
        self.assertTrue(math.isfinite(galv_degraded.acceleration_factor))
        self.assertTrue(math.isnan(galv_degraded.accelerated_cr0_nm_min))
        self.assertEqual(galv_degraded.coupled_risk_tier, "N/A")

        # Recipe comparison and all 6 report export formats must succeed cleanly on degraded payloads
        recipes = [
            CleaningRecipeSpec(name="Recipe A", pH=4.0, temperature_C=25.0, immersion_time_s=300.0),
            CleaningRecipeSpec(name="Recipe B", pH=5.0, temperature_C=30.0, immersion_time_s=300.0),
        ]
        comparison = compare_cleaning_recipes(controller, req, recipes)
        self.assertEqual(len(comparison.entries), 2)
        self.assertNotIn("∞", comparison.recommended_rationale)
        exports = generate_all_report_exports(
            payload,
            galvanic_payload=galv_degraded,
            comparison_payload=comparison,
        )
        for fmt in ("html", "pdf", "json", "csv", "png", "svg"):
            self.assertGreater(len(exports[fmt]), 0)

    def test_unseen_alloy_inherits_base_element_descriptors_in_ml(self) -> None:
        db = HarmonizedDatabase.from_seed()
        extractor = ElectrochemicalFeatureExtractor(material_properties=db.material_properties)
        extractor.fit(db.electrochemical_database)
        # TaN is not in seed material_properties; it should inherit Ta's descriptors rather than global median
        x_tan = extractor.transform_query(MLFeatureQuery(metal="TaN", pH=4.0, temperature_C=25.0))
        x_ta = extractor.transform_query(MLFeatureQuery(metal="Ta", pH=4.0, temperature_C=25.0))
        idx_e0 = extractor.feature_names.index("standard_potential_V_SHE")
        idx_rho = extractor.feature_names.index("density_g_cm3")
        self.assertAlmostEqual(float(x_tan[0, idx_e0]), float(x_ta[0, idx_e0]), places=6)
        self.assertAlmostEqual(float(x_tan[0, idx_rho]), float(x_ta[0, idx_rho]), places=6)


class TestLowSampleAndGPScaling(unittest.TestCase):
    def test_low_sample_material_is_forced_to_low_confidence(self) -> None:
        rows = [
            {
                "record_id": f"CU_{i}",
                "metal": "Cu",
                "pH": 3.0 + 0.1 * i,
                "temperature_C": 25.0,
                "oxidizer_type": "H2O2",
                "oxidizer_concentration": 1.0,
                "Ecorr_V_SHE": 0.25,
                "icorr_uA_cm2": 5.0,
                "beta_a_mV_dec": 60.0,
                "beta_c_mV_dec": 120.0,
            }
            for i in range(15)
        ] + [
            {
                "record_id": f"IR_{i}",
                "metal": "Ir",
                "pH": 3.0 + 0.1 * i,
                "temperature_C": 25.0,
                "oxidizer_type": "H2O2",
                "oxidizer_concentration": 1.0,
                "Ecorr_V_SHE": 0.75,
                "icorr_uA_cm2": 0.5,
                "beta_a_mV_dec": 60.0,
                "beta_c_mV_dec": 120.0,
            }
            for i in range(3)
        ]
        checker = ApplicabilityDomainChecker().fit(pd.DataFrame(rows))
        ir_res = checker.evaluate(
            MLFeatureQuery(metal="Ir", pH=3.1, temperature_C=25.0, oxidizer_type="H2O2", oxidizer_concentration=1.0)
        )
        self.assertEqual(ir_res.confidence_tier, "Low")
        self.assertTrue(any("Low training sample count" in w for w in ir_res.warnings))

    def test_gp_model_respects_max_gp_samples_cap(self) -> None:
        db = HarmonizedDatabase.from_seed()
        predictor = HybridMLPredictor(
            model_family="gaussian_process",
            random_state=42,
            max_gp_samples=20,
        )
        predictor.fit(db)
        self.assertTrue(any(m.gp_subsampled for m in predictor.models.values()))
        pred = predictor.predict(MLFeatureQuery(metal="Cu", pH=4.0, oxidizer_type="H2O2", oxidizer_concentration=1.0))
        self.assertTrue(math.isfinite(pred.Ecorr_V_SHE.mean))


class TestOfflineBuildPipelineAndStaleDetection(unittest.TestCase):
    def test_build_load_fresh_and_stale_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_dir = tmp / "raw"
            out_dir = tmp / "processed"
            raw_dir.mkdir()

            df = pd.DataFrame(
                [
                    {
                        "record_id": f"R_{i}",
                        "metal": "Cu" if i % 2 == 0 else "TiN",
                        "solution_id": f"SOL_{i % 3}",
                        "pH": 2.0 + (i % 5),
                        "temperature_C": 25.0 + (i % 3) * 10.0,
                        "oxidizer_type": "H2O2",
                        "oxidizer_concentration": 1.0,
                        "Ecorr_V_SHE": 0.25 if i % 2 == 0 else 0.10,
                        "icorr_uA_cm2": 5.0 + i * 0.2,
                        "beta_a_mV_dec": 60.0,
                        "beta_c_mV_dec": 120.0,
                    }
                    for i in range(16)
                ]
                + [
                    # Deliberately invalid row to test quarantine
                    {
                        "record_id": "R_BAD",
                        "metal": "Cu",
                        "solution_id": "SOL_0",
                        "pH": 3.0,
                        "temperature_C": 25.0,
                        "oxidizer_type": "H2O2",
                        "oxidizer_concentration": 1.0,
                        "Ecorr_V_SHE": None,
                        "icorr_uA_cm2": -1.0,
                        "beta_a_mV_dec": 60.0,
                        "beta_c_mV_dec": 120.0,
                    }
                ]
            )
            csv_path = raw_dir / "electrochemical.csv"
            df.to_csv(csv_path, index=False)

            mat_df = pd.DataFrame(
                [
                    {
                        "material_id": "TiN",
                        "material_name": "Titanium Nitride",
                        "category": "barrier",
                        "density_g_cm3": 5.22,
                        "molar_mass_g_mol": 61.87,
                        "valence": 3,
                    }
                ]
            )
            mat_df.to_csv(raw_dir / "material_properties.csv", index=False)

            summary = build_processed_artifacts(
                raw_dir=raw_dir,
                output_dir=out_dir,
                models=["random_forest"],
                export_csv=False,
            )
            self.assertEqual(summary.n_records, 16)
            self.assertEqual(summary.n_rejected, 1)
            self.assertEqual(set(summary.available_metals), {"Cu", "TiN"})
            self.assertTrue((out_dir / "electrochemical_database.parquet").exists())
            self.assertTrue((out_dir / "models" / "random_forest.joblib").exists())
            self.assertTrue((out_dir / "rejected_rows.csv").exists())
            self.assertIsNotNone(load_manifest(out_dir))

            # Freshness check + controller fast load
            status, _ = check_artifacts_freshness(raw_dir, out_dir)
            self.assertEqual(status, "fresh")
            ctrl = CorrosionPlatformController.from_default_data(
                raw_dir=raw_dir, processed_dir=out_dir
            )
            self.assertEqual(ctrl.artifact_status, "fresh")
            self.assertEqual(set(ctrl.available_options()["metals"]), {"Cu", "TiN"})

            # Mutating raw CSV marks artifacts as stale without breaking startup
            csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            stale_status, stale_msg = check_artifacts_freshness(raw_dir, out_dir)
            self.assertEqual(stale_status, "stale")
            self.assertIsNotNone(stale_msg)

            # Strict mode raises when invalid rows exist
            with self.assertRaises(RuntimeError):
                build_processed_artifacts(
                    raw_dir=raw_dir,
                    output_dir=out_dir,
                    models=["random_forest"],
                    strict=True,
                )


if __name__ == "__main__":
    unittest.main()
