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
        self.assertGreater(len(metals), len(CORE_METALS))
        self.assertIn("TiN", metals)
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


class TestPyInstallerExternalDataAndAutoRebuild(unittest.TestCase):
    def test_frozen_mode_resolves_external_data_next_to_executable_not_meipass(self) -> None:
        import os
        import sys
        from unittest.mock import patch

        from corrosion_simulator.paths import (
            DATA_DIR_ENV_VAR,
            get_bundle_dir,
            get_executable_dir,
            get_external_data_dir,
            iter_raw_csv_files,
            resolve_raw_and_processed_dirs,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dist_app = Path(tmpdir).resolve() / "dist" / "CorrosionSimulator"
            internal_dir = dist_app / "_internal"
            exe_path = dist_app / "CorrosionSimulator.exe"
            internal_dir.mkdir(parents=True)
            exe_path.write_text("", encoding="utf-8")

            env_clean = {k: v for k, v in os.environ.items() if k != DATA_DIR_ENV_VAR}
            with (
                patch.dict(os.environ, env_clean, clear=True),
                patch.object(sys, "frozen", True, create=True),
                patch.object(sys, "_MEIPASS", str(internal_dir), create=True),
                patch.object(sys, "executable", str(exe_path)),
            ):
                self.assertEqual(get_bundle_dir(), internal_dir)
                self.assertEqual(get_executable_dir(), dist_app)
                self.assertEqual(get_external_data_dir(), dist_app / "data")
                raw_d, proc_d = resolve_raw_and_processed_dirs()
                self.assertEqual(raw_d, dist_app / "data" / "raw")
                self.assertEqual(proc_d, dist_app / "data" / "processed")

            # Top-level data/*.csv fallback ignores processed/ artifacts
            flat_data = dist_app / "flat_data"
            (flat_data / "processed").mkdir(parents=True)
            (flat_data / "top_level.csv").write_text("metal,Ecorr_V_SHE,icorr_uA_cm2\nCu,0.2,3.0\n", encoding="utf-8")
            (flat_data / "processed" / "electrochemical_database.csv").write_text("ignored\n", encoding="utf-8")
            (flat_data / "rejected_rows.csv").write_text("ignored\n", encoding="utf-8")
            raw_flat, proc_flat = resolve_raw_and_processed_dirs(flat_data)
            self.assertEqual(raw_flat, flat_data.resolve())
            self.assertEqual(proc_flat, flat_data.resolve() / "processed")
            discovered = iter_raw_csv_files(raw_flat)
            self.assertEqual([p.name for p in discovered], ["top_level.csv"])

    def test_raw_csv_update_triggers_full_all_models_rebuild_and_processed_model_hot_swap(self) -> None:
        from corrosion_simulator.ml_predictor import SUPPORTED_MODEL_FAMILIES

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            raw_dir = data_dir / "raw"
            proc_dir = data_dir / "processed"
            raw_dir.mkdir(parents=True)

            base_rows = [
                {
                    "record_id": f"INIT_{i}",
                    "metal": "Cu" if i % 2 == 0 else "Co",
                    "solution_id": f"SOL_{i % 2}",
                    "pH": 3.0 + (i % 4) * 0.5,
                    "temperature_C": 25.0 + (i % 2) * 5.0,
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 1.0,
                    "Ecorr_V_SHE": 0.24 if i % 2 == 0 else -0.12,
                    "icorr_uA_cm2": 4.0 + i * 0.3,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                }
                for i in range(12)
            ]
            csv_file = raw_dir / "training.csv"
            pd.DataFrame(base_rows).to_csv(csv_file, index=False)

            progress_events: list[tuple[int, int, str]] = []
            ctrl_first = CorrosionPlatformController.from_default_data(
                data_dir=data_dir,
                progress_callback=lambda step, total, msg: progress_events.append((step, total, msg)),
            )
            self.assertEqual(ctrl_first.artifact_status, "rebuilt")
            self.assertEqual(set(ctrl_first.available_options()["metals"]), {"Cu", "Co"})
            self.assertGreater(len(progress_events), len(SUPPORTED_MODEL_FAMILIES))
            self.assertEqual(progress_events[-1][0], progress_events[-1][1])

            # Verify ALL 6 ML model families were built and saved
            for family in SUPPORTED_MODEL_FAMILIES:
                self.assertTrue(
                    (proc_dir / "models" / f"{family}.joblib").exists(),
                    msg=f"Expected prebuilt model {family}.joblib to exist",
                )

            # Second startup with unchanged raw/ hits fast 'fresh' path directly
            ctrl_second = CorrosionPlatformController.from_default_data(data_dir=data_dir)
            self.assertEqual(ctrl_second.artifact_status, "fresh")

            # Direct replacement of a model inside processed/models/ is used directly when raw/ is unchanged
            custom_et = HybridMLPredictor(model_family="extra_trees", n_estimators=17, random_state=7)
            custom_et.fit(ctrl_second.database)
            custom_et.save(proc_dir / "models" / "extra_trees.joblib")
            ctrl_custom = CorrosionPlatformController.from_default_data(data_dir=data_dir)
            self.assertEqual(ctrl_custom.artifact_status, "fresh")
            self.assertEqual(ctrl_custom.get_predictor("extra_trees").n_estimators, 17)

            # Corrupted .joblib auto-heals without crashing
            (proc_dir / "models" / "gradient_boosting.joblib").write_bytes(b"corrupted-pickle-bytes")
            ctrl_healed = CorrosionPlatformController.from_default_data(data_dir=data_dir)
            healed_pred = ctrl_healed.get_predictor("gradient_boosting")
            self.assertEqual(healed_pred.model_family, "gradient_boosting")
            self.assertIn("automatically retrained", ctrl_healed.artifact_warning or "")

            # Updating CSV in data/raw/ automatically rebuilds models & ingests new metal on next startup
            updated_rows = base_rows + [
                {
                    "record_id": f"RU_{j}",
                    "metal": "Ru",
                    "solution_id": "SOL_0",
                    "pH": 3.5,
                    "temperature_C": 25.0,
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 1.0,
                    "Ecorr_V_SHE": 0.48,
                    "icorr_uA_cm2": 0.6 + j * 0.1,
                    "beta_a_mV_dec": 65.0,
                    "beta_c_mV_dec": 115.0,
                }
                for j in range(4)
            ]
            pd.DataFrame(updated_rows).to_csv(csv_file, index=False)

            ctrl_updated = CorrosionPlatformController.from_default_data(data_dir=data_dir)
            self.assertEqual(ctrl_updated.artifact_status, "rebuilt")
            self.assertIn("Ru", ctrl_updated.available_options()["metals"])
            fresh_after, _ = check_artifacts_freshness(raw_dir, proc_dir)
            self.assertEqual(fresh_after, "fresh")

    def test_empty_external_data_dir_falls_back_to_seed_with_explicit_path_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_data = Path(tmpdir) / "empty_data"
            ctrl = CorrosionPlatformController.from_default_data(data_dir=empty_data)
            self.assertEqual(ctrl.artifact_status, "seed")
            self.assertIsNotNone(ctrl.artifact_warning)
            self.assertIn("empty_data", ctrl.artifact_warning or "")

    def test_main_entry_no_proxy_and_port_selection(self) -> None:
        import os
        from unittest.mock import patch

        from main import _configure_local_no_proxy, _find_available_port

        with patch.dict(os.environ, {"NO_PROXY": "corp.internal.local"}, clear=False):
            _configure_local_no_proxy()
            tokens = set(os.environ["NO_PROXY"].split(","))
            self.assertTrue({"corp.internal.local", "localhost", "127.0.0.1", "::1"}.issubset(tokens))

        port = _find_available_port("127.0.0.1", preferred_port=18501, max_tries=5)
        self.assertGreaterEqual(port, 18501)


class TestLabDirectoryAndDualFingerprintReload(unittest.TestCase):
    """Issue 04: external data/lab/<project>/ harmonization and dual-fingerprint Reload External Data."""

    def test_lab_path_resolution_in_dev_and_frozen_modes_excludes_lab_from_raw(self) -> None:
        import os
        import sys
        from unittest.mock import patch

        from corrosion_simulator.paths import (
            DATA_DIR_ENV_VAR,
            discover_lab_projects,
            iter_raw_csv_files,
            resolve_data_dirs,
            resolve_lab_dir,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dist_app = Path(tmpdir).resolve() / "dist" / "CorrosionSimulator"
            exe_path = dist_app / "CorrosionSimulator.exe"
            dist_app.mkdir(parents=True)
            exe_path.write_text("", encoding="utf-8")

            env_clean = {k: v for k, v in os.environ.items() if k != DATA_DIR_ENV_VAR}
            with (
                patch.dict(os.environ, env_clean, clear=True),
                patch.object(sys, "frozen", True, create=True),
                patch.object(sys, "_MEIPASS", str(dist_app / "_internal"), create=True),
                patch.object(sys, "executable", str(exe_path)),
            ):
                self.assertEqual(resolve_lab_dir(), dist_app / "data" / "lab")
                raw_d, proc_d, lab_d = resolve_data_dirs()
                self.assertEqual(raw_d, dist_app / "data" / "raw")
                self.assertEqual(proc_d, dist_app / "data" / "processed")
                self.assertEqual(lab_d, dist_app / "data" / "lab")

            # Development data_dir with top-level raw CSV + data/lab/<project>/ + top-level data/lab/*.csv
            data_dir = Path(tmpdir).resolve() / "workspace_data"
            (data_dir / "lab" / "study_dmso").mkdir(parents=True)
            (data_dir / "raw").mkdir(parents=True)
            (data_dir / "raw" / "lit.csv").write_text(
                "metal,Ecorr_V_SHE,icorr_uA_cm2\nCu,0.25,2.0\n", encoding="utf-8"
            )
            (data_dir / "lab" / "study_dmso" / "tafel.csv").write_text(
                "metal,Ecorr_V_SHE,icorr_uA_cm2\nCu,0.10,5.0\n", encoding="utf-8"
            )
            (data_dir / "lab" / "quick.csv").write_text(
                "metal,Ecorr_V_SHE,icorr_uA_cm2\nCo,-0.10,12.0\n", encoding="utf-8"
            )

            # iter_raw_csv_files scanning data_dir must skip lab/ completely
            raw_discovered = iter_raw_csv_files(data_dir)
            self.assertEqual([p.name for p in raw_discovered], ["lit.csv"])

            projects = discover_lab_projects(data_dir=data_dir)
            self.assertEqual(set(projects.keys()), {"default", "study_dmso"})
            self.assertEqual([p.name for p in projects["default"]], ["quick.csv"])
            self.assertEqual([p.name for p in projects["study_dmso"]], ["tafel.csv"])

    def test_lab_projects_harmonize_standalone_through_standard_pipeline(self) -> None:
        from corrosion_simulator.lab_data import discover_and_load_lab_projects

        with tempfile.TemporaryDirectory() as tmpdir:
            lab_dir = Path(tmpdir) / "lab"
            (lab_dir / "study_semoaq").mkdir(parents=True)
            (lab_dir / "study_invalid").mkdir(parents=True)

            # Ag/AgCl (+0.197 V), mA/cm2 -> uA/cm2, negative beta_c -> positive, 30 wt% DMSO -> 70 wt% water
            pd.DataFrame(
                [
                    {
                        "record_id": "LAB_1",
                        "metal": "Cu",
                        "solution_id": "SOL_LAB_DMSO",
                        "pH": 5.5,
                        "temperature_C": 30.0,
                        "reference_electrode": "Ag/AgCl",
                        "Ecorr_V_original": 0.103,
                        "icorr": 0.005,
                        "icorr_unit": "mA/cm2",
                        "beta_a": 55.0,
                        "beta_c": -115.0,
                        "organic_solvent_type": "DMSO",
                        "organic_solvent_concentration": 30.0,
                    },
                    {
                        "record_id": "LAB_BAD",
                        "metal": "Cu",
                        "solution_id": "SOL_LAB_DMSO",
                        "Ecorr_V_SHE": 0.2,
                        "icorr_uA_cm2": 1e9,
                    },
                ]
            ).to_csv(lab_dir / "study_semoaq" / "measurements.csv", index=False)

            # All-invalid lab project must stay empty (never fall back to seed dataset)
            pd.DataFrame(
                [
                    {
                        "record_id": "ALL_BAD",
                        "metal": "Cu",
                        "Ecorr_V_SHE": None,
                        "icorr_uA_cm2": -5.0,
                    }
                ]
            ).to_csv(lab_dir / "study_invalid" / "bad.csv", index=False)

            loaded = discover_and_load_lab_projects(lab_dir)
            self.assertEqual(set(loaded.keys()), {"study_semoaq", "study_invalid"})

            ds = loaded["study_semoaq"]
            self.assertEqual(ds.n_records, 1)
            self.assertEqual(ds.n_rejected, 1)
            row = ds.database.electrochemical_database.iloc[0]
            self.assertAlmostEqual(float(row["Ecorr_V_SHE"]), 0.300, places=6)
            self.assertAlmostEqual(float(row["icorr_uA_cm2"]), 5.0, places=6)
            self.assertAlmostEqual(float(row["beta_c_mV_dec"]), 115.0, places=6)
            self.assertAlmostEqual(float(row["water_content_wt_pct"]), 70.0, places=6)
            self.assertEqual(row["solvent_regime"], "semi_aqueous")
            self.assertEqual(row["solvent_subclass"], "polar_aprotic")

            empty_ds = loaded["study_invalid"]
            self.assertEqual(empty_ds.n_records, 0)
            self.assertEqual(empty_ds.n_rejected, 1)

    def test_dual_fingerprint_reload_skips_raw_rebuild_when_only_lab_changes_and_invalidates_on_raw_change(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            raw_dir = data_dir / "raw"
            lab_dir = data_dir / "lab"
            raw_dir.mkdir(parents=True)
            (lab_dir / "study_a").mkdir(parents=True)
            (lab_dir / "study_b").mkdir(parents=True)

            raw_rows = [
                {
                    "record_id": f"RAW_{i}",
                    "metal": "Cu" if i % 2 == 0 else "Co",
                    "solution_id": f"SOL_{i % 2}",
                    "pH": 3.0 + (i % 4) * 0.5,
                    "temperature_C": 25.0 + (i % 2) * 10.0,
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 1.0,
                    "Ecorr_V_SHE": 0.25 if i % 2 == 0 else -0.10,
                    "icorr_uA_cm2": 3.0 + i * 0.2,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                }
                for i in range(12)
            ]
            raw_csv = raw_dir / "lit.csv"
            pd.DataFrame(raw_rows).to_csv(raw_csv, index=False)

            lab_a_csv = lab_dir / "study_a" / "lab_a.csv"
            lab_a_rows = [
                {
                    "record_id": "LA_1",
                    "metal": "Cu",
                    "solution_id": "SOL_LAB_A",
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.28,
                    "icorr_uA_cm2": 2.5,
                    "beta_a_mV_dec": 62.0,
                    "beta_c_mV_dec": 118.0,
                }
            ]
            pd.DataFrame(lab_a_rows).to_csv(lab_a_csv, index=False)
            pd.DataFrame(
                [
                    {
                        "record_id": "LB_1",
                        "metal": "Co",
                        "solution_id": "SOL_LAB_B",
                        "pH": 4.0,
                        "temperature_C": 25.0,
                        "Ecorr_V_SHE": -0.09,
                        "icorr_uA_cm2": 15.0,
                        "beta_a_mV_dec": 65.0,
                        "beta_c_mV_dec": 122.0,
                    }
                ]
            ).to_csv(lab_dir / "study_b" / "lab_b.csv", index=False)

            ctrl = CorrosionPlatformController.from_default_data(
                data_dir=data_dir,
                rebuild_models=["random_forest"],
            )
            self.assertEqual(set(ctrl.available_options()["lab_projects"]), {"study_a", "study_b"})
            self.assertEqual(ctrl.get_lab_project("study_a").n_records, 1)
            base_pred_before = ctrl.predictor

            # Populate lab calibration cache entries
            ctrl._lab_calibration_cache["study_a:Cu"] = "cached_a"
            ctrl._lab_calibration_cache["study_b:Co"] = "cached_b"

            # 1. Modify only data/lab/study_a/ -> fast < 1 s reload, skips base rebuild, preserves base predictor
            lab_a_rows.append(
                {
                    "record_id": "LA_2",
                    "metal": "Cu",
                    "solution_id": "SOL_LAB_A",
                    "pH": 5.0,
                    "temperature_C": 35.0,
                    "Ecorr_V_SHE": 0.24,
                    "icorr_uA_cm2": 4.1,
                    "beta_a_mV_dec": 64.0,
                    "beta_c_mV_dec": 120.0,
                }
            )
            pd.DataFrame(lab_a_rows).to_csv(lab_a_csv, index=False)

            reload_lab_only = ctrl.reload_external_data(rebuild_models=["random_forest"])
            self.assertFalse(reload_lab_only.raw_rebuilt)
            self.assertEqual(reload_lab_only.lab_status, "reloaded")
            self.assertEqual(reload_lab_only.lab_added_or_updated, ("study_a",))
            self.assertLess(reload_lab_only.elapsed_s, 1.0)
            self.assertIs(ctrl.predictor, base_pred_before)
            self.assertEqual(ctrl.get_lab_project("study_a").n_records, 2)
            self.assertNotIn("study_a:Cu", ctrl._lab_calibration_cache)
            self.assertIn("study_b:Co", ctrl._lab_calibration_cache)

            # 2. Modify data/raw/ -> rebuilds literature artifacts and clears all cached lab calibrations
            raw_rows.append(
                {
                    "record_id": "RAW_NEW",
                    "metal": "Cu",
                    "solution_id": "SOL_0",
                    "pH": 4.5,
                    "temperature_C": 25.0,
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 1.0,
                    "Ecorr_V_SHE": 0.26,
                    "icorr_uA_cm2": 3.8,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                }
            )
            pd.DataFrame(raw_rows).to_csv(raw_csv, index=False)

            reload_raw = ctrl.reload_external_data(rebuild_models=["random_forest"])
            self.assertTrue(reload_raw.raw_rebuilt)
            self.assertEqual(reload_raw.raw_status, "rebuilt")
            self.assertTrue(reload_raw.lab_cache_invalidated)
            self.assertEqual(ctrl._lab_calibration_cache, {})
            self.assertIsNot(ctrl.predictor, base_pred_before)
            self.assertEqual(len(ctrl.database.electrochemical_database), 13)


if __name__ == "__main__":
    unittest.main()
