"""Behavioral tests for Phase 2 Hybrid Physics-ML predictor, MC uncertainty, and Applicability Domain."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from corrosion_simulator.database import ExposureCondition, HarmonizedDatabase
from corrosion_simulator.lab_calibration import (
    InsufficientLabSamplesError,
    LabCalibratedPredictor,
    ResidualBreakdown,
)
from corrosion_simulator.ml_predictor import (
    ApplicabilityDomainChecker,
    ApplicabilityDomainResult,
    ElectrochemicalFeatureExtractor,
    HybridMLPredictor,
    MLFeatureQuery,
    MLParameterPrediction,
    ParameterEstimate,
    plot_tafel_uncertainty_band,
    propagate_uncertainty_monte_carlo,
)
from corrosion_simulator.solvent_constants import (
    DEFAULT_UNKNOWN_ORGANIC_SOLVENT,
    SOLVENT_PHYSICAL_PROPERTIES,
    WATER_PROPERTIES,
    compute_mixture_solvent_descriptors,
    lookup_solvent_properties,
)


class TestFeatureExtractionAndApplicabilityDomain(unittest.TestCase):
    """Seam 2 ML slice 1: deterministic feature extraction and Applicability Domain checks."""

    def setUp(self) -> None:
        self.db = HarmonizedDatabase.from_directory("data/raw")

    def test_feature_extractor_produces_deterministic_physics_interaction_features(self) -> None:
        extractor = ElectrochemicalFeatureExtractor(
            material_properties=self.db.material_properties
        )
        X_train = extractor.fit_transform(self.db.electrochemical_database)
        self.assertEqual(X_train.shape[0], len(self.db.electrochemical_database))
        self.assertFalse(np.isnan(X_train).any())

        # Required engineered interaction terms must exist in feature names
        names = extractor.feature_names
        for required in (
            "pH_x_temperature_C",
            "E0_minus_ORP_V",
            "log1p_oxidizer_conc",
            "pourbaix_passivation_indicator",
        ):
            self.assertIn(required, names)

        # Deterministic transformation of a single ExposureCondition
        cond = ExposureCondition(
            metal="Co",
            pH=4.0,
            oxidizer_type="H2O2",
            oxidizer_concentration=1.0,
            temperature_C=25.0,
        )
        x1 = extractor.transform_query(cond)
        x2 = extractor.transform_query(cond)
        np.testing.assert_allclose(x1, x2, rtol=0.0, atol=0.0)
        self.assertEqual(x1.shape, (1, len(names)))

        # Verify interaction calculation: pH * T = 4.0 * 25.0 = 100.0
        idx_ph_t = names.index("pH_x_temperature_C")
        self.assertAlmostEqual(float(x1[0, idx_ph_t]), 100.0, places=6)

    def test_applicability_domain_distinguishes_inliers_from_extrapolation(self) -> None:
        checker = ApplicabilityDomainChecker(k=3).fit(self.db.electrochemical_database)

        # Inlier: take a dense Co training row (Co has 54 training samples)
        co_row = self.db.electrochemical_database[
            self.db.electrochemical_database["metal"] == "Co"
        ].iloc[4]
        inlier_query = MLFeatureQuery(
            metal="Co",
            pH=float(co_row["pH"]),
            temperature_C=float(co_row["temperature_C"]),
            oxidizer_type=str(co_row["oxidizer_type"]),
            oxidizer_concentration=float(co_row["oxidizer_concentration"]),
            acid_type=str(co_row["acid_type"]),
            acid_concentration=float(co_row["acid_concentration"]) if co_row["acid_concentration"] == co_row["acid_concentration"] else 0.0,
            inhibitor_type=str(co_row["inhibitor_type"]),
            inhibitor_concentration=float(co_row["inhibitor_concentration"]) if co_row["inhibitor_concentration"] == co_row["inhibitor_concentration"] else 0.0,
        )
        inlier_res = checker.evaluate(inlier_query)
        self.assertTrue(inlier_res.in_bounding_box)
        self.assertEqual(inlier_res.confidence_tier, "High")
        self.assertEqual(len(inlier_res.out_of_box_features), 0)

        # Medium confidence: in-box query in a sparser acidic H2O2 high-temperature region of Co
        med_query = MLFeatureQuery(
            metal="Co",
            pH=1.0,
            temperature_C=65.0,
            oxidizer_type="H2O2",
            oxidizer_concentration=1.0,
            acid_type="None",
            acid_concentration=0.0,
            inhibitor_type="None",
            inhibitor_concentration=0.0,
        )
        med_res = checker.evaluate(med_query)
        self.assertTrue(med_res.in_bounding_box)
        self.assertEqual(med_res.confidence_tier, "Medium")
        self.assertGreater(len(med_res.experimental_recommendations), 0)

        # Out-of-domain query: extreme temperature (150 C) and unseen oxidizer ("O3_plasma")
        ood_query = ExposureCondition(
            metal="Co",
            pH=13.5,
            oxidizer_type="O3_plasma",
            oxidizer_concentration=25.0,
            temperature_C=150.0,
        )
        ood_res = checker.evaluate(ood_query)
        self.assertFalse(ood_res.in_bounding_box)
        self.assertEqual(ood_res.confidence_tier, "Low")
        self.assertGreater(len(ood_res.out_of_box_features), 0)
        self.assertGreater(len(ood_res.unseen_categories), 0)
        self.assertGreater(len(ood_res.warnings), 0)
        self.assertGreater(len(ood_res.experimental_recommendations), 0)

        # Unseen metal also receives Low confidence and actionable guidance
        unseen_metal_res = checker.evaluate(ExposureCondition(metal="Zr", pH=4.0, temperature_C=25.0))
        self.assertEqual(unseen_metal_res.confidence_tier, "Low")
        self.assertFalse(unseen_metal_res.in_bounding_box)
        self.assertGreater(len(unseen_metal_res.experimental_recommendations), 0)


class TestHybridMLPredictorAndCrossValidation(unittest.TestCase):
    """Seam 2 ML slice 2: multi-family regressors, CV metrics, uncertainty estimates, and persistence."""

    def setUp(self) -> None:
        self.db = HarmonizedDatabase.from_seed()
        self.query = ExposureCondition(
            metal="Cu",
            pH=4.0,
            oxidizer_type="H2O2",
            oxidizer_concentration=1.0,
            temperature_C=25.0,
        )

    def test_all_three_model_families_predict_with_uncertainty_and_serialize(self) -> None:
        for family in (
            "random_forest",
            "hist_gradient_boosting",
            "rf_gp_ensemble",
        ):
            predictor = HybridMLPredictor(
                model_family=family,
                random_state=42,
                n_estimators=20,
                n_bootstrap=5,
            )
            predictor.fit(self.db)
            pred = predictor.predict(self.query)
            self.assertIsInstance(pred, MLParameterPrediction)
            self.assertEqual(pred.model_family, family)
            self.assertGreater(pred.icorr_uA_cm2_median, 0.0)

            for est in (
                pred.Ecorr_V_SHE,
                pred.log10_icorr_uA_cm2,
                pred.beta_a_mV_dec,
                pred.beta_c_mV_dec,
            ):
                self.assertTrue(np.isfinite(est.mean))
                self.assertTrue(np.isfinite(est.std))
                self.assertGreaterEqual(est.std, 0.0)

            # Raw point estimates must satisfy physical constraints (spec.md:52)
            self.assertGreaterEqual(pred.Ecorr_V_SHE.mean, -2.5)
            self.assertLessEqual(pred.Ecorr_V_SHE.mean, 2.5)
            self.assertGreaterEqual(pred.beta_a_mV_dec.mean, 20.0)
            self.assertLessEqual(pred.beta_a_mV_dec.mean, 500.0)
            self.assertGreaterEqual(pred.beta_c_mV_dec.mean, 20.0)
            self.assertLessEqual(pred.beta_c_mV_dec.mean, 500.0)

            # Verify persistence round-trip for Random Forest
            if family == "random_forest":
                with tempfile.TemporaryDirectory() as tmpdir:
                    model_path = Path(tmpdir) / "rf_bundle.joblib"
                    predictor.save(model_path)
                    self.assertTrue(model_path.exists())
                    loaded = HybridMLPredictor.load(model_path)
                    pred_loaded = loaded.predict(self.query)
                    self.assertAlmostEqual(
                        pred.Ecorr_V_SHE.mean, pred_loaded.Ecorr_V_SHE.mean, places=10
                    )
                    self.assertAlmostEqual(
                        pred.log10_icorr_uA_cm2.mean,
                        pred_loaded.log10_icorr_uA_cm2.mean,
                        places=10,
                    )
                    self.assertAlmostEqual(
                        pred.log10_icorr_uA_cm2.std,
                        pred_loaded.log10_icorr_uA_cm2.std,
                        places=10,
                    )

    def test_removed_model_families_raise_value_error(self) -> None:
        from corrosion_simulator.ml_predictor import SUPPORTED_MODEL_FAMILIES

        self.assertEqual(
            SUPPORTED_MODEL_FAMILIES,
            ("random_forest", "hist_gradient_boosting", "rf_gp_ensemble"),
        )
        for legacy in ("extra_trees", "gradient_boosting", "gaussian_process"):
            with self.assertRaises(ValueError):
                HybridMLPredictor(model_family=legacy)  # type: ignore[arg-type]

    def test_cross_validation_reports_mae_rmse_r2_and_mape_for_all_targets(self) -> None:
        predictor = HybridMLPredictor(
            model_family="random_forest", random_state=42, n_estimators=20
        )
        metrics_5fold = predictor.cross_validate(self.db, cv_strategy="stratified_5fold")
        self.assertEqual(len(metrics_5fold), 4)
        targets = {m.target for m in metrics_5fold}
        self.assertEqual(
            targets,
            {"Ecorr_V_SHE", "log10_icorr_uA_cm2", "beta_a_mV_dec", "beta_c_mV_dec"},
        )
        for m in metrics_5fold:
            self.assertEqual(m.cv_strategy, "stratified_5fold")
            self.assertGreaterEqual(m.mae, 0.0)
            self.assertGreaterEqual(m.rmse, m.mae - 1e-9)
            self.assertTrue(np.isfinite(m.r2))
            self.assertGreaterEqual(m.mape, 0.0)

        # Also verify LOOCV runs cleanly on small seed subset
        small_db = HarmonizedDatabase.from_raw_tables(
            self.db.electrochemical_database.head(10),
            material_df=self.db.material_properties,
            solution_df=self.db.solution_properties,
        )
        metrics_loo = predictor.cross_validate(small_db, cv_strategy="loocv")
        self.assertEqual(len(metrics_loo), 4)
        self.assertTrue(all(m.cv_strategy == "loocv" for m in metrics_loo))

    def test_imputed_tafel_exclusion_group_cv_per_metal_and_quality_weights(self) -> None:
        raw_db = HarmonizedDatabase.from_directory("data/raw")
        ml_df = raw_db.ml_training_dataframe

        # 1. Excluding imputed Tafel slopes reduces beta_a/beta_c training & CV counts to measured rows
        valid_ecorr = int(ml_df["Ecorr_V_SHE"].notna().sum())
        valid_icorr = int((ml_df["icorr_uA_cm2"].notna() & (ml_df["icorr_uA_cm2"] > 0)).sum())
        measured_ba = int((~ml_df["beta_a_imputed"]).sum())
        measured_bc = int((~ml_df["beta_c_imputed"]).sum())
        self.assertLess(measured_ba, len(ml_df))
        self.assertLess(measured_bc, len(ml_df))

        predictor_clean = HybridMLPredictor(
            model_family="random_forest",
            random_state=42,
            n_estimators=20,
            exclude_imputed_tafel=True,
            use_quality_weights=True,
        ).fit(raw_db)
        self.assertEqual(predictor_clean.target_training_counts["Ecorr_V_SHE"], valid_ecorr)
        self.assertEqual(predictor_clean.target_training_counts["log10_icorr_uA_cm2"], valid_icorr)
        self.assertEqual(predictor_clean.target_training_counts["beta_a_mV_dec"], measured_ba)
        self.assertEqual(predictor_clean.target_training_counts["beta_c_mV_dec"], measured_bc)

        # 2. GroupKFold by source_reference + per-metal breakdown
        metrics_group = predictor_clean.cross_validate(raw_db, cv_strategy="group_5fold")
        by_target = {m.target: m for m in metrics_group}
        self.assertEqual(by_target["Ecorr_V_SHE"].cv_strategy, "group_5fold")
        self.assertEqual(by_target["Ecorr_V_SHE"].n_samples, valid_ecorr)
        self.assertEqual(by_target["beta_a_mV_dec"].n_samples, measured_ba)
        self.assertEqual(by_target["beta_c_mV_dec"].n_samples, measured_bc)
        self.assertIn("Cu", by_target["Ecorr_V_SHE"].per_metal_metrics)
        self.assertEqual(
            int(sum(v["n_samples"] for v in by_target["Ecorr_V_SHE"].per_metal_metrics.values())),
            valid_ecorr,
        )

        # 3. Quality weights actively influence tree fitting (use_quality_weights=True vs False)
        df_weighted = self.db.electrochemical_database.copy()
        df_weighted["quality_score"] = np.where(
            np.arange(len(df_weighted)) % 2 == 0, 1.0, 0.05
        )
        weighted_db = HarmonizedDatabase(
            electrochemical_database=df_weighted,
            material_properties=self.db.material_properties,
            solution_properties=self.db.solution_properties,
            galvanic_series=self.db.galvanic_series,
            data_dictionary=self.db.data_dictionary,
        )
        pred_w = HybridMLPredictor(
            model_family="random_forest",
            random_state=42,
            n_estimators=25,
            use_quality_weights=True,
        ).fit(weighted_db).predict(self.query)
        pred_unw = HybridMLPredictor(
            model_family="random_forest",
            random_state=42,
            n_estimators=25,
            use_quality_weights=False,
        ).fit(weighted_db).predict(self.query)
        self.assertNotAlmostEqual(
            pred_w.log10_icorr_uA_cm2.mean,
            pred_unw.log10_icorr_uA_cm2.mean,
            places=6,
        )


class TestMonteCarloUncertaintyPropagation(unittest.TestCase):
    """Seam 2 ML slice 3: log-space and physics-clipped Monte Carlo uncertainty propagation."""

    def test_monte_carlo_enforces_physical_bounds_even_under_large_uncertainties(self) -> None:
        dummy_ad = ApplicabilityDomainResult(
            metal="Cu",
            confidence_tier="Medium",
            in_bounding_box=True,
            out_of_box_features=(),
            unseen_categories=(),
            knn_mean_distance=0.4,
            knn_k=3,
            n_metal_training_samples=5,
            warnings=(),
            experimental_recommendations=(),
        )
        # Intentionally extreme standard deviations to stress-test clipping & log-space sampling
        pred = MLParameterPrediction(
            metal="Cu",
            Ecorr_V_SHE=ParameterEstimate(mean=0.285, std=2.0),
            log10_icorr_uA_cm2=ParameterEstimate(mean=0.36, std=1.2),
            icorr_uA_cm2_median=10.0**0.36,
            beta_a_mV_dec=ParameterEstimate(mean=60.0, std=250.0),
            beta_c_mV_dec=ParameterEstimate(mean=120.0, std=300.0),
            model_family="random_forest",
            applicability=dummy_ad,
        )

        mc = propagate_uncertainty_monte_carlo(pred, n_draws=500, seed=42)
        self.assertEqual(mc.n_draws, 500)
        self.assertTrue(np.all(mc.icorr_samples_uA_cm2 > 0.0))
        self.assertTrue(np.all(mc.cr_nm_min_samples >= 0.0))
        self.assertTrue(np.all(mc.cr_mm_yr_samples >= 0.0))
        self.assertTrue(np.all((mc.beta_a_samples_mV_dec >= 20.0) & (mc.beta_a_samples_mV_dec <= 500.0)))
        self.assertTrue(np.all((mc.beta_c_samples_mV_dec >= 20.0) & (mc.beta_c_samples_mV_dec <= 500.0)))
        self.assertTrue(np.all((mc.Ecorr_samples_V_SHE >= -2.5) & (mc.Ecorr_samples_V_SHE <= 2.5)))

        self.assertLessEqual(mc.cr_nm_min_summary["p05"], mc.cr_nm_min_summary["p50"])
        self.assertLessEqual(mc.cr_nm_min_summary["p50"], mc.cr_nm_min_summary["p95"])

        band = mc.polarization_band
        valid = np.isfinite(band.log10_abs_net_p05) & np.isfinite(band.log10_abs_net_p95)
        self.assertTrue(np.all(band.log10_abs_net_p05[valid] <= band.log10_abs_net_p50[valid] + 1e-9))
        self.assertTrue(np.all(band.log10_abs_net_p50[valid] <= band.log10_abs_net_p95[valid] + 1e-9))

        # Overpotential-anchored V-cusp monotonicity and scalar summary intervals
        self.assertEqual(band.overpotential_V.shape, band.eta_log10_abs_net_p05.shape)
        self.assertTrue(np.all(band.eta_log10_abs_net_p05 <= band.eta_log10_abs_net_p25 + 1e-9))
        self.assertTrue(np.all(band.eta_log10_abs_net_p25 <= band.eta_log10_abs_net_p50 + 1e-9))
        self.assertTrue(np.all(band.eta_log10_abs_net_p50 <= band.eta_log10_abs_net_p75 + 1e-9))
        self.assertTrue(np.all(band.eta_log10_abs_net_p75 <= band.eta_log10_abs_net_p95 + 1e-9))
        zero_idx = int(np.argmin(np.abs(band.overpotential_V)))
        if abs(float(band.overpotential_V[zero_idx])) < 1e-9:
            self.assertLess(float(band.eta_log10_abs_net_p95[zero_idx]), float(band.eta_log10_abs_net_p05[0]))
        self.assertLessEqual(band.ecorr_p05_V_SHE, band.ecorr_p25_V_SHE)
        self.assertLessEqual(band.ecorr_p25_V_SHE, band.ecorr_p50_V_SHE)
        self.assertLessEqual(band.ecorr_p50_V_SHE, band.ecorr_p75_V_SHE)
        self.assertLessEqual(band.ecorr_p75_V_SHE, band.ecorr_p95_V_SHE)
        self.assertLessEqual(band.log10_icorr_p05, band.log10_icorr_p25)
        self.assertLessEqual(band.log10_icorr_p25, band.log10_icorr_p50)
        self.assertLessEqual(band.log10_icorr_p50, band.log10_icorr_p75)
        self.assertLessEqual(band.log10_icorr_p75, band.log10_icorr_p95)

        fig, ax = plot_tafel_uncertainty_band(mc)
        try:
            legend_labels = [t.get_text().lower() for t in ax.get_legend().get_texts()]
            self.assertTrue(any("90%" in lbl or "p05" in lbl or "band" in lbl for lbl in legend_labels))
        finally:
            plt.close(fig)


class TestPhase2MLNotebookArtifact(unittest.TestCase):
    """Verify the Phase 2 ML training and uncertainty notebook exists and is executed."""

    def test_phase2_ml_notebook_is_executed_with_outputs(self) -> None:
        nb_path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "03_phase2_ml_training_and_uncertainty.ipynb"
        )
        self.assertTrue(nb_path.exists(), f"Missing notebook: {nb_path}")
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 5)
        self.assertTrue(all(len(c.get("outputs", [])) > 0 for c in code_cells))


class TestPartialTargetMLTraining(unittest.TestCase):
    """Verify HybridMLPredictor utilizes partial-target records per target without NaN errors."""

    def test_fit_and_cv_use_partial_target_records_for_individual_targets(self) -> None:
        seed_db = HarmonizedDatabase.from_seed()
        base_n = len(seed_db.electrochemical_database)
        extra_partial = pd.DataFrame(
            [
                {
                    "record_id": f"P_ECORR_{i}",
                    "metal": "Ru",
                    "solution_id": "SOL_PARTIAL",
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 1.0,
                    "pH": 3.0 + 0.5 * i,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.45 + 0.02 * i,
                    "icorr_uA_cm2": None,
                    "beta_a_mV_dec": 80.0,
                    "beta_c_mV_dec": 120.0,
                }
                for i in range(4)
            ]
            + [
                {
                    "record_id": f"P_ICORR_{i}",
                    "metal": "Ru",
                    "solution_id": "SOL_PARTIAL",
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 1.0,
                    "pH": 3.5 + 0.5 * i,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": None,
                    "icorr_uA_cm2": 0.5 + 0.2 * i,
                    "beta_a_mV_dec": 85.0,
                    "beta_c_mV_dec": 125.0,
                }
                for i in range(3)
            ]
        )
        combined_raw = pd.concat(
            [seed_db.electrochemical_database, extra_partial], ignore_index=True
        )
        db = HarmonizedDatabase.from_raw_tables(
            combined_raw,
            material_df=seed_db.material_properties,
            solution_df=seed_db.solution_properties,
            supplement_missing_core_metals=False,
        )
        self.assertEqual(len(db.partial_target_records), 7)

        predictor = HybridMLPredictor(model_family="random_forest", n_estimators=20, random_state=42)
        predictor.fit(db)
        # Ecorr model receives complete pairs + 4 Ecorr-only rows; icorr model receives complete pairs + 3 icorr-only rows
        self.assertEqual(predictor.target_training_counts["Ecorr_V_SHE"], len(db.electrochemical_database) + 4)
        self.assertEqual(predictor.target_training_counts["log10_icorr_uA_cm2"], len(db.electrochemical_database) + 3)

        cv = predictor.cross_validate(db, cv_strategy="stratified_5fold")
        cv_map = {m.target: m for m in cv}
        self.assertEqual(cv_map["Ecorr_V_SHE"].n_samples, len(db.electrochemical_database) + 4)
        self.assertEqual(cv_map["log10_icorr_uA_cm2"].n_samples, len(db.electrochemical_database) + 3)


class TestSolventDescriptorsAndWaterWeightedFeatures(unittest.TestCase):
    """Issue 02: solvent lookup, continuous solvent descriptors, water-weighted Pourbaix features, and AD."""

    def test_solvent_lookup_covers_common_wet_clean_solvents_and_unseen_fallback(self) -> None:
        self.assertEqual(lookup_solvent_properties("DI water"), WATER_PROPERTIES)
        for name, expected_family, expected_protic in (
            ("EG", "glycol", 1.0),
            ("PG", "glycol", 1.0),
            ("IPA", "alcohol", 1.0),
            ("EtOH", "alcohol", 1.0),
            ("MeOH", "alcohol", 1.0),
            ("DMSO", "polar_aprotic", 0.0),
            ("NMP", "polar_aprotic", 0.0),
            ("DMF", "polar_aprotic", 0.0),
            ("acetone", "polar_aprotic", 0.0),
        ):
            props = lookup_solvent_properties(name)
            self.assertEqual(props.solvent_family, expected_family)
            self.assertEqual(props.is_protic, expected_protic)
            self.assertGreater(props.dielectric_constant, 0.0)
            self.assertGreater(props.polarity_index, 0.0)

        unseen = lookup_solvent_properties("exotic_novel_solvent_xyz")
        self.assertEqual(unseen, DEFAULT_UNKNOWN_ORGANIC_SOLVENT)

        # Composite labels mixing water and organic co-solvents must match the organic co-solvent first
        self.assertEqual(lookup_solvent_properties("50% DMSO / water"), SOLVENT_PHYSICAL_PROPERTIES["DMSO"])
        self.assertEqual(lookup_solvent_properties("ethanol-water"), SOLVENT_PHYSICAL_PROPERTIES["EtOH"])
        self.assertEqual(lookup_solvent_properties("IPA/water 30:70"), SOLVENT_PHYSICAL_PROPERTIES["IPA"])
        self.assertEqual(lookup_solvent_properties("deionized water rinse"), WATER_PROPERTIES)
        self.assertEqual(lookup_solvent_properties("negative_control_solvent"), DEFAULT_UNKNOWN_ORGANIC_SOLVENT)

    def test_feature_extractor_solvent_descriptors_pourbaix_scaling_and_fold_median_imputation(self) -> None:
        seed_db = HarmonizedDatabase.from_seed()
        extractor = ElectrochemicalFeatureExtractor(material_properties=seed_db.material_properties)
        extractor.fit(seed_db.electrochemical_database)
        names = extractor.feature_names

        for col in (
            "water_content_wt_pct",
            "effective_dielectric_constant",
            "solvent_polarity_index",
            "is_protic_solvent",
        ):
            self.assertIn(col, names)

        # Cu Pourbaix passive oxide window is [7.0, 12.0] -> pH=9.0 has indicator=1.0, pH=4.0 has distance=3.0
        x_aq_pass = extractor.transform_query(
            MLFeatureQuery(metal="Cu", pH=9.0, water_content_wt_pct=100.0)
        )
        x_semi_pass = extractor.transform_query(
            MLFeatureQuery(
                metal="Cu",
                pH=9.0,
                organic_solvent_type="DMSO",
                water_content_wt_pct=50.0,
                base_solvent="DI water",
            )
        )
        x_nonaq_pass = extractor.transform_query(
            MLFeatureQuery(
                metal="Cu",
                pH=9.0,
                organic_solvent_type="DMSO",
                water_content_wt_pct=0.0,
                base_solvent="DMSO",
            )
        )

        idx_water = names.index("water_content_wt_pct")
        idx_eps = names.index("effective_dielectric_constant")
        idx_pol = names.index("solvent_polarity_index")
        idx_prot = names.index("is_protic_solvent")
        idx_pb_ind = names.index("pourbaix_passivation_indicator")
        idx_pb_dist = names.index("pourbaix_pH_distance")

        self.assertAlmostEqual(float(x_aq_pass[0, idx_water]), 100.0, places=6)
        self.assertAlmostEqual(float(x_aq_pass[0, idx_eps]), WATER_PROPERTIES.dielectric_constant, places=4)
        self.assertAlmostEqual(float(x_aq_pass[0, idx_pb_ind]), 1.0, places=6)

        exp_eps_50, exp_pol_50, exp_prot_50 = compute_mixture_solvent_descriptors(50.0, "DMSO")
        self.assertAlmostEqual(float(x_semi_pass[0, idx_water]), 50.0, places=6)
        self.assertAlmostEqual(float(x_semi_pass[0, idx_eps]), exp_eps_50, places=6)
        self.assertAlmostEqual(float(x_semi_pass[0, idx_pol]), exp_pol_50, places=6)
        self.assertAlmostEqual(float(x_semi_pass[0, idx_prot]), exp_prot_50, places=6)
        self.assertAlmostEqual(float(x_semi_pass[0, idx_pb_ind]), 0.5, places=6)

        dmso_props = lookup_solvent_properties("DMSO")
        self.assertAlmostEqual(float(x_nonaq_pass[0, idx_water]), 0.0, places=6)
        self.assertAlmostEqual(float(x_nonaq_pass[0, idx_eps]), dmso_props.dielectric_constant, places=6)
        self.assertAlmostEqual(float(x_nonaq_pass[0, idx_prot]), 0.0, places=6)
        self.assertAlmostEqual(float(x_nonaq_pass[0, idx_pb_ind]), 0.0, places=6)

        x_aq_acid = extractor.transform_query(
            MLFeatureQuery(metal="Cu", pH=4.0, water_content_wt_pct=100.0)
        )
        x_semi_acid = extractor.transform_query(
            MLFeatureQuery(metal="Cu", pH=4.0, organic_solvent_type="DMSO", water_content_wt_pct=40.0)
        )
        self.assertAlmostEqual(float(x_aq_acid[0, idx_pb_dist]), 3.0, places=6)
        self.assertAlmostEqual(float(x_semi_acid[0, idx_pb_dist]), 3.0 * 0.4, places=6)

        # Leak-free fold median imputation: two training folds with different water medians produce different imputed values
        fold_a = pd.DataFrame(
            [
                {"metal": "Cu", "pH": 4.0, "temperature_C": 25.0, "water_content_wt_pct": 80.0},
                {"metal": "Cu", "pH": 5.0, "temperature_C": 25.0, "water_content_wt_pct": 80.0},
            ]
        )
        fold_b = pd.DataFrame(
            [
                {"metal": "Cu", "pH": 4.0, "temperature_C": 25.0, "water_content_wt_pct": 30.0},
                {"metal": "Cu", "pH": 5.0, "temperature_C": 25.0, "water_content_wt_pct": 30.0},
            ]
        )
        ext_a = ElectrochemicalFeatureExtractor(seed_db.material_properties).fit(fold_a)
        ext_b = ElectrochemicalFeatureExtractor(seed_db.material_properties).fit(fold_b)
        nan_query = MLFeatureQuery(metal="Cu", pH=4.5, water_content_wt_pct=float("nan"))
        val_a = float(ext_a.transform_query(nan_query)[0, ext_a.feature_names.index("water_content_wt_pct")])
        val_b = float(ext_b.transform_query(nan_query)[0, ext_b.feature_names.index("water_content_wt_pct")])
        self.assertAlmostEqual(val_a, 80.0, places=6)
        self.assertAlmostEqual(val_b, 30.0, places=6)

    def test_applicability_domain_checks_water_content_and_organic_solvent_type(self) -> None:
        seed_db = HarmonizedDatabase.from_seed()
        checker = ApplicabilityDomainChecker(k=3).fit(seed_db.electrochemical_database)

        # Seed is 100% aqueous with organic_solvent_type="None"; semi-aqueous DMSO query must trigger AD warnings
        ood_solvent = checker.evaluate(
            MLFeatureQuery(
                metal="Cu",
                pH=4.0,
                oxidizer_type="H2O2",
                oxidizer_concentration=1.0,
                temperature_C=25.0,
                organic_solvent_type="DMSO",
                water_content_wt_pct=50.0,
            )
        )
        self.assertFalse(ood_solvent.in_bounding_box)
        self.assertEqual(ood_solvent.confidence_tier, "Low")
        self.assertIn("water_content_wt_pct", ood_solvent.out_of_box_features)
        self.assertIn("organic_solvent_type=DMSO", ood_solvent.unseen_categories)

        # Non-finite / missing water_content_wt_pct must never silently report High confidence in-box
        for bad_water in (float("nan"), None):
            bad_res = checker.evaluate(
                MLFeatureQuery(
                    metal="Cu",
                    pH=4.0,
                    oxidizer_type="H2O2",
                    oxidizer_concentration=1.0,
                    temperature_C=25.0,
                    water_content_wt_pct=bad_water,  # type: ignore[arg-type]
                )
            )
            self.assertFalse(bad_res.in_bounding_box)
            self.assertEqual(bad_res.confidence_tier, "Low")
            self.assertIn("water_content_wt_pct", bad_res.out_of_box_features)
            self.assertTrue(math.isfinite(bad_res.knn_mean_distance))
            self.assertGreater(len(bad_res.warnings), 0)

    def test_all_six_model_families_train_save_load_and_predict_with_solvent_features(self) -> None:
        from corrosion_simulator.ml_predictor import SUPPORTED_MODEL_FAMILIES

        seed_db = HarmonizedDatabase.from_seed()
        semi_query = MLFeatureQuery(
            metal="Cu",
            pH=4.0,
            oxidizer_type="H2O2",
            oxidizer_concentration=1.0,
            temperature_C=25.0,
            organic_solvent_type="EG",
            water_content_wt_pct=60.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            for family in SUPPORTED_MODEL_FAMILIES:
                predictor = HybridMLPredictor(
                    model_family=family,
                    random_state=42,
                    n_estimators=15,
                    n_bootstrap=4,
                ).fit(seed_db)
                self.assertIn("effective_dielectric_constant", predictor.feature_extractor.feature_names)
                pred = predictor.predict(semi_query)
                self.assertTrue(np.isfinite(pred.Ecorr_V_SHE.mean))
                self.assertGreater(pred.icorr_uA_cm2_median, 0.0)
                bundle_path = Path(tmpdir) / f"{family}.joblib"
                predictor.save(bundle_path)
                loaded = HybridMLPredictor.load(bundle_path)
                pred_loaded = loaded.predict(semi_query)
                self.assertAlmostEqual(pred.Ecorr_V_SHE.mean, pred_loaded.Ecorr_V_SHE.mean, places=10)


class TestLabDeltaLearningAndLocalGP(unittest.TestCase):
    """Issue 05: parameter-level Delta-Learning GP, n<4 bias fallback, Lab-Only GP, and clipping."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.seed_db = HarmonizedDatabase.from_seed()
        cls.base_predictor = HybridMLPredictor(
            model_family="random_forest",
            random_state=42,
            n_estimators=25,
        ).fit(cls.seed_db)

    def _build_lab_db(self, rows: list[dict[str, object]]) -> HarmonizedDatabase:
        return HarmonizedDatabase.from_raw_tables(
            pd.DataFrame(rows),
            material_df=self.seed_db.material_properties,
            filter_to_core_metals=False,
            supplement_missing_core_metals=False,
            fallback_to_seed=False,
        )

    def test_delta_learning_gp_residual_near_vs_far_uncertainty_and_tafel_imputation(self) -> None:
        lab_rows: list[dict[str, object]] = []
        for idx, ph_val in enumerate([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]):
            q = MLFeatureQuery(
                metal="Cu",
                pH=ph_val,
                temperature_C=25.0,
                oxidizer_type="H2O2",
                oxidizer_concentration=1.0,
                organic_solvent_type="EG",
                water_content_wt_pct=50.0,
                base_solvent="DI water",
            )
            prior = self.base_predictor.predict(q)
            row: dict[str, object] = {
                "record_id": f"LAB_CU_{idx}",
                "metal": "Cu",
                "solution_id": f"LAB_EG_PH{int(ph_val)}",
                "pH": ph_val,
                "temperature_C": 25.0,
                "oxidizer_type": "H2O2",
                "oxidizer_concentration": 1.0,
                "acid_type": "None",
                "acid_concentration": 0.0,
                "inhibitor_type": "None",
                "inhibitor_concentration": 0.0,
                "organic_solvent_type": "EG",
                "water_content_wt_pct": 50.0,
                "base_solvent": "DI water",
                "Ecorr_V_SHE": prior.Ecorr_V_SHE.mean + 0.25,
                "icorr_uA_cm2": 10.0 ** (prior.log10_icorr_uA_cm2.mean + 0.90),
            }
            # Only 2 rows have measured beta_a; remaining 4 omit beta_a (imputed by harmonizer).
            # Zero rows measure beta_c (all imputed by harmonizer).
            if idx in (1, 3):
                row["beta_a_mV_dec"] = prior.beta_a_mV_dec.mean + 25.0
            lab_rows.append(row)

        lab_db = self._build_lab_db(lab_rows)
        cal = LabCalibratedPredictor(
            base_predictor=self.base_predictor,
            calibration_mode="delta_learning",
            lab_project="cu_eg_study",
        ).fit(lab_db)

        near_q = MLFeatureQuery(
            metal="Cu",
            pH=5.0,
            temperature_C=25.0,
            oxidizer_type="H2O2",
            oxidizer_concentration=1.0,
            organic_solvent_type="EG",
            water_content_wt_pct=50.0,
            base_solvent="DI water",
        )
        near_prior = self.base_predictor.predict(near_q)
        near_pred = cal.predict(near_q)
        self.assertIsInstance(near_pred.residual_breakdown, ResidualBreakdown)
        rb = near_pred.residual_breakdown
        assert rb is not None

        self.assertEqual(rb.Ecorr_V_SHE.method, "gp_residual")
        self.assertEqual(rb.Ecorr_V_SHE.n_lab, 6)
        self.assertEqual(rb.log10_icorr_uA_cm2.method, "gp_residual")
        self.assertEqual(rb.log10_icorr_uA_cm2.n_lab, 6)
        # Measured-only Tafel exclusion: 2 measured beta_a rows -> constant_bias; 0 measured beta_c -> prior_only
        self.assertEqual(rb.beta_a_mV_dec.method, "constant_bias")
        self.assertEqual(rb.beta_a_mV_dec.n_lab, 2)
        self.assertEqual(rb.beta_c_mV_dec.method, "prior_only")
        self.assertEqual(rb.beta_c_mV_dec.n_lab, 0)
        self.assertAlmostEqual(rb.beta_c_mV_dec.delta.mean, 0.0, places=9)
        self.assertAlmostEqual(rb.beta_c_mV_dec.delta.std, 0.0, places=9)

        # Near lab points, calibrated values closely track lab measurements (+0.25 V, +0.90 log10)
        self.assertAlmostEqual(
            near_pred.Ecorr_V_SHE.mean, near_prior.Ecorr_V_SHE.mean + 0.25, delta=0.04
        )
        self.assertAlmostEqual(
            near_pred.log10_icorr_uA_cm2.mean,
            near_prior.log10_icorr_uA_cm2.mean + 0.90,
            delta=0.10,
        )
        # Near lab points: Prior Variance Shrinkage strictly tightens sigma_total below sqrt(sigma_prior^2 + sigma_delta^2)
        self.assertLess(
            near_pred.Ecorr_V_SHE.std,
            math.hypot(rb.Ecorr_V_SHE.prior.std, rb.Ecorr_V_SHE.delta.std),
        )
        self.assertLess(
            near_pred.log10_icorr_uA_cm2.std,
            math.hypot(rb.log10_icorr_uA_cm2.prior.std, rb.log10_icorr_uA_cm2.delta.std),
        )
        # constant_bias (n=2) and prior_only (n=0) preserve unshrunk uncertainty propagation
        self.assertAlmostEqual(
            near_pred.beta_a_mV_dec.std,
            math.hypot(rb.beta_a_mV_dec.prior.std, rb.beta_a_mV_dec.delta.std),
            places=9,
        )
        self.assertAlmostEqual(
            near_pred.beta_c_mV_dec.std,
            rb.beta_c_mV_dec.prior.std,
            places=9,
        )

        # Far from lab points (pure aqueous, alkaline pH 13.5, 90 °C): residual smoothly reverts toward 0 and std expands
        far_q = MLFeatureQuery(
            metal="Cu",
            pH=13.5,
            temperature_C=90.0,
            oxidizer_type="H2O2",
            oxidizer_concentration=1.0,
            organic_solvent_type="None",
            water_content_wt_pct=100.0,
            base_solvent="DI water",
        )
        far_pred = cal.predict(far_q)
        far_rb = far_pred.residual_breakdown
        assert far_rb is not None
        self.assertLess(abs(far_rb.Ecorr_V_SHE.delta.mean), 0.05)
        self.assertLess(abs(far_rb.log10_icorr_uA_cm2.delta.mean), 0.15)
        self.assertGreater(far_rb.Ecorr_V_SHE.delta.std, rb.Ecorr_V_SHE.delta.std)
        self.assertGreater(far_rb.log10_icorr_uA_cm2.delta.std, rb.log10_icorr_uA_cm2.delta.std)
        self.assertAlmostEqual(
            far_pred.log10_icorr_uA_cm2.std,
            math.hypot(far_rb.log10_icorr_uA_cm2.prior.std, far_rb.log10_icorr_uA_cm2.delta.std),
            delta=0.02,
        )

    def test_delta_learning_constant_bias_quality_weighting_and_zero_record_passthrough(self) -> None:
        q = MLFeatureQuery(
            metal="Cu",
            pH=4.0,
            temperature_C=25.0,
            oxidizer_type="H2O2",
            oxidizer_concentration=1.0,
        )
        prior = self.base_predictor.predict(q)

        # Build 2 lab rows with different quality_score values (1.0 vs 0.2)
        raw_df = pd.DataFrame(
            [
                {
                    "record_id": "LOW_N_1",
                    "metal": "Cu",
                    "solution_id": "SOL_1",
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 1.0,
                    "acid_type": "None",
                    "acid_concentration": 0.0,
                    "inhibitor_type": "None",
                    "inhibitor_concentration": 0.0,
                    "Ecorr_V_SHE": prior.Ecorr_V_SHE.mean + 0.40,
                    "icorr_uA_cm2": 10.0 ** (prior.log10_icorr_uA_cm2.mean + 0.50),
                    "beta_a_mV_dec": prior.beta_a_mV_dec.mean + 20.0,
                    "beta_c_mV_dec": prior.beta_c_mV_dec.mean - 10.0,
                    "quality_score": 1.0,
                },
                {
                    "record_id": "LOW_N_2",
                    "metal": "Cu",
                    "solution_id": "SOL_1",
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 1.0,
                    "acid_type": "None",
                    "acid_concentration": 0.0,
                    "inhibitor_type": "None",
                    "inhibitor_concentration": 0.0,
                    "Ecorr_V_SHE": prior.Ecorr_V_SHE.mean + 0.10,
                    "icorr_uA_cm2": 10.0 ** (prior.log10_icorr_uA_cm2.mean + 0.20),
                    "beta_a_mV_dec": prior.beta_a_mV_dec.mean + 5.0,
                    "beta_c_mV_dec": prior.beta_c_mV_dec.mean - 40.0,
                    "quality_score": 0.2,
                },
            ]
        )
        lab_db = HarmonizedDatabase.from_raw_tables(
            raw_df,
            material_df=self.seed_db.material_properties,
            supplement_missing_core_metals=False,
            fallback_to_seed=False,
        )
        cal = LabCalibratedPredictor(
            base_predictor=self.base_predictor,
            calibration_mode="delta_learning",
            lab_project="small_n2",
        ).fit(lab_db)

        pred_cu = cal.predict(q)
        rb_cu = pred_cu.residual_breakdown
        assert rb_cu is not None
        self.assertEqual(rb_cu.Ecorr_V_SHE.method, "constant_bias")
        self.assertEqual(rb_cu.Ecorr_V_SHE.n_lab, 2)
        expected_weighted_de = (1.0 * 0.40 + 0.2 * 0.10) / 1.2
        self.assertAlmostEqual(rb_cu.Ecorr_V_SHE.delta.mean, expected_weighted_de, places=5)
        self.assertGreater(pred_cu.Ecorr_V_SHE.std, prior.Ecorr_V_SHE.std)

        # Unmeasured metal (Co has n=0 lab rows) passes through prior unchanged
        q_co = MLFeatureQuery(metal="Co", pH=4.0, temperature_C=25.0, oxidizer_type="H2O2", oxidizer_concentration=1.0)
        prior_co = self.base_predictor.predict(q_co)
        cal_co = cal.predict(q_co)
        assert cal_co.residual_breakdown is not None
        self.assertEqual(cal_co.residual_breakdown.Ecorr_V_SHE.method, "prior_only")
        self.assertEqual(cal_co.residual_breakdown.Ecorr_V_SHE.n_lab, 0)
        self.assertAlmostEqual(cal_co.Ecorr_V_SHE.mean, prior_co.Ecorr_V_SHE.mean, places=10)
        self.assertAlmostEqual(cal_co.log10_icorr_uA_cm2.mean, prior_co.log10_icorr_uA_cm2.mean, places=10)

    def test_lab_only_gp_trains_exclusively_on_lab_data_and_guards_min_samples(self) -> None:
        too_few_rows = [
            {
                "record_id": f"FEW_{i}",
                "metal": "Cu",
                "solution_id": "SOL_X",
                "pH": 4.0 + i,
                "temperature_C": 25.0,
                "Ecorr_V_SHE": -0.30,
                "icorr_uA_cm2": 25.0,
                "beta_a_mV_dec": 70.0,
                "beta_c_mV_dec": 140.0,
            }
            for i in range(2)
        ]
        too_few_db = self._build_lab_db(too_few_rows)
        with self.assertRaises(InsufficientLabSamplesError):
            LabCalibratedPredictor(
                base_predictor=self.base_predictor,
                calibration_mode="lab_only_gp",
                lab_project="too_few",
                allow_fallback=False,
            ).fit(too_few_db)

        fallback_pred = LabCalibratedPredictor(
            base_predictor=self.base_predictor,
            calibration_mode="lab_only_gp",
            lab_project="too_few",
            allow_fallback=True,
        ).fit(too_few_db).predict(MLFeatureQuery(metal="Cu", pH=4.5, temperature_C=25.0))
        self.assertEqual(fallback_pred.applicability.confidence_tier, "Low")
        self.assertTrue(
            any("at least 3" in w for w in fallback_pred.applicability.warnings)
        )

        # With n=5 lab records diverging sharply from literature, lab_only_gp fits without base_predictor
        lab_only_rows = [
            {
                "record_id": f"ONLY_{i}",
                "metal": "Cu",
                "solution_id": "SOL_SOLVENT",
                "pH": 3.0 + i * 1.0,
                "temperature_C": 30.0,
                "oxidizer_type": "None",
                "oxidizer_concentration": 0.0,
                "organic_solvent_type": "DMSO",
                "water_content_wt_pct": 25.0,
                "base_solvent": "DI water",
                "Ecorr_V_SHE": -0.45 - 0.02 * i,
                "icorr_uA_cm2": 100.0 + 10.0 * i,
                "beta_a_mV_dec": 90.0,
                "beta_c_mV_dec": 150.0,
            }
            for i in range(5)
        ]
        lab_only_db = self._build_lab_db(lab_only_rows)
        local_gp = LabCalibratedPredictor(
            base_predictor=None,
            calibration_mode="lab_only_gp",
            lab_project="dmso_local",
            allow_fallback=False,
        ).fit(lab_only_db)

        q_in = MLFeatureQuery(
            metal="Cu",
            pH=5.0,
            temperature_C=30.0,
            oxidizer_type="None",
            oxidizer_concentration=0.0,
            organic_solvent_type="DMSO",
            water_content_wt_pct=25.0,
            base_solvent="DI water",
        )
        pred_in = local_gp.predict(q_in)
        self.assertEqual(pred_in.calibration_mode, "lab_only_gp")
        self.assertEqual(pred_in.model_family, "lab_local_gp")
        self.assertEqual(pred_in.applicability.n_metal_training_samples, 5)
        self.assertTrue(pred_in.applicability.in_bounding_box)
        self.assertAlmostEqual(pred_in.Ecorr_V_SHE.mean, -0.49, delta=0.08)
        self.assertAlmostEqual(pred_in.log10_icorr_uA_cm2.mean, math.log10(120.0), delta=0.15)

    def test_calibrated_predictions_enforce_physical_bounds_clipping(self) -> None:
        # Extreme valid lab measurements near bounds push delta-learning past limits unless clipped
        extreme_rows = [
            {
                "record_id": f"EXT_{i}",
                "metal": "Cu",
                "solution_id": "SOL_EXT",
                "pH": 3.0 + i * 0.5,
                "temperature_C": 25.0,
                "oxidizer_type": "H2O2",
                "oxidizer_concentration": 1.0,
                "Ecorr_V_SHE": 2.45,
                "icorr_uA_cm2": 9e4,
                "beta_a_mV_dec": 495.0,
                "beta_c_mV_dec": 21.0,
            }
            for i in range(4)
        ]
        ext_db = self._build_lab_db(extreme_rows)
        cal = LabCalibratedPredictor(
            base_predictor=self.base_predictor,
            calibration_mode="delta_learning",
            lab_project="extreme_clip",
        ).fit(ext_db)

        pred = cal.predict(
            MLFeatureQuery(
                metal="Cu",
                pH=3.5,
                temperature_C=25.0,
                oxidizer_type="H2O2",
                oxidizer_concentration=1.0,
            )
        )
        self.assertGreaterEqual(pred.Ecorr_V_SHE.mean, -2.5)
        self.assertLessEqual(pred.Ecorr_V_SHE.mean, 2.5)
        self.assertGreaterEqual(pred.log10_icorr_uA_cm2.mean, -6.0)
        self.assertLessEqual(pred.log10_icorr_uA_cm2.mean, 6.0)
        self.assertGreaterEqual(pred.beta_a_mV_dec.mean, 20.0)
        self.assertLessEqual(pred.beta_a_mV_dec.mean, 500.0)
        self.assertGreaterEqual(pred.beta_c_mV_dec.mean, 20.0)
        self.assertLessEqual(pred.beta_c_mV_dec.mean, 500.0)

        mc = propagate_uncertainty_monte_carlo(pred, n_draws=200, seed=42)
        self.assertTrue(np.all((mc.Ecorr_samples_V_SHE >= -2.5) & (mc.Ecorr_samples_V_SHE <= 2.5)))
        self.assertTrue(np.all((mc.beta_a_samples_mV_dec >= 20.0) & (mc.beta_a_samples_mV_dec <= 500.0)))
        self.assertTrue(np.all((mc.beta_c_samples_mV_dec >= 20.0) & (mc.beta_c_samples_mV_dec <= 500.0)))
        self.assertTrue(np.all(mc.icorr_samples_uA_cm2 > 0.0))

    def test_mode_aware_applicability_domain_and_formatted_residual_breakdown(self) -> None:
        lab_rows: list[dict[str, object]] = []
        for idx, ph_val in enumerate([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]):
            q = MLFeatureQuery(
                metal="Cu",
                pH=ph_val,
                temperature_C=25.0,
                oxidizer_type="H2O2",
                oxidizer_concentration=1.0,
                acid_type="None",
                acid_concentration=0.0,
                inhibitor_type="None",
                inhibitor_concentration=0.0,
                organic_solvent_type="EG",
                water_content_wt_pct=50.0,
                base_solvent="DI water",
            )
            prior = self.base_predictor.predict(q)
            row: dict[str, object] = {
                "record_id": f"AD_CU_{idx}",
                "metal": "Cu",
                "solution_id": f"AD_EG_PH{int(ph_val)}",
                "pH": ph_val,
                "temperature_C": 25.0,
                "oxidizer_type": "H2O2",
                "oxidizer_concentration": 1.0,
                "acid_type": "None",
                "acid_concentration": 0.0,
                "inhibitor_type": "None",
                "inhibitor_concentration": 0.0,
                "organic_solvent_type": "EG",
                "water_content_wt_pct": 50.0,
                "base_solvent": "DI water",
                "Ecorr_V_SHE": prior.Ecorr_V_SHE.mean + 0.15,
                "icorr_uA_cm2": 10.0 ** (prior.log10_icorr_uA_cm2.mean + 0.40),
            }
            if idx in (1, 3):
                row["beta_a_mV_dec"] = prior.beta_a_mV_dec.mean + 15.0
            lab_rows.append(row)

        lab_db = self._build_lab_db(lab_rows)

        # 1. literature_only keeps literature AD untouched
        lit_cal = LabCalibratedPredictor(
            base_predictor=self.base_predictor,
            calibration_mode="literature_only",
        ).fit(lab_db)
        aq_query = MLFeatureQuery(
            metal="Cu",
            pH=4.0,
            temperature_C=25.0,
            oxidizer_type="H2O2",
            oxidizer_concentration=1.0,
            organic_solvent_type="None",
            water_content_wt_pct=100.0,
        )
        lit_pred = lit_cal.predict(aq_query)
        self.assertEqual(lit_pred.applicability.ad_mode, "literature_only")
        self.assertEqual(lit_pred.applicability.n_lab_samples, 0)
        self.assertIsNone(lit_pred.applicability.lab_anchoring_status)

        # 2. delta_learning reports Lab-Anchored near lab points, Reverting far from lab, and No Lab Coverage for unseen metal
        delta_cal = LabCalibratedPredictor(
            base_predictor=self.base_predictor,
            calibration_mode="delta_learning",
            lab_project="pilot_eg",
        ).fit(lab_db)
        near_q = MLFeatureQuery(
            metal="Cu",
            pH=5.0,
            temperature_C=25.0,
            oxidizer_type="H2O2",
            oxidizer_concentration=1.0,
            organic_solvent_type="EG",
            water_content_wt_pct=50.0,
            base_solvent="DI water",
        )
        near_pred = delta_cal.predict(near_q)
        self.assertEqual(near_pred.applicability.ad_mode, "delta_learning")
        self.assertEqual(near_pred.applicability.lab_anchoring_status, "Lab-Anchored")
        self.assertTrue(near_pred.applicability.lab_in_bounding_box)
        self.assertEqual(near_pred.applicability.n_lab_samples, 6)
        self.assertEqual(near_pred.applicability.confidence_tier, "High")
        self.assertIsNotNone(near_pred.applicability.lab_knn_distance)
        self.assertLessEqual(float(near_pred.applicability.lab_knn_distance), 0.35)

        assert near_pred.residual_breakdown is not None
        df_bd = near_pred.residual_breakdown.to_dataframe()
        self.assertEqual(
            list(df_bd.columns),
            [
                "Parameter",
                "Unit",
                "Prior Estimate",
                "Prior ±1σ",
                "Lab Residual (Δ)",
                "Residual ±1σ",
                "Calibrated Final",
                "Combined ±1σ",
                "Method",
                "Lab Samples (n)",
            ],
        )
        self.assertEqual(
            list(df_bd["Method"]),
            ["GP Residual", "GP Residual", "Constant Bias", "Prior Only"],
        )

        far_pred = delta_cal.predict(
            MLFeatureQuery(
                metal="Cu",
                pH=13.5,
                temperature_C=90.0,
                oxidizer_type="H2O2",
                oxidizer_concentration=1.0,
                organic_solvent_type="None",
                water_content_wt_pct=100.0,
            )
        )
        self.assertEqual(far_pred.applicability.lab_anchoring_status, "Reverting to Literature Prior")
        self.assertFalse(far_pred.applicability.lab_in_bounding_box)
        self.assertTrue(any("Reverting to Literature Prior" in w for w in far_pred.applicability.warnings))

        co_pred = delta_cal.predict(MLFeatureQuery(metal="Co", pH=5.0, temperature_C=25.0))
        self.assertEqual(co_pred.applicability.lab_anchoring_status, "No Lab Coverage")
        self.assertEqual(co_pred.applicability.n_lab_samples, 0)
        self.assertTrue(any("No Lab Coverage" in w for w in co_pred.applicability.warnings))

        # 3. lab_only_gp evaluates AD strictly on lab records only (aqueous query well covered by literature is still OOD)
        full_tafel_rows = [
            {**row, "beta_a_mV_dec": 70.0 + idx, "beta_c_mV_dec": 130.0 + idx}
            for idx, row in enumerate(lab_rows)
        ]
        lab_only = LabCalibratedPredictor(
            base_predictor=self.base_predictor,
            calibration_mode="lab_only_gp",
            lab_project="pilot_eg",
            allow_fallback=False,
        ).fit(self._build_lab_db(full_tafel_rows))
        lab_in_pred = lab_only.predict(near_q)
        self.assertEqual(lab_in_pred.applicability.ad_mode, "lab_only_gp")
        self.assertEqual(lab_in_pred.applicability.lab_anchoring_status, "Lab-Anchored")
        self.assertTrue(lab_in_pred.applicability.in_bounding_box)
        self.assertEqual(lab_in_pred.applicability.n_metal_training_samples, 6)

        lab_out_pred = lab_only.predict(aq_query)
        self.assertEqual(lab_out_pred.applicability.ad_mode, "lab_only_gp")
        self.assertEqual(lab_out_pred.applicability.lab_anchoring_status, "Outside Lab Domain")
        self.assertFalse(lab_out_pred.applicability.in_bounding_box)
        self.assertEqual(lab_out_pred.applicability.confidence_tier, "Low")

    def test_delta_learning_prior_variance_shrinkage_tightens_monte_carlo_intervals(self) -> None:
        # Build a clean lab curve in an aqueous H2O2 regime where the literature prior has non-trivial std
        lab_rows: list[dict[str, object]] = []
        for idx, ph_val in enumerate([2.5, 3.5, 4.5, 5.5, 6.5, 7.5]):
            q = MLFeatureQuery(
                metal="Cu",
                pH=ph_val,
                temperature_C=25.0,
                oxidizer_type="H2O2",
                oxidizer_concentration=1.0,
                organic_solvent_type="EG",
                water_content_wt_pct=50.0,
                base_solvent="DI water",
            )
            prior = self.base_predictor.predict(q)
            lab_rows.append(
                {
                    "record_id": f"SHRINK_{idx}",
                    "metal": "Cu",
                    "solution_id": f"SOL_S_{idx}",
                    "pH": ph_val,
                    "temperature_C": 25.0,
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 1.0,
                    "organic_solvent_type": "EG",
                    "water_content_wt_pct": 50.0,
                    "base_solvent": "DI water",
                    "Ecorr_V_SHE": prior.Ecorr_V_SHE.mean + 0.18,
                    "icorr_uA_cm2": 10.0 ** (prior.log10_icorr_uA_cm2.mean + 0.35),
                    "beta_a_mV_dec": 65.0 + idx * 1.5,
                    "beta_c_mV_dec": 120.0 + idx * 1.0,
                }
            )
        lab_db = self._build_lab_db(lab_rows)

        # Also verify delta_learning composes cleanly with rf_gp_ensemble as base prior
        rgp_base = HybridMLPredictor(
            model_family="rf_gp_ensemble",
            random_state=42,
            n_estimators=25,
        ).fit(self.seed_db)
        cal = LabCalibratedPredictor(
            base_predictor=rgp_base,
            calibration_mode="delta_learning",
            lab_project="shrinkage_check",
        ).fit(lab_db)

        anchored_q = MLFeatureQuery(
            metal="Cu",
            pH=4.5,
            temperature_C=25.0,
            oxidizer_type="H2O2",
            oxidizer_concentration=1.0,
            organic_solvent_type="EG",
            water_content_wt_pct=50.0,
            base_solvent="DI water",
        )
        prior_anchored = rgp_base.predict(anchored_q)
        cal_anchored = cal.predict(anchored_q)
        self.assertLess(cal_anchored.Ecorr_V_SHE.std, prior_anchored.Ecorr_V_SHE.std)
        self.assertLess(cal_anchored.log10_icorr_uA_cm2.std, prior_anchored.log10_icorr_uA_cm2.std)

        far_q = MLFeatureQuery(
            metal="Cu",
            pH=13.5,
            temperature_C=95.0,
            oxidizer_type="HNO3",
            oxidizer_concentration=5.0,
            organic_solvent_type="None",
            water_content_wt_pct=100.0,
            base_solvent="DI water",
        )
        cal_far = cal.predict(far_q)
        assert cal_far.residual_breakdown is not None
        self.assertAlmostEqual(
            cal_far.Ecorr_V_SHE.std,
            math.hypot(
                cal_far.residual_breakdown.Ecorr_V_SHE.prior.std,
                cal_far.residual_breakdown.Ecorr_V_SHE.delta.std,
            ),
            delta=0.01,
        )

        # End-to-end Monte Carlo propagation: tighter 90% CI at anchored lab point than uncalibrated prior or far extrapolation
        mc_anchored = propagate_uncertainty_monte_carlo(cal_anchored, n_draws=400, seed=42)
        mc_prior = propagate_uncertainty_monte_carlo(prior_anchored, n_draws=400, seed=42)
        mc_far = propagate_uncertainty_monte_carlo(cal_far, n_draws=400, seed=42)

        band_anchored_width = (
            mc_anchored.polarization_band.log10_icorr_p95
            - mc_anchored.polarization_band.log10_icorr_p05
        )
        band_prior_width = (
            mc_prior.polarization_band.log10_icorr_p95
            - mc_prior.polarization_band.log10_icorr_p05
        )
        band_far_width = (
            mc_far.polarization_band.log10_icorr_p95
            - mc_far.polarization_band.log10_icorr_p05
        )
        self.assertLess(band_anchored_width, band_prior_width)
        self.assertLess(band_anchored_width, band_far_width)

    def test_lab_only_gp_uses_low_dim_exposure_space_and_supports_multi_metal(self) -> None:
        rows: list[dict[str, object]] = []
        for idx, ph_val in enumerate([3.0, 4.0, 5.0, 6.0, 7.0]):
            rows.append(
                {
                    "record_id": f"LD_CU_{idx}",
                    "metal": "Cu",
                    "solution_id": f"SOL_CU_{idx}",
                    "pH": ph_val,
                    "temperature_C": 25.0 + 2.0 * idx,
                    "organic_solvent_type": "DMSO",
                    "water_content_wt_pct": 30.0,
                    "base_solvent": "DI water",
                    "Ecorr_V_SHE": 0.10 - 0.03 * ph_val,
                    "icorr_uA_cm2": 10.0 ** (1.2 - 0.1 * ph_val),
                    "beta_a_mV_dec": 65.0 + idx,
                    "beta_c_mV_dec": 115.0 + idx,
                }
            )
            rows.append(
                {
                    "record_id": f"LD_CO_{idx}",
                    "metal": "Co",
                    "solution_id": f"SOL_CO_{idx}",
                    "pH": ph_val,
                    "temperature_C": 25.0 + 2.0 * idx,
                    "organic_solvent_type": "DMSO",
                    "water_content_wt_pct": 30.0,
                    "base_solvent": "DI water",
                    "Ecorr_V_SHE": -0.35 - 0.02 * ph_val,
                    "icorr_uA_cm2": 10.0 ** (2.0 - 0.15 * ph_val),
                    "beta_a_mV_dec": 80.0 + idx,
                    "beta_c_mV_dec": 135.0 + idx,
                }
            )
        lab_db = self._build_lab_db(rows)
        local_gp = LabCalibratedPredictor(
            base_predictor=None,
            calibration_mode="lab_only_gp",
            lab_project="multi_metal_low_dim",
            allow_fallback=False,
        ).fit(lab_db)

        # Verify low-dimensional feature representation (< 20 dims vs 30-40+ global feature extractor)
        cu_fitted = local_gp._lab_only_metal_targets[("Cu", "Ecorr_V_SHE")]
        assert cu_fitted.transformer is not None
        X_cu = cu_fitted.transformer.transform_query(
            MLFeatureQuery(metal="Cu", pH=5.0, temperature_C=29.0, organic_solvent_type="DMSO", water_content_wt_pct=30.0)
        )
        self.assertLess(X_cu.shape[1], 20)

        cu_pred = local_gp.predict(
            MLFeatureQuery(metal="Cu", pH=5.0, temperature_C=29.0, organic_solvent_type="DMSO", water_content_wt_pct=30.0)
        )
        co_pred = local_gp.predict(
            MLFeatureQuery(metal="Co", pH=5.0, temperature_C=29.0, organic_solvent_type="DMSO", water_content_wt_pct=30.0)
        )
        self.assertEqual(cu_pred.model_family, "lab_local_gp")
        self.assertEqual(co_pred.model_family, "lab_local_gp")
        self.assertAlmostEqual(cu_pred.Ecorr_V_SHE.mean, 0.10 - 0.15, delta=0.04)
        self.assertAlmostEqual(co_pred.Ecorr_V_SHE.mean, -0.35 - 0.10, delta=0.04)

        # Unseen metal gracefully falls back to shared low-dim GP with Low confidence
        unseen_pred = local_gp.predict(
            MLFeatureQuery(metal="W", pH=5.0, temperature_C=29.0, organic_solvent_type="DMSO", water_content_wt_pct=30.0)
        )
        self.assertEqual(unseen_pred.model_family, "lab_local_gp")
        self.assertEqual(unseen_pred.applicability.confidence_tier, "Low")
        self.assertGreater(unseen_pred.Ecorr_V_SHE.std, cu_pred.Ecorr_V_SHE.std)


class TestPerMetalOOBResidualGPEnsemble(unittest.TestCase):
    """Issue 02: Global RF (oob_score=True) + per-Metal low-dim Zero-Mean OOB residual GP."""

    def _build_synthetic_db(
        self,
        *,
        n_cu: int = 24,
        n_co: int = 18,
        n_rare: int = 2,
    ) -> HarmonizedDatabase:
        seed_db = HarmonizedDatabase.from_seed()
        rows: list[dict[str, object]] = []
        rng = np.random.default_rng(123)
        for i in range(n_cu):
            ph = 2.0 + 8.0 * (i / max(n_cu - 1, 1))
            temp = 20.0 + 30.0 * ((i % 5) / 4.0)
            ecorr = 0.25 - 0.04 * ph + 0.001 * temp + float(rng.normal(0.0, 0.015))
            log_i = 0.8 - 0.08 * ph + 0.015 * temp + float(rng.normal(0.0, 0.05))
            rows.append(
                {
                    "record_id": f"SYN_CU_{i}",
                    "metal": "Cu",
                    "solution_id": f"S_CU_{i % 4}",
                    "pH": ph,
                    "temperature_C": temp,
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 1.0,
                    "acid_type": "None",
                    "acid_concentration": 0.0,
                    "inhibitor_type": "None",
                    "inhibitor_concentration": 0.0,
                    "Ecorr_V_SHE": ecorr,
                    "icorr_uA_cm2": 10.0**log_i,
                    "beta_a_mV_dec": 60.0 + 1.2 * ph,
                    "beta_c_mV_dec": 120.0 + 0.8 * ph,
                    "quality_score": 1.0 if i % 3 != 0 else 0.3,
                }
            )
        for i in range(n_co):
            ph = 2.0 + 7.0 * (i / max(n_co - 1, 1))
            temp = 25.0 + 20.0 * ((i % 4) / 3.0)
            rows.append(
                {
                    "record_id": f"SYN_CO_{i}",
                    "metal": "Co",
                    "solution_id": f"S_CO_{i % 3}",
                    "pH": ph,
                    "temperature_C": temp,
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 1.0,
                    "acid_type": "None",
                    "acid_concentration": 0.0,
                    "inhibitor_type": "None",
                    "inhibitor_concentration": 0.0,
                    "Ecorr_V_SHE": -0.20 - 0.03 * ph + float(rng.normal(0.0, 0.015)),
                    "icorr_uA_cm2": 10.0 ** (1.4 - 0.05 * ph + float(rng.normal(0.0, 0.05))),
                    "beta_a_mV_dec": 75.0 + ph,
                    "beta_c_mV_dec": 130.0 + ph,
                    "quality_score": 0.9,
                }
            )
        for i in range(n_rare):
            rows.append(
                {
                    "record_id": f"SYN_RU_{i}",
                    "metal": "Ru",
                    "solution_id": "S_RU",
                    "pH": 4.0 + i,
                    "temperature_C": 25.0,
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 1.0,
                    "acid_type": "None",
                    "acid_concentration": 0.0,
                    "inhibitor_type": "None",
                    "inhibitor_concentration": 0.0,
                    "Ecorr_V_SHE": 0.45 + 0.02 * i,
                    "icorr_uA_cm2": 0.5 + 0.1 * i,
                    "beta_a_mV_dec": 85.0,
                    "beta_c_mV_dec": 125.0,
                    "quality_score": 1.0,
                }
            )
        return HarmonizedDatabase.from_raw_tables(
            pd.DataFrame(rows),
            material_df=seed_db.material_properties,
            filter_to_core_metals=False,
            supplement_missing_core_metals=False,
            fallback_to_seed=False,
        )

    def test_oob_residual_gp_stage1_importances_rare_metal_fallback_and_extrapolation_std(self) -> None:
        from corrosion_simulator.ml_predictor import GP_MAX_TRAINING_SAMPLES

        self.assertEqual(GP_MAX_TRAINING_SAMPLES, 2000)

        db = self._build_synthetic_db(n_cu=24, n_co=18, n_rare=2)
        pred_ens = HybridMLPredictor(
            model_family="rf_gp_ensemble",
            random_state=42,
            n_estimators=30,
        ).fit(db)
        pred_rf = HybridMLPredictor(
            model_family="random_forest",
            random_state=42,
            n_estimators=30,
        ).fit(db)

        # Stage 1 RF fits 100% of records with oob_score=True and exposes valid feature importances
        self.assertEqual(pred_ens.target_training_counts["Ecorr_V_SHE"], 44)
        fi = pred_ens.feature_importances("log10_icorr_uA_cm2")
        self.assertGreater(len(fi), 10)
        self.assertAlmostEqual(float(fi["importance"].sum()), 1.0, places=6)

        # Metals with >=4 OOB rows get Stage 2 residual GPs; rare Metal (n=2 < 4) does not
        ecorr_model = pred_ens.models["Ecorr_V_SHE"]
        self.assertIn("Cu", ecorr_model._per_metal_residual_gps)
        self.assertIn("Co", ecorr_model._per_metal_residual_gps)
        self.assertNotIn("Ru", ecorr_model._per_metal_residual_gps)

        # Rare Metal (Ru, n=2) and unseen Metal (Ta, n=0) pass through Stage 1 RF mean and std identically
        for fallback_metal in ("Ru", "Ta"):
            q_fb = MLFeatureQuery(
                metal=fallback_metal,
                pH=4.5,
                temperature_C=25.0,
                oxidizer_type="H2O2",
                oxidizer_concentration=1.0,
            )
            out_ens = pred_ens.predict(q_fb)
            out_rf = pred_rf.predict(q_fb)
            self.assertAlmostEqual(out_ens.Ecorr_V_SHE.mean, out_rf.Ecorr_V_SHE.mean, places=10)
            self.assertAlmostEqual(out_ens.Ecorr_V_SHE.std, out_rf.Ecorr_V_SHE.std, places=10)
            self.assertAlmostEqual(
                out_ens.log10_icorr_uA_cm2.mean,
                out_rf.log10_icorr_uA_cm2.mean,
                places=10,
            )

        # Covered Metal (Cu): combined std = sqrt(rf_std^2 + gp_std^2) >= rf_std and expands far outside domain
        in_q = MLFeatureQuery(
            metal="Cu",
            pH=5.0,
            temperature_C=30.0,
            oxidizer_type="H2O2",
            oxidizer_concentration=1.0,
        )
        far_q = MLFeatureQuery(
            metal="Cu",
            pH=13.5,
            temperature_C=120.0,
            oxidizer_type="O3_plasma",
            oxidizer_concentration=25.0,
        )
        in_ens = pred_ens.predict(in_q)
        in_rf = pred_rf.predict(in_q)
        far_ens = pred_ens.predict(far_q)
        self.assertGreaterEqual(in_ens.Ecorr_V_SHE.std, in_rf.Ecorr_V_SHE.std)
        self.assertGreater(far_ens.Ecorr_V_SHE.std, in_ens.Ecorr_V_SHE.std)
        self.assertGreater(far_ens.log10_icorr_uA_cm2.std, in_ens.log10_icorr_uA_cm2.std)

    def test_per_metal_cap_subsamples_deterministically_without_global_dilution(self) -> None:
        # 24 Cu rows + 18 Co rows with max_gp_samples=15:
        # Cu (24 > 15) is capped at 15; Co (18 > 15) is capped at 15; neither steals budget from the other.
        db = self._build_synthetic_db(n_cu=24, n_co=18, n_rare=0)
        p1 = HybridMLPredictor(
            model_family="rf_gp_ensemble",
            random_state=42,
            n_estimators=25,
            max_gp_samples=15,
        ).fit(db)
        p2 = HybridMLPredictor(
            model_family="rf_gp_ensemble",
            random_state=42,
            n_estimators=25,
            max_gp_samples=15,
        ).fit(db)

        ecorr_m = p1.models["Ecorr_V_SHE"]
        self.assertTrue(ecorr_m.gp_subsampled)
        self.assertEqual(ecorr_m._per_metal_residual_gps["Cu"].n_samples, 15)
        self.assertEqual(ecorr_m._per_metal_residual_gps["Co"].n_samples, 15)

        q_cu = MLFeatureQuery(metal="Cu", pH=5.0, temperature_C=30.0, oxidizer_type="H2O2", oxidizer_concentration=1.0)
        self.assertAlmostEqual(
            p1.predict(q_cu).Ecorr_V_SHE.mean,
            p2.predict(q_cu).Ecorr_V_SHE.mean,
            places=12,
        )

        # When each Metal <= max_gp_samples (e.g. 24 <= 30, 18 <= 30), even though total=42 > 30, NO subsampling occurs
        p_full = HybridMLPredictor(
            model_family="rf_gp_ensemble",
            random_state=42,
            n_estimators=25,
            max_gp_samples=30,
        ).fit(db)
        ecorr_full = p_full.models["Ecorr_V_SHE"]
        self.assertFalse(ecorr_full.gp_subsampled)
        self.assertEqual(ecorr_full._per_metal_residual_gps["Cu"].n_samples, 24)
        self.assertEqual(ecorr_full._per_metal_residual_gps["Co"].n_samples, 18)

    def test_cross_validate_rf_gp_ensemble_is_leak_free_and_finite(self) -> None:
        db = self._build_synthetic_db(n_cu=20, n_co=15, n_rare=2)
        predictor = HybridMLPredictor(
            model_family="rf_gp_ensemble",
            random_state=42,
            n_estimators=20,
        )
        metrics = predictor.cross_validate(db, cv_strategy="stratified_5fold")
        self.assertEqual(len(metrics), 4)
        for m in metrics:
            self.assertEqual(m.model_family, "rf_gp_ensemble")
            self.assertTrue(np.isfinite(m.mae))
            self.assertTrue(np.isfinite(m.rmse))
            self.assertTrue(np.isfinite(m.r2))
            self.assertGreaterEqual(m.rmse, m.mae - 1e-9)

    def test_smooth_continuous_sweep_and_non_finite_oob_resilience(self) -> None:
        db = self._build_synthetic_db(n_cu=24, n_co=12, n_rare=2)
        pred_ens = HybridMLPredictor(
            model_family="rf_gp_ensemble",
            random_state=42,
            n_estimators=30,
        ).fit(db)

        # Fine pH sweep: rf_gp_ensemble produces continuous varying predictions (not piecewise-constant RF plateaus)
        ph_grid = np.linspace(3.0, 7.0, 21)
        ecorr_sweep = [
            pred_ens.predict(
                MLFeatureQuery(
                    metal="Cu",
                    pH=float(ph),
                    temperature_C=30.0,
                    oxidizer_type="H2O2",
                    oxidizer_concentration=1.0,
                )
            ).Ecorr_V_SHE.mean
            for ph in ph_grid
        ]
        unique_diffs = np.abs(np.diff(ecorr_sweep))
        self.assertGreater(int(np.sum(unique_diffs > 1e-6)), 15)

        # Very small n_estimators (n_estimators=2) produces non-finite OOB entries on some rows;
        # Stage 2 must filter them cleanly without raising or emitting NaN predictions.
        small_ens = HybridMLPredictor(
            model_family="rf_gp_ensemble",
            random_state=42,
            n_estimators=2,
        ).fit(db)
        out = small_ens.predict(
            MLFeatureQuery(
                metal="Cu",
                pH=5.0,
                temperature_C=30.0,
                oxidizer_type="H2O2",
                oxidizer_concentration=1.0,
            )
        )
        self.assertTrue(np.isfinite(out.Ecorr_V_SHE.mean))
        self.assertTrue(np.isfinite(out.Ecorr_V_SHE.std))
        self.assertTrue(np.isfinite(out.log10_icorr_uA_cm2.mean))
        self.assertTrue(np.isfinite(out.log10_icorr_uA_cm2.std))


if __name__ == "__main__":
    unittest.main()
