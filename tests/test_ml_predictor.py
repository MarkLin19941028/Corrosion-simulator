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
            "extra_trees",
            "gaussian_process",
            "rf_gp_ensemble",
            "hist_gradient_boosting",
            "gradient_boosting",
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


if __name__ == "__main__":
    unittest.main()
