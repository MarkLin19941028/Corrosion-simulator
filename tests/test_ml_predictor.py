"""Behavioral tests for Phase 2 Hybrid Physics-ML predictor, MC uncertainty, and Applicability Domain."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
        unseen_metal_res = checker.evaluate(ExposureCondition(metal="Ni", pH=4.0, temperature_C=25.0))
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
        for family in ("random_forest", "gradient_boosting", "gaussian_process"):
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


if __name__ == "__main__":
    unittest.main()
