"""Direct tests for corrosion_simulator.uncertainty and physics-to-ML dependency isolation."""

from __future__ import annotations

import ast
import math
from pathlib import Path
import unittest

import numpy as np

from corrosion_simulator.uncertainty import (
    DEFAULT_MASS_TRANSPORT_LIMIT_UA_CM2,
    ECORR_THERMO_MAX_V_SHE,
    ECORR_THERMO_MIN_V_SHE,
    TAFEL_SLOPE_MAX_MV_DEC,
    TAFEL_SLOPE_MIN_MV_DEC,
    MLParameterPrediction,
    ParameterEstimate,
    clip_target_mean,
    propagate_uncertainty_monte_carlo,
    sample_clipped_icorr,
    sample_lognormal_tafel_slope,
)


def _imported_module_names(file_path: Path) -> list[str]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
            for alias in node.names:
                modules.append(f"{node.module}.{alias.name}")
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return modules


class TestUncertaintyModule(unittest.TestCase):
    def test_physics_and_uncertainty_modules_have_zero_ml_or_sklearn_imports(self) -> None:
        pkg_dir = Path(__file__).resolve().parents[1] / "corrosion_simulator"
        for filename in ("uncertainty.py", "physics.py", "galvanic.py", "kinetics.py"):
            imports = _imported_module_names(pkg_dir / filename)
            self.assertFalse(
                any("ml_predictor" in m or "sklearn" in m for m in imports),
                msg=f"{filename} unexpectedly imports ML/sklearn: {imports}",
            )

    def test_lab_calibration_has_zero_private_ml_predictor_imports(self) -> None:
        lab_cal_path = Path(__file__).resolve().parents[1] / "corrosion_simulator" / "lab_calibration.py"
        tree = ast.parse(lab_cal_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "corrosion_simulator.ml_predictor":
                for alias in node.names:
                    self.assertFalse(
                        alias.name.startswith("_"),
                        msg=f"lab_calibration.py must not import private symbol {alias.name!r} from ml_predictor",
                    )

    def test_clip_target_mean_enforces_physical_bounds(self) -> None:
        self.assertEqual(clip_target_mean("Ecorr_V_SHE", 10.0), ECORR_THERMO_MAX_V_SHE)
        self.assertEqual(clip_target_mean("Ecorr_V_SHE", -10.0), ECORR_THERMO_MIN_V_SHE)
        self.assertEqual(clip_target_mean("beta_a_mV_dec", 5.0), TAFEL_SLOPE_MIN_MV_DEC)
        self.assertEqual(clip_target_mean("beta_c_mV_dec", 900.0), TAFEL_SLOPE_MAX_MV_DEC)
        self.assertEqual(clip_target_mean("log10_icorr_uA_cm2", 9.0), 6.0)
        self.assertEqual(clip_target_mean("log10_icorr_uA_cm2", -9.0), -6.0)
        with self.assertRaises(ValueError):
            clip_target_mean("unknown_target", 1.0)

    def test_sampling_and_propagation_respect_adr0003_guardrails(self) -> None:
        rng = np.random.default_rng(7)
        slopes = sample_lognormal_tafel_slope(60.0, 200.0, 500, rng)
        self.assertTrue(np.all(slopes >= TAFEL_SLOPE_MIN_MV_DEC))
        self.assertTrue(np.all(slopes <= TAFEL_SLOPE_MAX_MV_DEC))

        log_i, icorr = sample_clipped_icorr(3.0, 2.5, 500, rng)
        self.assertTrue(np.all(icorr > 0.0))
        self.assertTrue(np.all(icorr <= DEFAULT_MASS_TRANSPORT_LIMIT_UA_CM2))
        self.assertTrue(np.all(log_i <= math.log10(DEFAULT_MASS_TRANSPORT_LIMIT_UA_CM2) + 1e-12))

        pred = MLParameterPrediction(
            metal="Cu",
            Ecorr_V_SHE=ParameterEstimate(mean=0.25, std=0.05),
            log10_icorr_uA_cm2=ParameterEstimate(mean=0.5, std=0.2),
            icorr_uA_cm2_median=10.0**0.5,
            beta_a_mV_dec=ParameterEstimate(mean=60.0, std=8.0),
            beta_c_mV_dec=ParameterEstimate(mean=120.0, std=12.0),
            model_family="rf_gp_ensemble",
            applicability=None,
        )
        mc = propagate_uncertainty_monte_carlo(pred, n_draws=200, seed=42)
        self.assertEqual(mc.n_draws, 200)
        self.assertTrue(np.all(mc.icorr_samples_uA_cm2 > 0.0))
        self.assertTrue(np.all(mc.cr_nm_min_samples > 0.0))

    def test_hybrid_ml_predictor_batch_prior_interface_and_target_mask_fallback(self) -> None:
        import pandas as pd

        from corrosion_simulator.database import HarmonizedDatabase
        from corrosion_simulator.ml_predictor import (
            ML_TARGET_NAMES,
            HybridMLPredictor,
            extract_target_valid_mask,
        )

        unfitted = HybridMLPredictor(model_family="random_forest", n_estimators=10, random_state=42)
        self.assertFalse(unfitted.is_fitted)
        with self.assertRaises(RuntimeError):
            unfitted.predict_prior_means_dataframe(pd.DataFrame())

        db = HarmonizedDatabase.from_seed()
        fitted = unfitted.fit(db)
        self.assertTrue(fitted.is_fitted)
        df = db.ml_training_dataframe.head(4).reset_index(drop=True)
        priors = fitted.predict_prior_means_dataframe(df)
        self.assertEqual(set(priors.keys()), set(ML_TARGET_NAMES))
        for target_name, arr in priors.items():
            self.assertEqual(arr.shape, (len(df),))
            self.assertTrue(np.all(np.isfinite(arr)), msg=f"Non-finite prior for {target_name}")

        # Strict lab mask (fallback_min_measured=None) excludes imputed Tafel rows even when < 3 measured exist
        small_df = pd.DataFrame(
            {
                "beta_a_imputed": [False, True, True],
            }
        )
        y_vec = np.array([60.0, 65.0, 70.0])
        strict_mask = extract_target_valid_mask(
            small_df, "beta_a_mV_dec", y_vec, exclude_imputed_tafel=True, fallback_min_measured=None
        )
        fallback_mask = extract_target_valid_mask(
            small_df, "beta_a_mV_dec", y_vec, exclude_imputed_tafel=True, fallback_min_measured=3
        )
        self.assertEqual(strict_mask.tolist(), [True, False, False])
        self.assertEqual(fallback_mask.tolist(), [True, True, True])


if __name__ == "__main__":
    unittest.main()
