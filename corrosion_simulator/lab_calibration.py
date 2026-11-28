"""Parameter-level laboratory calibration: Delta-Learning Zero-Mean GP residual layer and Lab-Only Local GP."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Final, Literal, cast
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from corrosion_simulator.database import (
    ExposureCondition,
    HarmonizedDatabase,
)
from corrosion_simulator.ml_predictor import (
    ML_TARGET_NAMES,
    ApplicabilityDomainChecker,
    ApplicabilityDomainResult,
    ConfidenceTier,
    HybridMLPredictor,
    MLFeatureQuery,
    PredictedModelFamily,
    ResidualConditionTransformer,
    extract_sample_weights,
    extract_target_valid_mask,
    extract_target_vectors,
    fit_zero_mean_residual_gp,
)
from corrosion_simulator.uncertainty import (
    MLParameterPrediction,
    ParameterEstimate,
    clip_target_mean,
)

if TYPE_CHECKING:
    from corrosion_simulator.artifact_repository import LabProjectDataset

CalibrationMode = Literal["literature_only", "delta_learning", "lab_only_gp"]
ResidualMethod = Literal["gp_residual", "constant_bias", "prior_only"]

CALIBRATION_MODES: Final[tuple[CalibrationMode, ...]] = (
    "literature_only",
    "delta_learning",
    "lab_only_gp",
)
CALIBRATION_MODE_LABELS: Final[dict[CalibrationMode, str]] = {
    "literature_only": "Literature Only",
    "delta_learning": "Literature + Lab Calibration",
    "lab_only_gp": "Lab Data Only",
}
RESIDUAL_METHOD_LABELS: Final[dict[ResidualMethod, str]] = {
    "gp_residual": "GP Residual",
    "constant_bias": "Constant Bias",
    "prior_only": "Prior Only",
}
_TARGET_DISPLAY_INFO: Final[dict[str, tuple[str, str]]] = {
    "Ecorr_V_SHE": ("E_corr", "V vs. SHE"),
    "log10_icorr_uA_cm2": ("log10(i_corr)", "log10(µA/cm²)"),
    "beta_a_mV_dec": ("beta_a", "mV/dec"),
    "beta_c_mV_dec": ("beta_c", "mV/dec"),
}
MIN_DELTA_GP_SAMPLES: Final[int] = 4
MIN_LAB_ONLY_GP_SAMPLES: Final[int] = 3

_CONSTANT_BIAS_FLOOR_STD: Final[dict[str, float]] = {
    "Ecorr_V_SHE": 0.03,
    "log10_icorr_uA_cm2": 0.10,
    "beta_a_mV_dec": 8.0,
    "beta_c_mV_dec": 8.0,
}

_LAB_ONLY_FALLBACK_FLOOR_STD: Final[dict[str, float]] = {
    "Ecorr_V_SHE": 0.15,
    "log10_icorr_uA_cm2": 0.50,
    "beta_a_mV_dec": 30.0,
    "beta_c_mV_dec": 30.0,
}


class InsufficientLabSamplesError(ValueError):
    """Raised when Lab-Only Local GP lacks the minimum required measured laboratory records."""


@dataclass(frozen=True)
class TargetResidualBreakdown:
    """Per-target prior estimate, lab residual Δ, calibrated estimate, and calibration method."""

    target: str
    prior: ParameterEstimate
    delta: ParameterEstimate
    calibrated: ParameterEstimate
    method: ResidualMethod
    n_lab: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "prior_mean": float(self.prior.mean),
            "prior_std": float(self.prior.std),
            "delta_mean": float(self.delta.mean),
            "delta_std": float(self.delta.std),
            "calibrated_mean": float(self.calibrated.mean),
            "calibrated_std": float(self.calibrated.std),
            "method": self.method,
            "n_lab": int(self.n_lab),
        }


@dataclass(frozen=True)
class ResidualBreakdown:
    """Structured four-parameter transparency breakdown for laboratory calibration."""

    metal: str
    calibration_mode: CalibrationMode
    lab_project: str | None
    Ecorr_V_SHE: TargetResidualBreakdown
    log10_icorr_uA_cm2: TargetResidualBreakdown
    beta_a_mV_dec: TargetResidualBreakdown
    beta_c_mV_dec: TargetResidualBreakdown

    def by_target(self) -> dict[str, TargetResidualBreakdown]:
        return {
            "Ecorr_V_SHE": self.Ecorr_V_SHE,
            "log10_icorr_uA_cm2": self.log10_icorr_uA_cm2,
            "beta_a_mV_dec": self.beta_a_mV_dec,
            "beta_c_mV_dec": self.beta_c_mV_dec,
        }

    def __getitem__(self, target: str) -> TargetResidualBreakdown:
        mapping = self.by_target()
        if target not in mapping:
            raise KeyError(f"Unknown target {target!r}. Expected one of {tuple(mapping)}.")
        return mapping[target]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for target_name in ML_TARGET_NAMES:
            item = self[target_name]
            label, unit = _TARGET_DISPLAY_INFO[target_name]
            rows.append(
                {
                    "Parameter": label,
                    "Unit": unit,
                    "Prior Estimate": float(item.prior.mean),
                    "Prior ±1σ": float(item.prior.std),
                    "Lab Residual (Δ)": float(item.delta.mean),
                    "Residual ±1σ": float(item.delta.std),
                    "Calibrated Final": float(item.calibrated.mean),
                    "Combined ±1σ": float(item.calibrated.std),
                    "Method": RESIDUAL_METHOD_LABELS.get(item.method, item.method),
                    "Lab Samples (n)": int(item.n_lab),
                }
            )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metal": self.metal,
            "calibration_mode": self.calibration_mode,
            "lab_project": self.lab_project,
            "targets": {t: self[t].to_dict() for t in ML_TARGET_NAMES},
        }


def _extract_gp_kernel_variances(kernel: Any) -> tuple[float, float]:
    """Extract ``(prior_signal_var, noise_var)`` from a fitted ``Constant * Matern + WhiteKernel``."""
    signal_var = 0.0
    noise_var = 0.0
    k1 = getattr(kernel, "k1", None)
    k2 = getattr(kernel, "k2", None)
    if k1 is not None:
        c_kernel = getattr(k1, "k1", None)
        if c_kernel is not None and hasattr(c_kernel, "constant_value"):
            signal_var = max(float(c_kernel.constant_value), 0.0)
        elif hasattr(k1, "constant_value"):
            signal_var = max(float(k1.constant_value), 0.0)
    if k2 is not None and hasattr(k2, "noise_level"):
        noise_var = max(float(k2.noise_level), 0.0)
    return signal_var, noise_var


@dataclass(frozen=True)
class _FittedResidualTarget:
    method: ResidualMethod
    n_lab: int
    transformer: ResidualConditionTransformer | None = None
    gp_model: GaussianProcessRegressor | None = None
    prior_residual_var: float = 0.0
    noise_var: float = 0.0
    bias_mean: float = 0.0
    bias_std: float = 0.0


@dataclass(frozen=True)
class _FittedLabOnlyTarget:
    n_lab: int
    is_fallback: bool
    transformer: ResidualConditionTransformer | None = None
    gp_model: GaussianProcessRegressor | None = None
    fallback_mean: float = 0.0
    fallback_std: float = 0.0
    warning: str | None = None


class LabCalibratedPredictor:
    """Parameter-level calibration wrapper supporting ``literature_only``, ``delta_learning``, and ``lab_only_gp``."""

    def __init__(
        self,
        base_predictor: HybridMLPredictor | None = None,
        *,
        calibration_mode: CalibrationMode = "delta_learning",
        lab_project: str | None = None,
        random_state: int = 42,
        exclude_imputed_tafel: bool = True,
        use_quality_weights: bool = True,
        allow_fallback: bool = True,
    ) -> None:
        if calibration_mode not in CALIBRATION_MODES:
            raise ValueError(
                f"Unsupported calibration_mode {calibration_mode!r}. Supported: {CALIBRATION_MODES}"
            )
        self.base_predictor = base_predictor
        self.calibration_mode: CalibrationMode = calibration_mode
        self.lab_project: str | None = lab_project
        self.random_state = int(random_state)
        self.exclude_imputed_tafel = bool(exclude_imputed_tafel)
        self.use_quality_weights = bool(use_quality_weights)
        self.allow_fallback = bool(allow_fallback)

        self.model_family: PredictedModelFamily = (
            base_predictor.model_family if base_predictor is not None else "lab_local_gp"
        )
        self._residual_models: dict[tuple[str, str], _FittedResidualTarget] = {}
        self._delta_lab_ad: ApplicabilityDomainChecker | None = None
        self._lab_only_ad: ApplicabilityDomainChecker | None = None
        self._lab_only_targets: dict[str, _FittedLabOnlyTarget] = {}
        self._lab_only_metal_targets: dict[tuple[str, str], _FittedLabOnlyTarget] = {}
        self._lab_only_empty_fallback: bool = False
        self._lab_only_n_rows: int = 0
        self._is_fitted: bool = False

    def fit(
        self,
        lab_data: HarmonizedDatabase | LabProjectDataset | None = None,
    ) -> LabCalibratedPredictor:
        """Fit residual models (``delta_learning``) or standalone local GPs (``lab_only_gp``)."""
        lab_db: HarmonizedDatabase | None
        if lab_data is None:
            lab_db = None
        elif hasattr(lab_data, "database") and hasattr(lab_data, "project_name"):
            lab_db = lab_data.database
            if self.lab_project is None:
                self.lab_project = str(lab_data.project_name)
        elif isinstance(lab_data, HarmonizedDatabase):
            lab_db = lab_data
        else:
            raise TypeError(
                f"Expected HarmonizedDatabase, LabProjectDataset, or None, got {type(lab_data).__name__}"
            )

        if self.calibration_mode == "literature_only":
            if self.base_predictor is None or not self.base_predictor.is_fitted:
                raise RuntimeError(
                    "literature_only mode requires a fitted base HybridMLPredictor."
                )
            self._is_fitted = True
            return self

        if self.calibration_mode == "delta_learning":
            self._fit_delta_learning(lab_db)
            self._is_fitted = True
            return self

        self._fit_lab_only_gp(lab_db)
        self._is_fitted = True
        return self

    def _fit_delta_learning(self, lab_db: HarmonizedDatabase | None) -> None:
        if self.base_predictor is None or not self.base_predictor.is_fitted:
            raise RuntimeError("delta_learning mode requires a fitted base HybridMLPredictor.")

        self._residual_models = {}
        self._delta_lab_ad = None
        if lab_db is None:
            return
        df = lab_db.ml_training_dataframe.reset_index(drop=True)
        if df.empty:
            return

        self._delta_lab_ad = ApplicabilityDomainChecker(
            k=min(3, max(1, len(df))),
            min_high_confidence_samples=4,
            min_low_confidence_samples=1,
        ).fit(df)

        prior_means = self.base_predictor.predict_prior_means_dataframe(df)
        targets = extract_target_vectors(df)
        weights_all = extract_sample_weights(df, use_quality_weights=self.use_quality_weights)
        metals_col = df["metal"].astype(str).str.strip()

        for metal_idx, metal in enumerate(sorted(set(metals_col))):
            metal_mask = (metals_col == metal).to_numpy()
            df_metal = df.loc[metal_mask].reset_index(drop=True)
            w_metal = (
                weights_all[metal_mask]
                if weights_all is not None
                else np.ones(len(df_metal), dtype=float)
            )

            for t_idx, target_name in enumerate(ML_TARGET_NAMES):
                y_metal = targets[target_name][metal_mask]
                valid_mask = extract_target_valid_mask(
                    df_metal,
                    target_name,
                    y_metal,
                    exclude_imputed_tafel=self.exclude_imputed_tafel,
                    fallback_min_measured=None,
                )
                n_valid = int(np.sum(valid_mask))
                key = (metal, target_name)

                if n_valid == 0:
                    self._residual_models[key] = _FittedResidualTarget(
                        method="prior_only",
                        n_lab=0,
                    )
                    continue

                y_prior_valid = prior_means[target_name][metal_mask][valid_mask]
                residuals = y_metal[valid_mask] - y_prior_valid
                w_valid = np.clip(w_metal[valid_mask], 0.05, 1.0)

                if n_valid < MIN_DELTA_GP_SAMPLES:
                    bias_mean = float(np.average(residuals, weights=w_valid))
                    floor_std = _CONSTANT_BIAS_FLOOR_STD[target_name]
                    if n_valid == 1:
                        bias_std = max(floor_std, 0.35 * abs(bias_mean))
                    else:
                        weighted_var = float(
                            np.average((residuals - bias_mean) ** 2, weights=w_valid)
                        )
                        bias_std = max(
                            math.sqrt(max(weighted_var, 0.0)),
                            floor_std / math.sqrt(n_valid),
                            0.20 * abs(bias_mean) / math.sqrt(n_valid),
                        )
                    self._residual_models[key] = _FittedResidualTarget(
                        method="constant_bias",
                        n_lab=n_valid,
                        bias_mean=bias_mean,
                        bias_std=float(bias_std),
                    )
                    continue

                df_valid = df_metal.loc[valid_mask].reset_index(drop=True)
                transformer, gpr = fit_zero_mean_residual_gp(
                    df_valid,
                    residuals,
                    w_valid,
                    target_name=target_name,
                    length_scale_bounds=(0.2, 2.0),
                    alpha_scale=1e-6,
                    n_restarts_optimizer=2,
                    random_state=self.random_state + metal_idx * 41 + t_idx * 13,
                )

                prior_var, noise_var = _extract_gp_kernel_variances(gpr.kernel_)
                self._residual_models[key] = _FittedResidualTarget(
                    method="gp_residual",
                    n_lab=n_valid,
                    transformer=transformer,
                    gp_model=gpr,
                    prior_residual_var=prior_var,
                    noise_var=noise_var,
                )

    def _fit_single_lab_only_gp(
        self,
        df_sub: pd.DataFrame,
        y_sub: np.ndarray,
        w_sub: np.ndarray,
        *,
        include_metal: bool,
        seed: int,
    ) -> tuple[ResidualConditionTransformer, GaussianProcessRegressor]:
        transformer = ResidualConditionTransformer(include_metal=include_metal)
        X_low = transformer.fit_transform(df_sub)
        kernel = (
            ConstantKernel(
                constant_value=1.0,
                constant_value_bounds=(1e-3, 1e2),
            )
            * Matern(
                length_scale=1.0,
                length_scale_bounds=(0.1, 20.0),
                nu=2.5,
            )
            + WhiteKernel(
                noise_level=0.05,
                noise_level_bounds=(1e-6, 5.0),
            )
        )
        alpha = 1e-5 * ((1.0 / w_sub) ** 2 - 1.0) + 1e-9
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=True,
            random_state=seed,
            n_restarts_optimizer=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gpr.fit(X_low, y_sub)
        return transformer, gpr

    def _fit_lab_only_gp(self, lab_db: HarmonizedDatabase | None) -> None:
        self._lab_only_targets = {}
        self._lab_only_metal_targets = {}
        self._lab_only_ad = None
        self._lab_only_empty_fallback = False

        project_label = self.lab_project or "selected lab project"
        if lab_db is None or lab_db.ml_training_dataframe.empty:
            self._lab_only_n_rows = 0
            msg = (
                f"Lab-Only Local GP requires at least {MIN_LAB_ONLY_GP_SAMPLES} laboratory records "
                f"in project {project_label!r}, but found 0 records."
            )
            if not self.allow_fallback or self.base_predictor is None or not self.base_predictor.is_fitted:
                raise InsufficientLabSamplesError(
                    f"{msg} Add >= {MIN_LAB_ONLY_GP_SAMPLES} records to data/lab/ or switch to 'delta_learning'."
                )
            self._lab_only_empty_fallback = True
            return

        df = lab_db.ml_training_dataframe.reset_index(drop=True)
        self._lab_only_n_rows = len(df)
        self._lab_only_ad = ApplicabilityDomainChecker(
            k=min(3, max(1, len(df))),
            min_high_confidence_samples=5,
            min_low_confidence_samples=MIN_LAB_ONLY_GP_SAMPLES,
        ).fit(df)

        targets = extract_target_vectors(df)
        weights_all = extract_sample_weights(df, use_quality_weights=self.use_quality_weights)
        metals_col = df["metal"].astype(str).str.strip()

        for t_idx, target_name in enumerate(ML_TARGET_NAMES):
            y_all = targets[target_name]
            valid_mask = extract_target_valid_mask(
                df,
                target_name,
                y_all,
                exclude_imputed_tafel=self.exclude_imputed_tafel,
                fallback_min_measured=None,
            )
            n_valid = int(np.sum(valid_mask))

            if n_valid < MIN_LAB_ONLY_GP_SAMPLES:
                msg = (
                    f"Lab-Only Local GP requires at least {MIN_LAB_ONLY_GP_SAMPLES} valid lab records "
                    f"for {target_name}, but project {project_label!r} has n={n_valid}. "
                    f"Add at least {MIN_LAB_ONLY_GP_SAMPLES - n_valid} more measurement(s) or switch to 'delta_learning'."
                )
                if not self.allow_fallback:
                    raise InsufficientLabSamplesError(msg)

                if n_valid > 0:
                    y_sub = y_all[valid_mask]
                    w_sub = (
                        np.clip(weights_all[valid_mask], 0.05, 1.0)
                        if weights_all is not None
                        else np.ones(n_valid, dtype=float)
                    )
                    fb_mean = float(np.average(y_sub, weights=w_sub))
                    fb_std = max(
                        float(np.std(y_sub)) if n_valid > 1 else 0.0,
                        _LAB_ONLY_FALLBACK_FLOOR_STD[target_name],
                    )
                else:
                    finite_mask = np.isfinite(y_all)
                    if int(np.sum(finite_mask)) > 0:
                        fb_mean = float(np.mean(y_all[finite_mask]))
                    elif target_name == "beta_a_mV_dec":
                        fb_mean = 60.0
                    elif target_name == "beta_c_mV_dec":
                        fb_mean = 120.0
                    else:
                        fb_mean = 0.0
                    fb_std = _LAB_ONLY_FALLBACK_FLOOR_STD[target_name]

                self._lab_only_targets[target_name] = _FittedLabOnlyTarget(
                    n_lab=n_valid,
                    is_fallback=True,
                    fallback_mean=clip_target_mean(target_name, fb_mean),
                    fallback_std=float(fb_std),
                    warning=msg,
                )
                continue

            df_valid = df.loc[valid_mask].reset_index(drop=True)
            y_target = y_all[valid_mask]
            w_target = (
                np.clip(weights_all[valid_mask], 0.05, 1.0)
                if weights_all is not None
                else np.ones(n_valid, dtype=float)
            )

            # Shared low-dimensional Exposure Condition GP (handles unseen or low-n Metals cleanly)
            shared_transformer, shared_gpr = self._fit_single_lab_only_gp(
                df_valid,
                y_target,
                w_target,
                include_metal=True,
                seed=self.random_state + t_idx * 17,
            )
            self._lab_only_targets[target_name] = _FittedLabOnlyTarget(
                n_lab=n_valid,
                is_fallback=False,
                transformer=shared_transformer,
                gp_model=shared_gpr,
            )

            # Fit dedicated per-Metal low-dimensional GPs when multiple Metals have >= 3 valid records
            valid_metals = sorted(set(metals_col[valid_mask]))
            if len(valid_metals) > 1:
                for m_idx, metal in enumerate(valid_metals):
                    m_mask = valid_mask & (metals_col == metal).to_numpy()
                    n_m = int(np.sum(m_mask))
                    if n_m < MIN_LAB_ONLY_GP_SAMPLES:
                        continue
                    df_m = df.loc[m_mask].reset_index(drop=True)
                    y_m = y_all[m_mask]
                    w_m = (
                        np.clip(weights_all[m_mask], 0.05, 1.0)
                        if weights_all is not None
                        else np.ones(n_m, dtype=float)
                    )
                    m_transformer, m_gpr = self._fit_single_lab_only_gp(
                        df_m,
                        y_m,
                        w_m,
                        include_metal=False,
                        seed=self.random_state + t_idx * 17 + (m_idx + 1) * 53,
                    )
                    self._lab_only_metal_targets[(metal, target_name)] = _FittedLabOnlyTarget(
                        n_lab=n_m,
                        is_fallback=False,
                        transformer=m_transformer,
                        gp_model=m_gpr,
                    )

    def predict(self, query: ExposureCondition | MLFeatureQuery) -> MLParameterPrediction:
        """Predict calibrated four-parameter electrochemical estimates and uncertainty."""
        if not self._is_fitted:
            raise RuntimeError("LabCalibratedPredictor must be fitted before predict().")

        q = MLFeatureQuery.from_exposure_condition(query)
        metal = q.metal.strip()

        if self.calibration_mode == "literature_only":
            assert self.base_predictor is not None
            prior_pred = self.base_predictor.predict(q)
            return MLParameterPrediction(
                metal=prior_pred.metal,
                Ecorr_V_SHE=prior_pred.Ecorr_V_SHE,
                log10_icorr_uA_cm2=prior_pred.log10_icorr_uA_cm2,
                icorr_uA_cm2_median=prior_pred.icorr_uA_cm2_median,
                beta_a_mV_dec=prior_pred.beta_a_mV_dec,
                beta_c_mV_dec=prior_pred.beta_c_mV_dec,
                model_family=prior_pred.model_family,
                applicability=prior_pred.applicability,
                calibration_mode="literature_only",
                lab_project=self.lab_project,
                residual_breakdown=None,
            )

        if self.calibration_mode == "delta_learning":
            return self._predict_delta_learning(q, metal)

        return self._predict_lab_only_gp(q, metal)

    def _predict_delta_learning(self, q: MLFeatureQuery, metal: str) -> MLParameterPrediction:
        assert self.base_predictor is not None
        prior_pred = self.base_predictor.predict(q)
        prior_map = {
            "Ecorr_V_SHE": prior_pred.Ecorr_V_SHE,
            "log10_icorr_uA_cm2": prior_pred.log10_icorr_uA_cm2,
            "beta_a_mV_dec": prior_pred.beta_a_mV_dec,
            "beta_c_mV_dec": prior_pred.beta_c_mV_dec,
        }

        breakdowns: dict[str, TargetResidualBreakdown] = {}
        calibrated_map: dict[str, ParameterEstimate] = {}

        for target_name in ML_TARGET_NAMES:
            prior_est = prior_map[target_name]
            fitted = self._residual_models.get((metal, target_name))
            if fitted is None or fitted.method == "prior_only":
                delta_est = ParameterEstimate(mean=0.0, std=0.0)
                cal_est = ParameterEstimate(
                    mean=clip_target_mean(target_name, prior_est.mean),
                    std=float(max(prior_est.std, 0.0)),
                )
                breakdowns[target_name] = TargetResidualBreakdown(
                    target=target_name,
                    prior=prior_est,
                    delta=delta_est,
                    calibrated=cal_est,
                    method="prior_only",
                    n_lab=0,
                )
                calibrated_map[target_name] = cal_est
                continue

            if fitted.method == "constant_bias":
                delta_est = ParameterEstimate(
                    mean=float(fitted.bias_mean),
                    std=float(max(fitted.bias_std, 0.0)),
                )
                cal_std = math.hypot(max(float(prior_est.std), 0.0), delta_est.std)
            else:
                assert fitted.transformer is not None and fitted.gp_model is not None
                X_q = fitted.transformer.transform_query(q)
                d_mean, d_std = cast(
                    tuple[np.ndarray, np.ndarray],
                    fitted.gp_model.predict(X_q, return_std=True),
                )
                d_std_val = max(float(d_std[0]), 0.0)
                delta_est = ParameterEstimate(
                    mean=float(d_mean[0]),
                    std=d_std_val,
                )
                epistemic_var = max(d_std_val**2 - fitted.noise_var, 0.0)
                coverage = float(
                    np.clip(
                        1.0 - epistemic_var / max(fitted.prior_residual_var, 1e-12),
                        0.0,
                        1.0,
                    )
                )
                prior_std_val = max(float(prior_est.std), 0.0)
                cal_std = math.sqrt((1.0 - coverage) * (prior_std_val**2) + d_std_val**2)

            cal_mean = clip_target_mean(target_name, prior_est.mean + delta_est.mean)
            cal_est = ParameterEstimate(mean=cal_mean, std=float(cal_std))
            breakdowns[target_name] = TargetResidualBreakdown(
                target=target_name,
                prior=prior_est,
                delta=delta_est,
                calibrated=cal_est,
                method=fitted.method,
                n_lab=fitted.n_lab,
            )
            calibrated_map[target_name] = cal_est

        breakdown = ResidualBreakdown(
            metal=metal,
            calibration_mode="delta_learning",
            lab_project=self.lab_project,
            Ecorr_V_SHE=breakdowns["Ecorr_V_SHE"],
            log10_icorr_uA_cm2=breakdowns["log10_icorr_uA_cm2"],
            beta_a_mV_dec=breakdowns["beta_a_mV_dec"],
            beta_c_mV_dec=breakdowns["beta_c_mV_dec"],
        )

        lit_ad = prior_pred.applicability
        lab_ad = self._delta_lab_ad.evaluate(q) if self._delta_lab_ad is not None else None
        n_lab_samples = lab_ad.n_metal_training_samples if lab_ad is not None else 0
        lab_in_box = lab_ad.in_bounding_box if lab_ad is not None else False
        lab_knn = lab_ad.knn_mean_distance if lab_ad is not None else float("inf")
        project_label = self.lab_project or "selected lab project"

        if n_lab_samples == 0:
            anchoring_status = "No Lab Coverage"
        elif lab_in_box and lab_knn <= 0.75:
            anchoring_status = "Lab-Anchored"
        else:
            anchoring_status = "Reverting to Literature Prior"

        tier: ConfidenceTier = lit_ad.confidence_tier
        ad_warnings = list(lit_ad.warnings)
        ad_recs = list(lit_ad.experimental_recommendations)

        if anchoring_status == "Lab-Anchored":
            if n_lab_samples >= MIN_DELTA_GP_SAMPLES and lab_knn <= 0.35:
                tier = "High"
            elif tier == "Low":
                tier = "Medium"
            if not lit_ad.in_bounding_box:
                ad_warnings.insert(
                    0,
                    f"Lab-Anchored ({project_label}, n={n_lab_samples}, lab k-NN distance={lab_knn:.3f}): "
                    f"{metal} query is covered by laboratory measurements despite lying outside the literature domain.",
                )
        elif anchoring_status == "Reverting to Literature Prior":
            tier = lit_ad.confidence_tier
            knn_txt = f"{lab_knn:.3f}" if math.isfinite(lab_knn) else "inf"
            ad_warnings.append(
                f"Reverting to Literature Prior for {metal} ({project_label}, n={n_lab_samples}, "
                f"lab_in_box={lab_in_box}, lab k-NN distance={knn_txt}): query lies outside or far from "
                "lab coverage, so residual Δ → 0 with expanded combined uncertainty."
            )
            if lab_ad is not None:
                for rec in lab_ad.experimental_recommendations:
                    if rec not in ad_recs:
                        ad_recs.append(rec)
        else:
            ad_warnings.append(
                f"No Lab Coverage for {metal} in project {project_label!r} (n=0); using literature prior unchanged."
            )
            if lab_ad is not None:
                for rec in lab_ad.experimental_recommendations:
                    if rec not in ad_recs:
                        ad_recs.append(rec)

        adapted_ad = ApplicabilityDomainResult(
            metal=lit_ad.metal,
            confidence_tier=tier,
            in_bounding_box=lit_ad.in_bounding_box,
            out_of_box_features=lit_ad.out_of_box_features,
            unseen_categories=lit_ad.unseen_categories,
            knn_mean_distance=lit_ad.knn_mean_distance,
            knn_k=lit_ad.knn_k,
            n_metal_training_samples=lit_ad.n_metal_training_samples,
            warnings=tuple(dict.fromkeys(ad_warnings)),
            experimental_recommendations=tuple(dict.fromkeys(ad_recs)),
            ad_mode="delta_learning",
            lab_in_bounding_box=lab_in_box,
            lab_knn_distance=lab_knn,
            n_lab_samples=n_lab_samples,
            lab_anchoring_status=anchoring_status,
        )

        log_i_mean = calibrated_map["log10_icorr_uA_cm2"].mean
        return MLParameterPrediction(
            metal=metal,
            Ecorr_V_SHE=calibrated_map["Ecorr_V_SHE"],
            log10_icorr_uA_cm2=calibrated_map["log10_icorr_uA_cm2"],
            icorr_uA_cm2_median=float(10.0**log_i_mean),
            beta_a_mV_dec=calibrated_map["beta_a_mV_dec"],
            beta_c_mV_dec=calibrated_map["beta_c_mV_dec"],
            model_family=prior_pred.model_family,
            applicability=adapted_ad,
            calibration_mode="delta_learning",
            lab_project=self.lab_project,
            residual_breakdown=breakdown,
        )

    def _predict_lab_only_gp(self, q: MLFeatureQuery, metal: str) -> MLParameterPrediction:
        project_label = self.lab_project or "selected lab project"
        if self._lab_only_empty_fallback:
            assert self.base_predictor is not None
            prior_pred = self.base_predictor.predict(q)
            warn_msg = (
                f"Lab-Only Local GP requires at least {MIN_LAB_ONLY_GP_SAMPLES} laboratory records "
                f"in project {project_label!r}, but found 0; temporarily fell back to literature prior."
            )
            rec_msg = (
                f"Add >= {MIN_LAB_ONLY_GP_SAMPLES} measured records for {metal} in data/lab/ "
                f"or switch calibration mode to 'delta_learning'."
            )
            ad_res = ApplicabilityDomainResult(
                metal=metal,
                confidence_tier="Low",
                in_bounding_box=False,
                out_of_box_features=prior_pred.applicability.out_of_box_features,
                unseen_categories=prior_pred.applicability.unseen_categories,
                knn_mean_distance=float("inf"),
                knn_k=0,
                n_metal_training_samples=0,
                warnings=(warn_msg,),
                experimental_recommendations=(rec_msg,),
                ad_mode="lab_only_gp",
                lab_in_bounding_box=False,
                lab_knn_distance=float("inf"),
                n_lab_samples=0,
                lab_anchoring_status="Outside Lab Domain",
            )
            return MLParameterPrediction(
                metal=metal,
                Ecorr_V_SHE=prior_pred.Ecorr_V_SHE,
                log10_icorr_uA_cm2=prior_pred.log10_icorr_uA_cm2,
                icorr_uA_cm2_median=prior_pred.icorr_uA_cm2_median,
                beta_a_mV_dec=prior_pred.beta_a_mV_dec,
                beta_c_mV_dec=prior_pred.beta_c_mV_dec,
                model_family="lab_local_gp",
                applicability=ad_res,
                calibration_mode="lab_only_gp",
                lab_project=self.lab_project,
                residual_breakdown=None,
            )

        assert self._lab_only_ad is not None
        ad_res = self._lab_only_ad.evaluate(q)

        estimates: dict[str, ParameterEstimate] = {}
        extra_warnings: list[str] = []
        extra_recs: list[str] = []
        has_fallback = False

        for target_name in ML_TARGET_NAMES:
            fitted = self._lab_only_metal_targets.get((metal, target_name)) or self._lab_only_targets[target_name]
            if fitted.is_fallback:
                has_fallback = True
                estimates[target_name] = ParameterEstimate(
                    mean=fitted.fallback_mean,
                    std=fitted.fallback_std,
                )
                if fitted.warning:
                    extra_warnings.append(fitted.warning)
                    extra_recs.append(
                        f"Measure at least {MIN_LAB_ONLY_GP_SAMPLES} valid lab curves including {target_name} "
                        f"in project {project_label!r} or use 'delta_learning'."
                    )
            else:
                assert fitted.transformer is not None and fitted.gp_model is not None
                X_q = fitted.transformer.transform_query(q)
                mean_arr, std_arr = cast(
                    tuple[np.ndarray, np.ndarray],
                    fitted.gp_model.predict(X_q, return_std=True),
                )
                estimates[target_name] = ParameterEstimate(
                    mean=clip_target_mean(target_name, float(mean_arr[0])),
                    std=max(float(std_arr[0]), 0.0),
                )

        anchoring_status = (
            "Lab-Anchored"
            if (
                ad_res.in_bounding_box
                and ad_res.n_metal_training_samples >= MIN_LAB_ONLY_GP_SAMPLES
                and not has_fallback
            )
            else "Outside Lab Domain"
        )
        ad_res = ApplicabilityDomainResult(
            metal=ad_res.metal,
            confidence_tier="Low" if has_fallback else ad_res.confidence_tier,
            in_bounding_box=ad_res.in_bounding_box,
            out_of_box_features=ad_res.out_of_box_features,
            unseen_categories=ad_res.unseen_categories,
            knn_mean_distance=ad_res.knn_mean_distance,
            knn_k=ad_res.knn_k,
            n_metal_training_samples=ad_res.n_metal_training_samples,
            warnings=(
                tuple(dict.fromkeys([*extra_warnings, *ad_res.warnings]))
                if has_fallback
                else ad_res.warnings
            ),
            experimental_recommendations=(
                tuple(dict.fromkeys([*extra_recs, *ad_res.experimental_recommendations]))
                if has_fallback
                else ad_res.experimental_recommendations
            ),
            ad_mode="lab_only_gp",
            lab_in_bounding_box=ad_res.in_bounding_box,
            lab_knn_distance=ad_res.knn_mean_distance,
            n_lab_samples=ad_res.n_metal_training_samples,
            lab_anchoring_status=anchoring_status,
        )

        log_i_mean = estimates["log10_icorr_uA_cm2"].mean
        return MLParameterPrediction(
            metal=metal,
            Ecorr_V_SHE=estimates["Ecorr_V_SHE"],
            log10_icorr_uA_cm2=estimates["log10_icorr_uA_cm2"],
            icorr_uA_cm2_median=float(10.0**log_i_mean),
            beta_a_mV_dec=estimates["beta_a_mV_dec"],
            beta_c_mV_dec=estimates["beta_c_mV_dec"],
            model_family="lab_local_gp",
            applicability=ad_res,
            calibration_mode="lab_only_gp",
            lab_project=self.lab_project,
            residual_breakdown=None,
        )
