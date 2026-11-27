"""Parameter-level laboratory calibration: Delta-Learning Zero-Mean GP residual layer and Lab-Only Local GP."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Final, Literal
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from corrosion_simulator.database import (
    ExposureCondition,
    HarmonizedDatabase,
    normalize_category_label as _clean_category,
)
from corrosion_simulator.ml_predictor import (
    ECORR_THERMO_MAX_V_SHE,
    ECORR_THERMO_MIN_V_SHE,
    ML_TARGET_NAMES,
    TAFEL_SLOPE_MAX_MV_DEC,
    TAFEL_SLOPE_MIN_MV_DEC,
    ApplicabilityDomainChecker,
    ApplicabilityDomainResult,
    ConfidenceTier,
    ElectrochemicalFeatureExtractor,
    HybridMLPredictor,
    MLFeatureQuery,
    MLParameterPrediction,
    ModelFamily,
    ParameterEstimate,
    _extract_sample_weights,
    _extract_target_vectors,
)
from corrosion_simulator.solvent_constants import compute_mixture_solvent_descriptors

if TYPE_CHECKING:
    from corrosion_simulator.lab_data import LabProjectDataset

CalibrationMode = Literal["literature_only", "delta_learning", "lab_only_gp"]
ResidualMethod = Literal["gp_residual", "constant_bias", "prior_only"]

CALIBRATION_MODES: Final[tuple[CalibrationMode, ...]] = (
    "literature_only",
    "delta_learning",
    "lab_only_gp",
)
CALIBRATION_MODE_LABELS: Final[dict[CalibrationMode, str]] = {
    "literature_only": "Literature Only",
    "delta_learning": "Delta-Learning (Prior + Lab Residual)",
    "lab_only_gp": "Lab-Only Local GP",
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

_DELTA_MIN_AMPLITUDE: Final[dict[str, float]] = {
    "Ecorr_V_SHE": 0.02,
    "log10_icorr_uA_cm2": 0.05,
    "beta_a_mV_dec": 3.0,
    "beta_c_mV_dec": 3.0,
}

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

_RESIDUAL_NUMERIC_FLOORS: Final[tuple[tuple[str, float], ...]] = (
    ("pH", 1.0),
    ("temperature_C", 10.0),
    ("log1p_oxidizer_conc", 0.3),
    ("log1p_acid_conc", 0.3),
    ("log1p_inhibitor_conc", 0.3),
    ("water_content_wt_pct", 15.0),
    ("effective_dielectric_constant", 10.0),
    ("solvent_polarity_index", 1.5),
    ("is_protic_solvent", 0.5),
)

_RESIDUAL_CAT_COLS: Final[tuple[str, ...]] = (
    "oxidizer_type",
    "acid_type",
    "inhibitor_type",
    "organic_solvent_type",
)


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


def _clip_target_mean(target_name: str, value: float) -> float:
    """Enforce physical and thermodynamic bounds on predicted parameter means."""
    if target_name == "Ecorr_V_SHE":
        return float(np.clip(value, ECORR_THERMO_MIN_V_SHE, ECORR_THERMO_MAX_V_SHE))
    if target_name in ("beta_a_mV_dec", "beta_c_mV_dec"):
        return float(np.clip(value, TAFEL_SLOPE_MIN_MV_DEC, TAFEL_SLOPE_MAX_MV_DEC))
    if target_name == "log10_icorr_uA_cm2":
        return float(np.clip(value, -6.0, 6.0))
    raise ValueError(f"Unknown target_name: {target_name!r}")


def _lab_target_valid_mask(
    df: pd.DataFrame,
    target_name: str,
    y_vec: np.ndarray,
    *,
    exclude_imputed_tafel: bool = True,
) -> np.ndarray:
    """Return mask of valid measured lab observations (excluding imputed Tafel defaults when present)."""
    finite_mask = np.isfinite(y_vec)
    if not exclude_imputed_tafel:
        return finite_mask

    imputed_col: str | None = None
    if target_name == "beta_a_mV_dec" and "beta_a_imputed" in df.columns:
        imputed_col = "beta_a_imputed"
    elif target_name == "beta_c_mV_dec" and "beta_c_imputed" in df.columns:
        imputed_col = "beta_c_imputed"

    if imputed_col is None:
        return finite_mask

    return finite_mask & ~df[imputed_col].fillna(False).astype(bool).to_numpy()


def _clean_base_solvent_label(value: Any) -> str:
    label = _clean_category(value)
    return "DI water" if label == "None" else label


class _ResidualConditionTransformer:
    """Transforms Exposure Conditions into physically floored standardized vectors for residual GPs."""

    def __init__(self) -> None:
        self._means: np.ndarray = np.array([], dtype=float)
        self._scales: np.ndarray = np.array([], dtype=float)
        self._include_orp: bool = False
        self._orp_median: float = 0.0
        self._cat_vocab: dict[str, list[str]] = {}
        self._is_fitted: bool = False

    def _extract_numeric_matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, dict[str, list[str]]]:
        ph = pd.to_numeric(df.get("pH", 7.0), errors="coerce").fillna(7.0).to_numpy(dtype=float)
        temp = pd.to_numeric(df.get("temperature_C", 25.0), errors="coerce").fillna(25.0).to_numpy(dtype=float)
        ox_c = np.maximum(
            pd.to_numeric(df.get("oxidizer_concentration", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float),
            0.0,
        )
        acid_c = np.maximum(
            pd.to_numeric(df.get("acid_concentration", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float),
            0.0,
        )
        inh_c = np.maximum(
            pd.to_numeric(df.get("inhibitor_concentration", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float),
            0.0,
        )
        water_pct = np.clip(
            pd.to_numeric(df.get("water_content_wt_pct", 100.0), errors="coerce").fillna(100.0).to_numpy(dtype=float),
            0.0,
            100.0,
        )

        org_vals = [
            _clean_category(v)
            for v in (df["organic_solvent_type"].tolist() if "organic_solvent_type" in df.columns else ["None"] * len(df))
        ]
        base_vals = [
            _clean_base_solvent_label(v)
            for v in (df["base_solvent"].tolist() if "base_solvent" in df.columns else ["DI water"] * len(df))
        ]
        solvent_desc = [
            compute_mixture_solvent_descriptors(w_val, org_val, base_val)
            for w_val, org_val, base_val in zip(water_pct, org_vals, base_vals)
        ]

        cols: list[np.ndarray] = [
            ph,
            temp,
            np.log10(1.0 + ox_c),
            np.log10(1.0 + acid_c),
            np.log10(1.0 + inh_c),
            water_pct,
            np.array([d[0] for d in solvent_desc], dtype=float),
            np.array([d[1] for d in solvent_desc], dtype=float),
            np.array([d[2] for d in solvent_desc], dtype=float),
        ]
        if self._include_orp:
            orp_vals = (
                pd.to_numeric(df["ORP_mV"], errors="coerce")
                .fillna(self._orp_median)
                .to_numpy(dtype=float)
                if "ORP_mV" in df.columns
                else np.full(len(df), self._orp_median, dtype=float)
            )
            cols.append(orp_vals)

        cats: dict[str, list[str]] = {}
        for col in _RESIDUAL_CAT_COLS:
            if col in df.columns:
                cats[col] = [_clean_category(v) for v in df[col].tolist()]
            else:
                cats[col] = ["None"] * len(df)
        return np.column_stack(cols), cats

    def fit(self, df: pd.DataFrame) -> _ResidualConditionTransformer:
        if "ORP_mV" in df.columns:
            orp_series = pd.to_numeric(df["ORP_mV"], errors="coerce").dropna()
            if not orp_series.empty:
                self._include_orp = True
                self._orp_median = float(orp_series.median())
            else:
                self._include_orp = False
        else:
            self._include_orp = False

        num_mat, cats = self._extract_numeric_matrix(df)
        floors = [floor for _, floor in _RESIDUAL_NUMERIC_FLOORS]
        if self._include_orp:
            floors.append(100.0)
        floor_arr = np.asarray(floors, dtype=float)

        self._means = np.mean(num_mat, axis=0)
        emp_std = np.std(num_mat, axis=0)
        self._scales = np.maximum(emp_std, floor_arr)

        self._cat_vocab = {
            col: sorted(set(cats[col]))
            for col in _RESIDUAL_CAT_COLS
        }
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("_ResidualConditionTransformer must be fitted before transform().")
        num_mat, cats = self._extract_numeric_matrix(df)
        num_scaled = (num_mat - self._means) / self._scales

        cat_cols: list[np.ndarray] = []
        for col in _RESIDUAL_CAT_COLS:
            vocab = self._cat_vocab[col]
            vals = np.asarray(cats[col], dtype=object)
            if len(vocab) > 1:
                for token in vocab:
                    cat_cols.append(((vals == token).astype(float)) * 1.5)
            unseen_mask = np.array([v not in vocab for v in vals], dtype=float) * 8.0
            cat_cols.append(unseen_mask)

        return np.column_stack([num_scaled] + cat_cols)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def transform_query(self, query: MLFeatureQuery) -> np.ndarray:
        return self.transform(pd.DataFrame([query.to_record_dict()]))


@dataclass(frozen=True)
class _FittedResidualTarget:
    method: ResidualMethod
    n_lab: int
    transformer: _ResidualConditionTransformer | None = None
    gp_model: GaussianProcessRegressor | None = None
    bias_mean: float = 0.0
    bias_std: float = 0.0


@dataclass(frozen=True)
class _FittedLabOnlyTarget:
    n_lab: int
    is_fallback: bool
    gp_model: GaussianProcessRegressor | None = None
    scaler: StandardScaler | None = None
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

        self.model_family: ModelFamily = (
            base_predictor.model_family if base_predictor is not None else "gaussian_process"
        )
        self._residual_models: dict[tuple[str, str], _FittedResidualTarget] = {}
        self._delta_lab_ad: ApplicabilityDomainChecker | None = None
        self._lab_only_extractor: ElectrochemicalFeatureExtractor | None = None
        self._lab_only_ad: ApplicabilityDomainChecker | None = None
        self._lab_only_targets: dict[str, _FittedLabOnlyTarget] = {}
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
            if self.base_predictor is None or not self.base_predictor._is_fitted:
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
        if self.base_predictor is None or not self.base_predictor._is_fitted:
            raise RuntimeError("delta_learning mode requires a fitted base HybridMLPredictor.")
        assert self.base_predictor.feature_extractor is not None

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

        X_prior = self.base_predictor.feature_extractor.transform(df)
        prior_means: dict[str, np.ndarray] = {}
        for target_name in ML_TARGET_NAMES:
            raw_mean, _ = self.base_predictor.models[target_name].predict_mean_std(X_prior)
            prior_means[target_name] = np.array(
                [_clip_target_mean(target_name, float(v)) for v in raw_mean],
                dtype=float,
            )

        targets = _extract_target_vectors(df)
        weights_all = _extract_sample_weights(df, use_quality_weights=self.use_quality_weights)
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
                valid_mask = _lab_target_valid_mask(
                    df_metal,
                    target_name,
                    y_metal,
                    exclude_imputed_tafel=self.exclude_imputed_tafel,
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
                transformer = _ResidualConditionTransformer()
                X_res = transformer.fit_transform(df_valid)

                min_amp = _DELTA_MIN_AMPLITUDE[target_name]
                v = max(
                    float(np.average(residuals**2, weights=w_valid)),
                    min_amp**2,
                )
                kernel = (
                    ConstantKernel(
                        constant_value=v,
                        constant_value_bounds=(max(1e-6, 0.1 * v), max(1e2, 25.0 * v)),
                    )
                    * Matern(
                        length_scale=1.0,
                        length_scale_bounds=(0.2, 2.0),
                        nu=2.5,
                    )
                    + WhiteKernel(
                        noise_level=0.05 * v,
                        noise_level_bounds=(max(1e-8, 1e-5 * v), max(1e-3, 0.5 * v)),
                    )
                )
                alpha = 1e-6 * v * ((1.0 / w_valid) ** 2 - 1.0) + 1e-10
                gpr = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=alpha,
                    normalize_y=False,
                    random_state=self.random_state + metal_idx * 41 + t_idx * 13,
                    n_restarts_optimizer=2,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    gpr.fit(X_res, residuals)

                self._residual_models[key] = _FittedResidualTarget(
                    method="gp_residual",
                    n_lab=n_valid,
                    transformer=transformer,
                    gp_model=gpr,
                )

    def _fit_lab_only_gp(self, lab_db: HarmonizedDatabase | None) -> None:
        self._lab_only_targets = {}
        self._lab_only_extractor = None
        self._lab_only_ad = None
        self._lab_only_empty_fallback = False

        project_label = self.lab_project or "selected lab project"
        if lab_db is None or lab_db.ml_training_dataframe.empty:
            self._lab_only_n_rows = 0
            msg = (
                f"Lab-Only Local GP requires at least {MIN_LAB_ONLY_GP_SAMPLES} laboratory records "
                f"in project {project_label!r}, but found 0 records."
            )
            if not self.allow_fallback or self.base_predictor is None or not self.base_predictor._is_fitted:
                raise InsufficientLabSamplesError(
                    f"{msg} Add >= {MIN_LAB_ONLY_GP_SAMPLES} records to data/lab/ or switch to 'delta_learning'."
                )
            self._lab_only_empty_fallback = True
            return

        df = lab_db.ml_training_dataframe.reset_index(drop=True)
        self._lab_only_n_rows = len(df)
        self._lab_only_extractor = ElectrochemicalFeatureExtractor(
            material_properties=lab_db.material_properties
        )
        X_all = self._lab_only_extractor.fit_transform(df)
        self._lab_only_ad = ApplicabilityDomainChecker(
            k=min(3, max(1, len(df))),
            min_high_confidence_samples=5,
            min_low_confidence_samples=MIN_LAB_ONLY_GP_SAMPLES,
        ).fit(df)

        targets = _extract_target_vectors(df)
        weights_all = _extract_sample_weights(df, use_quality_weights=self.use_quality_weights)

        for t_idx, target_name in enumerate(ML_TARGET_NAMES):
            y_all = targets[target_name]
            valid_mask = _lab_target_valid_mask(
                df,
                target_name,
                y_all,
                exclude_imputed_tafel=self.exclude_imputed_tafel,
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
                    fallback_mean=_clip_target_mean(target_name, fb_mean),
                    fallback_std=float(fb_std),
                    warning=msg,
                )
                continue

            X_target = X_all[valid_mask]
            y_target = y_all[valid_mask]
            w_target = (
                np.clip(weights_all[valid_mask], 0.05, 1.0)
                if weights_all is not None
                else np.ones(n_valid, dtype=float)
            )

            scaler = StandardScaler()
            scaler.fit(X_target)
            scaler.scale_ = np.where(
                scaler.scale_ > 1e-9,
                np.maximum(scaler.scale_, 1e-3),
                1.0,
            )
            X_scaled = scaler.transform(X_target)

            y_var = max(float(np.var(y_target)), _DELTA_MIN_AMPLITUDE[target_name] ** 2)
            kernel = (
                ConstantKernel(
                    constant_value=y_var,
                    constant_value_bounds=(1e-4, max(1e3, 50.0 * y_var)),
                )
                * Matern(
                    length_scale=1.0,
                    length_scale_bounds=(0.1, 20.0),
                    nu=2.5,
                )
                + WhiteKernel(
                    noise_level=0.05 * y_var,
                    noise_level_bounds=(1e-6, max(10.0, 2.0 * y_var)),
                )
            )
            alpha = 1e-5 * y_var * ((1.0 / w_target) ** 2 - 1.0) + 1e-9
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                normalize_y=True,
                random_state=self.random_state + t_idx * 17,
                n_restarts_optimizer=2,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gpr.fit(X_scaled, y_target)

            self._lab_only_targets[target_name] = _FittedLabOnlyTarget(
                n_lab=n_valid,
                is_fallback=False,
                gp_model=gpr,
                scaler=scaler,
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
                    mean=_clip_target_mean(target_name, prior_est.mean),
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
            else:
                assert fitted.transformer is not None and fitted.gp_model is not None
                X_q = fitted.transformer.transform_query(q)
                d_mean, d_std = fitted.gp_model.predict(X_q, return_std=True)
                delta_est = ParameterEstimate(
                    mean=float(d_mean[0]),
                    std=max(float(d_std[0]), 0.0),
                )

            cal_mean = _clip_target_mean(target_name, prior_est.mean + delta_est.mean)
            cal_std = math.hypot(max(float(prior_est.std), 0.0), delta_est.std)
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
                model_family="gaussian_process",
                applicability=ad_res,
                calibration_mode="lab_only_gp",
                lab_project=self.lab_project,
                residual_breakdown=None,
            )

        assert self._lab_only_extractor is not None and self._lab_only_ad is not None
        X_q = self._lab_only_extractor.transform_query(q)
        ad_res = self._lab_only_ad.evaluate(q)

        estimates: dict[str, ParameterEstimate] = {}
        extra_warnings: list[str] = []
        extra_recs: list[str] = []
        has_fallback = False

        for target_name in ML_TARGET_NAMES:
            fitted = self._lab_only_targets[target_name]
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
                assert fitted.scaler is not None and fitted.gp_model is not None
                X_scaled = fitted.scaler.transform(X_q)
                mean_arr, std_arr = fitted.gp_model.predict(X_scaled, return_std=True)
                estimates[target_name] = ParameterEstimate(
                    mean=_clip_target_mean(target_name, float(mean_arr[0])),
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
            model_family="gaussian_process",
            applicability=ad_res,
            calibration_mode="lab_only_gp",
            lab_project=self.lab_project,
            residual_breakdown=None,
        )
