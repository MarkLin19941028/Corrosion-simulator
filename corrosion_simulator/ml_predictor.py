"""Hybrid Physics-ML parameter predictor, log-space Monte Carlo uncertainty propagation, and Applicability Domain."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from corrosion_simulator.database import ExposureCondition, HarmonizedDatabase
from corrosion_simulator.physics import (
    PolarizationParameters,
    calculate_faraday_corrosion_rate,
)
from corrosion_simulator.seed_data import SEED_MATERIAL_PROPERTIES

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

ConfidenceTier = Literal["High", "Medium", "Low"]
ModelFamily = Literal["random_forest", "gradient_boosting", "gaussian_process"]
CVStrategy = Literal["stratified_5fold", "loocv"]

ML_TARGET_NAMES: Final[tuple[str, ...]] = (
    "Ecorr_V_SHE",
    "log10_icorr_uA_cm2",
    "beta_a_mV_dec",
    "beta_c_mV_dec",
)

TAFEL_SLOPE_MIN_MV_DEC: Final[float] = 20.0
TAFEL_SLOPE_MAX_MV_DEC: Final[float] = 500.0
ECORR_THERMO_MIN_V_SHE: Final[float] = -2.5
ECORR_THERMO_MAX_V_SHE: Final[float] = 2.5

_MATERIAL_DESCRIPTOR_COLS: Final[tuple[str, ...]] = (
    "standard_potential_V_SHE",
    "density_g_cm3",
    "molar_mass_g_mol",
    "valence",
    "electronegativity_pauling",
    "work_function_eV",
    "passivation_tendency",
    "pourbaix_stable_pH_min",
    "pourbaix_stable_pH_max",
)

_CATEGORICAL_COLS: Final[tuple[str, ...]] = (
    "metal",
    "oxidizer_type",
    "acid_type",
    "inhibitor_type",
)

_APPLICABILITY_NUMERIC_COLS: Final[tuple[str, ...]] = (
    "pH",
    "temperature_C",
    "oxidizer_concentration",
    "acid_concentration",
    "inhibitor_concentration",
)

_APPLICABILITY_CATEGORICAL_COLS: Final[tuple[str, ...]] = (
    "oxidizer_type",
    "acid_type",
    "inhibitor_type",
)

_SEED_MATERIAL_MAP: Final[dict[str, dict[str, Any]]] = {
    row["metal"]: row for row in SEED_MATERIAL_PROPERTIES
}


def _clean_category(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return "None"
    text = str(value).strip()
    if not text or text.lower() in ("none", "nan", "null", "n/a", "na"):
        return "None"
    return text


def _safe_float(value: Any, default: float) -> float:
    if value is None or pd.isna(value):
        return default
    try:
        val = float(value)
        return val if math.isfinite(val) else default
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class MLFeatureQuery:
    """Complete Exposure Condition descriptor input for ML parameter prediction and Applicability Domain checks."""

    metal: str
    pH: float = 7.0
    temperature_C: float = 25.0
    oxidizer_type: str = "None"
    oxidizer_concentration: float = 0.0
    acid_type: str = "None"
    acid_concentration: float = 0.0
    inhibitor_type: str = "None"
    inhibitor_concentration: float = 0.0
    ORP_mV: float | None = None

    @classmethod
    def from_exposure_condition(cls, condition: ExposureCondition | MLFeatureQuery) -> MLFeatureQuery:
        if isinstance(condition, MLFeatureQuery):
            return condition
        if isinstance(condition, ExposureCondition):
            return cls(
                metal=condition.metal,
                pH=float(condition.pH),
                temperature_C=float(condition.temperature_C),
                oxidizer_type=_clean_category(condition.oxidizer_type),
                oxidizer_concentration=float(condition.oxidizer_concentration),
            )
        raise TypeError(
            f"Expected ExposureCondition or MLFeatureQuery, got {type(condition).__name__}"
        )

    def to_record_dict(self) -> dict[str, Any]:
        return {
            "metal": str(self.metal).strip(),
            "pH": float(self.pH),
            "temperature_C": float(self.temperature_C),
            "oxidizer_type": _clean_category(self.oxidizer_type),
            "oxidizer_concentration": float(self.oxidizer_concentration),
            "acid_type": _clean_category(self.acid_type),
            "acid_concentration": float(self.acid_concentration),
            "inhibitor_type": _clean_category(self.inhibitor_type),
            "inhibitor_concentration": float(self.inhibitor_concentration),
            "ORP_mV": np.nan if self.ORP_mV is None else float(self.ORP_mV),
        }


class ElectrochemicalFeatureExtractor:
    """Extracts material, solution, and physicochemical interaction features without data leakage."""

    def __init__(self, material_properties: pd.DataFrame | None = None) -> None:
        self._material_lookup: dict[str, dict[str, float]] = {}
        for metal, row in _SEED_MATERIAL_MAP.items():
            self._material_lookup[metal] = {
                col: float(row[col]) for col in _MATERIAL_DESCRIPTOR_COLS if col in row
            }
        if material_properties is not None and not material_properties.empty:
            col_key = "metal" if "metal" in material_properties.columns else "symbol"
            if col_key in material_properties.columns:
                for _, row in material_properties.iterrows():
                    metal = str(row[col_key]).strip()
                    if not metal:
                        continue
                    entry = self._material_lookup.setdefault(metal, {})
                    for col in _MATERIAL_DESCRIPTOR_COLS:
                        if col in row and not pd.isna(row[col]):
                            entry[col] = float(row[col])

        self._category_vocab: dict[str, list[str]] = {}
        self._numeric_medians: dict[str, float] = {}
        self._feature_names: list[str] = []
        self._is_fitted: bool = False

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def _enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        work["metal"] = work["metal"].astype(str).str.strip()

        for col in ("oxidizer_type", "acid_type", "inhibitor_type"):
            if col in work.columns:
                work[col] = work[col].apply(_clean_category)
            else:
                work[col] = "None"

        for col, default in (
            ("pH", 7.0),
            ("temperature_C", 25.0),
            ("oxidizer_concentration", 0.0),
            ("acid_concentration", 0.0),
            ("inhibitor_concentration", 0.0),
        ):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce").fillna(default)
            else:
                work[col] = default

        if "ORP_mV" in work.columns:
            work["ORP_mV"] = pd.to_numeric(work["ORP_mV"], errors="coerce")
        else:
            work["ORP_mV"] = np.nan

        # Populate material descriptors from lookup whenever missing
        for col in _MATERIAL_DESCRIPTOR_COLS:
            mapped = work["metal"].map(
                lambda m, c=col: self._material_lookup.get(m, {}).get(c, np.nan)
            )
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce").fillna(mapped)
            else:
                work[col] = mapped

        return work

    def fit(self, df: pd.DataFrame) -> ElectrochemicalFeatureExtractor:
        work = self._enrich_dataframe(df)

        # Learn training-only medians for numeric columns (notably ORP_mV)
        self._numeric_medians = {}
        for col in (
            *_MATERIAL_DESCRIPTOR_COLS,
            "pH",
            "temperature_C",
            "oxidizer_concentration",
            "acid_concentration",
            "inhibitor_concentration",
            "ORP_mV",
        ):
            med = float(work[col].median()) if work[col].notna().any() else 0.0
            self._numeric_medians[col] = med if math.isfinite(med) else 0.0

        # Learn deterministic sorted category vocabularies from training fold only
        self._category_vocab = {}
        for col in _CATEGORICAL_COLS:
            cats = sorted({str(v) for v in work[col].unique()})
            self._category_vocab[col] = cats

        # Build deterministic feature name ordering
        numeric_names = [
            *_MATERIAL_DESCRIPTOR_COLS,
            "pH",
            "temperature_C",
            "oxidizer_concentration",
            "acid_concentration",
            "inhibitor_concentration",
            "ORP_mV",
            "pH_x_temperature_C",
            "E0_minus_ORP_V",
            "log1p_oxidizer_conc",
            "log1p_acid_conc",
            "log1p_inhibitor_conc",
            "pourbaix_passivation_indicator",
            "pourbaix_pH_distance",
        ]
        cat_names = [
            f"{col}__{cat}"
            for col in _CATEGORICAL_COLS
            for cat in self._category_vocab[col]
        ]
        self._feature_names = numeric_names + cat_names
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("ElectrochemicalFeatureExtractor must be fitted before transform().")

        work = self._enrich_dataframe(df)
        for col, med in self._numeric_medians.items():
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(med)

        ph = work["pH"].to_numpy(dtype=float)
        temp = work["temperature_C"].to_numpy(dtype=float)
        ox_c = np.maximum(work["oxidizer_concentration"].to_numpy(dtype=float), 0.0)
        acid_c = np.maximum(work["acid_concentration"].to_numpy(dtype=float), 0.0)
        inh_c = np.maximum(work["inhibitor_concentration"].to_numpy(dtype=float), 0.0)
        orp_mv = work["ORP_mV"].to_numpy(dtype=float)
        e0 = work["standard_potential_V_SHE"].to_numpy(dtype=float)
        pb_min = work["pourbaix_stable_pH_min"].to_numpy(dtype=float)
        pb_max = work["pourbaix_stable_pH_max"].to_numpy(dtype=float)

        # Engineered physicochemical interaction features
        ph_x_t = ph * temp
        e0_minus_orp = e0 - (orp_mv / 1000.0)
        log1p_ox = np.log10(1.0 + ox_c)
        log1p_acid = np.log10(1.0 + acid_c)
        log1p_inh = np.log10(1.0 + inh_c)
        pb_indicator = ((ph >= pb_min) & (ph <= pb_max)).astype(float)
        pb_dist = np.where(
            ph < pb_min,
            pb_min - ph,
            np.where(ph > pb_max, ph - pb_max, 0.0),
        )

        numeric_cols: list[np.ndarray] = [
            work[col].to_numpy(dtype=float) for col in _MATERIAL_DESCRIPTOR_COLS
        ] + [
            ph,
            temp,
            ox_c,
            acid_c,
            inh_c,
            orp_mv,
            ph_x_t,
            e0_minus_orp,
            log1p_ox,
            log1p_acid,
            log1p_inh,
            pb_indicator,
            pb_dist,
        ]

        cat_cols: list[np.ndarray] = []
        for col in _CATEGORICAL_COLS:
            values = work[col].astype(str).to_numpy()
            for cat in self._category_vocab[col]:
                cat_cols.append((values == cat).astype(float))

        matrix = np.column_stack(numeric_cols + cat_cols)
        return np.asarray(matrix, dtype=float)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def transform_query(self, query: ExposureCondition | MLFeatureQuery) -> np.ndarray:
        q = MLFeatureQuery.from_exposure_condition(query)
        return self.transform(pd.DataFrame([q.to_record_dict()]))


@dataclass(frozen=True)
class ApplicabilityDomainResult:
    """Two-tier Applicability Domain assessment (per-metal Bounding Box + local k-NN density)."""

    metal: str
    confidence_tier: ConfidenceTier
    in_bounding_box: bool
    out_of_box_features: tuple[str, ...]
    unseen_categories: tuple[str, ...]
    knn_mean_distance: float
    knn_k: int
    n_metal_training_samples: int
    warnings: tuple[str, ...]
    experimental_recommendations: tuple[str, ...]


class ApplicabilityDomainChecker:
    """Evaluates per-metal Bounding Box coverage and local k-NN density to assign Confidence Tiers."""

    def __init__(
        self,
        *,
        k: int = 3,
        medium_distance_threshold: float = 0.35,
        high_distance_threshold: float = 0.75,
        min_high_confidence_samples: int = 5,
    ) -> None:
        if k < 1:
            raise ValueError("k must be at least 1.")
        self.k = int(k)
        self.medium_distance_threshold = float(medium_distance_threshold)
        self.high_distance_threshold = float(high_distance_threshold)
        self.min_high_confidence_samples = int(min_high_confidence_samples)

        self._metal_stats: dict[str, dict[str, Any]] = {}
        self._global_spans: dict[str, float] = {}
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> ApplicabilityDomainChecker:
        work = df.copy()
        work["metal"] = work["metal"].astype(str).str.strip()
        for col in _APPLICABILITY_NUMERIC_COLS:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce").fillna(
                    25.0 if col == "temperature_C" else (7.0 if col == "pH" else 0.0)
                )
            else:
                work[col] = 25.0 if col == "temperature_C" else (7.0 if col == "pH" else 0.0)
        for col in _APPLICABILITY_CATEGORICAL_COLS:
            if col in work.columns:
                work[col] = work[col].apply(_clean_category)
            else:
                work[col] = "None"

        self._global_spans = {}
        for col in _APPLICABILITY_NUMERIC_COLS:
            vals = work[col].to_numpy(dtype=float)
            span = float(np.max(vals) - np.min(vals)) if len(vals) > 0 else 1.0
            self._global_spans[col] = span if span > 1e-12 else 1.0

        self._metal_stats = {}
        for metal, grp in work.groupby("metal"):
            bounds: dict[str, tuple[float, float]] = {}
            scales: dict[str, float] = {}
            for col in _APPLICABILITY_NUMERIC_COLS:
                col_vals = grp[col].to_numpy(dtype=float)
                c_min = float(np.min(col_vals))
                c_max = float(np.max(col_vals))
                bounds[col] = (c_min, c_max)
                span = c_max - c_min
                scales[col] = span if span > 1e-12 else self._global_spans[col]

            cats = {
                col: sorted({str(v) for v in grp[col].unique()})
                for col in _APPLICABILITY_CATEGORICAL_COLS
            }
            points = np.column_stack(
                [grp[col].to_numpy(dtype=float) for col in _APPLICABILITY_NUMERIC_COLS]
            )
            self._metal_stats[str(metal)] = {
                "n_samples": len(grp),
                "bounds": bounds,
                "scales": scales,
                "categories": cats,
                "points": points,
            }

        self._is_fitted = True
        return self

    def evaluate(self, query: ExposureCondition | MLFeatureQuery) -> ApplicabilityDomainResult:
        if not self._is_fitted:
            raise RuntimeError("ApplicabilityDomainChecker must be fitted before evaluate().")

        q = MLFeatureQuery.from_exposure_condition(query)
        metal = q.metal.strip()
        stats = self._metal_stats.get(metal)
        if stats is None:
            return ApplicabilityDomainResult(
                metal=metal,
                confidence_tier="Low",
                in_bounding_box=False,
                out_of_box_features=_APPLICABILITY_NUMERIC_COLS,
                unseen_categories=(f"metal={metal}",),
                knn_mean_distance=float("inf"),
                knn_k=0,
                n_metal_training_samples=0,
                warnings=(f"No experimental training records exist for Metal {metal!r}.",),
                experimental_recommendations=(
                    f"Measure baseline Tafel Polarization Curves for {metal} in "
                    f"{q.oxidizer_type} (pH {q.pH:.1f}, {q.temperature_C:.1f} °C).",
                ),
            )

        rec = q.to_record_dict()
        bounds: dict[str, tuple[float, float]] = stats["bounds"]
        scales: dict[str, float] = stats["scales"]
        cats: dict[str, list[str]] = stats["categories"]
        points: np.ndarray = stats["points"]
        n_samples: int = stats["n_samples"]

        out_of_box: list[str] = []
        warnings: list[str] = []
        recommendations: list[str] = []

        for col in _APPLICABILITY_NUMERIC_COLS:
            val = float(rec[col])
            lo, hi = bounds[col]
            if val < lo - 1e-9 or val > hi + 1e-9:
                out_of_box.append(col)
                warnings.append(
                    f"{col}={val:.3g} lies outside the {metal} training range [{lo:.3g}, {hi:.3g}]."
                )
                recommendations.append(
                    f"Run targeted polarization experiments for {metal} near {col}={val:.3g} "
                    f"(current coverage: [{lo:.3g}, {hi:.3g}])."
                )

        unseen_cats: list[str] = []
        for col in _APPLICABILITY_CATEGORICAL_COLS:
            val = str(rec[col])
            if val != "None" and val not in cats[col]:
                unseen_cats.append(f"{col}={val}")
                warnings.append(
                    f"{col}={val!r} was never observed in the {metal} training set "
                    f"(observed: {', '.join(cats[col])})."
                )
                recommendations.append(
                    f"Collect experimental Tafel data for {metal} with {col}={val!r} "
                    f"at pH {q.pH:.1f} and {q.temperature_C:.1f} °C."
                )

        q_vec = np.array([float(rec[col]) for col in _APPLICABILITY_NUMERIC_COLS], dtype=float)
        scale_vec = np.array([scales[col] for col in _APPLICABILITY_NUMERIC_COLS], dtype=float)
        diffs = (points - q_vec) / scale_vec
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        eff_k = min(self.k, len(dists))
        nearest_dists = np.sort(dists)[:eff_k]
        knn_mean_dist = float(np.mean(nearest_dists))

        in_box = len(out_of_box) == 0 and len(unseen_cats) == 0
        if not in_box or knn_mean_dist > self.high_distance_threshold:
            tier: ConfidenceTier = "Low"
            if in_box:
                warnings.append(
                    f"Sparse local {metal} training density (k={eff_k} mean distance={knn_mean_dist:.3f})."
                )
                recommendations.append(
                    f"Add experimental polarization measurements for {metal} near pH {q.pH:.1f}, "
                    f"{q.oxidizer_type} ({q.oxidizer_concentration:.3g}), {q.temperature_C:.1f} °C."
                )
        elif (
            knn_mean_dist > self.medium_distance_threshold
            or n_samples < self.min_high_confidence_samples
        ):
            tier = "Medium"
            warnings.append(
                f"Moderate {metal} training coverage (n={n_samples}, k={eff_k} mean distance={knn_mean_dist:.3f})."
            )
            recommendations.append(
                f"Validate {metal} prediction with a confirmatory Tafel sweep at "
                f"pH {q.pH:.1f}, {q.oxidizer_type} ({q.oxidizer_concentration:.3g}), {q.temperature_C:.1f} °C."
            )
        else:
            tier = "High"

        return ApplicabilityDomainResult(
            metal=metal,
            confidence_tier=tier,
            in_bounding_box=in_box,
            out_of_box_features=tuple(out_of_box),
            unseen_categories=tuple(unseen_cats),
            knn_mean_distance=knn_mean_dist,
            knn_k=eff_k,
            n_metal_training_samples=n_samples,
            warnings=tuple(warnings),
            experimental_recommendations=tuple(recommendations),
        )


@dataclass(frozen=True)
class ParameterEstimate:
    """Predicted mean and ±1σ uncertainty for a single electrochemical parameter."""

    mean: float
    std: float


@dataclass(frozen=True)
class MLParameterPrediction:
    """Predicted four-parameter electrochemical kinetics with ±1σ uncertainties and Applicability Domain."""

    metal: str
    Ecorr_V_SHE: ParameterEstimate
    log10_icorr_uA_cm2: ParameterEstimate
    icorr_uA_cm2_median: float
    beta_a_mV_dec: ParameterEstimate
    beta_c_mV_dec: ParameterEstimate
    model_family: ModelFamily
    applicability: ApplicabilityDomainResult

    def to_polarization_parameters(self) -> PolarizationParameters:
        """Convert point estimates into physically clipped PolarizationParameters."""
        ecorr = float(np.clip(self.Ecorr_V_SHE.mean, ECORR_THERMO_MIN_V_SHE, ECORR_THERMO_MAX_V_SHE))
        icorr = float(10.0 ** float(np.clip(self.log10_icorr_uA_cm2.mean, -6.0, 6.0)))
        beta_a = float(np.clip(self.beta_a_mV_dec.mean, TAFEL_SLOPE_MIN_MV_DEC, TAFEL_SLOPE_MAX_MV_DEC))
        beta_c = float(np.clip(self.beta_c_mV_dec.mean, TAFEL_SLOPE_MIN_MV_DEC, TAFEL_SLOPE_MAX_MV_DEC))
        return PolarizationParameters(
            Ecorr_V_SHE=ecorr,
            icorr_uA_cm2=icorr,
            beta_a_mV_dec=beta_a,
            beta_c_mV_dec=beta_c,
            metal=self.metal,
        )


@dataclass(frozen=True)
class CrossValidationMetrics:
    """Cross-validated regression metrics for one electrochemical target."""

    target: str
    model_family: ModelFamily
    cv_strategy: CVStrategy
    mae: float
    rmse: float
    r2: float
    mape: float
    n_samples: int


class _UncertainTargetModel:
    """Wraps a single-target regressor providing (mean, std) predictions."""

    def __init__(
        self,
        model_family: ModelFamily,
        *,
        random_state: int = 42,
        n_estimators: int = 100,
        n_bootstrap: int = 10,
    ) -> None:
        self.model_family = model_family
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.n_bootstrap = int(n_bootstrap)
        self._scaler: StandardScaler | None = None
        self._model: Any = None
        self._ensemble: list[Any] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> _UncertainTargetModel:
        if self.model_family == "random_forest":
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                min_samples_leaf=1,
            )
            rf.fit(X, y)
            self._model = rf
        elif self.model_family == "gradient_boosting":
            rng = np.random.default_rng(self.random_state)
            n_samples = len(y)
            self._ensemble = []
            for b in range(self.n_bootstrap):
                idx = rng.integers(0, n_samples, size=n_samples)
                gbr = GradientBoostingRegressor(
                    n_estimators=max(20, self.n_estimators // 2),
                    max_depth=3,
                    learning_rate=0.08,
                    random_state=self.random_state + b,
                )
                gbr.fit(X[idx], y[idx])
                self._ensemble.append(gbr)
        elif self.model_family == "gaussian_process":
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5)
                + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-5, 1e1))
            )
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                random_state=self.random_state,
                n_restarts_optimizer=1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gpr.fit(X_scaled, y)
            self._model = gpr
        else:
            raise ValueError(f"Unsupported model_family: {self.model_family!r}")
        return self

    def predict_mean_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model_family == "random_forest":
            tree_preds = np.vstack([tree.predict(X) for tree in self._model.estimators_])
            return np.mean(tree_preds, axis=0), np.std(tree_preds, axis=0)
        if self.model_family == "gradient_boosting":
            boot_preds = np.vstack([m.predict(X) for m in self._ensemble])
            return np.mean(boot_preds, axis=0), np.std(boot_preds, axis=0)
        if self.model_family == "gaussian_process":
            assert self._scaler is not None
            X_scaled = self._scaler.transform(X)
            mean, std = self._model.predict(X_scaled, return_std=True)
            return np.asarray(mean, dtype=float), np.maximum(np.asarray(std, dtype=float), 0.0)
        raise ValueError(f"Unsupported model_family: {self.model_family!r}")


def _extract_target_vectors(df: pd.DataFrame) -> dict[str, np.ndarray]:
    ecorr = pd.to_numeric(df["Ecorr_V_SHE"], errors="coerce").to_numpy(dtype=float)
    icorr = pd.to_numeric(df["icorr_uA_cm2"], errors="coerce").to_numpy(dtype=float)
    log_icorr = np.log10(np.maximum(icorr, 1e-12))
    beta_a = pd.to_numeric(df["beta_a_mV_dec"], errors="coerce").to_numpy(dtype=float)
    beta_c = pd.to_numeric(df["beta_c_mV_dec"], errors="coerce").to_numpy(dtype=float)
    return {
        "Ecorr_V_SHE": ecorr,
        "log10_icorr_uA_cm2": log_icorr,
        "beta_a_mV_dec": beta_a,
        "beta_c_mV_dec": beta_c,
    }


def _compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


class HybridMLPredictor:
    """Trains and serves four electrochemical target regressors with uncertainty and Applicability Domain."""

    def __init__(
        self,
        *,
        model_family: ModelFamily = "random_forest",
        random_state: int = 42,
        n_estimators: int = 100,
        n_bootstrap: int = 10,
        knn_k: int = 3,
    ) -> None:
        if model_family not in ("random_forest", "gradient_boosting", "gaussian_process"):
            raise ValueError(f"Unsupported model_family: {model_family!r}")
        self.model_family: ModelFamily = model_family
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.n_bootstrap = int(n_bootstrap)
        self.knn_k = int(knn_k)

        self.feature_extractor: ElectrochemicalFeatureExtractor | None = None
        self.applicability_checker: ApplicabilityDomainChecker | None = None
        self.models: dict[str, _UncertainTargetModel] = {}
        self._material_properties: pd.DataFrame | None = None
        self._is_fitted: bool = False

    def fit(self, database: HarmonizedDatabase) -> HybridMLPredictor:
        df = database.electrochemical_database.reset_index(drop=True)
        if df.empty:
            raise ValueError("Cannot train HybridMLPredictor on an empty database.")

        self._material_properties = database.material_properties.copy()
        self.feature_extractor = ElectrochemicalFeatureExtractor(
            material_properties=self._material_properties
        )
        X = self.feature_extractor.fit_transform(df)
        targets = _extract_target_vectors(df)

        self.models = {}
        for idx, target_name in enumerate(ML_TARGET_NAMES):
            model = _UncertainTargetModel(
                self.model_family,
                random_state=self.random_state + idx * 17,
                n_estimators=self.n_estimators,
                n_bootstrap=self.n_bootstrap,
            )
            model.fit(X, targets[target_name])
            self.models[target_name] = model

        self.applicability_checker = ApplicabilityDomainChecker(k=self.knn_k).fit(df)
        self._is_fitted = True
        return self

    def predict(self, query: ExposureCondition | MLFeatureQuery) -> MLParameterPrediction:
        if not self._is_fitted or self.feature_extractor is None or self.applicability_checker is None:
            raise RuntimeError("HybridMLPredictor must be fitted before predict().")

        q = MLFeatureQuery.from_exposure_condition(query)
        X_q = self.feature_extractor.transform_query(q)
        ad_res = self.applicability_checker.evaluate(q)

        estimates: dict[str, ParameterEstimate] = {}
        for target_name in ML_TARGET_NAMES:
            mean_arr, std_arr = self.models[target_name].predict_mean_std(X_q)
            estimates[target_name] = ParameterEstimate(
                mean=float(mean_arr[0]),
                std=float(std_arr[0]),
            )

        log_i_mean = float(np.clip(estimates["log10_icorr_uA_cm2"].mean, -6.0, 6.0))
        return MLParameterPrediction(
            metal=q.metal.strip(),
            Ecorr_V_SHE=estimates["Ecorr_V_SHE"],
            log10_icorr_uA_cm2=estimates["log10_icorr_uA_cm2"],
            icorr_uA_cm2_median=float(10.0**log_i_mean),
            beta_a_mV_dec=estimates["beta_a_mV_dec"],
            beta_c_mV_dec=estimates["beta_c_mV_dec"],
            model_family=self.model_family,
            applicability=ad_res,
        )

    def cross_validate(
        self,
        database: HarmonizedDatabase,
        *,
        cv_strategy: CVStrategy = "stratified_5fold",
    ) -> list[CrossValidationMetrics]:
        """Run leak-free cross-validation (fitting feature extractor inside every fold)."""
        df = database.electrochemical_database.reset_index(drop=True)
        n_samples = len(df)
        if n_samples < 3:
            raise ValueError("At least 3 samples are required for cross-validation.")

        targets = _extract_target_vectors(df)
        oof_preds: dict[str, np.ndarray] = {
            t: np.zeros(n_samples, dtype=float) for t in ML_TARGET_NAMES
        }

        if cv_strategy == "loocv":
            splits = list(LeaveOneOut().split(df))
        elif cv_strategy == "stratified_5fold":
            n_splits = min(5, n_samples)
            metal_counts = df["metal"].value_counts()
            strat_labels = df["metal"].where(
                df["metal"].map(metal_counts) >= n_splits, "_rare_metal"
            )
            if strat_labels.value_counts().min() >= n_splits:
                splitter = StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=self.random_state
                )
                splits = list(splitter.split(df, strat_labels))
            else:
                splitter_kf = KFold(
                    n_splits=n_splits, shuffle=True, random_state=self.random_state
                )
                splits = list(splitter_kf.split(df))
        else:
            raise ValueError(f"Unsupported cv_strategy: {cv_strategy!r}")

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)

            # Strict featurization ordering: fit extractor on train fold only
            extractor = ElectrochemicalFeatureExtractor(
                material_properties=database.material_properties
            )
            X_train = extractor.fit_transform(train_df)
            X_val = extractor.transform(val_df)

            for t_idx, target_name in enumerate(ML_TARGET_NAMES):
                model = _UncertainTargetModel(
                    self.model_family,
                    random_state=self.random_state + fold_idx * 31 + t_idx * 7,
                    n_estimators=self.n_estimators,
                    n_bootstrap=self.n_bootstrap,
                )
                model.fit(X_train, targets[target_name][train_idx])
                mean_pred, _ = model.predict_mean_std(X_val)
                oof_preds[target_name][val_idx] = mean_pred

        results: list[CrossValidationMetrics] = []
        for target_name in ML_TARGET_NAMES:
            y_true = targets[target_name]
            y_pred = oof_preds[target_name]
            results.append(
                CrossValidationMetrics(
                    target=target_name,
                    model_family=self.model_family,
                    cv_strategy=cv_strategy,
                    mae=float(mean_absolute_error(y_true, y_pred)),
                    rmse=float(root_mean_squared_error(y_true, y_pred)),
                    r2=float(r2_score(y_true, y_pred)),
                    mape=_compute_mape(y_true, y_pred),
                    n_samples=n_samples,
                )
            )
        return results

    def feature_importances(self, target: str = "log10_icorr_uA_cm2") -> pd.DataFrame:
        """Return feature importance table for tree-based model families."""
        if not self._is_fitted or self.feature_extractor is None:
            raise RuntimeError("HybridMLPredictor must be fitted first.")
        if target not in self.models:
            raise ValueError(f"Unknown target: {target!r}")
        wrapper = self.models[target]
        if self.model_family == "random_forest":
            importances = np.asarray(wrapper._model.feature_importances_, dtype=float)
        elif self.model_family == "gradient_boosting":
            importances = np.mean(
                [m.feature_importances_ for m in wrapper._ensemble], axis=0
            )
        else:
            raise ValueError(
                f"Feature importances are only available for tree models, not {self.model_family!r}."
            )
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_extractor.feature_names,
                    "importance": importances,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: str | Path) -> Path:
        """Persist the fitted predictor bundle via joblib."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted HybridMLPredictor.")
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> HybridMLPredictor:
        """Load a persisted HybridMLPredictor bundle."""
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"Expected HybridMLPredictor in {path}, got {type(obj).__name__}")
        return obj


@dataclass(frozen=True)
class MonteCarloPolarizationBand:
    """Percentile uncertainty bands for activation-controlled Polarization Curves."""

    potential_V_SHE: np.ndarray
    log10_abs_net_mean: np.ndarray
    log10_abs_net_p05: np.ndarray
    log10_abs_net_p50: np.ndarray
    log10_abs_net_p95: np.ndarray
    anodic_p05_uA_cm2: np.ndarray
    anodic_p50_uA_cm2: np.ndarray
    anodic_p95_uA_cm2: np.ndarray
    cathodic_p05_uA_cm2: np.ndarray
    cathodic_p50_uA_cm2: np.ndarray
    cathodic_p95_uA_cm2: np.ndarray


@dataclass(frozen=True)
class MonteCarloPropagationResult:
    """Physics-clipped Monte Carlo uncertainty propagation results for single-metal kinetics."""

    metal: str
    n_draws: int
    Ecorr_samples_V_SHE: np.ndarray
    icorr_samples_uA_cm2: np.ndarray
    beta_a_samples_mV_dec: np.ndarray
    beta_c_samples_mV_dec: np.ndarray
    cr_nm_min_samples: np.ndarray
    cr_mm_yr_samples: np.ndarray
    cr_nm_min_summary: dict[str, float]
    cr_mm_yr_summary: dict[str, float]
    polarization_band: MonteCarloPolarizationBand


def _summarize_samples(samples: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "p05": float(np.percentile(samples, 5.0)),
        "p50": float(np.percentile(samples, 50.0)),
        "p95": float(np.percentile(samples, 95.0)),
    }


def propagate_uncertainty_monte_carlo(
    prediction: MLParameterPrediction,
    *,
    n_draws: int = 500,
    seed: int | None = 42,
    valence: int | None = None,
    material_properties: pd.DataFrame | None = None,
    overpotential_min_V: float = -0.25,
    overpotential_max_V: float = 0.25,
    num_potential_points: int = 201,
    potentials_V_SHE: np.ndarray | None = None,
) -> MonteCarloPropagationResult:
    """Propagate ML parameter uncertainties via log-space i_corr and physics-clipped Monte Carlo (ADR-0003)."""
    if n_draws < 1:
        raise ValueError(f"n_draws must be positive, got {n_draws!r}")

    rng = np.random.default_rng(seed)

    # Sample E_corr, beta_a, beta_c in linear space with thermodynamic/physical clipping
    ecorr_draws = np.clip(
        rng.normal(prediction.Ecorr_V_SHE.mean, max(prediction.Ecorr_V_SHE.std, 0.0), size=n_draws),
        ECORR_THERMO_MIN_V_SHE,
        ECORR_THERMO_MAX_V_SHE,
    )
    beta_a_draws = np.clip(
        rng.normal(
            prediction.beta_a_mV_dec.mean, max(prediction.beta_a_mV_dec.std, 0.0), size=n_draws
        ),
        TAFEL_SLOPE_MIN_MV_DEC,
        TAFEL_SLOPE_MAX_MV_DEC,
    )
    beta_c_draws = np.clip(
        rng.normal(
            prediction.beta_c_mV_dec.mean, max(prediction.beta_c_mV_dec.std, 0.0), size=n_draws
        ),
        TAFEL_SLOPE_MIN_MV_DEC,
        TAFEL_SLOPE_MAX_MV_DEC,
    )
    # Sample log10(i_corr) in log-space before exponentiating -> strictly positive i_corr
    log_i_draws = np.clip(
        rng.normal(
            prediction.log10_icorr_uA_cm2.mean,
            max(prediction.log10_icorr_uA_cm2.std, 0.0),
            size=n_draws,
        ),
        -6.0,
        6.0,
    )
    icorr_draws = np.power(10.0, log_i_draws)

    # Faraday's law conversion using exact Phase 1 unit-current scaling factor
    unit_faraday = calculate_faraday_corrosion_rate(
        prediction.metal,
        1.0,
        valence=valence,
        material_properties=material_properties,
    )
    cr_nm_min_draws = icorr_draws * unit_faraday.corrosion_rate_nm_min
    cr_mm_yr_draws = icorr_draws * unit_faraday.corrosion_rate_mm_yr

    # Vectorized Wagner-Traud polarization across (n_draws, num_potential_points)
    center_e = float(
        np.clip(prediction.Ecorr_V_SHE.mean, ECORR_THERMO_MIN_V_SHE, ECORR_THERMO_MAX_V_SHE)
    )
    if potentials_V_SHE is not None:
        potentials = np.asarray(potentials_V_SHE, dtype=float)
    else:
        potentials = np.linspace(
            center_e + overpotential_min_V,
            center_e + overpotential_max_V,
            num_potential_points,
        )

    eta = potentials[None, :] - ecorr_draws[:, None]
    ba_v = (beta_a_draws / 1000.0)[:, None]
    bc_v = (beta_c_draws / 1000.0)[:, None]
    i0 = icorr_draws[:, None]

    # Cap exponents to prevent floating-point overflow under wide E_corr variance
    i_anodic = i0 * np.power(10.0, np.clip(eta / ba_v, -30.0, 30.0))
    i_cathodic = i0 * np.power(10.0, np.clip(-eta / bc_v, -30.0, 30.0))
    abs_net = np.maximum(np.abs(i_anodic - i_cathodic), 1e-12)
    log_abs_net = np.log10(abs_net)

    band = MonteCarloPolarizationBand(
        potential_V_SHE=potentials,
        log10_abs_net_mean=np.mean(log_abs_net, axis=0),
        log10_abs_net_p05=np.percentile(log_abs_net, 5.0, axis=0),
        log10_abs_net_p50=np.percentile(log_abs_net, 50.0, axis=0),
        log10_abs_net_p95=np.percentile(log_abs_net, 95.0, axis=0),
        anodic_p05_uA_cm2=np.percentile(i_anodic, 5.0, axis=0),
        anodic_p50_uA_cm2=np.percentile(i_anodic, 50.0, axis=0),
        anodic_p95_uA_cm2=np.percentile(i_anodic, 95.0, axis=0),
        cathodic_p05_uA_cm2=np.percentile(i_cathodic, 5.0, axis=0),
        cathodic_p50_uA_cm2=np.percentile(i_cathodic, 50.0, axis=0),
        cathodic_p95_uA_cm2=np.percentile(i_cathodic, 95.0, axis=0),
    )

    return MonteCarloPropagationResult(
        metal=prediction.metal,
        n_draws=int(n_draws),
        Ecorr_samples_V_SHE=ecorr_draws,
        icorr_samples_uA_cm2=icorr_draws,
        beta_a_samples_mV_dec=beta_a_draws,
        beta_c_samples_mV_dec=beta_c_draws,
        cr_nm_min_samples=cr_nm_min_draws,
        cr_mm_yr_samples=cr_mm_yr_draws,
        cr_nm_min_summary=_summarize_samples(cr_nm_min_draws),
        cr_mm_yr_summary=_summarize_samples(cr_mm_yr_draws),
        polarization_band=band,
    )


def plot_tafel_uncertainty_band(
    mc_result: MonteCarloPropagationResult,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a Tafel Polarization Curve with Monte Carlo 5th-95th percentile (90%) uncertainty band."""
    import matplotlib.pyplot as plt

    band = mc_result.polarization_band
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
    else:
        fig = ax.figure

    ax.fill_betweenx(
        band.potential_V_SHE,
        band.log10_abs_net_p05,
        band.log10_abs_net_p95,
        color="#1f77b4",
        alpha=0.22,
        label=f"90% Uncertainty Band (P05–P95, N={mc_result.n_draws})",
    )
    ax.plot(
        band.log10_abs_net_p50,
        band.potential_V_SHE,
        color="#111111",
        linewidth=2.1,
        label="Median Polarization Curve (P50)",
    )
    ax.plot(
        np.log10(np.maximum(band.anodic_p50_uA_cm2, 1e-12)),
        band.potential_V_SHE,
        linestyle="--",
        color="#d62728",
        linewidth=1.3,
        alpha=0.8,
        label="Median Anodic Branch",
    )
    ax.plot(
        np.log10(np.maximum(band.cathodic_p50_uA_cm2, 1e-12)),
        band.potential_V_SHE,
        linestyle="--",
        color="#1f77b4",
        linewidth=1.3,
        alpha=0.8,
        label="Median Cathodic Branch",
    )

    ax.set_xlabel("log10(|i|) [µA/cm²]")
    ax.set_ylabel("Potential E [V vs. SHE]")
    ax.set_title(
        title or f"{mc_result.metal} Tafel Polarization Curve with Monte Carlo Uncertainty Band"
    )
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="best", fontsize=8.5)
    return fig, ax
