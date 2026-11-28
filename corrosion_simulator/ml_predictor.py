"""Hybrid Physics-ML parameter predictor, log-space Monte Carlo uncertainty propagation, and Applicability Domain."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, LeaveOneOut, StratifiedKFold

from corrosion_simulator.database import (
    ExposureCondition,
    HarmonizedDatabase,
    normalize_category_label as _clean_category,
)
from corrosion_simulator.element_constants import (
    PURE_ELEMENT_PROPERTIES,
    infer_base_element,
)
from corrosion_simulator.seed_data import SEED_MATERIAL_PROPERTIES
from corrosion_simulator.solvent_constants import compute_mixture_solvent_descriptors
from corrosion_simulator.uncertainty import (
    DEFAULT_ECORR_CORRELATION,
    DEFAULT_MASS_TRANSPORT_LIMIT_UA_CM2,
    ECORR_THERMO_MAX_V_SHE,
    ECORR_THERMO_MIN_V_SHE,
    TAFEL_SLOPE_MAX_MV_DEC,
    TAFEL_SLOPE_MIN_MV_DEC,
    MLParameterPrediction,
    MonteCarloPolarizationBand,
    MonteCarloPropagationResult,
    ParameterEstimate,
    clip_target_mean,
    propagate_uncertainty_monte_carlo,
    sample_clipped_icorr,
    sample_lognormal_tafel_slope,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

ConfidenceTier = Literal["High", "Medium", "Low"]
ModelFamily = Literal[
    "random_forest",
    "hist_gradient_boosting",
    "rf_gp_ensemble",
]
PredictedModelFamily = Literal[
    "random_forest",
    "hist_gradient_boosting",
    "rf_gp_ensemble",
    "lab_local_gp",
]
CVStrategy = Literal["stratified_5fold", "group_5fold", "loocv"]

SUPPORTED_MODEL_FAMILIES: Final[tuple[ModelFamily, ...]] = (
    "random_forest",
    "hist_gradient_boosting",
    "rf_gp_ensemble",
)

MODEL_FAMILY_LABELS: Final[dict[ModelFamily, str]] = {
    "random_forest": "Random Forest (Default — Balanced & Fast)",
    "hist_gradient_boosting": "Hist Gradient Boosting (L2 Regularized GBDT)",
    "rf_gp_ensemble": "RF + GP Hybrid Ensemble (Tree + Smooth Kernel)",
}

ML_TARGET_NAMES: Final[tuple[str, ...]] = (
    "Ecorr_V_SHE",
    "log10_icorr_uA_cm2",
    "beta_a_mV_dec",
    "beta_c_mV_dec",
)

GP_MAX_TRAINING_SAMPLES: Final[int] = 2000
MIN_RESIDUAL_GP_SAMPLES: Final[int] = 4
DEFAULT_LOW_SAMPLE_THRESHOLD: Final[int] = 10

RESIDUAL_MIN_AMPLITUDE: Final[dict[str, float]] = {
    "Ecorr_V_SHE": 0.02,
    "log10_icorr_uA_cm2": 0.05,
    "beta_a_mV_dec": 3.0,
    "beta_c_mV_dec": 3.0,
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
    "water_content_wt_pct",
)

_APPLICABILITY_CATEGORICAL_COLS: Final[tuple[str, ...]] = (
    "oxidizer_type",
    "acid_type",
    "inhibitor_type",
    "organic_solvent_type",
)

_SEED_MATERIAL_MAP: Final[dict[str, dict[str, Any]]] = {
    row["metal"]: row for row in SEED_MATERIAL_PROPERTIES
}


def _clean_base_solvent(value: Any) -> str:
    """Normalize a base-solvent string, defaulting blank or `'None'` entries to `'DI water'`."""
    label = _clean_category(value)
    return "DI water" if label == "None" else label


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
    organic_solvent_type: str = "None"
    water_content_wt_pct: float = 100.0
    base_solvent: str = "DI water"

    @classmethod
    def for_solution(
        cls,
        metal: str,
        solution: Any,
        temperature_C: float = 25.0,
    ) -> MLFeatureQuery:
        """Build an MLFeatureQuery from a Metal, Solution value object, and operating temperature."""
        return cls(
            metal=metal,
            pH=float(solution.pH),
            temperature_C=float(temperature_C),
            oxidizer_type=solution.oxidizer_type,
            oxidizer_concentration=float(solution.oxidizer_concentration),
            acid_type=solution.acid_type,
            acid_concentration=float(solution.acid_concentration),
            inhibitor_type=solution.inhibitor_type,
            inhibitor_concentration=float(solution.inhibitor_concentration),
            ORP_mV=solution.ORP_mV,
            organic_solvent_type=solution.organic_solvent_type,
            water_content_wt_pct=float(solution.water_content_wt_pct),
            base_solvent=solution.base_solvent,
        )

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
        water_val = (
            np.nan
            if (self.water_content_wt_pct is None or not math.isfinite(float(self.water_content_wt_pct)))
            else float(self.water_content_wt_pct)
        )
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
            "organic_solvent_type": _clean_category(self.organic_solvent_type),
            "water_content_wt_pct": water_val,
            "base_solvent": _clean_base_solvent(self.base_solvent),
        }


class ElectrochemicalFeatureExtractor:
    """Extracts material, solution, and physicochemical interaction features without data leakage."""

    def __init__(self, material_properties: pd.DataFrame | None = None) -> None:
        self._material_lookup: dict[str, dict[str, float]] = {}
        for metal, row in PURE_ELEMENT_PROPERTIES.items():
            self._material_lookup[metal] = {
                col: float(row[col])
                for col in _MATERIAL_DESCRIPTOR_COLS
                if col in row and not pd.isna(row[col])
            }
        if material_properties is not None and not material_properties.empty:
            col_key = next(
                (c for c in ("metal", "material_id", "symbol") if c in material_properties.columns),
                None,
            )
            if col_key is not None:
                for _, row in material_properties.iterrows():
                    metal = str(row[col_key]).strip()
                    if not metal:
                        continue
                    # For alloys/compounds (e.g., TiN, TaN), inherit base-element descriptors first
                    base_elem = (
                        str(row["base_element"]).strip()
                        if "base_element" in row and not pd.isna(row["base_element"])
                        else infer_base_element(metal)
                    )
                    entry = self._material_lookup.setdefault(metal, {})
                    if base_elem and base_elem in PURE_ELEMENT_PROPERTIES:
                        for col in _MATERIAL_DESCRIPTOR_COLS:
                            if col not in entry and col in PURE_ELEMENT_PROPERTIES[base_elem]:
                                val = PURE_ELEMENT_PROPERTIES[base_elem][col]
                                if not pd.isna(val):
                                    entry[col] = float(val)
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

        for col in ("oxidizer_type", "acid_type", "inhibitor_type", "organic_solvent_type"):
            if col in work.columns:
                work[col] = work[col].apply(_clean_category)
            else:
                work[col] = "None"

        if "base_solvent" in work.columns:
            work["base_solvent"] = work["base_solvent"].apply(_clean_base_solvent)
        else:
            work["base_solvent"] = "DI water"

        for col in (
            "pH",
            "temperature_C",
            "oxidizer_concentration",
            "acid_concentration",
            "inhibitor_concentration",
            "ORP_mV",
            "water_content_wt_pct",
        ):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
            else:
                work[col] = np.nan

        # Populate material descriptors from lookup (with base-element fallback for unseen alloys/compounds)
        def _resolve_descriptor(material_name: str, col_name: str) -> float:
            direct = self._material_lookup.get(material_name, {}).get(col_name)
            if direct is not None:
                return float(direct)
            base_elem = infer_base_element(material_name)
            if base_elem:
                base_val = self._material_lookup.get(base_elem, {}).get(col_name)
                if base_val is not None:
                    return float(base_val)
            return float("nan")

        for col in _MATERIAL_DESCRIPTOR_COLS:
            mapped = work["metal"].map(lambda m, c=col: _resolve_descriptor(m, c))
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce").fillna(mapped)
            else:
                work[col] = mapped

        return work

    def fit(self, df: pd.DataFrame) -> ElectrochemicalFeatureExtractor:
        work = self._enrich_dataframe(df)

        # Learn training-only medians for numeric columns (falling back to physical defaults only if all NaN)
        fallback_defaults = {"pH": 7.0, "temperature_C": 25.0, "water_content_wt_pct": 100.0}
        self._numeric_medians = {}
        for col in (
            *_MATERIAL_DESCRIPTOR_COLS,
            "pH",
            "temperature_C",
            "oxidizer_concentration",
            "acid_concentration",
            "inhibitor_concentration",
            "ORP_mV",
            "water_content_wt_pct",
        ):
            default_val = fallback_defaults.get(col, 0.0)
            med = float(work[col].median()) if work[col].notna().any() else default_val
            self._numeric_medians[col] = med if math.isfinite(med) else default_val

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
            "water_content_wt_pct",
            "effective_dielectric_constant",
            "solvent_polarity_index",
            "is_protic_solvent",
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

        # Engineered physicochemical interaction features & water-weighted Pourbaix passivation
        water_pct = np.clip(work["water_content_wt_pct"].to_numpy(dtype=float), 0.0, 100.0)
        water_frac = np.clip(water_pct / 100.0, 0.0, 1.0)
        solvent_desc = [
            compute_mixture_solvent_descriptors(w_val, org_val, base_val)
            for w_val, org_val, base_val in zip(
                water_pct,
                work["organic_solvent_type"].tolist(),
                work["base_solvent"].tolist(),
            )
        ]
        eff_dielectric = np.array([d[0] for d in solvent_desc], dtype=float)
        polarity_idx = np.array([d[1] for d in solvent_desc], dtype=float)
        is_protic = np.array([d[2] for d in solvent_desc], dtype=float)

        ph_x_t = ph * temp
        e0_minus_orp = e0 - (orp_mv / 1000.0)
        log1p_ox = np.log10(1.0 + ox_c)
        log1p_acid = np.log10(1.0 + acid_c)
        log1p_inh = np.log10(1.0 + inh_c)
        pb_indicator = ((ph >= pb_min) & (ph <= pb_max)).astype(float) * water_frac
        pb_dist = (
            np.where(
                ph < pb_min,
                pb_min - ph,
                np.where(ph > pb_max, ph - pb_max, 0.0),
            )
            * water_frac
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
            water_pct,
            eff_dielectric,
            polarity_idx,
            is_protic,
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


def _numeric_column_with_default(df: pd.DataFrame, col: str, default: float) -> np.ndarray:
    """Return numeric column array filled with ``default`` when missing or non-numeric."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(dtype=float)
    return np.full(len(df), default, dtype=float)


class ResidualConditionTransformer:
    """Transforms Exposure Conditions into physically floored low-dimensional vectors for GPs."""

    def __init__(self, *, include_metal: bool = False) -> None:
        self.include_metal = bool(include_metal)
        self._means: np.ndarray = np.array([], dtype=float)
        self._scales: np.ndarray = np.array([], dtype=float)
        self._include_orp: bool = False
        self._orp_median: float = 0.0
        self._cat_cols: tuple[str, ...] = (
            ("metal", *_RESIDUAL_CAT_COLS) if self.include_metal else _RESIDUAL_CAT_COLS
        )
        self._cat_vocab: dict[str, list[str]] = {}
        self._is_fitted: bool = False

    def _extract_numeric_matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, dict[str, list[str]]]:
        ph = _numeric_column_with_default(df, "pH", 7.0)
        temp = _numeric_column_with_default(df, "temperature_C", 25.0)
        ox_c = np.maximum(_numeric_column_with_default(df, "oxidizer_concentration", 0.0), 0.0)
        acid_c = np.maximum(_numeric_column_with_default(df, "acid_concentration", 0.0), 0.0)
        inh_c = np.maximum(_numeric_column_with_default(df, "inhibitor_concentration", 0.0), 0.0)
        water_pct = np.clip(_numeric_column_with_default(df, "water_content_wt_pct", 100.0), 0.0, 100.0)

        org_vals = [
            _clean_category(v)
            for v in (
                df["organic_solvent_type"].tolist()
                if "organic_solvent_type" in df.columns
                else ["None"] * len(df)
            )
        ]
        base_vals = [
            _clean_base_solvent(v)
            for v in (
                df["base_solvent"].tolist()
                if "base_solvent" in df.columns
                else ["DI water"] * len(df)
            )
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
            cols.append(_numeric_column_with_default(df, "ORP_mV", self._orp_median))

        cats: dict[str, list[str]] = {}
        for col in self._cat_cols:
            if col == "metal":
                cats[col] = (
                    [str(v).strip() for v in df["metal"].tolist()]
                    if "metal" in df.columns
                    else ["Unknown"] * len(df)
                )
            elif col in df.columns:
                cats[col] = [_clean_category(v) for v in df[col].tolist()]
            else:
                cats[col] = ["None"] * len(df)
        return np.column_stack(cols), cats

    def fit(self, df: pd.DataFrame) -> ResidualConditionTransformer:
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

        self._cat_vocab = {col: sorted(set(cats[col])) for col in self._cat_cols}
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("ResidualConditionTransformer must be fitted before transform().")
        num_mat, cats = self._extract_numeric_matrix(df)
        num_scaled = (num_mat - self._means) / self._scales

        cat_cols: list[np.ndarray] = []
        for col in self._cat_cols:
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

    def transform_query(self, query: ExposureCondition | MLFeatureQuery) -> np.ndarray:
        q = MLFeatureQuery.from_exposure_condition(query)
        return self.transform(pd.DataFrame([q.to_record_dict()]))


def fit_zero_mean_residual_gp(
    condition_df: pd.DataFrame,
    residuals: np.ndarray,
    weights: np.ndarray,
    *,
    target_name: str | None = None,
    length_scale_bounds: tuple[float, float] = (0.2, 5.0),
    alpha_scale: float = 0.01,
    n_restarts_optimizer: int = 1,
    random_state: int = 42,
) -> tuple[ResidualConditionTransformer, GaussianProcessRegressor]:
    """Fit a low-dimensional Zero-Mean Matérn(2.5) + WhiteKernel GP on weighted residuals."""
    transformer = ResidualConditionTransformer(include_metal=False)
    X_res = transformer.fit_transform(condition_df)

    min_amp = RESIDUAL_MIN_AMPLITUDE.get(target_name or "", 0.02)
    v = max(float(np.average(residuals**2, weights=weights)), min_amp**2)
    kernel = (
        ConstantKernel(
            constant_value=v,
            constant_value_bounds=(max(1e-6, 0.1 * v), max(1e2, 25.0 * v)),
        )
        * Matern(
            length_scale=1.0,
            length_scale_bounds=length_scale_bounds,
            nu=2.5,
        )
        + WhiteKernel(
            noise_level=0.05 * v,
            noise_level_bounds=(max(1e-8, 1e-5 * v), max(1e-3, 0.5 * v)),
        )
    )
    alpha = alpha_scale * v * ((1.0 / weights) ** 2 - 1.0) + 1e-10
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=False,
        random_state=random_state,
        n_restarts_optimizer=n_restarts_optimizer,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gpr.fit(X_res, residuals)
    return transformer, gpr


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
    ad_mode: str = "literature_only"
    lab_in_bounding_box: bool | None = None
    lab_knn_distance: float | None = None
    n_lab_samples: int = 0
    lab_anchoring_status: str | None = None
    cross_model_disagreement_log10_icorr: float | None = None
    cross_model_disagreement_ecorr_V: float | None = None
    disagreement_downgraded: bool = False


class ApplicabilityDomainChecker:
    """Evaluates per-metal Bounding Box coverage and local k-NN density to assign Confidence Tiers."""

    def __init__(
        self,
        *,
        k: int = 3,
        medium_distance_threshold: float = 0.35,
        high_distance_threshold: float = 0.75,
        min_high_confidence_samples: int = 5,
        min_low_confidence_samples: int = DEFAULT_LOW_SAMPLE_THRESHOLD,
    ) -> None:
        if k < 1:
            raise ValueError("k must be at least 1.")
        self.k = int(k)
        self.medium_distance_threshold = float(medium_distance_threshold)
        self.high_distance_threshold = float(high_distance_threshold)
        self.min_high_confidence_samples = int(min_high_confidence_samples)
        self.min_low_confidence_samples = int(min_low_confidence_samples)

        self._metal_stats: dict[str, dict[str, Any]] = {}
        self._global_spans: dict[str, float] = {}
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> ApplicabilityDomainChecker:
        work = df.copy()
        work["metal"] = work["metal"].astype(str).str.strip()
        ad_numeric_defaults = {
            "temperature_C": 25.0,
            "pH": 7.0,
            "water_content_wt_pct": 100.0,
        }
        for col in _APPLICABILITY_NUMERIC_COLS:
            default_val = ad_numeric_defaults.get(col, 0.0)
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce").fillna(default_val)
            else:
                work[col] = default_val
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
            cat_matrix = np.column_stack(
                [grp[col].astype(str).to_numpy() for col in _APPLICABILITY_CATEGORICAL_COLS]
            )
            self._metal_stats[str(metal)] = {
                "n_samples": len(grp),
                "bounds": bounds,
                "scales": scales,
                "categories": cats,
                "points": points,
                "cat_matrix": cat_matrix,
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
        q_numeric_vals: list[float] = []

        for col in _APPLICABILITY_NUMERIC_COLS:
            raw_val = rec[col]
            val = float("nan") if raw_val is None else float(raw_val)
            lo, hi = bounds[col]
            if not math.isfinite(val):
                out_of_box.append(col)
                warnings.append(
                    f"{col} is missing or non-finite; applicability domain coverage cannot be "
                    f"verified against the {metal} training range [{lo:.3g}, {hi:.3g}]."
                )
                recommendations.append(
                    f"Provide a finite {col} value for {metal} (current training coverage: "
                    f"[{lo:.3g}, {hi:.3g}])."
                )
                q_numeric_vals.append(0.5 * (lo + hi))
                continue
            q_numeric_vals.append(val)
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
            if val not in cats[col]:
                unseen_cats.append(f"{col}={val}")
                warnings.append(
                    f"{col}={val!r} was never observed in the {metal} training set "
                    f"(observed: {', '.join(cats[col])})."
                )
                recommendations.append(
                    f"Collect experimental Tafel data for {metal} with {col}={val!r} "
                    f"at pH {q.pH:.1f} and {q.temperature_C:.1f} °C."
                )

        q_vec = np.array(q_numeric_vals, dtype=float)
        scale_vec = np.array([scales[col] for col in _APPLICABILITY_NUMERIC_COLS], dtype=float)
        diffs = (points - q_vec) / scale_vec
        num_sq_dists = np.sum(diffs**2, axis=1)

        cat_matrix: np.ndarray = stats["cat_matrix"]
        q_cats = np.array([str(rec[col]) for col in _APPLICABILITY_CATEGORICAL_COLS], dtype=object)
        cat_mismatch_sq = 0.25 * np.sum(cat_matrix != q_cats[None, :], axis=1, dtype=float)

        dists = np.sqrt(num_sq_dists + cat_mismatch_sq)
        eff_k = min(self.k, len(dists))
        nearest_dists = np.sort(dists)[:eff_k]
        knn_mean_dist = float(np.mean(nearest_dists))

        in_box = len(out_of_box) == 0 and len(unseen_cats) == 0
        low_sample_threshold = getattr(self, "min_low_confidence_samples", DEFAULT_LOW_SAMPLE_THRESHOLD)
        is_low_sample = n_samples < low_sample_threshold
        if is_low_sample:
            warnings.append(
                f"Low training sample count for {metal} (n={n_samples} < {low_sample_threshold}); "
                "prediction relies heavily on cross-material physical descriptors."
            )
            recommendations.append(
                f"Collect additional experimental polarization data for {metal} (currently n={n_samples}) "
                f"to raise coverage above {low_sample_threshold} records."
            )

        if not in_box or knn_mean_dist > self.high_distance_threshold or is_low_sample:
            tier: ConfidenceTier = "Low"
            if in_box and not is_low_sample:
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
class CrossValidationMetrics:
    """Cross-validated regression metrics for one electrochemical target."""

    target: str
    model_family: PredictedModelFamily
    cv_strategy: CVStrategy
    mae: float
    rmse: float
    r2: float
    mape: float
    n_samples: int
    per_metal_metrics: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class _PerMetalResidualGP:
    """Per-Metal Stage 2 Zero-Mean OOB residual GP fitted in low-dimensional Exposure Condition space."""

    transformer: ResidualConditionTransformer
    gp_model: GaussianProcessRegressor
    n_samples: int
    subsampled: bool


class _UncertainTargetModel:
    """Wraps a single-target regressor providing (mean, std) predictions."""

    def __init__(
        self,
        model_family: ModelFamily,
        *,
        random_state: int = 42,
        n_estimators: int = 100,
        n_bootstrap: int = 10,
        max_gp_samples: int = GP_MAX_TRAINING_SAMPLES,
        target_name: str | None = None,
    ) -> None:
        if model_family not in SUPPORTED_MODEL_FAMILIES:
            raise ValueError(
                f"Unsupported model_family: {model_family!r}. Supported: {SUPPORTED_MODEL_FAMILIES}"
            )
        self.model_family: ModelFamily = model_family
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.n_bootstrap = int(n_bootstrap)
        self.max_gp_samples = int(max_gp_samples)
        self.target_name: str | None = target_name
        self.gp_subsampled: bool = False
        self._model: Any = None
        self._per_metal_residual_gps: dict[str, _PerMetalResidualGP] = {}
        self._ensemble: list[Any] = []

    def _select_stratified_indices(self, y: np.ndarray, *, seed_offset: int = 0) -> np.ndarray:
        """Deterministically cap per-Metal GP training size via quantile-stratified selection."""
        n = len(y)
        if n <= self.max_gp_samples:
            return np.arange(n, dtype=int)
        rng = np.random.default_rng(self.random_state + seed_offset)
        order = np.argsort(y)
        bins = np.array_split(order, min(20, self.max_gp_samples))
        selected: list[int] = []
        remaining = self.max_gp_samples
        for i, b in enumerate(bins):
            quota = remaining // (len(bins) - i)
            take = min(len(b), quota)
            if take > 0:
                chosen = rng.choice(b, size=take, replace=False)
                selected.extend(int(idx) for idx in chosen)
                remaining -= take
        return np.array(sorted(selected), dtype=int)

    def _fit_per_metal_oob_residual_gps(
        self,
        condition_df: pd.DataFrame,
        y: np.ndarray,
        oob_pred: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> None:
        """Fit Stage 2 per-Metal Zero-Mean Matérn(2.5) GPs on valid out-of-bag RF residuals."""
        self._per_metal_residual_gps = {}
        self.gp_subsampled = False

        valid_oob = np.isfinite(oob_pred) & np.isfinite(y)
        if not np.any(valid_oob):
            return

        residuals = y - oob_pred
        metals = condition_df["metal"].astype(str).str.strip().to_numpy()
        weights_all = (
            np.clip(np.asarray(sample_weight, dtype=float), 0.05, 1.0)
            if sample_weight is not None
            else np.ones(len(y), dtype=float)
        )

        for metal_idx, metal in enumerate(sorted(set(metals))):
            m_mask = valid_oob & (metals == metal)
            n_valid = int(np.sum(m_mask))
            if n_valid < MIN_RESIDUAL_GP_SAMPLES:
                continue

            df_m = condition_df.loc[m_mask].reset_index(drop=True)
            res_m = residuals[m_mask]
            w_m = weights_all[m_mask]

            if n_valid > self.max_gp_samples:
                idx_sub = self._select_stratified_indices(res_m, seed_offset=metal_idx * 19)
                df_m = df_m.iloc[idx_sub].reset_index(drop=True)
                res_m = res_m[idx_sub]
                w_m = w_m[idx_sub]
                subsampled = True
                self.gp_subsampled = True
            else:
                subsampled = False

            transformer, gpr = fit_zero_mean_residual_gp(
                df_m,
                res_m,
                w_m,
                target_name=self.target_name,
                length_scale_bounds=(0.2, 5.0),
                alpha_scale=0.01,
                n_restarts_optimizer=1,
                random_state=self.random_state + metal_idx * 29,
            )

            self._per_metal_residual_gps[metal] = _PerMetalResidualGP(
                transformer=transformer,
                gp_model=gpr,
                n_samples=len(res_m),
                subsampled=subsampled,
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        condition_df: pd.DataFrame | None = None,
    ) -> _UncertainTargetModel:
        if self.model_family == "random_forest":
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                min_samples_leaf=1,
            )
            rf.fit(X, y, sample_weight=sample_weight)
            self._model = rf
        elif self.model_family == "hist_gradient_boosting":
            rng = np.random.default_rng(self.random_state)
            n_samples = len(y)
            self._ensemble = []
            for b in range(self.n_bootstrap):
                idx = rng.integers(0, n_samples, size=n_samples)
                hgbr = HistGradientBoostingRegressor(
                    max_iter=max(50, self.n_estimators),
                    max_depth=5,
                    l2_regularization=1.0,
                    random_state=self.random_state + b,
                )
                w_boot = sample_weight[idx] if sample_weight is not None else None
                hgbr.fit(X[idx], y[idx], sample_weight=w_boot)
                self._ensemble.append(hgbr)
        elif self.model_family == "rf_gp_ensemble":
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                min_samples_leaf=1,
                oob_score=True,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                rf.fit(X, y, sample_weight=sample_weight)
            self._model = rf
            self._per_metal_residual_gps = {}
            self.gp_subsampled = False
            if condition_df is not None and len(condition_df) == len(y):
                oob_pred = np.asarray(rf.oob_prediction_, dtype=float)
                self._fit_per_metal_oob_residual_gps(
                    condition_df.reset_index(drop=True),
                    np.asarray(y, dtype=float),
                    oob_pred,
                    sample_weight,
                )
        else:
            raise ValueError(f"Unsupported model_family: {self.model_family!r}")
        return self

    def predict_mean_std(
        self,
        X: np.ndarray,
        *,
        condition_df: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.model_family == "random_forest":
            tree_preds = np.vstack([tree.predict(X) for tree in self._model.estimators_])
            return np.mean(tree_preds, axis=0), np.std(tree_preds, axis=0)
        if self.model_family == "hist_gradient_boosting":
            boot_preds = np.vstack([m.predict(X) for m in self._ensemble])
            return np.mean(boot_preds, axis=0), np.std(boot_preds, axis=0)
        if self.model_family == "rf_gp_ensemble":
            tree_preds = np.vstack([tree.predict(X) for tree in self._model.estimators_])
            rf_mean = np.mean(tree_preds, axis=0)
            rf_std = np.std(tree_preds, axis=0)
            if condition_df is None or not self._per_metal_residual_gps:
                return rf_mean, rf_std

            comb_mean = rf_mean.copy()
            comb_std = rf_std.copy()
            cond_reset = condition_df.reset_index(drop=True)
            metals = cond_reset["metal"].astype(str).str.strip().to_numpy()
            for metal in dict.fromkeys(metals):
                fitted_gp = self._per_metal_residual_gps.get(metal)
                if fitted_gp is None:
                    continue
                idx = np.flatnonzero(metals == metal)
                X_res = fitted_gp.transformer.transform(cond_reset.iloc[idx].reset_index(drop=True))
                r_mean, r_std = fitted_gp.gp_model.predict(X_res, return_std=True)
                r_mean_arr = np.asarray(r_mean, dtype=float)
                r_std_arr = np.maximum(np.asarray(r_std, dtype=float), 0.0)
                comb_mean[idx] = rf_mean[idx] + r_mean_arr
                comb_std[idx] = np.hypot(rf_std[idx], r_std_arr)
            return comb_mean, comb_std
        raise ValueError(f"Unsupported model_family: {self.model_family!r}")

    def feature_importances(self) -> np.ndarray:
        """Return feature importance vector for tree-based model families."""
        if self.model_family in ("random_forest", "rf_gp_ensemble"):
            return np.asarray(self._model.feature_importances_, dtype=float)
        raise ValueError(
            f"Feature importances are only available for tree models, not {self.model_family!r}."
        )


def extract_target_vectors(df: pd.DataFrame) -> dict[str, np.ndarray]:
    ecorr = pd.to_numeric(df["Ecorr_V_SHE"], errors="coerce").to_numpy(dtype=float)
    icorr = pd.to_numeric(df["icorr_uA_cm2"], errors="coerce").to_numpy(dtype=float)
    valid_icorr = np.isfinite(icorr) & (icorr > 0.0)
    log_icorr = np.where(
        valid_icorr,
        np.log10(np.where(valid_icorr, icorr, 1.0)),
        np.nan,
    )
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


def extract_target_valid_mask(
    df: pd.DataFrame,
    target_name: str,
    y_vec: np.ndarray,
    *,
    exclude_imputed_tafel: bool = True,
    fallback_min_measured: int | None = 3,
) -> np.ndarray:
    """Return boolean mask of valid (and non-imputed, when enabled) rows for a target."""
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

    measured_mask = finite_mask & ~df[imputed_col].fillna(False).astype(bool).to_numpy()
    if fallback_min_measured is None or int(np.sum(measured_mask)) >= fallback_min_measured:
        return measured_mask
    return finite_mask


def extract_sample_weights(
    df: pd.DataFrame,
    *,
    use_quality_weights: bool = True,
) -> np.ndarray | None:
    """Extract normalized quality_score sample weights when available."""
    if not use_quality_weights or "quality_score" not in df.columns:
        return None
    weights = pd.to_numeric(df["quality_score"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    return np.clip(weights, 0.05, 1.0)


def _build_cv_splits(
    df: pd.DataFrame,
    *,
    cv_strategy: CVStrategy,
    random_state: int,
    group_col: str = "source_reference",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build train/validation index splits for the requested CV strategy."""
    n_samples = len(df)
    if n_samples < 3:
        raise ValueError("At least 3 samples are required for cross-validation.")

    if cv_strategy == "loocv":
        return list(LeaveOneOut().split(df))

    n_splits = min(5, n_samples)
    if cv_strategy == "group_5fold":
        if group_col in df.columns:
            raw_groups = df[group_col].astype(object).to_numpy()
            groups = np.array(
                [
                    str(v).strip()
                    if (v is not None and not pd.isna(v) and str(v).strip() and str(v).strip().lower() != "nan")
                    else f"_ungrouped_{i}"
                    for i, v in enumerate(raw_groups)
                ],
                dtype=object,
            )
        else:
            groups = np.array([f"_ungrouped_{i}" for i in range(n_samples)], dtype=object)

        n_unique_groups = len(set(groups))
        if n_unique_groups >= 2:
            gkf = GroupKFold(n_splits=min(n_splits, n_unique_groups))
            return list(gkf.split(df, groups=groups))
        return list(KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(df))

    if cv_strategy == "stratified_5fold":
        metal_counts = df["metal"].value_counts()
        strat_labels = df["metal"].where(
            df["metal"].map(metal_counts) >= n_splits, "_rare_metal"
        )
        if strat_labels.value_counts().min() >= n_splits:
            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            return list(splitter.split(df, strat_labels))
        splitter_kf = KFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        return list(splitter_kf.split(df))

    raise ValueError(f"Unsupported cv_strategy: {cv_strategy!r}")


def _compute_per_metal_metrics(
    metals: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute regression metrics broken down by metal."""
    breakdown: dict[str, dict[str, float]] = {}
    for metal in sorted({str(m) for m in metals}):
        mask = metals == metal
        n_m = int(np.sum(mask))
        yt = y_true[mask]
        yp = y_pred[mask]
        r2_val = (
            float(r2_score(yt, yp))
            if (n_m >= 2 and float(np.var(yt)) > 1e-12)
            else float("nan")
        )
        breakdown[metal] = {
            "n_samples": float(n_m),
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(root_mean_squared_error(yt, yp)),
            "r2": r2_val,
            "mape": _compute_mape(yt, yp),
        }
    return breakdown


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
        exclude_imputed_tafel: bool = True,
        use_quality_weights: bool = True,
        max_gp_samples: int = GP_MAX_TRAINING_SAMPLES,
    ) -> None:
        if model_family not in SUPPORTED_MODEL_FAMILIES:
            raise ValueError(
                f"Unsupported model_family: {model_family!r}. Supported: {SUPPORTED_MODEL_FAMILIES}"
            )
        self.model_family: ModelFamily = model_family
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.n_bootstrap = int(n_bootstrap)
        self.knn_k = int(knn_k)
        self.exclude_imputed_tafel = bool(exclude_imputed_tafel)
        self.use_quality_weights = bool(use_quality_weights)
        self.max_gp_samples = int(max_gp_samples)

        self.feature_extractor: ElectrochemicalFeatureExtractor | None = None
        self.applicability_checker: ApplicabilityDomainChecker | None = None
        self.models: dict[str, _UncertainTargetModel] = {}
        self.target_training_counts: dict[str, int] = {}
        self._material_properties: pd.DataFrame | None = None
        self._is_fitted: bool = False

    def fit(self, database: HarmonizedDatabase) -> HybridMLPredictor:
        df = database.ml_training_dataframe.reset_index(drop=True)
        if df.empty:
            raise ValueError("Cannot train HybridMLPredictor on an empty database.")

        self._material_properties = database.material_properties.copy()
        self.feature_extractor = ElectrochemicalFeatureExtractor(
            material_properties=self._material_properties
        )
        X = self.feature_extractor.fit_transform(df)
        targets = extract_target_vectors(df)
        sample_weights = extract_sample_weights(df, use_quality_weights=self.use_quality_weights)

        self.models = {}
        self.target_training_counts = {}
        for idx, target_name in enumerate(ML_TARGET_NAMES):
            y_all = targets[target_name]
            mask = extract_target_valid_mask(
                df,
                target_name,
                y_all,
                exclude_imputed_tafel=self.exclude_imputed_tafel,
            )
            w_target = sample_weights[mask] if sample_weights is not None else None
            df_target = df.loc[mask].reset_index(drop=True)
            model = _UncertainTargetModel(
                self.model_family,
                random_state=self.random_state + idx * 17,
                n_estimators=self.n_estimators,
                n_bootstrap=self.n_bootstrap,
                max_gp_samples=self.max_gp_samples,
                target_name=target_name,
            )
            model.fit(X[mask], y_all[mask], sample_weight=w_target, condition_df=df_target)
            self.models[target_name] = model
            self.target_training_counts[target_name] = int(np.sum(mask))

        self.applicability_checker = ApplicabilityDomainChecker(k=self.knn_k).fit(df)
        self._is_fitted = True
        return self

    @property
    def is_fitted(self) -> bool:
        """Return True once all four electrochemical target models have been fitted."""
        return bool(self._is_fitted)

    def predict_prior_means_dataframe(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Predict physically clipped prior target means for a batch DataFrame of Exposure Conditions."""
        if not self._is_fitted or self.feature_extractor is None:
            raise RuntimeError("HybridMLPredictor must be fitted before predict_prior_means_dataframe().")
        X = self.feature_extractor.transform(df)
        prior_means: dict[str, np.ndarray] = {}
        for target_name in ML_TARGET_NAMES:
            raw_mean, _ = self.models[target_name].predict_mean_std(X, condition_df=df)
            prior_means[target_name] = np.array(
                [clip_target_mean(target_name, float(v)) for v in raw_mean],
                dtype=float,
            )
        return prior_means

    def predict(self, query: ExposureCondition | MLFeatureQuery) -> MLParameterPrediction:
        if not self._is_fitted or self.feature_extractor is None or self.applicability_checker is None:
            raise RuntimeError("HybridMLPredictor must be fitted before predict().")

        q = MLFeatureQuery.from_exposure_condition(query)
        q_df = pd.DataFrame([q.to_record_dict()])
        X_q = self.feature_extractor.transform(q_df)
        ad_res = self.applicability_checker.evaluate(q)

        estimates: dict[str, ParameterEstimate] = {}
        for target_name in ML_TARGET_NAMES:
            mean_arr, std_arr = self.models[target_name].predict_mean_std(X_q, condition_df=q_df)
            clipped_mean = clip_target_mean(target_name, float(mean_arr[0]))
            estimates[target_name] = ParameterEstimate(
                mean=clipped_mean,
                std=float(std_arr[0]),
            )

        log_i_mean = estimates["log10_icorr_uA_cm2"].mean
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
        group_col: str = "source_reference",
    ) -> list[CrossValidationMetrics]:
        """Run leak-free cross-validation (fitting feature extractor inside every fold).

        When ``exclude_imputed_tafel`` is enabled, ``beta_a_mV_dec`` and ``beta_c_mV_dec``
        are cross-validated exclusively on experimentally measured (non-imputed) records
        to prevent group-median imputation leakage across folds.
        """
        df = database.ml_training_dataframe.reset_index(drop=True)
        if len(df) < 3:
            raise ValueError("At least 3 samples are required for cross-validation.")

        all_targets = extract_target_vectors(df)
        all_weights = extract_sample_weights(df, use_quality_weights=self.use_quality_weights)

        results: list[CrossValidationMetrics] = []
        for t_idx, target_name in enumerate(ML_TARGET_NAMES):
            y_all = all_targets[target_name]
            mask = extract_target_valid_mask(
                df,
                target_name,
                y_all,
                exclude_imputed_tafel=self.exclude_imputed_tafel,
            )
            sub_df = df.loc[mask].reset_index(drop=True)
            y_true = y_all[mask]
            sub_weights = all_weights[mask] if all_weights is not None else None
            n_target_samples = len(sub_df)

            splits = _build_cv_splits(
                sub_df,
                cv_strategy=cv_strategy,
                random_state=self.random_state,
                group_col=group_col,
            )
            oof_pred = np.zeros(n_target_samples, dtype=float)

            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                train_df = sub_df.iloc[train_idx].reset_index(drop=True)
                val_df = sub_df.iloc[val_idx].reset_index(drop=True)

                # Strict featurization ordering: fit extractor on train fold only
                extractor = ElectrochemicalFeatureExtractor(
                    material_properties=database.material_properties
                )
                X_train = extractor.fit_transform(train_df)
                X_val = extractor.transform(val_df)

                w_train = sub_weights[train_idx] if sub_weights is not None else None
                model = _UncertainTargetModel(
                    self.model_family,
                    random_state=self.random_state + fold_idx * 31 + t_idx * 7,
                    n_estimators=self.n_estimators,
                    n_bootstrap=self.n_bootstrap,
                    max_gp_samples=self.max_gp_samples,
                    target_name=target_name,
                )
                model.fit(
                    X_train,
                    y_true[train_idx],
                    sample_weight=w_train,
                    condition_df=train_df,
                )
                mean_pred, _ = model.predict_mean_std(X_val, condition_df=val_df)
                oof_pred[val_idx] = mean_pred

            metals_arr = sub_df["metal"].astype(str).str.strip().to_numpy()
            results.append(
                CrossValidationMetrics(
                    target=target_name,
                    model_family=self.model_family,
                    cv_strategy=cv_strategy,
                    mae=float(mean_absolute_error(y_true, oof_pred)),
                    rmse=float(root_mean_squared_error(y_true, oof_pred)),
                    r2=float(r2_score(y_true, oof_pred)),
                    mape=_compute_mape(y_true, oof_pred),
                    n_samples=n_target_samples,
                    per_metal_metrics=_compute_per_metal_metrics(metals_arr, y_true, oof_pred),
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
        importances = wrapper.feature_importances()
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

    e_band = band.ecorr_p50_V_SHE + band.overpotential_V
    ax.fill_betweenx(
        e_band,
        band.eta_log10_abs_net_p05,
        band.eta_log10_abs_net_p95,
        color="#1f77b4",
        alpha=0.14,
        edgecolor="#1f77b4",
        linewidth=0.6,
        label=f"Net |i_net| 90% Band (p05–p95, N={mc_result.n_draws})",
    )
    ax.fill_betweenx(
        e_band,
        band.eta_log10_abs_net_p25,
        band.eta_log10_abs_net_p75,
        color="#1f77b4",
        alpha=0.28,
        edgecolor="#1f77b4",
        linewidth=0.8,
        label="Net |i_net| 50% Band (p25–p75)",
    )
    ax.plot(
        band.eta_log10_abs_net_p50,
        e_band,
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
