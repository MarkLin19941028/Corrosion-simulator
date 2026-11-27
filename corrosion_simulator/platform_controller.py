"""Application controller and interactive Plotly figure builders for the Corrosion Simulator UI."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from dataclasses import dataclass, field, replace, replace
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from corrosion_simulator.database import (
    CORE_METALS,
    DEFAULT_NON_AQUEOUS_WATER_WT_PCT,
    DEFAULT_SEMI_AQUEOUS_WATER_WT_PCT,
    SOLVENT_REGIMES,
    ExposureCondition,
    ExposureLookupResult,
    HarmonizedDatabase,
    SolventRegime,
    SolventSubclass,
    classify_solvent_regime,
    classify_solvent_subclass,
    normalize_category_label,
)
from corrosion_simulator.solvent_constants import SOLVENT_PHYSICAL_PROPERTIES
from corrosion_simulator.galvanic import (
    GalvanicCouplingResult,
    MonteCarloGalvanicResult,
    propagate_galvanic_uncertainty_monte_carlo,
    solve_galvanic_couple,
)
from corrosion_simulator.kinetics import (
    DEFAULT_CRITICAL_THICKNESS_NM,
    DEFAULT_PASSIVATION_LIMIT_NM,
    DEFAULT_POWER_LAW_N,
    MonteCarloKineticsResult,
    TimeResolvedKineticsResult,
    classify_thickness_loss_risk_tier,
    propagate_kinetics_uncertainty_monte_carlo,
    simulate_time_resolved_kinetics,
)
from corrosion_simulator.lab_data import (
    LabProjectDataset,
    compute_lab_project_fingerprint,
    discover_and_load_lab_projects,
    load_lab_project,
)
from corrosion_simulator.lab_calibration import (
    CALIBRATION_MODES,
    CalibrationMode,
    InsufficientLabSamplesError,
    LabCalibratedPredictor,
    ResidualBreakdown,
)
from corrosion_simulator.ml_predictor import (
    MODEL_FAMILY_LABELS,
    SUPPORTED_MODEL_FAMILIES,
    ApplicabilityDomainResult,
    ConfidenceTier,
    HybridMLPredictor,
    MLFeatureQuery,
    MLParameterPrediction,
    ModelFamily,
    MonteCarloPropagationResult,
    PredictedModelFamily,
    propagate_uncertainty_monte_carlo,
)
from corrosion_simulator.paths import (
    discover_lab_projects,
    iter_raw_csv_files,
    resolve_data_dirs,
    resolve_raw_and_processed_dirs,
)
from corrosion_simulator.physics import (
    FaradayCorrosionResult,
    GalvanicDrivingForceResult,
    PolarizationCurve,
    PolarizationParameters,
    calculate_faraday_corrosion_rate,
    generate_polarization_curve,
    has_faraday_properties,
    screen_galvanic_driving_force,
)

ParameterSource = Literal["database", "ml", "manual"]
KineticsSelection = Literal["auto", "linear", "passivation", "power_law"]
ArtifactStatus = Literal["fresh", "rebuilt", "stale", "live_fallback", "live", "seed"]
LabReloadStatus = Literal["none", "fresh", "reloaded"]


@dataclass(frozen=True)
class ExternalDataReloadSummary:
    """Structured result of dual-fingerprint Reload External Data."""

    raw_rebuilt: bool
    raw_status: ArtifactStatus
    lab_status: LabReloadStatus
    lab_projects: tuple[str, ...]
    lab_added_or_updated: tuple[str, ...]
    lab_removed: tuple[str, ...]
    lab_cache_invalidated: bool
    elapsed_s: float

_CONFIDENCE_RANK = {"Low": 0, "Medium": 1, "High": 2}
_CONFIDENCE_BY_RANK = {0: "Low", 1: "Medium", 2: "High"}
NON_AQUEOUS_WARNING_PREFIX = "Non-aqueous Exposure Condition"

CALIBRATION_MODE_DEFAULT_MODELS: Final[dict[CalibrationMode, PredictedModelFamily]] = {
    "literature_only": "rf_gp_ensemble",
    "delta_learning": "hist_gradient_boosting",
    "lab_only_gp": "lab_local_gp",
}

_AUTO_ROUTED_BASE_MODELS: Final[dict[CalibrationMode, ModelFamily]] = {
    "literature_only": "rf_gp_ensemble",
    "delta_learning": "hist_gradient_boosting",
    "lab_only_gp": "hist_gradient_boosting",
}

DISAGREEMENT_SEVERE_LOG10_ICORR: Final[float] = 1.0
DISAGREEMENT_MODERATE_LOG10_ICORR: Final[float] = 0.5
DISAGREEMENT_MODERATE_ECORR_V: Final[float] = 0.08


def resolve_model_family_for_mode(
    calibration_mode: CalibrationMode,
    override_model_family: ModelFamily | None = None,
) -> PredictedModelFamily:
    """Return the active model family for ``calibration_mode``, honoring valid base overrides."""
    if calibration_mode not in CALIBRATION_MODES:
        raise ValueError(
            f"Unsupported calibration_mode '{calibration_mode}'. Supported: {CALIBRATION_MODES}"
        )
    if calibration_mode == "lab_only_gp":
        return "lab_local_gp"
    if override_model_family is not None:
        if override_model_family not in SUPPORTED_MODEL_FAMILIES:
            raise ValueError(
                f"Unsupported ml_model_family '{override_model_family}'. Supported: {SUPPORTED_MODEL_FAMILIES}"
            )
        return override_model_family
    return CALIBRATION_MODE_DEFAULT_MODELS[calibration_mode]


def resolve_active_model_family(
    request: SimulationRequest,
    resolved: ResolvedMetalParameters | None,
) -> PredictedModelFamily:
    """Return the active ML model family from ``resolved`` or ``request`` mode routing."""
    if resolved is not None and resolved.ml_prediction is not None:
        return resolved.ml_prediction.model_family
    return resolve_model_family_for_mode(request.calibration_mode, request.ml_model_family)


def format_active_model_label(model_family: PredictedModelFamily | str) -> str:
    """Return human-readable display label for a global or local ML model family."""
    if model_family == "lab_local_gp":
        return "Lab-Only Local GP (Low-Dim Matérn 2.5)"
    return MODEL_FAMILY_LABELS.get(model_family, str(model_family))  # type: ignore[arg-type]


def resolve_consistent_solvent_settings(
    solvent_regime: str | None,
    organic_solvent_type: Any,
    water_content_wt_pct: float | None,
) -> tuple[SolventRegime, str, float]:
    """Resolve and validate mutually consistent ``(solvent_regime, organic_solvent_type, water_content_wt_pct)``."""
    org_type = normalize_category_label(organic_solvent_type)

    regime_val: SolventRegime | None = None
    if solvent_regime is not None:
        regime_str = str(solvent_regime).strip()
        if regime_str not in SOLVENT_REGIMES:
            raise ValueError(
                f"Unsupported solvent_regime '{solvent_regime}'. Supported: {SOLVENT_REGIMES}"
            )
        regime_val = regime_str  # type: ignore[assignment]

    water_num: float | None = None
    if water_content_wt_pct is not None:
        if isinstance(water_content_wt_pct, bool):
            raise ValueError("water_content_wt_pct must be a finite numeric percentage in [0, 100].")
        try:
            water_num = float(water_content_wt_pct)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"water_content_wt_pct must be numeric, got {water_content_wt_pct!r}."
            ) from exc
        if not math.isfinite(water_num) or not (0.0 <= water_num <= 100.0):
            raise ValueError(
                f"water_content_wt_pct must be finite and within [0.0, 100.0], got {water_content_wt_pct!r}."
            )

    if regime_val is None and water_num is None:
        if org_type != "None":
            return ("semi_aqueous", org_type, float(DEFAULT_SEMI_AQUEOUS_WATER_WT_PCT))
        return ("aqueous", "None", 100.0)

    if regime_val is not None and water_num is None:
        if regime_val == "aqueous":
            default_water = 100.0 if org_type == "None" else 95.0
        elif regime_val == "semi_aqueous":
            default_water = float(DEFAULT_SEMI_AQUEOUS_WATER_WT_PCT)
        else:
            default_water = float(DEFAULT_NON_AQUEOUS_WATER_WT_PCT)
        return (regime_val, org_type, default_water)

    if regime_val is None and water_num is not None:
        return (classify_solvent_regime(water_num), org_type, float(water_num))

    assert regime_val is not None and water_num is not None
    derived_regime = classify_solvent_regime(water_num)
    if derived_regime != regime_val:
        raise ValueError(
            f"Inconsistent solvent_regime='{regime_val}' and water_content_wt_pct={water_num:g}. "
            "Regime boundaries require aqueous >= 90.0 wt%, 1.0 <= semi_aqueous < 90.0 wt%, "
            "and non_aqueous < 1.0 wt% water."
        )
    return (regime_val, org_type, float(water_num))


def build_non_aqueous_physics_warning(
    water_content_wt_pct: float,
    organic_solvent_type: str = "None",
) -> str:
    """Build the standardized physics-limitation banner warning for non-aqueous (<1 wt% water) conditions."""
    org_label = normalize_category_label(organic_solvent_type)
    co_solvent = f" ({org_label})" if org_label != "None" else ""
    return (
        f"{NON_AQUEOUS_WARNING_PREFIX}{co_solvent} with {water_content_wt_pct:.2f} wt% water (< 1%): "
        "aqueous pH, V vs. SHE reference potential scale, and aqueous Pourbaix passivation assumptions "
        "are not strictly valid in this phase. Interpret simulated kinetics as exploratory screening only."
    )


@dataclass(frozen=True)
class ManualElectrochemicalInput:
    """User-supplied electrochemical parameters for manual override mode."""

    Ecorr_V_SHE: float
    icorr_uA_cm2: float
    beta_a_mV_dec: float = 60.0
    beta_c_mV_dec: float = 120.0
    valence: int | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.Ecorr_V_SHE):
            raise ValueError("Ecorr_V_SHE must be finite.")
        if self.icorr_uA_cm2 <= 0 or not np.isfinite(self.icorr_uA_cm2):
            raise ValueError("icorr_uA_cm2 must be strictly positive and finite.")
        if not (20.0 <= self.beta_a_mV_dec <= 500.0):
            raise ValueError("beta_a_mV_dec must lie in [20, 500] mV/dec.")
        if not (20.0 <= self.beta_c_mV_dec <= 500.0):
            raise ValueError("beta_c_mV_dec must lie in [20, 500] mV/dec.")


@dataclass(frozen=True)
class SimulationRequest:
    """Unified sidebar configuration for Single Metal and Galvanic Analysis."""

    primary_metal: str
    secondary_metal: str | None = None
    cathode_to_anode_area_ratio: float = 1.0
    oxidizer_type: str = "H2O2"
    oxidizer_concentration: float = 1.0
    acid_type: str = "None"
    acid_concentration: float = 0.0
    inhibitor_type: str = "None"
    inhibitor_concentration: float = 0.0
    pH: float = 3.0
    temperature_C: float = 25.0
    ORP_mV: float | None = None
    solvent_regime: SolventRegime | None = None
    organic_solvent_type: str = "None"
    water_content_wt_pct: float | None = None
    immersion_time_s: float = 300.0
    d_crit_nm: float = DEFAULT_CRITICAL_THICKNESS_NM
    kinetics_model: KineticsSelection = "auto"
    d_max_nm: float = DEFAULT_PASSIVATION_LIMIT_NM
    power_law_n: float = DEFAULT_POWER_LAW_N
    parameter_source: ParameterSource = "ml"
    ml_model_family: ModelFamily | None = None
    calibration_mode: CalibrationMode = "literature_only"
    lab_project: str | None = None
    manual_primary: ManualElectrochemicalInput | None = None
    manual_secondary: ManualElectrochemicalInput | None = None
    primary_valence: int | None = None
    secondary_valence: int | None = None
    mc_draws: int = 500
    mc_seed: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.primary_metal, str) or not self.primary_metal.strip():
            raise ValueError("primary_metal must be a non-empty material identifier.")
        if self.secondary_metal is not None and (
            not isinstance(self.secondary_metal, str) or not self.secondary_metal.strip()
        ):
            raise ValueError("secondary_metal must be None or a non-empty material identifier.")
        if self.cathode_to_anode_area_ratio <= 0 or not np.isfinite(self.cathode_to_anode_area_ratio):
            raise ValueError("cathode_to_anode_area_ratio must be strictly positive.")
        if self.immersion_time_s <= 0 or not np.isfinite(self.immersion_time_s):
            raise ValueError("immersion_time_s must be strictly positive.")
        if self.d_crit_nm <= 0 or not np.isfinite(self.d_crit_nm):
            raise ValueError("d_crit_nm must be strictly positive.")
        if self.parameter_source not in ("database", "ml", "manual"):
            raise ValueError(f"Unsupported parameter_source '{self.parameter_source}'.")
        if self.calibration_mode not in CALIBRATION_MODES:
            raise ValueError(
                f"Unsupported calibration_mode '{self.calibration_mode}'. Supported: {CALIBRATION_MODES}"
            )
        if self.ml_model_family is None:
            object.__setattr__(
                self,
                "ml_model_family",
                _AUTO_ROUTED_BASE_MODELS[self.calibration_mode],
            )
        elif self.ml_model_family not in SUPPORTED_MODEL_FAMILIES:
            raise ValueError(
                f"Unsupported ml_model_family '{self.ml_model_family}'. Supported: {SUPPORTED_MODEL_FAMILIES}"
            )
        if self.lab_project is not None:
            cleaned_proj = str(self.lab_project).strip()
            object.__setattr__(self, "lab_project", cleaned_proj or None)
        if self.kinetics_model not in ("auto", "linear", "passivation", "power_law"):
            raise ValueError(f"Unsupported kinetics_model '{self.kinetics_model}'.")
        if self.mc_draws < 50:
            raise ValueError("mc_draws must be at least 50.")

        resolved_regime, resolved_org, resolved_water = resolve_consistent_solvent_settings(
            self.solvent_regime,
            self.organic_solvent_type,
            self.water_content_wt_pct,
        )
        object.__setattr__(self, "solvent_regime", resolved_regime)
        object.__setattr__(self, "organic_solvent_type", resolved_org)
        object.__setattr__(self, "water_content_wt_pct", resolved_water)

    @property
    def base_solvent(self) -> str:
        """Canonical base solvent label passed to the ML feature extractor."""
        if self.solvent_regime in ("aqueous", "semi_aqueous"):
            return "DI water"
        return self.organic_solvent_type if self.organic_solvent_type != "None" else "non_aqueous"

    @property
    def solvent_subclass(self) -> SolventSubclass:
        """Chemical-family solvent subclass derived from solvent regime, pH, and co-solvent."""
        return classify_solvent_subclass(
            self.solvent_regime or "aqueous",
            pH=self.pH,
            organic_solvent_type=self.organic_solvent_type,
            base_solvent=self.base_solvent,
            acid_type=self.acid_type,
        )


@dataclass
class ResolvedMetalParameters:
    """Resolved electrochemical parameters and provenance for a single Metal."""

    metal: str
    source: ParameterSource
    parameters: PolarizationParameters
    Ecorr_std_V: float | None = None
    log10_icorr_std: float | None = None
    icorr_std_uA_cm2: float | None = None
    beta_a_std_mV_dec: float | None = None
    beta_c_std_mV_dec: float | None = None
    confidence_tier: str = "High"
    warnings: list[str] = field(default_factory=list)
    experimental_recommendations: list[str] = field(default_factory=list)
    lookup_result: ExposureLookupResult | None = None
    ml_prediction: MLParameterPrediction | None = None
    applicability: ApplicabilityDomainResult | None = None
    calibration_mode: CalibrationMode = "literature_only"
    lab_project: str | None = None
    residual_breakdown: ResidualBreakdown | None = None

    def to_parameter_table(self) -> pd.DataFrame:
        """Build a display-ready parameter summary table with ±1σ uncertainties."""
        if self.source == "ml":
            source_label = {
                "literature_only": "ML",
                "delta_learning": "ML (Delta-Learning)",
                "lab_only_gp": "ML (Lab-Only GP)",
            }[self.calibration_mode]
        else:
            source_label = {
                "database": "Database",
                "manual": "Manual",
            }[self.source]
        rows = [
            {
                "Parameter": "E_corr",
                "Value": float(self.parameters.Ecorr_V_SHE),
                "Std (±1σ)": self.Ecorr_std_V,
                "Unit": "V vs. SHE",
                "Source": source_label,
            },
            {
                "Parameter": "i_corr",
                "Value": float(self.parameters.icorr_uA_cm2),
                "Std (±1σ)": self.icorr_std_uA_cm2,
                "Log10 Std (±1σ dec)": self.log10_icorr_std,
                "Unit": "µA/cm²",
                "Source": source_label,
            },
            {
                "Parameter": "beta_a",
                "Value": float(self.parameters.beta_a_mV_dec),
                "Std (±1σ)": self.beta_a_std_mV_dec,
                "Unit": "mV/dec",
                "Source": source_label,
            },
            {
                "Parameter": "beta_c",
                "Value": float(self.parameters.beta_c_mV_dec),
                "Std (±1σ)": self.beta_c_std_mV_dec,
                "Unit": "mV/dec",
                "Source": source_label,
            },
        ]
        return pd.DataFrame(rows)


def _serialize_ad_summary(ad: ApplicabilityDomainResult | None) -> dict[str, Any] | None:
    if ad is None:
        return None
    return {
        "ad_mode": ad.ad_mode,
        "in_bounding_box": bool(ad.in_bounding_box),
        "lab_in_bounding_box": ad.lab_in_bounding_box,
        "lab_knn_distance": (
            None
            if (ad.lab_knn_distance is None or not math.isfinite(ad.lab_knn_distance))
            else float(ad.lab_knn_distance)
        ),
        "n_lab_samples": int(ad.n_lab_samples),
        "lab_anchoring_status": ad.lab_anchoring_status,
    }


@dataclass
class SingleMetalViewPayload:
    """Complete view payload for the Single Metal Analysis tab."""

    request: SimulationRequest
    resolved_parameters: ResolvedMetalParameters
    parameter_table: pd.DataFrame
    faraday_result: FaradayCorrosionResult | None
    polarization_curve: PolarizationCurve
    mc_propagation: MonteCarloPropagationResult | None
    kinetics_result: TimeResolvedKineticsResult | None
    mc_kinetics: MonteCarloKineticsResult | None
    initial_corrosion_rate_nm_min: float
    initial_corrosion_rate_mm_yr: float
    corrosion_depth_nm: float
    time_to_threshold_s: float
    thickness_loss_risk_score: float
    thickness_loss_risk_tier: str
    confidence_tier: str
    warnings: list[str]
    experimental_recommendations: list[str]
    polarization_figure: go.Figure
    kinetics_risk_figure: go.Figure
    depth_figure: go.Figure
    risk_figure: go.Figure

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the single-metal analysis."""
        return {
            "primary_metal": self.request.primary_metal,
            "parameter_source": self.request.parameter_source,
            "ml_model_family": (
                resolve_active_model_family(self.request, self.resolved_parameters)
                if self.request.parameter_source == "ml"
                else None
            ),
            "calibration_mode": (
                self.request.calibration_mode if self.request.parameter_source == "ml" else None
            ),
            "lab_project": (
                self.resolved_parameters.lab_project
                if self.request.parameter_source == "ml"
                else None
            ),
            "residual_breakdown": (
                self.resolved_parameters.residual_breakdown.to_dict()
                if self.resolved_parameters.residual_breakdown is not None
                else None
            ),
            "kinetics_model": self.kinetics_result.model if self.kinetics_result is not None else "unavailable",
            "pH": float(self.request.pH),
            "temperature_C": float(self.request.temperature_C),
            "solvent_regime": str(self.request.solvent_regime),
            "solvent_subclass": str(self.request.solvent_subclass),
            "organic_solvent_type": str(self.request.organic_solvent_type),
            "water_content_wt_pct": float(self.request.water_content_wt_pct),
            "immersion_time_s": float(self.request.immersion_time_s),
            "d_crit_nm": float(self.request.d_crit_nm),
            "Ecorr_V_SHE": float(self.resolved_parameters.parameters.Ecorr_V_SHE),
            "icorr_uA_cm2": float(self.resolved_parameters.parameters.icorr_uA_cm2),
            "beta_a_mV_dec": float(self.resolved_parameters.parameters.beta_a_mV_dec),
            "beta_c_mV_dec": float(self.resolved_parameters.parameters.beta_c_mV_dec),
            "initial_corrosion_rate_nm_min": (
                None if not math.isfinite(self.initial_corrosion_rate_nm_min) else float(self.initial_corrosion_rate_nm_min)
            ),
            "initial_corrosion_rate_mm_yr": (
                None if not math.isfinite(self.initial_corrosion_rate_mm_yr) else float(self.initial_corrosion_rate_mm_yr)
            ),
            "corrosion_depth_nm": (
                None if not math.isfinite(self.corrosion_depth_nm) else float(self.corrosion_depth_nm)
            ),
            "time_to_threshold_s": (
                None if not math.isfinite(self.time_to_threshold_s) else float(self.time_to_threshold_s)
            ),
            "thickness_loss_risk_score": (
                None if not math.isfinite(self.thickness_loss_risk_score) else float(self.thickness_loss_risk_score)
            ),
            "thickness_loss_risk_tier": str(self.thickness_loss_risk_tier),
            "confidence_tier": str(self.confidence_tier),
            "applicability_domain": _serialize_ad_summary(self.resolved_parameters.applicability),
            "warnings": list(self.warnings),
            "experimental_recommendations": list(self.experimental_recommendations),
        }


@dataclass
class GalvanicViewPayload:
    """Complete view payload for the Galvanic Analysis tab."""

    request: SimulationRequest
    primary_resolved: ResolvedMetalParameters
    secondary_resolved: ResolvedMetalParameters
    driving_force: GalvanicDrivingForceResult
    coupling_result: GalvanicCouplingResult
    mc_galvanic: MonteCarloGalvanicResult | None
    uncoupled_kinetics: TimeResolvedKineticsResult | None
    coupled_kinetics: TimeResolvedKineticsResult | None
    mc_coupled_kinetics: MonteCarloKineticsResult | None
    active_metal: str
    noble_metal: str
    galvanic_driving_force_risk: str
    mixed_potential_V_SHE: float
    galvanic_current_density_uA_cm2: float
    acceleration_factor: float
    uncoupled_cr0_nm_min: float
    accelerated_cr0_nm_min: float
    uncoupled_depth_nm: float
    coupled_depth_nm: float
    uncoupled_time_to_threshold_s: float
    coupled_time_to_threshold_s: float
    uncoupled_risk_score: float
    coupled_risk_score: float
    uncoupled_risk_tier: str
    coupled_risk_tier: str
    confidence_tier: str
    warnings: list[str]
    experimental_recommendations: list[str]
    galvanic_polarization_figure: go.Figure
    comparative_kinetics_figure: go.Figure
    galvanic_depth_figure: go.Figure
    galvanic_risk_figure: go.Figure

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the galvanic coupling analysis."""
        return {
            "primary_metal": self.request.primary_metal,
            "secondary_metal": self.request.secondary_metal,
            "active_metal": self.active_metal,
            "noble_metal": self.noble_metal,
            "cathode_to_anode_area_ratio": float(self.request.cathode_to_anode_area_ratio),
            "solvent_regime": str(self.request.solvent_regime),
            "solvent_subclass": str(self.request.solvent_subclass),
            "organic_solvent_type": str(self.request.organic_solvent_type),
            "water_content_wt_pct": float(self.request.water_content_wt_pct),
            "parameter_source": self.request.parameter_source,
            "ml_model_family": (
                resolve_active_model_family(self.request, self.primary_resolved)
                if self.request.parameter_source == "ml"
                else None
            ),
            "calibration_mode": (
                self.request.calibration_mode if self.request.parameter_source == "ml" else None
            ),
            "lab_project": (
                self.primary_resolved.lab_project
                if self.request.parameter_source == "ml"
                else None
            ),
            "primary_residual_breakdown": (
                self.primary_resolved.residual_breakdown.to_dict()
                if self.primary_resolved.residual_breakdown is not None
                else None
            ),
            "secondary_residual_breakdown": (
                self.secondary_resolved.residual_breakdown.to_dict()
                if self.secondary_resolved.residual_breakdown is not None
                else None
            ),
            "kinetics_model": self.coupled_kinetics.model if self.coupled_kinetics is not None else "unavailable",
            "delta_Ecorr_V": float(self.driving_force.delta_Ecorr_V),
            "galvanic_driving_force_risk": str(self.galvanic_driving_force_risk),
            "mixed_potential_V_SHE": float(self.mixed_potential_V_SHE),
            "galvanic_current_density_uA_cm2": float(self.galvanic_current_density_uA_cm2),
            "acceleration_factor": float(self.acceleration_factor),
            "uncoupled_cr0_nm_min": (
                None if not math.isfinite(self.uncoupled_cr0_nm_min) else float(self.uncoupled_cr0_nm_min)
            ),
            "accelerated_cr0_nm_min": (
                None if not math.isfinite(self.accelerated_cr0_nm_min) else float(self.accelerated_cr0_nm_min)
            ),
            "uncoupled_depth_nm": (
                None if not math.isfinite(self.uncoupled_depth_nm) else float(self.uncoupled_depth_nm)
            ),
            "coupled_depth_nm": (
                None if not math.isfinite(self.coupled_depth_nm) else float(self.coupled_depth_nm)
            ),
            "uncoupled_time_to_threshold_s": (
                None
                if not math.isfinite(self.uncoupled_time_to_threshold_s)
                else float(self.uncoupled_time_to_threshold_s)
            ),
            "coupled_time_to_threshold_s": (
                None
                if not math.isfinite(self.coupled_time_to_threshold_s)
                else float(self.coupled_time_to_threshold_s)
            ),
            "uncoupled_risk_score": (
                None if not math.isfinite(self.uncoupled_risk_score) else float(self.uncoupled_risk_score)
            ),
            "coupled_risk_score": (
                None if not math.isfinite(self.coupled_risk_score) else float(self.coupled_risk_score)
            ),
            "uncoupled_risk_tier": str(self.uncoupled_risk_tier),
            "coupled_risk_tier": str(self.coupled_risk_tier),
            "confidence_tier": str(self.confidence_tier),
            "primary_applicability_domain": _serialize_ad_summary(self.primary_resolved.applicability),
            "secondary_applicability_domain": _serialize_ad_summary(self.secondary_resolved.applicability),
            "warnings": list(self.warnings),
            "experimental_recommendations": list(self.experimental_recommendations),
        }


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _build_time_grid(immersion_time_s: float, num_points: int = 121) -> np.ndarray:
    return np.linspace(0.0, float(immersion_time_s), num_points)


def build_single_metal_polarization_figure(
    curve: PolarizationCurve,
    *,
    mc_result: MonteCarloPropagationResult | None = None,
    title: str | None = None,
) -> go.Figure:
    """Build an interactive Plotly Tafel Polarization Curve figure with optional 5th–95th %ile band."""
    fig = go.Figure()
    metal = curve.parameters.metal or "Metal"

    # Fill the exact E == E_corr NaN point with a finite cusp floor so the V-notch renders continuously
    net_log10 = np.array(curve.log10_abs_net_uA_cm2, dtype=float, copy=True)
    finite_mask = np.isfinite(net_log10)
    if np.any(finite_mask):
        cusp_floor = float(np.min(net_log10[finite_mask])) - 0.75
        net_log10[~finite_mask] = cusp_floor
        y_min = cusp_floor - 0.4
        y_max = float(np.max(net_log10[finite_mask])) + 1.2
    else:
        y_min, y_max = -2.0, 4.0

    if mc_result is not None and mc_result.polarization_band is not None:
        band = mc_result.polarization_band
        e_band = curve.parameters.Ecorr_V_SHE + band.overpotential_V
        p05 = np.maximum(band.eta_log10_abs_net_p05, cusp_floor)
        p25 = np.maximum(band.eta_log10_abs_net_p25, cusp_floor)
        p75 = np.maximum(band.eta_log10_abs_net_p75, cusp_floor)
        p95 = np.maximum(band.eta_log10_abs_net_p95, cusp_floor)
        fig.add_trace(
            go.Scatter(
                x=e_band,
                y=p95,
                mode="lines",
                line={"color": "rgba(31, 119, 180, 0.30)", "width": 0.8},
                showlegend=False,
                hoverinfo="skip",
                name="p95",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=e_band,
                y=p05,
                mode="lines",
                line={"color": "rgba(31, 119, 180, 0.30)", "width": 0.8},
                fill="tonexty",
                fillcolor="rgba(31, 119, 180, 0.14)",
                name="Net |i_net| 90% Band (p05–p95)",
                hovertemplate="E: %{x:.3f} V vs. SHE<br>log10|i| (p05): %{y:.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=e_band,
                y=p75,
                mode="lines",
                line={"color": "rgba(31, 119, 180, 0.45)", "width": 0.9},
                showlegend=False,
                hoverinfo="skip",
                name="p75",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=e_band,
                y=p25,
                mode="lines",
                line={"color": "rgba(31, 119, 180, 0.45)", "width": 0.9},
                fill="tonexty",
                fillcolor="rgba(31, 119, 180, 0.28)",
                name="Net |i_net| 50% Band (p25–p75)",
                hovertemplate="E: %{x:.3f} V vs. SHE<br>log10|i| (p25): %{y:.2f}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=curve.potential_V_SHE,
            y=curve.log10_anodic_uA_cm2,
            mode="lines",
            line={"color": "#d62728", "dash": "dash", "width": 1.6},
            name="Anodic Branch (i_a)",
            hovertemplate="E: %{x:.3f} V vs. SHE<br>log10(i_a): %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve.potential_V_SHE,
            y=curve.log10_cathodic_uA_cm2,
            mode="lines",
            line={"color": "#2ca02c", "dash": "dot", "width": 1.6},
            name="Cathodic Branch (i_c)",
            hovertemplate="E: %{x:.3f} V vs. SHE<br>log10(i_c): %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve.potential_V_SHE,
            y=net_log10,
            mode="lines",
            line={"color": "#1f77b4", "width": 2.6},
            name=f"{metal} Net |i_net|",
            hovertemplate="E: %{x:.3f} V vs. SHE<br>log10|i_net|: %{y:.2f}<extra></extra>",
        )
    )
    x_ecorr = float(curve.parameters.Ecorr_V_SHE)
    y_icorr = float(math.log10(curve.parameters.icorr_uA_cm2))
    marker_kwargs: dict[str, Any] = {
        "x": [x_ecorr],
        "y": [y_icorr],
        "mode": "markers",
        "marker": {"color": "#111111", "size": 9, "symbol": "diamond"},
        "name": f"(E_corr, i_corr) = ({x_ecorr:.3f} V, {curve.parameters.icorr_uA_cm2:.2g} µA/cm²)",
    }
    if mc_result is not None and mc_result.polarization_band is not None:
        band = mc_result.polarization_band
        marker_kwargs["error_x"] = {
            "type": "data",
            "symmetric": False,
            "array": [max(0.0, float(band.ecorr_p95_V_SHE) - x_ecorr)],
            "arrayminus": [max(0.0, x_ecorr - float(band.ecorr_p05_V_SHE))],
            "color": "#111111",
            "thickness": 1.5,
            "width": 4,
        }
        marker_kwargs["error_y"] = {
            "type": "data",
            "symmetric": False,
            "array": [max(0.0, float(band.log10_icorr_p95) - y_icorr)],
            "arrayminus": [max(0.0, y_icorr - float(band.log10_icorr_p05))],
            "color": "#111111",
            "thickness": 1.5,
            "width": 4,
        }
    fig.add_trace(go.Scatter(**marker_kwargs))

    fig.update_layout(
        title=title or f"{metal} Polarization Curve (Tafel Plot)",
        xaxis_title="Potential E (V vs. SHE)",
        yaxis_title="log10(|i| / µA·cm⁻²)",
        yaxis={"range": [y_min, y_max]},
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        margin={"l": 60, "r": 25, "t": 70, "b": 55},
        height=440,
    )
    return fig


def _add_dual_band_traces(
    fig: go.Figure,
    times: np.ndarray,
    p05: np.ndarray,
    p25: np.ndarray,
    p75: np.ndarray,
    p95: np.ndarray,
    *,
    label_prefix: str,
    fillcolor_90: str,
    fillcolor_50: str,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add layered 90% (p05-p95) and 50% (p25-p75) uncertainty band traces."""
    loc_kwargs = {"row": row, "col": col} if row is not None and col is not None else {}
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([times, times[::-1]]),
            y=np.concatenate([p95, p05[::-1]]),
            fill="toself",
            fillcolor=fillcolor_90,
            line={"width": 0},
            name=f"{label_prefix} 90% Band (p05–p95)",
            hoverinfo="skip",
        ),
        **loc_kwargs,
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([times, times[::-1]]),
            y=np.concatenate([p75, p25[::-1]]),
            fill="toself",
            fillcolor=fillcolor_50,
            line={"width": 0},
            name=f"{label_prefix} 50% Band (p25–p75)",
            hoverinfo="skip",
        ),
        **loc_kwargs,
    )


def build_single_metal_depth_figure(
    kinetics: TimeResolvedKineticsResult,
    *,
    mc_kinetics: MonteCarloKineticsResult | None = None,
    title: str | None = None,
) -> go.Figure:
    """Build standalone interactive Plotly Corrosion Depth figure with dual 50%/90% bands."""
    fig = go.Figure()
    times = kinetics.times_s
    metal = kinetics.metal or "Metal"

    if mc_kinetics is not None:
        _add_dual_band_traces(
            fig,
            times,
            mc_kinetics.depth_p05_nm,
            mc_kinetics.depth_p25_nm,
            mc_kinetics.depth_p75_nm,
            mc_kinetics.depth_p95_nm,
            label_prefix="Depth",
            fillcolor_90="rgba(31, 119, 180, 0.14)",
            fillcolor_50="rgba(31, 119, 180, 0.28)",
        )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=kinetics.depth_nm,
            mode="lines",
            line={"color": "#1f77b4", "dash": "solid", "width": 2.6},
            name=f"{metal} Depth ({kinetics.model})",
            hovertemplate="t: %{x:.1f} s<br>Depth: %{y:.3f} nm<extra></extra>",
        )
    )
    fig.add_hline(
        y=kinetics.d_crit_nm,
        line_dash="dash",
        line_color="#475569",
        annotation_text=f"d_crit = {kinetics.d_crit_nm:.2f} nm",
        annotation_position="top left",
    )
    fig.update_layout(
        title=title or f"{metal} Corrosion Depth vs. Immersion Time ({kinetics.model})",
        xaxis_title="Immersion Time (s)",
        yaxis_title="Corrosion Depth (nm)",
        yaxis={"rangemode": "tozero"},
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        margin={"l": 60, "r": 25, "t": 75, "b": 55},
        height=360,
    )
    return fig


def build_single_metal_risk_figure(
    kinetics: TimeResolvedKineticsResult,
    *,
    mc_kinetics: MonteCarloKineticsResult | None = None,
    title: str | None = None,
) -> go.Figure:
    """Build standalone interactive Plotly Thickness Loss Risk figure with dual 50%/90% bands."""
    fig = go.Figure()
    times = kinetics.times_s
    metal = kinetics.metal or "Metal"

    if mc_kinetics is not None:
        _add_dual_band_traces(
            fig,
            times,
            mc_kinetics.risk_p05,
            mc_kinetics.risk_p25,
            mc_kinetics.risk_p75,
            mc_kinetics.risk_p95,
            label_prefix="Risk",
            fillcolor_90="rgba(217, 95, 2, 0.14)",
            fillcolor_50="rgba(217, 95, 2, 0.28)",
        )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=kinetics.risk_score,
            mode="lines",
            line={"color": "#d95f02", "dash": "dashdot", "width": 2.6},
            name="Thickness Loss Risk R(t)",
            hovertemplate="t: %{x:.1f} s<br>Risk R(t): %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="#94a3b8",
        annotation_text="R(d_crit) = 0.50",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title=title or f"{metal} Thickness Loss Risk vs. Immersion Time ({kinetics.model})",
        xaxis_title="Immersion Time (s)",
        yaxis_title="Thickness Loss Risk R(t)",
        yaxis={"range": [0.0, 1.05]},
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        margin={"l": 60, "r": 25, "t": 75, "b": 55},
        height=360,
    )
    return fig


def build_single_metal_kinetics_figure(
    kinetics: TimeResolvedKineticsResult,
    *,
    mc_kinetics: MonteCarloKineticsResult | None = None,
    title: str | None = None,
) -> go.Figure:
    """Build dual-panel interactive Plotly Corrosion Depth and Thickness Loss Risk trajectories."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Corrosion Depth vs. Immersion Time", "Thickness Loss Risk vs. Immersion Time"),
        horizontal_spacing=0.12,
    )

    times = kinetics.times_s
    metal = kinetics.metal or "Metal"

    if mc_kinetics is not None:
        _add_dual_band_traces(
            fig,
            times,
            mc_kinetics.depth_p05_nm,
            mc_kinetics.depth_p25_nm,
            mc_kinetics.depth_p75_nm,
            mc_kinetics.depth_p95_nm,
            label_prefix="Depth",
            fillcolor_90="rgba(31, 119, 180, 0.14)",
            fillcolor_50="rgba(31, 119, 180, 0.28)",
            row=1,
            col=1,
        )
        _add_dual_band_traces(
            fig,
            times,
            mc_kinetics.risk_p05,
            mc_kinetics.risk_p25,
            mc_kinetics.risk_p75,
            mc_kinetics.risk_p95,
            label_prefix="Risk",
            fillcolor_90="rgba(217, 95, 2, 0.14)",
            fillcolor_50="rgba(217, 95, 2, 0.28)",
            row=1,
            col=2,
        )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=kinetics.depth_nm,
            mode="lines",
            line={"color": "#1f77b4", "dash": "solid", "width": 2.6},
            name=f"{metal} Depth ({kinetics.model})",
            hovertemplate="t: %{x:.1f} s<br>Depth: %{y:.3f} nm<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=kinetics.d_crit_nm,
        line_dash="dash",
        line_color="#475569",
        annotation_text=f"d_crit = {kinetics.d_crit_nm:.2f} nm",
        annotation_position="top left",
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=kinetics.risk_score,
            mode="lines",
            line={"color": "#d95f02", "dash": "dashdot", "width": 2.6},
            name="Thickness Loss Risk R(t)",
            hovertemplate="t: %{x:.1f} s<br>Risk R(t): %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="#94a3b8",
        annotation_text="R(d_crit) = 0.50",
        annotation_position="bottom right",
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Immersion Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Corrosion Depth (nm)", rangemode="tozero", row=1, col=1)
    fig.update_xaxes(title_text="Immersion Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Thickness Loss Risk R(t)", range=[0.0, 1.05], row=1, col=2)
    fig.update_layout(
        title=title or f"{metal} Time-Resolved Kinetics ({kinetics.model})",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.06, "xanchor": "left", "x": 0.0},
        margin={"l": 60, "r": 25, "t": 85, "b": 55},
        height=420,
    )
    return fig


def build_galvanic_polarization_figure(
    coupling: GalvanicCouplingResult,
    *,
    mc_galvanic: MonteCarloGalvanicResult | None = None,
    title: str | None = None,
    potential_padding_V: float = 0.15,
    num_points: int = 301,
) -> go.Figure:
    """Build interactive overlaid bimetallic Polarization Curves with Mixed Potential (E_galv) line."""
    e_min = min(coupling.active_parameters.Ecorr_V_SHE, coupling.noble_parameters.Ecorr_V_SHE) - potential_padding_V
    e_max = max(coupling.active_parameters.Ecorr_V_SHE, coupling.noble_parameters.Ecorr_V_SHE) + potential_padding_V
    potentials = np.linspace(e_min, e_max, num_points)

    active_curve = generate_polarization_curve(coupling.active_parameters, potentials_V_SHE=potentials)
    noble_curve = generate_polarization_curve(coupling.noble_parameters, potentials_V_SHE=potentials)
    area_ratio = coupling.cathode_to_anode_area_ratio
    noble_scaled_log10 = np.log10(np.maximum(noble_curve.abs_net_current_uA_cm2 * area_ratio, 1e-12))

    fig = go.Figure()

    if mc_galvanic is not None:
        e_p05 = float(mc_galvanic.mixed_potential_summary["p05"])
        e_p95 = float(mc_galvanic.mixed_potential_summary["p95"])
        fig.add_vrect(
            x0=e_p05,
            x1=e_p95,
            fillcolor="rgba(148, 103, 189, 0.14)",
            line_width=0,
            annotation_text="E_galv 90% CI",
            annotation_position="top right",
        )

    fig.add_trace(
        go.Scatter(
            x=potentials,
            y=active_curve.log10_abs_net_uA_cm2,
            mode="lines",
            line={"color": "#d62728", "width": 2.6},
            name=f"Active Metal: {coupling.active_metal} |i_net|",
            hovertemplate="E: %{x:.3f} V vs. SHE<br>log10|i|: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=potentials,
            y=noble_scaled_log10,
            mode="lines",
            line={"color": "#1f77b4", "width": 2.6},
            name=f"Noble Metal: {coupling.noble_metal} ({area_ratio:g}× |i_net|)",
            hovertemplate="E: %{x:.3f} V vs. SHE<br>log10(A_ratio·|i|): %{y:.2f}<extra></extra>",
        )
    )
    if not math.isclose(area_ratio, 1.0, rel_tol=1e-6):
        fig.add_trace(
            go.Scatter(
                x=potentials,
                y=noble_curve.log10_abs_net_uA_cm2,
                mode="lines",
                line={"color": "#1f77b4", "dash": "dot", "width": 1.5},
                name=f"Noble Metal: {coupling.noble_metal} (1× unscaled)",
                hovertemplate="E: %{x:.3f} V vs. SHE<br>log10|i|: %{y:.2f}<extra></extra>",
            )
        )

    fig.add_vline(
        x=coupling.mixed_potential_V_SHE,
        line_dash="dash",
        line_color="#9467bd",
        line_width=2.0,
        annotation_text=f"E_galv = {coupling.mixed_potential_V_SHE:.3f} V vs. SHE",
        annotation_position="top left",
    )

    fig.add_trace(
        go.Scatter(
            x=[coupling.mixed_potential_V_SHE],
            y=[math.log10(max(coupling.galvanic_current_density_uA_cm2, 1e-12))],
            mode="markers",
            marker={"color": "#9467bd", "size": 11, "symbol": "star"},
            name=(
                f"Galvanic Operating Point: i_galv = {coupling.galvanic_current_density_uA_cm2:.2g} µA/cm² "
                f"(AF = {coupling.acceleration_factor:.2f}×)"
            ),
        )
    )

    fig.update_layout(
        title=title
        or f"Galvanic Polarization Overlay: {coupling.active_metal} (Active) vs. {coupling.noble_metal} (Noble)",
        xaxis_title="Potential E (V vs. SHE)",
        yaxis_title="log10(|i| / µA·cm⁻²)",
        xaxis={"range": [float(e_min), float(e_max)]},
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        margin={"l": 60, "r": 25, "t": 75, "b": 55},
        height=460,
    )
    return fig


def build_galvanic_depth_comparison_figure(
    uncoupled_kinetics: TimeResolvedKineticsResult,
    coupled_kinetics: TimeResolvedKineticsResult,
    *,
    mc_coupled_kinetics: MonteCarloKineticsResult | None = None,
    active_metal: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Build standalone comparative uncoupled vs. coupled Corrosion Depth figure."""
    metal = active_metal or coupled_kinetics.metal or "Active Metal"
    times = coupled_kinetics.times_s
    fig = go.Figure()

    if mc_coupled_kinetics is not None:
        _add_dual_band_traces(
            fig,
            times,
            mc_coupled_kinetics.depth_p05_nm,
            mc_coupled_kinetics.depth_p25_nm,
            mc_coupled_kinetics.depth_p75_nm,
            mc_coupled_kinetics.depth_p95_nm,
            label_prefix="Coupled Depth",
            fillcolor_90="rgba(31, 119, 180, 0.14)",
            fillcolor_50="rgba(31, 119, 180, 0.28)",
        )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=uncoupled_kinetics.depth_nm,
            mode="lines",
            line={"color": "#64748b", "dash": "dash", "width": 2.2},
            name=f"{metal} Uncoupled Depth",
            hovertemplate="t: %{x:.1f} s<br>Uncoupled Depth: %{y:.3f} nm<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=coupled_kinetics.depth_nm,
            mode="lines",
            line={"color": "#1f77b4", "dash": "solid", "width": 2.8},
            name=f"{metal} Coupled Depth",
            hovertemplate="t: %{x:.1f} s<br>Coupled Depth: %{y:.3f} nm<extra></extra>",
        )
    )
    fig.add_hline(
        y=coupled_kinetics.d_crit_nm,
        line_dash="dash",
        line_color="#475569",
        annotation_text=f"d_crit = {coupled_kinetics.d_crit_nm:.2f} nm",
        annotation_position="top left",
    )
    fig.update_layout(
        title=title or f"{metal} Corrosion Depth: Uncoupled vs. Coupled ({coupled_kinetics.model})",
        xaxis_title="Immersion Time (s)",
        yaxis_title="Corrosion Depth (nm)",
        yaxis={"rangemode": "tozero"},
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        margin={"l": 60, "r": 25, "t": 75, "b": 55},
        height=360,
    )
    return fig


def build_galvanic_risk_comparison_figure(
    uncoupled_kinetics: TimeResolvedKineticsResult,
    coupled_kinetics: TimeResolvedKineticsResult,
    *,
    mc_coupled_kinetics: MonteCarloKineticsResult | None = None,
    active_metal: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Build standalone comparative uncoupled vs. coupled Thickness Loss Risk figure."""
    metal = active_metal or coupled_kinetics.metal or "Active Metal"
    times = coupled_kinetics.times_s
    fig = go.Figure()

    if mc_coupled_kinetics is not None:
        _add_dual_band_traces(
            fig,
            times,
            mc_coupled_kinetics.risk_p05,
            mc_coupled_kinetics.risk_p25,
            mc_coupled_kinetics.risk_p75,
            mc_coupled_kinetics.risk_p95,
            label_prefix="Coupled Risk",
            fillcolor_90="rgba(217, 95, 2, 0.14)",
            fillcolor_50="rgba(217, 95, 2, 0.28)",
        )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=uncoupled_kinetics.risk_score,
            mode="lines",
            line={"color": "#94a3b8", "dash": "dot", "width": 2.2},
            name=f"{metal} Uncoupled Risk",
            hovertemplate="t: %{x:.1f} s<br>Uncoupled Risk: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=coupled_kinetics.risk_score,
            mode="lines",
            line={"color": "#d95f02", "dash": "dashdot", "width": 2.8},
            name=f"{metal} Coupled Risk",
            hovertemplate="t: %{x:.1f} s<br>Coupled Risk: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="#94a3b8",
        annotation_text="R(d_crit) = 0.50",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title=title or f"{metal} Thickness Loss Risk: Uncoupled vs. Coupled ({coupled_kinetics.model})",
        xaxis_title="Immersion Time (s)",
        yaxis_title="Thickness Loss Risk R(t)",
        yaxis={"range": [0.0, 1.05]},
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        margin={"l": 60, "r": 25, "t": 75, "b": 55},
        height=360,
    )
    return fig


def build_galvanic_kinetics_comparison_figure(
    uncoupled_kinetics: TimeResolvedKineticsResult,
    coupled_kinetics: TimeResolvedKineticsResult,
    *,
    mc_coupled_kinetics: MonteCarloKineticsResult | None = None,
    active_metal: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Build comparative uncoupled vs. coupled Corrosion Depth and Thickness Loss Risk trajectories."""
    metal = active_metal or coupled_kinetics.metal or "Active Metal"
    times = coupled_kinetics.times_s

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"{metal} Corrosion Depth: Uncoupled vs. Coupled",
            f"{metal} Thickness Loss Risk: Uncoupled vs. Coupled",
        ),
        horizontal_spacing=0.12,
    )

    if mc_coupled_kinetics is not None:
        _add_dual_band_traces(
            fig,
            times,
            mc_coupled_kinetics.depth_p05_nm,
            mc_coupled_kinetics.depth_p25_nm,
            mc_coupled_kinetics.depth_p75_nm,
            mc_coupled_kinetics.depth_p95_nm,
            label_prefix="Coupled Depth",
            fillcolor_90="rgba(31, 119, 180, 0.14)",
            fillcolor_50="rgba(31, 119, 180, 0.28)",
            row=1,
            col=1,
        )
        _add_dual_band_traces(
            fig,
            times,
            mc_coupled_kinetics.risk_p05,
            mc_coupled_kinetics.risk_p25,
            mc_coupled_kinetics.risk_p75,
            mc_coupled_kinetics.risk_p95,
            label_prefix="Coupled Risk",
            fillcolor_90="rgba(217, 95, 2, 0.14)",
            fillcolor_50="rgba(217, 95, 2, 0.28)",
            row=1,
            col=2,
        )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=uncoupled_kinetics.depth_nm,
            mode="lines",
            line={"color": "#64748b", "dash": "dash", "width": 2.2},
            name=f"{metal} Uncoupled Depth",
            hovertemplate="t: %{x:.1f} s<br>Uncoupled Depth: %{y:.3f} nm<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=coupled_kinetics.depth_nm,
            mode="lines",
            line={"color": "#1f77b4", "dash": "solid", "width": 2.8},
            name=f"{metal} Coupled Depth",
            hovertemplate="t: %{x:.1f} s<br>Coupled Depth: %{y:.3f} nm<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=coupled_kinetics.d_crit_nm,
        line_dash="dash",
        line_color="#475569",
        annotation_text=f"d_crit = {coupled_kinetics.d_crit_nm:.2f} nm",
        annotation_position="top left",
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=uncoupled_kinetics.risk_score,
            mode="lines",
            line={"color": "#94a3b8", "dash": "dot", "width": 2.2},
            name=f"{metal} Uncoupled Risk",
            hovertemplate="t: %{x:.1f} s<br>Uncoupled Risk: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=coupled_kinetics.risk_score,
            mode="lines",
            line={"color": "#d95f02", "dash": "dashdot", "width": 2.8},
            name=f"{metal} Coupled Risk",
            hovertemplate="t: %{x:.1f} s<br>Coupled Risk: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="#94a3b8",
        annotation_text="R(d_crit) = 0.50",
        annotation_position="bottom right",
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Immersion Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Corrosion Depth (nm)", rangemode="tozero", row=1, col=1)
    fig.update_xaxes(title_text="Immersion Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Thickness Loss Risk R(t)", range=[0.0, 1.05], row=1, col=2)
    fig.update_layout(
        title=title or f"Uncoupled vs. Galvanically Coupled Kinetics ({coupled_kinetics.model})",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.06, "xanchor": "left", "x": 0.0},
        margin={"l": 60, "r": 25, "t": 85, "b": 55},
        height=430,
    )
    return fig


class CorrosionPlatformController:
    """High-level application controller backing the Streamlit UI and reporting workflows."""

    def __init__(
        self,
        database: HarmonizedDatabase,
        predictor: HybridMLPredictor | None = None,
        *,
        raw_dir: str | Path | None = None,
        artifact_dir: str | Path | None = None,
        artifact_status: ArtifactStatus = "live",
        artifact_warning: str | None = None,
        lab_dir: str | Path | None = None,
        raw_fingerprint: dict[str, Any] | None = None,
        lab_projects: dict[str, LabProjectDataset] | None = None,
    ) -> None:
        self.database = database
        self.raw_dir: Path | None = Path(raw_dir) if raw_dir is not None else None
        self.artifact_dir: Path | None = Path(artifact_dir) if artifact_dir is not None else None
        self.artifact_status: ArtifactStatus = artifact_status
        self.artifact_warning: str | None = artifact_warning
        self.lab_dir: Path | None = Path(lab_dir) if lab_dir is not None else None
        if raw_fingerprint is not None:
            self.raw_fingerprint: dict[str, Any] | None = raw_fingerprint
        elif self.raw_dir is not None and iter_raw_csv_files(self.raw_dir):
            self.raw_fingerprint = HarmonizedDatabase.compute_raw_directory_fingerprint(self.raw_dir)
        else:
            self.raw_fingerprint = None

        if lab_projects is not None:
            self.lab_projects: dict[str, LabProjectDataset] = dict(lab_projects)
        elif self.lab_dir is not None:
            self.lab_projects = discover_and_load_lab_projects(self.lab_dir)
        else:
            self.lab_projects = {}
        self.lab_fingerprints: dict[str, dict[str, Any]] = {
            name: ds.fingerprint for name, ds in self.lab_projects.items()
        }
        self.lab_status: LabReloadStatus = "fresh" if self.lab_projects else "none"
        self.lab_status_message: str | None = (
            f"Loaded {len(self.lab_projects)} lab project(s)."
            if self.lab_projects
            else None
        )
        self._lab_calibration_cache: dict[str, Any] = {}

        if predictor is None:
            self.predictor = self._load_or_fit_predictor("rf_gp_ensemble")
        else:
            self.predictor = predictor
        self._predictors: dict[ModelFamily, HybridMLPredictor] = {
            self.predictor.model_family: self.predictor
        }

    def _load_or_fit_predictor(self, model_family: ModelFamily) -> HybridMLPredictor:
        sample_metal = (
            self.database.available_metals[0]
            if self.database.available_metals
            else CORE_METALS[0]
        )
        if self.artifact_dir is not None:
            prebuilt_path = self.artifact_dir / "models" / f"{model_family}.joblib"
            if prebuilt_path.exists():
                try:
                    loaded = HybridMLPredictor.load(prebuilt_path)
                    if loaded.model_family != model_family:
                        raise ValueError(
                            f"Model family mismatch in {prebuilt_path.name}: expected {model_family!r}, got {loaded.model_family!r}"
                        )
                    # Smoke-test prediction to catch cross-OS / scikit-learn pickle incompatibilities early
                    loaded.predict(MLFeatureQuery(metal=sample_metal))
                    return loaded
                except Exception as exc:
                    heal_note = (
                        f"Prebuilt model '{prebuilt_path.name}' could not be loaded ({exc}); "
                        "automatically retrained on the current database."
                    )
                    self.artifact_warning = (
                        f"{self.artifact_warning} {heal_note}".strip()
                        if self.artifact_warning
                        else heal_note
                    )

        predictor = HybridMLPredictor(model_family=model_family, random_state=42)
        predictor.fit(self.database)
        if self.artifact_dir is not None:
            try:
                models_dir = self.artifact_dir / "models"
                models_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = models_dir / f".tmp_{model_family}.joblib"
                target_path = models_dir / f"{model_family}.joblib"
                predictor.save(tmp_path)
                tmp_path.replace(target_path)
            except OSError:
                pass
        return predictor

    def get_predictor(self, model_family: ModelFamily | None = None) -> HybridMLPredictor:
        """Return a fitted predictor for ``model_family``, loading prebuilt artifact or fitting on first use."""
        target_family = model_family or self.predictor.model_family
        if target_family not in SUPPORTED_MODEL_FAMILIES:
            raise ValueError(
                f"Unsupported model_family '{target_family}'. Supported: {SUPPORTED_MODEL_FAMILIES}"
            )
        cached = self._predictors.get(target_family)
        if cached is not None:
            return cached
        predictor = self._load_or_fit_predictor(target_family)
        self._predictors[target_family] = predictor
        return predictor

    def _get_cached_or_prebuilt_predictor(self, model_family: ModelFamily) -> HybridMLPredictor | None:
        """Return an in-memory or prebuilt on-disk predictor without ever triggering on-the-fly training."""
        cached = self._predictors.get(model_family)
        if cached is not None:
            return cached
        if self.artifact_dir is None:
            return None
        prebuilt_path = self.artifact_dir / "models" / f"{model_family}.joblib"
        if not prebuilt_path.exists():
            return None
        try:
            loaded = HybridMLPredictor.load(prebuilt_path)
            if loaded.model_family != model_family:
                return None
            self._predictors[model_family] = loaded
            return loaded
        except Exception:
            return None

    @classmethod
    def from_default_data(
        cls,
        raw_dir: str | Path | None = None,
        *,
        processed_dir: str | Path | None = None,
        lab_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        model_family: ModelFamily = "rf_gp_ensemble",
        auto_rebuild: bool = True,
        rebuild_models: Sequence[str] = ("all",),
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> CorrosionPlatformController:
        """Load prebuilt artifacts when fresh, auto-rebuild all models when raw CSVs change, or fall back cleanly."""
        from corrosion_simulator.build_artifacts import (
            build_processed_artifacts,
            check_artifacts_freshness,
        )

        raw_path, proc_path, lab_path = resolve_data_dirs(
            data_dir,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            lab_dir=lab_dir,
        )
        has_raw_csvs = bool(iter_raw_csv_files(raw_path))
        raw_fp = (
            HarmonizedDatabase.compute_raw_directory_fingerprint(raw_path)
            if has_raw_csvs
            else None
        )
        loaded_lab_projects = discover_and_load_lab_projects(lab_path)
        status, warning_msg = check_artifacts_freshness(raw_path, proc_path)

        if status == "fresh" or (status == "stale" and not auto_rebuild):
            try:
                database = HarmonizedDatabase.from_processed_directory(proc_path)
                ctrl = cls(
                    database=database,
                    predictor=None,
                    raw_dir=raw_path,
                    artifact_dir=proc_path,
                    artifact_status=status,
                    artifact_warning=warning_msg,
                    lab_dir=lab_path,
                    raw_fingerprint=raw_fp,
                    lab_projects=loaded_lab_projects,
                )
                if model_family != ctrl.predictor.model_family:
                    ctrl.predictor = ctrl.get_predictor(model_family)
                return ctrl
            except Exception as exc:
                if not has_raw_csvs:
                    raise
                warning_msg = f"Failed to read cached artifacts in '{proc_path}' ({exc}); rebuilding from '{raw_path}'."

        # Raw CSVs exist and artifacts are missing/stale -> rebuild full artifact & model suite when auto_rebuild=True
        if has_raw_csvs and auto_rebuild:
            try:
                summary = build_processed_artifacts(
                    raw_dir=raw_path,
                    output_dir=proc_path,
                    models=rebuild_models,
                    export_csv=False,
                    progress_callback=progress_callback,
                )
                database = HarmonizedDatabase.from_processed_directory(proc_path)
                rebuilt_msg = (
                    f"Detected updated CSV data in '{raw_path}' — rebuilt database "
                    f"({summary.n_records} records across {summary.n_materials} materials) "
                    f"and trained {len(summary.trained_models)} ML model(s)."
                )
                ctrl = cls(
                    database=database,
                    predictor=None,
                    raw_dir=raw_path,
                    artifact_dir=proc_path,
                    artifact_status="rebuilt",
                    artifact_warning=rebuilt_msg,
                    lab_dir=lab_path,
                    raw_fingerprint=raw_fp,
                    lab_projects=loaded_lab_projects,
                )
                if model_family != ctrl.predictor.model_family:
                    ctrl.predictor = ctrl.get_predictor(model_family)
                return ctrl
            except OSError as exc:
                warning_msg = (
                    f"Could not write prebuilt cache to '{proc_path}' ({exc}); "
                    f"loaded '{raw_path}' in memory instead."
                )

        # Fallback: build live in memory from raw_path, or from built-in seed data if no external CSVs exist
        if has_raw_csvs:
            database = HarmonizedDatabase.from_directory(raw_path)
            live_status: ArtifactStatus = "live_fallback"
            live_warning = warning_msg
        else:
            database = HarmonizedDatabase.from_seed()
            live_status = "seed"
            live_warning = (
                f"No CSV training files were found in '{raw_path}' (and no prebuilt cache in '{proc_path}'). "
                "Running on the built-in 8-metal seed dataset (n < 10 -> low data). "
                f"Place your CSV files in '{raw_path}' and restart or click Reload Data."
            )
        predictor = HybridMLPredictor(model_family=model_family, random_state=42)
        predictor.fit(database)
        return cls(
            database=database,
            predictor=predictor,
            raw_dir=raw_path,
            artifact_dir=None,
            artifact_status=live_status,
            artifact_warning=live_warning,
            lab_dir=lab_path,
            raw_fingerprint=raw_fp,
            lab_projects=loaded_lab_projects,
        )

    def _invalidate_lab_cache_for_project(self, project_name: str) -> bool:
        keys_to_drop = [
            key
            for key in self._lab_calibration_cache
            if key == project_name or str(key).startswith(f"{project_name}:")
        ]
        for key in keys_to_drop:
            del self._lab_calibration_cache[key]
        return bool(keys_to_drop)

    def reload_external_data(
        self,
        *,
        rebuild_models: Sequence[str] = ("all",),
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> ExternalDataReloadSummary:
        """Check independent ``data/raw/`` and ``data/lab/`` fingerprints and reload only what changed."""
        from corrosion_simulator.build_artifacts import (
            build_processed_artifacts,
            check_artifacts_freshness,
        )

        start_time = time.perf_counter()
        raw_path = self.raw_dir
        proc_path = self.artifact_dir
        has_raw_csvs = bool(raw_path is not None and iter_raw_csv_files(raw_path))
        current_raw_fp = (
            HarmonizedDatabase.compute_raw_directory_fingerprint(raw_path)
            if has_raw_csvs
            else None
        )
        prev_sha = self.raw_fingerprint.get("combined_sha256") if self.raw_fingerprint else None
        curr_sha = current_raw_fp.get("combined_sha256") if current_raw_fp else None

        raw_changed = prev_sha != curr_sha
        if not raw_changed and has_raw_csvs and proc_path is not None:
            disk_status, _ = check_artifacts_freshness(raw_path, proc_path)
            if disk_status in ("stale", "missing"):
                raw_changed = True

        active_family = self.predictor.model_family
        raw_rebuilt = False
        lab_cache_invalidated = False
        added_or_updated: list[str] = []
        removed: list[str] = []

        if raw_changed:
            if has_raw_csvs and raw_path is not None:
                target_proc = proc_path or (
                    raw_path.parent / "processed" if raw_path.name == "raw" else None
                )
                if target_proc is not None:
                    try:
                        summary = build_processed_artifacts(
                            raw_dir=raw_path,
                            output_dir=target_proc,
                            models=rebuild_models,
                            export_csv=False,
                            progress_callback=progress_callback,
                        )
                        self.database = HarmonizedDatabase.from_processed_directory(target_proc)
                        self.artifact_dir = target_proc
                        self.artifact_status = "rebuilt"
                        self.artifact_warning = (
                            f"Detected updated CSV data in '{raw_path}' — rebuilt database "
                            f"({summary.n_records} records across {summary.n_materials} materials) "
                            f"and trained {len(summary.trained_models)} ML model(s)."
                        )
                        raw_rebuilt = True
                    except OSError as exc:
                        self.database = HarmonizedDatabase.from_directory(raw_path)
                        self.artifact_dir = None
                        self.artifact_status = "live_fallback"
                        self.artifact_warning = (
                            f"Could not write prebuilt cache ({exc}); loaded '{raw_path}' in memory."
                        )
                else:
                    self.database = HarmonizedDatabase.from_directory(raw_path)
                    self.artifact_status = "live_fallback"
            else:
                self.database = HarmonizedDatabase.from_seed()
                self.artifact_dir = None
                self.artifact_status = "seed"
                self.artifact_warning = (
                    f"No CSV training files were found in '{raw_path}'. Running on built-in seed data."
                )

            self.raw_fingerprint = current_raw_fp
            self._predictors.clear()
            self.predictor = self._load_or_fit_predictor(active_family)
            self._predictors[active_family] = self.predictor

            # Clear all cached lab calibration models so they re-anchor to the rebuilt prior
            self._lab_calibration_cache.clear()
            lab_cache_invalidated = True

            if self.lab_dir is not None:
                reloaded_projects = discover_and_load_lab_projects(self.lab_dir)
                added_or_updated = sorted(reloaded_projects.keys())
                removed = sorted(set(self.lab_projects) - set(reloaded_projects))
                self.lab_projects = reloaded_projects
                self.lab_fingerprints = {
                    name: ds.fingerprint for name, ds in reloaded_projects.items()
                }
            self.lab_status = "reloaded" if self.lab_projects else "none"
            self.lab_status_message = (
                f"Re-anchored {len(self.lab_projects)} lab project(s) to updated literature prior."
                if self.lab_projects
                else None
            )
        else:
            # Fast path (< 1 s): literature prior unchanged -> skip base rebuild and rescan data/lab/ only
            if self.lab_dir is not None:
                discovered = discover_lab_projects(self.lab_dir)
                removed = sorted(set(self.lab_projects) - set(discovered))
                for name in removed:
                    self.lab_projects.pop(name, None)
                    self.lab_fingerprints.pop(name, None)
                    if self._invalidate_lab_cache_for_project(name):
                        lab_cache_invalidated = True

                for name, files in discovered.items():
                    new_fp = compute_lab_project_fingerprint(name, files, self.lab_dir)
                    old_sha = self.lab_fingerprints.get(name, {}).get("combined_sha256")
                    if old_sha != new_fp["combined_sha256"]:
                        self.lab_projects[name] = load_lab_project(
                            name, files, lab_dir=self.lab_dir
                        )
                        self.lab_fingerprints[name] = new_fp
                        added_or_updated.append(name)
                        if self._invalidate_lab_cache_for_project(name):
                            lab_cache_invalidated = True
                if added_or_updated and self._invalidate_lab_cache_for_project("__none__"):
                    lab_cache_invalidated = True

            if added_or_updated or removed:
                self.lab_status = "reloaded"
                self.lab_status_message = (
                    f"Reloaded {len(added_or_updated)} lab project(s), removed {len(removed)}."
                )
            elif self.lab_projects:
                self.lab_status = "fresh"
                self.lab_status_message = f"{len(self.lab_projects)} lab project(s) up to date."
            else:
                self.lab_status = "none"
                self.lab_status_message = None

        elapsed = time.perf_counter() - start_time
        return ExternalDataReloadSummary(
            raw_rebuilt=raw_rebuilt,
            raw_status=self.artifact_status,
            lab_status=self.lab_status,
            lab_projects=tuple(sorted(self.lab_projects.keys())),
            lab_added_or_updated=tuple(sorted(added_or_updated)),
            lab_removed=tuple(sorted(removed)),
            lab_cache_invalidated=lab_cache_invalidated,
            elapsed_s=elapsed,
        )

    def get_lab_project(self, project_name: str | None = None) -> LabProjectDataset | None:
        """Return a loaded lab project dataset by name (or the first sorted project if omitted)."""
        if not self.lab_projects:
            return None
        if project_name is None:
            return self.lab_projects[sorted(self.lab_projects)[0]]
        return self.lab_projects.get(project_name)

    def get_calibrated_predictor(
        self,
        *,
        calibration_mode: CalibrationMode = "literature_only",
        lab_project: str | None = None,
        model_family: ModelFamily | None = None,
    ) -> HybridMLPredictor | LabCalibratedPredictor:
        """Return a base or lab-calibrated predictor for the requested calibration mode and project."""
        if calibration_mode not in CALIBRATION_MODES:
            raise ValueError(
                f"Unsupported calibration_mode '{calibration_mode}'. Supported: {CALIBRATION_MODES}"
            )
        target_family = model_family or self.predictor.model_family
        base_predictor = self.get_predictor(target_family)
        if calibration_mode == "literature_only":
            return base_predictor

        lab_ds: LabProjectDataset | None
        if lab_project is not None:
            lab_ds = self.get_lab_project(lab_project)
            if lab_ds is None:
                raise ValueError(
                    f"Lab project {lab_project!r} was not found in {self.lab_dir}. "
                    f"Available projects: {sorted(self.lab_projects)}"
                )
        else:
            lab_ds = self.get_lab_project(None)

        proj_key = lab_ds.project_name if lab_ds is not None else "__none__"
        cache_key = f"{proj_key}:{calibration_mode}:{target_family}"
        cached = self._lab_calibration_cache.get(cache_key)
        if cached is not None:
            return cached

        calibrated = LabCalibratedPredictor(
            base_predictor=base_predictor,
            calibration_mode=calibration_mode,
            lab_project=lab_ds.project_name if lab_ds is not None else lab_project,
            allow_fallback=True,
        ).fit(lab_ds)
        self._lab_calibration_cache[cache_key] = calibrated
        return calibrated

    def available_options(self) -> dict[str, Any]:
        """Return sidebar selector options dynamically derived from the harmonized database."""
        df = self.database.electrochemical_database
        metals = self.database.available_metals or list(CORE_METALS)
        counts = self.database.metal_sample_counts()
        meta_df = self.database.material_metadata()
        metal_tiers: dict[str, str] = {}
        metal_categories: dict[str, str] = {}
        metal_has_faraday: dict[str, bool] = {}
        if not meta_df.empty:
            for _, row in meta_df.iterrows():
                m = str(row["metal"])
                metal_tiers[m] = str(row["tier"])
                metal_categories[m] = str(row["category"])
                metal_has_faraday[m] = bool(row["has_faraday_properties"])

        def _extract_categories(col: str, defaults: list[str]) -> list[str]:
            values = set(defaults)
            if col in df.columns:
                for raw in df[col].dropna().astype(str):
                    cleaned = raw.strip()
                    if cleaned and cleaned.lower() not in {"nan", "none"}:
                        values.add(cleaned)
            ordered = ["None"] + sorted(v for v in values if v != "None")
            return ordered

        categories = sorted({metal_categories.get(m, "metal") for m in metals})
        known_organic = {
            name for name in SOLVENT_PHYSICAL_PROPERTIES if name != "water"
        }
        if "organic_solvent_type" in df.columns:
            for raw in df["organic_solvent_type"].dropna().astype(str):
                cleaned = normalize_category_label(raw)
                if cleaned != "None":
                    known_organic.add(cleaned)
        return {
            "metals": metals,
            "core_metals": [m for m in metals if metal_tiers.get(m, "core" if m in CORE_METALS else "extended") == "core"],
            "extended_metals": [m for m in metals if metal_tiers.get(m, "core" if m in CORE_METALS else "extended") != "core"],
            "metal_counts": counts,
            "metal_tiers": metal_tiers,
            "metal_categories": metal_categories,
            "metal_has_faraday": metal_has_faraday,
            "categories": categories,
            "oxidizer_types": _extract_categories("oxidizer_type", ["None", "H2O2", "HNO3", "FeCl3"]),
            "acid_types": _extract_categories(
                "acid_type", ["None", "Citric acid", "H2SO4", "H3PO4", "HCl", "HF"]
            ),
            "inhibitor_types": _extract_categories("inhibitor_type", ["None", "BTA"]),
            "solvent_regimes": list(SOLVENT_REGIMES),
            "organic_solvent_types": ["None"] + sorted(known_organic),
            "kinetics_models": ["auto", "linear", "passivation", "power_law"],
            "parameter_sources": ["ml", "database", "manual"],
            "ml_model_families": list(SUPPORTED_MODEL_FAMILIES),
            "calibration_modes": list(CALIBRATION_MODES),
            "lab_projects": sorted(self.lab_projects.keys()),
        }

    def _build_ml_query(self, request: SimulationRequest, metal: str) -> MLFeatureQuery:
        return MLFeatureQuery(
            metal=metal,
            pH=request.pH,
            temperature_C=request.temperature_C,
            oxidizer_type=request.oxidizer_type,
            oxidizer_concentration=request.oxidizer_concentration,
            acid_type=request.acid_type,
            acid_concentration=request.acid_concentration,
            inhibitor_type=request.inhibitor_type,
            inhibitor_concentration=request.inhibitor_concentration,
            ORP_mV=request.ORP_mV,
            organic_solvent_type=request.organic_solvent_type,
            water_content_wt_pct=float(request.water_content_wt_pct),
            base_solvent=request.base_solvent,
        )

    def _resolve_database_lookup(
        self,
        request: SimulationRequest,
        metal: str,
        *,
        solution_id: str | None = None,
    ) -> ResolvedMetalParameters:
        warnings: list[str] = []
        recommendations: list[str] = []
        lookup = self.database.lookup_exposure_condition(
            ExposureCondition(
                metal=metal,
                pH=request.pH,
                oxidizer_type=request.oxidizer_type,
                oxidizer_concentration=request.oxidizer_concentration,
                temperature_C=request.temperature_C,
                solution_id=solution_id,
            )
        )
        params = PolarizationParameters.from_lookup_result(lookup)
        confidence_tier = "High"
        if not lookup.is_exact_match:
            warnings.append(
                f"{metal} Database lookup used nearest neighbor ({lookup.record_id}, distance={lookup.normalized_distance:.2f}): {lookup.mismatch_summary}"
            )
            confidence_tier = "Medium" if lookup.normalized_distance < 0.75 else "Low"
            recommendations.append(
                f"Measure polarization curve for {metal} at pH {request.pH:.1f}, {request.temperature_C:.1f}°C, "
                f"{request.oxidizer_type} ({request.oxidizer_concentration:g}) to replace nearest-neighbor database lookup."
            )
        if (
            not lookup.candidates.empty
            and "solvent_regime" in lookup.candidates.columns
        ):
            matched_regime = normalize_category_label(lookup.candidates.iloc[0]["solvent_regime"])
            if matched_regime != "None" and matched_regime != request.solvent_regime:
                warnings.append(
                    f"{metal} Database lookup matched a '{matched_regime}' record ({lookup.record_id}), "
                    f"whereas the requested Solvent Regime is '{request.solvent_regime}'."
                )
                confidence_tier = "Low"
        if lookup.is_imputed:
            warnings.append(
                f"{metal} Database record ({lookup.record_id}) contains imputed Tafel slope(s) (quality_score={lookup.quality_score:.2f})."
            )
            if confidence_tier == "High":
                confidence_tier = "Medium"
        return ResolvedMetalParameters(
            metal=metal,
            source="database",
            parameters=params,
            confidence_tier=confidence_tier,
            warnings=warnings,
            experimental_recommendations=recommendations,
            lookup_result=lookup,
        )

    def _find_shared_database_solution(
        self,
        request: SimulationRequest,
        metal_a: str,
        metal_b: str,
    ) -> str | None:
        """Find the closest shared solution_id for two metals matching the requested oxidizer."""
        df = self.database.electrochemical_database
        df_a = df[df["metal"] == metal_a]
        df_b = df[df["metal"] == metal_b]
        shared_ids = sorted(
            set(df_a["solution_id"].dropna().astype(str))
            & set(df_b["solution_id"].dropna().astype(str))
        )
        if not shared_ids:
            return None

        req_ox = normalize_category_label(request.oxidizer_type)
        best_sid: str | None = None
        best_score = float("inf")
        for sid in shared_ids:
            sub = df[df["solution_id"].astype(str) == sid]
            ox_vals = {
                normalize_category_label(v) for v in sub["oxidizer_type"].dropna().tolist()
            } if "oxidizer_type" in sub.columns else set()
            if req_ox not in ox_vals:
                continue
            ph_vals = pd.to_numeric(sub["pH"], errors="coerce").dropna()
            temp_vals = pd.to_numeric(sub["temperature_C"], errors="coerce").dropna()
            conc_vals = pd.to_numeric(sub["oxidizer_concentration"], errors="coerce").dropna()
            ph_diff = abs(float(ph_vals.median()) - request.pH) / 14.0 if len(ph_vals) else 0.5
            temp_diff = abs(float(temp_vals.median()) - request.temperature_C) / 100.0 if len(temp_vals) else 0.5
            conc_diff = abs(float(conc_vals.median()) - request.oxidizer_concentration) / max(request.oxidizer_concentration, 1.0) if len(conc_vals) else 0.5
            score = ph_diff + temp_diff + 0.5 * conc_diff
            if score < best_score:
                best_score = score
                best_sid = sid
        return best_sid

    def _evaluate_cross_model_disagreement(
        self,
        request: SimulationRequest,
        metal: str,
        query: MLFeatureQuery,
        pred: MLParameterPrediction,
    ) -> MLParameterPrediction:
        """Check HGB vs RF+RGP agreement on unanchored literature queries without training missing peers."""
        if request.calibration_mode == "lab_only_gp":
            return pred
        if (
            request.calibration_mode == "delta_learning"
            and pred.applicability.lab_anchoring_status == "Lab-Anchored"
        ):
            return pred

        active_base = request.ml_model_family or _AUTO_ROUTED_BASE_MODELS[request.calibration_mode]
        peer_preds: dict[ModelFamily, MLParameterPrediction | None] = {}
        for family in ("rf_gp_ensemble", "hist_gradient_boosting"):
            if family == active_base and request.calibration_mode == "literature_only":
                peer_preds[family] = pred
            elif family == active_base:
                peer_preds[family] = self.get_predictor(family).predict(query)
            else:
                peer_model = self._get_cached_or_prebuilt_predictor(family)
                peer_preds[family] = peer_model.predict(query) if peer_model is not None else None

        rgp_pred = peer_preds["rf_gp_ensemble"]
        hgb_pred = peer_preds["hist_gradient_boosting"]
        if hgb_pred is None or rgp_pred is None:
            return pred

        delta_log_i = abs(
            float(hgb_pred.log10_icorr_uA_cm2.mean) - float(rgp_pred.log10_icorr_uA_cm2.mean)
        )
        delta_e = abs(float(hgb_pred.Ecorr_V_SHE.mean) - float(rgp_pred.Ecorr_V_SHE.mean))

        ad = pred.applicability
        new_tier: ConfidenceTier = ad.confidence_tier
        downgraded = False
        if delta_log_i > DISAGREEMENT_SEVERE_LOG10_ICORR:
            new_tier = "Low"
            downgraded = True
        elif (
            delta_log_i > DISAGREEMENT_MODERATE_LOG10_ICORR
            or delta_e > DISAGREEMENT_MODERATE_ECORR_V
        ):
            new_tier = _CONFIDENCE_BY_RANK[max(0, _CONFIDENCE_RANK[ad.confidence_tier] - 1)]  # type: ignore[assignment]
            downgraded = True

        warnings_list = list(ad.warnings)
        recs_list = list(ad.experimental_recommendations)
        if downgraded:
            warnings_list.append(
                f"Cross-model disagreement between hist_gradient_boosting and rf_gp_ensemble for {metal} "
                f"(|Δlog10(i_corr)| = {delta_log_i:.2f} dec, |ΔE_corr| = {delta_e:.3f} V); "
                f"Confidence Tier set to {new_tier}."
            )
            recs_list.append(
                f"Run confirmatory Tafel polarization experiments for {metal} at pH {query.pH:.1f}, "
                f"{query.temperature_C:.1f} °C to resolve heterogeneous model divergence."
            )

        updated_ad = replace(
            ad,
            confidence_tier=new_tier,
            warnings=tuple(dict.fromkeys(warnings_list)),
            experimental_recommendations=tuple(dict.fromkeys(recs_list)),
            cross_model_disagreement_log10_icorr=delta_log_i,
            cross_model_disagreement_ecorr_V=delta_e,
            disagreement_downgraded=downgraded,
        )
        return replace(pred, applicability=updated_ad)

    def _resolve_metal_parameters(
        self,
        request: SimulationRequest,
        metal: str,
        *,
        manual_override: ManualElectrochemicalInput | None = None,
    ) -> ResolvedMetalParameters:
        warnings: list[str] = []
        recommendations: list[str] = []

        if request.parameter_source == "manual":
            if manual_override is None:
                # Fall back gracefully to database lookup values if caller omitted manual override
                lookup = self.database.lookup_exposure_condition(
                    metal=metal,
                    pH=request.pH,
                    oxidizer_type=request.oxidizer_type,
                    oxidizer_concentration=request.oxidizer_concentration,
                    temperature_C=request.temperature_C,
                )
                params = PolarizationParameters.from_lookup_result(lookup)
                warnings.append(
                    f"Manual override values were not provided for {metal}; defaulted to nearest Database record ({lookup.record_id})."
                )
            else:
                params = PolarizationParameters(
                    Ecorr_V_SHE=manual_override.Ecorr_V_SHE,
                    icorr_uA_cm2=manual_override.icorr_uA_cm2,
                    beta_a_mV_dec=manual_override.beta_a_mV_dec,
                    beta_c_mV_dec=manual_override.beta_c_mV_dec,
                    metal=metal,
                    temperature_C=request.temperature_C,
                )
            return ResolvedMetalParameters(
                metal=metal,
                source="manual",
                parameters=params,
                confidence_tier="High",
                warnings=warnings,
                experimental_recommendations=recommendations,
            )

        if request.parameter_source == "database":
            return self._resolve_database_lookup(request, metal)

        # ML prediction mode (with optional Delta-Learning or Lab-Only GP calibration)
        query = self._build_ml_query(request, metal)
        predictor = self.get_calibrated_predictor(
            calibration_mode=request.calibration_mode,
            lab_project=request.lab_project,
            model_family=request.ml_model_family,
        )
        pred = predictor.predict(query)
        pred = self._evaluate_cross_model_disagreement(request, metal, query, pred)
        params = pred.to_polarization_parameters()
        # Approximate linear-space 1-sigma for i_corr from log10-normal distribution:
        # sd(10^X) ≈ median * ln(10) * sigma_log10
        icorr_std = float(pred.icorr_uA_cm2_median * math.log(10.0) * pred.log10_icorr_uA_cm2.std)
        resolved_project = pred.lab_project
        if resolved_project is None and request.calibration_mode != "literature_only":
            lab_ds = self.get_lab_project(request.lab_project)
            resolved_project = lab_ds.project_name if lab_ds is not None else request.lab_project
        return ResolvedMetalParameters(
            metal=metal,
            source="ml",
            parameters=params,
            Ecorr_std_V=float(pred.Ecorr_V_SHE.std),
            log10_icorr_std=float(pred.log10_icorr_uA_cm2.std),
            icorr_std_uA_cm2=icorr_std,
            beta_a_std_mV_dec=float(pred.beta_a_mV_dec.std),
            beta_c_std_mV_dec=float(pred.beta_c_mV_dec.std),
            confidence_tier=pred.applicability.confidence_tier,
            warnings=list(pred.applicability.warnings),
            experimental_recommendations=list(pred.applicability.experimental_recommendations),
            ml_prediction=pred,
            applicability=pred.applicability,
            calibration_mode=request.calibration_mode,
            lab_project=resolved_project,
            residual_breakdown=pred.residual_breakdown,
        )

    def run_single_metal_analysis(self, request: SimulationRequest) -> SingleMetalViewPayload:
        """Execute end-to-end Single Metal Analysis and build interactive Plotly figures."""
        resolved = self._resolve_metal_parameters(
            request,
            request.primary_metal,
            manual_override=request.manual_primary,
        )
        valence = (
            request.manual_primary.valence
            if request.parameter_source == "manual" and request.manual_primary is not None and request.manual_primary.valence is not None
            else request.primary_valence
        )
        curve = generate_polarization_curve(resolved.parameters)
        warnings_list = list(resolved.warnings)
        if request.solvent_regime == "non_aqueous":
            warnings_list.insert(
                0,
                build_non_aqueous_physics_warning(
                    float(request.water_content_wt_pct),
                    request.organic_solvent_type,
                ),
            )
        recs_list = list(resolved.experimental_recommendations)

        if not has_faraday_properties(request.primary_metal, self.database.material_properties):
            warnings_list.append(
                f"Missing physical properties (density / equivalent weight) for {request.primary_metal}; "
                "Faraday corrosion rate (nm/min) and thickness-loss kinetics are unavailable. "
                "Add a row to data/raw/material_properties.csv to enable rate conversion."
            )
            pol_fig = build_single_metal_polarization_figure(
                curve,
                mc_result=None,
                title=f"{request.primary_metal} Polarization Curve ({resolved.source.upper()})",
            )
            placeholder_fig = go.Figure()
            placeholder_fig.add_annotation(
                text=f"Faraday corrosion rate & kinetics unavailable for {request.primary_metal}<br>"
                "(missing density / equivalent_weight_g_eq in material_properties.csv)",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14),
            )
            placeholder_fig.update_layout(
                title=f"{request.primary_metal} Time-Resolved Kinetics (Unavailable)",
                template="plotly_white",
                height=360,
            )
            return SingleMetalViewPayload(
                request=request,
                resolved_parameters=resolved,
                parameter_table=resolved.to_parameter_table(),
                faraday_result=None,
                polarization_curve=curve,
                mc_propagation=None,
                kinetics_result=None,
                mc_kinetics=None,
                initial_corrosion_rate_nm_min=float("nan"),
                initial_corrosion_rate_mm_yr=float("nan"),
                corrosion_depth_nm=float("nan"),
                time_to_threshold_s=float("nan"),
                thickness_loss_risk_score=float("nan"),
                thickness_loss_risk_tier="N/A",
                confidence_tier=resolved.confidence_tier,
                warnings=_dedupe_preserve_order(warnings_list),
                experimental_recommendations=_dedupe_preserve_order(recs_list),
                polarization_figure=pol_fig,
                kinetics_risk_figure=placeholder_fig,
                depth_figure=placeholder_fig,
                risk_figure=placeholder_fig,
            )

        faraday = calculate_faraday_corrosion_rate(
            metal=request.primary_metal,
            icorr_uA_cm2=resolved.parameters.icorr_uA_cm2,
            valence=valence,
            immersion_time_s=request.immersion_time_s,
            material_properties=self.database.material_properties,
        )

        mc_propagation: MonteCarloPropagationResult | None = None
        if resolved.ml_prediction is not None:
            mc_propagation = propagate_uncertainty_monte_carlo(
                resolved.ml_prediction,
                n_draws=request.mc_draws,
                seed=request.mc_seed,
                valence=valence,
                material_properties=self.database.material_properties,
            )

        times_s = _build_time_grid(request.immersion_time_s)
        inhibitor_present = (
            request.inhibitor_type.strip().lower() not in {"", "none"}
            and request.inhibitor_concentration > 0.0
        )
        model_arg = None if request.kinetics_model == "auto" else request.kinetics_model
        kinetics = simulate_time_resolved_kinetics(
            faraday,
            times_s,
            model=model_arg,
            metal=request.primary_metal,
            pH=request.pH,
            inhibitor_present=inhibitor_present,
            material_properties=self.database.material_properties,
            d_crit_nm=request.d_crit_nm,
            d_max_nm=request.d_max_nm,
            power_law_n=request.power_law_n,
        )

        mc_kinetics: MonteCarloKineticsResult | None = None
        if mc_propagation is not None:
            mc_kinetics = propagate_kinetics_uncertainty_monte_carlo(
                mc_propagation,
                times_s,
                model=kinetics.model,
                metal=request.primary_metal,
                pH=request.pH,
                inhibitor_present=inhibitor_present,
                material_properties=self.database.material_properties,
                d_crit_nm=request.d_crit_nm,
                d_max_nm=request.d_max_nm,
                power_law_n=request.power_law_n,
            )

        final_depth = float(kinetics.depth_nm[-1])
        final_risk = float(kinetics.risk_score[-1])
        final_tier = classify_thickness_loss_risk_tier(final_depth, request.d_crit_nm)

        pol_fig = build_single_metal_polarization_figure(
            curve,
            mc_result=mc_propagation,
            title=f"{request.primary_metal} Polarization Curve ({resolved.source.upper()})",
        )
        kin_fig = build_single_metal_kinetics_figure(
            kinetics,
            mc_kinetics=mc_kinetics,
            title=f"{request.primary_metal} Time-Resolved Kinetics & Thickness Loss Risk ({kinetics.model})",
        )
        depth_fig = build_single_metal_depth_figure(
            kinetics,
            mc_kinetics=mc_kinetics,
            title=f"{request.primary_metal} Corrosion Depth vs. Immersion Time ({kinetics.model})",
        )
        risk_fig = build_single_metal_risk_figure(
            kinetics,
            mc_kinetics=mc_kinetics,
            title=f"{request.primary_metal} Thickness Loss Risk vs. Immersion Time ({kinetics.model})",
        )

        return SingleMetalViewPayload(
            request=request,
            resolved_parameters=resolved,
            parameter_table=resolved.to_parameter_table(),
            faraday_result=faraday,
            polarization_curve=curve,
            mc_propagation=mc_propagation,
            kinetics_result=kinetics,
            mc_kinetics=mc_kinetics,
            initial_corrosion_rate_nm_min=float(faraday.corrosion_rate_nm_min),
            initial_corrosion_rate_mm_yr=float(faraday.corrosion_rate_mm_yr),
            corrosion_depth_nm=final_depth,
            time_to_threshold_s=float(kinetics.time_to_threshold_s),
            thickness_loss_risk_score=final_risk,
            thickness_loss_risk_tier=final_tier,
            confidence_tier=resolved.confidence_tier,
            warnings=_dedupe_preserve_order(warnings_list),
            experimental_recommendations=_dedupe_preserve_order(recs_list),
            polarization_figure=pol_fig,
            kinetics_risk_figure=kin_fig,
            depth_figure=depth_fig,
            risk_figure=risk_fig,
        )

    def run_galvanic_analysis(self, request: SimulationRequest) -> GalvanicViewPayload:
        """Execute end-to-end Bimetallic Galvanic Analysis and build interactive Plotly figures."""
        if request.secondary_metal is None:
            raise ValueError("secondary_metal must be specified for Galvanic Analysis.")
        if request.secondary_metal == request.primary_metal:
            raise ValueError("primary_metal and secondary_metal must be distinct for Galvanic Analysis.")

        primary_resolved = self._resolve_metal_parameters(
            request,
            request.primary_metal,
            manual_override=request.manual_primary,
        )
        secondary_resolved = self._resolve_metal_parameters(
            request,
            request.secondary_metal,
            manual_override=request.manual_secondary,
        )

        # In Database mode, try to pair both metals on a shared Solution if their independent
        # nearest-neighbor lookups landed on different solution_ids or temperatures.
        extra_warnings: list[str] = []
        if request.solvent_regime == "non_aqueous":
            extra_warnings.append(
                build_non_aqueous_physics_warning(
                    float(request.water_content_wt_pct),
                    request.organic_solvent_type,
                )
            )
        extra_recs: list[str] = []
        force_low_confidence = False
        solve_params_a = primary_resolved.parameters
        solve_params_b = secondary_resolved.parameters

        if request.parameter_source == "database":
            sol_mismatch = (
                solve_params_a.solution_id is not None
                and solve_params_b.solution_id is not None
                and solve_params_a.solution_id != solve_params_b.solution_id
            )
            temp_mismatch = (
                solve_params_a.temperature_C is not None
                and solve_params_b.temperature_C is not None
                and not math.isclose(solve_params_a.temperature_C, solve_params_b.temperature_C, abs_tol=1e-6)
            )
            if sol_mismatch or temp_mismatch:
                shared_sid = self._find_shared_database_solution(
                    request, request.primary_metal, request.secondary_metal
                )
                if shared_sid is not None:
                    primary_resolved = self._resolve_database_lookup(
                        request, request.primary_metal, solution_id=shared_sid
                    )
                    secondary_resolved = self._resolve_database_lookup(
                        request, request.secondary_metal, solution_id=shared_sid
                    )
                    solve_params_a = primary_resolved.parameters
                    solve_params_b = secondary_resolved.parameters
                    sol_mismatch = False
                    temp_mismatch = (
                        solve_params_a.temperature_C is not None
                        and solve_params_b.temperature_C is not None
                        and not math.isclose(
                            solve_params_a.temperature_C, solve_params_b.temperature_C, abs_tol=1e-6
                        )
                    )

            if sol_mismatch or temp_mismatch:
                force_low_confidence = True
                extra_warnings.append(
                    f"Database lookup matched different experimental conditions for {request.primary_metal} "
                    f"({solve_params_a.solution_id}, {solve_params_a.temperature_C}°C) and {request.secondary_metal} "
                    f"({solve_params_b.solution_id}, {solve_params_b.temperature_C}°C); galvanic coupling was "
                    f"approximated at the requested Exposure Condition ({request.temperature_C:.1f}°C). "
                    f"Consider switching to ML Prediction mode for matched-condition estimation."
                )
                extra_recs.append(
                    f"Measure polarization curves for both {request.primary_metal} and {request.secondary_metal} "
                    f"in the same Solution (pH {request.pH:.1f}, {request.oxidizer_type} "
                    f"{request.oxidizer_concentration:g}, {request.temperature_C:.1f}°C) to validate galvanic coupling."
                )
                solve_params_a = replace(
                    solve_params_a, solution_id=None, temperature_C=request.temperature_C
                )
                solve_params_b = replace(
                    solve_params_b, solution_id=None, temperature_C=request.temperature_C
                )

        driving_force = screen_galvanic_driving_force(
            metal_a=request.primary_metal,
            Ecorr_a_V_SHE=solve_params_a.Ecorr_V_SHE,
            metal_b=request.secondary_metal,
            Ecorr_b_V_SHE=solve_params_b.Ecorr_V_SHE,
            temperature_C=request.temperature_C,
        )

        # Determine active metal valence override if specified
        active_valence: int | None = None
        if driving_force.active_metal == request.primary_metal:
            active_valence = (
                request.manual_primary.valence
                if request.parameter_source == "manual"
                and request.manual_primary is not None
                and request.manual_primary.valence is not None
                else request.primary_valence
            )
        else:
            active_valence = (
                request.manual_secondary.valence
                if request.parameter_source == "manual"
                and request.manual_secondary is not None
                and request.manual_secondary.valence is not None
                else request.secondary_valence
            )

        coupling = solve_galvanic_couple(
            solve_params_a,
            solve_params_b,
            cathode_to_anode_area_ratio=request.cathode_to_anode_area_ratio,
            active_valence=active_valence,
            immersion_time_s=request.immersion_time_s,
            material_properties=self.database.material_properties,
            allow_missing_faraday=True,
        )

        galv_pol_fig = build_galvanic_polarization_figure(
            coupling,
            mc_galvanic=None,
        )

        if coupling.uncoupled_faraday is None or coupling.coupled_faraday is None:
            extra_warnings.append(
                f"Missing physical properties (density / equivalent weight) for Active Metal {coupling.active_metal}; "
                "Faraday corrosion rate (nm/min) and coupled thickness-loss kinetics are unavailable. "
                "Mixed potential (E_galv), galvanic current density (i_galv), and Acceleration Factor remain valid."
            )
            combined_rank = min(
                _CONFIDENCE_RANK.get(primary_resolved.confidence_tier, 0),
                _CONFIDENCE_RANK.get(secondary_resolved.confidence_tier, 0),
            )
            if force_low_confidence:
                combined_rank = 0
            combined_confidence = _CONFIDENCE_BY_RANK[combined_rank]
            combined_warnings = _dedupe_preserve_order(
                extra_warnings + primary_resolved.warnings + secondary_resolved.warnings
            )
            combined_recs = _dedupe_preserve_order(
                extra_recs
                + primary_resolved.experimental_recommendations
                + secondary_resolved.experimental_recommendations
            )
            placeholder_fig = go.Figure()
            placeholder_fig.add_annotation(
                text=f"Coupled Faraday corrosion rate & kinetics unavailable for {coupling.active_metal}<br>"
                "(missing density / equivalent_weight_g_eq in material_properties.csv)",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14),
            )
            placeholder_fig.update_layout(
                title=f"Galvanic Kinetics Comparison — {coupling.active_metal} (Unavailable)",
                template="plotly_white",
                height=360,
            )
            return GalvanicViewPayload(
                request=request,
                primary_resolved=primary_resolved,
                secondary_resolved=secondary_resolved,
                driving_force=driving_force,
                coupling_result=coupling,
                mc_galvanic=None,
                uncoupled_kinetics=None,
                coupled_kinetics=None,
                mc_coupled_kinetics=None,
                active_metal=coupling.active_metal,
                noble_metal=coupling.noble_metal,
                galvanic_driving_force_risk=driving_force.risk_tier,
                mixed_potential_V_SHE=float(coupling.mixed_potential_V_SHE),
                galvanic_current_density_uA_cm2=float(coupling.galvanic_current_density_uA_cm2),
                acceleration_factor=float(coupling.acceleration_factor),
                uncoupled_cr0_nm_min=float("nan"),
                accelerated_cr0_nm_min=float("nan"),
                uncoupled_depth_nm=float("nan"),
                coupled_depth_nm=float("nan"),
                uncoupled_time_to_threshold_s=float("nan"),
                coupled_time_to_threshold_s=float("nan"),
                uncoupled_risk_score=float("nan"),
                coupled_risk_score=float("nan"),
                uncoupled_risk_tier="N/A",
                coupled_risk_tier="N/A",
                confidence_tier=combined_confidence,
                warnings=combined_warnings,
                experimental_recommendations=combined_recs,
                galvanic_polarization_figure=galv_pol_fig,
                comparative_kinetics_figure=placeholder_fig,
                galvanic_depth_figure=placeholder_fig,
                galvanic_risk_figure=placeholder_fig,
            )

        mc_galvanic: MonteCarloGalvanicResult | None = None
        if primary_resolved.ml_prediction is not None and secondary_resolved.ml_prediction is not None:
            mc_galvanic = propagate_galvanic_uncertainty_monte_carlo(
                primary_resolved.ml_prediction,
                secondary_resolved.ml_prediction,
                cathode_to_anode_area_ratio=request.cathode_to_anode_area_ratio,
                n_draws=request.mc_draws,
                seed=request.mc_seed,
                active_valence=active_valence,
                material_properties=self.database.material_properties,
            )

        times_s = _build_time_grid(request.immersion_time_s)
        inhibitor_present = (
            request.inhibitor_type.strip().lower() not in {"", "none"}
            and request.inhibitor_concentration > 0.0
        )
        model_arg = None if request.kinetics_model == "auto" else request.kinetics_model

        uncoupled_kinetics = simulate_time_resolved_kinetics(
            coupling.uncoupled_faraday,
            times_s,
            model=model_arg,
            metal=coupling.active_metal,
            pH=request.pH,
            inhibitor_present=inhibitor_present,
            material_properties=self.database.material_properties,
            d_crit_nm=request.d_crit_nm,
            d_max_nm=request.d_max_nm,
            power_law_n=request.power_law_n,
        )
        coupled_kinetics = simulate_time_resolved_kinetics(
            coupling,
            times_s,
            model=uncoupled_kinetics.model,
            metal=coupling.active_metal,
            pH=request.pH,
            inhibitor_present=inhibitor_present,
            material_properties=self.database.material_properties,
            d_crit_nm=request.d_crit_nm,
            d_max_nm=request.d_max_nm,
            power_law_n=request.power_law_n,
        )

        mc_coupled_kinetics: MonteCarloKineticsResult | None = None
        if mc_galvanic is not None:
            mc_coupled_kinetics = propagate_kinetics_uncertainty_monte_carlo(
                mc_galvanic,
                times_s,
                model=coupled_kinetics.model,
                metal=coupling.active_metal,
                pH=request.pH,
                inhibitor_present=inhibitor_present,
                material_properties=self.database.material_properties,
                d_crit_nm=request.d_crit_nm,
                d_max_nm=request.d_max_nm,
                power_law_n=request.power_law_n,
            )

        uncoupled_depth = float(uncoupled_kinetics.depth_nm[-1])
        coupled_depth = float(coupled_kinetics.depth_nm[-1])
        uncoupled_risk = float(uncoupled_kinetics.risk_score[-1])
        coupled_risk = float(coupled_kinetics.risk_score[-1])
        uncoupled_tier = classify_thickness_loss_risk_tier(uncoupled_depth, request.d_crit_nm)
        coupled_tier = classify_thickness_loss_risk_tier(coupled_depth, request.d_crit_nm)

        combined_rank = min(
            _CONFIDENCE_RANK.get(primary_resolved.confidence_tier, 0),
            _CONFIDENCE_RANK.get(secondary_resolved.confidence_tier, 0),
        )
        if force_low_confidence:
            combined_rank = 0
        combined_confidence = _CONFIDENCE_BY_RANK[combined_rank]
        combined_warnings = _dedupe_preserve_order(
            extra_warnings + primary_resolved.warnings + secondary_resolved.warnings
        )
        combined_recs = _dedupe_preserve_order(
            extra_recs
            + primary_resolved.experimental_recommendations
            + secondary_resolved.experimental_recommendations
        )

        galv_pol_fig = build_galvanic_polarization_figure(
            coupling,
            mc_galvanic=mc_galvanic,
        )
        comp_kin_fig = build_galvanic_kinetics_comparison_figure(
            uncoupled_kinetics,
            coupled_kinetics,
            mc_coupled_kinetics=mc_coupled_kinetics,
            active_metal=coupling.active_metal,
        )
        galv_depth_fig = build_galvanic_depth_comparison_figure(
            uncoupled_kinetics,
            coupled_kinetics,
            mc_coupled_kinetics=mc_coupled_kinetics,
            active_metal=coupling.active_metal,
        )
        galv_risk_fig = build_galvanic_risk_comparison_figure(
            uncoupled_kinetics,
            coupled_kinetics,
            mc_coupled_kinetics=mc_coupled_kinetics,
            active_metal=coupling.active_metal,
        )

        return GalvanicViewPayload(
            request=request,
            primary_resolved=primary_resolved,
            secondary_resolved=secondary_resolved,
            driving_force=driving_force,
            coupling_result=coupling,
            mc_galvanic=mc_galvanic,
            uncoupled_kinetics=uncoupled_kinetics,
            coupled_kinetics=coupled_kinetics,
            mc_coupled_kinetics=mc_coupled_kinetics,
            active_metal=coupling.active_metal,
            noble_metal=coupling.noble_metal,
            galvanic_driving_force_risk=driving_force.risk_tier,
            mixed_potential_V_SHE=float(coupling.mixed_potential_V_SHE),
            galvanic_current_density_uA_cm2=float(coupling.galvanic_current_density_uA_cm2),
            acceleration_factor=float(coupling.acceleration_factor),
            uncoupled_cr0_nm_min=float(coupling.uncoupled_faraday.corrosion_rate_nm_min),
            accelerated_cr0_nm_min=float(coupling.coupled_faraday.corrosion_rate_nm_min),
            uncoupled_depth_nm=uncoupled_depth,
            coupled_depth_nm=coupled_depth,
            uncoupled_time_to_threshold_s=float(uncoupled_kinetics.time_to_threshold_s),
            coupled_time_to_threshold_s=float(coupled_kinetics.time_to_threshold_s),
            uncoupled_risk_score=uncoupled_risk,
            coupled_risk_score=coupled_risk,
            uncoupled_risk_tier=uncoupled_tier,
            coupled_risk_tier=coupled_tier,
            confidence_tier=combined_confidence,
            warnings=combined_warnings,
            experimental_recommendations=combined_recs,
            galvanic_polarization_figure=galv_pol_fig,
            comparative_kinetics_figure=comp_kin_fig,
            galvanic_depth_figure=galv_depth_fig,
            galvanic_risk_figure=galv_risk_fig,
        )
