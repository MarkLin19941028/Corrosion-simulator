"""Simulation orchestrator for Single Metal and Galvanic Corrosion Analysis."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from corrosion_simulator.artifact_repository import (
    ArtifactRepository,
    ArtifactStatus,
    LabProjectDataset,
)
from corrosion_simulator.database import (
    CORE_METALS,
    SOLVENT_REGIMES,
    ExposureCondition,
    ExposureLookupResult,
    HarmonizedDatabase,
    normalize_category_label,
)
from corrosion_simulator.solution import (
    NON_AQUEOUS_WARNING_PREFIX,
    CleaningRecipe,
    Solution,
    build_non_aqueous_physics_warning,
    resolve_consistent_solvent_settings,
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
from corrosion_simulator.lab_calibration import (
    CALIBRATION_MODES,
    CalibrationMode,
    ResidualBreakdown,
)
from corrosion_simulator.ml_predictor import (
    MODEL_FAMILY_LABELS,
    SUPPORTED_MODEL_FAMILIES,
    ApplicabilityDomainResult,
    ConfidenceTier,
    HybridMLPredictor,
    MLFeatureQuery,
    ModelFamily,
    PredictedModelFamily,
)
from corrosion_simulator.uncertainty import (
    MLParameterPrediction,
    MonteCarloPropagationResult,
    propagate_uncertainty_monte_carlo,
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

_CONFIDENCE_RANK = {"Low": 0, "Medium": 1, "High": 2}
_CONFIDENCE_BY_RANK = {0: "Low", 1: "Medium", 2: "High"}

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
    """Unified simulation configuration composed from Metal selection and a CleaningRecipe."""

    primary_metal: str
    secondary_metal: str | None = None
    cathode_to_anode_area_ratio: float = 1.0
    recipe: CleaningRecipe = field(default_factory=CleaningRecipe)
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
        if not isinstance(self.recipe, CleaningRecipe):
            raise TypeError(
                f"SimulationRequest.recipe must be a CleaningRecipe, got {type(self.recipe).__name__}."
            )
        if self.cathode_to_anode_area_ratio <= 0 or not np.isfinite(self.cathode_to_anode_area_ratio):
            raise ValueError("cathode_to_anode_area_ratio must be strictly positive.")
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

    @property
    def source_label(self) -> str:
        """Human-readable provenance label for this resolved parameter set."""
        if self.source == "ml":
            return {
                "literature_only": "ML",
                "delta_learning": "ML (Delta-Learning)",
                "lab_only_gp": "ML (Lab-Only GP)",
            }[self.calibration_mode]
        return {
            "database": "Database",
            "manual": "Manual",
        }[self.source]

    def to_parameter_table(self) -> pd.DataFrame:
        """Build a display-ready parameter summary table with ±1σ uncertainties."""
        source_label = self.source_label
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
class SingleMetalAnalysisResult:
    """Pure, Plotly-free simulation result for Single Metal Analysis."""

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
            "pH": float(self.request.recipe.solution.pH),
            "temperature_C": float(self.request.recipe.temperature_C),
            "solvent_regime": str(self.request.recipe.solution.solvent_regime),
            "solvent_subclass": str(self.request.recipe.solution.solvent_subclass),
            "organic_solvent_type": str(self.request.recipe.solution.organic_solvent_type),
            "water_content_wt_pct": float(self.request.recipe.solution.water_content_wt_pct),
            "immersion_time_s": float(self.request.recipe.immersion_time_s),
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
class GalvanicAnalysisResult:
    """Pure, Plotly-free simulation result for Bimetallic Galvanic Analysis."""

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

    @property
    def electrochemical_parameter_table(self) -> pd.DataFrame:
        """Build a summary table of Anode (Active) & Cathode (Noble) 4 parameters plus Mixed (E_galv, i_galv)."""
        if self.primary_resolved.metal == self.active_metal:
            active_resolved = self.primary_resolved
            noble_resolved = self.secondary_resolved
        else:
            active_resolved = self.secondary_resolved
            noble_resolved = self.primary_resolved

        active_params = self.coupling_result.active_parameters
        noble_params = self.coupling_result.noble_parameters

        return pd.DataFrame(
            [
                {
                    "Role": f"Anode (Active): {self.active_metal}",
                    "E_corr / E_galv (V vs. SHE)": round(float(active_params.Ecorr_V_SHE), 4),
                    "i_corr / i_galv (µA/cm²)": float(active_params.icorr_uA_cm2),
                    "β_a (mV/dec)": round(float(active_params.beta_a_mV_dec), 2),
                    "β_c (mV/dec)": round(float(active_params.beta_c_mV_dec), 2),
                    "Source": active_resolved.source_label,
                },
                {
                    "Role": f"Cathode (Noble): {self.noble_metal}",
                    "E_corr / E_galv (V vs. SHE)": round(float(noble_params.Ecorr_V_SHE), 4),
                    "i_corr / i_galv (µA/cm²)": float(noble_params.icorr_uA_cm2),
                    "β_a (mV/dec)": round(float(noble_params.beta_a_mV_dec), 2),
                    "β_c (mV/dec)": round(float(noble_params.beta_c_mV_dec), 2),
                    "Source": noble_resolved.source_label,
                },
                {
                    "Role": f"Mixed (Coupled, A_ratio={self.request.cathode_to_anode_area_ratio:g})",
                    "E_corr / E_galv (V vs. SHE)": round(float(self.mixed_potential_V_SHE), 4),
                    "i_corr / i_galv (µA/cm²)": float(self.galvanic_current_density_uA_cm2),
                    "β_a (mV/dec)": None,
                    "β_c (mV/dec)": None,
                    "Source": "Mixed-Potential Solver",
                },
            ]
        )

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the galvanic coupling analysis."""
        return {
            "primary_metal": self.request.primary_metal,
            "secondary_metal": self.request.secondary_metal,
            "active_metal": self.active_metal,
            "noble_metal": self.noble_metal,
            "cathode_to_anode_area_ratio": float(self.request.cathode_to_anode_area_ratio),
            "solvent_regime": str(self.request.recipe.solution.solvent_regime),
            "solvent_subclass": str(self.request.recipe.solution.solvent_subclass),
            "organic_solvent_type": str(self.request.recipe.solution.organic_solvent_type),
            "water_content_wt_pct": float(self.request.recipe.solution.water_content_wt_pct),
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


class CorrosionPlatformController:
    """Simulation orchestrator backing the Streamlit UI and reporting workflows."""

    def __init__(
        self,
        database_or_repository: ArtifactRepository | HarmonizedDatabase,
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
        if isinstance(database_or_repository, ArtifactRepository):
            extra_args = [
                name
                for name, val in (
                    ("predictor", predictor),
                    ("raw_dir", raw_dir),
                    ("artifact_dir", artifact_dir),
                    ("artifact_warning", artifact_warning),
                    ("lab_dir", lab_dir),
                    ("raw_fingerprint", raw_fingerprint),
                    ("lab_projects", lab_projects),
                )
                if val is not None
            ]
            if artifact_status != "live":
                extra_args.append("artifact_status")
            if extra_args:
                raise TypeError(
                    "Cannot pass repository configuration arguments "
                    f"({', '.join(extra_args)}) alongside an existing ArtifactRepository."
                )
            self.repository = database_or_repository
        else:
            self.repository = ArtifactRepository(
                database=database_or_repository,
                predictor=predictor,
                raw_dir=raw_dir,
                artifact_dir=artifact_dir,
                artifact_status=artifact_status,
                artifact_warning=artifact_warning,
                lab_dir=lab_dir,
                raw_fingerprint=raw_fingerprint,
                lab_projects=lab_projects,
            )

    @property
    def database(self) -> HarmonizedDatabase:
        """Return the active harmonized database owned by the underlying ArtifactRepository."""
        return self.repository.database

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
        """Build a controller backed by an ArtifactRepository loaded from the default data layout."""
        repo = ArtifactRepository.from_default_data(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            lab_dir=lab_dir,
            data_dir=data_dir,
            model_family=model_family,
            auto_rebuild=auto_rebuild,
            rebuild_models=rebuild_models,
            progress_callback=progress_callback,
        )
        return cls(repo)

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
            "lab_projects": sorted(self.repository.lab_projects.keys()),
        }

    def _resolve_database_lookup(
        self,
        request: SimulationRequest,
        metal: str,
        *,
        solution_id: str | None = None,
    ) -> ResolvedMetalParameters:
        sol = request.recipe.solution
        temp_c = request.recipe.temperature_C
        warnings: list[str] = []
        recommendations: list[str] = []
        lookup = self.database.lookup_exposure_condition(
            ExposureCondition(
                metal=metal,
                pH=sol.pH,
                oxidizer_type=sol.oxidizer_type,
                oxidizer_concentration=sol.oxidizer_concentration,
                temperature_C=temp_c,
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
                f"Measure polarization curve for {metal} at pH {sol.pH:.1f}, {temp_c:.1f}°C, "
                f"{sol.oxidizer_type} ({sol.oxidizer_concentration:g}) to replace nearest-neighbor database lookup."
            )
        if (
            not lookup.candidates.empty
            and "solvent_regime" in lookup.candidates.columns
        ):
            matched_regime = normalize_category_label(lookup.candidates.iloc[0]["solvent_regime"])
            if matched_regime != "None" and matched_regime != sol.solvent_regime:
                warnings.append(
                    f"{metal} Database lookup matched a '{matched_regime}' record ({lookup.record_id}), "
                    f"whereas the requested Solvent Regime is '{sol.solvent_regime}'."
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
        sol = request.recipe.solution
        temp_c = request.recipe.temperature_C
        df = self.database.electrochemical_database
        df_a = df[df["metal"] == metal_a]
        df_b = df[df["metal"] == metal_b]
        shared_ids = sorted(
            set(df_a["solution_id"].dropna().astype(str))
            & set(df_b["solution_id"].dropna().astype(str))
        )
        if not shared_ids:
            return None

        req_ox = normalize_category_label(sol.oxidizer_type)
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
            ph_diff = abs(float(ph_vals.median()) - sol.pH) / 14.0 if len(ph_vals) else 0.5
            temp_diff = abs(float(temp_vals.median()) - temp_c) / 100.0 if len(temp_vals) else 0.5
            conc_diff = abs(float(conc_vals.median()) - sol.oxidizer_concentration) / max(sol.oxidizer_concentration, 1.0) if len(conc_vals) else 0.5
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
                peer_preds[family] = self.repository.get_predictor(family).predict(query)
            else:
                peer_model = self.repository.get_cached_or_prebuilt_predictor(family)
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

        sol = request.recipe.solution
        temp_c = request.recipe.temperature_C
        if request.parameter_source == "manual":
            if manual_override is None:
                # Fall back gracefully to database lookup values if caller omitted manual override
                lookup = self.database.lookup_exposure_condition(
                    metal=metal,
                    pH=sol.pH,
                    oxidizer_type=sol.oxidizer_type,
                    oxidizer_concentration=sol.oxidizer_concentration,
                    temperature_C=temp_c,
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
                    temperature_C=temp_c,
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
        query = MLFeatureQuery.for_solution(metal, sol, temp_c)
        predictor = self.repository.get_calibrated_predictor(
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
            lab_ds = self.repository.get_lab_project(request.lab_project)
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

    def run_single_metal_analysis(self, request: SimulationRequest) -> SingleMetalAnalysisResult:
        """Execute end-to-end Single Metal Analysis and return a pure, Plotly-free result."""
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
        sol = request.recipe.solution
        immersion_time_s = request.recipe.immersion_time_s
        curve = generate_polarization_curve(resolved.parameters)
        warnings_list = list(resolved.warnings)
        non_aq_warning = sol.non_aqueous_warning()
        if non_aq_warning is not None:
            warnings_list.insert(0, non_aq_warning)
        recs_list = list(resolved.experimental_recommendations)

        if not has_faraday_properties(request.primary_metal, self.database.material_properties):
            warnings_list.append(
                f"Missing physical properties (density / equivalent weight) for {request.primary_metal}; "
                "Faraday corrosion rate (nm/min) and thickness-loss kinetics are unavailable. "
                "Add a row to data/raw/material_properties.csv to enable rate conversion."
            )
            return SingleMetalAnalysisResult(
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
            )

        faraday = calculate_faraday_corrosion_rate(
            metal=request.primary_metal,
            icorr_uA_cm2=resolved.parameters.icorr_uA_cm2,
            valence=valence,
            immersion_time_s=immersion_time_s,
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

        times_s = _build_time_grid(immersion_time_s)
        inhibitor_present = sol.has_inhibitor
        model_arg = None if request.kinetics_model == "auto" else request.kinetics_model
        kinetics = simulate_time_resolved_kinetics(
            faraday,
            times_s,
            model=model_arg,
            metal=request.primary_metal,
            pH=sol.pH,
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
                pH=sol.pH,
                inhibitor_present=inhibitor_present,
                material_properties=self.database.material_properties,
                d_crit_nm=request.d_crit_nm,
                d_max_nm=request.d_max_nm,
                power_law_n=request.power_law_n,
            )

        final_depth = float(kinetics.depth_nm[-1])
        final_risk = float(kinetics.risk_score[-1])
        final_tier = classify_thickness_loss_risk_tier(final_depth, request.d_crit_nm)

        return SingleMetalAnalysisResult(
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
        )

    def run_galvanic_analysis(self, request: SimulationRequest) -> GalvanicAnalysisResult:
        """Execute end-to-end Bimetallic Galvanic Analysis and return a pure, Plotly-free result."""
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

        sol = request.recipe.solution
        temp_c = request.recipe.temperature_C
        immersion_time_s = request.recipe.immersion_time_s

        # In Database mode, try to pair both metals on a shared Solution if their independent
        # nearest-neighbor lookups landed on different solution_ids or temperatures.
        extra_warnings: list[str] = []
        non_aq_warning = sol.non_aqueous_warning()
        if non_aq_warning is not None:
            extra_warnings.append(non_aq_warning)
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
                    f"approximated at the requested Exposure Condition ({temp_c:.1f}°C). "
                    f"Consider switching to ML Prediction mode for matched-condition estimation."
                )
                extra_recs.append(
                    f"Measure polarization curves for both {request.primary_metal} and {request.secondary_metal} "
                    f"in the same Solution (pH {sol.pH:.1f}, {sol.oxidizer_type} "
                    f"{sol.oxidizer_concentration:g}, {temp_c:.1f}°C) to validate galvanic coupling."
                )
                solve_params_a = replace(
                    solve_params_a, solution_id=None, temperature_C=temp_c
                )
                solve_params_b = replace(
                    solve_params_b, solution_id=None, temperature_C=temp_c
                )

        driving_force = screen_galvanic_driving_force(
            metal_a=request.primary_metal,
            Ecorr_a_V_SHE=solve_params_a.Ecorr_V_SHE,
            metal_b=request.secondary_metal,
            Ecorr_b_V_SHE=solve_params_b.Ecorr_V_SHE,
            temperature_C=temp_c,
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
            immersion_time_s=immersion_time_s,
            material_properties=self.database.material_properties,
            allow_missing_faraday=True,
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
            return GalvanicAnalysisResult(
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

        times_s = _build_time_grid(immersion_time_s)
        inhibitor_present = sol.has_inhibitor
        model_arg = None if request.kinetics_model == "auto" else request.kinetics_model

        uncoupled_kinetics = simulate_time_resolved_kinetics(
            coupling.uncoupled_faraday,
            times_s,
            model=model_arg,
            metal=coupling.active_metal,
            pH=sol.pH,
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
            pH=sol.pH,
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
                pH=sol.pH,
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

        return GalvanicAnalysisResult(
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
        )
