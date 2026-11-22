"""Application controller and interactive Plotly figure builders for the Corrosion Simulator UI."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from corrosion_simulator.database import (
    CORE_METALS,
    ExposureCondition,
    ExposureLookupResult,
    HarmonizedDatabase,
    normalize_category_label,
)
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
from corrosion_simulator.ml_predictor import (
    SUPPORTED_MODEL_FAMILIES,
    ApplicabilityDomainResult,
    HybridMLPredictor,
    MLFeatureQuery,
    MLParameterPrediction,
    ModelFamily,
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
ArtifactStatus = Literal["fresh", "stale", "live_fallback", "live", "seed"]

_CONFIDENCE_RANK = {"Low": 0, "Medium": 1, "High": 2}
_CONFIDENCE_BY_RANK = {0: "Low", 1: "Medium", 2: "High"}


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
    immersion_time_s: float = 300.0
    d_crit_nm: float = DEFAULT_CRITICAL_THICKNESS_NM
    kinetics_model: KineticsSelection = "auto"
    d_max_nm: float = DEFAULT_PASSIVATION_LIMIT_NM
    power_law_n: float = DEFAULT_POWER_LAW_N
    parameter_source: ParameterSource = "ml"
    ml_model_family: ModelFamily = "random_forest"
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
        if self.ml_model_family not in SUPPORTED_MODEL_FAMILIES:
            raise ValueError(
                f"Unsupported ml_model_family '{self.ml_model_family}'. Supported: {SUPPORTED_MODEL_FAMILIES}"
            )
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

    def to_parameter_table(self) -> pd.DataFrame:
        """Build a display-ready parameter summary table with ±1σ uncertainties."""
        source_label = {
            "database": "Database",
            "ml": "ML",
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

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the single-metal analysis."""
        return {
            "primary_metal": self.request.primary_metal,
            "parameter_source": self.request.parameter_source,
            "ml_model_family": (
                self.request.ml_model_family if self.request.parameter_source == "ml" else None
            ),
            "kinetics_model": self.kinetics_result.model if self.kinetics_result is not None else "unavailable",
            "pH": float(self.request.pH),
            "temperature_C": float(self.request.temperature_C),
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

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the galvanic coupling analysis."""
        return {
            "primary_metal": self.request.primary_metal,
            "secondary_metal": self.request.secondary_metal,
            "active_metal": self.active_metal,
            "noble_metal": self.noble_metal,
            "cathode_to_anode_area_ratio": float(self.request.cathode_to_anode_area_ratio),
            "parameter_source": self.request.parameter_source,
            "ml_model_family": (
                self.request.ml_model_family if self.request.parameter_source == "ml" else None
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
        fig.add_trace(
            go.Scatter(
                x=band.potential_V_SHE,
                y=band.log10_abs_net_p95,
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
                name="p95",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=band.potential_V_SHE,
                y=band.log10_abs_net_p05,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(31, 119, 180, 0.22)",
                name="90% Uncertainty Band (p05–p95)",
                hovertemplate="E: %{x:.3f} V vs. SHE<br>log10|i|: %{y:.2f}<extra></extra>",
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
    fig.add_trace(
        go.Scatter(
            x=[curve.parameters.Ecorr_V_SHE],
            y=[math.log10(curve.parameters.icorr_uA_cm2)],
            mode="markers",
            marker={"color": "#111111", "size": 9, "symbol": "diamond"},
            name=f"(E_corr, i_corr) = ({curve.parameters.Ecorr_V_SHE:.3f} V, {curve.parameters.icorr_uA_cm2:.2g} µA/cm²)",
        )
    )

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
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([times, times[::-1]]),
                y=np.concatenate([mc_kinetics.depth_p95_nm, mc_kinetics.depth_p05_nm[::-1]]),
                fill="toself",
                fillcolor="rgba(31, 119, 180, 0.20)",
                line={"width": 0},
                name="Depth 90% Band (p05–p95)",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([times, times[::-1]]),
                y=np.concatenate([mc_kinetics.risk_p95, mc_kinetics.risk_p05[::-1]]),
                fill="toself",
                fillcolor="rgba(214, 39, 40, 0.18)",
                line={"width": 0},
                name="Risk 90% Band (p05–p95)",
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=kinetics.depth_nm,
            mode="lines",
            line={"color": "#1f77b4", "width": 2.6},
            name=f"{metal} Depth ({kinetics.model})",
            hovertemplate="t: %{x:.1f} s<br>Depth: %{y:.3f} nm<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=kinetics.d_crit_nm,
        line_dash="dash",
        line_color="#d62728",
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
            line={"color": "#d62728", "width": 2.6},
            name="Thickness Loss Risk R(t)",
            hovertemplate="t: %{x:.1f} s<br>Risk R(t): %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="#ff7f0e",
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
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([times, times[::-1]]),
                y=np.concatenate([mc_coupled_kinetics.depth_p95_nm, mc_coupled_kinetics.depth_p05_nm[::-1]]),
                fill="toself",
                fillcolor="rgba(214, 39, 40, 0.16)",
                line={"width": 0},
                name="Coupled Depth 90% Band",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([times, times[::-1]]),
                y=np.concatenate([mc_coupled_kinetics.risk_p95, mc_coupled_kinetics.risk_p05[::-1]]),
                fill="toself",
                fillcolor="rgba(214, 39, 40, 0.16)",
                line={"width": 0},
                name="Coupled Risk 90% Band",
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=uncoupled_kinetics.depth_nm,
            mode="lines",
            line={"color": "#1f77b4", "dash": "dash", "width": 2.2},
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
            line={"color": "#d62728", "width": 2.8},
            name=f"{metal} Coupled Depth",
            hovertemplate="t: %{x:.1f} s<br>Coupled Depth: %{y:.3f} nm<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=coupled_kinetics.d_crit_nm,
        line_dash="dot",
        line_color="#555555",
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
            line={"color": "#1f77b4", "dash": "dash", "width": 2.2},
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
            line={"color": "#d62728", "width": 2.8},
            name=f"{metal} Coupled Risk",
            hovertemplate="t: %{x:.1f} s<br>Coupled Risk: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="#ff7f0e",
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
        artifact_dir: str | Path | None = None,
        artifact_status: ArtifactStatus = "live",
        artifact_warning: str | None = None,
    ) -> None:
        self.database = database
        self.artifact_dir: Path | None = Path(artifact_dir) if artifact_dir is not None else None
        self.artifact_status: ArtifactStatus = artifact_status
        self.artifact_warning: str | None = artifact_warning
        if predictor is None:
            self.predictor = self._load_or_fit_predictor("random_forest")
        else:
            self.predictor = predictor
        self._predictors: dict[ModelFamily, HybridMLPredictor] = {
            self.predictor.model_family: self.predictor
        }

    def _load_or_fit_predictor(self, model_family: ModelFamily) -> HybridMLPredictor:
        if self.artifact_dir is not None:
            prebuilt_path = self.artifact_dir / "models" / f"{model_family}.joblib"
            if prebuilt_path.exists():
                return HybridMLPredictor.load(prebuilt_path)
        predictor = HybridMLPredictor(model_family=model_family, random_state=42)
        predictor.fit(self.database)
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

    @classmethod
    def from_default_data(
        cls,
        raw_dir: str | Path = "data/raw",
        *,
        processed_dir: str | Path = "data/processed",
        model_family: ModelFamily = "random_forest",
    ) -> CorrosionPlatformController:
        """Load prebuilt artifacts when available (fast path), else ingest ``raw_dir`` or seed data."""
        from corrosion_simulator.build_artifacts import check_artifacts_freshness

        raw_path = Path(raw_dir)
        proc_path = Path(processed_dir)
        status, warning_msg = check_artifacts_freshness(raw_path, proc_path)

        if status in ("fresh", "stale"):
            database = HarmonizedDatabase.from_processed_directory(proc_path)
            ctrl = cls(
                database=database,
                predictor=None,
                artifact_dir=proc_path,
                artifact_status=status,
                artifact_warning=warning_msg,
            )
            if model_family != ctrl.predictor.model_family:
                ctrl.predictor = ctrl.get_predictor(model_family)
            return ctrl

        # Fallback: build live from raw_dir (or seed if raw_dir has no CSVs)
        if raw_path.exists() and any(raw_path.rglob("*.csv")):
            database = HarmonizedDatabase.from_directory(raw_path)
            live_status = "live_fallback"
            live_warning = warning_msg
        else:
            database = HarmonizedDatabase.from_seed()
            live_status = "seed"
            live_warning = None
        predictor = HybridMLPredictor(model_family=model_family, random_state=42)
        predictor.fit(database)
        return cls(
            database=database,
            predictor=predictor,
            artifact_dir=None,
            artifact_status=live_status,
            artifact_warning=live_warning,
        )

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
            "kinetics_models": ["auto", "linear", "passivation", "power_law"],
            "parameter_sources": ["ml", "database", "manual"],
            "ml_model_families": list(SUPPORTED_MODEL_FAMILIES),
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

        # ML prediction mode
        query = self._build_ml_query(request, metal)
        predictor = self.get_predictor(request.ml_model_family)
        pred = predictor.predict(query)
        params = pred.to_polarization_parameters()
        # Approximate linear-space 1-sigma for i_corr from log10-normal distribution:
        # sd(10^X) ≈ median * ln(10) * sigma_log10
        icorr_std = float(pred.icorr_uA_cm2_median * math.log(10.0) * pred.log10_icorr_uA_cm2.std)
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
            warnings=_dedupe_preserve_order(resolved.warnings),
            experimental_recommendations=_dedupe_preserve_order(resolved.experimental_recommendations),
            polarization_figure=pol_fig,
            kinetics_risk_figure=kin_fig,
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
        )
