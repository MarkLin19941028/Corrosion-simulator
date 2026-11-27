"""Interactive Plotly chart builders for Single Metal and Galvanic Corrosion Analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from corrosion_simulator.galvanic import (
    GalvanicCouplingResult,
    MonteCarloGalvanicResult,
)
from corrosion_simulator.kinetics import (
    MonteCarloKineticsResult,
    TimeResolvedKineticsResult,
)
from corrosion_simulator.ml_predictor import MonteCarloPropagationResult
from corrosion_simulator.physics import (
    PolarizationCurve,
    generate_polarization_curve,
)

if TYPE_CHECKING:
    from corrosion_simulator.platform_controller import (
        GalvanicAnalysisResult,
        SingleMetalAnalysisResult,
    )


@dataclass(frozen=True)
class SingleMetalFigures:
    """Interactive Plotly figure bundle for a Single Metal Analysis result."""

    polarization_figure: go.Figure
    depth_figure: go.Figure
    risk_figure: go.Figure
    kinetics_risk_figure: go.Figure | None = None


@dataclass(frozen=True)
class GalvanicFigures:
    """Interactive Plotly figure bundle for a Galvanic Analysis result."""

    galvanic_polarization_figure: go.Figure
    galvanic_depth_figure: go.Figure
    galvanic_risk_figure: go.Figure
    comparative_kinetics_figure: go.Figure | None = None


def _build_unavailable_kinetics_placeholder(*, title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14},
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=360,
    )
    return fig


def build_single_metal_polarization_figure(
    curve: PolarizationCurve,
    *,
    mc_result: MonteCarloPropagationResult | None = None,
    title: str | None = None,
) -> go.Figure:
    """Build an interactive Plotly Tafel Polarization Curve figure with optional 50% (p25–p75) band."""
    fig = go.Figure()
    metal = curve.parameters.metal or "Metal"

    net_log10 = np.array(curve.log10_abs_net_uA_cm2, dtype=float, copy=True)
    finite_mask = np.isfinite(net_log10)
    if np.any(finite_mask):
        cusp_floor = float(np.min(net_log10[finite_mask])) - 0.75
        net_log10[~finite_mask] = cusp_floor
        y_min = cusp_floor - 0.4
        y_max = float(np.max(net_log10[finite_mask])) + 1.2
    else:
        cusp_floor = -2.0
        y_min, y_max = -2.0, 4.0

    if mc_result is not None and mc_result.polarization_band is not None:
        band = mc_result.polarization_band
        e_band = curve.parameters.Ecorr_V_SHE + band.overpotential_V
        p25 = np.maximum(band.eta_log10_abs_net_p25, cusp_floor)
        p75 = np.maximum(band.eta_log10_abs_net_p75, cusp_floor)
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
    fig.add_trace(
        go.Scatter(
            x=[x_ecorr],
            y=[y_icorr],
            mode="markers",
            marker={"color": "#111111", "size": 9, "symbol": "diamond"},
            name=f"(E_corr, i_corr) = ({x_ecorr:.3f} V, {curve.parameters.icorr_uA_cm2:.2g} µA/cm²)",
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


def _add_50pct_band_trace(
    fig: go.Figure,
    times: np.ndarray,
    p25: np.ndarray,
    p75: np.ndarray,
    *,
    label_prefix: str,
    fillcolor_50: str,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add a 50% (p25–p75) interquartile uncertainty band trace."""
    loc_kwargs = {"row": row, "col": col} if row is not None and col is not None else {}
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


def apply_depth_yaxis_scale(
    fig: go.Figure,
    scale: Literal["linear", "log"] = "linear",
    *,
    log_floor_nm: float = 1e-4,
) -> go.Figure:
    """Return a copy of a Corrosion Depth Plotly figure formatted for linear or log Y-axis display."""
    if scale not in ("linear", "log"):
        raise ValueError(f"Unsupported depth Y-axis scale {scale!r}. Expected 'linear' or 'log'.")
    out = go.Figure(fig)
    if scale == "linear":
        out.update_yaxes(type="linear", rangemode="tozero")
        return out

    for trace in out.data:
        y_vals = getattr(trace, "y", None)
        if y_vals is None:
            continue
        y_arr = np.asarray(y_vals, dtype=float)
        if np.any(y_arr <= 0.0):
            trace.y = np.maximum(y_arr, log_floor_nm)
    out.update_yaxes(type="log", rangemode="normal")
    return out


def _apply_kinetics_panel_layout(
    fig: go.Figure,
    *,
    title_text: str,
    yaxis_title: str,
    yaxis: dict[str, Any],
) -> None:
    """Apply consistent top-title and top-legend layout for side-by-side Depth/Risk figures."""
    fig.update_layout(
        title={
            "text": title_text,
            "y": 0.96,
            "x": 0.0,
            "xanchor": "left",
            "yanchor": "top",
        },
        xaxis_title="Immersion Time (s)",
        yaxis_title=yaxis_title,
        yaxis=yaxis,
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        margin={"l": 55, "r": 20, "t": 90, "b": 55},
        height=380,
    )


def _add_single_metal_depth_traces(
    fig: go.Figure,
    kinetics: TimeResolvedKineticsResult,
    *,
    mc_kinetics: MonteCarloKineticsResult | None = None,
    row: int | None = None,
    col: int | None = None,
) -> None:
    times = kinetics.times_s
    metal = kinetics.metal or "Metal"
    loc_kwargs = {"row": row, "col": col} if row is not None and col is not None else {}

    if mc_kinetics is not None:
        _add_50pct_band_trace(
            fig,
            times,
            mc_kinetics.depth_p25_nm,
            mc_kinetics.depth_p75_nm,
            label_prefix="Depth",
            fillcolor_50="rgba(31, 119, 180, 0.28)",
            row=row,
            col=col,
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
        **loc_kwargs,
    )
    fig.add_hline(
        y=kinetics.d_crit_nm,
        line_dash="dash",
        line_color="#475569",
        annotation_text=f"d_crit = {kinetics.d_crit_nm:.2f} nm",
        annotation_position="top left",
        **loc_kwargs,
    )


def _add_single_metal_risk_traces(
    fig: go.Figure,
    kinetics: TimeResolvedKineticsResult,
    *,
    mc_kinetics: MonteCarloKineticsResult | None = None,
    row: int | None = None,
    col: int | None = None,
) -> None:
    times = kinetics.times_s
    loc_kwargs = {"row": row, "col": col} if row is not None and col is not None else {}

    if mc_kinetics is not None:
        _add_50pct_band_trace(
            fig,
            times,
            mc_kinetics.risk_p25,
            mc_kinetics.risk_p75,
            label_prefix="Risk",
            fillcolor_50="rgba(217, 95, 2, 0.28)",
            row=row,
            col=col,
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
        **loc_kwargs,
    )
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="#94a3b8",
        annotation_text="R(d_crit) = 0.50",
        annotation_position="bottom right",
        **loc_kwargs,
    )


def build_single_metal_depth_figure(
    kinetics: TimeResolvedKineticsResult,
    *,
    mc_kinetics: MonteCarloKineticsResult | None = None,
    title: str | None = None,
) -> go.Figure:
    """Build standalone interactive Plotly Corrosion Depth figure with 50% uncertainty band."""
    fig = go.Figure()
    metal = kinetics.metal or "Metal"
    _add_single_metal_depth_traces(fig, kinetics, mc_kinetics=mc_kinetics)
    _apply_kinetics_panel_layout(
        fig,
        title_text=title or f"{metal} Corrosion Depth vs. Immersion Time ({kinetics.model})",
        yaxis_title="Corrosion Depth (nm)",
        yaxis={"rangemode": "tozero"},
    )
    return fig


def build_single_metal_risk_figure(
    kinetics: TimeResolvedKineticsResult,
    *,
    mc_kinetics: MonteCarloKineticsResult | None = None,
    title: str | None = None,
) -> go.Figure:
    """Build standalone interactive Plotly Thickness Loss Risk figure with 50% uncertainty band."""
    fig = go.Figure()
    metal = kinetics.metal or "Metal"
    _add_single_metal_risk_traces(fig, kinetics, mc_kinetics=mc_kinetics)
    _apply_kinetics_panel_layout(
        fig,
        title_text=title or f"{metal} Thickness Loss Risk vs. Immersion Time ({kinetics.model})",
        yaxis_title="Thickness Loss Risk R(t)",
        yaxis={"range": [0.0, 1.05]},
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
    metal = kinetics.metal or "Metal"
    _add_single_metal_depth_traces(fig, kinetics, mc_kinetics=mc_kinetics, row=1, col=1)
    _add_single_metal_risk_traces(fig, kinetics, mc_kinetics=mc_kinetics, row=1, col=2)

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


def _add_galvanic_depth_traces(
    fig: go.Figure,
    uncoupled_kinetics: TimeResolvedKineticsResult,
    coupled_kinetics: TimeResolvedKineticsResult,
    *,
    mc_coupled_kinetics: MonteCarloKineticsResult | None = None,
    metal: str,
    row: int | None = None,
    col: int | None = None,
) -> None:
    times = coupled_kinetics.times_s
    loc_kwargs = {"row": row, "col": col} if row is not None and col is not None else {}

    if mc_coupled_kinetics is not None:
        _add_50pct_band_trace(
            fig,
            times,
            mc_coupled_kinetics.depth_p25_nm,
            mc_coupled_kinetics.depth_p75_nm,
            label_prefix="Coupled Depth",
            fillcolor_50="rgba(31, 119, 180, 0.28)",
            row=row,
            col=col,
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
        **loc_kwargs,
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
        **loc_kwargs,
    )
    fig.add_hline(
        y=coupled_kinetics.d_crit_nm,
        line_dash="dash",
        line_color="#475569",
        annotation_text=f"d_crit = {coupled_kinetics.d_crit_nm:.2f} nm",
        annotation_position="top left",
        **loc_kwargs,
    )


def _add_galvanic_risk_traces(
    fig: go.Figure,
    uncoupled_kinetics: TimeResolvedKineticsResult,
    coupled_kinetics: TimeResolvedKineticsResult,
    *,
    mc_coupled_kinetics: MonteCarloKineticsResult | None = None,
    metal: str,
    row: int | None = None,
    col: int | None = None,
) -> None:
    times = coupled_kinetics.times_s
    loc_kwargs = {"row": row, "col": col} if row is not None and col is not None else {}

    if mc_coupled_kinetics is not None:
        _add_50pct_band_trace(
            fig,
            times,
            mc_coupled_kinetics.risk_p25,
            mc_coupled_kinetics.risk_p75,
            label_prefix="Coupled Risk",
            fillcolor_50="rgba(217, 95, 2, 0.28)",
            row=row,
            col=col,
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
        **loc_kwargs,
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
        **loc_kwargs,
    )
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="#94a3b8",
        annotation_text="R(d_crit) = 0.50",
        annotation_position="bottom right",
        **loc_kwargs,
    )


def build_galvanic_depth_comparison_figure(
    uncoupled_kinetics: TimeResolvedKineticsResult,
    coupled_kinetics: TimeResolvedKineticsResult,
    *,
    mc_coupled_kinetics: MonteCarloKineticsResult | None = None,
    active_metal: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Build standalone comparative uncoupled vs. coupled Corrosion Depth figure with 50% band."""
    metal = active_metal or coupled_kinetics.metal or "Active Metal"
    fig = go.Figure()
    _add_galvanic_depth_traces(
        fig,
        uncoupled_kinetics,
        coupled_kinetics,
        mc_coupled_kinetics=mc_coupled_kinetics,
        metal=metal,
    )
    _apply_kinetics_panel_layout(
        fig,
        title_text=title or f"{metal} Corrosion Depth: Uncoupled vs. Coupled ({coupled_kinetics.model})",
        yaxis_title="Corrosion Depth (nm)",
        yaxis={"rangemode": "tozero"},
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
    """Build standalone comparative uncoupled vs. coupled Thickness Loss Risk figure with 50% band."""
    metal = active_metal or coupled_kinetics.metal or "Active Metal"
    fig = go.Figure()
    _add_galvanic_risk_traces(
        fig,
        uncoupled_kinetics,
        coupled_kinetics,
        mc_coupled_kinetics=mc_coupled_kinetics,
        metal=metal,
    )
    _apply_kinetics_panel_layout(
        fig,
        title_text=title or f"{metal} Thickness Loss Risk: Uncoupled vs. Coupled ({coupled_kinetics.model})",
        yaxis_title="Thickness Loss Risk R(t)",
        yaxis={"range": [0.0, 1.05]},
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
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"{metal} Corrosion Depth: Uncoupled vs. Coupled",
            f"{metal} Thickness Loss Risk: Uncoupled vs. Coupled",
        ),
        horizontal_spacing=0.12,
    )
    _add_galvanic_depth_traces(
        fig,
        uncoupled_kinetics,
        coupled_kinetics,
        mc_coupled_kinetics=mc_coupled_kinetics,
        metal=metal,
        row=1,
        col=1,
    )
    _add_galvanic_risk_traces(
        fig,
        uncoupled_kinetics,
        coupled_kinetics,
        mc_coupled_kinetics=mc_coupled_kinetics,
        metal=metal,
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


def build_single_metal_figures(
    result: SingleMetalAnalysisResult,
    *,
    include_combined: bool = False,
) -> SingleMetalFigures:
    """Build interactive Plotly figures for a SingleMetalAnalysisResult on demand."""
    metal = result.request.primary_metal
    source_upper = result.resolved_parameters.source.upper()
    pol_fig = build_single_metal_polarization_figure(
        result.polarization_curve,
        mc_result=result.mc_propagation,
        title=f"{metal} Polarization Curve ({source_upper})",
    )
    if result.kinetics_result is None:
        placeholder = _build_unavailable_kinetics_placeholder(
            title=f"{metal} Time-Resolved Kinetics (Unavailable)",
            message=(
                f"Faraday corrosion rate & kinetics unavailable for {metal}<br>"
                "(missing density / equivalent_weight_g_eq in material_properties.csv)"
            ),
        )
        return SingleMetalFigures(
            polarization_figure=pol_fig,
            depth_figure=placeholder,
            risk_figure=placeholder,
            kinetics_risk_figure=placeholder if include_combined else None,
        )

    kinetics = result.kinetics_result
    depth_fig = build_single_metal_depth_figure(
        kinetics,
        mc_kinetics=result.mc_kinetics,
        title=f"{metal} Corrosion Depth vs. Immersion Time ({kinetics.model})",
    )
    risk_fig = build_single_metal_risk_figure(
        kinetics,
        mc_kinetics=result.mc_kinetics,
        title=f"{metal} Thickness Loss Risk vs. Immersion Time ({kinetics.model})",
    )
    kin_fig = (
        build_single_metal_kinetics_figure(
            kinetics,
            mc_kinetics=result.mc_kinetics,
            title=f"{metal} Time-Resolved Kinetics & Thickness Loss Risk ({kinetics.model})",
        )
        if include_combined
        else None
    )
    return SingleMetalFigures(
        polarization_figure=pol_fig,
        depth_figure=depth_fig,
        risk_figure=risk_fig,
        kinetics_risk_figure=kin_fig,
    )


def build_galvanic_figures(
    result: GalvanicAnalysisResult,
    *,
    include_combined: bool = False,
) -> GalvanicFigures:
    """Build interactive Plotly figures for a GalvanicAnalysisResult on demand."""
    coupling = result.coupling_result
    pol_fig = build_galvanic_polarization_figure(
        coupling,
        mc_galvanic=result.mc_galvanic,
    )
    if result.uncoupled_kinetics is None or result.coupled_kinetics is None:
        placeholder = _build_unavailable_kinetics_placeholder(
            title=f"Galvanic Kinetics Comparison — {coupling.active_metal} (Unavailable)",
            message=(
                f"Coupled Faraday corrosion rate & kinetics unavailable for {coupling.active_metal}<br>"
                "(missing density / equivalent_weight_g_eq in material_properties.csv)"
            ),
        )
        return GalvanicFigures(
            galvanic_polarization_figure=pol_fig,
            galvanic_depth_figure=placeholder,
            galvanic_risk_figure=placeholder,
            comparative_kinetics_figure=placeholder if include_combined else None,
        )

    depth_fig = build_galvanic_depth_comparison_figure(
        result.uncoupled_kinetics,
        result.coupled_kinetics,
        mc_coupled_kinetics=result.mc_coupled_kinetics,
        active_metal=coupling.active_metal,
    )
    risk_fig = build_galvanic_risk_comparison_figure(
        result.uncoupled_kinetics,
        result.coupled_kinetics,
        mc_coupled_kinetics=result.mc_coupled_kinetics,
        active_metal=coupling.active_metal,
    )
    comp_kin_fig = (
        build_galvanic_kinetics_comparison_figure(
            result.uncoupled_kinetics,
            result.coupled_kinetics,
            mc_coupled_kinetics=result.mc_coupled_kinetics,
            active_metal=coupling.active_metal,
        )
        if include_combined
        else None
    )
    return GalvanicFigures(
        galvanic_polarization_figure=pol_fig,
        galvanic_depth_figure=depth_fig,
        galvanic_risk_figure=risk_fig,
        comparative_kinetics_figure=comp_kin_fig,
    )
