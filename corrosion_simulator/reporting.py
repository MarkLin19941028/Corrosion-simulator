"""Multi-recipe comparison engine and multi-format report exporter (HTML, PDF, CSV, JSON, PNG, SVG)."""

from __future__ import annotations

import base64
import csv
import html
import io
import json
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal, Sequence

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from corrosion_simulator.physics import generate_polarization_curve
from corrosion_simulator.platform_controller import (
    CorrosionPlatformController,
    GalvanicViewPayload,
    SimulationRequest,
    SingleMetalViewPayload,
)

ComparisonMode = Literal["single_metal", "galvanic"]


@dataclass(frozen=True)
class CleaningRecipeSpec:
    """Candidate Cleaning Recipe definition for side-by-side comparison."""

    name: str
    oxidizer_type: str = "H2O2"
    oxidizer_concentration: float = 1.0
    acid_type: str = "None"
    acid_concentration: float = 0.0
    inhibitor_type: str = "None"
    inhibitor_concentration: float = 0.0
    pH: float = 3.0
    temperature_C: float = 25.0
    immersion_time_s: float = 300.0

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("CleaningRecipeSpec.name must be non-empty.")
        if self.immersion_time_s <= 0 or not np.isfinite(self.immersion_time_s):
            raise ValueError("immersion_time_s must be strictly positive.")


@dataclass
class RecipeEvaluationEntry:
    """Evaluated performance and risk metrics for a single candidate Cleaning Recipe."""

    rank: int
    recipe: CleaningRecipeSpec
    mode: ComparisonMode
    metal: str
    active_metal: str | None
    noble_metal: str | None
    cr0_nm_min: float
    corrosion_depth_nm: float
    time_to_threshold_s: float
    risk_score: float
    risk_tier: str
    acceleration_factor: float | None
    confidence_tier: str
    parameter_source: str
    warnings: list[str] = field(default_factory=list)
    experimental_recommendations: list[str] = field(default_factory=list)
    single_payload: SingleMetalViewPayload | None = None
    galvanic_payload: GalvanicViewPayload | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": int(self.rank),
            "recipe_name": self.recipe.name,
            "mode": self.mode,
            "metal": self.metal,
            "active_metal": self.active_metal,
            "noble_metal": self.noble_metal,
            "oxidizer_type": self.recipe.oxidizer_type,
            "oxidizer_concentration": float(self.recipe.oxidizer_concentration),
            "acid_type": self.recipe.acid_type,
            "acid_concentration": float(self.recipe.acid_concentration),
            "inhibitor_type": self.recipe.inhibitor_type,
            "inhibitor_concentration": float(self.recipe.inhibitor_concentration),
            "pH": float(self.recipe.pH),
            "temperature_C": float(self.recipe.temperature_C),
            "immersion_time_s": float(self.recipe.immersion_time_s),
            "cr0_nm_min": float(self.cr0_nm_min),
            "corrosion_depth_nm": float(self.corrosion_depth_nm),
            "time_to_threshold_s": (
                None if not math.isfinite(self.time_to_threshold_s) else float(self.time_to_threshold_s)
            ),
            "thickness_loss_risk_score": float(self.risk_score),
            "thickness_loss_risk_tier": str(self.risk_tier),
            "acceleration_factor": (
                None if self.acceleration_factor is None else float(self.acceleration_factor)
            ),
            "confidence_tier": str(self.confidence_tier),
            "parameter_source": str(self.parameter_source),
            "warnings": list(self.warnings),
            "experimental_recommendations": list(self.experimental_recommendations),
        }


@dataclass
class RecipeComparisonPayload:
    """Complete view payload for the Recipe Comparison tab."""

    mode: ComparisonMode
    entries: list[RecipeEvaluationEntry]
    recommended_recipe: str
    recommended_rationale: str
    comparison_table: pd.DataFrame
    bar_figure: go.Figure
    radar_figure: go.Figure

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "recommended_recipe": self.recommended_recipe,
            "recommended_rationale": self.recommended_rationale,
            "rankings": [entry.to_dict() for entry in self.entries],
        }


def _format_ttt_short(seconds: float) -> str:
    if math.isnan(seconds):
        return "N/A"
    if not math.isfinite(seconds):
        return "∞"
    if seconds >= 3600.0:
        return f"{seconds / 3600.0:.2f} hr"
    if seconds >= 60.0:
        return f"{seconds / 60.0:.2f} min"
    return f"{seconds:.1f} s"


def _fmt_metric(value: float, fmt: str = ".3g", suffix: str = "") -> str:
    if not math.isfinite(value):
        return "N/A"
    return f"{value:{fmt}}{suffix}"


def build_recipe_comparison_bar_figure(
    entries: Sequence[RecipeEvaluationEntry],
    *,
    mode: ComparisonMode,
) -> go.Figure:
    """Build grouped interactive Plotly bar chart comparing key metrics across Cleaning Recipes."""
    names = [e.recipe.name for e in entries]
    cr0_vals = [e.cr0_nm_min for e in entries]
    depth_vals = [e.corrosion_depth_nm for e in entries]
    risk_vals = [e.risk_score for e in entries]
    finite_ttts = [e.time_to_threshold_s for e in entries if math.isfinite(e.time_to_threshold_s)]
    cap_ttt = max(finite_ttts) * 1.2 if finite_ttts else 600.0
    ttt_vals = [e.time_to_threshold_s if math.isfinite(e.time_to_threshold_s) else cap_ttt for e in entries]
    ttt_labels = [_format_ttt_short(e.time_to_threshold_s) for e in entries]

    cols = 3 if mode == "galvanic" else 2
    titles = [
        "Initial Corrosion Rate CR₀ (nm/min)",
        "Final Corrosion Depth (nm)",
        "Thickness Loss Risk Score R(t)",
        "Time to Threshold (s)",
    ]
    if mode == "galvanic":
        titles.append("Acceleration Factor (×)")
        titles.append("")

    fig = make_subplots(
        rows=2,
        cols=cols,
        subplot_titles=titles,
        horizontal_spacing=0.10,
        vertical_spacing=0.22,
    )
    colors = ["#2ca02c" if e.rank == 1 else "#1f77b4" for e in entries]

    fig.add_trace(
        go.Bar(x=names, y=cr0_vals, marker_color=colors, name="CR₀ (nm/min)", showlegend=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=names, y=depth_vals, marker_color=colors, name="Corrosion Depth (nm)", showlegend=False),
        row=1,
        col=2,
    )
    risk_pos = (1, 3) if mode == "galvanic" else (2, 1)
    ttt_pos = (2, 1) if mode == "galvanic" else (2, 2)
    fig.add_trace(
        go.Bar(x=names, y=risk_vals, marker_color=colors, name="Risk Score", showlegend=False),
        row=risk_pos[0],
        col=risk_pos[1],
    )
    fig.update_yaxes(range=[0.0, 1.05], row=risk_pos[0], col=risk_pos[1])

    fig.add_trace(
        go.Bar(
            x=names,
            y=ttt_vals,
            text=ttt_labels,
            textposition="auto",
            marker_color=colors,
            name="Time to Threshold (s)",
            showlegend=False,
        ),
        row=ttt_pos[0],
        col=ttt_pos[1],
    )

    if mode == "galvanic":
        af_vals = [e.acceleration_factor or 1.0 for e in entries]
        fig.add_trace(
            go.Bar(x=names, y=af_vals, marker_color=colors, name="Acceleration Factor", showlegend=False),
            row=2,
            col=2,
        )
        fig.update_xaxes(visible=False, row=2, col=3)
        fig.update_yaxes(visible=False, row=2, col=3)

    fig.update_layout(
        title="Cleaning Recipe Comparison — Key Metrics (Recommended Lowest-Risk in Green)",
        template="plotly_white",
        margin={"l": 45, "r": 25, "t": 75, "b": 60},
        height=560,
    )
    return fig


def build_recipe_comparison_radar_figure(
    entries: Sequence[RecipeEvaluationEntry],
    *,
    mode: ComparisonMode,
) -> go.Figure:
    """Build normalized multi-axis Plotly radar chart (higher score = safer / lower corrosion)."""
    finite_cr0 = [e.cr0_nm_min for e in entries if math.isfinite(e.cr0_nm_min)]
    finite_depth = [e.corrosion_depth_nm for e in entries if math.isfinite(e.corrosion_depth_nm)]
    max_cr0 = max(finite_cr0, default=1.0) or 1.0
    max_depth = max(finite_depth, default=1.0) or 1.0
    finite_ttts = [e.time_to_threshold_s for e in entries if math.isfinite(e.time_to_threshold_s)]
    max_ttt = max(finite_ttts) if finite_ttts else 1.0
    max_af = max(((e.acceleration_factor or 1.0) for e in entries), default=1.0) or 1.0

    axes = [
        "Low CR₀",
        "Low Corrosion Depth",
        "Low Thickness Loss Risk",
        "Long Time to Threshold",
    ]
    if mode == "galvanic":
        axes.append("Low Galvanic Acceleration")

    fig = go.Figure()
    for entry in entries:
        low_cr0 = max(0.0, 1.0 - entry.cr0_nm_min / (max_cr0 * 1.1)) if math.isfinite(entry.cr0_nm_min) else 0.0
        low_depth = max(0.0, 1.0 - entry.corrosion_depth_nm / (max_depth * 1.1)) if math.isfinite(entry.corrosion_depth_nm) else 0.0
        low_risk = max(0.0, 1.0 - entry.risk_score) if math.isfinite(entry.risk_score) else 0.0
        if math.isnan(entry.time_to_threshold_s):
            long_ttt = 0.0
        elif not math.isfinite(entry.time_to_threshold_s):
            long_ttt = 1.0
        else:
            long_ttt = min(1.0, entry.time_to_threshold_s / (max_ttt * 1.05))
        values = [low_cr0, low_depth, low_risk, long_ttt]
        if mode == "galvanic":
            af = entry.acceleration_factor or 1.0
            low_af = max(0.0, 1.0 - (af - 1.0) / max(max_af - 1.0 + 0.1, 0.1))
            values.append(low_af)

        closed_axes = axes + [axes[0]]
        closed_vals = values + [values[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=closed_vals,
                theta=closed_axes,
                fill="toself",
                name=f"#{entry.rank} {entry.recipe.name}",
            )
        )

    fig.update_layout(
        title="Normalized Safety Radar (Outer = Safer / Lower Risk)",
        polar={"radialaxis": {"visible": True, "range": [0.0, 1.0]}},
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.04, "xanchor": "left", "x": 0.0},
        margin={"l": 60, "r": 60, "t": 85, "b": 45},
        height=440,
    )
    return fig


def compare_cleaning_recipes(
    controller: CorrosionPlatformController,
    base_request: SimulationRequest,
    recipes: Sequence[CleaningRecipeSpec],
) -> RecipeComparisonPayload:
    """Evaluate 2 to 5 candidate Cleaning Recipes, rank lowest-risk first, and build comparison charts."""
    if not (2 <= len(recipes) <= 5):
        raise ValueError(f"Recipe comparison requires between 2 and 5 Cleaning Recipes, got {len(recipes)}.")

    seen_names: set[str] = set()
    for recipe in recipes:
        if recipe.name in seen_names:
            raise ValueError(f"Duplicate Cleaning Recipe name '{recipe.name}'. Recipe names must be unique.")
        seen_names.add(recipe.name)

    mode: ComparisonMode = "galvanic" if base_request.secondary_metal is not None else "single_metal"
    raw_entries: list[RecipeEvaluationEntry] = []

    for recipe in recipes:
        ox_conc = 0.0 if recipe.oxidizer_type == "None" else float(recipe.oxidizer_concentration)
        acid_conc = 0.0 if recipe.acid_type == "None" else float(recipe.acid_concentration)
        inh_conc = 0.0 if recipe.inhibitor_type == "None" else float(recipe.inhibitor_concentration)

        req = replace(
            base_request,
            oxidizer_type=recipe.oxidizer_type,
            oxidizer_concentration=ox_conc,
            acid_type=recipe.acid_type,
            acid_concentration=acid_conc,
            inhibitor_type=recipe.inhibitor_type,
            inhibitor_concentration=inh_conc,
            pH=float(recipe.pH),
            temperature_C=float(recipe.temperature_C),
            immersion_time_s=float(recipe.immersion_time_s),
        )

        if mode == "galvanic":
            galv = controller.run_galvanic_analysis(req)
            raw_entries.append(
                RecipeEvaluationEntry(
                    rank=0,
                    recipe=recipe,
                    mode=mode,
                    metal=galv.active_metal,
                    active_metal=galv.active_metal,
                    noble_metal=galv.noble_metal,
                    cr0_nm_min=galv.accelerated_cr0_nm_min,
                    corrosion_depth_nm=galv.coupled_depth_nm,
                    time_to_threshold_s=galv.coupled_time_to_threshold_s,
                    risk_score=galv.coupled_risk_score,
                    risk_tier=galv.coupled_risk_tier,
                    acceleration_factor=galv.acceleration_factor,
                    confidence_tier=galv.confidence_tier,
                    parameter_source=req.parameter_source,
                    warnings=list(galv.warnings),
                    experimental_recommendations=list(galv.experimental_recommendations),
                    galvanic_payload=galv,
                )
            )
        else:
            single = controller.run_single_metal_analysis(req)
            raw_entries.append(
                RecipeEvaluationEntry(
                    rank=0,
                    recipe=recipe,
                    mode=mode,
                    metal=req.primary_metal,
                    active_metal=None,
                    noble_metal=None,
                    cr0_nm_min=single.initial_corrosion_rate_nm_min,
                    corrosion_depth_nm=single.corrosion_depth_nm,
                    time_to_threshold_s=single.time_to_threshold_s,
                    risk_score=single.thickness_loss_risk_score,
                    risk_tier=single.thickness_loss_risk_tier,
                    acceleration_factor=None,
                    confidence_tier=single.confidence_tier,
                    parameter_source=req.parameter_source,
                    warnings=list(single.warnings),
                    experimental_recommendations=list(single.experimental_recommendations),
                    single_payload=single,
                )
            )

    def _sort_key(e: RecipeEvaluationEntry) -> tuple[float, float, float]:
        r = e.risk_score if math.isfinite(e.risk_score) else float("inf")
        d = e.corrosion_depth_nm if math.isfinite(e.corrosion_depth_nm) else float("inf")
        c = e.cr0_nm_min if math.isfinite(e.cr0_nm_min) else float("inf")
        return (r, d, c)

    raw_entries.sort(key=_sort_key)
    ranked_entries = [replace(entry, rank=idx) for idx, entry in enumerate(raw_entries, start=1)]

    best = ranked_entries[0]
    af_clause = (
        f", Acceleration Factor {best.acceleration_factor:.2f}×"
        if best.acceleration_factor is not None
        else ""
    )
    rationale = (
        f"Recommended Cleaning Recipe: '{best.recipe.name}' — lowest Thickness Loss Risk score "
        f"({_fmt_metric(best.risk_score, '.3f')}, {best.risk_tier}), "
        f"final Corrosion Depth {_fmt_metric(best.corrosion_depth_nm, '.3f', ' nm')}, "
        f"CR₀ {_fmt_metric(best.cr0_nm_min, '.3g', ' nm/min')}, "
        f"Time to Threshold {_format_ttt_short(best.time_to_threshold_s)}"
        f"{af_clause} (Confidence Tier: {best.confidence_tier})."
    )

    rows: list[dict[str, Any]] = []
    for e in ranked_entries:
        row: dict[str, Any] = {
            "Rank": e.rank,
            "Cleaning Recipe": e.recipe.name,
            "pH": e.recipe.pH,
            "Temp (°C)": e.recipe.temperature_C,
            "Immersion Time (s)": e.recipe.immersion_time_s,
            "CR₀ (nm/min)": round(e.cr0_nm_min, 4) if math.isfinite(e.cr0_nm_min) else None,
            "Corrosion Depth (nm)": round(e.corrosion_depth_nm, 4) if math.isfinite(e.corrosion_depth_nm) else None,
            "Time to Threshold": _format_ttt_short(e.time_to_threshold_s),
            "Thickness Loss Risk Score": round(e.risk_score, 4) if math.isfinite(e.risk_score) else None,
            "Risk Tier": e.risk_tier,
            "Confidence Tier": e.confidence_tier,
            "Source": e.parameter_source.upper(),
        }
        if mode == "galvanic":
            row["Acceleration Factor"] = round(e.acceleration_factor or 1.0, 3)
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    bar_fig = build_recipe_comparison_bar_figure(ranked_entries, mode=mode)
    radar_fig = build_recipe_comparison_radar_figure(ranked_entries, mode=mode)

    return RecipeComparisonPayload(
        mode=mode,
        entries=ranked_entries,
        recommended_recipe=best.recipe.name,
        recommended_rationale=rationale,
        comparison_table=comparison_df,
        bar_figure=bar_fig,
        radar_figure=radar_fig,
    )


def _render_composite_figure(
    single_payload: SingleMetalViewPayload,
    *,
    galvanic_payload: GalvanicViewPayload | None = None,
    comparison_payload: RecipeComparisonPayload | None = None,
) -> Figure:
    """Render a publication-ready composite Matplotlib figure (Single Tafel + Depth/Risk + optional Galvanic Tafel/Depth + Comparison)."""
    n_rows = 1 + int(galvanic_payload is not None) + int(comparison_payload is not None)
    fig = Figure(figsize=(11.5, 4.4 * n_rows), dpi=150)
    FigureCanvasAgg(fig)

    # Row 1: Single Metal Tafel (Potential on X, log10|i| on Y) + Depth/Risk trajectory
    ax_tafel = fig.add_subplot(n_rows, 2, 1)
    curve = single_payload.polarization_curve
    net_log10 = np.array(curve.log10_abs_net_uA_cm2, dtype=float, copy=True)
    finite_mask = np.isfinite(net_log10)
    cusp_floor = float(np.min(net_log10[finite_mask])) - 0.75 if np.any(finite_mask) else -2.0
    net_log10[~finite_mask] = cusp_floor

    if single_payload.mc_propagation is not None and single_payload.mc_propagation.polarization_band is not None:
        band = single_payload.mc_propagation.polarization_band
        ax_tafel.fill_between(
            band.potential_V_SHE,
            band.log10_abs_net_p05,
            band.log10_abs_net_p95,
            color="#1f77b4",
            alpha=0.18,
            label="90% Band",
        )
    ax_tafel.plot(curve.potential_V_SHE, net_log10, color="#1f77b4", linewidth=2.0, label="Net |i_net|")
    ax_tafel.scatter(
        [curve.parameters.Ecorr_V_SHE],
        [math.log10(curve.parameters.icorr_uA_cm2)],
        color="#111111",
        marker="D",
        zorder=5,
        label="E_corr, i_corr",
    )
    ax_tafel.set_title(f"{single_payload.request.primary_metal} Polarization Curve ({single_payload.request.parameter_source.upper()})")
    ax_tafel.set_xlabel("Potential E (V vs. SHE)")
    ax_tafel.set_ylabel("log10(|i| / µA·cm⁻²)")
    ax_tafel.set_ylim(cusp_floor - 0.4, float(np.max(net_log10[finite_mask])) + 1.2 if np.any(finite_mask) else 4.0)
    ax_tafel.grid(True, alpha=0.3)
    ax_tafel.legend(fontsize=8, loc="best")

    ax_kin = fig.add_subplot(n_rows, 2, 2)
    kin = single_payload.kinetics_result
    if kin is not None:
        ax_kin.plot(kin.times_s, kin.depth_nm, color="#1f77b4", linewidth=2.0, label=f"Depth ({kin.model})")
        if single_payload.mc_kinetics is not None:
            ax_kin.fill_between(
                kin.times_s,
                single_payload.mc_kinetics.depth_p05_nm,
                single_payload.mc_kinetics.depth_p95_nm,
                color="#1f77b4",
                alpha=0.18,
                label="Depth 90% Band",
            )
        ax_kin.axhline(kin.d_crit_nm, color="#d62728", linestyle="--", linewidth=1.2, label=f"d_crit={kin.d_crit_nm:.2f} nm")
        ax_kin.set_title(f"{single_payload.request.primary_metal} Corrosion Depth & Risk")
        ax_kin.set_xlabel("Immersion Time (s)")
        ax_kin.set_ylabel("Corrosion Depth (nm)", color="#1f77b4")
        ax_kin.set_ylim(bottom=0.0)
        ax_kin.grid(True, alpha=0.3)

        ax_risk = ax_kin.twinx()
        ax_risk.plot(kin.times_s, kin.risk_score, color="#d62728", linestyle=":", linewidth=1.8, label="Risk R(t)")
        ax_risk.set_ylabel("Thickness Loss Risk R(t)", color="#d62728")
        ax_risk.set_ylim(0.0, 1.05)
        ax_kin.legend(fontsize=8, loc="upper left")
    else:
        ax_kin.axis("off")
        ax_kin.text(
            0.5,
            0.5,
            f"Corrosion Depth & Risk unavailable for {single_payload.request.primary_metal}\n"
            "(missing density / equivalent weight)",
            ha="center",
            va="center",
            fontsize=10,
            color="#475569",
        )
        ax_kin.set_title(f"{single_payload.request.primary_metal} Corrosion Depth & Risk (N/A)")

    next_row = 2
    if galvanic_payload is not None:
        g = galvanic_payload
        coupling = g.coupling_result
        e_min = min(coupling.active_parameters.Ecorr_V_SHE, coupling.noble_parameters.Ecorr_V_SHE) - 0.15
        e_max = max(coupling.active_parameters.Ecorr_V_SHE, coupling.noble_parameters.Ecorr_V_SHE) + 0.15
        potentials = np.linspace(e_min, e_max, 241)
        active_c = generate_polarization_curve(coupling.active_parameters, potentials_V_SHE=potentials)
        noble_c = generate_polarization_curve(coupling.noble_parameters, potentials_V_SHE=potentials)
        noble_scaled = np.log10(np.maximum(noble_c.abs_net_current_uA_cm2 * coupling.cathode_to_anode_area_ratio, 1e-12))

        ax_gt = fig.add_subplot(n_rows, 2, (next_row - 1) * 2 + 1)
        ax_gt.plot(potentials, active_c.log10_abs_net_uA_cm2, color="#d62728", linewidth=2.0, label=f"Active: {g.active_metal}")
        ax_gt.plot(potentials, noble_scaled, color="#1f77b4", linewidth=2.0, label=f"Noble: {g.noble_metal} ({coupling.cathode_to_anode_area_ratio:g}×)")
        ax_gt.axvline(g.mixed_potential_V_SHE, color="#9467bd", linestyle="--", linewidth=1.5, label=f"E_galv={g.mixed_potential_V_SHE:.3f} V")
        ax_gt.scatter(
            [g.mixed_potential_V_SHE],
            [math.log10(max(g.galvanic_current_density_uA_cm2, 1e-12))],
            color="#9467bd",
            marker="*",
            s=90,
            zorder=5,
        )
        ax_gt.set_title(f"Galvanic Tafel Overlay: {g.active_metal} vs. {g.noble_metal}")
        ax_gt.set_xlabel("Potential E (V vs. SHE)")
        ax_gt.set_ylabel("log10(|i| / µA·cm⁻²)")
        ax_gt.set_xlim(e_min, e_max)
        ax_gt.grid(True, alpha=0.3)
        ax_gt.legend(fontsize=8, loc="best")

        ax_gd = fig.add_subplot(n_rows, 2, (next_row - 1) * 2 + 2)
        if g.uncoupled_kinetics is not None and g.coupled_kinetics is not None:
            ax_gd.plot(g.uncoupled_kinetics.times_s, g.uncoupled_kinetics.depth_nm, "--", color="#1f77b4", label=f"{g.active_metal} Uncoupled")
            ax_gd.plot(g.coupled_kinetics.times_s, g.coupled_kinetics.depth_nm, "-", color="#d62728", linewidth=2.2, label=f"{g.active_metal} Coupled ({g.acceleration_factor:.2f}×)")
            ax_gd.axhline(g.request.d_crit_nm, color="#555555", linestyle=":", linewidth=1.0, label=f"d_crit={g.request.d_crit_nm:.2f} nm")
            ax_gd.set_title(f"Galvanic Depth: Uncoupled vs. Coupled ({g.active_metal})")
            ax_gd.set_xlabel("Immersion Time (s)")
            ax_gd.set_ylabel("Corrosion Depth (nm)")
            ax_gd.set_ylim(bottom=0.0)
            ax_gd.grid(True, alpha=0.3)
            ax_gd.legend(fontsize=8, loc="upper left")
        else:
            ax_gd.axis("off")
            ax_gd.text(
                0.5,
                0.5,
                f"Galvanic Corrosion Depth unavailable for {g.active_metal}\n"
                "(missing density / equivalent weight)",
                ha="center",
                va="center",
                fontsize=10,
                color="#475569",
            )
            ax_gd.set_title(f"Galvanic Depth: {g.active_metal} (N/A)")
        next_row += 1

    if comparison_payload is not None:
        names = [e.recipe.name for e in comparison_payload.entries]
        risks = [e.risk_score if math.isfinite(e.risk_score) else 0.0 for e in comparison_payload.entries]
        depths = [e.corrosion_depth_nm if math.isfinite(e.corrosion_depth_nm) else 0.0 for e in comparison_payload.entries]
        bar_colors = ["#2ca02c" if e.rank == 1 else "#1f77b4" for e in comparison_payload.entries]

        ax_cmp1 = fig.add_subplot(n_rows, 2, (next_row - 1) * 2 + 1)
        ax_cmp1.barh(names[::-1], risks[::-1], color=bar_colors[::-1])
        ax_cmp1.set_xlim(0.0, 1.05)
        ax_cmp1.set_xlabel("Thickness Loss Risk Score R(t)")
        ax_cmp1.set_title("Recipe Comparison — Risk Score (Lowest First)")
        ax_cmp1.grid(True, axis="x", alpha=0.3)

        ax_cmp2 = fig.add_subplot(n_rows, 2, (next_row - 1) * 2 + 2)
        ax_cmp2.barh(names[::-1], depths[::-1], color=bar_colors[::-1])
        ax_cmp2.set_xlabel("Final Corrosion Depth (nm)")
        ax_cmp2.set_title("Recipe Comparison — Final Corrosion Depth")
        ax_cmp2.grid(True, axis="x", alpha=0.3)

    fig.tight_layout(pad=2.0)
    return fig


def _collect_warnings_and_recommendations(
    single_payload: SingleMetalViewPayload,
    galvanic_payload: GalvanicViewPayload | None = None,
) -> tuple[list[str], list[str]]:
    warnings = list(single_payload.warnings)
    recs = list(single_payload.experimental_recommendations)
    if galvanic_payload is not None:
        for w in galvanic_payload.warnings:
            if w not in warnings:
                warnings.append(w)
        for r in galvanic_payload.experimental_recommendations:
            if r not in recs:
                recs.append(r)
    return warnings, recs


def _build_uncertainty_intervals(
    single_payload: SingleMetalViewPayload,
    galvanic_payload: GalvanicViewPayload | None = None,
) -> dict[str, Any]:
    intervals: dict[str, Any] = {}
    if single_payload.mc_propagation is not None:
        intervals["single_cr0_nm_min"] = dict(single_payload.mc_propagation.cr_nm_min_summary)
    if single_payload.mc_kinetics is not None:
        intervals["single_corrosion_depth_nm"] = {
            "p05": float(single_payload.mc_kinetics.depth_p05_nm[-1]),
            "p50": float(single_payload.mc_kinetics.depth_p50_nm[-1]),
            "p95": float(single_payload.mc_kinetics.depth_p95_nm[-1]),
        }
        intervals["single_time_to_threshold_s"] = dict(
            single_payload.mc_kinetics.time_to_threshold_summary
        )
    if galvanic_payload is not None and galvanic_payload.mc_galvanic is not None:
        intervals["galvanic_mixed_potential_V_SHE"] = dict(
            galvanic_payload.mc_galvanic.mixed_potential_summary
        )
        intervals["galvanic_acceleration_factor"] = dict(
            galvanic_payload.mc_galvanic.acceleration_factor_summary
        )
        intervals["galvanic_coupled_cr0_nm_min"] = dict(
            galvanic_payload.mc_galvanic.coupled_cr_nm_min_summary
        )
    if galvanic_payload is not None and galvanic_payload.mc_coupled_kinetics is not None:
        intervals["galvanic_coupled_depth_nm"] = {
            "p05": float(galvanic_payload.mc_coupled_kinetics.depth_p05_nm[-1]),
            "p50": float(galvanic_payload.mc_coupled_kinetics.depth_p50_nm[-1]),
            "p95": float(galvanic_payload.mc_coupled_kinetics.depth_p95_nm[-1]),
        }
        intervals["galvanic_coupled_time_to_threshold_s"] = dict(
            galvanic_payload.mc_coupled_kinetics.time_to_threshold_summary
        )
    return intervals


def export_report_png(
    single_payload: SingleMetalViewPayload,
    *,
    galvanic_payload: GalvanicViewPayload | None = None,
    comparison_payload: RecipeComparisonPayload | None = None,
    _figure: Figure | None = None,
) -> bytes:
    """Export a high-DPI PNG composite figure of the current analysis."""
    fig = _figure or _render_composite_figure(
        single_payload,
        galvanic_payload=galvanic_payload,
        comparison_payload=comparison_payload,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


def export_report_svg(
    single_payload: SingleMetalViewPayload,
    *,
    galvanic_payload: GalvanicViewPayload | None = None,
    comparison_payload: RecipeComparisonPayload | None = None,
    _figure: Figure | None = None,
) -> bytes:
    """Export a vector SVG composite figure of the current analysis."""
    fig = _figure or _render_composite_figure(
        single_payload,
        galvanic_payload=galvanic_payload,
        comparison_payload=comparison_payload,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    return buf.getvalue()


def export_report_json(
    single_payload: SingleMetalViewPayload,
    *,
    galvanic_payload: GalvanicViewPayload | None = None,
    comparison_payload: RecipeComparisonPayload | None = None,
) -> bytes:
    """Export complete structured JSON report with provenance, uncertainties, warnings, and rankings."""
    req = single_payload.request
    warnings, recs = _collect_warnings_and_recommendations(single_payload, galvanic_payload)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "primary_metal": req.primary_metal,
            "secondary_metal": req.secondary_metal,
            "cathode_to_anode_area_ratio": float(req.cathode_to_anode_area_ratio),
            "oxidizer_type": req.oxidizer_type,
            "oxidizer_concentration": float(req.oxidizer_concentration),
            "acid_type": req.acid_type,
            "acid_concentration": float(req.acid_concentration),
            "inhibitor_type": req.inhibitor_type,
            "inhibitor_concentration": float(req.inhibitor_concentration),
            "pH": float(req.pH),
            "temperature_C": float(req.temperature_C),
            "immersion_time_s": float(req.immersion_time_s),
            "d_crit_nm": float(req.d_crit_nm),
            "kinetics_model": req.kinetics_model,
            "parameter_source": req.parameter_source,
        },
        "single_metal": single_payload.to_summary_dict(),
        "single_metal_parameters": single_payload.parameter_table.to_dict(orient="records"),
        "uncertainty_intervals": _build_uncertainty_intervals(single_payload, galvanic_payload),
        "warnings": warnings,
        "experimental_recommendations": recs,
    }
    if galvanic_payload is not None:
        payload["galvanic"] = galvanic_payload.to_summary_dict()
    if comparison_payload is not None:
        payload["recipe_comparison"] = comparison_payload.to_summary_dict()

    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def export_report_csv(
    single_payload: SingleMetalViewPayload,
    *,
    galvanic_payload: GalvanicViewPayload | None = None,
    comparison_payload: RecipeComparisonPayload | None = None,
) -> bytes:
    """Export tabular CSV summary of inputs, metrics, uncertainties, provenance, warnings, and rankings."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["section", "recipe", "metric", "value", "unit", "provenance"])
    writer.writeheader()

    req = single_payload.request
    src = req.parameter_source.upper()
    params = single_payload.resolved_parameters.parameters
    s_rows = [
        ("Inputs", "Active Query", "Primary Metal", req.primary_metal, "", src),
        ("Inputs", "Active Query", "Secondary Metal", req.secondary_metal or "None", "", src),
        ("Inputs", "Active Query", "Cathode-to-Anode Area Ratio", req.cathode_to_anode_area_ratio, "", src),
        ("Inputs", "Active Query", "Oxidizer", f"{req.oxidizer_type} ({req.oxidizer_concentration:g})", "", src),
        ("Inputs", "Active Query", "Acid / Complexing Agent", f"{req.acid_type} ({req.acid_concentration:g})", "", src),
        ("Inputs", "Active Query", "Corrosion Inhibitor", f"{req.inhibitor_type} ({req.inhibitor_concentration:g})", "", src),
        ("Inputs", "Active Query", "pH", req.pH, "", src),
        ("Inputs", "Active Query", "Temperature", req.temperature_C, "°C", src),
        ("Inputs", "Active Query", "Immersion Time", req.immersion_time_s, "s", src),
        ("Inputs", "Active Query", "Critical Thickness Threshold (d_crit)", req.d_crit_nm, "nm", src),
        ("Single Metal", "Active Query", "E_corr", params.Ecorr_V_SHE, "V vs. SHE", src),
        ("Single Metal", "Active Query", "i_corr", params.icorr_uA_cm2, "µA/cm²", src),
        ("Single Metal", "Active Query", "beta_a", params.beta_a_mV_dec, "mV/dec", src),
        ("Single Metal", "Active Query", "beta_c", params.beta_c_mV_dec, "mV/dec", src),
        ("Single Metal", "Active Query", "Initial Corrosion Rate (CR_0)", single_payload.initial_corrosion_rate_nm_min, "nm/min", src),
        ("Single Metal", "Active Query", "Corrosion Depth", single_payload.corrosion_depth_nm, "nm", src),
        ("Single Metal", "Active Query", "Time to Threshold", _format_ttt_short(single_payload.time_to_threshold_s), "", src),
        ("Single Metal", "Active Query", "Thickness Loss Risk Score", single_payload.thickness_loss_risk_score, "", src),
        ("Single Metal", "Active Query", "Thickness Loss Risk Tier", single_payload.thickness_loss_risk_tier, "", src),
        ("Single Metal", "Active Query", "Confidence Tier", single_payload.confidence_tier, "", src),
    ]
    for sec, rec, met, val, unit, prov in s_rows:
        writer.writerow({"section": sec, "recipe": rec, "metric": met, "value": val, "unit": unit, "provenance": prov})

    intervals = _build_uncertainty_intervals(single_payload, galvanic_payload)
    for key, stats in intervals.items():
        if "p05" in stats and "p95" in stats:
            writer.writerow(
                {
                    "section": "Uncertainty (90% CI)",
                    "recipe": "Active Query",
                    "metric": key,
                    "value": f"[{stats['p05']:.4g}, {stats['p95']:.4g}]",
                    "unit": "",
                    "provenance": src,
                }
            )

    if galvanic_payload is not None:
        g_rows = [
            ("Galvanic", "Active Query", "Active Metal", galvanic_payload.active_metal, "", src),
            ("Galvanic", "Active Query", "Noble Metal", galvanic_payload.noble_metal, "", src),
            ("Galvanic", "Active Query", "Galvanic Driving Force Risk", galvanic_payload.galvanic_driving_force_risk, "", src),
            ("Galvanic", "Active Query", "Mixed Potential (E_galv)", galvanic_payload.mixed_potential_V_SHE, "V vs. SHE", src),
            ("Galvanic", "Active Query", "Galvanic Current Density (i_galv)", galvanic_payload.galvanic_current_density_uA_cm2, "µA/cm²", src),
            ("Galvanic", "Active Query", "Acceleration Factor", galvanic_payload.acceleration_factor, "×", src),
            ("Galvanic", "Active Query", "Accelerated CR_0", galvanic_payload.accelerated_cr0_nm_min, "nm/min", src),
            ("Galvanic", "Active Query", "Coupled Corrosion Depth", galvanic_payload.coupled_depth_nm, "nm", src),
            ("Galvanic", "Active Query", "Coupled Time to Threshold", _format_ttt_short(galvanic_payload.coupled_time_to_threshold_s), "", src),
            ("Galvanic", "Active Query", "Coupled Risk Tier", galvanic_payload.coupled_risk_tier, "", src),
        ]
        for sec, rec, met, val, unit, prov in g_rows:
            writer.writerow({"section": sec, "recipe": rec, "metric": met, "value": val, "unit": unit, "provenance": prov})

    warnings, recs = _collect_warnings_and_recommendations(single_payload, galvanic_payload)
    for idx, w in enumerate(warnings, start=1):
        writer.writerow({"section": "Applicability Domain Warnings", "recipe": "Active Query", "metric": f"Warning {idx}", "value": w, "unit": "", "provenance": src})
    for idx, r in enumerate(recs, start=1):
        writer.writerow({"section": "Experimental Recommendations", "recipe": "Active Query", "metric": f"Recommendation {idx}", "value": r, "unit": "", "provenance": src})

    if comparison_payload is not None:
        for e in comparison_payload.entries:
            prov = e.parameter_source.upper()
            cmp_rows = [
                ("Recipe Comparison", e.recipe.name, "Rank", e.rank, "", prov),
                ("Recipe Comparison", e.recipe.name, "Initial Corrosion Rate (CR_0)", e.cr0_nm_min, "nm/min", prov),
                ("Recipe Comparison", e.recipe.name, "Corrosion Depth", e.corrosion_depth_nm, "nm", prov),
                ("Recipe Comparison", e.recipe.name, "Time to Threshold", _format_ttt_short(e.time_to_threshold_s), "", prov),
                ("Recipe Comparison", e.recipe.name, "Thickness Loss Risk Score", e.risk_score, "", prov),
                ("Recipe Comparison", e.recipe.name, "Thickness Loss Risk Tier", e.risk_tier, "", prov),
                ("Recipe Comparison", e.recipe.name, "Confidence Tier", e.confidence_tier, "", prov),
            ]
            if e.acceleration_factor is not None:
                cmp_rows.append(
                    ("Recipe Comparison", e.recipe.name, "Acceleration Factor", e.acceleration_factor, "×", prov)
                )
            for sec, rec, met, val, unit, pr in cmp_rows:
                writer.writerow({"section": sec, "recipe": rec, "metric": met, "value": val, "unit": unit, "provenance": pr})

    return buf.getvalue().encode("utf-8")


def export_report_pdf(
    single_payload: SingleMetalViewPayload,
    *,
    galvanic_payload: GalvanicViewPayload | None = None,
    comparison_payload: RecipeComparisonPayload | None = None,
    _figure: Figure | None = None,
) -> bytes:
    """Export a multi-page technical PDF report using Matplotlib PdfPages."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1: Executive Summary & Provenance
        summary_fig = Figure(figsize=(8.5, 11.0), dpi=150)
        FigureCanvasAgg(summary_fig)
        ax = summary_fig.add_axes([0.07, 0.05, 0.86, 0.90])
        ax.axis("off")

        req = single_payload.request
        params = single_payload.resolved_parameters.parameters
        intervals = _build_uncertainty_intervals(single_payload, galvanic_payload)

        def _ci_str(key: str, unit: str = "") -> str:
            st_dict = intervals.get(key)
            if not st_dict or "p05" not in st_dict or "p95" not in st_dict:
                return ""
            suffix = f" {unit}" if unit else ""
            return f" [90% CI: {st_dict['p05']:.3g}–{st_dict['p95']:.3g}{suffix}]"

        single_kin_model = (
            single_payload.kinetics_result.model
            if single_payload.kinetics_result is not None
            else "unavailable"
        )
        lines = [
            "CORROSION SIMULATOR — TECHNICAL ANALYSIS REPORT",
            f"Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "=" * 78,
            "",
            "1. INPUT CONDITIONS & PROVENANCE",
            f"  Primary Metal: {req.primary_metal}   |   Secondary Metal: {req.secondary_metal or 'None'} (Area Ratio: {req.cathode_to_anode_area_ratio:g})",
            f"  Solution: Oxidizer={req.oxidizer_type} ({req.oxidizer_concentration:g}), Acid={req.acid_type} ({req.acid_concentration:g}), Inhibitor={req.inhibitor_type} ({req.inhibitor_concentration:g})",
            f"  pH: {req.pH:.2f}   |   Temperature: {req.temperature_C:.1f} °C   |   Immersion Time: {req.immersion_time_s:.1f} s   |   d_crit: {req.d_crit_nm:.2f} nm",
            f"  Parameter Source: {req.parameter_source.upper()}   |   Kinetics Model: {single_kin_model}",
            "",
            "2. SINGLE METAL ANALYSIS",
            f"  E_corr: {params.Ecorr_V_SHE:.4f} V vs. SHE   |   i_corr: {params.icorr_uA_cm2:.3g} µA/cm²",
            f"  beta_a: {params.beta_a_mV_dec:.1f} mV/dec   |   beta_c: {params.beta_c_mV_dec:.1f} mV/dec",
            f"  Initial Corrosion Rate (CR_0): {_fmt_metric(single_payload.initial_corrosion_rate_nm_min, '.4g', ' nm/min')}{_ci_str('single_cr0_nm_min', 'nm/min')}",
            f"  Corrosion Depth @ {req.immersion_time_s:.0f} s: {_fmt_metric(single_payload.corrosion_depth_nm, '.4f', ' nm')}{_ci_str('single_corrosion_depth_nm', 'nm')}",
            f"  Time to Threshold: {_format_ttt_short(single_payload.time_to_threshold_s)}{_ci_str('single_time_to_threshold_s', 's')}",
            f"  Thickness Loss Risk: {single_payload.thickness_loss_risk_tier} (R = {_fmt_metric(single_payload.thickness_loss_risk_score, '.3f')})   |   Confidence Tier: {single_payload.confidence_tier}",
        ]

        if galvanic_payload is not None:
            g = galvanic_payload
            lines.extend(
                [
                    "",
                    "3. BIMETALLIC GALVANIC ANALYSIS",
                    f"  Active Metal: {g.active_metal}   |   Noble Metal: {g.noble_metal}   |   Driving Force Risk: {g.galvanic_driving_force_risk}",
                    f"  Mixed Potential (E_galv): {g.mixed_potential_V_SHE:.4f} V vs. SHE{_ci_str('galvanic_mixed_potential_V_SHE', 'V')}",
                    f"  Galvanic Current Density (i_galv): {g.galvanic_current_density_uA_cm2:.3g} µA/cm²",
                    f"  Acceleration Factor: {g.acceleration_factor:.2f}×{_ci_str('galvanic_acceleration_factor', '×')}   |   Accelerated CR_0: {_fmt_metric(g.accelerated_cr0_nm_min, '.4g', ' nm/min')}",
                    f"  Coupled Depth: {_fmt_metric(g.coupled_depth_nm, '.4f', ' nm')}{_ci_str('galvanic_coupled_depth_nm', 'nm')} (Uncoupled: {_fmt_metric(g.uncoupled_depth_nm, '.4f', ' nm')})",
                    f"  Coupled Time to Threshold: {_format_ttt_short(g.coupled_time_to_threshold_s)}   |   Risk Tier: {g.coupled_risk_tier}   |   Confidence: {g.confidence_tier}",
                ]
            )

        if comparison_payload is not None:
            lines.extend(["", "4. RECIPE COMPARISON RANKING (LOWEST RISK FIRST)"])
            lines.append(f"  {comparison_payload.recommended_rationale}")
            for e in comparison_payload.entries:
                af_txt = f" | AF={e.acceleration_factor:.2f}×" if e.acceleration_factor is not None else ""
                lines.append(
                    f"  #{e.rank} {e.recipe.name}: Risk={e.risk_score:.3f} ({e.risk_tier}), "
                    f"Depth={e.corrosion_depth_nm:.3f} nm, CR_0={e.cr0_nm_min:.3g} nm/min{af_txt}"
                )

        all_warnings, all_recs = _collect_warnings_and_recommendations(single_payload, galvanic_payload)
        if all_warnings:
            lines.extend(["", "APPLICABILITY DOMAIN WARNINGS"])
            for w in all_warnings:
                lines.append(f"  * {w}")

        if all_recs:
            lines.extend(["", "EXPERIMENTAL RECOMMENDATIONS"])
            for r in all_recs:
                lines.append(f"  - {r}")

        ax.text(
            0.0,
            0.98,
            "\n".join(lines),
            va="top",
            ha="left",
            family="monospace",
            fontsize=7.8,
            linespacing=1.4,
            wrap=True,
        )
        pdf.savefig(summary_fig)

        # Page 2: Figures
        comp_fig = _figure or _render_composite_figure(
            single_payload,
            galvanic_payload=galvanic_payload,
            comparison_payload=comparison_payload,
        )
        pdf.savefig(comp_fig, bbox_inches="tight")

    return buf.getvalue()


def export_report_html(
    single_payload: SingleMetalViewPayload,
    *,
    galvanic_payload: GalvanicViewPayload | None = None,
    comparison_payload: RecipeComparisonPayload | None = None,
    _png_bytes: bytes | None = None,
) -> bytes:
    """Export a self-contained HTML5 technical report with embedded figures, metrics, provenance, and warnings."""
    png_bytes = _png_bytes or export_report_png(
        single_payload,
        galvanic_payload=galvanic_payload,
        comparison_payload=comparison_payload,
    )
    png_b64 = base64.b64encode(png_bytes).decode("ascii")

    req = single_payload.request
    param_table_html = single_payload.parameter_table.to_html(index=False, classes="data-table", border=0)

    galv_section = ""
    if galvanic_payload is not None:
        g = galvanic_payload
        galv_section = f"""
        <section>
          <h2>Bimetallic Galvanic Analysis — {html.escape(g.active_metal)} (Active) vs. {html.escape(g.noble_metal)} (Noble)</h2>
          <div class="kpi-grid">
            <div class="kpi"><span>Driving Force Risk</span><strong>{html.escape(g.galvanic_driving_force_risk)}</strong></div>
            <div class="kpi"><span>Mixed Potential (E_galv)</span><strong>{g.mixed_potential_V_SHE:.3f} V vs. SHE</strong></div>
            <div class="kpi"><span>Galvanic Current Density (i_galv)</span><strong>{g.galvanic_current_density_uA_cm2:.3g} µA/cm²</strong></div>
            <div class="kpi"><span>Acceleration Factor</span><strong>{g.acceleration_factor:.2f}×</strong></div>
            <div class="kpi"><span>Coupled Depth</span><strong>{_fmt_metric(g.coupled_depth_nm, '.3f', ' nm')}</strong></div>
            <div class="kpi"><span>Coupled Risk Tier</span><strong>{html.escape(g.coupled_risk_tier)} ({_fmt_metric(g.coupled_risk_score, '.3f')})</strong></div>
            <div class="kpi"><span>Galvanic Confidence Tier</span><strong>{html.escape(g.confidence_tier)}</strong></div>
          </div>
        </section>
        """

    intervals = _build_uncertainty_intervals(single_payload, galvanic_payload)
    ci_section = ""
    if intervals:
        ci_rows = []
        for metric_key, stats in intervals.items():
            if "p05" in stats and "p95" in stats:
                p50 = stats.get("p50", float("nan"))
                ci_rows.append(
                    {
                        "Metric": metric_key,
                        "5th Percentile (p05)": f"{stats['p05']:.4g}",
                        "Median (p50)": f"{p50:.4g}" if math.isfinite(p50) else "—",
                        "95th Percentile (p95)": f"{stats['p95']:.4g}",
                    }
                )
        if ci_rows:
            ci_table_html = pd.DataFrame(ci_rows).to_html(index=False, classes="data-table", border=0)
            ci_section = f"""
            <section>
              <h2>Monte Carlo Uncertainty Intervals (90% CI)</h2>
              {ci_table_html}
            </section>
            """

    cmp_section = ""
    if comparison_payload is not None:
        cmp_table_html = comparison_payload.comparison_table.to_html(index=False, classes="data-table", border=0)
        cmp_section = f"""
        <section>
          <h2>Recipe Comparison</h2>
          <div class="callout-success">{html.escape(comparison_payload.recommended_rationale)}</div>
          {cmp_table_html}
        </section>
        """

    all_warnings, all_recs = _collect_warnings_and_recommendations(single_payload, galvanic_payload)

    warnings_html = (
        "<ul>" + "".join(f"<li>{html.escape(w)}</li>" for w in all_warnings) + "</ul>"
        if all_warnings
        else "<p>No Applicability Domain warnings triggered.</p>"
    )
    recs_html = (
        "<ul>" + "".join(f"<li>{html.escape(r)}</li>" for r in all_recs) + "</ul>"
        if all_recs
        else "<p>No additional laboratory experiments required.</p>"
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Corrosion Simulator Report — {html.escape(req.primary_metal)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #1f2937; max-width: 1120px; margin: 24px auto; padding: 0 20px; line-height: 1.5; }}
    h1 {{ border-bottom: 2px solid #1f77b4; padding-bottom: 8px; }}
    h2 {{ color: #0f4c81; margin-top: 28px; }}
    .meta {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 18px; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 14px 0; }}
    .kpi {{ background: #f1f5f9; border-radius: 8px; padding: 12px; }}
    .kpi span {{ display: block; font-size: 0.82rem; color: #475569; }}
    .kpi strong {{ font-size: 1.15rem; color: #0f172a; }}
    .data-table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.92rem; }}
    .data-table th, .data-table td {{ border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; }}
    .data-table th {{ background: #e2e8f0; }}
    .callout-success {{ background: #ecfdf5; border-left: 4px solid #10b981; padding: 12px 14px; border-radius: 4px; margin-bottom: 12px; }}
    .callout-warn {{ background: #fffbeb; border-left: 4px solid #f59e0b; padding: 12px 14px; border-radius: 4px; }}
    img.chart {{ width: 100%; height: auto; border: 1px solid #e2e8f0; border-radius: 8px; margin-top: 10px; }}
  </style>
</head>
<body>
  <h1>Semiconductor Wet-Cleaning Corrosion Simulator Report</h1>
  <div class="meta">
    <strong>Generated:</strong> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} |
    <strong>Parameter Provenance:</strong> {html.escape(req.parameter_source.upper())} |
    <strong>Confidence Tier:</strong> {html.escape(single_payload.confidence_tier)}<br/>
    <strong>Primary Metal:</strong> {html.escape(req.primary_metal)} |
    <strong>Secondary Metal:</strong> {html.escape(req.secondary_metal or 'None')} (Area Ratio: {req.cathode_to_anode_area_ratio:g}) |
    <strong>Solution:</strong> {html.escape(req.oxidizer_type)} ({req.oxidizer_concentration:g}), {html.escape(req.acid_type)} ({req.acid_concentration:g}), {html.escape(req.inhibitor_type)} ({req.inhibitor_concentration:g}) |
    <strong>pH:</strong> {req.pH:.2f} | <strong>Temp:</strong> {req.temperature_C:.1f} °C | <strong>Immersion Time:</strong> {req.immersion_time_s:.1f} s | <strong>d_crit:</strong> {req.d_crit_nm:.2f} nm
  </div>

  <section>
    <h2>Single Metal Analysis — {html.escape(req.primary_metal)}</h2>
    <div class="kpi-grid">
      <div class="kpi"><span>Initial Corrosion Rate (CR₀)</span><strong>{_fmt_metric(single_payload.initial_corrosion_rate_nm_min, '.3g', ' nm/min')}</strong></div>
      <div class="kpi"><span>Corrosion Depth</span><strong>{_fmt_metric(single_payload.corrosion_depth_nm, '.3f', ' nm')}</strong></div>
      <div class="kpi"><span>Time to Threshold</span><strong>{html.escape(_format_ttt_short(single_payload.time_to_threshold_s))}</strong></div>
      <div class="kpi"><span>Thickness Loss Risk</span><strong>{html.escape(single_payload.thickness_loss_risk_tier)} ({_fmt_metric(single_payload.thickness_loss_risk_score, '.3f')})</strong></div>
      <div class="kpi"><span>Kinetics Model</span><strong>{html.escape(single_payload.kinetics_result.model if single_payload.kinetics_result is not None else 'unavailable')}</strong></div>
    </div>
    {param_table_html}
  </section>

  {galv_section}
  {ci_section}
  {cmp_section}

  <section>
    <h2>Diagnostic Figures</h2>
    <img class="chart" src="data:image/png;base64,{png_b64}" alt="Corrosion Simulator Figures" />
  </section>

  <section>
    <h2>Applicability Domain Warnings & Experimental Recommendations</h2>
    <div class="callout-warn">
      <strong>Warnings:</strong>
      {warnings_html}
      <strong>Suggested Follow-Up Experiments:</strong>
      {recs_html}
    </div>
  </section>
</body>
</html>
"""
    return doc.encode("utf-8")


def generate_all_report_exports(
    single_payload: SingleMetalViewPayload,
    *,
    galvanic_payload: GalvanicViewPayload | None = None,
    comparison_payload: RecipeComparisonPayload | None = None,
) -> dict[str, bytes]:
    """Generate all six downloadable report formats: HTML, PDF, CSV, JSON, PNG, and SVG."""
    shared_fig = _render_composite_figure(
        single_payload,
        galvanic_payload=galvanic_payload,
        comparison_payload=comparison_payload,
    )
    png_bytes = export_report_png(
        single_payload,
        galvanic_payload=galvanic_payload,
        comparison_payload=comparison_payload,
        _figure=shared_fig,
    )
    return {
        "html": export_report_html(
            single_payload,
            galvanic_payload=galvanic_payload,
            comparison_payload=comparison_payload,
            _png_bytes=png_bytes,
        ),
        "pdf": export_report_pdf(
            single_payload,
            galvanic_payload=galvanic_payload,
            comparison_payload=comparison_payload,
            _figure=shared_fig,
        ),
        "csv": export_report_csv(
            single_payload,
            galvanic_payload=galvanic_payload,
            comparison_payload=comparison_payload,
        ),
        "json": export_report_json(
            single_payload,
            galvanic_payload=galvanic_payload,
            comparison_payload=comparison_payload,
        ),
        "png": png_bytes,
        "svg": export_report_svg(
            single_payload,
            galvanic_payload=galvanic_payload,
            comparison_payload=comparison_payload,
            _figure=shared_fig,
        ),
    }
