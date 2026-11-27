"""Multi-recipe comparison engine and multi-format report exporter (HTML, PDF, CSV, JSON, PNG, SVG)."""

from __future__ import annotations

import base64
import csv
import html
import io
import json
import math
import textwrap
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

from corrosion_simulator.database import SolventRegime, SolventSubclass, classify_solvent_subclass
from corrosion_simulator.lab_calibration import CALIBRATION_MODE_LABELS, ResidualBreakdown
from corrosion_simulator.physics import generate_polarization_curve
from corrosion_simulator.platform_controller import (
    CorrosionPlatformController,
    GalvanicViewPayload,
    SimulationRequest,
    SingleMetalViewPayload,
    resolve_consistent_solvent_settings,
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
    solvent_regime: SolventRegime | None = None
    organic_solvent_type: str = "None"
    water_content_wt_pct: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("CleaningRecipeSpec.name must be non-empty.")
        if self.immersion_time_s <= 0 or not np.isfinite(self.immersion_time_s):
            raise ValueError("immersion_time_s must be strictly positive.")
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
        """Canonical base solvent label for this Cleaning Recipe."""
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
            "solvent_regime": str(self.recipe.solvent_regime),
            "solvent_subclass": str(self.recipe.solvent_subclass),
            "organic_solvent_type": str(self.recipe.organic_solvent_type),
            "water_content_wt_pct": float(self.recipe.water_content_wt_pct),
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
            solvent_regime=recipe.solvent_regime,
            organic_solvent_type=recipe.organic_solvent_type,
            water_content_wt_pct=float(recipe.water_content_wt_pct),
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
            "Solvent Regime": str(e.recipe.solvent_regime),
            "Solvent Subclass": str(e.recipe.solvent_subclass),
            "Co-Solvent": str(e.recipe.organic_solvent_type),
            "Water (wt%)": round(float(e.recipe.water_content_wt_pct), 2),
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
        e_band = curve.parameters.Ecorr_V_SHE + band.overpotential_V
        p25 = np.maximum(band.eta_log10_abs_net_p25, cusp_floor)
        p75 = np.maximum(band.eta_log10_abs_net_p75, cusp_floor)
        ax_tafel.fill_between(
            e_band,
            p25,
            p75,
            color="#1f77b4",
            alpha=0.28,
            edgecolor="#1f77b4",
            linewidth=0.8,
            label="Net |i_net| 50% Band (p25–p75)",
        )
    ax_tafel.plot(
        curve.potential_V_SHE,
        curve.log10_anodic_uA_cm2,
        color="#d62728",
        linestyle="--",
        linewidth=1.3,
        label="Anodic Branch (i_a)",
    )
    ax_tafel.plot(
        curve.potential_V_SHE,
        curve.log10_cathodic_uA_cm2,
        color="#2ca02c",
        linestyle=":",
        linewidth=1.3,
        label="Cathodic Branch (i_c)",
    )
    ax_tafel.plot(curve.potential_V_SHE, net_log10, color="#1f77b4", linewidth=2.0, label="Net |i_net|")
    x_ecorr = float(curve.parameters.Ecorr_V_SHE)
    y_icorr = float(math.log10(curve.parameters.icorr_uA_cm2))
    ax_tafel.scatter(
        [x_ecorr],
        [y_icorr],
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
    ax_tafel.legend(fontsize=7.5, loc="best")

    ax_kin = fig.add_subplot(n_rows, 2, 2)
    kin = single_payload.kinetics_result
    if kin is not None:
        ax_kin.plot(kin.times_s, kin.depth_nm, color="#1f77b4", linestyle="-", linewidth=2.0, label=f"Depth ({kin.model}) (left)")
        if single_payload.mc_kinetics is not None:
            ax_kin.fill_between(
                kin.times_s,
                single_payload.mc_kinetics.depth_p25_nm,
                single_payload.mc_kinetics.depth_p75_nm,
                color="#1f77b4",
                alpha=0.28,
                label="Depth 50% Band (left)",
            )
        ax_kin.axhline(kin.d_crit_nm, color="#475569", linestyle="--", linewidth=1.2, label=f"d_crit={kin.d_crit_nm:.2f} nm (left)")
        ax_kin.set_title(f"{single_payload.request.primary_metal} Corrosion Depth & Risk")
        ax_kin.set_xlabel("Immersion Time (s)")
        ax_kin.set_ylabel("Corrosion Depth (nm)", color="#1f77b4")
        ax_kin.set_ylim(bottom=0.0)
        ax_kin.grid(True, alpha=0.3)

        ax_risk = ax_kin.twinx()
        if single_payload.mc_kinetics is not None:
            ax_risk.fill_between(
                kin.times_s,
                single_payload.mc_kinetics.risk_p25,
                single_payload.mc_kinetics.risk_p75,
                color="#d95f02",
                alpha=0.22,
                label="Risk 50% Band (right)",
            )
        ax_risk.plot(kin.times_s, kin.risk_score, color="#d95f02", linestyle="-.", linewidth=1.8, label="Risk R(t) (right)")
        ax_risk.set_ylabel("Thickness Loss Risk R(t)", color="#d95f02")
        ax_risk.set_ylim(0.0, 1.05)
        h_left, l_left = ax_kin.get_legend_handles_labels()
        h_right, l_right = ax_risk.get_legend_handles_labels()
        ax_kin.legend(h_left + h_right, l_left + l_right, fontsize=7.5, loc="upper left")
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
            if g.mc_coupled_kinetics is not None:
                ax_gd.fill_between(
                    g.coupled_kinetics.times_s,
                    g.mc_coupled_kinetics.depth_p25_nm,
                    g.mc_coupled_kinetics.depth_p75_nm,
                    color="#1f77b4",
                    alpha=0.28,
                    label="Coupled Depth 50% Band",
                )
            ax_gd.plot(g.uncoupled_kinetics.times_s, g.uncoupled_kinetics.depth_nm, "--", color="#64748b", label=f"{g.active_metal} Uncoupled")
            ax_gd.plot(g.coupled_kinetics.times_s, g.coupled_kinetics.depth_nm, "-", color="#1f77b4", linewidth=2.2, label=f"{g.active_metal} Coupled ({g.acceleration_factor:.2f}×)")
            ax_gd.axhline(g.request.d_crit_nm, color="#475569", linestyle="--", linewidth=1.0, label=f"d_crit={g.request.d_crit_nm:.2f} nm")
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


def _collect_calibration_metadata(
    single_payload: SingleMetalViewPayload,
    galvanic_payload: GalvanicViewPayload | None = None,
) -> dict[str, Any]:
    """Collect active calibration mode, lab project name, and ResidualBreakdown summaries."""
    req = single_payload.request
    is_ml = req.parameter_source == "ml"
    cal_mode = str(req.calibration_mode) if is_ml else None
    lab_proj = single_payload.resolved_parameters.lab_project if is_ml else None
    if lab_proj is None and is_ml and galvanic_payload is not None:
        lab_proj = galvanic_payload.primary_resolved.lab_project

    breakdowns: list[tuple[str, ResidualBreakdown]] = []
    if single_payload.resolved_parameters.residual_breakdown is not None:
        breakdowns.append(
            (
                f"Single Metal ({req.primary_metal})",
                single_payload.resolved_parameters.residual_breakdown,
            )
        )
    if galvanic_payload is not None:
        if galvanic_payload.primary_resolved.residual_breakdown is not None:
            breakdowns.append(
                (
                    f"Galvanic ({galvanic_payload.primary_resolved.metal})",
                    galvanic_payload.primary_resolved.residual_breakdown,
                )
            )
        if galvanic_payload.secondary_resolved.residual_breakdown is not None:
            breakdowns.append(
                (
                    f"Galvanic ({galvanic_payload.secondary_resolved.metal})",
                    galvanic_payload.secondary_resolved.residual_breakdown,
                )
            )
    return {
        "is_ml": is_ml,
        "calibration_mode": cal_mode,
        "calibration_mode_label": (
            CALIBRATION_MODE_LABELS.get(req.calibration_mode, req.calibration_mode)
            if is_ml
            else None
        ),
        "lab_project": lab_proj,
        "breakdowns": breakdowns,
    }


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


_METHODOLOGY_GUIDE_ENTRIES: tuple[dict[str, str], ...] = (
    {
        "Topic": "4 Electrochemical Parameters (E_corr, i_corr, β_a, β_c) & Generation",
        "Definition": (
            "E_corr (V vs. SHE): open-circuit corrosion potential where anodic = cathodic current. "
            "i_corr (µA/cm²): corrosion current density at E_corr. "
            "β_a, β_c (mV/dec): anodic & cathodic Tafel slopes. "
            "Generated via: (1) ML Prediction (ensemble / GP trained on harmonized database in log10(i_corr) space, "
            "optionally calibrated with lab data via Delta-Learning or Lab-Only GP), "
            "(2) Database Lookup (nearest experimental record), or (3) Manual Input."
        ),
        "Formula": "i_net(E) = i_corr · [10^((E − E_corr)/β_a) − 10^(−(E − E_corr)/β_c)]  (Wagner–Traud, β in V/dec)",
    },
    {
        "Topic": "Initial Corrosion Rate (CR₀)",
        "Definition": (
            "Instantaneous thickness-loss rate at t = 0 (nm/min or mm/yr) derived from dissolution current density "
            "(i = i_corr for single metal; i = i_galv for coupled anode) via Faraday's law."
        ),
        "Formula": (
            "CR₀ (nm/min) = [(i · 10⁻⁶ · EW) / (F · ρ)] · 10⁷ · 60,  "
            "EW = M / z (g/eq), F = 96,485.33 C/mol, ρ = density (g/cm³), z = valence"
        ),
    },
    {
        "Topic": "Corrosion Depth d(t)",
        "Definition": (
            "Cumulative metal thickness lost (nm) over immersion time t (s), anchored to initial velocity v₀ = CR₀ / 60 (nm/s). "
            "Selected automatically from Pourbaix stability or manually."
        ),
        "Formula": (
            "Linear: d(t) = v₀·t  |  "
            "Passivation: d(t) = d_max·[1 − exp(−v₀·t / d_max)]  |  "
            "Power-Law: d(t) = (v₀·t_ref)·(t / t_ref)^n  (t_ref = 60 s, 0.3 ≤ n ≤ 1.0)"
        ),
    },
    {
        "Topic": "Critical Thickness Threshold (d_crit) & Time to Threshold (t_crit)",
        "Definition": (
            "d_crit (nm, default 1.0 nm): maximum tolerable thickness loss before process/device impact. "
            "t_crit (s): immersion time required for d(t) to reach d_crit (∞ if d_max ≤ d_crit in passivation)."
        ),
        "Formula": (
            "Linear: t_crit = d_crit / v₀  |  "
            "Passivation: t_crit = −(d_max / v₀)·ln(1 − d_crit / d_max)  |  "
            "Power-Law: t_crit = t_ref·[d_crit / (v₀·t_ref)]^(1/n)"
        ),
    },
    {
        "Topic": "Thickness Loss Risk Score R(t) & Risk Tiers",
        "Definition": (
            "Bounded risk score R(t) ∈ [0, 1] calibrated so R(d_crit) = 0.50. "
            "Discrete Risk Tiers by ratio r = d(t) / d_crit: Low (r < 0.8), Medium (0.8 ≤ r < 2.0), "
            "High (2.0 ≤ r < 4.0), Critical (r ≥ 4.0)."
        ),
        "Formula": "R(t) = 1 − exp(−ln(2) · d(t) / d_crit)",
    },
    {
        "Topic": "Galvanic Coupling (E_galv, i_galv, Acceleration Factor AF)",
        "Definition": (
            "Active Metal (Anode, lower E_corr) dissolves faster when coupled to Noble Metal (Cathode, higher E_corr). "
            "Mixed potential E_galv satisfies area-weighted charge conservation; i_galv is the anodic dissolution current density on Active Metal. "
            "Galvanic Driving Force Risk by |ΔE_corr|: Low (< 0.1 V), Medium (0.1–0.3 V), High (≥ 0.3 V)."
        ),
        "Formula": (
            "i_net,active(E_galv) + (A_cathode / A_anode)·i_net,noble(E_galv) = 0;  "
            "i_galv = i_anodic,active(E_galv);  AF = i_galv / i_corr,active"
        ),
    },
    {
        "Topic": "Uncertainty Bands (50% Band) & Confidence Tier",
        "Definition": (
            "Monte Carlo propagation samples ML parameter uncertainties using log-normal Tafel slopes, "
            "mass-transport limiting current cap (i_lim = 1e5 µA/cm²), shared-solution E_corr correlation (ρ = 0.6), "
            "and fixed-anode tracking (AF < 1 when polarity flips) to form the shaded 50% Band (p25–p75 interquartile range). "
            "Confidence Tier (High / Medium / Low) indicates Applicability Domain coverage (bounding box + local k-NN density)."
        ),
        "Formula": "50% Band = [p25, p75] of Monte Carlo ensemble;  90% CI summary = [p05, p95];  i_galv ≤ i_lim",
    },
)


def _methodology_guide_dataframe() -> pd.DataFrame:
    """Return a display-ready DataFrame of report definitions and mathematical formulas."""
    return pd.DataFrame(list(_METHODOLOGY_GUIDE_ENTRIES))


def _format_galvanic_param_table_text(galvanic_payload: GalvanicViewPayload) -> list[str]:
    """Format Anode, Cathode & Mixed electrochemical parameters as compact aligned monospace lines (<=86 chars) for PDF."""
    if galvanic_payload.primary_resolved.metal == galvanic_payload.active_metal:
        active_src = galvanic_payload.primary_resolved.source_label
        noble_src = galvanic_payload.secondary_resolved.source_label
    else:
        active_src = galvanic_payload.secondary_resolved.source_label
        noble_src = galvanic_payload.primary_resolved.source_label

    active_params = galvanic_payload.coupling_result.active_parameters
    noble_params = galvanic_payload.coupling_result.noble_parameters
    a_ratio = galvanic_payload.request.cathode_to_anode_area_ratio

    rows = [
        (
            f"Anode ({galvanic_payload.active_metal})",
            f"{active_params.Ecorr_V_SHE:.4f}",
            f"{active_params.icorr_uA_cm2:.3g}",
            f"{active_params.beta_a_mV_dec:.1f}",
            f"{active_params.beta_c_mV_dec:.1f}",
            active_src,
        ),
        (
            f"Cathode ({galvanic_payload.noble_metal})",
            f"{noble_params.Ecorr_V_SHE:.4f}",
            f"{noble_params.icorr_uA_cm2:.3g}",
            f"{noble_params.beta_a_mV_dec:.1f}",
            f"{noble_params.beta_c_mV_dec:.1f}",
            noble_src,
        ),
        (
            f"Mixed (A={a_ratio:g})",
            f"{galvanic_payload.mixed_potential_V_SHE:.4f}",
            f"{galvanic_payload.galvanic_current_density_uA_cm2:.3g}",
            "-",
            "-",
            "Solver",
        ),
    ]
    lines = [
        "  Galvanic Electrochemical Parameters (Anode, Cathode & Mixed Point):",
        f"    {'Role':<18} {'E(V SHE)':>9} {'i(uA/cm2)':>10} {'ba':>6} {'bc':>6}  Source",
        "    " + "-" * 74,
    ]
    for role, e_s, i_s, ba_s, bc_s, src_s in rows:
        lines.append(f"    {role[:18]:<18} {e_s:>9} {i_s:>10} {ba_s:>6} {bc_s:>6}  {src_s[:22]}")
    return lines


def _write_paginated_text_pages(
    pdf: PdfPages,
    lines: Sequence[str],
    *,
    max_lines_per_page: int = 58,
    fontsize: float = 7.8,
    linespacing: float = 1.38,
    wrap_width: int = 92,
) -> None:
    """Render monospace text lines across one or more letter-sized PDF pages without bottom clipping."""
    expanded: list[str] = []
    for raw_line in lines:
        if not raw_line:
            expanded.append("")
        elif len(raw_line) <= wrap_width:
            expanded.append(raw_line)
        else:
            indent = " " * (len(raw_line) - len(raw_line.lstrip(" ")) + 2)
            expanded.extend(
                textwrap.wrap(
                    raw_line,
                    width=wrap_width,
                    subsequent_indent=indent,
                    break_long_words=False,
                )
            )

    for start in range(0, max(len(expanded), 1), max_lines_per_page):
        page_lines = expanded[start : start + max_lines_per_page]
        page_fig = Figure(figsize=(8.5, 11.0), dpi=150)
        FigureCanvasAgg(page_fig)
        ax = page_fig.add_axes([0.07, 0.05, 0.86, 0.90])
        ax.axis("off")
        ax.text(
            0.0,
            0.98,
            "\n".join(page_lines),
            va="top",
            ha="left",
            family="monospace",
            fontsize=fontsize,
            linespacing=linespacing,
        )
        pdf.savefig(page_fig)


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
    cal_meta = _collect_calibration_metadata(single_payload, galvanic_payload)
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
            "solvent_regime": str(req.solvent_regime),
            "solvent_subclass": str(req.solvent_subclass),
            "organic_solvent_type": str(req.organic_solvent_type),
            "water_content_wt_pct": float(req.water_content_wt_pct),
            "immersion_time_s": float(req.immersion_time_s),
            "d_crit_nm": float(req.d_crit_nm),
            "kinetics_model": req.kinetics_model,
            "parameter_source": req.parameter_source,
            "calibration_mode": cal_meta["calibration_mode"],
            "lab_project": cal_meta["lab_project"],
        },
        "single_metal": single_payload.to_summary_dict(),
        "single_metal_parameters": single_payload.parameter_table.to_dict(orient="records"),
        "uncertainty_intervals": _build_uncertainty_intervals(single_payload, galvanic_payload),
        "warnings": warnings,
        "experimental_recommendations": recs,
    }
    if cal_meta["is_ml"] and cal_meta["calibration_mode"] != "literature_only":
        lab_cal_obj: dict[str, Any] = {
            "calibration_mode": cal_meta["calibration_mode"],
            "calibration_mode_label": cal_meta["calibration_mode_label"],
            "lab_project": cal_meta["lab_project"],
        }
        if single_payload.resolved_parameters.residual_breakdown is not None:
            lab_cal_obj["single_metal_residual_breakdown"] = (
                single_payload.resolved_parameters.residual_breakdown.to_dict()
            )
        if galvanic_payload is not None and (
            galvanic_payload.primary_resolved.residual_breakdown is not None
            or galvanic_payload.secondary_resolved.residual_breakdown is not None
        ):
            lab_cal_obj["galvanic_residual_breakdown"] = {
                galvanic_payload.primary_resolved.metal: (
                    galvanic_payload.primary_resolved.residual_breakdown.to_dict()
                    if galvanic_payload.primary_resolved.residual_breakdown is not None
                    else None
                ),
                galvanic_payload.secondary_resolved.metal: (
                    galvanic_payload.secondary_resolved.residual_breakdown.to_dict()
                    if galvanic_payload.secondary_resolved.residual_breakdown is not None
                    else None
                ),
            }
        payload["lab_calibration"] = lab_cal_obj
    if galvanic_payload is not None:
        payload["galvanic"] = galvanic_payload.to_summary_dict()
        payload["galvanic_electrochemical_parameters"] = (
            galvanic_payload.electrochemical_parameter_table.where(
                pd.notna(galvanic_payload.electrochemical_parameter_table), None
            ).to_dict(orient="records")
        )
    if comparison_payload is not None:
        payload["recipe_comparison"] = comparison_payload.to_summary_dict()
    payload["methodology_and_formulas"] = list(_METHODOLOGY_GUIDE_ENTRIES)

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
    cal_meta = _collect_calibration_metadata(single_payload, galvanic_payload)
    s_rows = [
        ("Inputs", "Active Query", "Primary Metal", req.primary_metal, "", src),
        ("Inputs", "Active Query", "Secondary Metal", req.secondary_metal or "None", "", src),
        ("Inputs", "Active Query", "Cathode-to-Anode Area Ratio", req.cathode_to_anode_area_ratio, "", src),
        ("Inputs", "Active Query", "Oxidizer", f"{req.oxidizer_type} ({req.oxidizer_concentration:g})", "", src),
        ("Inputs", "Active Query", "Acid / Complexing Agent", f"{req.acid_type} ({req.acid_concentration:g})", "", src),
        ("Inputs", "Active Query", "Corrosion Inhibitor", f"{req.inhibitor_type} ({req.inhibitor_concentration:g})", "", src),
        ("Inputs", "Active Query", "pH", req.pH, "", src),
        ("Inputs", "Active Query", "Temperature", req.temperature_C, "°C", src),
        ("Inputs", "Active Query", "solvent_regime", str(req.solvent_regime), "", src),
        ("Inputs", "Active Query", "solvent_subclass", str(req.solvent_subclass), "", src),
        ("Inputs", "Active Query", "organic_solvent_type", str(req.organic_solvent_type), "", src),
        ("Inputs", "Active Query", "water_content_wt_pct", float(req.water_content_wt_pct), "wt%", src),
        ("Inputs", "Active Query", "Immersion Time", req.immersion_time_s, "s", src),
        ("Inputs", "Active Query", "Critical Thickness Threshold (d_crit)", req.d_crit_nm, "nm", src),
    ]
    if cal_meta["is_ml"]:
        s_rows.extend(
            [
                ("Inputs", "Active Query", "calibration_mode", cal_meta["calibration_mode"], "", src),
                ("Inputs", "Active Query", "lab_project", cal_meta["lab_project"] or "None", "", src),
            ]
        )
    s_rows.extend(
        [
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
    )
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
        active_params = galvanic_payload.coupling_result.active_parameters
        noble_params = galvanic_payload.coupling_result.noble_parameters
        g_rows = [
            ("Galvanic", "Active Query", "Active Metal", galvanic_payload.active_metal, "", src),
            ("Galvanic", "Active Query", "Noble Metal", galvanic_payload.noble_metal, "", src),
            ("Galvanic", "Active Query", f"Anode ({galvanic_payload.active_metal}) E_corr", active_params.Ecorr_V_SHE, "V vs. SHE", src),
            ("Galvanic", "Active Query", f"Anode ({galvanic_payload.active_metal}) i_corr", active_params.icorr_uA_cm2, "µA/cm²", src),
            ("Galvanic", "Active Query", f"Anode ({galvanic_payload.active_metal}) beta_a", active_params.beta_a_mV_dec, "mV/dec", src),
            ("Galvanic", "Active Query", f"Anode ({galvanic_payload.active_metal}) beta_c", active_params.beta_c_mV_dec, "mV/dec", src),
            ("Galvanic", "Active Query", f"Cathode ({galvanic_payload.noble_metal}) E_corr", noble_params.Ecorr_V_SHE, "V vs. SHE", src),
            ("Galvanic", "Active Query", f"Cathode ({galvanic_payload.noble_metal}) i_corr", noble_params.icorr_uA_cm2, "µA/cm²", src),
            ("Galvanic", "Active Query", f"Cathode ({galvanic_payload.noble_metal}) beta_a", noble_params.beta_a_mV_dec, "mV/dec", src),
            ("Galvanic", "Active Query", f"Cathode ({galvanic_payload.noble_metal}) beta_c", noble_params.beta_c_mV_dec, "mV/dec", src),
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

    if cal_meta["breakdowns"]:
        for scope_label, bd in cal_meta["breakdowns"]:
            for row in bd.to_dataframe().to_dict(orient="records"):
                writer.writerow(
                    {
                        "section": "Residual Breakdown",
                        "recipe": scope_label,
                        "metric": f"{row['Parameter']} ({row['Method']}, n={row['Lab Samples (n)']})",
                        "value": (
                            f"prior={row['Prior Estimate']:.4g} ± {row['Prior ±1σ']:.3g}, "
                            f"delta={row['Lab Residual (Δ)']:+.4g} ± {row['Residual ±1σ']:.3g}, "
                            f"calibrated={row['Calibrated Final']:.4g} ± {row['Combined ±1σ']:.3g}"
                        ),
                        "unit": row["Unit"],
                        "provenance": src,
                    }
                )

    if comparison_payload is not None:
        for e in comparison_payload.entries:
            prov = e.parameter_source.upper()
            cmp_rows = [
                ("Recipe Comparison", e.recipe.name, "Rank", e.rank, "", prov),
                ("Recipe Comparison", e.recipe.name, "solvent_regime", str(e.recipe.solvent_regime), "", prov),
                ("Recipe Comparison", e.recipe.name, "solvent_subclass", str(e.recipe.solvent_subclass), "", prov),
                ("Recipe Comparison", e.recipe.name, "organic_solvent_type", str(e.recipe.organic_solvent_type), "", prov),
                ("Recipe Comparison", e.recipe.name, "water_content_wt_pct", float(e.recipe.water_content_wt_pct), "wt%", prov),
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


def export_report_raw_data_csv(
    single_payload: SingleMetalViewPayload,
    *,
    galvanic_payload: GalvanicViewPayload | None = None,
    comparison_payload: RecipeComparisonPayload | None = None,
) -> bytes:
    """Export raw simulation curve data for all report charts in tidy CSV format."""
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["chart", "series", "x_name", "x_value", "x_unit", "y_name", "y_value", "y_unit"],
    )
    writer.writeheader()

    def _emit_series(
        chart: str,
        series: str,
        x_name: str,
        xs: Sequence[Any] | np.ndarray,
        x_unit: str,
        y_name: str,
        ys: Sequence[Any] | np.ndarray,
        y_unit: str,
    ) -> None:
        for xv, yv in zip(xs, ys):
            y_out: Any = "" if (isinstance(yv, (float, np.floating)) and not math.isfinite(float(yv))) else yv
            writer.writerow(
                {
                    "chart": chart,
                    "series": series,
                    "x_name": x_name,
                    "x_value": xv,
                    "x_unit": x_unit,
                    "y_name": y_name,
                    "y_value": y_out,
                    "y_unit": y_unit,
                }
            )

    # 1. Single Metal Polarization Curve (Tafel)
    curve = single_payload.polarization_curve
    metal = single_payload.request.primary_metal
    tafel_chart = f"Single Metal Polarization Curve ({metal})"
    _emit_series(tafel_chart, f"{metal} Net |i_net|", "Potential E", curve.potential_V_SHE, "V vs. SHE", "log10|i|", curve.log10_abs_net_uA_cm2, "log10(µA/cm²)")
    _emit_series(tafel_chart, "Anodic Branch (i_a)", "Potential E", curve.potential_V_SHE, "V vs. SHE", "log10(i_a)", curve.log10_anodic_uA_cm2, "log10(µA/cm²)")
    _emit_series(tafel_chart, "Cathodic Branch (i_c)", "Potential E", curve.potential_V_SHE, "V vs. SHE", "log10(i_c)", curve.log10_cathodic_uA_cm2, "log10(µA/cm²)")
    if single_payload.mc_propagation is not None and single_payload.mc_propagation.polarization_band is not None:
        band = single_payload.mc_propagation.polarization_band
        e_band = curve.parameters.Ecorr_V_SHE + band.overpotential_V
        _emit_series(tafel_chart, "Net |i_net| 50% Band p25", "Potential E", e_band, "V vs. SHE", "log10|i|", band.eta_log10_abs_net_p25, "log10(µA/cm²)")
        _emit_series(tafel_chart, "Net |i_net| 50% Band p75", "Potential E", e_band, "V vs. SHE", "log10|i|", band.eta_log10_abs_net_p75, "log10(µA/cm²)")

    # 2. Single Metal Corrosion Depth & Risk
    kin = single_payload.kinetics_result
    if kin is not None:
        kin_chart = f"Single Metal Depth & Risk ({metal})"
        _emit_series(kin_chart, f"Depth ({kin.model})", "Immersion Time", kin.times_s, "s", "Corrosion Depth", kin.depth_nm, "nm")
        _emit_series(kin_chart, "Thickness Loss Risk R(t)", "Immersion Time", kin.times_s, "s", "Risk R(t)", kin.risk_score, "unitless")
        if single_payload.mc_kinetics is not None:
            mc_k = single_payload.mc_kinetics
            _emit_series(kin_chart, "Depth 50% Band p25", "Immersion Time", kin.times_s, "s", "Corrosion Depth", mc_k.depth_p25_nm, "nm")
            _emit_series(kin_chart, "Depth 50% Band p75", "Immersion Time", kin.times_s, "s", "Corrosion Depth", mc_k.depth_p75_nm, "nm")
            _emit_series(kin_chart, "Risk 50% Band p25", "Immersion Time", kin.times_s, "s", "Risk R(t)", mc_k.risk_p25, "unitless")
            _emit_series(kin_chart, "Risk 50% Band p75", "Immersion Time", kin.times_s, "s", "Risk R(t)", mc_k.risk_p75, "unitless")

    # 3. Galvanic Polarization Overlay & Depth/Risk
    if galvanic_payload is not None:
        g = galvanic_payload
        coupling = g.coupling_result
        e_min = min(coupling.active_parameters.Ecorr_V_SHE, coupling.noble_parameters.Ecorr_V_SHE) - 0.15
        e_max = max(coupling.active_parameters.Ecorr_V_SHE, coupling.noble_parameters.Ecorr_V_SHE) + 0.15
        potentials = np.linspace(e_min, e_max, 241)
        active_c = generate_polarization_curve(coupling.active_parameters, potentials_V_SHE=potentials)
        noble_c = generate_polarization_curve(coupling.noble_parameters, potentials_V_SHE=potentials)
        noble_scaled = np.log10(np.maximum(noble_c.abs_net_current_uA_cm2 * coupling.cathode_to_anode_area_ratio, 1e-12))

        gt_chart = f"Galvanic Polarization Overlay ({g.active_metal} vs. {g.noble_metal})"
        _emit_series(gt_chart, f"Active: {g.active_metal} |i_net|", "Potential E", potentials, "V vs. SHE", "log10|i|", active_c.log10_abs_net_uA_cm2, "log10(µA/cm²)")
        _emit_series(gt_chart, f"Noble: {g.noble_metal} ({coupling.cathode_to_anode_area_ratio:g}x |i_net|)", "Potential E", potentials, "V vs. SHE", "log10(A_ratio*|i|)", noble_scaled, "log10(µA/cm²)")
        _emit_series(gt_chart, "Galvanic Operating Point", "Potential E", [g.mixed_potential_V_SHE], "V vs. SHE", "log10(i_galv)", [math.log10(max(g.galvanic_current_density_uA_cm2, 1e-12))], "log10(µA/cm²)")

        if g.uncoupled_kinetics is not None and g.coupled_kinetics is not None:
            gd_chart = f"Galvanic Depth & Risk ({g.active_metal})"
            t_g = g.coupled_kinetics.times_s
            _emit_series(gd_chart, f"{g.active_metal} Uncoupled Depth", "Immersion Time", t_g, "s", "Corrosion Depth", g.uncoupled_kinetics.depth_nm, "nm")
            _emit_series(gd_chart, f"{g.active_metal} Coupled Depth", "Immersion Time", t_g, "s", "Corrosion Depth", g.coupled_kinetics.depth_nm, "nm")
            _emit_series(gd_chart, f"{g.active_metal} Uncoupled Risk R(t)", "Immersion Time", t_g, "s", "Risk R(t)", g.uncoupled_kinetics.risk_score, "unitless")
            _emit_series(gd_chart, f"{g.active_metal} Coupled Risk R(t)", "Immersion Time", t_g, "s", "Risk R(t)", g.coupled_kinetics.risk_score, "unitless")
            if g.mc_coupled_kinetics is not None:
                mc_g = g.mc_coupled_kinetics
                _emit_series(gd_chart, "Coupled Depth 50% Band p25", "Immersion Time", t_g, "s", "Corrosion Depth", mc_g.depth_p25_nm, "nm")
                _emit_series(gd_chart, "Coupled Depth 50% Band p75", "Immersion Time", t_g, "s", "Corrosion Depth", mc_g.depth_p75_nm, "nm")
                _emit_series(gd_chart, "Coupled Risk 50% Band p25", "Immersion Time", t_g, "s", "Risk R(t)", mc_g.risk_p25, "unitless")
                _emit_series(gd_chart, "Coupled Risk 50% Band p75", "Immersion Time", t_g, "s", "Risk R(t)", mc_g.risk_p75, "unitless")

    # 4. Recipe Comparison
    if comparison_payload is not None:
        cmp_chart = "Recipe Comparison"
        r_names = [e.recipe.name for e in comparison_payload.entries]
        _emit_series(cmp_chart, "Initial Corrosion Rate CR_0", "Recipe", r_names, "", "CR_0", [e.cr0_nm_min for e in comparison_payload.entries], "nm/min")
        _emit_series(cmp_chart, "Final Corrosion Depth", "Recipe", r_names, "", "Corrosion Depth", [e.corrosion_depth_nm for e in comparison_payload.entries], "nm")
        _emit_series(cmp_chart, "Thickness Loss Risk Score R(t)", "Recipe", r_names, "", "Risk R(t)", [e.risk_score for e in comparison_payload.entries], "unitless")
        _emit_series(cmp_chart, "Time to Threshold", "Recipe", r_names, "", "Time to Threshold", [e.time_to_threshold_s for e in comparison_payload.entries], "s")
        if comparison_payload.mode == "galvanic":
            _emit_series(cmp_chart, "Acceleration Factor", "Recipe", r_names, "", "Acceleration Factor", [e.acceleration_factor or 1.0 for e in comparison_payload.entries], "x")

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
        # Executive Summary & Provenance (auto-paginated)
        req = single_payload.request
        params = single_payload.resolved_parameters.parameters
        intervals = _build_uncertainty_intervals(single_payload, galvanic_payload)
        cal_meta = _collect_calibration_metadata(single_payload, galvanic_payload)

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
            f"  Solvent System: solvent_regime={req.solvent_regime}, solvent_subclass={req.solvent_subclass}, organic_solvent_type={req.organic_solvent_type}, water_content_wt_pct={float(req.water_content_wt_pct):.2f} wt%",
            f"  pH: {req.pH:.2f}   |   Temperature: {req.temperature_C:.1f} °C   |   Immersion Time: {req.immersion_time_s:.1f} s   |   d_crit: {req.d_crit_nm:.2f} nm",
            f"  Parameter Source: {req.parameter_source.upper()}   |   Kinetics Model: {single_kin_model}",
            *(
                [
                    f"  Lab Calibration: calibration_mode={cal_meta['calibration_mode']} ({cal_meta['calibration_mode_label']}), lab_project={cal_meta['lab_project'] or 'None'}"
                ]
                if cal_meta["is_ml"]
                else []
            ),
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
                    *_format_galvanic_param_table_text(g),
                ]
            )

        if comparison_payload is not None:
            lines.extend(["", "4. RECIPE COMPARISON RANKING (LOWEST RISK FIRST)"])
            lines.append(f"  {comparison_payload.recommended_rationale}")
            for e in comparison_payload.entries:
                af_txt = f" | AF={e.acceleration_factor:.2f}×" if e.acceleration_factor is not None else ""
                lines.append(
                    f"  #{e.rank} {e.recipe.name} [{e.recipe.solvent_regime}/{e.recipe.solvent_subclass}, "
                    f"co-solvent={e.recipe.organic_solvent_type}, water={float(e.recipe.water_content_wt_pct):.1f}wt%]: "
                    f"Risk={e.risk_score:.3f} ({e.risk_tier}), "
                    f"Depth={e.corrosion_depth_nm:.3f} nm, CR_0={e.cr0_nm_min:.3g} nm/min{af_txt}"
                )

        if cal_meta["breakdowns"]:
            lines.extend(["", "LAB CALIBRATION — RESIDUAL BREAKDOWN"])
            for scope_label, bd in cal_meta["breakdowns"]:
                lines.append(f"  [{scope_label}]")
                for row in bd.to_dataframe().to_dict(orient="records"):
                    lines.append(
                        f"    {row['Parameter']} ({row['Unit']}): prior={row['Prior Estimate']:.4g}±{row['Prior ±1σ']:.3g}, "
                        f"delta={row['Lab Residual (Δ)']:+.4g}±{row['Residual ±1σ']:.3g}, "
                        f"final={row['Calibrated Final']:.4g}±{row['Combined ±1σ']:.3g} "
                        f"[{row['Method']}, n={row['Lab Samples (n)']}]"
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

        cal_subject = (
            f"; calibration_mode={cal_meta['calibration_mode']}; lab_project={cal_meta['lab_project'] or 'None'}"
            if cal_meta["is_ml"] and cal_meta["calibration_mode"] != "literature_only"
            else ""
        )
        pdf.infodict()["Subject"] = (
            f"solvent_regime={req.solvent_regime}; solvent_subclass={req.solvent_subclass}; "
            f"organic_solvent_type={req.organic_solvent_type}; "
            f"water_content_wt_pct={float(req.water_content_wt_pct):.2f}{cal_subject}"
        )
        pdf.infodict()["Keywords"] = (
            f"solvent_regime={req.solvent_regime}, solvent_subclass={req.solvent_subclass}, "
            f"organic_solvent_type={req.organic_solvent_type}, "
            f"water_content_wt_pct={float(req.water_content_wt_pct):.2f}"
            + (
                f", calibration_mode={cal_meta['calibration_mode']}, lab_project={cal_meta['lab_project'] or 'None'}"
                if cal_meta["is_ml"] and cal_meta["calibration_mode"] != "literature_only"
                else ""
            )
        )
        _write_paginated_text_pages(pdf, lines, max_lines_per_page=58, fontsize=7.8, linespacing=1.38)

        # Figures Page
        comp_fig = _figure or _render_composite_figure(
            single_payload,
            galvanic_payload=galvanic_payload,
            comparison_payload=comparison_payload,
        )
        pdf.savefig(comp_fig, bbox_inches="tight")

        # Methodology, Definitions & Mathematical Formulas (Quick-Start Guide)
        guide_lines = [
            "METHODOLOGY, DEFINITIONS & MATHEMATICAL FORMULAS (QUICK-START GUIDE)",
            "=" * 78,
            "Brief reference explaining how electrochemical parameters are generated and",
            "how each corrosion metric in this report is defined and calculated.",
            "",
        ]
        for idx, entry in enumerate(_METHODOLOGY_GUIDE_ENTRIES, start=1):
            guide_lines.append(f"{idx}. {entry['Topic']}")
            guide_lines.append(f"   {entry['Definition']}")
            guide_lines.append(f"   Formula: {entry['Formula']}")
            guide_lines.append("")

        _write_paginated_text_pages(pdf, guide_lines, max_lines_per_page=58, fontsize=7.6, linespacing=1.38, wrap_width=86)

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
    cal_meta = _collect_calibration_metadata(single_payload, galvanic_payload)

    cal_meta_html = ""
    if cal_meta["is_ml"]:
        cal_meta_html = (
            f"<br/><strong>Lab Calibration:</strong> "
            f"<code>calibration_mode={html.escape(str(cal_meta['calibration_mode']))}</code> "
            f"({html.escape(str(cal_meta['calibration_mode_label']))}) | "
            f"<strong>Lab Project:</strong> <code>{html.escape(str(cal_meta['lab_project'] or 'None'))}</code>"
        )

    residual_section = ""
    if cal_meta["breakdowns"]:
        blocks = []
        for scope_label, bd in cal_meta["breakdowns"]:
            bd_html = bd.to_dataframe().to_html(index=False, classes="data-table", border=0)
            blocks.append(f"<h3>{html.escape(scope_label)}</h3>{bd_html}")
        residual_section = f"""
        <section>
          <h2>Lab Calibration — Residual Breakdown</h2>
          {''.join(blocks)}
        </section>
        """

    galv_section = ""
    if galvanic_payload is not None:
        g = galvanic_payload
        galv_param_table_html = g.electrochemical_parameter_table.to_html(
            index=False,
            na_rep="—",
            classes="data-table",
            border=0,
        )
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
          <h3>Galvanic Electrochemical Parameters (Anode, Cathode &amp; Mixed Point)</h3>
          {galv_param_table_html}
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
    <strong>Solvent System:</strong> <code>solvent_regime={html.escape(str(req.solvent_regime))}</code>, <code>solvent_subclass={html.escape(str(req.solvent_subclass))}</code>, <code>organic_solvent_type={html.escape(str(req.organic_solvent_type))}</code>, <code>water_content_wt_pct={float(req.water_content_wt_pct):.2f} wt%</code><br/>
    <strong>pH:</strong> {req.pH:.2f} | <strong>Temp:</strong> {req.temperature_C:.1f} °C | <strong>Immersion Time:</strong> {req.immersion_time_s:.1f} s | <strong>d_crit:</strong> {req.d_crit_nm:.2f} nm{cal_meta_html}
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

  {residual_section}
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

  <section>
    <h2>Methodology, Definitions &amp; Mathematical Formulas (Quick-Start Guide)</h2>
    <p>Concise reference explaining how the four electrochemical parameters are generated and how each corrosion metric in this report is calculated.</p>
    {_methodology_guide_dataframe().to_html(index=False, classes="data-table", border=0)}
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
        "raw_data_csv": export_report_raw_data_csv(
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
