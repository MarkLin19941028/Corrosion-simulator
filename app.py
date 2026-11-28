"""Interactive Streamlit dashboard for the Corrosion Simulator.

Launch with:
    .venv/bin/streamlit run app.py
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import pandas as pd
import streamlit as st

from corrosion_simulator.artifact_repository import ArtifactRepository
from corrosion_simulator.lab_calibration import CALIBRATION_MODE_LABELS, ResidualBreakdown
from corrosion_simulator.ml_predictor import MODEL_FAMILY_LABELS, ApplicabilityDomainResult
from corrosion_simulator.paths import resolve_data_dirs
from corrosion_simulator.platform_controller import (
    CALIBRATION_MODE_DEFAULT_MODELS,
    CorrosionPlatformController,
    GalvanicAnalysisResult,
    ManualElectrochemicalInput,
    SimulationRequest,
    SingleMetalAnalysisResult,
    format_active_model_label,
    resolve_active_model_family,
)
from corrosion_simulator.plotly_charts import (
    apply_depth_yaxis_scale,
    build_galvanic_figures,
    build_single_metal_figures,
)
from corrosion_simulator.reporting import (
    RecipeCandidate,
    RecipeComparisonPayload,
    compare_cleaning_recipes,
    generate_all_report_exports,
)
from corrosion_simulator.solution import (
    NON_AQUEOUS_WARNING_PREFIX,
    CleaningRecipe,
    Solution,
    water_content_slider_bounds,
)

SOURCE_LABELS = {
    "ml": "ML Prediction",
    "database": "Database Lookup",
    "manual": "Manual Input",
}

KINETICS_LABELS = {
    "auto": "Auto (Pourbaix Stability Regime)",
    "linear": "Linear",
    "passivation": "Passivation (Self-Limiting)",
    "power_law": "Power-Law (Diffusion / Inhibited)",
}

SOLVENT_REGIME_LABELS = {
    "aqueous": "Aqueous (≥ 90 wt% Water)",
    "semi_aqueous": "Semi-aqueous (1–89.9 wt% Water)",
    "non_aqueous": "Non-aqueous (< 1 wt% Water)",
}


def _render_non_aqueous_warning_banner(request: SimulationRequest) -> None:
    """Render a prominent physics-limitation banner when simulating non-aqueous (<1% water) conditions."""
    warning = request.recipe.solution.non_aqueous_warning()
    if warning is not None:
        st.warning(warning, icon="⚠️")


@st.cache_resource(show_spinner="Loading prebuilt electrochemical database and ML models...")
def _load_cached_controller(data_dir_key: tuple[str, str, str]) -> CorrosionPlatformController:
    """Load database and ML predictors once per external data directory set."""
    _ = data_dir_key
    return CorrosionPlatformController.from_default_data()


def get_controller() -> CorrosionPlatformController:
    """Ensure external data/raw changes trigger a visible full-model rebuild before loading controller."""
    raw_path, proc_path, lab_path = resolve_data_dirs()
    data_dir_key = (str(raw_path), str(proc_path), str(lab_path))
    if not st.session_state.get("_controller_verified", False):
        progress_slot = st.empty()
        progress_bar = None

        def _on_progress(step: int, total: int, message: str) -> None:
            nonlocal progress_bar
            if progress_bar is None:
                progress_bar = progress_slot.progress(
                    0.0,
                    text="Detected updated CSV data in data/raw/ — preparing full ML model build...",
                )
            fraction = min(max(step / max(total, 1), 0.0), 1.0)
            progress_bar.progress(fraction, text=message)

        try:
            rebuild_banner = ArtifactRepository.ensure_fresh_on_disk(
                raw_dir=raw_path,
                processed_dir=proc_path,
                lab_dir=lab_path,
                rebuild_models=("all",),
                progress_callback=_on_progress,
                before_rebuild=lambda: _load_cached_controller.clear(data_dir_key),
            )
            if rebuild_banner:
                st.session_state["_rebuild_banner"] = rebuild_banner
        finally:
            progress_slot.empty()
        st.session_state["_controller_verified"] = True

    return _load_cached_controller(data_dir_key)


def _format_time_s(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "∞ (Not reached)"
    if seconds >= 3600.0:
        return f"{seconds / 3600.0:.2f} hr ({seconds:.0f} s)"
    if seconds >= 60.0:
        return f"{seconds / 60.0:.2f} min ({seconds:.1f} s)"
    return f"{seconds:.1f} s"


def _format_mc_time_interval(summary: dict[str, float] | None) -> str | None:
    if not summary:
        return None
    p05 = summary.get("p05", float("nan"))
    p95 = summary.get("p95", float("nan"))
    if math.isnan(p05) and math.isnan(p95):
        return None
    return f"90% CI: {_format_time_s(p05)} – {_format_time_s(p95)}"


def _render_manual_metal_inputs(
    metal: str,
    *,
    default_ecorr: float,
    default_icorr: float,
    key_prefix: str,
) -> ManualElectrochemicalInput:
    with st.sidebar.expander(f"Manual Parameters — {metal}", expanded=True):
        ecorr = st.number_input(
            f"{metal} E_corr (V vs. SHE)", value=default_ecorr, step=0.05, key=f"{key_prefix}_e"
        )
        icorr = st.number_input(
            f"{metal} i_corr (µA/cm²)", min_value=1e-4, value=default_icorr, step=0.5, key=f"{key_prefix}_i"
        )
        beta_a = st.number_input(
            f"{metal} β_a (mV/dec)", min_value=20.0, max_value=500.0, value=60.0, step=5.0, key=f"{key_prefix}_ba"
        )
        beta_c = st.number_input(
            f"{metal} β_c (mV/dec)", min_value=20.0, max_value=500.0, value=120.0, step=5.0, key=f"{key_prefix}_bc"
        )
    return ManualElectrochemicalInput(
        Ecorr_V_SHE=float(ecorr),
        icorr_uA_cm2=float(icorr),
        beta_a_mV_dec=float(beta_a),
        beta_c_mV_dec=float(beta_c),
    )


def _render_confidence_and_warnings(
    confidence_tier: str,
    warnings: list[str],
    recommendations: list[str],
    *,
    applicability_items: list[tuple[str, ApplicabilityDomainResult]] | None = None,
    calibration_mode: str = "literature_only",
    lab_project: str | None = None,
) -> None:
    if confidence_tier == "High":
        st.success(f"**Confidence Tier:** {confidence_tier} — Query lies well within the Applicability Domain.")
    elif confidence_tier == "Medium":
        st.warning(f"**Confidence Tier:** {confidence_tier} — Moderate coverage in the Applicability Domain.")
    else:
        st.error(f"**Confidence Tier:** {confidence_tier} — Extrapolation or sparse data detected.")

    if applicability_items and calibration_mode in ("delta_learning", "lab_only_gp"):
        for metal_label, ad in applicability_items:
            knn_str = (
                f"{ad.lab_knn_distance:.3f}"
                if (ad.lab_knn_distance is not None and math.isfinite(ad.lab_knn_distance))
                else "∞"
            )
            if calibration_mode == "delta_learning":
                lit_box = "In Literature Box" if ad.in_bounding_box else "Outside Literature Box"
                st.info(
                    f"**Applicability Domain ({metal_label} — Delta-Learning):** "
                    f"Literature coverage: `{lit_box}` (`n={ad.n_metal_training_samples}`) | "
                    f"Lab proximity (`{lab_project or 'None'}`): **{ad.lab_anchoring_status}** "
                    f"(`n_lab={ad.n_lab_samples}`, `lab_in_box={ad.lab_in_bounding_box}`, "
                    f"`lab k-NN distance={knn_str}`)."
                )
            else:
                st.info(
                    f"**Applicability Domain ({metal_label} — Lab-Only Local GP):** "
                    f"Evaluated strictly against lab project `{lab_project or 'None'}` — "
                    f"**{ad.lab_anchoring_status}** (`n_lab={ad.n_lab_samples}`, "
                    f"`lab_in_box={ad.lab_in_bounding_box}`, `lab k-NN distance={knn_str}`)."
                )

    secondary_warnings = [
        w for w in warnings if not w.startswith(NON_AQUEOUS_WARNING_PREFIX)
    ]
    if secondary_warnings:
        with st.expander(f"Applicability Domain & Data-Sufficiency Warnings ({len(secondary_warnings)})", expanded=(confidence_tier != "High")):
            for item in secondary_warnings:
                st.warning(item)

    if recommendations:
        with st.expander(f"Suggested Follow-Up Experiments ({len(recommendations)})", expanded=(confidence_tier == "Low")):
            for rec in recommendations:
                st.markdown(f"- {rec}")


def _render_residual_breakdown_section(
    breakdowns: list[tuple[str, ResidualBreakdown]],
) -> None:
    """Render an expandable Residual Breakdown table for Delta-Learning calibration."""
    if not breakdowns:
        return
    with st.expander("Residual Breakdown (Prior → Lab Δ → Calibrated Final)", expanded=False):
        for label, breakdown in breakdowns:
            if len(breakdowns) > 1:
                st.markdown(f"**{label}**")
            st.dataframe(breakdown.to_dataframe(), width="stretch", hide_index=True)


def _format_metal_label(metal: str, counts: dict[str, int], has_faraday: dict[str, bool]) -> str:
    n = counts.get(metal, 0)
    tags: list[str] = [f"n={n}"]
    if n < 10:
        tags.append("low data")
    if not has_faraday.get(metal, True):
        tags.append("no Faraday props")
    return f"{metal} ({', '.join(tags)})"


def _render_sidebar(controller: CorrosionPlatformController) -> SimulationRequest:
    options = controller.available_options()
    all_metals: list[str] = options["metals"]
    core_metals: list[str] = options.get("core_metals", all_metals)
    metal_counts: dict[str, int] = options.get("metal_counts", {})
    metal_categories: dict[str, str] = options.get("metal_categories", {})
    metal_has_faraday: dict[str, bool] = options.get("metal_has_faraday", {})
    categories: list[str] = options.get("categories", [])

    st.sidebar.title("Simulation Parameters")

    rebuild_banner = st.session_state.get("_rebuild_banner")
    if rebuild_banner:
        st.sidebar.success(rebuild_banner)

    repo = controller.repository
    if repo.artifact_status == "seed" and repo.artifact_warning:
        st.sidebar.error(repo.artifact_warning)
    elif repo.artifact_status in ("stale", "live_fallback") and repo.artifact_warning:
        st.sidebar.warning(repo.artifact_warning)
    elif repo.artifact_status == "rebuilt" and repo.artifact_warning and not rebuild_banner:
        st.sidebar.success(repo.artifact_warning)
    elif repo.artifact_warning:
        st.sidebar.info(repo.artifact_warning)

    with st.sidebar.expander("External Data & Model Cache", expanded=False):
        n_records = len(controller.database.electrochemical_database)
        st.markdown("**Literature Prior (`data/raw/`)**")
        st.caption(
            f"**Status:** `{repo.artifact_status}`  \n"
            f"**Records:** `{n_records}` across `{len(all_metals)}` materials  \n"
            f"**Raw CSV Dir:** `{repo.raw_dir or 'N/A'}`"
        )
        lab_project_names: list[str] = options.get("lab_projects", [])
        st.markdown("**Lab Data (`data/lab/`)**")
        if lab_project_names:
            selected_lab = st.selectbox(
                "Active Lab Project",
                options=lab_project_names,
                index=0,
                key="sidebar_active_lab_project",
            )
            active_ds = repo.get_lab_project(selected_lab)
            active_n = active_ds.n_records if active_ds is not None else 0
            st.caption(
                f"**Lab Status:** `{repo.lab_status}`  \n"
                f"**Active Project:** `{selected_lab}` (`{active_n}` records; `{len(lab_project_names)}` project(s))  \n"
                f"**Lab CSV Dir:** `{repo.lab_dir or 'N/A'}`"
            )
        else:
            st.caption(
                f"**Lab Status:** `{repo.lab_status}` (`0` projects)  \n"
                f"**Active Project:** `None` (`0` records)  \n"
                f"**Lab CSV Dir:** `{repo.lab_dir or 'N/A'}`"
            )
        if st.button("🔄 Reload External Data", width="stretch"):
            progress_slot = st.empty()
            progress_bar = None

            def _on_reload_progress(step: int, total: int, message: str) -> None:
                nonlocal progress_bar
                if progress_bar is None:
                    progress_bar = progress_slot.progress(0.0, text=message)
                fraction = min(max(step / max(total, 1), 0.0), 1.0)
                progress_bar.progress(fraction, text=message)

            reload_summary = repo.reload_external_data(
                progress_callback=_on_reload_progress
            )
            progress_slot.empty()
            if reload_summary.raw_rebuilt:
                st.session_state["_rebuild_banner"] = (
                    f"Literature raw data changed — rebuilt base models ({len(controller.database.electrochemical_database)} records) "
                    f"and refreshed {len(reload_summary.lab_projects)} lab project(s) in {reload_summary.elapsed_s:.2f}s."
                )
            elif reload_summary.lab_added_or_updated or reload_summary.lab_removed:
                st.session_state["_rebuild_banner"] = (
                    f"Lab data reloaded in {reload_summary.elapsed_s:.3f}s "
                    f"(literature prior unchanged, base rebuild skipped; "
                    f"{len(reload_summary.lab_projects)} project(s))."
                )
            else:
                st.session_state["_rebuild_banner"] = (
                    f"External data verified fresh in {reload_summary.elapsed_s:.3f}s (no changes detected)."
                )
            st.rerun()

    st.sidebar.subheader("1. Metal / Material Selection")
    scope_options: list[tuple[str, str, str]] = [
        ("all", "", f"All Materials ({len(all_metals)})"),
    ]
    if core_metals and len(core_metals) < len(all_metals):
        scope_options.append(("core", "", f"Core Semiconductor Metals ({len(core_metals)})"))
    if len(categories) > 1:
        for cat in categories:
            cat_count = sum(1 for m in all_metals if metal_categories.get(m, "metal") == cat)
            scope_options.append(("category", cat, f"Category: {cat} ({cat_count})"))

    if len(scope_options) > 1:
        selected_scope = st.sidebar.selectbox(
            "Primary Material Filter",
            options=scope_options,
            index=0,
            format_func=lambda item: item[2],
            help="Filter the Primary Material list by catalog tier or functional category.",
        )
        scope_kind, scope_val, _ = selected_scope
        if scope_kind == "core":
            metals = core_metals
        elif scope_kind == "category":
            metals = [m for m in all_metals if metal_categories.get(m, "metal") == scope_val]
        else:
            metals = all_metals
    else:
        metals = all_metals

    if not metals:
        metals = all_metals

    primary_metal = st.sidebar.selectbox(
        "Primary Material",
        options=metals,
        index=0,
        format_func=lambda m: _format_metal_label(m, metal_counts, metal_has_faraday),
    )

    # Allow cross-category galvanic pairing (e.g., TiN barrier paired with Cu metal)
    secondary_choices = ["None (Single Metal Only)"] + [m for m in all_metals if m != primary_metal]
    default_secondary_idx = secondary_choices.index("Co") if "Co" in secondary_choices else (1 if len(secondary_choices) > 1 else 0)
    secondary_selection = st.sidebar.selectbox(
        "Secondary Material (Optional Galvanic Partner)",
        options=secondary_choices,
        index=default_secondary_idx,
        format_func=lambda m: (
            m if m.startswith("None") else _format_metal_label(m, metal_counts, metal_has_faraday)
        ),
    )
    secondary_metal = None if secondary_selection.startswith("None") else secondary_selection

    area_ratio = st.sidebar.number_input(
        "Cathode-to-Anode Area Ratio (A_noble / A_active)",
        min_value=0.05,
        max_value=100.0,
        value=1.0,
        step=0.25,
        disabled=(secondary_metal is None),
    )

    st.sidebar.subheader("2. Solution & Operating Conditions")
    st.sidebar.markdown("**Solvent System**")
    solvent_regime = st.sidebar.selectbox(
        "Solvent Regime",
        options=options["solvent_regimes"],
        format_func=lambda key: SOLVENT_REGIME_LABELS.get(key, key),
        index=0,
        key="sidebar_solvent_regime",
    )
    organic_options = options["organic_solvent_types"]
    organic_solvent_type = st.sidebar.selectbox(
        "Organic Co-Solvent",
        options=organic_options,
        index=0,
        key="sidebar_organic_solvent",
    )
    w_min, w_max, w_default, w_step = water_content_slider_bounds(
        solvent_regime, organic_solvent_type
    )
    prev_regime = st.session_state.get("_last_sidebar_solvent_regime")
    if "sidebar_water_content" not in st.session_state:
        st.session_state["sidebar_water_content"] = w_default
    elif prev_regime != solvent_regime or not (
        w_min <= float(st.session_state["sidebar_water_content"]) <= w_max
    ):
        st.session_state["sidebar_water_content"] = w_default
    st.session_state["_last_sidebar_solvent_regime"] = solvent_regime
    water_content_wt_pct = st.sidebar.slider(
        "Water Content (wt%)",
        min_value=w_min,
        max_value=w_max,
        step=w_step,
        key="sidebar_water_content",
    )

    oxidizer_options = options["oxidizer_types"]
    default_ox = oxidizer_options.index("H2O2") if "H2O2" in oxidizer_options else 0
    oxidizer_type = st.sidebar.selectbox("Oxidizer Type", options=oxidizer_options, index=default_ox)
    oxidizer_conc = st.sidebar.number_input(
        "Oxidizer Concentration (wt% or M)",
        min_value=0.0,
        max_value=50.0,
        value=1.0,
        step=0.1,
        disabled=(oxidizer_type == "None"),
    )

    acid_options = options["acid_types"]
    acid_type = st.sidebar.selectbox("Acid / Complexing Agent", options=acid_options, index=0)
    acid_conc = st.sidebar.number_input(
        "Acid / Complexing Concentration (wt% or M)",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=0.1,
        disabled=(acid_type == "None"),
    )

    inhibitor_options = options["inhibitor_types"]
    inhibitor_type = st.sidebar.selectbox("Corrosion Inhibitor", options=inhibitor_options, index=0)
    inhibitor_conc = st.sidebar.number_input(
        "Inhibitor Concentration (mM or wt%)",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=0.1,
        disabled=(inhibitor_type == "None"),
    )

    pH = st.sidebar.slider("Solution pH", min_value=0.0, max_value=14.0, value=3.0, step=0.1)
    temperature_C = st.sidebar.slider("Operating Temperature (°C)", min_value=5.0, max_value=95.0, value=25.0, step=1.0)

    st.sidebar.subheader("3. Immersion Time & Kinetics")
    immersion_time_s = st.sidebar.number_input(
        "Immersion Time (s)",
        min_value=1.0,
        max_value=86400.0,
        value=300.0,
        step=30.0,
    )
    d_crit_nm = st.sidebar.number_input(
        "Critical Thickness Threshold d_crit (nm)",
        min_value=0.05,
        max_value=100.0,
        value=1.0,
        step=0.1,
    )
    kinetics_model = st.sidebar.selectbox(
        "Kinetics Model",
        options=options["kinetics_models"],
        format_func=lambda key: KINETICS_LABELS[key],
        index=0,
    )
    d_max_nm = 5.0
    power_law_n = 0.6
    if kinetics_model in ("auto", "passivation"):
        d_max_nm = st.sidebar.number_input(
            "Passivation Limiting Depth d_max (nm)",
            min_value=0.2,
            max_value=50.0,
            value=5.0,
            step=0.5,
        )
    if kinetics_model in ("auto", "power_law"):
        power_law_n = st.sidebar.slider(
            "Power-Law Exponent (n)",
            min_value=0.3,
            max_value=1.0,
            value=0.6,
            step=0.05,
        )

    st.sidebar.subheader("4. Parameter Source")
    parameter_source = st.sidebar.radio(
        "Electrochemical Parameter Source",
        options=options["parameter_sources"],
        format_func=lambda key: SOURCE_LABELS[key],
        index=0,
        help=(
            "- **ML Prediction**: No experimental data yet; predict from chemistry & conditions.\n"
            "- **Database Lookup**: Compare against the closest measured literature record.\n"
            "- **Manual Input**: You already measured Tafel parameters and want pure physics simulation."
        ),
    )

    ml_model_family: str | None = None
    calibration_mode = "literature_only"
    selected_lab_project: str | None = None
    if parameter_source == "ml":
        st.sidebar.markdown("**Lab Calibration**")
        calibration_mode = st.sidebar.radio(
            "Lab Calibration Mode",
            options=options["calibration_modes"],
            format_func=lambda key: CALIBRATION_MODE_LABELS.get(key, key),
            index=0,
            key="sidebar_calibration_mode",
            help=(
                "- **Literature Only**: General screening with no lab data.\n"
                "- **Literature + Lab Calibration**: Adjust literature trends using your small lab set (n < 20) in similar chemistries.\n"
                "- **Lab Data Only**: Brand-new or non-aqueous systems (n ≥ 3) where aqueous literature does not apply."
            ),
        )
        lab_project_choices: list[str] = options.get("lab_projects", [])
        if lab_project_choices:
            selected_lab_project = st.sidebar.selectbox(
                "Lab Dataset (data/lab/)",
                options=lab_project_choices,
                index=0,
                disabled=(calibration_mode == "literature_only"),
                key="sidebar_lab_project",
                help=(
                    "- Select a lab CSV folder under `data/lab/<project>/` for calibration.\n"
                    "- Disabled when using **Literature Only**."
                ),
            )
        elif calibration_mode != "literature_only":
            st.sidebar.info(
                "No lab projects found under `data/lab/<project>/`. Add lab CSVs and click **Reload External Data**."
            )

        auto_model = CALIBRATION_MODE_DEFAULT_MODELS.get(calibration_mode, "rf_gp_ensemble")
        if calibration_mode == "lab_only_gp":
            st.sidebar.caption(
                "Active ML Architecture: **Lab-Only Local GP** (auto-selected)"
            )
            ml_model_family = None
        else:
            auto_label = MODEL_FAMILY_LABELS.get(auto_model, auto_model)
            st.sidebar.caption(f"Active ML Architecture: **{auto_label}** (auto-selected)")
            with st.sidebar.expander("Advanced: Override Base Model", expanded=False):
                override_options = ["auto", *options["ml_model_families"]]
                selected_override = st.selectbox(
                    "Base Literature Model Override",
                    options=override_options,
                    format_func=lambda key: (
                        f"Auto ({auto_label})"
                        if key == "auto"
                        else MODEL_FAMILY_LABELS.get(key, key)
                    ),
                    index=0,
                    key="sidebar_base_model_override",
                )
                ml_model_family = None if selected_override == "auto" else selected_override

    manual_primary: ManualElectrochemicalInput | None = None
    manual_secondary: ManualElectrochemicalInput | None = None
    if parameter_source == "manual":
        manual_primary = _render_manual_metal_inputs(
            primary_metal, default_ecorr=0.20, default_icorr=2.5, key_prefix="man_p"
        )
        if secondary_metal is not None:
            manual_secondary = _render_manual_metal_inputs(
                secondary_metal, default_ecorr=-0.10, default_icorr=5.0, key_prefix="man_s"
            )

    return SimulationRequest(
        primary_metal=primary_metal,
        secondary_metal=secondary_metal,
        cathode_to_anode_area_ratio=float(area_ratio),
        recipe=CleaningRecipe(
            solution=Solution(
                oxidizer_type=oxidizer_type,
                oxidizer_concentration=float(oxidizer_conc),
                acid_type=acid_type,
                acid_concentration=float(acid_conc),
                inhibitor_type=inhibitor_type,
                inhibitor_concentration=float(inhibitor_conc),
                pH=float(pH),
                solvent_regime=solvent_regime,
                organic_solvent_type=organic_solvent_type,
                water_content_wt_pct=float(water_content_wt_pct),
            ),
            temperature_C=float(temperature_C),
            immersion_time_s=float(immersion_time_s),
        ),
        d_crit_nm=float(d_crit_nm),
        kinetics_model=kinetics_model,
        d_max_nm=float(d_max_nm),
        power_law_n=float(power_law_n),
        parameter_source=parameter_source,
        ml_model_family=ml_model_family,
        calibration_mode=calibration_mode,
        lab_project=(selected_lab_project if calibration_mode != "literature_only" else None),
        manual_primary=manual_primary,
        manual_secondary=manual_secondary,
    )


def _render_depth_and_risk_charts(
    depth_figure: Any,
    risk_figure: Any,
    *,
    scale_key: str,
) -> None:
    """Render side-by-side Depth and Risk Plotly charts with a Linear/Log Y-axis toggle for Depth."""
    depth_scale = st.radio(
        "Depth Y-Axis Scale",
        options=["linear", "log"],
        format_func=lambda s: "Linear" if s == "linear" else "Log",
        horizontal=True,
        key=scale_key,
    )
    col_depth, col_risk = st.columns(2)
    with col_depth:
        st.plotly_chart(
            apply_depth_yaxis_scale(depth_figure, depth_scale),
            width="stretch",
        )
    with col_risk:
        st.plotly_chart(risk_figure, width="stretch")


def _render_single_metal_view(payload: SingleMetalAnalysisResult) -> None:
    st.subheader(f"Single Metal Analysis — {payload.request.primary_metal}")
    _render_non_aqueous_warning_banner(payload.request)
    if payload.request.parameter_source == "ml":
        model_label = format_active_model_label(
            resolve_active_model_family(payload.request, payload.resolved_parameters)
        )
        cal_label = CALIBRATION_MODE_LABELS.get(
            payload.request.calibration_mode, payload.request.calibration_mode
        )
        lab_clause = (
            f" (`{payload.resolved_parameters.lab_project or 'None'}`)"
            if payload.request.calibration_mode != "literature_only"
            else ""
        )
        st.caption(
            f"Active ML Model: **{model_label}** | Calibration: **{cal_label}**{lab_clause} | "
            f"Solvent Regime: **{payload.request.recipe.solution.solvent_regime}** "
            f"({payload.request.recipe.solution.solvent_subclass}, {float(payload.request.recipe.solution.water_content_wt_pct):.1f} wt% water)"
        )
    ad_items = (
        [(payload.request.primary_metal, payload.resolved_parameters.applicability)]
        if payload.resolved_parameters.applicability is not None
        else None
    )
    _render_confidence_and_warnings(
        payload.confidence_tier,
        payload.warnings,
        payload.experimental_recommendations,
        applicability_items=ad_items,
        calibration_mode=payload.request.calibration_mode,
        lab_project=payload.resolved_parameters.lab_project,
    )

    c1, c2, c3, c4 = st.columns(4)
    has_rate = math.isfinite(payload.initial_corrosion_rate_nm_min)
    c1.metric(
        "Initial Corrosion Rate (CR₀)",
        f"{payload.initial_corrosion_rate_nm_min:.3g} nm/min" if has_rate else "N/A",
        delta=(
            f"{payload.initial_corrosion_rate_mm_yr:.3g} mm/yr"
            if has_rate
            else "Missing physical properties"
        ),
        delta_color="off",
    )
    c2.metric(
        f"Corrosion Depth @ {payload.request.recipe.immersion_time_s:.0f} s",
        f"{payload.corrosion_depth_nm:.3f} nm" if math.isfinite(payload.corrosion_depth_nm) else "N/A",
        delta=f"Threshold: {payload.request.d_crit_nm:.2f} nm",
        delta_color="off",
    )
    ttt_ci = (
        _format_mc_time_interval(payload.mc_kinetics.time_to_threshold_summary)
        if payload.mc_kinetics is not None
        else None
    )
    c3.metric(
        "Time to Threshold",
        _format_time_s(payload.time_to_threshold_s) if has_rate else "N/A",
        delta=ttt_ci,
        delta_color="off",
    )
    c4.metric(
        "Thickness Loss Risk",
        payload.thickness_loss_risk_tier,
        delta=(
            f"Score R(t) = {payload.thickness_loss_risk_score:.3f}"
            if math.isfinite(payload.thickness_loss_risk_score)
            else "Unavailable"
        ),
        delta_color="off",
    )

    st.markdown("#### Electrochemical Parameters & Uncertainty (±1σ)")
    st.dataframe(payload.parameter_table, width="stretch", hide_index=True)
    if (
        payload.request.calibration_mode == "delta_learning"
        and payload.resolved_parameters.residual_breakdown is not None
    ):
        _render_residual_breakdown_section(
            [(payload.request.primary_metal, payload.resolved_parameters.residual_breakdown)]
        )

    figs = build_single_metal_figures(payload)
    st.plotly_chart(figs.polarization_figure, width="stretch")
    _render_depth_and_risk_charts(
        figs.depth_figure,
        figs.risk_figure,
        scale_key="single_depth_yaxis_scale",
    )


def _render_galvanic_view(payload: GalvanicAnalysisResult | None) -> None:
    if payload is None:
        st.info("Select a Secondary Metal in the sidebar to run Bimetallic Galvanic Analysis.")
        return

    st.subheader(
        f"Galvanic Analysis — {payload.active_metal} (Active Metal) coupled with {payload.noble_metal} (Noble Metal)"
    )
    _render_non_aqueous_warning_banner(payload.request)
    if payload.request.parameter_source == "ml":
        model_label = format_active_model_label(
            resolve_active_model_family(payload.request, payload.primary_resolved)
        )
        cal_label = CALIBRATION_MODE_LABELS.get(
            payload.request.calibration_mode, payload.request.calibration_mode
        )
        lab_clause = (
            f" (`{payload.primary_resolved.lab_project or 'None'}`)"
            if payload.request.calibration_mode != "literature_only"
            else ""
        )
        st.caption(
            f"Active ML Model: **{model_label}** | Calibration: **{cal_label}**{lab_clause} | "
            f"Solvent Regime: **{payload.request.recipe.solution.solvent_regime}** "
            f"({payload.request.recipe.solution.solvent_subclass}, {float(payload.request.recipe.solution.water_content_wt_pct):.1f} wt% water)"
        )
    galv_ad_items: list[tuple[str, ApplicabilityDomainResult]] = []
    if payload.primary_resolved.applicability is not None:
        galv_ad_items.append((payload.primary_resolved.metal, payload.primary_resolved.applicability))
    if payload.secondary_resolved.applicability is not None:
        galv_ad_items.append((payload.secondary_resolved.metal, payload.secondary_resolved.applicability))
    _render_confidence_and_warnings(
        payload.confidence_tier,
        payload.warnings,
        payload.experimental_recommendations,
        applicability_items=galv_ad_items or None,
        calibration_mode=payload.request.calibration_mode,
        lab_project=payload.primary_resolved.lab_project,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Active / Noble Metal",
        f"{payload.active_metal} / {payload.noble_metal}",
        delta=f"Driving Force: {payload.galvanic_driving_force_risk} (|ΔE| = {payload.driving_force.delta_Ecorr_V:.3f} V)",
        delta_color="off",
    )
    c2.metric(
        "Mixed Potential (E_galv)",
        f"{payload.mixed_potential_V_SHE:.3f} V vs. SHE",
        delta=f"i_galv = {payload.galvanic_current_density_uA_cm2:.3g} µA/cm²",
        delta_color="off",
    )
    c3.metric(
        "Acceleration Factor",
        f"{payload.acceleration_factor:.2f}×",
        delta=(
            f"Accelerated CR₀ = {payload.accelerated_cr0_nm_min:.3g} nm/min"
            if math.isfinite(payload.accelerated_cr0_nm_min)
            else "CR₀: N/A (missing physical properties)"
        ),
        delta_color="off",
    )
    coupled_ttt_ci = (
        _format_mc_time_interval(payload.mc_coupled_kinetics.time_to_threshold_summary)
        if payload.mc_coupled_kinetics is not None
        else None
    )
    has_coupled_rate = math.isfinite(payload.accelerated_cr0_nm_min)
    c4.metric(
        "Coupled Time to Threshold",
        _format_time_s(payload.coupled_time_to_threshold_s) if has_coupled_rate else "N/A",
        delta=(
            coupled_ttt_ci
            or (
                f"Uncoupled: {_format_time_s(payload.uncoupled_time_to_threshold_s)}"
                if has_coupled_rate
                else "Unavailable"
            )
        ),
        delta_color="off",
    )

    comparison_df = pd.DataFrame(
        [
            {
                "Condition": f"{payload.active_metal} Uncoupled",
                "CR₀ (nm/min)": payload.uncoupled_cr0_nm_min if math.isfinite(payload.uncoupled_cr0_nm_min) else None,
                f"Corrosion Depth @ {payload.request.recipe.immersion_time_s:.0f} s (nm)": (
                    payload.uncoupled_depth_nm if math.isfinite(payload.uncoupled_depth_nm) else None
                ),
                "Time to Threshold": (
                    _format_time_s(payload.uncoupled_time_to_threshold_s) if has_coupled_rate else "N/A"
                ),
                "Thickness Loss Risk Score": (
                    round(payload.uncoupled_risk_score, 3) if math.isfinite(payload.uncoupled_risk_score) else None
                ),
                "Risk Tier": payload.uncoupled_risk_tier,
            },
            {
                "Condition": f"{payload.active_metal} Coupled to {payload.noble_metal} (A_ratio={payload.request.cathode_to_anode_area_ratio:g})",
                "CR₀ (nm/min)": payload.accelerated_cr0_nm_min if math.isfinite(payload.accelerated_cr0_nm_min) else None,
                f"Corrosion Depth @ {payload.request.recipe.immersion_time_s:.0f} s (nm)": (
                    payload.coupled_depth_nm if math.isfinite(payload.coupled_depth_nm) else None
                ),
                "Time to Threshold": (
                    _format_time_s(payload.coupled_time_to_threshold_s) if has_coupled_rate else "N/A"
                ),
                "Thickness Loss Risk Score": (
                    round(payload.coupled_risk_score, 3) if math.isfinite(payload.coupled_risk_score) else None
                ),
                "Risk Tier": payload.coupled_risk_tier,
            },
        ]
    )
    st.markdown("#### Galvanic Electrochemical Parameters (Anode, Cathode & Mixed Point)")
    st.dataframe(payload.electrochemical_parameter_table, width="stretch", hide_index=True)

    st.markdown("#### Uncoupled vs. Galvanically Coupled Summary")
    st.dataframe(comparison_df, width="stretch", hide_index=True)
    if payload.request.calibration_mode == "delta_learning":
        galv_breakdowns: list[tuple[str, ResidualBreakdown]] = []
        if payload.primary_resolved.residual_breakdown is not None:
            galv_breakdowns.append(
                (payload.primary_resolved.metal, payload.primary_resolved.residual_breakdown)
            )
        if payload.secondary_resolved.residual_breakdown is not None:
            galv_breakdowns.append(
                (payload.secondary_resolved.metal, payload.secondary_resolved.residual_breakdown)
            )
        _render_residual_breakdown_section(galv_breakdowns)

    figs = build_galvanic_figures(payload)
    st.plotly_chart(figs.galvanic_polarization_figure, width="stretch")
    _render_depth_and_risk_charts(
        figs.galvanic_depth_figure,
        figs.galvanic_risk_figure,
        scale_key="galvanic_depth_yaxis_scale",
    )


def _render_galvanic_series_explorer(controller: CorrosionPlatformController) -> None:
    """Allow users to inspect median E_corr ranking across a chosen material subset."""
    if controller.database.galvanic_series.empty:
        return
    options = controller.available_options()
    all_metals: list[str] = options["metals"]
    default_subset = options.get("core_metals", all_metals[:8])[:10]
    with st.expander("Galvanic Series Overview (Select Material Subset)", expanded=False):
        selected_metals = st.multiselect(
            "Materials to compare",
            options=all_metals,
            default=default_subset,
            help="Select a subset of materials to keep the galvanic ranking readable as the catalog grows.",
        )
        if selected_metals:
            sub = controller.database.summarize_galvanic_series(selected_metals)
            st.dataframe(sub, width="stretch", hide_index=True)


def _default_recipe_presets(base_request: SimulationRequest) -> list[RecipeCandidate]:
    base_rec = base_request.recipe
    base_sol = base_rec.solution
    return [
        RecipeCandidate(name="Baseline Recipe", recipe=base_rec),
        RecipeCandidate(
            name="Low-Oxidizer Mild Rinse",
            recipe=CleaningRecipe(
                solution=Solution(
                    oxidizer_type=base_sol.oxidizer_type,
                    oxidizer_concentration=max(0.1, round(base_sol.oxidizer_concentration * 0.4, 2)),
                    acid_type=base_sol.acid_type,
                    acid_concentration=base_sol.acid_concentration,
                    inhibitor_type="BTA",
                    inhibitor_concentration=1.0,
                    pH=min(12.0, round(base_sol.pH + 1.5, 1)),
                ),
                temperature_C=max(15.0, base_rec.temperature_C - 5.0),
                immersion_time_s=max(30.0, round(base_rec.immersion_time_s * 0.5, 0)),
            ),
        ),
        RecipeCandidate(
            name="High-Temp Aggressive Clean",
            recipe=CleaningRecipe(
                solution=Solution(
                    oxidizer_type=base_sol.oxidizer_type,
                    oxidizer_concentration=min(30.0, round(max(base_sol.oxidizer_concentration, 0.5) * 2.0, 2)),
                    acid_type=base_sol.acid_type,
                    acid_concentration=base_sol.acid_concentration,
                    inhibitor_type="None",
                    inhibitor_concentration=0.0,
                    pH=max(1.0, round(base_sol.pH - 0.5, 1)),
                ),
                temperature_C=min(85.0, base_rec.temperature_C + 20.0),
                immersion_time_s=round(base_rec.immersion_time_s * 1.5, 0),
            ),
        ),
        RecipeCandidate(
            name="Neutral Inhibited Clean",
            recipe=CleaningRecipe(
                solution=Solution(
                    oxidizer_type="None",
                    oxidizer_concentration=0.0,
                    acid_type="None",
                    acid_concentration=0.0,
                    inhibitor_type="BTA",
                    inhibitor_concentration=2.0,
                    pH=7.0,
                ),
                temperature_C=25.0,
                immersion_time_s=120.0,
            ),
        ),
        RecipeCandidate(
            name="Short High-pH Rinse",
            recipe=CleaningRecipe(
                solution=Solution(
                    oxidizer_type=base_sol.oxidizer_type,
                    oxidizer_concentration=0.5,
                    acid_type="None",
                    acid_concentration=0.0,
                    inhibitor_type="None",
                    inhibitor_concentration=0.0,
                    pH=9.0,
                ),
                temperature_C=25.0,
                immersion_time_s=60.0,
            ),
        ),
    ]


def _render_recipe_comparison_view(
    controller: CorrosionPlatformController,
    base_request: SimulationRequest,
) -> RecipeComparisonPayload:
    mode_label = (
        f"Galvanic Pair ({base_request.primary_metal} / {base_request.secondary_metal}, "
        f"Area Ratio = {base_request.cathode_to_anode_area_ratio:g})"
        if base_request.secondary_metal is not None
        else f"Single Metal ({base_request.primary_metal})"
    )
    st.subheader(f"Recipe Comparison — {mode_label}")
    st.caption(
        "Define 2 to 5 candidate Cleaning Recipes below to compare corrosion rate, cumulative depth, "
        "Thickness Loss Risk, Time to Threshold, and (when coupled) Acceleration Factor."
    )

    options = controller.available_options()
    presets = _default_recipe_presets(base_request)
    n_recipes = st.slider("Number of Candidate Cleaning Recipes", min_value=2, max_value=5, value=3)

    organic_options = options["organic_solvent_types"]
    regime_options = options["solvent_regimes"]

    recipes: list[RecipeCandidate] = []
    cols = st.columns(n_recipes)
    for idx in range(n_recipes):
        preset = presets[idx]
        preset_rec = preset.recipe
        preset_sol = preset_rec.solution
        with cols[idx]:
            st.markdown(f"**Recipe {idx + 1}**")
            name = st.text_input("Name", value=preset.name, key=f"cmp_name_{idx}")
            r_regime_key = f"cmp_regime_{idx}"
            r_org_key = f"cmp_org_{idx}"
            r_water_key = f"cmp_water_{idx}"
            r_last_key = f"_last_cmp_regime_{idx}"
            if r_regime_key not in st.session_state:
                st.session_state[r_regime_key] = preset_sol.solvent_regime
            cmp_regime = st.selectbox(
                "Solvent Regime",
                options=regime_options,
                format_func=lambda k: SOLVENT_REGIME_LABELS.get(k, k),
                key=r_regime_key,
            )
            if r_org_key not in st.session_state:
                st.session_state[r_org_key] = (
                    preset_sol.organic_solvent_type
                    if preset_sol.organic_solvent_type in organic_options
                    else "None"
                )
            cmp_org = st.selectbox(
                "Co-Solvent",
                options=organic_options,
                key=r_org_key,
            )
            cw_min, cw_max, cw_default, cw_step = water_content_slider_bounds(cmp_regime, cmp_org)
            if r_water_key not in st.session_state:
                st.session_state[r_water_key] = float(preset_sol.water_content_wt_pct)
            elif st.session_state.get(r_last_key) != cmp_regime or not (
                cw_min <= float(st.session_state[r_water_key]) <= cw_max
            ):
                st.session_state[r_water_key] = cw_default
            st.session_state[r_last_key] = cmp_regime
            cmp_water = st.number_input(
                "Water Content (wt%)",
                min_value=cw_min,
                max_value=cw_max,
                step=cw_step,
                key=r_water_key,
            )
            ox_idx = options["oxidizer_types"].index(preset_sol.oxidizer_type) if preset_sol.oxidizer_type in options["oxidizer_types"] else 0
            ox_type = st.selectbox("Oxidizer", options=options["oxidizer_types"], index=ox_idx, key=f"cmp_ox_{idx}")
            ox_conc = st.number_input(
                "Oxidizer Conc.",
                min_value=0.0,
                max_value=50.0,
                value=float(preset_sol.oxidizer_concentration),
                step=0.2,
                disabled=(ox_type == "None"),
                key=f"cmp_oxc_{idx}",
            )
            acid_idx = options["acid_types"].index(preset_sol.acid_type) if preset_sol.acid_type in options["acid_types"] else 0
            acid_type = st.selectbox("Acid / Complexing", options=options["acid_types"], index=acid_idx, key=f"cmp_acid_{idx}")
            acid_conc = st.number_input(
                "Acid Conc.",
                min_value=0.0,
                max_value=50.0,
                value=float(preset_sol.acid_concentration),
                step=0.2,
                disabled=(acid_type == "None"),
                key=f"cmp_acidc_{idx}",
            )
            inh_idx = options["inhibitor_types"].index(preset_sol.inhibitor_type) if preset_sol.inhibitor_type in options["inhibitor_types"] else 0
            inh_type = st.selectbox("Inhibitor", options=options["inhibitor_types"], index=inh_idx, key=f"cmp_inh_{idx}")
            inh_conc = st.number_input(
                "Inhibitor Conc.",
                min_value=0.0,
                max_value=50.0,
                value=float(preset_sol.inhibitor_concentration),
                step=0.2,
                disabled=(inh_type == "None"),
                key=f"cmp_inhc_{idx}",
            )
            ph_val = st.number_input("pH", min_value=0.0, max_value=14.0, value=float(preset_sol.pH), step=0.5, key=f"cmp_ph_{idx}")
            temp_val = st.number_input("Temp (°C)", min_value=5.0, max_value=95.0, value=float(preset_rec.temperature_C), step=5.0, key=f"cmp_t_{idx}")
            time_val = st.number_input("Time (s)", min_value=1.0, max_value=86400.0, value=float(preset_rec.immersion_time_s), step=30.0, key=f"cmp_s_{idx}")

            cleaned_name = name.strip() or f"Recipe {idx + 1}"
            recipes.append(
                RecipeCandidate(
                    name=cleaned_name,
                    recipe=CleaningRecipe(
                        solution=Solution(
                            oxidizer_type=ox_type,
                            oxidizer_concentration=float(ox_conc),
                            acid_type=acid_type,
                            acid_concentration=float(acid_conc),
                            inhibitor_type=inh_type,
                            inhibitor_concentration=float(inh_conc),
                            pH=float(ph_val),
                            solvent_regime=cmp_regime,
                            organic_solvent_type=cmp_org,
                            water_content_wt_pct=float(cmp_water),
                        ),
                        temperature_C=float(temp_val),
                        immersion_time_s=float(time_val),
                    ),
                )
            )

    # Ensure unique names if user typed duplicates
    unique_recipes: list[RecipeCandidate] = []
    used: set[str] = set()
    for i, r in enumerate(recipes, start=1):
        unique_name = r.name if r.name not in used else f"{r.name} ({i})"
        used.add(unique_name)
        unique_recipes.append(replace(r, name=unique_name))

    comparison = compare_cleaning_recipes(controller, base_request, unique_recipes)

    st.success(f"**Lowest-Risk Recommendation:** {comparison.recommended_rationale}")
    st.markdown("#### Side-by-Side Cleaning Recipe Rankings")
    st.dataframe(comparison.comparison_table, width="stretch", hide_index=True)

    st.plotly_chart(comparison.bar_figure, width="stretch")
    st.plotly_chart(comparison.radar_figure, width="stretch")
    return comparison


def _render_report_export_view(
    single_payload: SingleMetalAnalysisResult,
    galvanic_payload: GalvanicAnalysisResult | None,
    comparison_payload: RecipeComparisonPayload | None,
) -> None:
    st.subheader("Multi-Format Technical Report Export")
    st.caption(
        "Download complete analysis artifacts containing input conditions, parameter provenance "
        "(Database vs. ML vs. Manual), Tafel & time-resolved figures, uncertainty intervals, "
        "Confidence Tiers, Applicability Domain warnings, experimental recommendations, and recipe rankings."
    )

    exports = generate_all_report_exports(
        single_payload,
        galvanic_payload=galvanic_payload,
        comparison_payload=comparison_payload,
    )
    if galvanic_payload is not None:
        stem = f"corrosion_report_{galvanic_payload.active_metal.lower()}_{galvanic_payload.noble_metal.lower()}"
    else:
        stem = f"corrosion_report_{single_payload.request.primary_metal.lower()}"

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Download HTML Report (.html)",
        data=exports["html"],
        file_name=f"{stem}.html",
        mime="text/html",
        width="stretch",
    )
    c2.download_button(
        "Download PDF Report (.pdf)",
        data=exports["pdf"],
        file_name=f"{stem}.pdf",
        mime="application/pdf",
        width="stretch",
    )
    c3.download_button(
        "Download JSON Summary (.json)",
        data=exports["json"],
        file_name=f"{stem}.json",
        mime="application/json",
        width="stretch",
    )

    c4, c5, c6 = st.columns(3)
    c4.download_button(
        "Download CSV Table (.csv)",
        data=exports["csv"],
        file_name=f"{stem}.csv",
        mime="text/csv",
        width="stretch",
    )
    c5.download_button(
        "Download Composite Figure (.png)",
        data=exports["png"],
        file_name=f"{stem}.png",
        mime="image/png",
        width="stretch",
    )
    c6.download_button(
        "Download Raw Data (.csv)",
        data=exports["raw_data_csv"],
        file_name=f"{stem}_raw_data.csv",
        mime="text/csv",
        width="stretch",
    )

    with st.expander("Preview Composite Report Figure", expanded=True):
        st.image(exports["png"], caption="Composite Technical Report Figure")


def main() -> None:
    st.set_page_config(
        page_title="Corrosion Simulator",
        page_icon="⚗️",
        layout="wide",
    )
    st.title("Semiconductor Wet-Cleaning Corrosion Simulator")
    st.caption(
        "Physics-informed and ML-assisted electrochemical polarization, bimetallic galvanic coupling, "
        "time-resolved thickness loss risk, multi-recipe comparison, and automated report export."
    )

    controller = get_controller()
    request = _render_sidebar(controller)

    single_payload = controller.run_single_metal_analysis(request)
    galvanic_payload = (
        controller.run_galvanic_analysis(request)
        if request.secondary_metal is not None
        else None
    )

    tab_single, tab_galvanic, tab_compare, tab_export = st.tabs(
        ["Single Metal Analysis", "Galvanic Analysis", "Recipe Comparison", "Report Export"]
    )
    with tab_single:
        _render_single_metal_view(single_payload)
    with tab_galvanic:
        _render_galvanic_view(galvanic_payload)
        _render_galvanic_series_explorer(controller)
    with tab_compare:
        comparison_payload = _render_recipe_comparison_view(controller, request)
    with tab_export:
        _render_report_export_view(single_payload, galvanic_payload, comparison_payload)


if __name__ == "__main__":
    main()
