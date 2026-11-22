"""Interactive Streamlit dashboard for the Corrosion Simulator.

Launch with:
    .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from corrosion_simulator.platform_controller import (
    CorrosionPlatformController,
    GalvanicViewPayload,
    ManualElectrochemicalInput,
    SimulationRequest,
    SingleMetalViewPayload,
)

SOURCE_LABELS = {
    "ml": "ML Prediction (Hybrid Physics-ML)",
    "database": "Database Lookup (Nearest Neighbor)",
    "manual": "Manual Override",
}

KINETICS_LABELS = {
    "auto": "Auto (Pourbaix Stability Regime)",
    "linear": "Linear",
    "passivation": "Passivation (Self-Limiting)",
    "power_law": "Power-Law (Diffusion / Inhibited)",
}


@st.cache_resource(show_spinner="Loading harmonized electrochemical database and training ML models...")
def get_controller() -> CorrosionPlatformController:
    """Load database and fit ML predictor once per Streamlit server process."""
    return CorrosionPlatformController.from_default_data()


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
) -> None:
    if confidence_tier == "High":
        st.success(f"**Confidence Tier:** {confidence_tier} — Query lies well within the Applicability Domain.")
    elif confidence_tier == "Medium":
        st.warning(f"**Confidence Tier:** {confidence_tier} — Moderate coverage in the Applicability Domain.")
    else:
        st.error(f"**Confidence Tier:** {confidence_tier} — Extrapolation or sparse data detected.")

    if warnings:
        with st.expander(f"Applicability Domain & Data-Sufficiency Warnings ({len(warnings)})", expanded=(confidence_tier != "High")):
            for item in warnings:
                st.warning(item)

    if recommendations:
        with st.expander(f"Suggested Follow-Up Experiments ({len(recommendations)})", expanded=(confidence_tier == "Low")):
            for rec in recommendations:
                st.markdown(f"- {rec}")


def _render_sidebar(controller: CorrosionPlatformController) -> SimulationRequest:
    options = controller.available_options()
    metals = options["metals"]

    st.sidebar.title("Simulation Parameters")

    st.sidebar.subheader("1. Metal Selection")
    primary_metal = st.sidebar.selectbox("Primary Metal", options=metals, index=0)

    secondary_choices = ["None (Single Metal Only)"] + [m for m in metals if m != primary_metal]
    default_secondary_idx = secondary_choices.index("Co") if "Co" in secondary_choices else 1
    secondary_selection = st.sidebar.selectbox(
        "Secondary Metal (Optional Galvanic Partner)",
        options=secondary_choices,
        index=default_secondary_idx,
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
    oxidizer_options = options["oxidizer_types"]
    default_ox = oxidizer_options.index("H2O2") if "H2O2" in oxidizer_options else 0
    oxidizer_type = st.sidebar.selectbox("Oxidizer Type", options=oxidizer_options, index=default_ox)
    oxidizer_disabled = oxidizer_type == "None"
    oxidizer_conc_raw = st.sidebar.number_input(
        "Oxidizer Concentration (wt% or M)",
        min_value=0.0,
        max_value=50.0,
        value=1.0,
        step=0.1,
        disabled=oxidizer_disabled,
    )
    oxidizer_conc = 0.0 if oxidizer_disabled else float(oxidizer_conc_raw)

    acid_options = options["acid_types"]
    acid_type = st.sidebar.selectbox("Acid / Complexing Agent", options=acid_options, index=0)
    acid_disabled = acid_type == "None"
    acid_conc_raw = st.sidebar.number_input(
        "Acid / Complexing Concentration (wt% or M)",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=0.1,
        disabled=acid_disabled,
    )
    acid_conc = 0.0 if acid_disabled else float(acid_conc_raw)

    inhibitor_options = options["inhibitor_types"]
    inhibitor_type = st.sidebar.selectbox("Corrosion Inhibitor", options=inhibitor_options, index=0)
    inhibitor_disabled = inhibitor_type == "None"
    inhibitor_conc_raw = st.sidebar.number_input(
        "Inhibitor Concentration (mM or wt%)",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=0.1,
        disabled=inhibitor_disabled,
    )
    inhibitor_conc = 0.0 if inhibitor_disabled else float(inhibitor_conc_raw)

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
    )

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
        oxidizer_type=oxidizer_type,
        oxidizer_concentration=float(oxidizer_conc),
        acid_type=acid_type,
        acid_concentration=float(acid_conc),
        inhibitor_type=inhibitor_type,
        inhibitor_concentration=float(inhibitor_conc),
        pH=float(pH),
        temperature_C=float(temperature_C),
        immersion_time_s=float(immersion_time_s),
        d_crit_nm=float(d_crit_nm),
        kinetics_model=kinetics_model,
        d_max_nm=float(d_max_nm),
        power_law_n=float(power_law_n),
        parameter_source=parameter_source,
        manual_primary=manual_primary,
        manual_secondary=manual_secondary,
    )


def _render_single_metal_view(payload: SingleMetalViewPayload) -> None:
    st.subheader(f"Single Metal Analysis — {payload.request.primary_metal}")
    _render_confidence_and_warnings(
        payload.confidence_tier,
        payload.warnings,
        payload.experimental_recommendations,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Initial Corrosion Rate (CR₀)",
        f"{payload.initial_corrosion_rate_nm_min:.3g} nm/min",
        delta=f"{payload.initial_corrosion_rate_mm_yr:.3g} mm/yr",
        delta_color="off",
    )
    c2.metric(
        f"Corrosion Depth @ {payload.request.immersion_time_s:.0f} s",
        f"{payload.corrosion_depth_nm:.3f} nm",
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
        _format_time_s(payload.time_to_threshold_s),
        delta=ttt_ci,
        delta_color="off",
    )
    c4.metric(
        "Thickness Loss Risk",
        payload.thickness_loss_risk_tier,
        delta=f"Score R(t) = {payload.thickness_loss_risk_score:.3f}",
        delta_color="off",
    )

    st.markdown("#### Electrochemical Parameters & Uncertainty (±1σ)")
    st.dataframe(payload.parameter_table, width="stretch", hide_index=True)

    st.plotly_chart(payload.polarization_figure, width="stretch")
    st.plotly_chart(payload.kinetics_risk_figure, width="stretch")


def _render_galvanic_view(payload: GalvanicViewPayload | None) -> None:
    if payload is None:
        st.info("Select a Secondary Metal in the sidebar to run Bimetallic Galvanic Analysis.")
        return

    st.subheader(
        f"Galvanic Analysis — {payload.active_metal} (Active Metal) coupled with {payload.noble_metal} (Noble Metal)"
    )
    _render_confidence_and_warnings(
        payload.confidence_tier,
        payload.warnings,
        payload.experimental_recommendations,
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
        delta=f"Accelerated CR₀ = {payload.accelerated_cr0_nm_min:.3g} nm/min",
        delta_color="off",
    )
    coupled_ttt_ci = (
        _format_mc_time_interval(payload.mc_coupled_kinetics.time_to_threshold_summary)
        if payload.mc_coupled_kinetics is not None
        else None
    )
    c4.metric(
        "Coupled Time to Threshold",
        _format_time_s(payload.coupled_time_to_threshold_s),
        delta=coupled_ttt_ci or f"Uncoupled: {_format_time_s(payload.uncoupled_time_to_threshold_s)}",
        delta_color="off",
    )

    comparison_df = pd.DataFrame(
        [
            {
                "Condition": f"{payload.active_metal} Uncoupled",
                "CR₀ (nm/min)": payload.uncoupled_cr0_nm_min,
                f"Corrosion Depth @ {payload.request.immersion_time_s:.0f} s (nm)": payload.uncoupled_depth_nm,
                "Time to Threshold": _format_time_s(payload.uncoupled_time_to_threshold_s),
                "Thickness Loss Risk Score": round(payload.uncoupled_risk_score, 3),
                "Risk Tier": payload.uncoupled_risk_tier,
            },
            {
                "Condition": f"{payload.active_metal} Coupled to {payload.noble_metal} (A_ratio={payload.request.cathode_to_anode_area_ratio:g})",
                "CR₀ (nm/min)": payload.accelerated_cr0_nm_min,
                f"Corrosion Depth @ {payload.request.immersion_time_s:.0f} s (nm)": payload.coupled_depth_nm,
                "Time to Threshold": _format_time_s(payload.coupled_time_to_threshold_s),
                "Thickness Loss Risk Score": round(payload.coupled_risk_score, 3),
                "Risk Tier": payload.coupled_risk_tier,
            },
        ]
    )
    st.markdown("#### Uncoupled vs. Galvanically Coupled Summary")
    st.dataframe(comparison_df, width="stretch", hide_index=True)

    st.plotly_chart(payload.galvanic_polarization_figure, width="stretch")
    st.plotly_chart(payload.comparative_kinetics_figure, width="stretch")


def main() -> None:
    st.set_page_config(
        page_title="Corrosion Simulator",
        page_icon="⚗️",
        layout="wide",
    )
    st.title("Semiconductor Wet-Cleaning Corrosion Simulator")
    st.caption(
        "Physics-informed and ML-assisted electrochemical polarization, bimetallic galvanic coupling, and time-resolved thickness loss risk analysis."
    )

    controller = get_controller()
    request = _render_sidebar(controller)

    single_payload = controller.run_single_metal_analysis(request)
    galvanic_payload = (
        controller.run_galvanic_analysis(request)
        if request.secondary_metal is not None
        else None
    )

    tab_single, tab_galvanic = st.tabs(["Single Metal Analysis", "Galvanic Analysis"])
    with tab_single:
        _render_single_metal_view(single_payload)
    with tab_galvanic:
        _render_galvanic_view(galvanic_payload)


if __name__ == "__main__":
    main()
