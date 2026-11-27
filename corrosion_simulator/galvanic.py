"""Phase 2 bimetallic galvanic mixed-potential solver, Monte Carlo uncertainty propagation, and Tafel overlay plotting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from corrosion_simulator.database import (
    ExposureCondition,
    ExposureLookupResult,
    HarmonizedDatabase,
)
from corrosion_simulator.ml_predictor import (
    DEFAULT_ECORR_CORRELATION,
    DEFAULT_MASS_TRANSPORT_LIMIT_UA_CM2,
    ECORR_THERMO_MAX_V_SHE,
    ECORR_THERMO_MIN_V_SHE,
    MLParameterPrediction,
    sample_clipped_icorr,
    sample_lognormal_tafel_slope,
)
from corrosion_simulator.physics import (
    FaradayCorrosionResult,
    PolarizationParameters,
    calculate_faraday_corrosion_rate,
    generate_polarization_curve,
    has_faraday_properties,
    wagner_traud_anodic_current,
    wagner_traud_net_current,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class GalvanicCouplingResult:
    """Steady-state bimetallic galvanic coupling solution on an Active Metal / Noble Metal pair."""

    active_metal: str
    noble_metal: str
    active_parameters: PolarizationParameters
    noble_parameters: PolarizationParameters
    cathode_to_anode_area_ratio: float
    mixed_potential_V_SHE: float
    galvanic_current_density_uA_cm2: float
    active_net_current_uA_cm2: float
    noble_net_current_uA_cm2: float
    current_balance_residual_uA_cm2: float
    acceleration_factor: float
    uncoupled_faraday: FaradayCorrosionResult | None
    coupled_faraday: FaradayCorrosionResult | None
    solution_id: str | None = None
    temperature_C: float | None = None

    @property
    def uncoupled_cr_nm_min(self) -> float:
        return (
            self.uncoupled_faraday.corrosion_rate_nm_min
            if self.uncoupled_faraday is not None
            else float("nan")
        )

    @property
    def coupled_cr_nm_min(self) -> float:
        return (
            self.coupled_faraday.corrosion_rate_nm_min
            if self.coupled_faraday is not None
            else float("nan")
        )

    @property
    def uncoupled_cr_mm_yr(self) -> float:
        return self.uncoupled_faraday.corrosion_rate_mm_yr

    @property
    def coupled_cr_mm_yr(self) -> float:
        return self.coupled_faraday.corrosion_rate_mm_yr


def _coerce_polarization_params(
    params: PolarizationParameters | ExposureLookupResult,
) -> PolarizationParameters:
    if isinstance(params, ExposureLookupResult):
        return PolarizationParameters.from_lookup_result(params)
    if isinstance(params, PolarizationParameters):
        return params
    raise TypeError(
        f"Expected PolarizationParameters or ExposureLookupResult, got {type(params).__name__}"
    )


def _safe_wagner_traud_anodic(
    E_V: float,
    ecorr_V: float,
    icorr_uA_cm2: float,
    beta_a_V_dec: float,
) -> float:
    exp_a = max(-250.0, min(250.0, (E_V - ecorr_V) / beta_a_V_dec))
    return icorr_uA_cm2 * (10.0**exp_a)


def _safe_wagner_traud_net(
    E_V: float,
    ecorr_V: float,
    icorr_uA_cm2: float,
    beta_a_V_dec: float,
    beta_c_V_dec: float,
) -> float:
    exp_c = max(-250.0, min(250.0, -(E_V - ecorr_V) / beta_c_V_dec))
    return _safe_wagner_traud_anodic(E_V, ecorr_V, icorr_uA_cm2, beta_a_V_dec) - icorr_uA_cm2 * (
        10.0**exp_c
    )


def _solve_mixed_potential_scalar(
    ecorr_a: float,
    icorr_a: float,
    ba_a_V: float,
    bc_a_V: float,
    ecorr_b: float,
    icorr_b: float,
    ba_b_V: float,
    bc_b_V: float,
    area_ratio_b_over_a: float,
) -> float:
    """Solve i_net_a(E) + area_ratio_b_over_a * i_net_b(E) = 0 on [min(Ea, Eb), max(Ea, Eb)]."""
    if math.isclose(ecorr_a, ecorr_b, abs_tol=1e-15):
        return 0.5 * (ecorr_a + ecorr_b)

    e_low = min(ecorr_a, ecorr_b)
    e_high = max(ecorr_a, ecorr_b)

    def residual(E_val: float) -> float:
        return _safe_wagner_traud_net(
            E_val, ecorr_a, icorr_a, ba_a_V, bc_a_V
        ) + area_ratio_b_over_a * _safe_wagner_traud_net(E_val, ecorr_b, icorr_b, ba_b_V, bc_b_V)

    root = brentq(residual, e_low, e_high, xtol=1e-14, rtol=1e-14, maxiter=200)
    return float(root)


def solve_galvanic_couple(
    params_a: PolarizationParameters | ExposureLookupResult,
    params_b: PolarizationParameters | ExposureLookupResult,
    *,
    cathode_to_anode_area_ratio: float = 1.0,
    active_valence: int | None = None,
    immersion_time_s: float | None = None,
    material_properties: pd.DataFrame | None = None,
    allow_missing_faraday: bool = False,
) -> GalvanicCouplingResult:
    """Solve area-weighted bimetallic Mixed Potential (E_galv), Galvanic Current Density, and Faraday CR_0."""
    if not math.isfinite(cathode_to_anode_area_ratio) or cathode_to_anode_area_ratio <= 0.0:
        raise ValueError(
            f"cathode_to_anode_area_ratio must be strictly positive and finite, "
            f"got {cathode_to_anode_area_ratio!r}"
        )

    pa = _coerce_polarization_params(params_a)
    pb = _coerce_polarization_params(params_b)

    metal_a = (pa.metal or "").strip()
    metal_b = (pb.metal or "").strip()
    if not metal_a or not metal_b:
        raise ValueError(
            "Both PolarizationParameters inputs must specify a non-empty metal identifier."
        )
    if metal_a == metal_b:
        raise ValueError(
            f"Galvanic coupling requires two distinct metals, got {metal_a!r} twice."
        )

    if pa.solution_id is not None and pb.solution_id is not None and pa.solution_id != pb.solution_id:
        raise ValueError(
            f"Galvanic coupling requires the same Solution, got "
            f"{pa.solution_id!r} vs {pb.solution_id!r}."
        )
    if (
        pa.temperature_C is not None
        and pb.temperature_C is not None
        and not math.isclose(pa.temperature_C, pb.temperature_C, abs_tol=1e-6)
    ):
        raise ValueError(
            f"Galvanic coupling requires the same temperature, got "
            f"{pa.temperature_C:.2f} °C vs {pb.temperature_C:.2f} °C."
        )

    if pa.Ecorr_V_SHE <= pb.Ecorr_V_SHE:
        active_p, noble_p = pa, pb
        active_metal, noble_metal = metal_a, metal_b
    else:
        active_p, noble_p = pb, pa
        active_metal, noble_metal = metal_b, metal_a

    area_ratio = float(cathode_to_anode_area_ratio)
    e_galv = _solve_mixed_potential_scalar(
        active_p.Ecorr_V_SHE,
        active_p.icorr_uA_cm2,
        active_p.beta_a_V_dec,
        active_p.beta_c_V_dec,
        noble_p.Ecorr_V_SHE,
        noble_p.icorr_uA_cm2,
        noble_p.beta_a_V_dec,
        noble_p.beta_c_V_dec,
        area_ratio,
    )

    i_net_active = wagner_traud_net_current(e_galv, active_p)
    i_net_noble = wagner_traud_net_current(e_galv, noble_p)
    balance_residual = i_net_active + area_ratio * i_net_noble

    i_galv = wagner_traud_anodic_current(e_galv, active_p)
    accel_factor = i_galv / active_p.icorr_uA_cm2

    if allow_missing_faraday and not has_faraday_properties(active_metal, material_properties):
        uncoupled_faraday: FaradayCorrosionResult | None = None
        coupled_faraday: FaradayCorrosionResult | None = None
    else:
        uncoupled_faraday = calculate_faraday_corrosion_rate(
            active_metal,
            active_p.icorr_uA_cm2,
            valence=active_valence,
            immersion_time_s=immersion_time_s,
            material_properties=material_properties,
        )
        coupled_faraday = calculate_faraday_corrosion_rate(
            active_metal,
            i_galv,
            valence=active_valence,
            immersion_time_s=immersion_time_s,
            material_properties=material_properties,
        )

    sol_id = active_p.solution_id if active_p.solution_id is not None else noble_p.solution_id
    temp_c = active_p.temperature_C if active_p.temperature_C is not None else noble_p.temperature_C

    return GalvanicCouplingResult(
        active_metal=active_metal,
        noble_metal=noble_metal,
        active_parameters=active_p,
        noble_parameters=noble_p,
        cathode_to_anode_area_ratio=area_ratio,
        mixed_potential_V_SHE=e_galv,
        galvanic_current_density_uA_cm2=i_galv,
        active_net_current_uA_cm2=i_net_active,
        noble_net_current_uA_cm2=i_net_noble,
        current_balance_residual_uA_cm2=balance_residual,
        acceleration_factor=accel_factor,
        uncoupled_faraday=uncoupled_faraday,
        coupled_faraday=coupled_faraday,
        solution_id=sol_id,
        temperature_C=temp_c,
    )


def solve_galvanic_couple_from_database(
    database: HarmonizedDatabase,
    metal_a: str,
    metal_b: str,
    *,
    pH: float = 7.0,
    oxidizer_type: str = "None",
    oxidizer_concentration: float = 0.0,
    temperature_C: float = 25.0,
    solution_id: str | None = None,
    cathode_to_anode_area_ratio: float = 1.0,
    active_valence: int | None = None,
    immersion_time_s: float | None = None,
) -> GalvanicCouplingResult:
    """Look up two metals under the same Solution and Temperature and solve their galvanic coupling."""
    name_a = str(metal_a).strip()
    name_b = str(metal_b).strip()
    if not name_a or not name_b:
        raise ValueError("Both metal names must be non-empty.")
    if name_a == name_b:
        raise ValueError(f"Galvanic coupling requires two distinct metals, got {name_a!r} twice.")

    eff_pH = pH
    eff_ox_type = oxidizer_type
    eff_ox_conc = oxidizer_concentration
    if solution_id is not None:
        sol_attrs = database.get_solution_attributes(solution_id)
        eff_pH = float(sol_attrs["pH"])
        eff_ox_type = str(sol_attrs["oxidizer_type"])
        eff_ox_conc = float(sol_attrs["oxidizer_concentration"])

    def _lookup_for(metal_name: str) -> ExposureLookupResult:
        return database.lookup_exposure_condition(
            ExposureCondition(
                metal=metal_name,
                pH=eff_pH,
                oxidizer_type=eff_ox_type,
                oxidizer_concentration=eff_ox_conc,
                temperature_C=temperature_C,
                solution_id=solution_id,
            )
        )

    lookup_a = _lookup_for(name_a)
    lookup_b = _lookup_for(name_b)
    return solve_galvanic_couple(
        lookup_a,
        lookup_b,
        cathode_to_anode_area_ratio=cathode_to_anode_area_ratio,
        active_valence=active_valence,
        immersion_time_s=immersion_time_s,
        material_properties=database.material_properties,
    )


@dataclass(frozen=True)
class MonteCarloGalvanicResult:
    """Monte Carlo uncertainty propagation results for bimetallic galvanic coupling."""

    active_metal: str
    noble_metal: str
    cathode_to_anode_area_ratio: float
    n_draws: int
    nominal_result: GalvanicCouplingResult
    mixed_potential_samples_V_SHE: np.ndarray
    galvanic_current_samples_uA_cm2: np.ndarray
    acceleration_factor_samples: np.ndarray
    coupled_cr_nm_min_samples: np.ndarray
    coupled_cr_mm_yr_samples: np.ndarray
    mixed_potential_summary: dict[str, float]
    galvanic_current_summary: dict[str, float]
    acceleration_factor_summary: dict[str, float]
    coupled_cr_nm_min_summary: dict[str, float]
    coupled_cr_mm_yr_summary: dict[str, float]
    polarity_flip_fraction: float = 0.0


def _summarize_array(samples: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "p05": float(np.percentile(samples, 5.0)),
        "p50": float(np.percentile(samples, 50.0)),
        "p95": float(np.percentile(samples, 95.0)),
    }


def _to_nominal_params(
    item: MLParameterPrediction | PolarizationParameters | ExposureLookupResult,
) -> PolarizationParameters:
    if isinstance(item, MLParameterPrediction):
        return item.to_polarization_parameters()
    return _coerce_polarization_params(item)


def _sample_parameter_draws(
    item: MLParameterPrediction | PolarizationParameters | ExposureLookupResult,
    n_draws: int,
    rng: np.random.Generator,
    *,
    z_ecorr: np.ndarray | None = None,
    limiting_current_uA_cm2: float = DEFAULT_MASS_TRANSPORT_LIMIT_UA_CM2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample (E_corr, i_corr, beta_a_V_dec, beta_c_V_dec) arrays using log-normal Tafel slopes and i_lim cap."""
    if isinstance(item, MLParameterPrediction):
        z_e = z_ecorr if z_ecorr is not None else rng.standard_normal(n_draws)
        ecorr = np.clip(
            item.Ecorr_V_SHE.mean + max(item.Ecorr_V_SHE.std, 0.0) * z_e,
            ECORR_THERMO_MIN_V_SHE,
            ECORR_THERMO_MAX_V_SHE,
        )
        _, icorr = sample_clipped_icorr(
            item.log10_icorr_uA_cm2.mean,
            item.log10_icorr_uA_cm2.std,
            n_draws,
            rng,
            limiting_current_uA_cm2=limiting_current_uA_cm2,
        )
        ba_mv = sample_lognormal_tafel_slope(
            item.beta_a_mV_dec.mean, item.beta_a_mV_dec.std, n_draws, rng
        )
        bc_mv = sample_lognormal_tafel_slope(
            item.beta_c_mV_dec.mean, item.beta_c_mV_dec.std, n_draws, rng
        )
        return ecorr, icorr, ba_mv / 1000.0, bc_mv / 1000.0

    p = _coerce_polarization_params(item)
    return (
        np.full(n_draws, p.Ecorr_V_SHE, dtype=float),
        np.full(n_draws, min(p.icorr_uA_cm2, limiting_current_uA_cm2), dtype=float),
        np.full(n_draws, p.beta_a_V_dec, dtype=float),
        np.full(n_draws, p.beta_c_V_dec, dtype=float),
    )


def propagate_galvanic_uncertainty_monte_carlo(
    prediction_a: MLParameterPrediction | PolarizationParameters | ExposureLookupResult,
    prediction_b: MLParameterPrediction | PolarizationParameters | ExposureLookupResult,
    *,
    cathode_to_anode_area_ratio: float = 1.0,
    n_draws: int = 500,
    seed: int | None = 42,
    active_valence: int | None = None,
    material_properties: pd.DataFrame | None = None,
    ecorr_correlation: float = DEFAULT_ECORR_CORRELATION,
    limiting_current_uA_cm2: float = DEFAULT_MASS_TRANSPORT_LIMIT_UA_CM2,
) -> MonteCarloGalvanicResult:
    """Propagate parameter uncertainties through the area-weighted bimetallic Mixed Potential solver.

    Tracks the nominal Active Metal consistently across all draws (cathodic protection when
    polarity flips, i.e., AF < 1), applies shared-solution E_corr correlation, log-normal Tafel
    slope sampling, and a physical mass-transport limiting current cap.
    """
    if n_draws < 1:
        raise ValueError(f"n_draws must be positive, got {n_draws!r}")
    rho = float(ecorr_correlation)
    if not math.isfinite(rho) or not (0.0 <= rho <= 0.99):
        raise ValueError(f"ecorr_correlation must lie in [0.0, 0.99], got {ecorr_correlation!r}")
    i_lim = float(limiting_current_uA_cm2)
    if not math.isfinite(i_lim) or i_lim <= 0.0:
        raise ValueError(
            f"limiting_current_uA_cm2 must be strictly positive and finite, got {limiting_current_uA_cm2!r}"
        )

    nom_a = _to_nominal_params(prediction_a)
    nom_b = _to_nominal_params(prediction_b)
    nominal = solve_galvanic_couple(
        nom_a,
        nom_b,
        cathode_to_anode_area_ratio=cathode_to_anode_area_ratio,
        active_valence=active_valence,
        material_properties=material_properties,
    )

    # Align sampling order with nominal Active Metal and Noble Metal
    if nom_a.Ecorr_V_SHE <= nom_b.Ecorr_V_SHE:
        active_src, noble_src = prediction_a, prediction_b
    else:
        active_src, noble_src = prediction_b, prediction_a

    rng = np.random.default_rng(seed)
    z_act = rng.standard_normal(n_draws)
    z_ind = rng.standard_normal(n_draws)
    z_nob = rho * z_act + math.sqrt(max(0.0, 1.0 - rho * rho)) * z_ind

    e_act, i_act, ba_act, bc_act = _sample_parameter_draws(
        active_src, n_draws, rng, z_ecorr=z_act, limiting_current_uA_cm2=i_lim
    )
    e_nob, i_nob, ba_nob, bc_nob = _sample_parameter_draws(
        noble_src, n_draws, rng, z_ecorr=z_nob, limiting_current_uA_cm2=i_lim
    )

    unit_faraday_act = calculate_faraday_corrosion_rate(
        nominal.active_metal,
        1.0,
        valence=active_valence,
        material_properties=material_properties,
    )

    area_ratio = float(cathode_to_anode_area_ratio)
    e_galv_draws = np.empty(n_draws, dtype=float)
    i_galv_draws = np.empty(n_draws, dtype=float)
    af_draws = np.empty(n_draws, dtype=float)
    cr_nm_min_draws = np.empty(n_draws, dtype=float)
    cr_mm_yr_draws = np.empty(n_draws, dtype=float)

    for k in range(n_draws):
        ea_k, ia_k, baa_k, bca_k = float(e_act[k]), float(i_act[k]), float(ba_act[k]), float(bc_act[k])
        en_k, in_k, ban_k, bcn_k = float(e_nob[k]), float(i_nob[k]), float(ba_nob[k]), float(bc_nob[k])
        eg = _solve_mixed_potential_scalar(
            ea_k, ia_k, baa_k, bca_k, en_k, in_k, ban_k, bcn_k, area_ratio
        )
        e_galv_draws[k] = eg
        # Always evaluate dissolution of the nominal Active Metal at E_galv
        # (when ea_k > en_k, eg < ea_k -> nominal Active Metal is cathodically protected, AF < 1.0)
        ig = min(_safe_wagner_traud_anodic(eg, ea_k, ia_k, baa_k), i_lim)
        i_galv_draws[k] = ig
        af_draws[k] = ig / ia_k
        cr_nm_min_draws[k] = ig * unit_faraday_act.corrosion_rate_nm_min
        cr_mm_yr_draws[k] = ig * unit_faraday_act.corrosion_rate_mm_yr

    flip_fraction = float(np.mean(e_act > e_nob))

    return MonteCarloGalvanicResult(
        active_metal=nominal.active_metal,
        noble_metal=nominal.noble_metal,
        cathode_to_anode_area_ratio=area_ratio,
        n_draws=int(n_draws),
        nominal_result=nominal,
        mixed_potential_samples_V_SHE=e_galv_draws,
        galvanic_current_samples_uA_cm2=i_galv_draws,
        acceleration_factor_samples=af_draws,
        coupled_cr_nm_min_samples=cr_nm_min_draws,
        coupled_cr_mm_yr_samples=cr_mm_yr_draws,
        mixed_potential_summary=_summarize_array(e_galv_draws),
        galvanic_current_summary=_summarize_array(i_galv_draws),
        acceleration_factor_summary=_summarize_array(af_draws),
        coupled_cr_nm_min_summary=_summarize_array(cr_nm_min_draws),
        coupled_cr_mm_yr_summary=_summarize_array(cr_mm_yr_draws),
        polarity_flip_fraction=flip_fraction,
    )


def plot_galvanic_tafel_overlay(
    result: GalvanicCouplingResult,
    *,
    mc_result: MonteCarloGalvanicResult | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    show_area_weighted_noble: bool = True,
    potential_padding_V: float = 0.15,
    num_points: int = 401,
) -> tuple[Figure, Axes]:
    """Render overlaid bimetallic Tafel plots annotated with E_galv and the shift from i_corr,active to i_galv."""
    import matplotlib.pyplot as plt

    act = result.active_parameters
    nob = result.noble_parameters
    ratio = result.cathode_to_anode_area_ratio

    e_min = min(act.Ecorr_V_SHE, nob.Ecorr_V_SHE) - potential_padding_V
    e_max = max(act.Ecorr_V_SHE, nob.Ecorr_V_SHE) + potential_padding_V
    potentials = np.linspace(e_min, e_max, num_points)

    curve_act = generate_polarization_curve(act, potentials_V_SHE=potentials)
    curve_nob = generate_polarization_curve(nob, potentials_V_SHE=potentials)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 5.4))
    else:
        fig = ax.figure

    # Active Metal polarization curves (net + anodic branch)
    ax.plot(
        curve_act.log10_abs_net_uA_cm2,
        potentials,
        color="#d62728",
        linewidth=2.1,
        label=f"{result.active_metal} (Active Metal) |i_net|",
    )
    ax.plot(
        curve_act.log10_anodic_uA_cm2,
        potentials,
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
        alpha=0.65,
        label=f"{result.active_metal} Anodic Branch (β_a={act.beta_a_mV_dec:.0f} mV/dec)",
    )

    # Noble Metal polarization curves (per-area and area-weighted when ratio != 1)
    ax.plot(
        curve_nob.log10_abs_net_uA_cm2,
        potentials,
        color="#1f77b4",
        linewidth=2.1 if not (show_area_weighted_noble and not math.isclose(ratio, 1.0)) else 1.5,
        alpha=1.0 if not (show_area_weighted_noble and not math.isclose(ratio, 1.0)) else 0.55,
        label=f"{result.noble_metal} (Noble Metal) |i_net|",
    )
    if show_area_weighted_noble and not math.isclose(ratio, 1.0):
        weighted_log_net = np.where(
            curve_nob.abs_net_current_uA_cm2 > 0.0,
            np.log10(ratio * curve_nob.abs_net_current_uA_cm2),
            np.nan,
        )
        ax.plot(
            weighted_log_net,
            potentials,
            color="#0b4f8a",
            linestyle="-.",
            linewidth=2.0,
            label=f"{result.noble_metal} Area-Weighted ({ratio:.2g}× |i_net|)",
        )

    # Optional Monte Carlo 90% confidence band on E_galv
    if mc_result is not None:
        ax.axhspan(
            mc_result.mixed_potential_summary["p05"],
            mc_result.mixed_potential_summary["p95"],
            color="#2ca02c",
            alpha=0.14,
            label=f"E_galv 90% Band (P05–P95, N={mc_result.n_draws})",
        )

    # Mixed Potential operating line
    ax.axhline(
        result.mixed_potential_V_SHE,
        color="#2ca02c",
        linestyle="-",
        linewidth=1.6,
        alpha=0.9,
        label=f"Mixed Potential E_galv = {result.mixed_potential_V_SHE:+.3f} V vs. SHE",
    )

    # Isolated Active Metal corrosion point vs. coupled galvanic operating point
    log_icorr_act = math.log10(act.icorr_uA_cm2)
    log_ig = math.log10(result.galvanic_current_density_uA_cm2)

    ax.scatter(
        [log_icorr_act],
        [act.Ecorr_V_SHE],
        color="#ff7f0e",
        edgecolor="#111111",
        s=65,
        zorder=6,
        label=f"Isolated {result.active_metal} (i_corr={act.icorr_uA_cm2:.2g} µA/cm²)",
    )
    ax.scatter(
        [log_ig],
        [result.mixed_potential_V_SHE],
        color="#2ca02c",
        edgecolor="#111111",
        marker="D",
        s=75,
        zorder=7,
        label=f"Coupled {result.active_metal} (i_galv={result.galvanic_current_density_uA_cm2:.2g} µA/cm², AF={result.acceleration_factor:.2f}×)",
    )

    # Arrow illustrating shift from isolated (i_corr,active, E_corr,active) to coupled (i_galv, E_galv)
    ax.annotate(
        "",
        xy=(log_ig, result.mixed_potential_V_SHE),
        xytext=(log_icorr_act, act.Ecorr_V_SHE),
        arrowprops={"arrowstyle": "->", "color": "#2ca02c", "lw": 1.6, "linestyle": ":"},
        zorder=5,
    )
    ax.annotate(
        f"E_galv = {result.mixed_potential_V_SHE:+.3f} V\n"
        f"i_galv = {result.galvanic_current_density_uA_cm2:.3g} µA/cm²\n"
        f"AF = {result.acceleration_factor:.2f}× (CR₀ = {result.coupled_cr_nm_min:.3g} nm/min)",
        xy=(log_ig, result.mixed_potential_V_SHE),
        xytext=(14, -28),
        textcoords="offset points",
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "#eefaf0",
            "edgecolor": "#2ca02c",
            "alpha": 0.92,
        },
        arrowprops={"arrowstyle": "->", "color": "#2ca02c", "lw": 1.1},
    )

    ax.set_xlabel("log10(|i|) [µA/cm²]")
    ax.set_ylabel("Potential E [V vs. SHE]")
    ax.set_title(
        title
        or (
            f"Bimetallic Galvanic Tafel Overlay: {result.active_metal} (Active) vs. "
            f"{result.noble_metal} (Noble), A_noble/A_active = {ratio:.2g}"
        )
    )
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="best", fontsize=8.0)
    return fig, ax
