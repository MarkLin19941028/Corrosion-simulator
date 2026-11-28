"""Physics-clipped uncertainty contract and Monte Carlo propagation (ADR-0001 & ADR-0003).

This module depends strictly on ``numpy`` and ``corrosion_simulator.physics`` (zero ML/scikit-learn
dependencies), serving as the shared four-parameter uncertainty seam between predictors and physics solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from corrosion_simulator.physics import (
    PolarizationParameters,
    calculate_faraday_corrosion_rate,
)

if TYPE_CHECKING:
    import pandas as pd

TAFEL_SLOPE_MIN_MV_DEC: Final[float] = 20.0
TAFEL_SLOPE_MAX_MV_DEC: Final[float] = 500.0
ECORR_THERMO_MIN_V_SHE: Final[float] = -2.5
ECORR_THERMO_MAX_V_SHE: Final[float] = 2.5
LOG10_ICORR_MIN: Final[float] = -6.0
LOG10_ICORR_MAX: Final[float] = 6.0
DEFAULT_MASS_TRANSPORT_LIMIT_UA_CM2: Final[float] = 1e5
DEFAULT_ECORR_CORRELATION: Final[float] = 0.6


def clip_target_mean(target_name: str, value: float) -> float:
    """Enforce physical and thermodynamic bounds on a predicted electrochemical target mean."""
    if target_name == "Ecorr_V_SHE":
        return float(np.clip(value, ECORR_THERMO_MIN_V_SHE, ECORR_THERMO_MAX_V_SHE))
    if target_name in ("beta_a_mV_dec", "beta_c_mV_dec"):
        return float(np.clip(value, TAFEL_SLOPE_MIN_MV_DEC, TAFEL_SLOPE_MAX_MV_DEC))
    if target_name == "log10_icorr_uA_cm2":
        return float(np.clip(value, LOG10_ICORR_MIN, LOG10_ICORR_MAX))
    raise ValueError(f"Unknown target_name: {target_name!r}")


@dataclass(frozen=True)
class ParameterEstimate:
    """Predicted mean and ±1σ uncertainty for a single electrochemical parameter."""

    mean: float
    std: float


@dataclass(frozen=True)
class MLParameterPrediction:
    """Predicted four-parameter electrochemical kinetics with ±1σ uncertainties and Applicability Domain."""

    metal: str
    Ecorr_V_SHE: ParameterEstimate
    log10_icorr_uA_cm2: ParameterEstimate
    icorr_uA_cm2_median: float
    beta_a_mV_dec: ParameterEstimate
    beta_c_mV_dec: ParameterEstimate
    model_family: Any
    applicability: Any
    calibration_mode: str = "literature_only"
    lab_project: str | None = None
    residual_breakdown: Any | None = None

    def to_polarization_parameters(self) -> PolarizationParameters:
        """Convert point estimates into physically clipped PolarizationParameters."""
        ecorr = clip_target_mean("Ecorr_V_SHE", self.Ecorr_V_SHE.mean)
        icorr = float(10.0 ** clip_target_mean("log10_icorr_uA_cm2", self.log10_icorr_uA_cm2.mean))
        beta_a = clip_target_mean("beta_a_mV_dec", self.beta_a_mV_dec.mean)
        beta_c = clip_target_mean("beta_c_mV_dec", self.beta_c_mV_dec.mean)
        return PolarizationParameters(
            Ecorr_V_SHE=ecorr,
            icorr_uA_cm2=icorr,
            beta_a_mV_dec=beta_a,
            beta_c_mV_dec=beta_c,
            metal=self.metal,
        )


@dataclass(frozen=True)
class MonteCarloPolarizationBand:
    """Percentile uncertainty bands for activation-controlled Polarization Curves."""

    potential_V_SHE: np.ndarray
    log10_abs_net_mean: np.ndarray
    log10_abs_net_p05: np.ndarray
    log10_abs_net_p50: np.ndarray
    log10_abs_net_p95: np.ndarray
    anodic_p05_uA_cm2: np.ndarray
    anodic_p50_uA_cm2: np.ndarray
    anodic_p95_uA_cm2: np.ndarray
    cathodic_p05_uA_cm2: np.ndarray
    cathodic_p50_uA_cm2: np.ndarray
    cathodic_p95_uA_cm2: np.ndarray
    overpotential_V: np.ndarray
    eta_log10_abs_net_p05: np.ndarray
    eta_log10_abs_net_p25: np.ndarray
    eta_log10_abs_net_p50: np.ndarray
    eta_log10_abs_net_p75: np.ndarray
    eta_log10_abs_net_p95: np.ndarray
    ecorr_p05_V_SHE: float
    ecorr_p25_V_SHE: float
    ecorr_p50_V_SHE: float
    ecorr_p75_V_SHE: float
    ecorr_p95_V_SHE: float
    log10_icorr_p05: float
    log10_icorr_p25: float
    log10_icorr_p50: float
    log10_icorr_p75: float
    log10_icorr_p95: float


@dataclass(frozen=True)
class MonteCarloPropagationResult:
    """Physics-clipped Monte Carlo uncertainty propagation results for single-metal kinetics."""

    metal: str
    n_draws: int
    Ecorr_samples_V_SHE: np.ndarray
    icorr_samples_uA_cm2: np.ndarray
    beta_a_samples_mV_dec: np.ndarray
    beta_c_samples_mV_dec: np.ndarray
    cr_nm_min_samples: np.ndarray
    cr_mm_yr_samples: np.ndarray
    cr_nm_min_summary: dict[str, float]
    cr_mm_yr_summary: dict[str, float]
    polarization_band: MonteCarloPolarizationBand


def _summarize_samples(samples: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "p05": float(np.percentile(samples, 5.0)),
        "p50": float(np.percentile(samples, 50.0)),
        "p95": float(np.percentile(samples, 95.0)),
    }


def sample_lognormal_tafel_slope(
    mean_mV_dec: float,
    std_mV_dec: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample strictly positive Tafel slopes (mV/dec) via log-normal distribution with physical bounds."""
    center = float(np.clip(mean_mV_dec, TAFEL_SLOPE_MIN_MV_DEC, TAFEL_SLOPE_MAX_MV_DEC))
    sigma = max(float(std_mV_dec), 0.0)
    if sigma == 0.0:
        return np.full(size, center, dtype=float)
    sigma_ln = math.sqrt(math.log1p((sigma / center) ** 2))
    draws = rng.lognormal(mean=math.log(center), sigma=sigma_ln, size=size)
    return np.clip(draws, TAFEL_SLOPE_MIN_MV_DEC, TAFEL_SLOPE_MAX_MV_DEC)


def sample_clipped_icorr(
    mean_log10: float,
    std_log10: float,
    size: int,
    rng: np.random.Generator,
    *,
    limiting_current_uA_cm2: float = DEFAULT_MASS_TRANSPORT_LIMIT_UA_CM2,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (log10_icorr, icorr_uA_cm2) with log-space clipping and mass-transport limit i_lim."""
    i_lim = float(limiting_current_uA_cm2)
    if not math.isfinite(i_lim) or i_lim <= 0.0:
        raise ValueError(
            f"limiting_current_uA_cm2 must be strictly positive and finite, got {limiting_current_uA_cm2!r}"
        )
    log_i = np.clip(
        rng.normal(mean_log10, max(float(std_log10), 0.0), size=size),
        LOG10_ICORR_MIN,
        math.log10(i_lim),
    )
    return log_i, np.minimum(np.power(10.0, log_i), i_lim)


def propagate_uncertainty_monte_carlo(
    prediction: MLParameterPrediction,
    *,
    n_draws: int = 500,
    seed: int | None = 42,
    valence: int | None = None,
    material_properties: pd.DataFrame | None = None,
    overpotential_min_V: float = -0.25,
    overpotential_max_V: float = 0.25,
    num_potential_points: int = 201,
    potentials_V_SHE: np.ndarray | None = None,
    limiting_current_uA_cm2: float = DEFAULT_MASS_TRANSPORT_LIMIT_UA_CM2,
) -> MonteCarloPropagationResult:
    """Propagate ML parameter uncertainties via log-space i_corr and physics-clipped Monte Carlo (ADR-0003)."""
    if n_draws < 1:
        raise ValueError(f"n_draws must be positive, got {n_draws!r}")

    rng = np.random.default_rng(seed)

    ecorr_draws = np.clip(
        rng.normal(prediction.Ecorr_V_SHE.mean, max(prediction.Ecorr_V_SHE.std, 0.0), size=n_draws),
        ECORR_THERMO_MIN_V_SHE,
        ECORR_THERMO_MAX_V_SHE,
    )
    beta_a_draws = sample_lognormal_tafel_slope(
        prediction.beta_a_mV_dec.mean, prediction.beta_a_mV_dec.std, n_draws, rng
    )
    beta_c_draws = sample_lognormal_tafel_slope(
        prediction.beta_c_mV_dec.mean, prediction.beta_c_mV_dec.std, n_draws, rng
    )
    log_i_draws, icorr_draws = sample_clipped_icorr(
        prediction.log10_icorr_uA_cm2.mean,
        prediction.log10_icorr_uA_cm2.std,
        n_draws,
        rng,
        limiting_current_uA_cm2=limiting_current_uA_cm2,
    )

    unit_faraday = calculate_faraday_corrosion_rate(
        prediction.metal,
        1.0,
        valence=valence,
        material_properties=material_properties,
    )
    cr_nm_min_draws = icorr_draws * unit_faraday.corrosion_rate_nm_min
    cr_mm_yr_draws = icorr_draws * unit_faraday.corrosion_rate_mm_yr

    center_e = clip_target_mean("Ecorr_V_SHE", prediction.Ecorr_V_SHE.mean)
    if potentials_V_SHE is not None:
        potentials = np.asarray(potentials_V_SHE, dtype=float)
    else:
        potentials = np.linspace(
            center_e + overpotential_min_V,
            center_e + overpotential_max_V,
            num_potential_points,
        )

    eta = potentials[None, :] - ecorr_draws[:, None]
    ba_v = (beta_a_draws / 1000.0)[:, None]
    bc_v = (beta_c_draws / 1000.0)[:, None]
    icorr = icorr_draws[:, None]

    i_anodic = icorr * np.power(10.0, np.clip(eta / ba_v, -30.0, 30.0))
    i_cathodic = icorr * np.power(10.0, np.clip(-eta / bc_v, -30.0, 30.0))
    abs_net = np.maximum(np.abs(i_anodic - i_cathodic), 1e-12)
    log_abs_net = np.log10(abs_net)

    ecorr_p50 = float(np.percentile(ecorr_draws, 50.0))
    if potentials_V_SHE is not None:
        eta_grid = potentials - ecorr_p50
    else:
        eta_grid = np.linspace(overpotential_min_V, overpotential_max_V, num_potential_points)

    eta_mat = eta_grid[None, :]
    i_anodic_eta = icorr * np.power(10.0, np.clip(eta_mat / ba_v, -30.0, 30.0))
    i_cathodic_eta = icorr * np.power(10.0, np.clip(-eta_mat / bc_v, -30.0, 30.0))
    abs_net_eta = np.maximum(np.abs(i_anodic_eta - i_cathodic_eta), 1e-12)
    log_abs_net_eta = np.log10(abs_net_eta)

    band = MonteCarloPolarizationBand(
        potential_V_SHE=potentials,
        log10_abs_net_mean=np.mean(log_abs_net, axis=0),
        log10_abs_net_p05=np.percentile(log_abs_net, 5.0, axis=0),
        log10_abs_net_p50=np.percentile(log_abs_net, 50.0, axis=0),
        log10_abs_net_p95=np.percentile(log_abs_net, 95.0, axis=0),
        anodic_p05_uA_cm2=np.percentile(i_anodic, 5.0, axis=0),
        anodic_p50_uA_cm2=np.percentile(i_anodic, 50.0, axis=0),
        anodic_p95_uA_cm2=np.percentile(i_anodic, 95.0, axis=0),
        cathodic_p05_uA_cm2=np.percentile(i_cathodic, 5.0, axis=0),
        cathodic_p50_uA_cm2=np.percentile(i_cathodic, 50.0, axis=0),
        cathodic_p95_uA_cm2=np.percentile(i_cathodic, 95.0, axis=0),
        overpotential_V=eta_grid,
        eta_log10_abs_net_p05=np.percentile(log_abs_net_eta, 5.0, axis=0),
        eta_log10_abs_net_p25=np.percentile(log_abs_net_eta, 25.0, axis=0),
        eta_log10_abs_net_p50=np.percentile(log_abs_net_eta, 50.0, axis=0),
        eta_log10_abs_net_p75=np.percentile(log_abs_net_eta, 75.0, axis=0),
        eta_log10_abs_net_p95=np.percentile(log_abs_net_eta, 95.0, axis=0),
        ecorr_p05_V_SHE=float(np.percentile(ecorr_draws, 5.0)),
        ecorr_p25_V_SHE=float(np.percentile(ecorr_draws, 25.0)),
        ecorr_p50_V_SHE=ecorr_p50,
        ecorr_p75_V_SHE=float(np.percentile(ecorr_draws, 75.0)),
        ecorr_p95_V_SHE=float(np.percentile(ecorr_draws, 95.0)),
        log10_icorr_p05=float(np.percentile(log_i_draws, 5.0)),
        log10_icorr_p25=float(np.percentile(log_i_draws, 25.0)),
        log10_icorr_p50=float(np.percentile(log_i_draws, 50.0)),
        log10_icorr_p75=float(np.percentile(log_i_draws, 75.0)),
        log10_icorr_p95=float(np.percentile(log_i_draws, 95.0)),
    )

    return MonteCarloPropagationResult(
        metal=prediction.metal,
        n_draws=int(n_draws),
        Ecorr_samples_V_SHE=ecorr_draws,
        icorr_samples_uA_cm2=icorr_draws,
        beta_a_samples_mV_dec=beta_a_draws,
        beta_c_samples_mV_dec=beta_c_draws,
        cr_nm_min_samples=cr_nm_min_draws,
        cr_mm_yr_samples=cr_mm_yr_draws,
        cr_nm_min_summary=_summarize_samples(cr_nm_min_draws),
        cr_mm_yr_summary=_summarize_samples(cr_mm_yr_draws),
        polarization_band=band,
    )
