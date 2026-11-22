"""Phase 2 time-resolved corrosion kinetics, Time to Threshold, Thickness Loss Risk, and uncertainty bands."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Final, Literal, overload

import numpy as np
import pandas as pd

from corrosion_simulator.galvanic import GalvanicCouplingResult, MonteCarloGalvanicResult
from corrosion_simulator.ml_predictor import MonteCarloPropagationResult
from corrosion_simulator.physics import SECONDS_PER_MINUTE, FaradayCorrosionResult
from corrosion_simulator.seed_data import SEED_MATERIAL_PROPERTIES

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

KineticModelName = Literal["linear", "passivation", "power_law"]
PourbaixRegimeName = Literal["Active", "Passivation", "Transitional", "Inhibited"]
ThicknessLossRiskTier = Literal["Low", "Medium", "High", "Critical"]

DEFAULT_CRITICAL_THICKNESS_NM: Final[float] = 1.0
DEFAULT_PASSIVATION_LIMIT_NM: Final[float] = 5.0
DEFAULT_POWER_LAW_N: Final[float] = 0.6
DEFAULT_POWER_LAW_T_REF_S: Final[float] = 60.0
POWER_LAW_N_MIN: Final[float] = 0.3
POWER_LAW_N_MAX: Final[float] = 1.0

RISK_TIER_MEDIUM_MULTIPLIER: Final[float] = 0.8
RISK_TIER_HIGH_MULTIPLIER: Final[float] = 2.0
RISK_TIER_CRITICAL_MULTIPLIER: Final[float] = 4.0

_SEED_MATERIAL_MAP: Final[dict[str, dict[str, Any]]] = {
    row["metal"]: row for row in SEED_MATERIAL_PROPERTIES
}


def _validate_positive_finite(name: str, value: float) -> float:
    val = float(value)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be strictly positive and finite, got {value!r}")
    return val


def _validate_non_negative_finite(name: str, value: float) -> float:
    val = float(value)
    if not math.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value!r}")
    return val


def _validate_power_law_n(power_law_n: float) -> float:
    n_val = float(power_law_n)
    if not math.isfinite(n_val) or not (POWER_LAW_N_MIN <= n_val <= POWER_LAW_N_MAX):
        raise ValueError(
            f"power_law_n must be within [{POWER_LAW_N_MIN}, {POWER_LAW_N_MAX}], got {power_law_n!r}"
        )
    return n_val


def _extract_cr0_nm_min(
    cr0_source: float | FaradayCorrosionResult | GalvanicCouplingResult,
) -> float:
    if isinstance(cr0_source, GalvanicCouplingResult):
        return _validate_non_negative_finite(
            "cr0_nm_min", cr0_source.coupled_faraday.corrosion_rate_nm_min
        )
    if isinstance(cr0_source, FaradayCorrosionResult):
        return _validate_non_negative_finite("cr0_nm_min", cr0_source.corrosion_rate_nm_min)
    return _validate_non_negative_finite("cr0_nm_min", cr0_source)


@dataclass(frozen=True)
class PourbaixRegimeSelection:
    """Automatic or explicit Pourbaix stability regime to default kinetic model mapping."""

    regime: PourbaixRegimeName
    model: KineticModelName
    rationale: str


def _lookup_pourbaix_properties(
    metal: str,
    material_properties: pd.DataFrame | None = None,
) -> tuple[float, float, float]:
    """Return (pourbaix_stable_pH_min, pourbaix_stable_pH_max, passivation_tendency) for a Metal."""
    key = str(metal).strip()
    ph_min: float | None = None
    ph_max: float | None = None
    tendency: float | None = None

    if material_properties is not None and not material_properties.empty:
        col = "metal" if "metal" in material_properties.columns else "symbol"
        if col in material_properties.columns:
            match = material_properties[material_properties[col].astype(str).str.strip() == key]
            if not match.empty:
                row = match.iloc[0]
                if "pourbaix_stable_pH_min" in row and not pd.isna(row["pourbaix_stable_pH_min"]):
                    ph_min = float(row["pourbaix_stable_pH_min"])
                if "pourbaix_stable_pH_max" in row and not pd.isna(row["pourbaix_stable_pH_max"]):
                    ph_max = float(row["pourbaix_stable_pH_max"])
                if "passivation_tendency" in row and not pd.isna(row["passivation_tendency"]):
                    tendency = float(row["passivation_tendency"])

    seed_row = _SEED_MATERIAL_MAP.get(key)
    if seed_row is not None:
        if ph_min is None:
            ph_min = float(seed_row["pourbaix_stable_pH_min"])
        if ph_max is None:
            ph_max = float(seed_row["pourbaix_stable_pH_max"])
        if tendency is None:
            tendency = float(seed_row["passivation_tendency"])

    if ph_min is None or ph_max is None or tendency is None:
        raise ValueError(f"Unsupported Metal {metal!r}: missing Pourbaix stability descriptors.")
    return ph_min, ph_max, tendency


def select_kinetic_model_from_pourbaix(
    *,
    regime: str | None = None,
    metal: str | None = None,
    pH: float | None = None,
    inhibitor_present: bool = False,
    material_properties: pd.DataFrame | None = None,
) -> PourbaixRegimeSelection:
    """Map Pourbaix stability indicators (or explicit regime) to the default kinetic model."""
    if regime is not None:
        cleaned = str(regime).strip().lower()
        if cleaned == "active":
            return PourbaixRegimeSelection(
                regime="Active",
                model="linear",
                rationale="Active dissolution regime selected -> Linear kinetics.",
            )
        if cleaned in ("passivation", "passive"):
            return PourbaixRegimeSelection(
                regime="Passivation",
                model="passivation",
                rationale="Passivation regime selected -> Passivation (self-limiting) kinetics.",
            )
        if cleaned == "inhibited":
            return PourbaixRegimeSelection(
                regime="Inhibited",
                model="power_law",
                rationale="Inhibited regime selected -> Power-law kinetics.",
            )
        if cleaned == "transitional":
            return PourbaixRegimeSelection(
                regime="Transitional",
                model="power_law",
                rationale="Transitional regime selected -> Power-law kinetics.",
            )
        raise ValueError(
            f"Unknown Pourbaix regime {regime!r}. Expected 'Active', 'Passivation', 'Transitional', or 'Inhibited'."
        )

    if metal is None or pH is None:
        raise ValueError("Provide either an explicit regime or both metal and pH.")
    ph_val = float(pH)
    if not math.isfinite(ph_val):
        raise ValueError(f"pH must be finite, got {pH!r}")

    ph_min, ph_max, tendency = _lookup_pourbaix_properties(metal, material_properties)
    in_stable_window = ph_min <= ph_val <= ph_max
    dist_to_window = 0.0 if in_stable_window else min(abs(ph_val - ph_min), abs(ph_val - ph_max))

    if in_stable_window and tendency >= 0.6:
        return PourbaixRegimeSelection(
            regime="Passivation",
            model="passivation",
            rationale=(
                f"{metal} at pH {ph_val:.2f} lies inside its Pourbaix passive window "
                f"[{ph_min:.1f}, {ph_max:.1f}] with high passivation tendency ({tendency:.2f})."
            ),
        )
    if inhibitor_present:
        return PourbaixRegimeSelection(
            regime="Inhibited",
            model="power_law",
            rationale=(
                f"Corrosion inhibitor present for {metal} at pH {ph_val:.2f} -> "
                f"diffusion/adsorption-retarded Power-law kinetics."
            ),
        )
    if (in_stable_window and tendency >= 0.4) or (dist_to_window <= 1.0 and tendency >= 0.45):
        return PourbaixRegimeSelection(
            regime="Transitional",
            model="power_law",
            rationale=(
                f"{metal} at pH {ph_val:.2f} is near/in a transitional Pourbaix region "
                f"(window [{ph_min:.1f}, {ph_max:.1f}], passivation tendency {tendency:.2f})."
            ),
        )
    return PourbaixRegimeSelection(
        regime="Active",
        model="linear",
        rationale=(
            f"{metal} at pH {ph_val:.2f} lies in an active dissolution regime "
            f"(outside passive window [{ph_min:.1f}, {ph_max:.1f}] or low passivation tendency {tendency:.2f})."
        ),
    )


@overload
def compute_corrosion_depth(
    cr0_nm_min: float | FaradayCorrosionResult | GalvanicCouplingResult,
    immersion_time_s: float,
    *,
    model: KineticModelName = "linear",
    d_max_nm: float = DEFAULT_PASSIVATION_LIMIT_NM,
    power_law_n: float = DEFAULT_POWER_LAW_N,
    t_ref_s: float = DEFAULT_POWER_LAW_T_REF_S,
) -> float: ...


@overload
def compute_corrosion_depth(
    cr0_nm_min: float | FaradayCorrosionResult | GalvanicCouplingResult,
    immersion_time_s: np.ndarray | Sequence[float],
    *,
    model: KineticModelName = "linear",
    d_max_nm: float = DEFAULT_PASSIVATION_LIMIT_NM,
    power_law_n: float = DEFAULT_POWER_LAW_N,
    t_ref_s: float = DEFAULT_POWER_LAW_T_REF_S,
) -> np.ndarray: ...


def _validate_times_1d(immersion_times_s: np.ndarray | Sequence[float]) -> np.ndarray:
    times = np.asarray(immersion_times_s, dtype=float)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("immersion_times_s must be a non-empty 1D sequence of times.")
    if np.any(~np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("immersion_times_s must contain only finite, non-negative values.")
    return times


def _resolve_kinetic_model(
    model: KineticModelName | None,
    *,
    pourbaix_regime: str | None,
    metal: str | None,
    pH: float | None,
    inhibitor_present: bool,
    material_properties: pd.DataFrame | None,
) -> tuple[KineticModelName, PourbaixRegimeSelection | None]:
    if model is not None:
        return model, None
    selection = select_kinetic_model_from_pourbaix(
        regime=pourbaix_regime,
        metal=metal,
        pH=pH,
        inhibitor_present=inhibitor_present,
        material_properties=material_properties,
    )
    return selection.model, selection


def _evaluate_depth_kernel(
    v0_nm_s: np.ndarray,
    t_arr: np.ndarray,
    *,
    model: KineticModelName,
    d_max_nm: float,
    power_law_n: float,
    t_ref_s: float,
) -> np.ndarray:
    """Shared vectorized depth formula for both scalar and Monte Carlo array evaluations."""
    if model == "linear":
        depth = v0_nm_s * t_arr
    elif model == "passivation":
        d_max = _validate_positive_finite("d_max_nm", d_max_nm)
        depth = d_max * (-np.expm1(-(v0_nm_s / d_max) * t_arr))
    elif model == "power_law":
        n_val = _validate_power_law_n(power_law_n)
        t_ref = _validate_positive_finite("t_ref_s", t_ref_s)
        depth = (v0_nm_s * t_ref) * np.power(t_arr / t_ref, n_val)
    else:
        raise ValueError(
            f"Unknown kinetic model {model!r}. Expected 'linear', 'passivation', or 'power_law'."
        )
    return np.maximum(depth, 0.0)


def _evaluate_time_to_threshold_kernel(
    v0_nm_s: np.ndarray,
    *,
    d_crit_nm: float,
    model: KineticModelName,
    d_max_nm: float,
    power_law_n: float,
    t_ref_s: float,
) -> np.ndarray:
    """Shared vectorized Time to Threshold formula for scalar and Monte Carlo evaluations."""
    d_crit = _validate_positive_finite("d_crit_nm", d_crit_nm)
    if model == "linear":
        with np.errstate(divide="ignore"):
            return np.where(v0_nm_s > 0.0, d_crit / v0_nm_s, np.inf)
    if model == "passivation":
        d_max = _validate_positive_finite("d_max_nm", d_max_nm)
        if d_max <= d_crit:
            return np.full_like(v0_nm_s, np.inf, dtype=float)
        factor = -d_max * math.log1p(-d_crit / d_max)
        with np.errstate(divide="ignore"):
            return np.where(v0_nm_s > 0.0, factor / v0_nm_s, np.inf)
    if model == "power_law":
        n_val = _validate_power_law_n(power_law_n)
        t_ref = _validate_positive_finite("t_ref_s", t_ref_s)
        with np.errstate(divide="ignore"):
            return np.where(
                v0_nm_s > 0.0,
                t_ref * np.power(d_crit / (v0_nm_s * t_ref), 1.0 / n_val),
                np.inf,
            )
    raise ValueError(
        f"Unknown kinetic model {model!r}. Expected 'linear', 'passivation', or 'power_law'."
    )


def compute_corrosion_depth(
    cr0_nm_min: float | FaradayCorrosionResult | GalvanicCouplingResult,
    immersion_time_s: float | np.ndarray | Sequence[float],
    *,
    model: KineticModelName = "linear",
    d_max_nm: float = DEFAULT_PASSIVATION_LIMIT_NM,
    power_law_n: float = DEFAULT_POWER_LAW_N,
    t_ref_s: float = DEFAULT_POWER_LAW_T_REF_S,
) -> float | np.ndarray:
    """Compute cumulative Corrosion Depth d(t) in nm anchored to Initial Corrosion Rate CR_0 (per ADR-0002)."""
    cr0 = _extract_cr0_nm_min(cr0_nm_min)
    v0_nm_s = np.asarray(cr0 / SECONDS_PER_MINUTE, dtype=float)

    is_scalar = np.ndim(immersion_time_s) == 0
    t_arr = np.asarray(immersion_time_s, dtype=float)
    if np.any(~np.isfinite(t_arr)) or np.any(t_arr < 0.0):
        raise ValueError("immersion_time_s must contain only finite, non-negative values.")

    depth = _evaluate_depth_kernel(
        v0_nm_s,
        t_arr,
        model=model,
        d_max_nm=d_max_nm,
        power_law_n=power_law_n,
        t_ref_s=t_ref_s,
    )
    if is_scalar:
        return float(depth)
    return depth


def compute_time_to_threshold(
    cr0_nm_min: float | FaradayCorrosionResult | GalvanicCouplingResult,
    *,
    d_crit_nm: float = DEFAULT_CRITICAL_THICKNESS_NM,
    model: KineticModelName = "linear",
    d_max_nm: float = DEFAULT_PASSIVATION_LIMIT_NM,
    power_law_n: float = DEFAULT_POWER_LAW_N,
    t_ref_s: float = DEFAULT_POWER_LAW_T_REF_S,
) -> float:
    """Compute Immersion Time (in seconds) when d(t) reaches d_crit_nm, or +inf if unreached."""
    cr0 = _extract_cr0_nm_min(cr0_nm_min)
    v0_arr = np.asarray(cr0 / SECONDS_PER_MINUTE, dtype=float)
    res = _evaluate_time_to_threshold_kernel(
        v0_arr,
        d_crit_nm=d_crit_nm,
        model=model,
        d_max_nm=d_max_nm,
        power_law_n=power_law_n,
        t_ref_s=t_ref_s,
    )
    return float(res)


@overload
def compute_thickness_loss_risk_score(
    depth_nm: float,
    d_crit_nm: float = DEFAULT_CRITICAL_THICKNESS_NM,
) -> float: ...


@overload
def compute_thickness_loss_risk_score(
    depth_nm: np.ndarray | Sequence[float],
    d_crit_nm: float = DEFAULT_CRITICAL_THICKNESS_NM,
) -> np.ndarray: ...


def compute_thickness_loss_risk_score(
    depth_nm: float | np.ndarray | Sequence[float],
    d_crit_nm: float = DEFAULT_CRITICAL_THICKNESS_NM,
) -> float | np.ndarray:
    """Compute continuous Thickness Loss Risk R(t) = 1 - exp(-ln(2) * d(t) / d_crit) in [0, 1]."""
    d_crit = _validate_positive_finite("d_crit_nm", d_crit_nm)
    is_scalar = np.ndim(depth_nm) == 0
    d_arr = np.asarray(depth_nm, dtype=float)
    if np.any(~np.isfinite(d_arr)) or np.any(d_arr < 0.0):
        raise ValueError("depth_nm must contain only finite, non-negative values.")

    risk = -np.expm1(-math.log(2.0) * (d_arr / d_crit))
    risk = np.clip(risk, 0.0, 1.0)
    if is_scalar:
        return float(risk)
    return risk


def classify_thickness_loss_risk_tier(
    depth_nm: float,
    d_crit_nm: float = DEFAULT_CRITICAL_THICKNESS_NM,
) -> ThicknessLossRiskTier:
    """Classify discrete Thickness Loss Risk tier from d / d_crit ratio."""
    d_val = _validate_non_negative_finite("depth_nm", depth_nm)
    d_crit = _validate_positive_finite("d_crit_nm", d_crit_nm)
    ratio = d_val / d_crit
    if ratio < RISK_TIER_MEDIUM_MULTIPLIER:
        return "Low"
    if ratio < RISK_TIER_HIGH_MULTIPLIER:
        return "Medium"
    if ratio < RISK_TIER_CRITICAL_MULTIPLIER:
        return "High"
    return "Critical"


@dataclass(frozen=True)
class TimeResolvedKineticsResult:
    """Deterministic time-resolved Corrosion Depth, Time to Threshold, and Thickness Loss Risk trajectory."""

    model: KineticModelName
    cr0_nm_min: float
    d_crit_nm: float
    times_s: np.ndarray
    depth_nm: np.ndarray
    risk_score: np.ndarray
    risk_tiers: tuple[ThicknessLossRiskTier, ...]
    time_to_threshold_s: float
    d_max_nm: float = DEFAULT_PASSIVATION_LIMIT_NM
    power_law_n: float = DEFAULT_POWER_LAW_N
    t_ref_s: float = DEFAULT_POWER_LAW_T_REF_S
    metal: str | None = None
    pourbaix_selection: PourbaixRegimeSelection | None = None

    @property
    def threshold_reachable(self) -> bool:
        return math.isfinite(self.time_to_threshold_s)

    @property
    def final_depth_nm(self) -> float:
        return float(self.depth_nm[-1])

    @property
    def final_risk_score(self) -> float:
        return float(self.risk_score[-1])

    @property
    def final_risk_tier(self) -> ThicknessLossRiskTier:
        return self.risk_tiers[-1]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "immersion_time_s": self.times_s,
                "depth_nm": self.depth_nm,
                "risk_score": self.risk_score,
                "risk_tier": list(self.risk_tiers),
            }
        )


def simulate_time_resolved_kinetics(
    cr0_source: float | FaradayCorrosionResult | GalvanicCouplingResult,
    immersion_times_s: np.ndarray | Sequence[float],
    *,
    model: KineticModelName | None = None,
    pourbaix_regime: str | None = None,
    metal: str | None = None,
    pH: float | None = None,
    inhibitor_present: bool = False,
    material_properties: pd.DataFrame | None = None,
    d_crit_nm: float = DEFAULT_CRITICAL_THICKNESS_NM,
    d_max_nm: float = DEFAULT_PASSIVATION_LIMIT_NM,
    power_law_n: float = DEFAULT_POWER_LAW_N,
    t_ref_s: float = DEFAULT_POWER_LAW_T_REF_S,
) -> TimeResolvedKineticsResult:
    """Simulate time-resolved Corrosion Depth d(t), Time to Threshold, and Thickness Loss Risk R(t)."""
    cr0 = _extract_cr0_nm_min(cr0_source)
    resolved_metal = metal
    if resolved_metal is None:
        if isinstance(cr0_source, GalvanicCouplingResult):
            resolved_metal = cr0_source.active_metal
        elif isinstance(cr0_source, FaradayCorrosionResult):
            resolved_metal = cr0_source.metal

    chosen_model, selection = _resolve_kinetic_model(
        model,
        pourbaix_regime=pourbaix_regime,
        metal=resolved_metal,
        pH=pH,
        inhibitor_present=inhibitor_present,
        material_properties=material_properties,
    )
    times = _validate_times_1d(immersion_times_s)

    depths = compute_corrosion_depth(
        cr0,
        times,
        model=chosen_model,
        d_max_nm=d_max_nm,
        power_law_n=power_law_n,
        t_ref_s=t_ref_s,
    )
    risks = compute_thickness_loss_risk_score(depths, d_crit_nm=d_crit_nm)
    tiers = tuple(classify_thickness_loss_risk_tier(float(d), d_crit_nm=d_crit_nm) for d in depths)
    t_crit = compute_time_to_threshold(
        cr0,
        d_crit_nm=d_crit_nm,
        model=chosen_model,
        d_max_nm=d_max_nm,
        power_law_n=power_law_n,
        t_ref_s=t_ref_s,
    )

    return TimeResolvedKineticsResult(
        model=chosen_model,
        cr0_nm_min=cr0,
        d_crit_nm=float(d_crit_nm),
        times_s=times,
        depth_nm=depths,
        risk_score=risks,
        risk_tiers=tiers,
        time_to_threshold_s=t_crit,
        d_max_nm=float(d_max_nm),
        power_law_n=float(power_law_n),
        t_ref_s=float(t_ref_s),
        metal=resolved_metal,
        pourbaix_selection=selection,
    )


def compare_kinetic_models(
    cr0_source: float | FaradayCorrosionResult | GalvanicCouplingResult,
    immersion_times_s: np.ndarray | Sequence[float],
    *,
    models: Sequence[KineticModelName] = ("linear", "passivation", "power_law"),
    metal: str | None = None,
    d_crit_nm: float = DEFAULT_CRITICAL_THICKNESS_NM,
    d_max_nm: float = DEFAULT_PASSIVATION_LIMIT_NM,
    power_law_n: float = DEFAULT_POWER_LAW_N,
    t_ref_s: float = DEFAULT_POWER_LAW_T_REF_S,
) -> dict[KineticModelName, TimeResolvedKineticsResult]:
    """Run a side-by-side multi-model comparison across Linear, Passivation, and Power-law kinetics."""
    return {
        model_name: simulate_time_resolved_kinetics(
            cr0_source,
            immersion_times_s,
            model=model_name,
            metal=metal,
            d_crit_nm=d_crit_nm,
            d_max_nm=d_max_nm,
            power_law_n=power_law_n,
            t_ref_s=t_ref_s,
        )
        for model_name in models
    }


@dataclass(frozen=True)
class MonteCarloKineticsResult:
    """Monte Carlo uncertainty bands for time-resolved Corrosion Depth, Risk, and Time to Threshold."""

    model: KineticModelName
    n_draws: int
    d_crit_nm: float
    times_s: np.ndarray
    depth_mean_nm: np.ndarray
    depth_std_nm: np.ndarray
    depth_p05_nm: np.ndarray
    depth_p50_nm: np.ndarray
    depth_p95_nm: np.ndarray
    risk_mean: np.ndarray
    risk_std: np.ndarray
    risk_p05: np.ndarray
    risk_p50: np.ndarray
    risk_p95: np.ndarray
    median_risk_tiers: tuple[ThicknessLossRiskTier, ...]
    time_to_threshold_samples_s: np.ndarray
    time_to_threshold_summary: dict[str, float]
    threshold_reached_fraction: float
    metal: str | None = None
    d_max_nm: float = DEFAULT_PASSIVATION_LIMIT_NM
    power_law_n: float = DEFAULT_POWER_LAW_N
    t_ref_s: float = DEFAULT_POWER_LAW_T_REF_S


def _extract_cr0_samples(
    source: np.ndarray | Sequence[float] | MonteCarloPropagationResult | MonteCarloGalvanicResult,
    metal: str | None = None,
) -> tuple[np.ndarray, str | None]:
    if isinstance(source, MonteCarloGalvanicResult):
        return np.asarray(source.coupled_cr_nm_min_samples, dtype=float), metal or source.active_metal
    if isinstance(source, MonteCarloPropagationResult):
        return np.asarray(source.cr_nm_min_samples, dtype=float), metal or source.metal
    arr = np.asarray(source, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("cr0_nm_min_samples must be a non-empty 1D array.")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("cr0_nm_min_samples must contain only finite, non-negative values.")
    return arr, metal


def _summarize_threshold_times(samples_s: np.ndarray) -> tuple[dict[str, float], float]:
    finite_mask = np.isfinite(samples_s)
    reached_frac = float(np.mean(finite_mask))
    if reached_frac == 0.0:
        return (
            {
                "mean": math.inf,
                "std": math.nan,
                "p05": math.inf,
                "p50": math.inf,
                "p95": math.inf,
            },
            0.0,
        )
    if reached_frac == 1.0:
        return (
            {
                "mean": float(np.mean(samples_s)),
                "std": float(np.std(samples_s)),
                "p05": float(np.percentile(samples_s, 5.0)),
                "p50": float(np.percentile(samples_s, 50.0)),
                "p95": float(np.percentile(samples_s, 95.0)),
            },
            1.0,
        )
    finite_vals = samples_s[finite_mask]
    return (
        {
            "mean": float(np.mean(finite_vals)),
            "std": float(np.std(finite_vals)),
            "p05": float(np.percentile(samples_s, 5.0)),
            "p50": float(np.percentile(samples_s, 50.0)),
            "p95": float(np.percentile(samples_s, 95.0)),
        },
        reached_frac,
    )


def propagate_kinetics_uncertainty_monte_carlo(
    cr0_samples_source: np.ndarray
    | Sequence[float]
    | MonteCarloPropagationResult
    | MonteCarloGalvanicResult,
    immersion_times_s: np.ndarray | Sequence[float],
    *,
    model: KineticModelName | None = None,
    pourbaix_regime: str | None = None,
    metal: str | None = None,
    pH: float | None = None,
    inhibitor_present: bool = False,
    material_properties: pd.DataFrame | None = None,
    d_crit_nm: float = DEFAULT_CRITICAL_THICKNESS_NM,
    d_max_nm: float = DEFAULT_PASSIVATION_LIMIT_NM,
    power_law_n: float = DEFAULT_POWER_LAW_N,
    t_ref_s: float = DEFAULT_POWER_LAW_T_REF_S,
) -> MonteCarloKineticsResult:
    """Propagate CR_0 Monte Carlo samples through time-resolved kinetics, Risk R(t), and Time to Threshold."""
    cr0_arr, resolved_metal = _extract_cr0_samples(cr0_samples_source, metal=metal)
    d_crit = _validate_positive_finite("d_crit_nm", d_crit_nm)

    chosen_model, _ = _resolve_kinetic_model(
        model,
        pourbaix_regime=pourbaix_regime,
        metal=resolved_metal,
        pH=pH,
        inhibitor_present=inhibitor_present,
        material_properties=material_properties,
    )
    times = _validate_times_1d(immersion_times_s)

    v0_1d = cr0_arr / SECONDS_PER_MINUTE
    depth_matrix = _evaluate_depth_kernel(
        v0_1d[:, None],
        times[None, :],
        model=chosen_model,
        d_max_nm=d_max_nm,
        power_law_n=power_law_n,
        t_ref_s=t_ref_s,
    )
    t_crit_samples = _evaluate_time_to_threshold_kernel(
        v0_1d,
        d_crit_nm=d_crit,
        model=chosen_model,
        d_max_nm=d_max_nm,
        power_law_n=power_law_n,
        t_ref_s=t_ref_s,
    )
    risk_matrix = compute_thickness_loss_risk_score(depth_matrix, d_crit_nm=d_crit)

    depth_p50 = np.percentile(depth_matrix, 50.0, axis=0)
    median_tiers = tuple(classify_thickness_loss_risk_tier(float(d), d_crit) for d in depth_p50)
    t_summary, reached_frac = _summarize_threshold_times(t_crit_samples)

    return MonteCarloKineticsResult(
        model=chosen_model,
        n_draws=len(cr0_arr),
        d_crit_nm=d_crit,
        times_s=times,
        depth_mean_nm=np.mean(depth_matrix, axis=0),
        depth_std_nm=np.std(depth_matrix, axis=0),
        depth_p05_nm=np.percentile(depth_matrix, 5.0, axis=0),
        depth_p50_nm=depth_p50,
        depth_p95_nm=np.percentile(depth_matrix, 95.0, axis=0),
        risk_mean=np.mean(risk_matrix, axis=0),
        risk_std=np.std(risk_matrix, axis=0),
        risk_p05=np.percentile(risk_matrix, 5.0, axis=0),
        risk_p50=np.percentile(risk_matrix, 50.0, axis=0),
        risk_p95=np.percentile(risk_matrix, 95.0, axis=0),
        median_risk_tiers=median_tiers,
        time_to_threshold_samples_s=t_crit_samples,
        time_to_threshold_summary=t_summary,
        threshold_reached_fraction=reached_frac,
        metal=resolved_metal,
        d_max_nm=float(d_max_nm),
        power_law_n=float(power_law_n),
        t_ref_s=float(t_ref_s),
    )


def plot_kinetics_and_risk(
    result: TimeResolvedKineticsResult
    | MonteCarloKineticsResult
    | Sequence[TimeResolvedKineticsResult],
    *,
    axes: tuple[Axes, Axes] | None = None,
    title: str | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot time-resolved Corrosion Depth d(t) and Thickness Loss Risk R(t) with optional uncertainty bands."""
    import matplotlib.pyplot as plt

    if axes is None:
        fig, (ax_depth, ax_risk) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    else:
        ax_depth, ax_risk = axes
        fig = ax_depth.figure

    model_colors = {
        "linear": "#d62728",
        "passivation": "#2ca02c",
        "power_law": "#1f77b4",
    }

    if isinstance(result, MonteCarloKineticsResult):
        color = model_colors.get(result.model, "#1f77b4")
        ax_depth.fill_between(
            result.times_s,
            result.depth_p05_nm,
            result.depth_p95_nm,
            color=color,
            alpha=0.22,
            label=f"90% Uncertainty Band (P05–P95, N={result.n_draws})",
        )
        ax_depth.plot(
            result.times_s,
            result.depth_p50_nm,
            color=color,
            linewidth=2.2,
            label=f"{result.model.replace('_', ' ').title()} Median (P50)",
        )
        ax_depth.axhline(
            result.d_crit_nm,
            color="#111111",
            linestyle="--",
            linewidth=1.4,
            label=f"Critical Threshold d_crit = {result.d_crit_nm:.2g} nm",
        )
        t50 = result.time_to_threshold_summary["p50"]
        if math.isfinite(t50) and t50 <= float(result.times_s[-1]):
            ax_depth.axvline(
                t50,
                color="#ff7f0e",
                linestyle=":",
                linewidth=1.5,
                label=f"Median Time to Threshold = {t50:.1f} s",
            )

        ax_risk.fill_between(
            result.times_s,
            result.risk_p05,
            result.risk_p95,
            color=color,
            alpha=0.22,
            label=f"90% Risk Band (P05–P95, N={result.n_draws})",
        )
        ax_risk.plot(
            result.times_s,
            result.risk_p50,
            color=color,
            linewidth=2.2,
            label="Median Risk R(t) (P50)",
        )
        default_title = (
            f"{result.metal + ' ' if result.metal else ''}"
            f"Time-Resolved Corrosion Depth & Thickness Loss Risk ({result.model.replace('_', ' ').title()})"
        )
    else:
        items = [result] if isinstance(result, TimeResolvedKineticsResult) else list(result)
        if not items:
            raise ValueError("At least one TimeResolvedKineticsResult is required for plotting.")
        d_crit = items[0].d_crit_nm
        for item in items:
            color = model_colors.get(item.model, "#1f77b4")
            label_base = item.model.replace("_", " ").title()
            t_str = f"t_crit={item.time_to_threshold_s:.1f} s" if item.threshold_reachable else "unreached"
            ax_depth.plot(
                item.times_s,
                item.depth_nm,
                color=color,
                linewidth=2.1,
                label=f"{label_base} ({t_str})",
            )
            ax_risk.plot(
                item.times_s,
                item.risk_score,
                color=color,
                linewidth=2.1,
                label=f"{label_base} (final: {item.final_risk_tier})",
            )
        ax_depth.axhline(
            d_crit,
            color="#111111",
            linestyle="--",
            linewidth=1.4,
            label=f"Critical Threshold d_crit = {d_crit:.2g} nm",
        )
        metal_prefix = f"{items[0].metal} " if items[0].metal else ""
        default_title = f"{metal_prefix}Time-Resolved Corrosion Depth & Thickness Loss Risk"

    # Risk tier reference line at R(d_crit) = 0.5
    ax_risk.axhline(0.5, color="#111111", linestyle="--", linewidth=1.2, label="Threshold Risk R(d_crit) = 0.5")
    ax_risk.set_ylim(-0.02, 1.04)

    ax_depth.set_xlabel("Immersion Time [s]")
    ax_depth.set_ylabel("Corrosion Depth d(t) [nm]")
    ax_depth.set_title("Cumulative Corrosion Depth vs. Time")
    ax_depth.grid(True, linestyle=":", alpha=0.45)
    ax_depth.legend(loc="best", fontsize=8.2)

    ax_risk.set_xlabel("Immersion Time [s]")
    ax_risk.set_ylabel("Thickness Loss Risk R(t) [0–1]")
    ax_risk.set_title("Continuous Thickness Loss Risk vs. Time")
    ax_risk.grid(True, linestyle=":", alpha=0.45)
    ax_risk.legend(loc="best", fontsize=8.2)

    fig.suptitle(title or default_title, fontsize=11.5)
    fig.tight_layout()
    return fig, (ax_depth, ax_risk)
