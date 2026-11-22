"""Phase 1 physics solvers: four-parameter Wagner-Traud polarization, Faraday's law, and galvanic screening."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Final, Literal, overload

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from corrosion_simulator.database import (
    DEFAULT_DISSOLUTION_VALENCE,
    ExposureCondition,
    ExposureLookupResult,
    HarmonizedDatabase,
)
from corrosion_simulator.seed_data import SEED_MATERIAL_PROPERTIES

FARADAY_CONSTANT_C_MOL: Final[float] = 96485.33212
SECONDS_PER_MINUTE: Final[float] = 60.0
SECONDS_PER_YEAR: Final[float] = 365.0 * 24.0 * 3600.0  # ASTM G102 convention (31,536,000 s/yr)

_DEFAULT_METAL_CONSTANTS: Final[dict[str, tuple[float, float, int]]] = {
    row["metal"]: (
        float(row["molar_mass_g_mol"]),
        float(row["density_g_cm3"]),
        int(DEFAULT_DISSOLUTION_VALENCE[row["metal"]]),
    )
    for row in SEED_MATERIAL_PROPERTIES
}


@dataclass(frozen=True)
class PolarizationParameters:
    """Four-parameter activation-controlled Wagner-Traud kinetic parameters (per ADR-0001)."""

    Ecorr_V_SHE: float
    icorr_uA_cm2: float
    beta_a_mV_dec: float
    beta_c_mV_dec: float
    metal: str | None = None
    solution_id: str | None = None
    temperature_C: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.Ecorr_V_SHE):
            raise ValueError(f"Ecorr_V_SHE must be finite, got {self.Ecorr_V_SHE!r}")
        if not math.isfinite(self.icorr_uA_cm2) or self.icorr_uA_cm2 <= 0.0:
            raise ValueError(f"icorr_uA_cm2 must be strictly positive, got {self.icorr_uA_cm2!r}")
        if not math.isfinite(self.beta_a_mV_dec) or self.beta_a_mV_dec <= 0.0:
            raise ValueError(
                f"beta_a_mV_dec must be strictly positive, got {self.beta_a_mV_dec!r}"
            )
        if not math.isfinite(self.beta_c_mV_dec) or self.beta_c_mV_dec <= 0.0:
            raise ValueError(
                f"beta_c_mV_dec must be strictly positive, got {self.beta_c_mV_dec!r}"
            )

    @property
    def beta_a_V_dec(self) -> float:
        return self.beta_a_mV_dec / 1000.0

    @property
    def beta_c_V_dec(self) -> float:
        return self.beta_c_mV_dec / 1000.0

    @classmethod
    def from_lookup_result(cls, result: ExposureLookupResult) -> PolarizationParameters:
        """Build PolarizationParameters from a HarmonizedDatabase lookup result."""
        return cls(
            Ecorr_V_SHE=float(result.Ecorr_V_SHE),
            icorr_uA_cm2=float(result.icorr_uA_cm2),
            beta_a_mV_dec=float(result.beta_a_mV_dec),
            beta_c_mV_dec=float(result.beta_c_mV_dec),
            metal=result.metal,
            solution_id=result.solution_id,
            temperature_C=float(result.matched_condition.temperature_C),
        )


def _coerce_parameters(
    params: PolarizationParameters | ExposureLookupResult,
) -> PolarizationParameters:
    if isinstance(params, ExposureLookupResult):
        return PolarizationParameters.from_lookup_result(params)
    if isinstance(params, PolarizationParameters):
        return params
    raise TypeError(
        f"Expected PolarizationParameters or ExposureLookupResult, got {type(params).__name__}"
    )


@dataclass(frozen=True)
class PolarizationCurve:
    """Generated activation-controlled Polarization Curve (E vs. current density branches)."""

    parameters: PolarizationParameters
    potential_V_SHE: np.ndarray
    anodic_current_uA_cm2: np.ndarray
    cathodic_current_uA_cm2: np.ndarray
    net_current_uA_cm2: np.ndarray
    abs_net_current_uA_cm2: np.ndarray
    log10_anodic_uA_cm2: np.ndarray
    log10_cathodic_uA_cm2: np.ndarray
    log10_abs_net_uA_cm2: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "potential_V_SHE": self.potential_V_SHE,
                "anodic_current_uA_cm2": self.anodic_current_uA_cm2,
                "cathodic_current_uA_cm2": self.cathodic_current_uA_cm2,
                "net_current_uA_cm2": self.net_current_uA_cm2,
                "abs_net_current_uA_cm2": self.abs_net_current_uA_cm2,
                "log10_anodic_uA_cm2": self.log10_anodic_uA_cm2,
                "log10_cathodic_uA_cm2": self.log10_cathodic_uA_cm2,
                "log10_abs_net_uA_cm2": self.log10_abs_net_uA_cm2,
            }
        )


@overload
def wagner_traud_anodic_current(
    potential_V_SHE: float,
    params: PolarizationParameters | ExposureLookupResult,
) -> float: ...


@overload
def wagner_traud_anodic_current(
    potential_V_SHE: np.ndarray,
    params: PolarizationParameters | ExposureLookupResult,
) -> np.ndarray: ...


def wagner_traud_anodic_current(
    potential_V_SHE: float | np.ndarray,
    params: PolarizationParameters | ExposureLookupResult,
) -> float | np.ndarray:
    """Compute anodic dissolution current density i_anodic(E) in µA/cm²."""
    p = _coerce_parameters(params)
    eta = np.asarray(potential_V_SHE, dtype=float) - p.Ecorr_V_SHE
    values = p.icorr_uA_cm2 * np.power(10.0, eta / p.beta_a_V_dec)
    if np.ndim(potential_V_SHE) == 0:
        return float(values)
    return values


@overload
def wagner_traud_cathodic_current(
    potential_V_SHE: float,
    params: PolarizationParameters | ExposureLookupResult,
) -> float: ...


@overload
def wagner_traud_cathodic_current(
    potential_V_SHE: np.ndarray,
    params: PolarizationParameters | ExposureLookupResult,
) -> np.ndarray: ...


def wagner_traud_cathodic_current(
    potential_V_SHE: float | np.ndarray,
    params: PolarizationParameters | ExposureLookupResult,
) -> float | np.ndarray:
    """Compute cathodic reduction current density magnitude i_cathodic(E) in µA/cm²."""
    p = _coerce_parameters(params)
    eta = np.asarray(potential_V_SHE, dtype=float) - p.Ecorr_V_SHE
    values = p.icorr_uA_cm2 * np.power(10.0, -eta / p.beta_c_V_dec)
    if np.ndim(potential_V_SHE) == 0:
        return float(values)
    return values


@overload
def wagner_traud_net_current(
    potential_V_SHE: float,
    params: PolarizationParameters | ExposureLookupResult,
) -> float: ...


@overload
def wagner_traud_net_current(
    potential_V_SHE: np.ndarray,
    params: PolarizationParameters | ExposureLookupResult,
) -> np.ndarray: ...


def wagner_traud_net_current(
    potential_V_SHE: float | np.ndarray,
    params: PolarizationParameters | ExposureLookupResult,
) -> float | np.ndarray:
    """Compute net current density i_net(E) = i_anodic(E) - i_cathodic(E) in µA/cm²."""
    p = _coerce_parameters(params)
    i_a = wagner_traud_anodic_current(potential_V_SHE, p)
    i_c = wagner_traud_cathodic_current(potential_V_SHE, p)
    net = i_a - i_c
    if np.ndim(potential_V_SHE) == 0:
        return float(net)
    return np.asarray(net, dtype=float)


def generate_polarization_curve(
    params: PolarizationParameters | ExposureLookupResult,
    *,
    overpotential_min_V: float = -0.25,
    overpotential_max_V: float = 0.25,
    num_points: int = 401,
    potentials_V_SHE: np.ndarray | None = None,
) -> PolarizationCurve:
    """Generate anodic, cathodic, and net Polarization Curves over a potential sweep."""
    p = _coerce_parameters(params)
    if potentials_V_SHE is not None:
        potentials = np.asarray(potentials_V_SHE, dtype=float)
        if potentials.ndim != 1 or len(potentials) == 0:
            raise ValueError("potentials_V_SHE must be a non-empty 1D array.")
    else:
        if overpotential_min_V >= overpotential_max_V:
            raise ValueError("overpotential_min_V must be strictly less than overpotential_max_V.")
        if num_points < 3:
            raise ValueError("num_points must be at least 3.")
        potentials = np.linspace(
            p.Ecorr_V_SHE + overpotential_min_V,
            p.Ecorr_V_SHE + overpotential_max_V,
            num_points,
        )
        # Ensure exact E_corr sits on the grid when within sweep bounds
        if overpotential_min_V <= 0.0 <= overpotential_max_V:
            nearest_idx = int(np.argmin(np.abs(potentials - p.Ecorr_V_SHE)))
            potentials[nearest_idx] = p.Ecorr_V_SHE

    i_a = wagner_traud_anodic_current(potentials, p)
    i_c = wagner_traud_cathodic_current(potentials, p)
    i_net = i_a - i_c
    abs_net = np.abs(i_net)

    log_a = np.log10(i_a)
    log_c = np.log10(i_c)
    with np.errstate(divide="ignore"):
        log_abs_net = np.where(abs_net > 0.0, np.log10(abs_net), np.nan)

    return PolarizationCurve(
        parameters=p,
        potential_V_SHE=potentials,
        anodic_current_uA_cm2=i_a,
        cathodic_current_uA_cm2=i_c,
        net_current_uA_cm2=i_net,
        abs_net_current_uA_cm2=abs_net,
        log10_anodic_uA_cm2=log_a,
        log10_cathodic_uA_cm2=log_c,
        log10_abs_net_uA_cm2=log_abs_net,
    )


@dataclass(frozen=True)
class FaradayCorrosionResult:
    """Initial Corrosion Rate (CR_0) and optional linear Corrosion Depth derived via Faraday's law."""

    metal: str
    icorr_uA_cm2: float
    dissolution_valence: int
    molar_mass_g_mol: float
    density_g_cm3: float
    corrosion_rate_nm_min: float
    corrosion_rate_mm_yr: float
    immersion_time_s: float | None = None
    corrosion_depth_nm: float | None = None


def _resolve_metal_constants(
    metal: str,
    material_properties: pd.DataFrame | None = None,
) -> tuple[float, float, int]:
    """Return (molar_mass_g_mol, density_g_cm3, default_valence) for a Metal."""
    metal_key = str(metal).strip()
    molar_mass: float | None = None
    density: float | None = None
    default_valence: int | None = None

    if material_properties is not None and not material_properties.empty:
        col = "metal" if "metal" in material_properties.columns else "symbol"
        if col in material_properties.columns:
            match = material_properties[material_properties[col].astype(str).str.strip() == metal_key]
            if not match.empty:
                row = match.iloc[0]
                if "molar_mass_g_mol" in row and not pd.isna(row["molar_mass_g_mol"]):
                    molar_mass = float(row["molar_mass_g_mol"])
                if "density_g_cm3" in row and not pd.isna(row["density_g_cm3"]):
                    density = float(row["density_g_cm3"])
                if "valence" in row and not pd.isna(row["valence"]):
                    default_valence = int(row["valence"])

    fallback = _DEFAULT_METAL_CONSTANTS.get(metal_key)
    if fallback is not None:
        if molar_mass is None:
            molar_mass = fallback[0]
        if density is None:
            density = fallback[1]
        if default_valence is None:
            default_valence = fallback[2]

    if molar_mass is None or density is None or default_valence is None:
        raise ValueError(
            f"Unsupported Metal {metal!r}: missing molar mass, density, or Dissolution Valence."
        )
    return molar_mass, density, default_valence


def calculate_faraday_corrosion_rate(
    metal: str,
    icorr_uA_cm2: float,
    *,
    valence: int | None = None,
    immersion_time_s: float | None = None,
    material_properties: pd.DataFrame | None = None,
) -> FaradayCorrosionResult:
    """Convert Corrosion Current Density (µA/cm²) to Initial Corrosion Rate (nm/min, mm/yr) and linear Corrosion Depth."""
    if not math.isfinite(icorr_uA_cm2) or icorr_uA_cm2 < 0.0:
        raise ValueError(f"icorr_uA_cm2 must be non-negative and finite, got {icorr_uA_cm2!r}")
    if immersion_time_s is not None and (not math.isfinite(immersion_time_s) or immersion_time_s < 0.0):
        raise ValueError(
            f"immersion_time_s must be non-negative and finite, got {immersion_time_s!r}"
        )

    molar_mass, density, default_valence = _resolve_metal_constants(metal, material_properties)
    z = default_valence if valence is None else int(valence)
    if z <= 0:
        raise ValueError(f"Dissolution Valence (z) must be a positive integer, got {z!r}")

    # Volume dissolution velocity v (cm/s) = (i [A/cm2] * M [g/mol]) / (z * F [C/mol] * rho [g/cm3])
    current_A_cm2 = float(icorr_uA_cm2) * 1e-6
    velocity_cm_s = (current_A_cm2 * molar_mass) / (z * FARADAY_CONSTANT_C_MOL * density)

    # 1 cm = 1e7 nm; 1 min = 60 s -> CR [nm/min] = v [cm/s] * 1e7 * 60
    cr_nm_min = velocity_cm_s * 1e7 * SECONDS_PER_MINUTE
    # 1 cm = 10 mm; 1 yr = 365 * 86400 s -> CR [mm/yr] = v [cm/s] * 10 * 31,536,000
    cr_mm_yr = velocity_cm_s * 10.0 * SECONDS_PER_YEAR

    depth_nm: float | None = None
    if immersion_time_s is not None:
        depth_nm = cr_nm_min * (float(immersion_time_s) / SECONDS_PER_MINUTE)

    return FaradayCorrosionResult(
        metal=str(metal).strip(),
        icorr_uA_cm2=float(icorr_uA_cm2),
        dissolution_valence=z,
        molar_mass_g_mol=molar_mass,
        density_g_cm3=density,
        corrosion_rate_nm_min=cr_nm_min,
        corrosion_rate_mm_yr=cr_mm_yr,
        immersion_time_s=float(immersion_time_s) if immersion_time_s is not None else None,
        corrosion_depth_nm=depth_nm,
    )


GalvanicDrivingForceTier = Literal["Low", "Medium", "High"]

GALVANIC_LOW_THRESHOLD_V: Final[float] = 0.1
GALVANIC_HIGH_THRESHOLD_V: Final[float] = 0.3
_THRESHOLD_TOL_V: Final[float] = 1e-9


@dataclass(frozen=True)
class GalvanicDrivingForceResult:
    """Phase 1 bimetallic Galvanic Driving Force Risk screening result."""

    active_metal: str
    noble_metal: str
    active_Ecorr_V_SHE: float
    noble_Ecorr_V_SHE: float
    delta_Ecorr_V: float
    risk_tier: GalvanicDrivingForceTier
    solution_id: str | None = None
    temperature_C: float | None = None


def screen_galvanic_driving_force(
    metal_a: str,
    Ecorr_a_V_SHE: float,
    metal_b: str,
    Ecorr_b_V_SHE: float,
    *,
    solution_id: str | None = None,
    temperature_C: float | None = None,
) -> GalvanicDrivingForceResult:
    """Classify Galvanic Driving Force Risk from |Delta E_corr| and identify Active Metal vs. Noble Metal."""
    name_a = str(metal_a).strip()
    name_b = str(metal_b).strip()
    if not name_a or not name_b:
        raise ValueError("Both metal names must be non-empty.")
    if name_a == name_b:
        raise ValueError(f"Bimetallic screening requires two distinct metals, got {name_a!r} twice.")
    if not math.isfinite(Ecorr_a_V_SHE) or not math.isfinite(Ecorr_b_V_SHE):
        raise ValueError("Both Corrosion Potentials must be finite values in V vs. SHE.")

    ea = float(Ecorr_a_V_SHE)
    eb = float(Ecorr_b_V_SHE)
    if ea <= eb:
        active_metal, active_ecorr = name_a, ea
        noble_metal, noble_ecorr = name_b, eb
    else:
        active_metal, active_ecorr = name_b, eb
        noble_metal, noble_ecorr = name_a, ea

    delta_v = abs(noble_ecorr - active_ecorr)
    if delta_v < GALVANIC_LOW_THRESHOLD_V - _THRESHOLD_TOL_V:
        tier: GalvanicDrivingForceTier = "Low"
    elif delta_v <= GALVANIC_HIGH_THRESHOLD_V + _THRESHOLD_TOL_V:
        tier = "Medium"
    else:
        tier = "High"

    return GalvanicDrivingForceResult(
        active_metal=active_metal,
        noble_metal=noble_metal,
        active_Ecorr_V_SHE=active_ecorr,
        noble_Ecorr_V_SHE=noble_ecorr,
        delta_Ecorr_V=delta_v,
        risk_tier=tier,
        solution_id=solution_id,
        temperature_C=temperature_C,
    )


def screen_galvanic_pair_from_database(
    database: HarmonizedDatabase,
    metal_a: str,
    metal_b: str,
    *,
    pH: float = 7.0,
    oxidizer_type: str = "None",
    oxidizer_concentration: float = 0.0,
    temperature_C: float = 25.0,
    solution_id: str | None = None,
) -> GalvanicDrivingForceResult:
    """Look up two metals under the same Solution and Temperature and screen their Galvanic Driving Force Risk."""
    name_a = str(metal_a).strip()
    name_b = str(metal_b).strip()
    if name_a == name_b:
        raise ValueError(f"Bimetallic screening requires two distinct metals, got {name_a!r} twice.")

    # When solution_id is explicitly provided, resolve its formulation attributes from solution_properties
    eff_pH = pH
    eff_ox_type = oxidizer_type
    eff_ox_conc = oxidizer_concentration
    if solution_id is not None and not database.solution_properties.empty:
        sol_match = database.solution_properties[
            database.solution_properties["solution_id"].astype(str).str.strip() == solution_id.strip()
        ]
        if not sol_match.empty:
            sol_row = sol_match.iloc[0]
            if not pd.isna(sol_row.get("pH")):
                eff_pH = float(sol_row["pH"])
            if not pd.isna(sol_row.get("oxidizer_type")):
                eff_ox_type = str(sol_row["oxidizer_type"])
            if not pd.isna(sol_row.get("oxidizer_concentration")):
                eff_ox_conc = float(sol_row["oxidizer_concentration"])

    lookup_a = database.lookup_exposure_condition(
        ExposureCondition(
            metal=name_a,
            pH=eff_pH,
            oxidizer_type=eff_ox_type,
            oxidizer_concentration=eff_ox_conc,
            temperature_C=temperature_C,
            solution_id=solution_id,
        )
    )
    lookup_b = database.lookup_exposure_condition(
        ExposureCondition(
            metal=name_b,
            pH=eff_pH,
            oxidizer_type=eff_ox_type,
            oxidizer_concentration=eff_ox_conc,
            temperature_C=temperature_C,
            solution_id=solution_id,
        )
    )

    resolved_solution_id = (
        solution_id
        if solution_id is not None
        else (lookup_a.solution_id if lookup_a.solution_id == lookup_b.solution_id else None)
    )
    return screen_galvanic_driving_force(
        lookup_a.metal,
        lookup_a.Ecorr_V_SHE,
        lookup_b.metal,
        lookup_b.Ecorr_V_SHE,
        solution_id=resolved_solution_id,
        temperature_C=temperature_C,
    )


def plot_tafel_curve(
    curve_or_params: PolarizationCurve | PolarizationParameters | ExposureLookupResult,
    *,
    ax: Axes | None = None,
    show_branches: bool = True,
    title: str | None = None,
    overpotential_min_V: float = -0.25,
    overpotential_max_V: float = 0.25,
) -> tuple[Figure, Axes]:
    """Plot an E (V vs. SHE) vs. log10|i| (µA/cm²) Tafel Polarization Curve with annotations."""
    import matplotlib.pyplot as plt

    if isinstance(curve_or_params, PolarizationCurve):
        curve = curve_or_params
    else:
        curve = generate_polarization_curve(
            curve_or_params,
            overpotential_min_V=overpotential_min_V,
            overpotential_max_V=overpotential_max_V,
        )

    p = curve.parameters
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
    else:
        fig = ax.figure

    if show_branches:
        ax.plot(
            curve.log10_anodic_uA_cm2,
            curve.potential_V_SHE,
            linestyle="--",
            color="#d62728",
            linewidth=1.5,
            alpha=0.85,
            label=f"Anodic branch (β_a = {p.beta_a_mV_dec:.1f} mV/dec)",
        )
        ax.plot(
            curve.log10_cathodic_uA_cm2,
            curve.potential_V_SHE,
            linestyle="--",
            color="#1f77b4",
            linewidth=1.5,
            alpha=0.85,
            label=f"Cathodic branch (β_c = {p.beta_c_mV_dec:.1f} mV/dec)",
        )

    ax.plot(
        curve.log10_abs_net_uA_cm2,
        curve.potential_V_SHE,
        color="#111111",
        linewidth=2.2,
        label="Net Polarization Curve (|i_net|)",
    )

    log_icorr = math.log10(p.icorr_uA_cm2)
    ax.scatter(
        [log_icorr],
        [p.Ecorr_V_SHE],
        color="#ff7f0e",
        edgecolor="#111111",
        s=65,
        zorder=5,
        label="Corrosion point (E_corr, i_corr)",
    )
    ax.axhline(p.Ecorr_V_SHE, color="#ff7f0e", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axvline(log_icorr, color="#ff7f0e", linestyle=":", linewidth=1.0, alpha=0.7)

    ax.annotate(
        f"E_corr = {p.Ecorr_V_SHE:+.3f} V vs. SHE\ni_corr = {p.icorr_uA_cm2:.3g} µA/cm²",
        xy=(log_icorr, p.Ecorr_V_SHE),
        xytext=(14, 14),
        textcoords="offset points",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff8ec", "edgecolor": "#ff7f0e", "alpha": 0.9},
        arrowprops={"arrowstyle": "->", "color": "#ff7f0e", "lw": 1.1},
    )

    ax.set_xlabel("log10(|i|) [µA/cm²]")
    ax.set_ylabel("Potential E [V vs. SHE]")
    if title is None:
        metal_prefix = f"{p.metal} " if p.metal else ""
        title = f"{metal_prefix}Tafel Polarization Curve (Wagner-Traud)"
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="best", fontsize=8.5)
    return fig, ax
