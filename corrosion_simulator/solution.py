"""Deep domain value objects for Solution and Cleaning Recipe formulations."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Final

from corrosion_simulator.database import (
    AQUEOUS_WATER_MIN_WT_PCT,
    DEFAULT_NON_AQUEOUS_WATER_WT_PCT,
    DEFAULT_SEMI_AQUEOUS_WATER_WT_PCT,
    SEMI_AQUEOUS_WATER_MIN_WT_PCT,
    SOLVENT_REGIMES,
    SolventRegime,
    SolventSubclass,
    classify_solvent_regime,
    classify_solvent_subclass,
    normalize_category_label,
)

NON_AQUEOUS_WARNING_PREFIX: Final[str] = "Non-aqueous Exposure Condition"


def _default_water_content_for_regime(regime: SolventRegime, org_type: str) -> float:
    """Return the canonical default water content (wt%) for ``regime`` and normalized ``org_type``."""
    if regime == "aqueous":
        return 100.0 if org_type == "None" else 95.0
    if regime == "semi_aqueous":
        return float(DEFAULT_SEMI_AQUEOUS_WATER_WT_PCT)
    return float(DEFAULT_NON_AQUEOUS_WATER_WT_PCT)


def resolve_consistent_solvent_settings(
    solvent_regime: str | None,
    organic_solvent_type: Any,
    water_content_wt_pct: float | None,
) -> tuple[SolventRegime, str, float]:
    """Resolve and validate mutually consistent ``(solvent_regime, organic_solvent_type, water_content_wt_pct)``."""
    org_type = normalize_category_label(organic_solvent_type)

    regime_val: SolventRegime | None = None
    if solvent_regime is not None:
        regime_str = str(solvent_regime).strip()
        if regime_str not in SOLVENT_REGIMES:
            raise ValueError(
                f"Unsupported solvent_regime '{solvent_regime}'. Supported: {SOLVENT_REGIMES}"
            )
        regime_val = regime_str  # type: ignore[assignment]

    water_num: float | None = None
    if water_content_wt_pct is not None:
        if isinstance(water_content_wt_pct, bool):
            raise ValueError("water_content_wt_pct must be a finite numeric percentage in [0, 100].")
        try:
            water_num = float(water_content_wt_pct)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"water_content_wt_pct must be numeric, got {water_content_wt_pct!r}."
            ) from exc
        if not math.isfinite(water_num) or not (0.0 <= water_num <= 100.0):
            raise ValueError(
                f"water_content_wt_pct must be finite and within [0.0, 100.0], got {water_content_wt_pct!r}."
            )

    if regime_val is None and water_num is None:
        inferred: SolventRegime = "semi_aqueous" if org_type != "None" else "aqueous"
        return (inferred, org_type, _default_water_content_for_regime(inferred, org_type))

    if regime_val is not None and water_num is None:
        return (regime_val, org_type, _default_water_content_for_regime(regime_val, org_type))

    if regime_val is None and water_num is not None:
        return (classify_solvent_regime(water_num), org_type, float(water_num))

    assert regime_val is not None and water_num is not None
    derived_regime = classify_solvent_regime(water_num)
    if derived_regime != regime_val:
        raise ValueError(
            f"Inconsistent solvent_regime='{regime_val}' and water_content_wt_pct={water_num:g}. "
            f"Regime boundaries require aqueous >= {AQUEOUS_WATER_MIN_WT_PCT:g} wt%, "
            f"{SEMI_AQUEOUS_WATER_MIN_WT_PCT:g} <= semi_aqueous < {AQUEOUS_WATER_MIN_WT_PCT:g} wt%, "
            f"and non_aqueous < {SEMI_AQUEOUS_WATER_MIN_WT_PCT:g} wt% water."
        )
    return (regime_val, org_type, float(water_num))


def water_content_slider_bounds(
    regime: str,
    organic_solvent_type: str = "None",
) -> tuple[float, float, float, float]:
    """Return ``(min_wt_pct, max_wt_pct, default_wt_pct, step)`` consistent with ``regime``."""
    if regime not in SOLVENT_REGIMES:
        raise ValueError(f"Unsupported solvent_regime '{regime}'. Supported: {SOLVENT_REGIMES}")
    regime_val: SolventRegime = regime  # type: ignore[assignment]
    org_type = normalize_category_label(organic_solvent_type)
    default_wt = _default_water_content_for_regime(regime_val, org_type)
    if regime_val == "aqueous":
        return (float(AQUEOUS_WATER_MIN_WT_PCT), 100.0, default_wt, 0.5)
    if regime_val == "semi_aqueous":
        return (
            float(SEMI_AQUEOUS_WATER_MIN_WT_PCT),
            round(float(AQUEOUS_WATER_MIN_WT_PCT) - 0.1, 1),
            default_wt,
            0.5,
        )
    return (0.0, round(float(SEMI_AQUEOUS_WATER_MIN_WT_PCT) - 0.01, 2), default_wt, 0.01)


def build_non_aqueous_physics_warning(
    water_content_wt_pct: float,
    organic_solvent_type: str = "None",
) -> str:
    """Build the standardized physics-limitation banner warning for non-aqueous (<1 wt% water) conditions."""
    org_label = normalize_category_label(organic_solvent_type)
    co_solvent = f" ({org_label})" if org_label != "None" else ""
    return (
        f"{NON_AQUEOUS_WARNING_PREFIX}{co_solvent} with {water_content_wt_pct:.2f} wt% water (< 1%): "
        "aqueous pH, V vs. SHE reference potential scale, and aqueous Pourbaix passivation assumptions "
        "are not strictly valid in this phase. Interpret simulated kinetics as exploratory screening only."
    )


@dataclass(frozen=True)
class Solution:
    """Liquid chemical formulation independent of operating temperature or immersion duration."""

    oxidizer_type: str = "H2O2"
    oxidizer_concentration: float = 1.0
    acid_type: str = "None"
    acid_concentration: float = 0.0
    inhibitor_type: str = "None"
    inhibitor_concentration: float = 0.0
    pH: float = 3.0
    ORP_mV: float | None = None
    solvent_regime: SolventRegime | None = None
    organic_solvent_type: str = "None"
    water_content_wt_pct: float | None = None

    def __post_init__(self) -> None:
        ox_type = normalize_category_label(self.oxidizer_type)
        acid_type = normalize_category_label(self.acid_type)
        inh_type = normalize_category_label(self.inhibitor_type)

        ox_conc = 0.0 if ox_type == "None" else float(self.oxidizer_concentration)
        acid_conc = 0.0 if acid_type == "None" else float(self.acid_concentration)
        inh_conc = 0.0 if inh_type == "None" else float(self.inhibitor_concentration)

        if not math.isfinite(float(self.pH)):
            raise ValueError(f"pH must be finite, got {self.pH!r}.")
        if self.ORP_mV is not None and not math.isfinite(float(self.ORP_mV)):
            raise ValueError(f"ORP_mV must be None or finite, got {self.ORP_mV!r}.")

        resolved_regime, resolved_org, resolved_water = resolve_consistent_solvent_settings(
            self.solvent_regime,
            self.organic_solvent_type,
            self.water_content_wt_pct,
        )

        object.__setattr__(self, "oxidizer_type", ox_type)
        object.__setattr__(self, "oxidizer_concentration", ox_conc)
        object.__setattr__(self, "acid_type", acid_type)
        object.__setattr__(self, "acid_concentration", acid_conc)
        object.__setattr__(self, "inhibitor_type", inh_type)
        object.__setattr__(self, "inhibitor_concentration", inh_conc)
        object.__setattr__(self, "pH", float(self.pH))
        object.__setattr__(self, "ORP_mV", None if self.ORP_mV is None else float(self.ORP_mV))
        object.__setattr__(self, "solvent_regime", resolved_regime)
        object.__setattr__(self, "organic_solvent_type", resolved_org)
        object.__setattr__(self, "water_content_wt_pct", resolved_water)

    @property
    def base_solvent(self) -> str:
        """Canonical base solvent label passed to the ML feature extractor."""
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

    @property
    def has_inhibitor(self) -> bool:
        """Return True when a non-None corrosion inhibitor is present at positive concentration."""
        return self.inhibitor_type != "None" and self.inhibitor_concentration > 0.0

    @property
    def is_non_aqueous(self) -> bool:
        """Return True when the Solution lies in the non-aqueous (< 1 wt% water) regime."""
        return (
            self.solvent_regime == "non_aqueous"
            or float(self.water_content_wt_pct or 0.0) < SEMI_AQUEOUS_WATER_MIN_WT_PCT
        )

    def non_aqueous_warning(self) -> str | None:
        """Return the standardized physics-limitation warning if non-aqueous, else None."""
        if not self.is_non_aqueous:
            return None
        return build_non_aqueous_physics_warning(
            float(self.water_content_wt_pct or 0.0),
            self.organic_solvent_type,
        )


@dataclass(frozen=True)
class CleaningRecipe:
    """Process specification consisting of a Solution, operating Temperature, and Immersion Time."""

    solution: Solution = field(default_factory=Solution)
    temperature_C: float = 25.0
    immersion_time_s: float = 300.0

    def __post_init__(self) -> None:
        if not isinstance(self.solution, Solution):
            raise TypeError(f"CleaningRecipe.solution must be a Solution, got {type(self.solution).__name__}.")
        if not math.isfinite(float(self.temperature_C)):
            raise ValueError(f"temperature_C must be finite, got {self.temperature_C!r}.")
        if float(self.immersion_time_s) <= 0.0 or not math.isfinite(float(self.immersion_time_s)):
            raise ValueError("immersion_time_s must be strictly positive.")
        object.__setattr__(self, "temperature_C", float(self.temperature_C))
        object.__setattr__(self, "immersion_time_s", float(self.immersion_time_s))
