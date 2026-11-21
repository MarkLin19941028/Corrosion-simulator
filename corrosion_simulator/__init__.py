"""Corrosion Simulator package."""

from corrosion_simulator.database import (
    CORE_METALS,
    DEFAULT_DISSOLUTION_VALENCE,
    REFERENCE_ELECTRODE_OFFSETS_V_SHE,
    ExposureCondition,
    ExposureLookupResult,
    HarmonizedDatabase,
    convert_potential_to_she,
    normalize_current_density_to_uA_cm2,
    normalize_tafel_slope_to_mV_dec,
)

__all__ = [
    "CORE_METALS",
    "DEFAULT_DISSOLUTION_VALENCE",
    "REFERENCE_ELECTRODE_OFFSETS_V_SHE",
    "ExposureCondition",
    "ExposureLookupResult",
    "HarmonizedDatabase",
    "convert_potential_to_she",
    "normalize_current_density_to_uA_cm2",
    "normalize_tafel_slope_to_mV_dec",
]
