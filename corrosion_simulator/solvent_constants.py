"""Physicochemical solvent property constants and mixture descriptor calculations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Final

from corrosion_simulator.database import normalize_category_label


@dataclass(frozen=True)
class SolventPhysicalProperties:
    """Standard physicochemical constants for a pure liquid solvent at ~25 °C."""

    canonical_name: str
    dielectric_constant: float
    polarity_index: float
    is_protic: float
    solvent_family: str


WATER_PROPERTIES: Final[SolventPhysicalProperties] = SolventPhysicalProperties(
    canonical_name="water",
    dielectric_constant=78.4,
    polarity_index=10.2,
    is_protic=1.0,
    solvent_family="water",
)

DEFAULT_UNKNOWN_ORGANIC_SOLVENT: Final[SolventPhysicalProperties] = SolventPhysicalProperties(
    canonical_name="unknown_organic",
    dielectric_constant=25.0,
    polarity_index=5.0,
    is_protic=0.0,
    solvent_family="other_organic",
)

SOLVENT_PHYSICAL_PROPERTIES: Final[dict[str, SolventPhysicalProperties]] = {
    "water": WATER_PROPERTIES,
    # Glycols & glycol ethers
    "EG": SolventPhysicalProperties("EG", 37.7, 6.9, 1.0, "glycol"),
    "PG": SolventPhysicalProperties("PG", 32.0, 6.2, 1.0, "glycol"),
    "DEG": SolventPhysicalProperties("DEG", 31.7, 6.0, 1.0, "glycol"),
    "TEG": SolventPhysicalProperties("TEG", 23.7, 5.6, 1.0, "glycol"),
    "BDG": SolventPhysicalProperties("BDG", 14.1, 4.6, 1.0, "glycol"),
    "PGME": SolventPhysicalProperties("PGME", 12.3, 4.8, 1.0, "glycol"),
    "PGMEA": SolventPhysicalProperties("PGMEA", 8.3, 4.2, 0.0, "glycol"),
    "glycerol": SolventPhysicalProperties("glycerol", 42.5, 7.2, 1.0, "glycol"),
    # Alcohols
    "IPA": SolventPhysicalProperties("IPA", 19.9, 3.9, 1.0, "alcohol"),
    "EtOH": SolventPhysicalProperties("EtOH", 24.5, 4.3, 1.0, "alcohol"),
    "MeOH": SolventPhysicalProperties("MeOH", 32.7, 5.1, 1.0, "alcohol"),
    "1-propanol": SolventPhysicalProperties("1-propanol", 20.8, 4.0, 1.0, "alcohol"),
    "1-butanol": SolventPhysicalProperties("1-butanol", 17.8, 4.0, 1.0, "alcohol"),
    # Polar aprotic solvents
    "DMSO": SolventPhysicalProperties("DMSO", 46.7, 7.2, 0.0, "polar_aprotic"),
    "NMP": SolventPhysicalProperties("NMP", 32.2, 6.7, 0.0, "polar_aprotic"),
    "DMF": SolventPhysicalProperties("DMF", 36.7, 6.4, 0.0, "polar_aprotic"),
    "DMAc": SolventPhysicalProperties("DMAc", 37.8, 6.5, 0.0, "polar_aprotic"),
    "acetone": SolventPhysicalProperties("acetone", 20.7, 5.1, 0.0, "polar_aprotic"),
    "acetonitrile": SolventPhysicalProperties("acetonitrile", 37.5, 5.8, 0.0, "polar_aprotic"),
    "sulfolane": SolventPhysicalProperties("sulfolane", 43.3, 6.9, 0.0, "polar_aprotic"),
    "THF": SolventPhysicalProperties("THF", 7.6, 4.0, 0.0, "polar_aprotic"),
    "propylene carbonate": SolventPhysicalProperties("propylene carbonate", 64.9, 6.1, 0.0, "polar_aprotic"),
    "GBL": SolventPhysicalProperties("GBL", 39.0, 6.0, 0.0, "polar_aprotic"),
    "ethyl acetate": SolventPhysicalProperties("ethyl acetate", 6.0, 4.4, 0.0, "polar_aprotic"),
    # Organic acids
    "acetic acid": SolventPhysicalProperties("acetic acid", 6.2, 6.0, 1.0, "organic_acid"),
    "formic acid": SolventPhysicalProperties("formic acid", 58.5, 7.0, 1.0, "organic_acid"),
    "citric acid": SolventPhysicalProperties("citric acid", 25.0, 6.2, 1.0, "organic_acid"),
    "oxalic acid": SolventPhysicalProperties("oxalic acid", 25.0, 6.2, 1.0, "organic_acid"),
    "methanesulfonic acid": SolventPhysicalProperties("methanesulfonic acid", 25.0, 6.2, 1.0, "organic_acid"),
}

_SOLVENT_ALIASES: Final[dict[str, str]] = {
    "water": "water",
    "di water": "water",
    "di-water": "water",
    "h2o": "water",
    "aqueous": "water",
    "deionized water": "water",
    "deionised water": "water",
    "ultrapure water": "water",
    "upw": "water",
    "distilled water": "water",
    "eg": "EG",
    "ethylene glycol": "EG",
    "ethanediol": "EG",
    "pg": "PG",
    "propylene glycol": "PG",
    "1,2-propanediol": "PG",
    "deg": "DEG",
    "diethylene glycol": "DEG",
    "teg": "TEG",
    "triethylene glycol": "TEG",
    "bdg": "BDG",
    "butyl diglycol": "BDG",
    "pgme": "PGME",
    "pgmea": "PGMEA",
    "glycerol": "glycerol",
    "glycerin": "glycerol",
    "ipa": "IPA",
    "isopropanol": "IPA",
    "isopropyl alcohol": "IPA",
    "2-propanol": "IPA",
    "etoh": "EtOH",
    "ethanol": "EtOH",
    "ethyl alcohol": "EtOH",
    "meoh": "MeOH",
    "methanol": "MeOH",
    "methyl alcohol": "MeOH",
    "1-propanol": "1-propanol",
    "n-propanol": "1-propanol",
    "propanol": "1-propanol",
    "1-butanol": "1-butanol",
    "n-butanol": "1-butanol",
    "butanol": "1-butanol",
    "dmso": "DMSO",
    "dimethyl sulfoxide": "DMSO",
    "dimethylsulfoxide": "DMSO",
    "nmp": "NMP",
    "n-methyl-2-pyrrolidone": "NMP",
    "n-methylpyrrolidone": "NMP",
    "dmf": "DMF",
    "dimethylformamide": "DMF",
    "n,n-dimethylformamide": "DMF",
    "dmac": "DMAc",
    "dma": "DMAc",
    "dimethylacetamide": "DMAc",
    "acetone": "acetone",
    "propanone": "acetone",
    "acetonitrile": "acetonitrile",
    "acn": "acetonitrile",
    "mecn": "acetonitrile",
    "sulfolane": "sulfolane",
    "thf": "THF",
    "tetrahydrofuran": "THF",
    "propylene carbonate": "propylene carbonate",
    "pc": "propylene carbonate",
    "gbl": "GBL",
    "gamma-butyrolactone": "GBL",
    "ethyl acetate": "ethyl acetate",
    "acetic acid": "acetic acid",
    "glacial acetic acid": "acetic acid",
    "formic acid": "formic acid",
    "citric acid": "citric acid",
    "oxalic acid": "oxalic acid",
    "methanesulfonic acid": "methanesulfonic acid",
    "msa": "methanesulfonic acid",
}


def lookup_solvent_properties(solvent_name: Any) -> SolventPhysicalProperties:
    """Resolve physicochemical properties for water or an organic solvent with safe fallback."""
    label = normalize_category_label(solvent_name)
    if label == "None":
        return DEFAULT_UNKNOWN_ORGANIC_SOLVENT
    norm = " ".join(label.lower().split())
    canonical = _SOLVENT_ALIASES.get(norm)
    if canonical is not None:
        return SOLVENT_PHYSICAL_PROPERTIES[canonical]
    for alias, target in _SOLVENT_ALIASES.items():
        if len(alias) >= 4 and alias in norm:
            return SOLVENT_PHYSICAL_PROPERTIES[target]
    return DEFAULT_UNKNOWN_ORGANIC_SOLVENT


def compute_mixture_solvent_descriptors(
    water_content_wt_pct: float,
    organic_solvent_type: Any = None,
    base_solvent: Any = None,
) -> tuple[float, float, float]:
    """Compute water-fraction-weighted ``(effective_dielectric_constant, solvent_polarity_index, is_protic_solvent)``."""
    w_pct = 100.0 if (water_content_wt_pct is None or not math.isfinite(float(water_content_wt_pct))) else float(water_content_wt_pct)
    w_frac = min(1.0, max(0.0, w_pct / 100.0))
    if w_frac >= 1.0 - 1e-12:
        return (
            WATER_PROPERTIES.dielectric_constant,
            WATER_PROPERTIES.polarity_index,
            WATER_PROPERTIES.is_protic,
        )

    org_label = normalize_category_label(organic_solvent_type)
    if org_label != "None":
        org_props = lookup_solvent_properties(org_label)
    else:
        base_props = lookup_solvent_properties(base_solvent)
        org_props = (
            DEFAULT_UNKNOWN_ORGANIC_SOLVENT
            if base_props.canonical_name == "water"
            else base_props
        )

    o_frac = 1.0 - w_frac
    eff_dielectric = w_frac * WATER_PROPERTIES.dielectric_constant + o_frac * org_props.dielectric_constant
    eff_polarity = w_frac * WATER_PROPERTIES.polarity_index + o_frac * org_props.polarity_index
    eff_protic = w_frac * WATER_PROPERTIES.is_protic + o_frac * org_props.is_protic
    return float(eff_dielectric), float(eff_polarity), float(eff_protic)
