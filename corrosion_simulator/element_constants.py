"""Built-in physical constants for pure metallic elements and base-element inference.

Used as a fallback when a pure element appears in electrochemical datasets without an explicit
row in ``material_properties.csv``. Alloys and compounds (e.g., ``TiN``, ``TaN``, ``CoWP``)
should supply their own ``density_g_cm3`` and ``equivalent_weight_g_eq`` (or ``molar_mass_g_mol``
and ``valence``) in ``material_properties.csv``.
"""

from __future__ import annotations

import re
from typing import Any, Final

from corrosion_simulator.seed_data import SEED_MATERIAL_PROPERTIES

# Extended pure element physical constants beyond the 8 core semiconductor metals.
# Values reflect standard room-temperature metallic properties and common aqueous dissolution valence.
_EXTENDED_PURE_ELEMENTS: Final[dict[str, dict[str, Any]]] = {
    "Fe": {
        "material_name": "Iron",
        "category": "metal",
        "density_g_cm3": 7.874,
        "molar_mass_g_mol": 55.845,
        "valence": 2,
        "standard_potential_V_SHE": -0.447,
        "electronegativity_pauling": 1.83,
        "work_function_eV": 4.50,
        "passivation_tendency": 0.45,
        "pourbaix_stable_pH_min": 9.0,
        "pourbaix_stable_pH_max": 13.0,
    },
    "Cr": {
        "material_name": "Chromium",
        "category": "metal",
        "density_g_cm3": 7.19,
        "molar_mass_g_mol": 51.996,
        "valence": 3,
        "standard_potential_V_SHE": -0.744,
        "electronegativity_pauling": 1.66,
        "work_function_eV": 4.50,
        "passivation_tendency": 0.85,
        "pourbaix_stable_pH_min": 4.0,
        "pourbaix_stable_pH_max": 12.0,
    },
    "Ni": {
        "material_name": "Nickel",
        "category": "metal",
        "density_g_cm3": 8.908,
        "molar_mass_g_mol": 58.693,
        "valence": 2,
        "standard_potential_V_SHE": -0.257,
        "electronegativity_pauling": 1.91,
        "work_function_eV": 5.15,
        "passivation_tendency": 0.65,
        "pourbaix_stable_pH_min": 8.0,
        "pourbaix_stable_pH_max": 13.0,
    },
    "Sn": {
        "material_name": "Tin",
        "category": "metal",
        "density_g_cm3": 7.265,
        "molar_mass_g_mol": 118.71,
        "valence": 2,
        "standard_potential_V_SHE": -0.138,
        "electronegativity_pauling": 1.96,
        "work_function_eV": 4.42,
        "passivation_tendency": 0.55,
        "pourbaix_stable_pH_min": 3.0,
        "pourbaix_stable_pH_max": 10.0,
    },
    "Zn": {
        "material_name": "Zinc",
        "category": "metal",
        "density_g_cm3": 7.14,
        "molar_mass_g_mol": 65.38,
        "valence": 2,
        "standard_potential_V_SHE": -0.762,
        "electronegativity_pauling": 1.65,
        "work_function_eV": 4.33,
        "passivation_tendency": 0.35,
        "pourbaix_stable_pH_min": 8.5,
        "pourbaix_stable_pH_max": 11.5,
    },
    "Mg": {
        "material_name": "Magnesium",
        "category": "metal",
        "density_g_cm3": 1.738,
        "molar_mass_g_mol": 24.305,
        "valence": 2,
        "standard_potential_V_SHE": -2.372,
        "electronegativity_pauling": 1.31,
        "work_function_eV": 3.66,
        "passivation_tendency": 0.25,
        "pourbaix_stable_pH_min": 11.0,
        "pourbaix_stable_pH_max": 14.0,
    },
    "Au": {
        "material_name": "Gold",
        "category": "metal",
        "density_g_cm3": 19.30,
        "molar_mass_g_mol": 196.967,
        "valence": 3,
        "standard_potential_V_SHE": 1.498,
        "electronegativity_pauling": 2.54,
        "work_function_eV": 5.10,
        "passivation_tendency": 0.95,
        "pourbaix_stable_pH_min": 0.0,
        "pourbaix_stable_pH_max": 14.0,
    },
    "Mn": {
        "material_name": "Manganese",
        "category": "metal",
        "density_g_cm3": 7.21,
        "molar_mass_g_mol": 54.938,
        "valence": 2,
        "standard_potential_V_SHE": -1.185,
        "electronegativity_pauling": 1.55,
        "work_function_eV": 4.10,
        "passivation_tendency": 0.35,
        "pourbaix_stable_pH_min": 9.0,
        "pourbaix_stable_pH_max": 13.0,
    },
    "Pt": {
        "material_name": "Platinum",
        "category": "metal",
        "density_g_cm3": 21.45,
        "molar_mass_g_mol": 195.084,
        "valence": 2,
        "standard_potential_V_SHE": 1.188,
        "electronegativity_pauling": 2.28,
        "work_function_eV": 5.65,
        "passivation_tendency": 0.95,
        "pourbaix_stable_pH_min": 0.0,
        "pourbaix_stable_pH_max": 14.0,
    },
    "Pd": {
        "material_name": "Palladium",
        "category": "metal",
        "density_g_cm3": 12.023,
        "molar_mass_g_mol": 106.42,
        "valence": 2,
        "standard_potential_V_SHE": 0.951,
        "electronegativity_pauling": 2.20,
        "work_function_eV": 5.12,
        "passivation_tendency": 0.90,
        "pourbaix_stable_pH_min": 0.0,
        "pourbaix_stable_pH_max": 14.0,
    },
    "Ag": {
        "material_name": "Silver",
        "category": "metal",
        "density_g_cm3": 10.49,
        "molar_mass_g_mol": 107.868,
        "valence": 1,
        "standard_potential_V_SHE": 0.800,
        "electronegativity_pauling": 1.93,
        "work_function_eV": 4.26,
        "passivation_tendency": 0.60,
        "pourbaix_stable_pH_min": 0.0,
        "pourbaix_stable_pH_max": 14.0,
    },
    "Ir": {
        "material_name": "Iridium",
        "category": "metal",
        "density_g_cm3": 22.56,
        "molar_mass_g_mol": 192.217,
        "valence": 3,
        "standard_potential_V_SHE": 1.156,
        "electronegativity_pauling": 2.20,
        "work_function_eV": 5.27,
        "passivation_tendency": 0.95,
        "pourbaix_stable_pH_min": 0.0,
        "pourbaix_stable_pH_max": 14.0,
    },
    "Pb": {
        "material_name": "Lead",
        "category": "metal",
        "density_g_cm3": 11.34,
        "molar_mass_g_mol": 207.2,
        "valence": 2,
        "standard_potential_V_SHE": -0.126,
        "electronegativity_pauling": 2.33,
        "work_function_eV": 4.25,
        "passivation_tendency": 0.55,
        "pourbaix_stable_pH_min": 7.0,
        "pourbaix_stable_pH_max": 11.0,
    },
    "Nb": {
        "material_name": "Niobium",
        "category": "metal",
        "density_g_cm3": 8.57,
        "molar_mass_g_mol": 92.906,
        "valence": 5,
        "standard_potential_V_SHE": -1.099,
        "electronegativity_pauling": 1.60,
        "work_function_eV": 4.30,
        "passivation_tendency": 0.90,
        "pourbaix_stable_pH_min": 0.0,
        "pourbaix_stable_pH_max": 14.0,
    },
    "Zr": {
        "material_name": "Zirconium",
        "category": "metal",
        "density_g_cm3": 6.52,
        "molar_mass_g_mol": 91.224,
        "valence": 4,
        "standard_potential_V_SHE": -1.450,
        "electronegativity_pauling": 1.33,
        "work_function_eV": 4.05,
        "passivation_tendency": 0.92,
        "pourbaix_stable_pH_min": 0.0,
        "pourbaix_stable_pH_max": 14.0,
    },
    "Hf": {
        "material_name": "Hafnium",
        "category": "metal",
        "density_g_cm3": 13.31,
        "molar_mass_g_mol": 178.49,
        "valence": 4,
        "standard_potential_V_SHE": -1.550,
        "electronegativity_pauling": 1.30,
        "work_function_eV": 3.90,
        "passivation_tendency": 0.92,
        "pourbaix_stable_pH_min": 0.0,
        "pourbaix_stable_pH_max": 14.0,
    },
    "V": {
        "material_name": "Vanadium",
        "category": "metal",
        "density_g_cm3": 6.00,
        "molar_mass_g_mol": 50.942,
        "valence": 3,
        "standard_potential_V_SHE": -1.175,
        "electronegativity_pauling": 1.63,
        "work_function_eV": 4.30,
        "passivation_tendency": 0.65,
        "pourbaix_stable_pH_min": 4.0,
        "pourbaix_stable_pH_max": 8.0,
    },
    "In": {
        "material_name": "Indium",
        "category": "metal",
        "density_g_cm3": 7.31,
        "molar_mass_g_mol": 114.818,
        "valence": 3,
        "standard_potential_V_SHE": -0.338,
        "electronegativity_pauling": 1.78,
        "work_function_eV": 4.12,
        "passivation_tendency": 0.50,
        "pourbaix_stable_pH_min": 5.0,
        "pourbaix_stable_pH_max": 10.0,
    },
    "Si": {
        "material_name": "Silicon",
        "category": "semiconductor",
        "density_g_cm3": 2.329,
        "molar_mass_g_mol": 28.085,
        "valence": 4,
        "standard_potential_V_SHE": -0.857,
        "electronegativity_pauling": 1.90,
        "work_function_eV": 4.85,
        "passivation_tendency": 0.85,
        "pourbaix_stable_pH_min": 0.0,
        "pourbaix_stable_pH_max": 9.0,
    },
    "Ge": {
        "material_name": "Germanium",
        "category": "semiconductor",
        "density_g_cm3": 5.323,
        "molar_mass_g_mol": 72.630,
        "valence": 4,
        "standard_potential_V_SHE": 0.124,
        "electronegativity_pauling": 2.01,
        "work_function_eV": 5.00,
        "passivation_tendency": 0.50,
        "pourbaix_stable_pH_min": 2.0,
        "pourbaix_stable_pH_max": 8.0,
    },
    "Rh": {
        "material_name": "Rhodium",
        "category": "metal",
        "density_g_cm3": 12.41,
        "molar_mass_g_mol": 102.906,
        "valence": 3,
        "standard_potential_V_SHE": 0.758,
        "electronegativity_pauling": 2.28,
        "work_function_eV": 4.98,
        "passivation_tendency": 0.92,
        "pourbaix_stable_pH_min": 0.0,
        "pourbaix_stable_pH_max": 14.0,
    },
    "Re": {
        "material_name": "Rhenium",
        "category": "metal",
        "density_g_cm3": 21.02,
        "molar_mass_g_mol": 186.207,
        "valence": 4,
        "standard_potential_V_SHE": 0.300,
        "electronegativity_pauling": 1.90,
        "work_function_eV": 4.96,
        "passivation_tendency": 0.70,
        "pourbaix_stable_pH_min": 0.0,
        "pourbaix_stable_pH_max": 12.0,
    },
}


def _normalize_element_entry(sym: str, props: dict[str, Any]) -> dict[str, Any]:
    entry = dict(props)
    entry["metal"] = sym
    entry["symbol"] = sym
    entry["material_id"] = sym
    entry["base_element"] = sym
    entry["equivalent_weight_g_eq"] = round(
        float(entry["molar_mass_g_mol"]) / float(entry["valence"]), 4
    )
    return entry


def _build_pure_element_registry() -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for row in SEED_MATERIAL_PROPERTIES:
        sym = str(row["metal"]).strip()
        registry[sym] = _normalize_element_entry(sym, row)
    for sym, props in _EXTENDED_PURE_ELEMENTS.items():
        if sym not in registry:
            registry[sym] = _normalize_element_entry(sym, props)
    return registry


PURE_ELEMENT_PROPERTIES: Final[dict[str, dict[str, Any]]] = _build_pure_element_registry()

_ELEMENT_TOKEN_RE = re.compile(r"^([A-Z][a-z]?)")


def lookup_pure_element_properties(symbol: str) -> dict[str, Any] | None:
    """Return built-in physical properties for an exact pure element symbol, or ``None``."""
    key = str(symbol).strip()
    entry = PURE_ELEMENT_PROPERTIES.get(key)
    return dict(entry) if entry is not None else None


def infer_base_element(material_id: str) -> str | None:
    """Infer the primary metallic element symbol for a pure metal, compound, or alloy.

    Examples: ``"Cu" -> "Cu"``, ``"TiN" -> "Ti"``, ``"TaN" -> "Ta"``, ``"CoWP" -> "Co"``.
    Returns ``None`` if no known leading element symbol can be matched.
    """
    cleaned = str(material_id).strip()
    if not cleaned:
        return None
    if cleaned in PURE_ELEMENT_PROPERTIES:
        return cleaned
    match = _ELEMENT_TOKEN_RE.match(cleaned)
    if match:
        candidate = match.group(1)
        if candidate in PURE_ELEMENT_PROPERTIES:
            return candidate
        if len(candidate) == 2 and candidate[0] in PURE_ELEMENT_PROPERTIES:
            return candidate[0]
    return None
