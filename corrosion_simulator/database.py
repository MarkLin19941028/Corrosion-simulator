"""Database harmonization, unit normalization, Galvanic Series ranking, and Exposure Condition lookup."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from corrosion_simulator.element_constants import (
    PURE_ELEMENT_PROPERTIES,
    infer_base_element,
)
from corrosion_simulator.seed_data import SEED_MATERIAL_PROPERTIES, build_seed_tables

CORE_METALS: Final[tuple[str, ...]] = ("Cu", "Co", "W", "Ru", "Ti", "Ta", "Al", "Mo")

DEFAULT_DISSOLUTION_VALENCE: Final[dict[str, int]] = {
    "Cu": 2,
    "Co": 2,
    "W": 6,
    "Ru": 4,
    "Ti": 4,
    "Ta": 5,
    "Al": 3,
    "Mo": 6,
}

DEFAULT_BETA_A_MV_DEC: Final[float] = 60.0
DEFAULT_BETA_C_MV_DEC: Final[float] = 120.0
TAFEL_SLOPE_MIN_MV_DEC: Final[float] = 20.0
TAFEL_SLOPE_MAX_MV_DEC: Final[float] = 500.0
ECORR_MIN_V_SHE: Final[float] = -2.5
ECORR_MAX_V_SHE: Final[float] = 2.5
ICORR_MIN_UA_CM2: Final[float] = 1e-4
ICORR_MAX_UA_CM2: Final[float] = 1e5
IMPUTATION_QUALITY_PENALTY: Final[float] = 0.15
CR_DERIVED_QUALITY_PENALTY: Final[float] = 0.15
RP_DERIVED_QUALITY_PENALTY: Final[float] = 0.25
INFERRED_NEUTRAL_PH_WEIGHT_FACTOR: Final[float] = 0.75
_FARADAY_CONSTANT_C_MOL: Final[float] = 96485.33212

_CR_TO_NM_MIN: Final[dict[str, float]] = {
    "nm/min": 1.0,
    "nm min-1": 1.0,
    "a/min": 0.1,
    "å/min": 0.1,
    "mm/yr": 1e6 / 525600.0,
    "mm/year": 1e6 / 525600.0,
    "mm/y": 1e6 / 525600.0,
    "mmpy": 1e6 / 525600.0,
    "mm a": 1e6 / 525600.0,
    "mm y": 1e6 / 525600.0,
    "mm yr": 1e6 / 525600.0,
    "mm year-1": 1e6 / 525600.0,
    "mm/a": 1e6 / 525600.0,
    "mm/year x 10-3": 1e3 / 525600.0,
    "um/y": 1e3 / 525600.0,
    "um/yr": 1e3 / 525600.0,
    "um/year": 1e3 / 525600.0,
    "μm/y": 1e3 / 525600.0,
    "μm/yr": 1e3 / 525600.0,
    "µm/y": 1e3 / 525600.0,
    "µm/yr": 1e3 / 525600.0,
    "mpy": 25400.0 / 525600.0,
}

# Standard reference electrode potentials (V vs. SHE at 25 °C)
REFERENCE_ELECTRODE_OFFSETS_V_SHE: Final[dict[str, float]] = {
    "she": 0.000,
    "nhe": 0.000,
    "standard hydrogen electrode": 0.000,
    "ag/agcl": 0.197,
    "ag/agcl (sat)": 0.197,
    "ag/agcl (sat. kcl)": 0.197,
    "ag/agcl (3m kcl)": 0.210,
    "ag/agcl (3.5m kcl)": 0.205,
    "ag/agcl (1m kcl)": 0.235,
    "sce": 0.241,
    "saturated calomel electrode": 0.241,
    "mse": 0.640,
    "hg/hg2so4": 0.640,
    "mercurous sulfate": 0.640,
    "hg/hgo": 0.098,
    "cu/cuso4": 0.314,
    "cse": 0.314,
}

_CURRENT_DENSITY_TO_UA_CM2: Final[dict[str, float]] = {
    "ua/cm2": 1.0,
    "ua/cm^2": 1.0,
    "ua/cm²": 1.0,
    "µa/cm2": 1.0,
    "µa/cm^2": 1.0,
    "µa/cm²": 1.0,
    "μa/cm2": 1.0,
    "μa/cm^2": 1.0,
    "μa/cm²": 1.0,
    "ua cm-2": 1.0,
    "ma/cm2": 1_000.0,
    "ma/cm^2": 1_000.0,
    "ma/cm²": 1_000.0,
    "ma cm-2": 1_000.0,
    "a/cm2": 1_000_000.0,
    "a/cm^2": 1_000_000.0,
    "a/cm²": 1_000_000.0,
    "a cm-2": 1_000_000.0,
    "na/cm2": 1e-3,
    "na/cm^2": 1e-3,
    "na/cm²": 1e-3,
    "na cm-2": 1e-3,
    "a/m2": 100.0,
    "a/m^2": 100.0,
    "a/m²": 100.0,
    "ma/m2": 0.1,
    "ma/m^2": 0.1,
    "ma/m²": 0.1,
}

# Raw scraping columns that use avoided terminology from GLOSSARY.md
_RAW_COLUMN_RENAMES: Final[dict[str, str]] = {
    "electrolyte_additive": "supporting_salt_type",
    "electrolyte_additive_concentration": "supporting_salt_concentration",
    "electrolyte_additive_concentration_unit": "supporting_salt_concentration_unit",
}

_DROPPED_INTERNAL_COLUMNS: Final[set[str]] = {
    "is_target_metal",
    "_source_metal",
    "material_assignment_method",
    "paper_primary_material",
    "material_mention_counts",
}


def convert_potential_to_she(
    potential: float,
    reference_electrode: str = "SHE",
    *,
    unit: str = "V",
    pH: float = 7.0,
) -> float:
    """Convert an electrode potential to V vs. SHE."""
    if potential is None or pd.isna(potential):
        return float("nan")
    val_v = float(potential)
    unit_norm = (str(unit) if unit is not None and not pd.isna(unit) else "V").strip().lower()
    if unit_norm in ("mv", "millivolt", "millivolts"):
        val_v /= 1000.0
    elif unit_norm not in ("v", "volt", "volts", "v vs. she", "v_she"):
        raise ValueError(f"Unsupported potential unit: {unit!r}")

    if reference_electrode is None or pd.isna(reference_electrode):
        return float("nan")
    ref_norm = str(reference_electrode).strip().lower()
    if not ref_norm:
        return float("nan")

    if ref_norm == "rhe" or ref_norm.startswith("rhe"):
        ph_val = 7.0 if (pH is None or pd.isna(pH)) else float(pH)
        return val_v - 0.05916 * ph_val

    if ref_norm in REFERENCE_ELECTRODE_OFFSETS_V_SHE:
        return val_v + REFERENCE_ELECTRODE_OFFSETS_V_SHE[ref_norm]

    if "ag/agcl" in ref_norm or "agcl" in ref_norm:
        return val_v + REFERENCE_ELECTRODE_OFFSETS_V_SHE["ag/agcl"]
    if "sce" in ref_norm or "calomel" in ref_norm:
        return val_v + REFERENCE_ELECTRODE_OFFSETS_V_SHE["sce"]
    if "mse" in ref_norm or "hg2so4" in ref_norm:
        return val_v + REFERENCE_ELECTRODE_OFFSETS_V_SHE["mse"]
    if "hg/hgo" in ref_norm or "hgo" in ref_norm:
        return val_v + REFERENCE_ELECTRODE_OFFSETS_V_SHE["hg/hgo"]
    if "she" in ref_norm or "nhe" in ref_norm:
        return val_v

    raise ValueError(f"Unsupported reference electrode: {reference_electrode!r}")


def normalize_current_density_to_uA_cm2(
    current_density: float,
    unit: str = "uA/cm2",
) -> float:
    """Convert a current density value to µA/cm²."""
    if current_density is None or pd.isna(current_density):
        return float("nan")
    val = float(current_density)
    unit_str = "uA/cm2" if (unit is None or pd.isna(unit) or not str(unit).strip()) else str(unit)
    unit_norm = unit_str.strip().lower().replace("·", " ")
    if unit_norm not in _CURRENT_DENSITY_TO_UA_CM2:
        raise ValueError(f"Unsupported current density unit: {unit!r}")
    return val * _CURRENT_DENSITY_TO_UA_CM2[unit_norm]


def normalize_tafel_slope_to_mV_dec(
    slope: float,
    unit: str = "mV/dec",
) -> float:
    """Normalize an anodic or cathodic Tafel slope to strictly positive mV/dec."""
    if slope is None or pd.isna(slope):
        return float("nan")
    val = abs(float(slope))
    if val == 0.0:
        return float("nan")
    unit_str = "mV/dec" if (unit is None or pd.isna(unit) or not str(unit).strip()) else str(unit)
    unit_norm = unit_str.strip().lower()
    if unit_norm in ("v/dec", "v/decade", "v dec-1", "v"):
        val *= 1000.0
    elif unit_norm not in ("mv/dec", "mv/decade", "mv dec-1", "mv"):
        raise ValueError(f"Unsupported Tafel slope unit: {unit!r}")
    return val


def derive_icorr_from_cr(
    cr_value: float,
    cr_unit: str = "nm/min",
    *,
    molar_mass_g_mol: float,
    density_g_cm3: float,
    valence: float,
) -> float:
    """Invert Faraday's law to estimate i_corr (µA/cm²) from a reported Corrosion Rate.

    Returns ``nan`` if any physical property is missing/non-positive or the CR unit is unsupported.
    """
    if (
        cr_value is None
        or pd.isna(cr_value)
        or molar_mass_g_mol is None
        or pd.isna(molar_mass_g_mol)
        or density_g_cm3 is None
        or pd.isna(density_g_cm3)
        or valence is None
        or pd.isna(valence)
    ):
        return float("nan")
    cr_f = float(cr_value)
    m_f = float(molar_mass_g_mol)
    rho_f = float(density_g_cm3)
    z_f = float(valence)
    if cr_f <= 0.0 or m_f <= 0.0 or rho_f <= 0.0 or z_f <= 0.0:
        return float("nan")
    if cr_unit is None or pd.isna(cr_unit) or not str(cr_unit).strip():
        return float("nan")
    unit_key = " ".join(str(cr_unit).strip().lower().split())
    factor = _CR_TO_NM_MIN.get(unit_key)
    if factor is None:
        return float("nan")
    cr_nm_min = cr_f * factor
    # CR (nm/min) = i_corr (uA/cm2) * M * 600 / (z * F * rho)
    icorr_uA_cm2 = (cr_nm_min * z_f * _FARADAY_CONSTANT_C_MOL * rho_f) / (600.0 * m_f)
    return float(icorr_uA_cm2) if (math.isfinite(icorr_uA_cm2) and icorr_uA_cm2 > 0.0) else float("nan")


def derive_icorr_from_rp(
    rp_value: float,
    rp_unit: str = "ohm·cm2",
    *,
    beta_a_mV_dec: float = DEFAULT_BETA_A_MV_DEC,
    beta_c_mV_dec: float = DEFAULT_BETA_C_MV_DEC,
) -> float:
    """Apply the Stern-Geary relation to estimate i_corr (µA/cm²) from polarization resistance R_p.

    Returns ``nan`` if ``rp_value`` is non-positive or ``rp_unit`` cannot be parsed to Ω·cm².
    """
    if rp_value is None or pd.isna(rp_value):
        return float("nan")
    rp_f = float(rp_value)
    if not math.isfinite(rp_f) or rp_f <= 0.0:
        return float("nan")

    unit_raw = "ohm·cm2" if (rp_unit is None or pd.isna(rp_unit) or not str(rp_unit).strip()) else str(rp_unit).strip()
    unit_compact = "".join(unit_raw.split())
    unit_lower = unit_compact.lower()

    if unit_compact.startswith("MΩ") or unit_compact.startswith("Mohm") or unit_compact.startswith("MOhm") or unit_compact.startswith("Mῼ"):
        rp_ohm_cm2 = rp_f * 1e6
    elif unit_lower.startswith("kω") or unit_lower.startswith("kohm") or unit_lower.startswith("kῼ"):
        rp_ohm_cm2 = rp_f * 1e3
    elif any(tok in unit_lower for tok in ("ohm", "ω", "ῼ")):
        rp_ohm_cm2 = rp_f
    else:
        return float("nan")

    ba = float(beta_a_mV_dec) if (beta_a_mV_dec is not None and not pd.isna(beta_a_mV_dec) and TAFEL_SLOPE_MIN_MV_DEC <= float(beta_a_mV_dec) <= TAFEL_SLOPE_MAX_MV_DEC) else DEFAULT_BETA_A_MV_DEC
    bc = float(beta_c_mV_dec) if (beta_c_mV_dec is not None and not pd.isna(beta_c_mV_dec) and TAFEL_SLOPE_MIN_MV_DEC <= float(beta_c_mV_dec) <= TAFEL_SLOPE_MAX_MV_DEC) else DEFAULT_BETA_C_MV_DEC
    # Stern-Geary constant B in mV: B = (ba * bc) / (ln(10) * (ba + bc))
    B_mV = (ba * bc) / (math.log(10.0) * (ba + bc))
    # i_corr (uA/cm2) = (B_mV * 1e-3 V / R_p) * 1e6 = B_mV * 1000 / R_p
    icorr_uA_cm2 = (B_mV * 1000.0) / rp_ohm_cm2
    return float(icorr_uA_cm2) if (math.isfinite(icorr_uA_cm2) and icorr_uA_cm2 > 0.0) else float("nan")


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(c).lstrip("\ufeff").strip() for c in cleaned.columns]
    cleaned = cleaned.rename(columns=_RAW_COLUMN_RENAMES)
    drop_cols = [c for c in _DROPPED_INTERNAL_COLUMNS if c in cleaned.columns]
    if drop_cols:
        cleaned = cleaned.drop(columns=drop_cols)
    return cleaned


def normalize_category_label(value: Any) -> str:
    """Normalize a chemical or categorical label, mapping null/empty/none tokens to 'None'."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return "None"
    s = str(value).strip()
    if not s or s.lower() in ("none", "nan", "null", "n/a", "na"):
        return "None"
    return s


_normalize_oxidizer_label = normalize_category_label


def _coalesce_column(df: pd.DataFrame, candidates: list[str], default: Any = np.nan) -> pd.Series:
    result = pd.Series([np.nan] * len(df), index=df.index, dtype=object)
    for col in candidates:
        if col in df.columns:
            mask = result.isna() | (result.astype(str).str.strip() == "")
            result = result.where(~mask, df[col])
    if not pd.isna(default):
        mask = result.isna() | (result.astype(str).str.strip() == "")
        result = result.where(~mask, default)
    return result


def _normalize_slope_series(
    df: pd.DataFrame,
    normalized_col: str,
    raw_col: str,
    specific_unit_col: str,
) -> list[float]:
    pre_norm = (
        pd.to_numeric(df[normalized_col], errors="coerce")
        if normalized_col in df.columns
        else pd.Series([np.nan] * len(df), index=df.index)
    )
    raw_vals = (
        pd.to_numeric(df[raw_col], errors="coerce")
        if raw_col in df.columns
        else pd.Series([np.nan] * len(df), index=df.index)
    )
    unit_col = (
        df[specific_unit_col]
        if specific_unit_col in df.columns
        else (
            df["beta_unit"]
            if "beta_unit" in df.columns
            else pd.Series(["mV/dec"] * len(df), index=df.index)
        )
    )
    results: list[float] = []
    for idx in df.index:
        if not pd.isna(pre_norm.loc[idx]):
            # Value is already explicitly in mV/dec; only enforce strictly positive sign
            val = normalize_tafel_slope_to_mV_dec(float(pre_norm.loc[idx]), "mV/dec")
        elif not pd.isna(raw_vals.loc[idx]):
            unit_str = str(unit_col.loc[idx]) if not pd.isna(unit_col.loc[idx]) else "mV/dec"
            val = normalize_tafel_slope_to_mV_dec(float(raw_vals.loc[idx]), unit_str)
        else:
            val = float("nan")
        if math.isfinite(val) and (TAFEL_SLOPE_MIN_MV_DEC <= val <= TAFEL_SLOPE_MAX_MV_DEC):
            results.append(val)
        else:
            results.append(float("nan"))
    return results


@dataclass(frozen=True)
class ExposureCondition:
    """The combination of a specific Metal, a Solution formulation, and an operating Temperature."""

    metal: str
    pH: float = 7.0
    oxidizer_type: str = "None"
    oxidizer_concentration: float = 0.0
    temperature_C: float = 25.0
    solution_id: str | None = None
    solution_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "metal": self.metal,
            "oxidizer_type": self.oxidizer_type,
            "pH": self.pH,
            "oxidizer_concentration": self.oxidizer_concentration,
            "temperature_C": self.temperature_C,
        }
        if self.solution_id is not None:
            payload["solution_id"] = self.solution_id
        if self.solution_name is not None:
            payload["solution_name"] = self.solution_name
        return payload


@dataclass(frozen=True)
class ExposureLookupResult:
    """Result of an exact or nearest-neighbor Exposure Condition database lookup."""

    record_id: str
    solution_id: str
    metal: str
    oxidizer_type: str
    Ecorr_V_SHE: float
    icorr_uA_cm2: float
    beta_a_mV_dec: float
    beta_c_mV_dec: float
    is_imputed: bool
    quality_score: float
    is_exact_match: bool
    normalized_distance: float
    query_condition: ExposureCondition
    matched_condition: ExposureCondition
    condition_differences: dict[str, float]
    mismatch_summary: str
    candidates: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    @property
    def query_conditions(self) -> dict[str, Any]:
        return self.query_condition.to_dict()

    @property
    def matched_conditions(self) -> dict[str, Any]:
        return self.matched_condition.to_dict()


@dataclass(frozen=True)
class HarmonizedDatabase:
    """Unified, unit-standardized electrochemical, material, solution, and Galvanic Series database."""

    electrochemical_database: pd.DataFrame
    material_properties: pd.DataFrame
    solution_properties: pd.DataFrame
    galvanic_series: pd.DataFrame = field(default_factory=pd.DataFrame)
    data_dictionary: pd.DataFrame = field(default_factory=pd.DataFrame)
    rejected_records: pd.DataFrame = field(default_factory=pd.DataFrame)
    partial_target_records: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def ml_training_dataframe(self) -> pd.DataFrame:
        """Combined DataFrame of complete pairs plus partial-target rows for per-target ML training."""
        if self.partial_target_records.empty:
            return self.electrochemical_database
        if self.electrochemical_database.empty:
            return self.partial_target_records
        return pd.concat(
            [self.electrochemical_database, self.partial_target_records],
            ignore_index=True,
        )

    @property
    def available_metals(self) -> list[str]:
        """Return distinct materials present in the database, ordered with core metals first."""
        counts = self.metal_sample_counts()
        present = set(counts.keys())
        core_ordered = [m for m in CORE_METALS if m in present]
        extended = sorted(
            (m for m in present if m not in CORE_METALS),
            key=lambda m: (-counts.get(m, 0), m),
        )
        return core_ordered + extended

    def metal_sample_counts(self) -> dict[str, int]:
        """Return the number of harmonized electrochemical records per material."""
        if self.electrochemical_database.empty or "metal" not in self.electrochemical_database.columns:
            return {}
        vc = self.electrochemical_database["metal"].astype(str).str.strip().value_counts()
        return {str(k): int(v) for k, v in vc.items()}

    def summarize_galvanic_series(self, metals: list[str] | None = None) -> pd.DataFrame:
        """Return median E_corr and i_corr Galvanic Series summary for the requested material subset."""
        gs = self.galvanic_series
        if gs.empty or "metal" not in gs.columns:
            return pd.DataFrame(
                columns=["metal", "median_Ecorr_V_SHE", "median_icorr_uA_cm2", "solutions_covered"]
            )
        subset = gs[gs["metal"].isin(metals)] if metals is not None else gs
        if subset.empty:
            return pd.DataFrame(
                columns=["metal", "median_Ecorr_V_SHE", "median_icorr_uA_cm2", "solutions_covered"]
            )
        return (
            subset.groupby("metal", as_index=False)
            .agg(
                median_Ecorr_V_SHE=("Ecorr_V_SHE", "median"),
                median_icorr_uA_cm2=("icorr_uA_cm2", "median"),
                solutions_covered=("solution_id", "nunique"),
            )
            .sort_values("median_Ecorr_V_SHE", ascending=True)
            .reset_index(drop=True)
        )

    def material_metadata(self) -> pd.DataFrame:
        """Return per-material summary metadata (tier, category, sample count, Faraday readiness)."""
        counts = self.metal_sample_counts()
        metals = self.available_metals
        mat_idx = (
            self.material_properties.set_index("metal")
            if not self.material_properties.empty and "metal" in self.material_properties.columns
            else pd.DataFrame()
        )
        rows: list[dict[str, Any]] = []
        for metal in metals:
            row = mat_idx.loc[metal] if (not mat_idx.empty and metal in mat_idx.index) else None
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            tier = (
                str(row["tier"])
                if row is not None and "tier" in row and not pd.isna(row["tier"])
                else ("core" if metal in CORE_METALS else "extended")
            )
            category = (
                str(row["category"])
                if row is not None and "category" in row and not pd.isna(row["category"])
                else "metal"
            )
            has_faraday = (
                bool(row["has_faraday_properties"])
                if row is not None
                and "has_faraday_properties" in row
                and not pd.isna(row["has_faraday_properties"])
                else False
            )
            rows.append(
                {
                    "metal": metal,
                    "tier": tier,
                    "category": category,
                    "n_samples": counts.get(metal, 0),
                    "has_faraday_properties": has_faraday,
                }
            )
        return pd.DataFrame(rows)

    @classmethod
    def from_seed(cls) -> HarmonizedDatabase:
        """Build a HarmonizedDatabase directly from the built-in 8-metal seed dataset."""
        seed_echem, seed_mat, seed_sol = build_seed_tables()
        return cls.from_raw_tables(
            seed_echem,
            material_df=seed_mat,
            solution_df=seed_sol,
            filter_to_core_metals=True,
            supplement_missing_core_metals=False,
        )

    @classmethod
    def from_directory(
        cls,
        raw_dir: str | Path = "data/raw",
        *,
        filter_to_core_metals: bool = False,
        supplement_missing_core_metals: bool = False,
    ) -> HarmonizedDatabase:
        """Ingest raw CSV tables from ``raw_dir`` (or fall back to the built-in seed dataset)."""
        base_path = Path(raw_dir)
        if not base_path.exists():
            return cls.from_seed()

        csv_files = sorted(base_path.rglob("*.csv"))
        if not csv_files:
            return cls.from_seed()

        echem_frames: list[pd.DataFrame] = []
        material_frames: list[pd.DataFrame] = []
        solution_frames: list[pd.DataFrame] = []

        for csv_path in csv_files:
            df = _clean_column_names(pd.read_csv(csv_path, encoding="utf-8-sig"))
            if df.empty:
                continue
            cols = set(df.columns)
            name_lower = csv_path.name.lower()

            has_echem_cols = bool(
                cols
                & {
                    "Ecorr_V_SHE",
                    "Ecorr_V_original",
                    "Ecorr",
                    "E_corr",
                    "icorr_uA_cm2",
                    "icorr",
                    "i_corr",
                    "log_icorr",
                }
            )
            if has_echem_cols or "electrochemical" in name_lower or "ml_training" in name_lower:
                if not (cols & {"metal", "symbol", "material_id"}):
                    if csv_path.parent != base_path:
                        parent_name = csv_path.parent.name.strip()
                        if parent_name and (not filter_to_core_metals or parent_name in CORE_METALS):
                            df["metal"] = parent_name
                echem_frames.append(df)
            elif "material_properties" in name_lower or (
                "density_g_cm3" in cols and "molar_mass_g_mol" in cols and not has_echem_cols
            ):
                material_frames.append(df)
            elif "solution_properties" in name_lower or (
                "solution_id" in cols and "solution_name" in cols and not has_echem_cols
            ):
                solution_frames.append(df)

        echem_combined = pd.concat(echem_frames, ignore_index=True) if echem_frames else pd.DataFrame()
        mat_combined = pd.concat(material_frames, ignore_index=True) if material_frames else None
        sol_combined = pd.concat(solution_frames, ignore_index=True) if solution_frames else None

        return cls.from_raw_tables(
            echem_combined,
            material_df=mat_combined,
            solution_df=sol_combined,
            filter_to_core_metals=filter_to_core_metals,
            supplement_missing_core_metals=supplement_missing_core_metals,
        )

    @classmethod
    def from_raw_tables(
        cls,
        electrochemical_df: pd.DataFrame,
        material_df: pd.DataFrame | None = None,
        solution_df: pd.DataFrame | None = None,
        *,
        filter_to_core_metals: bool = False,
        supplement_missing_core_metals: bool = False,
    ) -> HarmonizedDatabase:
        """Harmonize, deduplicate, normalize units, filter, and impute electrochemical tables."""
        echem_raw = _clean_column_names(electrochemical_df)
        mat_raw = _clean_column_names(material_df) if material_df is not None else pd.DataFrame()
        sol_raw = _clean_column_names(solution_df) if solution_df is not None else pd.DataFrame()

        if echem_raw.empty:
            seed_echem, seed_mat, seed_sol = build_seed_tables()
            echem_raw = seed_echem
            mat_raw = pd.concat([mat_raw, seed_mat], ignore_index=True)
            sol_raw = pd.concat([sol_raw, seed_sol], ignore_index=True)
        elif supplement_missing_core_metals:
            # Determine which of the 8 core metals lack valid (Ecorr, icorr) records in echem_raw
            prelim_mat = cls._harmonize_materials(mat_raw, echem_raw, filter_to_core_metals=filter_to_core_metals)
            prelim_sol, prelim_echem_sol = cls._harmonize_solutions(sol_raw, echem_raw)
            prelim_valid, _, _ = cls._harmonize_electrochemical_records(
                prelim_echem_sol,
                prelim_mat,
                prelim_sol,
                filter_to_core_metals=filter_to_core_metals,
            )
            present_metals = set(prelim_valid["metal"].unique()) if not prelim_valid.empty else set()
            missing_metals = [m for m in CORE_METALS if m not in present_metals]
            if missing_metals:
                seed_echem, seed_mat, seed_sol = build_seed_tables()
                seed_echem_sub = seed_echem[seed_echem["metal"].isin(missing_metals)].copy()
                seed_sol_ids = set(seed_echem_sub["solution_id"].unique())
                seed_sol_sub = seed_sol[seed_sol["solution_id"].isin(seed_sol_ids)].copy()
                seed_mat_sub = seed_mat[seed_mat["metal"].isin(missing_metals)].copy()
                echem_raw = pd.concat([echem_raw, seed_echem_sub], ignore_index=True)
                mat_raw = pd.concat([mat_raw, seed_mat_sub], ignore_index=True)
                sol_raw = pd.concat([sol_raw, seed_sol_sub], ignore_index=True)

        # 1. Harmonize Material Properties (deduplicated by Metal)
        harmonized_materials = cls._harmonize_materials(
            mat_raw, echem_raw, filter_to_core_metals=filter_to_core_metals
        )

        # 2. Harmonize Solution Properties (deduplicated by solution_id, independent of temperature)
        harmonized_solutions, echem_with_sol = cls._harmonize_solutions(sol_raw, echem_raw)

        # 3. Harmonize Electrochemical Records (potentials -> V vs. SHE, currents -> uA/cm2, slopes -> +mV/dec)
        harmonized_echem, partial_records, rejected_records = cls._harmonize_electrochemical_records(
            echem_with_sol,
            harmonized_materials,
            harmonized_solutions,
            filter_to_core_metals=filter_to_core_metals,
        )

        # 4. Build Galvanic Series ranking per Solution
        galvanic_series = cls._build_galvanic_series(harmonized_echem)

        # 5. Build Data Dictionary
        data_dictionary = cls._build_data_dictionary(
            harmonized_echem,
            harmonized_materials,
            harmonized_solutions,
            galvanic_series,
        )

        return cls(
            electrochemical_database=harmonized_echem,
            material_properties=harmonized_materials,
            solution_properties=harmonized_solutions,
            galvanic_series=galvanic_series,
            data_dictionary=data_dictionary,
            rejected_records=rejected_records,
            partial_target_records=partial_records,
        )

    @staticmethod
    def _harmonize_materials(
        mat_raw: pd.DataFrame,
        echem_raw: pd.DataFrame,
        *,
        filter_to_core_metals: bool,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        observed_metals: set[str] = set()

        if not mat_raw.empty:
            df = mat_raw.copy()
            df["metal"] = _coalesce_column(df, ["metal", "symbol", "material_id"])
            observed_metals.update(
                m for m in df["metal"].dropna().astype(str).str.strip() if m and m.lower() != "nan"
            )
            frames.append(df)

        if not echem_raw.empty:
            mat_cols = [
                c
                for c in (
                    "metal",
                    "symbol",
                    "material_id",
                    "material_name",
                    "category",
                    "base_element",
                    "density_g_cm3",
                    "molar_mass_g_mol",
                    "valence",
                    "equivalent_weight_g_eq",
                    "standard_potential_V_SHE",
                    "melting_point_C",
                    "electronegativity_pauling",
                    "work_function_eV",
                    "oxide_type",
                    "oxide_stability",
                    "passivation_tendency",
                    "pourbaix_stable_pH_min",
                    "pourbaix_stable_pH_max",
                    "crystal_structure",
                    "thermal_conductivity_W_mK",
                    "electrical_resistivity_ohm_cm",
                    "common_corrosion_products",
                )
                if c in echem_raw.columns
            ]
            if mat_cols:
                sub = echem_raw[mat_cols].copy()
                sub["metal"] = _coalesce_column(sub, ["metal", "symbol", "material_id"])
                observed_metals.update(
                    m for m in sub["metal"].dropna().astype(str).str.strip() if m and m.lower() != "nan"
                )
                frames.append(sub)

        # Supplement missing physical attributes for pure elements that actually appear in the dataset
        # (and for all core metals when filter_to_core_metals=True)
        fallback_rows: list[dict[str, Any]] = []
        seed_mat_map = {row["metal"]: row for row in SEED_MATERIAL_PROPERTIES}
        candidate_fallbacks = set(observed_metals)
        if filter_to_core_metals:
            candidate_fallbacks.update(CORE_METALS)
        for metal_sym in sorted(candidate_fallbacks):
            if metal_sym in seed_mat_map:
                fallback_rows.append(dict(seed_mat_map[metal_sym]))
            elif metal_sym in PURE_ELEMENT_PROPERTIES:
                fallback_rows.append(dict(PURE_ELEMENT_PROPERTIES[metal_sym]))
        if fallback_rows:
            frames.append(pd.DataFrame(fallback_rows))

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined["metal"] = combined["metal"].astype(str).str.strip()
        combined = combined[
            combined["metal"].notna() & (combined["metal"] != "") & (combined["metal"] != "nan")
        ]

        if filter_to_core_metals:
            combined = combined[combined["metal"].isin(CORE_METALS)]

        # Deduplicate by Metal, preferring first non-null attribute values (explicit mat_raw > echem_raw > pure-element fallback)
        deduped = combined.groupby("metal", as_index=False, sort=False).first()
        deduped["symbol"] = deduped["metal"]
        deduped["material_id"] = deduped["metal"]
        deduped["tier"] = np.where(deduped["metal"].isin(CORE_METALS), "core", "extended")

        if "category" not in deduped.columns:
            deduped["category"] = "metal"
        else:
            deduped["category"] = deduped["category"].fillna("metal").astype(str).str.strip()
            deduped.loc[deduped["category"].isin({"", "nan"}), "category"] = "metal"

        if "base_element" not in deduped.columns:
            deduped["base_element"] = deduped["metal"].map(infer_base_element)
        else:
            inferred = deduped["metal"].map(infer_base_element)
            deduped["base_element"] = deduped["base_element"].where(
                deduped["base_element"].notna() & (deduped["base_element"].astype(str).str.strip() != ""),
                inferred,
            )

        # Enforce canonical Dissolution Valence (z) for core metals; preserve explicit valence for extended materials/alloys
        raw_valence = (
            pd.to_numeric(deduped["valence"], errors="coerce")
            if "valence" in deduped.columns
            else pd.Series([np.nan] * len(deduped), index=deduped.index)
        )
        canonical_valence = pd.to_numeric(deduped["metal"].map(DEFAULT_DISSOLUTION_VALENCE), errors="coerce")
        deduped["valence"] = canonical_valence.where(canonical_valence.notna(), raw_valence)

        molar_mass = (
            pd.to_numeric(deduped["molar_mass_g_mol"], errors="coerce")
            if "molar_mass_g_mol" in deduped.columns
            else pd.Series([np.nan] * len(deduped), index=deduped.index)
        )
        density = (
            pd.to_numeric(deduped["density_g_cm3"], errors="coerce")
            if "density_g_cm3" in deduped.columns
            else pd.Series([np.nan] * len(deduped), index=deduped.index)
        )
        explicit_ew = (
            pd.to_numeric(deduped["equivalent_weight_g_eq"], errors="coerce")
            if "equivalent_weight_g_eq" in deduped.columns
            else pd.Series([np.nan] * len(deduped), index=deduped.index)
        )
        computed_ew = (molar_mass / deduped["valence"]).round(4)
        # For core metals, always synchronize EW = M / canonical z; for alloys/compounds, honor explicit EW if provided
        is_core = deduped["metal"].isin(CORE_METALS)
        deduped["equivalent_weight_g_eq"] = np.where(
            is_core,
            computed_ew,
            explicit_ew.where(explicit_ew.notna() & (explicit_ew > 0.0), computed_ew),
        )
        deduped["has_faraday_properties"] = (
            density.notna()
            & (density > 0.0)
            & pd.to_numeric(deduped["equivalent_weight_g_eq"], errors="coerce").notna()
            & (pd.to_numeric(deduped["equivalent_weight_g_eq"], errors="coerce") > 0.0)
        )
        return deduped.reset_index(drop=True)

    @staticmethod
    def _harmonize_solutions(
        sol_raw: pd.DataFrame,
        echem_raw: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        echem = echem_raw.copy()
        if "solution_id" not in echem.columns:
            echem["solution_id"] = np.nan

        def _derive_sol_id(row: pd.Series) -> str:
            sid = row.get("solution_id")
            if (
                sid is not None
                and not pd.isna(sid)
                and str(sid).strip()
                and str(sid).strip().lower() != "nan"
            ):
                return str(sid).strip()
            key = "|".join(
                str(row.get(k, ""))
                for k in (
                    "solution_name",
                    "solution_description",
                    "oxidizer_type",
                    "oxidizer_concentration",
                    "acid_type",
                    "acid_concentration",
                    "pH",
                    "pH_effective",
                )
            )
            digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
            return f"SOL_{digest}"

        echem["solution_id"] = echem.apply(_derive_sol_id, axis=1)

        # Per GLOSSARY.md, a Solution is defined by its chemical components and pH, independent of operating temperature
        sol_cols = [
            "solution_id",
            "solution_name",
            "solution_category",
            "base_solvent",
            "pH",
            "oxidizer_type",
            "oxidizer_concentration",
            "oxidizer_concentration_unit",
            "acid_type",
            "acid_concentration",
            "acid_concentration_unit",
            "base_type",
            "base_concentration",
            "base_concentration_unit",
            "HF_concentration_wt_pct",
            "chelating_agent_type",
            "chelating_agent_concentration",
            "chelating_agent_concentration_unit",
            "inhibitor_type",
            "inhibitor_concentration",
            "inhibitor_concentration_unit",
            "organic_solvent_type",
            "organic_solvent_concentration",
            "supporting_salt_type",
            "supporting_salt_concentration",
            "supporting_salt_concentration_unit",
            "dissolved_oxygen_ppm",
            "conductivity_mS_cm",
            "ionic_strength_mol_L",
            "ORP_mV",
            "deaerated",
            "materials",
            "source",
            "notes",
        ]

        frames: list[pd.DataFrame] = []
        if not sol_raw.empty:
            s_df = sol_raw.copy()
            if "temperature_C" in s_df.columns:
                s_df = s_df.drop(columns=["temperature_C"])
            s_df = s_df[s_df["solution_id"].notna()]
            s_df["solution_id"] = s_df["solution_id"].astype(str).str.strip()
            frames.append(s_df)

        if not echem.empty:
            present_cols = [c for c in sol_cols if c in echem.columns]
            e_sol = echem[present_cols].copy()
            if "solution_name" not in e_sol.columns and "solution_description" in echem.columns:
                e_sol["solution_name"] = echem["solution_description"]
            if "pH" in e_sol.columns and "pH_effective" in echem.columns:
                e_sol["pH"] = e_sol["pH"].fillna(echem["pH_effective"])
            frames.append(e_sol)

        if not frames:
            return (
                pd.DataFrame(
                    columns=[
                        "solution_id",
                        "solution_name",
                        "pH",
                        "oxidizer_type",
                        "oxidizer_concentration",
                    ]
                ),
                echem,
            )

        combined = pd.concat(frames, ignore_index=True)
        combined["solution_id"] = combined["solution_id"].astype(str).str.strip()
        combined = combined[
            combined["solution_id"].notna()
            & (combined["solution_id"] != "")
            & (combined["solution_id"] != "nan")
        ]
        deduped = combined.groupby("solution_id", as_index=False, sort=False).first()
        if "oxidizer_type" in deduped.columns:
            deduped["oxidizer_type"] = deduped["oxidizer_type"].apply(_normalize_oxidizer_label)
        else:
            deduped["oxidizer_type"] = "None"
        if "oxidizer_concentration" in deduped.columns:
            deduped["oxidizer_concentration"] = pd.to_numeric(
                deduped["oxidizer_concentration"], errors="coerce"
            ).fillna(0.0)
        else:
            deduped["oxidizer_concentration"] = 0.0
        return deduped.reset_index(drop=True), echem

    @staticmethod
    def _harmonize_electrochemical_records(
        echem: pd.DataFrame,
        materials: pd.DataFrame,
        solutions: pd.DataFrame,
        *,
        filter_to_core_metals: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if echem.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        df = echem.copy()
        rejected_frames: list[pd.DataFrame] = []
        raw_metal = _coalesce_column(df, ["metal", "symbol", "material_id"])
        df["metal"] = raw_metal.astype(str).str.strip()
        invalid_metal_mask = raw_metal.isna() | df["metal"].isin({"", "nan", "None"})
        if invalid_metal_mask.any():
            rej = df.loc[invalid_metal_mask].copy()
            rej["rejection_reason"] = "missing_metal_identifier"
            rejected_frames.append(rej)
            df = df.loc[~invalid_metal_mask].copy()

        if filter_to_core_metals:
            non_core_mask = ~df["metal"].isin(CORE_METALS)
            if non_core_mask.any():
                rej = df.loc[non_core_mask].copy()
                rej["rejection_reason"] = "filtered_non_core_metal"
                rejected_frames.append(rej)
                df = df.loc[~non_core_mask].copy()

        if df.empty:
            rejected_all = pd.concat(rejected_frames, ignore_index=True) if rejected_frames else pd.DataFrame()
            return pd.DataFrame(), pd.DataFrame(), rejected_all

        if "record_id" not in df.columns:
            df["record_id"] = [f"REC_{i + 1:05d}" for i in range(len(df))]
        else:
            missing_id = df["record_id"].isna() | (df["record_id"].astype(str).str.strip() == "")
            df.loc[missing_id, "record_id"] = [
                f"REC_AUTO_{i + 1:05d}" for i in range(int(missing_id.sum()))
            ]

        # Join all key Solution formulation attributes if missing on echem rows
        sol_lookup = solutions.set_index("solution_id") if not solutions.empty else pd.DataFrame()
        for col in (
            "pH",
            "oxidizer_type",
            "oxidizer_concentration",
            "acid_type",
            "acid_concentration",
            "inhibitor_type",
            "inhibitor_concentration",
            "chelating_agent_type",
            "chelating_agent_concentration",
            "ORP_mV",
            "solution_name",
        ):
            if col not in df.columns:
                df[col] = np.nan
            if not sol_lookup.empty and col in sol_lookup.columns:
                mapped = df["solution_id"].map(sol_lookup[col])
                df[col] = df[col].where(df[col].notna(), mapped)

        # Synchronize canonical material properties (including valence and equivalent_weight_g_eq) onto echem records
        if not materials.empty:
            mat_lookup = materials.set_index("metal")
            for col in ("valence", "equivalent_weight_g_eq", "density_g_cm3", "molar_mass_g_mol"):
                if col in mat_lookup.columns:
                    df[col] = df["metal"].map(mat_lookup[col])

        if "pH_effective" in df.columns:
            df["pH"] = pd.to_numeric(df["pH"], errors="coerce").fillna(
                pd.to_numeric(df["pH_effective"], errors="coerce")
            )
        else:
            df["pH"] = pd.to_numeric(df["pH"], errors="coerce")
        df["pH"] = df["pH"].fillna(7.0)

        df["temperature_C"] = pd.to_numeric(
            _coalesce_column(df, ["temperature_C", "temperature"]), errors="coerce"
        ).fillna(25.0)

        df["oxidizer_type"] = df["oxidizer_type"].apply(_normalize_oxidizer_label)
        df["oxidizer_concentration"] = pd.to_numeric(
            df["oxidizer_concentration"], errors="coerce"
        ).fillna(0.0)
        df.loc[df["oxidizer_type"] == "None", "oxidizer_concentration"] = 0.0

        # Normalize Ecorr to V vs. SHE
        raw_ecorr_orig = pd.to_numeric(
            _coalesce_column(df, ["Ecorr_V_original", "Ecorr", "E_corr"]), errors="coerce"
        )
        raw_ecorr_she = (
            pd.to_numeric(df["Ecorr_V_SHE"], errors="coerce")
            if "Ecorr_V_SHE" in df.columns
            else pd.Series([np.nan] * len(df), index=df.index)
        )
        ref_col = (
            df["reference_electrode"]
            if "reference_electrode" in df.columns
            else pd.Series(["SHE"] * len(df), index=df.index)
        )
        ecorr_unit_col = (
            df["Ecorr_unit"]
            if "Ecorr_unit" in df.columns
            else pd.Series(["V"] * len(df), index=df.index)
        )

        converted_ecorr: list[float] = []
        for idx in df.index:
            orig_val = raw_ecorr_orig.loc[idx]
            she_val = raw_ecorr_she.loc[idx]
            ref_val = ref_col.loc[idx]
            unit_val = ecorr_unit_col.loc[idx]
            ph_val = df.loc[idx, "pH"]

            if (
                not pd.isna(orig_val)
                and ref_val is not None
                and not pd.isna(ref_val)
                and str(ref_val).strip()
            ):
                try:
                    converted_ecorr.append(
                        convert_potential_to_she(
                            float(orig_val),
                            str(ref_val),
                            unit=str(unit_val) if not pd.isna(unit_val) else "V",
                            pH=float(ph_val),
                        )
                    )
                    continue
                except ValueError:
                    pass
            if not pd.isna(she_val):
                converted_ecorr.append(float(she_val))
            else:
                converted_ecorr.append(float("nan"))

        df["Ecorr_V_SHE"] = converted_ecorr

        # Normalize icorr to uA/cm2 without double-scaling already-normalized icorr_uA_cm2 values
        pre_norm_icorr = (
            pd.to_numeric(df["icorr_uA_cm2"], errors="coerce")
            if "icorr_uA_cm2" in df.columns
            else pd.Series([np.nan] * len(df), index=df.index)
        )
        raw_icorr = pd.to_numeric(
            _coalesce_column(df, ["icorr", "i_corr"]), errors="coerce"
        )
        icorr_unit_col = (
            df["icorr_unit"]
            if "icorr_unit" in df.columns
            else pd.Series(["uA/cm2"] * len(df), index=df.index)
        )
        raw_log_icorr = (
            pd.to_numeric(df["log_icorr"], errors="coerce")
            if "log_icorr" in df.columns
            else pd.Series([np.nan] * len(df), index=df.index)
        )
        log_icorr_unit_col = (
            df["log_icorr_unit"]
            if "log_icorr_unit" in df.columns
            else pd.Series(["A/cm2"] * len(df), index=df.index)
        )

        normalized_icorr: list[float] = []
        for idx in df.index:
            pre_val = pre_norm_icorr.loc[idx]
            raw_val = raw_icorr.loc[idx]
            ic_unit = icorr_unit_col.loc[idx]
            if not pd.isna(pre_val):
                norm_val = float(pre_val)
            elif not pd.isna(raw_val):
                norm_val = normalize_current_density_to_uA_cm2(
                    float(raw_val),
                    str(ic_unit) if not pd.isna(ic_unit) else "uA/cm2",
                )
            elif not pd.isna(raw_log_icorr.loc[idx]):
                lv = float(raw_log_icorr.loc[idx])
                log_unit = (
                    str(log_icorr_unit_col.loc[idx])
                    if not pd.isna(log_icorr_unit_col.loc[idx])
                    else "A/cm2"
                )
                norm_val = normalize_current_density_to_uA_cm2(10.0**lv, log_unit)
            else:
                norm_val = float("nan")
            normalized_icorr.append(
                norm_val if (not math.isnan(norm_val) and norm_val > 0.0) else float("nan")
            )

        df["icorr_uA_cm2"] = normalized_icorr

        # Normalize Tafel slopes to positive mV/dec within physical bounds [20, 500] before Stern-Geary derivation
        df["beta_a_mV_dec"] = _normalize_slope_series(
            df, "beta_a_mV_dec", "beta_a", "beta_a_unit"
        )
        df["beta_c_mV_dec"] = _normalize_slope_series(
            df, "beta_c_mV_dec", "beta_c", "beta_c_unit"
        )

        # Track provenance of icorr and physically derive missing icorr via Faraday (CR) or Stern-Geary (Rp)
        icorr_sources: list[str] = [
            "measured" if (not pd.isna(v) and v > 0.0) else "missing"
            for v in df["icorr_uA_cm2"]
        ]
        cr_rep_vals = (
            pd.to_numeric(df["CR_reported_value"], errors="coerce")
            if "CR_reported_value" in df.columns
            else pd.Series([np.nan] * len(df), index=df.index)
        )
        cr_rep_units = (
            df["CR_reported_unit"]
            if "CR_reported_unit" in df.columns
            else pd.Series([np.nan] * len(df), index=df.index)
        )
        cr_calc_vals = (
            pd.to_numeric(df["CR_calculated_nm_min"], errors="coerce")
            if "CR_calculated_nm_min" in df.columns
            else pd.Series([np.nan] * len(df), index=df.index)
        )
        rp_vals = (
            pd.to_numeric(df["Rp_ohm_cm2"], errors="coerce")
            if "Rp_ohm_cm2" in df.columns
            else pd.Series([np.nan] * len(df), index=df.index)
        )
        rp_units = (
            df["Rp_unit"]
            if "Rp_unit" in df.columns
            else pd.Series(["ohm·cm2"] * len(df), index=df.index)
        )

        derived_icorr = list(df["icorr_uA_cm2"])
        for pos, idx in enumerate(df.index):
            if icorr_sources[pos] == "measured":
                continue
            m_val = df.loc[idx, "molar_mass_g_mol"] if "molar_mass_g_mol" in df.columns else np.nan
            rho_val = df.loc[idx, "density_g_cm3"] if "density_g_cm3" in df.columns else np.nan
            z_val = df.loc[idx, "valence"] if "valence" in df.columns else np.nan

            est_cr = float("nan")
            if not pd.isna(cr_rep_vals.loc[idx]) and not pd.isna(cr_rep_units.loc[idx]):
                est_cr = derive_icorr_from_cr(
                    float(cr_rep_vals.loc[idx]),
                    str(cr_rep_units.loc[idx]),
                    molar_mass_g_mol=m_val,
                    density_g_cm3=rho_val,
                    valence=z_val,
                )
            elif not pd.isna(cr_calc_vals.loc[idx]):
                est_cr = derive_icorr_from_cr(
                    float(cr_calc_vals.loc[idx]),
                    "nm/min",
                    molar_mass_g_mol=m_val,
                    density_g_cm3=rho_val,
                    valence=z_val,
                )
            if math.isfinite(est_cr) and (ICORR_MIN_UA_CM2 <= est_cr <= ICORR_MAX_UA_CM2):
                derived_icorr[pos] = est_cr
                icorr_sources[pos] = "derived_from_CR"
                continue

            if not pd.isna(rp_vals.loc[idx]):
                est_rp = derive_icorr_from_rp(
                    float(rp_vals.loc[idx]),
                    str(rp_units.loc[idx]) if not pd.isna(rp_units.loc[idx]) else "ohm·cm2",
                    beta_a_mV_dec=df.loc[idx, "beta_a_mV_dec"],
                    beta_c_mV_dec=df.loc[idx, "beta_c_mV_dec"],
                )
                if math.isfinite(est_rp) and (ICORR_MIN_UA_CM2 <= est_rp <= ICORR_MAX_UA_CM2):
                    derived_icorr[pos] = est_rp
                    icorr_sources[pos] = "derived_from_Rp"

        df["icorr_uA_cm2"] = derived_icorr
        df["icorr_source"] = icorr_sources

        # Physical bounds screening on Ecorr and icorr
        ecorr_present = df["Ecorr_V_SHE"].notna()
        ecorr_oob = ecorr_present & (
            (df["Ecorr_V_SHE"] < ECORR_MIN_V_SHE) | (df["Ecorr_V_SHE"] > ECORR_MAX_V_SHE)
        )
        icorr_present = df["icorr_uA_cm2"].notna() & (df["icorr_uA_cm2"] > 0.0)
        icorr_oob = icorr_present & (
            (df["icorr_uA_cm2"] < ICORR_MIN_UA_CM2) | (df["icorr_uA_cm2"] > ICORR_MAX_UA_CM2)
        )
        neither_valid = (~ecorr_present) & (~icorr_present)
        reject_mask = ecorr_oob | icorr_oob | neither_valid
        if reject_mask.any():
            rej = df.loc[reject_mask].copy()
            reasons: list[str] = []
            for idx in rej.index:
                parts: list[str] = []
                if bool(ecorr_oob.loc[idx]):
                    parts.append("Ecorr_out_of_physical_bounds")
                elif not bool(ecorr_present.loc[idx]):
                    parts.append("missing_or_unconvertible_Ecorr")
                if bool(icorr_oob.loc[idx]):
                    parts.append("icorr_out_of_physical_bounds")
                elif not bool(icorr_present.loc[idx]):
                    parts.append("missing_or_nonpositive_icorr")
                reasons.append(";".join(parts))
            rej["rejection_reason"] = reasons
            rejected_frames.append(rej)
            df = df.loc[~reject_mask].copy()

        rejected_all = (
            pd.concat(rejected_frames, ignore_index=True)
            if rejected_frames
            else pd.DataFrame(columns=["record_id", "metal", "solution_id", "rejection_reason"])
        )
        if df.empty:
            return df, pd.DataFrame(), rejected_all

        # Compute log10(icorr) where icorr is valid and positive
        valid_icorr_mask = df["icorr_uA_cm2"].notna() & (df["icorr_uA_cm2"] > 0.0)
        df["log10_icorr_uA_cm2"] = np.where(
            valid_icorr_mask,
            np.log10(np.where(valid_icorr_mask, df["icorr_uA_cm2"], 1.0)),
            np.nan,
        )
        df["has_complete_pair"] = df["Ecorr_V_SHE"].notna() & valid_icorr_mask

        # Pre-compute base quality and imputation flags for deduplication priority
        ba_missing = df["beta_a_mV_dec"].isna()
        bc_missing = df["beta_c_mV_dec"].isna()
        df["beta_a_imputed"] = ba_missing
        df["beta_c_imputed"] = bc_missing
        df["is_imputed"] = ba_missing | bc_missing

        base_quality = pd.to_numeric(
            _coalesce_column(df, ["quality_score", "data_quality_score"], default=1.0),
            errors="coerce",
        ).fillna(1.0)
        penalty = np.where(df["is_imputed"], IMPUTATION_QUALITY_PENALTY, 0.0)
        penalty = penalty + np.where(
            df["icorr_source"] == "derived_from_CR", CR_DERIVED_QUALITY_PENALTY, 0.0
        )
        penalty = penalty + np.where(
            df["icorr_source"] == "derived_from_Rp", RP_DERIVED_QUALITY_PENALTY, 0.0
        )
        q_score = np.clip(base_quality - penalty, 0.05, 1.0)
        if "pH_source" in df.columns:
            inferred_neutral_mask = df["pH_source"].astype(str).str.strip().eq("inferred_neutral")
            q_score = np.where(
                inferred_neutral_mask,
                np.clip(q_score * INFERRED_NEUTRAL_PH_WEIGHT_FACTOR, 0.05, 1.0),
                q_score,
            )
        df["quality_score"] = q_score

        # Split complete pairs vs. partial-target records and deduplicate by physical content (not record_id)
        complete_df = df.loc[df["has_complete_pair"]].copy()
        partial_df = df.loc[~df["has_complete_pair"]].copy()

        if not complete_df.empty:
            complete_df = (
                complete_df.sort_values(
                    by=["quality_score", "is_imputed"],
                    ascending=[False, True],
                    kind="mergesort",
                )
                .drop_duplicates(
                    subset=["metal", "solution_id", "Ecorr_V_SHE", "icorr_uA_cm2"],
                    keep="first",
                )
                .sort_index()
                .copy()
            )
        if not partial_df.empty:
            partial_df = (
                partial_df.sort_values(
                    by=["quality_score", "is_imputed"],
                    ascending=[False, True],
                    kind="mergesort",
                )
                .drop_duplicates(
                    subset=[
                        "metal",
                        "solution_id",
                        "Ecorr_V_SHE",
                        "icorr_uA_cm2",
                        "beta_a_mV_dec",
                        "beta_c_mV_dec",
                    ],
                    keep="first",
                )
                .sort_index()
                .copy()
            )

        # Impute missing beta_a and beta_c using same-metal, same-oxidizer medians from complete pairs (or 60 / 120 mV/dec defaults)
        if not complete_df.empty:
            ba_group_med = (
                complete_df.groupby(["metal", "oxidizer_type"])["beta_a_mV_dec"].median()
            )
            bc_group_med = (
                complete_df.groupby(["metal", "oxidizer_type"])["beta_c_mV_dec"].median()
            )
        else:
            ba_group_med = pd.Series(dtype=float)
            bc_group_med = pd.Series(dtype=float)

        def _apply_tafel_imputation(frame: pd.DataFrame) -> pd.DataFrame:
            if frame.empty:
                return frame
            keys = pd.MultiIndex.from_frame(frame[["metal", "oxidizer_type"]])
            mapped_ba = keys.map(ba_group_med).to_numpy(dtype=float) if not ba_group_med.empty else np.full(len(frame), np.nan)
            mapped_bc = keys.map(bc_group_med).to_numpy(dtype=float) if not bc_group_med.empty else np.full(len(frame), np.nan)
            frame["beta_a_mV_dec"] = (
                frame["beta_a_mV_dec"]
                .fillna(pd.Series(mapped_ba, index=frame.index))
                .fillna(DEFAULT_BETA_A_MV_DEC)
            )
            frame["beta_c_mV_dec"] = (
                frame["beta_c_mV_dec"]
                .fillna(pd.Series(mapped_bc, index=frame.index))
                .fillna(DEFAULT_BETA_C_MV_DEC)
            )
            return frame

        complete_df = _apply_tafel_imputation(complete_df)
        partial_df = _apply_tafel_imputation(partial_df)

        return (
            complete_df.reset_index(drop=True),
            partial_df.reset_index(drop=True),
            rejected_all.reset_index(drop=True),
        )

    @staticmethod
    def _build_galvanic_series(echem_df: pd.DataFrame) -> pd.DataFrame:
        if echem_df.empty:
            return pd.DataFrame(
                columns=[
                    "solution_id",
                    "solution_name",
                    "pH",
                    "oxidizer_type",
                    "oxidizer_concentration",
                    "temperature_C",
                    "metal",
                    "Ecorr_V_SHE",
                    "icorr_uA_cm2",
                    "beta_a_mV_dec",
                    "beta_c_mV_dec",
                    "galvanic_rank",
                    "metals_in_solution",
                    "relative_nobility",
                    "record_count",
                ]
            )

        grouped = (
            echem_df.groupby(["solution_id", "metal"], as_index=False)
            .agg(
                solution_name=("solution_name", "first"),
                pH=("pH", "median"),
                oxidizer_type=("oxidizer_type", "first"),
                oxidizer_concentration=("oxidizer_concentration", "median"),
                temperature_C=("temperature_C", "median"),
                Ecorr_V_SHE=("Ecorr_V_SHE", "median"),
                icorr_uA_cm2=("icorr_uA_cm2", "median"),
                beta_a_mV_dec=("beta_a_mV_dec", "median"),
                beta_c_mV_dec=("beta_c_mV_dec", "median"),
                record_count=("record_id", "count"),
            )
            .sort_values(["solution_id", "Ecorr_V_SHE", "metal"], ascending=[True, True, True])
            .reset_index(drop=True)
        )

        grouped["galvanic_rank"] = grouped.groupby("solution_id").cumcount() + 1
        grouped["metals_in_solution"] = grouped.groupby("solution_id")["metal"].transform("count")

        def _classify_nobility(row: pd.Series) -> str:
            n = int(row["metals_in_solution"])
            r = int(row["galvanic_rank"])
            if n <= 1:
                return "Isolated"
            if r == 1:
                return "Most Active (Anodic)"
            if r == n:
                return "Most Noble (Cathodic)"
            return "Intermediate"

        grouped["relative_nobility"] = grouped.apply(_classify_nobility, axis=1)
        return grouped

    @staticmethod
    def _build_data_dictionary(
        echem_df: pd.DataFrame,
        materials_df: pd.DataFrame,
        solutions_df: pd.DataFrame,
        galvanic_df: pd.DataFrame,
    ) -> pd.DataFrame:
        known_metadata: dict[str, tuple[str, str, str]] = {
            "record_id": ("identifier", "dimensionless", "Unique electrochemical measurement record identifier."),
            "material_id": ("identifier", "dimensionless", "Canonical Metal identifier matching symbol."),
            "metal": ("categorical", "dimensionless", "Conductive metallic element, compound, or alloy film identifier (e.g., Cu, Co, W, Ru, Ti, Ta, Al, Mo, TiN, Fe, Ni)."),
            "symbol": ("categorical", "dimensionless", "Chemical symbol of the Metal."),
            "material_name": ("text", "dimensionless", "Full English name of the Metal."),
            "category": ("categorical", "dimensionless", "Semiconductor film functional category (metal, barrier, liner, semiconductor)."),
            "material_class": ("categorical", "dimensionless", "High-level material classification (e.g., pure_metal, alloy, nitride, barrier)."),
            "components_json": ("text", "json", "Structured JSON breakdown of Solution chemical components and concentrations."),
            "tier": ("categorical", "dimensionless", "Material catalog tier (core semiconductor metal vs. extended material)."),
            "has_faraday_properties": ("boolean", "dimensionless", "True when density and equivalent weight are available for Faraday corrosion-rate conversion."),
            "sample_name": ("text", "dimensionless", "Original experimental sample or specimen designation reported in the source publication."),
            "metal_form": ("categorical", "dimensionless", "Physical morphology or deposition form of the Metal (thin film, PVD, ALD, bulk, foil, rod)."),
            "surface_preparation": ("text", "dimensionless", "Pre-exposure surface conditioning or polishing treatment (e.g., CMP, mechanical polish, annealed)."),
            "solution_id": ("identifier", "dimensionless", "Unique Solution formulation identifier."),
            "solution_name": ("text", "dimensionless", "Human-readable chemical formulation summary for the Solution."),
            "solution_description": ("text", "dimensionless", "Raw text description of the Solution composition from the literature source."),
            "solution_category": ("categorical", "dimensionless", "pH regime category of the Solution (acidic, neutral, alkaline, etc.)."),
            "materials": ("text", "dimensionless", "Semicolon-delimited list of Metals evaluated with this Solution."),
            "base_solvent": ("categorical", "dimensionless", "Primary liquid solvent (typically DI water)."),
            "pH": ("numeric", "pH units", "Hydrogen ion activity (-log10[H+]) of the Solution."),
            "pH_effective": ("numeric", "pH units", "Effective pH used for thermodynamic calculations (measured or inferred from acid/base molarity)."),
            "pH_source": ("categorical", "dimensionless", "Provenance of pH_effective (measured_echem, inferred_strong_acid, inferred_neutral, etc.)."),
            "temperature_C": ("numeric", "°C", "Operating temperature of the Exposure Condition in degrees Celsius."),
            "reference_electrode": ("categorical", "dimensionless", "Original experimental reference electrode before conversion to SHE."),
            "Ecorr_V_original": ("numeric", "V vs. ref", "Originally reported open-circuit Corrosion Potential prior to SHE conversion."),
            "Ecorr_unit": ("categorical", "unit string", "Original potential unit (V or mV) prior to conversion to V vs. SHE."),
            "Ecorr_V_SHE": ("numeric", "V vs. SHE", "Standardized Corrosion Potential (E_corr) converted to Volts vs. Standard Hydrogen Electrode."),
            "icorr_uA_cm2": ("numeric", "µA/cm²", "Standardized Corrosion Current Density (i_corr) at E_corr in microamperes per square centimeter."),
            "icorr_source": ("categorical", "dimensionless", "Provenance of icorr_uA_cm2 (measured, derived_from_CR via Faraday law, or derived_from_Rp via Stern-Geary)."),
            "has_complete_pair": ("boolean_flag", "boolean", "True when both Ecorr_V_SHE and icorr_uA_cm2 are present and valid for full polarization and lookup."),
            "icorr_unit": ("categorical", "unit string", "Original current density unit prior to normalization to µA/cm²."),
            "log_icorr": ("numeric", "log10(A/cm²)", "Raw base-10 logarithm of corrosion current density in A/cm² from literature compilation."),
            "log_icorr_unit": ("categorical", "unit string", "Current density unit underlying raw log_icorr (default A/cm²)."),
            "log10_icorr_uA_cm2": ("numeric", "log10(µA/cm²)", "Base-10 logarithm of standardized Corrosion Current Density i_corr in µA/cm²."),
            "beta_a_mV_dec": ("numeric", "mV/dec", "Strictly positive anodic Tafel Slope (beta_a) in millivolts per decade of current density."),
            "beta_c_mV_dec": ("numeric", "mV/dec", "Strictly positive cathodic Tafel Slope (beta_c) in millivolts per decade of current density."),
            "beta_a_unit": ("categorical", "unit string", "Original unit for anodic Tafel slope prior to mV/dec normalization."),
            "beta_c_unit": ("categorical", "unit string", "Original unit for cathodic Tafel slope prior to mV/dec normalization."),
            "beta_unit": ("categorical", "unit string", "Original unit for Tafel slopes prior to mV/dec normalization."),
            "tafel_slopes_general": ("text", "mV/dec", "Unassigned or general Tafel slope values extracted from literature text."),
            "beta_a_imputed": ("boolean_flag", "boolean", "True if anodic Tafel slope beta_a was missing and imputed via group median or 60 mV/dec fallback."),
            "beta_c_imputed": ("boolean_flag", "boolean", "True if cathodic Tafel slope beta_c was missing and imputed via group median or 120 mV/dec fallback."),
            "is_imputed": ("boolean_flag", "boolean", "Quality flag indicating whether either Tafel slope (beta_a or beta_c) was imputed."),
            "Rp_ohm_cm2": ("numeric", "Ω·cm²", "Linear polarization resistance (R_p) in ohm-square centimeters."),
            "Rp_unit": ("categorical", "unit string", "Reported physical unit for linear polarization resistance R_p."),
            "CR_reported_value": ("numeric", "reported unit", "Corrosion Rate value as originally reported in the source publication."),
            "CR_reported_unit": ("categorical", "unit string", "Physical unit for CR_reported_value (e.g., nm/min, mm/year, Å/min)."),
            "CR_calculated_nm_min": ("numeric", "nm/min", "Faraday-derived Corrosion Rate in nanometers per minute."),
            "measurement_method": ("text", "dimensionless", "Experimental electrochemical or analytical measurement techniques (potentiodynamic, EIS, LPR, ICP-MS)."),
            "scan_rate": ("text", "mV/s", "Potentiodynamic polarization potential sweep rate."),
            "immersion_time": ("text", "time unit", "Experimental immersion or pre-conditioning duration reported in the study."),
            "aeration_condition": ("categorical", "dimensionless", "Dissolved gas saturation state during testing (aerated, deaerated, N2/Ar purged)."),
            "reaction_type": ("categorical", "dimensionless", "Primary electrochemical process classification (corrosion, passivity, etching)."),
            "reactions": ("text", "dimensionless", "Half-cell reactions active under the Exposure Condition (HER, ORR, OER, metal dissolution)."),
            "source_type": ("categorical", "dimensionless", "Literature or benchmark source classification (journal, seed_benchmark)."),
            "source_reference": ("text", "DOI / ID", "Publication DOI, table row identifier, or benchmark reference string."),
            "source": ("text", "DOI / ID", "Aggregated publication DOI and table row provenance identifiers."),
            "data_extraction_method": ("categorical", "dimensionless", "Ingestion extraction pipeline method (table_column_mapped, regex, manual_curation)."),
            "quality_score": ("quality_metric", "score [0, 1]", "Data reliability score in [0, 1], automatically decremented when is_imputed is True."),
            "data_quality_score": ("quality_metric", "score [0, 1]", "Raw ingestion data quality score prior to imputation penalty."),
            "quality_notes": ("text", "dimensionless", "Automated ingestion audit notes regarding unit inference or missing metadata."),
            "notes": ("text", "dimensionless", "Curator annotations or supplementary notes."),
            "oxidizer_type": ("categorical", "dimensionless", "Primary oxidizing agent in the Solution (e.g., H2O2, HNO3, None)."),
            "oxidizer_concentration": ("numeric", "wt% or M", "Numerical concentration of the oxidizing agent (0.0 when oxidizer_type is None)."),
            "oxidizer_concentration_unit": ("categorical", "unit string", "Concentration unit for oxidizer_concentration (e.g., wt%, M, mM)."),
            "acid_type": ("categorical", "dimensionless", "Acid component in the Solution formulation (e.g., H2SO4, HCl, HF, HClO4, citric acid)."),
            "acid_concentration": ("numeric", "M or wt%", "Concentration of the acid component."),
            "acid_concentration_unit": ("categorical", "unit string", "Concentration unit for acid_concentration."),
            "base_type": ("categorical", "dimensionless", "Alkaline base component in the Solution formulation (e.g., KOH, NaOH, TMAH)."),
            "base_concentration": ("numeric", "M or wt%", "Concentration of the alkaline base component."),
            "base_concentration_unit": ("categorical", "unit string", "Concentration unit for base_concentration."),
            "HF_concentration_wt_pct": ("numeric", "wt%", "Hydrofluoric acid (HF) weight percent concentration in the Solution."),
            "chelating_agent_type": ("categorical", "dimensionless", "Complexing or chelating agent (e.g., citric acid, glycine, oxalic acid, EDTA)."),
            "chelating_agent_concentration": ("numeric", "mM or wt%", "Concentration of the complexing/chelating agent."),
            "chelating_agent_concentration_unit": ("categorical", "unit string", "Concentration unit for chelating_agent_concentration."),
            "inhibitor_type": ("categorical", "dimensionless", "Corrosion inhibitor chemical species (e.g., BTA)."),
            "inhibitor_concentration": ("numeric", "mM or wt%", "Concentration of the corrosion inhibitor."),
            "inhibitor_concentration_unit": ("categorical", "unit string", "Concentration unit for inhibitor_concentration."),
            "organic_solvent_type": ("categorical", "dimensionless", "Co-solvent organic species in the Solution (e.g., ethylene glycol, IPA)."),
            "organic_solvent_concentration": ("numeric", "wt% or mM", "Concentration of the organic co-solvent."),
            "supporting_salt_type": ("categorical", "dimensionless", "Supporting ionic salt species in the Solution (e.g., NaCl, Na2SO4, KNO3, KCl)."),
            "supporting_salt_concentration": ("numeric", "M or wt%", "Concentration of the supporting ionic salt."),
            "supporting_salt_concentration_unit": ("categorical", "unit string", "Concentration unit for supporting_salt_concentration."),
            "dissolved_oxygen_ppm": ("numeric", "ppm", "Dissolved oxygen concentration in parts per million."),
            "conductivity_mS_cm": ("numeric", "mS/cm", "Ionic electrical conductivity of the Solution in millisiemens per centimeter."),
            "ionic_strength_mol_L": ("numeric", "mol/L", "Total molar ionic strength of the Solution."),
            "ORP_mV": ("numeric", "mV", "Solution Oxidation-Reduction Potential in millivolts."),
            "deaerated": ("boolean_flag", "boolean", "True if the Solution was explicitly purged of dissolved oxygen prior to measurement."),
            "density_g_cm3": ("numeric", "g/cm³", "Bulk mass density of the Metal film."),
            "molar_mass_g_mol": ("numeric", "g/mol", "Atomic or molar mass of the Metal."),
            "valence": ("numeric", "e⁻/atom", "Dissolution Valence (z): electrons transferred per dissolved Metal atom in Faraday's law."),
            "equivalent_weight_g_eq": ("numeric", "g/eq", "Electrochemical equivalent weight (molar_mass_g_mol / valence)."),
            "standard_potential_V_SHE": ("numeric", "V vs. SHE", "Standard reduction potential E° of the Metal in V vs. SHE."),
            "melting_point_C": ("numeric", "°C", "Melting temperature of the Metal at 1 atm in degrees Celsius."),
            "electronegativity_pauling": ("numeric", "Pauling scale", "Pauling electronegativity of the Metal."),
            "work_function_eV": ("numeric", "eV", "Electronic work function of the Metal surface in electron-volts."),
            "oxide_type": ("text", "chemical formula", "Primary surface oxide formulas formed on the Metal."),
            "oxide_stability": ("categorical", "tier", "Qualitative thermodynamic stability tier of the native oxide (low, medium, high)."),
            "passivation_tendency": ("numeric", "index [0, 1]", "Dimensionless tendency of the Metal to form a self-limiting passive oxide film."),
            "pourbaix_stable_pH_min": ("numeric", "pH units", "Lower pH boundary of the Metal's passive oxide stability window."),
            "pourbaix_stable_pH_max": ("numeric", "pH units", "Upper pH boundary of the Metal's passive oxide stability window."),
            "crystal_structure": ("categorical", "lattice type", "Room-temperature crystal structure (FCC, BCC, HCP)."),
            "thermal_conductivity_W_mK": ("numeric", "W/(m·K)", "Bulk thermal conductivity of the Metal."),
            "electrical_resistivity_ohm_cm": ("numeric", "Ω·cm", "Bulk electrical resistivity of the Metal."),
            "common_corrosion_products": ("text", "chemical species", "Typical dissolved ionic or solid corrosion products."),
            "base_element": ("categorical", "dimensionless", "Primary metallic element symbol for elemental descriptor lookup."),
            "elem_atomic_number": ("numeric", "atomic number Z", "Atomic number (proton count) of the Metal element."),
            "elem_atomic_mass": ("numeric", "g/mol", "Standard atomic weight of the Metal element."),
            "elem_pauling_en": ("numeric", "Pauling scale", "Elemental Pauling electronegativity."),
            "elem_work_function_eV": ("numeric", "eV", "Polycrystalline elemental electronic work function in electron-volts."),
            "elem_standard_potential_V_SHE": ("numeric", "V vs. SHE", "Elemental standard reduction potential in V vs. SHE."),
            "elem_melting_point_K": ("numeric", "K", "Elemental melting temperature in Kelvin."),
            "elem_molar_volume_cm3": ("numeric", "cm³/mol", "Elemental molar volume in cubic centimeters per mole."),
            "mp_material_id": ("identifier", "Materials Project ID", "Materials Project crystal structure identifier."),
            "mp_work_function_eV": ("numeric", "eV", "DFT-computed surface work function from Materials Project in electron-volts."),
            "mp_surface_energy_J_m2": ("numeric", "J/m²", "DFT-computed cleavage/surface energy in Joules per square meter."),
            "mp_formation_energy_eV_atom": ("numeric", "eV/atom", "DFT-computed formation enthalpy per atom in electron-volts."),
            "mp_band_gap_eV": ("numeric", "eV", "Electronic band gap energy in electron-volts (0.0 eV for metallic conductors)."),
            "thermo_E_onset_V_SHE": ("numeric", "V vs. SHE", "Thermodynamic Pourbaix anodic dissolution or oxidation onset potential in V vs. SHE."),
            "thermo_corrosion_regime": ("categorical", "Pourbaix regime", "Thermodynamic Pourbaix stability regime (immunity, corrosion, passivation)."),
            "thermo_stable_phase": ("text", "chemical phase", "Thermodynamically favored aqueous ion or solid oxide phase at the Exposure Condition."),
            "thermo_source": ("categorical", "dimensionless", "Provenance of the Pourbaix thermodynamic assessment (e.g., mp_pourbaix)."),
            "mp_pourbaix_E_onset_V_SHE": ("numeric", "V vs. SHE", "Materials Project Pourbaix oxidation onset potential at pH_effective in V vs. SHE."),
            "mp_stable_phase_at_Ecorr": ("text", "chemical phase", "Materials Project Pourbaix stable phase evaluated at (pH_effective, Ecorr_V_SHE)."),
            "mp_corrosion_regime": ("categorical", "Pourbaix regime", "Materials Project Pourbaix stability classification at Ecorr_V_SHE (immunity, corrosion, passivation)."),
            "mp_delta_G_pbx_eV_atom": ("numeric", "eV/atom", "Pourbaix decomposition free energy (driving force above convex hull) at Ecorr_V_SHE in eV/atom."),
            "delta_E_overpotential_V": ("numeric", "V", "Anodic overpotential above thermodynamic oxidation onset (Ecorr_V_SHE - thermo_E_onset_V_SHE) in Volts."),
            "galvanic_rank": ("numeric", "integer rank (1..N)", "Monotonic Galvanic Series rank within a Solution (1 = most Active Metal, N = most Noble Metal)."),
            "metals_in_solution": ("numeric", "count", "Total distinct Metals evaluated in that Solution within the Galvanic Series."),
            "relative_nobility": ("categorical", "role label", "Qualitative galvanic role within the Solution (Most Active, Intermediate, Most Noble)."),
            "record_count": ("numeric", "count", "Number of harmonized experimental records aggregated for that (Solution, Metal) pair."),
        }

        rows: list[dict[str, Any]] = []
        table_map = [
            ("electrochemical_database", echem_df),
            ("material_properties", materials_df),
            ("solution_properties", solutions_df),
            ("galvanic_series", galvanic_df),
        ]

        for table_name, df in table_map:
            for col in df.columns:
                role, unit, desc = known_metadata.get(
                    col,
                    ("attribute", "dimensionless", f"Harmonized attribute '{col}' in {table_name}."),
                )
                rows.append(
                    {
                        "table_name": table_name,
                        "column_name": col,
                        "dtype": str(df[col].dtype),
                        "attribute_role": role,
                        "unit": unit,
                        "non_null_count": int(df[col].notna().sum()),
                        "description": desc,
                    }
                )
        return pd.DataFrame(rows)

    def render_data_dictionary_markdown(self) -> str:
        """Render the complete data dictionary as a human-readable Markdown document."""
        lines = [
            "# Harmonized Electrochemical Database — Data Dictionary",
            "",
            "All tables use standardized physical units: Corrosion Potentials ($E_{\\text{corr}}$) in **V vs. SHE**, "
            "Corrosion Current Densities ($i_{\\text{corr}}$) in **µA/cm²**, and Tafel Slopes ($\\beta_a, \\beta_c$) in "
            "strictly positive **mV/dec**.",
            "",
        ]
        if self.data_dictionary.empty:
            return "\n".join(lines)

        for table_name, group in self.data_dictionary.groupby("table_name", sort=False):
            lines.append(f"## Table: `{table_name}`")
            lines.append("")
            lines.append("| Column | Role | Unit | Type | Non-Null | Description |")
            lines.append("| :--- | :--- | :--- | :--- | ---: | :--- |")
            for _, row in group.iterrows():
                lines.append(
                    f"| `{row['column_name']}` | {row['attribute_role']} | {row['unit']} | "
                    f"`{row['dtype']}` | {row['non_null_count']} | {row['description']} |"
                )
            lines.append("")
        return "\n".join(lines)

    def lookup_exposure_condition(
        self,
        metal: str | ExposureCondition,
        *,
        pH: float = 7.0,
        oxidizer_type: str | None = None,
        oxidizer_concentration: float = 0.0,
        temperature_C: float = 25.0,
        top_k: int = 1,
    ) -> ExposureLookupResult:
        """Query the database for an exact or nearest-neighbor Exposure Condition.

        Filters by exact ``Metal`` and ``oxidizer_type`` category, then ranks
        candidate records by Min-Max normalized Euclidean distance across
        ``(pH, oxidizer_concentration, temperature_C)``.
        """
        if self.electrochemical_database.empty:
            raise ValueError(
                "Cannot lookup Exposure Condition on an empty electrochemical database."
            )

        if isinstance(metal, ExposureCondition):
            metal_norm = str(metal.metal).strip()
            ox_norm = _normalize_oxidizer_label(metal.oxidizer_type)
            ox_conc = 0.0 if ox_norm == "None" else float(metal.oxidizer_concentration)
            ph_query = float(metal.pH)
            temp_query = float(metal.temperature_C)
            sol_id_filter = metal.solution_id.strip() if metal.solution_id else None
        else:
            metal_norm = str(metal).strip()
            ox_norm = _normalize_oxidizer_label(oxidizer_type)
            ox_conc = 0.0 if ox_norm == "None" else float(oxidizer_concentration)
            ph_query = float(pH)
            temp_query = float(temperature_C)
            sol_id_filter = None

        query_cond = ExposureCondition(
            metal=metal_norm,
            pH=ph_query,
            oxidizer_type=ox_norm,
            oxidizer_concentration=ox_conc,
            temperature_C=temp_query,
            solution_id=sol_id_filter,
        )

        df = self.electrochemical_database
        metal_subset = df[df["metal"].str.lower() == metal_norm.lower()]
        if metal_subset.empty:
            raise ValueError(f"No records found for Metal {metal_norm!r}.")

        if sol_id_filter is not None:
            candidates = metal_subset[
                metal_subset["solution_id"].astype(str).str.strip() == sol_id_filter
            ].copy()
            if candidates.empty:
                raise ValueError(
                    f"No records found for Metal {metal_norm!r} in Solution {sol_id_filter!r}."
                )
        else:
            candidates = metal_subset[
                metal_subset["oxidizer_type"].str.lower() == ox_norm.lower()
            ].copy()
            if candidates.empty:
                raise ValueError(
                    f"No records found for Metal {metal_norm!r} with oxidizer_type {ox_norm!r}."
                )

        num_features = [
            ("pH", ph_query),
            ("oxidizer_concentration", ox_conc),
            ("temperature_C", temp_query),
        ]

        sq_dist = np.zeros(len(candidates), dtype=float)
        for col, q_val in num_features:
            col_vals = candidates[col].to_numpy(dtype=float)
            c_min = float(np.min(col_vals))
            c_max = float(np.max(col_vals))
            span = c_max - c_min
            if span > 1e-12:
                norm_delta = (col_vals - q_val) / span
            else:
                global_vals = df[col].to_numpy(dtype=float)
                g_span = float(np.max(global_vals) - np.min(global_vals))
                scale = g_span if g_span > 1e-12 else 1.0
                norm_delta = (col_vals - q_val) / scale
            sq_dist += norm_delta**2

        candidates["normalized_distance"] = np.sqrt(sq_dist)
        candidates = candidates.sort_values(
            ["normalized_distance", "quality_score", "record_id"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

        best = candidates.iloc[0]
        dist = float(best["normalized_distance"])
        matched_ph = float(best["pH"])
        matched_conc = float(best["oxidizer_concentration"])
        matched_temp = float(best["temperature_C"])

        diff_ph = matched_ph - ph_query
        diff_conc = matched_conc - ox_conc
        diff_temp = matched_temp - temp_query

        is_exact = (
            abs(diff_ph) <= 1e-9
            and abs(diff_conc) <= 1e-9
            and abs(diff_temp) <= 1e-9
        )

        condition_diffs = {
            "pH": diff_ph,
            "oxidizer_concentration": diff_conc,
            "temperature_C": diff_temp,
        }

        if is_exact:
            summary = (
                f"Exact match on {best['metal']} in {best['oxidizer_type']} "
                f"(pH={matched_ph:.2f}, conc={matched_conc:.3g}, T={matched_temp:.1f} °C)."
            )
        else:
            parts: list[str] = []
            if abs(diff_ph) > 1e-9:
                parts.append(
                    f"ΔpH = {diff_ph:+.2f} (matched {matched_ph:.2f} vs query {ph_query:.2f})"
                )
            if abs(diff_conc) > 1e-9:
                parts.append(
                    f"Δoxidizer_concentration = {diff_conc:+.3g} "
                    f"(matched {matched_conc:.3g} vs query {ox_conc:.3g})"
                )
            if abs(diff_temp) > 1e-9:
                parts.append(
                    f"Δtemperature_C = {diff_temp:+.1f} °C "
                    f"(matched {matched_temp:.1f} vs query {temp_query:.1f} °C)"
                )
            summary = "Nearest-neighbor match (" + "; ".join(parts) + f"; distance={dist:.4f})."

        matched_cond = ExposureCondition(
            metal=str(best["metal"]),
            pH=matched_ph,
            oxidizer_type=str(best["oxidizer_type"]),
            oxidizer_concentration=matched_conc,
            temperature_C=matched_temp,
            solution_id=str(best["solution_id"]),
            solution_name=str(best.get("solution_name", "")),
        )

        return ExposureLookupResult(
            record_id=str(best["record_id"]),
            solution_id=str(best["solution_id"]),
            metal=str(best["metal"]),
            oxidizer_type=str(best["oxidizer_type"]),
            Ecorr_V_SHE=float(best["Ecorr_V_SHE"]),
            icorr_uA_cm2=float(best["icorr_uA_cm2"]),
            beta_a_mV_dec=float(best["beta_a_mV_dec"]),
            beta_c_mV_dec=float(best["beta_c_mV_dec"]),
            is_imputed=bool(best["is_imputed"]),
            quality_score=float(best["quality_score"]),
            is_exact_match=is_exact,
            normalized_distance=0.0 if is_exact else dist,
            query_condition=query_cond,
            matched_condition=matched_cond,
            condition_differences=condition_diffs,
            mismatch_summary=summary,
            candidates=candidates.head(max(1, int(top_k))).copy(),
        )

    def get_solution_attributes(self, solution_id: str) -> dict[str, Any]:
        """Return normalized formulation attributes (pH, oxidizer_type, oxidizer_concentration) for a Solution ID."""
        sid = str(solution_id).strip()
        if not self.solution_properties.empty:
            match = self.solution_properties[
                self.solution_properties["solution_id"].astype(str).str.strip() == sid
            ]
            if not match.empty:
                row = match.iloc[0]
                return {
                    "solution_id": sid,
                    "pH": float(row["pH"]) if not pd.isna(row.get("pH")) else 7.0,
                    "oxidizer_type": _normalize_oxidizer_label(row.get("oxidizer_type")),
                    "oxidizer_concentration": (
                        float(row["oxidizer_concentration"])
                        if not pd.isna(row.get("oxidizer_concentration"))
                        else 0.0
                    ),
                }
        raise ValueError(f"Unknown solution_id: {solution_id!r}")

    @staticmethod
    def compute_raw_directory_fingerprint(raw_dir: str | Path = "data/raw") -> dict[str, Any]:
        """Compute a deterministic SHA-256 fingerprint of all CSV files under ``raw_dir``."""
        base_path = Path(raw_dir)
        files_info: dict[str, dict[str, Any]] = {}
        combined_hasher = hashlib.sha256()
        if base_path.exists():
            for csv_path in sorted(base_path.rglob("*.csv")):
                rel = str(csv_path.relative_to(base_path)).replace("\\", "/")
                data = csv_path.read_bytes()
                file_sha = hashlib.sha256(data).hexdigest()
                files_info[rel] = {
                    "sha256": file_sha,
                    "size_bytes": len(data),
                }
                combined_hasher.update(rel.encode("utf-8"))
                combined_hasher.update(b":")
                combined_hasher.update(file_sha.encode("utf-8"))
                combined_hasher.update(b"\n")
        return {
            "raw_dir": str(base_path),
            "combined_sha256": combined_hasher.hexdigest(),
            "file_count": len(files_info),
            "files": files_info,
        }

    def export_to_directory(self, output_dir: str | Path = "data/processed") -> dict[str, Path]:
        """Save all harmonized tables and the data dictionary to ``output_dir`` as CSV/Markdown."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        paths = {
            "electrochemical_database": out_path / "electrochemical_database.csv",
            "material_properties": out_path / "material_properties.csv",
            "solution_properties": out_path / "solution_properties.csv",
            "galvanic_series": out_path / "galvanic_series.csv",
            "data_dictionary": out_path / "data_dictionary.csv",
            "data_dictionary_markdown": out_path / "DATA_DICTIONARY.md",
            "rejected_records": out_path / "rejected_rows.csv",
            "partial_target_records": out_path / "partial_target_records.csv",
        }

        self.electrochemical_database.to_csv(paths["electrochemical_database"], index=False)
        self.material_properties.to_csv(paths["material_properties"], index=False)
        self.solution_properties.to_csv(paths["solution_properties"], index=False)
        self.galvanic_series.to_csv(paths["galvanic_series"], index=False)
        self.data_dictionary.to_csv(paths["data_dictionary"], index=False)
        paths["data_dictionary_markdown"].write_text(
            self.render_data_dictionary_markdown(), encoding="utf-8"
        )
        self.rejected_records.to_csv(paths["rejected_records"], index=False)
        self.partial_target_records.to_csv(paths["partial_target_records"], index=False)
        return paths

    def export_to_parquet(self, output_dir: str | Path = "data/processed") -> dict[str, Path]:
        """Save all harmonized tables to ``output_dir`` as Parquet for fast startup loading."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
            clean = df.copy()
            for col in clean.columns:
                if clean[col].dtype == object:
                    # Coerce mixed object columns to strings where non-null to satisfy Arrow typing
                    non_null = clean[col].dropna()
                    if not non_null.empty and len({type(v) for v in non_null}) > 1:
                        clean[col] = clean[col].where(clean[col].isna(), clean[col].astype(str))
            return clean

        paths = {
            "electrochemical_database": out_path / "electrochemical_database.parquet",
            "material_properties": out_path / "material_properties.parquet",
            "solution_properties": out_path / "solution_properties.parquet",
            "galvanic_series": out_path / "galvanic_series.parquet",
            "data_dictionary": out_path / "data_dictionary.parquet",
            "partial_target_records": out_path / "partial_target_records.parquet",
        }
        _sanitize_for_parquet(self.electrochemical_database).to_parquet(
            paths["electrochemical_database"], index=False
        )
        _sanitize_for_parquet(self.material_properties).to_parquet(
            paths["material_properties"], index=False
        )
        _sanitize_for_parquet(self.solution_properties).to_parquet(
            paths["solution_properties"], index=False
        )
        _sanitize_for_parquet(self.galvanic_series).to_parquet(
            paths["galvanic_series"], index=False
        )
        _sanitize_for_parquet(self.data_dictionary).to_parquet(
            paths["data_dictionary"], index=False
        )
        _sanitize_for_parquet(self.partial_target_records).to_parquet(
            paths["partial_target_records"], index=False
        )
        return paths

    @classmethod
    def from_processed_directory(cls, processed_dir: str | Path = "data/processed") -> HarmonizedDatabase:
        """Load prebuilt harmonized tables from Parquet (preferred) or CSV in ``processed_dir``."""
        base_path = Path(processed_dir)
        if not base_path.exists():
            raise FileNotFoundError(f"Processed artifact directory does not exist: {base_path}")

        def _load_table(stem: str, required: bool = True) -> pd.DataFrame:
            pq_path = base_path / f"{stem}.parquet"
            csv_path = base_path / f"{stem}.csv"
            if pq_path.exists():
                return pd.read_parquet(pq_path)
            if csv_path.exists():
                return pd.read_csv(csv_path, encoding="utf-8-sig")
            if required:
                raise FileNotFoundError(f"Missing processed table {stem}.parquet or {stem}.csv in {base_path}")
            return pd.DataFrame()

        echem = _load_table("electrochemical_database", required=True)
        materials = _load_table("material_properties", required=True)
        solutions = _load_table("solution_properties", required=True)
        galvanic = _load_table("galvanic_series", required=False)
        dictionary = _load_table("data_dictionary", required=False)
        partial_records = _load_table("partial_target_records", required=False)
        rejected_path = base_path / "rejected_rows.csv"
        rejected = pd.read_csv(rejected_path, encoding="utf-8-sig") if rejected_path.exists() else pd.DataFrame()

        return cls(
            electrochemical_database=echem,
            material_properties=materials,
            solution_properties=solutions,
            galvanic_series=galvanic,
            data_dictionary=dictionary,
            rejected_records=rejected,
            partial_target_records=partial_records,
        )
