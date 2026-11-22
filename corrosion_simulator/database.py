"""Database harmonization, unit normalization, Galvanic Series ranking, and Exposure Condition lookup."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

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
IMPUTATION_QUALITY_PENALTY: Final[float] = 0.15

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


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(c).lstrip("\ufeff").strip() for c in cleaned.columns]
    cleaned = cleaned.rename(columns=_RAW_COLUMN_RENAMES)
    drop_cols = [c for c in _DROPPED_INTERNAL_COLUMNS if c in cleaned.columns]
    if drop_cols:
        cleaned = cleaned.drop(columns=drop_cols)
    return cleaned


def _normalize_oxidizer_label(value: Any) -> str:
    if value is None or pd.isna(value):
        return "None"
    s = str(value).strip()
    if not s or s.lower() in ("none", "nan", "null", "n/a", "na"):
        return "None"
    return s


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
    value_candidates: list[str],
    specific_unit_col: str,
) -> list[float]:
    raw_slopes = pd.to_numeric(_coalesce_column(df, value_candidates), errors="coerce")
    unit_col = (
        df[specific_unit_col]
        if specific_unit_col in df.columns
        else (
            df["beta_unit"]
            if "beta_unit" in df.columns
            else pd.Series(["mV/dec"] * len(df), index=df.index)
        )
    )
    return [
        normalize_tafel_slope_to_mV_dec(
            raw_slopes.loc[idx],
            str(unit_col.loc[idx]) if not pd.isna(unit_col.loc[idx]) else "mV/dec",
        )
        for idx in df.index
    ]


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
        filter_to_core_metals: bool = True,
        supplement_missing_core_metals: bool = True,
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
                    parent_name = csv_path.parent.name
                    if parent_name in CORE_METALS:
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
        filter_to_core_metals: bool = True,
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
            prelim_valid = cls._harmonize_electrochemical_records(
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
        harmonized_echem = cls._harmonize_electrochemical_records(
            echem_with_sol,
            harmonized_materials,
            harmonized_solutions,
            filter_to_core_metals=filter_to_core_metals,
        )

        # 4. Build Galvanic Series ranking per Solution
        galvanic_series = cls._build_galvanic_series(harmonized_echem, harmonized_solutions)

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
        )

    @staticmethod
    def _harmonize_materials(
        mat_raw: pd.DataFrame,
        echem_raw: pd.DataFrame,
        *,
        filter_to_core_metals: bool,
    ) -> pd.DataFrame:
        seed_mat_df = pd.DataFrame(SEED_MATERIAL_PROPERTIES)
        frames: list[pd.DataFrame] = []

        if not mat_raw.empty:
            df = mat_raw.copy()
            df["metal"] = _coalesce_column(df, ["metal", "symbol", "material_id"])
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
                frames.append(sub)

        frames.append(seed_mat_df)
        combined = pd.concat(frames, ignore_index=True)
        combined["metal"] = combined["metal"].astype(str).str.strip()
        combined = combined[
            combined["metal"].notna() & (combined["metal"] != "") & (combined["metal"] != "nan")
        ]

        if filter_to_core_metals:
            combined = combined[combined["metal"].isin(CORE_METALS)]

        # Deduplicate by Metal, preferring first non-null attribute values
        deduped = combined.groupby("metal", as_index=False, sort=False).first()
        deduped["symbol"] = deduped["metal"]
        deduped["material_id"] = deduped["metal"]
        # Enforce canonical Dissolution Valence (z) from GLOSSARY.md and recompute equivalent_weight_g_eq
        deduped["valence"] = (
            deduped["metal"]
            .map(DEFAULT_DISSOLUTION_VALENCE)
            .fillna(deduped.get("valence", 2))
            .astype(int)
        )
        molar_mass = pd.to_numeric(deduped["molar_mass_g_mol"], errors="coerce")
        deduped["equivalent_weight_g_eq"] = (molar_mass / deduped["valence"]).round(4)
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
    ) -> pd.DataFrame:
        if echem.empty:
            return pd.DataFrame()

        df = echem.copy()
        df["metal"] = _coalesce_column(df, ["metal", "symbol", "material_id"]).astype(str).str.strip()
        if filter_to_core_metals:
            df = df[df["metal"].isin(CORE_METALS)].copy()

        if "record_id" not in df.columns:
            df["record_id"] = [f"REC_{i + 1:05d}" for i in range(len(df))]
        else:
            missing_id = df["record_id"].isna() | (df["record_id"].astype(str).str.strip() == "")
            df.loc[missing_id, "record_id"] = [
                f"REC_AUTO_{i + 1:05d}" for i in range(int(missing_id.sum()))
            ]

        # Join solution attributes if missing on echem rows
        sol_lookup = solutions.set_index("solution_id") if not solutions.empty else pd.DataFrame()
        for col in ("pH", "oxidizer_type", "oxidizer_concentration", "solution_name"):
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

        # Normalize icorr to uA/cm2
        raw_icorr = pd.to_numeric(
            _coalesce_column(df, ["icorr_uA_cm2", "icorr", "i_corr"]), errors="coerce"
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
            ic_val = raw_icorr.loc[idx]
            ic_unit = icorr_unit_col.loc[idx]
            if not pd.isna(ic_val):
                norm_val = normalize_current_density_to_uA_cm2(
                    float(ic_val),
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

        # Filter out records missing Ecorr or icorr
        df = df[
            df["Ecorr_V_SHE"].notna() & df["icorr_uA_cm2"].notna() & (df["icorr_uA_cm2"] > 0.0)
        ].copy()
        if df.empty:
            return df

        df["log10_icorr_uA_cm2"] = np.log10(df["icorr_uA_cm2"])

        # Normalize Tafel slopes to positive mV/dec
        df["beta_a_mV_dec"] = _normalize_slope_series(
            df, ["beta_a_mV_dec", "beta_a"], "beta_a_unit"
        )
        df["beta_c_mV_dec"] = _normalize_slope_series(
            df, ["beta_c_mV_dec", "beta_c"], "beta_c_unit"
        )

        # Deduplicate exact duplicate records before computing group medians
        df = df.drop_duplicates(
            subset=["record_id", "metal", "solution_id", "Ecorr_V_SHE", "icorr_uA_cm2"],
            keep="first",
        ).copy()

        # Impute missing beta_a and beta_c using same-metal, same-oxidizer medians (or 60 / 120 mV/dec defaults)
        ba_missing = df["beta_a_mV_dec"].isna()
        bc_missing = df["beta_c_mV_dec"].isna()
        df["beta_a_imputed"] = ba_missing
        df["beta_c_imputed"] = bc_missing
        df["is_imputed"] = ba_missing | bc_missing

        group_ba_medians = df.groupby(["metal", "oxidizer_type"])["beta_a_mV_dec"].transform(
            "median"
        )
        group_bc_medians = df.groupby(["metal", "oxidizer_type"])["beta_c_mV_dec"].transform(
            "median"
        )

        df["beta_a_mV_dec"] = (
            df["beta_a_mV_dec"].fillna(group_ba_medians).fillna(DEFAULT_BETA_A_MV_DEC)
        )
        df["beta_c_mV_dec"] = (
            df["beta_c_mV_dec"].fillna(group_bc_medians).fillna(DEFAULT_BETA_C_MV_DEC)
        )

        # Quality score calculation & imputation decrement
        base_quality = pd.to_numeric(
            _coalesce_column(df, ["quality_score", "data_quality_score"], default=1.0),
            errors="coerce",
        ).fillna(1.0)
        penalty = np.where(df["is_imputed"], IMPUTATION_QUALITY_PENALTY, 0.0)
        df["quality_score"] = np.clip(base_quality - penalty, 0.05, 1.0)

        return df.reset_index(drop=True)

    @staticmethod
    def _build_galvanic_series(
        echem_df: pd.DataFrame,
        solutions_df: pd.DataFrame,
    ) -> pd.DataFrame:
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
            "metal": ("categorical", "dimensionless", "Conductive metallic element or alloy film (Cu, Co, W, Ru, Ti, Ta, Al, Mo)."),
            "symbol": ("categorical", "dimensionless", "Chemical symbol of the Metal."),
            "material_name": ("text", "dimensionless", "Full English name of the Metal."),
            "category": ("categorical", "dimensionless", "Semiconductor film functional category (metal, barrier, liner)."),
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

    def export_to_directory(self, output_dir: str | Path = "data/processed") -> dict[str, Path]:
        """Save all harmonized tables and the data dictionary to ``output_dir``."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        paths = {
            "electrochemical_database": out_path / "electrochemical_database.csv",
            "material_properties": out_path / "material_properties.csv",
            "solution_properties": out_path / "solution_properties.csv",
            "galvanic_series": out_path / "galvanic_series.csv",
            "data_dictionary": out_path / "data_dictionary.csv",
            "data_dictionary_markdown": out_path / "DATA_DICTIONARY.md",
        }

        self.electrochemical_database.to_csv(paths["electrochemical_database"], index=False)
        self.material_properties.to_csv(paths["material_properties"], index=False)
        self.solution_properties.to_csv(paths["solution_properties"], index=False)
        self.galvanic_series.to_csv(paths["galvanic_series"], index=False)
        self.data_dictionary.to_csv(paths["data_dictionary"], index=False)
        paths["data_dictionary_markdown"].write_text(
            self.render_data_dictionary_markdown(), encoding="utf-8"
        )
        return paths
