import json
import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from corrosion_simulator.database import (
    AQUEOUS_SOLVENT_SUBCLASSES,
    CORE_METALS,
    INFERRED_REGIME_DEFAULT_WATER_WEIGHT_FACTOR,
    ORGANIC_SOLVENT_SUBCLASSES,
    SOLVENT_REGIMES,
    ExposureCondition,
    ExposureLookupResult,
    HarmonizedDatabase,
    classify_solvent_regime,
    classify_solvent_subclass,
    convert_potential_to_she,
    derive_icorr_from_cr,
    derive_icorr_from_rp,
    infer_water_content_and_regime,
    normalize_current_density_to_uA_cm2,
    normalize_tafel_slope_to_mV_dec,
)


class TestUnitAndReferenceElectrodeNormalization(unittest.TestCase):
    """Seam 1 slice 1: analytical verification of reference electrode and unit conversions."""

    def test_convert_potential_to_she_across_reference_electrodes(self) -> None:
        # SHE: 0.000 V offset
        self.assertAlmostEqual(convert_potential_to_she(0.150, "SHE"), 0.150, places=6)
        # Ag/AgCl: +0.197 V offset -> 0.379 + 0.197 = 0.576 V vs. SHE
        self.assertAlmostEqual(convert_potential_to_she(0.379, "Ag/AgCl"), 0.576, places=6)
        # SCE: +0.241 V offset -> 0.001 + 0.241 = 0.242 V vs. SHE
        self.assertAlmostEqual(convert_potential_to_she(0.001, "SCE"), 0.242, places=6)
        # MSE (Hg/Hg2SO4): +0.640 V offset -> -0.500 + 0.640 = 0.140 V vs. SHE
        self.assertAlmostEqual(convert_potential_to_she(-0.500, "MSE"), 0.140, places=6)
        self.assertAlmostEqual(convert_potential_to_she(-0.500, "Hg/Hg2SO4"), 0.140, places=6)
        # Hg/HgO: +0.098 V offset -> -0.100 + 0.098 = -0.002 V vs. SHE
        self.assertAlmostEqual(convert_potential_to_she(-0.100, "Hg/HgO"), -0.002, places=6)
        # RHE at pH 10.0: E_SHE = E_RHE - 0.05916 * pH -> 0.800 - 0.5916 = 0.2084 V vs. SHE
        self.assertAlmostEqual(
            convert_potential_to_she(0.800, "RHE", pH=10.0), 0.2084, places=6
        )
        # mV input unit with SCE: -178 mV = -0.178 V -> -0.178 + 0.241 = 0.063 V vs. SHE
        self.assertAlmostEqual(
            convert_potential_to_she(-178.0, "SCE", unit="mV"), 0.063, places=6
        )

    def test_normalize_current_density_to_uA_cm2(self) -> None:
        self.assertAlmostEqual(normalize_current_density_to_uA_cm2(2.3, "uA/cm2"), 2.3, places=6)
        self.assertAlmostEqual(normalize_current_density_to_uA_cm2(2.3, "µA/cm²"), 2.3, places=6)
        self.assertAlmostEqual(normalize_current_density_to_uA_cm2(0.025, "mA/cm2"), 25.0, places=6)
        self.assertAlmostEqual(normalize_current_density_to_uA_cm2(1.5e-5, "A/cm2"), 15.0, places=6)
        self.assertAlmostEqual(normalize_current_density_to_uA_cm2(500.0, "nA/cm2"), 0.5, places=6)
        self.assertAlmostEqual(normalize_current_density_to_uA_cm2(0.12, "A/m2"), 12.0, places=6)

    def test_normalize_tafel_slope_to_positive_mV_dec(self) -> None:
        # Positive mV/dec remains unchanged
        self.assertAlmostEqual(normalize_tafel_slope_to_mV_dec(65.0, "mV/dec"), 65.0, places=6)
        # Negative cathodic Tafel slope becomes strictly positive mV/dec
        self.assertAlmostEqual(normalize_tafel_slope_to_mV_dec(-118.5, "mV/dec"), 118.5, places=6)
        # V/dec converted to positive mV/dec
        self.assertAlmostEqual(normalize_tafel_slope_to_mV_dec(-0.120, "V/dec"), 120.0, places=6)
        self.assertAlmostEqual(normalize_tafel_slope_to_mV_dec(0.060, "V/dec"), 60.0, places=6)


class TestTableHarmonizationFilteringAndImputation(unittest.TestCase):
    """Seam 1 slice 2: deduplication, missing Ecorr/icorr filtering, and Tafel slope imputation."""

    def test_harmonize_deduplicates_filters_and_imputes_slopes(self) -> None:
        raw_echem = pd.DataFrame(
            [
                # Complete Cu record 1 in H2O2 (Ag/AgCl -> +0.197 V)
                {
                    "record_id": "R1",
                    "metal": "Cu",
                    "solution_id": "SOL_A",
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 1.0,
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "reference_electrode": "Ag/AgCl",
                    "Ecorr_V_original": 0.083,  # -> 0.280 V vs. SHE
                    "icorr_uA_cm2": 2.3,
                    "beta_a_mV_dec": 50.0,
                    "beta_c_mV_dec": -110.0,  # -> +110.0 mV/dec
                    "data_quality_score": 0.90,
                },
                # Complete Cu record 2 in H2O2 (SCE -> +0.241 V)
                {
                    "record_id": "R2",
                    "metal": "Cu",
                    "solution_id": "SOL_A",
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 1.0,
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "reference_electrode": "SCE",
                    "Ecorr_V_original": 0.059,  # -> 0.300 V vs. SHE
                    "icorr_uA_cm2": 3.1,
                    "beta_a_mV_dec": 70.0,
                    "beta_c_mV_dec": 150.0,
                    "data_quality_score": 0.90,
                },
                # Cu record in H2O2 missing beta_a and beta_c -> should impute same-metal/same-oxidizer medians (60.0, 130.0)
                {
                    "record_id": "R3",
                    "metal": "Cu",
                    "solution_id": "SOL_A",
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 2.0,
                    "pH": 4.5,
                    "temperature_C": 30.0,
                    "reference_electrode": "SHE",
                    "Ecorr_V_original": 0.310,
                    "icorr_uA_cm2": 4.5,
                    "beta_a_mV_dec": None,
                    "beta_c_mV_dec": None,
                    "data_quality_score": 0.90,
                },
                # W record in HNO3 with no other (W, HNO3) slopes -> fallback to 60.0 and 120.0 mV/dec
                {
                    "record_id": "R4",
                    "metal": "W",
                    "solution_id": "SOL_B",
                    "oxidizer_type": "HNO3",
                    "oxidizer_concentration": 0.5,
                    "pH": 2.0,
                    "temperature_C": 25.0,
                    "reference_electrode": "MSE",
                    "Ecorr_V_original": -0.340,  # -> +0.300 V vs. SHE
                    "icorr": 0.0012,
                    "icorr_unit": "mA/cm2",  # -> 1.2 uA/cm2
                    "beta_a_mV_dec": None,
                    "beta_c_mV_dec": None,
                    "data_quality_score": 0.85,
                },
                # Record missing Ecorr -> must be excluded
                {
                    "record_id": "R_MISSING_ECORR",
                    "metal": "Co",
                    "solution_id": "SOL_A",
                    "oxidizer_type": "H2O2",
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "reference_electrode": "SCE",
                    "Ecorr_V_original": None,
                    "icorr_uA_cm2": 10.0,
                    "beta_a_mV_dec": 55.0,
                    "beta_c_mV_dec": 120.0,
                    "data_quality_score": 0.80,
                },
                # Record missing icorr -> must be excluded
                {
                    "record_id": "R_MISSING_ICORR",
                    "metal": "Co",
                    "solution_id": "SOL_A",
                    "oxidizer_type": "H2O2",
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "reference_electrode": "SCE",
                    "Ecorr_V_original": -0.100,
                    "icorr_uA_cm2": None,
                    "beta_a_mV_dec": 55.0,
                    "beta_c_mV_dec": 120.0,
                    "data_quality_score": 0.80,
                },
            ]
        )

        raw_materials = pd.DataFrame(
            [
                {"symbol": "Cu", "material_name": "Copper", "density_g_cm3": 8.96, "valence": 2},
                {"symbol": "Cu", "material_name": "Copper duplicate", "density_g_cm3": 8.96, "valence": 2},
                {"symbol": "W", "material_name": "Tungsten", "density_g_cm3": 19.25, "valence": 6},
            ]
        )

        raw_solutions = pd.DataFrame(
            [
                {"solution_id": "SOL_A", "solution_name": "1 wt% H2O2", "pH": 4.0, "oxidizer_type": "H2O2"},
                {"solution_id": "SOL_A", "solution_name": "1 wt% H2O2 dup", "pH": 4.0, "oxidizer_type": "H2O2"},
                {"solution_id": "SOL_B", "solution_name": "0.5 wt% HNO3", "pH": 2.0, "oxidizer_type": "HNO3"},
            ]
        )

        db = HarmonizedDatabase.from_raw_tables(
            raw_echem,
            material_df=raw_materials,
            solution_df=raw_solutions,
            supplement_missing_core_metals=False,
        )

        echem = db.electrochemical_database.set_index("record_id")
        self.assertEqual(set(echem.index), {"R1", "R2", "R3", "R4"})

        # Check R1 normalizations
        self.assertAlmostEqual(echem.loc["R1", "Ecorr_V_SHE"], 0.280, places=6)
        self.assertAlmostEqual(echem.loc["R1", "beta_c_mV_dec"], 110.0, places=6)
        self.assertFalse(bool(echem.loc["R1", "is_imputed"]))
        self.assertAlmostEqual(echem.loc["R1", "quality_score"], 0.90, places=6)

        # Check R3 same-metal/same-oxidizer median imputation: median([50, 70]) = 60.0, median([110, 150]) = 130.0
        self.assertAlmostEqual(echem.loc["R3", "beta_a_mV_dec"], 60.0, places=6)
        self.assertAlmostEqual(echem.loc["R3", "beta_c_mV_dec"], 130.0, places=6)
        self.assertTrue(bool(echem.loc["R3", "is_imputed"]))
        self.assertLess(echem.loc["R3", "quality_score"], 0.90)

        # Check R4 physical fallback imputation (60 and 120 mV/dec) and current unit conversion (0.0012 mA/cm2 -> 1.2 uA/cm2)
        self.assertAlmostEqual(echem.loc["R4", "Ecorr_V_SHE"], 0.300, places=6)
        self.assertAlmostEqual(echem.loc["R4", "icorr_uA_cm2"], 1.2, places=6)
        self.assertAlmostEqual(echem.loc["R4", "beta_a_mV_dec"], 60.0, places=6)
        self.assertAlmostEqual(echem.loc["R4", "beta_c_mV_dec"], 120.0, places=6)
        self.assertTrue(bool(echem.loc["R4", "is_imputed"]))
        self.assertLess(echem.loc["R4", "quality_score"], 0.85)

        # Check material and solution deduplication
        self.assertEqual(db.material_properties["metal"].duplicated().sum(), 0)
        self.assertEqual(db.solution_properties["solution_id"].duplicated().sum(), 0)
        self.assertEqual(set(db.solution_properties["solution_id"]), {"SOL_A", "SOL_B"})


class TestGalvanicSeriesAndDataDictionary(unittest.TestCase):
    """Seam 1 slice 3: per-solution monotonic Galvanic Series and comprehensive data dictionary."""

    def test_galvanic_series_orders_metals_monotonically_by_ecorr_per_solution(self) -> None:
        raw_echem = pd.DataFrame(
            [
                {
                    "record_id": "G1",
                    "metal": "Cu",
                    "solution_id": "SOL_1",
                    "solution_name": "1 wt% H2O2 + citric acid",
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "reference_electrode": "SHE",
                    "Ecorr_V_original": 0.285,
                    "icorr_uA_cm2": 2.3,
                    "beta_a_mV_dec": 58.0,
                    "beta_c_mV_dec": 118.0,
                },
                {
                    "record_id": "G2",
                    "metal": "Al",
                    "solution_id": "SOL_1",
                    "solution_name": "1 wt% H2O2 + citric acid",
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "reference_electrode": "SHE",
                    "Ecorr_V_original": -0.580,
                    "icorr_uA_cm2": 4.2,
                    "beta_a_mV_dec": 68.0,
                    "beta_c_mV_dec": 126.0,
                },
                {
                    "record_id": "G3",
                    "metal": "Ru",
                    "solution_id": "SOL_1",
                    "solution_name": "1 wt% H2O2 + citric acid",
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "reference_electrode": "SHE",
                    "Ecorr_V_original": 0.520,
                    "icorr_uA_cm2": 0.12,
                    "beta_a_mV_dec": 92.0,
                    "beta_c_mV_dec": 128.0,
                },
                {
                    "record_id": "G4",
                    "metal": "Co",
                    "solution_id": "SOL_1",
                    "solution_name": "1 wt% H2O2 + citric acid",
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "reference_electrode": "SHE",
                    "Ecorr_V_original": -0.095,
                    "icorr_uA_cm2": 18.5,
                    "beta_a_mV_dec": 64.0,
                    "beta_c_mV_dec": 122.0,
                },
            ]
        )

        db = HarmonizedDatabase.from_raw_tables(raw_echem, supplement_missing_core_metals=False)
        gs = db.galvanic_series
        self.assertFalse(gs.empty)

        sol1_rows = gs[gs["solution_id"] == "SOL_1"]
        self.assertEqual(list(sol1_rows["metal"]), ["Al", "Co", "Cu", "Ru"])
        self.assertEqual(list(sol1_rows["galvanic_rank"]), [1, 2, 3, 4])
        # Verify monotonic ordering of Ecorr_V_SHE within the solution
        self.assertTrue(sol1_rows["Ecorr_V_SHE"].is_monotonic_increasing)

        # Verify data dictionary covers core tables, units, and quality flags
        dd = db.data_dictionary
        self.assertFalse(dd.empty)
        self.assertTrue({"table_name", "column_name", "unit", "description"}.issubset(dd.columns))
        documented_cols = set(dd["column_name"])
        for expected_col in (
            "Ecorr_V_SHE",
            "icorr_uA_cm2",
            "beta_a_mV_dec",
            "beta_c_mV_dec",
            "is_imputed",
            "quality_score",
            "galvanic_rank",
        ):
            self.assertIn(expected_col, documented_cols)

        md = db.render_data_dictionary_markdown()
        self.assertIn("Ecorr_V_SHE", md)
        self.assertIn("V vs. SHE", md)
        self.assertIn("is_imputed", md)


class TestExposureConditionNearestNeighborLookup(unittest.TestCase):
    """Seam 1 slice 4: exact and nearest-neighbor Exposure Condition lookup with Min-Max Euclidean distance."""

    def setUp(self) -> None:
        raw_echem = pd.DataFrame(
            [
                # Candidate 1: Cu in H2O2 at pH=4.0, conc=1.0, T=25.0
                {
                    "record_id": "CU_1",
                    "metal": "Cu",
                    "solution_id": "SOL_1",
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 1.0,
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "reference_electrode": "SHE",
                    "Ecorr_V_original": 0.285,
                    "icorr_uA_cm2": 2.3,
                    "beta_a_mV_dec": 58.0,
                    "beta_c_mV_dec": 118.0,
                },
                # Candidate 2: Cu in H2O2 at pH=10.0, conc=5.0, T=65.0
                {
                    "record_id": "CU_2",
                    "metal": "Cu",
                    "solution_id": "SOL_2",
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 5.0,
                    "pH": 10.0,
                    "temperature_C": 65.0,
                    "reference_electrode": "SHE",
                    "Ecorr_V_original": 0.195,
                    "icorr_uA_cm2": 0.9,
                    "beta_a_mV_dec": 70.0,
                    "beta_c_mV_dec": 135.0,
                },
                # Candidate 3: Cu in HNO3 at pH=4.2, conc=1.1, T=26.0 (closer numerically to query, but wrong oxidizer!)
                {
                    "record_id": "CU_HNO3",
                    "metal": "Cu",
                    "solution_id": "SOL_3",
                    "oxidizer_type": "HNO3",
                    "oxidizer_concentration": 1.1,
                    "pH": 4.2,
                    "temperature_C": 26.0,
                    "reference_electrode": "SHE",
                    "Ecorr_V_original": 0.340,
                    "icorr_uA_cm2": 12.0,
                    "beta_a_mV_dec": 55.0,
                    "beta_c_mV_dec": 110.0,
                },
                # Candidate 4: Co in H2O2 at pH=4.2, conc=1.1, T=26.0 (exact match to approx query, but wrong Metal!)
                {
                    "record_id": "CO_1",
                    "metal": "Co",
                    "solution_id": "SOL_1",
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 1.1,
                    "pH": 4.2,
                    "temperature_C": 26.0,
                    "reference_electrode": "SHE",
                    "Ecorr_V_original": -0.095,
                    "icorr_uA_cm2": 18.5,
                    "beta_a_mV_dec": 64.0,
                    "beta_c_mV_dec": 122.0,
                },
            ]
        )
        self.db = HarmonizedDatabase.from_raw_tables(raw_echem, supplement_missing_core_metals=False)

    def test_exact_match_returns_zero_distance_and_zero_deltas(self) -> None:
        res = self.db.lookup_exposure_condition(
            "Cu",
            pH=4.0,
            oxidizer_type="H2O2",
            oxidizer_concentration=1.0,
            temperature_C=25.0,
        )
        self.assertIsInstance(res, ExposureLookupResult)
        self.assertEqual(res.record_id, "CU_1")
        self.assertTrue(res.is_exact_match)
        self.assertAlmostEqual(res.normalized_distance, 0.0, places=9)
        self.assertAlmostEqual(res.Ecorr_V_SHE, 0.285, places=6)
        self.assertAlmostEqual(res.icorr_uA_cm2, 2.3, places=6)
        self.assertAlmostEqual(res.condition_differences["pH"], 0.0, places=6)
        self.assertAlmostEqual(res.condition_differences["oxidizer_concentration"], 0.0, places=6)
        self.assertAlmostEqual(res.condition_differences["temperature_C"], 0.0, places=6)

    def test_nearest_neighbor_filters_by_metal_and_oxidizer_and_computes_min_max_distance(self) -> None:
        # Query Cu in H2O2 at pH=5.5, conc=2.0, T=35.0
        # For (Cu, H2O2): pH range [4.0, 10.0] (span 6.0), conc range [1.0, 5.0] (span 4.0), T range [25.0, 65.0] (span 40.0)
        # Normalized deltas to CU_1 (4.0, 1.0, 25.0):
        #   dpH = (5.5 - 4.0)/6.0 = 0.25
        #   dConc = (2.0 - 1.0)/4.0 = 0.25
        #   dT = (35.0 - 25.0)/40.0 = 0.25
        #   Euclidean distance = sqrt(0.25^2 * 3) = 0.25 * sqrt(3) = 0.4330127018922193
        res = self.db.lookup_exposure_condition(
            "Cu",
            pH=5.5,
            oxidizer_type="H2O2",
            oxidizer_concentration=2.0,
            temperature_C=35.0,
        )
        self.assertEqual(res.record_id, "CU_1")
        self.assertFalse(res.is_exact_match)
        self.assertAlmostEqual(res.normalized_distance, 0.25 * math.sqrt(3.0), places=6)
        self.assertAlmostEqual(res.condition_differences["pH"], -1.5, places=6)
        self.assertAlmostEqual(res.condition_differences["oxidizer_concentration"], -1.0, places=6)
        self.assertAlmostEqual(res.condition_differences["temperature_C"], -10.0, places=6)
        self.assertIn("pH", res.mismatch_summary)


class TestDirectoryIngestionSeedFallbackAndExport(unittest.TestCase):
    """Seam 1 slice 5: directory ingestion of raw CSVs, seed fallback on empty/missing directory, and export."""

    def test_empty_directory_falls_back_cleanly_to_seed_covering_all_eight_core_metals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = HarmonizedDatabase.from_directory(Path(tmpdir) / "nonexistent_raw")
            self.assertEqual(set(db.material_properties["metal"]), set(CORE_METALS))
            self.assertEqual(set(db.electrochemical_database["metal"]), set(CORE_METALS))
            self.assertTrue((db.electrochemical_database["icorr_uA_cm2"] > 0).all())
            self.assertTrue((db.electrochemical_database["beta_a_mV_dec"] > 0).all())
            self.assertTrue((db.electrochemical_database["beta_c_mV_dec"] > 0).all())

    def test_workspace_raw_directory_ingests_and_exports_unified_tables(self) -> None:
        db = HarmonizedDatabase.from_directory("data/raw")
        # Data-driven ingestion includes all 8 core metals plus extended materials (e.g., Fe, Cr, Ni, TiN)
        self.assertTrue(set(CORE_METALS).issubset(set(db.material_properties["metal"])))
        self.assertTrue(set(CORE_METALS).issubset(set(db.electrochemical_database["metal"])))
        self.assertGreater(len(db.available_metals), len(CORE_METALS))
        # Raw dataset should have >1000 valid harmonized records across all materials
        self.assertGreater(len(db.electrochemical_database), 1000)
        self.assertEqual(db.material_properties["metal"].duplicated().sum(), 0)
        self.assertEqual(db.solution_properties["solution_id"].duplicated().sum(), 0)
        # Solution is independent of operating temperature per GLOSSARY.md
        self.assertNotIn("temperature_C", db.solution_properties.columns)
        # Every column in all 4 tables must have a non-generic data dictionary description
        self.assertFalse(
            db.data_dictionary["description"].str.startswith("Harmonized attribute").any()
        )
        # Canonical dissolution valence and equivalent weight must be synchronized for Ru (z=4)
        ru_mat = db.material_properties.set_index("metal").loc["Ru"]
        self.assertEqual(int(ru_mat["valence"]), 4)
        self.assertAlmostEqual(float(ru_mat["equivalent_weight_g_eq"]), 101.07 / 4.0, places=3)
        ru_echem_valences = set(
            db.electrochemical_database[db.electrochemical_database["metal"] == "Ru"]["valence"]
        )
        self.assertEqual(ru_echem_valences, {4})
        # Verify ExposureCondition object lookup against an actual Cu record in the database
        cu_row = db.electrochemical_database[
            db.electrochemical_database["metal"] == "Cu"
        ].iloc[0]
        cond_res = db.lookup_exposure_condition(
            ExposureCondition(
                metal="Cu",
                pH=float(cu_row["pH"]),
                oxidizer_type=str(cu_row["oxidizer_type"]),
                oxidizer_concentration=float(cu_row["oxidizer_concentration"]),
                temperature_C=float(cu_row["temperature_C"]),
            )
        )
        self.assertTrue(cond_res.is_exact_match)

        # Verify monotonic Ecorr_V_SHE ordering within every Solution in galvanic_series
        for _, grp in db.galvanic_series.groupby("solution_id"):
            self.assertTrue(grp["Ecorr_V_SHE"].is_monotonic_increasing)

        with tempfile.TemporaryDirectory() as out_dir:
            exported = db.export_to_directory(out_dir)
            for key in (
                "electrochemical_database",
                "material_properties",
                "solution_properties",
                "galvanic_series",
                "data_dictionary",
                "data_dictionary_markdown",
            ):
                self.assertIn(key, exported)
                self.assertTrue(exported[key].exists())

    def test_exploration_and_cleaning_notebook_is_valid_and_executed(self) -> None:
        nb_path = Path("notebooks/01_data_exploration_and_cleaning.ipynb")
        self.assertTrue(nb_path.exists())
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb.get("nbformat"), 4)
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertGreaterEqual(len(code_cells), 4)
        for cell in code_cells:
            self.assertTrue(cell.get("outputs"), "Every code cell in demonstration notebook must have outputs.")

    def test_solution_attributes_join_and_no_double_unit_scaling(self) -> None:
        raw_echem = pd.DataFrame(
            [
                {
                    "record_id": "R_PRE_NORM",
                    "metal": "Cu",
                    "solution_id": "SOL_REL",
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.285,
                    "icorr_uA_cm2": 2.3,
                    "icorr_unit": "mA/cm2",  # must NOT double-scale pre-normalized icorr_uA_cm2
                    "beta_a_mV_dec": 60.0,
                    "beta_a_unit": "V/dec",  # must NOT double-scale pre-normalized beta_a_mV_dec
                    "beta_c_mV_dec": -120.0,
                    "beta_c_unit": "V/dec",
                },
                {
                    "record_id": "R_RAW",
                    "metal": "Co",
                    "solution_id": "SOL_REL",
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": -0.095,
                    "icorr": 0.0185,
                    "icorr_unit": "mA/cm2",  # raw icorr -> 18.5 uA/cm2
                    "beta_a": 0.064,
                    "beta_a_unit": "V/dec",  # raw beta_a -> 64.0 mV/dec
                    "beta_c": -0.122,
                    "beta_c_unit": "V/dec",  # raw beta_c -> 122.0 mV/dec
                },
            ]
        )
        raw_sol = pd.DataFrame(
            [
                {
                    "solution_id": "SOL_REL",
                    "solution_name": "Relational Acidic H2O2 + BTA",
                    "pH": 4.0,
                    "oxidizer_type": "H2O2",
                    "oxidizer_concentration": 1.0,
                    "acid_type": "citric acid",
                    "acid_concentration": 50.0,
                    "inhibitor_type": "BTA",
                    "inhibitor_concentration": 1.0,
                    "ORP_mV": 480.0,
                }
            ]
        )
        db = HarmonizedDatabase.from_raw_tables(raw_echem, solution_df=raw_sol)
        echem = db.electrochemical_database.set_index("record_id")

        # Verify no double-scaling on pre-normalized row, and proper scaling on raw row
        self.assertAlmostEqual(float(echem.loc["R_PRE_NORM", "icorr_uA_cm2"]), 2.3, places=6)
        self.assertAlmostEqual(float(echem.loc["R_PRE_NORM", "beta_a_mV_dec"]), 60.0, places=6)
        self.assertAlmostEqual(float(echem.loc["R_PRE_NORM", "beta_c_mV_dec"]), 120.0, places=6)
        self.assertAlmostEqual(float(echem.loc["R_RAW", "icorr_uA_cm2"]), 18.5, places=6)
        self.assertAlmostEqual(float(echem.loc["R_RAW", "beta_a_mV_dec"]), 64.0, places=6)
        self.assertAlmostEqual(float(echem.loc["R_RAW", "beta_c_mV_dec"]), 122.0, places=6)

        # Verify relational Solution attributes (acid, inhibitor, ORP_mV) joined onto echem records
        self.assertEqual(str(echem.loc["R_PRE_NORM", "acid_type"]), "citric acid")
        self.assertEqual(str(echem.loc["R_PRE_NORM", "inhibitor_type"]), "BTA")
        self.assertAlmostEqual(float(echem.loc["R_PRE_NORM", "ORP_mV"]), 480.0, places=6)


class TestFilteringOptimizationAndPhysicalDerivation(unittest.TestCase):
    """Verify CR/Rp physical derivation, content dedup, physical bounds, and partial-target retention."""

    def test_derive_icorr_from_cr_and_rp_analytical(self) -> None:
        # Cu (M=63.546, rho=8.96, z=2): CR_nm_min = i_uA * 63.546 * 600 / (2 * 96485.33212 * 8.96)
        # For i_corr = 10 uA/cm2 -> CR = 0.220513... nm/min
        i_from_nm = derive_icorr_from_cr(
            0.2205130268,
            "nm/min",
            molar_mass_g_mol=63.546,
            density_g_cm3=8.96,
            valence=2.0,
        )
        self.assertAlmostEqual(i_from_nm, 10.0, places=3)

        # Stern-Geary with beta_a=60, beta_c=120 -> B = 40 / ln(10) = 17.37178 mV
        # At Rp = 1 kOhm*cm2 = 1000 Ohm*cm2 -> i_corr = 17.37178 uA/cm2
        i_from_rp = derive_icorr_from_rp(1.0, "kΩ·cm2", beta_a_mV_dec=60.0, beta_c_mV_dec=120.0)
        self.assertAlmostEqual(i_from_rp, 17.37178, places=3)
        self.assertTrue(math.isnan(derive_icorr_from_cr(1.0, "unsupported_unit", molar_mass_g_mol=63.546, density_g_cm3=8.96, valence=2.0)))

    def test_content_dedup_tafel_bounds_physical_quarantine_and_partial_targets(self) -> None:
        raw_echem = pd.DataFrame(
            [
                # Two duplicates with different record_ids (higher quality_score must be kept)
                {
                    "record_id": "DUP_LOW",
                    "metal": "Cu",
                    "solution_id": "SOL_1",
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.250,
                    "icorr_uA_cm2": 5.0,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                    "data_quality_score": 0.70,
                },
                {
                    "record_id": "DUP_HIGH",
                    "metal": "Cu",
                    "solution_id": "SOL_1",
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.250,
                    "icorr_uA_cm2": 5.0,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                    "data_quality_score": 0.95,
                },
                # Out-of-bounds Tafel slope (> 500 mV/dec) -> converted to imputed, kept as complete pair
                {
                    "record_id": "OOB_TAFEL",
                    "metal": "Cu",
                    "solution_id": "SOL_1",
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.260,
                    "icorr_uA_cm2": 6.0,
                    "beta_a_mV_dec": 850.0,
                    "beta_c_mV_dec": 120.0,
                    "pH_source": "inferred_neutral",
                    "data_quality_score": 0.90,
                },
                # Missing icorr but has CR_reported_value -> rescued into complete pair via Faraday
                {
                    "record_id": "RESCUE_CR",
                    "metal": "Cu",
                    "solution_id": "SOL_1",
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.270,
                    "icorr_uA_cm2": None,
                    "CR_reported_value": 0.220513,
                    "CR_reported_unit": "nm/min",
                    "beta_a_mV_dec": 65.0,
                    "beta_c_mV_dec": 125.0,
                    "data_quality_score": 0.90,
                },
                # Missing icorr but has Rp_ohm_cm2 -> rescued into complete pair via Stern-Geary
                {
                    "record_id": "RESCUE_RP",
                    "metal": "Cu",
                    "solution_id": "SOL_1",
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.275,
                    "icorr_uA_cm2": None,
                    "Rp_ohm_cm2": 2.0,
                    "Rp_unit": "kohm*cm2",
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                    "data_quality_score": 0.90,
                },
                # Partial-target: Ecorr only -> routed to partial_target_records, NOT complete or rejected
                {
                    "record_id": "PARTIAL_ECORR",
                    "metal": "Ru",
                    "solution_id": "SOL_1",
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.500,
                    "icorr_uA_cm2": None,
                    "beta_a_mV_dec": 75.0,
                    "beta_c_mV_dec": 115.0,
                },
                # Extreme icorr (> 1e5 uA/cm2) -> quarantined to rejected_records
                {
                    "record_id": "EXTREME_ICORR",
                    "metal": "Cu",
                    "solution_id": "SOL_1",
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.200,
                    "icorr_uA_cm2": 500000.0,
                },
                # Missing both Ecorr and icorr -> quarantined to rejected_records
                {
                    "record_id": "MISSING_BOTH",
                    "metal": "Cu",
                    "solution_id": "SOL_1",
                    "pH": 4.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": None,
                    "icorr_uA_cm2": None,
                },
            ]
        )

        db = HarmonizedDatabase.from_raw_tables(raw_echem, supplement_missing_core_metals=False)
        echem = db.electrochemical_database.set_index("record_id")
        partial = db.partial_target_records.set_index("record_id")
        rej = db.rejected_records.set_index("record_id")

        # Deduplication kept DUP_HIGH and dropped DUP_LOW
        self.assertIn("DUP_HIGH", echem.index)
        self.assertNotIn("DUP_LOW", echem.index)

        # Out-of-bounds Tafel was flagged as imputed and replaced within [20, 500], and inferred_neutral discounted quality_score
        self.assertTrue(bool(echem.loc["OOB_TAFEL", "beta_a_imputed"]))
        self.assertLessEqual(float(echem.loc["OOB_TAFEL", "beta_a_mV_dec"]), 500.0)
        # (0.90 - 0.15) * 0.75 = 0.5625
        self.assertAlmostEqual(float(echem.loc["OOB_TAFEL", "quality_score"]), 0.5625, places=4)

        # CR and Rp rescued records entered complete pairs with provenance and quality discount
        self.assertEqual(str(echem.loc["RESCUE_CR", "icorr_source"]), "derived_from_CR")
        self.assertAlmostEqual(float(echem.loc["RESCUE_CR", "icorr_uA_cm2"]), 10.0, places=2)
        self.assertEqual(str(echem.loc["RESCUE_RP", "icorr_source"]), "derived_from_Rp")
        self.assertAlmostEqual(float(echem.loc["RESCUE_RP", "icorr_uA_cm2"]), 8.6859, places=2)

        # Partial-target record is in partial_target_records and ml_training_dataframe, not in galvanic_series
        self.assertIn("PARTIAL_ECORR", partial.index)
        self.assertNotIn("PARTIAL_ECORR", echem.index)
        self.assertNotIn("Ru", set(db.galvanic_series["metal"]))
        self.assertIn("PARTIAL_ECORR", set(db.ml_training_dataframe["record_id"]))

        # Rejected records contain EXTREME_ICORR and MISSING_BOTH with explicit reasons
        self.assertIn("EXTREME_ICORR", rej.index)
        self.assertIn("icorr_out_of_physical_bounds", str(rej.loc["EXTREME_ICORR", "rejection_reason"]))
        self.assertIn("MISSING_BOTH", rej.index)


class TestSolventRegimeClassificationAndWaterInference(unittest.TestCase):
    """Issue 01: solvent_regime thresholds, solvent_subclass taxonomy, 3-tier water inference, and quality discount."""

    def test_water_content_regime_thresholds_and_boundaries(self) -> None:
        self.assertEqual(classify_solvent_regime(100.0), "aqueous")
        self.assertEqual(classify_solvent_regime(90.0), "aqueous")
        self.assertEqual(classify_solvent_regime(89.99), "semi_aqueous")
        self.assertEqual(classify_solvent_regime(50.0), "semi_aqueous")
        self.assertEqual(classify_solvent_regime(1.0), "semi_aqueous")
        self.assertEqual(classify_solvent_regime(0.99), "non_aqueous")
        self.assertEqual(classify_solvent_regime(0.0), "non_aqueous")
        with self.assertRaises(ValueError):
            classify_solvent_regime(float("nan"))

        raw_echem = pd.DataFrame(
            [
                {
                    "record_id": f"B_{idx}",
                    "metal": "Cu",
                    "solution_id": f"SOL_B_{idx}",
                    "pH": 7.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.25,
                    "icorr_uA_cm2": 2.0 + idx,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                    "water_content_wt_pct": water_pct,
                    "organic_solvent_type": "EG" if water_pct < 100.0 else "None",
                }
                for idx, water_pct in enumerate((100.0, 90.0, 89.99, 1.0, 0.99, 0.0))
            ]
        )
        db = HarmonizedDatabase.from_raw_tables(raw_echem, supplement_missing_core_metals=False)
        expected = ["aqueous", "aqueous", "semi_aqueous", "semi_aqueous", "non_aqueous", "non_aqueous"]
        sol = db.solution_properties.set_index("solution_id")
        echem = db.electrochemical_database.set_index("record_id")
        for idx, expected_regime in enumerate(expected):
            self.assertEqual(sol.loc[f"SOL_B_{idx}", "solvent_regime"], expected_regime)
            self.assertEqual(echem.loc[f"B_{idx}", "solvent_regime"], expected_regime)

    def test_solvent_subclass_taxonomy_for_aqueous_and_organic_regimes(self) -> None:
        # Aqueous pH and buffer subclasses
        self.assertEqual(classify_solvent_subclass("aqueous", pH=2.5), "acidic")
        self.assertEqual(classify_solvent_subclass("aqueous", pH=7.0), "neutral")
        self.assertEqual(classify_solvent_subclass("aqueous", pH=10.5), "alkaline")
        self.assertEqual(
            classify_solvent_subclass("aqueous", pH=7.4, solution_category="buffered"),
            "buffered",
        )
        self.assertEqual(
            classify_solvent_subclass("aqueous", pH=7.2, solution_name="Phosphate buffer"),
            "buffered",
        )

        # Semi-aqueous & non-aqueous chemical families
        for regime in ("semi_aqueous", "non_aqueous"):
            self.assertEqual(classify_solvent_subclass(regime, organic_solvent_type="EG"), "glycol")
            self.assertEqual(
                classify_solvent_subclass(regime, organic_solvent_type="propylene glycol"),
                "glycol",
            )
            self.assertEqual(classify_solvent_subclass(regime, organic_solvent_type="IPA"), "alcohol")
            self.assertEqual(
                classify_solvent_subclass(regime, organic_solvent_type="ethanol"), "alcohol"
            )
            self.assertEqual(
                classify_solvent_subclass(regime, organic_solvent_type="DMSO"), "polar_aprotic"
            )
            self.assertEqual(
                classify_solvent_subclass(regime, organic_solvent_type="NMP"), "polar_aprotic"
            )
            self.assertEqual(
                classify_solvent_subclass(regime, organic_solvent_type="DMF"), "polar_aprotic"
            )
            self.assertEqual(
                classify_solvent_subclass(regime, organic_solvent_type="acetone"),
                "polar_aprotic",
            )
            self.assertEqual(
                classify_solvent_subclass(regime, base_solvent="glacial acetic acid"),
                "organic_acid",
            )
            self.assertEqual(
                classify_solvent_subclass(regime, organic_solvent_type="glycolic acid"),
                "organic_acid",
            )
            self.assertEqual(
                classify_solvent_subclass(regime, organic_solvent_type="BMIM-BF4 ionic liquid"),
                "other_organic",
            )
            self.assertEqual(
                classify_solvent_subclass(
                    regime,
                    organic_solvent_type="BMIM-BF4",
                    acid_type="citric acid",
                ),
                "other_organic",
            )

    def test_three_tier_water_fallback_and_quality_penalty(self) -> None:
        # Direct helper checks for the 3 provenance tiers
        self.assertEqual(
            infer_water_content_and_regime(organic_solvent_type="DMSO", organic_solvent_concentration=30.0),
            (70.0, "reported", "semi_aqueous"),
        )
        self.assertEqual(
            infer_water_content_and_regime(base_solvent="DI water"),
            (100.0, "inferred_aqueous", "aqueous"),
        )
        self.assertEqual(
            infer_water_content_and_regime(base_solvent="DI-water"),
            (100.0, "inferred_aqueous", "aqueous"),
        )
        self.assertEqual(
            infer_water_content_and_regime(base_solvent="ultrapure DI water"),
            (100.0, "inferred_aqueous", "aqueous"),
        )
        for aq_label in ("ultra-pure water", "de-ionized water", "double-distilled water"):
            self.assertEqual(
                infer_water_content_and_regime(base_solvent=aq_label),
                (100.0, "inferred_aqueous", "aqueous"),
            )
        self.assertEqual(
            infer_water_content_and_regime(base_solvent="water-EG"),
            (50.0, "inferred_regime_default", "semi_aqueous"),
        )
        self.assertEqual(
            infer_water_content_and_regime(
                base_solvent="DMSO",
                organic_solvent_type="DMSO",
                organic_solvent_concentration=100.0,
            ),
            (0.0, "reported", "non_aqueous"),
        )
        self.assertEqual(
            infer_water_content_and_regime(base_solvent="DI water", organic_solvent_type="EG"),
            (50.0, "inferred_regime_default", "semi_aqueous"),
        )
        self.assertEqual(
            infer_water_content_and_regime(base_solvent="DMSO"),
            (0.0, "inferred_regime_default", "non_aqueous"),
        )
        self.assertEqual(
            infer_water_content_and_regime(base_solvent="non-aqueous"),
            (0.0, "inferred_regime_default", "non_aqueous"),
        )
        zero_org_nonaq = infer_water_content_and_regime(
            base_solvent="DMSO",
            organic_solvent_type="None",
            organic_solvent_concentration=0.0,
        )
        self.assertEqual(zero_org_nonaq, (0.0, "inferred_regime_default", "non_aqueous"))
        self.assertEqual(zero_org_nonaq.water_content_source, "inferred_regime_default")
        # Non-aqueous base_solvent + organic co-solvent concentration must NOT assume water = 100 - org_pct
        self.assertEqual(
            infer_water_content_and_regime(
                base_solvent="DMSO",
                organic_solvent_type="EG",
                organic_solvent_concentration=10.0,
            ),
            (0.0, "inferred_regime_default", "non_aqueous"),
        )

        raw_echem = pd.DataFrame(
            [
                # Tier 1a: explicit water_content_wt_pct -> reported, no penalty
                {
                    "record_id": "TIER1_WATER",
                    "metal": "Cu",
                    "solution_id": "SOL_T1W",
                    "pH": 7.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.20,
                    "icorr_uA_cm2": 1.0,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                    "base_solvent": "DI water",
                    "organic_solvent_type": "IPA",
                    "water_content_wt_pct": 65.0,
                    "data_quality_score": 0.90,
                },
                # Tier 1b: explicit organic concentration (wt%) -> 100 - 30 = 70 wt%, reported, no penalty
                {
                    "record_id": "TIER1_ORG_PCT",
                    "metal": "Cu",
                    "solution_id": "SOL_T1O",
                    "pH": 7.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.21,
                    "icorr_uA_cm2": 1.1,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                    "base_solvent": "DI water",
                    "organic_solvent_type": "DMSO",
                    "organic_solvent_concentration": 30.0,
                    "organic_solvent_concentration_unit": "wt%",
                    "data_quality_score": 0.90,
                },
                # Tier 1b: fraction unit (0.4 -> 40 wt% organic -> 60 wt% water), reported
                {
                    "record_id": "TIER1_FRACTION",
                    "metal": "Cu",
                    "solution_id": "SOL_T1F",
                    "pH": 7.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.215,
                    "icorr_uA_cm2": 1.15,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                    "base_solvent": "DI water",
                    "organic_solvent_type": "NMP",
                    "organic_solvent_concentration": 0.4,
                    "organic_solvent_concentration_unit": "fraction",
                    "data_quality_score": 0.90,
                },
                # Tier 2: DI water without organic co-solvent -> 100.0, inferred_aqueous, no penalty
                {
                    "record_id": "TIER2_AQ",
                    "metal": "Cu",
                    "solution_id": "SOL_T2",
                    "pH": 3.5,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.22,
                    "icorr_uA_cm2": 1.2,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                    "base_solvent": "DI water",
                    "data_quality_score": 0.90,
                },
                # Tier 3: organic co-solvent without numeric % concentration -> 50.0, inferred_regime_default, x0.85
                {
                    "record_id": "TIER3_SEMI",
                    "metal": "Cu",
                    "solution_id": "SOL_T3S",
                    "pH": 7.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.23,
                    "icorr_uA_cm2": 1.3,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                    "base_solvent": "DI water",
                    "organic_solvent_type": "EG",
                    "organic_solvent_concentration": None,
                    "data_quality_score": 0.90,
                },
                # Tier 3: mM concentration unit is not wt% -> falls back to 50.0, inferred_regime_default, x0.85
                {
                    "record_id": "TIER3_MM",
                    "metal": "Cu",
                    "solution_id": "SOL_T3M",
                    "pH": 7.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.235,
                    "icorr_uA_cm2": 1.35,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                    "base_solvent": "DI water",
                    "organic_solvent_type": "EG",
                    "organic_solvent_concentration": 10.0,
                    "organic_solvent_concentration_unit": "mM",
                    "data_quality_score": 0.90,
                },
                # Tier 3: non-aqueous base_solvent without concentration -> 0.0, inferred_regime_default, x0.85
                {
                    "record_id": "TIER3_NONAQ",
                    "metal": "Cu",
                    "solution_id": "SOL_T3N",
                    "pH": 7.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.24,
                    "icorr_uA_cm2": 1.4,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                    "base_solvent": "DMSO",
                    "data_quality_score": 0.90,
                },
                # Tier 3: floor clamping check at 0.05
                {
                    "record_id": "TIER3_FLOOR",
                    "metal": "Cu",
                    "solution_id": "SOL_T3F",
                    "pH": 7.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.245,
                    "icorr_uA_cm2": 1.45,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                    "base_solvent": "NMP",
                    "data_quality_score": 0.05,
                },
            ]
        )

        db = HarmonizedDatabase.from_raw_tables(raw_echem, supplement_missing_core_metals=False)
        echem = db.electrochemical_database.set_index("record_id")
        sol = db.solution_properties.set_index("solution_id")

        self.assertAlmostEqual(float(echem.loc["TIER1_WATER", "water_content_wt_pct"]), 65.0)
        self.assertEqual(echem.loc["TIER1_WATER", "water_content_source"], "reported")
        self.assertEqual(echem.loc["TIER1_WATER", "solvent_regime"], "semi_aqueous")
        self.assertEqual(echem.loc["TIER1_WATER", "solvent_subclass"], "alcohol")
        self.assertAlmostEqual(float(echem.loc["TIER1_WATER", "quality_score"]), 0.90, places=6)

        self.assertAlmostEqual(float(echem.loc["TIER1_ORG_PCT", "water_content_wt_pct"]), 70.0)
        self.assertEqual(echem.loc["TIER1_ORG_PCT", "water_content_source"], "reported")
        self.assertEqual(echem.loc["TIER1_ORG_PCT", "solvent_subclass"], "polar_aprotic")
        self.assertAlmostEqual(float(echem.loc["TIER1_ORG_PCT", "quality_score"]), 0.90, places=6)

        self.assertAlmostEqual(float(echem.loc["TIER1_FRACTION", "water_content_wt_pct"]), 60.0)
        self.assertEqual(echem.loc["TIER1_FRACTION", "water_content_source"], "reported")
        self.assertEqual(echem.loc["TIER1_FRACTION", "solvent_subclass"], "polar_aprotic")

        self.assertAlmostEqual(float(echem.loc["TIER2_AQ", "water_content_wt_pct"]), 100.0)
        self.assertEqual(echem.loc["TIER2_AQ", "water_content_source"], "inferred_aqueous")
        self.assertEqual(echem.loc["TIER2_AQ", "solvent_regime"], "aqueous")
        self.assertEqual(echem.loc["TIER2_AQ", "solvent_subclass"], "acidic")
        self.assertAlmostEqual(float(echem.loc["TIER2_AQ", "quality_score"]), 0.90, places=6)

        expected_discounted = 0.90 * INFERRED_REGIME_DEFAULT_WATER_WEIGHT_FACTOR
        for rid, sid, exp_water, exp_regime, exp_sub in (
            ("TIER3_SEMI", "SOL_T3S", 50.0, "semi_aqueous", "glycol"),
            ("TIER3_MM", "SOL_T3M", 50.0, "semi_aqueous", "glycol"),
            ("TIER3_NONAQ", "SOL_T3N", 0.0, "non_aqueous", "polar_aprotic"),
        ):
            self.assertAlmostEqual(float(echem.loc[rid, "water_content_wt_pct"]), exp_water)
            self.assertEqual(echem.loc[rid, "water_content_source"], "inferred_regime_default")
            self.assertEqual(echem.loc[rid, "solvent_regime"], exp_regime)
            self.assertEqual(echem.loc[rid, "solvent_subclass"], exp_sub)
            self.assertAlmostEqual(float(echem.loc[rid, "quality_score"]), expected_discounted, places=6)
            self.assertEqual(sol.loc[sid, "water_content_source"], "inferred_regime_default")
            self.assertAlmostEqual(float(sol.loc[sid, "water_content_wt_pct"]), exp_water)

        self.assertAlmostEqual(float(echem.loc["TIER3_FLOOR", "quality_score"]), 0.05, places=6)

    def test_data_dictionary_seed_and_raw_database_solvent_coverage(self) -> None:
        required_cols = {"solvent_regime", "solvent_subclass", "water_content_wt_pct", "water_content_source"}

        for label, db in (
            ("seed", HarmonizedDatabase.from_seed()),
            ("raw", HarmonizedDatabase.from_directory("data/raw")),
        ):
            with self.subTest(dataset=label):
                self.assertTrue(required_cols.issubset(db.solution_properties.columns))
                self.assertTrue(required_cols.issubset(db.electrochemical_database.columns))
                self.assertFalse(
                    db.data_dictionary["description"].str.startswith("Harmonized attribute").any()
                )
                for tbl in ("solution_properties", "electrochemical_database"):
                    tbl_dd = db.data_dictionary[db.data_dictionary["table_name"] == tbl]
                    self.assertTrue(required_cols.issubset(set(tbl_dd["column_name"])))
                    water_row = tbl_dd[tbl_dd["column_name"] == "water_content_wt_pct"].iloc[0]
                    self.assertEqual(water_row["unit"], "wt%")
                for df in (db.solution_properties, db.electrochemical_database):
                    self.assertTrue(set(df["solvent_regime"]).issubset(SOLVENT_REGIMES))
                    self.assertTrue(df["water_content_wt_pct"].between(0.0, 100.0).all())
                    self.assertTrue(
                        set(df["water_content_source"]).issubset(
                            {"reported", "inferred_aqueous", "inferred_regime_default"}
                        )
                    )
                    aq_subs = set(df.loc[df["solvent_regime"] == "aqueous", "solvent_subclass"])
                    self.assertTrue(aq_subs.issubset(AQUEOUS_SOLVENT_SUBCLASSES))
                    org_subs = set(df.loc[df["solvent_regime"] != "aqueous", "solvent_subclass"])
                    self.assertTrue(org_subs.issubset(ORGANIC_SOLVENT_SUBCLASSES))
            if label == "raw":
                raw_sol = db.solution_properties.set_index("solution_id")
                # SOL_eba7525a: 10 mM EG (unit="mM" in components_json) -> regime default 50 wt%, glycol
                self.assertEqual(raw_sol.loc["SOL_eba7525a", "water_content_source"], "inferred_regime_default")
                self.assertAlmostEqual(float(raw_sol.loc["SOL_eba7525a", "water_content_wt_pct"]), 50.0)
                self.assertEqual(raw_sol.loc["SOL_eba7525a", "solvent_regime"], "semi_aqueous")
                self.assertEqual(raw_sol.loc["SOL_eba7525a", "solvent_subclass"], "glycol")
                # SOL_5d8cf280: 10 % DMSO -> reported 90 wt% water
                self.assertEqual(raw_sol.loc["SOL_5d8cf280", "water_content_source"], "reported")
                self.assertAlmostEqual(float(raw_sol.loc["SOL_5d8cf280", "water_content_wt_pct"]), 90.0)
                # SOL_5cef03ca: 70 % IPA -> reported 30 wt% water
                self.assertEqual(raw_sol.loc["SOL_5cef03ca", "water_content_source"], "reported")
                self.assertAlmostEqual(float(raw_sol.loc["SOL_5cef03ca", "water_content_wt_pct"]), 30.0)

    def test_components_json_unit_fallback_and_row_level_solvent_override(self) -> None:
        raw_sol = pd.DataFrame(
            [
                {
                    "solution_id": "SOL_AQ_BASE",
                    "solution_name": "DI water pH 7",
                    "base_solvent": "DI water",
                    "organic_solvent_type": "None",
                    "pH": 7.0,
                },
                {
                    "solution_id": "SOL_DMSO_30",
                    "solution_name": "30% DMSO",
                    "base_solvent": "DI water",
                    "organic_solvent_type": "DMSO",
                    "organic_solvent_concentration": 30.0,
                    "organic_solvent_concentration_unit": "wt%",
                    "pH": 7.0,
                },
                {
                    "solution_id": "SOL_ACID_NO_PH",
                    "solution_name": "Acidic solution without measured pH",
                    "solution_category": "acidic",
                    "base_solvent": "DI water",
                    "pH": None,
                },
                {
                    "solution_id": "SOL_NONAQ_BASE",
                    "solution_name": "Neat DMSO",
                    "base_solvent": "DMSO",
                    "organic_solvent_type": "DMSO",
                    "pH": 7.0,
                },
            ]
        )
        raw_echem = pd.DataFrame(
            [
                # Overrides non-aqueous Solution (with organic_solvent_type="DMSO") via DI water -> must NOT inherit DMSO
                {
                    "record_id": "ROW_NONAQ_OVERRIDE_AQ",
                    "metal": "Cu",
                    "solution_id": "SOL_NONAQ_BASE",
                    "base_solvent": "DI water",
                    "pH": 7.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.205,
                    "icorr_uA_cm2": 1.05,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                    "data_quality_score": 0.90,
                },
                # Explicitly overrides 30% DMSO Solution with organic_solvent_type="None" -> aqueous, no penalty
                {
                    "record_id": "ROW_DMSO_OVERRIDE_NONE",
                    "metal": "Cu",
                    "solution_id": "SOL_DMSO_30",
                    "organic_solvent_type": "None",
                    "pH": 7.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.207,
                    "icorr_uA_cm2": 1.07,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                    "data_quality_score": 0.90,
                },
                # Overrides non-aqueous DMSO Solution with methanol -> alcohol (not polar_aprotic)
                {
                    "record_id": "ROW_DMSO_OVERRIDE_MEOH",
                    "metal": "Cu",
                    "solution_id": "SOL_NONAQ_BASE",
                    "base_solvent": "methanol",
                    "pH": 7.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.208,
                    "icorr_uA_cm2": 1.08,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                    "data_quality_score": 0.90,
                },
                # pH_effective only (no pH column value) with acidic value -> acidic in both tables
                {
                    "record_id": "ROW_PH_EFF_ONLY",
                    "metal": "Cu",
                    "solution_id": "SOL_PH_EFF_ONLY",
                    "pH_effective": 2.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.209,
                    "icorr_uA_cm2": 1.09,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                    "data_quality_score": 0.90,
                },
                # Matches SOL_AQ_BASE with no override -> inferred_aqueous, no penalty
                {
                    "record_id": "ROW_AQ_PLAIN",
                    "metal": "Cu",
                    "solution_id": "SOL_AQ_BASE",
                    "pH": 7.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.20,
                    "icorr_uA_cm2": 1.0,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                    "data_quality_score": 0.90,
                },
                # Same solution_id, but row specifies organic_solvent_type="EG" without conc -> must override to 50 wt%, x0.85
                {
                    "record_id": "ROW_AQ_OVERRIDE_EG",
                    "metal": "Cu",
                    "solution_id": "SOL_AQ_BASE",
                    "pH": 7.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.22,
                    "icorr_uA_cm2": 1.5,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                    "organic_solvent_type": "EG",
                    "data_quality_score": 0.90,
                },
                # Overrides organic_solvent_type against 30% DMSO solution -> must NOT inherit 30.0 conc
                {
                    "record_id": "ROW_DMSO_OVERRIDE_EG",
                    "metal": "Cu",
                    "solution_id": "SOL_DMSO_30",
                    "pH": 7.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.23,
                    "icorr_uA_cm2": 1.7,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                    "organic_solvent_type": "EG",
                    "data_quality_score": 0.90,
                },
                # Missing pH with solution_category="acidic" -> must stay "acidic" in both tables
                {
                    "record_id": "ROW_ACID_NO_PH",
                    "metal": "Cu",
                    "solution_id": "SOL_ACID_NO_PH",
                    "pH": None,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.235,
                    "icorr_uA_cm2": 1.8,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                    "data_quality_score": 0.90,
                },
                # Missing organic_solvent_concentration_unit column, unit="mM" inside components_json -> must NOT do 100 - 10
                {
                    "record_id": "ROW_COMP_MM",
                    "metal": "Cu",
                    "solution_id": "SOL_COMP_MM",
                    "pH": 7.0,
                    "temperature_C": 25.0,
                    "Ecorr_V_SHE": 0.24,
                    "icorr_uA_cm2": 2.0,
                    "beta_a_mV_dec": 60.0,
                    "beta_c_mV_dec": 120.0,
                    "base_solvent": "DI water",
                    "organic_solvent_type": "EG",
                    "organic_solvent_concentration": 10.0,
                    "components_json": '[{"species": "EG", "conc": "10", "unit": "mM", "role": "organic_solvent"}]',
                    "data_quality_score": 0.90,
                },
            ]
        )

        db = HarmonizedDatabase.from_raw_tables(
            raw_echem, solution_df=raw_sol, supplement_missing_core_metals=False
        )
        echem = db.electrochemical_database.set_index("record_id")
        sol = db.solution_properties.set_index("solution_id")

        self.assertEqual(echem.loc["ROW_AQ_PLAIN", "water_content_source"], "inferred_aqueous")
        self.assertAlmostEqual(float(echem.loc["ROW_AQ_PLAIN", "water_content_wt_pct"]), 100.0)
        self.assertAlmostEqual(float(echem.loc["ROW_AQ_PLAIN", "quality_score"]), 0.90, places=6)

        for rid in ("ROW_NONAQ_OVERRIDE_AQ", "ROW_DMSO_OVERRIDE_NONE"):
            self.assertEqual(echem.loc[rid, "water_content_source"], "inferred_aqueous")
            self.assertAlmostEqual(float(echem.loc[rid, "water_content_wt_pct"]), 100.0)
            self.assertEqual(echem.loc[rid, "solvent_regime"], "aqueous")
            self.assertAlmostEqual(float(echem.loc[rid, "quality_score"]), 0.90, places=6)

        self.assertEqual(sol.loc["SOL_ACID_NO_PH", "solvent_subclass"], "acidic")
        self.assertEqual(echem.loc["ROW_ACID_NO_PH", "solvent_subclass"], "acidic")
        self.assertEqual(sol.loc["SOL_PH_EFF_ONLY", "solvent_subclass"], "acidic")
        self.assertEqual(echem.loc["ROW_PH_EFF_ONLY", "solvent_subclass"], "acidic")

        self.assertEqual(echem.loc["ROW_DMSO_OVERRIDE_MEOH", "water_content_source"], "inferred_regime_default")
        self.assertEqual(echem.loc["ROW_DMSO_OVERRIDE_MEOH", "solvent_regime"], "non_aqueous")
        self.assertEqual(echem.loc["ROW_DMSO_OVERRIDE_MEOH", "solvent_subclass"], "alcohol")
        self.assertAlmostEqual(
            float(echem.loc["ROW_DMSO_OVERRIDE_MEOH", "quality_score"]),
            0.90 * INFERRED_REGIME_DEFAULT_WATER_WEIGHT_FACTOR,
            places=6,
        )

        # Verify pH_effective-only DataFrame (no pH column) agrees on acidic subclass in both tables
        db_ph_eff = HarmonizedDatabase.from_raw_tables(
            pd.DataFrame(
                [
                    {
                        "record_id": "R_PHEFF",
                        "metal": "Cu",
                        "solution_id": "SOL_PHEFF",
                        "pH_effective": 2.0,
                        "temperature_C": 25.0,
                        "Ecorr_V_SHE": 0.21,
                        "icorr_uA_cm2": 1.1,
                        "beta_a_mV_dec": 60.0,
                        "beta_c_mV_dec": 120.0,
                    }
                ]
            ),
            supplement_missing_core_metals=False,
        )
        self.assertEqual(db_ph_eff.solution_properties.iloc[0]["solvent_subclass"], "acidic")
        self.assertEqual(db_ph_eff.electrochemical_database.iloc[0]["solvent_subclass"], "acidic")

        expected_discounted = 0.90 * INFERRED_REGIME_DEFAULT_WATER_WEIGHT_FACTOR
        for rid in ("ROW_AQ_OVERRIDE_EG", "ROW_DMSO_OVERRIDE_EG", "ROW_COMP_MM"):
            self.assertEqual(echem.loc[rid, "water_content_source"], "inferred_regime_default")
            self.assertAlmostEqual(float(echem.loc[rid, "water_content_wt_pct"]), 50.0)
            self.assertEqual(echem.loc[rid, "solvent_regime"], "semi_aqueous")
            self.assertEqual(echem.loc[rid, "solvent_subclass"], "glycol")
            self.assertAlmostEqual(float(echem.loc[rid, "quality_score"]), expected_discounted, places=6)


if __name__ == "__main__":
    unittest.main()
