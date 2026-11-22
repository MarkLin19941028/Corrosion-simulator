"""Behavioral tests for Phase 1 polarization, Faraday corrosion rate, and galvanic screening."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from corrosion_simulator.database import CORE_METALS, DEFAULT_DISSOLUTION_VALENCE, ExposureCondition, HarmonizedDatabase
from corrosion_simulator.physics import (
    FaradayCorrosionResult,
    GalvanicDrivingForceResult,
    PolarizationParameters,
    calculate_faraday_corrosion_rate,
    generate_polarization_curve,
    plot_tafel_curve,
    screen_galvanic_driving_force,
    screen_galvanic_pair_from_database,
    wagner_traud_anodic_current,
    wagner_traud_cathodic_current,
    wagner_traud_net_current,
)


class TestWagnerTraudPolarization(unittest.TestCase):
    """Seam 2 slice 1: four-parameter activation-controlled Wagner-Traud invariants."""

    def test_wagner_traud_invariants_at_ecorr_and_one_decade_shifts(self) -> None:
        params = PolarizationParameters(
            Ecorr_V_SHE=0.285,
            icorr_uA_cm2=2.3,
            beta_a_mV_dec=60.0,
            beta_c_mV_dec=120.0,
            metal="Cu",
        )

        # At E = E_corr: i_anodic == i_cathodic == i_corr, i_net == 0
        self.assertAlmostEqual(wagner_traud_anodic_current(0.285, params), 2.3, places=9)
        self.assertAlmostEqual(wagner_traud_cathodic_current(0.285, params), 2.3, places=9)
        self.assertAlmostEqual(wagner_traud_net_current(0.285, params), 0.0, places=9)

        # Shifting E by +beta_a (0.060 V) scales i_anodic by 10x -> 23.0 uA/cm2
        self.assertAlmostEqual(
            wagner_traud_anodic_current(0.285 + 0.060, params), 23.0, places=9
        )
        # Shifting E by -beta_c (-0.120 V) scales i_cathodic by 10x -> 23.0 uA/cm2
        self.assertAlmostEqual(
            wagner_traud_cathodic_current(0.285 - 0.120, params), 23.0, places=9
        )

    def test_generate_polarization_curve_from_manual_and_lookup_inputs(self) -> None:
        db = HarmonizedDatabase.from_seed()
        lookup = db.lookup_exposure_condition(
            ExposureCondition(
                metal="Cu",
                pH=4.0,
                oxidizer_type="H2O2",
                oxidizer_concentration=1.0,
                temperature_C=25.0,
            )
        )

        curve = generate_polarization_curve(lookup, num_points=201)
        self.assertEqual(len(curve.potential_V_SHE), 201)
        self.assertTrue(np.all(curve.anodic_current_uA_cm2 > 0.0))
        self.assertTrue(np.all(curve.cathodic_current_uA_cm2 > 0.0))

        # E_corr must be included on the grid with exact net current == 0
        idx_ecorr = int(np.argmin(np.abs(curve.potential_V_SHE - lookup.Ecorr_V_SHE)))
        self.assertAlmostEqual(
            float(curve.potential_V_SHE[idx_ecorr]), lookup.Ecorr_V_SHE, places=12
        )
        self.assertAlmostEqual(
            float(curve.anodic_current_uA_cm2[idx_ecorr]), lookup.icorr_uA_cm2, places=9
        )
        self.assertAlmostEqual(
            float(curve.cathodic_current_uA_cm2[idx_ecorr]), lookup.icorr_uA_cm2, places=9
        )
        self.assertAlmostEqual(float(curve.net_current_uA_cm2[idx_ecorr]), 0.0, places=9)

        # Net current is strictly monotonically increasing with potential E
        self.assertTrue(np.all(np.diff(curve.net_current_uA_cm2) > 0.0))

    def test_invalid_polarization_parameters_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            PolarizationParameters(
                Ecorr_V_SHE=0.2, icorr_uA_cm2=0.0, beta_a_mV_dec=60.0, beta_c_mV_dec=120.0
            )
        with self.assertRaises(ValueError):
            PolarizationParameters(
                Ecorr_V_SHE=0.2, icorr_uA_cm2=2.0, beta_a_mV_dec=-60.0, beta_c_mV_dec=120.0
            )
        with self.assertRaises(ValueError):
            PolarizationParameters(
                Ecorr_V_SHE=0.2, icorr_uA_cm2=2.0, beta_a_mV_dec=60.0, beta_c_mV_dec=0.0
            )


class TestFaradayCorrosionRate(unittest.TestCase):
    """Seam 2 slice 2: Faraday's law conversion from i_corr to CR_0 and linear Corrosion Depth."""

    def test_faraday_conversion_matches_astm_g102_hand_calculations(self) -> None:
        # Cu (M=63.55 g/mol, rho=8.96 g/cm3, z=2) at 2.3 uA/cm2 for 60 s:
        # CR_nm_min = (600 * 2.3 * 63.55) / (2 * 96485.33212 * 8.96) = 0.05072187 nm/min
        # CR_mm_yr  = CR_nm_min * 0.5256 = 0.02665942 mm/yr
        res_cu = calculate_faraday_corrosion_rate("Cu", 2.3, immersion_time_s=60.0)
        self.assertIsInstance(res_cu, FaradayCorrosionResult)
        self.assertEqual(res_cu.dissolution_valence, 2)
        self.assertAlmostEqual(res_cu.corrosion_rate_nm_min, 0.05072187, places=6)
        self.assertAlmostEqual(res_cu.corrosion_rate_mm_yr, 0.02665942, places=6)
        self.assertAlmostEqual(res_cu.corrosion_depth_nm, 0.05072187, places=6)

        # W (M=183.84 g/mol, rho=19.25 g/cm3, z=6) at 10.0 uA/cm2 for 120 s:
        # CR_nm_min = (600 * 10.0 * 183.84) / (6 * 96485.33212 * 19.25) = 0.09898011 nm/min
        res_w = calculate_faraday_corrosion_rate("W", 10.0, immersion_time_s=120.0)
        self.assertEqual(res_w.dissolution_valence, 6)
        self.assertAlmostEqual(res_w.corrosion_rate_nm_min, 0.09898011, places=6)
        self.assertAlmostEqual(res_w.corrosion_depth_nm, 0.19796022, places=6)

    def test_faraday_valence_override_and_all_core_metal_defaults(self) -> None:
        res_cu_z2 = calculate_faraday_corrosion_rate("Cu", 2.3)
        res_cu_z1 = calculate_faraday_corrosion_rate("Cu", 2.3, valence=1)
        self.assertEqual(res_cu_z1.dissolution_valence, 1)
        self.assertAlmostEqual(
            res_cu_z1.corrosion_rate_nm_min, 2.0 * res_cu_z2.corrosion_rate_nm_min, places=9
        )
        self.assertIsNone(res_cu_z2.corrosion_depth_nm)

        for metal in CORE_METALS:
            res = calculate_faraday_corrosion_rate(metal, 5.0, immersion_time_s=60.0)
            self.assertEqual(res.dissolution_valence, DEFAULT_DISSOLUTION_VALENCE[metal])
            self.assertGreater(res.corrosion_rate_nm_min, 0.0)
            self.assertGreater(res.corrosion_rate_mm_yr, 0.0)
            self.assertAlmostEqual(
                res.corrosion_rate_mm_yr / res.corrosion_rate_nm_min, 0.5256, places=6
            )

    def test_faraday_zero_and_invalid_inputs(self) -> None:
        res_zero = calculate_faraday_corrosion_rate("Co", 0.0, immersion_time_s=60.0)
        self.assertEqual(res_zero.corrosion_rate_nm_min, 0.0)
        self.assertEqual(res_zero.corrosion_rate_mm_yr, 0.0)
        self.assertEqual(res_zero.corrosion_depth_nm, 0.0)

        with self.assertRaises(ValueError):
            calculate_faraday_corrosion_rate("Cu", -1.0)
        with self.assertRaises(ValueError):
            calculate_faraday_corrosion_rate("Cu", 2.0, valence=0)
        with self.assertRaises(ValueError):
            calculate_faraday_corrosion_rate("Cu", 2.0, immersion_time_s=-5.0)
        with self.assertRaises(ValueError):
            calculate_faraday_corrosion_rate("UnknownMetal", 2.0)


class TestGalvanicDrivingForceScreening(unittest.TestCase):
    """Seam 2 slice 3: bimetallic Galvanic Driving Force Risk classification."""

    def test_risk_tiers_and_active_noble_identification_are_order_independent(self) -> None:
        # Low: |Delta E_corr| < 0.1 V (0.030 V)
        low = screen_galvanic_driving_force("W", 0.140, "Mo", 0.110)
        self.assertIsInstance(low, GalvanicDrivingForceResult)
        self.assertEqual(low.active_metal, "Mo")
        self.assertEqual(low.noble_metal, "W")
        self.assertAlmostEqual(low.delta_Ecorr_V, 0.030, places=6)
        self.assertEqual(low.risk_tier, "Low")

        # Medium: 0.1 V <= |Delta E_corr| <= 0.3 V (0.100 V and 0.300 V inclusive)
        med_lower = screen_galvanic_driving_force("Cu", 0.200, "W", 0.100)
        self.assertEqual(med_lower.risk_tier, "Medium")
        self.assertEqual(med_lower.active_metal, "W")
        self.assertEqual(med_lower.noble_metal, "Cu")

        med_upper = screen_galvanic_driving_force("W", 0.100, "Ru", 0.400)
        self.assertEqual(med_upper.risk_tier, "Medium")
        self.assertEqual(med_upper.active_metal, "W")
        self.assertEqual(med_upper.noble_metal, "Ru")

        # High: |Delta E_corr| > 0.3 V (0.380 V), order-independent
        high_ab = screen_galvanic_driving_force("Cu", 0.285, "Co", -0.095)
        high_ba = screen_galvanic_driving_force("Co", -0.095, "Cu", 0.285)
        for res in (high_ab, high_ba):
            self.assertEqual(res.active_metal, "Co")
            self.assertEqual(res.noble_metal, "Cu")
            self.assertAlmostEqual(res.delta_Ecorr_V, 0.380, places=6)
            self.assertEqual(res.risk_tier, "High")

    def test_screen_galvanic_pair_from_database_same_solution_and_temperature(self) -> None:
        db = HarmonizedDatabase.from_seed()
        res = screen_galvanic_pair_from_database(
            db,
            "Cu",
            "Co",
            pH=4.0,
            oxidizer_type="H2O2",
            oxidizer_concentration=1.0,
            temperature_C=25.0,
        )
        self.assertEqual(res.active_metal, "Co")
        self.assertEqual(res.noble_metal, "Cu")
        self.assertAlmostEqual(res.delta_Ecorr_V, 0.380, places=6)
        self.assertEqual(res.risk_tier, "High")
        self.assertEqual(res.solution_id, "SOL_SEED_H2O2_CITRIC_PH4")
        self.assertAlmostEqual(res.temperature_C, 25.0, places=6)

        with self.assertRaises(ValueError):
            screen_galvanic_driving_force("Cu", 0.285, "Cu", 0.285)


class TestTafelPlotGeneration(unittest.TestCase):
    """Seam 2 slice 4: Tafel plot E vs. log10|i| visualization and annotations."""

    def test_plot_tafel_curve_annotates_ecorr_icorr_and_branches(self) -> None:
        params = PolarizationParameters(
            Ecorr_V_SHE=0.285,
            icorr_uA_cm2=2.3,
            beta_a_mV_dec=58.0,
            beta_c_mV_dec=118.0,
            metal="Cu",
        )
        fig, ax = plot_tafel_curve(params)
        try:
            self.assertIn("log", ax.get_xlabel().lower())
            self.assertIn("v vs. she", ax.get_ylabel().lower())

            legend_labels = [t.get_text().lower() for t in ax.get_legend().get_texts()]
            self.assertTrue(any("anodic" in lbl for lbl in legend_labels))
            self.assertTrue(any("cathodic" in lbl for lbl in legend_labels))
            self.assertTrue(any("net" in lbl for lbl in legend_labels))

            annotation_texts = " ".join(t.get_text() for t in ax.texts)
            self.assertIn("E_corr", annotation_texts)
            self.assertIn("i_corr", annotation_texts)
        finally:
            plt.close(fig)


class TestPhase1NotebookArtifact(unittest.TestCase):
    """Verify the Phase 1 Tafel demonstration notebook exists and contains executed outputs."""

    def test_phase1_notebook_is_executed_with_outputs(self) -> None:
        nb_path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "02_phase1_tafel_and_corrosion_calculations.ipynb"
        )
        self.assertTrue(nb_path.exists(), f"Missing notebook: {nb_path}")
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 5)
        self.assertTrue(all(len(c.get("outputs", [])) > 0 for c in code_cells))


if __name__ == "__main__":
    unittest.main()
