"""Behavioral tests for Issue 04: Area-Weighted Bimetallic Galvanic Mixed-Potential Solver."""

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

from corrosion_simulator.database import HarmonizedDatabase
from corrosion_simulator.galvanic import (
    GalvanicCouplingResult,
    MonteCarloGalvanicResult,
    plot_galvanic_tafel_overlay,
    propagate_galvanic_uncertainty_monte_carlo,
    solve_galvanic_couple,
    solve_galvanic_couple_from_database,
)
from corrosion_simulator.ml_predictor import (
    ApplicabilityDomainResult,
    MLParameterPrediction,
    ParameterEstimate,
)
from corrosion_simulator.physics import (
    PolarizationParameters,
    calculate_faraday_corrosion_rate,
    wagner_traud_anodic_current,
    wagner_traud_net_current,
)


class TestGalvanicMixedPotentialSolver(unittest.TestCase):
    """Seam 2: Area-weighted bimetallic mixed-potential root-finding and Faraday acceleration."""

    def setUp(self) -> None:
        # Benchmark semiconductor wet-clean couple: Co (active) vs. Cu (noble) in H2O2 + citric acid
        self.co_params = PolarizationParameters(
            Ecorr_V_SHE=-0.120,
            icorr_uA_cm2=12.0,
            beta_a_mV_dec=65.0,
            beta_c_mV_dec=120.0,
            metal="Co",
            solution_id="SOL_H2O2_CA",
            temperature_C=25.0,
        )
        self.cu_params = PolarizationParameters(
            Ecorr_V_SHE=0.285,
            icorr_uA_cm2=2.3,
            beta_a_mV_dec=60.0,
            beta_c_mV_dec=115.0,
            metal="Cu",
            solution_id="SOL_H2O2_CA",
            temperature_C=25.0,
        )

    def test_mixed_potential_root_is_strictly_bounded_and_conserves_current(self) -> None:
        res = solve_galvanic_couple(self.cu_params, self.co_params, cathode_to_anode_area_ratio=1.0)
        self.assertIsInstance(res, GalvanicCouplingResult)

        # Active vs Noble identification must be order-independent
        self.assertEqual(res.active_metal, "Co")
        self.assertEqual(res.noble_metal, "Cu")

        # E_galv strictly inside (E_corr,active, E_corr,noble)
        self.assertGreater(res.mixed_potential_V_SHE, self.co_params.Ecorr_V_SHE)
        self.assertLess(res.mixed_potential_V_SHE, self.cu_params.Ecorr_V_SHE)

        # Independent check of area-weighted current conservation < 1e-6 uA/cm2
        i_net_active = wagner_traud_net_current(res.mixed_potential_V_SHE, self.co_params)
        i_net_noble = wagner_traud_net_current(res.mixed_potential_V_SHE, self.cu_params)
        balance = i_net_active + 1.0 * i_net_noble
        self.assertLess(abs(balance), 1e-6)
        self.assertLess(abs(res.current_balance_residual_uA_cm2), 1e-6)

        # i_galv equals i_anodic,active(E_galv) and Acceleration Factor > 1.0
        expected_ig = wagner_traud_anodic_current(res.mixed_potential_V_SHE, self.co_params)
        self.assertAlmostEqual(res.galvanic_current_density_uA_cm2, expected_ig, places=8)
        self.assertGreater(res.acceleration_factor, 1.0)
        self.assertAlmostEqual(
            res.acceleration_factor,
            res.galvanic_current_density_uA_cm2 / self.co_params.icorr_uA_cm2,
            places=8,
        )

        # Coupled Faraday CR_0 matches Faraday calculation at i_galv
        expected_faraday = calculate_faraday_corrosion_rate("Co", expected_ig)
        self.assertAlmostEqual(
            res.coupled_faraday.corrosion_rate_nm_min,
            expected_faraday.corrosion_rate_nm_min,
            places=8,
        )
        self.assertAlmostEqual(
            res.coupled_faraday.corrosion_rate_nm_min / res.uncoupled_faraday.corrosion_rate_nm_min,
            res.acceleration_factor,
            places=8,
        )

    def test_acceleration_factor_increases_monotonically_with_cathode_to_anode_area_ratio(self) -> None:
        ratios = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        results = [
            solve_galvanic_couple(self.co_params, self.cu_params, cathode_to_anode_area_ratio=r)
            for r in ratios
        ]
        mixed_potentials = [r.mixed_potential_V_SHE for r in results]
        accel_factors = [r.acceleration_factor for r in results]
        coupled_crs = [r.coupled_faraday.corrosion_rate_nm_min for r in results]

        for r, res in zip(ratios, results):
            i_net_a = wagner_traud_net_current(res.mixed_potential_V_SHE, self.co_params)
            i_net_n = wagner_traud_net_current(res.mixed_potential_V_SHE, self.cu_params)
            self.assertLess(abs(i_net_a + r * i_net_n), 1e-6)
            self.assertGreater(res.acceleration_factor, 1.0)

        self.assertTrue(all(x < y for x, y in zip(mixed_potentials, mixed_potentials[1:])))
        self.assertTrue(all(x < y for x, y in zip(accel_factors, accel_factors[1:])))
        self.assertTrue(all(x < y for x, y in zip(coupled_crs, coupled_crs[1:])))

    def test_validation_rejects_invalid_area_ratio_same_metal_or_mismatched_conditions(self) -> None:
        with self.assertRaises(ValueError):
            solve_galvanic_couple(self.co_params, self.cu_params, cathode_to_anode_area_ratio=0.0)
        with self.assertRaises(ValueError):
            solve_galvanic_couple(self.co_params, self.cu_params, cathode_to_anode_area_ratio=-2.0)
        with self.assertRaises(ValueError):
            solve_galvanic_couple(self.co_params, self.co_params)

        mismatched_sol = PolarizationParameters(
            Ecorr_V_SHE=0.285,
            icorr_uA_cm2=2.3,
            beta_a_mV_dec=60.0,
            beta_c_mV_dec=115.0,
            metal="Cu",
            solution_id="SOL_OTHER",
            temperature_C=25.0,
        )
        with self.assertRaises(ValueError):
            solve_galvanic_couple(self.co_params, mismatched_sol)


class TestGalvanicDatabaseAndMonteCarlo(unittest.TestCase):
    """Seam 2: Database coupling helper and Monte Carlo uncertainty propagation."""

    def test_solve_galvanic_couple_from_database_same_solution(self) -> None:
        db = HarmonizedDatabase.from_seed()
        sol_id = str(db.solution_properties.iloc[0]["solution_id"])
        res = solve_galvanic_couple_from_database(
            db,
            "Cu",
            "Co",
            solution_id=sol_id,
            temperature_C=25.0,
            cathode_to_anode_area_ratio=2.5,
        )
        self.assertEqual(res.active_metal, "Co")
        self.assertEqual(res.noble_metal, "Cu")
        self.assertEqual(res.solution_id, sol_id)
        self.assertGreater(res.acceleration_factor, 1.0)
        self.assertLess(abs(res.current_balance_residual_uA_cm2), 1e-6)

    def test_monte_carlo_galvanic_uncertainty_produces_ordered_percentiles_and_physical_bounds(self) -> None:
        ad_dummy = ApplicabilityDomainResult(
            metal="Co",
            confidence_tier="High",
            in_bounding_box=True,
            out_of_box_features=(),
            unseen_categories=(),
            knn_mean_distance=0.1,
            knn_k=3,
            n_metal_training_samples=10,
            warnings=(),
            experimental_recommendations=(),
        )
        co_pred = MLParameterPrediction(
            metal="Co",
            Ecorr_V_SHE=ParameterEstimate(mean=-0.12, std=0.025),
            log10_icorr_uA_cm2=ParameterEstimate(mean=math.log10(12.0), std=0.15),
            icorr_uA_cm2_median=12.0,
            beta_a_mV_dec=ParameterEstimate(mean=65.0, std=8.0),
            beta_c_mV_dec=ParameterEstimate(mean=120.0, std=12.0),
            model_family="random_forest",
            applicability=ad_dummy,
        )
        cu_pred = MLParameterPrediction(
            metal="Cu",
            Ecorr_V_SHE=ParameterEstimate(mean=0.285, std=0.02),
            log10_icorr_uA_cm2=ParameterEstimate(mean=math.log10(2.3), std=0.15),
            icorr_uA_cm2_median=2.3,
            beta_a_mV_dec=ParameterEstimate(mean=60.0, std=7.0),
            beta_c_mV_dec=ParameterEstimate(mean=115.0, std=10.0),
            model_family="random_forest",
            applicability=ad_dummy,
        )

        mc = propagate_galvanic_uncertainty_monte_carlo(
            cu_pred,
            co_pred,
            cathode_to_anode_area_ratio=2.0,
            n_draws=200,
            seed=123,
        )
        self.assertIsInstance(mc, MonteCarloGalvanicResult)
        self.assertEqual(mc.active_metal, "Co")
        self.assertEqual(mc.noble_metal, "Cu")
        self.assertEqual(mc.n_draws, 200)
        self.assertEqual(len(mc.mixed_potential_samples_V_SHE), 200)

        # Physical bounds on every draw
        self.assertTrue(np.all(mc.galvanic_current_samples_uA_cm2 > 0.0))
        self.assertTrue(np.all(mc.coupled_cr_nm_min_samples >= 0.0))
        self.assertTrue(np.all(mc.acceleration_factor_samples >= 1.0))

        # Summary statistics present and monotonically ordered p05 <= p50 <= p95 with positive spread
        for summary in (
            mc.mixed_potential_summary,
            mc.galvanic_current_summary,
            mc.acceleration_factor_summary,
            mc.coupled_cr_nm_min_summary,
        ):
            self.assertEqual(set(summary.keys()), {"mean", "std", "p05", "p50", "p95"})
            self.assertLessEqual(summary["p05"], summary["p50"])
            self.assertLessEqual(summary["p50"], summary["p95"])
            self.assertGreater(summary["std"], 0.0)

        # Overlapping E_corr distributions must still guarantee Acceleration Factor >= 1.0 on every draw
        close_a = MLParameterPrediction(
            metal="Co",
            Ecorr_V_SHE=ParameterEstimate(mean=0.10, std=0.08),
            log10_icorr_uA_cm2=ParameterEstimate(mean=math.log10(5.0), std=0.1),
            icorr_uA_cm2_median=5.0,
            beta_a_mV_dec=ParameterEstimate(mean=60.0, std=5.0),
            beta_c_mV_dec=ParameterEstimate(mean=120.0, std=5.0),
            model_family="random_forest",
            applicability=ad_dummy,
        )
        close_b = MLParameterPrediction(
            metal="Cu",
            Ecorr_V_SHE=ParameterEstimate(mean=0.12, std=0.08),
            log10_icorr_uA_cm2=ParameterEstimate(mean=math.log10(5.0), std=0.1),
            icorr_uA_cm2_median=5.0,
            beta_a_mV_dec=ParameterEstimate(mean=60.0, std=5.0),
            beta_c_mV_dec=ParameterEstimate(mean=120.0, std=5.0),
            model_family="random_forest",
            applicability=ad_dummy,
        )
        mc_overlap = propagate_galvanic_uncertainty_monte_carlo(close_a, close_b, n_draws=200, seed=99)
        self.assertTrue(np.all(mc_overlap.acceleration_factor_samples >= 1.0))


class TestGalvanicTafelOverlayPlot(unittest.TestCase):
    """Seam 2: Overlaid bimetallic Tafel plot rendering with E_galv operating line and shift."""

    def test_plot_galvanic_tafel_overlay_renders_both_metals_and_operating_line(self) -> None:
        co = PolarizationParameters(
            Ecorr_V_SHE=-0.120,
            icorr_uA_cm2=12.0,
            beta_a_mV_dec=65.0,
            beta_c_mV_dec=120.0,
            metal="Co",
        )
        cu = PolarizationParameters(
            Ecorr_V_SHE=0.285,
            icorr_uA_cm2=2.3,
            beta_a_mV_dec=60.0,
            beta_c_mV_dec=115.0,
            metal="Cu",
        )
        res = solve_galvanic_couple(co, cu, cathode_to_anode_area_ratio=3.0)
        fig, ax = plot_galvanic_tafel_overlay(res)
        try:
            self.assertIn("log10", ax.get_xlabel().lower())
            self.assertIn("she", ax.get_ylabel().lower())
            legend = ax.get_legend()
            self.assertIsNotNone(legend)
            labels = [t.get_text() for t in legend.get_texts()]
            self.assertTrue(any("Co" in lbl for lbl in labels))
            self.assertTrue(any("Cu" in lbl for lbl in labels))
            self.assertTrue(any("E_galv" in lbl or "Mixed Potential" in lbl for lbl in labels))
            # Check annotations mention E_galv / i_galv / AF
            annotation_texts = [t.get_text() for t in ax.texts]
            self.assertTrue(any("E_galv" in txt and "i_galv" in txt for txt in annotation_texts))
        finally:
            plt.close(fig)


class TestGalvanicNotebookArtifact(unittest.TestCase):
    """Verify the bimetallic galvanic mixed-potential demonstration notebook exists and is executed."""

    def test_galvanic_notebook_is_executed_with_outputs(self) -> None:
        nb_path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "04_bimetallic_galvanic_mixed_potential.ipynb"
        )
        self.assertTrue(nb_path.exists(), f"Missing notebook: {nb_path}")
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 4)
        self.assertTrue(all(len(c.get("outputs", [])) > 0 for c in code_cells))


if __name__ == "__main__":
    unittest.main()
