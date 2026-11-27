"""Behavioral tests for Issue 05: Time-Resolved Corrosion Kinetics, Time to Threshold & Thickness Loss Risk."""

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

from corrosion_simulator.galvanic import (
    propagate_galvanic_uncertainty_monte_carlo,
    solve_galvanic_couple,
)
from corrosion_simulator.kinetics import (
    MonteCarloKineticsResult,
    TimeResolvedKineticsResult,
    classify_thickness_loss_risk_tier,
    compare_kinetic_models,
    compute_corrosion_depth,
    compute_thickness_loss_risk_score,
    compute_time_to_threshold,
    plot_kinetics_and_risk,
    propagate_kinetics_uncertainty_monte_carlo,
    select_kinetic_model_from_pourbaix,
    simulate_time_resolved_kinetics,
)
from corrosion_simulator.physics import PolarizationParameters


class TestTimeResolvedKineticsModels(unittest.TestCase):
    """Seam 2: Linear, Passivation, and Power-law depth invariants and slope anchoring."""

    def test_all_models_start_at_zero_are_monotonic_and_anchor_to_cr0(self) -> None:
        cr0_nm_min = 0.60  # 0.01 nm/s
        times_s = np.linspace(0.0, 600.0, 121)

        for model in ("linear", "passivation", "power_law"):
            depths = compute_corrosion_depth(
                cr0_nm_min,
                times_s,
                model=model,
                d_max_nm=5.0,
                power_law_n=0.6,
                t_ref_s=60.0,
            )
            self.assertAlmostEqual(float(depths[0]), 0.0, places=12)
            self.assertTrue(np.all(depths >= 0.0))
            self.assertTrue(np.all(np.diff(depths) >= 0.0))

        # Linear and Passivation share the exact initial slope dd/dt|_{t=0} = CR_0 / 60 (nm/s)
        dt = 1e-4
        d_lin_dt = float(compute_corrosion_depth(cr0_nm_min, dt, model="linear"))
        d_pas_dt = float(compute_corrosion_depth(cr0_nm_min, dt, model="passivation", d_max_nm=5.0))
        expected_slope_nm_s = cr0_nm_min / 60.0
        self.assertAlmostEqual(d_lin_dt / dt, expected_slope_nm_s, places=6)
        self.assertAlmostEqual(d_pas_dt / dt, expected_slope_nm_s, places=5)

        # Passivation is bounded above by d_max_nm
        d_pas_inf = float(
            compute_corrosion_depth(cr0_nm_min, 1e6, model="passivation", d_max_nm=3.5)
        )
        self.assertLessEqual(d_pas_inf, 3.5)
        self.assertAlmostEqual(d_pas_inf, 3.5, places=6)

        # Power-law equals CR_0 * t_ref at t = t_ref
        d_pow_ref = float(
            compute_corrosion_depth(cr0_nm_min, 60.0, model="power_law", power_law_n=0.6, t_ref_s=60.0)
        )
        self.assertAlmostEqual(d_pow_ref, cr0_nm_min * 1.0, places=9)

    def test_pourbaix_regime_selection_maps_regimes_to_default_models(self) -> None:
        self.assertEqual(select_kinetic_model_from_pourbaix(regime="Active").model, "linear")
        self.assertEqual(select_kinetic_model_from_pourbaix(regime="Passivation").model, "passivation")
        self.assertEqual(select_kinetic_model_from_pourbaix(regime="Transitional").model, "power_law")
        self.assertEqual(select_kinetic_model_from_pourbaix(regime="Inhibited").model, "power_law")

        # Metal + pH automatic Pourbaix classification:
        # Ti at pH 6 is strongly passivating (TiO2, stable pH 2-12, tendency 0.90) -> passivation
        ti_sel = select_kinetic_model_from_pourbaix(metal="Ti", pH=6.0)
        self.assertEqual(ti_sel.model, "passivation")

        # Co at pH 2 is active dissolution -> linear
        co_sel = select_kinetic_model_from_pourbaix(metal="Co", pH=2.0)
        self.assertEqual(co_sel.model, "linear")

        # Co at pH 2 with corrosion inhibitor -> power_law
        co_inh = select_kinetic_model_from_pourbaix(metal="Co", pH=2.0, inhibitor_present=True)
        self.assertEqual(co_inh.model, "power_law")

    def test_time_to_threshold_accuracy_and_unreached_passivation_limit(self) -> None:
        cr0_nm_min = 0.60  # 0.01 nm/s
        d_crit = 1.0

        # Linear: t_crit = 1.0 / 0.01 = 100.0 s
        t_lin = compute_time_to_threshold(cr0_nm_min, d_crit_nm=d_crit, model="linear")
        self.assertAlmostEqual(t_lin, 100.0, places=9)
        self.assertAlmostEqual(
            float(compute_corrosion_depth(cr0_nm_min, t_lin, model="linear")), d_crit, places=9
        )

        # Passivation (d_max = 5.0 > d_crit = 1.0): t_crit = -(5/0.01)*ln(1 - 1/5)
        t_pas = compute_time_to_threshold(
            cr0_nm_min, d_crit_nm=d_crit, model="passivation", d_max_nm=5.0
        )
        expected_t_pas = -(5.0 / 0.01) * math.log(1.0 - 1.0 / 5.0)
        self.assertAlmostEqual(t_pas, expected_t_pas, places=9)
        self.assertAlmostEqual(
            float(compute_corrosion_depth(cr0_nm_min, t_pas, model="passivation", d_max_nm=5.0)),
            d_crit,
            places=9,
        )

        # Passivation unreached when d_max <= d_crit -> +inf
        t_unreached_eq = compute_time_to_threshold(
            cr0_nm_min, d_crit_nm=1.0, model="passivation", d_max_nm=1.0
        )
        t_unreached_lt = compute_time_to_threshold(
            cr0_nm_min, d_crit_nm=1.0, model="passivation", d_max_nm=0.75
        )
        self.assertTrue(math.isinf(t_unreached_eq))
        self.assertTrue(math.isinf(t_unreached_lt))

        # Power-law: d(t_crit) == d_crit
        t_pow = compute_time_to_threshold(
            cr0_nm_min, d_crit_nm=d_crit, model="power_law", power_law_n=0.6, t_ref_s=60.0
        )
        self.assertAlmostEqual(
            float(
                compute_corrosion_depth(
                    cr0_nm_min, t_pow, model="power_law", power_law_n=0.6, t_ref_s=60.0
                )
            ),
            d_crit,
            places=9,
        )

    def test_thickness_loss_risk_score_and_discrete_tier_boundaries(self) -> None:
        d_crit = 1.0
        self.assertAlmostEqual(compute_thickness_loss_risk_score(0.0, d_crit), 0.0, places=12)
        self.assertAlmostEqual(compute_thickness_loss_risk_score(d_crit, d_crit), 0.5, places=12)
        r_large = compute_thickness_loss_risk_score(100.0, d_crit)
        self.assertGreaterEqual(r_large, 0.0)
        self.assertLessEqual(r_large, 1.0)

        # Discrete tier boundaries: Low < 0.8, Medium [0.8, 2.0), High [2.0, 4.0), Critical >= 4.0
        self.assertEqual(classify_thickness_loss_risk_tier(0.0, d_crit), "Low")
        self.assertEqual(classify_thickness_loss_risk_tier(0.799, d_crit), "Low")
        self.assertEqual(classify_thickness_loss_risk_tier(0.8, d_crit), "Medium")
        self.assertEqual(classify_thickness_loss_risk_tier(1.999, d_crit), "Medium")
        self.assertEqual(classify_thickness_loss_risk_tier(2.0, d_crit), "High")
        self.assertEqual(classify_thickness_loss_risk_tier(3.999, d_crit), "High")
        self.assertEqual(classify_thickness_loss_risk_tier(4.0, d_crit), "Critical")
        self.assertEqual(classify_thickness_loss_risk_tier(10.0, d_crit), "Critical")

    def test_simulate_time_resolved_kinetics_returns_complete_trajectory(self) -> None:
        times = np.linspace(0.0, 300.0, 61)
        res = simulate_time_resolved_kinetics(0.60, times, model="linear", d_crit_nm=1.0)
        self.assertIsInstance(res, TimeResolvedKineticsResult)
        self.assertEqual(res.model, "linear")
        self.assertTrue(res.threshold_reachable)
        self.assertAlmostEqual(res.time_to_threshold_s, 100.0, places=8)
        self.assertAlmostEqual(res.final_depth_nm, 3.0, places=8)
        self.assertEqual(res.final_risk_tier, "High")
        df = res.to_dataframe()
        self.assertEqual(len(df), 61)
        self.assertIn("depth_nm", df.columns)
        self.assertIn("risk_score", df.columns)
        self.assertIn("risk_tier", df.columns)


class TestKineticsMonteCarloAndPlotting(unittest.TestCase):
    """Seam 2: Multi-model comparison, MC uncertainty propagation, and depth/risk plotting."""

    def test_compare_kinetic_models_from_galvanic_coupling_result(self) -> None:
        co = PolarizationParameters(
            Ecorr_V_SHE=-0.120, icorr_uA_cm2=12.0, beta_a_mV_dec=65.0, beta_c_mV_dec=120.0, metal="Co"
        )
        cu = PolarizationParameters(
            Ecorr_V_SHE=0.285, icorr_uA_cm2=2.3, beta_a_mV_dec=60.0, beta_c_mV_dec=115.0, metal="Cu"
        )
        galv = solve_galvanic_couple(co, cu, cathode_to_anode_area_ratio=2.0)
        times = np.linspace(0.0, 180.0, 37)

        comparison = compare_kinetic_models(galv, times, d_crit_nm=1.0, d_max_nm=5.0, power_law_n=0.6)
        self.assertEqual(set(comparison.keys()), {"linear", "passivation", "power_law"})
        for model_name, res in comparison.items():
            self.assertEqual(res.model, model_name)
            self.assertEqual(res.metal, "Co")
            self.assertAlmostEqual(res.cr0_nm_min, galv.coupled_cr_nm_min, places=9)
            self.assertTrue(np.all(np.diff(res.depth_nm) >= 0.0))

    def test_monte_carlo_kinetics_propagates_depth_risk_and_time_to_threshold_bands(self) -> None:
        rng = np.random.default_rng(7)
        cr0_samples = np.exp(rng.normal(math.log(0.60), 0.25, size=250))
        times = np.linspace(0.0, 240.0, 49)

        mc = propagate_kinetics_uncertainty_monte_carlo(
            cr0_samples,
            times,
            model="passivation",
            d_crit_nm=1.0,
            d_max_nm=5.0,
        )
        self.assertIsInstance(mc, MonteCarloKineticsResult)
        self.assertEqual(mc.n_draws, 250)
        self.assertEqual(mc.depth_p50_nm.shape, times.shape)
        self.assertEqual(mc.depth_std_nm.shape, times.shape)
        self.assertEqual(mc.risk_p50.shape, times.shape)
        self.assertEqual(mc.risk_std.shape, times.shape)
        self.assertAlmostEqual(float(mc.depth_std_nm[0]), 0.0, places=12)
        self.assertGreater(float(mc.depth_std_nm[-1]), 0.0)
        self.assertGreater(float(mc.risk_std[-1]), 0.0)

        # Physical invariants across all time points
        self.assertAlmostEqual(float(mc.depth_p50_nm[0]), 0.0, places=12)
        self.assertTrue(np.all(mc.depth_p05_nm <= mc.depth_p25_nm + 1e-12))
        self.assertTrue(np.all(mc.depth_p25_nm <= mc.depth_p50_nm + 1e-12))
        self.assertTrue(np.all(mc.depth_p50_nm <= mc.depth_p75_nm + 1e-12))
        self.assertTrue(np.all(mc.depth_p75_nm <= mc.depth_p95_nm + 1e-12))
        self.assertTrue(np.all(mc.risk_p05 <= mc.risk_p25 + 1e-12))
        self.assertTrue(np.all(mc.risk_p25 <= mc.risk_p50 + 1e-12))
        self.assertTrue(np.all(mc.risk_p50 <= mc.risk_p75 + 1e-12))
        self.assertTrue(np.all(mc.risk_p75 <= mc.risk_p95 + 1e-12))
        self.assertTrue(np.all(np.diff(mc.depth_p50_nm) >= 0.0))
        self.assertTrue(np.all((mc.risk_p05 >= 0.0) & (mc.risk_p95 <= 1.0)))

        # Time to Threshold uncertainty summary
        self.assertAlmostEqual(mc.threshold_reached_fraction, 1.0, places=9)
        t_sum = mc.time_to_threshold_summary
        self.assertLess(t_sum["p05"], t_sum["p50"])
        self.assertLess(t_sum["p50"], t_sum["p95"])
        self.assertGreater(t_sum["std"], 0.0)

        # Unreached threshold case (d_max <= d_crit) reports fraction == 0.0 and inf percentiles
        mc_unreached = propagate_kinetics_uncertainty_monte_carlo(
            cr0_samples,
            times,
            model="passivation",
            d_crit_nm=1.0,
            d_max_nm=0.8,
        )
        self.assertEqual(mc_unreached.threshold_reached_fraction, 0.0)
        self.assertTrue(math.isinf(mc_unreached.time_to_threshold_summary["p50"]))

    def test_plot_kinetics_and_risk_renders_depth_and_risk_panels_with_bands(self) -> None:
        rng = np.random.default_rng(11)
        cr0_samples = np.exp(rng.normal(math.log(0.60), 0.2, size=100))
        times = np.linspace(0.0, 240.0, 49)
        mc = propagate_kinetics_uncertainty_monte_carlo(
            cr0_samples, times, model="linear", d_crit_nm=1.0
        )

        fig, (ax_depth, ax_risk) = plot_kinetics_and_risk(mc)
        try:
            self.assertIn("time", ax_depth.get_xlabel().lower())
            self.assertIn("depth", ax_depth.get_ylabel().lower())
            self.assertIn("risk", ax_risk.get_ylabel().lower())
            self.assertIsNotNone(ax_depth.get_legend())
            self.assertIsNotNone(ax_risk.get_legend())
        finally:
            plt.close(fig)


class TestKineticsNotebookArtifact(unittest.TestCase):
    """Verify the time-resolved kinetics & risk demonstration notebook exists and is executed."""

    def test_kinetics_notebook_is_executed_with_outputs(self) -> None:
        nb_path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "05_time_resolved_kinetics_and_risk.ipynb"
        )
        self.assertTrue(nb_path.exists(), f"Missing notebook: {nb_path}")
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 4)
        self.assertTrue(all(len(c.get("outputs", [])) > 0 for c in code_cells))


if __name__ == "__main__":
    unittest.main()
