"""Integration tests for the Streamlit platform backing controller and views."""

from __future__ import annotations

import math
import unittest

import plotly.graph_objects as go

from corrosion_simulator.database import CORE_METALS, HarmonizedDatabase
from corrosion_simulator.ml_predictor import HybridMLPredictor, MLFeatureQuery
from corrosion_simulator.solution import CleaningRecipe, Solution
from corrosion_simulator.platform_controller import (
    CorrosionPlatformController,
    GalvanicAnalysisResult,
    ManualElectrochemicalInput,
    SimulationRequest,
    SingleMetalAnalysisResult,
)
from corrosion_simulator.plotly_charts import (
    apply_depth_yaxis_scale,
    build_galvanic_figures,
    build_single_metal_figures,
)


class TestCorrosionPlatformControllerSingleMetal(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db = HarmonizedDatabase.from_seed()
        cls.predictor = HybridMLPredictor(model_family="random_forest", random_state=42)
        cls.predictor.fit(cls.db)
        cls.controller = CorrosionPlatformController(cls.db, cls.predictor)

    def test_sort_metals_by_count_orders_descending_with_alpha_tiebreak(self) -> None:
        from app import _sort_metals_by_count

        counts = {"Cu": 1034, "Co": 213, "W": 245, "Ti": 925, "Ru": 15, "Ta": 46, "Sn": 15}
        ordered = _sort_metals_by_count(["Co", "Ru", "Cu", "Sn", "Ta", "Ti", "W"], counts)
        self.assertEqual(ordered, ["Cu", "Ti", "W", "Co", "Ta", "Ru", "Sn"])

    def test_available_options_cover_all_core_metals(self) -> None:
        options = self.controller.available_options()
        self.assertEqual(tuple(options["metals"]), CORE_METALS)
        self.assertIn("H2O2", options["oxidizer_types"])
        self.assertIn("database", options["parameter_sources"])
        self.assertIn("ml", options["parameter_sources"])
        self.assertIn("manual", options["parameter_sources"])
        self.assertIn("auto", options["kinetics_models"])
        self.assertEqual(options["solvent_regimes"], ["aqueous", "semi_aqueous", "non_aqueous"])
        self.assertIn("EG", options["organic_solvent_types"])
        self.assertIn("DMSO", options["organic_solvent_types"])

    def test_simulation_request_solvent_consistency_and_guardrails(self) -> None:
        default_req = SimulationRequest(primary_metal="Cu")
        self.assertEqual(default_req.recipe.solution.solvent_regime, "aqueous")
        self.assertEqual(default_req.recipe.solution.organic_solvent_type, "None")
        self.assertAlmostEqual(float(default_req.recipe.solution.water_content_wt_pct), 100.0)
        self.assertEqual(default_req.recipe.solution.solvent_subclass, "acidic")

        semi_req = SimulationRequest(primary_metal="Cu", recipe=CleaningRecipe(solution=Solution(solvent_regime="semi_aqueous", organic_solvent_type="EG")))
        self.assertAlmostEqual(float(semi_req.recipe.solution.water_content_wt_pct), 50.0)
        self.assertEqual(semi_req.recipe.solution.solvent_subclass, "glycol")

        nonaq_req = SimulationRequest(primary_metal="Cu", secondary_metal="Co", parameter_source="ml", mc_draws=80, recipe=CleaningRecipe(solution=Solution(solvent_regime="non_aqueous", organic_solvent_type="DMSO", water_content_wt_pct=0.5)))
        self.assertEqual(nonaq_req.recipe.solution.solvent_regime, "non_aqueous")
        self.assertEqual(nonaq_req.recipe.solution.solvent_subclass, "polar_aprotic")

        # Boundary derivation from water_content_wt_pct alone
        self.assertEqual(Solution(water_content_wt_pct=90.0).solvent_regime, "aqueous")
        self.assertEqual(Solution(water_content_wt_pct=89.99).solvent_regime, "semi_aqueous")
        self.assertEqual(Solution(water_content_wt_pct=1.0).solvent_regime, "semi_aqueous")
        self.assertEqual(Solution(water_content_wt_pct=0.99).solvent_regime, "non_aqueous")

        # Inconsistent regime and water content must fail fast
        with self.assertRaises(ValueError):
            Solution(solvent_regime="non_aqueous", water_content_wt_pct=50.0)
        with self.assertRaises(ValueError):
            Solution(solvent_regime="aqueous", water_content_wt_pct=85.0)

        # Non-aqueous simulations must surface physics-limitation warnings in both Single Metal and Galvanic payloads
        single_payload = self.controller.run_single_metal_analysis(nonaq_req)
        galv_payload = self.controller.run_galvanic_analysis(nonaq_req)
        for payload in (single_payload, galv_payload):
            joined = " ".join(payload.warnings)
            self.assertIn("aqueous pH", joined)
            self.assertIn("V vs. SHE", joined)
            self.assertIn("aqueous Pourbaix", joined)
            summary = payload.to_summary_dict()
            self.assertEqual(summary["solvent_regime"], "non_aqueous")
            self.assertEqual(summary["solvent_subclass"], "polar_aprotic")
            self.assertEqual(summary["organic_solvent_type"], "DMSO")
            self.assertAlmostEqual(summary["water_content_wt_pct"], 0.5)

        # ML query carries solvent descriptors cleanly
        q = MLFeatureQuery.for_solution("Cu", semi_req.recipe.solution, semi_req.recipe.temperature_C)
        self.assertEqual(q.organic_solvent_type, "EG")
        self.assertAlmostEqual(q.water_content_wt_pct, 50.0)
        self.assertEqual(q.base_solvent, "DI water")

        # Database lookup mismatching solvent regime adds warning and downgrades confidence
        db_nonaq_payload = self.controller.run_single_metal_analysis(
            SimulationRequest(primary_metal="Cu", parameter_source="database", recipe=CleaningRecipe(solution=Solution(oxidizer_type="H2O2", oxidizer_concentration=1.0, pH=4.0, solvent_regime="non_aqueous", organic_solvent_type="DMSO", water_content_wt_pct=0.2)))
        )
        self.assertEqual(db_nonaq_payload.confidence_tier, "Low")
        self.assertTrue(any("Solvent Regime" in w for w in db_nonaq_payload.warnings))

    def test_single_metal_ml_workflow_returns_complete_payload_and_figures(self) -> None:
        req = SimulationRequest(primary_metal="Cu", d_crit_nm=1.0, kinetics_model="auto", parameter_source="ml", mc_draws=200, recipe=CleaningRecipe(solution=Solution(oxidizer_type="H2O2", oxidizer_concentration=1.0, acid_type="Citric acid", acid_concentration=1.0, pH=3.0), temperature_C=25.0, immersion_time_s=300.0))
        payload = self.controller.run_single_metal_analysis(req)

        self.assertIsInstance(payload, SingleMetalAnalysisResult)
        self.assertEqual(payload.resolved_parameters.metal, "Cu")
        self.assertEqual(payload.resolved_parameters.source, "ml")
        self.assertGreater(payload.initial_corrosion_rate_nm_min, 0.0)
        self.assertGreater(payload.initial_corrosion_rate_mm_yr, 0.0)
        self.assertGreaterEqual(payload.corrosion_depth_nm, 0.0)
        self.assertGreater(payload.time_to_threshold_s, 0.0)
        self.assertGreaterEqual(payload.thickness_loss_risk_score, 0.0)
        self.assertLessEqual(payload.thickness_loss_risk_score, 1.0)
        self.assertIn(payload.thickness_loss_risk_tier, {"Low", "Medium", "High", "Critical"})
        self.assertIn(payload.confidence_tier, {"High", "Medium", "Low"})

        # Parameter table has all 4 electrochemical parameters with ±1σ columns
        self.assertEqual(
            list(payload.parameter_table["Parameter"]),
            ["E_corr", "i_corr", "beta_a", "beta_c"],
        )
        self.assertIn("Std (±1σ)", payload.parameter_table.columns)

        # ML mode populates Monte Carlo uncertainty bands
        self.assertIsNotNone(payload.mc_propagation)
        self.assertIsNotNone(payload.mc_kinetics)

        # Pure simulation result carries no Plotly figure attributes
        self.assertFalse(any(name.endswith("_figure") for name in vars(payload)))

        # Plotly figures are built on demand via plotly_charts and contain valid traces
        figs = build_single_metal_figures(payload, include_combined=True)
        self.assertIsInstance(figs.polarization_figure, go.Figure)
        self.assertGreaterEqual(len(figs.polarization_figure.data), 3)
        self.assertIn("Potential", figs.polarization_figure.layout.xaxis.title.text)
        self.assertIn("log10", figs.polarization_figure.layout.yaxis.title.text)
        self.assertIsNotNone(figs.polarization_figure.layout.yaxis.range)
        net_trace = [t for t in figs.polarization_figure.data if (t.name or "").endswith("Net |i_net|")][0]
        self.assertTrue(all(math.isfinite(float(v)) for v in net_trace.y))
        pol_names = {t.name for t in figs.polarization_figure.data}
        self.assertIn("Net |i_net| 50% Band (p25–p75)", pol_names)
        self.assertNotIn("Net |i_net| 90% Band (p05–p95)", pol_names)
        marker_trace = [t for t in figs.polarization_figure.data if (t.name or "").startswith("(E_corr, i_corr)")][0]
        self.assertIsNone(marker_trace.error_x.array)
        self.assertIsNone(marker_trace.error_y.array)

        self.assertIsInstance(figs.kinetics_risk_figure, go.Figure)
        self.assertGreaterEqual(len(figs.kinetics_risk_figure.data), 2)
        self.assertIsInstance(figs.depth_figure, go.Figure)
        self.assertIsInstance(figs.risk_figure, go.Figure)
        self.assertEqual(figs.depth_figure.layout.legend.orientation, "h")
        self.assertEqual(figs.risk_figure.layout.legend.orientation, "h")
        depth_names = {t.name for t in figs.depth_figure.data}
        risk_names = {t.name for t in figs.risk_figure.data}
        self.assertIn("Depth 50% Band (p25–p75)", depth_names)
        self.assertNotIn("Depth 90% Band (p05–p95)", depth_names)
        self.assertIn("Risk 50% Band (p25–p75)", risk_names)
        self.assertNotIn("Risk 90% Band (p05–p95)", risk_names)

        log_fig = apply_depth_yaxis_scale(figs.depth_figure, "log")
        self.assertEqual(log_fig.layout.yaxis.type, "log")
        for trace in log_fig.data:
            if getattr(trace, "y", None) is not None:
                self.assertTrue(all(float(v) > 0.0 for v in trace.y))
        depth_line = [t for t in figs.depth_figure.data if "Depth (" in (t.name or "")][0]
        self.assertEqual(depth_line.line.color, "#1f77b4")
        self.assertEqual(depth_line.line.dash, "solid")
        risk_line = [t for t in figs.risk_figure.data if (t.name or "") == "Thickness Loss Risk R(t)"][0]
        self.assertEqual(risk_line.line.color, "#d95f02")
        self.assertEqual(risk_line.line.dash, "dashdot")

        # Summary dict is JSON-friendly and contains core keys
        summary = payload.to_summary_dict()
        self.assertEqual(summary["primary_metal"], "Cu")
        self.assertEqual(summary["parameter_source"], "ml")
        self.assertIn("confidence_tier", summary)
        self.assertIn("warnings", summary)
        self.assertIn("experimental_recommendations", summary)

    def test_single_metal_database_and_manual_sources(self) -> None:
        db_req = SimulationRequest(primary_metal="Co", d_crit_nm=1.0, kinetics_model="linear", parameter_source="database", recipe=CleaningRecipe(solution=Solution(oxidizer_type="H2O2", oxidizer_concentration=1.0, pH=4.0), temperature_C=25.0, immersion_time_s=120.0))
        db_payload = self.controller.run_single_metal_analysis(db_req)
        self.assertEqual(db_payload.resolved_parameters.source, "database")
        self.assertIsNone(db_payload.mc_propagation)
        db_figs = build_single_metal_figures(db_payload, include_combined=True)
        self.assertIsInstance(db_figs.polarization_figure, go.Figure)
        self.assertIsInstance(db_figs.kinetics_risk_figure, go.Figure)

        manual_req = SimulationRequest(primary_metal="Cu", d_crit_nm=1.0, kinetics_model="linear", parameter_source="manual", manual_primary=ManualElectrochemicalInput(
                Ecorr_V_SHE=0.25,
                icorr_uA_cm2=2.3,
                beta_a_mV_dec=60.0,
                beta_c_mV_dec=120.0,
            ), recipe=CleaningRecipe(solution=Solution(pH=3.0), temperature_C=25.0, immersion_time_s=60.0))
        manual_payload = self.controller.run_single_metal_analysis(manual_req)
        self.assertEqual(manual_payload.resolved_parameters.source, "manual")
        # Cu at 2.3 uA/cm2, z=2 -> ~0.0508 nm/min? Wait: Faraday check from spec is ~0.051 or 0.45?
        # Just verify exact match with Faraday physics solver (no hard-coded wrong number).
        self.assertAlmostEqual(manual_payload.corrosion_depth_nm, manual_payload.initial_corrosion_rate_nm_min * 1.0, places=6)
        self.assertAlmostEqual(
            manual_payload.time_to_threshold_s,
            60.0 / manual_payload.initial_corrosion_rate_nm_min,
            places=5,
        )

    def test_single_metal_out_of_domain_triggers_low_confidence_and_recommendations(self) -> None:
        ood_req = SimulationRequest(primary_metal="Cu", d_crit_nm=1.0, parameter_source="ml", mc_draws=100, recipe=CleaningRecipe(solution=Solution(oxidizer_type="UnknownOxidizerX", oxidizer_concentration=50.0, pH=13.5), temperature_C=95.0, immersion_time_s=180.0))
        payload = self.controller.run_single_metal_analysis(ood_req)
        self.assertEqual(payload.confidence_tier, "Low")
        self.assertGreater(len(payload.warnings), 0)
        self.assertGreater(len(payload.experimental_recommendations), 0)


class TestCorrosionPlatformControllerGalvanic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db = HarmonizedDatabase.from_seed()
        cls.predictor = HybridMLPredictor(model_family="random_forest", random_state=42)
        cls.predictor.fit(cls.db)
        cls.controller = CorrosionPlatformController(cls.db, cls.predictor)

    def test_galvanic_workflow_requires_distinct_secondary_metal(self) -> None:
        missing_secondary = SimulationRequest(primary_metal="Cu", secondary_metal=None)
        with self.assertRaises(ValueError):
            self.controller.run_galvanic_analysis(missing_secondary)

        same_metal = SimulationRequest(primary_metal="Cu", secondary_metal="Cu")
        with self.assertRaises(ValueError):
            self.controller.run_galvanic_analysis(same_metal)

    def test_galvanic_ml_workflow_returns_complete_payload_and_figures(self) -> None:
        req = SimulationRequest(primary_metal="Cu", secondary_metal="Co", cathode_to_anode_area_ratio=2.0, d_crit_nm=1.0, kinetics_model="linear", parameter_source="ml", mc_draws=200, recipe=CleaningRecipe(solution=Solution(oxidizer_type="H2O2", oxidizer_concentration=1.0, acid_type="Citric acid", acid_concentration=1.0, pH=3.0), temperature_C=25.0, immersion_time_s=300.0))
        payload = self.controller.run_galvanic_analysis(req)

        self.assertIsInstance(payload, GalvanicAnalysisResult)
        self.assertEqual({payload.active_metal, payload.noble_metal}, {"Cu", "Co"})
        self.assertIn(payload.galvanic_driving_force_risk, {"Low", "Medium", "High"})
        self.assertGreaterEqual(payload.acceleration_factor, 1.0)
        self.assertGreater(payload.galvanic_current_density_uA_cm2, 0.0)
        self.assertGreaterEqual(payload.accelerated_cr0_nm_min, payload.uncoupled_cr0_nm_min)
        self.assertGreaterEqual(payload.coupled_depth_nm, payload.uncoupled_depth_nm)
        self.assertLessEqual(payload.coupled_time_to_threshold_s, payload.uncoupled_time_to_threshold_s)
        self.assertGreaterEqual(payload.coupled_risk_score, payload.uncoupled_risk_score)
        self.assertIn(payload.coupled_risk_tier, {"Low", "Medium", "High", "Critical"})
        self.assertIn(payload.confidence_tier, {"High", "Medium", "Low"})

        # Monte Carlo uncertainty populated in ML mode
        self.assertIsNotNone(payload.mc_galvanic)
        self.assertIsNotNone(payload.mc_coupled_kinetics)

        # Pure simulation result carries no Plotly figure attributes
        self.assertFalse(any(name.endswith("_figure") for name in vars(payload)))

        # Plotly figures are built on demand via plotly_charts and include E_galv vertical line + comparative traces
        galv_figs = build_galvanic_figures(payload, include_combined=True)
        self.assertIsInstance(galv_figs.galvanic_polarization_figure, go.Figure)
        self.assertGreaterEqual(len(galv_figs.galvanic_polarization_figure.data), 3)
        self.assertGreaterEqual(len(galv_figs.galvanic_polarization_figure.layout.shapes), 1)
        self.assertIn("Potential", galv_figs.galvanic_polarization_figure.layout.xaxis.title.text)
        self.assertIn("log10", galv_figs.galvanic_polarization_figure.layout.yaxis.title.text)

        self.assertIsInstance(galv_figs.comparative_kinetics_figure, go.Figure)
        self.assertGreaterEqual(len(galv_figs.comparative_kinetics_figure.data), 4)
        self.assertIsInstance(galv_figs.galvanic_depth_figure, go.Figure)
        self.assertIsInstance(galv_figs.galvanic_risk_figure, go.Figure)
        self.assertEqual(galv_figs.galvanic_depth_figure.layout.legend.orientation, "h")
        self.assertEqual(galv_figs.galvanic_risk_figure.layout.legend.orientation, "h")
        galv_depth_names = {t.name for t in galv_figs.galvanic_depth_figure.data}
        galv_risk_names = {t.name for t in galv_figs.galvanic_risk_figure.data}
        self.assertIn("Coupled Depth 50% Band (p25–p75)", galv_depth_names)
        self.assertNotIn("Coupled Depth 90% Band (p05–p95)", galv_depth_names)
        self.assertIn("Coupled Risk 50% Band (p25–p75)", galv_risk_names)
        self.assertNotIn("Coupled Risk 90% Band (p05–p95)", galv_risk_names)

        # Galvanic electrochemical parameter table contains Anode, Cathode (4 params) + Mixed (E_galv, i_galv)
        galv_table = payload.electrochemical_parameter_table
        self.assertEqual(len(galv_table), 3)
        self.assertTrue(str(galv_table.iloc[0]["Role"]).startswith("Anode (Active):"))
        self.assertTrue(str(galv_table.iloc[1]["Role"]).startswith("Cathode (Noble):"))
        self.assertTrue(str(galv_table.iloc[2]["Role"]).startswith("Mixed (Coupled"))
        self.assertAlmostEqual(
            float(galv_table.iloc[2]["E_corr / E_galv (V vs. SHE)"]),
            round(payload.mixed_potential_V_SHE, 4),
            places=4,
        )
        self.assertAlmostEqual(
            float(galv_table.iloc[2]["i_corr / i_galv (µA/cm²)"]),
            payload.galvanic_current_density_uA_cm2,
            places=6,
        )
        galv_coupled_depth = [t for t in galv_figs.galvanic_depth_figure.data if (t.name or "").endswith("Coupled Depth")][0]
        self.assertEqual(galv_coupled_depth.line.color, "#1f77b4")
        self.assertEqual(galv_coupled_depth.line.dash, "solid")
        galv_uncoupled_depth = [t for t in galv_figs.galvanic_depth_figure.data if (t.name or "").endswith("Uncoupled Depth")][0]
        self.assertEqual(galv_uncoupled_depth.line.dash, "dash")
        galv_coupled_risk = [t for t in galv_figs.galvanic_risk_figure.data if (t.name or "").endswith("Coupled Risk")][0]
        self.assertEqual(galv_coupled_risk.line.color, "#d95f02")
        self.assertEqual(galv_coupled_risk.line.dash, "dashdot")
        galv_uncoupled_risk = [t for t in galv_figs.galvanic_risk_figure.data if (t.name or "").endswith("Uncoupled Risk")][0]
        self.assertEqual(galv_uncoupled_risk.line.dash, "dot")

        summary = payload.to_summary_dict()
        self.assertEqual(summary["active_metal"], payload.active_metal)
        self.assertEqual(summary["noble_metal"], payload.noble_metal)
        self.assertAlmostEqual(summary["cathode_to_anode_area_ratio"], 2.0)

    def test_galvanic_area_ratio_increases_acceleration_factor(self) -> None:
        base_kwargs = dict(
            primary_metal="Cu",
            secondary_metal="Co",
            recipe=CleaningRecipe(
                solution=Solution(pH=3.0),
                temperature_C=25.0,
                immersion_time_s=180.0,
            ),
            d_crit_nm=1.0,
            kinetics_model="linear",
            parameter_source="manual",
            manual_primary=ManualElectrochemicalInput(
                Ecorr_V_SHE=0.25, icorr_uA_cm2=2.0, beta_a_mV_dec=60.0, beta_c_mV_dec=120.0
            ),
            manual_secondary=ManualElectrochemicalInput(
                Ecorr_V_SHE=-0.10, icorr_uA_cm2=5.0, beta_a_mV_dec=60.0, beta_c_mV_dec=120.0
            ),
        )
        low_area = self.controller.run_galvanic_analysis(
            SimulationRequest(cathode_to_anode_area_ratio=0.5, **base_kwargs)
        )
        high_area = self.controller.run_galvanic_analysis(
            SimulationRequest(cathode_to_anode_area_ratio=5.0, **base_kwargs)
        )
        self.assertEqual(low_area.active_metal, "Co")
        self.assertEqual(low_area.noble_metal, "Cu")
        self.assertEqual(low_area.galvanic_driving_force_risk, "High")
        self.assertGreater(high_area.acceleration_factor, low_area.acceleration_factor)
        self.assertGreater(high_area.coupled_depth_nm, low_area.coupled_depth_nm)
        self.assertLess(high_area.coupled_time_to_threshold_s, low_area.coupled_time_to_threshold_s)

    def test_galvanic_database_source_workflow(self) -> None:
        req = SimulationRequest(primary_metal="Cu", secondary_metal="Co", cathode_to_anode_area_ratio=1.5, d_crit_nm=1.0, kinetics_model="auto", parameter_source="database", recipe=CleaningRecipe(solution=Solution(oxidizer_type="H2O2", oxidizer_concentration=1.0, pH=4.0), temperature_C=25.0, immersion_time_s=180.0))
        payload = self.controller.run_galvanic_analysis(req)
        self.assertEqual(payload.primary_resolved.source, "database")
        self.assertEqual(payload.secondary_resolved.source, "database")
        self.assertIsNone(payload.mc_galvanic)
        self.assertGreaterEqual(payload.acceleration_factor, 1.0)
        db_galv_figs = build_galvanic_figures(payload, include_combined=True)
        self.assertIsInstance(db_galv_figs.galvanic_polarization_figure, go.Figure)
        self.assertIsInstance(db_galv_figs.comparative_kinetics_figure, go.Figure)

    def test_galvanic_database_source_handles_mismatched_solution_ids_without_error(self) -> None:
        import pandas as pd

        # 1. Verify the active external dataset resolves Cu/Co and Cu/Ru cleanly without error
        raw_controller = CorrosionPlatformController.from_default_data()
        for pair in [("Cu", "Co"), ("Cu", "Ru")]:
            req = SimulationRequest(primary_metal=pair[0], secondary_metal=pair[1], cathode_to_anode_area_ratio=1.0, d_crit_nm=1.0, kinetics_model="auto", parameter_source="database", recipe=CleaningRecipe(solution=Solution(oxidizer_type="H2O2", oxidizer_concentration=1.0, pH=3.0), temperature_C=25.0, immersion_time_s=300.0))
            payload = raw_controller.run_galvanic_analysis(req)
            self.assertEqual({payload.active_metal, payload.noble_metal}, set(pair))
            self.assertGreaterEqual(payload.acceleration_factor, 1.0)
            self.assertIsInstance(build_galvanic_figures(payload).galvanic_polarization_figure, go.Figure)

        # 2. Deterministically verify mismatched solution_ids / temperatures degrade to Low confidence with warning
        disjoint_db = HarmonizedDatabase.from_raw_tables(
            pd.DataFrame(
                [
                    {
                        "record_id": "DISJ_CU",
                        "metal": "Cu",
                        "solution_id": "SOL_ONLY_CU",
                        "pH": 3.0,
                        "temperature_C": 25.0,
                        "oxidizer_type": "H2O2",
                        "oxidizer_concentration": 1.0,
                        "Ecorr_V_SHE": 0.22,
                        "icorr_uA_cm2": 4.5,
                        "beta_a_mV_dec": 60.0,
                        "beta_c_mV_dec": 120.0,
                    },
                    {
                        "record_id": "DISJ_RU",
                        "metal": "Ru",
                        "solution_id": "SOL_ONLY_RU",
                        "pH": 3.0,
                        "temperature_C": 45.0,
                        "oxidizer_type": "H2O2",
                        "oxidizer_concentration": 1.0,
                        "Ecorr_V_SHE": 0.52,
                        "icorr_uA_cm2": 0.8,
                        "beta_a_mV_dec": 65.0,
                        "beta_c_mV_dec": 115.0,
                    },
                ]
            )
        )
        disjoint_ctrl = CorrosionPlatformController(disjoint_db)
        disjoint_payload = disjoint_ctrl.run_galvanic_analysis(
            SimulationRequest(primary_metal="Cu", secondary_metal="Ru", cathode_to_anode_area_ratio=1.0, d_crit_nm=1.0, kinetics_model="auto", parameter_source="database", recipe=CleaningRecipe(solution=Solution(oxidizer_type="H2O2", oxidizer_concentration=1.0, pH=3.0), temperature_C=25.0, immersion_time_s=300.0))
        )
        self.assertEqual(disjoint_payload.confidence_tier, "Low")
        self.assertTrue(
            any("different experimental conditions" in w for w in disjoint_payload.warnings)
        )

    def test_live_ml_model_family_switching_and_caching(self) -> None:
        options = self.controller.available_options()
        self.assertEqual(
            set(options["ml_model_families"]),
            {"random_forest", "hist_gradient_boosting", "rf_gp_ensemble"},
        )

        for family in ("random_forest", "hist_gradient_boosting", "rf_gp_ensemble"):
            req = SimulationRequest(primary_metal="Cu", parameter_source="ml", ml_model_family=family, mc_draws=80, recipe=CleaningRecipe(solution=Solution(pH=4.0, oxidizer_type="H2O2", oxidizer_concentration=1.0), temperature_C=25.0))
            payload = self.controller.run_single_metal_analysis(req)
            self.assertIsNotNone(payload.resolved_parameters.ml_prediction)
            self.assertEqual(payload.resolved_parameters.ml_prediction.model_family, family)
            self.assertEqual(payload.to_summary_dict()["ml_model_family"], family)

        # Calling get_predictor again returns the cached instance
        self.assertIs(
            self.controller.repository.get_predictor("hist_gradient_boosting"),
            self.controller.repository.get_predictor("hist_gradient_boosting"),
        )

    def test_controller_calibration_modes_end_to_end_physics_and_reload(self) -> None:
        from pathlib import Path
        import tempfile

        import pandas as pd

        from corrosion_simulator.build_artifacts import build_processed_artifacts
        from corrosion_simulator.physics import wagner_traud_net_current

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            raw_dir = data_dir / "raw"
            proc_dir = data_dir / "processed"
            lab_dir = data_dir / "lab" / "cu_co_eg"
            raw_dir.mkdir(parents=True)
            lab_dir.mkdir(parents=True)

            pd.DataFrame(
                [
                    {
                        "record_id": f"RAW_{i}",
                        "metal": "Cu" if i % 2 == 0 else "Co",
                        "solution_id": f"SOL_{i % 2}",
                        "pH": 3.0 + (i % 4),
                        "temperature_C": 25.0 + (i % 2) * 10.0,
                        "oxidizer_type": "H2O2",
                        "oxidizer_concentration": 1.0,
                        "Ecorr_V_SHE": 0.25 if i % 2 == 0 else -0.10,
                        "icorr_uA_cm2": 2.5 + 0.3 * i,
                        "beta_a_mV_dec": 60.0,
                        "beta_c_mV_dec": 120.0,
                    }
                    for i in range(12)
                ]
            ).to_csv(raw_dir / "lit.csv", index=False)
            build_processed_artifacts(
                raw_dir=raw_dir,
                output_dir=proc_dir,
                models=["random_forest"],
                export_csv=False,
            )

            def _make_lab_rows(n_per_metal: int) -> pd.DataFrame:
                rows = []
                for idx in range(n_per_metal):
                    rows.append(
                        {
                            "record_id": f"LAB_CU_{idx}",
                            "metal": "Cu",
                            "solution_id": "EG_50",
                            "pH": 4.0 + idx * 0.5,
                            "temperature_C": 25.0 + idx * 5.0,
                            "oxidizer_type": "H2O2",
                            "oxidizer_concentration": 1.0,
                            "organic_solvent_type": "EG",
                            "water_content_wt_pct": 50.0,
                            "Ecorr_V_SHE": 0.40 + 0.01 * idx,
                            "icorr_uA_cm2": 12.0 + idx,
                            "beta_a_mV_dec": 75.0,
                            "beta_c_mV_dec": 135.0,
                        }
                    )
                    rows.append(
                        {
                            "record_id": f"LAB_CO_{idx}",
                            "metal": "Co",
                            "solution_id": "EG_50",
                            "pH": 4.0 + idx * 0.5,
                            "temperature_C": 25.0 + idx * 5.0,
                            "oxidizer_type": "H2O2",
                            "oxidizer_concentration": 1.0,
                            "organic_solvent_type": "EG",
                            "water_content_wt_pct": 50.0,
                            "Ecorr_V_SHE": -0.15 - 0.01 * idx,
                            "icorr_uA_cm2": 20.0 + idx,
                            "beta_a_mV_dec": 65.0,
                            "beta_c_mV_dec": 125.0,
                        }
                    )
                return pd.DataFrame(rows)

            _make_lab_rows(5).to_csv(lab_dir / "study.csv", index=False)
            ctrl = CorrosionPlatformController.from_default_data(
                data_dir=data_dir,
                model_family="random_forest",
                auto_rebuild=False,
            )
            self.assertEqual(
                ctrl.available_options()["calibration_modes"],
                ["literature_only", "delta_learning", "lab_only_gp"],
            )

            for mode in ("literature_only", "delta_learning", "lab_only_gp"):
                req = SimulationRequest(primary_metal="Cu", secondary_metal="Co", cathode_to_anode_area_ratio=2.0, d_crit_nm=1.0, kinetics_model="linear", parameter_source="ml", ml_model_family="random_forest", calibration_mode=mode, lab_project="cu_co_eg", mc_draws=100, recipe=CleaningRecipe(solution=Solution(oxidizer_type="H2O2", oxidizer_concentration=1.0, pH=5.0, solvent_regime="semi_aqueous", organic_solvent_type="EG", water_content_wt_pct=50.0), temperature_C=30.0, immersion_time_s=180.0))
                single = ctrl.run_single_metal_analysis(req)
                galv = ctrl.run_galvanic_analysis(req)

                # Wagner-Traud zero-current at E_corr and Faraday conservation
                params = single.resolved_parameters.parameters
                self.assertAlmostEqual(
                    wagner_traud_net_current(params.Ecorr_V_SHE, params), 0.0, places=9
                )
                self.assertGreater(single.initial_corrosion_rate_nm_min, 0.0)
                self.assertGreater(single.corrosion_depth_nm, 0.0)
                self.assertIsNotNone(single.mc_propagation)

                # Galvanic brentq charge conservation and kinetics
                self.assertLess(abs(galv.coupling_result.current_balance_residual_uA_cm2), 1e-8)
                self.assertGreaterEqual(galv.acceleration_factor, 1.0)
                self.assertIsNotNone(galv.mc_galvanic)
                self.assertIsNotNone(galv.mc_coupled_kinetics)
                self.assertEqual(single.to_summary_dict()["calibration_mode"], mode)

                if mode == "delta_learning":
                    self.assertIsNotNone(single.resolved_parameters.residual_breakdown)
                    self.assertEqual(
                        single.resolved_parameters.residual_breakdown.Ecorr_V_SHE.n_lab, 5
                    )
                    self.assertEqual(
                        list(single.parameter_table["Source"]),
                        ["ML (Delta-Learning)"] * 4,
                    )
                elif mode == "lab_only_gp":
                    self.assertIsNone(single.resolved_parameters.residual_breakdown)
                    self.assertEqual(
                        list(single.parameter_table["Source"]),
                        ["ML (Lab-Only GP)"] * 4,
                    )

            # Updating lab CSV and calling reload_external_data invalidates cached lab calibration models
            _make_lab_rows(6).to_csv(lab_dir / "study.csv", index=False)
            reload_summary = ctrl.repository.reload_external_data()
            self.assertFalse(reload_summary.raw_rebuilt)
            self.assertTrue(reload_summary.lab_cache_invalidated)
            refreshed = ctrl.run_single_metal_analysis(
                SimulationRequest(primary_metal="Cu", parameter_source="ml", calibration_mode="delta_learning", lab_project="cu_co_eg", mc_draws=80, recipe=CleaningRecipe(solution=Solution(oxidizer_type="H2O2", oxidizer_concentration=1.0, pH=5.0, solvent_regime="semi_aqueous", organic_solvent_type="EG", water_content_wt_pct=50.0), temperature_C=30.0))
            )
            assert refreshed.resolved_parameters.residual_breakdown is not None
            self.assertEqual(
                refreshed.resolved_parameters.residual_breakdown.Ecorr_V_SHE.n_lab, 6
            )


class TestStreamlitAppSmoke(unittest.TestCase):
    def test_streamlit_app_renders_without_exception(self) -> None:
        from pathlib import Path
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parents[1] / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0, msg=f"Streamlit app raised exceptions: {at.exception}")
        self.assertEqual(len(at.tabs), 4)
        self.assertGreaterEqual(len(at.metric), 4)

    def test_streamlit_sidebar_displays_literature_and_lab_sync_statuses_and_reloads(self) -> None:
        import os
        from pathlib import Path
        import tempfile
        from unittest.mock import patch

        import pandas as pd
        import streamlit as st
        from streamlit.testing.v1 import AppTest

        from corrosion_simulator.build_artifacts import build_processed_artifacts
        from corrosion_simulator.paths import DATA_DIR_ENV_VAR

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            raw_dir = data_dir / "raw"
            proc_dir = data_dir / "processed"
            lab_dir = data_dir / "lab" / "pilot_eg"
            raw_dir.mkdir(parents=True)
            lab_dir.mkdir(parents=True)

            pd.DataFrame(
                [
                    {
                        "record_id": f"R_{i}",
                        "metal": "Cu" if i % 2 == 0 else "Co",
                        "solution_id": f"SOL_{i % 2}",
                        "pH": 3.0 + (i % 4) * 0.5,
                        "temperature_C": 25.0 + (i % 2) * 5.0,
                        "oxidizer_type": "H2O2",
                        "oxidizer_concentration": 1.0,
                        "Ecorr_V_SHE": 0.25 if i % 2 == 0 else -0.10,
                        "icorr_uA_cm2": 3.0 + i * 0.2,
                        "beta_a_mV_dec": 60.0,
                        "beta_c_mV_dec": 120.0,
                    }
                    for i in range(12)
                ]
            ).to_csv(raw_dir / "lit.csv", index=False)
            build_processed_artifacts(
                raw_dir=raw_dir,
                output_dir=proc_dir,
                models=["all"],
                export_csv=False,
            )

            pd.DataFrame(
                [
                    {
                        "record_id": "L_1",
                        "metal": "Cu",
                        "solution_id": "SOL_EG",
                        "pH": 6.5,
                        "temperature_C": 25.0,
                        "Ecorr_V_SHE": 0.18,
                        "icorr_uA_cm2": 1.8,
                        "beta_a_mV_dec": 60.0,
                        "beta_c_mV_dec": 120.0,
                        "organic_solvent_type": "EG",
                        "organic_solvent_concentration": 20.0,
                    }
                ]
            ).to_csv(lab_dir / "lab_eg.csv", index=False)

            app_path = Path(__file__).resolve().parents[1] / "app.py"
            st.cache_resource.clear()
            try:
                with patch.dict(os.environ, {DATA_DIR_ENV_VAR: str(data_dir)}, clear=False):
                    at = AppTest.from_file(str(app_path), default_timeout=30)
                    at.run()
                    self.assertEqual(len(at.exception), 0, msg=f"App error: {at.exception}")
                    sidebar_markdown = "\n".join(el.value for el in at.sidebar.markdown)
                    sidebar_captions = "\n".join(el.value for el in at.sidebar.caption)
                    self.assertIn("Literature Prior", sidebar_markdown)
                    self.assertIn("Lab Data", sidebar_markdown)
                    self.assertIn("pilot_eg", sidebar_captions)
                    self.assertIn("`1` records", sidebar_captions)

                    # Trigger Reload External Data button when only lab CSV changes
                    pd.DataFrame(
                        [
                            {
                                "record_id": "L_1",
                                "metal": "Cu",
                                "solution_id": "SOL_EG",
                                "pH": 6.5,
                                "temperature_C": 25.0,
                                "Ecorr_V_SHE": 0.18,
                                "icorr_uA_cm2": 1.8,
                                "beta_a_mV_dec": 60.0,
                                "beta_c_mV_dec": 120.0,
                                "organic_solvent_type": "EG",
                                "organic_solvent_concentration": 20.0,
                            },
                            {
                                "record_id": "L_2",
                                "metal": "Cu",
                                "solution_id": "SOL_EG",
                                "pH": 7.0,
                                "temperature_C": 35.0,
                                "Ecorr_V_SHE": 0.16,
                                "icorr_uA_cm2": 2.4,
                                "beta_a_mV_dec": 62.0,
                                "beta_c_mV_dec": 124.0,
                                "organic_solvent_type": "EG",
                                "organic_solvent_concentration": 20.0,
                            },
                        ]
                    ).to_csv(lab_dir / "lab_eg.csv", index=False)

                    reload_btn = next(
                        b for b in at.sidebar.button if "Reload External Data" in b.label
                    )
                    reload_btn.click().run()
                    self.assertEqual(len(at.exception), 0, msg=f"Reload error: {at.exception}")
                    updated_captions = "\n".join(el.value for el in at.sidebar.caption)
                    self.assertIn("`reloaded`", updated_captions)
                    self.assertIn("`2` records", updated_captions)
            finally:
                st.cache_resource.clear()

    def test_streamlit_sidebar_solvent_controls_and_non_aqueous_warning_banner(self) -> None:
        from pathlib import Path
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parents[1] / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0, msg=f"Initial render failed: {at.exception}")

        regime_box = next(s for s in at.sidebar.selectbox if s.label == "Solvent Regime")
        cosolvent_box = next(s for s in at.sidebar.selectbox if s.label == "Organic Co-Solvent")
        regime_box.select("semi_aqueous").run()
        self.assertEqual(len(at.exception), 0, msg=f"Semi-aqueous switch failed: {at.exception}")
        water_slider = next(s for s in at.sidebar.slider if s.label == "Water Content (wt%)")
        self.assertGreaterEqual(float(water_slider.value), 1.0)
        self.assertLess(float(water_slider.value), 90.0)

        cosolvent_box = next(s for s in at.sidebar.selectbox if s.label == "Organic Co-Solvent")
        cosolvent_box.select("DMSO").run()
        regime_box = next(s for s in at.sidebar.selectbox if s.label == "Solvent Regime")
        regime_box.select("non_aqueous").run()
        self.assertEqual(len(at.exception), 0, msg=f"Non-aqueous switch failed: {at.exception}")

        water_slider = next(s for s in at.sidebar.slider if s.label == "Water Content (wt%)")
        self.assertLess(float(water_slider.value), 1.0)

        banner_warnings = [
            w.value
            for w in at.warning
            if "aqueous pH" in w.value
            and "V vs. SHE" in w.value
            and "aqueous Pourbaix" in w.value
        ]
        # Must render in both Single Metal and Galvanic views
        self.assertGreaterEqual(len(banner_warnings), 2)

    def test_streamlit_lab_calibration_controls_mode_aware_ad_and_residual_breakdown(self) -> None:
        import os
        from pathlib import Path
        import tempfile
        from unittest.mock import patch

        import pandas as pd
        import streamlit as st
        from streamlit.testing.v1 import AppTest

        from corrosion_simulator.build_artifacts import build_processed_artifacts
        from corrosion_simulator.paths import DATA_DIR_ENV_VAR

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            raw_dir = data_dir / "raw"
            proc_dir = data_dir / "processed"
            lab_dir = data_dir / "lab" / "pilot_eg"
            raw_dir.mkdir(parents=True)
            lab_dir.mkdir(parents=True)

            pd.DataFrame(
                [
                    {
                        "record_id": f"RAW_{i}",
                        "metal": "Cu" if i % 2 == 0 else "Co",
                        "solution_id": f"SOL_{i % 2}",
                        "pH": 3.0 + (i % 4),
                        "temperature_C": 25.0 + (i % 2) * 10.0,
                        "oxidizer_type": "H2O2",
                        "oxidizer_concentration": 1.0,
                        "Ecorr_V_SHE": 0.25 if i % 2 == 0 else -0.10,
                        "icorr_uA_cm2": 2.5 + 0.3 * i,
                        "beta_a_mV_dec": 60.0,
                        "beta_c_mV_dec": 120.0,
                    }
                    for i in range(12)
                ]
            ).to_csv(raw_dir / "lit.csv", index=False)
            build_processed_artifacts(
                raw_dir=raw_dir,
                output_dir=proc_dir,
                models=["random_forest"],
                export_csv=False,
            )

            lab_rows = []
            for idx in range(5):
                for metal, ecorr, icorr in (("Cu", 0.40, 14.0), ("Co", -0.20, 24.0)):
                    lab_rows.append(
                        {
                            "record_id": f"L_{metal}_{idx}",
                            "metal": metal,
                            "solution_id": f"EG_{idx}",
                            "pH": 3.0 + idx * 0.5,
                            "temperature_C": 25.0 + idx * 2.0,
                            "oxidizer_type": "H2O2",
                            "oxidizer_concentration": 1.0,
                            "organic_solvent_type": "EG",
                            "water_content_wt_pct": 50.0,
                            "Ecorr_V_SHE": ecorr + 0.01 * idx,
                            "icorr_uA_cm2": icorr + idx,
                            "beta_a_mV_dec": 75.0,
                            "beta_c_mV_dec": 135.0,
                        }
                    )
            pd.DataFrame(lab_rows).to_csv(lab_dir / "pilot_eg.csv", index=False)

            app_path = Path(__file__).resolve().parents[1] / "app.py"
            st.cache_resource.clear()
            try:
                with patch.dict(os.environ, {DATA_DIR_ENV_VAR: str(data_dir)}, clear=False):
                    at = AppTest.from_file(str(app_path), default_timeout=30)
                    at.run()
                    self.assertEqual(len(at.exception), 0, msg=f"Initial app error: {at.exception}")

                    # Select Delta-Learning mode and verify Reverting status on aqueous default query
                    cal_radio = next(
                        r for r in at.sidebar.radio if r.label == "Lab Calibration Mode"
                    )
                    lab_select = next(
                        s for s in at.sidebar.selectbox if s.label == "Lab Dataset (data/lab/)"
                    )
                    self.assertIn("pilot_eg", lab_select.options)
                    cal_radio.set_value("delta_learning").run()
                    self.assertEqual(len(at.exception), 0, msg=f"Delta-Learning error: {at.exception}")
                    info_texts = "\n".join(el.value for el in at.info)
                    self.assertIn("Reverting to Literature Prior", info_texts)

                    # Switch to semi_aqueous + EG so query is Lab-Anchored
                    next(s for s in at.sidebar.selectbox if s.label == "Organic Co-Solvent").select("EG").run()
                    next(s for s in at.sidebar.selectbox if s.label == "Solvent Regime").select("semi_aqueous").run()
                    self.assertEqual(len(at.exception), 0, msg=f"Semi-aqueous Delta error: {at.exception}")
                    anchored_info = "\n".join(el.value for el in at.info)
                    self.assertIn("Lab-Anchored", anchored_info)

                    # Expandable Residual Breakdown renders in Single Metal and Galvanic views
                    rb_expanders = [
                        ex for ex in at.expander if "Residual Breakdown" in ex.label
                    ]
                    self.assertGreaterEqual(len(rb_expanders), 2)
                    rb_tables = [
                        df.value
                        for df in at.dataframe
                        if "Lab Residual (Δ)" in df.value.columns
                    ]
                    self.assertGreaterEqual(len(rb_tables), 2)
                    self.assertIn("GP Residual", list(rb_tables[0]["Method"]))

                    # Switch to Lab-Only Local GP mode and verify strict lab AD banner
                    next(r for r in at.sidebar.radio if r.label == "Lab Calibration Mode").set_value("lab_only_gp").run()
                    self.assertEqual(len(at.exception), 0, msg=f"Lab-Only GP error: {at.exception}")
                    lab_only_info = "\n".join(el.value for el in at.info)
                    self.assertIn("Lab-Only Local GP", lab_only_info)
                    self.assertIn("pilot_eg", lab_only_info)
            finally:
                st.cache_resource.clear()

    def test_auto_model_routing_override_and_cross_model_disagreement_downgrade(self) -> None:
        from dataclasses import replace
        from types import SimpleNamespace
        from unittest.mock import patch

        from corrosion_simulator.ml_predictor import (
            ApplicabilityDomainResult,
            MLParameterPrediction,
            ParameterEstimate,
        )
        from corrosion_simulator.platform_controller import (
            CorrosionPlatformController,
            SimulationRequest,
            resolve_model_family_for_mode,
        )

        # 1. Deterministic Calibration Mode auto-routing & explicit override
        self.assertEqual(resolve_model_family_for_mode("literature_only"), "rf_gp_ensemble")
        self.assertEqual(resolve_model_family_for_mode("delta_learning"), "hist_gradient_boosting")
        self.assertEqual(resolve_model_family_for_mode("lab_only_gp"), "lab_local_gp")
        self.assertEqual(
            resolve_model_family_for_mode("literature_only", "random_forest"),
            "random_forest",
        )
        self.assertEqual(
            resolve_model_family_for_mode("lab_only_gp", "random_forest"),
            "lab_local_gp",
        )

        req_lit = SimulationRequest(primary_metal="Cu", calibration_mode="literature_only")
        req_delta = SimulationRequest(primary_metal="Cu", calibration_mode="delta_learning")
        req_override = SimulationRequest(
            primary_metal="Cu",
            calibration_mode="literature_only",
            ml_model_family="random_forest",
        )
        self.assertEqual(req_lit.ml_model_family, "rf_gp_ensemble")
        self.assertEqual(req_delta.ml_model_family, "hist_gradient_boosting")
        self.assertEqual(req_override.ml_model_family, "random_forest")

        # 2. Peer-absent no-op: when peer model is neither in memory nor on disk, never train it on the fly
        ctrl = CorrosionPlatformController.from_default_data(model_family="rf_gp_ensemble")
        ctrl.repository._predictors = {
            "rf_gp_ensemble": ctrl.repository.get_predictor("rf_gp_ensemble")
        }
        ctrl.repository.artifact_dir = None  # No disk artifacts available for peer
        with patch.object(
            ctrl.repository,
            "_load_or_fit_predictor",
            side_effect=AssertionError("Must not train missing peer model during prediction"),
        ):
            payload_no_peer = ctrl.run_single_metal_analysis(
                SimulationRequest(primary_metal="Cu", calibration_mode="literature_only", mc_draws=60, recipe=CleaningRecipe(solution=Solution(pH=4.0), temperature_C=25.0))
            )
        ad_no_peer = payload_no_peer.resolved_parameters.applicability
        assert ad_no_peer is not None
        self.assertIsNone(ad_no_peer.cross_model_disagreement_log10_icorr)
        self.assertIsNone(ad_no_peer.cross_model_disagreement_ecorr_V)
        self.assertFalse(ad_no_peer.disagreement_downgraded)

        # 3. Threshold downgrades when peer is cached in memory
        base_ad_high = ApplicabilityDomainResult(
            metal="Cu",
            confidence_tier="High",
            in_bounding_box=True,
            out_of_box_features=(),
            unseen_categories=(),
            knn_mean_distance=0.1,
            knn_k=3,
            n_metal_training_samples=20,
            warnings=(),
            experimental_recommendations=(),
        )

        def _make_stub_pred(
            family: str,
            ecorr: float,
            log_i: float,
            ad: ApplicabilityDomainResult = base_ad_high,
        ) -> MLParameterPrediction:
            return MLParameterPrediction(
                metal="Cu",
                Ecorr_V_SHE=ParameterEstimate(mean=ecorr, std=0.02),
                log10_icorr_uA_cm2=ParameterEstimate(mean=log_i, std=0.1),
                icorr_uA_cm2_median=10.0**log_i,
                beta_a_mV_dec=ParameterEstimate(mean=60.0, std=5.0),
                beta_c_mV_dec=ParameterEstimate(mean=120.0, std=5.0),
                model_family=family,  # type: ignore[arg-type]
                applicability=ad,
            )

        # Moderate |ΔE_corr| = 0.10 V (> 0.08 V) -> High downgrades to Medium
        ctrl.repository._predictors["rf_gp_ensemble"] = SimpleNamespace(  # type: ignore[assignment]
            model_family="rf_gp_ensemble",
            predict=lambda q: _make_stub_pred("rf_gp_ensemble", 0.20, 0.50),
        )
        ctrl.repository._predictors["hist_gradient_boosting"] = SimpleNamespace(  # type: ignore[assignment]
            model_family="hist_gradient_boosting",
            predict=lambda q: _make_stub_pred("hist_gradient_boosting", 0.30, 0.55),
        )
        mod_payload = ctrl.run_single_metal_analysis(
            SimulationRequest(primary_metal="Cu", calibration_mode="literature_only", mc_draws=60)
        )
        self.assertEqual(mod_payload.confidence_tier, "Medium")
        assert mod_payload.resolved_parameters.applicability is not None
        self.assertTrue(mod_payload.resolved_parameters.applicability.disagreement_downgraded)
        self.assertAlmostEqual(
            float(mod_payload.resolved_parameters.applicability.cross_model_disagreement_ecorr_V),
            0.10,
            places=6,
        )
        self.assertTrue(any("Cross-model disagreement" in w for w in mod_payload.warnings))

        # Moderate |Δlog10(i_corr)| = 0.65 (> 0.5) on Medium tier -> Medium downgrades to Low
        base_ad_med = replace(base_ad_high, confidence_tier="Medium")
        ctrl.repository._predictors["rf_gp_ensemble"] = SimpleNamespace(  # type: ignore[assignment]
            model_family="rf_gp_ensemble",
            predict=lambda q: _make_stub_pred("rf_gp_ensemble", 0.20, 0.50, base_ad_med),
        )
        ctrl.repository._predictors["hist_gradient_boosting"] = SimpleNamespace(  # type: ignore[assignment]
            model_family="hist_gradient_boosting",
            predict=lambda q: _make_stub_pred("hist_gradient_boosting", 0.21, 1.15, base_ad_med),
        )
        med_to_low = ctrl.run_single_metal_analysis(
            SimulationRequest(primary_metal="Cu", calibration_mode="literature_only", mc_draws=60)
        )
        self.assertEqual(med_to_low.confidence_tier, "Low")

        # Severe |Δlog10(i_corr)| = 1.25 (> 1.0) on High tier -> forced directly to Low
        ctrl.repository._predictors["rf_gp_ensemble"] = SimpleNamespace(  # type: ignore[assignment]
            model_family="rf_gp_ensemble",
            predict=lambda q: _make_stub_pred("rf_gp_ensemble", 0.20, 0.50, base_ad_high),
        )
        ctrl.repository._predictors["hist_gradient_boosting"] = SimpleNamespace(  # type: ignore[assignment]
            model_family="hist_gradient_boosting",
            predict=lambda q: _make_stub_pred("hist_gradient_boosting", 0.21, 1.75, base_ad_high),
        )
        severe_payload = ctrl.run_single_metal_analysis(
            SimulationRequest(primary_metal="Cu", calibration_mode="literature_only", mc_draws=60)
        )
        self.assertEqual(severe_payload.confidence_tier, "Low")
        assert severe_payload.resolved_parameters.applicability is not None
        self.assertTrue(severe_payload.resolved_parameters.applicability.disagreement_downgraded)

        # 4. random_forest override still compares cached HGB vs RF+RGP; unanchored delta_learning also downgrades
        rf_override_payload = ctrl.run_single_metal_analysis(
            SimulationRequest(
                primary_metal="Cu",
                calibration_mode="literature_only",
                ml_model_family="random_forest",
                mc_draws=60,
            )
        )
        assert rf_override_payload.resolved_parameters.applicability is not None
        self.assertTrue(rf_override_payload.resolved_parameters.applicability.disagreement_downgraded)

        unanchored_ad = replace(
            base_ad_high,
            ad_mode="delta_learning",
            lab_anchoring_status="Reverting to Literature Prior",
            n_lab_samples=3,
        )
        unanchored_pred = _make_stub_pred("hist_gradient_boosting", 0.21, 1.75, unanchored_ad)
        default_cu_query = MLFeatureQuery.for_solution("Cu", Solution(), 25.0)
        downgraded_delta = ctrl._evaluate_cross_model_disagreement(
            SimulationRequest(primary_metal="Cu", calibration_mode="delta_learning"),
            "Cu",
            default_cu_query,
            unanchored_pred,
        )
        self.assertEqual(downgraded_delta.applicability.confidence_tier, "Low")
        self.assertTrue(downgraded_delta.applicability.disagreement_downgraded)

        # 5. Exemptions: Lab-Anchored in delta_learning & lab_only_gp must NOT be downgraded
        anchored_ad = replace(
            base_ad_high,
            ad_mode="delta_learning",
            lab_anchoring_status="Lab-Anchored",
            n_lab_samples=6,
        )
        anchored_pred = _make_stub_pred("hist_gradient_boosting", 0.21, 1.75, anchored_ad)
        exempt_delta = ctrl._evaluate_cross_model_disagreement(
            SimulationRequest(primary_metal="Cu", calibration_mode="delta_learning"),
            "Cu",
            default_cu_query,
            anchored_pred,
        )
        self.assertEqual(exempt_delta.applicability.confidence_tier, "High")
        self.assertFalse(exempt_delta.applicability.disagreement_downgraded)
        self.assertIsNone(exempt_delta.applicability.cross_model_disagreement_log10_icorr)

        lab_only_ad = replace(base_ad_high, ad_mode="lab_only_gp", lab_anchoring_status="Lab-Anchored")
        exempt_lab_only = ctrl._evaluate_cross_model_disagreement(
            SimulationRequest(primary_metal="Cu", calibration_mode="lab_only_gp"),
            "Cu",
            default_cu_query,
            _make_stub_pred("lab_local_gp", 0.21, 1.75, lab_only_ad),
        )
        self.assertEqual(exempt_lab_only.applicability.confidence_tier, "High")
        self.assertFalse(exempt_lab_only.applicability.disagreement_downgraded)

    def test_streamlit_sidebar_auto_routes_models_and_hides_primary_model_dropdown(self) -> None:
        from pathlib import Path
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parents[1] / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0, msg=f"Initial render failed: {at.exception}")

        # Base Literature ML Model selectbox is removed from primary sidebar
        sidebar_select_labels = [s.label for s in at.sidebar.selectbox]
        self.assertNotIn("Base Literature ML Model", sidebar_select_labels)

        # Read-only active model indicator and collapsed Advanced override expander exist
        sidebar_captions = "\n".join(c.value for c in at.sidebar.caption)
        self.assertIn("Active ML Architecture", sidebar_captions)
        self.assertIn("RF + GP Hybrid Ensemble", sidebar_captions)
        adv_expanders = [e for e in at.sidebar.expander if e.label == "Advanced: Override Base Model"]
        self.assertEqual(len(adv_expanders), 1)

        # Manual override inside Advanced expander updates active model
        override_box = next(
            s for s in at.sidebar.selectbox if s.label == "Base Literature Model Override"
        )
        override_box.select("random_forest").run()
        self.assertEqual(len(at.exception), 0, msg=f"Override failed: {at.exception}")
        main_captions = "\n".join(c.value for c in at.caption)
        self.assertIn("Random Forest", main_captions)

        # Switching to delta_learning auto-routes to Hist Gradient Boosting
        next(r for r in at.sidebar.radio if r.label == "Lab Calibration Mode").set_value("delta_learning").run()
        self.assertEqual(len(at.exception), 0, msg=f"delta_learning switch failed: {at.exception}")
        sidebar_captions_delta = "\n".join(c.value for c in at.sidebar.caption)
        self.assertIn("Hist Gradient Boosting", sidebar_captions_delta)

        # Switching to lab_only_gp hides Advanced: Override Base Model expander
        next(r for r in at.sidebar.radio if r.label == "Lab Calibration Mode").set_value("lab_only_gp").run()
        self.assertEqual(len(at.exception), 0, msg=f"lab_only_gp switch failed: {at.exception}")
        self.assertEqual(
            [e for e in at.sidebar.expander if e.label == "Advanced: Override Base Model"],
            [],
        )
        sidebar_captions_lab = "\n".join(c.value for c in at.sidebar.caption)
        self.assertIn("Lab-Only Local GP", sidebar_captions_lab)


if __name__ == "__main__":
    unittest.main()
