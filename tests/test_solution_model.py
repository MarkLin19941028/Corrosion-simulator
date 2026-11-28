"""Unit tests for deep Solution and CleaningRecipe domain value objects."""

from __future__ import annotations

import unittest

from corrosion_simulator.ml_predictor import MLFeatureQuery
from corrosion_simulator.platform_controller import SimulationRequest
from corrosion_simulator.reporting import RecipeCandidate
from corrosion_simulator.solution import (
    CleaningRecipe,
    Solution,
    water_content_slider_bounds,
)


class TestSolutionAndCleaningRecipe(unittest.TestCase):
    def test_solution_defaults_and_solvent_regime_boundaries(self) -> None:
        default_sol = Solution()
        self.assertEqual(default_sol.solvent_regime, "aqueous")
        self.assertEqual(default_sol.organic_solvent_type, "None")
        self.assertAlmostEqual(float(default_sol.water_content_wt_pct), 100.0)
        self.assertEqual(default_sol.base_solvent, "DI water")
        self.assertEqual(default_sol.solvent_subclass, "acidic")
        self.assertFalse(default_sol.is_non_aqueous)
        self.assertIsNone(default_sol.non_aqueous_warning())

        self.assertEqual(Solution(water_content_wt_pct=90.0).solvent_regime, "aqueous")
        self.assertEqual(Solution(water_content_wt_pct=89.99).solvent_regime, "semi_aqueous")
        self.assertEqual(Solution(water_content_wt_pct=1.0).solvent_regime, "semi_aqueous")
        self.assertEqual(Solution(water_content_wt_pct=0.99).solvent_regime, "non_aqueous")

        semi_sol = Solution(solvent_regime="semi_aqueous", organic_solvent_type="EG")
        self.assertAlmostEqual(float(semi_sol.water_content_wt_pct), 50.0)
        self.assertEqual(semi_sol.solvent_subclass, "glycol")
        self.assertEqual(semi_sol.base_solvent, "DI water")

        nonaq_sol = Solution(
            solvent_regime="non_aqueous",
            organic_solvent_type="DMSO",
            water_content_wt_pct=0.5,
        )
        self.assertTrue(nonaq_sol.is_non_aqueous)
        self.assertEqual(nonaq_sol.base_solvent, "DMSO")
        self.assertEqual(nonaq_sol.solvent_subclass, "polar_aprotic")
        warning = nonaq_sol.non_aqueous_warning()
        self.assertIsNotNone(warning)
        self.assertIn("DMSO", warning or "")

    def test_solution_rejects_inconsistent_regime_and_water_content(self) -> None:
        with self.assertRaises(ValueError):
            Solution(solvent_regime="non_aqueous", water_content_wt_pct=50.0)
        with self.assertRaises(ValueError):
            Solution(solvent_regime="aqueous", water_content_wt_pct=85.0)
        with self.assertRaises(ValueError):
            Solution(pH=float("nan"))

    def test_solution_zeroes_concentration_when_species_type_is_none(self) -> None:
        sol = Solution(
            oxidizer_type="None",
            oxidizer_concentration=5.0,
            acid_type="",
            acid_concentration=2.5,
            inhibitor_type="none",
            inhibitor_concentration=1.2,
        )
        self.assertEqual(sol.oxidizer_type, "None")
        self.assertEqual(sol.oxidizer_concentration, 0.0)
        self.assertEqual(sol.acid_type, "None")
        self.assertEqual(sol.acid_concentration, 0.0)
        self.assertEqual(sol.inhibitor_type, "None")
        self.assertEqual(sol.inhibitor_concentration, 0.0)
        self.assertFalse(sol.has_inhibitor)

        inhibited = Solution(inhibitor_type="BTA", inhibitor_concentration=1.0)
        self.assertTrue(inhibited.has_inhibitor)

    def test_cleaning_recipe_and_recipe_candidate_validation(self) -> None:
        recipe = CleaningRecipe(
            solution=Solution(pH=4.5, oxidizer_type="H2O2", oxidizer_concentration=1.0),
            temperature_C=40.0,
            immersion_time_s=120.0,
        )
        self.assertAlmostEqual(recipe.temperature_C, 40.0)
        self.assertAlmostEqual(recipe.immersion_time_s, 120.0)

        with self.assertRaises(ValueError):
            CleaningRecipe(immersion_time_s=0.0)
        with self.assertRaises(TypeError):
            CleaningRecipe(solution="not-a-solution")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            RecipeCandidate(name="   ", recipe=recipe)
        with self.assertRaises(TypeError):
            SimulationRequest(primary_metal="Cu", recipe="not-a-recipe")  # type: ignore[arg-type]

    def test_water_content_slider_bounds_match_regime_thresholds(self) -> None:
        self.assertEqual(water_content_slider_bounds("aqueous", "None"), (90.0, 100.0, 100.0, 0.5))
        self.assertEqual(water_content_slider_bounds("aqueous", "EG"), (90.0, 100.0, 95.0, 0.5))
        self.assertEqual(water_content_slider_bounds("semi_aqueous", "EG"), (1.0, 89.9, 50.0, 0.5))
        self.assertEqual(water_content_slider_bounds("non_aqueous", "DMSO"), (0.0, 0.99, 0.0, 0.01))
        with self.assertRaises(ValueError):
            water_content_slider_bounds("supercritical")

    def test_ml_feature_query_for_solution_preserves_all_descriptors(self) -> None:
        sol = Solution(
            oxidizer_type="H2O2",
            oxidizer_concentration=1.5,
            acid_type="Citric acid",
            acid_concentration=0.8,
            inhibitor_type="BTA",
            inhibitor_concentration=0.2,
            pH=4.2,
            ORP_mV=320.0,
            solvent_regime="semi_aqueous",
            organic_solvent_type="EG",
            water_content_wt_pct=50.0,
        )
        q = MLFeatureQuery.for_solution("Cu", sol, temperature_C=35.0)
        self.assertEqual(q.metal, "Cu")
        self.assertAlmostEqual(q.pH, 4.2)
        self.assertAlmostEqual(q.temperature_C, 35.0)
        self.assertEqual(q.oxidizer_type, "H2O2")
        self.assertAlmostEqual(q.oxidizer_concentration, 1.5)
        self.assertEqual(q.acid_type, "Citric acid")
        self.assertAlmostEqual(q.acid_concentration, 0.8)
        self.assertEqual(q.inhibitor_type, "BTA")
        self.assertAlmostEqual(q.inhibitor_concentration, 0.2)
        self.assertAlmostEqual(float(q.ORP_mV), 320.0)
        self.assertEqual(q.organic_solvent_type, "EG")
        self.assertAlmostEqual(q.water_content_wt_pct, 50.0)
        self.assertEqual(q.base_solvent, "DI water")


if __name__ == "__main__":
    unittest.main()
