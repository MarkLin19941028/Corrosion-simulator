# 03: Solvent System UI controls, non-aqueous physics warnings, and report metadata

**What to build:** Expose Solvent System controls (`Aqueous / Semi-aqueous / Non-aqueous`, organic co-solvent selector, and water-content wt% slider) in the Streamlit sidebar and Cleaning Recipe comparison flow, display a prominent physics-limitation warning banner whenever a `non_aqueous` (`< 1%` wt% water) Exposure Condition is simulated, and include solvent regime metadata in all exported technical reports.

**Blocked by:** 02 (Solvent physical descriptors and water-weighted Pourbaix ML features)

**Status:** ready-for-agent

- [x] `SimulationRequest` and the Streamlit sidebar include a **Solvent System** section allowing users to choose `aqueous`, `semi_aqueous`, or `non_aqueous`, select an organic co-solvent, and adjust `water_content_wt_pct` (with regime and water content staying mutually consistent).
- [x] Selecting `non_aqueous` (`water_content_wt_pct < 1%`) renders a prominent warning banner in Single Metal and Galvanic views stating that aqueous pH, V vs. SHE, and aqueous Pourbaix assumptions are not strictly valid in this phase.
- [x] Cleaning Recipe comparison supports comparing recipes across different solvent regimes / water contents side by side.
- [x] Exported HTML, PDF, JSON, and CSV reports include `solvent_regime`, `solvent_subclass`, `organic_solvent_type`, and `water_content_wt_pct` metadata, verified by controller and reporting tests.

## Comments

Implemented and verified:
- `corrosion_simulator/platform_controller.py`: added `resolve_consistent_solvent_settings`, `build_non_aqueous_physics_warning`, and `SimulationRequest` solvent fields (`solvent_regime`, `organic_solvent_type`, `water_content_wt_pct`, plus `base_solvent` and `solvent_subclass` properties); extended `available_options()` with `solvent_regimes` and `organic_solvent_types`, updated `_build_ml_query` and `_resolve_database_lookup`, emitted non-aqueous physics warnings in Single Metal and Galvanic payloads, and added solvent metadata to `to_summary_dict()`.
- `corrosion_simulator/reporting.py`: extended `CleaningRecipeSpec`, `RecipeEvaluationEntry`, and `compare_cleaning_recipes` to support per-recipe solvent regimes and water contents, and added `solvent_regime`, `solvent_subclass`, `organic_solvent_type`, and `water_content_wt_pct` metadata to JSON, CSV, HTML, and PDF exports.
- `app.py`: added the sidebar **Solvent System** controls (`Solvent Regime`, `Organic Co-Solvent`, bounded `Water Content (wt%)` slider), prominent non-aqueous warning banners in Single Metal and Galvanic views, and per-recipe solvent controls in Recipe Comparison.
- `tests/test_streamlit_platform.py` and `tests/test_recipe_comparison_and_reporting.py`: verified all 20 tests passing (`MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest tests.test_streamlit_platform tests.test_recipe_comparison_and_reporting -v`).
