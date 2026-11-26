# 01: Solvent regime classification, water-content inference, and harmonization

**What to build:** Extend database harmonization so every Solution and electrochemical record is classified into `aqueous` ($\ge 90\%$ wt% water), `semi_aqueous` ($1\% \le \text{water} < 90\%$), or `non_aqueous` ($< 1\%$ wt% water) with a chemical-family `solvent_subclass`, a 3-tier `water_content_source` fallback (`reported`, `inferred_aqueous`, `inferred_regime_default`) that applies a `0.85` `quality_score` multiplier to regime-default inferences, and full backward compatibility with the existing ~2,000-row literature database and data dictionary.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [x] Harmonized Solution and electrochemical tables expose `solvent_regime`, `solvent_subclass`, `water_content_wt_pct`, and `water_content_source` columns, documented in the generated data dictionary.
- [x] Water-content thresholds classify `>= 90.0` wt% as `aqueous`, `1.0 <= water < 90.0` wt% as `semi_aqueous`, and `< 1.0` wt% as `non_aqueous`, with boundary values (`1.0`, `0.99`, `90.0`, `89.99`) verified by automated tests.
- [x] `solvent_subclass` assigns aqueous pH-based subclasses (`acidic`, `neutral`, `alkaline`, `buffered`) for `aqueous` Solutions and chemical-family subclasses (`glycol`, `alcohol`, `polar_aprotic`, `organic_acid`, `other_organic`) for `semi_aqueous` and `non_aqueous` Solutions.
- [x] Missing `water_content_wt_pct` follows the 3-tier fallback: explicit organic concentration converts to `100 - organic_pct` (`reported`, no penalty); DI water without organic co-solvent infers `100.0` (`inferred_aqueous`, no penalty); organic/semi/non-aqueous records without concentration infer regime defaults (`50.0` / `0.0`) with `inferred_regime_default` and multiply `quality_score` by `0.85` (clamped to `[0.05, 1.0]`).
- [x] Existing seed dataset and raw literature CSVs harmonize cleanly with zero regressions in existing harmonization tests.

## Comments

Implemented in `corrosion_simulator/database.py` (with exports in `corrosion_simulator/__init__.py`) and verified by `tests/test_database_harmonization.py` plus the full 78-test `unittest` suite (`0` failures, `0` warnings).
