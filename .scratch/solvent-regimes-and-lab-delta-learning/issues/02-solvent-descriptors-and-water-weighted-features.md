# 02: Solvent physical descriptors and water-weighted Pourbaix ML features

**What to build:** Add a built-in solvent physicochemical property lookup table and integrate four continuous solvent descriptors (`water_content_wt_pct`, `effective_dielectric_constant`, `solvent_polarity_index`, `is_protic_solvent`) plus water-content-weighted Pourbaix passivation features (`pourbaix_passivation_indicator` and `pourbaix_pH_distance` scaled by `water_content_wt_pct / 100`) into the ML feature extractor, Applicability Domain checker, and offline artifact build pipeline without data leakage.

**Blocked by:** 01 (Solvent regime classification, water-content inference, and harmonization)

**Status:** ready-for-agent

- [x] A built-in solvent property lookup covers water and common semiconductor wet-clean organic solvents (e.g., EG, PG, IPA, EtOH, MeOH, DMSO, NMP, DMF, acetone) with dielectric constant, polarity index, and protic indicator, plus safe fallback for unseen organic solvents.
- [x] `MLFeatureQuery` and `ElectrochemicalFeatureExtractor` compute the 4 continuous solvent descriptors and scale Pourbaix passivation features by `clip(water_content_wt_pct / 100, 0, 1)`, fitting all medians and category vocabularies strictly on training folds during cross-validation.
- [x] `ApplicabilityDomainChecker` incorporates `water_content_wt_pct` and `organic_solvent_type` so queries outside training water-content ranges or with unseen co-solvents trigger clear warnings and appropriate Confidence Tiers.
- [x] Offline artifact rebuilds and all 6 ML model families train, save, load, and predict cleanly with the extended feature schema, verified by automated ML and artifact tests.

## Comments

Implemented in `corrosion_simulator/solvent_constants.py`, `corrosion_simulator/ml_predictor.py`, and `corrosion_simulator/build_artifacts.py`; rebuilt `data/processed/` across all 6 ML model families (`3,182` records, `21` materials) and verified via `tests/test_ml_predictor.py` (`12/12`) and `tests/test_extensibility_and_artifacts.py` (`11/11`).
