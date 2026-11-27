# 01: Consolidate global ML models to 3 families, decouple lab_local_gp, and bump manifest schema to v3

**What to build:** Streamline the global literature ML predictor and offline artifact builder down to the three retained model families (`random_forest`, `hist_gradient_boosting`, and `rf_gp_ensemble`), removing obsolete or non-scalable families (`extra_trees`, `gradient_boosting`, and global `gaussian_process`). Decouple standalone laboratory GP predictions (`lab_only_gp`) onto a dedicated `lab_local_gp` identifier so calibration never relies on the removed global GP type, prebuild all three retained global models by default, and increment the processed artifact manifest schema version to `3` so legacy v2 caches automatically flag as `stale` and rebuild cleanly.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] Global literature `ModelFamily` and supported family registry include exactly `random_forest`, `hist_gradient_boosting`, and `rf_gp_ensemble`; passing `extra_trees`, `gradient_boosting`, or `gaussian_process` raises a clear `ValueError`.
- [ ] Training, prediction, cross-validation, and feature-importance paths remove obsolete branches for `extra_trees`, `gradient_boosting`, and global `gaussian_process` while preserving ADR-0003 $\log_{10}(i_{\text{corr}})$ regression and physical bounds clipping on the remaining families.
- [ ] `lab_only_gp` calibration reports `model_family="lab_local_gp"` on returned parameter predictions without violating global `ModelFamily` validation in predictors, controllers, or summary exports.
- [ ] Artifact prebuild defaults to all three retained global families (`random_forest`, `hist_gradient_boosting`, `rf_gp_ensemble`) and sets `MANIFEST_SCHEMA_VERSION` to `3`, flagging any existing v2 manifest as `stale` during freshness checks.
- [ ] Existing test suites and build/controller references are updated for the three-model roster and pass end-to-end.
