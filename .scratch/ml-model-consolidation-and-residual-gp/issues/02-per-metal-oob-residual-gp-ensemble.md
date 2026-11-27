# 02: Upgrade rf_gp_ensemble to Global RF + per-Metal low-dimensional OOB residual GP

**What to build:** Replace the 50/50 parallel blend in `rf_gp_ensemble` with a scalable two-stage **Global Random Forest + Per-Metal Low-Dimensional Zero-Mean Out-of-Bag (OOB) Residual Gaussian Process** architecture. Stage 1 fits a full-dataset weighted Random Forest (`oob_score=True`) to retain cross-Metal descriptor transfer and native feature importances; Stage 2 fits per-Metal Zero-Mean Matérn($\nu=2.5$) residual GPs on valid OOB residuals in a compact, physically floored Exposure Condition space so predictions stay smooth, utilize up to 2,000 rows per Metal without subsampling dilution, and widen uncertainty naturally when extrapolating.

**Blocked by:** 01 (Consolidate global ML models to 3 families, decouple lab_local_gp, and bump manifest schema to v3)

**Status:** ready-for-agent

- [ ] Stage 1 fits a global `RandomForestRegressor` with `oob_score=True` and normalized `quality_score` sample weights on 100% of filtered training records per target, exposing valid native `feature_importances()`.
- [ ] Out-of-bag residuals $r_i = y_i - \hat{y}_{\text{RF}}^{\text{OOB}}(x_i)$ are computed per target (filtering any non-finite OOB entries on small folds/datasets) to avoid in-sample residual collapse under `min_samples_leaf=1`.
- [ ] For each Metal with at least 4 valid OOB residual rows, Stage 2 transforms Exposure Conditions into the low-dimensional physically floored residual space and fits a Zero-Mean `ConstantKernel * Matern(nu=2.5) + WhiteKernel` GP (`normalize_y=False`) with `quality_score`-derived `alpha` noise weights, capped deterministically at 2,000 training rows per Metal.
- [ ] Metals with fewer than 4 valid OOB residual rows (and unseen Metals at inference time) gracefully pass through the Stage 1 global Random Forest mean and standard deviation without raising errors.
- [ ] Predictions combine Stage 1 and Stage 2 via $\hat{y} = \hat{y}_{\text{RF}} + \hat{r}_{\text{GP}}$ and $\sigma_{\text{total}} = \sqrt{\sigma_{\text{RF}}^2 + \sigma_{\text{GP}}^2}$ with ADR-0003 physical bounds clipping, producing smooth continuous response sweeps and expanding $\sigma_{\text{total}}$ far from observed Exposure Conditions.
- [ ] Leak-free cross-validation and unit tests verify OOB residual behavior, rare-Metal fallback, 2,000-row per-Metal cap, and combined mean/std calibration.
