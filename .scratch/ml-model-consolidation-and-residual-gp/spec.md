# Spec: ML Model Consolidation, Per-Metal Residual GP, Calibration Uncertainty & Plot Clarity

**Status:** ready-for-agent

## Problem Statement

As the Corrosion Simulator scales toward 1,000–2,000 literature records per core Metal (5,000–20,000+ combined rows) alongside small-sample user laboratory datasets ($n < 20$), researchers and process engineers encounter four coupled architectural and visualization limitations:

1. **Redundant and unscalable global ML model roster**: Offering six global literature models includes redundant or obsolete algorithms (`extra_trees`, `gradient_boosting`) and a global `gaussian_process` capped at 2,500 training rows that discards most data at $O(N^3)$ cost. Worse, the existing `rf_gp_ensemble` averages a full-data Random Forest 50/50 with a subsampled global GP, so as the literature database grows past several thousand rows, the GP half drags down accuracy and discards data while still taking half the prediction weight.
2. **Overly conservative or mis-scaled laboratory calibration uncertainty and small-sample feature space**: In `delta_learning` mode, total uncertainty is fixed as $\sigma_{\text{total}} = \sqrt{\sigma_{\text{prior}}^2 + \sigma_{\Delta}^2}$, so even when querying directly on a newly measured laboratory Exposure Condition, the combined $\pm 1\sigma$ uncertainty cannot shrink below the noisy literature prior's variance. Meanwhile, `lab_only_gp` uses the full 30–40+ dimensional global feature matrix for $n < 20$ datasets, diluting isotropic Matérn distance metrics with redundant or constant features, and reports its `model_family` under the global `"gaussian_process"` identifier slated for removal.
3. **Manual model selection friction and geometry-only Applicability Domain blind spots**: The sidebar forces users to manually select a `Base Literature ML Model` even though each Calibration Mode has a clear optimal pairing, and the Applicability Domain Confidence Tier (`High`, `Medium`, `Low`) evaluates only geometric coverage (Bounding Box and k-NN distance) without detecting regions where independent ML architectures sharply disagree (such as passivation transitions or conflicting literature).
4. **Unreadable 90% uncertainty bands and ambiguous Depth / Risk visual encoding**:
   - On single-Metal Polarization Curves, Monte Carlo sampling draws random $E_{\text{corr}}$ shifts across a fixed potential axis, smearing out the V-shaped cusp at $E_{\text{corr}}$ so the 90% band flattens into a full-height shaded wash across the background where the user cannot tell which curve it belongs to.
   - Across Single Metal and Galvanic views, high log-space $i_{\text{corr}}$ variance causes extreme 90th/95th-percentile Corrosion Depth trajectories to squash the median curve near the bottom axis without a central 50% core band for reference.
   - Corrosion Depth and Thickness Loss Risk plots currently sit side by side with a shared top legend rather than vertically separated plots with legends placed directly above each plot, and in both interactive charts and Technical Report dual-Y-axis overlays, `d_crit` shares the same red color family as Thickness Loss Risk while the overlay legend omits the Risk curve.

## Solution

Consolidate the global literature ML suite to three complementary architectures, upgrade their interplay with laboratory calibration and Applicability Domain scoring, automate mode-based model routing in the UI, and redesign polarization and kinetics uncertainty plots for immediate physical clarity:

1. **Three-Model Consolidation & Two-Stage Per-Metal OOB Residual GP**: Retain only `random_forest` (fast interpretable baseline), `hist_gradient_boosting` (primary high-accuracy prior for 5,000–20,000+ records and Delta-Learning), and `rf_gp_ensemble` (smooth response and Bayesian extrapolation uncertainty). Upgrade `rf_gp_ensemble` to a **Global Random Forest (`oob_score=True`) + Per-Metal Low-Dimensional Zero-Mean OOB Residual GP** so cross-metal descriptor transfer stays intact while each Metal's residual GP trains on up to 2,000 out-of-bag residuals in low-dimensional exposure space without subsampling loss.
2. **Calibration Uncertainty Shrinkage & Low-Dimensional Lab-Only GP**: Update `delta_learning` uncertainty combination with **Prior Variance Shrinkage** weighted by the residual GP's local epistemic variance reduction, allowing 90% bands to tighten toward laboratory precision near measured points while reverting smoothly to $\sqrt{\sigma_{\text{prior}}^2 + \sigma_{\Delta}^2}$ far away. Switch `lab_only_gp` to the low-dimensional physically floored Exposure Condition space and decouple its reported label (`lab_local_gp`) from the global literature model enum.
3. **Mode-Driven Auto-Selection & Cross-Model Disagreement Guardrail**: Remove `Base Literature ML Model` from the main sidebar and auto-select the best model per Calibration Mode (`literature_only` $\to$ `rf_gp_ensemble`, `delta_learning` $\to$ `hist_gradient_boosting`, `lab_only_gp` $\to$ `lab_local_gp`), keeping an optional manual override inside a collapsed `Advanced` expander. Add non-blocking cross-model disagreement checks between cached `hist_gradient_boosting` and `rf_gp_ensemble` predictions to downgrade `Confidence Tier` when unanchored literature models diverge sharply.
4. **Clearer Polarization, Depth, Risk, and Report Overlay Plots**: Render Polarization Curve uncertainty as overpotential-anchored ($\eta = E - E_{\text{corr}}$) dual V-shaped bands (50% core + 90% outer) paired with 2D $(E_{\text{corr}}, i_{\text{corr}})$ crosshair error bars; stack Corrosion Depth (top) and Thickness Loss Risk (bottom) as separate figures with individual top legends and dual 50%/90% bands; enforce distinct site-wide colors/line styles (`Depth` deep blue solid, `d_crit` dark slate-gray dashed, `Risk` burnt orange dash-dot); and unify left/right axis entries in the Technical Report dual-Y-axis overlay legend.
5. **Prebuilt Artifact & Cache Schema Upgrade**: Prebuild all three retained global models by default and bump the processed artifact manifest schema version so stale caches retrain automatically.

## User Stories

1. As a corrosion researcher scaling the literature database to 1,000–2,000 records per core Metal, I want redundant and non-scalable global models (`extra_trees`, `gradient_boosting`, global `gaussian_process`) removed, so that model selection is clear and offline artifact builds do not waste time training obsolete models.
2. As a corrosion researcher running fast exploratory checks or inspecting global feature importances, I want `random_forest` retained as a fast, interpretable baseline model, so that I can verify which chemical and material descriptors drive predictions.
3. As a process engineer working with 5,000–20,000+ combined literature rows and `quality_score` weighting, I want `hist_gradient_boosting` retained as the primary high-capacity literature model, so that large-scale trends and high-order interactions across pH, temperature, oxidizers, and solvent regimes are captured accurately.
4. As a user screening conditions in pure literature mode without laboratory data, I want `rf_gp_ensemble` to use a global Random Forest with out-of-bag (OOB) residuals feeding per-Metal low-dimensional Zero-Mean Matérn GPs, so that predictions remain smooth across continuous variables, use 100% of each Metal's data up to 2,000 rows, and expand uncertainty smoothly outside the training domain.
5. As a researcher evaluating a rare Metal with fewer than 4 valid OOB residual rows, I want `rf_gp_ensemble` to fall back gracefully to the global Random Forest prediction for that Metal, so that cross-material physical descriptor transfer still produces stable predictions without GP instability.
6. As an experimentalist calibrating literature trends with a small lab dataset ($n < 20$) in `delta_learning` mode, I want the combined uncertainty near my measured lab conditions to shrink toward my laboratory measurement precision via Prior Variance Shrinkage, so that my 90% confidence bands actually reflect the reduced uncertainty gained from running lab experiments.
7. As an experimentalist extrapolating far away from my lab points in `delta_learning` mode, I want the shrinkage factor to smoothly vanish toward zero so that $\Delta \to 0$ and total uncertainty expands back to $\sqrt{\sigma_{\text{prior}}^2 + \sigma_{\Delta}^2}$, preserving conservative extrapolation guardrails.
8. As an experimentalist studying a novel electrolyte in `lab_only_gp` mode with $n < 20$ points, I want the local GP to operate in a low-dimensional, physically floored Exposure Condition feature space rather than a bloated 30–40+ dimensional matrix, so that GP distance metrics remain sensitive and interpolation accuracy improves.
9. As a user of the Streamlit application, I want the `Base Literature ML Model` dropdown removed from the primary sidebar and replaced by automatic Calibration Mode routing (`literature_only` $\to$ `rf_gp_ensemble`, `delta_learning` $\to$ `hist_gradient_boosting`, `lab_only_gp` $\to$ `lab_local_gp`) with a concise active-model indicator, so that I always get the scientifically appropriate architecture without manual tuning.
10. As an advanced user who occasionally wants to inspect Random Forest feature importances or compare base priors manually, I want an optional model override inside a collapsed-by-default `Advanced: Override Base Model` expander (hidden in `lab_only_gp` mode), so that flexibility remains available without cluttering the primary workflow.
11. As an engineer assessing prediction reliability (`Low`, `Medium`, `High` Confidence Tier) in unanchored literature regions, I want the simulator to cross-check `hist_gradient_boosting` against `rf_gp_ensemble` whenever both cached models exist and downgrade the Confidence Tier if their predicted $\log_{10} i_{\text{corr}}$ or $E_{\text{corr}}$ diverges sharply, so that hidden regime boundaries or conflicting literature are never mislabeled as `High` confidence.
12. As an experimentalist evaluating a `Lab-Anchored` condition in `delta_learning` mode or running `lab_only_gp`, I want my prediction exempted from literature cross-model disagreement downgrades, so that conflicting literature models never penalize confidence at conditions directly measured in my laboratory.
13. As a user inspecting the Single Metal Polarization Curve, I want the uncertainty shading around `Net |i_net|` to follow the V-shaped curve along overpotential $\eta = E - E_{\text{corr}}$ with both a 50% core band (P25–P75) and a 90% outer band (P05–P95), paired with 2D crosshair error bars at $(E_{\text{corr}}, i_{\text{corr}})$, so that the V-cusp remains sharp and I can immediately separate potential uncertainty from current uncertainty.
14. As a user reviewing Single Metal and Galvanic time-resolved trajectories, I want Corrosion Depth and Thickness Loss Risk presented as two vertically stacked charts—each with its own title and legend positioned directly above the chart and displaying both 50% core and 90% outer bands—so that extreme P95 tails do not hide the median trend and each curve's band is unmistakable.
15. As a reader comparing interactive charts and exported Technical Reports, I want `Corrosion Depth` (deep blue solid), `d_crit` (dark slate-gray dashed), and `Thickness Loss Risk R(t)` (burnt orange dash-dot) to use distinct, non-colliding colors and line styles everywhere—and for the Report's dual-Y-axis overlay plot to include both left-axis and right-axis curves in a single unified legend—so that every curve and its corresponding axis are immediately legible.
16. As an application maintainer and Windows `.exe` user, I want all three retained models prebuilt by default and the artifact manifest schema version incremented, so that updating the application automatically flags legacy v2 caches as stale and rebuilds compatible v3 models without loading mismatched artifacts.

## Implementation Decisions

- **Global Model Family Consolidation**:
  - Restrict the global literature `ModelFamily` type and supported family registry to three families: `random_forest`, `hist_gradient_boosting`, and `rf_gp_ensemble`.
  - Remove `extra_trees`, `gradient_boosting`, and global `gaussian_process` training/prediction branches from the global ML predictor and artifact builder.
  - Define an explicit non-global label (`lab_local_gp`) for standalone laboratory GP predictions emitted by `lab_only_gp` calibration so `LabCalibratedPredictor` never depends on the removed global `gaussian_process` family.

- **Two-Stage Per-Metal OOB Residual GP (`rf_gp_ensemble`)**:
  - Stage 1 trains a global `RandomForestRegressor` on 100% of the filtered training matrix with `oob_score=True` and `quality_score` sample weights, retaining cross-material descriptor generalization and native `feature_importances()`.
  - Compute out-of-bag residuals $r_i = y_i - \hat{y}_{\text{RF}}^{\text{OOB}}(x_i)$ per target, dropping any rows where the OOB prediction is non-finite (which can occur on very small datasets). This prevents in-sample residual collapse under `min_samples_leaf=1`.
  - Stage 2 groups valid OOB residuals by `metal`. For each Metal with at least 4 valid OOB residual rows, fit a low-dimensional Zero-Mean Gaussian Process (`normalize_y=False`, `ConstantKernel * Matern(nu=2.5) + WhiteKernel`) using the physically floored exposure-condition transformer (reusing the residual condition transformer design from laboratory calibration), heteroscedastic `alpha` noise weights derived from `quality_score`, and a per-Metal cap of 2,000 training rows (deterministic quantile-stratified subsampling if a single Metal exceeds 2,000 rows).
  - For Metals with fewer than 4 valid OOB residual rows (or unseen Metals at inference), skip Stage 2 residual adjustment and pass through the global RF mean and std unchanged.
  - Combine Stage 1 and Stage 2 predictions via $\hat{y} = \hat{y}_{\text{RF}} + \hat{r}_{\text{GP}}$ and $\sigma_{\text{total}} = \sqrt{\sigma_{\text{RF}}^2 + \sigma_{\text{GP}}^2}$, preserving ADR-0003 log-space $i_{\text{corr}}$ semantics and physical bounds clipping.

- **Laboratory Calibration Formula & Low-Dimensional `lab_only_gp`**:
  - **Prior Variance Shrinkage in `delta_learning`**: During `delta_learning` fitting for targets using `gp_residual` ($n_{\text{lab}} \ge 4$), record the prior residual variance amplitude $\sigma_{\Delta,\text{prior}}^2$ (learned `ConstantKernel` signal variance) and observation noise floor $\sigma_{\text{noise}}^2$ (learned `WhiteKernel` noise level) from the fitted residual GP. At prediction time, compute epistemic coverage $c(x) = \text{clip}\!\left(1 - \frac{\max(\sigma_{\Delta}^2(x) - \sigma_{\text{noise}}^2,\; 0)}{\max(\sigma_{\Delta,\text{prior}}^2,\; 10^{-12})},\; 0,\; 1\right)$ and total standard deviation $\sigma_{\text{total}}(x) = \sqrt{(1 - c(x))\,\sigma_{\text{prior}}^2(x) + \sigma_{\Delta}^2(x)}$. Ensure `constant_bias` ($1 \le n_{\text{lab}} < 4$) and `prior_only` ($n_{\text{lab}} = 0$) modes retain their existing combination behavior without variance erasure.
  - **Low-Dimensional `lab_only_gp`**: Replace the full global feature extractor inside `lab_only_gp` with per-Metal (or shared low-dimensional when cross-metal) physically floored exposure-condition transformation matching the residual condition transformer dimensionality, and emit `model_family="lab_local_gp"` on returned `MLParameterPrediction` instances.

- **Mode-Driven Automatic Model Routing & UI Controls**:
  - Define a deterministic mode-to-model mapping at the controller/UI boundary:
    - `literature_only` $\to$ `rf_gp_ensemble`
    - `delta_learning` $\to$ `hist_gradient_boosting`
    - `lab_only_gp` $\to$ `lab_local_gp` (bypassing global base model routing)
  - Update `SimulationRequest` defaults and validation so automated mode routing sets the active base model cleanly when `ml_model_family` is not explicitly overridden, while still honoring explicit programmatic `ml_model_family` values from the three supported global families for API compatibility and cross-validation comparisons.
  - Remove the `Base Literature ML Model` selectbox from the primary sidebar under **Parameter Source**. Display a concise read-only status line showing the active model selected by the current Calibration Mode, and place an optional manual override inside a collapsed-by-default `Advanced: Override Base Model` expander (hidden when `calibration_mode == "lab_only_gp"`).

- **Cross-Model Disagreement Downgrade in Applicability Domain**:
  - Extend `ApplicabilityDomainResult` with optional backward-compatible fields: `cross_model_disagreement_log10_icorr` (`float | None = None`), `cross_model_disagreement_ecorr_V` (`float | None = None`), and `disagreement_downgraded` (`bool = False`).
  - In the platform controller's ML resolution path, evaluate cross-model disagreement between `hist_gradient_boosting` and `rf_gp_ensemble` **only** when:
    1. `calibration_mode == "literature_only"`, OR `calibration_mode == "delta_learning"` with `lab_anchoring_status != "Lab-Anchored"`; AND
    2. The peer model (`hist_gradient_boosting` when running `rf_gp_ensemble`, or vice versa; or both when running `random_forest`) is already loaded in memory or exists prebuilt on disk—never trigger synchronous training of a missing peer model during prediction.
  - Compute $\Delta_{\log i} = |\mu_{\log i, \text{HGB}} - \mu_{\log i, \text{RF+RGP}}|$ and $\Delta_E = |\mu_{E, \text{HGB}} - \mu_{E, \text{RF+RGP}}|$.
  - Apply deterministic downgrade rules after geometric AD evaluation:
    - If $\Delta_{\log i} > 1.0$: set `confidence_tier = "Low"` and `disagreement_downgraded = True`.
    - Else if $\Delta_{\log i} > 0.5$ or $\Delta_E > 0.08$: downgrade `confidence_tier` by one step (`High` $\to$ `Medium`, `Medium` $\to$ `Low`) and set `disagreement_downgraded = True`.
  - When downgraded, append a clear warning citing the disagreement magnitudes and an experimental verification recommendation.

- **Overpotential-Anchored Polarization Bands & Crosshair Error Bars**:
  - Extend `MonteCarloPolarizationBand` to store overpotential-anchored net polarization quantiles (`overpotential_V`, `eta_log10_abs_net_p05`, `eta_log10_abs_net_p25`, `eta_log10_abs_net_p75`, `eta_log10_abs_net_p95`) alongside scalar summary intervals for `Ecorr` (P05, P25, P75, P95) and `log10_icorr` (P05, P25, P75, P95), retaining backward-compatible potential-grid fields if downstream callers read them.
  - Anchor the Tafel plot band along $E = E_{\text{corr, median}} + \eta$ using the $(i_{\text{corr}}, \beta_a, \beta_c)$ Monte Carlo draws evaluated at $\eta$ so every draw shares its corrosion cusp at $\eta = 0$, rendering a deeper 50% core band (P25–P75) inside a lighter 90% outer band (P05–P95) with thin boundary traces labeled explicitly as `Net |i_net| 50% Band (p25–p75)` and `Net |i_net| 90% Band (p05–p95)`.
  - Attach horizontal (`Ecorr` 90% interval) and vertical (`icorr` 90% interval) error bars to the $(E_{\text{corr}}, i_{\text{corr}})$ diamond marker in both interactive Plotly charts and Matplotlib report figures.

- **Stacked Depth / Risk Layout, Dual Bands, Palette Standardization & Report Legend**:
  - Extend `MonteCarloKineticsResult` with `depth_p25_nm`, `depth_p75_nm`, `risk_p25`, and `risk_p75` arrays so kinetics charts render dual 50% (P25–P75) + 90% (P05–P95) uncertainty bands.
  - In the controller and Streamlit UI for both Single Metal and Galvanic views, split the side-by-side Depth/Risk figure into two standalone vertically stacked figures (`Corrosion Depth` above, `Thickness Loss Risk` below), each configured with its own title and horizontal legend placed directly above its plot area (while preserving backward compatibility on payload attributes or updating UI/controller contracts cleanly across Single Metal and Galvanic views).
  - Standardize color and line-style encoding across interactive Plotly figures and Technical Report Matplotlib figures:
    - `Corrosion Depth`: deep blue solid (`#1f77b4`, solid) with layered light-blue 50%/90% bands; in Galvanic comparisons, distinguish Uncoupled (dashed slate/steel blue) vs. Coupled (deep blue solid with dual bands).
    - `Critical Thickness Threshold (d_crit)`: dark slate-gray dashed (`#475569`, dashed).
    - `Thickness Loss Risk R(t)`: burnt orange dash-dot (`#d95f02`, dash-dot) with layered light-orange 50%/90% bands; in Galvanic comparisons, distinguish Uncoupled (dotted muted amber/gray) vs. Coupled (burnt orange dash-dot with dual bands).
    - `R(d_crit) = 0.50` threshold line: light gray dotted (`#94a3b8`, dotted).
  - In the Technical Report composite figure's dual-Y-axis Depth & Risk overlay panel, apply the exact same deep blue solid (`Depth`) / dark slate-gray dashed (`d_crit`) / burnt orange dash-dot (`Risk R(t)`) styling, include dual 50%/90% Depth shading (or clear Depth band), and merge handles/labels from both left and right axes into one unified legend explicitly denoting `(left)` and `(right)` axis association.

- **Artifact Prebuild & Manifest Schema Migration**:
  - Update the default prebuilt model tuple in the artifact builder to `("random_forest", "hist_gradient_boosting", "rf_gp_ensemble")`.
  - Increment `MANIFEST_SCHEMA_VERSION` from `2` to `3` so existing v2 processed caches automatically evaluate as `stale` and rebuild with the upgraded three-model suite.

## Testing Decisions

- **Testing Philosophy**: Verify observable external behavior, statistical contracts, and visual/serialization structures through the highest stable public seams (`HybridMLPredictor`, `LabCalibratedPredictor`, `PlatformController`, artifact build pipeline, and Technical Report renderer) rather than asserting private helper internals or scikit-learn object internals.
- **Primary Test Seams**:
  1. **Global ML Consolidation & Per-Metal OOB Residual GP Seam** (`HybridMLPredictor` & cross-validation):
     - Confirm only the three retained model families are accepted and rejected legacy names raise `ValueError`.
     - Verify `rf_gp_ensemble` fits 100% of rows on its RF stage, tracks per-Metal OOB residuals accurately without in-sample collapse, falls back smoothly to RF for rare Metals ($n < 4$), caps per-Metal GP at 2,000 samples without discarding data for Metals $\le 2,000$ rows, supports `feature_importances()`, and produces smooth continuous curves with expanding extrapolation uncertainty.
  2. **Lab Calibration Uncertainty & Feature-Space Seam** (`LabCalibratedPredictor` & `propagate_uncertainty_monte_carlo`):
     - Verify `delta_learning` Prior Variance Shrinkage strictly tightens $\sigma_{\text{total}}$ toward lab precision when querying directly on measured lab conditions (compared to uncalibrated noisy priors) while smoothly reverting to $\sqrt{\sigma_{\text{prior}}^2 + \sigma_{\Delta}^2}$ and $\Delta \to 0$ far outside lab coverage.
     - Verify `lab_only_gp` runs in low-dimensional Exposure Condition space, preserves $n < 3$ error/fallback handling, and emits `model_family="lab_local_gp"`.
  3. **Controller Routing, Cross-Model Disagreement & Artifact Freshness Seam** (`PlatformController`, artifact builder, `SimulationRequest`):
     - Verify mode-based default model routing and optional override behavior.
     - Verify cross-model disagreement downgrades `Confidence Tier` when cached `hist_gradient_boosting` and `rf_gp_ensemble` predictions diverge beyond thresholds in `literature_only` or unanchored `delta_learning`, stays silent/unblocked when a peer model is not prebuilt, and exempts `Lab-Anchored` and `lab_only_gp` queries.
     - Verify artifact prebuild writes only the three retained `.joblib` models and marks legacy v2 manifests as `stale` under `MANIFEST_SCHEMA_VERSION == 3`.
  4. **Polarization & Kinetics Plot Structure Seam** (`propagate_uncertainty_monte_carlo`, `propagate_kinetics_uncertainty_monte_carlo`, controller Plotly figure builders, and report Matplotlib composite renderer):
     - Verify `MonteCarloPolarizationBand` preserves the V-cusp at $\eta = 0$ (`eta_log10_abs_net_p05 <= eta_log10_abs_net_p25 <= median <= eta_log10_abs_net_p75 <= eta_log10_abs_net_p95`) and includes `Ecorr` / `icorr` interval bounds for crosshair error bars.
     - Verify Single Metal and Galvanic views return vertically separated Depth and Risk Plotly figures with top-positioned legends, distinct Depth vs. Risk vs. `d_crit` styling, and 50%/90% percentile traces.
     - Verify the Technical Report composite figure legend includes both Depth (left axis) and Risk (right axis) traces with distinct colors and line styles.
- **Prior Art in Codebase**: Follow the established `unittest` test suite structure covering ML prediction/calibration, kinetics Monte Carlo, artifact builds, report generation, and end-to-end Streamlit platform controller workflows, updating any existing assertions that enumerate all six legacy model families or expect single combined side-by-side Depth/Risk Plotly figures.

## Out of Scope

- Adding new ML algorithm families outside the three retained models (e.g., neural networks, XGBoost, LightGBM, or sparse variational GPs).
- Rewriting non-aqueous electrochemistry physics (reference electrode conversions, IR-drop compensation, or non-aqueous Pourbaix boundaries).
- Multi-output correlated GP covariances across $(E_{\text{corr}}, \log_{10} i_{\text{corr}}, \beta_a, \beta_c)$ during Monte Carlo sampling; parameters remain independently sampled per ADR-0003 while cross-branch V-cusp consistency is solved via overpotential anchoring.
- Automated hyperparameter optimization (grid/random search) for `HistGradientBoostingRegressor` or `RandomForestRegressor` beyond updating `oob_score=True` and per-Metal residual GP bounds.

## Further Notes

- All changes must continue to adhere strictly to ADR-0001 (four-parameter Wagner-Traud kinetic seam), ADR-0002 (time-resolved kinetics anchored to $\text{CR}_0$), ADR-0003 ($\log_{10} i_{\text{corr}}$ log-space regression and physics-clipped Monte Carlo propagation), and the PyInstaller external data directory contract (`data/raw/`, `data/processed/`, `data/lab/`).
- Existing pre-built tests or documentation referencing the legacy six-model list (`extra_trees`, `gradient_boosting`, global `gaussian_process`) or `MANIFEST_SCHEMA_VERSION = 2` will need synchronized updates during implementation.
