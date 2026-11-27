# 04: Mode-driven auto model routing, sidebar simplification & cross-model disagreement downgrade

**What to build:** Eliminate manual model-selection friction in the Streamlit sidebar by automatically routing each Calibration Mode to its optimal ML architecture (`literature_only` $\to$ `rf_gp_ensemble`, `delta_learning` $\to$ `hist_gradient_boosting`, `lab_only_gp` $\to$ `lab_local_gp`), while keeping an optional override tucked inside a collapsed `Advanced` expander. Pair this with a non-blocking cross-model disagreement check in the platform controller that cross-validates `hist_gradient_boosting` against `rf_gp_ensemble` on unanchored literature predictions and downgrades `Confidence Tier` when the two heterogeneous architectures diverge sharply.

**Blocked by:** 01 (Consolidate global ML models to 3 families, decouple lab_local_gp, and bump manifest schema to v3), 02 (Upgrade rf_gp_ensemble to Global RF + per-Metal low-dimensional OOB residual GP)

**Status:** ready-for-agent

- [ ] Controller and request helpers provide deterministic Calibration Mode auto-routing (`literature_only` $\to$ `rf_gp_ensemble`, `delta_learning` $\to$ `hist_gradient_boosting`, `lab_only_gp` $\to$ `lab_local_gp`) while honoring explicit programmatic overrides from the three retained `ModelFamily` options.
- [ ] The `Base Literature ML Model` selectbox is removed from the primary sidebar under Parameter Source, replaced by a concise read-only indicator of the automatically active model architecture and a collapsed-by-default `Advanced: Override Base Model` expander (hidden when `lab_only_gp` is active).
- [ ] `ApplicabilityDomainResult` includes backward-compatible optional diagnostics (`cross_model_disagreement_log10_icorr`, `cross_model_disagreement_ecorr_V`, and `disagreement_downgraded`).
- [ ] In `literature_only` mode, or in `delta_learning` when `lab_anchoring_status` is not `"Lab-Anchored"`, if the peer model (`hist_gradient_boosting` or `rf_gp_ensemble`) is already loaded in memory or prebuilt on disk, the controller compares predictions without ever triggering on-the-fly peer training when absent.
- [ ] Disagreement exceeding $1.0\text{ dec}$ in $|\Delta \log_{10} i_{\text{corr}}|$ sets `Confidence Tier` to `Low`; disagreement exceeding $0.5\text{ dec}$ in $|\Delta \log_{10} i_{\text{corr}}|$ or $0.08\text{ V}$ in $|\Delta E_{\text{corr}}|$ downgrades `Confidence Tier` by one tier (`High` $\to$ `Medium`, `Medium` $\to$ `Low`), appending an explanatory warning and experimental verification recommendation.
- [ ] Queries running `lab_only_gp` or marked `Lab-Anchored` in `delta_learning` are explicitly exempted from literature cross-model disagreement downgrades.
- [ ] Controller and UI tests cover auto-routing, manual override behavior, peer-absent no-op behavior, threshold downgrades, and `Lab-Anchored` / `lab_only_gp` exemptions.
