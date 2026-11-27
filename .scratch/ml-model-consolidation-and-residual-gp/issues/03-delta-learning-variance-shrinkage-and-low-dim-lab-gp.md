# 03: Delta-Learning Prior Variance Shrinkage & low-dimensional lab_only_gp

**What to build:** Refine small-sample laboratory calibration uncertainty and feature representation so `delta_learning` uncertainty tightens realistically around measured laboratory conditions and `lab_only_gp` avoids the curse of dimensionality on $n < 20$ datasets. When a `delta_learning` query lies near measured laboratory points, reduce the prior's contribution to total uncertainty proportionally to the residual GP's epistemic variance reduction; when extrapolating, smoothly expand back to $\sqrt{\sigma_{\text{prior}}^2 + \sigma_{\Delta}^2}$. Simultaneously transition `lab_only_gp` to the compact, physically floored Exposure Condition space.

**Blocked by:** 01 (Consolidate global ML models to 3 families, decouple lab_local_gp, and bump manifest schema to v3)

**Status:** ready-for-agent

- [ ] For `delta_learning` targets fitted with `gp_residual` ($n_{\text{lab}} \ge 4$), record the learned prior residual signal variance $\sigma_{\Delta,\text{prior}}^2$ and observation noise floor $\sigma_{\text{noise}}^2$ from the fitted kernel.
- [ ] During `delta_learning` prediction under `gp_residual`, compute epistemic variance coverage $c(x) \in [0, 1]$ and combine uncertainty as $\sigma_{\text{total}}(x) = \sqrt{(1 - c(x))\,\sigma_{\text{prior}}^2(x) + \sigma_{\Delta}^2(x)}$.
- [ ] Querying directly on or very near measured laboratory points yields strictly smaller $\sigma_{\text{total}}$ than a noisy uncalibrated prior, while extrapolating far from laboratory points smoothly reverts $\Delta \to 0$ and $\sigma_{\text{total}} \to \sqrt{\sigma_{\text{prior}}^2 + \sigma_{\Delta}^2}$.
- [ ] `constant_bias` ($1 \le n_{\text{lab}} < 4$) and `prior_only` ($n_{\text{lab}} = 0$) fallback paths preserve existing uncertainty propagation without applying variance shrinkage.
- [ ] `lab_only_gp` transforms Exposure Conditions into the low-dimensional physically floored feature space (supporting per-Metal or clean low-dimensional multi-Metal fitting) instead of the 30–40+ dimensional global feature matrix, while preserving $n < 3$ insufficient-sample errors/fallbacks and `model_family="lab_local_gp"`.
- [ ] End-to-end calibration and Monte Carlo propagation tests verify tighter 90% confidence intervals at anchored lab conditions and wide intervals on extrapolation.
