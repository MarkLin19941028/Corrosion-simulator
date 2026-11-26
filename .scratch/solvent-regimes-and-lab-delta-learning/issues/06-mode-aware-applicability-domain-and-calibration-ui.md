# 06: Mode-aware Applicability Domain, Lab Calibration UI, and Residual Breakdown diagnostics

**What to build:** Expose **Lab Calibration** controls (`Literature Only`, `Delta-Learning`, `Lab-Only Local GP` plus `data/lab/` project selector) in the Streamlit sidebar, adapt Applicability Domain and Confidence Tier reporting to each calibration mode, and add an expandable **Residual Breakdown** diagnostic table in Single Metal and Galvanic views (plus report export metadata) so every calibration step is transparent.

**Blocked by:** 03 (Solvent System UI controls, non-aqueous physics warnings, and report metadata), 05 (Parameter-level Delta-Learning GP residual correction and Lab-Only Local GP)

**Status:** ready-for-agent

- [ ] Sidebar **Parameter Source** / **Lab Calibration** allows selecting any discovered `data/lab/<project>/` dataset and toggling between `Literature Only`, `Delta-Learning (Prior + Lab Residual)`, and `Lab-Only Local GP`.
- [ ] Applicability Domain reporting adapts per mode: `literature_only` uses literature AD; `delta_learning` reports both literature coverage and proximity to lab measurements (`Lab-Anchored` vs. reverting to literature prior); `lab_only_gp` evaluates bounding box and k-NN density strictly against the selected lab project's own records.
- [ ] Single Metal and Galvanic views display an expandable **Residual Breakdown** table in `Delta-Learning` mode showing `Prior Estimate`, `Lab Residual (Δ)`, `Calibrated Final`, `±1σ` uncertainties, and calibration method (`GP Residual` / `Constant Bias` / `Prior Only`) for all four electrochemical parameters.
- [ ] Exported technical reports record the active calibration mode, lab project name, and residual breakdown summary when lab calibration is enabled, verified by end-to-end controller and UI tests.
