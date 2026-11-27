# 06: Mode-aware Applicability Domain, Lab Calibration UI, and Residual Breakdown diagnostics

**What to build:** Expose **Lab Calibration** controls (`Literature Only`, `Delta-Learning`, `Lab-Only Local GP` plus `data/lab/` project selector) in the Streamlit sidebar, adapt Applicability Domain and Confidence Tier reporting to each calibration mode, and add an expandable **Residual Breakdown** diagnostic table in Single Metal and Galvanic views (plus report export metadata) so every calibration step is transparent.

**Blocked by:** 03 (Solvent System UI controls, non-aqueous physics warnings, and report metadata), 05 (Parameter-level Delta-Learning GP residual correction and Lab-Only Local GP)

**Status:** ready-for-agent

- [x] Sidebar **Parameter Source** / **Lab Calibration** allows selecting any discovered `data/lab/<project>/` dataset and toggling between `Literature Only`, `Delta-Learning (Prior + Lab Residual)`, and `Lab-Only Local GP`.
- [x] Applicability Domain reporting adapts per mode: `literature_only` uses literature AD; `delta_learning` reports both literature coverage and proximity to lab measurements (`Lab-Anchored` vs. reverting to literature prior); `lab_only_gp` evaluates bounding box and k-NN density strictly against the selected lab project's own records.
- [x] Single Metal and Galvanic views display an expandable **Residual Breakdown** table in `Delta-Learning` mode showing `Prior Estimate`, `Lab Residual (Δ)`, `Calibrated Final`, `±1σ` uncertainties, and calibration method (`GP Residual` / `Constant Bias` / `Prior Only`) for all four electrochemical parameters.
- [x] Exported technical reports record the active calibration mode, lab project name, and residual breakdown summary when lab calibration is enabled, verified by end-to-end controller and UI tests.

## Comments

Implemented and verified:
- `corrosion_simulator/ml_predictor.py`: extended `ApplicabilityDomainResult` with backward-compatible mode-aware fields (`ad_mode`, `lab_in_bounding_box`, `lab_knn_distance`, `n_lab_samples`, `lab_anchoring_status`).
- `corrosion_simulator/lab_calibration.py` & `corrosion_simulator/__init__.py`: added `CALIBRATION_MODE_LABELS`, `RESIDUAL_METHOD_LABELS`, formatted `ResidualBreakdown.to_dataframe()` (`Parameter`, `Unit`, `Prior Estimate`, `Prior ±1σ`, `Lab Residual (Δ)`, `Residual ±1σ`, `Calibrated Final`, `Combined ±1σ`, `Method`, `Lab Samples (n)`), dual literature + lab AD proximity evaluation (`Lab-Anchored` / `Reverting to Literature Prior` / `No Lab Coverage`) in `delta_learning`, and strict lab-only AD metadata (`Lab-Anchored` / `Outside Lab Domain`) in `lab_only_gp`.
- `corrosion_simulator/platform_controller.py`: exposed `applicability_domain` / `primary_applicability_domain` / `secondary_applicability_domain` and galvanic residual breakdown dicts in `to_summary_dict()`.
- `corrosion_simulator/reporting.py`: added `_collect_calibration_metadata` and exported `calibration_mode`, `lab_project`, and residual breakdown summaries across JSON, CSV, HTML, and PDF reports.
- `app.py`: added sidebar **Lab Calibration** mode radio & `data/lab/` dataset selector, mode-aware AD status callouts in `_render_confidence_and_warnings`, and expandable **Residual Breakdown (Prior → Lab Δ → Calibrated Final)** tables in Single Metal and Galvanic views.
- `tests/test_ml_predictor.py`, `tests/test_recipe_comparison_and_reporting.py`, and `tests/test_streamlit_platform.py`: added mode-aware AD, multi-format report export, and Streamlit `AppTest` UI tests; full suite `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest discover -s tests -v` passed (98/98 tests OK).
