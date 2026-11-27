# 05: Parameter-level Delta-Learning GP residual correction and Lab-Only Local GP

**What to build:** Implement parameter-level laboratory calibration supporting `literature_only`, `delta_learning` (Literature Base Prior + Zero-Mean Gaussian Process residual layer), and `lab_only_gp` (Local GP trained exclusively on the selected lab project) on the four electrochemical kinetic parameters $(E_{\text{corr}}, \log_{10} i_{\text{corr}}, \beta_a, \beta_c)$ before the physics solvers, with `n < 4` constant-bias fallback, combined uncertainty propagation $\sigma_{\text{total}} = \sqrt{\sigma_{\text{prior}}^2 + \sigma_{\Delta}^2}$, and ADR-0003 physical clipping.

**Blocked by:** 02 (Solvent physical descriptors and water-weighted Pourbaix ML features), 04 (External `data/lab/` project directory and dual-fingerprint Reload External Data)

**Status:** ready-for-agent

- [x] `delta_learning` computes per-target residuals $\Delta y = y_{\text{lab}} - \hat{y}_{\text{prior}}$ (using $\log_{10} i_{\text{corr}}$ in log space and excluding imputed Tafel slopes when measured lab slopes exist), fits a Zero-Mean Matérn + WhiteKernel GP when $n \ge 4$, and produces structured `ResidualBreakdown` metadata (`prior`, `delta`, `calibrated`, `method`, `n_lab`).
- [x] Queries near lab data points closely match lab measurements, while queries far from lab points smoothly revert toward $\Delta \to 0$ (literature prior) with expanded combined uncertainty $\sigma_{\text{total}} = \sqrt{\sigma_{\text{prior}}^2 + \sigma_{\Delta}^2}$.
- [x] When a target has $1 \le n < 4$ valid lab records, `delta_learning` automatically falls back to a `quality_score`-weighted constant bias offset rather than overfitting a GP; when $n = 0$, it passes through the prior unchanged.
- [x] `lab_only_gp` trains a standalone GP exclusively on the selected lab project's records (requiring $n \ge 3$ valid samples per target, surfacing a clear actionable error/fallback if insufficient).
- [x] All calibrated predictions enforce physical bounds clipping (`E_corr` in `[-2.5, 2.5]` V vs. SHE, `log10(i_corr)` in `[-6, 6]`, Tafel slopes in `[20, 500]` mV/dec) and flow seamlessly into Wagner-Traud, Faraday, galvanic `brentq`, time-resolved kinetics, and Monte Carlo propagation while preserving charge and mass conservation.

## Comments

Implemented and verified:
- `corrosion_simulator/lab_calibration.py`: added `CalibrationMode`, `CALIBRATION_MODES`, `InsufficientLabSamplesError`, `TargetResidualBreakdown`, `ResidualBreakdown`, and `LabCalibratedPredictor` supporting `literature_only`, `delta_learning` (Zero-Mean `ConstantKernel * Matern(nu=2.5) + WhiteKernel` GP when $n \ge 4$, `quality_score`-weighted constant bias fallback for $1 \le n < 4$, prior pass-through for $n = 0$, measured-only Tafel slope masking, $\sigma_{\text{total}} = \sqrt{\sigma_{\text{prior}}^2 + \sigma_{\Delta}^2}$ propagation, and smooth $\Delta \to 0$ reversion with expanded uncertainty on extrapolation), and `lab_only_gp` (standalone GP trained strictly on the selected lab project with $n \ge 3$ enforcement/actionable fallback).
- `corrosion_simulator/ml_predictor.py`: extended `MLParameterPrediction` with backward-compatible `calibration_mode`, `lab_project`, and `residual_breakdown` fields.
- `corrosion_simulator/platform_controller.py` and `corrosion_simulator/__init__.py`: added `calibration_mode` and `lab_project` to `SimulationRequest`, `ResolvedMetalParameters`, `to_parameter_table()`, and `to_summary_dict()`, added cached `get_calibrated_predictor(...)` integrated with dual-fingerprint `reload_external_data(...)` cache invalidation, and exported calibration types.
- `tests/test_ml_predictor.py` and `tests/test_streamlit_platform.py`: added `TestLabDeltaLearningAndLocalGP` and `test_controller_calibration_modes_end_to_end_physics_and_reload`; full suite `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest discover -s tests -v` passed (95/95 tests OK).
