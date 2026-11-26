# Spec: Solvent Regime Modeling & Small-Sample Lab Delta-Learning

**Status:** ready-for-agent

## Problem Statement

Researchers and process engineers using the Corrosion Simulator face two related limitations when evaluating real semiconductor wet cleaning and etching processes:

1. **No solvent-system distinction or continuous water-content modeling**: The current harmonized database and ML predictor treat all Solutions primarily by pH regime (`acidic`, `neutral`, `alkaline`) and ignore solvent composition (`base_solvent`, `organic_solvent_type`, `organic_solvent_concentration`). When evaluating semi-aqueous formulations (water + organic co-solvents such as EG, DMSO, NMP, IPA) or screening non-aqueous formulations, the simulator cannot represent water content or solvent physicochemical properties, cannot warn when aqueous thermodynamic assumptions break down, and would suffer severe data sparsity if each organic solvent were split into isolated one-hot categories.
2. **Ineffective integration of small-sample laboratory datasets ($n < 20$)**: When a researcher investigates a specific Metal in a specific Solution and generates a small, high-precision set of Tafel Polarization Curve measurements ($n < 20$), mixing those rows directly into the ~2,000-row literature database dilutes their influence, triggers slow full-database retraining on every edit, and risks skewing predictions when the lab system differs from aqueous literature trends. Conversely, applying ML residual correction to *final* simulated outputs (Corrosion Depth or Galvanic Current Density) would break charge conservation, Faraday consistency, and area-ratio/time extrapolation.

## Solution

Introduce a two-phase extension that preserves the simulator's physics-informed grey-box architecture (ADR-0001, ADR-0002, ADR-0003):

- **Phase 1 — Solvent Regimes & Physicochemical Solvent Descriptors**:
  Classify every Solution into three water-content-anchored regimes—`aqueous` ($\ge 90\%$ wt% water), `semi_aqueous` ($1\% \le \text{water} < 90\%$), and `non_aqueous` ($< 1\%$ wt% water)—paired with a chemical-family `solvent_subclass`. Represent solvent chemistry to the ML predictor via continuous physicochemical descriptors (water content, effective dielectric constant, polarity index, protic indicator) and decay aqueous Pourbaix passivation features proportionally to water content. Handle missing literature water-content values with a transparent 3-tier provenance fallback and quality-score weighting. Reserve `non_aqueous` as a classified regime with explicit UI physics warnings rather than rewriting non-aqueous reference-electrode or IR-drop physics in this phase.
- **Phase 2 — Isolated Lab Data Management & Parameter-Level Delta-Learning**:
  Add an external `data/lab/<project>/` directory alongside `data/raw/` and `data/processed/` so laboratory small-sample datasets remain cleanly separated from the literature prior and reload in under a second via independent fingerprinting. Provide three transparent calibration modes operating strictly on the four electrochemical kinetic parameters $(E_{\text{corr}}, \log_{10} i_{\text{corr}}, \beta_a, \beta_c)$ **before** the physics solvers:
  1. `Literature Only` (existing base prior)
  2. `Delta-Learning (Prior + Lab Residual)` using a Zero-Mean Gaussian Process residual layer (falling back to weighted constant bias offset when $n < 4$) that smoothly reverts to the literature prior on extrapolation while expanding total uncertainty $\sigma_{\text{total}} = \sqrt{\sigma_{\text{prior}}^2 + \sigma_{\Delta}^2}$
  3. `Lab-Only Local GP` trained exclusively on the selected lab project when the chemical system is fundamentally distinct from aqueous literature.
  Expose mode-aware Applicability Domain diagnostics and an expandable **Residual Breakdown** table (`Prior → Lab Δ → Calibrated Final`) to prevent black-box operation.

## User Stories

1. As a process engineer, I want every Solution in the harmonized database to be classified into `aqueous` ($\ge 90\%$ wt% water), `semi_aqueous` ($1\% \le \text{water} < 90\%$), or `non_aqueous` ($< 1\%$ wt% water), so that I can filter, compare, and analyze Cleaning Recipes by solvent regime.
2. As a process engineer, I want each solvent regime to include a chemical-family `solvent_subclass` (such as `acidic`, `neutral`, `alkaline`, `buffered` for aqueous, and `glycol`, `alcohol`, `polar_aprotic`, `organic_acid`, `other_organic` for semi-aqueous and non-aqueous), so that I can group similar chemistries without fragmenting sparse data into dozens of isolated categories.
3. As a data curator, I want the database harmonizer to infer `water_content_wt_pct` when literature records omit exact percentages (via a 3-tier fallback tracked in `water_content_source`) and apply a `0.85` `quality_score` multiplier to regime-default inferences, so that existing ~2,000 literature records remain usable without masking uncertainty.
4. As a corrosion researcher, I want the ML predictor to use continuous solvent descriptors (`water_content_wt_pct`, `effective_dielectric_constant`, `solvent_polarity_index`, `is_protic_solvent`) derived from a built-in solvent property lookup table, so that predictions can generalize across co-solvents even with limited semi-aqueous training data.
5. As a corrosion researcher, I want aqueous Pourbaix passivation ML features (`pourbaix_passivation_indicator`, `pourbaix_pH_distance`) to be scaled by `water_content_wt_pct / 100`, so that the model reflects diminishing aqueous oxide passivation stability as water content decreases.
6. As a user of the Streamlit application, I want a **Solvent System** section in the sidebar where I can choose the solvent regime, organic co-solvent type, and water content (wt%), so that I can interactively simulate aqueous and semi-aqueous Exposure Conditions.
7. As a user of the Streamlit application, I want a prominent physics-limitation warning banner whenever I select `Non-aqueous` (`< 1%` wt% water), so that I am clearly informed that aqueous pH, V vs. SHE, and aqueous Pourbaix assumptions are not strictly valid in this phase.
8. As a process engineer, I want Cleaning Recipe comparisons and exported technical reports (HTML, PDF, JSON, CSV) to record solvent regime, co-solvent, and water content metadata, so that comparative assessments remain reproducible and auditable.
9. As an experimentalist, I want to store my small-sample laboratory Tafel datasets ($n < 20$) in project-specific subdirectories under `data/lab/<project>/`, so that I can manage multiple studies cleanly without polluting `data/raw/`.
10. As a Windows `.exe` user, I want `data/lab/` to live externally alongside `data/raw/` and `data/processed/`, so that I can add or edit lab CSVs without rebuilding the packaged application.
11. As an experimentalist, I want clicking **Reload External Data** to check independent fingerprints for `data/raw/` and `data/lab/`, so that editing lab CSVs refreshes the lab project list and residual models in under a second without triggering a 2–4 minute literature base-model rebuild.
12. As a user, I want the sidebar data status panel to display separate sync statuses for the literature prior (`data/raw/`) and the active lab project (`data/lab/<project>/`), so that I always know which dataset was reloaded and how many records it contains.
13. As an experimentalist, I want to run **Delta-Learning (Prior + Lab Residual)** on the four electrochemical parameters $(E_{\text{corr}}, \log_{10} i_{\text{corr}}, \beta_a, \beta_c)$ before the physics engine, so that my small dataset calibrates the local Exposure Condition while preserving 100% of Faraday's law, Wagner-Traud Polarization Curves, area-weighted galvanic current balance, and time-resolved kinetics extrapolation.
14. As an experimentalist, I want the Delta-Learning residual layer to use a Zero-Mean Gaussian Process (Matérn + WhiteKernel) in parameter space (with $\log_{10} i_{\text{corr}}$ in log space per ADR-0003), so that predictions fit my lab points smoothly near measured conditions and safely revert to the literature prior ($\Delta \to 0$) when extrapolating far from lab data.
15. As an experimentalist, I want Delta-Learning uncertainty to combine base-prior and residual uncertainty via $\sigma_{\text{total}} = \sqrt{\sigma_{\text{prior}}^2 + \sigma_{\Delta}^2}$ and propagate through Monte Carlo sampling with physical bounds clipping, so that 90% confidence bands (P05–P95) honestly reflect both literature and lab uncertainty.
16. As an experimentalist with very few early measurements ($n < 4$ valid lab records for a target), I want Delta-Learning to automatically fall back to a quality-weighted constant bias offset instead of overfitting a GP, so that calibration remains stable from the very first experiments.
17. As an experimentalist studying a fundamentally new electrolyte system, I want a one-click switch to **Lab-Only Local GP** mode, so that I can train a local GP solely on my selected `data/lab/<project>/` dataset without being pulled toward incompatible aqueous literature trends.
18. As a user in **Delta-Learning** mode, I want the Applicability Domain banner to report both literature coverage and distance to the nearest lab measurement (indicating when a query is `Lab-Anchored` vs. reverting to the literature prior), so that I know whether my prediction is directly supported by lab data.
19. As a user in **Lab-Only Local GP** mode, I want the Applicability Domain and Confidence Tier to be evaluated strictly against the selected lab project's own bounding box and k-NN density, so that literature coverage never masks extrapolation outside my lab experiments.
20. As a user inspecting calibrated results, I want an expandable **Residual Breakdown** table showing `Prior Estimate`, `Lab Residual (Δ)`, `Calibrated Final`, and `±1σ` uncertainties for all four parameters, so that the ML calibration remains completely transparent and auditable rather than a black box.

## Implementation Decisions

### Phase 1: Solvent Regimes & Physicochemical Descriptors

- **Solvent Regime Classification & Thresholds**:
  - Add canonical fields to harmonized Solution and electrochemical tables: `solvent_regime` (`aqueous`, `semi_aqueous`, `non_aqueous`), `solvent_subclass`, `water_content_wt_pct`, and `water_content_source`.
  - Regime boundaries:
    - `aqueous`: `water_content_wt_pct >= 90.0` (or no organic co-solvent present)
    - `semi_aqueous`: `1.0 <= water_content_wt_pct < 90.0`
    - `non_aqueous`: `water_content_wt_pct < 1.0`
  - Chemical-family `solvent_subclass` taxonomy:
    - For `aqueous`: `acidic`, `neutral`, `alkaline`, `buffered` (derived from pH / solution metadata).
    - For `semi_aqueous` and `non_aqueous`: `glycol`, `alcohol`, `polar_aprotic`, `organic_acid`, `other_organic` (mapped from normalized `organic_solvent_type` / `base_solvent`).
- **Missing Water-Content 3-Tier Fallback & Quality Weighting**:
  1. `"reported"`: Explicit water percentage or organic co-solvent concentration (wt% / vol% converted to `100 - organic_pct`). No quality penalty.
  2. `"inferred_aqueous"`: `base_solvent` is DI water (or aqueous default) with no organic co-solvent → `100.0` wt%. No quality penalty.
  3. `"inferred_regime_default"`: Organic co-solvent or semi/non-aqueous regime indicated without numeric concentration → assign regime default (`50.0` wt% for `semi_aqueous`, `0.0` wt% for `non_aqueous`) and multiply `quality_score` by `0.85` (clamped to `[0.05, 1.0]`).
  4. Truly unknown rows retain `NaN` for `water_content_wt_pct` and are imputed with training-fold medians inside the feature extractor.
- **Built-In Solvent Property Lookup & ML Feature Engineering**:
  - Create a dedicated solvent constants/properties module containing standard physicochemical properties for water and common semiconductor wet-clean organic solvents (e.g., EG, PG, IPA, EtOH, MeOH, DMSO, NMP, DMF, acetone, citric/oxalic organic media): pure dielectric constant $\varepsilon_r$, polarity index, and protic flag (`0.0` or `1.0`).
  - Extend `MLFeatureQuery` and `ElectrochemicalFeatureExtractor` (strictly leak-free: fit medians/vocabularies on training folds only) with 4 continuous solvent features:
    - `water_content_wt_pct`
    - `effective_dielectric_constant` (water-fraction-weighted mixture estimate between water and co-solvent)
    - `solvent_polarity_index` (water-fraction-weighted mixture polarity)
    - `is_protic_solvent` (fraction-weighted or binary protic indicator)
  - Scale `pourbaix_passivation_indicator` and `pourbaix_pH_distance` by `np.clip(water_content_wt_pct / 100.0, 0.0, 1.0)` so aqueous Pourbaix features smoothly decay toward zero as water content drops.
  - Extend `ApplicabilityDomainChecker` numeric/categorical checks to include `water_content_wt_pct` and `organic_solvent_type`.
- **Non-Aqueous Physics Guardrail**:
  - Do not alter Wagner-Traud, Faraday, or Pourbaix equations for non-aqueous electrochemistry in this phase; instead surface a clear warning in simulation results and UI whenever `solvent_regime == "non_aqueous"` (`water_content_wt_pct < 1%`).

### Phase 2: External `data/lab/` Directory, Dual-Fingerprint Reload & Delta-Learning

- **Directory & Path Resolution**:
  - Extend external data path resolution to manage three sibling directories under `<data_dir>/`: `raw/` (literature prior), `processed/` (cached Parquet & base `.joblib` models), and `lab/` (user laboratory projects).
  - Each subdirectory under `data/lab/<project>/` represents a selectable Lab Dataset (top-level CSVs directly under `data/lab/` map to a `"default"` project if present).
  - Harmonize lab CSVs using the same unit/reference-electrode/physical-bounds pipeline as `HarmonizedDatabase.from_directory`, preserving `quality_score` and `rejected_records` without merging into `data/raw/`.
- **Dual-Fingerprint Reload Mechanism**:
  - Maintain independent SHA-256 fingerprints: `raw_fingerprint` (persisted in `manifest.json`) and per-project `lab_fingerprint`.
  - Clicking **Reload External Data** checks both fingerprints:
    - If `raw_fingerprint` is unchanged, skip rebuilding `data/processed/` and base `.joblib` models; only rescan `data/lab/` and invalidate in-memory lab residual caches (completes in `< 1 s`).
    - If `raw_fingerprint` changed, rebuild literature artifacts first, then refresh lab residual models against the updated prior.
- **Parameter-Level Calibration Architecture**:
  - Introduce a `LabCalibratedPredictor` (or calibrated mode wrapper around `HybridMLPredictor`) supporting `calibration_mode`: `"literature_only" | "delta_learning" | "lab_only_gp"`.
  - Calibration operates strictly on the 4-tuple $(E_{\text{corr}}, \log_{10} i_{\text{corr}}, \beta_a, \beta_c)$ **prior to** `PolarizationParameters`, `solve_galvanic_couple`, `calculate_faraday_corrosion_rate`, and `simulate_time_resolved_kinetics`, enforcing the same physical bounds clipping (`E_corr` in `[-2.5, 2.5]` V vs. SHE, `log10(i_corr)` in `[-6, 6]`, Tafel slopes in `[20, 500]` mV/dec) and ADR-0003 log-space Monte Carlo propagation.
  - **Delta-Learning (`delta_learning`)**:
    - For each target $t$, compute residuals $\Delta y_i = y_{\text{lab},i} - \hat{y}_{\text{prior}}(x_i)$ on valid (and non-imputed for $\beta_a, \beta_c$) lab rows.
    - When $n_{\text{lab}, t} \ge 4$: fit a Zero-Mean Gaussian Process (`normalize_y=False`, `ConstantKernel * Matern(nu=2.5) + WhiteKernel`) on standardized features so that $\Delta \hat{y}(x) \to 0$ smoothly as query distance from lab points grows.
    - When $1 \le n_{\text{lab}, t} < 4$: fall back to a `quality_score`-weighted mean residual offset (constant bias) with empirical/floor uncertainty, preventing GP overfitting.
    - Combine mean and uncertainty: $\hat{y}_{\text{cal}} = \hat{y}_{\text{prior}} + \Delta \hat{y}$, $\sigma_{\text{total}} = \sqrt{\sigma_{\text{prior}}^2 + \sigma_{\Delta}^2}$.
    - Return structured `ResidualBreakdown` metadata (prior mean/std, delta mean/std, final mean/std, method used: `gp_residual` / `constant_bias` / `prior_only`, and lab sample count per target).
  - **Lab-Only Local GP (`lab_only_gp`)**:
    - Fit a standard GP directly on the selected lab project's harmonized records (requiring $n \ge 3$ valid samples per target, otherwise raising a clear actionable warning/fallback).
- **Mode-Aware Applicability Domain**:
  - `literature_only`: existing literature AD behavior.
  - `delta_learning`: evaluate both literature AD and lab local proximity (`lab_knn_distance`, `lab_in_bounding_box`, `n_lab_samples`); label queries inside/near lab coverage as `Lab-Anchored` and queries far from lab points as reverting to the literature prior.
  - `lab_only_gp`: fit and evaluate `ApplicabilityDomainChecker` strictly on the selected lab project's records only.
- **UI & Controller Integration**:
  - Extend `SimulationRequest` with solvent fields (`solvent_regime`, `organic_solvent_type`, `water_content_wt_pct`) and lab calibration fields (`calibration_mode`, `lab_project`).
  - Add sidebar **Solvent System** controls, sidebar **Lab Calibration** controls, dual status indicators under **External Data & Model Cache**, a non-aqueous physics-limitation banner, and an expandable **Residual Breakdown** diagnostic table in Single Metal and Galvanic tabs.

## Testing Decisions

- **Testing Philosophy**: Test observable external behavior through the highest stable seams—database harmonization contracts, predictor prediction/uncertainty/AD contracts, controller end-to-end simulation requests, and report serialization—rather than asserting internal private helper structures or scikit-learn internals.
- **Primary Test Seams**:
  1. **Database harmonization seam** (`HarmonizedDatabase.from_raw_tables` / `from_directory`): Verify `aqueous` / `semi_aqueous` / `non_aqueous` boundary thresholds (`90%`, `1%`), `solvent_subclass` mapping, 3-tier `water_content_source` inference, `0.85` quality discount, and full backward compatibility on seed/existing tables.
  2. **ML & Calibration seam** (`HybridMLPredictor`, `LabCalibratedPredictor`, `ApplicabilityDomainChecker`, `propagate_uncertainty_monte_carlo`): Verify leak-free solvent descriptor extraction, water-weighted Pourbaix decay, Zero-Mean GP residual convergence near lab points and smooth reversion to prior far from lab points, `n < 4` constant-bias fallback, $\sigma_{\text{total}}$ inflation, physical bounds clipping, and strict lab-only AD isolation.
  3. **Path & Controller seam** (`PlatformController`, external `data/lab/` discovery, dual-fingerprint reload): Verify sub-second lab-only reload skips `data/raw/` retraining, mode switching (`literature_only` / `delta_learning` / `lab_only_gp`) flows cleanly into Wagner-Traud, Faraday, galvanic `brentq`, kinetics, and report export, and non-aqueous queries emit physics limitation warnings.
- **Prior Art in Codebase**: Follow the existing `unittest` suites covering database harmonization, ML predictor cross-validation/uncertainty, galvanic/kinetics solvers, extensibility/artifacts, and Streamlit platform controller tests.

## Out of Scope

- Rewriting non-aqueous electrochemistry physics (non-aqueous reference electrode conversions such as $\text{Fc}/\text{Fc}^+$ or $\text{Ag}/\text{Ag}^+$, high-impedance IR-drop compensation, or non-aqueous Pourbaix diagram generation).
- Downstream residual correction on final etch depth or weight-loss-only datasets (when Tafel polarization parameters are unavailable); this may be evaluated in a future spec.
- Automated literature scraping/NLP extraction of solvent concentrations from PDF papers.
- Bayesian Optimization / automated next-experiment suggestion loops beyond surfacing Applicability Domain warnings and uncertainty bands.

## Further Notes

- Respects all three existing Architecture Decision Records: four-parameter Wagner-Traud seam (ADR-0001), time-resolved kinetics anchored to $\text{CR}_0$ (ADR-0002), and $\log_{10}(i_{\text{corr}})$ prediction with physics-clipped Monte Carlo propagation (ADR-0003).
- All new features must remain compatible with the PyInstaller `-D` Windows `.exe` packaging layout where `data/raw/`, `data/processed/`, and `data/lab/` reside externally beside the executable.
