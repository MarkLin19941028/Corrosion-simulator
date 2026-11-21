# Corrosion Simulator Specification

Status: ready-for-agent

## Problem Statement

Semiconductor wet cleaning and post-CMP etching processes expose thin metallic interconnect, liner, barrier, and gate films (such as Cu, Co, W, Ru, Ti, Ta, Al, and Mo) to complex chemical formulations containing oxidizers, organic or inorganic acids, complexing agents, and corrosion inhibitors. Even sub-nanometer to few-nanometer metal loss during seconds-to-minutes of exposure can degrade device performance, while electrical contact between dissimilar metals on a patterned wafer creates bimetallic galvanic couples that severely accelerate dissolution of the more active film.

Process engineers currently face three connected problems:
1. **Fragmented and inconsistent literature data**: Experimental electrochemical data across metals and solutions are scattered across tables with inconsistent reference electrodes, current density units, Tafel slope conventions, and missing values.
2. **Inability to evaluate untested formulations with physical consistency and uncertainty bounds**: Pure lookup tables fail when exploring novel formulations (e.g., tweaking oxidizer/acid concentrations, pH, or temperature), whereas naive black-box machine learning can violate physical laws (producing negative dissolution rates, non-monotonic thickness loss over time, or silent extrapolation outside the training domain).
3. **Lack of integrated recipe decision support**: Engineers lack an interactive platform to compare candidate wet cleaning formulations side by side across single-metal loss, bimetallic galvanic acceleration, time-to-threshold limits, confidence warnings, and shareable technical reports.

## Solution

Build a three-phase, physics-informed and machine-learning-assisted **Corrosion Simulator** for semiconductor wet cleaning evaluation:

1. **Phase 1 — Electrochemical Parameter Database & Physics-Based Engine**:
   Consolidate and normalize per-metal raw datasets for the eight core semiconductor metals (Cu, Co, W, Ru, Ti, Ta, Al, Mo) into a unified, unit-standardized database ($\text{V vs. SHE}$, $\mu\text{A/cm}^2$, $\text{mV/dec}$) with data quality scoring, missing-slope imputation tracking, and per-solution Galvanic Series rankings. Provide deterministic physics solvers for generating four-parameter activation-controlled **Polarization Curves**, converting **Corrosion Current Density ($i_{\text{corr}}$)** to **Initial Corrosion Rate ($\text{CR}_0$)** and linear **Corrosion Depth** via Faraday's law, and screening bimetallic pairs via **Galvanic Driving Force Risk** ($|\Delta E_{\text{corr}}|$).

2. **Phase 2 — Hybrid Physics-ML Predictor, Galvanic Mixed-Potential Solver & Time-Resolved Kinetics**:
   Train machine-learning models on material descriptors, solution descriptors, and engineered physicochemical interactions to predict $(E_{\text{corr}}, \log_{10} i_{\text{corr}}, \beta_a, \beta_c)$ with uncertainty estimates (tree variance, Gaussian Process posterior variance, or bootstrap ensembles). Propagate parameter uncertainty into downstream physical predictions via physics-clipped Monte Carlo sampling. Solve area-weighted bimetallic **Mixed Potential ($E_{\text{galv}}$)**, **Galvanic Current Density ($i_{\text{galv}}$)**, and **Acceleration Factor** via root-finding. Model time-dependent **Corrosion Depth** $d(t)$ using Linear, Power-law, and Passivation kinetics anchored to $\text{CR}_0$, compute **Time to Threshold** against a configurable **Critical Thickness Threshold ($d_{\text{crit}}$)**, evaluate continuous and categorical **Thickness Loss Risk**, and enforce **Applicability Domain** checks with **Confidence Tier** ratings.

3. **Phase 3 — Interactive Decision Support Platform & Automated Reporting**:
   Deliver an interactive web dashboard enabling single-metal analysis, bimetallic galvanic coupling analysis, multi-recipe comparison (2–5 **Cleaning Recipes** with bar/radar visualizations and lowest-risk ranking), automated applicability and data-sufficiency guidance suggesting targeted follow-up experiments, and multi-format report export (HTML, PDF, CSV, JSON, PNG, SVG).

## User Stories

1. As a data engineer, I want to merge per-metal electrochemical datasets across Cu, Co, W, Ru, Ti, Ta, Al, and Mo into a single unified electrochemical database, so that all downstream tools query one consistent source of truth.
2. As a data engineer, I want to deduplicate and merge per-metal material property tables and solution property tables (keyed by unique solution identifiers), so that redundant records across metal directories are eliminated.
3. As an electrochemist, I want all reference electrode potentials (such as Ag/AgCl, SCE, or MSE) automatically converted to $\text{V vs. SHE}$ for every **Corrosion Potential ($E_{\text{corr}}$)** record, so that potentials are directly comparable across studies.
4. As an electrochemist, I want all **Corrosion Current Density ($i_{\text{corr}}$)** values converted to $\mu\text{A/cm}^2$ and all anodic and cathodic **Tafel Slopes ($\beta_a, \beta_c$)** normalized to strictly positive $\text{mV/dec}$, so that unit mismatches never corrupt physics calculations.
5. As a data curator, I want records missing $E_{\text{corr}}$ or $i_{\text{corr}}$ to be filtered out, and records missing $\beta_a$ or $\beta_c$ to be imputed using same-metal/same-oxidizer medians (or physical fallbacks of $60$ and $120\text{ mV/dec}$) while flagging them as imputed and lowering their quality score, so that data provenance and reliability remain transparent.
6. As a process engineer, I want a generated Galvanic Series table ranking all metals by their $E_{\text{corr}}$ within each **Solution**, so that I can immediately inspect relative nobility across chemical environments.
7. As a researcher, I want a comprehensive data dictionary documenting every table attribute, physical unit, and quality/imputation flag, so that I can interpret the processed dataset accurately.
8. As a process engineer, I want to query the database for an exact or nearest-neighbor **Exposure Condition** (matching the same **Metal** and oxidizer category, then minimizing normalized distance across pH, oxidizer concentration, and temperature), so that I can retrieve baseline electrochemical parameters without machine learning.
9. As a process engineer, I want to manually input custom $(E_{\text{corr}}, i_{\text{corr}}, \beta_a, \beta_c)$ parameters, so that I can simulate hypothetical or newly measured lab curves directly.
10. As an electrochemist, I want to generate anodic, cathodic, and net **Polarization Curves** ($E$ vs. $\log_{10}|i|$) from $(E_{\text{corr}}, i_{\text{corr}}, \beta_a, \beta_c)$ using the four-parameter Wagner-Traud equation, so that I can visualize Tafel plots for any **Exposure Condition**.
11. As a process engineer, I want to convert $i_{\text{corr}}$ into **Corrosion Rate ($\text{CR}$)** in both $\text{nm/min}$ and $\text{mm/yr}$ using Faraday's law and built-in **Dissolution Valence ($z$)** defaults (with optional per-call valence overrides), so that current densities translate directly into semiconductor film thickness loss rates.
12. As a process engineer, I want to calculate linear **Corrosion Depth** over a specified immersion duration from the Faraday **Corrosion Rate**, so that I can estimate nanometer film loss in a wet cleaning step.
13. As a process engineer, I want to screen any two metals under the same **Solution** and temperature using their open-circuit potential difference $|\Delta E_{\text{corr}}|$ to classify **Galvanic Driving Force Risk** (`Low` $< 0.1\text{ V}$, `Medium` $0.1\text{--}0.3\text{ V}$, `High` $> 0.3\text{ V}$) and identify the **Active Metal** and **Noble Metal**, so that I can quickly flag hazardous bimetallic pairings.
14. As a machine-learning engineer, I want to extract material descriptors (standard potential, density, electronegativity, work function, passivation tendency), solution descriptors (pH, temperature, encoded oxidizer/acid/inhibitor types and concentrations), and engineered interaction features ($\text{pH} \times T$, standard potential minus ORP, log-concentration, and Pourbaix stability indicators), so that predictive models capture physicochemical driving forces.
15. As a machine-learning engineer, I want to train and cross-validate (via leave-one-out or metal-stratified 5-fold CV) multiple model families (Random Forest baseline, gradient boosted trees, and Gaussian Processes) for predicting $E_{\text{corr}}$, $\log_{10}(i_{\text{corr}})$, $\beta_a$, and $\beta_c$ evaluated by MAE, RMSE, $R^2$, and MAPE, so that we select accurate and well-calibrated predictors.
16. As a process engineer, I want every ML-predicted electrochemical parameter to include an uncertainty estimate ($\pm 1\sigma$) derived from tree variance, Gaussian Process posterior variance, or bootstrap ensembles, so that I know the precision of each prediction.
17. As a process engineer, I want parameter uncertainties propagated through the physics solvers via log-space and physics-clipped Monte Carlo sampling ($N \ge 500$ draws with $\log_{10} i_{\text{corr}}$ sampled in log-space and $\beta_a, \beta_c$ clipped to $[20, 500]\text{ mV/dec}$), so that **Polarization Curves**, **Corrosion Rate**, **Corrosion Depth**, and galvanic coupling metrics carry physically valid uncertainty bands.
18. As a process engineer, I want an **Applicability Domain** checker that verifies whether a query's categorical and numerical features fall within the per-metal training bounding box and evaluates local $k$-NN training density to assign a **Confidence Tier** (`High`, `Medium`, `Low`), so that I am warned whenever a prediction relies on extrapolation.
19. As a process engineer, I want to solve for the bimetallic **Mixed Potential ($E_{\text{galv}}$)** of an **Active Metal** and **Noble Metal** pair using numerical root-finding (`brentq`) under area-weighted current conservation ($A_{\text{active}} i_{\text{net, active}}(E_{\text{galv}}) + A_{\text{noble}} i_{\text{net, noble}}(E_{\text{galv}}) = 0$) with a configurable cathode-to-anode area ratio, so that I can quantify real-world galvanic coupling on patterned wafers.
20. As a process engineer, I want to compute the coupled **Galvanic Current Density ($i_{\text{galv}}$)**, the **Acceleration Factor** ($i_{\text{galv}} / i_{\text{corr, active}} \ge 1.0$), and the resulting accelerated **Initial Corrosion Rate ($\text{CR}_0$)** on the **Active Metal**, so that I know how much faster the vulnerable film dissolves when coupled.
21. As a process engineer, I want to simulate time-resolved **Corrosion Depth** $d(t)$ using Linear ($d(t) = \text{CR}_0 t$), Passivation ($d(t) = d_{\max}(1 - \exp(-\text{CR}_0 t / d_{\max}))$ using **Passivation Limiting Depth ($d_{\max}$)**), and Power-law ($d(t) = (\text{CR}_0 t_{\text{ref}})(t / t_{\text{ref}})^n$) kinetics automatically selected by Pourbaix stability regime or manually toggled, so that surface passivation and diffusion slowing are reflected over time.
22. As a process engineer, I want to calculate the **Time to Threshold** (with uncertainty bounds) required for a metal's **Corrosion Depth** to reach a target **Critical Thickness Threshold ($d_{\text{crit}}$)** (default $1.0\text{ nm}$), so that I can set safe maximum immersion times for a cleaning process.
23. As a process engineer, I want a time-resolved **Thickness Loss Risk** evaluation providing both a continuous score $R(t) = 1 - \exp(-\ln(2)\cdot d(t)/d_{\text{crit}}) \in [0, 1]$ and a discrete risk tier (`Low` $< 0.8 d_{\text{crit}}$, `Medium` $0.8\text{--}2.0 d_{\text{crit}}$, `High` $2.0\text{--}4.0 d_{\text{crit}}$, `Critical` $\ge 4.0 d_{\text{crit}}$), so that I can track how risk escalates with immersion time.
24. As a process engineer, I want all outputs to strictly satisfy physical constraints ($\text{CR} \ge 0$, $d(t) \ge 0$ and monotonically non-decreasing with $t$, $R(t) \in [0, 1]$, $E_{\text{corr}}$ within thermodynamically plausible Pourbaix bounds, and $\beta_a, \beta_c \in [20, 500]\text{ mV/dec}$), so that no simulation ever produces unphysical artifacts.
25. As a process engineer, I want an interactive dashboard sidebar where I can select a primary **Metal**, optional secondary **Metal** (and area ratio), **Solution** components and concentrations, pH, temperature, **Immersion Time**, and **Critical Thickness Threshold ($d_{\text{crit}}$)**, so that I can interactively explore process windows.
26. As a process engineer, I want a Single Metal Analysis view displaying interactive **Polarization Curves** with uncertainty bands, predicted or looked-up electrochemical parameters, **Corrosion Rate**, time-resolved **Corrosion Depth** and **Thickness Loss Risk** curves, **Confidence Tier**, and domain warnings, so that I can thoroughly assess a single film's behavior.
27. As a process engineer, I want a Galvanic Analysis view displaying overlaid bimetallic **Polarization Curves** with the **Mixed Potential ($E_{\text{galv}}$)** operating line, **Galvanic Current Density ($i_{\text{galv}}$)**, **Acceleration Factor**, **Time to Threshold**, and comparative uncoupled vs. coupled depth/risk curves, so that I can evaluate bimetallic compatibility.
28. As a process engineer, I want a Recipe Comparison view where I can define 2 to 5 candidate **Cleaning Recipes**, compare their **Corrosion Rate**, **Corrosion Depth**, **Acceleration Factor**, and **Thickness Loss Risk** via summary tables, bar charts, and radar plots, and receive an automatic recommendation for the lowest-risk **Cleaning Recipe**, so that I can make data-driven formulation decisions.
29. As a process engineer, I want actionable data-sufficiency warnings and experimental recommendations whenever a query receives a `Low` or `Medium` **Confidence Tier**, telling me which metal/chemical/pH/temperature regime lacks experimental coverage, so that I know which lab polarization experiments to run next.
30. As a process engineer, I want to export complete analysis and comparison reports in HTML, PDF, CSV, JSON, PNG, and SVG formats containing input summaries, parameter provenance (Database vs. ML), figures, metrics, uncertainties, and warnings, so that I can archive and share results in design reviews.
31. As a researcher or developer, I want five step-by-step demonstration notebooks covering data exploration/cleaning, Phase 1 Tafel and corrosion calculations, Phase 2 ML training and feature importance, bimetallic galvanic mixed-potential analysis, and time-resolved risk modeling, so that every stage of the methodology is reproducible and inspectable.

## Implementation Decisions

- **Adaptive Schema & Data Harmonization Module**:
  - Reads raw per-metal tables for the eight supported metals (`Cu`, `Co`, `W`, `Ru`, `Ti`, `Ta`, `Al`, `Mo`), inspecting actual CSV columns present in the raw files rather than assuming rigid non-existent columns.
  - Deduplicates material properties by **Metal** identifier and solution properties by `solution_id`.
  - Normalizes all potentials to $\text{V vs. SHE}$ via a reference electrode conversion interface supporting standard reference electrodes (SHE, Ag/AgCl, SCE, MSE, etc.).
  - Normalizes current densities to $\mu\text{A/cm}^2$ and Tafel slopes to positive $\text{mV/dec}$.
  - Drops records missing $E_{\text{corr}}$ or $i_{\text{corr}}$; imputes missing $\beta_a$ or $\beta_c$ using same-metal, same-oxidizer median values (falling back to $\beta_a = 60\text{ mV/dec}, \beta_c = 120\text{ mV/dec}$), sets an `is_imputed` boolean flag, and decrements the record's `quality_score`.
  - Produces unified processed tables (`electrochemical_database`, `material_properties`, `solution_properties`, `galvanic_series`) and a human-readable data dictionary.
  - Supports Phase 1 nearest-neighbor parameter retrieval by filtering on exact **Metal** and oxidizer type, then computing Min-Max normalized Euclidean distance over numerical exposure conditions (`pH`, oxidizer concentration, temperature) and returning the closest record along with a condition-mismatch summary.

- **Four-Parameter Wagner-Traud Polarization & Faraday Module (per ADR-0001)**:
  - Implements activation-controlled base-10 anodic, cathodic, and net current density functions parameterized by $(E_{\text{corr}}, i_{\text{corr}}, \beta_a, \beta_c)$:
    $$i_{\text{anodic}}(E) = i_{\text{corr}} \cdot 10^{\frac{E - E_{\text{corr}}}{\beta_a}}, \quad i_{\text{cathodic}}(E) = i_{\text{corr}} \cdot 10^{-\frac{E - E_{\text{corr}}}{\beta_c}}, \quad i_{\text{net}}(E) = i_{\text{anodic}}(E) - i_{\text{cathodic}}(E)$$
    where $\beta_a, \beta_c$ are converted from $\text{mV/dec}$ to $\text{V/dec}$ internally.
  - Implements Faraday's law conversion from current density ($\mu\text{A/cm}^2$) to **Corrosion Rate** in both $\text{nm/min}$ and $\text{mm/yr}$. Uses a built-in dictionary of standard **Dissolution Valence ($z$)** values (`Cu: 2, Co: 2, W: 6, Ru: 4, Ti: 4, Ta: 5, Al: 3, Mo: 6`) and standard atomic weights/densities as fallbacks whenever those columns are absent from `material_properties`, while allowing explicit per-call or per-record valence overrides.
  - Implements Phase 1 **Galvanic Driving Force Risk** screening from $|\Delta E_{\text{corr}}|$ with thresholds at $0.1\text{ V}$ (`Low` / `Medium`) and $0.3\text{ V}$ (`Medium` / `High`), identifying the metal with the lower $E_{\text{corr}}$ as the **Active Metal**.

- **Galvanic Mixed-Potential Solver Module (per ADR-0001)**:
  - Identifies the **Active Metal** ($\min E_{\text{corr}}$) and **Noble Metal** ($\max E_{\text{corr}}$).
  - Solves for the bimetallic **Mixed Potential ($E_{\text{galv}}$)** within the bracket $[E_{\text{corr, active}}, E_{\text{corr, noble}}]$ using Brent's root-finding method (`scipy.optimize.brentq`) on the area-weighted total current conservation equation:
    $$i_{\text{net, active}}(E_{\text{galv}}) + \left(\frac{A_{\text{noble}}}{A_{\text{active}}}\right) i_{\text{net, noble}}(E_{\text{galv}}) = 0$$
    where the cathode-to-anode area ratio $A_{\text{noble}} / A_{\text{active}}$ defaults to $1.0$ and is user-configurable.
  - Evaluates **Galvanic Current Density ($i_{\text{galv}}$)** as $i_{\text{anodic, active}}(E_{\text{galv}})$ and **Acceleration Factor** as $i_{\text{galv}} / i_{\text{corr, active}}$ (guaranteed $\ge 1.0$).

- **Time-Resolved Kinetics & Thickness Loss Risk Module (per ADR-0002)**:
  - Anchors all three time-resolved **Corrosion Depth** models at $t = 0$ to the instantaneous Faraday **Initial Corrosion Rate ($\text{CR}_0$)** (using $\text{CR}_0$ from $i_{\text{corr}}$ for single metals or from $i_{\text{galv}}$ for the coupled **Active Metal**):
    - **Linear**: $d(t) = \text{CR}_0 \cdot t$
    - **Passivation**: $d(t) = d_{\max} \left(1 - \exp\left(-\frac{\text{CR}_0}{d_{\max}} t\right)\right)$, parameterized by **Passivation Limiting Depth ($d_{\max}$)**
    - **Power-law**: $d(t) = (\text{CR}_0 \cdot t_{\text{ref}}) \left(\frac{t}{t_{\text{ref}}}\right)^n$ with $t_{\text{ref}} = 60\text{ s}$ and $n \in [0.3, 1.0]$ (default $0.6$)
  - Automatically selects the default kinetic regime from the Pourbaix stability indicator (`Active` $\to$ Linear, `Passivation` $\to$ Passivation, inhibited/transitional $\to$ Power-law) while supporting explicit user selection or multi-model overlay.
  - Solves analytically or numerically for **Time to Threshold** where $d(t) = d_{\text{crit}}$ (returning $\infty$ when $d_{\max} \le d_{\text{crit}}$ in the Passivation regime).
  - Computes continuous **Thickness Loss Risk** score $R(t) = 1 - \exp(-\ln(2) \cdot d(t) / d_{\text{crit}}) \in [0, 1]$ and discrete risk tiers (`Low`: $d < 0.8 d_{\text{crit}}$, `Medium`: $0.8 d_{\text{crit}} \le d < 2.0 d_{\text{crit}}$, `High`: $2.0 d_{\text{crit}} \le d < 4.0 d_{\text{crit}}$, `Critical`: $d \ge 4.0 d_{\text{crit}}$).

- **Hybrid Physics-ML, Uncertainty & Applicability Domain Module (per ADR-0003)**:
  - Constructs a unified feature pipeline combining material descriptors, encoded solution descriptors, and engineered physicochemical interaction terms ($\text{pH} \times T$, standard potential minus ORP, log-concentrations, Pourbaix stability indicator).
  - Trains four target regressors for $E_{\text{corr}}$, $\log_{10}(i_{\text{corr}})$, $\beta_a$, and $\beta_c$ supporting Random Forest, gradient boosted decision trees (XGBoost/LightGBM), and Gaussian Process regressors, evaluated under LOOCV or metal-stratified 5-fold cross-validation and persisted as serialized model bundles.
  - Estimates predictive mean and standard deviation $(\mu, \sigma)$ per target using tree variance across ensemble members, Gaussian Process posterior variance, or bootstrap ensembles.
  - Propagates uncertainty via Monte Carlo sampling ($N = 500$ draws by default): samples $\log_{10}(i_{\text{corr}}) \sim \mathcal{N}(\mu_{\log i}, \sigma_{\log i}^2)$ and exponentiates to $i_{\text{corr}} = 10^x$ (strictly positive), clips sampled $\beta_a, \beta_c$ to $[20, 500]\text{ mV/dec}$ and $E_{\text{corr}}$ to thermodynamically reasonable bounds, and computes mean, standard deviation, and percentile bands for all downstream quantities ($\text{Polarization Curve}$, $\text{CR}_0$, $E_{\text{galv}}$, $i_{\text{galv}}$, $\text{Acceleration Factor}$, $d(t)$, $R(t)$, and $\text{Time to Threshold}$).
  - Evaluates **Applicability Domain** per metal via a two-tier check:
    1. **Bounding Box Check**: checks whether categorical chemicals exist in that metal's training subset and whether numeric features fall within observed `[min, max]` ranges.
    2. **Local Density Check**: computes normalized $k$-NN distance ($k = 3$) to the closest training examples for that metal.
    Assigns `High`, `Medium`, or `Low` **Confidence Tier** and generates human-readable experimental gap recommendations when confidence is `Medium` or `Low`.

- **Interactive Platform & Multi-Format Reporting Module**:
  - Exposes a cohesive high-level simulation and decision service consumed by both the demonstration notebooks and the interactive Streamlit dashboard.
  - Organizes the Streamlit application into a persistent parameter sidebar and dedicated views for **Single Metal Analysis**, **Galvanic Analysis**, **Recipe Comparison** (2–5 **Cleaning Recipes** with bar chart, radar chart, comparison table, and lowest-risk recommendation), and **Report Export**.
  - Uses interactive Plotly figures for Tafel overlays, uncertainty bands, time-resolved depth/risk trajectories, and multi-recipe comparisons (in addition to static Matplotlib helpers for Phase 1 notebooks and static image exports).
  - Generates self-contained HTML reports and exports structured CSV, JSON, PNG, SVG, and PDF summaries containing full input metadata, parameter provenance (`Database` vs. `ML`), physics/risk metrics, uncertainty intervals, and applicability warnings.

## Testing Decisions

- **Philosophy — Behavioral Testing at High-Level Seams**:
  Tests must verify observable physical, statistical, and contract behaviors at public module boundaries rather than internal implementation details or private helper structures.

- **Seam 1 — Database Harmonization & Lookup Seam**:
  - Verify that raw multi-metal tables are merged and deduplicated without data loss for valid records.
  - Verify reference electrode conversions to $\text{V vs. SHE}$ and unit conversions to $\mu\text{A/cm}^2$ and positive $\text{mV/dec}$ against analytical hand-calculated values.
  - Verify that records missing $E_{\text{corr}}$ or $i_{\text{corr}}$ are rejected, while records missing $\beta_a$ or $\beta_c$ receive imputed values, `is_imputed = True`, and a reduced `quality_score`.
  - Verify that `galvanic_series` orders metals monotonically by $E_{\text{corr}}$ within each solution, and that nearest-neighbor lookup returns the exact record on an exact match and reports condition deltas on an approximate match.

- **Seam 2 — Core Physics, ML Uncertainty & Decision Engine Seam**:
  - **Polarization & Faraday invariants**: Verify that at $E = E_{\text{corr}}$, $i_{\text{anodic}} = i_{\text{cathodic}} = i_{\text{corr}}$ and $i_{\text{net}}(E_{\text{corr}}) = 0$; verify that shifting $E$ by $+\beta_a$ increases $i_{\text{anodic}}$ by a factor of $10$; verify Faraday's law conversion against reference hand calculations (e.g., Cu at $i_{\text{corr}} = 2.3\text{ }\mu\text{A/cm}^2, z = 2 \to \text{CR} \approx 0.45\text{ nm/min}$).
  - **Galvanic coupling invariants**: Verify that for any two distinct metals, $E_{\text{corr, active}} < E_{\text{galv}} < E_{\text{corr, noble}}$; verify area-weighted current conservation at $E_{\text{galv}}$ to within numerical tolerance ($< 10^{-6}\text{ }\mu\text{A/cm}^2$); verify that **Acceleration Factor** $> 1.0$ and increases monotonically as the cathode-to-anode area ratio $A_{\text{noble}} / A_{\text{active}}$ increases.
  - **Time-resolved kinetics & risk invariants**: Verify that Linear, Passivation, and Power-law models all satisfy $d(0) = 0$, $d(t)$ is monotonically non-decreasing for $t \ge 0$, and Linear vs. Passivation share the same initial slope $\text{CR}_0$ as $t \to 0$; verify that **Thickness Loss Risk** $R(t) \in [0, 1]$, $R(d_{\text{crit}}) = 0.5$, and discrete tiers match the $0.8\times, 2.0\times, 4.0\times d_{\text{crit}}$ boundaries.
  - **ML, Uncertainty & Applicability Domain invariants**: Verify that all Monte Carlo propagated predictions satisfy physical constraints ($\text{CR} \ge 0$, $\beta_a, \beta_c \in [20, 500]\text{ mV/dec}$, $d(t) \ge 0$); verify that training data inliers receive a `High` **Confidence Tier** while out-of-range queries trigger an out-of-domain warning, a `Low` **Confidence Tier**, and non-empty experimental guidance.
  - **Recipe comparison & report generation**: Verify that comparing multiple **Cleaning Recipes** ranks the lowest-risk formulation first and exports valid HTML, JSON, and CSV artifacts containing all required provenance and warning fields.

- **Prior Art**:
  This is a greenfield repository; the test suite will be established using `pytest` with deterministic synthetic fixtures and regression checks against the specification's benchmark examples (e.g., Cu and Co in $\text{H}_2\text{O}_2$ + citric acid).

## Out of Scope

- Spatial 2D/3D finite-element or finite-difference partial differential equation (PDE) modeling of potential, current, or concentration fields across wafer topography.
- Microstructural localized corrosion simulation (such as stochastic pitting nucleation, grain-boundary diffusion, or crevice geometry transport).
- Molecular dynamics (MD) or density functional theory (DFT) quantum-chemical calculations.
- Multi-user authentication, cloud database hosting, or live manufacturing execution system (MES) hardware integration.

## Further Notes

- The system must gracefully handle both the user's incoming raw CSV files (once placed into the raw data directory) and a built-in fallback/seed fixture so that automated tests and demonstration notebooks remain runnable out of the box.
- Domain vocabulary defined in `GLOSSARY.md` and architectural constraints recorded in `docs/adr/0001` through `docs/adr/0003` govern all module naming and mathematical formulations.
