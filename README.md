# Corrosion Simulator

A physics-informed and machine-learning-assisted electrochemical corrosion simulator for evaluating metal loss, bimetallic galvanic coupling acceleration, time-resolved **Thickness Loss Risk**, and wet **Cleaning Recipe** trade-offs across semiconductor interconnect, liner, barrier, and gate metals (`Cu, Co, W, Ru, Ti, Ta, Al, Mo`).

## Key Capabilities

1. **Electrochemical Database Harmonization (`corrosion_simulator/database.py`)**:
   - Ingests and merges per-metal CSV tables from `data/raw/` (with automatic fallback to built-in seed data in `seed_data.py`), deduplicates material and solution tables, converts reference electrodes (SHE, Ag/AgCl, SCE, MSE) to $\text{V vs. SHE}$, normalizes $i_{\text{corr}}$ to $\mu\text{A/cm}^2$ and Tafel slopes to positive $\text{mV/dec}$, tracks slope imputation and data quality scores, builds per-solution Galvanic Series rankings, and supports nearest-neighbor **Exposure Condition** lookup.
2. **Wagner–Traud Polarization & Faraday Engine (`corrosion_simulator/physics.py`)**:
   - Four-parameter activation-controlled **Polarization Curves**, Faraday's law **Initial Corrosion Rate ($\text{CR}_0$)** in `nm/min` and `mm/yr`, linear **Corrosion Depth**, and **Galvanic Driving Force Risk** screening ($|\Delta E_{\text{corr}}|$).
3. **Hybrid Physics-ML Predictor & Applicability Domain (`corrosion_simulator/ml_predictor.py`)**:
   - Predicts $(E_{\text{corr}}, \log_{10} i_{\text{corr}}, \beta_a, \beta_c)$ with uncertainty using Random Forest, Gradient Boosting, or Gaussian Process models; propagates log-space, physics-clipped Monte Carlo uncertainty; and assigns **Confidence Tiers** (`High`, `Medium`, `Low`) with targeted experimental recommendations.
4. **Area-Weighted Galvanic Mixed-Potential Solver (`corrosion_simulator/galvanic.py`)**:
   - Solves area-weighted current conservation via Brent root-finding (`brentq`) to obtain **Mixed Potential ($E_{\text{galv}}$)**, **Galvanic Current Density ($i_{\text{galv}}$)**, **Acceleration Factor**, and accelerated $\text{CR}_0$ with Monte Carlo uncertainty propagation.
5. **Time-Resolved Kinetics & Risk (`corrosion_simulator/kinetics.py`)**:
   - Simulates **Corrosion Depth** $d(t)$ under Linear, Passivation (with **Passivation Limiting Depth ($d_{\max}$)**), and Power-law models anchored to $\text{CR}_0$, computes **Time to Threshold** against **Critical Thickness Threshold ($d_{\text{crit}}$)**, and evaluates continuous and discrete **Thickness Loss Risk**.
6. **Interactive Streamlit Platform & Multi-Format Reporting (`app.py`, `corrosion_simulator/platform_controller.py`, `corrosion_simulator/reporting.py`)**:
   - Unified sidebar + four interactive tabs: **Single Metal Analysis**, **Galvanic Analysis**, **Recipe Comparison** (2–5 candidate **Cleaning Recipes** with bar/radar charts and automatic lowest-risk ranking), and **Report Export** (HTML, PDF, CSV, JSON, PNG, SVG).

---

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Raw Data Placement

Place per-metal raw electrochemical, material property, and solution property CSV files under `data/raw/` (e.g. `data/raw/<metal>/*.csv`).
If `data/raw/` is empty or missing a core metal, `HarmonizedDatabase.from_directory("data/raw")` automatically supplements or falls back to the built-in seed dataset (`corrosion_simulator/seed_data.py`), so the dashboard, notebooks, and tests run out of the box.

## Launching the Interactive Streamlit App

```bash
.venv/bin/streamlit run app.py
```

## Running the Automated Test Suite

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Demonstration Notebooks

Five executable Jupyter notebooks under `notebooks/` document each stage of the methodology:

1. `notebooks/01_data_exploration_and_cleaning.ipynb` — Raw ingestion, unit/reference-electrode harmonization, imputation, and Galvanic Series generation.
2. `notebooks/02_phase1_tafel_and_corrosion_calculations.ipynb` — Wagner–Traud Tafel curves, Faraday corrosion rates, and Phase 1 galvanic driving-force screening.
3. `notebooks/03_phase2_ml_training_and_uncertainty.ipynb` — Feature engineering, cross-validation across model families, Monte Carlo uncertainty bands, and Applicability Domain checks.
4. `notebooks/04_bimetallic_galvanic_mixed_potential.ipynb` — Area-weighted galvanic mixed-potential solving, area-ratio sweeps, and overlaid Tafel plots.
5. `notebooks/05_time_resolved_kinetics_and_risk.ipynb` — Linear, Passivation, and Power-law depth trajectories, Time to Threshold, and Thickness Loss Risk curves.

To open and run the notebooks interactively (or execute headless via `nbconvert`):

```bash
# Interactive Jupyter Lab (install jupyterlab if not already present)
.venv/bin/pip install jupyterlab
.venv/bin/jupyter lab notebooks/

# Or re-execute a notebook headlessly
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_exploration_and_cleaning.ipynb
```

## Domain Glossary & Architecture Decisions

- Domain vocabulary: [GLOSSARY.md](GLOSSARY.md)
- Architecture Decision Records: `docs/adr/0001` through `docs/adr/0003`
