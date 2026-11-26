# 04: External `data/lab/` project directory and dual-fingerprint Reload External Data

**What to build:** Support an external `data/lab/<project>/` directory alongside `data/raw/` and `data/processed/` for small-sample laboratory CSVs (compatible with both source execution and frozen Windows `.exe` deployments), with independent SHA-256 fingerprinting so clicking **Reload External Data** refreshes lab projects in under a second when `data/raw/` is unchanged and displays separate sync statuses in the sidebar.

**Blocked by:** 01 (Solvent regime classification, water-content inference, and harmonization)

**Status:** ready-for-agent

- [ ] External data path resolution discovers `data/lab/` alongside `data/raw/` and `data/processed/` in both development and PyInstaller frozen modes, enumerating named projects from subdirectories (`data/lab/<project>/*.csv`) without merging them into `data/raw/`.
- [ ] Lab project CSVs are harmonized through the same unit, reference-electrode, solvent-regime, and physical-bounds validation pipeline as literature records, producing a standalone `HarmonizedDatabase` per project.
- [ ] Independent fingerprints are computed for `data/raw/` and each `data/lab/<project>/`; clicking **Reload External Data** skips rebuilding `data/processed/` base models when `data/raw/` is unchanged and refreshes lab datasets in `< 1 s`.
- [ ] When `data/raw/` does change, **Reload External Data** rebuilds the literature artifacts and automatically invalidates cached lab calibration models so they re-anchor to the new prior.
- [ ] Sidebar **External Data & Model Cache** clearly displays separate status lines for the literature prior and the active lab project (status, record count, project name), verified by automated tests.
