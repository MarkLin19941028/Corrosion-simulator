# 04: External `data/lab/` project directory and dual-fingerprint Reload External Data

**What to build:** Support an external `data/lab/<project>/` directory alongside `data/raw/` and `data/processed/` for small-sample laboratory CSVs (compatible with both source execution and frozen Windows `.exe` deployments), with independent SHA-256 fingerprinting so clicking **Reload External Data** refreshes lab projects in under a second when `data/raw/` is unchanged and displays separate sync statuses in the sidebar.

**Blocked by:** 01 (Solvent regime classification, water-content inference, and harmonization)

**Status:** ready-for-agent

- [x] External data path resolution discovers `data/lab/` alongside `data/raw/` and `data/processed/` in both development and PyInstaller frozen modes, enumerating named projects from subdirectories (`data/lab/<project>/*.csv`) without merging them into `data/raw/`.
- [x] Lab project CSVs are harmonized through the same unit, reference-electrode, solvent-regime, and physical-bounds validation pipeline as literature records, producing a standalone `HarmonizedDatabase` per project.
- [x] Independent fingerprints are computed for `data/raw/` and each `data/lab/<project>/`; clicking **Reload External Data** skips rebuilding `data/processed/` base models when `data/raw/` is unchanged and refreshes lab datasets in `< 1 s`.
- [x] When `data/raw/` does change, **Reload External Data** rebuilds the literature artifacts and automatically invalidates cached lab calibration models so they re-anchor to the new prior.
- [x] Sidebar **External Data & Model Cache** clearly displays separate status lines for the literature prior and the active lab project (status, record count, project name), verified by automated tests.

## Comments

Implemented and verified:
- `corrosion_simulator/paths.py`: added `resolve_lab_dir`, `resolve_data_dirs`, `discover_lab_projects`, and `compute_csv_fingerprint`, and excluded `lab/` alongside `processed/` in `iter_raw_csv_files`.
- `corrosion_simulator/database.py`: extracted `HarmonizedDatabase.from_csv_files(...)` with `fallback_to_seed=False` support for standalone lab datasets and unified `compute_raw_directory_fingerprint`.
- `corrosion_simulator/lab_data.py`: added `LabProjectDataset`, `compute_lab_project_fingerprint`, `compute_lab_directory_fingerprint`, `load_lab_project`, and `discover_and_load_lab_projects`.
- `corrosion_simulator/platform_controller.py`: added `LabReloadStatus`, `ExternalDataReloadSummary`, dual-fingerprint `reload_external_data(...)`, `get_lab_project(...)`, `_lab_calibration_cache` invalidation on raw rebuilds, and `"lab_projects"` in `available_options()`.
- `app.py`: parameterized `_load_cached_controller` by resolved `(raw_dir, processed_dir, lab_dir)` paths and updated sidebar **External Data & Model Cache** to display separate Literature Prior and Lab Data statuses plus fast reload via `controller.reload_external_data(...)`.
- `tests/test_extensibility_and_artifacts.py` and `tests/test_streamlit_platform.py`: added `TestLabDirectoryAndDualFingerprintReload` and `test_streamlit_sidebar_displays_literature_and_lab_sync_statuses_and_reloads`; full suite `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest discover -s tests -v` passed (86/86 tests OK).
