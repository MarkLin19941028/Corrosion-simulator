"""Workspace artifact, lab-project discovery, fingerprinting, and ML predictor cache lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Any, Literal

from corrosion_simulator.database import CORE_METALS, HarmonizedDatabase
from corrosion_simulator.lab_calibration import (
    CALIBRATION_MODES,
    CalibrationMode,
    LabCalibratedPredictor,
)
from corrosion_simulator.ml_predictor import (
    SUPPORTED_MODEL_FAMILIES,
    HybridMLPredictor,
    MLFeatureQuery,
    ModelFamily,
)
from corrosion_simulator.paths import (
    compute_csv_fingerprint,
    discover_lab_projects,
    iter_raw_csv_files,
    resolve_data_dirs,
    resolve_lab_dir,
)

ArtifactStatus = Literal["fresh", "rebuilt", "stale", "live_fallback", "live", "seed"]
LabReloadStatus = Literal["none", "fresh", "reloaded"]


@dataclass(frozen=True)
class LabProjectDataset:
    """Harmonized laboratory project loaded from ``data/lab/<project>/``."""

    project_name: str
    lab_dir: Path
    csv_files: tuple[Path, ...]
    fingerprint: dict[str, Any]
    database: HarmonizedDatabase
    n_records: int
    n_partial_records: int
    n_rejected: int


@dataclass(frozen=True)
class ExternalDataReloadSummary:
    """Structured result of dual-fingerprint Reload External Data."""

    raw_rebuilt: bool
    raw_status: ArtifactStatus
    lab_status: LabReloadStatus
    lab_projects: tuple[str, ...]
    lab_added_or_updated: tuple[str, ...]
    lab_removed: tuple[str, ...]
    lab_cache_invalidated: bool
    elapsed_s: float


def compute_lab_project_fingerprint(
    project_name: str,
    csv_files: Sequence[Path],
    base_dir: Path,
) -> dict[str, Any]:
    """Compute a deterministic SHA-256 fingerprint for one lab project's CSV files."""
    fp = compute_csv_fingerprint(csv_files, base_dir)
    return {
        "project_name": project_name,
        **fp,
    }


def compute_lab_directory_fingerprint(
    lab_dir: str | Path | None = None,
    *,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compute deterministic per-project and aggregate SHA-256 fingerprints under ``data/lab/``."""
    base = resolve_lab_dir(data_dir, lab_dir=lab_dir)
    project_files = discover_lab_projects(base)
    projects_fp: dict[str, dict[str, Any]] = {}
    combined_hasher = hashlib.sha256()

    for project_name in sorted(project_files):
        fp = compute_lab_project_fingerprint(project_name, project_files[project_name], base)
        projects_fp[project_name] = fp
        combined_hasher.update(project_name.encode("utf-8"))
        combined_hasher.update(b":")
        combined_hasher.update(str(fp["combined_sha256"]).encode("utf-8"))
        combined_hasher.update(b"\n")

    return {
        "lab_dir": str(base),
        "combined_sha256": combined_hasher.hexdigest(),
        "project_count": len(projects_fp),
        "projects": projects_fp,
    }


def load_lab_project(
    project_name: str,
    csv_files: Sequence[Path],
    *,
    lab_dir: Path,
) -> LabProjectDataset:
    """Load and harmonize a single lab project through the standard harmonization pipeline."""
    sorted_files = tuple(sorted(csv_files))
    fingerprint = compute_lab_project_fingerprint(project_name, sorted_files, lab_dir)
    project_base = lab_dir if project_name == "default" else (lab_dir / project_name)
    database = HarmonizedDatabase.from_csv_files(
        sorted_files,
        base_path=project_base,
        filter_to_core_metals=False,
        supplement_missing_core_metals=False,
        fallback_to_seed=False,
    )
    return LabProjectDataset(
        project_name=project_name,
        lab_dir=lab_dir,
        csv_files=sorted_files,
        fingerprint=fingerprint,
        database=database,
        n_records=int(len(database.electrochemical_database)),
        n_partial_records=int(len(database.partial_target_records)),
        n_rejected=int(len(database.rejected_records)),
    )


def discover_and_load_lab_projects(
    lab_dir: str | Path | None = None,
    *,
    data_dir: str | Path | None = None,
) -> dict[str, LabProjectDataset]:
    """Discover and harmonize all lab projects under ``data/lab/<project>/``."""
    base = resolve_lab_dir(data_dir, lab_dir=lab_dir)
    project_files = discover_lab_projects(base)
    return {
        project_name: load_lab_project(project_name, files, lab_dir=base)
        for project_name, files in project_files.items()
    }


class ArtifactRepository:
    """Deep repository owning harmonized database artifacts, lab projects, and ML predictor caches."""

    def __init__(
        self,
        database: HarmonizedDatabase,
        predictor: HybridMLPredictor | None = None,
        *,
        raw_dir: str | Path | None = None,
        artifact_dir: str | Path | None = None,
        artifact_status: ArtifactStatus = "live",
        artifact_warning: str | None = None,
        lab_dir: str | Path | None = None,
        raw_fingerprint: dict[str, Any] | None = None,
        lab_projects: dict[str, LabProjectDataset] | None = None,
    ) -> None:
        self.database = database
        self.raw_dir: Path | None = Path(raw_dir) if raw_dir is not None else None
        self.artifact_dir: Path | None = Path(artifact_dir) if artifact_dir is not None else None
        self.artifact_status: ArtifactStatus = artifact_status
        self.artifact_warning: str | None = artifact_warning
        self.lab_dir: Path | None = Path(lab_dir) if lab_dir is not None else None
        if raw_fingerprint is not None:
            self.raw_fingerprint: dict[str, Any] | None = raw_fingerprint
        elif self.raw_dir is not None and iter_raw_csv_files(self.raw_dir):
            self.raw_fingerprint = HarmonizedDatabase.compute_raw_directory_fingerprint(self.raw_dir)
        else:
            self.raw_fingerprint = None

        if lab_projects is not None:
            self.lab_projects: dict[str, LabProjectDataset] = dict(lab_projects)
        elif self.lab_dir is not None:
            self.lab_projects = discover_and_load_lab_projects(self.lab_dir)
        else:
            self.lab_projects = {}
        self.lab_fingerprints: dict[str, dict[str, Any]] = {
            name: ds.fingerprint for name, ds in self.lab_projects.items()
        }
        self.lab_status: LabReloadStatus = "fresh" if self.lab_projects else "none"
        self.lab_status_message: str | None = (
            f"Loaded {len(self.lab_projects)} lab project(s)."
            if self.lab_projects
            else None
        )
        self._lab_calibration_cache: dict[
            tuple[str | None, CalibrationMode, ModelFamily], LabCalibratedPredictor
        ] = {}

        if predictor is None:
            self.predictor = self._load_or_fit_predictor("rf_gp_ensemble")
        else:
            self.predictor = predictor
        self._predictors: dict[ModelFamily, HybridMLPredictor] = {
            self.predictor.model_family: self.predictor
        }

    def _load_or_fit_predictor(self, model_family: ModelFamily) -> HybridMLPredictor:
        sample_metal = (
            self.database.available_metals[0]
            if self.database.available_metals
            else CORE_METALS[0]
        )
        if self.artifact_dir is not None:
            prebuilt_path = self.artifact_dir / "models" / f"{model_family}.joblib"
            if prebuilt_path.exists():
                try:
                    loaded = HybridMLPredictor.load(prebuilt_path)
                    if loaded.model_family != model_family:
                        raise ValueError(
                            f"Model family mismatch in {prebuilt_path.name}: expected {model_family!r}, got {loaded.model_family!r}"
                        )
                    # Smoke-test prediction to catch cross-OS / scikit-learn pickle incompatibilities early
                    loaded.predict(MLFeatureQuery(metal=sample_metal))
                    return loaded
                except Exception as exc:
                    heal_note = (
                        f"Prebuilt model '{prebuilt_path.name}' could not be loaded ({exc}); "
                        "automatically retrained on the current database."
                    )
                    self.artifact_warning = (
                        f"{self.artifact_warning} {heal_note}".strip()
                        if self.artifact_warning
                        else heal_note
                    )

        predictor = HybridMLPredictor(model_family=model_family, random_state=42)
        predictor.fit(self.database)
        if self.artifact_dir is not None:
            try:
                models_dir = self.artifact_dir / "models"
                models_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = models_dir / f".tmp_{model_family}.joblib"
                target_path = models_dir / f"{model_family}.joblib"
                predictor.save(tmp_path)
                tmp_path.replace(target_path)
            except OSError:
                pass
        return predictor

    def get_predictor(self, model_family: ModelFamily | None = None) -> HybridMLPredictor:
        """Return a fitted predictor for ``model_family``, loading prebuilt artifact or fitting on first use."""
        target_family = model_family or self.predictor.model_family
        if target_family not in SUPPORTED_MODEL_FAMILIES:
            raise ValueError(
                f"Unsupported model_family '{target_family}'. Supported: {SUPPORTED_MODEL_FAMILIES}"
            )
        cached = self._predictors.get(target_family)
        if cached is not None:
            return cached
        predictor = self._load_or_fit_predictor(target_family)
        self._predictors[target_family] = predictor
        return predictor

    def get_cached_or_prebuilt_predictor(self, model_family: ModelFamily) -> HybridMLPredictor | None:
        """Return an in-memory or prebuilt on-disk predictor without ever triggering on-the-fly training."""
        cached = self._predictors.get(model_family)
        if cached is not None:
            return cached
        if self.artifact_dir is None:
            return None
        prebuilt_path = self.artifact_dir / "models" / f"{model_family}.joblib"
        if not prebuilt_path.exists():
            return None
        try:
            loaded = HybridMLPredictor.load(prebuilt_path)
            if loaded.model_family != model_family:
                return None
            self._predictors[model_family] = loaded
            return loaded
        except Exception:
            return None

    @classmethod
    def ensure_fresh_on_disk(
        cls,
        raw_dir: str | Path | None = None,
        *,
        processed_dir: str | Path | None = None,
        lab_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        rebuild_models: Sequence[str] = ("all",),
        progress_callback: Callable[[int, int, str], None] | None = None,
        before_rebuild: Callable[[], None] | None = None,
    ) -> str | None:
        """Rebuild on-disk processed artifacts if raw CSVs are newer; return a status banner if rebuilt."""
        from corrosion_simulator.build_artifacts import (
            build_processed_artifacts,
            check_artifacts_freshness,
        )

        raw_path, proc_path, _ = resolve_data_dirs(
            data_dir,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            lab_dir=lab_dir,
        )
        status, _ = check_artifacts_freshness(raw_path, proc_path)
        if status not in ("stale", "missing") or not iter_raw_csv_files(raw_path):
            return None

        if before_rebuild is not None:
            before_rebuild()
        try:
            summary = build_processed_artifacts(
                raw_dir=raw_path,
                output_dir=proc_path,
                models=rebuild_models,
                export_csv=False,
                progress_callback=progress_callback,
            )
            return (
                f"Updated CSV data loaded: harmonized {summary.n_records} records across "
                f"{summary.n_materials} materials and prebuilt all {len(summary.trained_models)} ML models."
            )
        except OSError:
            # Read-only external directory; from_default_data() will fall back to in-memory training
            return None

    @staticmethod
    def _rebuild_or_fallback_raw(
        raw_path: Path,
        target_proc: Path | None,
        *,
        rebuild_models: Sequence[str],
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> tuple[HarmonizedDatabase, Path | None, ArtifactStatus, str | None, bool]:
        """Rebuild processed artifacts from ``raw_path`` or fall back to in-memory load on OSError."""
        from corrosion_simulator.build_artifacts import build_processed_artifacts

        if target_proc is not None:
            try:
                summary = build_processed_artifacts(
                    raw_dir=raw_path,
                    output_dir=target_proc,
                    models=rebuild_models,
                    export_csv=False,
                    progress_callback=progress_callback,
                )
                database = HarmonizedDatabase.from_processed_directory(target_proc)
                rebuilt_msg = (
                    f"Detected updated CSV data in '{raw_path}' — rebuilt database "
                    f"({summary.n_records} records across {summary.n_materials} materials) "
                    f"and trained {len(summary.trained_models)} ML model(s)."
                )
                return database, target_proc, "rebuilt", rebuilt_msg, True
            except OSError as exc:
                warning_msg = (
                    f"Could not write prebuilt cache to '{target_proc}' ({exc}); "
                    f"loaded '{raw_path}' in memory instead."
                )
                return HarmonizedDatabase.from_directory(raw_path), None, "live_fallback", warning_msg, False

        return HarmonizedDatabase.from_directory(raw_path), None, "live_fallback", None, False

    @classmethod
    def from_default_data(
        cls,
        raw_dir: str | Path | None = None,
        *,
        processed_dir: str | Path | None = None,
        lab_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        model_family: ModelFamily = "rf_gp_ensemble",
        auto_rebuild: bool = True,
        rebuild_models: Sequence[str] = ("all",),
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> ArtifactRepository:
        """Load prebuilt artifacts when fresh, auto-rebuild all models when raw CSVs change, or fall back cleanly."""
        from corrosion_simulator.build_artifacts import check_artifacts_freshness

        raw_path, proc_path, lab_path = resolve_data_dirs(
            data_dir,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            lab_dir=lab_dir,
        )
        has_raw_csvs = bool(iter_raw_csv_files(raw_path))
        raw_fp = (
            HarmonizedDatabase.compute_raw_directory_fingerprint(raw_path)
            if has_raw_csvs
            else None
        )
        loaded_lab_projects = discover_and_load_lab_projects(lab_path)
        status, warning_msg = check_artifacts_freshness(raw_path, proc_path)

        if status == "fresh" or (status == "stale" and not auto_rebuild):
            try:
                database = HarmonizedDatabase.from_processed_directory(proc_path)
                repo = cls(
                    database=database,
                    predictor=None,
                    raw_dir=raw_path,
                    artifact_dir=proc_path,
                    artifact_status=status,
                    artifact_warning=warning_msg,
                    lab_dir=lab_path,
                    raw_fingerprint=raw_fp,
                    lab_projects=loaded_lab_projects,
                )
                if model_family != repo.predictor.model_family:
                    repo.predictor = repo.get_predictor(model_family)
                return repo
            except Exception as exc:
                if not has_raw_csvs:
                    raise
                warning_msg = f"Failed to read cached artifacts in '{proc_path}' ({exc}); rebuilding from '{raw_path}'."

        if has_raw_csvs and auto_rebuild:
            database, active_proc, live_status, live_warning, rebuilt = cls._rebuild_or_fallback_raw(
                raw_path,
                proc_path,
                rebuild_models=rebuild_models,
                progress_callback=progress_callback,
            )
            if rebuilt:
                repo = cls(
                    database=database,
                    predictor=None,
                    raw_dir=raw_path,
                    artifact_dir=active_proc,
                    artifact_status=live_status,
                    artifact_warning=live_warning,
                    lab_dir=lab_path,
                    raw_fingerprint=raw_fp,
                    lab_projects=loaded_lab_projects,
                )
                if model_family != repo.predictor.model_family:
                    repo.predictor = repo.get_predictor(model_family)
                return repo
            predictor = HybridMLPredictor(model_family=model_family, random_state=42)
            predictor.fit(database)
            return cls(
                database=database,
                predictor=predictor,
                raw_dir=raw_path,
                artifact_dir=None,
                artifact_status=live_status,
                artifact_warning=live_warning,
                lab_dir=lab_path,
                raw_fingerprint=raw_fp,
                lab_projects=loaded_lab_projects,
            )

        if has_raw_csvs:
            database = HarmonizedDatabase.from_directory(raw_path)
            live_status = "live_fallback"
            live_warning = warning_msg
        else:
            database = HarmonizedDatabase.from_seed()
            live_status = "seed"
            live_warning = (
                f"No CSV training files were found in '{raw_path}' (and no prebuilt cache in '{proc_path}'). "
                "Running on the built-in 8-metal seed dataset (n < 10 -> low data). "
                f"Place your CSV files in '{raw_path}' and restart or click Reload Data."
            )
        predictor = HybridMLPredictor(model_family=model_family, random_state=42)
        predictor.fit(database)
        return cls(
            database=database,
            predictor=predictor,
            raw_dir=raw_path,
            artifact_dir=None,
            artifact_status=live_status,
            artifact_warning=live_warning,
            lab_dir=lab_path,
            raw_fingerprint=raw_fp,
            lab_projects=loaded_lab_projects,
        )

    def _invalidate_lab_cache_for_project(self, project_name: str | None) -> bool:
        keys_to_drop = [
            key for key in self._lab_calibration_cache if key[0] == project_name
        ]
        for key in keys_to_drop:
            del self._lab_calibration_cache[key]
        return bool(keys_to_drop)

    def reload_external_data(
        self,
        *,
        rebuild_models: Sequence[str] = ("all",),
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> ExternalDataReloadSummary:
        """Check independent ``data/raw/`` and ``data/lab/`` fingerprints and reload only what changed."""
        from corrosion_simulator.build_artifacts import check_artifacts_freshness

        start_time = time.perf_counter()
        raw_path = self.raw_dir
        proc_path = self.artifact_dir
        has_raw_csvs = bool(raw_path is not None and iter_raw_csv_files(raw_path))
        current_raw_fp = (
            HarmonizedDatabase.compute_raw_directory_fingerprint(raw_path)
            if has_raw_csvs
            else None
        )
        prev_sha = self.raw_fingerprint.get("combined_sha256") if self.raw_fingerprint else None
        curr_sha = current_raw_fp.get("combined_sha256") if current_raw_fp else None

        raw_changed = prev_sha != curr_sha
        if not raw_changed and has_raw_csvs and proc_path is not None:
            disk_status, _ = check_artifacts_freshness(raw_path, proc_path)
            if disk_status in ("stale", "missing"):
                raw_changed = True

        active_family = self.predictor.model_family
        raw_rebuilt = False
        lab_cache_invalidated = False
        added_or_updated: list[str] = []
        removed: list[str] = []

        if raw_changed:
            if has_raw_csvs and raw_path is not None:
                target_proc = proc_path or (
                    raw_path.parent / "processed" if raw_path.name == "raw" else None
                )
                (
                    self.database,
                    self.artifact_dir,
                    self.artifact_status,
                    self.artifact_warning,
                    raw_rebuilt,
                ) = self._rebuild_or_fallback_raw(
                    raw_path,
                    target_proc,
                    rebuild_models=rebuild_models,
                    progress_callback=progress_callback,
                )
            else:
                self.database = HarmonizedDatabase.from_seed()
                self.artifact_dir = None
                self.artifact_status = "seed"
                self.artifact_warning = (
                    f"No CSV training files were found in '{raw_path}'. Running on built-in seed data."
                )

            self.raw_fingerprint = current_raw_fp
            self._predictors.clear()
            self.predictor = self._load_or_fit_predictor(active_family)
            self._predictors[active_family] = self.predictor

            self._lab_calibration_cache.clear()
            lab_cache_invalidated = True

            if self.lab_dir is not None:
                reloaded_projects = discover_and_load_lab_projects(self.lab_dir)
                added_or_updated = sorted(reloaded_projects.keys())
                removed = sorted(set(self.lab_projects) - set(reloaded_projects))
                self.lab_projects = reloaded_projects
                self.lab_fingerprints = {
                    name: ds.fingerprint for name, ds in reloaded_projects.items()
                }
            self.lab_status = "reloaded" if self.lab_projects else "none"
            self.lab_status_message = (
                f"Re-anchored {len(self.lab_projects)} lab project(s) to updated literature prior."
                if self.lab_projects
                else None
            )
        else:
            if self.lab_dir is not None:
                discovered = discover_lab_projects(self.lab_dir)
                removed = sorted(set(self.lab_projects) - set(discovered))
                for name in removed:
                    self.lab_projects.pop(name, None)
                    self.lab_fingerprints.pop(name, None)
                    if self._invalidate_lab_cache_for_project(name):
                        lab_cache_invalidated = True

                for name, files in discovered.items():
                    new_fp = compute_lab_project_fingerprint(name, files, self.lab_dir)
                    old_sha = self.lab_fingerprints.get(name, {}).get("combined_sha256")
                    if old_sha != new_fp["combined_sha256"]:
                        self.lab_projects[name] = load_lab_project(
                            name, files, lab_dir=self.lab_dir
                        )
                        self.lab_fingerprints[name] = new_fp
                        added_or_updated.append(name)
                        if self._invalidate_lab_cache_for_project(name):
                            lab_cache_invalidated = True
                if added_or_updated and self._invalidate_lab_cache_for_project(None):
                    lab_cache_invalidated = True

            if added_or_updated or removed:
                self.lab_status = "reloaded"
                self.lab_status_message = (
                    f"Reloaded {len(added_or_updated)} lab project(s), removed {len(removed)}."
                )
            elif self.lab_projects:
                self.lab_status = "fresh"
                self.lab_status_message = f"{len(self.lab_projects)} lab project(s) up to date."
            else:
                self.lab_status = "none"
                self.lab_status_message = None

        elapsed = time.perf_counter() - start_time
        return ExternalDataReloadSummary(
            raw_rebuilt=raw_rebuilt,
            raw_status=self.artifact_status,
            lab_status=self.lab_status,
            lab_projects=tuple(sorted(self.lab_projects.keys())),
            lab_added_or_updated=tuple(sorted(added_or_updated)),
            lab_removed=tuple(sorted(removed)),
            lab_cache_invalidated=lab_cache_invalidated,
            elapsed_s=elapsed,
        )

    def get_lab_project(self, project_name: str | None = None) -> LabProjectDataset | None:
        """Return a loaded lab project dataset by name (or the first sorted project if omitted)."""
        if not self.lab_projects:
            return None
        if project_name is None:
            return self.lab_projects[sorted(self.lab_projects)[0]]
        return self.lab_projects.get(project_name)

    def get_calibrated_predictor(
        self,
        *,
        calibration_mode: CalibrationMode = "literature_only",
        lab_project: str | None = None,
        model_family: ModelFamily | None = None,
    ) -> HybridMLPredictor | LabCalibratedPredictor:
        """Return a base or lab-calibrated predictor for the requested calibration mode and project."""
        if calibration_mode not in CALIBRATION_MODES:
            raise ValueError(
                f"Unsupported calibration_mode '{calibration_mode}'. Supported: {CALIBRATION_MODES}"
            )
        target_family = model_family or self.predictor.model_family
        base_predictor = self.get_predictor(target_family)
        if calibration_mode == "literature_only":
            return base_predictor

        lab_ds: LabProjectDataset | None
        if lab_project is not None:
            lab_ds = self.get_lab_project(lab_project)
            if lab_ds is None:
                raise ValueError(
                    f"Lab project {lab_project!r} was not found in {self.lab_dir}. "
                    f"Available projects: {sorted(self.lab_projects)}"
                )
        else:
            lab_ds = self.get_lab_project(None)

        proj_key = lab_ds.project_name if lab_ds is not None else None
        cache_key: tuple[str | None, CalibrationMode, ModelFamily] = (
            proj_key,
            calibration_mode,
            target_family,
        )
        cached = self._lab_calibration_cache.get(cache_key)
        if cached is not None:
            return cached

        calibrated = LabCalibratedPredictor(
            base_predictor=base_predictor,
            calibration_mode=calibration_mode,
            lab_project=lab_ds.project_name if lab_ds is not None else lab_project,
            allow_fallback=True,
        ).fit(lab_ds)
        self._lab_calibration_cache[cache_key] = calibrated
        return calibrated
