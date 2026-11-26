"""External ``data/lab/<project>/`` discovery, SHA-256 fingerprinting, and standalone harmonization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from corrosion_simulator.database import HarmonizedDatabase
from corrosion_simulator.paths import (
    compute_csv_fingerprint,
    discover_lab_projects,
    resolve_lab_dir,
)


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
