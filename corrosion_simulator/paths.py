"""Runtime path resolution separating bundled resources (_MEIPASS) from external user data."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import os
from pathlib import Path
import sys
from typing import Any, Final

DATA_DIR_ENV_VAR: Final[str] = "CORROSION_SIMULATOR_DATA_DIR"
_IGNORED_RAW_CSV_NAMES: Final[frozenset[str]] = frozenset({"rejected_rows.csv"})


def is_frozen() -> bool:
    """Return True when running inside a PyInstaller frozen executable."""
    return bool(getattr(sys, "frozen", False))


def get_bundle_dir() -> Path:
    """Return directory containing bundled internal code/assets (e.g., ``app.py``).

    In PyInstaller mode this points to ``sys._MEIPASS`` (``_internal/`` in PyInstaller >= 6).
    In normal Python development mode this points to the repository root.
    """
    if is_frozen() and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")).resolve()
    return Path(__file__).resolve().parents[1]


def get_executable_dir() -> Path:
    """Return directory containing the outer executable (or repository root in dev mode)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def get_external_data_dir(data_dir: str | Path | None = None) -> Path:
    """Resolve the external ``data/`` directory.

    Resolution priority:
    1. Explicit ``data_dir`` argument when provided.
    2. ``CORROSION_SIMULATOR_DATA_DIR`` environment variable when set.
    3. ``<exe_dir>/data`` when running as a PyInstaller executable.
    4. ``<repo_root>/data`` when running from Python source.
    """
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()
    env_override = os.environ.get(DATA_DIR_ENV_VAR, "").strip()
    if env_override:
        return Path(env_override).expanduser().resolve()
    if is_frozen():
        return get_executable_dir() / "data"
    return get_bundle_dir() / "data"


def iter_raw_csv_files(raw_dir: str | Path) -> list[Path]:
    """Recursively list raw CSV files under ``raw_dir``, excluding ``processed/`` and ``lab/``."""
    base = Path(raw_dir)
    if not base.exists() or not base.is_dir():
        return []
    discovered: list[Path] = []
    for csv_path in sorted(base.rglob("*.csv")):
        if not csv_path.is_file():
            continue
        rel_parts = {part.lower() for part in csv_path.relative_to(base).parts[:-1]}
        if rel_parts & {"processed", "lab"}:
            continue
        if csv_path.name.lower() in _IGNORED_RAW_CSV_NAMES:
            continue
        discovered.append(csv_path)
    return discovered


def resolve_raw_and_processed_dirs(
    data_dir: str | Path | None = None,
    *,
    raw_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve ``(raw_dir, processed_dir)`` for external user CSVs and prebuilt artifacts.

    Defaults to ``(<data_dir>/raw, <data_dir>/processed)``. If ``<data_dir>/raw`` has no CSV
    files yet ``<data_dir>`` itself contains top-level ``*.csv`` files, ``raw_dir`` falls back
    to ``<data_dir>`` (while ``iter_raw_csv_files`` automatically skips ``processed/`` and ``lab/``).
    """
    base = get_external_data_dir(data_dir)
    resolved_proc = Path(processed_dir) if processed_dir is not None else (base / "processed")
    if raw_dir is not None:
        resolved_raw = Path(raw_dir)
    else:
        candidate_raw = base / "raw"
        if iter_raw_csv_files(candidate_raw):
            resolved_raw = candidate_raw
        elif base.exists() and any(
            p.is_file() and p.name.lower() not in _IGNORED_RAW_CSV_NAMES for p in base.glob("*.csv")
        ):
            resolved_raw = base
        else:
            resolved_raw = candidate_raw
    return resolved_raw, resolved_proc


def resolve_lab_dir(
    data_dir: str | Path | None = None,
    *,
    lab_dir: str | Path | None = None,
) -> Path:
    """Resolve the external ``data/lab/`` directory for small-sample laboratory datasets."""
    if lab_dir is not None:
        return Path(lab_dir).expanduser().resolve()
    return get_external_data_dir(data_dir) / "lab"


def resolve_data_dirs(
    data_dir: str | Path | None = None,
    *,
    raw_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
    lab_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve ``(raw_dir, processed_dir, lab_dir)`` under the external data directory."""
    resolved_raw, resolved_proc = resolve_raw_and_processed_dirs(
        data_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )
    resolved_lab = resolve_lab_dir(data_dir, lab_dir=lab_dir)
    return resolved_raw, resolved_proc, resolved_lab


def discover_lab_projects(
    lab_dir: str | Path | None = None,
    *,
    data_dir: str | Path | None = None,
) -> dict[str, list[Path]]:
    """Enumerate named laboratory projects and their CSV files under ``data/lab/<project>/``.

    Top-level CSV files placed directly under ``data/lab/`` (outside subdirectories) map to
    the ``"default"`` project name.
    """
    base = resolve_lab_dir(data_dir, lab_dir=lab_dir)
    if not base.exists() or not base.is_dir():
        return {}

    projects: dict[str, list[Path]] = {}
    top_level = [
        p
        for p in sorted(base.glob("*.csv"))
        if p.is_file() and p.name.lower() not in _IGNORED_RAW_CSV_NAMES
    ]
    if top_level:
        projects["default"] = top_level

    for sub in sorted(base.iterdir()):
        if not sub.is_dir() or sub.name.startswith(".") or sub.name.lower() == "processed":
            continue
        csvs = iter_raw_csv_files(sub)
        if not csvs:
            continue
        if sub.name in projects:
            merged = sorted({*projects[sub.name], *csvs})
            projects[sub.name] = merged
        else:
            projects[sub.name] = csvs
    return projects


def compute_csv_fingerprint(
    csv_files: Sequence[Path],
    base_path: Path,
) -> dict[str, Any]:
    """Compute a deterministic SHA-256 fingerprint across a sorted collection of CSV files."""
    files_info: dict[str, dict[str, Any]] = {}
    combined_hasher = hashlib.sha256()
    for csv_path in sorted(csv_files):
        try:
            rel = str(csv_path.relative_to(base_path)).replace("\\", "/")
        except ValueError:
            rel = csv_path.name
        data = csv_path.read_bytes()
        file_sha = hashlib.sha256(data).hexdigest()
        files_info[rel] = {
            "sha256": file_sha,
            "size_bytes": len(data),
        }
        combined_hasher.update(rel.encode("utf-8"))
        combined_hasher.update(b":")
        combined_hasher.update(file_sha.encode("utf-8"))
        combined_hasher.update(b"\n")
    return {
        "combined_sha256": combined_hasher.hexdigest(),
        "file_count": len(files_info),
        "files": files_info,
    }
