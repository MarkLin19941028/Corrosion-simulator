"""Runtime path resolution separating bundled resources (_MEIPASS) from external user data."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Final

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
    """Recursively list raw CSV files under ``raw_dir``, excluding ``processed/`` artifacts."""
    base = Path(raw_dir)
    if not base.exists() or not base.is_dir():
        return []
    discovered: list[Path] = []
    for csv_path in sorted(base.rglob("*.csv")):
        if not csv_path.is_file():
            continue
        rel_parts = {part.lower() for part in csv_path.relative_to(base).parts[:-1]}
        if "processed" in rel_parts:
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
    to ``<data_dir>`` (while ``iter_raw_csv_files`` automatically skips ``processed/``).
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
