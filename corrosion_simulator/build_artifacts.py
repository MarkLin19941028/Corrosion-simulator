"""Offline artifact build pipeline (ETL + Parquet tables + prebuilt ML models + manifest).

Run from the repository root whenever files under ``data/raw/`` change:

    .venv/bin/python -m corrosion_simulator.build_artifacts

Outputs to ``data/processed/`` by default:
- ``electrochemical_database.parquet`` (plus ``material_properties``, ``solution_properties``,
  ``galvanic_series``, ``data_dictionary``)
- Optional CSV/Markdown mirrors and ``rejected_rows.csv``
- ``models/<model_family>.joblib`` for prebuilt ML predictors
- ``manifest.json`` recording input SHA-256 fingerprint, material coverage, and build diagnostics
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Final, Literal, Sequence

from corrosion_simulator.database import CORE_METALS, HarmonizedDatabase
from corrosion_simulator.ml_predictor import (
    GP_MAX_TRAINING_SAMPLES,
    SUPPORTED_MODEL_FAMILIES,
    HybridMLPredictor,
    ModelFamily,
)

ArtifactFreshnessStatus = Literal["fresh", "stale", "missing"]
MANIFEST_SCHEMA_VERSION: Final[int] = 1
DEFAULT_PREBUILT_MODELS: Final[tuple[ModelFamily, ...]] = (
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
)


@dataclass(frozen=True)
class ArtifactBuildSummary:
    """Structured summary produced by :func:`build_processed_artifacts`."""

    output_dir: Path
    manifest_path: Path
    n_records: int
    n_rejected: int
    n_materials: int
    available_metals: list[str]
    missing_faraday_materials: list[str]
    trained_models: list[str]
    warnings: list[str]
    manifest: dict[str, Any]


def load_manifest(processed_dir: str | Path = "data/processed") -> dict[str, Any] | None:
    """Read ``manifest.json`` from ``processed_dir``, or return ``None`` if absent/invalid."""
    manifest_path = Path(processed_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def check_artifacts_freshness(
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
) -> tuple[ArtifactFreshnessStatus, str | None]:
    """Check whether prebuilt artifacts exist and match the current ``raw_dir`` fingerprint.

    Returns ``(status, message)`` where ``status`` is one of:
    - ``"fresh"``: prebuilt Parquet tables and manifest exist and match ``raw_dir``.
    - ``"stale"``: prebuilt artifacts exist, but ``raw_dir`` has changed since build.
    - ``"missing"``: no complete prebuilt artifact bundle exists.
    """
    proc_path = Path(processed_dir)
    manifest = load_manifest(proc_path)
    required_parquet = proc_path / "electrochemical_database.parquet"
    if manifest is None or not required_parquet.exists():
        return (
            "missing",
            "Prebuilt artifacts not found in data/processed/. Run `.venv/bin/python -m corrosion_simulator.build_artifacts` for faster startup.",
        )

    raw_path = Path(raw_dir)
    if not raw_path.exists() or not any(raw_path.rglob("*.csv")):
        return ("fresh", None)

    current_fp = HarmonizedDatabase.compute_raw_directory_fingerprint(raw_path)
    recorded_sha = manifest.get("raw_fingerprint", {}).get("combined_sha256")
    if recorded_sha and recorded_sha == current_fp["combined_sha256"]:
        return ("fresh", None)

    return (
        "stale",
        "Raw CSV files in `data/raw/` have changed since the last build. "
        "Running on cached artifacts — execute `.venv/bin/python -m corrosion_simulator.build_artifacts` to refresh.",
    )


def _parse_model_families(models: Sequence[str] | None) -> list[ModelFamily]:
    if not models:
        return list(DEFAULT_PREBUILT_MODELS)
    parsed: list[ModelFamily] = []
    for item in models:
        for token in str(item).split(","):
            name = token.strip()
            if not name:
                continue
            if name.lower() == "all":
                return list(SUPPORTED_MODEL_FAMILIES)
            if name not in SUPPORTED_MODEL_FAMILIES:
                raise ValueError(
                    f"Unsupported model family {name!r}. Supported: {SUPPORTED_MODEL_FAMILIES}"
                )
            if name not in parsed:
                parsed.append(name)  # type: ignore[arg-type]
    return parsed or list(DEFAULT_PREBUILT_MODELS)


def build_processed_artifacts(
    raw_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/processed",
    *,
    models: Sequence[str] | None = None,
    gp_max_samples: int = GP_MAX_TRAINING_SAMPLES,
    export_csv: bool = True,
    filter_to_core_metals: bool = False,
    supplement_missing_core_metals: bool = False,
    strict: bool = False,
    random_state: int = 42,
) -> ArtifactBuildSummary:
    """Execute the full offline ETL + ML model build pipeline and persist to ``output_dir``."""
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fingerprint = HarmonizedDatabase.compute_raw_directory_fingerprint(raw_path)
    database = HarmonizedDatabase.from_directory(
        raw_path,
        filter_to_core_metals=filter_to_core_metals,
        supplement_missing_core_metals=supplement_missing_core_metals,
    )

    # Persist Parquet tables (primary fast runtime format) + optional CSV/Markdown mirrors
    parquet_paths = database.export_to_parquet(out_path)
    if export_csv:
        database.export_to_directory(out_path)
    else:
        database.rejected_records.to_csv(out_path / "rejected_rows.csv", index=False)

    meta_df = database.material_metadata()
    missing_faraday = (
        meta_df.loc[~meta_df["has_faraday_properties"], "metal"].tolist()
        if not meta_df.empty
        else []
    )
    build_warnings: list[str] = []
    n_rejected = len(database.rejected_records)
    if n_rejected > 0:
        build_warnings.append(
            f"Quarantined {n_rejected} invalid/incomplete raw record(s) to rejected_rows.csv."
        )
    if missing_faraday:
        build_warnings.append(
            f"Materials lacking Faraday physical properties (density/equivalent weight): {', '.join(missing_faraday)}."
        )

    selected_families = _parse_model_families(models)
    models_dir = out_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    trained_models: list[str] = []
    model_artifacts: dict[str, dict[str, Any]] = {}

    for family in selected_families:
        predictor = HybridMLPredictor(
            model_family=family,
            random_state=random_state,
            max_gp_samples=gp_max_samples,
        )
        predictor.fit(database)
        model_path = models_dir / f"{family}.joblib"
        predictor.save(model_path)
        trained_models.append(family)
        gp_capped = any(
            getattr(m, "gp_subsampled", False) for m in predictor.models.values()
        )
        if gp_capped:
            build_warnings.append(
                f"Model {family!r} subsampled GP training set to {gp_max_samples} samples for scalability."
            )
        model_artifacts[family] = {
            "path": f"models/{family}.joblib",
            "target_training_counts": dict(predictor.target_training_counts),
            "gp_subsampled": gp_capped,
        }

    counts = database.metal_sample_counts()
    combined_ml = database.ml_training_dataframe
    icorr_src = combined_ml["icorr_source"] if "icorr_source" in combined_ml.columns else None
    n_derived_cr = int((icorr_src == "derived_from_CR").sum()) if icorr_src is not None else 0
    n_derived_rp = int((icorr_src == "derived_from_Rp").sum()) if icorr_src is not None else 0
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_fingerprint": fingerprint,
        "filter_to_core_metals": bool(filter_to_core_metals),
        "supplement_missing_core_metals": bool(supplement_missing_core_metals),
        "n_records": int(len(database.electrochemical_database)),
        "n_complete_pairs": int(len(database.electrochemical_database)),
        "n_partial_records": int(len(database.partial_target_records)),
        "n_derived_icorr_cr": n_derived_cr,
        "n_derived_icorr_rp": n_derived_rp,
        "n_rejected": int(n_rejected),
        "n_solutions": int(len(database.solution_properties)),
        "n_materials": int(len(database.available_metals)),
        "available_metals": database.available_metals,
        "core_metals_present": [m for m in CORE_METALS if m in counts],
        "extended_metals_present": [m for m in database.available_metals if m not in CORE_METALS],
        "metal_sample_counts": counts,
        "missing_faraday_materials": missing_faraday,
        "tables": {k: v.name for k, v in parquet_paths.items()},
        "models": model_artifacts,
        "warnings": build_warnings,
    }
    manifest_path = out_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = ArtifactBuildSummary(
        output_dir=out_path,
        manifest_path=manifest_path,
        n_records=int(len(database.electrochemical_database)),
        n_rejected=int(n_rejected),
        n_materials=int(len(database.available_metals)),
        available_metals=database.available_metals,
        missing_faraday_materials=missing_faraday,
        trained_models=trained_models,
        warnings=build_warnings,
        manifest=manifest,
    )

    if strict and (n_rejected > 0 or missing_faraday):
        raise RuntimeError(
            "Strict build failed: " + "; ".join(build_warnings)
        )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build harmonized Parquet tables and prebuilt ML models from data/raw/."
    )
    parser.add_argument("--raw-dir", default="data/raw", help="Directory containing raw CSV files.")
    parser.add_argument(
        "--out-dir",
        default="data/processed",
        help="Destination directory for Parquet tables, models, and manifest.json.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=(
            "Model families to prebuild (comma-separated or space-separated, or 'all'). "
            f"Default: {','.join(DEFAULT_PREBUILT_MODELS)}"
        ),
    )
    parser.add_argument(
        "--gp-max-samples",
        type=int,
        default=GP_MAX_TRAINING_SAMPLES,
        help=f"Maximum training samples for Gaussian Process models (default: {GP_MAX_TRAINING_SAMPLES}).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing CSV mirrors alongside Parquet tables.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail with non-zero exit code if any rows are rejected or any material lacks Faraday properties.",
    )
    args = parser.parse_args(argv)

    try:
        summary = build_processed_artifacts(
            raw_dir=args.raw_dir,
            output_dir=args.out_dir,
            models=args.models,
            gp_max_samples=args.gp_max_samples,
            export_csv=not args.no_csv,
            strict=args.strict,
        )
    except Exception as exc:
        print(f"[build_artifacts] ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"[build_artifacts] Built {summary.n_records} records across {summary.n_materials} materials "
        f"({summary.n_rejected} rejected) -> {summary.output_dir}"
    )
    print(f"[build_artifacts] Materials: {', '.join(summary.available_metals)}")
    print(f"[build_artifacts] Prebuilt models: {', '.join(summary.trained_models)}")
    for w in summary.warnings:
        print(f"[build_artifacts] WARNING: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
