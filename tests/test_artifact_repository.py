"""Direct interface tests for ArtifactRepository (disk artifacts, lab projects, and caching)."""

from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from corrosion_simulator.artifact_repository import (
    ArtifactRepository,
    compute_lab_directory_fingerprint,
)
from corrosion_simulator.platform_controller import CorrosionPlatformController


def _write_minimal_raw_csv(raw_dir: Path, n_rows: int = 12) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "record_id": f"REPO_{i}",
            "metal": "Cu" if i % 2 == 0 else "Co",
            "solution_id": f"SOL_{i % 2}",
            "pH": 3.0 + (i % 4) * 0.5,
            "temperature_C": 25.0 + (i % 2) * 5.0,
            "oxidizer_type": "H2O2",
            "oxidizer_concentration": 1.0,
            "Ecorr_V_SHE": 0.24 if i % 2 == 0 else -0.12,
            "icorr_uA_cm2": 3.0 + 0.2 * i,
            "beta_a_mV_dec": 60.0,
            "beta_c_mV_dec": 120.0,
        }
        for i in range(n_rows)
    ]
    csv_path = raw_dir / "training.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _write_lab_project_csv(lab_dir: Path, project: str, n_rows: int = 3) -> Path:
    proj_dir = lab_dir / project
    proj_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "record_id": f"{project}_{i}",
            "metal": "Cu",
            "solution_id": f"SOL_{project}",
            "pH": 4.0 + 0.5 * i,
            "temperature_C": 25.0 + 5.0 * i,
            "oxidizer_type": "H2O2",
            "oxidizer_concentration": 1.0,
            "Ecorr_V_SHE": 0.27 + 0.01 * i,
            "icorr_uA_cm2": 3.5 + 0.5 * i,
            "beta_a_mV_dec": 62.0,
            "beta_c_mV_dec": 118.0,
        }
        for i in range(n_rows)
    ]
    csv_path = proj_dir / "lab.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


class TestArtifactRepository(unittest.TestCase):
    def test_platform_controller_module_has_zero_plotly_imports(self) -> None:
        src_path = Path(__file__).resolve().parents[1] / "corrosion_simulator" / "platform_controller.py"
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        imported_modules = [
            node.module if isinstance(node, ast.ImportFrom) else alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in (node.names if isinstance(node, ast.Import) else [None])
        ]
        self.assertFalse(any(m and "plotly" in m for m in imported_modules))

    def test_ensure_fresh_on_disk_rebuilds_once_and_skips_when_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            raw_dir = data_dir / "raw"
            proc_dir = data_dir / "processed"
            _write_minimal_raw_csv(raw_dir)

            before_called: list[bool] = []
            progress_events: list[tuple[int, int, str]] = []
            banner = ArtifactRepository.ensure_fresh_on_disk(
                raw_dir=raw_dir,
                processed_dir=proc_dir,
                rebuild_models=("random_forest",),
                before_rebuild=lambda: before_called.append(True),
                progress_callback=lambda s, t, m: progress_events.append((s, t, m)),
            )
            self.assertIsNotNone(banner)
            self.assertEqual(len(before_called), 1)
            self.assertGreater(len(progress_events), 0)

            second_banner = ArtifactRepository.ensure_fresh_on_disk(
                raw_dir=raw_dir,
                processed_dir=proc_dir,
                rebuild_models=("random_forest",),
                before_rebuild=lambda: before_called.append(True),
            )
            self.assertIsNone(second_banner)
            self.assertEqual(len(before_called), 1)

    def test_ensure_fresh_on_disk_and_from_default_data_handle_readonly_oserror(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            raw_dir = data_dir / "raw"
            proc_dir = data_dir / "processed"
            _write_minimal_raw_csv(raw_dir)

            with patch(
                "corrosion_simulator.build_artifacts.build_processed_artifacts",
                side_effect=PermissionError("read-only filesystem"),
            ):
                self.assertIsNone(
                    ArtifactRepository.ensure_fresh_on_disk(
                        raw_dir=raw_dir,
                        processed_dir=proc_dir,
                        rebuild_models=("random_forest",),
                    )
                )
                repo = ArtifactRepository.from_default_data(
                    raw_dir=raw_dir,
                    processed_dir=proc_dir,
                    model_family="random_forest",
                    rebuild_models=("random_forest",),
                )
            self.assertEqual(repo.artifact_status, "live_fallback")
            self.assertIsNone(repo.artifact_dir)
            self.assertIn("read-only filesystem", repo.artifact_warning or "")

    def test_corrupted_prebuilt_joblib_auto_heals_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            raw_dir = data_dir / "raw"
            proc_dir = data_dir / "processed"
            _write_minimal_raw_csv(raw_dir)

            ArtifactRepository.from_default_data(
                data_dir=data_dir,
                model_family="random_forest",
                rebuild_models=("random_forest",),
            )
            (proc_dir / "models" / "random_forest.joblib").write_bytes(b"corrupted-pickle")

            healed = ArtifactRepository.from_default_data(
                data_dir=data_dir,
                model_family="random_forest",
                rebuild_models=("random_forest",),
            )
            self.assertEqual(healed.get_predictor("random_forest").model_family, "random_forest")
            self.assertIn("automatically retrained", healed.artifact_warning or "")

    def test_dual_fingerprint_reload_and_selective_lab_calibration_cache_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            raw_dir = data_dir / "raw"
            lab_dir = data_dir / "lab"
            _write_minimal_raw_csv(raw_dir, n_rows=12)
            _write_lab_project_csv(lab_dir, "proj_a", n_rows=3)
            _write_lab_project_csv(lab_dir, "proj_b", n_rows=3)

            repo = ArtifactRepository.from_default_data(
                data_dir=data_dir,
                model_family="random_forest",
                rebuild_models=("random_forest",),
            )
            base_before = repo.predictor
            cal_a_before = repo.get_calibrated_predictor(
                calibration_mode="delta_learning", lab_project="proj_a", model_family="random_forest"
            )
            cal_b_before = repo.get_calibrated_predictor(
                calibration_mode="delta_learning", lab_project="proj_b", model_family="random_forest"
            )

            # Lab-only change invalidates proj_a cache but preserves proj_b cache and base predictor
            _write_lab_project_csv(lab_dir, "proj_a", n_rows=4)
            lab_reload = repo.reload_external_data(rebuild_models=("random_forest",))
            self.assertFalse(lab_reload.raw_rebuilt)
            self.assertEqual(lab_reload.lab_added_or_updated, ("proj_a",))
            self.assertTrue(lab_reload.lab_cache_invalidated)
            self.assertIs(repo.predictor, base_before)
            self.assertIsNot(
                repo.get_calibrated_predictor(
                    calibration_mode="delta_learning", lab_project="proj_a", model_family="random_forest"
                ),
                cal_a_before,
            )
            self.assertIs(
                repo.get_calibrated_predictor(
                    calibration_mode="delta_learning", lab_project="proj_b", model_family="random_forest"
                ),
                cal_b_before,
            )

            # Raw CSV change rebuilds base predictor and invalidates all lab calibration caches
            _write_minimal_raw_csv(raw_dir, n_rows=14)
            raw_reload = repo.reload_external_data(rebuild_models=("random_forest",))
            self.assertTrue(raw_reload.raw_rebuilt)
            self.assertTrue(raw_reload.lab_cache_invalidated)
            self.assertIsNot(repo.predictor, base_before)
            self.assertIsNot(
                repo.get_calibrated_predictor(
                    calibration_mode="delta_learning", lab_project="proj_b", model_family="random_forest"
                ),
                cal_b_before,
            )

    def test_repository_injects_cleanly_into_controller_and_rejects_mixed_kwargs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            raw_dir = data_dir / "raw"
            lab_dir = data_dir / "lab"
            _write_minimal_raw_csv(raw_dir)
            _write_lab_project_csv(lab_dir, "proj_x", n_rows=1)

            fp = compute_lab_directory_fingerprint(lab_dir)
            self.assertEqual(fp["project_count"], 1)
            self.assertIn("proj_x", fp["projects"])

            repo = ArtifactRepository.from_default_data(
                data_dir=data_dir,
                model_family="random_forest",
                rebuild_models=("random_forest",),
            )
            ctrl = CorrosionPlatformController(repo)
            self.assertIs(ctrl.repository, repo)
            self.assertIs(ctrl.database, repo.database)
            self.assertEqual(repo.get_lab_project("proj_x").n_records, 1)  # type: ignore[union-attr]
            self.assertIsNone(repo.get_cached_or_prebuilt_predictor("hist_gradient_boosting"))

            with self.assertRaises(TypeError):
                CorrosionPlatformController(repo, raw_dir=raw_dir)


if __name__ == "__main__":
    unittest.main()
