import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class QuasiRealMapDomainGapEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        for path in (self.repo_root / "scripts", self.repo_root / "model-explorer" / "src"):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="qreal-domain-gap-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_data_prepare_downloads_missing_manifest_files_and_validates_hashes(self) -> None:
        from scripts.run_quasi_real_lola_data_prepare import run_quasi_real_lola_data_prepare

        source_dir = self.temp_dir / "source"
        source_dir.mkdir()
        dem = source_dir / "dem.jp2"
        count = source_dir / "count.jp2"
        dem.write_bytes(b"dem-fixture")
        count.write_bytes(b"count-fixture")
        raw_dir = self.temp_dir / "raw"
        manifest = self._data_manifest(raw_dir, dem, count)

        summary = run_quasi_real_lola_data_prepare(
            manifest_path=manifest,
            output_root=self.temp_dir / "prepare-out",
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["downloaded_file_count"], 2)
        self.assertEqual(summary["checked_file_count"], 2)
        self.assertEqual(summary["sha256_mismatch_count"], 0)
        self.assertTrue((raw_dir / "dem.jp2").is_file())
        self.assertTrue((raw_dir / "count.jp2").is_file())

    def test_bridge_generates_path_feedback_manifest_with_lola_sidecars_and_context_ids(self) -> None:
        from model_explorer.policy.path_feedback import validate_path_feedback_manifest
        from scripts.run_quasi_real_map_path_feedback_bridge import run_quasi_real_map_path_feedback_bridge

        matrix = self._matrix_manifest()

        summary = run_quasi_real_map_path_feedback_bridge(
            matrix_manifest_path=matrix,
            output_root=self.temp_dir / "bridge-out",
            config={"top_k": 3, "max_slices": 12},
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertGreaterEqual(summary["slice_count"], 12)
        self.assertEqual(summary["context_id_missing_count"], 0)
        self.assertEqual(summary["legacy_identity_fallback_count"], 0)
        self.assertEqual(
            set(summary["roi_groups"]),
            {"smooth_high_confidence", "rim_or_steep_slope", "low_observation_count", "mixed_risk"},
        )
        manifest_path = Path(summary["path_feedback_manifest"])
        validation = validate_path_feedback_manifest(manifest_path)
        self.assertEqual(validation["status"], "valid")
        sidecar_path = Path(json.loads(manifest_path.read_text(encoding="utf-8"))["scenarios"][0]["sidecar"])
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        self.assertEqual(sidecar["metadata"]["map_source"]["kind"], "lola_quasi_real_roi")
        self.assertIn("risk", sidecar["terrain_layers"])

    def test_domain_gap_summary_passes_for_clean_quasi_real_path_feedback(self) -> None:
        from scripts.run_quasi_real_map_domain_gap_evaluation import run_quasi_real_map_domain_gap_evaluation

        output_root = self.temp_dir / "domain-gap"
        bridge_summary = {
            "schema_version": "quasi-real-map-path-feedback-bridge-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "slice_count": 12,
            "context_id_missing_count": 0,
            "legacy_identity_fallback_count": 0,
            "roi_groups": ["smooth_high_confidence", "rim_or_steep_slope", "low_observation_count", "mixed_risk"],
        }
        quasi_summary = self._path_feedback_summary(scenario_set="quasi_real_map_domain_gap")
        generated_summary = self._path_feedback_summary(scenario_set="policy_canary_sequential_multi_step_opportunity")

        summary = run_quasi_real_map_domain_gap_evaluation(
            bridge_summary=bridge_summary,
            quasi_real_path_feedback_summary=quasi_summary,
            generated_reference_summary=generated_summary,
            output_root=output_root,
            config={"validation": {"min_slice_count": 12, "min_roi_group_count": 4}},
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["domain_gap_verdict"], "acceptable_for_next_pilot")
        self.assertEqual(summary["context_id_missing_count"], 0)
        self.assertEqual(summary["fallback_count"], 0)
        self.assertEqual(summary["open_grid_fallback_count"], 0)
        self.assertEqual(summary["fallback_or_open_grid_count"], 0)
        self.assertEqual(summary["contract_regression_count"], 0)
        self.assertTrue((output_root / "quasi-real-map-domain-gap-report.md").is_file())

    def test_readiness_accepts_passed_quasi_real_domain_gap_summary(self) -> None:
        from scripts.run_policy_training_readiness_review import _quasi_real_map_domain_gap_readiness

        readiness = _quasi_real_map_domain_gap_readiness(
            {
                "schema_version": "quasi-real-map-domain-gap-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "domain_gap_verdict": "acceptable_for_next_pilot",
                "slice_count": 12,
                "roi_group_count": 4,
                "context_id_missing_count": 0,
                "legacy_identity_fallback_count": 0,
                "invalid_action_mask_count": 0,
                "fallback_or_open_grid_count": 0,
                "safety_regression_count": 0,
                "contract_violation_count": 0,
                "path_cost_regression_count": 0,
                "risk_regression_count": 0,
                "source_selection_regression_count": 0,
                "git_provenance": {"current_matches_sources": True},
            }
        )

        self.assertTrue(readiness["present"])
        self.assertTrue(readiness["completed"])
        self.assertEqual(readiness["training_blockers"], [])

    def _matrix_manifest(self) -> Path:
        try:
            from PIL import Image
        except ImportError as exc:
            raise unittest.SkipTest("Pillow is not available") from exc

        raw_dir = self.temp_dir / "raw"
        raw_dir.mkdir()
        dem_path = raw_dir / "dem.jp2"
        count_path = raw_dir / "count.jp2"
        dem_values = []
        count_values = []
        for y in range(16):
            for x in range(16):
                dem_values.append((x + y) * 3 + (24 if x >= 8 and y >= 8 else 0))
                count_values.append(max(0, 15 - x - (3 if y >= 8 else 0)))
        dem_image = Image.new("L", (16, 16))
        dem_image.putdata(dem_values)
        dem_image.save(dem_path, format="PNG")
        count_image = Image.new("L", (16, 16))
        count_image.putdata(count_values)
        count_image.save(count_path, format="PNG")
        data_manifest = self._data_manifest(raw_dir, dem_path, count_path, urls=False)
        matrix = self.temp_dir / "matrix.json"
        rois = [
            ("smooth_high_confidence", "train", 0, 0),
            ("smooth_high_confidence", "validation", 1, 0),
            ("smooth_high_confidence", "test", 0, 1),
            ("rim_or_steep_slope", "train", 8, 8),
            ("rim_or_steep_slope", "validation", 9, 8),
            ("rim_or_steep_slope", "test", 8, 9),
            ("low_observation_count", "train", 8, 0),
            ("low_observation_count", "validation", 9, 0),
            ("low_observation_count", "test", 8, 1),
            ("mixed_risk", "train", 0, 8),
            ("mixed_risk", "validation", 1, 8),
            ("mixed_risk", "test", 0, 9),
        ]
        matrix.write_text(
            json.dumps(
                {
                    "schema_version": "model-explorer-quasi-real-evaluation/v1",
                    "name": "fixture-qreal-domain-gap",
                    "run_id": "fixture-domain-gap",
                    "dataset_manifest": str(data_manifest),
                    "output_root": str(self.temp_dir / "processed"),
                    "candidate_count": 4,
                    "episode_count": 1,
                    "seed": 7,
                    "rois": [
                        {
                            "name": name,
                            "split": split,
                            "roi_x": x,
                            "roi_y": y,
                            "roi_width": 4,
                            "roi_height": 4,
                        }
                        for name, split, x, y in rois
                    ],
                }
            ),
            encoding="utf-8",
        )
        return matrix

    def _data_manifest(self, raw_dir: Path, dem: Path, count: Path, *, urls: bool = True) -> Path:
        manifest = self.temp_dir / f"manifest-{len(list(self.temp_dir.glob('manifest-*.json')))}.json"
        products = []
        for role, source in (("shape_map_radius", dem), ("observation_count", count)):
            entry = {
                "name": source.name,
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest().upper(),
            }
            if urls:
                entry["url"] = source.resolve().as_uri()
            products.append({"product_id": role, "role": role, "files": [entry]})
        manifest.write_text(
            json.dumps(
                {
                    "dataset_id": "fixture_lola",
                    "data_class": "quasi_real",
                    "region": "lunar_south_pole",
                    "local_raw_dir": str(raw_dir),
                    "projection": {"map_scale_meters_per_pixel": 20},
                    "products": products,
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def _path_feedback_summary(self, *, scenario_set: str) -> dict:
        scenarios = [
            {
                "scenario_id": f"{scenario_set}-{index}",
                "scenario_group": "smooth_high_confidence",
                "candidate_count": 3,
                "reachable_count": 3,
                "open_grid_fallback_used": False,
                "path_planning_failure_count": 0,
                "replan_count": 0,
                "tracking_safety_violation_count": 0,
                "trajectory_optimization_fallback_count": 0,
                "region_graph_disconnected_count": 0,
                "selected_path_cost_after_feedback": 10.0 + index,
                "path_cost_delta_after_feedback": -0.5,
                "coverage_rate_delta": 0.1,
                "path_feedback": {
                    "candidates": [
                        {"reachable": True, "path_cost": 10.0, "risk": 0.1, "source_selected": True}
                    ]
                },
            }
            for index in range(12)
        ]
        return {
            "schema_version": "path-feedback-summary/v1",
            "status": "passed",
            "scenario_set": scenario_set,
            "scenario_count": 12,
            "candidate_count": 36,
            "reachable_count": 36,
            "path_planning_failure_count": 0,
            "replan_count": 0,
            "open_grid_fallback_used": False,
            "open_grid_fallback_used_count": 0,
            "tracking_safety_violation_count": 0,
            "trajectory_optimization_fallback_count": 0,
            "region_graph_disconnected_count": 0,
            "average_path_cost": 10.0,
            "coverage_per_path_cost": 0.1,
            "scenarios": scenarios,
        }


if __name__ == "__main__":
    unittest.main()
