import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceTrueIncumbentSelectionBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="xunce-true-incumbent-binding-"))
        self.materialized_root = self.temp_dir / "stage18e"
        self.inference_root = self.temp_dir / "stage18b"
        self.output_root = self.temp_dir / "stage18g0"
        self.config_path = self.temp_dir / "config.json"
        self._write_materialized_root()
        self._write_inference_root()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_binds_true_checkpoint_selection_without_fallback(self) -> None:
        from scripts.run_xunce_true_incumbent_selection_binding import run_xunce_true_incumbent_selection_binding

        summary = run_xunce_true_incumbent_selection_binding(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["true_incumbent_selection_bound"])
        self.assertEqual(summary["incumbent_selected_action_index_missing_count"], 0)
        self.assertEqual(summary["fallback_action_index_0_count"], 0)
        self.assertEqual(summary["candidate_cell_mismatch_count"], 0)
        self.assertTrue(summary["true_model_inference_executed"])
        self.assertFalse(summary["proxy_selection_used"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["connects_real_executor"])

        bound = json.loads((self.output_root / "xunce-high-fidelity-path-feedback-audit.json").read_text(encoding="utf-8"))
        scenario = bound["scenarios"][0]
        self.assertEqual(scenario["incumbent_selected_action_index"], 1)
        self.assertEqual(scenario["incumbent_selection_source"], "true_checkpoint_inference")
        self.assertEqual(scenario["xunce_selected_action_index"], 2)
        self.assertEqual(scenario["xunce_selection_source"], "true_checkpoint_inference")
        self.assertTrue((self.output_root / "xunce-true-incumbent-selection-binding-summary.json").is_file())
        self.assertTrue((self.output_root / "xunce-true-incumbent-selection-binding-audit.json").is_file())
        self.assertTrue((self.output_root / "xunce-true-incumbent-selection-binding-report.md").is_file())
        self.assertTrue((self.output_root / "xunce-candidate-level-coverage-opportunity-summary.json").is_file())

    def test_candidate_cell_mismatch_fails_without_fallback(self) -> None:
        from scripts.run_xunce_true_incumbent_selection_binding import run_xunce_true_incumbent_selection_binding

        rows = self._read_jsonl(self.inference_root / "xunce-high-fidelity-model-inference-results.jsonl")
        rows[0]["candidate_cells"][1] = [999, 999]
        self._write_jsonl(self.inference_root / "xunce-high-fidelity-model-inference-results.jsonl", rows)

        summary = run_xunce_true_incumbent_selection_binding(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("candidate_cell_mismatch", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "repair_stage18e_candidate_alignment")

    def test_accepts_true_frontier_nbv_candidate_source_summary_alias(self) -> None:
        from scripts.run_xunce_true_incumbent_selection_binding import run_xunce_true_incumbent_selection_binding

        (self.materialized_root / "xunce-candidate-level-coverage-opportunity-summary.json").unlink()
        self._write_json(
            self.materialized_root / "xunce-true-frontier-nbv-candidate-source-summary.json",
            {
                "schema_version": "xunce-true-frontier-nbv-candidate-source-summary/v1",
                "status": "passed",
                "candidate_validation_mode": "in_process_evaluate_candidate_paths",
                "next_required_change": "rerun_true_model_inference_and_binding",
                "canary_traffic_fraction": 0.0,
            },
        )

        summary = run_xunce_true_incumbent_selection_binding(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["true_incumbent_selection_bound"])
        self.assertEqual(summary["source_materialization_status"], "passed")

    def test_non_true_inference_fails_to_stage18b_route(self) -> None:
        from scripts.run_xunce_true_incumbent_selection_binding import run_xunce_true_incumbent_selection_binding

        rows = self._read_jsonl(self.inference_root / "xunce-high-fidelity-model-inference-results.jsonl")
        rows[0]["true_model_inference_executed"] = False
        self._write_jsonl(self.inference_root / "xunce-high-fidelity-model-inference-results.jsonl", rows)

        summary = run_xunce_true_incumbent_selection_binding(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("true_inference_not_executed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "run_xunce_high_fidelity_real_map_comparison")

    def _write_config(self) -> None:
        payload = {
            "schema_version": "xunce-true-incumbent-selection-binding-config/v1",
            "source_materialized_coverage_root": str(self.materialized_root),
            "source_model_inference_root": str(self.inference_root),
            "required_scenario_count": 3,
            "require_true_model_inference": True,
            "require_candidate_cell_match": True,
            "allow_fallback_action_index": False,
            "canary_traffic_fraction": 0.0,
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_materialized_root(self) -> None:
        self.materialized_root.mkdir(parents=True, exist_ok=True)
        scenarios = []
        slices = []
        for index in range(3):
            scenario_id = f"scenario_{index:03d}"
            cells = [[index * 10 + 1, 1], [index * 10 + 2, 2], [index * 10 + 3, 3]]
            candidates = [
                {"action_index": i, "cell": cell, "reachable": True, "path_cost": 10.0 - i, "risk": 0.1, "roi_weighted_coverage_delta": 0.02 + i * 0.02, "path_line_coverage_delta": 0.02 + i * 0.02, "revisit_penalty": 0.0}
                for i, cell in enumerate(cells)
            ]
            scenarios.append({"scenario_id": scenario_id, "roi_group": f"roi_{index}", "path_feedback": {"candidates": candidates}, "open_grid_fallback_used": False})
            slices.append({"scenario_id": scenario_id, "roi_name": f"roi_{index}", "split": "train", "context_id": f"context-{scenario_id}", "contract": "c.json", "sidecar": "s.json"})
        self._write_json(self.materialized_root / "xunce-candidate-level-coverage-opportunity-summary.json", {"schema_version": "xunce-candidate-level-coverage-opportunity-summary/v1", "status": "passed", "canary_traffic_fraction": 0.0})
        self._write_json(self.materialized_root / "xunce-high-fidelity-real-map-roi-expansion-summary.json", {"schema_version": "xunce-high-fidelity-real-map-roi-expansion-summary/v1", "status": "passed", "slice_count": 3, "roi_group_count": 3, "canary_traffic_fraction": 0.0})
        self._write_jsonl(self.materialized_root / "xunce-high-fidelity-real-map-slices.jsonl", slices)
        self._write_json(self.materialized_root / "xunce-high-fidelity-path-feedback-audit.json", {"schema_version": "xunce-high-fidelity-path-feedback-audit/v1", "scenario_count": 3, "candidate_count": 9, "reachable_count": 9, "scenarios": scenarios})

    def _write_inference_root(self) -> None:
        self.inference_root.mkdir(parents=True, exist_ok=True)
        rows = []
        for index in range(3):
            scenario_id = f"scenario_{index:03d}"
            cells = [[index * 10 + 1, 1], [index * 10 + 2, 2], [index * 10 + 3, 3]]
            rows.append(
                {
                    "schema_version": "xunce-high-fidelity-model-inference-result/v1",
                    "scenario_id": scenario_id,
                    "candidate_cells": cells,
                    "true_model_inference_executed": True,
                    "action_mask": [True, True, True],
                    "incumbent": {"selected_action_index": 1, "selected_probability": 0.7, "selected_rank": 1, "value": 1.0, "latency_ms": 0.1, "finite_outputs": True},
                    "xunce": {"selected_action_index": 2, "selected_probability": 0.8, "selected_rank": 1, "value": 1.2, "latency_ms": 0.2, "finite_outputs": True},
                }
            )
        self._write_jsonl(self.inference_root / "xunce-high-fidelity-model-inference-results.jsonl", rows)
        self._write_json(self.inference_root / "xunce-high-fidelity-model-inference-audit.json", {"schema_version": "xunce-high-fidelity-model-inference-audit/v1", "true_model_inference_executed": True, "proxy_selection_used": False, "xunce_checkpoint_loaded": True, "incumbent_checkpoint_loaded": True, "reason_codes": []})
        self._write_json(self.inference_root / "xunce-high-fidelity-real-map-comparison-summary.json", {"schema_version": "xunce-high-fidelity-real-map-comparison-summary/v1", "status": "passed", "true_model_inference_executed": True, "proxy_selection_used": False, "xunce_checkpoint_loaded": True, "incumbent_checkpoint_loaded": True})

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
