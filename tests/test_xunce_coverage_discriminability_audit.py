import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceCoverageDiscriminabilityAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="xunce-discriminability-"))
        self.expansion_root = self.temp_dir / "expansion"
        self.coverage_root = self.temp_dir / "coverage"
        self.output_root = self.temp_dir / "audit"
        self.config_path = self.temp_dir / "config.json"
        self.xunce_checkpoint = self.temp_dir / "xunce.pt"
        self.incumbent_checkpoint = self.temp_dir / "incumbent.pt"
        self.xunce_checkpoint.write_bytes(b"xunce")
        self.incumbent_checkpoint.write_bytes(b"incumbent")
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_low_spread_routes_to_candidate_coverage_spread_insufficient(self) -> None:
        from scripts.run_xunce_coverage_discriminability_audit import run_xunce_coverage_discriminability_audit

        self._write_evidence(coverage_values=[0.01, 0.01, 0.01], incumbent_return=0.01, xunce_return=0.01)

        summary = run_xunce_coverage_discriminability_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertIn("candidate_coverage_spread_insufficient", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "repair_coverage_evaluation_discriminability")
        self.assertEqual(summary["root_cause_route"], "candidate_coverage_spread_insufficient")
        self.assertEqual(summary["candidate_coverage_spread_range"], 0.0)
        self.assertTrue((self.output_root / "xunce-coverage-discriminability-audit-summary.json").is_file())
        self.assertTrue((self.output_root / "xunce-coverage-candidate-spread.jsonl").is_file())
        self.assertTrue((self.output_root / "xunce-coverage-oracle-baseline.jsonl").is_file())
        self.assertTrue((self.output_root / "xunce-coverage-root-cause-audit.json").is_file())
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])

    def test_static_candidate_reuse_routes_to_static_reuse(self) -> None:
        from scripts.run_xunce_coverage_discriminability_audit import run_xunce_coverage_discriminability_audit

        self._write_evidence(
            coverage_values=[0.01, 0.03, 0.05],
            incumbent_return=0.01,
            xunce_return=0.01,
            static_reuse=True,
        )

        summary = run_xunce_coverage_discriminability_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertIn("coverage_task_not_discriminative_static_candidate_reuse", summary["reason_codes"])
        self.assertEqual(summary["root_cause_route"], "coverage_task_not_discriminative_static_candidate_reuse")
        self.assertGreaterEqual(summary["static_candidate_reuse_rate"], 0.5)

    def test_oracle_cannot_beat_incumbent_routes_to_map_or_roi_complexity(self) -> None:
        from scripts.run_xunce_coverage_discriminability_audit import run_xunce_coverage_discriminability_audit

        self._write_evidence(
            coverage_values=[0.02, 0.03, 0.04],
            incumbent_return=0.5,
            xunce_return=0.5,
            static_reuse=False,
        )

        summary = run_xunce_coverage_discriminability_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertIn("map_or_roi_complexity_insufficient", summary["reason_codes"])
        self.assertEqual(summary["root_cause_route"], "map_or_roi_complexity_insufficient")
        self.assertLessEqual(summary["oracle_vs_incumbent_coverage_delta"], 0.0)

    def test_oracle_can_beat_models_routes_to_training_or_adapter_iteration(self) -> None:
        from scripts.run_xunce_coverage_discriminability_audit import run_xunce_coverage_discriminability_audit

        self._write_evidence(
            coverage_values=[0.01, 0.04, 0.09],
            incumbent_return=0.02,
            xunce_return=0.02,
            static_reuse=False,
            useful_disagreement=True,
        )

        summary = run_xunce_coverage_discriminability_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertIn("models_genuinely_no_coverage_advantage", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "xunce_training_or_adapter_iteration_required")
        self.assertGreater(summary["oracle_vs_incumbent_coverage_delta"], 0.0)
        self.assertGreater(summary["xunce_oracle_regret"], 0.0)

    def test_endpoint_tie_with_path_line_spread_marks_metric_too_coarse(self) -> None:
        from scripts.run_xunce_coverage_discriminability_audit import run_xunce_coverage_discriminability_audit

        self._write_evidence(
            coverage_values=[0.02, 0.04, 0.08],
            path_line_values=[0.02, 0.12, 0.2],
            incumbent_return=0.08,
            xunce_return=0.08,
            endpoint_tie=True,
            static_reuse=False,
        )

        summary = run_xunce_coverage_discriminability_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertIn("coverage_metric_too_coarse", summary["reason_codes"])
        self.assertEqual(summary["root_cause_route"], "coverage_metric_too_coarse")
        self.assertGreater(summary["path_line_coverage_spread_range"], summary["candidate_coverage_spread_range"])

    def _write_config(self) -> None:
        payload = {
            "schema_version": "xunce-coverage-discriminability-audit-config/v1",
            "source_roi_expansion_root": str(self.expansion_root),
            "source_coverage_comparison_root": str(self.coverage_root),
            "xunce_candidate_checkpoint": str(self.xunce_checkpoint),
            "incumbent_policy_checkpoint": str(self.incumbent_checkpoint),
            "required_scenario_count": 4,
            "rollout_steps": 4,
            "min_candidate_coverage_spread": 0.01,
            "max_static_candidate_reuse_rate": 0.5,
            "min_roi_group_with_nonzero_spread_count": 1,
            "canary_traffic_fraction": 0.0,
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_evidence(
        self,
        *,
        coverage_values: list[float],
        incumbent_return: float,
        xunce_return: float,
        path_line_values: list[float] | None = None,
        static_reuse: bool = True,
        useful_disagreement: bool = False,
        endpoint_tie: bool = False,
    ) -> None:
        self.expansion_root.mkdir(parents=True, exist_ok=True)
        self.coverage_root.mkdir(parents=True, exist_ok=True)
        path_line_values = path_line_values or coverage_values
        scenarios = []
        slices = []
        for index in range(4):
            scenario_id = f"scenario_{index:03d}"
            roi_group = f"roi_{index % 2}"
            candidates = []
            for action_index, coverage in enumerate(coverage_values):
                candidates.append(
                    {
                        "action_index": action_index,
                        "cell": [index * 10 + action_index, action_index],
                        "reachable": True,
                        "path_cost": 10.0 + action_index,
                        "risk": 0.1 + action_index * 0.01,
                        "expected_coverage_rate_delta": coverage,
                        "expected_new_coverage_area": coverage * 1000.0,
                        "expected_path_line_coverage_delta": path_line_values[action_index],
                    }
                )
            scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "roi_group": roi_group,
                    "path_feedback": {"candidates": candidates},
                    "coverage_endpoint_tie_observed": endpoint_tie,
                }
            )
            slices.append(
                {
                    "scenario_id": scenario_id,
                    "roi_name": roi_group,
                    "split": "train",
                    "context_id": f"context-{scenario_id}",
                    "contract": str(self.expansion_root / f"{scenario_id}.contract.json"),
                    "sidecar": str(self.expansion_root / f"{scenario_id}.sidecar.json"),
                }
            )
        self._write_json(
            self.expansion_root / "xunce-high-fidelity-real-map-roi-expansion-summary.json",
            {
                "schema_version": "xunce-high-fidelity-real-map-roi-expansion-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "slice_count": 4,
                "roi_group_count": 2,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
            },
        )
        (self.expansion_root / "xunce-high-fidelity-real-map-slices.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in slices) + "\n",
            encoding="utf-8",
        )
        self._write_json(
            self.expansion_root / "xunce-high-fidelity-path-feedback-audit.json",
            {
                "schema_version": "xunce-high-fidelity-path-feedback-audit/v1",
                "scenario_count": 4,
                "candidate_count": 4 * len(coverage_values),
                "reachable_count": 4 * len(coverage_values),
                "fallback_or_open_grid_count": 0,
                "scenarios": scenarios,
            },
        )
        self._write_json(
            self.coverage_root / "xunce-exploration-coverage-comparison-summary.json",
            {
                "schema_version": "xunce-exploration-coverage-comparison-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "scenario_count": 4,
                "rollout_steps": 4,
                "xunce_coverage_return_delta_vs_incumbent": xunce_return - incumbent_return,
                "xunce_coverage_curve_auc_delta_vs_incumbent": xunce_return - incumbent_return,
                "policy_disagreement_count": 4 if useful_disagreement else 0,
                "useful_disagreement_count": 0,
                "proxy_selection_used": False,
                "true_model_inference_executed": True,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
            },
        )
        episodes = []
        for index in range(4):
            scenario_id = f"scenario_{index:03d}"
            for policy, coverage_return in (("xunce", xunce_return), ("incumbent", incumbent_return)):
                episodes.append(
                    {
                        "scenario_id": scenario_id,
                        "roi_group": f"roi_{index % 2}",
                        "policy": policy,
                        "coverage_return": coverage_return,
                        "coverage_curve_auc": coverage_return,
                        "final_coverage_rate": coverage_return,
                        "new_covered_cell_count": int(coverage_return * 1000),
                        "path_cost": 10.0,
                        "risk": 0.1,
                    }
                )
        self._write_jsonl(self.coverage_root / "xunce-exploration-coverage-episodes.jsonl", episodes)
        steps = []
        for index in range(4):
            scenario_id = f"scenario_{index:03d}"
            for policy in ("xunce", "incumbent"):
                for step in range(4):
                    if static_reuse:
                        candidate_cells = [[index, 0], [index, 1], [index, 2]]
                    else:
                        candidate_cells = [[index, step], [index, step + 1], [index, step + 2]]
                    action = 2 if policy == "xunce" and useful_disagreement else 0
                    steps.append(
                        {
                            "scenario_id": scenario_id,
                            "policy": policy,
                            "step_index": step,
                            "selected_action_index": action,
                            "coverage_rate_delta": xunce_return if policy == "xunce" else incumbent_return,
                            "candidate_cells": candidate_cells,
                            "model_inference_mask_violation": False,
                            "executed": True,
                        }
                    )
        self._write_jsonl(self.coverage_root / "xunce-exploration-coverage-steps.jsonl", steps)

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
