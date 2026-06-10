from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


class QuasiRealShadowFailureTaxonomyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="qreal-shadow-taxonomy-"))
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_dir = str(self.repo_root / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        self.shadow_root = self.temp_dir / "shadow"
        self.quasi_root = self.temp_dir / "quasi"
        self.output_root = self.temp_dir / "taxonomy"
        self.shadow_root.mkdir()
        self.quasi_root.mkdir()
        self._write_shadow_artifacts()
        self._write_quasi_artifacts()

    def test_taxonomy_classifies_joint_path_risk_regression_and_feature_delta(self) -> None:
        from scripts.run_quasi_real_shadow_failure_taxonomy import (
            run_quasi_real_shadow_failure_taxonomy,
        )

        summary = run_quasi_real_shadow_failure_taxonomy(
            shadow_root=self.shadow_root,
            quasi_real_root=self.quasi_root,
            output_root=self.output_root,
            config=self._config(),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(summary["path_risk_joint_regression_count"], 1)
        self.assertEqual(summary["bridge_or_feedback_gap_count"], 0)
        self.assertEqual(summary["action_mask_or_contract_gap_count"], 0)
        records = self._read_jsonl(self.output_root / "quasi-real-shadow-failure-taxonomy.jsonl")
        self.assertEqual(records[0]["failure_class"], "path_risk_joint_regression")
        self.assertEqual(records[0]["scenario_id"], "lola_qreal_mixed_risk_test_011")
        self.assertEqual(records[0]["roi_group"], "mixed_risk")
        self.assertEqual(records[0]["source_action_index"], 1)
        self.assertEqual(records[0]["raw_policy_action_index"], 2)
        self.assertGreater(records[0]["path_cost_delta"], 0.0)
        self.assertGreater(records[0]["risk_delta"], 0.0)

        features = self._read_jsonl(self.output_root / "quasi-real-shadow-feature-audit.jsonl")
        self.assertEqual(features[0]["scenario_id"], "lola_qreal_mixed_risk_test_011")
        self.assertEqual(features[0]["source_candidate"]["context_id"], "ctx-source")
        self.assertEqual(features[0]["alternative_candidate"]["context_id"], "ctx-alt")
        self.assertEqual(features[0]["metric_delta"]["path_cost"], 1.9)
        self.assertEqual(features[0]["metric_delta"]["risk"], 0.014)

    def test_taxonomy_separates_action_mask_or_contract_gap(self) -> None:
        self._write_shadow_artifacts(gate_reasons=["invalid_action_mask"])
        from scripts.run_quasi_real_shadow_failure_taxonomy import (
            run_quasi_real_shadow_failure_taxonomy,
        )

        summary = run_quasi_real_shadow_failure_taxonomy(
            shadow_root=self.shadow_root,
            quasi_real_root=self.quasi_root,
            output_root=self.output_root,
            config=self._config(),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["action_mask_or_contract_gap_count"], 1)
        self.assertIn("quasi_real_shadow_action_mask_or_contract_gap", summary["reason_codes"])

    def _write_shadow_artifacts(self, gate_reasons: list[str] | None = None) -> None:
        gate_reasons = gate_reasons or ["path_cost_regression", "risk_regression"]
        decision = {
            "schema_version": "quasi-real-shadow-policy-decision/v1",
            "context_id": "ctx-alt",
            "scenario_id": "lola_qreal_mixed_risk_test_011",
            "roi_group": "mixed_risk",
            "roi_name": "mixed_risk",
            "split": "test",
            "map_id": "lola",
            "slice_id": "slice-011",
            "source_action_index": 1,
            "raw_policy_action_index": 2,
            "logit_margin": 0.269,
            "action_mask_valid": "invalid_action_mask" not in gate_reasons,
            "path_cost_delta": 1.9,
            "risk_delta": 0.014,
            "gate_reason_codes": gate_reasons,
            "decision_class": "policy_changed_gate_rejected",
        }
        (self.shadow_root / "quasi-real-shadow-policy-decisions.jsonl").write_text(
            json.dumps(decision, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (self.shadow_root / "quasi-real-shadow-policy-rejection-report.json").write_text(
            json.dumps(
                {
                    "schema_version": "quasi-real-shadow-policy-rejection-report/v1",
                    "status": "failed",
                    "rejected_count": 1,
                    "rejections": [decision],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.shadow_root / "quasi-real-shadow-policy-behavior-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "quasi-real-shadow-policy-behavior-summary/v1",
                    "status": "failed",
                    "reason_codes": ["quasi_real_shadow_gate_regression"],
                    "behavior_verdict": "policy_real_map_alignment_refinement_required",
                    "shadow_context_count": 1,
                    "policy_decision_count": 1,
                    "policy_changed_gate_rejected_count": 1,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_quasi_artifacts(self) -> None:
        scenario = {
            "scenario_id": "lola_qreal_mixed_risk_test_011",
            "scenario_group": "mixed_risk",
            "path_feedback": {
                "candidates": [
                    self._candidate("ctx-source", 1, 10.0, 0.10, "source_selected"),
                    self._candidate("ctx-alt", 2, 11.9, 0.114, "not_source_selected"),
                ]
            },
        }
        (self.quasi_root / "quasi-real-map-path-feedback-summary.json").write_text(
            json.dumps({"schema_version": "path-feedback-summary/v1", "scenarios": [scenario]}),
            encoding="utf-8",
        )
        slice_record = {
            "schema_version": "quasi-real-map-slice/v1",
            "scenario_id": "lola_qreal_mixed_risk_test_011",
            "scenario_group": "mixed_risk",
            "roi_name": "mixed_risk",
            "split": "test",
            "map_id": "lola",
            "slice_id": "slice-011",
            "context_id": "slice-ctx",
        }
        (self.quasi_root / "quasi-real-map-slices.jsonl").write_text(
            json.dumps(slice_record) + "\n",
            encoding="utf-8",
        )

    def _candidate(
        self,
        context_id: str,
        action_index: int,
        path_cost: float,
        risk: float,
        source_status: str,
    ) -> dict[str, object]:
        return {
            "context_id": context_id,
            "action_index": action_index,
            "source_action_index": action_index,
            "policy_target_cell": [action_index, 4],
            "execution_goal_cell": [action_index, 4],
            "reachable": True,
            "replan_required": False,
            "path_cost": path_cost,
            "risk": risk,
            "utility": 1.0 - risk,
            "candidate_generation": {"source_selection_status": source_status},
            "platform_goal_feasibility": {"contract_reachable": True},
        }

    def _config(self) -> dict[str, object]:
        return {
            "input_files": {
                "shadow_decisions": "quasi-real-shadow-policy-decisions.jsonl",
                "shadow_rejection_report": "quasi-real-shadow-policy-rejection-report.json",
                "quasi_real_slices": "quasi-real-map-slices.jsonl",
                "quasi_real_path_feedback_summary": "quasi-real-map-path-feedback-summary.json",
            },
            "output_files": {
                "summary": "quasi-real-shadow-failure-taxonomy-summary.json",
                "taxonomy": "quasi-real-shadow-failure-taxonomy.jsonl",
                "feature_audit": "quasi-real-shadow-feature-audit.jsonl",
                "report": "quasi-real-shadow-failure-report.md",
            },
            "validation": {
                "max_bridge_or_feedback_gap_count": 0,
                "max_action_mask_or_contract_gap_count": 0,
            },
        }

    def _read_jsonl(self, path: Path) -> list[dict[str, object]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
