from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


class QuasiRealGuardedTeacherFollowingPilotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="quasi-real-teacher-following-"))
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_dir = str(self.repo_root / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        self.source_root = self.temp_dir / "source"
        self.candidate_root = self.temp_dir / "candidate"
        self.quasi_root = self.temp_dir / "quasi"
        self.output_root = self.temp_dir / "out"
        for path in (self.source_root, self.candidate_root, self.quasi_root):
            path.mkdir(parents=True, exist_ok=True)
        self._write_candidate_files()
        self._write_quasi_files()

    def test_source_aligned_policy_passes_as_teacher_following(self) -> None:
        from scripts.run_quasi_real_guarded_teacher_following_pilot import (
            run_quasi_real_guarded_teacher_following_pilot,
        )

        summary = run_quasi_real_guarded_teacher_following_pilot(
            source_root=self.source_root,
            candidate_root=self.candidate_root,
            quasi_real_root=self.quasi_root,
            output_root=self.output_root,
            config=self._config(),
            repo_root=self.repo_root,
            score_decisions=lambda **_: [
                self._decision("smooth-001", "smooth_high_confidence", 0, 0, []),
                self._decision("rim-001", "rim_or_steep_slope", 1, 1, []),
            ],
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["teacher_following_pilot_verdict"], "teacher_following_pilot_validated")
        self.assertEqual(summary["quasi_real_context_count"], 2)
        self.assertEqual(summary["policy_decision_count"], 2)
        self.assertEqual(summary["teacher_following_step_count"], 2)
        self.assertEqual(summary["teacher_agreement_rate"], 1.0)
        self.assertEqual(summary["safe_disagreement_count"], 0)
        self.assertEqual(summary["unsafe_disagreement_count"], 0)
        self.assertEqual(summary["policy_changed_gate_rejected_count"], 0)
        self.assertFalse(summary["runs_ppo_update"])
        self.assertFalse(summary["writes_ppo_transition"])

        records = self._read_decisions()
        self.assertEqual(
            {record["controlled_choice_source"] for record in records},
            {"policy_teacher_aligned"},
        )

    def test_safe_disagreement_is_allowed(self) -> None:
        from scripts.run_quasi_real_guarded_teacher_following_pilot import (
            run_quasi_real_guarded_teacher_following_pilot,
        )

        summary = run_quasi_real_guarded_teacher_following_pilot(
            source_root=self.source_root,
            candidate_root=self.candidate_root,
            quasi_real_root=self.quasi_root,
            output_root=self.output_root,
            config=self._config(),
            repo_root=self.repo_root,
            score_decisions=lambda **_: [
                self._decision("smooth-001", "smooth_high_confidence", 0, 0, []),
                self._decision("rim-001", "rim_or_steep_slope", 0, 1, []),
            ],
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["teacher_following_step_count"], 1)
        self.assertEqual(summary["safe_disagreement_count"], 1)
        self.assertEqual(summary["unsafe_disagreement_count"], 0)
        records = self._read_decisions()
        self.assertIn(
            "policy_safe_disagreement",
            {record["controlled_choice_source"] for record in records},
        )

    def test_unsafe_disagreement_fails_and_falls_back_to_teacher(self) -> None:
        from scripts.run_quasi_real_guarded_teacher_following_pilot import (
            run_quasi_real_guarded_teacher_following_pilot,
        )

        summary = run_quasi_real_guarded_teacher_following_pilot(
            source_root=self.source_root,
            candidate_root=self.candidate_root,
            quasi_real_root=self.quasi_root,
            output_root=self.output_root,
            config=self._config(),
            repo_root=self.repo_root,
            score_decisions=lambda **_: [
                self._decision("smooth-001", "smooth_high_confidence", 0, 0, []),
                self._decision("rim-001", "rim_or_steep_slope", 0, 1, ["path_cost_regression"]),
            ],
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(
            summary["teacher_following_pilot_verdict"],
            "quasi_real_guarded_teacher_following_gate_regression",
        )
        self.assertIn("quasi_real_guarded_teacher_following_gate_regression", summary["reason_codes"])
        self.assertEqual(summary["unsafe_disagreement_count"], 1)
        self.assertEqual(summary["policy_changed_gate_rejected_count"], 1)
        self.assertEqual(summary["source_fallback_count"], 1)
        self.assertEqual(summary["path_cost_regression_count"], 1)
        records = self._read_decisions()
        fallback = [record for record in records if record["decision_class"] == "policy_changed_gate_rejected"]
        self.assertEqual(fallback[0]["controlled_choice_source"], "teacher_fallback")
        self.assertEqual(fallback[0]["controlled_action_index"], fallback[0]["teacher_action_index"])

    def test_readiness_accepts_passed_teacher_following_summary(self) -> None:
        from scripts.run_policy_training_readiness_review import (
            _quasi_real_guarded_teacher_following_pilot_readiness,
        )

        readiness = _quasi_real_guarded_teacher_following_pilot_readiness(
            {
                "schema_version": "quasi-real-guarded-teacher-following-pilot-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "teacher_following_pilot_verdict": "teacher_following_pilot_validated",
                "quasi_real_context_count": 108,
                "policy_decision_count": 108,
                "roi_group_count": 4,
                "context_id_missing_count": 0,
                "teacher_agreement_rate": 1.0,
                "teacher_following_step_count": 108,
                "unsafe_disagreement_count": 0,
                "policy_changed_gate_rejected_count": 0,
                "invalid_action_mask_count": 0,
                "fallback_or_open_grid_count": 0,
                "safety_regression_count": 0,
                "contract_violation_count": 0,
                "path_cost_regression_count": 0,
                "risk_regression_count": 0,
                "source_selection_regression_count": 0,
                "runs_ppo_update": False,
                "writes_ppo_transition": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
            }
        )

        self.assertTrue(readiness["completed"])
        self.assertEqual(readiness["training_blockers"], [])

    def _write_candidate_files(self) -> None:
        (self.source_root / "batch-evaluation-summary.json").write_text(
            json.dumps({"schema_version": "batch-evaluation-summary/v1", "failed_count": 0}),
            encoding="utf-8",
        )
        (self.candidate_root / "raw-policy-generalization-candidate-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "raw-policy-generalization-candidate-summary/v1",
                    "status": "passed",
                    "candidate_training_status": "passed",
                    "reason_codes": [],
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                }
            ),
            encoding="utf-8",
        )
        (self.candidate_root / "experimental-hybrid-policy-candidate-metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": "controlled-hybrid-policy-candidate-checkpoint-metadata/v1",
                    "experimental": True,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                }
            ),
            encoding="utf-8",
        )
        (self.candidate_root / "experimental-hybrid-policy-candidate.pt").write_bytes(b"dummy")

    def _write_quasi_files(self) -> None:
        scenarios = [
            self._scenario("smooth-001", "smooth_high_confidence"),
            self._scenario("rim-001", "rim_or_steep_slope"),
        ]
        (self.quasi_root / "quasi-real-map-path-feedback-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "path-feedback-summary/v1",
                    "status": "completed",
                    "scenario_count": 2,
                    "candidate_count": 4,
                    "reachable_count": 4,
                    "open_grid_fallback_used": False,
                    "scenarios": scenarios,
                }
            ),
            encoding="utf-8",
        )
        slices = [
            self._slice("smooth-001", "smooth_high_confidence", "train"),
            self._slice("rim-001", "rim_or_steep_slope", "test"),
        ]
        (self.quasi_root / "quasi-real-map-slices.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in slices),
            encoding="utf-8",
        )
        (self.quasi_root / "quasi-real-map-domain-gap-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "quasi-real-map-domain-gap-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "domain_gap_verdict": "acceptable_for_next_pilot",
                    "slice_count": 2,
                    "roi_group_count": 2,
                }
            ),
            encoding="utf-8",
        )

    def _scenario(self, scenario_id: str, group: str) -> dict[str, object]:
        return {
            "scenario_id": scenario_id,
            "scenario_group": group,
            "scenario_seed": 7,
            "scenario_variant_id": f"{scenario_id}-v1",
            "path_feedback": {
                "candidates": [
                    self._candidate(scenario_id, 0, "source_selected", 10.0),
                    self._candidate(scenario_id, 1, "not_source_candidate", 9.0),
                ]
            },
        }

    def _candidate(
        self,
        scenario_id: str,
        action_index: int,
        source_status: str,
        path_cost: float,
    ) -> dict[str, object]:
        return {
            "action_index": action_index,
            "source_action_index": action_index,
            "cell": [action_index, action_index + 1],
            "candidate_role": "policy_target",
            "policy_target_cell": [action_index, action_index + 1],
            "execution_goal_cell": [action_index, action_index + 1],
            "reachable": True,
            "replan_required": False,
            "path_cost": path_cost,
            "risk": 0.1 + action_index * 0.01,
            "utility": 1.0 - action_index * 0.01,
            "context_id": f"{scenario_id}-{action_index}",
            "context_id_schema_version": "policy-context-id/v1",
            "context_id_source": "stable_semantic_fields",
            "candidate_generation": {"source_selection_status": source_status},
            "platform_goal_feasibility": {"contract_reachable": True},
        }

    def _slice(self, scenario_id: str, group: str, split: str) -> dict[str, object]:
        return {
            "schema_version": "quasi-real-map-slice/v1",
            "scenario_id": scenario_id,
            "scenario_group": group,
            "roi_group": group,
            "roi_name": group,
            "split": split,
            "map_id": "lola-test",
            "slice_id": scenario_id,
            "dataset_id": "lunar_south_pole_lro_lola_gdr_875s_20m",
            "context_id": f"{scenario_id}-slice",
            "context_id_schema_version": "policy-context-id/v1",
            "context_id_source": "stable_semantic_fields",
            "legacy_identity_fallback_used": False,
        }

    def _decision(
        self,
        scenario_id: str,
        group: str,
        source_action: int,
        raw_action: int,
        gate_reasons: list[str],
    ) -> dict[str, object]:
        return {
            "schema_version": "scenario-disjoint-policy-rollout-decision/v1",
            "scenario_id": scenario_id,
            "scenario_group": group,
            "context_id": f"{scenario_id}-{raw_action}",
            "source_selected_context_id": f"{scenario_id}-{source_action}",
            "raw_policy_selected_context_id": f"{scenario_id}-{raw_action}",
            "source_selected_action_index": source_action,
            "raw_policy_selected_action_index": raw_action,
            "source_selected_policy_logit": 0.2,
            "raw_policy_selected_policy_logit": 0.6,
            "raw_policy_logit_margin_vs_source": 0.4,
            "action_mask_valid": "invalid_action_mask" not in gate_reasons,
            "raw_policy_selected_path_cost_delta": -1.0 if not gate_reasons else 2.0,
            "raw_policy_selected_risk_delta": 0.0,
            "raw_policy_regression_reason_codes": gate_reasons,
        }

    def _config(self) -> dict[str, object]:
        return {
            "schema_version": "quasi-real-guarded-teacher-following-pilot-config/v1",
            "input_files": {
                "source_batch_summary": "batch-evaluation-summary.json",
                "quasi_real_path_feedback_summary": "quasi-real-map-path-feedback-summary.json",
                "quasi_real_slices": "quasi-real-map-slices.jsonl",
                "quasi_real_domain_gap_summary": "quasi-real-map-domain-gap-summary.json",
                "candidate_summary": "raw-policy-generalization-candidate-summary.json",
                "checkpoint": "experimental-hybrid-policy-candidate.pt",
                "checkpoint_metadata": "experimental-hybrid-policy-candidate-metadata.json",
            },
            "output_files": {
                "decisions": "quasi-real-guarded-teacher-following-decisions.jsonl",
                "summary": "quasi-real-guarded-teacher-following-pilot-summary.json",
                "rejection_report": "quasi-real-guarded-teacher-following-rejection-report.json",
                "group_report": "quasi-real-guarded-teacher-following-group-report.md",
            },
            "validation": {
                "min_quasi_real_context_count": 2,
                "min_roi_group_count": 2,
                "min_teacher_agreement_rate": 0.5,
                "min_teacher_following_step_count": 1,
                "max_invalid_action_mask_count": 0,
                "max_fallback_or_open_grid_count": 0,
                "max_safety_regression_count": 0,
                "max_contract_violation_count": 0,
                "max_path_cost_regression_count": 0,
                "max_risk_regression_count": 0,
                "max_source_selection_regression_count": 0,
                "max_policy_changed_gate_rejected_count": 0,
            },
            "evaluation": {"hidden_size": 16, "max_path_cost_regression": 0.0, "max_risk_regression": 0.0},
            "non_goals": ["no_ppo_update", "no_checkpoint_publication"],
        }

    def _read_decisions(self) -> list[dict[str, object]]:
        path = self.output_root / "quasi-real-guarded-teacher-following-decisions.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    unittest.main()
