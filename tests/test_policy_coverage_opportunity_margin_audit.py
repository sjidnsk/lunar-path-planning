import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class PolicyCoverageOpportunityMarginAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="policy-coverage-margin-"))
        self.coverage_driven_root = self.temp_dir / "coverage-driven"
        self.reward_root = self.temp_dir / "reward-refinement"
        self.signal_root = self.temp_dir / "coverage-signal"
        self.performance_root = self.temp_dir / "coverage-performance"
        self.output_root = self.temp_dir / "policy-margin-output"
        for path in (
            self.coverage_driven_root,
            self.reward_root,
            self.signal_root,
            self.performance_root,
        ):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_routes_to_collect_more_policy_differentiating_coverage_when_decision_features_are_zero(self) -> None:
        from scripts.run_policy_coverage_opportunity_margin_audit import (
            run_policy_coverage_opportunity_margin_audit,
        )

        transition = self._transition(
            context_id="ctx-zero",
            teacher_action_index=0,
            features=[
                self._candidate(expected=0.0, information=0.0, value=0.0, risk=0.20, path_cost=0.50),
                self._candidate(expected=0.0, information=0.0, value=0.0, risk=0.18, path_cost=0.40),
            ],
            pre_logits=[1.0, 0.5],
            post_logits=[1.1, 0.6],
        )
        self._write_inputs([transition], replay_rows=[self._replay_row("ctx-zero", teacher=0, raw=0, controlled=0)])

        summary = run_policy_coverage_opportunity_margin_audit(
            coverage_driven_root=self.coverage_driven_root,
            reward_refinement_root=self.reward_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(
            summary["next_required_change"],
            "collect_more_policy_differentiating_coverage",
        )
        self.assertIn("candidate_coverage_features_missing", summary["reason_codes"])
        self.assertEqual(summary["safe_better_than_teacher_count"], 0)
        self.assertEqual(summary["candidate_expected_coverage_nonzero_count"], 0)
        self.assertEqual(summary["candidate_information_gain_nonzero_count"], 0)
        self.assertEqual(summary["candidate_value_nonzero_count"], 0)
        self.assertEqual(summary["post_update_teacher_equal_raw_count"], 1)
        self.assertEqual(summary["post_update_teacher_equal_controlled_count"], 1)
        self.assertEqual(summary["policy_argmax_changed_count"], 0)
        self.assertEqual(summary["teacher_margin_sample_count"], 1)
        self.assertEqual(summary["missing_teacher_signal_count"], 0)
        self.assertEqual(summary["controlled_regression_count"], 0)
        self.assertEqual(summary["fallback_gain_contamination_count"], 0)
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])
        self.assertFalse(summary["formal_release_claimed"])

        for filename in (
            "policy-coverage-opportunity-margin-audit-summary.json",
            "policy-coverage-action-level-audit.jsonl",
            "policy-margin-audit.json",
            "candidate-feature-audit.json",
            "policy-coverage-opportunity-rejection-report.json",
            "policy-coverage-opportunity-margin-audit-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_routes_to_reward_refinement_when_safe_non_teacher_candidate_has_better_decision_signal(self) -> None:
        from scripts.run_policy_coverage_opportunity_margin_audit import (
            run_policy_coverage_opportunity_margin_audit,
        )

        transition = self._transition(
            context_id="ctx-better",
            teacher_action_index=0,
            features=[
                self._candidate(expected=0.02, information=0.01, value=0.00, risk=0.20, path_cost=0.50),
                self._candidate(expected=0.20, information=0.10, value=0.05, risk=0.19, path_cost=0.45),
            ],
            pre_logits=[1.0, 0.4],
            post_logits=[0.6, 1.2],
        )
        self._write_inputs([transition], replay_rows=[self._replay_row("ctx-better", teacher=0, raw=0, controlled=0)])

        summary = run_policy_coverage_opportunity_margin_audit(
            coverage_driven_root=self.coverage_driven_root,
            reward_refinement_root=self.reward_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["next_required_change"], "refine_coverage_reward_or_advantage")
        self.assertEqual(summary["safe_better_than_teacher_count"], 1)
        self.assertEqual(summary["safe_better_than_teacher_family_count"], 1)
        self.assertEqual(summary["policy_argmax_changed_count"], 1)
        self.assertGreater(summary["candidate_expected_coverage_nonzero_count"], 0)
        self.assertGreater(summary["candidate_information_gain_nonzero_count"], 0)
        self.assertGreater(summary["candidate_value_nonzero_count"], 0)

        action_rows = self._read_jsonl(self.output_root / "policy-coverage-action-level-audit.jsonl")
        self.assertEqual(len(action_rows), 2)
        better_rows = [row for row in action_rows if row["is_best_safe_non_teacher_alternative"]]
        self.assertEqual(len(better_rows), 1)
        self.assertTrue(better_rows[0]["safe_better_than_teacher"])
        self.assertEqual(better_rows[0]["action_index"], 1)

    def test_candidate_coverage_overlay_can_supply_decision_time_coverage_signal(self) -> None:
        from scripts.run_policy_coverage_opportunity_margin_audit import (
            run_policy_coverage_opportunity_margin_audit,
        )

        transition = self._transition(
            context_id="ctx-overlay",
            teacher_action_index=0,
            features=[
                self._candidate(expected=0.0, information=0.0, value=0.0, risk=0.20, path_cost=0.50),
                self._candidate(expected=0.0, information=0.0, value=0.0, risk=0.19, path_cost=0.45),
            ],
            pre_logits=[1.0, 0.4],
            post_logits=[1.0, 0.4],
        )
        self._write_inputs([transition], replay_rows=[self._replay_row("ctx-overlay", teacher=0, raw=0, controlled=0)])
        overlay_path = self.temp_dir / "candidate-coverage-overlay.jsonl"
        self._write_jsonl(
            overlay_path,
            [
                {
                    "schema_version": "candidate-level-coverage-overlay-row/v1",
                    "context_id": "ctx-overlay",
                    "scenario_id": "scenario-a",
                    "scenario_family": "family-a",
                    "split": "train",
                    "action_index": 0,
                    "candidate_cell": [0, 0],
                    "coverage_source_available": True,
                    "expected_coverage_rate_delta": 0.01,
                    "expected_new_coverage_area": 0.01,
                    "information_gain": 0.0,
                    "value": 0.0,
                    "valuable_coverage_proxy": 0.005,
                    "path_cost": 0.50,
                    "risk": 0.20,
                    "energy_cost": 0.0,
                    "source_path": "fixtures/source.json",
                    "match_method": "executed_action_actual_path_feedback",
                    "source_confidence": 1.0,
                },
                {
                    "schema_version": "candidate-level-coverage-overlay-row/v1",
                    "context_id": "ctx-overlay",
                    "scenario_id": "scenario-a",
                    "scenario_family": "family-a",
                    "split": "train",
                    "action_index": 1,
                    "candidate_cell": [1, 1],
                    "coverage_source_available": True,
                    "expected_coverage_rate_delta": 0.20,
                    "expected_new_coverage_area": 0.20,
                    "information_gain": 0.05,
                    "value": 0.03,
                    "valuable_coverage_proxy": 0.10,
                    "path_cost": 0.45,
                    "risk": 0.19,
                    "energy_cost": 0.0,
                    "source_path": "fixtures/source.json",
                    "match_method": "candidate_counterfactual_path_feedback",
                    "source_confidence": 1.0,
                },
            ],
        )

        summary = run_policy_coverage_opportunity_margin_audit(
            coverage_driven_root=self.coverage_driven_root,
            reward_refinement_root=self.reward_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            candidate_coverage_overlay_path=overlay_path,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["next_required_change"], "refine_coverage_reward_or_advantage")
        self.assertEqual(summary["safe_better_than_teacher_count"], 1)
        self.assertEqual(summary["candidate_coverage_overlay_connected_count"], 2)
        self.assertEqual(summary["candidate_expected_coverage_nonzero_count"], 2)
        self.assertEqual(summary["candidate_information_gain_nonzero_count"], 1)
        self.assertEqual(summary["candidate_value_nonzero_count"], 1)

        action_rows = self._read_jsonl(self.output_root / "policy-coverage-action-level-audit.jsonl")
        better_rows = [row for row in action_rows if row["is_best_safe_non_teacher_alternative"]]
        self.assertEqual(len(better_rows), 1)
        self.assertEqual(better_rows[0]["action_index"], 1)
        self.assertTrue(better_rows[0]["candidate_coverage_overlay_connected"])
        self.assertEqual(
            better_rows[0]["candidate_coverage_match_method"],
            "candidate_counterfactual_path_feedback",
        )

    def _write_inputs(self, transitions: list[dict], *, replay_rows: list[dict]) -> None:
        self._write_json(
            self.coverage_driven_root / "coverage-driven-ppo-improvement-run-summary.json",
            {
                "schema_version": "coverage-driven-ppo-improvement-run-summary/v1",
                "status": "failed",
                "reason_codes": ["no_coverage_return_improvement"],
                "coverage_driven_ppo_improvement_status": "failed",
                "coverage_aware_batch_episodes": str(
                    self.coverage_driven_root / "coverage-aware-ppo-batch" / "ppo-rollout-episodes.jsonl"
                ),
                "replay_audit": str(self.coverage_driven_root / "coverage-driven-ppo-replay-audit.json"),
                "ppo_update_summary": str(self.coverage_driven_root / "coverage-driven-ppo-update-summary.json"),
                "coverage_return_improvement": 0.0,
                "cumulative_coverage_rate_delta_improvement": 0.0,
                "valuable_area_covered_improvement": 0.0,
                "fallback_rate": 0.0,
                "teacher_agreement_rate": 1.0,
                "controlled_regression_count": 0,
                "runs_new_ppo_update": True,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "formal_release_claimed": False,
            },
        )
        self._write_jsonl(
            self.coverage_driven_root / "coverage-aware-ppo-batch" / "ppo-rollout-episodes.jsonl",
            [
                {
                    "schema_version": "coverage-aware-ppo-rollout-episode/v1",
                    "episode_id": "episode-a",
                    "transitions": transitions,
                }
            ],
        )
        self._write_json(
            self.coverage_driven_root / "coverage-driven-ppo-replay-audit.json",
            {
                "schema_version": "coverage-driven-ppo-replay-audit/v1",
                "status": "passed",
                "reason_codes": [],
                "rows": replay_rows,
                "guard_rejected_action_count": 0,
                "accepted_policy_action_count": len(replay_rows),
            },
        )
        self._write_json(
            self.coverage_driven_root / "coverage-driven-ppo-update-summary.json",
            {
                "schema_version": "limited-ppo-update-smoke-summary/v1",
                "status": "passed",
                "optimizer_train_transition_count": len(transitions),
                "runs_formal_ppo_rollout": False,
            },
        )
        self._write_json(
            self.reward_root / "coverage-aware-reward-refinement-summary.json",
            {
                "schema_version": "coverage-aware-reward-refinement-summary/v1",
                "status": "passed",
                "reward_refinement_status": "passed",
                "source_field_missing_component_count": 0,
                "fallback_coverage_gain_claimed_as_policy_gain_count": 0,
                "controlled_regression_count": 0,
                "performance_claimed": False,
            },
        )
        self._write_json(
            self.signal_root / "exploration-coverage-signal-audit-summary.json",
            {
                "schema_version": "exploration-coverage-signal-audit-summary/v1",
                "status": "passed",
                "coverage_signal_status": "passed",
                "fallback_coverage_gain_claimed_as_policy_gain_count": 0,
                "controlled_regression_count": 0,
                "nonzero_actual_coverage_delta_count": len(transitions),
                "performance_claimed": False,
            },
        )
        self._write_json(
            self.performance_root / "exploration-coverage-performance-evaluation-summary.json",
            {
                "schema_version": "exploration-coverage-performance-evaluation-summary/v1",
                "status": "failed",
                "coverage_performance_status": "failed",
                "reason_codes": ["coverage_performance_not_improved"],
                "coverage_return_improvement": 0.0,
                "valuable_area_covered_improvement": 0.0,
                "performance_claimed": False,
            },
        )

    def _transition(
        self,
        *,
        context_id: str,
        teacher_action_index: int,
        features: list[list[float]],
        pre_logits: list[float],
        post_logits: list[float],
    ) -> dict:
        return {
            "action_index": teacher_action_index,
            "controlled_action_index": teacher_action_index,
            "teacher_action_index": teacher_action_index,
            "pre_policy_logits": pre_logits,
            "post_policy_logits": post_logits,
            "info": {
                "context_id": context_id,
                "episode_id": "episode-a",
                "step_index": 0,
                "scenario_id": "scenario-a",
                "scenario_family": "family-a",
                "controlled_action_index": teacher_action_index,
                "teacher_action_index": teacher_action_index,
                "coverage_rate_delta": 0.05,
                "valuable_area_covered": 0.02,
                "path_cost": features[teacher_action_index][3],
                "risk": features[teacher_action_index][2],
                "energy_cost": features[teacher_action_index][4],
                "actual_coverage_gain_source": "path_feedback",
                "fallback_like": False,
                "controlled_regression_reason_codes": [],
                "split": "train",
            },
            "observation": {
                "candidate_feature_names": [
                    "expected_coverage_rate_delta",
                    "information_gain",
                    "risk",
                    "path_cost",
                    "energy_cost",
                    "value",
                    "utility",
                    "reachable",
                ],
                "candidate_features": features,
                "global_feature_names": ["coverage_rate"],
                "global_features": [0.0],
                "action_mask": [True for _ in features],
                "candidate_cells": [[index, index] for index, _ in enumerate(features)],
                "candidate_missing_feature_names": [[] for _ in features],
                "candidate_missing_indicator_names": [],
                "candidate_missing_indicators": [[] for _ in features],
            },
        }

    def _candidate(
        self,
        *,
        expected: float,
        information: float,
        value: float,
        risk: float,
        path_cost: float,
    ) -> list[float]:
        return [expected, information, risk, path_cost, 0.0, value, 0.5, 1.0]

    def _replay_row(self, context_id: str, *, teacher: int, raw: int, controlled: int) -> dict:
        return {
            "schema_version": "coverage-driven-ppo-replay-row/v1",
            "context_id": context_id,
            "episode_id": "episode-a",
            "step_index": 0,
            "scenario_id": "scenario-a",
            "scenario_family": "family-a",
            "teacher_action_index": teacher,
            "raw_policy_action_index": raw,
            "controlled_action_index": controlled,
            "policy_action_accepted": True,
            "guard_rejected": False,
            "controlled_regression_reason_codes": [],
            "fallback_like": False,
        }

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
