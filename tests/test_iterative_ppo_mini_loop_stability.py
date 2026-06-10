import json
import sys
import tempfile
import unittest
from pathlib import Path


class IterativePpoMiniLoopStabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts = self.repo_root / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        self.temp_dir = Path(tempfile.mkdtemp(prefix="iterative-ppo-mini-loop-"))

    def test_round_plan_chains_update_candidate_between_rounds(self) -> None:
        from scripts.run_iterative_ppo_mini_loop_stability import build_round_plan

        plan = build_round_plan(
            output_root=self.temp_dir / "iterative",
            initial_candidate_root=Path("outputs/start-candidate"),
            round_count=3,
        )

        self.assertEqual(plan[0]["base_candidate_root"], Path("outputs/start-candidate"))
        self.assertEqual(plan[1]["base_candidate_root"], plan[0]["update_root"])
        self.assertEqual(plan[2]["base_candidate_root"], plan[1]["update_root"])
        self.assertEqual(plan[2]["round_index"], 2)
        self.assertEqual(plan[2]["post_collector_root"], self.temp_dir / "iterative" / "round-02" / "post-collector")

    def test_summary_passes_for_three_stable_rounds(self) -> None:
        from scripts.run_iterative_ppo_mini_loop_stability import summarize_iterative_rounds

        summary, drift_report, rejection_report = summarize_iterative_rounds(
            round_records=[self._round_record(index) for index in range(3)],
            output_root=self.temp_dir / "iterative",
            config=self._config(),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["stability_passed"])
        self.assertEqual(summary["round_count"], 3)
        self.assertEqual(summary["failed_round_count"], 0)
        self.assertEqual(summary["max_abs_approx_kl"], 0.01)
        self.assertAlmostEqual(summary["cumulative_parameter_l2_delta"], 0.006)
        self.assertEqual(drift_report["max_abs_approx_kl"], 0.01)
        self.assertEqual(rejection_report["records"], [])

    def test_summary_fails_when_update_is_not_on_collector_policy(self) -> None:
        from scripts.run_iterative_ppo_mini_loop_stability import summarize_iterative_rounds

        record = self._round_record(0)
        record["update_summary"]["old_log_prob_max_abs_error"] = 0.01

        summary, _, rejection_report = summarize_iterative_rounds(
            round_records=[record, self._round_record(1), self._round_record(2)],
            output_root=self.temp_dir / "iterative",
            config=self._config(),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("iterative_ppo_on_policy_contract_invalid", summary["reason_codes"])
        self.assertEqual(summary["failed_round_count"], 1)
        self.assertEqual(rejection_report["records"][0]["round_index"], 0)

    def test_summary_fails_when_post_update_sequential_gate_regresses(self) -> None:
        from scripts.run_iterative_ppo_mini_loop_stability import summarize_iterative_rounds

        record = self._round_record(1)
        record["post_update_sequential_summary"]["canary_rejected_policy_choice_count"] = 1

        summary, _, _ = summarize_iterative_rounds(
            round_records=[self._round_record(0), record, self._round_record(2)],
            output_root=self.temp_dir / "iterative",
            config=self._config(),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("iterative_ppo_post_update_gate_regression", summary["reason_codes"])

    def test_readiness_accepts_passed_iterative_summary(self) -> None:
        from scripts.run_policy_training_readiness_review import (
            _iterative_ppo_mini_loop_stability_readiness,
        )

        readiness = _iterative_ppo_mini_loop_stability_readiness(
            {
                "schema_version": "iterative-ppo-mini-loop-stability-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "round_count": 3,
                "failed_round_count": 0,
                "stability_passed": True,
                "max_abs_approx_kl": 0.01,
                "cumulative_parameter_l2_delta": 0.006,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "git_provenance": {"current_matches_sources": True},
            }
        )

        self.assertTrue(readiness["present"])
        self.assertTrue(readiness["completed"])
        self.assertEqual(readiness["training_blockers"], [])
        self.assertEqual(readiness["round_count"], 3)

    def _round_record(self, index: int) -> dict:
        return {
            "round_index": index,
            "base_candidate_root": f"round-{index - 1:02d}/update" if index else "initial",
            "update_root": f"round-{index:02d}/update",
            "collector_summary": {
                "schema_version": "ppo-rollout-collector-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "ppo_trainable_transition_count": 37,
                "invalid_action_mask_count": 0,
                "empty_action_mask_count": 0,
                "missing_log_prob_count": 0,
                "missing_value_count": 0,
                "non_finite_reward_count": 0,
                "state_continuity_violation_count": 0,
                "path_cost_regression_count": 0,
                "risk_regression_count": 0,
                "source_selection_regression_count": 0,
            },
            "update_summary": {
                "schema_version": "limited-ppo-update-smoke-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "input_ppo_trainable_transition_count": 37,
                "optimizer_train_transition_count": 37,
                "source_fallback_trainable_count": 0,
                "old_log_prob_max_abs_error": 0.0,
                "old_value_max_abs_error": 0.0,
                "loss_non_finite_count": 0,
                "non_finite_gradient_count": 0,
                "non_finite_reward_count": 0,
                "non_finite_return_count": 0,
                "non_finite_advantage_count": 0,
                "parameter_l2_delta": 0.002,
                "approx_kl": (-0.01 if index == 1 else 0.01),
                "max_grad_norm_after_clip": 1.0,
            },
            "raw_generalization_summary": {
                "schema_version": "raw-policy-generalization-evaluation-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "test_raw_policy_regression_count": 0,
                "overfit_gap": 0.0,
            },
            "post_update_sequential_summary": {
                "schema_version": "policy-gated-sequential-canary-rollout-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "episode_count": 36,
                "step_count": 108,
                "accepted_takeover_family_count": 6,
                "multi_step_accepted_episode_count": 12,
                "canary_rejected_policy_choice_count": 0,
                "state_continuity_violation_count": 0,
                "episode_fallback_count": 0,
                "cumulative_path_cost_regression_count": 0,
                "cumulative_risk_regression_count": 0,
                "cumulative_source_selection_regression_count": 0,
            },
            "post_update_collector_summary": {
                "schema_version": "ppo-rollout-collector-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "ppo_trainable_transition_count": 37,
                "invalid_action_mask_count": 0,
                "empty_action_mask_count": 0,
                "missing_log_prob_count": 0,
                "missing_value_count": 0,
                "non_finite_reward_count": 0,
                "state_continuity_violation_count": 0,
                "path_cost_regression_count": 0,
                "risk_regression_count": 0,
                "source_selection_regression_count": 0,
            },
        }

    def _config(self) -> dict:
        return {
            "schema_version": "iterative-ppo-mini-loop-stability-config/v1",
            "validation": {
                "round_count": 3,
                "min_optimizer_train_transition_count": 24,
                "min_ppo_trainable_transition_count": 24,
                "max_old_log_prob_abs_error": 1.0e-4,
                "max_old_value_abs_error": 1.0e-4,
                "max_abs_approx_kl": 0.25,
                "max_grad_norm_after_clip": 1.0,
                "max_cumulative_parameter_l2_delta": 0.05,
                "max_raw_test_regression_count": 0,
                "min_sequential_episode_count": 36,
                "min_sequential_step_count": 108,
                "min_accepted_takeover_family_count": 6,
                "min_multi_step_accepted_episode_count": 12,
                "max_canary_rejected_policy_choice_count": 0,
            },
            "output_files": {
                "summary": "iterative-ppo-mini-loop-stability-summary.json",
                "rounds": "iterative-ppo-mini-loop-rounds.jsonl",
                "drift_report": "iterative-ppo-mini-loop-drift-report.json",
                "rejection_report": "iterative-ppo-mini-loop-rejection-report.json",
            },
        }


if __name__ == "__main__":
    unittest.main()
