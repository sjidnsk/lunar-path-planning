import sys
import tempfile
import unittest
from pathlib import Path


class QuasiRealIterativePpoMiniLoopStabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts = self.repo_root / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        self.temp_dir = Path(tempfile.mkdtemp(prefix="quasi-real-iterative-ppo-"))

    def test_round_plan_chains_experimental_candidate_between_rounds(self) -> None:
        from scripts.run_quasi_real_iterative_ppo_mini_loop_stability import (
            build_quasi_real_round_plan,
        )

        plan = build_quasi_real_round_plan(
            output_root=self.temp_dir / "loop",
            initial_candidate_root=Path("outputs/base-candidate"),
            round_count=3,
        )

        self.assertEqual(plan[0]["base_candidate_root"], Path("outputs/base-candidate"))
        self.assertEqual(plan[1]["base_candidate_root"], plan[0]["update_root"])
        self.assertEqual(plan[2]["base_candidate_root"], plan[1]["update_root"])
        self.assertEqual(plan[0]["teacher_following_root"], self.temp_dir / "loop" / "round-00" / "teacher-following")
        self.assertEqual(plan[2]["long_horizon_root"], self.temp_dir / "loop" / "round-02" / "long-horizon")

    def test_summary_passes_when_strict_generated_wrapper_failure_is_overridden_by_long_horizon(self) -> None:
        from scripts.run_quasi_real_iterative_ppo_mini_loop_stability import (
            summarize_quasi_real_iterative_rounds,
        )

        summary, drift_report, rejection_report = summarize_quasi_real_iterative_rounds(
            round_records=[self._round_record(index) for index in range(3)],
            output_root=self.temp_dir / "loop",
            config=self._config(),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["schema_version"], "iterative-ppo-mini-loop-stability-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["stability_passed"])
        self.assertEqual(summary["round_count"], 3)
        self.assertEqual(summary["failed_round_count"], 0)
        self.assertEqual(summary["min_optimizer_train_transition_count"], 36)
        self.assertEqual(summary["min_ppo_trainable_transition_count"], 36)
        self.assertAlmostEqual(summary["cumulative_parameter_l2_delta"], 0.006)
        self.assertEqual(summary["long_horizon_controlled_regression_count"], 0)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["formal_training_ready_claimed"])
        self.assertEqual(drift_report["records"][2]["round_index"], 2)
        self.assertEqual(rejection_report["records"], [])

    def test_summary_fails_when_validation_or_test_transition_enters_optimizer(self) -> None:
        from scripts.run_quasi_real_iterative_ppo_mini_loop_stability import (
            summarize_quasi_real_iterative_rounds,
        )

        record = self._round_record(0)
        record["update_summary"]["validation_test_optimizer_transition_count"] = 1

        summary, _, rejection_report = summarize_quasi_real_iterative_rounds(
            round_records=[record, self._round_record(1), self._round_record(2)],
            output_root=self.temp_dir / "loop",
            config=self._config(),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("iterative_ppo_trainable_transition_count_insufficient", summary["reason_codes"])
        self.assertEqual(rejection_report["records"][0]["round_index"], 0)

    def test_summary_fails_when_old_policy_reconstruction_does_not_match_collector(self) -> None:
        from scripts.run_quasi_real_iterative_ppo_mini_loop_stability import (
            summarize_quasi_real_iterative_rounds,
        )

        record = self._round_record(1)
        record["update_summary"]["old_value_max_abs_error"] = 0.01

        summary, _, _ = summarize_quasi_real_iterative_rounds(
            round_records=[self._round_record(0), record, self._round_record(2)],
            output_root=self.temp_dir / "loop",
            config=self._config(),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("iterative_ppo_on_policy_contract_invalid", summary["reason_codes"])

    def test_summary_fails_when_long_horizon_reports_controlled_regression(self) -> None:
        from scripts.run_quasi_real_iterative_ppo_mini_loop_stability import (
            summarize_quasi_real_iterative_rounds,
        )

        record = self._round_record(2)
        record["long_horizon_summary"]["controlled_regression_episode_count"] = 1

        summary, _, _ = summarize_quasi_real_iterative_rounds(
            round_records=[self._round_record(0), self._round_record(1), record],
            output_root=self.temp_dir / "loop",
            config=self._config(),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("iterative_ppo_post_update_gate_regression", summary["reason_codes"])

    def test_summary_fails_when_accounting_audit_does_not_pass(self) -> None:
        from scripts.run_quasi_real_iterative_ppo_mini_loop_stability import (
            summarize_quasi_real_iterative_rounds,
        )

        record = self._round_record(0)
        record["accounting_summary"]["status"] = "failed"
        record["accounting_summary"]["reason_codes"] = ["legacy_mismatch_rows_missing"]

        summary, _, _ = summarize_quasi_real_iterative_rounds(
            round_records=[record, self._round_record(1), self._round_record(2)],
            output_root=self.temp_dir / "loop",
            config=self._config(),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("iterative_ppo_post_update_gate_regression", summary["reason_codes"])

    def test_readiness_accepts_quasi_real_iterative_summary_via_existing_status(self) -> None:
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
                "min_optimizer_train_transition_count": 36,
                "min_ppo_trainable_transition_count": 36,
                "max_abs_approx_kl": 0.01,
                "cumulative_parameter_l2_delta": 0.006,
                "raw_test_regression_count": 0,
                "sequential_rejected_count": 0,
                "collector_regression_count": 0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "formal_training_ready_claimed": False,
                "git_provenance": {"current_matches_sources": True},
            }
        )

        self.assertTrue(readiness["present"])
        self.assertTrue(readiness["completed"])
        self.assertEqual(readiness["training_blockers"], [])

    def _round_record(self, index: int) -> dict:
        return {
            "round_index": index,
            "base_candidate_root": "initial" if index == 0 else f"round-{index - 1:02d}/update",
            "update_root": f"round-{index:02d}/update",
            "teacher_following_summary": {
                "schema_version": "quasi-real-guarded-teacher-following-pilot-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "teacher_agreement_rate": 1.0,
                "unsafe_disagreement_count": 0,
                "policy_changed_gate_rejected_count": 0,
            },
            "collector_summary": {
                "schema_version": "ppo-rollout-collector-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "ppo_trainable_transition_count": 36,
                "diagnostic_transition_count": 72,
                "source_fallback_trainable_count": 0,
                "invalid_action_mask_count": 0,
                "empty_action_mask_count": 0,
                "missing_log_prob_count": 0,
                "missing_value_count": 0,
                "non_finite_reward_count": 0,
                "fallback_or_open_grid_count": 0,
                "safety_regression_count": 0,
                "contract_violation_count": 0,
                "path_cost_regression_count": 0,
                "risk_regression_count": 0,
                "source_selection_regression_count": 0,
            },
            "update_summary": {
                "schema_version": "limited-quasi-real-ppo-update-smoke-summary/v1",
                "status": "failed",
                "reason_codes": ["limited_quasi_real_ppo_update_post_update_gate_regression"],
                "input_ppo_trainable_transition_count": 36,
                "optimizer_train_transition_count": 36,
                "source_fallback_trainable_count": 0,
                "validation_test_optimizer_transition_count": 0,
                "non_empty_gate_reason_optimizer_transition_count": 0,
                "disallowed_source_optimizer_transition_count": 0,
                "old_log_prob_max_abs_error": 0.0,
                "old_value_max_abs_error": 0.0,
                "loss_non_finite_count": 0,
                "non_finite_gradient_count": 0,
                "non_finite_reward_count": 0,
                "non_finite_return_count": 0,
                "non_finite_advantage_count": 0,
                "parameter_l2_delta": 0.002,
                "approx_kl": 0.01,
                "max_grad_norm_after_clip": 1.0,
                "post_update_quasi_real_teacher_following_status": "passed",
                "post_update_quasi_real_teacher_agreement_rate": 1.0,
                "post_update_quasi_real_unsafe_disagreement_count": 0,
                "post_update_quasi_real_collector_status": "passed",
                "post_update_quasi_real_collector_trainable_transition_count": 36,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "formal_training_ready_claimed": False,
            },
            "post_update_quasi_real_collector_summary": {
                "schema_version": "ppo-rollout-collector-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "ppo_trainable_transition_count": 36,
                "diagnostic_transition_count": 72,
                "source_fallback_trainable_count": 0,
                "invalid_action_mask_count": 0,
                "empty_action_mask_count": 0,
                "missing_log_prob_count": 0,
                "missing_value_count": 0,
                "non_finite_reward_count": 0,
                "fallback_or_open_grid_count": 0,
                "safety_regression_count": 0,
                "contract_violation_count": 0,
                "path_cost_regression_count": 0,
                "risk_regression_count": 0,
                "source_selection_regression_count": 0,
            },
            "compatibility_summary": {
                "schema_version": "quasi-real-generated-sequential-contract-compatibility-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "diagnosis_verdict": "pre_existing_generated_sequential_contract_mismatch",
                "failed_step_count": 6,
            },
            "accounting_summary": {
                "schema_version": "generated-sequential-gate-metric-accounting-audit-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "diagnosis_verdict_after_origin_split": "pre_existing_generated_sequential_contract_mismatch",
                "controlled_path_cost_regression_count": 0,
                "controlled_risk_regression_count": 0,
            },
            "long_horizon_summary": {
                "schema_version": "generated-sequential-long-horizon-teacher-skill-contract-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "verdict": "long_horizon_teacher_skill_contract_aligned",
                "teacher_equivalent_episode_count": 36,
                "beyond_teacher_episode_count": 15,
                "dominated_raw_choice_count": 6,
                "controlled_regression_episode_count": 0,
            },
        }

    def _config(self) -> dict:
        return {
            "schema_version": "quasi-real-iterative-ppo-mini-loop-stability-config/v1",
            "output_files": {
                "summary": "quasi-real-iterative-ppo-mini-loop-stability-summary.json",
                "rounds": "quasi-real-iterative-ppo-mini-loop-rounds.jsonl",
                "drift_report": "quasi-real-iterative-ppo-mini-loop-drift-report.json",
                "rejection_report": "quasi-real-iterative-ppo-mini-loop-rejection-report.json",
            },
            "validation": {
                "round_count": 3,
                "min_optimizer_train_transition_count": 24,
                "min_ppo_trainable_transition_count": 24,
                "max_old_log_prob_abs_error": 1.0e-4,
                "max_old_value_abs_error": 1.0e-4,
                "max_abs_approx_kl": 0.25,
                "max_grad_norm_after_clip": 1.0,
                "max_cumulative_parameter_l2_delta": 0.05,
                "min_teacher_agreement_rate": 0.9,
            },
        }


if __name__ == "__main__":
    unittest.main()
