import tempfile
import unittest
import sys
from pathlib import Path


class GuardedPpoRolloutPilotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts = self.repo_root / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        self.temp_dir = Path(tempfile.mkdtemp(prefix="guarded-ppo-rollout-pilot-"))

    def test_build_pilot_plan_uses_base_then_update_for_post_gates(self) -> None:
        from scripts.run_guarded_ppo_rollout_pilot import build_guarded_pilot_plan

        plan = build_guarded_pilot_plan(
            output_root=self.temp_dir / "pilot",
            base_candidate_root=Path("outputs/base-candidate"),
        )

        self.assertEqual(plan["base_candidate_root"], Path("outputs/base-candidate"))
        self.assertEqual(plan["sequential_root"], self.temp_dir / "pilot" / "pilot" / "sequential")
        self.assertEqual(plan["collector_root"], self.temp_dir / "pilot" / "pilot" / "collector")
        self.assertEqual(plan["update_root"], self.temp_dir / "pilot" / "update")
        self.assertEqual(plan["post_sequential_root"], self.temp_dir / "pilot" / "final" / "sequential")
        self.assertEqual(plan["post_collector_root"], self.temp_dir / "pilot" / "final" / "collector")

    def test_summary_passes_for_guarded_rollout_and_post_update_gates(self) -> None:
        from scripts.run_guarded_ppo_rollout_pilot import summarize_guarded_ppo_rollout_pilot

        summary, rejection_report = summarize_guarded_ppo_rollout_pilot(
            output_root=self.temp_dir / "pilot",
            config=self._config(),
            repo_root=self.repo_root,
            base_candidate_root=Path("outputs/base"),
            update_root=Path("outputs/update"),
            pilot_sequential_summary=self._sequential_summary(),
            pilot_collector_summary=self._collector_summary(),
            update_summary=self._update_summary(),
            raw_generalization_summary=self._raw_summary(),
            post_update_sequential_summary=self._sequential_summary(),
            post_update_collector_summary=self._collector_summary(),
        )

        self.assertEqual(summary["schema_version"], "guarded-ppo-rollout-pilot-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["ppo_trainable_transition_count"], 37)
        self.assertEqual(summary["optimizer_train_transition_count"], 37)
        self.assertEqual(summary["post_update_raw_test_regression_count"], 0)
        self.assertTrue(summary["guarded_rollout_pilot_passed"])
        self.assertEqual(rejection_report["records"], [])

    def test_summary_fails_when_guarded_rollout_rejects_policy_choice(self) -> None:
        from scripts.run_guarded_ppo_rollout_pilot import summarize_guarded_ppo_rollout_pilot

        sequential = self._sequential_summary()
        sequential["canary_rejected_policy_choice_count"] = 1

        summary, rejection_report = summarize_guarded_ppo_rollout_pilot(
            output_root=self.temp_dir / "pilot",
            config=self._config(),
            repo_root=self.repo_root,
            base_candidate_root=Path("outputs/base"),
            update_root=Path("outputs/update"),
            pilot_sequential_summary=sequential,
            pilot_collector_summary=self._collector_summary(),
            update_summary=self._update_summary(),
            raw_generalization_summary=self._raw_summary(),
            post_update_sequential_summary=self._sequential_summary(),
            post_update_collector_summary=self._collector_summary(),
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("guarded_ppo_rollout_gate_regression", summary["reason_codes"])
        self.assertEqual(rejection_report["records"][0]["stage"], "pilot_sequential")

    def test_readiness_accepts_passed_guarded_pilot_summary(self) -> None:
        from scripts.run_policy_training_readiness_review import _guarded_ppo_rollout_pilot_readiness

        readiness = _guarded_ppo_rollout_pilot_readiness(
            {
                "schema_version": "guarded-ppo-rollout-pilot-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "guarded_rollout_pilot_passed": True,
                "episode_count": 36,
                "step_count": 108,
                "ppo_trainable_transition_count": 37,
                "optimizer_train_transition_count": 37,
                "source_fallback_trainable_count": 0,
                "state_continuity_violation_count": 0,
                "invalid_action_mask_count": 0,
                "empty_action_mask_count": 0,
                "missing_log_prob_count": 0,
                "missing_value_count": 0,
                "non_finite_reward_count": 0,
                "post_update_raw_test_regression_count": 0,
                "post_update_sequential_rejected_count": 0,
                "post_update_collector_regression_count": 0,
                "old_log_prob_max_abs_error": 0.0,
                "old_value_max_abs_error": 0.0,
                "parameter_l2_delta": 0.001,
                "approx_kl": 0.01,
                "max_grad_norm_after_clip": 1.0,
                "experimental_checkpoint": True,
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
        self.assertEqual(readiness["ppo_trainable_transition_count"], 37)

    def _config(self) -> dict:
        return {
            "schema_version": "guarded-ppo-rollout-pilot-config/v1",
            "validation": {
                "min_episode_count": 36,
                "min_step_count": 108,
                "min_ppo_trainable_transition_count": 24,
                "min_optimizer_train_transition_count": 24,
                "max_old_log_prob_abs_error": 1.0e-4,
                "max_old_value_abs_error": 1.0e-4,
                "max_abs_approx_kl": 0.25,
                "max_grad_norm_after_clip": 1.0,
                "max_raw_test_regression_count": 0,
                "min_accepted_takeover_family_count": 6,
                "min_multi_step_accepted_episode_count": 12,
            },
            "output_files": {
                "summary": "guarded-ppo-rollout-pilot-summary.json",
                "rejection_report": "guarded-ppo-rollout-rejection-report.json",
            },
            "non_goals": [],
        }

    def _sequential_summary(self) -> dict:
        return {
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
            "invalid_action_mask_count": 0,
            "fallback_or_open_grid_count": 0,
            "cumulative_safety_regression_count": 0,
            "cumulative_contract_violation_count": 0,
            "cumulative_path_cost_regression_count": 0,
            "cumulative_risk_regression_count": 0,
            "cumulative_source_selection_regression_count": 0,
        }

    def _collector_summary(self) -> dict:
        return {
            "schema_version": "ppo-rollout-collector-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "episode_count": 36,
            "step_count": 108,
            "ppo_trainable_transition_count": 37,
            "source_fallback_trainable_count": 0,
            "invalid_action_mask_count": 0,
            "empty_action_mask_count": 0,
            "missing_log_prob_count": 0,
            "missing_value_count": 0,
            "non_finite_reward_count": 0,
            "state_continuity_violation_count": 0,
            "fallback_or_open_grid_count": 0,
            "safety_regression_count": 0,
            "contract_violation_count": 0,
            "path_cost_regression_count": 0,
            "risk_regression_count": 0,
            "source_selection_regression_count": 0,
        }

    def _update_summary(self) -> dict:
        return {
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
            "parameter_l2_delta": 0.001,
            "approx_kl": 0.01,
            "max_grad_norm_after_clip": 1.0,
            "experimental_checkpoint": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
        }

    def _raw_summary(self) -> dict:
        return {
            "schema_version": "raw-policy-generalization-evaluation-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "test_raw_policy_regression_count": 0,
        }


if __name__ == "__main__":
    unittest.main()
