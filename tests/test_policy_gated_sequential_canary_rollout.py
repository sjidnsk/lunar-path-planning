import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class PolicyGatedSequentialCanaryRolloutTests(unittest.TestCase):
    def test_summary_fails_when_next_step_does_not_start_from_previous_controlled_goal(self) -> None:
        from scripts.run_policy_gated_sequential_canary_rollout import summarize_sequential_steps

        steps = [
            self._step("ep-1", 0, "mixed_stress_detour", [1, 6], [4, 6], "policy"),
            self._step("ep-1", 1, "mixed_stress_detour", [5, 6], [8, 6], "policy"),
        ]

        summary, _ = summarize_sequential_steps(steps, config=self._config())

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["state_continuity_violation_count"], 1)
        self.assertIn("state_continuity_violation", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "sequential_canary_state_continuity_required")

    def test_rejected_policy_choice_falls_back_to_source_goal_for_next_step(self) -> None:
        from scripts.run_policy_gated_sequential_canary_rollout import build_sequential_step_record

        decision = {
            "episode_id": "ep-1",
            "step_index": 0,
            "scenario_group": "dense_choke_safe_bypass",
            "decision_class": "canary_rejected_policy_choice",
            "accepted_choice_value_class": None,
            "source_selected_execution_goal_cell": [3, 6],
            "policy_selected_execution_goal_cell": [4, 6],
            "canary_rejection_reason_codes": ["risk_regression"],
            "controlled_regression_reason_codes": [],
            "action_mask_valid": True,
        }

        step = build_sequential_step_record(
            decision,
            episode_id="ep-1",
            step_index=0,
            input_start_cell=[1, 6],
        )

        self.assertEqual(step["controlled_choice_source"], "source_fallback")
        self.assertEqual(step["controlled_execution_goal_cell"], [3, 6])
        self.assertEqual(step["source_execution_goal_cell"], [3, 6])
        self.assertEqual(step["policy_execution_goal_cell"], [4, 6])

    def test_summary_passes_for_multi_family_continuous_safe_takeover(self) -> None:
        from scripts.run_policy_gated_sequential_canary_rollout import summarize_sequential_steps

        steps = []
        for family_index, family in enumerate(
            (
                "mixed_stress_detour",
                "near_blocked_safe_alt",
                "high_risk_tradeoff",
                "dense_choke_safe_bypass",
                "channel_contrast",
                "path_complexity_benefit",
            )
        ):
            x = family_index * 3
            steps.extend(
                [
                    self._step(f"ep-{family_index}", 0, family, [x, 1], [x + 1, 1], "policy"),
                    self._step(f"ep-{family_index}", 1, family, [x + 1, 1], [x + 2, 1], "policy"),
                    self._step(f"ep-{family_index}", 2, family, [x + 2, 1], [x + 3, 1], "policy"),
                ]
            )

        summary, _ = summarize_sequential_steps(
            steps,
            config=self._config(
                min_episode_count=6,
                min_step_count=18,
                min_completed_episode_count=6,
                min_policy_takeover_step_count=18,
                min_accepted_takeover_step_count=18,
                min_accepted_better_step_count=18,
                min_multi_step_accepted_episode_count=6,
            ),
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["episode_count"], 6)
        self.assertEqual(summary["step_count"], 18)
        self.assertEqual(summary["state_continuity_violation_count"], 0)
        self.assertEqual(summary["family_with_multi_step_accepted_episode_count"], 6)
        self.assertEqual(summary["accepted_takeover_family_count"], 6)

    def test_summary_counts_each_regression_reason_once_per_step(self) -> None:
        from scripts.run_policy_gated_sequential_canary_rollout import summarize_sequential_steps

        step = self._step("ep-1", 0, "channel_contrast", [1, 6], [2, 6], "source_fallback")
        step["decision_class"] = "canary_rejected_policy_choice"
        step["canary_rejection_reason_codes"] = ["path_cost_regression", "risk_regression"]
        step["raw_policy_regression_reason_codes"] = ["path_cost_regression", "risk_regression"]

        summary, _ = summarize_sequential_steps(
            [step],
            config=self._config(
                min_policy_takeover_step_count=0,
                min_accepted_takeover_step_count=0,
                min_accepted_better_step_count=0,
                min_accepted_takeover_family_count=0,
                min_multi_step_accepted_episode_count=0,
                min_family_with_multi_step_accepted_episode_count=0,
                max_canary_rejected_policy_choice_count=1,
                max_cumulative_path_cost_regression_count=1,
                max_cumulative_risk_regression_count=1,
            ),
        )

        self.assertEqual(summary["cumulative_path_cost_regression_count"], 1)
        self.assertEqual(summary["cumulative_risk_regression_count"], 1)

    def test_readiness_advances_after_sequential_canary_passes(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        root = Path(tempfile.mkdtemp(prefix="sequential-readiness-"))
        source_root = root / "source"
        candidate_root = root / "candidate"
        sequential_root = root / "sequential"
        source_root.mkdir()
        candidate_root.mkdir()
        sequential_root.mkdir()
        self._write_minimal_source_root(source_root)
        self._write_raw_generalization_summary(candidate_root)
        git_provenance = {"current": self._current_git_snapshot(repo_root), "current_matches_sources": True}
        (sequential_root / "policy-gated-sequential-canary-rollout-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "policy-gated-sequential-canary-rollout-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "episode_count": 36,
                    "step_count": 108,
                    "completed_episode_count": 36,
                    "policy_takeover_step_count": 48,
                    "accepted_takeover_step_count": 48,
                    "accepted_better_step_count": 48,
                    "multi_step_accepted_episode_count": 18,
                    "family_with_multi_step_accepted_episode_count": 6,
                    "accepted_takeover_family_count": 6,
                    "source_fallback_step_count": 0,
                    "state_continuity_violation_count": 0,
                    "episode_fallback_count": 0,
                    "canary_rejected_policy_choice_count": 0,
                    "cumulative_safety_regression_count": 0,
                    "cumulative_contract_violation_count": 0,
                    "cumulative_path_cost_regression_count": 0,
                    "cumulative_risk_regression_count": 0,
                    "cumulative_source_selection_regression_count": 0,
                    "invalid_action_mask_count": 0,
                    "fallback_or_open_grid_count": 0,
                    "candidate_git_current_matches_sources": True,
                    "checkpoint_metadata_git_current_matches_sources": True,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                    "git_provenance": git_provenance,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                "bash",
                str(repo_root / "scripts" / "run_policy_training_readiness_review.sh"),
                "--batch-root",
                str(source_root),
                "--config",
                str(repo_root / "configs" / "policy_training_readiness_review_v1.json"),
                "--raw-policy-generalization-evaluation-summary",
                str(candidate_root / "raw-policy-generalization-evaluation-summary.json"),
                "--policy-gated-sequential-canary-rollout-summary",
                str(sequential_root / "policy-gated-sequential-canary-rollout-summary.json"),
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads((source_root / "policy-training-readiness-review-summary.json").read_text())
        self.assertEqual(summary["training_readiness_status"], "policy_gated_sequential_canary_rollout_evaluated")
        self.assertEqual(summary["training_blockers"], [])

    def test_readiness_advances_after_sequential_safe_choice_calibration_passes(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        root = Path(tempfile.mkdtemp(prefix="sequential-safe-choice-readiness-"))
        source_root = root / "source"
        candidate_root = root / "candidate"
        sequential_root = root / "sequential_safe_choice"
        source_root.mkdir()
        candidate_root.mkdir()
        sequential_root.mkdir()
        self._write_minimal_source_root(source_root)
        self._write_raw_generalization_summary(candidate_root)
        git_provenance = {"current": self._current_git_snapshot(repo_root), "current_matches_sources": True}
        (sequential_root / "policy-gated-sequential-canary-rollout-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "policy-gated-sequential-canary-rollout-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "batch_root": "outputs/path_feedback_batch_policy_gated_sequential_safe_choice_rollout_v1",
                    "episode_count": 36,
                    "step_count": 108,
                    "completed_episode_count": 36,
                    "policy_takeover_step_count": 48,
                    "accepted_takeover_step_count": 48,
                    "accepted_better_step_count": 48,
                    "multi_step_accepted_episode_count": 18,
                    "family_with_multi_step_accepted_episode_count": 6,
                    "accepted_takeover_family_count": 6,
                    "source_fallback_step_count": 0,
                    "state_continuity_violation_count": 0,
                    "episode_fallback_count": 0,
                    "canary_rejected_policy_choice_count": 0,
                    "cumulative_safety_regression_count": 0,
                    "cumulative_contract_violation_count": 0,
                    "cumulative_path_cost_regression_count": 0,
                    "cumulative_risk_regression_count": 0,
                    "cumulative_source_selection_regression_count": 0,
                    "invalid_action_mask_count": 0,
                    "fallback_or_open_grid_count": 0,
                    "candidate_git_current_matches_sources": True,
                    "checkpoint_metadata_git_current_matches_sources": True,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                    "git_provenance": git_provenance,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                "bash",
                str(repo_root / "scripts" / "run_policy_training_readiness_review.sh"),
                "--batch-root",
                str(source_root),
                "--config",
                str(repo_root / "configs" / "policy_training_readiness_review_v1.json"),
                "--raw-policy-generalization-evaluation-summary",
                str(candidate_root / "raw-policy-generalization-evaluation-summary.json"),
                "--policy-gated-sequential-canary-rollout-summary",
                str(sequential_root / "policy-gated-sequential-canary-rollout-summary.json"),
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads((source_root / "policy-training-readiness-review-summary.json").read_text())
        self.assertEqual(summary["training_readiness_status"], "policy_gated_sequential_safe_choice_calibrated")
        self.assertEqual(summary["training_blockers"], [])

    def _step(
        self,
        episode_id: str,
        step_index: int,
        family: str,
        input_start_cell: list[int],
        controlled_goal: list[int],
        controlled_source: str,
    ) -> dict:
        return {
            "schema_version": "policy-gated-sequential-canary-step/v1",
            "episode_id": episode_id,
            "step_index": step_index,
            "scenario_group": family,
            "input_start_cell": input_start_cell,
            "source_execution_goal_cell": controlled_goal,
            "policy_execution_goal_cell": controlled_goal,
            "controlled_execution_goal_cell": controlled_goal,
            "controlled_choice_source": controlled_source,
            "decision_class": "canary_accepted_policy_choice",
            "accepted_choice_value_class": "accepted_better",
            "canary_rejection_reason_codes": [],
            "controlled_regression_reason_codes": [],
            "raw_policy_regression_reason_codes": [],
            "action_mask_valid": True,
        }

    def _config(self, **overrides: int) -> dict:
        validation = {
            "min_episode_count": 1,
            "min_step_count": 1,
            "min_completed_episode_count": 1,
            "min_policy_takeover_step_count": 1,
            "min_accepted_takeover_step_count": 1,
            "min_accepted_better_step_count": 1,
            "min_accepted_takeover_family_count": 1,
            "min_multi_step_accepted_episode_count": 1,
            "min_family_with_multi_step_accepted_episode_count": 1,
            "max_state_continuity_violation_count": 0,
            "max_episode_fallback_count": 0,
            "max_canary_rejected_policy_choice_count": 0,
            "max_invalid_action_mask_count": 0,
            "max_fallback_or_open_grid_count": 0,
            "max_cumulative_safety_regression_count": 0,
            "max_cumulative_contract_violation_count": 0,
            "max_cumulative_path_cost_regression_count": 0,
            "max_cumulative_risk_regression_count": 0,
            "max_cumulative_source_selection_regression_count": 0,
        }
        validation.update(overrides)
        return {"validation": validation, "non_goals": ["no_formal_ppo_rollout"]}

    def _current_git_snapshot(self, repo_root: Path) -> dict:
        import sys

        scripts_dir = repo_root / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from git_provenance import git_snapshot

        return git_snapshot(repo_root)

    def _write_minimal_source_root(self, source_root: Path) -> None:
        git_provenance = {
            "current": self._current_git_snapshot(Path(__file__).resolve().parents[1]),
            "current_matches_sources": True,
        }
        common = {
            "status": "passed",
            "reason_codes": [],
            "source_selected_candidate_changed_rate": 0.0,
            "calibrated_selected_candidate_changed_rate": 0.5,
            "git_provenance": git_provenance,
        }
        (source_root / "batch-evaluation-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "path-feedback-batch-evaluation-summary/v1",
                    "status": "passed",
                    "failed_count": 0,
                    "reason_codes": [],
                    "open_grid_fallback_used_count": 0,
                    "safety_regression_count": 0,
                    "git_provenance": git_provenance,
                }
            ),
            encoding="utf-8",
        )
        payloads = {
            "calibrated-policy-application-smoke-summary.json": {
                **common,
                "schema_version": "calibrated-policy-application-smoke-summary/v1",
                "rejected_goal_blocked_count": 0,
                "safety_regression_count": 0,
                "open_grid_fallback_used_count": 0,
                "applied_calibrated_candidate_count": 1,
                "recommended_next_action": "ready_for_policy_training_readiness_review",
                "does_not_modify_default_astar": True,
                "does_not_modify_ppo": True,
                "does_not_modify_network": True,
                "does_not_modify_action_space": True,
                "does_not_modify_model_explorer_contract": True,
                "does_not_modify_path_planner_route_contract": True,
                "does_not_modify_path_planner_sidecar_contract": True,
                "no_ackermann_feasible_trajectory_claim": True,
            },
            "channel-aware-training-readiness-summary.json": {
                **common,
                "schema_version": "channel-aware-training-readiness-summary/v1",
            },
            "channel-aware-contrast-coverage-summary.json": {
                **common,
                "schema_version": "channel-aware-contrast-coverage-summary/v1",
            },
            "channel-aware-selection-contrast-calibration-summary.json": {
                **common,
                "schema_version": "channel-aware-selection-contrast-calibration-summary/v1",
            },
        }
        for name, payload in payloads.items():
            (source_root / name).write_text(json.dumps(payload), encoding="utf-8")

    def _write_raw_generalization_summary(self, candidate_root: Path) -> None:
        git_provenance = {
            "current": self._current_git_snapshot(Path(__file__).resolve().parents[1]),
            "current_matches_sources": True,
        }
        (candidate_root / "raw-policy-generalization-evaluation-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "raw-policy-generalization-evaluation-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "test_generalization_passed": True,
                    "test_raw_policy_regression_reduction_rate": 1.0,
                    "overfit_gap": 0.0,
                    "test_regression_count": 0,
                    "test_invalid_action_mask_count": 0,
                    "test_fallback_or_open_grid_count": 0,
                    "test_safety_regression_count": 0,
                    "test_contract_violation_count": 0,
                    "test_path_cost_regression_count": 0,
                    "test_risk_regression_count": 0,
                    "test_source_selection_regression_count": 0,
                    "candidate_git_current_matches_sources": True,
                    "checkpoint_metadata_git_current_matches_sources": True,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                    "git_provenance": git_provenance,
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
