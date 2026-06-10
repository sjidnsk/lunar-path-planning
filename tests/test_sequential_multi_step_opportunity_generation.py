import json
import tempfile
import unittest
from pathlib import Path


class SequentialMultiStepOpportunityGenerationTests(unittest.TestCase):
    def test_diagnosis_marks_source_aligned_step_with_safe_better_alternative_as_policy_missed(self) -> None:
        from scripts.run_sequential_multi_step_opportunity_diagnosis import (
            run_sequential_multi_step_opportunity_diagnosis,
        )

        batch_root = Path(tempfile.mkdtemp(prefix="seq-multi-step-opportunity-"))
        self._write_rollout_files(
            batch_root,
            steps=[
                self._step(
                    episode_id="ep-mixed-a",
                    step_index=0,
                    scenario_id="npz_seq_canary_mixed_a_step00",
                    family="mixed_stress_detour",
                    decision_class="source_aligned",
                )
            ],
        )
        self._write_path_feedback(
            batch_root,
            "npz_seq_canary_mixed_a_step00",
            "mixed_stress_detour",
            [
                self._candidate("source-ctx", 0, [4, 6], path_cost=10.0, risk=0.20, utility=1.0, source=True),
                self._candidate("alt-ctx", 1, [5, 6], path_cost=9.5, risk=0.18, utility=1.02),
            ],
        )

        summary, diagnostics, exclusions = run_sequential_multi_step_opportunity_diagnosis(
            batch_root=batch_root,
            config=self._config(
                min_episode_count=1,
                min_step_count=1,
                min_multi_step_opportunity_episode_count=0,
                min_family_with_multi_step_opportunity_count=0,
                min_safe_better_alternative_step_count=1,
            ),
            repo_root=Path(__file__).resolve().parents[1],
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["safe_better_alternative_step_count"], 1)
        self.assertEqual(summary["policy_missed_existing_opportunity_count"], 1)
        self.assertEqual(diagnostics[0]["opportunity_class"], "policy_missed_existing_opportunity")
        self.assertEqual(exclusions["opportunity_exclusion_count"], 0)

    def test_diagnosis_marks_step_without_safe_better_alternative_as_opportunity_missing(self) -> None:
        from scripts.run_sequential_multi_step_opportunity_diagnosis import (
            run_sequential_multi_step_opportunity_diagnosis,
        )

        batch_root = Path(tempfile.mkdtemp(prefix="seq-multi-step-opportunity-missing-"))
        self._write_rollout_files(
            batch_root,
            steps=[
                self._step(
                    episode_id="ep-dense-a",
                    step_index=0,
                    scenario_id="npz_seq_canary_dense_a_step00",
                    family="dense_choke_safe_bypass",
                    decision_class="source_aligned",
                )
            ],
        )
        self._write_path_feedback(
            batch_root,
            "npz_seq_canary_dense_a_step00",
            "dense_choke_safe_bypass",
            [
                self._candidate("source-ctx", 0, [4, 6], path_cost=10.0, risk=0.20, utility=1.0, source=True),
                self._candidate("bad-alt", 1, [5, 6], path_cost=11.0, risk=0.25, utility=1.0),
            ],
        )

        summary, diagnostics, _ = run_sequential_multi_step_opportunity_diagnosis(
            batch_root=batch_root,
            config=self._config(
                min_episode_count=1,
                min_step_count=1,
                min_multi_step_opportunity_episode_count=0,
                min_family_with_multi_step_opportunity_count=0,
                min_safe_better_alternative_step_count=1,
            ),
            repo_root=Path(__file__).resolve().parents[1],
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("safe_better_alternative_step_count_below_threshold", summary["reason_codes"])
        self.assertEqual(summary["opportunity_missing_count"], 1)
        self.assertEqual(diagnostics[0]["opportunity_class"], "opportunity_missing")
        self.assertEqual(summary["next_required_change"], "sequential_multi_step_opportunity_generation_gap")

    def test_diagnosis_counts_multi_step_opportunity_by_episode_and_family(self) -> None:
        from scripts.run_sequential_multi_step_opportunity_diagnosis import (
            run_sequential_multi_step_opportunity_diagnosis,
        )

        batch_root = Path(tempfile.mkdtemp(prefix="seq-multi-step-opportunity-counts-"))
        steps = [
            self._step("ep-channel-a", 0, "npz_seq_canary_channel_a_step00", "channel_contrast", "source_aligned"),
            self._step("ep-channel-a", 1, "npz_seq_canary_channel_a_step01", "channel_contrast", "source_aligned"),
            self._step("ep-channel-a", 2, "npz_seq_canary_channel_a_step02", "channel_contrast", "source_aligned"),
        ]
        self._write_rollout_files(batch_root, steps=steps)
        for step in steps:
            self._write_path_feedback(
                batch_root,
                step["scenario_id"],
                step["scenario_group"],
                [
                    self._candidate(f"source-{step['step_index']}", 0, [4 + step["step_index"], 6], 10.0, 0.20, 1.0, True),
                    self._candidate(f"alt-{step['step_index']}", 1, [5 + step["step_index"], 6], 9.5, 0.18, 1.02),
                ],
            )

        summary, _, _ = run_sequential_multi_step_opportunity_diagnosis(
            batch_root=batch_root,
            config=self._config(
                min_episode_count=1,
                min_step_count=3,
                min_multi_step_opportunity_episode_count=1,
                min_family_with_multi_step_opportunity_count=1,
                min_safe_better_alternative_step_count=2,
            ),
            repo_root=Path(__file__).resolve().parents[1],
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["safe_better_alternative_step_count"], 3)
        self.assertEqual(summary["multi_step_opportunity_episode_count"], 1)
        self.assertEqual(summary["family_with_multi_step_opportunity_count"], 1)
        self.assertEqual(
            summary["family_opportunity_summary"]["channel_contrast"],
            {
                "episode_count": 1,
                "step_count": 3,
                "safe_better_alternative_step_count": 3,
                "multi_step_opportunity_episode_count": 1,
                "policy_used_existing_opportunity_count": 0,
                "policy_missed_existing_opportunity_count": 3,
                "policy_rejected_existing_opportunity_count": 0,
                "opportunity_missing_count": 0,
                "accepted_takeover_step_count": 0,
                "source_aligned_step_count": 3,
                "multi_step_accepted_episode_count": 0,
                "gap_reason": "policy_missed_existing_opportunity",
            },
        )

    def test_diagnosis_requires_min_multi_step_opportunity_episode_count_per_family(self) -> None:
        from scripts.run_sequential_multi_step_opportunity_diagnosis import (
            run_sequential_multi_step_opportunity_diagnosis,
        )

        batch_root = Path(tempfile.mkdtemp(prefix="seq-multi-step-opportunity-family-min-"))
        steps = [
            self._step("ep-channel-a", 0, "npz_seq_canary_channel_a_step00", "channel_contrast", "source_aligned"),
            self._step("ep-channel-a", 1, "npz_seq_canary_channel_a_step01", "channel_contrast", "source_aligned"),
            self._step("ep-channel-b", 0, "npz_seq_canary_channel_b_step00", "channel_contrast", "source_aligned"),
            self._step("ep-channel-b", 1, "npz_seq_canary_channel_b_step01", "channel_contrast", "source_aligned"),
        ]
        self._write_rollout_files(batch_root, steps=steps)
        for step in steps:
            candidates = [
                self._candidate(f"source-{step['episode_id']}-{step['step_index']}", 0, [4, 6], 10.0, 0.20, 1.0, True)
            ]
            if step["episode_id"] == "ep-channel-a":
                candidates.append(
                    self._candidate(f"alt-{step['step_index']}", 1, [5, 6], 9.5, 0.18, 1.02)
                )
            self._write_path_feedback(
                batch_root,
                step["scenario_id"],
                step["scenario_group"],
                candidates,
            )

        summary, _, _ = run_sequential_multi_step_opportunity_diagnosis(
            batch_root=batch_root,
            config=self._config(
                min_episode_count=2,
                min_step_count=4,
                min_multi_step_opportunity_episode_count=1,
                min_family_with_multi_step_opportunity_count=1,
                min_safe_better_alternative_step_count=2,
                min_multi_step_opportunity_episode_count_per_family=2,
            ),
            repo_root=Path(__file__).resolve().parents[1],
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["multi_step_opportunity_episode_count_by_family"]["channel_contrast"], 1)
        self.assertEqual(
            summary["families_below_min_multi_step_opportunity_episode_count"],
            ["channel_contrast"],
        )
        self.assertEqual(
            summary["family_opportunity_summary"]["channel_contrast"]["gap_reason"],
            "multi_step_opportunity_episode_count_below_family_min",
        )
        self.assertIn(
            "multi_step_opportunity_episode_count_per_family_below_threshold",
            summary["reason_codes"],
        )

    def _write_rollout_files(self, batch_root: Path, *, steps: list[dict]) -> None:
        batch_root.mkdir(parents=True, exist_ok=True)
        (batch_root / "policy-gated-sequential-canary-rollout-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "policy-gated-sequential-canary-rollout-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "episode_count": len({step["episode_id"] for step in steps}),
                    "step_count": len(steps),
                }
            ),
            encoding="utf-8",
        )
        episodes = [
            {"episode_id": episode_id, "schema_version": "policy-gated-sequential-canary-episode/v1"}
            for episode_id in sorted({step["episode_id"] for step in steps})
        ]
        (batch_root / "policy-gated-sequential-canary-episodes.jsonl").write_text(
            "".join(json.dumps(episode) + "\n" for episode in episodes),
            encoding="utf-8",
        )
        (batch_root / "policy-gated-sequential-canary-steps.jsonl").write_text(
            "".join(json.dumps(step) + "\n" for step in steps),
            encoding="utf-8",
        )
        (batch_root / "policy-gated-sequential-canary-rejection-report.json").write_text(
            json.dumps({"failed_steps": []}),
            encoding="utf-8",
        )

    def _write_path_feedback(
        self,
        batch_root: Path,
        scenario_id: str,
        family: str,
        candidates: list[dict],
    ) -> None:
        step_root = batch_root / scenario_id
        step_root.mkdir(parents=True, exist_ok=True)
        (step_root / "path-feedback-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "path-feedback-summary/v1",
                    "diagnostic_profile": "execution",
                    "planner_extra_args": ["--planning-backend", "astar"],
                    "scenarios": [
                        {
                            "scenario_id": scenario_id,
                            "scenario_group": family,
                            "scenario_seed": 12001,
                            "scenario_variant_id": f"{scenario_id}-variant",
                            "path_feedback": {
                                "best_by_path_cost": {"context_id": "source-ctx"},
                                "candidates": candidates,
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def _candidate(
        self,
        context_id: str,
        action_index: int,
        goal: list[int],
        path_cost: float,
        risk: float,
        utility: float,
        source: bool = False,
    ) -> dict:
        return {
            "context_id": context_id,
            "action_index": action_index,
            "source_action_index": action_index,
            "policy_target_cell": goal,
            "execution_goal_cell": goal,
            "target_binding_mode": "same_action_execution_substitute",
            "reachable": True,
            "replan_required": False,
            "open_grid_fallback_used": False,
            "tracking_safety_violation_count": 0,
            "contract_safe": True,
            "path_cost": path_cost,
            "risk": risk,
            "utility": utility,
            "candidate_generation": {
                "source_selection_status": "source_selected" if source else "candidate"
            },
        }

    def _step(
        self,
        episode_id: str,
        step_index: int,
        scenario_id: str,
        family: str,
        decision_class: str,
    ) -> dict:
        return {
            "schema_version": "policy-gated-sequential-canary-step/v1",
            "episode_id": episode_id,
            "step_index": step_index,
            "scenario_id": scenario_id,
            "scenario_group": family,
            "source_selected_context_id": "source-ctx",
            "decision_class": decision_class,
            "input_start_cell": [1 + step_index, 6],
            "source_execution_goal_cell": [4 + step_index, 6],
            "controlled_execution_goal_cell": [4 + step_index, 6],
            "canary_rejection_reason_codes": [],
            "raw_policy_regression_reason_codes": [],
            "controlled_regression_reason_codes": [],
        }

    def _config(self, **validation_overrides: int) -> dict:
        validation = {
            "min_episode_count": 36,
            "min_step_count": 108,
            "min_multi_step_opportunity_episode_count": 12,
            "min_family_with_multi_step_opportunity_count": 6,
            "min_safe_better_alternative_step_count": 24,
            "max_opportunity_exclusion_count": 0,
        }
        validation.update(validation_overrides)
        return {
            "schema_version": "sequential-multi-step-opportunity-diagnosis-config/v1",
            "input_files": {
                "summary": "policy-gated-sequential-canary-rollout-summary.json",
                "steps": "policy-gated-sequential-canary-steps.jsonl",
                "episodes": "policy-gated-sequential-canary-episodes.jsonl",
                "rejection_report": "policy-gated-sequential-canary-rejection-report.json",
            },
            "output_files": {
                "summary": "sequential-multi-step-opportunity-diagnosis-summary.json",
                "diagnostics": "sequential-multi-step-opportunity-diagnostics.jsonl",
                "exclusion_report": "sequential-multi-step-opportunity-exclusion-report.json",
            },
            "evaluation": {
                "max_path_cost_regression": 0.0,
                "max_risk_regression": 0.0,
                "min_better_path_cost_delta": 0.25,
                "min_better_risk_delta": 0.01,
                "min_better_utility_delta": 0.005,
            },
            "validation": validation,
            "non_goals": ["no_formal_ppo_rollout"],
        }


if __name__ == "__main__":
    unittest.main()
