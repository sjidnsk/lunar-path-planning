import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class PolicyDifferentiatingCounterfactualCoverageRolloutsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="policy-counterfactual-coverage-"))
        self.policy_margin_root = self.temp_dir / "policy-margin"
        self.stage5a1_root = self.temp_dir / "stage5a1"
        self.coverage_driven_root = self.temp_dir / "coverage-driven"
        self.reward_root = self.temp_dir / "reward"
        self.signal_root = self.temp_dir / "signal"
        self.performance_root = self.temp_dir / "performance"
        self.quasi_real_root = self.temp_dir / "quasi-real"
        self.output_root = self.temp_dir / "counterfactual-rollouts"
        for path in (
            self.policy_margin_root,
            self.stage5a1_root,
            self.coverage_driven_root,
            self.reward_root,
            self.signal_root,
            self.performance_root,
            self.quasi_real_root,
        ):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_generates_counterfactual_coverage_from_candidate_expanded_cells_and_sidecar(self) -> None:
        from scripts.run_policy_differentiating_counterfactual_coverage_rollouts import (
            run_policy_differentiating_counterfactual_coverage_rollouts,
        )

        self._write_stage5a_inputs(candidate_has_missing_coverage=True)
        self._write_stage5a1_inputs()
        self._write_support_summaries()
        self._write_coverage_driven_inputs()
        self._write_quasi_real_summary(include_counterfactual_path=True)

        summary = run_policy_differentiating_counterfactual_coverage_rollouts(
            policy_margin_root=self.policy_margin_root,
            candidate_materialization_root=self.stage5a1_root,
            coverage_driven_root=self.coverage_driven_root,
            reward_refinement_root=self.reward_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            quasi_real_root=self.quasi_real_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(
            summary["next_required_change"],
            "rerun_coverage_driven_ppo_with_refined_reward_or_advantage",
        )
        self.assertEqual(summary["input_missing_candidate_count"], 1)
        self.assertEqual(summary["counterfactual_coverage_candidate_count"], 1)
        self.assertGreater(summary["candidate_expected_coverage_nonzero_count"], 0)
        self.assertGreater(summary["candidate_information_gain_nonzero_count"], 0)
        self.assertEqual(summary["safe_better_than_teacher_candidate_count"], 1)
        self.assertEqual(summary["safe_better_than_teacher_family_count"], 1)
        self.assertEqual(summary["missing_counterfactual_source_count"], 0)
        self.assertEqual(summary["fallback_gain_contamination_count"], 0)
        self.assertEqual(summary["controlled_regression_count"], 0)
        self.assertEqual(summary["stage5a1_rerun_status"], "passed")
        self.assertEqual(summary["stage5a_overlay_rerun_status"], "passed")
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["performance_claimed"])

        counterfactual_rows = self._read_jsonl(
            self.output_root / "counterfactual-coverage-rollouts.jsonl"
        )
        self.assertEqual(len(counterfactual_rows), 1)
        row = counterfactual_rows[0]
        self.assertTrue(row["coverage_source_available"])
        self.assertEqual(row["match_method"], "counterfactual_candidate_expanded_cells_sidecar")
        self.assertGreater(row["expected_coverage_rate_delta"], 0.0)
        self.assertGreater(row["expected_new_coverage_area"], 0.0)
        self.assertGreater(row["information_gain"], 0.0)
        self.assertIsNotNone(row["valuable_coverage_proxy"])

        for filename in (
            "policy-differentiating-counterfactual-coverage-rollouts-summary.json",
            "counterfactual-coverage-rollouts.jsonl",
            "candidate-level-coverage-overlay.jsonl",
            "source-link-audit.json",
            "family-action-gap-report.json",
            "stage5a1-rerun-summary.json",
            "policy-differentiating-counterfactual-coverage-rollouts-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_keeps_counterfactual_missing_when_route_or_sidecar_evidence_is_absent(self) -> None:
        from scripts.run_policy_differentiating_counterfactual_coverage_rollouts import (
            run_policy_differentiating_counterfactual_coverage_rollouts,
        )

        self._write_stage5a_inputs(candidate_has_missing_coverage=True)
        self._write_stage5a1_inputs()
        self._write_support_summaries()
        self._write_coverage_driven_inputs()
        self._write_quasi_real_summary(include_counterfactual_path=False)

        summary = run_policy_differentiating_counterfactual_coverage_rollouts(
            policy_margin_root=self.policy_margin_root,
            candidate_materialization_root=self.stage5a1_root,
            coverage_driven_root=self.coverage_driven_root,
            reward_refinement_root=self.reward_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            quasi_real_root=self.quasi_real_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["next_required_change"], "expand_counterfactual_coverage_source_generation")
        self.assertIn("counterfactual_coverage_source_missing", summary["reason_codes"])
        self.assertEqual(summary["input_missing_candidate_count"], 1)
        self.assertEqual(summary["counterfactual_coverage_candidate_count"], 0)
        self.assertEqual(summary["missing_counterfactual_source_count"], 1)
        self.assertEqual(summary["safe_better_than_teacher_candidate_count"], 0)

        counterfactual_rows = self._read_jsonl(
            self.output_root / "counterfactual-coverage-rollouts.jsonl"
        )
        self.assertEqual(len(counterfactual_rows), 1)
        row = counterfactual_rows[0]
        self.assertFalse(row["coverage_source_available"])
        self.assertIsNone(row["expected_coverage_rate_delta"])
        self.assertIsNone(row["expected_new_coverage_area"])
        self.assertIn("candidate_route_cells_missing", row["missing_reason_codes"])

    def _write_stage5a_inputs(self, *, candidate_has_missing_coverage: bool) -> None:
        rows = [
            self._action_row(action=0, teacher=0, coverage=0.02, observed=True),
            self._action_row(
                action=1,
                teacher=0,
                coverage=None if candidate_has_missing_coverage else 0.05,
                observed=False,
            ),
        ]
        self._write_jsonl(self.policy_margin_root / "policy-coverage-action-level-audit.jsonl", rows)
        self._write_json(
            self.policy_margin_root / "policy-coverage-opportunity-margin-audit-summary.json",
            {
                "schema_version": "policy-coverage-opportunity-margin-audit-summary/v1",
                "status": "passed",
                "reason_codes": ["candidate_coverage_features_missing"],
                "next_required_change": "collect_more_policy_differentiating_coverage",
                "context_count": 1,
                "action_candidate_row_count": 2,
                "safe_better_than_teacher_count": 0,
                "fallback_gain_contamination_count": 0,
                "controlled_regression_count": 0,
                "runs_new_ppo_update": False,
                "performance_claimed": False,
            },
        )

    def _write_stage5a1_inputs(self) -> None:
        self._write_json(
            self.stage5a1_root / "candidate-level-coverage-materialization-summary.json",
            {
                "schema_version": "candidate-level-coverage-materialization-summary/v1",
                "status": "failed",
                "next_required_change": "generate_policy_differentiating_coverage_rollouts",
                "missing_candidate_coverage_source_count": 1,
                "counterfactual_coverage_candidate_count": 0,
                "safe_better_than_teacher_candidate_count": 0,
                "runs_new_ppo_update": False,
                "performance_claimed": False,
            },
        )
        self._write_jsonl(
            self.stage5a1_root / "candidate-level-coverage-overlay.jsonl",
            [
                self._overlay_row(action=0, available=True),
                self._overlay_row(action=1, available=False),
            ],
        )
        self._write_json(
            self.stage5a1_root / "source-link-audit.json",
            {
                "schema_version": "candidate-level-coverage-source-link-audit/v1",
                "missing_by_family": {"family-a": 1},
                "missing_by_action_index": {"1": 1},
                "match_method_counts": {
                    "executed_action_actual_path_feedback": 1,
                    "candidate_source_without_coverage": 1,
                },
            },
        )

    def _write_support_summaries(self) -> None:
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
                "performance_claimed": False,
            },
        )
        self._write_json(
            self.performance_root / "exploration-coverage-performance-evaluation-summary.json",
            {
                "schema_version": "exploration-coverage-performance-evaluation-summary/v1",
                "status": "failed",
                "coverage_performance_status": "failed",
                "performance_claimed": False,
            },
        )

    def _write_coverage_driven_inputs(self) -> None:
        features = [
            [0.0, 0.0, 0.20, 0.50, 0.0, 0.0, 0.5, 1.0],
            [0.0, 0.0, 0.19, 0.45, 0.0, 0.0, 0.6, 1.0],
        ]
        self._write_json(
            self.coverage_driven_root / "coverage-driven-ppo-improvement-run-summary.json",
            {
                "schema_version": "coverage-driven-ppo-improvement-run-summary/v1",
                "status": "failed",
                "reason_codes": ["no_coverage_return_improvement"],
                "coverage_aware_batch_episodes": str(
                    self.coverage_driven_root / "coverage-aware-ppo-batch" / "ppo-rollout-episodes.jsonl"
                ),
                "replay_audit": str(self.coverage_driven_root / "coverage-driven-ppo-replay-audit.json"),
                "ppo_update_summary": str(self.coverage_driven_root / "coverage-driven-ppo-update-summary.json"),
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
                    "transitions": [
                        {
                            "action_index": 0,
                            "controlled_action_index": 0,
                            "teacher_action_index": 0,
                            "pre_policy_logits": [1.0, 0.4],
                            "post_policy_logits": [1.0, 0.4],
                            "info": {
                                "context_id": "ctx-a",
                                "episode_id": "episode-a",
                                "step_index": 0,
                                "scenario_id": "scenario-a",
                                "scenario_family": "family-a",
                                "controlled_action_index": 0,
                                "teacher_action_index": 0,
                                "coverage_rate_delta": 0.02,
                                "valuable_area_covered": 0.01,
                                "new_area_covered": 0.02,
                                "information_gain": 0.02,
                                "path_cost": 0.50,
                                "risk": 0.20,
                                "energy_cost": 0.0,
                                "actual_coverage_gain_source": "path_feedback",
                                "fallback_like": False,
                                "controlled_regression_reason_codes": [],
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
                                "action_mask": [True, True],
                                "candidate_cells": [[0, 0], [1, 1]],
                                "candidate_missing_feature_names": [[], []],
                                "candidate_missing_indicator_names": [],
                                "candidate_missing_indicators": [[], []],
                            },
                        }
                    ],
                }
            ],
        )
        self._write_json(
            self.coverage_driven_root / "coverage-driven-ppo-replay-audit.json",
            {
                "schema_version": "coverage-driven-ppo-replay-audit/v1",
                "status": "passed",
                "reason_codes": [],
                "rows": [
                    {
                        "schema_version": "coverage-driven-ppo-replay-row/v1",
                        "context_id": "ctx-a",
                        "episode_id": "episode-a",
                        "step_index": 0,
                        "scenario_id": "scenario-a",
                        "scenario_family": "family-a",
                        "teacher_action_index": 0,
                        "raw_policy_action_index": 0,
                        "controlled_action_index": 0,
                        "policy_action_accepted": True,
                        "guard_rejected": False,
                        "controlled_regression_reason_codes": [],
                        "fallback_like": False,
                    }
                ],
            },
        )
        self._write_json(
            self.coverage_driven_root / "coverage-driven-ppo-update-summary.json",
            {
                "schema_version": "limited-ppo-update-smoke-summary/v1",
                "status": "passed",
                "optimizer_train_transition_count": 1,
                "runs_formal_ppo_rollout": False,
            },
        )

    def _write_quasi_real_summary(self, *, include_counterfactual_path: bool) -> None:
        candidate_one = {
            "action_index": 1,
            "cell": [1, 1],
            "utility": 0.6,
            "reachable": True,
            "path_cost": 0.45,
            "risk": 0.19,
            "energy_cost": 0.0,
            "open_grid_fallback_used": False,
            "replan_required": False,
            "candidate_role": "policy_target",
        }
        if include_counterfactual_path:
            candidate_one["diagnostics"] = {
                "expanded_cells": [[1, 1], [1, 2], [2, 1], [2, 2], [3, 1], [3, 2]]
            }
        candidates = [
            {
                "action_index": 0,
                "cell": [0, 0],
                "utility": 0.5,
                "reachable": True,
                "path_cost": 0.50,
                "risk": 0.20,
                "energy_cost": 0.0,
                "open_grid_fallback_used": False,
                "replan_required": False,
                "candidate_role": "policy_target",
                "diagnostics": {"expanded_cells": [[0, 0], [0, 1]]},
            },
            candidate_one,
        ]
        self._write_json(
            self.quasi_real_root / "quasi-real-map-path-feedback-summary.json",
            {
                "schema_version": "quasi-real-map-path-feedback-summary/v1",
                "scenario_count": 1,
                "scenarios": [
                    {
                        "scenario_id": "scenario-a",
                        "scenario_group": "family-a",
                        "coverage_rate_delta": 0.02,
                        "open_grid_fallback_used": False,
                        "path_feedback": {
                            "candidate_count": 2,
                            "reachable_count": 2,
                            "failure_count": 0,
                            "replan_count": 0,
                            "candidates": candidates,
                        },
                    }
                ],
            },
        )
        sidecar_dir = self.quasi_real_root / "path_planner_sidecars"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(
            sidecar_dir / "scenario-a.path-planner-sidecar.json",
            {
                "schema_version": "path-planner-sidecar/v1",
                "cost": [[1.0 for _ in range(4)] for _ in range(4)],
                "passable_mask": [[True for _ in range(4)] for _ in range(4)],
                "terrain_layers": {
                    "risk": [[0.1 for _ in range(4)] for _ in range(4)],
                    "confidence": [[0.9 for _ in range(4)] for _ in range(4)],
                    "observation_count": [[1.0 for _ in range(4)] for _ in range(4)],
                    "dem": [[0.0 for _ in range(4)] for _ in range(4)],
                },
                "metadata": {
                    "map_source": {"roi_name": "family-a", "split": "train", "resolution_m": 20.0},
                    "passable_ratio": 1.0,
                },
            },
        )
        self._write_json(
            self.quasi_real_root / "quasi-real-safe-alternative-opportunity-summary.json",
            {
                "schema_version": "quasi-real-safe-alternative-opportunity-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "runs_ppo_update": False,
                "performance_claimed": False,
            },
        )

    def _action_row(self, *, action: int, teacher: int, coverage: float | None, observed: bool) -> dict:
        return {
            "schema_version": "policy-coverage-action-level-audit-row/v1",
            "context_id": "ctx-a",
            "episode_id": "episode-a",
            "step_index": 0,
            "scenario_id": "scenario-a",
            "scenario_family": "family-a",
            "action_index": action,
            "candidate_cell": [action, action],
            "action_mask_valid": True,
            "teacher_action_index": teacher,
            "is_teacher_action": action == teacher,
            "is_pre_improvement_selected_action": action == teacher,
            "is_post_update_raw_action": action == teacher,
            "is_post_update_controlled_action": action == teacher,
            "path_cost": 0.50 if action == 0 else 0.45,
            "risk": 0.20 if action == 0 else 0.19,
            "energy_cost": 0.0,
            "coverage_rate_delta": coverage,
            "valuable_area_covered": coverage / 2.0 if coverage is not None else None,
            "new_area_covered": coverage,
            "information_gain_executed": coverage,
            "actual_coverage_gain_source": "path_feedback",
            "fallback_like": False,
            "guard_rejected": False,
            "policy_action_accepted": True,
            "controlled_regression_reason_codes": [],
            "counterfactual_coverage_observed": observed,
        }

    def _overlay_row(self, *, action: int, available: bool) -> dict:
        return {
            "schema_version": "candidate-level-coverage-overlay-row/v1",
            "context_id": "ctx-a",
            "episode_id": "episode-a",
            "step_index": 0,
            "scenario_id": "scenario-a",
            "scenario_family": "family-a",
            "action_index": action,
            "candidate_cell": [action, action],
            "is_teacher_action": action == 0,
            "coverage_source_available": available,
            "expected_coverage_rate_delta": 0.02 if available else None,
            "information_gain": 0.02 if available else None,
            "value": None,
            "missing_reason_codes": [] if available else ["counterfactual_candidate_coverage_source_missing"],
            "match_method": "executed_action_actual_path_feedback" if available else "candidate_source_without_coverage",
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
