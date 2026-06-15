import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class CandidateLevelCoverageMaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="candidate-coverage-materialization-"))
        self.policy_margin_root = self.temp_dir / "policy-margin"
        self.coverage_driven_root = self.temp_dir / "coverage-driven"
        self.reward_root = self.temp_dir / "reward"
        self.signal_root = self.temp_dir / "signal"
        self.performance_root = self.temp_dir / "performance"
        self.quasi_real_root = self.temp_dir / "quasi-real"
        self.output_root = self.temp_dir / "materialization"
        for path in (
            self.policy_margin_root,
            self.coverage_driven_root,
            self.reward_root,
            self.signal_root,
            self.performance_root,
            self.quasi_real_root,
        ):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_fails_and_routes_to_rollout_generation_when_only_executed_teacher_has_coverage(self) -> None:
        from scripts.run_candidate_level_coverage_materialization import (
            run_candidate_level_coverage_materialization,
        )

        self._write_stage5a_action_rows(
            [
                self._action_row(action=0, teacher=0, coverage=0.07, observed=True),
                self._action_row(action=1, teacher=0, coverage=None, observed=False),
            ]
        )
        self._write_policy_margin_summary()
        self._write_coverage_driven_inputs(
            features=[
                self._candidate(expected=0.0, information=0.0, value=0.0, risk=0.20, path_cost=0.50),
                self._candidate(expected=0.0, information=0.0, value=0.0, risk=0.19, path_cost=0.45),
            ],
            teacher_action_index=0,
        )
        self._write_support_summaries()
        self._write_quasi_real_summary(candidate_coverages={})

        summary = run_candidate_level_coverage_materialization(
            policy_margin_root=self.policy_margin_root,
            coverage_driven_root=self.coverage_driven_root,
            reward_refinement_root=self.reward_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            quasi_real_root=self.quasi_real_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["next_required_change"], "generate_policy_differentiating_coverage_rollouts")
        self.assertIn("counterfactual_candidate_coverage_source_missing", summary["reason_codes"])
        self.assertEqual(summary["context_count"], 1)
        self.assertEqual(summary["action_candidate_row_count"], 2)
        self.assertEqual(summary["overlay_connected_candidate_count"], 1)
        self.assertEqual(summary["missing_candidate_coverage_source_count"], 1)
        self.assertEqual(summary["candidate_expected_coverage_nonzero_count"], 1)
        self.assertEqual(summary["counterfactual_coverage_candidate_count"], 0)
        self.assertEqual(summary["safe_better_than_teacher_candidate_count"], 0)
        self.assertEqual(summary["fallback_gain_contamination_count"], 0)
        self.assertEqual(summary["controlled_regression_count"], 0)
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["performance_claimed"])
        self.assertEqual(summary["stage5a_overlay_rerun_status"], "passed")

        overlay_rows = self._read_jsonl(self.output_root / "candidate-level-coverage-overlay.jsonl")
        self.assertEqual(len(overlay_rows), 2)
        missing_rows = [row for row in overlay_rows if not row["coverage_source_available"]]
        self.assertEqual(len(missing_rows), 1)
        self.assertIsNone(missing_rows[0]["expected_coverage_rate_delta"])
        self.assertIn("counterfactual_candidate_coverage_source_missing", missing_rows[0]["missing_reason_codes"])

        for filename in (
            "candidate-level-coverage-materialization-summary.json",
            "candidate-level-coverage-overlay.jsonl",
            "counterfactual-opportunity-audit.jsonl",
            "source-link-audit.json",
            "candidate-level-coverage-rejection-report.json",
            "candidate-level-coverage-materialization-report.md",
            "overlay-fed-stage5a-rerun-summary.json",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_passes_when_counterfactual_candidate_coverage_proves_safe_better_alternative(self) -> None:
        from scripts.run_candidate_level_coverage_materialization import (
            run_candidate_level_coverage_materialization,
        )

        self._write_stage5a_action_rows(
            [
                self._action_row(action=0, teacher=0, coverage=0.02, observed=True),
                self._action_row(action=1, teacher=0, coverage=None, observed=False),
            ]
        )
        self._write_policy_margin_summary()
        self._write_coverage_driven_inputs(
            features=[
                self._candidate(expected=0.0, information=0.0, value=0.0, risk=0.20, path_cost=0.50),
                self._candidate(expected=0.0, information=0.0, value=0.0, risk=0.19, path_cost=0.45),
            ],
            teacher_action_index=0,
        )
        self._write_support_summaries()
        self._write_quasi_real_summary(
            candidate_coverages={
                1: {
                    "expected_coverage_rate_delta": 0.30,
                    "expected_new_coverage_area": 0.30,
                    "information_gain": 0.08,
                    "value": 0.04,
                    "valuable_coverage_proxy": 0.15,
                }
            }
        )

        summary = run_candidate_level_coverage_materialization(
            policy_margin_root=self.policy_margin_root,
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
        self.assertEqual(summary["overlay_connected_candidate_count"], 2)
        self.assertEqual(summary["missing_candidate_coverage_source_count"], 0)
        self.assertEqual(summary["counterfactual_coverage_candidate_count"], 1)
        self.assertEqual(summary["safe_better_than_teacher_candidate_count"], 1)
        self.assertEqual(summary["safe_better_than_teacher_family_count"], 1)
        self.assertEqual(summary["stage5a_overlay_rerun_status"], "passed")

        rerun_summary = self._read_json(self.output_root / "overlay-fed-stage5a-rerun-summary.json")
        self.assertEqual(rerun_summary["safe_better_than_teacher_count"], 1)
        self.assertFalse(rerun_summary["runs_new_ppo_update"])

    def _write_stage5a_action_rows(self, rows: list[dict]) -> None:
        self._write_jsonl(self.policy_margin_root / "policy-coverage-action-level-audit.jsonl", rows)

    def _write_policy_margin_summary(self) -> None:
        self._write_json(
            self.policy_margin_root / "policy-coverage-opportunity-margin-audit-summary.json",
            {
                "schema_version": "policy-coverage-opportunity-margin-audit-summary/v1",
                "status": "passed",
                "reason_codes": ["candidate_coverage_features_missing"],
                "next_required_change": "collect_more_policy_differentiating_coverage",
                "context_count": 1,
                "action_candidate_row_count": 2,
                "fallback_gain_contamination_count": 0,
                "controlled_regression_count": 0,
                "runs_new_ppo_update": False,
                "performance_claimed": False,
            },
        )

    def _write_coverage_driven_inputs(self, *, features: list[list[float]], teacher_action_index: int) -> None:
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
                            "action_index": teacher_action_index,
                            "controlled_action_index": teacher_action_index,
                            "teacher_action_index": teacher_action_index,
                            "pre_policy_logits": [1.0, 0.4],
                            "post_policy_logits": [1.0, 0.4],
                            "info": {
                                "context_id": "ctx-materialize",
                                "episode_id": "episode-a",
                                "step_index": 0,
                                "scenario_id": "scenario-a",
                                "scenario_family": "family-a",
                                "controlled_action_index": teacher_action_index,
                                "teacher_action_index": teacher_action_index,
                                "coverage_rate_delta": 0.02,
                                "valuable_area_covered": 0.01,
                                "new_area_covered": 0.02,
                                "information_gain": 0.02,
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
                        "context_id": "ctx-materialize",
                        "episode_id": "episode-a",
                        "step_index": 0,
                        "scenario_id": "scenario-a",
                        "scenario_family": "family-a",
                        "teacher_action_index": teacher_action_index,
                        "raw_policy_action_index": teacher_action_index,
                        "controlled_action_index": teacher_action_index,
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
                "nonzero_actual_coverage_delta_count": 1,
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
                "performance_claimed": False,
            },
        )

    def _write_quasi_real_summary(self, *, candidate_coverages: dict[int, dict]) -> None:
        candidates = []
        for action_index, cell in enumerate(([0, 0], [1, 1])):
            candidate = {
                "action_index": action_index,
                "cell": cell,
                "utility": 0.5,
                "reachable": True,
                "path_cost": 0.50 if action_index == 0 else 0.45,
                "risk": 0.20 if action_index == 0 else 0.19,
                "open_grid_fallback_used": False,
                "replan_required": False,
                "candidate_role": "policy_target",
            }
            candidate.update(candidate_coverages.get(action_index, {}))
            candidates.append(candidate)
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
            "context_id": "ctx-materialize",
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
            "candidate_expected_coverage_rate_delta": 0.0,
            "candidate_expected_new_coverage_area": 0.0,
            "candidate_information_gain": 0.0,
            "candidate_value": 0.0,
            "candidate_utility": 0.5,
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

    def _candidate(self, *, expected: float, information: float, value: float, risk: float, path_cost: float) -> list[float]:
        return [expected, information, risk, path_cost, 0.0, value, 0.5, 1.0]

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _read_json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
