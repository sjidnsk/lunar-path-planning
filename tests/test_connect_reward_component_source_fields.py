import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class ConnectRewardComponentSourceFieldsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_dir = str(self.repo_root / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="connect-reward-source-fields-"))
        self.signal_root = self.temp_dir / "coverage-signal"
        self.performance_root = self.temp_dir / "coverage-performance"
        self.reward_root = self.temp_dir / "reward-refinement"
        self.path_feedback_root = self.temp_dir / "path-feedback"
        self.output_root = self.temp_dir / "connect-output"
        for path in (
            self.signal_root,
            self.performance_root,
            self.reward_root,
            self.path_feedback_root,
        ):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_connects_actual_value_information_and_risk_sources_with_read_only_boundary(self) -> None:
        from scripts.run_connect_reward_component_source_fields import (
            run_connect_reward_component_source_fields,
        )

        row = self._delta_row("selected-context", delta=0.25)
        shadow = self._shadow_step("selected-context", action_index=1)
        self._write_inputs([row], [shadow])
        self._write_path_feedback(
            [
                self._path_feedback_candidate(
                    scenario_id="scenario-a",
                    action_index=1,
                    context_id="path-feedback-context-a1",
                    utility=0.8,
                    risk=0.4,
                )
            ]
        )

        summary = run_connect_reward_component_source_fields(
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            reward_refinement_root=self.reward_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            path_feedback_roots=[self.path_feedback_root],
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reward_component_source_field_status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["next_required_change"], "rerun_coverage_aware_reward_refinement")
        self.assertEqual(summary["expected_actual_coverage_confusion_count"], 0)
        self.assertEqual(summary["fallback_coverage_gain_claimed_as_policy_gain_count"], 0)
        self.assertEqual(summary["controlled_regression_count"], 0)
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])
        self.assertFalse(summary["formal_release_claimed"])

        for filename in (
            "connect-reward-component-source-fields-summary.json",
            "source-field-wiring-audit.json",
            "component-provenance.jsonl",
            "replay-validation.json",
            "connect-reward-component-source-fields-rejection-report.json",
            "connect-reward-component-source-fields-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

        provenance = self._read_jsonl(self.output_root / "component-provenance.jsonl")
        self.assertEqual(len(provenance), 1)
        connected = provenance[0]
        self.assertAlmostEqual(connected["actual_valuable_area_covered"], 0.2)
        self.assertAlmostEqual(connected["actual_information_gain"], 0.25)
        self.assertAlmostEqual(connected["risk"], 0.4)
        self.assertEqual(connected["source_fields"]["valuable_area_bonus"], "path_feedback.coverage_rate_delta*path_feedback.candidates.utility")
        self.assertEqual(connected["source_fields"]["information_gain_bonus"], "path_feedback.coverage_rate_delta")
        self.assertEqual(connected["source_fields"]["risk_penalty"], "path_feedback.candidates.risk")
        self.assertEqual(connected["path_feedback_match_method"], "scenario_action")

    def test_stage4_consumes_matching_connector_overlay_and_clears_missing_sources(self) -> None:
        from scripts.run_connect_reward_component_source_fields import (
            run_connect_reward_component_source_fields,
        )
        from scripts.run_coverage_aware_reward_refinement import (
            run_coverage_aware_reward_refinement,
        )

        row = self._delta_row("selected-context", delta=0.25)
        shadow = self._shadow_step("selected-context", action_index=1)
        self._write_inputs([row], [shadow])
        self._write_path_feedback(
            [
                self._path_feedback_candidate(
                    scenario_id="scenario-a",
                    action_index=1,
                    context_id="path-feedback-context-a1",
                    utility=0.8,
                    risk=0.4,
                )
            ]
        )
        connector_summary = run_connect_reward_component_source_fields(
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            reward_refinement_root=self.reward_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            path_feedback_roots=[self.path_feedback_root],
        )
        self.assertEqual(connector_summary["status"], "passed")

        formal_root, replay_root, selected_root = self._write_stage4_inputs(row, shadow)
        stage4_output = self.temp_dir / "stage4-rerun"
        summary = run_coverage_aware_reward_refinement(
            formal_training_root=formal_root,
            post_training_replay_root=replay_root,
            selected_candidate_root=selected_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            output_root=stage4_output,
            repo_root=self.repo_root,
            reward_component_source_root=self.output_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertNotIn("valuable_coverage_signal_missing", summary["reason_codes"])
        self.assertNotIn("information_gain_signal_missing", summary["reason_codes"])
        self.assertNotIn("risk_signal_missing", summary["reason_codes"])
        source_audit = self._load_json(stage4_output / "source-field-audit.json")
        self.assertEqual(source_audit["missing_required_component_source_count"], 0)

    def test_rejects_expected_only_coverage_as_information_or_valuable_source(self) -> None:
        from scripts.run_connect_reward_component_source_fields import (
            run_connect_reward_component_source_fields,
        )

        row = self._delta_row("selected-context", delta=None, expected_delta=0.25)
        shadow = self._shadow_step("selected-context", action_index=1)
        self._write_inputs([row], [shadow], expected_actual_confusion_count=1)
        self._write_path_feedback(
            [
                self._path_feedback_candidate(
                    scenario_id="scenario-a",
                    action_index=1,
                    context_id="path-feedback-context-a1",
                    utility=0.8,
                    risk=0.4,
                )
            ]
        )

        summary = run_connect_reward_component_source_fields(
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            reward_refinement_root=self.reward_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            path_feedback_roots=[self.path_feedback_root],
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("expected_actual_coverage_confusion", summary["reason_codes"])
        provenance = self._read_jsonl(self.output_root / "component-provenance.jsonl")
        self.assertIsNone(provenance[0]["actual_valuable_area_covered"])
        self.assertIsNone(provenance[0]["actual_information_gain"])

    def test_missing_candidate_source_fails_without_manufacturing_positive_reward_sources(self) -> None:
        from scripts.run_connect_reward_component_source_fields import (
            run_connect_reward_component_source_fields,
        )

        row = self._delta_row("selected-context", delta=0.25)
        shadow = self._shadow_step("selected-context", action_index=1)
        self._write_inputs([row], [shadow])
        self._write_path_feedback([])

        summary = run_connect_reward_component_source_fields(
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            reward_refinement_root=self.reward_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            path_feedback_roots=[self.path_feedback_root],
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("reward_component_source_missing", summary["reason_codes"])
        self.assertIn("path_feedback_candidate_source_missing", summary["reason_codes"])
        provenance = self._read_jsonl(self.output_root / "component-provenance.jsonl")
        self.assertIsNone(provenance[0]["actual_valuable_area_covered"])
        self.assertAlmostEqual(provenance[0]["actual_information_gain"], 0.25)
        self.assertIsNone(provenance[0]["risk"])

    def test_rejects_fallback_policy_gain_contamination_and_controlled_regression(self) -> None:
        from scripts.run_connect_reward_component_source_fields import (
            run_connect_reward_component_source_fields,
        )

        fallback = self._delta_row(
            "fallback-context",
            delta=0.20,
            choice_source="source_fallback",
            claimed_actor="policy",
        )
        regression = self._delta_row(
            "regression-context",
            delta=0.10,
            controlled_reasons=["risk_regression"],
        )
        shadows = [
            self._shadow_step("fallback-context", action_index=0, choice_source="source_fallback"),
            self._shadow_step("regression-context", action_index=1, controlled_reasons=["risk_regression"]),
        ]
        self._write_inputs(
            [fallback, regression],
            shadows,
            fallback_claimed_policy_count=1,
            controlled_regression_count=1,
        )
        self._write_path_feedback(
            [
                self._path_feedback_candidate("scenario-a", 0, "fallback-pf", utility=0.5, risk=0.3),
                self._path_feedback_candidate("scenario-a", 1, "regression-pf", utility=0.7, risk=0.4),
            ]
        )

        summary = run_connect_reward_component_source_fields(
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            reward_refinement_root=self.reward_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            path_feedback_roots=[self.path_feedback_root],
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("fallback_policy_gain_contamination", summary["reason_codes"])
        self.assertIn("controlled_regression_present", summary["reason_codes"])
        self.assertEqual(summary["fallback_coverage_gain_claimed_as_policy_gain_count"], 1)
        self.assertEqual(summary["controlled_regression_count"], 1)

    def _write_inputs(
        self,
        delta_rows: list[dict],
        shadow_steps: list[dict],
        *,
        expected_actual_confusion_count: int = 0,
        fallback_claimed_policy_count: int = 0,
        controlled_regression_count: int = 0,
    ) -> None:
        self._write_jsonl(self.signal_root / "coverage-delta-audit.jsonl", delta_rows)
        self._write_json(
            self.signal_root / "exploration-coverage-signal-audit-summary.json",
            {
                "schema_version": "exploration-coverage-signal-audit-summary/v1",
                "status": "passed" if expected_actual_confusion_count == 0 else "failed",
                "reason_codes": [],
                "coverage_signal_status": "passed" if expected_actual_confusion_count == 0 else "failed",
                "actual_coverage_gain_source": "path_feedback",
                "nonzero_actual_coverage_delta_count": sum(
                    1 for row in delta_rows if row.get("coverage_rate_delta")
                ),
                "coverage_delta_default_zero_count": 0,
                "expected_actual_coverage_confusion_count": expected_actual_confusion_count,
                "fallback_coverage_gain_claimed_as_policy_gain_count": fallback_claimed_policy_count,
                "controlled_regression_count": controlled_regression_count,
                "coverage_delta_audit": str(self.signal_root / "coverage-delta-audit.jsonl"),
            },
        )
        self._write_jsonl(self.performance_root / "shadow-steps.jsonl", shadow_steps)
        self._write_jsonl(self.performance_root / "coverage-performance-metric-table.jsonl", [])
        self._write_json(
            self.performance_root / "coverage-performance-comparison-audit.json",
            {
                "schema_version": "coverage-performance-comparison-audit/v1",
                "selected_actor": "selected_ppo_candidate",
                "best_baseline_actor": "teacher",
                "selected_teacher_equivalent": True,
            },
        )
        self._write_json(
            self.performance_root / "exploration-coverage-performance-evaluation-summary.json",
            {
                "schema_version": "exploration-coverage-performance-evaluation-summary/v1",
                "status": "failed",
                "reason_codes": ["coverage_performance_not_improved"],
                "coverage_performance_status": "failed",
                "coverage_delta_audit": str(self.signal_root / "coverage-delta-audit.jsonl"),
                "shadow_steps": str(self.performance_root / "shadow-steps.jsonl"),
                "metric_table": str(self.performance_root / "coverage-performance-metric-table.jsonl"),
                "comparison_audit": str(self.performance_root / "coverage-performance-comparison-audit.json"),
                "runs_new_ppo_update": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "formal_release_claimed": False,
            },
        )
        self._write_json(
            self.reward_root / "source-field-audit.json",
            {
                "schema_version": "coverage-aware-reward-source-field-audit/v1",
                "audited_row_count": len(delta_rows),
                "missing_required_components": [
                    "valuable_area_bonus",
                    "information_gain_bonus",
                    "risk_penalty",
                ],
                "missing_required_component_source_count": 3,
            },
        )
        self._write_json(
            self.reward_root / "coverage-aware-reward-refinement-summary.json",
            {
                "schema_version": "coverage-aware-reward-refinement-summary/v1",
                "status": "failed",
                "reason_codes": [
                    "reward_component_source_missing",
                    "valuable_coverage_signal_missing",
                    "information_gain_signal_missing",
                    "risk_signal_missing",
                ],
                "next_required_change": "connect_reward_component_source_fields",
                "source_field_audit": str(self.reward_root / "source-field-audit.json"),
                "runs_new_ppo_update": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "formal_release_claimed": False,
            },
        )

    def _write_stage4_inputs(self, row: dict, shadow: dict) -> tuple[Path, Path, Path]:
        formal_root = self.temp_dir / "formal-training"
        replay_root = self.temp_dir / "post-training-replay"
        selected_root = self.temp_dir / "selected-candidate"
        collector_root = formal_root / "seed-00" / "collector"
        self._write_json(
            formal_root / "formal-ppo-training-run-summary.json",
            {
                "schema_version": "guarded-formal-ppo-training-run-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "seed_count": 1,
                "optimizer_train_transition_count": 1,
                "teacher_agreement_rate": 1.0,
                "controlled_regression_count": 0,
                "performance_claimed": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
            },
        )
        self._write_jsonl(
            formal_root / "formal-ppo-training-run-seed-summaries.jsonl",
            [{"seed": 0, "status": "passed", "collector_root": str(collector_root)}],
        )
        self._write_jsonl(
            collector_root / "ppo-rollout-episodes.jsonl",
            [{"transitions": [{"reward": 1.0, "info": {"context_id": row["context_id"]}}]}],
        )
        self._write_json(
            replay_root / "formal-ppo-post-training-stability-replay-summary.json",
            {
                "schema_version": "guarded-formal-ppo-post-training-stability-replay-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "teacher_agreement_rate": 1.0,
                "controlled_regression_count": 0,
                "runs_new_ppo_update": False,
            },
        )
        self._write_json(
            selected_root / "selected-formal-ppo-candidate-promotion-preflight-summary.json",
            {
                "schema_version": "selected-formal-ppo-candidate-promotion-preflight-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "selected_seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "teacher_agreement_rate": 1.0,
                "controlled_regression_count": 0,
                "multihorizon_steps": str(selected_root / "multihorizon-shadow-rollout-steps.jsonl"),
                "runs_new_ppo_update": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
            },
        )
        self._write_jsonl(selected_root / "multihorizon-shadow-rollout-steps.jsonl", [shadow])
        selected_metric = {
            "schema_version": "coverage-performance-metric-row/v1",
            "actor": "selected_ppo_candidate",
            "row_count": 1,
            "coverage_return": row["coverage_rate_delta"],
            "cumulative_coverage_rate_delta": row["coverage_rate_delta"],
            "valuable_area_covered": 0.0,
            "path_cost": 1.0,
            "risk": 1.0,
            "fallback_rate": 0.0,
            "teacher_agreement_rate": 1.0,
            "controlled_regression_count": 0,
        }
        teacher_metric = dict(selected_metric, actor="teacher")
        self._write_jsonl(
            self.performance_root / "coverage-performance-metric-table.jsonl",
            [selected_metric, teacher_metric],
        )
        return formal_root, replay_root, selected_root

    def _delta_row(
        self,
        context_id: str,
        *,
        delta: float | None,
        expected_delta: float = 0.0,
        choice_source: str = "policy",
        claimed_actor: str = "policy",
        controlled_reasons: list[str] | None = None,
    ) -> dict:
        return {
            "schema_version": "exploration-coverage-delta-audit-row/v1",
            "audit_index": 0,
            "episode_id": "shadow-episode",
            "source_episode_id": "source-episode",
            "step_index": 0,
            "source_step_index": 0,
            "context_id": context_id,
            "scenario_id": "scenario-a",
            "scenario_family": "family-a",
            "split": "train",
            "controlled_choice_source": choice_source,
            "controlled_choice_detail": "policy_teacher_aligned",
            "canonical_coverage_actor": "selected_ppo_candidate",
            "coverage_gain_claimed_actor": claimed_actor,
            "ppo_trainable": True,
            "initial_coverage_rate": 0.0,
            "final_coverage_rate": delta,
            "coverage_rate_delta": delta,
            "cumulative_coverage_rate_delta": delta,
            "expected_coverage_rate_delta": expected_delta,
            "actual_coverage_gain_source": "path_feedback" if delta is not None else "missing",
            "actual_delta_present": delta is not None,
            "actual_delta_finite": delta is not None,
            "actual_delta_nonzero": bool(delta),
            "actual_delta_default_zero": delta == 0.0,
            "expected_delta_nonzero": expected_delta != 0.0,
            "controlled_regression_reason_codes": controlled_reasons or [],
        }

    def _shadow_step(
        self,
        context_id: str,
        *,
        action_index: int,
        choice_source: str = "policy",
        controlled_reasons: list[str] | None = None,
    ) -> dict:
        return {
            "episode_id": "source-episode",
            "shadow_episode_id": "shadow-episode",
            "step_index": 0,
            "shadow_step_index": 0,
            "context_id": context_id,
            "scenario_id": "scenario-a",
            "scenario_family": "family-a",
            "split": "train",
            "controlled_choice_source": choice_source,
            "controlled_choice_detail": "policy_teacher_aligned",
            "controlled_action_index": action_index,
            "teacher_action_index": action_index,
            "controlled_regression_reason_codes": controlled_reasons or [],
            "policy_takes_control": choice_source == "policy",
            "reward": 1.0,
            "reward_components": {"teacher_following_bonus": 1.0},
            "observation": {
                "candidate_feature_names": ["value", "information_gain", "risk", "path_cost", "energy_cost"],
                "candidate_features": [
                    [0.0, 0.0, 0.0, 1.0, 0.5],
                    [0.0, 0.0, 0.0, 1.0, 0.5],
                ],
                "candidate_missing_indicator_names": [
                    "value_missing",
                    "information_gain_missing",
                    "risk_missing",
                    "path_cost_missing",
                    "energy_cost_missing",
                ],
                "candidate_missing_indicators": [
                    [1.0, 1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0],
                ],
                "action_mask": [True, True],
            },
        }

    def _path_feedback_candidate(
        self,
        scenario_id: str,
        action_index: int,
        context_id: str,
        *,
        utility: float,
        risk: float,
    ) -> dict:
        return {
            "scenario_id": scenario_id,
            "scenario_group": "family-a",
            "context_id": context_id,
            "action_index": action_index,
            "utility": utility,
            "risk": risk,
            "path_cost": 12.0,
            "reachable": True,
            "open_grid_fallback_used": False,
            "tracking_safety_violation_count": 0,
            "contract_safe": True,
        }

    def _write_path_feedback(self, candidates: list[dict]) -> None:
        self._write_json(
            self.path_feedback_root / "path-feedback-summary.json",
            {
                "schema_version": "path-feedback-summary/v1",
                "status": "passed",
                "scenarios": [
                    {
                        "scenario_id": "scenario-a",
                        "scenario_group": "family-a",
                        "coverage_rate_delta": 0.25,
                        "path_feedback": {"candidates": candidates},
                    }
                ],
            },
        )

    def _load_json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
