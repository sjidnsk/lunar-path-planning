import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class CoverageAwareRewardRefinementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_dir = str(self.repo_root / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="coverage-aware-reward-"))
        self.formal_root = self.temp_dir / "formal-training"
        self.replay_root = self.temp_dir / "post-training-replay"
        self.selected_root = self.temp_dir / "selected-candidate"
        self.signal_root = self.temp_dir / "coverage-signal"
        self.performance_root = self.temp_dir / "coverage-performance"
        self.output_root = self.temp_dir / "reward-refinement"
        for path in (
            self.formal_root,
            self.replay_root,
            self.selected_root,
            self.signal_root,
            self.performance_root,
        ):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_passes_reward_contract_with_actual_component_sources_and_read_only_boundary(self) -> None:
        from scripts.run_coverage_aware_reward_refinement import (
            run_coverage_aware_reward_refinement,
        )

        rows = [
            self._row("selected_ppo_candidate", 0, delta=0.10, value=2.0, information_gain=0.30),
            self._row("selected_ppo_candidate", 1, delta=0.08, value=1.5, information_gain=0.20),
            self._row("teacher", 0, delta=0.05, value=1.0, information_gain=0.10),
            self._row("teacher", 1, delta=0.04, value=1.0, information_gain=0.10),
        ]
        self._write_inputs(rows, selected_teacher_equivalent=False, performance_reason_codes=[])

        summary = run_coverage_aware_reward_refinement(
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reward_refinement_status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["next_required_change"], "coverage_driven_ppo_improvement_run")
        self.assertGreater(summary["reward_component_source_status"]["coverage_gain_bonus"]["positive_count"], 0)
        self.assertGreater(summary["reward_component_source_status"]["valuable_area_bonus"]["positive_count"], 0)
        self.assertGreater(summary["reward_component_source_status"]["information_gain_bonus"]["positive_count"], 0)
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])
        self.assertFalse(summary["formal_release_claimed"])

        for filename in (
            "coverage-aware-reward-refinement-summary.json",
            "reward-component-audit.jsonl",
            "source-field-audit.json",
            "reward-rescore-comparison.json",
            "reward-refinement-rejection-report.json",
            "coverage-aware-reward-refinement-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_rejects_expected_coverage_as_actual_coverage_bonus(self) -> None:
        from scripts.run_coverage_aware_reward_refinement import (
            run_coverage_aware_reward_refinement,
        )

        rows = [
            self._row(
                "selected_ppo_candidate",
                0,
                delta=None,
                expected_delta=0.12,
                value=2.0,
                information_gain=0.20,
            ),
            self._row("teacher", 0, delta=0.04, value=1.0, information_gain=0.10),
        ]
        self._write_inputs(rows, expected_actual_confusion_count=1)

        summary = run_coverage_aware_reward_refinement(
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("expected_actual_coverage_confusion", summary["reason_codes"])
        audit_rows = self._read_jsonl(self.output_root / "reward-component-audit.jsonl")
        selected = next(row for row in audit_rows if row["actor"] == "selected_ppo_candidate")
        self.assertEqual(selected["reward_components"]["coverage_gain_bonus"], 0.0)

    def test_missing_valuable_and_information_sources_do_not_create_positive_bonus(self) -> None:
        from scripts.run_coverage_aware_reward_refinement import (
            run_coverage_aware_reward_refinement,
        )

        rows = [
            self._row(
                "selected_ppo_candidate",
                0,
                delta=0.10,
                value=0.0,
                information_gain=0.0,
                value_missing=True,
                information_missing=True,
            ),
            self._row(
                "teacher",
                0,
                delta=0.10,
                value=0.0,
                information_gain=0.0,
                value_missing=True,
                information_missing=True,
            ),
        ]
        self._write_inputs(rows)

        summary = run_coverage_aware_reward_refinement(
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("reward_component_source_missing", summary["reason_codes"])
        self.assertIn("valuable_coverage_signal_missing", summary["reason_codes"])
        self.assertIn("information_gain_signal_missing", summary["reason_codes"])
        audit_rows = self._read_jsonl(self.output_root / "reward-component-audit.jsonl")
        self.assertTrue(all(row["reward_components"]["valuable_area_bonus"] == 0.0 for row in audit_rows))
        self.assertTrue(all(row["reward_components"]["information_gain_bonus"] == 0.0 for row in audit_rows))

    def test_path_risk_energy_penalties_use_candidate_features_when_step_deltas_are_zero(self) -> None:
        from scripts.run_coverage_aware_reward_refinement import (
            run_coverage_aware_reward_refinement,
        )

        rows = [
            self._row(
                "selected_ppo_candidate",
                0,
                delta=0.10,
                value=1.0,
                information_gain=0.10,
                path_cost=7.0,
                risk=3.0,
                energy=2.0,
            ),
            self._row("teacher", 0, delta=0.05, value=1.0, information_gain=0.10),
        ]
        self._write_inputs(rows, zero_step_cost_fields=True)

        summary = run_coverage_aware_reward_refinement(
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        selected = next(
            row
            for row in self._read_jsonl(self.output_root / "reward-component-audit.jsonl")
            if row["actor"] == "selected_ppo_candidate"
        )
        self.assertLess(selected["reward_components"]["path_cost_penalty"], 0.0)
        self.assertLess(selected["reward_components"]["risk_penalty"], 0.0)
        self.assertLess(selected["reward_components"]["energy_penalty"], 0.0)
        self.assertEqual(selected["source_fields"]["path_cost_penalty"], "observation.candidate_features.path_cost")
        self.assertEqual(selected["source_fields"]["risk_penalty"], "observation.candidate_features.risk")
        self.assertEqual(selected["source_fields"]["energy_penalty"], "observation.candidate_features.energy_cost")

    def test_fallback_gain_claimed_as_policy_is_penalized_and_rejected(self) -> None:
        from scripts.run_coverage_aware_reward_refinement import (
            run_coverage_aware_reward_refinement,
        )

        rows = [
            self._row(
                "source_fallback",
                0,
                delta=0.20,
                value=1.0,
                information_gain=0.10,
                choice_source="source_fallback",
                claimed_actor="policy",
            ),
            self._row("teacher", 0, delta=0.05, value=1.0, information_gain=0.10),
        ]
        self._write_inputs(rows, fallback_claimed_policy_count=1)

        summary = run_coverage_aware_reward_refinement(
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("fallback_policy_gain_contamination", summary["reason_codes"])
        fallback_row = next(row for row in self._read_jsonl(self.output_root / "reward-component-audit.jsonl") if row["actor"] == "source_fallback")
        self.assertLess(fallback_row["reward_components"]["fallback_penalty"], 0.0)

    def test_controlled_regression_gets_penalty_and_blocks_refinement(self) -> None:
        from scripts.run_coverage_aware_reward_refinement import (
            run_coverage_aware_reward_refinement,
        )

        rows = [
            self._row(
                "selected_ppo_candidate",
                0,
                delta=0.10,
                value=1.0,
                information_gain=0.10,
                controlled_reasons=["risk_regression"],
            ),
            self._row("teacher", 0, delta=0.05, value=1.0, information_gain=0.10),
        ]
        self._write_inputs(rows, controlled_regression_count=1)

        summary = run_coverage_aware_reward_refinement(
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("controlled_regression_present", summary["reason_codes"])
        selected = next(row for row in self._read_jsonl(self.output_root / "reward-component-audit.jsonl") if row["actor"] == "selected_ppo_candidate")
        self.assertLess(selected["reward_components"]["controlled_regression_penalty"], 0.0)

    def test_teacher_skill_retention_bonus_requires_action_evidence(self) -> None:
        from scripts.run_coverage_aware_reward_refinement import (
            run_coverage_aware_reward_refinement,
        )

        rows = [
            self._row("selected_ppo_candidate", 0, delta=0.10, value=1.0, information_gain=0.10),
            self._row(
                "selected_ppo_candidate",
                1,
                delta=0.10,
                value=1.0,
                information_gain=0.10,
                teacher_action_index=1,
                controlled_action_index=0,
                controlled_detail="policy_safe_disagreement",
            ),
            self._row("teacher", 0, delta=0.05, value=1.0, information_gain=0.10),
        ]
        self._write_inputs(rows, selected_teacher_equivalent=False)

        summary = run_coverage_aware_reward_refinement(
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        audit_rows = self._read_jsonl(self.output_root / "reward-component-audit.jsonl")
        aligned = next(row for row in audit_rows if row["context_id"] == "selected_ppo_candidate-context-0")
        safe_disagreement = next(row for row in audit_rows if row["context_id"] == "selected_ppo_candidate-context-1")
        self.assertGreater(aligned["reward_components"]["teacher_skill_retention_bonus"], 0.0)
        self.assertGreater(safe_disagreement["reward_components"]["teacher_skill_retention_bonus"], 0.0)
        self.assertLess(
            safe_disagreement["reward_components"]["teacher_skill_retention_bonus"],
            aligned["reward_components"]["teacher_skill_retention_bonus"],
        )

    def test_teacher_equivalent_candidate_is_not_marked_as_performance_improved(self) -> None:
        from scripts.run_coverage_aware_reward_refinement import (
            run_coverage_aware_reward_refinement,
        )

        rows = [
            self._row("selected_ppo_candidate", 0, delta=0.10, value=1.0, information_gain=0.10),
            self._row("selected_ppo_candidate", 1, delta=0.08, value=1.0, information_gain=0.10),
        ]
        self._write_inputs(
            rows,
            selected_teacher_equivalent=True,
            performance_reason_codes=["coverage_performance_not_improved"],
        )

        summary = run_coverage_aware_reward_refinement(
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["selected_teacher_equivalent"])
        self.assertFalse(summary["selected_candidate_performance_improved"])
        comparison = self._load_json(self.output_root / "reward-rescore-comparison.json")
        self.assertEqual(comparison["coverage_aware_reward_improvement"], 0.0)

    def _write_inputs(
        self,
        rows: list[dict],
        *,
        expected_actual_confusion_count: int = 0,
        fallback_claimed_policy_count: int = 0,
        controlled_regression_count: int = 0,
        selected_teacher_equivalent: bool = False,
        performance_reason_codes: list[str] | None = None,
        zero_step_cost_fields: bool = False,
    ) -> None:
        self._write_json(
            self.formal_root / "formal-ppo-training-run-summary.json",
            {
                "schema_version": "guarded-formal-ppo-training-run-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "seed_count": 5,
                "optimizer_train_transition_count": 684,
                "teacher_agreement_rate": 1.0,
                "controlled_regression_count": controlled_regression_count,
                "performance_claimed": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
            },
        )
        collector_root = self.formal_root / "seed-00" / "collector"
        self._write_jsonl(
            self.formal_root / "formal-ppo-training-run-seed-summaries.jsonl",
            [
                {
                    "seed": 0,
                    "status": "passed",
                    "collector_root": str(collector_root),
                    "controlled_regression_count": controlled_regression_count,
                }
            ],
        )
        self._write_jsonl(
            collector_root / "ppo-rollout-episodes.jsonl",
            [
                {
                    "metrics": {"final_coverage_rate": 0.0, "cumulative_coverage_rate_delta": 0.0},
                    "transitions": [
                        {
                            "reward": 1.0,
                            "info": {
                                "context_id": "collector-teacher-following",
                                "controlled_choice_source": "policy",
                                "controlled_choice_detail": "policy_teacher_aligned",
                                "coverage_rate_delta": 0.0,
                            },
                        }
                    ],
                }
            ],
        )
        self._write_json(
            self.replay_root / "formal-ppo-post-training-stability-replay-summary.json",
            {
                "schema_version": "guarded-formal-ppo-post-training-stability-replay-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "teacher_agreement_rate": 1.0,
                "controlled_regression_count": controlled_regression_count,
                "runs_new_ppo_update": False,
            },
        )
        self._write_json(
            self.selected_root / "selected-formal-ppo-candidate-promotion-preflight-summary.json",
            {
                "schema_version": "selected-formal-ppo-candidate-promotion-preflight-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "selected_seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "teacher_agreement_rate": 1.0,
                "controlled_regression_count": controlled_regression_count,
                "multihorizon_steps": str(self.selected_root / "multihorizon-shadow-rollout-steps.jsonl"),
                "runs_new_ppo_update": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
            },
        )
        self._write_jsonl(
            self.selected_root / "multihorizon-shadow-rollout-steps.jsonl",
            [self._shadow_step(row, zero_step_cost_fields=zero_step_cost_fields) for row in rows],
        )
        nonzero_count = sum(
            1
            for row in rows
            if isinstance(row.get("coverage_rate_delta"), (int, float)) and row["coverage_rate_delta"] != 0
        )
        signal_passed = expected_actual_confusion_count == 0 and fallback_claimed_policy_count == 0
        self._write_json(
            self.signal_root / "exploration-coverage-signal-audit-summary.json",
            {
                "schema_version": "exploration-coverage-signal-audit-summary/v1",
                "status": "passed" if signal_passed else "failed",
                "reason_codes": [],
                "coverage_signal_status": "passed" if signal_passed else "failed",
                "actual_coverage_gain_source": "path_feedback",
                "nonzero_actual_coverage_delta_count": nonzero_count,
                "coverage_delta_default_zero_count": 0,
                "expected_actual_coverage_confusion_count": expected_actual_confusion_count,
                "fallback_coverage_gain_claimed_as_policy_gain_count": fallback_claimed_policy_count,
                "controlled_regression_count": controlled_regression_count,
                "coverage_delta_audit": str(self.signal_root / "coverage-delta-audit.jsonl"),
            },
        )
        delta_rows = [dict(row) for row in rows]
        if zero_step_cost_fields:
            for row in delta_rows:
                row.pop("path_cost", None)
                row.pop("risk", None)
                row.pop("energy_cost", None)
        self._write_jsonl(self.signal_root / "coverage-delta-audit.jsonl", delta_rows)
        selected_rows = [row for row in rows if row["canonical_coverage_actor"] == "selected_ppo_candidate"]
        teacher_rows = [row for row in rows if row["canonical_coverage_actor"] == "teacher"]
        if selected_teacher_equivalent and not teacher_rows:
            teacher_rows = [dict(row, canonical_coverage_actor="teacher") for row in selected_rows]
        selected_metric = self._metric("selected_ppo_candidate", selected_rows)
        teacher_metric = self._metric("teacher", teacher_rows)
        reason_codes = performance_reason_codes if performance_reason_codes is not None else ["coverage_performance_not_improved"]
        self._write_json(
            self.performance_root / "exploration-coverage-performance-evaluation-summary.json",
            {
                "schema_version": "exploration-coverage-performance-evaluation-summary/v1",
                "status": "passed" if not reason_codes else "failed",
                "reason_codes": reason_codes,
                "coverage_performance_status": "passed" if not reason_codes else "failed",
                "next_required_change": "coverage_aware_reward_refinement",
                "selected_actor": "selected_ppo_candidate",
                "best_baseline_actor": "teacher",
                "selected_metrics": selected_metric,
                "best_baseline_metrics": teacher_metric,
                "coverage_return_improvement": round(
                    selected_metric["coverage_return"] - teacher_metric["coverage_return"],
                    12,
                ),
                "cumulative_coverage_rate_delta_improvement": round(
                    selected_metric["cumulative_coverage_rate_delta"]
                    - teacher_metric["cumulative_coverage_rate_delta"],
                    12,
                ),
                "valuable_area_covered_improvement": round(
                    selected_metric["valuable_area_covered"] - teacher_metric["valuable_area_covered"],
                    12,
                ),
                "controlled_regression_count": controlled_regression_count,
                "runs_new_ppo_update": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "formal_release_claimed": False,
                "metric_table": str(self.performance_root / "coverage-performance-metric-table.jsonl"),
                "comparison_audit": str(self.performance_root / "coverage-performance-comparison-audit.json"),
            },
        )
        self._write_jsonl(self.performance_root / "coverage-performance-metric-table.jsonl", [selected_metric, teacher_metric])
        self._write_json(
            self.performance_root / "coverage-performance-comparison-audit.json",
            {
                "schema_version": "coverage-performance-comparison-audit/v1",
                "selected_actor": "selected_ppo_candidate",
                "best_baseline_actor": "teacher",
                "baseline_actors": ["teacher"],
                "selected_teacher_equivalent": selected_teacher_equivalent,
                "coverage_return_improvement": round(
                    selected_metric["coverage_return"] - teacher_metric["coverage_return"],
                    12,
                ),
                "cumulative_coverage_rate_delta_improvement": round(
                    selected_metric["cumulative_coverage_rate_delta"]
                    - teacher_metric["cumulative_coverage_rate_delta"],
                    12,
                ),
                "valuable_area_covered_improvement": round(
                    selected_metric["valuable_area_covered"] - teacher_metric["valuable_area_covered"],
                    12,
                ),
            },
        )

    def _row(
        self,
        actor: str,
        step_index: int,
        *,
        delta: float | None,
        expected_delta: float = 0.0,
        path_cost: float = 2.0,
        risk: float = 0.2,
        energy: float = 0.5,
        value: float = 1.0,
        information_gain: float = 0.1,
        value_missing: bool = False,
        information_missing: bool = False,
        claimed_actor: str | None = None,
        choice_source: str | None = None,
        controlled_reasons: list[str] | None = None,
        teacher_action_index: int = 0,
        controlled_action_index: int = 0,
        controlled_detail: str = "policy_teacher_aligned",
    ) -> dict:
        episode_id = f"{actor}-episode"
        cumulative = delta if delta is not None else None
        return {
            "schema_version": "exploration-coverage-delta-audit-row/v1",
            "audit_index": len(actor) + step_index,
            "episode_id": episode_id,
            "source_episode_id": episode_id,
            "step_index": step_index,
            "source_step_index": step_index,
            "context_id": f"{actor}-context-{step_index}",
            "scenario_id": f"scenario-{step_index}",
            "scenario_family": "family-a",
            "split": "train",
            "controlled_choice_source": choice_source or ("policy" if actor == "selected_ppo_candidate" else actor),
            "controlled_choice_detail": controlled_detail,
            "controlled_action_index": controlled_action_index,
            "teacher_action_index": teacher_action_index,
            "canonical_coverage_actor": actor,
            "coverage_gain_claimed_actor": claimed_actor or actor,
            "ppo_trainable": actor == "selected_ppo_candidate",
            "initial_coverage_rate": 0.0,
            "final_coverage_rate": delta,
            "coverage_rate_delta": delta,
            "cumulative_coverage_rate_delta": cumulative,
            "expected_coverage_rate_delta": expected_delta,
            "actual_coverage_gain_source": "path_feedback" if delta is not None else "missing",
            "actual_delta_present": delta is not None,
            "actual_delta_finite": delta is not None,
            "actual_delta_nonzero": bool(delta),
            "actual_delta_default_zero": delta == 0.0,
            "expected_delta_nonzero": expected_delta != 0.0,
            "controlled_regression_reason_codes": controlled_reasons or [],
            "path_cost": path_cost,
            "risk": risk,
            "energy_cost": energy,
            "value": value,
            "information_gain": information_gain,
            "value_missing": value_missing,
            "information_gain_missing": information_missing,
        }

    def _shadow_step(self, row: dict, *, zero_step_cost_fields: bool) -> dict:
        return {
            "episode_id": row["source_episode_id"],
            "shadow_episode_id": row["episode_id"],
            "step_index": row["source_step_index"],
            "shadow_step_index": row["step_index"],
            "context_id": row["context_id"],
            "scenario_id": row["scenario_id"],
            "scenario_family": row["scenario_family"],
            "split": row["split"],
            "controlled_choice_source": row["controlled_choice_source"],
            "controlled_choice_detail": row["controlled_choice_detail"],
            "controlled_action_index": row["controlled_action_index"],
            "teacher_action_index": row["teacher_action_index"],
            "policy_takes_control": row["canonical_coverage_actor"] == "selected_ppo_candidate",
            "path_cost_delta": 0.0 if zero_step_cost_fields else row["path_cost"],
            "risk_delta": 0.0 if zero_step_cost_fields else row["risk"],
            "energy_cost_delta": 0.0 if zero_step_cost_fields else row["energy_cost"],
            "controlled_regression_reason_codes": row["controlled_regression_reason_codes"],
            "reward": 1.0,
            "reward_components": {"teacher_following_bonus": 1.0},
            "observation": {
                "candidate_feature_names": ["value", "risk", "path_cost", "energy_cost", "information_gain"],
                "candidate_features": [
                    [
                        row["value"],
                        row["risk"],
                        row["path_cost"],
                        row["energy_cost"],
                        row["information_gain"],
                    ]
                ],
                "candidate_missing_indicator_names": [
                    "value_missing",
                    "risk_missing",
                    "path_cost_missing",
                    "energy_cost_missing",
                    "information_gain_missing",
                ],
                "candidate_missing_indicators": [
                    [
                        1.0 if row["value_missing"] else 0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0 if row["information_gain_missing"] else 0.0,
                    ]
                ],
                "action_mask": [True],
            },
        }

    def _metric(self, actor: str, rows: list[dict]) -> dict:
        total_delta = sum(row["coverage_rate_delta"] or 0.0 for row in rows)
        total_value = sum((row["coverage_rate_delta"] or 0.0) * row["value"] for row in rows)
        return {
            "schema_version": "coverage-performance-metric-row/v1",
            "actor": actor,
            "row_count": len(rows),
            "actual_coverage_row_count": len(rows),
            "coverage_return": round(total_delta, 12),
            "cumulative_coverage_rate_delta": round(total_delta, 12),
            "valuable_area_covered": round(total_value, 12),
            "path_cost": sum(row["path_cost"] for row in rows),
            "risk": sum(row["risk"] for row in rows),
            "energy_cost": sum(row["energy_cost"] for row in rows),
            "fallback_rate": 0.0,
            "teacher_agreement_rate": 1.0,
            "controlled_regression_count": sum(
                1 for row in rows if row["controlled_regression_reason_codes"]
            ),
        }

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
