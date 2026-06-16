import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class FamilyBalancedCoverageDrivenPpoRerunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts = str(self.repo_root / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="family-balanced-rerun-"))
        self.gap_root = self.temp_dir / "gap"
        self.coverage_root = self.temp_dir / "coverage"
        self.formal_root = self.temp_dir / "formal"
        self.replay_root = self.temp_dir / "replay"
        self.selected_root = self.temp_dir / "selected"
        self.signal_root = self.temp_dir / "signal"
        self.performance_root = self.temp_dir / "performance"
        self.reward_root = self.temp_dir / "reward"
        self.output_root = self.temp_dir / "output"
        self.base_candidate_root = self.temp_dir / "base-candidate"
        for path in (
            self.gap_root,
            self.coverage_root,
            self.formal_root,
            self.replay_root,
            self.selected_root,
            self.signal_root,
            self.performance_root,
            self.reward_root,
            self.base_candidate_root,
        ):
            path.mkdir(parents=True)
        self._write_docs()
        self._write_support_summaries()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_runner_materializes_compatible_inputs_synthetic_old_transitions_and_accepts_refined_pass(self) -> None:
        import scripts.run_expanded_four_family_refined_coverage_driven_ppo_improvement as expanded
        import scripts.run_family_balanced_coverage_driven_ppo_rerun as module

        pairs, counterfactual_rows = self._write_gap_inputs(low_observation_count=36)
        self._write_old_coverage_inputs(existing_pair=pairs[0])

        original_refined = module.run_refined_coverage_driven_ppo_improvement_run
        original_policy = expanded._policy_log_prob_and_value

        def fake_policy(**kwargs):
            action_index = int(kwargs["action_index"])
            return -0.25 - action_index, 0.5 + action_index

        def fake_refined(**kwargs):
            stage5a2_root = Path(kwargs["stage5a2_root"])
            coverage_driven_root = Path(kwargs["coverage_driven_root"])
            output_root = Path(kwargs["output_root"])
            overlay_rows = self._read_jsonl(stage5a2_root / "candidate-level-coverage-overlay.jsonl")
            counter_rows = self._read_jsonl(stage5a2_root / "counterfactual-coverage-rollouts.jsonl")
            episodes = self._read_jsonl(
                coverage_driven_root / "coverage-aware-ppo-batch" / "ppo-rollout-episodes.jsonl"
            )
            transitions = [transition for episode in episodes for transition in episode["transitions"]]
            self.assertEqual(len(counter_rows), len(pairs))
            self.assertEqual({row["scenario_family"] for row in counter_rows}, self._families())
            self.assertTrue(any(row.get("family_balanced_ppo_advantage", 0.0) > 0.0 for row in overlay_rows))
            self.assertTrue(any(transition["info"].get("synthetic_old_transition") for transition in transitions))
            self.assertTrue(
                all(
                    math.isfinite(float(transition["log_prob"])) and math.isfinite(float(transition["value"]))
                    for transition in transitions
                )
            )
            self._write_json(output_root / "coverage-driven-ppo-replay-audit.json", {"status": "passed"})
            return {
                "status": "passed",
                "reason_codes": [],
                "summary": str(output_root / "refined-coverage-driven-ppo-improvement-run-summary.json"),
                "compatibility_summary": str(output_root / "coverage-driven-ppo-improvement-run-summary.json"),
                "refined_trainable_transition_count": len(pairs),
                "safe_better_training_pair_count": len(pairs),
                "safe_better_training_family_count": 4,
                "counterfactual_advantage_nonzero_count": len(pairs),
                "policy_argmax_changed_count": 7,
                "coverage_return_improvement": 3.5,
                "cumulative_coverage_rate_delta_improvement": 4.0,
                "valuable_area_covered_improvement": 1.2,
                "coverage_efficiency_regression": False,
                "fallback_rate": 0.1,
                "controlled_regression_count": 0,
                "fallback_gain_contamination_count": 0,
                "ppo_update_status": "passed",
                "guard_replay_audit": str(output_root / "coverage-driven-ppo-replay-audit.json"),
                "checkpoint_path": str(output_root / "coverage-driven-experimental-policy-candidate.pt"),
                "checkpoint_metadata_path": str(output_root / "coverage-driven-experimental-policy-candidate-metadata.json"),
                "runs_new_ppo_update": True,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "performance_claimed": False,
            }

        expanded._policy_log_prob_and_value = fake_policy
        module.run_refined_coverage_driven_ppo_improvement_run = fake_refined
        try:
            summary = module.run_family_balanced_coverage_driven_ppo_rerun(
                family_balanced_gap_root=self.gap_root,
                coverage_driven_root=self.coverage_root,
                formal_training_root=self.formal_root,
                post_training_replay_root=self.replay_root,
                selected_candidate_root=self.selected_root,
                coverage_signal_root=self.signal_root,
                coverage_performance_root=self.performance_root,
                reward_refinement_root=self.reward_root,
                output_root=self.output_root,
                repo_root=self.temp_dir,
                minimum_low_observation_count=32,
            )
        finally:
            module.run_refined_coverage_driven_ppo_improvement_run = original_refined
            expanded._policy_log_prob_and_value = original_policy

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["rerun_verdict"], "eligible_for_family_balanced_shadow_canary_preflight")
        self.assertTrue(summary["family_balanced_coverage_driven_ppo_rerun_passed"])
        self.assertTrue(summary["family_balanced_shadow_canary_preflight_approved"])
        self.assertTrue(summary["family_balanced_input_audit_passed"])
        self.assertTrue(summary["old_transition_materialization_audit_passed"])
        self.assertEqual(summary["ppo_update_status"], "passed")
        self.assertTrue(summary["guard_replay_audit_passed"])
        self.assertGreater(summary["coverage_return_improvement"], 0.0)
        self.assertGreater(summary["cumulative_coverage_rate_delta_improvement"], 0.0)
        self.assertGreater(summary["valuable_area_covered_improvement"], 0.0)
        self.assertFalse(summary["coverage_efficiency_regression"])
        self.assertEqual(summary["safe_better_training_family_count"], 4)
        self.assertGreaterEqual(summary["low_observation_trainable_transition_count"], 32)
        self.assertEqual(summary["next_required_change"], "family_balanced_shadow_canary_preflight")
        self.assertFalse(summary["checkpoint_publication_approved"])
        self.assertFalse(summary["default_policy_replacement_approved"])
        self.assertFalse(summary["real_executor_connection_approved"])
        for filename in (
            "family-balanced-coverage-driven-ppo-rerun-summary.json",
            "family-balanced-ppo-source-ledger.json",
            "family-balanced-compatible-input-audit.json",
            "family-balanced-old-transition-materialization-audit.json",
            "family-balanced-ppo-advantage-audit.jsonl",
            "family-balanced-performance-audit.json",
            "family-balanced-release-boundary-audit.json",
            "family-balanced-coverage-driven-ppo-rerun-rejection-report.json",
            "family-balanced-coverage-driven-ppo-rerun-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_fails_when_gap_closure_not_passed(self) -> None:
        import scripts.run_family_balanced_coverage_driven_ppo_rerun as module

        self._write_gap_inputs(low_observation_count=36, status="failed", reason_codes=["fixture"])
        self._write_old_coverage_inputs()

        summary = module.run_family_balanced_coverage_driven_ppo_rerun(
            family_balanced_gap_root=self.gap_root,
            coverage_driven_root=self.coverage_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            reward_refinement_root=self.reward_root,
            output_root=self.output_root,
            repo_root=self.temp_dir,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("family_balanced_gap_closure_not_passed", summary["reason_codes"])

    def test_fails_when_low_observation_transition_gap_remains(self) -> None:
        import scripts.run_family_balanced_coverage_driven_ppo_rerun as module

        self._write_gap_inputs(low_observation_count=4)
        self._write_old_coverage_inputs()

        summary = module.run_family_balanced_coverage_driven_ppo_rerun(
            family_balanced_gap_root=self.gap_root,
            coverage_driven_root=self.coverage_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            reward_refinement_root=self.reward_root,
            output_root=self.output_root,
            repo_root=self.temp_dir,
            minimum_low_observation_count=32,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("low_observation_transition_gap", summary["reason_codes"])

    def test_maps_refined_ppo_performance_failures_to_rerun_reasons(self) -> None:
        import scripts.run_expanded_four_family_refined_coverage_driven_ppo_improvement as expanded
        import scripts.run_family_balanced_coverage_driven_ppo_rerun as module

        self._write_gap_inputs(low_observation_count=36)
        self._write_old_coverage_inputs()
        original_refined = module.run_refined_coverage_driven_ppo_improvement_run
        original_policy = expanded._policy_log_prob_and_value

        def fake_refined(**kwargs):
            return {
                "status": "failed",
                "reason_codes": ["coverage_efficiency_regression"],
                "refined_trainable_transition_count": 10,
                "safe_better_training_pair_count": 10,
                "safe_better_training_family_count": 4,
                "counterfactual_advantage_nonzero_count": 10,
                "policy_argmax_changed_count": 0,
                "coverage_return_improvement": 0.0,
                "cumulative_coverage_rate_delta_improvement": 0.0,
                "valuable_area_covered_improvement": -1.0,
                "coverage_efficiency_regression": True,
                "fallback_rate": 0.8,
                "controlled_regression_count": 1,
                "fallback_gain_contamination_count": 1,
                "ppo_update_status": "failed",
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            }

        expanded._policy_log_prob_and_value = lambda **kwargs: (-0.1, 0.2)
        module.run_refined_coverage_driven_ppo_improvement_run = fake_refined
        try:
            summary = module.run_family_balanced_coverage_driven_ppo_rerun(
                family_balanced_gap_root=self.gap_root,
                coverage_driven_root=self.coverage_root,
                formal_training_root=self.formal_root,
                post_training_replay_root=self.replay_root,
                selected_candidate_root=self.selected_root,
                coverage_signal_root=self.signal_root,
                coverage_performance_root=self.performance_root,
                reward_refinement_root=self.reward_root,
                output_root=self.output_root,
                repo_root=self.temp_dir,
            )
        finally:
            module.run_refined_coverage_driven_ppo_improvement_run = original_refined
            expanded._policy_log_prob_and_value = original_policy

        self.assertEqual(summary["status"], "failed")
        for reason in (
            "ppo_update_failed",
            "post_update_policy_teacher_equivalent",
            "no_coverage_return_improvement",
            "no_cumulative_coverage_rate_delta_improvement",
            "valuable_coverage_regressed",
            "coverage_efficiency_regression",
            "fallback_dominates",
            "fallback_gain_contamination",
            "controlled_regression_detected",
        ):
            self.assertIn(reason, summary["reason_codes"])

    def _write_docs(self) -> None:
        self._write_text(
            self.temp_dir / "README.md",
            "Family-Balanced Coverage-Driven PPO Rerun v1\n"
            "outputs/path_feedback_batch_family_balanced_coverage_driven_ppo_rerun_v1/\n"
            "family_balanced_shadow_canary_preflight\n"
            "checkpoint_publication_approved=false\n"
            "default_policy_replacement_approved=false\n"
            "real_executor_connection_approved=false\n",
        )
        self._write_text(
            self.temp_dir / "docs" / "算法设计与系统架构报告.md",
            "Family-Balanced Coverage-Driven PPO Rerun v1\n"
            "离线 guarded PPO rerun\n"
            "不发布 checkpoint、不替换 default policy、不连接真实执行器\n",
        )
        self._write_text(
            self.temp_dir
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-06-16-family-balanced-coverage-driven-ppo-rerun.md",
            "Family-Balanced Coverage-Driven PPO Rerun v1\n"
            "family-balanced-coverage-driven-ppo-rerun-summary.json\n"
            "family_balanced_shadow_canary_preflight\n",
        )

    def _write_gap_inputs(
        self,
        *,
        low_observation_count: int,
        status: str = "passed",
        reason_codes: list[str] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        pairs = []
        for family, count in (
            ("low_observation_count", low_observation_count),
            ("mixed_risk", 3),
            ("rim_or_steep_slope", 3),
            ("smooth_high_confidence", 3),
        ):
            for index in range(count):
                pairs.append(self._pair(family=family, index=len(pairs)))
        counterfactual_rows = [self._counterfactual(row) for row in pairs]
        pairs_path = self.gap_root / "family-balanced-safe-better-pairs.jsonl"
        counterfactual_path = self.gap_root / "family-balanced-counterfactual-rollouts.jsonl"
        manifest_path = self.gap_root / "family-balanced-coverage-driven-input-manifest.json"
        self._write_jsonl(pairs_path, pairs)
        self._write_jsonl(counterfactual_path, counterfactual_rows)
        family_counts = self._family_counts(pairs)
        self._write_json(
            manifest_path,
            {
                "status": status,
                "family_balanced_safe_better_pairs": str(pairs_path),
                "family_balanced_counterfactual_rollouts": str(counterfactual_path),
                "family_safe_better_counts": family_counts,
                "next_required_change": "family_balanced_coverage_driven_ppo_rerun",
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.gap_root / "family-balanced-algorithm-gap-closure-summary.json",
            {
                "status": status,
                "reason_codes": reason_codes or [],
                "family_balanced_algorithm_gap_closure_passed": status == "passed",
                "family_balanced_coverage_driven_ppo_rerun_approved": status == "passed",
                "next_required_change": "family_balanced_coverage_driven_ppo_rerun",
                "family_balanced_safe_better_pairs": str(pairs_path),
                "family_balanced_counterfactual_rollouts": str(counterfactual_path),
                "family_balanced_coverage_driven_input_manifest": str(manifest_path),
                "family_safe_better_counts": family_counts,
                "low_observation_count": family_counts.get("low_observation_count", 0),
                "checkpoint_publication_approved": False,
                "default_policy_replacement_approved": False,
                "real_executor_connection_approved": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        return pairs, counterfactual_rows

    def _write_old_coverage_inputs(self, existing_pair: dict | None = None) -> None:
        if existing_pair is None:
            existing_pair = self._pair(family="mixed_risk", index=999)
        transition = {
            "observation": self._observation_payload(),
            "action_index": existing_pair["teacher_action_index"],
            "action_mask": [True for _ in range(8)],
            "log_prob": -0.5,
            "value": 0.25,
            "reward": existing_pair["teacher_expected_coverage_rate_delta"],
            "reward_components": {"teacher_skill_retention_bonus": 0.1},
            "done": True,
            "next_observation": None,
            "info": {
                "context_id": existing_pair["context_id"],
                "episode_id": existing_pair["episode_id"],
                "step_index": existing_pair["step_index"],
                "scenario_id": existing_pair["scenario_id"],
                "scenario_family": existing_pair["scenario_family"],
                "split": "train",
                "controlled_action_index": existing_pair["teacher_action_index"],
                "teacher_action_index": existing_pair["teacher_action_index"],
                "coverage_rate_delta": existing_pair["teacher_expected_coverage_rate_delta"],
                "controlled_regression_reason_codes": [],
                "ppo_trainable": True,
            },
        }
        episodes_path = self.coverage_root / "coverage-aware-ppo-batch" / "ppo-rollout-episodes.jsonl"
        self._write_jsonl(
            episodes_path,
            [{"schema_version": "coverage-aware-ppo-rollout-episode/v1", "episode_id": "old", "transitions": [transition]}],
        )
        self._write_json(
            self.coverage_root / "coverage-driven-ppo-improvement-run-summary.json",
            {
                "status": "failed",
                "reason_codes": ["fixture_old_baseline"],
                "coverage_aware_batch_episodes": str(episodes_path),
                "base_candidate_root": str(self.base_candidate_root),
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )

    def _write_support_summaries(self) -> None:
        self._write_json(self.formal_root / "formal-ppo-training-run-summary.json", {"status": "passed"})
        self._write_json(self.replay_root / "formal-ppo-post-training-stability-replay-summary.json", {"status": "passed"})
        self._write_json(
            self.selected_root / "selected-formal-ppo-candidate-promotion-preflight-summary.json",
            {"status": "passed", "selected_candidate_root": str(self.base_candidate_root)},
        )
        self._write_json(
            self.signal_root / "exploration-coverage-signal-audit-summary.json",
            {
                "status": "passed",
                "coverage_signal_status": "passed",
                "fallback_coverage_gain_claimed_as_policy_gain_count": 0,
            },
        )
        metric_path = self.performance_root / "coverage-performance-metric-table.jsonl"
        self._write_jsonl(metric_path, [{"actor": "teacher", "coverage_return": 1.0}])
        self._write_json(
            self.performance_root / "exploration-coverage-performance-evaluation-summary.json",
            {"status": "passed", "metric_table": str(metric_path), "performance_claimed": False},
        )
        self._write_json(
            self.reward_root / "coverage-aware-reward-refinement-summary.json",
            {
                "status": "passed",
                "reward_refinement_status": "passed",
                "source_field_missing_component_count": 0,
            },
        )

    def _pair(self, *, family: str, index: int) -> dict:
        candidate_action = (index % 5) + 1
        teacher_action = 0
        return {
            "schema_version": "family-balanced-safe-better-pair-row/v1",
            "context_id": f"ctx-{family}-{index}",
            "episode_id": f"episode-{family}-{index}",
            "step_index": index,
            "scenario_id": f"scenario-{family}-{index}",
            "scenario_family": family,
            "split": "train",
            "ppo_trainable": True,
            "coverage_source_available": True,
            "safe_better_than_teacher_candidate": True,
            "fallback_like": False,
            "guard_rejected": False,
            "controlled_regression_reason_codes": [],
            "candidate_action_index": candidate_action,
            "teacher_action_index": teacher_action,
            "candidate_cell": [candidate_action, index],
            "teacher_candidate_cell": [teacher_action, index],
            "expected_coverage_rate_delta": 0.08,
            "teacher_expected_coverage_rate_delta": 0.04,
            "expected_new_coverage_area": 10.0,
            "teacher_expected_new_coverage_area": 5.0,
            "information_gain": 0.03,
            "teacher_information_gain": 0.01,
            "valuable_coverage_proxy": 0.6,
            "teacher_valuable_coverage_proxy": 0.4,
            "coverage_advantage": 0.04,
            "valuable_coverage_advantage": 0.2,
            "path_cost": 12.0,
            "teacher_path_cost": 10.0,
            "path_cost_delta": 2.0,
            "risk": 0.2,
            "teacher_risk": 0.2,
            "risk_delta": 0.0,
            "energy_cost": 100.0,
            "teacher_energy_cost": 100.0,
            "energy_delta": 0.0,
            "match_method": "fixture",
            "source_path": str(self.temp_dir / "source.json"),
            "source_confidence": 1.0,
        }

    def _counterfactual(self, row: dict) -> dict:
        return {
            "schema_version": "family-balanced-counterfactual-coverage-row/v1",
            "context_id": row["context_id"],
            "episode_id": row["episode_id"],
            "step_index": row["step_index"],
            "scenario_id": row["scenario_id"],
            "scenario_family": row["scenario_family"],
            "split": "train",
            "action_index": row["candidate_action_index"],
            "teacher_action_index": row["teacher_action_index"],
            "expected_coverage_rate_delta": row["expected_coverage_rate_delta"],
            "teacher_expected_coverage_rate_delta": row["teacher_expected_coverage_rate_delta"],
            "expected_new_coverage_area": row["expected_new_coverage_area"],
            "valuable_coverage_proxy": row["valuable_coverage_proxy"],
            "information_gain": row["information_gain"],
            "coverage_advantage": row["coverage_advantage"],
            "coverage_source_available": True,
            "safe_better_than_teacher_candidate": True,
            "fallback_like": False,
            "guard_rejected": False,
            "controlled_regression_reason_codes": [],
            "source_path": row["source_path"],
        }

    def _observation_payload(self) -> dict:
        feature_names = [
            "cell_x",
            "cell_y",
            "expected_coverage_rate_delta",
            "expected_new_coverage_area",
            "information_gain",
            "value",
            "risk",
            "path_cost",
            "energy_cost",
        ]
        return {
            "candidate_feature_names": feature_names,
            "candidate_features": [[0.0 for _ in feature_names] for _ in range(8)],
            "candidate_missing_indicator_names": [f"{name}_missing" for name in feature_names],
            "candidate_missing_indicators": [[0.0 for _ in feature_names] for _ in range(8)],
            "candidate_cells": [[i, i] for i in range(8)],
            "global_feature_names": ["coverage_rate", "step_index"],
            "global_features": [0.0, 0.0],
            "action_mask": [True for _ in range(8)],
        }

    def _families(self) -> set[str]:
        return {"low_observation_count", "mixed_risk", "rim_or_steep_slope", "smooth_high_confidence"}

    def _family_counts(self, rows: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            family = row["scenario_family"]
            counts[family] = counts.get(family, 0) + 1
        return dict(sorted(counts.items()))

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
