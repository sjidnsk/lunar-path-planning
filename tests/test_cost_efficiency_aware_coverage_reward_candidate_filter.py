import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class CostEfficiencyAwareCoverageRewardCandidateFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts = str(self.repo_root / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="cost-efficiency-filter-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_cost_efficiency_batch_keeps_four_families_and_demotes_low_gain_high_cost_rows(self) -> None:
        from scripts.run_cost_efficiency_aware_coverage_reward_candidate_filter import (
            build_cost_efficiency_aware_batch,
        )

        pairs = []
        counterfactual_rows = []
        families = (
            "low_observation_count",
            "mixed_risk",
            "rim_or_steep_slope",
            "smooth_high_confidence",
        )
        for family_index, family in enumerate(families):
            for row_index in range(3):
                row = self._pair(
                    family=family,
                    index=family_index * 10 + row_index,
                    coverage_advantage=0.12 if row_index != 2 else 0.004,
                    path_delta=2.0 if row_index != 2 else 95.0,
                    risk_delta=0.0 if row_index != 2 else 0.02,
                    energy_delta=0.0 if row_index != 2 else 120.0,
                )
                pairs.append(row)
                counterfactual_rows.append(self._counterfactual(row))

        batch = build_cost_efficiency_aware_batch(
            pair_rows=pairs,
            counterfactual_rows=counterfactual_rows,
            minimum_trainable_pair_count=8,
            minimum_family_pair_count=2,
        )

        self.assertEqual(batch["status"], "passed")
        self.assertEqual(batch["trainable_pair_count"], 8)
        self.assertEqual(batch["trainable_family_count"], 4)
        self.assertEqual(
            batch["family_trainable_counts"],
            {
                "low_observation_count": 2,
                "mixed_risk": 2,
                "rim_or_steep_slope": 2,
                "smooth_high_confidence": 2,
            },
        )
        self.assertGreater(batch["diagnostic_pair_count"], 0)
        self.assertTrue(
            any("low_gain_high_cost" in row["row_reason_codes"] for row in batch["diagnostic_rows"])
        )
        for row in batch["trainable_pairs"]:
            self.assertGreater(row["cost_efficiency_ppo_advantage"], 0.0)
            self.assertLessEqual(row["cost_efficiency_ppo_advantage"], row["coverage_advantage"])
            components = row["cost_efficiency_reward_components"]
            self.assertGreater(components["coverage_gain_bonus"], 0.0)
            self.assertLessEqual(components["path_cost_efficiency_penalty"], 0.0)
            self.assertIn(row["cost_efficiency_sample_class"], {"strong", "weighted"})

    def test_runner_writes_summary_audit_filtered_batch_and_comparable_metric_input(self) -> None:
        import scripts.run_cost_efficiency_aware_coverage_reward_candidate_filter as module

        safe_better_root = self.temp_dir / "safe-better"
        expanded_root = self.temp_dir / "expanded-four-family"
        output_root = self.temp_dir / "output"
        selected_root = self.temp_dir / "selected"
        for path in (safe_better_root, expanded_root, selected_root):
            path.mkdir(parents=True)

        pairs = []
        counterfactual_rows = []
        for family_index, family in enumerate(
            (
                "low_observation_count",
                "mixed_risk",
                "rim_or_steep_slope",
                "smooth_high_confidence",
            )
        ):
            for row_index in range(2):
                row = self._pair(
                    family=family,
                    index=family_index * 10 + row_index,
                    coverage_advantage=0.2,
                    path_delta=1.0,
                    risk_delta=0.0,
                    energy_delta=0.0,
                )
                pairs.append(row)
                counterfactual_rows.append(self._counterfactual(row))
        self._write_json(
            safe_better_root / "safe-better-pair-expansion-summary.json",
            {
                "schema_version": "safe-better-pair-expansion-summary/v1",
                "status": "passed",
                "safe_better_than_teacher_candidate_count": len(pairs),
                "safe_better_than_teacher_family_count": 4,
                "missing_counterfactual_source_count": 0,
                "fallback_gain_contamination_count": 0,
                "controlled_regression_count": 0,
                "performance_claimed": False,
            },
        )
        self._write_jsonl(safe_better_root / "expanded-safe-better-pairs.jsonl", pairs)
        self._write_jsonl(
            safe_better_root / "expanded-counterfactual-coverage-rollouts.jsonl",
            counterfactual_rows,
        )
        self._write_json(
            expanded_root / "expanded-four-family-refined-coverage-driven-ppo-improvement-summary.json",
            {
                "schema_version": "expanded-four-family-refined-coverage-driven-ppo-improvement-summary/v1",
                "status": "failed",
                "reason_codes": ["coverage_efficiency_regression"],
                "next_required_change": "tune_cost_efficiency_aware_reward_or_candidate_filter",
                "refined_trainable_transitions": str(expanded_root / "refined.jsonl"),
                "guard_replay_audit": str(expanded_root / "replay.json"),
                "performance_metric_table": str(expanded_root / "metric.jsonl"),
                "compatible_coverage_driven_root": str(expanded_root / "coverage-input"),
                "coverage_aware_batch_episodes": str(expanded_root / "episodes.jsonl"),
                "base_candidate_root": str(selected_root),
                "safe_better_training_pair_count": 723,
                "safe_better_training_family_count": 4,
                "coverage_efficiency_regression": True,
                "performance_claimed": False,
            },
        )
        self._write_jsonl(expanded_root / "refined.jsonl", [{"schema_version": "fixture"}])
        self._write_json(expanded_root / "replay.json", {"schema_version": "fixture"})
        self._write_jsonl(expanded_root / "metric.jsonl", [{"actor": "post_improvement_ppo"}])
        (expanded_root / "coverage-input").mkdir()

        original = module.run_refined_coverage_driven_ppo_improvement_run

        def fake_refined_run(**kwargs):
            stage5a2_root = Path(kwargs["stage5a2_root"])
            performance_root = Path(kwargs["coverage_performance_root"])
            self.assertTrue((stage5a2_root / "candidate-level-coverage-overlay.jsonl").is_file())
            compatible_counterfactual_rows = self._read_jsonl(
                stage5a2_root / "counterfactual-coverage-rollouts.jsonl"
            )
            self.assertTrue(all(row.get("source_path") for row in compatible_counterfactual_rows))
            self.assertTrue(
                all(row.get("cost_efficiency_ppo_advantage", 0.0) > 0.0 for row in compatible_counterfactual_rows)
            )
            metric_rows = self._read_jsonl(performance_root / "coverage-performance-metric-table.jsonl")
            self.assertEqual({row["actor"] for row in metric_rows}, {"teacher", "pre_improvement_selected_ppo"})
            self.assertGreater(metric_rows[0]["coverage_gain_per_path_cost"], 0.0)
            return {
                "status": "passed",
                "reason_codes": [],
                "next_required_change": "shadow_canary_release_performance_validation_preflight",
                "summary": str(output_root / "refined-summary.json"),
                "compatibility_summary": str(output_root / "coverage-driven-ppo-improvement-run-summary.json"),
                "refined_trainable_transitions": str(output_root / "refined-coverage-ppo-batch" / "refined-trainable-transitions.jsonl"),
                "advantage_margin_audit": str(output_root / "advantage-margin-audit.jsonl"),
                "ppo_update_summary": str(output_root / "coverage-driven-ppo-update-summary.json"),
                "guard_replay_audit": str(output_root / "coverage-driven-ppo-replay-audit.json"),
                "performance_metric_table": str(output_root / "refined-coverage-driven-ppo-performance-metric-table.jsonl"),
                "stage5a_rerun_summary": str(output_root / "stage5a-rerun-summary.json"),
                "safe_better_training_pair_count": 8,
                "counterfactual_advantage_nonzero_count": 8,
                "policy_argmax_changed_count": 3,
                "coverage_return_improvement": 1.25,
                "cumulative_coverage_rate_delta_improvement": 1.5,
                "valuable_area_covered_improvement": 0.7,
                "coverage_efficiency_regression": False,
                "fallback_rate": 0.25,
                "controlled_regression_count": 0,
                "fallback_gain_contamination_count": 0,
                "runs_new_ppo_update": True,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
            }

        module.run_refined_coverage_driven_ppo_improvement_run = fake_refined_run
        try:
            summary = module.run_cost_efficiency_aware_coverage_reward_candidate_filter(
                safe_better_root=safe_better_root,
                expanded_four_family_root=expanded_root,
                formal_training_root=self.temp_dir / "formal",
                post_training_replay_root=self.temp_dir / "replay",
                selected_candidate_root=selected_root,
                coverage_signal_root=self.temp_dir / "signal",
                reward_refinement_root=self.temp_dir / "reward",
                output_root=output_root,
                repo_root=self.repo_root,
                minimum_trainable_pair_count=8,
                minimum_family_pair_count=2,
            )
        finally:
            module.run_refined_coverage_driven_ppo_improvement_run = original

        self.assertEqual(summary["status"], "passed")
        self.assertFalse(summary["coverage_efficiency_regression"])
        self.assertEqual(summary["safe_better_training_family_count"], 4)
        self.assertEqual(summary["trainable_pair_count"], 8)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])
        for filename in (
            "cost-efficiency-aware-coverage-reward-candidate-filter-summary.json",
            "cost-efficiency-efficiency-audit.jsonl",
            "cost-efficiency-filtered-batch.jsonl",
            "cost-efficiency-advantage-audit.jsonl",
            "cost-efficiency-aware-coverage-reward-candidate-filter-report.md",
        ):
            self.assertTrue((output_root / filename).is_file(), filename)

    def test_cost_efficiency_reward_component_merge_preserves_policy_value(self) -> None:
        import scripts.run_refined_coverage_driven_ppo_improvement_run as module

        original = module._policy_log_prob_and_value
        module._policy_log_prob_and_value = lambda **kwargs: (-0.75, 1.25)
        try:
            transition = module._refined_transition(
                old_transition={
                    "observation": {
                        "candidate_feature_names": ["expected_coverage_rate_delta", "value", "risk", "path_cost", "energy_cost"],
                        "candidate_features": [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
                        "candidate_missing_indicator_names": [],
                        "candidate_missing_indicators": [[], []],
                        "action_mask": [True, True],
                    },
                    "done": True,
                    "info": {"context_id": "ctx", "episode_id": "ep", "step_index": 0},
                },
                candidate={
                    "context_id": "ctx",
                    "episode_id": "ep",
                    "step_index": 0,
                    "action_index": 1,
                    "teacher_action_index": 0,
                    "expected_coverage_rate_delta": 0.3,
                    "information_gain": 0.1,
                    "value": 0.2,
                    "path_cost": 2.0,
                    "risk": 0.1,
                    "energy_cost": 3.0,
                    "expanded_family_weight": 1.0,
                    "cost_efficiency_reward_components": {
                        "path_cost_efficiency_penalty": -0.02,
                        "controlled_regression_penalty": 0.0,
                    },
                },
                teacher={
                    "action_index": 0,
                    "expected_coverage_rate_delta": 0.1,
                    "information_gain": 0.1,
                    "value": 0.1,
                    "path_cost": 1.0,
                    "risk": 0.1,
                    "energy_cost": 1.0,
                },
                counterfactual={"expected_coverage_rate_delta": 0.3},
                action_index=1,
                advantage=0.2,
                coverage_rank_margin=0.2,
                base_candidate_root=self.temp_dir,
                repo_root=self.repo_root,
                ppo_advantage_scale=10.0,
                ppo_advantage_signal=0.15,
            )
        finally:
            module._policy_log_prob_and_value = original

        self.assertEqual(transition["log_prob"], -0.75)
        self.assertEqual(transition["value"], 1.25)
        self.assertEqual(transition["ppo_return"], 2.75)

    def _pair(
        self,
        *,
        family: str,
        index: int,
        coverage_advantage: float,
        path_delta: float,
        risk_delta: float,
        energy_delta: float,
    ) -> dict:
        teacher_coverage = 0.05
        teacher_path = 20.0
        teacher_risk = 0.2
        teacher_energy = 100.0
        return {
            "schema_version": "safe-better-pair-expansion-row/v1",
            "context_id": f"context-{index}",
            "episode_id": f"episode-{index}",
            "step_index": index,
            "scenario_id": f"scenario-{index}",
            "scenario_family": family,
            "split": "train",
            "ppo_trainable": True,
            "coverage_source_available": True,
            "safe_better_than_teacher_candidate": True,
            "fallback_like": False,
            "guard_rejected": False,
            "controlled_regression_reason_codes": [],
            "candidate_action_index": 2,
            "teacher_action_index": 1,
            "candidate_cell": [index, 2],
            "teacher_candidate_cell": [index, 1],
            "expected_coverage_rate_delta": teacher_coverage + coverage_advantage,
            "teacher_expected_coverage_rate_delta": teacher_coverage,
            "expected_new_coverage_area": 10.0 + coverage_advantage,
            "teacher_expected_new_coverage_area": 10.0,
            "information_gain": 0.01,
            "teacher_information_gain": 0.01,
            "valuable_coverage_proxy": 0.4 + coverage_advantage,
            "teacher_valuable_coverage_proxy": 0.4,
            "coverage_advantage": coverage_advantage,
            "valuable_coverage_advantage": coverage_advantage,
            "path_cost": teacher_path + path_delta,
            "teacher_path_cost": teacher_path,
            "path_cost_delta": path_delta,
            "risk": teacher_risk + risk_delta,
            "teacher_risk": teacher_risk,
            "risk_delta": risk_delta,
            "energy_cost": teacher_energy + energy_delta,
            "teacher_energy_cost": teacher_energy,
            "energy_delta": energy_delta,
            "match_method": "fixture_sidecar",
            "source_path": str(self.temp_dir / "source.json"),
            "source_confidence": 1.0,
        }

    def _counterfactual(self, row: dict) -> dict:
        return {
            "schema_version": "expanded-counterfactual-coverage-row/v1",
            "context_id": row["context_id"],
            "episode_id": row["episode_id"],
            "step_index": row["step_index"],
            "scenario_id": row["scenario_id"],
            "scenario_family": row["scenario_family"],
            "action_index": row["candidate_action_index"],
            "expected_coverage_rate_delta": row["expected_coverage_rate_delta"],
            "expected_new_coverage_area": row["expected_new_coverage_area"],
            "valuable_coverage_proxy": row["valuable_coverage_proxy"],
            "information_gain": row["information_gain"],
            "coverage_source_available": True,
            "match_method": row["match_method"],
            "source_path": row["source_path"],
            "source_confidence": 1.0,
        }

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
