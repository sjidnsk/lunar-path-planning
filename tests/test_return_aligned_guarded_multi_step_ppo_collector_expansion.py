import json
import sys
import tempfile
import unittest
from pathlib import Path


class ReturnAlignedGuardedMultiStepPpoCollectorExpansionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_dir = str(self.repo_root / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="return-aligned-collector-"))
        self.guarded_root = self.temp_dir / "guarded"
        self.output_root = self.temp_dir / "return-aligned"
        self.collector_root = self.guarded_root / "pilot" / "collector"
        self.collector_root.mkdir(parents=True)
        self._write_guarded_inputs()

    def test_multistep_return_audit_counts_full_horizon_episode_and_strict_trainable_steps(self) -> None:
        from scripts.run_return_aligned_guarded_multi_step_ppo_collector_expansion import (
            run_return_aligned_guarded_multi_step_ppo_collector_expansion,
        )

        summary = run_return_aligned_guarded_multi_step_ppo_collector_expansion(
            guarded_root=self.guarded_root,
            evidence_freeze_summary_path=self.temp_dir / "freeze.json",
            output_root=self.output_root,
            config=self._config(min_episode_count=1, min_transition_count=1),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["schema_version"], "return-aligned-guarded-multistep-collector-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["horizon"], 3)
        self.assertEqual(summary["episode_count"], 3)
        self.assertEqual(summary["step_count"], 9)
        self.assertEqual(summary["trainable_episode_count"], 1)
        self.assertEqual(summary["trainable_transition_count"], 2)
        self.assertEqual(summary["diagnostic_transition_count"], 7)
        self.assertEqual(summary["validation_trainable_count"], 0)
        self.assertEqual(summary["test_trainable_count"], 0)
        self.assertEqual(summary["source_fallback_trainable_count"], 0)
        self.assertEqual(summary["non_finite_reward_count"], 0)
        self.assertEqual(summary["non_finite_return_count"], 0)
        self.assertEqual(summary["non_finite_advantage_count"], 0)
        self.assertEqual(summary["controlled_regression_count"], 0)
        self.assertEqual(summary["teacher_equivalent_episode_count"], 1)
        self.assertEqual(summary["safe_better_episode_count"], 1)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["formal_training_ready_claimed"])

        episodes = self._read_jsonl(self.output_root / "return-aligned-ppo-episodes.jsonl")
        self.assertEqual(len(episodes), 3)
        trainable = [row for row in episodes if row["ppo_trainable_episode"]]
        self.assertEqual(len(trainable), 1)
        self.assertAlmostEqual(trainable[0]["discounted_episode_return"], 1.0 + 0.99 * 0.0 + 0.99**2 * 2.0)
        self.assertTrue(trainable[0]["uses_multistep_discounted_return"])
        self.assertTrue(trainable[0]["not_single_step_best_action"])
        self.assertTrue(trainable[0]["safe_better_episode"])
        self.assertTrue(trainable[0]["teacher_equivalent_episode"])

        reward_audit = json.loads((self.output_root / "return-aligned-reward-audit.json").read_text(encoding="utf-8"))
        self.assertTrue(reward_audit["uses_multistep_discounted_return"])
        self.assertTrue(reward_audit["not_single_step_best_action"])
        self.assertIn("teacher_following_return", reward_audit["component_names"])
        self.assertIn("safe_better_return", reward_audit["component_names"])
        self.assertEqual(reward_audit["episode_count"], 3)

    def test_validation_test_source_fallback_and_gate_reason_stay_diagnostic(self) -> None:
        from scripts.run_return_aligned_guarded_multi_step_ppo_collector_expansion import (
            run_return_aligned_guarded_multi_step_ppo_collector_expansion,
        )

        summary = run_return_aligned_guarded_multi_step_ppo_collector_expansion(
            guarded_root=self.guarded_root,
            evidence_freeze_summary_path=self.temp_dir / "freeze.json",
            output_root=self.output_root,
            config=self._config(min_episode_count=1, min_transition_count=1),
            repo_root=self.repo_root,
        )

        transitions = self._read_jsonl(self.output_root / "return-aligned-ppo-transitions.jsonl")
        by_episode = {row["episode_id"]: [] for row in transitions}
        for row in transitions:
            by_episode[row["episode_id"]].append(row)

        self.assertTrue(any(row["split"] == "validation" for row in by_episode["validation-episode"]))
        self.assertTrue(all(not row["ppo_trainable"] for row in by_episode["validation-episode"]))
        self.assertTrue(any(row["controlled_choice_source"] == "source_fallback" for row in by_episode["diagnostic-episode"]))
        self.assertTrue(all(not row["ppo_trainable"] for row in by_episode["diagnostic-episode"]))
        self.assertEqual(summary["validation_trainable_count"], 0)
        self.assertEqual(summary["test_trainable_count"], 0)
        self.assertEqual(summary["source_fallback_trainable_count"], 0)

    def test_non_finite_reward_or_incomplete_horizon_fails(self) -> None:
        from scripts.run_return_aligned_guarded_multi_step_ppo_collector_expansion import (
            run_return_aligned_guarded_multi_step_ppo_collector_expansion,
        )

        records = [self._transition("bad-episode", 0, reward="NaN")]
        self._write_transition_records(records)

        summary = run_return_aligned_guarded_multi_step_ppo_collector_expansion(
            guarded_root=self.guarded_root,
            evidence_freeze_summary_path=self.temp_dir / "freeze.json",
            output_root=self.output_root,
            config=self._config(min_episode_count=1, min_transition_count=1),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("non_finite_return_detected", summary["reason_codes"])
        self.assertIn("horizon_complete_episode_count_below_threshold", summary["reason_codes"])

    def test_readiness_accepts_return_aligned_summary(self) -> None:
        from scripts.run_policy_training_readiness_review import (
            _return_aligned_guarded_multistep_collector_readiness,
        )

        readiness = _return_aligned_guarded_multistep_collector_readiness(
            {
                "schema_version": "return-aligned-guarded-multistep-collector-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "trainable_episode_count": 24,
                "trainable_transition_count": 24,
                "validation_trainable_count": 0,
                "test_trainable_count": 0,
                "source_fallback_trainable_count": 0,
                "non_finite_reward_count": 0,
                "non_finite_return_count": 0,
                "non_finite_advantage_count": 0,
                "controlled_regression_count": 0,
                "controlled_safety_regression_count": 0,
                "controlled_contract_regression_count": 0,
                "controlled_path_risk_regression_count": 0,
                "controlled_source_selection_regression_count": 0,
                "uses_multistep_discounted_return": True,
                "not_single_step_best_action": True,
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

    def test_config_declares_outputs_and_non_goals(self) -> None:
        config = json.loads(
            (self.repo_root / "configs" / "return_aligned_guarded_multi_step_ppo_collector_expansion_v1.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            config["schema_version"],
            "return-aligned-guarded-multistep-collector-config/v1",
        )
        self.assertEqual(config["horizon"], 3)
        self.assertEqual(config["output_files"]["summary"], "return-aligned-collector-summary.json")
        self.assertIn("no_new_ppo_update", config["non_goals"])
        self.assertIn("no_checkpoint_publication", config["non_goals"])

    def _write_guarded_inputs(self) -> None:
        self._write_json(
            self.guarded_root / "guarded-ppo-rollout-pilot-summary.json",
            {
                "schema_version": "guarded-ppo-rollout-pilot-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "ppo_trainable_transition_count": 2,
                "optimizer_train_transition_count": 2,
                "post_update_controlled_sequential_regression_count": 0,
                "post_update_quasi_real_collector_trainable_transition_count": 2,
                "git_provenance": {"current_matches_sources": True},
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "formal_training_ready_claimed": False,
            },
        )
        self._write_json(
            self.temp_dir / "freeze.json",
            {
                "schema_version": "guarded-ppo-evidence-freeze-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "training_readiness_status": "guarded_ppo_rollout_pilot_evaluated",
                "training_blockers": [],
                "git_provenance": {"current_matches_sources": True},
            },
        )
        records = [
            self._transition("train-episode", 0, source="policy", reward=1.0, ppo_trainable=True),
            self._transition("train-episode", 1, source="source", reward=0.0),
            self._transition("train-episode", 2, source="policy", reward=2.0, ppo_trainable=True),
            self._transition("validation-episode", 0, split="validation", source="policy", reward=1.0, ppo_trainable=False),
            self._transition("validation-episode", 1, split="validation", source="policy", reward=1.0, ppo_trainable=False),
            self._transition("validation-episode", 2, split="validation", source="policy", reward=1.0, ppo_trainable=False),
            self._transition("diagnostic-episode", 0, source="source_fallback", reward=0.0, reasons=["path_cost_regression"]),
            self._transition("diagnostic-episode", 1, source="policy", reward=1.0, reasons=["risk_regression"]),
            self._transition("diagnostic-episode", 2, source="none", reward=0.0),
        ]
        self._write_transition_records(records)
        self._write_json(
            self.collector_root / "ppo-rollout-collector-summary.json",
            {
                "schema_version": "ppo-rollout-collector-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "episode_count": 3,
                "step_count": 9,
                "ppo_trainable_transition_count": 2,
                "diagnostic_transition_count": 7,
                "source_fallback_trainable_count": 0,
                "missing_log_prob_count": 0,
                "missing_value_count": 0,
                "non_finite_reward_count": 0,
                "fallback_or_open_grid_count": 0,
                "safety_regression_count": 0,
                "contract_violation_count": 0,
                "path_cost_regression_count": 0,
                "risk_regression_count": 0,
                "source_selection_regression_count": 0,
                "git_provenance": {"current_matches_sources": True},
            },
        )

    def _transition(
        self,
        episode_id: str,
        step_index: int,
        *,
        split: str = "train",
        source: str = "policy",
        reward=1.0,
        ppo_trainable: bool = False,
        reasons: list[str] | None = None,
    ) -> dict:
        reasons = list(reasons or [])
        return {
            "schema_version": "ppo-rollout-transition-record/v1",
            "episode_id": episode_id,
            "step_index": step_index,
            "scenario_id": f"{episode_id}-{step_index}",
            "scenario_family": "synthetic",
            "context_id": f"{episode_id}-{step_index}",
            "split": split,
            "controlled_choice_source": source,
            "controlled_action_index": 0,
            "ppo_trainable": ppo_trainable,
            "diagnostic_only": not ppo_trainable,
            "rejection_reason_codes": reasons,
            "reward": reward,
            "reward_components": {
                "better_choice_bonus": 1.0 if ppo_trainable else 0.0,
                "path_improvement": 0.0,
                "risk_improvement": 0.0,
                "utility_improvement": 0.0,
                "gate_penalty": -1.0 if reasons else 0.0,
            },
            "reward_audit": {
                "episode_id": episode_id,
                "step_index": step_index,
                "reward": reward,
                "components": {},
                "reason_codes": [],
            },
            "counter_deltas": {"ppo_trainable_transition_count": 1} if ppo_trainable else {"diagnostic_transition_count": 1},
        }

    def _config(self, *, min_episode_count: int, min_transition_count: int) -> dict:
        return {
            "schema_version": "return-aligned-guarded-multistep-collector-config/v1",
            "horizon": 3,
            "discount_factor": 0.99,
            "input_files": {
                "guarded_summary": "guarded-ppo-rollout-pilot-summary.json",
                "collector_summary": "pilot/collector/ppo-rollout-collector-summary.json",
                "collector_transitions": "pilot/collector/ppo-rollout-transitions.jsonl",
            },
            "validation": {
                "min_trainable_episode_count": min_episode_count,
                "min_trainable_transition_count": min_transition_count,
            },
            "output_files": {
                "episodes": "return-aligned-ppo-episodes.jsonl",
                "transitions": "return-aligned-ppo-transitions.jsonl",
                "reward_audit": "return-aligned-reward-audit.json",
                "rejection_report": "return-aligned-rejection-report.json",
                "summary": "return-aligned-collector-summary.json",
            },
            "non_goals": ["no_new_ppo_update", "no_checkpoint_publication"],
        }

    def _write_transition_records(self, records: list[dict]) -> None:
        self.collector_root.mkdir(parents=True, exist_ok=True)
        (self.collector_root / "ppo-rollout-transitions.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
