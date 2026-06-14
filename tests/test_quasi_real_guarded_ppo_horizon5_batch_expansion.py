import json
import tempfile
import unittest
from pathlib import Path


class QuasiRealGuardedPpoHorizon5BatchExpansionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="quasi-real-guarded-horizon5-"))
        self.stability_root = self.temp_dir / "stability"
        self.freeze_root = self.temp_dir / "freeze"
        self.pilot_root = self.temp_dir / "pilot"
        self.output_root = self.temp_dir / "horizon5"
        self.batch_root = self.temp_dir / "batch"
        for path in (self.stability_root, self.freeze_root, self.pilot_root, self.batch_root):
            path.mkdir(parents=True)
        self._write_input_artifacts()

    def test_horizon5_expansion_passes_and_writes_outputs(self) -> None:
        from scripts.run_quasi_real_guarded_ppo_horizon5_batch_expansion import (
            run_quasi_real_guarded_ppo_horizon5_batch_expansion,
        )

        result = run_quasi_real_guarded_ppo_horizon5_batch_expansion(
            stability_root=self.stability_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(
            result["schema_version"],
            "quasi-real-guarded-ppo-horizon5-batch-expansion-summary/v1",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(result["horizon"], 5)
        self.assertGreaterEqual(result["episode_count"], 96)
        self.assertGreaterEqual(result["step_count"], 480)
        self.assertGreaterEqual(result["ppo_trainable_transition_count"], 96)
        self.assertEqual(result["replay_count"], 3)
        self.assertEqual(result["passed_replay_count"], 3)
        self.assertEqual(result["baseline_replay_behavior_drift_count"], 0)
        self.assertEqual(result["validation_trainable_count"], 0)
        self.assertEqual(result["test_trainable_count"], 0)
        self.assertEqual(result["source_fallback_trainable_count"], 0)
        self.assertEqual(result["teacher_fallback_trainable_count"], 0)
        self.assertEqual(result["missing_observation_count"], 0)
        self.assertEqual(result["missing_log_prob_count"], 0)
        self.assertEqual(result["missing_value_count"], 0)
        self.assertEqual(result["non_finite_reward_count"], 0)
        self.assertEqual(result["non_finite_return_count"], 0)
        self.assertEqual(result["non_finite_advantage_count"], 0)
        self.assertEqual(result["controlled_regression_count"], 0)
        self.assertGreaterEqual(result["teacher_agreement_rate"], 0.95)
        self.assertEqual(result["quasi_real_collector_replay_status"], "passed")
        self.assertGreaterEqual(
            result["quasi_real_collector_replay_trainable_transition_count"],
            96,
        )
        self.assertEqual(
            result["long_horizon_verdict"],
            "long_horizon_teacher_skill_contract_aligned",
        )
        self.assertEqual(
            result["readiness_status"],
            "quasi_real_guarded_ppo_horizon5_batch_expansion_evaluated",
        )
        self.assertFalse(result["runs_ppo_update"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])

        steps = [
            json.loads(line)
            for line in (
                self.output_root / "quasi-real-guarded-ppo-horizon5-batch-expansion-steps.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        episodes = [
            json.loads(line)
            for line in (
                self.output_root / "quasi-real-guarded-ppo-horizon5-batch-expansion-episodes.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(episodes), result["episode_count"])
        self.assertEqual(len(steps), result["step_count"])
        self.assertTrue(all(episode["horizon"] == 5 for episode in episodes))
        self.assertTrue(all(len(episode["steps"]) == 5 for episode in episodes))
        self.assertTrue(all(step["step_index"] in range(5) for step in steps))
        first_episode_steps = episodes[0]["steps"]
        self.assertGreater(
            first_episode_steps[0]["discounted_return"],
            first_episode_steps[0]["reward"],
        )
        diagnostic_split_steps = [
            step for step in steps if step["split"] in {"validation", "test"}
        ]
        self.assertTrue(diagnostic_split_steps)
        self.assertFalse(any(step["ppo_trainable"] for step in diagnostic_split_steps))

        comparison = [
            json.loads(line)
            for line in (self.output_root / "horizon5-batch-expansion-comparison.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(comparison), 3)
        self.assertTrue(all(row["status"] == "matched" for row in comparison))
        report = (self.output_root / "horizon5-batch-expansion-report.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Quasi-Real Guarded PPO Horizon-5 Batch Expansion", report)
        self.assertIn("not formal PPO", report)

    def test_horizon5_expansion_fails_when_stability_input_not_passed(self) -> None:
        from scripts.run_quasi_real_guarded_ppo_horizon5_batch_expansion import (
            run_quasi_real_guarded_ppo_horizon5_batch_expansion,
        )

        summary_path = self.stability_root / "quasi-real-guarded-ppo-stability-replay-summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["status"] = "failed"
        summary["reason_codes"] = ["synthetic_regression"]
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        result = run_quasi_real_guarded_ppo_horizon5_batch_expansion(
            stability_root=self.stability_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("input_stability_replay_not_passed", result["reason_codes"])

    def test_config_declares_horizon5_contract_and_non_goals(self) -> None:
        config_path = (
            self.repo_root
            / "configs"
            / "quasi_real_guarded_ppo_horizon5_batch_expansion_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["schema_version"],
            "quasi-real-guarded-ppo-horizon5-batch-expansion-config/v1",
        )
        self.assertEqual(config["horizon"], 5)
        self.assertEqual(config["validation"]["min_episode_count"], 96)
        self.assertEqual(config["validation"]["min_step_count"], 480)
        self.assertEqual(config["validation"]["min_ppo_trainable_transition_count"], 96)
        self.assertIn("does_not_start_formal_ppo", config["non_goals"])

    def test_readiness_accepts_passed_horizon5_summary(self) -> None:
        from scripts.run_policy_training_readiness_review import (
            _quasi_real_guarded_ppo_horizon5_batch_expansion_readiness,
        )

        readiness = _quasi_real_guarded_ppo_horizon5_batch_expansion_readiness(
            self._horizon5_summary()
        )

        self.assertTrue(readiness["present"])
        self.assertTrue(readiness["completed"])
        self.assertEqual(readiness["training_blockers"], [])
        self.assertEqual(readiness["horizon"], 5)
        self.assertEqual(readiness["trainable_transition_count"], 160)

    def _config(self) -> dict:
        return {
            "schema_version": "quasi-real-guarded-ppo-horizon5-batch-expansion-config/v1",
            "horizon": 5,
            "discount_factor": 0.99,
            "expansion": {"episode_count": 96, "replay_count": 3},
            "validation": {
                "min_episode_count": 96,
                "min_step_count": 480,
                "min_ppo_trainable_transition_count": 96,
                "min_teacher_agreement_rate": 0.95,
            },
            "readiness": {
                "config": "configs/policy_training_readiness_review_v1.json",
                "expected_status": "quasi_real_guarded_ppo_horizon5_batch_expansion_evaluated",
            },
            "output_files": {
                "summary": "quasi-real-guarded-ppo-horizon5-batch-expansion-summary.json",
                "episodes": "quasi-real-guarded-ppo-horizon5-batch-expansion-episodes.jsonl",
                "steps": "quasi-real-guarded-ppo-horizon5-batch-expansion-steps.jsonl",
                "reward_audit": "quasi-real-guarded-ppo-horizon5-batch-expansion-reward-audit.json",
                "rejection_report": "quasi-real-guarded-ppo-horizon5-batch-expansion-rejection-report.json",
                "comparison": "horizon5-batch-expansion-comparison.jsonl",
                "progress_events": "horizon5-batch-expansion-progress-events.jsonl",
                "readiness_validate_only": "horizon5-batch-expansion-readiness-validate-only.json",
                "report": "horizon5-batch-expansion-report.md",
            },
        }

    def _write_input_artifacts(self) -> None:
        pilot_steps = [self._baseline_step(index) for index in range(15)]
        (self.pilot_root / "quasi-real-guarded-ppo-rollout-steps.jsonl").write_text(
            "".join(json.dumps(step, sort_keys=True) + "\n" for step in pilot_steps),
            encoding="utf-8",
        )
        pilot_summary = {
            "schema_version": "quasi-real-guarded-ppo-rollout-pilot-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "episode_count": 5,
            "step_count": 15,
            "ppo_trainable_transition_count": 5,
            "diagnostic_transition_count": 10,
        }
        (self.pilot_root / "quasi-real-guarded-ppo-rollout-pilot-summary.json").write_text(
            json.dumps(pilot_summary, indent=2),
            encoding="utf-8",
        )
        freeze_summary = {
            "schema_version": "quasi-real-guarded-ppo-evidence-freeze-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "pilot_root": str(self.pilot_root),
            "pilot_status": "passed",
            "required_artifact_missing_count": 0,
        }
        freeze_manifest = {
            "schema_version": "quasi-real-guarded-ppo-evidence-manifest/v1",
            "required_artifact_missing_count": 0,
            "artifacts": [],
        }
        (self.freeze_root / "quasi-real-guarded-ppo-evidence-freeze-summary.json").write_text(
            json.dumps(freeze_summary, indent=2),
            encoding="utf-8",
        )
        (self.freeze_root / "quasi-real-guarded-ppo-evidence-manifest.json").write_text(
            json.dumps(freeze_manifest, indent=2),
            encoding="utf-8",
        )
        stability_summary = {
            "schema_version": "quasi-real-guarded-ppo-stability-replay-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "freeze_root": str(self.freeze_root),
            "freeze_summary": str(
                self.freeze_root / "quasi-real-guarded-ppo-evidence-freeze-summary.json"
            ),
            "freeze_manifest": str(
                self.freeze_root / "quasi-real-guarded-ppo-evidence-manifest.json"
            ),
            "batch_root": str(self.batch_root),
            "candidate_root": str(self.temp_dir / "candidate"),
            "quasi_real_root": str(self.temp_dir / "quasi-real"),
            "replay_count": 3,
            "passed_replay_count": 3,
            "episode_count": 36,
            "step_count": 108,
            "ppo_trainable_transition_count": 36,
            "diagnostic_transition_count": 72,
            "validation_trainable_count": 0,
            "test_trainable_count": 0,
            "source_fallback_trainable_count": 0,
            "teacher_fallback_trainable_count": 0,
            "missing_observation_count": 0,
            "missing_log_prob_count": 0,
            "missing_value_count": 0,
            "non_finite_reward_count": 0,
            "non_finite_return_count": 0,
            "non_finite_advantage_count": 0,
            "controlled_regression_count": 0,
            "controlled_safety_regression_count": 0,
            "controlled_contract_regression_count": 0,
            "controlled_path_risk_regression_count": 0,
            "controlled_source_selection_regression_count": 0,
            "teacher_agreement_rate": 1.0,
            "baseline_replay_behavior_drift_count": 0,
            "quasi_real_collector_replay_status": "passed",
            "long_horizon_verdict": "long_horizon_teacher_skill_contract_aligned",
            "acceptance_contract_refined": True,
            "runs_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
        }
        (
            self.stability_root
            / "quasi-real-guarded-ppo-stability-replay-summary.json"
        ).write_text(json.dumps(stability_summary, indent=2), encoding="utf-8")
        (self.stability_root / "acceptance-contract-refinement.json").write_text(
            json.dumps({"schema_version": "contract/v1", "hard_gates": []}, indent=2),
            encoding="utf-8",
        )

    def _baseline_step(self, index: int) -> dict:
        split = ("train", "validation", "test")[index % 3]
        return {
            "schema_version": "quasi-real-guarded-ppo-rollout-step/v1",
            "episode_id": f"baseline-{index // 3:04d}",
            "step_index": index % 3,
            "decision_index": index,
            "context_id": f"context-{index:04d}",
            "scenario_id": f"scenario-{index:04d}",
            "scenario_family": "unit_family",
            "split": split,
            "raw_policy_action_index": 1,
            "teacher_action_index": 1,
            "controlled_action_index": 1,
            "controlled_choice_source": "policy",
            "controlled_choice_detail": "policy_teacher_aligned",
            "policy_takes_control": True,
            "gate_reason_codes": [],
            "controlled_regression_reason_codes": [],
            "ppo_trainable_candidate": split == "train",
            "ppo_trainable": split == "train",
            "diagnostic_only": split != "train",
            "rejection_reason_codes": [] if split == "train" else ["non_train_split"],
            "observation": {"action_mask": [True, True, True]},
            "missing_observation": False,
            "log_prob": -0.25,
            "value": 0.1,
            "reward": 1.0,
            "discounted_return": 1.0,
            "advantage": 0.9,
            "done": index % 3 == 2,
            "reward_components": {"teacher_following_bonus": 1.0},
            "path_cost_delta": 0.0,
            "risk_delta": 0.0,
        }

    def _passing_readiness(self, *_args, **_kwargs) -> dict:
        return {
            "schema_version": "policy-training-readiness-review-summary/v1",
            "status": "config validated",
            "reason_codes": [],
            "training_readiness_status": "quasi_real_guarded_ppo_horizon5_batch_expansion_evaluated",
            "training_blockers": [],
            "recommended_next_action": "quasi_real_guarded_ppo_horizon5_batch_expansion_evaluated",
        }

    def _horizon5_summary(self) -> dict:
        return {
            "schema_version": "quasi-real-guarded-ppo-horizon5-batch-expansion-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "horizon": 5,
            "episode_count": 96,
            "step_count": 480,
            "ppo_trainable_transition_count": 160,
            "diagnostic_transition_count": 320,
            "replay_count": 3,
            "passed_replay_count": 3,
            "baseline_replay_behavior_drift_count": 0,
            "validation_trainable_count": 0,
            "test_trainable_count": 0,
            "source_fallback_trainable_count": 0,
            "teacher_fallback_trainable_count": 0,
            "missing_observation_count": 0,
            "missing_log_prob_count": 0,
            "missing_value_count": 0,
            "non_finite_reward_count": 0,
            "non_finite_return_count": 0,
            "non_finite_advantage_count": 0,
            "controlled_regression_count": 0,
            "controlled_safety_regression_count": 0,
            "controlled_contract_regression_count": 0,
            "controlled_path_risk_regression_count": 0,
            "controlled_source_selection_regression_count": 0,
            "teacher_agreement_rate": 1.0,
            "quasi_real_collector_replay_status": "passed",
            "quasi_real_collector_replay_trainable_transition_count": 160,
            "long_horizon_verdict": "long_horizon_teacher_skill_contract_aligned",
            "uses_multistep_discounted_return": True,
            "not_single_step_best_action": True,
            "runs_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "git_provenance": {"current": {}, "current_matches_sources": True},
        }


if __name__ == "__main__":
    unittest.main()
