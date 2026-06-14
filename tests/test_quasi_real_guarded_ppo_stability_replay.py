import json
import tempfile
import unittest
from pathlib import Path


class QuasiRealGuardedPpoStabilityReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="quasi-real-guarded-stability-"))
        self.freeze_root = self.temp_dir / "freeze"
        self.output_root = self.temp_dir / "stability"
        self.batch_root = self.temp_dir / "batch"
        self.freeze_root.mkdir(parents=True)
        self.batch_root.mkdir(parents=True)
        self._write_freeze_artifacts()

    def test_stability_replay_passes_and_writes_comparison_contract_and_report(self) -> None:
        from scripts.run_quasi_real_guarded_ppo_stability_replay import (
            run_quasi_real_guarded_ppo_stability_replay,
        )

        result = run_quasi_real_guarded_ppo_stability_replay(
            freeze_root=self.freeze_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            pilot_replay_runner=self._passing_pilot_replay,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["schema_version"], "quasi-real-guarded-ppo-stability-replay-summary/v1")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(result["replay_count"], 3)
        self.assertEqual(result["passed_replay_count"], 3)
        self.assertEqual(result["episode_count"], 36)
        self.assertEqual(result["step_count"], 108)
        self.assertEqual(result["ppo_trainable_transition_count"], 36)
        self.assertEqual(result["diagnostic_transition_count"], 72)
        self.assertEqual(result["controlled_regression_count"], 0)
        self.assertEqual(result["teacher_agreement_rate"], 1.0)
        self.assertEqual(result["readiness_status"], "quasi_real_guarded_ppo_stability_replay_evaluated")
        self.assertFalse(result["runs_ppo_update"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])

        summary = json.loads(
            (self.output_root / "quasi-real-guarded-ppo-stability-replay-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary, result)
        comparison_lines = [
            json.loads(line)
            for line in (self.output_root / "stability-replay-comparison.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        self.assertEqual(len(comparison_lines), 3)
        self.assertTrue(all(row["status"] == "matched" for row in comparison_lines))
        contract = json.loads(
            (self.output_root / "acceptance-contract-refinement.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("controlled_regression_count_zero", contract["hard_gates"])
        self.assertIn("validation_test_diagnostic_only", contract["diagnostic_only"])
        report = (self.output_root / "stability-replay-report.md").read_text(encoding="utf-8")
        self.assertIn("Quasi-Real Guarded PPO Stability Replay", report)
        self.assertIn("quasi_real_guarded_ppo_stability_replay_evaluated", report)

    def test_stability_replay_fails_when_replay_behavior_drifts(self) -> None:
        from scripts.run_quasi_real_guarded_ppo_stability_replay import (
            run_quasi_real_guarded_ppo_stability_replay,
        )

        def drifting_runner(*, output_root: Path, replay_index: int, **_kwargs) -> dict:
            summary = self._pilot_summary(output_root)
            if replay_index == 1:
                summary["controlled_regression_count"] = 1
            return summary

        result = run_quasi_real_guarded_ppo_stability_replay(
            freeze_root=self.freeze_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            pilot_replay_runner=drifting_runner,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("replay_controlled_regression_detected", result["reason_codes"])
        self.assertIn("baseline_replay_behavior_drift_detected", result["reason_codes"])

    def test_config_declares_stability_replay_contract_and_non_goals(self) -> None:
        config_path = self.repo_root / "configs" / "quasi_real_guarded_ppo_stability_replay_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["schema_version"], "quasi-real-guarded-ppo-stability-replay-config/v1")
        self.assertEqual(config["replay"]["replay_count"], 3)
        self.assertEqual(
            config["output_files"]["summary"],
            "quasi-real-guarded-ppo-stability-replay-summary.json",
        )
        self.assertIn("does_not_start_new_ppo_update", config["non_goals"])

    def test_readiness_accepts_passed_stability_replay_summary(self) -> None:
        from scripts.run_policy_training_readiness_review import (
            _quasi_real_guarded_ppo_stability_replay_readiness,
        )

        readiness = _quasi_real_guarded_ppo_stability_replay_readiness(
            self._stability_summary()
        )

        self.assertTrue(readiness["present"])
        self.assertTrue(readiness["completed"])
        self.assertEqual(readiness["training_blockers"], [])
        self.assertEqual(readiness["replay_count"], 3)
        self.assertEqual(readiness["passed_replay_count"], 3)

    def _config(self) -> dict:
        return {
            "schema_version": "quasi-real-guarded-ppo-stability-replay-config/v1",
            "readiness": {
                "config": "configs/policy_training_readiness_review_v1.json",
                "expected_status": "quasi_real_guarded_ppo_stability_replay_evaluated",
            },
            "replay": {"replay_count": 3},
            "validation": {
                "expected_episode_count": 36,
                "expected_step_count": 108,
                "min_ppo_trainable_transition_count": 24,
                "expected_ppo_trainable_transition_count": 36,
                "expected_diagnostic_transition_count": 72,
                "min_teacher_agreement_rate": 0.9,
            },
            "output_files": {
                "summary": "quasi-real-guarded-ppo-stability-replay-summary.json",
                "comparison": "stability-replay-comparison.jsonl",
                "acceptance_contract": "acceptance-contract-refinement.json",
                "readiness_validate_only": "quasi-real-guarded-ppo-stability-readiness-validate-only.json",
                "report": "stability-replay-report.md",
            },
        }

    def _write_freeze_artifacts(self) -> None:
        pilot_root = self.temp_dir / "pilot"
        pilot_root.mkdir()
        update_root = self.temp_dir / "update"
        update_root.mkdir()
        freeze_summary = {
            "schema_version": "quasi-real-guarded-ppo-evidence-freeze-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "pilot_root": str(pilot_root),
            "batch_root": str(self.batch_root),
            "update_root": str(update_root),
            "pilot_status": "passed",
            "readiness_status": "quasi_real_guarded_ppo_rollout_pilot_evaluated",
            "pilot_episode_count": 36,
            "pilot_step_count": 108,
            "pilot_ppo_trainable_transition_count": 36,
            "pilot_diagnostic_transition_count": 72,
            "pilot_controlled_regression_count": 0,
            "pilot_teacher_agreement_rate": 1.0,
            "collector_replay_status": "passed",
            "collector_replay_trainable_transition_count": 36,
            "long_horizon_verdict": "long_horizon_teacher_skill_contract_aligned",
            "required_artifact_missing_count": 0,
            "runs_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
        }
        (self.freeze_root / "quasi-real-guarded-ppo-evidence-freeze-summary.json").write_text(
            json.dumps(freeze_summary, indent=2),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": "quasi-real-guarded-ppo-evidence-manifest/v1",
            "required_artifact_count": 9,
            "required_artifact_missing_count": 0,
            "artifacts": [
                {
                    "label": "pilot_summary",
                    "path": str(pilot_root / "quasi-real-guarded-ppo-rollout-pilot-summary.json"),
                    "required": True,
                    "exists": True,
                    "sha256": "abc",
                    "schema_version": "quasi-real-guarded-ppo-rollout-pilot-summary/v1",
                    "status": "passed",
                }
            ],
        }
        (self.freeze_root / "quasi-real-guarded-ppo-evidence-manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    def _passing_pilot_replay(self, *, output_root: Path, replay_index: int, **_kwargs) -> dict:
        return self._pilot_summary(output_root)

    def _pilot_summary(self, output_root: Path) -> dict:
        output_root.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema_version": "quasi-real-guarded-ppo-rollout-pilot-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "episode_count": 36,
            "step_count": 108,
            "trainable_transition_count": 36,
            "ppo_trainable_transition_count": 36,
            "diagnostic_transition_count": 72,
            "validation_trainable_count": 0,
            "test_trainable_count": 0,
            "source_fallback_trainable_count": 0,
            "teacher_fallback_trainable_count": 0,
            "non_empty_gate_reason_trainable_count": 0,
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
            "quasi_real_collector_replay_trainable_transition_count": 36,
            "post_pilot_long_horizon_status": "passed",
            "post_pilot_long_horizon_verdict": "long_horizon_teacher_skill_contract_aligned",
            "uses_multistep_discounted_return": True,
            "not_single_step_best_action": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "summary": str(output_root / "quasi-real-guarded-ppo-rollout-pilot-summary.json"),
        }
        (output_root / "quasi-real-guarded-ppo-rollout-pilot-summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary

    def _stability_summary(self) -> dict:
        return {
            "schema_version": "quasi-real-guarded-ppo-stability-replay-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "replay_count": 3,
            "passed_replay_count": 3,
            "episode_count": 36,
            "step_count": 108,
            "ppo_trainable_transition_count": 36,
            "diagnostic_transition_count": 72,
            "validation_trainable_count": 0,
            "test_trainable_count": 0,
            "source_fallback_trainable_count": 0,
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
            "git_provenance": {"current": {}, "current_matches_sources": True},
        }

    def _passing_readiness(self, *_args, **_kwargs) -> dict:
        return {
            "schema_version": "policy-training-readiness-review-summary/v1",
            "status": "config validated",
            "reason_codes": [],
            "training_readiness_status": "quasi_real_guarded_ppo_stability_replay_evaluated",
            "training_blockers": [],
            "recommended_next_action": "quasi_real_guarded_ppo_stability_replay_evaluated",
        }


if __name__ == "__main__":
    unittest.main()
