import json
import tempfile
import unittest
from pathlib import Path


class QuasiRealGuardedPpoEvidenceFreezeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="quasi-real-guarded-freeze-"))
        self.pilot_root = self.temp_dir / "pilot"
        self.batch_root = self.temp_dir / "batch"
        self.update_root = self.temp_dir / "update"
        self.output_root = self.temp_dir / "freeze"
        for path in (self.pilot_root, self.batch_root, self.update_root):
            path.mkdir(parents=True)
        self._write_pilot_artifacts()
        self._write_update_summary()
        self._write_long_horizon_summary()
        self._write_collector_replay_summary()
        self._write_stale_readiness_summary()

    def test_freeze_passes_and_writes_manifest_readiness_and_report(self) -> None:
        from scripts.run_quasi_real_guarded_ppo_evidence_freeze import (
            run_quasi_real_guarded_ppo_evidence_freeze,
        )

        result = run_quasi_real_guarded_ppo_evidence_freeze(
            pilot_root=self.pilot_root,
            batch_root=self.batch_root,
            update_root=self.update_root,
            output_root=self.output_root,
            config=self._config(),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(result["pilot_status"], "passed")
        self.assertEqual(result["readiness_status"], "quasi_real_guarded_ppo_rollout_pilot_evaluated")
        self.assertTrue(result["stale_written_readiness_summary_detected"])
        self.assertEqual(result["pilot_episode_count"], 36)
        self.assertEqual(result["pilot_step_count"], 108)
        self.assertEqual(result["pilot_ppo_trainable_transition_count"], 36)
        self.assertEqual(result["pilot_diagnostic_transition_count"], 72)
        self.assertEqual(result["pilot_controlled_regression_count"], 0)
        self.assertEqual(result["pilot_teacher_agreement_rate"], 1.0)
        self.assertFalse(result["runs_ppo_update"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])

        summary = json.loads(
            (self.output_root / "quasi-real-guarded-ppo-evidence-freeze-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary, result)
        readiness = json.loads(
            (self.output_root / "quasi-real-guarded-ppo-readiness-validate-only.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(readiness["training_blockers"], [])
        manifest = json.loads(
            (self.output_root / "quasi-real-guarded-ppo-evidence-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["schema_version"], "quasi-real-guarded-ppo-evidence-manifest/v1")
        self.assertEqual(manifest["required_artifact_missing_count"], 0)
        self.assertGreaterEqual(len(manifest["artifacts"]), 8)
        self.assertTrue(all(item["sha256"] for item in manifest["artifacts"] if item["required"]))
        report = (
            self.output_root / "quasi-real-guarded-ppo-evidence-freeze-report.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Quasi-Real Guarded PPO Evidence Freeze", report)
        self.assertIn("quasi_real_guarded_ppo_rollout_pilot_evaluated", report)
        self.assertIn("stale written readiness summary", report)

    def test_freeze_fails_when_pilot_contract_regresses(self) -> None:
        from scripts.run_quasi_real_guarded_ppo_evidence_freeze import (
            run_quasi_real_guarded_ppo_evidence_freeze,
        )

        summary_path = self.pilot_root / "quasi-real-guarded-ppo-rollout-pilot-summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["controlled_regression_count"] = 1
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        result = run_quasi_real_guarded_ppo_evidence_freeze(
            pilot_root=self.pilot_root,
            batch_root=self.batch_root,
            update_root=self.update_root,
            output_root=self.output_root,
            config=self._config(),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("pilot_controlled_regression_detected", result["reason_codes"])

    def test_freeze_fails_when_explicit_readiness_blocks(self) -> None:
        from scripts.run_quasi_real_guarded_ppo_evidence_freeze import (
            run_quasi_real_guarded_ppo_evidence_freeze,
        )

        result = run_quasi_real_guarded_ppo_evidence_freeze(
            pilot_root=self.pilot_root,
            batch_root=self.batch_root,
            update_root=self.update_root,
            output_root=self.output_root,
            config=self._config(),
            repo_root=self.repo_root,
            readiness_runner=lambda *_args, **_kwargs: {
                "training_readiness_status": "blocked_by_validation",
                "reason_codes": ["current_git_provenance_mismatch"],
                "training_blockers": ["current_git_provenance_mismatch"],
            },
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("readiness_not_quasi_real_guarded_ppo_rollout_pilot_evaluated", result["reason_codes"])
        self.assertIn("readiness_reason_codes_non_empty", result["reason_codes"])

    def test_config_declares_freeze_contract_and_non_goals(self) -> None:
        config_path = self.repo_root / "configs" / "quasi_real_guarded_ppo_evidence_freeze_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["schema_version"], "quasi-real-guarded-ppo-evidence-freeze-config/v1")
        self.assertEqual(
            config["output_files"]["summary"],
            "quasi-real-guarded-ppo-evidence-freeze-summary.json",
        )
        self.assertIn(
            "quasi-real-guarded-ppo-rollout-pilot-summary.json",
            config["required_artifacts"],
        )
        self.assertIn("does_not_start_new_ppo_update", config["non_goals"])

    def _config(self) -> dict:
        return {
            "schema_version": "quasi-real-guarded-ppo-evidence-freeze-config/v1",
            "readiness": {
                "config": "configs/policy_training_readiness_review_v1.json",
                "expected_status": "quasi_real_guarded_ppo_rollout_pilot_evaluated",
            },
            "output_files": {
                "summary": "quasi-real-guarded-ppo-evidence-freeze-summary.json",
                "manifest": "quasi-real-guarded-ppo-evidence-manifest.json",
                "readiness_validate_only": "quasi-real-guarded-ppo-readiness-validate-only.json",
                "report": "quasi-real-guarded-ppo-evidence-freeze-report.md",
            },
        }

    def _write_pilot_artifacts(self) -> None:
        summary = {
            "schema_version": "quasi-real-guarded-ppo-rollout-pilot-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "episode_count": 36,
            "step_count": 108,
            "ppo_trainable_transition_count": 36,
            "trainable_transition_count": 36,
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
            "quasi_real_collector_replay_status": "passed",
            "quasi_real_collector_replay_trainable_transition_count": 36,
            "post_pilot_long_horizon_verdict": "long_horizon_teacher_skill_contract_aligned",
            "runs_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "summary": str(self.pilot_root / "quasi-real-guarded-ppo-rollout-pilot-summary.json"),
            "episodes": str(self.pilot_root / "quasi-real-guarded-ppo-rollout-episodes.jsonl"),
            "steps": str(self.pilot_root / "quasi-real-guarded-ppo-rollout-steps.jsonl"),
            "rejection_report": str(self.pilot_root / "quasi-real-guarded-ppo-rollout-rejection-report.json"),
            "reward_audit": str(self.pilot_root / "quasi-real-guarded-ppo-rollout-reward-audit.json"),
            "long_horizon_summary": str(self.update_root / "post_update_long_horizon" / "long-horizon-teacher-skill-contract-summary.json"),
        }
        (self.pilot_root / "quasi-real-guarded-ppo-rollout-pilot-summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        for name in (
            "quasi-real-guarded-ppo-rollout-episodes.jsonl",
            "quasi-real-guarded-ppo-rollout-steps.jsonl",
        ):
            (self.pilot_root / name).write_text(
                json.dumps({"schema_version": "quasi-real-guarded-ppo-rollout-step/v1"}) + "\n",
                encoding="utf-8",
            )
        for name in (
            "quasi-real-guarded-ppo-rollout-rejection-report.json",
            "quasi-real-guarded-ppo-rollout-reward-audit.json",
        ):
            (self.pilot_root / name).write_text(
                json.dumps({"schema_version": name.replace(".json", "/v1"), "status": "present"}),
                encoding="utf-8",
            )

    def _write_update_summary(self) -> None:
        data = {
            "schema_version": "return-aligned-guarded-ppo-update-smoke-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "optimizer_train_transition_count": 30,
            "parameter_l2_delta": 0.0004,
            "approx_kl": 0.001,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
        }
        (self.update_root / "return-aligned-guarded-ppo-update-smoke-summary.json").write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    def _write_long_horizon_summary(self) -> None:
        path = self.update_root / "post_update_long_horizon"
        path.mkdir(parents=True)
        (path / "long-horizon-teacher-skill-contract-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "generated-sequential-long-horizon-teacher-skill-contract-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "verdict": "long_horizon_teacher_skill_contract_aligned",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_collector_replay_summary(self) -> None:
        path = self.pilot_root / "quasi_real_collector_replay"
        path.mkdir(parents=True)
        (path / "ppo-rollout-collector-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "ppo-rollout-collector-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "ppo_trainable_transition_count": 36,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_stale_readiness_summary(self) -> None:
        (self.batch_root / "policy-training-readiness-review-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "policy-training-readiness-review-summary/v1",
                    "status": "failed",
                    "reason_codes": ["current_git_provenance_mismatch"],
                    "training_readiness_status": "blocked_by_validation",
                    "training_blockers": ["current_git_provenance_mismatch"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _passing_readiness(self, *_args, **_kwargs) -> dict:
        return {
            "schema_version": "policy-training-readiness-review-summary/v1",
            "status": "config validated",
            "reason_codes": [],
            "training_readiness_status": "quasi_real_guarded_ppo_rollout_pilot_evaluated",
            "training_blockers": [],
            "recommended_next_action": "quasi_real_guarded_ppo_rollout_pilot_evaluated",
        }


if __name__ == "__main__":
    unittest.main()
