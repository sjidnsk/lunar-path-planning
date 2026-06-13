import json
import tempfile
import unittest
from pathlib import Path


class GuardedPpoEvidenceFreezeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="guarded-ppo-evidence-freeze-"))
        self.guarded_root = self.temp_dir / "guarded"
        self.batch_root = self.temp_dir / "batch"
        self.output_root = self.temp_dir / "freeze"
        self.guarded_root.mkdir()
        self.batch_root.mkdir()
        self._write_guarded_summary()
        self._write_progress_summary()
        self._write_progress_events()
        self._write_stale_readiness_summary()

    def test_freeze_passes_and_writes_manifest_readiness_and_reports(self) -> None:
        from scripts.run_guarded_ppo_evidence_freeze import run_guarded_ppo_evidence_freeze

        result = run_guarded_ppo_evidence_freeze(
            guarded_root=self.guarded_root,
            batch_root=self.batch_root,
            output_root=self.output_root,
            config={
                "schema_version": "guarded-ppo-evidence-freeze-config/v1",
                "readiness": {"command": ["python", "-m", "fake"]},
            },
            repo_root=self.repo_root,
            readiness_runner=self._guarded_readiness,
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(result["training_readiness_status"], "guarded_ppo_rollout_pilot_evaluated")
        self.assertEqual(result["guarded_pilot_status"], "passed")
        self.assertEqual(result["progress_status"], "passed")
        self.assertEqual(result["progress_failed_stage_count"], 0)
        self.assertEqual(result["progress_event_count"], 2)
        self.assertEqual(result["required_artifact_missing_count"], 0)
        self.assertEqual(result["manifest_required_artifact_count"], 4)
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])

        summary = json.loads(
            (self.output_root / "guarded-ppo-evidence-freeze-summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary, result)
        manifest = json.loads((self.output_root / "evidence-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "guarded-ppo-evidence-manifest/v1")
        self.assertEqual(len(manifest["artifacts"]), 4)
        self.assertTrue(all(item["exists"] for item in manifest["artifacts"]))
        self.assertTrue(all(item["sha256"] for item in manifest["artifacts"]))

        readiness = json.loads((self.output_root / "readiness-final.json").read_text(encoding="utf-8"))
        self.assertEqual(readiness["training_blockers"], [])
        consistency = json.loads(
            (self.output_root / "progress-consistency-report.json").read_text(encoding="utf-8")
        )
        self.assertTrue(consistency["stale_readiness_detected"])
        self.assertEqual(consistency["final_readiness_source"], "explicit_guarded_summary_validate_only")
        report = (self.output_root / "reproducibility-report.md").read_text(encoding="utf-8")
        self.assertIn("Guarded PPO Evidence Freeze", report)
        self.assertIn("guarded_ppo_rollout_pilot_evaluated", report)

    def test_freeze_fails_when_required_artifact_is_missing(self) -> None:
        from scripts.run_guarded_ppo_evidence_freeze import run_guarded_ppo_evidence_freeze

        (self.guarded_root / "training-progress-events.jsonl").unlink()

        result = run_guarded_ppo_evidence_freeze(
            guarded_root=self.guarded_root,
            batch_root=self.batch_root,
            output_root=self.output_root,
            config={"schema_version": "guarded-ppo-evidence-freeze-config/v1"},
            repo_root=self.repo_root,
            readiness_runner=self._guarded_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("required_artifact_missing", result["reason_codes"])
        self.assertEqual(result["required_artifact_missing_count"], 1)
        self.assertIn("training-progress-events.jsonl", result["recommended_debug_artifact"])

    def test_freeze_fails_when_readiness_is_not_guarded(self) -> None:
        from scripts.run_guarded_ppo_evidence_freeze import run_guarded_ppo_evidence_freeze

        result = run_guarded_ppo_evidence_freeze(
            guarded_root=self.guarded_root,
            batch_root=self.batch_root,
            output_root=self.output_root,
            config={"schema_version": "guarded-ppo-evidence-freeze-config/v1"},
            repo_root=self.repo_root,
            readiness_runner=lambda *_args, **_kwargs: {
                "training_readiness_status": "iterative_ppo_mini_loop_stability_evaluated",
                "reason_codes": [],
                "training_blockers": [],
            },
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("readiness_not_guarded_ppo_rollout_pilot_evaluated", result["reason_codes"])
        self.assertEqual(
            result["training_readiness_status"],
            "iterative_ppo_mini_loop_stability_evaluated",
        )

    def test_freeze_config_declares_artifact_contract_and_non_goals(self) -> None:
        config = json.loads(
            (self.repo_root / "configs" / "guarded_ppo_evidence_freeze_v1.json").read_text(encoding="utf-8")
        )

        self.assertEqual(config["schema_version"], "guarded-ppo-evidence-freeze-config/v1")
        self.assertEqual(config["output_files"]["summary"], "guarded-ppo-evidence-freeze-summary.json")
        self.assertIn("guarded-ppo-rollout-pilot-summary.json", config["required_artifacts"])
        self.assertIn("does_not_publish_checkpoint", config["non_goals"])

    def test_nonzero_readiness_with_json_stdout_preserves_real_blockers(self) -> None:
        from scripts.run_guarded_ppo_evidence_freeze import _readiness_result_from_process

        result = _readiness_result_from_process(
            returncode=1,
            stdout=json.dumps(
                {
                    "training_readiness_status": "blocked_by_validation",
                    "reason_codes": ["current_git_provenance_mismatch"],
                    "training_blockers": ["current_git_provenance_mismatch"],
                }
            ),
            stderr="",
            command=["bash", "readiness"],
        )

        self.assertEqual(result["training_readiness_status"], "blocked_by_validation")
        self.assertEqual(result["reason_codes"], ["current_git_provenance_mismatch"])
        self.assertEqual(result["training_blockers"], ["current_git_provenance_mismatch"])
        self.assertEqual(result["command"], ["bash", "readiness"])

    def _write_guarded_summary(self) -> None:
        data = {
            "schema_version": "guarded-ppo-rollout-pilot-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "guarded_rollout_pilot_passed": True,
            "ppo_trainable_transition_count": 30,
            "optimizer_train_transition_count": 30,
            "post_update_controlled_sequential_regression_count": 0,
            "post_update_quasi_real_collector_trainable_transition_count": 36,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
        }
        (self.guarded_root / "guarded-ppo-rollout-pilot-summary.json").write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    def _write_progress_summary(self) -> None:
        data = {
            "schema_version": "training-progress-summary/v1",
            "status": "passed",
            "failed_stage_count": 0,
            "readiness_status": "guarded_ppo_rollout_pilot_evaluated",
            "event_count": 2,
            "last_stage": "guarded_ppo_rollout_pilot_closure",
            "last_status": "passed",
            "last_reason_codes": [],
        }
        (self.guarded_root / "training-progress-summary.json").write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    def _write_progress_events(self) -> None:
        events = [
            {"schema_version": "training-progress-event/v1", "stage": "guarded_pilot", "status": "passed"},
            {"schema_version": "training-progress-event/v1", "stage": "readiness", "status": "passed"},
        ]
        (self.guarded_root / "training-progress-events.jsonl").write_text(
            "\n".join(json.dumps(event) for event in events) + "\n",
            encoding="utf-8",
        )

    def _write_stale_readiness_summary(self) -> None:
        data = {
            "schema_version": "policy-training-readiness-review-summary/v1",
            "training_readiness_status": "iterative_ppo_mini_loop_stability_evaluated",
            "reason_codes": [],
            "training_blockers": [],
        }
        (self.batch_root / "policy-training-readiness-review-summary.json").write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    def _guarded_readiness(self, *_args, **_kwargs) -> dict:
        return {
            "schema_version": "policy-training-readiness-review-summary/v1",
            "training_readiness_status": "guarded_ppo_rollout_pilot_evaluated",
            "reason_codes": [],
            "training_blockers": [],
            "recommended_next_action": "guarded_ppo_rollout_pilot_evaluated",
        }
