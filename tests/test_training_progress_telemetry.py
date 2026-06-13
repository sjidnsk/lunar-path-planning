import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TrainingProgressTelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="training-progress-"))
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_reporter_writes_events_summary_and_plain_stderr(self) -> None:
        from scripts.training_progress import ProgressReporter

        stream = io.StringIO()
        reporter = ProgressReporter(
            output_root=self.temp_dir,
            mode="plain",
            run_id="run-1",
            stream=stream,
        )

        reporter.emit(
            stage="ppo_update",
            status="start",
            current=1,
            total=3,
            round_index=0,
            step_index=None,
            message="ppo epoch 1/1",
            metrics={"optimizer_train_transition_count": 30},
        )
        reporter.emit(
            stage="ppo_update",
            status="passed",
            current=3,
            total=3,
            summary_path="update/limited-ppo-update-smoke-summary.json",
            metrics={"approx_kl": 0.001, "max_grad_norm_after_clip": 1.0},
        )
        summary = reporter.finalize(status="passed", readiness_status="guarded_ppo_rollout_pilot_evaluated")

        events_path = self.temp_dir / "training-progress-events.jsonl"
        self.assertTrue(events_path.is_file())
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["schema_version"], "training-progress-event/v1")
        self.assertEqual(events[0]["run_id"], "run-1")
        self.assertEqual(events[0]["stage"], "ppo_update")
        self.assertEqual(events[0]["status"], "start")
        self.assertEqual(events[0]["current"], 1)
        self.assertEqual(events[0]["total"], 3)
        self.assertEqual(events[0]["round_index"], 0)
        self.assertIn("elapsed_seconds", events[0])
        self.assertEqual(events[0]["metrics"]["optimizer_train_transition_count"], 30)
        self.assertIn("ppo_update", stream.getvalue())

        summary_path = self.temp_dir / "training-progress-summary.json"
        self.assertTrue(summary_path.is_file())
        saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary, saved_summary)
        self.assertEqual(saved_summary["schema_version"], "training-progress-summary/v1")
        self.assertEqual(saved_summary["status"], "passed")
        self.assertEqual(saved_summary["event_count"], 2)
        self.assertEqual(saved_summary["failed_stage_count"], 0)
        self.assertEqual(saved_summary["last_stage"], "ppo_update")
        self.assertEqual(saved_summary["last_status"], "passed")
        self.assertEqual(saved_summary["readiness_status"], "guarded_ppo_rollout_pilot_evaluated")

    def test_reporter_off_preserves_old_behavior_by_writing_nothing(self) -> None:
        from scripts.training_progress import ProgressReporter

        stream = io.StringIO()
        reporter = ProgressReporter(output_root=self.temp_dir, mode="off", stream=stream)
        reporter.emit(stage="collector", status="start", current=1, total=2)
        summary = reporter.finalize(status="passed")

        self.assertEqual(summary, {})
        self.assertEqual(stream.getvalue(), "")
        self.assertFalse((self.temp_dir / "training-progress-events.jsonl").exists())
        self.assertFalse((self.temp_dir / "training-progress-summary.json").exists())

    def test_failure_summary_points_to_last_stage_and_debug_artifact(self) -> None:
        from scripts.training_progress import ProgressReporter

        reporter = ProgressReporter(output_root=self.temp_dir, mode="jsonl", run_id="run-failed")

        reporter.emit(stage="generated_sequential", status="start", current=2, total=9)
        reporter.emit(
            stage="generated_sequential",
            status="failed",
            current=2,
            total=9,
            reason_codes=["canary_rejected_policy_choice_count_above_threshold"],
            summary_path="final/sequential/policy-gated-sequential-canary-rollout-summary.json",
        )
        summary = reporter.finalize(status="failed")

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["last_stage"], "generated_sequential")
        self.assertEqual(summary["last_status"], "failed")
        self.assertEqual(
            summary["last_reason_codes"],
            ["canary_rejected_policy_choice_count_above_threshold"],
        )
        self.assertEqual(summary["failed_stage_count"], 1)
        self.assertEqual(
            summary["recommended_debug_artifact"],
            "final/sequential/policy-gated-sequential-canary-rollout-summary.json",
        )

    def test_update_summary_metrics_are_projected_for_progress_events(self) -> None:
        from scripts.training_progress import ppo_update_progress_metrics

        metrics = ppo_update_progress_metrics(
            {
                "optimizer_train_transition_count": 30,
                "training_result": {"epochs": 1, "final_total_loss": 24.9},
                "approx_kl": 0.001,
                "max_grad_norm_after_clip": 1.0,
                "parameter_l2_delta": 0.0004,
                "loss_non_finite_count": 0,
                "non_finite_gradient_count": 0,
                "non_finite_reward_count": 0,
                "non_finite_return_count": 0,
                "non_finite_advantage_count": 0,
            }
        )

        self.assertEqual(metrics["optimizer_train_transition_count"], 30)
        self.assertEqual(metrics["epochs"], 1)
        self.assertEqual(metrics["loss"], 24.9)
        self.assertEqual(metrics["approx_kl"], 0.001)
        self.assertEqual(metrics["max_grad_norm_after_clip"], 1.0)
        self.assertEqual(metrics["parameter_l2_delta"], 0.0004)
        self.assertEqual(metrics["non_finite_count"], 0)

    def test_training_progress_config_declares_output_contract(self) -> None:
        config = json.loads(
            (self.repo_root / "configs" / "training_progress_telemetry_v1.json").read_text(encoding="utf-8")
        )

        self.assertEqual(config["schema_version"], "training-progress-telemetry-config/v1")
        self.assertEqual(config["progress"]["plain_progress_stream"], "stderr")
        self.assertEqual(config["progress"]["structured_events_filename"], "training-progress-events.jsonl")
        self.assertEqual(config["progress"]["summary_filename"], "training-progress-summary.json")
        self.assertIn("does_not_change_readiness_semantics", config["non_goals"])

    def test_sequential_progress_treats_raw_diagnostics_as_passed_stage(self) -> None:
        from scripts.run_policy_gated_sequential_canary_rollout import _progress_status_for_sequential_summary

        self.assertEqual(
            _progress_status_for_sequential_summary(
                {
                    "status": "failed",
                    "reason_codes": [
                        "multi_step_accepted_episode_count_below_threshold",
                        "family_with_multi_step_accepted_episode_count_below_threshold",
                        "canary_rejected_policy_choice_count_above_threshold",
                    ],
                }
            ),
            "passed",
        )
        self.assertEqual(
            _progress_status_for_sequential_summary(
                {"status": "failed", "reason_codes": ["sequential_step_path_feedback_failed"]}
            ),
            "failed",
        )

    def test_python_entrypoints_accept_progress_off_in_validate_only_mode(self) -> None:
        commands = [
            [
                sys.executable,
                "scripts/run_guarded_ppo_rollout_pilot.py",
                "--source-root",
                str(self.temp_dir / "source"),
                "--base-candidate-root",
                str(self.temp_dir / "base"),
                "--raw-baseline-candidate-root",
                str(self.temp_dir / "raw-base"),
                "--dev-root",
                str(self.temp_dir / "dev"),
                "--val-root",
                str(self.temp_dir / "val"),
                "--test-root",
                str(self.temp_dir / "test"),
                "--output-root",
                str(self.temp_dir / "guarded"),
                "--config",
                "configs/guarded_ppo_rollout_pilot_v1.json",
                "--progress",
                "off",
                "--validate-only",
            ],
            [
                sys.executable,
                "scripts/run_quasi_real_iterative_ppo_mini_loop_stability.py",
                "--source-root",
                str(self.temp_dir / "source"),
                "--initial-candidate-root",
                str(self.temp_dir / "initial"),
                "--quasi-real-root",
                str(self.temp_dir / "quasi"),
                "--output-root",
                str(self.temp_dir / "iterative"),
                "--config",
                "configs/quasi_real_iterative_ppo_mini_loop_stability_v1.json",
                "--progress",
                "off",
                "--validate-only",
            ],
            [
                sys.executable,
                "scripts/run_limited_ppo_update_smoke.py",
                "--source-root",
                str(self.temp_dir / "source"),
                "--base-candidate-root",
                str(self.temp_dir / "base"),
                "--collector-root",
                str(self.temp_dir / "collector"),
                "--output-root",
                str(self.temp_dir / "update"),
                "--config",
                "configs/guarded_ppo_rollout_update_v1.json",
                "--progress",
                "off",
                "--validate-only",
            ],
            [
                sys.executable,
                "scripts/run_policy_gated_sequential_canary_rollout.py",
                "--source-root",
                str(self.temp_dir / "source"),
                "--candidate-root",
                str(self.temp_dir / "candidate"),
                "--batch-root",
                str(self.temp_dir / "sequential"),
                "--config",
                "configs/policy_gated_sequential_multi_step_opportunity_rollout_v1.json",
                "--progress",
                "off",
                "--validate-only",
            ],
        ]

        for command in commands:
            with self.subTest(command=command[1]):
                result = subprocess.run(
                    command,
                    cwd=self.repo_root,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('"status": "config validated"', result.stdout)
                self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
