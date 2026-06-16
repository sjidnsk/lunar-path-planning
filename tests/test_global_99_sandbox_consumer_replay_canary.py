import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class Global99SandboxConsumerReplayCanaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="global-99-sandbox-consumer-"))
        self.install_root = self.temp_dir / "install"
        self.output_root = self.temp_dir / "out"
        self.config_path = self.temp_dir / "config.json"
        self.install_root.mkdir(parents=True)
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_sandbox_consumer_replay_canary_passes_without_online_canary(self) -> None:
        from scripts.run_global_99_sandbox_consumer_replay_canary import run_global_99_sandbox_consumer_replay_canary

        self._write_install()
        summary = run_global_99_sandbox_consumer_replay_canary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(summary["sandbox_consumer_replay_canary_passed"])
        self.assertEqual(summary["sandbox_consumer_verdict"], "eligible_for_controlled_default_policy_candidate_installation_preflight")
        self.assertEqual(summary["next_required_change"], "controlled_default_policy_candidate_installation_preflight")
        self.assertEqual(summary["source_sandbox_install_status"], "passed")
        self.assertEqual(summary["source_sandbox_install_next_required_change"], "sandbox_consumer_replay_canary")
        self.assertEqual(summary["consumer_step_count"], 64)
        self.assertEqual(summary["fallback_rate"], 0.0)
        self.assertEqual(summary["controlled_regression_count"], 0)
        self.assertTrue(summary["candidate_load_audit_passed"])
        self.assertTrue(summary["fallback_audit_passed"])
        self.assertTrue(summary["telemetry_audit_passed"])
        self.assertTrue(summary["rollback_audit_passed"])
        self.assertTrue(summary["boundary_audit_passed"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertEqual(summary["canary_traffic_fraction"], 0.0)
        self.assertFalse(summary["default_policy_replacement_approved"])
        self.assertFalse(summary["real_executor_connection_approved"])

        for filename in (
            "global-99-sandbox-consumer-replay-canary-summary.json",
            "global-99-sandbox-consumer-manifest.json",
            "global-99-sandbox-consumer-replay-trace.jsonl",
            "global-99-sandbox-consumer-load-audit.json",
            "global-99-sandbox-consumer-fallback-audit.json",
            "global-99-sandbox-consumer-telemetry-audit.json",
            "global-99-sandbox-consumer-rollback-audit.json",
            "global-99-sandbox-consumer-boundary-audit.json",
            "global-99-sandbox-consumer-rejection-report.json",
            "global-99-sandbox-consumer-replay-canary-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_install_routes_to_install_fix(self) -> None:
        from scripts.run_global_99_sandbox_consumer_replay_canary import run_global_99_sandbox_consumer_replay_canary

        summary = run_global_99_sandbox_consumer_replay_canary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_sandbox_install_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_sandbox_candidate_installation_dry_run")

    def test_failed_install_blocks_consumer(self) -> None:
        from scripts.run_global_99_sandbox_consumer_replay_canary import run_global_99_sandbox_consumer_replay_canary

        self._write_install({"status": "failed", "reason_codes": ["install_failed"]})
        summary = run_global_99_sandbox_consumer_replay_canary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sandbox_install_not_passed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_sandbox_candidate_installation_dry_run")

    def test_high_fallback_blocks_consumer(self) -> None:
        from scripts.run_global_99_sandbox_consumer_replay_canary import run_global_99_sandbox_consumer_replay_canary

        self._write_config({"max_fallback_rate": 0.0})
        self._write_install()
        summary = run_global_99_sandbox_consumer_replay_canary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sandbox_consumer_fallback_rate_too_high", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_sandbox_consumer_replay_canary")

    def test_boundary_violation_blocks_consumer(self) -> None:
        from scripts.run_global_99_sandbox_consumer_replay_canary import run_global_99_sandbox_consumer_replay_canary

        self._write_install({"starts_online_canary": True})
        summary = run_global_99_sandbox_consumer_replay_canary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sandbox_consumer_boundary_violation", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "resolve_sandbox_consumer_boundary_rejections")

    def _write_config(self, updates: dict | None = None) -> None:
        payload = {
            "schema_version": "global-99-sandbox-consumer-replay-canary-config/v1",
            "source_sandbox_install_root": str(self.install_root),
            "consumer_step_count": 64,
            "max_fallback_rate": 0.05,
            "require_sandbox_install_passed": True,
            "require_candidate_load": True,
            "require_fallback_available": True,
            "require_telemetry": True,
            "require_rollback": True,
            "sandbox_consumer_only": True,
            "canary_traffic_fraction": 0.0,
        }
        if updates:
            payload.update(updates)
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_install(self, updates: dict | None = None) -> None:
        payload = {
            "schema_version": "global-99-sandbox-candidate-installation-dry-run-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "sandbox_installation_verdict": "eligible_for_sandbox_consumer_replay_canary",
            "next_required_change": "sandbox_consumer_replay_canary",
            "sandbox_candidate_installation_dry_run_passed": True,
            "sandbox_candidate_hash_audit_passed": True,
            "sandbox_candidate_load_audit_passed": True,
            "rollback_audit_passed": True,
            "kill_switch_audit_passed": True,
            "telemetry_audit_passed": True,
            "boundary_audit_passed": True,
            "sandbox_candidate_sha256": "a" * 64,
            "sandbox_candidate_size_bytes": 123,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
            "default_policy_replacement_approved": False,
            "real_executor_connection_approved": False,
        }
        if updates:
            payload.update(updates)
        (self.install_root / "global-99-sandbox-candidate-installation-dry-run-summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
