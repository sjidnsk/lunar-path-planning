import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class Global99SandboxCandidateInstallationDryRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="global-99-sandbox-install-"))
        self.auth_root = self.temp_dir / "authorization"
        self.output_root = self.temp_dir / "out"
        self.config_path = self.temp_dir / "config.json"
        self.auth_root.mkdir(parents=True)
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_authorized_candidate_runs_sandbox_install_dry_run_only(self) -> None:
        from scripts.run_global_99_sandbox_candidate_installation_dry_run import run_global_99_sandbox_candidate_installation_dry_run

        self._write_authorization()
        summary = run_global_99_sandbox_candidate_installation_dry_run(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(summary["sandbox_candidate_installation_dry_run_passed"])
        self.assertEqual(summary["sandbox_installation_verdict"], "eligible_for_sandbox_consumer_replay_canary")
        self.assertEqual(summary["next_required_change"], "sandbox_consumer_replay_canary")
        self.assertEqual(summary["source_authorization_status"], "passed")
        self.assertEqual(summary["source_authorization_next_required_change"], "sandbox_candidate_installation_dry_run")
        self.assertTrue(summary["sandbox_candidate_hash_audit_passed"])
        self.assertTrue(summary["sandbox_candidate_load_audit_passed"])
        self.assertTrue(summary["rollback_audit_passed"])
        self.assertTrue(summary["kill_switch_audit_passed"])
        self.assertTrue(summary["telemetry_audit_passed"])
        self.assertTrue(summary["boundary_audit_passed"])
        self.assertGreater(summary["sandbox_candidate_size_bytes"], 0)
        self.assertEqual(len(summary["sandbox_candidate_sha256"]), 64)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertEqual(summary["canary_traffic_fraction"], 0.0)
        self.assertFalse(summary["default_policy_replacement_approved"])
        self.assertFalse(summary["real_executor_connection_approved"])

        for filename in (
            "global-99-sandbox-candidate-installation-dry-run-summary.json",
            "global-99-sandbox-candidate-installation-manifest.json",
            "global-99-sandbox-candidate-provenance-audit.json",
            "global-99-sandbox-candidate-hash-audit.json",
            "global-99-sandbox-candidate-load-audit.json",
            "global-99-sandbox-candidate-rollback-audit.json",
            "global-99-sandbox-candidate-kill-switch-audit.json",
            "global-99-sandbox-candidate-telemetry-audit.json",
            "global-99-sandbox-candidate-boundary-audit.json",
            "global-99-sandbox-candidate-rejection-report.json",
            "global-99-sandbox-candidate-installation-dry-run-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_authorization_routes_to_authorization_fix(self) -> None:
        from scripts.run_global_99_sandbox_candidate_installation_dry_run import run_global_99_sandbox_candidate_installation_dry_run

        summary = run_global_99_sandbox_candidate_installation_dry_run(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_authorization_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_default_policy_candidate_authorization_preflight")

    def test_failed_authorization_blocks_installation_dry_run(self) -> None:
        from scripts.run_global_99_sandbox_candidate_installation_dry_run import run_global_99_sandbox_candidate_installation_dry_run

        self._write_authorization({"status": "failed", "reason_codes": ["authorization_failed"]})
        summary = run_global_99_sandbox_candidate_installation_dry_run(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("authorization_not_passed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_default_policy_candidate_authorization_preflight")

    def test_missing_hash_blocks_sandbox_installation(self) -> None:
        from scripts.run_global_99_sandbox_candidate_installation_dry_run import run_global_99_sandbox_candidate_installation_dry_run

        self._write_config({"require_candidate_hash": False})
        self._write_authorization()
        summary = run_global_99_sandbox_candidate_installation_dry_run(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sandbox_candidate_hash_missing", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_sandbox_candidate_installation_dry_run")

    def test_boundary_violation_blocks_sandbox_installation(self) -> None:
        from scripts.run_global_99_sandbox_candidate_installation_dry_run import run_global_99_sandbox_candidate_installation_dry_run

        self._write_authorization({"replaces_default_policy": True})
        summary = run_global_99_sandbox_candidate_installation_dry_run(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sandbox_installation_boundary_violation", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "resolve_sandbox_candidate_installation_boundary_rejections")

    def _write_config(self, updates: dict | None = None) -> None:
        payload = {
            "schema_version": "global-99-sandbox-candidate-installation-dry-run-config/v1",
            "source_authorization_root": str(self.auth_root),
            "sandbox_candidate_id": "test-candidate",
            "sandbox_installation_root": str(self.output_root / "sandbox_candidate"),
            "require_authorization_passed": True,
            "require_candidate_hash": True,
            "require_sandbox_load": True,
            "require_rollback": True,
            "require_kill_switch": True,
            "require_telemetry": True,
            "sandbox_installation_only": True,
            "canary_traffic_fraction": 0.0,
        }
        if updates:
            payload.update(updates)
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_authorization(self, updates: dict | None = None) -> None:
        payload = {
            "schema_version": "global-99-default-policy-candidate-authorization-preflight-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "authorization_verdict": "eligible_for_sandbox_candidate_installation_dry_run",
            "next_required_change": "sandbox_candidate_installation_dry_run",
            "default_policy_candidate_authorization_preflight_passed": True,
            "default_policy_candidate_installation_approved": False,
            "candidate_read_only_audit_passed": True,
            "default_policy_read_only_audit_passed": True,
            "executor_isolation_audit_passed": True,
            "path_planner_isolation_audit_passed": True,
            "kill_switch_audit_passed": True,
            "rollback_audit_passed": True,
            "telemetry_audit_passed": True,
            "boundary_audit_passed": True,
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
        (self.auth_root / "global-99-default-policy-candidate-authorization-preflight-summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
