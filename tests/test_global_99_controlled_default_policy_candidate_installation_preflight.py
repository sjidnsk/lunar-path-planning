import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class Global99ControlledDefaultPolicyCandidateInstallationPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="global-99-controlled-install-"))
        self.consumer_root = self.temp_dir / "consumer"
        self.output_root = self.temp_dir / "out"
        self.config_path = self.temp_dir / "config.json"
        self.consumer_root.mkdir(parents=True)
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_controlled_installation_review_eligibility_passes_without_replacement(self) -> None:
        from scripts.run_global_99_controlled_default_policy_candidate_installation_preflight import (
            run_global_99_controlled_default_policy_candidate_installation_preflight,
        )

        self._write_consumer()
        summary = run_global_99_controlled_default_policy_candidate_installation_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(summary["controlled_default_policy_candidate_installation_preflight_passed"])
        self.assertEqual(summary["controlled_installation_verdict"], "eligible_for_controlled_default_policy_candidate_installation_review")
        self.assertEqual(summary["next_required_change"], "eligible_for_controlled_default_policy_candidate_installation_review")
        self.assertEqual(summary["source_sandbox_consumer_status"], "passed")
        self.assertEqual(summary["source_sandbox_consumer_next_required_change"], "controlled_default_policy_candidate_installation_preflight")
        self.assertTrue(summary["review_scope_audit_passed"])
        self.assertTrue(summary["default_policy_read_only_audit_passed"])
        self.assertTrue(summary["executor_isolation_audit_passed"])
        self.assertTrue(summary["online_canary_audit_passed"])
        self.assertTrue(summary["kill_switch_audit_passed"])
        self.assertTrue(summary["rollback_audit_passed"])
        self.assertTrue(summary["telemetry_audit_passed"])
        self.assertTrue(summary["boundary_audit_passed"])
        self.assertFalse(summary["controlled_installation_executed"])
        self.assertFalse(summary["default_policy_replacement_approved"])
        self.assertFalse(summary["real_executor_connection_approved"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertEqual(summary["canary_traffic_fraction"], 0.0)
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])
        self.assertFalse(summary["modifies_action_space"])
        self.assertFalse(summary["modifies_default_astar"])

        for filename in (
            "global-99-controlled-default-policy-candidate-installation-preflight-summary.json",
            "global-99-controlled-default-policy-candidate-installation-manifest.json",
            "global-99-controlled-installation-lineage-audit.json",
            "global-99-controlled-installation-review-scope-audit.json",
            "global-99-controlled-installation-boundary-audit.json",
            "global-99-controlled-installation-kill-switch-audit.json",
            "global-99-controlled-installation-rollback-audit.json",
            "global-99-controlled-installation-telemetry-audit.json",
            "global-99-controlled-installation-rejection-report.json",
            "global-99-controlled-default-policy-candidate-installation-preflight-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_consumer_routes_to_consumer_fix(self) -> None:
        from scripts.run_global_99_controlled_default_policy_candidate_installation_preflight import (
            run_global_99_controlled_default_policy_candidate_installation_preflight,
        )

        summary = run_global_99_controlled_default_policy_candidate_installation_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_sandbox_consumer_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_sandbox_consumer_replay_canary")

    def test_failed_consumer_blocks_controlled_preflight(self) -> None:
        from scripts.run_global_99_controlled_default_policy_candidate_installation_preflight import (
            run_global_99_controlled_default_policy_candidate_installation_preflight,
        )

        self._write_consumer({"status": "failed", "reason_codes": ["consumer_failed"]})
        summary = run_global_99_controlled_default_policy_candidate_installation_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sandbox_consumer_not_passed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_sandbox_consumer_replay_canary")

    def test_review_scope_failure_blocks_preflight(self) -> None:
        from scripts.run_global_99_controlled_default_policy_candidate_installation_preflight import (
            run_global_99_controlled_default_policy_candidate_installation_preflight,
        )

        self._write_config({"require_installation_review_only": False})
        self._write_consumer()
        summary = run_global_99_controlled_default_policy_candidate_installation_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("controlled_installation_review_scope_invalid", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "resolve_controlled_default_policy_candidate_installation_preflight_rejections")

    def test_boundary_violation_blocks_preflight(self) -> None:
        from scripts.run_global_99_controlled_default_policy_candidate_installation_preflight import (
            run_global_99_controlled_default_policy_candidate_installation_preflight,
        )

        self._write_consumer({"replaces_default_policy": True})
        summary = run_global_99_controlled_default_policy_candidate_installation_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("controlled_installation_boundary_violation", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "resolve_controlled_default_policy_candidate_installation_preflight_rejections")

    def _write_config(self, updates: dict | None = None) -> None:
        payload = {
            "schema_version": "global-99-controlled-default-policy-candidate-installation-preflight-config/v1",
            "source_sandbox_consumer_root": str(self.consumer_root),
            "require_sandbox_consumer_passed": True,
            "require_installation_review_only": True,
            "require_default_policy_read_only": True,
            "require_executor_isolated": True,
            "require_online_canary_disabled": True,
            "require_kill_switch": True,
            "require_rollback": True,
            "require_telemetry": True,
            "controlled_installation_preflight_only": True,
            "canary_traffic_fraction": 0.0,
        }
        if updates:
            payload.update(updates)
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_consumer(self, updates: dict | None = None) -> None:
        payload = {
            "schema_version": "global-99-sandbox-consumer-replay-canary-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "sandbox_consumer_verdict": "eligible_for_controlled_default_policy_candidate_installation_preflight",
            "next_required_change": "controlled_default_policy_candidate_installation_preflight",
            "sandbox_consumer_replay_canary_passed": True,
            "consumer_step_count": 64,
            "fallback_rate": 0.0,
            "controlled_regression_count": 0,
            "candidate_load_audit_passed": True,
            "fallback_audit_passed": True,
            "telemetry_audit_passed": True,
            "rollback_audit_passed": True,
            "boundary_audit_passed": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
            "default_policy_replacement_approved": False,
            "real_executor_connection_approved": False,
            "default_policy_candidate_installation_approved": False,
        }
        if updates:
            payload.update(updates)
        (self.consumer_root / "global-99-sandbox-consumer-replay-canary-summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
