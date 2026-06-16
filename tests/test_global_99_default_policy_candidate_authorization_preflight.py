import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class Global99DefaultPolicyCandidateAuthorizationPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="global-99-default-auth-"))
        self.multi_roi_root = self.temp_dir / "multi-roi"
        self.output_root = self.temp_dir / "out"
        self.config_path = self.temp_dir / "config.json"
        self.multi_roi_root.mkdir(parents=True)
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_current_multi_roi_evidence_authorizes_sandbox_dry_run_only(self) -> None:
        from scripts.run_global_99_default_policy_candidate_authorization_preflight import (
            run_global_99_default_policy_candidate_authorization_preflight,
        )

        self._write_multi_roi()
        summary = run_global_99_default_policy_candidate_authorization_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(summary["default_policy_candidate_authorization_preflight_passed"])
        self.assertEqual(summary["authorization_verdict"], "eligible_for_sandbox_candidate_installation_dry_run")
        self.assertEqual(summary["next_required_change"], "sandbox_candidate_installation_dry_run")
        self.assertEqual(summary["source_multi_roi_status"], "passed")
        self.assertEqual(summary["source_multi_roi_next_required_change"], "default_policy_candidate_authorization_preflight")
        self.assertTrue(summary["candidate_read_only_audit_passed"])
        self.assertTrue(summary["default_policy_read_only_audit_passed"])
        self.assertTrue(summary["executor_isolation_audit_passed"])
        self.assertTrue(summary["path_planner_isolation_audit_passed"])
        self.assertTrue(summary["kill_switch_audit_passed"])
        self.assertTrue(summary["rollback_audit_passed"])
        self.assertTrue(summary["telemetry_audit_passed"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertEqual(summary["canary_traffic_fraction"], 0.0)
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])
        self.assertFalse(summary["modifies_action_space"])
        self.assertFalse(summary["modifies_default_astar"])
        self.assertFalse(summary["default_policy_replacement_approved"])
        self.assertFalse(summary["real_executor_connection_approved"])

        for filename in (
            "global-99-default-policy-candidate-authorization-preflight-summary.json",
            "global-99-default-policy-candidate-authorization-manifest.json",
            "global-99-default-policy-candidate-lineage-audit.json",
            "global-99-default-policy-candidate-scope-audit.json",
            "global-99-default-policy-candidate-read-only-audit.json",
            "global-99-default-policy-read-only-audit.json",
            "global-99-default-policy-candidate-isolation-audit.json",
            "global-99-default-policy-candidate-kill-switch-audit.json",
            "global-99-default-policy-candidate-rollback-audit.json",
            "global-99-default-policy-candidate-telemetry-audit.json",
            "global-99-default-policy-candidate-rejection-report.json",
            "global-99-default-policy-candidate-authorization-preflight-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_multi_roi_routes_to_multi_roi_fix(self) -> None:
        from scripts.run_global_99_default_policy_candidate_authorization_preflight import (
            run_global_99_default_policy_candidate_authorization_preflight,
        )

        summary = run_global_99_default_policy_candidate_authorization_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_multi_roi_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_real_map_multi_roi_generalization")

    def test_failed_multi_roi_routes_to_multi_roi_fix(self) -> None:
        from scripts.run_global_99_default_policy_candidate_authorization_preflight import (
            run_global_99_default_policy_candidate_authorization_preflight,
        )

        self._write_multi_roi({"status": "failed", "reason_codes": ["roi_failed"]})
        summary = run_global_99_default_policy_candidate_authorization_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("multi_roi_not_passed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_real_map_multi_roi_generalization")

    def test_read_only_failure_blocks_authorization(self) -> None:
        from scripts.run_global_99_default_policy_candidate_authorization_preflight import (
            run_global_99_default_policy_candidate_authorization_preflight,
        )

        self._write_config({"require_candidate_read_only": False})
        self._write_multi_roi()
        summary = run_global_99_default_policy_candidate_authorization_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("candidate_read_only_boundary_invalid", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "resolve_default_policy_candidate_authorization_rejections")

    def test_boundary_violation_blocks_authorization(self) -> None:
        from scripts.run_global_99_default_policy_candidate_authorization_preflight import (
            run_global_99_default_policy_candidate_authorization_preflight,
        )

        self._write_multi_roi({"connects_real_executor": True})
        summary = run_global_99_default_policy_candidate_authorization_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("authorization_boundary_violation", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "resolve_default_policy_candidate_authorization_rejections")

    def _write_config(self, updates: dict | None = None) -> None:
        payload = {
            "schema_version": "global-99-default-policy-candidate-authorization-preflight-config/v1",
            "source_multi_roi_root": str(self.multi_roi_root),
            "require_multi_roi_passed": True,
            "require_default_policy_read_only": True,
            "require_candidate_read_only": True,
            "require_executor_isolated": True,
            "require_path_planner_isolated": True,
            "require_kill_switch": True,
            "require_rollback": True,
            "require_telemetry": True,
            "authorization_preflight_only": True,
            "canary_traffic_fraction": 0.0,
        }
        if updates:
            payload.update(updates)
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_multi_roi(self, updates: dict | None = None) -> None:
        payload = {
            "schema_version": "global-99-real-map-multi-roi-generalization-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "real_map_multi_roi_generalization_verdict": "eligible_for_default_policy_candidate_authorization_preflight",
            "next_required_change": "default_policy_candidate_authorization_preflight",
            "slice_count": 12,
            "roi_group_count": 4,
            "passed_roi_group_count": 4,
            "failed_roi_group_count": 0,
            "passed_required_scenario_count": 12,
            "failed_required_scenario_count": 0,
            "split_coverage_complete": True,
            "context_id_missing_count": 0,
            "legacy_identity_fallback_count": 0,
            "missing_contract_count": 0,
            "missing_sidecar_count": 0,
            "fallback_or_open_grid_count": 0,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
            "real_world_release_approved": False,
            "real_world_performance_claimed": False,
            "default_policy_replacement_approved": False,
            "real_executor_connection_approved": False,
        }
        if updates:
            payload.update(updates)
        (self.multi_roi_root / "global-99-real-map-multi-roi-generalization-summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
