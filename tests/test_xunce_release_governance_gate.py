import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceReleaseGovernanceGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(tempfile.mkdtemp(prefix="xunce-release-governance-"))
        scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.sandbox_root = self.repo_root / "outputs" / "stage16"
        self.output_root = self.repo_root / "outputs" / "stage17"
        self.config_path = self.repo_root / "configs" / "xunce_release_governance_gate_v1.json"
        self._write_sandbox_summary()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.repo_root)

    def test_default_fixture_marks_research_chain_complete_without_release_approval(self) -> None:
        from scripts.run_xunce_release_governance_gate import run_xunce_release_governance_gate

        summary = run_xunce_release_governance_gate(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["next_required_change"], "xunce_research_track_complete")
        self.assertTrue(summary["release_governance_gate_passed"])
        self.assertEqual(summary["release_governance_verdict"], "research_candidate_ready_for_human_governance_review")
        self.assertTrue(summary["xunce_research_chain_complete"])
        self.assertFalse(summary["default_policy_replacement_approved"])
        self.assertFalse(summary["real_world_release_approved"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])

        for filename in (
            "xunce-release-governance-gate-summary.json",
            "xunce-release-governance-gate-manifest.json",
            "xunce-release-evidence-lineage-audit.json",
            "xunce-release-scope-audit.json",
            "xunce-release-boundary-audit.json",
            "xunce-release-governance-decision-audit.json",
            "xunce-release-rejection-report.json",
            "xunce-release-governance-gate-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_sandbox_source_routes_to_sandbox_fix(self) -> None:
        from scripts.run_xunce_release_governance_gate import run_xunce_release_governance_gate

        shutil.rmtree(self.sandbox_root)
        summary = run_xunce_release_governance_gate(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_sandbox_candidate_preflight_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_sandbox_candidate_preflight")

    def test_wrong_sandbox_next_routes_to_sandbox_fix(self) -> None:
        from scripts.run_xunce_release_governance_gate import run_xunce_release_governance_gate

        self._write_sandbox_summary({"next_required_change": "something_else"})
        summary = run_xunce_release_governance_gate(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sandbox_candidate_wrong_next_required_change", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_sandbox_candidate_preflight")

    def test_boundary_violation_blocks_release_governance(self) -> None:
        from scripts.run_xunce_release_governance_gate import run_xunce_release_governance_gate

        self._write_sandbox_summary({"replaces_default_policy": True})
        summary = run_xunce_release_governance_gate(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("release_governance_boundary_violation", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "resolve_xunce_release_governance_boundary_rejections")

    def test_missing_governance_safety_audit_fails_gate(self) -> None:
        from scripts.run_xunce_release_governance_gate import run_xunce_release_governance_gate

        self._write_sandbox_summary({"kill_switch_audit_passed": False})
        summary = run_xunce_release_governance_gate(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sandbox_kill_switch_not_passed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_release_governance_gate")

    def _write_config(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "xunce-release-governance-gate-config/v1",
            "source_sandbox_candidate_root": str(self.sandbox_root),
            "require_sandbox_preflight_passed": True,
            "require_kill_switch": True,
            "require_rollback": True,
            "require_telemetry": True,
            "require_default_policy_read_only": True,
            "require_executor_isolation": True,
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_sandbox_summary(self, updates: dict | None = None) -> None:
        payload = {
            "schema_version": "xunce-sandbox-candidate-preflight-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "xunce_release_governance_gate",
            "sandbox_candidate_preflight_passed": True,
            "sandbox_package_created": True,
            "sandbox_checkpoint_hash_verified": True,
            "sandbox_load_verified": True,
            "kill_switch_audit_passed": True,
            "rollback_audit_passed": True,
            "telemetry_audit_passed": True,
            "default_policy_read_only": True,
            "executor_isolation_passed": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "runs_new_training_update": False,
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
        path = self.sandbox_root / "xunce-sandbox-candidate-preflight-summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
