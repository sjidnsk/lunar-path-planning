import json
import shutil
import tempfile
import unittest
from pathlib import Path


class DefaultPolicyCandidateAuthorizationPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="default-policy-candidate-"))
        self.stage15_root = self.temp_dir / "stage15"
        self.output_root = self.temp_dir / "s16"
        self.stage15_root.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_authorizes_default_policy_candidate_sandbox_install_preflight_only(self) -> None:
        from scripts.run_default_policy_candidate_authorization_preflight import (
            run_default_policy_candidate_authorization_preflight,
        )

        self._write_stage15()

        summary = run_default_policy_candidate_authorization_preflight(
            stage15_root=self.stage15_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(
            summary["authorization_verdict"],
            "eligible_for_default_policy_candidate_sandbox_install_preflight",
        )
        self.assertTrue(summary["default_policy_candidate_authorization_preflight_passed"])
        self.assertTrue(summary["default_policy_candidate_sandbox_install_preflight_approved"])
        self.assertTrue(summary["kill_switch_audit_passed"])
        self.assertTrue(summary["rollback_audit_passed"])
        self.assertTrue(summary["default_policy_boundary_audit_passed"])
        self.assertTrue(summary["path_planner_isolation_audit_passed"])
        self.assertTrue(summary["release_boundary_audit_passed"])
        self.assertFalse(summary["checkpoint_publication_approved"])
        self.assertFalse(summary["default_policy_replacement_approved"])
        self.assertFalse(summary["real_executor_connection_approved"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertEqual(summary["next_required_change"], "default_policy_candidate_sandbox_install_preflight")

    def test_stage15_failure_blocks_authorization(self) -> None:
        from scripts.run_default_policy_candidate_authorization_preflight import (
            run_default_policy_candidate_authorization_preflight,
        )

        self._write_stage15({"status": "failed", "reason_codes": ["fallback_dominates_consumer_replay"]})

        summary = run_default_policy_candidate_authorization_preflight(
            stage15_root=self.stage15_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("stage15_not_passed", summary["reason_codes"])

    def test_unstable_consumer_evidence_blocks_authorization(self) -> None:
        from scripts.run_default_policy_candidate_authorization_preflight import (
            run_default_policy_candidate_authorization_preflight,
        )

        self._write_stage15({"fallback_rate": 0.75})

        summary = run_default_policy_candidate_authorization_preflight(
            stage15_root=self.stage15_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("consumer_evidence_unstable", summary["reason_codes"])

    def test_kill_switch_missing_blocks_authorization(self) -> None:
        from scripts.run_default_policy_candidate_authorization_preflight import (
            run_default_policy_candidate_authorization_preflight,
        )

        self._write_stage15({"default_policy_candidate_authorization_preflight_approved": False})

        summary = run_default_policy_candidate_authorization_preflight(
            stage15_root=self.stage15_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("kill_switch_missing", summary["reason_codes"])

    def test_default_policy_boundary_violation_blocks_authorization(self) -> None:
        from scripts.run_default_policy_candidate_authorization_preflight import (
            run_default_policy_candidate_authorization_preflight,
        )

        self._write_stage15({"replaces_default_policy": True})

        summary = run_default_policy_candidate_authorization_preflight(
            stage15_root=self.stage15_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("default_policy_boundary_violation", summary["reason_codes"])

    def test_path_planner_isolation_violation_blocks_authorization(self) -> None:
        from scripts.run_default_policy_candidate_authorization_preflight import (
            run_default_policy_candidate_authorization_preflight,
        )

        self._write_stage15({"connects_real_executor": True})

        summary = run_default_policy_candidate_authorization_preflight(
            stage15_root=self.stage15_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("path_planner_isolation_violation", summary["reason_codes"])

    def test_rollback_boundary_invalid_blocks_authorization(self) -> None:
        from scripts.run_default_policy_candidate_authorization_preflight import (
            run_default_policy_candidate_authorization_preflight,
        )

        self._write_stage15({"rollback_audit_passed": False})

        summary = run_default_policy_candidate_authorization_preflight(
            stage15_root=self.stage15_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("rollback_boundary_invalid", summary["reason_codes"])

    def test_release_boundary_violation_blocks_authorization(self) -> None:
        from scripts.run_default_policy_candidate_authorization_preflight import (
            run_default_policy_candidate_authorization_preflight,
        )

        self._write_stage15({"default_policy_replacement_approved": True})

        summary = run_default_policy_candidate_authorization_preflight(
            stage15_root=self.stage15_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("release_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["default_policy_replacement_approved"])

    def test_docs_spec_and_shell_are_declared(self) -> None:
        shell_path = self.repo_root / "scripts" / "run_default_policy_candidate_authorization_preflight.sh"
        spec_path = (
            self.repo_root
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-06-16-default-policy-candidate-authorization-preflight.md"
        )
        self.assertTrue(shell_path.is_file())
        self.assertTrue(spec_path.is_file())

    def test_docs_missing_blocks_authorization(self) -> None:
        from scripts.run_default_policy_candidate_authorization_preflight import (
            run_default_policy_candidate_authorization_preflight,
        )

        self._write_stage15()
        fake_repo = self.temp_dir / "docless"
        fake_repo.mkdir()

        summary = run_default_policy_candidate_authorization_preflight(
            stage15_root=self.stage15_root,
            output_root=self.output_root,
            repo_root=fake_repo,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("docs_not_updated", summary["reason_codes"])

    def _write_stage15(self, updates: dict | None = None) -> None:
        summary = {
            "status": "passed",
            "reason_codes": [],
            "consumer_replay_canary_verdict": "eligible_for_default_policy_candidate_authorization_preflight",
            "checkpoint_publication_sandbox_consumer_replay_canary_passed": True,
            "default_policy_candidate_authorization_preflight_approved": True,
            "consumer_step_count": 64,
            "fallback_rate": 0.0,
            "controlled_regression_count": 0,
            "telemetry_audit_passed": True,
            "rollback_audit_passed": True,
            "release_boundary_audit_passed": True,
            "sandbox_consumer_checkpoint_sha256": "abc123",
            "sandbox_consumer_checkpoint_size_bytes": 17189,
            "checkpoint_publication_approved": False,
            "default_policy_replacement_approved": False,
            "real_executor_connection_approved": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "next_required_change": "default_policy_candidate_authorization_preflight",
        }
        if updates:
            summary.update(updates)
        self._write_json(
            self.stage15_root / "checkpoint-publication-sandbox-consumer-replay-canary-summary.json",
            summary,
        )
        for filename, payload in {
            "checkpoint-publication-sandbox-consumer-telemetry-audit.json": {"telemetry_audit_passed": True},
            "checkpoint-publication-sandbox-consumer-rollback-audit.json": {"rollback_audit_passed": True},
            "checkpoint-publication-sandbox-consumer-release-boundary-audit.json": {"release_boundary_audit_passed": True},
        }.items():
            self._write_json(self.stage15_root / filename, payload)

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
