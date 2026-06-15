import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class CheckpointPublicationSandboxInstallDryRunPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts = str(self.repo_root / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="checkpoint-sandbox-preflight-"))
        self.stage11_root = self.temp_dir / "stage11"
        self.stage10_root = self.temp_dir / "stage10"
        self.stage9_root = self.temp_dir / "stage9"
        self.stage8_root = self.temp_dir / "stage8"
        self.stage7_root = self.temp_dir / "stage7"
        self.output_root = self.temp_dir / "output"
        for path in (
            self.stage11_root,
            self.stage10_root,
            self.stage9_root,
            self.stage8_root,
            self.stage7_root,
        ):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_preflight_approves_sandbox_dry_run_without_install_or_publication(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_preflight import (
            run_checkpoint_publication_sandbox_install_dry_run_preflight,
        )

        self._write_inputs()

        summary = run_checkpoint_publication_sandbox_install_dry_run_preflight(
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(
            summary["schema_version"],
            "checkpoint-publication-sandbox-install-dry-run-preflight-summary/v1",
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(
            summary["preflight_verdict"],
            "eligible_for_checkpoint_publication_sandbox_install_dry_run",
        )
        self.assertTrue(summary["checkpoint_publication_sandbox_install_dry_run_preflight_passed"])
        self.assertTrue(summary["checkpoint_publication_sandbox_install_dry_run_approved"])
        self.assertTrue(summary["sandbox_preflight_manifest_passed"])
        self.assertTrue(summary["package_consumer_audit_passed"])
        self.assertTrue(summary["default_policy_boundary_audit_passed"])
        self.assertTrue(summary["sandbox_path_boundary_audit_passed"])
        self.assertTrue(summary["lineage_audit_passed"])
        self.assertTrue(summary["rollback_preflight_audit_passed"])
        self.assertTrue(summary["release_boundary_audit_passed"])
        self.assertEqual(summary["selected_seed"], 0)
        self.assertEqual(summary["selected_budget"], "epochs1_lr3e-6")
        self.assertEqual(summary["package_checkpoint_sha256"], self.package_sha256)
        self.assertEqual(summary["package_checkpoint_size_bytes"], self.package_size)
        self.assertEqual(summary["next_required_change"], "checkpoint_publication_sandbox_install_dry_run")
        self.assertFalse(summary["checkpoint_publication_approved"])
        self.assertFalse(summary["default_policy_replacement_approved"])
        self.assertFalse(summary["real_executor_connection_approved"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])

        for field in (
            "sandbox_preflight_manifest",
            "package_consumer_audit",
            "default_policy_boundary_audit",
            "sandbox_path_boundary_audit",
            "sandbox_lineage_audit",
            "sandbox_release_boundary_audit",
            "rollback_preflight_audit",
            "rejection_report",
            "report",
        ):
            self.assertTrue(Path(summary[field]).is_file(), field)

        manifest = json.loads(Path(summary["sandbox_preflight_manifest"]).read_text(encoding="utf-8"))
        planned_checkpoint = Path(manifest["planned_consumer_checkpoint_path"])
        self.assertTrue(str(planned_checkpoint).startswith(str(self.output_root)))
        self.assertFalse(planned_checkpoint.exists(), "Stage 12 must not copy checkpoint into sandbox")

    def test_stage11_failure_blocks_preflight(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_preflight import (
            run_checkpoint_publication_sandbox_install_dry_run_preflight,
        )

        self._write_inputs()
        self._patch_stage11_summary({"status": "failed", "reason_codes": ["package_hash_mismatch"]})

        summary = run_checkpoint_publication_sandbox_install_dry_run_preflight(
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("stage11_not_passed", summary["reason_codes"])

    def test_wrong_stage11_next_gate_blocks_preflight(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_preflight import (
            run_checkpoint_publication_sandbox_install_dry_run_preflight,
        )

        self._write_inputs()
        self._patch_stage11_summary({"next_required_change": "checkpoint_publication_sandbox_install_dry_run"})

        summary = run_checkpoint_publication_sandbox_install_dry_run_preflight(
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("stage11_not_authorized_for_sandbox_preflight", summary["reason_codes"])

    def test_missing_consumer_manifest_blocks_preflight(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_preflight import (
            run_checkpoint_publication_sandbox_install_dry_run_preflight,
        )

        self._write_inputs()
        self.consumer_manifest_path.unlink()

        summary = run_checkpoint_publication_sandbox_install_dry_run_preflight(
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("consumer_manifest_missing", summary["reason_codes"])

    def test_missing_package_checkpoint_blocks_preflight(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_preflight import (
            run_checkpoint_publication_sandbox_install_dry_run_preflight,
        )

        self._write_inputs()
        self.package_checkpoint_path.unlink()

        summary = run_checkpoint_publication_sandbox_install_dry_run_preflight(
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("package_checkpoint_missing", summary["reason_codes"])

    def test_package_identity_mismatch_blocks_preflight(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_preflight import (
            run_checkpoint_publication_sandbox_install_dry_run_preflight,
        )

        self._write_inputs()
        self.package_checkpoint_path.write_bytes(b"mutated-package")

        summary = run_checkpoint_publication_sandbox_install_dry_run_preflight(
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("package_identity_mismatch", summary["reason_codes"])

    def test_package_load_not_verified_blocks_preflight(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_preflight import (
            run_checkpoint_publication_sandbox_install_dry_run_preflight,
        )

        self._write_inputs()
        self._write_json(
            self.load_audit_path,
            {"package_load_verification_audit_passed": False, "reason_codes": ["package_load_verification_failed"]},
        )

        summary = run_checkpoint_publication_sandbox_install_dry_run_preflight(
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("package_load_not_verified", summary["reason_codes"])

    def test_sandbox_path_boundary_violation_blocks_preflight(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_preflight import (
            run_checkpoint_publication_sandbox_install_dry_run_preflight,
        )

        self._write_inputs()
        unsafe_output_root = self.temp_dir / "default_policy_live_output"

        summary = run_checkpoint_publication_sandbox_install_dry_run_preflight(
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=unsafe_output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sandbox_path_boundary_violation", summary["reason_codes"])

    def test_default_policy_boundary_violation_blocks_preflight(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_preflight import (
            run_checkpoint_publication_sandbox_install_dry_run_preflight,
        )

        self._write_inputs()
        default_policy_path = self.temp_dir / "simulated-default-policy.pt"
        default_policy_path.write_bytes(b"default-policy-before")

        summary = run_checkpoint_publication_sandbox_install_dry_run_preflight(
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            default_policy_path=default_policy_path,
            default_policy_mutator=lambda path: path.write_bytes(b"default-policy-after"),
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("default_policy_boundary_violation", summary["reason_codes"])

    def test_lineage_failure_blocks_preflight(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_preflight import (
            run_checkpoint_publication_sandbox_install_dry_run_preflight,
        )

        self._write_inputs()
        self._write_json(self.lineage_audit_path, {"lineage_verification_audit_passed": False})

        summary = run_checkpoint_publication_sandbox_install_dry_run_preflight(
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("lineage_incomplete", summary["reason_codes"])

    def test_rollback_preflight_failure_blocks_preflight(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_preflight import (
            run_checkpoint_publication_sandbox_install_dry_run_preflight,
        )

        self._write_inputs()

        summary = run_checkpoint_publication_sandbox_install_dry_run_preflight(
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            assume_rollback_preflight_valid=False,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("rollback_preflight_invalid", summary["reason_codes"])

    def test_release_boundary_violation_blocks_preflight(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_preflight import (
            run_checkpoint_publication_sandbox_install_dry_run_preflight,
        )

        self._write_inputs()
        self._patch_stage11_summary({"checkpoint_publication_approved": True})

        summary = run_checkpoint_publication_sandbox_install_dry_run_preflight(
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("release_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["checkpoint_publication_approved"])

    def test_docs_missing_blocks_preflight(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_preflight import (
            run_checkpoint_publication_sandbox_install_dry_run_preflight,
        )

        self._write_inputs()
        fake_repo = self.temp_dir / "docless_repo"
        fake_repo.mkdir()

        summary = run_checkpoint_publication_sandbox_install_dry_run_preflight(
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=fake_repo,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("docs_not_updated", summary["reason_codes"])

    def test_shell_docs_and_spec_contract_are_declared(self) -> None:
        shell_path = self.repo_root / "scripts" / "run_checkpoint_publication_sandbox_install_dry_run_preflight.sh"
        spec_path = (
            self.repo_root
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-06-16-checkpoint-publication-sandbox-install-dry-run-preflight.md"
        )
        self.assertTrue(shell_path.is_file())
        self.assertTrue(spec_path.is_file())
        spec_text = spec_path.read_text(encoding="utf-8")
        self.assertIn("checkpoint-publication-sandbox-install-dry-run-preflight-summary.json", spec_text)
        self.assertIn("README.md", spec_text)
        self.assertIn("docs/算法设计与系统架构报告.md", spec_text)
        self.assertIn("checkpoint_publication_sandbox_install_dry_run", spec_text)
        self.assertIn("不复制到发布/default/live/executor 路径", spec_text)

    def _write_inputs(self) -> None:
        self.package_root = self.stage10_root / "checkpoint-publication-package"
        self.package_root.mkdir(parents=True)
        self.package_checkpoint_path = self.package_root / "experimental-hybrid-policy-candidate.pt"
        self.package_checkpoint_path.write_bytes(b"stage12-package-checkpoint")
        self.package_sha256 = hashlib.sha256(self.package_checkpoint_path.read_bytes()).hexdigest()
        self.package_size = self.package_checkpoint_path.stat().st_size
        self.package_metadata_path = self.package_root / "experimental-hybrid-policy-candidate-metadata.json"
        self._write_json(
            self.package_metadata_path,
            {
                "schema_version": "controlled-hybrid-policy-candidate-checkpoint-metadata/v1",
                "experimental": True,
                "seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "checkpoint_path": "source/provenance/checkpoint.pt",
                "checkpoint_publication_approved": False,
                "default_policy_replacement_approved": False,
                "real_executor_connection_approved": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "performance_claimed": False,
                "final_release_approved": False,
            },
        )
        self.stage10_manifest_path = self.stage10_root / "checkpoint-publication-package-manifest.json"
        self._write_json(
            self.stage10_manifest_path,
            {
                "schema_version": "checkpoint-publication-package-manifest/v1",
                "selected_seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "package_checkpoint_path": str(self.package_checkpoint_path),
                "package_metadata_path": str(self.package_metadata_path),
                "package_checkpoint_sha256": self.package_sha256,
                "package_checkpoint_size_bytes": self.package_size,
                "next_required_change": "checkpoint_publication_package_verification",
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "non_goals": [
                    "do_not_publish_checkpoint",
                    "do_not_replace_default_policy",
                    "do_not_connect_real_executor",
                ],
            },
        )
        self.consumer_manifest_path = self.stage11_root / "checkpoint-publication-package-consumer-manifest.json"
        self.integrity_audit_path = self.stage11_root / "checkpoint-publication-package-integrity-audit.json"
        self.load_audit_path = self.stage11_root / "checkpoint-publication-package-load-verification-audit.json"
        self.lineage_audit_path = self.stage11_root / "checkpoint-publication-package-lineage-verification-audit.json"
        self.release_audit_path = self.stage11_root / "checkpoint-publication-package-release-boundary-audit.json"
        self.rollback_audit_path = self.stage11_root / "checkpoint-publication-package-rollback-verification-audit.json"
        self._write_json(
            self.consumer_manifest_path,
            {
                "schema_version": "checkpoint-publication-package-consumer-manifest/v1",
                "verification_verdict": "verified_for_checkpoint_publication_sandbox_install_dry_run_preflight",
                "stage10_manifest": str(self.stage10_manifest_path),
                "selected_seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "authoritative_package_checkpoint_path": str(self.package_checkpoint_path),
                "package_metadata_path": str(self.package_metadata_path),
                "package_checkpoint_sha256": self.package_sha256,
                "package_checkpoint_size_bytes": self.package_size,
                "package_load_verification_audit_passed": True,
                "next_required_change": "checkpoint_publication_sandbox_install_dry_run_preflight",
                "checkpoint_publication_approved": False,
                "default_policy_replacement_approved": False,
                "real_executor_connection_approved": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.integrity_audit_path,
            {
                "package_integrity_audit_passed": True,
                "reason_codes": [],
                "package_checkpoint_path": str(self.package_checkpoint_path),
                "package_checkpoint_sha256": self.package_sha256,
                "package_checkpoint_size_bytes": self.package_size,
            },
        )
        self._write_json(self.load_audit_path, {"package_load_verification_audit_passed": True, "reason_codes": []})
        self._write_json(self.lineage_audit_path, {"lineage_verification_audit_passed": True, "reason_codes": []})
        self._write_json(
            self.release_audit_path,
            {
                "release_boundary_audit_passed": True,
                "reason_codes": [],
                "checkpoint_publication_approved": False,
                "default_policy_replacement_approved": False,
                "real_executor_connection_approved": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.rollback_audit_path,
            {
                "rollback_verification_audit_passed": True,
                "reason_codes": [],
                "package_checkpoint_unchanged": True,
                "package_metadata_unchanged": True,
                "default_policy_touched": False,
            },
        )
        self._write_json(
            self.stage11_root / "checkpoint-publication-package-verification-summary.json",
            {
                "schema_version": "checkpoint-publication-package-verification-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "verification_verdict": "verified_for_checkpoint_publication_sandbox_install_dry_run_preflight",
                "checkpoint_publication_package_verification_passed": True,
                "checkpoint_publication_sandbox_install_dry_run_preflight_approved": True,
                "selected_seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "package_checkpoint_sha256": self.package_sha256,
                "package_checkpoint_size_bytes": self.package_size,
                "consumer_manifest": str(self.consumer_manifest_path),
                "package_integrity_audit": str(self.integrity_audit_path),
                "package_load_verification_audit": str(self.load_audit_path),
                "package_lineage_verification_audit": str(self.lineage_audit_path),
                "package_release_boundary_audit": str(self.release_audit_path),
                "package_rollback_verification_audit": str(self.rollback_audit_path),
                "next_required_change": "checkpoint_publication_sandbox_install_dry_run_preflight",
                "checkpoint_publication_approved": False,
                "default_policy_replacement_approved": False,
                "real_executor_connection_approved": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.stage9_root / "checkpoint-publication-authorization-preflight-summary.json",
            {"status": "passed", "reason_codes": [], "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False},
        )
        self._write_json(
            self.stage8_root / "scoped-claim-publication-evidence-freeze-summary.json",
            {"status": "passed", "reason_codes": [], "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False},
        )
        self._write_json(
            self.stage7_root / "formal-performance-claim-release-decision-summary.json",
            {"status": "passed", "reason_codes": [], "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False},
        )

    def _patch_stage11_summary(self, updates: dict) -> None:
        path = self.stage11_root / "checkpoint-publication-package-verification-summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(updates)
        self._write_json(path, payload)

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
