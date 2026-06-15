import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class CheckpointPublicationSandboxInstallDryRunVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts = str(self.repo_root / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="checkpoint-sandbox-install-verification-"))
        self.stage13_root = self.temp_dir / "stage13"
        self.stage12_root = self.temp_dir / "stage12"
        self.stage11_root = self.temp_dir / "stage11"
        self.stage10_root = self.temp_dir / "stage10"
        self.stage9_root = self.temp_dir / "stage9"
        self.stage8_root = self.temp_dir / "stage8"
        self.stage7_root = self.temp_dir / "stage7"
        self.output_root = self.temp_dir / "output"
        for path in (
            self.stage13_root,
            self.stage12_root,
            self.stage11_root,
            self.stage10_root,
            self.stage9_root,
            self.stage8_root,
            self.stage7_root,
        ):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_verifies_sandbox_install_without_mutating_files_or_publication(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_verification import (
            run_checkpoint_publication_sandbox_install_dry_run_verification,
        )

        self._write_inputs()
        source_before = self.package_checkpoint_path.read_bytes()
        sandbox_before = self.sandbox_checkpoint_path.read_bytes()

        summary = run_checkpoint_publication_sandbox_install_dry_run_verification(
            stage13_root=self.stage13_root,
            stage12_root=self.stage12_root,
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
            "checkpoint-publication-sandbox-install-dry-run-verification-summary/v1",
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(
            summary["verification_verdict"],
            "verified_for_checkpoint_publication_sandbox_consumer_smoke_preflight",
        )
        self.assertTrue(summary["checkpoint_publication_sandbox_install_dry_run_verification_passed"])
        self.assertTrue(summary["checkpoint_publication_sandbox_consumer_smoke_preflight_approved"])
        self.assertTrue(summary["sandbox_installed_file_identity_audit_passed"])
        self.assertTrue(summary["sandbox_install_manifest_consistency_audit_passed"])
        self.assertTrue(summary["sandbox_consumer_load_reverification_audit_passed"])
        self.assertTrue(summary["sandbox_metadata_verification_audit_passed"])
        self.assertTrue(summary["lineage_verification_audit_passed"])
        self.assertTrue(summary["rollback_verification_audit_passed"])
        self.assertTrue(summary["release_boundary_audit_passed"])
        self.assertEqual(summary["selected_seed"], 0)
        self.assertEqual(summary["selected_budget"], "epochs1_lr3e-6")
        self.assertEqual(summary["source_package_checkpoint_sha256"], self.checkpoint_sha256)
        self.assertEqual(summary["sandbox_consumer_checkpoint_sha256"], self.checkpoint_sha256)
        self.assertEqual(summary["source_package_checkpoint_size_bytes"], self.checkpoint_size)
        self.assertEqual(summary["sandbox_consumer_checkpoint_size_bytes"], self.checkpoint_size)
        self.assertEqual(summary["next_required_change"], "checkpoint_publication_sandbox_consumer_smoke_preflight")
        self.assertFalse(summary["checkpoint_publication_approved"])
        self.assertFalse(summary["default_policy_replacement_approved"])
        self.assertFalse(summary["real_executor_connection_approved"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertEqual(self.package_checkpoint_path.read_bytes(), source_before)
        self.assertEqual(self.sandbox_checkpoint_path.read_bytes(), sandbox_before)

        for field in (
            "consumer_verification_manifest",
            "installed_file_identity_audit",
            "install_manifest_consistency_audit",
            "consumer_load_reverification_audit",
            "metadata_verification_audit",
            "lineage_verification_audit",
            "sandbox_release_boundary_audit",
            "rollback_verification_audit",
            "rejection_report",
            "report",
        ):
            self.assertTrue(Path(summary[field]).is_file(), field)

    def test_stage13_failure_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_verification import (
            run_checkpoint_publication_sandbox_install_dry_run_verification,
        )

        self._write_inputs()
        self._patch_stage13_summary({"status": "failed", "reason_codes": ["sandbox_consumer_load_failed"]})

        summary = run_checkpoint_publication_sandbox_install_dry_run_verification(
            stage13_root=self.stage13_root,
            stage12_root=self.stage12_root,
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("stage13_not_passed", summary["reason_codes"])

    def test_wrong_stage13_next_gate_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_verification import (
            run_checkpoint_publication_sandbox_install_dry_run_verification,
        )

        self._write_inputs()
        self._patch_stage13_summary({"next_required_change": "checkpoint_publication"})

        summary = run_checkpoint_publication_sandbox_install_dry_run_verification(
            stage13_root=self.stage13_root,
            stage12_root=self.stage12_root,
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("stage13_not_ready_for_verification", summary["reason_codes"])

    def test_missing_install_manifest_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_verification import (
            run_checkpoint_publication_sandbox_install_dry_run_verification,
        )

        self._write_inputs()
        self.install_manifest_path.unlink()

        summary = run_checkpoint_publication_sandbox_install_dry_run_verification(
            stage13_root=self.stage13_root,
            stage12_root=self.stage12_root,
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sandbox_install_manifest_missing", summary["reason_codes"])

    def test_missing_sandbox_checkpoint_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_verification import (
            run_checkpoint_publication_sandbox_install_dry_run_verification,
        )

        self._write_inputs()
        self.sandbox_checkpoint_path.unlink()

        summary = run_checkpoint_publication_sandbox_install_dry_run_verification(
            stage13_root=self.stage13_root,
            stage12_root=self.stage12_root,
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sandbox_checkpoint_missing", summary["reason_codes"])

    def test_missing_sandbox_metadata_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_verification import (
            run_checkpoint_publication_sandbox_install_dry_run_verification,
        )

        self._write_inputs()
        self.sandbox_metadata_path.unlink()

        summary = run_checkpoint_publication_sandbox_install_dry_run_verification(
            stage13_root=self.stage13_root,
            stage12_root=self.stage12_root,
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sandbox_metadata_missing", summary["reason_codes"])

    def test_missing_source_checkpoint_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_verification import (
            run_checkpoint_publication_sandbox_install_dry_run_verification,
        )

        self._write_inputs()
        self.package_checkpoint_path.unlink()

        summary = run_checkpoint_publication_sandbox_install_dry_run_verification(
            stage13_root=self.stage13_root,
            stage12_root=self.stage12_root,
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("source_package_checkpoint_missing", summary["reason_codes"])

    def test_missing_source_metadata_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_verification import (
            run_checkpoint_publication_sandbox_install_dry_run_verification,
        )

        self._write_inputs()
        self.package_metadata_path.unlink()

        summary = run_checkpoint_publication_sandbox_install_dry_run_verification(
            stage13_root=self.stage13_root,
            stage12_root=self.stage12_root,
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("source_package_metadata_missing", summary["reason_codes"])

    def test_sandbox_identity_mismatch_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_verification import (
            run_checkpoint_publication_sandbox_install_dry_run_verification,
        )

        self._write_inputs()
        self.sandbox_checkpoint_path.write_bytes(b"mutated-sandbox")

        summary = run_checkpoint_publication_sandbox_install_dry_run_verification(
            stage13_root=self.stage13_root,
            stage12_root=self.stage12_root,
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sandbox_identity_mismatch", summary["reason_codes"])

    def test_source_identity_mismatch_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_verification import (
            run_checkpoint_publication_sandbox_install_dry_run_verification,
        )

        self._write_inputs()
        self.package_checkpoint_path.write_bytes(b"mutated-source")

        summary = run_checkpoint_publication_sandbox_install_dry_run_verification(
            stage13_root=self.stage13_root,
            stage12_root=self.stage12_root,
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("source_package_identity_mismatch", summary["reason_codes"])

    def test_manifest_inconsistency_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_verification import (
            run_checkpoint_publication_sandbox_install_dry_run_verification,
        )

        self._write_inputs()
        manifest = json.loads(self.install_manifest_path.read_text(encoding="utf-8"))
        manifest["sandbox_consumer_checkpoint_sha256"] = "wrong-hash"
        self._write_json(self.install_manifest_path, manifest)

        summary = run_checkpoint_publication_sandbox_install_dry_run_verification(
            stage13_root=self.stage13_root,
            stage12_root=self.stage12_root,
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sandbox_install_manifest_inconsistent", summary["reason_codes"])

    def test_load_reverification_failure_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_verification import (
            run_checkpoint_publication_sandbox_install_dry_run_verification,
        )

        self._write_inputs(loadable=False)

        summary = run_checkpoint_publication_sandbox_install_dry_run_verification(
            stage13_root=self.stage13_root,
            stage12_root=self.stage12_root,
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sandbox_load_reverification_failed", summary["reason_codes"])

    def test_invalid_metadata_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_verification import (
            run_checkpoint_publication_sandbox_install_dry_run_verification,
        )

        self._write_inputs()
        metadata = json.loads(self.sandbox_metadata_path.read_text(encoding="utf-8"))
        metadata["performance_claimed"] = True
        self._write_json(self.sandbox_metadata_path, metadata)

        summary = run_checkpoint_publication_sandbox_install_dry_run_verification(
            stage13_root=self.stage13_root,
            stage12_root=self.stage12_root,
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sandbox_metadata_invalid", summary["reason_codes"])

    def test_default_policy_boundary_violation_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_verification import (
            run_checkpoint_publication_sandbox_install_dry_run_verification,
        )

        self._write_inputs()
        default_policy_path = self.temp_dir / "simulated-default-policy.pt"
        default_policy_path.write_bytes(b"default-policy-before")

        summary = run_checkpoint_publication_sandbox_install_dry_run_verification(
            stage13_root=self.stage13_root,
            stage12_root=self.stage12_root,
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

    def test_rollback_failure_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_verification import (
            run_checkpoint_publication_sandbox_install_dry_run_verification,
        )

        self._write_inputs()

        summary = run_checkpoint_publication_sandbox_install_dry_run_verification(
            stage13_root=self.stage13_root,
            stage12_root=self.stage12_root,
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            assume_rollback_valid=False,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("rollback_boundary_invalid", summary["reason_codes"])

    def test_lineage_failure_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_verification import (
            run_checkpoint_publication_sandbox_install_dry_run_verification,
        )

        self._write_inputs()
        self._write_json(self.stage13_root / "checkpoint-publication-sandbox-lineage-audit.json", {"lineage_audit_passed": False})

        summary = run_checkpoint_publication_sandbox_install_dry_run_verification(
            stage13_root=self.stage13_root,
            stage12_root=self.stage12_root,
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("lineage_verification_incomplete", summary["reason_codes"])

    def test_release_boundary_violation_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_verification import (
            run_checkpoint_publication_sandbox_install_dry_run_verification,
        )

        self._write_inputs()
        self._patch_stage13_summary({"checkpoint_publication_approved": True})

        summary = run_checkpoint_publication_sandbox_install_dry_run_verification(
            stage13_root=self.stage13_root,
            stage12_root=self.stage12_root,
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

    def test_docs_missing_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run_verification import (
            run_checkpoint_publication_sandbox_install_dry_run_verification,
        )

        self._write_inputs()
        fake_repo = self.temp_dir / "docless_repo"
        fake_repo.mkdir()

        summary = run_checkpoint_publication_sandbox_install_dry_run_verification(
            stage13_root=self.stage13_root,
            stage12_root=self.stage12_root,
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
        shell_path = self.repo_root / "scripts" / "run_checkpoint_publication_sandbox_install_dry_run_verification.sh"
        spec_path = (
            self.repo_root
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-06-16-checkpoint-publication-sandbox-install-dry-run-verification.md"
        )
        self.assertTrue(shell_path.is_file())
        self.assertTrue(spec_path.is_file())
        spec_text = spec_path.read_text(encoding="utf-8")
        self.assertIn("checkpoint-publication-sandbox-install-dry-run-verification-summary.json", spec_text)
        self.assertIn("README.md", spec_text)
        self.assertIn("docs/算法设计与系统架构报告.md", spec_text)
        self.assertIn("checkpoint_publication_sandbox_consumer_smoke_preflight", spec_text)
        self.assertIn("不发布 checkpoint", spec_text)

    def _write_inputs(self, *, loadable: bool = True) -> None:
        import torch

        self.package_root = self.stage10_root / "checkpoint-publication-package"
        self.package_root.mkdir(parents=True)
        self.package_checkpoint_path = self.package_root / "experimental-hybrid-policy-candidate.pt"
        payload = {
            "schema_version": "experimental-hybrid-policy-candidate-checkpoint/v1",
            "experimental": True,
        }
        if loadable:
            payload["model_state_dict"] = {"layer.weight": torch.ones(2, 2)}
        else:
            payload["not_a_state_dict"] = {"layer.weight": torch.ones(2, 2)}
        torch.save(payload, self.package_checkpoint_path)
        self.checkpoint_sha256 = hashlib.sha256(self.package_checkpoint_path.read_bytes()).hexdigest()
        self.checkpoint_size = self.package_checkpoint_path.stat().st_size
        self.package_metadata_path = self.package_root / "experimental-hybrid-policy-candidate-metadata.json"
        self._write_metadata(self.package_metadata_path)
        self.stage10_manifest_path = self.stage10_root / "checkpoint-publication-package-manifest.json"
        self._write_json(
            self.stage10_manifest_path,
            {
                "selected_seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "package_checkpoint_path": str(self.package_checkpoint_path),
                "package_metadata_path": str(self.package_metadata_path),
                "package_checkpoint_sha256": self.checkpoint_sha256,
                "package_checkpoint_size_bytes": self.checkpoint_size,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self.sandbox_root = self.stage12_root / "checkpoint-publication-sandbox-install-dry-run-sandbox"
        self.sandbox_root.mkdir(parents=True)
        self.sandbox_checkpoint_path = self.sandbox_root / "experimental-hybrid-policy-candidate.pt"
        self.sandbox_metadata_path = self.sandbox_root / "experimental-hybrid-policy-candidate-metadata.json"
        shutil.copy2(self.package_checkpoint_path, self.sandbox_checkpoint_path)
        shutil.copy2(self.package_metadata_path, self.sandbox_metadata_path)
        self.stage12_manifest_path = self.stage12_root / "checkpoint-publication-sandbox-preflight-manifest.json"
        self._write_json(
            self.stage12_manifest_path,
            {
                "sandbox_preflight_manifest_passed": True,
                "sandbox_root": str(self.sandbox_root),
                "planned_consumer_checkpoint_path": str(self.sandbox_checkpoint_path),
                "planned_consumer_metadata_path": str(self.sandbox_metadata_path),
                "source_package_checkpoint_path": str(self.package_checkpoint_path),
                "source_package_metadata_path": str(self.package_metadata_path),
                "package_checkpoint_sha256": self.checkpoint_sha256,
                "package_checkpoint_size_bytes": self.checkpoint_size,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.stage12_root / "checkpoint-publication-sandbox-install-dry-run-preflight-summary.json",
            {
                "status": "passed",
                "reason_codes": [],
                "preflight_verdict": "eligible_for_checkpoint_publication_sandbox_install_dry_run",
                "checkpoint_publication_sandbox_install_dry_run_preflight_passed": True,
                "checkpoint_publication_sandbox_install_dry_run_approved": True,
                "sandbox_preflight_manifest": str(self.stage12_manifest_path),
                "selected_seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "package_checkpoint_sha256": self.checkpoint_sha256,
                "package_checkpoint_size_bytes": self.checkpoint_size,
                "next_required_change": "checkpoint_publication_sandbox_install_dry_run",
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self.install_manifest_path = self.stage13_root / "checkpoint-publication-sandbox-install-manifest.json"
        self.copy_audit_path = self.stage13_root / "checkpoint-publication-sandbox-copy-audit.json"
        self.load_audit_path = self.stage13_root / "checkpoint-publication-sandbox-consumer-load-audit.json"
        self._write_json(
            self.install_manifest_path,
            {
                "schema_version": "checkpoint-publication-sandbox-install-manifest/v1",
                "install_dry_run_verdict": "installed_in_sandbox_for_checkpoint_publication_sandbox_install_dry_run_verification",
                "sandbox_install_manifest_passed": True,
                "stage12_preflight_manifest": str(self.stage12_manifest_path),
                "sandbox_root": str(self.sandbox_root),
                "source_package_checkpoint_path": str(self.package_checkpoint_path),
                "source_package_metadata_path": str(self.package_metadata_path),
                "sandbox_consumer_checkpoint_path": str(self.sandbox_checkpoint_path),
                "sandbox_consumer_metadata_path": str(self.sandbox_metadata_path),
                "source_package_checkpoint_sha256": self.checkpoint_sha256,
                "sandbox_consumer_checkpoint_sha256": self.checkpoint_sha256,
                "source_package_checkpoint_size_bytes": self.checkpoint_size,
                "sandbox_consumer_checkpoint_size_bytes": self.checkpoint_size,
                "next_required_change": "checkpoint_publication_sandbox_install_dry_run_verification",
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.copy_audit_path,
            {
                "sandbox_copy_audit_passed": True,
                "reason_codes": [],
                "source_package_checkpoint_sha256": self.checkpoint_sha256,
                "sandbox_consumer_checkpoint_sha256": self.checkpoint_sha256,
                "source_package_checkpoint_size_bytes": self.checkpoint_size,
                "sandbox_consumer_checkpoint_size_bytes": self.checkpoint_size,
                "copied_checkpoint": True,
                "copied_metadata": True,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.load_audit_path,
            {
                "sandbox_consumer_load_audit_passed": loadable,
                "reason_codes": [] if loadable else ["sandbox_consumer_load_failed"],
                "sandbox_consumer_checkpoint_sha256": self.checkpoint_sha256,
                "sandbox_consumer_checkpoint_size_bytes": self.checkpoint_size,
                "tensor_count": 1 if loadable else 0,
                "non_finite_tensor_count": 0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        for filename, payload in {
            "checkpoint-publication-sandbox-default-policy-boundary-audit.json": {"default_policy_boundary_audit_passed": True, "reason_codes": []},
            "checkpoint-publication-sandbox-path-boundary-audit.json": {"sandbox_path_boundary_audit_passed": True, "reason_codes": []},
            "checkpoint-publication-sandbox-lineage-audit.json": {"lineage_audit_passed": True, "reason_codes": []},
            "checkpoint-publication-sandbox-release-boundary-audit.json": {"release_boundary_audit_passed": True, "reason_codes": [], "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False},
            "checkpoint-publication-sandbox-rollback-audit.json": {"rollback_audit_passed": True, "reason_codes": []},
        }.items():
            self._write_json(self.stage13_root / filename, payload)
        self._write_json(
            self.stage13_root / "checkpoint-publication-sandbox-install-dry-run-summary.json",
            {
                "status": "passed",
                "reason_codes": [],
                "install_dry_run_verdict": "installed_in_sandbox_for_checkpoint_publication_sandbox_install_dry_run_verification",
                "checkpoint_publication_sandbox_install_dry_run_passed": True,
                "checkpoint_publication_sandbox_install_dry_run_verification_approved": True,
                "selected_seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "source_package_checkpoint_sha256": self.checkpoint_sha256,
                "sandbox_consumer_checkpoint_sha256": self.checkpoint_sha256,
                "source_package_checkpoint_size_bytes": self.checkpoint_size,
                "sandbox_consumer_checkpoint_size_bytes": self.checkpoint_size,
                "sandbox_install_manifest": str(self.install_manifest_path),
                "sandbox_copy_audit": str(self.copy_audit_path),
                "sandbox_consumer_load_audit": str(self.load_audit_path),
                "default_policy_boundary_audit": str(self.stage13_root / "checkpoint-publication-sandbox-default-policy-boundary-audit.json"),
                "sandbox_path_boundary_audit": str(self.stage13_root / "checkpoint-publication-sandbox-path-boundary-audit.json"),
                "sandbox_lineage_audit": str(self.stage13_root / "checkpoint-publication-sandbox-lineage-audit.json"),
                "sandbox_release_boundary_audit": str(self.stage13_root / "checkpoint-publication-sandbox-release-boundary-audit.json"),
                "rollback_audit": str(self.stage13_root / "checkpoint-publication-sandbox-rollback-audit.json"),
                "next_required_change": "checkpoint_publication_sandbox_install_dry_run_verification",
                "checkpoint_publication_approved": False,
                "default_policy_replacement_approved": False,
                "real_executor_connection_approved": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.stage11_root / "checkpoint-publication-package-consumer-manifest.json",
            {
                "authoritative_package_checkpoint_path": str(self.package_checkpoint_path),
                "package_metadata_path": str(self.package_metadata_path),
                "package_checkpoint_sha256": self.checkpoint_sha256,
                "package_checkpoint_size_bytes": self.checkpoint_size,
                "package_load_verification_audit_passed": True,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        for filename, payload in {
            "checkpoint-publication-package-integrity-audit.json": {"package_integrity_audit_passed": True, "reason_codes": []},
            "checkpoint-publication-package-load-verification-audit.json": {"package_load_verification_audit_passed": True, "reason_codes": []},
            "checkpoint-publication-package-release-boundary-audit.json": {"release_boundary_audit_passed": True, "reason_codes": []},
            "checkpoint-publication-package-rollback-verification-audit.json": {"rollback_verification_audit_passed": True, "reason_codes": []},
        }.items():
            self._write_json(self.stage11_root / filename, payload)
        self._write_json(self.stage9_root / "checkpoint-publication-authorization-preflight-summary.json", {"status": "passed", "reason_codes": [], "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False})
        self._write_json(self.stage8_root / "scoped-claim-publication-evidence-freeze-summary.json", {"status": "passed", "reason_codes": [], "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False})
        self._write_json(self.stage7_root / "formal-performance-claim-release-decision-summary.json", {"status": "passed", "reason_codes": [], "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False})

    def _write_metadata(self, path: Path) -> None:
        self._write_json(
            path,
            {
                "schema_version": "controlled-hybrid-policy-candidate-checkpoint-metadata/v1",
                "experimental": True,
                "seed": 0,
                "selected_budget": "epochs1_lr3e-6",
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

    def _patch_stage13_summary(self, updates: dict) -> None:
        path = self.stage13_root / "checkpoint-publication-sandbox-install-dry-run-summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(updates)
        self._write_json(path, payload)

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
