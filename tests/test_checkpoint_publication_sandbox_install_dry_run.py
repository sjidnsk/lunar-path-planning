import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class CheckpointPublicationSandboxInstallDryRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts = str(self.repo_root / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="checkpoint-sandbox-install-"))
        self.stage12_root = self.temp_dir / "stage12"
        self.stage11_root = self.temp_dir / "stage11"
        self.stage10_root = self.temp_dir / "stage10"
        self.stage9_root = self.temp_dir / "stage9"
        self.stage8_root = self.temp_dir / "stage8"
        self.stage7_root = self.temp_dir / "stage7"
        self.output_root = self.temp_dir / "output"
        for path in (
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

    def test_installs_package_into_sandbox_without_publication(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run import (
            run_checkpoint_publication_sandbox_install_dry_run,
        )

        self._write_inputs()

        summary = run_checkpoint_publication_sandbox_install_dry_run(
            stage12_root=self.stage12_root,
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["schema_version"], "checkpoint-publication-sandbox-install-dry-run-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(
            summary["install_dry_run_verdict"],
            "installed_in_sandbox_for_checkpoint_publication_sandbox_install_dry_run_verification",
        )
        self.assertTrue(summary["checkpoint_publication_sandbox_install_dry_run_passed"])
        self.assertTrue(summary["checkpoint_publication_sandbox_install_dry_run_verification_approved"])
        self.assertTrue(summary["sandbox_install_manifest_passed"])
        self.assertTrue(summary["sandbox_copy_audit_passed"])
        self.assertTrue(summary["sandbox_consumer_load_audit_passed"])
        self.assertTrue(summary["default_policy_boundary_audit_passed"])
        self.assertTrue(summary["sandbox_path_boundary_audit_passed"])
        self.assertTrue(summary["lineage_audit_passed"])
        self.assertTrue(summary["rollback_audit_passed"])
        self.assertTrue(summary["release_boundary_audit_passed"])
        self.assertEqual(summary["selected_seed"], 0)
        self.assertEqual(summary["selected_budget"], "epochs1_lr3e-6")
        self.assertEqual(summary["source_package_checkpoint_sha256"], self.package_sha256)
        self.assertEqual(summary["sandbox_consumer_checkpoint_sha256"], self.package_sha256)
        self.assertEqual(summary["source_package_checkpoint_size_bytes"], self.package_size)
        self.assertEqual(summary["sandbox_consumer_checkpoint_size_bytes"], self.package_size)
        self.assertEqual(summary["next_required_change"], "checkpoint_publication_sandbox_install_dry_run_verification")
        self.assertFalse(summary["checkpoint_publication_approved"])
        self.assertFalse(summary["default_policy_replacement_approved"])
        self.assertFalse(summary["real_executor_connection_approved"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])

        self.assertTrue(self.planned_checkpoint_path.is_file())
        self.assertTrue(self.planned_metadata_path.is_file())
        self.assertEqual(hashlib.sha256(self.planned_checkpoint_path.read_bytes()).hexdigest(), self.package_sha256)

        for field in (
            "sandbox_install_manifest",
            "sandbox_copy_audit",
            "sandbox_consumer_load_audit",
            "default_policy_boundary_audit",
            "sandbox_path_boundary_audit",
            "sandbox_lineage_audit",
            "sandbox_release_boundary_audit",
            "rollback_audit",
            "rejection_report",
            "report",
        ):
            self.assertTrue(Path(summary[field]).is_file(), field)

    def test_stage12_failure_blocks_install(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run import (
            run_checkpoint_publication_sandbox_install_dry_run,
        )

        self._write_inputs()
        self._patch_stage12_summary({"status": "failed", "reason_codes": ["package_identity_mismatch"]})

        summary = run_checkpoint_publication_sandbox_install_dry_run(
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
        self.assertIn("stage12_not_passed", summary["reason_codes"])

    def test_wrong_stage12_next_gate_blocks_install(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run import (
            run_checkpoint_publication_sandbox_install_dry_run,
        )

        self._write_inputs()
        self._patch_stage12_summary({"next_required_change": "checkpoint_publication_sandbox_install_dry_run_verification"})

        summary = run_checkpoint_publication_sandbox_install_dry_run(
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
        self.assertIn("stage12_not_authorized_for_sandbox_install_dry_run", summary["reason_codes"])

    def test_missing_preflight_manifest_blocks_install(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run import (
            run_checkpoint_publication_sandbox_install_dry_run,
        )

        self._write_inputs()
        self.preflight_manifest_path.unlink()

        summary = run_checkpoint_publication_sandbox_install_dry_run(
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
        self.assertIn("preflight_manifest_missing", summary["reason_codes"])

    def test_missing_source_checkpoint_blocks_install(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run import (
            run_checkpoint_publication_sandbox_install_dry_run,
        )

        self._write_inputs()
        self.package_checkpoint_path.unlink()

        summary = run_checkpoint_publication_sandbox_install_dry_run(
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

    def test_missing_source_metadata_blocks_install(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run import (
            run_checkpoint_publication_sandbox_install_dry_run,
        )

        self._write_inputs()
        self.package_metadata_path.unlink()

        summary = run_checkpoint_publication_sandbox_install_dry_run(
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

    def test_source_identity_mismatch_blocks_install(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run import (
            run_checkpoint_publication_sandbox_install_dry_run,
        )

        self._write_inputs()
        self.package_checkpoint_path.write_bytes(b"mutated-source")

        summary = run_checkpoint_publication_sandbox_install_dry_run(
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

    def test_unsafe_sandbox_path_blocks_install(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run import (
            run_checkpoint_publication_sandbox_install_dry_run,
        )

        self._write_inputs()
        manifest = json.loads(self.preflight_manifest_path.read_text(encoding="utf-8"))
        manifest["planned_consumer_checkpoint_path"] = str(self.temp_dir / "live" / "candidate.pt")
        self._write_json(self.preflight_manifest_path, manifest)

        summary = run_checkpoint_publication_sandbox_install_dry_run(
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
        self.assertIn("sandbox_path_boundary_violation", summary["reason_codes"])
        self.assertFalse((self.temp_dir / "live" / "candidate.pt").exists())

    def test_existing_mismatched_sandbox_file_blocks_install_without_overwrite(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run import (
            run_checkpoint_publication_sandbox_install_dry_run,
        )

        self._write_inputs()
        self.planned_checkpoint_path.parent.mkdir(parents=True)
        self.planned_checkpoint_path.write_bytes(b"wrong-existing-checkpoint")

        summary = run_checkpoint_publication_sandbox_install_dry_run(
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
        self.assertIn("sandbox_copy_failed", summary["reason_codes"])
        self.assertEqual(self.planned_checkpoint_path.read_bytes(), b"wrong-existing-checkpoint")

    def test_sandbox_consumer_identity_mismatch_blocks_install(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run import (
            run_checkpoint_publication_sandbox_install_dry_run,
        )

        self._write_inputs()

        summary = run_checkpoint_publication_sandbox_install_dry_run(
            stage12_root=self.stage12_root,
            stage11_root=self.stage11_root,
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            sandbox_checkpoint_mutator=lambda path: path.write_bytes(b"mutated-sandbox"),
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sandbox_consumer_identity_mismatch", summary["reason_codes"])

    def test_sandbox_load_failure_blocks_install(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run import (
            run_checkpoint_publication_sandbox_install_dry_run,
        )

        self._write_inputs(loadable=False)

        summary = run_checkpoint_publication_sandbox_install_dry_run(
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
        self.assertIn("sandbox_consumer_load_failed", summary["reason_codes"])

    def test_default_policy_boundary_violation_blocks_install(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run import (
            run_checkpoint_publication_sandbox_install_dry_run,
        )

        self._write_inputs()
        default_policy_path = self.temp_dir / "simulated-default-policy.pt"
        default_policy_path.write_bytes(b"default-policy-before")

        summary = run_checkpoint_publication_sandbox_install_dry_run(
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

    def test_lineage_failure_blocks_install(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run import (
            run_checkpoint_publication_sandbox_install_dry_run,
        )

        self._write_inputs()
        self._write_json(self.stage12_root / "checkpoint-publication-sandbox-lineage-audit.json", {"lineage_audit_passed": False})

        summary = run_checkpoint_publication_sandbox_install_dry_run(
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
        self.assertIn("lineage_incomplete", summary["reason_codes"])

    def test_rollback_failure_blocks_install(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run import (
            run_checkpoint_publication_sandbox_install_dry_run,
        )

        self._write_inputs()

        summary = run_checkpoint_publication_sandbox_install_dry_run(
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

    def test_release_boundary_violation_blocks_install(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run import (
            run_checkpoint_publication_sandbox_install_dry_run,
        )

        self._write_inputs()
        self._patch_stage12_summary({"checkpoint_publication_approved": True})

        summary = run_checkpoint_publication_sandbox_install_dry_run(
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

    def test_docs_missing_blocks_install(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_install_dry_run import (
            run_checkpoint_publication_sandbox_install_dry_run,
        )

        self._write_inputs()
        fake_repo = self.temp_dir / "docless_repo"
        fake_repo.mkdir()

        summary = run_checkpoint_publication_sandbox_install_dry_run(
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
        shell_path = self.repo_root / "scripts" / "run_checkpoint_publication_sandbox_install_dry_run.sh"
        spec_path = (
            self.repo_root
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-06-16-checkpoint-publication-sandbox-install-dry-run.md"
        )
        self.assertTrue(shell_path.is_file())
        self.assertTrue(spec_path.is_file())
        spec_text = spec_path.read_text(encoding="utf-8")
        self.assertIn("checkpoint-publication-sandbox-install-dry-run-summary.json", spec_text)
        self.assertIn("README.md", spec_text)
        self.assertIn("docs/算法设计与系统架构报告.md", spec_text)
        self.assertIn("checkpoint_publication_sandbox_install_dry_run_verification", spec_text)
        self.assertIn("不发布 checkpoint", spec_text)

    def _write_inputs(self, *, loadable: bool = True) -> None:
        import torch

        self.package_root = self.stage10_root / "checkpoint-publication-package"
        self.package_root.mkdir(parents=True)
        self.package_checkpoint_path = self.package_root / "experimental-hybrid-policy-candidate.pt"
        if loadable:
            torch.save(
                {
                    "schema_version": "experimental-hybrid-policy-candidate-checkpoint/v1",
                    "experimental": True,
                    "model_state_dict": {"layer.weight": torch.ones(2, 2)},
                },
                self.package_checkpoint_path,
            )
        else:
            torch.save(
                {
                    "schema_version": "experimental-hybrid-policy-candidate-checkpoint/v1",
                    "experimental": True,
                    "not_a_state_dict": {"layer.weight": torch.ones(2, 2)},
                },
                self.package_checkpoint_path,
            )
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
            },
        )
        self.sandbox_root = self.stage12_root / "checkpoint-publication-sandbox-install-dry-run-sandbox"
        self.planned_checkpoint_path = self.sandbox_root / "experimental-hybrid-policy-candidate.pt"
        self.planned_metadata_path = self.sandbox_root / "experimental-hybrid-policy-candidate-metadata.json"
        self.preflight_manifest_path = self.stage12_root / "checkpoint-publication-sandbox-preflight-manifest.json"
        self._write_json(
            self.preflight_manifest_path,
            {
                "schema_version": "checkpoint-publication-sandbox-preflight-manifest/v1",
                "preflight_verdict": "eligible_for_checkpoint_publication_sandbox_install_dry_run",
                "sandbox_preflight_manifest_passed": True,
                "sandbox_root": str(self.sandbox_root),
                "planned_consumer_checkpoint_path": str(self.planned_checkpoint_path),
                "planned_consumer_metadata_path": str(self.planned_metadata_path),
                "source_package_checkpoint_path": str(self.package_checkpoint_path),
                "source_package_metadata_path": str(self.package_metadata_path),
                "package_checkpoint_sha256": self.package_sha256,
                "package_checkpoint_size_bytes": self.package_size,
                "stage12_copies_checkpoint_to_sandbox": False,
                "stage12_installs_policy": False,
                "stage12_runs_rollout": False,
                "next_required_change": "checkpoint_publication_sandbox_install_dry_run",
                "checkpoint_publication_approved": False,
                "default_policy_replacement_approved": False,
                "real_executor_connection_approved": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self.package_consumer_audit_path = self.stage12_root / "checkpoint-publication-sandbox-package-consumer-audit.json"
        self._write_json(
            self.package_consumer_audit_path,
            {
                "package_consumer_audit_passed": True,
                "reason_codes": [],
                "authoritative_package_checkpoint_path": str(self.package_checkpoint_path),
                "package_checkpoint_exists": True,
                "package_checkpoint_sha256": self.package_sha256,
                "package_checkpoint_size_bytes": self.package_size,
                "package_load_verified": True,
            },
        )
        for filename, payload in {
            "checkpoint-publication-sandbox-default-policy-boundary-audit.json": {"default_policy_boundary_audit_passed": True, "reason_codes": []},
            "checkpoint-publication-sandbox-path-boundary-audit.json": {"sandbox_path_boundary_audit_passed": True, "reason_codes": []},
            "checkpoint-publication-sandbox-lineage-audit.json": {"lineage_audit_passed": True, "reason_codes": []},
            "checkpoint-publication-sandbox-release-boundary-audit.json": {"release_boundary_audit_passed": True, "reason_codes": [], "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False},
            "checkpoint-publication-sandbox-rollback-preflight-audit.json": {"rollback_preflight_audit_passed": True, "reason_codes": []},
        }.items():
            self._write_json(self.stage12_root / filename, payload)
        self._write_json(
            self.stage12_root / "checkpoint-publication-sandbox-install-dry-run-preflight-summary.json",
            {
                "schema_version": "checkpoint-publication-sandbox-install-dry-run-preflight-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "preflight_verdict": "eligible_for_checkpoint_publication_sandbox_install_dry_run",
                "checkpoint_publication_sandbox_install_dry_run_preflight_passed": True,
                "checkpoint_publication_sandbox_install_dry_run_approved": True,
                "selected_seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "package_checkpoint_sha256": self.package_sha256,
                "package_checkpoint_size_bytes": self.package_size,
                "sandbox_preflight_manifest": str(self.preflight_manifest_path),
                "package_consumer_audit": str(self.package_consumer_audit_path),
                "default_policy_boundary_audit": str(self.stage12_root / "checkpoint-publication-sandbox-default-policy-boundary-audit.json"),
                "sandbox_path_boundary_audit": str(self.stage12_root / "checkpoint-publication-sandbox-path-boundary-audit.json"),
                "sandbox_lineage_audit": str(self.stage12_root / "checkpoint-publication-sandbox-lineage-audit.json"),
                "sandbox_release_boundary_audit": str(self.stage12_root / "checkpoint-publication-sandbox-release-boundary-audit.json"),
                "rollback_preflight_audit": str(self.stage12_root / "checkpoint-publication-sandbox-rollback-preflight-audit.json"),
                "next_required_change": "checkpoint_publication_sandbox_install_dry_run",
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
                "verification_verdict": "verified_for_checkpoint_publication_sandbox_install_dry_run_preflight",
                "stage10_manifest": str(self.stage10_manifest_path),
                "selected_seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "authoritative_package_checkpoint_path": str(self.package_checkpoint_path),
                "package_metadata_path": str(self.package_metadata_path),
                "package_checkpoint_sha256": self.package_sha256,
                "package_checkpoint_size_bytes": self.package_size,
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

    def _patch_stage12_summary(self, updates: dict) -> None:
        path = self.stage12_root / "checkpoint-publication-sandbox-install-dry-run-preflight-summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(updates)
        self._write_json(path, payload)

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
