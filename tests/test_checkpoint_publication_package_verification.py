import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class CheckpointPublicationPackageVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts = str(self.repo_root / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="checkpoint-package-verification-"))
        self.stage10_root = self.temp_dir / "stage10"
        self.stage9_root = self.temp_dir / "stage9"
        self.stage8_root = self.temp_dir / "stage8"
        self.stage7_root = self.temp_dir / "stage7"
        self.selected_root = self.temp_dir / "selected"
        self.output_root = self.temp_dir / "output"
        for path in (
            self.stage10_root,
            self.stage9_root,
            self.stage8_root,
            self.stage7_root,
            self.selected_root,
        ):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_verifies_prepared_package_without_publication(self) -> None:
        from scripts.run_checkpoint_publication_package_verification import (
            run_checkpoint_publication_package_verification,
        )

        self._write_inputs()

        summary = run_checkpoint_publication_package_verification(
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["schema_version"], "checkpoint-publication-package-verification-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(
            summary["verification_verdict"],
            "verified_for_checkpoint_publication_sandbox_install_dry_run_preflight",
        )
        self.assertTrue(summary["checkpoint_publication_package_verification_passed"])
        self.assertTrue(summary["checkpoint_publication_sandbox_install_dry_run_preflight_approved"])
        self.assertTrue(summary["package_integrity_audit_passed"])
        self.assertTrue(summary["package_manifest_consistency_audit_passed"])
        self.assertTrue(summary["package_metadata_verification_audit_passed"])
        self.assertTrue(summary["package_load_verification_audit_passed"])
        self.assertTrue(summary["lineage_verification_audit_passed"])
        self.assertTrue(summary["rollback_verification_audit_passed"])
        self.assertTrue(summary["release_boundary_audit_passed"])
        self.assertEqual(summary["selected_seed"], 0)
        self.assertEqual(summary["selected_budget"], "epochs1_lr3e-6")
        self.assertEqual(summary["package_checkpoint_sha256"], self.checkpoint_sha256)
        self.assertEqual(summary["package_checkpoint_size_bytes"], self.checkpoint_size)
        self.assertEqual(summary["next_required_change"], "checkpoint_publication_sandbox_install_dry_run_preflight")
        self.assertFalse(summary["checkpoint_publication_approved"])
        self.assertFalse(summary["default_policy_replacement_approved"])
        self.assertFalse(summary["real_executor_connection_approved"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])

        for field in (
            "consumer_manifest",
            "package_integrity_audit",
            "package_manifest_consistency_audit",
            "package_metadata_verification_audit",
            "package_load_verification_audit",
            "package_lineage_verification_audit",
            "package_release_boundary_audit",
            "package_rollback_verification_audit",
            "rejection_report",
            "report",
        ):
            self.assertTrue(Path(summary[field]).is_file(), field)

    def test_stage10_failure_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_package_verification import (
            run_checkpoint_publication_package_verification,
        )

        self._write_inputs()
        self._patch_stage10_summary({"status": "failed", "reason_codes": ["package_hash_mismatch"]})

        summary = run_checkpoint_publication_package_verification(
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("stage10_not_passed", summary["reason_codes"])

    def test_wrong_stage10_next_gate_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_package_verification import (
            run_checkpoint_publication_package_verification,
        )

        self._write_inputs()
        self._patch_stage10_summary({"next_required_change": "checkpoint_publication"})

        summary = run_checkpoint_publication_package_verification(
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("stage10_not_prepared_for_package_verification", summary["reason_codes"])

    def test_missing_package_manifest_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_package_verification import (
            run_checkpoint_publication_package_verification,
        )

        self._write_inputs()
        self.manifest_path.unlink()

        summary = run_checkpoint_publication_package_verification(
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("package_manifest_missing", summary["reason_codes"])

    def test_missing_package_checkpoint_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_package_verification import (
            run_checkpoint_publication_package_verification,
        )

        self._write_inputs()
        self.package_checkpoint_path.unlink()

        summary = run_checkpoint_publication_package_verification(
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("package_checkpoint_missing", summary["reason_codes"])

    def test_missing_package_metadata_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_package_verification import (
            run_checkpoint_publication_package_verification,
        )

        self._write_inputs()
        self.package_metadata_path.unlink()

        summary = run_checkpoint_publication_package_verification(
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("package_metadata_missing", summary["reason_codes"])

    def test_package_hash_mismatch_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_package_verification import (
            run_checkpoint_publication_package_verification,
        )

        self._write_inputs()
        self.package_checkpoint_path.write_bytes(b"mutated-package")

        summary = run_checkpoint_publication_package_verification(
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("package_hash_mismatch", summary["reason_codes"])

    def test_package_size_mismatch_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_package_verification import (
            run_checkpoint_publication_package_verification,
        )

        self._write_inputs()
        self._patch_stage10_summary({"package_checkpoint_size_bytes": self.checkpoint_size + 1})

        summary = run_checkpoint_publication_package_verification(
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("package_size_mismatch", summary["reason_codes"])

    def test_manifest_inconsistency_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_package_verification import (
            run_checkpoint_publication_package_verification,
        )

        self._write_inputs()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["selected_budget"] = "wrong-budget"
        self._write_json(self.manifest_path, manifest)

        summary = run_checkpoint_publication_package_verification(
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("package_manifest_inconsistent", summary["reason_codes"])

    def test_invalid_package_metadata_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_package_verification import (
            run_checkpoint_publication_package_verification,
        )

        self._write_inputs()
        metadata = json.loads(self.package_metadata_path.read_text(encoding="utf-8"))
        metadata["performance_claimed"] = True
        self._write_json(self.package_metadata_path, metadata)

        summary = run_checkpoint_publication_package_verification(
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("package_metadata_invalid", summary["reason_codes"])

    def test_load_failure_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_package_verification import (
            run_checkpoint_publication_package_verification,
        )

        self._write_inputs(loadable=False)

        summary = run_checkpoint_publication_package_verification(
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("package_load_verification_failed", summary["reason_codes"])

    def test_lineage_failure_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_package_verification import (
            run_checkpoint_publication_package_verification,
        )

        self._write_inputs()
        self._write_json(self.stage9_root / "checkpoint-publication-load-evidence-audit.json", {"checkpoint_load_evidence_audit_passed": False})

        summary = run_checkpoint_publication_package_verification(
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("lineage_verification_incomplete", summary["reason_codes"])

    def test_rollback_boundary_failure_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_package_verification import (
            run_checkpoint_publication_package_verification,
        )

        self._write_inputs()

        summary = run_checkpoint_publication_package_verification(
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            assume_package_unchanged=False,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("rollback_boundary_invalid", summary["reason_codes"])

    def test_release_boundary_violation_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_package_verification import (
            run_checkpoint_publication_package_verification,
        )

        self._write_inputs()
        self._patch_stage10_summary({"checkpoint_publication_approved": True})

        summary = run_checkpoint_publication_package_verification(
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("release_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["checkpoint_publication_approved"])

    def test_docs_missing_blocks_verification(self) -> None:
        from scripts.run_checkpoint_publication_package_verification import (
            run_checkpoint_publication_package_verification,
        )

        self._write_inputs()
        fake_repo = self.temp_dir / "docless_repo"
        fake_repo.mkdir()

        summary = run_checkpoint_publication_package_verification(
            stage10_root=self.stage10_root,
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=fake_repo,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("docs_not_updated", summary["reason_codes"])

    def test_shell_docs_and_spec_contract_are_declared(self) -> None:
        shell_path = self.repo_root / "scripts" / "run_checkpoint_publication_package_verification.sh"
        spec_path = (
            self.repo_root
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-06-15-checkpoint-publication-package-verification.md"
        )
        self.assertTrue(shell_path.is_file())
        self.assertTrue(spec_path.is_file())
        spec_text = spec_path.read_text(encoding="utf-8")
        self.assertIn("checkpoint-publication-package-verification-summary.json", spec_text)
        self.assertIn("README.md", spec_text)
        self.assertIn("docs/算法设计与系统架构报告.md", spec_text)
        self.assertIn("checkpoint_publication_sandbox_install_dry_run_preflight", spec_text)
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
        checkpoint_bytes = self.package_checkpoint_path.read_bytes()
        self.checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        self.checkpoint_size = len(checkpoint_bytes)
        self.package_metadata_path = self.package_root / "experimental-hybrid-policy-candidate-metadata.json"
        self._write_json(
            self.package_metadata_path,
            {
                "schema_version": "controlled-hybrid-policy-candidate-checkpoint-metadata/v1",
                "experimental": True,
                "seed": 0,
                "checkpoint_path": "source/provenance/checkpoint.pt",
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "performance_claimed": False,
                "formal_training_ready_claimed": False,
            },
        )
        self.manifest_path = self.stage10_root / "checkpoint-publication-package-manifest.json"
        self.hash_audit_path = self.stage10_root / "checkpoint-publication-package-hash-audit.json"
        self.metadata_audit_path = self.stage10_root / "checkpoint-publication-package-metadata-audit.json"
        self.lineage_audit_path = self.stage10_root / "checkpoint-publication-package-lineage-audit.json"
        self.release_audit_path = self.stage10_root / "checkpoint-publication-package-release-boundary-audit.json"
        self.rollback_audit_path = self.stage10_root / "checkpoint-publication-package-rollback-audit.json"
        self.stage9_summary_path = self.stage9_root / "checkpoint-publication-authorization-preflight-summary.json"
        self.stage9_candidate_path = self.stage9_root / "checkpoint-publication-candidate-manifest.json"
        self.stage9_load_path = self.stage9_root / "checkpoint-publication-load-evidence-audit.json"
        self._write_json(
            self.manifest_path,
            {
                "schema_version": "checkpoint-publication-package-manifest/v1",
                "package_preparation_verdict": "prepared_for_checkpoint_publication_package_verification",
                "selected_seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "stage9_summary": str(self.stage9_summary_path),
                "package_root": str(self.package_root),
                "source_checkpoint_path": str(self.selected_root / "source.pt"),
                "package_checkpoint_path": str(self.package_checkpoint_path),
                "source_metadata_path": str(self.selected_root / "source-metadata.json"),
                "package_metadata_path": str(self.package_metadata_path),
                "source_checkpoint_sha256": self.checkpoint_sha256,
                "package_checkpoint_sha256": self.checkpoint_sha256,
                "source_checkpoint_size_bytes": self.checkpoint_size,
                "package_checkpoint_size_bytes": self.checkpoint_size,
                "next_required_change": "checkpoint_publication_package_verification",
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "non_goals": [
                    "do_not_publish_checkpoint",
                    "do_not_replace_default_policy",
                    "do_not_connect_real_executor",
                    "do_not_claim_ackermann_feasible_trajectory",
                ],
            },
        )
        self._write_json(
            self.hash_audit_path,
            {
                "package_hash_audit_passed": True,
                "reason_codes": [],
                "package_checkpoint_path": str(self.package_checkpoint_path),
                "package_checkpoint_sha256": self.checkpoint_sha256,
                "package_checkpoint_size_bytes": self.checkpoint_size,
                "source_checkpoint_sha256": self.checkpoint_sha256,
                "source_checkpoint_size_bytes": self.checkpoint_size,
                "package_checkpoint_exists": True,
            },
        )
        self._write_json(
            self.metadata_audit_path,
            {
                "package_metadata_audit_passed": True,
                "reason_codes": [],
                "package_metadata_path": str(self.package_metadata_path),
                "package_metadata_experimental": True,
            },
        )
        self._write_json(
            self.lineage_audit_path,
            {
                "lineage_audit_passed": True,
                "missing_or_unreadable_inputs": [],
                "sources": [{"name": "stage9_summary", "passed": True}],
            },
        )
        self._write_json(
            self.release_audit_path,
            {
                "release_boundary_audit_passed": True,
                "violations": [],
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.rollback_audit_path,
            {
                "rollback_audit_passed": True,
                "package_deletable": True,
                "source_checkpoint_unchanged": True,
                "source_metadata_unchanged": True,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.stage10_root / "checkpoint-publication-package-preparation-summary.json",
            {
                "schema_version": "checkpoint-publication-package-preparation-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "package_preparation_verdict": "prepared_for_checkpoint_publication_package_verification",
                "checkpoint_publication_package_prepared": True,
                "selected_seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "package_checkpoint_path": str(self.package_checkpoint_path),
                "package_metadata_path": str(self.package_metadata_path),
                "package_checkpoint_sha256": self.checkpoint_sha256,
                "package_checkpoint_size_bytes": self.checkpoint_size,
                "source_checkpoint_sha256": self.checkpoint_sha256,
                "source_checkpoint_size_bytes": self.checkpoint_size,
                "package_manifest": str(self.manifest_path),
                "package_hash_audit": str(self.hash_audit_path),
                "package_metadata_audit": str(self.metadata_audit_path),
                "package_lineage_audit": str(self.lineage_audit_path),
                "package_release_boundary_audit": str(self.release_audit_path),
                "package_rollback_audit": str(self.rollback_audit_path),
                "next_required_change": "checkpoint_publication_package_verification",
                "checkpoint_publication_approved": False,
                "default_policy_replacement_approved": False,
                "real_executor_connection_approved": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.stage9_summary_path,
            {
                "schema_version": "checkpoint-publication-authorization-preflight-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "authorization_verdict": "eligible_for_checkpoint_publication_package_preparation",
                "checkpoint_publication_authorization_preflight_passed": True,
                "checkpoint_publication_package_preparation_approved": True,
                "selected_seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "checkpoint_sha256": self.checkpoint_sha256,
                "checkpoint_size_bytes": self.checkpoint_size,
                "candidate_manifest": str(self.stage9_candidate_path),
                "load_evidence_audit": str(self.stage9_load_path),
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.stage9_candidate_path,
            {
                "authorization_verdict": "eligible_for_checkpoint_publication_package_preparation",
                "selected_seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "checkpoint_sha256": self.checkpoint_sha256,
                "checkpoint_size_bytes": self.checkpoint_size,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(self.stage9_load_path, {"checkpoint_load_evidence_audit_passed": True})
        self._write_json(
            self.stage8_root / "scoped-claim-publication-evidence-freeze-summary.json",
            {"status": "passed", "reason_codes": [], "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False},
        )
        self._write_json(
            self.stage7_root / "formal-performance-claim-release-decision-summary.json",
            {"status": "passed", "reason_codes": [], "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False},
        )
        self._write_json(
            self.selected_root / "selected-formal-ppo-candidate-promotion-preflight-summary.json",
            {"status": "passed", "reason_codes": [], "checkpoint_sha256": self.checkpoint_sha256, "checkpoint_size_bytes": self.checkpoint_size, "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False},
        )

    def _patch_stage10_summary(self, updates: dict) -> None:
        path = self.stage10_root / "checkpoint-publication-package-preparation-summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(updates)
        self._write_json(path, payload)

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
