import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class CheckpointPublicationPackagePreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts = str(self.repo_root / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="checkpoint-publication-package-"))
        self.stage9_root = self.temp_dir / "stage9"
        self.stage8_root = self.temp_dir / "stage8"
        self.stage7_root = self.temp_dir / "stage7"
        self.selected_root = self.temp_dir / "selected"
        self.output_root = self.temp_dir / "output"
        for path in (self.stage9_root, self.stage8_root, self.stage7_root, self.selected_root):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_prepares_isolated_checkpoint_package_without_publication(self) -> None:
        from scripts.run_checkpoint_publication_package_preparation import (
            run_checkpoint_publication_package_preparation,
        )

        self._write_inputs()

        summary = run_checkpoint_publication_package_preparation(
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["schema_version"], "checkpoint-publication-package-preparation-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(
            summary["package_preparation_verdict"],
            "prepared_for_checkpoint_publication_package_verification",
        )
        self.assertTrue(summary["checkpoint_publication_package_prepared"])
        self.assertTrue(summary["package_manifest_audit_passed"])
        self.assertTrue(summary["package_hash_audit_passed"])
        self.assertTrue(summary["package_metadata_audit_passed"])
        self.assertTrue(summary["lineage_audit_passed"])
        self.assertTrue(summary["rollback_audit_passed"])
        self.assertTrue(summary["release_boundary_audit_passed"])
        self.assertEqual(summary["selected_seed"], 0)
        self.assertEqual(summary["selected_budget"], "epochs1_lr3e-6")
        self.assertEqual(summary["source_checkpoint_sha256"], self.checkpoint_sha256)
        self.assertEqual(summary["package_checkpoint_sha256"], self.checkpoint_sha256)
        self.assertEqual(summary["source_checkpoint_size_bytes"], self.checkpoint_size)
        self.assertEqual(summary["package_checkpoint_size_bytes"], self.checkpoint_size)
        self.assertEqual(summary["next_required_change"], "checkpoint_publication_package_verification")
        self.assertFalse(summary["checkpoint_publication_approved"])
        self.assertFalse(summary["default_policy_replacement_approved"])
        self.assertFalse(summary["real_executor_connection_approved"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])

        package_checkpoint = Path(summary["package_checkpoint_path"])
        package_metadata = Path(summary["package_metadata_path"])
        self.assertTrue(package_checkpoint.is_file())
        self.assertTrue(package_metadata.is_file())
        self.assertEqual(package_checkpoint.read_bytes(), self.checkpoint_bytes)
        self.assertNotEqual(str(package_checkpoint), str(self.checkpoint_path))

        for field in (
            "package_manifest",
            "package_hash_audit",
            "package_metadata_audit",
            "package_lineage_audit",
            "package_release_boundary_audit",
            "package_rollback_audit",
            "rejection_report",
            "report",
        ):
            self.assertTrue(Path(summary[field]).is_file(), field)

    def test_stage9_failure_blocks_package_preparation(self) -> None:
        from scripts.run_checkpoint_publication_package_preparation import (
            run_checkpoint_publication_package_preparation,
        )

        self._write_inputs()
        self._patch_stage9_summary({"status": "failed", "reason_codes": ["checkpoint_hash_mismatch"]})

        summary = run_checkpoint_publication_package_preparation(
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("stage9_not_passed", summary["reason_codes"])

    def test_wrong_stage9_next_gate_blocks_package_preparation(self) -> None:
        from scripts.run_checkpoint_publication_package_preparation import (
            run_checkpoint_publication_package_preparation,
        )

        self._write_inputs()
        self._patch_stage9_summary({"next_required_change": "default_policy_install_dry_run"})

        summary = run_checkpoint_publication_package_preparation(
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("stage9_not_authorized_for_package_preparation", summary["reason_codes"])

    def test_missing_source_checkpoint_blocks_package_preparation(self) -> None:
        from scripts.run_checkpoint_publication_package_preparation import (
            run_checkpoint_publication_package_preparation,
        )

        self._write_inputs()
        self.checkpoint_path.unlink()

        summary = run_checkpoint_publication_package_preparation(
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("source_checkpoint_missing", summary["reason_codes"])

    def test_invalid_source_metadata_blocks_package_preparation(self) -> None:
        from scripts.run_checkpoint_publication_package_preparation import (
            run_checkpoint_publication_package_preparation,
        )

        self._write_inputs()
        self._write_json(
            self.metadata_path,
            {
                "schema_version": "controlled-hybrid-policy-candidate-checkpoint-metadata/v1",
                "experimental": False,
                "checkpoint_path": str(self.checkpoint_path),
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
            },
        )

        summary = run_checkpoint_publication_package_preparation(
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("source_metadata_invalid", summary["reason_codes"])

    def test_source_identity_mismatch_blocks_package_preparation(self) -> None:
        from scripts.run_checkpoint_publication_package_preparation import (
            run_checkpoint_publication_package_preparation,
        )

        self._write_inputs()
        self.checkpoint_path.write_bytes(b"mutated-source-checkpoint")

        summary = run_checkpoint_publication_package_preparation(
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("source_checkpoint_identity_mismatch", summary["reason_codes"])

    def test_package_checkpoint_hash_mismatch_blocks_package_preparation(self) -> None:
        from scripts.run_checkpoint_publication_package_preparation import (
            run_checkpoint_publication_package_preparation,
        )

        self._write_inputs()

        def mutate_package(path: Path) -> None:
            path.write_bytes(b"mutated-package-checkpoint")

        summary = run_checkpoint_publication_package_preparation(
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            package_checkpoint_mutator=mutate_package,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("package_checkpoint_hash_mismatch", summary["reason_codes"])

    def test_package_checkpoint_size_mismatch_blocks_package_preparation(self) -> None:
        from scripts.run_checkpoint_publication_package_preparation import (
            run_checkpoint_publication_package_preparation,
        )

        self._write_inputs()

        def mutate_package(path: Path) -> None:
            path.write_bytes(self.checkpoint_bytes + b"-extra")

        summary = run_checkpoint_publication_package_preparation(
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            package_checkpoint_mutator=mutate_package,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("package_checkpoint_size_mismatch", summary["reason_codes"])

    def test_package_metadata_mismatch_blocks_package_preparation(self) -> None:
        from scripts.run_checkpoint_publication_package_preparation import (
            run_checkpoint_publication_package_preparation,
        )

        self._write_inputs()

        def mutate_metadata(path: Path) -> None:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["experimental"] = False
            self._write_json(path, payload)

        summary = run_checkpoint_publication_package_preparation(
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            package_metadata_mutator=mutate_metadata,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("package_metadata_mismatch", summary["reason_codes"])

    def test_manifest_incomplete_blocks_package_preparation(self) -> None:
        from scripts.run_checkpoint_publication_package_preparation import (
            run_checkpoint_publication_package_preparation,
        )

        self._write_inputs()

        summary = run_checkpoint_publication_package_preparation(
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            omit_manifest_fields_for_test=["stage9_summary"],
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("package_manifest_incomplete", summary["reason_codes"])

    def test_unexpected_publication_path_blocks_package_preparation(self) -> None:
        from scripts.run_checkpoint_publication_package_preparation import (
            run_checkpoint_publication_package_preparation,
        )

        self._write_inputs()

        summary = run_checkpoint_publication_package_preparation(
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.temp_dir / "default_policy_live_executor",
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("unexpected_publication_path", summary["reason_codes"])

    def test_rollback_boundary_failure_blocks_package_preparation(self) -> None:
        from scripts.run_checkpoint_publication_package_preparation import (
            run_checkpoint_publication_package_preparation,
        )

        self._write_inputs()

        summary = run_checkpoint_publication_package_preparation(
            stage9_root=self.stage9_root,
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            assume_package_deletable=False,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("rollback_boundary_invalid", summary["reason_codes"])

    def test_release_boundary_violation_blocks_package_preparation(self) -> None:
        from scripts.run_checkpoint_publication_package_preparation import (
            run_checkpoint_publication_package_preparation,
        )

        self._write_inputs()
        self._patch_stage9_summary({"checkpoint_publication_approved": True})

        summary = run_checkpoint_publication_package_preparation(
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

    def test_docs_missing_blocks_package_preparation(self) -> None:
        from scripts.run_checkpoint_publication_package_preparation import (
            run_checkpoint_publication_package_preparation,
        )

        self._write_inputs()
        fake_repo = self.temp_dir / "docless_repo"
        fake_repo.mkdir()

        summary = run_checkpoint_publication_package_preparation(
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
        shell_path = self.repo_root / "scripts" / "run_checkpoint_publication_package_preparation.sh"
        spec_path = (
            self.repo_root
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-06-15-checkpoint-publication-package-preparation.md"
        )
        self.assertTrue(shell_path.is_file())
        self.assertTrue(spec_path.is_file())
        spec_text = spec_path.read_text(encoding="utf-8")
        self.assertIn("checkpoint-publication-package-preparation-summary.json", spec_text)
        self.assertIn("README.md", spec_text)
        self.assertIn("docs/算法设计与系统架构报告.md", spec_text)
        self.assertIn("checkpoint_publication_package_verification", spec_text)
        self.assertIn("不发布 checkpoint", spec_text)

    def _write_inputs(self) -> None:
        self.checkpoint_path = self.selected_root / "experimental-hybrid-policy-candidate.pt"
        self.checkpoint_bytes = b"selected-experimental-checkpoint-v1"
        self.checkpoint_path.write_bytes(self.checkpoint_bytes)
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint_bytes).hexdigest()
        self.checkpoint_size = len(self.checkpoint_bytes)
        self.metadata_path = self.selected_root / "experimental-hybrid-policy-candidate-metadata.json"
        self._write_json(
            self.metadata_path,
            {
                "schema_version": "controlled-hybrid-policy-candidate-checkpoint-metadata/v1",
                "experimental": True,
                "selected_seed": 0,
                "seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "checkpoint_path": str(self.checkpoint_path),
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "performance_claimed": False,
                "formal_training_ready_claimed": False,
            },
        )
        self._write_json(
            self.stage9_root / "checkpoint-publication-authorization-preflight-summary.json",
            {
                "schema_version": "checkpoint-publication-authorization-preflight-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "authorization_verdict": "eligible_for_checkpoint_publication_package_preparation",
                "checkpoint_publication_authorization_preflight_passed": True,
                "checkpoint_publication_package_preparation_approved": True,
                "next_required_change": "checkpoint_publication_package_preparation",
                "selected_seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "checkpoint_path": str(self.checkpoint_path),
                "checkpoint_metadata_path": str(self.metadata_path),
                "checkpoint_sha256": self.checkpoint_sha256,
                "checkpoint_size_bytes": self.checkpoint_size,
                "candidate_manifest": str(self.stage9_root / "checkpoint-publication-candidate-manifest.json"),
                "identity_audit": str(self.stage9_root / "checkpoint-publication-identity-audit.json"),
                "metadata_audit": str(self.stage9_root / "checkpoint-publication-metadata-audit.json"),
                "load_evidence_audit": str(self.stage9_root / "checkpoint-publication-load-evidence-audit.json"),
                "lineage_audit": str(self.stage9_root / "checkpoint-publication-lineage-audit.json"),
                "release_boundary_audit": str(self.stage9_root / "checkpoint-publication-release-boundary-audit.json"),
                "checkpoint_publication_approved": False,
                "default_policy_replacement_approved": False,
                "real_executor_connection_approved": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.stage9_root / "checkpoint-publication-candidate-manifest.json",
            {
                "authorization_verdict": "eligible_for_checkpoint_publication_package_preparation",
                "selected_seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "checkpoint_path": str(self.checkpoint_path),
                "checkpoint_metadata_path": str(self.metadata_path),
                "checkpoint_sha256": self.checkpoint_sha256,
                "checkpoint_size_bytes": self.checkpoint_size,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(self.stage9_root / "checkpoint-publication-identity-audit.json", {"checkpoint_identity_audit_passed": True})
        self._write_json(self.stage9_root / "checkpoint-publication-metadata-audit.json", {"checkpoint_metadata_audit_passed": True})
        self._write_json(self.stage9_root / "checkpoint-publication-load-evidence-audit.json", {"checkpoint_load_evidence_audit_passed": True})
        self._write_json(self.stage9_root / "checkpoint-publication-lineage-audit.json", {"lineage_audit_passed": True})
        self._write_json(
            self.stage9_root / "checkpoint-publication-release-boundary-audit.json",
            {"release_boundary_audit_passed": True, "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False},
        )
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
            {"status": "passed", "reason_codes": [], "checkpoint_path": str(self.checkpoint_path), "checkpoint_metadata_path": str(self.metadata_path), "checkpoint_sha256": self.checkpoint_sha256, "checkpoint_size_bytes": self.checkpoint_size, "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False},
        )

    def _patch_stage9_summary(self, updates: dict) -> None:
        path = self.stage9_root / "checkpoint-publication-authorization-preflight-summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(updates)
        self._write_json(path, payload)

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
