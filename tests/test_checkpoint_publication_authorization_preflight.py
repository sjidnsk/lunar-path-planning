import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class CheckpointPublicationAuthorizationPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts = str(self.repo_root / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="checkpoint-publication-auth-"))
        self.stage8_root = self.temp_dir / "stage8"
        self.stage7_root = self.temp_dir / "stage7"
        self.stage6_root = self.temp_dir / "stage6"
        self.cost_root = self.temp_dir / "cost"
        self.formal_root = self.temp_dir / "formal"
        self.replay_root = self.temp_dir / "replay"
        self.selected_root = self.temp_dir / "selected"
        self.output_root = self.temp_dir / "output"
        for path in (
            self.stage8_root,
            self.stage7_root,
            self.stage6_root,
            self.cost_root,
            self.formal_root,
            self.replay_root,
            self.selected_root,
        ):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_authorizes_checkpoint_package_preparation_without_publication(self) -> None:
        from scripts.run_checkpoint_publication_authorization_preflight import (
            run_checkpoint_publication_authorization_preflight,
        )

        self._write_inputs()

        summary = run_checkpoint_publication_authorization_preflight(
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["schema_version"], "checkpoint-publication-authorization-preflight-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["authorization_verdict"], "eligible_for_checkpoint_publication_package_preparation")
        self.assertTrue(summary["checkpoint_publication_authorization_preflight_passed"])
        self.assertTrue(summary["checkpoint_publication_package_preparation_approved"])
        self.assertTrue(summary["checkpoint_identity_audit_passed"])
        self.assertTrue(summary["checkpoint_metadata_audit_passed"])
        self.assertTrue(summary["checkpoint_load_evidence_audit_passed"])
        self.assertTrue(summary["lineage_audit_passed"])
        self.assertTrue(summary["release_boundary_audit_passed"])
        self.assertEqual(summary["selected_seed"], 0)
        self.assertEqual(summary["selected_budget"], "epochs1_lr3e-6")
        self.assertEqual(summary["checkpoint_sha256"], self.checkpoint_sha256)
        self.assertEqual(summary["next_required_change"], "checkpoint_publication_package_preparation")
        self.assertFalse(summary["checkpoint_publication_approved"])
        self.assertFalse(summary["default_policy_replacement_approved"])
        self.assertFalse(summary["real_executor_connection_approved"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])

        for field in (
            "candidate_manifest",
            "identity_audit",
            "metadata_audit",
            "load_evidence_audit",
            "lineage_audit",
            "release_boundary_audit",
            "authorization_matrix",
            "rejection_report",
            "report",
        ):
            self.assertTrue(Path(summary[field]).is_file(), field)

    def test_stage8_failure_blocks_authorization(self) -> None:
        from scripts.run_checkpoint_publication_authorization_preflight import (
            run_checkpoint_publication_authorization_preflight,
        )

        self._write_inputs()
        self._patch_stage8_summary({"status": "failed", "reason_codes": ["docs_not_updated"]})

        summary = run_checkpoint_publication_authorization_preflight(
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("stage8_not_passed", summary["reason_codes"])

    def test_overbroad_stage8_claim_scope_blocks_authorization(self) -> None:
        from scripts.run_checkpoint_publication_authorization_preflight import (
            run_checkpoint_publication_authorization_preflight,
        )

        self._write_inputs()
        self._patch_stage8_summary({"performance_claim_scope": "all_real_world_scenarios"})

        summary = run_checkpoint_publication_authorization_preflight(
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("claim_scope_overbroad", summary["reason_codes"])

    def test_missing_checkpoint_blocks_authorization(self) -> None:
        from scripts.run_checkpoint_publication_authorization_preflight import (
            run_checkpoint_publication_authorization_preflight,
        )

        self._write_inputs()
        self.checkpoint_path.unlink()

        summary = run_checkpoint_publication_authorization_preflight(
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("checkpoint_missing", summary["reason_codes"])

    def test_checkpoint_hash_mismatch_blocks_authorization(self) -> None:
        from scripts.run_checkpoint_publication_authorization_preflight import (
            run_checkpoint_publication_authorization_preflight,
        )

        self._write_inputs()
        self.checkpoint_path.write_bytes(b"mutated-checkpoint-v1")

        summary = run_checkpoint_publication_authorization_preflight(
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("checkpoint_hash_mismatch", summary["reason_codes"])

    def test_checkpoint_size_mismatch_blocks_authorization(self) -> None:
        from scripts.run_checkpoint_publication_authorization_preflight import (
            run_checkpoint_publication_authorization_preflight,
        )

        self._write_inputs()
        self._patch_selected_summary({"checkpoint_size_bytes": self.checkpoint_size + 1})

        summary = run_checkpoint_publication_authorization_preflight(
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("checkpoint_size_mismatch", summary["reason_codes"])

    def test_invalid_metadata_blocks_authorization(self) -> None:
        from scripts.run_checkpoint_publication_authorization_preflight import (
            run_checkpoint_publication_authorization_preflight,
        )

        self._write_inputs()
        self._write_json(
            self.metadata_path,
            {
                "schema_version": "controlled-hybrid-policy-candidate-checkpoint-metadata/v1",
                "experimental": False,
                "selected_seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "checkpoint_path": str(self.checkpoint_path),
            },
        )

        summary = run_checkpoint_publication_authorization_preflight(
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("checkpoint_metadata_invalid", summary["reason_codes"])

    def test_failed_load_evidence_blocks_authorization(self) -> None:
        from scripts.run_checkpoint_publication_authorization_preflight import (
            run_checkpoint_publication_authorization_preflight,
        )

        self._write_inputs()
        self._patch_selected_summary({"checkpoint_load_passed": False, "non_finite_logits_count": 1})

        summary = run_checkpoint_publication_authorization_preflight(
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("checkpoint_load_evidence_not_passed", summary["reason_codes"])

    def test_incomplete_lineage_blocks_authorization(self) -> None:
        from scripts.run_checkpoint_publication_authorization_preflight import (
            run_checkpoint_publication_authorization_preflight,
        )

        self._write_inputs()
        self._write_json(
            self.formal_root / "formal-ppo-training-run-summary.json",
            {"status": "failed", "reason_codes": ["controlled_regression"]},
        )

        summary = run_checkpoint_publication_authorization_preflight(
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("lineage_incomplete", summary["reason_codes"])

    def test_release_boundary_violation_blocks_authorization(self) -> None:
        from scripts.run_checkpoint_publication_authorization_preflight import (
            run_checkpoint_publication_authorization_preflight,
        )

        self._write_inputs()
        self._patch_stage8_summary({"checkpoint_publication_approved": True})

        summary = run_checkpoint_publication_authorization_preflight(
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("release_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["checkpoint_publication_approved"])

    def test_docs_missing_blocks_authorization(self) -> None:
        from scripts.run_checkpoint_publication_authorization_preflight import (
            run_checkpoint_publication_authorization_preflight,
        )

        self._write_inputs()
        fake_repo = self.temp_dir / "docless_repo"
        fake_repo.mkdir()

        summary = run_checkpoint_publication_authorization_preflight(
            stage8_root=self.stage8_root,
            stage7_root=self.stage7_root,
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=fake_repo,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("docs_not_updated", summary["reason_codes"])

    def test_shell_docs_and_spec_contract_are_declared(self) -> None:
        shell_path = self.repo_root / "scripts" / "run_checkpoint_publication_authorization_preflight.sh"
        spec_path = (
            self.repo_root
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-06-15-checkpoint-publication-authorization-preflight.md"
        )
        self.assertTrue(shell_path.is_file())
        self.assertTrue(spec_path.is_file())
        spec_text = spec_path.read_text(encoding="utf-8")
        self.assertIn("checkpoint-publication-authorization-preflight-summary.json", spec_text)
        self.assertIn("README.md", spec_text)
        self.assertIn("docs/算法设计与系统架构报告.md", spec_text)
        self.assertIn("checkpoint_publication_package_preparation", spec_text)
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
        self._write_stage8_inputs()
        self._write_json(
            self.stage7_root / "formal-performance-claim-release-decision-summary.json",
            {
                "schema_version": "formal-performance-claim-release-decision-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "decision_verdict": "approved_for_scoped_offline_performance_claim",
                "scoped_offline_performance_claim_approved": True,
                "performance_claim_scope": "scoped_offline_guarded_shadow_canary_only",
                "checkpoint_publication_approved": False,
                "default_policy_replacement_approved": False,
                "real_executor_connection_approved": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.stage6_root / "shadow-canary-release-performance-validation-preflight-summary.json",
            {
                "schema_version": "shadow-canary-release-performance-validation-preflight-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "long_horizon_shadow_passed": True,
                "checkpoint_publication_approved": False,
                "default_policy_replacement_approved": False,
                "real_executor_connection_approved": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.cost_root / "cost-efficiency-aware-coverage-reward-candidate-filter-summary.json",
            {"status": "passed", "reason_codes": [], "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False},
        )
        self._write_json(
            self.formal_root / "formal-ppo-training-run-summary.json",
            {"status": "passed", "reason_codes": [], "publishes_checkpoint": False, "replaces_default_policy": False},
        )
        self._write_json(
            self.replay_root / "formal-ppo-post-training-stability-replay-summary.json",
            {"status": "passed", "reason_codes": [], "publishes_checkpoint": False, "replaces_default_policy": False},
        )
        self._write_json(
            self.selected_root / "selected-formal-ppo-candidate-promotion-preflight-summary.json",
            {
                "schema_version": "selected-formal-ppo-candidate-promotion-preflight-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "selected_seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "checkpoint_path": str(self.checkpoint_path),
                "checkpoint_metadata_path": str(self.metadata_path),
                "checkpoint_sha256": self.checkpoint_sha256,
                "checkpoint_size_bytes": self.checkpoint_size,
                "checkpoint_load_passed": True,
                "invalid_action_mask_count": 0,
                "missing_observation_count": 0,
                "non_finite_logits_count": 0,
                "non_finite_log_prob_count": 0,
                "non_finite_value_count": 0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "performance_claimed": False,
            },
        )

    def _write_stage8_inputs(self) -> None:
        stage8_summary = {
            "schema_version": "scoped-claim-publication-evidence-freeze-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "publication_verdict": "approved_for_scoped_claim_publication",
            "scoped_claim_publication_approved": True,
            "performance_claim_scope": "scoped_offline_guarded_shadow_canary_only",
            "next_required_change": "checkpoint_publication_authorization_preflight",
            "evidence_freeze_complete": True,
            "doc_consistency_audit_passed": True,
            "release_boundary_audit_passed": True,
            "metric_mismatch_count": 0,
            "checkpoint_publication_approved": False,
            "default_policy_replacement_approved": False,
            "real_executor_connection_approved": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "bundle_manifest": str(self.stage8_root / "scoped-claim-publication-bundle-manifest.json"),
            "claim_text": str(self.stage8_root / "scoped-claim-publication-claim-text.md"),
            "scope_audit": str(self.stage8_root / "scoped-claim-publication-scope-audit.json"),
            "evidence_freeze_ledger": str(self.stage8_root / "scoped-claim-publication-evidence-freeze-ledger.json"),
            "doc_consistency_audit": str(self.stage8_root / "scoped-claim-publication-doc-consistency-audit.json"),
            "release_boundary_audit": str(self.stage8_root / "scoped-claim-publication-release-boundary-audit.json"),
        }
        self._write_json(self.stage8_root / "scoped-claim-publication-evidence-freeze-summary.json", stage8_summary)
        self._write_json(
            self.stage8_root / "scoped-claim-publication-bundle-manifest.json",
            {"publication_verdict": "approved_for_scoped_claim_publication", "blocked_release_actions": ["checkpoint_publication", "default_policy_replacement", "real_executor_connection"]},
        )
        self._write_text(
            self.stage8_root / "scoped-claim-publication-claim-text.md",
            "Allowed scoped offline claim within current offline guarded shadow/canary only.\n",
        )
        self._write_json(self.stage8_root / "scoped-claim-publication-scope-audit.json", {"scope_audit_passed": True})
        self._write_json(
            self.stage8_root / "scoped-claim-publication-evidence-freeze-ledger.json",
            {"evidence_freeze_complete": True, "all_sources_passed": True},
        )
        self._write_json(
            self.stage8_root / "scoped-claim-publication-doc-consistency-audit.json",
            {"doc_consistency_audit_passed": True},
        )
        self._write_json(
            self.stage8_root / "scoped-claim-publication-release-boundary-audit.json",
            {"release_boundary_audit_passed": True, "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False},
        )

    def _patch_stage8_summary(self, updates: dict) -> None:
        path = self.stage8_root / "scoped-claim-publication-evidence-freeze-summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(updates)
        self._write_json(path, payload)

    def _patch_selected_summary(self, updates: dict) -> None:
        path = self.selected_root / "selected-formal-ppo-candidate-promotion-preflight-summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(updates)
        self._write_json(path, payload)

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
