import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path


class GuardedExperimentalPolicyReleaseCandidatePackagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="guarded-rc-package-"))
        self.decision_root = self.temp_dir / "decision-review"
        self.candidate_root = self.temp_dir / "selected-candidate"
        self.output_root = self.temp_dir / "release-candidate-package"
        self.batch_root = self.temp_dir / "batch"
        for path in (
            self.decision_root,
            self.candidate_root,
            self.output_root,
            self.batch_root,
        ):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_packages_reviewed_candidate_with_identity_load_and_rollback_audits(self) -> None:
        from scripts.run_guarded_experimental_policy_release_candidate_packaging import (
            run_guarded_experimental_policy_release_candidate_packaging,
        )

        decision_summary_path = self._write_decision_review_summary()

        result = run_guarded_experimental_policy_release_candidate_packaging(
            decision_root=self.decision_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
            load_audit_runner=self._passing_load_audit,
        )

        self.assertEqual(
            result["schema_version"],
            "guarded-experimental-policy-release-candidate-packaging-summary/v1",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(
            result["package_verdict"],
            "eligible_for_guarded_install_dry_run",
        )
        self.assertEqual(result["decision_review_summary"], str(decision_summary_path))
        self.assertEqual(result["original_checkpoint_path"], str(self.checkpoint_path))
        self.assertNotEqual(result["package_checkpoint_path"], str(self.checkpoint_path))
        self.assertTrue(Path(result["package_checkpoint_path"]).is_file())
        self.assertEqual(Path(result["package_checkpoint_path"]).read_bytes(), self.checkpoint_bytes)
        self.assertEqual(result["checkpoint_sha256"], self.checkpoint_sha256)
        self.assertEqual(result["package_checkpoint_sha256"], self.checkpoint_sha256)
        self.assertEqual(result["checkpoint_size_bytes"], len(self.checkpoint_bytes))
        self.assertEqual(result["package_checkpoint_size_bytes"], len(self.checkpoint_bytes))
        self.assertTrue(result["checkpoint_identity_audit_passed"])
        self.assertTrue(result["checkpoint_load_passed"])
        self.assertEqual(result["checkpoint_load_sample_count"], 64)
        self.assertEqual(result["invalid_action_mask_count"], 0)
        self.assertEqual(result["missing_observation_count"], 0)
        self.assertEqual(result["non_finite_logits_count"], 0)
        self.assertEqual(result["non_finite_log_prob_count"], 0)
        self.assertEqual(result["non_finite_value_count"], 0)
        self.assertTrue(result["rollback_audit_passed"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])
        self.assertEqual(
            result["readiness_status"],
            "guarded_experimental_policy_release_candidate_packaging_evaluated",
        )

        manifest = json.loads(Path(result["package_manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["schema_version"],
            "guarded-experimental-policy-release-candidate-package-manifest/v1",
        )
        self.assertEqual(manifest["original_checkpoint_sha256"], self.checkpoint_sha256)
        self.assertEqual(manifest["package_checkpoint_sha256"], self.checkpoint_sha256)
        self.assertEqual(manifest["selected_seed"], 0)
        self.assertEqual(manifest["selected_budget"], "epochs1_lr3e-6")
        self.assertFalse(manifest["publishes_checkpoint"])
        self.assertFalse(manifest["replaces_default_policy"])
        self.assertTrue(manifest["rollback_source_traceable"])
        self.assertEqual(len(manifest["source_lineage"]), 4)
        self.assertTrue(all(source["passed"] for source in manifest["source_lineage"]))
        self.assertTrue(
            all(
                source["git_current_matches_sources"] is True
                for source in manifest["source_lineage"]
            )
        )

        for filename in (
            "guarded-experimental-policy-release-candidate-packaging-summary.json",
            "release-candidate-package-manifest.json",
            "checkpoint-hash-audit.json",
            "checkpoint-load-audit.json",
            "rollback-audit.json",
            "packaging-readiness-validate-only.json",
            "release-candidate-packaging-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_hash_mismatch_blocks_packaging(self) -> None:
        from scripts.run_guarded_experimental_policy_release_candidate_packaging import (
            run_guarded_experimental_policy_release_candidate_packaging,
        )

        self._write_decision_review_summary()
        self.checkpoint_path.write_bytes(b"mutated-after-review")

        result = run_guarded_experimental_policy_release_candidate_packaging(
            decision_root=self.decision_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
            load_audit_runner=self._passing_load_audit,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("release_candidate_checkpoint_hash_mismatch", result["reason_codes"])
        self.assertEqual(result["package_verdict"], "blocked_by_checkpoint_identity")
        self.assertFalse(result["checkpoint_identity_audit_passed"])

    def test_config_declares_outputs_docs_and_non_goals(self) -> None:
        config_path = (
            self.repo_root
            / "configs"
            / "guarded_experimental_policy_release_candidate_packaging_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["schema_version"],
            "guarded-experimental-policy-release-candidate-packaging-config/v1",
        )
        self.assertIn(
            "guarded-experimental-policy-release-candidate-packaging-summary.json",
            config["output_files"].values(),
        )
        self.assertIn("README.md", config["documentation_updates"])
        self.assertIn("docs/算法设计与系统架构报告.md", config["documentation_updates"])
        self.assertIn("does_not_run_new_ppo_update", config["non_goals"])
        self.assertIn("does_not_execute_install_or_canary", config["non_goals"])
        self.assertIn("does_not_replace_default_policy", config["non_goals"])

    def _write_decision_review_summary(self) -> Path:
        self.checkpoint_bytes = b"experimental-checkpoint-v1"
        self.checkpoint_path = self.candidate_root / "experimental-hybrid-policy-candidate.pt"
        self.checkpoint_path.write_bytes(self.checkpoint_bytes)
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint_bytes).hexdigest()

        self.metadata_path = (
            self.candidate_root / "experimental-hybrid-policy-candidate-metadata.json"
        )
        self.metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": "controlled-hybrid-policy-candidate-checkpoint-metadata/v1",
                    "experimental": True,
                    "checkpoint_path": str(self.checkpoint_path),
                    "seed": 0,
                    "selected_seed": 0,
                    "selected_budget": "epochs1_lr3e-6",
                    "selected_candidate_root": str(self.candidate_root),
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        lineage_report_path = self.decision_root / "evidence-lineage-report.json"
        lineage_report_path.write_text(
            json.dumps(
                {
                    "schema_version": "selected-formal-ppo-candidate-promotion-decision-lineage/v1",
                    "sources": [
                        {
                            "name": "promotion_preflight",
                            "passed": True,
                            "git_current_matches_sources": True,
                            "publication_flags": {},
                        },
                        {
                            "name": "multihorizon_shadow_rollout",
                            "passed": True,
                            "git_current_matches_sources": True,
                            "publication_flags": {},
                        },
                        {
                            "name": "candidate_selection_long_horizon_holdout",
                            "passed": True,
                            "git_current_matches_sources": True,
                            "publication_flags": {},
                        },
                        {
                            "name": "formal_stability_holdout",
                            "passed": True,
                            "git_current_matches_sources": True,
                            "publication_flags": {},
                        },
                    ],
                    "git_provenance": {"current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        checkpoint_identity_path = self.decision_root / "checkpoint-identity-audit.json"
        checkpoint_identity_path.write_text(
            json.dumps(
                {
                    "schema_version": "selected-formal-ppo-candidate-promotion-checkpoint-identity-audit/v1",
                    "checkpoint_identity_audit_passed": True,
                    "reason_codes": [],
                    "checkpoint_path": str(self.checkpoint_path),
                    "checkpoint_metadata_path": str(self.metadata_path),
                    "checkpoint_exists": True,
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "checkpoint_size_bytes": len(self.checkpoint_bytes),
                    "metadata_schema_version": "controlled-hybrid-policy-candidate-checkpoint-metadata/v1",
                    "metadata_experimental": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        release_boundary_path = self.decision_root / "release-boundary-audit.json"
        release_boundary_path.write_text(
            json.dumps(
                {
                    "schema_version": "selected-formal-ppo-candidate-promotion-release-boundary-audit/v1",
                    "release_boundary_audit_passed": True,
                    "experimental_candidate_only": True,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        decision_summary_path = (
            self.decision_root
            / "selected-formal-ppo-candidate-promotion-decision-review-summary.json"
        )
        decision_summary_path.write_text(
            json.dumps(
                {
                    "schema_version": "selected-formal-ppo-candidate-promotion-decision-review-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "decision_verdict": "eligible_for_guarded_release_candidate_packaging",
                    "readiness_status": "selected_formal_ppo_candidate_promotion_decision_review_evaluated",
                    "selected_seed": 0,
                    "selected_budget": "epochs1_lr3e-6",
                    "selected_candidate_root": str(self.candidate_root),
                    "checkpoint_path": str(self.checkpoint_path),
                    "checkpoint_metadata_path": str(self.metadata_path),
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "checkpoint_size_bytes": len(self.checkpoint_bytes),
                    "lineage_audit_passed": True,
                    "source_lineage_count": 4,
                    "checkpoint_identity_audit_passed": True,
                    "release_boundary_audit_passed": True,
                    "preflight_summary": str(self.decision_root / "preflight-summary.json"),
                    "lineage_report": str(lineage_report_path),
                    "checkpoint_identity_audit": str(checkpoint_identity_path),
                    "release_boundary_audit": str(release_boundary_path),
                    "runs_new_ppo_update": False,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                    "git_provenance": {"current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return decision_summary_path

    def _config(self) -> dict:
        return {
            "schema_version": "guarded-experimental-policy-release-candidate-packaging-config/v1",
            "input_files": {
                "decision_review_summary": "selected-formal-ppo-candidate-promotion-decision-review-summary.json"
            },
            "validation": {"min_load_sample_count": 64},
            "readiness": {
                "config": "configs/policy_training_readiness_review_v1.json",
                "expected_status": "guarded_experimental_policy_release_candidate_packaging_evaluated",
            },
            "output_files": {
                "summary": "guarded-experimental-policy-release-candidate-packaging-summary.json",
                "package_manifest": "release-candidate-package-manifest.json",
                "checkpoint_hash_audit": "checkpoint-hash-audit.json",
                "checkpoint_load_audit": "checkpoint-load-audit.json",
                "rollback_audit": "rollback-audit.json",
                "readiness_validate_only": "packaging-readiness-validate-only.json",
                "report": "release-candidate-packaging-report.md",
            },
        }

    def _passing_load_audit(self, **kwargs) -> dict:
        checkpoint_path = Path(kwargs["checkpoint_path"])
        self.assertTrue(checkpoint_path.is_relative_to(self.output_root))
        return {
            "checkpoint_load_passed": True,
            "checkpoint_load_sample_count": 64,
            "invalid_action_mask_count": 0,
            "missing_observation_count": 0,
            "non_finite_logits_count": 0,
            "non_finite_log_prob_count": 0,
            "non_finite_value_count": 0,
            "sampled_rows": [],
        }

    def _passing_readiness(self, **_kwargs) -> dict:
        return {
            "training_readiness_status": "guarded_experimental_policy_release_candidate_packaging_evaluated",
            "training_blockers": [],
            "reason_codes": [],
        }


if __name__ == "__main__":
    unittest.main()
