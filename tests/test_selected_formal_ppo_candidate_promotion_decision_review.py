import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path


class SelectedFormalPpoCandidatePromotionDecisionReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="selected-formal-decision-"))
        self.preflight_root = self.temp_dir / "promotion-preflight"
        self.candidate_root = self.temp_dir / "selected-candidate"
        self.multihorizon_root = self.temp_dir / "multihorizon"
        self.selection_root = self.temp_dir / "candidate-selection"
        self.stability_root = self.temp_dir / "stability"
        self.output_root = self.temp_dir / "decision-review"
        self.batch_root = self.temp_dir / "batch"
        for path in (
            self.preflight_root,
            self.candidate_root,
            self.multihorizon_root,
            self.selection_root,
            self.stability_root,
            self.batch_root,
        ):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_builds_decision_summary_lineage_identity_and_release_audits(self) -> None:
        from scripts.run_selected_formal_ppo_candidate_promotion_decision_review import (
            run_selected_formal_ppo_candidate_promotion_decision_review,
        )

        preflight_summary_path = self._write_preflight_lineage()

        result = run_selected_formal_ppo_candidate_promotion_decision_review(
            preflight_root=self.preflight_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(
            result["schema_version"],
            "selected-formal-ppo-candidate-promotion-decision-review-summary/v1",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(
            result["decision_verdict"],
            "eligible_for_guarded_release_candidate_packaging",
        )
        self.assertEqual(result["preflight_summary"], str(preflight_summary_path))
        self.assertEqual(result["selected_seed"], 0)
        self.assertEqual(result["selected_budget"], "epochs1_lr3e-6")
        self.assertEqual(result["checkpoint_sha256"], self.checkpoint_sha256)
        self.assertEqual(result["checkpoint_size_bytes"], len(self.checkpoint_bytes))
        self.assertTrue(result["lineage_audit_passed"])
        self.assertTrue(result["checkpoint_identity_audit_passed"])
        self.assertTrue(result["release_boundary_audit_passed"])
        self.assertFalse(result["runs_new_ppo_update"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])
        self.assertEqual(
            result["readiness_status"],
            "selected_formal_ppo_candidate_promotion_decision_review_evaluated",
        )

        lineage = json.loads(Path(result["lineage_report"]).read_text(encoding="utf-8"))
        self.assertEqual(
            [source["name"] for source in lineage["sources"]],
            [
                "promotion_preflight",
                "multihorizon_shadow_rollout",
                "candidate_selection_long_horizon_holdout",
                "formal_stability_holdout",
            ],
        )
        self.assertTrue(all(source["passed"] for source in lineage["sources"]))

        for filename in (
            "selected-formal-ppo-candidate-promotion-decision-review-summary.json",
            "evidence-lineage-report.json",
            "checkpoint-identity-audit.json",
            "release-boundary-audit.json",
            "promotion-decision-readiness-validate-only.json",
            "promotion-decision-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_lineage_source_blocks_decision(self) -> None:
        from scripts.run_selected_formal_ppo_candidate_promotion_decision_review import (
            run_selected_formal_ppo_candidate_promotion_decision_review,
        )

        self._write_preflight_lineage()
        (self.stability_root / "stability-summary.json").unlink()

        result = run_selected_formal_ppo_candidate_promotion_decision_review(
            preflight_root=self.preflight_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("promotion_decision_lineage_source_missing", result["reason_codes"])
        self.assertEqual(result["decision_verdict"], "blocked_by_preflight_or_provenance")
        self.assertFalse(result["lineage_audit_passed"])

    def test_checkpoint_hash_mismatch_blocks_decision(self) -> None:
        from scripts.run_selected_formal_ppo_candidate_promotion_decision_review import (
            run_selected_formal_ppo_candidate_promotion_decision_review,
        )

        self._write_preflight_lineage()
        (self.candidate_root / "experimental-hybrid-policy-candidate.pt").write_bytes(
            b"changed-checkpoint"
        )

        result = run_selected_formal_ppo_candidate_promotion_decision_review(
            preflight_root=self.preflight_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("promotion_decision_checkpoint_hash_mismatch", result["reason_codes"])
        self.assertFalse(result["checkpoint_identity_audit_passed"])

    def test_release_boundary_claim_blocks_decision(self) -> None:
        from scripts.run_selected_formal_ppo_candidate_promotion_decision_review import (
            run_selected_formal_ppo_candidate_promotion_decision_review,
        )

        preflight_summary_path = self._write_preflight_lineage()
        payload = json.loads(preflight_summary_path.read_text(encoding="utf-8"))
        payload["publishes_checkpoint"] = True
        preflight_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        result = run_selected_formal_ppo_candidate_promotion_decision_review(
            preflight_root=self.preflight_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("promotion_decision_release_boundary_invalid", result["reason_codes"])
        self.assertFalse(result["release_boundary_audit_passed"])
        self.assertFalse(result["publishes_checkpoint"])

    def test_config_declares_outputs_docs_and_non_goals(self) -> None:
        config_path = (
            self.repo_root
            / "configs"
            / "selected_formal_ppo_candidate_promotion_decision_review_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["schema_version"],
            "selected-formal-ppo-candidate-promotion-decision-review-config/v1",
        )
        self.assertIn(
            "selected-formal-ppo-candidate-promotion-decision-review-summary.json",
            config["output_files"].values(),
        )
        self.assertIn("README.md", config["documentation_updates"])
        self.assertIn("docs/算法设计与系统架构报告.md", config["documentation_updates"])
        self.assertIn("does_not_publish_checkpoint", config["non_goals"])
        self.assertIn("does_not_claim_formal_training_ready", config["non_goals"])

    def _write_preflight_lineage(self) -> Path:
        self.checkpoint_bytes = b"experimental-checkpoint-v1"
        checkpoint_path = self.candidate_root / "experimental-hybrid-policy-candidate.pt"
        checkpoint_path.write_bytes(self.checkpoint_bytes)
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint_bytes).hexdigest()

        checkpoint_metadata_path = (
            self.candidate_root / "experimental-hybrid-policy-candidate-metadata.json"
        )
        checkpoint_metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": "controlled-hybrid-policy-candidate-checkpoint-metadata/v1",
                    "experimental": True,
                    "checkpoint_path": str(checkpoint_path),
                    "seed": 0,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        stability_summary_path = self.stability_root / "stability-summary.json"
        stability_summary_path.write_text(
            json.dumps(
                self._source_summary(
                    "quasi-real-guarded-formal-ppo-stability-holdout-validation-summary/v1"
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

        selected_manifest_path = self.selection_root / "selected-candidate-manifest.json"
        selected_manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "quasi-real-guarded-formal-ppo-selected-candidate-manifest/v1",
                    "selected_seed": 0,
                    "selected_budget": "epochs1_lr3e-6",
                    "selected_candidate_root": str(self.candidate_root),
                    "stability_summary": str(stability_summary_path),
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

        selection_summary_path = self.selection_root / "candidate-selection-summary.json"
        selection_summary_path.write_text(
            json.dumps(
                {
                    **self._source_summary(
                        "quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-summary/v1"
                    ),
                    "selected_seed": 0,
                    "selected_budget": "epochs1_lr3e-6",
                    "selected_candidate_root": str(self.candidate_root),
                    "stability_summary": str(stability_summary_path),
                    "candidate_manifest": str(selected_manifest_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        multihorizon_summary_path = self.multihorizon_root / "multihorizon-summary.json"
        multihorizon_summary_path.write_text(
            json.dumps(
                {
                    **self._source_summary(
                        "selected-formal-ppo-candidate-multihorizon-shadow-rollout-summary/v1"
                    ),
                    "selected_seed": 0,
                    "selected_budget": "epochs1_lr3e-6",
                    "selected_candidate_root": str(self.candidate_root),
                    "selected_candidate_from_candidate_selection": True,
                    "candidate_selection_summary": str(selection_summary_path),
                    "candidate_manifest": str(selected_manifest_path),
                    "input_trainable_transition_count": 684,
                    "shadow_trainable_transition_count": 2052,
                    "unique_trainable_context_count": 684,
                    "horizons": [10, 20, 30],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        manifest_path = self.preflight_root / "promotion-candidate-manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "selected-formal-ppo-candidate-promotion-manifest/v1",
                    "selected_seed": 0,
                    "selected_budget": "epochs1_lr3e-6",
                    "selected_candidate_root": str(self.candidate_root),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_metadata_path": str(checkpoint_metadata_path),
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "checkpoint_size_bytes": len(self.checkpoint_bytes),
                    "multihorizon_summary": str(multihorizon_summary_path),
                    "candidate_summary": str(self.candidate_root / "raw-policy-generalization-candidate-summary.json"),
                    "limited_ppo_summary": str(self.candidate_root / "limited-ppo-update-smoke-summary.json"),
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

        hash_audit_path = self.preflight_root / "checkpoint-hash-audit.json"
        hash_audit_path.write_text(
            json.dumps(
                {
                    "checkpoint_exists": True,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "checkpoint_size_bytes": len(self.checkpoint_bytes),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        rollback_audit_path = self.preflight_root / "rollback-audit.json"
        rollback_audit_path.write_text(
            json.dumps(
                {
                    "rollback_audit_passed": True,
                    "experimental_candidate_only": True,
                    "default_policy_replaced": False,
                    "publication_flags": {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        preflight_summary_path = (
            self.preflight_root
            / "selected-formal-ppo-candidate-promotion-preflight-summary.json"
        )
        preflight_summary_path.write_text(
            json.dumps(
                {
                    **self._source_summary(
                        "selected-formal-ppo-candidate-promotion-preflight-summary/v1"
                    ),
                    "readiness_status": "selected_formal_ppo_candidate_promotion_preflight_evaluated",
                    "selected_seed": 0,
                    "selected_budget": "epochs1_lr3e-6",
                    "selected_candidate_root": str(self.candidate_root),
                    "selected_candidate_from_multihorizon_shadow": True,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_metadata_path": str(checkpoint_metadata_path),
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "checkpoint_size_bytes": len(self.checkpoint_bytes),
                    "checkpoint_load_passed": True,
                    "inference_audit_count": 64,
                    "invalid_action_mask_count": 0,
                    "missing_observation_count": 0,
                    "non_finite_logits_count": 0,
                    "non_finite_log_prob_count": 0,
                    "non_finite_value_count": 0,
                    "controlled_regression_count": 0,
                    "family_regression_count": 0,
                    "teacher_agreement_rate": 1.0,
                    "rollback_audit_passed": True,
                    "promotion_manifest": str(manifest_path),
                    "checkpoint_hash_audit": str(hash_audit_path),
                    "rollback_audit": str(rollback_audit_path),
                    "multihorizon_summary": str(multihorizon_summary_path),
                    "input_trainable_transition_count": 684,
                    "shadow_trainable_transition_count": 2052,
                    "unique_trainable_context_count": 684,
                    "horizons": [10, 20, 30],
                    "runs_promotion_preflight": True,
                    "runs_new_ppo_update": False,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return preflight_summary_path

    def _source_summary(self, schema: str) -> dict:
        return {
            "schema_version": schema,
            "status": "passed",
            "reason_codes": [],
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "git_provenance": {"current_matches_sources": True},
        }

    def _config(self) -> dict:
        return {
            "schema_version": "selected-formal-ppo-candidate-promotion-decision-review-config/v1",
            "readiness": {
                "config": "configs/policy_training_readiness_review_v1.json",
                "expected_status": "selected_formal_ppo_candidate_promotion_decision_review_evaluated",
            },
            "input_files": {
                "preflight_summary": "selected-formal-ppo-candidate-promotion-preflight-summary.json"
            },
            "output_files": {
                "summary": "selected-formal-ppo-candidate-promotion-decision-review-summary.json",
                "lineage_report": "evidence-lineage-report.json",
                "checkpoint_identity_audit": "checkpoint-identity-audit.json",
                "release_boundary_audit": "release-boundary-audit.json",
                "readiness_validate_only": "promotion-decision-readiness-validate-only.json",
                "report": "promotion-decision-report.md",
            },
        }

    def _passing_readiness(self, **_kwargs) -> dict:
        return {
            "training_readiness_status": "selected_formal_ppo_candidate_promotion_decision_review_evaluated",
            "training_blockers": [],
            "reason_codes": [],
        }


if __name__ == "__main__":
    unittest.main()
