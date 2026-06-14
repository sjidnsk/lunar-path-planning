import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path


class GuardedExperimentalPolicyInstallCanaryDryRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="guarded-install-canary-"))
        self.packaging_root = self.temp_dir / "packaging"
        self.package_dir = self.packaging_root / "release-candidate-package"
        self.output_root = self.temp_dir / "install-canary"
        self.batch_root = self.temp_dir / "batch"
        self.default_policy = self.temp_dir / "default-policy.pt"
        for path in (self.package_dir, self.output_root, self.batch_root):
            path.mkdir(parents=True)
        self.default_policy.write_bytes(b"default-policy-v1")
        self.default_policy_sha256 = hashlib.sha256(self.default_policy.read_bytes()).hexdigest()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_installs_packaged_candidate_in_sandbox_and_runs_guarded_canary(self) -> None:
        from scripts.run_guarded_experimental_policy_install_canary_dry_run import (
            run_guarded_experimental_policy_install_canary_dry_run,
        )

        packaging_summary_path = self._write_packaging_summary()

        result = run_guarded_experimental_policy_install_canary_dry_run(
            packaging_root=self.packaging_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            canary_runner=self._passing_canary,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(
            result["schema_version"],
            "guarded-experimental-policy-install-canary-dry-run-summary/v1",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(
            result["install_canary_verdict"],
            "eligible_for_guarded_shadow_release_trial",
        )
        self.assertEqual(result["packaging_summary"], str(packaging_summary_path))
        self.assertTrue(result["sandbox_manifest_passed"])
        self.assertTrue(Path(result["sandbox_manifest"]).is_file())
        self.assertEqual(result["package_checkpoint_sha256"], self.checkpoint_sha256)
        self.assertEqual(result["consumer_checkpoint_sha256"], self.checkpoint_sha256)
        self.assertEqual(result["package_checkpoint_size_bytes"], len(self.checkpoint_bytes))
        self.assertEqual(result["consumer_checkpoint_size_bytes"], len(self.checkpoint_bytes))
        self.assertEqual(result["canary_step_count"], 64)
        self.assertEqual(result["missing_observation_count"], 0)
        self.assertEqual(result["invalid_action_mask_count"], 0)
        self.assertEqual(result["non_finite_logits_count"], 0)
        self.assertEqual(result["non_finite_log_prob_count"], 0)
        self.assertEqual(result["non_finite_value_count"], 0)
        self.assertEqual(result["non_finite_reward_count"], 0)
        self.assertEqual(result["controlled_regression_count"], 0)
        self.assertEqual(result["controlled_safety_regression_count"], 0)
        self.assertEqual(result["controlled_contract_regression_count"], 0)
        self.assertEqual(result["controlled_path_risk_regression_count"], 0)
        self.assertEqual(result["controlled_source_selection_regression_count"], 0)
        self.assertTrue(result["rollback_default_audit_passed"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])
        self.assertEqual(
            result["readiness_status"],
            "guarded_experimental_policy_install_canary_dry_run_evaluated",
        )

        manifest = json.loads(Path(result["sandbox_manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["schema_version"],
            "guarded-experimental-policy-install-canary-sandbox-manifest/v1",
        )
        self.assertEqual(manifest["package_checkpoint_sha256"], self.checkpoint_sha256)
        self.assertEqual(manifest["default_policy_sha256_before"], self.default_policy_sha256)
        self.assertEqual(manifest["default_policy_sha256_after"], self.default_policy_sha256)
        self.assertFalse(manifest["default_policy_replaced"])

        for filename in (
            "guarded-experimental-policy-install-canary-dry-run-summary.json",
            "install-canary-sandbox-manifest.json",
            "install-canary-package-consumer-audit.json",
            "install-canary-step-audit.jsonl",
            "install-canary-rollback-audit.json",
            "install-canary-readiness-validate-only.json",
            "install-canary-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_controlled_regression_and_default_policy_mutation_block_canary(self) -> None:
        from scripts.run_guarded_experimental_policy_install_canary_dry_run import (
            run_guarded_experimental_policy_install_canary_dry_run,
        )

        self._write_packaging_summary()

        result = run_guarded_experimental_policy_install_canary_dry_run(
            packaging_root=self.packaging_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            canary_runner=self._regressing_canary_that_mutates_default,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["install_canary_verdict"], "blocked_by_guarded_canary")
        self.assertIn("install_canary_controlled_regression", result["reason_codes"])
        self.assertIn("install_canary_default_policy_modified", result["reason_codes"])
        self.assertFalse(result["rollback_default_audit_passed"])
        self.assertNotEqual(
            result["default_policy_sha256_before"],
            result["default_policy_sha256_after"],
        )

    def test_config_declares_outputs_docs_and_non_goals(self) -> None:
        config_path = (
            self.repo_root
            / "configs"
            / "guarded_experimental_policy_install_canary_dry_run_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["schema_version"],
            "guarded-experimental-policy-install-canary-dry-run-config/v1",
        )
        self.assertIn(
            "guarded-experimental-policy-install-canary-dry-run-summary.json",
            config["output_files"].values(),
        )
        self.assertIn("README.md", config["documentation_updates"])
        self.assertIn("docs/算法设计与系统架构报告.md", config["documentation_updates"])
        self.assertIn("does_not_run_new_ppo_update", config["non_goals"])
        self.assertIn("does_not_publish_checkpoint", config["non_goals"])
        self.assertIn("does_not_replace_default_policy", config["non_goals"])

    def _write_packaging_summary(self) -> Path:
        self.checkpoint_bytes = b"experimental-checkpoint-v1"
        self.package_checkpoint_path = self.package_dir / "experimental-hybrid-policy-candidate.pt"
        self.package_checkpoint_path.write_bytes(self.checkpoint_bytes)
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint_bytes).hexdigest()
        self.metadata_path = self.package_dir / "experimental-hybrid-policy-candidate-metadata.json"
        self.metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": "controlled-hybrid-policy-candidate-checkpoint-metadata/v1",
                    "experimental": True,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest_path = self.packaging_root / "release-candidate-package-manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "guarded-experimental-policy-release-candidate-package-manifest/v1",
                    "package_root": str(self.package_dir),
                    "package_checkpoint_path": str(self.package_checkpoint_path),
                    "package_checkpoint_sha256": self.checkpoint_sha256,
                    "package_checkpoint_size_bytes": len(self.checkpoint_bytes),
                    "package_checkpoint_metadata_path": str(self.metadata_path),
                    "selected_seed": 0,
                    "selected_budget": "epochs1_lr3e-6",
                    "source_lineage_count": 4,
                    "rollback_source_traceable": True,
                    "runs_new_ppo_update": False,
                    "executes_install_or_canary": False,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary_path = (
            self.packaging_root
            / "guarded-experimental-policy-release-candidate-packaging-summary.json"
        )
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": "guarded-experimental-policy-release-candidate-packaging-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "package_verdict": "eligible_for_guarded_install_dry_run",
                    "package_manifest": str(manifest_path),
                    "package_root": str(self.package_dir),
                    "package_checkpoint_path": str(self.package_checkpoint_path),
                    "package_checkpoint_metadata_path": str(self.metadata_path),
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "package_checkpoint_sha256": self.checkpoint_sha256,
                    "checkpoint_size_bytes": len(self.checkpoint_bytes),
                    "package_checkpoint_size_bytes": len(self.checkpoint_bytes),
                    "checkpoint_identity_audit_passed": True,
                    "checkpoint_load_passed": True,
                    "checkpoint_load_sample_count": 64,
                    "invalid_action_mask_count": 0,
                    "missing_observation_count": 0,
                    "non_finite_logits_count": 0,
                    "non_finite_log_prob_count": 0,
                    "non_finite_value_count": 0,
                    "rollback_audit_passed": True,
                    "runs_release_candidate_packaging": True,
                    "runs_new_ppo_update": False,
                    "executes_install_or_canary": False,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                    "readiness_status": "guarded_experimental_policy_release_candidate_packaging_evaluated",
                    "git_provenance": {"current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return summary_path

    def _config(self) -> dict:
        return {
            "schema_version": "guarded-experimental-policy-install-canary-dry-run-config/v1",
            "input_files": {
                "packaging_summary": "guarded-experimental-policy-release-candidate-packaging-summary.json"
            },
            "validation": {"min_canary_step_count": 64},
            "default_policy": {"path": str(self.default_policy)},
            "readiness": {
                "config": "configs/policy_training_readiness_review_v1.json",
                "expected_status": "guarded_experimental_policy_install_canary_dry_run_evaluated",
            },
            "output_files": {
                "summary": "guarded-experimental-policy-install-canary-dry-run-summary.json",
                "sandbox_manifest": "install-canary-sandbox-manifest.json",
                "package_consumer_audit": "install-canary-package-consumer-audit.json",
                "step_audit": "install-canary-step-audit.jsonl",
                "rollback_audit": "install-canary-rollback-audit.json",
                "readiness_validate_only": "install-canary-readiness-validate-only.json",
                "report": "install-canary-report.md",
            },
        }

    def _passing_canary(self, **kwargs) -> dict:
        self.assertEqual(Path(kwargs["checkpoint_path"]), self.package_checkpoint_path)
        self.assertTrue(Path(kwargs["sandbox_manifest_path"]).is_file())
        return {
            "canary_step_count": 64,
            "missing_observation_count": 0,
            "invalid_action_mask_count": 0,
            "non_finite_logits_count": 0,
            "non_finite_log_prob_count": 0,
            "non_finite_value_count": 0,
            "non_finite_reward_count": 0,
            "controlled_regression_count": 0,
            "controlled_safety_regression_count": 0,
            "controlled_contract_regression_count": 0,
            "controlled_path_risk_regression_count": 0,
            "controlled_source_selection_regression_count": 0,
            "raw_policy_rejection_count": 0,
            "fallback_count": 0,
            "step_rows": [
                {
                    "step_index": i,
                    "raw_policy_action_index": 0,
                    "controlled_action_index": 0,
                    "controlled_choice_source": "policy",
                    "gate_reason_codes": [],
                    "controlled_regression_reason_codes": [],
                    "path_cost_delta": 0.0,
                    "risk_delta": 0.0,
                    "log_prob": -0.1,
                    "value": 0.2,
                    "reward": 1.0,
                }
                for i in range(64)
            ],
        }

    def _regressing_canary_that_mutates_default(self, **_kwargs) -> dict:
        self.default_policy.write_bytes(b"default-policy-mutated")
        payload = self._passing_canary(
            checkpoint_path=self.package_checkpoint_path,
            sandbox_manifest_path=self.output_root / "install-canary-sandbox-manifest.json",
        )
        payload["controlled_regression_count"] = 1
        payload["controlled_path_risk_regression_count"] = 1
        payload["step_rows"][0]["controlled_regression_reason_codes"] = [
            "path_cost_regression"
        ]
        payload["step_rows"][0]["path_cost_delta"] = 1.0
        return payload

    def _passing_readiness(self, **_kwargs) -> dict:
        return {
            "training_readiness_status": "guarded_experimental_policy_install_canary_dry_run_evaluated",
            "training_blockers": [],
            "reason_codes": [],
        }


if __name__ == "__main__":
    unittest.main()
