import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path


class GuardedExperimentalPolicyShadowReleaseTrialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="guarded-shadow-release-"))
        self.install_canary_root = self.temp_dir / "install-canary"
        self.multihorizon_root = self.temp_dir / "multihorizon"
        self.output_root = self.temp_dir / "shadow-release"
        self.batch_root = self.temp_dir / "batch"
        self.default_policy = self.temp_dir / "default-policy.pt"
        for path in (
            self.install_canary_root,
            self.multihorizon_root,
            self.output_root,
            self.batch_root,
        ):
            path.mkdir(parents=True)
        self.default_policy.write_bytes(b"default-policy-v1")
        self.default_policy_sha256 = hashlib.sha256(
            self.default_policy.read_bytes()
        ).hexdigest()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_runs_shadow_release_trial_without_taking_control(self) -> None:
        from scripts.run_guarded_experimental_policy_shadow_release_trial import (
            run_guarded_experimental_policy_shadow_release_trial,
        )

        self._write_install_canary_summary()
        self._write_multihorizon_shadow_artifacts(step_count=300)

        result = run_guarded_experimental_policy_shadow_release_trial(
            install_canary_root=self.install_canary_root,
            multihorizon_shadow_root=self.multihorizon_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(
            result["schema_version"],
            "guarded-experimental-policy-shadow-release-trial-summary/v1",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(
            result["shadow_release_trial_verdict"],
            "eligible_for_guarded_staged_release_preflight",
        )
        self.assertEqual(result["shadow_step_count"], 300)
        self.assertEqual(result["unique_shadow_context_count"], 300)
        self.assertEqual(result["shadow_fallback_diagnostic_count"], 2)
        self.assertEqual(result["shadow_rejection_diagnostic_count"], 2)
        self.assertEqual(result["controlled_regression_count"], 0)
        self.assertTrue(result["default_policy_unchanged"])
        self.assertTrue(result["rollback_default_audit_passed"])
        self.assertFalse(result["shadow_policy_takes_control"])
        self.assertFalse(result["runs_new_ppo_update"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])
        self.assertEqual(
            result["readiness_status"],
            "guarded_experimental_policy_shadow_release_trial_evaluated",
        )

        steps = [
            json.loads(line)
            for line in Path(result["step_comparison"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(steps), 300)
        self.assertTrue(all(step["shadow_policy_takes_control"] is False for step in steps))
        self.assertTrue(any(step["shadow_diagnostic_only"] for step in steps))

        for filename in (
            "guarded-experimental-policy-shadow-release-trial-summary.json",
            "shadow-release-runtime-manifest.json",
            "shadow-release-step-comparison.jsonl",
            "shadow-release-rejection-report.json",
            "shadow-release-risk-reward-audit.json",
            "shadow-release-readiness-validate-only.json",
            "shadow-release-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_controlled_regression_blocks_trial_but_shadow_rejection_remains_diagnostic(self) -> None:
        from scripts.run_guarded_experimental_policy_shadow_release_trial import (
            run_guarded_experimental_policy_shadow_release_trial,
        )

        self._write_install_canary_summary()
        self._write_multihorizon_shadow_artifacts(
            step_count=260,
            extra_step=self._shadow_step(
                999,
                gate_reason_codes=["path_cost_regression"],
                controlled_regression_reason_codes=["path_cost_regression"],
                path_cost_delta=0.5,
            ),
        )

        result = run_guarded_experimental_policy_shadow_release_trial(
            install_canary_root=self.install_canary_root,
            multihorizon_shadow_root=self.multihorizon_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("shadow_release_trial_controlled_regression", result["reason_codes"])
        self.assertEqual(result["controlled_regression_count"], 1)
        self.assertGreaterEqual(result["shadow_rejection_diagnostic_count"], 3)

    def test_config_declares_outputs_docs_and_non_goals(self) -> None:
        config_path = (
            self.repo_root
            / "configs"
            / "guarded_experimental_policy_shadow_release_trial_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["schema_version"],
            "guarded-experimental-policy-shadow-release-trial-config/v1",
        )
        self.assertEqual(config["validation"]["min_shadow_step_count"], 256)
        self.assertIn(
            "guarded-experimental-policy-shadow-release-trial-summary.json",
            config["output_files"].values(),
        )
        self.assertIn("README.md", config["documentation_updates"])
        self.assertIn("docs/算法设计与系统架构报告.md", config["documentation_updates"])
        self.assertIn("does_not_run_new_ppo_update", config["non_goals"])
        self.assertIn("does_not_replace_default_policy", config["non_goals"])

    def _config(self) -> dict:
        return {
            "schema_version": "guarded-experimental-policy-shadow-release-trial-config/v1",
            "input_files": {
                "install_canary_summary": "guarded-experimental-policy-install-canary-dry-run-summary.json",
                "multihorizon_shadow_summary": "multihorizon-shadow-rollout-summary.json",
                "multihorizon_shadow_steps": "multihorizon-shadow-rollout-steps.jsonl",
            },
            "validation": {
                "min_shadow_step_count": 256,
                "min_unique_shadow_context_count": 256,
            },
            "default_policy": {"path": str(self.default_policy)},
            "readiness": {
                "config": "configs/policy_training_readiness_review_v1.json",
                "expected_status": "guarded_experimental_policy_shadow_release_trial_evaluated",
            },
            "output_files": {
                "summary": "guarded-experimental-policy-shadow-release-trial-summary.json",
                "runtime_manifest": "shadow-release-runtime-manifest.json",
                "step_comparison": "shadow-release-step-comparison.jsonl",
                "rejection_report": "shadow-release-rejection-report.json",
                "risk_reward_audit": "shadow-release-risk-reward-audit.json",
                "readiness_validate_only": "shadow-release-readiness-validate-only.json",
                "report": "shadow-release-report.md",
            },
        }

    def _write_install_canary_summary(self) -> None:
        checkpoint_bytes = b"experimental-checkpoint-v1"
        checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        (self.install_canary_root / "checkpoint.pt").write_bytes(checkpoint_bytes)
        summary = {
            "schema_version": "guarded-experimental-policy-install-canary-dry-run-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "install_canary_verdict": "eligible_for_guarded_shadow_release_trial",
            "package_checkpoint_path": str(self.install_canary_root / "checkpoint.pt"),
            "package_checkpoint_sha256": checkpoint_sha256,
            "consumer_checkpoint_sha256": checkpoint_sha256,
            "package_checkpoint_size_bytes": len(checkpoint_bytes),
            "consumer_checkpoint_size_bytes": len(checkpoint_bytes),
            "package_consumer_audit_passed": True,
            "sandbox_manifest_passed": True,
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
            "rollback_default_audit_passed": True,
            "default_policy_unchanged": True,
            "default_policy_sha256_before": self.default_policy_sha256,
            "default_policy_sha256_after": self.default_policy_sha256,
            "runs_install_canary_dry_run": True,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "readiness_status": "guarded_experimental_policy_install_canary_dry_run_evaluated",
            "git_provenance": {"current_matches_sources": True},
        }
        (
            self.install_canary_root
            / "guarded-experimental-policy-install-canary-dry-run-summary.json"
        ).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    def _write_multihorizon_shadow_artifacts(
        self,
        *,
        step_count: int,
        extra_step: dict | None = None,
    ) -> None:
        steps = [self._shadow_step(index) for index in range(step_count)]
        steps[10]["gate_reason_codes"] = ["raw_policy_action_rejected"]
        steps[10]["controlled_choice_source"] = "source_fallback"
        steps[11]["gate_reason_codes"] = ["raw_policy_action_rejected"]
        steps[11]["controlled_choice_source"] = "source_fallback"
        if extra_step:
            steps.append(extra_step)
        steps_path = self.multihorizon_root / "multihorizon-shadow-rollout-steps.jsonl"
        steps_path.write_text(
            "".join(json.dumps(step, sort_keys=True) + "\n" for step in steps),
            encoding="utf-8",
        )
        summary = {
            "schema_version": "selected-formal-ppo-candidate-multihorizon-shadow-rollout-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "shadow_trainable_transition_count": len(steps),
            "unique_trainable_context_count": len({step["context_id"] for step in steps}),
            "controlled_regression_count": 0 if not extra_step else 1,
            "controlled_safety_regression_count": 0,
            "controlled_contract_regression_count": 0,
            "controlled_path_risk_regression_count": 0 if not extra_step else 1,
            "controlled_source_selection_regression_count": 0,
            "teacher_agreement_rate": 1.0,
            "runs_multihorizon_shadow_rollout": True,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "readiness_status": "selected_formal_ppo_candidate_multihorizon_shadow_rollout_evaluated",
            "steps": str(steps_path),
            "git_provenance": {"current_matches_sources": True},
        }
        (self.multihorizon_root / "multihorizon-shadow-rollout-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _shadow_step(
        self,
        index: int,
        *,
        gate_reason_codes: list[str] | None = None,
        controlled_regression_reason_codes: list[str] | None = None,
        path_cost_delta: float = 0.0,
    ) -> dict:
        return {
            "schema_version": "selected-formal-ppo-candidate-multihorizon-shadow-step/v1",
            "context_id": f"context-{index:04d}",
            "scenario_id": f"scenario-{index % 5}",
            "scenario_family": f"family-{index % 5}",
            "split": "validation" if index % 2 == 0 else "test",
            "controlled_choice_source": "policy",
            "raw_policy_action_index": 0,
            "controlled_action_index": 0,
            "teacher_action_index": 0,
            "gate_reason_codes": gate_reason_codes or [],
            "controlled_regression_reason_codes": controlled_regression_reason_codes or [],
            "observation": {"action_mask": [True], "candidate_cells": [[1, 2]]},
            "log_prob": -0.1,
            "value": 0.2,
            "reward": 1.0,
            "shadow_discounted_return": 1.0,
            "shadow_advantage": 0.8,
            "path_cost_delta": path_cost_delta,
            "risk_delta": 0.0,
        }

    def _passing_readiness(self, **_kwargs) -> dict:
        return {
            "training_readiness_status": "guarded_experimental_policy_shadow_release_trial_evaluated",
            "training_blockers": [],
            "reason_codes": [],
        }


if __name__ == "__main__":
    unittest.main()
