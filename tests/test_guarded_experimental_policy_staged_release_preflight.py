import json
import shutil
import tempfile
import unittest
from pathlib import Path


class GuardedExperimentalPolicyStagedReleasePreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="guarded-staged-release-preflight-"))
        self.shadow_root = self.temp_dir / "shadow-release"
        self.output_root = self.temp_dir / "staged-preflight"
        self.batch_root = self.temp_dir / "batch"
        for path in (self.shadow_root, self.output_root, self.batch_root):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_runs_preflight_without_enabling_staged_release(self) -> None:
        from scripts.run_guarded_experimental_policy_staged_release_preflight import (
            run_guarded_experimental_policy_staged_release_preflight,
        )

        self._write_shadow_summary()

        result = run_guarded_experimental_policy_staged_release_preflight(
            shadow_release_trial_root=self.shadow_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(
            result["schema_version"],
            "guarded-experimental-policy-staged-release-preflight-summary/v1",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(
            result["staged_release_preflight_verdict"],
            "eligible_for_guarded_staged_release_trial",
        )
        self.assertFalse(result["staged_release_enabled"])
        self.assertTrue(result["default_policy_authoritative"])
        self.assertEqual(result["experimental_control_activation_count"], 0)
        self.assertEqual(result["shadow_step_count"], 700)
        self.assertEqual(result["unique_shadow_context_count"], 500)
        self.assertEqual(result["controlled_regression_count"], 0)
        self.assertTrue(result["gate_threshold_audit_passed"])
        self.assertTrue(result["kill_switch_audit_passed"])
        self.assertTrue(result["rollback_audit_passed"])
        self.assertTrue(result["telemetry_audit_passed"])
        self.assertFalse(result["runs_new_ppo_update"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])
        self.assertEqual(
            result["readiness_status"],
            "guarded_experimental_policy_staged_release_preflight_evaluated",
        )

        manifest = json.loads(Path(result["preflight_manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["stage_plan"][0]["stage"], "stage0_shadow_only")
        self.assertTrue(all(stage["executes_in_preflight"] is False for stage in manifest["stage_plan"]))

        for filename in (
            "guarded-experimental-policy-staged-release-preflight-summary.json",
            "staged-release-preflight-manifest.json",
            "staged-release-gate-threshold-audit.json",
            "staged-release-kill-switch-audit.json",
            "staged-release-rollback-audit.json",
            "staged-release-telemetry-audit.json",
            "staged-release-preflight-readiness-validate-only.json",
            "staged-release-preflight-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_shadow_regression_blocks_preflight_and_keeps_control_disabled(self) -> None:
        from scripts.run_guarded_experimental_policy_staged_release_preflight import (
            run_guarded_experimental_policy_staged_release_preflight,
        )

        payload = self._shadow_summary()
        payload["controlled_regression_count"] = 1
        payload["controlled_path_risk_regression_count"] = 1
        self._write_shadow_summary(payload)

        result = run_guarded_experimental_policy_staged_release_preflight(
            shadow_release_trial_root=self.shadow_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("staged_release_preflight_shadow_controlled_regression", result["reason_codes"])
        self.assertFalse(result["staged_release_enabled"])
        self.assertTrue(result["default_policy_authoritative"])
        self.assertEqual(result["experimental_control_activation_count"], 0)

    def test_config_declares_outputs_docs_and_non_goals(self) -> None:
        config_path = (
            self.repo_root
            / "configs"
            / "guarded_experimental_policy_staged_release_preflight_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["schema_version"],
            "guarded-experimental-policy-staged-release-preflight-config/v1",
        )
        self.assertEqual(config["validation"]["min_shadow_step_count"], 512)
        self.assertEqual(config["validation"]["min_unique_shadow_context_count"], 256)
        self.assertIn(
            "guarded-experimental-policy-staged-release-preflight-summary.json",
            config["output_files"].values(),
        )
        self.assertIn("README.md", config["documentation_updates"])
        self.assertIn("docs/算法设计与系统架构报告.md", config["documentation_updates"])
        self.assertIn("does_not_execute_actual_staged_release", config["non_goals"])
        self.assertIn("does_not_replace_default_policy", config["non_goals"])

    def _config(self) -> dict:
        return {
            "schema_version": "guarded-experimental-policy-staged-release-preflight-config/v1",
            "input_files": {
                "shadow_release_trial_summary": "guarded-experimental-policy-shadow-release-trial-summary.json",
            },
            "validation": {
                "min_shadow_step_count": 512,
                "min_unique_shadow_context_count": 256,
            },
            "readiness": {
                "config": "configs/policy_training_readiness_review_v1.json",
                "expected_status": "guarded_experimental_policy_staged_release_preflight_evaluated",
            },
            "output_files": {
                "summary": "guarded-experimental-policy-staged-release-preflight-summary.json",
                "preflight_manifest": "staged-release-preflight-manifest.json",
                "gate_threshold_audit": "staged-release-gate-threshold-audit.json",
                "kill_switch_audit": "staged-release-kill-switch-audit.json",
                "rollback_audit": "staged-release-rollback-audit.json",
                "telemetry_audit": "staged-release-telemetry-audit.json",
                "readiness_validate_only": "staged-release-preflight-readiness-validate-only.json",
                "report": "staged-release-preflight-report.md",
            },
        }

    def _write_shadow_summary(self, payload: dict | None = None) -> None:
        summary = payload or self._shadow_summary()
        (
            self.shadow_root
            / "guarded-experimental-policy-shadow-release-trial-summary.json"
        ).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    def _shadow_summary(self) -> dict:
        return {
            "schema_version": "guarded-experimental-policy-shadow-release-trial-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "shadow_release_trial_verdict": "eligible_for_guarded_staged_release_preflight",
            "shadow_step_count": 700,
            "unique_shadow_context_count": 500,
            "shadow_policy_takes_control": False,
            "missing_observation_count": 0,
            "missing_log_prob_count": 0,
            "missing_value_count": 0,
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
            "runtime_manifest": "shadow-release-runtime-manifest.json",
            "step_comparison": "shadow-release-step-comparison.jsonl",
            "rejection_report": "shadow-release-rejection-report.json",
            "risk_reward_audit": "shadow-release-risk-reward-audit.json",
            "runs_shadow_release_trial": True,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "readiness_status": "guarded_experimental_policy_shadow_release_trial_evaluated",
            "git_provenance": {"current_matches_sources": True},
        }

    def _passing_readiness(self, **_kwargs) -> dict:
        return {
            "training_readiness_status": "guarded_experimental_policy_staged_release_preflight_evaluated",
            "training_blockers": [],
            "reason_codes": [],
        }


if __name__ == "__main__":
    unittest.main()
