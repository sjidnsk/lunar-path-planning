import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path


class GuardedExperimentalPolicyStagedReleaseCanaryPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="guarded-staged-canary-preflight-"))
        self.trial_root = self.temp_dir / "trial"
        self.output_root = self.temp_dir / "canary"
        self.batch_root = self.temp_dir / "batch"
        for path in (self.trial_root, self.output_root, self.batch_root):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_preflight_filters_gate_clean_trial_activations_and_closes_audits(self) -> None:
        from scripts.run_guarded_experimental_policy_staged_release_canary_preflight import (
            run_guarded_experimental_policy_staged_release_canary_preflight,
        )

        self._write_trial_summary()
        self._write_activation_ledger(
            [
                self._activation("ctx-1"),
                self._activation("ctx-2"),
                self._activation("ctx-3", controlled_choice_source="source_fallback"),
                self._activation("ctx-4", gate_reasons=["path_cost_regression"]),
                self._activation("ctx-5", missing_observation=True),
                self._activation("ctx-6", value=math.inf),
            ]
        )

        result = run_guarded_experimental_policy_staged_release_canary_preflight(
            staged_release_trial_root=self.trial_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(max_activation_count=2, traffic_fraction=0.01),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(
            result["schema_version"],
            "guarded-experimental-policy-staged-release-canary-preflight-summary/v1",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(
            result["staged_release_canary_preflight_verdict"],
            "eligible_for_guarded_staged_release_canary_dry_run",
        )
        self.assertFalse(result["staged_canary_enabled"])
        self.assertFalse(result["connects_real_executor"])
        self.assertTrue(result["default_policy_authoritative"])
        self.assertLessEqual(result["canary_traffic_fraction"], 0.01)
        self.assertEqual(result["canary_eligible_activation_count"], 2)
        self.assertEqual(result["max_canary_control_activation_count"], 2)
        self.assertEqual(result["diagnostic_fallback_rejected_canary_eligible_count"], 0)
        self.assertEqual(result["missing_observation_canary_eligible_count"], 0)
        self.assertEqual(result["non_finite_canary_eligible_count"], 0)
        self.assertEqual(result["controlled_regression_canary_eligible_count"], 0)
        self.assertEqual(result["controlled_regression_count"], 0)
        self.assertEqual(result["controlled_safety_regression_count"], 0)
        self.assertEqual(result["controlled_contract_regression_count"], 0)
        self.assertEqual(result["controlled_path_risk_regression_count"], 0)
        self.assertEqual(result["controlled_source_selection_regression_count"], 0)
        self.assertTrue(result["kill_switch_audit_passed"])
        self.assertTrue(result["rollback_audit_passed"])
        self.assertTrue(result["telemetry_audit_passed"])
        self.assertTrue(result["automatic_downgrade_audit_passed"])
        self.assertTrue(result["canary_budget_audit_passed"])
        self.assertTrue(result["operator_approval_audit_passed"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])
        self.assertEqual(
            result["readiness_status"],
            "guarded_experimental_policy_staged_release_canary_preflight_evaluated",
        )

        ledger_rows = [
            json.loads(line)
            for line in (self.output_root / "staged-canary-eligibility-ledger.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual({row["context_id"] for row in ledger_rows}, {"ctx-1", "ctx-2"})
        self.assertEqual(
            [row["canary_control_mode"] for row in ledger_rows],
            ["offline_dry_run_candidate", "offline_dry_run_candidate"],
        )
        self.assertTrue((self.output_root / "staged-canary-preflight-manifest.json").is_file())
        self.assertTrue((self.output_root / "staged-canary-automatic-downgrade-audit.json").is_file())
        self.assertTrue((self.output_root / "staged-canary-budget-audit.json").is_file())
        self.assertTrue((self.output_root / "staged-canary-operator-approval-audit.json").is_file())
        self.assertTrue((self.output_root / "staged-canary-preflight-report.md").is_file())

    def test_controlled_regression_in_trial_ledger_blocks_canary_preflight(self) -> None:
        from scripts.run_guarded_experimental_policy_staged_release_canary_preflight import (
            run_guarded_experimental_policy_staged_release_canary_preflight,
        )

        self._write_trial_summary()
        self._write_activation_ledger(
            [
                self._activation(
                    "ctx-regression",
                    controlled_reasons=["path_cost_regression"],
                )
            ]
        )

        result = run_guarded_experimental_policy_staged_release_canary_preflight(
            staged_release_trial_root=self.trial_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(max_activation_count=4, traffic_fraction=0.01),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("staged_release_canary_preflight_controlled_regression", result["reason_codes"])
        self.assertEqual(result["canary_eligible_activation_count"], 0)
        self.assertEqual(result["controlled_regression_count"], 1)

    def test_config_declares_outputs_docs_and_non_goals(self) -> None:
        config_path = (
            self.repo_root
            / "configs"
            / "guarded_experimental_policy_staged_release_canary_preflight_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["schema_version"],
            "guarded-experimental-policy-staged-release-canary-preflight-config/v1",
        )
        self.assertEqual(config["validation"]["max_canary_control_activation_count"], 16)
        self.assertLessEqual(config["validation"]["canary_traffic_fraction"], 0.01)
        self.assertIn(
            "guarded-experimental-policy-staged-release-canary-preflight-summary.json",
            config["output_files"].values(),
        )
        self.assertIn("README.md", config["documentation_updates"])
        self.assertIn("docs/算法设计与系统架构报告.md", config["documentation_updates"])
        self.assertIn("does_not_connect_real_executor", config["non_goals"])
        self.assertIn("does_not_launch_online_canary", config["non_goals"])
        self.assertIn("does_not_replace_default_policy", config["non_goals"])

    def _config(
        self,
        *,
        max_activation_count: int = 16,
        traffic_fraction: float = 0.01,
    ) -> dict:
        return {
            "schema_version": "guarded-experimental-policy-staged-release-canary-preflight-config/v1",
            "input_files": {
                "staged_release_trial_summary": "guarded-experimental-policy-staged-release-trial-summary.json",
                "activation_ledger": "staged-release-activation-ledger.jsonl",
            },
            "validation": {
                "max_canary_control_activation_count": max_activation_count,
                "canary_traffic_fraction": traffic_fraction,
            },
            "readiness": {
                "config": "configs/policy_training_readiness_review_v1.json",
                "expected_status": "guarded_experimental_policy_staged_release_canary_preflight_evaluated",
            },
            "output_files": {
                "summary": "guarded-experimental-policy-staged-release-canary-preflight-summary.json",
                "manifest": "staged-canary-preflight-manifest.json",
                "eligibility_ledger": "staged-canary-eligibility-ledger.jsonl",
                "rejection_report": "staged-canary-rejection-report.json",
                "kill_switch_audit": "staged-canary-kill-switch-audit.json",
                "rollback_audit": "staged-canary-rollback-audit.json",
                "telemetry_audit": "staged-canary-telemetry-audit.json",
                "automatic_downgrade_audit": "staged-canary-automatic-downgrade-audit.json",
                "budget_audit": "staged-canary-budget-audit.json",
                "operator_approval_audit": "staged-canary-operator-approval-audit.json",
                "readiness_validate_only": "staged-canary-preflight-readiness-validate-only.json",
                "report": "staged-canary-preflight-report.md",
            },
        }

    def _write_trial_summary(self, payload: dict | None = None) -> None:
        summary = payload or self._trial_summary()
        (self.trial_root / "guarded-experimental-policy-staged-release-trial-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_activation_ledger(self, rows: list[dict]) -> None:
        (self.trial_root / "staged-release-activation-ledger.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _trial_summary(self) -> dict:
        return {
            "schema_version": "guarded-experimental-policy-staged-release-trial-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "staged_release_trial_verdict": "eligible_for_guarded_staged_release_canary",
            "source_staged_release_preflight_summary": str(
                self.trial_root / "guarded-experimental-policy-staged-release-preflight-summary.json"
            ),
            "activation_ledger": str(self.trial_root / "staged-release-activation-ledger.jsonl"),
            "controlled_regression_audit": str(self.trial_root / "staged-release-controlled-regression-audit.json"),
            "fallback_rejection_report": str(self.trial_root / "staged-release-fallback-rejection-report.json"),
            "kill_switch_drill": str(self.trial_root / "staged-release-kill-switch-drill.json"),
            "rollback_drill": str(self.trial_root / "staged-release-rollback-drill.json"),
            "telemetry_drill": str(self.trial_root / "staged-release-telemetry-drill.json"),
            "shadow_step_count": 6,
            "unique_shadow_context_count": 6,
            "staged_release_enabled": True,
            "default_policy_authoritative": True,
            "experimental_control_activation_count": 6,
            "max_experimental_control_activation_count": 64,
            "diagnostic_fallback_rejected_control_activation_count": 0,
            "missing_observation_count": 0,
            "missing_log_prob_count": 0,
            "missing_value_count": 0,
            "invalid_action_mask_count": 0,
            "non_finite_logits_count": 0,
            "non_finite_log_prob_count": 0,
            "non_finite_value_count": 0,
            "non_finite_reward_count": 0,
            "missing_observation_control_activation_count": 0,
            "non_finite_control_activation_count": 0,
            "controlled_regression_count": 0,
            "controlled_safety_regression_count": 0,
            "controlled_contract_regression_count": 0,
            "controlled_path_risk_regression_count": 0,
            "controlled_source_selection_regression_count": 0,
            "kill_switch_drill_passed": True,
            "post_kill_switch_activation_count": 0,
            "rollback_drill_passed": True,
            "telemetry_drill_passed": True,
            "default_policy_unchanged": True,
            "runs_staged_release_trial": True,
            "connects_real_executor": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "readiness_status": "guarded_experimental_policy_staged_release_trial_evaluated",
            "git_provenance": {"current_matches_sources": True},
        }

    def _activation(
        self,
        context_id: str,
        *,
        controlled_choice_source: str = "policy",
        gate_reasons: list[str] | None = None,
        controlled_reasons: list[str] | None = None,
        missing_observation: bool = False,
        invalid_action_mask: bool = False,
        log_prob: float = -0.1,
        value: float = 0.2,
        reward: float = 1.0,
    ) -> dict:
        return {
            "schema_version": "guarded-experimental-policy-staged-release-trial-activation/v1",
            "activation_index": 0,
            "source_index": 0,
            "context_id": context_id,
            "scenario_id": f"scenario-{context_id}",
            "scenario_family": "unit",
            "split": "train",
            "control_mode": "experimental",
            "default_policy_action_index": 1,
            "experimental_action_index": 1,
            "controlled_choice_source": controlled_choice_source,
            "gate_reason_codes": gate_reasons or [],
            "controlled_regression_reason_codes": controlled_reasons or [],
            "missing_observation": missing_observation,
            "invalid_action_mask": invalid_action_mask,
            "log_prob": log_prob,
            "value": value,
            "reward": reward,
            "path_cost_delta": 0.0,
            "risk_delta": 0.0,
        }

    def _passing_readiness(self, **_kwargs) -> dict:
        return {
            "training_readiness_status": "guarded_experimental_policy_staged_release_canary_preflight_evaluated",
            "training_blockers": [],
            "reason_codes": [],
        }


if __name__ == "__main__":
    unittest.main()
