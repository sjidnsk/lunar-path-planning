import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path


class GuardedExperimentalPolicyStagedReleaseTrialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="guarded-staged-release-trial-"))
        self.preflight_root = self.temp_dir / "preflight"
        self.shadow_root = self.temp_dir / "shadow"
        self.output_root = self.temp_dir / "trial"
        self.batch_root = self.temp_dir / "batch"
        for path in (self.preflight_root, self.shadow_root, self.output_root, self.batch_root):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_trial_activates_only_gate_clean_steps_and_closes_drills(self) -> None:
        from scripts.run_guarded_experimental_policy_staged_release_trial import (
            run_guarded_experimental_policy_staged_release_trial,
        )

        self._write_preflight_summary()
        self._write_shadow_steps(
            [
                self._step("ctx-1", source="policy"),
                self._step("ctx-2", source="policy"),
                self._step("ctx-3", source="source_fallback"),
                self._step("ctx-4", source="policy", gate_reasons=["path_cost_regression"]),
                self._step("ctx-5", source="policy", missing_observation=True),
                self._step("ctx-6", source="policy", value=math.inf),
            ]
        )

        result = run_guarded_experimental_policy_staged_release_trial(
            preflight_root=self.preflight_root,
            shadow_release_trial_root=self.shadow_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(max_activation_count=2),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(
            result["schema_version"],
            "guarded-experimental-policy-staged-release-trial-summary/v1",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(
            result["staged_release_trial_verdict"],
            "eligible_for_guarded_staged_release_canary",
        )
        self.assertTrue(result["staged_release_enabled"])
        self.assertTrue(result["default_policy_authoritative"])
        self.assertEqual(result["experimental_control_activation_count"], 2)
        self.assertEqual(result["diagnostic_fallback_rejected_control_activation_count"], 0)
        self.assertEqual(result["controlled_regression_count"], 0)
        self.assertEqual(result["missing_observation_control_activation_count"], 0)
        self.assertEqual(result["non_finite_control_activation_count"], 0)
        self.assertTrue(result["kill_switch_drill_passed"])
        self.assertEqual(result["post_kill_switch_activation_count"], 0)
        self.assertTrue(result["rollback_drill_passed"])
        self.assertTrue(result["telemetry_drill_passed"])
        self.assertTrue(result["default_policy_unchanged"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])
        self.assertEqual(
            result["readiness_status"],
            "guarded_experimental_policy_staged_release_trial_evaluated",
        )

        ledger_rows = [
            json.loads(line)
            for line in (self.output_root / "staged-release-activation-ledger.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual([row["control_mode"] for row in ledger_rows], ["experimental", "experimental"])
        self.assertEqual({row["context_id"] for row in ledger_rows}, {"ctx-1", "ctx-2"})
        self.assertTrue((self.output_root / "staged-release-kill-switch-drill.json").is_file())
        self.assertTrue((self.output_root / "staged-release-rollback-drill.json").is_file())
        self.assertTrue((self.output_root / "staged-release-telemetry-drill.json").is_file())
        self.assertTrue((self.output_root / "staged-release-trial-report.md").is_file())

    def test_controlled_regression_blocks_activation_and_trial(self) -> None:
        from scripts.run_guarded_experimental_policy_staged_release_trial import (
            run_guarded_experimental_policy_staged_release_trial,
        )

        self._write_preflight_summary()
        self._write_shadow_steps(
            [
                self._step(
                    "ctx-regression",
                    source="policy",
                    controlled_reasons=["path_cost_regression"],
                )
            ]
        )

        result = run_guarded_experimental_policy_staged_release_trial(
            preflight_root=self.preflight_root,
            shadow_release_trial_root=self.shadow_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(max_activation_count=4),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("staged_release_trial_controlled_regression", result["reason_codes"])
        self.assertEqual(result["experimental_control_activation_count"], 0)
        self.assertEqual(result["controlled_regression_count"], 1)

    def test_config_declares_outputs_docs_and_non_goals(self) -> None:
        config_path = (
            self.repo_root
            / "configs"
            / "guarded_experimental_policy_staged_release_trial_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["schema_version"],
            "guarded-experimental-policy-staged-release-trial-config/v1",
        )
        self.assertEqual(config["validation"]["max_experimental_control_activation_count"], 64)
        self.assertIn(
            "guarded-experimental-policy-staged-release-trial-summary.json",
            config["output_files"].values(),
        )
        self.assertIn("README.md", config["documentation_updates"])
        self.assertIn("docs/算法设计与系统架构报告.md", config["documentation_updates"])
        self.assertIn("does_not_connect_real_executor", config["non_goals"])
        self.assertIn("does_not_replace_default_policy", config["non_goals"])

    def _config(self, *, max_activation_count: int = 64) -> dict:
        return {
            "schema_version": "guarded-experimental-policy-staged-release-trial-config/v1",
            "input_files": {
                "preflight_summary": "guarded-experimental-policy-staged-release-preflight-summary.json",
                "shadow_step_comparison": "shadow-release-step-comparison.jsonl",
            },
            "validation": {
                "max_experimental_control_activation_count": max_activation_count,
            },
            "readiness": {
                "config": "configs/policy_training_readiness_review_v1.json",
                "expected_status": "guarded_experimental_policy_staged_release_trial_evaluated",
            },
            "output_files": {
                "summary": "guarded-experimental-policy-staged-release-trial-summary.json",
                "activation_ledger": "staged-release-activation-ledger.jsonl",
                "controlled_regression_audit": "staged-release-controlled-regression-audit.json",
                "fallback_rejection_report": "staged-release-fallback-rejection-report.json",
                "kill_switch_drill": "staged-release-kill-switch-drill.json",
                "rollback_drill": "staged-release-rollback-drill.json",
                "telemetry_drill": "staged-release-telemetry-drill.json",
                "readiness_validate_only": "staged-release-trial-readiness-validate-only.json",
                "report": "staged-release-trial-report.md",
            },
        }

    def _write_preflight_summary(self, payload: dict | None = None) -> None:
        summary = payload or self._preflight_summary()
        (self.preflight_root / "guarded-experimental-policy-staged-release-preflight-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_shadow_steps(self, rows: list[dict]) -> None:
        (self.shadow_root / "shadow-release-step-comparison.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _preflight_summary(self) -> dict:
        return {
            "schema_version": "guarded-experimental-policy-staged-release-preflight-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "staged_release_preflight_verdict": "eligible_for_guarded_staged_release_trial",
            "source_shadow_release_trial_summary": str(
                self.shadow_root / "guarded-experimental-policy-shadow-release-trial-summary.json"
            ),
            "shadow_step_count": 6,
            "unique_shadow_context_count": 6,
            "staged_release_enabled": False,
            "default_policy_authoritative": True,
            "experimental_control_activation_count": 0,
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
            "gate_threshold_audit_passed": True,
            "kill_switch_audit_passed": True,
            "rollback_audit_passed": True,
            "telemetry_audit_passed": True,
            "runs_staged_release_preflight": True,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "readiness_status": "guarded_experimental_policy_staged_release_preflight_evaluated",
            "git_provenance": {"current_matches_sources": True},
        }

    def _step(
        self,
        context_id: str,
        *,
        source: str,
        gate_reasons: list[str] | None = None,
        controlled_reasons: list[str] | None = None,
        missing_observation: bool = False,
        invalid_action_mask: bool = False,
        log_prob: float = -0.1,
        value: float = 0.2,
        reward: float = 1.0,
    ) -> dict:
        return {
            "schema_version": "guarded-experimental-policy-shadow-release-trial-step/v1",
            "context_id": context_id,
            "scenario_id": f"scenario-{context_id}",
            "scenario_family": "unit",
            "split": "train",
            "default_policy_action_index": 1,
            "experimental_shadow_action_index": 1,
            "controlled_choice_source": source,
            "shadow_policy_takes_control": False,
            "shadow_diagnostic_only": source != "policy" or bool(gate_reasons),
            "shadow_gate_reason_codes": gate_reasons or [],
            "controlled_regression_reason_codes": controlled_reasons or [],
            "missing_observation": missing_observation,
            "invalid_action_mask": invalid_action_mask,
            "non_finite_logits": False,
            "non_finite_log_prob": not math.isfinite(log_prob),
            "non_finite_value": not math.isfinite(value),
            "non_finite_reward": not math.isfinite(reward),
            "log_prob": log_prob,
            "value": value,
            "reward": reward,
            "path_cost_delta": 0.0,
            "risk_delta": 0.0,
        }

    def _passing_readiness(self, **_kwargs) -> dict:
        return {
            "training_readiness_status": "guarded_experimental_policy_staged_release_trial_evaluated",
            "training_blockers": [],
            "reason_codes": [],
        }


if __name__ == "__main__":
    unittest.main()
