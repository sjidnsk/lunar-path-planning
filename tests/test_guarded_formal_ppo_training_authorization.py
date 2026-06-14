import json
import shutil
import tempfile
import unittest
from pathlib import Path


class GuardedFormalPpoTrainingAuthorizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="guarded-formal-ppo-authorization-"))
        self.batch_root = self.temp_dir / "batch"
        self.output_root = self.temp_dir / "authorization"
        self.formal_preflight_root = self.temp_dir / "formal-preflight"
        self.formal_rollout_canary_root = self.temp_dir / "formal-rollout-canary"
        self.formal_stability_holdout_root = self.temp_dir / "formal-stability"
        self.candidate_selection_root = self.temp_dir / "candidate-selection"
        self.promotion_decision_review_root = self.temp_dir / "promotion-decision"
        self.canary_preflight_root = self.temp_dir / "canary-preflight"
        for path in (
            self.batch_root,
            self.output_root,
            self.formal_preflight_root,
            self.formal_rollout_canary_root,
            self.formal_stability_holdout_root,
            self.candidate_selection_root,
            self.promotion_decision_review_root,
            self.canary_preflight_root,
        ):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_authorizes_clean_formal_training_inputs_and_writes_manifests(self) -> None:
        from scripts.run_guarded_formal_ppo_training_authorization import (
            run_guarded_formal_ppo_training_authorization,
        )

        self._write_all_source_summaries()

        result = run_guarded_formal_ppo_training_authorization(
            formal_preflight_root=self.formal_preflight_root,
            formal_rollout_canary_root=self.formal_rollout_canary_root,
            formal_stability_holdout_root=self.formal_stability_holdout_root,
            candidate_selection_root=self.candidate_selection_root,
            promotion_decision_review_root=self.promotion_decision_review_root,
            canary_preflight_root=self.canary_preflight_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(
            result["schema_version"],
            "guarded-formal-ppo-training-authorization-summary/v1",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(
            result["authorization_verdict"],
            "authorized_for_guarded_formal_ppo_training_run",
        )
        self.assertEqual(result["authorized_trainable_transition_count"], 684)
        self.assertEqual(result["authorized_optimizer_train_transition_count"], 684)
        self.assertEqual(result["unique_authorized_trainable_context_count"], 684)
        self.assertEqual(result["source_summary_count"], 6)
        self.assertEqual(result["passed_source_summary_count"], 6)
        self.assertEqual(result["source_git_provenance_present_count"], 6)
        self.assertEqual(result["source_git_provenance_mismatch_count"], 0)
        self.assertEqual(result["validation_trainable_count"], 0)
        self.assertEqual(result["test_trainable_count"], 0)
        self.assertEqual(result["fallback_trainable_count"], 0)
        self.assertEqual(result["diagnostic_trainable_count"], 0)
        self.assertEqual(result["missing_observation_count"], 0)
        self.assertEqual(result["missing_log_prob_count"], 0)
        self.assertEqual(result["missing_value_count"], 0)
        self.assertEqual(result["invalid_action_mask_count"], 0)
        self.assertEqual(result["non_finite_reward_count"], 0)
        self.assertEqual(result["non_finite_return_count"], 0)
        self.assertEqual(result["non_finite_advantage_count"], 0)
        self.assertEqual(result["controlled_regression_count"], 0)
        self.assertEqual(result["controlled_safety_regression_count"], 0)
        self.assertEqual(result["controlled_contract_regression_count"], 0)
        self.assertEqual(result["controlled_path_risk_regression_count"], 0)
        self.assertEqual(result["controlled_source_selection_regression_count"], 0)
        self.assertEqual(result["seed_count"], 5)
        self.assertEqual(result["seeds"], [0, 1, 2, 3, 4])
        self.assertTrue(result["budget_manifest_passed"])
        self.assertTrue(result["seed_plan_passed"])
        self.assertTrue(result["stop_condition_manifest_passed"])
        self.assertTrue(result["rollback_manifest_passed"])
        self.assertTrue(result["post_training_gate_plan_passed"])
        self.assertFalse(result["runs_new_ppo_update"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])
        self.assertEqual(result["readiness_status"], "guarded_formal_ppo_training_authorized")

        for filename in (
            "formal-ppo-training-input-audit.json",
            "formal-ppo-training-budget-manifest.json",
            "formal-ppo-training-seed-plan.json",
            "formal-ppo-training-stop-condition-manifest.json",
            "formal-ppo-training-rollback-manifest.json",
            "formal-ppo-post-training-gate-plan.json",
            "formal-ppo-training-authorization-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_blocks_authorization_for_split_fallback_or_diagnostic_leakage(self) -> None:
        from scripts.run_guarded_formal_ppo_training_authorization import (
            run_guarded_formal_ppo_training_authorization,
        )

        self._write_all_source_summaries(
            stability_overrides={
                "validation_trainable_count": 1,
                "fallback_trainable_count": 1,
                "non_empty_gate_reason_trainable_count": 1,
            }
        )

        result = run_guarded_formal_ppo_training_authorization(
            formal_preflight_root=self.formal_preflight_root,
            formal_rollout_canary_root=self.formal_rollout_canary_root,
            formal_stability_holdout_root=self.formal_stability_holdout_root,
            candidate_selection_root=self.candidate_selection_root,
            promotion_decision_review_root=self.promotion_decision_review_root,
            canary_preflight_root=self.canary_preflight_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("guarded_formal_ppo_training_authorization_split_leakage", result["reason_codes"])
        self.assertIn("guarded_formal_ppo_training_authorization_fallback_trainable", result["reason_codes"])
        self.assertIn("guarded_formal_ppo_training_authorization_gate_reason_trainable", result["reason_codes"])

    def test_blocks_authorization_for_non_finite_or_controlled_regression(self) -> None:
        from scripts.run_guarded_formal_ppo_training_authorization import (
            run_guarded_formal_ppo_training_authorization,
        )

        self._write_all_source_summaries(
            stability_overrides={
                "missing_log_prob_count": 1,
                "non_finite_return_count": 1,
                "controlled_path_risk_regression_count": 1,
            }
        )

        result = run_guarded_formal_ppo_training_authorization(
            formal_preflight_root=self.formal_preflight_root,
            formal_rollout_canary_root=self.formal_rollout_canary_root,
            formal_stability_holdout_root=self.formal_stability_holdout_root,
            candidate_selection_root=self.candidate_selection_root,
            promotion_decision_review_root=self.promotion_decision_review_root,
            canary_preflight_root=self.canary_preflight_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("guarded_formal_ppo_training_authorization_contract_invalid", result["reason_codes"])
        self.assertIn("guarded_formal_ppo_training_authorization_non_finite", result["reason_codes"])
        self.assertIn("guarded_formal_ppo_training_authorization_controlled_regression", result["reason_codes"])

    def test_blocks_authorization_for_failed_or_stale_source_summary(self) -> None:
        from scripts.run_guarded_formal_ppo_training_authorization import (
            run_guarded_formal_ppo_training_authorization,
        )

        self._write_all_source_summaries(
            rollout_overrides={
                "status": "failed",
                "reason_codes": ["rollout_failed"],
                "git_provenance": {"current_matches_sources": False},
            }
        )

        result = run_guarded_formal_ppo_training_authorization(
            formal_preflight_root=self.formal_preflight_root,
            formal_rollout_canary_root=self.formal_rollout_canary_root,
            formal_stability_holdout_root=self.formal_stability_holdout_root,
            candidate_selection_root=self.candidate_selection_root,
            promotion_decision_review_root=self.promotion_decision_review_root,
            canary_preflight_root=self.canary_preflight_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("guarded_formal_ppo_training_authorization_source_not_passed", result["reason_codes"])
        self.assertIn(
            "guarded_formal_ppo_training_authorization_source_git_provenance_mismatch",
            result["reason_codes"],
        )

    def test_config_declares_outputs_docs_and_non_goals(self) -> None:
        config_path = self.repo_root / "configs" / "guarded_formal_ppo_training_authorization_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["schema_version"],
            "guarded-formal-ppo-training-authorization-config/v1",
        )
        self.assertEqual(config["validation"]["min_authorized_trainable_transition_count"], 684)
        self.assertEqual(config["training_plan"]["seeds"], [0, 1, 2, 3, 4])
        self.assertIn("formal-ppo-training-authorization-summary.json", config["output_files"].values())
        self.assertIn("README.md", config["documentation_updates"])
        self.assertIn("docs/算法设计与系统架构报告.md", config["documentation_updates"])
        self.assertIn(
            "docs/superpowers/specs/2026-06-14-guarded-formal-ppo-training-authorization.md",
            config["documentation_updates"],
        )
        self.assertIn("does_not_run_new_ppo_update", config["non_goals"])
        self.assertIn("does_not_publish_checkpoint", config["non_goals"])
        self.assertIn("does_not_replace_default_policy", config["non_goals"])

    def _write_all_source_summaries(
        self,
        *,
        preflight_overrides: dict | None = None,
        rollout_overrides: dict | None = None,
        stability_overrides: dict | None = None,
        candidate_overrides: dict | None = None,
        promotion_overrides: dict | None = None,
        canary_overrides: dict | None = None,
    ) -> None:
        self._write_json(
            self.formal_preflight_root / "quasi-real-guarded-formal-ppo-preflight-summary.json",
            self._summary(
                "quasi-real-guarded-formal-ppo-preflight-summary/v1",
                "quasi_real_guarded_formal_ppo_preflight_evaluated",
                overrides=preflight_overrides,
            ),
        )
        self._write_json(
            self.formal_rollout_canary_root / "quasi-real-guarded-formal-ppo-rollout-canary-summary.json",
            self._summary(
                "quasi-real-guarded-formal-ppo-rollout-canary-summary/v1",
                "quasi_real_guarded_formal_ppo_rollout_canary_evaluated",
                overrides=rollout_overrides,
            ),
        )
        self._write_json(
            self.formal_stability_holdout_root
            / "quasi-real-guarded-formal-ppo-stability-holdout-validation-summary.json",
            self._formal_stability_summary(overrides=stability_overrides),
        )
        self._write_json(
            self.candidate_selection_root
            / "quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-summary.json",
            self._summary(
                "quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-summary/v1",
                "quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_evaluated",
                overrides={
                    "input_trainable_transition_count": 684,
                    "long_horizon_trainable_transition_count": 684,
                    "optimizer_train_transition_count": 0,
                    "unique_trainable_context_count": 684,
                    "eligible_candidate_count": 30,
                    "horizon": 10,
                    "runs_formal_ppo_candidate_selection_long_horizon_holdout": True,
                    **(candidate_overrides or {}),
                },
            ),
        )
        self._write_json(
            self.promotion_decision_review_root
            / "selected-formal-ppo-candidate-promotion-decision-review-summary.json",
            self._summary(
                "selected-formal-ppo-candidate-promotion-decision-review-summary/v1",
                "selected_formal_ppo_candidate_promotion_decision_review_evaluated",
                overrides={
                    "decision_verdict": "eligible_for_guarded_release_candidate_packaging",
                    "selected_seed": 0,
                    "selected_budget": "epochs1_lr3e-6",
                    "selected_candidate_root": "outputs/selected-candidate",
                    **(promotion_overrides or {}),
                },
            ),
        )
        self._write_json(
            self.canary_preflight_root
            / "guarded-experimental-policy-staged-release-canary-preflight-summary.json",
            self._summary(
                "guarded-experimental-policy-staged-release-canary-preflight-summary/v1",
                "guarded_experimental_policy_staged_release_canary_preflight_evaluated",
                overrides={
                    "staged_release_canary_preflight_verdict": (
                        "eligible_for_guarded_staged_release_canary_dry_run"
                    ),
                    "canary_eligible_activation_count": 16,
                    "controlled_regression_count": 0,
                    "runs_staged_release_canary_preflight": True,
                    "runs_online_canary": False,
                    "connects_real_executor": False,
                    **(canary_overrides or {}),
                },
            ),
        )

    def _formal_stability_summary(self, *, overrides: dict | None = None) -> dict:
        payload = self._summary(
            "quasi-real-guarded-formal-ppo-stability-holdout-validation-summary/v1",
            "quasi_real_guarded_formal_ppo_stability_holdout_validated",
            overrides={
                "input_trainable_transition_count": 684,
                "optimizer_train_transition_count": 684,
                "unique_trainable_context_count": 684,
                "seed_count": 5,
                "budget_count": 6,
                "run_count": 30,
                "passed_run_count": 30,
                "validation_trainable_count": 0,
                "test_trainable_count": 0,
                "fallback_trainable_count": 0,
                "source_fallback_trainable_count": 0,
                "teacher_fallback_trainable_count": 0,
                "non_empty_gate_reason_trainable_count": 0,
                "missing_observation_count": 0,
                "missing_log_prob_count": 0,
                "missing_value_count": 0,
                "invalid_action_mask_count": 0,
                "non_finite_reward_count": 0,
                "non_finite_return_count": 0,
                "non_finite_advantage_count": 0,
                "loss_non_finite_count": 0,
                "non_finite_gradient_count": 0,
                "teacher_agreement_rate": 1.0,
                "controlled_regression_count": 0,
                "controlled_safety_regression_count": 0,
                "controlled_contract_regression_count": 0,
                "controlled_path_risk_regression_count": 0,
                "controlled_source_selection_regression_count": 0,
                "runs_formal_ppo_stability_holdout_validation": True,
                **(overrides or {}),
            },
        )
        return payload

    def _summary(self, schema_version: str, readiness_status: str, *, overrides: dict | None = None) -> dict:
        payload = {
            "schema_version": schema_version,
            "status": "passed",
            "reason_codes": [],
            "readiness_status": readiness_status,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "git_provenance": {"current_matches_sources": True},
        }
        payload.update(overrides or {})
        return payload

    def _config(self) -> dict:
        return {
            "schema_version": "guarded-formal-ppo-training-authorization-config/v1",
            "validation": {
                "min_authorized_trainable_transition_count": 684,
                "min_seed_count": 5,
            },
            "training_plan": {
                "seeds": [0, 1, 2, 3, 4],
                "epochs": 1,
                "learning_rate_max": 0.00001,
                "clip_ratio": 0.2,
                "max_grad_norm": 1.0,
                "checkpoint_scope": "experimental_candidate_only",
            },
            "readiness": {
                "config": "configs/policy_training_readiness_review_v1.json",
                "expected_status": "guarded_formal_ppo_training_authorized",
            },
            "output_files": {
                "summary": "formal-ppo-training-authorization-summary.json",
                "training_input_audit": "formal-ppo-training-input-audit.json",
                "budget_manifest": "formal-ppo-training-budget-manifest.json",
                "seed_plan": "formal-ppo-training-seed-plan.json",
                "stop_condition_manifest": "formal-ppo-training-stop-condition-manifest.json",
                "rollback_manifest": "formal-ppo-training-rollback-manifest.json",
                "post_training_gate_plan": "formal-ppo-post-training-gate-plan.json",
                "readiness_validate_only": "formal-ppo-training-authorization-readiness-validate-only.json",
                "report": "formal-ppo-training-authorization-report.md",
            },
        }

    def _passing_readiness(self, **_: object) -> dict:
        return {
            "training_readiness_status": "guarded_formal_ppo_training_authorized",
            "training_blockers": [],
            "reason_codes": [],
            "recommended_next_action": "guarded_formal_ppo_training_authorized",
        }

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
