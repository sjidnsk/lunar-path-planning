import json
import tempfile
import unittest
from pathlib import Path


class QuasiRealGuardedFormalPpoStabilityHoldoutValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="qreal-formal-stability-"))
        self.canary_root = self.temp_dir / "formal-canary"
        self.output_root = self.temp_dir / "formal-stability"
        self.batch_root = self.temp_dir / "batch"
        self.canary_root.mkdir(parents=True)
        self.batch_root.mkdir(parents=True)

    def test_stability_holdout_passes_from_canary_with_seed_budget_matrix(self) -> None:
        from scripts.run_quasi_real_guarded_formal_ppo_stability_holdout_validation import (
            run_quasi_real_guarded_formal_ppo_stability_holdout_validation,
        )

        self._write_canary_artifacts(trainable_count=4)

        result = run_quasi_real_guarded_formal_ppo_stability_holdout_validation(
            canary_root=self.canary_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(expected_trainable=4),
            repo_root=self.repo_root,
            run_holdout_runner=self._passing_holdout_run,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(
            result["schema_version"],
            "quasi-real-guarded-formal-ppo-stability-holdout-validation-summary/v1",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(result["input_trainable_transition_count"], 4)
        self.assertEqual(result["optimizer_train_transition_count"], 4)
        self.assertEqual(result["unique_trainable_context_count"], 4)
        self.assertEqual(result["seed_count"], 2)
        self.assertEqual(result["budget_count"], 2)
        self.assertEqual(result["run_count"], 4)
        self.assertEqual(result["passed_run_count"], 4)
        self.assertEqual(result["max_old_log_prob_abs_error"], 0.0)
        self.assertEqual(result["max_old_value_abs_error"], 0.0)
        self.assertEqual(result["loss_non_finite_count"], 0)
        self.assertEqual(result["non_finite_gradient_count"], 0)
        self.assertEqual(result["non_finite_reward_count"], 0)
        self.assertEqual(result["non_finite_return_count"], 0)
        self.assertEqual(result["non_finite_advantage_count"], 0)
        self.assertGreater(result["min_parameter_l2_delta"], 0.0)
        self.assertLessEqual(result["max_abs_approx_kl"], 0.25)
        self.assertLessEqual(result["max_grad_norm_after_clip"], 1.0)
        self.assertEqual(result["validation_trainable_count"], 0)
        self.assertEqual(result["test_trainable_count"], 0)
        self.assertEqual(result["fallback_trainable_count"], 0)
        self.assertEqual(result["non_empty_gate_reason_trainable_count"], 0)
        self.assertEqual(result["controlled_regression_count"], 0)
        self.assertEqual(result["validation_controlled_regression_count"], 0)
        self.assertEqual(result["test_controlled_regression_count"], 0)
        self.assertEqual(result["family_regression_count"], 0)
        self.assertEqual(result["teacher_agreement_rate"], 1.0)
        self.assertEqual(
            result["readiness_status"],
            "quasi_real_guarded_formal_ppo_stability_holdout_validated",
        )
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])

        for filename in (
            "formal-ppo-stability-baseline-manifest.json",
            "formal-ppo-stability-matrix.jsonl",
            "formal-ppo-stability-training-curves.json",
            "formal-ppo-stability-holdout-audit.json",
            "formal-ppo-stability-family-regression-report.json",
            "formal-ppo-stability-rollback-manifest.json",
            "formal-ppo-stability-readiness-validate-only.json",
            "formal-ppo-stability-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_stability_holdout_rejects_diagnostic_or_fallback_trainable_leakage(self) -> None:
        from scripts.run_quasi_real_guarded_formal_ppo_stability_holdout_validation import (
            run_quasi_real_guarded_formal_ppo_stability_holdout_validation,
        )

        self._write_canary_artifacts(
            trainable_count=4,
            extra_steps=[
                self._step(100, split="validation", ppo_trainable=True),
                self._step(101, controlled_choice_source="source_fallback", ppo_trainable=True),
                self._step(102, gate_reason_codes=["risk_regression"], ppo_trainable=True),
            ],
        )

        result = run_quasi_real_guarded_formal_ppo_stability_holdout_validation(
            canary_root=self.canary_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(expected_trainable=4),
            repo_root=self.repo_root,
            run_holdout_runner=self._passing_holdout_run,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("formal_ppo_stability_holdout_split_leakage", result["reason_codes"])
        self.assertIn("formal_ppo_stability_holdout_fallback_trainable", result["reason_codes"])
        self.assertIn("formal_ppo_stability_holdout_gate_reason_trainable", result["reason_codes"])
        self.assertEqual(result["validation_trainable_count"], 1)
        self.assertEqual(result["fallback_trainable_count"], 1)
        self.assertEqual(result["non_empty_gate_reason_trainable_count"], 1)
        self.assertEqual(result["run_count"], 0)

    def test_stability_holdout_fails_when_any_budget_run_regresses(self) -> None:
        from scripts.run_quasi_real_guarded_formal_ppo_stability_holdout_validation import (
            run_quasi_real_guarded_formal_ppo_stability_holdout_validation,
        )

        self._write_canary_artifacts(trainable_count=4)

        def runner(*, seed: int, budget: dict, **kwargs) -> dict:
            summary = self._passing_holdout_run(seed=seed, budget=budget, **kwargs)
            if seed == 1 and budget["epochs"] == 2:
                summary["status"] = "failed"
                summary["reason_codes"] = ["validation_controlled_regression"]
                summary["validation_controlled_regression_count"] = 1
                summary["controlled_regression_count"] = 1
            return summary

        result = run_quasi_real_guarded_formal_ppo_stability_holdout_validation(
            canary_root=self.canary_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(expected_trainable=4),
            repo_root=self.repo_root,
            run_holdout_runner=runner,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("formal_ppo_stability_holdout_run_not_all_passed", result["reason_codes"])
        self.assertIn("formal_ppo_stability_holdout_controlled_regression", result["reason_codes"])
        self.assertEqual(result["passed_run_count"], 3)
        self.assertEqual(result["validation_controlled_regression_count"], 1)

    def test_config_declares_stability_holdout_docs_matrix_and_non_goals(self) -> None:
        config_path = (
            self.repo_root
            / "configs"
            / "quasi_real_guarded_formal_ppo_stability_holdout_validation_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["schema_version"],
            "quasi-real-guarded-formal-ppo-stability-holdout-validation-config/v1",
        )
        self.assertEqual(config["validation"]["expected_trainable_transition_count"], 684)
        self.assertGreaterEqual(len(config["seeds"]), 5)
        epochs = {budget["epochs"] for budget in config["budgets"]}
        learning_rates = {budget["learning_rate"] for budget in config["budgets"]}
        self.assertEqual(epochs, {1, 2, 3})
        self.assertEqual(learning_rates, {3e-6, 1e-5})
        self.assertIn("README.md", config["documentation_updates"])
        self.assertIn("docs/算法设计与系统架构报告.md", config["documentation_updates"])
        self.assertIn("does_not_publish_checkpoint", config["non_goals"])
        self.assertIn("does_not_claim_formal_training_ready", config["non_goals"])

    def _config(self, *, expected_trainable: int) -> dict:
        return {
            "schema_version": "quasi-real-guarded-formal-ppo-stability-holdout-validation-config/v1",
            "seeds": [0, 1],
            "budgets": [
                {"name": "epoch1_lr3e-6", "epochs": 1, "learning_rate": 3e-6},
                {"name": "epoch2_lr1e-5", "epochs": 2, "learning_rate": 1e-5},
            ],
            "training": {
                "clip_ratio": 0.2,
                "max_grad_norm": 1.0,
                "discount_factor": 0.99,
                "device": "cpu",
            },
            "validation": {
                "expected_trainable_transition_count": expected_trainable,
                "min_seed_count": 2,
                "min_budget_count": 2,
                "max_old_log_prob_abs_error": 1.0e-4,
                "max_old_value_abs_error": 1.0e-4,
                "max_abs_approx_kl": 0.25,
                "max_grad_norm_after_clip": 1.0,
                "min_teacher_agreement_rate": 0.95,
            },
            "readiness": {
                "config": "configs/policy_training_readiness_review_v1.json",
                "expected_status": "quasi_real_guarded_formal_ppo_stability_holdout_validated",
            },
            "output_files": {
                "summary": "quasi-real-guarded-formal-ppo-stability-holdout-validation-summary.json",
                "baseline_manifest": "formal-ppo-stability-baseline-manifest.json",
                "stability_matrix": "formal-ppo-stability-matrix.jsonl",
                "training_curves": "formal-ppo-stability-training-curves.json",
                "holdout_audit": "formal-ppo-stability-holdout-audit.json",
                "family_regression_report": "formal-ppo-stability-family-regression-report.json",
                "rollback_manifest": "formal-ppo-stability-rollback-manifest.json",
                "readiness_validate_only": "formal-ppo-stability-readiness-validate-only.json",
                "report": "formal-ppo-stability-report.md",
            },
        }

    def _write_canary_artifacts(
        self,
        *,
        trainable_count: int,
        extra_steps: list[dict] | None = None,
    ) -> None:
        steps_path = self.canary_root / "canary-steps.jsonl"
        steps = [self._step(index) for index in range(trainable_count)]
        steps.extend(extra_steps or [])
        steps_path.write_text(
            "".join(json.dumps(step, sort_keys=True) + "\n" for step in steps),
            encoding="utf-8",
        )
        seed_summaries_path = self.canary_root / "formal-rollout-canary-seed-summaries.jsonl"
        seed_summaries = [
            {
                "schema_version": "quasi-real-guarded-formal-ppo-rollout-canary-seed-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "seed": seed,
                "optimizer_train_transition_count": trainable_count,
                "controlled_regression_count": 0,
                "teacher_agreement_rate": 1.0,
            }
            for seed in (0, 1, 2)
        ]
        seed_summaries_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in seed_summaries),
            encoding="utf-8",
        )
        progress_path = self.canary_root / "formal-rollout-canary-progress.jsonl"
        progress_path.write_text(
            "".join(
                json.dumps({"event": event, "seed": seed}, sort_keys=True) + "\n"
                for seed in (0, 1, 2)
                for event in ("seed_started", "seed_finished")
            ),
            encoding="utf-8",
        )
        rollback_path = self.canary_root / "formal-rollout-canary-rollback-manifest.json"
        rollback_path.write_text(
            json.dumps(
                {
                    "schema_version": "quasi-real-guarded-formal-ppo-rollout-canary-rollback-manifest/v1",
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary = {
            "schema_version": "quasi-real-guarded-formal-ppo-rollout-canary-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "input_trainable_transition_count": trainable_count,
            "optimizer_train_transition_count": trainable_count,
            "unique_trainable_context_count": trainable_count,
            "validation_trainable_count": 0,
            "test_trainable_count": 0,
            "fallback_trainable_count": 0,
            "source_fallback_trainable_count": 0,
            "teacher_fallback_trainable_count": 0,
            "non_empty_gate_reason_trainable_count": 0,
            "missing_observation_count": 0,
            "missing_log_prob_count": 0,
            "missing_value_count": 0,
            "non_finite_reward_count": 0,
            "non_finite_return_count": 0,
            "non_finite_advantage_count": 0,
            "loss_non_finite_count": 0,
            "non_finite_gradient_count": 0,
            "seed_count": 3,
            "passed_seed_count": 3,
            "max_old_log_prob_abs_error": 0.0,
            "max_old_value_abs_error": 0.0,
            "max_abs_approx_kl": 0.01,
            "max_grad_norm_after_clip": 0.5,
            "min_parameter_l2_delta": 0.001,
            "teacher_agreement_rate": 1.0,
            "controlled_regression_count": 0,
            "controlled_safety_regression_count": 0,
            "controlled_contract_regression_count": 0,
            "controlled_path_risk_regression_count": 0,
            "controlled_source_selection_regression_count": 0,
            "steps": str(steps_path),
            "seed_summaries": str(seed_summaries_path),
            "progress": str(progress_path),
            "rollback_manifest": str(rollback_path),
            "runs_formal_ppo_rollout_canary": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "git_provenance": {"current_matches_sources": True},
        }
        (self.canary_root / "quasi-real-guarded-formal-ppo-rollout-canary-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _step(
        self,
        index: int,
        *,
        split: str = "train",
        controlled_choice_source: str = "policy",
        ppo_trainable: bool = True,
        gate_reason_codes: list[str] | None = None,
    ) -> dict:
        return {
            "schema_version": "quasi-real-guarded-ppo-horizon5-batch-expansion-step/v1",
            "context_id": f"context-{index:04d}",
            "scenario_id": f"scenario-{index % 2}",
            "scenario_family": f"family-{index % 2}",
            "episode_id": f"episode-{index:04d}",
            "step_index": index % 5,
            "horizon": 5,
            "split": split,
            "ppo_trainable": ppo_trainable,
            "controlled_choice_source": controlled_choice_source,
            "controlled_action_index": 0,
            "gate_reason_codes": gate_reason_codes or [],
            "controlled_regression_reason_codes": [],
            "observation": {
                "candidate_feature_names": ["feature"],
                "candidate_features": [[1.0]],
                "global_feature_names": ["global"],
                "global_features": [0.0],
                "action_mask": [True],
                "candidate_cells": [[1, 2]],
                "candidate_missing_feature_names": [[]],
                "candidate_missing_indicator_names": [],
                "candidate_missing_indicators": [[]],
            },
            "log_prob": -0.1,
            "value": 0.2,
            "reward": 1.0,
            "discounted_return": 1.0,
            "advantage": 0.8,
        }

    def _passing_holdout_run(
        self,
        *,
        seed: int,
        budget: dict,
        trainable_steps: list[dict],
        **_kwargs,
    ) -> dict:
        return {
            "schema_version": "quasi-real-guarded-formal-ppo-stability-holdout-run-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "seed": seed,
            "budget_name": budget["name"],
            "epochs": budget["epochs"],
            "learning_rate": budget["learning_rate"],
            "optimizer_train_transition_count": len(trainable_steps),
            "post_update_guarded_collector_trainable_transition_count": len(trainable_steps),
            "old_log_prob_max_abs_error": 0.0,
            "old_value_max_abs_error": 0.0,
            "loss_non_finite_count": 0,
            "non_finite_gradient_count": 0,
            "non_finite_reward_count": 0,
            "non_finite_return_count": 0,
            "non_finite_advantage_count": 0,
            "parameter_l2_delta": 0.001,
            "approx_kl": 0.01,
            "max_grad_norm_after_clip": 1.0,
            "teacher_agreement_rate": 1.0,
            "controlled_regression_count": 0,
            "train_controlled_regression_count": 0,
            "validation_controlled_regression_count": 0,
            "test_controlled_regression_count": 0,
            "family_regression_count": 0,
            "controlled_safety_regression_count": 0,
            "controlled_contract_regression_count": 0,
            "controlled_path_risk_regression_count": 0,
            "controlled_source_selection_regression_count": 0,
            "training_curve_records": [
                {
                    "seed": seed,
                    "budget_name": budget["name"],
                    "epoch": 0,
                    "approx_kl": 0.01,
                }
            ],
            "holdout_splits_evaluated": ["train", "validation", "test"],
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
        }

    def _passing_readiness(self, **_kwargs) -> dict:
        return {
            "training_readiness_status": "quasi_real_guarded_formal_ppo_stability_holdout_validated",
            "training_blockers": [],
            "reason_codes": [],
        }


if __name__ == "__main__":
    unittest.main()
