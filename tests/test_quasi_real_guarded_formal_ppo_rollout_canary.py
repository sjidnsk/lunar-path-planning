import json
import tempfile
import unittest
from pathlib import Path


class QuasiRealGuardedFormalPpoRolloutCanaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="qreal-formal-canary-"))
        self.preflight_root = self.temp_dir / "formal-preflight"
        self.output_root = self.temp_dir / "formal-canary"
        self.batch_root = self.temp_dir / "batch"
        self.preflight_root.mkdir(parents=True)
        self.batch_root.mkdir(parents=True)

    def test_canary_passes_from_formal_preflight_with_three_guarded_seeds(self) -> None:
        from scripts.run_quasi_real_guarded_formal_ppo_rollout_canary import (
            run_quasi_real_guarded_formal_ppo_rollout_canary,
        )

        self._write_preflight_artifacts(trainable_count=684)

        result = run_quasi_real_guarded_formal_ppo_rollout_canary(
            preflight_root=self.preflight_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            seed_canary_runner=self._passing_seed_canary,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(
            result["schema_version"],
            "quasi-real-guarded-formal-ppo-rollout-canary-summary/v1",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(result["input_trainable_transition_count"], 684)
        self.assertEqual(result["optimizer_train_transition_count"], 684)
        self.assertEqual(result["unique_trainable_context_count"], 684)
        self.assertEqual(result["validation_trainable_count"], 0)
        self.assertEqual(result["test_trainable_count"], 0)
        self.assertEqual(result["fallback_trainable_count"], 0)
        self.assertEqual(result["non_empty_gate_reason_trainable_count"], 0)
        self.assertEqual(result["seed_count"], 3)
        self.assertEqual(result["passed_seed_count"], 3)
        self.assertEqual(result["max_old_log_prob_abs_error"], 0.0)
        self.assertEqual(result["max_old_value_abs_error"], 0.0)
        self.assertEqual(result["controlled_regression_count"], 0)
        self.assertEqual(result["teacher_agreement_rate"], 1.0)
        self.assertEqual(
            result["readiness_status"],
            "quasi_real_guarded_formal_ppo_rollout_canary_evaluated",
        )
        self.assertTrue(result["runs_formal_ppo_rollout_canary"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])

        self.assertTrue((self.output_root / "formal-rollout-canary-seed-summaries.jsonl").is_file())
        self.assertTrue((self.output_root / "formal-rollout-canary-training-curves.json").is_file())
        self.assertTrue((self.output_root / "formal-rollout-canary-gate-audit.json").is_file())
        rollback = json.loads(
            (self.output_root / "formal-rollout-canary-rollback-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            rollback["schema_version"],
            "quasi-real-guarded-formal-ppo-rollout-canary-rollback-manifest/v1",
        )
        self.assertEqual(rollback["formal_preflight_summary"], str(self.preflight_root / "quasi-real-guarded-formal-ppo-preflight-summary.json"))
        self.assertFalse(rollback["publishes_checkpoint"])
        self.assertFalse(rollback["replaces_default_policy"])

    def test_canary_fails_when_formal_preflight_is_not_passed(self) -> None:
        from scripts.run_quasi_real_guarded_formal_ppo_rollout_canary import (
            run_quasi_real_guarded_formal_ppo_rollout_canary,
        )

        self._write_preflight_artifacts(trainable_count=684)
        summary_path = self.preflight_root / "quasi-real-guarded-formal-ppo-preflight-summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["status"] = "failed"
        summary["reason_codes"] = ["preflight_regression"]
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        result = run_quasi_real_guarded_formal_ppo_rollout_canary(
            preflight_root=self.preflight_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            seed_canary_runner=self._passing_seed_canary,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("input_formal_preflight_not_passed", result["reason_codes"])
        self.assertEqual(result["passed_seed_count"], 0)

    def test_canary_fails_when_diagnostic_or_fallback_step_leaks_into_trainable(self) -> None:
        from scripts.run_quasi_real_guarded_formal_ppo_rollout_canary import (
            run_quasi_real_guarded_formal_ppo_rollout_canary,
        )

        self._write_preflight_artifacts(
            trainable_count=684,
            extra_steps=[
                self._step(9_000, split="validation", ppo_trainable=True),
                self._step(9_001, controlled_choice_source="source_fallback", ppo_trainable=True),
                self._step(9_002, gate_reason_codes=["path_cost_regression"], ppo_trainable=True),
            ],
        )

        result = run_quasi_real_guarded_formal_ppo_rollout_canary(
            preflight_root=self.preflight_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            seed_canary_runner=self._passing_seed_canary,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("formal_rollout_canary_split_leakage", result["reason_codes"])
        self.assertIn("formal_rollout_canary_fallback_trainable", result["reason_codes"])
        self.assertIn("formal_rollout_canary_gate_reason_trainable", result["reason_codes"])
        self.assertEqual(result["validation_trainable_count"], 1)
        self.assertEqual(result["fallback_trainable_count"], 1)
        self.assertEqual(result["non_empty_gate_reason_trainable_count"], 1)

    def test_canary_fails_when_seed_metrics_violate_guarded_contract(self) -> None:
        from scripts.run_quasi_real_guarded_formal_ppo_rollout_canary import (
            run_quasi_real_guarded_formal_ppo_rollout_canary,
        )

        self._write_preflight_artifacts(trainable_count=684)

        def runner(*, seed: int, **kwargs) -> dict:
            summary = self._passing_seed_canary(seed=seed, **kwargs)
            if seed == 1:
                summary["status"] = "failed"
                summary["reason_codes"] = ["controlled_path_regression"]
                summary["controlled_regression_count"] = 1
            return summary

        result = run_quasi_real_guarded_formal_ppo_rollout_canary(
            preflight_root=self.preflight_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            seed_canary_runner=runner,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("formal_rollout_canary_seed_not_all_passed", result["reason_codes"])
        self.assertIn("formal_rollout_canary_controlled_regression", result["reason_codes"])
        self.assertEqual(result["passed_seed_count"], 2)

    def test_default_seed_canary_runner_materializes_experimental_candidate_only(self) -> None:
        from scripts.run_quasi_real_guarded_formal_ppo_rollout_canary import (
            _run_seed_canary,
        )

        trainable_steps = [self._full_observation_step(index) for index in range(2)]

        def fake_update_runner(*, collector_root: Path, output_root: Path, config: dict, **_kwargs) -> dict:
            self.assertTrue((collector_root / "ppo-rollout-episodes.jsonl").is_file())
            self.assertTrue((collector_root / "ppo-rollout-collector-summary.json").is_file())
            self.assertEqual(config["training"]["epochs"], 3)
            self.assertLessEqual(config["training"]["learning_rate"], 1.0e-5)
            output_root.mkdir(parents=True, exist_ok=True)
            return {
                "schema_version": "limited-ppo-update-smoke-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "optimizer_train_transition_count": 2,
                "old_log_prob_max_abs_error": 0.0,
                "old_value_max_abs_error": 0.0,
                "loss_non_finite_count": 0,
                "non_finite_gradient_count": 0,
                "non_finite_reward_count": 0,
                "non_finite_return_count": 0,
                "non_finite_advantage_count": 0,
                "parameter_l2_delta": 0.001,
                "approx_kl": 0.01,
                "max_grad_norm_after_clip": 0.5,
            }

        config = self._config()
        config["update_smoke"] = {"base_candidate_root": str(self.temp_dir / "missing-candidate")}
        summary = _run_seed_canary(
            seed=7,
            trainable_steps=trainable_steps,
            output_root=self.output_root,
            config=config,
            repo_root=self.repo_root,
            batch_root=self.batch_root,
            ppo_update_runner=fake_update_runner,
        )

        self.assertEqual(summary["schema_version"], "quasi-real-guarded-formal-ppo-rollout-canary-seed-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["optimizer_train_transition_count"], 2)
        self.assertEqual(summary["post_update_guarded_collector_trainable_transition_count"], 2)
        self.assertTrue(summary["runs_formal_ppo_rollout_canary"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])
        self.assertFalse(summary["formal_training_ready_claimed"])

    def test_readiness_accepts_passed_formal_rollout_canary_summary(self) -> None:
        from scripts.run_policy_training_readiness_review import (
            _quasi_real_guarded_formal_ppo_rollout_canary_readiness,
        )

        readiness = _quasi_real_guarded_formal_ppo_rollout_canary_readiness(
            self._canary_summary()
        )

        self.assertTrue(readiness["present"])
        self.assertTrue(readiness["completed"])
        self.assertEqual(readiness["training_blockers"], [])
        self.assertEqual(readiness["trainable_transition_count"], 684)
        self.assertEqual(readiness["passed_seed_count"], 3)

    def test_config_declares_canary_contract_docs_and_non_goals(self) -> None:
        config_path = (
            self.repo_root
            / "configs"
            / "quasi_real_guarded_formal_ppo_rollout_canary_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["schema_version"],
            "quasi-real-guarded-formal-ppo-rollout-canary-config/v1",
        )
        self.assertEqual(config["validation"]["expected_trainable_transition_count"], 684)
        self.assertEqual(config["training"]["epochs"], 3)
        self.assertLessEqual(config["training"]["learning_rate"], 1.0e-5)
        self.assertIn("README.md", config["documentation_updates"])
        self.assertIn("does_not_publish_checkpoint", config["non_goals"])
        self.assertIn("does_not_claim_formal_training_ready", config["non_goals"])

    def _config(self) -> dict:
        return {
            "schema_version": "quasi-real-guarded-formal-ppo-rollout-canary-config/v1",
            "seeds": [0, 1, 2],
            "training": {
                "epochs": 3,
                "learning_rate": 1.0e-5,
                "clip_ratio": 0.2,
                "max_grad_norm": 1.0,
                "discount_factor": 0.99,
                "device": "cpu",
            },
            "validation": {
                "expected_trainable_transition_count": 684,
                "max_old_log_prob_abs_error": 1.0e-4,
                "max_old_value_abs_error": 1.0e-4,
                "max_abs_approx_kl": 0.25,
                "max_grad_norm_after_clip": 1.0,
                "min_teacher_agreement_rate": 0.95,
            },
            "readiness": {
                "config": "configs/policy_training_readiness_review_v1.json",
                "expected_status": "quasi_real_guarded_formal_ppo_rollout_canary_evaluated",
            },
            "output_files": {
                "summary": "quasi-real-guarded-formal-ppo-rollout-canary-summary.json",
                "seed_summaries": "formal-rollout-canary-seed-summaries.jsonl",
                "training_curves": "formal-rollout-canary-training-curves.json",
                "gate_audit": "formal-rollout-canary-gate-audit.json",
                "rollback_manifest": "formal-rollout-canary-rollback-manifest.json",
                "readiness_validate_only": "formal-rollout-canary-readiness-validate-only.json",
                "progress": "formal-rollout-canary-progress.jsonl",
                "report": "formal-rollout-canary-report.md",
            },
        }

    def _write_preflight_artifacts(
        self,
        *,
        trainable_count: int,
        extra_steps: list[dict] | None = None,
    ) -> None:
        steps_path = self.preflight_root / "formal-preflight-steps.jsonl"
        steps = [self._step(index) for index in range(trainable_count)]
        steps.extend(extra_steps or [])
        steps_path.write_text(
            "".join(json.dumps(step, sort_keys=True) + "\n" for step in steps),
            encoding="utf-8",
        )
        rollback_path = self.preflight_root / "formal-preflight-rollback-manifest.json"
        rollback_path.write_text(
            json.dumps(
                {
                    "schema_version": "quasi-real-guarded-formal-ppo-preflight-rollback-manifest/v1",
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "base_candidate_root": "outputs/path_feedback_batch_quasi_real_teacher_distillation_candidate_v1",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        freeze_manifest_path = self.preflight_root / "freeze-manifest.json"
        freeze_manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "quasi-real-guarded-ppo-iterative-miniloop-evidence-manifest/v1",
                    "required_artifact_missing_count": 0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary = {
            "schema_version": "quasi-real-guarded-formal-ppo-preflight-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "input_trainable_transition_count": trainable_count,
            "optimizer_train_transition_count": trainable_count,
            "unique_trainable_context_count": trainable_count,
            "seed_count": 3,
            "passed_seed_count": 3,
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
            "freeze_manifest": str(freeze_manifest_path),
            "rollback_manifest": str(rollback_path),
            "runs_formal_ppo_rollout": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
        }
        (self.preflight_root / "quasi-real-guarded-formal-ppo-preflight-summary.json").write_text(
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
            "schema_version": "quasi-real-trainable-context-expansion-step/v1",
            "context_id": f"context-{index:04d}",
            "scenario_id": f"scenario-{index % 8}",
            "scenario_family": "family-a",
            "episode_id": f"episode-{index:04d}",
            "step_index": 0,
            "split": split,
            "ppo_trainable": ppo_trainable,
            "controlled_choice_source": controlled_choice_source,
            "controlled_action_index": 0,
            "gate_reason_codes": gate_reason_codes or [],
            "controlled_regression_reason_codes": [],
            "observation": {"present": True},
            "log_prob": -0.1,
            "value": 0.2,
            "reward": 1.0,
            "discounted_return": 1.0,
            "advantage": 0.8,
        }

    def _full_observation_step(self, index: int) -> dict:
        step = self._step(index)
        step["observation"] = {
            "candidate_feature_names": ["feature"],
            "candidate_features": [[1.0]],
            "global_feature_names": ["global"],
            "global_features": [0.0],
            "action_mask": [True],
            "candidate_cells": [[1, 2]],
            "candidate_missing_feature_names": [[]],
            "candidate_missing_indicator_names": [],
            "candidate_missing_indicators": [[]],
        }
        return step

    def _passing_seed_canary(self, *, seed: int, trainable_steps: list[dict], **_kwargs) -> dict:
        return {
            "schema_version": "quasi-real-guarded-formal-ppo-rollout-canary-seed-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "seed": seed,
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
            "controlled_safety_regression_count": 0,
            "controlled_contract_regression_count": 0,
            "controlled_path_risk_regression_count": 0,
            "controlled_source_selection_regression_count": 0,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "runs_formal_ppo_rollout_canary": True,
            "training_curve_records": [
                {"seed": seed, "epoch": 0, "total_loss": 0.1, "approx_kl": 0.01}
            ],
            "updated_candidate_root": str(self.output_root / f"seed-{seed:02d}" / "candidate"),
        }

    def _passing_readiness(self, **_kwargs) -> dict:
        return {
            "training_readiness_status": "quasi_real_guarded_formal_ppo_rollout_canary_evaluated",
            "training_blockers": [],
            "reason_codes": [],
        }

    def _canary_summary(self) -> dict:
        return {
            "schema_version": "quasi-real-guarded-formal-ppo-rollout-canary-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "input_trainable_transition_count": 684,
            "optimizer_train_transition_count": 684,
            "unique_trainable_context_count": 684,
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
            "max_grad_norm_after_clip": 1.0,
            "min_parameter_l2_delta": 0.001,
            "teacher_agreement_rate": 1.0,
            "controlled_regression_count": 0,
            "controlled_safety_regression_count": 0,
            "controlled_contract_regression_count": 0,
            "controlled_path_risk_regression_count": 0,
            "controlled_source_selection_regression_count": 0,
            "rollback_manifest": "formal-rollout-canary-rollback-manifest.json",
            "runs_formal_ppo_rollout_canary": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
        }


if __name__ == "__main__":
    unittest.main()
