import json
import tempfile
import unittest
from pathlib import Path


class QuasiRealGuardedPpoIterativeMiniLoopStabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="qreal-guarded-iterative-"))
        self.expansion_root = self.temp_dir / "expansion"
        self.scale512_root = self.temp_dir / "scale512"
        self.output_root = self.temp_dir / "iterative"
        self.batch_root = self.temp_dir / "batch"
        self.expansion_root.mkdir(parents=True)
        self.scale512_root.mkdir(parents=True)
        self.batch_root.mkdir(parents=True)

    def test_stability_passes_with_three_seeds_three_iterations(self) -> None:
        from scripts.run_quasi_real_guarded_ppo_iterative_miniloop_stability import (
            run_quasi_real_guarded_ppo_iterative_miniloop_stability,
        )

        self._write_input_artifacts(trainable_unique_count=684)

        result = run_quasi_real_guarded_ppo_iterative_miniloop_stability(
            expansion_root=self.expansion_root,
            scale512_root=self.scale512_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            iteration_runner=self._passing_iteration,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(
            result["schema_version"],
            "quasi-real-guarded-ppo-iterative-miniloop-stability-summary/v1",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(result["input_trainable_transition_count"], 684)
        self.assertEqual(result["unique_trainable_context_count"], 684)
        self.assertEqual(result["seed_count"], 3)
        self.assertEqual(result["iteration_count"], 3)
        self.assertEqual(result["passed_iteration_count"], 9)
        self.assertEqual(result["failed_iteration_count"], 0)
        self.assertEqual(result["min_optimizer_train_transition_count"], 684)
        self.assertEqual(result["validation_trainable_count"], 0)
        self.assertEqual(result["test_trainable_count"], 0)
        self.assertEqual(result["source_fallback_trainable_count"], 0)
        self.assertEqual(result["teacher_fallback_trainable_count"], 0)
        self.assertEqual(result["max_old_log_prob_abs_error"], 0.0)
        self.assertEqual(result["max_old_value_abs_error"], 0.0)
        self.assertEqual(result["loss_non_finite_count"], 0)
        self.assertEqual(result["non_finite_gradient_count"], 0)
        self.assertLessEqual(abs(result["max_abs_approx_kl"]), 0.25)
        self.assertLessEqual(result["max_grad_norm_after_clip"], 1.0)
        self.assertEqual(result["controlled_regression_count"], 0)
        self.assertEqual(result["behavior_drift_count"], 0)
        self.assertEqual(result["min_teacher_agreement_rate"], 1.0)
        self.assertEqual(
            result["readiness_status"],
            "quasi_real_guarded_ppo_iterative_miniloop_stability_evaluated",
        )
        self.assertFalse(result["runs_formal_ppo_rollout"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])

        progress_rows = [
            json.loads(line)
            for line in (self.output_root / "iterative-miniloop-progress.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(progress_rows), 9)
        self.assertEqual(progress_rows[0]["seed"], 0)
        self.assertEqual(progress_rows[0]["iteration"], 0)
        self.assertIn("teacher_agreement_rate", progress_rows[0])

    def test_stability_fails_when_any_iteration_uses_too_few_trainable_steps(self) -> None:
        from scripts.run_quasi_real_guarded_ppo_iterative_miniloop_stability import (
            run_quasi_real_guarded_ppo_iterative_miniloop_stability,
        )

        self._write_input_artifacts(trainable_unique_count=684)

        def runner(**kwargs) -> dict:
            summary = self._passing_iteration(**kwargs)
            if kwargs["seed"] == 1 and kwargs["iteration"] == 2:
                summary["optimizer_train_transition_count"] = 100
            return summary

        result = run_quasi_real_guarded_ppo_iterative_miniloop_stability(
            expansion_root=self.expansion_root,
            scale512_root=self.scale512_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            iteration_runner=runner,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn(
            "quasi_real_guarded_iterative_optimizer_train_count_mismatch",
            result["reason_codes"],
        )
        self.assertEqual(result["passed_iteration_count"], 8)

    def test_stability_fails_when_split_or_fallback_leaks_into_trainable_input(self) -> None:
        from scripts.run_quasi_real_guarded_ppo_iterative_miniloop_stability import (
            run_quasi_real_guarded_ppo_iterative_miniloop_stability,
        )

        self._write_input_artifacts(trainable_unique_count=684, leak_validation=True)

        result = run_quasi_real_guarded_ppo_iterative_miniloop_stability(
            expansion_root=self.expansion_root,
            scale512_root=self.scale512_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            iteration_runner=self._passing_iteration,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("quasi_real_guarded_iterative_split_leakage", result["reason_codes"])
        self.assertEqual(result["validation_trainable_count"], 1)
        self.assertEqual(result["passed_iteration_count"], 0)

    def test_readiness_accepts_passed_guarded_iterative_summary(self) -> None:
        from scripts.run_policy_training_readiness_review import (
            _quasi_real_guarded_ppo_iterative_miniloop_stability_readiness,
        )

        readiness = _quasi_real_guarded_ppo_iterative_miniloop_stability_readiness(
            self._iterative_summary()
        )

        self.assertTrue(readiness["present"])
        self.assertTrue(readiness["completed"])
        self.assertEqual(readiness["training_blockers"], [])
        self.assertEqual(readiness["passed_iteration_count"], 9)

    def test_config_declares_guarded_iterative_contract_and_progress_tracking(self) -> None:
        config_path = (
            self.repo_root
            / "configs"
            / "quasi_real_guarded_ppo_iterative_miniloop_stability_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["schema_version"],
            "quasi-real-guarded-ppo-iterative-miniloop-stability-config/v1",
        )
        self.assertEqual(config["iteration_count"], 3)
        self.assertEqual(config["seeds"], [0, 1, 2])
        self.assertEqual(config["validation"]["expected_optimizer_train_transition_count"], 684)
        self.assertIn("progress_jsonl", config["output_files"])
        self.assertIn("does_not_download_new_raw_data", config["non_goals"])

    def _config(self) -> dict:
        return {
            "schema_version": "quasi-real-guarded-ppo-iterative-miniloop-stability-config/v1",
            "iteration_count": 3,
            "seeds": [0, 1, 2],
            "validation": {
                "expected_optimizer_train_transition_count": 684,
                "min_unique_trainable_context_count": 684,
                "min_teacher_agreement_rate": 0.95,
                "max_old_log_prob_abs_error": 1.0e-4,
                "max_old_value_abs_error": 1.0e-4,
                "max_abs_approx_kl": 0.25,
                "max_grad_norm_after_clip": 1.0,
            },
            "training": {
                "epochs": 1,
                "learning_rate": 1.0e-5,
                "clip_ratio": 0.2,
                "max_grad_norm": 1.0,
            },
            "readiness": {
                "config": "configs/policy_training_readiness_review_v1.json",
                "expected_status": "quasi_real_guarded_ppo_iterative_miniloop_stability_evaluated",
            },
            "output_files": {
                "summary": "quasi-real-guarded-ppo-iterative-miniloop-stability-summary.json",
                "iteration_summaries": "iterative-miniloop-iteration-summaries.jsonl",
                "progress_jsonl": "iterative-miniloop-progress.jsonl",
                "readiness_validate_only": "iterative-miniloop-readiness-validate-only.json",
                "report": "iterative-miniloop-stability-report.md",
            },
        }

    def _write_input_artifacts(
        self, *, trainable_unique_count: int, leak_validation: bool = False
    ) -> None:
        steps = [self._step(index, trainable=True) for index in range(trainable_unique_count)]
        if leak_validation:
            steps.append(self._step(99_999, trainable=True, split="validation"))
        steps.extend(self._step(index + 200_000, trainable=False) for index in range(8))
        steps_path = self.expansion_root / "quasi-real-trainable-context-expansion-steps.jsonl"
        steps_path.write_text(
            "".join(json.dumps(step, sort_keys=True) + "\n" for step in steps),
            encoding="utf-8",
        )
        expansion_summary = {
            "schema_version": "quasi-real-trainable-context-expansion-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "unique_trainable_context_count": trainable_unique_count,
            "ppo_trainable_transition_count": trainable_unique_count,
            "validation_trainable_count": 1 if leak_validation else 0,
            "test_trainable_count": 0,
            "source_fallback_trainable_count": 0,
            "teacher_fallback_trainable_count": 0,
            "controlled_regression_count": 0,
            "teacher_agreement_rate": 1.0,
            "steps": str(steps_path),
            "scale512_summary": str(
                self.scale512_root
                / "quasi-real-guarded-ppo-scale512-multiseed-preflight-summary.json"
            ),
        }
        (self.expansion_root / "quasi-real-trainable-context-expansion-summary.json").write_text(
            json.dumps(expansion_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (self.scale512_root / "quasi-real-guarded-ppo-scale512-multiseed-preflight-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "quasi-real-guarded-ppo-scale512-multiseed-preflight-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "horizon": 5,
                    "ppo_trainable_transition_count": trainable_unique_count,
                    "unique_trainable_context_count": trainable_unique_count,
                    "seed_count": 3,
                    "passed_seed_count": 3,
                    "controlled_regression_count": 0,
                    "teacher_agreement_rate": 1.0,
                    "readiness_status": "quasi_real_guarded_ppo_scale512_multiseed_preflight_evaluated",
                    "git_provenance": {"current": {}, "current_matches_sources": True},
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _step(self, index: int, *, trainable: bool, split: str = "train") -> dict:
        return {
            "schema_version": "quasi-real-guarded-ppo-horizon5-batch-expansion-step/v1",
            "episode_id": f"episode-{index // 5:04d}",
            "step_index": index % 5,
            "context_id": f"context-{index:06d}",
            "scenario_id": f"scenario-{index:06d}",
            "scenario_family": "unit_family",
            "split": split if trainable else "validation",
            "controlled_choice_source": "policy",
            "ppo_trainable": trainable,
            "gate_reason_codes": [],
            "controlled_regression_reason_codes": [],
            "observation": {"action_mask": [True, True, True]},
            "missing_observation": False,
            "log_prob": -0.25,
            "value": 0.1,
            "reward": 1.0,
            "discounted_return": 1.0,
            "advantage": 0.9,
        }

    def _passing_iteration(
        self,
        *,
        seed: int,
        iteration: int,
        trainable_steps: list[dict],
        base_candidate_root: Path,
        output_root: Path,
        **_kwargs,
    ) -> dict:
        iteration_root = output_root / f"seed-{seed:02d}" / f"iteration-{iteration:02d}"
        update_root = iteration_root / "limited_ppo_update_smoke"
        update_root.mkdir(parents=True, exist_ok=True)
        return {
            "schema_version": "quasi-real-guarded-ppo-iterative-miniloop-iteration-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "seed": seed,
            "iteration": iteration,
            "base_candidate_root": str(base_candidate_root),
            "updated_candidate_root": str(update_root),
            "optimizer_train_transition_count": len(trainable_steps),
            "post_update_guarded_collector_trainable_transition_count": len(trainable_steps),
            "old_log_prob_max_abs_error": 0.0,
            "old_value_max_abs_error": 0.0,
            "loss_non_finite_count": 0,
            "non_finite_gradient_count": 0,
            "non_finite_reward_count": 0,
            "non_finite_return_count": 0,
            "non_finite_advantage_count": 0,
            "approx_kl": 0.01,
            "max_grad_norm_after_clip": 0.5,
            "controlled_regression_count": 0,
            "behavior_drift_count": 0,
            "teacher_agreement_rate": 1.0,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
        }

    def _passing_readiness(self, *_args, **_kwargs) -> dict:
        return {
            "schema_version": "policy-training-readiness-review-summary/v1",
            "status": "config validated",
            "reason_codes": [],
            "training_readiness_status": "quasi_real_guarded_ppo_iterative_miniloop_stability_evaluated",
            "training_blockers": [],
            "recommended_next_action": "quasi_real_guarded_ppo_iterative_miniloop_stability_evaluated",
        }

    def _iterative_summary(self) -> dict:
        return {
            "schema_version": "quasi-real-guarded-ppo-iterative-miniloop-stability-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "input_trainable_transition_count": 684,
            "unique_trainable_context_count": 684,
            "seed_count": 3,
            "iteration_count": 3,
            "passed_iteration_count": 9,
            "failed_iteration_count": 0,
            "min_optimizer_train_transition_count": 684,
            "validation_trainable_count": 0,
            "test_trainable_count": 0,
            "source_fallback_trainable_count": 0,
            "teacher_fallback_trainable_count": 0,
            "max_old_log_prob_abs_error": 0.0,
            "max_old_value_abs_error": 0.0,
            "loss_non_finite_count": 0,
            "non_finite_gradient_count": 0,
            "non_finite_reward_count": 0,
            "non_finite_return_count": 0,
            "non_finite_advantage_count": 0,
            "max_abs_approx_kl": 0.01,
            "max_grad_norm_after_clip": 0.5,
            "min_teacher_agreement_rate": 1.0,
            "controlled_regression_count": 0,
            "behavior_drift_count": 0,
            "runs_formal_ppo_rollout": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "git_provenance": {"current": {}, "current_matches_sources": True},
        }


if __name__ == "__main__":
    unittest.main()
