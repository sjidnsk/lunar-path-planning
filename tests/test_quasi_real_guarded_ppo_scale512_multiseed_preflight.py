import json
import tempfile
import unittest
from pathlib import Path


class QuasiRealGuardedPpoScale512MultiSeedPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="quasi-real-scale512-"))
        self.horizon5_root = self.temp_dir / "horizon5"
        self.output_root = self.temp_dir / "scale512"
        self.batch_root = self.temp_dir / "batch"
        self.horizon5_root.mkdir(parents=True)
        self.batch_root.mkdir(parents=True)

    def test_preflight_passes_with_512_unique_trainable_contexts_and_three_seeds(self) -> None:
        from scripts.run_quasi_real_guarded_ppo_scale512_multiseed_preflight import (
            run_quasi_real_guarded_ppo_scale512_multiseed_preflight,
        )

        self._write_horizon5_artifacts(trainable_unique_count=512)

        result = run_quasi_real_guarded_ppo_scale512_multiseed_preflight(
            horizon5_root=self.horizon5_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            seed_smoke_runner=self._passing_seed_smoke,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(
            result["schema_version"],
            "quasi-real-guarded-ppo-scale512-multiseed-preflight-summary/v1",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertGreaterEqual(result["horizon"], 5)
        self.assertEqual(result["ppo_trainable_transition_count"], 512)
        self.assertEqual(result["unique_trainable_context_count"], 512)
        self.assertEqual(result["seed_count"], 3)
        self.assertEqual(result["passed_seed_count"], 3)
        self.assertEqual(result["controlled_regression_count"], 0)
        self.assertEqual(result["validation_trainable_count"], 0)
        self.assertEqual(result["test_trainable_count"], 0)
        self.assertEqual(result["source_fallback_trainable_count"], 0)
        self.assertEqual(result["teacher_fallback_trainable_count"], 0)
        self.assertEqual(result["non_finite_reward_count"], 0)
        self.assertEqual(result["non_finite_return_count"], 0)
        self.assertEqual(result["non_finite_advantage_count"], 0)
        self.assertEqual(
            result["readiness_status"],
            "quasi_real_guarded_ppo_scale512_multiseed_preflight_evaluated",
        )
        self.assertFalse(result["runs_formal_ppo_rollout"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])

        seed_summaries = [
            json.loads(line)
            for line in (self.output_root / "scale512-seed-summaries.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(seed_summaries), 3)
        self.assertTrue(all(row["status"] == "passed" for row in seed_summaries))
        report = (self.output_root / "scale512-multiseed-preflight-report.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Scale-512 Multi-Seed Preflight", report)
        self.assertIn("not formal PPO", report)

    def test_preflight_fails_when_unique_trainable_capacity_is_below_512(self) -> None:
        from scripts.run_quasi_real_guarded_ppo_scale512_multiseed_preflight import (
            run_quasi_real_guarded_ppo_scale512_multiseed_preflight,
        )

        self._write_horizon5_artifacts(trainable_unique_count=36)

        result = run_quasi_real_guarded_ppo_scale512_multiseed_preflight(
            horizon5_root=self.horizon5_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            seed_smoke_runner=self._passing_seed_smoke,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("insufficient_quasi_real_trainable_capacity", result["reason_codes"])
        self.assertEqual(result["unique_trainable_context_count"], 36)
        self.assertEqual(result["passed_seed_count"], 0)
        self.assertEqual(result["seed_smoke_skipped"], True)

    def test_preflight_fails_when_any_seed_smoke_fails(self) -> None:
        from scripts.run_quasi_real_guarded_ppo_scale512_multiseed_preflight import (
            run_quasi_real_guarded_ppo_scale512_multiseed_preflight,
        )

        self._write_horizon5_artifacts(trainable_unique_count=512)

        def runner(*, seed: int, **kwargs) -> dict:
            summary = self._passing_seed_smoke(seed=seed, **kwargs)
            if seed == 1:
                summary["status"] = "failed"
                summary["reason_codes"] = ["seed_controlled_regression"]
                summary["controlled_regression_count"] = 1
            return summary

        result = run_quasi_real_guarded_ppo_scale512_multiseed_preflight(
            horizon5_root=self.horizon5_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            seed_smoke_runner=runner,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("scale512_seed_smoke_not_all_passed", result["reason_codes"])
        self.assertIn("scale512_seed_controlled_regression", result["reason_codes"])
        self.assertEqual(result["passed_seed_count"], 2)

    def test_default_seed_smoke_runner_materializes_collector_and_maps_update_summary(self) -> None:
        from scripts.run_quasi_real_guarded_ppo_scale512_multiseed_preflight import (
            _run_seed_smoke,
        )

        trainable_steps = [self._full_observation_step(index) for index in range(2)]

        def fake_update_runner(*, collector_root: Path, output_root: Path, **_kwargs) -> dict:
            self.assertTrue((collector_root / "ppo-rollout-episodes.jsonl").is_file())
            self.assertTrue((collector_root / "ppo-rollout-collector-summary.json").is_file())
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
        summary = _run_seed_smoke(
            seed=7,
            trainable_steps=trainable_steps,
            output_root=self.output_root,
            config=config,
            repo_root=self.repo_root,
            batch_root=self.batch_root,
            ppo_update_runner=fake_update_runner,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["seed"], 7)
        self.assertEqual(summary["optimizer_train_transition_count"], 2)
        self.assertEqual(summary["post_update_guarded_collector_trainable_transition_count"], 2)
        self.assertEqual(summary["old_log_prob_max_abs_error"], 0.0)
        self.assertEqual(summary["max_grad_norm_after_clip"], 0.5)

    def test_seed_policy_estimate_refresh_overwrites_stale_log_prob_and_value(self) -> None:
        from scripts.run_quasi_real_guarded_ppo_scale512_multiseed_preflight import (
            _refresh_trainable_step_policy_estimates,
        )

        step = self._full_observation_step(0)
        step["log_prob"] = -9.0
        step["value"] = -9.0

        class FakeEvaluator:
            def evaluate(self, _step: dict) -> dict:
                return {"log_prob": -0.33, "value": 0.77}

        refreshed = _refresh_trainable_step_policy_estimates([step], policy_evaluator=FakeEvaluator())

        self.assertEqual(refreshed[0]["log_prob"], -0.33)
        self.assertEqual(refreshed[0]["value"], 0.77)
        self.assertEqual(step["log_prob"], -9.0)

    def test_readiness_accepts_passed_scale512_preflight_summary(self) -> None:
        from scripts.run_policy_training_readiness_review import (
            _quasi_real_guarded_ppo_scale512_multiseed_preflight_readiness,
        )

        readiness = _quasi_real_guarded_ppo_scale512_multiseed_preflight_readiness(
            self._scale512_summary()
        )

        self.assertTrue(readiness["present"])
        self.assertTrue(readiness["completed"])
        self.assertEqual(readiness["training_blockers"], [])
        self.assertEqual(readiness["unique_trainable_context_count"], 512)
        self.assertEqual(readiness["passed_seed_count"], 3)

    def test_config_declares_scale512_multiseed_contract_and_non_goals(self) -> None:
        config_path = (
            self.repo_root
            / "configs"
            / "quasi_real_guarded_ppo_scale512_multiseed_preflight_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["schema_version"],
            "quasi-real-guarded-ppo-scale512-multiseed-preflight-config/v1",
        )
        self.assertEqual(config["validation"]["min_ppo_trainable_transition_count"], 512)
        self.assertEqual(config["validation"]["min_unique_trainable_context_count"], 512)
        self.assertEqual(config["seeds"], [0, 1, 2])
        self.assertIn("does_not_start_formal_ppo", config["non_goals"])

    def _config(self) -> dict:
        return {
            "schema_version": "quasi-real-guarded-ppo-scale512-multiseed-preflight-config/v1",
            "horizon": 5,
            "seeds": [0, 1, 2],
            "validation": {
                "min_horizon": 5,
                "min_ppo_trainable_transition_count": 512,
                "min_unique_trainable_context_count": 512,
                "min_teacher_agreement_rate": 0.95,
                "max_old_log_prob_abs_error": 1.0e-4,
                "max_old_value_abs_error": 1.0e-4,
                "max_abs_approx_kl": 0.25,
                "max_grad_norm_after_clip": 1.0,
            },
            "readiness": {
                "config": "configs/policy_training_readiness_review_v1.json",
                "expected_status": "quasi_real_guarded_ppo_scale512_multiseed_preflight_evaluated",
            },
            "output_files": {
                "summary": "quasi-real-guarded-ppo-scale512-multiseed-preflight-summary.json",
                "capacity_report": "scale512-trainable-capacity-report.json",
                "trainable_contexts": "scale512-trainable-contexts.jsonl",
                "seed_summaries": "scale512-seed-summaries.jsonl",
                "readiness_validate_only": "scale512-readiness-validate-only.json",
                "report": "scale512-multiseed-preflight-report.md",
            },
        }

    def _write_horizon5_artifacts(self, *, trainable_unique_count: int) -> None:
        steps = [self._step(index, trainable=True) for index in range(trainable_unique_count)]
        steps.extend(self._step(index + 10_000, trainable=False) for index in range(16))
        (self.horizon5_root / "quasi-real-guarded-ppo-horizon5-batch-expansion-steps.jsonl").write_text(
            "".join(json.dumps(step, sort_keys=True) + "\n" for step in steps),
            encoding="utf-8",
        )
        summary = {
            "schema_version": "quasi-real-guarded-ppo-horizon5-batch-expansion-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "horizon": 5,
            "episode_count": 128,
            "step_count": len(steps),
            "ppo_trainable_transition_count": trainable_unique_count,
            "diagnostic_transition_count": 16,
            "replay_count": 3,
            "passed_replay_count": 3,
            "controlled_regression_count": 0,
            "teacher_agreement_rate": 1.0,
            "baseline_replay_behavior_drift_count": 0,
            "runs_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "steps": str(
                self.horizon5_root / "quasi-real-guarded-ppo-horizon5-batch-expansion-steps.jsonl"
            ),
        }
        (self.horizon5_root / "quasi-real-guarded-ppo-horizon5-batch-expansion-summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    def _step(self, index: int, *, trainable: bool) -> dict:
        return {
            "schema_version": "quasi-real-guarded-ppo-horizon5-batch-expansion-step/v1",
            "episode_id": f"episode-{index // 5:04d}",
            "step_index": index % 5,
            "context_id": f"context-{index:04d}",
            "scenario_id": f"scenario-{index:04d}",
            "scenario_family": "unit_family",
            "split": "train" if trainable else "validation",
            "controlled_choice_source": "policy",
            "controlled_choice_detail": "policy_teacher_aligned",
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

    def _passing_seed_smoke(self, *, seed: int, trainable_steps: list[dict], output_root: Path, **_kwargs) -> dict:
        seed_root = output_root / f"seed-{seed:02d}"
        seed_root.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema_version": "quasi-real-guarded-ppo-scale512-seed-smoke-summary/v1",
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
            "approx_kl": 0.01,
            "max_grad_norm_after_clip": 0.5,
            "controlled_regression_count": 0,
            "teacher_agreement_rate": 1.0,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
        }
        (seed_root / "seed-smoke-summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary

    def _full_observation_step(self, index: int) -> dict:
        step = self._step(index, trainable=True)
        step["controlled_action_index"] = 0
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

    def _passing_readiness(self, *_args, **_kwargs) -> dict:
        return {
            "schema_version": "policy-training-readiness-review-summary/v1",
            "status": "config validated",
            "reason_codes": [],
            "training_readiness_status": "quasi_real_guarded_ppo_scale512_multiseed_preflight_evaluated",
            "training_blockers": [],
            "recommended_next_action": "quasi_real_guarded_ppo_scale512_multiseed_preflight_evaluated",
        }

    def _scale512_summary(self) -> dict:
        return {
            "schema_version": "quasi-real-guarded-ppo-scale512-multiseed-preflight-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "horizon": 5,
            "ppo_trainable_transition_count": 512,
            "unique_trainable_context_count": 512,
            "validation_trainable_count": 0,
            "test_trainable_count": 0,
            "source_fallback_trainable_count": 0,
            "teacher_fallback_trainable_count": 0,
            "missing_observation_count": 0,
            "missing_log_prob_count": 0,
            "missing_value_count": 0,
            "non_finite_reward_count": 0,
            "non_finite_return_count": 0,
            "non_finite_advantage_count": 0,
            "controlled_regression_count": 0,
            "controlled_safety_regression_count": 0,
            "controlled_contract_regression_count": 0,
            "controlled_path_risk_regression_count": 0,
            "controlled_source_selection_regression_count": 0,
            "teacher_agreement_rate": 1.0,
            "seed_count": 3,
            "passed_seed_count": 3,
            "seed_failure_count": 0,
            "seed_max_old_log_prob_abs_error": 0.0,
            "seed_max_old_value_abs_error": 0.0,
            "seed_loss_non_finite_count": 0,
            "seed_non_finite_gradient_count": 0,
            "seed_non_finite_reward_count": 0,
            "seed_non_finite_return_count": 0,
            "seed_non_finite_advantage_count": 0,
            "seed_max_abs_approx_kl": 0.01,
            "seed_max_grad_norm_after_clip": 0.5,
            "min_post_update_guarded_collector_trainable_transition_count": 512,
            "runs_formal_ppo_rollout": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "git_provenance": {"current": {}, "current_matches_sources": True},
        }


if __name__ == "__main__":
    unittest.main()
