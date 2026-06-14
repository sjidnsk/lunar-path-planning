import json
import shutil
import tempfile
import unittest
from pathlib import Path


class GuardedFormalPpoTrainingRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="guarded-formal-ppo-training-run-"))
        self.authorization_root = self.temp_dir / "authorization"
        self.stability_root = self.temp_dir / "stability"
        self.output_root = self.temp_dir / "training-run"
        self.batch_root = self.temp_dir / "batch"
        for path in (self.authorization_root, self.stability_root, self.output_root, self.batch_root):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_runs_authorized_formal_training_across_all_seeds(self) -> None:
        from scripts.run_guarded_formal_ppo_training_run import (
            run_guarded_formal_ppo_training_run,
        )

        self._write_authorized_inputs()
        seen_seeds: list[int] = []

        def seed_runner(**kwargs: object) -> dict:
            seed = int(kwargs["seed"])
            seen_seeds.append(seed)
            trainable_steps = list(kwargs["trainable_steps"])
            return self._seed_summary(seed=seed, trainable_count=len(trainable_steps))

        result = run_guarded_formal_ppo_training_run(
            authorization_root=self.authorization_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            seed_training_runner=seed_runner,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["schema_version"], "guarded-formal-ppo-training-run-summary/v1")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertTrue(result["runs_guarded_formal_ppo_training_run"])
        self.assertTrue(result["runs_new_ppo_update"])
        self.assertEqual(result["input_authorization_status"], "passed")
        self.assertEqual(result["authorization_verdict"], "authorized_for_guarded_formal_ppo_training_run")
        self.assertEqual(result["authorized_trainable_transition_count"], 3)
        self.assertEqual(result["optimizer_train_transition_count"], 3)
        self.assertEqual(result["unique_trainable_context_count"], 3)
        self.assertEqual(result["validation_trainable_count"], 0)
        self.assertEqual(result["test_trainable_count"], 0)
        self.assertEqual(result["fallback_trainable_count"], 0)
        self.assertEqual(result["diagnostic_trainable_count"], 0)
        self.assertEqual(result["seed_count"], 5)
        self.assertEqual(result["passed_seed_count"], 5)
        self.assertEqual(result["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(seen_seeds, [0, 1, 2, 3, 4])
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
        self.assertGreaterEqual(result["teacher_agreement_rate"], 0.95)
        self.assertEqual(result["controlled_regression_count"], 0)
        self.assertEqual(result["post_training_holdout_status"], "passed")
        self.assertEqual(result["post_training_canary_status"], "passed")
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])
        self.assertEqual(result["readiness_status"], "guarded_formal_ppo_training_run_evaluated")

        for filename in (
            "formal-ppo-training-run-summary.json",
            "formal-ppo-training-run-seed-summaries.jsonl",
            "formal-ppo-training-run-progress.jsonl",
            "formal-ppo-training-run-gate-audit.json",
            "formal-ppo-training-run-readiness-validate-only.json",
            "formal-ppo-training-run-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_blocks_seed_gate_regression_or_publication_claims(self) -> None:
        from scripts.run_guarded_formal_ppo_training_run import (
            run_guarded_formal_ppo_training_run,
        )

        self._write_authorized_inputs()

        def seed_runner(**kwargs: object) -> dict:
            seed = int(kwargs["seed"])
            summary = self._seed_summary(seed=seed, trainable_count=3)
            if seed == 2:
                summary["controlled_path_risk_regression_count"] = 1
                summary["controlled_regression_count"] = 1
                summary["post_training_holdout_status"] = "failed"
                summary["publishes_checkpoint"] = True
            return summary

        result = run_guarded_formal_ppo_training_run(
            authorization_root=self.authorization_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            seed_training_runner=seed_runner,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("guarded_formal_ppo_training_run_controlled_regression", result["reason_codes"])
        self.assertIn("guarded_formal_ppo_training_run_post_training_gate_failed", result["reason_codes"])
        self.assertIn("limited_ppo_update_checkpoint_publication_claimed", result["reason_codes"])

    def test_config_declares_docs_outputs_and_non_goals(self) -> None:
        config_path = self.repo_root / "configs" / "guarded_formal_ppo_training_run_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["schema_version"], "guarded-formal-ppo-training-run-config/v1")
        self.assertEqual(config["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(config["training"]["epochs"], 1)
        self.assertLessEqual(config["training"]["learning_rate"], 0.00001)
        self.assertIn("formal-ppo-training-run-summary.json", config["output_files"].values())
        self.assertIn("README.md", config["documentation_updates"])
        self.assertIn("docs/算法设计与系统架构报告.md", config["documentation_updates"])
        self.assertIn(
            "docs/superpowers/specs/2026-06-14-guarded-formal-ppo-training-run.md",
            config["documentation_updates"],
        )
        self.assertIn("does_not_launch_online_canary", config["non_goals"])
        self.assertIn("does_not_publish_checkpoint", config["non_goals"])
        self.assertIn("does_not_replace_default_policy", config["non_goals"])

    def _write_authorized_inputs(self) -> None:
        steps_path = self.stability_root / "steps.jsonl"
        steps = [
            self._step("ctx-1"),
            self._step("ctx-2"),
            self._step("ctx-3"),
            {**self._step("diag-validation"), "split": "validation", "ppo_trainable": False},
        ]
        steps_path.write_text("".join(json.dumps(step, sort_keys=True) + "\n" for step in steps), encoding="utf-8")
        stability_summary_path = self.stability_root / "quasi-real-guarded-formal-ppo-stability-holdout-validation-summary.json"
        stability_summary_path.write_text(
            json.dumps(
                {
                    "schema_version": "quasi-real-guarded-formal-ppo-stability-holdout-validation-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "steps": str(steps_path),
                    "input_trainable_transition_count": 3,
                    "optimizer_train_transition_count": 3,
                    "unique_trainable_context_count": 3,
                    "teacher_agreement_rate": 1.0,
                    "controlled_regression_count": 0,
                    "git_provenance": {"current_matches_sources": True, "current": {"dirty": False}},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        authorization = {
            "schema_version": "guarded-formal-ppo-training-authorization-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "authorization_verdict": "authorized_for_guarded_formal_ppo_training_run",
            "readiness_status": "guarded_formal_ppo_training_authorized",
            "formal_stability_holdout_summary": str(stability_summary_path),
            "authorized_trainable_transition_count": 3,
            "authorized_optimizer_train_transition_count": 3,
            "unique_authorized_trainable_context_count": 3,
            "validation_trainable_count": 0,
            "test_trainable_count": 0,
            "fallback_trainable_count": 0,
            "source_fallback_trainable_count": 0,
            "teacher_fallback_trainable_count": 0,
            "diagnostic_trainable_count": 0,
            "non_empty_gate_reason_trainable_count": 0,
            "missing_observation_count": 0,
            "missing_log_prob_count": 0,
            "missing_value_count": 0,
            "invalid_action_mask_count": 0,
            "non_finite_reward_count": 0,
            "non_finite_return_count": 0,
            "non_finite_advantage_count": 0,
            "controlled_regression_count": 0,
            "controlled_safety_regression_count": 0,
            "controlled_contract_regression_count": 0,
            "controlled_path_risk_regression_count": 0,
            "controlled_source_selection_regression_count": 0,
            "seed_count": 5,
            "seeds": [0, 1, 2, 3, 4],
            "budget_manifest_passed": True,
            "seed_plan_passed": True,
            "stop_condition_manifest_passed": True,
            "rollback_manifest_passed": True,
            "post_training_gate_plan_passed": True,
            "source_git_provenance_mismatch_count": 0,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "git_provenance": {"current_matches_sources": True, "current": {"dirty": False}},
        }
        (self.authorization_root / "formal-ppo-training-authorization-summary.json").write_text(
            json.dumps(authorization, indent=2),
            encoding="utf-8",
        )

    def _step(self, context_id: str) -> dict:
        return {
            "context_id": context_id,
            "episode_id": f"episode-{context_id}",
            "scenario_id": f"scenario-{context_id}",
            "scenario_family": "unit",
            "split": "train",
            "ppo_trainable": True,
            "controlled_choice_source": "policy",
            "controlled_action_index": 0,
            "raw_policy_action_index": 0,
            "source_action_index": 0,
            "gate_reason_codes": [],
            "controlled_regression_reason_codes": [],
            "observation": {"present": True},
            "log_prob": -0.1,
            "value": 0.2,
            "reward": 1.0,
            "discounted_return": 1.0,
            "advantage": 0.8,
            "done": True,
        }

    def _seed_summary(self, *, seed: int, trainable_count: int) -> dict:
        return {
            "schema_version": "guarded-formal-ppo-training-run-seed-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "seed": seed,
            "optimizer_train_transition_count": trainable_count,
            "post_update_guarded_collector_trainable_transition_count": trainable_count,
            "old_log_prob_max_abs_error": 0.0,
            "old_value_max_abs_error": 0.0,
            "loss_non_finite_count": 0,
            "non_finite_gradient_count": 0,
            "non_finite_reward_count": 0,
            "non_finite_return_count": 0,
            "non_finite_advantage_count": 0,
            "parameter_l2_delta": 0.001 + seed * 0.0001,
            "approx_kl": 0.01,
            "max_grad_norm_after_clip": 0.5,
            "teacher_agreement_rate": 1.0,
            "controlled_regression_count": 0,
            "controlled_safety_regression_count": 0,
            "controlled_contract_regression_count": 0,
            "controlled_path_risk_regression_count": 0,
            "controlled_source_selection_regression_count": 0,
            "post_training_holdout_status": "passed",
            "post_training_canary_status": "passed",
            "experimental_checkpoint": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "training_curve_records": [
                {
                    "seed": seed,
                    "epoch": 1,
                    "optimizer_train_transition_count": trainable_count,
                    "approx_kl": 0.01,
                    "max_grad_norm_after_clip": 0.5,
                }
            ],
        }

    def _config(self) -> dict:
        return {
            "schema_version": "guarded-formal-ppo-training-run-config/v1",
            "seeds": [0, 1, 2, 3, 4],
            "training": {
                "epochs": 1,
                "learning_rate": 0.00001,
                "clip_ratio": 0.2,
                "discount_factor": 0.99,
                "max_grad_norm": 1.0,
                "device": "cpu",
            },
            "validation": {
                "expected_trainable_transition_count": 3,
                "min_seed_count": 5,
                "max_old_log_prob_abs_error": 0.0001,
                "max_old_value_abs_error": 0.0001,
                "max_abs_approx_kl": 0.25,
                "max_grad_norm_after_clip": 1.0,
                "min_teacher_agreement_rate": 0.95,
            },
            "readiness": {
                "config": "configs/policy_training_readiness_review_v1.json",
                "expected_status": "guarded_formal_ppo_training_run_evaluated",
            },
            "output_files": {
                "summary": "formal-ppo-training-run-summary.json",
                "seed_summaries": "formal-ppo-training-run-seed-summaries.jsonl",
                "progress": "formal-ppo-training-run-progress.jsonl",
                "training_curves": "formal-ppo-training-run-training-curves.json",
                "gate_audit": "formal-ppo-training-run-gate-audit.json",
                "rollback_manifest": "formal-ppo-training-run-rollback-manifest.json",
                "readiness_validate_only": "formal-ppo-training-run-readiness-validate-only.json",
                "report": "formal-ppo-training-run-report.md",
            },
        }

    def _passing_readiness(self, **_: object) -> dict:
        return {
            "training_readiness_status": "guarded_formal_ppo_training_run_evaluated",
            "training_blockers": [],
            "reason_codes": [],
            "recommended_next_action": "guarded_formal_ppo_training_run_evaluated",
            "returncode": 0,
        }
