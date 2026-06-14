import json
import tempfile
import unittest
from pathlib import Path


class QuasiRealGuardedFormalPpoPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="qreal-formal-preflight-"))
        self.freeze_root = self.temp_dir / "freeze"
        self.output_root = self.temp_dir / "formal-preflight"
        self.batch_root = self.temp_dir / "batch"
        self.freeze_root.mkdir(parents=True)
        self.batch_root.mkdir(parents=True)

    def test_preflight_passes_from_frozen_miniloop_baseline_and_three_seeds(self) -> None:
        from scripts.run_quasi_real_guarded_formal_ppo_preflight import (
            run_quasi_real_guarded_formal_ppo_preflight,
        )

        self._write_freeze_artifacts(trainable_count=684)

        result = run_quasi_real_guarded_formal_ppo_preflight(
            freeze_root=self.freeze_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            seed_preflight_runner=self._passing_seed_preflight,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(
            result["schema_version"],
            "quasi-real-guarded-formal-ppo-preflight-summary/v1",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(result["input_trainable_transition_count"], 684)
        self.assertEqual(result["optimizer_train_transition_count"], 684)
        self.assertEqual(result["unique_trainable_context_count"], 684)
        self.assertEqual(result["seed_count"], 3)
        self.assertEqual(result["passed_seed_count"], 3)
        self.assertEqual(result["validation_trainable_count"], 0)
        self.assertEqual(result["test_trainable_count"], 0)
        self.assertEqual(result["fallback_trainable_count"], 0)
        self.assertEqual(result["non_empty_gate_reason_trainable_count"], 0)
        self.assertEqual(result["controlled_regression_count"], 0)
        self.assertEqual(result["teacher_agreement_rate"], 1.0)
        self.assertEqual(
            result["readiness_status"],
            "quasi_real_guarded_formal_ppo_preflight_evaluated",
        )
        self.assertFalse(result["runs_formal_ppo_rollout"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])

        self.assertTrue((self.output_root / "formal-preflight-seed-summaries.jsonl").is_file())
        self.assertTrue((self.output_root / "formal-preflight-training-curves.json").is_file())
        self.assertTrue((self.output_root / "formal-preflight-gate-audit.json").is_file())
        rollback = json.loads(
            (self.output_root / "formal-preflight-rollback-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(rollback["schema_version"], "quasi-real-guarded-formal-ppo-preflight-rollback-manifest/v1")
        self.assertFalse(rollback["publishes_checkpoint"])
        self.assertFalse(rollback["replaces_default_policy"])

    def test_preflight_fails_when_frozen_baseline_is_not_passed(self) -> None:
        from scripts.run_quasi_real_guarded_formal_ppo_preflight import (
            run_quasi_real_guarded_formal_ppo_preflight,
        )

        self._write_freeze_artifacts(trainable_count=684)
        freeze_summary_path = (
            self.freeze_root
            / "quasi-real-guarded-ppo-iterative-miniloop-evidence-freeze-summary.json"
        )
        summary = json.loads(freeze_summary_path.read_text(encoding="utf-8"))
        summary["status"] = "failed"
        summary["reason_codes"] = ["baseline_not_frozen"]
        freeze_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        result = run_quasi_real_guarded_formal_ppo_preflight(
            freeze_root=self.freeze_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            seed_preflight_runner=self._passing_seed_preflight,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("input_freeze_summary_not_passed", result["reason_codes"])
        self.assertEqual(result["passed_seed_count"], 0)

    def test_preflight_fails_when_validation_or_fallback_leaks_into_trainable(self) -> None:
        from scripts.run_quasi_real_guarded_formal_ppo_preflight import (
            run_quasi_real_guarded_formal_ppo_preflight,
        )

        self._write_freeze_artifacts(
            trainable_count=684,
            extra_steps=[
                self._step(9_000, split="validation", ppo_trainable=True),
                self._step(9_001, controlled_choice_source="source_fallback", ppo_trainable=True),
            ],
        )

        result = run_quasi_real_guarded_formal_ppo_preflight(
            freeze_root=self.freeze_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            seed_preflight_runner=self._passing_seed_preflight,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("formal_preflight_split_leakage", result["reason_codes"])
        self.assertIn("formal_preflight_fallback_trainable", result["reason_codes"])
        self.assertEqual(result["validation_trainable_count"], 1)
        self.assertEqual(result["fallback_trainable_count"], 1)

    def test_preflight_fails_when_seed_metrics_violate_contract(self) -> None:
        from scripts.run_quasi_real_guarded_formal_ppo_preflight import (
            run_quasi_real_guarded_formal_ppo_preflight,
        )

        self._write_freeze_artifacts(trainable_count=684)

        def runner(*, seed: int, **kwargs) -> dict:
            summary = self._passing_seed_preflight(seed=seed, **kwargs)
            if seed == 2:
                summary["status"] = "failed"
                summary["reason_codes"] = ["seed_kl_too_large"]
                summary["approx_kl"] = 0.5
            return summary

        result = run_quasi_real_guarded_formal_ppo_preflight(
            freeze_root=self.freeze_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            seed_preflight_runner=runner,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("formal_preflight_seed_not_all_passed", result["reason_codes"])
        self.assertIn("formal_preflight_seed_kl_too_large", result["reason_codes"])
        self.assertEqual(result["passed_seed_count"], 2)

    def test_readiness_accepts_passed_formal_preflight_summary(self) -> None:
        from scripts.run_policy_training_readiness_review import (
            _quasi_real_guarded_formal_ppo_preflight_readiness,
        )

        readiness = _quasi_real_guarded_formal_ppo_preflight_readiness(
            self._formal_summary()
        )

        self.assertTrue(readiness["present"])
        self.assertTrue(readiness["completed"])
        self.assertEqual(readiness["training_blockers"], [])
        self.assertEqual(readiness["trainable_transition_count"], 684)
        self.assertEqual(readiness["passed_seed_count"], 3)

    def test_config_declares_formal_preflight_contract_docs_and_non_goals(self) -> None:
        config_path = (
            self.repo_root
            / "configs"
            / "quasi_real_guarded_formal_ppo_preflight_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["schema_version"],
            "quasi-real-guarded-formal-ppo-preflight-config/v1",
        )
        self.assertEqual(config["validation"]["expected_trainable_transition_count"], 684)
        self.assertEqual(config["seeds"], [0, 1, 2])
        self.assertIn("README.md", config["documentation_updates"])
        self.assertIn("does_not_publish_checkpoint", config["non_goals"])
        self.assertIn("does_not_claim_formal_training_ready", config["non_goals"])

    def _config(self) -> dict:
        return {
            "schema_version": "quasi-real-guarded-formal-ppo-preflight-config/v1",
            "seeds": [0, 1, 2],
            "training": {
                "epochs": 2,
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
                "expected_status": "quasi_real_guarded_formal_ppo_preflight_evaluated",
            },
            "output_files": {
                "summary": "quasi-real-guarded-formal-ppo-preflight-summary.json",
                "seed_summaries": "formal-preflight-seed-summaries.jsonl",
                "training_curves": "formal-preflight-training-curves.json",
                "gate_audit": "formal-preflight-gate-audit.json",
                "rollback_manifest": "formal-preflight-rollback-manifest.json",
                "readiness_validate_only": "formal-preflight-readiness-validate-only.json",
                "report": "formal-preflight-report.md",
            },
        }

    def _write_freeze_artifacts(
        self,
        *,
        trainable_count: int,
        extra_steps: list[dict] | None = None,
    ) -> None:
        miniloop_root = self.temp_dir / "miniloop"
        miniloop_root.mkdir(parents=True)
        steps_path = miniloop_root / "quasi-real-trainable-context-expansion-steps.jsonl"
        steps = [self._step(index) for index in range(trainable_count)]
        steps.extend(extra_steps or [])
        steps_path.write_text(
            "".join(json.dumps(step, sort_keys=True) + "\n" for step in steps),
            encoding="utf-8",
        )
        miniloop_summary_path = (
            miniloop_root
            / "quasi-real-guarded-ppo-iterative-miniloop-stability-summary.json"
        )
        miniloop_summary = {
            "schema_version": "quasi-real-guarded-ppo-iterative-miniloop-stability-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "input_trainable_transition_count": trainable_count,
            "unique_trainable_context_count": trainable_count,
            "seed_count": 3,
            "iteration_count": 3,
            "passed_iteration_count": 9,
            "failed_iteration_count": 0,
            "readiness_status": "quasi_real_guarded_ppo_iterative_miniloop_stability_evaluated",
            "controlled_regression_count": 0,
            "behavior_drift_count": 0,
            "steps": str(steps_path),
        }
        miniloop_summary_path.write_text(
            json.dumps(miniloop_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        manifest_path = (
            self.freeze_root
            / "quasi-real-guarded-ppo-iterative-miniloop-evidence-manifest.json"
        )
        manifest = {
            "schema_version": "quasi-real-guarded-ppo-iterative-miniloop-evidence-manifest/v1",
            "required_artifact_missing_count": 0,
            "artifacts": [
                {
                    "name": "miniloop_summary",
                    "path": str(miniloop_summary_path),
                    "required": True,
                    "exists": True,
                    "sha256": "test",
                }
            ],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        freeze_summary = {
            "schema_version": "quasi-real-guarded-ppo-iterative-miniloop-evidence-freeze-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "baseline_frozen": True,
            "input_trainable_transition_count": trainable_count,
            "unique_trainable_context_count": trainable_count,
            "seed_count": 3,
            "iteration_count": 3,
            "passed_iteration_count": 9,
            "required_artifact_missing_count": 0,
            "readiness_status": "quasi_real_guarded_ppo_iterative_miniloop_stability_evaluated",
            "controlled_regression_count": 0,
            "behavior_drift_count": 0,
            "manifest": str(manifest_path),
            "miniloop_summary": str(miniloop_summary_path),
        }
        (
            self.freeze_root
            / "quasi-real-guarded-ppo-iterative-miniloop-evidence-freeze-summary.json"
        ).write_text(json.dumps(freeze_summary, indent=2, sort_keys=True), encoding="utf-8")

    def _step(
        self,
        index: int,
        *,
        split: str = "train",
        controlled_choice_source: str = "policy",
        ppo_trainable: bool = True,
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
            "gate_reason_codes": [],
            "controlled_regression_reason_codes": [],
            "observation": {"present": True},
            "log_prob": -0.1,
            "value": 0.2,
            "reward": 1.0,
            "discounted_return": 1.0,
            "advantage": 0.8,
        }

    def _passing_seed_preflight(self, *, seed: int, trainable_steps: list[dict], **_kwargs) -> dict:
        return {
            "schema_version": "quasi-real-guarded-formal-ppo-preflight-seed-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "seed": seed,
            "optimizer_train_transition_count": len(trainable_steps),
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
            "training_curve_records": [
                {"seed": seed, "epoch": 0, "total_loss": 0.1, "approx_kl": 0.01}
            ],
            "updated_candidate_root": str(self.output_root / f"seed-{seed:02d}" / "candidate"),
        }

    def _passing_readiness(self, **_kwargs) -> dict:
        return {
            "training_readiness_status": "quasi_real_guarded_formal_ppo_preflight_evaluated",
            "training_blockers": [],
            "reason_codes": [],
        }

    def _formal_summary(self) -> dict:
        return {
            "schema_version": "quasi-real-guarded-formal-ppo-preflight-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "input_trainable_transition_count": 684,
            "optimizer_train_transition_count": 684,
            "unique_trainable_context_count": 684,
            "validation_trainable_count": 0,
            "test_trainable_count": 0,
            "fallback_trainable_count": 0,
            "non_empty_gate_reason_trainable_count": 0,
            "max_old_log_prob_abs_error": 0.0,
            "max_old_value_abs_error": 0.0,
            "loss_non_finite_count": 0,
            "non_finite_gradient_count": 0,
            "non_finite_reward_count": 0,
            "non_finite_return_count": 0,
            "non_finite_advantage_count": 0,
            "seed_count": 3,
            "passed_seed_count": 3,
            "max_abs_approx_kl": 0.01,
            "max_grad_norm_after_clip": 1.0,
            "min_parameter_l2_delta": 0.001,
            "teacher_agreement_rate": 1.0,
            "controlled_regression_count": 0,
            "controlled_safety_regression_count": 0,
            "controlled_contract_regression_count": 0,
            "controlled_path_risk_regression_count": 0,
            "controlled_source_selection_regression_count": 0,
            "rollback_manifest": "rollback.json",
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
        }
