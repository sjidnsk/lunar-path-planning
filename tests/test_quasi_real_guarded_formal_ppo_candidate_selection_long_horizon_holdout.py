import json
import tempfile
import unittest
from pathlib import Path


class QuasiRealGuardedFormalPpoCandidateSelectionLongHorizonHoldoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="qreal-formal-candidate-holdout-"))
        self.stability_root = self.temp_dir / "formal-stability"
        self.output_root = self.temp_dir / "candidate-holdout"
        self.batch_root = self.temp_dir / "batch"
        self.stability_root.mkdir(parents=True)
        self.batch_root.mkdir(parents=True)

    def test_selects_reproducible_candidate_and_builds_long_horizon_holdout(self) -> None:
        from scripts.run_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout import (
            run_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout,
        )

        self._write_stability_artifacts(trainable_count=12)

        result = run_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout(
            stability_root=self.stability_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(expected_trainable=12, horizon=10),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(
            result["schema_version"],
            "quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-summary/v1",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(result["selected_seed"], 0)
        self.assertEqual(result["selected_budget"], "epochs1_lr3e-6")
        self.assertTrue(result["selected_candidate_from_stability_matrix"])
        self.assertTrue(result["candidate_selection_reproducible"])
        self.assertEqual(result["eligible_candidate_count"], 4)
        self.assertEqual(result["horizon"], 10)
        self.assertEqual(result["long_horizon_step_count"], 12)
        self.assertEqual(result["completed_long_horizon_episode_count"], 1)
        self.assertEqual(result["leftover_step_count"], 2)
        self.assertEqual(result["optimizer_train_transition_count"], 0)
        self.assertEqual(result["validation_trainable_count"], 0)
        self.assertEqual(result["test_trainable_count"], 0)
        self.assertEqual(result["controlled_regression_count"], 0)
        self.assertEqual(result["family_regression_count"], 0)
        self.assertEqual(result["teacher_agreement_rate"], 1.0)
        self.assertEqual(
            result["readiness_status"],
            "quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_evaluated",
        )
        self.assertFalse(result["runs_new_ppo_update"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])

        for filename in (
            "candidate-selection-audit.json",
            "long-horizon-holdout-episodes.jsonl",
            "long-horizon-holdout-steps.jsonl",
            "long-horizon-return-audit.json",
            "long-horizon-holdout-split-report.json",
            "long-horizon-family-report.json",
            "selected-candidate-manifest.json",
            "candidate-selection-rollback-manifest.json",
            "candidate-selection-readiness-validate-only.json",
            "candidate-selection-long-horizon-holdout-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_rejects_split_fallback_and_gate_reason_trainable_leakage(self) -> None:
        from scripts.run_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout import (
            run_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout,
        )

        self._write_stability_artifacts(
            trainable_count=10,
            extra_steps=[
                self._step(200, split="validation", ppo_trainable=True),
                self._step(201, split="test", ppo_trainable=True),
                self._step(202, controlled_choice_source="source_fallback", ppo_trainable=True),
                self._step(203, gate_reason_codes=["risk_regression"], ppo_trainable=True),
            ],
        )

        result = run_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout(
            stability_root=self.stability_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(expected_trainable=10, horizon=10),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("candidate_selection_holdout_split_leakage", result["reason_codes"])
        self.assertIn("candidate_selection_holdout_fallback_trainable", result["reason_codes"])
        self.assertIn("candidate_selection_holdout_gate_reason_trainable", result["reason_codes"])
        self.assertEqual(result["validation_trainable_count"], 1)
        self.assertEqual(result["test_trainable_count"], 1)
        self.assertEqual(result["fallback_trainable_count"], 1)
        self.assertEqual(result["non_empty_gate_reason_trainable_count"], 1)

    def test_rejects_matrix_candidate_with_regression_or_numeric_failure(self) -> None:
        from scripts.run_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout import (
            run_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout,
        )

        self._write_stability_artifacts(
            trainable_count=10,
            matrix_overrides={
                (0, "epochs1_lr3e-6"): {"controlled_regression_count": 1},
                (1, "epochs1_lr3e-6"): {"old_log_prob_max_abs_error": 2.0e-4},
                (0, "epochs2_lr1e-5"): {"non_finite_gradient_count": 1},
                (1, "epochs2_lr1e-5"): {"parameter_l2_delta": 0.0},
            },
        )

        result = run_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout(
            stability_root=self.stability_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(expected_trainable=10, horizon=10),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("candidate_selection_no_eligible_candidate", result["reason_codes"])
        self.assertEqual(result["eligible_candidate_count"], 0)
        audit = json.loads((self.output_root / "candidate-selection-audit.json").read_text(encoding="utf-8"))
        rejected_reasons = {
            reason
            for row in audit["candidate_rows"]
            for reason in row["rejection_reasons"]
        }
        self.assertIn("controlled_regression", rejected_reasons)
        self.assertIn("old_policy_reconstruction_error", rejected_reasons)
        self.assertIn("non_finite_numeric", rejected_reasons)
        self.assertIn("parameter_delta_missing", rejected_reasons)

    def test_config_declares_docs_outputs_and_non_goals(self) -> None:
        config_path = (
            self.repo_root
            / "configs"
            / "quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["schema_version"],
            "quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-config/v1",
        )
        self.assertGreaterEqual(config["holdout"]["horizon"], 10)
        self.assertEqual(config["validation"]["expected_trainable_transition_count"], 684)
        self.assertIn("candidate-selection-audit.json", config["output_files"].values())
        self.assertIn("README.md", config["documentation_updates"])
        self.assertIn("docs/算法设计与系统架构报告.md", config["documentation_updates"])
        self.assertIn("does_not_run_new_ppo_update", config["non_goals"])
        self.assertIn("does_not_claim_formal_training_ready", config["non_goals"])

    def _config(self, *, expected_trainable: int, horizon: int) -> dict:
        return {
            "schema_version": "quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-config/v1",
            "holdout": {"horizon": horizon, "discount_factor": 0.99},
            "validation": {
                "expected_trainable_transition_count": expected_trainable,
                "min_horizon": horizon,
                "max_old_log_prob_abs_error": 1.0e-4,
                "max_old_value_abs_error": 1.0e-4,
                "max_abs_approx_kl": 0.25,
                "max_grad_norm_after_clip": 1.0,
                "min_teacher_agreement_rate": 0.95,
            },
            "readiness": {
                "config": "configs/policy_training_readiness_review_v1.json",
                "expected_status": "quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_evaluated",
            },
            "output_files": {
                "summary": "quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-summary.json",
                "selection_audit": "candidate-selection-audit.json",
                "holdout_episodes": "long-horizon-holdout-episodes.jsonl",
                "holdout_steps": "long-horizon-holdout-steps.jsonl",
                "return_audit": "long-horizon-return-audit.json",
                "split_report": "long-horizon-holdout-split-report.json",
                "family_report": "long-horizon-family-report.json",
                "candidate_manifest": "selected-candidate-manifest.json",
                "rollback_manifest": "candidate-selection-rollback-manifest.json",
                "readiness_validate_only": "candidate-selection-readiness-validate-only.json",
                "report": "candidate-selection-long-horizon-holdout-report.md",
            },
        }

    def _write_stability_artifacts(
        self,
        *,
        trainable_count: int,
        extra_steps: list[dict] | None = None,
        matrix_overrides: dict[tuple[int, str], dict] | None = None,
    ) -> None:
        steps_path = self.stability_root / "steps.jsonl"
        steps = [self._step(index) for index in range(trainable_count)]
        steps.extend(extra_steps or [])
        steps_path.write_text(
            "".join(json.dumps(step, sort_keys=True) + "\n" for step in steps),
            encoding="utf-8",
        )
        baseline_path = self.stability_root / "formal-ppo-stability-baseline-manifest.json"
        baseline_path.write_text(json.dumps({"schema_version": "baseline/v1"}, indent=2), encoding="utf-8")
        rollback_path = self.stability_root / "formal-ppo-stability-rollback-manifest.json"
        rollback_path.write_text(
            json.dumps(
                {
                    "schema_version": "rollback/v1",
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "run_candidate_roots": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        matrix_path = self.stability_root / "formal-ppo-stability-matrix.jsonl"
        rows = []
        for budget in (
            {"name": "epochs1_lr3e-6", "epochs": 1, "learning_rate": 3e-6, "approx_kl": 0.001},
            {"name": "epochs2_lr1e-5", "epochs": 2, "learning_rate": 1e-5, "approx_kl": 0.01},
        ):
            for seed in (0, 1):
                row = self._matrix_row(seed=seed, budget=budget, trainable_count=trainable_count)
                row.update((matrix_overrides or {}).get((seed, budget["name"]), {}))
                rows.append(row)
        matrix_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        summary = {
            "schema_version": "quasi-real-guarded-formal-ppo-stability-holdout-validation-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "input_trainable_transition_count": trainable_count,
            "optimizer_train_transition_count": trainable_count,
            "unique_trainable_context_count": trainable_count,
            "seed_count": 2,
            "budget_count": 2,
            "run_count": 4,
            "passed_run_count": 4,
            "run_failure_count": 0,
            "controlled_regression_count": 0,
            "family_regression_count": 0,
            "teacher_agreement_rate": 1.0,
            "steps": str(steps_path),
            "stability_matrix": str(matrix_path),
            "baseline_manifest": str(baseline_path),
            "rollback_manifest": str(rollback_path),
            "runs_formal_ppo_stability_holdout_validation": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "git_provenance": {"current_matches_sources": True},
        }
        (self.stability_root / "quasi-real-guarded-formal-ppo-stability-holdout-validation-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _matrix_row(self, *, seed: int, budget: dict, trainable_count: int) -> dict:
        return {
            "schema_version": "quasi-real-guarded-formal-ppo-stability-holdout-run-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "seed": seed,
            "budget_name": budget["name"],
            "epochs": budget["epochs"],
            "learning_rate": budget["learning_rate"],
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
            "approx_kl": budget["approx_kl"] + seed * 0.0001,
            "max_grad_norm_after_clip": 0.5 + seed * 0.01,
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
            "updated_candidate_root": str(self.stability_root / budget["name"] / f"seed-{seed:02d}"),
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
        }

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
            "episode_id": f"episode-{index // 5:04d}",
            "step_index": index % 5,
            "horizon": 5,
            "split": split,
            "ppo_trainable": ppo_trainable,
            "controlled_choice_source": controlled_choice_source,
            "controlled_action_index": 0,
            "teacher_action_index": 0,
            "gate_reason_codes": gate_reason_codes or [],
            "controlled_regression_reason_codes": [],
            "observation": {"action_mask": [True], "candidate_cells": [[1, 2]]},
            "log_prob": -0.1,
            "value": 0.2,
            "reward": 1.0,
            "discounted_return": 1.0,
            "advantage": 0.8,
            "path_cost_delta": 0.0,
            "risk_delta": 0.0,
        }

    def _passing_readiness(self, **_kwargs) -> dict:
        return {
            "training_readiness_status": "quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_evaluated",
            "training_blockers": [],
            "reason_codes": [],
        }


if __name__ == "__main__":
    unittest.main()
