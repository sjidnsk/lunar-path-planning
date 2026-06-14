import json
import tempfile
import unittest
from pathlib import Path


class SelectedFormalPpoCandidateMultihorizonShadowRolloutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="selected-formal-shadow-"))
        self.candidate_root = self.temp_dir / "candidate-selection"
        self.output_root = self.temp_dir / "shadow-rollout"
        self.batch_root = self.temp_dir / "batch"
        self.candidate_root.mkdir(parents=True)
        self.batch_root.mkdir(parents=True)

    def test_builds_multihorizon_shadow_rollout_from_selected_candidate(self) -> None:
        from scripts.run_selected_formal_ppo_candidate_multihorizon_shadow_rollout import (
            run_selected_formal_ppo_candidate_multihorizon_shadow_rollout,
        )

        self._write_candidate_selection_artifacts(trainable_count=60)

        result = run_selected_formal_ppo_candidate_multihorizon_shadow_rollout(
            candidate_selection_root=self.candidate_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(expected_trainable=60),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(
            result["schema_version"],
            "selected-formal-ppo-candidate-multihorizon-shadow-rollout-summary/v1",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(result["selected_seed"], 0)
        self.assertEqual(result["selected_budget"], "epochs1_lr3e-6")
        self.assertEqual(result["horizons"], [10, 20, 30])
        self.assertEqual(result["input_trainable_transition_count"], 60)
        self.assertEqual(result["shadow_trainable_transition_count"], 180)
        self.assertEqual(result["unique_trainable_context_count"], 60)
        self.assertEqual(result["per_horizon_completed_episode_count"], {"10": 6, "20": 3, "30": 2})
        self.assertEqual(result["controlled_regression_count"], 0)
        self.assertEqual(result["family_regression_count"], 0)
        self.assertEqual(result["teacher_agreement_rate"], 1.0)
        self.assertTrue(result["uses_multistep_discounted_return"])
        self.assertTrue(result["not_single_step_best_action"])
        self.assertEqual(
            result["readiness_status"],
            "selected_formal_ppo_candidate_multihorizon_shadow_rollout_evaluated",
        )
        self.assertFalse(result["runs_new_ppo_update"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])

        for filename in (
            "multihorizon-shadow-rollout-summary.json",
            "multihorizon-shadow-rollout-episodes.jsonl",
            "multihorizon-shadow-rollout-steps.jsonl",
            "multihorizon-return-audit.json",
            "multihorizon-rejection-report.json",
            "multihorizon-family-report.json",
            "multihorizon-readiness-validate-only.json",
            "multihorizon-shadow-rollout-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_rejects_split_fallback_and_gate_reason_trainable_leakage(self) -> None:
        from scripts.run_selected_formal_ppo_candidate_multihorizon_shadow_rollout import (
            run_selected_formal_ppo_candidate_multihorizon_shadow_rollout,
        )

        self._write_candidate_selection_artifacts(
            trainable_count=30,
            extra_steps=[
                self._step(200, split="validation", ppo_trainable=True),
                self._step(201, split="test", ppo_trainable=True),
                self._step(202, controlled_choice_source="source_fallback", ppo_trainable=True),
                self._step(203, gate_reason_codes=["risk_regression"], ppo_trainable=True),
            ],
        )

        result = run_selected_formal_ppo_candidate_multihorizon_shadow_rollout(
            candidate_selection_root=self.candidate_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(expected_trainable=30),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("selected_formal_ppo_shadow_split_leakage", result["reason_codes"])
        self.assertIn("selected_formal_ppo_shadow_fallback_trainable", result["reason_codes"])
        self.assertIn("selected_formal_ppo_shadow_gate_reason_trainable", result["reason_codes"])
        self.assertEqual(result["validation_trainable_count"], 1)
        self.assertEqual(result["test_trainable_count"], 1)
        self.assertEqual(result["source_fallback_trainable_count"], 1)
        self.assertEqual(result["non_empty_gate_reason_trainable_count"], 1)

    def test_config_declares_docs_outputs_and_non_goals(self) -> None:
        config_path = (
            self.repo_root
            / "configs"
            / "selected_formal_ppo_candidate_multihorizon_shadow_rollout_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["schema_version"],
            "selected-formal-ppo-candidate-multihorizon-shadow-rollout-config/v1",
        )
        self.assertEqual(config["shadow_rollout"]["horizons"], [10, 20, 30])
        self.assertEqual(config["validation"]["expected_trainable_transition_count"], 684)
        self.assertIn("multihorizon-shadow-rollout-summary.json", config["output_files"].values())
        self.assertIn("README.md", config["documentation_updates"])
        self.assertIn("docs/算法设计与系统架构报告.md", config["documentation_updates"])
        self.assertIn("does_not_run_new_ppo_update", config["non_goals"])
        self.assertIn("does_not_claim_formal_training_ready", config["non_goals"])

    def _config(self, *, expected_trainable: int) -> dict:
        return {
            "schema_version": "selected-formal-ppo-candidate-multihorizon-shadow-rollout-config/v1",
            "shadow_rollout": {"horizons": [10, 20, 30], "discount_factor": 0.99},
            "validation": {
                "expected_trainable_transition_count": expected_trainable,
                "min_teacher_agreement_rate": 0.95,
            },
            "readiness": {
                "config": "configs/policy_training_readiness_review_v1.json",
                "expected_status": "selected_formal_ppo_candidate_multihorizon_shadow_rollout_evaluated",
            },
            "output_files": {
                "summary": "multihorizon-shadow-rollout-summary.json",
                "episodes": "multihorizon-shadow-rollout-episodes.jsonl",
                "steps": "multihorizon-shadow-rollout-steps.jsonl",
                "return_audit": "multihorizon-return-audit.json",
                "rejection_report": "multihorizon-rejection-report.json",
                "family_report": "multihorizon-family-report.json",
                "readiness_validate_only": "multihorizon-readiness-validate-only.json",
                "report": "multihorizon-shadow-rollout-report.md",
            },
        }

    def _write_candidate_selection_artifacts(
        self,
        *,
        trainable_count: int,
        extra_steps: list[dict] | None = None,
    ) -> None:
        steps_path = self.candidate_root / "long-horizon-holdout-steps.jsonl"
        steps = [self._step(index) for index in range(trainable_count)]
        steps.extend(extra_steps or [])
        steps_path.write_text(
            "".join(json.dumps(step, sort_keys=True) + "\n" for step in steps),
            encoding="utf-8",
        )
        manifest_path = self.candidate_root / "selected-candidate-manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "quasi-real-guarded-formal-ppo-selected-candidate-manifest/v1",
                    "selected_seed": 0,
                    "selected_budget": "epochs1_lr3e-6",
                    "selected_candidate_root": str(self.candidate_root / "selected-candidate"),
                    "selected_candidate_from_stability_matrix": True,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                    "git_provenance": {"current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary = {
            "schema_version": "quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "input_trainable_transition_count": trainable_count,
            "long_horizon_trainable_transition_count": trainable_count,
            "unique_trainable_context_count": trainable_count,
            "eligible_candidate_count": 4,
            "selected_seed": 0,
            "selected_budget": "epochs1_lr3e-6",
            "selected_candidate_root": str(self.candidate_root / "selected-candidate"),
            "selected_candidate_from_stability_matrix": True,
            "candidate_selection_reproducible": True,
            "horizon": 10,
            "long_horizon_step_count": trainable_count,
            "completed_long_horizon_episode_count": trainable_count // 10,
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
            "controlled_regression_count": 0,
            "family_regression_count": 0,
            "teacher_agreement_rate": 1.0,
            "holdout_steps": str(steps_path),
            "candidate_manifest": str(manifest_path),
            "runs_formal_ppo_candidate_selection_long_horizon_holdout": True,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "git_provenance": {"current_matches_sources": True},
        }
        (
            self.candidate_root
            / "quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-summary.json"
        ).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

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
            "schema_version": "quasi-real-guarded-formal-ppo-long-horizon-holdout-step/v1",
            "context_id": f"context-{index:04d}",
            "scenario_id": f"scenario-{index % 3}",
            "scenario_family": f"family-{index % 3}",
            "episode_id": f"candidate-holdout-{index // 10:04d}",
            "step_index": index % 10,
            "split": split,
            "ppo_trainable": ppo_trainable,
            "controlled_choice_source": controlled_choice_source,
            "controlled_choice_detail": "policy_teacher_aligned",
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
            "training_readiness_status": "selected_formal_ppo_candidate_multihorizon_shadow_rollout_evaluated",
            "training_blockers": [],
            "reason_codes": [],
        }


if __name__ == "__main__":
    unittest.main()
