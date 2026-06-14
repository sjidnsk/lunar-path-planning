import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class SelectedFormalPpoCandidatePromotionPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="selected-formal-promotion-"))
        self.shadow_root = self.temp_dir / "multihorizon"
        self.candidate_root = self.temp_dir / "selected-candidate"
        self.output_root = self.temp_dir / "promotion-preflight"
        self.batch_root = self.temp_dir / "batch"
        self.shadow_root.mkdir(parents=True)
        self.candidate_root.mkdir(parents=True)
        self.batch_root.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_builds_promotion_preflight_manifest_and_inference_audit(self) -> None:
        from scripts.run_selected_formal_ppo_candidate_promotion_preflight import (
            run_selected_formal_ppo_candidate_promotion_preflight,
        )

        steps_path = self._write_candidate_and_shadow_artifacts(step_count=70)

        result = run_selected_formal_ppo_candidate_promotion_preflight(
            multihorizon_root=self.shadow_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(expected_trainable=70, min_inference_audit=64),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(
            result["schema_version"],
            "selected-formal-ppo-candidate-promotion-preflight-summary/v1",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(result["selected_seed"], 0)
        self.assertEqual(result["selected_budget"], "epochs1_lr3e-6")
        self.assertEqual(result["selected_candidate_root"], str(self.candidate_root))
        self.assertEqual(result["multihorizon_steps"], str(steps_path))
        self.assertTrue(result["checkpoint_load_passed"])
        self.assertEqual(result["inference_audit_count"], 64)
        self.assertEqual(result["invalid_action_mask_count"], 0)
        self.assertEqual(result["missing_observation_count"], 0)
        self.assertEqual(result["non_finite_logits_count"], 0)
        self.assertEqual(result["non_finite_log_prob_count"], 0)
        self.assertEqual(result["non_finite_value_count"], 0)
        self.assertLessEqual(result["log_prob_reconstruction_max_abs_error"], 1.0e-4)
        self.assertLessEqual(result["value_reconstruction_max_abs_error"], 1.0e-4)
        self.assertEqual(len(result["checkpoint_sha256"]), 64)
        self.assertGreater(result["checkpoint_size_bytes"], 0)
        self.assertEqual(result["controlled_regression_count"], 0)
        self.assertEqual(result["family_regression_count"], 0)
        self.assertEqual(result["teacher_agreement_rate"], 1.0)
        self.assertTrue(result["rollback_audit_passed"])
        self.assertFalse(result["runs_new_ppo_update"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])
        self.assertEqual(
            result["readiness_status"],
            "selected_formal_ppo_candidate_promotion_preflight_evaluated",
        )

        for filename in (
            "selected-formal-ppo-candidate-promotion-preflight-summary.json",
            "promotion-candidate-manifest.json",
            "checkpoint-hash-audit.json",
            "checkpoint-load-inference-audit.json",
            "rollback-audit.json",
            "promotion-preflight-readiness-validate-only.json",
            "promotion-preflight-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_checkpoint_blocks_promotion_preflight(self) -> None:
        from scripts.run_selected_formal_ppo_candidate_promotion_preflight import (
            run_selected_formal_ppo_candidate_promotion_preflight,
        )

        self._write_candidate_and_shadow_artifacts(step_count=70)
        (self.candidate_root / "experimental-hybrid-policy-candidate.pt").unlink()

        result = run_selected_formal_ppo_candidate_promotion_preflight(
            multihorizon_root=self.shadow_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(expected_trainable=70, min_inference_audit=64),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("promotion_preflight_checkpoint_missing", result["reason_codes"])
        self.assertFalse(result["checkpoint_load_passed"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])

    def test_missing_candidate_progress_log_blocks_promotion_preflight(self) -> None:
        from scripts.run_selected_formal_ppo_candidate_promotion_preflight import (
            run_selected_formal_ppo_candidate_promotion_preflight,
        )

        self._write_candidate_and_shadow_artifacts(step_count=70)
        (self.candidate_root / "training-progress-events.jsonl").unlink()

        result = run_selected_formal_ppo_candidate_promotion_preflight(
            multihorizon_root=self.shadow_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(expected_trainable=70, min_inference_audit=64),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("promotion_preflight_candidate_artifacts_missing", result["reason_codes"])

    def test_shadow_log_prob_value_drift_is_reported_not_blocking_when_inference_is_finite(self) -> None:
        from scripts.run_selected_formal_ppo_candidate_promotion_preflight import (
            run_selected_formal_ppo_candidate_promotion_preflight,
        )

        steps_path = self._write_candidate_and_shadow_artifacts(step_count=70)
        rows = [
            json.loads(line)
            for line in steps_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            row["log_prob"] = float(row["log_prob"]) + 0.01
            row["value"] = float(row["value"]) - 0.01
        steps_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

        result = run_selected_formal_ppo_candidate_promotion_preflight(
            multihorizon_root=self.shadow_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(expected_trainable=70, min_inference_audit=64),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertGreater(result["log_prob_reconstruction_max_abs_error"], 1.0e-4)
        self.assertGreater(result["value_reconstruction_max_abs_error"], 1.0e-4)
        self.assertTrue(result["reconstruction_difference_explained"])

    def test_config_declares_docs_outputs_and_non_goals(self) -> None:
        config_path = (
            self.repo_root
            / "configs"
            / "selected_formal_ppo_candidate_promotion_preflight_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["schema_version"],
            "selected-formal-ppo-candidate-promotion-preflight-config/v1",
        )
        self.assertEqual(config["validation"]["expected_trainable_transition_count"], 684)
        self.assertEqual(config["validation"]["min_inference_audit_count"], 64)
        self.assertIn(
            "selected-formal-ppo-candidate-promotion-preflight-summary.json",
            config["output_files"].values(),
        )
        self.assertIn("README.md", config["documentation_updates"])
        self.assertIn("docs/算法设计与系统架构报告.md", config["documentation_updates"])
        self.assertIn("does_not_run_new_ppo_update", config["non_goals"])
        self.assertIn("does_not_claim_formal_training_ready", config["non_goals"])

    def _config(self, *, expected_trainable: int, min_inference_audit: int) -> dict:
        return {
            "schema_version": "selected-formal-ppo-candidate-promotion-preflight-config/v1",
            "validation": {
                "expected_trainable_transition_count": expected_trainable,
                "min_inference_audit_count": min_inference_audit,
                "max_log_prob_abs_error": 1.0e-4,
                "max_value_abs_error": 1.0e-4,
                "min_teacher_agreement_rate": 0.95,
            },
            "readiness": {
                "config": "configs/policy_training_readiness_review_v1.json",
                "expected_status": "selected_formal_ppo_candidate_promotion_preflight_evaluated",
            },
            "input_files": {
                "multihorizon_summary": "multihorizon-shadow-rollout-summary.json",
                "checkpoint": "experimental-hybrid-policy-candidate.pt",
                "checkpoint_metadata": "experimental-hybrid-policy-candidate-metadata.json",
                "candidate_summary": "raw-policy-generalization-candidate-summary.json",
                "limited_ppo_summary": "limited-ppo-update-smoke-summary.json",
                "diagnostics": "limited-ppo-update-diagnostics.json",
                "training_curves": "limited-ppo-update-training-curves.json",
                "progress_events": "training-progress-events.jsonl",
                "progress_summary": "training-progress-summary.json",
            },
            "output_files": {
                "summary": "selected-formal-ppo-candidate-promotion-preflight-summary.json",
                "manifest": "promotion-candidate-manifest.json",
                "checkpoint_hash_audit": "checkpoint-hash-audit.json",
                "inference_audit": "checkpoint-load-inference-audit.json",
                "rollback_audit": "rollback-audit.json",
                "readiness_validate_only": "promotion-preflight-readiness-validate-only.json",
                "report": "promotion-preflight-report.md",
            },
        }

    def _write_candidate_and_shadow_artifacts(self, *, step_count: int) -> Path:
        sys.path.insert(0, str(self.repo_root / "model-explorer" / "src"))
        import torch
        from model_explorer.policy.architectures import build_policy_network_from_metadata
        from model_explorer.policy.rollout_io import _observation_from_dict
        from model_explorer.policy.torch_policy import observation_to_tensors

        torch.manual_seed(0)
        network = build_policy_network_from_metadata(
            "mlp_v1",
            candidate_feature_count=15,
            global_feature_count=8,
            missing_indicator_count=8,
            hidden_size=16,
            architecture_config={"hidden_dim": 16, "dropout": 0.0},
        )
        checkpoint_path = self.candidate_root / "experimental-hybrid-policy-candidate.pt"
        torch.save(
            {
                "schema_version": "controlled-hybrid-policy-candidate-checkpoint/v1",
                "experimental": True,
                "architecture": "mlp_v1",
                "architecture_config": {"hidden_dim": 16, "dropout": 0.0},
                "model_state_dict": network.state_dict(),
            },
            checkpoint_path,
        )

        steps = []
        network.eval()
        with torch.no_grad():
            for index in range(step_count):
                observation = self._observation(index)
                tensors = observation_to_tensors(_observation_from_dict(observation), device="cpu")
                output = network(**tensors)
                action_index = index % 3
                distribution = torch.distributions.Categorical(logits=output.masked_logits)
                log_prob = float(distribution.log_prob(torch.tensor([action_index])).item())
                value = float(output.value[0].item())
                steps.append(
                    {
                        "schema_version": "selected-formal-ppo-candidate-multihorizon-shadow-step/v1",
                        "context_id": f"context-{index:04d}",
                        "scenario_id": f"scenario-{index % 5}",
                        "scenario_family": f"family-{index % 4}",
                        "split": "train",
                        "ppo_trainable": True,
                        "shadow_trainable": True,
                        "controlled_choice_source": "policy",
                        "controlled_choice_detail": "policy_teacher_aligned",
                        "controlled_action_index": action_index,
                        "teacher_action_index": action_index,
                        "gate_reason_codes": [],
                        "controlled_regression_reason_codes": [],
                        "observation": observation,
                        "log_prob": log_prob,
                        "value": value,
                        "reward": 1.0,
                        "discounted_return": 1.0,
                        "advantage": 1.0 - value,
                        "shadow_discounted_return": 1.0,
                        "shadow_advantage": 1.0 - value,
                    }
                )
        steps_path = self.shadow_root / "multihorizon-shadow-rollout-steps.jsonl"
        steps_path.write_text(
            "".join(json.dumps(step, sort_keys=True) + "\n" for step in steps),
            encoding="utf-8",
        )
        summary = self._multihorizon_summary(step_count=step_count, steps_path=steps_path)
        (self.shadow_root / "multihorizon-shadow-rollout-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self._write_candidate_metadata_and_summaries(checkpoint_path, sample_count=step_count)
        return steps_path

    def _write_candidate_metadata_and_summaries(self, checkpoint_path: Path, *, sample_count: int) -> None:
        metadata_path = self.candidate_root / "experimental-hybrid-policy-candidate-metadata.json"
        metadata = {
            "schema_version": "controlled-hybrid-policy-candidate-checkpoint-metadata/v1",
            "experimental": True,
            "checkpoint_path": str(checkpoint_path),
            "architecture": "mlp_v1",
            "architecture_config": {"hidden_dim": 16, "dropout": 0.0},
            "sample_count": sample_count,
            "ppo_update_sample_count": sample_count,
            "epochs": 1,
            "seed": 0,
            "hidden_size": 16,
            "learning_rate": 0.000003,
            "clip_ratio": 0.2,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        candidate_summary = {
            "schema_version": "raw-policy-generalization-candidate-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_metadata_path": str(metadata_path),
            "experimental_checkpoint": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "candidate_training_status": "passed",
            "train_pair_count": sample_count,
        }
        (self.candidate_root / "raw-policy-generalization-candidate-summary.json").write_text(
            json.dumps(candidate_summary, indent=2),
            encoding="utf-8",
        )
        limited_summary = {
            "schema_version": "limited-ppo-update-smoke-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "optimizer_train_transition_count": sample_count,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
        }
        (self.candidate_root / "limited-ppo-update-smoke-summary.json").write_text(
            json.dumps(limited_summary, indent=2),
            encoding="utf-8",
        )
        (self.candidate_root / "limited-ppo-update-diagnostics.json").write_text(
            json.dumps({"status": "passed"}, indent=2),
            encoding="utf-8",
        )
        (self.candidate_root / "limited-ppo-update-training-curves.json").write_text(
            json.dumps({"epochs": 1, "losses": []}, indent=2),
            encoding="utf-8",
        )
        (self.candidate_root / "training-progress-events.jsonl").write_text(
            json.dumps({"stage": "ppo_update", "status": "complete"}) + "\n",
            encoding="utf-8",
        )
        (self.candidate_root / "training-progress-summary.json").write_text(
            json.dumps({"status": "complete"}, indent=2),
            encoding="utf-8",
        )

    def _multihorizon_summary(self, *, step_count: int, steps_path: Path) -> dict:
        return {
            "schema_version": "selected-formal-ppo-candidate-multihorizon-shadow-rollout-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "selected_seed": 0,
            "selected_budget": "epochs1_lr3e-6",
            "selected_candidate_root": str(self.candidate_root),
            "selected_candidate_from_candidate_selection": True,
            "horizons": [10, 20, 30],
            "input_trainable_transition_count": step_count,
            "shadow_trainable_transition_count": step_count * 3,
            "unique_trainable_context_count": step_count,
            "teacher_agreement_rate": 1.0,
            "controlled_regression_count": 0,
            "family_regression_count": 0,
            "controlled_safety_regression_count": 0,
            "controlled_contract_regression_count": 0,
            "controlled_path_risk_regression_count": 0,
            "controlled_source_selection_regression_count": 0,
            "steps": str(steps_path),
            "candidate_manifest": str(self.shadow_root / "stale-selected-candidate-manifest.json"),
            "runs_multihorizon_shadow_rollout": True,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "git_provenance": {"current_matches_sources": True},
        }

    def _observation(self, index: int) -> dict:
        cells = [[index % 50, 0], [index % 50, 1], [index % 50, 2]]
        return {
            "candidate_feature_names": [
                "cell_x",
                "cell_y",
                "relative_dx",
                "relative_dy",
                "relative_distance",
                "utility",
                "reachable",
                "expected_coverage_rate_delta",
                "expected_new_coverage_area",
                "information_gain",
                "confidence_gain",
                "value",
                "risk",
                "path_cost",
                "energy_cost",
            ],
            "candidate_features": [
                [0.1 + row * 0.01, 0.2, 0.1, 0.2, 0.3, 0.4 + row * 0.01, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.0]
                for row in range(3)
            ],
            "global_feature_names": [
                "grid_width",
                "grid_height",
                "grid_resolution",
                "passable_ratio",
                "violation_count",
                "coverage_rate",
                "step_index",
                "remaining_steps",
            ],
            "global_features": [0.02, 0.02, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "action_mask": [True, True, True],
            "candidate_cells": cells,
            "candidate_missing_feature_names": [[], [], []],
            "candidate_missing_indicator_names": [
                "expected_coverage_rate_delta_missing",
                "expected_new_coverage_area_missing",
                "information_gain_missing",
                "confidence_gain_missing",
                "value_missing",
                "risk_missing",
                "path_cost_missing",
                "energy_cost_missing",
            ],
            "candidate_missing_indicators": [[0.0] * 8, [0.0] * 8, [0.0] * 8],
        }

    def _passing_readiness(self, **_kwargs) -> dict:
        return {
            "training_readiness_status": "selected_formal_ppo_candidate_promotion_preflight_evaluated",
            "training_blockers": [],
            "reason_codes": [],
        }


if __name__ == "__main__":
    unittest.main()
