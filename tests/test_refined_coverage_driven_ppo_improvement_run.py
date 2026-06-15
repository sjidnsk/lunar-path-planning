import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class RefinedCoverageDrivenPpoImprovementRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        for path in (self.repo_root / "scripts", self.repo_root / "model-explorer" / "src"):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="refined-coverage-ppo-"))
        self.stage5a2_root = self.temp_dir / "stage5a2"
        self.coverage_driven_root = self.temp_dir / "coverage-driven"
        self.formal_root = self.temp_dir / "formal-training"
        self.replay_root = self.temp_dir / "post-training-replay"
        self.selected_root = self.temp_dir / "selected-candidate"
        self.base_candidate_root = self.temp_dir / "base-candidate"
        self.signal_root = self.temp_dir / "coverage-signal"
        self.performance_root = self.temp_dir / "coverage-performance"
        self.reward_root = self.temp_dir / "reward-refinement"
        self.output_root = self.temp_dir / "refined-output"
        for path in (
            self.stage5a2_root,
            self.coverage_driven_root,
            self.formal_root,
            self.replay_root,
            self.selected_root,
            self.base_candidate_root,
            self.signal_root,
            self.performance_root,
            self.reward_root,
        ):
            path.mkdir(parents=True)
        self.observation = self._observation()
        self._write_base_candidate()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_materializes_safe_better_counterfactual_as_nonzero_advantage_training_signal(self) -> None:
        from scripts.run_refined_coverage_driven_ppo_improvement_run import (
            run_refined_coverage_driven_ppo_improvement_run,
        )

        self._write_inputs(safe_better=True)

        summary = run_refined_coverage_driven_ppo_improvement_run(
            stage5a2_root=self.stage5a2_root,
            coverage_driven_root=self.coverage_driven_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            reward_refinement_root=self.reward_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["safe_better_training_pair_count"], 1)
        self.assertEqual(summary["refined_trainable_transition_count"], 1)
        self.assertEqual(summary["counterfactual_advantage_nonzero_count"], 1)
        self.assertEqual(summary["fallback_gain_contamination_count"], 0)
        self.assertEqual(summary["controlled_regression_count"], 0)
        self.assertTrue(summary["runs_new_ppo_update"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])
        self.assertNotEqual(summary["next_required_change"], "fix_refined_advantage_materialization")

        refined_rows = self._read_jsonl(
            self.output_root / "refined-coverage-ppo-batch" / "refined-trainable-transitions.jsonl"
        )
        self.assertEqual(len(refined_rows), 1)
        refined = refined_rows[0]
        self.assertEqual(refined["action_index"], 1)
        self.assertEqual(refined["teacher_action_index"], 0)
        self.assertGreater(refined["counterfactual_coverage_advantage"], 0.0)
        self.assertGreater(refined["coverage_rank_margin"], 0.0)
        self.assertGreater(refined["teacher_margin_target"], 0.0)
        self.assertTrue(math.isfinite(refined["log_prob"]))
        self.assertTrue(math.isfinite(refined["value"]))

        episodes = self._read_jsonl(
            self.output_root / "coverage-aware-ppo-batch" / "ppo-rollout-episodes.jsonl"
        )
        transition = episodes[0]["transitions"][0]
        self.assertEqual(transition["action_index"], 1)
        self.assertEqual(transition["info"]["controlled_action_index"], 1)
        self.assertEqual(transition["info"]["teacher_action_index"], 0)
        self.assertGreater(transition["info"]["counterfactual_coverage_advantage"], 0.0)
        self.assertIn("counterfactual_coverage_advantage_bonus", transition["reward_components"])

        for filename in (
            "refined-coverage-driven-ppo-improvement-run-summary.json",
            "coverage-driven-ppo-improvement-run-summary.json",
            "advantage-margin-audit.jsonl",
            "coverage-driven-ppo-update-summary.json",
            "coverage-driven-experimental-policy-candidate-metadata.json",
            "coverage-driven-ppo-replay-audit.json",
            "refined-coverage-driven-ppo-performance-metric-table.jsonl",
            "stage5a-rerun-summary.json",
            "refined-coverage-driven-ppo-improvement-run-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_fails_without_fake_zero_when_no_safe_better_pair_can_be_materialized(self) -> None:
        from scripts.run_refined_coverage_driven_ppo_improvement_run import (
            run_refined_coverage_driven_ppo_improvement_run,
        )

        self._write_inputs(safe_better=False, candidate_risk=0.50)

        summary = run_refined_coverage_driven_ppo_improvement_run(
            stage5a2_root=self.stage5a2_root,
            coverage_driven_root=self.coverage_driven_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            reward_refinement_root=self.reward_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["safe_better_training_pair_count"], 0)
        self.assertEqual(summary["counterfactual_advantage_nonzero_count"], 0)
        self.assertIn("insufficient_refined_advantage_materialization", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_refined_advantage_materialization")
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["performance_claimed"])

        audit_rows = self._read_jsonl(self.output_root / "advantage-margin-audit.jsonl")
        self.assertEqual(len(audit_rows), 1)
        self.assertIn("not_safe_better_than_teacher", audit_rows[0]["row_reason_codes"])
        self.assertIsNone(audit_rows[0]["counterfactual_coverage_advantage"])

    def _write_inputs(
        self,
        *,
        safe_better: bool,
        candidate_risk: float = 0.19,
    ) -> None:
        self._write_json(
            self.stage5a2_root / "policy-differentiating-counterfactual-coverage-rollouts-summary.json",
            {
                "schema_version": "policy-differentiating-counterfactual-coverage-rollouts-summary/v1",
                "status": "passed" if safe_better else "failed",
                "reason_codes": [] if safe_better else ["no_safe_better_than_teacher_alternatives"],
                "next_required_change": "rerun_coverage_driven_ppo_with_refined_reward_or_advantage",
                "counterfactual_coverage_rollouts": str(
                    self.stage5a2_root / "counterfactual-coverage-rollouts.jsonl"
                ),
                "candidate_coverage_overlay": str(self.stage5a2_root / "candidate-level-coverage-overlay.jsonl"),
                "safe_better_than_teacher_candidate_count": 1 if safe_better else 0,
                "fallback_gain_contamination_count": 0,
                "controlled_regression_count": 0,
                "runs_new_ppo_update": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
            },
        )
        self._write_jsonl(
            self.stage5a2_root / "candidate-level-coverage-overlay.jsonl",
            [
                self._overlay_row(
                    action=0,
                    teacher=0,
                    expected=0.02,
                    valuable=0.01,
                    risk=0.20,
                    path_cost=0.50,
                    is_teacher=True,
                    safe_better=False,
                ),
                self._overlay_row(
                    action=1,
                    teacher=0,
                    expected=0.20,
                    valuable=0.10,
                    risk=candidate_risk,
                    path_cost=0.45,
                    is_teacher=False,
                    safe_better=safe_better,
                ),
            ],
        )
        self._write_jsonl(
            self.stage5a2_root / "counterfactual-coverage-rollouts.jsonl",
            [
                {
                    **self._overlay_row(
                        action=1,
                        teacher=0,
                        expected=0.20,
                        valuable=0.10,
                        risk=candidate_risk,
                        path_cost=0.45,
                        is_teacher=False,
                        safe_better=safe_better,
                    ),
                    "schema_version": "policy-differentiating-counterfactual-coverage-row/v1",
                    "match_method": "counterfactual_candidate_expanded_cells_sidecar",
                    "source_confidence": 0.75,
                    "new_coverage_cell_count": 20,
                    "coverage_universe_cell_count": 100,
                }
            ],
        )
        self._write_old_coverage_driven_inputs()
        self._write_support_summaries()

    def _write_old_coverage_driven_inputs(self) -> None:
        log_prob, value = self._policy_log_prob_and_value(action_index=0)
        transition = {
            "observation": self._observation_payload(),
            "action_index": 0,
            "action_mask": [True, True],
            "log_prob": log_prob,
            "value": value,
            "reward": 0.12,
            "reward_components": {"teacher_skill_retention_bonus": 0.1, "coverage_gain_bonus": 0.02},
            "done": True,
            "next_observation": None,
            "info": {
                "context_id": "ctx-a",
                "episode_id": "episode-a",
                "step_index": 0,
                "scenario_id": "scenario-a",
                "scenario_family": "family-a",
                "split": "train",
                "controlled_choice_source": "policy",
                "controlled_choice_detail": "policy_teacher_aligned",
                "controlled_action_index": 0,
                "teacher_action_index": 0,
                "coverage_rate_delta": 0.02,
                "cumulative_coverage_rate_delta": 0.02,
                "final_coverage_rate": 0.02,
                "new_area_covered": 0.02,
                "valuable_area_covered": 0.01,
                "information_gain": 0.01,
                "path_cost": 0.50,
                "risk": 0.20,
                "energy_cost": 0.0,
                "actual_coverage_gain_source": "path_feedback",
                "fallback_like": False,
                "gate_reason_codes": [],
                "controlled_regression_reason_codes": [],
                "ppo_trainable": True,
            },
        }
        self._write_jsonl(
            self.coverage_driven_root / "coverage-aware-ppo-batch" / "ppo-rollout-episodes.jsonl",
            [
                {
                    "schema_version": "coverage-aware-ppo-rollout-episode/v1",
                    "episode_id": "episode-a",
                    "transitions": [transition],
                }
            ],
        )
        self._write_json(
            self.coverage_driven_root / "coverage-driven-ppo-improvement-run-summary.json",
            {
                "schema_version": "coverage-driven-ppo-improvement-run-summary/v1",
                "status": "failed",
                "reason_codes": ["no_coverage_return_improvement"],
                "next_required_change": "refine_coverage_reward_or_collect_more_policy_coverage",
                "coverage_aware_batch_episodes": str(
                    self.coverage_driven_root / "coverage-aware-ppo-batch" / "ppo-rollout-episodes.jsonl"
                ),
                "coverage_return_improvement": 0.0,
                "cumulative_coverage_rate_delta_improvement": 0.0,
                "valuable_area_covered_improvement": 0.0,
                "fallback_rate": 0.0,
                "teacher_agreement_rate": 1.0,
                "controlled_regression_count": 0,
                "base_candidate_root": str(self.base_candidate_root),
                "runs_new_ppo_update": True,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
            },
        )

    def _write_support_summaries(self) -> None:
        self._write_json(
            self.formal_root / "formal-ppo-training-run-summary.json",
            {
                "schema_version": "guarded-formal-ppo-training-run-summary/v1",
                "status": "passed",
                "seed_count": 5,
                "optimizer_train_transition_count": 684,
                "controlled_regression_count": 0,
                "performance_claimed": False,
            },
        )
        self._write_json(
            self.replay_root / "formal-ppo-post-training-stability-replay-summary.json",
            {
                "schema_version": "guarded-formal-ppo-post-training-stability-replay-summary/v1",
                "status": "passed",
                "controlled_regression_count": 0,
            },
        )
        self._write_json(
            self.selected_root / "selected-formal-ppo-candidate-promotion-preflight-summary.json",
            {
                "schema_version": "selected-formal-ppo-candidate-promotion-preflight-summary/v1",
                "status": "passed",
                "selected_seed": 0,
                "selected_candidate_root": str(self.base_candidate_root),
                "checkpoint_path": str(self.base_candidate_root / "experimental-hybrid-policy-candidate.pt"),
                "checkpoint_metadata_path": str(
                    self.base_candidate_root / "experimental-hybrid-policy-candidate-metadata.json"
                ),
                "controlled_regression_count": 0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
            },
        )
        self._write_json(
            self.signal_root / "exploration-coverage-signal-audit-summary.json",
            {
                "schema_version": "exploration-coverage-signal-audit-summary/v1",
                "status": "passed",
                "coverage_signal_status": "passed",
                "nonzero_actual_coverage_delta_count": 1,
                "fallback_coverage_gain_claimed_as_policy_gain_count": 0,
                "controlled_regression_count": 0,
            },
        )
        metric_rows = [
            self._metric_row("teacher", coverage_return=0.02, valuable=0.01),
            self._metric_row("selected_ppo_candidate", coverage_return=0.02, valuable=0.01),
            self._metric_row("source_default", coverage_return=0.01, valuable=0.005),
        ]
        self._write_json(
            self.performance_root / "exploration-coverage-performance-evaluation-summary.json",
            {
                "schema_version": "exploration-coverage-performance-evaluation-summary/v1",
                "status": "failed",
                "coverage_performance_status": "failed",
                "best_baseline_actor": "teacher",
                "best_baseline_metrics": metric_rows[0],
                "selected_metrics": metric_rows[1],
                "metric_table": str(self.performance_root / "coverage-performance-metric-table.jsonl"),
                "controlled_regression_count": 0,
                "performance_claimed": False,
            },
        )
        self._write_jsonl(self.performance_root / "coverage-performance-metric-table.jsonl", metric_rows)
        self._write_json(
            self.reward_root / "coverage-aware-reward-refinement-summary.json",
            {
                "schema_version": "coverage-aware-reward-refinement-summary/v1",
                "status": "passed",
                "reward_refinement_status": "passed",
                "source_field_missing_component_count": 0,
                "fallback_coverage_gain_claimed_as_policy_gain_count": 0,
                "controlled_regression_count": 0,
                "performance_claimed": False,
            },
        )

    def _overlay_row(
        self,
        *,
        action: int,
        teacher: int,
        expected: float,
        valuable: float,
        risk: float,
        path_cost: float,
        is_teacher: bool,
        safe_better: bool,
    ) -> dict:
        return {
            "schema_version": "candidate-level-coverage-overlay-row/v1",
            "context_id": "ctx-a",
            "episode_id": "episode-a",
            "step_index": 0,
            "scenario_id": "scenario-a",
            "scenario_family": "family-a",
            "split": "train",
            "action_index": action,
            "candidate_cell": [action, action],
            "action_mask_valid": True,
            "teacher_action_index": teacher,
            "is_teacher_action": is_teacher,
            "coverage_source_available": True,
            "expected_coverage_rate_delta": expected,
            "expected_new_coverage_area": expected,
            "information_gain": expected / 2.0,
            "valuable_coverage_proxy": valuable,
            "value": valuable,
            "path_cost": path_cost,
            "risk": risk,
            "energy_cost": 0.0,
            "fallback_like": False,
            "guard_rejected": False,
            "policy_action_accepted": True,
            "controlled_regression_reason_codes": [],
            "missing_reason_codes": [],
            "match_method": "candidate_counterfactual_path_feedback" if not is_teacher else "executed_action_actual_path_feedback",
            "source_confidence": 1.0,
            "source_execution_type": "counterfactual_candidate" if not is_teacher else "executed_policy_action",
            "safe_better_than_teacher_candidate": safe_better,
        }

    def _observation(self):
        from model_explorer.policy.features import PolicyObservation

        return PolicyObservation(
            candidate_feature_names=(
                "expected_coverage_rate_delta",
                "information_gain",
                "risk",
                "path_cost",
                "energy_cost",
                "value",
                "utility",
                "reachable",
            ),
            candidate_features=(
                (0.02, 0.01, 0.20, 0.50, 0.0, 0.01, 0.50, 1.0),
                (0.20, 0.10, 0.19, 0.45, 0.0, 0.10, 0.50, 1.0),
            ),
            global_feature_names=("coverage_rate",),
            global_features=(0.0,),
            action_mask=(True, True),
            candidate_cells=((0, 0), (1, 1)),
            candidate_missing_feature_names=((), ()),
            candidate_missing_indicator_names=(),
            candidate_missing_indicators=((), ()),
        )

    def _observation_payload(self) -> dict:
        return {
            "candidate_feature_names": list(self.observation.candidate_feature_names),
            "candidate_features": [list(item) for item in self.observation.candidate_features],
            "global_feature_names": list(self.observation.global_feature_names),
            "global_features": list(self.observation.global_features),
            "action_mask": list(self.observation.action_mask),
            "candidate_cells": [list(item) for item in self.observation.candidate_cells],
            "candidate_missing_feature_names": [
                list(item) for item in self.observation.candidate_missing_feature_names
            ],
            "candidate_missing_indicator_names": list(self.observation.candidate_missing_indicator_names),
            "candidate_missing_indicators": [
                list(item) for item in self.observation.candidate_missing_indicators
            ],
        }

    def _write_base_candidate(self) -> None:
        import torch
        from model_explorer.policy.architectures import build_policy_network

        torch.manual_seed(31)
        network = build_policy_network(None, observation=self.observation, hidden_size=8)
        checkpoint = {
            "schema_version": "controlled-hybrid-policy-candidate-checkpoint/v1",
            "experimental": True,
            "architecture": network.architecture_name,
            "model_state_dict": network.state_dict(),
            "training": {"hidden_size": 8, "seed": 31},
            "git_provenance": {"current_matches_sources": True},
        }
        torch.save(checkpoint, self.base_candidate_root / "experimental-hybrid-policy-candidate.pt")
        self._write_json(
            self.base_candidate_root / "experimental-hybrid-policy-candidate-metadata.json",
            {
                "schema_version": "controlled-hybrid-policy-candidate-checkpoint-metadata/v1",
                "experimental": True,
                "checkpoint_path": "experimental-hybrid-policy-candidate.pt",
                "architecture": network.architecture_name,
                "hidden_size": 8,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "git_provenance": {"current_matches_sources": True},
            },
        )
        self._write_json(
            self.base_candidate_root / "raw-policy-generalization-candidate-summary.json",
            {
                "schema_version": "raw-policy-generalization-candidate-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "checkpoint_path": "experimental-hybrid-policy-candidate.pt",
                "checkpoint_metadata_path": "experimental-hybrid-policy-candidate-metadata.json",
                "experimental_checkpoint": True,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
            },
        )

    def _policy_log_prob_and_value(self, *, action_index: int) -> tuple[float, float]:
        import torch
        from model_explorer.policy.architectures import build_policy_network
        from model_explorer.policy.torch_policy import observation_to_tensors

        checkpoint = torch.load(
            self.base_candidate_root / "experimental-hybrid-policy-candidate.pt",
            map_location="cpu",
            weights_only=False,
        )
        network = build_policy_network(None, observation=self.observation, hidden_size=8)
        network.load_state_dict(checkpoint["model_state_dict"])
        with torch.no_grad():
            output = network(**observation_to_tensors(self.observation))
            distribution = torch.distributions.Categorical(logits=output.masked_logits)
            log_prob = float(distribution.log_prob(torch.tensor([action_index])).item())
            value = float(output.value[0].item())
        return log_prob, value

    def _metric_row(self, actor: str, *, coverage_return: float, valuable: float) -> dict:
        return {
            "schema_version": "coverage-performance-metric-row/v1",
            "actor": actor,
            "row_count": 1,
            "episode_count": 1,
            "coverage_return": coverage_return,
            "cumulative_coverage_rate_delta": coverage_return,
            "final_coverage_rate": coverage_return,
            "new_area_covered": coverage_return,
            "valuable_area_covered": valuable,
            "information_gain": coverage_return / 2.0,
            "path_cost": 0.50,
            "risk": 0.20,
            "energy_cost": 0.0,
            "coverage_gain_per_path_cost": coverage_return / 0.50,
            "coverage_gain_per_risk": coverage_return / 0.20,
            "coverage_gain_per_energy": None,
            "accepted_policy_activation_rate": 1.0 if actor == "selected_ppo_candidate" else 0.0,
            "fallback_rate": 0.0,
            "teacher_agreement_rate": 1.0,
            "controlled_regression_count": 0,
            "fallback_coverage_gain": 0.0,
        }

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
