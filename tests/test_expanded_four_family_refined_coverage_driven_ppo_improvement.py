import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class ExpandedFourFamilyRefinedCoverageDrivenPpoImprovementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        for path in (self.repo_root / "scripts", self.repo_root / "model-explorer" / "src"):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="expanded-four-family-ppo-"))
        self.safe_better_root = self.temp_dir / "safe-better"
        self.stage5a2_root = self.temp_dir / "old-stage5a2"
        self.coverage_driven_root = self.temp_dir / "coverage-driven"
        self.formal_root = self.temp_dir / "formal-training"
        self.replay_root = self.temp_dir / "post-training-replay"
        self.selected_root = self.temp_dir / "selected-candidate"
        self.base_candidate_root = self.temp_dir / "base-candidate"
        self.signal_root = self.temp_dir / "coverage-signal"
        self.performance_root = self.temp_dir / "coverage-performance"
        self.reward_root = self.temp_dir / "reward-refinement"
        self.output_root = self.temp_dir / "expanded-output"
        for path in (
            self.safe_better_root,
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

    def test_consumes_expanded_four_family_pairs_instead_of_old_stage5a2_pairs(self) -> None:
        from scripts.run_expanded_four_family_refined_coverage_driven_ppo_improvement import (
            run_expanded_four_family_refined_coverage_driven_ppo_improvement,
        )

        self._write_support_inputs()
        self._write_old_stage5a2_single_pair()
        pairs = []
        counterfactual_rows = []
        for offset, family in enumerate(
            (
                "low_observation_count",
                "mixed_risk",
                "rim_or_steep_slope",
                "smooth_high_confidence",
            )
        ):
            for index in range(2):
                row = self._safe_better_pair(family=family, index=offset * 10 + index)
                pairs.append(row)
                counterfactual_rows.append(self._counterfactual_row(row))
        self._write_safe_better_inputs(pairs, counterfactual_rows)

        summary = run_expanded_four_family_refined_coverage_driven_ppo_improvement(
            safe_better_root=self.safe_better_root,
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
            minimum_safe_better_pair_count=8,
        )

        self.assertEqual(summary["expanded_safe_better_pair_count"], 8)
        self.assertEqual(summary["safe_better_training_pair_count"], 8)
        self.assertEqual(summary["safe_better_training_family_count"], 4)
        self.assertEqual(summary["counterfactual_advantage_nonzero_count"], 8)
        self.assertEqual(
            summary["family_safe_better_training_counts"],
            {
                "low_observation_count": 2,
                "mixed_risk": 2,
                "rim_or_steep_slope": 2,
                "smooth_high_confidence": 2,
            },
        )
        self.assertEqual(summary["old_stage5a2_safe_better_pair_count"], 1)
        self.assertFalse(summary["uses_old_stage5a2_pairs_as_training_source"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])

        refined_rows = self._read_jsonl(
            self.output_root / "refined-coverage-ppo-batch" / "refined-trainable-transitions.jsonl"
        )
        self.assertEqual(len(refined_rows), 8)
        self.assertEqual({row["scenario_family"] for row in refined_rows}, set(summary["family_safe_better_training_counts"]))
        self.assertTrue(all(row["counterfactual_coverage_advantage"] > 0.0 for row in refined_rows))
        self.assertTrue(all(row["reward_components"]["expanded_family_weight_bonus"] > 0.0 for row in refined_rows))

        synthetic_rows = self._read_jsonl(
            self.output_root
            / "expanded-four-family-compatible-coverage-driven-input"
            / "coverage-aware-ppo-batch"
            / "ppo-rollout-episodes.jsonl"
        )
        synthetic_transitions = [
            transition
            for episode in synthetic_rows
            for transition in episode["transitions"]
            if transition["info"]["scenario_family"] == "low_observation_count"
        ]
        self.assertTrue(synthetic_transitions)
        for transition in synthetic_transitions:
            action_mask = transition["observation"]["action_mask"]
            self.assertGreaterEqual(len(action_mask), transition["info"]["teacher_action_index"] + 1)
            self.assertGreaterEqual(len(action_mask), transition["action_index"] + 1)
            self.assertTrue(math.isfinite(transition["log_prob"]))
            self.assertTrue(math.isfinite(transition["value"]))

        for filename in (
            "expanded-four-family-refined-coverage-driven-ppo-improvement-summary.json",
            "expanded-four-family-source-audit.json",
            "expanded-four-family-refined-coverage-driven-ppo-improvement-report.md",
            "coverage-driven-ppo-update-summary.json",
            "coverage-driven-ppo-replay-audit.json",
            "stage5a-rerun-summary.json",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_fails_when_expanded_pairs_do_not_cover_four_training_families(self) -> None:
        from scripts.run_expanded_four_family_refined_coverage_driven_ppo_improvement import (
            run_expanded_four_family_refined_coverage_driven_ppo_improvement,
        )

        self._write_support_inputs()
        self._write_old_stage5a2_single_pair()
        pairs = [self._safe_better_pair(family="mixed_risk", index=index) for index in range(8)]
        self._write_safe_better_inputs(pairs, [self._counterfactual_row(row) for row in pairs])

        summary = run_expanded_four_family_refined_coverage_driven_ppo_improvement(
            safe_better_root=self.safe_better_root,
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
            minimum_safe_better_pair_count=8,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("family_balance_failed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "expand_safe_better_pair_generation_across_families")
        self.assertEqual(summary["safe_better_training_family_count"], 1)
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["performance_claimed"])

    def _write_support_inputs(self) -> None:
        self._write_old_coverage_driven_inputs()
        self._write_json(
            self.formal_root / "formal-ppo-training-run-summary.json",
            {
                "schema_version": "guarded-formal-ppo-training-run-summary/v1",
                "status": "passed",
                "seed_count": 5,
                "controlled_regression_count": 0,
                "performance_claimed": False,
            },
        )
        self._write_json(
            self.replay_root / "formal-ppo-post-training-stability-replay-summary.json",
            {"schema_version": "formal-replay/v1", "status": "passed", "controlled_regression_count": 0},
        )
        self._write_json(
            self.selected_root / "selected-formal-ppo-candidate-promotion-preflight-summary.json",
            {
                "schema_version": "selected-formal-ppo-candidate-promotion-preflight-summary/v1",
                "status": "passed",
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
                "fallback_coverage_gain_claimed_as_policy_gain_count": 0,
                "controlled_regression_count": 0,
            },
        )
        metric_rows = [
            self._metric_row("teacher", coverage_return=0.02, valuable=0.01),
            self._metric_row("selected_ppo_candidate", coverage_return=0.02, valuable=0.01),
        ]
        self._write_json(
            self.performance_root / "exploration-coverage-performance-evaluation-summary.json",
            {
                "schema_version": "exploration-coverage-performance-evaluation-summary/v1",
                "status": "failed",
                "coverage_performance_status": "failed",
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
                "controlled_regression_count": 0,
                "performance_claimed": False,
            },
        )

    def _write_old_stage5a2_single_pair(self) -> None:
        self._write_json(
            self.stage5a2_root / "policy-differentiating-counterfactual-coverage-rollouts-summary.json",
            {
                "schema_version": "policy-differentiating-counterfactual-coverage-rollouts-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "next_required_change": "rerun_coverage_driven_ppo_with_refined_reward_or_advantage",
                "safe_better_than_teacher_candidate_count": 1,
                "candidate_coverage_overlay": str(self.stage5a2_root / "candidate-level-coverage-overlay.jsonl"),
                "counterfactual_coverage_rollouts": str(self.stage5a2_root / "counterfactual-coverage-rollouts.jsonl"),
                "fallback_gain_contamination_count": 0,
                "controlled_regression_count": 0,
                "performance_claimed": False,
            },
        )
        old_pair = self._safe_better_pair(family="old_stage5a2_family", index=99)
        self._write_jsonl(
            self.stage5a2_root / "candidate-level-coverage-overlay.jsonl",
            [self._teacher_overlay(old_pair), self._candidate_overlay(old_pair)],
        )
        self._write_jsonl(self.stage5a2_root / "counterfactual-coverage-rollouts.jsonl", [self._counterfactual_row(old_pair)])

    def _write_safe_better_inputs(self, pairs: list[dict], counterfactual_rows: list[dict]) -> None:
        from collections import Counter

        family_counts = Counter(row["scenario_family"] for row in pairs if row.get("ppo_trainable"))
        self._write_json(
            self.safe_better_root / "safe-better-pair-expansion-summary.json",
            {
                "schema_version": "safe-better-pair-expansion-summary/v1",
                "status": "passed" if len(family_counts) >= 4 else "failed",
                "reason_codes": [] if len(family_counts) >= 4 else ["safe_better_family_count_below_threshold"],
                "next_required_change": "rerun_refined_coverage_driven_ppo_improvement",
                "safe_better_than_teacher_candidate_count": len(pairs),
                "trainable_safe_better_pair_count": len(pairs),
                "safe_better_than_teacher_family_count": len(family_counts),
                "family_safe_better_counts": dict(family_counts),
                "missing_counterfactual_source_count": 0,
                "fallback_gain_contamination_count": 0,
                "controlled_regression_count": 0,
                "runs_new_ppo_update": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
            },
        )
        self._write_jsonl(self.safe_better_root / "expanded-safe-better-pairs.jsonl", pairs)
        self._write_jsonl(
            self.safe_better_root / "expanded-counterfactual-coverage-rollouts.jsonl",
            counterfactual_rows,
        )

    def _safe_better_pair(self, *, family: str, index: int) -> dict:
        action_index = 7 if family == "low_observation_count" else 1
        teacher_index = 4 if family == "low_observation_count" else 0
        return {
            "schema_version": "safe-better-pair-expansion-row/v1",
            "context_id": f"ctx-{family}-{index}",
            "episode_id": f"episode-{family}-{index}",
            "step_index": index,
            "scenario_id": f"scenario-{family}-{index}",
            "scenario_family": family,
            "split": "train",
            "candidate_action_index": action_index,
            "teacher_action_index": teacher_index,
            "coverage_source_available": True,
            "expected_coverage_rate_delta": 0.20,
            "teacher_expected_coverage_rate_delta": 0.02,
            "expected_new_coverage_area": 0.20,
            "teacher_expected_new_coverage_area": 0.02,
            "information_gain": 0.10,
            "teacher_information_gain": 0.01,
            "valuable_coverage_proxy": 0.10,
            "teacher_valuable_coverage_proxy": 0.01,
            "path_cost": 0.45,
            "teacher_path_cost": 0.50,
            "risk": 0.19,
            "teacher_risk": 0.20,
            "energy_cost": 0.0,
            "teacher_energy_cost": 0.0,
            "coverage_advantage": 0.18,
            "valuable_coverage_advantage": 0.09,
            "source_confidence": 1.0,
            "source_path": str(self.safe_better_root / "source.json"),
            "match_method": "expanded_four_family_fixture",
            "counterfactual_coverage_match_method": "expanded_four_family_fixture",
            "counterfactual_coverage_source_path": str(self.safe_better_root / "source.json"),
            "safe_better_than_teacher_candidate": True,
            "safe_better_basis": "counterfactual_coverage_advantage_positive",
            "ppo_trainable": True,
            "diagnostic_only": False,
        }

    def _candidate_overlay(self, row: dict) -> dict:
        return {
            "schema_version": "candidate-level-coverage-overlay-row/v1",
            "context_id": row["context_id"],
            "episode_id": row["episode_id"],
            "step_index": row["step_index"],
            "scenario_id": row["scenario_id"],
            "scenario_family": row["scenario_family"],
            "split": row["split"],
            "action_index": row["candidate_action_index"],
            "candidate_cell": [row["candidate_action_index"], row["candidate_action_index"]],
            "action_mask_valid": True,
            "teacher_action_index": row["teacher_action_index"],
            "is_teacher_action": False,
            "coverage_source_available": True,
            "expected_coverage_rate_delta": row["expected_coverage_rate_delta"],
            "expected_new_coverage_area": row["expected_new_coverage_area"],
            "information_gain": row["information_gain"],
            "valuable_coverage_proxy": row["valuable_coverage_proxy"],
            "value": row["valuable_coverage_proxy"],
            "path_cost": row["path_cost"],
            "risk": row["risk"],
            "energy_cost": row["energy_cost"],
            "fallback_like": False,
            "guard_rejected": False,
            "policy_action_accepted": True,
            "controlled_regression_reason_codes": [],
            "missing_reason_codes": [],
            "match_method": row["match_method"],
            "source_confidence": 1.0,
            "source_execution_type": "counterfactual_candidate",
            "safe_better_than_teacher_candidate": True,
        }

    def _teacher_overlay(self, row: dict) -> dict:
        teacher = self._candidate_overlay(row)
        teacher.update(
            {
                "action_index": row["teacher_action_index"],
                "candidate_cell": [row["teacher_action_index"], row["teacher_action_index"]],
                "is_teacher_action": True,
                "expected_coverage_rate_delta": row["teacher_expected_coverage_rate_delta"],
                "expected_new_coverage_area": row["teacher_expected_new_coverage_area"],
                "information_gain": row["teacher_information_gain"],
                "valuable_coverage_proxy": row["teacher_valuable_coverage_proxy"],
                "value": row["teacher_valuable_coverage_proxy"],
                "path_cost": row["teacher_path_cost"],
                "risk": row["teacher_risk"],
                "energy_cost": row["teacher_energy_cost"],
                "source_execution_type": "executed_policy_action",
                "safe_better_than_teacher_candidate": False,
            }
        )
        return teacher

    def _counterfactual_row(self, row: dict) -> dict:
        return {
            **self._candidate_overlay(row),
            "schema_version": "safe-better-expanded-counterfactual-coverage-row/v1",
            "action_index": row["candidate_action_index"],
            "coverage_advantage": row["coverage_advantage"],
            "source_path": row["source_path"],
            "source_confidence": 1.0,
        }

    def _write_old_coverage_driven_inputs(self) -> None:
        rows = []
        for family in ("mixed_risk", "rim_or_steep_slope", "smooth_high_confidence"):
            for index in range(20):
                rows.append(self._old_transition(self._safe_better_pair(family=family, index=index)))
        self._write_jsonl(
            self.coverage_driven_root / "coverage-aware-ppo-batch" / "ppo-rollout-episodes.jsonl",
            [
                {
                    "schema_version": "coverage-aware-ppo-rollout-episode/v1",
                    "episode_id": transition["info"]["episode_id"],
                    "transitions": [transition],
                }
                for transition in rows
            ],
        )
        self._write_json(
            self.coverage_driven_root / "coverage-driven-ppo-improvement-run-summary.json",
            {
                "schema_version": "coverage-driven-ppo-improvement-run-summary/v1",
                "status": "failed",
                "reason_codes": ["no_coverage_return_improvement"],
                "coverage_aware_batch_episodes": str(
                    self.coverage_driven_root / "coverage-aware-ppo-batch" / "ppo-rollout-episodes.jsonl"
                ),
                "base_candidate_root": str(self.base_candidate_root),
                "controlled_regression_count": 0,
                "performance_claimed": False,
            },
        )

    def _old_transition(self, row: dict) -> dict:
        log_prob, value = self._policy_log_prob_and_value(action_index=row["teacher_action_index"])
        return {
            "observation": self._observation_payload(action_count=max(row["candidate_action_index"], row["teacher_action_index"]) + 1),
            "action_index": row["teacher_action_index"],
            "action_mask": [True] * (max(row["candidate_action_index"], row["teacher_action_index"]) + 1),
            "log_prob": log_prob,
            "value": value,
            "reward": 0.02,
            "reward_components": {"teacher_skill_retention_bonus": 0.02},
            "done": True,
            "next_observation": None,
            "info": {
                "context_id": row["context_id"],
                "episode_id": row["episode_id"],
                "step_index": row["step_index"],
                "scenario_id": row["scenario_id"],
                "scenario_family": row["scenario_family"],
                "split": "train",
                "controlled_action_index": row["teacher_action_index"],
                "teacher_action_index": row["teacher_action_index"],
                "coverage_rate_delta": row["teacher_expected_coverage_rate_delta"],
                "cumulative_coverage_rate_delta": row["teacher_expected_coverage_rate_delta"],
                "final_coverage_rate": row["teacher_expected_coverage_rate_delta"],
                "new_area_covered": row["teacher_expected_new_coverage_area"],
                "valuable_area_covered": row["teacher_valuable_coverage_proxy"],
                "information_gain": row["teacher_information_gain"],
                "path_cost": row["teacher_path_cost"],
                "risk": row["teacher_risk"],
                "energy_cost": row["teacher_energy_cost"],
                "fallback_like": False,
                "gate_reason_codes": [],
                "controlled_regression_reason_codes": [],
                "ppo_trainable": True,
            },
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
            candidate_features=tuple((0.02, 0.01, 0.20, 0.50, 0.0, 0.01, 0.50, 1.0) for _ in range(8)),
            global_feature_names=("coverage_rate",),
            global_features=(0.0,),
            action_mask=tuple(True for _ in range(8)),
            candidate_cells=tuple((index, index) for index in range(8)),
            candidate_missing_feature_names=tuple(() for _ in range(8)),
            candidate_missing_indicator_names=(),
            candidate_missing_indicators=tuple(() for _ in range(8)),
        )

    def _observation_payload(self, *, action_count: int = 8) -> dict:
        return {
            "candidate_feature_names": list(self.observation.candidate_feature_names),
            "candidate_features": [list(item) for item in self.observation.candidate_features[:action_count]],
            "global_feature_names": list(self.observation.global_feature_names),
            "global_features": list(self.observation.global_features),
            "action_mask": [True] * action_count,
            "candidate_cells": [list(item) for item in self.observation.candidate_cells[:action_count]],
            "candidate_missing_feature_names": [
                list(item) for item in self.observation.candidate_missing_feature_names[:action_count]
            ],
            "candidate_missing_indicator_names": list(self.observation.candidate_missing_indicator_names),
            "candidate_missing_indicators": [
                list(item) for item in self.observation.candidate_missing_indicators[:action_count]
            ],
        }

    def _write_base_candidate(self) -> None:
        import torch
        from model_explorer.policy.architectures import build_policy_network

        torch.manual_seed(37)
        network = build_policy_network(None, observation=self.observation, hidden_size=8)
        checkpoint = {
            "schema_version": "controlled-hybrid-policy-candidate-checkpoint/v1",
            "experimental": True,
            "architecture": network.architecture_name,
            "model_state_dict": network.state_dict(),
            "training": {"hidden_size": 8, "seed": 37},
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
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
