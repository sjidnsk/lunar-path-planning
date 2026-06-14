import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class ReturnAlignedGuardedPpoUpdateSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        for path in (self.repo_root / "scripts", self.repo_root / "model-explorer" / "src"):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="return-aligned-ppo-smoke-"))
        self.source_root = self.temp_dir / "source"
        self.base_candidate_root = self.temp_dir / "base-candidate"
        self.guarded_collector_root = self.temp_dir / "guarded-collector"
        self.return_aligned_root = self.temp_dir / "return-aligned"
        self.output_root = self.temp_dir / "output"
        for path in (
            self.source_root,
            self.base_candidate_root,
            self.guarded_collector_root,
            self.return_aligned_root,
        ):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_authorized_return_aligned_train_rows_enter_optimizer_with_multistep_values(self) -> None:
        from model_explorer.policy.rollout_io import read_rollout_episodes
        from scripts.run_return_aligned_guarded_ppo_update_smoke import (
            run_return_aligned_guarded_ppo_update_smoke,
        )

        observation = self._observation()
        self._write_base_candidate(observation=observation)
        transitions = [
            self._transition_payload(
                observation=observation,
                action_index=1,
                reward=1.0,
                context_id="ctx-train-a",
                episode_id="ep-train",
                step_index=0,
                split="train",
                controlled_choice_source="policy",
            ),
            self._transition_payload(
                observation=observation,
                action_index=0,
                reward=1.0,
                context_id="ctx-train-b",
                episode_id="ep-train",
                step_index=1,
                split="train",
                controlled_choice_source="policy",
            ),
        ]
        self._write_guarded_collector_episodes(transitions)
        self._write_guarded_collector_summary(trainable_count=2)
        self._write_return_aligned_records(
            [
                self._return_aligned_row(transitions[0], discounted_return=3.5, advantage=3.25, ppo_trainable=True),
                self._return_aligned_row(transitions[1], discounted_return=2.0, advantage=1.75, ppo_trainable=True),
            ],
            trainable_count=2,
        )

        summary = run_return_aligned_guarded_ppo_update_smoke(
            source_root=self.source_root,
            base_candidate_root=self.base_candidate_root,
            guarded_collector_root=self.guarded_collector_root,
            return_aligned_root=self.return_aligned_root,
            output_root=self.output_root,
            config=self._config(expected_count=2),
            repo_root=self.repo_root,
            post_update_runner=self._passing_post_update_runner(),
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["schema_version"], "return-aligned-guarded-ppo-update-smoke-summary/v1")
        self.assertEqual(summary["input_return_aligned_trainable_transition_count"], 2)
        self.assertEqual(summary["optimizer_train_transition_count"], 2)
        self.assertEqual(summary["optimizer_transition_split_counts"], {"train": 2})
        self.assertEqual(summary["optimizer_transition_source_counts"], {"policy": 2})
        self.assertEqual(summary["validation_test_optimizer_transition_count"], 0)
        self.assertTrue(summary["uses_multistep_discounted_return"])
        self.assertTrue(summary["not_single_step_best_action"])
        self.assertEqual(summary["optimizer_return_source"], "return_aligned_collector")
        materialized = read_rollout_episodes(self.output_root / "optimizer-input" / "ppo-rollout-episodes.jsonl")
        joined = [transition for episode in materialized for transition in episode.transitions]
        self.assertEqual([transition.info.extra["ppo_return"] for transition in joined], [3.5, 2.0])
        self.assertEqual([transition.info.extra["ppo_advantage"] for transition in joined], [3.25, 1.75])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])
        self.assertFalse(summary["formal_training_ready_claimed"])

    def test_diagnostic_return_aligned_rows_do_not_enter_optimizer(self) -> None:
        from scripts.run_return_aligned_guarded_ppo_update_smoke import (
            run_return_aligned_guarded_ppo_update_smoke,
        )

        observation = self._observation()
        self._write_base_candidate(observation=observation)
        transitions = [
            self._transition_payload(
                observation=observation,
                action_index=1,
                reward=1.0,
                context_id="ctx-train",
                episode_id="ep-train",
                step_index=0,
                split="train",
                controlled_choice_source="policy",
            ),
            self._transition_payload(
                observation=observation,
                action_index=0,
                reward=1.0,
                context_id="ctx-val",
                episode_id="ep-val",
                step_index=0,
                split="validation",
                controlled_choice_source="policy",
            ),
            self._transition_payload(
                observation=observation,
                action_index=0,
                reward=0.0,
                context_id="ctx-fallback",
                episode_id="ep-fallback",
                step_index=0,
                split="train",
                controlled_choice_source="source_fallback",
            ),
            self._transition_payload(
                observation=observation,
                action_index=0,
                reward=0.0,
                context_id="ctx-gated",
                episode_id="ep-gated",
                step_index=0,
                split="train",
                controlled_choice_source="policy",
                gate_reason_codes=["path_cost_regression"],
            ),
        ]
        self._write_guarded_collector_episodes(transitions)
        self._write_guarded_collector_summary(trainable_count=1)
        self._write_return_aligned_records(
            [
                self._return_aligned_row(transitions[0], discounted_return=1.0, advantage=0.75, ppo_trainable=True),
                self._return_aligned_row(transitions[1], discounted_return=1.0, advantage=0.75, ppo_trainable=False),
                self._return_aligned_row(transitions[2], discounted_return=0.0, advantage=0.0, ppo_trainable=False),
                self._return_aligned_row(
                    transitions[3],
                    discounted_return=0.0,
                    advantage=0.0,
                    ppo_trainable=False,
                    rejection_reason_codes=["path_cost_regression"],
                ),
            ],
            trainable_count=1,
            diagnostic_count=3,
        )

        summary = run_return_aligned_guarded_ppo_update_smoke(
            source_root=self.source_root,
            base_candidate_root=self.base_candidate_root,
            guarded_collector_root=self.guarded_collector_root,
            return_aligned_root=self.return_aligned_root,
            output_root=self.output_root,
            config=self._config(expected_count=1),
            repo_root=self.repo_root,
            post_update_runner=self._passing_post_update_runner(),
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["optimizer_train_transition_count"], 1)
        self.assertEqual(summary["optimizer_transition_split_counts"], {"train": 1})
        self.assertEqual(summary["validation_test_optimizer_transition_count"], 0)
        self.assertEqual(summary["source_fallback_optimizer_transition_count"], 0)
        self.assertEqual(summary["non_empty_gate_reason_optimizer_transition_count"], 0)

    def test_non_finite_return_or_advantage_blocks_update(self) -> None:
        from scripts.run_return_aligned_guarded_ppo_update_smoke import (
            run_return_aligned_guarded_ppo_update_smoke,
        )

        observation = self._observation()
        self._write_base_candidate(observation=observation)
        transition = self._transition_payload(
            observation=observation,
            action_index=1,
            reward=1.0,
            context_id="ctx-bad-return",
            episode_id="ep-bad-return",
            step_index=0,
            split="train",
            controlled_choice_source="policy",
        )
        self._write_guarded_collector_episodes([transition])
        self._write_guarded_collector_summary(trainable_count=1)
        self._write_return_aligned_records(
            [self._return_aligned_row(transition, discounted_return=float("nan"), advantage=0.0, ppo_trainable=True)],
            trainable_count=1,
            non_finite_return_count=1,
        )

        summary = run_return_aligned_guarded_ppo_update_smoke(
            source_root=self.source_root,
            base_candidate_root=self.base_candidate_root,
            guarded_collector_root=self.guarded_collector_root,
            return_aligned_root=self.return_aligned_root,
            output_root=self.output_root,
            config=self._config(expected_count=1),
            repo_root=self.repo_root,
            post_update_runner=self._passing_post_update_runner(),
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("return_aligned_ppo_update_input_contract_invalid", summary["reason_codes"])
        self.assertEqual(summary["optimizer_train_transition_count"], 0)

    def test_missing_post_update_gate_summary_blocks_smoke(self) -> None:
        from scripts.run_return_aligned_guarded_ppo_update_smoke import (
            run_return_aligned_guarded_ppo_update_smoke,
        )

        observation = self._observation()
        self._write_base_candidate(observation=observation)
        transition = self._transition_payload(
            observation=observation,
            action_index=1,
            reward=1.0,
            context_id="ctx-train",
            episode_id="ep-train",
            step_index=0,
            split="train",
            controlled_choice_source="policy",
        )
        self._write_guarded_collector_episodes([transition])
        self._write_guarded_collector_summary(trainable_count=1)
        self._write_return_aligned_records(
            [self._return_aligned_row(transition, discounted_return=1.0, advantage=0.75, ppo_trainable=True)],
            trainable_count=1,
        )

        summary = run_return_aligned_guarded_ppo_update_smoke(
            source_root=self.source_root,
            base_candidate_root=self.base_candidate_root,
            guarded_collector_root=self.guarded_collector_root,
            return_aligned_root=self.return_aligned_root,
            output_root=self.output_root,
            config=self._config(expected_count=1),
            repo_root=self.repo_root,
            post_update_runner=lambda _context: {"quasi_real_teacher_following": {"status": "passed", "teacher_agreement_rate": 1.0}},
        )

        self.assertEqual(summary["status"], "failed")
        self.assertFalse(summary["post_update_gates_evaluated"])
        self.assertIn("return_aligned_ppo_update_post_update_gate_regression", summary["reason_codes"])

    def test_post_update_controlled_regression_blocks_smoke(self) -> None:
        from scripts.run_return_aligned_guarded_ppo_update_smoke import (
            run_return_aligned_guarded_ppo_update_smoke,
        )

        observation = self._observation()
        self._write_base_candidate(observation=observation)
        transition = self._transition_payload(
            observation=observation,
            action_index=1,
            reward=1.0,
            context_id="ctx-train",
            episode_id="ep-train",
            step_index=0,
            split="train",
            controlled_choice_source="policy",
        )
        self._write_guarded_collector_episodes([transition])
        self._write_guarded_collector_summary(trainable_count=1)
        self._write_return_aligned_records(
            [self._return_aligned_row(transition, discounted_return=1.0, advantage=0.75, ppo_trainable=True)],
            trainable_count=1,
        )

        post = self._passing_post_update_payload()
        post["sequential_canary"]["controlled_path_cost_regression_count"] = 1

        summary = run_return_aligned_guarded_ppo_update_smoke(
            source_root=self.source_root,
            base_candidate_root=self.base_candidate_root,
            guarded_collector_root=self.guarded_collector_root,
            return_aligned_root=self.return_aligned_root,
            output_root=self.output_root,
            config=self._config(expected_count=1),
            repo_root=self.repo_root,
            post_update_runner=lambda _context: post,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["post_update_controlled_regression_count"], 1)
        self.assertIn("return_aligned_ppo_update_post_update_gate_regression", summary["reason_codes"])

    def test_readiness_accepts_passed_return_aligned_update_smoke_summary(self) -> None:
        from scripts.run_policy_training_readiness_review import (
            _return_aligned_guarded_ppo_update_smoke_readiness,
        )

        readiness = _return_aligned_guarded_ppo_update_smoke_readiness(self._wrapper_summary(status="passed", reason_codes=[]))

        self.assertTrue(readiness["present"])
        self.assertTrue(readiness["completed"])
        self.assertEqual(readiness["training_blockers"], [])
        self.assertEqual(readiness["optimizer_train_transition_count"], 30)

    def test_readiness_requires_post_update_gates(self) -> None:
        from scripts.run_policy_training_readiness_review import (
            _return_aligned_guarded_ppo_update_smoke_readiness,
        )

        summary = self._wrapper_summary(status="passed", reason_codes=[])
        summary["post_update_gates_evaluated"] = False

        readiness = _return_aligned_guarded_ppo_update_smoke_readiness(summary)

        self.assertFalse(readiness["completed"])
        self.assertIn("return_aligned_ppo_update_post_update_gate_regression", readiness["training_blockers"])

    def _observation(self):
        from model_explorer.policy.features import PolicyObservation

        return PolicyObservation(
            candidate_feature_names=("path_cost", "risk"),
            candidate_features=((1.0, 0.2), (0.4, 0.05)),
            global_feature_names=("step_index",),
            global_features=(0.0,),
            action_mask=(True, True),
            candidate_cells=((4, 6), (5, 6)),
            candidate_missing_feature_names=((), ()),
            candidate_missing_indicator_names=(),
            candidate_missing_indicators=((), ()),
        )

    def _write_base_candidate(self, *, observation) -> None:
        import torch
        from model_explorer.policy.architectures import build_policy_network

        torch.manual_seed(7)
        network = build_policy_network(None, observation=observation, hidden_size=8)
        checkpoint = {
            "schema_version": "controlled-hybrid-policy-candidate-checkpoint/v1",
            "experimental": True,
            "architecture": network.architecture_name,
            "model_state_dict": network.state_dict(),
            "training": {"hidden_size": 8, "seed": 7},
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

    def _transition_payload(
        self,
        *,
        observation,
        action_index: int,
        reward: float,
        context_id: str,
        episode_id: str,
        step_index: int,
        split: str,
        controlled_choice_source: str,
        gate_reason_codes: list[str] | None = None,
    ) -> dict:
        import torch
        from model_explorer.policy.architectures import build_policy_network
        from model_explorer.policy.torch_policy import observation_to_tensors

        checkpoint = torch.load(
            self.base_candidate_root / "experimental-hybrid-policy-candidate.pt",
            map_location="cpu",
            weights_only=False,
        )
        network = build_policy_network(None, observation=observation, hidden_size=8)
        network.load_state_dict(checkpoint["model_state_dict"])
        with torch.no_grad():
            output = network(**observation_to_tensors(observation))
            distribution = torch.distributions.Categorical(logits=output.masked_logits)
            log_prob = float(distribution.log_prob(torch.tensor([action_index])).item())
            value = float(output.value[0].item())
        return {
            "observation": {
                "candidate_feature_names": list(observation.candidate_feature_names),
                "candidate_features": [list(row) for row in observation.candidate_features],
                "global_feature_names": list(observation.global_feature_names),
                "global_features": list(observation.global_features),
                "action_mask": list(observation.action_mask),
                "candidate_cells": [list(cell) for cell in observation.candidate_cells],
                "candidate_missing_feature_names": [list(row) for row in observation.candidate_missing_feature_names],
                "candidate_missing_indicator_names": list(observation.candidate_missing_indicator_names),
                "candidate_missing_indicators": [list(row) for row in observation.candidate_missing_indicators],
            },
            "action_index": action_index,
            "action_mask": list(observation.action_mask),
            "log_prob": log_prob,
            "value": value,
            "reward": reward,
            "next_observation": None,
            "done": True,
            "info": {
                "selected_cell": list(observation.candidate_cells[action_index]),
                "coverage_rate_delta": 0.0,
                "path_cost": 0.0,
                "risk": 0.0,
                "failure_reason": None,
                "final_coverage_rate": 0.0,
                "total_cost": 0.0,
                "failure_count": 0,
                "replan_count": 0,
                "ppo_trainable": True,
                "controlled_choice_source": controlled_choice_source,
                "context_id": context_id,
                "episode_id": episode_id,
                "step_index": step_index,
                "split": split,
                "gate_reason_codes": list(gate_reason_codes or []),
            },
        }

    def _write_guarded_collector_episodes(self, transitions: list[dict]) -> None:
        episode = {
            "transitions": transitions,
            "metrics": {
                "final_coverage_rate": 0.0,
                "cumulative_coverage_rate_delta": 0.0,
                "total_path_cost": 0.0,
                "average_risk": 0.0,
                "failure_count": 0,
                "replan_count": 0,
                "value_coverage": 0.0,
            },
        }
        (self.guarded_collector_root / "ppo-rollout-episodes.jsonl").write_text(
            json.dumps(episode) + "\n",
            encoding="utf-8",
        )

    def _write_guarded_collector_summary(self, *, trainable_count: int) -> None:
        self._write_json(
            self.guarded_collector_root / "ppo-rollout-collector-summary.json",
            {
                "schema_version": "ppo-rollout-collector-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "ppo_trainable_transition_count": trainable_count,
                "source_fallback_trainable_count": 0,
                "invalid_action_mask_count": 0,
                "empty_action_mask_count": 0,
                "missing_log_prob_count": 0,
                "missing_value_count": 0,
                "non_finite_reward_count": 0,
                "git_provenance": {"current_matches_sources": True},
            },
        )

    def _return_aligned_row(
        self,
        transition: dict,
        *,
        discounted_return: float,
        advantage: float,
        ppo_trainable: bool,
        rejection_reason_codes: list[str] | None = None,
    ) -> dict:
        info = transition["info"]
        return {
            "schema_version": "return-aligned-guarded-multistep-transition/v1",
            "episode_id": info["episode_id"],
            "step_index": info["step_index"],
            "scenario_id": f"scenario-{info['context_id']}",
            "scenario_family": "unit",
            "context_id": info["context_id"],
            "split": info["split"],
            "controlled_choice_source": info["controlled_choice_source"],
            "controlled_action_index": transition["action_index"],
            "input_ppo_trainable": bool(info.get("ppo_trainable")),
            "ppo_trainable": ppo_trainable,
            "diagnostic_only": not ppo_trainable,
            "reward": transition["reward"],
            "discounted_episode_return": discounted_return,
            "advantage_reference_value": 0.0,
            "advantage": advantage,
            "gate_reason_codes": list(info.get("gate_reason_codes") or []),
            "rejection_reason_codes": list(rejection_reason_codes or []),
            "reward_components": {"unit_reward": transition["reward"]},
        }

    def _write_return_aligned_records(
        self,
        records: list[dict],
        *,
        trainable_count: int,
        diagnostic_count: int = 0,
        non_finite_return_count: int = 0,
    ) -> None:
        self._write_jsonl(self.return_aligned_root / "return-aligned-ppo-transitions.jsonl", records)
        self._write_jsonl(self.return_aligned_root / "return-aligned-ppo-episodes.jsonl", [])
        self._write_json(
            self.return_aligned_root / "return-aligned-collector-summary.json",
            {
                "schema_version": "return-aligned-guarded-multistep-collector-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "guarded_root": str(self.temp_dir / "guarded-root"),
                "horizon": 3,
                "episode_count": 1,
                "step_count": len(records),
                "trainable_episode_count": trainable_count,
                "trainable_transition_count": trainable_count,
                "ppo_trainable_transition_count": trainable_count,
                "diagnostic_transition_count": diagnostic_count,
                "validation_trainable_count": 0,
                "test_trainable_count": 0,
                "source_fallback_trainable_count": 0,
                "invalid_action_mask_count": 0,
                "empty_action_mask_count": 0,
                "missing_log_prob_count": 0,
                "missing_value_count": 0,
                "non_finite_reward_count": 0,
                "non_finite_return_count": non_finite_return_count,
                "non_finite_advantage_count": 0,
                "controlled_regression_count": 0,
                "controlled_safety_regression_count": 0,
                "controlled_contract_regression_count": 0,
                "controlled_path_risk_regression_count": 0,
                "controlled_source_selection_regression_count": 0,
                "uses_multistep_discounted_return": True,
                "not_single_step_best_action": True,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "formal_training_ready_claimed": False,
                "git_provenance": {"current_matches_sources": True},
            },
        )

    def _config(self, *, expected_count: int) -> dict:
        return {
            "schema_version": "return-aligned-guarded-ppo-update-smoke-config/v1",
            "input_files": {
                "return_aligned_summary": "return-aligned-collector-summary.json",
                "return_aligned_transitions": "return-aligned-ppo-transitions.jsonl",
                "guarded_rollout_episodes": "ppo-rollout-episodes.jsonl",
                "guarded_collector_summary": "ppo-rollout-collector-summary.json",
                "base_checkpoint": "experimental-hybrid-policy-candidate.pt",
                "base_checkpoint_metadata": "experimental-hybrid-policy-candidate-metadata.json",
                "base_candidate_summary": "raw-policy-generalization-candidate-summary.json",
            },
            "output_files": {
                "summary": "return-aligned-guarded-ppo-update-smoke-summary.json",
                "optimizer_input_root": "optimizer-input",
                "update_summary": "limited-ppo-update-smoke-summary.json",
                "training_curves": "limited-ppo-update-training-curves.json",
                "diagnostics": "limited-ppo-update-diagnostics.json",
                "checkpoint": "experimental-hybrid-policy-candidate.pt",
                "checkpoint_metadata": "experimental-hybrid-policy-candidate-metadata.json",
                "candidate_summary": "raw-policy-generalization-candidate-summary.json",
            },
            "trainable_filter": {
                "splits": ["train"],
                "controlled_choice_sources": ["policy"],
                "require_empty_gate_reason_codes": True,
                "require_return_aligned_ppo_trainable": True,
            },
            "training": {
                "seed": 0,
                "epochs": 1,
                "learning_rate": 1.0e-5,
                "clip_ratio": 0.2,
                "discount_factor": 0.99,
                "max_grad_norm": 1.0,
                "hidden_size": 8,
                "return_source": "transition_info",
                "return_field": "ppo_return",
                "advantage_field": "ppo_advantage",
            },
            "validation": {
                "expected_input_ppo_trainable_transition_count": expected_count,
                "min_optimizer_train_transition_count": expected_count,
                "max_old_log_prob_abs_error": 1.0e-4,
                "max_old_value_abs_error": 1.0e-4,
                "max_approx_kl": 0.25,
                "max_grad_norm_after_clip": 1.0,
                "min_post_update_teacher_agreement_rate": 0.9,
            },
            "evaluation": {"hidden_size": 8},
            "post_update_gates": {"enabled": True},
            "non_goals": ["no_formal_ppo_rollout", "no_checkpoint_publication"],
        }

    def _passing_post_update_payload(self) -> dict:
        return {
            "raw_generalization": {
                "status": "passed",
                "reason_codes": [],
                "raw_test_regression_count": 0,
                "test_raw_policy_regression_count": 0,
            },
            "sequential_canary": {
                "status": "failed",
                "reason_codes": [
                    "multi_step_accepted_episode_count_below_threshold",
                    "family_with_multi_step_accepted_episode_count_below_threshold",
                    "canary_rejected_policy_choice_count_above_threshold",
                ],
                "canary_rejected_policy_choice_count": 6,
                "controlled_path_cost_regression_count": 0,
                "controlled_risk_regression_count": 0,
                "cumulative_path_cost_regression_count": 0,
                "cumulative_risk_regression_count": 0,
            },
            "generated_collector": {
                "status": "passed",
                "reason_codes": [],
                "ppo_trainable_transition_count": 24,
                "source_fallback_trainable_count": 0,
                "invalid_action_mask_count": 0,
                "empty_action_mask_count": 0,
                "missing_log_prob_count": 0,
                "missing_value_count": 0,
                "non_finite_reward_count": 0,
                "path_cost_regression_count": 0,
                "risk_regression_count": 0,
                "controlled_regression_count": 0,
            },
            "quasi_real_teacher_following": {
                "status": "passed",
                "reason_codes": [],
                "teacher_agreement_rate": 1.0,
                "unsafe_disagreement_count": 0,
                "policy_changed_gate_rejected_count": 0,
            },
            "quasi_real_collector": {
                "status": "passed",
                "reason_codes": [],
                "ppo_trainable_transition_count": 24,
                "source_fallback_trainable_count": 0,
                "invalid_action_mask_count": 0,
                "empty_action_mask_count": 0,
                "missing_log_prob_count": 0,
                "missing_value_count": 0,
                "non_finite_reward_count": 0,
                "path_cost_regression_count": 0,
                "risk_regression_count": 0,
                "controlled_regression_count": 0,
            },
            "long_horizon": {
                "status": "passed",
                "reason_codes": [],
                "verdict": "long_horizon_teacher_skill_contract_aligned",
                "controlled_regression_episode_count": 0,
            },
            "return_aligned_replay": {
                "status": "passed",
                "reason_codes": [],
                "trainable_transition_count": 24,
                "controlled_regression_count": 0,
                "non_finite_reward_count": 0,
                "non_finite_return_count": 0,
                "non_finite_advantage_count": 0,
            },
        }

    def _passing_post_update_runner(self):
        def run_post_update_gates(_context: dict) -> dict:
            return self._passing_post_update_payload()

        return run_post_update_gates

    def _wrapper_summary(self, *, status: str, reason_codes: list[str]) -> dict:
        return {
            "schema_version": "return-aligned-guarded-ppo-update-smoke-summary/v1",
            "status": status,
            "reason_codes": reason_codes,
            "input_return_aligned_trainable_transition_count": 30,
            "optimizer_train_transition_count": 30,
            "validation_test_optimizer_transition_count": 0,
            "source_fallback_optimizer_transition_count": 0,
            "non_empty_gate_reason_optimizer_transition_count": 0,
            "source_fallback_trainable_count": 0,
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
            "uses_multistep_discounted_return": True,
            "not_single_step_best_action": True,
            "post_update_controlled_regression_count": 0,
            "post_update_gates_evaluated": True,
            "post_update_teacher_agreement_rate": 1.0,
            "post_update_raw_generalization_status": "passed",
            "post_update_generated_sequential_status": "failed",
            "post_update_generated_collector_status": "passed",
            "post_update_quasi_real_teacher_following_status": "passed",
            "post_update_quasi_real_collector_status": "passed",
            "post_update_return_aligned_replay_status": "passed",
            "post_update_long_horizon_verdict": "long_horizon_teacher_skill_contract_aligned",
            "experimental_checkpoint": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "git_provenance": {"current_matches_sources": True},
        }

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, allow_nan=True) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
