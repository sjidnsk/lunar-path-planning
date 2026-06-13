import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class LimitedQuasiRealPpoUpdateSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        for path in (self.repo_root / "scripts", self.repo_root / "model-explorer" / "src"):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="limited-quasi-ppo-smoke-"))
        self.source_root = self.temp_dir / "source"
        self.base_candidate_root = self.temp_dir / "base-candidate"
        self.collector_root = self.temp_dir / "collector"
        self.output_root = self.temp_dir / "output"
        for path in (self.source_root, self.base_candidate_root, self.collector_root):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_teacher_aligned_and_safe_disagreement_train_transitions_enter_optimizer(self) -> None:
        from scripts.run_limited_quasi_real_ppo_update_smoke import (
            run_limited_quasi_real_ppo_update_smoke,
        )

        observation = self._observation()
        self._write_base_candidate(observation=observation)
        transitions = [
            self._transition_payload(
                observation=observation,
                action_index=0,
                reward=1.0,
                context_id="ctx-teacher",
                controlled_choice_source="policy_teacher_aligned",
                split="train",
            ),
            self._transition_payload(
                observation=observation,
                action_index=1,
                reward=1.0,
                context_id="ctx-safe",
                controlled_choice_source="policy_safe_disagreement",
                split="train",
            ),
        ]
        self._write_collector_episodes(transitions)
        self._write_collector_transition_records(transitions)
        self._write_collector_summary(trainable_count=2, diagnostic_count=0)

        summary = run_limited_quasi_real_ppo_update_smoke(
            source_root=self.source_root,
            base_candidate_root=self.base_candidate_root,
            collector_root=self.collector_root,
            output_root=self.output_root,
            config=self._config(expected_count=2),
            repo_root=self.repo_root,
            post_update_runner=self._passing_post_update_runner(),
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["schema_version"], "limited-quasi-real-ppo-update-smoke-summary/v1")
        self.assertEqual(summary["input_ppo_trainable_transition_count"], 2)
        self.assertEqual(summary["optimizer_train_transition_count"], 2)
        self.assertEqual(summary["optimizer_transition_split_counts"], {"train": 2})
        self.assertEqual(
            summary["optimizer_transition_source_counts"],
            {"policy_safe_disagreement": 1, "policy_teacher_aligned": 1},
        )
        self.assertEqual(summary["validation_test_optimizer_transition_count"], 0)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])
        self.assertFalse(summary["formal_training_ready_claimed"])

    def test_validation_and_rejected_sources_do_not_enter_optimizer(self) -> None:
        from scripts.run_limited_quasi_real_ppo_update_smoke import (
            run_limited_quasi_real_ppo_update_smoke,
        )

        observation = self._observation()
        self._write_base_candidate(observation=observation)
        transitions = [
            self._transition_payload(
                observation=observation,
                action_index=0,
                reward=1.0,
                context_id="ctx-train",
                controlled_choice_source="policy_teacher_aligned",
                split="train",
            ),
            self._transition_payload(
                observation=observation,
                action_index=1,
                reward=1.0,
                context_id="ctx-validation",
                controlled_choice_source="policy_teacher_aligned",
                split="validation",
            ),
            self._transition_payload(
                observation=observation,
                action_index=1,
                reward=0.0,
                context_id="ctx-fallback",
                controlled_choice_source="teacher_fallback",
                split="train",
            ),
            self._transition_payload(
                observation=observation,
                action_index=1,
                reward=0.0,
                context_id="ctx-gated",
                controlled_choice_source="policy_teacher_aligned",
                split="train",
                gate_reason_codes=["path_cost_regression"],
            ),
        ]
        self._write_collector_episodes(transitions)
        self._write_collector_transition_records(transitions, diagnostic_count=3)
        self._write_collector_summary(trainable_count=1, diagnostic_count=3)

        summary = run_limited_quasi_real_ppo_update_smoke(
            source_root=self.source_root,
            base_candidate_root=self.base_candidate_root,
            collector_root=self.collector_root,
            output_root=self.output_root,
            config=self._config(expected_count=1),
            repo_root=self.repo_root,
            post_update_runner=self._passing_post_update_runner(),
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["optimizer_train_transition_count"], 1)
        self.assertEqual(summary["optimizer_transition_split_counts"], {"train": 1})
        self.assertEqual(summary["validation_test_optimizer_transition_count"], 0)
        self.assertEqual(summary["non_empty_gate_reason_optimizer_transition_count"], 0)
        self.assertEqual(summary["disallowed_source_optimizer_transition_count"], 0)

    def test_log_prob_mismatch_fails_before_wrapper_can_pass(self) -> None:
        from scripts.run_limited_quasi_real_ppo_update_smoke import (
            run_limited_quasi_real_ppo_update_smoke,
        )

        observation = self._observation()
        self._write_base_candidate(observation=observation)
        transition = self._transition_payload(
            observation=observation,
            action_index=0,
            reward=1.0,
            context_id="ctx-mismatch",
            controlled_choice_source="policy_teacher_aligned",
            split="train",
        )
        transition["log_prob"] += 0.25
        self._write_collector_episodes([transition])
        self._write_collector_transition_records([transition])
        self._write_collector_summary(trainable_count=1, diagnostic_count=0)

        summary = run_limited_quasi_real_ppo_update_smoke(
            source_root=self.source_root,
            base_candidate_root=self.base_candidate_root,
            collector_root=self.collector_root,
            output_root=self.output_root,
            config=self._config(expected_count=1),
            repo_root=self.repo_root,
            post_update_runner=self._passing_post_update_runner(),
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("ppo_update_not_on_collector_policy", summary["reason_codes"])

    def test_readiness_accepts_passed_quasi_real_update_smoke_summary(self) -> None:
        from scripts.run_policy_training_readiness_review import (
            _limited_quasi_real_ppo_update_smoke_readiness,
        )

        readiness = _limited_quasi_real_ppo_update_smoke_readiness(
            self._wrapper_summary(
                status="passed",
                reason_codes=[],
                post_update_quasi_real_teacher_following_status="passed",
                post_update_quasi_real_collector_status="passed",
            )
        )

        self.assertTrue(readiness["present"])
        self.assertTrue(readiness["completed"])
        self.assertEqual(readiness["training_blockers"], [])
        self.assertEqual(readiness["optimizer_train_transition_count"], 36)

    def test_readiness_blocks_post_update_quasi_real_gate_regression(self) -> None:
        from scripts.run_policy_training_readiness_review import (
            _limited_quasi_real_ppo_update_smoke_readiness,
        )

        readiness = _limited_quasi_real_ppo_update_smoke_readiness(
            self._wrapper_summary(
                status="passed",
                reason_codes=[],
                post_update_quasi_real_teacher_following_status="failed",
                post_update_quasi_real_collector_status="passed",
            )
        )

        self.assertFalse(readiness["completed"])
        self.assertIn("limited_quasi_real_ppo_update_post_update_gate_regression", readiness["training_blockers"])

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
        controlled_choice_source: str,
        split: str,
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
                "episode_id": f"ep-{context_id}",
                "step_index": 0,
                "split": split,
                "gate_reason_codes": list(gate_reason_codes or []),
            },
        }

    def _write_collector_episodes(self, transitions: list[dict]) -> None:
        lines = []
        for transition in transitions:
            episode = {
                "transitions": [transition],
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
            lines.append(json.dumps(episode))
        (self.collector_root / "ppo-rollout-episodes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_collector_transition_records(self, transitions: list[dict], diagnostic_count: int = 0) -> None:
        records = []
        for index, transition in enumerate(transitions):
            extra = transition["info"]
            optimizer_candidate = (
                extra.get("split") == "train"
                and extra.get("controlled_choice_source")
                in {"policy_teacher_aligned", "policy_safe_disagreement"}
                and not extra.get("gate_reason_codes")
            )
            records.append(
                {
                    "schema_version": "ppo-rollout-transition-record/v1",
                    "episode_id": extra["episode_id"],
                    "step_index": 0,
                    "decision_index": index,
                    "context_id": extra["context_id"],
                    "split": extra["split"],
                    "controlled_choice_source": extra["controlled_choice_source"],
                    "controlled_action_index": transition["action_index"],
                    "ppo_trainable": optimizer_candidate,
                    "diagnostic_only": not optimizer_candidate,
                    "rejection_reason_codes": [] if optimizer_candidate else ["diagnostic_only"],
                    "reward": transition["reward"],
                    "reward_components": {"teacher_following_bonus": transition["reward"]},
                    "reward_audit": {"reason_codes": []},
                    "counter_deltas": {
                        "ppo_trainable_transition_count": 1 if optimizer_candidate else 0,
                        "diagnostic_transition_count": 0 if optimizer_candidate else 1,
                    },
                }
            )
        (self.collector_root / "ppo-rollout-transitions.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def _write_collector_summary(self, *, trainable_count: int, diagnostic_count: int) -> None:
        self._write_json(
            self.collector_root / "ppo-rollout-collector-summary.json",
            {
                "schema_version": "ppo-rollout-collector-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "candidate_root": str(self.base_candidate_root),
                "quasi_real_root": str(self.temp_dir / "quasi-real"),
                "guarded_teacher_following_root": str(self.temp_dir / "teacher-following"),
                "episode_count": trainable_count + diagnostic_count,
                "step_count": trainable_count + diagnostic_count,
                "ppo_trainable_transition_count": trainable_count,
                "diagnostic_transition_count": diagnostic_count,
                "source_fallback_trainable_count": 0,
                "invalid_action_mask_count": 0,
                "empty_action_mask_count": 0,
                "missing_log_prob_count": 0,
                "missing_value_count": 0,
                "non_finite_reward_count": 0,
                "state_continuity_violation_count": 0,
                "fallback_or_open_grid_count": 0,
                "safety_regression_count": 0,
                "contract_violation_count": 0,
                "path_cost_regression_count": 0,
                "risk_regression_count": 0,
                "source_selection_regression_count": 0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "formal_training_ready_claimed": False,
                "git_provenance": {"current_matches_sources": True},
            },
        )

    def _config(self, *, expected_count: int) -> dict:
        return {
            "schema_version": "limited-quasi-real-ppo-update-smoke-config/v1",
            "input_files": {
                "rollout_episodes": "ppo-rollout-episodes.jsonl",
                "rollout_transitions": "ppo-rollout-transitions.jsonl",
                "collector_summary": "ppo-rollout-collector-summary.json",
                "base_checkpoint": "experimental-hybrid-policy-candidate.pt",
                "base_checkpoint_metadata": "experimental-hybrid-policy-candidate-metadata.json",
                "base_candidate_summary": "raw-policy-generalization-candidate-summary.json",
            },
            "output_files": {
                "summary": "limited-quasi-real-ppo-update-smoke-summary.json",
                "update_summary": "limited-ppo-update-smoke-summary.json",
                "training_curves": "limited-ppo-update-training-curves.json",
                "diagnostics": "limited-ppo-update-diagnostics.json",
                "checkpoint": "experimental-hybrid-policy-candidate.pt",
                "checkpoint_metadata": "experimental-hybrid-policy-candidate-metadata.json",
                "candidate_summary": "raw-policy-generalization-candidate-summary.json",
            },
            "training": {
                "seed": 0,
                "epochs": 1,
                "learning_rate": 1.0e-5,
                "clip_ratio": 0.2,
                "discount_factor": 0.99,
                "max_grad_norm": 1.0,
            },
            "validation": {
                "expected_input_ppo_trainable_transition_count": expected_count,
                "min_optimizer_train_transition_count": expected_count,
                "max_old_log_prob_abs_error": 1.0e-4,
                "max_old_value_abs_error": 1.0e-4,
                "max_approx_kl": 0.25,
                "max_grad_norm_after_clip": 1.0,
                "min_post_update_ppo_trainable_transition_count": 1,
                "min_post_update_teacher_agreement_rate": 0.9,
            },
            "trainable_filter": {
                "splits": ["train"],
                "controlled_choice_sources": ["policy_teacher_aligned", "policy_safe_disagreement"],
                "require_empty_gate_reason_codes": True,
            },
            "post_update_gates": {"enabled": True},
            "non_goals": [
                "no_formal_ppo_rollout",
                "no_checkpoint_publication",
                "no_default_policy_replacement",
            ],
        }

    def _passing_post_update_runner(self):
        def run_post_update_gates(_context: dict) -> dict:
            return {
                "raw_generalization": {"status": "passed", "reason_codes": [], "raw_test_regression_count": 0},
                "sequential_canary": {"status": "passed", "reason_codes": [], "rejected_choice_count": 0},
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
                    "diagnostic_transition_count": 72,
                    "source_fallback_trainable_count": 0,
                    "invalid_action_mask_count": 0,
                    "empty_action_mask_count": 0,
                    "missing_log_prob_count": 0,
                    "missing_value_count": 0,
                    "non_finite_reward_count": 0,
                },
            }

        return run_post_update_gates

    def _wrapper_summary(
        self,
        *,
        status: str,
        reason_codes: list[str],
        post_update_quasi_real_teacher_following_status: str,
        post_update_quasi_real_collector_status: str,
    ) -> dict:
        return {
            "schema_version": "limited-quasi-real-ppo-update-smoke-summary/v1",
            "status": status,
            "reason_codes": reason_codes,
            "input_ppo_trainable_transition_count": 36,
            "optimizer_train_transition_count": 36,
            "validation_test_optimizer_transition_count": 0,
            "non_empty_gate_reason_optimizer_transition_count": 0,
            "disallowed_source_optimizer_transition_count": 0,
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
            "experimental_checkpoint": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "post_update_raw_generalization_status": "passed",
            "post_update_raw_test_regression_count": 0,
            "post_update_sequential_canary_status": "passed",
            "post_update_sequential_rejected_choice_count": 0,
            "post_update_generated_collector_status": "passed",
            "post_update_generated_collector_trainable_transition_count": 24,
            "post_update_quasi_real_teacher_following_status": post_update_quasi_real_teacher_following_status,
            "post_update_quasi_real_teacher_agreement_rate": 1.0,
            "post_update_quasi_real_unsafe_disagreement_count": 0,
            "post_update_quasi_real_collector_status": post_update_quasi_real_collector_status,
            "post_update_quasi_real_collector_trainable_transition_count": 24,
            "git_provenance": {"current_matches_sources": True},
        }

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
