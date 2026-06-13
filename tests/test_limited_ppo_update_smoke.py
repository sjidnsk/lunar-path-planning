import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class LimitedPpoUpdateSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        for path in (self.repo_root / "scripts", self.repo_root / "model-explorer" / "src"):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="limited-ppo-smoke-"))
        self.base_candidate_root = self.temp_dir / "base-candidate"
        self.collector_root = self.temp_dir / "collector"
        self.output_root = self.temp_dir / "smoke-output"
        self.base_candidate_root.mkdir(parents=True)
        self.collector_root.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_smoke_update_loads_base_checkpoint_and_writes_experimental_candidate(self) -> None:
        from model_explorer.policy.rollout_io import read_rollout_episodes
        from scripts.run_limited_ppo_update_smoke import run_limited_ppo_update_smoke

        observation = self._observation()
        self._write_base_candidate(observation=observation)
        self._write_collector_episodes(
            [
                self._transition_payload(
                    observation=observation,
                    action_index=1,
                    reward=1.0,
                    context_id="ctx-train-1",
                    ppo_trainable=True,
                    controlled_choice_source="policy",
                )
            ]
        )
        self._write_collector_summary(trainable_count=1)

        summary = run_limited_ppo_update_smoke(
            source_root=self.temp_dir / "source",
            base_candidate_root=self.base_candidate_root,
            collector_root=self.collector_root,
            output_root=self.output_root,
            config=self._config(expected_count=1),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["input_ppo_trainable_transition_count"], 1)
        self.assertEqual(summary["optimizer_train_transition_count"], 1)
        self.assertEqual(summary["source_fallback_trainable_count"], 0)
        self.assertLessEqual(summary["old_log_prob_max_abs_error"], 1.0e-4)
        self.assertLessEqual(summary["old_value_max_abs_error"], 1.0e-4)
        self.assertGreater(summary["parameter_l2_delta"], 0.0)
        self.assertLessEqual(summary["approx_kl"], 0.25)
        self.assertLessEqual(summary["max_grad_norm_after_clip"], 1.0)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])
        self.assertTrue((self.output_root / "experimental-hybrid-policy-candidate.pt").is_file())
        candidate_summary = self._load_json(self.output_root / "raw-policy-generalization-candidate-summary.json")
        self.assertEqual(candidate_summary["schema_version"], "raw-policy-generalization-candidate-summary/v1")
        self.assertTrue(candidate_summary["experimental_checkpoint"])
        self.assertFalse(candidate_summary["publishes_checkpoint"])
        episodes = read_rollout_episodes(self.collector_root / "ppo-rollout-episodes.jsonl")
        self.assertEqual(len(episodes), 1)

    def test_smoke_rejects_log_prob_or_value_mismatch(self) -> None:
        from scripts.run_limited_ppo_update_smoke import run_limited_ppo_update_smoke

        observation = self._observation()
        self._write_base_candidate(observation=observation)
        transition = self._transition_payload(
            observation=observation,
            action_index=1,
            reward=1.0,
            context_id="ctx-mismatch",
            ppo_trainable=True,
            controlled_choice_source="policy",
        )
        transition["log_prob"] += 0.25
        self._write_collector_episodes([transition])
        self._write_collector_summary(trainable_count=1)

        summary = run_limited_ppo_update_smoke(
            source_root=self.temp_dir / "source",
            base_candidate_root=self.base_candidate_root,
            collector_root=self.collector_root,
            output_root=self.output_root,
            config=self._config(expected_count=1),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("ppo_update_not_on_collector_policy", summary["reason_codes"])
        self.assertGreater(summary["old_log_prob_max_abs_error"], 1.0e-4)

    def test_source_fallback_transition_never_enters_optimizer(self) -> None:
        from scripts.run_limited_ppo_update_smoke import run_limited_ppo_update_smoke

        observation = self._observation()
        self._write_base_candidate(observation=observation)
        self._write_collector_episodes(
            [
                self._transition_payload(
                    observation=observation,
                    action_index=0,
                    reward=0.0,
                    context_id="ctx-source",
                    ppo_trainable=False,
                    controlled_choice_source="source_fallback",
                )
            ]
        )
        self._write_collector_summary(trainable_count=0)

        summary = run_limited_ppo_update_smoke(
            source_root=self.temp_dir / "source",
            base_candidate_root=self.base_candidate_root,
            collector_root=self.collector_root,
            output_root=self.output_root,
            config=self._config(expected_count=1),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["optimizer_train_transition_count"], 0)
        self.assertIn("limited_ppo_update_input_contract_invalid", summary["reason_codes"])

    def test_configured_trainable_filter_accepts_quasi_real_teacher_following_sources(self) -> None:
        from scripts.run_limited_ppo_update_smoke import run_limited_ppo_update_smoke

        observation = self._observation()
        self._write_base_candidate(observation=observation)
        self._write_collector_episodes(
            [
                self._transition_payload(
                    observation=observation,
                    action_index=1,
                    reward=1.0,
                    context_id="ctx-quasi-train",
                    ppo_trainable=True,
                    controlled_choice_source="policy_teacher_aligned",
                    split="train",
                ),
                self._transition_payload(
                    observation=observation,
                    action_index=0,
                    reward=1.0,
                    context_id="ctx-quasi-validation",
                    ppo_trainable=True,
                    controlled_choice_source="policy_teacher_aligned",
                    split="validation",
                ),
            ]
        )
        self._write_collector_summary(trainable_count=1)
        config = self._config(expected_count=1)
        config["trainable_filter"] = {
            "splits": ["train"],
            "controlled_choice_sources": ["policy_teacher_aligned", "policy_safe_disagreement"],
            "require_empty_gate_reason_codes": True,
        }

        summary = run_limited_ppo_update_smoke(
            source_root=self.temp_dir / "source",
            base_candidate_root=self.base_candidate_root,
            collector_root=self.collector_root,
            output_root=self.output_root,
            config=config,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["input_ppo_trainable_transition_count"], 1)
        self.assertEqual(summary["optimizer_train_transition_count"], 1)
        self.assertEqual(summary["optimizer_transition_split_counts"], {"train": 1})
        self.assertEqual(summary["optimizer_transition_source_counts"], {"policy_teacher_aligned": 1})

    def test_readiness_accepts_passed_limited_ppo_update_smoke_summary(self) -> None:
        from scripts.run_policy_training_readiness_review import _limited_ppo_update_smoke_readiness

        readiness = _limited_ppo_update_smoke_readiness(
            {
                "schema_version": "limited-ppo-update-smoke-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "input_ppo_trainable_transition_count": 37,
                "optimizer_train_transition_count": 37,
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
            }
        )

        self.assertTrue(readiness["present"])
        self.assertTrue(readiness["completed"])
        self.assertEqual(readiness["training_blockers"], [])
        self.assertEqual(readiness["optimizer_train_transition_count"], 37)

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
        ppo_trainable: bool,
        controlled_choice_source: str,
        split: str | None = None,
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
            "observation": observation.to_dict()
            if hasattr(observation, "to_dict")
            else {
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
                "selected_cell": [5, 6],
                "coverage_rate_delta": 0.0,
                "path_cost": 0.0,
                "risk": 0.0,
                "failure_reason": None,
                "final_coverage_rate": 0.0,
                "total_cost": 0.0,
                "failure_count": 0,
                "replan_count": 0,
                "ppo_trainable": ppo_trainable,
                "controlled_choice_source": controlled_choice_source,
                "context_id": context_id,
                "episode_id": "ep-smoke",
                "step_index": 0,
                "split": split,
                "gate_reason_codes": list(gate_reason_codes or []),
            },
        }

    def _write_collector_episodes(self, transitions: list[dict]) -> None:
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
        path = self.collector_root / "ppo-rollout-episodes.jsonl"
        path.write_text(json.dumps(episode) + "\n", encoding="utf-8")

    def _write_collector_summary(self, *, trainable_count: int) -> None:
        self._write_json(
            self.collector_root / "ppo-rollout-collector-summary.json",
            {
                "schema_version": "ppo-rollout-collector-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "ppo_trainable_transition_count": trainable_count,
                "source_fallback_trainable_count": 0,
                "git_provenance": {"current_matches_sources": True},
            },
        )

    def _config(self, *, expected_count: int) -> dict:
        return {
            "schema_version": "limited-ppo-update-smoke-config/v1",
            "input_files": {
                "rollout_episodes": "ppo-rollout-episodes.jsonl",
                "collector_summary": "ppo-rollout-collector-summary.json",
                "base_checkpoint": "experimental-hybrid-policy-candidate.pt",
                "base_checkpoint_metadata": "experimental-hybrid-policy-candidate-metadata.json",
                "base_candidate_summary": "raw-policy-generalization-candidate-summary.json",
            },
            "output_files": {
                "summary": "limited-ppo-update-smoke-summary.json",
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
            },
        }

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))
