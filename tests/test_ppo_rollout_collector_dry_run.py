import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


class PpoRolloutCollectorDryRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        for path in (self.repo_root / "scripts", self.repo_root / "model-explorer" / "src"):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)

    def test_consistency_check_rejects_passed_rollout_with_failed_final_diagnosis(self) -> None:
        from scripts.run_sequential_evidence_consistency_check import (
            run_sequential_evidence_consistency_check,
        )

        root = Path(tempfile.mkdtemp(prefix="seq-consistency-"))
        self._write_json(
            root / "policy-gated-sequential-canary-rollout-summary.json",
            {
                "schema_version": "policy-gated-sequential-canary-rollout-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "evaluation_stage": "sequential_multi_step_opportunity",
            },
        )
        self._write_json(
            root / "sequential-multi-step-opportunity-diagnosis-summary.json",
            {
                "schema_version": "sequential-multi-step-opportunity-diagnosis-summary/v1",
                "status": "failed",
                "reason_codes": ["multi_step_opportunity_episode_count_below_threshold"],
            },
        )
        readiness = root / "policy-training-readiness-review-summary.json"
        self._write_json(
            readiness,
            {
                "schema_version": "policy-training-readiness-review-summary/v1",
                "status": "passed",
                "training_readiness_status": "policy_gated_sequential_multi_step_opportunity_evaluated",
                "training_blockers": [],
                "reason_codes": [],
            },
        )

        summary = run_sequential_evidence_consistency_check(
            batch_root=root,
            readiness_summary_path=readiness,
            config={
                "schema_version": "sequential-evidence-consistency-config/v1",
                "diagnosis_role": "final_gate",
                "validation": {"require_readiness_status": "policy_gated_sequential_multi_step_opportunity_evaluated"},
            },
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sequential_diagnosis_final_gate_failed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "sequential_evidence_consistency_required")

    def test_consistency_check_accepts_passed_preflight_diagnosis_outside_final_root(self) -> None:
        from scripts.run_sequential_evidence_consistency_check import (
            run_sequential_evidence_consistency_check,
        )

        root = Path(tempfile.mkdtemp(prefix="seq-consistency-final-"))
        preflight_root = Path(tempfile.mkdtemp(prefix="seq-consistency-preflight-"))
        self._write_json(
            root / "policy-gated-sequential-canary-rollout-summary.json",
            {
                "schema_version": "policy-gated-sequential-canary-rollout-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "episode_count": 36,
                "step_count": 108,
            },
        )
        self._write_json(
            preflight_root / "sequential-multi-step-opportunity-diagnosis-summary.json",
            {
                "schema_version": "sequential-multi-step-opportunity-diagnosis-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "episode_count": 36,
                "step_count": 108,
                "multi_step_opportunity_episode_count": 12,
                "family_with_multi_step_opportunity_count": 6,
            },
        )
        readiness = root / "policy-training-readiness-review-summary.json"
        self._write_json(
            readiness,
            {
                "schema_version": "policy-training-readiness-review-summary/v1",
                "status": "passed",
                "training_readiness_status": "policy_gated_sequential_multi_step_opportunity_evaluated",
                "training_blockers": [],
                "reason_codes": [],
            },
        )

        summary = run_sequential_evidence_consistency_check(
            batch_root=root,
            readiness_summary_path=readiness,
            config={
                "schema_version": "sequential-evidence-consistency-config/v1",
                "diagnosis_role": "preflight_only",
                "validation": {
                    "require_readiness_status": "policy_gated_sequential_multi_step_opportunity_evaluated",
                    "require_diagnosis_summary": True,
                },
            },
            repo_root=self.repo_root,
            diagnosis_root=preflight_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(summary["diagnosis_summary"]["preflight_only"])
        self.assertEqual(summary["diagnosis_summary"]["root"], str(preflight_root))

    def test_consistency_check_still_requires_diagnosis_without_preflight_root(self) -> None:
        from scripts.run_sequential_evidence_consistency_check import (
            run_sequential_evidence_consistency_check,
        )

        root = Path(tempfile.mkdtemp(prefix="seq-consistency-missing-diagnosis-"))
        self._write_json(
            root / "policy-gated-sequential-canary-rollout-summary.json",
            {
                "schema_version": "policy-gated-sequential-canary-rollout-summary/v1",
                "status": "passed",
                "reason_codes": [],
            },
        )
        readiness = root / "policy-training-readiness-review-summary.json"
        self._write_json(
            readiness,
            {
                "schema_version": "policy-training-readiness-review-summary/v1",
                "status": "passed",
                "training_readiness_status": "policy_gated_sequential_multi_step_opportunity_evaluated",
                "training_blockers": [],
                "reason_codes": [],
            },
        )

        summary = run_sequential_evidence_consistency_check(
            batch_root=root,
            readiness_summary_path=readiness,
            config={
                "schema_version": "sequential-evidence-consistency-config/v1",
                "diagnosis_role": "preflight_only",
                "validation": {
                    "require_readiness_status": "policy_gated_sequential_multi_step_opportunity_evaluated",
                    "require_diagnosis_summary": True,
                },
            },
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sequential_diagnosis_summary_missing", summary["reason_codes"])

    def test_collector_materializes_only_policy_controlled_steps_as_ppo_trainable(self) -> None:
        from model_explorer.policy.rollout_io import read_rollout_episodes
        from scripts.run_ppo_rollout_collector_dry_run import run_ppo_rollout_collector_dry_run

        sequential_root = Path(tempfile.mkdtemp(prefix="ppo-collector-seq-"))
        output_root = Path(tempfile.mkdtemp(prefix="ppo-collector-out-"))
        steps = [
            self._step(
                episode_id="ep-a",
                step_index=0,
                controlled_choice_source="policy",
                controlled_action_index=1,
                context_id="ctx-policy",
                log_prob=-0.25,
                value=0.4,
                path_delta=-0.5,
                risk_delta=-0.02,
            ),
            self._step(
                episode_id="ep-a",
                step_index=1,
                controlled_choice_source="source_fallback",
                controlled_action_index=0,
                context_id="ctx-source",
                log_prob=-0.50,
                value=0.2,
                path_delta=0.0,
                risk_delta=0.0,
            ),
        ]
        self._write_jsonl(sequential_root / "policy-gated-sequential-canary-steps.jsonl", steps)
        self._write_json(
            sequential_root / "policy-gated-sequential-canary-rollout-summary.json",
            {
                "schema_version": "policy-gated-sequential-canary-rollout-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "episode_count": 1,
                "step_count": 2,
                "state_continuity_violation_count": 0,
                "canary_rejected_policy_choice_count": 0,
            },
        )

        summary = run_ppo_rollout_collector_dry_run(
            sequential_root=sequential_root,
            output_root=output_root,
            candidate_root=None,
            config={
                "schema_version": "ppo-rollout-collector-dry-run-config/v1",
                "validation": {
                    "min_ppo_trainable_transition_count": 1,
                    "max_invalid_action_mask_count": 0,
                    "max_empty_action_mask_count": 0,
                    "max_source_fallback_trainable_count": 0,
                },
                "reward": {"better_choice_bonus": 1.0, "path_improvement_weight": 1.0, "risk_improvement_weight": 10.0},
            },
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["step_count"], 2)
        self.assertEqual(summary["ppo_trainable_transition_count"], 1)
        self.assertEqual(summary["source_fallback_trainable_count"], 0)
        self.assertEqual(summary["missing_log_prob_count"], 0)
        self.assertEqual(summary["missing_value_count"], 0)
        self.assertEqual(summary["non_finite_reward_count"], 0)
        episodes = read_rollout_episodes(output_root / "ppo-rollout-episodes.jsonl")
        self.assertEqual(len(episodes), 1)
        self.assertEqual(len(episodes[0].transitions), 1)
        transition = episodes[0].transitions[0]
        self.assertEqual(transition.action_index, 1)
        self.assertTrue(transition.info.extra["ppo_trainable"])
        self.assertEqual(transition.info.extra["controlled_choice_source"], "policy")
        self.assertGreaterEqual(transition.reward, 0.0)

    def test_collector_rejects_non_finite_reward(self) -> None:
        from scripts.run_ppo_rollout_collector_dry_run import run_ppo_rollout_collector_dry_run

        sequential_root = Path(tempfile.mkdtemp(prefix="ppo-collector-bad-reward-"))
        output_root = Path(tempfile.mkdtemp(prefix="ppo-collector-bad-reward-out-"))
        step = self._step(
            episode_id="ep-a",
            step_index=0,
            controlled_choice_source="policy",
            controlled_action_index=1,
            context_id="ctx-policy",
            log_prob=-0.25,
            value=0.4,
            path_delta=math.inf,
            risk_delta=0.0,
        )
        self._write_jsonl(sequential_root / "policy-gated-sequential-canary-steps.jsonl", [step])
        self._write_json(
            sequential_root / "policy-gated-sequential-canary-rollout-summary.json",
            {"schema_version": "policy-gated-sequential-canary-rollout-summary/v1", "status": "passed", "reason_codes": []},
        )

        summary = run_ppo_rollout_collector_dry_run(
            sequential_root=sequential_root,
            output_root=output_root,
            candidate_root=None,
            config={
                "schema_version": "ppo-rollout-collector-dry-run-config/v1",
                "validation": {"min_ppo_trainable_transition_count": 1},
                "reward": {},
            },
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("ppo_reward_contract_invalid", summary["reason_codes"])
        self.assertEqual(summary["non_finite_reward_count"], 1)

    def test_readiness_accepts_passed_ppo_collector_summary(self) -> None:
        from scripts.run_policy_training_readiness_review import _ppo_rollout_collector_readiness

        readiness = _ppo_rollout_collector_readiness(
            {
                "schema_version": "ppo-rollout-collector-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "episode_count": 36,
                "step_count": 108,
                "ppo_trainable_transition_count": 24,
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
                "git_provenance": {"current_matches_sources": True},
            }
        )

        self.assertTrue(readiness["present"])
        self.assertTrue(readiness["completed"])
        self.assertEqual(readiness["training_blockers"], [])
        self.assertEqual(readiness["ppo_trainable_transition_count"], 24)

    def _step(
        self,
        *,
        episode_id: str,
        step_index: int,
        controlled_choice_source: str,
        controlled_action_index: int,
        context_id: str,
        log_prob: float,
        value: float,
        path_delta: float,
        risk_delta: float,
    ) -> dict:
        observation = {
            "candidate_feature_names": ["path_cost", "risk"],
            "candidate_features": [[1.0, 0.2], [0.5, 0.1]],
            "global_feature_names": ["step_index"],
            "global_features": [float(step_index)],
            "action_mask": [True, True],
            "candidate_cells": [[4 + step_index, 6], [5 + step_index, 6]],
            "candidate_missing_feature_names": [[], []],
            "candidate_missing_indicator_names": [],
            "candidate_missing_indicators": [[], []],
        }
        return {
            "schema_version": "policy-gated-sequential-canary-step/v1",
            "episode_id": episode_id,
            "step_index": step_index,
            "context_id": context_id,
            "scenario_id": f"scenario-{episode_id}-{step_index}",
            "scenario_group": "mixed_stress_detour",
            "input_start_cell": [1 + step_index, 6],
            "raw_policy_selected_action_index": controlled_action_index,
            "source_selected_action_index": 0,
            "controlled_action_index": controlled_action_index,
            "controlled_choice_source": controlled_choice_source,
            "controlled_execution_goal_cell": [5 + step_index, 6],
            "policy_execution_goal_cell": [5 + step_index, 6],
            "source_execution_goal_cell": [4 + step_index, 6],
            "decision_class": "canary_accepted_policy_choice" if controlled_choice_source == "policy" else "source_aligned",
            "canary_gate_passed": controlled_choice_source == "policy",
            "canary_rejection_reason_codes": [],
            "controlled_regression_reason_codes": [],
            "raw_policy_regression_reason_codes": [],
            "policy_selected_path_cost_delta": path_delta,
            "policy_selected_risk_delta": risk_delta,
            "policy_selected_utility_delta": 0.01,
            "accepted_choice_value_class": "accepted_better" if controlled_choice_source == "policy" else None,
            "policy_action_log_prob": log_prob,
            "policy_value": value,
            "observation": observation,
        }

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
