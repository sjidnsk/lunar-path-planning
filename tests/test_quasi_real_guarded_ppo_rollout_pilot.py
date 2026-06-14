import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class QuasiRealGuardedPpoRolloutPilotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_dir = str(self.repo_root / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="quasi-real-guarded-ppo-rollout-"))
        self.update_root = self.temp_dir / "update"
        self.teacher_root = self.update_root / "post_update_quasi_real_teacher_following"
        self.collector_root = self.update_root / "post_update_quasi_real_collector"
        self.long_horizon_root = self.update_root / "post_update_long_horizon"
        self.replay_root = self.update_root / "post_update_return_aligned_replay"
        self.candidate_root = self.update_root
        self.quasi_real_root = self.temp_dir / "quasi"
        self.output_root = self.temp_dir / "out"
        for path in (
            self.teacher_root,
            self.collector_root,
            self.long_horizon_root,
            self.replay_root,
            self.candidate_root,
            self.quasi_real_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self._write_inputs()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_horizon_three_rollout_materializes_trainable_policy_steps(self) -> None:
        from scripts.run_quasi_real_guarded_ppo_rollout_pilot import (
            run_quasi_real_guarded_ppo_rollout_pilot,
        )

        decisions = [
            self._decision("ctx-train-0", split="train", action=0),
            self._decision("ctx-train-1", split="train", action=1),
            self._decision("ctx-train-2", split="train", action=0),
        ]
        self._write_decisions(decisions)

        summary = run_quasi_real_guarded_ppo_rollout_pilot(
            update_smoke_root=self.update_root,
            candidate_root=self.candidate_root,
            quasi_real_root=self.quasi_real_root,
            output_root=self.output_root,
            config=self._config(min_episode_count=1, min_step_count=3, min_trainable=3),
            repo_root=self.repo_root,
            collector_replay_runner=self._passing_collector_replay(),
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["episode_count"], 1)
        self.assertEqual(summary["step_count"], 3)
        self.assertEqual(summary["trainable_transition_count"], 3)
        self.assertEqual(summary["diagnostic_transition_count"], 0)
        self.assertEqual(summary["controlled_regression_count"], 0)
        self.assertEqual(summary["teacher_agreement_rate"], 1.0)
        self.assertTrue(summary["uses_multistep_discounted_return"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])
        self.assertFalse(summary["formal_training_ready_claimed"])

        steps = self._read_jsonl(self.output_root / "quasi-real-guarded-ppo-rollout-steps.jsonl")
        self.assertEqual([step["step_index"] for step in steps], [0, 1, 2])
        self.assertEqual({step["controlled_choice_source"] for step in steps}, {"policy"})
        self.assertEqual([step["done"] for step in steps], [False, False, True])
        self.assertTrue(all(step["ppo_trainable"] for step in steps))
        self.assertTrue(all(math.isfinite(step["log_prob"]) for step in steps))
        self.assertTrue(all(math.isfinite(step["value"]) for step in steps))
        self.assertAlmostEqual(steps[0]["discounted_return"], 1.0 + 0.99 + 0.99 * 0.99)

    def test_validation_test_fallback_and_gate_reason_are_diagnostic_only(self) -> None:
        from scripts.run_quasi_real_guarded_ppo_rollout_pilot import (
            run_quasi_real_guarded_ppo_rollout_pilot,
        )

        decisions = [
            self._decision("ctx-train", split="train", action=0),
            self._decision("ctx-val", split="validation", action=0),
            self._decision("ctx-test", split="test", action=0),
            self._decision(
                "ctx-fallback",
                split="train",
                action=0,
                controlled_choice_source="teacher_fallback",
                decision_class="policy_changed_gate_rejected",
                gate_reason_codes=["path_cost_regression"],
                policy_takes_control=False,
            ),
        ]
        self._write_decisions(decisions)

        summary = run_quasi_real_guarded_ppo_rollout_pilot(
            update_smoke_root=self.update_root,
            candidate_root=self.candidate_root,
            quasi_real_root=self.quasi_real_root,
            output_root=self.output_root,
            config=self._config(min_episode_count=1, min_step_count=4, min_trainable=1),
            repo_root=self.repo_root,
            collector_replay_runner=self._passing_collector_replay(trainable_count=1),
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("quasi_real_guarded_ppo_rollout_controlled_regression", summary["reason_codes"])
        self.assertEqual(summary["trainable_transition_count"], 1)
        self.assertEqual(summary["validation_trainable_count"], 0)
        self.assertEqual(summary["test_trainable_count"], 0)
        self.assertEqual(summary["source_fallback_trainable_count"], 0)
        self.assertEqual(summary["controlled_path_risk_regression_count"], 1)
        steps = self._read_jsonl(self.output_root / "quasi-real-guarded-ppo-rollout-steps.jsonl")
        diagnostic = [step for step in steps if step["diagnostic_only"]]
        self.assertEqual(len(diagnostic), 3)
        self.assertTrue(any("gate_reason_codes_present" in step["rejection_reason_codes"] for step in diagnostic))

    def test_missing_logprob_value_or_nonfinite_reward_fails_pilot(self) -> None:
        from scripts.run_quasi_real_guarded_ppo_rollout_pilot import (
            run_quasi_real_guarded_ppo_rollout_pilot,
        )

        decisions = [
            self._decision("ctx-missing", split="train", action=0, log_prob=None, value=None),
            self._decision("ctx-bad", split="train", action=1, path_cost_delta=float("inf")),
            self._decision("ctx-good", split="train", action=0),
        ]
        self._write_decisions(decisions)

        summary = run_quasi_real_guarded_ppo_rollout_pilot(
            update_smoke_root=self.update_root,
            candidate_root=self.candidate_root,
            quasi_real_root=self.quasi_real_root,
            output_root=self.output_root,
            config=self._config(min_episode_count=1, min_step_count=3, min_trainable=1),
            repo_root=self.repo_root,
            collector_replay_runner=self._passing_collector_replay(trainable_count=1),
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("quasi_real_guarded_ppo_rollout_contract_invalid", summary["reason_codes"])
        self.assertIn("quasi_real_guarded_ppo_rollout_non_finite_return", summary["reason_codes"])
        self.assertEqual(summary["missing_log_prob_count"], 1)
        self.assertEqual(summary["missing_value_count"], 1)
        self.assertEqual(summary["non_finite_reward_count"], 1)

    def test_readiness_accepts_passed_quasi_real_guarded_ppo_rollout_pilot_summary(self) -> None:
        from scripts.run_policy_training_readiness_review import (
            _quasi_real_guarded_ppo_rollout_pilot_readiness,
        )

        readiness = _quasi_real_guarded_ppo_rollout_pilot_readiness(
            {
                "schema_version": "quasi-real-guarded-ppo-rollout-pilot-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "episode_count": 36,
                "step_count": 108,
                "trainable_transition_count": 36,
                "quasi_real_collector_replay_status": "passed",
                "quasi_real_collector_replay_trainable_transition_count": 36,
                "validation_trainable_count": 0,
                "test_trainable_count": 0,
                "source_fallback_trainable_count": 0,
                "missing_observation_count": 0,
                "missing_log_prob_count": 0,
                "missing_value_count": 0,
                "non_finite_reward_count": 0,
                "non_finite_return_count": 0,
                "non_finite_advantage_count": 0,
                "controlled_regression_count": 0,
                "controlled_safety_regression_count": 0,
                "controlled_contract_regression_count": 0,
                "controlled_path_risk_regression_count": 0,
                "controlled_source_selection_regression_count": 0,
                "teacher_agreement_rate": 1.0,
                "post_pilot_long_horizon_verdict": "long_horizon_teacher_skill_contract_aligned",
                "uses_multistep_discounted_return": True,
                "not_single_step_best_action": True,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "formal_training_ready_claimed": False,
                "git_provenance": {"current_matches_sources": True},
            }
        )

        self.assertTrue(readiness["present"])
        self.assertTrue(readiness["completed"])
        self.assertEqual(readiness["training_blockers"], [])

    def _write_inputs(self) -> None:
        self._write_json(
            self.update_root / "return-aligned-guarded-ppo-update-smoke-summary.json",
            {
                "schema_version": "return-aligned-guarded-ppo-update-smoke-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "output_root": str(self.update_root),
                "post_update_gates_evaluated": True,
                "post_update_controlled_regression_count": 0,
                "post_update_teacher_agreement_rate": 1.0,
                "post_update_quasi_real_collector_summary": str(
                    self.collector_root / "ppo-rollout-collector-summary.json"
                ),
                "post_update_long_horizon_summary": str(
                    self.long_horizon_root / "long-horizon-teacher-skill-contract-summary.json"
                ),
                "post_update_return_aligned_replay_summary": str(
                    self.replay_root / "return-aligned-collector-summary.json"
                ),
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "formal_training_ready_claimed": False,
                "git_provenance": {"current_matches_sources": True},
            },
        )
        self._write_json(
            self.teacher_root / "quasi-real-guarded-teacher-following-pilot-summary.json",
            {
                "schema_version": "quasi-real-guarded-teacher-following-pilot-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "quasi_real_context_count": 0,
                "policy_decision_count": 0,
                "teacher_agreement_rate": 1.0,
                "decisions_path": str(self.teacher_root / "quasi-real-guarded-teacher-following-decisions.jsonl"),
                "source_root": "outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1",
                "candidate_root": str(self.candidate_root),
                "quasi_real_root": str(self.quasi_real_root),
                "git_provenance": {"current_matches_sources": True},
            },
        )
        self._write_json(
            self.collector_root / "ppo-rollout-collector-summary.json",
            {
                "schema_version": "ppo-rollout-collector-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "ppo_trainable_transition_count": 3,
                "source_fallback_trainable_count": 0,
                "missing_log_prob_count": 0,
                "missing_value_count": 0,
                "non_finite_reward_count": 0,
            },
        )
        self._write_json(
            self.long_horizon_root / "long-horizon-teacher-skill-contract-summary.json",
            {
                "schema_version": "generated-sequential-long-horizon-teacher-skill-contract-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "verdict": "long_horizon_teacher_skill_contract_aligned",
                "controlled_regression_episode_count": 0,
            },
        )
        self._write_json(
            self.replay_root / "return-aligned-collector-summary.json",
            {
                "schema_version": "return-aligned-guarded-multistep-collector-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "trainable_transition_count": 3,
                "controlled_regression_count": 0,
            },
        )
        self._write_json(
            self.quasi_real_root / "quasi-real-map-path-feedback-summary.json",
            {
                "schema_version": "path-feedback-summary/v1",
                "status": "completed",
                "scenarios": [],
            },
        )
        self._write_jsonl(self.quasi_real_root / "quasi-real-map-slices.jsonl", [])

    def _write_decisions(self, decisions: list[dict]) -> None:
        self._write_jsonl(self.teacher_root / "quasi-real-guarded-teacher-following-decisions.jsonl", decisions)
        summary_path = self.teacher_root / "quasi-real-guarded-teacher-following-pilot-summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["quasi_real_context_count"] = len(decisions)
        payload["policy_decision_count"] = len(decisions)
        payload["teacher_agreement_rate"] = (
            sum(1 for decision in decisions if decision.get("teacher_following")) / len(decisions)
            if decisions
            else 0.0
        )
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _decision(
        self,
        context_id: str,
        *,
        split: str,
        action: int,
        controlled_choice_source: str = "policy_teacher_aligned",
        decision_class: str = "source_aligned",
        gate_reason_codes: list[str] | None = None,
        policy_takes_control: bool = True,
        log_prob: float | None = -0.1,
        value: float | None = 0.25,
        path_cost_delta: float = 0.0,
    ) -> dict:
        observation = {
            "candidate_feature_names": ["path_cost", "risk"],
            "candidate_features": [[1.0, 0.1], [0.5, 0.05]],
            "global_feature_names": ["step_index"],
            "global_features": [0.0],
            "action_mask": [True, True],
            "candidate_cells": [[0, 1], [1, 2]],
            "candidate_missing_feature_names": [[], []],
            "candidate_missing_indicator_names": [],
            "candidate_missing_indicators": [[], []],
        }
        return {
            "schema_version": "quasi-real-guarded-teacher-following-decision/v1",
            "context_id": context_id,
            "scenario_id": context_id.replace("ctx", "scenario"),
            "roi_group": "smooth_high_confidence",
            "split": split,
            "source_action_index": action,
            "teacher_action_index": action,
            "raw_policy_action_index": action,
            "controlled_action_index": action,
            "controlled_choice_source": controlled_choice_source,
            "decision_class": decision_class,
            "gate_reason_codes": list(gate_reason_codes or []),
            "teacher_following": controlled_choice_source == "policy_teacher_aligned",
            "safe_disagreement": controlled_choice_source == "policy_safe_disagreement",
            "unsafe_disagreement": controlled_choice_source == "teacher_fallback",
            "policy_takes_control": policy_takes_control,
            "path_cost_delta": path_cost_delta,
            "risk_delta": 0.0,
            "policy_action_log_prob": log_prob,
            "policy_value": value,
            "observation": observation,
        }

    def _config(self, *, min_episode_count: int, min_step_count: int, min_trainable: int) -> dict:
        return {
            "schema_version": "quasi-real-guarded-ppo-rollout-pilot-config/v1",
            "horizon": 3,
            "discount_factor": 0.99,
            "input_files": {
                "update_smoke_summary": "return-aligned-guarded-ppo-update-smoke-summary.json",
                "teacher_following_summary": "post_update_quasi_real_teacher_following/quasi-real-guarded-teacher-following-pilot-summary.json",
                "teacher_following_decisions": "post_update_quasi_real_teacher_following/quasi-real-guarded-teacher-following-decisions.jsonl",
                "long_horizon_summary": "post_update_long_horizon/long-horizon-teacher-skill-contract-summary.json",
            },
            "output_files": {
                "summary": "quasi-real-guarded-ppo-rollout-pilot-summary.json",
                "episodes": "quasi-real-guarded-ppo-rollout-episodes.jsonl",
                "steps": "quasi-real-guarded-ppo-rollout-steps.jsonl",
                "rejection_report": "quasi-real-guarded-ppo-rollout-rejection-report.json",
                "reward_audit": "quasi-real-guarded-ppo-rollout-reward-audit.json",
            },
            "trainable_filter": {
                "splits": ["train"],
                "controlled_choice_sources": ["policy"],
                "require_empty_gate_reason_codes": True,
            },
            "validation": {
                "min_episode_count": min_episode_count,
                "min_step_count": min_step_count,
                "min_trainable_transition_count": min_trainable,
                "min_teacher_agreement_rate": 0.9,
                "min_quasi_real_collector_replay_trainable_transition_count": min_trainable,
            },
            "reward": {
                "teacher_following_bonus": 1.0,
                "safe_disagreement_bonus": 1.0,
                "gate_regression_penalty": 1.0,
            },
            "non_goals": ["no_formal_ppo_training", "no_checkpoint_publication"],
        }

    def _passing_collector_replay(self, *, trainable_count: int = 3):
        def run(**_: object) -> dict:
            return {
                "schema_version": "ppo-rollout-collector-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "episode_count": 3,
                "step_count": 3,
                "ppo_trainable_transition_count": trainable_count,
                "diagnostic_transition_count": 0,
                "source_fallback_trainable_count": 0,
                "invalid_action_mask_count": 0,
                "empty_action_mask_count": 0,
                "missing_log_prob_count": 0,
                "missing_value_count": 0,
                "non_finite_reward_count": 0,
            }

        return run

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, allow_nan=True) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
