import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class ExplorationCoveragePerformanceEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_dir = str(self.repo_root / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="coverage-performance-eval-"))
        self.formal_root = self.temp_dir / "formal-training"
        self.replay_root = self.temp_dir / "post-training-replay"
        self.selected_root = self.temp_dir / "selected-candidate"
        self.shadow_root = self.temp_dir / "shadow-rollout"
        self.signal_root = self.temp_dir / "coverage-signal"
        self.output_root = self.temp_dir / "performance-evaluation"
        for path in (
            self.formal_root,
            self.replay_root,
            self.selected_root,
            self.shadow_root,
            self.signal_root,
        ):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_passes_when_selected_ppo_improves_coverage_and_efficiency(self) -> None:
        from scripts.run_exploration_coverage_performance_evaluation import (
            run_exploration_coverage_performance_evaluation,
        )

        self._write_inputs(
            [
                self._row("selected_ppo_candidate", 0, delta=0.10, path_cost=2.0, risk=0.10),
                self._row("selected_ppo_candidate", 1, delta=0.08, path_cost=1.5, risk=0.10),
                self._row("teacher", 0, delta=0.04, path_cost=2.0, risk=0.10),
                self._row("teacher", 1, delta=0.03, path_cost=1.5, risk=0.10),
                self._row("source_default", 0, delta=0.05, path_cost=2.0, risk=0.10),
                self._row("source_default", 1, delta=0.04, path_cost=1.5, risk=0.10),
            ]
        )

        summary = run_exploration_coverage_performance_evaluation(
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            shadow_root=self.shadow_root,
            coverage_signal_root=self.signal_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["coverage_performance_status"], "passed")
        self.assertEqual(summary["selected_actor"], "selected_ppo_candidate")
        self.assertGreater(summary["selected_metrics"]["coverage_return"], summary["best_baseline_metrics"]["coverage_return"])
        self.assertGreater(
            summary["selected_metrics"]["coverage_gain_per_path_cost"],
            summary["best_baseline_metrics"]["coverage_gain_per_path_cost"],
        )
        self.assertEqual(summary["controlled_regression_count"], 0)
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["formal_release_claimed"])

        for filename in (
            "exploration-coverage-performance-evaluation-summary.json",
            "coverage-performance-metric-table.jsonl",
            "coverage-performance-comparison-audit.json",
            "coverage-performance-rejection-report.json",
            "exploration-coverage-performance-evaluation-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_fails_when_selected_coverage_is_not_improved(self) -> None:
        from scripts.run_exploration_coverage_performance_evaluation import (
            run_exploration_coverage_performance_evaluation,
        )

        self._write_inputs(
            [
                self._row("selected_ppo_candidate", 0, delta=0.04, path_cost=1.0),
                self._row("selected_ppo_candidate", 1, delta=0.03, path_cost=1.0),
                self._row("teacher", 0, delta=0.04, path_cost=1.0),
                self._row("teacher", 1, delta=0.03, path_cost=1.0),
            ]
        )

        summary = run_exploration_coverage_performance_evaluation(
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            shadow_root=self.shadow_root,
            coverage_signal_root=self.signal_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["coverage_performance_status"], "failed")
        self.assertIn("coverage_performance_not_improved", summary["reason_codes"])

    def test_fails_when_coverage_improves_but_efficiency_regresses(self) -> None:
        from scripts.run_exploration_coverage_performance_evaluation import (
            run_exploration_coverage_performance_evaluation,
        )

        self._write_inputs(
            [
                self._row("selected_ppo_candidate", 0, delta=0.20, path_cost=20.0, risk=5.0),
                self._row("selected_ppo_candidate", 1, delta=0.20, path_cost=20.0, risk=5.0),
                self._row("teacher", 0, delta=0.10, path_cost=1.0, risk=0.1),
                self._row("teacher", 1, delta=0.10, path_cost=1.0, risk=0.1),
            ]
        )

        summary = run_exploration_coverage_performance_evaluation(
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            shadow_root=self.shadow_root,
            coverage_signal_root=self.signal_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("coverage_efficiency_regression", summary["reason_codes"])
        self.assertIn("path_cost_regression", summary["reason_codes"])
        self.assertIn("risk_regression", summary["reason_codes"])

    def test_fails_when_expected_coverage_is_confused_with_actual_coverage(self) -> None:
        from scripts.run_exploration_coverage_performance_evaluation import (
            run_exploration_coverage_performance_evaluation,
        )

        rows = [
            self._row("selected_ppo_candidate", 0, delta=None, expected_delta=0.10),
            self._row("teacher", 0, delta=0.05, expected_delta=0.05),
        ]
        self._write_inputs(rows, expected_actual_confusion_count=1)

        summary = run_exploration_coverage_performance_evaluation(
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            shadow_root=self.shadow_root,
            coverage_signal_root=self.signal_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("expected_actual_coverage_confusion", summary["reason_codes"])

    def test_fails_when_fallback_gain_is_claimed_as_policy_gain(self) -> None:
        from scripts.run_exploration_coverage_performance_evaluation import (
            run_exploration_coverage_performance_evaluation,
        )

        rows = [
            self._row("selected_ppo_candidate", 0, delta=0.03),
            self._row("teacher", 0, delta=0.02),
            self._row(
                "source_fallback",
                0,
                delta=0.30,
                claimed_actor="policy",
                choice_source="source_fallback",
            ),
        ]
        self._write_inputs(rows, fallback_claimed_policy_count=1)

        summary = run_exploration_coverage_performance_evaluation(
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            shadow_root=self.shadow_root,
            coverage_signal_root=self.signal_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("fallback_coverage_gain_claimed_as_policy_gain", summary["reason_codes"])
        self.assertIn("fallback_dominates_gain", summary["reason_codes"])

    def test_fails_when_controlled_regression_is_present(self) -> None:
        from scripts.run_exploration_coverage_performance_evaluation import (
            run_exploration_coverage_performance_evaluation,
        )

        rows = [
            self._row("selected_ppo_candidate", 0, delta=0.10, controlled_reasons=["risk_regression"]),
            self._row("teacher", 0, delta=0.05),
        ]
        self._write_inputs(rows, controlled_regression_count=1)

        summary = run_exploration_coverage_performance_evaluation(
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            shadow_root=self.shadow_root,
            coverage_signal_root=self.signal_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("controlled_regression_present", summary["reason_codes"])
        self.assertEqual(summary["controlled_regression_count"], 1)

    def test_fails_without_comparator_evidence(self) -> None:
        from scripts.run_exploration_coverage_performance_evaluation import (
            run_exploration_coverage_performance_evaluation,
        )

        self._write_inputs([self._row("selected_ppo_candidate", 0, delta=0.10)])

        summary = run_exploration_coverage_performance_evaluation(
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            shadow_root=self.shadow_root,
            coverage_signal_root=self.signal_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("insufficient_comparator_evidence", summary["reason_codes"])

    def test_uses_candidate_feature_costs_when_delta_cost_fields_are_zero_placeholders(self) -> None:
        from scripts.run_exploration_coverage_performance_evaluation import (
            run_exploration_coverage_performance_evaluation,
        )

        rows = [
            self._row("selected_ppo_candidate", 0, delta=0.10, path_cost=0.0, risk=0.0, energy=0.0),
            self._row("teacher", 0, delta=0.05, path_cost=0.0, risk=0.0, energy=0.0),
        ]
        self._write_inputs(rows)
        shadow_steps = [
            json.loads(line)
            for line in (self.shadow_root / "multihorizon-shadow-rollout-steps.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        for step in shadow_steps:
            is_selected = "selected_ppo_candidate" in step["context_id"]
            path_cost = 2.0 if is_selected else 4.0
            risk = 0.2 if is_selected else 0.4
            energy = 1.0 if is_selected else 2.0
            step["path_cost_delta"] = 0.0
            step["risk_delta"] = 0.0
            step["energy_cost_delta"] = 0.0
            step["observation"] = {
                "candidate_feature_names": ["value", "risk", "path_cost", "energy_cost"],
                "candidate_features": [[1.0, risk, path_cost, energy]],
                "action_mask": [True],
            }
        self._write_jsonl(self.shadow_root / "multihorizon-shadow-rollout-steps.jsonl", shadow_steps)

        summary = run_exploration_coverage_performance_evaluation(
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            shadow_root=self.shadow_root,
            coverage_signal_root=self.signal_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["selected_metrics"]["path_cost"], 2.0)
        self.assertEqual(summary["selected_metrics"]["risk"], 0.2)
        self.assertEqual(summary["selected_metrics"]["energy_cost"], 1.0)
        self.assertEqual(summary["selected_metrics"]["coverage_gain_per_path_cost"], 0.05)

    def _write_inputs(
        self,
        rows: list[dict],
        *,
        expected_actual_confusion_count: int = 0,
        fallback_claimed_policy_count: int = 0,
        controlled_regression_count: int = 0,
    ) -> None:
        self._write_json(
            self.formal_root / "formal-ppo-training-run-summary.json",
            {
                "schema_version": "guarded-formal-ppo-training-run-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "seed_count": 5,
                "optimizer_train_transition_count": 684,
                "teacher_agreement_rate": 1.0,
                "controlled_regression_count": controlled_regression_count,
                "performance_claimed": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
            },
        )
        self._write_jsonl(
            self.formal_root / "formal-ppo-training-run-seed-summaries.jsonl",
            [{"seed": seed, "status": "passed", "controlled_regression_count": 0} for seed in range(5)],
        )
        self._write_json(
            self.replay_root / "formal-ppo-post-training-stability-replay-summary.json",
            {
                "schema_version": "guarded-formal-ppo-post-training-stability-replay-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "passed_replay_count": 15,
                "teacher_agreement_rate": 1.0,
                "controlled_regression_count": controlled_regression_count,
            },
        )
        self._write_json(
            self.selected_root / "selected-formal-ppo-candidate-promotion-preflight-summary.json",
            {
                "schema_version": "selected-formal-ppo-candidate-promotion-preflight-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "selected_seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "teacher_agreement_rate": 1.0,
                "controlled_regression_count": controlled_regression_count,
            },
        )
        self._write_json(
            self.shadow_root / "multihorizon-shadow-rollout-summary.json",
            {
                "schema_version": "selected-formal-ppo-candidate-multihorizon-shadow-rollout-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "teacher_agreement_rate": 1.0,
                "controlled_regression_count": controlled_regression_count,
                "steps": str(self.shadow_root / "multihorizon-shadow-rollout-steps.jsonl"),
                "runs_new_ppo_update": False,
            },
        )
        shadow_steps = [self._shadow_step(row) for row in rows]
        self._write_jsonl(self.shadow_root / "multihorizon-shadow-rollout-steps.jsonl", shadow_steps)

        nonzero_count = sum(
            1 for row in rows if isinstance(row.get("coverage_rate_delta"), (int, float)) and row["coverage_rate_delta"] != 0
        )
        self._write_json(
            self.signal_root / "exploration-coverage-signal-audit-summary.json",
            {
                "schema_version": "exploration-coverage-signal-audit-summary/v1",
                "status": "passed" if expected_actual_confusion_count == 0 else "failed",
                "reason_codes": [],
                "coverage_signal_status": "passed" if expected_actual_confusion_count == 0 else "failed",
                "actual_coverage_gain_source": "path_feedback",
                "nonzero_actual_coverage_delta_count": nonzero_count,
                "coverage_delta_default_zero_count": 0,
                "expected_actual_coverage_confusion_count": expected_actual_confusion_count,
                "fallback_coverage_gain_claimed_as_policy_gain_count": fallback_claimed_policy_count,
                "controlled_regression_count": controlled_regression_count,
                "coverage_delta_audit": str(self.signal_root / "coverage-delta-audit.jsonl"),
            },
        )
        self._write_jsonl(self.signal_root / "coverage-delta-audit.jsonl", rows)

    def _row(
        self,
        actor: str,
        step_index: int,
        *,
        delta: float | None,
        expected_delta: float = 0.0,
        path_cost: float = 1.0,
        risk: float = 0.1,
        energy: float = 0.5,
        value: float = 1.0,
        claimed_actor: str | None = None,
        choice_source: str | None = None,
        controlled_reasons: list[str] | None = None,
    ) -> dict:
        episode_id = f"{actor}-episode"
        cumulative = delta if delta is not None else None
        return {
            "schema_version": "exploration-coverage-delta-audit-row/v1",
            "audit_index": len(actor) + step_index,
            "episode_id": episode_id,
            "source_episode_id": episode_id,
            "step_index": step_index,
            "source_step_index": step_index,
            "context_id": f"{actor}-context-{step_index}",
            "scenario_id": f"scenario-{step_index}",
            "scenario_family": "family-a",
            "split": "train",
            "controlled_choice_source": choice_source or ("policy" if actor == "selected_ppo_candidate" else actor),
            "canonical_coverage_actor": actor,
            "coverage_gain_claimed_actor": claimed_actor or actor,
            "ppo_trainable": actor == "selected_ppo_candidate",
            "initial_coverage_rate": 0.0,
            "final_coverage_rate": delta,
            "coverage_rate_delta": delta,
            "cumulative_coverage_rate_delta": cumulative,
            "expected_coverage_rate_delta": expected_delta,
            "actual_coverage_gain_source": "path_feedback" if delta is not None else "missing",
            "actual_delta_present": delta is not None,
            "actual_delta_finite": delta is not None,
            "actual_delta_nonzero": bool(delta),
            "actual_delta_default_zero": delta == 0.0,
            "expected_delta_nonzero": expected_delta != 0.0,
            "controlled_regression_reason_codes": controlled_reasons or [],
            "path_cost": path_cost,
            "risk": risk,
            "energy_cost": energy,
            "valuable_area_covered": (delta or 0.0) * value,
        }

    def _shadow_step(self, row: dict) -> dict:
        return {
            "episode_id": row["source_episode_id"],
            "shadow_episode_id": row["episode_id"],
            "step_index": row["source_step_index"],
            "shadow_step_index": row["step_index"],
            "context_id": row["context_id"],
            "scenario_id": row["scenario_id"],
            "scenario_family": row["scenario_family"],
            "split": row["split"],
            "controlled_choice_source": row["controlled_choice_source"],
            "controlled_action_index": 0,
            "teacher_action_index": 0,
            "policy_takes_control": row["canonical_coverage_actor"] == "selected_ppo_candidate",
            "path_cost_delta": row["path_cost"],
            "risk_delta": row["risk"],
            "energy_cost_delta": row["energy_cost"],
            "controlled_regression_reason_codes": row["controlled_regression_reason_codes"],
            "observation": {
                "candidate_feature_names": ["value", "risk", "path_cost", "energy_cost"],
                "candidate_features": [[1.0, row["risk"], row["path_cost"], row["energy_cost"]]],
                "action_mask": [True],
            },
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


if __name__ == "__main__":
    unittest.main()
