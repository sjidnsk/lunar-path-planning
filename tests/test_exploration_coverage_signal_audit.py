import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class ExplorationCoverageSignalAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_dir = str(self.repo_root / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="coverage-signal-audit-"))
        self.formal_root = self.temp_dir / "formal-training"
        self.selected_root = self.temp_dir / "selected-candidate"
        self.shadow_root = self.temp_dir / "shadow-rollout"
        self.output_root = self.temp_dir / "audit"
        self.formal_root.mkdir(parents=True)
        self.selected_root.mkdir(parents=True)
        self.shadow_root.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_passes_when_actual_coverage_gain_is_nonzero_sourced_and_attributed(self) -> None:
        from scripts.run_exploration_coverage_signal_audit import (
            run_exploration_coverage_signal_audit,
        )

        self._write_inputs(
            [
                self._step(0, initial=0.10, final=0.15, delta=0.05, cumulative=0.05),
                self._step(1, initial=0.15, final=0.18, delta=0.03, cumulative=0.08),
                self._step(
                    2,
                    initial=0.18,
                    final=0.20,
                    delta=0.02,
                    cumulative=0.10,
                    choice_source="source_fallback",
                    claimed_actor="source_fallback",
                    ppo_trainable=False,
                ),
            ]
        )

        summary = run_exploration_coverage_signal_audit(
            formal_training_root=self.formal_root,
            selected_candidate_root=self.selected_root,
            shadow_root=self.shadow_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["coverage_signal_status"], "passed")
        self.assertEqual(summary["coverage_field_presence_status"], "passed")
        self.assertEqual(summary["actual_coverage_gain_source"], "sidecar")
        self.assertEqual(summary["nonzero_actual_coverage_delta_count"], 3)
        self.assertEqual(summary["coverage_delta_default_zero_count"], 0)
        self.assertEqual(summary["expected_actual_coverage_confusion_count"], 0)
        self.assertTrue(summary["multi_step_state_update_verified"])
        self.assertEqual(summary["policy_source_fallback_coverage_attribution_status"], "passed")
        self.assertEqual(summary["fallback_coverage_gain_claimed_as_policy_gain_count"], 0)
        self.assertEqual(summary["controlled_regression_count"], 0)
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])
        self.assertFalse(summary["formal_training_ready_claimed"])

        for filename in (
            "exploration-coverage-signal-audit-summary.json",
            "coverage-field-presence-audit.json",
            "coverage-delta-audit.jsonl",
            "coverage-attribution-audit.json",
            "coverage-state-transition-audit.json",
            "coverage-signal-rejection-report.json",
            "exploration-coverage-signal-audit-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_fails_with_insufficient_signal_when_actual_coverage_is_all_zero(self) -> None:
        from scripts.run_exploration_coverage_signal_audit import (
            run_exploration_coverage_signal_audit,
        )

        self._write_inputs(
            [
                self._step(0, initial=0.0, final=0.0, delta=0.0, cumulative=0.0),
                self._step(1, initial=0.0, final=0.0, delta=0.0, cumulative=0.0),
            ]
        )

        summary = run_exploration_coverage_signal_audit(
            formal_training_root=self.formal_root,
            selected_candidate_root=self.selected_root,
            shadow_root=self.shadow_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["coverage_signal_status"], "failed")
        self.assertIn("insufficient_exploration_coverage_signal", summary["reason_codes"])
        self.assertEqual(summary["nonzero_actual_coverage_delta_count"], 0)
        self.assertEqual(summary["coverage_delta_default_zero_count"], 2)

        rejection = self._load_json(self.output_root / "coverage-signal-rejection-report.json")
        self.assertIn("actual_coverage_delta_all_zero", rejection["rejection_reason_counts"])

    def test_flags_expected_actual_confusion_without_actual_delta(self) -> None:
        from scripts.run_exploration_coverage_signal_audit import (
            run_exploration_coverage_signal_audit,
        )

        step = self._step(0, initial=None, final=None, delta=None, cumulative=None)
        step["expected_coverage_rate_delta"] = 0.07
        step["observation"]["candidate_features"][0][7] = 0.07
        self._write_inputs([step])

        summary = run_exploration_coverage_signal_audit(
            formal_training_root=self.formal_root,
            selected_candidate_root=self.selected_root,
            shadow_root=self.shadow_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("expected_actual_coverage_confusion", summary["reason_codes"])
        self.assertEqual(summary["expected_actual_coverage_confusion_count"], 1)

    def test_flags_fallback_gain_claimed_as_policy_gain(self) -> None:
        from scripts.run_exploration_coverage_signal_audit import (
            run_exploration_coverage_signal_audit,
        )

        self._write_inputs(
            [
                self._step(
                    0,
                    initial=0.20,
                    final=0.25,
                    delta=0.05,
                    cumulative=0.05,
                    choice_source="source_fallback",
                    claimed_actor="policy",
                    ppo_trainable=False,
                )
            ]
        )

        summary = run_exploration_coverage_signal_audit(
            formal_training_root=self.formal_root,
            selected_candidate_root=self.selected_root,
            shadow_root=self.shadow_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("fallback_coverage_gain_claimed_as_policy_gain", summary["reason_codes"])
        self.assertEqual(summary["fallback_coverage_gain_claimed_as_policy_gain_count"], 1)
        self.assertEqual(summary["policy_source_fallback_coverage_attribution_status"], "failed")

    def test_flags_cumulative_delta_mismatch_and_static_multistep_state(self) -> None:
        from scripts.run_exploration_coverage_signal_audit import (
            run_exploration_coverage_signal_audit,
        )

        self._write_inputs(
            [
                self._step(0, initial=0.10, final=0.12, delta=0.02, cumulative=0.02),
                self._step(1, initial=0.10, final=0.12, delta=0.02, cumulative=0.09),
            ]
        )

        summary = run_exploration_coverage_signal_audit(
            formal_training_root=self.formal_root,
            selected_candidate_root=self.selected_root,
            shadow_root=self.shadow_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("cumulative_coverage_delta_mismatch", summary["reason_codes"])
        self.assertIn("multi_step_state_not_updated", summary["reason_codes"])
        self.assertFalse(summary["multi_step_state_update_verified"])

    def test_backfills_actual_coverage_from_path_feedback_summary_when_steps_lack_fields(self) -> None:
        from scripts.run_exploration_coverage_signal_audit import (
            run_exploration_coverage_signal_audit,
        )

        shadow_steps = [
            self._step(0, initial=None, final=None, delta=None, cumulative=None),
            self._step(1, initial=None, final=None, delta=None, cumulative=None),
        ]
        shadow_steps[0]["episode_id"] = "source-episode-a"
        shadow_steps[0]["step_index"] = 0
        shadow_steps[0]["shadow_episode_id"] = "shadow-a"
        shadow_steps[0]["shadow_step_index"] = 0
        shadow_steps[1]["episode_id"] = "source-episode-b"
        shadow_steps[1]["step_index"] = 0
        shadow_steps[1]["shadow_episode_id"] = "shadow-a"
        shadow_steps[1]["shadow_step_index"] = 1
        self._write_inputs(shadow_steps)
        path_feedback_root = (
            self.temp_dir
            / "outputs"
            / "path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1"
        )
        self._write_json(
            path_feedback_root / "quasi-real-map-path-feedback-summary.json",
            {
                "schema_version": "path-feedback-summary/v1",
                "scenarios": [
                    {
                        "scenario_id": "scenario-0",
                        "coverage_rate_delta": 0.04,
                        "baseline_vs_feedback": {"coverage_rate_delta": 0.04},
                    },
                    {
                        "scenario_id": "scenario-1",
                        "coverage_rate_delta": 0.03,
                        "baseline_vs_feedback": {"coverage_rate_delta": 0.03},
                    },
                ],
            },
        )

        summary = run_exploration_coverage_signal_audit(
            formal_training_root=self.formal_root,
            selected_candidate_root=self.selected_root,
            shadow_root=self.shadow_root,
            output_root=self.output_root,
            repo_root=self.temp_dir,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["coverage_signal_status"], "passed")
        self.assertEqual(summary["coverage_field_presence_status"], "passed")
        self.assertEqual(summary["actual_coverage_gain_source"], "path_feedback")
        self.assertEqual(summary["nonzero_actual_coverage_delta_count"], 2)
        self.assertEqual(summary["coverage_delta_default_zero_count"], 0)
        self.assertTrue(summary["multi_step_state_update_verified"])
        delta_rows = [
            json.loads(line)
            for line in (self.output_root / "coverage-delta-audit.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual([row["initial_coverage_rate"] for row in delta_rows], [0.0, 0.04])
        self.assertEqual([row["final_coverage_rate"] for row in delta_rows], [0.04, 0.07])
        self.assertEqual([row["cumulative_coverage_rate_delta"] for row in delta_rows], [0.04, 0.07])

    def _write_inputs(self, shadow_steps: list[dict]) -> None:
        self._write_json(
            self.formal_root / "formal-ppo-training-run-summary.json",
            {
                "schema_version": "guarded-formal-ppo-training-run-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "seed_count": 1,
                "passed_seed_count": 1,
                "optimizer_train_transition_count": len(shadow_steps),
                "teacher_agreement_rate": 1.0,
                "controlled_regression_count": 0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "formal_training_ready_claimed": False,
            },
        )
        collector_root = self.formal_root / "seed-00" / "collector"
        collector_root.mkdir(parents=True)
        self._write_jsonl(
            collector_root / "ppo-rollout-episodes.jsonl",
            [
                {
                    "episode_id": "episode-a",
                    "transitions": [
                        {
                            "info": dict(step),
                            "observation": step["observation"],
                            "action_index": step["controlled_action_index"],
                        }
                        for step in shadow_steps
                    ],
                    "metrics": {
                        "final_coverage_rate": shadow_steps[-1].get("final_coverage_rate"),
                        "cumulative_coverage_rate_delta": shadow_steps[-1].get(
                            "cumulative_coverage_rate_delta"
                        ),
                    },
                }
            ],
        )
        self._write_jsonl(
            self.formal_root / "formal-ppo-training-run-seed-summaries.jsonl",
            [
                {
                    "schema_version": "guarded-formal-ppo-training-run-seed-summary/v1",
                    "status": "passed",
                    "seed": 0,
                    "collector_root": str(collector_root),
                    "controlled_regression_count": 0,
                    "teacher_agreement_rate": 1.0,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                }
            ],
        )
        steps_path = self.shadow_root / "multihorizon-shadow-rollout-steps.jsonl"
        self._write_jsonl(steps_path, shadow_steps)
        self._write_json(
            self.shadow_root / "multihorizon-shadow-rollout-summary.json",
            {
                "schema_version": "selected-formal-ppo-candidate-multihorizon-shadow-rollout-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "steps": str(steps_path),
                "controlled_regression_count": 0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "runs_new_ppo_update": False,
                "performance_claimed": False,
                "formal_training_ready_claimed": False,
            },
        )
        self._write_json(
            self.selected_root / "promotion-candidate-manifest.json",
            {
                "schema_version": "selected-formal-ppo-candidate-promotion-manifest/v1",
                "selected_seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "multihorizon_root": str(self.shadow_root),
                "multihorizon_steps": str(steps_path),
                "experimental_candidate_only": True,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "formal_training_ready_claimed": False,
            },
        )

    def _step(
        self,
        index: int,
        *,
        initial: float | None,
        final: float | None,
        delta: float | None,
        cumulative: float | None,
        choice_source: str = "policy",
        claimed_actor: str = "policy",
        ppo_trainable: bool = True,
    ) -> dict:
        step = {
            "schema_version": "selected-formal-ppo-candidate-multihorizon-shadow-step/v1",
            "episode_id": "episode-a",
            "shadow_episode_id": "shadow-a",
            "step_index": index,
            "shadow_step_index": index,
            "scenario_id": f"scenario-{index}",
            "scenario_family": "coverage_fixture",
            "split": "train",
            "controlled_choice_source": choice_source,
            "controlled_choice_detail": "policy_teacher_aligned"
            if choice_source == "policy"
            else choice_source,
            "controlled_action_index": 0,
            "teacher_action_index": 0,
            "source_action_index": 1 if choice_source == "source_fallback" else None,
            "ppo_trainable": ppo_trainable,
            "shadow_trainable": ppo_trainable,
            "controlled_regression_reason_codes": [],
            "coverage_gain_claimed_actor": claimed_actor,
            "coverage_gain_source": "sidecar",
            "observation": {
                "candidate_feature_names": [
                    "cell_x",
                    "cell_y",
                    "relative_dx",
                    "relative_dy",
                    "relative_distance",
                    "utility",
                    "reachable",
                    "expected_coverage_rate_delta",
                ],
                "candidate_features": [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.01]],
                "global_feature_names": ["coverage_rate", "step_index"],
                "global_features": [initial or 0.0, float(index)],
            },
        }
        if initial is not None:
            step["initial_coverage_rate"] = initial
        if final is not None:
            step["final_coverage_rate"] = final
        if delta is not None:
            step["coverage_rate_delta"] = delta
        if cumulative is not None:
            step["cumulative_coverage_rate_delta"] = cumulative
        step["expected_coverage_rate_delta"] = 0.01
        return step

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _load_json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
