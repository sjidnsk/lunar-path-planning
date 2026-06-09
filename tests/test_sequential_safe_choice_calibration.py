import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class SequentialSafeChoiceCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="seq-safe-choice-"))
        self.batch_root = self.temp_dir / "sequential"
        self.source_root = self.temp_dir / "source"
        self.train_root = self.temp_dir / "train"
        self.output_root = self.temp_dir / "candidate"
        for path in (self.batch_root, self.source_root, self.train_root, self.output_root):
            path.mkdir(parents=True)
        self.git_snapshot = self._current_git_snapshot()
        self.mining_script = self.repo_root / "scripts" / "run_sequential_canary_failure_mining.sh"
        self.mining_config = self.repo_root / "configs" / "sequential_canary_failure_mining_v1.json"
        self.candidate_script = (
            self.repo_root / "scripts" / "run_sequential_safe_choice_calibration_candidate.sh"
        )
        self.candidate_config = (
            self.repo_root / "configs" / "sequential_safe_choice_calibration_candidate_v1.json"
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHON"] = str(Path("/home/kai/anaconda3/envs/lunar-explorer/bin/python"))
        return env

    def _current_git_snapshot(self) -> dict:
        def git(path: Path, *args: str) -> str | None:
            completed = subprocess.run(
                ["git", "-C", str(path), *args],
                cwd=self.repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            if completed.returncode != 0:
                return None
            return completed.stdout.strip() or None

        return {
            "parent": {
                "path": ".",
                "sha": git(self.repo_root, "rev-parse", "HEAD") or "unknown",
                "branch": git(self.repo_root, "branch", "--show-current"),
            },
            "submodules": {
                name: {
                    "path": name,
                    "sha": git(self.repo_root / name, "rev-parse", "HEAD") or "unknown",
                    "branch": git(self.repo_root / name, "branch", "--show-current"),
                }
                for name in ("dev-platform-constraints", "model-explorer", "path-planner")
            },
        }

    def _candidate(self, context_id: str, action: int, path_cost: float, risk: float) -> dict:
        return {
            "context_id": context_id,
            "context_id_schema_version": "policy-context-id/v1",
            "context_id_source": "stable_semantic_fields",
            "action_index": action,
            "source_action_index": action,
            "cell": [8 + action, 8],
            "policy_target_cell": [8 + action, 8],
            "execution_goal_cell": [8 + action, 8],
            "candidate_role": "sequential_canary_candidate",
            "reachable": True,
            "replan_required": False,
            "open_grid_fallback_used": False,
            "action_mask_valid": True,
            "path_cost": path_cost,
            "risk": risk,
            "utility": 1.0 - action * 0.1,
            "platform_goal_feasibility": {"contract_reachable": True},
            "candidate_generation": {
                "source_selection_quality_regression": False,
                "source_selection_status": "source_selected" if action == 0 else "not_source_selected",
            },
        }

    def _write_sequential_failure_artifacts(self, *, missing_alternative_context: bool = False) -> None:
        source = self._candidate("ctx-source", 0, 1.0, 0.10)
        alternative = self._candidate("ctx-policy", 1, 3.0, 0.35)
        if missing_alternative_context:
            alternative.pop("context_id")
        step = {
            "schema_version": "policy-gated-sequential-canary-step/v1",
            "episode_id": "seq-high_risk_tradeoff-f",
            "scenario_group": "high_risk_tradeoff",
            "scenario_id": "npz_seq_canary_high_risk_tradeoff_f_step00",
            "step_index": 0,
            "input_start_cell": [1, 6],
            "decision_class": "canary_rejected_policy_choice",
            "controlled_choice_source": "source_fallback",
            "canary_rejection_reason_codes": ["path_cost_regression", "risk_regression"],
            "source_selected_context_id": "ctx-source",
            "raw_policy_selected_context_id": "ctx-policy",
            "policy_selected_context_id": "ctx-policy",
            "source_selected_action_index": 0,
            "policy_selected_action_index": 1,
            "source_execution_goal_cell": [8, 8],
            "policy_execution_goal_cell": [9, 8],
            "controlled_execution_goal_cell": [8, 8],
            "path_cost_delta": 2.0,
            "risk_delta": 0.25,
            "utility_delta": -0.1,
            "preferred_candidate": source,
            "alternative_candidate": alternative,
        }
        (self.batch_root / "policy-gated-sequential-canary-steps.jsonl").write_text(
            json.dumps(step, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (self.batch_root / "policy-gated-sequential-canary-episodes.jsonl").write_text(
            json.dumps(
                {
                    "schema_version": "policy-gated-sequential-canary-episode/v1",
                    "episode_id": "seq-high_risk_tradeoff-f",
                    "scenario_group": "high_risk_tradeoff",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (self.batch_root / "policy-gated-sequential-canary-rejection-report.json").write_text(
            json.dumps(
                {
                    "schema_version": "policy-gated-sequential-canary-rejection-report/v1",
                    "failed_steps": [step],
                    "canary_rejection_reason_counts": {
                        "path_cost_regression": 1,
                        "risk_regression": 1,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.batch_root / "policy-gated-sequential-canary-rollout-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "policy-gated-sequential-canary-rollout-summary/v1",
                    "status": "failed",
                    "reason_codes": ["cumulative_path_cost_regression_count_above_threshold"],
                    "episode_count": 1,
                    "step_count": 1,
                    "canary_rejected_policy_choice_count": 1,
                    "cumulative_path_cost_regression_count": 1,
                    "cumulative_risk_regression_count": 1,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _run_mining(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                str(self.mining_script),
                "--batch-root",
                str(self.batch_root),
                "--config",
                str(self.mining_config),
            ],
            cwd=self.repo_root,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_mining_converts_rejected_sequential_step_to_hard_negative_pair(self) -> None:
        self._write_sequential_failure_artifacts()

        completed = self._run_mining()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "sequential-canary-failure-mining-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["sequential_rejected_step_count"], 1)
        self.assertEqual(summary["sequential_hard_negative_preference_pair_count"], 1)
        self.assertEqual(summary["hard_positive_added_count"], 0)
        samples = [
            json.loads(line)
            for line in (
                self.batch_root / "sequential-canary-hard-negative-preference-samples.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(samples[0]["sample_type"], "raw_policy_regression_preference_pair")
        self.assertEqual(samples[0]["sequential_sample_type"], "sequential_hard_negative_preference_pair")
        self.assertEqual(samples[0]["preferred"]["context_id"], "ctx-source")
        self.assertEqual(samples[0]["alternative"]["context_id"], "ctx-policy")
        self.assertEqual(samples[0]["episode_id"], "seq-high_risk_tradeoff-f")
        self.assertEqual(samples[0]["step_index"], 0)
        self.assertEqual(samples[0]["input_start_cell"], [1, 6])
        self.assertEqual(samples[0]["hard_positive_added_count"], 0)
        self.assertEqual(
            samples[0]["raw_policy_regression_reason_codes"],
            ["path_cost_regression", "risk_regression"],
        )

    def test_mining_excludes_step_with_missing_alternative_context(self) -> None:
        self._write_sequential_failure_artifacts(missing_alternative_context=True)

        completed = self._run_mining()

        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
        report = json.loads(
            (self.batch_root / "sequential-canary-failure-exclusion-report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["exclusion_count"], 1)
        self.assertEqual(
            report["exclusion_reason_counts"],
            {"alternative_context_id_missing": 1},
        )

    def test_mining_converts_source_aligned_safe_better_alternative_to_missed_choice_pair(self) -> None:
        self._write_sequential_failure_artifacts()
        self._append_source_aligned_missed_safe_choice_artifacts()

        completed = self._run_mining()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "sequential-canary-failure-mining-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["sequential_hard_negative_preference_pair_count"], 1)
        self.assertEqual(summary["sequential_missed_safe_choice_preference_pair_count"], 1)
        self.assertEqual(summary["sequential_preference_pair_count"], 2)
        samples = [
            json.loads(line)
            for line in (
                self.batch_root / "sequential-canary-hard-negative-preference-samples.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        missed = [
            sample
            for sample in samples
            if sample["sequential_sample_type"] == "sequential_missed_safe_choice_preference_pair"
        ]
        self.assertEqual(len(missed), 1)
        self.assertEqual(missed[0]["preferred"]["context_id"], "ctx-safe-better")
        self.assertEqual(missed[0]["alternative"]["context_id"], "ctx-source-aligned")
        self.assertEqual(missed[0]["raw_policy_regression_reason_codes"], ["missed_safe_better_choice"])
        self.assertEqual(missed[0]["hard_positive_added_count"], 0)

    def test_mining_skips_optional_source_aligned_exclusions_without_failing(self) -> None:
        self._write_sequential_failure_artifacts()
        source = self._candidate("ctx-source-invalid-mask", 0, 10.0, 0.20)
        source["reachable"] = False
        path_feedback_path = self.batch_root / "invalid-source-step" / "path-feedback-summary.json"
        path_feedback_path.parent.mkdir(parents=True, exist_ok=True)
        path_feedback_path.write_text(
            json.dumps(
                {
                    "schema_version": "path-feedback-summary/v1",
                    "diagnostic_profile": "execution",
                    "planner_extra_args": [],
                    "scenarios": [
                        {
                            "scenario_id": "npz_seq_canary_near_invalid_source_step00",
                            "scenario_group": "near_blocked_safe_alt",
                            "scenario_seed": 902,
                            "scenario_variant_id": "invalid-source-mask",
                            "path_feedback": {"candidates": [source]},
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        step = {
            "schema_version": "policy-gated-sequential-canary-step/v1",
            "episode_id": "seq-near-invalid-source",
            "scenario_group": "near_blocked_safe_alt",
            "scenario_id": "npz_seq_canary_near_invalid_source_step00",
            "run_id": "invalid-source-step",
            "source_path": str(path_feedback_path),
            "step_index": 0,
            "decision_class": "source_aligned",
            "controlled_choice_source": "source",
            "canary_rejection_reason_codes": [],
            "source_selected_context_id": "ctx-source-invalid-mask",
            "raw_policy_selected_context_id": "ctx-source-invalid-mask",
            "policy_selected_context_id": "ctx-source-invalid-mask",
            "source_selected_action_index": 0,
            "raw_policy_selected_action_index": 0,
            "policy_selected_action_index": 0,
            "source_execution_goal_cell": [8, 8],
            "policy_execution_goal_cell": [8, 8],
            "controlled_execution_goal_cell": [8, 8],
        }
        with (self.batch_root / "policy-gated-sequential-canary-steps.jsonl").open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(json.dumps(step, ensure_ascii=False) + "\n")

        completed = self._run_mining()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "sequential-canary-failure-mining-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["exclusion_count"], 0)
        self.assertEqual(
            summary["source_aligned_skipped_reason_counts"],
            {"source_action_mask_invalid": 1},
        )

    def test_mining_allows_missed_safe_choice_only_when_no_rejected_steps(self) -> None:
        self._write_sequential_coverage_failure_artifacts()
        self._append_source_aligned_missed_safe_choice_artifacts()

        completed = self._run_mining()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "sequential-canary-failure-mining-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["sequential_rejected_step_count"], 0)
        self.assertEqual(summary["sequential_hard_negative_preference_pair_count"], 0)
        self.assertEqual(summary["sequential_missed_safe_choice_preference_pair_count"], 1)
        self.assertEqual(summary["sequential_preference_pair_count"], 1)

    def test_candidate_summary_reports_sequential_weights_and_leakage(self) -> None:
        self._write_minimal_candidate_inputs()

        completed = subprocess.run(
            [
                "bash",
                str(self.candidate_script),
                "--source-root",
                str(self.source_root),
                "--sequential-mining-root",
                str(self.train_root),
                "--output-root",
                str(self.output_root),
                "--config",
                str(self.candidate_config),
                "--validate-only",
            ],
            cwd=self.repo_root,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.output_root / "sequential-safe-choice-calibration-candidate-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["sequential_hard_negative_preference_pair_count"], 2)
        self.assertEqual(summary["hard_positive_added_count"], 0)
        self.assertEqual(summary["leaked_context_id_count"], 0)
        self.assertEqual(summary["sequential_hard_negative_loss_weight"], 1.0)
        self.assertEqual(summary["path_cost_regression_negative_weight"], 1.0)
        self.assertEqual(summary["risk_regression_negative_weight"], 1.0)

    def test_candidate_includes_missed_safe_choice_pairs_with_separate_count(self) -> None:
        self._write_minimal_candidate_inputs()
        missed = {
            "schema_version": "sequential-canary-hard-negative-preference-sample/v1",
            "sample_type": "raw_policy_regression_preference_pair",
            "sequential_sample_type": "sequential_missed_safe_choice_preference_pair",
            "context_id": "ctx-safe-better",
            "alternative_context_id": "ctx-source-aligned",
            "preferred": self._candidate("ctx-safe-better", 1, 9.5, 0.18),
            "alternative": self._candidate("ctx-source-aligned", 0, 10.0, 0.2),
            "raw_policy_regression_reason_codes": ["missed_safe_better_choice"],
            "hard_positive_added_count": 0,
        }
        with self.train_root.joinpath("sequential-canary-hard-negative-preference-samples.jsonl").open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(json.dumps(missed, ensure_ascii=False) + "\n")
        summary_path = self.train_root / "sequential-canary-failure-mining-summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["sequential_missed_safe_choice_preference_pair_count"] = 1
        summary["sequential_preference_pair_count"] = 3
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        completed = subprocess.run(
            [
                "bash",
                str(self.candidate_script),
                "--source-root",
                str(self.source_root),
                "--sequential-mining-root",
                str(self.train_root),
                "--output-root",
                str(self.output_root),
                "--config",
                str(self.candidate_config),
                "--validate-only",
            ],
            cwd=self.repo_root,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        candidate_summary = json.loads(
            (self.output_root / "sequential-safe-choice-calibration-candidate-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(candidate_summary["sequential_hard_negative_preference_pair_count"], 2)
        self.assertEqual(candidate_summary["sequential_missed_safe_choice_preference_pair_count"], 1)
        self.assertEqual(candidate_summary["sequential_preference_pair_count"], 3)
        self.assertEqual(candidate_summary["train_pair_count"], 3)

    def test_candidate_rejects_val_context_leakage(self) -> None:
        self._write_minimal_candidate_inputs()
        val_root = self.temp_dir / "val"
        val_root.mkdir()
        val_root.joinpath("raw-policy-regression-diagnostics.jsonl").write_text(
            json.dumps(
                {
                    "sample_type": "raw_policy_regression_diagnostic",
                    "context_id": "ctx-source",
                    "alternative_context_id": "ctx-policy",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                "bash",
                str(self.candidate_script),
                "--source-root",
                str(self.source_root),
                "--sequential-mining-root",
                str(self.train_root),
                "--val-diagnostic-root",
                str(val_root),
                "--output-root",
                str(self.output_root),
                "--config",
                str(self.candidate_config),
                "--validate-only",
            ],
            cwd=self.repo_root,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.output_root / "sequential-safe-choice-calibration-candidate-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("train_eval_context_leakage_detected", summary["reason_codes"])
        self.assertEqual(summary["leaked_context_id_count"], 2)

    def _write_minimal_candidate_inputs(self) -> None:
        sample = {
            "schema_version": "sequential-canary-hard-negative-preference-sample/v1",
            "sample_type": "raw_policy_regression_preference_pair",
            "sequential_sample_type": "sequential_hard_negative_preference_pair",
            "context_id": "ctx-source",
            "alternative_context_id": "ctx-policy",
            "preferred": self._candidate("ctx-source", 0, 1.0, 0.1),
            "alternative": self._candidate("ctx-policy", 1, 3.0, 0.3),
            "raw_policy_regression_reason_codes": ["path_cost_regression"],
            "hard_positive_added_count": 0,
        }
        self.train_root.joinpath("sequential-canary-failure-mining-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "sequential-canary-failure-mining-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "sequential_rejected_step_count": 2,
                    "sequential_hard_negative_preference_pair_count": 2,
                    "hard_positive_added_count": 0,
                    "exclusion_count": 0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.train_root.joinpath("sequential-canary-hard-negative-preference-samples.jsonl").write_text(
            json.dumps(sample, ensure_ascii=False)
            + "\n"
            + json.dumps({**sample, "context_id": "ctx-source-2", "alternative_context_id": "ctx-policy-2"}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

    def _append_source_aligned_missed_safe_choice_artifacts(self) -> None:
        source = self._candidate("ctx-source-aligned", 0, 10.0, 0.20)
        better = self._candidate("ctx-safe-better", 1, 9.5, 0.18)
        better["utility"] = 1.1
        source["candidate_generation"]["source_selection_status"] = "source_selected"
        better["candidate_generation"]["source_selection_status"] = "not_source_selected"
        path_feedback_path = self.batch_root / "missed-step" / "path-feedback-summary.json"
        path_feedback_path.parent.mkdir(parents=True, exist_ok=True)
        path_feedback_path.write_text(
            json.dumps(
                {
                    "schema_version": "path-feedback-summary/v1",
                    "diagnostic_profile": "execution",
                    "planner_extra_args": [],
                    "scenarios": [
                        {
                            "scenario_id": "npz_seq_canary_channel_contrast_missed_step00",
                            "scenario_group": "channel_contrast",
                            "scenario_seed": 901,
                            "scenario_variant_id": "missed-safe-choice",
                            "path_feedback": {
                                "candidates": [source, better],
                            },
                        }
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        step = {
            "schema_version": "policy-gated-sequential-canary-step/v1",
            "episode_id": "seq-channel_contrast-missed",
            "scenario_group": "channel_contrast",
            "scenario_id": "npz_seq_canary_channel_contrast_missed_step00",
            "run_id": "missed-step",
            "source_path": str(path_feedback_path),
            "step_index": 0,
            "input_start_cell": [2, 6],
            "decision_class": "source_aligned",
            "controlled_choice_source": "source",
            "canary_rejection_reason_codes": [],
            "source_selected_context_id": "ctx-source-aligned",
            "raw_policy_selected_context_id": "ctx-source-aligned",
            "policy_selected_context_id": "ctx-source-aligned",
            "source_selected_action_index": 0,
            "raw_policy_selected_action_index": 0,
            "policy_selected_action_index": 0,
            "source_execution_goal_cell": [8, 8],
            "policy_execution_goal_cell": [8, 8],
            "controlled_execution_goal_cell": [8, 8],
            "path_cost_delta": 0.0,
            "risk_delta": 0.0,
            "utility_delta": 0.0,
        }
        with (self.batch_root / "policy-gated-sequential-canary-steps.jsonl").open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(json.dumps(step, ensure_ascii=False) + "\n")

    def _write_sequential_coverage_failure_artifacts(self) -> None:
        (self.batch_root / "policy-gated-sequential-canary-steps.jsonl").write_text(
            "",
            encoding="utf-8",
        )
        (self.batch_root / "policy-gated-sequential-canary-episodes.jsonl").write_text(
            json.dumps(
                {
                    "schema_version": "policy-gated-sequential-canary-episode/v1",
                    "episode_id": "seq-channel_contrast-missed",
                    "scenario_group": "channel_contrast",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (self.batch_root / "policy-gated-sequential-canary-rejection-report.json").write_text(
            json.dumps(
                {
                    "schema_version": "policy-gated-sequential-canary-rejection-report/v1",
                    "failed_steps": [],
                    "canary_rejection_reason_counts": {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.batch_root / "policy-gated-sequential-canary-rollout-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "policy-gated-sequential-canary-rollout-summary/v1",
                    "status": "failed",
                    "reason_codes": ["multi_step_accepted_episode_count_below_threshold"],
                    "episode_count": 1,
                    "step_count": 1,
                    "canary_rejected_policy_choice_count": 0,
                    "cumulative_path_cost_regression_count": 0,
                    "cumulative_risk_regression_count": 0,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
