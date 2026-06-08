import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class LimitedPolicyTrainingDryRunInputMaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        model_explorer_src = self.repo_root / "model-explorer" / "src"
        if str(model_explorer_src) not in sys.path:
            sys.path.insert(0, str(model_explorer_src))
        self.temp_dir = Path(tempfile.mkdtemp(prefix="limited-training-dry-run-"))
        self.batch_root = self.temp_dir / "batch"
        self.batch_root.mkdir(parents=True)
        self.materialization_script = (
            self.repo_root / "scripts" / "run_planner_validated_training_input_materialization.sh"
        )
        self.dry_run_script = self.repo_root / "scripts" / "run_limited_policy_training_dry_run.sh"
        self.materialization_config = self.temp_dir / "materialization-config.json"
        self.dry_run_config = self.temp_dir / "dry-run-config.json"
        self._write_materialization_config(expected_positive_count=2)
        self._write_dry_run_config(expected_positive_count=2)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHON"] = str(Path("/home/kai/anaconda3/envs/lunar-explorer/bin/python"))
        return env

    def _run_materialization(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                str(self.materialization_script),
                "--batch-root",
                str(self.batch_root),
                "--config",
                str(self.materialization_config),
            ],
            cwd=self.repo_root,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _run_dry_run(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                str(self.dry_run_script),
                "--batch-root",
                str(self.batch_root),
                "--config",
                str(self.dry_run_config),
            ],
            cwd=self.repo_root,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _write_materialization_config(self, *, expected_positive_count: int) -> None:
        self.materialization_config.write_text(
            json.dumps(
                {
                    "schema_version": "planner-validated-training-input-materialization-config/v1",
                    "output_files": {
                        "rollout_episodes": "planner-validated-rollout-episodes.jsonl",
                        "summary": "planner-validated-training-input-materialization-summary.json",
                        "exclusion_report": "planner-validated-training-exclusion-report.json",
                    },
                    "validation": {
                        "fail_on_input_failure": True,
                        "fail_on_provenance_mismatch": True,
                        "fail_on_fallback_or_open_grid": True,
                        "fail_on_safety_regression": True,
                    },
                    "expected_counts": {
                        "input_positive_count": expected_positive_count,
                    },
                    "materialization": {
                        "reward": 1.0,
                        "max_action_count": 4,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_dry_run_config(self, *, expected_positive_count: int) -> None:
        self.dry_run_config.write_text(
            json.dumps(
                {
                    "schema_version": "limited-policy-training-dry-run-config/v1",
                    "input_files": {
                        "rollout_episodes": "planner-validated-rollout-episodes.jsonl",
                        "materialization_summary": "planner-validated-training-input-materialization-summary.json",
                    },
                    "output_files": {
                        "summary": "limited-policy-training-dry-run-summary.json",
                    },
                    "validation": {
                        "expected_input_positive_count": expected_positive_count,
                        "max_invalid_action_mask_count": 0,
                        "max_empty_action_mask_count": 0,
                    },
                    "training": {
                        "seed": 0,
                        "hidden_size": 8,
                        "learning_rate": 0.001,
                        "epochs": 1,
                        "checkpoint_path": None,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_summaries(self, records: list[dict], *, mining_status: str = "passed") -> None:
        positive_records = [
            item
            for item in records
            if item["final_training_decision"]
            in {
                "selected_default_contract_trainable",
                "selected_planner_validated_distance_exception",
            }
        ]
        default_count = sum(
            1 for item in positive_records if item["final_training_decision"] == "selected_default_contract_trainable"
        )
        exception_count = len(positive_records) - default_count
        rejected_count = len(records) - len(positive_records)
        mining_summary = {
            "schema_version": "planner-validated-trainable-target-mining-summary/v1",
            "status": mining_status,
            "reason_codes": [] if mining_status == "passed" else ["synthetic_failure"],
            "current_git_provenance_mismatch_count": 0,
            "git_provenance_mismatch_count": 0,
            "fallback_or_open_grid_count": 0,
            "safety_regression_count": 0,
            "planner_validated_trainable_target_count": len(positive_records),
            "default_contract_trainable_target_count": default_count,
            "planner_validated_distance_exception_count": exception_count,
            "nontrainable_blocked_target_count": rejected_count,
            "distance_contract_blocked_count": sum(
                1 for item in records if item["final_training_decision"] == "rejected_distance_contract"
            ),
            "source_selection_not_selected_count": sum(
                1 for item in records if item["final_training_decision"] == "rejected_not_source_selected"
            ),
            "quality_regression_rejected_count": sum(
                1 for item in records if item["final_training_decision"] == "rejected_quality_regression"
            ),
            "final_decision_records": records,
        }
        candidate_summary = {
            "schema_version": "anchor-projection-candidate-generation-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "current_git_provenance_mismatch_count": 0,
            "git_provenance_mismatch_count": 0,
            "fallback_or_open_grid_count": 0,
            "safety_regression_count": 0,
            "candidate_contract_alignment_gap_count": 0,
            "context_records": records,
        }
        (self.batch_root / "planner-validated-trainable-target-mining-summary.json").write_text(
            json.dumps(mining_summary, indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "anchor-projection-candidate-generation-summary.json").write_text(
            json.dumps(candidate_summary, indent=2),
            encoding="utf-8",
        )

    def _record(
        self,
        scenario_id: str,
        decision: str,
        *,
        action_index: int = 0,
        action_mask: list[bool] | None = None,
    ) -> dict:
        return {
            "run_id": "run-a",
            "scenario_id": scenario_id,
            "source_action_index": action_index,
            "policy_target_cell": [10 + action_index, 5],
            "execution_goal_cell": [8 + action_index, 5],
            "final_training_decision": decision,
            "projection_distance_cells": 2.0,
            "projection_distance_m": 1.0,
            "source_selection_status": "source_selected",
            "target_binding_mode": "same_action_execution_substitute",
            "ppo_consumable_action": True,
            "contract_safe": decision == "selected_default_contract_trainable",
            "planner_validated_distance_exception": decision
            == "selected_planner_validated_distance_exception",
            "source_selection_quality_regression": False,
            "selected_candidate_path_cost": 1.25,
            "selected_candidate_risk": 0.2,
            "source_path": str(self.batch_root / "run-a" / "path-feedback-summary.json"),
            **({} if action_mask is None else {"action_mask": action_mask}),
        }

    def test_materialization_writes_only_selected_positive_rollout_samples(self) -> None:
        records = [
            self._record("default-positive", "selected_default_contract_trainable", action_index=0),
            self._record("exception-positive", "selected_planner_validated_distance_exception", action_index=1),
            self._record("not-selected", "rejected_not_source_selected", action_index=2),
            self._record("distance", "rejected_distance_contract", action_index=3),
        ]
        self._write_summaries(records)

        completed = self._run_materialization()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "planner-validated-training-input-materialization-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["input_positive_count"], 2)
        self.assertEqual(summary["default_contract_positive_count"], 1)
        self.assertEqual(summary["planner_validated_exception_positive_count"], 1)
        self.assertEqual(summary["excluded_nontrainable_count"], 2)
        self.assertEqual(summary["invalid_action_mask_count"], 0)
        self.assertEqual(summary["empty_action_mask_count"], 0)
        self.assertFalse(summary["runs_training"])

        from model_explorer.policy.rollout_io import read_rollout_episodes

        episodes = read_rollout_episodes(self.batch_root / "planner-validated-rollout-episodes.jsonl")
        self.assertEqual(len(episodes), 2)
        transitions = [episode.transitions[0] for episode in episodes]
        self.assertEqual([transition.action_index for transition in transitions], [0, 1])
        self.assertEqual(transitions[0].info.extra["final_training_decision"], "selected_default_contract_trainable")
        self.assertTrue(transitions[1].info.extra["planner_validated_distance_exception"])

    def test_materialization_fails_when_positive_action_mask_is_invalid(self) -> None:
        records = [
            self._record(
                "masked-positive",
                "selected_default_contract_trainable",
                action_index=1,
                action_mask=[True, False],
            ),
            self._record("exception-positive", "selected_planner_validated_distance_exception", action_index=0),
        ]
        self._write_summaries(records)

        completed = self._run_materialization()

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("invalid_action_mask", completed.stdout + completed.stderr)

    def test_dry_run_validates_and_trains_materialized_samples_without_publishing_checkpoint(self) -> None:
        records = [
            self._record("default-positive", "selected_default_contract_trainable", action_index=0),
            self._record("exception-positive", "selected_planner_validated_distance_exception", action_index=1),
        ]
        self._write_summaries(records)
        materialization = self._run_materialization()
        self.assertEqual(materialization.returncode, 0, materialization.stdout + materialization.stderr)

        completed = self._run_dry_run()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "limited-policy-training-dry-run-summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["dry_run_status"], "passed")
        self.assertEqual(summary["input_positive_count"], 2)
        self.assertEqual(summary["train_policy_sample_count"], 2)
        self.assertEqual(summary["invalid_action_mask_count"], 0)
        self.assertEqual(summary["empty_action_mask_count"], 0)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["performance_claimed"])

    def test_dry_run_refuses_missing_or_failed_materialization_input(self) -> None:
        self._write_summaries(
            [
                self._record("default-positive", "selected_default_contract_trainable", action_index=0),
                self._record("exception-positive", "selected_planner_validated_distance_exception", action_index=1),
            ],
            mining_status="failed",
        )

        completed = self._run_materialization()

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("planner_validated_mining_summary_failed", completed.stdout + completed.stderr)
