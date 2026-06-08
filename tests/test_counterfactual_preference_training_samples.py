import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class CounterfactualPreferenceTrainingSamplesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="counterfactual-preference-"))
        self.batch_root = self.temp_dir / "batch"
        self.batch_root.mkdir(parents=True)
        self.sample_script = self.repo_root / "scripts" / "run_counterfactual_preference_training_samples.sh"
        self.dry_run_script = (
            self.repo_root / "scripts" / "run_counterfactual_preference_training_dry_run.sh"
        )
        self.sample_config = self.temp_dir / "counterfactual-config.json"
        self.dry_run_config = self.temp_dir / "counterfactual-dry-run-config.json"
        self._write_sample_config()
        self._write_dry_run_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHON"] = str(Path("/home/kai/anaconda3/envs/lunar-explorer/bin/python"))
        return env

    def _run_samples(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                str(self.sample_script),
                "--batch-root",
                str(self.batch_root),
                "--config",
                str(self.sample_config),
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

    def _write_sample_config(self) -> None:
        self.sample_config.write_text(
            json.dumps(
                {
                    "schema_version": "counterfactual-preference-training-samples-config/v1",
                    "input_files": {
                        "planner_validated_trainable_target_mining_summary": "planner-validated-trainable-target-mining-summary.json",
                        "anchor_projection_candidate_generation_summary": "anchor-projection-candidate-generation-summary.json",
                    },
                    "output_files": {
                        "samples": "counterfactual-preference-training-samples.jsonl",
                        "summary": "counterfactual-preference-training-summary.json",
                        "exclusion_report": "counterfactual-preference-exclusion-report.json",
                    },
                    "validation": {
                        "fail_on_input_failure": True,
                        "fail_on_provenance_mismatch": True,
                        "fail_on_fallback_or_open_grid": True,
                        "fail_on_safety_regression": True,
                    },
                    "expected_counts": {
                        "source_selection_not_selected_count": 36,
                        "preference_pair_count": 24,
                        "selected_over_alternative_negative_count": 12,
                        "tradeoff_preference_pair_count": 12,
                        "rejected_binding_or_distance_required_count": 12,
                        "hard_positive_added_count": 0,
                    },
                    "preference": {
                        "selected_over_alternative_weight": 1.0,
                        "tradeoff_weight": 0.35,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_dry_run_config(self) -> None:
        self.dry_run_config.write_text(
            json.dumps(
                {
                    "schema_version": "counterfactual-preference-training-dry-run-config/v1",
                    "input_files": {
                        "samples": "counterfactual-preference-training-samples.jsonl",
                        "summary": "counterfactual-preference-training-summary.json",
                    },
                    "output_files": {
                        "summary": "counterfactual-preference-training-dry-run-summary.json",
                    },
                    "validation": {
                        "expected_preference_pair_count": 24,
                    },
                    "training": {
                        "seed": 0,
                        "hidden_size": 8,
                        "learning_rate": 0.001,
                        "epochs": 1,
                        "margin": 0.1,
                        "checkpoint_path": None,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_summaries(self) -> None:
        records = []
        records.extend(
            self._record(
                scenario_id=f"npz_high_risk_value_trap_{index}",
                path_margin=2.0 + index * 0.1,
                risk_margin=0.01,
                contract_safe=True,
            )
            for index in range(12)
        )
        records.extend(
            self._record(
                scenario_id=f"npz_path_complexity_benefit_probe_{index}",
                path_margin=3.0 + index * 0.1,
                risk_margin=-0.02,
                contract_safe=True,
            )
            for index in range(12)
        )
        records.extend(
            self._record(
                scenario_id=f"npz_near_blocked_corridor_{index}",
                path_margin=0.2,
                risk_margin=-0.01,
                contract_safe=False,
            )
            for index in range(12)
        )
        mining = {
            "schema_version": "planner-validated-trainable-target-mining-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "current_git_provenance_mismatch_count": 0,
            "git_provenance_mismatch_count": 0,
            "fallback_or_open_grid_count": 0,
            "safety_regression_count": 0,
            "planner_validated_trainable_target_count": 24,
            "default_contract_trainable_target_count": 18,
            "planner_validated_distance_exception_count": 6,
            "nontrainable_blocked_target_count": 54,
            "source_selection_not_selected_count": 36,
            "final_decision_records": records,
        }
        candidate = {
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
            json.dumps(mining, indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "anchor-projection-candidate-generation-summary.json").write_text(
            json.dumps(candidate, indent=2),
            encoding="utf-8",
        )

    def _record(
        self,
        *,
        scenario_id: str,
        path_margin: float,
        risk_margin: float,
        contract_safe: bool,
    ) -> dict:
        selected_path_cost = 10.0
        selected_risk = 0.2
        return {
            "run_id": "run-a",
            "scenario_id": scenario_id,
            "source_action_index": 1,
            "policy_target_cell": [18, 6],
            "execution_goal_cell": [17, 6],
            "selected_action_index": 0,
            "selected_cell": [19, 12],
            "source_selection_best_alternative_action_index": 3,
            "source_selection_best_alternative_cell": [19, 13],
            "final_training_decision": "rejected_not_source_selected",
            "source_selection_status": "not_source_selected",
            "target_binding_mode": "synthetic_projection",
            "ppo_consumable_action": False,
            "contract_safe": contract_safe,
            "planner_validated_distance_exception": False,
            "source_selection_quality_regression": False,
            "projection_distance_cells": 2.0 if contract_safe else 3.0,
            "projection_distance_m": 1.0 if contract_safe else 1.5,
            "source_selection_path_cost_margin_vs_best_alternative": path_margin,
            "source_selection_risk_margin_vs_best_alternative": risk_margin,
            "selected_candidate_path_cost": selected_path_cost,
            "selected_candidate_risk": selected_risk,
            "selected_candidate_utility": 0.8,
            "projected_candidate_path_cost": selected_path_cost + path_margin,
            "projected_candidate_risk": selected_risk + risk_margin,
            "projected_candidate_utility": 0.6,
        }

    def test_mining_classifies_all_not_selected_records_without_hard_positives(self) -> None:
        self._write_summaries()

        completed = self._run_samples()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "counterfactual-preference-training-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["source_selection_not_selected_count"], 36)
        self.assertEqual(summary["preference_pair_count"], 24)
        self.assertEqual(summary["selected_over_alternative_negative_count"], 12)
        self.assertEqual(summary["tradeoff_preference_pair_count"], 12)
        self.assertEqual(summary["rejected_binding_or_distance_required_count"], 12)
        self.assertEqual(summary["hard_positive_added_count"], 0)
        self.assertFalse(summary["runs_training"])

        samples = [
            json.loads(line)
            for line in (self.batch_root / "counterfactual-preference-training-samples.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(samples), 24)
        self.assertEqual(
            {sample["preference_decision"] for sample in samples},
            {"selected_over_alternative_negative", "tradeoff_preference_pair"},
        )
        self.assertTrue(all(sample["hard_positive"] is False for sample in samples))
        self.assertTrue(all(sample["alternative"]["ppo_consumable_action"] is False for sample in samples))

    def test_dry_run_trains_pairwise_preference_without_checkpoint_or_policy_claim(self) -> None:
        self._write_summaries()
        sample_run = self._run_samples()
        self.assertEqual(sample_run.returncode, 0, sample_run.stdout + sample_run.stderr)

        completed = self._run_dry_run()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "counterfactual-preference-training-dry-run-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["preference_dry_run_status"], "passed")
        self.assertEqual(summary["preference_train_sample_count"], 24)
        self.assertEqual(summary["hard_positive_added_count"], 0)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["performance_claimed"])
