import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class UnifiedPolicySampleRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="unified-policy-samples-"))
        self.batch_root = self.temp_dir / "batch"
        self.batch_root.mkdir(parents=True)
        self.registry_script = self.repo_root / "scripts" / "run_unified_policy_sample_registry.sh"
        self.dry_run_script = (
            self.repo_root / "scripts" / "run_residual_boundary_preference_training_dry_run.sh"
        )
        self.registry_config = self.temp_dir / "unified-registry-config.json"
        self.dry_run_config = self.temp_dir / "residual-dry-run-config.json"
        self._write_registry_config()
        self._write_dry_run_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHON"] = str(Path("/home/kai/anaconda3/envs/lunar-explorer/bin/python"))
        return env

    def _run_registry(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                str(self.registry_script),
                "--batch-root",
                str(self.batch_root),
                "--config",
                str(self.registry_config),
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

    def _write_registry_config(self) -> None:
        self.registry_config.write_text(
            json.dumps(
                {
                    "schema_version": "unified-policy-sample-registry-config/v1",
                    "input_files": {
                        "planner_validated_trainable_target_mining_summary": "planner-validated-trainable-target-mining-summary.json",
                        "anchor_projection_candidate_generation_summary": "anchor-projection-candidate-generation-summary.json",
                        "planner_validated_training_input_materialization_summary": "planner-validated-training-input-materialization-summary.json",
                        "planner_validated_rollout_episodes": "planner-validated-rollout-episodes.jsonl",
                        "counterfactual_preference_training_summary": "counterfactual-preference-training-summary.json",
                        "counterfactual_preference_training_samples": "counterfactual-preference-training-samples.jsonl",
                        "counterfactual_preference_exclusion_report": "counterfactual-preference-exclusion-report.json",
                    },
                    "output_files": {
                        "registry": "unified-policy-sample-registry.jsonl",
                        "summary": "unified-policy-sample-registry-summary.json",
                        "exclusion_report": "unified-policy-sample-exclusion-report.json",
                    },
                    "validation": {
                        "fail_on_input_failure": True,
                        "fail_on_provenance_mismatch": True,
                        "fail_on_fallback_or_open_grid": True,
                        "fail_on_safety_regression": True,
                    },
                    "expected_counts": {
                        "action_label_positive_count": 24,
                        "existing_preference_pair_count": 24,
                        "boundary_negative_preference_pair_count": 12,
                        "blocked_target_negative_pair_count": 18,
                        "residual_trainable_signal_count": 30,
                        "pairwise_preference_signal_count": 54,
                        "unified_context_coverage_count": 78,
                        "hard_positive_added_count": 0,
                    },
                    "residual_preference": {
                        "boundary_negative_weight": 0.25,
                        "blocked_target_negative_weight": 1.0,
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
                    "schema_version": "residual-boundary-preference-training-dry-run-config/v1",
                    "input_files": {
                        "registry": "unified-policy-sample-registry.jsonl",
                        "summary": "unified-policy-sample-registry-summary.json",
                    },
                    "output_files": {
                        "summary": "residual-boundary-preference-training-dry-run-summary.json",
                    },
                    "validation": {
                        "expected_residual_train_sample_count": 30,
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

    def _write_artifacts(self, *, missing_dense_metric: bool = False) -> None:
        records = []
        records.extend(
            self._record(
                scenario_id=f"default-positive-{index}",
                decision="selected_default_contract_trainable",
                source_action_index=index % 4,
            )
            for index in range(18)
        )
        records.extend(
            self._record(
                scenario_id=f"exception-positive-{index}",
                decision="selected_planner_validated_distance_exception",
                source_action_index=index % 4,
                contract_safe=False,
                planner_exception=True,
                projection_distance_cells=3.0,
                projection_distance_m=1.5,
            )
            for index in range(6)
        )
        records.extend(
            self._record(
                scenario_id=f"existing-preference-{index}",
                decision="rejected_not_source_selected",
                source_action_index=1,
                contract_safe=True,
                projection_distance_cells=1.0,
                projection_distance_m=0.5,
            )
            for index in range(24)
        )
        records.extend(
            self._record(
                scenario_id=f"npz_near_blocked_corridor_{index}",
                decision="rejected_not_source_selected",
                source_action_index=1,
                contract_safe=False,
                projection_distance_cells=3.0,
                projection_distance_m=1.5,
            )
            for index in range(12)
        )
        dense = [
            self._record(
                scenario_id=f"npz_dense_rock_choke_{index}",
                decision="rejected_distance_contract",
                source_action_index=index % 4,
                contract_safe=False,
                projection_distance_cells=7.0 if index < 12 else 11.0,
                projection_distance_m=3.5 if index < 12 else 5.5,
                source_selection_status="source_selected_not_ppo_consumable",
            )
            for index in range(18)
        ]
        if missing_dense_metric:
            dense[0].pop("projected_candidate_utility")
        records.extend(dense)
        self._write_mining_and_candidate_summaries(records)
        self._write_materialization_artifacts()
        self._write_counterfactual_preference_artifacts()

    def _write_mining_and_candidate_summaries(self, records: list[dict]) -> None:
        mining_summary = {
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
            "distance_contract_blocked_count": 18,
            "source_selection_not_selected_count": 36,
            "quality_regression_rejected_count": 0,
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

    def _write_materialization_artifacts(self) -> None:
        summary = {
            "schema_version": "planner-validated-training-input-materialization-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "input_positive_count": 24,
            "default_contract_positive_count": 18,
            "planner_validated_exception_positive_count": 6,
            "excluded_nontrainable_count": 54,
            "publishes_checkpoint": False,
            "performance_claimed": False,
        }
        (self.batch_root / "planner-validated-training-input-materialization-summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "planner-validated-rollout-episodes.jsonl").write_text(
            "".join(json.dumps({"sample_index": index}) + "\n" for index in range(24)),
            encoding="utf-8",
        )

    def _write_counterfactual_preference_artifacts(self) -> None:
        summary = {
            "schema_version": "counterfactual-preference-training-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "source_selection_not_selected_count": 36,
            "preference_pair_count": 24,
            "selected_over_alternative_negative_count": 12,
            "tradeoff_preference_pair_count": 12,
            "rejected_binding_or_distance_required_count": 12,
            "hard_positive_added_count": 0,
        }
        samples = [
            self._preference_sample(
                index=index,
                decision=(
                    "selected_over_alternative_negative"
                    if index < 12
                    else "tradeoff_preference_pair"
                ),
            )
            for index in range(24)
        ]
        exclusions = [
            {
                "run_id": "run-a",
                "scenario_id": f"npz_near_blocked_corridor_{index}",
                "source_action_index": 1,
                "policy_target_cell": [20, index],
                "execution_goal_cell": [17, index],
                "preference_decision": "rejected_binding_or_distance_required",
                "contract_safe": False,
                "ppo_consumable_action": False,
                "target_binding_mode": "synthetic_projection",
                "projection_distance_cells": 3.0,
                "projection_distance_m": 1.5,
            }
            for index in range(12)
        ]
        (self.batch_root / "counterfactual-preference-training-summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "counterfactual-preference-training-samples.jsonl").write_text(
            "".join(json.dumps(sample) + "\n" for sample in samples),
            encoding="utf-8",
        )
        (self.batch_root / "counterfactual-preference-exclusion-report.json").write_text(
            json.dumps(
                {
                    "schema_version": "counterfactual-preference-exclusion-report/v1",
                    "status": "passed",
                    "excluded_count": 12,
                    "excluded_decision_counts": {"rejected_binding_or_distance_required": 12},
                    "excluded_records": exclusions,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _record(
        self,
        *,
        scenario_id: str,
        decision: str,
        source_action_index: int,
        contract_safe: bool = True,
        planner_exception: bool = False,
        projection_distance_cells: float = 2.0,
        projection_distance_m: float = 1.0,
        source_selection_status: str = "not_source_selected",
    ) -> dict:
        selected_path_cost = 8.0 + source_action_index
        selected_risk = 0.2 + source_action_index * 0.01
        return {
            "run_id": "run-a",
            "scenario_id": scenario_id,
            "source_action_index": source_action_index,
            "policy_target_cell": [20, source_action_index],
            "execution_goal_cell": [17, source_action_index],
            "selected_action_index": 0,
            "selected_cell": [19, source_action_index],
            "final_training_decision": decision,
            "source_selection_status": source_selection_status,
            "target_binding_mode": (
                "same_action_execution_substitute"
                if decision.startswith("selected_")
                else "synthetic_projection"
            ),
            "ppo_consumable_action": decision.startswith("selected_"),
            "contract_safe": contract_safe,
            "planner_validated_distance_exception": planner_exception,
            "source_selection_quality_regression": False,
            "projection_distance_cells": projection_distance_cells,
            "projection_distance_m": projection_distance_m,
            "selected_candidate_path_cost": selected_path_cost,
            "selected_candidate_risk": selected_risk,
            "selected_candidate_utility": 0.8,
            "projected_candidate_path_cost": selected_path_cost + projection_distance_cells,
            "projected_candidate_risk": selected_risk + 0.05,
            "projected_candidate_utility": 0.5,
        }

    def _preference_sample(self, *, index: int, decision: str) -> dict:
        return {
            "schema_version": "counterfactual-preference-training-summary/v1",
            "sample_index": index,
            "run_id": "run-a",
            "scenario_id": f"existing-preference-{index}",
            "preference_decision": decision,
            "hard_positive": False,
            "sample_weight": 1.0,
            "selected": {
                "action_index": 0,
                "cell": [19, index % 4],
                "path_cost": 1.0,
                "risk": 0.2,
                "utility": 0.8,
                "candidate_features": [0.1] * 15,
            },
            "alternative": {
                "source_action_index": 1,
                "policy_target_cell": [20, index % 4],
                "execution_goal_cell": [17, index % 4],
                "target_binding_mode": "synthetic_projection",
                "ppo_consumable_action": False,
                "contract_safe": True,
                "planner_validated_distance_exception": False,
                "path_cost": 2.0,
                "risk": 0.3,
                "utility": 0.5,
                "candidate_features": [0.2] * 15,
            },
            "margins": {
                "projection_distance_cells": 1.0,
                "projection_distance_m": 0.5,
            },
            "global_features": [0.0] * 8,
            "candidate_missing_indicators": [[0.0] * 8, [0.0] * 8],
            "selection_contract": {
                "source_selection_status": "not_source_selected",
                "not_source_selected_is_not_hard_positive": True,
            },
        }

    def test_registry_unifies_24_24_12_18_without_adding_hard_positives(self) -> None:
        self._write_artifacts()

        completed = self._run_registry()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "unified-policy-sample-registry-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["action_label_positive_count"], 24)
        self.assertEqual(summary["existing_preference_pair_count"], 24)
        self.assertEqual(summary["boundary_negative_preference_pair_count"], 12)
        self.assertEqual(summary["blocked_target_negative_pair_count"], 18)
        self.assertEqual(summary["residual_trainable_signal_count"], 30)
        self.assertEqual(summary["pairwise_preference_signal_count"], 54)
        self.assertEqual(summary["unified_context_coverage_count"], 78)
        self.assertEqual(summary["hard_positive_added_count"], 0)

        registry = [
            json.loads(line)
            for line in (self.batch_root / "unified-policy-sample-registry.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(registry), 78)
        boundary = [
            item
            for item in registry
            if item["sample_type"] == "boundary_negative_preference_pair"
        ]
        blocked = [
            item for item in registry if item["sample_type"] == "blocked_target_negative_pair"
        ]
        self.assertEqual(len(boundary), 12)
        self.assertEqual(len(blocked), 18)
        self.assertTrue(all(item["hard_positive"] is False for item in boundary + blocked))
        self.assertTrue(all(item["binding_required"] is True for item in boundary))
        self.assertTrue(all(item["hierarchical_subgoal_required"] is True for item in blocked))

    def test_registry_fails_and_excludes_residual_records_with_missing_metrics(self) -> None:
        self._write_artifacts(missing_dense_metric=True)

        completed = self._run_registry()

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("candidate_metrics_missing", completed.stdout + completed.stderr)
        exclusion = json.loads(
            (self.batch_root / "unified-policy-sample-exclusion-report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(exclusion["excluded_count"], 1)
        self.assertEqual(exclusion["excluded_records"][0]["reason"], "candidate_metrics_missing")

    def test_residual_dry_run_trains_30_pairwise_samples_without_checkpoint(self) -> None:
        self._write_artifacts()
        registry = self._run_registry()
        self.assertEqual(registry.returncode, 0, registry.stdout + registry.stderr)

        completed = self._run_dry_run()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "residual-boundary-preference-training-dry-run-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["residual_preference_dry_run_status"], "passed")
        self.assertEqual(summary["residual_train_sample_count"], 30)
        self.assertEqual(summary["boundary_negative_preference_pair_count"], 12)
        self.assertEqual(summary["blocked_target_negative_pair_count"], 18)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["performance_claimed"])
