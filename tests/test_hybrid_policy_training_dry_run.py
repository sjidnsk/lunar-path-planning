import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class HybridPolicyTrainingDryRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        model_explorer_src = self.repo_root / "model-explorer" / "src"
        if str(model_explorer_src) not in sys.path:
            sys.path.insert(0, str(model_explorer_src))
        self.temp_dir = Path(tempfile.mkdtemp(prefix="hybrid-policy-training-"))
        self.batch_root = self.temp_dir / "batch"
        self.batch_root.mkdir(parents=True)
        self.script = self.repo_root / "scripts" / "run_hybrid_policy_training_dry_run.sh"
        self.config = self.temp_dir / "hybrid-config.json"
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHON"] = str(Path("/home/kai/anaconda3/envs/lunar-explorer/bin/python"))
        return env

    def _run_hybrid(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                str(self.script),
                "--batch-root",
                str(self.batch_root),
                "--config",
                str(self.config),
            ],
            cwd=self.repo_root,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _write_config(self) -> None:
        self.config.write_text(
            json.dumps(
                {
                    "schema_version": "hybrid-policy-training-dry-run-config/v1",
                    "input_files": {
                        "rollout_episodes": "planner-validated-rollout-episodes.jsonl",
                        "materialization_summary": "planner-validated-training-input-materialization-summary.json",
                        "counterfactual_preference_samples": "counterfactual-preference-training-samples.jsonl",
                        "counterfactual_preference_summary": "counterfactual-preference-training-summary.json",
                        "unified_policy_sample_registry": "unified-policy-sample-registry.jsonl",
                        "unified_policy_sample_registry_summary": "unified-policy-sample-registry-summary.json",
                    },
                    "output_files": {
                        "summary": "hybrid-policy-training-dry-run-summary.json",
                    },
                    "validation": {
                        "expected_action_label_positive_count": 24,
                        "expected_existing_preference_pair_count": 24,
                        "expected_residual_preference_pair_count": 30,
                        "expected_pairwise_preference_signal_count": 54,
                        "expected_hybrid_train_signal_count": 78,
                        "max_invalid_action_mask_count": 0,
                        "max_empty_action_mask_count": 0,
                    },
                    "training": {
                        "seed": 0,
                        "hidden_size": 8,
                        "learning_rate": 0.001,
                        "epochs": 1,
                        "margin": 0.1,
                        "action_label_loss_weight": 1.0,
                        "preference_loss_weight": 0.5,
                        "residual_negative_loss_weight": 0.25,
                        "checkpoint_path": None,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_artifacts(
        self,
        *,
        materialization_status: str = "passed",
        registry_status: str = "passed",
        invalid_action_mask_count: int = 0,
        empty_action_mask_count: int = 0,
    ) -> None:
        self._write_rollout_episodes()
        materialization_summary = {
            "schema_version": "planner-validated-training-input-materialization-summary/v1",
            "status": materialization_status,
            "reason_codes": [] if materialization_status == "passed" else ["synthetic_failure"],
            "input_positive_count": 24,
            "default_contract_positive_count": 18,
            "planner_validated_exception_positive_count": 6,
            "excluded_nontrainable_count": 54,
            "invalid_action_mask_count": invalid_action_mask_count,
            "empty_action_mask_count": empty_action_mask_count,
            "publishes_checkpoint": False,
            "performance_claimed": False,
        }
        counterfactual_summary = {
            "schema_version": "counterfactual-preference-training-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "preference_pair_count": 24,
            "selected_over_alternative_negative_count": 12,
            "tradeoff_preference_pair_count": 12,
            "rejected_binding_or_distance_required_count": 12,
            "hard_positive_added_count": 0,
        }
        registry_summary = {
            "schema_version": "unified-policy-sample-registry-summary/v1",
            "status": registry_status,
            "reason_codes": [] if registry_status == "passed" else ["synthetic_failure"],
            "action_label_positive_count": 24,
            "existing_preference_pair_count": 24,
            "boundary_negative_preference_pair_count": 12,
            "blocked_target_negative_pair_count": 18,
            "residual_trainable_signal_count": 30,
            "pairwise_preference_signal_count": 54,
            "unified_context_coverage_count": 78,
            "hard_positive_added_count": 0,
            "publishes_checkpoint": False,
            "performance_claimed": False,
        }
        (self.batch_root / "planner-validated-training-input-materialization-summary.json").write_text(
            json.dumps(materialization_summary, indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "counterfactual-preference-training-summary.json").write_text(
            json.dumps(counterfactual_summary, indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "unified-policy-sample-registry-summary.json").write_text(
            json.dumps(registry_summary, indent=2),
            encoding="utf-8",
        )
        preference_samples = [
            self._pairwise_record(
                index=index,
                sample_type="counterfactual_preference_pair",
                preference_decision=(
                    "selected_over_alternative_negative"
                    if index < 12
                    else "tradeoff_preference_pair"
                ),
            )
            for index in range(24)
        ]
        registry_records = [
            self._action_label_registry_record(index) for index in range(24)
        ]
        registry_records.extend(preference_samples)
        registry_records.extend(
            self._pairwise_record(
                index=index,
                sample_type="boundary_negative_preference_pair",
                preference_decision="boundary_negative_preference_pair",
                sample_weight=0.25,
            )
            for index in range(12)
        )
        registry_records.extend(
            self._pairwise_record(
                index=index,
                sample_type="blocked_target_negative_pair",
                preference_decision="blocked_target_negative_pair",
                sample_weight=1.0,
            )
            for index in range(18)
        )
        (self.batch_root / "counterfactual-preference-training-samples.jsonl").write_text(
            "".join(json.dumps(sample) + "\n" for sample in preference_samples),
            encoding="utf-8",
        )
        (self.batch_root / "unified-policy-sample-registry.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in registry_records),
            encoding="utf-8",
        )

    def _write_rollout_episodes(self) -> None:
        from model_explorer.policy.features import (
            CANDIDATE_FEATURE_NAMES,
            GLOBAL_FEATURE_NAMES,
            MISSING_INDICATOR_NAMES,
            PolicyObservation,
        )
        from model_explorer.policy.rollout import (
            EpisodeMetrics,
            RolloutEpisode,
            RolloutInfo,
            RolloutTransition,
        )
        from model_explorer.policy.rollout_io import write_rollout_episodes_jsonl

        episodes = []
        for index in range(24):
            action_index = index % 2
            observation = PolicyObservation(
                candidate_feature_names=CANDIDATE_FEATURE_NAMES,
                candidate_features=(
                    tuple([0.1 + index * 0.001] * len(CANDIDATE_FEATURE_NAMES)),
                    tuple([0.2 + index * 0.001] * len(CANDIDATE_FEATURE_NAMES)),
                ),
                global_feature_names=GLOBAL_FEATURE_NAMES,
                global_features=tuple([0.0] * len(GLOBAL_FEATURE_NAMES)),
                action_mask=(True, True),
                candidate_cells=((index, 0), (index, 1)),
                candidate_missing_indicator_names=MISSING_INDICATOR_NAMES,
                candidate_missing_indicators=(
                    tuple([0.0] * len(MISSING_INDICATOR_NAMES)),
                    tuple([0.0] * len(MISSING_INDICATOR_NAMES)),
                ),
            )
            transition = RolloutTransition(
                observation=observation,
                action_index=action_index,
                log_prob=0.0,
                value=0.0,
                reward=1.0,
                next_observation=None,
                done=True,
                info=RolloutInfo(
                    selected_cell=(index, action_index),
                    coverage_rate_delta=1.0,
                    extra={"sample_type": "action_label_positive"},
                ),
            )
            episodes.append(
                RolloutEpisode(
                    transitions=(transition,),
                    metrics=EpisodeMetrics(cumulative_coverage_rate_delta=1.0),
                )
            )
        write_rollout_episodes_jsonl(
            self.batch_root / "planner-validated-rollout-episodes.jsonl",
            tuple(episodes),
        )

    def _action_label_registry_record(self, index: int) -> dict:
        return {
            "schema_version": "unified-policy-sample-registry-summary/v1",
            "sample_index": index,
            "sample_type": "action_label_positive",
            "training_signal_type": "rollout_action_label",
            "run_id": "run-a",
            "scenario_id": f"positive-{index}",
            "hard_positive": True,
        }

    def _pairwise_record(
        self,
        *,
        index: int,
        sample_type: str,
        preference_decision: str,
        sample_weight: float = 1.0,
    ) -> dict:
        selected_key = "selected" if sample_type == "counterfactual_preference_pair" else "preferred"
        record = {
            "schema_version": "unified-policy-sample-registry-summary/v1",
            "sample_index": index,
            "sample_type": sample_type,
            "training_signal_type": "pairwise_preference",
            "run_id": "run-a",
            "scenario_id": f"{sample_type}-{index}",
            "preference_decision": preference_decision,
            "sample_weight": sample_weight,
            "alternative": {
                "candidate_features": [0.2 + index * 0.001] * 15,
            },
            "global_features": [0.0] * 8,
            "candidate_missing_indicators": [[0.0] * 8, [0.0] * 8],
            "hard_positive": False,
        }
        record[selected_key] = {"candidate_features": [0.1 + index * 0.001] * 15}
        return record

    def test_hybrid_dry_run_consumes_24_action_labels_and_54_pairwise_signals(self) -> None:
        self._write_artifacts()

        completed = self._run_hybrid()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "hybrid-policy-training-dry-run-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["dry_run_status"], "passed")
        self.assertEqual(summary["action_label_positive_count"], 24)
        self.assertEqual(summary["existing_preference_pair_count"], 24)
        self.assertEqual(summary["residual_preference_pair_count"], 30)
        self.assertEqual(summary["pairwise_preference_signal_count"], 54)
        self.assertEqual(summary["hybrid_train_signal_count"], 78)
        self.assertEqual(summary["hard_positive_added_count"], 0)
        self.assertEqual(summary["invalid_action_mask_count"], 0)
        self.assertEqual(summary["empty_action_mask_count"], 0)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["performance_claimed"])
        self.assertIn("action_label_loss", summary["training_result"])
        self.assertIn("pairwise_preference_loss", summary["training_result"])

    def test_hybrid_dry_run_rejects_invalid_or_empty_action_mask_summary(self) -> None:
        self._write_artifacts(invalid_action_mask_count=1)

        completed = self._run_hybrid()

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("invalid_action_mask", completed.stdout + completed.stderr)

    def test_hybrid_dry_run_rejects_failed_registry_summary(self) -> None:
        self._write_artifacts(registry_status="failed")

        completed = self._run_hybrid()

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("unified_policy_sample_registry_summary_failed", completed.stdout + completed.stderr)
