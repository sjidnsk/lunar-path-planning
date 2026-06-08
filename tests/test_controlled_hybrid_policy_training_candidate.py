import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ControlledHybridPolicyTrainingCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        model_explorer_src = self.repo_root / "model-explorer" / "src"
        if str(model_explorer_src) not in sys.path:
            sys.path.insert(0, str(model_explorer_src))
        self.temp_dir = Path(tempfile.mkdtemp(prefix="controlled-hybrid-candidate-"))
        self.source_root = self.temp_dir / "source"
        self.output_root = self.temp_dir / "candidate"
        self.source_root.mkdir(parents=True)
        self.candidate_script = (
            self.repo_root / "scripts" / "run_controlled_hybrid_policy_training_candidate.sh"
        )
        self.holdout_script = (
            self.repo_root / "scripts" / "run_controlled_hybrid_policy_holdout_evaluation.sh"
        )
        self.readiness_script = self.repo_root / "scripts" / "run_policy_training_readiness_review.sh"
        self.candidate_config = (
            self.repo_root / "configs" / "controlled_hybrid_policy_training_candidate_v1.json"
        )
        self.holdout_config = (
            self.repo_root / "configs" / "controlled_hybrid_policy_holdout_evaluation_v1.json"
        )
        self.readiness_config = (
            self.repo_root / "configs" / "policy_training_readiness_review_v1.json"
        )
        self.git_snapshot = self._current_git_snapshot()

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

    def _run_candidate(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                str(self.candidate_script),
                "--source-root",
                str(self.source_root),
                "--output-root",
                str(self.output_root),
                "--config",
                str(self.candidate_config),
            ],
            cwd=self.repo_root,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _run_holdout(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                str(self.holdout_script),
                "--source-root",
                str(self.source_root),
                "--candidate-root",
                str(self.output_root),
                "--config",
                str(self.holdout_config),
            ],
            cwd=self.repo_root,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _run_readiness(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                str(self.readiness_script),
                "--batch-root",
                str(self.source_root),
                "--config",
                str(self.readiness_config),
                "--hybrid-policy-training-dry-run-summary",
                str(self.source_root / "hybrid-policy-training-dry-run-summary.json"),
                "--controlled-hybrid-policy-training-candidate-summary",
                str(self.output_root / "controlled-hybrid-policy-training-candidate-summary.json"),
                "--controlled-hybrid-policy-holdout-evaluation-summary",
                str(self.output_root / "controlled-hybrid-policy-holdout-evaluation-summary.json"),
            ],
            cwd=self.repo_root,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _write_source_artifacts(self) -> None:
        self._write_readiness_sources()
        self._write_rollout_episodes()
        self._write_json("batch-evaluation-summary.json", {
            "schema_version": "path-feedback-batch-evaluation-summary/v1",
            "status": "passed",
            "run_count": 8,
            "passed_count": 8,
            "failed_count": 0,
            "open_grid_fallback_used_count": 0,
            "safety_regression_count": 0,
            "reason_codes": [],
            "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
        })
        self._write_json("anchor-projection-candidate-generation-summary.json", {
            "schema_version": "anchor-projection-candidate-generation-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "current_git_provenance_mismatch_count": 0,
            "git_provenance_mismatch_count": 0,
            "fallback_or_open_grid_count": 0,
            "safety_regression_count": 0,
            "candidate_contract_alignment_gap_count": 0,
            "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
        })
        self._write_json("planner-validated-trainable-target-mining-summary.json", {
            "schema_version": "planner-validated-trainable-target-mining-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "planner_validated_trainable_target_count": 24,
            "nontrainable_blocked_target_count": 54,
            "fallback_or_open_grid_count": 0,
            "safety_regression_count": 0,
            "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
        })
        self._write_json("planner-validated-training-input-materialization-summary.json", {
            "schema_version": "planner-validated-training-input-materialization-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "input_positive_count": 24,
            "default_contract_positive_count": 18,
            "planner_validated_exception_positive_count": 6,
            "excluded_nontrainable_count": 54,
            "invalid_action_mask_count": 0,
            "empty_action_mask_count": 0,
            "publishes_checkpoint": False,
            "performance_claimed": False,
        })
        self._write_json("unified-policy-sample-registry-summary.json", {
            "schema_version": "unified-policy-sample-registry-summary/v1",
            "status": "passed",
            "reason_codes": [],
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
        })
        self._write_json("hybrid-policy-training-dry-run-summary.json", {
            "schema_version": "hybrid-policy-training-dry-run-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "dry_run_status": "passed",
            "action_label_positive_count": 24,
            "existing_preference_pair_count": 24,
            "residual_preference_pair_count": 30,
            "pairwise_preference_signal_count": 54,
            "hybrid_train_signal_count": 78,
            "hard_positive_added_count": 0,
            "invalid_action_mask_count": 0,
            "empty_action_mask_count": 0,
            "publishes_checkpoint": False,
            "performance_claimed": False,
            "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
            "runs_large_scale_training": False,
            "dry_run_only": True,
        })
        registry_records = [self._action_label_registry_record(index) for index in range(24)]
        registry_records.extend(
            self._pairwise_record(index, "counterfactual_preference_pair") for index in range(24)
        )
        registry_records.extend(
            self._pairwise_record(index, "boundary_negative_preference_pair", sample_weight=0.25)
            for index in range(12)
        )
        registry_records.extend(
            self._pairwise_record(index, "blocked_target_negative_pair")
            for index in range(18)
        )
        (self.source_root / "unified-policy-sample-registry.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in registry_records),
            encoding="utf-8",
        )

    def _write_readiness_sources(self) -> None:
        common = {
            "generated_at": "2026-06-08T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "source_selected_candidate_changed_rate": 0.0,
            "safety_regression_count": 0,
            "open_grid_fallback_used_count": 0,
            "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
        }
        self._write_json("calibrated-policy-application-smoke-summary.json", {
            **common,
            "schema_version": "calibrated-policy-application-smoke-summary/v1",
            "calibrated_selected_candidate_changed_rate": 0.5,
            "applied_calibrated_candidate_count": 2,
            "rejected_goal_blocked_count": 0,
            "platform_goal_contract_mismatch_count": 0,
            "recommended_next_action": "ready_for_policy_training_readiness_review",
            "does_not_modify_default_astar": True,
            "does_not_modify_ppo": True,
            "does_not_modify_network": True,
            "does_not_modify_action_space": True,
            "does_not_modify_model_explorer_contract": True,
            "does_not_modify_path_planner_route_contract": True,
            "does_not_modify_path_planner_sidecar_contract": True,
            "no_ackermann_feasible_trajectory_claim": True,
        })
        self._write_json("channel-aware-training-readiness-summary.json", {
            **common,
            "schema_version": "channel-aware-training-readiness-summary/v1",
            "readiness_status": "ready_for_calibrated_policy_application_smoke",
            "calibrated_readiness_status": "ready_for_calibrated_policy_application_smoke",
            "calibration_selected_candidate_changed_rate": 0.5,
            "calibration_safety_regression_count": 0,
        })
        self._write_json("channel-aware-contrast-coverage-summary.json", {
            **common,
            "schema_version": "channel-aware-contrast-coverage-summary/v1",
            "calibrated_selected_candidate_changed_rate": 0.5,
            "blocked_candidate_rate": 0.0,
            "recommended_next_action": "ready_for_calibrated_policy_application_smoke",
        })
        self._write_json("channel-aware-selection-contrast-calibration-summary.json", {
            **common,
            "schema_version": "channel-aware-selection-contrast-calibration-summary/v1",
            "selected_candidate_changed_count": 2,
            "selected_candidate_changed_rate": 0.5,
            "goal_blocked_count": 0,
            "platform_goal_contract_mismatch_count": 0,
        })

    def _write_json(self, relative_path: str, payload: dict) -> None:
        (self.source_root / relative_path).write_text(
            json.dumps(payload, indent=2),
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
            episodes.append(
                RolloutEpisode(
                    transitions=(
                        RolloutTransition(
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
                        ),
                    ),
                    metrics=EpisodeMetrics(cumulative_coverage_rate_delta=1.0),
                )
            )
        write_rollout_episodes_jsonl(
            self.source_root / "planner-validated-rollout-episodes.jsonl",
            tuple(episodes),
        )

    def _action_label_registry_record(self, index: int) -> dict:
        return {
            "schema_version": "unified-policy-sample-registry-summary/v1",
            "sample_index": index,
            "sample_type": "action_label_positive",
            "training_signal_type": "rollout_action_label",
            "scenario_id": f"positive-{index}",
            "hard_positive": True,
        }

    def _pairwise_record(
        self,
        index: int,
        sample_type: str,
        *,
        sample_weight: float = 1.0,
    ) -> dict:
        selected_key = "selected" if sample_type == "counterfactual_preference_pair" else "preferred"
        record = {
            "schema_version": "unified-policy-sample-registry-summary/v1",
            "sample_index": index,
            "sample_type": sample_type,
            "training_signal_type": "pairwise_preference",
            "scenario_id": f"{sample_type}-{index}",
            "sample_weight": sample_weight,
            "alternative": {"candidate_features": [0.2 + index * 0.001] * 15},
            "global_features": [0.0] * 8,
            "candidate_missing_indicators": [[0.0] * 8, [0.0] * 8],
            "hard_positive": False,
        }
        record[selected_key] = {"candidate_features": [0.1 + index * 0.001] * 15}
        return record

    def test_training_candidate_writes_experimental_checkpoint_metadata_without_publishing(self) -> None:
        self._write_source_artifacts()

        completed = self._run_candidate()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.output_root / "controlled-hybrid-policy-training-candidate-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["candidate_training_status"], "passed")
        self.assertEqual(summary["action_label_positive_count"], 24)
        self.assertEqual(summary["pairwise_preference_signal_count"], 54)
        self.assertEqual(summary["hybrid_train_signal_count"], 78)
        self.assertEqual(summary["hard_positive_added_count"], 0)
        self.assertTrue(summary["experimental_checkpoint"])
        self.assertTrue((self.output_root / "experimental-hybrid-policy-candidate.pt").is_file())
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])
        metadata = json.loads(
            (self.output_root / "experimental-hybrid-policy-candidate-metadata.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(metadata["experimental"])
        self.assertFalse(metadata["publishes_checkpoint"])
        self.assertFalse(metadata["replaces_default_policy"])
        self.assertEqual(
            metadata["git_provenance"]["current"]["parent"]["sha"],
            self.git_snapshot["parent"]["sha"],
        )
        self.assertEqual(
            metadata["git_provenance"]["current"]["submodules"]["model-explorer"]["sha"],
            self.git_snapshot["submodules"]["model-explorer"]["sha"],
        )
        self.assertTrue(metadata["git_provenance"]["current_matches_sources"])

    def test_holdout_evaluation_reports_zero_safety_contract_and_action_mask_regressions(self) -> None:
        self._write_source_artifacts()
        self.assertEqual(self._run_candidate().returncode, 0)

        completed = self._run_holdout()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.output_root / "controlled-hybrid-policy-holdout-evaluation-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["action_mask_invalid_count"], 0)
        self.assertEqual(summary["empty_action_mask_count"], 0)
        self.assertEqual(summary["fallback_or_open_grid_count"], 0)
        self.assertEqual(summary["safety_regression_count"], 0)
        self.assertEqual(summary["contract_violation_count"], 0)
        self.assertIn("preference_margin_improved_count", summary)
        self.assertFalse(summary["performance_claimed"])

    def test_readiness_records_controlled_candidate_evaluated_only_after_holdout_passes(self) -> None:
        self._write_source_artifacts()
        self.assertEqual(self._run_candidate().returncode, 0)
        self.assertEqual(self._run_holdout().returncode, 0)

        completed = self._run_readiness()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.source_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            summary["training_readiness_status"],
            "controlled_hybrid_training_candidate_evaluated",
        )
        self.assertEqual(
            summary["recommended_next_action"],
            "controlled_hybrid_training_candidate_evaluated",
        )
        self.assertFalse(summary["controlled_hybrid_training_candidate_readiness"]["formal_training_ready_claimed"])
        self.assertFalse(summary["controlled_hybrid_training_candidate_readiness"]["performance_claimed"])

    def test_readiness_blocks_controlled_candidate_when_holdout_regresses_path_cost(self) -> None:
        self._write_source_artifacts()
        self.assertEqual(self._run_candidate().returncode, 0)
        self.assertEqual(self._run_holdout().returncode, 0)
        holdout_path = self.output_root / "controlled-hybrid-policy-holdout-evaluation-summary.json"
        holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
        holdout["path_cost_regression_count"] = 1
        holdout["next_required_change"] = "training_objective_or_sample_weight_refinement_required"
        holdout_path.write_text(json.dumps(holdout, indent=2), encoding="utf-8")

        completed = self._run_readiness()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.source_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["training_readiness_status"], "needs_training_contract_refinement")
        self.assertEqual(
            summary["next_required_change"],
            "training_objective_or_sample_weight_refinement_required",
        )
        self.assertIn(
            "controlled_hybrid_holdout_path_cost_regression",
            summary["training_blockers"],
        )
