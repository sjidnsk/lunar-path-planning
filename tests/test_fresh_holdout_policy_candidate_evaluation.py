import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class FreshHoldoutPolicyCandidateEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        model_explorer_src = self.repo_root / "model-explorer" / "src"
        if str(model_explorer_src) not in sys.path:
            sys.path.insert(0, str(model_explorer_src))
        self.temp_dir = Path(tempfile.mkdtemp(prefix="fresh-holdout-"))
        self.source_root = self.temp_dir / "source"
        self.candidate_root = self.temp_dir / "candidate"
        self.holdout_root = self.temp_dir / "holdout"
        for path in (self.source_root, self.candidate_root, self.holdout_root):
            path.mkdir(parents=True)
        self.script = self.repo_root / "scripts" / "run_fresh_holdout_policy_candidate_evaluation.sh"
        self.readiness_script = self.repo_root / "scripts" / "run_policy_training_readiness_review.sh"
        self.config = self.repo_root / "configs" / "fresh_holdout_policy_candidate_evaluation_v1.json"
        self.readiness_config = self.repo_root / "configs" / "policy_training_readiness_review_v1.json"
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

    def _run_fresh(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                str(self.script),
                "--source-root",
                str(self.source_root),
                "--candidate-root",
                str(self.candidate_root),
                "--batch-root",
                str(self.holdout_root),
                "--config",
                str(self.config),
            ],
            cwd=self.repo_root,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _write_common_artifacts(self) -> None:
        self._write_batch_root(self.source_root, run_id="source-run", candidate_cells=[(10, 10)])
        self._write_batch_root(self.holdout_root, run_id="fresh-run", candidate_cells=[(10, 10), (11, 10)])
        self._write_candidate_artifacts()
        self._write_readiness_sources()

    def _write_batch_root(
        self,
        root: Path,
        *,
        run_id: str,
        candidate_cells: list[tuple[int, int]],
    ) -> None:
        run_root = root / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        candidates = [
            {
                "action_index": index,
                "source_action_index": index,
                "cell": list(cell),
                "candidate_role": "policy_target",
                "policy_target_cell": list(cell),
                "execution_goal_cell": list(cell),
                "utility": 0.8 - index * 0.1,
                "reachable": True,
                "path_cost": 10.0 + index,
                "risk": 0.1 + index * 0.01,
                "failure_reason": None,
                "replan_required": False,
                "open_grid_fallback_used": False,
                "platform_goal_feasibility": {"contract_reachable": True},
            }
            for index, cell in enumerate(candidate_cells)
        ]
        summary = {
            "schema_version": "path-feedback-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "scenario_set": "stress",
            "diagnostic_profile": "execution",
            "top_k": len(candidate_cells),
            "open_grid_fallback_used": False,
            "open_grid_fallback_used_gate": "passed",
            "safety_regression_count": 0,
            "candidate_contract_alignment_gap_count": 0,
            "scenarios": [
                {
                    "scenario_id": "shared-scenario",
                    "scenario_group": "stress",
                    "open_grid_fallback_used": False,
                    "tracking_safety_violation_count": 0,
                    "trajectory_optimization_fallback_count": 0,
                    "path_feedback": {
                        "candidate_count": len(candidates),
                        "reachable_count": len(candidates),
                        "failure_count": 0,
                        "replan_count": 0,
                        "best_by_path_cost": candidates[0],
                        "candidates": candidates,
                    },
                }
            ],
            "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
        }
        (run_root / "path-feedback-summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        batch_summary = {
            "schema_version": "path-feedback-batch-evaluation-summary/v1",
            "status": "passed",
            "run_count": 1,
            "passed_count": 1,
            "failed_count": 0,
            "failed_run_ids": [],
            "open_grid_fallback_used_count": 0,
            "safety_regression_count": 0,
            "reason_codes": [],
            "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
        }
        (root / "batch-evaluation-summary.json").write_text(
            json.dumps(batch_summary, indent=2),
            encoding="utf-8",
        )
        run_index = {
            "schema_version": "path-feedback-batch-run-index/v1",
            "runs": [
                {
                    "run_id": run_id,
                    "status": "passed",
                    "summary_path": str(run_root / "path-feedback-summary.json"),
                    "open_grid_fallback_used": False,
                    "reason_codes": [],
                }
            ],
        }
        (root / "batch-run-index.json").write_text(
            json.dumps(run_index, indent=2),
            encoding="utf-8",
        )

    def _write_candidate_artifacts(self) -> None:
        from model_explorer.policy.architectures import build_policy_network
        from model_explorer.policy.features import (
            CANDIDATE_FEATURE_NAMES,
            GLOBAL_FEATURE_NAMES,
            MISSING_INDICATOR_NAMES,
            PolicyObservation,
        )
        import torch

        observation = PolicyObservation(
            candidate_feature_names=CANDIDATE_FEATURE_NAMES,
            candidate_features=(
                tuple([0.1] * len(CANDIDATE_FEATURE_NAMES)),
                tuple([0.2] * len(CANDIDATE_FEATURE_NAMES)),
            ),
            global_feature_names=GLOBAL_FEATURE_NAMES,
            global_features=tuple([0.0] * len(GLOBAL_FEATURE_NAMES)),
            action_mask=(True, True),
            candidate_cells=((10, 10), (11, 10)),
            candidate_missing_indicator_names=MISSING_INDICATOR_NAMES,
            candidate_missing_indicators=(
                tuple([0.0] * len(MISSING_INDICATOR_NAMES)),
                tuple([0.0] * len(MISSING_INDICATOR_NAMES)),
            ),
        )
        network = build_policy_network(None, observation=observation, hidden_size=16)
        torch.save(
            {
                "schema_version": "controlled-hybrid-policy-candidate-checkpoint/v1",
                "model_state_dict": network.state_dict(),
                "training": {"hidden_size": 16},
            },
            self.candidate_root / "experimental-hybrid-policy-candidate.pt",
        )
        summary = {
            "schema_version": "controlled-hybrid-policy-training-candidate-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "candidate_training_status": "passed",
            "action_label_positive_count": 24,
            "pairwise_preference_signal_count": 54,
            "hybrid_train_signal_count": 78,
            "hard_positive_added_count": 0,
            "invalid_action_mask_count": 0,
            "empty_action_mask_count": 0,
            "experimental_checkpoint": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
        }
        (self.candidate_root / "controlled-hybrid-policy-training-candidate-summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        metadata = {
            "schema_version": "controlled-hybrid-policy-candidate-checkpoint-metadata/v1",
            "experimental": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
        }
        (self.candidate_root / "experimental-hybrid-policy-candidate-metadata.json").write_text(
            json.dumps(metadata, indent=2),
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
        payloads = {
            "calibrated-policy-application-smoke-summary.json": {
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
            },
            "channel-aware-training-readiness-summary.json": {
                **common,
                "schema_version": "channel-aware-training-readiness-summary/v1",
                "readiness_status": "ready_for_calibrated_policy_application_smoke",
                "calibrated_readiness_status": "ready_for_calibrated_policy_application_smoke",
                "calibration_selected_candidate_changed_rate": 0.5,
                "calibration_safety_regression_count": 0,
            },
            "channel-aware-contrast-coverage-summary.json": {
                **common,
                "schema_version": "channel-aware-contrast-coverage-summary/v1",
                "calibrated_selected_candidate_changed_rate": 0.5,
                "blocked_candidate_rate": 0.0,
                "recommended_next_action": "ready_for_calibrated_policy_application_smoke",
            },
            "channel-aware-selection-contrast-calibration-summary.json": {
                **common,
                "schema_version": "channel-aware-selection-contrast-calibration-summary/v1",
                "selected_candidate_changed_count": 2,
                "selected_candidate_changed_rate": 0.5,
                "goal_blocked_count": 0,
                "platform_goal_contract_mismatch_count": 0,
            },
        }
        for filename, payload in payloads.items():
            (self.source_root / filename).write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )

    def test_fresh_holdout_accepts_only_identity_disjoint_candidates(self) -> None:
        self._write_common_artifacts()

        completed = self._run_fresh()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary_path = self.holdout_root / "fresh-holdout-policy-candidate-evaluation-summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertGreater(summary["fresh_disjoint_context_count"], 0)
        self.assertEqual(summary["accepted_identity_overlap_count"], 0)
        self.assertEqual(summary["accepted_identity_key_missing_count"], 0)
        self.assertEqual(summary["scenario_overlap_count"], 1)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])
        self.assertTrue((self.holdout_root / "fresh-holdout-overlap-report.json").is_file())
        self.assertTrue((self.holdout_root / "fresh-holdout-candidate-score-report.json").is_file())

    def test_fresh_holdout_fails_when_no_disjoint_context_exists(self) -> None:
        self._write_common_artifacts()
        self._write_batch_root(self.holdout_root, run_id="fresh-run", candidate_cells=[(10, 10)])

        completed = self._run_fresh()

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.holdout_root / "fresh-holdout-policy-candidate-evaluation-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "failed")
        self.assertIn("fresh_disjoint_context_count_zero", summary["reason_codes"])
        self.assertEqual(
            summary["next_required_change"],
            "fresh_holdout_scenario_or_candidate_generation_required",
        )

    def test_readiness_advances_only_after_fresh_holdout_passes(self) -> None:
        self._write_common_artifacts()
        self.assertEqual(self._run_fresh().returncode, 0)

        completed = subprocess.run(
            [
                "bash",
                str(self.readiness_script),
                "--batch-root",
                str(self.source_root),
                "--config",
                str(self.readiness_config),
                "--fresh-holdout-policy-candidate-evaluation-summary",
                str(self.holdout_root / "fresh-holdout-policy-candidate-evaluation-summary.json"),
            ],
            cwd=self.repo_root,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        readiness = json.loads(
            (self.source_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            readiness["training_readiness_status"],
            "fresh_holdout_policy_candidate_evaluated",
        )
        self.assertEqual(
            readiness["recommended_next_action"],
            "fresh_holdout_policy_candidate_evaluated",
        )
        self.assertFalse(readiness["fresh_holdout_policy_candidate_readiness"]["performance_claimed"])


if __name__ == "__main__":
    unittest.main()
