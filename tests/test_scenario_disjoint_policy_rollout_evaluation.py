import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ScenarioDisjointPolicyRolloutEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        model_explorer_src = self.repo_root / "model-explorer" / "src"
        if str(model_explorer_src) not in sys.path:
            sys.path.insert(0, str(model_explorer_src))
        self.temp_dir = Path(tempfile.mkdtemp(prefix="scenario-rollout-"))
        self.source_root = self.temp_dir / "source"
        self.candidate_root = self.temp_dir / "candidate"
        self.holdout_root = self.temp_dir / "holdout"
        for path in (self.source_root, self.candidate_root, self.holdout_root):
            path.mkdir(parents=True)
        self.script = self.repo_root / "scripts" / "run_scenario_disjoint_policy_rollout_evaluation.sh"
        self.readiness_script = self.repo_root / "scripts" / "run_policy_training_readiness_review.sh"
        self.config = self.repo_root / "configs" / "scenario_disjoint_policy_rollout_evaluation_v1.json"
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

    def _run_rollout(self) -> subprocess.CompletedProcess[str]:
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

    def _run_readiness(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                str(self.readiness_script),
                "--batch-root",
                str(self.source_root),
                "--config",
                str(self.readiness_config),
                "--controlled-hybrid-policy-training-candidate-summary",
                str(self.candidate_root / "controlled-hybrid-policy-training-candidate-summary.json"),
                "--fresh-holdout-policy-candidate-evaluation-summary",
                str(self.holdout_root / "fresh-holdout-policy-candidate-evaluation-summary.json"),
                "--scenario-disjoint-policy-rollout-evaluation-summary",
                str(self.holdout_root / "scenario-disjoint-policy-rollout-evaluation-summary.json"),
            ],
            cwd=self.repo_root,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _write_artifacts(
        self,
        *,
        missing_context_id: bool = False,
        all_unreachable: bool = False,
        source_contract_safe: bool = True,
    ) -> None:
        self._write_readiness_sources()
        self._write_candidate_artifacts()
        self._write_holdout_artifacts(
            missing_context_id=missing_context_id,
            all_unreachable=all_unreachable,
            source_contract_safe=source_contract_safe,
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
            (self.source_root / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (self.source_root / "batch-evaluation-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "path-feedback-batch-evaluation-summary/v1",
                    "status": "passed",
                    "run_count": 1,
                    "passed_count": 1,
                    "failed_count": 0,
                    "reason_codes": [],
                    "open_grid_fallback_used_count": 0,
                    "safety_regression_count": 0,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
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
                "experimental": True,
                "model_state_dict": network.state_dict(),
                "training": {"hidden_size": 16},
            },
            self.candidate_root / "experimental-hybrid-policy-candidate.pt",
        )
        candidate_summary = {
            "schema_version": "controlled-hybrid-policy-training-candidate-summary/v1",
            "status": "passed",
            "candidate_training_status": "passed",
            "reason_codes": [],
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
            "formal_training_ready_claimed": False,
            "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
        }
        (self.candidate_root / "controlled-hybrid-policy-training-candidate-summary.json").write_text(
            json.dumps(candidate_summary, indent=2),
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

    def _write_holdout_artifacts(
        self,
        *,
        missing_context_id: bool,
        all_unreachable: bool,
        source_contract_safe: bool,
    ) -> None:
        run_root = self.holdout_root / "holdout-run"
        run_root.mkdir(parents=True, exist_ok=True)
        candidates = [
            self._candidate(
                0,
                "ctx-source",
                source_selected=True,
                missing_context_id=missing_context_id,
                reachable=not all_unreachable,
                contract_safe=source_contract_safe,
            ),
            self._candidate(
                1,
                "ctx-alt",
                source_selected=False,
                missing_context_id=missing_context_id,
                reachable=not all_unreachable,
                contract_safe=source_contract_safe,
            ),
        ]
        path_feedback = {
            "schema_version": "path-feedback-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "scenario_set": "holdout",
            "diagnostic_profile": "execution",
            "top_k": 2,
            "open_grid_fallback_used": False,
            "safety_regression_count": 0,
            "candidate_contract_alignment_gap_count": 0,
            "planner_extra_args": ["--planning-backend", "astar"],
            "scenarios": [
                {
                    "scenario_id": "npz_holdout_unit",
                    "scenario_group": "holdout_unit",
                    "scenario_seed": 8601,
                    "scenario_variant_id": "npz_holdout_unit-seed-8601",
                    "open_grid_fallback_used": False,
                    "tracking_safety_violation_count": 0,
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
            json.dumps(path_feedback, indent=2),
            encoding="utf-8",
        )
        (self.holdout_root / "batch-evaluation-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "path-feedback-batch-evaluation-summary/v1",
                    "status": "passed",
                    "run_count": 1,
                    "passed_count": 1,
                    "failed_count": 0,
                    "reason_codes": [],
                    "open_grid_fallback_used_count": 0,
                    "safety_regression_count": 0,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.holdout_root / "fresh-holdout-policy-candidate-evaluation-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "fresh-holdout-policy-candidate-evaluation-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "fresh_disjoint_context_count": 2,
                    "raw_holdout_context_count": 2,
                    "require_context_id": True,
                    "require_scenario_disjoint": True,
                    "scenario_overlap_count": 0,
                    "identity_overlap_count": 0,
                    "context_id_missing_count": 0,
                    "legacy_identity_fallback_count": 0,
                    "scenario_disjoint": True,
                    "fallback_or_open_grid_count": 0,
                    "safety_regression_count": 0,
                    "contract_violation_count": 0,
                    "path_cost_regression_count": 0,
                    "risk_regression_count": 0,
                    "source_selection_regression_count": 0,
                    "experimental_checkpoint": True,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "candidate_git_current_matches_sources": True,
                    "checkpoint_metadata_git_current_matches_sources": True,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _candidate(
        self,
        action_index: int,
        context_id: str,
        *,
        source_selected: bool,
        missing_context_id: bool,
        reachable: bool,
        contract_safe: bool,
    ) -> dict:
        candidate = {
            "action_index": action_index,
            "source_action_index": action_index,
            "cell": [10 + action_index, 10],
            "candidate_role": "policy_target",
            "policy_target_cell": [10 + action_index, 10],
            "execution_goal_cell": [10 + action_index, 10],
            "reachable": reachable,
            "replan_required": False,
            "open_grid_fallback_used": False,
            "path_cost": 10.0,
            "risk": 0.1,
            "utility": 0.8,
            "platform_goal_feasibility": {"contract_reachable": contract_safe},
            "candidate_generation": {
                "source_selection_status": "source_selected" if source_selected else "not_source_selected",
                "source_selection_path_cost_margin_vs_best_alternative": 0.0,
                "source_selection_risk_margin_vs_best_alternative": 0.0,
                "source_selection_quality_regression": False,
            },
        }
        if not missing_context_id:
            candidate.update(
                {
                    "context_id": context_id,
                    "context_id_schema_version": "policy-context-id/v1",
                    "context_id_source": "stable_semantic_fields",
                    "legacy_identity_fallback_used": False,
                }
            )
        return candidate

    def test_rollout_evaluation_writes_shadow_decisions_and_summary(self) -> None:
        self._write_artifacts()

        completed = self._run_rollout()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.holdout_root / "scenario-disjoint-policy-rollout-evaluation-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["scenario_disjoint_context_count"], 2)
        self.assertEqual(summary["policy_decision_count"], 1)
        self.assertEqual(summary["invalid_action_mask_count"], 0)
        self.assertEqual(summary["regression_count"], 0)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])
        decisions = [
            json.loads(line)
            for line in (self.holdout_root / "scenario-disjoint-policy-rollout-decisions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(decisions), 1)
        self.assertIn(decisions[0]["decision_class"], {"aligned", "acceptable_alternative"})
        self.assertIn("context_id", decisions[0])
        self.assertEqual(decisions[0]["regression_reason_codes"], [])

    def test_rollout_evaluation_rejects_missing_context_id(self) -> None:
        self._write_artifacts(missing_context_id=True)

        completed = self._run_rollout()

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.holdout_root / "scenario-disjoint-policy-rollout-evaluation-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "failed")
        self.assertIn("context_id_missing", summary["reason_codes"])

    def test_rollout_evaluation_rejects_invalid_action_mask(self) -> None:
        self._write_artifacts(all_unreachable=True)

        completed = self._run_rollout()

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.holdout_root / "scenario-disjoint-policy-rollout-evaluation-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["invalid_action_mask_count"], 1)
        self.assertIn("invalid_action_mask", summary["reason_codes"])

    def test_rollout_evaluation_does_not_treat_source_contract_as_policy_regression(self) -> None:
        self._write_artifacts(source_contract_safe=False)

        completed = self._run_rollout()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.holdout_root / "scenario-disjoint-policy-rollout-evaluation-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["contract_violation_count"], 0)
        self.assertEqual(summary["regression_count"], 0)
        self.assertEqual(summary["reason_codes"], [])

    def test_readiness_advances_after_scenario_disjoint_rollout_passes(self) -> None:
        self._write_artifacts()
        self.assertEqual(self._run_rollout().returncode, 0)

        completed = self._run_readiness()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.source_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            summary["training_readiness_status"],
            "scenario_disjoint_policy_rollout_evaluated",
        )
        self.assertEqual(summary["training_blockers"], [])


if __name__ == "__main__":
    unittest.main()
