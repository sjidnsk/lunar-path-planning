import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class RawPolicyDecisionAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="raw-policy-align-"))
        self.source_root = self.temp_dir / "source"
        self.holdout_root = self.temp_dir / "holdout"
        self.candidate_root = self.temp_dir / "candidate"
        for path in (self.source_root, self.holdout_root, self.candidate_root):
            path.mkdir(parents=True)
        self.git_snapshot = self._current_git_snapshot()
        self.mining_script = self.repo_root / "scripts" / "run_raw_policy_regression_mining.sh"
        self.readiness_script = self.repo_root / "scripts" / "run_policy_training_readiness_review.sh"
        self.mining_config = self.repo_root / "configs" / "raw_policy_regression_mining_v1.json"
        self.readiness_config = self.repo_root / "configs" / "policy_training_readiness_review_v1.json"

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

    def _run_mining(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                str(self.mining_script),
                "--source-root",
                str(self.source_root),
                "--holdout-root",
                str(self.holdout_root),
                "--config",
                str(self.mining_config),
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
                "--raw-policy-strict-rollout-evaluation-summary",
                str(self.holdout_root / "raw-policy-strict-rollout-evaluation-summary.json"),
            ],
            cwd=self.repo_root,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _write_mining_artifacts(self) -> None:
        self._write_source_batch_summary()
        run_root = self.holdout_root / "holdout-run"
        run_root.mkdir(parents=True)
        source = self._candidate("ctx-source", action=0, path_cost=1.0, risk=0.1, source_selected=True)
        raw = self._candidate("ctx-raw", action=1, path_cost=3.0, risk=0.3, source_selected=False)
        path_feedback = {
            "schema_version": "path-feedback-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "diagnostic_profile": "execution",
            "planner_extra_args": ["--planning-backend", "astar"],
            "scenarios": [
                {
                    "scenario_id": "raw_regression_case",
                    "scenario_group": "holdout_unit",
                    "scenario_seed": 8601,
                    "scenario_variant_id": "raw-regression-case",
                    "path_feedback": {
                        "best_by_path_cost": source,
                        "candidates": [source, raw],
                    },
                }
            ],
            "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
        }
        (run_root / "path-feedback-summary.json").write_text(
            json.dumps(path_feedback, indent=2),
            encoding="utf-8",
        )
        (self.holdout_root / "scenario-disjoint-policy-rollout-decisions.jsonl").write_text(
            json.dumps(
                {
                    "schema_version": "scenario-disjoint-policy-rollout-decision/v1",
                    "source_path": str(run_root / "path-feedback-summary.json"),
                    "context_id": "ctx-source",
                    "scenario_id": "raw_regression_case",
                    "source_selected_action_index": 0,
                    "raw_policy_selected_action_index": 1,
                    "policy_selected_action_index": 0,
                    "source_selected_context_id": "ctx-source",
                    "raw_policy_selected_context_id": "ctx-raw",
                    "policy_selected_context_id": "ctx-source",
                    "raw_policy_decision_class": "regression",
                    "raw_policy_regression_reason_codes": ["path_cost_regression", "risk_regression"],
                    "raw_policy_logit_margin_vs_source": 0.7,
                    "decision_class": "aligned",
                    "regression_reason_codes": [],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (self.holdout_root / "scenario-disjoint-policy-rollout-evaluation-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "scenario-disjoint-policy-rollout-evaluation-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "raw_policy_regression_count": 1,
                    "regression_count": 0,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_readiness_artifacts(self) -> None:
        self._write_source_batch_summary()
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
        (self.holdout_root / "raw-policy-strict-rollout-evaluation-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "raw-policy-strict-rollout-evaluation-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "scenario_disjoint_context_count": 2,
                    "policy_decision_count": 1,
                    "regression_count": 0,
                    "raw_policy_regression_count": 0,
                    "invalid_action_mask_count": 0,
                    "fallback_or_open_grid_count": 0,
                    "safety_regression_count": 0,
                    "contract_violation_count": 0,
                    "path_cost_regression_count": 0,
                    "risk_regression_count": 0,
                    "source_selection_regression_count": 0,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_source_batch_summary(self) -> None:
        (self.source_root / "batch-evaluation-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "path-feedback-batch-evaluation-summary/v1",
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

    def _candidate(
        self,
        context_id: str,
        *,
        action: int,
        path_cost: float,
        risk: float,
        source_selected: bool,
    ) -> dict:
        return {
            "context_id": context_id,
            "context_id_schema_version": "policy-context-id/v1",
            "context_id_source": "stable_semantic_fields",
            "action_index": action,
            "source_action_index": action,
            "cell": [10 + action, 10],
            "policy_target_cell": [10 + action, 10],
            "execution_goal_cell": [10 + action, 10],
            "candidate_role": "projected_execution_target",
            "reachable": True,
            "replan_required": False,
            "open_grid_fallback_used": False,
            "path_cost": path_cost,
            "risk": risk,
            "utility": 0.8 - action * 0.1,
            "platform_goal_feasibility": {"contract_reachable": True},
            "candidate_generation": {
                "source_selection_status": "source_selected" if source_selected else "not_source_selected",
                "source_selection_quality_regression": False,
            },
        }

    def test_mining_converts_raw_regression_to_pairwise_sample(self) -> None:
        self._write_mining_artifacts()

        completed = self._run_mining()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.holdout_root / "raw-policy-regression-mining-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["raw_policy_regression_input_count"], 1)
        self.assertEqual(summary["raw_policy_regression_preference_pair_count"], 1)
        self.assertEqual(summary["hard_positive_added_count"], 0)
        samples = [
            json.loads(line)
            for line in (self.holdout_root / "raw-policy-regression-preference-samples.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["sample_type"], "raw_policy_regression_preference_pair")
        self.assertEqual(samples[0]["preferred"]["context_id"], "ctx-source")
        self.assertEqual(samples[0]["alternative"]["context_id"], "ctx-raw")
        self.assertEqual(samples[0]["raw_policy_regression_reason_codes"], ["path_cost_regression", "risk_regression"])

    def test_readiness_advances_after_raw_policy_strict_rollout_passes(self) -> None:
        self._write_readiness_artifacts()

        completed = self._run_readiness()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.source_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["training_readiness_status"], "raw_policy_decision_alignment_evaluated")
        self.assertEqual(summary["training_blockers"], [])


if __name__ == "__main__":
    unittest.main()
