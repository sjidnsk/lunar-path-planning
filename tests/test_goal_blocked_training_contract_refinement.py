import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class GoalBlockedTrainingContractRefinementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "run_goal_blocked_training_contract_refinement.sh"
        self.config = self.repo_root / "configs" / "goal_blocked_training_contract_refinement_v1.json"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="goal-blocked-contract-"))
        self.batch_root = self.temp_dir / "batch"
        self.batch_root.mkdir(parents=True)
        self.git_snapshot = self._current_git_snapshot()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _run_refinement(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHON"] = str(Path("/home/kai/anaconda3/envs/lunar-explorer/bin/python"))
        return subprocess.run(
            ["bash", str(self.script), *args],
            cwd=self.repo_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

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

    def _git_snapshot(self, *, git_mismatch: bool = False) -> dict:
        if not git_mismatch:
            return self.git_snapshot
        return {
            **self.git_snapshot,
            "parent": {**self.git_snapshot["parent"], "sha": "0" * 40},
        }

    def _write_sources(
        self,
        records: list[dict],
        *,
        git_mismatch: bool = False,
        safety_regression_count: int = 0,
        fallback_count: int = 0,
        contract_mutation: bool = False,
    ) -> None:
        git_snapshot = self._git_snapshot(git_mismatch=git_mismatch)
        excluded_count = sum(
            1
            for record in records
            if record.get("application_action") == "exclude_blocked_candidate_evidence"
            or record.get("recommendation") == "reject"
        )
        goal_blocked_count = sum(1 for record in records if "goal_blocked" in record.get("reason_codes", []))
        contract_guard = not contract_mutation
        review = {
            "schema_version": "policy-training-readiness-review-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "training_readiness_status": "needs_training_contract_refinement",
            "recommended_next_action": "needs_training_contract_refinement",
            "training_blockers": ["goal_blocked_candidates_excluded_from_training_positive_evidence"],
            "rejected_goal_blocked_count": goal_blocked_count,
            "excluded_candidate_count": excluded_count,
            "safety_regression_count": safety_regression_count,
            "fallback_or_open_grid_count": fallback_count,
            "audit_only": True,
            "no_ppo_training": True,
            "runs_training": False,
            "git_provenance": {"current": git_snapshot, "current_matches_sources": not git_mismatch},
            "contract_impact": {
                "training_contract_status": "compatible_audit_only",
                "contract_mutations": [],
            },
        }
        smoke = {
            "schema_version": "calibrated-policy-application-smoke-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "recommended_next_action": "ready_for_policy_training_readiness_review",
            "rejected_goal_blocked_count": goal_blocked_count,
            "safety_regression_count": safety_regression_count,
            "open_grid_fallback_used_count": fallback_count,
            "audit_only": True,
            "no_ppo_training": True,
            "runs_training": False,
            "does_not_modify_default_astar": True,
            "does_not_modify_ppo": contract_guard,
            "does_not_modify_network": contract_guard,
            "does_not_modify_action_space": contract_guard,
            "does_not_modify_model_explorer_contract": contract_guard,
            "does_not_modify_path_planner_route_contract": contract_guard,
            "does_not_modify_path_planner_sidecar_contract": contract_guard,
            "no_ackermann_feasible_trajectory_claim": True,
            "git_provenance": {"current": git_snapshot, "current_matches_sources": not git_mismatch},
        }
        application = {
            "schema_version": "policy-robustness-application-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "git_provenance": {"current": git_snapshot},
            "channel_aware_application": {
                "schema_version": "channel-aware-application-smoke/v1",
                "record_count": len(records),
                "records": records,
            },
        }
        calibration = {
            "schema_version": "channel-aware-selection-contrast-calibration-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "goal_blocked_count": goal_blocked_count,
            "safety_regression_count": safety_regression_count,
            "git_provenance": {"current": git_snapshot},
        }
        files = {
            "policy-training-readiness-review-summary.json": review,
            "calibrated-policy-application-smoke-summary.json": smoke,
            "policy-robustness-application-summary.json": application,
            "channel-aware-selection-contrast-calibration-summary.json": calibration,
        }
        for filename, payload in files.items():
            (self.batch_root / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _excluded_record(self, scenario_id: str, *, comparison: dict | None, reason: str = "goal_blocked") -> dict:
        return {
            "scenario_id": scenario_id,
            "pair_key": "all-all-k3",
            "action_index": 1,
            "cell": [10, 12],
            "recommendation": "reject",
            "application_action": "exclude_blocked_candidate_evidence",
            "application_sample_weight": 0.0,
            "selected_candidate_changed": False,
            "path_cost_tradeoff": False,
            "reason_codes": [reason],
            "application_reason_codes": ["channel_aware_application_exclude_blocked_candidate_evidence"],
            "comparison": comparison
            if comparison is not None
            else {
                "path_changed": False,
                "path_cost_delta": None,
                "channel_cost_delta": None,
                "high_cost_exposure_delta": None,
                "risk_delta": None,
            },
        }

    def test_refinement_keeps_ambiguous_goal_blocked_out_of_negative_training(self) -> None:
        self._write_sources(
            [
                self._excluded_record("ambiguous-goal-blocked", comparison=None),
                self._excluded_record(
                    "explicit-negative",
                    comparison={
                        "path_changed": True,
                        "path_cost_delta": 1.5,
                        "channel_cost_delta": 3.0,
                        "high_cost_exposure_delta": 2.0,
                        "risk_delta": 0.5,
                    },
                ),
                self._excluded_record("plain-reject", comparison=None, reason="same_as_baseline"),
            ]
        )

        completed = self._run_refinement("--batch-root", str(self.batch_root), "--config", str(self.config))

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "goal-blocked-training-contract-refinement-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["schema_version"], "goal-blocked-training-contract-refinement-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["goal_blocked_count"], 2)
        self.assertEqual(summary["excluded_candidate_count"], 3)
        self.assertEqual(summary["negative_evidence_candidate_count"], 1)
        self.assertEqual(summary["needs_regeneration_count"], 1)
        self.assertEqual(summary["contract_decision_counts"]["eligible_negative_evidence"], 1)
        self.assertEqual(summary["contract_decision_counts"]["needs_regeneration"], 1)
        self.assertEqual(summary["contract_decision_counts"]["excluded_from_positive_training"], 1)
        self.assertIn("goal_blocked_records_need_regeneration", summary["contract_blockers"])
        self.assertEqual(summary["recommended_next_action"], "needs_goal_blocked_contract_refinement")
        self.assertTrue(summary["audit_only"])
        self.assertTrue(summary["no_ppo_training"])

    def test_refinement_recommends_readiness_rerun_when_excluded_records_have_contracts(self) -> None:
        self._write_sources(
            [
                self._excluded_record(
                    "explicit-negative",
                    comparison={
                        "path_changed": True,
                        "path_cost_delta": 2.5,
                        "channel_cost_delta": 4.0,
                        "high_cost_exposure_delta": 1.0,
                        "risk_delta": 0.25,
                    },
                ),
                self._excluded_record("plain-reject", comparison=None, reason="same_as_baseline"),
            ]
        )

        completed = self._run_refinement("--batch-root", str(self.batch_root), "--config", str(self.config))

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "goal-blocked-training-contract-refinement-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["contract_blockers"], [])
        self.assertEqual(summary["negative_evidence_candidate_count"], 1)
        self.assertEqual(summary["needs_regeneration_count"], 0)
        self.assertEqual(summary["recommended_next_action"], "rerun_policy_training_readiness_review")
        self.assertFalse(summary["runs_training"])
        self.assertTrue(summary["no_ppo_training"])

    def test_validate_only_blocks_current_git_mismatch_without_writing_summary(self) -> None:
        self._write_sources([self._excluded_record("ambiguous-goal-blocked", comparison=None)], git_mismatch=True)

        completed = self._run_refinement(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--validate-only",
        )

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        validation = json.loads(completed.stdout.splitlines()[0])
        self.assertEqual(validation["status"], "validation failed")
        self.assertIn("current_git_provenance_mismatch", validation["reason_codes"])
        self.assertFalse((self.batch_root / "goal-blocked-training-contract-refinement-summary.json").exists())


if __name__ == "__main__":
    unittest.main()
