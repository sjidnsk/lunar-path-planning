import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class PolicyTrainingReadinessReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "run_policy_training_readiness_review.sh"
        self.config = self.repo_root / "configs" / "policy_training_readiness_review_v1.json"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="policy-training-readiness-"))
        self.batch_root = self.temp_dir / "batch"
        self.batch_root.mkdir(parents=True)
        self.git_snapshot = self._current_git_snapshot()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _run_review(self, *args: str) -> subprocess.CompletedProcess[str]:
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
        *,
        source_rate: float = 0.0,
        calibrated_rate: float = 0.5,
        applied_count: int = 2,
        rejected_goal_blocked_count: int = 0,
        safety_regression_count: int = 0,
        smoke_recommended: str = "ready_for_policy_training_readiness_review",
        open_grid_fallback_used_count: int = 0,
        contract_mutation: bool = False,
        git_mismatch: bool = False,
    ) -> None:
        git_snapshot = self._git_snapshot(git_mismatch=git_mismatch)
        changed_ids = [
            "npz_blocked_nearby_clearance_detour",
            "npz_high_cost_exposure_rock_detour",
        ]
        contract_guard = not contract_mutation
        smoke = {
            "schema_version": "calibrated-policy-application-smoke-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "source_selected_candidate_changed_rate": source_rate,
            "calibrated_selected_candidate_changed_rate": calibrated_rate,
            "calibrated_selection_rate_delta": calibrated_rate - source_rate,
            "applied_calibrated_candidate_count": applied_count,
            "changed_scenario_ids": changed_ids,
            "rejected_goal_blocked_count": rejected_goal_blocked_count,
            "safety_regression_count": safety_regression_count,
            "application_gate_reason_codes": [],
            "recommended_next_action": smoke_recommended,
            "audit_only": True,
            "runs_training": False,
            "no_ppo_training": True,
            "no_large_scale_training": True,
            "channel_aware_backend_opt_in": True,
            "does_not_modify_default_astar": True,
            "does_not_modify_ppo": contract_guard,
            "does_not_modify_network": contract_guard,
            "does_not_modify_action_space": contract_guard,
            "does_not_modify_model_explorer_contract": contract_guard,
            "does_not_modify_path_planner_route_contract": contract_guard,
            "does_not_modify_path_planner_sidecar_contract": contract_guard,
            "no_ackermann_feasible_trajectory_claim": True,
            "open_grid_fallback_used_count": open_grid_fallback_used_count,
            "git_provenance": {"current": git_snapshot, "current_matches_sources": not git_mismatch},
        }
        readiness = {
            "schema_version": "channel-aware-training-readiness-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "readiness_status": "ready_for_calibrated_policy_application_smoke",
            "calibrated_readiness_status": "ready_for_calibrated_policy_application_smoke",
            "source_selected_candidate_changed_rate": source_rate,
            "calibration_selected_candidate_changed_rate": calibrated_rate,
            "calibration_safety_regression_count": safety_regression_count,
            "git_provenance": {"current": git_snapshot},
        }
        coverage = {
            "schema_version": "channel-aware-contrast-coverage-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "source_selected_candidate_changed_rate": source_rate,
            "calibrated_selected_candidate_changed_rate": calibrated_rate,
            "changed_scenario_ids": changed_ids,
            "blocked_candidate_rate": 0.0,
            "safety_regression_count": safety_regression_count,
            "recommended_next_action": "ready_for_calibrated_policy_application_smoke",
            "open_grid_fallback_used_count": open_grid_fallback_used_count,
            "git_provenance": {"current": git_snapshot},
        }
        calibration = {
            "schema_version": "channel-aware-selection-contrast-calibration-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "source_selected_candidate_changed_rate": source_rate,
            "selected_candidate_changed_count": applied_count,
            "selected_candidate_changed_rate": calibrated_rate,
            "changed_scenario_ids": changed_ids,
            "goal_blocked_count": rejected_goal_blocked_count,
            "safety_regression_count": safety_regression_count,
            "git_provenance": {"current": git_snapshot},
        }
        (self.batch_root / "calibrated-policy-application-smoke-summary.json").write_text(
            json.dumps(smoke, indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "channel-aware-training-readiness-summary.json").write_text(
            json.dumps(readiness, indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "channel-aware-contrast-coverage-summary.json").write_text(
            json.dumps(coverage, indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "channel-aware-selection-contrast-calibration-summary.json").write_text(
            json.dumps(calibration, indent=2),
            encoding="utf-8",
        )

    def test_review_allows_limited_training_dry_run_when_contract_is_clear(self) -> None:
        self._write_sources()

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["schema_version"], "policy-training-readiness-review-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["training_readiness_status"], "ready_for_limited_policy_training_dry_run")
        self.assertEqual(summary["recommended_next_action"], "ready_for_limited_policy_training_dry_run")
        self.assertEqual(summary["source_selected_candidate_changed_rate"], 0.0)
        self.assertEqual(summary["calibrated_selected_candidate_changed_rate"], 0.5)
        self.assertEqual(summary["applied_calibrated_candidate_count"], 2)
        self.assertEqual(summary["training_positive_candidate_count"], 2)
        self.assertEqual(summary["excluded_candidate_count"], 0)
        self.assertEqual(summary["training_blockers"], [])
        self.assertEqual(summary["contract_impact"]["training_contract_status"], "compatible_audit_only")
        self.assertTrue(summary["git_provenance"]["current_matches_sources"])
        self.assertTrue(summary["audit_only"])
        self.assertTrue(summary["no_ppo_training"])
        self.assertTrue(summary["does_not_modify_default_astar"])
        self.assertTrue(summary["no_ackermann_feasible_trajectory_claim"])

    def test_review_requires_contract_refinement_when_goal_blocked_candidates_exist(self) -> None:
        self._write_sources(rejected_goal_blocked_count=3)

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["training_readiness_status"], "needs_training_contract_refinement")
        self.assertEqual(summary["recommended_next_action"], "needs_training_contract_refinement")
        self.assertEqual(summary["training_positive_candidate_count"], 2)
        self.assertEqual(summary["excluded_candidate_count"], 3)
        self.assertIn(
            "goal_blocked_candidates_excluded_from_training_positive_evidence",
            summary["training_blockers"],
        )

    def test_review_blocks_fallback_safety_and_contract_mutation_from_training_readiness(self) -> None:
        for kwargs, expected_reason in (
            (
                {"open_grid_fallback_used_count": 1},
                "fallback_or_open_grid_evidence_blocks_training_readiness",
            ),
            (
                {"safety_regression_count": 1},
                "safety_regression_blocks_training_readiness",
            ),
            (
                {"contract_mutation": True},
                "contract_mutation_blocks_training_readiness",
            ),
        ):
            with self.subTest(expected_reason=expected_reason):
                shutil.rmtree(self.batch_root)
                self.batch_root.mkdir(parents=True)
                self._write_sources(**kwargs)

                completed = self._run_review(
                    "--batch-root",
                    str(self.batch_root),
                    "--config",
                    str(self.config),
                )

                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                summary = json.loads(
                    (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(summary["training_readiness_status"], "needs_training_contract_refinement")
                self.assertEqual(summary["recommended_next_action"], "needs_training_contract_refinement")
                self.assertIn(expected_reason, summary["training_blockers"])

    def test_validate_only_blocks_current_git_mismatch_without_writing_summary(self) -> None:
        self._write_sources(git_mismatch=True)

        completed = self._run_review(
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
        self.assertFalse((self.batch_root / "policy-training-readiness-review-summary.json").exists())


if __name__ == "__main__":
    unittest.main()
