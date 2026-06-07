import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class CalibratedPolicyApplicationSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "run_calibrated_policy_application_smoke.sh"
        self.config = self.repo_root / "configs" / "calibrated_policy_application_smoke_v1.json"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="calibrated-policy-smoke-"))
        self.batch_root = self.temp_dir / "batch"
        self.batch_root.mkdir(parents=True)
        self.git_snapshot = self._current_git_snapshot()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _run_smoke(self, *args: str) -> subprocess.CompletedProcess[str]:
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

    def _write_sources(
        self,
        *,
        source_rate: float = 0.0,
        calibrated_rate: float = 0.5,
        safety_regression_count: int = 0,
        coverage_recommended: str = "ready_for_calibrated_policy_application_smoke",
        readiness_status: str = "ready_for_calibrated_policy_application_smoke",
        platform_goal_contract_mismatch_count: int = 0,
        git_mismatch: bool = False,
    ) -> None:
        git_snapshot = self.git_snapshot
        if git_mismatch:
            git_snapshot = {
                **self.git_snapshot,
                "parent": {**self.git_snapshot["parent"], "sha": "0" * 40},
            }
        changed_ids = [
            "npz_blocked_nearby_clearance_detour",
            "npz_high_cost_exposure_rock_detour",
        ]
        records = [
            self._calibrated_record(
                "npz_blocked_nearby_clearance_detour",
                [15, 9],
                [15, 10],
                changed=True,
            ),
            self._calibrated_record(
                "npz_high_cost_exposure_rock_detour",
                [14, 7],
                [14, 9],
                changed=True,
                safety_regression=safety_regression_count > 0,
            ),
            self._calibrated_record(
                "npz_no_contrast",
                [10, 7],
                [10, 7],
                changed=False,
                reason="no_eligible_channel_quality_contrast_candidate",
                recommendation=None,
            ),
        ]
        calibration = {
            "schema_version": "channel-aware-selection-contrast-calibration-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "source_selected_candidate_changed_rate": source_rate,
            "selected_candidate_changed_count": 2,
            "selected_candidate_changed_rate": calibrated_rate,
            "changed_scenario_ids": changed_ids,
            "goal_blocked_count": 3,
            "platform_goal_contract_mismatch_count": platform_goal_contract_mismatch_count,
            "platform_goal_anchor_available_count": platform_goal_contract_mismatch_count,
            "platform_goal_unresolved_count": 0,
            "platform_goal_feasibility_class_counts": (
                {"platform_inflated_goal_blocked": platform_goal_contract_mismatch_count}
                if platform_goal_contract_mismatch_count
                else {}
            ),
            "blocked_candidate_rate": 0.25,
            "safety_regression_count": safety_regression_count,
            "calibrated_selection_records": records,
            "git_provenance": {"current": git_snapshot},
        }
        coverage = {
            "schema_version": "channel-aware-contrast-coverage-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "scenario_count": 5,
            "contrast_eligible_context_count": 2,
            "source_selected_candidate_changed_rate": source_rate,
            "calibrated_selected_candidate_changed_rate": calibrated_rate,
            "changed_scenario_ids": changed_ids,
            "blocked_candidate_rate": 0.25,
            "safety_regression_count": safety_regression_count,
            "recommended_next_action": coverage_recommended,
            "git_provenance": {"current": git_snapshot},
        }
        readiness = {
            "schema_version": "channel-aware-training-readiness-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "readiness_status": readiness_status,
            "calibrated_readiness_status": readiness_status,
            "source_selected_candidate_changed_rate": source_rate,
            "calibration_selected_candidate_changed_rate": calibrated_rate,
            "calibration_safety_regression_count": safety_regression_count,
            "git_provenance": {"current": git_snapshot},
        }
        (self.batch_root / "channel-aware-selection-contrast-calibration-summary.json").write_text(
            json.dumps(calibration, indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "channel-aware-contrast-coverage-summary.json").write_text(
            json.dumps(coverage, indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "channel-aware-training-readiness-summary.json").write_text(
            json.dumps(readiness, indent=2),
            encoding="utf-8",
        )

    def _calibrated_record(
        self,
        scenario_id: str,
        source_cell: list[int],
        calibrated_cell: list[int],
        *,
        changed: bool,
        safety_regression: bool = False,
        reason: str = "channel_quality_contrast_selected",
        recommendation: str | None = "keep",
    ) -> dict:
        return {
            "pair_key": "all-all-k3",
            "scenario_id": scenario_id,
            "astar_selected_cell": source_cell,
            "calibrated_channel_aware_selected_cell": calibrated_cell,
            "selected_candidate_changed": changed,
            "selection_reason": reason,
            "selected_action_index": 1 if changed else None,
            "selected_candidate_score": 12.0 if changed else None,
            "recommendation": recommendation,
            "path_cost_tradeoff": changed,
            "safety_regression": safety_regression,
        }

    def test_smoke_applies_calibrated_candidates_without_training(self) -> None:
        self._write_sources()

        completed = self._run_smoke(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "calibrated-policy-application-smoke-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["schema_version"], "calibrated-policy-application-smoke-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["source_selected_candidate_changed_rate"], 0.0)
        self.assertEqual(summary["calibrated_selected_candidate_changed_rate"], 0.5)
        self.assertEqual(summary["applied_calibrated_candidate_count"], 2)
        self.assertEqual(
            summary["changed_scenario_ids"],
            ["npz_blocked_nearby_clearance_detour", "npz_high_cost_exposure_rock_detour"],
        )
        self.assertEqual(summary["rejected_goal_blocked_count"], 3)
        self.assertEqual(summary["safety_regression_count"], 0)
        self.assertEqual(summary["recommended_next_action"], "ready_for_policy_training_readiness_review")
        self.assertTrue(summary["audit_only"])
        self.assertTrue(summary["no_ppo_training"])
        self.assertTrue(summary["does_not_modify_default_astar"])
        self.assertTrue(summary["channel_aware_backend_opt_in"])
        self.assertTrue(summary["git_provenance"]["current_matches_sources"])

    def test_smoke_propagates_platform_goal_contract_mismatch_counts(self) -> None:
        self._write_sources(platform_goal_contract_mismatch_count=3)

        completed = self._run_smoke(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "calibrated-policy-application-smoke-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["rejected_goal_blocked_count"], 3)
        self.assertEqual(summary["platform_goal_contract_mismatch_count"], 3)
        self.assertEqual(summary["platform_goal_anchor_available_count"], 3)
        self.assertEqual(summary["platform_goal_unresolved_count"], 0)
        self.assertEqual(
            summary["platform_goal_feasibility_class_counts"]["platform_inflated_goal_blocked"],
            3,
        )

    def test_smoke_recommends_gate_refinement_when_safety_regresses(self) -> None:
        self._write_sources(safety_regression_count=1)

        completed = self._run_smoke(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "calibrated-policy-application-smoke-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["safety_regression_count"], 1)
        self.assertEqual(summary["applied_calibrated_candidate_count"], 1)
        self.assertEqual(summary["recommended_next_action"], "needs_application_gate_refinement")
        self.assertIn("safety_regression_blocks_policy_training", summary["application_gate_reason_codes"])

    def test_validate_only_blocks_current_git_mismatch_without_writing_summary(self) -> None:
        self._write_sources(git_mismatch=True)

        completed = self._run_smoke(
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
        self.assertFalse((self.batch_root / "calibrated-policy-application-smoke-summary.json").exists())


if __name__ == "__main__":
    unittest.main()
