import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class ChannelAwareContrastCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "run_channel_aware_contrast_coverage.sh"
        self.config = self.repo_root / "configs" / "channel_aware_contrast_coverage_v1.json"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="channel-aware-contrast-coverage-"))
        self.batch_root = self.temp_dir / "batch"
        self.batch_root.mkdir(parents=True)
        self.git_snapshot = self._current_git_snapshot()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _run_coverage(self, *args: str) -> subprocess.CompletedProcess[str]:
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

    def _write_calibration_summary(
        self,
        *,
        changed_scenario_ids: list[str],
        context_count: int,
        eligible_context_count: int,
        source_rate: float = 0.0,
        calibrated_rate: float | None = None,
        blocked_candidate_rate: float = 0.25,
        safety_regression_count: int = 0,
        status: str = "passed",
        reason_codes: list[str] | None = None,
        git_mismatch: bool = False,
    ) -> Path:
        git_snapshot = self.git_snapshot
        if git_mismatch:
            git_snapshot = {
                **self.git_snapshot,
                "parent": {**self.git_snapshot["parent"], "sha": "0" * 40},
            }
        records = []
        for index in range(context_count):
            changed = index < len(changed_scenario_ids)
            eligible = index < eligible_context_count
            scenario_id = (
                changed_scenario_ids[index]
                if changed
                else f"npz_no_contrast_{index}"
            )
            records.append(
                {
                    "pair_key": "all-all-k3",
                    "scenario_id": scenario_id,
                    "astar_selected_cell": [10, index],
                    "calibrated_channel_aware_selected_cell": [11, index] if changed else [10, index],
                    "selected_candidate_changed": changed,
                    "selection_reason": (
                        "channel_quality_contrast_selected"
                        if eligible
                        else "no_eligible_channel_quality_contrast_candidate"
                    ),
                    "selected_candidate_score": 2.0 if eligible else None,
                    "path_cost_tradeoff": eligible,
                    "safety_regression": False,
                }
            )
        calibrated_rate = (
            len(changed_scenario_ids) / context_count
            if calibrated_rate is None and context_count
            else float(calibrated_rate or 0.0)
        )
        payload = {
            "schema_version": "channel-aware-selection-contrast-calibration-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": status,
            "reason_codes": list(reason_codes or []),
            "source_selected_candidate_changed_rate": source_rate,
            "selected_candidate_changed_count": len(changed_scenario_ids),
            "selected_candidate_changed_rate": calibrated_rate,
            "changed_scenario_ids": list(changed_scenario_ids),
            "blocked_candidate_rate": blocked_candidate_rate,
            "channel_cost_delta_stats": {
                "count": eligible_context_count,
                "min": -5.0,
                "max": -1.0,
                "mean": -3.0,
            },
            "high_cost_exposure_delta_stats": {
                "count": eligible_context_count,
                "min": -6.0,
                "max": -2.0,
                "mean": -4.0,
            },
            "safety_regression_count": safety_regression_count,
            "calibrated_selection_records": records,
            "git_provenance": {"current": git_snapshot},
        }
        path = self.batch_root / "channel-aware-selection-contrast-calibration-summary.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def test_coverage_summary_recommends_more_scenarios_when_contrast_is_still_low(self) -> None:
        self._write_calibration_summary(
            changed_scenario_ids=["npz_low_confidence_risk_band", "npz_shadow_corridor"],
            context_count=8,
            eligible_context_count=3,
            blocked_candidate_rate=0.6,
        )

        completed = self._run_coverage(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "channel-aware-contrast-coverage-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["schema_version"], "channel-aware-contrast-coverage-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["scenario_count"], 8)
        self.assertEqual(summary["contrast_eligible_context_count"], 3)
        self.assertEqual(summary["source_selected_candidate_changed_rate"], 0.0)
        self.assertEqual(summary["calibrated_selected_candidate_changed_rate"], 0.25)
        self.assertEqual(
            summary["changed_scenario_ids"],
            ["npz_low_confidence_risk_band", "npz_shadow_corridor"],
        )
        self.assertEqual(summary["blocked_candidate_rate"], 0.6)
        self.assertEqual(summary["no_eligible_contrast_count"], 5)
        self.assertEqual(summary["channel_cost_delta_stats"]["mean"], -3.0)
        self.assertEqual(summary["high_cost_exposure_delta_stats"]["mean"], -4.0)
        self.assertEqual(summary["safety_regression_count"], 0)
        self.assertEqual(summary["recommended_next_action"], "needs_more_contrast_scenarios")
        self.assertFalse(summary["runs_training"])
        self.assertTrue(summary["channel_aware_backend_opt_in"])
        self.assertTrue(summary["does_not_modify_default_astar"])
        self.assertFalse(summary["policy_target_selection_improvement_claimed"])

    def test_validate_only_blocks_current_git_mismatch_without_writing_summary(self) -> None:
        self._write_calibration_summary(
            changed_scenario_ids=[
                "npz_low_confidence_risk_band",
                "npz_shadow_corridor",
                "npz_rock_field_multi_pose",
                "npz_low_centerline_bad_channel",
            ],
            context_count=8,
            eligible_context_count=5,
            git_mismatch=True,
        )

        completed = self._run_coverage(
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
        self.assertFalse(
            (self.batch_root / "channel-aware-contrast-coverage-summary.json").exists()
        )


if __name__ == "__main__":
    unittest.main()
