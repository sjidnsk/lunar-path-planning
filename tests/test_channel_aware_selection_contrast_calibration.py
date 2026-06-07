import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class ChannelAwareSelectionContrastCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "run_channel_aware_selection_contrast_calibration.sh"
        self.config = self.repo_root / "configs" / "channel_aware_selection_contrast_calibration_v1.json"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="channel-aware-selection-contrast-"))
        self.batch_root = self.temp_dir / "batch"
        self.batch_root.mkdir(parents=True)
        self.git_snapshot = self._current_git_snapshot()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _run_calibration(self, *args: str) -> subprocess.CompletedProcess[str]:
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

    def _write_sources(self, *, git_mismatch: bool = False) -> None:
        source_git = self.git_snapshot
        if git_mismatch:
            source_git = {
                **self.git_snapshot,
                "parent": {**self.git_snapshot["parent"], "sha": "0" * 40},
            }
        records = [
            self._audit_record(
                "all-all-k3",
                "npz_tradeoff_contrast",
                0,
                cell=[2, 2],
                astar_selected_cell=[2, 2],
                recommendation="keep",
                reason_codes=["channel_aware_quality_improved", "path_cost_tradeoff"],
                path_cost_delta=0.1,
                channel_cost_delta=-1.0,
                high_cost_exposure_delta=-1.0,
            ),
            self._audit_record(
                "all-all-k3",
                "npz_tradeoff_contrast",
                1,
                cell=[1, 3],
                astar_selected_cell=[2, 2],
                recommendation="keep",
                reason_codes=["channel_aware_quality_improved", "path_cost_tradeoff"],
                path_cost_delta=0.8,
                channel_cost_delta=-4.0,
                high_cost_exposure_delta=-3.0,
            ),
            self._audit_record(
                "all-all-k3",
                "npz_blocked",
                0,
                cell=[5, 5],
                astar_selected_cell=[5, 5],
                recommendation="reject",
                blocker_reason="goal_blocked",
                reason_codes=["goal_blocked"],
            ),
            self._audit_record(
                "all-all-k3",
                "npz_same_as_baseline",
                0,
                cell=[6, 6],
                astar_selected_cell=[6, 6],
                recommendation="downweight",
                blocker_reason="same_as_baseline",
                reason_codes=["same_as_baseline"],
            ),
        ]
        comparison = {
            "schema_version": "policy-decision-selection-comparison-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "batch_root": str(self.batch_root),
            "channel_aware_decision_audit": {
                "schema_version": "channel-aware-decision-audit/v1",
                "paired_scenario_count": 3,
                "channel_aware_candidate_count": len(records),
                "selected_candidate_changed_count": 0,
                "selected_candidate_changed_rate": 0.0,
                "recommendation_counts": {
                    "downweight": 1,
                    "keep": 2,
                    "needs_more_evidence": 0,
                    "reject": 1,
                },
                "records": records,
            },
        }
        evidence = {
            "schema_version": "channel-aware-policy-target-selection-evidence-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "selected_candidate_changed_rate": 0.0,
            "supports_policy_target_selection_improvement_claim": False,
            "git_provenance": {
                "current": source_git,
                "readiness": {"current": source_git},
            },
        }
        (self.batch_root / "policy-decision-selection-comparison-summary.json").write_text(
            json.dumps(comparison, indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "channel-aware-policy-target-selection-evidence-summary.json").write_text(
            json.dumps(evidence, indent=2),
            encoding="utf-8",
        )

    def _audit_record(
        self,
        pair_key: str,
        scenario_id: str,
        action_index: int,
        *,
        cell: list[int],
        astar_selected_cell: list[int],
        recommendation: str,
        reason_codes: list[str],
        path_cost_delta: float | None = None,
        channel_cost_delta: float | None = None,
        high_cost_exposure_delta: float | None = None,
        blocker_reason: str | None = "selected",
    ) -> dict:
        return {
            "pair_key": pair_key,
            "astar_run_id": f"{pair_key}-astar",
            "channel_aware_run_id": f"{pair_key}-channel-aware",
            "scenario_id": scenario_id,
            "scenario_group": "stress",
            "action_index": action_index,
            "cell": cell,
            "astar_selected_cell": astar_selected_cell,
            "channel_aware_selected_cell": astar_selected_cell,
            "selected_candidate_changed": False,
            "selected": recommendation == "keep",
            "quality_improvement": recommendation == "keep",
            "risk_or_high_cost_improvement": recommendation == "keep",
            "path_cost_tradeoff": "path_cost_tradeoff" in reason_codes,
            "blocker_reason": blocker_reason,
            "recommendation": recommendation,
            "reason_codes": reason_codes,
            "comparison": {
                "path_changed": recommendation == "keep",
                "path_cost_delta": path_cost_delta,
                "channel_cost_delta": channel_cost_delta,
                "high_cost_exposure_delta": high_cost_exposure_delta,
            },
        }

    def test_calibration_selects_channel_quality_contrast_without_claiming_training_ready(self) -> None:
        self._write_sources()

        completed = self._run_calibration(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "channel-aware-selection-contrast-calibration-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["schema_version"], "channel-aware-selection-contrast-calibration-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["source_selected_candidate_changed_rate"], 0.0)
        self.assertEqual(summary["selected_candidate_changed_count"], 1)
        self.assertEqual(summary["selected_candidate_changed_rate"], 1 / 3)
        self.assertEqual(summary["changed_scenario_ids"], ["npz_tradeoff_contrast"])
        self.assertEqual(summary["keep_selected_candidate_count"], 1)
        self.assertEqual(summary["keep_selected_candidate_rate"], 1.0)
        self.assertEqual(summary["goal_blocked_count"], 1)
        self.assertEqual(summary["same_as_baseline_count"], 1)
        self.assertEqual(summary["blocked_candidate_count"], 1)
        self.assertEqual(summary["blocked_candidate_rate"], 0.25)
        self.assertEqual(summary["path_cost_tradeoff_count"], 2)
        self.assertEqual(summary["channel_cost_delta_stats"]["count"], 2)
        self.assertEqual(summary["channel_cost_delta_stats"]["min"], -4.0)
        self.assertEqual(summary["high_cost_exposure_delta_stats"]["mean"], -2.0)
        self.assertEqual(summary["safety_regression_count"], 0)
        self.assertEqual(
            summary["recommended_training_readiness_action"],
            "rerun_training_readiness_gate_after_selection_contrast_calibration",
        )
        self.assertFalse(summary["runs_training"])
        self.assertTrue(summary["channel_aware_backend_opt_in"])
        self.assertTrue(summary["does_not_modify_default_astar"])
        self.assertFalse(summary["policy_target_selection_improvement_claimed_without_evidence"])

    def test_calibration_counts_platform_goal_contract_mismatch(self) -> None:
        self._write_sources()
        comparison_path = self.batch_root / "policy-decision-selection-comparison-summary.json"
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        blocked = comparison["channel_aware_decision_audit"]["records"][2]
        blocked["reason_codes"] = ["goal_blocked", "platform_inflated_goal_blocked"]
        blocked["failure_taxonomy"] = "platform_inflated_goal_blocked"
        blocked["platform_goal_classification"] = "platform_inflated_goal_blocked"
        blocked["platform_goal_feasibility"] = {
            "classification": "platform_inflated_goal_blocked",
            "nearest_inflated_passable_anchor": [6, 5],
            "proxy_route_comparison": {
                "scope": "audit_proxy_anchor_not_same_cell",
                "same_cell_positive_evidence": False,
            },
        }
        comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

        completed = self._run_calibration(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "channel-aware-selection-contrast-calibration-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["goal_blocked_count"], 1)
        self.assertEqual(summary["platform_goal_contract_mismatch_count"], 1)
        self.assertEqual(summary["platform_goal_anchor_available_count"], 1)
        self.assertEqual(summary["platform_goal_unresolved_count"], 0)
        self.assertEqual(
            summary["platform_goal_feasibility_class_counts"]["platform_inflated_goal_blocked"],
            1,
        )

    def test_validate_only_reports_git_mismatch_without_writing_summary(self) -> None:
        self._write_sources(git_mismatch=True)

        completed = self._run_calibration(
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
            (self.batch_root / "channel-aware-selection-contrast-calibration-summary.json").exists()
        )
