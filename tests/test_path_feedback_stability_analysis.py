import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class PathFeedbackStabilityAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "run_path_feedback_stability_analysis.sh"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="path-feedback-stability-"))
        self.batch_root = self.temp_dir / "batch"
        self.batch_root.mkdir(parents=True)
        self.git_snapshot = {
            "parent": {"path": ".", "sha": "1" * 40, "branch": "main"},
            "submodules": {
                "dev-platform-constraints": {
                    "path": "dev-platform-constraints",
                    "sha": "2" * 40,
                    "branch": "dev-platform-constraints",
                },
                "model-explorer": {
                    "path": "model-explorer",
                    "sha": "3" * 40,
                    "branch": "codex/candidate-list-policy-baseline",
                },
                "path-planner": {
                    "path": "path-planner",
                    "sha": "4" * 40,
                    "branch": "codex/phase2-postprocess",
                },
            },
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _run_stability(self, *args: str) -> subprocess.CompletedProcess[str]:
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

    def _write_batch(
        self,
        runs: list[dict],
        *,
        evaluation_override: dict | None = None,
        index_override: dict | None = None,
    ) -> None:
        run_index_runs = []
        source_summary_paths = []
        for run in runs:
            run_root = self.batch_root / run["run_id"]
            run_root.mkdir(parents=True, exist_ok=True)
            summary_path = run_root / "path-feedback-summary.json"
            report_path = run_root / "path-feedback-summary.md"
            summary = run.get("summary")
            if summary is not None:
                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                source_summary_paths.append(str(summary_path))
            report_path.write_text("# report\n", encoding="utf-8")
            command_args = {
                "scenario_set": run["scenario_set"],
                "diagnostic_profile": run["diagnostic_profile"],
                "top_k": run["top_k"],
                "output_root": str(run_root),
            }
            acceptance_metadata = run.get("acceptance_metadata")
            if acceptance_metadata is None and isinstance(summary, dict):
                acceptance_metadata = summary.get("acceptance_metadata", {})
            run_index_runs.append(
                {
                    "run_id": run["run_id"],
                    "status": run.get("status", "passed"),
                    "reason_codes": list(run.get("reason_codes", [])),
                    "command_args": command_args,
                    "source_paths": {
                        "output_root": str(run_root),
                        "summary": str(summary_path),
                        "report": str(report_path),
                    },
                    "summary_path": str(summary_path),
                    "report_path": str(report_path),
                    "acceptance_metadata": acceptance_metadata or {},
                    "git": run.get("git", self.git_snapshot),
                }
            )
        failed_run_ids = [run["run_id"] for run in run_index_runs if run["status"] == "failed"]
        run_index = {
            "schema_version": "path-feedback-batch-run-index/v1",
            "generated_at": "2026-06-02T00:00:00Z",
            "matrix_path": "configs/path_feedback_batch_dataset_v1.json",
            "output_root": str(self.batch_root),
            "run_count": len(run_index_runs),
            "passed_count": len(run_index_runs) - len(failed_run_ids),
            "failed_count": len(failed_run_ids),
            "failed_run_ids": failed_run_ids,
            "git": self.git_snapshot,
            "runs": run_index_runs,
        }
        if index_override:
            run_index.update(index_override)
        evaluation_summary = {
            "schema_version": "path-feedback-batch-evaluation-summary/v1",
            "generated_at": "2026-06-02T00:00:00Z",
            "matrix_path": "configs/path_feedback_batch_dataset_v1.json",
            "output_root": str(self.batch_root),
            "run_count": len(run_index_runs),
            "passed_count": len(run_index_runs) - len(failed_run_ids),
            "failed_count": len(failed_run_ids),
            "failed_run_ids": failed_run_ids,
            "failure_reason_code_counts": {},
            "open_grid_fallback_used_count": sum(
                1 for run in runs if isinstance(run.get("summary"), dict) and run["summary"].get("open_grid_fallback_used")
            ),
            "source_summary_paths": source_summary_paths,
            "scenario_group_summary": {},
            "runs": [],
        }
        if evaluation_override:
            evaluation_summary.update(evaluation_override)
        (self.batch_root / "batch-run-index.json").write_text(
            json.dumps(run_index, indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "batch-evaluation-summary.json").write_text(
            json.dumps(evaluation_summary, indent=2),
            encoding="utf-8",
        )

    def _summary(
        self,
        *,
        scenario_set: str,
        diagnostic_profile: str,
        top_k: int,
        scenario_id: str,
        scenario_group: str,
        failure_count: int = 0,
        replan_count: int = 0,
        iris_fallback_count: int = 0,
        region_graph_fallback_count: int = 0,
        region_graph_disconnected_count: int = 0,
        selection_changed: bool = False,
        replacement_reason: str = "unchanged",
        failure_sources: list[str] | None = None,
        open_grid: bool = False,
        channel_aware_report_count: int = 0,
        channel_aware_selected_count: int = 0,
        channel_aware_fallback_count: int = 0,
    ) -> dict:
        failure_sources = list(failure_sources or [])
        gate = {
            "status": "failed" if open_grid else "passed",
            "expected": False,
            "actual": open_grid,
            "reason_codes": ["open_grid_fallback_used"] if open_grid else ["open_grid_fallback_not_used"],
        }
        acceptance_metadata = {
            "schema_version": "path-feedback-acceptance-metadata/v1",
            "scenario_set": scenario_set,
            "diagnostic_profile": diagnostic_profile,
            "acceptance_gate": "semi-real-closed-loop"
            if scenario_set == "all" and diagnostic_profile == "all" and top_k == 3
            else "custom",
            "top_k": top_k,
            "planner_extra_args": [],
            "open_grid_fallback_used": open_grid,
            "open_grid_fallback_used_gate": gate,
        }
        candidate = {
            "action_index": 0,
            "cell": [1, 1],
            "reachable": failure_count == 0,
            "failure_reason": "goal_blocked" if failure_count else None,
            "replan_required": replan_count > 0,
            "diagnostic_interpretation": {
                "diagnostic_flags": failure_sources,
                "open_grid_fallback_used": open_grid,
                "iris_fallback_used": "iris_fallback" in failure_sources,
                "region_graph_fallback_used": "region_graph_fallback" in failure_sources,
                "region_graph_start_goal_connected": False
                if "region_graph_disconnected" in failure_sources
                else True,
            },
        }
        return {
            "schema_version": "path-feedback-summary/v1",
            "scenario_set": scenario_set,
            "diagnostic_profile": diagnostic_profile,
            "acceptance_gate": acceptance_metadata["acceptance_gate"],
            "top_k": top_k,
            "scenario_count": 1,
            "candidate_count": top_k,
            "path_planning_failure_count": failure_count,
            "replan_count": replan_count,
            "iris_fallback_count": iris_fallback_count,
            "region_graph_fallback_count": region_graph_fallback_count,
            "region_graph_disconnected_count": region_graph_disconnected_count,
            "open_grid_fallback_used": open_grid,
            "open_grid_fallback_used_gate": gate,
            "acceptance_metadata": acceptance_metadata,
            "scenario_group_summary": {
                scenario_group: {
                    "scenario_count": 1,
                    "candidate_count": top_k,
                    "reachable_count": top_k - failure_count,
                    "failure_count": failure_count,
                    "replan_count": replan_count,
                    "selection_changed_count": 1 if selection_changed else 0,
                    "iris_report_count": top_k,
                    "iris_fallback_count": iris_fallback_count,
                    "region_graph_fallback_count": region_graph_fallback_count,
                    "region_graph_start_goal_disconnected_count": region_graph_disconnected_count,
                    "channel_aware_astar_report_count": channel_aware_report_count,
                    "channel_aware_astar_selected_count": channel_aware_selected_count,
                    "channel_aware_astar_fallback_count": channel_aware_fallback_count,
                }
            },
            "channel_aware_astar_report_count": channel_aware_report_count,
            "channel_aware_astar_selected_count": channel_aware_selected_count,
            "channel_aware_astar_fallback_count": channel_aware_fallback_count,
            "channel_aware_astar_requested_backend_counts": (
                {"channel_aware_astar": channel_aware_report_count}
                if channel_aware_report_count
                else {}
            ),
            "channel_aware_astar_selected_backend_counts": (
                {
                    "channel_aware_astar": channel_aware_selected_count,
                    "astar": channel_aware_fallback_count,
                }
                if channel_aware_report_count
                else {}
            ),
            "channel_aware_astar_status_counts": (
                {
                    "selected": channel_aware_selected_count,
                    "fallback": channel_aware_fallback_count,
                }
                if channel_aware_report_count
                else {}
            ),
            "channel_aware_astar_fallback_reason_counts": (
                {"channel_search_failed:goal_blocked": channel_aware_fallback_count}
                if channel_aware_fallback_count
                else {}
            ),
            "channel_aware_astar_blocker_class_counts": (
                {
                    "selected": channel_aware_selected_count,
                    "goal_blocked": channel_aware_fallback_count,
                }
                if channel_aware_report_count
                else {}
            ),
            "channel_aware_astar_path_changed_count": channel_aware_selected_count,
            "channel_aware_astar_path_changed_rate": (
                channel_aware_selected_count / channel_aware_report_count
                if channel_aware_report_count
                else 0.0
            ),
            "channel_aware_astar_path_cost_delta_count": channel_aware_selected_count,
            "channel_aware_astar_path_cost_delta_min": 2.0 if channel_aware_selected_count else None,
            "channel_aware_astar_path_cost_delta_max": 2.0 if channel_aware_selected_count else None,
            "channel_aware_astar_path_cost_delta_mean": 2.0 if channel_aware_selected_count else None,
            "channel_aware_astar_channel_cost_delta_count": channel_aware_selected_count,
            "channel_aware_astar_channel_cost_delta_min": -4.0 if channel_aware_selected_count else None,
            "channel_aware_astar_channel_cost_delta_max": -4.0 if channel_aware_selected_count else None,
            "channel_aware_astar_channel_cost_delta_mean": -4.0 if channel_aware_selected_count else None,
            "channel_aware_astar_high_cost_exposure_delta_count": channel_aware_selected_count,
            "channel_aware_astar_high_cost_exposure_delta_min": -3.0 if channel_aware_selected_count else None,
            "channel_aware_astar_high_cost_exposure_delta_max": -3.0 if channel_aware_selected_count else None,
            "channel_aware_astar_high_cost_exposure_delta_mean": -3.0 if channel_aware_selected_count else None,
            "channel_aware_astar_candidate_audit": [],
            "scenarios": [
                {
                    "scenario_id": scenario_id,
                    "scenario_group": scenario_group,
                    "selected_cell_before_path_feedback": [2, 2],
                    "selected_cell_after_path_feedback": [1, 1] if selection_changed else [2, 2],
                    "selection_changed_by_path_feedback": selection_changed,
                    "open_grid_fallback_used": open_grid,
                    "path_feedback": {
                        "candidate_count": top_k,
                        "reachable_count": top_k - failure_count,
                        "failure_count": failure_count,
                        "replan_count": replan_count,
                        "failure_reasons": ["goal_blocked"] if failure_count else [],
                        "candidates": [candidate],
                    },
                    "diagnostic_interpretation": {
                        "target_replacement_reason": replacement_reason,
                        "failure_sources": failure_sources,
                        "primary_failure_reason": "goal_blocked" if failure_count else None,
                        "open_grid_fallback_used": open_grid,
                    },
                    "iris_diagnostics": {
                        "fallback_count": iris_fallback_count,
                    },
                    "region_graph_diagnostics": {
                        "fallback_count": region_graph_fallback_count,
                        "start_goal_disconnected_count": region_graph_disconnected_count,
                    },
                }
            ],
        }

    def test_stability_analysis_writes_three_machine_readable_summaries(self) -> None:
        smoke_summary = self._summary(
            scenario_set="smoke",
            diagnostic_profile="baseline",
            top_k=1,
            scenario_id="npz_shadow_corridor",
            scenario_group="smoke",
            selection_changed=False,
        )
        stress_summary = self._summary(
            scenario_set="all",
            diagnostic_profile="all",
            top_k=3,
            scenario_id="npz_near_blocked_corridor",
            scenario_group="stress",
            failure_count=3,
            replan_count=3,
            iris_fallback_count=3,
            region_graph_fallback_count=3,
            region_graph_disconnected_count=3,
            selection_changed=True,
            replacement_reason="no_feasible_candidate_after_path_feedback",
            channel_aware_report_count=6,
            channel_aware_selected_count=3,
            channel_aware_fallback_count=3,
            failure_sources=[
                "path_planning_failure",
                "replan_required",
                "iris_fallback",
                "region_graph_fallback",
                "region_graph_disconnected",
            ],
        )
        self._write_batch(
            [
                {
                    "run_id": "smoke-baseline-k1",
                    "scenario_set": "smoke",
                    "diagnostic_profile": "baseline",
                    "top_k": 1,
                    "summary": smoke_summary,
                },
                {
                    "run_id": "all-all-k3",
                    "scenario_set": "all",
                    "diagnostic_profile": "all",
                    "top_k": 3,
                    "summary": stress_summary,
                },
            ],
            evaluation_override={
                "channel_aware_astar_report_count": 6,
                "channel_aware_astar_selected_count": 3,
                "channel_aware_astar_fallback_count": 3,
                "channel_aware_astar_requested_backend_counts": {"channel_aware_astar": 6},
                "channel_aware_astar_selected_backend_counts": {"channel_aware_astar": 3, "astar": 3},
                "channel_aware_astar_status_counts": {"selected": 3, "fallback": 3},
                "channel_aware_astar_fallback_reason_counts": {"channel_search_failed:goal_blocked": 3},
                "channel_aware_astar_blocker_class_counts": {"selected": 3, "goal_blocked": 3},
                "channel_aware_astar_path_changed_count": 3,
                "channel_aware_astar_path_changed_rate": 0.5,
                "channel_aware_astar_path_cost_delta_count": 3,
                "channel_aware_astar_path_cost_delta_min": 2.0,
                "channel_aware_astar_path_cost_delta_max": 2.0,
                "channel_aware_astar_path_cost_delta_mean": 2.0,
                "channel_aware_astar_channel_cost_delta_count": 3,
                "channel_aware_astar_channel_cost_delta_min": -4.0,
                "channel_aware_astar_channel_cost_delta_max": -4.0,
                "channel_aware_astar_channel_cost_delta_mean": -4.0,
                "channel_aware_astar_high_cost_exposure_delta_count": 3,
                "channel_aware_astar_high_cost_exposure_delta_min": -3.0,
                "channel_aware_astar_high_cost_exposure_delta_max": -3.0,
                "channel_aware_astar_high_cost_exposure_delta_mean": -3.0,
            },
        )

        completed = self._run_stability("--batch-root", str(self.batch_root))

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        batch_summary = json.loads((self.batch_root / "batch-stability-summary.json").read_text(encoding="utf-8"))
        dataset_summary = json.loads(
            (self.batch_root / "dataset-quality-stability-summary.json").read_text(encoding="utf-8")
        )
        decision_summary = json.loads(
            (self.batch_root / "decision-stability-summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(batch_summary["schema_version"], "batch-stability-summary/v1")
        self.assertEqual(batch_summary["status"], "passed")
        self.assertEqual(batch_summary["run_count"], 2)
        self.assertEqual(batch_summary["by_run"]["all-all-k3"]["path_planning_failure_count"], 3)
        self.assertEqual(batch_summary["by_scenario_set"]["all"]["replan_count"], 3)
        self.assertEqual(batch_summary["by_diagnostic_profile"]["all"]["iris_fallback_count"], 3)
        self.assertEqual(batch_summary["by_top_k"]["3"]["region_graph_disconnected_count"], 3)
        self.assertEqual(batch_summary["by_scenario_group"]["stress"]["region_graph_fallback_count"], 3)
        self.assertEqual(batch_summary["by_run"]["all-all-k3"]["channel_aware_astar_report_count"], 6)
        self.assertEqual(batch_summary["by_scenario_set"]["all"]["channel_aware_astar_selected_count"], 3)
        self.assertEqual(batch_summary["by_scenario_group"]["stress"]["channel_aware_astar_fallback_count"], 3)
        self.assertEqual(batch_summary["batch_evaluation_counts"]["channel_aware_astar_fallback_count"], 3)
        channel_evidence = batch_summary["channel_aware_astar_evidence"]
        self.assertEqual(channel_evidence["channel_aware_astar_report_count"], 6)
        self.assertEqual(channel_evidence["channel_aware_astar_high_cost_exposure_delta_count"], 3)
        self.assertEqual(channel_evidence["channel_aware_astar_high_cost_exposure_delta_mean"], -3.0)
        self.assertEqual(batch_summary["source_paths"]["batch_run_index"], str(self.batch_root / "batch-run-index.json"))
        self.assertEqual(batch_summary["git_provenance"]["batch"]["parent"]["sha"], "1" * 40)

        self.assertEqual(dataset_summary["schema_version"], "dataset-quality-stability-summary/v1")
        self.assertEqual(dataset_summary["status"], "passed")
        self.assertEqual(dataset_summary["record_count"], 2)
        self.assertEqual(dataset_summary["by_action"]["keep"]["record_count"], 1)
        self.assertEqual(dataset_summary["by_action"]["downweight"]["record_count"], 1)
        self.assertEqual(
            dataset_summary["by_reason_code"]["region_graph_disconnected"]["action_counts"]["downweight"],
            1,
        )
        stress_bucket = dataset_summary["by_scenario_id"]["npz_near_blocked_corridor"]
        self.assertTrue(stress_bucket["stable_action"])
        self.assertIn("path_planning_failure", stress_bucket["stable_reason_codes"])

        self.assertEqual(decision_summary["schema_version"], "decision-stability-summary/v1")
        self.assertEqual(decision_summary["status"], "passed")
        stress_decision = decision_summary["by_scenario_id"]["npz_near_blocked_corridor"]
        self.assertTrue(stress_decision["stable_selection_changed"])
        self.assertTrue(stress_decision["stable_target_replacement_reason"])
        self.assertTrue(stress_decision["stable_failure_sources"])
        self.assertEqual(
            stress_decision["target_replacement_reason_counts"],
            {"no_feasible_candidate_after_path_feedback": 1},
        )

    def test_validation_failures_are_written_with_machine_reason_codes(self) -> None:
        open_grid_summary = self._summary(
            scenario_set="smoke",
            diagnostic_profile="baseline",
            top_k=1,
            scenario_id="npz_open_grid",
            scenario_group="smoke",
            open_grid=True,
        )
        mismatched_summary = self._summary(
            scenario_set="all",
            diagnostic_profile="all",
            top_k=3,
            scenario_id="npz_mismatch",
            scenario_group="stress",
        )
        self._write_batch(
            [
                {
                    "run_id": "open-grid-fail",
                    "scenario_set": "smoke",
                    "diagnostic_profile": "baseline",
                    "top_k": 1,
                    "summary": open_grid_summary,
                },
                {
                    "run_id": "failed-run",
                    "scenario_set": "stress",
                    "diagnostic_profile": "execution",
                    "top_k": 3,
                    "status": "failed",
                    "reason_codes": ["single_run_exit_nonzero"],
                    "summary": None,
                },
                {
                    "run_id": "metadata-mismatch",
                    "scenario_set": "stress",
                    "diagnostic_profile": "execution",
                    "top_k": 3,
                    "summary": mismatched_summary,
                },
            ],
            evaluation_override={"failed_count": 1, "failed_run_ids": ["failed-run"]},
        )

        completed = self._run_stability("--batch-root", str(self.batch_root))

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        batch_summary = json.loads((self.batch_root / "batch-stability-summary.json").read_text(encoding="utf-8"))
        dataset_summary = json.loads(
            (self.batch_root / "dataset-quality-stability-summary.json").read_text(encoding="utf-8")
        )
        decision_summary = json.loads(
            (self.batch_root / "decision-stability-summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(batch_summary["status"], "failed")
        by_run = batch_summary["by_run"]
        self.assertIn("open_grid_fallback_used", by_run["open-grid-fail"]["reason_codes"])
        self.assertIn("source_summary_missing", by_run["failed-run"]["reason_codes"])
        self.assertIn("batch_run_failed", by_run["failed-run"]["reason_codes"])
        self.assertIn("acceptance_metadata_mismatch", by_run["metadata-mismatch"]["reason_codes"])
        self.assertEqual(batch_summary["failure_reason_code_counts"]["batch_run_failed"], 1)
        self.assertEqual(dataset_summary["status"], "failed")
        self.assertEqual(decision_summary["status"], "failed")
        self.assertIn("open_grid_fallback", dataset_summary["by_action"]["exclude"]["reason_code_counts"])

    def test_schema_mismatch_is_reported_without_crashing(self) -> None:
        bad_summary = self._summary(
            scenario_set="smoke",
            diagnostic_profile="baseline",
            top_k=1,
            scenario_id="npz_bad_schema",
            scenario_group="smoke",
        )
        bad_summary["schema_version"] = "path-feedback-summary/v0"
        self._write_batch(
            [
                {
                    "run_id": "bad-schema",
                    "scenario_set": "smoke",
                    "diagnostic_profile": "baseline",
                    "top_k": 1,
                    "summary": bad_summary,
                }
            ]
        )

        completed = self._run_stability("--batch-root", str(self.batch_root))

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        batch_summary = json.loads((self.batch_root / "batch-stability-summary.json").read_text(encoding="utf-8"))
        self.assertIn("source_summary_schema_mismatch", batch_summary["by_run"]["bad-schema"]["reason_codes"])


class PathFeedbackSingleRunCompatibilityDuringStabilityTests(unittest.TestCase):
    def test_single_run_default_dry_run_behavior_remains_unchanged(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_path_feedback_validation.sh"

        completed = subprocess.run(
            ["bash", str(script), "--dry-run"],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Scenario set: smoke", completed.stdout)
        self.assertIn("Diagnostic profile: baseline", completed.stdout)
        self.assertIn("Top-K: 3", completed.stdout)


if __name__ == "__main__":
    unittest.main()
