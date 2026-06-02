import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class SampleQualityTrainingApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "run_sample_quality_training_application.sh"
        self.config = self.repo_root / "configs" / "sample_quality_training_application_v1.json"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="sample-quality-training-"))
        self.batch_root = self.temp_dir / "batch"
        self.batch_root.mkdir(parents=True)
        self.git_snapshot = self._current_git_snapshot()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _run_application(self, *args: str) -> subprocess.CompletedProcess[str]:
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

    def _write_batch(
        self,
        runs: list[dict],
        *,
        batch_status: str = "passed",
        stability_status: str = "passed",
        omit_sources: tuple[str, ...] = (),
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
                    "sample_quality_profile": run.get("sample_quality_profile", "fixture"),
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
            "source_summary_paths": source_summary_paths,
            "scenario_group_summary": {},
            "runs": [],
        }
        batch_stability = {
            "schema_version": "batch-stability-summary/v1",
            "generated_at": "2026-06-02T00:00:00Z",
            "status": stability_status,
            "reason_codes": [] if stability_status == "passed" else ["fixture_stability_failed"],
            "run_count": len(run_index_runs),
            "failure_reason_code_counts": {},
        }
        dataset_stability = {
            "schema_version": "dataset-quality-stability-summary/v1",
            "generated_at": "2026-06-02T00:00:00Z",
            "status": stability_status,
            "reason_codes": [] if stability_status == "passed" else ["fixture_dataset_stability_failed"],
            "record_count": len(source_summary_paths),
            "records": [],
        }
        decision_stability = {
            "schema_version": "decision-stability-summary/v1",
            "generated_at": "2026-06-02T00:00:00Z",
            "status": stability_status,
            "reason_codes": [] if stability_status == "passed" else ["fixture_decision_stability_failed"],
            "record_count": len(source_summary_paths),
            "records": [],
        }
        if batch_status == "failed":
            evaluation_summary["failed_count"] = max(1, evaluation_summary["failed_count"])
            evaluation_summary["failed_run_ids"] = failed_run_ids or [run_index_runs[0]["run_id"]]

        payloads = {
            "batch-run-index.json": run_index,
            "batch-evaluation-summary.json": evaluation_summary,
            "batch-stability-summary.json": batch_stability,
            "dataset-quality-stability-summary.json": dataset_stability,
            "decision-stability-summary.json": decision_stability,
        }
        for filename, payload in payloads.items():
            if filename in omit_sources:
                continue
            (self.batch_root / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")

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
        failure_sources: list[str] | None = None,
        open_grid: bool = False,
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
                    "path_planning_failure_count": failure_count,
                    "replan_count": replan_count,
                    "selection_changed_count": 1 if selection_changed else 0,
                    "iris_fallback_count": iris_fallback_count,
                    "region_graph_fallback_count": region_graph_fallback_count,
                    "region_graph_disconnected_count": region_graph_disconnected_count,
                }
            },
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
                        "candidates": [
                            {
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
                        ],
                    },
                    "diagnostic_interpretation": {
                        "target_replacement_reason": "no_feasible_candidate_after_path_feedback"
                        if failure_count
                        else "unchanged",
                        "failure_sources": failure_sources,
                        "open_grid_fallback_used": open_grid,
                    },
                    "iris_diagnostics": {"fallback_count": iris_fallback_count},
                    "region_graph_diagnostics": {
                        "fallback_count": region_graph_fallback_count,
                        "start_goal_disconnected_count": region_graph_disconnected_count,
                    },
                }
            ],
        }

    def test_config_validate_and_dry_run_do_not_write_application_outputs(self) -> None:
        self._write_batch(
            [
                {
                    "run_id": "smoke-baseline-k1",
                    "scenario_set": "smoke",
                    "diagnostic_profile": "baseline",
                    "top_k": 1,
                    "summary": self._summary(
                        scenario_set="smoke",
                        diagnostic_profile="baseline",
                        top_k=1,
                        scenario_id="npz_shadow_corridor",
                        scenario_group="smoke",
                    ),
                }
            ]
        )

        validate = self._run_application(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--validate-only",
        )
        dry_run = self._run_application(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--dry-run",
        )

        self.assertEqual(validate.returncode, 0, validate.stdout + validate.stderr)
        self.assertIn("config validated", validate.stdout)
        self.assertEqual(dry_run.returncode, 0, dry_run.stdout + dry_run.stderr)
        self.assertIn("legacy", dry_run.stdout)
        self.assertFalse((self.batch_root / "sample-quality-training-application-summary.json").exists())
        self.assertFalse((self.batch_root / "training-selection-stability-summary.json").exists())

    def test_application_consumes_sources_and_writes_profile_summaries(self) -> None:
        self._write_batch(
            [
                {
                    "run_id": "smoke-baseline-k1",
                    "scenario_set": "smoke",
                    "diagnostic_profile": "baseline",
                    "top_k": 1,
                    "sample_quality_profile": "legacy",
                    "summary": self._summary(
                        scenario_set="smoke",
                        diagnostic_profile="baseline",
                        top_k=1,
                        scenario_id="npz_shadow_corridor",
                        scenario_group="smoke",
                    ),
                },
                {
                    "run_id": "stress-execution-k3",
                    "scenario_set": "stress",
                    "diagnostic_profile": "execution",
                    "top_k": 3,
                    "sample_quality_profile": "soft_downweight_diagnostics",
                    "summary": self._summary(
                        scenario_set="stress",
                        diagnostic_profile="execution",
                        top_k=3,
                        scenario_id="npz_near_blocked_corridor",
                        scenario_group="stress",
                        failure_count=3,
                        replan_count=3,
                        iris_fallback_count=3,
                        region_graph_fallback_count=3,
                        region_graph_disconnected_count=3,
                        selection_changed=True,
                        failure_sources=[
                            "path_planning_failure",
                            "replan_required",
                            "iris_fallback",
                            "region_graph_fallback",
                            "region_graph_disconnected",
                        ],
                    ),
                },
            ]
        )

        completed = self._run_application("--batch-root", str(self.batch_root), "--config", str(self.config))

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        application = json.loads(
            (self.batch_root / "sample-quality-training-application-summary.json").read_text(encoding="utf-8")
        )
        selection = json.loads(
            (self.batch_root / "training-selection-stability-summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(application["schema_version"], "sample-quality-training-application-summary/v1")
        self.assertEqual(application["status"], "passed")
        self.assertEqual(application["source_summaries"]["batch_run_index"]["schema_version"], "path-feedback-batch-run-index/v1")
        self.assertEqual(len(application["source_summaries"]["path_feedback_summary_paths"]), 2)
        self.assertEqual(application["git_provenance"]["batch"]["parent"]["sha"], self.git_snapshot["parent"]["sha"])

        legacy = application["profile_results"]["legacy"]
        aware = application["profile_results"]["soft_downweight_diagnostics"]
        self.assertEqual(legacy["kept_sample_count"], 2)
        self.assertEqual(legacy["downweighted_sample_count"], 0)
        self.assertEqual(legacy["excluded_sample_count"], 0)
        self.assertEqual(aware["kept_sample_count"], 1)
        self.assertEqual(aware["downweighted_sample_count"], 1)
        self.assertEqual(aware["excluded_sample_count"], 0)
        self.assertEqual(aware["by_scenario_group"]["stress"]["action_counts"]["downweight"], 1)
        self.assertEqual(aware["by_reason_code"]["region_graph_disconnected"]["record_count"], 1)
        self.assertEqual(aware["records"][1]["sample_weight"], 0.5)

        self.assertEqual(selection["schema_version"], "training-selection-stability-summary/v1")
        self.assertEqual(selection["status"], "passed")
        self.assertEqual(selection["comparison"]["legacy_profile_id"], "legacy")
        self.assertEqual(selection["comparison"]["sample_quality_aware_profile_id"], "soft_downweight_diagnostics")
        self.assertIn("no_training_metric_evaluated", selection["comparison"]["reason_codes"])

    def test_open_grid_is_hard_excluded_and_failed_inputs_return_nonzero(self) -> None:
        self._write_batch(
            [
                {
                    "run_id": "open-grid-fail",
                    "scenario_set": "smoke",
                    "diagnostic_profile": "baseline",
                    "top_k": 1,
                    "status": "failed",
                    "reason_codes": ["open_grid_fallback_used"],
                    "summary": self._summary(
                        scenario_set="smoke",
                        diagnostic_profile="baseline",
                        top_k=1,
                        scenario_id="npz_open_grid",
                        scenario_group="smoke",
                        open_grid=True,
                    ),
                }
            ],
            batch_status="failed",
            stability_status="failed",
        )

        completed = self._run_application("--batch-root", str(self.batch_root), "--config", str(self.config))

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        application = json.loads(
            (self.batch_root / "sample-quality-training-application-summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(application["status"], "failed")
        self.assertIn("batch_run_failed", application["by_run"]["open-grid-fail"]["reason_codes"])
        self.assertIn("open_grid_fallback_used", application["failure_reason_code_counts"])
        aware = application["profile_results"]["soft_downweight_diagnostics"]
        self.assertEqual(aware["excluded_sample_count"], 1)
        self.assertEqual(aware["records"][0]["action"], "exclude")
        self.assertIn("open_grid_fallback", aware["records"][0]["reason_codes"])

    def test_validation_failures_cover_missing_schema_metadata_and_git_mismatch(self) -> None:
        bad_schema = self._summary(
            scenario_set="smoke",
            diagnostic_profile="baseline",
            top_k=1,
            scenario_id="npz_bad_schema",
            scenario_group="smoke",
        )
        bad_schema["schema_version"] = "path-feedback-summary/v0"
        metadata_mismatch = self._summary(
            scenario_set="all",
            diagnostic_profile="all",
            top_k=3,
            scenario_id="npz_mismatch",
            scenario_group="stress",
        )
        mismatched_git = dict(self.git_snapshot)
        mismatched_git["parent"] = dict(self.git_snapshot["parent"], sha="0" * 40)
        self._write_batch(
            [
                {
                    "run_id": "bad-schema",
                    "scenario_set": "smoke",
                    "diagnostic_profile": "baseline",
                    "top_k": 1,
                    "summary": bad_schema,
                },
                {
                    "run_id": "missing-summary",
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
                    "summary": metadata_mismatch,
                    "git": mismatched_git,
                },
            ],
            batch_status="failed",
        )

        completed = self._run_application("--batch-root", str(self.batch_root), "--config", str(self.config))

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        application = json.loads(
            (self.batch_root / "sample-quality-training-application-summary.json").read_text(encoding="utf-8")
        )
        self.assertIn("source_summary_schema_mismatch", application["by_run"]["bad-schema"]["reason_codes"])
        self.assertIn("source_summary_missing", application["by_run"]["missing-summary"]["reason_codes"])
        self.assertIn("acceptance_metadata_mismatch", application["by_run"]["metadata-mismatch"]["reason_codes"])
        self.assertIn("git_provenance_mismatch", application["by_run"]["metadata-mismatch"]["reason_codes"])

    def test_missing_required_source_jsons_are_reported(self) -> None:
        self._write_batch(
            [
                {
                    "run_id": "smoke-baseline-k1",
                    "scenario_set": "smoke",
                    "diagnostic_profile": "baseline",
                    "top_k": 1,
                    "summary": self._summary(
                        scenario_set="smoke",
                        diagnostic_profile="baseline",
                        top_k=1,
                        scenario_id="npz_shadow_corridor",
                        scenario_group="smoke",
                    ),
                }
            ],
            omit_sources=("decision-stability-summary.json",),
        )

        completed = self._run_application("--batch-root", str(self.batch_root), "--config", str(self.config))

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        application = json.loads(
            (self.batch_root / "sample-quality-training-application-summary.json").read_text(encoding="utf-8")
        )
        self.assertIn("decision_stability_summary_missing", application["reason_codes"])
        self.assertEqual(application["source_summaries"]["decision_stability_summary"]["status"], "missing")

    def test_application_does_not_mutate_stable_input_contract_files(self) -> None:
        self._write_batch(
            [
                {
                    "run_id": "smoke-baseline-k1",
                    "scenario_set": "smoke",
                    "diagnostic_profile": "baseline",
                    "top_k": 1,
                    "summary": self._summary(
                        scenario_set="smoke",
                        diagnostic_profile="baseline",
                        top_k=1,
                        scenario_id="npz_shadow_corridor",
                        scenario_group="smoke",
                    ),
                }
            ]
        )
        protected_files = [
            self.batch_root / "batch-run-index.json",
            self.batch_root / "batch-evaluation-summary.json",
            self.batch_root / "batch-stability-summary.json",
            self.batch_root / "dataset-quality-stability-summary.json",
            self.batch_root / "decision-stability-summary.json",
            self.batch_root / "smoke-baseline-k1" / "path-feedback-summary.json",
        ]
        before = {path: path.read_text(encoding="utf-8") for path in protected_files}

        completed = self._run_application("--batch-root", str(self.batch_root), "--config", str(self.config))

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        after = {path: path.read_text(encoding="utf-8") for path in protected_files}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
