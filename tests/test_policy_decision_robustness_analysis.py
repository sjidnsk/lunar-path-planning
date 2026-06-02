import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class PolicyDecisionRobustnessAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "run_policy_decision_robustness_analysis.sh"
        self.config = self.repo_root / "configs" / "policy_decision_robustness_v1.json"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="policy-decision-robustness-"))
        self.batch_root = self.temp_dir / "batch"
        self.batch_root.mkdir(parents=True)
        self.git_snapshot = self._current_git_snapshot()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _run_analysis(self, *args: str) -> subprocess.CompletedProcess[str]:
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
        application_status: str = "passed",
        omit_sources: tuple[str, ...] = (),
    ) -> None:
        run_index_runs = []
        source_summary_paths = []
        sample_quality_records = []
        for run in runs:
            run_root = self.batch_root / run["run_id"]
            run_root.mkdir(parents=True, exist_ok=True)
            summary_path = run_root / "path-feedback-summary.json"
            report_path = run_root / "path-feedback-summary.md"
            summary = run.get("summary")
            if summary is not None:
                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                source_summary_paths.append(str(summary_path))
                sample_quality_records.extend(self._sample_quality_records(run, summary, str(summary_path)))
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
        failed_count = len(failed_run_ids)
        if batch_status == "failed":
            failed_count = max(1, failed_count)
        evaluation_summary = {
            "schema_version": "path-feedback-batch-evaluation-summary/v1",
            "generated_at": "2026-06-02T00:00:00Z",
            "matrix_path": "configs/path_feedback_batch_dataset_v1.json",
            "output_root": str(self.batch_root),
            "run_count": len(run_index_runs),
            "passed_count": len(run_index_runs) - failed_count,
            "failed_count": failed_count,
            "failed_run_ids": failed_run_ids or (["fixture-failed"] if batch_status == "failed" else []),
            "failure_reason_code_counts": {},
            "open_grid_fallback_used_count": sum(
                1 for run in runs if isinstance(run.get("summary"), dict) and run["summary"].get("open_grid_fallback_used")
            ),
            "source_summary_paths": source_summary_paths,
            "scenario_group_summary": {},
            "runs": [],
        }
        batch_stability = {
            "schema_version": "batch-stability-summary/v1",
            "generated_at": "2026-06-02T00:00:00Z",
            "status": stability_status,
            "reason_codes": [] if stability_status == "passed" else ["fixture_stability_failed"],
            "failure_reason_code_counts": {},
            "run_count": len(run_index_runs),
        }
        dataset_stability = {
            "schema_version": "dataset-quality-stability-summary/v1",
            "generated_at": "2026-06-02T00:00:00Z",
            "status": stability_status,
            "reason_codes": [] if stability_status == "passed" else ["fixture_dataset_stability_failed"],
            "record_count": len(sample_quality_records),
            "records": sample_quality_records,
        }
        decision_stability = {
            "schema_version": "decision-stability-summary/v1",
            "generated_at": "2026-06-02T00:00:00Z",
            "status": stability_status,
            "reason_codes": [] if stability_status == "passed" else ["fixture_decision_stability_failed"],
            "record_count": len(sample_quality_records),
            "records": [],
        }
        application = {
            "schema_version": "sample-quality-training-application-summary/v1",
            "generated_at": "2026-06-02T00:00:00Z",
            "status": application_status,
            "reason_codes": [] if application_status == "passed" else ["fixture_application_failed"],
            "failure_reason_code_counts": {},
            "batch_root": str(self.batch_root),
            "git_provenance": {
                "batch": self.git_snapshot,
                "current": self.git_snapshot,
                "runs_match_batch": True,
                "current_matches_batch": True,
            },
            "profile_results": {
                "legacy": {"records": [dict(record, action="keep", sample_weight=1.0) for record in sample_quality_records]},
                "soft_downweight_diagnostics": {"records": sample_quality_records},
            },
        }
        training_selection = {
            "schema_version": "training-selection-stability-summary/v1",
            "generated_at": "2026-06-02T00:00:00Z",
            "status": application_status,
            "reason_codes": [] if application_status == "passed" else ["fixture_selection_failed"],
            "comparison": {
                "reason_codes": ["no_training_metric_evaluated"],
                "selection_changed": False,
            },
        }
        payloads = {
            "batch-run-index.json": run_index,
            "batch-evaluation-summary.json": evaluation_summary,
            "batch-stability-summary.json": batch_stability,
            "dataset-quality-stability-summary.json": dataset_stability,
            "decision-stability-summary.json": decision_stability,
            "sample-quality-training-application-summary.json": application,
            "training-selection-stability-summary.json": training_selection,
        }
        for filename, payload in payloads.items():
            if filename in omit_sources:
                continue
            (self.batch_root / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _sample_quality_records(self, run: dict, summary: dict, source_summary_path: str) -> list[dict]:
        records = []
        command_args = {
            "scenario_set": run["scenario_set"],
            "diagnostic_profile": run["diagnostic_profile"],
            "top_k": run["top_k"],
        }
        for scenario in summary.get("scenarios", []):
            reason_codes = self._scenario_reason_codes(scenario)
            action = "exclude" if "open_grid_fallback" in reason_codes else "downweight" if len(reason_codes) > 1 else "keep"
            sample_weight = 0.0 if action == "exclude" else 0.5 if action == "downweight" else 1.0
            records.append(
                {
                    "run_id": run["run_id"],
                    "scenario_id": scenario["scenario_id"],
                    "scenario_group": scenario["scenario_group"],
                    "scenario_set": command_args["scenario_set"],
                    "diagnostic_profile": command_args["diagnostic_profile"],
                    "top_k": command_args["top_k"],
                    "source_summary_path": source_summary_path,
                    "reason_codes": reason_codes,
                    "action": action,
                    "decision": action,
                    "sample_weight": sample_weight,
                }
            )
        return records

    def _scenario_reason_codes(self, scenario: dict) -> list[str]:
        codes = []
        feedback = scenario.get("path_feedback", {})
        for candidate in feedback.get("candidates", []):
            if candidate.get("failure_reason") or candidate.get("reachable") is False:
                codes.append("path_planning_failure")
            if candidate.get("replan_required"):
                codes.append("replan_required")
            interpretation = candidate.get("diagnostic_interpretation", {})
            if interpretation.get("open_grid_fallback_used"):
                codes.append("open_grid_fallback")
            if interpretation.get("iris_fallback_used"):
                codes.append("iris_fallback")
            if interpretation.get("region_graph_fallback_used"):
                codes.append("region_graph_fallback")
            if interpretation.get("region_graph_start_goal_connected") is False:
                codes.append("region_graph_disconnected")
        if scenario.get("open_grid_fallback_used"):
            codes.append("open_grid_fallback")
        result = []
        for code in codes or ["sample_quality_passed"]:
            if code not in result:
                result.append(code)
        return result

    def _summary(
        self,
        *,
        scenario_set: str,
        diagnostic_profile: str,
        top_k: int,
        scenario_id: str,
        scenario_group: str,
        candidates: list[dict],
        selected_before: list[int],
        selected_after: list[int],
        open_grid: bool = False,
    ) -> dict:
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
        failure_count = sum(1 for candidate in candidates if candidate.get("failure_reason") or candidate.get("reachable") is False)
        replan_count = sum(1 for candidate in candidates if candidate.get("replan_required"))
        iris_fallback_count = sum(1 for candidate in candidates if candidate.get("diagnostic_interpretation", {}).get("iris_fallback_used"))
        region_graph_fallback_count = sum(
            1 for candidate in candidates if candidate.get("diagnostic_interpretation", {}).get("region_graph_fallback_used")
        )
        region_graph_disconnected_count = sum(
            1
            for candidate in candidates
            if candidate.get("diagnostic_interpretation", {}).get("region_graph_start_goal_connected") is False
        )
        return {
            "schema_version": "path-feedback-summary/v1",
            "scenario_set": scenario_set,
            "diagnostic_profile": diagnostic_profile,
            "acceptance_gate": acceptance_metadata["acceptance_gate"],
            "top_k": top_k,
            "scenario_count": 1,
            "candidate_count": len(candidates),
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
                    "candidate_count": len(candidates),
                    "path_planning_failure_count": failure_count,
                    "replan_count": replan_count,
                    "iris_fallback_count": iris_fallback_count,
                    "region_graph_fallback_count": region_graph_fallback_count,
                    "region_graph_disconnected_count": region_graph_disconnected_count,
                }
            },
            "scenarios": [
                {
                    "scenario_id": scenario_id,
                    "scenario_group": scenario_group,
                    "selected_cell_before_path_feedback": selected_before,
                    "selected_cell_after_path_feedback": selected_after,
                    "selection_changed_by_path_feedback": selected_before != selected_after,
                    "open_grid_fallback_used": open_grid,
                    "path_feedback": {
                        "candidate_count": len(candidates),
                        "reachable_count": len(candidates) - failure_count,
                        "failure_count": failure_count,
                        "replan_count": replan_count,
                        "failure_reasons": ["goal_blocked"] if failure_count else [],
                        "candidates": candidates,
                    },
                    "diagnostic_interpretation": {
                        "target_replacement_reason": "no_feasible_candidate_after_path_feedback"
                        if failure_count == len(candidates)
                        else "unchanged",
                        "failure_sources": self._scenario_reason_codes({"path_feedback": {"candidates": candidates}}),
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

    def _candidate(
        self,
        action_index: int,
        cell: list[int],
        *,
        utility: float,
        path_cost: float,
        risk: float = 0.0,
        reachable: bool = True,
        failure_reason: str | None = None,
        replan_required: bool = False,
        iris_fallback: bool = False,
        region_graph_fallback: bool = False,
        region_graph_connected: bool = True,
        open_grid: bool = False,
    ) -> dict:
        return {
            "action_index": action_index,
            "cell": cell,
            "utility": utility,
            "path_cost": path_cost,
            "risk": risk,
            "reachable": reachable,
            "failure_reason": failure_reason,
            "replan_required": replan_required,
            "diagnostic_interpretation": {
                "diagnostic_flags": [],
                "open_grid_fallback_used": open_grid,
                "iris_fallback_used": iris_fallback,
                "region_graph_fallback_used": region_graph_fallback,
                "region_graph_start_goal_connected": region_graph_connected,
            },
        }

    def test_config_validate_and_dry_run_do_not_write_outputs(self) -> None:
        self._write_batch(
            [
                {
                    "run_id": "smoke-baseline-k1",
                    "scenario_set": "smoke",
                    "diagnostic_profile": "baseline",
                    "top_k": 2,
                    "summary": self._summary(
                        scenario_set="smoke",
                        diagnostic_profile="baseline",
                        top_k=2,
                        scenario_id="npz_shadow_corridor",
                        scenario_group="smoke",
                        selected_before=[2, 2],
                        selected_after=[2, 2],
                        candidates=[
                            self._candidate(0, [2, 2], utility=0.9, path_cost=12.0),
                            self._candidate(1, [3, 2], utility=0.7, path_cost=10.0),
                        ],
                    ),
                }
            ]
        )

        validate = self._run_analysis("--batch-root", str(self.batch_root), "--config", str(self.config), "--validate-only")
        dry_run = self._run_analysis("--batch-root", str(self.batch_root), "--config", str(self.config), "--dry-run")

        self.assertEqual(validate.returncode, 0, validate.stdout + validate.stderr)
        self.assertIn("config validated", validate.stdout)
        self.assertEqual(dry_run.returncode, 0, dry_run.stdout + dry_run.stderr)
        self.assertIn("feedback_aware", dry_run.stdout)
        self.assertFalse((self.batch_root / "policy-decision-robustness-summary.json").exists())
        self.assertFalse((self.batch_root / "policy-decision-selection-comparison-summary.json").exists())

    def test_wrapper_uses_default_config_when_config_is_omitted(self) -> None:
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
                        selected_before=[2, 2],
                        selected_after=[2, 2],
                        candidates=[self._candidate(0, [2, 2], utility=0.9, path_cost=12.0)],
                    ),
                }
            ]
        )

        completed = self._run_analysis("--batch-root", str(self.batch_root), "--validate-only")

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("config validated", completed.stdout)
        self.assertIn('"config": "configs/policy_decision_robustness_v1.json"', completed.stdout)

    def test_analysis_consumes_sources_and_writes_profile_decision_summaries(self) -> None:
        self._write_batch(
            [
                {
                    "run_id": "all-all-k3",
                    "scenario_set": "all",
                    "diagnostic_profile": "all",
                    "top_k": 3,
                    "summary": self._summary(
                        scenario_set="all",
                        diagnostic_profile="all",
                        top_k=3,
                        scenario_id="npz_near_blocked_corridor",
                        scenario_group="stress",
                        selected_before=[2, 2],
                        selected_after=[1, 1],
                        candidates=[
                            self._candidate(
                                0,
                                [2, 2],
                                utility=0.95,
                                path_cost=250.0,
                                risk=0.8,
                                reachable=False,
                                failure_reason="goal_blocked",
                                replan_required=True,
                                iris_fallback=True,
                                region_graph_fallback=True,
                                region_graph_connected=False,
                            ),
                            self._candidate(1, [1, 1], utility=0.75, path_cost=14.0, risk=0.1),
                            self._candidate(2, [3, 1], utility=0.6, path_cost=40.0, risk=0.2),
                        ],
                    ),
                }
            ]
        )

        completed = self._run_analysis("--batch-root", str(self.batch_root), "--config", str(self.config))

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        robustness = json.loads((self.batch_root / "policy-decision-robustness-summary.json").read_text(encoding="utf-8"))
        comparison = json.loads(
            (self.batch_root / "policy-decision-selection-comparison-summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(robustness["schema_version"], "policy-decision-robustness-summary/v1")
        self.assertEqual(robustness["status"], "passed")
        self.assertEqual(robustness["source_summaries"]["sample_quality_training_application_summary"]["schema_version"], "sample-quality-training-application-summary/v1")
        self.assertEqual(robustness["git_provenance"]["batch"]["parent"]["sha"], self.git_snapshot["parent"]["sha"])
        self.assertEqual(set(robustness["profiles"]), {"legacy", "feedback_aware", "sample_quality_aware"})

        legacy_decision = robustness["profiles"]["legacy"]["decisions"][0]
        feedback_decision = robustness["profiles"]["feedback_aware"]["decisions"][0]
        sample_quality_decision = robustness["profiles"]["sample_quality_aware"]["decisions"][0]
        self.assertEqual(legacy_decision["selected_action_after_profile"], 0)
        self.assertEqual(feedback_decision["selected_action_after_profile"], 1)
        self.assertEqual(sample_quality_decision["selected_action_after_profile"], 1)
        self.assertTrue(feedback_decision["selection_changed_by_profile"])

        penalized = {
            candidate["action_index"]: candidate
            for candidate in feedback_decision["candidate_comparisons"]
        }[0]
        self.assertGreater(penalized["rank_delta"], 0)
        self.assertIn("path_planning_failure", penalized["reason_codes"])
        self.assertIn("replan_required", penalized["reason_codes"])
        self.assertIn("high_path_cost", penalized["reason_codes"])
        self.assertIn("iris_fallback", penalized["reason_codes"])
        self.assertIn("region_graph_fallback", penalized["reason_codes"])
        self.assertIn("region_graph_disconnected", penalized["reason_codes"])
        self.assertGreater(penalized["penalty_components"]["path_feedback_penalty"], 0)

        self.assertEqual(robustness["profiles"]["feedback_aware"]["by_scenario_group"]["stress"]["selection_changed_count"], 1)
        self.assertEqual(robustness["profiles"]["feedback_aware"]["reason_code_counts"]["path_planning_failure"], 1)
        self.assertEqual(comparison["schema_version"], "policy-decision-selection-comparison-summary/v1")
        self.assertIn("no_training_metric_evaluated", comparison["comparison"]["reason_codes"])
        self.assertTrue(comparison["by_scenario_id"]["npz_near_blocked_corridor"]["selection_changed_between_profiles"])

    def test_open_grid_failed_inputs_return_nonzero_and_machine_reason_codes(self) -> None:
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
                        selected_before=[1, 1],
                        selected_after=[1, 1],
                        open_grid=True,
                        candidates=[
                            self._candidate(0, [1, 1], utility=0.5, path_cost=12.0, open_grid=True),
                        ],
                    ),
                }
            ],
            batch_status="failed",
            stability_status="failed",
            application_status="failed",
        )

        completed = self._run_analysis("--batch-root", str(self.batch_root), "--config", str(self.config))

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        robustness = json.loads((self.batch_root / "policy-decision-robustness-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(robustness["status"], "failed")
        self.assertIn("batch_evaluation_failed", robustness["reason_codes"])
        self.assertIn("batch_run_failed", robustness["by_run"]["open-grid-fail"]["reason_codes"])
        self.assertIn("open_grid_fallback_used", robustness["failure_reason_code_counts"])
        candidate = robustness["profiles"]["sample_quality_aware"]["decisions"][0]["candidate_comparisons"][0]
        self.assertIn("open_grid_fallback", candidate["reason_codes"])
        self.assertGreater(candidate["penalty_components"]["sample_quality_penalty"], 0)

    def test_selection_comparison_aggregates_repeated_scenario_ids_without_overcounting(self) -> None:
        candidates = [
            self._candidate(0, [2, 2], utility=0.9, path_cost=180.0, reachable=False, failure_reason="goal_blocked"),
            self._candidate(1, [1, 1], utility=0.8, path_cost=12.0),
        ]
        self._write_batch(
            [
                {
                    "run_id": "all-iris-k2",
                    "scenario_set": "all",
                    "diagnostic_profile": "iris",
                    "top_k": 2,
                    "summary": self._summary(
                        scenario_set="all",
                        diagnostic_profile="iris",
                        top_k=2,
                        scenario_id="npz_shadow_corridor",
                        scenario_group="smoke",
                        selected_before=[2, 2],
                        selected_after=[1, 1],
                        candidates=candidates,
                    ),
                },
                {
                    "run_id": "all-all-k2",
                    "scenario_set": "all",
                    "diagnostic_profile": "all",
                    "top_k": 2,
                    "summary": self._summary(
                        scenario_set="all",
                        diagnostic_profile="all",
                        top_k=2,
                        scenario_id="npz_shadow_corridor",
                        scenario_group="smoke",
                        selected_before=[2, 2],
                        selected_after=[1, 1],
                        candidates=candidates,
                    ),
                },
            ]
        )

        completed = self._run_analysis("--batch-root", str(self.batch_root), "--config", str(self.config))

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        comparison = json.loads(
            (self.batch_root / "policy-decision-selection-comparison-summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(comparison["comparison"]["scenario_count"], 1)
        self.assertEqual(comparison["comparison"]["selection_changed_scenario_count"], 1)
        self.assertEqual(comparison["comparison"]["selection_stable_scenario_count"], 0)
        scenario = comparison["by_scenario_id"]["npz_shadow_corridor"]
        self.assertEqual(scenario["observation_count"], 2)
        self.assertTrue(scenario["selection_changed_between_profiles"])

    def test_validation_failures_cover_missing_schema_metadata_provenance_and_failed_sources(self) -> None:
        bad_schema = self._summary(
            scenario_set="smoke",
            diagnostic_profile="baseline",
            top_k=1,
            scenario_id="npz_bad_schema",
            scenario_group="smoke",
            selected_before=[1, 1],
            selected_after=[1, 1],
            candidates=[self._candidate(0, [1, 1], utility=0.5, path_cost=12.0)],
        )
        bad_schema["schema_version"] = "path-feedback-summary/v0"
        metadata_mismatch = self._summary(
            scenario_set="all",
            diagnostic_profile="all",
            top_k=3,
            scenario_id="npz_mismatch",
            scenario_group="stress",
            selected_before=[1, 1],
            selected_after=[1, 1],
            candidates=[self._candidate(0, [1, 1], utility=0.5, path_cost=12.0)],
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
            stability_status="failed",
            application_status="failed",
            omit_sources=("training-selection-stability-summary.json",),
        )

        completed = self._run_analysis("--batch-root", str(self.batch_root), "--config", str(self.config))

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        robustness = json.loads((self.batch_root / "policy-decision-robustness-summary.json").read_text(encoding="utf-8"))
        self.assertIn("training_selection_stability_summary_missing", robustness["reason_codes"])
        self.assertIn("batch_stability_summary_failed", robustness["reason_codes"])
        self.assertIn("sample_quality_training_application_summary_failed", robustness["reason_codes"])
        self.assertIn("source_summary_schema_mismatch", robustness["by_run"]["bad-schema"]["reason_codes"])
        self.assertIn("source_summary_missing", robustness["by_run"]["missing-summary"]["reason_codes"])
        self.assertIn("acceptance_metadata_mismatch", robustness["by_run"]["metadata-mismatch"]["reason_codes"])
        self.assertIn("git_provenance_mismatch", robustness["by_run"]["metadata-mismatch"]["reason_codes"])

    def test_analysis_does_not_mutate_stable_input_contract_files(self) -> None:
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
                        selected_before=[1, 1],
                        selected_after=[1, 1],
                        candidates=[self._candidate(0, [1, 1], utility=0.5, path_cost=12.0)],
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
            self.batch_root / "sample-quality-training-application-summary.json",
            self.batch_root / "training-selection-stability-summary.json",
            self.batch_root / "smoke-baseline-k1" / "path-feedback-summary.json",
        ]
        before = {path: path.read_text(encoding="utf-8") for path in protected_files}

        completed = self._run_analysis("--batch-root", str(self.batch_root), "--config", str(self.config))

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        after = {path: path.read_text(encoding="utf-8") for path in protected_files}
        self.assertEqual(before, after)


class PolicyDecisionRobustnessCompatibilityTests(unittest.TestCase):
    def test_existing_cli_default_behaviors_remain_unchanged(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        single = subprocess.run(
            ["bash", str(repo_root / "scripts" / "run_path_feedback_validation.sh"), "--dry-run"],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        batch = subprocess.run(
            [
                "bash",
                str(repo_root / "scripts" / "run_batch_path_feedback_validation.sh"),
                "--matrix",
                str(repo_root / "configs" / "path_feedback_batch_dataset_v1.json"),
                "--validate-only",
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(single.returncode, 0, single.stdout + single.stderr)
        self.assertIn("Scenario set: smoke", single.stdout)
        self.assertEqual(batch.returncode, 0, batch.stdout + batch.stderr)
        self.assertIn("matrix validated", batch.stdout)


if __name__ == "__main__":
    unittest.main()
