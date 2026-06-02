import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


class BatchPathFeedbackValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "run_batch_path_feedback_validation.sh"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="path-feedback-batch-"))
        self.fake_single_run_log = self.temp_dir / "single-run.log"
        self.fake_single_run = self._write_fake_single_run_script(self.temp_dir / "fake_single_run.sh")

    def _write_matrix(self, payload: dict) -> Path:
        path = self.temp_dir / "matrix.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _run_batch(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["FAKE_SINGLE_RUN_LOG"] = str(self.fake_single_run_log)
        return subprocess.run(
            ["bash", str(self.script), *args],
            cwd=self.repo_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _write_fake_single_run_script(self, path: Path) -> Path:
        path.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env bash
                set -euo pipefail

                python_bin="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"
                output_root=""
                scenario_set=""
                diagnostic_profile=""
                top_k=""
                while [[ $# -gt 0 ]]; do
                  case "$1" in
                    --output-root)
                      output_root="$2"
                      shift 2
                      ;;
                    --scenario-set)
                      scenario_set="$2"
                      shift 2
                      ;;
                    --diagnostic-profile)
                      diagnostic_profile="$2"
                      shift 2
                      ;;
                    --top-k)
                      top_k="$2"
                      shift 2
                      ;;
                    --simulate-tracking|--optimize-trajectory|--drake-iris-regions)
                      shift
                      ;;
                    *)
                      echo "unexpected argument: $1" >&2
                      exit 2
                      ;;
                  esac
                done

                if [[ -z "$output_root" || -z "$scenario_set" || -z "$diagnostic_profile" || -z "$top_k" ]]; then
                  echo "missing required fake single-run argument" >&2
                  exit 2
                fi

                run_id="$(basename "$output_root")"
                if [[ -n "${FAKE_SINGLE_RUN_LOG:-}" ]]; then
                  printf '%s|%s|%s|%s|%s\\n' "$run_id" "$scenario_set" "$diagnostic_profile" "$top_k" "$output_root" >> "$FAKE_SINGLE_RUN_LOG"
                fi

                mkdir -p "$output_root/maps" "$output_root/path_planner_sidecars"
                if [[ "$run_id" == "exit-fail" ]]; then
                  exit 7
                fi

                "$python_bin" - "$output_root" "$scenario_set" "$diagnostic_profile" "$top_k" <<'PY'
                import json
                import sys
                from pathlib import Path

                output_root = Path(sys.argv[1])
                scenario_set = sys.argv[2]
                diagnostic_profile = sys.argv[3]
                top_k = int(sys.argv[4])
                run_id = output_root.name
                open_grid = run_id == "open-grid-fail"

                def group_payload(group):
                    failure_count = top_k if group == "stress" else 0
                    return {
                        "scenario_count": 1,
                        "candidate_count": top_k,
                        "reachable_count": 0 if group == "stress" else top_k,
                        "failure_count": failure_count,
                        "replan_count": failure_count,
                        "selection_changed_count": 1,
                        "iris_report_count": top_k,
                        "iris_fallback_count": top_k,
                        "region_graph_fallback_count": top_k,
                        "region_graph_start_goal_disconnected_count": failure_count,
                    }

                if scenario_set == "all":
                    scenario_groups = {
                        "smoke": group_payload("smoke"),
                        "stress": group_payload("stress"),
                    }
                else:
                    scenario_groups = {scenario_set: group_payload(scenario_set)}

                path_failures = sum(item["failure_count"] for item in scenario_groups.values())
                replans = sum(item["replan_count"] for item in scenario_groups.values())
                iris_fallback = sum(item["iris_fallback_count"] for item in scenario_groups.values())
                region_fallback = sum(item["region_graph_fallback_count"] for item in scenario_groups.values())
                region_disconnect = sum(
                    item["region_graph_start_goal_disconnected_count"]
                    for item in scenario_groups.values()
                )
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
                    "acceptance_gate": "semi-real-closed-loop" if scenario_set == "all" and diagnostic_profile == "all" and top_k == 3 else "custom",
                    "top_k": top_k,
                    "python_executable": sys.executable,
                    "planner_extra_args": [],
                    "open_grid_fallback_used": open_grid,
                    "open_grid_fallback_used_gate": gate,
                }
                summary = {
                    "schema_version": "path-feedback-summary/v1",
                    "scenario_count": sum(item["scenario_count"] for item in scenario_groups.values()),
                    "scenario_set": scenario_set,
                    "diagnostic_profile": diagnostic_profile,
                    "acceptance_gate": acceptance_metadata["acceptance_gate"],
                    "top_k": top_k,
                    "candidate_count": sum(item["candidate_count"] for item in scenario_groups.values()),
                    "path_planning_failure_count": path_failures,
                    "replan_count": replans,
                    "iris_fallback_count": iris_fallback,
                    "region_graph_fallback_count": region_fallback,
                    "region_graph_start_goal_disconnected_count": region_disconnect,
                    "region_graph_disconnected_count": region_disconnect,
                    "open_grid_fallback_used": open_grid,
                    "open_grid_fallback_used_gate": gate,
                    "acceptance_metadata": acceptance_metadata,
                    "scenario_group_summary": scenario_groups,
                    "scenarios": [],
                }
                manifest = {
                    "schema_version": "path-feedback-manifest/v1",
                    "scenario_set": scenario_set,
                    "diagnostic_profile": diagnostic_profile,
                    "top_k": top_k,
                    "outputs": {
                        "summary": str(output_root / "path-feedback-summary.json"),
                        "report": str(output_root / "path-feedback-summary.md"),
                    },
                }
                (output_root / "path-feedback-summary.json").write_text(
                    json.dumps(summary, indent=2),
                    encoding="utf-8",
                )
                (output_root / "path-feedback-manifest.json").write_text(
                    json.dumps(manifest, indent=2),
                    encoding="utf-8",
                )
                (output_root / "path-feedback-summary.md").write_text(
                    f"# fake report for {run_id}\\n",
                    encoding="utf-8",
                )
                PY
                """
            ),
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path

    def test_matrix_validate_and_dry_run_do_not_write_outputs(self) -> None:
        output_root = self.temp_dir / "batch"
        matrix = self._write_matrix(
            {
                "schema_version": "path-feedback-batch-matrix/v1",
                "output_root": str(output_root),
                "defaults": {
                    "planner_extra_args": ["--planning-backend", "region_graph_guided"],
                },
                "runs": [
                    {
                        "run_id": "smoke-baseline-k1",
                        "scenario_set": "smoke",
                        "diagnostic_profile": "baseline",
                        "top_k": 1,
                        "sample_quality_profile": "audit-only",
                    }
                ],
            }
        )

        validate = self._run_batch("--matrix", str(matrix), "--validate-only")

        self.assertEqual(validate.returncode, 0, validate.stdout + validate.stderr)
        self.assertIn("matrix validated", validate.stdout)
        self.assertFalse(output_root.exists())

        dry_run = self._run_batch("--matrix", str(matrix), "--dry-run")

        self.assertEqual(dry_run.returncode, 0, dry_run.stdout + dry_run.stderr)
        self.assertIn("[DRY RUN]", dry_run.stdout)
        self.assertIn("smoke-baseline-k1", dry_run.stdout)
        self.assertIn("--planning-backend region_graph_guided", dry_run.stdout)
        self.assertFalse(output_root.exists())

    def test_invalid_matrix_is_rejected_before_execution(self) -> None:
        matrix = self._write_matrix(
            {
                "schema_version": "path-feedback-batch-matrix/v1",
                "runs": [
                    {
                        "run_id": "bad-top-k",
                        "scenario_set": "smoke",
                        "diagnostic_profile": "baseline",
                        "top_k": 0,
                    }
                ],
            }
        )

        completed = self._run_batch("--matrix", str(matrix), "--validate-only")

        self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
        self.assertIn("top_k must be a positive integer", completed.stderr)
        self.assertFalse(self.fake_single_run_log.exists())

    def test_batch_calls_single_script_with_independent_output_roots_and_writes_index(self) -> None:
        output_root = self.temp_dir / "batch"
        matrix = self._write_matrix(
            {
                "schema_version": "path-feedback-batch-matrix/v1",
                "output_root": "outputs/ignored-by-cli",
                "runs": [
                    {
                        "run_id": "smoke-baseline-k1",
                        "scenario_set": "smoke",
                        "diagnostic_profile": "baseline",
                        "top_k": 1,
                        "sample_quality_profile": "audit-only",
                    },
                    {
                        "run_id": "all-all-k3",
                        "scenario_set": "all",
                        "diagnostic_profile": "all",
                        "top_k": 3,
                    },
                ],
            }
        )

        completed = self._run_batch(
            "--matrix",
            str(matrix),
            "--output-root",
            str(output_root),
            "--single-run-script",
            str(self.fake_single_run),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertTrue((output_root / "smoke-baseline-k1" / "path-feedback-summary.json").is_file())
        self.assertTrue((output_root / "all-all-k3" / "path-feedback-summary.json").is_file())
        log_lines = self.fake_single_run_log.read_text(encoding="utf-8").splitlines()
        self.assertIn(f"smoke-baseline-k1|smoke|baseline|1|{output_root / 'smoke-baseline-k1'}", log_lines)
        self.assertIn(f"all-all-k3|all|all|3|{output_root / 'all-all-k3'}", log_lines)

        run_index = json.loads((output_root / "batch-run-index.json").read_text(encoding="utf-8"))
        self.assertEqual(run_index["schema_version"], "path-feedback-batch-run-index/v1")
        self.assertEqual(run_index["run_count"], 2)
        first_run = run_index["runs"][0]
        self.assertEqual(first_run["status"], "passed")
        self.assertEqual(first_run["reason_codes"], [])
        self.assertEqual(first_run["command_args"]["scenario_set"], "smoke")
        self.assertEqual(first_run["command_args"]["diagnostic_profile"], "baseline")
        self.assertEqual(first_run["command_args"]["top_k"], 1)
        self.assertEqual(first_run["command_args"]["python_executable"], sys.executable)
        self.assertEqual(first_run["sample_quality_profile"], "audit-only")
        self.assertTrue(first_run["source_paths"]["summary"].endswith("smoke-baseline-k1/path-feedback-summary.json"))
        self.assertTrue(first_run["source_paths"]["report"].endswith("smoke-baseline-k1/path-feedback-summary.md"))
        self.assertEqual(first_run["acceptance_metadata"]["scenario_set"], "smoke")
        self.assertRegex(first_run["git"]["parent"]["sha"], r"^[0-9a-f]{40}$")
        self.assertEqual(
            set(first_run["git"]["submodules"]),
            {"dev-platform-constraints", "model-explorer", "path-planner"},
        )
        for submodule in first_run["git"]["submodules"].values():
            self.assertRegex(submodule["sha"], r"^[0-9a-f]{40}$")

    def test_batch_evaluation_summary_aggregates_run_and_scenario_metrics(self) -> None:
        output_root = self.temp_dir / "batch"
        matrix = self._write_matrix(
            {
                "schema_version": "path-feedback-batch-matrix/v1",
                "output_root": str(output_root),
                "runs": [
                    {
                        "run_id": "smoke-baseline-k1",
                        "scenario_set": "smoke",
                        "diagnostic_profile": "baseline",
                        "top_k": 1,
                    },
                    {
                        "run_id": "all-all-k3",
                        "scenario_set": "all",
                        "diagnostic_profile": "all",
                        "top_k": 3,
                    },
                ],
            }
        )

        completed = self._run_batch(
            "--matrix",
            str(matrix),
            "--single-run-script",
            str(self.fake_single_run),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads((output_root / "batch-evaluation-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["schema_version"], "path-feedback-batch-evaluation-summary/v1")
        self.assertEqual(summary["run_count"], 2)
        self.assertEqual(summary["passed_count"], 2)
        self.assertEqual(summary["failed_count"], 0)
        self.assertEqual(summary["open_grid_fallback_used_count"], 0)
        self.assertEqual(summary["path_planning_failure_count"], 3)
        self.assertEqual(summary["replan_count"], 3)
        self.assertEqual(summary["iris_fallback_count"], 7)
        self.assertEqual(summary["region_graph_fallback_count"], 7)
        self.assertEqual(summary["region_graph_disconnected_count"], 3)
        self.assertEqual(summary["scenario_group_summary"]["smoke"]["scenario_count"], 2)
        self.assertEqual(summary["scenario_group_summary"]["stress"]["failure_count"], 3)
        self.assertEqual(
            len(summary["source_summary_paths"]),
            2,
        )
        self.assertTrue(summary["source_summary_paths"][0].endswith("smoke-baseline-k1/path-feedback-summary.json"))

    def test_failed_single_run_still_writes_failure_records_and_nonzero_exit(self) -> None:
        output_root = self.temp_dir / "batch"
        matrix = self._write_matrix(
            {
                "schema_version": "path-feedback-batch-matrix/v1",
                "output_root": str(output_root),
                "runs": [
                    {
                        "run_id": "smoke-baseline-k1",
                        "scenario_set": "smoke",
                        "diagnostic_profile": "baseline",
                        "top_k": 1,
                    },
                    {
                        "run_id": "exit-fail",
                        "scenario_set": "stress",
                        "diagnostic_profile": "execution",
                        "top_k": 3,
                    },
                    {
                        "run_id": "open-grid-fail",
                        "scenario_set": "smoke",
                        "diagnostic_profile": "baseline",
                        "top_k": 1,
                    },
                ],
            }
        )

        completed = self._run_batch(
            "--matrix",
            str(matrix),
            "--single-run-script",
            str(self.fake_single_run),
        )

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        run_index = json.loads((output_root / "batch-run-index.json").read_text(encoding="utf-8"))
        runs_by_id = {item["run_id"]: item for item in run_index["runs"]}
        self.assertEqual(runs_by_id["smoke-baseline-k1"]["status"], "passed")
        self.assertEqual(runs_by_id["exit-fail"]["status"], "failed")
        self.assertIn("single_run_exit_nonzero", runs_by_id["exit-fail"]["reason_codes"])
        self.assertIn("summary_missing", runs_by_id["exit-fail"]["reason_codes"])
        self.assertEqual(runs_by_id["open-grid-fail"]["status"], "failed")
        self.assertIn("open_grid_fallback_used", runs_by_id["open-grid-fail"]["reason_codes"])

        summary = json.loads((output_root / "batch-evaluation-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["run_count"], 3)
        self.assertEqual(summary["passed_count"], 1)
        self.assertEqual(summary["failed_count"], 2)
        self.assertEqual(summary["open_grid_fallback_used_count"], 1)
        self.assertIn("exit-fail", summary["failed_run_ids"])
        self.assertIn("open-grid-fail", summary["failed_run_ids"])


class PathFeedbackSingleRunCompatibilityTests(unittest.TestCase):
    def test_single_run_default_dry_run_behavior_is_unchanged(self) -> None:
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
        self.assertIn("Output root: " + str(repo_root / "outputs" / "path_feedback_validation"), completed.stdout)
        self.assertIn("Acceptance gate: custom", completed.stdout)
        self.assertIn("Top-K: 3", completed.stdout)
        self.assertIn("Scenario set: smoke", completed.stdout)
        self.assertIn("Diagnostic profile: baseline", completed.stdout)


if __name__ == "__main__":
    unittest.main()
