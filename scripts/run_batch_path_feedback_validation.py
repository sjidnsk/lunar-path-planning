from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot


MATRIX_SCHEMA_VERSION = "path-feedback-batch-matrix/v1"
RUN_INDEX_SCHEMA_VERSION = "path-feedback-batch-run-index/v1"
EVALUATION_SUMMARY_SCHEMA_VERSION = "path-feedback-batch-evaluation-summary/v1"
SUMMARY_SCHEMA_VERSION = "path-feedback-summary/v1"
GCS_CONTROL_POINT_CANDIDATE_TRIAGE_SCHEMA_VERSION = "gcs-control-point-candidate-triage-summary/v1"
GCS_CONTROL_POINT_CANDIDATE_ARTIFACT_INDEX_SCHEMA_VERSION = (
    "gcs-control-point-candidate-artifact-index/v1"
)
GCS_CONTROL_POINT_CANDIDATE_CALIBRATION_SWEEP_SCHEMA_VERSION = (
    "gcs-control-point-candidate-calibration-sweep/v1"
)
SCENARIO_SETS = {
    "smoke",
    "stress",
    "holdout",
    "raw_align_train",
    "raw_align_val",
    "raw_align_test",
    "policy_canary",
    "policy_canary_diversity",
    "policy_canary_opportunity_quality",
    "all",
}
DIAGNOSTIC_PROFILES = {"baseline", "execution", "iris", "all"}
SUBMODULES = ("dev-platform-constraints", "model-explorer", "path-planner")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
OPTIONAL_BOOL_ARGS = {
    "anchor_projection_candidate_generation": "--anchor-projection-candidate-generation",
    "anchor_projection_contract_aware_trainable_target_generation": (
        "--anchor-projection-contract-aware-trainable-target-generation"
    ),
    "anchor_projection_prefer_contract_safe_trainable_targets": (
        "--anchor-projection-prefer-contract-safe-trainable-targets"
    ),
    "anchor_projection_planner_validated_trainable_target_mining": (
        "--anchor-projection-planner-validated-trainable-target-mining"
    ),
    "anchor_projection_allow_planner_validated_distance_exception": (
        "--anchor-projection-allow-planner-validated-distance-exception"
    ),
}
OPTIONAL_VALUE_ARGS = {
    "anchor_projection_selection_path_cost_bonus": "--anchor-projection-selection-path-cost-bonus",
    "anchor_projection_max_selection_path_cost_regression": (
        "--anchor-projection-max-selection-path-cost-regression"
    ),
    "anchor_projection_max_selection_risk_regression": (
        "--anchor-projection-max-selection-risk-regression"
    ),
    "anchor_projection_max_trainable_distance_cells": (
        "--anchor-projection-max-trainable-distance-cells"
    ),
    "anchor_projection_max_trainable_distance_m": "--anchor-projection-max-trainable-distance-m",
    "anchor_projection_max_planner_validated_distance_cells": (
        "--anchor-projection-max-planner-validated-distance-cells"
    ),
    "anchor_projection_max_planner_validated_distance_m": (
        "--anchor-projection-max-planner-validated-distance-m"
    ),
}


class MatrixError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a batch of semi-real path-feedback validation commands."
    )
    parser.add_argument("--matrix", required=True, help="Path to path-feedback batch matrix JSON.")
    parser.add_argument(
        "--output-root",
        help="Override the batch output root. Each run writes to <output-root>/<run_id>/.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print planned commands without writing outputs.")
    parser.add_argument("--validate-only", action="store_true", help="Validate the matrix and exit without running commands.")
    parser.add_argument(
        "--single-run-script",
        help="Single-run validation script to orchestrate. Defaults to scripts/run_path_feedback_validation.sh.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    matrix_path = _resolve_path(args.matrix, repo_root)
    single_run_script = (
        _resolve_path(args.single_run_script, repo_root)
        if args.single_run_script
        else repo_root / "scripts" / "run_path_feedback_validation.sh"
    )

    try:
        matrix_payload = _load_matrix_json(matrix_path)
        batch_plan = _build_batch_plan(
            matrix_payload,
            matrix_path=matrix_path,
            repo_root=repo_root,
            cli_output_root=args.output_root,
            single_run_script=single_run_script,
        )
    except MatrixError as exc:
        print(f"matrix error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": "matrix validated",
                "matrix": _display_path(matrix_path, repo_root),
                "run_count": len(batch_plan["runs"]),
                "output_root": _display_path(batch_plan["output_root"], repo_root),
            },
            ensure_ascii=False,
        )
    )

    if args.validate_only:
        return 0

    if args.dry_run:
        for run in batch_plan["runs"]:
            print(f"[DRY RUN] {run['run_id']}: (cd {repo_root} && {shlex.join(run['argv'])})")
        return 0

    git_snapshot = _git_snapshot(repo_root)
    batch_plan["output_root"].mkdir(parents=True, exist_ok=True)

    run_records: list[dict[str, Any]] = []
    for run in batch_plan["runs"]:
        run_records.append(_execute_run(run, repo_root=repo_root, git_snapshot=git_snapshot))

    run_index = _build_run_index(batch_plan, run_records, repo_root=repo_root, git_snapshot=git_snapshot)
    evaluation_summary = _build_evaluation_summary(batch_plan, run_records, repo_root=repo_root)

    index_path = batch_plan["output_root"] / "batch-run-index.json"
    summary_path = batch_plan["output_root"] / "batch-evaluation-summary.json"
    _write_json(index_path, run_index)
    _write_json(summary_path, evaluation_summary)

    print(
        json.dumps(
            {
                "status": "batch complete" if evaluation_summary["failed_count"] == 0 else "batch failed",
                "run_count": evaluation_summary["run_count"],
                "passed_count": evaluation_summary["passed_count"],
                "failed_count": evaluation_summary["failed_count"],
                "batch_run_index": _display_path(index_path, repo_root),
                "batch_evaluation_summary": _display_path(summary_path, repo_root),
            },
            ensure_ascii=False,
        )
    )
    return 1 if evaluation_summary["failed_count"] else 0


def _load_matrix_json(matrix_path: Path) -> dict[str, Any]:
    if not matrix_path.is_file():
        raise MatrixError(f"matrix file does not exist: {matrix_path}")
    try:
        payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MatrixError(f"matrix JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise MatrixError("matrix root must be an object")
    schema_version = payload.get("schema_version")
    if schema_version != MATRIX_SCHEMA_VERSION:
        raise MatrixError(f"schema_version must be {MATRIX_SCHEMA_VERSION!r}")
    return payload


def _build_batch_plan(
    payload: dict[str, Any],
    *,
    matrix_path: Path,
    repo_root: Path,
    cli_output_root: str | None,
    single_run_script: Path,
) -> dict[str, Any]:
    defaults = payload.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        raise MatrixError("defaults must be an object when present")

    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise MatrixError("runs must be a non-empty array")

    output_root_value = (
        cli_output_root
        if cli_output_root is not None
        else payload.get("output_root", defaults.get("output_root", "outputs/path_feedback_batch"))
    )
    if not isinstance(output_root_value, str) or not output_root_value:
        raise MatrixError("output_root must be a non-empty string")
    batch_output_root = _resolve_path(output_root_value, repo_root)

    if not single_run_script.is_file():
        raise MatrixError(f"single-run script does not exist: {single_run_script}")

    seen_run_ids: set[str] = set()
    runs: list[dict[str, Any]] = []
    for index, raw_run in enumerate(raw_runs):
        if not isinstance(raw_run, dict):
            raise MatrixError(f"runs[{index}] must be an object")
        merged = {**defaults, **raw_run}
        run_id = _require_run_id(merged.get("run_id"), index)
        if run_id in seen_run_ids:
            raise MatrixError(f"run_id must be unique: {run_id}")
        seen_run_ids.add(run_id)

        scenario_set = _require_choice(merged.get("scenario_set"), SCENARIO_SETS, f"runs[{index}].scenario_set")
        diagnostic_profile = _require_choice(
            merged.get("diagnostic_profile"),
            DIAGNOSTIC_PROFILES,
            f"runs[{index}].diagnostic_profile",
        )
        top_k = _require_positive_int(merged.get("top_k"), f"runs[{index}].top_k")
        sample_quality_profile = merged.get("sample_quality_profile")
        if sample_quality_profile is not None and not isinstance(sample_quality_profile, (str, dict)):
            raise MatrixError(f"runs[{index}].sample_quality_profile must be a string or object when present")
        planner_extra_args = _optional_string_list(
            merged.get("planner_extra_args", ()),
            f"runs[{index}].planner_extra_args",
        )

        if cli_output_root is None and isinstance(raw_run.get("output_root"), str):
            run_output_root = _resolve_path(raw_run["output_root"], repo_root)
        else:
            run_output_root = batch_output_root / run_id

        argv = [
            "bash",
            str(single_run_script),
            "--scenario-set",
            scenario_set,
            "--diagnostic-profile",
            diagnostic_profile,
            "--top-k",
            str(top_k),
            "--output-root",
            str(run_output_root),
        ]
        optional_cli_args = _optional_cli_args(merged, index=index)
        argv.extend(optional_cli_args)
        argv.extend(planner_extra_args)
        runs.append(
            {
                "run_id": run_id,
                "scenario_set": scenario_set,
                "diagnostic_profile": diagnostic_profile,
                "top_k": top_k,
                "sample_quality_profile": sample_quality_profile,
                "optional_cli_args": optional_cli_args,
                "planner_extra_args": planner_extra_args,
                "output_root": run_output_root,
                "argv": argv,
            }
        )

    return {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_path": matrix_path,
        "output_root": batch_output_root,
        "single_run_script": single_run_script,
        "runs": runs,
    }


def _require_run_id(value: Any, index: int) -> str:
    if not isinstance(value, str) or not value:
        raise MatrixError(f"runs[{index}].run_id must be a non-empty string")
    if not RUN_ID_RE.match(value):
        raise MatrixError(f"runs[{index}].run_id may contain only letters, numbers, dot, underscore, and dash")
    return value


def _require_choice(value: Any, choices: set[str], label: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise MatrixError(f"{label} must be one of: {', '.join(sorted(choices))}")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MatrixError(f"{label} must be a positive integer")
    return value


def _optional_string_list(value: Any, label: str) -> list[str]:
    if value in (None, ()):
        return []
    if not isinstance(value, list):
        raise MatrixError(f"{label} must be an array of strings when present")
    if any(not isinstance(item, str) or item == "" for item in value):
        raise MatrixError(f"{label} must contain only non-empty strings")
    return list(value)


def _optional_cli_args(merged: dict[str, Any], *, index: int) -> list[str]:
    args: list[str] = []
    for field, cli_arg in OPTIONAL_BOOL_ARGS.items():
        if field not in merged:
            continue
        value = merged[field]
        if not isinstance(value, bool):
            raise MatrixError(f"runs[{index}].{field} must be a boolean when present")
        if value:
            args.append(cli_arg)
    for field, cli_arg in OPTIONAL_VALUE_ARGS.items():
        if field not in merged:
            continue
        value = merged[field]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise MatrixError(f"runs[{index}].{field} must be a string or number when present")
        text = str(value)
        if not text:
            raise MatrixError(f"runs[{index}].{field} must not be empty")
        args.extend([cli_arg, text])
    return args


def _execute_run(run: dict[str, Any], *, repo_root: Path, git_snapshot: dict[str, Any]) -> dict[str, Any]:
    run_root = run["output_root"]
    run_root.mkdir(parents=True, exist_ok=True)
    stdout_log = run_root / "batch-run.stdout.log"
    stderr_log = run_root / "batch-run.stderr.log"

    print(f"==> batch run {run['run_id']}: {shlex.join(run['argv'])}")
    with stdout_log.open("w", encoding="utf-8") as stdout_file, stderr_log.open("w", encoding="utf-8") as stderr_file:
        completed = subprocess.run(
            run["argv"],
            cwd=repo_root,
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
        )

    record = _record_run_result(
        run,
        repo_root=repo_root,
        git_snapshot=git_snapshot,
        return_code=completed.returncode,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
    )
    print(
        json.dumps(
            {
                "run_id": record["run_id"],
                "status": record["status"],
                "return_code": record["return_code"],
                "reason_codes": record["reason_codes"],
            },
            ensure_ascii=False,
        )
    )
    return record


def _record_run_result(
    run: dict[str, Any],
    *,
    repo_root: Path,
    git_snapshot: dict[str, Any],
    return_code: int,
    stdout_log: Path,
    stderr_log: Path,
) -> dict[str, Any]:
    run_root = run["output_root"]
    manifest_path = run_root / "path-feedback-manifest.json"
    summary_path = run_root / "path-feedback-summary.json"
    report_path = run_root / "path-feedback-summary.md"
    reason_codes: list[str] = []
    summary: dict[str, Any] | None = None

    if return_code != 0:
        _append_reason(reason_codes, "single_run_exit_nonzero")

    if not summary_path.is_file():
        _append_reason(reason_codes, "summary_missing")
    else:
        try:
            loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            _append_reason(reason_codes, "summary_invalid_json")
        else:
            if isinstance(loaded_summary, dict):
                summary = loaded_summary
                _inspect_summary_for_reasons(summary, reason_codes)
            else:
                _append_reason(reason_codes, "summary_not_object")

    if not manifest_path.is_file():
        _append_reason(reason_codes, "manifest_missing")
    if not report_path.is_file():
        _append_reason(reason_codes, "report_missing")

    status = "failed" if reason_codes else "passed"
    acceptance_metadata = _acceptance_metadata_from_summary(summary)
    python_executable = (
        acceptance_metadata.get("python_executable")
        if isinstance(acceptance_metadata.get("python_executable"), str)
        else None
    )
    return {
        "run_id": run["run_id"],
        "status": status,
        "return_code": return_code,
        "reason_codes": reason_codes,
        "command_argv": list(run["argv"]),
        "command_args": {
            "scenario_set": run["scenario_set"],
            "diagnostic_profile": run["diagnostic_profile"],
            "top_k": run["top_k"],
            "python_executable": python_executable,
            "output_root": _display_path(run_root, repo_root),
            "planner_extra_args": list(run["planner_extra_args"]),
        },
        "sample_quality_profile": run["sample_quality_profile"],
        "source_paths": {
            "output_root": _display_path(run_root, repo_root),
            "manifest": _display_path(manifest_path, repo_root),
            "summary": _display_path(summary_path, repo_root),
            "report": _display_path(report_path, repo_root),
            "stdout_log": _display_path(stdout_log, repo_root),
            "stderr_log": _display_path(stderr_log, repo_root),
        },
        "summary_path": _display_path(summary_path, repo_root),
        "report_path": _display_path(report_path, repo_root),
        "acceptance_metadata": acceptance_metadata,
        "open_grid_fallback_used": bool(summary.get("open_grid_fallback_used")) if summary else None,
        "summary": summary,
        "git": git_snapshot,
    }


def _inspect_summary_for_reasons(summary: dict[str, Any], reason_codes: list[str]) -> None:
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        _append_reason(reason_codes, "summary_schema_mismatch")

    open_grid_used = summary.get("open_grid_fallback_used")
    if open_grid_used is not False:
        _append_reason(reason_codes, "open_grid_fallback_used" if open_grid_used is True else "open_grid_fallback_unknown")

    acceptance_metadata = _acceptance_metadata_from_summary(summary)
    gate = acceptance_metadata.get("open_grid_fallback_used_gate")
    if not isinstance(gate, dict):
        gate = summary.get("open_grid_fallback_used_gate")
    if not isinstance(gate, dict):
        _append_reason(reason_codes, "open_grid_fallback_gate_missing")
        return
    if gate.get("status") != "passed":
        _append_reason(reason_codes, "open_grid_fallback_gate_failed")
    for code in gate.get("reason_codes", []):
        if isinstance(code, str) and code != "open_grid_fallback_not_used":
            _append_reason(reason_codes, code)


def _acceptance_metadata_from_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    metadata = summary.get("acceptance_metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _build_run_index(
    batch_plan: dict[str, Any],
    run_records: list[dict[str, Any]],
    *,
    repo_root: Path,
    git_snapshot: dict[str, Any],
) -> dict[str, Any]:
    public_runs = [_public_run_record(record) for record in run_records]
    failed_run_ids = [record["run_id"] for record in public_runs if record["status"] == "failed"]
    return {
        "schema_version": RUN_INDEX_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "matrix_path": _display_path(batch_plan["matrix_path"], repo_root),
        "output_root": _display_path(batch_plan["output_root"], repo_root),
        "single_run_script": _display_path(batch_plan["single_run_script"], repo_root),
        "run_count": len(public_runs),
        "passed_count": len(public_runs) - len(failed_run_ids),
        "failed_count": len(failed_run_ids),
        "failed_run_ids": failed_run_ids,
        "git": git_snapshot,
        "runs": public_runs,
    }


def _public_run_record(record: dict[str, Any]) -> dict[str, Any]:
    public = dict(record)
    public.pop("summary", None)
    return public


def _build_evaluation_summary(
    batch_plan: dict[str, Any],
    run_records: list[dict[str, Any]],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    public_runs = [_summary_run_record(record) for record in run_records]
    failed_run_ids = [record["run_id"] for record in run_records if record["status"] == "failed"]
    parsed_summaries = [record["summary"] for record in run_records if isinstance(record.get("summary"), dict)]
    reason_code_counts = Counter(
        code
        for record in run_records
        for code in record["reason_codes"]
    )
    open_grid_count = sum(1 for summary in parsed_summaries if summary.get("open_grid_fallback_used") is True)
    control_point_artifacts = _aggregate_control_point_artifacts(parsed_summaries)
    control_point_triage = _aggregate_control_point_triage(
        parsed_summaries,
        route_artifact_count=int(control_point_artifacts["route_artifact_count"]),
    )
    channel_aware_report_count = _sum_summary_int(parsed_summaries, "channel_aware_astar_report_count")
    channel_aware_path_changed_count = _sum_summary_int(
        parsed_summaries,
        "channel_aware_astar_path_changed_count",
    )

    return {
        "schema_version": EVALUATION_SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "matrix_path": _display_path(batch_plan["matrix_path"], repo_root),
        "output_root": _display_path(batch_plan["output_root"], repo_root),
        "run_count": len(run_records),
        "passed_count": len(run_records) - len(failed_run_ids),
        "failed_count": len(failed_run_ids),
        "failed_run_ids": failed_run_ids,
        "failure_reason_code_counts": dict(sorted(reason_code_counts.items())),
        "open_grid_fallback_used_count": open_grid_count,
        "open_grid_fallback_gate": _aggregate_open_grid_gates(run_records),
        "path_planning_failure_count": _sum_summary_int(parsed_summaries, "path_planning_failure_count"),
        "replan_count": _sum_summary_int(parsed_summaries, "replan_count"),
        "iris_fallback_count": _sum_summary_int(parsed_summaries, "iris_fallback_count"),
        "region_graph_fallback_count": _sum_summary_int(parsed_summaries, "region_graph_fallback_count"),
        "region_graph_disconnected_count": _sum_region_graph_disconnected(parsed_summaries),
        "convex_region_report_count": _sum_summary_int(
            parsed_summaries,
            "convex_region_report_count",
        ),
        "convex_region_count_total": _sum_summary_int(
            parsed_summaries,
            "convex_region_count_total",
        ),
        "convex_region_backend_counts": _aggregate_summary_counter(
            parsed_summaries,
            "convex_region_backend_counts",
        ),
        "convex_region_fallback_used_count": _sum_summary_int(
            parsed_summaries,
            "convex_region_fallback_used_count",
        ),
        "convex_region_gcs_ready_count": _sum_summary_int(
            parsed_summaries,
            "convex_region_gcs_ready_count",
        ),
        "convex_region_blocked_cell_violation_count": _sum_summary_int(
            parsed_summaries,
            "convex_region_blocked_cell_violation_count",
        ),
        "convex_region_coverage_status_counts": _aggregate_summary_counter(
            parsed_summaries,
            "convex_region_coverage_status_counts",
        ),
        "convex_region_gcs_ready_reason_counts": _aggregate_summary_counter(
            parsed_summaries,
            "convex_region_gcs_ready_reason_counts",
        ),
        "convex_region_start_contained_count": _sum_summary_int(
            parsed_summaries,
            "convex_region_start_contained_count",
        ),
        "convex_region_goal_contained_count": _sum_summary_int(
            parsed_summaries,
            "convex_region_goal_contained_count",
        ),
        "convex_region_adjacent_overlap_count": _sum_summary_int(
            parsed_summaries,
            "convex_region_adjacent_overlap_count",
        ),
        "convex_region_portal_count": _sum_summary_int(
            parsed_summaries,
            "convex_region_portal_count",
        ),
        "convex_region_candidate_audit": _aggregate_summary_list(
            parsed_summaries,
            "convex_region_candidate_audit",
        ),
        "gcs_trajectory_report_count": _sum_summary_int(
            parsed_summaries,
            "gcs_trajectory_report_count",
        ),
        "gcs_trajectory_attempted_count": _sum_summary_int(
            parsed_summaries,
            "gcs_trajectory_attempted_count",
        ),
        "gcs_trajectory_success_count": _sum_summary_int(
            parsed_summaries,
            "gcs_trajectory_success_count",
        ),
        "gcs_trajectory_collision_count": _sum_summary_int(
            parsed_summaries,
            "gcs_trajectory_collision_count",
        ),
        "gcs_trajectory_region_count_total": _sum_summary_int(
            parsed_summaries,
            "gcs_trajectory_region_count_total",
        ),
        "gcs_trajectory_sample_count_total": _sum_summary_int(
            parsed_summaries,
            "gcs_trajectory_sample_count_total",
        ),
        "gcs_trajectory_backend_counts": _aggregate_summary_counter(
            parsed_summaries,
            "gcs_trajectory_backend_counts",
        ),
        "gcs_trajectory_reason_counts": _aggregate_summary_counter(
            parsed_summaries,
            "gcs_trajectory_reason_counts",
        ),
        "gcs_trajectory_result_status_counts": _aggregate_summary_counter(
            parsed_summaries,
            "gcs_trajectory_result_status_counts",
        ),
        "gcs_trajectory_candidate_audit": _aggregate_summary_list(
            parsed_summaries,
            "gcs_trajectory_candidate_audit",
        ),
        "gcs_candidate_report_count": _sum_summary_int(
            parsed_summaries,
            "gcs_candidate_report_count",
        ),
        "gcs_candidate_attempted_count": _sum_summary_int(
            parsed_summaries,
            "gcs_candidate_attempted_count",
        ),
        "gcs_candidate_available_count": _sum_summary_int(
            parsed_summaries,
            "gcs_candidate_available_count",
        ),
        "gcs_candidate_selected_count": _sum_summary_int(
            parsed_summaries,
            "gcs_candidate_selected_count",
        ),
        "gcs_candidate_collision_count": _sum_summary_int(
            parsed_summaries,
            "gcs_candidate_collision_count",
        ),
        "gcs_candidate_fallback_reason_counts": _aggregate_summary_counter(
            parsed_summaries,
            "gcs_candidate_fallback_reason_counts",
        ),
        "gcs_candidate_selection_reason_counts": _aggregate_summary_counter(
            parsed_summaries,
            "gcs_candidate_selection_reason_counts",
        ),
        "gcs_candidate_cost_delta_vs_baseline_negative_count": _sum_summary_int(
            parsed_summaries,
            "gcs_candidate_cost_delta_vs_baseline_negative_count",
        ),
        "gcs_candidate_cost_delta_vs_baseline_positive_count": _sum_summary_int(
            parsed_summaries,
            "gcs_candidate_cost_delta_vs_baseline_positive_count",
        ),
        "gcs_candidate_cost_delta_vs_baseline_zero_count": _sum_summary_int(
            parsed_summaries,
            "gcs_candidate_cost_delta_vs_baseline_zero_count",
        ),
        "gcs_candidate_audit": _aggregate_summary_list(
            parsed_summaries,
            "gcs_candidate_audit",
        ),
        "gcs_motion_feasibility_report_count": _sum_summary_int(
            parsed_summaries,
            "gcs_motion_feasibility_report_count",
        ),
        "gcs_motion_feasibility_evaluated_count": _sum_summary_int(
            parsed_summaries,
            "gcs_motion_feasibility_evaluated_count",
        ),
        "gcs_motion_feasibility_feasible_count": _sum_summary_int(
            parsed_summaries,
            "gcs_motion_feasibility_feasible_count",
        ),
        "gcs_motion_feasibility_infeasible_count": _sum_summary_int(
            parsed_summaries,
            "gcs_motion_feasibility_infeasible_count",
        ),
        "gcs_motion_feasibility_diagnostic_only_count": _sum_summary_int(
            parsed_summaries,
            "gcs_motion_feasibility_diagnostic_only_count",
        ),
        "gcs_motion_feasibility_curvature_violation_count": _sum_summary_int(
            parsed_summaries,
            "gcs_motion_feasibility_curvature_violation_count",
        ),
        "gcs_motion_feasibility_heading_violation_count": _sum_summary_int(
            parsed_summaries,
            "gcs_motion_feasibility_heading_violation_count",
        ),
        "gcs_motion_feasibility_status_counts": _aggregate_summary_counter(
            parsed_summaries,
            "gcs_motion_feasibility_status_counts",
        ),
        "gcs_motion_feasibility_fallback_reason_counts": _aggregate_summary_counter(
            parsed_summaries,
            "gcs_motion_feasibility_fallback_reason_counts",
        ),
        "gcs_motion_feasibility_motion_model_counts": _aggregate_summary_counter(
            parsed_summaries,
            "gcs_motion_feasibility_motion_model_counts",
        ),
        "gcs_motion_feasibility_audit": _aggregate_summary_list(
            parsed_summaries,
            "gcs_motion_feasibility_audit",
        ),
        "gcs_curvature_constrained_report_count": _sum_summary_int(
            parsed_summaries,
            "gcs_curvature_constrained_report_count",
        ),
        "gcs_curvature_constrained_attempted_count": _sum_summary_int(
            parsed_summaries,
            "gcs_curvature_constrained_attempted_count",
        ),
        "gcs_curvature_constrained_available_count": _sum_summary_int(
            parsed_summaries,
            "gcs_curvature_constrained_available_count",
        ),
        "gcs_curvature_constrained_selected_count": _sum_summary_int(
            parsed_summaries,
            "gcs_curvature_constrained_selected_count",
        ),
        "gcs_curvature_constrained_repair_success_count": _sum_summary_int(
            parsed_summaries,
            "gcs_curvature_constrained_repair_success_count",
        ),
        "gcs_curvature_constrained_infeasible_count": _sum_summary_int(
            parsed_summaries,
            "gcs_curvature_constrained_infeasible_count",
        ),
        "gcs_curvature_constrained_diagnostic_only_count": _sum_summary_int(
            parsed_summaries,
            "gcs_curvature_constrained_diagnostic_only_count",
        ),
        "gcs_curvature_constrained_curvature_violation_count_before": _sum_summary_int(
            parsed_summaries,
            "gcs_curvature_constrained_curvature_violation_count_before",
        ),
        "gcs_curvature_constrained_curvature_violation_count_after": _sum_summary_int(
            parsed_summaries,
            "gcs_curvature_constrained_curvature_violation_count_after",
        ),
        "gcs_curvature_constrained_heading_violation_count_before": _sum_summary_int(
            parsed_summaries,
            "gcs_curvature_constrained_heading_violation_count_before",
        ),
        "gcs_curvature_constrained_heading_violation_count_after": _sum_summary_int(
            parsed_summaries,
            "gcs_curvature_constrained_heading_violation_count_after",
        ),
        "gcs_curvature_constrained_collision_count": _sum_summary_int(
            parsed_summaries,
            "gcs_curvature_constrained_collision_count",
        ),
        "gcs_curvature_constrained_region_containment_violation_count": _sum_summary_int(
            parsed_summaries,
            "gcs_curvature_constrained_region_containment_violation_count",
        ),
        "gcs_curvature_constrained_status_before_counts": _aggregate_summary_counter(
            parsed_summaries,
            "gcs_curvature_constrained_status_before_counts",
        ),
        "gcs_curvature_constrained_status_after_counts": _aggregate_summary_counter(
            parsed_summaries,
            "gcs_curvature_constrained_status_after_counts",
        ),
        "gcs_curvature_constrained_fallback_reason_counts": _aggregate_summary_counter(
            parsed_summaries,
            "gcs_curvature_constrained_fallback_reason_counts",
        ),
        "gcs_curvature_constrained_repair_strategy_counts": _aggregate_summary_counter(
            parsed_summaries,
            "gcs_curvature_constrained_repair_strategy_counts",
        ),
        "gcs_curvature_constrained_audit": _aggregate_summary_list(
            parsed_summaries,
            "gcs_curvature_constrained_audit",
        ),
        "gcs_control_point_report_count": _sum_summary_int(
            parsed_summaries,
            "gcs_control_point_report_count",
        ),
        "gcs_control_point_attempted_count": _sum_summary_int(
            parsed_summaries,
            "gcs_control_point_attempted_count",
        ),
        "gcs_control_point_success_count": _sum_summary_int(
            parsed_summaries,
            "gcs_control_point_success_count",
        ),
        "gcs_control_point_backend_counts": _aggregate_summary_counter(
            parsed_summaries,
            "gcs_control_point_backend_counts",
        ),
        "gcs_control_point_candidate_selected_count": _sum_summary_int(
            parsed_summaries,
            "gcs_control_point_candidate_selected_count",
        ),
        "gcs_control_point_candidate_fallback_reason_counts": _aggregate_summary_counter(
            parsed_summaries,
            "gcs_control_point_candidate_fallback_reason_counts",
        ),
        "gcs_control_point_terrain_objective_source_counts": _aggregate_summary_counter(
            parsed_summaries,
            "gcs_control_point_terrain_objective_source_counts",
        ),
        **_aggregate_summary_metric_stats(
            parsed_summaries,
            "gcs_control_point_sampled_terrain_cost",
        ),
        **_aggregate_summary_metric_stats(
            parsed_summaries,
            "gcs_control_point_high_cost_exposure_delta",
        ),
        "gcs_control_point_candidate_audit": _aggregate_summary_list(
            parsed_summaries,
            "gcs_control_point_candidate_audit",
        ),
        "gcs_control_point_candidate_artifacts": control_point_artifacts,
        "gcs_control_point_candidate_triage": control_point_triage,
        "channel_aware_astar_report_count": channel_aware_report_count,
        "channel_aware_astar_selected_count": _sum_summary_int(
            parsed_summaries,
            "channel_aware_astar_selected_count",
        ),
        "channel_aware_astar_fallback_count": _sum_summary_int(
            parsed_summaries,
            "channel_aware_astar_fallback_count",
        ),
        "channel_aware_astar_requested_backend_counts": _aggregate_summary_counter(
            parsed_summaries,
            "channel_aware_astar_requested_backend_counts",
        ),
        "channel_aware_astar_selected_backend_counts": _aggregate_summary_counter(
            parsed_summaries,
            "channel_aware_astar_selected_backend_counts",
        ),
        "channel_aware_astar_status_counts": _aggregate_summary_counter(
            parsed_summaries,
            "channel_aware_astar_status_counts",
        ),
        "channel_aware_astar_fallback_reason_counts": _aggregate_summary_counter(
            parsed_summaries,
            "channel_aware_astar_fallback_reason_counts",
        ),
        "channel_aware_astar_blocker_class_counts": _aggregate_summary_counter(
            parsed_summaries,
            "channel_aware_astar_blocker_class_counts",
        ),
        "channel_aware_astar_path_changed_count": channel_aware_path_changed_count,
        "channel_aware_astar_path_changed_rate": (
            channel_aware_path_changed_count / channel_aware_report_count
            if channel_aware_report_count
            else 0.0
        ),
        **_aggregate_summary_metric_stats(
            parsed_summaries,
            "channel_aware_astar_path_cost_delta",
        ),
        **_aggregate_summary_metric_stats(
            parsed_summaries,
            "channel_aware_astar_channel_cost_delta",
        ),
        **_aggregate_summary_metric_stats(
            parsed_summaries,
            "channel_aware_astar_high_cost_exposure_delta",
        ),
        "channel_aware_astar_candidate_audit": _aggregate_summary_list(
            parsed_summaries,
            "channel_aware_astar_candidate_audit",
        ),
        "sampled_region_path_selected_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_selected_count",
        ),
        "sampled_region_path_fallback_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_fallback_count",
        ),
        "sampled_region_path_status_counts": _aggregate_summary_counter(
            parsed_summaries,
            "sampled_region_path_status_counts",
        ),
        "sampled_region_path_source_counts": _aggregate_summary_counter(
            parsed_summaries,
            "sampled_region_path_source_counts",
        ),
        "sampled_region_path_fallback_reasons": _aggregate_summary_counter(
            parsed_summaries,
            "sampled_region_path_fallback_reasons",
        ),
        "sampled_region_path_sample_attempt_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_sample_attempt_count",
        ),
        "sampled_region_path_candidate_ranking_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_candidate_ranking_count",
        ),
        "sampled_region_path_anchor_region_added_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_anchor_region_added_count",
        ),
        "sampled_region_path_anchor_region_connected_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_anchor_region_connected_count",
        ),
        "sampled_region_path_anchor_closure_attempt_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_anchor_closure_attempt_count",
        ),
        "sampled_region_path_anchor_closure_connected_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_anchor_closure_connected_count",
        ),
        "sampled_region_path_anchor_closure_status_counts": _aggregate_summary_counter(
            parsed_summaries,
            "sampled_region_path_anchor_closure_status_counts",
        ),
        "sampled_region_path_anchor_closure_reason_counts": _aggregate_summary_counter(
            parsed_summaries,
            "sampled_region_path_anchor_closure_reason_counts",
        ),
        "sampled_region_path_anchor_closure_connection_kind_counts": _aggregate_summary_counter(
            parsed_summaries,
            "sampled_region_path_anchor_closure_connection_kind_counts",
        ),
        "sampled_region_path_start_classification_counts": _aggregate_summary_counter(
            parsed_summaries,
            "sampled_region_path_start_classification_counts",
        ),
        "sampled_region_path_goal_classification_counts": _aggregate_summary_counter(
            parsed_summaries,
            "sampled_region_path_goal_classification_counts",
        ),
        "sampled_region_path_connector_attempt_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_connector_attempt_count",
        ),
        "sampled_region_path_connector_strategy_counts": _aggregate_summary_counter(
            parsed_summaries,
            "sampled_region_path_connector_strategy_counts",
        ),
        "sampled_region_path_bridge_aware_connector_attempt_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_bridge_aware_connector_attempt_count",
        ),
        "sampled_region_path_bridge_aware_connector_available_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_bridge_aware_connector_available_count",
        ),
        "sampled_region_path_bridge_aware_connector_selected_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_bridge_aware_connector_selected_count",
        ),
        "sampled_region_path_bridge_aware_connector_rejected_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_bridge_aware_connector_rejected_count",
        ),
        "sampled_region_path_bridge_aware_connector_status_counts": _aggregate_summary_counter(
            parsed_summaries,
            "sampled_region_path_bridge_aware_connector_status_counts",
        ),
        "sampled_region_path_bridge_aware_fallback_reasons": _aggregate_summary_counter(
            parsed_summaries,
            "sampled_region_path_bridge_aware_fallback_reasons",
        ),
        "sampled_region_path_bridge_aware_bridge_cell_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_bridge_aware_bridge_cell_count",
        ),
        "sampled_region_path_bridge_aware_mask_added_cell_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_bridge_aware_mask_added_cell_count",
        ),
        "sampled_region_path_bridge_corridor_connector_attempt_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_bridge_corridor_connector_attempt_count",
        ),
        "sampled_region_path_bridge_corridor_connector_available_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_bridge_corridor_connector_available_count",
        ),
        "sampled_region_path_bridge_corridor_connector_selected_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_bridge_corridor_connector_selected_count",
        ),
        "sampled_region_path_bridge_corridor_connector_rejected_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_bridge_corridor_connector_rejected_count",
        ),
        "sampled_region_path_bridge_corridor_status_counts": _aggregate_summary_counter(
            parsed_summaries,
            "sampled_region_path_bridge_corridor_status_counts",
        ),
        "sampled_region_path_bridge_corridor_fallback_reasons": _aggregate_summary_counter(
            parsed_summaries,
            "sampled_region_path_bridge_corridor_fallback_reasons",
        ),
        "sampled_region_path_bridge_corridor_radius_counts": _aggregate_summary_counter(
            parsed_summaries,
            "sampled_region_path_bridge_corridor_radius_counts",
        ),
        "sampled_region_path_bridge_corridor_added_cell_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_bridge_corridor_added_cell_count",
        ),
        "sampled_region_path_terminal_adjusted_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_terminal_adjusted_count",
        ),
        "sampled_region_path_terminal_adjustment_candidate_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_terminal_adjustment_candidate_count",
        ),
        "sampled_region_path_terminal_adjustment_status_counts": _aggregate_summary_counter(
            parsed_summaries,
            "sampled_region_path_terminal_adjustment_status_counts",
        ),
        "sampled_region_path_terminal_adjustment_reason_counts": _aggregate_summary_counter(
            parsed_summaries,
            "sampled_region_path_terminal_adjustment_reason_counts",
        ),
        "sampled_region_path_reachable_component_status_counts": _aggregate_summary_counter(
            parsed_summaries,
            "sampled_region_path_reachable_component_status_counts",
        ),
        "sampled_region_path_reachable_component_reason_counts": _aggregate_summary_counter(
            parsed_summaries,
            "sampled_region_path_reachable_component_reason_counts",
        ),
        "sampled_region_path_reachable_component_disconnected_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_reachable_component_disconnected_count",
        ),
        "sampled_region_path_reachable_component_replacement_selected_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_reachable_component_replacement_selected_count",
        ),
        "sampled_region_path_reachable_component_terminal_candidate_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_reachable_component_terminal_candidate_count",
        ),
        "sampled_region_path_reachable_terminal_rescue_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_reachable_terminal_rescue_count",
        ),
        "sampled_region_path_proxy_goal_anchor_selected_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_proxy_goal_anchor_selected_count",
        ),
        "sampled_region_path_goal_rescue_candidate_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_goal_rescue_candidate_count",
        ),
        "sampled_region_path_benefit_surface_present_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_benefit_surface_present_count",
        ),
        "sampled_region_path_path_duplicate_with_baseline_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_path_duplicate_with_baseline_count",
        ),
        "sampled_region_path_baseline_equivalent_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_baseline_equivalent_count",
        ),
        "sampled_region_path_no_quality_gain_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_no_quality_gain_count",
        ),
        "sampled_region_path_fixture_no_benefit_surface_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_fixture_no_benefit_surface_count",
        ),
        "sampled_region_path_candidate_missing_metrics_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_candidate_missing_metrics_count",
        ),
        "sampled_region_path_constrained_connector_failed_count": _sum_summary_int(
            parsed_summaries,
            "sampled_region_path_constrained_connector_failed_count",
        ),
        "sampled_region_path_complexity_reason_counts": _aggregate_summary_counter(
            parsed_summaries,
            "sampled_region_path_complexity_reason_counts",
        ),
        "sampled_region_path_execution_tie_break_status_counts": _aggregate_summary_counter(
            parsed_summaries,
            "sampled_region_path_execution_tie_break_status_counts",
        ),
        "sampled_region_path_execution_tie_break_reason_counts": _aggregate_summary_counter(
            parsed_summaries,
            "sampled_region_path_execution_tie_break_reason_counts",
        ),
        "sampled_region_path_candidate_audit": _aggregate_summary_list(
            parsed_summaries,
            "sampled_region_path_candidate_audit",
        ),
        "scenario_group_summary": _aggregate_scenario_groups(parsed_summaries),
        "source_summary_paths": [
            record["source_paths"]["summary"]
            for record in run_records
            if isinstance(record.get("summary"), dict)
        ],
        "runs": public_runs,
    }


def _summary_run_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": record["run_id"],
        "status": record["status"],
        "reason_codes": list(record["reason_codes"]),
        "command_args": dict(record["command_args"]),
        "sample_quality_profile": record["sample_quality_profile"],
        "summary_path": record["summary_path"],
        "report_path": record["report_path"],
        "acceptance_metadata": dict(record["acceptance_metadata"]),
        "open_grid_fallback_used": record["open_grid_fallback_used"],
    }


def _aggregate_open_grid_gates(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    by_run: dict[str, Any] = {}
    failed_run_ids: list[str] = []
    for record in run_records:
        metadata = record.get("acceptance_metadata", {})
        gate = metadata.get("open_grid_fallback_used_gate") if isinstance(metadata, dict) else None
        if not isinstance(gate, dict) and isinstance(record.get("summary"), dict):
            gate = record["summary"].get("open_grid_fallback_used_gate")
        if isinstance(gate, dict):
            status = str(gate.get("status", "unknown"))
            gate_payload = dict(gate)
        else:
            status = "missing"
            gate_payload = {
                "status": "missing",
                "expected": False,
                "actual": record.get("open_grid_fallback_used"),
                "reason_codes": ["open_grid_fallback_gate_missing"],
            }
        status_counts[status] += 1
        by_run[record["run_id"]] = gate_payload
        if status != "passed":
            failed_run_ids.append(record["run_id"])
    return {
        "status_counts": dict(sorted(status_counts.items())),
        "failed_run_ids": failed_run_ids,
        "by_run": by_run,
    }


def _aggregate_scenario_groups(summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        groups = summary.get("scenario_group_summary", {})
        if not isinstance(groups, dict):
            continue
        for group_name, group_payload in groups.items():
            if not isinstance(group_payload, dict):
                continue
            bucket = aggregate.setdefault(str(group_name), {})
            for key, value in group_payload.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, int):
                    bucket[key] = int(bucket.get(key, 0)) + value
                elif isinstance(value, float):
                    bucket[key] = float(bucket.get(key, 0.0)) + value
            if "failure_count" in bucket:
                bucket["path_planning_failure_count"] = bucket["failure_count"]
    return dict(sorted(aggregate.items()))


def _aggregate_control_point_artifacts(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    artifact_roots: list[str] = []
    for summary in summaries:
        artifact_index = summary.get("gcs_control_point_candidate_artifacts")
        if not isinstance(artifact_index, dict):
            continue
        artifact_root = artifact_index.get("artifact_root")
        if artifact_root:
            artifact_roots.append(str(artifact_root))
        raw_entries = artifact_index.get("entries")
        if isinstance(raw_entries, list):
            entries.extend(entry for entry in raw_entries if isinstance(entry, dict))
    return {
        "schema_version": GCS_CONTROL_POINT_CANDIDATE_ARTIFACT_INDEX_SCHEMA_VERSION,
        "artifact_roots": sorted(artifact_roots),
        "candidate_count": len(entries),
        "route_artifact_count": sum(1 for entry in entries if entry.get("route_artifact")),
        "entries": entries,
    }


def _aggregate_control_point_triage(
    summaries: list[dict[str, Any]],
    *,
    route_artifact_count: int,
) -> dict[str, Any]:
    fallback_reason_counts: Counter[str] = Counter()
    terrain_source_counts: Counter[str] = Counter()
    blocker_class_counts: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    attempted_count = 0
    success_count = 0
    selected_count = 0
    for summary in summaries:
        triage = summary.get("gcs_control_point_candidate_triage")
        if not isinstance(triage, dict):
            continue
        attempted_count += _int_or_zero(triage.get("attempted_count"))
        success_count += _int_or_zero(triage.get("success_count"))
        selected_count += _int_or_zero(triage.get("selected_count"))
        fallback_reason_counts.update(_counter_payload(triage.get("fallback_reason_counts")))
        terrain_source_counts.update(_counter_payload(triage.get("terrain_objective_source_counts")))
        blocker_class_counts.update(_counter_payload(triage.get("blocker_class_counts")))
        raw_candidates = triage.get("candidates")
        if isinstance(raw_candidates, list):
            candidates.extend(item for item in raw_candidates if isinstance(item, dict))
    return {
        "schema_version": GCS_CONTROL_POINT_CANDIDATE_TRIAGE_SCHEMA_VERSION,
        "candidate_count": len(candidates),
        "attempted_count": attempted_count,
        "success_count": success_count,
        "selected_count": selected_count,
        "route_artifact_count": route_artifact_count,
        "fallback_reason_counts": dict(sorted(fallback_reason_counts.items())),
        "terrain_objective_source_counts": dict(sorted(terrain_source_counts.items())),
        "blocker_class_counts": dict(sorted(blocker_class_counts.items())),
        "sampled_terrain_cost": _aggregate_nested_metric_stats(
            summaries,
            "gcs_control_point_candidate_triage",
            "sampled_terrain_cost",
        ),
        "high_cost_exposure_delta_vs_baseline": _aggregate_nested_metric_stats(
            summaries,
            "gcs_control_point_candidate_triage",
            "high_cost_exposure_delta_vs_baseline",
        ),
        "calibration_sweep": _control_point_calibration_sweep(
            candidates,
            fallback_reason_counts=fallback_reason_counts,
        ),
        "candidates": candidates,
    }


def _control_point_calibration_sweep(
    candidates: list[dict[str, Any]],
    *,
    fallback_reason_counts: Counter[str],
) -> dict[str, Any]:
    quality_blocked = [
        candidate
        for candidate in candidates
        if candidate.get("candidate_fallback_reason") == "cost_dominated"
    ]
    direction_blocked = [
        candidate
        for candidate in candidates
        if candidate.get("candidate_fallback_reason") == "direction_cone_constraint_violation"
        or _int_or_zero(candidate.get("direction_cone_violation_count")) > 0
        or bool(candidate.get("direction_cone_risk_flags"))
    ]
    unsupported = [
        candidate
        for candidate in candidates
        if candidate.get("candidate_fallback_reason") == "unsupported_route_replacement"
    ]
    conservative_gate_candidates = [
        candidate
        for candidate in quality_blocked
        if _non_positive_summary_number(candidate.get("cost_delta_vs_baseline"))
        and _non_positive_summary_number(candidate.get("high_cost_exposure_delta_vs_baseline"))
        and candidate.get("motion_feasibility_status") in {None, "feasible"}
        and _int_or_zero(candidate.get("direction_cone_violation_count")) == 0
        and not candidate.get("direction_cone_risk_flags")
    ]
    default_change_reason = "no_control_point_candidates_reported"
    if candidates:
        default_change_reason = "requires_solver_rerun_and_no_safety_diagnostic_degradation"
        if quality_blocked or direction_blocked:
            default_change_reason = "recorded_candidates_remain_blocked_by_quality_or_direction_cone_gate"
    return {
        "schema_version": GCS_CONTROL_POINT_CANDIDATE_CALIBRATION_SWEEP_SCHEMA_VERSION,
        "mode": "recorded_candidate_gate_diagnostics",
        "solver_rerun_required": True,
        "default_change_recommended": False,
        "default_change_reason": default_change_reason,
        "sweep_dimensions": [
            "terrain_objective_weight",
            "control_point_second_difference_quadratic_weight",
            "direction_cone_rho_eta_tolerance",
            "quality_gate_thresholds",
        ],
        "observed_current_values": {
            "terrain_objective_weight": _unique_summary_numbers(candidates, "terrain_objective_weight"),
            "second_difference_weight": _unique_summary_numbers(candidates, "second_difference_weight"),
            "direction_cone_eta": _unique_summary_numbers(candidates, "direction_cone_eta"),
            "direction_cone_rho_min": _unique_summary_numbers(candidates, "direction_cone_rho_min"),
            "direction_cone_tolerance_deg": _unique_summary_numbers(
                candidates,
                "direction_cone_tolerance_deg",
            ),
            "direction_cone_rho_source_counts": _aggregate_candidate_counter(
                candidates,
                "direction_cone_rho_source_counts",
            ),
        },
        "candidate_gate_outcomes": {
            "quality_gate_blocked_count": len(quality_blocked),
            "direction_cone_blocked_count": len(direction_blocked),
            "expected_not_evaluated_count": len(unsupported),
            "fallback_reason_counts": dict(sorted(fallback_reason_counts.items())),
            "conservative_gate_relaxation_candidate_count": len(conservative_gate_candidates),
            "unsafe_or_unproven_quality_relaxation_count": max(
                0,
                len(quality_blocked) - len(conservative_gate_candidates),
            ),
        },
        "safety_regression_guard": {
            "terrain_cost_degradation_allowed": False,
            "high_cost_exposure_degradation_allowed": False,
            "collision_degradation_allowed": False,
            "direction_cone_degradation_allowed": False,
            "motion_diagnostic_degradation_allowed": False,
            "default_gate_relaxation_allowed_without_evidence": False,
        },
        "next_solver_rerun_matrix": [
            {
                "dimension": "terrain_objective_weight",
                "target_blocker": "cost_dominated",
                "acceptance": "lower_sampled_terrain_cost_and_high_cost_exposure_without_collision_or_direction_cone_regression",
            },
            {
                "dimension": "control_point_second_difference_quadratic_weight",
                "target_blocker": "cost_dominated_or_motion_diagnostic_regression",
                "acceptance": "smoother_control_points_without_region_or_motion_feasibility_regression",
            },
            {
                "dimension": "direction_cone_rho_eta_tolerance",
                "target_blocker": "direction_cone_constraint_violation",
                "acceptance": "fewer_direction_cone_violations_without_motion_or_collision_regression",
            },
            {
                "dimension": "quality_gate_thresholds",
                "target_blocker": "candidate_selected_count_zero",
                "acceptance": "selected_count_can_increase_only_when_cost_and_safety_metrics_do_not_degrade",
            },
        ],
    }


def _non_positive_summary_number(value: Any) -> bool:
    number = _summary_float(value)
    return number is not None and number <= 0.0


def _unique_summary_numbers(candidates: list[dict[str, Any]], key: str) -> list[float]:
    values = {_summary_float(candidate.get(key)) for candidate in candidates}
    return sorted(value for value in values if value is not None)


def _aggregate_candidate_counter(candidates: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for candidate in candidates:
        payload = candidate.get(key)
        if not isinstance(payload, dict):
            continue
        counts.update(_counter_payload(payload))
    return dict(sorted(counts.items()))


def _int_or_zero(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _counter_payload(value: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not isinstance(value, dict):
        return counts
    for key, count in value.items():
        if isinstance(count, int) and not isinstance(count, bool):
            counts[str(key)] += count
    return counts


def _sum_summary_int(summaries: list[dict[str, Any]], key: str) -> int:
    total = 0
    for summary in summaries:
        value = summary.get(key, 0)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            total += value
    return total


def _aggregate_summary_counter(summaries: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for summary in summaries:
        payload = summary.get(key, {})
        if not isinstance(payload, dict):
            continue
        for item_key, value in payload.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                counts[str(item_key)] += value
    return dict(sorted(counts.items()))


def _aggregate_summary_list(summaries: list[dict[str, Any]], key: str) -> list[Any]:
    items: list[Any] = []
    for summary in summaries:
        payload = summary.get(key, [])
        if isinstance(payload, list):
            items.extend(payload)
    return items


def _aggregate_nested_metric_stats(
    summaries: list[dict[str, Any]],
    parent_key: str,
    metric_key: str,
) -> dict[str, Any]:
    count = 0
    weighted_total = 0.0
    minimum: float | None = None
    maximum: float | None = None
    for summary in summaries:
        parent = summary.get(parent_key)
        if not isinstance(parent, dict):
            continue
        metric = parent.get(metric_key)
        if not isinstance(metric, dict):
            continue
        item_count = metric.get("count", 0)
        if isinstance(item_count, bool) or not isinstance(item_count, int) or item_count <= 0:
            continue
        item_mean = _summary_float(metric.get("mean"))
        if item_mean is None:
            continue
        count += item_count
        weighted_total += item_mean * item_count
        item_min = _summary_float(metric.get("min"))
        item_max = _summary_float(metric.get("max"))
        if item_min is not None:
            minimum = item_min if minimum is None else min(minimum, item_min)
        if item_max is not None:
            maximum = item_max if maximum is None else max(maximum, item_max)
    if count == 0:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": count,
        "min": minimum,
        "max": maximum,
        "mean": weighted_total / count,
    }


def _aggregate_summary_metric_stats(summaries: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    count_key = f"{prefix}_count"
    min_key = f"{prefix}_min"
    max_key = f"{prefix}_max"
    mean_key = f"{prefix}_mean"
    count = 0
    weighted_total = 0.0
    minimum: float | None = None
    maximum: float | None = None
    for summary in summaries:
        item_count = summary.get(count_key, 0)
        if isinstance(item_count, bool) or not isinstance(item_count, int) or item_count <= 0:
            continue
        item_mean = _summary_float(summary.get(mean_key))
        if item_mean is None:
            continue
        count += item_count
        weighted_total += item_mean * item_count
        item_min = _summary_float(summary.get(min_key))
        item_max = _summary_float(summary.get(max_key))
        if item_min is not None:
            minimum = item_min if minimum is None else min(minimum, item_min)
        if item_max is not None:
            maximum = item_max if maximum is None else max(maximum, item_max)
    if count == 0:
        return {
            count_key: 0,
            min_key: None,
            max_key: None,
            mean_key: None,
        }
    return {
        count_key: count,
        min_key: minimum,
        max_key: maximum,
        mean_key: float(weighted_total / count),
    }


def _summary_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _sum_region_graph_disconnected(summaries: list[dict[str, Any]]) -> int:
    total = _sum_summary_int(summaries, "region_graph_disconnected_count")
    if total:
        return total
    return _sum_summary_int(summaries, "region_graph_start_goal_disconnected_count")


def _append_reason(reason_codes: list[str], code: str) -> None:
    if code not in reason_codes:
        reason_codes.append(code)


def _resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
