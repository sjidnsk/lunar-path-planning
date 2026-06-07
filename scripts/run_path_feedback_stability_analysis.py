from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_INDEX_SCHEMA_VERSION = "path-feedback-batch-run-index/v1"
EVALUATION_SUMMARY_SCHEMA_VERSION = "path-feedback-batch-evaluation-summary/v1"
PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION = "path-feedback-summary/v1"
BATCH_STABILITY_SCHEMA_VERSION = "batch-stability-summary/v1"
DATASET_QUALITY_STABILITY_SCHEMA_VERSION = "dataset-quality-stability-summary/v1"
DECISION_STABILITY_SCHEMA_VERSION = "decision-stability-summary/v1"
SUBMODULES = ("dev-platform-constraints", "model-explorer", "path-planner")
HARD_EXCLUDE_REASON_CODES = {"open_grid_fallback"}
DEFAULT_DOWNWEIGHT_REASON_CODES = {
    "path_planning_failure",
    "replan_required",
    "iris_fallback",
    "region_graph_disconnected",
    "region_graph_fallback",
}
CHANNEL_AWARE_COUNT_METRIC_KEYS = (
    "channel_aware_astar_report_count",
    "channel_aware_astar_selected_count",
    "channel_aware_astar_fallback_count",
    "channel_aware_astar_path_changed_count",
)
CHANNEL_AWARE_EVIDENCE_KEYS = (
    "channel_aware_astar_report_count",
    "channel_aware_astar_selected_count",
    "channel_aware_astar_fallback_count",
    "channel_aware_astar_requested_backend_counts",
    "channel_aware_astar_selected_backend_counts",
    "channel_aware_astar_status_counts",
    "channel_aware_astar_fallback_reason_counts",
    "channel_aware_astar_blocker_class_counts",
    "channel_aware_astar_path_changed_count",
    "channel_aware_astar_path_changed_rate",
    "channel_aware_astar_path_cost_delta_count",
    "channel_aware_astar_path_cost_delta_min",
    "channel_aware_astar_path_cost_delta_max",
    "channel_aware_astar_path_cost_delta_mean",
    "channel_aware_astar_channel_cost_delta_count",
    "channel_aware_astar_channel_cost_delta_min",
    "channel_aware_astar_channel_cost_delta_max",
    "channel_aware_astar_channel_cost_delta_mean",
    "channel_aware_astar_high_cost_exposure_delta_count",
    "channel_aware_astar_high_cost_exposure_delta_min",
    "channel_aware_astar_high_cost_exposure_delta_max",
    "channel_aware_astar_high_cost_exposure_delta_mean",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze Dataset / Decision Stability v1 from a path-feedback batch root."
    )
    parser.add_argument("--batch-root", required=True, help="Batch output root containing batch-run-index.json.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    batch_root = _resolve_path(args.batch_root, repo_root)
    batch_root.mkdir(parents=True, exist_ok=True)

    outputs = analyze_batch_stability(batch_root=batch_root, repo_root=repo_root)
    _write_json(batch_root / "batch-stability-summary.json", outputs["batch"])
    _write_json(batch_root / "dataset-quality-stability-summary.json", outputs["dataset"])
    _write_json(batch_root / "decision-stability-summary.json", outputs["decision"])

    print(
        json.dumps(
            {
                "status": outputs["batch"]["status"],
                "run_count": outputs["batch"]["run_count"],
                "failure_reason_code_counts": outputs["batch"]["failure_reason_code_counts"],
                "batch_stability_summary": _display_path(
                    batch_root / "batch-stability-summary.json",
                    repo_root,
                ),
                "dataset_quality_stability_summary": _display_path(
                    batch_root / "dataset-quality-stability-summary.json",
                    repo_root,
                ),
                "decision_stability_summary": _display_path(
                    batch_root / "decision-stability-summary.json",
                    repo_root,
                ),
            },
            ensure_ascii=False,
        )
    )
    return 1 if outputs["batch"]["status"] == "failed" else 0


def analyze_batch_stability(*, batch_root: Path, repo_root: Path) -> dict[str, dict[str, Any]]:
    index_path = batch_root / "batch-run-index.json"
    evaluation_path = batch_root / "batch-evaluation-summary.json"
    global_reason_codes: list[str] = []
    run_index = _load_json_object(index_path, global_reason_codes, "batch_run_index")
    evaluation_summary = _load_json_object(
        evaluation_path,
        global_reason_codes,
        "batch_evaluation_summary",
    )

    if run_index.get("schema_version") != RUN_INDEX_SCHEMA_VERSION:
        _append_reason(global_reason_codes, "batch_run_index_schema_mismatch")
    if evaluation_summary.get("schema_version") != EVALUATION_SUMMARY_SCHEMA_VERSION:
        _append_reason(global_reason_codes, "batch_evaluation_summary_schema_mismatch")

    runs = run_index.get("runs", [])
    if not isinstance(runs, list):
        _append_reason(global_reason_codes, "batch_run_index_runs_invalid")
        runs = []

    evaluation_source_paths = _string_set(evaluation_summary.get("source_summary_paths", []))
    batch_git = run_index.get("git") if isinstance(run_index.get("git"), dict) else {}
    git_reason_codes = _validate_git_snapshot(batch_git, prefix="batch_git")
    for reason in git_reason_codes:
        _append_reason(global_reason_codes, reason)

    run_records = [
        _analyze_run(
            run,
            batch_git=batch_git,
            evaluation_source_paths=evaluation_source_paths,
            repo_root=repo_root,
        )
        for run in runs
        if isinstance(run, dict)
    ]
    if len(run_records) != len(runs):
        _append_reason(global_reason_codes, "batch_run_index_run_not_object")

    status = "failed" if global_reason_codes or any(record["reason_codes"] for record in run_records) else "passed"
    batch_summary = _build_batch_stability_summary(
        status=status,
        global_reason_codes=global_reason_codes,
        run_records=run_records,
        run_index=run_index,
        evaluation_summary=evaluation_summary,
        batch_root=batch_root,
        index_path=index_path,
        evaluation_path=evaluation_path,
        repo_root=repo_root,
    )
    dataset_summary = _build_dataset_quality_stability_summary(
        status=status,
        global_reason_codes=global_reason_codes,
        run_records=run_records,
        batch_root=batch_root,
        repo_root=repo_root,
    )
    decision_summary = _build_decision_stability_summary(
        status=status,
        global_reason_codes=global_reason_codes,
        run_records=run_records,
        batch_root=batch_root,
        repo_root=repo_root,
    )
    return {"batch": batch_summary, "dataset": dataset_summary, "decision": decision_summary}


def _load_json_object(path: Path, reason_codes: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _append_reason(reason_codes, f"{label}_invalid_json")
        return {}
    if not isinstance(payload, dict):
        _append_reason(reason_codes, f"{label}_not_object")
        return {}
    return payload


def _analyze_run(
    run: dict[str, Any],
    *,
    batch_git: dict[str, Any],
    evaluation_source_paths: set[str],
    repo_root: Path,
) -> dict[str, Any]:
    run_id = str(run.get("run_id", "unknown"))
    reason_codes: list[str] = []
    original_reason_codes = _string_list(run.get("reason_codes", []))
    if run.get("status") != "passed" or original_reason_codes:
        _append_reason(reason_codes, "batch_run_failed")
        for reason in original_reason_codes:
            _append_reason(reason_codes, reason)

    command_args = run.get("command_args")
    if not isinstance(command_args, dict):
        command_args = {}
        _append_reason(reason_codes, "command_args_missing")

    source_summary_value = _first_non_empty(
        _nested_string(run, ("source_paths", "summary")),
        run.get("summary_path"),
        default="",
    )
    summary_path = _resolve_path(source_summary_value, repo_root) if source_summary_value else None
    source_summary_display = (
        _display_path(summary_path, repo_root)
        if summary_path is not None
        else str(source_summary_value)
    )
    summary = _load_path_feedback_summary(summary_path, reason_codes) if summary_path is not None else {}
    if not source_summary_value:
        _append_reason(reason_codes, "source_summary_path_missing")
    elif evaluation_source_paths and source_summary_value not in evaluation_source_paths and source_summary_display not in evaluation_source_paths:
        _append_reason(reason_codes, "source_summary_not_listed_in_batch_evaluation")

    if summary.get("schema_version") != PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION and summary:
        _append_reason(reason_codes, "source_summary_schema_mismatch")

    acceptance_metadata = summary.get("acceptance_metadata") if isinstance(summary.get("acceptance_metadata"), dict) else {}
    run_acceptance_metadata = run.get("acceptance_metadata") if isinstance(run.get("acceptance_metadata"), dict) else {}
    mismatches = _acceptance_metadata_mismatches(
        command_args=command_args,
        run_acceptance_metadata=run_acceptance_metadata,
        summary=summary,
        summary_acceptance_metadata=acceptance_metadata,
    )
    if not acceptance_metadata and summary:
        _append_reason(reason_codes, "acceptance_metadata_missing")
    if mismatches:
        _append_reason(reason_codes, "acceptance_metadata_mismatch")

    open_grid_used = _open_grid_fallback_used(summary, acceptance_metadata)
    if open_grid_used is True:
        _append_reason(reason_codes, "open_grid_fallback_used")
    gate = acceptance_metadata.get("open_grid_fallback_used_gate") if isinstance(acceptance_metadata, dict) else None
    if not isinstance(gate, dict):
        gate = summary.get("open_grid_fallback_used_gate")
    if summary and not isinstance(gate, dict):
        _append_reason(reason_codes, "open_grid_fallback_gate_missing")
    elif isinstance(gate, dict):
        if gate.get("status") != "passed":
            _append_reason(reason_codes, "open_grid_fallback_gate_failed")
        if gate.get("actual") is not False:
            _append_reason(reason_codes, "open_grid_fallback_used")

    run_git = run.get("git") if isinstance(run.get("git"), dict) else {}
    for reason in _validate_git_snapshot(run_git, prefix="run_git"):
        _append_reason(reason_codes, reason)
    if batch_git and run_git and not _git_snapshots_match(batch_git, run_git):
        _append_reason(reason_codes, "git_provenance_mismatch")

    metrics = _summary_metrics(summary)
    return {
        "run_id": run_id,
        "status": "failed" if reason_codes else "passed",
        "batch_run_status": str(run.get("status", "unknown")),
        "reason_codes": reason_codes,
        "batch_reason_codes": original_reason_codes,
        "command_args": dict(command_args),
        "source_summary_path": source_summary_display,
        "report_path": str(_first_non_empty(_nested_string(run, ("source_paths", "report")), run.get("report_path"), default="")),
        "acceptance_metadata": dict(acceptance_metadata),
        "run_acceptance_metadata": dict(run_acceptance_metadata),
        "acceptance_metadata_mismatches": mismatches,
        "open_grid_fallback_used": open_grid_used,
        "metrics": metrics,
        "scenario_group_summary": _scenario_group_summary(summary),
        "summary": summary if isinstance(summary, dict) else {},
        "git": run_git,
    }


def _load_path_feedback_summary(path: Path | None, reason_codes: list[str]) -> dict[str, Any]:
    if path is None or not path.is_file():
        _append_reason(reason_codes, "source_summary_missing")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _append_reason(reason_codes, "source_summary_invalid_json")
        return {}
    if not isinstance(payload, dict):
        _append_reason(reason_codes, "source_summary_not_object")
        return {}
    return payload


def _acceptance_metadata_mismatches(
    *,
    command_args: dict[str, Any],
    run_acceptance_metadata: dict[str, Any],
    summary: dict[str, Any],
    summary_acceptance_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for field in ("scenario_set", "diagnostic_profile", "top_k"):
        expected = command_args.get(field)
        if expected is None:
            continue
        summary_metadata_value = summary_acceptance_metadata.get(field)
        summary_value = summary.get(field)
        if str(summary_metadata_value) != str(expected) or str(summary_value) != str(expected):
            mismatches.append(
                {
                    "field": field,
                    "expected": expected,
                    "summary": summary_value,
                    "acceptance_metadata": summary_metadata_value,
                }
            )
        if run_acceptance_metadata and str(run_acceptance_metadata.get(field)) != str(summary_metadata_value):
            mismatches.append(
                {
                    "field": f"run_acceptance_metadata.{field}",
                    "expected": run_acceptance_metadata.get(field),
                    "actual": summary_metadata_value,
                }
            )
    expected_open_grid = False
    actual_open_grid = _open_grid_fallback_used(summary, summary_acceptance_metadata)
    if actual_open_grid is not None and bool(actual_open_grid) is not expected_open_grid:
        mismatches.append(
            {
                "field": "open_grid_fallback_used",
                "expected": expected_open_grid,
                "actual": actual_open_grid,
            }
        )
    return mismatches


def _open_grid_fallback_used(summary: dict[str, Any], acceptance_metadata: dict[str, Any]) -> bool | None:
    values = [
        summary.get("open_grid_fallback_used"),
        acceptance_metadata.get("open_grid_fallback_used"),
    ]
    gate = acceptance_metadata.get("open_grid_fallback_used_gate")
    if isinstance(gate, dict):
        values.append(gate.get("actual"))
    summary_gate = summary.get("open_grid_fallback_used_gate")
    if isinstance(summary_gate, dict):
        values.append(summary_gate.get("actual"))
    for value in values:
        if value is True:
            return True
    for value in values:
        if value is False:
            return False
    return None


def _summary_metrics(summary: dict[str, Any]) -> dict[str, int]:
    metrics = {
        "scenario_count": _int_value(summary.get("scenario_count")),
        "candidate_count": _int_value(summary.get("candidate_count")),
        "open_grid_fallback_used_count": 1 if summary.get("open_grid_fallback_used") is True else 0,
        "path_planning_failure_count": _int_value(summary.get("path_planning_failure_count")),
        "replan_count": _int_value(summary.get("replan_count")),
        "iris_fallback_count": _int_value(summary.get("iris_fallback_count")),
        "region_graph_fallback_count": _int_value(summary.get("region_graph_fallback_count")),
        "region_graph_disconnected_count": _region_graph_disconnected_count(summary),
    }
    for key in CHANNEL_AWARE_COUNT_METRIC_KEYS:
        metrics[key] = _int_value(summary.get(key))
    return metrics


def _scenario_group_summary(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_groups = summary.get("scenario_group_summary")
    if not isinstance(raw_groups, dict):
        return {}
    groups: dict[str, dict[str, Any]] = {}
    for group_name, payload in raw_groups.items():
        if not isinstance(payload, dict):
            continue
        group_metrics = {
            "scenario_count": _int_value(payload.get("scenario_count")),
            "candidate_count": _int_value(payload.get("candidate_count")),
            "open_grid_fallback_used_count": 1 if summary.get("open_grid_fallback_used") is True else 0,
            "path_planning_failure_count": _int_value(payload.get("path_planning_failure_count", payload.get("failure_count"))),
            "replan_count": _int_value(payload.get("replan_count")),
            "iris_fallback_count": _int_value(payload.get("iris_fallback_count")),
            "region_graph_fallback_count": _int_value(payload.get("region_graph_fallback_count")),
            "region_graph_disconnected_count": _int_value(
                payload.get("region_graph_disconnected_count", payload.get("region_graph_start_goal_disconnected_count"))
            ),
        }
        for key in CHANNEL_AWARE_COUNT_METRIC_KEYS:
            group_metrics[key] = _int_value(payload.get(key))
        extras = {
            key: value
            for key, value in payload.items()
            if key not in group_metrics and isinstance(value, int) and not isinstance(value, bool)
        }
        groups[str(group_name)] = {**extras, **group_metrics}
    return dict(sorted(groups.items()))


def _build_batch_stability_summary(
    *,
    status: str,
    global_reason_codes: list[str],
    run_records: list[dict[str, Any]],
    run_index: dict[str, Any],
    evaluation_summary: dict[str, Any],
    batch_root: Path,
    index_path: Path,
    evaluation_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    by_run = {record["run_id"]: _batch_run_public_record(record) for record in run_records}
    reason_code_counts = Counter(global_reason_codes)
    for record in run_records:
        reason_code_counts.update(record["reason_codes"])

    source_summary_paths = [
        record["source_summary_path"]
        for record in run_records
        if record["source_summary_path"]
    ]
    return {
        "schema_version": BATCH_STABILITY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(global_reason_codes),
        "failure_reason_code_counts": dict(sorted(reason_code_counts.items())),
        "batch_root": _display_path(batch_root, repo_root),
        "source_paths": {
            "batch_run_index": _display_path(index_path, repo_root),
            "batch_evaluation_summary": _display_path(evaluation_path, repo_root),
            "source_summary_paths": source_summary_paths,
        },
        "run_count": len(run_records),
        "passed_count": sum(1 for record in run_records if record["status"] == "passed"),
        "failed_count": sum(1 for record in run_records if record["status"] == "failed"),
        "batch_run_index_counts": _index_counts(run_index),
        "batch_evaluation_counts": _evaluation_counts(evaluation_summary),
        "channel_aware_astar_evidence": _channel_aware_astar_evidence(evaluation_summary),
        "git_provenance": {
            "batch": _public_git_snapshot(run_index.get("git", {})),
            "current": _git_snapshot(repo_root),
            "runs_match_batch": all(
                _git_snapshots_match(run_index.get("git", {}), record.get("git", {}))
                for record in run_records
                if run_index.get("git")
            ),
        },
        "by_run": by_run,
        "by_scenario_set": _aggregate_runs_by_dimension(run_records, ("command_args", "scenario_set")),
        "by_diagnostic_profile": _aggregate_runs_by_dimension(run_records, ("command_args", "diagnostic_profile")),
        "by_top_k": _aggregate_runs_by_dimension(run_records, ("command_args", "top_k")),
        "by_scenario_group": _aggregate_scenario_groups(run_records),
        "quality_signal_use": "stability_analysis_only",
        "not_real_world_performance_claim": True,
    }


def _batch_run_public_record(record: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(record["metrics"])
    return {
        "run_id": record["run_id"],
        "status": record["status"],
        "batch_run_status": record["batch_run_status"],
        "reason_codes": list(record["reason_codes"]),
        "batch_reason_codes": list(record["batch_reason_codes"]),
        "command_args": dict(record["command_args"]),
        "source_summary_path": record["source_summary_path"],
        "report_path": record["report_path"],
        "acceptance_metadata": dict(record["acceptance_metadata"]),
        "acceptance_metadata_mismatches": list(record["acceptance_metadata_mismatches"]),
        "open_grid_fallback_used": record["open_grid_fallback_used"],
        "scenario_group_summary": dict(record["scenario_group_summary"]),
        **metrics,
    }


def _aggregate_runs_by_dimension(records: list[dict[str, Any]], key_path: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for record in records:
        key = _nested_value(record, key_path)
        _add_run_to_bucket(buckets, str(key if key is not None else "unknown"), record, record["metrics"])
    return dict(sorted(buckets.items()))


def _aggregate_scenario_groups(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for record in records:
        for group_name, metrics in record["scenario_group_summary"].items():
            _add_run_to_bucket(buckets, group_name, record, metrics)
    return dict(sorted(buckets.items()))


def _add_run_to_bucket(
    buckets: dict[str, dict[str, Any]],
    key: str,
    record: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    bucket = buckets.setdefault(key, _empty_metric_bucket())
    bucket["run_count"] += 1
    if record["status"] == "passed":
        bucket["passed_run_count"] += 1
    else:
        bucket["failed_run_count"] += 1
    _append_unique(bucket["run_ids"], record["run_id"])
    _append_unique(bucket["source_summary_paths"], record["source_summary_path"])
    for reason in record["reason_codes"]:
        bucket["reason_code_counts"][reason] = bucket["reason_code_counts"].get(reason, 0) + 1
    for metric_key in (
        "scenario_count",
        "candidate_count",
        "open_grid_fallback_used_count",
        "path_planning_failure_count",
        "replan_count",
        "iris_fallback_count",
        "region_graph_fallback_count",
        "region_graph_disconnected_count",
        *CHANNEL_AWARE_COUNT_METRIC_KEYS,
    ):
        bucket[metric_key] += _int_value(metrics.get(metric_key))


def _empty_metric_bucket() -> dict[str, Any]:
    return {
        "run_count": 0,
        "passed_run_count": 0,
        "failed_run_count": 0,
        "run_ids": [],
        "source_summary_paths": [],
        "reason_code_counts": {},
        "scenario_count": 0,
        "candidate_count": 0,
        "open_grid_fallback_used_count": 0,
        "path_planning_failure_count": 0,
        "replan_count": 0,
        "iris_fallback_count": 0,
        "region_graph_fallback_count": 0,
        "region_graph_disconnected_count": 0,
        **{key: 0 for key in CHANNEL_AWARE_COUNT_METRIC_KEYS},
    }


def _build_dataset_quality_stability_summary(
    *,
    status: str,
    global_reason_codes: list[str],
    run_records: list[dict[str, Any]],
    batch_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    records = []
    for run_record in run_records:
        summary = run_record.get("summary", {})
        if not isinstance(summary, dict):
            continue
        for scenario in summary.get("scenarios", []):
            if isinstance(scenario, dict):
                records.append(_dataset_quality_record(run_record, scenario))
    action_counts = Counter(record["action"] for record in records)
    return {
        "schema_version": DATASET_QUALITY_STABILITY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(global_reason_codes),
        "batch_root": _display_path(batch_root, repo_root),
        "quality_signal_use": "stability_analysis_only",
        "not_real_world_performance_claim": True,
        "record_count": len(records),
        "excluded_record_count": int(action_counts.get("exclude", 0)),
        "downweighted_record_count": int(action_counts.get("downweight", 0)),
        "kept_record_count": int(action_counts.get("keep", 0)),
        "by_scenario_id": _aggregate_quality_records_by_field(records, "scenario_id"),
        "by_scenario_group": _aggregate_quality_records_by_field(records, "scenario_group"),
        "by_reason_code": _aggregate_quality_records_by_reason_code(records),
        "by_action": _aggregate_quality_records_by_field(records, "action"),
        "records": records,
    }


def _dataset_quality_record(run_record: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    reason_codes = _scenario_quality_reason_codes(scenario)
    action, sample_weight = _quality_action(reason_codes)
    command_args = run_record["command_args"]
    return {
        "run_id": run_record["run_id"],
        "scenario_id": str(scenario.get("scenario_id", "")),
        "scenario_group": str(scenario.get("scenario_group", "unknown")),
        "scenario_set": str(command_args.get("scenario_set", "unknown")),
        "diagnostic_profile": str(command_args.get("diagnostic_profile", "unknown")),
        "top_k": command_args.get("top_k"),
        "source_summary_path": run_record["source_summary_path"],
        "acceptance_metadata": dict(run_record["acceptance_metadata"]),
        "reason_codes": reason_codes,
        "action": action,
        "sample_weight": sample_weight,
        "quality_signal_use": "stability_analysis_only",
        "not_real_world_performance_claim": True,
    }


def _scenario_quality_reason_codes(scenario: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    if scenario.get("open_grid_fallback_used") is True:
        codes.append("open_grid_fallback")
    feedback = scenario.get("path_feedback")
    if isinstance(feedback, dict):
        if _int_value(feedback.get("failure_count")) > 0:
            codes.append("path_planning_failure")
        if _int_value(feedback.get("replan_count")) > 0:
            codes.append("replan_required")
        for candidate in feedback.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            if candidate.get("failure_reason"):
                codes.append("path_planning_failure")
            if candidate.get("replan_required") is True:
                codes.append("replan_required")
            interpretation = candidate.get("diagnostic_interpretation")
            if isinstance(interpretation, dict):
                codes.extend(_normalize_reason(flag) for flag in _string_list(interpretation.get("diagnostic_flags", [])))
                if interpretation.get("open_grid_fallback_used") is True:
                    codes.append("open_grid_fallback")
                if interpretation.get("iris_fallback_used") is True:
                    codes.append("iris_fallback")
                if interpretation.get("region_graph_fallback_used") is True:
                    codes.append("region_graph_fallback")
                if interpretation.get("region_graph_start_goal_connected") is False:
                    codes.append("region_graph_disconnected")
    interpretation = scenario.get("diagnostic_interpretation")
    if isinstance(interpretation, dict):
        codes.extend(_normalize_reason(source) for source in _string_list(interpretation.get("failure_sources", [])))
        if interpretation.get("open_grid_fallback_used") is True:
            codes.append("open_grid_fallback")
    iris_diagnostics = scenario.get("iris_diagnostics")
    if isinstance(iris_diagnostics, dict) and _int_value(iris_diagnostics.get("fallback_count")) > 0:
        codes.append("iris_fallback")
    region_graph_diagnostics = scenario.get("region_graph_diagnostics")
    if isinstance(region_graph_diagnostics, dict):
        if _int_value(region_graph_diagnostics.get("fallback_count")) > 0:
            codes.append("region_graph_fallback")
        if _int_value(region_graph_diagnostics.get("start_goal_disconnected_count")) > 0:
            codes.append("region_graph_disconnected")
    return _dedupe(code for code in codes if code and code != "none") or ["sample_quality_passed"]


def _quality_action(reason_codes: list[str]) -> tuple[str, float]:
    reason_set = set(reason_codes)
    if HARD_EXCLUDE_REASON_CODES.intersection(reason_set):
        return "exclude", 0.0
    if DEFAULT_DOWNWEIGHT_REASON_CODES.intersection(reason_set):
        return "downweight", 0.5
    return "keep", 1.0


def _aggregate_quality_records_by_field(records: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(str(record.get(field, "unknown")), []).append(record)
    return {
        key: _quality_bucket(bucket_records)
        for key, bucket_records in sorted(buckets.items())
    }


def _aggregate_quality_records_by_reason_code(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        for reason in record["reason_codes"]:
            buckets.setdefault(reason, []).append(record)
    return {
        key: _quality_bucket(bucket_records)
        for key, bucket_records in sorted(buckets.items())
    }


def _quality_bucket(records: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts = Counter(record["action"] for record in records)
    reason_code_counts = Counter(
        reason
        for record in records
        for reason in record["reason_codes"]
    )
    reason_sets = [set(record["reason_codes"]) for record in records]
    stable_reason_codes = set.intersection(*reason_sets) if reason_sets else set()
    all_reason_codes = set.union(*reason_sets) if reason_sets else set()
    return {
        "record_count": len(records),
        "action_counts": dict(sorted(action_counts.items())),
        "reason_code_counts": dict(sorted(reason_code_counts.items())),
        "scenario_ids": _unique(record["scenario_id"] for record in records),
        "run_ids": _unique(record["run_id"] for record in records),
        "source_summary_paths": _unique(record["source_summary_path"] for record in records),
        "stable_action": len(action_counts) <= 1,
        "stable_reason_codes": sorted(stable_reason_codes),
        "varying_reason_codes": sorted(all_reason_codes - stable_reason_codes),
    }


def _build_decision_stability_summary(
    *,
    status: str,
    global_reason_codes: list[str],
    run_records: list[dict[str, Any]],
    batch_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    records = []
    for run_record in run_records:
        summary = run_record.get("summary", {})
        if not isinstance(summary, dict):
            continue
        for scenario in summary.get("scenarios", []):
            if isinstance(scenario, dict):
                records.append(_decision_record(run_record, scenario))
    by_scenario_id = _aggregate_decision_records(records)
    stable_scenario_count = sum(
        1
        for bucket in by_scenario_id.values()
        if bucket["stable_selection_changed"]
        and bucket["stable_target_replacement_reason"]
        and bucket["stable_failure_sources"]
    )
    return {
        "schema_version": DECISION_STABILITY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(global_reason_codes),
        "batch_root": _display_path(batch_root, repo_root),
        "quality_signal_use": "stability_analysis_only",
        "not_real_world_performance_claim": True,
        "record_count": len(records),
        "scenario_count": len(by_scenario_id),
        "stable_scenario_count": stable_scenario_count,
        "unstable_scenario_count": len(by_scenario_id) - stable_scenario_count,
        "by_scenario_id": by_scenario_id,
        "records": records,
    }


def _decision_record(run_record: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    command_args = run_record["command_args"]
    interpretation = scenario.get("diagnostic_interpretation")
    if not isinstance(interpretation, dict):
        interpretation = {}
    failure_sources = _dedupe(_normalize_reason(source) for source in _string_list(interpretation.get("failure_sources", [])))
    return {
        "run_id": run_record["run_id"],
        "scenario_id": str(scenario.get("scenario_id", "")),
        "scenario_group": str(scenario.get("scenario_group", "unknown")),
        "scenario_set": str(command_args.get("scenario_set", "unknown")),
        "diagnostic_profile": str(command_args.get("diagnostic_profile", "unknown")),
        "top_k": command_args.get("top_k"),
        "source_summary_path": run_record["source_summary_path"],
        "selected_cell_before_path_feedback": scenario.get("selected_cell_before_path_feedback"),
        "selected_cell_after_path_feedback": scenario.get("selected_cell_after_path_feedback"),
        "selection_changed": bool(scenario.get("selection_changed_by_path_feedback", False)),
        "target_replacement_reason": str(interpretation.get("target_replacement_reason", "unknown")),
        "failure_sources": failure_sources,
    }


def _aggregate_decision_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(record["scenario_id"], []).append(record)
    result = {}
    for scenario_id, bucket_records in sorted(buckets.items()):
        selection_changed_counts = Counter(str(record["selection_changed"]).lower() for record in bucket_records)
        target_replacement_reason_counts = Counter(record["target_replacement_reason"] for record in bucket_records)
        failure_source_counts = Counter(
            source
            for record in bucket_records
            for source in record["failure_sources"]
        )
        failure_source_sets = [set(record["failure_sources"]) for record in bucket_records]
        result[scenario_id] = {
            "observation_count": len(bucket_records),
            "scenario_group": _first_non_empty(*(record["scenario_group"] for record in bucket_records), default="unknown"),
            "run_ids": _unique(record["run_id"] for record in bucket_records),
            "scenario_sets": _unique(record["scenario_set"] for record in bucket_records),
            "diagnostic_profiles": _unique(record["diagnostic_profile"] for record in bucket_records),
            "top_k_values": _unique(str(record["top_k"]) for record in bucket_records),
            "source_summary_paths": _unique(record["source_summary_path"] for record in bucket_records),
            "selection_changed_counts": dict(sorted(selection_changed_counts.items())),
            "target_replacement_reason_counts": dict(sorted(target_replacement_reason_counts.items())),
            "failure_source_counts": dict(sorted(failure_source_counts.items())),
            "stable_selection_changed": len(selection_changed_counts) <= 1,
            "stable_target_replacement_reason": len(target_replacement_reason_counts) <= 1,
            "stable_failure_sources": len({tuple(sorted(values)) for values in failure_source_sets}) <= 1,
            "stable_failure_sources_value": sorted(failure_source_sets[0]) if failure_source_sets else [],
        }
    return result


def _index_counts(run_index: dict[str, Any]) -> dict[str, int]:
    return {
        "run_count": _int_value(run_index.get("run_count")),
        "passed_count": _int_value(run_index.get("passed_count")),
        "failed_count": _int_value(run_index.get("failed_count")),
    }


def _evaluation_counts(evaluation_summary: dict[str, Any]) -> dict[str, int]:
    counts = {
        "run_count": _int_value(evaluation_summary.get("run_count")),
        "passed_count": _int_value(evaluation_summary.get("passed_count")),
        "failed_count": _int_value(evaluation_summary.get("failed_count")),
        "open_grid_fallback_used_count": _int_value(evaluation_summary.get("open_grid_fallback_used_count")),
        "path_planning_failure_count": _int_value(evaluation_summary.get("path_planning_failure_count")),
        "replan_count": _int_value(evaluation_summary.get("replan_count")),
        "iris_fallback_count": _int_value(evaluation_summary.get("iris_fallback_count")),
        "region_graph_fallback_count": _int_value(evaluation_summary.get("region_graph_fallback_count")),
        "region_graph_disconnected_count": _int_value(evaluation_summary.get("region_graph_disconnected_count")),
    }
    for key in CHANNEL_AWARE_COUNT_METRIC_KEYS:
        counts[key] = _int_value(evaluation_summary.get(key))
    return counts


def _channel_aware_astar_evidence(evaluation_summary: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for key in CHANNEL_AWARE_EVIDENCE_KEYS:
        value = evaluation_summary.get(key)
        if isinstance(value, dict):
            evidence[key] = dict(value)
        else:
            evidence[key] = value
    return evidence


def _validate_git_snapshot(snapshot: Any, *, prefix: str) -> list[str]:
    reason_codes: list[str] = []
    if not isinstance(snapshot, dict) or not snapshot:
        return [f"{prefix}_missing"]
    parent = snapshot.get("parent")
    if not isinstance(parent, dict) or not _looks_like_sha(parent.get("sha")):
        reason_codes.append(f"{prefix}_parent_sha_missing")
    submodules = snapshot.get("submodules")
    if not isinstance(submodules, dict):
        reason_codes.append(f"{prefix}_submodules_missing")
        return reason_codes
    for name in SUBMODULES:
        item = submodules.get(name)
        if not isinstance(item, dict) or not _looks_like_sha(item.get("sha")):
            reason_codes.append(f"{prefix}_{name}_sha_missing")
    return reason_codes


def _git_snapshots_match(left: Any, right: Any) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    if _nested_value(left, ("parent", "sha")) != _nested_value(right, ("parent", "sha")):
        return False
    for name in SUBMODULES:
        if _nested_value(left, ("submodules", name, "sha")) != _nested_value(right, ("submodules", name, "sha")):
            return False
    return True


def _public_git_snapshot(snapshot: Any) -> dict[str, Any]:
    return dict(snapshot) if isinstance(snapshot, dict) else {}


def _git_snapshot(repo_root: Path) -> dict[str, Any]:
    return {
        "parent": _git_repo_state(repo_root, repo_root=repo_root),
        "submodules": {
            name: _git_repo_state(repo_root / name, repo_root=repo_root)
            for name in SUBMODULES
        },
    }


def _git_repo_state(path: Path, *, repo_root: Path) -> dict[str, Any]:
    sha = _run_git(path, "rev-parse", "HEAD")
    branch = _run_git(path, "branch", "--show-current")
    return {
        "path": _display_path(path, repo_root),
        "sha": sha or "unknown",
        "branch": branch or None,
    }


def _run_git(path: Path, *args: str) -> str | None:
    if not path.exists():
        return None
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _region_graph_disconnected_count(summary: dict[str, Any]) -> int:
    direct = _int_value(summary.get("region_graph_disconnected_count"))
    if direct:
        return direct
    return _int_value(summary.get("region_graph_start_goal_disconnected_count"))


def _normalize_reason(value: Any) -> str:
    text = str(value).strip()
    aliases = {
        "path_failure": "path_planning_failure",
        "failure": "path_planning_failure",
        "replan": "replan_required",
        "iris_region_fallback": "iris_fallback",
        "region_graph_start_goal_disconnected": "region_graph_disconnected",
        "open_grid_fallback_used": "open_grid_fallback",
    }
    return aliases.get(text, text)


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return []


def _string_set(value: Any) -> set[str]:
    return set(_string_list(value))


def _nested_string(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    value = _nested_value(payload, keys)
    return str(value) if value not in (None, "") else None


def _nested_value(payload: Any, keys: tuple[str, ...]) -> Any:
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _first_non_empty(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return default


def _looks_like_sha(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 40 and all(char in "0123456789abcdef" for char in text.lower())


def _append_reason(reason_codes: list[str], code: str) -> None:
    if code not in reason_codes:
        reason_codes.append(code)


def _append_unique(values: list[str], value: Any) -> None:
    text = str(value)
    if text and text not in values:
        values.append(text)


def _unique(values) -> list[str]:
    result: list[str] = []
    for value in values:
        _append_unique(result, value)
    return result


def _dedupe(values) -> list[str]:
    result: list[str] = []
    for value in values:
        _append_unique(result, value)
    return result


def _resolve_path(value: Any, repo_root: Path) -> Path:
    path = Path(str(value)).expanduser()
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
