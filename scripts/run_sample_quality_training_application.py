from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIG_SCHEMA_VERSION = "sample-quality-training-application-config/v1"
APPLICATION_SCHEMA_VERSION = "sample-quality-training-application-summary/v1"
SELECTION_SCHEMA_VERSION = "training-selection-stability-summary/v1"
RUN_INDEX_SCHEMA_VERSION = "path-feedback-batch-run-index/v1"
EVALUATION_SUMMARY_SCHEMA_VERSION = "path-feedback-batch-evaluation-summary/v1"
BATCH_STABILITY_SCHEMA_VERSION = "batch-stability-summary/v1"
DATASET_STABILITY_SCHEMA_VERSION = "dataset-quality-stability-summary/v1"
DECISION_STABILITY_SCHEMA_VERSION = "decision-stability-summary/v1"
PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION = "path-feedback-summary/v1"
SUBMODULES = ("dev-platform-constraints", "model-explorer", "path-planner")
SOURCE_FILES = {
    "batch_run_index": ("batch-run-index.json", RUN_INDEX_SCHEMA_VERSION),
    "batch_evaluation_summary": ("batch-evaluation-summary.json", EVALUATION_SUMMARY_SCHEMA_VERSION),
    "batch_stability_summary": ("batch-stability-summary.json", BATCH_STABILITY_SCHEMA_VERSION),
    "dataset_quality_stability_summary": ("dataset-quality-stability-summary.json", DATASET_STABILITY_SCHEMA_VERSION),
    "decision_stability_summary": ("decision-stability-summary.json", DECISION_STABILITY_SCHEMA_VERSION),
}

class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply Sample-Quality-Aware Training Application v1 to path-feedback batch outputs."
    )
    parser.add_argument("--batch-root", required=True, help="Batch root containing batch/stability JSON outputs.")
    parser.add_argument("--config", required=True, help="Sample-quality training application config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned output paths.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without writing outputs.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    batch_root = _resolve_path(args.batch_root, repo_root)
    config_path = _resolve_path(args.config, repo_root)

    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    outputs = analyze_training_application(batch_root=batch_root, config=config, repo_root=repo_root)
    output_files = _output_files(batch_root, config)
    status = outputs["application"]["status"]
    validation_message = {
        "status": "config validated" if status == "passed" else "validation failed",
        "batch_root": _display_path(batch_root, repo_root),
        "config": _display_path(config_path, repo_root),
        "profiles": list(outputs["application"]["profile_results"]),
        "reason_codes": outputs["application"]["reason_codes"],
        "application_summary": _display_path(output_files["application"], repo_root),
        "selection_stability_summary": _display_path(output_files["selection"], repo_root),
    }
    print(json.dumps(validation_message, ensure_ascii=False))

    if args.validate_only or args.dry_run:
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "dry-run",
                        "would_write": {
                            "application_summary": _display_path(output_files["application"], repo_root),
                            "selection_stability_summary": _display_path(output_files["selection"], repo_root),
                        },
                        "profile_count": len(outputs["application"]["profile_results"]),
                    },
                    ensure_ascii=False,
                )
            )
        return 1 if status == "failed" else 0

    _write_json(output_files["application"], outputs["application"])
    _write_json(output_files["selection"], outputs["selection"])
    print(
        json.dumps(
            {
                "status": status,
                "sample_quality_training_application_summary": _display_path(output_files["application"], repo_root),
                "training_selection_stability_summary": _display_path(output_files["selection"], repo_root),
                "failure_reason_code_counts": outputs["application"]["failure_reason_code_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if status == "failed" else 0


def analyze_training_application(*, batch_root: Path, config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    global_reason_codes: list[str] = []
    sources = _load_sources(batch_root, repo_root=repo_root, reason_codes=global_reason_codes)
    run_index = sources["payloads"].get("batch_run_index", {})
    evaluation_summary = sources["payloads"].get("batch_evaluation_summary", {})
    stability_payloads = {
        key: sources["payloads"].get(key, {})
        for key in (
            "batch_stability_summary",
            "dataset_quality_stability_summary",
            "decision_stability_summary",
        )
    }
    _inspect_source_statuses(
        evaluation_summary=evaluation_summary,
        stability_payloads=stability_payloads,
        reason_codes=global_reason_codes,
    )

    runs = run_index.get("runs", [])
    if not isinstance(runs, list):
        _append_reason(global_reason_codes, "batch_run_index_runs_invalid")
        runs = []
    evaluation_source_paths = _string_set(evaluation_summary.get("source_summary_paths", []))
    batch_git = run_index.get("git") if isinstance(run_index.get("git"), dict) else {}
    current_git = _git_snapshot(repo_root)
    if _require_current_git_match(config) and batch_git and not _git_snapshots_match(batch_git, current_git):
        _append_reason(global_reason_codes, "current_git_provenance_mismatch")

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

    base_records = _base_sample_records(run_records)
    profile_results = {
        profile["id"]: _build_profile_result(profile, base_records)
        for profile in config["profiles"]
    }
    all_reason_codes = list(global_reason_codes)
    for record in run_records:
        for reason in record["reason_codes"]:
            _append_reason(all_reason_codes, reason)
    status = "failed" if all_reason_codes else "passed"
    failure_reason_code_counts = Counter(global_reason_codes)
    for record in run_records:
        failure_reason_code_counts.update(record["reason_codes"])

    application_summary = {
        "schema_version": APPLICATION_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(global_reason_codes),
        "failure_reason_code_counts": dict(sorted(failure_reason_code_counts.items())),
        "batch_root": _display_path(batch_root, repo_root),
        "quality_signal_use": "training_sample_selection_audit_only",
        "training_application_scope": "sample_selection_and_best_run_profile_audit_only",
        "not_real_world_performance_claim": True,
        "does_not_modify_ppo": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "non_goals": list(config.get("non_goals", [])),
        "config": _public_config(config),
        "source_summaries": _source_summary_records(sources, run_records, repo_root=repo_root),
        "acceptance_metadata": {
            "by_run": {
                record["run_id"]: dict(record["acceptance_metadata"])
                for record in run_records
            },
        },
        "git_provenance": {
            "batch": _public_git_snapshot(batch_git),
            "current": current_git,
            "runs_match_batch": all(
                _git_snapshots_match(batch_git, record.get("git", {}))
                for record in run_records
                if batch_git
            ),
            "current_matches_batch": _git_snapshots_match(batch_git, current_git) if batch_git else None,
        },
        "run_count": len(run_records),
        "record_count": len(base_records),
        "by_run": {record["run_id"]: _run_public_record(record) for record in run_records},
        "profile_results": profile_results,
    }
    selection_summary = _build_selection_summary(
        status=status,
        global_reason_codes=global_reason_codes,
        profile_results=profile_results,
        config=config,
        batch_root=batch_root,
        repo_root=repo_root,
    )
    return {"application": application_summary, "selection": selection_summary}


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ConfigError("profiles must be a non-empty array")
    normalized_profiles = [_normalize_profile(profile, index) for index, profile in enumerate(profiles)]
    profile_ids = [profile["id"] for profile in normalized_profiles]
    if len(profile_ids) != len(set(profile_ids)):
        raise ConfigError("profile ids must be unique")
    legacy_profile_id = _required_string(payload.get("legacy_profile_id"), "legacy_profile_id")
    aware_profile_id = _required_string(payload.get("sample_quality_aware_profile_id"), "sample_quality_aware_profile_id")
    if legacy_profile_id not in profile_ids:
        raise ConfigError("legacy_profile_id must reference a configured profile")
    if aware_profile_id not in profile_ids:
        raise ConfigError("sample_quality_aware_profile_id must reference a configured profile")
    config = dict(payload)
    config["profiles"] = normalized_profiles
    return config


def _normalize_profile(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"profiles[{index}] must be an object")
    profile_id = _required_string(value.get("id"), f"profiles[{index}].id")
    downweight_factor = _float_value(value.get("downweight_factor", 0.5), f"profiles[{index}].downweight_factor")
    if downweight_factor < 0.0:
        raise ConfigError(f"profiles[{index}].downweight_factor must be >= 0")
    return {
        "id": profile_id,
        "description": str(value.get("description", "")),
        "legacy_mode": bool(value.get("legacy_mode", False)),
        "exclude_reason_codes": sorted(_string_set(value.get("exclude_reason_codes", []))),
        "downweight_reason_codes": sorted(_string_set(value.get("downweight_reason_codes", []))),
        "downweight_factor": float(downweight_factor),
    }


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{label} must be a non-empty string")
    return value


def _float_value(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be numeric") from exc


def _load_sources(batch_root: Path, *, repo_root: Path, reason_codes: list[str]) -> dict[str, Any]:
    payloads: dict[str, dict[str, Any]] = {}
    source_records: dict[str, dict[str, Any]] = {}
    for key, (filename, expected_schema) in SOURCE_FILES.items():
        path = batch_root / filename
        payload = _load_json_object(path, reason_codes, key)
        payloads[key] = payload
        schema_version = payload.get("schema_version") if isinstance(payload, dict) else None
        status = "passed"
        if not path.is_file():
            status = "missing"
        elif schema_version != expected_schema:
            status = "failed"
        source_records[key] = {
            "path": _display_path(path, repo_root),
            "status": status,
            "schema_version": schema_version,
            "expected_schema_version": expected_schema,
        }
        if path.is_file() and schema_version != expected_schema:
            _append_reason(reason_codes, f"{key}_schema_mismatch")
    return {"payloads": payloads, "source_records": source_records}


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


def _inspect_source_statuses(
    *,
    evaluation_summary: dict[str, Any],
    stability_payloads: dict[str, dict[str, Any]],
    reason_codes: list[str],
) -> None:
    if _int_value(evaluation_summary.get("failed_count")) > 0:
        _append_reason(reason_codes, "batch_evaluation_failed")
    for key, payload in stability_payloads.items():
        if payload and payload.get("status") != "passed":
            _append_reason(reason_codes, f"{key}_failed")
            for reason in _string_list(payload.get("reason_codes", [])):
                _append_reason(reason_codes, reason)


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

    command_args = run.get("command_args") if isinstance(run.get("command_args"), dict) else {}
    if not command_args:
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

    if summary and summary.get("schema_version") != PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION:
        _append_reason(reason_codes, "source_summary_schema_mismatch")

    acceptance_metadata = summary.get("acceptance_metadata") if isinstance(summary.get("acceptance_metadata"), dict) else {}
    run_acceptance_metadata = run.get("acceptance_metadata") if isinstance(run.get("acceptance_metadata"), dict) else {}
    mismatches = _acceptance_metadata_mismatches(
        command_args=command_args,
        run_acceptance_metadata=run_acceptance_metadata,
        summary=summary,
        summary_acceptance_metadata=acceptance_metadata,
    )
    if summary and not acceptance_metadata:
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
    if batch_git and run_git and not _git_snapshots_match(batch_git, run_git):
        _append_reason(reason_codes, "git_provenance_mismatch")

    return {
        "run_id": run_id,
        "status": "failed" if reason_codes else "passed",
        "reason_codes": reason_codes,
        "batch_reason_codes": original_reason_codes,
        "command_args": dict(command_args),
        "sample_quality_profile": run.get("sample_quality_profile"),
        "source_summary_path": source_summary_display,
        "acceptance_metadata": dict(acceptance_metadata),
        "acceptance_metadata_mismatches": mismatches,
        "open_grid_fallback_used": open_grid_used,
        "summary": summary if isinstance(summary, dict) else {},
        "git": run_git,
        "metrics": _summary_metrics(summary),
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
    actual_open_grid = _open_grid_fallback_used(summary, summary_acceptance_metadata)
    if actual_open_grid is not None and bool(actual_open_grid) is not False:
        mismatches.append(
            {
                "field": "open_grid_fallback_used",
                "expected": False,
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
    return {
        "scenario_count": _int_value(summary.get("scenario_count")),
        "candidate_count": _int_value(summary.get("candidate_count")),
        "path_planning_failure_count": _int_value(summary.get("path_planning_failure_count")),
        "replan_count": _int_value(summary.get("replan_count")),
        "iris_fallback_count": _int_value(summary.get("iris_fallback_count")),
        "region_graph_fallback_count": _int_value(summary.get("region_graph_fallback_count")),
        "region_graph_disconnected_count": _int_value(
            summary.get("region_graph_disconnected_count", summary.get("region_graph_start_goal_disconnected_count"))
        ),
    }


def _base_sample_records(run_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_record in run_records:
        summary = run_record.get("summary", {})
        if not isinstance(summary, dict):
            continue
        scenarios = summary.get("scenarios", [])
        if not isinstance(scenarios, list):
            continue
        for scenario in scenarios:
            if not isinstance(scenario, dict):
                continue
            records.append(_base_sample_record(run_record, scenario))
    return records


def _base_sample_record(run_record: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    command_args = run_record["command_args"]
    reason_codes = _sample_quality_reason_codes(scenario)
    return {
        "run_id": run_record["run_id"],
        "scenario_id": str(scenario.get("scenario_id", "")),
        "scenario_group": str(scenario.get("scenario_group", "unknown")),
        "scenario_set": str(command_args.get("scenario_set", "unknown")),
        "diagnostic_profile": str(command_args.get("diagnostic_profile", "unknown")),
        "top_k": command_args.get("top_k"),
        "source_summary_path": run_record["source_summary_path"],
        "acceptance_metadata": dict(run_record["acceptance_metadata"]),
        "sample_quality_profile_from_batch": run_record.get("sample_quality_profile"),
        "reason_codes": reason_codes,
        "quality_signal_use": "training_sample_selection_audit_only",
        "not_real_world_performance_claim": True,
    }


def _sample_quality_reason_codes(scenario: dict[str, Any]) -> list[str]:
    codes: list[str] = []
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
    if scenario.get("open_grid_fallback_used") is True:
        codes.append("open_grid_fallback")
    return _dedupe(code for code in codes if code and code != "none") or ["sample_quality_passed"]


def _normalize_reason(value: Any) -> str:
    text = str(value).strip()
    aliases = {
        "replan": "replan_required",
        "path_failure": "path_planning_failure",
        "failure": "path_planning_failure",
        "iris_region_fallback": "iris_fallback",
        "region_graph_start_goal_disconnected": "region_graph_disconnected",
    }
    return aliases.get(text, text)


def _build_profile_result(profile: dict[str, Any], base_records: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for base_record in base_records:
        action, sample_weight = _profile_action(base_record["reason_codes"], profile)
        record = dict(base_record)
        record.update(
            {
                "profile_id": profile["id"],
                "action": action,
                "decision": action,
                "sample_weight": sample_weight,
            }
        )
        records.append(record)
    action_counts = Counter(record["action"] for record in records)
    reason_code_counts = Counter(reason for record in records for reason in record["reason_codes"])
    return {
        "profile": dict(profile),
        "record_count": len(records),
        "excluded_sample_count": int(action_counts.get("exclude", 0)),
        "downweighted_sample_count": int(action_counts.get("downweight", 0)),
        "kept_sample_count": int(action_counts.get("keep", 0)),
        "action_counts": dict(sorted(action_counts.items())),
        "reason_code_counts": dict(sorted(reason_code_counts.items())),
        "by_action": _aggregate_records_by_field(records, "action"),
        "by_reason_code": _aggregate_records_by_reason_code(records),
        "by_scenario_id": _aggregate_records_by_field(records, "scenario_id"),
        "by_scenario_group": _aggregate_records_by_field(records, "scenario_group"),
        "by_run": _aggregate_records_by_field(records, "run_id"),
        "run_quality": _run_quality_records(records),
        "records": records,
    }


def _profile_action(reason_codes: list[str], profile: dict[str, Any]) -> tuple[str, float]:
    if profile.get("legacy_mode"):
        return "keep", 1.0
    reason_set = set(reason_codes)
    exclude_reasons = set(profile.get("exclude_reason_codes", []))
    downweight_reasons = set(profile.get("downweight_reason_codes", []))
    if exclude_reasons.intersection(reason_set):
        return "exclude", 0.0
    if downweight_reasons.intersection(reason_set):
        return "downweight", max(0.0, float(profile.get("downweight_factor", 0.5)))
    return "keep", 1.0


def _aggregate_records_by_field(records: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(str(record.get(field, "unknown")), []).append(record)
    return {
        key: _quality_bucket(bucket_records)
        for key, bucket_records in sorted(buckets.items())
    }


def _aggregate_records_by_reason_code(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        for reason in record["reason_codes"]:
            buckets.setdefault(str(reason), []).append(record)
    return {
        key: _quality_bucket(bucket_records)
        for key, bucket_records in sorted(buckets.items())
    }


def _quality_bucket(records: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts = Counter(record["action"] for record in records)
    reason_code_counts = Counter(reason for record in records for reason in record["reason_codes"])
    return {
        "record_count": len(records),
        "action_counts": dict(sorted(action_counts.items())),
        "reason_code_counts": dict(sorted(reason_code_counts.items())),
        "scenario_ids": _unique(record["scenario_id"] for record in records),
        "run_ids": _unique(record["run_id"] for record in records),
        "source_summary_paths": _unique(record["source_summary_path"] for record in records),
        "average_sample_weight": _average(record["sample_weight"] for record in records),
    }


def _run_quality_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for run_id, bucket in _group_records(records, "run_id").items():
        action_counts = Counter(record["action"] for record in bucket)
        reason_code_counts = Counter(reason for record in bucket for reason in record["reason_codes"])
        first = bucket[0]
        result[run_id] = {
            "run_id": run_id,
            "scenario_set": first["scenario_set"],
            "diagnostic_profile": first["diagnostic_profile"],
            "top_k": first["top_k"],
            "sample_quality_profile_from_batch": first.get("sample_quality_profile_from_batch"),
            "profile_key": _run_profile_key(first),
            "record_count": len(bucket),
            "excluded_sample_count": int(action_counts.get("exclude", 0)),
            "downweighted_sample_count": int(action_counts.get("downweight", 0)),
            "kept_sample_count": int(action_counts.get("keep", 0)),
            "average_sample_weight": _average(record["sample_weight"] for record in bucket),
            "total_sample_weight": sum(float(record["sample_weight"]) for record in bucket),
            "reason_code_counts": dict(sorted(reason_code_counts.items())),
            "source_summary_paths": _unique(record["source_summary_path"] for record in bucket),
        }
    return dict(sorted(result.items()))


def _run_profile_key(record: dict[str, Any]) -> str:
    return "|".join(
        (
            f"scenario_set={record.get('scenario_set', 'unknown')}",
            f"diagnostic_profile={record.get('diagnostic_profile', 'unknown')}",
            f"top_k={record.get('top_k')}",
            f"sample_quality_profile={record.get('sample_quality_profile_from_batch', 'unknown')}",
        )
    )


def _build_selection_summary(
    *,
    status: str,
    global_reason_codes: list[str],
    profile_results: dict[str, dict[str, Any]],
    config: dict[str, Any],
    batch_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    profile_selections = {
        profile_id: _select_best_run_for_profile(profile_id, result)
        for profile_id, result in profile_results.items()
    }
    legacy_id = config["legacy_profile_id"]
    aware_id = config["sample_quality_aware_profile_id"]
    legacy_selection = profile_selections.get(legacy_id, _empty_selection(legacy_id))
    aware_selection = profile_selections.get(aware_id, _empty_selection(aware_id))
    comparison_reason_codes = ["no_training_metric_evaluated"]
    changed = legacy_selection.get("run_id") != aware_selection.get("run_id") or legacy_selection.get("profile_key") != aware_selection.get("profile_key")
    comparison_reason_codes.append(
        "selection_changed_by_sample_quality_profile" if changed else "selection_stable_across_profiles"
    )
    min_observations = _int_value(config.get("selection", {}).get("min_observations_for_stable_selection", 2))
    if _int_value(aware_selection.get("record_count")) < max(1, min_observations):
        comparison_reason_codes.append("single_observation_selection_risk")
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(global_reason_codes),
        "batch_root": _display_path(batch_root, repo_root),
        "quality_signal_use": "best_run_profile_selection_audit_only",
        "not_real_world_performance_claim": True,
        "does_not_run_training": True,
        "policy": str(config.get("selection", {}).get("policy", "audit_quality_score")),
        "metric": str(config.get("selection", {}).get("metric", "average_sample_weight")),
        "profile_selections": profile_selections,
        "comparison": {
            "legacy_profile_id": legacy_id,
            "sample_quality_aware_profile_id": aware_id,
            "legacy_selection": legacy_selection,
            "sample_quality_aware_selection": aware_selection,
            "selection_changed": changed,
            "reason_codes": comparison_reason_codes,
            "single_metric_accidental_improvement": "not_evaluated_no_training_metric",
            "stability_status": "insufficient_observations"
            if "single_observation_selection_risk" in comparison_reason_codes
            else "audit_comparable",
        },
    }


def _select_best_run_for_profile(profile_id: str, result: dict[str, Any]) -> dict[str, Any]:
    run_quality = result.get("run_quality", {})
    if not isinstance(run_quality, dict) or not run_quality:
        return _empty_selection(profile_id)
    candidates = list(run_quality.values())
    eligible = [
        record
        for record in candidates
        if not (record["record_count"] > 0 and record["excluded_sample_count"] == record["record_count"])
    ]
    if not eligible:
        eligible = candidates
    selected = max(
        eligible,
        key=lambda record: (
            float(record["average_sample_weight"]),
            -int(record["excluded_sample_count"]),
            -int(record["downweighted_sample_count"]),
            -sum(int(value) for value in record["reason_code_counts"].values()),
            str(record["run_id"]),
        ),
    )
    reason_codes = ["selected_best_quality_audit_run"]
    if selected["excluded_sample_count"]:
        reason_codes.append("selected_run_has_exclusions")
    if selected["downweighted_sample_count"]:
        reason_codes.append("selected_run_has_downweights")
    return {
        "profile_id": profile_id,
        "status": "selected",
        "run_id": selected["run_id"],
        "profile_key": selected["profile_key"],
        "reason_codes": reason_codes,
        "record_count": selected["record_count"],
        "excluded_sample_count": selected["excluded_sample_count"],
        "downweighted_sample_count": selected["downweighted_sample_count"],
        "kept_sample_count": selected["kept_sample_count"],
        "average_sample_weight": selected["average_sample_weight"],
        "total_sample_weight": selected["total_sample_weight"],
        "source_summary_paths": list(selected["source_summary_paths"]),
    }


def _empty_selection(profile_id: str) -> dict[str, Any]:
    return {
        "profile_id": profile_id,
        "status": "no_candidate",
        "run_id": None,
        "profile_key": None,
        "reason_codes": ["no_quality_records"],
        "record_count": 0,
        "excluded_sample_count": 0,
        "downweighted_sample_count": 0,
        "kept_sample_count": 0,
        "average_sample_weight": 0.0,
        "total_sample_weight": 0.0,
        "source_summary_paths": [],
    }


def _source_summary_records(sources: dict[str, Any], run_records: list[dict[str, Any]], *, repo_root: Path) -> dict[str, Any]:
    records = dict(sources["source_records"])
    records["path_feedback_summary_paths"] = [
        record["source_summary_path"]
        for record in run_records
        if record["source_summary_path"]
    ]
    records["path_feedback_summaries"] = {
        record["run_id"]: {
            "path": record["source_summary_path"],
            "status": record["status"],
            "schema_version": record["summary"].get("schema_version") if isinstance(record.get("summary"), dict) else None,
        }
        for record in run_records
    }
    return records


def _run_public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": record["run_id"],
        "status": record["status"],
        "reason_codes": list(record["reason_codes"]),
        "batch_reason_codes": list(record["batch_reason_codes"]),
        "command_args": dict(record["command_args"]),
        "sample_quality_profile": record.get("sample_quality_profile"),
        "source_summary_path": record["source_summary_path"],
        "acceptance_metadata": dict(record["acceptance_metadata"]),
        "acceptance_metadata_mismatches": list(record["acceptance_metadata_mismatches"]),
        "open_grid_fallback_used": record["open_grid_fallback_used"],
        "metrics": dict(record["metrics"]),
    }


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": config["schema_version"],
        "legacy_profile_id": config["legacy_profile_id"],
        "sample_quality_aware_profile_id": config["sample_quality_aware_profile_id"],
        "validation": dict(config.get("validation", {})),
        "selection": dict(config.get("selection", {})),
        "profiles": [dict(profile) for profile in config["profiles"]],
    }


def _output_files(batch_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    output_config = config.get("output_files", {})
    if not isinstance(output_config, dict):
        output_config = {}
    application_name = str(output_config.get("application_summary", "sample-quality-training-application-summary.json"))
    selection_name = str(output_config.get("selection_stability_summary", "training-selection-stability-summary.json"))
    return {
        "application": batch_root / application_name,
        "selection": batch_root / selection_name,
    }


def _require_current_git_match(config: dict[str, Any]) -> bool:
    validation = config.get("validation", {})
    if not isinstance(validation, dict):
        return True
    return bool(validation.get("require_current_git_match", True))


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


def _public_git_snapshot(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _git_snapshots_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _git_sha_map(left) == _git_sha_map(right)


def _git_sha_map(snapshot: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    parent = snapshot.get("parent")
    if isinstance(parent, dict):
        result[str(parent.get("path", "."))] = str(parent.get("sha", "unknown"))
    submodules = snapshot.get("submodules")
    if isinstance(submodules, dict):
        for name, payload in submodules.items():
            if isinstance(payload, dict):
                result[str(name)] = str(payload.get("sha", "unknown"))
    return result


def _group_records(records: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(str(record.get(field, "unknown")), []).append(record)
    return dict(sorted(buckets.items()))


def _string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {str(item) for item in value}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _nested_string(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return str(value) if value not in (None, "") else None


def _first_non_empty(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return default


def _dedupe(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _unique(values) -> list[str]:
    return _dedupe(str(value) for value in values if value not in (None, ""))


def _average(values) -> float:
    numbers = [float(value) for value in values]
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def _append_reason(reason_codes: list[str], code: str) -> None:
    if code not in reason_codes:
        reason_codes.append(code)


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
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
