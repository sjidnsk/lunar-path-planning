from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any


CONFIG_SCHEMA_VERSION = "policy-decision-robustness-config/v1"
ROBUSTNESS_SCHEMA_VERSION = "policy-decision-robustness-summary/v1"
COMPARISON_SCHEMA_VERSION = "policy-decision-selection-comparison-summary/v1"
RUN_INDEX_SCHEMA_VERSION = "path-feedback-batch-run-index/v1"
EVALUATION_SUMMARY_SCHEMA_VERSION = "path-feedback-batch-evaluation-summary/v1"
BATCH_STABILITY_SCHEMA_VERSION = "batch-stability-summary/v1"
DATASET_STABILITY_SCHEMA_VERSION = "dataset-quality-stability-summary/v1"
DECISION_STABILITY_SCHEMA_VERSION = "decision-stability-summary/v1"
APPLICATION_SCHEMA_VERSION = "sample-quality-training-application-summary/v1"
TRAINING_SELECTION_SCHEMA_VERSION = "training-selection-stability-summary/v1"
PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION = "path-feedback-summary/v1"
SUBMODULES = ("dev-platform-constraints", "model-explorer", "path-planner")
SOURCE_FILES = {
    "batch_run_index": ("batch-run-index.json", RUN_INDEX_SCHEMA_VERSION),
    "batch_evaluation_summary": ("batch-evaluation-summary.json", EVALUATION_SUMMARY_SCHEMA_VERSION),
    "batch_stability_summary": ("batch-stability-summary.json", BATCH_STABILITY_SCHEMA_VERSION),
    "dataset_quality_stability_summary": ("dataset-quality-stability-summary.json", DATASET_STABILITY_SCHEMA_VERSION),
    "decision_stability_summary": ("decision-stability-summary.json", DECISION_STABILITY_SCHEMA_VERSION),
    "sample_quality_training_application_summary": (
        "sample-quality-training-application-summary.json",
        APPLICATION_SCHEMA_VERSION,
    ),
    "training_selection_stability_summary": (
        "training-selection-stability-summary.json",
        TRAINING_SELECTION_SCHEMA_VERSION,
    ),
}


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Policy Decision Robustness v1 analysis over path-feedback batch outputs."
    )
    parser.add_argument("--batch-root", required=True, help="Batch root containing refreshed policy input JSON.")
    parser.add_argument("--config", required=True, help="Policy decision robustness config JSON.")
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

    outputs = analyze_policy_decision_robustness(batch_root=batch_root, config=config, repo_root=repo_root)
    output_files = _output_files(batch_root, config)
    status = outputs["robustness"]["status"]
    validation_message = {
        "status": "config validated" if status == "passed" else "validation failed",
        "batch_root": _display_path(batch_root, repo_root),
        "config": _display_path(config_path, repo_root),
        "profiles": list(outputs["robustness"]["profiles"]),
        "reason_codes": outputs["robustness"]["reason_codes"],
        "robustness_summary": _display_path(output_files["robustness"], repo_root),
        "selection_comparison_summary": _display_path(output_files["comparison"], repo_root),
    }
    print(json.dumps(validation_message, ensure_ascii=False))

    if args.validate_only or args.dry_run:
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "dry-run",
                        "would_write": {
                            "robustness_summary": _display_path(output_files["robustness"], repo_root),
                            "selection_comparison_summary": _display_path(output_files["comparison"], repo_root),
                        },
                        "profile_count": len(outputs["robustness"]["profiles"]),
                    },
                    ensure_ascii=False,
                )
            )
        return 1 if status == "failed" else 0

    _write_json(output_files["robustness"], outputs["robustness"])
    _write_json(output_files["comparison"], outputs["comparison"])
    print(
        json.dumps(
            {
                "status": status,
                "policy_decision_robustness_summary": _display_path(output_files["robustness"], repo_root),
                "policy_decision_selection_comparison_summary": _display_path(output_files["comparison"], repo_root),
                "failure_reason_code_counts": outputs["robustness"]["failure_reason_code_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if status == "failed" else 0


def analyze_policy_decision_robustness(*, batch_root: Path, config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    global_reason_codes: list[str] = []
    sources = _load_sources(batch_root, repo_root=repo_root, reason_codes=global_reason_codes)
    payloads = sources["payloads"]
    run_index = payloads.get("batch_run_index", {})
    evaluation_summary = payloads.get("batch_evaluation_summary", {})
    stability_payloads = {
        key: payloads.get(key, {})
        for key in (
            "batch_stability_summary",
            "dataset_quality_stability_summary",
            "decision_stability_summary",
            "sample_quality_training_application_summary",
            "training_selection_stability_summary",
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
    batch_git = run_index.get("git") if isinstance(run_index.get("git"), dict) else {}
    current_git = _git_snapshot(repo_root)
    if _require_current_git_match(config) and batch_git and not _git_snapshots_match(batch_git, current_git):
        _append_reason(global_reason_codes, "current_git_provenance_mismatch")

    evaluation_source_paths = _string_set(evaluation_summary.get("source_summary_paths", []))
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

    sample_quality_records = _sample_quality_records_by_key(
        payloads.get("sample_quality_training_application_summary", {}),
        config=config,
    )
    profile_results = {
        profile["id"]: _build_profile_result(
            profile,
            run_records=run_records,
            sample_quality_records=sample_quality_records,
            config=config,
        )
        for profile in config["profiles"]
    }

    all_reason_codes = list(global_reason_codes)
    for record in run_records:
        for reason in record["reason_codes"]:
            _append_reason(all_reason_codes, reason)
    status = "failed" if all_reason_codes else "passed"
    failure_counts = Counter(global_reason_codes)
    for record in run_records:
        failure_counts.update(record["reason_codes"])

    robustness = {
        "schema_version": ROBUSTNESS_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(global_reason_codes),
        "failure_reason_code_counts": dict(sorted(failure_counts.items())),
        "batch_root": _display_path(batch_root, repo_root),
        "policy_decision_scope": "candidate_sorting_robustness_audit_only",
        "quality_signal_use": "policy_candidate_sorting_audit_only",
        "not_real_world_performance_claim": True,
        "does_not_modify_ppo": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "no_training_metric_evaluated": True,
        "non_goals": list(config.get("non_goals", [])),
        "config": _public_config(config),
        "source_summaries": _source_summary_records(sources, run_records),
        "acceptance_metadata": {
            "by_run": {record["run_id"]: dict(record["acceptance_metadata"]) for record in run_records},
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
        "scenario_count": sum(result["scenario_count"] for result in profile_results.values()) // max(1, len(profile_results)),
        "candidate_count": sum(result["candidate_count"] for result in profile_results.values()) // max(1, len(profile_results)),
        "by_run": {record["run_id"]: _run_public_record(record) for record in run_records},
        "profiles": profile_results,
    }
    comparison = _build_selection_comparison(
        status=status,
        global_reason_codes=global_reason_codes,
        profile_results=profile_results,
        config=config,
        batch_root=batch_root,
        repo_root=repo_root,
    )
    return {"robustness": robustness, "comparison": comparison}


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
    for field in ("legacy_profile_id", "feedback_aware_profile_id", "sample_quality_aware_profile_id"):
        value = _required_string(payload.get(field), field)
        if value not in profile_ids:
            raise ConfigError(f"{field} must reference a configured profile")
    config = dict(payload)
    config["profiles"] = normalized_profiles
    config["scoring"] = _normalize_scoring(config.get("scoring", {}))
    return config


def _normalize_profile(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"profiles[{index}] must be an object")
    profile_id = _required_string(value.get("id"), f"profiles[{index}].id")
    return {
        "id": profile_id,
        "description": str(value.get("description", "")),
        "use_path_feedback": bool(value.get("use_path_feedback", False)),
        "use_sample_quality": bool(value.get("use_sample_quality", False)),
        "sample_quality_source_profile_id": str(value.get("sample_quality_source_profile_id", "")),
    }


def _normalize_scoring(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        value = {}
    defaults = {
        "path_cost_normalizer": 100.0,
        "high_path_cost_threshold": 100.0,
        "utility_weight": 1.0,
        "path_cost_weight": 0.25,
        "risk_weight": 0.25,
        "unreachable_penalty": 1.0,
        "failure_penalty": 1.0,
        "replan_penalty": 0.5,
        "high_path_cost_penalty": 0.35,
        "iris_fallback_penalty": 0.1,
        "region_graph_fallback_penalty": 0.2,
        "region_graph_disconnected_penalty": 0.3,
        "open_grid_fallback_penalty": 2.0,
        "sample_quality_downweight_penalty": 0.5,
        "sample_quality_exclude_penalty": 2.0,
    }
    return {key: _float_value(value.get(key, default), f"scoring.{key}") for key, default in defaults.items()}


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
    source_summary_display = _display_path(summary_path, repo_root) if summary_path is not None else str(source_summary_value)
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


def _sample_quality_records_by_key(application_summary: dict[str, Any], *, config: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    profile_id = ""
    for profile in config["profiles"]:
        if profile.get("use_sample_quality"):
            profile_id = str(profile.get("sample_quality_source_profile_id") or "")
            break
    profile_results = application_summary.get("profile_results") if isinstance(application_summary, dict) else {}
    profile = profile_results.get(profile_id) if isinstance(profile_results, dict) else {}
    records = profile.get("records") if isinstance(profile, dict) else []
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, dict):
            continue
        key = (str(record.get("run_id", "")), str(record.get("scenario_id", "")))
        result[key] = dict(record)
    return result


def _build_profile_result(
    profile: dict[str, Any],
    *,
    run_records: list[dict[str, Any]],
    sample_quality_records: dict[tuple[str, str], dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for run_record in run_records:
        summary = run_record.get("summary", {})
        scenarios = summary.get("scenarios", []) if isinstance(summary, dict) else []
        for scenario in scenarios if isinstance(scenarios, list) else []:
            if not isinstance(scenario, dict):
                continue
            decisions.append(
                _profile_decision(
                    profile,
                    run_record=run_record,
                    scenario=scenario,
                    sample_quality_record=sample_quality_records.get(
                        (run_record["run_id"], str(scenario.get("scenario_id", ""))),
                        {},
                    ),
                    config=config,
                )
            )
    reason_counts = Counter(
        reason
        for decision in decisions
        for candidate in decision["candidate_comparisons"]
        for reason in candidate["reason_codes"]
    )
    selection_changed_count = sum(1 for decision in decisions if decision["selection_changed_by_profile"])
    return {
        "profile": dict(profile),
        "scenario_count": len(decisions),
        "candidate_count": sum(len(decision["candidate_comparisons"]) for decision in decisions),
        "selection_changed_count": selection_changed_count,
        "selection_stable_count": len(decisions) - selection_changed_count,
        "average_abs_rank_delta": _average(
            abs(candidate["rank_delta"]) for decision in decisions for candidate in decision["candidate_comparisons"]
        ),
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "by_scenario_group": _decision_aggregate(decisions, "scenario_group"),
        "by_scenario_id": _decision_aggregate(decisions, "scenario_id"),
        "decisions": decisions,
    }


def _profile_decision(
    profile: dict[str, Any],
    *,
    run_record: dict[str, Any],
    scenario: dict[str, Any],
    sample_quality_record: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    feedback = scenario.get("path_feedback") if isinstance(scenario.get("path_feedback"), dict) else {}
    candidates = [candidate for candidate in feedback.get("candidates", []) if isinstance(candidate, dict)]
    base_scores = {
        _action_index(candidate): _legacy_score(candidate, config=config)
        for candidate in candidates
    }
    profile_scores = {
        _action_index(candidate): _score_candidate(
            candidate,
            profile=profile,
            sample_quality_record=sample_quality_record,
            config=config,
        )
        for candidate in candidates
    }
    before_ranking = _ranking(candidates, base_scores)
    after_ranking = _ranking(candidates, profile_scores)
    before_rank_by_action = {action: index + 1 for index, action in enumerate(before_ranking)}
    after_rank_by_action = {action: index + 1 for index, action in enumerate(after_ranking)}
    selected_before = before_ranking[0] if before_ranking else None
    selected_after = after_ranking[0] if after_ranking else None
    comparisons = []
    for candidate in candidates:
        action_index = _action_index(candidate)
        score_record = profile_scores[action_index]
        before_rank = before_rank_by_action.get(action_index)
        after_rank = after_rank_by_action.get(action_index)
        comparisons.append(
            {
                "action_index": action_index,
                "cell": _cell(candidate),
                "before_rank": before_rank,
                "after_rank": after_rank,
                "rank_delta": (after_rank or 0) - (before_rank or 0),
                "before_score": base_scores[action_index]["total_score"],
                "after_score": score_record["total_score"],
                "score_components": score_record["score_components"],
                "penalty_components": score_record["penalty_components"],
                "reason_codes": score_record["reason_codes"],
            }
        )
    comparisons.sort(key=lambda item: (item["before_rank"] or 9999, item["action_index"]))
    return {
        "profile_id": profile["id"],
        "run_id": run_record["run_id"],
        "scenario_id": str(scenario.get("scenario_id", "")),
        "scenario_group": str(scenario.get("scenario_group", "unknown")),
        "scenario_set": str(run_record["command_args"].get("scenario_set", "unknown")),
        "diagnostic_profile": str(run_record["command_args"].get("diagnostic_profile", "unknown")),
        "top_k": run_record["command_args"].get("top_k"),
        "source_summary_path": run_record["source_summary_path"],
        "selected_cell_before_path_feedback": _list_value(scenario.get("selected_cell_before_path_feedback")),
        "selected_cell_after_path_feedback": _list_value(scenario.get("selected_cell_after_path_feedback")),
        "source_selection_changed_by_path_feedback": bool(scenario.get("selection_changed_by_path_feedback", False)),
        "selected_action_before_profile": selected_before,
        "selected_cell_before_profile": _cell(_candidate_by_action(candidates, selected_before)),
        "selected_action_after_profile": selected_after,
        "selected_cell_after_profile": _cell(_candidate_by_action(candidates, selected_after)),
        "selection_changed_by_profile": selected_before != selected_after,
        "sample_quality_record": _sample_quality_public_record(sample_quality_record),
        "candidate_comparisons": comparisons,
    }


def _score_candidate(
    candidate: dict[str, Any],
    *,
    profile: dict[str, Any],
    sample_quality_record: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    scoring = config["scoring"]
    utility = _finite_float(candidate.get("utility"))
    path_cost = max(0.0, _finite_float(candidate.get("path_cost")))
    path_cost_normalizer = max(1.0, scoring["path_cost_normalizer"])
    risk = max(0.0, _finite_float(candidate.get("risk")))
    score_components = {
        "utility": scoring["utility_weight"] * utility,
        "path_cost": 0.0,
        "risk": 0.0,
    }
    penalty_components = {
        "path_feedback_penalty": 0.0,
        "sample_quality_penalty": 0.0,
        "total_penalty": 0.0,
    }
    reason_codes: list[str] = []
    if profile.get("use_path_feedback"):
        score_components["path_cost"] = -scoring["path_cost_weight"] * min(path_cost / path_cost_normalizer, 10.0)
        score_components["risk"] = -scoring["risk_weight"] * risk
        penalty_components["path_feedback_penalty"] += _path_feedback_penalty(
            candidate,
            scoring=scoring,
            reason_codes=reason_codes,
        )
    if profile.get("use_sample_quality"):
        penalty_components["sample_quality_penalty"] += _sample_quality_penalty(
            sample_quality_record,
            scoring=scoring,
            reason_codes=reason_codes,
        )
    penalty_components["total_penalty"] = (
        penalty_components["path_feedback_penalty"] + penalty_components["sample_quality_penalty"]
    )
    total_score = sum(score_components.values()) - penalty_components["total_penalty"]
    return {
        "total_score": _finite_float(total_score),
        "score_components": {key: _finite_float(value) for key, value in score_components.items()},
        "penalty_components": {key: _finite_float(value) for key, value in penalty_components.items()},
        "reason_codes": _dedupe(reason_codes) or ["candidate_passed"],
    }


def _legacy_score(candidate: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
    utility = config["scoring"]["utility_weight"] * _finite_float(candidate.get("utility"))
    return {
        "total_score": _finite_float(utility),
        "score_components": {"utility": _finite_float(utility), "path_cost": 0.0, "risk": 0.0},
        "penalty_components": {"path_feedback_penalty": 0.0, "sample_quality_penalty": 0.0, "total_penalty": 0.0},
        "reason_codes": ["candidate_passed"],
    }


def _path_feedback_penalty(candidate: dict[str, Any], *, scoring: dict[str, float], reason_codes: list[str]) -> float:
    penalty = 0.0
    interpretation = candidate.get("diagnostic_interpretation")
    interpretation = interpretation if isinstance(interpretation, dict) else {}
    if candidate.get("reachable") is False:
        penalty += scoring["unreachable_penalty"]
        _append_reason(reason_codes, "unreachable_candidate")
        _append_reason(reason_codes, "path_planning_failure")
    if candidate.get("failure_reason"):
        penalty += scoring["failure_penalty"]
        _append_reason(reason_codes, "path_planning_failure")
    if candidate.get("replan_required") is True:
        penalty += scoring["replan_penalty"]
        _append_reason(reason_codes, "replan_required")
    if _finite_float(candidate.get("path_cost")) >= scoring["high_path_cost_threshold"]:
        penalty += scoring["high_path_cost_penalty"]
        _append_reason(reason_codes, "high_path_cost")
    if interpretation.get("open_grid_fallback_used") is True:
        penalty += scoring["open_grid_fallback_penalty"]
        _append_reason(reason_codes, "open_grid_fallback")
    if interpretation.get("iris_fallback_used") is True:
        penalty += scoring["iris_fallback_penalty"]
        _append_reason(reason_codes, "iris_fallback")
    if interpretation.get("region_graph_fallback_used") is True:
        penalty += scoring["region_graph_fallback_penalty"]
        _append_reason(reason_codes, "region_graph_fallback")
    if interpretation.get("region_graph_start_goal_connected") is False:
        penalty += scoring["region_graph_disconnected_penalty"]
        _append_reason(reason_codes, "region_graph_disconnected")
    return penalty


def _sample_quality_penalty(
    sample_quality_record: dict[str, Any],
    *,
    scoring: dict[str, float],
    reason_codes: list[str],
) -> float:
    if not isinstance(sample_quality_record, dict) or not sample_quality_record:
        _append_reason(reason_codes, "sample_quality_record_missing")
        return 0.0
    action = str(sample_quality_record.get("action", sample_quality_record.get("decision", "keep")))
    for reason in _string_list(sample_quality_record.get("reason_codes", [])):
        _append_reason(reason_codes, reason)
    if action == "exclude":
        _append_reason(reason_codes, "sample_quality_exclude")
        return scoring["sample_quality_exclude_penalty"]
    if action == "downweight":
        _append_reason(reason_codes, "sample_quality_downweight")
        return scoring["sample_quality_downweight_penalty"]
    return 0.0


def _ranking(candidates: list[dict[str, Any]], score_records: dict[int, dict[str, Any]]) -> list[int]:
    return [
        _action_index(candidate)
        for candidate in sorted(
            candidates,
            key=lambda item: (
                -score_records[_action_index(item)]["total_score"],
                _action_index(item),
                _cell(item),
            ),
        )
    ]


def _decision_aggregate(decisions: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for decision in decisions:
        buckets.setdefault(str(decision.get(field, "unknown")), []).append(decision)
    return {key: _decision_bucket(records) for key, records in sorted(buckets.items())}


def _decision_bucket(records: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts = Counter(
        reason
        for decision in records
        for candidate in decision["candidate_comparisons"]
        for reason in candidate["reason_codes"]
    )
    return {
        "scenario_count": len(records),
        "candidate_count": sum(len(record["candidate_comparisons"]) for record in records),
        "selection_changed_count": sum(1 for record in records if record["selection_changed_by_profile"]),
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "run_ids": _unique(record["run_id"] for record in records),
        "scenario_ids": _unique(record["scenario_id"] for record in records),
        "source_summary_paths": _unique(record["source_summary_path"] for record in records),
    }


def _build_selection_comparison(
    *,
    status: str,
    global_reason_codes: list[str],
    profile_results: dict[str, dict[str, Any]],
    config: dict[str, Any],
    batch_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for profile_id, result in profile_results.items():
        for decision in result.get("decisions", []):
            key = (str(decision.get("run_id", "")), str(decision.get("scenario_id", "")))
            record = by_key.setdefault(
                key,
                {
                    "run_id": key[0],
                    "scenario_id": key[1],
                    "scenario_group": decision.get("scenario_group", "unknown"),
                    "selected_actions_by_profile": {},
                    "selected_cells_by_profile": {},
                    "selection_changed_by_profile": {},
                    "source_summary_paths": [],
                },
            )
            record["selected_actions_by_profile"][profile_id] = decision.get("selected_action_after_profile")
            record["selected_cells_by_profile"][profile_id] = decision.get("selected_cell_after_profile")
            record["selection_changed_by_profile"][profile_id] = bool(decision.get("selection_changed_by_profile"))
            _append_unique(record["source_summary_paths"], decision.get("source_summary_path"))

    observation_records = {}
    by_scenario: dict[str, dict[str, Any]] = {}
    for key, record in sorted(by_key.items()):
        actions = {json.dumps(value) for value in record["selected_actions_by_profile"].values()}
        cells = {json.dumps(value) for value in record["selected_cells_by_profile"].values()}
        changed = len(actions) > 1 or len(cells) > 1
        reason_codes = ["no_training_metric_evaluated"]
        reason_codes.append("selection_changed_between_profiles" if changed else "selection_stable_across_profiles")
        observation = {
            **record,
            "selection_changed_between_profiles": changed,
            "reason_codes": reason_codes,
        }
        observation_key = f"{key[0]}::{key[1]}"
        observation_records[observation_key] = observation
        scenario = by_scenario.setdefault(
            key[1],
            {
                "scenario_id": key[1],
                "scenario_group": record.get("scenario_group", "unknown"),
                "observation_count": 0,
                "selection_changed_observation_count": 0,
                "run_ids": [],
                "source_summary_paths": [],
                "selected_actions_by_profile": {},
                "selected_cells_by_profile": {},
            },
        )
        scenario["observation_count"] += 1
        if changed:
            scenario["selection_changed_observation_count"] += 1
        _append_unique(scenario["run_ids"], key[0])
        for source_path in record["source_summary_paths"]:
            _append_unique(scenario["source_summary_paths"], source_path)
        for profile_id, action in record["selected_actions_by_profile"].items():
            scenario["selected_actions_by_profile"].setdefault(profile_id, [])
            _append_unique(scenario["selected_actions_by_profile"][profile_id], action)
        for profile_id, cell in record["selected_cells_by_profile"].items():
            scenario["selected_cells_by_profile"].setdefault(profile_id, [])
            _append_unique(scenario["selected_cells_by_profile"][profile_id], cell)

    scenario_records = {}
    for scenario_id, scenario in sorted(by_scenario.items()):
        changed = bool(scenario["selection_changed_observation_count"])
        reason_codes = ["no_training_metric_evaluated"]
        reason_codes.append("selection_changed_between_profiles" if changed else "selection_stable_across_profiles")
        scenario_records[scenario_id] = {
            **scenario,
            "selection_changed_between_profiles": changed,
            "reason_codes": reason_codes,
        }
    changed_count = sum(1 for record in scenario_records.values() if record["selection_changed_between_profiles"])
    comparison_reason_codes = ["no_training_metric_evaluated"]
    comparison_reason_codes.append(
        "selection_changed_by_policy_decision_profile" if changed_count else "selection_stable_across_profiles"
    )
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(global_reason_codes),
        "batch_root": _display_path(batch_root, repo_root),
        "quality_signal_use": "policy_candidate_sorting_comparison_audit_only",
        "not_real_world_performance_claim": True,
        "does_not_run_training": True,
        "no_training_metric_evaluated": True,
        "profile_ids": [profile["id"] for profile in config["profiles"]],
        "profile_summaries": {
            profile_id: {
                "scenario_count": result["scenario_count"],
                "candidate_count": result["candidate_count"],
                "selection_changed_count": result["selection_changed_count"],
                "average_abs_rank_delta": result["average_abs_rank_delta"],
                "reason_code_counts": dict(result["reason_code_counts"]),
            }
            for profile_id, result in profile_results.items()
        },
        "by_observation": observation_records,
        "by_scenario_id": scenario_records,
        "comparison": {
            "legacy_profile_id": config["legacy_profile_id"],
            "feedback_aware_profile_id": config["feedback_aware_profile_id"],
            "sample_quality_aware_profile_id": config["sample_quality_aware_profile_id"],
            "scenario_count": len(scenario_records),
            "selection_changed_scenario_count": changed_count,
            "selection_stable_scenario_count": len(scenario_records) - changed_count,
            "reason_codes": comparison_reason_codes,
            "best_decision_profile_stability": "audit_comparable_no_training_metric_evaluated",
            "single_metric_accidental_improvement": "not_evaluated_no_training_metric",
        },
    }


def _source_summary_records(sources: dict[str, Any], run_records: list[dict[str, Any]]) -> dict[str, Any]:
    records = dict(sources["source_records"])
    records["path_feedback_summary_paths"] = [
        record["source_summary_path"] for record in run_records if record["source_summary_path"]
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


def _sample_quality_public_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict) or not record:
        return {}
    return {
        "action": record.get("action", record.get("decision")),
        "decision": record.get("decision", record.get("action")),
        "sample_weight": _finite_float(record.get("sample_weight", 1.0)),
        "reason_codes": _string_list(record.get("reason_codes", [])),
    }


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": config["schema_version"],
        "legacy_profile_id": config["legacy_profile_id"],
        "feedback_aware_profile_id": config["feedback_aware_profile_id"],
        "sample_quality_aware_profile_id": config["sample_quality_aware_profile_id"],
        "validation": dict(config.get("validation", {})),
        "scoring": dict(config.get("scoring", {})),
        "profiles": [dict(profile) for profile in config["profiles"]],
    }


def _output_files(batch_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    output_config = config.get("output_files", {})
    if not isinstance(output_config, dict):
        output_config = {}
    robustness_name = str(output_config.get("robustness_summary", "policy-decision-robustness-summary.json"))
    comparison_name = str(
        output_config.get("selection_comparison_summary", "policy-decision-selection-comparison-summary.json")
    )
    return {
        "robustness": batch_root / robustness_name,
        "comparison": batch_root / comparison_name,
    }


def _require_current_git_match(config: dict[str, Any]) -> bool:
    validation = config.get("validation", {})
    if not isinstance(validation, dict):
        return True
    return bool(validation.get("require_current_git_match", True))


def _git_snapshot(repo_root: Path) -> dict[str, Any]:
    return {
        "parent": _git_repo_state(repo_root, repo_root=repo_root),
        "submodules": {name: _git_repo_state(repo_root / name, repo_root=repo_root) for name in SUBMODULES},
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


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{label} must be a non-empty string")
    return value


def _float_value(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be numeric") from exc
    if not isfinite(number):
        raise ConfigError(f"{label} must be finite")
    return number


def _finite_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if isfinite(number) else 0.0


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _action_index(candidate: dict[str, Any]) -> int:
    return _int_value(candidate.get("action_index"))


def _candidate_by_action(candidates: list[dict[str, Any]], action_index: int | None) -> dict[str, Any]:
    for candidate in candidates:
        if action_index is not None and _action_index(candidate) == action_index:
            return candidate
    return {}


def _cell(candidate: dict[str, Any]) -> list[int] | None:
    cell = candidate.get("cell") if isinstance(candidate, dict) else None
    if not isinstance(cell, list | tuple) or len(cell) < 2:
        return None
    return [_int_value(cell[0]), _int_value(cell[1])]


def _list_value(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list | tuple) else []


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


def _append_unique(values: list[Any], value: Any) -> None:
    if value in (None, ""):
        return
    if value not in values:
        values.append(value)


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
