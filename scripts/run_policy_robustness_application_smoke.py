from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIG_SCHEMA_VERSION = "policy-robustness-application-smoke-config/v1"
APPLICATION_SCHEMA_VERSION = "policy-robustness-application-summary/v1"
COMPARISON_SCHEMA_VERSION = "policy-robustness-application-comparison-summary/v1"
ROBUSTNESS_SCHEMA_VERSION = "policy-decision-robustness-summary/v1"
ROBUSTNESS_COMPARISON_SCHEMA_VERSION = "policy-decision-selection-comparison-summary/v1"
SAMPLE_QUALITY_SCHEMA_VERSION = "sample-quality-training-application-summary/v1"
TRAINING_SELECTION_SCHEMA_VERSION = "training-selection-stability-summary/v1"
PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION = "path-feedback-summary/v1"
SUBMODULES = ("dev-platform-constraints", "model-explorer", "path-planner")
BATCH_SOURCE_FILES = {
    "policy_decision_selection_comparison_summary": (
        "policy-decision-selection-comparison-summary.json",
        ROBUSTNESS_COMPARISON_SCHEMA_VERSION,
    ),
    "sample_quality_training_application_summary": (
        "sample-quality-training-application-summary.json",
        SAMPLE_QUALITY_SCHEMA_VERSION,
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
        description="Apply Policy-Robustness Application Smoke v1 to current robustness JSON outputs."
    )
    parser.add_argument("--batch-root", required=True, help="Batch root containing policy robustness inputs.")
    parser.add_argument("--robustness-summary", required=True, help="policy-decision-robustness-summary/v1 JSON.")
    parser.add_argument("--config", required=True, help="Policy robustness application smoke config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned output paths.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without writing outputs.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    batch_root = _resolve_path(args.batch_root, repo_root)
    robustness_summary_path = _resolve_path(args.robustness_summary, repo_root)
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    outputs = analyze_application_smoke(
        batch_root=batch_root,
        robustness_summary_path=robustness_summary_path,
        config=config,
        repo_root=repo_root,
    )
    output_files = _output_files(batch_root, config)
    status = outputs["application"]["status"]
    validation_message = {
        "status": "config validated" if status == "passed" else "validation failed",
        "batch_root": _display_path(batch_root, repo_root),
        "robustness_summary": _display_path(robustness_summary_path, repo_root),
        "config": _display_path(config_path, repo_root),
        "baseline_profile_id": config["baseline_profile_id"],
        "selected_profile_id": config["selected_profile_id"],
        "reason_codes": outputs["application"]["reason_codes"],
        "application_summary": _display_path(output_files["application"], repo_root),
        "comparison_summary": _display_path(output_files["comparison"], repo_root),
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
                            "comparison_summary": _display_path(output_files["comparison"], repo_root),
                        },
                        "selected_profile_id": config["selected_profile_id"],
                    },
                    ensure_ascii=False,
                )
            )
        return 1 if status == "failed" else 0

    _write_json(output_files["application"], outputs["application"])
    _write_json(output_files["comparison"], outputs["comparison"])
    print(
        json.dumps(
            {
                "status": status,
                "policy_robustness_application_summary": _display_path(output_files["application"], repo_root),
                "policy_robustness_application_comparison_summary": _display_path(output_files["comparison"], repo_root),
                "failure_reason_code_counts": outputs["application"]["failure_reason_code_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if status == "failed" else 0


def analyze_application_smoke(
    *,
    batch_root: Path,
    robustness_summary_path: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    global_reason_codes: list[str] = []
    source_records: dict[str, Any] = {}
    robustness = _load_source_json(
        robustness_summary_path,
        label="policy_decision_robustness_summary",
        expected_schema=ROBUSTNESS_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=global_reason_codes,
        source_records=source_records,
    )
    batch_payloads = _load_batch_sources(
        batch_root,
        repo_root=repo_root,
        reason_codes=global_reason_codes,
        source_records=source_records,
    )
    _inspect_upstream_statuses(
        robustness=robustness,
        batch_payloads=batch_payloads,
        reason_codes=global_reason_codes,
    )

    current_git = _git_snapshot(repo_root)
    robustness_git = robustness.get("git_provenance") if isinstance(robustness.get("git_provenance"), dict) else {}
    robustness_current_git = robustness_git.get("current") if isinstance(robustness_git.get("current"), dict) else {}
    if _require_current_git_match(config) and robustness_current_git and not _git_snapshots_match(robustness_current_git, current_git):
        _append_reason(global_reason_codes, "current_git_provenance_mismatch")
    if robustness_git and robustness_git.get("current_matches_batch") is not True:
        _append_reason(global_reason_codes, "git_provenance_mismatch")
    if robustness_git and robustness_git.get("runs_match_batch") is not True:
        _append_reason(global_reason_codes, "git_provenance_mismatch")

    path_feedback = _load_path_feedback_sources(
        robustness,
        repo_root=repo_root,
        reason_codes=global_reason_codes,
    )
    source_records["path_feedback_summaries"] = path_feedback["source_records"]

    baseline_decisions = _decisions_by_key(robustness, config["baseline_profile_id"])
    selected_decisions = _decisions_by_key(robustness, config["selected_profile_id"])
    if not baseline_decisions:
        _append_reason(global_reason_codes, "baseline_profile_missing_or_empty")
    if not selected_decisions:
        _append_reason(global_reason_codes, "selected_profile_missing_or_empty")

    decision_records = _build_decision_records(
        baseline_decisions=baseline_decisions,
        selected_decisions=selected_decisions,
        path_feedback_summaries=path_feedback["payloads"],
        config=config,
        reason_codes=global_reason_codes,
    )
    all_decision_reason_codes = Counter(
        reason for record in decision_records for reason in record["reason_codes"]
    )
    failure_reason_counts = Counter(global_reason_codes)
    status = "failed" if global_reason_codes else "passed"

    application = {
        "schema_version": APPLICATION_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(global_reason_codes),
        "failure_reason_code_counts": dict(sorted(failure_reason_counts.items())),
        "batch_root": _display_path(batch_root, repo_root),
        "baseline_profile_id": config["baseline_profile_id"],
        "applied_profile_id": config["selected_profile_id"],
        "application_scope": "lightweight_decision_log_smoke_only",
        "quality_signal_use": "policy_robustness_application_smoke_only",
        "no_large_scale_training": True,
        "no_real_world_performance_claim": True,
        "no_single_metric_improvement_claim": True,
        "does_not_modify_ppo": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "non_goals": list(config.get("non_goals", [])),
        "config": _public_config(config),
        "source_summaries": source_records,
        "acceptance_metadata": _acceptance_metadata_records(robustness, path_feedback["payloads"]),
        "git_provenance": {
            "robustness": dict(robustness_git),
            "current": current_git,
            "current_matches_robustness": _git_snapshots_match(robustness_current_git, current_git)
            if robustness_current_git
            else None,
        },
        "episode_count": len(decision_records),
        "decision_count": len(decision_records),
        "decision_changed_count": sum(1 for record in decision_records if record["decision_changed"]),
        "decision_stable_count": sum(1 for record in decision_records if not record["decision_changed"]),
        "reason_code_counts": dict(sorted(all_decision_reason_codes.items())),
        "by_scenario_group": _aggregate_decisions(decision_records, "scenario_group"),
        "by_scenario_id": _aggregate_decisions(decision_records, "scenario_id"),
        "decision_records": decision_records,
    }
    comparison = _build_comparison_summary(
        status=status,
        global_reason_codes=global_reason_codes,
        decision_records=decision_records,
        config=config,
        batch_root=batch_root,
        repo_root=repo_root,
    )
    return {"application": application, "comparison": comparison}


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
    baseline_profile_id = _required_string(payload.get("baseline_profile_id"), "baseline_profile_id")
    selected_profile_id = _required_string(payload.get("selected_profile_id"), "selected_profile_id")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ConfigError("profiles must be a non-empty array")
    profile_ids = []
    normalized_profiles = []
    for index, profile in enumerate(profiles):
        if not isinstance(profile, dict):
            raise ConfigError(f"profiles[{index}] must be an object")
        profile_id = _required_string(profile.get("id"), f"profiles[{index}].id")
        profile_ids.append(profile_id)
        normalized_profiles.append({"id": profile_id, "description": str(profile.get("description", ""))})
    if len(profile_ids) != len(set(profile_ids)):
        raise ConfigError("profile ids must be unique")
    if baseline_profile_id not in profile_ids:
        raise ConfigError("baseline_profile_id must reference a configured profile")
    if selected_profile_id not in profile_ids:
        raise ConfigError("selected_profile_id must reference a configured profile")
    config = dict(payload)
    config["profiles"] = normalized_profiles
    return config


def _load_batch_sources(
    batch_root: Path,
    *,
    repo_root: Path,
    reason_codes: list[str],
    source_records: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for key, (filename, expected_schema) in BATCH_SOURCE_FILES.items():
        payloads[key] = _load_source_json(
            batch_root / filename,
            label=key,
            expected_schema=expected_schema,
            repo_root=repo_root,
            reason_codes=reason_codes,
            source_records=source_records,
        )
    return payloads


def _load_source_json(
    path: Path,
    *,
    label: str,
    expected_schema: str,
    repo_root: Path,
    reason_codes: list[str],
    source_records: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    status = "passed"
    schema_version = None
    if not path.is_file():
        status = "missing"
        _append_reason(reason_codes, f"{label}_missing")
    else:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status = "failed"
            _append_reason(reason_codes, f"{label}_invalid_json")
        else:
            if not isinstance(loaded, dict):
                status = "failed"
                _append_reason(reason_codes, f"{label}_not_object")
            else:
                payload = loaded
                schema_version = payload.get("schema_version")
                if schema_version != expected_schema:
                    status = "failed"
                    _append_reason(reason_codes, f"{label}_schema_mismatch")
    source_records[label] = {
        "path": _display_path(path, repo_root),
        "status": status,
        "schema_version": schema_version,
        "expected_schema_version": expected_schema,
    }
    return payload


def _inspect_upstream_statuses(
    *,
    robustness: dict[str, Any],
    batch_payloads: dict[str, dict[str, Any]],
    reason_codes: list[str],
) -> None:
    if robustness and robustness.get("status") != "passed":
        _append_reason(reason_codes, "policy_decision_robustness_summary_failed")
        for reason in _string_list(robustness.get("reason_codes", [])):
            _append_reason(reason_codes, reason)
    for key, payload in batch_payloads.items():
        if payload and payload.get("status") != "passed":
            _append_reason(reason_codes, f"{key}_failed")
            for reason in _string_list(payload.get("reason_codes", [])):
                _append_reason(reason_codes, reason)


def _load_path_feedback_sources(
    robustness: dict[str, Any],
    *,
    repo_root: Path,
    reason_codes: list[str],
) -> dict[str, Any]:
    paths = _path_feedback_paths(robustness, repo_root=repo_root)
    payloads: dict[str, dict[str, Any]] = {}
    source_records: dict[str, dict[str, Any]] = {}
    for path in paths:
        display = _display_path(path, repo_root)
        payload = _load_path_feedback_summary(path, reason_codes)
        payloads[display] = payload
        schema_version = payload.get("schema_version") if payload else None
        status = "missing" if not path.is_file() else "failed" if schema_version != PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION else "passed"
        if path.is_file() and schema_version != PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION:
            _append_reason(reason_codes, "source_summary_schema_mismatch")
        if payload:
            _inspect_open_grid(payload, reason_codes)
        source_records[display] = {
            "path": display,
            "status": status,
            "schema_version": schema_version,
            "expected_schema_version": PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION,
        }
    return {"payloads": payloads, "source_records": source_records}


def _path_feedback_paths(robustness: dict[str, Any], *, repo_root: Path) -> list[Path]:
    values: list[str] = []
    source_summaries = robustness.get("source_summaries") if isinstance(robustness.get("source_summaries"), dict) else {}
    for value in source_summaries.get("path_feedback_summary_paths", []):
        if value not in values:
            values.append(str(value))
    profiles = robustness.get("profiles") if isinstance(robustness.get("profiles"), dict) else {}
    for profile in profiles.values():
        decisions = profile.get("decisions") if isinstance(profile, dict) else []
        for decision in decisions if isinstance(decisions, list) else []:
            if isinstance(decision, dict) and decision.get("source_summary_path") not in (None, ""):
                text = str(decision["source_summary_path"])
                if text not in values:
                    values.append(text)
    return [_resolve_path(value, repo_root) for value in values]


def _load_path_feedback_summary(path: Path, reason_codes: list[str]) -> dict[str, Any]:
    if not path.is_file():
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


def _inspect_open_grid(summary: dict[str, Any], reason_codes: list[str]) -> None:
    acceptance_metadata = summary.get("acceptance_metadata") if isinstance(summary.get("acceptance_metadata"), dict) else {}
    values = [summary.get("open_grid_fallback_used"), acceptance_metadata.get("open_grid_fallback_used")]
    gate = acceptance_metadata.get("open_grid_fallback_used_gate")
    if isinstance(gate, dict):
        values.append(gate.get("actual"))
        if gate.get("status") != "passed":
            _append_reason(reason_codes, "open_grid_fallback_gate_failed")
    summary_gate = summary.get("open_grid_fallback_used_gate")
    if isinstance(summary_gate, dict):
        values.append(summary_gate.get("actual"))
    if any(value is True for value in values):
        _append_reason(reason_codes, "open_grid_fallback_used")


def _decisions_by_key(robustness: dict[str, Any], profile_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    profiles = robustness.get("profiles") if isinstance(robustness.get("profiles"), dict) else {}
    profile = profiles.get(profile_id) if isinstance(profiles, dict) else {}
    decisions = profile.get("decisions") if isinstance(profile, dict) else []
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for decision in decisions if isinstance(decisions, list) else []:
        if not isinstance(decision, dict):
            continue
        result[(str(decision.get("run_id", "")), str(decision.get("scenario_id", "")))] = decision
    return result


def _build_decision_records(
    *,
    baseline_decisions: dict[tuple[str, str], dict[str, Any]],
    selected_decisions: dict[tuple[str, str], dict[str, Any]],
    path_feedback_summaries: dict[str, dict[str, Any]],
    config: dict[str, Any],
    reason_codes: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key, selected in sorted(selected_decisions.items()):
        baseline = baseline_decisions.get(key)
        if not baseline:
            _append_reason(reason_codes, "baseline_decision_missing")
            continue
        source_path = str(selected.get("source_summary_path", baseline.get("source_summary_path", "")))
        path_summary = _matching_path_summary(source_path, path_feedback_summaries)
        scenario = _matching_scenario(path_summary, str(selected.get("scenario_id", "")))
        mismatches = _metadata_mismatches(selected, path_summary)
        if mismatches:
            _append_reason(reason_codes, "acceptance_metadata_mismatch")
        selected_candidate = _candidate_record(
            selected.get("candidate_comparisons"),
            selected.get("selected_action_after_profile"),
        )
        sample_quality = selected.get("sample_quality_record") if isinstance(selected.get("sample_quality_record"), dict) else {}
        sample_quality_reason_codes = _string_list(sample_quality.get("reason_codes", []))
        record_reason_codes = _dedupe(
            list(_string_list(selected_candidate.get("reason_codes", [])))
            + sample_quality_reason_codes
            + [
                "decision_changed_by_robustness_profile"
                if _decision_changed(baseline, selected)
                else "decision_stable_across_smoke_profiles"
            ]
        )
        records.append(
            {
                "episode_id": f"{key[0]}::{key[1]}",
                "run_id": key[0],
                "scenario_id": key[1],
                "scenario_group": str(selected.get("scenario_group", "unknown")),
                "scenario_set": str(selected.get("scenario_set", "unknown")),
                "diagnostic_profile": str(selected.get("diagnostic_profile", "unknown")),
                "top_k": selected.get("top_k"),
                "source_summary_path": source_path,
                "baseline_profile_id": config["baseline_profile_id"],
                "applied_profile_id": config["selected_profile_id"],
                "selected_action_before": baseline.get("selected_action_after_profile"),
                "selected_action_after": selected.get("selected_action_after_profile"),
                "selected_cell_before": baseline.get("selected_cell_after_profile"),
                "selected_cell_after": selected.get("selected_cell_after_profile"),
                "decision_changed": _decision_changed(baseline, selected),
                "rollback_available": True,
                "metadata_mismatches": mismatches,
                "failure_replan_exposure": _failure_replan_exposure(scenario, selected_candidate),
                "sample_quality_action": sample_quality.get("action", sample_quality.get("decision")),
                "sample_quality_weight": _finite_float(sample_quality.get("sample_weight", 1.0)),
                "sample_quality_reason_codes": sample_quality_reason_codes,
                "reason_codes": record_reason_codes,
                "candidate_comparisons": list(selected.get("candidate_comparisons", []))
                if isinstance(selected.get("candidate_comparisons"), list)
                else [],
            }
        )
    return records


def _matching_path_summary(source_path: str, path_feedback_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for path, summary in path_feedback_summaries.items():
        if path == source_path or Path(path).as_posix() == Path(source_path).as_posix():
            return summary
    if len(path_feedback_summaries) == 1:
        return next(iter(path_feedback_summaries.values()))
    return {}


def _matching_scenario(summary: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    scenarios = summary.get("scenarios") if isinstance(summary, dict) else []
    for scenario in scenarios if isinstance(scenarios, list) else []:
        if isinstance(scenario, dict) and str(scenario.get("scenario_id", "")) == scenario_id:
            return scenario
    return {}


def _metadata_mismatches(decision: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, Any]]:
    if not summary:
        return []
    acceptance_metadata = summary.get("acceptance_metadata") if isinstance(summary.get("acceptance_metadata"), dict) else {}
    mismatches = []
    for field in ("scenario_set", "diagnostic_profile", "top_k"):
        expected = decision.get(field)
        if expected is None:
            continue
        summary_value = summary.get(field)
        metadata_value = acceptance_metadata.get(field)
        if str(summary_value) != str(expected) or str(metadata_value) != str(expected):
            mismatches.append(
                {
                    "field": field,
                    "expected": expected,
                    "summary": summary_value,
                    "acceptance_metadata": metadata_value,
                }
            )
    return mismatches


def _candidate_record(candidates: Any, action_index: Any) -> dict[str, Any]:
    for candidate in candidates if isinstance(candidates, list) else []:
        if isinstance(candidate, dict) and str(candidate.get("action_index")) == str(action_index):
            return candidate
    return {}


def _failure_replan_exposure(scenario: dict[str, Any], selected_candidate: dict[str, Any]) -> dict[str, Any]:
    feedback = scenario.get("path_feedback") if isinstance(scenario.get("path_feedback"), dict) else {}
    reason_codes = _string_list(selected_candidate.get("reason_codes", []))
    return {
        "path_feedback_failure_count": _int_value(feedback.get("failure_count")),
        "path_feedback_replan_count": _int_value(feedback.get("replan_count")),
        "selected_candidate_path_planning_failure": "path_planning_failure" in reason_codes,
        "selected_candidate_replan_required": "replan_required" in reason_codes,
        "selected_candidate_iris_fallback": "iris_fallback" in reason_codes,
        "selected_candidate_region_graph_fallback": "region_graph_fallback" in reason_codes,
        "selected_candidate_region_graph_disconnected": "region_graph_disconnected" in reason_codes,
    }


def _decision_changed(baseline: dict[str, Any], selected: dict[str, Any]) -> bool:
    return (
        baseline.get("selected_action_after_profile") != selected.get("selected_action_after_profile")
        or baseline.get("selected_cell_after_profile") != selected.get("selected_cell_after_profile")
    )


def _aggregate_decisions(records: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(str(record.get(field, "unknown")), []).append(record)
    return {key: _decision_bucket(bucket) for key, bucket in sorted(buckets.items())}


def _decision_bucket(records: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts = Counter(reason for record in records for reason in record["reason_codes"])
    return {
        "decision_count": len(records),
        "decision_changed_count": sum(1 for record in records if record["decision_changed"]),
        "decision_stable_count": sum(1 for record in records if not record["decision_changed"]),
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "run_ids": _unique(record["run_id"] for record in records),
        "scenario_ids": _unique(record["scenario_id"] for record in records),
        "source_summary_paths": _unique(record["source_summary_path"] for record in records),
    }


def _build_comparison_summary(
    *,
    status: str,
    global_reason_codes: list[str],
    decision_records: list[dict[str, Any]],
    config: dict[str, Any],
    batch_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    changed_count = sum(1 for record in decision_records if record["decision_changed"])
    comparison_reason_codes = [
        "no_large_scale_training",
        "no_real_world_performance_claim",
        "no_single_metric_improvement_claim",
    ]
    comparison_reason_codes.append(
        "decision_changed_by_robustness_profile" if changed_count else "decision_stable_across_smoke_profiles"
    )
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(global_reason_codes),
        "batch_root": _display_path(batch_root, repo_root),
        "baseline_profile_id": config["baseline_profile_id"],
        "applied_profile_id": config["selected_profile_id"],
        "application_scope": "lightweight_decision_log_smoke_only",
        "no_large_scale_training": True,
        "no_real_world_performance_claim": True,
        "no_single_metric_improvement_claim": True,
        "rollback_profile_id": config["baseline_profile_id"],
        "rollback_available": True,
        "decision_count": len(decision_records),
        "decision_changed_count": changed_count,
        "decision_stable_count": len(decision_records) - changed_count,
        "by_scenario_group": _aggregate_decisions(decision_records, "scenario_group"),
        "by_scenario_id": _aggregate_decisions(decision_records, "scenario_id"),
        "comparison": {
            "baseline_profile_id": config["baseline_profile_id"],
            "applied_profile_id": config["selected_profile_id"],
            "decision_count": len(decision_records),
            "decision_changed_count": changed_count,
            "decision_stable_count": len(decision_records) - changed_count,
            "reason_codes": comparison_reason_codes,
            "single_metric_accidental_improvement": "not_evaluated_smoke_only",
        },
    }


def _acceptance_metadata_records(
    robustness: dict[str, Any],
    path_feedback_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    metadata = robustness.get("acceptance_metadata") if isinstance(robustness.get("acceptance_metadata"), dict) else {}
    if metadata:
        result["from_robustness"] = metadata
    result["path_feedback"] = {
        path: summary.get("acceptance_metadata", {})
        for path, summary in path_feedback_summaries.items()
        if isinstance(summary, dict)
    }
    return result


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": config["schema_version"],
        "baseline_profile_id": config["baseline_profile_id"],
        "selected_profile_id": config["selected_profile_id"],
        "validation": dict(config.get("validation", {})),
        "profiles": [dict(profile) for profile in config["profiles"]],
    }


def _output_files(batch_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    output_config = config.get("output_files", {})
    if not isinstance(output_config, dict):
        output_config = {}
    return {
        "application": batch_root / str(output_config.get("application_summary", "policy-robustness-application-summary.json")),
        "comparison": batch_root / str(output_config.get("comparison_summary", "policy-robustness-application-comparison-summary.json")),
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


def _finite_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


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


def _append_reason(reason_codes: list[str], code: str) -> None:
    if code not in reason_codes:
        reason_codes.append(code)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
