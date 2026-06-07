from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIG_SCHEMA_VERSION = "channel-aware-policy-target-selection-evidence-config/v1"
SUMMARY_SCHEMA_VERSION = "channel-aware-policy-target-selection-evidence-summary/v1"
READINESS_SCHEMA_VERSION = "channel-aware-training-readiness-summary/v1"
APPLICATION_SCHEMA_VERSION = "policy-robustness-application-summary/v1"
SELECTION_COMPARISON_SCHEMA_VERSION = "policy-decision-selection-comparison-summary/v1"
CHANNEL_AUDIT_SCHEMA_VERSION = "channel-aware-decision-audit/v1"
SUBMODULES = ("dev-platform-constraints", "model-explorer", "path-planner")


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Explain channel-aware policy target selection evidence without changing training or planner behavior."
    )
    parser.add_argument("--batch-root", required=True, help="Batch root containing channel-aware audit summaries.")
    parser.add_argument(
        "--readiness-summary",
        help="channel-aware-training-readiness-summary/v1 JSON. Defaults to <batch-root>/channel-aware-training-readiness-summary.json.",
    )
    parser.add_argument(
        "--application-summary",
        help="policy-robustness-application-summary/v1 JSON. Defaults to <batch-root>/policy-robustness-application-summary.json.",
    )
    parser.add_argument(
        "--selection-comparison-summary",
        help="policy-decision-selection-comparison-summary/v1 JSON. Defaults to <batch-root>/policy-decision-selection-comparison-summary.json.",
    )
    parser.add_argument("--config", required=True, help="Policy target selection evidence config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned output paths.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without writing outputs.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    batch_root = _resolve_path(args.batch_root, repo_root)
    readiness_summary_path = (
        _resolve_path(args.readiness_summary, repo_root)
        if args.readiness_summary
        else batch_root / "channel-aware-training-readiness-summary.json"
    )
    application_summary_path = (
        _resolve_path(args.application_summary, repo_root)
        if args.application_summary
        else batch_root / "policy-robustness-application-summary.json"
    )
    selection_comparison_summary_path = (
        _resolve_path(args.selection_comparison_summary, repo_root)
        if args.selection_comparison_summary
        else batch_root / "policy-decision-selection-comparison-summary.json"
    )
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    summary = analyze_policy_target_selection_evidence(
        batch_root=batch_root,
        readiness_summary_path=readiness_summary_path,
        application_summary_path=application_summary_path,
        selection_comparison_summary_path=selection_comparison_summary_path,
        config=config,
        repo_root=repo_root,
    )
    output_file = _output_file(batch_root, config)
    validation_message = {
        "status": "config validated" if summary["status"] == "passed" else "validation failed",
        "batch_root": _display_path(batch_root, repo_root),
        "readiness_summary": _display_path(readiness_summary_path, repo_root),
        "application_summary": _display_path(application_summary_path, repo_root),
        "selection_comparison_summary": _display_path(selection_comparison_summary_path, repo_root),
        "config": _display_path(config_path, repo_root),
        "reason_codes": summary["reason_codes"],
        "selected_candidate_changed_rate": summary["selected_candidate_changed_rate"],
        "supports_policy_target_selection_improvement_claim": summary[
            "supports_policy_target_selection_improvement_claim"
        ],
        "policy_target_selection_evidence_summary": _display_path(output_file, repo_root),
    }
    print(json.dumps(validation_message, ensure_ascii=False))

    if args.validate_only or args.dry_run:
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "dry-run",
                        "would_write": {
                            "policy_target_selection_evidence_summary": _display_path(output_file, repo_root),
                        },
                        "recommended_next_adjustment": summary["recommended_next_adjustment"],
                    },
                    ensure_ascii=False,
                )
            )
        return 1 if summary["status"] == "failed" else 0

    _write_json(output_file, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "policy_target_selection_evidence_summary": _display_path(output_file, repo_root),
                "selected_candidate_changed_rate": summary["selected_candidate_changed_rate"],
                "recommended_next_adjustment": summary["recommended_next_adjustment"],
                "failure_reason_code_counts": summary["failure_reason_code_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def analyze_policy_target_selection_evidence(
    *,
    batch_root: Path,
    readiness_summary_path: Path,
    application_summary_path: Path,
    selection_comparison_summary_path: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    global_reason_codes: list[str] = []
    source_summaries: dict[str, Any] = {}
    readiness = _load_source(
        readiness_summary_path,
        label="channel_aware_training_readiness_summary",
        expected_schema=READINESS_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=global_reason_codes,
        source_summaries=source_summaries,
    )
    application = _load_source(
        application_summary_path,
        label="policy_robustness_application_summary",
        expected_schema=APPLICATION_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=global_reason_codes,
        source_summaries=source_summaries,
    )
    selection_comparison = _load_source(
        selection_comparison_summary_path,
        label="policy_decision_selection_comparison_summary",
        expected_schema=SELECTION_COMPARISON_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=global_reason_codes,
        source_summaries=source_summaries,
    )
    if _fail_on_input_failure(config):
        for label, payload in (
            ("policy_robustness_application_summary", application),
            ("policy_decision_selection_comparison_summary", selection_comparison),
        ):
            if payload.get("status") == "failed":
                _append_reason(global_reason_codes, f"{label}_failed")

    current_git = _git_snapshot(repo_root)
    _inspect_git(
        readiness,
        label="channel_aware_training_readiness_summary",
        current_git=current_git,
        config=config,
        reason_codes=global_reason_codes,
    )
    _inspect_git(
        application,
        label="policy_robustness_application_summary",
        current_git=current_git,
        config=config,
        reason_codes=global_reason_codes,
    )
    _inspect_git(
        selection_comparison,
        label="policy_decision_selection_comparison_summary",
        current_git=current_git,
        config=config,
        reason_codes=global_reason_codes,
    )

    channel_application = _channel_application(application, reason_codes=global_reason_codes)
    application_records = [
        _normalize_application_record(record)
        for record in channel_application.get("records", [])
        if isinstance(record, dict)
    ]
    channel_audit = _channel_audit(selection_comparison, reason_codes=global_reason_codes)
    audit_records = [
        _normalize_audit_record(record)
        for record in channel_audit.get("records", [])
        if isinstance(record, dict)
    ]
    if application_records and audit_records and len(application_records) != len(audit_records):
        _append_reason(global_reason_codes, "channel_aware_application_audit_record_count_mismatch")

    selection_evidence = _summarize_selection_evidence(
        audit_records,
        application_records=application_records,
        readiness=readiness,
        config=config,
    )
    status = "failed" if global_reason_codes else "passed"
    failure_reason_counts = Counter(global_reason_codes)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(global_reason_codes),
        "failure_reason_code_counts": dict(sorted(failure_reason_counts.items())),
        "batch_root": _display_path(batch_root, repo_root),
        "application_scope": "channel_aware_policy_target_selection_evidence_audit_only",
        "audit_only": True,
        "quality_signal_use": "policy_target_selection_explanation_only",
        "source_summaries": source_summaries,
        "config": _public_config(config),
        "git_provenance": {
            "current": current_git,
            "readiness": _public_git(readiness),
            "application": _public_git(application),
            "selection_comparison": _public_git(selection_comparison),
        },
        **selection_evidence,
        "path_cost_tradeoff_interpretation": "tradeoff_reason_not_failure",
        "audit_boundaries": {
            "does_not_train_ppo": True,
            "does_not_modify_default_astar": True,
            "does_not_modify_ppo": True,
            "does_not_modify_network": True,
            "does_not_modify_action_space": True,
            "does_not_modify_path_planner_route_contract": True,
            "does_not_modify_model_explorer_contract": True,
            "does_not_modify_path_planner_sidecar_contract": True,
            "no_ackermann_feasible_trajectory_claim": True,
            "no_gcs_control_point_candidate_as_default_execution_trajectory": True,
        },
        "no_ppo_training": True,
        "no_large_scale_training": True,
        "no_real_world_performance_claim": True,
        "does_not_modify_default_astar": True,
        "does_not_modify_ppo": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "does_not_modify_path_planner_route_contract": True,
        "does_not_modify_model_explorer_contract": True,
        "does_not_modify_path_planner_sidecar_contract": True,
        "no_ackermann_feasible_trajectory_claim": True,
        "non_goals": list(config.get("non_goals", [])),
    }


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
    output_files = payload.get("output_files")
    if not isinstance(output_files, dict) or not _nonempty_string(
        output_files.get("policy_target_selection_evidence_summary")
    ):
        raise ConfigError("output_files.policy_target_selection_evidence_summary must be a non-empty string")
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ConfigError("thresholds must be an object")
    blocked_candidate_high_rate = _float_value(
        thresholds.get("blocked_candidate_high_rate", 0.5),
        "thresholds.blocked_candidate_high_rate",
    )
    if blocked_candidate_high_rate < 0.0 or blocked_candidate_high_rate > 1.0:
        raise ConfigError("thresholds.blocked_candidate_high_rate must be between 0 and 1")
    config = dict(payload)
    config["thresholds"] = {"blocked_candidate_high_rate": blocked_candidate_high_rate}
    return config


def _load_source(
    path: Path,
    *,
    label: str,
    expected_schema: str,
    repo_root: Path,
    reason_codes: list[str],
    source_summaries: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {"path": _display_path(path, repo_root), "exists": path.is_file()}
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        source_summaries[label] = record
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _append_reason(reason_codes, f"{label}_invalid_json")
        source_summaries[label] = record
        return {}
    if not isinstance(payload, dict):
        _append_reason(reason_codes, f"{label}_not_object")
        source_summaries[label] = record
        return {}
    record.update(
        {
            "schema_version": payload.get("schema_version"),
            "status": payload.get("status"),
            "reason_codes": _string_list(payload.get("reason_codes", [])),
        }
    )
    if payload.get("schema_version") != expected_schema:
        _append_reason(reason_codes, f"{label}_schema_mismatch")
    source_summaries[label] = record
    return payload


def _inspect_git(
    payload: dict[str, Any],
    *,
    label: str,
    current_git: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if not _require_current_git_match(config):
        return
    git = payload.get("git_provenance") if isinstance(payload.get("git_provenance"), dict) else {}
    current = git.get("current") if isinstance(git.get("current"), dict) else {}
    if current and not _git_snapshots_match(current, current_git):
        _append_reason(reason_codes, "current_git_provenance_mismatch")
        _append_reason(reason_codes, f"{label}_current_git_provenance_mismatch")
    for key in ("current_matches_application", "current_matches_robustness", "current_matches_batch"):
        if git.get(key) is False:
            _append_reason(reason_codes, "git_provenance_mismatch")
            _append_reason(reason_codes, f"{label}_git_provenance_mismatch")
    robustness = git.get("robustness") if isinstance(git.get("robustness"), dict) else {}
    for key in ("current_matches_batch", "runs_match_batch"):
        if robustness.get(key) is False:
            _append_reason(reason_codes, "git_provenance_mismatch")
            _append_reason(reason_codes, f"{label}_git_provenance_mismatch")


def _channel_application(application: dict[str, Any], *, reason_codes: list[str]) -> dict[str, Any]:
    channel = application.get("channel_aware_application")
    if not isinstance(channel, dict):
        _append_reason(reason_codes, "channel_aware_application_missing")
        return {}
    records = channel.get("records")
    if not isinstance(records, list):
        _append_reason(reason_codes, "channel_aware_application_records_invalid")
        return dict(channel, records=[])
    return channel


def _channel_audit(selection_comparison: dict[str, Any], *, reason_codes: list[str]) -> dict[str, Any]:
    audit = selection_comparison.get("channel_aware_decision_audit")
    if not isinstance(audit, dict):
        _append_reason(reason_codes, "channel_aware_decision_audit_missing")
        return {}
    if audit.get("schema_version") != CHANNEL_AUDIT_SCHEMA_VERSION:
        _append_reason(reason_codes, "channel_aware_decision_audit_schema_mismatch")
    records = audit.get("records")
    if not isinstance(records, list):
        _append_reason(reason_codes, "channel_aware_decision_audit_records_invalid")
        return dict(audit, records=[])
    return audit


def _normalize_application_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": _record_key(record),
        "recommendation": str(record.get("recommendation", "needs_more_evidence")),
        "application_action": str(record.get("application_action", "downweight_needs_more_evidence")),
        "application_sample_weight": _safe_float(record.get("application_sample_weight"), 0.5),
        "reason_codes": _string_list(record.get("reason_codes", [])),
        "application_reason_codes": _string_list(record.get("application_reason_codes", [])),
    }


def _normalize_audit_record(record: dict[str, Any]) -> dict[str, Any]:
    reason_codes = _string_list(record.get("reason_codes", []))
    if bool(record.get("path_cost_tradeoff", False)) and "path_cost_tradeoff" not in reason_codes:
        reason_codes.append("path_cost_tradeoff")
    return {
        "key": _record_key(record),
        "context_key": _context_key(record),
        "recommendation": str(record.get("recommendation", "needs_more_evidence")),
        "selected": bool(record.get("selected", False)),
        "selected_candidate_changed": bool(record.get("selected_candidate_changed", False)),
        "astar_selected_cell": _list_value(record.get("astar_selected_cell")),
        "channel_aware_selected_cell": _list_value(record.get("channel_aware_selected_cell")),
        "path_cost_tradeoff": bool(record.get("path_cost_tradeoff", False)),
        "blocker_reason": str(record.get("blocker_reason")) if record.get("blocker_reason") is not None else None,
        "reason_codes": reason_codes,
    }


def _summarize_selection_evidence(
    audit_records: list[dict[str, Any]],
    *,
    application_records: list[dict[str, Any]],
    readiness: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    app_by_key = {record["key"]: record for record in application_records}
    merged_records = []
    for record in audit_records:
        app = app_by_key.get(record["key"], {})
        merged = dict(record)
        merged["application_action"] = app.get("application_action", _application_action(record["recommendation"]))
        merged["application_sample_weight"] = app.get("application_sample_weight", _application_weight(merged["application_action"]))
        merged["application_reason_codes"] = app.get("application_reason_codes", [])
        merged_records.append(merged)

    record_count = len(merged_records)
    contexts = _selection_contexts(merged_records)
    selected_changed_count = sum(1 for context in contexts.values() if context["selected_candidate_changed"])
    selected_unchanged_count = len(contexts) - selected_changed_count
    selected_candidate_changed_rate = _rate(selected_changed_count, len(contexts))
    keep_records = [record for record in merged_records if record["recommendation"] == "keep"]
    keep_selected_count = sum(1 for record in keep_records if record["selected"])
    keep_non_selected_count = len(keep_records) - keep_selected_count
    blocked_records = [record for record in merged_records if _is_blocked_candidate(record)]
    blocked_selected_count = sum(1 for record in blocked_records if record["selected"])
    same_as_baseline_count = sum(1 for record in merged_records if _has_reason(record, "same_as_baseline"))
    goal_blocked_count = sum(1 for record in merged_records if _has_reason(record, "goal_blocked"))
    path_cost_tradeoff_count = sum(
        1 for record in merged_records if record["path_cost_tradeoff"] or _has_reason(record, "path_cost_tradeoff")
    )
    recommendation_counts = Counter(record["recommendation"] for record in merged_records)
    action_counts = Counter(record["application_action"] for record in merged_records)
    reason_counts = Counter(
        reason
        for record in merged_records
        for reason in record["reason_codes"] + record["application_reason_codes"]
    )
    for recommendation in ("keep", "downweight", "reject", "needs_more_evidence"):
        recommendation_counts.setdefault(recommendation, 0)
    for action in (
        "keep_quality_evidence",
        "downweight_conservative_application",
        "exclude_blocked_candidate_evidence",
        "downweight_needs_more_evidence",
    ):
        action_counts.setdefault(action, 0)
    supports_improvement = selected_candidate_changed_rate > 0.0 and any(
        record["selected_candidate_changed"] for record in merged_records
    )
    explanation_codes = _selection_change_explanation_codes(
        selected_candidate_changed_rate=selected_candidate_changed_rate,
        keep_non_selected_count=keep_non_selected_count,
        blocked_candidate_rate=_rate(len(blocked_records), record_count),
        same_as_baseline_count=same_as_baseline_count,
        goal_blocked_count=goal_blocked_count,
        path_cost_tradeoff_count=path_cost_tradeoff_count,
        config=config,
    )
    recommended_next_adjustment = _recommended_next_adjustment(
        supports_improvement=supports_improvement,
        keep_selected_count=keep_selected_count,
        keep_non_selected_count=keep_non_selected_count,
        blocked_candidate_rate=_rate(len(blocked_records), record_count),
        same_as_baseline_count=same_as_baseline_count,
        goal_blocked_count=goal_blocked_count,
        config=config,
    )
    return {
        "record_count": record_count,
        "selection_context_count": len(contexts),
        "selected_candidate_changed_count": selected_changed_count,
        "selected_candidate_changed_rate": selected_candidate_changed_rate,
        "selected_candidate_unchanged_count": selected_unchanged_count,
        "selected_candidate_unchanged_rate": _rate(selected_unchanged_count, len(contexts)),
        "candidate_ranking_unchanged_count": selected_unchanged_count,
        "candidate_ranking_unchanged_rate": _rate(selected_unchanged_count, len(contexts)),
        "keep_candidate_count": len(keep_records),
        "keep_selected_candidate_count": keep_selected_count,
        "keep_selected_candidate_rate": _rate(keep_selected_count, len(keep_records)),
        "keep_non_selected_candidate_count": keep_non_selected_count,
        "keep_non_selected_candidate_rate": _rate(keep_non_selected_count, len(keep_records)),
        "blocked_candidate_count": len(blocked_records),
        "blocked_candidate_rate": _rate(len(blocked_records), record_count),
        "blocked_selected_candidate_count": blocked_selected_count,
        "blocked_selected_candidate_rate": _rate(blocked_selected_count, len(blocked_records)),
        "same_as_baseline_count": same_as_baseline_count,
        "goal_blocked_count": goal_blocked_count,
        "path_cost_tradeoff_count": path_cost_tradeoff_count,
        "recommendation_counts": dict(sorted(recommendation_counts.items())),
        "application_action_counts": dict(sorted(action_counts.items())),
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "readiness_gate_input": {
            "status": readiness.get("status"),
            "readiness_status": readiness.get("readiness_status"),
            "readiness_reason_codes": _string_list(readiness.get("readiness_reason_codes", [])),
            "selected_candidate_changed_rate": readiness.get("selected_candidate_changed_rate"),
        },
        "supports_policy_target_selection_improvement_claim": supports_improvement,
        "policy_target_selection_improvement_claim": "supported" if supports_improvement else "not_supported",
        "improvement_claim_reason_codes": (
            ["policy_target_selection_improvement_evidence_present"]
            if supports_improvement
            else ["policy_target_selection_not_improved"]
        ),
        "selection_change_explanation_codes": explanation_codes,
        "recommended_next_adjustment": recommended_next_adjustment,
        "training_readiness_gate_rerun_recommendation": (
            "rerun_training_readiness_gate_after_policy_target_selection_evidence_changes"
            if supports_improvement
            else "not_until_policy_target_selection_evidence_changes"
        ),
    }


def _selection_contexts(records: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    contexts: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        context = contexts.setdefault(
            record["context_key"],
            {
                "selected_candidate_changed": False,
                "selected_cell_unchanged": True,
            },
        )
        if record["selected_candidate_changed"]:
            context["selected_candidate_changed"] = True
        if record["astar_selected_cell"] != record["channel_aware_selected_cell"]:
            context["selected_cell_unchanged"] = False
    return contexts


def _selection_change_explanation_codes(
    *,
    selected_candidate_changed_rate: float,
    keep_non_selected_count: int,
    blocked_candidate_rate: float,
    same_as_baseline_count: int,
    goal_blocked_count: int,
    path_cost_tradeoff_count: int,
    config: dict[str, Any],
) -> list[str]:
    codes: list[str] = []
    if selected_candidate_changed_rate == 0.0:
        codes.append("selected_candidate_unchanged")
    if keep_non_selected_count > 0:
        codes.append("keep_evidence_not_selected_by_policy_target")
    if blocked_candidate_rate >= config["thresholds"]["blocked_candidate_high_rate"]:
        codes.append("reject_or_blocked_candidate_rate_high")
    if same_as_baseline_count > 0:
        codes.append("same_as_baseline_present")
    if goal_blocked_count > 0:
        codes.append("goal_blocked_present")
    if path_cost_tradeoff_count > 0:
        codes.append("path_cost_tradeoff_recorded_not_failure")
    return codes


def _recommended_next_adjustment(
    *,
    supports_improvement: bool,
    keep_selected_count: int,
    keep_non_selected_count: int,
    blocked_candidate_rate: float,
    same_as_baseline_count: int,
    goal_blocked_count: int,
    config: dict[str, Any],
) -> str:
    if supports_improvement:
        return "rerun_training_readiness_gate_with_policy_target_selection_improvement_evidence"
    if keep_non_selected_count > 0:
        return "inspect_candidate_ranking_weights_to_promote_keep_evidence_before_policy_smoke"
    if keep_selected_count > 0:
        return "increase_channel_aware_selection_contrast_against_baseline_before_policy_smoke"
    if blocked_candidate_rate >= config["thresholds"]["blocked_candidate_high_rate"] and goal_blocked_count > 0:
        return "reduce_goal_blocked_channel_aware_candidates_before_policy_smoke"
    if same_as_baseline_count > 0:
        return "add_channel_aware_scenarios_that_differ_from_baseline_before_policy_smoke"
    return "collect_more_channel_aware_policy_target_selection_evidence"


def _is_blocked_candidate(record: dict[str, Any]) -> bool:
    return (
        record["recommendation"] == "reject"
        or record.get("application_action") == "exclude_blocked_candidate_evidence"
        or _has_reason(record, "goal_blocked")
    )


def _has_reason(record: dict[str, Any], reason: str) -> bool:
    return record.get("blocker_reason") == reason or reason in record.get("reason_codes", [])


def _record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("pair_key"),
        record.get("scenario_id"),
        record.get("action_index"),
        tuple(_list_value(record.get("cell"))),
    )


def _context_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (record.get("pair_key"), record.get("scenario_id"))


def _application_action(recommendation: str) -> str:
    return {
        "keep": "keep_quality_evidence",
        "downweight": "downweight_conservative_application",
        "reject": "exclude_blocked_candidate_evidence",
        "needs_more_evidence": "downweight_needs_more_evidence",
    }.get(recommendation, "downweight_needs_more_evidence")


def _application_weight(action: str) -> float:
    if action == "keep_quality_evidence":
        return 1.0
    if action == "exclude_blocked_candidate_evidence":
        return 0.0
    return 0.5


def _output_file(batch_root: Path, config: dict[str, Any]) -> Path:
    return batch_root / config["output_files"]["policy_target_selection_evidence_summary"]


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": config.get("schema_version"),
        "validation": dict(config.get("validation", {})) if isinstance(config.get("validation"), dict) else {},
        "thresholds": dict(config.get("thresholds", {})),
        "output_files": dict(config.get("output_files", {})),
    }


def _public_git(payload: dict[str, Any]) -> dict[str, Any]:
    git = payload.get("git_provenance") if isinstance(payload.get("git_provenance"), dict) else {}
    return dict(git)


def _require_current_git_match(config: dict[str, Any]) -> bool:
    validation = config.get("validation")
    if not isinstance(validation, dict):
        return True
    return bool(validation.get("require_current_git_match", True))


def _fail_on_input_failure(config: dict[str, Any]) -> bool:
    validation = config.get("validation")
    if not isinstance(validation, dict):
        return True
    return bool(validation.get("fail_on_input_failure", True))


def _git_snapshot(repo_root: Path) -> dict[str, Any]:
    def git(path: Path, *args: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            cwd=repo_root,
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
            "sha": git(repo_root, "rev-parse", "HEAD") or "unknown",
            "branch": git(repo_root, "branch", "--show-current"),
        },
        "submodules": {
            name: {
                "path": name,
                "sha": git(repo_root / name, "rev-parse", "HEAD") or "unknown",
                "branch": git(repo_root / name, "branch", "--show-current"),
            }
            for name in SUBMODULES
        },
    }


def _git_snapshots_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not left or not right:
        return False
    left_parent = left.get("parent") if isinstance(left.get("parent"), dict) else {}
    right_parent = right.get("parent") if isinstance(right.get("parent"), dict) else {}
    if left_parent.get("sha") != right_parent.get("sha"):
        return False
    left_modules = left.get("submodules") if isinstance(left.get("submodules"), dict) else {}
    right_modules = right.get("submodules") if isinstance(right.get("submodules"), dict) else {}
    for name in SUBMODULES:
        left_module = left_modules.get(name) if isinstance(left_modules.get(name), dict) else {}
        right_module = right_modules.get(name) if isinstance(right_modules.get(name), dict) else {}
        if left_module.get("sha") != right_module.get("sha"):
            return False
    return True


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_reason(reason_codes: list[str], code: str) -> None:
    if code not in reason_codes:
        reason_codes.append(code)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return []


def _list_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    return []


def _float_value(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a number") from exc


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
