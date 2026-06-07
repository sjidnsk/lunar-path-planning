from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _shared_git_snapshot
from git_provenance import git_snapshots_match as _shared_git_snapshots_match
from git_provenance import inspect_source_git_provenance as _inspect_source_git_provenance
from git_provenance import public_git as _shared_public_git


CONFIG_SCHEMA_VERSION = "channel-aware-training-readiness-gate-config/v1"
SUMMARY_SCHEMA_VERSION = "channel-aware-training-readiness-summary/v1"
APPLICATION_SCHEMA_VERSION = "policy-robustness-application-summary/v1"
CHANNEL_APPLICATION_SCHEMA_VERSION = "channel-aware-application-smoke/v1"
ROBUSTNESS_SCHEMA_VERSION = "policy-decision-robustness-summary/v1"
POLICY_TARGET_EVIDENCE_SCHEMA_VERSION = "channel-aware-policy-target-selection-evidence-summary/v1"
SELECTION_CONTRAST_CALIBRATION_SCHEMA_VERSION = "channel-aware-selection-contrast-calibration-summary/v1"
SUBMODULES = ("dev-platform-constraints", "model-explorer", "path-planner")


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply Channel-Aware Training Readiness Gate v1 to application smoke evidence."
    )
    parser.add_argument("--batch-root", required=True, help="Batch root containing application smoke JSON outputs.")
    parser.add_argument(
        "--application-summary",
        help="policy-robustness-application-summary/v1 JSON. Defaults to <batch-root>/policy-robustness-application-summary.json.",
    )
    parser.add_argument(
        "--robustness-summary",
        help="policy-decision-robustness-summary/v1 JSON. Defaults to <batch-root>/policy-decision-robustness-summary.json.",
    )
    parser.add_argument(
        "--policy-target-evidence-summary",
        help="channel-aware-policy-target-selection-evidence-summary/v1 JSON. Defaults to <batch-root>/channel-aware-policy-target-selection-evidence-summary.json.",
    )
    parser.add_argument(
        "--selection-contrast-calibration-summary",
        help="channel-aware-selection-contrast-calibration-summary/v1 JSON. Defaults to <batch-root>/channel-aware-selection-contrast-calibration-summary.json.",
    )
    parser.add_argument("--config", required=True, help="Channel-aware training readiness gate config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned output paths.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without writing outputs.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    batch_root = _resolve_path(args.batch_root, repo_root)
    application_summary_path = (
        _resolve_path(args.application_summary, repo_root)
        if args.application_summary
        else batch_root / "policy-robustness-application-summary.json"
    )
    robustness_summary_path = (
        _resolve_path(args.robustness_summary, repo_root)
        if args.robustness_summary
        else batch_root / "policy-decision-robustness-summary.json"
    )
    policy_target_evidence_path = (
        _resolve_path(args.policy_target_evidence_summary, repo_root)
        if args.policy_target_evidence_summary
        else batch_root / "channel-aware-policy-target-selection-evidence-summary.json"
    )
    selection_contrast_calibration_path = (
        _resolve_path(args.selection_contrast_calibration_summary, repo_root)
        if args.selection_contrast_calibration_summary
        else batch_root / "channel-aware-selection-contrast-calibration-summary.json"
    )
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    summary = analyze_training_readiness(
        batch_root=batch_root,
        application_summary_path=application_summary_path,
        robustness_summary_path=robustness_summary_path,
        policy_target_evidence_path=policy_target_evidence_path,
        selection_contrast_calibration_path=selection_contrast_calibration_path,
        config=config,
        repo_root=repo_root,
    )
    output_file = _output_file(batch_root, config)
    validation_message = {
        "status": "config validated" if summary["status"] == "passed" else "validation failed",
        "batch_root": _display_path(batch_root, repo_root),
        "application_summary": _display_path(application_summary_path, repo_root),
        "robustness_summary": _display_path(robustness_summary_path, repo_root),
        "policy_target_evidence_summary": _display_path(policy_target_evidence_path, repo_root),
        "selection_contrast_calibration_summary": _display_path(
            selection_contrast_calibration_path,
            repo_root,
        ),
        "config": _display_path(config_path, repo_root),
        "reason_codes": summary["reason_codes"],
        "readiness_status": summary["readiness_status"],
        "readiness_reason_codes": summary["readiness_reason_codes"],
        "calibrated_readiness_status": summary["calibrated_readiness_status"],
        "calibrated_readiness_reason_codes": summary["calibrated_readiness_reason_codes"],
        "training_readiness_summary": _display_path(output_file, repo_root),
    }
    print(json.dumps(validation_message, ensure_ascii=False))

    if args.validate_only or args.dry_run:
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "dry-run",
                        "would_write": {
                            "training_readiness_summary": _display_path(output_file, repo_root),
                        },
                        "readiness_status": summary["readiness_status"],
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
                "readiness_status": summary["readiness_status"],
                "channel_aware_training_readiness_summary": _display_path(output_file, repo_root),
                "failure_reason_code_counts": summary["failure_reason_code_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def analyze_training_readiness(
    *,
    batch_root: Path,
    application_summary_path: Path,
    robustness_summary_path: Path,
    policy_target_evidence_path: Path,
    selection_contrast_calibration_path: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    global_reason_codes: list[str] = []
    source_summaries: dict[str, Any] = {}
    application = _load_application_summary(
        application_summary_path,
        repo_root=repo_root,
        reason_codes=global_reason_codes,
        source_summaries=source_summaries,
    )
    robustness = _load_source_summary(
        robustness_summary_path,
        label="policy_decision_robustness_summary",
        expected_schema=ROBUSTNESS_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=global_reason_codes,
        source_summaries=source_summaries,
    )
    policy_target_evidence = _load_source_summary(
        policy_target_evidence_path,
        label="channel_aware_policy_target_selection_evidence_summary",
        expected_schema=POLICY_TARGET_EVIDENCE_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=global_reason_codes,
        source_summaries=source_summaries,
    )
    selection_contrast_calibration = _load_source_summary(
        selection_contrast_calibration_path,
        label="channel_aware_selection_contrast_calibration_summary",
        expected_schema=SELECTION_CONTRAST_CALIBRATION_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=global_reason_codes,
        source_summaries=source_summaries,
    )
    current_git = _git_snapshot(repo_root)
    app_git = application.get("git_provenance") if isinstance(application.get("git_provenance"), dict) else {}
    app_current_git = app_git.get("current") if isinstance(app_git.get("current"), dict) else {}
    if _require_current_git_match(config) and app_current_git and not _git_snapshots_match(app_current_git, current_git):
        _append_reason(global_reason_codes, "current_git_provenance_mismatch")
    if app_git and app_git.get("current_matches_robustness") is False:
        _append_reason(global_reason_codes, "git_provenance_mismatch")
    robustness_git = app_git.get("robustness") if isinstance(app_git.get("robustness"), dict) else {}
    if robustness_git and robustness_git.get("current_matches_batch") is False:
        _append_reason(global_reason_codes, "git_provenance_mismatch")
    if robustness_git and robustness_git.get("runs_match_batch") is False:
        _append_reason(global_reason_codes, "git_provenance_mismatch")
    _inspect_summary_git(
        robustness,
        label="policy_decision_robustness_summary",
        current_git=current_git,
        config=config,
        reason_codes=global_reason_codes,
    )
    _inspect_summary_git(
        policy_target_evidence,
        label="channel_aware_policy_target_selection_evidence_summary",
        current_git=current_git,
        config=config,
        reason_codes=global_reason_codes,
    )
    _inspect_summary_git(
        selection_contrast_calibration,
        label="channel_aware_selection_contrast_calibration_summary",
        current_git=current_git,
        config=config,
        reason_codes=global_reason_codes,
    )

    if _fail_on_input_failure(config) and application.get("status") == "failed":
        _append_reason(global_reason_codes, "policy_robustness_application_summary_failed")
    if _fail_on_input_failure(config) and robustness.get("status") == "failed":
        _append_reason(global_reason_codes, "policy_decision_robustness_summary_failed")
    if _fail_on_input_failure(config) and policy_target_evidence.get("status") == "failed":
        _append_reason(global_reason_codes, "channel_aware_policy_target_selection_evidence_summary_failed")
    if _fail_on_input_failure(config) and selection_contrast_calibration.get("status") == "failed":
        _append_reason(global_reason_codes, "channel_aware_selection_contrast_calibration_summary_failed")

    channel = _channel_application(application, reason_codes=global_reason_codes)
    records = [
        _normalize_record(record)
        for record in channel.get("records", [])
        if isinstance(record, dict)
    ]
    evidence = _summarize_evidence(records)
    policy_target = _policy_target_evidence(policy_target_evidence)
    calibration = _calibration_evidence(selection_contrast_calibration)
    readiness = _readiness_status(
        evidence=evidence,
        policy_target=policy_target,
        calibration=calibration,
        validation_reason_codes=global_reason_codes,
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
        "application_summary_path": _display_path(application_summary_path, repo_root),
        "robustness_summary_path": _display_path(robustness_summary_path, repo_root),
        "policy_target_evidence_summary_path": _display_path(policy_target_evidence_path, repo_root),
        "calibration_summary_path": _display_path(selection_contrast_calibration_path, repo_root),
        "application_scope": "channel_aware_training_readiness_audit_only",
        "quality_signal_use": "calibration_aware_pre_training_readiness_gate_only",
        "readiness_status": readiness["status"],
        "readiness_reason_codes": readiness["reason_codes"],
        "calibrated_readiness_status": readiness["status"],
        "calibrated_readiness_reason_codes": readiness["reason_codes"],
        "conservative_next_step": readiness["conservative_next_step"],
        "recommended_next_action": readiness["recommended_next_action"],
        "training_evidence_policy": {
            "keep_quality_evidence": "positive_pre_training_evidence",
            "downweight_conservative_application": "downweighted_evidence_not_strong_positive",
            "downweight_needs_more_evidence": "downweighted_until_more_evidence",
            "exclude_blocked_candidate_evidence": "excluded_from_training_positive_evidence",
            "path_cost_tradeoff": "tradeoff_reason_not_failure_by_itself",
            "selected_candidate_changed_rate_zero": "does_not_support_policy_target_selection_improvement_claim",
        },
        "config": _public_config(config),
        "source_summaries": source_summaries,
        "git_provenance": {
            "application": dict(app_git),
            "robustness": _public_git(robustness),
            "policy_target_evidence": _public_git(policy_target_evidence),
            "selection_contrast_calibration": _public_git(selection_contrast_calibration),
            "current": current_git,
            "current_matches_application": _git_snapshots_match(app_current_git, current_git)
            if app_current_git
            else None,
        },
        "source_selected_candidate_changed_rate": policy_target["selected_candidate_changed_rate"],
        "source_supports_policy_target_selection_improvement_claim": policy_target[
            "supports_policy_target_selection_improvement_claim"
        ],
        "policy_target_selection_improvement_claimed": False,
        "calibration_selected_candidate_changed_count": calibration["selected_candidate_changed_count"],
        "calibration_selected_candidate_changed_rate": calibration["selected_candidate_changed_rate"],
        "calibration_changed_scenario_ids": calibration["changed_scenario_ids"],
        "calibration_safety_regression_count": calibration["safety_regression_count"],
        "record_count": evidence["record_count"],
        "recommendation_counts": evidence["recommendation_counts"],
        "application_action_counts": evidence["application_action_counts"],
        "sample_weight_distribution": evidence["sample_weight_distribution"],
        "sample_weight_stats": evidence["sample_weight_stats"],
        "reason_code_counts": evidence["reason_code_counts"],
        "excluded_candidate_count": evidence["excluded_candidate_count"],
        "excluded_candidate_rate": evidence["excluded_candidate_rate"],
        "positive_evidence_count": evidence["positive_evidence_count"],
        "positive_evidence_rate": evidence["positive_evidence_rate"],
        "downweighted_evidence_count": evidence["downweighted_evidence_count"],
        "downweighted_evidence_rate": evidence["downweighted_evidence_rate"],
        "selected_candidate_changed_count": evidence["selected_candidate_changed_count"],
        "selected_candidate_changed_rate": evidence["selected_candidate_changed_rate"],
        "path_cost_tradeoff_count": evidence["path_cost_tradeoff_count"],
        "path_cost_tradeoff_rate": evidence["path_cost_tradeoff_rate"],
        "route_replacement_default_changed": bool(channel.get("route_replacement_default_changed", False)),
        "no_ppo_training": True,
        "no_large_scale_training": True,
        "no_real_world_performance_claim": True,
        "not_real_world_performance_claim": True,
        "does_not_modify_ppo": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "does_not_modify_default_astar": True,
        "does_not_modify_path_planner_route_contract": True,
        "does_not_modify_model_explorer_contract": True,
        "does_not_modify_path_planner_sidecar_contract": True,
        "no_ackermann_feasible_trajectory_claim": True,
        "no_gcs_control_point_candidate_as_default_execution_trajectory": True,
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
    if not isinstance(output_files, dict) or not _string_value(output_files.get("training_readiness_summary")):
        raise ConfigError("output_files.training_readiness_summary must be a non-empty string")
    thresholds = payload.get("readiness_thresholds")
    if not isinstance(thresholds, dict):
        raise ConfigError("readiness_thresholds must be an object")
    normalized_thresholds = {
        "min_positive_evidence_count": _int_value(
            thresholds.get("min_positive_evidence_count", 1),
            "readiness_thresholds.min_positive_evidence_count",
        ),
        "min_positive_evidence_rate": _float_value(
            thresholds.get("min_positive_evidence_rate", 0.25),
            "readiness_thresholds.min_positive_evidence_rate",
        ),
        "max_excluded_candidate_rate_for_policy_smoke": _float_value(
            thresholds.get("max_excluded_candidate_rate_for_policy_smoke", 0.75),
            "readiness_thresholds.max_excluded_candidate_rate_for_policy_smoke",
        ),
        "block_excluded_candidate_rate": _float_value(
            thresholds.get("block_excluded_candidate_rate", 0.9),
            "readiness_thresholds.block_excluded_candidate_rate",
        ),
        "require_selected_candidate_change_for_ready": bool(
            thresholds.get("require_selected_candidate_change_for_ready", True)
        ),
    }
    if normalized_thresholds["min_positive_evidence_count"] < 0:
        raise ConfigError("readiness_thresholds.min_positive_evidence_count must be >= 0")
    for key in (
        "min_positive_evidence_rate",
        "max_excluded_candidate_rate_for_policy_smoke",
        "block_excluded_candidate_rate",
    ):
        if normalized_thresholds[key] < 0.0 or normalized_thresholds[key] > 1.0:
            raise ConfigError(f"readiness_thresholds.{key} must be between 0 and 1")
    config = dict(payload)
    config["readiness_thresholds"] = normalized_thresholds
    return config


def _load_application_summary(
    path: Path,
    *,
    repo_root: Path,
    reason_codes: list[str],
    source_summaries: dict[str, Any],
) -> dict[str, Any]:
    label = "policy_robustness_application_summary"
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
    schema_version = payload.get("schema_version")
    record.update(
        {
            "schema_version": schema_version,
            "status": payload.get("status"),
            "reason_codes": _string_list(payload.get("reason_codes", [])),
        }
    )
    if schema_version != APPLICATION_SCHEMA_VERSION:
        _append_reason(reason_codes, f"{label}_schema_mismatch")
    source_summaries[label] = record
    return payload


def _load_source_summary(
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


def _inspect_summary_git(
    payload: dict[str, Any],
    *,
    label: str,
    current_git: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    _inspect_source_git_provenance(
        payload,
        label=label,
        current_git=current_git,
        require_current_git_match=_require_current_git_match(config),
        reason_codes=reason_codes,
        submodules=SUBMODULES,
    )


def _channel_application(application: dict[str, Any], *, reason_codes: list[str]) -> dict[str, Any]:
    channel = application.get("channel_aware_application")
    if not isinstance(channel, dict):
        _append_reason(reason_codes, "channel_aware_application_missing")
        return {}
    if channel.get("schema_version") != CHANNEL_APPLICATION_SCHEMA_VERSION:
        _append_reason(reason_codes, "channel_aware_application_schema_mismatch")
    records = channel.get("records")
    if not isinstance(records, list):
        _append_reason(reason_codes, "channel_aware_application_records_invalid")
        return dict(channel, records=[])
    return channel


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    action = str(record.get("application_action", "downweight_needs_more_evidence"))
    recommendation = str(record.get("recommendation", "needs_more_evidence"))
    reason_codes = _string_list(record.get("reason_codes", []))
    application_reason_codes = _string_list(record.get("application_reason_codes", []))
    if bool(record.get("path_cost_tradeoff", False)) and "path_cost_tradeoff" not in reason_codes:
        reason_codes.append("path_cost_tradeoff")
    return {
        "recommendation": recommendation,
        "application_action": action,
        "application_sample_weight": _safe_float(record.get("application_sample_weight"), 0.5),
        "selected_candidate_changed": bool(record.get("selected_candidate_changed", False)),
        "path_cost_tradeoff": bool(record.get("path_cost_tradeoff", False)),
        "reason_codes": reason_codes,
        "application_reason_codes": application_reason_codes,
    }


def _summarize_evidence(records: list[dict[str, Any]]) -> dict[str, Any]:
    record_count = len(records)
    recommendation_counts = Counter(record["recommendation"] for record in records)
    action_counts = Counter(record["application_action"] for record in records)
    for key in ("keep", "downweight", "reject", "needs_more_evidence"):
        recommendation_counts.setdefault(key, 0)
    for key in (
        "keep_quality_evidence",
        "downweight_conservative_application",
        "exclude_blocked_candidate_evidence",
        "downweight_needs_more_evidence",
    ):
        action_counts.setdefault(key, 0)
    reason_counts = Counter(
        reason
        for record in records
        for reason in record["reason_codes"] + record["application_reason_codes"]
    )
    weight_distribution = Counter(_weight_key(record["application_sample_weight"]) for record in records)
    weights = [record["application_sample_weight"] for record in records]
    excluded_count = action_counts["exclude_blocked_candidate_evidence"]
    positive_count = action_counts["keep_quality_evidence"]
    downweighted_count = (
        action_counts["downweight_conservative_application"]
        + action_counts["downweight_needs_more_evidence"]
    )
    selected_changed_count = sum(1 for record in records if record["selected_candidate_changed"])
    path_cost_tradeoff_count = sum(
        1
        for record in records
        if record["path_cost_tradeoff"] or "path_cost_tradeoff" in record["reason_codes"]
    )
    return {
        "record_count": record_count,
        "recommendation_counts": dict(sorted(recommendation_counts.items())),
        "application_action_counts": dict(sorted(action_counts.items())),
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "sample_weight_distribution": dict(sorted(weight_distribution.items(), key=lambda item: float(item[0]))),
        "sample_weight_stats": _sample_weight_stats(weights),
        "excluded_candidate_count": excluded_count,
        "excluded_candidate_rate": _rate(excluded_count, record_count),
        "positive_evidence_count": positive_count,
        "positive_evidence_rate": _rate(positive_count, record_count),
        "downweighted_evidence_count": downweighted_count,
        "downweighted_evidence_rate": _rate(downweighted_count, record_count),
        "selected_candidate_changed_count": selected_changed_count,
        "selected_candidate_changed_rate": _rate(selected_changed_count, record_count),
        "path_cost_tradeoff_count": path_cost_tradeoff_count,
        "path_cost_tradeoff_rate": _rate(path_cost_tradeoff_count, record_count),
    }


def _policy_target_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_candidate_changed_rate": _safe_float(
            payload.get("selected_candidate_changed_rate"),
            0.0,
        ),
        "supports_policy_target_selection_improvement_claim": bool(
            payload.get("supports_policy_target_selection_improvement_claim", False)
        ),
    }


def _calibration_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_selected_candidate_changed_rate": _safe_float(
            payload.get("source_selected_candidate_changed_rate"),
            0.0,
        ),
        "selected_candidate_changed_count": _safe_int(
            payload.get("selected_candidate_changed_count"),
            0,
        ),
        "selected_candidate_changed_rate": _safe_float(
            payload.get("selected_candidate_changed_rate"),
            0.0,
        ),
        "changed_scenario_ids": _string_list(payload.get("changed_scenario_ids", [])),
        "safety_regression_count": _safe_int(payload.get("safety_regression_count"), 0),
    }


def _readiness_status(
    *,
    evidence: dict[str, Any],
    policy_target: dict[str, Any],
    calibration: dict[str, Any],
    validation_reason_codes: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    thresholds = config["readiness_thresholds"]
    reason_codes: list[str] = []
    record_count = evidence["record_count"]
    positive_count = evidence["positive_evidence_count"]
    positive_rate = evidence["positive_evidence_rate"]
    excluded_rate = evidence["excluded_candidate_rate"]
    selected_changed_rate = evidence["selected_candidate_changed_rate"]
    if validation_reason_codes:
        for reason in validation_reason_codes:
            _append_reason(reason_codes, reason)
        return {
            "status": "blocked",
            "reason_codes": reason_codes,
            "conservative_next_step": "fix_validation_failures_before_policy_smoke",
            "recommended_next_action": "fix_validation_failures_before_policy_smoke",
        }
    if record_count == 0:
        _append_reason(reason_codes, "channel_aware_application_records_empty")
    if positive_count == 0:
        _append_reason(reason_codes, "positive_channel_aware_evidence_missing")
    if excluded_rate >= thresholds["block_excluded_candidate_rate"] and record_count > 0:
        _append_reason(reason_codes, "blocked_candidate_rate_blocks_training_readiness")
    if reason_codes:
        return {
            "status": "blocked",
            "reason_codes": reason_codes,
            "conservative_next_step": "regenerate_channel_aware_application_evidence_before_policy_smoke",
            "recommended_next_action": "regenerate_channel_aware_application_evidence_before_policy_smoke",
        }

    if calibration["safety_regression_count"] > 0:
        _append_reason(reason_codes, "calibration_safety_regression_blocks_policy_smoke")
        return {
            "status": "blocked",
            "reason_codes": reason_codes,
            "conservative_next_step": "resolve_calibration_safety_regressions_before_policy_smoke",
            "recommended_next_action": "resolve_calibration_safety_regressions_before_policy_smoke",
        }

    if positive_count < thresholds["min_positive_evidence_count"] or positive_rate < thresholds["min_positive_evidence_rate"]:
        _append_reason(reason_codes, "positive_channel_aware_evidence_below_threshold")
    if excluded_rate > thresholds["max_excluded_candidate_rate_for_policy_smoke"]:
        _append_reason(reason_codes, "blocked_candidate_rate_high")
    calibration_changed_rate = calibration["selected_candidate_changed_rate"]
    if thresholds["require_selected_candidate_change_for_ready"] and calibration_changed_rate == 0.0:
        _append_reason(reason_codes, "calibrated_target_contrast_missing")

    if reason_codes:
        next_step = "collect_calibrated_policy_smoke_evidence_before_training"
        return {
            "status": "needs_policy_smoke_before_training",
            "reason_codes": reason_codes,
            "conservative_next_step": next_step,
            "recommended_next_action": next_step,
        }
    calibrated_reason_codes = ["calibrated_target_contrast_available"]
    if policy_target["selected_candidate_changed_rate"] == 0.0:
        _append_reason(calibrated_reason_codes, "source_policy_target_selection_not_improved")
    return {
        "status": "ready_for_calibrated_policy_application_smoke",
        "reason_codes": calibrated_reason_codes,
        "conservative_next_step": "run_calibrated_policy_application_smoke_before_training",
        "recommended_next_action": "run_calibrated_policy_application_smoke_before_training",
    }


def _sample_weight_stats(weights: list[float]) -> dict[str, Any]:
    if not weights:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(weights),
        "min": min(weights),
        "max": max(weights),
        "mean": sum(weights) / len(weights),
    }


def _output_file(batch_root: Path, config: dict[str, Any]) -> Path:
    return batch_root / config["output_files"]["training_readiness_summary"]


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": config.get("schema_version"),
        "validation": dict(config.get("validation", {})) if isinstance(config.get("validation"), dict) else {},
        "readiness_thresholds": dict(config.get("readiness_thresholds", {})),
        "output_files": dict(config.get("output_files", {})),
    }


def _public_git(payload: dict[str, Any]) -> dict[str, Any]:
    return _shared_public_git(payload)


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
    return _shared_git_snapshot(repo_root, submodules=SUBMODULES)


def _git_snapshots_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _shared_git_snapshots_match(left, right, submodules=SUBMODULES)


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


def _string_value(value: Any) -> str:
    return value if isinstance(value, str) and value else ""


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return []


def _int_value(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be an integer") from exc


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


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _weight_key(value: float) -> str:
    if value == int(value):
        return f"{value:.1f}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
