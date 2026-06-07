from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _shared_git_snapshot
from git_provenance import git_snapshots_match as _shared_git_snapshots_match
from git_provenance import inspect_source_git_provenance as _inspect_source_git_provenance
from git_provenance import public_git as _shared_public_git


CONFIG_SCHEMA_VERSION = "goal-blocked-training-contract-refinement-config/v1"
SUMMARY_SCHEMA_VERSION = "goal-blocked-training-contract-refinement-summary/v1"
REVIEW_SCHEMA_VERSION = "policy-training-readiness-review-summary/v1"
SMOKE_SCHEMA_VERSION = "calibrated-policy-application-smoke-summary/v1"
APPLICATION_SCHEMA_VERSION = "policy-robustness-application-summary/v1"
CALIBRATION_SCHEMA_VERSION = "channel-aware-selection-contrast-calibration-summary/v1"
CHANNEL_APPLICATION_SCHEMA_VERSION = "channel-aware-application-smoke/v1"
SUBMODULES = ("dev-platform-constraints", "model-explorer", "path-planner")
FALLBACK_COUNT_FIELDS = ("open_grid_fallback_used_count", "open_grid_fallback_count", "fallback_used_count")
CONTRACT_GUARD_FIELDS = (
    "does_not_modify_default_astar",
    "does_not_modify_ppo",
    "does_not_modify_network",
    "does_not_modify_action_space",
    "does_not_modify_model_explorer_contract",
    "does_not_modify_path_planner_route_contract",
    "does_not_modify_path_planner_sidecar_contract",
    "no_ackermann_feasible_trajectory_claim",
)
CONTRACT_DECISIONS = (
    "excluded_from_positive_training",
    "eligible_negative_evidence",
    "needs_regeneration",
    "blocked_by_contract",
)
PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES = frozenset(
    {
        "platform_inflated_goal_blocked",
        "original_goal_blocked",
        "out_of_bounds",
        "unknown_contract_mismatch",
    }
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify goal-blocked and excluded candidates before policy training."
    )
    parser.add_argument("--batch-root", required=True, help="Batch root containing current evidence summaries.")
    parser.add_argument(
        "--policy-training-readiness-review-summary",
        help="policy-training-readiness-review-summary/v1 JSON. Defaults to <batch-root>/policy-training-readiness-review-summary.json.",
    )
    parser.add_argument(
        "--calibrated-policy-application-smoke-summary",
        help="calibrated-policy-application-smoke-summary/v1 JSON. Defaults to <batch-root>/calibrated-policy-application-smoke-summary.json.",
    )
    parser.add_argument(
        "--policy-robustness-application-summary",
        help="policy-robustness-application-summary/v1 JSON. Defaults to <batch-root>/policy-robustness-application-summary.json.",
    )
    parser.add_argument(
        "--selection-contrast-calibration-summary",
        help="channel-aware-selection-contrast-calibration-summary/v1 JSON. Defaults to <batch-root>/channel-aware-selection-contrast-calibration-summary.json.",
    )
    parser.add_argument("--config", required=True, help="Goal-blocked contract refinement config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned output paths.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without writing outputs.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    batch_root = _resolve_path(args.batch_root, repo_root)
    review_path = (
        _resolve_path(args.policy_training_readiness_review_summary, repo_root)
        if args.policy_training_readiness_review_summary
        else batch_root / "policy-training-readiness-review-summary.json"
    )
    smoke_path = (
        _resolve_path(args.calibrated_policy_application_smoke_summary, repo_root)
        if args.calibrated_policy_application_smoke_summary
        else batch_root / "calibrated-policy-application-smoke-summary.json"
    )
    application_path = (
        _resolve_path(args.policy_robustness_application_summary, repo_root)
        if args.policy_robustness_application_summary
        else batch_root / "policy-robustness-application-summary.json"
    )
    calibration_path = (
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

    summary = analyze_goal_blocked_contract_refinement(
        batch_root=batch_root,
        review_path=review_path,
        smoke_path=smoke_path,
        application_path=application_path,
        calibration_path=calibration_path,
        config=config,
        repo_root=repo_root,
    )
    output_file = _output_file(batch_root, config)
    validation_message = {
        "status": "config validated" if summary["status"] == "passed" else "validation failed",
        "batch_root": _display_path(batch_root, repo_root),
        "policy_training_readiness_review_summary": _display_path(review_path, repo_root),
        "calibrated_policy_application_smoke_summary": _display_path(smoke_path, repo_root),
        "policy_robustness_application_summary": _display_path(application_path, repo_root),
        "selection_contrast_calibration_summary": _display_path(calibration_path, repo_root),
        "config": _display_path(config_path, repo_root),
        "reason_codes": summary["reason_codes"],
        "contract_blockers": summary["contract_blockers"],
        "recommended_next_action": summary["recommended_next_action"],
        "goal_blocked_training_contract_refinement_summary": _display_path(output_file, repo_root),
    }
    print(json.dumps(validation_message, ensure_ascii=False))

    if args.validate_only or args.dry_run:
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "dry-run",
                        "would_write": {
                            "goal_blocked_training_contract_refinement_summary": _display_path(
                                output_file,
                                repo_root,
                            ),
                        },
                        "recommended_next_action": summary["recommended_next_action"],
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
                "goal_blocked_training_contract_refinement_summary": _display_path(output_file, repo_root),
                "recommended_next_action": summary["recommended_next_action"],
                "failure_reason_code_counts": summary["failure_reason_code_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def analyze_goal_blocked_contract_refinement(
    *,
    batch_root: Path,
    review_path: Path,
    smoke_path: Path,
    application_path: Path,
    calibration_path: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    source_summaries: dict[str, Any] = {}
    review = _load_source(
        review_path,
        label="policy_training_readiness_review_summary",
        expected_schema=REVIEW_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
    )
    smoke = _load_source(
        smoke_path,
        label="calibrated_policy_application_smoke_summary",
        expected_schema=SMOKE_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
    )
    application = _load_source(
        application_path,
        label="policy_robustness_application_summary",
        expected_schema=APPLICATION_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
    )
    calibration = _load_source(
        calibration_path,
        label="channel_aware_selection_contrast_calibration_summary",
        expected_schema=CALIBRATION_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
    )
    if _fail_on_input_failure(config):
        for label, payload in (
            ("policy_training_readiness_review_summary", review),
            ("calibrated_policy_application_smoke_summary", smoke),
            ("policy_robustness_application_summary", application),
            ("channel_aware_selection_contrast_calibration_summary", calibration),
        ):
            if payload.get("status") == "failed":
                _append_reason(reason_codes, f"{label}_failed")

    current_git = _git_snapshot(repo_root)
    source_git_matches = [
        _inspect_git(review, label="policy_training_readiness_review_summary", current_git=current_git, config=config, reason_codes=reason_codes),
        _inspect_git(smoke, label="calibrated_policy_application_smoke_summary", current_git=current_git, config=config, reason_codes=reason_codes),
        _inspect_git(application, label="policy_robustness_application_summary", current_git=current_git, config=config, reason_codes=reason_codes),
        _inspect_git(calibration, label="channel_aware_selection_contrast_calibration_summary", current_git=current_git, config=config, reason_codes=reason_codes),
    ]

    contract = _contract_metrics(
        review=review,
        smoke=smoke,
        application=application,
        calibration=calibration,
        validation_reason_codes=reason_codes,
        config=config,
    )
    status = "failed" if reason_codes else "passed"
    failure_reason_counts = Counter(reason_codes)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "failure_reason_code_counts": dict(sorted(failure_reason_counts.items())),
        "batch_root": _display_path(batch_root, repo_root),
        "policy_training_readiness_review_summary_path": _display_path(review_path, repo_root),
        "calibrated_policy_application_smoke_summary_path": _display_path(smoke_path, repo_root),
        "policy_robustness_application_summary_path": _display_path(application_path, repo_root),
        "selection_contrast_calibration_summary_path": _display_path(calibration_path, repo_root),
        "application_scope": "goal_blocked_training_contract_refinement_audit_only",
        "quality_signal_use": "goal_blocked_contract_classification_only",
        "config": _public_config(config),
        "source_summaries": source_summaries,
        "git_provenance": {
            "current": current_git,
            "policy_training_readiness_review": _public_git(review),
            "calibrated_policy_application_smoke": _public_git(smoke),
            "policy_robustness_application": _public_git(application),
            "selection_contrast_calibration": _public_git(calibration),
            "current_matches_sources": all(source_git_matches),
        },
        **contract,
        "runs_training": False,
        "audit_only": True,
        "no_ppo_training": True,
        "no_large_scale_training": True,
        "no_real_world_performance_claim": True,
        "does_not_modify_default_astar": True,
        "does_not_modify_ppo": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "does_not_modify_model_explorer_contract": True,
        "does_not_modify_path_planner_route_contract": True,
        "does_not_modify_path_planner_sidecar_contract": True,
        "no_ackermann_feasible_trajectory_claim": True,
        "non_goals": list(config.get("non_goals", [])),
    }


def _contract_metrics(
    *,
    review: dict[str, Any],
    smoke: dict[str, Any],
    application: dict[str, Any],
    calibration: dict[str, Any],
    validation_reason_codes: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    records = _excluded_records(application, validation_reason_codes)
    safety_regression_count = max(
        _int_value_or_default(review.get("safety_regression_count"), 0),
        _int_value_or_default(smoke.get("safety_regression_count"), 0),
        _int_value_or_default(calibration.get("safety_regression_count"), 0),
    )
    fallback_count = max(_fallback_count(review), _fallback_count(smoke), _fallback_count(calibration))
    contract_mutations = _contract_mutations(
        {
            "policy_training_readiness_review": review,
            "calibrated_policy_application_smoke": smoke,
            "policy_robustness_application": application,
            "selection_contrast_calibration": calibration,
        }
    )
    global_contract_blocked = safety_regression_count > 0 or fallback_count > 0 or bool(contract_mutations)
    contract_decisions = [
        _classify_record(record, global_contract_blocked=global_contract_blocked, config=config)
        for record in records
    ]
    decision_counts = Counter(decision["contract_decision"] for decision in contract_decisions)
    for decision in CONTRACT_DECISIONS:
        decision_counts.setdefault(decision, 0)
    platform_goal_contract_mismatch_count = sum(
        1 for decision in contract_decisions if decision.get("platform_goal_contract_mismatch")
    )
    platform_goal_trainable_anchor_projection_count = sum(
        1 for decision in contract_decisions if _platform_goal_trainable_anchor_projection(decision)
    )
    platform_goal_nontrainable_blocked_target_count = (
        platform_goal_contract_mismatch_count - platform_goal_trainable_anchor_projection_count
    )
    platform_goal_anchor_available_count = sum(
        1
        for decision in contract_decisions
        if decision.get("platform_goal_anchor_available")
    )
    platform_goal_unresolved_count = sum(
        1
        for decision in contract_decisions
        if decision.get("platform_goal_classification") == "unknown_contract_mismatch"
    )
    record_goal_blocked_count = sum(1 for record in records if _is_goal_blocked(record))
    goal_blocked_count = max(
        record_goal_blocked_count,
        _int_value_or_default(review.get("rejected_goal_blocked_count"), 0),
        _int_value_or_default(smoke.get("rejected_goal_blocked_count"), 0),
        _int_value_or_default(calibration.get("goal_blocked_count"), 0),
    )
    excluded_candidate_count = max(
        len(records),
        _int_value_or_default(review.get("excluded_candidate_count"), 0),
    )
    contract_blockers = _contract_blockers(
        validation_reason_codes=validation_reason_codes,
        safety_regression_count=safety_regression_count,
        fallback_count=fallback_count,
        contract_mutations=contract_mutations,
        decision_counts=decision_counts,
    )
    recommended = (
        "fix_validation_failures_before_goal_blocked_contract_refinement"
        if validation_reason_codes
        else "needs_goal_blocked_contract_refinement"
        if contract_blockers
        else "rerun_policy_training_readiness_review"
    )
    return {
        "goal_blocked_count": goal_blocked_count,
        "excluded_candidate_count": excluded_candidate_count,
        "negative_evidence_candidate_count": decision_counts["eligible_negative_evidence"],
        "needs_regeneration_count": decision_counts["needs_regeneration"],
        "contract_decision_counts": dict(sorted(decision_counts.items())),
        "platform_goal_contract_mismatch_count": platform_goal_contract_mismatch_count,
        "platform_goal_trainable_anchor_projection_count": platform_goal_trainable_anchor_projection_count,
        "platform_goal_nontrainable_blocked_target_count": platform_goal_nontrainable_blocked_target_count,
        "platform_goal_anchor_available_count": platform_goal_anchor_available_count,
        "platform_goal_unresolved_count": platform_goal_unresolved_count,
        "contract_blockers": contract_blockers,
        "contract_mutations": contract_mutations,
        "safety_regression_count": safety_regression_count,
        "fallback_or_open_grid_count": fallback_count,
        "contract_decisions": contract_decisions,
        "recommended_next_action": recommended,
    }


def _excluded_records(application: dict[str, Any], reason_codes: list[str]) -> list[dict[str, Any]]:
    channel = application.get("channel_aware_application")
    if not isinstance(channel, dict):
        _append_reason(reason_codes, "channel_aware_application_missing")
        return []
    if channel.get("schema_version") not in (None, CHANNEL_APPLICATION_SCHEMA_VERSION):
        _append_reason(reason_codes, "channel_aware_application_schema_mismatch")
    records = channel.get("records")
    if not isinstance(records, list):
        _append_reason(reason_codes, "channel_aware_application_records_invalid")
        return []
    return [record for record in records if isinstance(record, dict) and _is_excluded(record)]


def _is_excluded(record: dict[str, Any]) -> bool:
    return (
        record.get("application_action") == "exclude_blocked_candidate_evidence"
        or record.get("recommendation") == "reject"
        or _is_goal_blocked(record)
    )


def _is_goal_blocked(record: dict[str, Any]) -> bool:
    return "goal_blocked" in _string_list(record.get("reason_codes")) or "goal_blocked" in _string_list(
        record.get("application_reason_codes")
    )


def _classify_record(
    record: dict[str, Any],
    *,
    global_contract_blocked: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    reason_codes = _string_list(record.get("reason_codes")) + _string_list(record.get("application_reason_codes"))
    platform_goal_classification = _platform_goal_failure_class(record)
    platform_goal_feasibility = record.get("platform_goal_feasibility")
    platform_goal_feasibility = (
        platform_goal_feasibility
        if isinstance(platform_goal_feasibility, dict)
        else {}
    )
    if global_contract_blocked:
        decision = "blocked_by_contract"
        basis = "global_contract_blocker"
    elif platform_goal_classification is not None:
        decision = "needs_regeneration"
        basis = "platform_goal_contract_mismatch"
    elif _is_goal_blocked(record):
        if _has_explicit_candidate_contrast(record, config):
            decision = "eligible_negative_evidence"
            basis = "goal_blocked_with_explicit_candidate_contrast"
        else:
            decision = "needs_regeneration"
            basis = "goal_blocked_without_explicit_candidate_contrast"
    else:
        decision = "excluded_from_positive_training"
        basis = "excluded_non_positive_not_goal_blocked"
    payload = {
        "scenario_id": record.get("scenario_id"),
        "pair_key": record.get("pair_key"),
        "action_index": record.get("action_index"),
        "cell": _list_value(record.get("cell")),
        "recommendation": record.get("recommendation"),
        "application_action": record.get("application_action"),
        "contract_decision": decision,
        "decision_basis": basis,
        "reason_codes": _unique([str(reason) for reason in reason_codes]),
    }
    if platform_goal_classification is not None:
        payload.update(
            {
                "platform_goal_contract_mismatch": True,
                "platform_goal_classification": platform_goal_classification,
                "platform_goal_anchor_available": _platform_goal_anchor_available(
                    platform_goal_feasibility
                ),
                "failure_taxonomy": record.get("failure_taxonomy"),
                "failure_taxonomy_source": record.get("failure_taxonomy_source"),
                "platform_goal_feasibility": platform_goal_feasibility,
            }
        )
    return payload


def _has_explicit_candidate_contrast(record: dict[str, Any], config: dict[str, Any]) -> bool:
    if _platform_goal_failure_class(record) is not None:
        return False
    classification = config.get("classification") if isinstance(config.get("classification"), dict) else {}
    if not bool(classification.get("require_finite_comparison_for_negative_evidence", True)):
        return True
    if not _list_value(record.get("cell")):
        return False
    if record.get("action_index") is None:
        return False
    comparison = record.get("comparison") if isinstance(record.get("comparison"), dict) else {}
    for key in ("path_cost_delta", "channel_cost_delta", "high_cost_exposure_delta", "risk_delta"):
        if _finite_number(comparison.get(key)):
            return True
    return False


def _platform_goal_failure_class(record: dict[str, Any]) -> str | None:
    for key in ("failure_taxonomy", "platform_goal_classification"):
        value = record.get(key)
        if isinstance(value, str) and value in PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES:
            return value
    for reason in _string_list(record.get("reason_codes")):
        if reason in PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES:
            return reason
    feasibility = record.get("platform_goal_feasibility")
    feasibility = feasibility if isinstance(feasibility, dict) else {}
    classification = feasibility.get("classification")
    if isinstance(classification, str) and classification in PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES:
        return classification
    return None


def _platform_goal_anchor_available(feasibility: dict[str, Any]) -> bool:
    anchor = feasibility.get("nearest_inflated_passable_anchor")
    return isinstance(anchor, list) and len(anchor) == 2


def _platform_goal_trainable_anchor_projection(decision: dict[str, Any]) -> bool:
    if decision.get("platform_goal_classification") != "platform_inflated_goal_blocked":
        return False
    feasibility = decision.get("platform_goal_feasibility")
    feasibility = feasibility if isinstance(feasibility, dict) else {}
    projection = feasibility.get("anchor_projection")
    projection = projection if isinstance(projection, dict) else {}
    return projection.get("training_use") not in (None, "", "not_positive_evidence")


def _contract_blockers(
    *,
    validation_reason_codes: list[str],
    safety_regression_count: int,
    fallback_count: int,
    contract_mutations: list[str],
    decision_counts: Counter[str],
) -> list[str]:
    blockers: list[str] = []
    for reason in validation_reason_codes:
        _append_reason(blockers, reason)
    if safety_regression_count > 0:
        _append_reason(blockers, "safety_regression_blocks_goal_blocked_contract")
    if fallback_count > 0:
        _append_reason(blockers, "fallback_or_open_grid_blocks_goal_blocked_contract")
    if contract_mutations:
        _append_reason(blockers, "contract_mutation_blocks_goal_blocked_contract")
    if decision_counts["needs_regeneration"] > 0:
        _append_reason(blockers, "goal_blocked_records_need_regeneration")
    if decision_counts["blocked_by_contract"] > 0:
        _append_reason(blockers, "goal_blocked_records_blocked_by_contract")
    return blockers


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
        output_files.get("goal_blocked_training_contract_refinement_summary")
    ):
        raise ConfigError("output_files.goal_blocked_training_contract_refinement_summary must be a non-empty string")
    classification = payload.get("classification", {})
    if not isinstance(classification, dict):
        raise ConfigError("classification must be an object")
    config = dict(payload)
    config["classification"] = {
        "require_finite_comparison_for_negative_evidence": bool(
            classification.get("require_finite_comparison_for_negative_evidence", True)
        )
    }
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
            "reason_codes": _string_list(payload.get("reason_codes")),
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
) -> bool:
    return _inspect_source_git_provenance(
        payload,
        label=label,
        current_git=current_git,
        require_current_git_match=_require_current_git_match(config),
        reason_codes=reason_codes,
        submodules=SUBMODULES,
    )


def _fallback_count(payload: dict[str, Any]) -> int:
    return max(_int_value_or_default(payload.get(field), 0) for field in FALLBACK_COUNT_FIELDS)


def _contract_mutations(labeled_payloads: dict[str, dict[str, Any]]) -> list[str]:
    mutations: list[str] = []
    for label, payload in labeled_payloads.items():
        for field in CONTRACT_GUARD_FIELDS:
            if payload.get(field) is False:
                mutations.append(f"{label}.{field}")
    return sorted(set(mutations))


def _output_file(batch_root: Path, config: dict[str, Any]) -> Path:
    return batch_root / config["output_files"]["goal_blocked_training_contract_refinement_summary"]


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": config.get("schema_version"),
        "validation": dict(config.get("validation", {})) if isinstance(config.get("validation"), dict) else {},
        "classification": dict(config.get("classification", {})),
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_reason(reason_codes: list[str], code: str) -> None:
    if code not in reason_codes:
        reason_codes.append(code)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _unique(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})


def _list_value(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _int_value_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


if __name__ == "__main__":
    raise SystemExit(main())
