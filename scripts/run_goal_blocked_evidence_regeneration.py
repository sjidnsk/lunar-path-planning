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


CONFIG_SCHEMA_VERSION = "goal-blocked-evidence-regeneration-config/v1"
SUMMARY_SCHEMA_VERSION = "goal-blocked-evidence-regeneration-summary/v1"
REFINEMENT_SCHEMA_VERSION = "goal-blocked-training-contract-refinement-summary/v1"
APPLICATION_SCHEMA_VERSION = "policy-robustness-application-summary/v1"
REVIEW_SCHEMA_VERSION = "policy-training-readiness-review-summary/v1"
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
FAILURE_TAXONOMY = (
    "target_unreachable",
    "candidate_generation_failed",
    "candidate_mask_empty",
    "route_generation_failed",
    "missing_candidate_contrast",
    "blocked_by_contract",
    "platform_inflated_goal_blocked",
    "original_goal_blocked",
    "out_of_bounds",
    "unknown_contract_mismatch",
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
        description="Regenerate goal-blocked evidence diagnostics without running policy training."
    )
    parser.add_argument("--batch-root", required=True, help="Batch root containing current evidence summaries.")
    parser.add_argument(
        "--goal-blocked-training-contract-refinement-summary",
        help="goal-blocked-training-contract-refinement-summary/v1 JSON. Defaults to <batch-root>/goal-blocked-training-contract-refinement-summary.json.",
    )
    parser.add_argument(
        "--policy-robustness-application-summary",
        help="policy-robustness-application-summary/v1 JSON. Defaults to <batch-root>/policy-robustness-application-summary.json.",
    )
    parser.add_argument(
        "--policy-training-readiness-review-summary",
        help="policy-training-readiness-review-summary/v1 JSON. Defaults to <batch-root>/policy-training-readiness-review-summary.json.",
    )
    parser.add_argument("--config", required=True, help="Goal-blocked evidence regeneration config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned output paths.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without writing outputs.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    batch_root = _resolve_path(args.batch_root, repo_root)
    refinement_path = (
        _resolve_path(args.goal_blocked_training_contract_refinement_summary, repo_root)
        if args.goal_blocked_training_contract_refinement_summary
        else batch_root / "goal-blocked-training-contract-refinement-summary.json"
    )
    application_path = (
        _resolve_path(args.policy_robustness_application_summary, repo_root)
        if args.policy_robustness_application_summary
        else batch_root / "policy-robustness-application-summary.json"
    )
    review_path = (
        _resolve_path(args.policy_training_readiness_review_summary, repo_root)
        if args.policy_training_readiness_review_summary
        else batch_root / "policy-training-readiness-review-summary.json"
    )
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    summary = analyze_goal_blocked_evidence_regeneration(
        batch_root=batch_root,
        refinement_path=refinement_path,
        application_path=application_path,
        review_path=review_path,
        config=config,
        repo_root=repo_root,
    )
    output_file = _output_file(batch_root, config)
    validation_message = {
        "status": "config validated" if summary["status"] == "passed" else "validation failed",
        "batch_root": _display_path(batch_root, repo_root),
        "goal_blocked_training_contract_refinement_summary": _display_path(refinement_path, repo_root),
        "policy_robustness_application_summary": _display_path(application_path, repo_root),
        "policy_training_readiness_review_summary": _display_path(review_path, repo_root),
        "config": _display_path(config_path, repo_root),
        "reason_codes": summary["reason_codes"],
        "needs_regeneration_input_count": summary["needs_regeneration_input_count"],
        "regenerated_record_count": summary["regenerated_record_count"],
        "still_unresolved_count": summary["still_unresolved_count"],
        "contract_blockers": summary["contract_blockers"],
        "recommended_next_action": summary["recommended_next_action"],
        "goal_blocked_evidence_regeneration_summary": _display_path(output_file, repo_root),
    }
    print(json.dumps(validation_message, ensure_ascii=False))

    if args.validate_only or args.dry_run:
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "dry-run",
                        "would_write": {
                            "goal_blocked_evidence_regeneration_summary": _display_path(
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
                "goal_blocked_evidence_regeneration_summary": _display_path(output_file, repo_root),
                "recommended_next_action": summary["recommended_next_action"],
                "failure_reason_code_counts": summary["failure_reason_code_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def analyze_goal_blocked_evidence_regeneration(
    *,
    batch_root: Path,
    refinement_path: Path,
    application_path: Path,
    review_path: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    source_summaries: dict[str, Any] = {}
    refinement = _load_source(
        refinement_path,
        label="goal_blocked_training_contract_refinement_summary",
        expected_schema=REFINEMENT_SCHEMA_VERSION,
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
    review = _load_source(
        review_path,
        label="policy_training_readiness_review_summary",
        expected_schema=REVIEW_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
    )
    if _fail_on_input_failure(config):
        for label, payload in (
            ("goal_blocked_training_contract_refinement_summary", refinement),
            ("policy_robustness_application_summary", application),
            ("policy_training_readiness_review_summary", review),
        ):
            if payload.get("status") == "failed":
                _append_reason(reason_codes, f"{label}_failed")

    current_git = _git_snapshot(repo_root)
    source_git_matches = [
        _inspect_git(
            refinement,
            label="goal_blocked_training_contract_refinement_summary",
            current_git=current_git,
            config=config,
            reason_codes=reason_codes,
        ),
        _inspect_git(
            application,
            label="policy_robustness_application_summary",
            current_git=current_git,
            config=config,
            reason_codes=reason_codes,
        ),
        _inspect_git(
            review,
            label="policy_training_readiness_review_summary",
            current_git=current_git,
            config=config,
            reason_codes=reason_codes,
        ),
    ]

    regeneration = _regeneration_metrics(
        refinement=refinement,
        application=application,
        review=review,
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
        "goal_blocked_training_contract_refinement_summary_path": _display_path(refinement_path, repo_root),
        "policy_robustness_application_summary_path": _display_path(application_path, repo_root),
        "policy_training_readiness_review_summary_path": _display_path(review_path, repo_root),
        "application_scope": "goal_blocked_evidence_regeneration_audit_only",
        "quality_signal_use": "goal_blocked_failure_diagnostic_only",
        "config": _public_config(config),
        "source_summaries": source_summaries,
        "git_provenance": {
            "current": current_git,
            "goal_blocked_training_contract_refinement": _public_git(refinement),
            "policy_robustness_application": _public_git(application),
            "policy_training_readiness_review": _public_git(review),
            "current_matches_sources": all(source_git_matches),
        },
        **regeneration,
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


def _regeneration_metrics(
    *,
    refinement: dict[str, Any],
    application: dict[str, Any],
    review: dict[str, Any],
    validation_reason_codes: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    decisions = _needs_regeneration_decisions(refinement)
    application_records = _application_records(application, validation_reason_codes)
    records_by_key = {_record_key(record): record for record in application_records}
    safety_regression_count = max(
        _int_value_or_default(refinement.get("safety_regression_count"), 0),
        _int_value_or_default(application.get("safety_regression_count"), 0),
        _int_value_or_default(review.get("safety_regression_count"), 0),
    )
    fallback_count = max(_fallback_count(refinement), _fallback_count(application), _fallback_count(review))
    contract_mutations = _contract_mutations(
        {
            "goal_blocked_training_contract_refinement": refinement,
            "policy_robustness_application": application,
            "policy_training_readiness_review": review,
        }
    )
    global_contract_blocked = safety_regression_count > 0 or fallback_count > 0 or bool(contract_mutations)
    regenerated_records = [
        _regenerate_record(
            decision=decision,
            source_record=records_by_key.get(_record_key(decision)),
            global_contract_blocked=global_contract_blocked,
            config=config,
        )
        for decision in decisions
    ]
    taxonomy_counts = Counter(
        record["failure_category"] for record in regenerated_records if record["failure_category"] is not None
    )
    contrast_status_counts = Counter(
        record["candidate_contrast_status"]
        for record in regenerated_records
        if record.get("candidate_contrast_status") is not None
    )
    upstream_blocker_reason_counts = Counter(
        record["upstream_blocker_reason"]
        for record in regenerated_records
        if record.get("upstream_blocker_reason") is not None
    )
    failure_taxonomy_source_counts = Counter(
        record["source_failure_taxonomy_source"]
        for record in regenerated_records
        if record.get("source_failure_taxonomy_source") is not None
    )
    for category in FAILURE_TAXONOMY:
        taxonomy_counts.setdefault(category, 0)
    for status in ("finite_candidate_comparison", "missing_candidate_contrast"):
        contrast_status_counts.setdefault(status, 0)
    eligible_count = sum(
        1
        for record in regenerated_records
        if record["diagnostic_decision"] == "eligible_negative_evidence_candidate"
    )
    platform_goal_contract_mismatch_count = sum(
        1
        for record in regenerated_records
        if record["diagnostic_decision"] == "platform_goal_contract_mismatch"
    )
    platform_goal_trainable_anchor_projection_count = sum(
        1 for record in regenerated_records if _platform_goal_trainable_anchor_projection(record)
    )
    platform_goal_nontrainable_blocked_target_count = (
        platform_goal_contract_mismatch_count - platform_goal_trainable_anchor_projection_count
    )
    platform_goal_anchor_available_count = sum(
        1 for record in regenerated_records if record.get("platform_goal_anchor_available")
    )
    platform_goal_unresolved_count = sum(
        1
        for record in regenerated_records
        if record.get("platform_goal_classification") == "unknown_contract_mismatch"
    )
    still_unresolved_count = sum(
        1
        for record in regenerated_records
        if record["diagnostic_decision"] in {"unresolved", "blocked_by_contract"}
    )
    contract_blockers = _contract_blockers(
        validation_reason_codes=validation_reason_codes,
        safety_regression_count=safety_regression_count,
        fallback_count=fallback_count,
        contract_mutations=contract_mutations,
        regenerated_records=regenerated_records,
        still_unresolved_count=still_unresolved_count,
    )
    recommended = (
        "rerun_goal_blocked_training_contract_refinement"
        if still_unresolved_count == 0 and not contract_blockers
        else "needs_goal_blocked_diagnostic_refinement"
    )
    return {
        "needs_regeneration_input_count": len(decisions),
        "regenerated_record_count": len(regenerated_records),
        "failure_taxonomy_counts": dict(sorted(taxonomy_counts.items())),
        "candidate_contrast_status_counts": dict(sorted(contrast_status_counts.items())),
        "upstream_blocker_reason_counts": dict(sorted(upstream_blocker_reason_counts.items())),
        "failure_taxonomy_source_counts": dict(sorted(failure_taxonomy_source_counts.items())),
        "upstream_diagnostic_blockers": _upstream_diagnostic_blockers(
            regenerated_records=regenerated_records,
            eligible_count=eligible_count,
        ),
        "eligible_negative_evidence_candidate_count": eligible_count,
        "platform_goal_contract_mismatch_count": platform_goal_contract_mismatch_count,
        "platform_goal_trainable_anchor_projection_count": platform_goal_trainable_anchor_projection_count,
        "platform_goal_nontrainable_blocked_target_count": platform_goal_nontrainable_blocked_target_count,
        "platform_goal_anchor_available_count": platform_goal_anchor_available_count,
        "platform_goal_unresolved_count": platform_goal_unresolved_count,
        "still_unresolved_count": still_unresolved_count,
        "contract_blockers": contract_blockers,
        "contract_mutations": contract_mutations,
        "safety_regression_count": safety_regression_count,
        "fallback_or_open_grid_count": fallback_count,
        "regenerated_records": regenerated_records,
        "recommended_next_action": recommended,
    }


def _needs_regeneration_decisions(refinement: dict[str, Any]) -> list[dict[str, Any]]:
    decisions = refinement.get("contract_decisions")
    if not isinstance(decisions, list):
        return []
    return [
        decision
        for decision in decisions
        if isinstance(decision, dict) and decision.get("contract_decision") == "needs_regeneration"
    ]


def _application_records(application: dict[str, Any], reason_codes: list[str]) -> list[dict[str, Any]]:
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
    return [record for record in records if isinstance(record, dict)]


def _regenerate_record(
    *,
    decision: dict[str, Any],
    source_record: dict[str, Any] | None,
    global_contract_blocked: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    source = source_record if isinstance(source_record, dict) else {}
    reason_codes = _all_reason_codes(decision, source)
    platform_goal_classification = _platform_goal_failure_class(decision, source)
    platform_goal_feasibility = _platform_goal_feasibility(decision, source)
    if global_contract_blocked:
        diagnostic_decision = "blocked_by_contract"
        failure_category: str | None = "blocked_by_contract"
        basis = "global_hard_blocker_prevents_training_readiness"
    elif platform_goal_classification is not None:
        diagnostic_decision = "platform_goal_contract_mismatch"
        failure_category = platform_goal_classification
        basis = "platform_goal_contract_mismatch_diagnostic"
    elif source and _has_explicit_candidate_contrast(source, config):
        diagnostic_decision = "eligible_negative_evidence_candidate"
        failure_category = None
        basis = "finite_candidate_comparison_available"
    else:
        diagnostic_decision = "unresolved"
        failure_category = _failure_category(decision, source)
        basis = f"{failure_category}_diagnostic"
    payload = {
        "scenario_id": decision.get("scenario_id"),
        "pair_key": decision.get("pair_key"),
        "action_index": decision.get("action_index"),
        "cell": _list_value(decision.get("cell")) or _list_value(source.get("cell")),
        "original_contract_decision": decision.get("contract_decision"),
        "diagnostic_decision": diagnostic_decision,
        "failure_category": failure_category,
        "diagnostic_basis": basis,
        "eligible_negative_evidence_candidate": diagnostic_decision == "eligible_negative_evidence_candidate",
        "has_source_record": bool(source),
        "has_finite_candidate_comparison": bool(source and _has_explicit_candidate_contrast(source, config)),
        "candidate_contrast_status": source.get("candidate_contrast_status")
        if isinstance(source.get("candidate_contrast_status"), str)
        else "finite_candidate_comparison"
        if bool(source and _has_explicit_candidate_contrast(source, config))
        else "missing_candidate_contrast",
        "upstream_blocker_reason": source.get("upstream_blocker_reason"),
        "source_failure_taxonomy": source.get("failure_taxonomy"),
        "source_failure_taxonomy_source": source.get("failure_taxonomy_source"),
        "reason_codes": _unique(reason_codes),
    }
    if platform_goal_classification is not None:
        payload.update(
            {
                "platform_goal_contract_mismatch": True,
                "platform_goal_classification": platform_goal_classification,
                "platform_goal_anchor_available": _platform_goal_anchor_available(
                    platform_goal_feasibility
                ),
                "platform_goal_feasibility": platform_goal_feasibility,
            }
        )
    return payload


def _failure_category(decision: dict[str, Any], source: dict[str, Any]) -> str:
    platform_class = _platform_goal_failure_class(decision, source)
    if platform_class is not None:
        return platform_class
    for payload in (source, decision):
        taxonomy = payload.get("failure_taxonomy")
        if isinstance(taxonomy, str) and taxonomy in FAILURE_TAXONOMY:
            return taxonomy
    text = " ".join(str(item).lower() for item in _diagnostic_tokens(decision, source))
    if "target_unreachable" in text or "goal_unreachable" in text:
        return "target_unreachable"
    if "candidate_mask_empty" in text or "mask_empty" in text or "candidate_list_empty" in text:
        return "candidate_mask_empty"
    if "candidate_generation_failed" in text or "candidate_generation" in text or "candidate_failed" in text:
        return "candidate_generation_failed"
    if (
        "route_generation_failed" in text
        or "route_generation" in text
        or "route_failed" in text
        or "planner_route_failed" in text
        or "path_planner_failed" in text
        or "search_failed" in text
    ):
        return "route_generation_failed"
    if not source:
        return "candidate_generation_failed"
    return "missing_candidate_contrast"


def _diagnostic_tokens(decision: dict[str, Any], source: dict[str, Any]) -> list[Any]:
    tokens: list[Any] = []
    for payload in (decision, source):
        for key in (
            "decision_basis",
            "reason_codes",
            "application_reason_codes",
            "blocker_reason",
            "failure_reason",
            "failure_reasons",
            "fallback_reason",
            "diagnostic_reason",
            "route_failure_reason",
            "failure_taxonomy",
            "platform_goal_classification",
            "platform_goal_feasibility",
            "upstream_blocker_reason",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                tokens.extend(value)
            elif value is not None:
                tokens.append(value)
    return tokens


def _platform_goal_failure_class(decision: dict[str, Any], source: dict[str, Any]) -> str | None:
    for payload in (source, decision):
        for key in ("failure_taxonomy", "platform_goal_classification"):
            value = payload.get(key)
            if isinstance(value, str) and value in PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES:
                return value
        for reason in _string_list(payload.get("reason_codes")):
            if reason in PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES:
                return reason
        feasibility = payload.get("platform_goal_feasibility")
        feasibility = feasibility if isinstance(feasibility, dict) else {}
        classification = feasibility.get("classification")
        if isinstance(classification, str) and classification in PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES:
            return classification
    return None


def _platform_goal_feasibility(decision: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    for payload in (source, decision):
        feasibility = payload.get("platform_goal_feasibility")
        if isinstance(feasibility, dict):
            return dict(feasibility)
    return {}


def _platform_goal_anchor_available(feasibility: dict[str, Any]) -> bool:
    anchor = feasibility.get("nearest_inflated_passable_anchor")
    return isinstance(anchor, list) and len(anchor) == 2


def _platform_goal_trainable_anchor_projection(record: dict[str, Any]) -> bool:
    if record.get("platform_goal_classification") != "platform_inflated_goal_blocked":
        return False
    feasibility = record.get("platform_goal_feasibility")
    feasibility = feasibility if isinstance(feasibility, dict) else {}
    projection = feasibility.get("anchor_projection")
    projection = projection if isinstance(projection, dict) else {}
    return projection.get("training_use") not in (None, "", "not_positive_evidence")


def _all_reason_codes(decision: dict[str, Any], source: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    for payload in (decision, source):
        codes.extend(_string_list(payload.get("reason_codes")))
        codes.extend(_string_list(payload.get("application_reason_codes")))
    return codes


def _upstream_diagnostic_blockers(
    *,
    regenerated_records: list[dict[str, Any]],
    eligible_count: int,
) -> list[str]:
    blockers: list[str] = []
    if any(
        record.get("candidate_contrast_status") == "missing_candidate_contrast"
        and record.get("diagnostic_decision") != "platform_goal_contract_mismatch"
        for record in regenerated_records
    ):
        _append_reason(blockers, "upstream_goal_blocked_records_without_finite_candidate_comparison")
    if eligible_count == 0 and any(
        record.get("diagnostic_decision") != "platform_goal_contract_mismatch"
        for record in regenerated_records
    ):
        _append_reason(blockers, "upstream_goal_blocked_records_have_no_eligible_negative_evidence")
    if any(not record.get("source_failure_taxonomy") for record in regenerated_records):
        _append_reason(blockers, "upstream_goal_blocked_records_without_failure_taxonomy")
    return blockers


def _has_explicit_candidate_contrast(record: dict[str, Any], config: dict[str, Any]) -> bool:
    diagnostic = config.get("diagnostic") if isinstance(config.get("diagnostic"), dict) else {}
    if not bool(diagnostic.get("require_finite_comparison_for_negative_evidence", True)):
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


def _contract_blockers(
    *,
    validation_reason_codes: list[str],
    safety_regression_count: int,
    fallback_count: int,
    contract_mutations: list[str],
    regenerated_records: list[dict[str, Any]],
    still_unresolved_count: int,
) -> list[str]:
    blockers: list[str] = []
    for reason in validation_reason_codes:
        _append_reason(blockers, reason)
    if safety_regression_count > 0:
        _append_reason(blockers, "safety_regression_blocks_goal_blocked_regeneration")
    if fallback_count > 0:
        _append_reason(blockers, "fallback_or_open_grid_blocks_goal_blocked_regeneration")
    if contract_mutations:
        _append_reason(blockers, "contract_mutation_blocks_goal_blocked_regeneration")
    if any(record["diagnostic_decision"] == "blocked_by_contract" for record in regenerated_records):
        _append_reason(blockers, "goal_blocked_records_blocked_by_contract")
    if still_unresolved_count > 0:
        _append_reason(blockers, "goal_blocked_records_still_unresolved")
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
        output_files.get("goal_blocked_evidence_regeneration_summary")
    ):
        raise ConfigError("output_files.goal_blocked_evidence_regeneration_summary must be a non-empty string")
    diagnostic = payload.get("diagnostic", {})
    if not isinstance(diagnostic, dict):
        raise ConfigError("diagnostic must be an object")
    config = dict(payload)
    config["diagnostic"] = {
        "require_finite_comparison_for_negative_evidence": bool(
            diagnostic.get("require_finite_comparison_for_negative_evidence", True)
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
        impact = payload.get("contract_impact") if isinstance(payload.get("contract_impact"), dict) else {}
        impact_mutations = impact.get("contract_mutations")
        if isinstance(impact_mutations, list):
            for mutation in impact_mutations:
                if mutation:
                    mutations.append(f"{label}.contract_impact.{mutation}")
    return sorted(set(mutations))


def _output_file(batch_root: Path, config: dict[str, Any]) -> Path:
    return batch_root / config["output_files"]["goal_blocked_evidence_regeneration_summary"]


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": config.get("schema_version"),
        "validation": dict(config.get("validation", {})) if isinstance(config.get("validation"), dict) else {},
        "diagnostic": dict(config.get("diagnostic", {})),
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


def _record_key(record: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (record.get("scenario_id"), record.get("pair_key"), record.get("action_index"))


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
