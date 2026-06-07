from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot
from git_provenance import git_snapshots_match as _git_snapshots_match
from git_provenance import inspect_source_git_provenance as _inspect_source_git_provenance
from git_provenance import public_git as _public_git


CONFIG_SCHEMA_VERSION = "anchor-projection-evidence-contract-config/v1"
SUMMARY_SCHEMA_VERSION = "anchor-projection-evidence-contract-summary/v1"
REGENERATION_SCHEMA_VERSION = "goal-blocked-evidence-regeneration-summary/v1"
REVIEW_SCHEMA_VERSION = "policy-training-readiness-review-summary/v1"
SUBMODULES = ("dev-platform-constraints", "model-explorer", "path-planner")
FALLBACK_COUNT_FIELDS = (
    "open_grid_fallback_used_count",
    "open_grid_fallback_count",
    "fallback_used_count",
    "fallback_or_open_grid_count",
)
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
        description="Classify platform anchor projections before any policy training."
    )
    parser.add_argument("--batch-root", required=True, help="Batch root containing current evidence summaries.")
    parser.add_argument(
        "--goal-blocked-evidence-regeneration-summary",
        help="goal-blocked-evidence-regeneration-summary/v1 JSON. Defaults to <batch-root>/goal-blocked-evidence-regeneration-summary.json.",
    )
    parser.add_argument(
        "--policy-training-readiness-review-summary",
        help="policy-training-readiness-review-summary/v1 JSON. Defaults to <batch-root>/policy-training-readiness-review-summary.json.",
    )
    parser.add_argument("--config", required=True, help="Anchor projection contract config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned output paths.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without writing outputs.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    batch_root = _resolve_path(args.batch_root, repo_root)
    regeneration_path = (
        _resolve_path(args.goal_blocked_evidence_regeneration_summary, repo_root)
        if args.goal_blocked_evidence_regeneration_summary
        else batch_root / "goal-blocked-evidence-regeneration-summary.json"
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

    summary = analyze_anchor_projection_contract(
        batch_root=batch_root,
        regeneration_path=regeneration_path,
        review_path=review_path,
        config=config,
        repo_root=repo_root,
    )
    output_file = _output_file(batch_root, config)
    validation_message = {
        "status": "config validated" if summary["status"] == "passed" else "validation failed",
        "batch_root": _display_path(batch_root, repo_root),
        "goal_blocked_evidence_regeneration_summary": _display_path(regeneration_path, repo_root),
        "policy_training_readiness_review_summary": _display_path(review_path, repo_root),
        "config": _display_path(config_path, repo_root),
        "reason_codes": summary["reason_codes"],
        "current_git_provenance_mismatch_count": summary["current_git_provenance_mismatch_count"],
        "git_provenance_mismatch_count": summary["git_provenance_mismatch_count"],
        "platform_goal_contract_mismatch_count": summary["platform_goal_contract_mismatch_count"],
        "trainable_anchor_projection_count": summary["trainable_anchor_projection_count"],
        "nontrainable_blocked_target_count": summary["nontrainable_blocked_target_count"],
        "platform_goal_unresolved_count": summary["platform_goal_unresolved_count"],
        "contract_blockers": summary["contract_blockers"],
        "recommended_next_action": summary["recommended_next_action"],
        "anchor_projection_evidence_contract_summary": _display_path(output_file, repo_root),
    }
    print(json.dumps(validation_message, ensure_ascii=False))

    if args.validate_only or args.dry_run:
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "dry-run",
                        "would_write": {
                            "anchor_projection_evidence_contract_summary": _display_path(
                                output_file,
                                repo_root,
                            )
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
                "anchor_projection_evidence_contract_summary": _display_path(output_file, repo_root),
                "recommended_next_action": summary["recommended_next_action"],
                "failure_reason_code_counts": summary["failure_reason_code_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def analyze_anchor_projection_contract(
    *,
    batch_root: Path,
    regeneration_path: Path,
    review_path: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    source_summaries: dict[str, Any] = {}
    regeneration = _load_source(
        regeneration_path,
        label="goal_blocked_evidence_regeneration_summary",
        expected_schema=REGENERATION_SCHEMA_VERSION,
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
            ("goal_blocked_evidence_regeneration_summary", regeneration),
            ("policy_training_readiness_review_summary", review),
        ):
            if payload.get("status") == "failed":
                _append_reason(reason_codes, f"{label}_failed")

    current_git = _git_snapshot(repo_root)
    source_git_matches = [
        _inspect_git(
            regeneration,
            label="goal_blocked_evidence_regeneration_summary",
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
    contract = _contract_metrics(
        regeneration=regeneration,
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
        "current_git_provenance_mismatch_count": failure_reason_counts.get("current_git_provenance_mismatch", 0),
        "git_provenance_mismatch_count": failure_reason_counts.get("git_provenance_mismatch", 0),
        "batch_root": _display_path(batch_root, repo_root),
        "goal_blocked_evidence_regeneration_summary_path": _display_path(regeneration_path, repo_root),
        "policy_training_readiness_review_summary_path": _display_path(review_path, repo_root),
        "application_scope": "anchor_projection_evidence_contract_audit_only",
        "quality_signal_use": "anchor_projection_training_contract_classification_only",
        "source_summaries": source_summaries,
        "config": _public_config(config),
        "git_provenance": {
            "current": current_git,
            "goal_blocked_evidence_regeneration": _public_git(regeneration),
            "policy_training_readiness_review": _public_git(review),
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
    regeneration: dict[str, Any],
    review: dict[str, Any],
    validation_reason_codes: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    records = _platform_goal_records(regeneration, validation_reason_codes)
    decisions = [_classify_anchor_projection(record, config=config) for record in records]
    decision_counts = Counter(decision["contract_decision"] for decision in decisions)
    for decision in (
        "trainable_anchor_projection_contrast",
        "nontrainable_blocked_target",
        "unresolved",
    ):
        decision_counts.setdefault(decision, 0)
    safety_regression_count = max(
        _int_value_or_default(regeneration.get("safety_regression_count"), 0),
        _int_value_or_default(review.get("safety_regression_count"), 0),
    )
    fallback_count = max(_fallback_count(regeneration), _fallback_count(review))
    contract_mutations = _contract_mutations(
        {
            "goal_blocked_evidence_regeneration": regeneration,
            "policy_training_readiness_review": review,
        }
    )
    platform_goal_contract_mismatch_count = len(records)
    trainable_count = decision_counts["trainable_anchor_projection_contrast"]
    nontrainable_count = decision_counts["nontrainable_blocked_target"]
    unresolved_count = decision_counts["unresolved"]
    anchor_available_count = sum(1 for decision in decisions if decision.get("anchor_available"))
    audit_proxy_positive_count = sum(
        1
        for decision in decisions
        if decision["contract_decision"] == "trainable_anchor_projection_contrast"
        and decision.get("comparison_scope") in _forbidden_positive_scopes(config)
    )
    negative_scope_violation_count = sum(
        1
        for record in _records(regeneration)
        if bool(record.get("eligible_negative_evidence_candidate"))
        and _projection_scope(record) in _forbidden_positive_scopes(config)
    )
    contract_blockers = _contract_blockers(
        validation_reason_codes=validation_reason_codes,
        safety_regression_count=safety_regression_count,
        fallback_count=fallback_count,
        contract_mutations=contract_mutations,
        unresolved_count=unresolved_count,
        max_unresolved_count=_max_unresolved_count(config),
        audit_proxy_positive_count=audit_proxy_positive_count,
        negative_scope_violation_count=negative_scope_violation_count,
    )
    for blocker in contract_blockers:
        if blocker not in {
            "anchor_projection_records_unresolved",
        }:
            _append_reason(validation_reason_codes, blocker)
    if unresolved_count > _max_unresolved_count(config):
        _append_reason(validation_reason_codes, "platform_goal_unresolved_count_exceeds_threshold")
    recommended = (
        "fix_validation_failures_before_anchor_projection_contract"
        if validation_reason_codes
        else "rerun_policy_training_readiness_review_with_anchor_projection_contract"
        if trainable_count > 0
        else "keep_platform_blocked_targets_out_of_training"
    )
    return {
        "platform_goal_contract_mismatch_count": platform_goal_contract_mismatch_count,
        "trainable_anchor_projection_count": trainable_count,
        "platform_goal_trainable_anchor_projection_count": trainable_count,
        "nontrainable_blocked_target_count": nontrainable_count,
        "platform_goal_nontrainable_blocked_target_count": nontrainable_count,
        "platform_goal_anchor_available_count": anchor_available_count,
        "platform_goal_unresolved_count": unresolved_count,
        "positive_training_evidence_contains_audit_proxy_anchor_count": audit_proxy_positive_count,
        "negative_evidence_scope_violation_count": negative_scope_violation_count,
        "eligible_negative_evidence_candidate_count": _int_value_or_default(
            regeneration.get("eligible_negative_evidence_candidate_count"),
            0,
        ),
        "contract_decision_counts": dict(sorted(decision_counts.items())),
        "contract_blockers": contract_blockers,
        "contract_mutations": contract_mutations,
        "safety_regression_count": safety_regression_count,
        "fallback_or_open_grid_count": fallback_count,
        "source_selected_candidate_changed_rate": review.get("source_selected_candidate_changed_rate"),
        "training_readiness_status": review.get("training_readiness_status"),
        "source_selection_improvement_supported": _finite_number(
            review.get("source_selected_candidate_changed_rate")
        )
        and float(review["source_selected_candidate_changed_rate"]) > 0.0,
        "anchor_projection_decisions": decisions,
        "recommended_next_action": recommended,
    }


def _platform_goal_records(regeneration: dict[str, Any], reason_codes: list[str]) -> list[dict[str, Any]]:
    records = _records(regeneration)
    platform_records = []
    for record in records:
        if not isinstance(record, dict):
            continue
        classification = _platform_goal_classification(record)
        if record.get("platform_goal_contract_mismatch") or classification in PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES:
            platform_records.append(record)
    expected = _int_value_or_default(regeneration.get("platform_goal_contract_mismatch_count"), len(platform_records))
    if expected != len(platform_records):
        _append_reason(reason_codes, "platform_goal_contract_mismatch_count_inconsistent")
    return platform_records


def _classify_anchor_projection(record: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
    feasibility = record.get("platform_goal_feasibility")
    feasibility = feasibility if isinstance(feasibility, dict) else {}
    projection = feasibility.get("anchor_projection")
    projection = projection if isinstance(projection, dict) else {}
    classification = _platform_goal_classification(record)
    anchor = projection.get("nearest_inflated_passable_anchor")
    if not _cell(anchor):
        anchor = feasibility.get("nearest_inflated_passable_anchor")
    comparison_scope = _projection_scope(record)
    training_use = projection.get("training_use")
    distance_m = _number_or_none(projection.get("projection_distance_m"))
    if distance_m is None:
        distance_m = _number_or_none(feasibility.get("anchor_distance_m"))
    distance_cells = _int_or_none(projection.get("projection_distance_cells"))
    if distance_cells is None:
        distance_cells = _int_or_none(feasibility.get("anchor_distance_cells"))
    anchor_reachable = _anchor_reachable(feasibility, projection)
    reject_reasons: list[str] = []
    if classification not in PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES:
        reject_reasons.append("not_platform_goal_contract_mismatch")
    if classification == "unknown_contract_mismatch":
        decision = "unresolved"
        reject_reasons.append("platform_goal_classification_unknown")
    else:
        if classification != "platform_inflated_goal_blocked":
            reject_reasons.append("platform_goal_classification_not_trainable_projection")
        if not _cell(anchor):
            reject_reasons.append("nearest_inflated_passable_anchor_missing")
        if _require_anchor_reachable(config) and not anchor_reachable:
            reject_reasons.append("anchor_not_reachable")
        if distance_m is None or not _finite_number(distance_m):
            reject_reasons.append("projection_distance_m_missing")
        elif distance_m > _max_projection_distance_m(config):
            reject_reasons.append("projection_distance_m_exceeds_contract")
        if distance_cells is None:
            reject_reasons.append("projection_distance_cells_missing")
        elif distance_cells > _max_projection_distance_cells(config):
            reject_reasons.append("projection_distance_cells_exceeds_contract")
        if training_use not in _allowed_training_uses(config):
            reject_reasons.append("source_training_use_not_trainable")
        if comparison_scope not in _allowed_training_scopes(config):
            reject_reasons.append("comparison_scope_not_trainable")
        if comparison_scope in _forbidden_positive_scopes(config):
            reject_reasons.append("audit_proxy_scope_not_positive_evidence")
        decision = (
            "trainable_anchor_projection_contrast"
            if not reject_reasons
            else "nontrainable_blocked_target"
        )
    return {
        "scenario_id": record.get("scenario_id"),
        "pair_key": record.get("pair_key"),
        "action_index": record.get("action_index"),
        "policy_target_cell": _cell(feasibility.get("policy_target_cell"))
        or _cell(feasibility.get("cell"))
        or _cell(record.get("cell")),
        "projected_anchor_cell": _cell(anchor),
        "platform_goal_classification": classification,
        "anchor_available": _cell(anchor) is not None,
        "anchor_reachable": anchor_reachable,
        "projection_distance_m": distance_m,
        "projection_distance_cells": distance_cells,
        "comparison_scope": comparison_scope,
        "same_cell_positive_evidence": bool(projection.get("same_cell_positive_evidence")),
        "source_training_use": training_use,
        "evidence_boundary": projection.get("evidence_boundary"),
        "contract_decision": decision,
        "sample_weight": 1.0 if decision == "trainable_anchor_projection_contrast" else 0.0,
        "training_use": decision if decision == "trainable_anchor_projection_contrast" else "not_positive_evidence",
        "reject_reasons": _unique(reject_reasons),
    }


def _contract_blockers(
    *,
    validation_reason_codes: list[str],
    safety_regression_count: int,
    fallback_count: int,
    contract_mutations: list[str],
    unresolved_count: int,
    max_unresolved_count: int,
    audit_proxy_positive_count: int,
    negative_scope_violation_count: int,
) -> list[str]:
    blockers: list[str] = []
    for reason in validation_reason_codes:
        _append_reason(blockers, reason)
    if safety_regression_count > 0:
        _append_reason(blockers, "safety_regression_blocks_anchor_projection_contract")
    if fallback_count > 0:
        _append_reason(blockers, "fallback_or_open_grid_blocks_anchor_projection_contract")
    if contract_mutations:
        _append_reason(blockers, "contract_mutation_blocks_anchor_projection_contract")
    if unresolved_count > max_unresolved_count:
        _append_reason(blockers, "anchor_projection_records_unresolved")
    if audit_proxy_positive_count > 0:
        _append_reason(blockers, "audit_proxy_anchor_positive_evidence_forbidden")
    if negative_scope_violation_count > 0:
        _append_reason(blockers, "negative_evidence_scope_violation")
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
        output_files.get("anchor_projection_evidence_contract_summary")
    ):
        raise ConfigError("output_files.anchor_projection_evidence_contract_summary must be a non-empty string")
    contract = payload.get("anchor_projection_contract", {})
    if not isinstance(contract, dict):
        raise ConfigError("anchor_projection_contract must be an object")
    config = dict(payload)
    config["anchor_projection_contract"] = {
        "max_projection_distance_m": _number_value(
            contract.get("max_projection_distance_m", 1.0),
            "anchor_projection_contract.max_projection_distance_m",
        ),
        "max_projection_distance_cells": _int_value(
            contract.get("max_projection_distance_cells", 2),
            "anchor_projection_contract.max_projection_distance_cells",
        ),
        "max_unresolved_count": _int_value(
            contract.get("max_unresolved_count", 0),
            "anchor_projection_contract.max_unresolved_count",
        ),
        "require_anchor_reachable": bool(contract.get("require_anchor_reachable", True)),
        "allowed_training_use_values": _string_set(
            contract.get("allowed_training_use_values"),
            default=("trainable_anchor_projection_contrast",),
        ),
        "allowed_comparison_scopes_for_training": _string_set(
            contract.get("allowed_comparison_scopes_for_training"),
            default=("projected_target_anchor_contrast",),
        ),
        "forbidden_positive_evidence_scopes": _string_set(
            contract.get("forbidden_positive_evidence_scopes"),
            default=("audit_proxy_anchor_not_same_cell",),
        ),
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


def _records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("regenerated_records")
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def _platform_goal_classification(record: dict[str, Any]) -> str | None:
    for key in ("platform_goal_classification", "failure_category", "failure_taxonomy"):
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


def _projection_scope(record: dict[str, Any]) -> str | None:
    feasibility = record.get("platform_goal_feasibility")
    feasibility = feasibility if isinstance(feasibility, dict) else {}
    projection = feasibility.get("anchor_projection")
    projection = projection if isinstance(projection, dict) else {}
    for payload in (projection, feasibility.get("proxy_route_comparison")):
        payload = payload if isinstance(payload, dict) else {}
        for key in ("comparison_scope", "scope"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _anchor_reachable(feasibility: dict[str, Any], projection: dict[str, Any]) -> bool:
    if isinstance(projection.get("anchor_reachable"), bool):
        return bool(projection["anchor_reachable"])
    proxy = feasibility.get("proxy_route_comparison")
    proxy = proxy if isinstance(proxy, dict) else {}
    return bool(proxy.get("anchor_route_feasible"))


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
    return batch_root / config["output_files"]["anchor_projection_evidence_contract_summary"]


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    contract = config.get("anchor_projection_contract", {})
    public_contract = dict(contract) if isinstance(contract, dict) else {}
    for key in (
        "allowed_training_use_values",
        "allowed_comparison_scopes_for_training",
        "forbidden_positive_evidence_scopes",
    ):
        if isinstance(public_contract.get(key), set):
            public_contract[key] = sorted(public_contract[key])
    return {
        "schema_version": config.get("schema_version"),
        "validation": dict(config.get("validation", {})) if isinstance(config.get("validation"), dict) else {},
        "anchor_projection_contract": public_contract,
        "output_files": dict(config.get("output_files", {})),
    }


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
    if code and code not in reason_codes:
        reason_codes.append(code)


def _unique(values: list[str]) -> list[str]:
    return sorted(set(values))


def _cell(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    if not all(isinstance(item, int) for item in value):
        return None
    return list(value)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _string_set(value: Any, *, default: tuple[str, ...]) -> set[str]:
    if value is None:
        return set(default)
    if not isinstance(value, list):
        raise ConfigError("string set config values must be arrays")
    return {str(item) for item in value if str(item)}


def _allowed_training_uses(config: dict[str, Any]) -> set[str]:
    return set(config["anchor_projection_contract"]["allowed_training_use_values"])


def _allowed_training_scopes(config: dict[str, Any]) -> set[str]:
    return set(config["anchor_projection_contract"]["allowed_comparison_scopes_for_training"])


def _forbidden_positive_scopes(config: dict[str, Any]) -> set[str]:
    return set(config["anchor_projection_contract"]["forbidden_positive_evidence_scopes"])


def _max_projection_distance_m(config: dict[str, Any]) -> float:
    return float(config["anchor_projection_contract"]["max_projection_distance_m"])


def _max_projection_distance_cells(config: dict[str, Any]) -> int:
    return int(config["anchor_projection_contract"]["max_projection_distance_cells"])


def _max_unresolved_count(config: dict[str, Any]) -> int:
    return int(config["anchor_projection_contract"]["max_unresolved_count"])


def _require_anchor_reachable(config: dict[str, Any]) -> bool:
    return bool(config["anchor_projection_contract"]["require_anchor_reachable"])


def _number_value(value: Any, label: str) -> float:
    number = _number_or_none(value)
    if number is None:
        raise ConfigError(f"{label} must be a finite number")
    return number


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _int_value(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{label} must be a non-negative integer")
    return value


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _int_value_or_default(value: Any, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


if __name__ == "__main__":
    raise SystemExit(main())
