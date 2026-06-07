from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot
from git_provenance import git_snapshots_match as _git_snapshots_match


CONFIG_SCHEMA_VERSION = "anchor-projection-candidate-generation-config/v1"
SUMMARY_SCHEMA_VERSION = "anchor-projection-candidate-generation-summary/v1"
RUN_INDEX_SCHEMA_VERSION = "path-feedback-batch-run-index/v1"
BATCH_EVALUATION_SCHEMA_VERSION = "path-feedback-batch-evaluation-summary/v1"
PATH_FEEDBACK_SCHEMA_VERSION = "path-feedback-summary/v1"
SUBMODULES = ("dev-platform-constraints", "model-explorer", "path-planner")
MISMATCH_CLASSES = {
    "platform_inflated_goal_blocked",
    "original_goal_blocked",
    "out_of_bounds",
    "unknown_contract_mismatch",
}


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate opt-in anchor projection candidate generation evidence."
    )
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    batch_root = _resolve_path(args.batch_root, repo_root)
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    summary = analyze_anchor_projection_candidate_generation(
        batch_root=batch_root,
        config=config,
        repo_root=repo_root,
    )
    output_file = batch_root / config["output_files"]["anchor_projection_candidate_generation_summary"]
    print(
        json.dumps(
            {
                "status": "config validated" if summary["status"] == "passed" else "validation failed",
                "batch_root": _display_path(batch_root, repo_root),
                "reason_codes": summary["reason_codes"],
                "current_git_provenance_mismatch_count": summary[
                    "current_git_provenance_mismatch_count"
                ],
                "git_provenance_mismatch_count": summary["git_provenance_mismatch_count"],
                "source_selected_candidate_changed_rate": summary[
                    "source_selected_candidate_changed_rate"
                ],
                "trainable_anchor_projection_count": summary["trainable_anchor_projection_count"],
                "anchor_projection_candidate_generation_summary": _display_path(
                    output_file,
                    repo_root,
                ),
            },
            ensure_ascii=False,
        )
    )
    if args.validate_only or args.dry_run:
        return 1 if summary["status"] == "failed" else 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 1 if summary["status"] == "failed" else 0


def analyze_anchor_projection_candidate_generation(
    *,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    run_index_path = batch_root / "batch-run-index.json"
    batch_summary_path = batch_root / "batch-evaluation-summary.json"
    run_index = _load_source(
        run_index_path,
        expected_schema=RUN_INDEX_SCHEMA_VERSION,
        label="batch_run_index",
        reason_codes=reason_codes,
    )
    batch_summary = _load_source(
        batch_summary_path,
        expected_schema=BATCH_EVALUATION_SCHEMA_VERSION,
        label="batch_evaluation_summary",
        reason_codes=reason_codes,
    )
    current_git = _git_snapshot(repo_root)
    current_git_mismatch = 0
    git_mismatch = 0
    if config["validation"]["require_current_git_match"]:
        stored_git = run_index.get("git") if isinstance(run_index.get("git"), dict) else {}
        if stored_git and not _git_snapshots_match(stored_git, current_git):
            current_git_mismatch = 1
            git_mismatch = 1
            _append_reason(reason_codes, "current_git_provenance_mismatch")
            _append_reason(reason_codes, "git_provenance_mismatch")

    if int(batch_summary.get("failed_count", 0) or 0) > 0:
        _append_reason(reason_codes, "batch_evaluation_failed")
    open_grid_count = int(batch_summary.get("open_grid_fallback_used_count", 0) or 0)
    fallback_or_open_grid_count = open_grid_count
    if config["validation"]["fail_on_fallback_or_open_grid"] and fallback_or_open_grid_count > 0:
        _append_reason(reason_codes, "fallback_or_open_grid_blocks_anchor_projection_candidate_generation")
    safety_regression_count = int(
        batch_summary.get("safety_regression_count", batch_summary.get("tracking_safety_violation_count", 0)) or 0
    )
    if config["validation"]["fail_on_safety_regression"] and safety_regression_count > 0:
        _append_reason(reason_codes, "safety_regression_blocks_anchor_projection_candidate_generation")

    path_feedback_sources = _path_feedback_summary_paths(run_index, batch_root=batch_root, repo_root=repo_root)
    contexts: dict[tuple[Any, ...], dict[str, Any]] = {}
    source_summary_records = []
    for path in path_feedback_sources:
        payload = _load_path_feedback_summary(path, reason_codes=reason_codes)
        if not payload:
            continue
        source_summary_records.append(
            {
                "path": _display_path(path, repo_root),
                "scenario_count": payload.get("scenario_count"),
                "open_grid_fallback_used": payload.get("open_grid_fallback_used"),
            }
        )
        _collect_contexts(payload, contexts=contexts, source_path=path, config=config)

    platform_goal_contract_mismatch_count = len(contexts)
    trainable_contexts = [context for context in contexts.values() if context["trainable"]]
    trainable_count = len(trainable_contexts)
    nontrainable_count = platform_goal_contract_mismatch_count - trainable_count
    source_changed_count = sum(1 for context in trainable_contexts if context["source_selected_candidate_changed"])
    source_changed_rate = _rate(source_changed_count, platform_goal_contract_mismatch_count)
    positive_audit_proxy_count = sum(1 for context in trainable_contexts if context["positive_audit_proxy"])
    source_selection_quality_regression_count = sum(
        1 for context in contexts.values() if context["source_selection_quality_regression"]
    )
    max_source_selection_path_margin = _max_context_value(
        contexts.values(),
        "source_selection_path_cost_margin_vs_best_alternative",
    )
    max_source_selection_risk_margin = _max_context_value(
        contexts.values(),
        "source_selection_risk_margin_vs_best_alternative",
    )
    anchor_available_count = sum(1 for context in contexts.values() if context["anchor_available"])
    unresolved_count = sum(1 for context in contexts.values() if context["classification"] == "unknown_contract_mismatch")
    reject_reason_counts = Counter(
        reason
        for context in contexts.values()
        for reason in context["reject_reasons"]
        if reason
    )
    coverage_diagnosis = _anchor_projection_coverage_diagnosis(contexts.values())
    nontrainable_reason_counts = coverage_diagnosis["nontrainable_primary_reason_counts"]
    reachable_substitute_anchor_found_count = int(
        coverage_diagnosis["reachable_substitute_anchor_found_count"]
    )
    anchor_unreachable_repaired_count = int(
        coverage_diagnosis["anchor_unreachable_repaired_by_reachable_substitute_count"]
    )
    true_geometry_unreachable_count = int(coverage_diagnosis["true_geometry_unreachable_count"])
    if trainable_count < config["thresholds"]["min_trainable_anchor_projection_count"]:
        _append_reason(reason_codes, "trainable_anchor_projection_count_below_threshold")
    if source_changed_rate < config["thresholds"]["min_source_selected_candidate_changed_rate"]:
        _append_reason(reason_codes, "source_selected_candidate_changed_rate_below_threshold")
    if positive_audit_proxy_count > 0:
        _append_reason(reason_codes, "positive_training_evidence_contains_audit_proxy_anchor")
    if (
        source_selection_quality_regression_count
        > config["thresholds"]["max_source_selection_quality_regression_count"]
    ):
        _append_reason(reason_codes, "source_selection_quality_regression_blocks_anchor_projection_candidate_generation")
    if trainable_count + nontrainable_count != platform_goal_contract_mismatch_count:
        _append_reason(reason_codes, "anchor_projection_context_accounting_mismatch")

    status = "failed" if reason_codes else "passed"
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "batch_root": _display_path(batch_root, repo_root),
        "source_summaries": source_summary_records,
        "git_provenance": {"current": current_git, "batch_run_index": run_index.get("git")},
        "current_git_provenance_mismatch_count": current_git_mismatch,
        "git_provenance_mismatch_count": git_mismatch,
        "open_grid_fallback_used_count": open_grid_count,
        "fallback_or_open_grid_count": fallback_or_open_grid_count,
        "safety_regression_count": safety_regression_count,
        "platform_goal_contract_mismatch_count": platform_goal_contract_mismatch_count,
        "platform_goal_anchor_available_count": anchor_available_count,
        "platform_goal_unresolved_count": unresolved_count,
        "trainable_anchor_projection_count": trainable_count,
        "nontrainable_blocked_target_count": nontrainable_count,
        "nontrainable_anchor_unreachable_count": int(
            nontrainable_reason_counts.get("anchor_unreachable", 0) or 0
        ),
        "nontrainable_source_candidate_not_selected_count": int(
            nontrainable_reason_counts.get("source_candidate_not_selected", 0) or 0
        ),
        "reachable_substitute_anchor_found_count": reachable_substitute_anchor_found_count,
        "anchor_unreachable_repaired_by_reachable_substitute_count": anchor_unreachable_repaired_count,
        "true_geometry_unreachable_count": true_geometry_unreachable_count,
        "source_selected_candidate_changed_count": source_changed_count,
        "source_selected_candidate_changed_rate": source_changed_rate,
        "source_selection_quality_regression_count": source_selection_quality_regression_count,
        "max_source_selection_path_cost_margin_vs_best_alternative": max_source_selection_path_margin,
        "max_source_selection_risk_margin_vs_best_alternative": max_source_selection_risk_margin,
        "positive_training_evidence_contains_audit_proxy_anchor_count": positive_audit_proxy_count,
        "audit_proxy_positive_count": positive_audit_proxy_count,
        "anchor_projection_candidate_reject_reason_counts": dict(sorted(reject_reason_counts.items())),
        "anchor_projection_coverage_diagnosis": coverage_diagnosis,
        "context_count": len(contexts),
        "context_records": list(contexts.values()),
        "audit_boundaries": {
            "audit_proxy_anchor_not_same_cell_is_positive_evidence": False,
            "does_not_train_ppo": True,
            "does_not_modify_network": True,
            "does_not_modify_action_space": True,
            "does_not_modify_default_astar": True,
            "does_not_claim_ackermann_feasible_trajectory": True,
        },
        "non_goals": list(config.get("non_goals", [])),
    }


def _collect_contexts(
    summary: dict[str, Any],
    *,
    contexts: dict[tuple[Any, ...], dict[str, Any]],
    source_path: Path,
    config: dict[str, Any],
) -> None:
    scenarios = summary.get("scenarios")
    if not isinstance(scenarios, list):
        return
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        scenario_id = scenario.get("scenario_id")
        scenario_group = scenario.get("scenario_group")
        before_cell = _cell_tuple(scenario.get("selected_cell_before_path_feedback"))
        after_cell = _cell_tuple(scenario.get("selected_cell_after_path_feedback"))
        feedback = scenario.get("path_feedback") if isinstance(scenario.get("path_feedback"), dict) else {}
        candidates = feedback.get("candidates")
        if not isinstance(candidates, list):
            continue
        selected_candidate = _candidate_for_cell(candidates, after_cell)
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            feasibility = candidate.get("platform_goal_feasibility")
            feasibility = feasibility if isinstance(feasibility, dict) else {}
            classification = str(feasibility.get("classification", "unavailable"))
            if classification not in MISMATCH_CLASSES:
                continue
            policy_target = _cell_tuple(
                feasibility.get("policy_target_cell") or candidate.get("policy_target_cell")
            )
            if policy_target is None:
                continue
            source_action_index = candidate.get("source_action_index")
            if source_action_index is None:
                source_action_index = candidate.get("action_index")
            key = (str(source_path), scenario_id, source_action_index, policy_target)
            context = contexts.setdefault(
                key,
                {
                    "source_path": str(source_path),
                    "run_id": source_path.parent.name,
                    "scenario_id": scenario_id,
                    "scenario_group": scenario_group,
                    "source_action_index": source_action_index,
                    "policy_target_cell": list(policy_target),
                    "execution_goal_cell": None,
                    "projected_anchor_cell": None,
                    "classification": classification,
                    "anchor_available": False,
                    "anchor_reachable": False,
                    "trainable": False,
                    "source_selected_candidate_changed": False,
                    "positive_audit_proxy": False,
                    "training_use": "not_positive_evidence",
                    "comparison_scope": None,
                    "reject_reasons": [],
                    "evidence_boundary": None,
                    "projected_candidate_generated": False,
                    "projected_candidate_source_selected": False,
                    "generated_action_index": None,
                    "selected_action_index": None,
                    "selected_candidate_role": None,
                    "selected_cell": None if after_cell is None else list(after_cell),
                    "projected_candidate_path_cost": None,
                    "projected_candidate_risk": None,
                    "projected_candidate_utility": None,
                    "selected_candidate_path_cost": None,
                    "selected_candidate_risk": None,
                    "selected_candidate_utility": None,
                    "path_cost_margin_vs_selected": None,
                    "risk_margin_vs_selected": None,
                    "source_selection_quality_regression": False,
                    "source_selection_best_alternative_action_index": None,
                    "source_selection_best_alternative_cell": None,
                    "source_selection_path_cost_margin_vs_best_alternative": None,
                    "source_selection_risk_margin_vs_best_alternative": None,
                    "projection_distance_cells": None,
                    "projection_distance_m": None,
                    "nearest_inflated_passable_anchor": None,
                    "nearest_anchor_reachable": None,
                    "nearest_anchor_distance_cells": None,
                    "nearest_anchor_distance_m": None,
                    "anchor_selection_status": None,
                    "start_component_id": None,
                    "target_component_id": None,
                    "nearest_anchor_component_id": None,
                    "projected_anchor_component_id": None,
                    "start_component_size": None,
                    "target_component_size": None,
                    "nearest_anchor_component_size": None,
                    "projected_anchor_component_size": None,
                    "reachable_substitute_anchor_available": False,
                    "reachable_substitute_anchor_count": 0,
                },
            )
            projection = feasibility.get("anchor_projection")
            projection = projection if isinstance(projection, dict) else {}
            _copy_anchor_reachability_fields(context, projection)
            anchor = _cell_tuple(
                projection.get("projected_anchor_cell")
                or projection.get("nearest_inflated_passable_anchor")
                or feasibility.get("nearest_inflated_passable_anchor")
            )
            if anchor is not None:
                context["anchor_available"] = True
                context["projected_anchor_cell"] = list(anchor)
            if projection.get("anchor_reachable") is True:
                context["anchor_reachable"] = True
            distance_cells = _float_optional(projection.get("projection_distance_cells"))
            if distance_cells is not None:
                context["projection_distance_cells"] = distance_cells
            distance_m = _float_optional(projection.get("projection_distance_m"))
            if distance_m is not None:
                context["projection_distance_m"] = distance_m
            reject_reason = projection.get("reject_reason")
            if reject_reason:
                context["reject_reasons"].append(str(reject_reason))
            generation = candidate.get("candidate_generation")
            generation = generation if isinstance(generation, dict) else {}
            if generation.get("candidate_role") != "projected_execution_target":
                continue
            context["projected_candidate_generated"] = True
            context["generated_action_index"] = candidate.get("action_index")
            context["projected_candidate_path_cost"] = _float_optional(candidate.get("path_cost"))
            context["projected_candidate_risk"] = _float_optional(candidate.get("risk"))
            context["projected_candidate_utility"] = _float_optional(candidate.get("utility"))
            generation_distance_cells = _float_optional(generation.get("projection_distance_cells"))
            if generation_distance_cells is not None:
                context["projection_distance_cells"] = generation_distance_cells
            generation_distance_m = _float_optional(generation.get("projection_distance_m"))
            if generation_distance_m is not None:
                context["projection_distance_m"] = generation_distance_m
            _copy_anchor_reachability_fields(context, generation)
            if selected_candidate is not None:
                context["selected_action_index"] = selected_candidate.get("action_index")
                context["selected_candidate_role"] = selected_candidate.get("candidate_role")
                context["selected_candidate_path_cost"] = _float_optional(selected_candidate.get("path_cost"))
                context["selected_candidate_risk"] = _float_optional(selected_candidate.get("risk"))
                context["selected_candidate_utility"] = _float_optional(selected_candidate.get("utility"))
                context["path_cost_margin_vs_selected"] = _delta(
                    context["projected_candidate_path_cost"],
                    context["selected_candidate_path_cost"],
                )
                context["risk_margin_vs_selected"] = _delta(
                    context["projected_candidate_risk"],
                    context["selected_candidate_risk"],
                )
            source_selection_status = generation.get("source_selection_status") or projection.get(
                "source_selection_status"
            )
            context["source_selection_quality_regression"] = bool(
                source_selection_status == "source_selected_quality_regression"
                or generation.get("reject_reason") == "source_selection_quality_regression"
                or projection.get("reject_reason") == "source_selection_quality_regression"
            )
            context["source_selection_best_alternative_action_index"] = generation.get(
                "source_selection_best_alternative_action_index",
                projection.get("source_selection_best_alternative_action_index"),
            )
            for field in (
                "source_selection_best_alternative_scope",
                "source_selection_best_alternative_candidate_role",
            ):
                value = generation.get(field, projection.get(field))
                if value:
                    context[field] = str(value)
            best_alternative_cell = _cell_tuple(
                generation.get("source_selection_best_alternative_cell")
                or projection.get("source_selection_best_alternative_cell")
            )
            context["source_selection_best_alternative_cell"] = (
                None if best_alternative_cell is None else list(best_alternative_cell)
            )
            path_margin_vs_best = _float_optional(
                generation.get("source_selection_path_cost_margin_vs_best_alternative")
                if generation.get("source_selection_path_cost_margin_vs_best_alternative") is not None
                else projection.get("source_selection_path_cost_margin_vs_best_alternative")
            )
            if path_margin_vs_best is not None:
                context["source_selection_path_cost_margin_vs_best_alternative"] = path_margin_vs_best
            risk_margin_vs_best = _float_optional(
                generation.get("source_selection_risk_margin_vs_best_alternative")
                if generation.get("source_selection_risk_margin_vs_best_alternative") is not None
                else projection.get("source_selection_risk_margin_vs_best_alternative")
            )
            if risk_margin_vs_best is not None:
                context["source_selection_risk_margin_vs_best_alternative"] = risk_margin_vs_best
            generation_reject_reason = generation.get("reject_reason")
            if generation_reject_reason:
                context["reject_reasons"].append(str(generation_reject_reason))
            training_use = generation.get("training_use") or projection.get("training_use")
            comparison_scope = generation.get("comparison_scope") or projection.get("comparison_scope")
            source_selected = generation.get("source_selection_status") == "source_selected"
            context["projected_candidate_source_selected"] = source_selected
            execution_goal = _cell_tuple(generation.get("execution_goal_cell"))
            distance_reject_reasons = _projection_distance_reject_reasons(context, config=config)
            for reason in distance_reject_reasons:
                if reason not in context["reject_reasons"]:
                    context["reject_reasons"].append(reason)
            trainable = (
                training_use == "trainable_anchor_projection_contrast"
                and source_selected
                and comparison_scope == "projected_target_anchor_contrast"
                and not distance_reject_reasons
            )
            context["execution_goal_cell"] = None if execution_goal is None else list(execution_goal)
            context["training_use"] = str(training_use)
            context["comparison_scope"] = str(comparison_scope)
            context["evidence_boundary"] = generation.get("evidence_boundary")
            if trainable:
                context["trainable"] = True
                context["source_selected_candidate_changed"] = bool(
                    before_cell is not None
                    and after_cell is not None
                    and before_cell != after_cell
                    and execution_goal == after_cell
                )
                context["positive_audit_proxy"] = comparison_scope == "audit_proxy_anchor_not_same_cell"


def _copy_anchor_reachability_fields(context: dict[str, Any], source: dict[str, Any]) -> None:
    nearest_anchor = _cell_tuple(source.get("nearest_inflated_passable_anchor"))
    if nearest_anchor is not None:
        context["nearest_inflated_passable_anchor"] = list(nearest_anchor)
    if "nearest_anchor_reachable" in source:
        context["nearest_anchor_reachable"] = bool(source.get("nearest_anchor_reachable"))
    status = source.get("anchor_selection_status")
    if status:
        context["anchor_selection_status"] = str(status)
    for field in ("nearest_anchor_distance_cells", "nearest_anchor_distance_m"):
        value = _float_optional(source.get(field))
        if value is not None:
            context[field] = value
    for field in (
        "start_component_id",
        "target_component_id",
        "nearest_anchor_component_id",
        "projected_anchor_component_id",
        "start_component_size",
        "target_component_size",
        "nearest_anchor_component_size",
        "projected_anchor_component_size",
        "reachable_substitute_anchor_count",
    ):
        value = _int_optional(source.get(field))
        if value is not None:
            context[field] = value
    if "reachable_substitute_anchor_available" in source:
        context["reachable_substitute_anchor_available"] = bool(
            source.get("reachable_substitute_anchor_available")
        )


def _projection_distance_reject_reasons(context: dict[str, Any], *, config: dict[str, Any]) -> list[str]:
    thresholds = config.get("thresholds") if isinstance(config.get("thresholds"), dict) else {}
    reasons: list[str] = []
    max_cells = thresholds.get("max_trainable_projection_distance_cells")
    distance_cells = _float_optional(context.get("projection_distance_cells"))
    if max_cells is not None and distance_cells is not None and distance_cells > float(max_cells):
        reasons.append("projection_distance_cells_exceeds_contract")
    max_m = thresholds.get("max_trainable_projection_distance_m")
    distance_m = _float_optional(context.get("projection_distance_m"))
    if max_m is not None and distance_m is not None and distance_m > float(max_m):
        reasons.append("projection_distance_m_exceeds_contract")
    return reasons


def _anchor_projection_coverage_diagnosis(context_values: Any) -> dict[str, Any]:
    contexts = list(context_values)
    trainable = [context for context in contexts if context["trainable"]]
    nontrainable = [context for context in contexts if not context["trainable"]]
    generated = [context for context in contexts if context["projected_candidate_generated"]]
    generated_source_selected = [
        context for context in generated if context["projected_candidate_source_selected"]
    ]
    generated_not_source_selected = [
        context for context in generated if not context["projected_candidate_source_selected"]
    ]
    anchor_unreachable_not_generated = [
        context
        for context in contexts
        if not context["projected_candidate_generated"] and not context["anchor_reachable"]
    ]
    generated_nontrainable = [
        context for context in generated if not context["trainable"]
    ]
    quality_regressions = [
        context for context in generated if context["source_selection_quality_regression"]
    ]
    primary_reason_counts = Counter(
        _primary_nontrainable_reason(context) for context in nontrainable
    )
    primary_reason_counts.pop(None, None)
    anchor_selection_status_counts = Counter(
        str(context.get("anchor_selection_status"))
        for context in contexts
        if context.get("anchor_selection_status")
    )
    reachable_substitute_anchor_found = [
        context
        for context in contexts
        if context.get("anchor_selection_status") == "reachable_substitute_anchor_found"
    ]
    true_geometry_unreachable = [
        context
        for context in contexts
        if context.get("anchor_selection_status") == "true_geometry_unreachable"
    ]
    repaired_by_reachable_substitute = [
        context
        for context in reachable_substitute_anchor_found
        if context.get("nearest_anchor_reachable") is False
        and context.get("anchor_reachable") is True
        and context.get("projected_candidate_generated") is True
    ]
    projection_distance_contract_rejected = [
        context
        for context in contexts
        if "projection_distance_cells_exceeds_contract" in context.get("reject_reasons", [])
        or "projection_distance_m_exceeds_contract" in context.get("reject_reasons", [])
    ]
    distance_cells = [
        context["projection_distance_cells"]
        for context in contexts
        if isinstance(context.get("projection_distance_cells"), int | float)
    ]
    distance_m = [
        context["projection_distance_m"]
        for context in contexts
        if isinstance(context.get("projection_distance_m"), int | float)
    ]
    margins = [
        context
        for context in generated_not_source_selected
        if context.get("path_cost_margin_vs_selected") is not None
    ]
    return {
        "schema_version": "anchor-projection-coverage-diagnosis/v1",
        "context_count": len(contexts),
        "trainable": len(trainable),
        "nontrainable": len(nontrainable),
        "projected_candidate_generated_count": len(generated),
        "projected_candidate_generated_rate": _rate(len(generated), len(contexts)),
        "projected_candidate_source_selected_count": len(generated_source_selected),
        "projected_candidate_not_source_selected_count": len(generated_not_source_selected),
        "generated_nontrainable_count": len(generated_nontrainable),
        "source_selection_quality_regression_count": len(quality_regressions),
        "anchor_unreachable_not_generated_count": len(anchor_unreachable_not_generated),
        "anchor_selection_status_counts": dict(sorted(anchor_selection_status_counts.items())),
        "reachable_substitute_anchor_found_count": len(reachable_substitute_anchor_found),
        "true_geometry_unreachable_count": len(true_geometry_unreachable),
        "anchor_unreachable_repaired_by_reachable_substitute_count": len(
            repaired_by_reachable_substitute
        ),
        "projection_distance_contract_rejected_count": len(projection_distance_contract_rejected),
        "nontrainable_primary_reason_counts": dict(sorted(primary_reason_counts.items())),
        "scenario_diagnosis_counts": dict(
            sorted(Counter("trainable" if context["trainable"] else "nontrainable" for context in contexts).items())
        ),
        "nontrainable_by_run_id": _nested_count(nontrainable, "run_id"),
        "nontrainable_by_scenario_id": _nested_count(nontrainable, "scenario_id"),
        "nontrainable_by_scenario_group": _nested_count(nontrainable, "scenario_group"),
        "generated_not_source_selected_by_scenario_id": _nested_count(
            generated_not_source_selected,
            "scenario_id",
        ),
        "projection_distance_cells": _numeric_summary(distance_cells),
        "projection_distance_m": _numeric_summary(distance_m),
        "source_selection_margin": _selection_margin_summary(margins),
        "source_selection_quality_regression_margin": _selection_quality_regression_margin_summary(
            quality_regressions
        ),
        "audit_proxy_positive_evidence_count": sum(
            1 for context in trainable if context["positive_audit_proxy"]
        ),
    }


def _primary_nontrainable_reason(context: dict[str, Any]) -> str | None:
    if context["trainable"]:
        return None
    if not context["projected_candidate_generated"]:
        if not context["anchor_reachable"]:
            return "anchor_unreachable"
        return "projected_candidate_not_generated"
    if not context["projected_candidate_source_selected"]:
        return "source_candidate_not_selected"
    if context["source_selection_quality_regression"]:
        return "source_selection_quality_regression"
    reasons = [reason for reason in context.get("reject_reasons", []) if reason]
    return str(reasons[0]) if reasons else "unknown_nontrainable_anchor_projection"


def _nested_count(contexts: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(str(context.get(field) or "unknown") for context in contexts).items()
        )
    )


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    clean = sorted(float(value) for value in values if isfinite(float(value)))
    if not clean:
        return {"count": 0, "min": None, "max": None, "mean": None, "bins": {}}
    return {
        "count": len(clean),
        "min": clean[0],
        "max": clean[-1],
        "mean": sum(clean) / len(clean),
        "bins": dict(sorted(Counter(str(int(value)) if value.is_integer() else str(value) for value in clean).items())),
    }


def _selection_margin_summary(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    path_cost_margins = [
        context["path_cost_margin_vs_selected"]
        for context in contexts
        if isinstance(context.get("path_cost_margin_vs_selected"), int | float)
    ]
    risk_margins = [
        context["risk_margin_vs_selected"]
        for context in contexts
        if isinstance(context.get("risk_margin_vs_selected"), int | float)
    ]
    path_summary = _numeric_summary(path_cost_margins)
    risk_summary = _numeric_summary(risk_margins)
    return {
        "count": len(path_cost_margins),
        "min_path_cost_margin": path_summary["min"],
        "max_path_cost_margin": path_summary["max"],
        "mean_path_cost_margin": path_summary["mean"],
        "min_risk_margin": risk_summary["min"],
        "max_risk_margin": risk_summary["max"],
        "mean_risk_margin": risk_summary["mean"],
    }


def _selection_quality_regression_margin_summary(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    path_cost_margins = [
        context["source_selection_path_cost_margin_vs_best_alternative"]
        for context in contexts
        if isinstance(context.get("source_selection_path_cost_margin_vs_best_alternative"), int | float)
    ]
    risk_margins = [
        context["source_selection_risk_margin_vs_best_alternative"]
        for context in contexts
        if isinstance(context.get("source_selection_risk_margin_vs_best_alternative"), int | float)
    ]
    path_summary = _numeric_summary(path_cost_margins)
    risk_summary = _numeric_summary(risk_margins)
    return {
        "count": len(contexts),
        "min_path_cost_margin_vs_best_alternative": path_summary["min"],
        "max_path_cost_margin_vs_best_alternative": path_summary["max"],
        "mean_path_cost_margin_vs_best_alternative": path_summary["mean"],
        "min_risk_margin_vs_best_alternative": risk_summary["min"],
        "max_risk_margin_vs_best_alternative": risk_summary["max"],
        "mean_risk_margin_vs_best_alternative": risk_summary["mean"],
    }


def _max_context_value(contexts: Any, field: str) -> float | None:
    values = [
        context.get(field)
        for context in contexts
        if isinstance(context, dict) and isinstance(context.get(field), int | float)
    ]
    if not values:
        return None
    return max(float(value) for value in values)


def _candidate_for_cell(candidates: list[Any], cell: tuple[int, int] | None) -> dict[str, Any] | None:
    if cell is None:
        return None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if _cell_tuple(candidate.get("cell")) == cell:
            return candidate
    return None


def _float_optional(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _int_optional(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _delta(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    return first - second


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    output_files = payload.get("output_files")
    if not isinstance(output_files, dict) or not isinstance(
        output_files.get("anchor_projection_candidate_generation_summary"),
        str,
    ):
        raise ConfigError("output_files.anchor_projection_candidate_generation_summary is required")
    validation = payload.get("validation") if isinstance(payload.get("validation"), dict) else {}
    thresholds = payload.get("thresholds") if isinstance(payload.get("thresholds"), dict) else {}
    config = dict(payload)
    config["validation"] = {
        "require_current_git_match": bool(validation.get("require_current_git_match", True)),
        "fail_on_fallback_or_open_grid": bool(validation.get("fail_on_fallback_or_open_grid", True)),
        "fail_on_safety_regression": bool(validation.get("fail_on_safety_regression", True)),
    }
    config["thresholds"] = {
        "min_source_selected_candidate_changed_rate": _float_value(
            thresholds.get("min_source_selected_candidate_changed_rate", 1.0e-9),
            default=1.0e-9,
        ),
        "min_trainable_anchor_projection_count": int(
            thresholds.get("min_trainable_anchor_projection_count", 1) or 1
        ),
        "max_source_selection_quality_regression_count": int(
            thresholds.get("max_source_selection_quality_regression_count", 0) or 0
        ),
        "max_trainable_projection_distance_m": _float_value(
            thresholds.get("max_trainable_projection_distance_m", 1.0),
            default=1.0,
        ),
        "max_trainable_projection_distance_cells": int(
            thresholds.get("max_trainable_projection_distance_cells", 2) or 2
        ),
    }
    if config["thresholds"]["max_source_selection_quality_regression_count"] < 0:
        raise ConfigError("thresholds.max_source_selection_quality_regression_count must be >= 0")
    if config["thresholds"]["max_trainable_projection_distance_m"] < 0.0:
        raise ConfigError("thresholds.max_trainable_projection_distance_m must be >= 0")
    if config["thresholds"]["max_trainable_projection_distance_cells"] < 0:
        raise ConfigError("thresholds.max_trainable_projection_distance_cells must be >= 0")
    return config


def _load_source(path: Path, *, expected_schema: str, label: str, reason_codes: list[str]) -> dict[str, Any]:
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
    if payload.get("schema_version") != expected_schema:
        _append_reason(reason_codes, f"{label}_schema_mismatch")
    return payload


def _load_path_feedback_summary(path: Path, *, reason_codes: list[str]) -> dict[str, Any]:
    payload = _load_source(
        path,
        expected_schema=PATH_FEEDBACK_SCHEMA_VERSION,
        label="path_feedback_summary",
        reason_codes=reason_codes,
    )
    if payload.get("open_grid_fallback_used") is True:
        _append_reason(reason_codes, "path_feedback_summary_open_grid_fallback_used")
    return payload


def _path_feedback_summary_paths(run_index: dict[str, Any], *, batch_root: Path, repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    runs = run_index.get("runs")
    if isinstance(runs, list):
        for run in runs:
            if not isinstance(run, dict):
                continue
            source_paths = run.get("source_paths") if isinstance(run.get("source_paths"), dict) else {}
            value = source_paths.get("summary") or run.get("summary_path")
            if isinstance(value, str) and value:
                paths.append(_resolve_path(value, repo_root))
    if not paths:
        paths.extend(sorted(batch_root.glob("*/path-feedback-summary.json")))
    return paths


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _cell_tuple(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    try:
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return None


def _float_value(value: Any, *, default: float) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else float(numerator) / float(denominator)


def _append_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


if __name__ == "__main__":
    raise SystemExit(main())
