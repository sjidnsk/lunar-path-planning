from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot
from run_controlled_hybrid_policy_training_candidate import (
    CHECKPOINT_METADATA_SCHEMA_VERSION,
)
from run_scenario_disjoint_policy_rollout_evaluation import (
    _candidate_action_mask_valid,
    _collect_holdout_scenarios,
    _display_path,
    _fallback_or_open_grid_count,
    _float_or_none,
    _int_value,
    _regression_reasons,
    _score_scenarios,
    _same_candidate,
    _source_selected_candidate,
    _string_list,
    _summary_git_matches_current,
)


CONFIG_SCHEMA_VERSION = "policy-gated-canary-rollout-config/v1"
SUMMARY_SCHEMA_VERSION = "policy-gated-canary-rollout-summary/v1"
DECISION_SCHEMA_VERSION = "policy-gated-canary-decision/v1"
REJECTION_REPORT_SCHEMA_VERSION = "policy-gated-canary-rejection-report/v1"
OPPORTUNITY_SUMMARY_SCHEMA_VERSION = "policy-gated-canary-opportunity-summary/v1"

NEXT_NO_OPPORTUNITY = "canary_scenario_or_candidate_generation_required"
NEXT_TOO_CONSERVATIVE = "policy_objective_too_conservative_requires_canary_alignment_refinement"
NEXT_GATE_FAILED = "policy_candidate_fails_canary_gate_requires_objective_or_feature_refinement"
NEXT_CONTROLLED_GATE_FAILED = "policy_canary_controlled_gate_regression_requires_refinement"
NEXT_PROVENANCE_REFRESH = "clean_head_evidence_refresh_required"
NEXT_OPPORTUNITY_INSUFFICIENT = "canary_opportunity_insufficient"
NEXT_ACCEPTANCE_INSUFFICIENT = "policy_safe_alternative_acceptance_insufficient"
NEXT_FAMILY_COVERAGE_INSUFFICIENT = "scenario_family_coverage_insufficient"
NEXT_POLICY_GATE_REGRESSION = "policy_gate_regression_detected"
NEXT_OPPORTUNITY_GENERATION_GAP = "canary_opportunity_generation_gap"
NEXT_DENSE_CHOKE_OPPORTUNITY_GAP = "dense_choke_opportunity_generation_gap"
NEXT_SAFE_CHOICE_ALIGNMENT = "policy_safe_choice_alignment_insufficient"
NEXT_VALUE_OPPORTUNITY_GAP = "canary_value_opportunity_generation_gap"
NEXT_VALUE_ALIGNMENT = "policy_value_alignment_or_objective_refinement_required"
DENSE_CHOKE_FAMILY = "dense_choke_safe_bypass"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run policy-gated canary rollout evaluation.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    _install_model_explorer_path(repo_root)
    source_root = _resolve_path(args.source_root, repo_root)
    candidate_root = _resolve_path(args.candidate_root, repo_root)
    batch_root = _resolve_path(args.batch_root, repo_root)
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    output_paths = _output_paths(batch_root, config)
    summary, decisions, rejection_report, opportunity_summary = run_policy_gated_canary_rollout(
        source_root=source_root,
        candidate_root=candidate_root,
        batch_root=batch_root,
        config=config,
        repo_root=repo_root,
        output_paths=output_paths,
    )
    print(
        json.dumps(
            {
                "status": "config validated" if summary["status"] == "passed" else "validation failed",
                "source_root": _display_path(source_root, repo_root),
                "candidate_root": _display_path(candidate_root, repo_root),
                "batch_root": _display_path(batch_root, repo_root),
                "reason_codes": summary["reason_codes"],
                "policy_decision_count": summary["policy_decision_count"],
                "canary_opportunity_context_count": summary["canary_opportunity_context_count"],
                "canary_accepted_policy_choice_count": summary[
                    "canary_accepted_policy_choice_count"
                ],
                "summary": _display_path(output_paths["summary"], repo_root),
            },
            ensure_ascii=False,
        )
    )
    if not args.validate_only:
        batch_root.mkdir(parents=True, exist_ok=True)
        _write_json(output_paths["summary"], summary)
        _write_json(output_paths["rejection_report"], rejection_report)
        _write_json(output_paths["opportunity_summary"], opportunity_summary)
        output_paths["decisions"].write_text(
            "".join(json.dumps(decision, ensure_ascii=False) + "\n" for decision in decisions),
            encoding="utf-8",
        )
    return 1 if summary["status"] == "failed" else 0


def run_policy_gated_canary_rollout(
    *,
    source_root: Path,
    candidate_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    output_paths: dict[str, Path],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    reason_codes: list[str] = []
    paths = _input_paths(source_root, candidate_root, batch_root, config)
    source_batch = _load_summary(
        paths["source_batch_summary"],
        expected_schema=None,
        label="source_batch_summary",
        reason_codes=reason_codes,
    )
    holdout_batch = _load_summary(
        paths["holdout_batch_summary"],
        expected_schema=None,
        label="holdout_batch_summary",
        reason_codes=reason_codes,
    )
    candidate = _load_summary(
        paths["candidate_summary"],
        expected_schema=None,
        label="candidate_summary",
        reason_codes=reason_codes,
    )
    metadata = _load_summary(
        paths["checkpoint_metadata"],
        expected_schema=CHECKPOINT_METADATA_SCHEMA_VERSION,
        label="checkpoint_metadata",
        reason_codes=reason_codes,
    )

    if source_batch.get("failed_count", 0) or _string_list(source_batch.get("reason_codes")):
        _append_reason(reason_codes, "source_batch_failed")
    if holdout_batch.get("failed_count", 0) or _string_list(holdout_batch.get("reason_codes")):
        _append_reason(reason_codes, "canary_batch_failed")
    if candidate.get("status") != "passed" or candidate.get("candidate_training_status") != "passed":
        _append_reason(reason_codes, "candidate_summary_failed")
    if _string_list(candidate.get("reason_codes")):
        _append_reason(reason_codes, "candidate_summary_failed")
    if candidate.get("schema_version") != "raw-policy-generalization-candidate-summary/v1":
        _append_reason(reason_codes, "candidate_summary_schema_version_mismatch")
    if not metadata.get("experimental", False):
        _append_reason(reason_codes, "checkpoint_metadata_not_experimental")
    if metadata.get("publishes_checkpoint") or candidate.get("publishes_checkpoint"):
        _append_reason(reason_codes, "checkpoint_publication_detected")
    if metadata.get("replaces_default_policy") or candidate.get("replaces_default_policy"):
        _append_reason(reason_codes, "default_policy_replacement_detected")
    if metadata.get("performance_claimed") or candidate.get("performance_claimed"):
        _append_reason(reason_codes, "performance_claim_detected")
    if not paths["checkpoint"].is_file():
        _append_reason(reason_codes, "experimental_checkpoint_missing")

    current_git = _git_snapshot(repo_root)
    allow_dirty_match = bool(config["validation"].get("allow_dirty_current_git_match"))
    candidate_git_current_matches_sources = _summary_git_matches_current(
        candidate,
        current_git,
        allow_dirty_match=allow_dirty_match,
    )
    checkpoint_metadata_git_current_matches_sources = _summary_git_matches_current(
        metadata,
        current_git,
        allow_dirty_match=allow_dirty_match,
    )
    if config["validation"].get("require_candidate_git_current_match"):
        if not candidate_git_current_matches_sources:
            _append_reason(reason_codes, "candidate_git_current_mismatch")
        if not checkpoint_metadata_git_current_matches_sources:
            _append_reason(reason_codes, "checkpoint_metadata_git_current_mismatch")

    scenario_groups = _collect_holdout_scenarios(batch_root, repo_root)
    context_records = [record for group in scenario_groups for record in group["candidates"]]
    context_id_missing_count = sum(1 for record in context_records if not record.get("context_id"))
    if config["validation"].get("require_context_id") and context_id_missing_count:
        _append_reason(reason_codes, "context_id_missing")

    base_config = {
        "evaluation": dict(config["evaluation"]),
        "validation": dict(config["validation"]),
    }
    opportunities = _canary_opportunities(scenario_groups, config=base_config)

    scoring_error = None
    raw_decisions: list[dict[str, Any]] = []
    if paths["checkpoint"].is_file() and not any(
        reason
        in {
            "checkpoint_metadata_not_experimental",
            "checkpoint_publication_detected",
            "default_policy_replacement_detected",
            "performance_claim_detected",
            "experimental_checkpoint_missing",
        }
        for reason in reason_codes
    ):
        try:
            raw_decisions = _score_scenarios(
                checkpoint_path=paths["checkpoint"],
                scenario_groups=scenario_groups,
                config=base_config,
                repo_root=repo_root,
            )
        except Exception as exc:  # noqa: BLE001
            _append_reason(reason_codes, "policy_canary_scoring_failed")
            scoring_error = str(exc)

    decisions = [_canary_decision_record(decision, config=base_config) for decision in raw_decisions]
    class_counts = Counter(decision["decision_class"] for decision in decisions)
    value_class_counts = Counter(
        decision.get("accepted_choice_value_class")
        for decision in decisions
        if decision.get("decision_class") == "canary_accepted_policy_choice"
    )
    rejection_reason_counts: Counter[str] = Counter()
    for decision in decisions:
        if decision["decision_class"] == "canary_rejected_policy_choice":
            rejection_reason_counts.update(decision.get("canary_rejection_reason_codes", []))
    scenario_family_summary = _scenario_family_summary(decisions, opportunities)
    accepted_decision_family_distribution = {
        family: metrics["canary_accepted_policy_choice_count"]
        for family, metrics in scenario_family_summary.items()
        if metrics["canary_accepted_policy_choice_count"] > 0
    }
    scenario_family_count = len(scenario_family_summary)
    accepted_scenario_family_count = len(accepted_decision_family_distribution)
    families_with_acceptable_alternative = sorted(
        family
        for family, metrics in scenario_family_summary.items()
        if metrics["acceptable_alternative_count"] > 0
    )
    missing_acceptable_alternative_families = sorted(
        family
        for family, metrics in scenario_family_summary.items()
        if metrics["canary_opportunity_context_count"] > 0
        and metrics["acceptable_alternative_count"] <= 0
    )
    opportunity_by_scenario = {
        (opportunity.get("run_id"), opportunity.get("scenario_id")): opportunity
        for opportunity in opportunities
    }
    source_aligned_with_acceptable = [
        decision
        for decision in decisions
        if decision.get("decision_class") == "source_aligned"
        and int(
            opportunity_by_scenario.get(
                (decision.get("run_id"), decision.get("scenario_id")),
                {},
            ).get("acceptable_alternative_count", 0)
        )
        > 0
    ]
    source_aligned_with_acceptable_alternative_count = len(source_aligned_with_acceptable)
    missed_safe_choice_families = sorted(
        {
            str(decision.get("scenario_group") or "unknown")
            for decision in source_aligned_with_acceptable
        }
    )
    missed_safe_choice_family_count = len(missed_safe_choice_families)
    dense_choke_family_metrics = scenario_family_summary.get(DENSE_CHOKE_FAMILY, {})
    dense_choke_acceptable_alternative_count = int(
        dense_choke_family_metrics.get("acceptable_alternative_count", 0)
    )
    dense_choke_accepted_policy_choice_count = int(
        dense_choke_family_metrics.get("canary_accepted_policy_choice_count", 0)
    )
    accepted_better_family_count = sum(
        1
        for metrics in scenario_family_summary.values()
        if int(metrics.get("accepted_better_choice_count", 0)) > 0
    )
    accepted_value_delta_summary = _value_delta_summary(
        [
            decision
            for decision in decisions
            if decision.get("decision_class") == "canary_accepted_policy_choice"
        ]
    )

    controlled_regression_count = sum(
        1 for decision in decisions if decision.get("controlled_decision_class") == "regression"
    )
    invalid_action_mask_count = sum(
        1 for decision in decisions if not decision.get("action_mask_valid")
    )
    fallback_or_open_grid_count = max(
        _fallback_or_open_grid_count(holdout_batch),
        sum(
            1
            for decision in decisions
            if "fallback_or_open_grid" in decision.get("controlled_regression_reason_codes", [])
        ),
    )
    safety_regression_count = max(
        _int_value(holdout_batch.get("safety_regression_count")),
        sum(
            1
            for decision in decisions
            if "safety_regression" in decision.get("controlled_regression_reason_codes", [])
        ),
    )
    contract_violation_count = max(
        _int_value(holdout_batch.get("candidate_contract_alignment_gap_count")),
        _int_value(holdout_batch.get("contract_violation_count")),
        sum(
            1
            for decision in decisions
            if "contract_violation" in decision.get("controlled_regression_reason_codes", [])
        ),
    )
    path_cost_regression_count = sum(
        1
        for decision in decisions
        if "path_cost_regression" in decision.get("controlled_regression_reason_codes", [])
    )
    risk_regression_count = sum(
        1
        for decision in decisions
        if "risk_regression" in decision.get("controlled_regression_reason_codes", [])
    )
    source_selection_regression_count = sum(
        1
        for decision in decisions
        if "source_selection_regression" in decision.get("controlled_regression_reason_codes", [])
    )
    raw_policy_regression_count = sum(
        1 for decision in decisions if decision.get("raw_policy_decision_class") == "regression"
    )

    validation = config["validation"]
    if len(decisions) < _int_value(validation.get("min_policy_decision_count")):
        _append_reason(reason_codes, "policy_decision_count_below_threshold")
    if len(opportunities) < _int_value(validation.get("min_canary_opportunity_context_count")):
        _append_reason(reason_codes, "canary_opportunity_context_count_below_threshold")
    if class_counts.get("canary_accepted_policy_choice", 0) + class_counts.get(
        "canary_rejected_policy_choice",
        0,
    ) < _int_value(validation.get("min_policy_changed_decision_count")):
        _append_reason(reason_codes, "policy_changed_decision_count_below_threshold")
    if class_counts.get("canary_accepted_policy_choice", 0) < _int_value(
        validation.get("min_canary_accepted_policy_choice_count")
    ):
        _append_reason(reason_codes, "canary_accepted_policy_choice_count_below_threshold")
    min_scenario_family_count = _int_value(validation.get("min_scenario_family_count"))
    min_accepted_scenario_family_count = _int_value(
        validation.get("min_accepted_scenario_family_count")
    )
    if scenario_family_count < min_scenario_family_count:
        _append_reason(reason_codes, "scenario_family_count_below_threshold")
    if accepted_scenario_family_count < min_accepted_scenario_family_count:
        _append_reason(reason_codes, "accepted_scenario_family_count_below_threshold")
    min_family_with_acceptable_alternative_count = _int_value(
        validation.get("min_family_with_acceptable_alternative_count")
    )
    if (
        len(families_with_acceptable_alternative)
        < min_family_with_acceptable_alternative_count
    ):
        _append_reason(reason_codes, "family_with_acceptable_alternative_count_below_threshold")
    if controlled_regression_count > _int_value(validation.get("max_controlled_regression_count")):
        _append_reason(reason_codes, "controlled_regression")
    if invalid_action_mask_count > _int_value(validation.get("max_invalid_action_mask_count")):
        _append_reason(reason_codes, "invalid_action_mask")
    if fallback_or_open_grid_count > _int_value(validation.get("max_fallback_or_open_grid_count")):
        _append_reason(reason_codes, "fallback_or_open_grid")
    if safety_regression_count > _int_value(validation.get("max_safety_regression_count")):
        _append_reason(reason_codes, "safety_regression")
    if contract_violation_count > _int_value(validation.get("max_contract_violation_count")):
        _append_reason(reason_codes, "contract_violation")
    if path_cost_regression_count > _int_value(validation.get("max_path_cost_regression_count")):
        _append_reason(reason_codes, "path_cost_regression")
    if risk_regression_count > _int_value(validation.get("max_risk_regression_count")):
        _append_reason(reason_codes, "risk_regression")
    if source_selection_regression_count > _int_value(
        validation.get("max_source_selection_regression_count")
    ):
        _append_reason(reason_codes, "source_selection_regression")
    if (
        "max_raw_policy_regression_count" in validation
        and raw_policy_regression_count > _int_value(validation.get("max_raw_policy_regression_count"))
    ):
        _append_reason(reason_codes, "raw_policy_regression")
    min_accepted_better_choice_count = _int_value(
        validation.get("min_accepted_better_choice_count")
    )
    min_accepted_better_family_count = _int_value(
        validation.get("min_accepted_better_family_count")
    )
    if value_class_counts.get("accepted_better", 0) < min_accepted_better_choice_count:
        _append_reason(reason_codes, "accepted_better_choice_count_below_threshold")
    if accepted_better_family_count < min_accepted_better_family_count:
        _append_reason(reason_codes, "accepted_better_family_count_below_threshold")

    status = "failed" if reason_codes else "passed"
    diversity_requested = min_scenario_family_count > 0 or min_accepted_scenario_family_count > 0
    opportunity_quality_requested = min_family_with_acceptable_alternative_count > 0
    canary_diversity_passed = (
        status == "passed"
        and diversity_requested
        and scenario_family_count >= min_scenario_family_count
        and accepted_scenario_family_count >= min_accepted_scenario_family_count
    )
    canary_opportunity_quality_passed = (
        canary_diversity_passed
        and opportunity_quality_requested
        and len(families_with_acceptable_alternative)
        >= min_family_with_acceptable_alternative_count
    )
    full_family_opportunity_requested = (
        min_scenario_family_count >= 6
        and min_family_with_acceptable_alternative_count >= 6
        and min_accepted_scenario_family_count >= 6
    )
    canary_full_family_opportunity_passed = (
        canary_opportunity_quality_passed
        and full_family_opportunity_requested
        and dense_choke_acceptable_alternative_count > 0
        and dense_choke_accepted_policy_choice_count > 0
    )
    value_stability_requested = bool(config["evaluation"].get("value_stability_evaluation")) or (
        min_accepted_better_choice_count > 0 or min_accepted_better_family_count > 0
    )
    canary_value_stability_passed = (
        status == "passed"
        and value_stability_requested
        and canary_opportunity_quality_passed
        and value_class_counts.get("accepted_better", 0) >= min_accepted_better_choice_count
        and accepted_better_family_count >= min_accepted_better_family_count
    )
    next_required_change = _next_required_change(
        reason_codes=reason_codes,
        opportunity_count=len(opportunities),
        changed_count=class_counts.get("canary_accepted_policy_choice", 0)
        + class_counts.get("canary_rejected_policy_choice", 0),
        accepted_count=class_counts.get("canary_accepted_policy_choice", 0),
        rejected_count=class_counts.get("canary_rejected_policy_choice", 0),
        controlled_regression_count=controlled_regression_count,
        family_with_acceptable_alternative_count=len(families_with_acceptable_alternative),
        missing_acceptable_alternative_families=missing_acceptable_alternative_families,
        missed_safe_choice_count=source_aligned_with_acceptable_alternative_count,
        value_stability_requested=value_stability_requested,
    )
    policy_changed_decision_count = class_counts.get("canary_accepted_policy_choice", 0) + class_counts.get(
        "canary_rejected_policy_choice",
        0,
    )
    accepted_count = class_counts.get("canary_accepted_policy_choice", 0)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "source_root": _display_path(source_root, repo_root),
        "candidate_root": _display_path(candidate_root, repo_root),
        "batch_root": _display_path(batch_root, repo_root),
        "summary": _display_path(output_paths["summary"], repo_root),
        "decisions_path": _display_path(output_paths["decisions"], repo_root),
        "rejection_report_path": _display_path(output_paths["rejection_report"], repo_root),
        "opportunity_summary_path": _display_path(output_paths["opportunity_summary"], repo_root),
        "policy_decision_count": len(decisions),
        "canary_opportunity_context_count": len(opportunities),
        "policy_changed_decision_count": policy_changed_decision_count,
        "policy_change_rate": (
            policy_changed_decision_count / len(decisions) if decisions else 0.0
        ),
        "source_aligned_count": class_counts.get("source_aligned", 0),
        "canary_accepted_policy_choice_count": class_counts.get(
            "canary_accepted_policy_choice",
            0,
        ),
        "accepted_choice_rate": accepted_count / len(decisions) if decisions else 0.0,
        "canary_rejected_policy_choice_count": class_counts.get(
            "canary_rejected_policy_choice",
            0,
        ),
        "accepted_equal_choice_count": value_class_counts.get("accepted_equal", 0),
        "accepted_better_choice_count": value_class_counts.get("accepted_better", 0),
        "accepted_better_family_count": accepted_better_family_count,
        "accepted_value_delta_summary": accepted_value_delta_summary,
        "family_value_stability_summary": {
            family: {
                "accepted_equal_choice_count": metrics["accepted_equal_choice_count"],
                "accepted_better_choice_count": metrics["accepted_better_choice_count"],
                "policy_change_rate": (
                    metrics["policy_changed_decision_count"] / metrics["policy_decision_count"]
                    if metrics["policy_decision_count"]
                    else 0.0
                ),
                "accepted_choice_rate": (
                    metrics["canary_accepted_policy_choice_count"]
                    / metrics["policy_decision_count"]
                    if metrics["policy_decision_count"]
                    else 0.0
                ),
            }
            for family, metrics in scenario_family_summary.items()
        },
        "canary_rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "scenario_family_count": scenario_family_count,
        "accepted_scenario_family_count": accepted_scenario_family_count,
        "accepted_decision_family_distribution": dict(
            sorted(accepted_decision_family_distribution.items())
        ),
        "family_with_acceptable_alternative_count": len(
            families_with_acceptable_alternative
        ),
        "families_with_acceptable_alternative": families_with_acceptable_alternative,
        "dense_choke_acceptable_alternative_count": dense_choke_acceptable_alternative_count,
        "dense_choke_accepted_policy_choice_count": dense_choke_accepted_policy_choice_count,
        "missing_acceptable_alternative_family_count": len(
            missing_acceptable_alternative_families
        ),
        "missing_acceptable_alternative_families": missing_acceptable_alternative_families,
        "source_aligned_with_acceptable_alternative_count": (
            source_aligned_with_acceptable_alternative_count
        ),
        "canary_missed_opportunity_preference_pair_count": (
            source_aligned_with_acceptable_alternative_count
        ),
        "missed_safe_choice_family_count": missed_safe_choice_family_count,
        "missed_safe_choice_families": missed_safe_choice_families,
        "hard_positive_added_count": 0,
        "scenario_family_summary": scenario_family_summary,
        "canary_diversity_passed": canary_diversity_passed,
        "canary_opportunity_quality_passed": canary_opportunity_quality_passed,
        "canary_full_family_opportunity_passed": canary_full_family_opportunity_passed,
        "canary_value_stability_passed": canary_value_stability_passed,
        "controlled_regression_count": controlled_regression_count,
        "raw_policy_regression_count": raw_policy_regression_count,
        "invalid_action_mask_count": invalid_action_mask_count,
        "fallback_or_open_grid_count": fallback_or_open_grid_count,
        "safety_regression_count": safety_regression_count,
        "contract_violation_count": contract_violation_count,
        "path_cost_regression_count": path_cost_regression_count,
        "risk_regression_count": risk_regression_count,
        "source_selection_regression_count": source_selection_regression_count,
        "context_id_missing_count": context_id_missing_count,
        "candidate_git_current_matches_sources": candidate_git_current_matches_sources,
        "checkpoint_metadata_git_current_matches_sources": checkpoint_metadata_git_current_matches_sources,
        "scoring_error": scoring_error,
        "next_required_change": next_required_change,
        "experimental_checkpoint": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": current_git, "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }
    rejection_report = {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "status": status,
        "reason_codes": reason_codes,
        "canary_rejected_policy_choice_count": summary["canary_rejected_policy_choice_count"],
        "canary_rejection_reason_counts": summary["canary_rejection_reason_counts"],
        "rejected_decisions": [
            decision
            for decision in decisions
            if decision.get("decision_class") == "canary_rejected_policy_choice"
        ],
    }
    opportunity_summary = {
        "schema_version": OPPORTUNITY_SUMMARY_SCHEMA_VERSION,
        "status": status,
        "reason_codes": reason_codes,
        "canary_opportunity_context_count": len(opportunities),
        "opportunities": opportunities[:50],
        "scenario_family_summary": scenario_family_summary,
        "family_with_acceptable_alternative_count": len(
            families_with_acceptable_alternative
        ),
        "dense_choke_acceptable_alternative_count": dense_choke_acceptable_alternative_count,
        "dense_choke_accepted_policy_choice_count": dense_choke_accepted_policy_choice_count,
        "missing_acceptable_alternative_families": missing_acceptable_alternative_families,
        "source_aligned_with_acceptable_alternative_count": (
            source_aligned_with_acceptable_alternative_count
        ),
        "canary_missed_opportunity_preference_pair_count": (
            source_aligned_with_acceptable_alternative_count
        ),
        "missed_safe_choice_families": missed_safe_choice_families,
        "hard_positive_added_count": 0,
        "next_required_change": NEXT_NO_OPPORTUNITY if not opportunities else None,
    }
    return summary, decisions, rejection_report, opportunity_summary


def _canary_opportunities(
    scenario_groups: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    opportunities: list[dict[str, Any]] = []
    for group in scenario_groups:
        source = _source_selected_candidate(group)
        if not source:
            continue
        alternatives = []
        for candidate in group.get("candidates", []):
            if _same_candidate(candidate, source):
                continue
            regression_reasons = _regression_reasons(candidate, source, config=config)
            alternatives.append(
                {
                    "context_id": candidate.get("context_id"),
                    "source_action_index": candidate.get("source_action_index"),
                    "action_index": candidate.get("action_index"),
                    "policy_target_cell": candidate.get("policy_target_cell"),
                    "path_cost_delta": _delta(candidate.get("path_cost"), source.get("path_cost")),
                    "risk_delta": _delta(candidate.get("risk"), source.get("risk")),
                    "utility_delta": _delta(candidate.get("utility"), source.get("utility")),
                    "canary_gate_acceptable": not regression_reasons,
                    "canary_gate_rejection_reason_codes": list(regression_reasons),
                }
            )
        if alternatives:
            opportunities.append(
                {
                    "run_id": group.get("run_id"),
                    "scenario_id": group.get("scenario_id"),
                    "scenario_group": group.get("scenario_group"),
                    "source_context_id": source.get("context_id"),
                    "alternative_count": len(alternatives),
                    "acceptable_alternative_count": sum(
                        1 for alternative in alternatives if alternative["canary_gate_acceptable"]
                    ),
                    "alternatives": alternatives,
                }
            )
    return opportunities


def _canary_decision_record(
    decision: dict[str, Any],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    raw_reasons = list(decision.get("raw_policy_regression_reason_codes", []))
    raw_changed = (
        decision.get("raw_policy_selected_context_id") != decision.get("source_selected_context_id")
        if decision.get("raw_policy_selected_context_id") and decision.get("source_selected_context_id")
        else decision.get("raw_policy_selected_action_index")
        != decision.get("source_selected_action_index")
    )
    if not raw_changed:
        decision_class = "source_aligned"
        rejection_reasons: list[str] = []
    elif raw_reasons:
        decision_class = "canary_rejected_policy_choice"
        rejection_reasons = raw_reasons
    else:
        decision_class = "canary_accepted_policy_choice"
        rejection_reasons = []
    record = dict(decision)
    record["schema_version"] = DECISION_SCHEMA_VERSION
    record["controlled_decision_class"] = decision.get("decision_class")
    record["controlled_regression_reason_codes"] = list(decision.get("regression_reason_codes", []))
    record["decision_class"] = decision_class
    record["canary_rejection_reason_codes"] = rejection_reasons
    record["canary_gate_passed"] = decision_class == "canary_accepted_policy_choice"
    record["accepted_choice_value_class"] = _accepted_choice_value_class(
        record,
        config=config,
    )
    record["publishes_checkpoint"] = False
    record["replaces_default_policy"] = False
    record["performance_claimed"] = False
    return record


def _accepted_choice_value_class(
    decision: dict[str, Any],
    *,
    config: dict[str, Any],
) -> str | None:
    if decision.get("decision_class") != "canary_accepted_policy_choice":
        return None
    blocked_reasons = set(decision.get("controlled_regression_reason_codes", [])) | set(
        decision.get("canary_rejection_reason_codes", [])
    )
    if blocked_reasons:
        return None
    evaluation = config.get("evaluation", {})
    path_threshold = float(evaluation.get("min_better_path_cost_delta", 0.25))
    risk_threshold = float(evaluation.get("min_better_risk_delta", 0.01))
    utility_threshold = float(evaluation.get("min_better_utility_delta", 0.005))
    path_delta = _float_or_none(decision.get("policy_selected_path_cost_delta"))
    risk_delta = _float_or_none(decision.get("policy_selected_risk_delta"))
    utility_delta = _float_or_none(decision.get("policy_selected_utility_delta"))
    better = (
        (path_delta is not None and path_delta <= -path_threshold)
        or (risk_delta is not None and risk_delta <= -risk_threshold)
        or (utility_delta is not None and utility_delta >= utility_threshold)
    )
    return "accepted_better" if better else "accepted_equal"


def _next_required_change(
    *,
    reason_codes: list[str],
    opportunity_count: int,
    changed_count: int,
    accepted_count: int,
    rejected_count: int,
    controlled_regression_count: int,
    family_with_acceptable_alternative_count: int = 0,
    missing_acceptable_alternative_families: list[str] | None = None,
    missed_safe_choice_count: int = 0,
    value_stability_requested: bool = False,
) -> str | None:
    if not reason_codes:
        return None
    if set(reason_codes) <= {"candidate_git_current_mismatch", "checkpoint_metadata_git_current_mismatch"}:
        return NEXT_PROVENANCE_REFRESH
    if value_stability_requested and "family_with_acceptable_alternative_count_below_threshold" in reason_codes:
        return NEXT_VALUE_OPPORTUNITY_GAP
    if value_stability_requested and (
        "accepted_better_choice_count_below_threshold" in reason_codes
        or "accepted_better_family_count_below_threshold" in reason_codes
    ):
        return NEXT_VALUE_ALIGNMENT
    if "family_with_acceptable_alternative_count_below_threshold" in reason_codes:
        if DENSE_CHOKE_FAMILY in set(missing_acceptable_alternative_families or []):
            return NEXT_DENSE_CHOKE_OPPORTUNITY_GAP
        return NEXT_OPPORTUNITY_GENERATION_GAP
    if missed_safe_choice_count > 0 and (
        "policy_changed_decision_count_below_threshold" in reason_codes
        or "canary_accepted_policy_choice_count_below_threshold" in reason_codes
        or "accepted_scenario_family_count_below_threshold" in reason_codes
    ):
        return NEXT_SAFE_CHOICE_ALIGNMENT
    if "scenario_family_count_below_threshold" in reason_codes or (
        "accepted_scenario_family_count_below_threshold" in reason_codes
        and accepted_count > 0
    ):
        return NEXT_FAMILY_COVERAGE_INSUFFICIENT
    if accepted_count <= 0:
        if changed_count > 0 or rejected_count > 0:
            return NEXT_GATE_FAILED
    if "canary_opportunity_context_count_below_threshold" in reason_codes or opportunity_count <= 0:
        return NEXT_OPPORTUNITY_INSUFFICIENT
    if accepted_count <= 0:
        return NEXT_TOO_CONSERVATIVE
    if "canary_accepted_policy_choice_count_below_threshold" in reason_codes:
        return NEXT_ACCEPTANCE_INSUFFICIENT
    if changed_count <= 0:
        return NEXT_TOO_CONSERVATIVE
    if opportunity_count <= 0:
        return NEXT_GATE_FAILED
    if controlled_regression_count > 0:
        return NEXT_POLICY_GATE_REGRESSION
    if any(reason.endswith("_regression") for reason in reason_codes):
        return NEXT_POLICY_GATE_REGRESSION
    return "policy_gated_canary_rollout_refinement_required"


def _scenario_family_summary(
    decisions: list[dict[str, Any]],
    opportunities: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    families: dict[str, dict[str, Any]] = {}

    def metrics_for(family: Any) -> dict[str, Any]:
        key = str(family or "unknown")
        if key not in families:
            families[key] = {
                "canary_opportunity_context_count": 0,
                "policy_decision_count": 0,
                "policy_changed_decision_count": 0,
                "source_aligned_count": 0,
                "canary_accepted_policy_choice_count": 0,
                "canary_rejected_policy_choice_count": 0,
                "accepted_equal_choice_count": 0,
                "accepted_better_choice_count": 0,
                "canary_rejection_reason_counts": Counter(),
                "acceptable_alternative_count": 0,
                "alternative_count": 0,
                "alternative_rejection_reason_counts": Counter(),
                "scenario_ids": set(),
            }
        return families[key]

    for opportunity in opportunities:
        metrics = metrics_for(opportunity.get("scenario_group"))
        metrics["canary_opportunity_context_count"] += 1
        metrics["alternative_count"] += int(opportunity.get("alternative_count", 0))
        metrics["acceptable_alternative_count"] += int(
            opportunity.get("acceptable_alternative_count", 0)
        )
        for alternative in opportunity.get("alternatives", []):
            if not isinstance(alternative, dict):
                continue
            if alternative.get("canary_gate_acceptable") is True:
                continue
            metrics["alternative_rejection_reason_counts"].update(
                alternative.get("canary_gate_rejection_reason_codes", [])
            )
        scenario_id = opportunity.get("scenario_id")
        if scenario_id:
            metrics["scenario_ids"].add(str(scenario_id))

    for decision in decisions:
        metrics = metrics_for(decision.get("scenario_group"))
        metrics["policy_decision_count"] += 1
        decision_class = decision.get("decision_class")
        if decision_class in {"canary_accepted_policy_choice", "canary_rejected_policy_choice"}:
            metrics["policy_changed_decision_count"] += 1
        if decision_class == "source_aligned":
            metrics["source_aligned_count"] += 1
        elif decision_class == "canary_accepted_policy_choice":
            metrics["canary_accepted_policy_choice_count"] += 1
            if decision.get("accepted_choice_value_class") == "accepted_better":
                metrics["accepted_better_choice_count"] += 1
            elif decision.get("accepted_choice_value_class") == "accepted_equal":
                metrics["accepted_equal_choice_count"] += 1
        elif decision_class == "canary_rejected_policy_choice":
            metrics["canary_rejected_policy_choice_count"] += 1
            metrics["canary_rejection_reason_counts"].update(
                decision.get("canary_rejection_reason_codes", [])
            )
        scenario_id = decision.get("scenario_id")
        if scenario_id:
            metrics["scenario_ids"].add(str(scenario_id))

    normalized: dict[str, dict[str, Any]] = {}
    for family, metrics in sorted(families.items()):
        normalized[family] = {
            "canary_opportunity_context_count": metrics["canary_opportunity_context_count"],
            "policy_decision_count": metrics["policy_decision_count"],
            "policy_changed_decision_count": metrics["policy_changed_decision_count"],
            "source_aligned_count": metrics["source_aligned_count"],
            "canary_accepted_policy_choice_count": metrics[
                "canary_accepted_policy_choice_count"
            ],
            "canary_rejected_policy_choice_count": metrics[
                "canary_rejected_policy_choice_count"
            ],
            "accepted_equal_choice_count": metrics["accepted_equal_choice_count"],
            "accepted_better_choice_count": metrics["accepted_better_choice_count"],
            "canary_rejection_reason_counts": dict(
                sorted(metrics["canary_rejection_reason_counts"].items())
            ),
            "alternative_count": metrics["alternative_count"],
            "acceptable_alternative_count": metrics["acceptable_alternative_count"],
            "alternative_rejection_reason_counts": dict(
                sorted(metrics["alternative_rejection_reason_counts"].items())
            ),
            "scenario_ids": sorted(metrics["scenario_ids"]),
        }
    return normalized


def _value_delta_summary(decisions: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    return {
        "path_cost_delta": _numeric_summary(
            decision.get("policy_selected_path_cost_delta") for decision in decisions
        ),
        "risk_delta": _numeric_summary(
            decision.get("policy_selected_risk_delta") for decision in decisions
        ),
        "utility_delta": _numeric_summary(
            decision.get("policy_selected_utility_delta") for decision in decisions
        ),
    }


def _numeric_summary(values: Any) -> dict[str, float | None]:
    numeric = [
        value
        for value in (_float_or_none(item) for item in values)
        if value is not None
    ]
    if not numeric:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(numeric),
        "max": max(numeric),
        "mean": sum(numeric) / len(numeric),
    }


def _input_paths(source_root: Path, candidate_root: Path, batch_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    inputs = config["input_files"]
    return {
        "source_batch_summary": source_root / inputs["source_batch_summary"],
        "holdout_batch_summary": batch_root / inputs["holdout_batch_summary"],
        "candidate_summary": candidate_root / inputs["candidate_summary"],
        "checkpoint": candidate_root / inputs["checkpoint"],
        "checkpoint_metadata": candidate_root / inputs["checkpoint_metadata"],
    }


def _output_paths(batch_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["output_files"]
    return {
        "decisions": batch_root / outputs["decisions"],
        "rejection_report": batch_root / outputs["rejection_report"],
        "opportunity_summary": batch_root / outputs["opportunity_summary"],
        "summary": batch_root / outputs["summary"],
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
    for section in ("input_files", "output_files", "validation", "evaluation"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


def _load_summary(path: Path, *, expected_schema: str | None, label: str, reason_codes: list[str]) -> dict[str, Any]:
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _append_reason(reason_codes, f"{label}_invalid_json")
        return {}
    if not isinstance(payload, dict):
        _append_reason(reason_codes, f"{label}_invalid_json_root")
        return {}
    if expected_schema is not None and payload.get("schema_version") != expected_schema:
        _append_reason(reason_codes, f"{label}_schema_version_mismatch")
    return payload


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _delta(value: Any, reference: Any) -> float:
    try:
        numeric = float(value)
        baseline = float(reference)
    except (TypeError, ValueError):
        return 0.0
    return numeric - baseline


def _install_model_explorer_path(repo_root: Path) -> None:
    source = repo_root / "model-explorer" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
