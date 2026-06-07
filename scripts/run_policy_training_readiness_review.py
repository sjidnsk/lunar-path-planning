from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot
from git_provenance import git_snapshots_match as _git_snapshots_match
from git_provenance import inspect_source_git_provenance as _inspect_source_git_provenance
from git_provenance import public_git as _public_git


CONFIG_SCHEMA_VERSION = "policy-training-readiness-review-config/v1"
SUMMARY_SCHEMA_VERSION = "policy-training-readiness-review-summary/v1"
SMOKE_SCHEMA_VERSION = "calibrated-policy-application-smoke-summary/v1"
READINESS_SCHEMA_VERSION = "channel-aware-training-readiness-summary/v1"
COVERAGE_SCHEMA_VERSION = "channel-aware-contrast-coverage-summary/v1"
CALIBRATION_SCHEMA_VERSION = "channel-aware-selection-contrast-calibration-summary/v1"
ANCHOR_CANDIDATE_SCHEMA_VERSION = "anchor-projection-candidate-generation-summary/v1"
ANCHOR_CONTRACT_SCHEMA_VERSION = "anchor-projection-evidence-contract-summary/v1"
SUBMODULES = ("dev-platform-constraints", "model-explorer", "path-planner")
READY_SMOKE_ACTION = "ready_for_policy_training_readiness_review"
READY_DRY_RUN_ACTION = "ready_for_limited_policy_training_dry_run"
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
FALLBACK_COUNT_FIELDS = (
    "open_grid_fallback_used_count",
    "open_grid_fallback_count",
    "fallback_used_count",
    "fallback_or_open_grid_count",
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Review calibrated policy target selection evidence before any policy training."
    )
    parser.add_argument("--batch-root", required=True, help="Batch root containing channel-aware summaries.")
    parser.add_argument(
        "--calibrated-policy-application-smoke-summary",
        help="calibrated-policy-application-smoke-summary/v1 JSON. Defaults to <batch-root>/calibrated-policy-application-smoke-summary.json.",
    )
    parser.add_argument(
        "--readiness-summary",
        help="channel-aware-training-readiness-summary/v1 JSON. Defaults to <batch-root>/channel-aware-training-readiness-summary.json.",
    )
    parser.add_argument(
        "--contrast-coverage-summary",
        help="channel-aware-contrast-coverage-summary/v1 JSON. Defaults to <batch-root>/channel-aware-contrast-coverage-summary.json.",
    )
    parser.add_argument(
        "--selection-contrast-calibration-summary",
        help="channel-aware-selection-contrast-calibration-summary/v1 JSON. Defaults to <batch-root>/channel-aware-selection-contrast-calibration-summary.json.",
    )
    parser.add_argument(
        "--anchor-projection-candidate-generation-summary",
        help="Optional anchor-projection-candidate-generation-summary/v1 JSON. Defaults to <batch-root>/anchor-projection-candidate-generation-summary.json when present.",
    )
    parser.add_argument(
        "--anchor-projection-evidence-contract-summary",
        help="Optional anchor-projection-evidence-contract-summary/v1 JSON. Defaults to <batch-root>/anchor-projection-evidence-contract-summary.json when present.",
    )
    parser.add_argument("--config", required=True, help="Policy training readiness review config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned output paths.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without writing outputs.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    batch_root = _resolve_path(args.batch_root, repo_root)
    smoke_path = (
        _resolve_path(args.calibrated_policy_application_smoke_summary, repo_root)
        if args.calibrated_policy_application_smoke_summary
        else batch_root / "calibrated-policy-application-smoke-summary.json"
    )
    readiness_path = (
        _resolve_path(args.readiness_summary, repo_root)
        if args.readiness_summary
        else batch_root / "channel-aware-training-readiness-summary.json"
    )
    coverage_path = (
        _resolve_path(args.contrast_coverage_summary, repo_root)
        if args.contrast_coverage_summary
        else batch_root / "channel-aware-contrast-coverage-summary.json"
    )
    calibration_path = (
        _resolve_path(args.selection_contrast_calibration_summary, repo_root)
        if args.selection_contrast_calibration_summary
        else batch_root / "channel-aware-selection-contrast-calibration-summary.json"
    )
    anchor_candidate_path = (
        _resolve_path(args.anchor_projection_candidate_generation_summary, repo_root)
        if args.anchor_projection_candidate_generation_summary
        else batch_root / "anchor-projection-candidate-generation-summary.json"
    )
    anchor_contract_path = (
        _resolve_path(args.anchor_projection_evidence_contract_summary, repo_root)
        if args.anchor_projection_evidence_contract_summary
        else batch_root / "anchor-projection-evidence-contract-summary.json"
    )
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    summary = analyze_policy_training_readiness_review(
        batch_root=batch_root,
        smoke_path=smoke_path,
        readiness_path=readiness_path,
        coverage_path=coverage_path,
        calibration_path=calibration_path,
        anchor_candidate_path=anchor_candidate_path,
        anchor_contract_path=anchor_contract_path,
        anchor_candidate_required=bool(args.anchor_projection_candidate_generation_summary),
        anchor_contract_required=bool(args.anchor_projection_evidence_contract_summary),
        config=config,
        repo_root=repo_root,
    )
    output_file = _output_file(batch_root, config)
    validation_message = {
        "status": "config validated" if summary["status"] == "passed" else "validation failed",
        "batch_root": _display_path(batch_root, repo_root),
        "calibrated_policy_application_smoke_summary": _display_path(smoke_path, repo_root),
        "readiness_summary": _display_path(readiness_path, repo_root),
        "contrast_coverage_summary": _display_path(coverage_path, repo_root),
        "selection_contrast_calibration_summary": _display_path(calibration_path, repo_root),
        "anchor_projection_candidate_generation_summary": (
            _display_path(anchor_candidate_path, repo_root)
            if anchor_candidate_path.is_file() or args.anchor_projection_candidate_generation_summary
            else None
        ),
        "anchor_projection_evidence_contract_summary": (
            _display_path(anchor_contract_path, repo_root)
            if anchor_contract_path.is_file() or args.anchor_projection_evidence_contract_summary
            else None
        ),
        "config": _display_path(config_path, repo_root),
        "reason_codes": summary["reason_codes"],
        "training_readiness_status": summary["training_readiness_status"],
        "training_blockers": summary["training_blockers"],
        "recommended_next_action": summary["recommended_next_action"],
        "policy_training_readiness_review_summary": _display_path(output_file, repo_root),
    }
    print(json.dumps(validation_message, ensure_ascii=False))

    if args.validate_only or args.dry_run:
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "dry-run",
                        "would_write": {
                            "policy_training_readiness_review_summary": _display_path(
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
                "training_readiness_status": summary["training_readiness_status"],
                "policy_training_readiness_review_summary": _display_path(output_file, repo_root),
                "recommended_next_action": summary["recommended_next_action"],
                "failure_reason_code_counts": summary["failure_reason_code_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def analyze_policy_training_readiness_review(
    *,
    batch_root: Path,
    smoke_path: Path,
    readiness_path: Path,
    coverage_path: Path,
    calibration_path: Path,
    anchor_candidate_path: Path,
    anchor_contract_path: Path,
    anchor_candidate_required: bool = False,
    anchor_contract_required: bool = False,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    source_summaries: dict[str, Any] = {}
    anchor_only_mode = (
        anchor_candidate_required
        and anchor_contract_required
        and anchor_candidate_path.is_file()
        and anchor_contract_path.is_file()
        and not any(path.is_file() for path in (smoke_path, readiness_path, coverage_path, calibration_path))
    )
    if anchor_only_mode:
        smoke = _load_optional_source(
            smoke_path,
            label="calibrated_policy_application_smoke_summary",
            expected_schema=SMOKE_SCHEMA_VERSION,
            repo_root=repo_root,
            reason_codes=reason_codes,
            source_summaries=source_summaries,
        )
        readiness = _load_optional_source(
            readiness_path,
            label="channel_aware_training_readiness_summary",
            expected_schema=READINESS_SCHEMA_VERSION,
            repo_root=repo_root,
            reason_codes=reason_codes,
            source_summaries=source_summaries,
        )
        coverage = _load_optional_source(
            coverage_path,
            label="channel_aware_contrast_coverage_summary",
            expected_schema=COVERAGE_SCHEMA_VERSION,
            repo_root=repo_root,
            reason_codes=reason_codes,
            source_summaries=source_summaries,
        )
        calibration = _load_optional_source(
            calibration_path,
            label="channel_aware_selection_contrast_calibration_summary",
            expected_schema=CALIBRATION_SCHEMA_VERSION,
            repo_root=repo_root,
            reason_codes=reason_codes,
            source_summaries=source_summaries,
        )
    else:
        smoke = _load_source(
            smoke_path,
            label="calibrated_policy_application_smoke_summary",
            expected_schema=SMOKE_SCHEMA_VERSION,
            repo_root=repo_root,
            reason_codes=reason_codes,
            source_summaries=source_summaries,
        )
        readiness = _load_source(
            readiness_path,
            label="channel_aware_training_readiness_summary",
            expected_schema=READINESS_SCHEMA_VERSION,
            repo_root=repo_root,
            reason_codes=reason_codes,
            source_summaries=source_summaries,
        )
        coverage = _load_source(
            coverage_path,
            label="channel_aware_contrast_coverage_summary",
            expected_schema=COVERAGE_SCHEMA_VERSION,
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
    anchor_candidate = _load_optional_source(
        anchor_candidate_path,
        label="anchor_projection_candidate_generation_summary",
        expected_schema=ANCHOR_CANDIDATE_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=anchor_candidate_required,
    )
    anchor_contract = _load_optional_source(
        anchor_contract_path,
        label="anchor_projection_evidence_contract_summary",
        expected_schema=ANCHOR_CONTRACT_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=anchor_contract_required,
    )
    if _fail_on_input_failure(config):
        for label, payload in (
            ("calibrated_policy_application_smoke_summary", smoke),
            ("channel_aware_training_readiness_summary", readiness),
            ("channel_aware_contrast_coverage_summary", coverage),
            ("channel_aware_selection_contrast_calibration_summary", calibration),
            ("anchor_projection_candidate_generation_summary", anchor_candidate),
            ("anchor_projection_evidence_contract_summary", anchor_contract),
        ):
            if payload.get("status") == "failed":
                _append_reason(reason_codes, f"{label}_failed")

    current_git = _git_snapshot(repo_root)
    source_git_matches = []
    if not anchor_only_mode:
        source_git_matches.extend(
            [
                _inspect_git(smoke, label="calibrated_policy_application_smoke_summary", current_git=current_git, config=config, reason_codes=reason_codes),
                _inspect_git(readiness, label="channel_aware_training_readiness_summary", current_git=current_git, config=config, reason_codes=reason_codes),
                _inspect_git(coverage, label="channel_aware_contrast_coverage_summary", current_git=current_git, config=config, reason_codes=reason_codes),
                _inspect_git(calibration, label="channel_aware_selection_contrast_calibration_summary", current_git=current_git, config=config, reason_codes=reason_codes),
            ]
        )
    if anchor_candidate:
        source_git_matches.append(
            _inspect_git(
                anchor_candidate,
                label="anchor_projection_candidate_generation_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if anchor_contract:
        source_git_matches.append(
            _inspect_git(
                anchor_contract,
                label="anchor_projection_evidence_contract_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )

    review = _review_metrics(
        smoke=smoke,
        readiness=readiness,
        coverage=coverage,
        calibration=calibration,
        anchor_candidate=anchor_candidate,
        anchor_contract=anchor_contract,
        validation_reason_codes=reason_codes,
        anchor_only_mode=anchor_only_mode,
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
        "calibrated_policy_application_smoke_summary_path": _display_path(smoke_path, repo_root),
        "readiness_summary_path": _display_path(readiness_path, repo_root),
        "contrast_coverage_summary_path": _display_path(coverage_path, repo_root),
        "selection_contrast_calibration_summary_path": _display_path(calibration_path, repo_root),
        "anchor_projection_candidate_generation_summary_path": (
            _display_path(anchor_candidate_path, repo_root) if anchor_candidate else None
        ),
        "anchor_projection_evidence_contract_summary_path": (
            _display_path(anchor_contract_path, repo_root) if anchor_contract else None
        ),
        "application_scope": (
            "anchor_projection_readiness_contract_review_only"
            if anchor_only_mode
            else "calibrated_policy_training_readiness_review_audit_only"
        ),
        "quality_signal_use": "calibrated_policy_target_training_contract_review_only",
        "source_summaries": source_summaries,
        "config": _public_config(config),
        "git_provenance": {
            "current": current_git,
            "calibrated_policy_application_smoke": _public_git(smoke),
            "training_readiness": _public_git(readiness),
            "contrast_coverage": _public_git(coverage),
            "selection_contrast_calibration": _public_git(calibration),
            "anchor_projection_candidate_generation": _public_git(anchor_candidate),
            "anchor_projection_evidence_contract": _public_git(anchor_contract),
            "current_matches_sources": all(source_git_matches),
        },
        **review,
        "runs_training": False,
        "audit_only": True,
        "no_ppo_training": True,
        "no_large_scale_training": True,
        "no_real_world_performance_claim": True,
        "channel_aware_backend_opt_in": True,
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


def _review_metrics(
    *,
    smoke: dict[str, Any],
    readiness: dict[str, Any],
    coverage: dict[str, Any],
    calibration: dict[str, Any],
    anchor_candidate: dict[str, Any],
    anchor_contract: dict[str, Any],
    validation_reason_codes: list[str],
    anchor_only_mode: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    thresholds = config["readiness_thresholds"]
    source_rate = _source_rate(smoke, readiness, coverage, calibration)
    calibrated_rate = _calibrated_rate(smoke, readiness, coverage, calibration)
    applied_count = _int_value_or_default(smoke.get("applied_calibrated_candidate_count"), 0)
    rejected_goal_blocked_count = max(
        _int_value_or_default(smoke.get("rejected_goal_blocked_count"), 0),
        _int_value_or_default(calibration.get("goal_blocked_count"), 0),
    )
    platform_goal_contract_mismatch_count = max(
        _int_value_or_default(smoke.get("platform_goal_contract_mismatch_count"), 0),
        _int_value_or_default(calibration.get("platform_goal_contract_mismatch_count"), 0),
    )
    platform_goal_trainable_anchor_projection_count = max(
        _int_value_or_default(smoke.get("platform_goal_trainable_anchor_projection_count"), 0),
        _int_value_or_default(calibration.get("platform_goal_trainable_anchor_projection_count"), 0),
    )
    platform_goal_nontrainable_blocked_target_count = max(
        _int_value_or_default(
            smoke.get("platform_goal_nontrainable_blocked_target_count"),
            platform_goal_contract_mismatch_count - platform_goal_trainable_anchor_projection_count,
        ),
        _int_value_or_default(
            calibration.get("platform_goal_nontrainable_blocked_target_count"),
            platform_goal_contract_mismatch_count - platform_goal_trainable_anchor_projection_count,
        ),
    )
    platform_goal_anchor_available_count = max(
        _int_value_or_default(smoke.get("platform_goal_anchor_available_count"), 0),
        _int_value_or_default(calibration.get("platform_goal_anchor_available_count"), 0),
    )
    platform_goal_unresolved_count = max(
        _int_value_or_default(smoke.get("platform_goal_unresolved_count"), 0),
        _int_value_or_default(calibration.get("platform_goal_unresolved_count"), 0),
    )
    platform_goal_class_counts = _max_counter_dict(
        smoke.get("platform_goal_feasibility_class_counts"),
        calibration.get("platform_goal_feasibility_class_counts"),
    )
    safety_regression_count = max(
        _int_value_or_default(smoke.get("safety_regression_count"), 0),
        _int_value_or_default(readiness.get("calibration_safety_regression_count"), 0),
        _int_value_or_default(coverage.get("safety_regression_count"), 0),
        _int_value_or_default(calibration.get("safety_regression_count"), 0),
        _int_value_or_default(anchor_candidate.get("safety_regression_count"), 0),
        _int_value_or_default(anchor_contract.get("safety_regression_count"), 0),
    )
    fallback_or_open_grid_count = max(
        _fallback_or_open_grid_count(smoke),
        _fallback_or_open_grid_count(readiness),
        _fallback_or_open_grid_count(coverage),
        _fallback_or_open_grid_count(calibration),
        _fallback_or_open_grid_count(anchor_candidate),
        _fallback_or_open_grid_count(anchor_contract),
    )
    changed_scenario_ids = _unique(
        _string_list(smoke.get("changed_scenario_ids"))
        or _string_list(coverage.get("changed_scenario_ids"))
        or _string_list(calibration.get("changed_scenario_ids"))
    )
    contract_mutations = _contract_mutations(
        {
            "calibrated_policy_application_smoke": smoke,
            "training_readiness": readiness,
            "contrast_coverage": coverage,
            "selection_contrast_calibration": calibration,
            "anchor_projection_candidate_generation": anchor_candidate,
            "anchor_projection_evidence_contract": anchor_contract,
        }
    )
    anchor_projection_readiness = _anchor_projection_readiness(
        candidate=anchor_candidate,
        contract=anchor_contract,
        thresholds=thresholds,
    )
    training_blockers: list[str] = []
    if validation_reason_codes:
        for reason in validation_reason_codes:
            _append_reason(training_blockers, reason)
    if (
        not anchor_only_mode
        and thresholds["require_smoke_ready_for_training_review"]
        and smoke.get("recommended_next_action") != READY_SMOKE_ACTION
    ):
        _append_reason(training_blockers, "calibrated_application_smoke_not_ready_for_training_review")
    if not anchor_only_mode and applied_count < thresholds["min_applied_calibrated_candidate_count"]:
        _append_reason(training_blockers, "applied_calibrated_candidate_count_below_training_threshold")
    if not anchor_only_mode and calibrated_rate - source_rate < thresholds["min_calibrated_selection_rate_delta"]:
        _append_reason(training_blockers, "calibrated_selection_rate_delta_below_training_threshold")
    if rejected_goal_blocked_count > thresholds["max_rejected_goal_blocked_count"]:
        _append_reason(training_blockers, "goal_blocked_candidates_excluded_from_training_positive_evidence")
    if safety_regression_count > thresholds["max_safety_regression_count"]:
        _append_reason(training_blockers, "safety_regression_blocks_training_readiness")
    if fallback_or_open_grid_count > thresholds["max_fallback_or_open_grid_count"]:
        _append_reason(training_blockers, "fallback_or_open_grid_evidence_blocks_training_readiness")
    if contract_mutations:
        _append_reason(training_blockers, "contract_mutation_blocks_training_readiness")
    if anchor_only_mode and anchor_projection_readiness["candidate_generation_nontrainable_count"] > 0:
        _append_reason(training_blockers, "anchor_projection_nontrainable_contexts_remain")
    for reason in anchor_projection_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)

    hard_validation_failed = bool(validation_reason_codes)
    if hard_validation_failed:
        training_readiness_status = "blocked_by_validation"
        recommended_next_action = "fix_validation_failures_before_training_readiness_review"
    elif training_blockers:
        training_readiness_status = "needs_training_contract_refinement"
        recommended_next_action = "needs_training_contract_refinement"
    else:
        training_readiness_status = READY_DRY_RUN_ACTION
        recommended_next_action = READY_DRY_RUN_ACTION

    excluded_candidate_count = rejected_goal_blocked_count + safety_regression_count + fallback_or_open_grid_count
    contract_status = "compatible_audit_only" if not contract_mutations else "contract_mutation_detected"
    return {
        "training_readiness_status": training_readiness_status,
        "source_selected_candidate_changed_rate": source_rate,
        "calibrated_selected_candidate_changed_rate": calibrated_rate,
        "calibrated_selection_rate_delta": calibrated_rate - source_rate,
        "applied_calibrated_candidate_count": applied_count,
        "changed_scenario_ids": changed_scenario_ids,
        "rejected_goal_blocked_count": rejected_goal_blocked_count,
        "platform_goal_contract_mismatch_count": platform_goal_contract_mismatch_count,
        "platform_goal_trainable_anchor_projection_count": platform_goal_trainable_anchor_projection_count,
        "platform_goal_nontrainable_blocked_target_count": platform_goal_nontrainable_blocked_target_count,
        "platform_goal_anchor_available_count": platform_goal_anchor_available_count,
        "platform_goal_unresolved_count": platform_goal_unresolved_count,
        "platform_goal_feasibility_class_counts": platform_goal_class_counts,
        "safety_regression_count": safety_regression_count,
        "fallback_or_open_grid_count": fallback_or_open_grid_count,
        "training_positive_candidate_count": applied_count,
        "excluded_candidate_count": excluded_candidate_count,
        "anchor_projection_readiness": anchor_projection_readiness,
        "anchor_projection_candidate_generation_trainable_count": anchor_projection_readiness[
            "candidate_generation_trainable_count"
        ],
        "anchor_projection_contract_trainable_count": anchor_projection_readiness["contract_trainable_count"],
        "anchor_projection_readiness_trainable_count": anchor_projection_readiness["readiness_trainable_count"],
        "anchor_projection_candidate_contract_alignment_gap_count": anchor_projection_readiness[
            "candidate_contract_alignment_gap_count"
        ],
        "anchor_projection_anchor_unreachable_count": anchor_projection_readiness["anchor_unreachable_count"],
        "anchor_projection_source_candidate_not_selected_count": anchor_projection_readiness[
            "source_candidate_not_selected_count"
        ],
        "anchor_projection_reachable_substitute_anchor_found_count": anchor_projection_readiness[
            "reachable_substitute_anchor_found_count"
        ],
        "anchor_projection_anchor_unreachable_repaired_by_reachable_substitute_count": (
            anchor_projection_readiness["anchor_unreachable_repaired_by_reachable_substitute_count"]
        ),
        "anchor_projection_true_geometry_unreachable_count": anchor_projection_readiness[
            "true_geometry_unreachable_count"
        ],
        "anchor_projection_audit_proxy_positive_count": anchor_projection_readiness[
            "audit_proxy_positive_count"
        ],
        "training_blockers": training_blockers,
        "contract_impact": {
            "training_contract_status": contract_status,
            "contract_mutations": contract_mutations,
            "source_policy_target_selection_improvement_claimed": bool(
                smoke.get(
                    "source_policy_target_selection_improvement_claimed",
                    calibration.get("source_supports_policy_target_selection_improvement_claim", False),
                )
            ),
            "calibrated_selection_only": source_rate == 0.0 and calibrated_rate > 0.0,
            "policy_training_scope": (
                "limited_policy_training_dry_run_only"
                if recommended_next_action == READY_DRY_RUN_ACTION
                else "audit_contract_refinement_only"
            ),
        },
        "recommended_next_action": recommended_next_action,
        "readiness_source_status": {
            "smoke_recommended_next_action": str(smoke.get("recommended_next_action", "")),
            "readiness_status": str(readiness.get("readiness_status", "")),
            "calibrated_readiness_status": str(readiness.get("calibrated_readiness_status", "")),
            "coverage_recommended_next_action": str(coverage.get("recommended_next_action", "")),
        },
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
        output_files.get("policy_training_readiness_review_summary")
    ):
        raise ConfigError("output_files.policy_training_readiness_review_summary must be a non-empty string")
    thresholds = payload.get("readiness_thresholds")
    if not isinstance(thresholds, dict):
        raise ConfigError("readiness_thresholds must be an object")
    normalized_thresholds = {
        "min_applied_calibrated_candidate_count": _int_value(
            thresholds.get("min_applied_calibrated_candidate_count", 1),
            "readiness_thresholds.min_applied_calibrated_candidate_count",
        ),
        "min_calibrated_selection_rate_delta": _float_value(
            thresholds.get("min_calibrated_selection_rate_delta", 0.01),
            "readiness_thresholds.min_calibrated_selection_rate_delta",
        ),
        "max_rejected_goal_blocked_count": _int_value(
            thresholds.get("max_rejected_goal_blocked_count", 0),
            "readiness_thresholds.max_rejected_goal_blocked_count",
        ),
        "max_safety_regression_count": _int_value(
            thresholds.get("max_safety_regression_count", 0),
            "readiness_thresholds.max_safety_regression_count",
        ),
        "max_fallback_or_open_grid_count": _int_value(
            thresholds.get("max_fallback_or_open_grid_count", 0),
            "readiness_thresholds.max_fallback_or_open_grid_count",
        ),
        "max_anchor_projection_source_selection_quality_regression_count": _int_value(
            thresholds.get("max_anchor_projection_source_selection_quality_regression_count", 0),
            "readiness_thresholds.max_anchor_projection_source_selection_quality_regression_count",
        ),
        "max_anchor_projection_path_cost_regression": _optional_nonnegative_float(
            thresholds.get("max_anchor_projection_path_cost_regression"),
            "readiness_thresholds.max_anchor_projection_path_cost_regression",
        ),
        "max_anchor_projection_risk_regression": _optional_nonnegative_float(
            thresholds.get("max_anchor_projection_risk_regression"),
            "readiness_thresholds.max_anchor_projection_risk_regression",
        ),
        "require_smoke_ready_for_training_review": bool(
            thresholds.get("require_smoke_ready_for_training_review", True)
        ),
    }
    if normalized_thresholds["min_applied_calibrated_candidate_count"] < 0:
        raise ConfigError("readiness_thresholds.min_applied_calibrated_candidate_count must be >= 0")
    if normalized_thresholds["min_calibrated_selection_rate_delta"] < 0.0:
        raise ConfigError("readiness_thresholds.min_calibrated_selection_rate_delta must be >= 0")
    for key in (
        "max_rejected_goal_blocked_count",
        "max_safety_regression_count",
        "max_fallback_or_open_grid_count",
        "max_anchor_projection_source_selection_quality_regression_count",
    ):
        if normalized_thresholds[key] < 0:
            raise ConfigError(f"readiness_thresholds.{key} must be >= 0")
    config = dict(payload)
    config["readiness_thresholds"] = normalized_thresholds
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


def _load_optional_source(
    path: Path,
    *,
    label: str,
    expected_schema: str,
    repo_root: Path,
    reason_codes: list[str],
    source_summaries: dict[str, Any],
    required: bool = False,
) -> dict[str, Any]:
    if path.is_file() or required:
        return _load_source(
            path,
            label=label,
            expected_schema=expected_schema,
            repo_root=repo_root,
            reason_codes=reason_codes,
            source_summaries=source_summaries,
        )
    source_summaries[label] = {
        "path": _display_path(path, repo_root),
        "exists": False,
        "optional": True,
    }
    return {}


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


def _source_rate(
    smoke: dict[str, Any],
    readiness: dict[str, Any],
    coverage: dict[str, Any],
    calibration: dict[str, Any],
) -> float:
    for payload, key in (
        (smoke, "source_selected_candidate_changed_rate"),
        (readiness, "source_selected_candidate_changed_rate"),
        (coverage, "source_selected_candidate_changed_rate"),
        (calibration, "source_selected_candidate_changed_rate"),
    ):
        if payload.get(key) is not None:
            return _float_value_or_default(payload.get(key), 0.0)
    return 0.0


def _calibrated_rate(
    smoke: dict[str, Any],
    readiness: dict[str, Any],
    coverage: dict[str, Any],
    calibration: dict[str, Any],
) -> float:
    for payload, key in (
        (smoke, "calibrated_selected_candidate_changed_rate"),
        (coverage, "calibrated_selected_candidate_changed_rate"),
        (readiness, "calibration_selected_candidate_changed_rate"),
        (calibration, "selected_candidate_changed_rate"),
    ):
        if payload.get(key) is not None:
            return _float_value_or_default(payload.get(key), 0.0)
    return 0.0


def _fallback_or_open_grid_count(payload: dict[str, Any]) -> int:
    return max(_int_value_or_default(payload.get(field), 0) for field in FALLBACK_COUNT_FIELDS)


def _contract_mutations(labeled_payloads: dict[str, dict[str, Any]]) -> list[str]:
    mutations: list[str] = []
    for label, payload in labeled_payloads.items():
        for field in CONTRACT_GUARD_FIELDS:
            if payload.get(field) is False:
                mutations.append(f"{label}.{field}")
    return sorted(set(mutations))


def _anchor_projection_readiness(
    *,
    candidate: dict[str, Any],
    contract: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    candidate_present = bool(candidate)
    contract_present = bool(contract)
    candidate_trainable = _int_value_or_default(candidate.get("trainable_anchor_projection_count"), 0)
    candidate_nontrainable = _int_value_or_default(
        candidate.get("nontrainable_blocked_target_count"),
        _int_value_or_default(candidate.get("nontrainable_anchor_projection_count"), 0),
    )
    contract_trainable = _int_value_or_default(contract.get("trainable_anchor_projection_count"), 0)
    contract_nontrainable = _int_value_or_default(
        contract.get("nontrainable_blocked_target_count"),
        _int_value_or_default(contract.get("nontrainable_anchor_projection_count"), 0),
    )
    if candidate_present and contract_present:
        readiness_trainable = min(candidate_trainable, contract_trainable)
        alignment_gap = max(candidate_trainable - contract_trainable, 0)
    else:
        readiness_trainable = 0
        alignment_gap = candidate_trainable if candidate_present else 0
    anchor_unreachable_count = _candidate_nontrainable_reason_count(candidate, "anchor_unreachable")
    source_candidate_not_selected_count = _candidate_nontrainable_reason_count(
        candidate,
        "source_candidate_not_selected",
    )
    reachable_substitute_anchor_found_count = max(
        _int_value_or_default(candidate.get("reachable_substitute_anchor_found_count"), 0),
        _int_value_or_default(contract.get("reachable_substitute_anchor_found_count"), 0),
        _coverage_diagnosis_int(candidate, "reachable_substitute_anchor_found_count"),
    )
    anchor_unreachable_repaired_count = max(
        _int_value_or_default(
            candidate.get("anchor_unreachable_repaired_by_reachable_substitute_count"),
            0,
        ),
        _int_value_or_default(
            contract.get("anchor_unreachable_repaired_by_reachable_substitute_count"),
            0,
        ),
        _coverage_diagnosis_int(
            candidate,
            "anchor_unreachable_repaired_by_reachable_substitute_count",
        ),
    )
    true_geometry_unreachable_count = max(
        _int_value_or_default(candidate.get("true_geometry_unreachable_count"), 0),
        _int_value_or_default(contract.get("true_geometry_unreachable_count"), 0),
        _coverage_diagnosis_int(candidate, "true_geometry_unreachable_count"),
    )
    audit_proxy_positive_count = max(
        _int_value_or_default(candidate.get("positive_training_evidence_contains_audit_proxy_anchor_count"), 0),
        _int_value_or_default(candidate.get("audit_proxy_positive_count"), 0),
        _int_value_or_default(contract.get("positive_training_evidence_contains_audit_proxy_anchor_count"), 0),
        _int_value_or_default(contract.get("audit_proxy_positive_count"), 0),
    )
    source_quality_regression_count = _int_value_or_default(
        candidate.get("source_selection_quality_regression_count"),
        _quality_regression_count_from_contexts(candidate),
    )
    diagnostic_max_path_margin = _max_numeric(
        candidate.get("max_source_selection_path_cost_margin_vs_best_alternative"),
        _max_context_field(candidate, "source_selection_path_cost_margin_vs_best_alternative"),
    )
    diagnostic_max_risk_margin = _max_numeric(
        candidate.get("max_source_selection_risk_margin_vs_best_alternative"),
        _max_context_field(candidate, "source_selection_risk_margin_vs_best_alternative"),
    )
    max_path_margin = _max_context_field(
        candidate,
        "source_selection_path_cost_margin_vs_best_alternative",
        predicate=_context_margin_can_block_readiness,
    )
    max_risk_margin = _max_context_field(
        candidate,
        "source_selection_risk_margin_vs_best_alternative",
        predicate=_context_margin_can_block_readiness,
    )
    if source_quality_regression_count > 0:
        max_path_margin = _max_numeric(
            max_path_margin,
            candidate.get("max_source_selection_path_cost_margin_vs_best_alternative"),
        )
        max_risk_margin = _max_numeric(
            max_risk_margin,
            candidate.get("max_source_selection_risk_margin_vs_best_alternative"),
        )
    quality_blockers: list[str] = []
    training_blockers: list[str] = []
    max_quality_regressions = thresholds.get("max_anchor_projection_source_selection_quality_regression_count")
    if source_quality_regression_count > _int_value_or_default(max_quality_regressions, 0):
        _append_reason(quality_blockers, "anchor_projection_source_selection_quality_regression")
    max_allowed_path_margin = thresholds.get("max_anchor_projection_path_cost_regression")
    if (
        max_allowed_path_margin is not None
        and max_path_margin is not None
        and max_path_margin > float(max_allowed_path_margin)
    ):
        _append_reason(quality_blockers, "anchor_projection_source_selection_path_cost_regression")
    max_allowed_risk_margin = thresholds.get("max_anchor_projection_risk_regression")
    if (
        max_allowed_risk_margin is not None
        and max_risk_margin is not None
        and max_risk_margin > float(max_allowed_risk_margin)
    ):
        _append_reason(quality_blockers, "anchor_projection_source_selection_risk_regression")
    if contract_present and candidate_present and contract_trainable < candidate_trainable:
        _append_reason(training_blockers, "anchor_projection_contract_trainable_count_below_candidate_generation")
    if audit_proxy_positive_count > 0:
        _append_reason(training_blockers, "anchor_projection_positive_evidence_contains_audit_proxy_anchor")
    for reason in quality_blockers:
        _append_reason(training_blockers, reason)
    return {
        "candidate_generation_present": candidate_present,
        "contract_present": contract_present,
        "candidate_generation_trainable_count": candidate_trainable,
        "candidate_generation_nontrainable_count": candidate_nontrainable,
        "contract_trainable_count": contract_trainable,
        "contract_nontrainable_count": contract_nontrainable,
        "readiness_trainable_count": readiness_trainable,
        "candidate_contract_alignment_gap_count": alignment_gap,
        "anchor_unreachable_count": anchor_unreachable_count,
        "source_candidate_not_selected_count": source_candidate_not_selected_count,
        "reachable_substitute_anchor_found_count": reachable_substitute_anchor_found_count,
        "anchor_unreachable_repaired_by_reachable_substitute_count": anchor_unreachable_repaired_count,
        "true_geometry_unreachable_count": true_geometry_unreachable_count,
        "audit_proxy_positive_count": audit_proxy_positive_count,
        "source_selection_quality_regression_count": source_quality_regression_count,
        "max_source_selection_path_cost_margin_vs_best_alternative": max_path_margin,
        "max_source_selection_risk_margin_vs_best_alternative": max_risk_margin,
        "diagnostic_max_source_selection_path_cost_margin_vs_best_alternative": diagnostic_max_path_margin,
        "diagnostic_max_source_selection_risk_margin_vs_best_alternative": diagnostic_max_risk_margin,
        "quality_regression_blockers": quality_blockers,
        "training_blockers": training_blockers,
    }


def _candidate_nontrainable_reason_count(payload: dict[str, Any], reason: str) -> int:
    explicit_fields = {
        "anchor_unreachable": (
            "nontrainable_anchor_unreachable_count",
            "anchor_unreachable_count",
        ),
        "source_candidate_not_selected": (
            "nontrainable_source_candidate_not_selected_count",
            "source_candidate_not_selected_count",
        ),
    }
    for field in explicit_fields.get(reason, ()):
        if payload.get(field) is not None:
            return _int_value_or_default(payload.get(field), 0)
    diagnosis = payload.get("anchor_projection_coverage_diagnosis")
    diagnosis = diagnosis if isinstance(diagnosis, dict) else {}
    reason_counts = diagnosis.get("nontrainable_primary_reason_counts")
    if isinstance(reason_counts, dict) and reason_counts.get(reason) is not None:
        return _int_value_or_default(reason_counts.get(reason), 0)
    fallback_fields = {
        "anchor_unreachable": "anchor_unreachable_not_generated_count",
        "source_candidate_not_selected": "projected_candidate_not_source_selected_count",
    }
    field = fallback_fields.get(reason)
    if field:
        return _int_value_or_default(diagnosis.get(field), 0)
    return 0


def _coverage_diagnosis_int(payload: dict[str, Any], field: str) -> int:
    diagnosis = payload.get("anchor_projection_coverage_diagnosis")
    diagnosis = diagnosis if isinstance(diagnosis, dict) else {}
    return _int_value_or_default(diagnosis.get(field), 0)


def _quality_regression_count_from_contexts(payload: dict[str, Any]) -> int:
    contexts = payload.get("context_records")
    if not isinstance(contexts, list):
        return 0
    return sum(
        1
        for context in contexts
        if isinstance(context, dict)
        and (
            context.get("source_selection_quality_regression") is True
            or "source_selection_quality_regression" in _string_list(context.get("reject_reasons"))
        )
    )


def _max_context_field(
    payload: dict[str, Any],
    field: str,
    *,
    predicate: Any | None = None,
) -> float | None:
    contexts = payload.get("context_records")
    if not isinstance(contexts, list):
        return None
    return _max_numeric(
        *(
            context.get(field)
            for context in contexts
            if isinstance(context, dict) and (predicate is None or predicate(context))
        )
    )


def _context_margin_can_block_readiness(context: dict[str, Any]) -> bool:
    if (
        context.get("source_selection_quality_regression") is True
        or "source_selection_quality_regression" in _string_list(context.get("reject_reasons"))
    ):
        return True
    return (
        context.get("training_use") == "trainable_anchor_projection_contrast"
        and context.get("source_selection_status") in {"source_selected", "source_selected_quality_regression", None}
    )


def _max_numeric(*values: Any) -> float | None:
    numeric_values = []
    for value in values:
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numeric_values:
        return None
    return max(numeric_values)


def _output_file(batch_root: Path, config: dict[str, Any]) -> Path:
    return batch_root / config["output_files"]["policy_training_readiness_review_summary"]


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": config.get("schema_version"),
        "validation": dict(config.get("validation", {})) if isinstance(config.get("validation"), dict) else {},
        "readiness_thresholds": dict(config.get("readiness_thresholds", {})),
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


def _max_counter_dict(*values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        for key, count in value.items():
            parsed = _int_value_or_default(count, 0)
            key_text = str(key)
            result[key_text] = max(result.get(key_text, 0), parsed)
    return dict(sorted(result.items()))


def _int_value(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be an integer") from exc


def _float_value(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be a number") from exc


def _optional_nonnegative_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    parsed = _float_value(value, label)
    if parsed < 0.0:
        raise ConfigError(f"{label} must be >= 0")
    return parsed


def _int_value_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
