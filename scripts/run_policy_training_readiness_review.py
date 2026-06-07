from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIG_SCHEMA_VERSION = "policy-training-readiness-review-config/v1"
SUMMARY_SCHEMA_VERSION = "policy-training-readiness-review-summary/v1"
SMOKE_SCHEMA_VERSION = "calibrated-policy-application-smoke-summary/v1"
READINESS_SCHEMA_VERSION = "channel-aware-training-readiness-summary/v1"
COVERAGE_SCHEMA_VERSION = "channel-aware-contrast-coverage-summary/v1"
CALIBRATION_SCHEMA_VERSION = "channel-aware-selection-contrast-calibration-summary/v1"
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
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    source_summaries: dict[str, Any] = {}
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
    if _fail_on_input_failure(config):
        for label, payload in (
            ("calibrated_policy_application_smoke_summary", smoke),
            ("channel_aware_training_readiness_summary", readiness),
            ("channel_aware_contrast_coverage_summary", coverage),
            ("channel_aware_selection_contrast_calibration_summary", calibration),
        ):
            if payload.get("status") == "failed":
                _append_reason(reason_codes, f"{label}_failed")

    current_git = _git_snapshot(repo_root)
    source_git_matches = [
        _inspect_git(smoke, label="calibrated_policy_application_smoke_summary", current_git=current_git, config=config, reason_codes=reason_codes),
        _inspect_git(readiness, label="channel_aware_training_readiness_summary", current_git=current_git, config=config, reason_codes=reason_codes),
        _inspect_git(coverage, label="channel_aware_contrast_coverage_summary", current_git=current_git, config=config, reason_codes=reason_codes),
        _inspect_git(calibration, label="channel_aware_selection_contrast_calibration_summary", current_git=current_git, config=config, reason_codes=reason_codes),
    ]

    review = _review_metrics(
        smoke=smoke,
        readiness=readiness,
        coverage=coverage,
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
        "calibrated_policy_application_smoke_summary_path": _display_path(smoke_path, repo_root),
        "readiness_summary_path": _display_path(readiness_path, repo_root),
        "contrast_coverage_summary_path": _display_path(coverage_path, repo_root),
        "selection_contrast_calibration_summary_path": _display_path(calibration_path, repo_root),
        "application_scope": "calibrated_policy_training_readiness_review_audit_only",
        "quality_signal_use": "calibrated_policy_target_training_contract_review_only",
        "source_summaries": source_summaries,
        "config": _public_config(config),
        "git_provenance": {
            "current": current_git,
            "calibrated_policy_application_smoke": _public_git(smoke),
            "training_readiness": _public_git(readiness),
            "contrast_coverage": _public_git(coverage),
            "selection_contrast_calibration": _public_git(calibration),
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
    validation_reason_codes: list[str],
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
    )
    fallback_or_open_grid_count = max(
        _fallback_or_open_grid_count(smoke),
        _fallback_or_open_grid_count(readiness),
        _fallback_or_open_grid_count(coverage),
        _fallback_or_open_grid_count(calibration),
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
        }
    )
    training_blockers: list[str] = []
    if validation_reason_codes:
        for reason in validation_reason_codes:
            _append_reason(training_blockers, reason)
    if thresholds["require_smoke_ready_for_training_review"] and smoke.get("recommended_next_action") != READY_SMOKE_ACTION:
        _append_reason(training_blockers, "calibrated_application_smoke_not_ready_for_training_review")
    if applied_count < thresholds["min_applied_calibrated_candidate_count"]:
        _append_reason(training_blockers, "applied_calibrated_candidate_count_below_training_threshold")
    if calibrated_rate - source_rate < thresholds["min_calibrated_selection_rate_delta"]:
        _append_reason(training_blockers, "calibrated_selection_rate_delta_below_training_threshold")
    if rejected_goal_blocked_count > thresholds["max_rejected_goal_blocked_count"]:
        _append_reason(training_blockers, "goal_blocked_candidates_excluded_from_training_positive_evidence")
    if safety_regression_count > thresholds["max_safety_regression_count"]:
        _append_reason(training_blockers, "safety_regression_blocks_training_readiness")
    if fallback_or_open_grid_count > thresholds["max_fallback_or_open_grid_count"]:
        _append_reason(training_blockers, "fallback_or_open_grid_evidence_blocks_training_readiness")
    if contract_mutations:
        _append_reason(training_blockers, "contract_mutation_blocks_training_readiness")

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
        "platform_goal_anchor_available_count": platform_goal_anchor_available_count,
        "platform_goal_unresolved_count": platform_goal_unresolved_count,
        "platform_goal_feasibility_class_counts": platform_goal_class_counts,
        "safety_regression_count": safety_regression_count,
        "fallback_or_open_grid_count": fallback_or_open_grid_count,
        "training_positive_candidate_count": applied_count,
        "excluded_candidate_count": excluded_candidate_count,
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


def _inspect_git(
    payload: dict[str, Any],
    *,
    label: str,
    current_git: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> bool:
    if not _require_current_git_match(config):
        return True
    git = payload.get("git_provenance") if isinstance(payload.get("git_provenance"), dict) else {}
    source_current = git.get("current") if isinstance(git.get("current"), dict) else {}
    source_matches = True
    if source_current and not _git_snapshots_match(source_current, current_git):
        _append_reason(reason_codes, "current_git_provenance_mismatch")
        _append_reason(reason_codes, f"{label}_current_git_provenance_mismatch")
        source_matches = False
    for key in (
        "current_matches_sources",
        "current_matches_application",
        "current_matches_robustness",
        "current_matches_batch",
        "runs_match_batch",
    ):
        if git.get(key) is False:
            _append_reason(reason_codes, "git_provenance_mismatch")
            _append_reason(reason_codes, f"{label}_git_provenance_mismatch")
            source_matches = False
    return bool(source_current) and source_matches


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


def _output_file(batch_root: Path, config: dict[str, Any]) -> Path:
    return batch_root / config["output_files"]["policy_training_readiness_review_summary"]


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": config.get("schema_version"),
        "validation": dict(config.get("validation", {})) if isinstance(config.get("validation"), dict) else {},
        "readiness_thresholds": dict(config.get("readiness_thresholds", {})),
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
