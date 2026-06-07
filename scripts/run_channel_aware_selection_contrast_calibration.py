from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIG_SCHEMA_VERSION = "channel-aware-selection-contrast-calibration-config/v1"
SUMMARY_SCHEMA_VERSION = "channel-aware-selection-contrast-calibration-summary/v1"
EVIDENCE_SCHEMA_VERSION = "channel-aware-policy-target-selection-evidence-summary/v1"
SELECTION_COMPARISON_SCHEMA_VERSION = "policy-decision-selection-comparison-summary/v1"
CHANNEL_AUDIT_SCHEMA_VERSION = "channel-aware-decision-audit/v1"
SUBMODULES = ("dev-platform-constraints", "model-explorer", "path-planner")
SAFETY_REGRESSION_REASONS = {
    "safety_regression",
    "tracking_safety_violation",
    "collision",
    "collision_regression",
    "motion_diagnostic_regression",
}
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
        description="Calibrate opt-in channel-aware policy target selection contrast from audit records."
    )
    parser.add_argument("--batch-root", required=True, help="Batch root containing channel-aware audit summaries.")
    parser.add_argument(
        "--policy-target-evidence-summary",
        help="channel-aware-policy-target-selection-evidence-summary/v1 JSON. Defaults to <batch-root>/channel-aware-policy-target-selection-evidence-summary.json.",
    )
    parser.add_argument(
        "--selection-comparison-summary",
        help="policy-decision-selection-comparison-summary/v1 JSON. Defaults to <batch-root>/policy-decision-selection-comparison-summary.json.",
    )
    parser.add_argument("--config", required=True, help="Selection contrast calibration config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned output paths.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without writing outputs.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    batch_root = _resolve_path(args.batch_root, repo_root)
    evidence_path = (
        _resolve_path(args.policy_target_evidence_summary, repo_root)
        if args.policy_target_evidence_summary
        else batch_root / "channel-aware-policy-target-selection-evidence-summary.json"
    )
    comparison_path = (
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

    summary = analyze_selection_contrast_calibration(
        batch_root=batch_root,
        evidence_path=evidence_path,
        comparison_path=comparison_path,
        config=config,
        repo_root=repo_root,
    )
    output_file = _output_file(batch_root, config)
    validation_message = {
        "status": "config validated" if summary["status"] == "passed" else "validation failed",
        "batch_root": _display_path(batch_root, repo_root),
        "policy_target_evidence_summary": _display_path(evidence_path, repo_root),
        "selection_comparison_summary": _display_path(comparison_path, repo_root),
        "config": _display_path(config_path, repo_root),
        "reason_codes": summary["reason_codes"],
        "selected_candidate_changed_rate": summary["selected_candidate_changed_rate"],
        "changed_scenario_ids": summary["changed_scenario_ids"],
        "selection_contrast_calibration_summary": _display_path(output_file, repo_root),
    }
    print(json.dumps(validation_message, ensure_ascii=False))

    if args.validate_only or args.dry_run:
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "dry-run",
                        "would_write": {
                            "selection_contrast_calibration_summary": _display_path(output_file, repo_root),
                        },
                        "recommended_training_readiness_action": summary[
                            "recommended_training_readiness_action"
                        ],
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
                "selection_contrast_calibration_summary": _display_path(output_file, repo_root),
                "selected_candidate_changed_rate": summary["selected_candidate_changed_rate"],
                "recommended_training_readiness_action": summary["recommended_training_readiness_action"],
                "failure_reason_code_counts": summary["failure_reason_code_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def analyze_selection_contrast_calibration(
    *,
    batch_root: Path,
    evidence_path: Path,
    comparison_path: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    global_reason_codes: list[str] = []
    source_summaries: dict[str, Any] = {}
    evidence = _load_source(
        evidence_path,
        label="channel_aware_policy_target_selection_evidence_summary",
        expected_schema=EVIDENCE_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=global_reason_codes,
        source_summaries=source_summaries,
    )
    comparison = _load_source(
        comparison_path,
        label="policy_decision_selection_comparison_summary",
        expected_schema=SELECTION_COMPARISON_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=global_reason_codes,
        source_summaries=source_summaries,
    )
    if _fail_on_input_failure(config):
        for label, payload in (
            ("channel_aware_policy_target_selection_evidence_summary", evidence),
            ("policy_decision_selection_comparison_summary", comparison),
        ):
            if payload.get("status") == "failed":
                _append_reason(global_reason_codes, f"{label}_failed")

    current_git = _git_snapshot(repo_root)
    _inspect_git(evidence, label="channel_aware_policy_target_selection_evidence_summary", current_git=current_git, config=config, reason_codes=global_reason_codes)
    _inspect_git(comparison, label="policy_decision_selection_comparison_summary", current_git=current_git, config=config, reason_codes=global_reason_codes)

    audit = _channel_audit(comparison, reason_codes=global_reason_codes)
    records = [_normalize_record(record) for record in audit.get("records", []) if isinstance(record, dict)]
    calibration = _calibrate_selection(records, evidence=evidence, config=config)
    status = "failed" if global_reason_codes else "passed"
    failure_reason_counts = Counter(global_reason_codes)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(global_reason_codes),
        "failure_reason_code_counts": dict(sorted(failure_reason_counts.items())),
        "batch_root": _display_path(batch_root, repo_root),
        "application_scope": "channel_aware_selection_contrast_calibration_audit_only",
        "quality_signal_use": "opt_in_selection_contrast_calibration_only",
        "source_summaries": source_summaries,
        "config": _public_config(config),
        "git_provenance": {
            "current": current_git,
            "policy_target_evidence": _public_git(evidence),
            "selection_comparison": _public_git(comparison),
        },
        **calibration,
        "runs_training": False,
        "channel_aware_backend_opt_in": True,
        "does_not_modify_default_astar": True,
        "does_not_modify_ppo": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "does_not_modify_path_planner_route_contract": True,
        "does_not_modify_model_explorer_contract": True,
        "does_not_modify_path_planner_sidecar_contract": True,
        "no_gcs_control_point_candidate_as_default_execution_trajectory": True,
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
        output_files.get("selection_contrast_calibration_summary")
    ):
        raise ConfigError("output_files.selection_contrast_calibration_summary must be a non-empty string")
    raw_calibration = payload.get("calibration")
    if not isinstance(raw_calibration, dict):
        raise ConfigError("calibration must be an object")
    eligible = raw_calibration.get("eligible_recommendations", ["keep"])
    if not isinstance(eligible, list) or not eligible:
        raise ConfigError("calibration.eligible_recommendations must be a non-empty array")
    calibration = {
        "eligible_recommendations": [str(item) for item in eligible],
        "channel_cost_improvement_weight": _float_value(
            raw_calibration.get("channel_cost_improvement_weight", 1.0),
            "calibration.channel_cost_improvement_weight",
        ),
        "high_cost_exposure_improvement_weight": _float_value(
            raw_calibration.get("high_cost_exposure_improvement_weight", 1.0),
            "calibration.high_cost_exposure_improvement_weight",
        ),
        "path_cost_tradeoff_weight": _float_value(
            raw_calibration.get("path_cost_tradeoff_weight", 1.0),
            "calibration.path_cost_tradeoff_weight",
        ),
        "min_contrast_score": _float_value(
            raw_calibration.get("min_contrast_score", 0.0),
            "calibration.min_contrast_score",
        ),
    }
    for key in (
        "channel_cost_improvement_weight",
        "high_cost_exposure_improvement_weight",
        "path_cost_tradeoff_weight",
    ):
        if calibration[key] < 0.0:
            raise ConfigError(f"calibration.{key} must be nonnegative")
    config = dict(payload)
    config["calibration"] = calibration
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


def _channel_audit(comparison: dict[str, Any], *, reason_codes: list[str]) -> dict[str, Any]:
    audit = comparison.get("channel_aware_decision_audit")
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


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    comparison = record.get("comparison") if isinstance(record.get("comparison"), dict) else {}
    reason_codes = _string_list(record.get("reason_codes", []))
    if bool(record.get("path_cost_tradeoff", False)) and "path_cost_tradeoff" not in reason_codes:
        reason_codes.append("path_cost_tradeoff")
    return {
        "pair_key": record.get("pair_key"),
        "scenario_id": str(record.get("scenario_id", "")),
        "scenario_group": str(record.get("scenario_group", "unknown")),
        "action_index": _safe_int(record.get("action_index"), 0),
        "cell": _list_value(record.get("cell")),
        "astar_selected_cell": _list_value(record.get("astar_selected_cell")),
        "source_channel_aware_selected_cell": _list_value(record.get("channel_aware_selected_cell")),
        "source_selected_candidate_changed": bool(record.get("selected_candidate_changed", False)),
        "recommendation": str(record.get("recommendation", "needs_more_evidence")),
        "blocker_reason": str(record.get("blocker_reason")) if record.get("blocker_reason") is not None else None,
        "path_cost_tradeoff": bool(record.get("path_cost_tradeoff", False)) or "path_cost_tradeoff" in reason_codes,
        "reason_codes": reason_codes,
        "path_cost_delta": _safe_float_or_none(comparison.get("path_cost_delta")),
        "channel_cost_delta": _safe_float_or_none(comparison.get("channel_cost_delta")),
        "high_cost_exposure_delta": _safe_float_or_none(comparison.get("high_cost_exposure_delta")),
        "platform_goal_classification": _platform_goal_failure_class(record),
        "platform_goal_feasibility": dict(record.get("platform_goal_feasibility"))
        if isinstance(record.get("platform_goal_feasibility"), dict)
        else {},
    }


def _calibrate_selection(
    records: list[dict[str, Any]],
    *,
    evidence: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    context_records = _records_by_context(records)
    calibrated_records = []
    changed_scenario_ids: set[str] = set()
    changed_context_count = 0
    keep_selected_count = 0
    safety_regression_count = 0
    for context_key, bucket in sorted(context_records.items()):
        selected = _calibrated_context_selection(bucket, config=config)
        if selected["selected_candidate_changed"]:
            changed_context_count += 1
            changed_scenario_ids.add(selected["scenario_id"])
            if selected["recommendation"] == "keep":
                keep_selected_count += 1
        if selected["safety_regression"]:
            safety_regression_count += 1
        calibrated_records.append(selected)

    record_count = len(records)
    goal_blocked_count = sum(1 for record in records if _has_reason(record, "goal_blocked"))
    platform_class_counts = Counter(
        record["platform_goal_classification"]
        for record in records
        if record.get("platform_goal_classification") is not None
    )
    platform_goal_contract_mismatch_count = sum(
        platform_class_counts[classification]
        for classification in PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES
    )
    platform_goal_anchor_available_count = sum(
        1 for record in records if _platform_goal_anchor_available(record)
    )
    platform_goal_unresolved_count = platform_class_counts["unknown_contract_mismatch"]
    same_as_baseline_count = sum(1 for record in records if _has_reason(record, "same_as_baseline"))
    blocked_candidate_count = sum(1 for record in records if _is_blocked(record))
    path_cost_tradeoff_count = sum(1 for record in records if record["path_cost_tradeoff"])
    channel_deltas = [record["channel_cost_delta"] for record in records if record["channel_cost_delta"] is not None]
    high_cost_deltas = [
        record["high_cost_exposure_delta"]
        for record in records
        if record["high_cost_exposure_delta"] is not None
    ]
    selected_candidate_changed_rate = _rate(changed_context_count, len(context_records))
    recommendation = _training_readiness_action(
        selected_candidate_changed_rate=selected_candidate_changed_rate,
        safety_regression_count=safety_regression_count,
    )
    return {
        "source_selected_candidate_changed_rate": _safe_float(
            evidence.get("selected_candidate_changed_rate"),
            0.0,
        ),
        "source_supports_policy_target_selection_improvement_claim": bool(
            evidence.get("supports_policy_target_selection_improvement_claim", False)
        ),
        "selection_context_count": len(context_records),
        "selected_candidate_changed_count": changed_context_count,
        "selected_candidate_changed_rate": selected_candidate_changed_rate,
        "changed_scenario_ids": sorted(changed_scenario_ids),
        "keep_selected_candidate_count": keep_selected_count,
        "keep_selected_candidate_rate": _rate(keep_selected_count, changed_context_count),
        "goal_blocked_count": goal_blocked_count,
        "platform_goal_contract_mismatch_count": platform_goal_contract_mismatch_count,
        "platform_goal_anchor_available_count": platform_goal_anchor_available_count,
        "platform_goal_unresolved_count": platform_goal_unresolved_count,
        "platform_goal_feasibility_class_counts": dict(sorted(platform_class_counts.items())),
        "same_as_baseline_count": same_as_baseline_count,
        "blocked_candidate_count": blocked_candidate_count,
        "blocked_candidate_rate": _rate(blocked_candidate_count, record_count),
        "path_cost_tradeoff_count": path_cost_tradeoff_count,
        "channel_cost_delta_stats": _numeric_stats(channel_deltas),
        "high_cost_exposure_delta_stats": _numeric_stats(high_cost_deltas),
        "safety_regression_count": safety_regression_count,
        "recommended_training_readiness_action": recommendation,
        "policy_target_selection_improvement_claimed_without_evidence": False,
        "calibrated_selection_records": calibrated_records,
    }


def _records_by_context(records: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault((record["pair_key"], record["scenario_id"]), []).append(record)
    return buckets


def _calibrated_context_selection(bucket: list[dict[str, Any]], *, config: dict[str, Any]) -> dict[str, Any]:
    astar_selected_cell = _first_nonempty(record["astar_selected_cell"] for record in bucket)
    eligible = [_scored_record(record, config=config) for record in bucket if _eligible(record, config=config)]
    eligible = [record for record in eligible if record["contrast_score"] >= config["calibration"]["min_contrast_score"]]
    if eligible:
        selected = sorted(
            eligible,
            key=lambda item: (
                -item["contrast_score"],
                item["path_cost_delta"] if item["path_cost_delta"] is not None else 0.0,
                item["action_index"],
                item["cell"],
            ),
        )[0]
        calibrated_cell = selected["cell"]
        selection_reason = "channel_quality_contrast_selected"
    else:
        selected = None
        calibrated_cell = astar_selected_cell
        selection_reason = "no_eligible_channel_quality_contrast_candidate"
    changed = bool(calibrated_cell and astar_selected_cell and calibrated_cell != astar_selected_cell)
    safety_regression = bool(selected and _has_safety_regression(selected))
    scenario_id = bucket[0]["scenario_id"] if bucket else ""
    pair_key = bucket[0]["pair_key"] if bucket else None
    return {
        "pair_key": pair_key,
        "scenario_id": scenario_id,
        "astar_selected_cell": astar_selected_cell,
        "calibrated_channel_aware_selected_cell": calibrated_cell,
        "selected_candidate_changed": changed,
        "selection_reason": selection_reason,
        "selected_action_index": None if selected is None else selected["action_index"],
        "selected_candidate_score": None if selected is None else selected["contrast_score"],
        "recommendation": None if selected is None else selected["recommendation"],
        "path_cost_tradeoff": bool(selected and selected["path_cost_tradeoff"]),
        "safety_regression": safety_regression,
    }


def _scored_record(record: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
    calibration = config["calibration"]
    channel_improvement = max(-float(record["channel_cost_delta"] or 0.0), 0.0)
    exposure_improvement = max(-float(record["high_cost_exposure_delta"] or 0.0), 0.0)
    path_tradeoff = max(float(record["path_cost_delta"] or 0.0), 0.0)
    score = (
        calibration["channel_cost_improvement_weight"] * channel_improvement
        + calibration["high_cost_exposure_improvement_weight"] * exposure_improvement
        - calibration["path_cost_tradeoff_weight"] * path_tradeoff
    )
    scored = dict(record)
    scored["contrast_score"] = float(score)
    return scored


def _eligible(record: dict[str, Any], *, config: dict[str, Any]) -> bool:
    if record["recommendation"] not in set(config["calibration"]["eligible_recommendations"]):
        return False
    if _is_blocked(record):
        return False
    if record["channel_cost_delta"] is None and record["high_cost_exposure_delta"] is None:
        return False
    return True


def _is_blocked(record: dict[str, Any]) -> bool:
    return record["recommendation"] == "reject" or _has_reason(record, "goal_blocked")


def _has_reason(record: dict[str, Any], reason: str) -> bool:
    return record.get("blocker_reason") == reason or reason in record.get("reason_codes", [])


def _platform_goal_failure_class(record: dict[str, Any]) -> str | None:
    for key in ("failure_taxonomy", "platform_goal_classification"):
        value = record.get(key)
        if isinstance(value, str) and value in PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES:
            return value
    for reason in _string_list(record.get("reason_codes", [])):
        if reason in PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES:
            return reason
    feasibility = record.get("platform_goal_feasibility")
    feasibility = feasibility if isinstance(feasibility, dict) else {}
    classification = feasibility.get("classification")
    if isinstance(classification, str) and classification in PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES:
        return classification
    return None


def _platform_goal_anchor_available(record: dict[str, Any]) -> bool:
    feasibility = record.get("platform_goal_feasibility")
    feasibility = feasibility if isinstance(feasibility, dict) else {}
    anchor = feasibility.get("nearest_inflated_passable_anchor")
    return isinstance(anchor, list) and len(anchor) == 2


def _has_safety_regression(record: dict[str, Any]) -> bool:
    return any(reason in SAFETY_REGRESSION_REASONS for reason in record.get("reason_codes", []))


def _training_readiness_action(
    *,
    selected_candidate_changed_rate: float,
    safety_regression_count: int,
) -> str:
    if safety_regression_count:
        return "block_training_readiness_until_safety_regressions_resolved"
    if selected_candidate_changed_rate > 0.0:
        return "rerun_training_readiness_gate_after_selection_contrast_calibration"
    return "needs_more_selection_contrast_before_training_readiness_gate"


def _numeric_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _output_file(batch_root: Path, config: dict[str, Any]) -> Path:
    return batch_root / config["output_files"]["selection_contrast_calibration_summary"]


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": config.get("schema_version"),
        "validation": dict(config.get("validation", {})) if isinstance(config.get("validation"), dict) else {},
        "calibration": dict(config.get("calibration", {})),
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


def _first_nonempty(values) -> list[Any]:
    for value in values:
        if value:
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


def _safe_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
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
