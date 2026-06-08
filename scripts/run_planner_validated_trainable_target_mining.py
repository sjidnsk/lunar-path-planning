from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot
from git_provenance import inspect_source_git_provenance as _inspect_source_git_provenance
from git_provenance import public_git as _public_git


CONFIG_SCHEMA_VERSION = "planner-validated-trainable-target-mining-config/v1"
SUMMARY_SCHEMA_VERSION = "planner-validated-trainable-target-mining-summary/v1"
CANDIDATE_SCHEMA_VERSION = "anchor-projection-candidate-generation-summary/v1"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mine planner-validated trainable anchor-projection targets from path feedback evidence."
    )
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--anchor-projection-candidate-generation-summary",
        help="Defaults to <batch-root>/anchor-projection-candidate-generation-summary.json.",
    )
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

    candidate_path = (
        _resolve_path(args.anchor_projection_candidate_generation_summary, repo_root)
        if args.anchor_projection_candidate_generation_summary
        else batch_root / "anchor-projection-candidate-generation-summary.json"
    )
    summary = analyze_planner_validated_trainable_target_mining(
        batch_root=batch_root,
        candidate_path=candidate_path,
        config=config,
        repo_root=repo_root,
    )
    output_file = batch_root / config["output_files"]["planner_validated_trainable_target_mining_summary"]
    print(
        json.dumps(
            {
                "status": "config validated" if summary["status"] == "passed" else "validation failed",
                "batch_root": _display_path(batch_root, repo_root),
                "reason_codes": summary["reason_codes"],
                "planner_validated_trainable_target_count": summary[
                    "planner_validated_trainable_target_count"
                ],
                "nontrainable_blocked_target_count": summary["nontrainable_blocked_target_count"],
                "next_required_change": summary["next_required_change"],
                "planner_validated_trainable_target_mining_summary": _display_path(
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


def analyze_planner_validated_trainable_target_mining(
    *,
    batch_root: Path,
    candidate_path: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    candidate = _load_source(
        candidate_path,
        expected_schema=CANDIDATE_SCHEMA_VERSION,
        label="anchor_projection_candidate_generation_summary",
        reason_codes=reason_codes,
    )
    validation = config["validation"]
    if validation["fail_on_input_failure"] and candidate.get("status") == "failed":
        _append_reason(reason_codes, "anchor_projection_candidate_generation_summary_failed")

    current_git = _git_snapshot(repo_root)
    current_matches_sources = _inspect_source_git_provenance(
        candidate,
        label="anchor_projection_candidate_generation_summary",
        current_git=current_git,
        require_current_git_match=validation["require_current_git_match"],
        reason_codes=reason_codes,
        allow_dirty_current_git_match=validation.get("allow_dirty_current_git_match", False),
    )
    fallback_or_open_grid_count = _int_value(
        candidate.get("fallback_or_open_grid_count", candidate.get("open_grid_fallback_used_count"))
    )
    if validation["fail_on_fallback_or_open_grid"] and fallback_or_open_grid_count > 0:
        _append_reason(reason_codes, "fallback_or_open_grid_blocks_planner_validated_mining")
    safety_regression_count = _int_value(candidate.get("safety_regression_count"))
    if validation["fail_on_safety_regression"] and safety_regression_count > 0:
        _append_reason(reason_codes, "safety_regression_blocks_planner_validated_mining")

    contexts = [
        context for context in candidate.get("context_records", []) if isinstance(context, dict)
    ] if isinstance(candidate.get("context_records"), list) else []
    decisions = [_final_decision(context, config=config) for context in contexts]
    decision_counts = Counter(decisions)
    default_count = int(decision_counts["selected_default_contract_trainable"])
    exception_count = int(decision_counts["selected_planner_validated_distance_exception"])
    trainable_count = default_count + exception_count
    nontrainable_count = len(contexts) - trainable_count
    baselines = config["baseline_counts"]
    nontrainable_delta = nontrainable_count - _int_value(
        baselines.get("nontrainable_blocked_target_count")
    )
    alignment_gap_count = _int_value(candidate.get("candidate_contract_alignment_gap_count"))
    main_success_gate_failures: list[str] = []
    if trainable_count <= _int_value(baselines.get("planner_validated_trainable_target_count_floor")):
        main_success_gate_failures.append("planner_validated_trainable_target_count_not_increased")
    if nontrainable_count >= _int_value(baselines.get("nontrainable_blocked_target_count")):
        main_success_gate_failures.append("nontrainable_blocked_target_count_not_reduced")
    if alignment_gap_count > 0:
        main_success_gate_failures.append("candidate_contract_alignment_gap_count_nonzero")

    next_required_change = (
        "source_selection_or_target_contract_change_required"
        if main_success_gate_failures
        else None
    )
    recommended_blockers = (
        []
        if not main_success_gate_failures
        else ["anchor_projection_nontrainable_contexts_remain"]
    )
    status = "failed" if reason_codes else "passed"
    final_decision_records = [
        _decision_record(context, decision=decision)
        for context, decision in zip(contexts, decisions)
    ]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason_codes": list(reason_codes),
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "batch_root": _display_path(batch_root, repo_root),
        "source_summaries": {
            "anchor_projection_candidate_generation_summary": {
                "path": _display_path(candidate_path, repo_root),
                "exists": candidate_path.is_file(),
                "schema_version": candidate.get("schema_version"),
                "status": candidate.get("status"),
            }
        },
        "git_provenance": {
            "current": current_git,
            "anchor_projection_candidate_generation": _public_git(candidate),
            "current_matches_sources": current_matches_sources,
        },
        "current_git_provenance_mismatch_count": int("current_git_provenance_mismatch" in reason_codes),
        "git_provenance_mismatch_count": int("git_provenance_mismatch" in reason_codes),
        "planner_validated_trainable_target_count": trainable_count,
        "default_contract_trainable_target_count": default_count,
        "planner_validated_distance_exception_count": exception_count,
        "nontrainable_blocked_target_count": nontrainable_count,
        "nontrainable_blocked_target_count_delta": nontrainable_delta,
        "distance_contract_blocked_count": int(decision_counts["rejected_distance_contract"]),
        "source_selection_not_selected_count": int(decision_counts["rejected_not_source_selected"]),
        "quality_regression_rejected_count": int(decision_counts["rejected_quality_regression"]),
        "not_ppo_consumable_rejected_count": int(decision_counts["rejected_not_ppo_consumable"]),
        "final_decision_counts": dict(sorted(decision_counts.items())),
        "final_decision_records": final_decision_records,
        "candidate_contract_alignment_gap_count": alignment_gap_count,
        "main_success_gate_failures": main_success_gate_failures,
        "next_required_change": next_required_change,
        "readiness_impact": {
            "recommended_training_blockers": recommended_blockers,
            "recommended_training_readiness_status": (
                "needs_training_contract_refinement"
                if recommended_blockers
                else "planner_validated_trainable_targets_available_for_limited_dry_run_review"
            ),
            "summary_passed_is_not_ppo_readiness": True,
        },
        "fallback_or_open_grid_count": fallback_or_open_grid_count,
        "safety_regression_count": safety_regression_count,
        "runs_training": False,
        "audit_only": True,
        "no_ppo_training": True,
        "does_not_modify_default_astar": True,
        "does_not_modify_ppo": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "does_not_relax_default_distance_contract": True,
        "planner_validated_distance_exception_is_opt_in": True,
        "no_ackermann_feasible_trajectory_claim": True,
        "non_goals": list(config.get("non_goals", [])),
    }


def _final_decision(context: dict[str, Any], *, config: dict[str, Any]) -> str:
    if _selected_default_contract_trainable(context):
        return "selected_default_contract_trainable"
    if _quality_regression(context):
        return "rejected_quality_regression"
    if _selected_planner_validated_distance_exception(context, config=config):
        return "selected_planner_validated_distance_exception"
    if _distance_contract_blocked(context, config=config):
        return "rejected_distance_contract"
    if not _source_selected(context):
        return "rejected_not_source_selected"
    if not _ppo_consumable_action(context):
        return "rejected_not_ppo_consumable"
    return "rejected_not_ppo_consumable"


def _selected_default_contract_trainable(context: dict[str, Any]) -> bool:
    return (
        context.get("trainable") is True
        and _source_selected(context)
        and _ppo_consumable_action(context)
        and _contract_safe(context)
        and _feasible_no_replan(context)
        and not _quality_regression(context)
    )


def _selected_planner_validated_distance_exception(
    context: dict[str, Any],
    *,
    config: dict[str, Any],
) -> bool:
    thresholds = config["thresholds"]
    return (
        config["mining"].get("allow_planner_validated_distance_exception") is True
        and _source_selected(context)
        and _ppo_consumable_action(context)
        and _feasible_no_replan(context)
        and not _quality_regression(context)
        and _planner_exception_marker(context)
        and _within_distance(
            context,
            max_cells=thresholds["max_planner_validated_distance_cells"],
            max_m=thresholds["max_planner_validated_distance_m"],
        )
        and _within_quality_regression(context, thresholds=thresholds)
    )


def _distance_contract_blocked(context: dict[str, Any], *, config: dict[str, Any]) -> bool:
    thresholds = config["thresholds"]
    if _within_distance(
        context,
        max_cells=thresholds["max_trainable_projection_distance_cells"],
        max_m=thresholds["max_trainable_projection_distance_m"],
    ):
        return False
    if (
        config["mining"].get("allow_planner_validated_distance_exception") is True
        and _within_distance(
            context,
            max_cells=thresholds["max_planner_validated_distance_cells"],
            max_m=thresholds["max_planner_validated_distance_m"],
        )
    ):
        return False
    return True


def _within_distance(context: dict[str, Any], *, max_cells: int, max_m: float) -> bool:
    distance_cells = _float_value(context.get("projection_distance_cells"))
    distance_m = _float_value(context.get("projection_distance_m"))
    return distance_cells is not None and distance_m is not None and distance_cells <= max_cells and distance_m <= max_m


def _within_quality_regression(context: dict[str, Any], *, thresholds: dict[str, Any]) -> bool:
    path_margin = _float_value(context.get("source_selection_path_cost_margin_vs_best_alternative"))
    risk_margin = _float_value(context.get("source_selection_risk_margin_vs_best_alternative"))
    max_path = thresholds.get("max_source_selection_path_cost_regression")
    max_risk = thresholds.get("max_source_selection_risk_regression")
    if max_path is not None and path_margin is not None and path_margin > float(max_path):
        return False
    if max_risk is not None and risk_margin is not None and risk_margin > float(max_risk):
        return False
    return True


def _source_selected(context: dict[str, Any]) -> bool:
    return str(context.get("source_selection_status") or "").startswith("source_selected")


def _quality_regression(context: dict[str, Any]) -> bool:
    reasons = _string_list(context.get("reject_reasons"))
    return (
        context.get("source_selection_quality_regression") is True
        or str(context.get("source_selection_status") or "") == "source_selected_quality_regression"
        or "source_selection_quality_regression" in reasons
    )


def _ppo_consumable_action(context: dict[str, Any]) -> bool:
    if context.get("ppo_consumable_action") is True:
        return True
    generation = context.get("candidate_generation")
    if isinstance(generation, dict) and generation.get("ppo_consumable_action") is True:
        return True
    gate = context.get("trainability_gate")
    return isinstance(gate, dict) and gate.get("ppo_consumable_action") is True


def _contract_safe(context: dict[str, Any]) -> bool:
    if context.get("contract_safe") is True:
        return True
    generation = context.get("candidate_generation")
    if isinstance(generation, dict) and generation.get("contract_safe") is True:
        return True
    gate = context.get("trainability_gate")
    return isinstance(gate, dict) and gate.get("contract_safe") is True


def _planner_exception_marker(context: dict[str, Any]) -> bool:
    if context.get("planner_validated_distance_exception") is True:
        return True
    if context.get("planner_validated_exception_safe") is True:
        return True
    generation = context.get("candidate_generation")
    if isinstance(generation, dict):
        return (
            generation.get("planner_validated_distance_exception") is True
            or generation.get("planner_validated_exception_safe") is True
        )
    return False


def _feasible_no_replan(context: dict[str, Any]) -> bool:
    reachable = context.get("reachable")
    replan = context.get("replan_required")
    return (reachable is not False) and (replan is not True)


def _decision_record(context: dict[str, Any], *, decision: str) -> dict[str, Any]:
    return {
        "context_id": context.get("context_id"),
        "context_id_schema_version": context.get("context_id_schema_version"),
        "context_id_source": context.get("context_id_source"),
        "legacy_identity_fallback_used": context.get("legacy_identity_fallback_used"),
        "scenario_id": context.get("scenario_id"),
        "scenario_group": context.get("scenario_group"),
        "scenario_seed": context.get("scenario_seed"),
        "scenario_variant_id": context.get("scenario_variant_id"),
        "diagnostic_profile": context.get("diagnostic_profile"),
        "planning_backend": context.get("planning_backend"),
        "top_k": context.get("top_k"),
        "run_id": context.get("run_id"),
        "source_action_index": context.get("source_action_index"),
        "policy_target_cell": context.get("policy_target_cell"),
        "execution_goal_cell": context.get("execution_goal_cell"),
        "final_training_decision": decision,
        "projection_distance_cells": context.get("projection_distance_cells"),
        "projection_distance_m": context.get("projection_distance_m"),
        "source_selection_status": context.get("source_selection_status"),
        "target_binding_mode": context.get("target_binding_mode"),
        "ppo_consumable_action": _ppo_consumable_action(context),
        "contract_safe": _contract_safe(context),
        "planner_validated_distance_exception": _planner_exception_marker(context),
        "source_selection_quality_regression": _quality_regression(context),
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
    for section in ("output_files", "validation", "baseline_counts", "thresholds", "mining"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    if not payload["output_files"].get("planner_validated_trainable_target_mining_summary"):
        raise ConfigError("output_files.planner_validated_trainable_target_mining_summary is required")
    return payload


def _load_source(
    path: Path,
    *,
    expected_schema: str,
    label: str,
    reason_codes: list[str],
) -> dict[str, Any]:
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
    if payload.get("schema_version") != expected_schema:
        _append_reason(reason_codes, f"{label}_schema_version_mismatch")
    return payload


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


if __name__ == "__main__":
    raise SystemExit(main())
