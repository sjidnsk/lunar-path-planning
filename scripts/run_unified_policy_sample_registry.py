from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIG_SCHEMA_VERSION = "unified-policy-sample-registry-config/v1"
SUMMARY_SCHEMA_VERSION = "unified-policy-sample-registry-summary/v1"
EXCLUSION_SCHEMA_VERSION = "unified-policy-sample-exclusion-report/v1"
MINING_SCHEMA_VERSION = "planner-validated-trainable-target-mining-summary/v1"
CANDIDATE_SCHEMA_VERSION = "anchor-projection-candidate-generation-summary/v1"
MATERIALIZATION_SCHEMA_VERSION = "planner-validated-training-input-materialization-summary/v1"
COUNTERFACTUAL_SCHEMA_VERSION = "counterfactual-preference-training-summary/v1"
COUNTERFACTUAL_EXCLUSION_SCHEMA_VERSION = "counterfactual-preference-exclusion-report/v1"

POSITIVE_DECISIONS = {
    "selected_default_contract_trainable",
    "selected_planner_validated_distance_exception",
}
RESIDUAL_SAMPLE_TYPES = {
    "boundary_negative_preference_pair",
    "blocked_target_negative_pair",
}


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a unified registry of action-label, preference, and residual boundary samples."
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

    result = build_unified_policy_sample_registry(
        batch_root=batch_root,
        config=config,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": "config validated" if result["summary"]["status"] == "passed" else "validation failed",
                "batch_root": _display_path(batch_root, repo_root),
                "reason_codes": result["summary"]["reason_codes"],
                "action_label_positive_count": result["summary"]["action_label_positive_count"],
                "pairwise_preference_signal_count": result["summary"][
                    "pairwise_preference_signal_count"
                ],
                "residual_trainable_signal_count": result["summary"][
                    "residual_trainable_signal_count"
                ],
                "unified_context_coverage_count": result["summary"][
                    "unified_context_coverage_count"
                ],
                "registry": _display_path(result["registry_path"], repo_root),
                "summary": _display_path(result["summary_path"], repo_root),
                "exclusion_report": _display_path(result["exclusion_path"], repo_root),
            },
            ensure_ascii=False,
        )
    )
    if args.validate_only or args.dry_run:
        return 1 if result["summary"]["status"] == "failed" else 0
    _write_outputs(result)
    return 1 if result["summary"]["status"] == "failed" else 0


def build_unified_policy_sample_registry(
    *,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    input_files = config["input_files"]
    mining_path = batch_root / input_files["planner_validated_trainable_target_mining_summary"]
    candidate_path = batch_root / input_files["anchor_projection_candidate_generation_summary"]
    materialization_path = batch_root / input_files[
        "planner_validated_training_input_materialization_summary"
    ]
    rollout_path = batch_root / input_files["planner_validated_rollout_episodes"]
    counterfactual_path = batch_root / input_files["counterfactual_preference_training_summary"]
    counterfactual_samples_path = batch_root / input_files[
        "counterfactual_preference_training_samples"
    ]
    counterfactual_exclusion_path = batch_root / input_files[
        "counterfactual_preference_exclusion_report"
    ]

    mining = _load_source(
        mining_path,
        expected_schema=MINING_SCHEMA_VERSION,
        label="planner_validated_mining_summary",
        reason_codes=reason_codes,
    )
    candidate = _load_source(
        candidate_path,
        expected_schema=CANDIDATE_SCHEMA_VERSION,
        label="anchor_projection_candidate_generation_summary",
        reason_codes=reason_codes,
    )
    materialization = _load_source(
        materialization_path,
        expected_schema=MATERIALIZATION_SCHEMA_VERSION,
        label="planner_validated_training_input_materialization_summary",
        reason_codes=reason_codes,
    )
    counterfactual = _load_source(
        counterfactual_path,
        expected_schema=COUNTERFACTUAL_SCHEMA_VERSION,
        label="counterfactual_preference_training_summary",
        reason_codes=reason_codes,
    )
    counterfactual_exclusion = _load_source(
        counterfactual_exclusion_path,
        expected_schema=COUNTERFACTUAL_EXCLUSION_SCHEMA_VERSION,
        label="counterfactual_preference_exclusion_report",
        reason_codes=reason_codes,
    )
    counterfactual_samples = _load_jsonl(
        counterfactual_samples_path,
        label="counterfactual_preference_training_samples",
        reason_codes=reason_codes,
    )
    rollout_line_count = _count_jsonl_lines(
        rollout_path,
        label="planner_validated_rollout_episodes",
        reason_codes=reason_codes,
    )

    validation = config["validation"]
    if validation.get("fail_on_input_failure", True):
        _require_passed(mining, "planner_validated_mining_summary", reason_codes)
        _require_passed(candidate, "anchor_projection_candidate_generation_summary", reason_codes)
        _require_passed(
            materialization,
            "planner_validated_training_input_materialization_summary",
            reason_codes,
        )
        _require_passed(counterfactual, "counterfactual_preference_training_summary", reason_codes)
        _require_passed(
            counterfactual_exclusion,
            "counterfactual_preference_exclusion_report",
            reason_codes,
        )
    if validation.get("fail_on_provenance_mismatch", True):
        if _int_value(mining.get("current_git_provenance_mismatch_count")) > 0 or _int_value(
            mining.get("git_provenance_mismatch_count")
        ) > 0:
            _append_reason(reason_codes, "current_git_provenance_mismatch")
    if validation.get("fail_on_fallback_or_open_grid", True):
        if _int_value(mining.get("fallback_or_open_grid_count")) > 0:
            _append_reason(reason_codes, "fallback_or_open_grid_blocks_unified_registry")
    if validation.get("fail_on_safety_regression", True):
        if _int_value(mining.get("safety_regression_count")) > 0:
            _append_reason(reason_codes, "safety_regression_blocks_unified_registry")
    if _int_value(candidate.get("candidate_contract_alignment_gap_count")) > 0:
        _append_reason(reason_codes, "candidate_contract_alignment_gap_count_nonzero")

    records = _merged_records(mining, candidate)
    registry: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    positive_records = [record for record in records if _decision(record) in POSITIVE_DECISIONS]
    for sample_index, record in enumerate(positive_records):
        registry.append(_action_label_positive_record(record, sample_index=sample_index))

    for sample_index, sample in enumerate(counterfactual_samples):
        registry.append(_existing_preference_record(sample, sample_index=sample_index))

    merged_by_key = {_fallback_record_key(record): record for record in records}
    boundary_sources = counterfactual_exclusion.get("excluded_records", [])
    boundary_sample_index = 0
    residual_config = config.get("residual_preference", {})
    for source in boundary_sources:
        if not isinstance(source, dict):
            continue
        if source.get("preference_decision") != "rejected_binding_or_distance_required":
            continue
        record = merged_by_key.get(_fallback_record_key(source))
        if record is None:
            _append_reason(reason_codes, "residual_source_record_missing")
            exclusions.append(
                _exclusion_record(
                    source,
                    sample_type="boundary_negative_preference_pair",
                    reason="residual_source_record_missing",
                )
            )
            continue
        sample = _residual_preference_record(
            record,
            sample_type="boundary_negative_preference_pair",
            sample_index=boundary_sample_index,
            sample_weight=float(residual_config.get("boundary_negative_weight", 0.25)),
            binding_required=True,
            hierarchical_subgoal_required=False,
            reason_codes=reason_codes,
            exclusions=exclusions,
        )
        if sample is not None:
            registry.append(sample)
            boundary_sample_index += 1

    blocked_sample_index = 0
    for record in records:
        if _decision(record) != "rejected_distance_contract":
            continue
        sample = _residual_preference_record(
            record,
            sample_type="blocked_target_negative_pair",
            sample_index=blocked_sample_index,
            sample_weight=float(residual_config.get("blocked_target_negative_weight", 1.0)),
            binding_required=False,
            hierarchical_subgoal_required=True,
            reason_codes=reason_codes,
            exclusions=exclusions,
        )
        if sample is not None:
            registry.append(sample)
            blocked_sample_index += 1

    counts = Counter(item["sample_type"] for item in registry)
    action_label_positive_count = counts["action_label_positive"]
    existing_preference_pair_count = counts["counterfactual_preference_pair"]
    boundary_negative_preference_pair_count = counts["boundary_negative_preference_pair"]
    blocked_target_negative_pair_count = counts["blocked_target_negative_pair"]
    residual_trainable_signal_count = (
        boundary_negative_preference_pair_count + blocked_target_negative_pair_count
    )
    pairwise_preference_signal_count = (
        existing_preference_pair_count + residual_trainable_signal_count
    )
    hard_positive_added_count = sum(
        1
        for item in registry
        if item["sample_type"] != "action_label_positive" and item.get("hard_positive") is True
    )
    unified_context_coverage_count = len(registry)

    if _int_value(materialization.get("input_positive_count")) != action_label_positive_count:
        _append_reason(reason_codes, "materialized_positive_count_mismatch")
    if rollout_line_count != action_label_positive_count:
        _append_reason(reason_codes, "materialized_rollout_count_mismatch")
    if _int_value(counterfactual.get("preference_pair_count")) != existing_preference_pair_count:
        _append_reason(reason_codes, "existing_preference_pair_count_mismatch")
    if _int_value(counterfactual_exclusion.get("excluded_count")) != len(boundary_sources):
        _append_reason(reason_codes, "counterfactual_exclusion_count_mismatch")

    expected = config.get("expected_counts", {})
    _check_expected_count(
        reason_codes,
        expected,
        "action_label_positive_count",
        action_label_positive_count,
    )
    _check_expected_count(
        reason_codes,
        expected,
        "existing_preference_pair_count",
        existing_preference_pair_count,
    )
    _check_expected_count(
        reason_codes,
        expected,
        "boundary_negative_preference_pair_count",
        boundary_negative_preference_pair_count,
    )
    _check_expected_count(
        reason_codes,
        expected,
        "blocked_target_negative_pair_count",
        blocked_target_negative_pair_count,
    )
    _check_expected_count(
        reason_codes,
        expected,
        "residual_trainable_signal_count",
        residual_trainable_signal_count,
    )
    _check_expected_count(
        reason_codes,
        expected,
        "pairwise_preference_signal_count",
        pairwise_preference_signal_count,
    )
    _check_expected_count(
        reason_codes,
        expected,
        "unified_context_coverage_count",
        unified_context_coverage_count,
    )
    _check_expected_count(
        reason_codes,
        expected,
        "hard_positive_added_count",
        hard_positive_added_count,
    )

    output_files = config["output_files"]
    registry_path = batch_root / output_files["registry"]
    summary_path = batch_root / output_files["summary"]
    exclusion_path = batch_root / output_files["exclusion_report"]
    status = "failed" if reason_codes else "passed"
    generated_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "batch_root": _display_path(batch_root, repo_root),
        "source_summaries": {
            "planner_validated_trainable_target_mining_summary": _source_descriptor(
                mining_path,
                mining,
                repo_root,
            ),
            "anchor_projection_candidate_generation_summary": _source_descriptor(
                candidate_path,
                candidate,
                repo_root,
            ),
            "planner_validated_training_input_materialization_summary": _source_descriptor(
                materialization_path,
                materialization,
                repo_root,
            ),
            "planner_validated_rollout_episodes": {
                "path": _display_path(rollout_path, repo_root),
                "exists": rollout_path.is_file(),
                "line_count": rollout_line_count,
            },
            "counterfactual_preference_training_summary": _source_descriptor(
                counterfactual_path,
                counterfactual,
                repo_root,
            ),
            "counterfactual_preference_training_samples": {
                "path": _display_path(counterfactual_samples_path, repo_root),
                "exists": counterfactual_samples_path.is_file(),
                "line_count": len(counterfactual_samples),
            },
            "counterfactual_preference_exclusion_report": _source_descriptor(
                counterfactual_exclusion_path,
                counterfactual_exclusion,
                repo_root,
            ),
        },
        "action_label_positive_count": action_label_positive_count,
        "default_contract_action_label_positive_count": sum(
            1 for item in registry if item.get("final_training_decision") == "selected_default_contract_trainable"
        ),
        "planner_validated_exception_action_label_positive_count": sum(
            1
            for item in registry
            if item.get("final_training_decision") == "selected_planner_validated_distance_exception"
        ),
        "existing_preference_pair_count": existing_preference_pair_count,
        "boundary_negative_preference_pair_count": boundary_negative_preference_pair_count,
        "blocked_target_negative_pair_count": blocked_target_negative_pair_count,
        "residual_trainable_signal_count": residual_trainable_signal_count,
        "pairwise_preference_signal_count": pairwise_preference_signal_count,
        "unified_context_coverage_count": unified_context_coverage_count,
        "hard_positive_added_count": hard_positive_added_count,
        "planner_validated_trainable_target_count": _int_value(
            mining.get("planner_validated_trainable_target_count")
        ),
        "does_not_add_rollout_hard_positive": True,
        "registry": _display_path(registry_path, repo_root),
        "exclusion_report": _display_path(exclusion_path, repo_root),
        "runs_training": False,
        "dry_run_only": False,
        "publishes_checkpoint": False,
        "performance_claimed": False,
        "does_not_modify_default_astar": True,
        "does_not_modify_ppo": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "does_not_relax_default_distance_contract": True,
        "no_ackermann_feasible_trajectory_claim": True,
        "non_goals": list(config.get("non_goals", [])),
    }
    exclusion_report = {
        "schema_version": EXCLUSION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "excluded_count": len(exclusions),
        "excluded_reason_counts": dict(sorted(Counter(item["reason"] for item in exclusions).items())),
        "excluded_records": exclusions,
    }
    return {
        "registry": registry,
        "summary": summary,
        "exclusion_report": exclusion_report,
        "registry_path": registry_path,
        "summary_path": summary_path,
        "exclusion_path": exclusion_path,
    }


def _action_label_positive_record(record: dict[str, Any], *, sample_index: int) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "sample_index": sample_index,
        "sample_type": "action_label_positive",
        "training_signal_type": "rollout_action_label",
        "context_id": _context_id(record),
        "context_id_schema_version": record.get("context_id_schema_version"),
        "context_id_source": record.get("context_id_source"),
        "legacy_identity_fallback_used": bool(record.get("legacy_identity_fallback_used")),
        "run_id": record.get("run_id"),
        "scenario_id": record.get("scenario_id"),
        "scenario_group": record.get("scenario_group"),
        "scenario_seed": record.get("scenario_seed"),
        "scenario_variant_id": record.get("scenario_variant_id"),
        "diagnostic_profile": record.get("diagnostic_profile"),
        "planning_backend": record.get("planning_backend"),
        "top_k": record.get("top_k"),
        "source_action_index": _int_value(record.get("source_action_index")),
        "policy_target_cell": _list_cell(record.get("policy_target_cell")),
        "execution_goal_cell": _list_cell(record.get("execution_goal_cell")),
        "final_training_decision": _decision(record),
        "target_binding_mode": record.get("target_binding_mode"),
        "ppo_consumable_action": bool(record.get("ppo_consumable_action")),
        "contract_safe": bool(record.get("contract_safe")),
        "planner_validated_distance_exception": bool(record.get("planner_validated_distance_exception")),
        "projection_distance_cells": _float_value(record.get("projection_distance_cells")),
        "projection_distance_m": _float_value(record.get("projection_distance_m")),
        "hard_positive": True,
        "binding_required": False,
        "hierarchical_subgoal_required": False,
    }


def _existing_preference_record(sample: dict[str, Any], *, sample_index: int) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "sample_index": sample_index,
        "sample_type": "counterfactual_preference_pair",
        "training_signal_type": "pairwise_preference",
        "context_id": _sample_context_id(sample),
        "context_id_schema_version": sample.get("context_id_schema_version"),
        "context_id_source": sample.get("context_id_source"),
        "legacy_identity_fallback_used": bool(sample.get("legacy_identity_fallback_used")),
        "run_id": sample.get("run_id"),
        "scenario_id": sample.get("scenario_id"),
        "scenario_group": sample.get("scenario_group"),
        "scenario_seed": sample.get("scenario_seed"),
        "scenario_variant_id": sample.get("scenario_variant_id"),
        "diagnostic_profile": sample.get("diagnostic_profile"),
        "planning_backend": sample.get("planning_backend"),
        "top_k": sample.get("top_k"),
        "preference_decision": sample.get("preference_decision"),
        "sample_weight": float(sample.get("sample_weight", 1.0)),
        "preferred": sample.get("selected"),
        "alternative": sample.get("alternative"),
        "margins": sample.get("margins", {}),
        "global_features": sample.get("global_features", _global_features()),
        "candidate_missing_indicators": sample.get(
            "candidate_missing_indicators",
            _missing_indicators(),
        ),
        "hard_positive": False,
        "binding_required": False,
        "hierarchical_subgoal_required": False,
    }


def _residual_preference_record(
    record: dict[str, Any],
    *,
    sample_type: str,
    sample_index: int,
    sample_weight: float,
    binding_required: bool,
    hierarchical_subgoal_required: bool,
    reason_codes: list[str],
    exclusions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    missing_reason = _residual_missing_reason(record)
    if missing_reason is not None:
        _append_reason(reason_codes, missing_reason)
        exclusions.append(_exclusion_record(record, sample_type=sample_type, reason=missing_reason))
        return None
    selected_cell = _list_cell(record.get("selected_cell"))
    policy_target_cell = _list_cell(record.get("policy_target_cell"))
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "sample_index": sample_index,
        "sample_type": sample_type,
        "training_signal_type": "pairwise_preference",
        "context_id": _context_id(record),
        "context_id_schema_version": record.get("context_id_schema_version"),
        "context_id_source": record.get("context_id_source"),
        "legacy_identity_fallback_used": bool(record.get("legacy_identity_fallback_used")),
        "run_id": record.get("run_id"),
        "scenario_id": record.get("scenario_id"),
        "scenario_group": record.get("scenario_group"),
        "scenario_seed": record.get("scenario_seed"),
        "scenario_variant_id": record.get("scenario_variant_id"),
        "diagnostic_profile": record.get("diagnostic_profile"),
        "planning_backend": record.get("planning_backend"),
        "top_k": record.get("top_k"),
        "preference_decision": sample_type,
        "sample_weight": sample_weight,
        "preferred": {
            "action_index": _int_value(record.get("selected_action_index")),
            "cell": selected_cell,
            "path_cost": _float_value(record.get("selected_candidate_path_cost")),
            "risk": _float_value(record.get("selected_candidate_risk")),
            "utility": _float_value(record.get("selected_candidate_utility")),
            "candidate_features": _candidate_features(
                cell=selected_cell,
                utility=_float_value(record.get("selected_candidate_utility")),
                path_cost=_float_value(record.get("selected_candidate_path_cost")),
                risk=_float_value(record.get("selected_candidate_risk")),
            ),
        },
        "alternative": {
            "source_action_index": _int_value(record.get("source_action_index")),
            "policy_target_cell": policy_target_cell,
            "execution_goal_cell": _list_cell(record.get("execution_goal_cell")),
            "target_binding_mode": record.get("target_binding_mode"),
            "ppo_consumable_action": bool(record.get("ppo_consumable_action")),
            "contract_safe": bool(record.get("contract_safe")),
            "planner_validated_distance_exception": bool(
                record.get("planner_validated_distance_exception")
            ),
            "path_cost": _float_value(record.get("projected_candidate_path_cost")),
            "risk": _float_value(record.get("projected_candidate_risk")),
            "utility": _float_value(record.get("projected_candidate_utility")),
            "candidate_features": _candidate_features(
                cell=policy_target_cell,
                utility=_float_value(record.get("projected_candidate_utility")),
                path_cost=_float_value(record.get("projected_candidate_path_cost")),
                risk=_float_value(record.get("projected_candidate_risk")),
            ),
        },
        "margins": {
            "projection_distance_cells": _float_value(record.get("projection_distance_cells")),
            "projection_distance_m": _float_value(record.get("projection_distance_m")),
            "path_cost_margin_vs_selected": _float_value(record.get("path_cost_margin_vs_selected")),
            "risk_margin_vs_selected": _float_value(record.get("risk_margin_vs_selected")),
        },
        "global_features": _global_features(),
        "candidate_missing_indicators": _missing_indicators(),
        "source_contract": {
            "source_selection_status": record.get("source_selection_status"),
            "final_training_decision": _decision(record),
            "not_action_label_positive": True,
            "does_not_relax_default_distance_contract": True,
        },
        "hard_positive": False,
        "binding_required": binding_required,
        "hierarchical_subgoal_required": hierarchical_subgoal_required,
    }


def _residual_missing_reason(record: dict[str, Any]) -> str | None:
    if record.get("selected_action_index") is None or _list_cell(record.get("selected_cell")) is None:
        return "selected_reference_missing"
    selected_metrics = (
        _float_value(record.get("selected_candidate_path_cost")),
        _float_value(record.get("selected_candidate_risk")),
        _float_value(record.get("selected_candidate_utility")),
    )
    alternative_metrics = (
        _float_value(record.get("projected_candidate_path_cost")),
        _float_value(record.get("projected_candidate_risk")),
        _float_value(record.get("projected_candidate_utility")),
    )
    if any(value is None for value in selected_metrics + alternative_metrics):
        return "candidate_metrics_missing"
    if _list_cell(record.get("policy_target_cell")) is None or _list_cell(record.get("execution_goal_cell")) is None:
        return "candidate_metrics_missing"
    return None


def _exclusion_record(record: dict[str, Any], *, sample_type: str, reason: str) -> dict[str, Any]:
    return {
        "reason": reason,
        "sample_type": sample_type,
        "run_id": record.get("run_id"),
        "scenario_id": record.get("scenario_id"),
        "source_action_index": record.get("source_action_index"),
        "policy_target_cell": _list_cell(record.get("policy_target_cell")),
        "execution_goal_cell": _list_cell(record.get("execution_goal_cell")),
        "final_training_decision": _decision(record),
        "target_binding_mode": record.get("target_binding_mode"),
        "ppo_consumable_action": bool(record.get("ppo_consumable_action")),
        "contract_safe": bool(record.get("contract_safe")),
        "projection_distance_cells": record.get("projection_distance_cells"),
        "projection_distance_m": record.get("projection_distance_m"),
    }


def _merged_records(mining: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = {
        _record_key(record): record
        for record in candidate.get("context_records", [])
        if isinstance(record, dict)
    }
    records: list[dict[str, Any]] = []
    for record in mining.get("final_decision_records", []):
        if not isinstance(record, dict):
            continue
        merged = dict(candidates.get(_record_key(record), {}))
        merged.update(record)
        records.append(merged)
    return records


def _record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("run_id"),
        record.get("scenario_id"),
        _int_value(record.get("source_action_index")),
        tuple(_list_cell(record.get("policy_target_cell")) or []),
        tuple(_list_cell(record.get("execution_goal_cell")) or []),
    )


def _fallback_record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("run_id"),
        record.get("scenario_id"),
        _int_value(record.get("source_action_index")),
    )


def _context_id(record: dict[str, Any]) -> str:
    if record.get("context_id"):
        return str(record["context_id"])
    parts = [
        str(record.get("run_id")),
        str(record.get("scenario_id")),
        str(_int_value(record.get("source_action_index"))),
        str(_list_cell(record.get("policy_target_cell"))),
        str(_list_cell(record.get("execution_goal_cell"))),
    ]
    return "|".join(parts)


def _sample_context_id(sample: dict[str, Any]) -> str:
    if sample.get("context_id"):
        return str(sample["context_id"])
    alternative = sample.get("alternative") if isinstance(sample.get("alternative"), dict) else {}
    parts = [
        str(sample.get("run_id")),
        str(sample.get("scenario_id")),
        str(alternative.get("source_action_index")),
        str(alternative.get("policy_target_cell")),
        str(alternative.get("execution_goal_cell")),
    ]
    return "|".join(parts)


def _candidate_features(
    *,
    cell: list[int] | None,
    utility: float | None,
    path_cost: float | None,
    risk: float | None,
) -> list[float]:
    if cell is None:
        return [0.0 for _ in range(15)]
    x, y = cell
    return [
        _clip01(x / 100.0),
        _clip01(y / 100.0),
        _clip_signed(x / 100.0),
        _clip_signed(y / 100.0),
        _clip01(((x * x + y * y) ** 0.5) / 141.5),
        _clip01(utility or 0.0),
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        _clip01(utility or 0.0),
        _clip01(risk or 0.0),
        _clip01((path_cost or 0.0) / 100.0),
        _clip01((path_cost or 0.0) / 100.0),
    ]


def _global_features() -> list[float]:
    return [0.0 for _ in range(8)]


def _missing_indicators() -> list[list[float]]:
    return [[0.0 for _ in range(8)], [0.0 for _ in range(8)]]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clip_signed(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _write_outputs(result: dict[str, Any]) -> None:
    result["summary_path"].parent.mkdir(parents=True, exist_ok=True)
    result["registry_path"].write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in result["registry"]),
        encoding="utf-8",
    )
    result["summary_path"].write_text(
        json.dumps(result["summary"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result["exclusion_path"].write_text(
        json.dumps(result["exclusion_report"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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
    for section in ("input_files", "output_files", "validation", "expected_counts"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
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


def _load_jsonl(path: Path, *, label: str, reason_codes: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    except json.JSONDecodeError:
        _append_reason(reason_codes, f"{label}_invalid_json")
    return rows


def _count_jsonl_lines(path: Path, *, label: str, reason_codes: list[str]) -> int:
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _source_descriptor(path: Path, payload: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path, repo_root),
        "exists": path.is_file(),
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status"),
    }


def _require_passed(payload: dict[str, Any], label: str, reason_codes: list[str]) -> None:
    if payload.get("status") != "passed":
        _append_reason(reason_codes, f"{label}_failed")


def _check_expected_count(
    reason_codes: list[str],
    expected: dict[str, Any],
    field: str,
    actual: int,
) -> None:
    if field in expected and actual != _int_value(expected.get(field)):
        _append_reason(reason_codes, f"{field}_mismatch")


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _decision(record: dict[str, Any]) -> str:
    return str(record.get("final_training_decision") or "")


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _list_cell(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return [int(value[0]), int(value[1])]
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
