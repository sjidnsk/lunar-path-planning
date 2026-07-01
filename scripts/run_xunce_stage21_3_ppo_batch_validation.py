from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_EXPLORER_SRC = SCRIPT_DIR.parent / "model-explorer" / "src"
if str(MODEL_EXPLORER_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_EXPLORER_SRC))

from model_explorer.policy.training import compute_returns_and_advantages
from xunce_synthetic_exploration_credit import BEHAVIOR_POLICY_ID, synthetic_credit_behavior_logprob
from xunce_theta_viewpoint_candidates import row_has_theta_viewpoint_contract

THETA_COVERAGE_SOURCE = "theta_aware_sensor_footprint/v1"
SLOPE_OBSTACLE_COVERAGE_SOURCE = "endpoint_theta_slope_obstacle_los/v1"
HYBRID_ASTAR_PATH_COST_SOURCE = "hybrid_astar_pose_path/v1"
CONTINUOUS_THETA_ACTION_SPACE = "hybrid_discrete_xy_continuous_theta/v1"
SLOPE_OBSTACLE_MAX_TRAVERSABLE_SLOPE_DEG = 30.0


CONFIG_SCHEMA_VERSION = "xunce-stage21-3-ppo-batch-validation-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage21-3-ppo-batch-validation-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage21-3-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage21-3-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage21_3_ppo_batch_validation_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/"
    "outputs/path_feedback_batch_xunce_stage21_3_ppo_batch_validation_v1"
)

SUMMARY_FILE = "xunce-stage21-3-ppo-batch-validation-summary.json"
BATCH_FILE = "xunce-stage21-3-ppo-trainable-batch.jsonl"
SPLITS_FILE = "xunce-stage21-3-ppo-batch-splits.json"
RETURN_AUDIT_FILE = "xunce-stage21-3-return-advantage-audit.jsonl"
LINEAGE_FILE = "xunce-stage21-3-lineage-audit.json"
ROUTING_FILE = "xunce-stage21-3-next-stage-routing.json"
REPORT_FILE = "xunce-stage21-3-report.md"
MANIFEST_FILE = "xunce-stage21-3-manifest.json"

ROUTE_BOUNDARY = "resolve_stage21_3_ppo_batch_validation_boundary_rejections"
ROUTE_INPUTS = "rerun_stage21_3_required_inputs"
ROUTE_DUPLICATE_LINEAGE = "repair_stage21_3_duplicate_lineage_ids"
ROUTE_LINEAGE = "repair_stage21_3_batch_lineage"
ROUTE_EPISODE = "repair_stage21_3_episode_boundary_contract"
ROUTE_REWARD_TRAINABILITY = "repair_stage21_3_reward_trainability_contract"
ROUTE_SPLIT = "expand_stage21_3_batch_for_train_validation_split"
ROUTE_CONTRACT = "repair_stage21_3_ppo_batch_contract"
ROUTE_STAGE21_4 = "implement_stage21_4_tiny_ppo_update_smoke"

BOUNDARY_FIELDS = (
    "stage21_3_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 21.3 PPO batch validation.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    try:
        summary = run_xunce_stage21_3_ppo_batch_validation(
            config_path=Path(args.config),
            output_root=Path(args.output_root),
            repo_root=Path(args.repo_root),
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "next_required_change": summary["next_required_change"],
                "trainable_transition_count": summary["trainable_transition_count"],
                "return_finite_count": summary["return_finite_count"],
                "advantage_finite_count": summary["advantage_finite_count"],
                "stage21_3_authorized": summary["stage21_3_authorized"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage21_3_ppo_batch_validation(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root=repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    boundary_reasons = _boundary_rejections(config)
    input_reasons = _input_rejections(config)

    stage21_1_summary: dict[str, Any] = {}
    stage21_2_summary: dict[str, Any] = {}
    batch_rows: list[dict[str, Any]] = []
    return_audit: list[dict[str, Any]] = []
    splits: dict[str, Any] = {"train": [], "validation": []}
    lineage: dict[str, Any] = {}
    if not boundary_reasons and not input_reasons:
        stage21_1_root = Path(config["stage21_1_collector_root"])
        stage21_2_root = Path(config["stage21_2_reward_contract_root"])
        stage21_1_summary = _read_json(stage21_1_root / "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json")
        stage21_2_summary = _read_json(stage21_2_root / "xunce-stage21-2-coverage-first-reward-summary.json")
        transitions = _read_jsonl(stage21_1_root / "xunce-stage21-1-ppo-trainable-batch.jsonl")
        reward_rows = _read_jsonl(stage21_2_root / "xunce-stage21-2-reward-contract-evaluation.jsonl")
        batch_rows, return_audit = _build_batch(
            transitions,
            reward_rows,
            discount_factor=float(config["discount_factor"]),
        )
        splits = _split_rows(batch_rows, validation_fraction=float(config["validation_fraction"]))
        lineage = _lineage_audit(stage21_1_summary, stage21_2_summary, transitions, reward_rows)
        _apply_splits_and_advantages(
            batch_rows,
            return_audit,
            splits,
            normalize_advantages=bool(config["normalize_advantages"]),
        )

    contract_reasons = _contract_rejections(batch_rows, return_audit, splits, config)
    duplicate_lineage_reasons = _duplicate_lineage_rejections(lineage)
    lineage_reasons = _lineage_rejections(lineage)
    episode_reasons = _episode_rejections(lineage)
    reward_trainability_reasons = _reward_trainability_rejections(lineage)
    split_reasons = _split_rejections(splits, config)
    route = ROUTE_STAGE21_4
    status = "passed"
    blocking = list(
        boundary_reasons
        + input_reasons
        + duplicate_lineage_reasons
        + lineage_reasons
        + episode_reasons
        + reward_trainability_reasons
        + split_reasons
        + contract_reasons
    )
    if boundary_reasons:
        route = ROUTE_BOUNDARY
        status = "failed"
    elif input_reasons:
        route = ROUTE_INPUTS
        status = "failed"
    elif duplicate_lineage_reasons:
        route = ROUTE_DUPLICATE_LINEAGE
        status = "failed"
    elif lineage_reasons:
        route = ROUTE_LINEAGE
        status = "failed"
    elif episode_reasons:
        route = ROUTE_EPISODE
        status = "failed"
    elif reward_trainability_reasons:
        route = ROUTE_REWARD_TRAINABILITY
        status = "failed"
    elif split_reasons:
        route = ROUTE_SPLIT
        status = "failed"
    elif contract_reasons:
        route = ROUTE_CONTRACT
        status = "failed"
    return _write_outputs(
        config=config,
        output_root=output_root,
        config_path=_resolve_path(config_path, repo_root),
        stage21_1_summary=stage21_1_summary,
        stage21_2_summary=stage21_2_summary,
        batch_rows=batch_rows,
        return_audit=return_audit,
        splits=splits,
        lineage=lineage,
        blocking_reason_codes=blocking,
        route=route,
        status=status,
    )


def _build_batch(
    transitions: list[dict[str, Any]],
    reward_rows: list[dict[str, Any]],
    *,
    discount_factor: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reward_by_id = _first_row_by_transition_id(reward_rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for transition in transitions:
        scenario_id = str(transition.get("scenario_id"))
        grouped.setdefault(scenario_id, []).append(transition)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row.get("step_index", 0)))

    returns_by_id: dict[str, float] = {}
    raw_advantage_by_id: dict[str, float] = {}
    audit_rows: list[dict[str, Any]] = []
    for scenario_id, rows in grouped.items():
        rewards: list[float] = []
        dones: list[bool] = []
        values: list[float] = []
        for row in rows:
            reward_row = reward_by_id.get(str(row.get("transition_id")), {})
            reward_value = _finite(reward_row.get("reward"))
            rewards.append(reward_value if reward_value is not None else 0.0)
            dones.append(bool(row.get("done")))
            old_value = _finite(row.get("old_value"))
            values.append(old_value if old_value is not None else 0.0)
        computed = compute_returns_and_advantages(
            rewards=rewards,
            dones=dones,
            values=values,
            discount_factor=discount_factor,
            mode="discounted",
        )
        for index, row in enumerate(rows):
            transition_id = str(row.get("transition_id"))
            reward_row = reward_by_id.get(transition_id, {})
            ret = float(computed.returns[index])
            advantage = float(computed.advantages[index])
            old_value = _finite(row.get("old_value")) or 0.0
            returns_by_id[transition_id] = ret
            raw_advantage_by_id[transition_id] = advantage
            audit_rows.append(
                {
                    "schema_version": "xunce-stage21-3-return-advantage-audit/v1",
                    "transition_id": transition_id,
                    "scenario_id": scenario_id,
                    "step_index": row.get("step_index"),
                    "reward": reward_row.get("reward", row.get("reward")),
                    "return": ret,
                    "old_value": old_value,
                    "advantage": advantage,
                    "raw_advantage": advantage,
                    "normalized_advantage": advantage,
                    "stage21_3_split": None,
                    "advantage_normalization_applied": False,
                    "advantage_normalization_scope": "disabled",
                    "done": bool(row.get("done")),
                }
            )

    batch_rows: list[dict[str, Any]] = []
    for transition in transitions:
        transition_id = str(transition.get("transition_id"))
        reward_row = reward_by_id.get(transition_id, {})
        info = transition.get("info") if isinstance(transition.get("info"), dict) else {}
        batch_rows.append(
            {
                "schema_version": "xunce-stage21-3-ppo-trainable-transition/v1",
                "transition_id": transition_id,
                "scenario_id": transition.get("scenario_id"),
                "step_index": transition.get("step_index"),
                "observation": transition.get("observation"),
                "xunce_batch": transition.get("xunce_batch"),
                "action_index": transition.get("action_index"),
                "action_space_type": transition.get("action_space_type"),
                "selected_base_candidate_index": transition.get("selected_base_candidate_index"),
                "selected_theta_rad": transition.get("selected_theta_rad"),
                "selected_theta_deg": transition.get("selected_theta_deg"),
                "old_point_log_prob": transition.get("old_point_log_prob"),
                "old_theta_log_prob": transition.get("old_theta_log_prob"),
                "old_policy_point_log_prob": transition.get("old_policy_point_log_prob", info.get("old_policy_point_log_prob")),
                "old_policy_theta_log_prob": transition.get("old_policy_theta_log_prob", info.get("old_policy_theta_log_prob")),
                "old_policy_log_prob": transition.get("old_policy_log_prob", info.get("old_policy_log_prob")),
                "old_behavior_point_log_prob": transition.get(
                    "old_behavior_point_log_prob",
                    info.get("old_behavior_point_log_prob"),
                ),
                "old_behavior_theta_log_prob": transition.get(
                    "old_behavior_theta_log_prob",
                    info.get("old_behavior_theta_log_prob"),
                ),
                "old_behavior_log_prob": transition.get("old_behavior_log_prob", info.get("old_behavior_log_prob")),
                "behavior_policy_id": transition.get("behavior_policy_id", info.get("behavior_policy_id")),
                "synthetic_credit_mixture_probability": transition.get(
                    "synthetic_credit_mixture_probability",
                    info.get("synthetic_credit_mixture_probability"),
                ),
                "synthetic_credit_target_index": transition.get(
                    "synthetic_credit_target_index",
                    info.get("synthetic_credit_target_index"),
                ),
                "synthetic_credit_target_theta_deg": transition.get(
                    "synthetic_credit_target_theta_deg",
                    info.get("synthetic_credit_target_theta_deg"),
                ),
                "synthetic_credit_theta_policy_id": transition.get(
                    "synthetic_credit_theta_policy_id",
                    info.get("synthetic_credit_theta_policy_id"),
                ),
                "synthetic_credit_theta_proposals_deg": transition.get(
                    "synthetic_credit_theta_proposals_deg",
                    info.get("synthetic_credit_theta_proposals_deg"),
                ),
                "synthetic_credit_theta_selected_proposal_index": transition.get(
                    "synthetic_credit_theta_selected_proposal_index",
                    info.get("synthetic_credit_theta_selected_proposal_index"),
                ),
                "synthetic_credit_theta_reachable_proposal_count": transition.get(
                    "synthetic_credit_theta_reachable_proposal_count",
                    info.get("synthetic_credit_theta_reachable_proposal_count"),
                ),
                "synthetic_credit_target_selected": transition.get(
                    "synthetic_credit_target_selected",
                    info.get("synthetic_credit_target_selected"),
                ),
                "synthetic_credit_score": transition.get("synthetic_credit_score", info.get("synthetic_credit_score")),
                "synthetic_credit_score_version": transition.get(
                    "synthetic_credit_score_version",
                    info.get("synthetic_credit_score_version"),
                ),
                "path_efficiency_filter_relaxed": transition.get(
                    "path_efficiency_filter_relaxed",
                    info.get("path_efficiency_filter_relaxed"),
                ),
                "selected_target_hybrid_cost_norm": transition.get(
                    "selected_target_hybrid_cost_norm",
                    info.get("selected_target_hybrid_cost_norm"),
                ),
                "selected_target_gain_per_cost_norm": transition.get(
                    "selected_target_gain_per_cost_norm",
                    info.get("selected_target_gain_per_cost_norm"),
                ),
                "xunce_batch_feature_semantic_map": info.get("xunce_batch_feature_semantic_map"),
                "base_candidate_set_hash": transition.get("base_candidate_set_hash"),
                "action_sample_hash": transition.get("action_sample_hash"),
                "old_log_prob": transition.get("old_log_prob"),
                "old_value": transition.get("old_value"),
                "reward": reward_row.get("reward", transition.get("reward")),
                "return": returns_by_id.get(transition_id),
                "advantage": raw_advantage_by_id.get(transition_id),
                "raw_advantage": raw_advantage_by_id.get(transition_id),
                "stage21_3_split": None,
                "next_observation": transition.get("next_observation"),
                "next_xunce_batch": transition.get("next_xunce_batch"),
                "done": bool(transition.get("done")),
                "info": transition.get("info"),
                "transition_trainable": bool(transition.get("trainable")),
                "reward_trainable": reward_row.get("trainable"),
                "reward_reason_codes": reward_row.get("reason_codes"),
                "reward_scenario_id": reward_row.get("scenario_id"),
                "reward_step_index": reward_row.get("step_index"),
                "reward_done": reward_row.get("done"),
                "reward_profile_id": reward_row.get("profile_id"),
                "reward_profile_version": reward_row.get("profile_version"),
                "reward_profile_hash": reward_row.get("profile_hash"),
                "reward_components": reward_row.get("components"),
                "theta_aware_reward_contract": reward_row.get("theta_aware_reward_contract"),
                "coverage_source": reward_row.get("coverage_source"),
                "candidate_viewpoint": reward_row.get("candidate_viewpoint"),
                "candidate_theta_deg": reward_row.get("candidate_theta_deg"),
                "theta_new_visible_cell_count": reward_row.get("theta_new_visible_cell_count"),
                "theta_coverage_hash": reward_row.get("theta_coverage_hash"),
                "theta_coverage_gain_per_path_cost": reward_row.get("theta_coverage_gain_per_path_cost"),
                "theta_coverage_denominator_cells": reward_row.get("theta_coverage_denominator_cells"),
                "slope_obstacle_aware_theta_reward_contract": reward_row.get("slope_obstacle_aware_theta_reward_contract"),
                "obstacle_aware_new_visible_cell_count": reward_row.get("obstacle_aware_new_visible_cell_count"),
                "obstacle_aware_theta_coverage_hash": reward_row.get("obstacle_aware_theta_coverage_hash"),
                "obstacle_aware_theta_coverage_gain_per_path_cost": reward_row.get(
                    "obstacle_aware_theta_coverage_gain_per_path_cost"
                ),
                "obstacle_aware_theta_coverage_denominator_cells": reward_row.get(
                    "obstacle_aware_theta_coverage_denominator_cells"
                ),
                "slope_obstacle_source_hash": reward_row.get("slope_obstacle_source_hash"),
                "platform_contract_hash": reward_row.get("platform_contract_hash"),
                "max_traversable_slope_deg": reward_row.get("max_traversable_slope_deg"),
                "slope_blocked_source_kind": reward_row.get("slope_blocked_source_kind"),
                "unobstructed_theta_reward_fallback_used": reward_row.get("unobstructed_theta_reward_fallback_used"),
                "hybrid_astar_path_cost_reward_contract": reward_row.get(
                    "hybrid_astar_path_cost_reward_contract"
                ),
                "path_cost_source": reward_row.get("path_cost_source"),
                "hybrid_astar_path_cost": reward_row.get("hybrid_astar_path_cost"),
                "hybrid_astar_pose_path_hash": reward_row.get("hybrid_astar_pose_path_hash"),
                "hybrid_astar_trajectory_kind": reward_row.get("hybrid_astar_trajectory_kind"),
                "legacy_grid_astar_path_cost": reward_row.get("legacy_grid_astar_path_cost"),
                "hybrid_vs_grid_path_cost_delta": reward_row.get("hybrid_vs_grid_path_cost_delta"),
                "default_astar_replaced": reward_row.get("default_astar_replaced"),
                "hybrid_astar_ackermann_feasible_claimed": reward_row.get(
                    "hybrid_astar_ackermann_feasible_claimed"
                ),
                "point_grid_path_cost_fallback_used": reward_row.get("point_grid_path_cost_fallback_used"),
                "synthetic_terrain_reward_provenance": reward_row.get("synthetic_terrain_reward_provenance"),
                "synthetic_terrain_model_id": reward_row.get("synthetic_terrain_model_id"),
                "synthetic_terrain_hash": reward_row.get("synthetic_terrain_hash"),
                "synthetic_source_kind": reward_row.get("synthetic_source_kind"),
                "synthetic_hard_obstacle_cells_used": reward_row.get("synthetic_hard_obstacle_cells_used"),
                "synthetic_los_blocker_cells_used": reward_row.get("synthetic_los_blocker_cells_used"),
                "synthetic_high_risk_cells_available": reward_row.get("synthetic_high_risk_cells_available"),
                "physical_obstacle_cells_written": reward_row.get("physical_obstacle_cells_written"),
                "effective_hard_obstacle_source": reward_row.get("effective_hard_obstacle_source"),
                "effective_los_blocker_source": reward_row.get("effective_los_blocker_source"),
                "reward_metrics": reward_row.get("metrics"),
                "point_only_reward_fallback_used": reward_row.get("point_only_reward_fallback_used"),
                "old_log_prob_recompute_abs_error": info.get("old_log_prob_recompute_abs_error"),
            }
        )
    return batch_rows, audit_rows


def _split_rows(rows: list[dict[str, Any]], *, validation_fraction: float) -> dict[str, Any]:
    train: list[str] = []
    validation: list[str] = []
    scenario_ids = sorted({str(row.get("scenario_id")) for row in rows})
    for scenario_id in scenario_ids:
        bucket = _stable_bucket(scenario_id)
        if bucket < validation_fraction:
            validation.append(scenario_id)
        else:
            train.append(scenario_id)
    if rows and not train and validation:
        train.append(validation.pop())
    train_ids = {row["transition_id"] for row in rows if str(row.get("scenario_id")) in set(train)}
    validation_ids = {row["transition_id"] for row in rows if str(row.get("scenario_id")) in set(validation)}
    return {
        "schema_version": "xunce-stage21-3-ppo-batch-splits/v1",
        "validation_fraction": validation_fraction,
        "train_scenario_ids": train,
        "validation_scenario_ids": validation,
        "train_transition_ids": sorted(train_ids),
        "validation_transition_ids": sorted(validation_ids),
        "overlap_transition_count": len(train_ids & validation_ids),
    }


def _apply_splits_and_advantages(
    rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    splits: dict[str, Any],
    *,
    normalize_advantages: bool,
) -> None:
    split_by_id: dict[str, str] = {}
    for transition_id in splits.get("train_transition_ids", []):
        split_by_id[str(transition_id)] = "train"
    for transition_id in splits.get("validation_transition_ids", []):
        split_by_id[str(transition_id)] = "validation"

    train_advantages = [
        float(row["raw_advantage"])
        for row in rows
        if split_by_id.get(str(row.get("transition_id"))) == "train" and _finite(row.get("raw_advantage")) is not None
    ]
    mean_adv = sum(train_advantages) / len(train_advantages) if train_advantages else 0.0
    variance = sum((value - mean_adv) ** 2 for value in train_advantages) / len(train_advantages) if train_advantages else 0.0
    std_adv = variance ** 0.5
    normalize = normalize_advantages and len(train_advantages) >= 2 and std_adv > 1.0e-12

    normalized_by_id: dict[str, float] = {}
    for row in rows:
        transition_id = str(row.get("transition_id"))
        split = split_by_id.get(transition_id)
        row["stage21_3_split"] = split
        raw = _finite(row.get("raw_advantage"))
        normalized = (raw - mean_adv) / std_adv if normalize and raw is not None else raw
        row["advantage"] = normalized
        row["advantage_normalization_applied"] = normalize
        row["advantage_normalization_scope"] = "train_split" if normalize else "disabled"
        row["advantage_normalization_mean"] = mean_adv if normalize else None
        row["advantage_normalization_std"] = std_adv if normalize else None
        if normalized is not None:
            normalized_by_id[transition_id] = float(normalized)

    for row in audit_rows:
        transition_id = str(row.get("transition_id"))
        split = split_by_id.get(transition_id)
        row["stage21_3_split"] = split
        row["normalized_advantage"] = normalized_by_id.get(transition_id, row.get("raw_advantage"))
        row["advantage"] = row["normalized_advantage"]
        row["advantage_normalization_applied"] = normalize
        row["advantage_normalization_scope"] = "train_split" if normalize else "disabled"
        row["advantage_normalization_mean"] = mean_adv if normalize else None
        row["advantage_normalization_std"] = std_adv if normalize else None


def _lineage_audit(
    stage21_1_summary: dict[str, Any],
    stage21_2_summary: dict[str, Any],
    transitions: list[dict[str, Any]],
    reward_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    transition_ids_list = [str(row.get("transition_id") or "") for row in transitions]
    reward_ids_list = [str(row.get("transition_id") or "") for row in reward_rows]
    transition_ids = {transition_id for transition_id in transition_ids_list if transition_id}
    reward_ids = {transition_id for transition_id in reward_ids_list if transition_id}
    transition_by_id = _first_row_by_transition_id(transitions)
    reward_by_id = _first_row_by_transition_id(reward_rows)
    common_ids = sorted(transition_ids & reward_ids)

    scenario_mismatch_count = 0
    step_mismatch_count = 0
    done_mismatch_count = 0
    transition_trainable_false_count = 0
    reward_trainable_false_count = 0
    hard_risk_reward_rejection_count = 0
    reward_reason_code_nonempty_count = 0
    profile_hashes = set()
    for transition_id in common_ids:
        transition = transition_by_id[transition_id]
        reward = reward_by_id[transition_id]
        if str(transition.get("scenario_id")) != str(reward.get("scenario_id")):
            scenario_mismatch_count += 1
        if _int_or_none(transition.get("step_index")) != _int_or_none(reward.get("step_index")):
            step_mismatch_count += 1
        if bool(transition.get("done")) != bool(reward.get("done")):
            done_mismatch_count += 1
        if transition.get("trainable") is not True:
            transition_trainable_false_count += 1
        if reward.get("trainable") is not True:
            reward_trainable_false_count += 1
        reason_codes = reward.get("reason_codes") if isinstance(reward.get("reason_codes"), list) else []
        if reason_codes:
            reward_reason_code_nonempty_count += 1
        if "hard_risk_rejected" in reason_codes:
            hard_risk_reward_rejection_count += 1
        if reward.get("profile_hash"):
            profile_hashes.add(str(reward.get("profile_hash")))

    episode_audit = _episode_boundary_audit(transitions)
    return {
        "schema_version": "xunce-stage21-3-lineage-audit/v1",
        "stage21_1_status": stage21_1_summary.get("status"),
        "stage21_2_status": stage21_2_summary.get("status"),
        "stage21_1_next_required_change": stage21_1_summary.get("next_required_change"),
        "stage21_2_next_required_change": stage21_2_summary.get("next_required_change"),
        "stage21_1_transition_count": len(transitions),
        "stage21_2_reward_row_count": len(reward_rows),
        "missing_reward_transition_count": len(transition_ids - reward_ids),
        "extra_reward_transition_count": len(reward_ids - transition_ids),
        "blank_transition_id_count": sum(1 for value in transition_ids_list if not value),
        "blank_reward_transition_id_count": sum(1 for value in reward_ids_list if not value),
        "duplicate_transition_id_count": _duplicate_count(transition_ids_list),
        "duplicate_reward_transition_id_count": _duplicate_count(reward_ids_list),
        "duplicate_transition_ids": _duplicate_values(transition_ids_list),
        "duplicate_reward_transition_ids": _duplicate_values(reward_ids_list),
        "scenario_id_mismatch_count": scenario_mismatch_count,
        "step_index_mismatch_count": step_mismatch_count,
        "done_mismatch_count": done_mismatch_count,
        "transition_trainable_false_count": transition_trainable_false_count,
        "reward_trainable_false_count": reward_trainable_false_count,
        "reward_reason_code_nonempty_count": reward_reason_code_nonempty_count,
        "hard_risk_reward_rejection_count": hard_risk_reward_rejection_count,
        "reward_profile_hash_count": len(profile_hashes),
        "reward_profile_hashes": sorted(profile_hashes),
        "reward_profile_hash": stage21_2_summary.get("profile_hash"),
        "episode_audit": episode_audit,
    }


def _contract_rejections(rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]], splits: dict[str, Any], config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if len(rows) < int(config["min_trainable_transition_count"]):
        reasons.append("trainable_transition_count_below_minimum")
    if not rows:
        reasons.append("empty_ppo_batch")
        return reasons
    if any(_finite(row.get("old_log_prob")) is None for row in rows):
        reasons.append("non_finite_old_log_prob")
    if any(_finite(row.get("old_value")) is None for row in rows):
        reasons.append("non_finite_old_value")
    if any(_finite(row.get("reward")) is None for row in rows):
        reasons.append("non_finite_reward")
    if any(_finite(row.get("return")) is None for row in rows):
        reasons.append("non_finite_return")
    if any(_finite(row.get("advantage")) is None for row in rows):
        reasons.append("non_finite_advantage")
    if _invalid_action_count(rows) > 0:
        reasons.append("invalid_action_detected")
    if bool(config.get("require_theta_aware_viewpoint_contract")) and _theta_viewpoint_contract_missing_count(rows) > 0:
        reasons.append("theta_viewpoint_contract_missing")
    if bool(config.get("require_theta_aware_reward_contract")) and _theta_reward_contract_missing_count(rows) > 0:
        reasons.append("theta_reward_contract_missing")
    if bool(config.get("require_slope_obstacle_aware_theta_reward_contract")) and _slope_obstacle_reward_contract_missing_count(rows) > 0:
        reasons.append("slope_obstacle_aware_theta_reward_contract_missing")
    if bool(config.get("require_hybrid_astar_path_cost_contract")) and _hybrid_astar_path_cost_contract_missing_count(rows) > 0:
        reasons.append("hybrid_astar_path_cost_contract_missing")
    if bool(config.get("require_synthetic_terrain_contract")) and _synthetic_terrain_contract_missing_count(rows) > 0:
        reasons.append("synthetic_terrain_contract_missing")
    if bool(config.get("require_continuous_theta_action_contract")) and _continuous_theta_action_contract_missing_count(rows) > 0:
        reasons.append("continuous_theta_action_contract_missing")
    if not bool(config.get("allow_synthetic_credit_behavior_policy")) and any(
        _row_uses_synthetic_credit_behavior_policy(row) for row in rows
    ):
        reasons.append("synthetic_credit_behavior_policy_not_allowed")
    if bool(config.get("allow_synthetic_credit_behavior_policy")) and _synthetic_credit_behavior_logprob_missing_count(rows) > 0:
        reasons.append("synthetic_credit_behavior_logprob_contract_missing")
    if _hard_risk_count(rows) > 0:
        reasons.append("hard_risk_violation_detected")
    if _old_log_prob_recompute_violation_count(rows, float(config["max_old_log_prob_recompute_abs_error"])) > 0:
        reasons.append("old_log_prob_recompute_error_exceeds_threshold")
    if _missing_next_observation_count(rows) > 0:
        reasons.append("missing_next_observation_for_non_terminal")
    if splits.get("overlap_transition_count", 0) != 0:
        reasons.append("train_validation_split_overlap")
    if not audit_rows:
        reasons.append("missing_return_advantage_audit")
    return reasons


def _duplicate_lineage_rejections(lineage: dict[str, Any]) -> list[str]:
    if not lineage:
        return []
    reasons: list[str] = []
    if lineage.get("blank_transition_id_count", 0) != 0:
        reasons.append("blank_transition_ids")
    if lineage.get("blank_reward_transition_id_count", 0) != 0:
        reasons.append("blank_reward_transition_ids")
    if lineage.get("duplicate_transition_id_count", 0) != 0:
        reasons.append("duplicate_transition_ids")
    if lineage.get("duplicate_reward_transition_id_count", 0) != 0:
        reasons.append("duplicate_reward_transition_ids")
    return reasons


def _lineage_rejections(lineage: dict[str, Any]) -> list[str]:
    if not lineage:
        return []
    reasons: list[str] = []
    if lineage.get("stage21_1_status") != "passed":
        reasons.append("stage21_1_lineage_not_passed")
    if lineage.get("stage21_2_status") != "passed":
        reasons.append("stage21_2_lineage_not_passed")
    if lineage.get("stage21_1_next_required_change") != "implement_stage21_2_coverage_first_ppo_reward_contract":
        reasons.append("stage21_1_route_mismatch")
    if lineage.get("stage21_2_next_required_change") != "implement_stage21_3_ppo_batch_validation":
        reasons.append("stage21_2_route_mismatch")
    if lineage.get("missing_reward_transition_count") != 0:
        reasons.append("missing_reward_transition_rows")
    if lineage.get("extra_reward_transition_count") != 0:
        reasons.append("extra_reward_transition_rows")
    if lineage.get("scenario_id_mismatch_count", 0) != 0:
        reasons.append("scenario_id_mismatch_between_transition_and_reward")
    if lineage.get("step_index_mismatch_count", 0) != 0:
        reasons.append("step_index_mismatch_between_transition_and_reward")
    if lineage.get("done_mismatch_count", 0) != 0:
        reasons.append("done_mismatch_between_transition_and_reward")
    if lineage.get("reward_profile_hash_count", 0) != 1:
        reasons.append("reward_profile_hash_count_not_one")
    elif str(lineage.get("reward_profile_hash")) not in set(lineage.get("reward_profile_hashes", [])):
        reasons.append("reward_profile_hash_mismatch_with_stage21_2_summary")
    return reasons


def _episode_rejections(lineage: dict[str, Any]) -> list[str]:
    if not lineage:
        return []
    audit = lineage.get("episode_audit") if isinstance(lineage.get("episode_audit"), dict) else {}
    reasons: list[str] = []
    if audit.get("non_monotonic_episode_count", 0) != 0:
        reasons.append("episode_step_index_not_monotonic")
    if audit.get("duplicate_step_index_episode_count", 0) != 0:
        reasons.append("episode_duplicate_step_index")
    if audit.get("non_contiguous_step_index_episode_count", 0) != 0:
        reasons.append("episode_step_index_not_contiguous")
    if audit.get("terminal_count_mismatch_episode_count", 0) != 0:
        reasons.append("episode_terminal_count_mismatch")
    if audit.get("terminal_not_last_episode_count", 0) != 0:
        reasons.append("episode_terminal_not_last")
    return reasons


def _reward_trainability_rejections(lineage: dict[str, Any]) -> list[str]:
    if not lineage:
        return []
    reasons: list[str] = []
    if lineage.get("transition_trainable_false_count", 0) != 0:
        reasons.append("transition_trainable_false")
    if lineage.get("reward_trainable_false_count", 0) != 0:
        reasons.append("reward_trainable_false")
    if lineage.get("hard_risk_reward_rejection_count", 0) != 0:
        reasons.append("hard_risk_reward_rejection")
    return reasons


def _split_rejections(splits: dict[str, Any], config: dict[str, Any]) -> list[str]:
    if not splits:
        return []
    reasons: list[str] = []
    if config.get("require_validation_split") is True and not splits.get("validation_transition_ids"):
        reasons.append("validation_split_empty")
    if not splits.get("train_transition_ids"):
        reasons.append("train_split_empty")
    return reasons


def _write_outputs(
    *,
    config: dict[str, Any],
    output_root: Path,
    config_path: Path,
    stage21_1_summary: dict[str, Any],
    stage21_2_summary: dict[str, Any],
    batch_rows: list[dict[str, Any]],
    return_audit: list[dict[str, Any]],
    splits: dict[str, Any],
    lineage: dict[str, Any],
    blocking_reason_codes: list[str],
    route: str,
    status: str,
) -> dict[str, Any]:
    generated_at = _utc_now()
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "next_required_change": route,
        "stage21_1_collector_root": config["stage21_1_collector_root"],
        "stage21_2_reward_contract_root": config["stage21_2_reward_contract_root"],
        "stage21_1_status": stage21_1_summary.get("status"),
        "stage21_2_status": stage21_2_summary.get("status"),
        "reward_profile_hash": stage21_2_summary.get("profile_hash"),
        "trainable_transition_count": len(batch_rows),
        "theta_aware_viewpoint_contract_required": bool(config.get("require_theta_aware_viewpoint_contract")),
        "theta_aware_viewpoint_contract_missing_count": _theta_viewpoint_contract_missing_count(batch_rows),
        "theta_aware_reward_contract_required": bool(config.get("require_theta_aware_reward_contract")),
        "theta_aware_reward_contract_missing_count": _theta_reward_contract_missing_count(batch_rows),
        "slope_obstacle_aware_theta_reward_contract_required": bool(
            config.get("require_slope_obstacle_aware_theta_reward_contract")
        ),
        "slope_obstacle_aware_theta_reward_contract_missing_count": _slope_obstacle_reward_contract_missing_count(
            batch_rows
        ),
        "point_only_reward_fallback_used_count": _point_only_reward_fallback_used_count(batch_rows),
        "unobstructed_theta_reward_fallback_used_count": _unobstructed_theta_reward_fallback_used_count(batch_rows),
        "hybrid_astar_path_cost_contract_required": bool(config.get("require_hybrid_astar_path_cost_contract")),
        "hybrid_astar_path_cost_contract_missing_count": _hybrid_astar_path_cost_contract_missing_count(batch_rows),
        "synthetic_terrain_contract_required": bool(config.get("require_synthetic_terrain_contract")),
        "synthetic_terrain_contract_missing_count": _synthetic_terrain_contract_missing_count(batch_rows),
        "synthetic_terrain_reward_provenance_count": sum(
            1 for row in batch_rows if row.get("synthetic_terrain_reward_provenance") is True
        ),
        "synthetic_physical_obstacle_pollution_count": sum(
            1 for row in batch_rows if row.get("physical_obstacle_cells_written") is True
        ),
        "continuous_theta_action_contract_required": bool(config.get("require_continuous_theta_action_contract")),
        "continuous_theta_action_contract_missing_count": _continuous_theta_action_contract_missing_count(batch_rows),
        "synthetic_credit_behavior_policy_allowed": bool(config.get("allow_synthetic_credit_behavior_policy")),
        "synthetic_credit_behavior_logprob_missing_count": _synthetic_credit_behavior_logprob_missing_count(batch_rows),
        "synthetic_credit_behavior_policy_row_count": sum(
            1 for row in batch_rows if _row_uses_synthetic_credit_behavior_policy(row)
        ),
        "point_grid_path_cost_fallback_used_count": _point_grid_path_cost_fallback_used_count(batch_rows),
        "return_finite_count": sum(1 for row in batch_rows if _finite(row.get("return")) is not None),
        "advantage_finite_count": sum(1 for row in batch_rows if _finite(row.get("advantage")) is not None),
        "invalid_action_count": _invalid_action_count(batch_rows),
        "hard_risk_violation_count": _hard_risk_count(batch_rows),
        "old_log_prob_recompute_max_abs_error": _old_log_prob_recompute_max_abs_error(batch_rows),
        "old_log_prob_recompute_violation_count": _old_log_prob_recompute_violation_count(
            batch_rows,
            float(config["max_old_log_prob_recompute_abs_error"]),
        ),
        "missing_next_observation_count": _missing_next_observation_count(batch_rows),
        "duplicate_transition_id_count": lineage.get("duplicate_transition_id_count", 0),
        "duplicate_reward_transition_id_count": lineage.get("duplicate_reward_transition_id_count", 0),
        "blank_transition_id_count": lineage.get("blank_transition_id_count", 0),
        "blank_reward_transition_id_count": lineage.get("blank_reward_transition_id_count", 0),
        "scenario_id_mismatch_count": lineage.get("scenario_id_mismatch_count", 0),
        "step_index_mismatch_count": lineage.get("step_index_mismatch_count", 0),
        "done_mismatch_count": lineage.get("done_mismatch_count", 0),
        "transition_trainable_false_count": lineage.get("transition_trainable_false_count", 0),
        "reward_trainable_false_count": lineage.get("reward_trainable_false_count", 0),
        "hard_risk_reward_rejection_count": lineage.get("hard_risk_reward_rejection_count", 0),
        "episode_audit": lineage.get("episode_audit", {}),
        "train_transition_count": len(splits.get("train_transition_ids", [])),
        "validation_transition_count": len(splits.get("validation_transition_ids", [])),
        "validation_split_required": bool(config.get("require_validation_split")),
        "blocking_reason_codes": _unique(blocking_reason_codes),
        "reason_codes": _unique(blocking_reason_codes),
        "stage21_3_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "summary": str((output_root / SUMMARY_FILE).resolve()),
        "batch": str((output_root / BATCH_FILE).resolve()),
        "splits": str((output_root / SPLITS_FILE).resolve()),
        "return_advantage_audit": str((output_root / RETURN_AUDIT_FILE).resolve()),
        "lineage": str((output_root / LINEAGE_FILE).resolve()),
        "routing": str((output_root / ROUTING_FILE).resolve()),
        "report": str((output_root / REPORT_FILE).resolve()),
        "manifest": str((output_root / MANIFEST_FILE).resolve()),
    }
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "primary_route": route,
        "stage21_3_authorized": False,
        "stage21_4_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config": str(config_path.resolve()),
        "artifacts": {
            "summary": summary["summary"],
            "batch": summary["batch"],
            "splits": summary["splits"],
            "return_advantage_audit": summary["return_advantage_audit"],
            "lineage": summary["lineage"],
            "routing": summary["routing"],
            "report": summary["report"],
        },
        "summary_status": status,
        "next_required_change": route,
    }
    _write_jsonl(output_root / BATCH_FILE, batch_rows)
    _write_jsonl(output_root / RETURN_AUDIT_FILE, return_audit)
    _write_json(output_root / SPLITS_FILE, splits)
    _write_json(output_root / LINEAGE_FILE, lineage)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _load_config(path: Path, *, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    for key in ("stage21_1_collector_root", "stage21_2_reward_contract_root"):
        if not isinstance(config.get(key), str) or not str(config[key]).strip():
            raise ConfigError(f"{key} must be a non-empty path string")
        config[key] = str(_resolve_path(Path(config[key]), repo_root))
    config["discount_factor"] = _range_float(config.get("discount_factor", 0.99), "discount_factor")
    config["validation_fraction"] = _fraction(config.get("validation_fraction", 0.25), "validation_fraction")
    config["normalize_advantages"] = bool(config.get("normalize_advantages", True))
    config["min_trainable_transition_count"] = _nonnegative_int(config.get("min_trainable_transition_count", 2), "min_trainable_transition_count")
    config["max_old_log_prob_recompute_abs_error"] = _nonnegative_float(
        config.get("max_old_log_prob_recompute_abs_error", 1.0e-6),
        "max_old_log_prob_recompute_abs_error",
    )
    config["require_validation_split"] = bool(config.get("require_validation_split", False))
    config["require_theta_aware_viewpoint_contract"] = bool(config.get("require_theta_aware_viewpoint_contract", False))
    config["require_theta_aware_reward_contract"] = bool(config.get("require_theta_aware_reward_contract", False))
    config["require_slope_obstacle_aware_theta_reward_contract"] = bool(
        config.get("require_slope_obstacle_aware_theta_reward_contract", False)
    )
    config["require_hybrid_astar_path_cost_contract"] = bool(
        config.get("require_hybrid_astar_path_cost_contract", False)
    )
    config["require_synthetic_terrain_contract"] = bool(config.get("require_synthetic_terrain_contract", False))
    config["require_continuous_theta_action_contract"] = bool(
        config.get("require_continuous_theta_action_contract", False)
    )
    config["allow_synthetic_credit_behavior_policy"] = bool(
        config.get("allow_synthetic_credit_behavior_policy", False)
    )
    config["canary_traffic_fraction"] = _nonnegative_float(config.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    for field in BOUNDARY_FIELDS:
        config.setdefault(field, False)
    return config


def _input_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    stage21_1_root = Path(config["stage21_1_collector_root"])
    stage21_2_root = Path(config["stage21_2_reward_contract_root"])
    required = [
        stage21_1_root / "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json",
        stage21_1_root / "xunce-stage21-1-ppo-trainable-batch.jsonl",
        stage21_2_root / "xunce-stage21-2-coverage-first-reward-summary.json",
        stage21_2_root / "xunce-stage21-2-reward-contract-evaluation.jsonl",
    ]
    for path in required:
        if not path.is_file():
            reasons.append(f"missing_{path.name}")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in BOUNDARY_FIELDS:
        if config.get(field) is not False:
            reasons.append(field)
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _invalid_action_count(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        action = row.get("action_index")
        try:
            action_index = int(action)
        except (TypeError, ValueError):
            count += 1
            continue
        action_mask = info.get("action_mask")
        sampling_mask = info.get("sampling_mask")
        hard_risk_clean_mask = info.get("hard_risk_clean_mask")
        if action_index < 0:
            count += 1
            continue
        if not isinstance(action_mask, list) or action_index >= len(action_mask) or action_mask[action_index] is not True:
            count += 1
            continue
        if not isinstance(sampling_mask, list) or action_index >= len(sampling_mask) or sampling_mask[action_index] is not True:
            count += 1
            continue
        if isinstance(hard_risk_clean_mask, list) and (
            action_index >= len(hard_risk_clean_mask) or hard_risk_clean_mask[action_index] is not True
        ):
            count += 1
    return count


def _theta_viewpoint_contract_missing_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if not _row_has_transition_theta_viewpoint_contract(row))


def _theta_reward_contract_missing_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if not _row_has_theta_reward_contract(row))


def _slope_obstacle_reward_contract_missing_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if not _row_has_slope_obstacle_theta_reward_contract(row))


def _hybrid_astar_path_cost_contract_missing_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if not _row_has_hybrid_astar_path_cost_contract(row))


def _synthetic_terrain_contract_missing_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if not _row_has_synthetic_terrain_contract(row))


def _continuous_theta_action_contract_missing_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if not _row_has_continuous_theta_action_contract(row))


def _synthetic_credit_behavior_logprob_missing_count(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if _row_uses_synthetic_credit_behavior_policy(row)
        and not _row_has_synthetic_credit_behavior_logprob(row)
    )


def _point_only_reward_fallback_used_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("point_only_reward_fallback_used") is True)


def _unobstructed_theta_reward_fallback_used_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("unobstructed_theta_reward_fallback_used") is True)


def _point_grid_path_cost_fallback_used_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("point_grid_path_cost_fallback_used") is True)


def _row_has_theta_reward_contract(row: dict[str, Any]) -> bool:
    if not (
        row.get("theta_aware_reward_contract") is True
        and row.get("coverage_source") in {THETA_COVERAGE_SOURCE, SLOPE_OBSTACLE_COVERAGE_SOURCE}
        and _present_viewpoint(row.get("candidate_viewpoint"))
        and _finite(row.get("candidate_theta_deg")) is not None
        and row.get("point_only_reward_fallback_used") is False
    ):
        return False
    if row.get("coverage_source") == SLOPE_OBSTACLE_COVERAGE_SOURCE:
        if not _row_has_slope_obstacle_theta_reward_contract(row):
            return False
    else:
        if not (
            _finite(row.get("theta_new_visible_cell_count")) is not None
            and _finite(row.get("theta_coverage_gain_per_path_cost")) is not None
            and _finite(row.get("theta_coverage_denominator_cells")) is not None
            and isinstance(row.get("theta_coverage_hash"), str)
            and bool(str(row.get("theta_coverage_hash")).strip())
        ):
            return False
    if not _theta_reward_matches_selected_action(row):
        return False
    if not _theta_reward_metrics_consistent(row):
        return False
    return True


def _row_has_slope_obstacle_theta_reward_contract(row: dict[str, Any]) -> bool:
    if not (
        row.get("theta_aware_reward_contract") is True
        and row.get("slope_obstacle_aware_theta_reward_contract") is True
        and row.get("coverage_source") == SLOPE_OBSTACLE_COVERAGE_SOURCE
        and _present_viewpoint(row.get("candidate_viewpoint"))
        and _finite(row.get("candidate_theta_deg")) is not None
        and _finite(row.get("obstacle_aware_new_visible_cell_count")) is not None
        and _finite(row.get("obstacle_aware_theta_coverage_gain_per_path_cost")) is not None
        and _finite(row.get("obstacle_aware_theta_coverage_denominator_cells")) is not None
        and isinstance(row.get("obstacle_aware_theta_coverage_hash"), str)
        and bool(str(row.get("obstacle_aware_theta_coverage_hash")).strip())
        and isinstance(row.get("slope_obstacle_source_hash"), str)
        and bool(str(row.get("slope_obstacle_source_hash")).strip())
        and isinstance(row.get("platform_contract_hash"), str)
        and bool(str(row.get("platform_contract_hash")).strip())
        and _finite(row.get("max_traversable_slope_deg")) is not None
        and abs(_finite(row.get("max_traversable_slope_deg")) - SLOPE_OBSTACLE_MAX_TRAVERSABLE_SLOPE_DEG) <= 1.0e-9
        and _valid_obstacle_source_kind(row)
        and row.get("point_only_reward_fallback_used") is False
        and row.get("unobstructed_theta_reward_fallback_used") is False
    ):
        return False
    if not _theta_reward_matches_selected_action(row):
        return False
    if not _slope_obstacle_reward_metrics_consistent(row):
        return False
    return True


def _row_has_hybrid_astar_path_cost_contract(row: dict[str, Any]) -> bool:
    if not (
        row.get("hybrid_astar_path_cost_reward_contract") is True
        and row.get("path_cost_source") == HYBRID_ASTAR_PATH_COST_SOURCE
        and _finite(row.get("hybrid_astar_path_cost")) is not None
        and _finite(row.get("hybrid_astar_path_cost")) > 0.0
        and isinstance(row.get("hybrid_astar_pose_path_hash"), str)
        and bool(str(row.get("hybrid_astar_pose_path_hash")).strip())
        and row.get("hybrid_astar_trajectory_kind") == "hybrid_astar_pose_path"
        and _finite(row.get("legacy_grid_astar_path_cost")) is not None
        and _finite(row.get("legacy_grid_astar_path_cost")) > 0.0
        and _finite(row.get("hybrid_vs_grid_path_cost_delta")) is not None
        and row.get("default_astar_replaced") is False
        and row.get("hybrid_astar_ackermann_feasible_claimed") is False
        and row.get("point_grid_path_cost_fallback_used") is False
        and isinstance(row.get("platform_contract_hash"), str)
        and bool(str(row.get("platform_contract_hash")).strip())
        and isinstance(row.get("slope_obstacle_source_hash"), str)
        and bool(str(row.get("slope_obstacle_source_hash")).strip())
        and _finite(row.get("max_traversable_slope_deg")) is not None
        and abs(_finite(row.get("max_traversable_slope_deg")) - SLOPE_OBSTACLE_MAX_TRAVERSABLE_SLOPE_DEG) <= 1.0e-9
    ):
        return False
    metrics = row.get("reward_metrics") if isinstance(row.get("reward_metrics"), dict) else {}
    path_cost = _finite(metrics.get("path_cost_m"))
    hybrid_cost = _finite(row.get("hybrid_astar_path_cost"))
    return path_cost is not None and hybrid_cost is not None and abs(path_cost - hybrid_cost) <= 1.0e-9


def _row_has_synthetic_terrain_contract(row: dict[str, Any]) -> bool:
    return (
        row.get("synthetic_terrain_reward_provenance") is True
        and row.get("synthetic_terrain_model_id") == "synthetic_rock_pit_terrain/v1"
        and isinstance(row.get("synthetic_terrain_hash"), str)
        and bool(str(row.get("synthetic_terrain_hash")).strip())
        and row.get("synthetic_source_kind") == "synthetic_terrain_obstacle_proxy/v1"
        and row.get("synthetic_hard_obstacle_cells_used") is True
        and row.get("synthetic_los_blocker_cells_used") is True
        and row.get("synthetic_high_risk_cells_available") is True
        and row.get("physical_obstacle_cells_written") is False
        and isinstance(row.get("effective_hard_obstacle_source"), list)
        and "synthetic_hard_obstacle_cells" in row.get("effective_hard_obstacle_source")
        and isinstance(row.get("effective_los_blocker_source"), list)
        and "synthetic_los_blocker_cells" in row.get("effective_los_blocker_source")
        and row.get("point_only_reward_fallback_used") is False
        and row.get("unobstructed_theta_reward_fallback_used") is False
        and row.get("point_grid_path_cost_fallback_used") is False
        and row.get("default_astar_replaced") is False
        and row.get("hybrid_astar_ackermann_feasible_claimed") is False
    )


def _valid_obstacle_source_kind(row: dict[str, Any]) -> bool:
    kind = row.get("slope_blocked_source_kind")
    if kind == "slope_blocked_as_obstacle_proxy":
        return True
    if kind == "synthetic_terrain_obstacle_proxy/v1" and row.get("synthetic_terrain_reward_provenance") is True:
        return True
    return False


def _row_has_continuous_theta_action_contract(row: dict[str, Any]) -> bool:
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    action_space = row.get("action_space_type") or info.get("action_space_type")
    selected_index = _int_or_none(row.get("selected_base_candidate_index"))
    if selected_index is None:
        selected_index = _int_or_none(info.get("selected_base_candidate_index"))
    action_index = _int_or_none(row.get("action_index"))
    theta_rad = _finite(row.get("selected_theta_rad"))
    if theta_rad is None:
        theta_rad = _finite(info.get("selected_theta_rad"))
    theta_deg = _finite(row.get("selected_theta_deg"))
    if theta_deg is None:
        theta_deg = _finite(info.get("selected_theta_deg"))
    point_log_prob = _finite(row.get("old_point_log_prob"))
    if point_log_prob is None:
        point_log_prob = _finite(info.get("old_point_log_prob"))
    theta_log_prob = _finite(row.get("old_theta_log_prob"))
    if theta_log_prob is None:
        theta_log_prob = _finite(info.get("old_theta_log_prob"))
    total_log_prob = _finite(row.get("old_log_prob"))
    base_hash = row.get("base_candidate_set_hash") or info.get("base_candidate_set_hash")
    sample_hash = row.get("action_sample_hash") or info.get("action_sample_hash")
    if not (
        action_space == CONTINUOUS_THETA_ACTION_SPACE
        and selected_index is not None
        and action_index is not None
        and selected_index == action_index
        and theta_rad is not None
        and theta_deg is not None
        and point_log_prob is not None
        and theta_log_prob is not None
        and total_log_prob is not None
        and isinstance(base_hash, str)
        and bool(base_hash.strip())
        and isinstance(sample_hash, str)
        and bool(sample_hash.strip())
    ):
        return False
    if abs(total_log_prob - (point_log_prob + theta_log_prob)) > 1.0e-5:
        return False
    if _angle_deg_delta_abs(theta_deg, math.degrees(theta_rad)) > 1.0e-5:
        return False
    if _row_uses_synthetic_credit_behavior_policy(row):
        policy_point_log_prob = _finite(row.get("old_policy_point_log_prob"))
        if policy_point_log_prob is None:
            policy_point_log_prob = _finite(info.get("old_policy_point_log_prob"))
        policy_theta_log_prob = _finite(row.get("old_policy_theta_log_prob"))
        if policy_theta_log_prob is None:
            policy_theta_log_prob = _finite(info.get("old_policy_theta_log_prob"))
        if policy_theta_log_prob is None:
            policy_theta_log_prob = theta_log_prob
        return (
            policy_point_log_prob is not None
            and _continuous_theta_old_log_prob_recomputes(
                info=info,
                action_index=action_index,
                theta_rad=theta_rad,
                old_point_log_prob=policy_point_log_prob,
                old_theta_log_prob=policy_theta_log_prob,
            )
            and _row_has_synthetic_credit_behavior_logprob(row)
        )
    return _continuous_theta_old_log_prob_recomputes(
        info=info,
        action_index=action_index,
        theta_rad=theta_rad,
        old_point_log_prob=point_log_prob,
        old_theta_log_prob=theta_log_prob,
    )


def _row_uses_synthetic_credit_behavior_policy(row: dict[str, Any]) -> bool:
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    return (row.get("behavior_policy_id") or info.get("behavior_policy_id")) == BEHAVIOR_POLICY_ID


def _row_has_synthetic_credit_behavior_logprob(row: dict[str, Any]) -> bool:
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    action_index = _int_or_none(row.get("action_index"))
    if action_index is None:
        action_index = _int_or_none(info.get("action_index"))
    old_action_probs = info.get("old_action_probs")
    if not isinstance(old_action_probs, list):
        old_action_probs = row.get("old_action_probs")
    if action_index is None or not isinstance(old_action_probs, list):
        return False
    mixture_probability = _finite(
        row.get("synthetic_credit_mixture_probability", info.get("synthetic_credit_mixture_probability"))
    )
    target_index = _int_or_none(row.get("synthetic_credit_target_index"))
    if target_index is None:
        target_index = _int_or_none(info.get("synthetic_credit_target_index"))
    theta_log_prob = _finite(row.get("old_policy_theta_log_prob"))
    if theta_log_prob is None:
        theta_log_prob = _finite(info.get("old_policy_theta_log_prob"))
    if theta_log_prob is None:
        theta_log_prob = _finite(row.get("old_theta_log_prob"))
    if theta_log_prob is None:
        theta_log_prob = _finite(info.get("old_theta_log_prob"))
    behavior_theta_log_prob = _finite(row.get("old_behavior_theta_log_prob"))
    if behavior_theta_log_prob is None:
        behavior_theta_log_prob = _finite(info.get("old_behavior_theta_log_prob"))
    theta_policy_id = (
        row.get("synthetic_credit_theta_policy_id")
        or info.get("synthetic_credit_theta_policy_id")
        or row.get("synthetic_credit_theta_behavior")
        or info.get("synthetic_credit_theta_behavior")
    )
    if behavior_theta_log_prob is None and theta_policy_id != "reachability_theta_proposal_mixture/v1":
        behavior_theta_log_prob = theta_log_prob
    old_behavior_point = _finite(row.get("old_behavior_point_log_prob"))
    if old_behavior_point is None:
        old_behavior_point = _finite(info.get("old_behavior_point_log_prob"))
    old_policy_point = _finite(row.get("old_policy_point_log_prob"))
    if old_policy_point is None:
        old_policy_point = _finite(info.get("old_policy_point_log_prob"))
    old_policy_total = _finite(row.get("old_policy_log_prob"))
    if old_policy_total is None:
        old_policy_total = _finite(info.get("old_policy_log_prob"))
    old_behavior_total = _finite(row.get("old_behavior_log_prob"))
    if old_behavior_total is None:
        old_behavior_total = _finite(info.get("old_behavior_log_prob"))
    old_total = _finite(row.get("old_log_prob"))
    if old_total is None:
        old_total = _finite(info.get("old_log_prob"))
    if (
        mixture_probability is None
        or old_behavior_point is None
        or old_policy_point is None
        or old_policy_total is None
        or old_behavior_total is None
        or old_total is None
        or behavior_theta_log_prob is None
    ):
        return False
    theta_info = dict(info)
    for key in (
        "synthetic_credit_theta_policy_id",
        "synthetic_credit_theta_behavior",
        "synthetic_credit_theta_proposals_deg",
        "synthetic_credit_theta_selected_proposal_index",
        "synthetic_credit_theta_reachable_proposal_count",
    ):
        if row.get(key) is not None:
            theta_info[key] = row.get(key)
    if not _behavior_theta_logprob_recomputes(info=theta_info, expected=behavior_theta_log_prob):
        return False
    try:
        recomputed = synthetic_credit_behavior_logprob(
            policy_probs=old_action_probs,
            action_index=action_index,
            target_index=target_index,
            mixture_probability=mixture_probability,
            theta_log_prob=theta_log_prob,
            behavior_theta_log_prob=behavior_theta_log_prob,
        )
    except (TypeError, ValueError, OverflowError):
        return False
    if abs(float(recomputed["old_behavior_point_log_prob"]) - float(old_behavior_point)) > 1.0e-5:
        return False
    if abs(float(recomputed["old_policy_point_log_prob"]) - float(old_policy_point)) > 1.0e-5:
        return False
    if abs(float(recomputed["old_policy_log_prob"]) - float(old_policy_total)) > 1.0e-5:
        return False
    if abs(float(recomputed["old_log_prob"]) - float(old_behavior_total)) > 1.0e-5:
        return False
    return abs(float(recomputed["old_log_prob"]) - float(old_total)) <= 1.0e-5


def _behavior_theta_logprob_recomputes(*, info: dict[str, Any], expected: float) -> bool:
    policy_id = info.get("synthetic_credit_theta_policy_id") or info.get("synthetic_credit_theta_behavior")
    if policy_id != "reachability_theta_proposal_mixture/v1":
        return True
    proposals = info.get("synthetic_credit_theta_proposals_deg")
    selected_index = _int_or_none(info.get("synthetic_credit_theta_selected_proposal_index"))
    reachable_count = _int_or_none(info.get("synthetic_credit_theta_reachable_proposal_count"))
    if not isinstance(proposals, list) or selected_index is None:
        return False
    if selected_index < 0 or selected_index >= len(proposals) or not proposals:
        return False
    support_count = reachable_count if reachable_count is not None and reachable_count > 0 else len(proposals)
    recomputed = -math.log(support_count)
    return abs(float(recomputed) - float(expected)) <= 1.0e-5


def _continuous_theta_old_log_prob_recomputes(
    *,
    info: dict[str, Any],
    action_index: int,
    theta_rad: float,
    old_point_log_prob: float,
    old_theta_log_prob: float,
) -> bool:
    logits = info.get("old_sampling_logits")
    theta_mu = info.get("theta_mu_rad")
    theta_kappa = info.get("theta_kappa")
    if not isinstance(logits, list) or not isinstance(theta_mu, list) or not isinstance(theta_kappa, list):
        return False
    if action_index < 0 or action_index >= len(logits) or action_index >= len(theta_mu) or action_index >= len(theta_kappa):
        return False
    try:
        logits_tensor = torch.tensor([float(value) for value in logits], dtype=torch.float64)
        action_tensor = torch.tensor(int(action_index), dtype=torch.long)
        point_log_prob = torch.distributions.Categorical(logits=logits_tensor).log_prob(action_tensor)
        theta_dist = torch.distributions.VonMises(
            torch.tensor(float(theta_mu[action_index]), dtype=torch.float64),
            torch.tensor(float(theta_kappa[action_index]), dtype=torch.float64),
        )
        theta_log_prob = theta_dist.log_prob(torch.tensor(float(theta_rad), dtype=torch.float64))
    except (TypeError, ValueError, RuntimeError):
        return False
    return (
        abs(float(point_log_prob.item()) - float(old_point_log_prob)) <= 1.0e-5
        and abs(float(theta_log_prob.item()) - float(old_theta_log_prob)) <= 1.0e-5
    )


def _angle_deg_delta_abs(lhs: float, rhs: float) -> float:
    return abs(((float(lhs) - float(rhs) + 180.0) % 360.0) - 180.0)


def _present_viewpoint(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 3


def _theta_reward_matches_selected_action(row: dict[str, Any]) -> bool:
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    action_index = _int_or_none(row.get("action_index"))
    selected_viewpoint = _selected_value(info, "candidate_viewpoint", "candidate_viewpoints", action_index)
    if selected_viewpoint is None:
        selected_viewpoint = info.get("selected_viewpoint")
    if not _present_viewpoint(selected_viewpoint):
        return False
    selected_theta = _selected_value(info, "candidate_theta_deg", "candidate_theta_degs", action_index)
    if selected_theta is None:
        selected_theta = info.get("selected_theta_deg")
    if selected_theta is None and _present_viewpoint(selected_viewpoint):
        selected_theta = selected_viewpoint[2]
    if _present_viewpoint(selected_viewpoint) and list(selected_viewpoint) != list(row.get("candidate_viewpoint")):
        return False
    selected_theta_value = _finite(selected_theta)
    reward_theta_value = _finite(row.get("candidate_theta_deg"))
    if (
        selected_theta_value is not None
        and reward_theta_value is not None
        and abs(float(selected_theta_value) - float(reward_theta_value)) > 1.0e-6
    ):
        return False
    return True


def _theta_reward_metrics_consistent(row: dict[str, Any]) -> bool:
    if row.get("coverage_source") == SLOPE_OBSTACLE_COVERAGE_SOURCE:
        return _slope_obstacle_reward_metrics_consistent(row)
    metrics = row.get("reward_metrics") if isinstance(row.get("reward_metrics"), dict) else {}
    coverage_rate_delta = _finite(metrics.get("coverage_rate_delta"))
    denominator = _finite(row.get("theta_coverage_denominator_cells"))
    theta_new = _finite(row.get("theta_new_visible_cell_count"))
    if coverage_rate_delta is None or denominator is None or denominator <= 0.0 or theta_new is None:
        return False
    return abs(coverage_rate_delta - (theta_new / denominator)) <= 1.0e-9


def _slope_obstacle_reward_metrics_consistent(row: dict[str, Any]) -> bool:
    metrics = row.get("reward_metrics") if isinstance(row.get("reward_metrics"), dict) else {}
    coverage_rate_delta = _finite(metrics.get("coverage_rate_delta"))
    denominator = _finite(row.get("obstacle_aware_theta_coverage_denominator_cells"))
    obstacle_new = _finite(row.get("obstacle_aware_new_visible_cell_count"))
    if coverage_rate_delta is None or denominator is None or denominator <= 0.0 or obstacle_new is None:
        return False
    return abs(coverage_rate_delta - (obstacle_new / denominator)) <= 1.0e-9


def _row_has_transition_theta_viewpoint_contract(row: dict[str, Any]) -> bool:
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    observation = row.get("observation") if isinstance(row.get("observation"), dict) else {}
    return row_has_theta_viewpoint_contract({"info": info, "observation": observation})


def _selected_value(info: dict[str, Any], singular_key: str, plural_key: str, action_index: int | None) -> Any:
    if singular_key in info:
        value = info.get(singular_key)
        if action_index is not None and isinstance(value, list):
            if singular_key == "candidate_viewpoint":
                if value and isinstance(value[0], list) and 0 <= action_index < len(value):
                    return value[action_index]
                return value
            if 0 <= action_index < len(value):
                return value[action_index]
        return value
    values = info.get(plural_key)
    if action_index is not None and isinstance(values, list) and 0 <= action_index < len(values):
        return values[action_index]
    return None


def _hard_risk_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if isinstance(row.get("info"), dict) and row["info"].get("hard_risk_violation") is True)


def _missing_next_observation_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("done") is not True and row.get("next_observation") is None)


def _old_log_prob_recompute_violation_count(rows: list[dict[str, Any]], threshold: float) -> int:
    count = 0
    for row in rows:
        value = _finite(row.get("old_log_prob_recompute_abs_error"))
        if value is None or value > threshold:
            count += 1
    return count


def _old_log_prob_recompute_max_abs_error(rows: list[dict[str, Any]]) -> float | None:
    values = [_finite(row.get("old_log_prob_recompute_abs_error")) for row in rows]
    finite_values = [float(value) for value in values if value is not None]
    return max(finite_values) if finite_values else None


def _episode_boundary_audit(transitions: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in transitions:
        grouped.setdefault(str(row.get("scenario_id") or ""), []).append(row)
    audit = {
        "scenario_count": len(grouped),
        "non_monotonic_episode_count": 0,
        "duplicate_step_index_episode_count": 0,
        "non_contiguous_step_index_episode_count": 0,
        "terminal_count_mismatch_episode_count": 0,
        "terminal_not_last_episode_count": 0,
        "terminal_after_transition_count": 0,
        "scenario_summaries": [],
    }
    for scenario_id, rows in sorted(grouped.items()):
        original_steps = [_int_or_none(row.get("step_index")) for row in rows]
        valid_steps = [step for step in original_steps if step is not None]
        sorted_rows = sorted(rows, key=lambda row: _int_or_none(row.get("step_index")) if _int_or_none(row.get("step_index")) is not None else -1)
        sorted_steps = [_int_or_none(row.get("step_index")) for row in sorted_rows]
        duplicate_step_count = len(valid_steps) - len(set(valid_steps))
        terminal_positions = [index for index, row in enumerate(sorted_rows) if row.get("done") is True]
        contiguous = bool(valid_steps) and sorted(valid_steps) == list(range(min(valid_steps), max(valid_steps) + 1))
        monotonic = original_steps == sorted(valid_steps) if len(original_steps) == len(valid_steps) else False
        terminal_count = len(terminal_positions)
        terminal_not_last = terminal_count == 1 and terminal_positions[0] != len(sorted_rows) - 1
        if not monotonic:
            audit["non_monotonic_episode_count"] += 1
        if duplicate_step_count:
            audit["duplicate_step_index_episode_count"] += 1
        if not contiguous:
            audit["non_contiguous_step_index_episode_count"] += 1
        if terminal_count != 1:
            audit["terminal_count_mismatch_episode_count"] += 1
        if terminal_not_last:
            audit["terminal_not_last_episode_count"] += 1
            audit["terminal_after_transition_count"] += len(sorted_rows) - terminal_positions[0] - 1
        audit["scenario_summaries"].append(
            {
                "scenario_id": scenario_id,
                "transition_count": len(rows),
                "step_index_min": min(valid_steps) if valid_steps else None,
                "step_index_max": max(valid_steps) if valid_steps else None,
                "duplicate_step_count": duplicate_step_count,
                "terminal_count": terminal_count,
                "terminal_not_last": terminal_not_last,
            }
        )
    return audit


def _first_row_by_transition_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        transition_id = str(row.get("transition_id") or "")
        if transition_id and transition_id not in by_id:
            by_id[transition_id] = row
    return by_id


def _duplicate_count(values: list[str]) -> int:
    counts = Counter(value for value in values if value)
    return sum(count - 1 for count in counts.values() if count > 1)


def _duplicate_values(values: list[str]) -> list[str]:
    counts = Counter(value for value in values if value)
    return sorted(value for value, count in counts.items() if count > 1)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stable_bucket(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage 21.3 PPO Batch Validation",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- trainable_transition_count: `{summary['trainable_transition_count']}`",
            f"- return_finite_count: `{summary['return_finite_count']}`",
            f"- advantage_finite_count: `{summary['advantage_finite_count']}`",
            f"- invalid_action_count: `{summary['invalid_action_count']}`",
            f"- hard_risk_violation_count: `{summary['hard_risk_violation_count']}`",
            f"- old_log_prob_recompute_max_abs_error: `{summary['old_log_prob_recompute_max_abs_error']}`",
            f"- duplicate_transition_id_count: `{summary['duplicate_transition_id_count']}`",
            f"- duplicate_reward_transition_id_count: `{summary['duplicate_reward_transition_id_count']}`",
            f"- reward_trainable_false_count: `{summary['reward_trainable_false_count']}`",
            f"- train_transition_count: `{summary['train_transition_count']}`",
            f"- validation_transition_count: `{summary['validation_transition_count']}`",
            "",
            "Stage 21.3 validates the PPO batch only. It does not run PPO, publish checkpoints, replace default policy, connect a real executor, or start canary traffic.",
        ]
    ) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"JSON file is invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"JSON root must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ConfigError(f"JSONL file does not exist: {path}") from exc
    for line in lines:
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _range_float(value: Any, field: str) -> float:
    parsed = _finite(value)
    if parsed is None or parsed <= 0.0 or parsed > 1.0:
        raise ConfigError(f"{field} must be in (0, 1]")
    return parsed


def _fraction(value: Any, field: str) -> float:
    parsed = _finite(value)
    if parsed is None or parsed < 0.0 or parsed >= 1.0:
        raise ConfigError(f"{field} must be in [0, 1)")
    return parsed


def _nonnegative_float(value: Any, field: str) -> float:
    parsed = _finite(value)
    if parsed is None or parsed < 0.0:
        raise ConfigError(f"{field} must be finite and >= 0")
    return parsed


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be an integer") from exc
    if parsed < 0:
        raise ConfigError(f"{field} must be >= 0")
    return parsed


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
