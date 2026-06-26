from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:  # pragma: no cover - direct script execution
    import run_xunce_stage21_5_post_update_offline_trajectory_evaluation as stage21_5
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage21_5_post_update_offline_trajectory_evaluation as stage21_5


CONFIG_SCHEMA_VERSION = "xunce-stage24-5-hybrid-astar-path-cost-post-update-trajectory-eval-smoke-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage24-5-summary/v1"
ACTION_AUDIT_SCHEMA_VERSION = "xunce-stage24-5-hybrid-path-action-change-audit/v1"
DELTA_AUDIT_SCHEMA_VERSION = "xunce-stage24-5-trajectory-delta-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage24-5-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage24-5-manifest/v1"

STAGE_ID = "xunce-stage24-5-hybrid-astar-path-cost-post-update-trajectory-eval-smoke"
DEFAULT_CONFIG = "configs/xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage24_hybrid_astar_pose_planner/"
    "outputs/path_feedback_batch_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke_v1"
)

SLOPE_COVERAGE_SOURCE = "endpoint_theta_slope_obstacle_los/v1"
SLOPE_SOURCE_KIND = "slope_blocked_as_obstacle_proxy"
HYBRID_ASTAR_PATH_COST_SOURCE = "hybrid_astar_pose_path/v1"
ROUTE_INPUTS = "rerun_stage24_5_required_inputs"
ROUTE_BINDING = "repair_stage24_5_hybrid_path_inference_binding"
ROUTE_SAFETY = "repair_stage24_5_hybrid_path_eval_safety_regression"
ROUTE_SIGNAL = "repair_stage24_hybrid_path_policy_update_signal_strength"
ROUTE_MARGIN = "calibrate_stage24_hybrid_path_discrete_margin_crossing"
ROUTE_CREDIT = "repair_stage24_hybrid_path_credit_assignment"
ROUTE_STAGE24_6 = "run_stage24_6_hybrid_astar_path_cost_multi_seed_ppo_pilot"
ROUTE_BOUNDARY = "resolve_stage24_5_boundary_rejections"
ROUTE_FROM_STAGE24_4 = "run_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke"

SUMMARY_FILE = "xunce-stage24-5-summary.json"
STAGE21_5_CONFIG_FILE = "xunce-stage24-5-stage21-5-config.json"
STAGE21_5_SUMMARY_FILE = "xunce-stage24-5-stage21-5-summary.json"
ACTION_AUDIT_FILE = "xunce-stage24-5-hybrid-path-action-change-audit.json"
DELTA_AUDIT_FILE = "xunce-stage24-5-trajectory-delta-audit.json"
ROUTING_FILE = "xunce-stage24-5-next-stage-routing.json"
REPORT_FILE = "xunce-stage24-5-report.md"
MANIFEST_FILE = "xunce-stage24-5-manifest.json"
GENERATED_HIGH_FIDELITY_CONFIG_FILE = "xunce-stage24-5-high-fidelity-config.json"
MODEL_INFERENCE_FILE = "xunce-exploration-coverage-model-inference.jsonl"

BOUNDARY_FIELDS = (
    "stage24_5_authorized",
    "release_or_training_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)
HYBRID_ASTAR_PLANNER_FIELDS = (
    "hybrid_astar_theta_bin_count",
    "hybrid_astar_goal_position_tolerance_m",
    "hybrid_astar_goal_theta_tolerance_deg",
    "hybrid_astar_max_iterations",
    "hybrid_astar_primitive_duration_s",
    "hybrid_astar_integration_dt_s",
    "hybrid_astar_max_speed_mps",
    "hybrid_astar_max_angular_speed_degps",
    "hybrid_astar_rotation_cost_weight",
    "hybrid_astar_reverse_penalty_weight",
    "hybrid_astar_turn_penalty_weight",
)

REQUIRED_INFERENCE_FIELDS = (
    "scenario_id",
    "step_index",
    "current_cell",
    "covered_cells_hash",
    "candidate_set_hash",
    "selected_action_index",
    "candidate_viewpoint",
    "candidate_theta_deg",
    "selected_viewpoint",
    "selected_theta_deg",
    "action_probs",
    "logits",
    "coverage_source",
    "path_cost_source",
    "hybrid_astar_path_cost",
    "hybrid_astar_pose_path_hash",
    "hybrid_astar_trajectory_kind",
    "legacy_grid_astar_path_cost",
    "hybrid_vs_grid_path_cost_delta",
    "point_grid_path_cost_fallback_used",
    "default_astar_replaced",
    "hybrid_astar_ackermann_feasible_claimed",
    "obstacle_occlusion_enabled",
    "obstacle_source_kind",
    "obstacle_aware_theta_coverage_hash",
    "slope_obstacle_source_hash",
    "platform_contract_hash",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage24.5 Hybrid A* path-cost post-update trajectory eval smoke.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    summary = run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(config_path, repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    boundary_reasons = _boundary_rejections(config)
    stage24_4_summary = _read_json_if_exists(Path(config["stage24_4_root"]) / "xunce-stage24-4-summary.json")
    stage24_3_summary = _stage24_3_summary(stage24_4_summary)
    input_reasons = _input_rejections(config, stage24_4_summary, stage24_3_summary)

    stage21_5_summary: dict[str, Any] = {}
    stage21_5_config_path = output_root / STAGE21_5_CONFIG_FILE
    high_fidelity_config_path = output_root / GENERATED_HIGH_FIDELITY_CONFIG_FILE
    if not boundary_reasons and not input_reasons:
        stage21_5_summary = _run_stage21_5(
            config,
            stage24_3_summary,
            output_root,
            repo_root,
            stage21_5_config_path,
            high_fidelity_config_path,
        )

    action_audit = _hybrid_path_action_change_audit(stage21_5_summary, output_root / "s21_5")
    delta_audit = _trajectory_delta_audit(stage21_5_summary, output_root / "s21_5")
    stage21_5_execution_reasons = _stage21_5_execution_rejections(stage21_5_summary)
    safety_reasons = _safety_rejections(stage21_5_summary)
    status, route, route_reason = _route(
        boundary_reasons,
        input_reasons,
        stage21_5_execution_reasons,
        safety_reasons,
        stage21_5_summary,
        action_audit,
        delta_audit,
        float(config["min_mean_abs_probability_delta_for_signal"]),
    )
    blocking = _unique(
        boundary_reasons
        + input_reasons
        + stage21_5_execution_reasons
        + safety_reasons
        + _route_blocking_reasons(route, action_audit, stage21_5_summary)
    )

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": blocking,
        "stage24_4_root": config["stage24_4_root"],
        "stage24_4_status": stage24_4_summary.get("status"),
        "stage24_4_next_required_change": stage24_4_summary.get("next_required_change"),
        "stage24_3_root": stage24_4_summary.get("stage24_3_root"),
        "stage23_2b_high_res_root": stage24_3_summary.get("stage23_2b_high_res_root"),
        "stage21_5_root": str(output_root / "s21_5"),
        "stage21_5_status": stage21_5_summary.get("status"),
        "stage21_5_next_required_change": stage21_5_summary.get("next_required_change"),
        "stage21_5_execution_incomplete": bool(stage21_5_execution_reasons),
        "stage21_5_execution_reason_codes": stage21_5_execution_reasons,
        "action_probability_audit_is_partial_diagnostic": bool(stage21_5_execution_reasons),
        "coverage_source": SLOPE_COVERAGE_SOURCE,
        "path_cost_source": HYBRID_ASTAR_PATH_COST_SOURCE,
        "max_traversable_slope_deg": 30.0,
        "slope_blocked_source_kind": SLOPE_SOURCE_KIND,
        "default_astar_replaced": False,
        "ackermann_feasible_claimed": False,
        "stage24_4_pre_clip_grad_norm": stage24_4_summary.get("pre_clip_grad_norm"),
        **_summary_counts(action_audit, delta_audit, stage21_5_summary),
        "release_or_training_authorized": False,
        "stage24_5_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "stage24_5_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "summary": str(output_root / SUMMARY_FILE),
        "stage21_5_config": str(stage21_5_config_path),
        "stage21_5_summary": str(output_root / STAGE21_5_SUMMARY_FILE),
        "hybrid_path_action_change_audit": str(output_root / ACTION_AUDIT_FILE),
        "trajectory_delta_audit": str(output_root / DELTA_AUDIT_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / STAGE21_5_SUMMARY_FILE, stage21_5_summary)
    _write_json(output_root / ACTION_AUDIT_FILE, action_audit)
    _write_json(output_root / DELTA_AUDIT_FILE, delta_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    _write_json(output_root / MANIFEST_FILE, manifest)
    return summary


def _run_stage21_5(
    config: dict[str, Any],
    stage24_3_summary: dict[str, Any],
    output_root: Path,
    repo_root: Path,
    stage21_5_config_path: Path,
    high_fidelity_config_path: Path,
) -> dict[str, Any]:
    cfg = _load_json_template(config["stage21_5_base_config"], repo_root)
    high_fidelity_cfg = _load_json_template(config["high_fidelity_config"], repo_root)
    high_res_root = str(stage24_3_summary.get("stage23_2b_high_res_root") or stage24_3_summary.get("stage21_1_actual_source_roi_expansion_root") or "")
    high_fidelity_cfg.update(
        {
            "theta_aware_candidate_viewpoints_enabled": True,
            "slope_obstacle_aware_theta_reward_enabled": True,
            "hybrid_astar_pose_path_cost_enabled": True,
            "obstacle_occlusion_enabled": True,
            "coverage_source": SLOPE_COVERAGE_SOURCE,
            "path_cost_source": HYBRID_ASTAR_PATH_COST_SOURCE,
            "max_traversable_slope_deg": 30.0,
            "theta_bin_count": int(config["theta_bin_count"]),
            "theta_step_deg": int(config["theta_step_deg"]),
            "sensor_model_id": str(config["sensor_model_id"]),
            "sensor_fov_deg": float(config["sensor_fov_deg"]),
            "sensor_range_cells": int(config["sensor_range_cells"]),
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            **{field: config[field] for field in HYBRID_ASTAR_PLANNER_FIELDS},
            "dynamic_validation_work_root": str(output_root / "_xunce_dynamic_validation_work"),
            "source_roi_expansion_root": high_res_root,
            "emit_candidate_metric_audit": True,
            "default_astar_replaced": False,
            "hybrid_astar_ackermann_feasible_claimed": False,
        }
    )
    _write_json(high_fidelity_config_path, high_fidelity_cfg)
    stage24_4_root = Path(config["stage24_4_root"])
    cfg.update(
        {
            "stage21_4_tiny_ppo_update_smoke_root": str(stage24_4_root / "s21_4"),
            "high_fidelity_config": str(high_fidelity_config_path),
            "execute_high_fidelity_evaluations": True,
            "pre_ppo_evaluation_root": str(output_root / "pre"),
            "post_ppo_evaluation_root": str(output_root / "post"),
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "theta_aware_candidate_viewpoints_enabled": True,
            "slope_obstacle_aware_theta_reward_enabled": True,
            "hybrid_astar_pose_path_cost_enabled": True,
            "obstacle_occlusion_enabled": True,
            "coverage_source": SLOPE_COVERAGE_SOURCE,
            "path_cost_source": HYBRID_ASTAR_PATH_COST_SOURCE,
            "max_traversable_slope_deg": 30.0,
            "theta_bin_count": int(config["theta_bin_count"]),
            "theta_step_deg": int(config["theta_step_deg"]),
            "sensor_model_id": str(config["sensor_model_id"]),
            "sensor_fov_deg": float(config["sensor_fov_deg"]),
            "sensor_range_cells": int(config["sensor_range_cells"]),
            **{field: config[field] for field in HYBRID_ASTAR_PLANNER_FIELDS},
            "default_astar_replaced": False,
            "hybrid_astar_ackermann_feasible_claimed": False,
            "stage21_5_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    _write_json(stage21_5_config_path, cfg)
    return stage21_5.run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=stage21_5_config_path,
        output_root=output_root / "s21_5",
        repo_root=repo_root,
    )


def _hybrid_path_action_change_audit(stage21_5_summary: dict[str, Any], stage21_5_root: Path) -> dict[str, Any]:
    pre_root = Path(str(stage21_5_summary.get("pre_evaluation_root") or stage21_5_root / "pre_ppo_xunce"))
    post_root = Path(str(stage21_5_summary.get("post_evaluation_root") or stage21_5_root / "post_ppo_xunce"))
    pre_rows = _read_jsonl_if_exists(pre_root / MODEL_INFERENCE_FILE)
    post_rows = _read_jsonl_if_exists(post_root / MODEL_INFERENCE_FILE)
    pre_index, pre_duplicate_count = _index_inference_rows(pre_rows)
    post_index, post_duplicate_count = _index_inference_rows(post_rows)
    joined = missing_post = required_missing = 0
    coverage_source_mismatch = path_cost_source_mismatch = slope_mismatch = obstacle_contract_mismatch = 0
    hybrid_contract_mismatch = grid_fallback_count = default_astar_replaced_count = ackermann_claimed_count = 0
    viewpoint_changed = theta_changed = cell_changed = action_changed = 0
    probability_abs_delta_sum = probability_value_count = 0
    selected_probability_delta_sum = selected_probability_delta_count = 0
    blocking_examples: list[dict[str, Any]] = []
    for key, pre in pre_index.items():
        post = post_index.get(key)
        if post is None:
            missing_post += 1
            continue
        joined += 1
        for row in (pre, post):
            if row.get("coverage_source") != SLOPE_COVERAGE_SOURCE:
                coverage_source_mismatch += 1
            if row.get("path_cost_source") != HYBRID_ASTAR_PATH_COST_SOURCE:
                path_cost_source_mismatch += 1
            if _finite_float(row.get("max_traversable_slope_deg")) != 30.0:
                slope_mismatch += 1
            if row.get("obstacle_occlusion_enabled") is not True:
                obstacle_contract_mismatch += 1
            if row.get("obstacle_source_kind") != SLOPE_SOURCE_KIND:
                obstacle_contract_mismatch += 1
            if not str(row.get("obstacle_aware_theta_coverage_hash") or "").strip():
                obstacle_contract_mismatch += 1
            if not str(row.get("slope_obstacle_source_hash") or "").strip():
                obstacle_contract_mismatch += 1
            if not str(row.get("platform_contract_hash") or "").strip():
                obstacle_contract_mismatch += 1
            if row.get("hybrid_astar_trajectory_kind") != "hybrid_astar_pose_path":
                hybrid_contract_mismatch += 1
            if _finite_float(row.get("hybrid_astar_path_cost")) is None:
                hybrid_contract_mismatch += 1
            if not str(row.get("hybrid_astar_pose_path_hash") or "").strip():
                hybrid_contract_mismatch += 1
            if _finite_float(row.get("legacy_grid_astar_path_cost")) is None:
                hybrid_contract_mismatch += 1
            if _finite_float(row.get("hybrid_vs_grid_path_cost_delta")) is None:
                hybrid_contract_mismatch += 1
            if row.get("point_grid_path_cost_fallback_used") is True:
                grid_fallback_count += 1
            if row.get("default_astar_replaced") is True:
                default_astar_replaced_count += 1
            if row.get("hybrid_astar_ackermann_feasible_claimed") is True:
                ackermann_claimed_count += 1
        row_missing = _missing_required_fields(pre) + _missing_required_fields(post)
        if row_missing:
            required_missing += len(row_missing)
            if len(blocking_examples) < 5:
                blocking_examples.append({"key": key, "missing": row_missing})
            continue
        if _normalize_json_value(pre.get("selected_viewpoint")) != _normalize_json_value(post.get("selected_viewpoint")):
            viewpoint_changed += 1
        if _cell(pre) != _cell(post):
            cell_changed += 1
        if _theta(pre) != _theta(post):
            theta_changed += 1
        if pre.get("selected_action_index") != post.get("selected_action_index"):
            action_changed += 1
        deltas = _probability_deltas(pre.get("action_probs"), post.get("action_probs"))
        probability_abs_delta_sum += sum(deltas)
        probability_value_count += len(deltas)
        selected_index = _int_or_none(pre.get("selected_action_index"))
        if selected_index is not None:
            pre_probs = pre.get("action_probs") if isinstance(pre.get("action_probs"), list) else []
            post_probs = post.get("action_probs") if isinstance(post.get("action_probs"), list) else []
            if selected_index < len(pre_probs) and selected_index < len(post_probs):
                pre_value = _finite_float(pre_probs[selected_index])
                post_value = _finite_float(post_probs[selected_index])
                if pre_value is not None and post_value is not None:
                    selected_probability_delta_sum += abs(post_value - pre_value)
                    selected_probability_delta_count += 1
    return {
        "schema_version": ACTION_AUDIT_SCHEMA_VERSION,
        "pre_model_inference_root": str(pre_root),
        "post_model_inference_root": str(post_root),
        "pre_model_inference_row_count": len(pre_rows),
        "post_model_inference_row_count": len(post_rows),
        "strong_state_join_available_count": joined,
        "missing_post_counterpart_count": missing_post,
        "pre_duplicate_strong_key_count": pre_duplicate_count,
        "post_duplicate_strong_key_count": post_duplicate_count,
        "hybrid_path_inference_required_field_missing_count": required_missing,
        "coverage_source_mismatch_count": coverage_source_mismatch,
        "path_cost_source_mismatch_count": path_cost_source_mismatch,
        "max_traversable_slope_mismatch_count": slope_mismatch,
        "obstacle_contract_mismatch_count": obstacle_contract_mismatch,
        "hybrid_path_contract_mismatch_count": hybrid_contract_mismatch,
        "grid_fallback_count": grid_fallback_count,
        "default_astar_replaced_count": default_astar_replaced_count,
        "ackermann_feasible_claimed_count": ackermann_claimed_count,
        "selected_viewpoint_changed_count": viewpoint_changed,
        "selected_theta_changed_count": theta_changed,
        "selected_cell_changed_count": cell_changed,
        "selected_action_changed_count": action_changed,
        "mean_abs_probability_delta": probability_abs_delta_sum / probability_value_count if probability_value_count else 0.0,
        "selected_action_probability_delta_mean": (
            selected_probability_delta_sum / selected_probability_delta_count if selected_probability_delta_count else 0.0
        ),
        "selected_action_probability_delta": (
            selected_probability_delta_sum / selected_probability_delta_count if selected_probability_delta_count else 0.0
        ),
        "blocking_examples": blocking_examples,
    }


def _trajectory_delta_audit(stage21_5_summary: dict[str, Any], stage21_5_root: Path) -> dict[str, Any]:
    delta = _read_json_if_exists(stage21_5_root / stage21_5.DELTA_FILE)
    return {
        "schema_version": DELTA_AUDIT_SCHEMA_VERSION,
        "final_coverage_delta": _first_finite(
            delta.get("final_coverage_rate_mean_delta"),
            delta.get("final_coverage_rate_delta"),
            stage21_5_summary.get("final_coverage_rate_delta"),
            stage21_5_summary.get("mean_final_coverage_delta"),
        ),
        "coverage_auc_delta": _first_finite(
            delta.get("coverage_curve_auc_mean_delta"),
            delta.get("coverage_curve_auc_delta"),
            stage21_5_summary.get("coverage_curve_auc_delta"),
            stage21_5_summary.get("mean_coverage_auc_delta"),
        ),
        "path_cost_delta": _first_finite(
            delta.get("path_cost_total_m_mean_delta"),
            delta.get("path_cost_total_m_delta"),
            stage21_5_summary.get("path_cost_total_m_delta"),
            stage21_5_summary.get("mean_path_cost_delta_m"),
        ),
        "coverage_per_100m_delta": _first_finite(
            delta.get("coverage_per_100m_mean_delta"),
            delta.get("coverage_per_100m_delta"),
            stage21_5_summary.get("coverage_per_100m_delta"),
            stage21_5_summary.get("mean_coverage_per_100m_delta"),
        ),
        "raw_delta": delta,
    }


def _route(
    boundary_reasons: list[str],
    input_reasons: list[str],
    stage21_5_execution_reasons: list[str],
    safety_reasons: list[str],
    stage21_5_summary: dict[str, Any],
    action_audit: dict[str, Any],
    delta_audit: dict[str, Any],
    min_probability_delta: float,
) -> tuple[str, str, str]:
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "boundary flag enabled"
    if input_reasons:
        return "failed", ROUTE_INPUTS, "Stage24.4 input is not ready for Stage24.5"
    if stage21_5_execution_reasons:
        return "failed", ROUTE_INPUTS, "Stage21.5 evaluation did not satisfy Stage24.5 smoke input requirements"
    if safety_reasons:
        return "failed", ROUTE_SAFETY, "post-update trajectory safety regression detected"
    if stage21_5_summary.get("status") not in {"passed", "failed"}:
        return "failed", ROUTE_INPUTS, "Stage21.5 did not produce a readable summary"
    if _binding_failed(action_audit):
        return "failed", ROUTE_BINDING, "Hybrid path pre/post inference strong binding is incomplete"
    changed = (
        int(action_audit.get("selected_viewpoint_changed_count") or 0) > 0
        or int(action_audit.get("selected_theta_changed_count") or 0) > 0
    )
    prob_delta = float(action_audit.get("mean_abs_probability_delta") or 0.0)
    final_delta = float(delta_audit.get("final_coverage_delta") or 0.0)
    auc_delta = float(delta_audit.get("coverage_auc_delta") or 0.0)
    if not changed and prob_delta < min_probability_delta:
        return "failed", ROUTE_SIGNAL, "Hybrid path PPO update did not move probabilities or selected viewpoint"
    if not changed:
        return "failed", ROUTE_MARGIN, "Hybrid path probabilities moved but did not cross the discrete viewpoint boundary"
    if int(stage21_5_summary.get("scenario_regression_count") or 0) > 0:
        return "failed", ROUTE_CREDIT, "Hybrid path smoke has at least one scenario-level regression"
    if final_delta > 0.0 and auc_delta > 0.0:
        return "passed", ROUTE_STAGE24_6, "Hybrid path post-update smoke improved coverage and AUC without safety regression"
    return "failed", ROUTE_CREDIT, "Hybrid path action changed but coverage/AUC did not improve in the smoke"


def _binding_failed(action_audit: dict[str, Any]) -> bool:
    return (
        int(action_audit.get("strong_state_join_available_count") or 0) <= 0
        or int(action_audit.get("hybrid_path_inference_required_field_missing_count") or 0) > 0
        or int(action_audit.get("coverage_source_mismatch_count") or 0) > 0
        or int(action_audit.get("path_cost_source_mismatch_count") or 0) > 0
        or int(action_audit.get("max_traversable_slope_mismatch_count") or 0) > 0
        or int(action_audit.get("obstacle_contract_mismatch_count") or 0) > 0
        or int(action_audit.get("hybrid_path_contract_mismatch_count") or 0) > 0
        or int(action_audit.get("grid_fallback_count") or 0) > 0
        or int(action_audit.get("default_astar_replaced_count") or 0) > 0
        or int(action_audit.get("ackermann_feasible_claimed_count") or 0) > 0
        or int(action_audit.get("pre_duplicate_strong_key_count") or 0) > 0
        or int(action_audit.get("post_duplicate_strong_key_count") or 0) > 0
    )


def _safety_rejections(stage21_5_summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    checks = {
        "post_ppo_hard_risk_violation": stage21_5_summary.get("post_hard_risk_violation_count"),
        "post_ppo_mask_violation": stage21_5_summary.get("post_mask_violation_count"),
        "post_ppo_path_planning_failure": stage21_5_summary.get("post_path_planning_failure_count"),
        "post_ppo_open_grid_fallback": stage21_5_summary.get("post_open_grid_fallback_count"),
    }
    for reason, value in checks.items():
        if int(value or 0) > 0:
            reasons.append(reason)
    if stage21_5_summary.get("next_required_change") == stage21_5.ROUTE_HARD_RISK:
        reasons.append("stage21_5_hard_risk_route")
    return reasons


def _stage21_5_execution_rejections(stage21_5_summary: dict[str, Any]) -> list[str]:
    if not stage21_5_summary:
        return []
    reasons: list[str] = []
    reason_codes = [str(value) for value in stage21_5_summary.get("reason_codes") or []]
    incomplete_markers = (
        "scenario_count_short",
        "xunce_episode_count_short",
        "evaluation_summary_missing",
        "evaluation_episodes_missing",
        "high_fidelity_roi_expansion_slice_count_short",
    )
    for code in reason_codes:
        if any(marker in code for marker in incomplete_markers):
            reasons.append(f"stage21_5_{code}")
    if stage21_5_summary.get("next_required_change") == stage21_5.ROUTE_EXECUTION:
        reasons.append("stage21_5_offline_evaluation_execution_incomplete")
    if stage21_5_summary.get("status") == "failed" and reasons:
        return _unique(reasons)
    return []


def _summary_counts(action_audit: dict[str, Any], delta_audit: dict[str, Any], stage21_5_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "strong_state_join_available_count": int(action_audit.get("strong_state_join_available_count") or 0),
        "pre_duplicate_strong_key_count": int(action_audit.get("pre_duplicate_strong_key_count") or 0),
        "post_duplicate_strong_key_count": int(action_audit.get("post_duplicate_strong_key_count") or 0),
        "hybrid_path_inference_required_field_missing_count": int(action_audit.get("hybrid_path_inference_required_field_missing_count") or 0),
        "coverage_source_mismatch_count": int(action_audit.get("coverage_source_mismatch_count") or 0),
        "path_cost_source_mismatch_count": int(action_audit.get("path_cost_source_mismatch_count") or 0),
        "max_traversable_slope_mismatch_count": int(action_audit.get("max_traversable_slope_mismatch_count") or 0),
        "obstacle_contract_mismatch_count": int(action_audit.get("obstacle_contract_mismatch_count") or 0),
        "hybrid_path_contract_mismatch_count": int(action_audit.get("hybrid_path_contract_mismatch_count") or 0),
        "grid_fallback_count": int(action_audit.get("grid_fallback_count") or 0),
        "default_astar_replaced_count": int(action_audit.get("default_astar_replaced_count") or 0),
        "ackermann_feasible_claimed_count": int(action_audit.get("ackermann_feasible_claimed_count") or 0),
        "selected_viewpoint_changed_count": int(action_audit.get("selected_viewpoint_changed_count") or 0),
        "selected_theta_changed_count": int(action_audit.get("selected_theta_changed_count") or 0),
        "selected_cell_changed_count": int(action_audit.get("selected_cell_changed_count") or 0),
        "selected_action_changed_count": int(action_audit.get("selected_action_changed_count") or 0),
        "mean_abs_probability_delta": float(action_audit.get("mean_abs_probability_delta") or 0.0),
        "selected_action_probability_delta_mean": float(action_audit.get("selected_action_probability_delta_mean") or 0.0),
        "selected_action_probability_delta": float(action_audit.get("selected_action_probability_delta") or 0.0),
        "final_coverage_delta": _first_finite(delta_audit.get("final_coverage_delta"), 0.0),
        "coverage_auc_delta": _first_finite(delta_audit.get("coverage_auc_delta"), 0.0),
        "path_cost_delta": _first_finite(delta_audit.get("path_cost_delta"), 0.0),
        "coverage_per_100m_delta": _first_finite(delta_audit.get("coverage_per_100m_delta"), 0.0),
        "hard_risk_violation_count": int(stage21_5_summary.get("post_hard_risk_violation_count") or 0),
        "mask_violation_count": int(stage21_5_summary.get("post_mask_violation_count") or 0),
        "path_planning_failure_count": int(stage21_5_summary.get("post_path_planning_failure_count") or 0),
        "open_grid_fallback_count": int(stage21_5_summary.get("post_open_grid_fallback_count") or 0),
        "scenario_regression_count": int(stage21_5_summary.get("scenario_regression_count") or 0),
        "sample_count_too_low_for_performance_claim": bool(stage21_5_summary.get("sample_count_too_low_for_performance_claim", True)),
    }


def _route_blocking_reasons(route: str, action_audit: dict[str, Any], stage21_5_summary: dict[str, Any]) -> list[str]:
    if route == ROUTE_BINDING:
        reasons = []
        if int(action_audit.get("strong_state_join_available_count") or 0) <= 0:
            reasons.append("hybrid_path_pre_post_strong_state_join_unavailable")
        if int(action_audit.get("pre_duplicate_strong_key_count") or 0) > 0:
            reasons.append("hybrid_path_pre_duplicate_strong_keys")
        if int(action_audit.get("post_duplicate_strong_key_count") or 0) > 0:
            reasons.append("hybrid_path_post_duplicate_strong_keys")
        if int(action_audit.get("hybrid_path_inference_required_field_missing_count") or 0) > 0:
            reasons.append("hybrid_path_inference_required_fields_missing")
        if int(action_audit.get("coverage_source_mismatch_count") or 0) > 0:
            reasons.append("hybrid_path_inference_coverage_source_mismatch")
        if int(action_audit.get("path_cost_source_mismatch_count") or 0) > 0:
            reasons.append("hybrid_path_inference_path_cost_source_mismatch")
        if int(action_audit.get("max_traversable_slope_mismatch_count") or 0) > 0:
            reasons.append("hybrid_path_inference_max_slope_mismatch")
        if int(action_audit.get("obstacle_contract_mismatch_count") or 0) > 0:
            reasons.append("hybrid_path_obstacle_contract_mismatch")
        if int(action_audit.get("hybrid_path_contract_mismatch_count") or 0) > 0:
            reasons.append("hybrid_path_inference_contract_mismatch")
        if int(action_audit.get("grid_fallback_count") or 0) > 0:
            reasons.append("hybrid_path_inference_grid_fallback_used")
        if int(action_audit.get("default_astar_replaced_count") or 0) > 0:
            reasons.append("hybrid_path_inference_default_astar_replaced")
        if int(action_audit.get("ackermann_feasible_claimed_count") or 0) > 0:
            reasons.append("hybrid_path_inference_ackermann_feasible_claimed")
        return reasons
    if route == ROUTE_SIGNAL:
        return ["hybrid_path_action_probability_and_viewpoint_unchanged"]
    if route == ROUTE_MARGIN:
        return ["hybrid_path_probability_changed_without_discrete_action_change"]
    if route == ROUTE_CREDIT:
        reasons = ["hybrid_path_action_changed_without_coverage_auc_improvement"]
        if int(stage21_5_summary.get("scenario_regression_count") or 0) > 0:
            reasons.append("hybrid_path_post_update_scenario_regression_detected")
        return reasons
    if stage21_5_summary.get("status") == "failed":
        return ["stage21_5_failed"]
    return []


def _input_rejections(config: dict[str, Any], summary: dict[str, Any], stage24_3_summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    root = Path(config["stage24_4_root"])
    if not summary:
        return ["stage24_4_summary_missing"]
    if summary.get("status") != "passed":
        reasons.append("stage24_4_status_not_passed")
    if summary.get("next_required_change") != ROUTE_FROM_STAGE24_4:
        reasons.append("stage24_4_route_not_stage24_5")
    if summary.get("coverage_source") != SLOPE_COVERAGE_SOURCE:
        reasons.append("stage24_4_coverage_source_not_slope_obstacle_los")
    if summary.get("path_cost_source") != HYBRID_ASTAR_PATH_COST_SOURCE:
        reasons.append("stage24_4_path_cost_source_not_hybrid_astar")
    if _finite_float(summary.get("max_traversable_slope_deg")) != 30.0:
        reasons.append("stage24_4_max_traversable_slope_not_30")
    for field in ("checkpoint_exists", "checkpoint_reload_passed", "checkpoint_boundary_passed", "stage21_3_lineage_passed", "experimental_only", "source_checkpoint_sha_consistent"):
        if summary.get(field) is not True:
            reasons.append(f"stage24_4_{field}_not_true")
    zero_fields = (
        "hybrid_path_batch_contract_missing_count",
        "xunce_batch_viewpoint_shape_mismatch_count",
        "action_viewpoint_path_cost_binding_mismatch_count",
        "selected_info_viewpoint_mismatch_count",
        "hybrid_path_provenance_array_length_mismatch_count",
        "selected_action_mask_violation_count",
        "point_grid_path_cost_fallback_used_count",
        "default_astar_replaced_count",
        "ackermann_feasible_claimed_count",
    )
    for field in zero_fields:
        if int(summary.get(field) or 0) != 0:
            reasons.append(f"stage24_4_{field}_nonzero")
    for field in ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary"):
        if summary.get(field) is True:
            reasons.append(f"stage24_4_{field}_enabled")
    if not (root / "s21_4").exists():
        reasons.append("stage24_4_stage21_4_root_missing")
    if not stage24_3_summary:
        reasons.append("stage24_3_summary_missing")
    elif not stage24_3_summary.get("stage23_2b_high_res_root") and not stage24_3_summary.get("stage21_1_actual_source_roi_expansion_root"):
        reasons.append("stage24_3_high_res_source_root_missing")
    return reasons


def _stage24_3_summary(stage24_4_summary: dict[str, Any]) -> dict[str, Any]:
    root_value = stage24_4_summary.get("stage24_3_root")
    if not root_value:
        return {}
    return _read_json_if_exists(Path(str(root_value)) / "xunce-stage24-3-summary.json")


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if bool(config.get(field, False))]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"config schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    config["stage24_4_root"] = str(_resolve_path(Path(str(payload["stage24_4_root"])), repo_root))
    config["stage21_5_base_config"] = str(_resolve_path(Path(str(payload["stage21_5_base_config"])), repo_root))
    config["high_fidelity_config"] = str(_resolve_path(Path(str(payload["high_fidelity_config"])), repo_root))
    config["required_scenario_count"] = _positive_int(payload.get("required_scenario_count", 2), "required_scenario_count")
    config["rollout_steps"] = _positive_int(payload.get("rollout_steps", 4), "rollout_steps")
    config["dynamic_max_candidates_per_step"] = _positive_int(payload.get("dynamic_max_candidates_per_step", 36), "dynamic_max_candidates_per_step")
    config["dynamic_proposal_pool_limit_per_step"] = _positive_int(payload.get("dynamic_proposal_pool_limit_per_step", 288), "dynamic_proposal_pool_limit_per_step")
    config["theta_bin_count"] = _positive_int(payload.get("theta_bin_count", 8), "theta_bin_count")
    config["theta_step_deg"] = _positive_int(payload.get("theta_step_deg", 45), "theta_step_deg")
    config["sensor_model_id"] = str(payload.get("sensor_model_id", "theta-fov-90-range-radius/v1"))
    config["sensor_fov_deg"] = _positive_float(payload.get("sensor_fov_deg", 90.0), "sensor_fov_deg")
    config["sensor_range_cells"] = _positive_int(payload.get("sensor_range_cells", 2), "sensor_range_cells")
    config["hybrid_astar_theta_bin_count"] = _positive_int(
        payload.get("hybrid_astar_theta_bin_count", 72),
        "hybrid_astar_theta_bin_count",
    )
    config["hybrid_astar_goal_position_tolerance_m"] = _positive_float(
        payload.get("hybrid_astar_goal_position_tolerance_m", 2.0),
        "hybrid_astar_goal_position_tolerance_m",
    )
    config["hybrid_astar_goal_theta_tolerance_deg"] = _positive_float(
        payload.get("hybrid_astar_goal_theta_tolerance_deg", 5.0),
        "hybrid_astar_goal_theta_tolerance_deg",
    )
    config["hybrid_astar_max_iterations"] = _positive_int(
        payload.get("hybrid_astar_max_iterations", 100_000),
        "hybrid_astar_max_iterations",
    )
    config["hybrid_astar_primitive_duration_s"] = _positive_float(
        payload.get("hybrid_astar_primitive_duration_s", 1.5),
        "hybrid_astar_primitive_duration_s",
    )
    config["hybrid_astar_integration_dt_s"] = _positive_float(
        payload.get("hybrid_astar_integration_dt_s", 0.25),
        "hybrid_astar_integration_dt_s",
    )
    config["hybrid_astar_max_speed_mps"] = _positive_float(
        payload.get("hybrid_astar_max_speed_mps", 3.0),
        "hybrid_astar_max_speed_mps",
    )
    config["hybrid_astar_max_angular_speed_degps"] = _positive_float(
        payload.get("hybrid_astar_max_angular_speed_degps", 45.0),
        "hybrid_astar_max_angular_speed_degps",
    )
    config["hybrid_astar_rotation_cost_weight"] = _nonnegative_float(
        payload.get("hybrid_astar_rotation_cost_weight", 0.2),
        "hybrid_astar_rotation_cost_weight",
    )
    config["hybrid_astar_reverse_penalty_weight"] = _nonnegative_float(
        payload.get("hybrid_astar_reverse_penalty_weight", 0.5),
        "hybrid_astar_reverse_penalty_weight",
    )
    config["hybrid_astar_turn_penalty_weight"] = _nonnegative_float(
        payload.get("hybrid_astar_turn_penalty_weight", 0.05),
        "hybrid_astar_turn_penalty_weight",
    )
    config["min_mean_abs_probability_delta_for_signal"] = _positive_float(
        payload.get("min_mean_abs_probability_delta_for_signal", 1.0e-5),
        "min_mean_abs_probability_delta_for_signal",
    )
    return config


def _index_inference_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], int]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for row in rows:
        if row.get("policy") is not None and row.get("policy") != "xunce":
            continue
        key = _strong_key(row)
        if key is None:
            continue
        if key in indexed:
            duplicate_count += 1
            continue
        indexed[key] = row
    return indexed, duplicate_count


def _strong_key(row: dict[str, Any]) -> str | None:
    fields = ("scenario_id", "step_index", "current_cell", "covered_cells_hash", "candidate_set_hash")
    if any(row.get(field) is None for field in fields):
        return None
    return "|".join(str(_normalize_json_value(row.get(field))) for field in fields)


def _missing_required_fields(row: dict[str, Any]) -> list[str]:
    missing = [field for field in REQUIRED_INFERENCE_FIELDS if row.get(field) is None]
    if not isinstance(row.get("action_probs"), list) or not row.get("action_probs"):
        missing.append("action_probs_nonempty")
    if not isinstance(row.get("logits"), list):
        missing.append("logits_list")
    return missing


def _probability_deltas(pre: Any, post: Any) -> list[float]:
    if not isinstance(pre, list) or not isinstance(post, list):
        return []
    deltas: list[float] = []
    for index in range(min(len(pre), len(post))):
        pre_value = _finite_float(pre[index])
        post_value = _finite_float(post[index])
        if pre_value is not None and post_value is not None:
            deltas.append(abs(post_value - pre_value))
    return deltas


def _cell(row: dict[str, Any]) -> tuple[Any, Any] | None:
    viewpoint = row.get("selected_viewpoint") or row.get("candidate_viewpoint")
    if isinstance(viewpoint, list) and len(viewpoint) >= 2:
        return viewpoint[0], viewpoint[1]
    return None


def _theta(row: dict[str, Any]) -> float | None:
    value = _finite_float(row.get("selected_theta_deg"))
    if value is not None:
        return value
    viewpoint = row.get("selected_viewpoint") or row.get("candidate_viewpoint")
    if isinstance(viewpoint, list) and len(viewpoint) >= 3:
        return _finite_float(viewpoint[2])
    return None


def _load_json_template(path: str, repo_root: Path) -> dict[str, Any]:
    return _read_json(_resolve_path(Path(path), repo_root))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.exists() else {}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path)


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _positive_float(value: Any, name: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _nonnegative_float(value: Any, name: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative")
    return parsed


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_finite(*values: Any) -> float:
    for value in values:
        parsed = _finite_float(value)
        if parsed is not None:
            return parsed
    return 0.0


def _normalize_json_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage24.5 Hybrid A* Path-Cost Post-Update Trajectory Eval Smoke",
            "",
            f"- status: {summary['status']}",
            f"- next_required_change: {summary['next_required_change']}",
            f"- strong_state_join_available_count: {summary['strong_state_join_available_count']}",
            f"- selected_viewpoint_changed_count: {summary['selected_viewpoint_changed_count']}",
            f"- selected_theta_changed_count: {summary['selected_theta_changed_count']}",
            f"- mean_abs_probability_delta: {summary['mean_abs_probability_delta']}",
            f"- final_coverage_delta: {summary['final_coverage_delta']}",
            f"- coverage_auc_delta: {summary['coverage_auc_delta']}",
            f"- path_cost_delta: {summary['path_cost_delta']}",
            "",
            "This is a bounded offline smoke. It does not publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
            "",
        ]
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

