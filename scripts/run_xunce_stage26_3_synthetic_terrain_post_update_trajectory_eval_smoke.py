from __future__ import annotations

import argparse
import json
import multiprocessing
import queue
from pathlib import Path
from typing import Any

try:  # pragma: no cover - direct script execution
    import run_xunce_stage21_5_post_update_offline_trajectory_evaluation as stage21_5
    import run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke as stage24_5
    import run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as stage26_2
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage21_5_post_update_offline_trajectory_evaluation as stage21_5
    import scripts.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke as stage24_5
    import scripts.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as stage26_2


CONFIG_SCHEMA_VERSION = "xunce-stage26-3-synthetic-terrain-post-update-trajectory-eval-smoke-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-3-summary/v1"
ACTION_AUDIT_SCHEMA_VERSION = "xunce-stage26-3-synthetic-action-change-audit/v1"
DELTA_AUDIT_SCHEMA_VERSION = "xunce-stage26-3-trajectory-delta-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-3-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-3-manifest/v1"

STAGE_ID = "xunce-stage26-3-synthetic-terrain-post-update-trajectory-eval-smoke"
DEFAULT_CONFIG = "configs/xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke_v1"
)

SYNTHETIC_MODEL_ID = "synthetic_rock_pit_terrain/v1"
SYNTHETIC_SOURCE_KIND = "synthetic_terrain_obstacle_proxy/v1"
COVERAGE_SOURCE = stage26_2.COVERAGE_SOURCE
PATH_COST_SOURCE = stage26_2.PATH_COST_SOURCE

ROUTE_INPUTS = "rerun_stage26_3_required_inputs"
ROUTE_BINDING = "repair_stage26_3_synthetic_inference_binding"
ROUTE_LINEAGE = "repair_stage26_3_synthetic_lineage_binding"
ROUTE_SEMANTICS = "repair_stage26_3_synthetic_source_semantics"
ROUTE_SAFETY = "repair_stage26_3_synthetic_eval_safety_regression"
ROUTE_SIGNAL = "repair_stage26_synthetic_policy_update_signal_strength"
ROUTE_MARGIN = "calibrate_stage26_synthetic_discrete_margin_crossing"
ROUTE_CREDIT = "repair_stage26_synthetic_credit_assignment"
ROUTE_STAGE26_4 = "run_stage26_4_synthetic_terrain_multi_seed_ppo_pilot"
ROUTE_BOUNDARY = "resolve_stage26_3_boundary_rejections"
ROUTE_FROM_STAGE26_2 = "run_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke"

SUMMARY_FILE = "xunce-stage26-3-summary.json"
STAGE21_5_CONFIG_FILE = "xunce-stage26-3-stage21-5-config.json"
STAGE21_5_SUMMARY_FILE = "xunce-stage26-3-stage21-5-summary.json"
ACTION_AUDIT_FILE = "xunce-stage26-3-synthetic-action-change-audit.json"
DELTA_AUDIT_FILE = "xunce-stage26-3-trajectory-delta-audit.json"
ROUTING_FILE = "xunce-stage26-3-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-3-report.md"
MANIFEST_FILE = "xunce-stage26-3-manifest.json"
GENERATED_HIGH_FIDELITY_CONFIG_FILE = "xunce-stage26-3-high-fidelity-config.json"
MODEL_INFERENCE_FILE = stage24_5.MODEL_INFERENCE_FILE

BOUNDARY_FIELDS = (
    "stage26_3_authorized",
    "release_or_training_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)
HYBRID_ASTAR_PLANNER_FIELDS = stage24_5.HYBRID_ASTAR_PLANNER_FIELDS

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
    "synthetic_terrain_hash",
    "synthetic_source_kind",
    "synthetic_los_blocker_cells_used",
    "synthetic_hard_obstacle_cells_used",
    "physical_obstacle_cells_written",
    "default_astar_replaced",
    "hybrid_astar_ackermann_feasible_claimed",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.3 synthetic terrain post-update trajectory eval smoke.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    summary = run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(config_path, repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage26_2_summary = _read_json_if_exists(Path(config["stage26_2_root"]) / stage26_2.SUMMARY_FILE)
    stage26_1_summary = _stage26_1_summary(stage26_2_summary)
    boundary_reasons = _boundary_rejections(config)
    input_reasons = _input_rejections(config, stage26_2_summary, stage26_1_summary)

    stage21_5_summary: dict[str, Any] = {}
    stage21_5_config_path = output_root / STAGE21_5_CONFIG_FILE
    high_fidelity_config_path = output_root / GENERATED_HIGH_FIDELITY_CONFIG_FILE
    if not boundary_reasons and not input_reasons:
        stage21_5_summary = _run_stage21_5(
            config,
            stage26_2_summary,
            stage26_1_summary,
            output_root,
            repo_root,
            stage21_5_config_path,
            high_fidelity_config_path,
        )

    action_audit = _synthetic_action_change_audit(stage21_5_summary, output_root / "s21_5", stage26_2_summary)
    delta_audit = _trajectory_delta_audit(stage21_5_summary, output_root / "s21_5")
    stage21_5_execution_reasons = stage24_5._stage21_5_execution_rejections(stage21_5_summary)
    if stage21_5_summary.get("stage26_3_runtime_blocker") is True:
        stage21_5_execution_reasons = _unique(stage21_5_execution_reasons + ["stage21_5_runtime_timeout_or_process_failure"])
    safety_reasons = stage24_5._safety_rejections(stage21_5_summary)
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
        "stage26_2_root": config["stage26_2_root"],
        "stage26_2_status": stage26_2_summary.get("status"),
        "stage26_2_next_required_change": stage26_2_summary.get("next_required_change"),
        "stage26_1_root": stage26_2_summary.get("stage26_1_root"),
        "stage26_1_status": stage26_1_summary.get("status"),
        "stage21_5_root": str(output_root / "s21_5"),
        "stage21_5_status": stage21_5_summary.get("status"),
        "stage21_5_next_required_change": stage21_5_summary.get("next_required_change"),
        "stage21_5_execution_incomplete": bool(stage21_5_execution_reasons),
        "stage21_5_execution_reason_codes": stage21_5_execution_reasons,
        "stage21_5_runtime_blocker": bool(stage21_5_summary.get("stage26_3_runtime_blocker", False)),
        "stage21_5_runtime_timeout_seconds": stage21_5_summary.get("runtime_timeout_seconds"),
        "action_probability_audit_is_partial_diagnostic": bool(stage21_5_execution_reasons),
        "coverage_source": COVERAGE_SOURCE,
        "path_cost_source": PATH_COST_SOURCE,
        "synthetic_terrain_model_id": SYNTHETIC_MODEL_ID,
        "synthetic_terrain_hash": stage26_2_summary.get("synthetic_terrain_hash") or config.get("synthetic_terrain_hash"),
        "synthetic_source_kind": SYNTHETIC_SOURCE_KIND,
        "max_traversable_slope_deg": 30.0,
        "default_astar_replaced": False,
        "ackermann_feasible_claimed": False,
        "stage26_2_pre_clip_grad_norm": stage26_2_summary.get("pre_clip_grad_norm"),
        **_summary_counts(action_audit, delta_audit, stage21_5_summary),
        "release_or_training_authorized": False,
        "stage26_3_authorized": False,
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
        "stage26_3_authorized": False,
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
        "synthetic_action_change_audit": str(output_root / ACTION_AUDIT_FILE),
        "trajectory_delta_audit": str(output_root / DELTA_AUDIT_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / STAGE21_5_SUMMARY_FILE, stage21_5_summary)
    _write_json(output_root / ACTION_AUDIT_FILE, action_audit)
    _write_json(output_root / DELTA_AUDIT_FILE, delta_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _run_stage21_5(
    config: dict[str, Any],
    stage26_2_summary: dict[str, Any],
    stage26_1_summary: dict[str, Any],
    output_root: Path,
    repo_root: Path,
    stage21_5_config_path: Path,
    high_fidelity_config_path: Path,
) -> dict[str, Any]:
    cfg = _load_json_template(config["stage21_5_base_config"], repo_root)
    high_fidelity_cfg = _load_json_template(config["high_fidelity_config"], repo_root)
    source_roi_root = _source_roi_expansion_root(stage26_1_summary, stage26_2_summary)
    synthetic_hash = str(stage26_2_summary.get("synthetic_terrain_hash") or config.get("synthetic_terrain_hash") or "")
    common = {
        "theta_aware_candidate_viewpoints_enabled": True,
        "slope_obstacle_aware_theta_reward_enabled": True,
        "hybrid_astar_pose_path_cost_enabled": True,
        "synthetic_terrain_contract_enabled": True,
        "obstacle_occlusion_enabled": True,
        "coverage_source": COVERAGE_SOURCE,
        "path_cost_source": PATH_COST_SOURCE,
        "synthetic_terrain_model_id": SYNTHETIC_MODEL_ID,
        "synthetic_terrain_hash": synthetic_hash,
        "synthetic_source_kind": SYNTHETIC_SOURCE_KIND,
        "synthetic_los_blocker_cells_used": True,
        "synthetic_hard_obstacle_cells_used": True,
        "physical_obstacle_cells_written": False,
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
        "default_astar_replaced": False,
        "hybrid_astar_ackermann_feasible_claimed": False,
        "hybrid_astar_candidate_eval_workers": int(config.get("hybrid_astar_candidate_eval_workers", 1)),
    }
    if bool(config.get("continuous_theta_action_space_enabled")):
        common["continuous_theta_action_space_enabled"] = True
        common["action_space_type"] = str(config.get("action_space_type") or "hybrid_discrete_xy_continuous_theta/v1")
    if bool(config.get("synthetic_credit_feature_exposure_enabled")):
        common["synthetic_credit_feature_exposure_enabled"] = True
    high_fidelity_cfg.update(
        {
            **common,
            "source_roi_expansion_root": source_roi_root,
            "dynamic_validation_work_root": str(output_root / "_xunce_dynamic_validation_work"),
            "emit_candidate_metric_audit": True,
            "include_oracle_baselines": False,
            "include_canonical_reward_rerank_oracle": False,
            "xunce_only_evaluation": True,
        }
    )
    _write_json(high_fidelity_config_path, high_fidelity_cfg)

    cfg.update(
        {
            **common,
            "stage21_4_tiny_ppo_update_smoke_root": str(Path(config["stage26_2_root"]) / "s21_4"),
            "high_fidelity_config": str(high_fidelity_config_path),
            "execute_high_fidelity_evaluations": True,
            "pre_ppo_evaluation_root": str(output_root / "pre"),
            "post_ppo_evaluation_root": str(output_root / "post"),
            "include_oracle_baselines": False,
            "include_canonical_reward_rerank_oracle": False,
            "xunce_only_evaluation": True,
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
    return _invoke_stage21_5(
        config_path=stage21_5_config_path,
        output_root=output_root / "s21_5",
        repo_root=repo_root,
        timeout_seconds=float(config.get("stage21_5_timeout_seconds") or 0.0),
    )


def _invoke_stage21_5(*, config_path: Path, output_root: Path, repo_root: Path, timeout_seconds: float) -> dict[str, Any]:
    if timeout_seconds <= 0.0:
        return stage21_5.run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
            config_path=config_path,
            output_root=output_root,
            repo_root=repo_root,
        )
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_stage21_5_process_entry, args=(str(config_path), str(output_root), str(repo_root), result_queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(10.0)
        return {
            "schema_version": "xunce-stage21-5-post-update-evaluation-summary/v1",
            "status": "failed",
            "next_required_change": stage21_5.ROUTE_EXECUTION,
            "reason_codes": ["stage21_5_runtime_timeout_seconds_exceeded"],
            "stage26_3_runtime_blocker": True,
            "runtime_timeout_seconds": timeout_seconds,
            "pre_evaluation_root": str(output_root / "pre_ppo_xunce"),
            "post_evaluation_root": str(output_root / "post_ppo_xunce"),
            "post_hard_risk_violation_count": 0,
            "post_mask_violation_count": 0,
            "post_path_planning_failure_count": 0,
            "post_open_grid_fallback_count": 0,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    try:
        payload = result_queue.get_nowait()
    except queue.Empty:
        payload = {
            "schema_version": "xunce-stage21-5-post-update-evaluation-summary/v1",
            "status": "failed",
            "next_required_change": stage21_5.ROUTE_EXECUTION,
            "reason_codes": ["stage21_5_runtime_process_returned_without_summary"],
            "stage26_3_runtime_blocker": True,
        }
    if isinstance(payload, dict):
        return payload
    return {
        "schema_version": "xunce-stage21-5-post-update-evaluation-summary/v1",
        "status": "failed",
        "next_required_change": stage21_5.ROUTE_EXECUTION,
        "reason_codes": ["stage21_5_runtime_process_invalid_summary"],
        "stage26_3_runtime_blocker": True,
    }


def _stage21_5_process_entry(config_path: str, output_root: str, repo_root: str, result_queue: Any) -> None:
    try:
        summary = stage21_5.run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
            config_path=Path(config_path),
            output_root=Path(output_root),
            repo_root=Path(repo_root),
        )
    except BaseException as exc:  # pragma: no cover - subprocess crash path
        summary = {
            "schema_version": "xunce-stage21-5-post-update-evaluation-summary/v1",
            "status": "failed",
            "next_required_change": stage21_5.ROUTE_EXECUTION,
            "reason_codes": [f"stage21_5_runtime_exception:{type(exc).__name__}"],
            "runtime_exception_message": str(exc),
            "stage26_3_runtime_blocker": True,
        }
    result_queue.put(summary)


def _synthetic_action_change_audit(
    stage21_5_summary: dict[str, Any],
    stage21_5_root: Path,
    stage26_2_summary: dict[str, Any],
) -> dict[str, Any]:
    pre_root = Path(str(stage21_5_summary.get("pre_evaluation_root") or stage21_5_root / "pre_ppo_xunce"))
    post_root = Path(str(stage21_5_summary.get("post_evaluation_root") or stage21_5_root / "post_ppo_xunce"))
    pre_rows = _read_jsonl_if_exists(pre_root / MODEL_INFERENCE_FILE)
    post_rows = _read_jsonl_if_exists(post_root / MODEL_INFERENCE_FILE)
    expected_hash = str(stage26_2_summary.get("synthetic_terrain_hash") or "")
    pre_missing = sum(len(_missing_required_fields(row)) for row in _xunce_rows(pre_rows))
    post_missing = sum(len(_missing_required_fields(row)) for row in _xunce_rows(post_rows))
    pre_index, pre_duplicate_count, pre_key_missing = _index_inference_rows(pre_rows)
    post_index, post_duplicate_count, post_key_missing = _index_inference_rows(post_rows)
    post_base_index = _index_inference_rows_by_base_key(post_rows)
    joined = missing_post = 0
    coverage_source_mismatch = path_cost_source_mismatch = hybrid_contract_mismatch = 0
    synthetic_contract_mismatch = physical_payload_count = grid_fallback_count = 0
    default_astar_replaced_count = ackermann_claimed_count = 0
    viewpoint_changed = theta_changed = cell_changed = action_changed = 0
    probability_abs_delta_sum = probability_value_count = 0
    selected_probability_delta_sum = selected_probability_delta_count = 0
    blocking_examples: list[dict[str, Any]] = []
    for key, pre in pre_index.items():
        post = post_index.get(key)
        if post is None:
            missing_post += 1
            base_post = post_base_index.get(_base_strong_key(pre) or "")
            if base_post is not None:
                post_hash = base_post.get("synthetic_terrain_hash")
                post_kind = base_post.get("synthetic_source_kind")
                if (post_hash is not None and pre.get("synthetic_terrain_hash") != post_hash) or (
                    post_kind is not None and pre.get("synthetic_source_kind") != post_kind
                ):
                    synthetic_contract_mismatch += 1
            continue
        joined += 1
        for row in (pre, post):
            if row.get("coverage_source") != COVERAGE_SOURCE:
                coverage_source_mismatch += 1
            if row.get("path_cost_source") != PATH_COST_SOURCE:
                path_cost_source_mismatch += 1
            if row.get("hybrid_astar_trajectory_kind") != "hybrid_astar_pose_path":
                hybrid_contract_mismatch += 1
            if _finite_float(row.get("hybrid_astar_path_cost")) is None:
                hybrid_contract_mismatch += 1
            if not str(row.get("hybrid_astar_pose_path_hash") or "").strip():
                hybrid_contract_mismatch += 1
            if row.get("point_grid_path_cost_fallback_used") is True:
                grid_fallback_count += 1
            if row.get("default_astar_replaced") is True:
                default_astar_replaced_count += 1
            if row.get("hybrid_astar_ackermann_feasible_claimed") is True:
                ackermann_claimed_count += 1
            if _synthetic_contract_mismatch(row, expected_hash=expected_hash):
                synthetic_contract_mismatch += 1
            if row.get("physical_obstacle_cells_written") is True or _physical_obstacle_payload_present(row):
                physical_payload_count += 1
        if len(blocking_examples) < 5:
            row_missing = _missing_required_fields(pre) + _missing_required_fields(post)
            if row_missing:
                blocking_examples.append({"key": key, "missing": row_missing})
        if _normalize_json_value(pre.get("selected_viewpoint")) != _normalize_json_value(post.get("selected_viewpoint")):
            viewpoint_changed += 1
        if _cell(pre) != _cell(post):
            cell_changed += 1
        if _theta(pre) != _theta(post):
            theta_changed += 1
        if pre.get("selected_action_index") != post.get("selected_action_index"):
            action_changed += 1
        deltas = stage24_5._probability_deltas(pre.get("action_probs"), post.get("action_probs"))
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
        "pre_strong_key_unavailable_count": pre_key_missing,
        "post_strong_key_unavailable_count": post_key_missing,
        "synthetic_inference_required_field_missing_count": pre_missing + post_missing,
        "coverage_source_mismatch_count": coverage_source_mismatch,
        "path_cost_source_mismatch_count": path_cost_source_mismatch,
        "hybrid_path_contract_mismatch_count": hybrid_contract_mismatch,
        "synthetic_contract_mismatch_count": synthetic_contract_mismatch,
        "physical_obstacle_payload_count": physical_payload_count,
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
    audit = stage24_5._trajectory_delta_audit(stage21_5_summary, stage21_5_root)
    audit["schema_version"] = DELTA_AUDIT_SCHEMA_VERSION
    audit["hybrid_astar_path_cost_delta"] = audit.get("path_cost_delta")
    return audit


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
        return "failed", ROUTE_INPUTS, "Stage26.2 input is not ready for Stage26.3"
    if stage21_5_execution_reasons:
        return "failed", ROUTE_INPUTS, "Stage21.5 evaluation did not satisfy Stage26.3 smoke input requirements"
    if safety_reasons:
        return "failed", ROUTE_SAFETY, "post-update trajectory safety regression detected"
    if stage21_5_summary.get("status") not in {"passed", "failed"}:
        return "failed", ROUTE_INPUTS, "Stage21.5 did not produce a readable summary"
    if int(action_audit.get("physical_obstacle_payload_count") or 0) > 0:
        return "failed", ROUTE_SEMANTICS, "synthetic terrain polluted physical obstacle fields"
    if int(action_audit.get("synthetic_contract_mismatch_count") or 0) > 0:
        return "failed", ROUTE_LINEAGE, "synthetic terrain hash/source kind is inconsistent between pre/post rows"
    if _binding_failed(action_audit):
        return "failed", ROUTE_BINDING, "synthetic pre/post inference binding is incomplete"
    changed = (
        int(action_audit.get("selected_viewpoint_changed_count") or 0) > 0
        or int(action_audit.get("selected_theta_changed_count") or 0) > 0
    )
    prob_delta = float(action_audit.get("mean_abs_probability_delta") or 0.0)
    final_delta = float(delta_audit.get("final_coverage_delta") or 0.0)
    auc_delta = float(delta_audit.get("coverage_auc_delta") or 0.0)
    path_delta = float(delta_audit.get("hybrid_astar_path_cost_delta") or delta_audit.get("path_cost_delta") or 0.0)
    if not changed and prob_delta < min_probability_delta:
        return "failed", ROUTE_SIGNAL, "synthetic terrain PPO update did not move probabilities or selected viewpoint"
    if not changed:
        return "failed", ROUTE_MARGIN, "synthetic terrain probabilities moved but did not cross the discrete viewpoint boundary"
    if int(stage21_5_summary.get("scenario_regression_count") or 0) > 0:
        return "failed", ROUTE_CREDIT, "synthetic terrain smoke has at least one scenario-level regression"
    if final_delta > 0.0 and auc_delta > 0.0 and path_delta <= 0.0:
        return "passed", ROUTE_STAGE26_4, "synthetic terrain smoke improved coverage/AUC/path cost without safety regression"
    return "failed", ROUTE_CREDIT, "synthetic terrain action changed but coverage/AUC/path cost did not improve in the smoke"


def _binding_failed(action_audit: dict[str, Any]) -> bool:
    return (
        int(action_audit.get("strong_state_join_available_count") or 0) <= 0
        or int(action_audit.get("synthetic_inference_required_field_missing_count") or 0) > 0
        or int(action_audit.get("coverage_source_mismatch_count") or 0) > 0
        or int(action_audit.get("path_cost_source_mismatch_count") or 0) > 0
        or int(action_audit.get("hybrid_path_contract_mismatch_count") or 0) > 0
        or int(action_audit.get("grid_fallback_count") or 0) > 0
        or int(action_audit.get("default_astar_replaced_count") or 0) > 0
        or int(action_audit.get("ackermann_feasible_claimed_count") or 0) > 0
        or int(action_audit.get("pre_duplicate_strong_key_count") or 0) > 0
        or int(action_audit.get("post_duplicate_strong_key_count") or 0) > 0
    )


def _summary_counts(action_audit: dict[str, Any], delta_audit: dict[str, Any], stage21_5_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "strong_state_join_available_count": int(action_audit.get("strong_state_join_available_count") or 0),
        "pre_duplicate_strong_key_count": int(action_audit.get("pre_duplicate_strong_key_count") or 0),
        "post_duplicate_strong_key_count": int(action_audit.get("post_duplicate_strong_key_count") or 0),
        "pre_strong_key_unavailable_count": int(action_audit.get("pre_strong_key_unavailable_count") or 0),
        "post_strong_key_unavailable_count": int(action_audit.get("post_strong_key_unavailable_count") or 0),
        "synthetic_inference_required_field_missing_count": int(action_audit.get("synthetic_inference_required_field_missing_count") or 0),
        "coverage_source_mismatch_count": int(action_audit.get("coverage_source_mismatch_count") or 0),
        "path_cost_source_mismatch_count": int(action_audit.get("path_cost_source_mismatch_count") or 0),
        "hybrid_path_contract_mismatch_count": int(action_audit.get("hybrid_path_contract_mismatch_count") or 0),
        "synthetic_contract_mismatch_count": int(action_audit.get("synthetic_contract_mismatch_count") or 0),
        "physical_obstacle_payload_count": int(action_audit.get("physical_obstacle_payload_count") or 0),
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
        "hybrid_astar_path_cost_delta": _first_finite(delta_audit.get("hybrid_astar_path_cost_delta"), 0.0),
        "path_cost_delta": _first_finite(delta_audit.get("path_cost_delta"), 0.0),
        "coverage_per_100m_delta": _first_finite(delta_audit.get("coverage_per_100m_delta"), 0.0),
        "synthetic_contract_mismatch_count": int(action_audit.get("synthetic_contract_mismatch_count") or 0),
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
            reasons.append("synthetic_pre_post_strong_state_join_unavailable")
        if int(action_audit.get("pre_duplicate_strong_key_count") or 0) > 0:
            reasons.append("synthetic_pre_duplicate_strong_keys")
        if int(action_audit.get("post_duplicate_strong_key_count") or 0) > 0:
            reasons.append("synthetic_post_duplicate_strong_keys")
        if int(action_audit.get("synthetic_inference_required_field_missing_count") or 0) > 0:
            reasons.append("synthetic_inference_required_fields_missing")
        if int(action_audit.get("coverage_source_mismatch_count") or 0) > 0:
            reasons.append("synthetic_inference_coverage_source_mismatch")
        if int(action_audit.get("path_cost_source_mismatch_count") or 0) > 0:
            reasons.append("synthetic_inference_path_cost_source_mismatch")
        if int(action_audit.get("hybrid_path_contract_mismatch_count") or 0) > 0:
            reasons.append("synthetic_hybrid_path_contract_mismatch")
        if int(action_audit.get("grid_fallback_count") or 0) > 0:
            reasons.append("synthetic_grid_fallback_used")
        return reasons
    if route == ROUTE_LINEAGE:
        return ["synthetic_hash_or_source_kind_mismatch"]
    if route == ROUTE_SEMANTICS:
        return ["synthetic_physical_obstacle_payload_or_pollution"]
    if route == ROUTE_SIGNAL:
        return ["synthetic_action_probability_and_viewpoint_unchanged"]
    if route == ROUTE_MARGIN:
        return ["synthetic_probability_changed_without_discrete_action_change"]
    if route == ROUTE_CREDIT:
        reasons = ["synthetic_action_changed_without_coverage_auc_path_cost_improvement"]
        if int(stage21_5_summary.get("scenario_regression_count") or 0) > 0:
            reasons.append("synthetic_post_update_scenario_regression_detected")
        return reasons
    if stage21_5_summary.get("status") == "failed":
        return ["stage21_5_failed"]
    return []


def _input_rejections(config: dict[str, Any], summary: dict[str, Any], stage26_1_summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    root = Path(config["stage26_2_root"])
    if not summary:
        return ["stage26_2_summary_missing"]
    if summary.get("status") != "passed":
        reasons.append("stage26_2_status_not_passed")
    if summary.get("next_required_change") != ROUTE_FROM_STAGE26_2:
        reasons.append("stage26_2_route_not_stage26_3")
    if summary.get("coverage_source") != COVERAGE_SOURCE:
        reasons.append("stage26_2_coverage_source_mismatch")
    if summary.get("path_cost_source") != PATH_COST_SOURCE:
        reasons.append("stage26_2_path_cost_source_mismatch")
    if summary.get("synthetic_source_kind") != SYNTHETIC_SOURCE_KIND:
        reasons.append("stage26_2_synthetic_source_kind_mismatch")
    if not str(summary.get("synthetic_terrain_hash") or "").strip():
        reasons.append("stage26_2_synthetic_hash_missing")
    if _finite_float(summary.get("max_traversable_slope_deg")) != 30.0:
        reasons.append("stage26_2_max_traversable_slope_not_30")
    for field in (
        "checkpoint_exists",
        "checkpoint_reload_passed",
        "checkpoint_boundary_passed",
        "stage21_3_lineage_passed",
        "experimental_only",
        "source_checkpoint_sha_consistent",
        "collector_source_checkpoint_match",
    ):
        if summary.get(field) is not True:
            reasons.append(f"stage26_2_{field}_not_true")
    for field in (
        "synthetic_physical_obstacle_payload_count",
        "stage26_1_upstream_physical_obstacle_payload_count",
        "synthetic_physical_obstacle_pollution_count",
        "point_grid_path_cost_fallback_used_count",
        "default_astar_replaced_count",
        "ackermann_feasible_claimed_count",
    ):
        if int(summary.get(field) or 0) != 0:
            reasons.append(f"stage26_2_{field}_nonzero")
    for field in ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary"):
        if summary.get(field) is True:
            reasons.append(f"stage26_2_{field}_enabled")
    if not (root / "s21_4").exists():
        reasons.append("stage26_2_stage21_4_root_missing")
    if not stage26_1_summary:
        reasons.append("stage26_1_summary_missing")
    elif not _source_roi_expansion_root(stage26_1_summary, summary):
        reasons.append("stage26_1_synthetic_source_root_missing")
    return _unique(reasons)


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
    config["stage26_2_root"] = str(_resolve_path(Path(str(payload["stage26_2_root"])), repo_root))
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
    for field in HYBRID_ASTAR_PLANNER_FIELDS:
        if field in {"hybrid_astar_theta_bin_count", "hybrid_astar_max_iterations"}:
            config[field] = _positive_int(payload.get(field, _default_hybrid_value(field)), field)
        else:
            config[field] = _positive_float(payload.get(field, _default_hybrid_value(field)), field)
    config["min_mean_abs_probability_delta_for_signal"] = _positive_float(
        payload.get("min_mean_abs_probability_delta_for_signal", 1.0e-5),
        "min_mean_abs_probability_delta_for_signal",
    )
    config["stage21_5_timeout_seconds"] = float(payload.get("stage21_5_timeout_seconds", 0.0) or 0.0)
    if config["stage21_5_timeout_seconds"] < 0.0:
        raise ValueError("stage21_5_timeout_seconds must be nonnegative")
    return config


def _default_hybrid_value(field: str) -> float | int:
    defaults: dict[str, float | int] = {
        "hybrid_astar_theta_bin_count": 72,
        "hybrid_astar_goal_position_tolerance_m": 2.0,
        "hybrid_astar_goal_theta_tolerance_deg": 5.0,
        "hybrid_astar_max_iterations": 100_000,
        "hybrid_astar_primitive_duration_s": 1.5,
        "hybrid_astar_integration_dt_s": 0.25,
        "hybrid_astar_max_speed_mps": 3.0,
        "hybrid_astar_max_angular_speed_degps": 45.0,
        "hybrid_astar_rotation_cost_weight": 0.2,
        "hybrid_astar_reverse_penalty_weight": 0.5,
        "hybrid_astar_turn_penalty_weight": 0.05,
    }
    return defaults[field]


def _stage26_1_summary(stage26_2_summary: dict[str, Any]) -> dict[str, Any]:
    root_value = stage26_2_summary.get("stage26_1_root")
    if not root_value:
        return {}
    return _read_json_if_exists(Path(str(root_value)) / "xunce-stage26-1-summary.json")


def _source_roi_expansion_root(stage26_1_summary: dict[str, Any], stage26_2_summary: dict[str, Any]) -> str:
    for value in (
        stage26_1_summary.get("synthetic_source_root"),
        stage26_1_summary.get("stage21_1_actual_source_roi_expansion_root"),
        stage26_2_summary.get("stage26_1_source_roi_expansion_root"),
    ):
        if value:
            return str(value)
    root = stage26_2_summary.get("stage26_1_root")
    if not root:
        return ""
    manifest = _read_json_if_exists(Path(str(root)) / "s21_1" / "xunce-stage21-1-manifest.json")
    model_audit = manifest.get("model_audit") if isinstance(manifest.get("model_audit"), dict) else {}
    return str(model_audit.get("source_roi_expansion_root") or "")


def _index_inference_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], int, int]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicate_count = key_missing_count = 0
    for row in _xunce_rows(rows):
        key = _strong_key(row)
        if key is None:
            key_missing_count += 1
            continue
        if key in indexed:
            duplicate_count += 1
            continue
        indexed[key] = row
    return indexed, duplicate_count, key_missing_count


def _index_inference_rows_by_base_key(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in _xunce_rows(rows):
        key = _base_strong_key(row)
        if key is not None and key not in indexed:
            indexed[key] = row
    return indexed


def _strong_key(row: dict[str, Any]) -> str | None:
    fields = ("scenario_id", "step_index", "current_cell", "covered_cells_hash", "candidate_set_hash", "synthetic_terrain_hash")
    if any(row.get(field) is None for field in fields):
        return None
    return "|".join(str(_normalize_json_value(row.get(field))) for field in fields)


def _base_strong_key(row: dict[str, Any]) -> str | None:
    fields = ("scenario_id", "step_index", "current_cell", "covered_cells_hash", "candidate_set_hash")
    if any(row.get(field) is None for field in fields):
        return None
    return "|".join(str(_normalize_json_value(row.get(field))) for field in fields)


def _xunce_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("policy") is None or row.get("policy") == "xunce"]


def _missing_required_fields(row: dict[str, Any]) -> list[str]:
    missing = [field for field in REQUIRED_INFERENCE_FIELDS if row.get(field) is None]
    if not isinstance(row.get("action_probs"), list) or not row.get("action_probs"):
        missing.append("action_probs_nonempty")
    if not isinstance(row.get("logits"), list):
        missing.append("logits_list")
    return missing


def _synthetic_contract_mismatch(row: dict[str, Any], *, expected_hash: str) -> bool:
    if expected_hash and row.get("synthetic_terrain_hash") != expected_hash:
        return True
    return (
        row.get("synthetic_source_kind") != SYNTHETIC_SOURCE_KIND
        or row.get("synthetic_los_blocker_cells_used") is not True
        or row.get("synthetic_hard_obstacle_cells_used") is not True
        or row.get("physical_obstacle_cells_written") is not False
    )


def _physical_obstacle_payload_present(row: dict[str, Any]) -> bool:
    return stage26_2._physical_obstacle_payload_present(row)


def _cell(row: dict[str, Any]) -> tuple[Any, Any] | None:
    return stage24_5._cell(row)


def _theta(row: dict[str, Any]) -> float | None:
    return stage24_5._theta(row)


def _load_json_template(path: str, repo_root: Path) -> dict[str, Any]:
    return _read_json(_resolve_path(Path(path), repo_root))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.exists() else {}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return stage24_5._read_jsonl_if_exists(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    stage24_5._write_json(path, payload)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path)


def _positive_int(value: Any, name: str) -> int:
    return stage24_5._positive_int(value, name)


def _positive_float(value: Any, name: str) -> float:
    return stage24_5._positive_float(value, name)


def _finite_float(value: Any) -> float | None:
    return stage24_5._finite_float(value)


def _int_or_none(value: Any) -> int | None:
    return stage24_5._int_or_none(value)


def _first_finite(*values: Any) -> float:
    return stage24_5._first_finite(*values)


def _normalize_json_value(value: Any) -> str:
    return stage24_5._normalize_json_value(value)


def _unique(values: list[str]) -> list[str]:
    return stage24_5._unique(values)


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.3 Synthetic Terrain Post-Update Trajectory Eval Smoke",
            "",
            f"- status: {summary['status']}",
            f"- next_required_change: {summary['next_required_change']}",
            f"- synthetic_terrain_hash: {summary['synthetic_terrain_hash']}",
            f"- strong_state_join_available_count: {summary['strong_state_join_available_count']}",
            f"- selected_viewpoint_changed_count: {summary['selected_viewpoint_changed_count']}",
            f"- selected_theta_changed_count: {summary['selected_theta_changed_count']}",
            f"- mean_abs_probability_delta: {summary['mean_abs_probability_delta']}",
            f"- final_coverage_delta: {summary['final_coverage_delta']}",
            f"- coverage_auc_delta: {summary['coverage_auc_delta']}",
            f"- hybrid_astar_path_cost_delta: {summary['hybrid_astar_path_cost_delta']}",
            "",
            "This is a bounded offline smoke. It does not publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
            "",
        ]
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
