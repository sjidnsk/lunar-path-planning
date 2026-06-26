from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

try:  # pragma: no cover
    import run_xunce_stage26_1_synthetic_terrain_collector_smoke as stage26_1
    import run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as stage26_2
    import run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as stage26_3
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage26_1_synthetic_terrain_collector_smoke as stage26_1
    import scripts.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as stage26_2
    import scripts.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as stage26_3


STAGE_ID = "xunce-stage26-4-synthetic-policy-update-signal-strength-repair"
CONFIG_SCHEMA_VERSION = "xunce-stage26-4-synthetic-policy-update-signal-strength-repair-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-4-summary/v1"
GRADIENT_AUDIT_SCHEMA_VERSION = "xunce-stage26-4-policy-value-gradient-audit/v1"
ACTION_AUDIT_SCHEMA_VERSION = "xunce-stage26-4-action-signal-audit/v1"
CREDIT_AUDIT_SCHEMA_VERSION = "xunce-stage26-4-synthetic-credit-signal-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-4-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-4-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_4_synthetic_policy_update_signal_strength_repair_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_4_synthetic_policy_update_signal_strength_repair_v1"
)
DEFAULT_STAGE26_3_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke_v1"
)

SUMMARY_FILE = "xunce-stage26-4-summary.json"
SWEEP_FILE = "xunce-stage26-4-sweep-results.jsonl"
GRADIENT_AUDIT_FILE = "xunce-stage26-4-policy-value-gradient-audit.json"
ACTION_AUDIT_FILE = "xunce-stage26-4-action-signal-audit.json"
CREDIT_AUDIT_FILE = "xunce-stage26-4-synthetic-credit-signal-audit.json"
RECOMMENDED_26_2_CONFIG = "xunce-stage26-4-recommended-stage26-2-config.json"
RECOMMENDED_26_3_CONFIG = "xunce-stage26-4-recommended-stage26-3-config.json"
ROUTING_FILE = "xunce-stage26-4-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-4-report.md"
MANIFEST_FILE = "xunce-stage26-4-manifest.json"
COLLECTOR_CONFIG_FILE = "xunce-stage26-4-stage26-1-config.json"

SYNTHETIC_MODEL_ID = stage26_2.SYNTHETIC_MODEL_ID
SYNTHETIC_SOURCE_KIND = stage26_2.SYNTHETIC_SOURCE_KIND
COVERAGE_SOURCE = stage26_2.COVERAGE_SOURCE
PATH_COST_SOURCE = stage26_2.PATH_COST_SOURCE
SYNTHETIC_HASH = "e628d24c6fb0f961c50294ea0a018a2558a79a0b30ad21dd31b6e34fc0867e85"

ROUTE_INPUTS = "rerun_stage26_4_required_inputs"
ROUTE_COLLECTOR = "expand_stage26_synthetic_collector_samples"
ROUTE_LINEAGE = "repair_stage26_4_synthetic_lineage_binding"
ROUTE_STABILITY = "repair_stage26_synthetic_ppo_update_stability"
ROUTE_VALUE_BALANCE = "repair_stage26_policy_value_loss_balance"
ROUTE_STRENGTH = "increase_stage26_synthetic_update_strength_or_sample_count"
ROUTE_MARGIN = "calibrate_stage26_synthetic_discrete_margin_crossing"
ROUTE_CREDIT = "repair_stage26_synthetic_credit_assignment"
ROUTE_STAGE26_5 = "run_stage26_5_synthetic_terrain_multi_seed_ppo_pilot"
ROUTE_BOUNDARY = "resolve_stage26_4_boundary_rejections"
ROUTE_FROM_STAGE26_3 = "repair_stage26_synthetic_policy_update_signal_strength"

BOUNDARY_FIELDS = (
    "stage26_4_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.4 synthetic terrain PPO update signal strength repair.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_4_synthetic_policy_update_signal_strength_repair(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_4_synthetic_policy_update_signal_strength_repair(
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
    input_reasons, stage26_3_summary = _input_rejections(config)
    collector_summary: dict[str, Any] = {}
    sweep_rows: list[dict[str, Any]] = []

    if not boundary_reasons and not input_reasons:
        collector_summary = _run_stage26_1(config, output_root, repo_root, stage26_3_summary)
        if _collector_ready(collector_summary, int(config["min_trainable_transition_count"])):
            sweep_rows = _run_sweep(config, output_root=output_root, repo_root=repo_root)

    gradient_audit = _policy_value_gradient_audit(sweep_rows)
    action_audit = _action_signal_audit(sweep_rows)
    credit_audit = _synthetic_credit_signal_audit(output_root / "s26_1")
    status, route, route_reason = _route(
        boundary_reasons=boundary_reasons,
        input_reasons=input_reasons,
        collector_summary=collector_summary,
        sweep_rows=sweep_rows,
        gradient_audit=gradient_audit,
        action_audit=action_audit,
        credit_audit=credit_audit,
        min_transitions=int(config["min_trainable_transition_count"]),
        min_probability_delta=float(config["min_mean_abs_probability_delta_for_signal"]),
        value_to_policy_threshold=float(config["value_to_policy_grad_ratio_dominance_threshold"]),
    )
    blockers = _blocking_reasons(
        route=route,
        boundary_reasons=boundary_reasons,
        input_reasons=input_reasons,
        collector_summary=collector_summary,
        sweep_rows=sweep_rows,
        gradient_audit=gradient_audit,
        action_audit=action_audit,
        min_transitions=int(config["min_trainable_transition_count"]),
    )
    recommended = _recommended_row(sweep_rows)
    _write_recommended_configs(recommended, output_root)

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": blockers,
        "stage26_3_root": config["stage26_3_root"],
        "stage26_3_status": stage26_3_summary.get("status"),
        "stage26_3_next_required_change": stage26_3_summary.get("next_required_change"),
        "stage26_1_root": str(output_root / "s26_1"),
        "stage26_1_status": collector_summary.get("status"),
        "stage26_1_next_required_change": collector_summary.get("next_required_change"),
        "trainable_transition_count": _collector_transition_count(collector_summary),
        "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
        "sweep_combo_count": len(sweep_rows),
        "stable_combo_count": int(gradient_audit["stable_combo_count"]),
        "best_combo_id": recommended.get("combo_id") if recommended else None,
        "best_combo_route": recommended.get("stage26_3_next_required_change") if recommended else None,
        "max_mean_abs_probability_delta": action_audit["max_mean_abs_probability_delta"],
        "max_selected_action_probability_delta_mean": action_audit["max_selected_action_probability_delta_mean"],
        "max_selected_viewpoint_changed_count": action_audit["max_selected_viewpoint_changed_count"],
        "max_selected_theta_changed_count": action_audit["max_selected_theta_changed_count"],
        "max_selected_action_changed_count": action_audit["max_selected_action_changed_count"],
        "max_final_coverage_delta": action_audit["max_final_coverage_delta"],
        "max_coverage_auc_delta": action_audit["max_coverage_auc_delta"],
        "min_hybrid_astar_path_cost_delta": action_audit["min_hybrid_astar_path_cost_delta"],
        "min_value_to_policy_grad_norm_ratio": gradient_audit["min_value_to_policy_grad_norm_ratio"],
        "advantage_gap": credit_audit["advantage_gap"],
        "synthetic_reward_component_gap": credit_audit["synthetic_reward_component_gap"],
        "coverage_source": COVERAGE_SOURCE,
        "path_cost_source": PATH_COST_SOURCE,
        "synthetic_terrain_model_id": SYNTHETIC_MODEL_ID,
        "synthetic_terrain_hash": stage26_3_summary.get("synthetic_terrain_hash") or SYNTHETIC_HASH,
        "synthetic_source_kind": SYNTHETIC_SOURCE_KIND,
        "max_traversable_slope_deg": 30.0,
        "physical_obstacle_cells_written": False,
        "stage26_4_authorized": False,
        "runs_new_ppo_update": bool(sweep_rows),
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
        "stage26_4_authorized": False,
        "runs_new_ppo_update": bool(sweep_rows),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "summary": str(output_root / SUMMARY_FILE),
        "sweep_results": str(output_root / SWEEP_FILE),
        "policy_value_gradient_audit": str(output_root / GRADIENT_AUDIT_FILE),
        "action_signal_audit": str(output_root / ACTION_AUDIT_FILE),
        "synthetic_credit_signal_audit": str(output_root / CREDIT_AUDIT_FILE),
        "recommended_stage26_2_config": str(output_root / RECOMMENDED_26_2_CONFIG),
        "recommended_stage26_3_config": str(output_root / RECOMMENDED_26_3_CONFIG),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }

    _write_json(output_root / SUMMARY_FILE, summary)
    _write_jsonl(output_root / SWEEP_FILE, sweep_rows)
    _write_json(output_root / GRADIENT_AUDIT_FILE, gradient_audit)
    _write_json(output_root / ACTION_AUDIT_FILE, action_audit)
    _write_json(output_root / CREDIT_AUDIT_FILE, credit_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _run_stage26_1(config: dict[str, Any], output_root: Path, repo_root: Path, stage26_3_summary: dict[str, Any]) -> dict[str, Any]:
    cfg = _load_json_template(config["stage26_1_base_config"], repo_root)
    stage26_0_root = config.get("stage26_0_root") or _stage26_0_root_from_stage26_3(stage26_3_summary)
    cfg.update(
        {
            "stage26_0_root": stage26_0_root,
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
            "theta_bin_count": int(config["theta_bin_count"]),
            "theta_step_deg": int(config["theta_step_deg"]),
            "sensor_model_id": str(config["sensor_model_id"]),
            "sensor_fov_deg": float(config["sensor_fov_deg"]),
            "sensor_range_cells": int(config["sensor_range_cells"]),
            "max_traversable_slope_deg": 30.0,
            "stage26_1_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    cfg_path = output_root / COLLECTOR_CONFIG_FILE
    _write_json(cfg_path, cfg)
    return stage26_1.run_xunce_stage26_1_synthetic_terrain_collector_smoke(
        config_path=cfg_path,
        output_root=output_root / "s26_1",
        repo_root=repo_root,
    )


def _run_sweep(config: dict[str, Any], *, output_root: Path, repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, combo in enumerate(config["sweep_combinations"], start=1):
        combo_id = str(combo["combo_id"])
        combo_work_id = _combo_work_id(combo, index)
        combo_root = output_root / "c" / combo_work_id
        stage26_2_config_path = combo_root / "s2.json"
        stage26_3_config_path = combo_root / "s3.json"
        stage26_2_config = _build_stage26_2_config(config, combo, output_root)
        stage26_3_config = _build_stage26_3_config(config, combo_root)
        _write_json(stage26_2_config_path, stage26_2_config)
        stage26_2_summary = stage26_2.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke(
            config_path=stage26_2_config_path,
            output_root=combo_root / "s2",
            repo_root=repo_root,
        )
        stage26_3_summary: dict[str, Any] = {}
        if stage26_2_summary.get("status") == "passed":
            _write_json(stage26_3_config_path, stage26_3_config)
            stage26_3_summary = stage26_3.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke(
                config_path=stage26_3_config_path,
                output_root=combo_root / "s3",
                repo_root=repo_root,
            )
        rows.append(_combo_row(combo, combo_work_id, combo_root, stage26_2_summary, stage26_3_summary))
    return rows


def _build_stage26_2_config(config: dict[str, Any], combo: dict[str, Any], output_root: Path) -> dict[str, Any]:
    cfg = _load_json_template(config["stage26_2_base_config"], Path(config["repo_root"]))
    cfg.update(
        {
            "stage26_1_root": str(output_root / "s26_1"),
            "epochs": int(combo["epochs"]),
            "learning_rate": float(combo["learning_rate"]),
            "clip_ratio": float(combo.get("clip_ratio", config["clip_ratio"])),
            "policy_loss_coefficient": float(combo["policy_loss_coefficient"]),
            "value_loss_coefficient": float(combo["value_loss_coefficient"]),
            "entropy_coefficient": float(combo.get("entropy_coefficient", config["entropy_coefficient"])),
            "advantage_clip_abs": float(config["advantage_clip_abs"]),
            "normalize_minibatch_advantages": True,
            "loss_scale": float(combo["loss_scale"]),
            "max_grad_norm": float(config["max_grad_norm"]),
            "max_abs_approx_kl": float(config["max_abs_approx_kl"]),
            "stage26_2_authorized": False,
            "runs_new_ppo_update": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _build_stage26_3_config(config: dict[str, Any], combo_root: Path) -> dict[str, Any]:
    cfg = _load_json_template(config["stage26_3_base_config"], Path(config["repo_root"]))
    cfg.update(
        {
            "stage26_2_root": str(combo_root / "s2"),
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["eval_rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "theta_bin_count": int(config["theta_bin_count"]),
            "theta_step_deg": int(config["theta_step_deg"]),
            "sensor_model_id": str(config["sensor_model_id"]),
            "sensor_fov_deg": float(config["sensor_fov_deg"]),
            "sensor_range_cells": int(config["sensor_range_cells"]),
            "min_mean_abs_probability_delta_for_signal": float(config["min_mean_abs_probability_delta_for_signal"]),
            "include_oracle_baselines": False,
            "include_canonical_reward_rerank_oracle": False,
            "xunce_only_evaluation": True,
            "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
            "stage21_5_timeout_seconds": float(config["stage21_5_timeout_seconds"]),
            "stage26_3_authorized": False,
            "release_or_training_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _combo_row(
    combo: dict[str, Any],
    combo_work_id: str,
    combo_root: Path,
    stage26_2_summary: dict[str, Any],
    stage26_3_summary: dict[str, Any],
) -> dict[str, Any]:
    loss_gradient = _read_json_if_exists(combo_root / "s2" / stage26_2.LOSS_GRADIENT_AUDIT_FILE)
    component = loss_gradient.get("component_grad_norms") if isinstance(loss_gradient.get("component_grad_norms"), dict) else {}
    policy_grad = _finite_or_default(component.get("policy_loss_grad_norm"), 0.0)
    value_grad = _finite_or_default(component.get("value_loss_grad_norm"), 0.0)
    ratio = value_grad / policy_grad if policy_grad > 0.0 else math.inf
    margin_audit = _candidate_margin_audit(combo_root / "s3")
    return {
        "combo_id": str(combo["combo_id"]),
        "combo_work_id": combo_work_id,
        "combo_root": str(combo_root),
        "epochs": int(combo["epochs"]),
        "learning_rate": float(combo["learning_rate"]),
        "clip_ratio": float(combo.get("clip_ratio", 0.2)),
        "policy_loss_coefficient": float(combo["policy_loss_coefficient"]),
        "value_loss_coefficient": float(combo["value_loss_coefficient"]),
        "loss_scale": float(combo["loss_scale"]),
        "stage26_2_status": stage26_2_summary.get("status"),
        "stage26_2_next_required_change": stage26_2_summary.get("next_required_change"),
        "stage26_2_blocking_reason_codes": stage26_2_summary.get("blocking_reason_codes", []),
        "stage26_3_status": stage26_3_summary.get("status"),
        "stage26_3_next_required_change": stage26_3_summary.get("next_required_change"),
        "stage26_3_blocking_reason_codes": stage26_3_summary.get("blocking_reason_codes", []),
        "batch_row_count": int(stage26_2_summary.get("batch_row_count", 0) or 0),
        "synthetic_contract_mismatch_count": int(stage26_3_summary.get("synthetic_contract_mismatch_count", 0) or 0),
        "synthetic_inference_required_field_missing_count": int(
            stage26_3_summary.get("synthetic_inference_required_field_missing_count", 0) or 0
        ),
        "physical_obstacle_payload_count": int(stage26_3_summary.get("physical_obstacle_payload_count", 0) or 0),
        "grid_fallback_count": int(stage26_3_summary.get("grid_fallback_count", 0) or 0),
        "default_astar_replaced_count": int(stage26_3_summary.get("default_astar_replaced_count", 0) or 0),
        "ackermann_feasible_claimed_count": int(stage26_3_summary.get("ackermann_feasible_claimed_count", 0) or 0),
        "action_probability_audit_is_partial_diagnostic": bool(
            stage26_3_summary.get("action_probability_audit_is_partial_diagnostic", False)
        ),
        "loss_finite": bool(stage26_2_summary.get("loss_finite")),
        "gradient_finite": bool(stage26_2_summary.get("gradient_finite")),
        "checkpoint_reload_passed": bool(stage26_2_summary.get("checkpoint_reload_passed")),
        "checkpoint_boundary_passed": bool(stage26_2_summary.get("checkpoint_boundary_passed")),
        "final_post_update_approx_kl": _finite_or_default(stage26_2_summary.get("final_post_update_approx_kl"), 0.0),
        "pre_clip_grad_norm": _finite_or_default(stage26_2_summary.get("pre_clip_grad_norm"), 0.0),
        "post_clip_grad_norm": _finite_or_default(stage26_2_summary.get("post_clip_grad_norm"), 0.0),
        "parameter_delta_l2": _finite_or_default(stage26_2_summary.get("parameter_delta_l2"), 0.0),
        "policy_loss_grad_norm": policy_grad,
        "value_loss_grad_norm": value_grad,
        "entropy_loss_grad_norm": _finite_or_default(component.get("entropy_loss_grad_norm"), 0.0),
        "value_to_policy_grad_norm_ratio": ratio,
        "strong_state_join_available_count": int(stage26_3_summary.get("strong_state_join_available_count", 0) or 0),
        "mean_abs_probability_delta": _finite_or_default(stage26_3_summary.get("mean_abs_probability_delta"), 0.0),
        "selected_action_probability_delta_mean": _finite_or_default(stage26_3_summary.get("selected_action_probability_delta_mean"), 0.0),
        "selected_action_probability_delta": _finite_or_default(stage26_3_summary.get("selected_action_probability_delta"), 0.0),
        "selected_viewpoint_changed_count": int(stage26_3_summary.get("selected_viewpoint_changed_count", 0) or 0),
        "selected_theta_changed_count": int(stage26_3_summary.get("selected_theta_changed_count", 0) or 0),
        "selected_action_changed_count": int(stage26_3_summary.get("selected_action_changed_count", 0) or 0),
        "final_coverage_delta": _finite_or_default(stage26_3_summary.get("final_coverage_delta"), 0.0),
        "coverage_auc_delta": _finite_or_default(stage26_3_summary.get("coverage_auc_delta"), 0.0),
        "hybrid_astar_path_cost_delta": _finite_or_default(stage26_3_summary.get("hybrid_astar_path_cost_delta"), 0.0),
        "coverage_per_100m_delta": _finite_or_default(stage26_3_summary.get("coverage_per_100m_delta"), 0.0),
        "hard_risk_violation_count": int(stage26_3_summary.get("hard_risk_violation_count", 0) or 0),
        "mask_violation_count": int(stage26_3_summary.get("mask_violation_count", 0) or 0),
        "path_planning_failure_count": int(stage26_3_summary.get("path_planning_failure_count", 0) or 0),
        "open_grid_fallback_count": int(stage26_3_summary.get("open_grid_fallback_count", 0) or 0),
        "selected_vs_best_probability_margin": margin_audit["selected_vs_best_probability_margin_mean"],
        "selected_vs_best_rank_gap": margin_audit["selected_vs_best_rank_gap_mean"],
        "candidate_margin_audit": margin_audit,
        "release_boundary_clean": _release_boundary_clean(stage26_2_summary, stage26_3_summary),
        "coverage_source": COVERAGE_SOURCE,
        "path_cost_source": PATH_COST_SOURCE,
        "synthetic_terrain_hash": stage26_2_summary.get("synthetic_terrain_hash"),
        "synthetic_source_kind": stage26_2_summary.get("synthetic_source_kind"),
        "max_traversable_slope_deg": 30.0,
    }


def _candidate_margin_audit(stage26_3_combo_root: Path) -> dict[str, Any]:
    rows = _read_jsonl_if_exists(stage26_3_combo_root / "post" / stage26_3.MODEL_INFERENCE_FILE)
    margins: list[float] = []
    rank_gaps: list[float] = []
    best_changed_count = 0
    eligible_count = 0
    for row in rows:
        if row.get("policy") not in (None, "xunce"):
            continue
        action_probs = _float_list(row.get("action_probs"))
        cpc = _float_list(_first_present(row.get("theta_coverage_gain_per_path_costs"), row.get("coverage_gain_per_path_costs")))
        selected_index = _int_or_none(row.get("selected_action_index"))
        if selected_index is None or selected_index < 0 or selected_index >= len(action_probs) or len(cpc) != len(action_probs):
            continue
        action_mask = row.get("action_mask")
        valid_indices = [
            index
            for index, _ in enumerate(action_probs)
            if not isinstance(action_mask, list) or (index < len(action_mask) and action_mask[index] is True)
        ]
        if not valid_indices:
            continue
        best_index = max(valid_indices, key=lambda index: cpc[index])
        sorted_by_prob = sorted(valid_indices, key=lambda index: action_probs[index], reverse=True)
        selected_rank = sorted_by_prob.index(selected_index) + 1 if selected_index in sorted_by_prob else None
        best_rank = sorted_by_prob.index(best_index) + 1 if best_index in sorted_by_prob else None
        if selected_rank is None or best_rank is None:
            continue
        eligible_count += 1
        if best_index != selected_index:
            best_changed_count += 1
        margins.append(float(action_probs[best_index] - action_probs[selected_index]))
        rank_gaps.append(float(selected_rank - best_rank))
    return {
        "schema_version": "xunce-stage26-4-candidate-margin-audit/v1",
        "post_inference_row_count": len(rows),
        "eligible_row_count": eligible_count,
        "best_cpc_differs_from_selected_count": best_changed_count,
        "selected_vs_best_probability_margin_mean": mean(margins) if margins else None,
        "selected_vs_best_probability_margin_max": max(margins) if margins else None,
        "selected_vs_best_rank_gap_mean": mean(rank_gaps) if rank_gaps else None,
        "selected_vs_best_rank_gap_max": max(rank_gaps) if rank_gaps else None,
    }


def _route(
    *,
    boundary_reasons: list[str],
    input_reasons: list[str],
    collector_summary: dict[str, Any],
    sweep_rows: list[dict[str, Any]],
    gradient_audit: dict[str, Any],
    action_audit: dict[str, Any],
    credit_audit: dict[str, Any],
    min_transitions: int,
    min_probability_delta: float,
    value_to_policy_threshold: float,
) -> tuple[str, str, str]:
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "Stage26.4 boundary fields are not clean"
    if input_reasons:
        return "failed", ROUTE_INPUTS, "Stage26.3 inputs are missing or not the synthetic signal route"
    if not _collector_ready(collector_summary, min_transitions):
        return "failed", ROUTE_COLLECTOR, "collector produced fewer trainable transitions than required"
    if _synthetic_lineage_failed(sweep_rows):
        return "failed", ROUTE_LINEAGE, "synthetic terrain lineage mismatch in sweep"
    stable = _stable_rows(sweep_rows)
    if not stable:
        return "failed", ROUTE_STABILITY, "no stable Stage26.2/26.3 combo completed"
    if all(float(row.get("value_to_policy_grad_norm_ratio", math.inf)) > value_to_policy_threshold for row in stable):
        return "failed", ROUTE_VALUE_BALANCE, "value loss still dominates policy gradient across stable combos"
    if action_audit["max_mean_abs_probability_delta"] < min_probability_delta and action_audit["max_selected_viewpoint_changed_count"] <= 0:
        return "failed", ROUTE_STRENGTH, "stable combos still produce negligible synthetic action probability change"
    if action_audit["max_selected_viewpoint_changed_count"] <= 0 and action_audit["max_selected_theta_changed_count"] <= 0:
        return "failed", ROUTE_MARGIN, "probabilities moved but did not cross viewpoint/theta selection boundary"
    if _same_combo_improved(stable):
        return "passed", ROUTE_STAGE26_5, "synthetic terrain update moved actions and improved bounded coverage/AUC/path cost"
    if credit_audit.get("advantage_gap") is not None and float(credit_audit["advantage_gap"]) <= 0.0:
        return "failed", ROUTE_CREDIT, "synthetic advantage gap is not positive"
    return "failed", ROUTE_CREDIT, "synthetic action changed without bounded coverage/AUC/path-cost improvement"


def _blocking_reasons(
    *,
    route: str,
    boundary_reasons: list[str],
    input_reasons: list[str],
    collector_summary: dict[str, Any],
    sweep_rows: list[dict[str, Any]],
    gradient_audit: dict[str, Any],
    action_audit: dict[str, Any],
    min_transitions: int,
) -> list[str]:
    reasons = list(boundary_reasons) + list(input_reasons)
    if route == ROUTE_COLLECTOR:
        if _collector_transition_count(collector_summary) < min_transitions:
            reasons.append("stage26_1_trainable_transition_count_below_minimum")
        if collector_summary.get("status") != "passed":
            reasons.extend(f"stage26_1_{code}" for code in collector_summary.get("blocking_reason_codes", []))
    if route == ROUTE_LINEAGE:
        reasons.append("synthetic_lineage_or_inference_binding_mismatch")
    if route == ROUTE_STABILITY:
        reasons.append("no_stable_stage26_2_3_combo")
        for row in sweep_rows:
            if row.get("stage26_2_status") != "passed":
                reasons.extend(f"{row['combo_id']}_stage26_2_{code}" for code in row.get("stage26_2_blocking_reason_codes", []))
            if row.get("hard_risk_violation_count") or row.get("mask_violation_count") or row.get("path_planning_failure_count") or row.get("open_grid_fallback_count"):
                reasons.append(f"{row['combo_id']}_safety_counter_nonzero")
    if route == ROUTE_VALUE_BALANCE:
        reasons.append("value_loss_grad_norm_ratio_above_threshold")
    if route == ROUTE_STRENGTH:
        reasons.append("synthetic_probability_delta_below_threshold")
    if route == ROUTE_MARGIN:
        reasons.append("synthetic_probability_changed_without_viewpoint_change")
    if route == ROUTE_CREDIT:
        reasons.append("synthetic_action_changed_without_coverage_auc_path_cost_improvement")
    return _unique(reasons)


def _collector_ready(summary: dict[str, Any], min_transitions: int) -> bool:
    return (
        summary.get("status") == "passed"
        and summary.get("next_required_change") == stage26_1.ROUTE_STAGE26_2
        and _collector_transition_count(summary) >= min_transitions
    )


def _collector_transition_count(summary: dict[str, Any]) -> int:
    for key in ("trainable_transition_count", "transition_count", "batch_row_count"):
        try:
            value = int(summary.get(key, 0) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return 0


def _stable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("stage26_2_status") == "passed"
        and row.get("loss_finite") is True
        and row.get("gradient_finite") is True
        and row.get("checkpoint_reload_passed") is True
        and row.get("checkpoint_boundary_passed") is True
        and row.get("release_boundary_clean") is True
        and int(row.get("strong_state_join_available_count", 0) or 0) > 0
        and int(row.get("synthetic_contract_mismatch_count", 0) or 0) == 0
        and int(row.get("synthetic_inference_required_field_missing_count", 0) or 0) == 0
        and int(row.get("physical_obstacle_payload_count", 0) or 0) == 0
        and int(row.get("grid_fallback_count", 0) or 0) == 0
        and int(row.get("default_astar_replaced_count", 0) or 0) == 0
        and int(row.get("ackermann_feasible_claimed_count", 0) or 0) == 0
        and row.get("action_probability_audit_is_partial_diagnostic") is not True
        and int(row.get("hard_risk_violation_count", 0) or 0) == 0
        and int(row.get("mask_violation_count", 0) or 0) == 0
        and int(row.get("path_planning_failure_count", 0) or 0) == 0
        and int(row.get("open_grid_fallback_count", 0) or 0) == 0
        and _finite_float(row.get("final_post_update_approx_kl")) is not None
        and row.get("synthetic_terrain_hash") == SYNTHETIC_HASH
        and row.get("synthetic_source_kind") == SYNTHETIC_SOURCE_KIND
    ]


def _synthetic_lineage_failed(rows: list[dict[str, Any]]) -> bool:
    return any(
        row.get("synthetic_terrain_hash") not in (None, SYNTHETIC_HASH)
        or row.get("synthetic_source_kind") not in (None, SYNTHETIC_SOURCE_KIND)
        or int(row.get("synthetic_contract_mismatch_count", 0) or 0) > 0
        or int(row.get("synthetic_inference_required_field_missing_count", 0) or 0) > 0
        or int(row.get("physical_obstacle_payload_count", 0) or 0) > 0
        or int(row.get("grid_fallback_count", 0) or 0) > 0
        or int(row.get("default_astar_replaced_count", 0) or 0) > 0
        or int(row.get("ackermann_feasible_claimed_count", 0) or 0) > 0
        or row.get("action_probability_audit_is_partial_diagnostic") is True
        for row in rows
    )


def _same_combo_improved(rows: list[dict[str, Any]]) -> bool:
    return any(
        float(row.get("final_coverage_delta", 0.0) or 0.0) > 0.0
        and float(row.get("coverage_auc_delta", 0.0) or 0.0) > 0.0
        and float(row.get("hybrid_astar_path_cost_delta", 0.0) or 0.0) < 0.0
        and int(row.get("selected_viewpoint_changed_count", 0) or 0) > 0
        for row in rows
    )


def _policy_value_gradient_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stable = _stable_rows(rows)
    ratios = [float(row["value_to_policy_grad_norm_ratio"]) for row in stable if _finite_float(row.get("value_to_policy_grad_norm_ratio")) is not None]
    return {
        "schema_version": GRADIENT_AUDIT_SCHEMA_VERSION,
        "combo_count": len(rows),
        "stable_combo_count": len(stable),
        "min_value_to_policy_grad_norm_ratio": min(ratios) if ratios else None,
        "max_value_to_policy_grad_norm_ratio": max(ratios) if ratios else None,
        "mean_value_to_policy_grad_norm_ratio": sum(ratios) / len(ratios) if ratios else None,
        "rows": [
            {
                "combo_id": row["combo_id"],
                "policy_loss_grad_norm": row["policy_loss_grad_norm"],
                "value_loss_grad_norm": row["value_loss_grad_norm"],
                "entropy_loss_grad_norm": row["entropy_loss_grad_norm"],
                "value_to_policy_grad_norm_ratio": row["value_to_policy_grad_norm_ratio"],
                "pre_clip_grad_norm": row["pre_clip_grad_norm"],
                "post_clip_grad_norm": row["post_clip_grad_norm"],
                "parameter_delta_l2": row["parameter_delta_l2"],
                "final_post_update_approx_kl": row["final_post_update_approx_kl"],
            }
            for row in rows
        ],
    }


def _action_signal_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stable = _stable_rows(rows)
    source = stable or rows
    probability_margins = [_finite_float(row.get("selected_vs_best_probability_margin")) for row in source]
    rank_gaps = [_finite_float(row.get("selected_vs_best_rank_gap")) for row in source]
    return {
        "schema_version": ACTION_AUDIT_SCHEMA_VERSION,
        "combo_count": len(rows),
        "stable_combo_count": len(stable),
        "max_mean_abs_probability_delta": max((_finite_or_default(row.get("mean_abs_probability_delta"), 0.0) for row in source), default=0.0),
        "max_selected_action_probability_delta_mean": max(
            (_finite_or_default(row.get("selected_action_probability_delta_mean"), 0.0) for row in source),
            default=0.0,
        ),
        "max_selected_viewpoint_changed_count": max((int(row.get("selected_viewpoint_changed_count", 0) or 0) for row in source), default=0),
        "max_selected_theta_changed_count": max((int(row.get("selected_theta_changed_count", 0) or 0) for row in source), default=0),
        "max_selected_action_changed_count": max((int(row.get("selected_action_changed_count", 0) or 0) for row in source), default=0),
        "max_final_coverage_delta": max((_finite_or_default(row.get("final_coverage_delta"), 0.0) for row in source), default=0.0),
        "max_coverage_auc_delta": max((_finite_or_default(row.get("coverage_auc_delta"), 0.0) for row in source), default=0.0),
        "min_hybrid_astar_path_cost_delta": min((_finite_or_default(row.get("hybrid_astar_path_cost_delta"), 0.0) for row in source), default=0.0),
        "min_selected_vs_best_probability_margin": min((value for value in probability_margins if value is not None), default=None),
        "max_selected_vs_best_rank_gap": max((value for value in rank_gaps if value is not None), default=None),
        "rows": [
            {
                "combo_id": row["combo_id"],
                "strong_state_join_available_count": row["strong_state_join_available_count"],
                "mean_abs_probability_delta": row["mean_abs_probability_delta"],
                "selected_action_probability_delta_mean": row["selected_action_probability_delta_mean"],
                "selected_viewpoint_changed_count": row["selected_viewpoint_changed_count"],
                "selected_theta_changed_count": row["selected_theta_changed_count"],
                "selected_action_changed_count": row["selected_action_changed_count"],
                "final_coverage_delta": row["final_coverage_delta"],
                "coverage_auc_delta": row["coverage_auc_delta"],
                "hybrid_astar_path_cost_delta": row["hybrid_astar_path_cost_delta"],
            }
            for row in rows
        ],
    }


def _synthetic_credit_signal_audit(stage26_1_root: Path) -> dict[str, Any]:
    batch_rows = _read_jsonl_if_exists(stage26_1_root / "s21_3" / stage26_2.stage21_3.BATCH_FILE)
    reward_rows = _read_jsonl_if_exists(stage26_1_root / "s21_2" / stage26_1.stage21_2.EVALUATION_FILE)
    advantages = [_finite_float(_first_present(row.get("advantage"), row.get("gae_advantage"), row.get("normalized_advantage"))) for row in batch_rows]
    advantages = [value for value in advantages if value is not None]
    positive = [value for value in advantages if value > 0.0]
    negative = [value for value in advantages if value < 0.0]
    rewards = [_finite_float(_first_present(row.get("reward"), row.get("total_reward"), row.get("coverage_first_reward"))) for row in reward_rows]
    rewards = [value for value in rewards if value is not None]
    synthetic_components = [
        _finite_float(
            _first_present(
                row.get("obstacle_aware_theta_coverage_gain_per_path_cost"),
                row.get("theta_coverage_gain_per_path_cost"),
                row.get("coverage_gain_per_path_cost"),
            )
        )
        for row in reward_rows
    ]
    synthetic_components = [value for value in synthetic_components if value is not None]
    advantage_gap = (mean(positive) - mean(negative)) if positive and negative else None
    reward_gap = (max(rewards) - min(rewards)) if rewards else None
    synthetic_gap = (max(synthetic_components) - min(synthetic_components)) if synthetic_components else None
    return {
        "schema_version": CREDIT_AUDIT_SCHEMA_VERSION,
        "batch_row_count": len(batch_rows),
        "reward_row_count": len(reward_rows),
        "advantage_count": len(advantages),
        "advantage_mean": mean(advantages) if advantages else None,
        "advantage_std": pstdev(advantages) if len(advantages) > 1 else 0.0 if advantages else None,
        "advantage_min": min(advantages) if advantages else None,
        "advantage_max": max(advantages) if advantages else None,
        "positive_advantage_mean": mean(positive) if positive else None,
        "negative_advantage_mean": mean(negative) if negative else None,
        "advantage_gap": advantage_gap,
        "reward_gap": reward_gap,
        "synthetic_reward_component_gap": synthetic_gap,
        "synthetic_component_count": len(synthetic_components),
    }


def _recommended_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stable = _stable_rows(rows)
    if not stable:
        return rows[0] if rows else {}
    return sorted(
        stable,
        key=lambda row: (
            -float(row.get("final_coverage_delta", 0.0) or 0.0),
            -float(row.get("coverage_auc_delta", 0.0) or 0.0),
            float(row.get("hybrid_astar_path_cost_delta", 0.0) or 0.0),
            -int(row.get("selected_viewpoint_changed_count", 0) or 0),
            -float(row.get("mean_abs_probability_delta", 0.0) or 0.0),
            float(row.get("value_to_policy_grad_norm_ratio", math.inf)),
        ),
    )[0]


def _write_recommended_configs(row: dict[str, Any], output_root: Path) -> None:
    if not row:
        _write_json(output_root / RECOMMENDED_26_2_CONFIG, {})
        _write_json(output_root / RECOMMENDED_26_3_CONFIG, {})
        return
    combo_root = Path(str(row["combo_root"]))
    _write_json(output_root / RECOMMENDED_26_2_CONFIG, _read_json_if_exists(combo_root / "s2.json"))
    _write_json(output_root / RECOMMENDED_26_3_CONFIG, _read_json_if_exists(combo_root / "s3.json"))


def _input_rejections(config: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    summary_path = Path(config["stage26_3_root"]) / stage26_3.SUMMARY_FILE
    if not summary_path.is_file():
        return ["missing_stage26_3_summary"], {}
    summary = _read_json(summary_path)
    reasons: list[str] = []
    if summary.get("status") != "failed":
        reasons.append("stage26_3_status_not_failed_signal_smoke")
    if summary.get("next_required_change") != ROUTE_FROM_STAGE26_3:
        reasons.append("stage26_3_route_not_synthetic_signal_repair")
    if summary.get("action_probability_audit_is_partial_diagnostic") is not False:
        reasons.append("stage26_3_action_probability_audit_partial")
    if int(summary.get("strong_state_join_available_count", 0) or 0) <= 0:
        reasons.append("stage26_3_strong_join_unavailable")
    if summary.get("coverage_source") != COVERAGE_SOURCE:
        reasons.append("stage26_3_coverage_source_mismatch")
    if summary.get("path_cost_source") != PATH_COST_SOURCE:
        reasons.append("stage26_3_path_cost_source_mismatch")
    if summary.get("synthetic_source_kind") != SYNTHETIC_SOURCE_KIND:
        reasons.append("stage26_3_synthetic_source_kind_mismatch")
    if summary.get("synthetic_terrain_hash") != SYNTHETIC_HASH:
        reasons.append("stage26_3_synthetic_hash_mismatch")
    if _finite_float(summary.get("max_traversable_slope_deg")) != 30.0:
        reasons.append("stage26_3_max_traversable_slope_not_30")
    for key in ("synthetic_contract_mismatch_count", "physical_obstacle_payload_count", "grid_fallback_count"):
        if int(summary.get(key, 0) or 0) != 0:
            reasons.append(f"stage26_3_{key}_nonzero")
    for key in ("hard_risk_violation_count", "mask_violation_count", "path_planning_failure_count", "open_grid_fallback_count"):
        if int(summary.get(key, 0) or 0) != 0:
            reasons.append(f"stage26_3_{key}_nonzero")
    for key in ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary"):
        if summary.get(key) is True:
            reasons.append(f"stage26_3_{key}_enabled")
    stage26_1_root = summary.get("stage26_1_root")
    if not stage26_1_root or not (Path(str(stage26_1_root)) / stage26_1.SUMMARY_FILE).is_file():
        reasons.append("missing_stage26_1_summary_from_stage26_3_lineage")
    return reasons, summary


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        reasons.append("canary_traffic_fraction_nonzero")
    return reasons


def _stage26_0_root_from_stage26_3(stage26_3_summary: dict[str, Any]) -> str:
    stage26_1_root = stage26_3_summary.get("stage26_1_root")
    if stage26_1_root:
        stage26_1_summary = _read_json_if_exists(Path(str(stage26_1_root)) / stage26_1.SUMMARY_FILE)
        value = stage26_1_summary.get("stage26_0_root")
        if value:
            return str(value)
    return stage26_1.DEFAULT_STAGE26_0_ROOT


def _release_boundary_clean(stage26_2_summary: dict[str, Any], stage26_3_summary: dict[str, Any]) -> bool:
    for payload in (stage26_2_summary, stage26_3_summary):
        for key in ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary"):
            if payload.get(key) is True:
                return False
        if float(payload.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
            return False
    return True


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"config schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    defaults: dict[str, Any] = {
        "stage26_3_root": DEFAULT_STAGE26_3_ROOT,
        "stage26_0_root": "",
        "stage26_1_base_config": "configs/xunce_stage26_1_synthetic_terrain_collector_smoke_v1.json",
        "stage26_2_base_config": "configs/xunce_stage26_2_synthetic_terrain_ppo_update_smoke_v1.json",
        "stage26_3_base_config": "configs/xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke_v1.json",
        "required_scenario_count": 2,
        "rollout_steps": 8,
        "eval_rollout_steps": 4,
        "dynamic_max_candidates_per_step": 36,
        "dynamic_proposal_pool_limit_per_step": 288,
        "min_trainable_transition_count": 16,
        "theta_bin_count": 8,
        "theta_step_deg": 45,
        "sensor_model_id": "theta-fov-90-range-radius/v1",
        "sensor_fov_deg": 90.0,
        "sensor_range_cells": 3,
        "clip_ratio": 0.2,
        "entropy_coefficient": 0.01,
        "advantage_clip_abs": 5.0,
        "max_grad_norm": 1.0,
        "max_abs_approx_kl": 0.5,
        "min_mean_abs_probability_delta_for_signal": 1.0e-5,
        "value_to_policy_grad_ratio_dominance_threshold": 10.0,
        "hybrid_astar_candidate_eval_workers": 4,
        "stage21_5_timeout_seconds": 7200.0,
        "sweep_combinations": _default_sweep_combinations(),
        "stage26_4_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    config["repo_root"] = str(repo_root)
    for key in ("stage26_3_root", "stage26_1_base_config", "stage26_2_base_config", "stage26_3_base_config"):
        config[key] = str(_resolve_path(Path(str(config[key])), repo_root))
    if config.get("stage26_0_root"):
        config["stage26_0_root"] = str(_resolve_path(Path(str(config["stage26_0_root"])), repo_root))
    for key in (
        "required_scenario_count",
        "rollout_steps",
        "eval_rollout_steps",
        "dynamic_max_candidates_per_step",
        "dynamic_proposal_pool_limit_per_step",
        "min_trainable_transition_count",
        "theta_bin_count",
        "theta_step_deg",
        "sensor_range_cells",
        "hybrid_astar_candidate_eval_workers",
    ):
        config[key] = _positive_int(config[key], key)
    for key in (
        "sensor_fov_deg",
        "clip_ratio",
        "entropy_coefficient",
        "advantage_clip_abs",
        "max_grad_norm",
        "max_abs_approx_kl",
        "min_mean_abs_probability_delta_for_signal",
        "value_to_policy_grad_ratio_dominance_threshold",
        "stage21_5_timeout_seconds",
        "canary_traffic_fraction",
    ):
        config[key] = _finite_number(config[key], key)
    return config


def _default_sweep_combinations() -> list[dict[str, Any]]:
    return [
        {
            "combo_id": "baseline_current",
            "epochs": 1,
            "learning_rate": 2.0e-6,
            "clip_ratio": 0.2,
            "value_loss_coefficient": 0.1,
            "policy_loss_coefficient": 1.0,
            "loss_scale": 0.25,
        },
        {
            "combo_id": "value_off_depth",
            "epochs": 4,
            "learning_rate": 2.0e-5,
            "clip_ratio": 0.2,
            "value_loss_coefficient": 0.0,
            "policy_loss_coefficient": 1.0,
            "loss_scale": 0.25,
        },
        {
            "combo_id": "policy_amp_depth",
            "epochs": 8,
            "learning_rate": 2.0e-5,
            "clip_ratio": 0.2,
            "value_loss_coefficient": 0.02,
            "policy_loss_coefficient": 2.0,
            "loss_scale": 0.5,
        },
    ]


def _combo_work_id(combo: dict[str, Any], index: int) -> str:
    raw = str(combo.get("work_id") or f"c{index}")
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)
    return safe[:12] or f"c{index}"


def _load_json_template(path: str, repo_root: Path) -> dict[str, Any]:
    return _read_json(_resolve_path(Path(path), repo_root))


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _finite_number(value: Any, name: str) -> float:
    parsed = _finite_float(value)
    if parsed is None:
        raise ValueError(f"{name} must be finite")
    return parsed


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_or_default(value: Any, default: float) -> float:
    parsed = _finite_float(value)
    return parsed if parsed is not None else float(default)


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    output: list[float] = []
    for item in value:
        parsed = _finite_float(item)
        if parsed is None:
            return []
        output.append(parsed)
    return output


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.4 Synthetic Policy Update Signal Strength Repair",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- trainable_transition_count: `{summary['trainable_transition_count']}`",
            f"- stable_combo_count: `{summary['stable_combo_count']}`",
            f"- best_combo_id: `{summary['best_combo_id']}`",
            f"- max_mean_abs_probability_delta: `{summary['max_mean_abs_probability_delta']}`",
            f"- max_selected_viewpoint_changed_count: `{summary['max_selected_viewpoint_changed_count']}`",
            f"- max_final_coverage_delta: `{summary['max_final_coverage_delta']}`",
            f"- max_coverage_auc_delta: `{summary['max_coverage_auc_delta']}`",
            f"- min_hybrid_astar_path_cost_delta: `{summary['min_hybrid_astar_path_cost_delta']}`",
            "",
            "Stage26.4 is a bounded offline diagnostic. It reuses Stage26.1/26.2/26.3 and does not publish checkpoints, replace default policy, connect an executor, start canary traffic, or change reward/network/default A*/Hybrid A*/candidate generation semantics.",
            "",
        ]
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
