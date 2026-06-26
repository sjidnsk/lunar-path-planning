from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke as stage23_3
    import run_xunce_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke as stage23_4
    import run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke as stage23_5
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke as stage23_3
    import scripts.run_xunce_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke as stage23_4
    import scripts.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke as stage23_5


STAGE_ID = "xunce-stage23-6-slope-theta-policy-update-signal-strength-repair"
CONFIG_SCHEMA_VERSION = "xunce-stage23-6-slope-theta-policy-update-signal-strength-repair-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage23-6-summary/v1"
GRADIENT_AUDIT_SCHEMA_VERSION = "xunce-stage23-6-policy-value-gradient-audit/v1"
ACTION_AUDIT_SCHEMA_VERSION = "xunce-stage23-6-action-signal-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage23-6-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage23-6-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage23_6_slope_theta_policy_update_signal_strength_repair_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage23_slope_obstacle_aware_theta_reward/"
    "outputs/path_feedback_batch_xunce_stage23_6_slope_theta_policy_update_signal_strength_repair_v1"
)

SLOPE_COVERAGE_SOURCE = "endpoint_theta_slope_obstacle_los/v1"
SUMMARY_FILE = "xunce-stage23-6-summary.json"
SWEEP_FILE = "xunce-stage23-6-sweep-results.jsonl"
GRADIENT_AUDIT_FILE = "xunce-stage23-6-policy-value-gradient-audit.json"
ACTION_AUDIT_FILE = "xunce-stage23-6-action-signal-audit.json"
RECOMMENDED_23_4_CONFIG = "xunce-stage23-6-recommended-stage23-4-config.json"
RECOMMENDED_23_5_CONFIG = "xunce-stage23-6-recommended-stage23-5-config.json"
ROUTING_FILE = "xunce-stage23-6-next-stage-routing.json"
REPORT_FILE = "xunce-stage23-6-report.md"
MANIFEST_FILE = "xunce-stage23-6-manifest.json"
COLLECTOR_CONFIG_FILE = "xunce-stage23-6-stage23-3-config.json"

ROUTE_INPUTS = "rerun_stage23_6_required_inputs"
ROUTE_COLLECTOR = "expand_stage23_slope_theta_collector_samples"
ROUTE_STABILITY = "repair_stage23_slope_theta_ppo_update_stability"
ROUTE_VALUE_BALANCE = "repair_stage23_policy_value_loss_balance"
ROUTE_SIGNAL = "repair_stage23_slope_theta_policy_update_signal_strength"
ROUTE_STRENGTH = "increase_stage23_slope_theta_update_strength_or_sample_count"
ROUTE_MARGIN = "calibrate_stage23_slope_theta_discrete_margin_crossing"
ROUTE_CREDIT = "repair_stage23_slope_theta_credit_assignment"
ROUTE_STAGE23_7 = "run_stage23_7_slope_obstacle_aware_theta_multi_seed_ppo_pilot"
ROUTE_BOUNDARY = "resolve_stage23_6_boundary_rejections"

BOUNDARY_FIELDS = (
    "stage23_6_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage23.6 slope-theta PPO update signal strength repair.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage23_6_slope_theta_policy_update_signal_strength_repair(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage23_6_slope_theta_policy_update_signal_strength_repair(
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
    input_reasons, stage23_5a_summary = _input_rejections(config)

    collector_summary: dict[str, Any] = {}
    sweep_rows: list[dict[str, Any]] = []
    if not boundary_reasons and not input_reasons:
        collector_summary = _run_stage23_3(config, output_root, repo_root)
        if _collector_ready(collector_summary, int(config["min_trainable_transition_count"])):
            sweep_rows = _run_sweep(config, output_root=output_root, repo_root=repo_root)

    gradient_audit = _policy_value_gradient_audit(sweep_rows)
    action_audit = _action_signal_audit(sweep_rows)
    status, route, route_reason = _route(
        boundary_reasons=boundary_reasons,
        input_reasons=input_reasons,
        collector_summary=collector_summary,
        sweep_rows=sweep_rows,
        gradient_audit=gradient_audit,
        action_audit=action_audit,
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
        "stage23_5a_root": config["stage23_5a_root"],
        "stage23_5a_status": stage23_5a_summary.get("status"),
        "stage23_5a_next_required_change": stage23_5a_summary.get("next_required_change"),
        "stage23_3_root": str(output_root / "s23_3"),
        "stage23_3_status": collector_summary.get("status"),
        "stage23_3_next_required_change": collector_summary.get("next_required_change"),
        "trainable_transition_count": int(collector_summary.get("trainable_transition_count", 0) or 0),
        "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
        "sweep_combo_count": len(sweep_rows),
        "stable_combo_count": int(gradient_audit["stable_combo_count"]),
        "best_combo_id": recommended.get("combo_id") if recommended else None,
        "best_combo_route": recommended.get("stage23_5_next_required_change") if recommended else None,
        "max_mean_abs_probability_delta": action_audit["max_mean_abs_probability_delta"],
        "max_selected_action_probability_delta_mean": action_audit["max_selected_action_probability_delta_mean"],
        "max_selected_viewpoint_changed_count": action_audit["max_selected_viewpoint_changed_count"],
        "max_selected_theta_changed_count": action_audit["max_selected_theta_changed_count"],
        "max_selected_action_changed_count": action_audit["max_selected_action_changed_count"],
        "max_final_coverage_delta": action_audit["max_final_coverage_delta"],
        "max_coverage_auc_delta": action_audit["max_coverage_auc_delta"],
        "min_value_to_policy_grad_norm_ratio": gradient_audit["min_value_to_policy_grad_norm_ratio"],
        "coverage_source": SLOPE_COVERAGE_SOURCE,
        "max_traversable_slope_deg": 30.0,
        "platform_contract_hash": _first_present(
            collector_summary.get("platform_contract_hash"),
            stage23_5a_summary.get("platform_contract_hash"),
        ),
        "stage23_6_authorized": False,
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
        "stage23_6_authorized": False,
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
        "recommended_stage23_4_config": str(output_root / RECOMMENDED_23_4_CONFIG),
        "recommended_stage23_5_config": str(output_root / RECOMMENDED_23_5_CONFIG),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }

    _write_json(output_root / SUMMARY_FILE, summary)
    _write_jsonl(output_root / SWEEP_FILE, sweep_rows)
    _write_json(output_root / GRADIENT_AUDIT_FILE, gradient_audit)
    _write_json(output_root / ACTION_AUDIT_FILE, action_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _run_stage23_3(config: dict[str, Any], output_root: Path, repo_root: Path) -> dict[str, Any]:
    cfg = _load_json_template(config["stage23_3_base_config"], repo_root)
    cfg.update(
        {
            "stage23_2_root": config["stage23_2_root"],
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
            "stage23_3_authorized": False,
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
    return stage23_3.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke(
        config_path=cfg_path,
        output_root=output_root / "s23_3",
        repo_root=repo_root,
    )


def _run_sweep(config: dict[str, Any], *, output_root: Path, repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, combo in enumerate(config["sweep_combinations"], start=1):
        combo_id = str(combo["combo_id"])
        combo_work_id = _combo_work_id(combo, index)
        combo_root = output_root / "c" / combo_work_id
        stage23_4_config_path = combo_root / "s4.json"
        stage23_5_config_path = combo_root / "s5.json"
        stage23_4_config = _build_stage23_4_config(config, combo, output_root)
        stage23_5_config = _build_stage23_5_config(config, combo_root)
        _write_json(stage23_4_config_path, stage23_4_config)
        stage23_4_summary = stage23_4.run_xunce_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke(
            config_path=stage23_4_config_path,
            output_root=combo_root / "s4",
            repo_root=repo_root,
        )
        stage23_5_summary: dict[str, Any] = {}
        if stage23_4_summary.get("status") == "passed":
            _write_json(stage23_5_config_path, stage23_5_config)
            stage23_5_summary = stage23_5.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke(
                config_path=stage23_5_config_path,
                output_root=combo_root / "s5",
                repo_root=repo_root,
            )
        rows.append(_combo_row(combo, combo_work_id, combo_root, stage23_4_summary, stage23_5_summary))
    return rows


def _build_stage23_4_config(config: dict[str, Any], combo: dict[str, Any], output_root: Path) -> dict[str, Any]:
    cfg = _load_json_template(config["stage23_4_base_config"], Path(config["repo_root"]))
    cfg.update(
        {
            "stage23_3_root": str(output_root / "s23_3"),
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
            "stage23_4_authorized": False,
            "runs_new_ppo_update": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _build_stage23_5_config(config: dict[str, Any], combo_root: Path) -> dict[str, Any]:
    cfg = _load_json_template(config["stage23_5_base_config"], Path(config["repo_root"]))
    cfg.update(
        {
            "stage23_4_root": str(combo_root / "s4"),
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "theta_bin_count": int(config["theta_bin_count"]),
            "theta_step_deg": int(config["theta_step_deg"]),
            "sensor_model_id": str(config["sensor_model_id"]),
            "sensor_fov_deg": float(config["sensor_fov_deg"]),
            "sensor_range_cells": int(config["sensor_range_cells"]),
            "min_mean_abs_probability_delta_for_signal": float(config["min_mean_abs_probability_delta_for_signal"]),
            "coverage_source": SLOPE_COVERAGE_SOURCE,
            "max_traversable_slope_deg": 30.0,
            "stage23_5_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _combo_work_id(combo: dict[str, Any], index: int) -> str:
    raw = str(combo.get("work_id") or f"c{index}")
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)
    return safe[:12] or f"c{index}"


def _combo_row(
    combo: dict[str, Any],
    combo_work_id: str,
    combo_root: Path,
    stage23_4_summary: dict[str, Any],
    stage23_5_summary: dict[str, Any],
) -> dict[str, Any]:
    loss_gradient = _read_json_if_exists(combo_root / "s4" / stage23_4.LOSS_GRADIENT_AUDIT_FILE)
    component = loss_gradient.get("component_grad_norms") if isinstance(loss_gradient.get("component_grad_norms"), dict) else {}
    policy_grad = _finite_float(component.get("policy_loss_grad_norm")) or 0.0
    value_grad = _finite_float(component.get("value_loss_grad_norm")) or 0.0
    ratio = value_grad / policy_grad if policy_grad > 0.0 else math.inf
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
        "stage23_4_status": stage23_4_summary.get("status"),
        "stage23_4_next_required_change": stage23_4_summary.get("next_required_change"),
        "stage23_4_blocking_reason_codes": stage23_4_summary.get("blocking_reason_codes", []),
        "stage23_5_status": stage23_5_summary.get("status"),
        "stage23_5_next_required_change": stage23_5_summary.get("next_required_change"),
        "stage23_5_blocking_reason_codes": stage23_5_summary.get("blocking_reason_codes", []),
        "loss_finite": bool(stage23_4_summary.get("loss_finite")),
        "gradient_finite": bool(stage23_4_summary.get("gradient_finite")),
        "checkpoint_reload_passed": bool(stage23_4_summary.get("checkpoint_reload_passed")),
        "checkpoint_boundary_passed": bool(stage23_4_summary.get("checkpoint_boundary_passed")),
        "final_post_update_approx_kl": _finite_or_default(stage23_4_summary.get("final_post_update_approx_kl"), 0.0),
        "pre_clip_grad_norm": _finite_or_default(stage23_4_summary.get("pre_clip_grad_norm"), 0.0),
        "post_clip_grad_norm": _finite_or_default(stage23_4_summary.get("post_clip_grad_norm"), 0.0),
        "parameter_delta_l2": _finite_or_default(stage23_4_summary.get("parameter_delta_l2"), 0.0),
        "policy_loss_grad_norm": policy_grad,
        "value_loss_grad_norm": value_grad,
        "entropy_loss_grad_norm": _finite_or_default(component.get("entropy_loss_grad_norm"), 0.0),
        "value_to_policy_grad_norm_ratio": ratio,
        "strong_state_join_available_count": int(stage23_5_summary.get("strong_state_join_available_count", 0) or 0),
        "mean_abs_probability_delta": _finite_or_default(stage23_5_summary.get("mean_abs_probability_delta"), 0.0),
        "selected_action_probability_delta_mean": _finite_or_default(stage23_5_summary.get("selected_action_probability_delta_mean"), 0.0),
        "selected_viewpoint_changed_count": int(stage23_5_summary.get("selected_viewpoint_changed_count", 0) or 0),
        "selected_theta_changed_count": int(stage23_5_summary.get("selected_theta_changed_count", 0) or 0),
        "selected_action_changed_count": int(stage23_5_summary.get("selected_action_changed_count", 0) or 0),
        "final_coverage_delta": _finite_or_default(stage23_5_summary.get("final_coverage_delta"), 0.0),
        "coverage_auc_delta": _finite_or_default(stage23_5_summary.get("coverage_auc_delta"), 0.0),
        "path_cost_delta": _finite_or_default(stage23_5_summary.get("path_cost_delta"), 0.0),
        "hard_risk_violation_count": int(stage23_5_summary.get("hard_risk_violation_count", 0) or 0),
        "mask_violation_count": int(stage23_5_summary.get("mask_violation_count", 0) or 0),
        "path_planning_failure_count": int(stage23_5_summary.get("path_planning_failure_count", 0) or 0),
        "open_grid_fallback_count": int(stage23_5_summary.get("open_grid_fallback_count", 0) or 0),
        "release_boundary_clean": _release_boundary_clean(stage23_4_summary, stage23_5_summary),
        "coverage_source": SLOPE_COVERAGE_SOURCE,
        "max_traversable_slope_deg": 30.0,
    }


def _route(
    *,
    boundary_reasons: list[str],
    input_reasons: list[str],
    collector_summary: dict[str, Any],
    sweep_rows: list[dict[str, Any]],
    gradient_audit: dict[str, Any],
    action_audit: dict[str, Any],
    min_transitions: int,
    min_probability_delta: float,
    value_to_policy_threshold: float,
) -> tuple[str, str, str]:
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "Stage23.6 boundary fields are not clean"
    if input_reasons:
        return "failed", ROUTE_INPUTS, "Stage23.5A inputs are missing or not the slope-theta signal route"
    if not _collector_ready(collector_summary, min_transitions):
        return "failed", ROUTE_COLLECTOR, "collector produced fewer trainable transitions than required"
    stable = _stable_rows(sweep_rows)
    if not stable:
        return "failed", ROUTE_STABILITY, "no stable Stage23.4/23.5 combo completed"
    if all(float(row.get("value_to_policy_grad_norm_ratio", math.inf)) > value_to_policy_threshold for row in stable):
        return "failed", ROUTE_VALUE_BALANCE, "value loss still dominates policy gradient across stable combos"
    if action_audit["max_mean_abs_probability_delta"] < min_probability_delta and action_audit["max_selected_viewpoint_changed_count"] <= 0:
        return "failed", ROUTE_STRENGTH, "stable combos still produce negligible slope-theta action probability change"
    if action_audit["max_selected_viewpoint_changed_count"] <= 0 and action_audit["max_selected_theta_changed_count"] <= 0:
        return "failed", ROUTE_MARGIN, "probabilities moved but did not cross viewpoint/theta selection boundary"
    if action_audit["max_final_coverage_delta"] > 0.0 and action_audit["max_coverage_auc_delta"] > 0.0:
        return "passed", ROUTE_STAGE23_7, "slope-theta update signal moved actions and improved bounded coverage/AUC"
    return "failed", ROUTE_CREDIT, "slope-theta action changed without bounded coverage/AUC improvement"


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
        if int(collector_summary.get("trainable_transition_count", 0) or 0) < min_transitions:
            reasons.append("stage23_3_trainable_transition_count_below_minimum")
        if collector_summary.get("status") != "passed":
            reasons.extend(f"stage23_3_{code}" for code in collector_summary.get("blocking_reason_codes", []))
    if route == ROUTE_STABILITY:
        reasons.append("no_stable_stage23_4_5_combo")
        for row in sweep_rows:
            if row.get("stage23_4_status") != "passed":
                reasons.extend(f"{row['combo_id']}_stage23_4_{code}" for code in row.get("stage23_4_blocking_reason_codes", []))
            if row.get("hard_risk_violation_count") or row.get("mask_violation_count") or row.get("path_planning_failure_count") or row.get("open_grid_fallback_count"):
                reasons.append(f"{row['combo_id']}_safety_counter_nonzero")
    if route == ROUTE_VALUE_BALANCE:
        reasons.append("value_loss_grad_norm_ratio_above_threshold")
    if route == ROUTE_STRENGTH:
        reasons.append("slope_theta_probability_delta_below_threshold")
    if route == ROUTE_MARGIN:
        reasons.append("slope_theta_probability_changed_without_viewpoint_change")
    if route == ROUTE_CREDIT:
        reasons.append("slope_theta_action_changed_without_coverage_auc_improvement")
    return _unique(reasons)


def _collector_ready(summary: dict[str, Any], min_transitions: int) -> bool:
    return (
        summary.get("status") == "passed"
        and summary.get("next_required_change") == stage23_3.ROUTE_STAGE23_4
        and int(summary.get("trainable_transition_count", 0) or 0) >= min_transitions
    )


def _stable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("stage23_4_status") == "passed"
        and row.get("loss_finite") is True
        and row.get("gradient_finite") is True
        and row.get("checkpoint_reload_passed") is True
        and row.get("checkpoint_boundary_passed") is True
        and row.get("release_boundary_clean") is True
        and int(row.get("hard_risk_violation_count", 0) or 0) == 0
        and int(row.get("mask_violation_count", 0) or 0) == 0
        and int(row.get("path_planning_failure_count", 0) or 0) == 0
        and int(row.get("open_grid_fallback_count", 0) or 0) == 0
        and _finite_float(row.get("final_post_update_approx_kl")) is not None
    ]


def _policy_value_gradient_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stable = _stable_rows(rows)
    ratios = [float(row.get("value_to_policy_grad_norm_ratio")) for row in stable if _finite_float(row.get("value_to_policy_grad_norm_ratio")) is not None]
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
            }
            for row in rows
        ],
    }


def _action_signal_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stable = _stable_rows(rows)
    source = stable or rows
    return {
        "schema_version": ACTION_AUDIT_SCHEMA_VERSION,
        "combo_count": len(rows),
        "stable_combo_count": len(stable),
        "max_mean_abs_probability_delta": max((_finite_or_default(row.get("mean_abs_probability_delta"), 0.0) for row in source), default=0.0),
        "max_selected_action_probability_delta_mean": max((_finite_or_default(row.get("selected_action_probability_delta_mean"), 0.0) for row in source), default=0.0),
        "max_selected_viewpoint_changed_count": max((int(row.get("selected_viewpoint_changed_count", 0) or 0) for row in source), default=0),
        "max_selected_theta_changed_count": max((int(row.get("selected_theta_changed_count", 0) or 0) for row in source), default=0),
        "max_selected_action_changed_count": max((int(row.get("selected_action_changed_count", 0) or 0) for row in source), default=0),
        "max_final_coverage_delta": max((_finite_or_default(row.get("final_coverage_delta"), 0.0) for row in source), default=0.0),
        "max_coverage_auc_delta": max((_finite_or_default(row.get("coverage_auc_delta"), 0.0) for row in source), default=0.0),
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
            }
            for row in rows
        ],
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
            -int(row.get("selected_viewpoint_changed_count", 0) or 0),
            -float(row.get("mean_abs_probability_delta", 0.0) or 0.0),
            float(row.get("value_to_policy_grad_norm_ratio", math.inf)),
        ),
    )[0]


def _write_recommended_configs(row: dict[str, Any], output_root: Path) -> None:
    if not row:
        _write_json(output_root / RECOMMENDED_23_4_CONFIG, {})
        _write_json(output_root / RECOMMENDED_23_5_CONFIG, {})
        return
    combo_root = Path(str(row["combo_root"]))
    stage23_4_config = _read_json_if_exists(combo_root / "s4.json")
    stage23_5_config = _read_json_if_exists(combo_root / "s5.json")
    _write_json(output_root / RECOMMENDED_23_4_CONFIG, stage23_4_config)
    _write_json(output_root / RECOMMENDED_23_5_CONFIG, stage23_5_config)


def _input_rejections(config: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    summary_path = Path(config["stage23_5a_root"]) / "xunce-stage23-5a-summary.json"
    if not summary_path.is_file():
        return ["missing_stage23_5a_summary"], {}
    summary = _read_json(summary_path)
    reasons: list[str] = []
    if summary.get("prior_stage23_5_required_input_short_blocker") is not True:
        reasons.append("stage23_5a_did_not_repair_required_input_short_blocker")
    if int(summary.get("repaired_high_res_roi_expansion_slice_count", 0) or 0) < 2:
        reasons.append("stage23_5a_repaired_slice_count_below_two")
    if int(summary.get("repaired_high_res_roi_expansion_scenario_count", 0) or 0) < 2:
        reasons.append("stage23_5a_repaired_scenario_count_below_two")
    if summary.get("stage23_5_action_probability_audit_is_partial_diagnostic") is not False:
        reasons.append("stage23_5a_stage23_5_still_partial_diagnostic")
    if summary.get("next_required_change") != ROUTE_SIGNAL:
        reasons.append("stage23_5a_route_not_slope_theta_signal_repair")
    if summary.get("coverage_source") != SLOPE_COVERAGE_SOURCE:
        reasons.append("stage23_5a_coverage_source_mismatch")
    if _finite_float(summary.get("max_traversable_slope_deg")) != 30.0:
        reasons.append("stage23_5a_max_traversable_slope_not_30")
    for key in ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary"):
        if summary.get(key) is True:
            reasons.append(f"stage23_5a_{key}_enabled")
    stage23_2_root = Path(config["stage23_2_root"])
    if not (stage23_2_root / "xunce-stage23-2-summary.json").is_file():
        reasons.append("missing_repaired_stage23_2_summary")
    return reasons, summary


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        reasons.append("canary_traffic_fraction_nonzero")
    return reasons


def _release_boundary_clean(stage23_4_summary: dict[str, Any], stage23_5_summary: dict[str, Any]) -> bool:
    for payload in (stage23_4_summary, stage23_5_summary):
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
    stage23_5a_root = str(_resolve_path(Path(str(payload.get("stage23_5a_root", _default_stage23_5a_root()))), repo_root))
    defaults: dict[str, Any] = {
        "stage23_5a_root": stage23_5a_root,
        "stage23_2_root": str(Path(stage23_5a_root) / "r2"),
        "stage23_3_base_config": "configs/xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke_v1.json",
        "stage23_4_base_config": "configs/xunce_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke_v1.json",
        "stage23_5_base_config": "configs/xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke_v1.json",
        "required_scenario_count": 2,
        "rollout_steps": 8,
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
        "sweep_combinations": _default_sweep_combinations(),
        "stage23_6_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    config["repo_root"] = str(repo_root)
    for key in ("stage23_5a_root", "stage23_2_root", "stage23_3_base_config", "stage23_4_base_config", "stage23_5_base_config"):
        config[key] = str(_resolve_path(Path(str(config[key])), repo_root))
    for key in ("required_scenario_count", "rollout_steps", "dynamic_max_candidates_per_step", "dynamic_proposal_pool_limit_per_step", "min_trainable_transition_count", "theta_bin_count", "theta_step_deg", "sensor_range_cells"):
        config[key] = _positive_int(config[key], key)
    for key in ("sensor_fov_deg", "clip_ratio", "entropy_coefficient", "advantage_clip_abs", "max_grad_norm", "max_abs_approx_kl", "min_mean_abs_probability_delta_for_signal", "value_to_policy_grad_ratio_dominance_threshold", "canary_traffic_fraction"):
        config[key] = _finite_number(config[key], key)
    return config


def _default_stage23_5a_root() -> str:
    return (
        "D:/CodexDownloads/lunar-path-planning/stage23_slope_obstacle_aware_theta_reward/"
        "outputs/path_feedback_batch_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage_v1"
    )


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


def _load_json_template(path: str, repo_root: Path) -> dict[str, Any]:
    return _read_json(_resolve_path(Path(path), repo_root))


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


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
            "# Stage23.6 Slope-Theta Policy Update Signal Strength Repair",
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
            "",
            "Stage23.6 is a bounded offline repair smoke. It reuses Stage23.3/23.4/23.5 and does not publish checkpoints, replace default policy, connect an executor, start canary traffic, or change network/default A*/candidate generation/reward semantics.",
            "",
        ]
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
