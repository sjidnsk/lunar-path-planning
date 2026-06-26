from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import run_xunce_stage21_1_on_policy_ppo_rollout_collector as stage21_1
    import run_xunce_stage21_2_coverage_first_ppo_reward_contract as stage21_2
    import run_xunce_stage21_3_ppo_batch_validation as stage21_3
    import run_xunce_stage21_4_tiny_ppo_update_smoke as stage21_4
    import run_xunce_stage21_5_post_update_offline_trajectory_evaluation as stage21_5
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage21_1_on_policy_ppo_rollout_collector as stage21_1
    import scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract as stage21_2
    import scripts.run_xunce_stage21_3_ppo_batch_validation as stage21_3
    import scripts.run_xunce_stage21_4_tiny_ppo_update_smoke as stage21_4
    import scripts.run_xunce_stage21_5_post_update_offline_trajectory_evaluation as stage21_5


CONFIG_SCHEMA_VERSION = "xunce-stage25-0-continuous-theta-hybrid-action-space-foundation-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage25-0-summary/v1"
AUDIT_SCHEMA_VERSION = "xunce-stage25-0-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage25-0-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage25-0-manifest/v1"

STAGE_ID = "xunce-stage25-0-continuous-theta-hybrid-action-space-foundation"
DEFAULT_CONFIG = "configs/xunce_stage25_0_continuous_theta_hybrid_action_space_foundation_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage25_continuous_theta_action_space/"
    "outputs/path_feedback_batch_xunce_stage25_0_continuous_theta_hybrid_action_space_foundation_v1"
)

ACTION_SPACE_TYPE = "hybrid_discrete_xy_continuous_theta/v1"
COVERAGE_SOURCE = "endpoint_theta_slope_obstacle_los/v1"
PATH_COST_SOURCE = "hybrid_astar_pose_path/v1"
MODEL_INFERENCE_FILE = "xunce-exploration-coverage-model-inference.jsonl"

SUMMARY_FILE = "xunce-stage25-0-summary.json"
ACTION_AUDIT_FILE = "xunce-stage25-0-action-contract-audit.json"
THETA_AUDIT_FILE = "xunce-stage25-0-theta-distribution-audit.json"
COLLECTOR_BATCH_AUDIT_FILE = "xunce-stage25-0-collector-batch-audit.json"
PPO_AUDIT_FILE = "xunce-stage25-0-ppo-logprob-kl-audit.json"
EVAL_AUDIT_FILE = "xunce-stage25-0-post-update-eval-audit.json"
ROUTING_FILE = "xunce-stage25-0-next-stage-routing.json"
REPORT_FILE = "xunce-stage25-0-report.md"
MANIFEST_FILE = "xunce-stage25-0-manifest.json"

ROUTE_INPUTS = "rerun_stage25_0_required_inputs"
ROUTE_ACTION = "repair_stage25_0_continuous_theta_action_contract"
ROUTE_BATCH = "repair_stage25_0_collector_batch_continuous_theta_contract"
ROUTE_PPO = "repair_stage25_0_continuous_theta_ppo_update_stability"
ROUTE_EVAL = "repair_stage25_0_continuous_theta_eval_binding"
ROUTE_SIGNAL = "calibrate_stage25_continuous_theta_policy_signal_strength"
ROUTE_CREDIT = "repair_stage25_continuous_theta_credit_assignment"
ROUTE_STAGE25_1 = "run_stage25_1_continuous_theta_multi_seed_ppo_pilot"
ROUTE_BOUNDARY = "resolve_stage25_0_boundary_rejections"

BOUNDARY_FIELDS = (
    "stage25_0_authorized",
    "release_or_training_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage25.0 continuous theta hybrid action-space foundation.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    summary = run_xunce_stage25_0_continuous_theta_hybrid_action_space_foundation(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage25_0_continuous_theta_hybrid_action_space_foundation(
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
    input_reasons = _input_rejections(config, repo_root)
    stage_summaries: dict[str, dict[str, Any]] = {}
    generated: dict[str, str] = {}
    if not boundary_reasons and not input_reasons:
        stage_summaries, generated = _run_chain(config, output_root, repo_root)

    s21_1 = stage_summaries.get("stage21_1", {})
    s21_2 = stage_summaries.get("stage21_2", {})
    s21_3 = stage_summaries.get("stage21_3", {})
    s21_4 = stage_summaries.get("stage21_4", {})
    s21_5 = stage_summaries.get("stage21_5", {})
    action_audit = _action_contract_audit(output_root / "s21_1")
    theta_audit = _theta_distribution_audit(output_root / "s21_1")
    collector_batch_audit = _collector_batch_audit(s21_1, s21_2, s21_3)
    ppo_audit = _ppo_logprob_kl_audit(output_root / "s21_4", output_root / "s21_3", s21_3, s21_4)
    eval_audit = _post_update_eval_audit(output_root / "s21_5")

    status, route, route_reason = _route(
        boundary_reasons,
        input_reasons,
        stage_summaries,
        action_audit,
        collector_batch_audit,
        ppo_audit,
        eval_audit,
        min_probability_delta=float(config["min_mean_abs_probability_delta_for_signal"]),
    )
    blocking = _unique(
        boundary_reasons
        + input_reasons
        + _route_blocking_reasons(route, action_audit, collector_batch_audit, ppo_audit, eval_audit)
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": blocking,
        "stage24_5a_root": config["stage24_5a_root"],
        "stage21_1_status": s21_1.get("status"),
        "stage21_2_status": s21_2.get("status"),
        "stage21_3_status": s21_3.get("status"),
        "stage21_4_status": s21_4.get("status"),
        "stage21_5_status": s21_5.get("status"),
        "action_space_type": ACTION_SPACE_TYPE,
        "coverage_source": COVERAGE_SOURCE,
        "path_cost_source": PATH_COST_SOURCE,
        "max_traversable_slope_deg": 30.0,
        "trainable_transition_count": collector_batch_audit["stage21_3_trainable_transition_count"],
        "continuous_theta_transition_count": action_audit["continuous_theta_transition_count"],
        "continuous_theta_batch_missing_count": collector_batch_audit["continuous_theta_action_contract_missing_count"],
        "old_new_logprob_recomputable": ppo_audit["continuous_theta_logprob_audit_passed"],
        "checkpoint_reload_passed": bool(s21_4.get("checkpoint_reload_passed")),
        "strong_state_join_available_count": eval_audit["strong_state_join_available_count"],
        "mean_abs_probability_delta": eval_audit["mean_abs_probability_delta"],
        "selected_theta_changed_count": eval_audit["selected_theta_changed_count"],
        "selected_action_changed_count": eval_audit["selected_action_changed_count"],
        "final_coverage_delta": eval_audit["final_coverage_delta"],
        "coverage_auc_delta": eval_audit["coverage_auc_delta"],
        "path_cost_delta": eval_audit["path_cost_delta"],
        "stage25_0_authorized": False,
        "release_or_training_authorized": False,
        "runs_new_ppo_update": bool(s21_4.get("status") == "passed"),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "generated_configs": generated,
    }
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "stage25_0_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "summary": str(output_root / SUMMARY_FILE),
        "action_contract_audit": str(output_root / ACTION_AUDIT_FILE),
        "theta_distribution_audit": str(output_root / THETA_AUDIT_FILE),
        "collector_batch_audit": str(output_root / COLLECTOR_BATCH_AUDIT_FILE),
        "ppo_logprob_kl_audit": str(output_root / PPO_AUDIT_FILE),
        "post_update_eval_audit": str(output_root / EVAL_AUDIT_FILE),
        "routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
        "substage_roots": {
            "s21_1": str(output_root / "s21_1"),
            "s21_2": str(output_root / "s21_2"),
            "s21_3": str(output_root / "s21_3"),
            "s21_4": str(output_root / "s21_4"),
            "s21_5": str(output_root / "s21_5"),
        },
        "generated_configs": generated,
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / ACTION_AUDIT_FILE, action_audit)
    _write_json(output_root / THETA_AUDIT_FILE, theta_audit)
    _write_json(output_root / COLLECTOR_BATCH_AUDIT_FILE, collector_batch_audit)
    _write_json(output_root / PPO_AUDIT_FILE, ppo_audit)
    _write_json(output_root / EVAL_AUDIT_FILE, eval_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    _write_json(output_root / MANIFEST_FILE, manifest)
    return summary


def _run_chain(config: dict[str, Any], output_root: Path, repo_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    configs_dir = output_root / "generated_configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    high_fidelity_path = configs_dir / "xunce-stage25-0-high-fidelity-config.json"
    stage21_1_config = configs_dir / "xunce-stage25-0-stage21-1-config.json"
    stage21_2_config = configs_dir / "xunce-stage25-0-stage21-2-config.json"
    stage21_3_config = configs_dir / "xunce-stage25-0-stage21-3-config.json"
    stage21_4_config = configs_dir / "xunce-stage25-0-stage21-4-config.json"
    stage21_5_config = configs_dir / "xunce-stage25-0-stage21-5-config.json"

    high_fidelity = _stage25_high_fidelity_config(config, repo_root, output_root)
    _write_json(high_fidelity_path, high_fidelity)
    s21_1_cfg = _stage21_1_config(config, repo_root, high_fidelity_path, output_root)
    _write_json(stage21_1_config, s21_1_cfg)
    s21_1 = stage21_1.run_xunce_stage21_1_on_policy_ppo_rollout_collector(
        config_path=stage21_1_config,
        output_root=output_root / "s21_1",
        repo_root=repo_root,
    )
    s21_2_cfg = _stage21_2_config(config, repo_root, output_root)
    _write_json(stage21_2_config, s21_2_cfg)
    s21_2 = stage21_2.run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=stage21_2_config,
        output_root=output_root / "s21_2",
        repo_root=repo_root,
    )
    s21_3_cfg = _stage21_3_config(config, repo_root, output_root)
    _write_json(stage21_3_config, s21_3_cfg)
    s21_3 = stage21_3.run_xunce_stage21_3_ppo_batch_validation(
        config_path=stage21_3_config,
        output_root=output_root / "s21_3",
        repo_root=repo_root,
    )
    s21_4_cfg = _stage21_4_config(config, repo_root, high_fidelity_path, output_root)
    _write_json(stage21_4_config, s21_4_cfg)
    s21_4 = stage21_4.run_xunce_stage21_4_tiny_ppo_update_smoke(
        config_path=stage21_4_config,
        output_root=output_root / "s21_4",
        repo_root=repo_root,
    )
    s21_5_cfg = _stage21_5_config(config, repo_root, high_fidelity_path, output_root)
    _write_json(stage21_5_config, s21_5_cfg)
    s21_5 = stage21_5.run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=stage21_5_config,
        output_root=output_root / "s21_5",
        repo_root=repo_root,
    )
    return (
        {"stage21_1": s21_1, "stage21_2": s21_2, "stage21_3": s21_3, "stage21_4": s21_4, "stage21_5": s21_5},
        {
            "high_fidelity_config": str(high_fidelity_path),
            "stage21_1_config": str(stage21_1_config),
            "stage21_2_config": str(stage21_2_config),
            "stage21_3_config": str(stage21_3_config),
            "stage21_4_config": str(stage21_4_config),
            "stage21_5_config": str(stage21_5_config),
        },
    )


def _stage25_high_fidelity_config(config: dict[str, Any], repo_root: Path, output_root: Path) -> dict[str, Any]:
    cfg = _read_json(_resolve_path(Path(config["high_fidelity_config"]), repo_root))
    cfg.update(
        {
            "continuous_theta_action_space_enabled": True,
            "action_space_type": ACTION_SPACE_TYPE,
            "theta_aware_candidate_viewpoints_enabled": False,
            "slope_obstacle_aware_theta_reward_enabled": True,
            "hybrid_astar_pose_path_cost_enabled": True,
            "obstacle_occlusion_enabled": True,
            "coverage_source": COVERAGE_SOURCE,
            "path_cost_source": PATH_COST_SOURCE,
            "max_traversable_slope_deg": 30.0,
            "continuous_theta_head_init_seed": int(config["sampling_seed"]),
            "initial_theta_deg": float(config["initial_theta_deg"]),
            "sensor_model_id": str(config["sensor_model_id"]),
            "sensor_fov_deg": float(config["sensor_fov_deg"]),
            "sensor_range_cells": int(config["sensor_range_cells"]),
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "source_roi_expansion_root": config.get("source_roi_expansion_root") or _stage24_5a_high_res_root(config),
            "dynamic_validation_work_root": str(output_root / "_xunce_dynamic_validation_work"),
            "emit_candidate_metric_audit": True,
            "default_astar_replaced": False,
            "hybrid_astar_ackermann_feasible_claimed": False,
        }
    )
    for field in _hybrid_astar_fields():
        if field in config:
            cfg[field] = config[field]
    return cfg


def _stage21_1_config(config: dict[str, Any], repo_root: Path, high_fidelity_path: Path, output_root: Path) -> dict[str, Any]:
    cfg = _read_json(_resolve_path(Path(config["stage21_1_base_config"]), repo_root))
    cfg.update(_common_action_fields(config))
    cfg.update(
        {
            "high_fidelity_config": str(high_fidelity_path),
            "canonical_reward_profile": config["canonical_reward_profile"],
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "dynamic_validation_work_root": str(output_root / "_stage21_1_dynamic_validation_work"),
            "sampling_seed": int(config["sampling_seed"]),
            "sampling_temperature": float(config["sampling_temperature"]),
            "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
            "stage21_1_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _stage21_2_config(config: dict[str, Any], repo_root: Path, output_root: Path) -> dict[str, Any]:
    cfg = _read_json(_resolve_path(Path(config["stage21_2_base_config"]), repo_root))
    cfg.update(
        {
            "stage21_1_collector_root": str(output_root / "s21_1"),
            "coverage_first_reward_profile": config["coverage_first_reward_profile"],
            "require_theta_aware_reward_contract": True,
            "slope_obstacle_aware_theta_reward_enabled": True,
            "require_hybrid_astar_path_cost_contract": True,
            "stage21_2_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _stage21_3_config(config: dict[str, Any], repo_root: Path, output_root: Path) -> dict[str, Any]:
    cfg = _read_json(_resolve_path(Path(config["stage21_3_base_config"]), repo_root))
    cfg.update(
        {
            "stage21_1_collector_root": str(output_root / "s21_1"),
            "stage21_2_reward_contract_root": str(output_root / "s21_2"),
            "require_continuous_theta_action_contract": True,
            "require_theta_aware_reward_contract": True,
            "require_slope_obstacle_aware_theta_reward_contract": True,
            "require_hybrid_astar_path_cost_contract": True,
            "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
            "stage21_3_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _stage21_4_config(config: dict[str, Any], repo_root: Path, high_fidelity_path: Path, output_root: Path) -> dict[str, Any]:
    cfg = _read_json(_resolve_path(Path(config["stage21_4_base_config"]), repo_root))
    cfg.update(
        {
            "stage21_3_ppo_batch_validation_root": str(output_root / "s21_3"),
            "xunce_candidate_checkpoint": config["xunce_candidate_checkpoint"],
            "high_fidelity_config": str(high_fidelity_path),
            "continuous_theta_head_init_seed": int(config["sampling_seed"]),
            "epochs": int(config["epochs"]),
            "learning_rate": float(config["learning_rate"]),
            "clip_ratio": float(config["clip_ratio"]),
            "policy_loss_coefficient": float(config["policy_loss_coefficient"]),
            "value_loss_coefficient": float(config["value_loss_coefficient"]),
            "entropy_coefficient": float(config["entropy_coefficient"]),
            "advantage_clip_abs": float(config["advantage_clip_abs"]),
            "normalize_minibatch_advantages": bool(config["normalize_minibatch_advantages"]),
            "loss_scale": float(config["loss_scale"]),
            "max_grad_norm": float(config["max_grad_norm"]),
            "max_abs_approx_kl": float(config["max_abs_approx_kl"]),
            "offline_ppo_update_smoke_authorized": True,
            "stage21_4_authorized": False,
            "runs_new_ppo_update": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _stage21_5_config(config: dict[str, Any], repo_root: Path, high_fidelity_path: Path, output_root: Path) -> dict[str, Any]:
    cfg = _read_json(_resolve_path(Path(config["stage21_5_base_config"]), repo_root))
    cfg.update(_common_action_fields(config))
    cfg.update(
        {
            "stage21_4_tiny_ppo_update_smoke_root": str(output_root / "s21_4"),
            "high_fidelity_config": str(high_fidelity_path),
            "execute_high_fidelity_evaluations": True,
            "pre_ppo_evaluation_root": str(output_root / "s21_5" / "pre"),
            "post_ppo_evaluation_root": str(output_root / "s21_5" / "post"),
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "stage21_5_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    for field in _hybrid_astar_fields():
        if field in config:
            cfg[field] = config[field]
    return cfg


def _common_action_fields(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "continuous_theta_action_space_enabled": True,
        "action_space_type": ACTION_SPACE_TYPE,
        "theta_aware_candidate_viewpoints_enabled": False,
        "slope_obstacle_aware_theta_reward_enabled": True,
        "hybrid_astar_pose_path_cost_enabled": True,
        "obstacle_occlusion_enabled": True,
        "coverage_source": COVERAGE_SOURCE,
        "path_cost_source": PATH_COST_SOURCE,
        "max_traversable_slope_deg": 30.0,
        "continuous_theta_head_init_seed": int(config["sampling_seed"]),
        "initial_theta_deg": float(config["initial_theta_deg"]),
        "sensor_model_id": str(config["sensor_model_id"]),
        "sensor_fov_deg": float(config["sensor_fov_deg"]),
        "sensor_range_cells": int(config["sensor_range_cells"]),
        "default_astar_replaced": False,
        "hybrid_astar_ackermann_feasible_claimed": False,
    }


def _action_contract_audit(stage21_1_root: Path) -> dict[str, Any]:
    rows = _read_jsonl_if_exists(stage21_1_root / stage21_1.TRANSITIONS_FILE)
    continuous = [row for row in rows if row.get("action_space_type") == ACTION_SPACE_TYPE]
    missing = 0
    discrete_theta_rows = 0
    for row in rows:
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        if row.get("action_space_type") != ACTION_SPACE_TYPE:
            missing += 1
        if info.get("candidate_viewpoints") and not row.get("selected_theta_rad"):
            discrete_theta_rows += 1
        if not (
            _finite(row.get("selected_theta_rad")) is not None
            and _finite(row.get("old_point_log_prob")) is not None
            and _finite(row.get("old_theta_log_prob")) is not None
            and str(row.get("base_candidate_set_hash") or "").strip()
            and str(row.get("action_sample_hash") or "").strip()
        ):
            missing += 1
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "transition_count": len(rows),
        "continuous_theta_transition_count": len(continuous),
        "continuous_theta_action_contract_missing_count": missing,
        "discrete_theta_viewpoint_action_row_count": discrete_theta_rows,
        "action_space_type": ACTION_SPACE_TYPE,
    }


def _theta_distribution_audit(stage21_1_root: Path) -> dict[str, Any]:
    rows = _read_jsonl_if_exists(stage21_1_root / stage21_1.SAMPLING_AUDIT_FILE)
    kappa_values: list[float] = []
    mu_values: list[float] = []
    finite_theta = 0
    for row in rows:
        if _finite(row.get("selected_theta_rad")) is not None:
            finite_theta += 1
        kappas = row.get("theta_kappa")
        if isinstance(kappas, list):
            kappa_values.extend(float(v) for v in kappas if _finite(v) is not None)
        mus = row.get("theta_mu_rad")
        if isinstance(mus, list):
            mu_values.extend(float(v) for v in mus if _finite(v) is not None)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "sampling_audit_row_count": len(rows),
        "finite_selected_theta_count": finite_theta,
        "theta_mu_count": len(mu_values),
        "theta_kappa_count": len(kappa_values),
        "theta_kappa_min": min(kappa_values) if kappa_values else None,
        "theta_kappa_max": max(kappa_values) if kappa_values else None,
        "theta_distribution": "von_mises",
    }


def _collector_batch_audit(s21_1: dict[str, Any], s21_2: dict[str, Any], s21_3: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "stage21_1_status": s21_1.get("status"),
        "stage21_2_status": s21_2.get("status"),
        "stage21_3_status": s21_3.get("status"),
        "stage21_1_trainable_transition_count": _int_value(s21_1.get("trainable_transition_count")),
        "stage21_2_reward_row_count": _int_value(s21_2.get("reward_evaluation_row_count")),
        "stage21_3_trainable_transition_count": _int_value(s21_3.get("trainable_transition_count")),
        "continuous_theta_action_contract_missing_count": _int_value(
            s21_3.get("continuous_theta_action_contract_missing_count")
        ),
        "slope_obstacle_reward_contract_missing_count": _int_value(
            s21_3.get("slope_obstacle_reward_contract_missing_count")
        ),
        "hybrid_astar_path_cost_contract_missing_count": _int_value(
            s21_3.get("hybrid_astar_path_cost_contract_missing_count")
        ),
        "coverage_source": COVERAGE_SOURCE,
        "path_cost_source": PATH_COST_SOURCE,
    }


def _ppo_logprob_kl_audit(
    stage21_4_root: Path,
    stage21_3_root: Path,
    s21_3: dict[str, Any],
    s21_4: dict[str, Any],
) -> dict[str, Any]:
    loss_rows = _read_jsonl_if_exists(stage21_4_root / stage21_4.LOSS_AUDIT_FILE)
    train_rows = _read_jsonl_if_exists(stage21_3_root / stage21_3.BATCH_FILE)
    train_count = sum(1 for row in train_rows if row.get("stage21_3_split") == "train")
    continuous_count = sum(1 for row in train_rows if row.get("action_space_type") == ACTION_SPACE_TYPE)
    continuous_missing = _int_value(s21_3.get("continuous_theta_action_contract_missing_count"))
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "stage21_4_status": s21_4.get("status"),
        "train_transition_count": _int_value(s21_4.get("train_transition_count")),
        "stage21_3_train_row_count": train_count,
        "stage21_3_continuous_theta_row_count": continuous_count,
        "stage21_3_continuous_theta_contract_missing_count": continuous_missing,
        "final_post_update_approx_kl": _finite(s21_4.get("final_post_update_approx_kl")),
        "pre_clip_grad_norm": _finite(s21_4.get("pre_clip_grad_norm")),
        "post_clip_grad_norm": _finite(s21_4.get("post_clip_grad_norm")),
        "parameter_delta_l2": _finite(s21_4.get("parameter_delta_l2")),
        "checkpoint_reload_passed": bool(s21_4.get("checkpoint_reload_passed")),
        "loss_audit_row_count": len(loss_rows),
        "continuous_theta_logprob_audit_passed": bool(
            s21_4.get("status") == "passed"
            and loss_rows
            and train_count > 0
            and continuous_count == len(train_rows)
            and continuous_missing == 0
        ),
    }


def _post_update_eval_audit(stage21_5_root: Path) -> dict[str, Any]:
    summary = _read_json_if_exists(stage21_5_root / stage21_5.SUMMARY_FILE)
    pre_root = Path(str(summary.get("pre_evaluation_root") or stage21_5_root / "pre_ppo_xunce"))
    post_root = Path(str(summary.get("post_evaluation_root") or stage21_5_root / "post_ppo_xunce"))
    pre_rows_all = _read_jsonl_if_exists(pre_root / MODEL_INFERENCE_FILE)
    post_rows_all = _read_jsonl_if_exists(post_root / MODEL_INFERENCE_FILE)
    pre_rows = [row for row in pre_rows_all if row.get("policy") == "xunce"]
    post_rows = [row for row in post_rows_all if row.get("policy") == "xunce"]
    pre_index = {_strong_key(row): row for row in pre_rows if _strong_key(row) is not None}
    post_index = {_strong_key(row): row for row in post_rows if _strong_key(row) is not None}
    joined = 0
    theta_changed = action_changed = 0
    prob_delta_sum = prob_delta_count = 0
    field_missing = 0
    for key, pre in pre_index.items():
        post = post_index.get(key)
        if post is None:
            continue
        joined += 1
        for row in (pre, post):
            if not _eval_row_has_stage25_contract(row):
                field_missing += 1
        if _angle_delta(_finite(pre.get("selected_theta_rad")), _finite(post.get("selected_theta_rad"))) > 1.0e-6:
            theta_changed += 1
        if pre.get("selected_action_index") != post.get("selected_action_index"):
            action_changed += 1
        deltas = _probability_deltas(pre.get("action_probs"), post.get("action_probs"))
        prob_delta_sum += sum(deltas)
        prob_delta_count += len(deltas)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "stage21_5_status": summary.get("status"),
        "pre_inference_row_count": len(pre_rows_all),
        "post_inference_row_count": len(post_rows_all),
        "pre_xunce_inference_row_count": len(pre_rows),
        "post_xunce_inference_row_count": len(post_rows),
        "strong_state_join_available_count": joined,
        "continuous_theta_eval_required_field_missing_count": field_missing,
        "selected_theta_changed_count": theta_changed,
        "selected_action_changed_count": action_changed,
        "mean_abs_probability_delta": (prob_delta_sum / prob_delta_count) if prob_delta_count else 0.0,
        "final_coverage_delta": _finite(summary.get("final_coverage_delta")) or 0.0,
        "coverage_auc_delta": _finite(summary.get("coverage_auc_delta")) or 0.0,
        "path_cost_delta": _finite(summary.get("path_cost_delta")) or 0.0,
        "hard_risk_violation_count": _int_value(summary.get("hard_risk_violation_count")),
        "mask_violation_count": _int_value(summary.get("mask_violation_count")),
        "path_planning_failure_count": _int_value(summary.get("path_planning_failure_count")),
        "open_grid_fallback_count": _int_value(summary.get("open_grid_fallback_count")),
    }


def _route(
    boundary_reasons: list[str],
    input_reasons: list[str],
    summaries: dict[str, dict[str, Any]],
    action_audit: dict[str, Any],
    batch_audit: dict[str, Any],
    ppo_audit: dict[str, Any],
    eval_audit: dict[str, Any],
    *,
    min_probability_delta: float,
) -> tuple[str, str, str]:
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "boundary fields are open"
    if input_reasons:
        return "failed", ROUTE_INPUTS, "required Stage24.5A lineage or config inputs are missing"
    for name in ("stage21_1", "stage21_2", "stage21_3", "stage21_4"):
        if summaries.get(name, {}).get("status") != "passed":
            return "failed", _route_for_failed_substage(name), f"{name} did not pass"
    if summaries.get("stage21_5", {}).get("status") != "passed" and (
        eval_audit["strong_state_join_available_count"] <= 0
        or eval_audit["continuous_theta_eval_required_field_missing_count"] > 0
    ):
        return "failed", ROUTE_EVAL, "stage21_5 did not produce complete continuous theta eval binding"
    if action_audit["continuous_theta_action_contract_missing_count"] > 0:
        return "failed", ROUTE_ACTION, "continuous theta transition contract is incomplete"
    if batch_audit["continuous_theta_action_contract_missing_count"] > 0:
        return "failed", ROUTE_BATCH, "Stage21.3 rejected or missed continuous theta rows"
    if not ppo_audit["continuous_theta_logprob_audit_passed"]:
        return "failed", ROUTE_PPO, "continuous theta PPO logprob audit did not pass"
    if eval_audit["strong_state_join_available_count"] <= 0 or eval_audit["continuous_theta_eval_required_field_missing_count"] > 0:
        return "failed", ROUTE_EVAL, "continuous theta eval binding is incomplete"
    if (
        eval_audit["final_coverage_delta"] > 0.0
        and eval_audit["coverage_auc_delta"] > 0.0
        and eval_audit["path_cost_delta"] <= 0.0
        and _safety_clean(eval_audit)
        and eval_audit["selected_theta_changed_count"] > 0
    ):
        return "passed", ROUTE_STAGE25_1, "coverage/AUC improved with continuous-theta action movement"
    if eval_audit["mean_abs_probability_delta"] < min_probability_delta and eval_audit["selected_theta_changed_count"] == 0:
        return "failed", ROUTE_SIGNAL, "continuous theta policy signal is still too small"
    return "failed", ROUTE_CREDIT, "theta changed or probability moved but coverage/path-cost did not improve"


def _route_for_failed_substage(name: str) -> str:
    return {
        "stage21_1": ROUTE_ACTION,
        "stage21_2": ROUTE_ACTION,
        "stage21_3": ROUTE_BATCH,
        "stage21_4": ROUTE_PPO,
        "stage21_5": ROUTE_EVAL,
    }.get(name, ROUTE_INPUTS)


def _route_blocking_reasons(
    route: str,
    action_audit: dict[str, Any],
    batch_audit: dict[str, Any],
    ppo_audit: dict[str, Any],
    eval_audit: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if route == ROUTE_ACTION:
        if action_audit["continuous_theta_action_contract_missing_count"] > 0:
            reasons.append("continuous_theta_action_contract_missing")
    if route == ROUTE_BATCH:
        if batch_audit["continuous_theta_action_contract_missing_count"] > 0:
            reasons.append("continuous_theta_batch_contract_missing")
    if route == ROUTE_PPO and not ppo_audit["continuous_theta_logprob_audit_passed"]:
        reasons.append("continuous_theta_ppo_logprob_audit_failed")
    if route == ROUTE_EVAL:
        if eval_audit["strong_state_join_available_count"] <= 0:
            reasons.append("continuous_theta_eval_strong_join_missing")
        if eval_audit["continuous_theta_eval_required_field_missing_count"] > 0:
            reasons.append("continuous_theta_eval_required_fields_missing")
    return _unique(reasons)


def _eval_row_has_stage25_contract(row: dict[str, Any]) -> bool:
    if row.get("action_space_type") != ACTION_SPACE_TYPE:
        return False
    required = (
        "selected_theta_rad",
        "selected_theta_deg",
        "theta_mu_rad",
        "theta_kappa",
        "candidate_viewpoint",
        "candidate_theta_deg",
        "hybrid_astar_path_cost",
        "hybrid_astar_pose_path_hash",
        "legacy_grid_astar_path_cost",
        "hybrid_vs_grid_path_cost_delta",
        "slope_obstacle_source_hash",
        "platform_contract_hash",
    )
    if any(row.get(field) in (None, "") for field in required):
        return False
    return (
        row.get("coverage_source") == COVERAGE_SOURCE
        and row.get("path_cost_source") == PATH_COST_SOURCE
        and row.get("hybrid_astar_trajectory_kind") == "hybrid_astar_pose_path"
        and row.get("point_grid_path_cost_fallback_used") is False
        and row.get("default_astar_replaced") is False
        and row.get("hybrid_astar_ackermann_feasible_claimed") is False
        and _finite(row.get("max_traversable_slope_deg")) == 30.0
        and _finite(row.get("hybrid_astar_path_cost")) is not None
    )


def _input_rejections(config: dict[str, Any], repo_root: Path) -> list[str]:
    reasons: list[str] = []
    stage24_5a_root = Path(config["stage24_5a_root"])
    stage24_5a_summary = _read_json_if_exists(stage24_5a_root / "xunce-stage24-5a-summary.json")
    if stage24_5a_summary.get("status") != "passed":
        reasons.append("stage24_5a_status_not_passed")
    for field in (
        "stage21_1_base_config",
        "stage21_2_base_config",
        "stage21_3_base_config",
        "stage21_4_base_config",
        "stage21_5_base_config",
        "high_fidelity_config",
        "canonical_reward_profile",
        "coverage_first_reward_profile",
    ):
        if not _resolve_path(Path(config[field]), repo_root).is_file():
            reasons.append(f"{field}_missing")
    return _unique(reasons)


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0)) > 0.0:
        reasons.append("canary_traffic_fraction")
    return _unique(reasons)


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(path, repo_root))
    payload.setdefault("schema_version", CONFIG_SCHEMA_VERSION)
    payload.setdefault("stage_id", STAGE_ID)
    payload.setdefault("stage24_5a_root", "D:/CodexDownloads/lunar-path-planning/stage24_hybrid_astar_pose_planner/outputs/path_feedback_batch_xunce_stage24_5a_repair_hybrid_path_inference_binding_v1")
    payload.setdefault("stage21_1_base_config", "configs/xunce_stage21_1_on_policy_ppo_rollout_collector_v1.json")
    payload.setdefault("stage21_2_base_config", "configs/xunce_stage21_2_coverage_first_ppo_reward_contract_v1.json")
    payload.setdefault("stage21_3_base_config", "configs/xunce_stage21_3_ppo_batch_validation_v1.json")
    payload.setdefault("stage21_4_base_config", "configs/xunce_stage21_4_tiny_ppo_update_smoke_v1.json")
    payload.setdefault("stage21_5_base_config", "configs/xunce_stage21_5_post_update_offline_trajectory_evaluation_v1.json")
    payload.setdefault("high_fidelity_config", "configs/xunce_high_fidelity_exploration_coverage_comparison_stage18_9_strict_v3.json")
    payload.setdefault("canonical_reward_profile", "configs/xunce_canonical_reward_guard_profile_v3.json")
    payload.setdefault("coverage_first_reward_profile", "configs/xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json")
    payload.setdefault("xunce_candidate_checkpoint", "outputs/path_feedback_batch_xunce_sandbox_candidate_preflight_v1/sandbox_package/xunce-controlled-training-candidate.pt")
    payload.setdefault("required_scenario_count", 2)
    payload.setdefault("rollout_steps", 4)
    payload.setdefault("dynamic_max_candidates_per_step", 36)
    payload.setdefault("dynamic_proposal_pool_limit_per_step", 288)
    payload.setdefault("sampling_seed", 2101)
    payload.setdefault("sampling_temperature", 1.0)
    payload.setdefault("initial_theta_deg", 0.0)
    payload.setdefault("sensor_model_id", "endpoint-theta-slope-obstacle-los/v1")
    payload.setdefault("sensor_fov_deg", 90.0)
    payload.setdefault("sensor_range_cells", 3)
    payload.setdefault("min_trainable_transition_count", 1)
    payload.setdefault("epochs", 1)
    payload.setdefault("learning_rate", 2.0e-6)
    payload.setdefault("clip_ratio", 0.2)
    payload.setdefault("policy_loss_coefficient", 1.0)
    payload.setdefault("value_loss_coefficient", 0.1)
    payload.setdefault("entropy_coefficient", 0.01)
    payload.setdefault("advantage_clip_abs", 5.0)
    payload.setdefault("normalize_minibatch_advantages", True)
    payload.setdefault("loss_scale", 0.25)
    payload.setdefault("max_grad_norm", 1.0)
    payload.setdefault("max_abs_approx_kl", 0.5)
    payload.setdefault("min_mean_abs_probability_delta_for_signal", 1.0e-6)
    payload.setdefault("hybrid_astar_theta_bin_count", 72)
    payload.setdefault("hybrid_astar_goal_position_tolerance_m", 2.0)
    payload.setdefault("hybrid_astar_goal_theta_tolerance_deg", 5.0)
    payload.setdefault("hybrid_astar_max_iterations", 100000)
    payload.setdefault("hybrid_astar_primitive_duration_s", 1.5)
    payload.setdefault("hybrid_astar_integration_dt_s", 0.25)
    payload.setdefault("hybrid_astar_max_speed_mps", 3.0)
    payload.setdefault("hybrid_astar_max_angular_speed_degps", 45.0)
    payload.setdefault("hybrid_astar_rotation_cost_weight", 0.2)
    payload.setdefault("hybrid_astar_reverse_penalty_weight", 0.5)
    payload.setdefault("hybrid_astar_turn_penalty_weight", 0.05)
    for field in BOUNDARY_FIELDS:
        payload.setdefault(field, False)
    payload.setdefault("runs_new_ppo_update", False)
    payload.setdefault("canary_traffic_fraction", 0.0)
    return payload


def _stage24_5a_high_res_root(config: dict[str, Any]) -> str:
    root = Path(config["stage24_5a_root"])
    summary = _read_json_if_exists(root / "xunce-stage24-5a-rerun-stage24-5-summary.json")
    return str(summary.get("stage23_2b_high_res_root") or summary.get("stage21_1_actual_source_roi_expansion_root") or "")


def _hybrid_astar_fields() -> tuple[str, ...]:
    return (
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


def _safety_clean(eval_audit: dict[str, Any]) -> bool:
    return all(
        _int_value(eval_audit.get(field)) == 0
        for field in (
            "hard_risk_violation_count",
            "mask_violation_count",
            "path_planning_failure_count",
            "open_grid_fallback_count",
        )
    )


def _strong_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    fields = ("scenario_id", "step_index", "current_cell", "covered_cells_hash", "candidate_set_hash")
    if any(row.get(field) is None for field in fields):
        return None
    return tuple(json.dumps(row.get(field), sort_keys=True, separators=(",", ":")) for field in fields)


def _probability_deltas(pre: Any, post: Any) -> list[float]:
    if not isinstance(pre, list) or not isinstance(post, list):
        return []
    deltas = []
    for lhs, rhs in zip(pre, post):
        left = _finite(lhs)
        right = _finite(rhs)
        if left is not None and right is not None:
            deltas.append(abs(right - left))
    return deltas


def _angle_delta(lhs: float | None, rhs: float | None) -> float:
    if lhs is None or rhs is None:
        return 0.0
    return abs(((rhs - lhs + math.pi) % (2.0 * math.pi)) - math.pi)


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage25.0 Continuous Theta Hybrid Action Space Foundation",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- action_space_type: `{summary['action_space_type']}`",
            f"- trainable_transition_count: `{summary['trainable_transition_count']}`",
            f"- continuous_theta_transition_count: `{summary['continuous_theta_transition_count']}`",
            f"- strong_state_join_available_count: `{summary['strong_state_join_available_count']}`",
            f"- mean_abs_probability_delta: `{summary['mean_abs_probability_delta']}`",
            f"- selected_theta_changed_count: `{summary['selected_theta_changed_count']}`",
            "",
            "This is an offline smoke foundation. It does not publish checkpoints, replace the default policy, connect a real executor, or start canary traffic.",
            "",
        ]
    )


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _int_value(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _unique(values: list[str]) -> list[str]:
    return sorted({str(value) for value in values if value})


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
