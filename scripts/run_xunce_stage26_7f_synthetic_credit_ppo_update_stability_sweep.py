from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as stage26_2
    import run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as stage26_3
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as stage26_2
    import scripts.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as stage26_3


STAGE_ID = "xunce-stage26-7f-synthetic-credit-ppo-update-stability-sweep"
CONFIG_SCHEMA_VERSION = "xunce-stage26-7f-synthetic-credit-ppo-update-stability-sweep-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-7f-summary/v1"
SWEEP_ROW_SCHEMA_VERSION = "xunce-stage26-7f-update-sweep-result/v1"
KL_AUDIT_SCHEMA_VERSION = "xunce-stage26-7f-kl-stability-audit/v1"
CHECKPOINT_AUDIT_SCHEMA_VERSION = "xunce-stage26-7f-checkpoint-audit/v1"
POST_EVAL_AUDIT_SCHEMA_VERSION = "xunce-stage26-7f-post-update-eval-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-7f-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-7f-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep_v1"
)
DEFAULT_STAGE26_7D_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_7d_repair_synthetic_credit_sampler_continuous_theta_reachability_v1"
)

SUMMARY_FILE = "xunce-stage26-7f-summary.json"
SWEEP_FILE = "xunce-stage26-7f-update-sweep-results.jsonl"
KL_AUDIT_FILE = "xunce-stage26-7f-kl-stability-audit.json"
CHECKPOINT_AUDIT_FILE = "xunce-stage26-7f-checkpoint-audit.json"
POST_EVAL_AUDIT_FILE = "xunce-stage26-7f-post-update-eval-audit.json"
RECOMMENDED_STAGE26_2_CONFIG_FILE = "xunce-stage26-7f-recommended-stage26-2-config.json"
RECOMMENDED_STAGE26_3_CONFIG_FILE = "xunce-stage26-7f-recommended-stage26-3-config.json"
ROUTING_FILE = "xunce-stage26-7f-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-7f-report.md"
MANIFEST_FILE = "xunce-stage26-7f-manifest.json"

ROUTE_INPUTS = "rerun_stage26_7f_required_inputs"
ROUTE_REDUCE = "reduce_stage26_7_credit_update_strength"
ROUTE_NUMERICS = "repair_stage26_7_credit_ppo_update_numerics"
ROUTE_CHECKPOINT = "repair_stage26_7_credit_checkpoint_boundary"
ROUTE_EVAL_BINDING = "repair_stage26_7_credit_post_update_eval_binding"
ROUTE_KL_BASELINE = "repair_stage26_7_behavior_policy_kl_baseline"
ROUTE_MARGIN = "calibrate_stage26_synthetic_discrete_margin_crossing_after_credit"
ROUTE_CREDIT = "repair_stage26_synthetic_credit_assignment"
ROUTE_MULTI_SEED = "run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot"
ROUTE_BOUNDARY = "resolve_stage26_7f_boundary_rejections"

ROUTE_FROM_STAGE26_7D = "repair_stage26_7_credit_ppo_update_stability"
BOUNDARY_FIELDS = ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary")
MAX_ALLOWED_KL_LIMIT = 1.5
REQUIRED_STAGE26_7D_TRAINABLE_TRANSITION_COUNT = 36
REQUIRED_STAGE26_7D_CREDIT_TARGET_SELECTED_COUNT = 36


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.7F synthetic credit PPO update stability sweep.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage26_7d_summary = _read_json_if_exists(Path(config["stage26_7d_root"]) / "xunce-stage26-7d-summary.json")
    boundary_rejections = _boundary_rejections(config)
    primary_stage26_1_root = _primary_stage26_1_root(config, stage26_7d_summary)
    input_rejections = _input_rejections(stage26_7d_summary, primary_stage26_1_root)

    sweep_rows: list[dict[str, Any]] = []
    combo_details: list[dict[str, Any]] = []
    if not boundary_rejections and not input_rejections and bool(config["run_stage26_chain"]):
        for combo in config["update_sweep"]:
            detail = _run_combo(config=config, combo=combo, primary_stage26_1_root=primary_stage26_1_root, output_root=output_root, repo_root=repo_root)
            combo_details.append(detail)
            sweep_rows.append(detail["sweep_row"])

    stable_rows = [row for row in sweep_rows if _combo_is_stable(row, config["max_abs_approx_kl"])]
    best_combo = _best_stable_combo(combo_details, config["max_abs_approx_kl"])
    kl_audit = _kl_stability_audit(sweep_rows, float(config["max_abs_approx_kl"]))
    checkpoint_audit = _checkpoint_audit(sweep_rows)
    post_eval_audit = _post_eval_audit(best_combo)
    route = _route(
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        sweep_rows=sweep_rows,
        stable_rows=stable_rows,
        best_combo=best_combo,
        kl_audit=kl_audit,
        checkpoint_audit=checkpoint_audit,
        post_eval_audit=post_eval_audit,
    )
    status = "passed" if route == ROUTE_MULTI_SEED else "failed"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "stage26_7d_root": config["stage26_7d_root"],
        "stage26_7d_status": stage26_7d_summary.get("status"),
        "stage26_7d_next_required_change": stage26_7d_summary.get("next_required_change"),
        "primary_stage26_1_root": str(primary_stage26_1_root) if primary_stage26_1_root else None,
        "combo_count": len(sweep_rows),
        "stable_combo_count": len(stable_rows),
        "best_stable_combo_id": best_combo.get("combo_id"),
        "best_stable_final_post_update_approx_kl": best_combo.get("final_post_update_approx_kl"),
        "best_stable_parameter_delta_l2": best_combo.get("parameter_delta_l2"),
        "min_pre_update_approx_kl": kl_audit.get("min_pre_update_approx_kl"),
        "max_pre_update_approx_kl": kl_audit.get("max_pre_update_approx_kl"),
        "pre_update_kl_baseline_exceeded": kl_audit.get("pre_update_kl_baseline_exceeded"),
        "post_eval_stage26_3_status": best_combo.get("stage26_3_status"),
        "selected_action_changed_count": best_combo.get("selected_action_changed_count", 0),
        "main_final_coverage_delta": best_combo.get("main_final_coverage_delta", 0.0),
        "main_coverage_auc_delta": best_combo.get("main_coverage_auc_delta", 0.0),
        "main_coverage_per_100m_delta": best_combo.get("main_coverage_per_100m_delta", 0.0),
        "hybrid_astar_path_cost_delta": best_combo.get("hybrid_astar_path_cost_delta", 0.0),
        "hybrid_astar_path_cost_delta_is_diagnostic_only": True,
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "coverage_denominator_source": "main_coverable_cells/v1",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "max_traversable_slope_deg": 30.0,
        "max_abs_approx_kl": config["max_abs_approx_kl"],
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "summary": str(output_root / SUMMARY_FILE),
        "update_sweep_results": str(output_root / SWEEP_FILE),
        "kl_stability_audit": str(output_root / KL_AUDIT_FILE),
        "checkpoint_audit": str(output_root / CHECKPOINT_AUDIT_FILE),
        "post_update_eval_audit": str(output_root / POST_EVAL_AUDIT_FILE),
        "recommended_stage26_2_config": str(output_root / RECOMMENDED_STAGE26_2_CONFIG_FILE),
        "recommended_stage26_3_config": str(output_root / RECOMMENDED_STAGE26_3_CONFIG_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }

    _write_jsonl(output_root / SWEEP_FILE, sweep_rows)
    _write_json(output_root / KL_AUDIT_FILE, kl_audit)
    _write_json(output_root / CHECKPOINT_AUDIT_FILE, checkpoint_audit)
    _write_json(output_root / POST_EVAL_AUDIT_FILE, post_eval_audit)
    _write_json(output_root / RECOMMENDED_STAGE26_2_CONFIG_FILE, best_combo.get("stage26_2_config", {}))
    _write_json(output_root / RECOMMENDED_STAGE26_3_CONFIG_FILE, best_combo.get("stage26_3_config", {}))
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary, sweep_rows), encoding="utf-8")
    return summary


def _run_combo(
    *,
    config: dict[str, Any],
    combo: dict[str, Any],
    primary_stage26_1_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    combo_root = output_root / str(combo["work_dir"])
    combo_root.mkdir(parents=True, exist_ok=True)
    stage26_2_config = _build_stage26_2_config(config, combo, primary_stage26_1_root)
    stage26_2_config_path = combo_root / "xunce-stage26-7f-stage26-2-config.json"
    _write_json(stage26_2_config_path, stage26_2_config)
    stage26_2_summary = stage26_2.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke(
        config_path=stage26_2_config_path,
        output_root=combo_root / "s26_2",
        repo_root=repo_root,
    )
    stage26_3_config: dict[str, Any] = {}
    stage26_3_summary: dict[str, Any] = {}
    if _stage26_2_is_stable(stage26_2_summary, float(config["max_abs_approx_kl"])):
        stage26_3_config = _build_stage26_3_config(config, combo_root / "s26_2")
        stage26_3_config_path = combo_root / "xunce-stage26-7f-stage26-3-config.json"
        _write_json(stage26_3_config_path, stage26_3_config)
        stage26_3_summary = stage26_3.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke(
            config_path=stage26_3_config_path,
            output_root=combo_root / "s26_3",
            repo_root=repo_root,
        )
    sweep_row = _sweep_row(combo, combo_root, stage26_2_summary, stage26_3_summary)
    return {
        "combo": combo,
        "combo_root": combo_root,
        "stage26_2_config": stage26_2_config,
        "stage26_3_config": stage26_3_config,
        "stage26_2_summary": stage26_2_summary,
        "stage26_3_summary": stage26_3_summary,
        "sweep_row": sweep_row,
    }


def _build_stage26_2_config(config: dict[str, Any], combo: dict[str, Any], stage26_1_root: Path) -> dict[str, Any]:
    cfg = _read_json(Path(config["stage26_2_base_config"]))
    cfg.update(
        {
            "stage26_1_root": str(stage26_1_root),
            "epochs": int(combo["epochs"]),
            "learning_rate": float(combo["learning_rate"]),
            "clip_ratio": float(config["clip_ratio"]),
            "policy_loss_coefficient": float(combo["policy_loss_coefficient"]),
            "value_loss_coefficient": float(combo["value_loss_coefficient"]),
            "entropy_coefficient": float(config["entropy_coefficient"]),
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


def _build_stage26_3_config(config: dict[str, Any], stage26_2_root: Path) -> dict[str, Any]:
    cfg = _read_json(Path(config["stage26_3_base_config"]))
    cfg.update(
        {
            "stage26_2_root": str(stage26_2_root),
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["eval_rollout_steps"]),
            "coverage_denominator_mode": "main_coverable_cells",
            "coverage_denominator_source": "main_coverable_cells/v1",
            "continuous_theta_action_space_enabled": True,
            "synthetic_credit_feature_exposure_enabled": True,
            "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
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


def _sweep_row(
    combo: dict[str, Any],
    combo_root: Path,
    stage26_2_summary: dict[str, Any],
    stage26_3_summary: dict[str, Any],
) -> dict[str, Any]:
    loss_stats = _stage21_4_loss_audit_stats(stage26_2_summary)
    return {
        "schema_version": SWEEP_ROW_SCHEMA_VERSION,
        "combo_id": combo["combo_id"],
        "combo_work_dir": combo["work_dir"],
        "epochs": int(combo["epochs"]),
        "learning_rate": float(combo["learning_rate"]),
        "policy_loss_coefficient": float(combo["policy_loss_coefficient"]),
        "value_loss_coefficient": float(combo["value_loss_coefficient"]),
        "loss_scale": float(combo["loss_scale"]),
        "stage26_2_status": stage26_2_summary.get("status"),
        "stage26_2_next_required_change": stage26_2_summary.get("next_required_change"),
        "stage21_4_status": stage26_2_summary.get("stage21_4_status"),
        "stage21_4_next_required_change": stage26_2_summary.get("stage21_4_next_required_change"),
        "blocking_reason_codes": stage26_2_summary.get("blocking_reason_codes", []),
        "final_post_update_approx_kl": _finite(stage26_2_summary.get("final_post_update_approx_kl")),
        "pre_update_approx_kl": loss_stats.get("pre_update_approx_kl"),
        "post_minus_pre_update_approx_kl": loss_stats.get("post_minus_pre_update_approx_kl"),
        "pre_update_clip_fraction": loss_stats.get("pre_update_clip_fraction"),
        "loss_finite": bool(stage26_2_summary.get("loss_finite", False)),
        "gradient_finite": bool(stage26_2_summary.get("gradient_finite", False)),
        "final_total_loss": _finite(stage26_2_summary.get("final_total_loss")),
        "final_policy_loss": _finite(stage26_2_summary.get("final_policy_loss")),
        "final_value_loss": _finite(stage26_2_summary.get("final_value_loss")),
        "final_entropy": _finite(stage26_2_summary.get("final_entropy")),
        "pre_clip_grad_norm": _finite(stage26_2_summary.get("pre_clip_grad_norm")),
        "post_clip_grad_norm": _finite(stage26_2_summary.get("post_clip_grad_norm")),
        "parameter_delta_l2": _finite(stage26_2_summary.get("parameter_delta_l2")),
        "checkpoint_exists": bool(stage26_2_summary.get("checkpoint_exists", False)),
        "checkpoint_reload_passed": bool(stage26_2_summary.get("checkpoint_reload_passed", False)),
        "experimental_only": bool(stage26_2_summary.get("experimental_only", False)),
        "checkpoint_boundary_passed": bool(stage26_2_summary.get("checkpoint_boundary_passed", False)),
        "old_log_prob_source": _stage21_4_old_log_prob_source(stage26_2_summary),
        "coverage_source_mismatch_count": int(stage26_2_summary.get("coverage_source_mismatch_count") or 0),
        "path_cost_source_mismatch_count": int(stage26_2_summary.get("path_cost_source_mismatch_count") or 0),
        "synthetic_terrain_contract_missing_count": int(stage26_2_summary.get("synthetic_terrain_contract_missing_count") or 0),
        "publishes_checkpoint": bool(stage26_2_summary.get("publishes_checkpoint", False)),
        "replaces_default_policy": bool(stage26_2_summary.get("replaces_default_policy", False)),
        "connects_real_executor": bool(stage26_2_summary.get("connects_real_executor", False)),
        "starts_online_canary": bool(stage26_2_summary.get("starts_online_canary", False)),
        "canary_traffic_fraction": float(stage26_2_summary.get("canary_traffic_fraction") or 0.0),
        "stage26_3_status": stage26_3_summary.get("status"),
        "stage26_3_next_required_change": stage26_3_summary.get("next_required_change"),
        **_eval_metrics(stage26_3_summary),
        "combo_root": str(combo_root),
    }


def _stage21_4_old_log_prob_source(stage26_2_summary: dict[str, Any]) -> str | None:
    root = stage26_2_summary.get("stage21_4_root")
    if not root:
        return None
    summary = _read_json_if_exists(Path(str(root)) / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json")
    return summary.get("old_log_prob_source")


def _stage21_4_loss_audit_stats(stage26_2_summary: dict[str, Any]) -> dict[str, Any]:
    root = stage26_2_summary.get("stage21_4_root")
    if not root:
        return {}
    summary = _read_json_if_exists(Path(str(root)) / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json")
    loss_audit_path = summary.get("loss_audit") or str(Path(str(root)) / "xunce-stage21-4-ppo-loss-audit.jsonl")
    rows = _read_jsonl_if_exists(Path(str(loss_audit_path)))
    if not rows:
        return {}
    first = rows[0]
    last = rows[-1]
    pre_kl = _finite(first.get("pre_update_approx_kl"))
    post_kl = _finite(last.get("post_update_approx_kl"))
    return {
        "pre_update_approx_kl": pre_kl,
        "post_update_approx_kl_from_loss_audit": post_kl,
        "post_minus_pre_update_approx_kl": None if pre_kl is None or post_kl is None else post_kl - pre_kl,
        "pre_update_clip_fraction": _finite(first.get("clip_fraction")),
    }


def _eval_metrics(stage26_3_summary: dict[str, Any]) -> dict[str, Any]:
    main_final = _first_finite(stage26_3_summary.get("main_final_coverage_delta"), stage26_3_summary.get("final_coverage_delta"), 0.0)
    main_auc = _first_finite(stage26_3_summary.get("main_coverage_auc_delta"), stage26_3_summary.get("coverage_auc_delta"), 0.0)
    main_per_100m = _first_finite(
        stage26_3_summary.get("main_coverage_per_100m_delta"),
        stage26_3_summary.get("coverage_per_100m_delta"),
        0.0,
    )
    return {
        "selected_action_changed_count": int(stage26_3_summary.get("selected_action_changed_count") or 0),
        "selected_viewpoint_changed_count": int(stage26_3_summary.get("selected_viewpoint_changed_count") or 0),
        "selected_theta_changed_count": int(stage26_3_summary.get("selected_theta_changed_count") or 0),
        "mean_abs_probability_delta": float(stage26_3_summary.get("mean_abs_probability_delta") or 0.0),
        "main_final_coverage_delta": main_final,
        "main_coverage_auc_delta": main_auc,
        "main_coverage_per_100m_delta": main_per_100m,
        "hybrid_astar_path_cost_delta": _first_finite(stage26_3_summary.get("hybrid_astar_path_cost_delta"), 0.0),
        "hard_risk_violation_count": int(stage26_3_summary.get("hard_risk_violation_count") or 0),
        "mask_violation_count": int(stage26_3_summary.get("mask_violation_count") or 0),
        "path_planning_failure_count": int(stage26_3_summary.get("path_planning_failure_count") or 0),
        "open_grid_fallback_count": int(stage26_3_summary.get("open_grid_fallback_count") or 0),
        "scenario_regression_count": int(stage26_3_summary.get("scenario_regression_count") or 0),
    }


def _kl_stability_audit(rows: list[dict[str, Any]], max_abs_approx_kl: float) -> dict[str, Any]:
    kl_values = [_finite(row.get("final_post_update_approx_kl")) for row in rows]
    pre_kl_values = [_finite(row.get("pre_update_approx_kl")) for row in rows]
    clean = [value for value in kl_values if value is not None]
    clean_pre = [value for value in pre_kl_values if value is not None]
    return {
        "schema_version": KL_AUDIT_SCHEMA_VERSION,
        "combo_count": len(rows),
        "max_abs_approx_kl": max_abs_approx_kl,
        "stable_combo_count": sum(1 for row in rows if _combo_is_stable(row, max_abs_approx_kl)),
        "kl_exceeded_count": sum(1 for value in clean if abs(value) > max_abs_approx_kl),
        "pre_update_kl_exceeded_count": sum(1 for value in clean_pre if abs(value) > max_abs_approx_kl),
        "pre_update_kl_baseline_exceeded": _pre_update_kl_baseline_exceeded(rows, max_abs_approx_kl),
        "min_pre_update_approx_kl": min(clean_pre) if clean_pre else None,
        "max_pre_update_approx_kl": max(clean_pre) if clean_pre else None,
        "min_final_post_update_approx_kl": min(clean) if clean else None,
        "max_final_post_update_approx_kl": max(clean) if clean else None,
        "loss_nonfinite_count": sum(1 for row in rows if row.get("loss_finite") is not True),
        "gradient_nonfinite_count": sum(1 for row in rows if row.get("gradient_finite") is not True),
    }


def _checkpoint_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_AUDIT_SCHEMA_VERSION,
        "combo_count": len(rows),
        "checkpoint_exists_count": sum(1 for row in rows if row.get("checkpoint_exists") is True),
        "checkpoint_reload_passed_count": sum(1 for row in rows if row.get("checkpoint_reload_passed") is True),
        "experimental_only_count": sum(1 for row in rows if row.get("experimental_only") is True),
        "checkpoint_boundary_passed_count": sum(1 for row in rows if row.get("checkpoint_boundary_passed") is True),
        "boundary_enabled_count": sum(1 for row in rows if _boundary_enabled_in_row(row)),
    }


def _post_eval_audit(best_combo: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": POST_EVAL_AUDIT_SCHEMA_VERSION,
        "best_stable_combo_id": best_combo.get("combo_id"),
        "stage26_3_status": best_combo.get("stage26_3_status"),
        "stage26_3_next_required_change": best_combo.get("stage26_3_next_required_change"),
        "selected_action_changed_count": best_combo.get("selected_action_changed_count", 0),
        "selected_viewpoint_changed_count": best_combo.get("selected_viewpoint_changed_count", 0),
        "selected_theta_changed_count": best_combo.get("selected_theta_changed_count", 0),
        "mean_abs_probability_delta": best_combo.get("mean_abs_probability_delta", 0.0),
        "main_final_coverage_delta": best_combo.get("main_final_coverage_delta", 0.0),
        "main_coverage_auc_delta": best_combo.get("main_coverage_auc_delta", 0.0),
        "main_coverage_per_100m_delta": best_combo.get("main_coverage_per_100m_delta", 0.0),
        "hybrid_astar_path_cost_delta": best_combo.get("hybrid_astar_path_cost_delta", 0.0),
        "hybrid_astar_path_cost_delta_is_diagnostic_only": True,
        "safety_count_nonzero": any(
            int(best_combo.get(key) or 0) != 0
            for key in ("hard_risk_violation_count", "mask_violation_count", "path_planning_failure_count", "open_grid_fallback_count")
        ),
    }


def _route(
    *,
    boundary_rejections: list[str],
    input_rejections: list[str],
    sweep_rows: list[dict[str, Any]],
    stable_rows: list[dict[str, Any]],
    best_combo: dict[str, Any],
    kl_audit: dict[str, Any],
    checkpoint_audit: dict[str, Any],
    post_eval_audit: dict[str, Any],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections:
        return ROUTE_INPUTS
    if int(kl_audit.get("loss_nonfinite_count") or 0) > 0 or int(kl_audit.get("gradient_nonfinite_count") or 0) > 0:
        return ROUTE_NUMERICS
    if _pre_update_kl_baseline_exceeded(sweep_rows, float(kl_audit.get("max_abs_approx_kl") or MAX_ALLOWED_KL_LIMIT)):
        return ROUTE_KL_BASELINE
    kl_stable_rows = [row for row in sweep_rows if _combo_has_stable_kl_and_finite_update(row, float(kl_audit.get("max_abs_approx_kl") or MAX_ALLOWED_KL_LIMIT))]
    if kl_stable_rows and any(not _combo_has_checkpoint_boundary(row) for row in kl_stable_rows):
        return ROUTE_CHECKPOINT
    if not stable_rows:
        return ROUTE_REDUCE
    if best_combo.get("stage26_3_status") is None:
        return ROUTE_EVAL_BINDING
    if best_combo.get("stage26_3_status") == "failed" and best_combo.get("stage26_3_next_required_change") in {
        "repair_stage26_3_synthetic_inference_binding",
        "repair_stage26_3_synthetic_lineage_binding",
    }:
        return ROUTE_EVAL_BINDING
    if int(post_eval_audit.get("selected_action_changed_count") or 0) <= 0:
        return ROUTE_MARGIN
    if float(post_eval_audit.get("main_coverage_per_100m_delta") or 0.0) < 0.0:
        return ROUTE_CREDIT
    if _post_eval_success(post_eval_audit, best_combo):
        return ROUTE_MULTI_SEED
    return ROUTE_CREDIT


def _post_eval_success(post_eval_audit: dict[str, Any], best_combo: dict[str, Any]) -> bool:
    return (
        best_combo.get("stage26_3_status") == "passed"
        and int(post_eval_audit.get("selected_action_changed_count") or 0) > 0
        and float(post_eval_audit.get("main_final_coverage_delta") or 0.0) > 0.0
        and float(post_eval_audit.get("main_coverage_auc_delta") or 0.0) > 0.0
        and float(post_eval_audit.get("main_coverage_per_100m_delta") or 0.0) >= 0.0
        and int(best_combo.get("scenario_regression_count") or 0) == 0
        and post_eval_audit.get("safety_count_nonzero") is False
    )


def _pre_update_kl_baseline_exceeded(rows: list[dict[str, Any]], max_abs_approx_kl: float) -> bool:
    values = [_finite(row.get("pre_update_approx_kl")) for row in rows]
    clean = [value for value in values if value is not None]
    return bool(clean) and min(abs(value) for value in clean) > max_abs_approx_kl


def _stage26_2_is_stable(summary: dict[str, Any], max_abs_approx_kl: float) -> bool:
    row = {
        "stage26_2_status": summary.get("status"),
        "stage21_4_status": summary.get("stage21_4_status"),
        "checkpoint_reload_passed": summary.get("checkpoint_reload_passed"),
        "experimental_only": summary.get("experimental_only"),
        "checkpoint_boundary_passed": summary.get("checkpoint_boundary_passed"),
        "final_post_update_approx_kl": summary.get("final_post_update_approx_kl"),
        "loss_finite": summary.get("loss_finite"),
        "gradient_finite": summary.get("gradient_finite"),
        "publishes_checkpoint": summary.get("publishes_checkpoint", False),
        "replaces_default_policy": summary.get("replaces_default_policy", False),
        "connects_real_executor": summary.get("connects_real_executor", False),
        "starts_online_canary": summary.get("starts_online_canary", False),
        "canary_traffic_fraction": summary.get("canary_traffic_fraction", 0.0),
    }
    return _combo_is_stable(row, max_abs_approx_kl)


def _combo_is_stable(row: dict[str, Any], max_abs_approx_kl: float) -> bool:
    return _combo_has_stable_kl_and_finite_update(row, max_abs_approx_kl) and _combo_has_checkpoint_boundary(row)


def _combo_has_stable_kl_and_finite_update(row: dict[str, Any], max_abs_approx_kl: float) -> bool:
    kl = _finite(row.get("final_post_update_approx_kl"))
    return (
        row.get("stage21_4_status") == "passed"
        and row.get("loss_finite") is True
        and row.get("gradient_finite") is True
        and kl is not None
        and abs(kl) <= max_abs_approx_kl
    )


def _combo_has_checkpoint_boundary(row: dict[str, Any]) -> bool:
    return (
        row.get("checkpoint_reload_passed") is True
        and row.get("experimental_only") is True
        and row.get("checkpoint_boundary_passed") is True
        and not _boundary_enabled_in_row(row)
    )


def _best_stable_combo(combo_details: list[dict[str, Any]], max_abs_approx_kl: float) -> dict[str, Any]:
    stable = [detail for detail in combo_details if _combo_is_stable(detail["sweep_row"], max_abs_approx_kl)]
    if not stable:
        return {}
    best = max(
        stable,
        key=lambda detail: (
            _kl_not_close_to_limit(detail["sweep_row"], max_abs_approx_kl),
            float(detail["sweep_row"].get("parameter_delta_l2") or 0.0),
            -abs(float(detail["sweep_row"].get("final_post_update_approx_kl") or 0.0)),
        ),
    )
    return dict(best["sweep_row"], stage26_2_config=best.get("stage26_2_config", {}), stage26_3_config=best.get("stage26_3_config", {}))


def _kl_not_close_to_limit(row: dict[str, Any], max_abs_approx_kl: float) -> bool:
    kl = abs(float(row.get("final_post_update_approx_kl") or 0.0))
    return kl <= 0.95 * max_abs_approx_kl


def _boundary_enabled_in_row(row: dict[str, Any]) -> bool:
    return (
        row.get("publishes_checkpoint") is True
        or row.get("replaces_default_policy") is True
        or row.get("connects_real_executor") is True
        or row.get("starts_online_canary") is True
        or float(row.get("canary_traffic_fraction") or 0.0) != 0.0
    )


def _primary_stage26_1_root(config: dict[str, Any], stage26_7d_summary: dict[str, Any]) -> Path | None:
    if config.get("primary_stage26_1_root"):
        return Path(str(config["primary_stage26_1_root"]))
    chain_root = stage26_7d_summary.get("stage26_7c_rerun_output_root")
    if not chain_root:
        return None
    return Path(str(chain_root)) / "r" / str(config["primary_combo_work_dir"]) / "s26_1"


def _input_rejections(summary: dict[str, Any], primary_stage26_1_root: Path | None) -> list[str]:
    if not summary:
        return ["missing_stage26_7d_summary"]
    reasons: list[str] = []
    if summary.get("stage_id") != "xunce-stage26-7d-repair-synthetic-credit-sampler-continuous-theta-reachability":
        reasons.append("stage26_7d_wrong_stage_id")
    if summary.get("next_required_change") != ROUTE_FROM_STAGE26_7D:
        reasons.append("stage26_7d_route_mismatch")
    if int(summary.get("trainable_transition_count") or 0) < REQUIRED_STAGE26_7D_TRAINABLE_TRANSITION_COUNT:
        reasons.append("stage26_7d_trainable_transition_count_short")
    if int(summary.get("synthetic_credit_target_selected_count") or 0) < REQUIRED_STAGE26_7D_CREDIT_TARGET_SELECTED_COUNT:
        reasons.append("stage26_7d_credit_target_selected_count_short")
    if int(summary.get("selected_continuous_theta_hybrid_astar_unreachable_count") or 0) != 0:
        reasons.append("stage26_7d_unreachable_theta_count_nonzero")
    if summary.get("behavior_theta_logprob_recomputable") is not True:
        reasons.append("stage26_7d_behavior_theta_logprob_not_recomputable")
    if primary_stage26_1_root is None or not primary_stage26_1_root.exists():
        reasons.append("primary_stage26_1_root_missing")
    else:
        stage26_1_summary = _read_json_if_exists(primary_stage26_1_root / "xunce-stage26-1-summary.json")
        if stage26_1_summary.get("status") != "passed":
            reasons.append("primary_stage26_1_status_not_passed")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    if payload.get("stage_id") != STAGE_ID:
        raise ValueError(f"stage_id must be {STAGE_ID}")
    config = dict(payload)
    config["stage26_7d_root"] = str(_resolve_path(Path(config.get("stage26_7d_root") or DEFAULT_STAGE26_7D_ROOT), repo_root))
    config["stage26_2_base_config"] = str(_resolve_path(Path(config.get("stage26_2_base_config") or stage26_2.DEFAULT_CONFIG), repo_root))
    config["stage26_3_base_config"] = str(_resolve_path(Path(config.get("stage26_3_base_config") or stage26_3.DEFAULT_CONFIG), repo_root))
    config["run_stage26_chain"] = bool(config.get("run_stage26_chain", True))
    config["primary_combo_work_dir"] = str(config.get("primary_combo_work_dir", "c1"))
    if config.get("primary_stage26_1_root"):
        config["primary_stage26_1_root"] = str(_resolve_path(Path(str(config["primary_stage26_1_root"])), repo_root))
    config["required_scenario_count"] = int(config.get("required_scenario_count", 2))
    config["eval_rollout_steps"] = int(config.get("eval_rollout_steps", 4))
    config["hybrid_astar_candidate_eval_workers"] = int(config.get("hybrid_astar_candidate_eval_workers", 4))
    config["clip_ratio"] = float(config.get("clip_ratio", 0.2))
    config["entropy_coefficient"] = float(config.get("entropy_coefficient", 0.01))
    config["advantage_clip_abs"] = float(config.get("advantage_clip_abs", 5.0))
    config["max_grad_norm"] = float(config.get("max_grad_norm", 1.0))
    config["max_abs_approx_kl"] = float(config.get("max_abs_approx_kl", MAX_ALLOWED_KL_LIMIT))
    if not math.isfinite(config["max_abs_approx_kl"]) or config["max_abs_approx_kl"] > MAX_ALLOWED_KL_LIMIT:
        raise ValueError("max_abs_approx_kl must be finite and must not exceed 1.5 for Stage26.7F")
    config["update_sweep"] = _update_sweep(config.get("update_sweep"))
    config["canary_traffic_fraction"] = float(config.get("canary_traffic_fraction", 0.0))
    for field in BOUNDARY_FIELDS:
        config[field] = bool(config.get(field, False))
    return config


def _update_sweep(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        value = [
            {"combo_id": "current_repro", "work_dir": "u0", "epochs": 4, "learning_rate": 1.0e-5, "policy_loss_coefficient": 1.0, "value_loss_coefficient": 0.02, "loss_scale": 0.25},
            {"combo_id": "lr_half", "work_dir": "u1", "epochs": 4, "learning_rate": 5.0e-6, "policy_loss_coefficient": 1.0, "value_loss_coefficient": 0.02, "loss_scale": 0.25},
            {"combo_id": "shallow_lr_half", "work_dir": "u2", "epochs": 2, "learning_rate": 5.0e-6, "policy_loss_coefficient": 1.0, "value_loss_coefficient": 0.02, "loss_scale": 0.25},
            {"combo_id": "conservative", "work_dir": "u3", "epochs": 1, "learning_rate": 5.0e-6, "policy_loss_coefficient": 1.0, "value_loss_coefficient": 0.02, "loss_scale": 0.25},
            {"combo_id": "ultra_conservative", "work_dir": "u4", "epochs": 1, "learning_rate": 2.5e-6, "policy_loss_coefficient": 1.0, "value_loss_coefficient": 0.02, "loss_scale": 0.25},
            {"combo_id": "policy_soft", "work_dir": "u5", "epochs": 2, "learning_rate": 5.0e-6, "policy_loss_coefficient": 0.5, "value_loss_coefficient": 0.02, "loss_scale": 0.25},
        ]
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError("update_sweep entries must be objects")
        result.append(
            {
                "combo_id": str(item.get("combo_id") or f"combo_{index:02d}"),
                "work_dir": str(item.get("work_dir") or f"u{index}"),
                "epochs": _positive_int(item.get("epochs", 1), "epochs"),
                "learning_rate": _positive_float(item.get("learning_rate", 5.0e-6), "learning_rate"),
                "policy_loss_coefficient": _nonnegative_float(item.get("policy_loss_coefficient", 1.0), "policy_loss_coefficient"),
                "value_loss_coefficient": _nonnegative_float(item.get("value_loss_coefficient", 0.02), "value_loss_coefficient"),
                "loss_scale": _positive_float(item.get("loss_scale", 0.25), "loss_scale"),
            }
        )
    return result


def _render_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage26.7F Synthetic Credit PPO Update Stability Sweep",
        "",
        f"- status: `{summary['status']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- stable_combo_count: `{summary['stable_combo_count']}`",
        f"- best_stable_combo_id: `{summary.get('best_stable_combo_id')}`",
        f"- best_stable_final_post_update_approx_kl: `{summary.get('best_stable_final_post_update_approx_kl')}`",
        f"- pre_update_kl_baseline_exceeded: `{summary.get('pre_update_kl_baseline_exceeded')}`",
        f"- min_pre_update_approx_kl: `{summary.get('min_pre_update_approx_kl')}`",
        f"- main_coverage_per_100m_delta: `{summary.get('main_coverage_per_100m_delta')}`",
        "",
        "Combo results:",
    ]
    for row in rows:
        lines.append(
            "- "
            f"{row.get('combo_id')}: s26_2={row.get('stage26_2_status')}, "
            f"kl={row.get('final_post_update_approx_kl')}, "
            f"checkpoint_reload={row.get('checkpoint_reload_passed')}, "
            f"s26_3={row.get('stage26_3_status')}, "
            f"main_per_100m_delta={row.get('main_coverage_per_100m_delta')}"
        )
    lines.extend(
        [
            "",
            "This is a bounded offline smoke. It does not publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    try:
        return _read_json(path)
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
        return {}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    try:
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
        return rows
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
        return []


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _positive_int(value: Any, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _nonnegative_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_finite(*values: Any) -> float:
    for value in values:
        number = _finite(value)
        if number is not None:
            return number
    return 0.0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
