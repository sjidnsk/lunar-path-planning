from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import run_xunce_stage26_1_synthetic_terrain_collector_smoke as stage26_1
    import run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as stage26_2
    import run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as stage26_3
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage26_1_synthetic_terrain_collector_smoke as stage26_1
    import scripts.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as stage26_2
    import scripts.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as stage26_3


STAGE_ID = "xunce-stage26-8-synthetic-terrain-multi-seed-coverage-efficiency-pilot"
CONFIG_SCHEMA_VERSION = "xunce-stage26-8-synthetic-terrain-multi-seed-coverage-efficiency-pilot-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-8-summary/v1"
SEED_RESULT_SCHEMA_VERSION = "xunce-stage26-8-seed-result/v1"
AGGREGATE_SCHEMA_VERSION = "xunce-stage26-8-efficiency-aggregate-audit/v1"
SCENARIO_AUDIT_SCHEMA_VERSION = "xunce-stage26-8-scenario-efficiency-audit/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "xunce-stage26-8-boundary-safety-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-8-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-8-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot_v1"
)
DEFAULT_STAGE26_7H_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_7h_repair_credit_post_update_eval_binding_v1"
)
DEFAULT_STAGE26_0_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract_v1"
)

SUMMARY_FILE = "xunce-stage26-8-summary.json"
SEED_RESULTS_FILE = "xunce-stage26-8-seed-results.jsonl"
AGGREGATE_AUDIT_FILE = "xunce-stage26-8-efficiency-aggregate-audit.json"
SCENARIO_AUDIT_FILE = "xunce-stage26-8-scenario-efficiency-audit.json"
BOUNDARY_AUDIT_FILE = "xunce-stage26-8-boundary-safety-audit.json"
RECOMMENDED_CONFIG_FILE = "xunce-stage26-8-recommended-next-config.json"
ROUTING_FILE = "xunce-stage26-8-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-8-report.md"
MANIFEST_FILE = "xunce-stage26-8-manifest.json"

COVERAGE_SOURCE = "endpoint_theta_slope_obstacle_los/v1"
PATH_COST_SOURCE = "hybrid_astar_pose_path/v1"
SYNTHETIC_SOURCE_KIND = "synthetic_terrain_obstacle_proxy/v1"
ACTION_SPACE_TYPE = "hybrid_discrete_xy_continuous_theta/v1"
COVERAGE_DENOMINATOR_SOURCE = "main_coverable_cells/v1"
SUCCESS_METRIC = "main_coverable_coverage_efficiency/v1"

ROUTE_INPUTS = "rerun_stage26_8_required_inputs"
ROUTE_METRIC = "repair_stage26_8_efficiency_metric_contract"
ROUTE_BINDING_OR_SAFETY = "repair_stage26_8_multiseed_eval_binding_or_safety"
ROUTE_UPDATE = "repair_stage26_8_update_stability"
ROUTE_CREDIT = "repair_stage26_synthetic_credit_assignment"
ROUTE_EXPAND = "expand_stage26_8_seed_or_horizon_budget"
ROUTE_STAGE26_9 = "run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot"
ROUTE_BOUNDARY = "resolve_stage26_8_boundary_rejections"
ACCEPTED_STAGE26_7H_ROUTES = {
    ("passed", "run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot"),
    ("failed", "repair_stage26_synthetic_credit_assignment"),
}

BOUNDARY_FIELDS = (
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)
COUNT_FIELDS_REQUIRING_ZERO = (
    "synthetic_inference_required_field_missing_count",
    "hybrid_path_missing_provenance_count",
    "hybrid_path_contract_mismatch_count",
    "explicit_unreachable_selected_provenance_count",
    "pre_unreachable_selected_count",
    "post_unreachable_selected_count",
    "hard_risk_violation_count",
    "mask_violation_count",
    "path_planning_failure_count",
    "open_grid_fallback_count",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.8 multi-seed coverage efficiency pilot.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(config_path, repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage26_7h_summary = _read_json_if_exists(Path(config["stage26_7h_root"]) / "xunce-stage26-7h-summary.json")
    boundary_rejections = _boundary_rejections(config)
    input_rejections = _input_rejections(stage26_7h_summary)
    seed_rows: list[dict[str, Any]] = []
    if not boundary_rejections and not input_rejections and bool(config["run_stage26_chain"]):
        for seed_index, seed in enumerate(config["seeds"]):
            row = _run_seed(
                config=config,
                seed=int(seed),
                seed_index=int(seed_index),
                output_root=output_root,
                repo_root=repo_root,
            )
            seed_rows.append(row)
            if bool(config.get("stop_on_first_seed_blocker")) and (
                row.get("binding_or_safety_failure") or row.get("execution_failure") or row.get("update_failure")
            ):
                break

    aggregate = _efficiency_aggregate(seed_rows)
    scenario_audit = _scenario_efficiency_audit(output_root, seed_rows)
    boundary_audit = _boundary_safety_audit(seed_rows, boundary_rejections)
    route = _route(boundary_rejections=boundary_rejections, input_rejections=input_rejections, seed_rows=seed_rows, aggregate=aggregate)
    status = "passed" if route == ROUTE_STAGE26_9 else "failed"
    summary = {
        **aggregate,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "stage26_7h_root": config["stage26_7h_root"],
        "stage26_7h_status": stage26_7h_summary.get("status"),
        "stage26_7h_next_required_change": stage26_7h_summary.get("next_required_change"),
        "coverage_denominator_source": COVERAGE_DENOMINATOR_SOURCE,
        "coverage_source": COVERAGE_SOURCE,
        "path_cost_source": PATH_COST_SOURCE,
        "synthetic_source_kind": SYNTHETIC_SOURCE_KIND,
        "action_space_type": ACTION_SPACE_TYPE,
        "max_traversable_slope_deg": 30.0,
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "summary": str(output_root / SUMMARY_FILE),
    }
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "summary_status": status,
        "next_required_change": route,
        "artifacts": {
            "summary": str(output_root / SUMMARY_FILE),
            "seed_results": str(output_root / SEED_RESULTS_FILE),
            "efficiency_aggregate_audit": str(output_root / AGGREGATE_AUDIT_FILE),
            "scenario_efficiency_audit": str(output_root / SCENARIO_AUDIT_FILE),
            "boundary_safety_audit": str(output_root / BOUNDARY_AUDIT_FILE),
            "recommended_next_config": str(output_root / RECOMMENDED_CONFIG_FILE),
            "routing": str(output_root / ROUTING_FILE),
            "report": str(output_root / REPORT_FILE),
        },
    }
    _write_jsonl(output_root / SEED_RESULTS_FILE, seed_rows)
    _write_json(output_root / AGGREGATE_AUDIT_FILE, aggregate)
    _write_json(output_root / SCENARIO_AUDIT_FILE, scenario_audit)
    _write_json(output_root / BOUNDARY_AUDIT_FILE, boundary_audit)
    _write_json(output_root / RECOMMENDED_CONFIG_FILE, _recommended_next_config(config, aggregate))
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary, seed_rows), encoding="utf-8")
    return summary


def _run_seed(*, config: dict[str, Any], seed: int, seed_index: int, output_root: Path, repo_root: Path) -> dict[str, Any]:
    seed_root = output_root / f"s{seed_index}"
    seed_root.mkdir(parents=True, exist_ok=True)
    stage26_1_config_path = seed_root / "xunce-stage26-8-stage26-1-config.json"
    stage26_2_config_path = seed_root / "xunce-stage26-8-stage26-2-config.json"
    stage26_3_config_path = seed_root / "xunce-stage26-8-stage26-3-config.json"
    stage26_1_root = seed_root / "s26_1"
    stage26_2_root = seed_root / "s26_2"
    stage26_3_root = seed_root / "s26_3"
    reuse_existing = bool(config.get("reuse_existing_seed_outputs"))
    reused_stage26_1 = False
    reused_stage26_2 = False
    reused_stage26_3 = False

    stage26_1_config = _build_stage26_1_config(config, seed, seed_root, repo_root)
    _write_json(stage26_1_config_path, stage26_1_config)
    stage26_1_summary = _read_json_if_exists(stage26_1_root / stage26_1.SUMMARY_FILE) if reuse_existing else {}
    if stage26_1_summary.get("status") == "passed":
        reused_stage26_1 = True
    else:
        stage26_1_summary = stage26_1.run_xunce_stage26_1_synthetic_terrain_collector_smoke(
            config_path=stage26_1_config_path,
            output_root=stage26_1_root,
            repo_root=repo_root,
        )

    stage26_2_summary: dict[str, Any] = _read_json_if_exists(stage26_2_root / stage26_2.SUMMARY_FILE) if reuse_existing else {}
    stage26_3_summary: dict[str, Any] = _read_json_if_exists(stage26_3_root / stage26_3.SUMMARY_FILE) if reuse_existing else {}
    if stage26_1_summary.get("status") == "passed":
        stage26_2_config = _build_stage26_2_config(config, seed=seed, stage26_1_root=stage26_1_root, repo_root=repo_root)
        _write_json(stage26_2_config_path, stage26_2_config)
        if stage26_2_summary.get("status") == "passed":
            reused_stage26_2 = True
        else:
            stage26_2_summary = stage26_2.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke(
                config_path=stage26_2_config_path,
                output_root=stage26_2_root,
                repo_root=repo_root,
            )
    if stage26_3_summary:
        reused_stage26_3 = True
    if stage26_2_summary.get("status") == "passed" and not stage26_3_summary:
        stage26_3_config = _build_stage26_3_config(config, seed=seed, stage26_2_root=stage26_2_root, repo_root=repo_root)
        _write_json(stage26_3_config_path, stage26_3_config)
        stage26_3_summary = stage26_3.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke(
            config_path=stage26_3_config_path,
            output_root=stage26_3_root,
            repo_root=repo_root,
        )

    collector_failure = stage26_1_summary.get("status") != "passed"
    update_failure = (not collector_failure) and stage26_2_summary.get("status") != "passed"
    row = {
        "schema_version": SEED_RESULT_SCHEMA_VERSION,
        "seed": seed,
        "seed_index": seed_index,
        "seed_root": str(seed_root),
        "stage26_1_root": str(stage26_1_root),
        "stage26_1_status": stage26_1_summary.get("status"),
        "stage26_1_next_required_change": stage26_1_summary.get("next_required_change"),
        "stage26_2_root": str(stage26_2_root),
        "stage26_2_status": stage26_2_summary.get("status"),
        "stage26_2_next_required_change": stage26_2_summary.get("next_required_change"),
        "stage26_3_root": str(stage26_3_root),
        "stage26_3_status": stage26_3_summary.get("status"),
        "stage26_3_next_required_change": stage26_3_summary.get("next_required_change"),
        "main_final_coverage_delta": _float(stage26_3_summary.get("final_coverage_delta")),
        "main_coverage_auc_delta": _float(stage26_3_summary.get("coverage_auc_delta")),
        "main_coverage_per_100m_delta": _float(stage26_3_summary.get("coverage_per_100m_delta")),
        "hybrid_astar_path_cost_delta": _float(stage26_3_summary.get("hybrid_astar_path_cost_delta")),
        "coverage_denominator_source": stage26_3_summary.get("coverage_denominator_source"),
        "coverage_source": stage26_3_summary.get("coverage_source"),
        "path_cost_source": stage26_3_summary.get("path_cost_source"),
        "synthetic_source_kind": stage26_3_summary.get("synthetic_source_kind"),
        "action_space_type": stage26_3_summary.get("action_space_type"),
        "synthetic_terrain_hash": stage26_3_summary.get("synthetic_terrain_hash"),
        "platform_contract_hash": stage26_3_summary.get("platform_contract_hash"),
        "max_traversable_slope_deg": stage26_3_summary.get("max_traversable_slope_deg"),
        "lineage_mismatch": bool(stage26_3_summary) and _lineage_mismatch(stage26_3_summary),
        "efficiency_seed_passed": _efficiency_seed_passed(stage26_3_summary),
        "binding_or_safety_failure": (
            bool(collector_failure)
            or (
                bool(stage26_3_summary)
                and (_binding_or_safety_failure(stage26_3_summary) or _lineage_mismatch(stage26_3_summary))
            )
        ),
        "execution_failure": _stage26_3_execution_failure(stage26_3_summary),
        "collector_failure": collector_failure,
        "update_failure": update_failure,
        "reused_existing_stage26_1": reused_stage26_1,
        "reused_existing_stage26_2": reused_stage26_2,
        "reused_existing_stage26_3": reused_stage26_3,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    return row


def _build_stage26_1_config(config: dict[str, Any], seed: int, seed_root: Path, repo_root: Path) -> dict[str, Any]:
    cfg = _load_json_template(config["stage26_1_base_config"], repo_root)
    stage21_1_base = _load_json_template(cfg["stage21_1_base_config"], repo_root)
    stage21_1_base["sampling_seed"] = int(seed)
    stage21_1_base["continuous_theta_head_init_seed"] = int(seed)
    generated_stage21_1_base = seed_root / "xunce-stage26-8-stage21-1-base-config.json"
    _write_json(generated_stage21_1_base, stage21_1_base)
    cfg.update(
        {
            "stage26_8_seed": int(seed),
            "stage26_0_root": config["stage26_0_root"],
            "stage21_1_base_config": str(generated_stage21_1_base),
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["collector_rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "theta_bin_count": int(config["theta_bin_count"]),
            "theta_step_deg": int(config["theta_step_deg"]),
            "sensor_fov_deg": float(config["sensor_fov_deg"]),
            "sensor_range_cells": int(config["sensor_range_cells"]),
            "action_space_type": ACTION_SPACE_TYPE,
            "continuous_theta_action_space_enabled": True,
            "synthetic_credit_feature_exposure_enabled": True,
            "synthetic_exploration_credit_enabled": True,
            "allow_synthetic_credit_behavior_policy": True,
            "synthetic_credit_mixture_probability": float(config["synthetic_credit_mixture_probability"]),
            "synthetic_credit_score_version": str(config["synthetic_credit_score_version"]),
            "path_efficiency_max_cost_norm": float(config["path_efficiency_max_cost_norm"]),
            "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
            "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
            "scenario_diversity_contract_enabled": bool(config.get("scenario_diversity_contract_enabled", False)),
            "scenario_diversity_source": str(
                config.get("scenario_diversity_source", "synthetic_roi_start_seed_matrix/v1")
            ),
            "scenario_seed_base": int(config.get("scenario_seed_base", seed)),
            "min_scenario_start_separation_cells": int(config.get("min_scenario_start_separation_cells", 5)),
            "min_scenario_start_clearance_cells": int(config.get("min_scenario_start_clearance_cells", 1)),
            "stage26_1_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _build_stage26_2_config(config: dict[str, Any], *, seed: int, stage26_1_root: Path, repo_root: Path) -> dict[str, Any]:
    cfg = _load_json_template(config["stage26_2_base_config"], repo_root)
    cfg.update(
        {
            "stage26_8_seed": int(seed),
            "stage26_1_root": str(stage26_1_root),
            "epochs": int(config["epochs"]),
            "learning_rate": float(config["learning_rate"]),
            "clip_ratio": float(config["clip_ratio"]),
            "policy_loss_coefficient": float(config["policy_loss_coefficient"]),
            "value_loss_coefficient": float(config["value_loss_coefficient"]),
            "loss_scale": float(config["loss_scale"]),
            "max_grad_norm": float(config["max_grad_norm"]),
            "max_abs_approx_kl": float(config["max_abs_approx_kl"]),
            "stage26_2_authorized": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _build_stage26_3_config(config: dict[str, Any], *, seed: int, stage26_2_root: Path, repo_root: Path) -> dict[str, Any]:
    cfg = _load_json_template(config["stage26_3_base_config"], repo_root)
    cfg.update(
        {
            "stage26_8_seed": int(seed),
            "stage26_2_root": str(stage26_2_root),
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["eval_rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "coverage_denominator_mode": "main_coverable_cells",
            "coverage_denominator_source": COVERAGE_DENOMINATOR_SOURCE,
            "post_update_success_metric": SUCCESS_METRIC,
            "coverage_source": COVERAGE_SOURCE,
            "path_cost_source": PATH_COST_SOURCE,
            "synthetic_source_kind": SYNTHETIC_SOURCE_KIND,
            "action_space_type": ACTION_SPACE_TYPE,
            "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
            "stage21_5_timeout_seconds": 0.0,
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


def _efficiency_aggregate(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    seed_count = len(seed_rows)
    positive = [
        row
        for row in seed_rows
        if bool(row.get("efficiency_seed_passed")) and _float(row.get("main_coverage_per_100m_delta")) > 0.0
    ]
    negative = [
        row
        for row in seed_rows
        if not row.get("binding_or_safety_failure")
        and not row.get("update_failure")
        and not row.get("efficiency_seed_passed")
        and _float(row.get("main_coverage_per_100m_delta")) < 0.0
    ]
    clean = [
        row
        for row in seed_rows
        if not row.get("binding_or_safety_failure") and not row.get("execution_failure") and not row.get("update_failure")
    ]
    nonnegative = [
        row
        for row in clean
        if _float(row.get("main_coverage_per_100m_delta")) >= 0.0
    ]
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "seed_count": seed_count,
        "completed_seed_count": len([row for row in seed_rows if row.get("stage26_3_status")]),
        "clean_seed_count": len(clean),
        "positive_efficiency_seed_count": len(positive),
        "nonnegative_efficiency_seed_count": len(nonnegative),
        "negative_efficiency_seed_count": len(negative),
        "binding_or_safety_failure_seed_count": len([row for row in seed_rows if row.get("binding_or_safety_failure")]),
        "execution_failure_seed_count": len([row for row in seed_rows if row.get("execution_failure")]),
        "update_failure_seed_count": len([row for row in seed_rows if row.get("update_failure")]),
        "mean_main_coverage_per_100m_delta": _mean([row.get("main_coverage_per_100m_delta") for row in seed_rows]),
        "mean_main_final_coverage_delta": _mean([row.get("main_final_coverage_delta") for row in seed_rows]),
        "mean_main_coverage_auc_delta_diagnostic": _mean([row.get("main_coverage_auc_delta") for row in seed_rows]),
        "mean_hybrid_astar_path_cost_delta_diagnostic": _mean([row.get("hybrid_astar_path_cost_delta") for row in seed_rows]),
    }


def _scenario_efficiency_audit(output_root: Path, seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_rows = []
    for row in seed_rows:
        path = Path(row["stage26_3_root"]) / "s21_5" / stage26_3.stage21_5.SCENARIO_DELTA_FILE
        scenario_rows.extend(dict(item, seed=row["seed"]) for item in _read_jsonl_if_exists(path))
    scenario_rows_path = output_root / "xunce-stage26-8-scenario-efficiency-rows.jsonl"
    _write_jsonl(scenario_rows_path, scenario_rows)
    return {
        "schema_version": SCENARIO_AUDIT_SCHEMA_VERSION,
        "scenario_row_count": len(scenario_rows),
        "scenario_negative_efficiency_count": len([row for row in scenario_rows if _float(row.get("coverage_per_100m_delta")) < 0.0]),
        "scenario_rows_path": str(scenario_rows_path),
    }


def _boundary_safety_audit(seed_rows: list[dict[str, Any]], boundary_rejections: list[str]) -> dict[str, Any]:
    return {
        "schema_version": BOUNDARY_AUDIT_SCHEMA_VERSION,
        "boundary_rejections": boundary_rejections,
        "binding_or_safety_failure_seed_count": len([row for row in seed_rows if row.get("binding_or_safety_failure")]),
        "execution_failure_seed_count": len([row for row in seed_rows if row.get("execution_failure")]),
        "update_failure_seed_count": len([row for row in seed_rows if row.get("update_failure")]),
    }


def _route(
    *,
    boundary_rejections: list[str],
    input_rejections: list[str],
    seed_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections or not seed_rows:
        return ROUTE_INPUTS
    if int(aggregate.get("binding_or_safety_failure_seed_count") or 0) > 0:
        return ROUTE_BINDING_OR_SAFETY
    if int(aggregate.get("execution_failure_seed_count") or 0) > 0:
        return ROUTE_BINDING_OR_SAFETY
    if int(aggregate.get("update_failure_seed_count") or 0) > 0:
        return ROUTE_UPDATE
    majority = math.floor(len(seed_rows) / 2) + 1
    if int(aggregate.get("positive_efficiency_seed_count") or 0) >= majority:
        return ROUTE_STAGE26_9
    if int(aggregate.get("negative_efficiency_seed_count") or 0) >= majority:
        return ROUTE_CREDIT
    return ROUTE_EXPAND


def _input_rejections(stage26_7h_summary: dict[str, Any]) -> list[str]:
    if not stage26_7h_summary:
        return ["missing_stage26_7h_summary"]
    reasons: list[str] = []
    if stage26_7h_summary.get("schema_version") != "xunce-stage26-7h-summary/v1":
        reasons.append("stage26_7h_schema_version_mismatch")
    if stage26_7h_summary.get("stage_id") != "xunce-stage26-7h-repair-credit-post-update-eval-binding":
        reasons.append("stage26_7h_stage_id_mismatch")
    status = str(stage26_7h_summary.get("status") or "")
    route = str(stage26_7h_summary.get("next_required_change") or "")
    if (status, route) not in ACCEPTED_STAGE26_7H_ROUTES:
        reasons.append("stage26_7h_route_not_accepted_for_efficiency_pilot")
    if stage26_7h_summary.get("post_update_success_metric") != SUCCESS_METRIC:
        reasons.append("stage26_7h_post_update_success_metric_mismatch")
    for field in COUNT_FIELDS_REQUIRING_ZERO:
        if field not in stage26_7h_summary:
            reasons.append(f"stage26_7h_{field.removesuffix('_count')}_missing")
        elif int(stage26_7h_summary.get(field) or 0) != 0:
            reasons.append(f"stage26_7h_{field.removesuffix('_count')}_nonzero")
    if stage26_7h_summary.get("coverage_denominator_source") != COVERAGE_DENOMINATOR_SOURCE:
        reasons.append("stage26_7h_coverage_denominator_source_mismatch")
    if stage26_7h_summary.get("coverage_source") != COVERAGE_SOURCE:
        reasons.append("stage26_7h_coverage_source_mismatch")
    if stage26_7h_summary.get("path_cost_source") != PATH_COST_SOURCE:
        reasons.append("stage26_7h_path_cost_source_mismatch")
    if stage26_7h_summary.get("synthetic_source_kind") != SYNTHETIC_SOURCE_KIND:
        reasons.append("stage26_7h_synthetic_source_kind_mismatch")
    if stage26_7h_summary.get("action_space_type") != ACTION_SPACE_TYPE:
        reasons.append("stage26_7h_action_space_type_mismatch")
    if abs(_float(stage26_7h_summary.get("max_traversable_slope_deg")) - 30.0) > 1.0e-9:
        reasons.append("stage26_7h_max_traversable_slope_deg_mismatch")
    if not str(stage26_7h_summary.get("synthetic_terrain_hash") or "").strip():
        reasons.append("stage26_7h_synthetic_terrain_hash_missing")
    if not str(stage26_7h_summary.get("platform_contract_hash") or "").strip():
        reasons.append("stage26_7h_platform_contract_hash_missing")
    for field in BOUNDARY_FIELDS:
        if stage26_7h_summary.get(field) is True:
            reasons.append(f"stage26_7h_{field}")
    if _float(stage26_7h_summary.get("canary_traffic_fraction")) > 0.0:
        reasons.append("stage26_7h_canary_traffic_fraction")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0)) > 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _binding_or_safety_failure(summary: dict[str, Any]) -> bool:
    if not summary:
        return False
    return any(int(summary.get(field) or 0) != 0 for field in COUNT_FIELDS_REQUIRING_ZERO)


def _stage26_3_execution_failure(summary: dict[str, Any]) -> bool:
    if not summary:
        return False
    reason_codes = summary.get("stage21_5_execution_reason_codes") or []
    return (
        summary.get("stage21_5_runtime_blocker") is True
        or summary.get("next_required_change") == "rerun_stage26_3_required_inputs"
        or "stage21_5_runtime_timeout_or_process_failure" in reason_codes
    )


def _lineage_mismatch(summary: dict[str, Any]) -> bool:
    if not summary:
        return True
    return (
        summary.get("post_update_success_metric") != SUCCESS_METRIC
        or
        summary.get("coverage_denominator_source") != COVERAGE_DENOMINATOR_SOURCE
        or summary.get("coverage_source") != COVERAGE_SOURCE
        or summary.get("path_cost_source") != PATH_COST_SOURCE
        or summary.get("synthetic_source_kind") != SYNTHETIC_SOURCE_KIND
        or summary.get("action_space_type") != ACTION_SPACE_TYPE
        or abs(_float(summary.get("max_traversable_slope_deg")) - 30.0) > 1.0e-9
        or not str(summary.get("synthetic_terrain_hash") or "").strip()
        or not str(summary.get("platform_contract_hash") or "").strip()
    )


def _efficiency_seed_passed(summary: dict[str, Any]) -> bool:
    if not summary:
        return False
    final_delta = summary.get("final_coverage_delta")
    if final_delta is None:
        final_delta = summary.get("main_final_coverage_delta")
    return (
        summary.get("status") == "passed"
        and summary.get("next_required_change") == stage26_3.ROUTE_STAGE26_8
        and not _binding_or_safety_failure(summary)
        and not _lineage_mismatch(summary)
        and _float(final_delta) >= 0.0
        and _float(summary.get("coverage_per_100m_delta")) >= 0.0
    )


def _recommended_next_config(config: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "coverage_denominator_source": COVERAGE_DENOMINATOR_SOURCE,
        "post_update_success_metric": SUCCESS_METRIC,
        "recommended_seed_count": int(aggregate.get("seed_count") or len(config["seeds"])),
        "recommended_eval_rollout_steps": int(config["eval_rollout_steps"]),
        "recommended_collector_rollout_steps": int(config["collector_rollout_steps"]),
    }


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    defaults = {
        "stage26_7h_root": DEFAULT_STAGE26_7H_ROOT,
        "stage26_0_root": DEFAULT_STAGE26_0_ROOT,
        "stage26_1_base_config": "configs/xunce_stage26_1_synthetic_terrain_collector_smoke_v1.json",
        "stage26_2_base_config": "configs/xunce_stage26_2_synthetic_terrain_ppo_update_smoke_v1.json",
        "stage26_3_base_config": "configs/xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke_v1.json",
        "seeds": [260801, 260802, 260803],
        "run_stage26_chain": True,
        "required_scenario_count": 3,
        "collector_rollout_steps": 8,
        "eval_rollout_steps": 8,
        "dynamic_max_candidates_per_step": 36,
        "dynamic_proposal_pool_limit_per_step": 288,
        "theta_bin_count": 8,
        "theta_step_deg": 45,
        "sensor_fov_deg": 90.0,
        "sensor_range_cells": 2,
        "hybrid_astar_candidate_eval_workers": 4,
        "min_trainable_transition_count": 12,
        "synthetic_credit_mixture_probability": 1.0,
        "synthetic_credit_score_version": "path_efficiency_v2",
        "path_efficiency_max_cost_norm": 0.70,
        "epochs": 4,
        "learning_rate": 1.0e-5,
        "clip_ratio": 0.2,
        "policy_loss_coefficient": 1.0,
        "value_loss_coefficient": 0.02,
        "loss_scale": 0.25,
        "max_grad_norm": 1.0,
        "max_abs_approx_kl": 1.5,
        "stop_on_first_seed_blocker": False,
        "reuse_existing_seed_outputs": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    for field in ("stage26_7h_root", "stage26_0_root", "stage26_1_base_config", "stage26_2_base_config", "stage26_3_base_config"):
        config[field] = str(_resolve_path(Path(str(config[field])), repo_root))
    config["seeds"] = [int(seed) for seed in config.get("seeds", [])]
    if not config["seeds"]:
        raise ValueError("seeds must not be empty")
    if float(config.get("max_abs_approx_kl", 1.5)) > 1.5:
        raise ValueError("max_abs_approx_kl must not exceed 1.5")
    return config


def _load_json_template(path: str | Path, repo_root: Path) -> dict[str, Any]:
    resolved = _resolve_path(Path(path), repo_root)
    if not resolved.is_file():
        return {}
    return _read_json(resolved)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return _read_json(path)


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isfinite(parsed):
        return parsed
    return 0.0


def _mean(values: list[Any]) -> float:
    parsed = [_float(value) for value in values]
    if not parsed:
        return 0.0
    return sum(parsed) / len(parsed)


def _render_report(summary: dict[str, Any], seed_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage26.8 Synthetic Terrain Multi-Seed Coverage Efficiency Pilot",
        "",
        f"- status: `{summary['status']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- seed_count: `{summary['seed_count']}`",
        f"- positive_efficiency_seed_count: `{summary['positive_efficiency_seed_count']}`",
        f"- mean_main_coverage_per_100m_delta: `{summary['mean_main_coverage_per_100m_delta']}`",
        "",
        "Seed results:",
    ]
    for row in seed_rows:
        lines.append(
            f"- seed `{row.get('seed')}`: stage26_3={row.get('stage26_3_status')}, "
            f"per100m={row.get('main_coverage_per_100m_delta')}, "
            f"final={row.get('main_final_coverage_delta')}, auc_diagnostic={row.get('main_coverage_auc_delta')}"
        )
    lines.extend(
        [
            "",
            "AUC and total path cost are diagnostic only in this stage. This bounded pilot does not publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
