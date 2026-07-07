from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_xunce_high_fidelity_exploration_coverage_comparison import (  # noqa: E402
    run_xunce_high_fidelity_exploration_coverage_comparison,
)
import xunce_artifact_io as artifact_io
from xunce_hybrid_astar_candidate_path_cost import (  # noqa: E402
    CANDIDATE_REACHABILITY_GATE_SOURCE,
    CANDIDATE_REACHABILITY_PROVENANCE_SCHEMA_VERSION,
    PATH_COST_SOURCE as HYBRID_ASTAR_PATH_COST_SOURCE,
)
from xunce_artifact_paths import (
    STAGE21_1_SUMMARY,
    STAGE21_3_SUMMARY,
    STAGE21_4_CHECKPOINT,
    STAGE21_4_ROUTING,
    STAGE21_4_SUMMARY,
    STAGE21_5_DELTA,
    STAGE21_5_MANIFEST,
    STAGE21_5_RESULTS,
    STAGE21_5_ROUTING,
    STAGE21_5_SCENARIO_DELTA,
    STAGE21_5_SELECTED_POSE_EVIDENCE_AUDIT,
    STAGE21_5_SUMMARY,
    artifact_path,
    read_json_artifact,
    write_json_artifact,
    write_jsonl_artifact,
)


CONFIG_SCHEMA_VERSION = "xunce-stage21-5-post-update-offline-trajectory-evaluation-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage21-5-post-update-evaluation-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage21-5-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage21-5-manifest/v1"
CANDIDATE_REACHABILITY_GATE_LEGACY = "legacy_action_mask_validation/v1"
SELECTED_POSE_EVIDENCE_AUDIT_EXCLUDED_TERMINALS = {
    "candidate_generation_exhausted",
    "no_valid_action",
}

DEFAULT_CONFIG = "configs/xunce_stage21_5_post_update_offline_trajectory_evaluation_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/"
    "outputs/path_feedback_batch_xunce_stage21_5_post_update_offline_trajectory_evaluation_v1"
)

SUMMARY_FILE = "xunce-stage21-5-post-update-evaluation-summary.json"
RESULTS_FILE = "xunce-stage21-5-policy-evaluation-results.json"
DELTA_FILE = "xunce-stage21-5-trajectory-delta.json"
SCENARIO_DELTA_FILE = "xunce-stage21-5-scenario-trajectory-delta.jsonl"
ROUTING_FILE = "xunce-stage21-5-next-stage-routing.json"
REPORT_FILE = "xunce-stage21-5-report.md"
MANIFEST_FILE = "xunce-stage21-5-manifest.json"

ROUTE_BOUNDARY = "resolve_stage21_5_post_update_evaluation_boundary_rejections"
ROUTE_INPUTS = "rerun_stage21_5_required_inputs"
ROUTE_EXECUTION = "repair_stage21_5_offline_evaluation_execution"
ROUTE_PRE_UNREACHABLE = "repair_stage21_5_pre_policy_unreachable_baseline"
ROUTE_POST_UNREACHABLE = "repair_stage21_5_post_policy_unreachable_regression"
ROUTE_HARD_RISK = "repair_stage21_5_hard_risk_regression"
ROUTE_REPAIR = "repair_stage21_5_reward_collector_advantage_or_horizon"
ROUTE_STAGE21_6 = "implement_stage21_6_multi_seed_ppo_pilot"
SUCCESS_METRIC_DEFAULT = "coverage_auc_and_final/v1"
SUCCESS_METRIC_MAIN_COVERABLE_EFFICIENCY = "main_coverable_coverage_efficiency/v1"

ACCEPTED_STAGE21_4_ROUTES = {
    "implement_stage21_5_single_seed_ppo_pilot",
    "implement_stage21_5_post_update_offline_trajectory_evaluation",
}

FORBIDDEN_TRUE_FIELDS = (
    "stage21_5_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 21.5 post-update offline trajectory evaluation.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    try:
        summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
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
                "pre_final_coverage_rate_mean": summary["pre_final_coverage_rate_mean"],
                "post_final_coverage_rate_mean": summary["post_final_coverage_rate_mean"],
                "final_coverage_rate_delta": summary["final_coverage_rate_delta"],
                "post_hard_risk_violation_count": summary["post_hard_risk_violation_count"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    config_path = _resolve_path(config_path, repo_root)
    config = _load_config(config_path, repo_root=repo_root)
    output_root = _resolve_path(output_root, repo_root)
    artifact_io.make_dirs(output_root)

    boundary_reasons = _boundary_rejections(config)
    input_reasons = _input_rejections(config)
    stage21_4_summary: dict[str, Any] = {}
    stage21_4_checkpoint_audit: dict[str, Any] = {}
    pre_root = Path(config["pre_ppo_evaluation_root"]) if config.get("pre_ppo_evaluation_root") else output_root / "pre_ppo_xunce"
    post_root = Path(config["post_ppo_evaluation_root"]) if config.get("post_ppo_evaluation_root") else output_root / "post_ppo_xunce"
    pre_eval_summary: dict[str, Any] = {}
    post_eval_summary: dict[str, Any] = {}
    pre_episodes: list[dict[str, Any]] = []
    post_episodes: list[dict[str, Any]] = []
    pre_evidence_audit: dict[str, Any] = _empty_selected_pose_evidence_audit("pre")
    post_evidence_audit: dict[str, Any] = _empty_selected_pose_evidence_audit("post")

    if not boundary_reasons and not input_reasons:
        stage21_4_root = Path(config["stage21_4_tiny_ppo_update_smoke_root"])
        stage21_4_summary, _ = read_json_artifact(stage21_4_root, STAGE21_4_SUMMARY)
        stage21_4_checkpoint_audit, _ = read_json_artifact(stage21_4_root, STAGE21_4_CHECKPOINT)
        stage21_4_routing, _ = read_json_artifact(stage21_4_root, STAGE21_4_ROUTING)
        input_reasons.extend(_stage21_4_rejections(stage21_4_summary, stage21_4_checkpoint_audit, stage21_4_routing))
        theta_head_init_seed = _resolve_continuous_theta_head_init_seed(
            config,
            stage21_4_summary=stage21_4_summary,
            repo_root=repo_root,
        )
        if theta_head_init_seed is not None:
            config["continuous_theta_head_init_seed"] = int(theta_head_init_seed)

    if not boundary_reasons and not input_reasons:
        if config["execute_high_fidelity_evaluations"]:
            artifact_io.make_dirs(pre_root)
            artifact_io.make_dirs(post_root)
            _run_high_fidelity_eval(
                config,
                repo_root=repo_root,
                output_root=pre_root,
                checkpoint_path=Path(stage21_4_summary["source_xunce_candidate_checkpoint"]),
                label="pre",
            )
            _run_high_fidelity_eval(
                config,
                repo_root=repo_root,
                output_root=post_root,
                checkpoint_path=Path(stage21_4_summary["experimental_checkpoint_path"]),
                label="post",
            )
        pre_eval_summary, pre_episodes = _read_evaluation_root(pre_root)
        post_eval_summary, post_episodes = _read_evaluation_root(post_root)
        pre_evidence_audit = _selected_pose_evidence_audit(pre_root, label="pre", config=config)
        post_evidence_audit = _selected_pose_evidence_audit(post_root, label="post", config=config)

    execution_reasons = _evaluation_execution_rejections(pre_eval_summary, post_eval_summary, pre_episodes, post_episodes, config)
    execution_reasons.extend(_episode_alignment_rejections(pre_episodes, post_episodes, config))
    execution_reasons.extend(_selected_pose_evidence_rejections(pre_evidence_audit, post_evidence_audit))
    pre_metrics = _policy_metrics(pre_eval_summary, pre_episodes, policy_name="xunce")
    post_metrics = _policy_metrics(post_eval_summary, post_episodes, policy_name="xunce")
    delta = _trajectory_delta(pre_metrics, post_metrics)
    scenario_delta_rows = _scenario_delta_rows(pre_episodes, post_episodes)
    status = "passed"
    route = ROUTE_STAGE21_6
    blocking = list(boundary_reasons + input_reasons + execution_reasons)
    if boundary_reasons:
        status = "failed"
        route = ROUTE_BOUNDARY
    elif input_reasons:
        status = "failed"
        route = ROUTE_INPUTS
    elif "post_unreachable_selected_count_nonzero" in execution_reasons:
        status = "failed"
        route = ROUTE_POST_UNREACHABLE
    elif "post_selected_reachability_provenance_invalid_count_nonzero" in execution_reasons:
        status = "failed"
        route = ROUTE_POST_UNREACHABLE
    elif "post_selected_reachability_provenance_missing" in execution_reasons:
        status = "failed"
        route = ROUTE_POST_UNREACHABLE
    elif "pre_unreachable_selected_count_nonzero" in execution_reasons:
        status = "failed"
        route = ROUTE_PRE_UNREACHABLE
    elif "pre_selected_reachability_provenance_invalid_count_nonzero" in execution_reasons:
        status = "failed"
        route = ROUTE_PRE_UNREACHABLE
    elif "pre_selected_reachability_provenance_missing" in execution_reasons:
        status = "failed"
        route = ROUTE_PRE_UNREACHABLE
    elif execution_reasons:
        status = "failed"
        route = ROUTE_EXECUTION
    elif _hard_risk_or_safety_boundary_regressed(pre_metrics, post_metrics, delta, scenario_delta_rows):
        status = "failed"
        route = ROUTE_HARD_RISK
        blocking.append("post_ppo_hard_risk_or_safety_boundary_regression_detected")
    elif _regressed(delta, scenario_delta_rows, config):
        status = "failed"
        route = ROUTE_REPAIR
        blocking.append(_coverage_regression_reason(config))

    return _write_outputs(
        config=config,
        config_path=config_path,
        output_root=output_root,
        pre_root=pre_root,
        post_root=post_root,
        stage21_4_summary=stage21_4_summary,
        pre_eval_summary=pre_eval_summary,
        post_eval_summary=post_eval_summary,
        pre_metrics=pre_metrics,
        post_metrics=post_metrics,
        delta=delta,
        scenario_delta_rows=scenario_delta_rows,
        pre_evidence_audit=pre_evidence_audit,
        post_evidence_audit=post_evidence_audit,
        status=status,
        route=route,
        blocking_reason_codes=blocking,
    )


def _run_high_fidelity_eval(
    config: dict[str, Any],
    *,
    repo_root: Path,
    output_root: Path,
    checkpoint_path: Path,
    label: str,
) -> None:
    overrides = {
        "xunce_candidate_checkpoint": str(checkpoint_path),
        "required_scenario_count": int(config["required_scenario_count"]),
        "rollout_steps": int(config["rollout_steps"]),
        "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
        "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
        "dynamic_validation_work_root": str(output_root / "_xunce_dynamic_validation_work"),
        "include_oracle_baselines": bool(config["include_oracle_baselines"]),
        "xunce_only_evaluation": bool(config["xunce_only_evaluation"]),
        "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
        "continuous_theta_head_init_seed": config.get("continuous_theta_head_init_seed"),
        "include_roi_weighted_coverage": True,
        "include_canonical_reward_rerank_oracle": bool(config["include_canonical_reward_rerank_oracle"]),
        "canonical_reward_rerank_profile": str(config["canonical_reward_rerank_profile"])
        if config["include_canonical_reward_rerank_oracle"]
        else None,
        "emit_candidate_metric_audit": bool(config["emit_candidate_metric_audit"]),
    }
    for key in (
        "hybrid_astar_pose_path_cost_enabled",
        "candidate_reachability_gate_source",
        "candidate_reachability_max_theta_proposals_per_candidate",
        "candidate_reachability_theta_proposal_policy",
        "hybrid_astar_planning_grid_source",
        "planner_grid_resolution_m",
        "hybrid_astar_closed_key_xy_resolution_m",
        "hybrid_astar_goal_position_tolerance_m",
        "hybrid_astar_goal_theta_tolerance_deg",
        "hybrid_astar_max_iterations",
        "hybrid_astar_primitive_duration_s",
        "hybrid_astar_integration_dt_s",
        "hybrid_astar_max_speed_mps",
        "hybrid_astar_max_angular_speed_degps",
    ):
        if key in config:
            overrides[key] = config.get(key)
    run_xunce_high_fidelity_exploration_coverage_comparison(
        config_path=Path(config["high_fidelity_config"]),
        output_root=output_root,
        repo_root=repo_root,
        config_overrides={key: value for key, value in overrides.items() if value is not None},
    )


def _resolve_continuous_theta_head_init_seed(
    config: dict[str, Any],
    *,
    stage21_4_summary: dict[str, Any],
    repo_root: Path,
) -> int | None:
    explicit = _nonnegative_int_or_none(config.get("continuous_theta_head_init_seed"))
    if explicit is not None:
        return explicit
    summary_seed = _nonnegative_int_or_none(stage21_4_summary.get("continuous_theta_head_init_seed"))
    if summary_seed is not None:
        return summary_seed
    stage21_3_root_raw = stage21_4_summary.get("stage21_3_ppo_batch_validation_root")
    if not stage21_3_root_raw:
        return None
    stage21_3_root = _resolve_path(Path(str(stage21_3_root_raw)), repo_root)
    try:
        stage21_3_summary, _ = read_json_artifact(stage21_3_root, STAGE21_3_SUMMARY)
    except FileNotFoundError:
        return None
    stage21_1_root_raw = stage21_3_summary.get("stage21_1_collector_root")
    if not stage21_1_root_raw:
        return None
    stage21_1_root = _resolve_path(Path(str(stage21_1_root_raw)), repo_root)
    try:
        stage21_1_summary, _ = read_json_artifact(stage21_1_root, STAGE21_1_SUMMARY)
    except FileNotFoundError:
        return None
    return _nonnegative_int_or_none(stage21_1_summary.get("sampling_seed"))


def _nonnegative_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric >= 0 else None


def _policy_metrics(summary: dict[str, Any], episodes: list[dict[str, Any]], *, policy_name: str) -> dict[str, Any]:
    rows = [row for row in episodes if row.get("policy") == policy_name]
    return {
        "policy": policy_name,
        "episode_count": len(rows),
        "summary_status": summary.get("status"),
        "true_model_inference_executed": bool(summary.get("true_model_inference_executed")),
        "checkpoint_loaded": bool(summary.get("xunce_checkpoint_loaded")),
        "final_coverage_rate_mean": _mean([row.get("final_coverage_rate") for row in rows]),
        "final_coverage_rate_capped_mean": _mean([row.get("final_coverage_rate_capped") for row in rows]),
        "coverage_curve_auc_mean": _mean([row.get("coverage_curve_auc") for row in rows]),
        "coverage_curve_auc_capped_mean": _mean([row.get("coverage_curve_auc_capped") for row in rows]),
        "coverage_return_mean": _mean([row.get("coverage_return") for row in rows]),
        "new_covered_cell_count_mean": _mean([row.get("new_covered_cell_count") for row in rows]),
        "path_cost_total_m_mean": _mean([row.get("path_cost_total_m") for row in rows]),
        "coverage_per_100m_mean": _mean([row.get("coverage_per_100m") for row in rows]),
        "soft_risk_exposure_total_mean": _mean([row.get("soft_risk_exposure_total") for row in rows]),
        "hard_risk_violation_count": int(sum(_int_value(row.get("hard_risk_violation_count")) for row in rows)),
        "model_inference_failure_count": int(sum(_int_value(row.get("model_inference_failure_count")) for row in rows)),
        "mask_violation_count": int(sum(_int_value(row.get("mask_violation_count")) for row in rows)),
        "unreachable_selected_count": int(sum(_int_value(row.get("unreachable_selected_count")) for row in rows)),
        "path_planning_failure_count": int(sum(_int_value(row.get("path_planning_failure_count")) for row in rows)),
        "open_grid_fallback_count": int(sum(_int_value(row.get("open_grid_fallback_count")) for row in rows)),
        "safety_boundary_violation_count": int(
            sum(
                _int_value(row.get("hard_risk_violation_count"))
                + _int_value(row.get("model_inference_failure_count"))
                + _int_value(row.get("mask_violation_count"))
                + _int_value(row.get("unreachable_selected_count"))
                + _int_value(row.get("path_planning_failure_count"))
                + _int_value(row.get("open_grid_fallback_count"))
                for row in rows
            )
        ),
        "candidate_generation_exhausted_count": int(sum(_int_value(row.get("candidate_generation_exhausted_count")) for row in rows)),
    }


def _trajectory_delta(pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "final_coverage_rate_mean",
        "final_coverage_rate_capped_mean",
        "coverage_curve_auc_mean",
        "coverage_curve_auc_capped_mean",
        "coverage_return_mean",
        "new_covered_cell_count_mean",
        "path_cost_total_m_mean",
        "coverage_per_100m_mean",
        "soft_risk_exposure_total_mean",
    )
    delta = {"schema_version": "xunce-stage21-5-trajectory-delta/v1"}
    for field in fields:
        delta[f"{field}_delta"] = _nullable_delta(post.get(field), pre.get(field))
    delta["hard_risk_violation_count_delta"] = int(post.get("hard_risk_violation_count", 0)) - int(pre.get("hard_risk_violation_count", 0))
    for field in (
        "model_inference_failure_count",
        "mask_violation_count",
        "unreachable_selected_count",
        "path_planning_failure_count",
        "open_grid_fallback_count",
        "safety_boundary_violation_count",
        "candidate_generation_exhausted_count",
    ):
        delta[f"{field}_delta"] = int(post.get(field, 0)) - int(pre.get(field, 0))
    return delta


def _scenario_delta_rows(pre_episodes: list[dict[str, Any]], post_episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pre_rows = _xunce_episode_map(pre_episodes)
    post_rows = _xunce_episode_map(post_episodes)
    rows: list[dict[str, Any]] = []
    for scenario_id in sorted(set(pre_rows) & set(post_rows)):
        if scenario_id.startswith("__duplicate__"):
            continue
        pre = pre_rows[scenario_id]
        post = post_rows[scenario_id]
        row = {
            "schema_version": "xunce-stage21-5-scenario-trajectory-delta/v1",
            "scenario_id": scenario_id,
            "pre_final_coverage_rate": pre.get("final_coverage_rate"),
            "post_final_coverage_rate": post.get("final_coverage_rate"),
            "final_coverage_rate_delta": _nullable_delta(post.get("final_coverage_rate"), pre.get("final_coverage_rate")),
            "final_coverage_rate_capped_delta": _nullable_delta(post.get("final_coverage_rate_capped"), pre.get("final_coverage_rate_capped")),
            "coverage_curve_auc_delta": _nullable_delta(post.get("coverage_curve_auc"), pre.get("coverage_curve_auc")),
            "coverage_curve_auc_capped_delta": _nullable_delta(post.get("coverage_curve_auc_capped"), pre.get("coverage_curve_auc_capped")),
            "new_covered_cell_count_delta": _nullable_delta(post.get("new_covered_cell_count"), pre.get("new_covered_cell_count")),
            "path_cost_total_m_delta": _nullable_delta(post.get("path_cost_total_m"), pre.get("path_cost_total_m")),
            "coverage_per_100m_delta": _nullable_delta(post.get("coverage_per_100m"), pre.get("coverage_per_100m")),
            "soft_risk_exposure_total_delta": _nullable_delta(
                post.get("soft_risk_exposure_total"),
                pre.get("soft_risk_exposure_total"),
            ),
        }
        for field in (
            "hard_risk_violation_count",
            "model_inference_failure_count",
            "mask_violation_count",
            "unreachable_selected_count",
            "path_planning_failure_count",
            "open_grid_fallback_count",
            "candidate_generation_exhausted_count",
        ):
            row[f"pre_{field}"] = _int_value(pre.get(field))
            row[f"post_{field}"] = _int_value(post.get(field))
            row[f"{field}_delta"] = _int_value(post.get(field)) - _int_value(pre.get(field))
        row["post_safety_boundary_violation_count"] = (
            row["post_hard_risk_violation_count"]
            + row["post_model_inference_failure_count"]
            + row["post_mask_violation_count"]
            + row["post_unreachable_selected_count"]
            + row["post_path_planning_failure_count"]
            + row["post_open_grid_fallback_count"]
        )
        rows.append(row)
    return rows


def _scenario_regression_count(rows: list[dict[str, Any]], config: dict[str, Any]) -> int:
    min_delta = float(config["min_post_update_coverage_delta"])
    fields = _scenario_regression_fields(config)
    count = 0
    for row in rows:
        if any(_finite(row.get(field)) is None or float(_finite(row.get(field))) < -min_delta for field in fields):
            count += 1
    return count


def _scenario_safety_boundary_regression_count(rows: list[dict[str, Any]]) -> int:
    fields = (
        "hard_risk_violation_count_delta",
        "model_inference_failure_count_delta",
        "mask_violation_count_delta",
        "unreachable_selected_count_delta",
        "path_planning_failure_count_delta",
        "open_grid_fallback_count_delta",
    )
    count = 0
    for row in rows:
        if int(row.get("post_safety_boundary_violation_count", 0)) > 0:
            count += 1
        elif any(int(row.get(field, 0)) > 0 for field in fields):
            count += 1
    return count


def _regressed(delta: dict[str, Any], scenario_delta_rows: list[dict[str, Any]], config: dict[str, Any]) -> bool:
    min_delta = float(config["min_post_update_coverage_delta"])
    fields = _aggregate_regression_fields(config)
    values = [_finite(delta.get(field)) for field in fields]
    if any(value is None for value in values):
        return True
    if any(float(value) < -min_delta for value in values if value is not None):
        return True
    return _scenario_regression_count(scenario_delta_rows, config) > 0


def _aggregate_regression_fields(config: dict[str, Any]) -> tuple[str, ...]:
    if _uses_main_coverable_efficiency_metric(config):
        return (
            "final_coverage_rate_mean_delta",
            "final_coverage_rate_capped_mean_delta",
            "coverage_per_100m_mean_delta",
        )
    return (
        "final_coverage_rate_mean_delta",
        "final_coverage_rate_capped_mean_delta",
        "coverage_curve_auc_mean_delta",
        "coverage_curve_auc_capped_mean_delta",
    )


def _scenario_regression_fields(config: dict[str, Any]) -> tuple[str, ...]:
    if _uses_main_coverable_efficiency_metric(config):
        return ("coverage_per_100m_delta",)
    return (
        "final_coverage_rate_delta",
        "final_coverage_rate_capped_delta",
        "coverage_curve_auc_delta",
        "coverage_curve_auc_capped_delta",
    )


def _uses_main_coverable_efficiency_metric(config: dict[str, Any]) -> bool:
    return config.get("post_update_success_metric") == SUCCESS_METRIC_MAIN_COVERABLE_EFFICIENCY


def _coverage_regression_reason(config: dict[str, Any]) -> str:
    if _uses_main_coverable_efficiency_metric(config):
        return "post_ppo_main_coverable_coverage_efficiency_regressed"
    return "post_ppo_coverage_or_auc_regressed"


def _hard_risk_or_safety_boundary_regressed(
    pre: dict[str, Any],
    post: dict[str, Any],
    delta: dict[str, Any],
    scenario_delta_rows: list[dict[str, Any]],
) -> bool:
    if int(post.get("hard_risk_violation_count", 0)) > 0:
        return True
    if int(post.get("safety_boundary_violation_count", 0)) > 0:
        return True
    for field in (
        "model_inference_failure_count_delta",
        "mask_violation_count_delta",
        "unreachable_selected_count_delta",
        "path_planning_failure_count_delta",
        "open_grid_fallback_count_delta",
    ):
        if int(delta.get(field, 0)) > 0:
            return True
    return _scenario_safety_boundary_regression_count(scenario_delta_rows) > 0


def _evaluation_execution_rejections(
    pre_summary: dict[str, Any],
    post_summary: dict[str, Any],
    pre_episodes: list[dict[str, Any]],
    post_episodes: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    xunce_only = bool(config.get("xunce_only_evaluation", False))
    for label, summary, episodes in (("pre", pre_summary, pre_episodes), ("post", post_summary, post_episodes)):
        if not summary:
            reasons.append(f"{label}_evaluation_summary_missing")
            continue
        if not summary.get("xunce_checkpoint_loaded"):
            reasons.append(f"{label}_xunce_checkpoint_not_loaded")
        if not xunce_only and not summary.get("incumbent_checkpoint_loaded"):
            reasons.append(f"{label}_incumbent_checkpoint_not_loaded")
        if not summary.get("true_model_inference_executed"):
            reasons.append(f"{label}_true_model_inference_not_executed")
        scenario_count = len(_xunce_episode_map(episodes)) if xunce_only else int(summary.get("scenario_count", 0))
        if scenario_count < int(config["required_scenario_count"]):
            reasons.append(f"{label}_scenario_count_short")
        if int(summary.get("rollout_steps", 0)) != int(config["rollout_steps"]):
            reasons.append(f"{label}_rollout_steps_mismatch")
        if summary.get("proxy_selection_used") is True:
            reasons.append(f"{label}_proxy_selection_used")
        for field in (
            "model_inference_failure_count",
            "model_inference_mask_violation_count",
            "open_grid_fallback_count",
            "unreachable_selected_count",
            "path_planning_failure_count",
        ):
            if _int_value(summary.get(field)) > 0:
                reasons.append(f"{label}_{field}_nonzero")
    return reasons


def _empty_selected_pose_evidence_audit(label: str) -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage21-5-selected-pose-evidence-audit/v1",
        "label": label,
        "candidate_reachability_gate_source": CANDIDATE_REACHABILITY_GATE_LEGACY,
        "model_inference_row_count": 0,
        "selected_reachability_audited_count": 0,
        "selected_reachability_provenance_pass_count": 0,
        "selected_reachability_provenance_invalid_count": 0,
        "selected_reachability_provenance_missing_count": 0,
        "selected_reachability_provenance_source_mismatch_count": 0,
        "selected_reachability_planner_hash_mismatch_count": 0,
        "selected_reachability_candidate_set_hash_mismatch_count": 0,
        "selected_reachability_candidate_set_hash_conflict_count": 0,
        "selected_reachability_candidate_index_mismatch_count": 0,
        "selected_reachability_unreachable_count": 0,
        "reason_codes": [],
        "invalid_samples": [],
    }


def _selected_pose_evidence_audit(root: Path, *, label: str, config: dict[str, Any]) -> dict[str, Any]:
    rows = _read_jsonl(root / "xunce-exploration-coverage-model-inference.jsonl")
    gate_source = str(config.get("candidate_reachability_gate_source") or CANDIDATE_REACHABILITY_GATE_LEGACY)
    audit = _empty_selected_pose_evidence_audit(label)
    audit["candidate_reachability_gate_source"] = gate_source
    audit["model_inference_row_count"] = len(rows)
    if gate_source != CANDIDATE_REACHABILITY_GATE_SOURCE:
        return audit
    xunce_rows = [
        row
        for row in rows
        if row.get("policy") == "xunce" and not _selected_pose_evidence_audit_excluded_terminal(row)
    ]
    audit["selected_reachability_audited_count"] = len(xunce_rows)
    reason_counts: dict[str, int] = {}
    invalid_samples: list[dict[str, Any]] = []
    candidate_set_hash_conflict_count = 0
    for row in xunce_rows:
        if _selected_candidate_set_hash_conflict(row):
            candidate_set_hash_conflict_count += 1
        row_reasons = _selected_pose_evidence_row_reasons(row)
        if row_reasons:
            audit["selected_reachability_provenance_invalid_count"] += 1
            for reason in row_reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if len(invalid_samples) < 5:
                invalid_samples.append(_selected_pose_evidence_invalid_sample(row, row_reasons))
        else:
            audit["selected_reachability_provenance_pass_count"] += 1
    audit["selected_reachability_provenance_missing_count"] = reason_counts.get("selected_reachability_provenance_missing", 0)
    audit["selected_reachability_provenance_source_mismatch_count"] = reason_counts.get(
        "selected_reachability_provenance_source_mismatch",
        0,
    )
    audit["selected_reachability_planner_hash_mismatch_count"] = reason_counts.get(
        "selected_reachability_planner_hash_mismatch",
        0,
    )
    audit["selected_reachability_candidate_set_hash_mismatch_count"] = reason_counts.get(
        "selected_reachability_candidate_set_hash_mismatch",
        0,
    )
    audit["selected_reachability_candidate_set_hash_conflict_count"] = candidate_set_hash_conflict_count
    audit["selected_reachability_candidate_index_mismatch_count"] = reason_counts.get(
        "selected_reachability_candidate_index_mismatch",
        0,
    )
    audit["selected_reachability_unreachable_count"] = reason_counts.get(
        "selected_reachability_provenance_unreachable",
        0,
    )
    audit["reason_codes"] = _unique_sorted(list(reason_counts))
    audit["invalid_samples"] = invalid_samples
    return audit


def _selected_pose_evidence_audit_excluded_terminal(row: dict[str, Any]) -> bool:
    skipped_reason = str(row.get("inference_skipped_reason") or "")
    if skipped_reason not in SELECTED_POSE_EVIDENCE_AUDIT_EXCLUDED_TERMINALS:
        return False
    if row.get("true_model_inference_executed") is True:
        return False
    if row.get("selected_action_index") is not None:
        return False
    if skipped_reason == "no_valid_action" and row.get("has_valid_action") is not None and row.get("has_valid_action") is not False:
        return False
    return True


def _selected_pose_evidence_invalid_sample(row: dict[str, Any], row_reasons: list[str]) -> dict[str, Any]:
    provenance = row.get("selected_candidate_reachability_provenance")
    selected_base = row.get("selected_base_candidate_index")
    if selected_base is None:
        selected_base = row.get("base_candidate_index")
    preferred_hash_field, preferred_hash_value = _selected_candidate_set_hash_binding(row)
    return {
        "scenario_id": row.get("scenario_id"),
        "step_index": row.get("step_index"),
        "selected_action_index": row.get("selected_action_index"),
        "row_selected_action_index": _int_or_none(row.get("selected_action_index")),
        "row_selected_base_candidate_index": _int_or_none(selected_base),
        "row_selected_base_candidate_set_hash": row.get("selected_base_candidate_set_hash"),
        "row_base_candidate_set_hash": row.get("base_candidate_set_hash"),
        "provenance_candidate_index": _int_or_none(provenance.get("candidate_index"))
        if isinstance(provenance, dict)
        else None,
        "candidate_set_hash_match": _candidate_set_hash_match(row, provenance),
        "candidate_set_hash_preferred_field": preferred_hash_field,
        "candidate_set_hash_preferred_value": preferred_hash_value,
        "candidate_set_hash_conflict": _selected_candidate_set_hash_conflict(row),
        "planner_hash_match": _planner_hash_match(row, provenance),
        "reason_codes": row_reasons,
    }


def _candidate_set_hash_match(row: dict[str, Any], provenance: Any) -> bool | None:
    if not isinstance(provenance, dict):
        return None
    provenance_hash = provenance.get("candidate_set_hash")
    _, row_hash = _selected_candidate_set_hash_binding(row)
    if row_hash is None or provenance_hash is None or not str(provenance_hash):
        return None
    return row_hash == str(provenance_hash)


def _selected_candidate_set_hash_binding(row: dict[str, Any]) -> tuple[str | None, str | None]:
    for field in ("selected_base_candidate_set_hash", "base_candidate_set_hash", "candidate_set_hash"):
        value = row.get(field)
        if value is None:
            continue
        text = str(value)
        if text:
            return field, text
    return None, None


def _selected_candidate_set_hash_conflict(row: dict[str, Any]) -> bool:
    values: list[str] = []
    for field in ("selected_base_candidate_set_hash", "base_candidate_set_hash", "candidate_set_hash"):
        value = row.get(field)
        if value is None:
            continue
        text = str(value)
        if text and text not in values:
            values.append(text)
    return len(values) > 1


def _planner_hash_match(row: dict[str, Any], provenance: Any) -> bool | None:
    if not isinstance(provenance, dict):
        return None
    provenance_hash = provenance.get("planner_config_hash")
    if provenance_hash is None:
        return None
    row_hashes = [
        str(row[field])
        for field in ("selected_reachability_planner_config_hash", "candidate_reachability_planner_config_hash")
        if row.get(field) is not None
    ]
    if not row_hashes:
        return None
    return all(row_hash == str(provenance_hash) for row_hash in row_hashes)


def _selected_pose_evidence_row_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    provenance = row.get("selected_candidate_reachability_provenance")
    if not isinstance(provenance, dict):
        return ["selected_reachability_provenance_missing"]
    if provenance.get("schema_version") != CANDIDATE_REACHABILITY_PROVENANCE_SCHEMA_VERSION:
        reasons.append("selected_reachability_provenance_schema_mismatch")
    if provenance.get("source") != CANDIDATE_REACHABILITY_GATE_SOURCE:
        reasons.append("selected_reachability_provenance_source_mismatch")
    if provenance.get("backend") != HYBRID_ASTAR_PATH_COST_SOURCE:
        reasons.append("selected_reachability_provenance_backend_mismatch")
    if provenance.get("reachable") is not True:
        reasons.append("selected_reachability_provenance_unreachable")
    if _finite(provenance.get("path_cost")) is None:
        reasons.append("selected_reachability_provenance_path_cost_missing")
    if not str(provenance.get("pose_path_hash") or "").strip():
        reasons.append("selected_reachability_provenance_pose_path_hash_missing")
    planner_hash = str(provenance.get("planner_config_hash") or "")
    if not planner_hash:
        reasons.append("selected_reachability_provenance_planner_hash_missing")
    for field in ("selected_reachability_planner_config_hash", "candidate_reachability_planner_config_hash"):
        row_hash = row.get(field)
        if row_hash is not None and str(row_hash) != planner_hash:
            reasons.append("selected_reachability_planner_hash_mismatch")
            break
    provenance_candidate_set_hash = provenance.get("candidate_set_hash")
    if provenance_candidate_set_hash is None or not str(provenance_candidate_set_hash):
        reasons.append("selected_reachability_candidate_set_hash_missing")
    elif _selected_candidate_set_hash_binding(row)[1] is not None and not _candidate_set_hash_match(row, provenance):
        reasons.append("selected_reachability_candidate_set_hash_mismatch")
    row_action_index = _int_or_none(row.get("selected_action_index"))
    provenance_candidate_index = _int_or_none(provenance.get("candidate_index"))
    if provenance_candidate_index is None:
        reasons.append("selected_reachability_candidate_index_missing")
    elif row_action_index is not None and row_action_index != provenance_candidate_index:
        reasons.append("selected_reachability_candidate_index_mismatch")
    row_theta = _finite(row.get("candidate_theta_deg"))
    provenance_theta = _finite(provenance.get("candidate_theta_deg"))
    if provenance_theta is None:
        reasons.append("selected_reachability_theta_missing")
    elif row_theta is not None and abs(row_theta - provenance_theta) > 1.0e-6:
        reasons.append("selected_reachability_theta_mismatch")
    return _unique_sorted(reasons)


def _selected_pose_evidence_rejections(
    pre_audit: dict[str, Any],
    post_audit: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    for label, audit in (("pre", pre_audit), ("post", post_audit)):
        if audit.get("candidate_reachability_gate_source") != CANDIDATE_REACHABILITY_GATE_SOURCE:
            continue
        if _int_value(audit.get("selected_reachability_audited_count")) <= 0:
            reasons.append(f"{label}_selected_reachability_provenance_missing")
        if _int_value(audit.get("selected_reachability_provenance_invalid_count")) > 0:
            reasons.append(f"{label}_selected_reachability_provenance_invalid_count_nonzero")
    return reasons


def _episode_alignment_rejections(pre_episodes: list[dict[str, Any]], post_episodes: list[dict[str, Any]], config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    pre_rows = _xunce_episode_map(pre_episodes)
    post_rows = _xunce_episode_map(post_episodes)
    if len(pre_rows) < int(config["required_scenario_count"]):
        reasons.append("pre_xunce_episode_count_short")
    if len(post_rows) < int(config["required_scenario_count"]):
        reasons.append("post_xunce_episode_count_short")
    if set(pre_rows) != set(post_rows):
        reasons.append("pre_post_xunce_scenario_id_mismatch")
    return reasons


def _xunce_episode_map(episodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in episodes:
        if row.get("policy") != "xunce":
            continue
        scenario_id = str(row.get("scenario_id", ""))
        if not scenario_id or scenario_id in rows:
            rows[f"__duplicate__{scenario_id}"] = row
            continue
        rows[scenario_id] = row
    return rows


def _stage21_4_rejections(summary: dict[str, Any], checkpoint_audit: dict[str, Any], routing: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if summary.get("status") != "passed":
        reasons.append("stage21_4_status_not_passed")
    if summary.get("next_required_change") not in ACCEPTED_STAGE21_4_ROUTES:
        reasons.append("stage21_4_route_not_stage21_5")
    if routing.get("status") != summary.get("status"):
        reasons.append("stage21_4_routing_status_mismatch")
    if routing.get("next_required_change") != summary.get("next_required_change"):
        reasons.append("stage21_4_routing_next_required_change_mismatch")
    if not checkpoint_audit.get("checkpoint_reload_passed"):
        reasons.append("stage21_4_experimental_checkpoint_not_reloadable")
    metadata = checkpoint_audit.get("metadata") if isinstance(checkpoint_audit.get("metadata"), dict) else {}
    if metadata.get("experimental_only") is not True:
        reasons.append("stage21_4_checkpoint_not_experimental_only")
    for field in ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary"):
        if summary.get(field) is True or metadata.get(field) is True:
            reasons.append(f"stage21_4_{field}_true")
    if not summary.get("source_xunce_candidate_checkpoint") or not summary.get("experimental_checkpoint_path"):
        reasons.append("stage21_4_checkpoint_paths_missing")
    if not summary.get("experimental_checkpoint_sha256"):
        reasons.append("stage21_4_summary_experimental_checkpoint_sha256_missing")
    if not checkpoint_audit.get("experimental_checkpoint_path"):
        reasons.append("stage21_4_checkpoint_audit_path_missing")
    elif str(checkpoint_audit.get("experimental_checkpoint_path")) != str(summary.get("experimental_checkpoint_path")):
        reasons.append("stage21_4_checkpoint_audit_path_mismatch")
    reload_audit = checkpoint_audit.get("reload_audit") if isinstance(checkpoint_audit.get("reload_audit"), dict) else {}
    if not reload_audit.get("checkpoint_path"):
        reasons.append("stage21_4_checkpoint_reload_path_missing")
    elif str(reload_audit.get("checkpoint_path")) != str(summary.get("experimental_checkpoint_path")):
        reasons.append("stage21_4_checkpoint_reload_path_mismatch")
    if not checkpoint_audit.get("experimental_checkpoint_sha256"):
        reasons.append("stage21_4_checkpoint_audit_sha256_missing")
    elif checkpoint_audit.get("experimental_checkpoint_sha256") != summary.get("experimental_checkpoint_sha256"):
        reasons.append("stage21_4_checkpoint_audit_sha256_mismatch")
    if not reload_audit.get("checkpoint_sha256"):
        reasons.append("stage21_4_checkpoint_reload_sha256_missing")
    elif reload_audit.get("checkpoint_sha256") != summary.get("experimental_checkpoint_sha256"):
        reasons.append("stage21_4_checkpoint_reload_sha256_mismatch")
    return reasons


def _read_evaluation_root(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = _read_json(root / "xunce-exploration-coverage-comparison-summary.json")
    episodes = _read_jsonl(root / "xunce-exploration-coverage-episodes.jsonl")
    return summary, episodes


def _write_outputs(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    pre_root: Path,
    post_root: Path,
    stage21_4_summary: dict[str, Any],
    pre_eval_summary: dict[str, Any],
    post_eval_summary: dict[str, Any],
    pre_metrics: dict[str, Any],
    post_metrics: dict[str, Any],
    delta: dict[str, Any],
    scenario_delta_rows: list[dict[str, Any]],
    pre_evidence_audit: dict[str, Any],
    post_evidence_audit: dict[str, Any],
    status: str,
    route: str,
    blocking_reason_codes: list[str],
) -> dict[str, Any]:
    paths = {
        "summary": output_root / SUMMARY_FILE,
        "results": output_root / RESULTS_FILE,
        "delta": output_root / DELTA_FILE,
        "scenario_delta": output_root / SCENARIO_DELTA_FILE,
        "routing": output_root / ROUTING_FILE,
        "report": output_root / REPORT_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "selected_pose_evidence_audit": artifact_path(output_root, STAGE21_5_SELECTED_POSE_EVIDENCE_AUDIT),
    }
    results = {
        "schema_version": "xunce-stage21-5-policy-evaluation-results/v1",
        "pre_ppo_xunce": pre_metrics,
        "post_ppo_xunce": post_metrics,
        "pre_evaluation_root": str(pre_root),
        "post_evaluation_root": str(post_root),
    }
    write_json_artifact(output_root, STAGE21_5_RESULTS, results)
    write_json_artifact(output_root, STAGE21_5_DELTA, delta)
    write_jsonl_artifact(output_root, STAGE21_5_SCENARIO_DELTA, scenario_delta_rows)
    selected_pose_evidence_audit = {
        "schema_version": "xunce-stage21-5-selected-pose-evidence-audit-bundle/v1",
        "candidate_reachability_gate_source": config.get("candidate_reachability_gate_source"),
        "pre": pre_evidence_audit,
        "post": post_evidence_audit,
        "pre_selected_reachability_provenance_invalid_count": pre_evidence_audit.get(
            "selected_reachability_provenance_invalid_count",
            0,
        ),
        "post_selected_reachability_provenance_invalid_count": post_evidence_audit.get(
            "selected_reachability_provenance_invalid_count",
            0,
        ),
        "pre_selected_reachability_candidate_index_mismatch_count": pre_evidence_audit.get(
            "selected_reachability_candidate_index_mismatch_count",
            0,
        ),
        "post_selected_reachability_candidate_index_mismatch_count": post_evidence_audit.get(
            "selected_reachability_candidate_index_mismatch_count",
            0,
        ),
    }
    write_json_artifact(output_root, STAGE21_5_SELECTED_POSE_EVIDENCE_AUDIT, selected_pose_evidence_audit)
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "reason_codes": _unique_sorted(blocking_reason_codes),
        "stage21_5_authorized": False,
        "training_or_release_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    write_json_artifact(output_root, STAGE21_5_ROUTING, routing)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "next_required_change": route,
        "reason_codes": _unique_sorted(blocking_reason_codes),
        "blocking_reason_codes": _unique_sorted(blocking_reason_codes),
        "stage21_4_tiny_ppo_update_smoke_root": str(config["stage21_4_tiny_ppo_update_smoke_root"]),
        "stage21_4_status": stage21_4_summary.get("status"),
        "stage21_4_experimental_checkpoint_path": stage21_4_summary.get("experimental_checkpoint_path"),
        "execute_high_fidelity_evaluations": bool(config["execute_high_fidelity_evaluations"]),
        "required_scenario_count": int(config["required_scenario_count"]),
        "rollout_steps": int(config["rollout_steps"]),
        "pre_evaluation_root": str(pre_root),
        "post_evaluation_root": str(post_root),
        "pre_evaluation_status": pre_eval_summary.get("status"),
        "post_evaluation_status": post_eval_summary.get("status"),
        "pre_final_coverage_rate_mean": pre_metrics.get("final_coverage_rate_mean"),
        "post_final_coverage_rate_mean": post_metrics.get("final_coverage_rate_mean"),
        "final_coverage_rate_delta": delta.get("final_coverage_rate_mean_delta"),
        "final_coverage_rate_capped_delta": delta.get("final_coverage_rate_capped_mean_delta"),
        "coverage_curve_auc_delta": delta.get("coverage_curve_auc_mean_delta"),
        "coverage_curve_auc_capped_delta": delta.get("coverage_curve_auc_capped_mean_delta"),
        "new_covered_cell_count_delta": delta.get("new_covered_cell_count_mean_delta"),
        "path_cost_total_m_delta": delta.get("path_cost_total_m_mean_delta"),
        "coverage_per_100m_delta": delta.get("coverage_per_100m_mean_delta"),
        "post_update_success_metric": config.get("post_update_success_metric", SUCCESS_METRIC_DEFAULT),
        "continuous_theta_head_init_seed": config.get("continuous_theta_head_init_seed"),
        "candidate_reachability_gate_source": config.get("candidate_reachability_gate_source"),
        "selected_pose_evidence_audit": str(paths["selected_pose_evidence_audit"]),
        "pre_selected_reachability_audited_count": pre_evidence_audit.get(
            "selected_reachability_audited_count",
            0,
        ),
        "post_selected_reachability_audited_count": post_evidence_audit.get(
            "selected_reachability_audited_count",
            0,
        ),
        "pre_selected_reachability_provenance_pass_count": pre_evidence_audit.get(
            "selected_reachability_provenance_pass_count",
            0,
        ),
        "post_selected_reachability_provenance_pass_count": post_evidence_audit.get(
            "selected_reachability_provenance_pass_count",
            0,
        ),
        "pre_selected_reachability_provenance_invalid_count": pre_evidence_audit.get(
            "selected_reachability_provenance_invalid_count",
            0,
        ),
        "post_selected_reachability_provenance_invalid_count": post_evidence_audit.get(
            "selected_reachability_provenance_invalid_count",
            0,
        ),
        "pre_selected_reachability_candidate_index_mismatch_count": pre_evidence_audit.get(
            "selected_reachability_candidate_index_mismatch_count",
            0,
        ),
        "post_selected_reachability_candidate_index_mismatch_count": post_evidence_audit.get(
            "selected_reachability_candidate_index_mismatch_count",
            0,
        ),
        "soft_risk_exposure_total_delta": delta.get("soft_risk_exposure_total_mean_delta"),
        "post_hard_risk_violation_count": post_metrics.get("hard_risk_violation_count", 0),
        "pre_unreachable_selected_count": pre_metrics.get("unreachable_selected_count", 0),
        "post_unreachable_selected_count": post_metrics.get("unreachable_selected_count", 0),
        "post_safety_boundary_violation_count": post_metrics.get("safety_boundary_violation_count", 0),
        "safety_boundary_violation_count_delta": delta.get("safety_boundary_violation_count_delta"),
        "scenario_delta_row_count": len(scenario_delta_rows),
        "scenario_regression_count": _scenario_regression_count(scenario_delta_rows, config),
        "scenario_safety_boundary_regression_count": _scenario_safety_boundary_regression_count(scenario_delta_rows),
        "sample_count_too_low_for_performance_claim": bool(stage21_4_summary.get("sample_count_too_low_for_performance_claim", True)),
        "summary": str(paths["summary"]),
        "results": str(paths["results"]),
        "trajectory_delta": str(paths["delta"]),
        "scenario_trajectory_delta": str(paths["scenario_delta"]),
        "routing": str(paths["routing"]),
        "report": str(paths["report"]),
        "manifest": str(paths["manifest"]),
        "stage21_5_authorized": False,
        "training_or_release_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    write_json_artifact(output_root, STAGE21_5_SUMMARY, summary)
    artifact_io.write_text(paths["report"], _report_markdown(summary))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": summary["generated_at"],
        "config": str(config_path),
        "summary_status": status,
        "next_required_change": route,
        "artifacts": {key: str(path) for key, path in paths.items()},
        "pre_evaluation_root": str(pre_root),
        "post_evaluation_root": str(post_root),
        "selected_pose_evidence_audit": str(paths["selected_pose_evidence_audit"]),
    }
    write_json_artifact(output_root, STAGE21_5_MANIFEST, manifest)
    return summary


def _report_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage 21.5 Post-Update Offline Trajectory Evaluation",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- pre_final_coverage_rate_mean: `{summary['pre_final_coverage_rate_mean']}`",
            f"- post_final_coverage_rate_mean: `{summary['post_final_coverage_rate_mean']}`",
            f"- final_coverage_rate_delta: `{summary['final_coverage_rate_delta']}`",
            f"- coverage_curve_auc_delta: `{summary['coverage_curve_auc_delta']}`",
            f"- coverage_per_100m_delta: `{summary['coverage_per_100m_delta']}`",
            f"- post_update_success_metric: `{summary['post_update_success_metric']}`",
            f"- post_hard_risk_violation_count: `{summary['post_hard_risk_violation_count']}`",
            f"- pre_selected_reachability_provenance_invalid_count: `{summary['pre_selected_reachability_provenance_invalid_count']}`",
            f"- post_selected_reachability_provenance_invalid_count: `{summary['post_selected_reachability_provenance_invalid_count']}`",
            f"- scenario_regression_count: `{summary['scenario_regression_count']}`",
            f"- scenario_safety_boundary_regression_count: `{summary['scenario_safety_boundary_regression_count']}`",
            "",
            "Stage 21.5 is an offline trajectory evaluation only. It does not run a PPO update, publish a checkpoint, replace the default policy, connect a real executor, or start canary traffic.",
            "",
        ]
    )


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in FORBIDDEN_TRUE_FIELDS:
        if config.get(field) is True:
            reasons.append(field)
    if float(config.get("canary_traffic_fraction", 0.0)) > 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _input_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    stage21_4_root = Path(config["stage21_4_tiny_ppo_update_smoke_root"])
    for artifact in (STAGE21_4_SUMMARY, STAGE21_4_CHECKPOINT, STAGE21_4_ROUTING):
        try:
            read_json_artifact(stage21_4_root, artifact)
        except FileNotFoundError:
            reasons.append(f"missing_{artifact.legacy[0] if artifact.legacy else artifact.canonical}")
    if not artifact_io.path_is_file(Path(config["high_fidelity_config"])):
        reasons.append("missing_high_fidelity_config")
    if config["include_canonical_reward_rerank_oracle"] and not artifact_io.path_is_file(Path(config["canonical_reward_rerank_profile"])):
        reasons.append("missing_canonical_reward_rerank_profile")
    if not config["execute_high_fidelity_evaluations"]:
        for label, key in (("pre", "pre_ppo_evaluation_root"), ("post", "post_ppo_evaluation_root")):
            root = Path(str(config.get(key, "")))
            if not artifact_io.path_is_file(root / "xunce-exploration-coverage-comparison-summary.json"):
                reasons.append(f"missing_{label}_evaluation_summary")
            if not artifact_io.path_is_file(root / "xunce-exploration-coverage-episodes.jsonl"):
                reasons.append(f"missing_{label}_evaluation_episodes")
    return reasons


def _load_config(path: Path, *, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    for field in ("stage21_4_tiny_ppo_update_smoke_root", "high_fidelity_config", "canonical_reward_rerank_profile"):
        if field in config and config[field] is not None:
            config[field] = str(_resolve_path(Path(config[field]), repo_root))
    for field in ("pre_ppo_evaluation_root", "post_ppo_evaluation_root"):
        if config.get(field):
            config[field] = str(_resolve_path(Path(config[field]), repo_root))
    config["execute_high_fidelity_evaluations"] = bool(config.get("execute_high_fidelity_evaluations", True))
    config["required_scenario_count"] = _positive_int(config.get("required_scenario_count", 2), "required_scenario_count")
    config["rollout_steps"] = _positive_int(config.get("rollout_steps", 4), "rollout_steps")
    config["dynamic_max_candidates_per_step"] = _positive_int(
        config.get("dynamic_max_candidates_per_step", 36),
        "dynamic_max_candidates_per_step",
    )
    config["dynamic_proposal_pool_limit_per_step"] = _positive_int(
        config.get("dynamic_proposal_pool_limit_per_step", 288),
        "dynamic_proposal_pool_limit_per_step",
    )
    config["include_canonical_reward_rerank_oracle"] = bool(config.get("include_canonical_reward_rerank_oracle", True))
    config["include_oracle_baselines"] = bool(config.get("include_oracle_baselines", True))
    config["xunce_only_evaluation"] = bool(config.get("xunce_only_evaluation", False))
    config["hybrid_astar_candidate_eval_workers"] = _positive_int(
        config.get("hybrid_astar_candidate_eval_workers", 1),
        "hybrid_astar_candidate_eval_workers",
    )
    config["candidate_reachability_gate_source"] = str(
        config.get("candidate_reachability_gate_source") or CANDIDATE_REACHABILITY_GATE_LEGACY
    )
    if config["candidate_reachability_gate_source"] not in {
        CANDIDATE_REACHABILITY_GATE_LEGACY,
        CANDIDATE_REACHABILITY_GATE_SOURCE,
    }:
        raise ConfigError("candidate_reachability_gate_source is invalid")
    config["candidate_reachability_max_theta_proposals_per_candidate"] = _nonnegative_int(
        config.get("candidate_reachability_max_theta_proposals_per_candidate", 0),
        "candidate_reachability_max_theta_proposals_per_candidate",
    )
    config["candidate_reachability_theta_proposal_policy"] = str(
        config.get("candidate_reachability_theta_proposal_policy") or "candidate_viewpoint_current_step/v1"
    )
    if config["candidate_reachability_theta_proposal_policy"] not in {
        "candidate_viewpoint_current_step/v1",
        "candidate_current_bearing_sweep/v1",
    }:
        raise ConfigError("candidate_reachability_theta_proposal_policy is invalid")
    if config.get("hybrid_astar_pose_path_cost_enabled") is not None:
        config["hybrid_astar_pose_path_cost_enabled"] = bool(config.get("hybrid_astar_pose_path_cost_enabled"))
    if config.get("hybrid_astar_planning_grid_source") is not None:
        config["hybrid_astar_planning_grid_source"] = str(config["hybrid_astar_planning_grid_source"])
    for key in (
        "planner_grid_resolution_m",
        "hybrid_astar_closed_key_xy_resolution_m",
        "hybrid_astar_goal_position_tolerance_m",
        "hybrid_astar_goal_theta_tolerance_deg",
        "hybrid_astar_primitive_duration_s",
        "hybrid_astar_integration_dt_s",
        "hybrid_astar_max_speed_mps",
        "hybrid_astar_max_angular_speed_degps",
    ):
        if config.get(key) is not None:
            numeric = _finite(config.get(key))
            if numeric is None or numeric <= 0.0:
                raise ConfigError(f"{key} must be a positive finite number")
            config[key] = float(numeric)
    if config.get("hybrid_astar_max_iterations") is not None:
        config["hybrid_astar_max_iterations"] = _positive_int(
            config.get("hybrid_astar_max_iterations"),
            "hybrid_astar_max_iterations",
        )
    config["emit_candidate_metric_audit"] = bool(config.get("emit_candidate_metric_audit", True))
    config["min_post_update_coverage_delta"] = _nonnegative_float(
        config.get("min_post_update_coverage_delta", 0.0),
        "min_post_update_coverage_delta",
    )
    config["post_update_success_metric"] = str(config.get("post_update_success_metric") or SUCCESS_METRIC_DEFAULT)
    config["canary_traffic_fraction"] = _nonnegative_float(config.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    return config


def _read_json(path: Path) -> dict[str, Any]:
    return artifact_io.read_json(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not artifact_io.path_is_file(Path(path)):
        return []
    rows: list[dict[str, Any]] = []
    for line in artifact_io.read_text(Path(path)).splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a positive integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a positive integer") from exc
    if numeric <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return numeric


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a non-negative integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a non-negative integer") from exc
    if numeric < 0:
        raise ConfigError(f"{name} must be a non-negative integer")
    return numeric


def _nonnegative_float(value: Any, name: str) -> float:
    numeric = _finite(value)
    if numeric is None or numeric < 0.0:
        raise ConfigError(f"{name} must be a non-negative finite number")
    return float(numeric)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _mean(values: list[Any]) -> float | None:
    finite = [_finite(value) for value in values]
    numbers = [float(value) for value in finite if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _nullable_delta(post: Any, pre: Any) -> float | None:
    post_value = _finite(post)
    pre_value = _finite(pre)
    if post_value is None or pre_value is None:
        return None
    return float(post_value) - float(pre_value)


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        return int(value)
    if isinstance(value, str) and not value.strip():
        return None
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return parsed


def _unique_sorted(values: list[str]) -> list[str]:
    return sorted({str(value) for value in values if value})


if __name__ == "__main__":
    raise SystemExit(main())
