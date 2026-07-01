from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
REPO_ROOT = SCRIPT_DIR.parent
MODEL_EXPLORER_SRC = REPO_ROOT / "model-explorer" / "src"
if MODEL_EXPLORER_SRC.is_dir() and str(MODEL_EXPLORER_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_EXPLORER_SRC))

try:
    from git_provenance import git_snapshot
    from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from global_99_governance_common import global_99_boundary_defaults
    from xunce_dynamic_frontier_nbv import (
        GENERATION_SOURCE as DYNAMIC_FRONTIER_NBV_GENERATION_SOURCE,
        build_dynamic_frontier_nbv_candidates,
        candidate_set_hash,
        covered_cells_hash,
        resolve_roi_group,
    )
    from xunce_frontier_nbv_validation import IN_PROCESS_VALIDATION_MODE, validate_candidate_cells
    from xunce_stage18_guard_thresholds import stage18_guard_thresholds
    from xunce_theta_viewpoint_candidates import (
        candidate_observation_cells,
        expand_theta_aware_candidates,
        theta_metadata,
        theta_viewpoints_enabled,
    )
    from xunce_obstacle_aware_theta_sensor_coverage import (
        extract_obstacle_cells,
        stable_obstacle_source_hash,
        visible_cells_for_viewpoint_with_obstacles,
    )
    from xunce_hybrid_astar_candidate_path_cost import (
        PATH_COST_SOURCE as HYBRID_ASTAR_PATH_COST_SOURCE,
        Cell as HYBRID_CELL,
        WorldPoint as HYBRID_WORLD_POINT,
        build_cost_grid_from_sidecar,
        evaluate_hybrid_astar_candidate_path_cost,
    )
    from xunce_continuous_theta_action import CONTINUOUS_THETA_ACTION_SPACE, continuous_theta_enabled, normalize_theta_rad
    from xunce_synthetic_exploration_credit import (
        apply_feature_rows_to_xunce_batch,
        build_synthetic_credit_feature_rows,
        candidate_pressure_values,
        feature_semantic_map,
    )
    from xunce_platform_contract import apply_stage23_platform_defaults
    from xunce_theta_sensor_coverage import visible_cells_for_viewpoint
    from run_xunce_high_fidelity_real_map_comparison import (
        _boundary_audit as _stage18b_boundary_audit,
        _candidate_at,
        _candidate_cell,
        _candidate_cost,
        _candidate_is_valid,
        _candidate_rows,
        _finite_or_none,
        _float_default,
        _int_value,
        _load_model_bundle,
        _load_source,
        _mask_violation,
        _model_score,
        _policy_detail_to_dict,
        _scenario_to_model_inputs,
        _score_xunce_model,
        _selected_index,
    )
    from model_explorer.policy.canonical_reward import compute_canonical_reward_components, load_canonical_reward_profile
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from scripts.global_99_governance_common import global_99_boundary_defaults
    from scripts.xunce_dynamic_frontier_nbv import (
        GENERATION_SOURCE as DYNAMIC_FRONTIER_NBV_GENERATION_SOURCE,
        build_dynamic_frontier_nbv_candidates,
        candidate_set_hash,
        covered_cells_hash,
        resolve_roi_group,
    )
    from scripts.xunce_frontier_nbv_validation import IN_PROCESS_VALIDATION_MODE, validate_candidate_cells
    from scripts.xunce_stage18_guard_thresholds import stage18_guard_thresholds
    from scripts.xunce_theta_viewpoint_candidates import (
        candidate_observation_cells,
        expand_theta_aware_candidates,
        theta_metadata,
        theta_viewpoints_enabled,
    )
    from scripts.xunce_obstacle_aware_theta_sensor_coverage import (
        extract_obstacle_cells,
        stable_obstacle_source_hash,
        visible_cells_for_viewpoint_with_obstacles,
    )
    from scripts.xunce_hybrid_astar_candidate_path_cost import (
        PATH_COST_SOURCE as HYBRID_ASTAR_PATH_COST_SOURCE,
        Cell as HYBRID_CELL,
        WorldPoint as HYBRID_WORLD_POINT,
        build_cost_grid_from_sidecar,
        evaluate_hybrid_astar_candidate_path_cost,
    )
    from scripts.xunce_continuous_theta_action import CONTINUOUS_THETA_ACTION_SPACE, continuous_theta_enabled, normalize_theta_rad
    from scripts.xunce_synthetic_exploration_credit import (
        apply_feature_rows_to_xunce_batch,
        build_synthetic_credit_feature_rows,
        candidate_pressure_values,
        feature_semantic_map,
    )
    from scripts.xunce_platform_contract import apply_stage23_platform_defaults
    from scripts.xunce_theta_sensor_coverage import visible_cells_for_viewpoint
    from scripts.run_xunce_high_fidelity_real_map_comparison import (
        _boundary_audit as _stage18b_boundary_audit,
        _candidate_at,
        _candidate_cell,
        _candidate_cost,
        _candidate_is_valid,
        _candidate_rows,
        _finite_or_none,
        _float_default,
        _int_value,
        _load_model_bundle,
        _load_source,
        _mask_violation,
        _model_score,
        _policy_detail_to_dict,
        _scenario_to_model_inputs,
        _score_xunce_model,
        _selected_index,
    )
    from model_explorer.policy.canonical_reward import compute_canonical_reward_components, load_canonical_reward_profile


CONFIG_SCHEMA_VERSION = "xunce-high-fidelity-exploration-coverage-comparison-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-exploration-coverage-comparison-summary/v1"
DEFAULT_CONFIG = "configs/xunce_high_fidelity_exploration_coverage_comparison_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_high_fidelity_exploration_coverage_comparison_v1"
DEFAULT_CANONICAL_PROFILE = "configs/xunce_canonical_reward_guard_profile_v2.json"

SUMMARY_FILE = "xunce-exploration-coverage-comparison-summary.json"
EPISODES_FILE = "xunce-exploration-coverage-episodes.jsonl"
STEPS_FILE = "xunce-exploration-coverage-steps.jsonl"
COMPARISON_PAIRS_FILE = "xunce-exploration-coverage-comparison-pairs.jsonl"
COMPARISON_AGGREGATE_FILE = "xunce-exploration-coverage-comparison-aggregate.json"
DYNAMIC_PROPOSALS_FILE = "xunce-exploration-coverage-dynamic-proposals.jsonl"
DYNAMIC_VALIDATION_RESULTS_FILE = "xunce-exploration-coverage-dynamic-validation-results.jsonl"
DYNAMIC_VALIDATION_AUDIT_FILE = "xunce-exploration-coverage-dynamic-validation-audit.json"
PAIRED_DECISION_AUDIT_FILE = "xunce-exploration-coverage-paired-decision-audit.jsonl"
CANDIDATE_METRIC_AUDIT_FILE = "xunce-exploration-coverage-candidate-metric-audit.jsonl"
OBSTACLE_SOURCES_FILE = "xunce-exploration-coverage-obstacle-sources.json"
ON_POLICY_ORACLE_TEACHER_LABELS_FILE = "xunce-exploration-coverage-on-policy-oracle-teacher-labels.jsonl"
MODEL_INFERENCE_FILE = "xunce-exploration-coverage-model-inference.jsonl"
ROI_BREAKDOWN_FILE = "xunce-exploration-coverage-roi-breakdown.json"
DECISION_AUDIT_FILE = "xunce-exploration-coverage-decision-audit.json"
MANIFEST_FILE = "xunce-exploration-coverage-comparison-manifest.json"
REPORT_FILE = "xunce-exploration-coverage-comparison-report.md"
V2_SUMMARY_FILE = "xunce-exploration-coverage-comparison-v2-summary.json"
V2_EPISODES_FILE = "xunce-exploration-coverage-v2-episodes.jsonl"
V2_STEPS_FILE = "xunce-exploration-coverage-v2-steps.jsonl"

FIX_XUNCE_CHECKPOINT_NEXT_REQUIRED_CHANGE = "fix_xunce_sandbox_candidate_preflight"
FIX_INCUMBENT_CHECKPOINT_NEXT_REQUIRED_CHANGE = "fix_incumbent_policy_checkpoint"
FIX_RUNNER_NEXT_REQUIRED_CHANGE = "fix_xunce_coverage_comparison_runner"
BOUNDARY_NEXT_REQUIRED_CHANGE = "resolve_xunce_coverage_boundary_rejections"
EFFICIENCY_NEXT_REQUIRED_CHANGE = "refine_coverage_reward_and_cost_guard"
ADVANTAGE_NEXT_REQUIRED_CHANGE = "xunce_default_policy_candidate_authorization_preflight"
NO_ADVANTAGE_NEXT_REQUIRED_CHANGE = "xunce_research_iteration_required"
FIX_SOURCE_NEXT_REQUIRED_CHANGE = "fix_xunce_high_fidelity_real_map_roi_expansion"
REVIEW_COMPARISON_NEXT_REQUIRED_CHANGE = "review_xunce_incumbent_comparison_metrics"

POLICIES = ("xunce", "incumbent")
ORACLE_POLICIES = ("greedy_coverage_oracle", "cost_aware_coverage_oracle")
CANONICAL_REWARD_RERANK_ORACLE = "canonical_reward_rerank_oracle"
MODEL_POLICY_INFERENCE_KIND = "true_checkpoint_inference"
ORACLE_POLICY_INFERENCE_KIND = "oracle_offline_policy"
TOLERANCE = 1.0e-12

DEFAULT_COMPARISON_UTILITY_PROFILES = {
    "coverage_first": {"cost_weight": 0.0, "risk_weight": 0.0},
    "cost_aware": {"cost_weight": 0.0001, "risk_weight": 0.0},
    "risk_aware": {"cost_weight": 0.0001, "risk_weight": 0.01},
}

BOUNDARY_FIELDS = tuple(global_99_boundary_defaults()) + (
    "real_world_release_approved",
    "real_world_performance_claimed",
    "default_policy_replacement_approved",
    "real_executor_connection_approved",
    "starts_online_canary",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Xunce high-fidelity exploration coverage comparison v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--source-roi-expansion-root")
    parser.add_argument("--xunce-candidate-checkpoint")
    parser.add_argument("--incumbent-policy-checkpoint")
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--candidate-refresh-mode")
    parser.add_argument("--coverage-metric-mode")
    parser.add_argument("--dynamic-candidate-validation-mode")
    parser.add_argument("--dynamic-candidate-generation-mode")
    parser.add_argument("--dynamic-candidate-selection-mode")
    parser.add_argument("--dynamic-validation-work-root")
    parser.add_argument("--dynamic-validation-max-path-length", type=int)
    parser.add_argument("--dynamic-sidecar-fallback-mode")
    parser.add_argument("--dynamic-proposal-pool-limit-per-step", type=int)
    parser.add_argument("--dynamic-max-candidates-per-step", type=int)
    parser.add_argument("--dynamic-min-frontier-candidates-per-step", type=int)
    parser.add_argument("--dynamic-min-low-cost-candidates-per-step", type=int)
    parser.add_argument("--dynamic-min-efficiency-candidates-per-step", type=int)
    parser.add_argument("--dynamic-frontier-min-uncovered-component-cells", type=int)
    parser.add_argument("--dynamic-frontier-max-components", type=int)
    parser.add_argument("--dynamic-adapter-audit-enabled", action="store_true")
    parser.add_argument("--dynamic-adapter-audit-max-routes", type=int)
    parser.add_argument("--dynamic-adapter-audit-min-per-frontier-source", type=int)
    parser.add_argument("--include-oracle-baselines", action="store_true")
    parser.add_argument("--include-roi-weighted-coverage", action="store_true")
    parser.add_argument("--emit-candidate-metric-audit", action="store_true")
    parser.add_argument("--include-canonical-reward-rerank-oracle", action="store_true")
    parser.add_argument("--canonical-reward-rerank-profile")
    parser.add_argument("--emit-on-policy-oracle-teacher-labels", action="store_true")
    parser.add_argument("--on-policy-oracle-teacher-baseline-policy")
    parser.add_argument("--on-policy-oracle-teacher-profile")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    overrides = {
        key: value
        for key, value in {
            "source_roi_expansion_root": args.source_roi_expansion_root,
            "xunce_candidate_checkpoint": args.xunce_candidate_checkpoint,
            "incumbent_policy_checkpoint": args.incumbent_policy_checkpoint,
            "rollout_steps": args.rollout_steps,
            "candidate_refresh_mode": args.candidate_refresh_mode,
            "coverage_metric_mode": args.coverage_metric_mode,
            "dynamic_candidate_validation_mode": args.dynamic_candidate_validation_mode,
            "dynamic_candidate_generation_mode": args.dynamic_candidate_generation_mode,
            "dynamic_candidate_selection_mode": args.dynamic_candidate_selection_mode,
            "dynamic_validation_work_root": args.dynamic_validation_work_root,
            "dynamic_validation_max_path_length": args.dynamic_validation_max_path_length,
            "dynamic_sidecar_fallback_mode": args.dynamic_sidecar_fallback_mode,
            "dynamic_proposal_pool_limit_per_step": args.dynamic_proposal_pool_limit_per_step,
            "dynamic_max_candidates_per_step": args.dynamic_max_candidates_per_step,
            "dynamic_min_frontier_candidates_per_step": args.dynamic_min_frontier_candidates_per_step,
            "dynamic_min_low_cost_candidates_per_step": args.dynamic_min_low_cost_candidates_per_step,
            "dynamic_min_efficiency_candidates_per_step": args.dynamic_min_efficiency_candidates_per_step,
            "dynamic_frontier_min_uncovered_component_cells": args.dynamic_frontier_min_uncovered_component_cells,
            "dynamic_frontier_max_components": args.dynamic_frontier_max_components,
            "dynamic_adapter_audit_enabled": True if args.dynamic_adapter_audit_enabled else None,
            "dynamic_adapter_audit_max_routes": args.dynamic_adapter_audit_max_routes,
            "dynamic_adapter_audit_min_per_frontier_source": args.dynamic_adapter_audit_min_per_frontier_source,
            "include_oracle_baselines": True if args.include_oracle_baselines else None,
            "include_roi_weighted_coverage": True if args.include_roi_weighted_coverage else None,
            "emit_candidate_metric_audit": True if args.emit_candidate_metric_audit else None,
            "include_canonical_reward_rerank_oracle": True if args.include_canonical_reward_rerank_oracle else None,
            "canonical_reward_rerank_profile": args.canonical_reward_rerank_profile,
            "emit_on_policy_oracle_teacher_labels": True if args.emit_on_policy_oracle_teacher_labels else None,
            "on_policy_oracle_teacher_baseline_policy": args.on_policy_oracle_teacher_baseline_policy,
            "on_policy_oracle_teacher_profile": args.on_policy_oracle_teacher_profile,
        }.items()
        if value is not None
    }
    try:
        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
            config_overrides=overrides,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "xunce_coverage_advantage_established": summary["xunce_coverage_advantage_established"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_high_fidelity_exploration_coverage_comparison(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path, repo_root, config_overrides=config_overrides)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_root)

    source = _load_source(config, repo_root)
    obstacle_source_audit = _obstacle_source_audit(source, config, repo_root=repo_root)
    source["obstacle_source_count"] = obstacle_source_audit["source_count"]
    boundary = _boundary_audit(config, source)
    model_bundle = _load_model_bundle(config, source, repo_root)
    source_match = _source_match_audit(config, source)
    (
        episodes,
        steps,
        inference_rows,
        dynamic_proposals,
        dynamic_validation_rows,
        paired_decision_rows,
        candidate_metric_rows,
        on_policy_teacher_label_rows,
    ) = _run_coverage_rollouts(
        config,
        source,
        model_bundle,
        obstacle_source_audit=obstacle_source_audit,
        repo_root=repo_root,
        output_root=output_root,
    )
    comparison_pairs = _comparison_pairs(episodes, config)
    comparison_aggregate = _comparison_aggregate(comparison_pairs, config)
    comparison = _coverage_comparison_audit(
        episodes,
        steps,
        comparison_pairs,
        comparison_aggregate,
        dynamic_proposals=dynamic_proposals,
        dynamic_validation_rows=dynamic_validation_rows,
        paired_decision_rows=paired_decision_rows,
        candidate_metric_rows=candidate_metric_rows,
        on_policy_teacher_label_rows=on_policy_teacher_label_rows,
        source=source,
        config=config,
        repo_root=repo_root,
        output_root=output_root,
    )
    roi_breakdown = _roi_breakdown(episodes)
    model_inference = _model_inference_audit(model_bundle, inference_rows)
    decision = _decision(
        config=config,
        source=source,
        boundary=boundary,
        source_match=source_match,
        model_inference=model_inference,
        comparison=comparison,
    )
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=config_path,
        output_root=output_root,
        paths=paths,
        config=config,
        source=source,
        boundary=boundary,
        source_match=source_match,
        model_inference=model_inference,
        comparison=comparison,
        roi_breakdown=roi_breakdown,
        decision=decision,
        repo_root=repo_root,
    )
    manifest = {
        "schema_version": "xunce-exploration-coverage-comparison-manifest/v1",
        "generated_at": generated_at,
        "config": str(config_path),
        "normalized_config": _normalized_config_snapshot(config),
        "output_root": str(output_root),
        "profile_id": config["profile_id"],
        "profile_version": config["profile_version"],
        "profile_hash": config["profile_hash"],
        "artifacts": {key: str(value) for key, value in paths.items()},
        "candidate_metric_audit": str(paths["candidate_metric_audit"]) if config["emit_candidate_metric_audit"] else None,
        "candidate_metric_audit_row_count": len(candidate_metric_rows) if config["emit_candidate_metric_audit"] else 0,
        "obstacle_source_audit": str(paths["obstacle_sources"]),
        "obstacle_source_count": obstacle_source_audit["source_count"],
        "obstacle_source_missing_scenario_count": obstacle_source_audit["missing_source_scenario_count"],
        "obstacle_source_proxy_count": obstacle_source_audit["proxy_source_count"],
        "on_policy_oracle_teacher_label_audit": str(paths["on_policy_oracle_teacher_labels"])
        if config["emit_on_policy_oracle_teacher_labels"]
        else None,
        "on_policy_oracle_teacher_label_row_count": len(on_policy_teacher_label_rows)
        if config["emit_on_policy_oracle_teacher_labels"]
        else 0,
        "summary_status": summary["status"],
        "next_required_change": summary["next_required_change"],
    }

    write_jsonl(paths["episodes"], [_public_episode(row) for row in episodes])
    write_jsonl(paths["steps"], steps)
    write_jsonl(paths["comparison_pairs"], comparison_pairs)
    write_json(paths["comparison_aggregate"], comparison_aggregate)
    write_jsonl(paths["dynamic_proposals"], dynamic_proposals)
    write_jsonl(paths["dynamic_validation_results"], dynamic_validation_rows)
    write_json(paths["dynamic_validation_audit"], comparison["dynamic_validation_audit"])
    write_jsonl(paths["paired_decision_audit"], paired_decision_rows)
    write_json(paths["obstacle_sources"], obstacle_source_audit)
    if config["emit_candidate_metric_audit"]:
        write_jsonl(paths["candidate_metric_audit"], candidate_metric_rows)
    if config["emit_on_policy_oracle_teacher_labels"]:
        write_jsonl(paths["on_policy_oracle_teacher_labels"], on_policy_teacher_label_rows)
    write_jsonl(paths["model_inference"], inference_rows)
    write_json(paths["roi_breakdown"], roi_breakdown)
    write_json(paths["decision_audit"], decision)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    if _v2_enabled(config):
        write_json(paths["v2_summary"], summary)
        write_jsonl(paths["v2_episodes"], [_public_episode(row) for row in episodes])
        write_jsonl(paths["v2_steps"], steps)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


def _load_config(
    path: Path,
    repo_root: Path,
    *,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    if config_overrides:
        payload = {**payload, **config_overrides}
    payload = apply_stage23_platform_defaults(payload, repo_root=repo_root)
    normalized = dict(payload)
    for key in ("source_roi_expansion_root", "xunce_candidate_checkpoint", "incumbent_policy_checkpoint"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
        normalized[key] = str(resolve_path(Path(payload[key]), repo_root))
    normalized["required_scenario_count"] = _positive_int(payload.get("required_scenario_count", 24), "required_scenario_count")
    normalized["rollout_steps"] = _positive_int(payload.get("rollout_steps", 10), "rollout_steps")
    normalized["coverage_radius_cells"] = _nonnegative_int(payload.get("coverage_radius_cells", 1), "coverage_radius_cells")
    sensitivity = payload.get("coverage_radius_sensitivity", [3])
    if not isinstance(sensitivity, list) or any(not isinstance(item, int) or item < 0 for item in sensitivity):
        raise ConfigError("coverage_radius_sensitivity must be a list of non-negative integers")
    normalized["coverage_radius_sensitivity"] = list(sensitivity)
    normalized["coverage_denominator_cells"] = _positive_int(payload.get("coverage_denominator_cells", 1000), "coverage_denominator_cells")
    normalized["coverage_denominator_mode"] = _require_string(
        payload.get("coverage_denominator_mode", "fixed_config_cells"),
        "coverage_denominator_mode",
    )
    if normalized["coverage_denominator_mode"] not in {
        "roi_valid_cells",
        "fixed_config_cells",
        "raw_cells_only",
        "main_coverable_cells",
    }:
        raise ConfigError(
            "coverage_denominator_mode must be roi_valid_cells, fixed_config_cells, raw_cells_only, or main_coverable_cells"
        )
    normalized["path_budget_m"] = _positive_float(payload.get("path_budget_m", 5000.0), "path_budget_m")
    normalized["planning_backend"] = _require_string(payload.get("planning_backend", "channel_aware_astar"), "planning_backend")
    normalized["diagnostic_reason_codes"] = list(payload.get("diagnostic_reason_codes", [])) if isinstance(payload.get("diagnostic_reason_codes", []), list) else []
    normalized["candidate_refresh_mode"] = _require_string(payload.get("candidate_refresh_mode", "static_from_source"), "candidate_refresh_mode")
    if normalized["candidate_refresh_mode"] == "dynamic_from_coverage_memory":
        normalized["candidate_refresh_mode"] = "dynamic_validated_only"
        normalized["diagnostic_reason_codes"].append("dynamic_from_coverage_memory_legacy_mode")
    if normalized["candidate_refresh_mode"] not in {"static_from_source", "dynamic_validated_only", "dynamic_frontier_nbv_in_process"}:
        raise ConfigError("candidate_refresh_mode must be static_from_source, dynamic_validated_only, or dynamic_frontier_nbv_in_process")
    normalized["coverage_metric_mode"] = _require_string(payload.get("coverage_metric_mode", "endpoint_footprint"), "coverage_metric_mode")
    if normalized["coverage_metric_mode"] not in {"endpoint_footprint", "path_line_plus_endpoint"}:
        raise ConfigError("coverage_metric_mode must be endpoint_footprint or path_line_plus_endpoint")
    normalized["include_oracle_baselines"] = _require_bool(payload.get("include_oracle_baselines", False), "include_oracle_baselines")
    normalized["include_roi_weighted_coverage"] = _require_bool(payload.get("include_roi_weighted_coverage", False), "include_roi_weighted_coverage")
    normalized["emit_candidate_metric_audit"] = _require_bool(payload.get("emit_candidate_metric_audit", False), "emit_candidate_metric_audit")
    normalized["emit_obstacle_source_audit"] = _require_bool(
        payload.get("emit_obstacle_source_audit", True),
        "emit_obstacle_source_audit",
    )
    normalized["obstacle_occlusion_enabled"] = _require_bool(
        payload.get("obstacle_occlusion_enabled", False),
        "obstacle_occlusion_enabled",
    )
    normalized["no_go_blocks_los"] = _require_bool(
        payload.get("no_go_blocks_los", False),
        "no_go_blocks_los",
    )
    normalized["theta_aware_candidate_viewpoints_enabled"] = _require_bool(
        payload.get("theta_aware_candidate_viewpoints_enabled", False),
        "theta_aware_candidate_viewpoints_enabled",
    )
    normalized["continuous_theta_action_space_enabled"] = continuous_theta_enabled(payload)
    if normalized["continuous_theta_action_space_enabled"]:
        normalized["theta_aware_candidate_viewpoints_enabled"] = False
        normalized["action_space_type"] = CONTINUOUS_THETA_ACTION_SPACE
    normalized["theta_bin_count"] = _positive_int(payload.get("theta_bin_count", 8), "theta_bin_count")
    normalized["theta_step_deg"] = _positive_int(payload.get("theta_step_deg", 45), "theta_step_deg")
    normalized["sensor_model_id"] = _require_string(
        payload.get("sensor_model_id", "theta-fov-90-range-radius/v1"),
        "sensor_model_id",
    )
    normalized["sensor_fov_deg"] = _positive_float(payload.get("sensor_fov_deg", 90.0), "sensor_fov_deg")
    normalized["sensor_range_cells"] = _nonnegative_int(
        payload.get("sensor_range_cells", normalized["coverage_radius_cells"]),
        "sensor_range_cells",
    )
    normalized["hybrid_astar_pose_path_cost_enabled"] = _require_bool(
        payload.get("hybrid_astar_pose_path_cost_enabled", False),
        "hybrid_astar_pose_path_cost_enabled",
    )
    normalized["path_cost_source"] = _require_string(
        payload.get(
            "path_cost_source",
            HYBRID_ASTAR_PATH_COST_SOURCE
            if normalized["hybrid_astar_pose_path_cost_enabled"]
            else "legacy_grid_astar_path/v1",
        ),
        "path_cost_source",
    )
    normalized["initial_theta_deg"] = _finite_float_or_default(payload.get("initial_theta_deg", 0.0), 0.0)
    normalized["hybrid_astar_theta_bin_count"] = _positive_int(
        payload.get("hybrid_astar_theta_bin_count", 72),
        "hybrid_astar_theta_bin_count",
    )
    hybrid_goal_position_tolerance = payload.get("hybrid_astar_goal_position_tolerance_m")
    normalized["hybrid_astar_goal_position_tolerance_m"] = (
        None
        if hybrid_goal_position_tolerance is None
        else _positive_float(hybrid_goal_position_tolerance, "hybrid_astar_goal_position_tolerance_m")
    )
    normalized["hybrid_astar_goal_theta_tolerance_deg"] = _positive_float(
        payload.get("hybrid_astar_goal_theta_tolerance_deg", 5.0),
        "hybrid_astar_goal_theta_tolerance_deg",
    )
    normalized["hybrid_astar_max_iterations"] = _positive_int(
        payload.get("hybrid_astar_max_iterations", 100_000),
        "hybrid_astar_max_iterations",
    )
    normalized["hybrid_astar_candidate_eval_workers"] = _positive_int(
        payload.get("hybrid_astar_candidate_eval_workers", 1),
        "hybrid_astar_candidate_eval_workers",
    )
    normalized["hybrid_astar_primitive_duration_s"] = _positive_float(
        payload.get("hybrid_astar_primitive_duration_s", 1.0),
        "hybrid_astar_primitive_duration_s",
    )
    normalized["hybrid_astar_integration_dt_s"] = _positive_float(
        payload.get("hybrid_astar_integration_dt_s", 0.25),
        "hybrid_astar_integration_dt_s",
    )
    normalized["hybrid_astar_max_speed_mps"] = _positive_float(
        payload.get("hybrid_astar_max_speed_mps", 1.0),
        "hybrid_astar_max_speed_mps",
    )
    normalized["hybrid_astar_max_angular_speed_degps"] = _positive_float(
        payload.get("hybrid_astar_max_angular_speed_degps", 45.0),
        "hybrid_astar_max_angular_speed_degps",
    )
    normalized["hybrid_astar_rotation_cost_weight"] = _nonnegative_float(
        payload.get("hybrid_astar_rotation_cost_weight", 0.2),
        "hybrid_astar_rotation_cost_weight",
    )
    normalized["hybrid_astar_reverse_penalty_weight"] = _nonnegative_float(
        payload.get("hybrid_astar_reverse_penalty_weight", 0.5),
        "hybrid_astar_reverse_penalty_weight",
    )
    normalized["hybrid_astar_turn_penalty_weight"] = _nonnegative_float(
        payload.get("hybrid_astar_turn_penalty_weight", 0.05),
        "hybrid_astar_turn_penalty_weight",
    )
    canonical_profile_path = resolve_path(
        Path(str(payload.get("canonical_reward_profile", DEFAULT_CANONICAL_PROFILE))),
        repo_root,
    )
    canonical_profile = load_canonical_reward_profile(canonical_profile_path)
    guard_thresholds = stage18_guard_thresholds(canonical_profile)
    normalized["canonical_reward_profile"] = str(canonical_profile_path)
    normalized["profile_id"] = canonical_profile.profile_id
    normalized["profile_version"] = canonical_profile.profile_version
    normalized["profile_hash"] = canonical_profile.profile_hash
    normalized["canonical_guard_thresholds"] = guard_thresholds.to_artifact_dict()
    normalized["include_canonical_reward_rerank_oracle"] = _require_bool(
        payload.get("include_canonical_reward_rerank_oracle", False),
        "include_canonical_reward_rerank_oracle",
    )
    normalized["xunce_only_evaluation"] = _require_bool(
        payload.get("xunce_only_evaluation", False),
        "xunce_only_evaluation",
    )
    rerank_profile_path = payload.get("canonical_reward_rerank_profile")
    if normalized["include_canonical_reward_rerank_oracle"]:
        if rerank_profile_path is None:
            rerank_profile_path = canonical_profile_path
        else:
            rerank_profile_path = resolve_path(Path(str(rerank_profile_path)), repo_root)
        rerank_profile = load_canonical_reward_profile(Path(rerank_profile_path))
        if rerank_profile.profile_version != "v3":
            raise ConfigError("canonical_reward_rerank_profile must use profile_version v3")
        normalized["canonical_reward_rerank_profile"] = str(Path(rerank_profile_path))
        normalized["canonical_reward_rerank_profile_id"] = rerank_profile.profile_id
        normalized["canonical_reward_rerank_profile_version"] = rerank_profile.profile_version
        normalized["canonical_reward_rerank_profile_hash"] = rerank_profile.profile_hash
    else:
        normalized["canonical_reward_rerank_profile"] = None
        normalized["canonical_reward_rerank_profile_id"] = None
        normalized["canonical_reward_rerank_profile_version"] = None
        normalized["canonical_reward_rerank_profile_hash"] = None
    normalized["emit_on_policy_oracle_teacher_labels"] = _require_bool(
        payload.get("emit_on_policy_oracle_teacher_labels", False),
        "emit_on_policy_oracle_teacher_labels",
    )
    normalized["on_policy_oracle_teacher_baseline_policy"] = _require_string(
        payload.get("on_policy_oracle_teacher_baseline_policy", "xunce"),
        "on_policy_oracle_teacher_baseline_policy",
    )
    if normalized["on_policy_oracle_teacher_baseline_policy"] != "xunce":
        raise ConfigError("on_policy_oracle_teacher_baseline_policy currently supports only xunce")
    teacher_profile_path = payload.get("on_policy_oracle_teacher_profile")
    if normalized["emit_on_policy_oracle_teacher_labels"]:
        if teacher_profile_path is None:
            teacher_profile_path = rerank_profile_path if rerank_profile_path is not None else canonical_profile_path
        else:
            teacher_profile_path = resolve_path(Path(str(teacher_profile_path)), repo_root)
        teacher_profile = load_canonical_reward_profile(Path(teacher_profile_path))
        if teacher_profile.profile_version != "v3":
            raise ConfigError("on_policy_oracle_teacher_profile must use profile_version v3")
        normalized["on_policy_oracle_teacher_profile"] = str(Path(teacher_profile_path))
        normalized["on_policy_oracle_teacher_profile_id"] = teacher_profile.profile_id
        normalized["on_policy_oracle_teacher_profile_version"] = teacher_profile.profile_version
        normalized["on_policy_oracle_teacher_profile_hash"] = teacher_profile.profile_hash
    else:
        normalized["on_policy_oracle_teacher_profile"] = None
        normalized["on_policy_oracle_teacher_profile_id"] = None
        normalized["on_policy_oracle_teacher_profile_version"] = None
        normalized["on_policy_oracle_teacher_profile_hash"] = None
    utility_profile_payload = payload.get("comparison_utility_profiles")
    if utility_profile_payload is None:
        utility_profile_payload = _comparison_utility_profiles_from_canonical(canonical_profile)
    normalized["comparison_utility_profiles"] = _comparison_utility_profiles(utility_profile_payload)
    normalized["comparison_utility_profiles_diagnostic_only"] = True
    normalized["stage19_readiness_source"] = "canonical_guard_not_utility_profiles"
    normalized["dynamic_candidate_validation_mode"] = _require_string(
        payload.get("dynamic_candidate_validation_mode", "in_process_path_planner_astar_batch"),
        "dynamic_candidate_validation_mode",
    )
    normalized["dynamic_candidate_generation_mode"] = _require_string(
        payload.get("dynamic_candidate_generation_mode", "map_aware_coverage_frontier_nbv"),
        "dynamic_candidate_generation_mode",
    )
    normalized["dynamic_candidate_selection_mode"] = _require_string(
        payload.get("dynamic_candidate_selection_mode", "validated_pareto_diverse"),
        "dynamic_candidate_selection_mode",
    )
    dynamic_work_root = payload.get("dynamic_validation_work_root", "outputs/_xunce_dynamic_validation_work")
    if not isinstance(dynamic_work_root, str) or not dynamic_work_root.strip():
        raise ConfigError("dynamic_validation_work_root must be a non-empty path string")
    normalized["dynamic_validation_work_root"] = str(resolve_path(Path(dynamic_work_root), repo_root))
    normalized["dynamic_validation_max_path_length"] = _positive_int(
        payload.get("dynamic_validation_max_path_length", 180),
        "dynamic_validation_max_path_length",
    )
    normalized["dynamic_validation_short_path_retry_enabled"] = _require_bool(
        payload.get("dynamic_validation_short_path_retry_enabled", True),
        "dynamic_validation_short_path_retry_enabled",
    )
    normalized["dynamic_sidecar_fallback_mode"] = _require_string(
        payload.get("dynamic_sidecar_fallback_mode", "diagnostic_only"),
        "dynamic_sidecar_fallback_mode",
    )
    if normalized["dynamic_sidecar_fallback_mode"] not in {"diagnostic_only", "formal_screening"}:
        raise ConfigError("dynamic_sidecar_fallback_mode must be diagnostic_only or formal_screening")
    radii = payload.get("dynamic_frontier_radius_cells", [2, 4, 6])
    if not isinstance(radii, list) or any(not isinstance(item, int) or item <= 0 for item in radii):
        raise ConfigError("dynamic_frontier_radius_cells must be a list of positive integers")
    normalized["dynamic_frontier_radius_cells"] = list(radii)
    normalized["dynamic_frontier_direction_count"] = _positive_int(payload.get("dynamic_frontier_direction_count", 8), "dynamic_frontier_direction_count")
    normalized["dynamic_proposal_pool_limit_per_step"] = _positive_int(
        payload.get("dynamic_proposal_pool_limit_per_step", 24),
        "dynamic_proposal_pool_limit_per_step",
    )
    normalized["dynamic_max_candidates_per_step"] = _positive_int(
        payload.get("dynamic_max_candidates_per_step", 6),
        "dynamic_max_candidates_per_step",
    )
    normalized["dynamic_min_frontier_candidates_per_step"] = _positive_int(
        payload.get("dynamic_min_frontier_candidates_per_step", 1),
        "dynamic_min_frontier_candidates_per_step",
    )
    normalized["dynamic_min_low_cost_candidates_per_step"] = _positive_int(
        payload.get("dynamic_min_low_cost_candidates_per_step", 1),
        "dynamic_min_low_cost_candidates_per_step",
    )
    normalized["dynamic_min_efficiency_candidates_per_step"] = _positive_int(
        payload.get("dynamic_min_efficiency_candidates_per_step", 1),
        "dynamic_min_efficiency_candidates_per_step",
    )
    normalized["dynamic_frontier_min_uncovered_component_cells"] = _positive_int(
        payload.get("dynamic_frontier_min_uncovered_component_cells", 4),
        "dynamic_frontier_min_uncovered_component_cells",
    )
    normalized["dynamic_frontier_max_components"] = _positive_int(
        payload.get("dynamic_frontier_max_components", 8),
        "dynamic_frontier_max_components",
    )
    weight_map = payload.get("roi_group_weight_map", {})
    if weight_map is not None and not isinstance(weight_map, dict):
        raise ConfigError("roi_group_weight_map must be an object when provided")
    normalized["roi_group_weight_map"] = dict(weight_map or {})
    normalized["dynamic_validation_cache_enabled"] = _require_bool(payload.get("dynamic_validation_cache_enabled", True), "dynamic_validation_cache_enabled")
    normalized["dynamic_adapter_audit_enabled"] = _require_bool(payload.get("dynamic_adapter_audit_enabled", False), "dynamic_adapter_audit_enabled")
    normalized["dynamic_adapter_audit_max_routes"] = _positive_int(
        payload.get("dynamic_adapter_audit_max_routes", 128),
        "dynamic_adapter_audit_max_routes",
    )
    normalized["dynamic_adapter_audit_min_per_frontier_source"] = _positive_int(
        payload.get("dynamic_adapter_audit_min_per_frontier_source", 1),
        "dynamic_adapter_audit_min_per_frontier_source",
    )
    normalized["debug_validation_artifacts"] = _require_bool(payload.get("debug_validation_artifacts", False), "debug_validation_artifacts")
    normalized["state_conditioned_candidate_generation"] = _require_bool(
        payload.get("state_conditioned_candidate_generation", True),
        "state_conditioned_candidate_generation",
    )
    normalized["allow_open_grid_fallback"] = _require_bool(payload.get("allow_open_grid_fallback", False), "allow_open_grid_fallback")
    normalized["require_context_ids"] = _require_bool(payload.get("require_context_ids", True), "require_context_ids")
    normalized["require_contract_and_sidecar_paths"] = _require_bool(payload.get("require_contract_and_sidecar_paths", True), "require_contract_and_sidecar_paths")
    normalized["max_xunce_parameter_count"] = _positive_int(payload.get("max_xunce_parameter_count", 10_000_000), "max_xunce_parameter_count")
    normalized["max_median_inference_latency_ms"] = _nonnegative_float(payload.get("max_median_inference_latency_ms", 5.0), "max_median_inference_latency_ms")
    normalized["max_latency_ratio_vs_incumbent"] = _nonnegative_float(payload.get("max_latency_ratio_vs_incumbent", 10.0), "max_latency_ratio_vs_incumbent")
    normalized["canary_traffic_fraction"] = _nonnegative_float(payload.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    return normalized


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "episodes": output_root / EPISODES_FILE,
        "steps": output_root / STEPS_FILE,
        "comparison_pairs": output_root / COMPARISON_PAIRS_FILE,
        "comparison_aggregate": output_root / COMPARISON_AGGREGATE_FILE,
        "dynamic_proposals": output_root / DYNAMIC_PROPOSALS_FILE,
        "dynamic_validation_results": output_root / DYNAMIC_VALIDATION_RESULTS_FILE,
        "dynamic_validation_audit": output_root / DYNAMIC_VALIDATION_AUDIT_FILE,
        "paired_decision_audit": output_root / PAIRED_DECISION_AUDIT_FILE,
        "candidate_metric_audit": output_root / CANDIDATE_METRIC_AUDIT_FILE,
        "obstacle_sources": output_root / OBSTACLE_SOURCES_FILE,
        "on_policy_oracle_teacher_labels": output_root / ON_POLICY_ORACLE_TEACHER_LABELS_FILE,
        "model_inference": output_root / MODEL_INFERENCE_FILE,
        "roi_breakdown": output_root / ROI_BREAKDOWN_FILE,
        "decision_audit": output_root / DECISION_AUDIT_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "report": output_root / REPORT_FILE,
        "v2_summary": output_root / V2_SUMMARY_FILE,
        "v2_episodes": output_root / V2_EPISODES_FILE,
        "v2_steps": output_root / V2_STEPS_FILE,
    }


def _obstacle_source_audit(source: dict[str, Any], config: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    scenarios = source.get("path_feedback", {}).get("scenarios", [])
    if not isinstance(scenarios, list):
        scenarios = []
    slice_by_id = {str(row.get("scenario_id")): row for row in source.get("slices", []) if isinstance(row, dict)}
    sources: list[dict[str, Any]] = []
    source_by_scenario: dict[str, dict[str, Any]] = {}
    missing_scenarios: list[str] = []
    proxy_count = 0
    physical_count = 0
    for index, scenario in enumerate(scenarios[: int(config.get("required_scenario_count", len(scenarios)))]):
        if not isinstance(scenario, dict):
            continue
        scenario_id = str(scenario.get("scenario_id", f"scenario-{index:04d}"))
        selected = _select_obstacle_source_for_scenario(
            scenario_id=scenario_id,
            scenario=scenario,
            slice_row=slice_by_id.get(scenario_id, {}),
            config=config,
            repo_root=repo_root,
        )
        if selected is None:
            missing_scenarios.append(scenario_id)
            continue
        if selected["obstacle_source_is_proxy"]:
            proxy_count += 1
        else:
            physical_count += 1
        sources.append(selected)
        source_by_scenario[scenario_id] = selected
    return {
        "schema_version": "xunce-exploration-coverage-obstacle-sources/v1",
        "source_count": len(sources),
        "missing_source_scenario_count": len(missing_scenarios),
        "missing_source_scenario_ids": missing_scenarios,
        "proxy_source_count": proxy_count,
        "physical_source_count": physical_count,
        "no_go_blocks_los": bool(config.get("no_go_blocks_los", False)),
        "platform_contract_id": config.get("platform_contract_id"),
        "platform_contract_hash": config.get("platform_contract_hash"),
        "platform_max_climb_deg": config.get("platform_max_climb_deg"),
        "max_traversable_slope_deg": config.get("max_traversable_slope_deg"),
        "sources": sources,
        "source_by_scenario": source_by_scenario,
    }


def _select_obstacle_source_for_scenario(
    *,
    scenario_id: str,
    scenario: dict[str, Any],
    slice_row: dict[str, Any],
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any] | None:
    payloads = (
        ("scenario", scenario),
        ("slice", slice_row),
        ("config", config),
    )
    include_no_go = bool(config.get("no_go_blocks_los", False))
    sidecar_payload = _read_sidecar_payload(slice_row, repo_root)
    effective_source = _effective_synthetic_los_obstacle_source(
        scenario_id=scenario_id,
        payloads=payloads + (("sidecar", sidecar_payload),),
        include_no_go=include_no_go,
        config=config,
        sidecar_payload=sidecar_payload,
    )
    if effective_source is not None:
        return effective_source
    source_fields: list[tuple[str, str, bool]] = [
        ("obstacle_cells", "physical_obstacle_cells", False),
        ("obstacle_rectangles", "physical_obstacle_cells", False),
        ("slope_blocked_cells", "slope_blocked_as_obstacle_proxy", True),
        ("synthetic_los_blocker_cells", "synthetic_terrain_obstacle_proxy/v1", True),
        ("synthetic_hard_obstacle_cells", "synthetic_terrain_obstacle_proxy/v1", True),
        ("blocked_cells", "blocked_as_obstacle_proxy", True),
        ("blocked_rectangles", "blocked_as_obstacle_proxy", True),
    ]
    if include_no_go:
        source_fields.extend(
            [
                ("no_go_cells", "no_go_as_obstacle_proxy", True),
                ("no_go_rectangles", "no_go_as_obstacle_proxy", True),
            ]
        )
    for field, source_kind, is_proxy in source_fields:
        for payload_name, payload in payloads:
            if not isinstance(payload, dict):
                continue
            cells, source_field = _extract_specific_obstacle_field(payload, field, include_no_go=include_no_go)
            if not cells:
                continue
            sorted_cells = [list(cell) for cell in sorted(cells)]
            source_hash = stable_obstacle_source_hash(
                source_kind=source_kind,
                obstacle_cells=sorted_cells,
                no_go_blocks_los=include_no_go,
            )
            source_id = f"scenario:{scenario_id}:{source_kind}"
            return {
                "schema_version": "xunce-exploration-coverage-obstacle-source/v1",
                "scenario_id": scenario_id,
                "obstacle_source_id": source_id,
                "obstacle_source_hash": source_hash,
                "obstacle_source_kind": source_kind,
                "obstacle_source_field": source_field or field,
                "obstacle_source_payload": payload_name,
                "obstacle_source_is_proxy": bool(is_proxy),
                "platform_contract_id": config.get("platform_contract_id"),
                "platform_contract_hash": config.get("platform_contract_hash"),
                "platform_max_climb_deg": config.get("platform_max_climb_deg"),
                "max_traversable_slope_deg": config.get("max_traversable_slope_deg"),
                "obstacle_cell_count": len(sorted_cells),
                "obstacle_cells": sorted_cells,
                "no_go_blocks_los": include_no_go,
            }
    sidecar_source = _obstacle_source_from_sidecar(
        slice_row,
        repo_root,
        include_no_go=include_no_go,
        derive_slope_blocked=bool(config.get("derive_slope_blocked_cells_from_sidecar_dem", False)),
        max_traversable_slope_deg=float(config.get("max_traversable_slope_deg", 30.0)),
    )
    if sidecar_source is not None:
        sidecar_cells, source_kind, source_field, is_proxy = sidecar_source
        sorted_cells = [list(cell) for cell in sorted(sidecar_cells)]
        source_hash = stable_obstacle_source_hash(
            source_kind=source_kind,
            obstacle_cells=sorted_cells,
            no_go_blocks_los=include_no_go,
        )
        return {
            "schema_version": "xunce-exploration-coverage-obstacle-source/v1",
            "scenario_id": scenario_id,
            "obstacle_source_id": f"scenario:{scenario_id}:{source_kind}",
            "obstacle_source_hash": source_hash,
            "obstacle_source_kind": source_kind,
            "obstacle_source_field": source_field,
            "obstacle_source_payload": "sidecar",
            "obstacle_source_is_proxy": bool(is_proxy),
            "platform_contract_id": config.get("platform_contract_id"),
            "platform_contract_hash": config.get("platform_contract_hash"),
            "platform_max_climb_deg": config.get("platform_max_climb_deg"),
            "max_traversable_slope_deg": config.get("max_traversable_slope_deg"),
            "obstacle_cell_count": len(sorted_cells),
            "obstacle_cells": sorted_cells,
            "no_go_blocks_los": include_no_go,
        }
    return None


def _obstacle_source_from_sidecar(
    slice_row: dict[str, Any],
    repo_root: Path,
    *,
    include_no_go: bool,
    derive_slope_blocked: bool = False,
    max_traversable_slope_deg: float = 30.0,
) -> tuple[set[tuple[int, int]], str, str, bool] | None:
    payload = _read_sidecar_payload(slice_row, repo_root)
    if not payload:
        return None
    physical_fields: list[tuple[str, str, bool]] = [
        ("obstacle_cells", "physical_obstacle_cells", False),
        ("obstacle_rectangles", "physical_obstacle_cells", False),
    ]
    for field, source_kind, is_proxy in physical_fields:
        cells, source_field = _extract_specific_obstacle_field(payload, field, include_no_go=include_no_go)
        if cells:
            return cells, source_kind, source_field or field, is_proxy
    slope_fields: list[tuple[str, str, bool]] = [
        ("slope_blocked_cells", "slope_blocked_as_obstacle_proxy", True),
    ]
    for field, source_kind, is_proxy in slope_fields:
        cells, source_field = _extract_specific_obstacle_field(payload, field, include_no_go=include_no_go)
        if cells:
            return cells, source_kind, source_field or field, is_proxy
    synthetic_fields: list[tuple[str, str, bool]] = [
        ("synthetic_los_blocker_cells", "synthetic_terrain_obstacle_proxy/v1", True),
        ("synthetic_hard_obstacle_cells", "synthetic_terrain_obstacle_proxy/v1", True),
    ]
    for field, source_kind, is_proxy in synthetic_fields:
        cells, source_field = _extract_specific_obstacle_field(payload, field, include_no_go=include_no_go)
        if cells:
            return cells, source_kind, source_field or field, is_proxy
    if derive_slope_blocked:
        cells = _slope_blocked_cells_from_sidecar_dem_payload(
            payload,
            max_traversable_slope_deg=max_traversable_slope_deg,
        )
        if cells:
            return cells, "slope_blocked_as_obstacle_proxy", "sidecar_dem_slope_gt_max_traversable_deg", True
    blocked_fields: list[tuple[str, str, bool]] = [
        ("blocked_cells", "blocked_as_obstacle_proxy", True),
        ("blocked_rectangles", "blocked_as_obstacle_proxy", True),
    ]
    for field, source_kind, is_proxy in blocked_fields:
        cells, source_field = _extract_specific_obstacle_field(payload, field, include_no_go=include_no_go)
        if cells:
            return cells, source_kind, source_field or field, is_proxy
    cells = _blocked_cells_from_sidecar_passable_mask_payload(payload)
    if cells:
        return cells, "blocked_as_obstacle_proxy", "passable_mask_false", True
    if include_no_go:
        no_go_fields: list[tuple[str, str, bool]] = [
            ("no_go_cells", "no_go_as_obstacle_proxy", True),
            ("no_go_rectangles", "no_go_as_obstacle_proxy", True),
        ]
        for field, source_kind, is_proxy in no_go_fields:
            cells, source_field = _extract_specific_obstacle_field(payload, field, include_no_go=include_no_go)
            if cells:
                return cells, source_kind, source_field or field, is_proxy
    return None


def _read_sidecar_payload(slice_row: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    sidecar_path = _resolved_file(slice_row.get("sidecar"), repo_root)
    if sidecar_path is None:
        return {}
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _effective_synthetic_los_obstacle_source(
    *,
    scenario_id: str | None,
    payloads: tuple[tuple[str, dict[str, Any]], ...],
    include_no_go: bool,
    config: dict[str, Any],
    sidecar_payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    synthetic_cells = _collect_obstacle_cells_from_payloads(
        payloads,
        ("synthetic_los_blocker_cells",),
        include_no_go=include_no_go,
    )
    if not synthetic_cells:
        return None
    physical_cells = _collect_obstacle_cells_from_payloads(
        payloads,
        ("obstacle_cells", "obstacle_rectangles"),
        include_no_go=include_no_go,
    )
    slope_cells = _collect_obstacle_cells_from_payloads(
        payloads,
        ("slope_blocked_cells",),
        include_no_go=include_no_go,
    )
    if sidecar_payload and bool(config.get("derive_slope_blocked_cells_from_sidecar_dem", False)):
        slope_cells.update(
            _slope_blocked_cells_from_sidecar_dem_payload(
                sidecar_payload,
                max_traversable_slope_deg=float(config.get("max_traversable_slope_deg", 30.0)),
            )
        )
    if include_no_go:
        physical_cells.update(
            _collect_obstacle_cells_from_payloads(
                payloads,
                ("no_go_cells", "no_go_rectangles"),
                include_no_go=include_no_go,
            )
        )
    effective_cells = physical_cells | slope_cells | synthetic_cells
    if not effective_cells:
        return None
    source_kind = "synthetic_terrain_obstacle_proxy/v1"
    sorted_cells = [list(cell) for cell in sorted(effective_cells)]
    source_hash = stable_obstacle_source_hash(
        source_kind=source_kind,
        obstacle_cells=sorted_cells,
        no_go_blocks_los=include_no_go,
    )
    source_id = f"scenario:{scenario_id}:{source_kind}" if scenario_id is not None else None
    return {
        "schema_version": "xunce-exploration-coverage-obstacle-source/v1",
        "scenario_id": scenario_id,
        "obstacle_source_id": source_id,
        "obstacle_source_hash": source_hash,
        "obstacle_source_kind": source_kind,
        "obstacle_source_field": "effective_los_blocker_cells",
        "obstacle_source_payload": "effective",
        "obstacle_source_is_proxy": True,
        "obstacle_source_components": [
            component
            for component, cells in (
                ("physical_obstacle_cells", physical_cells),
                ("slope_blocked_cells", slope_cells),
                ("synthetic_los_blocker_cells", synthetic_cells),
            )
            if cells
        ],
        "platform_contract_id": config.get("platform_contract_id"),
        "platform_contract_hash": config.get("platform_contract_hash"),
        "platform_max_climb_deg": config.get("platform_max_climb_deg"),
        "max_traversable_slope_deg": config.get("max_traversable_slope_deg"),
        "obstacle_cell_count": len(sorted_cells),
        "obstacle_cells": sorted_cells,
        "synthetic_obstacle_cell_count": len(synthetic_cells),
        "slope_obstacle_cell_count": len(slope_cells),
        "physical_obstacle_cell_count": len(physical_cells),
        "no_go_blocks_los": include_no_go,
    }


def _collect_obstacle_cells_from_payloads(
    payloads: tuple[tuple[str, dict[str, Any]], ...],
    fields: tuple[str, ...],
    *,
    include_no_go: bool,
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for _, payload in payloads:
        if not isinstance(payload, dict):
            continue
        for field in fields:
            extracted, _ = _extract_specific_obstacle_field(payload, field, include_no_go=include_no_go)
            cells.update(extracted)
    return cells


def _blocked_cells_from_sidecar_passable_mask(slice_row: dict[str, Any], repo_root: Path) -> set[tuple[int, int]]:
    sidecar_path = _resolved_file(slice_row.get("sidecar"), repo_root)
    if sidecar_path is None:
        return set()
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return _blocked_cells_from_sidecar_passable_mask_payload(payload)


def _blocked_cells_from_sidecar_passable_mask_payload(payload: dict[str, Any]) -> set[tuple[int, int]]:
    mask = _find_nested_key(payload, "passable_mask")
    if not isinstance(mask, list):
        return set()
    cells: set[tuple[int, int]] = set()
    for y, row in enumerate(mask):
        if not isinstance(row, list):
            continue
        for x, value in enumerate(row):
            if value is False:
                cells.add((int(x), int(y)))
    return cells


def _slope_blocked_cells_from_sidecar_dem_payload(
    payload: dict[str, Any],
    *,
    max_traversable_slope_deg: float,
) -> set[tuple[int, int]]:
    dem = _find_nested_key(payload, "dem")
    if not _is_rectangular_number_grid(dem):
        return set()
    resolution = _sidecar_resolution_m(payload)
    cells: set[tuple[int, int]] = set()
    for y, row in enumerate(dem):
        for x, value in enumerate(row):
            if _max_neighbor_slope_deg(dem, x=x, y=y, resolution_m=resolution) > max_traversable_slope_deg:
                cells.add((int(x), int(y)))
    return cells


def _is_rectangular_number_grid(value: Any) -> bool:
    if not isinstance(value, list) or not value or not isinstance(value[0], list) or not value[0]:
        return False
    width = len(value[0])
    for row in value:
        if not isinstance(row, list) or len(row) != width:
            return False
        for item in row:
            if not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                return False
    return True


def _sidecar_resolution_m(payload: dict[str, Any]) -> float:
    for key in ("resolution_m",):
        value = _find_nested_key(payload, key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0:
            return float(value)
    return 1.0


def _max_neighbor_slope_deg(dem: list[list[float]], *, x: int, y: int, resolution_m: float) -> float:
    max_angle = 0.0
    center = float(dem[y][x])
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            yy = y + dy
            xx = x + dx
            if 0 <= yy < len(dem) and 0 <= xx < len(dem[yy]):
                distance = resolution_m * math.sqrt(float(dx * dx + dy * dy))
                if distance <= 0:
                    continue
                grade = abs(center - float(dem[yy][xx])) / distance
                max_angle = max(max_angle, math.degrees(math.atan(grade)))
    return max_angle


def _extract_specific_obstacle_field(payload: dict[str, Any], field: str, *, include_no_go: bool) -> tuple[set[tuple[int, int]], str | None]:
    return extract_obstacle_cells({field: payload.get(field)}, include_no_go=include_no_go)


def _v2_enabled(config: dict[str, Any]) -> bool:
    return (
        config["candidate_refresh_mode"] in {"dynamic_validated_only", "dynamic_frontier_nbv_in_process"}
        or config["coverage_metric_mode"] == "path_line_plus_endpoint"
        or bool(config["include_oracle_baselines"])
        or bool(config["include_roi_weighted_coverage"])
    )


def resolve_coverage_denominator(
    scenario: dict[str, Any],
    slice_row: dict[str, Any],
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    mode = str(config.get("coverage_denominator_mode", "fixed_config_cells"))
    fixed_cells = int(config["coverage_denominator_cells"])
    if mode == "raw_cells_only":
        return {
            "coverage_denominator_cells": None,
            "legacy_coverage_denominator_cells": fixed_cells,
            "coverage_denominator_mode": mode,
            "coverage_denominator_source": "raw_cells_only_no_rate_denominator/v1",
            "coverage_denominator_reason_codes": ["raw_cells_only"],
        }
    sidecar_path = _resolved_file(slice_row.get("sidecar"), repo_root)
    if mode == "main_coverable_cells":
        if sidecar_path is not None:
            semantics = _sidecar_coverable_cell_semantics(
                sidecar_path,
                derive_slope_blocked=bool(config.get("derive_slope_blocked_cells_from_sidecar_dem", False)),
                max_traversable_slope_deg=float(config.get("max_traversable_slope_deg", 30.0)),
            )
            main_coverable_count = int(semantics.get("main_coverable_denominator_cells") or 0)
            if main_coverable_count > 0:
                return {
                    **semantics,
                    "coverage_denominator_cells": main_coverable_count,
                    "legacy_coverage_denominator_cells": main_coverable_count,
                    "coverage_denominator_mode": mode,
                    "coverage_denominator_source": "main_coverable_cells/v1",
                    "coverage_denominator_reason_codes": [],
                }
        return {
            "coverage_denominator_cells": fixed_cells,
            "legacy_coverage_denominator_cells": fixed_cells,
            "coverage_denominator_mode": mode,
            "coverage_denominator_source": "fixed_config_cells_fallback_due_missing_main_coverable_cells/v1",
            "coverage_denominator_reason_codes": ["main_coverable_cells_unavailable", "fixed_config_cells_fallback"],
        }
    if mode == "roi_valid_cells":
        if sidecar_path is not None:
            valid_cells = _count_passable_mask_cells(sidecar_path)
            if valid_cells > 0:
                return {
                    "coverage_denominator_cells": valid_cells,
                    "legacy_coverage_denominator_cells": valid_cells,
                    "coverage_denominator_mode": mode,
                    "coverage_denominator_source": "sidecar_passable_mask_valid_cells/v1",
                    "coverage_denominator_reason_codes": [],
                }
        return {
            "coverage_denominator_cells": fixed_cells,
            "legacy_coverage_denominator_cells": fixed_cells,
            "coverage_denominator_mode": mode,
            "coverage_denominator_source": "fixed_config_cells_fallback_due_missing_roi_valid_cells/v1",
            "coverage_denominator_reason_codes": ["roi_valid_cells_unavailable", "fixed_config_cells_fallback"],
        }
    return {
        "coverage_denominator_cells": fixed_cells,
        "legacy_coverage_denominator_cells": fixed_cells,
        "coverage_denominator_mode": mode,
        "coverage_denominator_source": "fixed_config_cells/v1",
        "coverage_denominator_reason_codes": [],
    }


def _count_passable_mask_cells(sidecar_path: Path) -> int:
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    mask = _find_nested_key(payload, "passable_mask")
    if not isinstance(mask, list):
        return 0
    return _count_truthy_nested(mask)


def _sidecar_coverable_cell_semantics(
    sidecar_path: Path,
    *,
    derive_slope_blocked: bool,
    max_traversable_slope_deg: float,
) -> dict[str, Any]:
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    passable_cells, raw_roi_cells = _passable_and_raw_cells_from_payload(payload)
    if not passable_cells:
        return {}
    physical_cells = _cells_from_payload_fields(
        payload,
        ("physical_obstacle_cells", "obstacle_cells", "obstacle_rectangles"),
    )
    slope_cells = _cells_from_payload_fields(payload, ("slope_blocked_cells",))
    if derive_slope_blocked:
        slope_cells.update(
            _slope_blocked_cells_from_sidecar_dem_payload(
                payload,
                max_traversable_slope_deg=max_traversable_slope_deg,
            )
        )
    blocked_cells = _cells_from_payload_fields(payload, ("blocked_cells", "blocked_rectangles"))
    blocked_cells.update(_blocked_cells_from_sidecar_passable_mask_payload(payload))
    synthetic_hard_cells = _cells_from_payload_fields(
        payload,
        (
            "synthetic_hard_obstacle_cells",
            "synthetic_rock_hard_obstacle_cells",
            "synthetic_pit_hard_obstacle_cells",
        ),
    )
    synthetic_los_cells = _cells_from_payload_fields(payload, ("synthetic_los_blocker_cells",))
    synthetic_high_risk_cells = _cells_from_payload_fields(
        payload,
        ("synthetic_high_risk_cells", "synthetic_pit_high_risk_cells"),
    )
    hard_obstacle_cells = physical_cells | slope_cells | blocked_cells | synthetic_hard_cells
    traversable_cells = passable_cells - hard_obstacle_cells
    los_blocker_cells = physical_cells | slope_cells | synthetic_los_cells | synthetic_hard_cells
    main_coverable_cells = passable_cells - hard_obstacle_cells
    hazard_observable_cells = physical_cells | slope_cells | synthetic_hard_cells | synthetic_high_risk_cells
    los_only_cells = synthetic_los_cells - synthetic_hard_cells - physical_cells - slope_cells - blocked_cells
    hash_payload = {
        "schema_version": "coverable_cell_semantics_hash/v1",
        "passable_cells": _sorted_cell_lists(passable_cells),
        "hard_obstacle_cells": _sorted_cell_lists(hard_obstacle_cells),
        "los_blocker_cells": _sorted_cell_lists(los_blocker_cells),
        "main_coverable_cells": _sorted_cell_lists(main_coverable_cells),
        "hazard_observable_cells": _sorted_cell_lists(hazard_observable_cells),
    }
    semantics_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {
        "coverable_cell_semantics_hash": semantics_hash,
        "raw_roi_denominator_cells": len(raw_roi_cells),
        "passable_denominator_cells": len(passable_cells),
        "main_coverable_denominator_cells": len(main_coverable_cells),
        "hazard_observable_denominator_cells": len(hazard_observable_cells),
        "traversable_cell_count": len(traversable_cells),
        "hard_obstacle_cell_count": len(hard_obstacle_cells),
        "main_coverable_hard_obstacle_overlap_count": len(main_coverable_cells & hard_obstacle_cells),
        "los_blocker_cell_count": len(los_blocker_cells),
        "synthetic_los_only_blocker_cell_count": len(los_only_cells),
        "synthetic_los_only_blocker_passable_count": len(los_only_cells & passable_cells),
        "synthetic_los_only_blocker_main_coverable_count": len(los_only_cells & main_coverable_cells),
        "_raw_roi_cells": raw_roi_cells,
        "_passable_cells": passable_cells,
        "_main_coverable_cells": main_coverable_cells,
        "_hazard_observable_cells": hazard_observable_cells,
        "_hard_obstacle_cells": hard_obstacle_cells,
        "_synthetic_los_only_blocker_cells": los_only_cells,
    }


def _passable_and_raw_cells_from_payload(payload: dict[str, Any]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    mask = _find_nested_key(payload, "passable_mask")
    if not isinstance(mask, list):
        return set(), set()
    passable_cells: set[tuple[int, int]] = set()
    raw_cells: set[tuple[int, int]] = set()
    for y, row in enumerate(mask):
        if not isinstance(row, list):
            continue
        for x, value in enumerate(row):
            cell = (int(x), int(y))
            raw_cells.add(cell)
            if bool(value):
                passable_cells.add(cell)
    return passable_cells, raw_cells


def _cells_from_payload_fields(payload: dict[str, Any], fields: tuple[str, ...]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for field in fields:
        value = payload.get(field)
        if field.endswith("rectangles"):
            cells.update(_cells_from_rectangles_value(value))
        else:
            cells.update(_cells_from_cell_list_value(value))
    return cells


def _cells_from_cell_list_value(value: Any) -> set[tuple[int, int]]:
    if not isinstance(value, list):
        return set()
    cells: set[tuple[int, int]] = set()
    for item in value:
        cell = _cell_tuple(item)
        if cell is not None:
            cells.add(cell)
    return cells


def _cells_from_rectangles_value(value: Any) -> set[tuple[int, int]]:
    if not isinstance(value, list):
        return set()
    cells: set[tuple[int, int]] = set()
    for raw in value:
        bounds: tuple[int, int, int, int] | None = None
        if isinstance(raw, dict):
            if all(key in raw for key in ("min_x", "min_y", "max_x", "max_y")):
                bounds = (int(raw["min_x"]), int(raw["min_y"]), int(raw["max_x"]), int(raw["max_y"]))
            elif all(key in raw for key in ("x0", "y0", "x1", "y1")):
                bounds = (int(raw["x0"]), int(raw["y0"]), int(raw["x1"]), int(raw["y1"]))
        elif isinstance(raw, (list, tuple)) and len(raw) >= 4:
            try:
                bounds = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
            except (TypeError, ValueError):
                bounds = None
        if bounds is None:
            continue
        x0, y0, x1, y1 = bounds
        for x in range(min(x0, x1), max(x0, x1) + 1):
            for y in range(min(y0, y1), max(y0, y1) + 1):
                cells.add((x, y))
    return cells


def _sorted_cell_lists(cells: set[tuple[int, int]]) -> list[list[int]]:
    return [[int(x), int(y)] for x, y in sorted(cells)]


def _find_nested_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_nested_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_nested_key(child, key)
            if found is not None:
                return found
    return None


def _count_truthy_nested(value: Any) -> int:
    if isinstance(value, list):
        return sum(_count_truthy_nested(item) for item in value)
    return 1 if bool(value) else 0


def _coverage_rate_from_count(cell_count: int, denominator_context: dict[str, Any]) -> float | None:
    denominator = denominator_context.get("coverage_denominator_cells")
    if denominator is None:
        return None
    numeric = _finite_or_none(denominator)
    if numeric is None or float(numeric) <= TOLERANCE:
        return None
    return float(cell_count) / float(numeric)


def _coverage_count_for_denominator(covered_cells: set[tuple[int, int]], denominator_context: dict[str, Any]) -> int:
    main_coverable_cells = denominator_context.get("_main_coverable_cells")
    if denominator_context.get("coverage_denominator_mode") == "main_coverable_cells" and isinstance(main_coverable_cells, set):
        return len(covered_cells & main_coverable_cells)
    return len(covered_cells)


def _coverage_denominator_episode_fields(
    covered_cells: set[tuple[int, int]],
    denominator_context: dict[str, Any],
) -> dict[str, Any]:
    main_cells = denominator_context.get("_main_coverable_cells")
    passable_cells = denominator_context.get("_passable_cells")
    raw_cells = denominator_context.get("_raw_roi_cells")
    hazard_cells = denominator_context.get("_hazard_observable_cells")

    main_count = len(covered_cells & main_cells) if isinstance(main_cells, set) else _coverage_count_for_denominator(covered_cells, denominator_context)
    passable_count = len(covered_cells & passable_cells) if isinstance(passable_cells, set) else None
    raw_count = len(covered_cells & raw_cells) if isinstance(raw_cells, set) else len(covered_cells)
    hazard_count = len(covered_cells & hazard_cells) if isinstance(hazard_cells, set) else 0

    return {
        "main_coverable_denominator_cells": denominator_context.get("main_coverable_denominator_cells"),
        "main_covered_cell_count": main_count,
        "main_coverage_rate": _safe_ratio(main_count, denominator_context.get("main_coverable_denominator_cells")),
        "raw_roi_denominator_cells": denominator_context.get("raw_roi_denominator_cells"),
        "raw_roi_coverage_rate": _safe_ratio(raw_count, denominator_context.get("raw_roi_denominator_cells")),
        "passable_denominator_cells": denominator_context.get("passable_denominator_cells"),
        "passable_covered_cell_count": passable_count,
        "passable_coverage_rate": _safe_ratio(passable_count, denominator_context.get("passable_denominator_cells")),
        "hazard_observable_denominator_cells": denominator_context.get("hazard_observable_denominator_cells"),
        "hazard_observed_cell_count": hazard_count,
        "hazard_observation_rate": _safe_ratio(hazard_count, denominator_context.get("hazard_observable_denominator_cells")),
        "coverable_cell_semantics_hash": denominator_context.get("coverable_cell_semantics_hash"),
        "traversable_cell_count": denominator_context.get("traversable_cell_count"),
        "hard_obstacle_cell_count": denominator_context.get("hard_obstacle_cell_count"),
        "main_coverable_hard_obstacle_overlap_count": denominator_context.get("main_coverable_hard_obstacle_overlap_count"),
        "los_blocker_cell_count": denominator_context.get("los_blocker_cell_count"),
        "synthetic_los_only_blocker_cell_count": denominator_context.get("synthetic_los_only_blocker_cell_count"),
        "synthetic_los_only_blocker_passable_count": denominator_context.get("synthetic_los_only_blocker_passable_count"),
        "synthetic_los_only_blocker_main_coverable_count": denominator_context.get(
            "synthetic_los_only_blocker_main_coverable_count"
        ),
    }


def _public_coverage_denominator_context(denominator_context: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in denominator_context.items() if not str(key).startswith("_")}


def _coverage_rate_capped(rate: float | None) -> float | None:
    if rate is None:
        return None
    return min(float(rate), 1.0)


def _coverage_saturation_excess(rate: float | None) -> float:
    if rate is None:
        return 0.0
    return max(0.0, float(rate) - 1.0)


def _coverage_curve_auc_nullable(coverage_rates: list[float | None], rollout_steps: int) -> float | None:
    if any(value is None for value in coverage_rates):
        return None
    return _coverage_curve_auc([float(value) for value in coverage_rates if value is not None], rollout_steps)


def _run_coverage_rollouts(
    config: dict[str, Any],
    source: dict[str, Any],
    model_bundle: dict[str, Any],
    *,
    obstacle_source_audit: dict[str, Any],
    repo_root: Path,
    output_root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    scenarios = source["path_feedback"].get("scenarios", [])
    if not isinstance(scenarios, list):
        scenarios = []
    slice_by_id = {str(row.get("scenario_id")): row for row in source["slices"] if isinstance(row, dict)}
    obstacle_source_by_scenario = obstacle_source_audit.get("source_by_scenario", {})
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    dynamic_proposals: list[dict[str, Any]] = []
    dynamic_validation_rows: list[dict[str, Any]] = []
    paired_decision_rows: list[dict[str, Any]] = []
    candidate_metric_rows: list[dict[str, Any]] = []
    on_policy_teacher_label_rows: list[dict[str, Any]] = []
    validation_cache: dict[str, list[dict[str, Any]]] = {}
    worker_count = int(config.get("hybrid_astar_candidate_eval_workers", 1))
    executor_context = (
        ProcessPoolExecutor(max_workers=worker_count)
        if bool(config.get("hybrid_astar_pose_path_cost_enabled")) and worker_count > 1
        else nullcontext(None)
    )
    with executor_context as hybrid_astar_executor:
        for scenario_index, scenario in enumerate(scenarios[: config["required_scenario_count"]]):
            if not isinstance(scenario, dict):
                continue
            scenario_id = str(scenario.get("scenario_id", f"scenario-{scenario_index:04d}"))
            slice_row = slice_by_id.get(scenario_id, {})
            obstacle_source_linkage = obstacle_source_by_scenario.get(scenario_id)
            roi_group = resolve_roi_group(scenario, slice_row)
            if config.get("xunce_only_evaluation"):
                policy_names = ("xunce",)
            else:
                policy_names = POLICIES + (ORACLE_POLICIES if config["include_oracle_baselines"] else ())
                if config["include_canonical_reward_rerank_oracle"]:
                    policy_names = policy_names + (CANONICAL_REWARD_RERANK_ORACLE,)
            for policy_name in policy_names:
                episode = _run_policy_episode(
                    policy_name=policy_name,
                    scenario=scenario,
                    scenario_index=scenario_index,
                    scenario_id=scenario_id,
                    roi_group=roi_group,
                    split=slice_row.get("split"),
                    config=config,
                    model_bundle=model_bundle,
                    slice_row=slice_row,
                    obstacle_source_linkage=obstacle_source_linkage,
                    repo_root=repo_root,
                    output_root=output_root,
                    validation_cache=validation_cache,
                    hybrid_astar_executor=hybrid_astar_executor,
                )
                episodes.append(episode)
                steps.extend(episode.get("steps", []))
                inference_rows.extend(episode.get("inference_rows", []))
                dynamic_proposals.extend(episode.get("dynamic_proposals", []))
                dynamic_validation_rows.extend(episode.get("dynamic_validation_rows", []))
                paired_decision_rows.extend(episode.get("paired_decision_rows", []))
                candidate_metric_rows.extend(episode.get("candidate_metric_rows", []))
                on_policy_teacher_label_rows.extend(episode.get("on_policy_teacher_label_rows", []))
    return (
        episodes,
        steps,
        inference_rows,
        dynamic_proposals,
        dynamic_validation_rows,
        paired_decision_rows,
        candidate_metric_rows,
        on_policy_teacher_label_rows,
    )


def _run_policy_episode(
    *,
    policy_name: str,
    scenario: dict[str, Any],
    scenario_index: int,
    scenario_id: str,
    roi_group: str,
    split: Any,
    config: dict[str, Any],
    model_bundle: dict[str, Any],
    slice_row: dict[str, Any],
    obstacle_source_linkage: dict[str, Any] | None,
    repo_root: Path,
    output_root: Path,
    validation_cache: dict[str, list[dict[str, Any]]],
    hybrid_astar_executor: Any | None = None,
) -> dict[str, Any]:
    denominator_context = resolve_coverage_denominator(scenario, slice_row, config, repo_root)
    denominator = float(denominator_context["legacy_coverage_denominator_cells"])
    radius = int(config["coverage_radius_cells"])
    start_cell = _cell_tuple(scenario.get("start_cell")) or (0, 0)
    scenario_diversity_metadata = {
        "scenario_seed": scenario.get("scenario_seed"),
        "scenario_start_cell": scenario.get("scenario_start_cell") or list(start_cell),
        "scenario_start_cell_source": scenario.get("scenario_start_cell_source"),
        "scenario_roi_id": scenario.get("scenario_roi_id"),
        "scenario_candidate_seed": scenario.get("scenario_candidate_seed"),
        "scenario_diversity_source": scenario.get("scenario_diversity_source")
        or config.get("scenario_diversity_source"),
        "scenario_diversity_signature_hash": scenario.get("scenario_diversity_signature_hash"),
        "scenario_diversity_content_hash": scenario.get("scenario_diversity_content_hash"),
    }
    covered_cells = set(_footprint(start_cell, radius=radius))
    initial_coverage_count = _coverage_count_for_denominator(covered_cells, denominator_context)
    coverage_rates = [initial_coverage_count / denominator]
    coverage_rate_raw_values: list[float | None] = [_coverage_rate_from_count(initial_coverage_count, denominator_context)]
    coverage_rate_capped_values: list[float | None] = [_coverage_rate_capped(coverage_rate_raw_values[-1])]
    steps: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    reason_codes: list[str] = []
    path_cost_total = 0.0
    risk_total = 0.0
    risk_cost_weighted_total = 0.0
    soft_risk_exposure_total = 0.0
    high_risk_distance_total_m = 0.0
    path_risk_peaks: list[float] = []
    hard_risk_violation_count = 0
    risk_boundary_violation_steps = 0
    energy_total = 0.0
    new_cell_total = 0
    raw_new_cell_total = 0
    revisited_cell_total = 0
    coverage_return = 0.0
    valuable_area_covered = 0.0
    selected_probabilities: list[float] = []
    selected_ranks: list[int] = []
    entropies: list[float] = []
    latencies: list[float] = []
    action_indices: list[int | None] = []
    mask_violation_count = 0
    unreachable_selected_count = 0
    path_planning_failure_count = 0
    open_grid_fallback_count = 0
    model_inference_failure_count = 0
    candidate_generation_exhausted_count = 0
    episode_termination_reason: str | None = None
    candidate_generation_exhausted_step: int | None = None
    current_cell = start_cell
    dynamic_proposals: list[dict[str, Any]] = []
    dynamic_validation_rows: list[dict[str, Any]] = []
    paired_decision_rows: list[dict[str, Any]] = []
    candidate_metric_rows: list[dict[str, Any]] = []
    on_policy_teacher_label_rows: list[dict[str, Any]] = []
    current_theta_deg = float(config.get("initial_theta_deg", 0.0))

    for step_index in range(config["rollout_steps"]):
        cell_before = current_cell
        candidate_batch = _candidate_rows_for_step(
            scenario,
            current_cell=current_cell,
            current_theta_deg=current_theta_deg,
            covered_cells=covered_cells,
            step_index=step_index,
            config=config,
            scenario_id=scenario_id,
            slice_row=slice_row,
            repo_root=repo_root,
            output_root=output_root,
            validation_cache=validation_cache,
            hybrid_astar_executor=hybrid_astar_executor,
        )
        candidates = candidate_batch["candidates"]
        candidate_set_id = candidate_batch["candidate_set_id"]
        candidate_set_hash_value = candidate_batch["candidate_set_hash"]
        covered_hash = covered_cells_hash(covered_cells)
        dynamic_proposals.extend(
            [
                _dynamic_artifact_row(
                    row,
                    scenario_id=scenario_id,
                    policy=policy_name,
                    policy_step=step_index,
                    current_cell=cell_before,
                    covered_cells_hash_value=covered_hash,
                    candidate_set_id=candidate_set_id,
                    candidate_set_hash_value=candidate_set_hash_value,
                )
                for row in candidate_batch["dynamic_proposals"]
            ]
        )
        dynamic_validation_rows.extend(
            [
                _dynamic_artifact_row(
                    row,
                    scenario_id=scenario_id,
                    policy=policy_name,
                    policy_step=step_index,
                    current_cell=cell_before,
                    covered_cells_hash_value=covered_hash,
                    candidate_set_id=candidate_set_id,
                    candidate_set_hash_value=candidate_set_hash_value,
                )
                for row in candidate_batch["dynamic_validation_rows"]
            ]
        )
        remaining_budget = max(0.0, float(config["path_budget_m"]) - path_cost_total)
        if (
            config["candidate_refresh_mode"] == "dynamic_frontier_nbv_in_process"
            and candidate_batch["dynamic_generation_executed"]
            and not candidates
        ):
            step_reasons = ["candidate_generation_exhausted", "no_valid_dynamic_candidates"]
            coverage_rates.append(coverage_rates[-1])
            coverage_rate_raw_values.append(coverage_rate_raw_values[-1])
            coverage_rate_capped_values.append(coverage_rate_capped_values[-1])
            action_indices.append(None)
            candidate_generation_exhausted_count += 1
            episode_termination_reason = "candidate_generation_exhausted"
            candidate_generation_exhausted_step = step_index
            step_row = {
                "schema_version": "xunce-exploration-coverage-step/v1",
                "scenario_id": scenario_id,
                "roi_group": roi_group,
                "split": split,
                "policy": policy_name,
                "step_index": step_index,
                "current_cell_before": list(cell_before),
                "remaining_budget_m": remaining_budget,
                "selected_action_index": None,
                "selected_cell": None,
                "candidate_cells": [],
                "candidate_generation_source": candidate_batch["candidate_generation_source"],
                "candidate_set_id": candidate_set_id,
                "candidate_set_hash": candidate_set_hash_value,
                "covered_cells_hash": covered_hash,
                "state_conditioned_candidate_generation": bool(config.get("state_conditioned_candidate_generation", True)),
                "dynamic_proposal_count": candidate_batch["dynamic_proposal_count"],
                "dynamic_validated_candidate_count": 0,
                "dynamic_validation_cache_hit": candidate_batch["dynamic_validation_cache_hit"],
                "path_feedback_validation_source_counts": candidate_batch["path_feedback_validation_source_counts"],
                "frontier_candidate_source_counts": candidate_batch["frontier_candidate_source_counts"],
                "selected_probability": 0.0,
                "selected_rank": 0,
                "action_entropy": 0.0,
                "path_cost": None,
                "risk": None,
                "energy_cost": None,
                "coverage_rate_delta": 0.0,
                "cumulative_coverage_rate_delta": coverage_rates[-1] - coverage_rates[0],
                "final_coverage_rate": coverage_rates[-1],
                "raw_covered_cell_count": len(covered_cells),
                "raw_new_covered_cell_count": 0,
                "coverage_rate_raw": coverage_rate_raw_values[-1],
                "coverage_rate_capped": coverage_rate_capped_values[-1],
                "coverage_saturation_exceeded": bool(
                    coverage_rate_raw_values[-1] is not None and coverage_rate_raw_values[-1] > 1.0 + TOLERANCE
                ),
                "coverage_saturation_excess": _coverage_saturation_excess(coverage_rate_raw_values[-1]),
                "coverage_denominator_cells": denominator_context["coverage_denominator_cells"],
                "coverage_denominator_mode": denominator_context["coverage_denominator_mode"],
                "coverage_denominator_source": denominator_context["coverage_denominator_source"],
                "coverage_denominator_reason_codes": denominator_context["coverage_denominator_reason_codes"],
                **_coverage_denominator_episode_fields(covered_cells, denominator_context),
                "new_covered_cell_count": 0,
                "revisited_cell_count": 0,
                "coverage_gain_per_path_cost": None,
                "coverage_gain_per_risk": None,
                "policy_inference_kind": MODEL_POLICY_INFERENCE_KIND if policy_name not in ORACLE_POLICIES else ORACLE_POLICY_INFERENCE_KIND,
                "oracle_rollout_executed": False,
                "true_model_inference_executed": False,
                "dynamic_candidate_validation_missing": False,
                "finite_outputs": False,
                "model_inference_failure": False,
                "model_inference_mask_violation": False,
                "candidate_generation_exhausted": True,
                "terminal_reason": "candidate_generation_exhausted",
                "executed": False,
                "reason_codes": step_reasons,
                **scenario_diversity_metadata,
            }
            steps.append(step_row)
            reason_codes.extend(step_reasons)
            break
        scenario_state = dict(scenario)
        scenario_state["coverage_rate"] = coverage_rates[-1]
        scenario_state["coverage_rate_delta"] = steps[-1]["coverage_rate_delta"] if steps else 0.0
        adapter = _scenario_to_model_inputs(
            scenario_state,
            candidates,
            scenario_index + step_index,
            model_bundle.get("xunce_config") or {},
        )
        synthetic_feature_metadata = _apply_synthetic_credit_feature_exposure_to_adapter(
            adapter,
            candidates=candidates,
            current_cell=cell_before,
            current_theta_deg=current_theta_deg,
            covered_cells=covered_cells,
            candidate_set_hash_value=candidate_set_hash_value,
            config=config,
            slice_row=slice_row,
            obstacle_source_linkage=obstacle_source_linkage,
            repo_root=repo_root,
            hybrid_astar_executor=hybrid_astar_executor,
        )
        if config["emit_candidate_metric_audit"]:
            candidate_metric_rows.extend(
                _candidate_metric_audit_rows(
                    candidates,
                    scenario_id=scenario_id,
                    roi_group=roi_group,
                    split=split,
                    policy_name=policy_name,
                    step_index=step_index,
                    current_cell=cell_before,
                    covered_cells=covered_cells,
                    covered_cells_hash_value=covered_hash,
                    candidate_set_id=candidate_set_id,
                    candidate_set_hash_value=candidate_set_hash_value,
                    action_mask=adapter["action_mask"],
                    config=config,
                    obstacle_source_linkage=obstacle_source_linkage,
                )
            )
        step_reasons: list[str] = []
        detail: dict[str, Any] | None = None
        true_model_inference_executed = False
        is_oracle_policy = _is_oracle_policy(policy_name)
        oracle_rollout_executed = False
        policy_inference_kind = ORACLE_POLICY_INFERENCE_KIND if is_oracle_policy else MODEL_POLICY_INFERENCE_KIND
        model_inference_failure = False
        dynamic_candidate_validation_missing = any(candidate.get("dynamic_candidate_validation_missing") is True for candidate in candidates)
        if dynamic_candidate_validation_missing:
            step_reasons.append("dynamic_candidate_validation_missing")
        if config["candidate_refresh_mode"] == "dynamic_frontier_nbv_in_process":
            if not candidate_batch["dynamic_generation_executed"]:
                step_reasons.append("dynamic_candidate_generation_missing")
        if is_oracle_policy:
            oracle_detail = _oracle_policy_detail(
                policy_name,
                candidates,
                current_cell=current_cell,
                covered_cells=covered_cells,
                coverage_denominator=float(denominator),
                config=config,
            )
            true_model_inference_executed = False
            oracle_rollout_executed = True
            detail = oracle_detail
        elif not model_bundle["xunce_checkpoint_loaded"] or not model_bundle["incumbent_checkpoint_loaded"]:
            step_reasons.append("true_model_inference_not_executed")
        elif not adapter["has_valid_action"]:
            step_reasons.append("no_valid_action")
        else:
            try:
                if policy_name == "xunce":
                    detail = _score_xunce_model(model_bundle["xunce_model"], adapter["xunce_batch"])
                else:
                    detail = _policy_detail_to_dict(model_bundle["incumbent_scorer"].score_detail(adapter["incumbent_observation"]))
                true_model_inference_executed = True
            except Exception as exc:  # pragma: no cover
                model_inference_failure = True
                model_inference_failure_count += 1
                step_reasons.append("model_inference_failure")
                step_reasons.append(f"true_model_inference_failed:{type(exc).__name__}")
                step_reasons.append("true_model_inference_not_executed")

        paired_row = _paired_decision_audit_row(
            scenario_id=scenario_id,
            roi_group=roi_group,
            split=split,
            executing_policy=policy_name,
            step_index=step_index,
            current_cell=cell_before,
            candidate_set_id=candidate_set_id,
            candidate_set_hash_value=candidate_set_hash_value,
            covered_cells_hash_value=covered_hash,
            candidates=candidates,
            adapter=adapter,
            model_bundle=model_bundle,
        )
        if paired_row:
            paired_decision_rows.append(paired_row)

        detail_payload = detail or _empty_model_detail()
        if (
            not is_oracle_policy
            and policy_name == "xunce"
            and true_model_inference_executed
            and continuous_theta_enabled(config)
            and detail_payload.get("theta_mu_rad")
        ):
            _bind_continuous_theta_mu_to_candidate_set(
                candidates,
                detail_payload=detail_payload,
                candidate_set_hash_value=candidate_set_hash_value,
            )
            if config.get("hybrid_astar_pose_path_cost_enabled"):
                _enrich_candidates_with_hybrid_astar_path_cost(
                    candidates,
                    current_cell=cell_before,
                    current_theta_deg=current_theta_deg,
                    candidate_set_hash_value=candidate_set_hash_value,
                    step_index=step_index,
                    scenario_id=scenario_id,
                    config=config,
                    slice_row=slice_row,
                    repo_root=repo_root,
                    hybrid_astar_executor=hybrid_astar_executor,
                )
                _apply_continuous_theta_hybrid_reachable_eval_policy(
                    detail_payload,
                    candidates=candidates,
                    action_mask=adapter["action_mask"],
                    current_cell=cell_before,
                    current_theta_deg=current_theta_deg,
                    candidate_set_hash_value=candidate_set_hash_value,
                    step_index=step_index,
                    scenario_id=scenario_id,
                    config=config,
                    slice_row=slice_row,
                    repo_root=repo_root,
                    hybrid_astar_executor=hybrid_astar_executor,
                )
        selected_index = detail["selected_action_index"] if is_oracle_policy and detail is not None else _selected_index(detail)
        if (
            not is_oracle_policy
            and true_model_inference_executed
            and not bool(detail_payload.get("finite_outputs"))
        ):
            model_inference_failure = True
            model_inference_failure_count += 1
            step_reasons.append("model_inference_failure")
            step_reasons.append("model_inference_non_finite_output")
        selected_candidate = _candidate_at(candidates, selected_index)
        if (
            not is_oracle_policy
            and selected_candidate is not None
            and selected_index is not None
            and continuous_theta_enabled(config)
            and detail_payload.get("selected_theta_rad") is not None
        ):
            _bind_continuous_theta_to_selected_candidate(
                selected_candidate,
                selected_index=int(selected_index),
                detail_payload=detail_payload,
                candidate_set_hash_value=candidate_set_hash_value,
            )
            if config.get("hybrid_astar_pose_path_cost_enabled"):
                _enrich_candidates_with_hybrid_astar_path_cost(
                    candidates,
                    current_cell=cell_before,
                    current_theta_deg=current_theta_deg,
                    candidate_set_hash_value=candidate_set_hash_value,
                    step_index=step_index,
                    scenario_id=scenario_id,
                    config=config,
                    slice_row=slice_row,
                    repo_root=repo_root,
                    hybrid_astar_executor=hybrid_astar_executor,
                )
        selected_cell = _cell_tuple(_candidate_cell(selected_candidate)) if selected_candidate is not None else None
        selected_cost = _candidate_cost(selected_candidate) if selected_candidate is not None else None
        selected_risk = _finite_or_none(selected_candidate.get("risk")) if selected_candidate is not None else None
        selected_soft_risk_exposure = _finite_or_none((selected_candidate or {}).get("soft_risk_exposure"))
        if selected_soft_risk_exposure is None:
            selected_soft_risk_exposure = _finite_or_none((selected_candidate or {}).get("path_risk_exposure"))
        if selected_soft_risk_exposure is None and selected_cost is not None and selected_risk is not None:
            selected_soft_risk_exposure = float(selected_cost) * float(selected_risk)
        selected_path_risk_peak = _finite_or_none((selected_candidate or {}).get("path_risk_peak"))
        selected_high_risk_distance_m = _finite_or_none((selected_candidate or {}).get("high_risk_distance_m")) or 0.0
        selected_path_allowed_by_risk = (selected_candidate or {}).get("path_allowed_by_risk")
        selected_hard_risk_flags = [str(flag) for flag in ((selected_candidate or {}).get("hard_risk_flags") or [])]
        selected_hard_risk_violation = bool(
            selected_candidate is not None
            and (selected_path_allowed_by_risk is False or selected_hard_risk_flags)
        )
        selected_risk_source = str((selected_candidate or {}).get("risk_source") or "unavailable")
        selected_risk_route_derived = bool((selected_candidate or {}).get("risk_route_derived", False))
        selected_planner_validation_backend = str((selected_candidate or {}).get("planner_validation_backend") or "")
        selected_validation_evidence_kind = str((selected_candidate or {}).get("validation_evidence_kind") or "")
        selected_energy = _finite_or_none(selected_candidate.get("energy_cost")) if selected_candidate is not None else None
        selected_value = _finite_or_none(selected_candidate.get("value")) if selected_candidate is not None else None
        mask_violation = selected_index is not None and _mask_violation(adapter["action_mask"], selected_index)
        if mask_violation:
            mask_violation_count += 1
            step_reasons.append("model_inference_mask_violation")
        if model_inference_failure:
            pass
        elif selected_candidate is None or selected_cell is None:
            path_planning_failure_count += 1
            step_reasons.append("path_planning_failure")
        elif not _candidate_is_valid(selected_candidate):
            unreachable_selected_count += 1
            step_reasons.append("unreachable_selected_candidate")
        if (scenario.get("open_grid_fallback_used") is True or (selected_candidate or {}).get("open_grid_fallback_used") is True) and not config["allow_open_grid_fallback"]:
            open_grid_fallback_count += 1
            step_reasons.append("open_grid_fallback_forbidden")
        if selected_cost is not None and path_cost_total + float(selected_cost) > float(config["path_budget_m"]):
            step_reasons.append("path_budget_exhausted")

        teacher_label_row = _on_policy_oracle_teacher_label_row(
            scenario_id=scenario_id,
            roi_group=roi_group,
            split=split,
            policy_name=policy_name,
            step_index=step_index,
            current_cell=cell_before,
            covered_cells=covered_cells,
            covered_cells_hash_value=covered_hash,
            candidate_set_id=candidate_set_id,
            candidate_set_hash_value=candidate_set_hash_value,
            candidates=candidates,
            xunce_selected_index=selected_index,
            true_model_inference_executed=true_model_inference_executed,
            model_inference_failure=model_inference_failure,
            detail_payload=detail_payload,
            coverage_denominator=float(denominator),
            config=config,
        )
        if teacher_label_row is not None:
            on_policy_teacher_label_rows.append(teacher_label_row)

        policy_execution_ready = oracle_rollout_executed if is_oracle_policy else true_model_inference_executed
        executed = (
            policy_execution_ready
            and detail_payload["finite_outputs"]
            and selected_candidate is not None
            and selected_cell is not None
            and selected_cost is not None
            and not mask_violation
            and "unreachable_selected_candidate" not in step_reasons
            and "open_grid_fallback_forbidden" not in step_reasons
            and "path_budget_exhausted" not in step_reasons
        )
        new_cells: set[tuple[int, int]] = set()
        revisited_cells: set[tuple[int, int]] = set()
        coverage_delta = 0.0
        if executed and selected_cell is not None and selected_cost is not None:
            footprint = _candidate_coverage_cells(
                start=cell_before,
                end=selected_cell,
                candidate=selected_candidate or {},
                config=config,
                obstacle_source_linkage=obstacle_source_linkage,
            )
            new_cells = footprint - covered_cells
            revisited_cells = footprint & covered_cells
            covered_cells.update(footprint)
            current_cell = selected_cell
            path_cost_total += float(selected_cost)
            risk_total += _float_default(selected_risk)
            risk_cost_weighted_total += float(selected_cost) * _float_default(selected_risk)
            soft_risk_exposure_total += _float_default(selected_soft_risk_exposure)
            high_risk_distance_total_m += _float_default(selected_high_risk_distance_m)
            if selected_path_risk_peak is not None:
                path_risk_peaks.append(float(selected_path_risk_peak))
            if selected_hard_risk_violation:
                hard_risk_violation_count += 1
                risk_boundary_violation_steps += 1
            energy_total += _float_default(selected_energy)
            new_cells_for_rate = _coverage_count_for_denominator(new_cells, denominator_context)
            raw_new_cell_total += len(new_cells)
            new_cell_total += new_cells_for_rate
            revisited_cell_total += len(revisited_cells)
            coverage_delta = new_cells_for_rate / denominator
            coverage_return += coverage_delta
            valuable_area_covered += new_cells_for_rate * _float_default(selected_value)
            selected_theta = _finite_or_none((selected_candidate or {}).get("candidate_theta_deg"))
            if selected_theta is not None:
                current_theta_deg = float(selected_theta)
        covered_count_for_rate = _coverage_count_for_denominator(covered_cells, denominator_context)
        coverage_rates.append(covered_count_for_rate / denominator)
        coverage_rate_raw_values.append(_coverage_rate_from_count(covered_count_for_rate, denominator_context))
        coverage_rate_capped_values.append(_coverage_rate_capped(coverage_rate_raw_values[-1]))
        if detail is not None:
            selected_probabilities.append(float(detail_payload["selected_probability"]))
            selected_ranks.append(int(detail_payload["selected_rank"]))
            entropies.append(_entropy(detail_payload["action_probs"]))
            latencies.append(float(detail_payload["latency_ms"]))
        action_indices.append(selected_index)

        cumulative_delta = coverage_rates[-1] - coverage_rates[0]
        step_row = {
            "schema_version": "xunce-exploration-coverage-step/v1",
            "scenario_id": scenario_id,
            **scenario_diversity_metadata,
            "roi_group": roi_group,
            "split": split,
            "policy": policy_name,
            "step_index": step_index,
            "current_cell_before": list(cell_before),
            "remaining_budget_m": remaining_budget,
            "selected_action_index": selected_index,
            "selected_cell": list(selected_cell) if selected_cell is not None else None,
            "candidate_cells": candidate_observation_cells(candidates),
            **theta_metadata(candidates),
            "candidate_generation_source": candidate_batch["candidate_generation_source"],
            "candidate_set_id": candidate_set_id,
            "candidate_set_hash": candidate_set_hash_value,
            "covered_cells_hash": covered_hash,
            "state_conditioned_candidate_generation": bool(config.get("state_conditioned_candidate_generation", True))
            and config["candidate_refresh_mode"] == "dynamic_frontier_nbv_in_process",
            "dynamic_proposal_count": candidate_batch["dynamic_proposal_count"],
            "dynamic_validated_candidate_count": candidate_batch["dynamic_validated_candidate_count"],
            "dynamic_validation_cache_hit": candidate_batch["dynamic_validation_cache_hit"],
            "path_feedback_validation_source_counts": candidate_batch["path_feedback_validation_source_counts"],
            "frontier_candidate_source_counts": candidate_batch["frontier_candidate_source_counts"],
            "selected_probability": detail_payload["selected_probability"],
            "selected_rank": detail_payload["selected_rank"],
            "action_entropy": entropies[-1] if entropies else 0.0,
            "selected_reward_components": detail_payload.get("reward_components"),
            "selected_reward_profile_id": detail_payload.get("profile_id"),
            "selected_reward_profile_hash": detail_payload.get("profile_hash"),
            "path_cost": selected_cost,
            "risk": selected_risk,
            "path_allowed_by_risk": selected_path_allowed_by_risk,
            "hard_risk_flags": selected_hard_risk_flags,
            "soft_risk_exposure": selected_soft_risk_exposure,
            "path_risk_exposure": selected_soft_risk_exposure,
            "path_risk_peak": selected_path_risk_peak,
            "high_risk_distance_m": selected_high_risk_distance_m,
            "risk_boundary_violation": selected_hard_risk_violation,
            "selected_risk_source": selected_risk_source,
            "selected_risk_route_derived": selected_risk_route_derived,
            "selected_planner_validation_backend": selected_planner_validation_backend,
            "selected_validation_evidence_kind": selected_validation_evidence_kind,
            "energy_cost": selected_energy,
            "coverage_rate_delta": coverage_delta,
            "cumulative_coverage_rate_delta": cumulative_delta,
            "final_coverage_rate": coverage_rates[-1],
            "raw_covered_cell_count": len(covered_cells),
            "raw_new_covered_cell_count": len(new_cells),
            "coverage_rate_raw": coverage_rate_raw_values[-1],
            "coverage_rate_capped": coverage_rate_capped_values[-1],
            "coverage_saturation_exceeded": bool(
                coverage_rate_raw_values[-1] is not None and coverage_rate_raw_values[-1] > 1.0 + TOLERANCE
            ),
            "coverage_saturation_excess": _coverage_saturation_excess(coverage_rate_raw_values[-1]),
            "coverage_denominator_cells": denominator_context["coverage_denominator_cells"],
            "coverage_denominator_mode": denominator_context["coverage_denominator_mode"],
            "coverage_denominator_source": denominator_context["coverage_denominator_source"],
            "coverage_denominator_reason_codes": denominator_context["coverage_denominator_reason_codes"],
            **_coverage_denominator_episode_fields(covered_cells, denominator_context),
            "new_covered_cell_count": _coverage_count_for_denominator(new_cells, denominator_context),
            "revisited_cell_count": len(revisited_cells),
            "coverage_gain_per_path_cost": _safe_ratio(coverage_delta, selected_cost),
            "coverage_gain_per_risk": _safe_ratio(coverage_delta, selected_risk),
            "policy_inference_kind": policy_inference_kind,
            "oracle_rollout_executed": oracle_rollout_executed,
            "true_model_inference_executed": true_model_inference_executed,
            "dynamic_candidate_validation_missing": dynamic_candidate_validation_missing,
            "finite_outputs": detail_payload["finite_outputs"],
            "model_inference_failure": model_inference_failure,
            "model_inference_mask_violation": mask_violation,
            "candidate_generation_exhausted": False,
            "terminal_reason": None,
            "executed": executed,
            "reason_codes": unique_sorted(step_reasons),
        }
        steps.append(step_row)
        selected_candidate = candidates[selected_index] if isinstance(selected_index, int) and 0 <= selected_index < len(candidates) else {}
        selected_obstacle_audit = _candidate_obstacle_audit(
            candidate=selected_candidate,
            config=config,
            cell=selected_cell,
            obstacle_source_linkage=obstacle_source_linkage,
        )
        if not is_oracle_policy:
            inference_rows.append(
                {
                    "schema_version": "xunce-exploration-coverage-model-inference/v1",
                    "input_source": "high_fidelity_scenario_adapter/v1",
                    "scenario_id": scenario_id,
                    **scenario_diversity_metadata,
                    "roi_group": roi_group,
                    "policy": policy_name,
                    "step_index": step_index,
                    "current_cell": list(cell_before),
                    "current_cell_before": list(cell_before),
                    "covered_cells_hash": covered_hash,
                    "candidate_cells": candidate_observation_cells(candidates),
                    **theta_metadata(candidates),
                    "candidate_set_id": candidate_set_id,
                    "candidate_set_hash": candidate_set_hash_value,
                    "action_mask": list(adapter["action_mask"]),
                    "selected_action_index": selected_index,
                    "candidate_viewpoint": selected_candidate.get("candidate_viewpoint"),
                    "candidate_theta_deg": selected_candidate.get("candidate_theta_deg"),
                    "selected_viewpoint": selected_candidate.get("candidate_viewpoint"),
                    "selected_theta_rad": selected_candidate.get("selected_theta_rad"),
                    "selected_theta_deg": selected_candidate.get("candidate_theta_deg"),
                    "reachable": selected_candidate.get("reachable"),
                    "path_cost": selected_cost,
                    "selected_path_cost": selected_cost,
                    "action_space_type": selected_candidate.get("action_space_type")
                    or detail_payload.get("action_space_type")
                    or config.get("action_space_type"),
                    "theta_mu_rad": list(detail_payload.get("theta_mu_rad") or []),
                    "theta_kappa": list(detail_payload.get("theta_kappa") or []),
                    "coverage_source": config.get(
                        "coverage_source",
                        "endpoint_theta_slope_obstacle_los/v1"
                        if config.get("slope_obstacle_aware_theta_reward_enabled")
                        else (
                            "theta_aware_sensor_footprint/v1"
                            if config.get("theta_aware_candidate_viewpoints_enabled")
                            else config.get("coverage_metric_mode")
                        ),
                    ),
                    "path_cost_source": selected_candidate.get("path_cost_source")
                    or config.get("path_cost_source")
                    or (
                        "hybrid_astar_pose_path/v1"
                        if config.get("hybrid_astar_pose_path_cost_enabled")
                        else selected_candidate.get("path_cost_source")
                    ),
                    "hybrid_astar_path_cost": selected_candidate.get("hybrid_astar_path_cost"),
                    "hybrid_astar_pose_path_hash": selected_candidate.get("hybrid_astar_pose_path_hash"),
                    "hybrid_astar_reachable": selected_candidate.get("hybrid_astar_reachable"),
                    "hybrid_astar_failure_reason": selected_candidate.get("hybrid_astar_failure_reason"),
                    "hybrid_astar_trajectory_kind": selected_candidate.get("hybrid_astar_trajectory_kind"),
                    "legacy_grid_astar_path_cost": selected_candidate.get("legacy_grid_astar_path_cost"),
                    "hybrid_vs_grid_path_cost_delta": selected_candidate.get("hybrid_vs_grid_path_cost_delta"),
                    "hybrid_astar_current_pose": selected_candidate.get("hybrid_astar_current_pose"),
                    "hybrid_astar_current_pose_provenance": selected_candidate.get("hybrid_astar_current_pose_provenance"),
                    "point_grid_path_cost_fallback_used": bool(selected_candidate.get("point_grid_path_cost_fallback_used", False)),
                    "default_astar_replaced": bool(selected_candidate.get("default_astar_replaced", False)),
                    "hybrid_astar_ackermann_feasible_claimed": bool(
                        selected_candidate.get("hybrid_astar_ackermann_feasible_claimed", False)
                    ),
                    "synthetic_terrain_model_id": selected_candidate.get("synthetic_terrain_model_id")
                    or config.get("synthetic_terrain_model_id"),
                    "synthetic_terrain_hash": selected_candidate.get("synthetic_terrain_hash")
                    or config.get("synthetic_terrain_hash"),
                    "synthetic_source_kind": selected_candidate.get("synthetic_source_kind")
                    or config.get("synthetic_source_kind"),
                    "synthetic_los_blocker_cells_used": bool(
                        selected_candidate.get(
                            "synthetic_los_blocker_cells_used",
                            config.get("synthetic_los_blocker_cells_used", bool(config.get("synthetic_terrain_hash"))),
                        )
                    ),
                    "synthetic_hard_obstacle_cells_used": bool(
                        selected_candidate.get(
                            "synthetic_hard_obstacle_cells_used",
                            config.get("synthetic_hard_obstacle_cells_used", bool(config.get("synthetic_terrain_hash"))),
                        )
                    ),
                    "physical_obstacle_cells_written": bool(
                        selected_candidate.get("physical_obstacle_cells_written", config.get("physical_obstacle_cells_written", False))
                    ),
                    "max_traversable_slope_deg": config.get("max_traversable_slope_deg"),
                    "obstacle_occlusion_enabled": selected_obstacle_audit.get("obstacle_occlusion_enabled"),
                    "obstacle_source_kind": selected_obstacle_audit.get("obstacle_source_kind"),
                    "obstacle_aware_theta_coverage_hash": selected_obstacle_audit.get("obstacle_aware_theta_coverage_hash"),
                    "slope_obstacle_source_hash": selected_candidate.get("slope_obstacle_source_hash")
                    or selected_candidate.get("obstacle_source_hash")
                    or selected_obstacle_audit.get("obstacle_source_hash"),
                    "platform_contract_hash": selected_candidate.get("platform_contract_hash")
                    or config.get("platform_contract_hash"),
                    "selected_rank": detail_payload.get("selected_rank"),
                    "selected_probability": detail_payload.get("selected_probability"),
                    "action_probs": list(detail_payload.get("action_probs") or []),
                    "logits": list(detail_payload.get("logits") or []),
                    "masked_logits": list(detail_payload.get("masked_logits") or []),
                    "xunce_batch_feature_semantic_map": synthetic_feature_metadata.get(
                        "xunce_batch_feature_semantic_map"
                    ),
                    "synthetic_credit_feature_rows": synthetic_feature_metadata.get("synthetic_credit_feature_rows"),
                    "synthetic_los_blocker_candidate_counts": synthetic_feature_metadata.get(
                        "synthetic_los_blocker_candidate_counts"
                    ),
                    "synthetic_hard_obstacle_candidate_counts": synthetic_feature_metadata.get(
                        "synthetic_hard_obstacle_candidate_counts"
                    ),
                    "value": detail_payload.get("value"),
                    "finite_outputs": detail_payload.get("finite_outputs"),
                    "latency_ms": detail_payload.get("latency_ms"),
                    "policy_inference_kind": policy_inference_kind,
                    "oracle_rollout_executed": oracle_rollout_executed,
                    "true_model_inference_executed": true_model_inference_executed,
                    "dynamic_candidate_validation_missing": dynamic_candidate_validation_missing,
                    "model_inference_failure_count": int(model_inference_failure),
                    "model_inference_mask_violation_count": int(mask_violation),
                    "detail": detail_payload,
                    "reason_codes": unique_sorted(step_reasons),
                }
            )
        reason_codes.extend(step_reasons)
        if not executed:
            break

    executed_step_count = sum(1 for row in steps if row["executed"])
    coverage_curve_auc = _coverage_curve_auc(coverage_rates, config["rollout_steps"])
    coverage_curve_auc_capped = _coverage_curve_auc_nullable(coverage_rate_capped_values, config["rollout_steps"])
    cumulative_delta = coverage_rates[-1] - coverage_rates[0]
    coverage_rate_raw = coverage_rate_raw_values[-1]
    coverage_rate_capped = coverage_rate_capped_values[-1]
    initial_coverage_rate_capped = coverage_rate_capped_values[0]
    coverage_rate_delta_capped = (
        None
        if coverage_rate_capped is None or initial_coverage_rate_capped is None
        else float(coverage_rate_capped) - float(initial_coverage_rate_capped)
    )
    total_seen_cells = new_cell_total + revisited_cell_total
    coverage_per_100m = _safe_ratio_v2(new_cell_total * 100.0, path_cost_total, "coverage_per_100m")
    risk_per_100m = _safe_ratio_v2(risk_total * 100.0, path_cost_total, "risk_per_100m")
    if valuable_area_covered > TOLERANCE:
        roi_weighted_coverage_total = valuable_area_covered
        roi_weighted_coverage_source = "candidate_value_weighted_new_cells/v1"
    else:
        roi_weighted_coverage_total = float(new_cell_total)
        roi_weighted_coverage_source = "unweighted_new_cell_count_fallback_due_missing_candidate_value/v1"
    return {
        "schema_version": "xunce-exploration-coverage-episode/v1",
        "scenario_id": scenario_id,
        **scenario_diversity_metadata,
        "roi_group": roi_group,
        "split": split,
        "policy": policy_name,
        "rollout_steps": config["rollout_steps"],
        "executed_step_count": executed_step_count,
        "episode_termination_reason": episode_termination_reason,
        "candidate_generation_exhausted": candidate_generation_exhausted_count > 0,
        "candidate_generation_exhausted_step": candidate_generation_exhausted_step,
        "candidate_generation_exhausted_count": candidate_generation_exhausted_count,
        "model_inference_failure_count": model_inference_failure_count,
        "initial_coverage_rate": coverage_rates[0],
        "final_coverage_rate": coverage_rates[-1],
        "coverage_rate_delta": cumulative_delta,
        "cumulative_coverage_rate_delta": cumulative_delta,
        "coverage_return": coverage_return,
        "coverage_curve_auc": coverage_curve_auc,
        "raw_covered_cell_count": len(covered_cells),
        "raw_new_covered_cell_count": raw_new_cell_total,
        "coverage_rate_raw": coverage_rate_raw,
        "coverage_rate_capped": coverage_rate_capped,
        "final_coverage_rate_capped": coverage_rate_capped,
        "coverage_rate_delta_capped": coverage_rate_delta_capped,
        "coverage_curve_auc_capped": coverage_curve_auc_capped,
        "coverage_saturation_exceeded": bool(coverage_rate_raw is not None and coverage_rate_raw > 1.0 + TOLERANCE),
        "coverage_saturation_excess": _coverage_saturation_excess(coverage_rate_raw),
        "coverage_denominator_cells": denominator_context["coverage_denominator_cells"],
        "coverage_denominator_mode": denominator_context["coverage_denominator_mode"],
        "coverage_denominator_source": denominator_context["coverage_denominator_source"],
        "coverage_denominator_reason_codes": denominator_context["coverage_denominator_reason_codes"],
        **_coverage_denominator_episode_fields(covered_cells, denominator_context),
        "new_covered_cell_count": new_cell_total,
        "total_new_cell_count": new_cell_total,
        "revisited_cell_count": revisited_cell_total,
        "revisit_rate": _safe_ratio(revisited_cell_total, total_seen_cells) or 0.0,
        "coverage_overlap_ratio": _safe_ratio(revisited_cell_total, total_seen_cells) or 0.0,
        "min_roi_group_coverage_rate": coverage_rates[-1],
        "valuable_area_covered": valuable_area_covered,
        "roi_weighted_coverage_total": roi_weighted_coverage_total,
        "roi_weighted_coverage_source": roi_weighted_coverage_source,
        "path_cost": path_cost_total,
        "path_cost_total_m": path_cost_total,
        "risk": risk_total,
        "risk_total": risk_total,
        "risk_source": "candidate_risk_scalar_rollout_sum/v1",
        "risk_cost_weighted_total": risk_cost_weighted_total,
        "risk_cost_weighted_source": "sum_path_cost_times_candidate_risk_scalar/v1",
        "soft_risk_exposure_total": soft_risk_exposure_total,
        "soft_risk_exposure_source": "sum_selected_path_risk_exposure_or_path_cost_times_risk_proxy/v1",
        "path_risk_peak_max": max(path_risk_peaks) if path_risk_peaks else None,
        "high_risk_distance_total_m": high_risk_distance_total_m,
        "hard_risk_violation_count": hard_risk_violation_count,
        "risk_boundary_violation_steps": risk_boundary_violation_steps,
        "energy_cost": energy_total,
        "coverage_gain_per_path_cost": _safe_ratio(coverage_return, path_cost_total),
        "coverage_gain_per_meter": _safe_ratio(coverage_return, path_cost_total),
        "coverage_gain_per_risk": _safe_ratio(coverage_return, risk_total),
        "coverage_gain_per_energy": _safe_ratio(coverage_return, energy_total),
        "coverage_per_100m": coverage_per_100m["value"],
        "risk_per_100m": risk_per_100m["value"],
        "undefined_metric_reason_codes": unique_sorted(coverage_per_100m["reason_codes"] + risk_per_100m["reason_codes"]),
        "path_budget_used_ratio": _safe_ratio(path_cost_total, config["path_budget_m"]) or 0.0,
        "latency_per_coverage_gain": _safe_ratio(sum(latencies), coverage_return),
        "selected_probability_median": statistics.median(selected_probabilities) if selected_probabilities else 0.0,
        "selected_rank_median": statistics.median(selected_ranks) if selected_ranks else 0.0,
        "action_entropy_median": statistics.median(entropies) if entropies else 0.0,
        "median_inference_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "mask_violation_count": mask_violation_count,
        "unreachable_selected_count": unreachable_selected_count,
        "path_planning_failure_count": path_planning_failure_count,
        "open_grid_fallback_count": open_grid_fallback_count,
        "reason_codes": unique_sorted(reason_codes),
        "action_indices": action_indices,
        "steps": steps,
        "inference_rows": inference_rows,
        "dynamic_proposals": dynamic_proposals,
        "dynamic_validation_rows": dynamic_validation_rows,
        "paired_decision_rows": paired_decision_rows,
        "candidate_metric_rows": candidate_metric_rows,
        "on_policy_teacher_label_rows": on_policy_teacher_label_rows,
    }


def _comparison_pairs(episodes: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for episode in episodes:
        grouped[str(episode["scenario_id"])][str(episode["policy"])] = episode
    rows: list[dict[str, Any]] = []
    profiles = config["comparison_utility_profiles"]
    for scenario_id in sorted(grouped):
        policies = grouped[scenario_id]
        xunce = policies.get("xunce")
        incumbent = policies.get("incumbent")
        if not xunce or not incumbent:
            continue
        greedy_oracle = policies.get("greedy_coverage_oracle")
        cost_aware_oracle = policies.get("cost_aware_coverage_oracle")
        coverage_delta = _episode_number(xunce, "total_new_cell_count") - _episode_number(incumbent, "total_new_cell_count")
        roi_delta = _episode_number(xunce, "roi_weighted_coverage_total") - _episode_number(incumbent, "roi_weighted_coverage_total")
        path_cost_delta = _episode_number(xunce, "path_cost_total_m") - _episode_number(incumbent, "path_cost_total_m")
        risk_delta = _episode_number(xunce, "risk_total") - _episode_number(incumbent, "risk_total")
        risk_cost_delta = _episode_number(xunce, "risk_cost_weighted_total") - _episode_number(incumbent, "risk_cost_weighted_total")
        soft_risk_exposure_delta = _episode_number(xunce, "soft_risk_exposure_total") - _episode_number(incumbent, "soft_risk_exposure_total")
        hard_risk_violation_delta = _episode_number(xunce, "hard_risk_violation_count") - _episode_number(incumbent, "hard_risk_violation_count")
        path_risk_peak_delta = _nullable_delta(xunce.get("path_risk_peak_max"), incumbent.get("path_risk_peak_max"))
        xunce_coverage_per_100m = _safe_ratio_v2(_episode_number(xunce, "total_new_cell_count") * 100.0, _episode_number(xunce, "path_cost_total_m"), "xunce_coverage_per_100m")
        incumbent_coverage_per_100m = _safe_ratio_v2(_episode_number(incumbent, "total_new_cell_count") * 100.0, _episode_number(incumbent, "path_cost_total_m"), "incumbent_coverage_per_100m")
        xunce_risk_per_100m = _safe_ratio_v2(_episode_number(xunce, "risk_total") * 100.0, _episode_number(xunce, "path_cost_total_m"), "xunce_risk_per_100m")
        incumbent_risk_per_100m = _safe_ratio_v2(_episode_number(incumbent, "risk_total") * 100.0, _episode_number(incumbent, "path_cost_total_m"), "incumbent_risk_per_100m")
        coverage_per_100m_delta = _nullable_delta(xunce_coverage_per_100m["value"], incumbent_coverage_per_100m["value"])
        incremental_cost = _incremental_ratio(path_cost_delta, coverage_delta, "incremental_cost_per_extra_cell")
        greedy_xunce_regret = _oracle_coverage_regret(greedy_oracle, xunce)
        greedy_incumbent_regret = _oracle_coverage_regret(greedy_oracle, incumbent)
        cost_xunce_regret = _oracle_utility_regret(cost_aware_oracle, xunce, profiles["cost_aware"])
        cost_incumbent_regret = _oracle_utility_regret(cost_aware_oracle, incumbent, profiles["cost_aware"])
        profile_outcomes = {
            name: _utility_pair_outcome(xunce, incumbent, profile)
            for name, profile in profiles.items()
        }
        reason_codes = unique_sorted(
            list(xunce.get("undefined_metric_reason_codes", []))
            + list(incumbent.get("undefined_metric_reason_codes", []))
            + xunce_coverage_per_100m["reason_codes"]
            + incumbent_coverage_per_100m["reason_codes"]
            + xunce_risk_per_100m["reason_codes"]
            + incumbent_risk_per_100m["reason_codes"]
            + incremental_cost["reason_codes"]
            + _missing_oracle_reasons(greedy_oracle, cost_aware_oracle)
        )
        guard_thresholds = config["canonical_guard_thresholds"]
        path_cost_budget_exceeded = path_cost_delta > float(guard_thresholds["max_path_cost_delta_m"]) + TOLERANCE
        risk_budget_exceeded = (
            bool(guard_thresholds.get("risk_delta_hard_gate_enabled", True))
            and guard_thresholds.get("max_risk_delta") is not None
            and risk_delta > float(guard_thresholds["max_risk_delta"]) + TOLERANCE
        )
        risk_cost_weighted_budget_exceeded = (
            risk_cost_delta > float(
                guard_thresholds.get("max_soft_risk_exposure_delta")
                if guard_thresholds.get("candidate_level_risk_delta_guard_is_diagnostic_only")
                else guard_thresholds["max_risk_cost_weighted_delta"]
            )
            + TOLERANCE
        )
        coverage_efficiency_regression = (
            coverage_per_100m_delta is not None
            and coverage_per_100m_delta < float(guard_thresholds["min_coverage_per_100m_delta"]) - TOLERANCE
        )
        row = {
            "schema_version": "xunce-exploration-coverage-comparison-pair/v1",
            "scenario_id": scenario_id,
            "roi_group": xunce.get("roi_group"),
            "split": xunce.get("split"),
            "rollout_steps": xunce.get("rollout_steps"),
            "xunce_total_new_cell_count": _episode_number(xunce, "total_new_cell_count"),
            "incumbent_total_new_cell_count": _episode_number(incumbent, "total_new_cell_count"),
            "coverage_delta_cells": coverage_delta,
            "xunce_raw_covered_cell_count": _episode_number(xunce, "raw_covered_cell_count"),
            "incumbent_raw_covered_cell_count": _episode_number(incumbent, "raw_covered_cell_count"),
            "xunce_final_coverage_rate_raw": xunce.get("coverage_rate_raw"),
            "incumbent_final_coverage_rate_raw": incumbent.get("coverage_rate_raw"),
            "coverage_rate_delta_raw": _nullable_delta(xunce.get("coverage_rate_raw"), incumbent.get("coverage_rate_raw")),
            "xunce_final_coverage_rate_capped": xunce.get("final_coverage_rate_capped"),
            "incumbent_final_coverage_rate_capped": incumbent.get("final_coverage_rate_capped"),
            "coverage_rate_delta_capped": _nullable_delta(xunce.get("final_coverage_rate_capped"), incumbent.get("final_coverage_rate_capped")),
            "xunce_coverage_saturation_exceeded": bool(xunce.get("coverage_saturation_exceeded")),
            "incumbent_coverage_saturation_exceeded": bool(incumbent.get("coverage_saturation_exceeded")),
            "xunce_coverage_saturation_excess": _episode_number(xunce, "coverage_saturation_excess"),
            "incumbent_coverage_saturation_excess": _episode_number(incumbent, "coverage_saturation_excess"),
            "xunce_roi_weighted_coverage_total": _episode_number(xunce, "roi_weighted_coverage_total"),
            "incumbent_roi_weighted_coverage_total": _episode_number(incumbent, "roi_weighted_coverage_total"),
            "roi_weighted_coverage_delta": roi_delta,
            "xunce_path_cost_total_m": _episode_number(xunce, "path_cost_total_m"),
            "incumbent_path_cost_total_m": _episode_number(incumbent, "path_cost_total_m"),
            "path_cost_delta_m": path_cost_delta,
            "xunce_risk_total": _episode_number(xunce, "risk_total"),
            "incumbent_risk_total": _episode_number(incumbent, "risk_total"),
            "risk_delta": risk_delta,
            "xunce_risk_cost_weighted_total": _episode_number(xunce, "risk_cost_weighted_total"),
            "incumbent_risk_cost_weighted_total": _episode_number(incumbent, "risk_cost_weighted_total"),
            "risk_cost_weighted_delta": risk_cost_delta,
            "xunce_soft_risk_exposure_total": _episode_number(xunce, "soft_risk_exposure_total"),
            "incumbent_soft_risk_exposure_total": _episode_number(incumbent, "soft_risk_exposure_total"),
            "soft_risk_exposure_delta": soft_risk_exposure_delta,
            "xunce_hard_risk_violation_count": _episode_number(xunce, "hard_risk_violation_count"),
            "incumbent_hard_risk_violation_count": _episode_number(incumbent, "hard_risk_violation_count"),
            "hard_risk_violation_delta": hard_risk_violation_delta,
            "hard_risk_violation_count": _episode_number(xunce, "hard_risk_violation_count"),
            "xunce_path_risk_peak_max": xunce.get("path_risk_peak_max"),
            "incumbent_path_risk_peak_max": incumbent.get("path_risk_peak_max"),
            "path_risk_peak_delta": path_risk_peak_delta,
            "xunce_coverage_per_100m": xunce_coverage_per_100m["value"],
            "incumbent_coverage_per_100m": incumbent_coverage_per_100m["value"],
            "coverage_per_100m_delta": coverage_per_100m_delta,
            "xunce_risk_per_100m": xunce_risk_per_100m["value"],
            "incumbent_risk_per_100m": incumbent_risk_per_100m["value"],
            "risk_per_100m_delta": _nullable_delta(xunce_risk_per_100m["value"], incumbent_risk_per_100m["value"]),
            "incremental_cost_per_extra_cell": incremental_cost["value"],
            "greedy_oracle_coverage_regret_xunce": greedy_xunce_regret,
            "greedy_oracle_coverage_regret_incumbent": greedy_incumbent_regret,
            "cost_aware_oracle_utility_regret_xunce": cost_xunce_regret,
            "cost_aware_oracle_utility_regret_incumbent": cost_incumbent_regret,
            "utility_profile_outcomes": profile_outcomes,
            "canonical_guard_profile_id": config["profile_id"],
            "canonical_guard_profile_version": config["profile_version"],
            "canonical_guard_profile_hash": config["profile_hash"],
            "canonical_guard_thresholds": guard_thresholds,
            "path_cost_budget_exceeded": path_cost_budget_exceeded,
            "risk_budget_exceeded": risk_budget_exceeded,
            "risk_cost_weighted_budget_exceeded": risk_cost_weighted_budget_exceeded,
            "soft_risk_exposure_budget_exceeded": risk_cost_weighted_budget_exceeded
            if guard_thresholds.get("candidate_level_risk_delta_guard_is_diagnostic_only")
            else False,
            "coverage_efficiency_regression": coverage_efficiency_regression,
            "coverage_gain_per_path_cost_delta_audit_only": (
                guard_thresholds["coverage_gain_per_path_cost_delta_mode"] == "audit_only"
            ),
            "pairwise_outcome": _coverage_pairwise_outcome(coverage_delta),
            "undefined_metric_reason_codes": reason_codes,
        }
        rows.append(row)
    return rows


def _comparison_aggregate(pairs: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    win_count = sum(1 for row in pairs if row["pairwise_outcome"] == "xunce_coverage_win")
    tie_count = sum(1 for row in pairs if row["pairwise_outcome"] == "coverage_tie")
    loss_count = sum(1 for row in pairs if row["pairwise_outcome"] == "xunce_coverage_loss")
    profile_summary: dict[str, dict[str, int]] = {}
    for profile_name in config["comparison_utility_profiles"]:
        outcomes = [row.get("utility_profile_outcomes", {}).get(profile_name) for row in pairs]
        profile_summary[profile_name] = {
            "xunce_win_count": sum(1 for outcome in outcomes if outcome == "xunce_win"),
            "incumbent_win_count": sum(1 for outcome in outcomes if outcome == "incumbent_win"),
            "tie_count": sum(1 for outcome in outcomes if outcome == "tie"),
        }
    final_raw_rates = [
        value
        for row in pairs
        for value in (row.get("xunce_final_coverage_rate_raw"), row.get("incumbent_final_coverage_rate_raw"))
        if _finite_or_none(value) is not None
    ]
    saturation_excesses = [
        value
        for row in pairs
        for value in (row.get("xunce_coverage_saturation_excess"), row.get("incumbent_coverage_saturation_excess"))
        if _finite_or_none(value) is not None
    ]
    return {
        "schema_version": "xunce-exploration-coverage-comparison-aggregate/v1",
        "scenario_count": len(pairs),
        "profile_id": config["profile_id"],
        "profile_version": config["profile_version"],
        "profile_hash": config["profile_hash"],
        "xunce_coverage_win_count": win_count,
        "xunce_coverage_tie_count": tie_count,
        "xunce_coverage_loss_count": loss_count,
        "xunce_coverage_win_rate": _safe_ratio(win_count, len(pairs)) or 0.0,
        "coverage_rate_saturation_episode_count": sum(
            int(bool(row.get("xunce_coverage_saturation_exceeded"))) + int(bool(row.get("incumbent_coverage_saturation_exceeded")))
            for row in pairs
        ),
        "max_final_coverage_rate_raw": max((float(value) for value in final_raw_rates), default=0.0),
        "max_coverage_rate_saturation_excess": max((float(value) for value in saturation_excesses), default=0.0),
        **_distribution_fields("coverage_delta_cells", [row.get("coverage_delta_cells") for row in pairs]),
        **_distribution_fields("path_cost_delta_m", [row.get("path_cost_delta_m") for row in pairs]),
        **_distribution_fields("risk_delta", [row.get("risk_delta") for row in pairs]),
        **_distribution_fields("risk_cost_weighted_delta", [row.get("risk_cost_weighted_delta") for row in pairs]),
        **_distribution_fields("soft_risk_exposure_delta", [row.get("soft_risk_exposure_delta") for row in pairs]),
        **_distribution_fields("hard_risk_violation_delta", [row.get("hard_risk_violation_delta") for row in pairs]),
        "hard_risk_violation_count": sum(int(row.get("hard_risk_violation_count", 0) or 0) for row in pairs),
        **_distribution_fields("coverage_per_100m_delta", [row.get("coverage_per_100m_delta") for row in pairs]),
        "greedy_oracle_coverage_regret_delta_mean": _mean(
            [
                _nullable_delta(row.get("greedy_oracle_coverage_regret_xunce"), row.get("greedy_oracle_coverage_regret_incumbent"))
                for row in pairs
            ]
        ),
        "cost_aware_oracle_utility_regret_delta_mean": _mean(
            [
                _nullable_delta(row.get("cost_aware_oracle_utility_regret_xunce"), row.get("cost_aware_oracle_utility_regret_incumbent"))
                for row in pairs
            ]
        ),
        "utility_profile_summary": profile_summary,
        "comparison_utility_profiles": config["comparison_utility_profiles"],
    }


def _coverage_comparison_audit(
    episodes: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    comparison_pairs: list[dict[str, Any]],
    comparison_aggregate: dict[str, Any],
    *,
    dynamic_proposals: list[dict[str, Any]],
    dynamic_validation_rows: list[dict[str, Any]],
    paired_decision_rows: list[dict[str, Any]],
    candidate_metric_rows: list[dict[str, Any]],
    on_policy_teacher_label_rows: list[dict[str, Any]],
    source: dict[str, Any],
    config: dict[str, Any],
    repo_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for episode in episodes:
        pairs[str(episode["scenario_id"])][str(episode["policy"])] = episode
    xunce_better = 0
    xunce_worse = 0
    xunce_tie = 0
    efficiency_regression = 0
    safety_regression = 0
    risk_regression = 0
    path_cost_regression = 0
    coverage_return_deltas: list[float] = []
    coverage_auc_deltas: list[float] = []
    final_coverage_deltas: list[float] = []
    new_cell_deltas: list[float] = []
    min_roi_deltas: list[float] = []
    gain_per_cost_deltas: list[float] = []
    gain_per_risk_deltas: list[float] = []
    pair_by_scenario = {str(row["scenario_id"]): row for row in comparison_pairs}
    for policies in pairs.values():
        xunce = policies.get("xunce")
        incumbent = policies.get("incumbent")
        if not xunce or not incumbent:
            continue
        pair_row = pair_by_scenario.get(str(xunce["scenario_id"]), {})
        coverage_delta = float(xunce["coverage_return"]) - float(incumbent["coverage_return"])
        coverage_return_deltas.append(coverage_delta)
        coverage_auc_deltas.append(float(xunce["coverage_curve_auc"]) - float(incumbent["coverage_curve_auc"]))
        final_coverage_deltas.append(float(xunce["final_coverage_rate"]) - float(incumbent["final_coverage_rate"]))
        new_cell_deltas.append(float(xunce["new_covered_cell_count"]) - float(incumbent["new_covered_cell_count"]))
        min_roi_deltas.append(float(xunce["min_roi_group_coverage_rate"]) - float(incumbent["min_roi_group_coverage_rate"]))
        cost_delta = _none_to_zero(xunce.get("coverage_gain_per_path_cost")) - _none_to_zero(incumbent.get("coverage_gain_per_path_cost"))
        risk_delta = _none_to_zero(xunce.get("coverage_gain_per_risk")) - _none_to_zero(incumbent.get("coverage_gain_per_risk"))
        gain_per_cost_deltas.append(cost_delta)
        gain_per_risk_deltas.append(risk_delta)
        if coverage_delta > TOLERANCE:
            xunce_better += 1
        elif coverage_delta < -TOLERANCE:
            xunce_worse += 1
        else:
            xunce_tie += 1
        if pair_row.get("risk_budget_exceeded") is True or pair_row.get("risk_cost_weighted_budget_exceeded") is True:
            risk_regression += 1
        if pair_row.get("path_cost_budget_exceeded") is True:
            path_cost_regression += 1
        if pair_row.get("path_cost_budget_exceeded") is True or pair_row.get("coverage_efficiency_regression") is True:
            efficiency_regression += 1
        if (
            int(xunce["mask_violation_count"]) > int(incumbent["mask_violation_count"])
            or int(xunce["unreachable_selected_count"]) > int(incumbent["unreachable_selected_count"])
            or int(xunce["path_planning_failure_count"]) > int(incumbent["path_planning_failure_count"])
            or int(xunce["open_grid_fallback_count"]) > int(incumbent["open_grid_fallback_count"])
            or pair_row.get("risk_budget_exceeded") is True
            or pair_row.get("risk_cost_weighted_budget_exceeded") is True
        ):
            safety_regression += 1
    if config["candidate_refresh_mode"] == "dynamic_frontier_nbv_in_process":
        disagreement_count, useful_disagreement_count = _paired_disagreement_counts(paired_decision_rows)
    else:
        disagreement_count, useful_disagreement_count = _disagreement_counts(steps)
    xunce_episodes = [row for row in episodes if row["policy"] == "xunce"]
    incumbent_episodes = [row for row in episodes if row["policy"] == "incumbent"]
    oracle_episodes = [row for row in episodes if row["policy"] == "greedy_coverage_oracle"]
    oracle_by_scenario = {str(row["scenario_id"]): row for row in oracle_episodes}
    xunce_oracle_regrets = []
    incumbent_oracle_regrets = []
    oracle_vs_incumbent_deltas = []
    for policies in pairs.values():
        xunce = policies.get("xunce")
        incumbent = policies.get("incumbent")
        if not xunce or not incumbent:
            continue
        oracle = oracle_by_scenario.get(str(xunce["scenario_id"]))
        if not oracle:
            continue
        oracle_return = float(oracle["coverage_return"])
        xunce_oracle_regrets.append(max(0.0, oracle_return - float(xunce["coverage_return"])))
        incumbent_oracle_regrets.append(max(0.0, oracle_return - float(incumbent["coverage_return"])))
        oracle_vs_incumbent_deltas.append(oracle_return - float(incumbent["coverage_return"]))
    evaluation_task_discriminative = bool(
        oracle_vs_incumbent_deltas
        and _mean(oracle_vs_incumbent_deltas) > TOLERANCE
        and useful_disagreement_count > 0
        and sum(1 for row in episodes if row["policy"] == "greedy_coverage_oracle") > 0
    )
    dynamic_audit = _dynamic_validation_audit(
        config,
        source,
        dynamic_proposals,
        dynamic_validation_rows,
        steps,
        repo_root=repo_root,
        output_root=output_root,
    )
    candidate_generation_exhausted_count = sum(int(row.get("candidate_generation_exhausted_count", 0) or 0) for row in episodes)
    model_inference_failure_count = sum(int(row.get("model_inference_failure_count", 0) or 0) for row in episodes)
    coverage_rate_saturation_episode_count = sum(1 for row in episodes if row.get("coverage_saturation_exceeded") is True)
    final_raw_rates = [row.get("coverage_rate_raw") for row in episodes if _finite_or_none(row.get("coverage_rate_raw")) is not None]
    saturation_excesses = [
        row.get("coverage_saturation_excess")
        for row in episodes
        if _finite_or_none(row.get("coverage_saturation_excess")) is not None
    ]
    return {
        "schema_version": "xunce-exploration-coverage-comparison-audit/v1",
        "dynamic_validation_audit": dynamic_audit,
        "scenario_count": comparison_aggregate["scenario_count"],
        "xunce_coverage_better_count": comparison_aggregate["xunce_coverage_win_count"],
        "xunce_coverage_worse_count": comparison_aggregate["xunce_coverage_loss_count"],
        "xunce_coverage_tie_count": comparison_aggregate["xunce_coverage_tie_count"],
        "xunce_efficiency_regression_count": efficiency_regression,
        "xunce_safety_regression_count": safety_regression,
        "risk_regression_count": risk_regression,
        "path_cost_regression_count": path_cost_regression,
        "xunce_oracle_regret": _mean(xunce_oracle_regrets),
        "incumbent_oracle_regret": _mean(incumbent_oracle_regrets),
        "oracle_vs_incumbent_coverage_delta": _mean(oracle_vs_incumbent_deltas),
        "evaluation_task_discriminative": evaluation_task_discriminative,
        "xunce_final_coverage_rate_delta_vs_incumbent": _mean(final_coverage_deltas),
        "xunce_coverage_return_delta_vs_incumbent": _mean(coverage_return_deltas),
        "xunce_coverage_curve_auc_delta_vs_incumbent": _mean(coverage_auc_deltas),
        "xunce_new_covered_cell_delta_vs_incumbent": comparison_aggregate["coverage_delta_cells_mean"],
        "xunce_min_roi_group_coverage_delta_vs_incumbent": _mean(min_roi_deltas),
        "coverage_gain_per_path_cost_delta_vs_incumbent": _mean(gain_per_cost_deltas),
        "coverage_gain_per_risk_delta_vs_incumbent": _mean(gain_per_risk_deltas),
        "xunce_path_cost_delta_vs_incumbent": comparison_aggregate["path_cost_delta_m_mean"],
        "xunce_risk_delta_vs_incumbent": comparison_aggregate["risk_delta_mean"],
        "risk_cost_weighted_delta_vs_incumbent": _mean([row.get("risk_cost_weighted_delta") for row in comparison_pairs]),
        "coverage_per_100m_delta_vs_incumbent": comparison_aggregate["coverage_per_100m_delta_mean"],
        "risk_per_100m_delta_vs_incumbent": _mean([row.get("risk_per_100m_delta") for row in comparison_pairs]),
        "comparison_aggregate": comparison_aggregate,
        "policy_disagreement_count": disagreement_count,
        "useful_disagreement_count": useful_disagreement_count,
        "selected_probability_median": _median([row["selected_probability_median"] for row in episodes]),
        "action_entropy_median": _median([row["action_entropy_median"] for row in episodes]),
        "selected_rank_median": _median([row["selected_rank_median"] for row in episodes]),
        "xunce_median_inference_latency_ms": _median([row["median_inference_latency_ms"] for row in xunce_episodes]),
        "incumbent_median_inference_latency_ms": _median([row["median_inference_latency_ms"] for row in incumbent_episodes]),
        "xunce_open_grid_fallback_count": sum(int(row["open_grid_fallback_count"]) for row in xunce_episodes),
        "open_grid_fallback_count": sum(int(row["open_grid_fallback_count"]) for row in episodes),
        "unreachable_selected_count": sum(int(row["unreachable_selected_count"]) for row in episodes),
        "path_planning_failure_count": sum(int(row["path_planning_failure_count"]) for row in episodes),
        "dynamic_candidate_validation_missing_count": sum(
            1
            for row in steps
            if row.get("dynamic_candidate_validation_missing") is True
        ),
        "dynamic_candidate_generation_executed": dynamic_audit["dynamic_candidate_generation_executed"],
        "dynamic_candidate_generation_source": dynamic_audit["dynamic_candidate_generation_source"],
        "dynamic_candidate_validation_mode": dynamic_audit["dynamic_candidate_validation_mode"],
        "dynamic_validation_work_root": dynamic_audit["dynamic_validation_work_root"],
        "dynamic_validation_work_root_path_length": dynamic_audit["dynamic_validation_work_root_path_length"],
        "dynamic_validation_max_path_length": dynamic_audit["dynamic_validation_max_path_length"],
        "dynamic_proposal_count": dynamic_audit["dynamic_proposal_count"],
        "dynamic_validation_attempt_count": dynamic_audit["dynamic_validation_attempt_count"],
        "dynamic_validation_success_count": dynamic_audit["dynamic_validation_success_count"],
        "dynamic_validation_failure_count": dynamic_audit["dynamic_validation_failure_count"],
        "dynamic_validation_cache_hit_count": dynamic_audit["dynamic_validation_cache_hit_count"],
        "dynamic_contract_sidecar_missing_count": dynamic_audit["dynamic_contract_sidecar_missing_count"],
        "dynamic_path_length_preflight_failure_count": dynamic_audit["dynamic_path_length_preflight_failure_count"],
        "in_process_batch_astar_validation_count": dynamic_audit["in_process_batch_astar_validation_count"],
        "path_planner_route_adapter_success_count": dynamic_audit["path_planner_route_adapter_success_count"],
        "path_planner_route_adapter_failure_count": dynamic_audit["path_planner_route_adapter_failure_count"],
        "path_planner_route_adapter_audit_sample_count": dynamic_audit["path_planner_route_adapter_audit_sample_count"],
        "path_planner_route_adapter_audit_failure_count": dynamic_audit["path_planner_route_adapter_audit_failure_count"],
        "adapter_batch_astar_mismatch_count": dynamic_audit["adapter_batch_astar_mismatch_count"],
        "adapter_audit_passed": dynamic_audit["adapter_audit_passed"],
        "sidecar_grid_astar_screening_count": dynamic_audit["sidecar_grid_astar_screening_count"],
        "sidecar_grid_astar_diagnostic_count": dynamic_audit["sidecar_grid_astar_diagnostic_count"],
        "adapter_error_type_counts": dynamic_audit["adapter_error_type_counts"],
        "adapter_error_message_samples": dynamic_audit["adapter_error_message_samples"],
        "planner_validation_backend_counts": dynamic_audit["planner_validation_backend_counts"],
        "validation_evidence_kind_counts": dynamic_audit["validation_evidence_kind_counts"],
        "dynamic_validation_full_adapter_evidence_passed": dynamic_audit["dynamic_validation_full_adapter_evidence_passed"],
        "dynamic_planner_validation_backend_counts": dynamic_audit["planner_validation_backend_counts"],
        "dynamic_sidecar_grid_astar_fallback_count": dynamic_audit["sidecar_grid_astar_fallback_count"],
        "coverage_frontier_candidate_count": dynamic_audit["coverage_frontier_candidate_count"],
        "undercovered_component_candidate_count": dynamic_audit["undercovered_component_candidate_count"],
        "candidate_generation_algorithm_source_counts": dynamic_audit["candidate_generation_algorithm_source_counts"],
        "low_cost_bridge_candidate_count": dynamic_audit["low_cost_bridge_candidate_count"],
        "conservative_local_candidate_count": dynamic_audit["conservative_local_candidate_count"],
        "validated_pareto_frontier_count": dynamic_audit["validated_pareto_frontier_count"],
        "validated_low_cost_candidate_count": dynamic_audit["validated_low_cost_candidate_count"],
        "validated_efficiency_candidate_count": dynamic_audit["validated_efficiency_candidate_count"],
        "prevalidation_proposal_drop_count": dynamic_audit["prevalidation_proposal_drop_count"],
        "postvalidation_candidate_drop_count": dynamic_audit["postvalidation_candidate_drop_count"],
        "risk_source_counts": dynamic_audit["risk_source_counts"],
        "formal_risk_source_counts": dynamic_audit["formal_risk_source_counts"],
        "selected_risk_source_counts": dynamic_audit["selected_risk_source_counts"],
        "route_derived_risk_count": dynamic_audit["route_derived_risk_count"],
        "roi_weight_source_counts": dynamic_audit["roi_weight_source_counts"],
        "candidate_selection_mode": dynamic_audit["candidate_selection_mode"],
        "dynamic_validation_source_root": dynamic_audit["dynamic_validation_source_root"],
        "dynamic_candidate_generation_missing_count": dynamic_audit["dynamic_candidate_generation_missing_count"],
        "candidate_generation_exhausted_count": candidate_generation_exhausted_count,
        "model_inference_failure_count": model_inference_failure_count,
        "coverage_rate_saturation_episode_count": coverage_rate_saturation_episode_count,
        "max_final_coverage_rate_raw": max((float(value) for value in final_raw_rates), default=0.0),
        "max_coverage_rate_saturation_excess": max((float(value) for value in saturation_excesses), default=0.0),
        "state_conditioned_candidate_generation": dynamic_audit["state_conditioned_candidate_generation"],
        "candidate_set_hash_mismatch_count": _candidate_set_hash_mismatch_count(steps),
        "paired_decision_audit_row_count": len(paired_decision_rows),
        "candidate_metric_audit_row_count": len(candidate_metric_rows) if config["emit_candidate_metric_audit"] else 0,
        "on_policy_oracle_teacher_label_row_count": len(on_policy_teacher_label_rows)
        if config["emit_on_policy_oracle_teacher_labels"]
        else 0,
        "candidate_generation_effect_scope": (
            "dynamic_generator_plus_policy_closed_loop"
            if config["candidate_refresh_mode"] == "dynamic_frontier_nbv_in_process"
            else "static_candidate_set"
        ),
        "model_selection_evidence_scope": (
            "same_state_same_candidate_set_paired_decision_audit"
            if config["candidate_refresh_mode"] == "dynamic_frontier_nbv_in_process"
            else "static_candidate_set"
        ),
        "closed_loop_dynamic_rollout_summary": {
            "enabled": config["candidate_refresh_mode"] == "dynamic_frontier_nbv_in_process",
            "scenario_count": comparison_aggregate["scenario_count"],
            "xunce_coverage_win_count": comparison_aggregate["xunce_coverage_win_count"],
            "xunce_coverage_loss_count": comparison_aggregate["xunce_coverage_loss_count"],
            "coverage_delta_cells_mean": comparison_aggregate["coverage_delta_cells_mean"],
            "path_cost_delta_m_mean": comparison_aggregate["path_cost_delta_m_mean"],
            "risk_delta_mean": comparison_aggregate["risk_delta_mean"],
        },
        "same_candidate_set_policy_selection_summary": {
            "paired_decision_audit_row_count": len(paired_decision_rows),
            "policy_disagreement_count": disagreement_count,
            "useful_disagreement_count": useful_disagreement_count,
        },
        "closed_loop_dynamic_rollout_advantage_established": False,
        "same_candidate_set_policy_selection_advantage_established": bool(useful_disagreement_count > 0),
    }


def _dynamic_validation_audit(
    config: dict[str, Any],
    source: dict[str, Any],
    dynamic_proposals: list[dict[str, Any]],
    dynamic_validation_rows: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    *,
    repo_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    generation_executed = config["candidate_refresh_mode"] == "dynamic_frontier_nbv_in_process"
    validation_attempts = len(dynamic_validation_rows)
    validation_success = sum(1 for row in dynamic_validation_rows if row.get("proposal_validated_by_path_feedback") is True and row.get("proposal_only") is False)
    contract_sidecar_missing = sum(
        1
        for row in dynamic_validation_rows
        if row.get("dynamic_contract_missing") is True
        or row.get("dynamic_sidecar_missing") is True
        or row.get("path_feedback_validation_source") == "dynamic_contract_or_sidecar_missing"
    )
    generation_missing = sum(
        1
        for row in steps
        if "dynamic_candidate_generation_missing" in set(row.get("reason_codes", []))
    )
    candidate_generation_exhausted = sum(
        1
        for row in steps
        if row.get("candidate_generation_exhausted") is True
        or row.get("terminal_reason") == "candidate_generation_exhausted"
        or "candidate_generation_exhausted" in set(row.get("reason_codes", []))
    )
    backend_counts = _count_by_field(dynamic_validation_rows, "planner_validation_backend")
    evidence_kind_counts = _count_by_field(dynamic_validation_rows, "validation_evidence_kind")
    batch_astar_count = sum(
        1
        for row in dynamic_validation_rows
        if row.get("planner_validation_backend") == "in_process_path_planner_astar_batch"
        and row.get("proposal_validated_by_path_feedback") is True
        and row.get("proposal_only") is False
    )
    path_length_preflight_failures = sum(
        1
        for row in dynamic_validation_rows
        if row.get("path_feedback_validation_source") == "path_length_preflight_failed"
        or row.get("failure_reason") == "path_length_preflight_failed"
        or row.get("path_length_gate_passed") is False
    )
    route_success = sum(
        1
        for row in dynamic_validation_rows
        if row.get("planner_validation_backend") == "path_planner_route_adapter"
        and row.get("proposal_validated_by_path_feedback") is True
        and row.get("proposal_only") is False
    )
    route_failure = sum(
        1
        for row in dynamic_validation_rows
        if row.get("planner_validation_backend") == "path_planner_route_adapter"
        and (
            row.get("proposal_validated_by_path_feedback") is not True
            or row.get("path_planner_adapter_audit_status") in {"failed", "not_attempted_path_length_preflight_failed"}
        )
    )
    sidecar_screening = sum(
        1 for row in dynamic_validation_rows if row.get("planner_validation_backend") == "sidecar_grid_astar_screening"
    )
    sidecar_diagnostic = sum(
        1 for row in dynamic_validation_rows if row.get("planner_validation_backend") == "sidecar_grid_astar_diagnostic"
    )
    old_sidecar_fallback = sum(
        1 for row in dynamic_validation_rows if row.get("planner_validation_backend") == "sidecar_grid_astar_fallback"
    )
    full_adapter_evidence = bool(
        route_success > 0
        and route_failure == 0
        and sidecar_screening == 0
        and old_sidecar_fallback == 0
        and batch_astar_count == 0
        and path_length_preflight_failures == 0
    )
    adapter_error_samples = unique_sorted(
        [str(row.get("adapter_error_message_tail")) for row in dynamic_validation_rows if row.get("adapter_error_message_tail")]
    )[:3]
    sample_audit = _path_planner_route_adapter_sample_audit(
        config,
        source,
        dynamic_validation_rows,
        steps,
        repo_root=repo_root,
        output_root=output_root,
    )
    dynamic_work_root = Path(str(config.get("dynamic_validation_work_root", "")))
    formal_validation_rows = [
        row
        for row in dynamic_validation_rows
        if row.get("proposal_validated_by_path_feedback") is True and row.get("proposal_only") is False
    ]
    selected_step_rows = [row for row in steps if row.get("executed") is True and row.get("selected_cell") is not None]
    return {
        "schema_version": "xunce-exploration-coverage-dynamic-validation-audit/v1",
        "dynamic_candidate_generation_executed": generation_executed,
        "dynamic_candidate_generation_source": DYNAMIC_FRONTIER_NBV_GENERATION_SOURCE if generation_executed else "",
        "dynamic_candidate_validation_mode": config.get("dynamic_candidate_validation_mode", ""),
        "dynamic_validation_work_root": str(dynamic_work_root),
        "dynamic_validation_work_root_path_length": len(str(dynamic_work_root.resolve())) if str(dynamic_work_root) else 0,
        "dynamic_validation_max_path_length": int(config.get("dynamic_validation_max_path_length", 180)),
        "dynamic_proposal_count": len(dynamic_proposals),
        "dynamic_validation_attempt_count": validation_attempts,
        "dynamic_validation_success_count": validation_success,
        "dynamic_validation_failure_count": max(0, validation_attempts - validation_success),
        "dynamic_validation_cache_hit_count": sum(1 for row in dynamic_validation_rows if row.get("dynamic_validation_cache_hit") is True),
        "dynamic_contract_sidecar_missing_count": contract_sidecar_missing,
        "dynamic_path_length_preflight_failure_count": path_length_preflight_failures,
        "path_planner_route_adapter_success_count": route_success,
        "path_planner_route_adapter_failure_count": route_failure,
        "path_planner_route_adapter_audit_sample_count": sample_audit["sample_count"],
        "path_planner_route_adapter_audit_failure_count": sample_audit["failure_count"],
        "adapter_batch_astar_mismatch_count": sample_audit["mismatch_count"],
        "adapter_audit_passed": sample_audit["passed"],
        "adapter_audit_rows": sample_audit["rows"],
        "in_process_batch_astar_validation_count": batch_astar_count,
        "sidecar_grid_astar_screening_count": sidecar_screening,
        "sidecar_grid_astar_diagnostic_count": sidecar_diagnostic,
        "sidecar_grid_astar_fallback_count": old_sidecar_fallback,
        "adapter_error_type_counts": _count_by_field(dynamic_validation_rows, "adapter_error_type"),
        "adapter_error_message_samples": adapter_error_samples,
        "planner_validation_backend_counts": backend_counts,
        "validation_evidence_kind_counts": evidence_kind_counts,
        "dynamic_validation_full_adapter_evidence_passed": full_adapter_evidence,
        "dynamic_validation_source_root": str(source["root"]),
        "dynamic_candidate_generation_missing_count": generation_missing,
        "candidate_generation_exhausted_count": candidate_generation_exhausted,
        "state_conditioned_candidate_generation": bool(config.get("state_conditioned_candidate_generation", True)) and generation_executed,
        "path_feedback_validation_source_counts": _count_by_field(dynamic_validation_rows, "path_feedback_validation_source"),
        "frontier_candidate_source_counts": _count_by_field(dynamic_validation_rows, "frontier_candidate_source"),
        "candidate_generation_algorithm_source_counts": _count_by_field(dynamic_validation_rows, "candidate_generation_algorithm_source"),
        "coverage_frontier_candidate_count": sum(1 for row in dynamic_validation_rows if row.get("frontier_candidate_source") == "coverage_frontier_boundary"),
        "undercovered_component_candidate_count": sum(
            1
            for row in dynamic_validation_rows
            if row.get("frontier_candidate_source") in {"undercovered_component_centroid", "undercovered_component_boundary"}
        ),
        "low_cost_bridge_candidate_count": sum(1 for row in dynamic_validation_rows if row.get("frontier_candidate_source") == "low_cost_bridge_candidate"),
        "conservative_local_candidate_count": sum(1 for row in dynamic_validation_rows if row.get("frontier_candidate_source") == "conservative_local_candidate"),
        "validated_pareto_frontier_count": sum(1 for row in formal_validation_rows if row.get("candidate_selection_role") == "pareto_frontier"),
        "validated_low_cost_candidate_count": sum(1 for row in formal_validation_rows if row.get("candidate_selection_role") == "low_cost"),
        "validated_efficiency_candidate_count": sum(1 for row in formal_validation_rows if row.get("candidate_selection_role") == "coverage_efficiency"),
        "prevalidation_proposal_drop_count": max(0, len(dynamic_proposals) - validation_attempts),
        "postvalidation_candidate_drop_count": max(0, validation_success - sum(int(row.get("dynamic_validated_candidate_count", 0) or 0) for row in steps)),
        "risk_source_counts": _count_by_field(dynamic_validation_rows, "risk_source"),
        "formal_risk_source_counts": _count_by_field(formal_validation_rows, "risk_source"),
        "selected_risk_source_counts": _count_by_field(selected_step_rows, "selected_risk_source"),
        "route_derived_risk_count": sum(1 for row in dynamic_validation_rows if row.get("risk_route_derived") is True),
        "roi_weight_source_counts": _count_by_field(dynamic_validation_rows, "roi_weight_source"),
        "candidate_selection_mode": "validated_pareto_diverse" if generation_executed else "",
    }


def _path_planner_route_adapter_sample_audit(
    config: dict[str, Any],
    source: dict[str, Any],
    dynamic_validation_rows: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    *,
    repo_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if not bool(config.get("dynamic_adapter_audit_enabled", False)):
        return {"sample_count": 0, "failure_count": 0, "mismatch_count": 0, "passed": False, "rows": []}
    if config.get("candidate_refresh_mode") != "dynamic_frontier_nbv_in_process":
        return {"sample_count": 0, "failure_count": 0, "mismatch_count": 0, "passed": False, "rows": []}

    samples = _adapter_audit_sample_rows(
        dynamic_validation_rows,
        steps,
        max_routes=int(config.get("dynamic_adapter_audit_max_routes", 128)),
        min_per_frontier_source=int(config.get("dynamic_adapter_audit_min_per_frontier_source", 1)),
    )
    if not samples:
        return {"sample_count": 0, "failure_count": 0, "mismatch_count": 0, "passed": False, "rows": []}

    scenario_by_id = {
        str(row.get("scenario_id")): row
        for row in source["path_feedback"].get("scenarios", [])
        if isinstance(row, dict)
    }
    slice_by_id = {str(row.get("scenario_id")): row for row in source["slices"] if isinstance(row, dict)}
    audit_rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        scenario_id = str(sample.get("scenario_id") or "")
        scenario = scenario_by_id.get(scenario_id, {"scenario_id": scenario_id})
        slice_row = slice_by_id.get(scenario_id, {})
        contract_path = _resolved_file(slice_row.get("contract"), repo_root)
        sidecar_path = _resolved_file(slice_row.get("sidecar"), repo_root)
        batch_hash = _short_hash(
            {
                "scenario_id": scenario_id,
                "policy": sample.get("policy"),
                "step_index": sample.get("step_index"),
                "cell": sample.get("cell"),
                "candidate_set_hash": sample.get("candidate_set_hash"),
            }
        )
        if contract_path is None or sidecar_path is None:
            audit_rows.append(
                _adapter_audit_result_row(
                    sample,
                    batch_hash=batch_hash,
                    audit_status="failed",
                    failure_reason="adapter_audit_contract_or_sidecar_missing",
                    mismatch=True,
                )
            )
            continue
        current_cell = _cell_tuple(sample.get("current_cell_before")) or (0, 0)
        output_work_root = Path(str(config.get("dynamic_validation_work_root"))) / "_adapter_audit" / batch_hash
        try:
            adapter_rows = validate_candidate_cells(
                scenario=scenario,
                proposal_rows=[dict(sample)],
                contract_path=contract_path,
                sidecar_path=sidecar_path,
                current_cell=current_cell,
                repo_root=repo_root,
                output_work_root=output_work_root,
                validation_mode=IN_PROCESS_VALIDATION_MODE,
                top_k=1,
                allow_open_grid_fallback=bool(config.get("allow_open_grid_fallback", False)),
                debug_validation_artifacts=bool(config.get("debug_validation_artifacts", False)),
                max_validation_path_length=int(config.get("dynamic_validation_max_path_length", 180)),
                sidecar_fallback_mode="diagnostic_only",
                validation_batch_hash=batch_hash,
            )
        except Exception as exc:  # pragma: no cover - defensive audit isolation
            audit_rows.append(
                _adapter_audit_result_row(
                    sample,
                    batch_hash=batch_hash,
                    audit_status="failed",
                    failure_reason=f"adapter_audit_exception:{type(exc).__name__}",
                    mismatch=True,
                )
            )
            continue
        adapter_row = adapter_rows[0] if adapter_rows else {}
        mismatch_reasons = _adapter_audit_mismatch_reasons(sample, adapter_row)
        audit_rows.append(
            _adapter_audit_result_row(
                sample,
                adapter_row=adapter_row,
                batch_hash=batch_hash,
                audit_status="passed" if not mismatch_reasons else "mismatch",
                failure_reason=";".join(mismatch_reasons),
                mismatch=bool(mismatch_reasons),
                sample_index=index,
            )
        )

    failure_count = sum(1 for row in audit_rows if row.get("adapter_audit_status") == "failed")
    mismatch_count = sum(1 for row in audit_rows if row.get("adapter_batch_astar_mismatch") is True)
    return {
        "sample_count": len(audit_rows),
        "failure_count": failure_count,
        "mismatch_count": mismatch_count,
        "passed": bool(audit_rows) and failure_count == 0 and mismatch_count == 0,
        "rows": audit_rows,
    }


def _adapter_audit_sample_rows(
    dynamic_validation_rows: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    *,
    max_routes: int,
    min_per_frontier_source: int,
) -> list[dict[str, Any]]:
    formal_rows = [
        row
        for row in dynamic_validation_rows
        if row.get("planner_validation_backend") == "in_process_path_planner_astar_batch"
        and row.get("proposal_validated_by_path_feedback") is True
        and row.get("proposal_only") is False
        and _cell_tuple(row.get("cell")) is not None
    ]
    if not formal_rows or max_routes <= 0:
        return []
    selected_keys = {
        (
            str(row.get("scenario_id")),
            str(row.get("policy")),
            int(row.get("step_index", 0) or 0),
            str(row.get("candidate_set_hash") or ""),
            tuple(_cell_tuple(row.get("selected_cell")) or (-1, -1)),
        )
        for row in steps
        if row.get("selected_cell") is not None
    }
    selected: list[dict[str, Any]] = []
    family: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def add(row: dict[str, Any], bucket: list[dict[str, Any]]) -> None:
        key = _adapter_audit_row_key(row)
        if key in seen:
            return
        seen.add(key)
        bucket.append(row)

    for row in sorted(formal_rows, key=_adapter_audit_sort_key):
        cell = _cell_tuple(row.get("cell")) or (-1, -1)
        selected_key = (
            str(row.get("scenario_id")),
            str(row.get("policy")),
            int(row.get("step_index", 0) or 0),
            str(row.get("candidate_set_hash") or ""),
            cell,
        )
        if selected_key in selected_keys:
            add(row, selected)

    for source_name in unique_sorted([str(row.get("frontier_candidate_source") or "missing") for row in formal_rows]):
        count = 0
        for row in sorted(formal_rows, key=_adapter_audit_sort_key):
            if str(row.get("frontier_candidate_source") or "missing") != source_name:
                continue
            add(row, family)
            count += 1
            if count >= min_per_frontier_source:
                break

    for row in sorted(formal_rows, key=_adapter_audit_sort_key):
        add(row, remaining)

    ordered = selected + family + remaining
    deduped: list[dict[str, Any]] = []
    emitted: set[tuple[Any, ...]] = set()
    for row in ordered:
        key = _adapter_audit_row_key(row)
        if key in emitted:
            continue
        emitted.add(key)
        deduped.append(dict(row))
        if len(deduped) >= max_routes:
            break
    return deduped


def _adapter_audit_mismatch_reasons(batch_row: dict[str, Any], adapter_row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if adapter_row.get("proposal_validated_by_path_feedback") is not True or adapter_row.get("proposal_only") is True:
        reasons.append("adapter_candidate_not_formal")
    for field in ("reachable", "open_grid_fallback_used"):
        if bool(batch_row.get(field)) != bool(adapter_row.get(field)):
            reasons.append(f"{field}_mismatch")
    for field in ("path_cost", "path_length", "risk"):
        left = _finite_or_none(batch_row.get(field))
        right = _finite_or_none(adapter_row.get(field))
        if left is None or right is None:
            reasons.append(f"{field}_missing")
        elif abs(left - right) > max(1.0e-6, 1.0e-6 * max(abs(left), abs(right), 1.0)):
            reasons.append(f"{field}_mismatch")
    return unique_sorted(reasons)


def _adapter_audit_result_row(
    batch_row: dict[str, Any],
    *,
    batch_hash: str,
    audit_status: str,
    failure_reason: str,
    mismatch: bool,
    adapter_row: dict[str, Any] | None = None,
    sample_index: int = 0,
) -> dict[str, Any]:
    adapter_row = adapter_row or {}
    return {
        "schema_version": "xunce-exploration-coverage-adapter-sample-audit/v1",
        "sample_index": int(sample_index),
        "scenario_id": batch_row.get("scenario_id"),
        "policy": batch_row.get("policy"),
        "step_index": batch_row.get("step_index"),
        "candidate_set_hash": batch_row.get("candidate_set_hash"),
        "cell": batch_row.get("cell"),
        "frontier_candidate_source": batch_row.get("frontier_candidate_source"),
        "validation_batch_hash": batch_hash,
        "batch_backend": batch_row.get("planner_validation_backend"),
        "adapter_backend": adapter_row.get("planner_validation_backend"),
        "batch_path_cost": batch_row.get("path_cost"),
        "adapter_path_cost": adapter_row.get("path_cost"),
        "batch_path_length": batch_row.get("path_length"),
        "adapter_path_length": adapter_row.get("path_length"),
        "batch_risk": batch_row.get("risk"),
        "adapter_risk": adapter_row.get("risk"),
        "adapter_audit_status": audit_status,
        "adapter_batch_astar_mismatch": bool(mismatch),
        "failure_reason": failure_reason,
    }


def _adapter_audit_row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("scenario_id")),
        str(row.get("policy")),
        int(row.get("step_index", 0) or 0),
        str(row.get("candidate_set_hash") or ""),
        _cell_tuple(row.get("cell")),
    )


def _adapter_audit_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    cell = _cell_tuple(row.get("cell")) or (1_000_000, 1_000_000)
    return (
        str(row.get("scenario_id")),
        int(row.get("step_index", 0) or 0),
        str(row.get("policy")),
        str(row.get("frontier_candidate_source") or ""),
        cell[0],
        cell[1],
    )


def _resolved_file(value: Any, repo_root: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = resolve_path(Path(value), repo_root)
    return path if path.is_file() else None


def _short_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _candidate_set_hash_mismatch_count(steps: list[dict[str, Any]]) -> int:
    by_key: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in steps:
        by_key[(str(row.get("scenario_id")), int(row.get("step_index", 0)))].add(str(row.get("candidate_set_hash") or ""))
    return sum(1 for hashes in by_key.values() if len(hashes - {""}) > 1)


def _model_inference_audit(model_bundle: dict[str, Any], inference_rows: list[dict[str, Any]]) -> dict[str, Any]:
    reason_codes = list(model_bundle["reason_codes"])
    if not model_bundle["xunce_checkpoint_loaded"] or not model_bundle["incumbent_checkpoint_loaded"]:
        reason_codes.append("true_model_inference_not_executed")
    auditable_rows = [
        row
        for row in inference_rows
        if row.get("inference_skipped_reason") != "candidate_generation_exhausted"
        and "candidate_generation_exhausted" not in set(row.get("reason_codes", []))
    ]
    executed_rows = [row for row in auditable_rows if row.get("true_model_inference_executed")]
    finite_count = sum(1 for row in auditable_rows if row.get("detail", {}).get("finite_outputs") is True)
    mask_violation_count = sum(_int_value(row.get("model_inference_mask_violation_count")) for row in inference_rows)
    model_inference_failure_count = sum(
        _int_value(row.get("model_inference_failure_count"))
        for row in auditable_rows
    )
    model_inference_failure_count += sum(
        1
        for row in auditable_rows
        if "model_inference_failure" in set(row.get("reason_codes", []))
        and _int_value(row.get("model_inference_failure_count")) == 0
    )
    if auditable_rows and len(executed_rows) != len(auditable_rows):
        reason_codes.append("true_model_inference_not_executed")
    if auditable_rows and finite_count != len(auditable_rows):
        reason_codes.append("model_inference_non_finite_output")
    if model_inference_failure_count:
        reason_codes.append("model_inference_failure")
    if mask_violation_count:
        reason_codes.append("model_inference_mask_violation")
    true_model_inference_executed = bool(
        len(executed_rows) == len(auditable_rows)
        and model_bundle["xunce_checkpoint_loaded"]
        and model_bundle["incumbent_checkpoint_loaded"]
        and model_inference_failure_count == 0
    )
    return {
        "schema_version": "xunce-exploration-coverage-model-inference-audit/v1",
        "true_model_inference_executed": true_model_inference_executed,
        "proxy_selection_used": False,
        "input_source": "high_fidelity_scenario_adapter/v1",
        "xunce_checkpoint_loaded": model_bundle["xunce_checkpoint_loaded"],
        "incumbent_checkpoint_loaded": model_bundle["incumbent_checkpoint_loaded"],
        "xunce_parameter_count": model_bundle["xunce_parameter_count"],
        "incumbent_parameter_count": model_bundle["incumbent_parameter_count"],
        "model_inference_finite_output_count": finite_count,
        "model_inference_failure_count": model_inference_failure_count,
        "model_inference_mask_violation_count": mask_violation_count,
        "passed": not reason_codes,
        "reason_codes": unique_sorted(reason_codes),
    }


def _source_match_audit(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    reasons = list(source["read_reason_codes"])
    expansion = source["expansion"]
    slices = source["slices"]
    if expansion.get("status") != "passed":
        reasons.append("xunce_high_fidelity_roi_expansion_not_passed")
    if _int_value(expansion.get("slice_count")) < config["required_scenario_count"]:
        reasons.append("xunce_high_fidelity_roi_expansion_slice_count_short")
    if config["require_context_ids"]:
        missing = sum(1 for row in slices[: config["required_scenario_count"]] if not row.get("context_id"))
        if missing:
            reasons.append("xunce_high_fidelity_context_id_missing")
    if config["require_contract_and_sidecar_paths"]:
        missing_paths = sum(
            1
            for row in slices[: config["required_scenario_count"]]
            if not row.get("contract") or not row.get("sidecar")
        )
        if missing_paths:
            reasons.append("xunce_high_fidelity_contract_or_sidecar_missing")
    return {
        "schema_version": "xunce-exploration-coverage-source-match-audit/v1",
        "passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "source_slice_count": _int_value(expansion.get("slice_count")),
        "required_scenario_count": config["required_scenario_count"],
    }


def _boundary_audit(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    boundary = _stage18b_boundary_audit(config, source)
    if config["canary_traffic_fraction"] > 0:
        boundary["passed"] = False
        boundary["reason_codes"] = unique_sorted(list(boundary.get("reason_codes", [])) + ["xunce_coverage_comparison_boundary_violation"])
    return {**boundary, **_boundary_fields()}


def _decision(
    *,
    config: dict[str, Any],
    source: dict[str, Any],
    boundary: dict[str, Any],
    source_match: dict[str, Any],
    model_inference: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    reasons = unique_sorted(
        list(source_match["reason_codes"])
        + list(boundary.get("reason_codes", []))
        + list(model_inference["reason_codes"])
    )
    if "missing_xunce_candidate_checkpoint" in source["read_reason_codes"]:
        reasons.append("missing_xunce_candidate_checkpoint")
    if "missing_incumbent_policy_checkpoint" in source["read_reason_codes"]:
        reasons.append("missing_incumbent_policy_checkpoint")
    if config["candidate_refresh_mode"] == "dynamic_frontier_nbv_in_process":
        if comparison.get("dynamic_contract_sidecar_missing_count", 0) > 0:
            reasons.append("dynamic_contract_sidecar_missing")
        if comparison.get("dynamic_candidate_generation_missing_count", 0) > 0:
            reasons.append("dynamic_candidate_generation_missing")
        if comparison.get("dynamic_validation_attempt_count", 0) <= 0:
            reasons.append("dynamic_candidate_validation_not_executed")
    if model_inference.get("model_inference_failure_count", 0) > 0:
        reasons.append("model_inference_failure")
    reasons = unique_sorted(reasons)
    coverage_advantage = (
        not reasons
        and comparison["xunce_coverage_return_delta_vs_incumbent"] > TOLERANCE
        and comparison["xunce_coverage_curve_auc_delta_vs_incumbent"] > TOLERANCE
        and comparison["xunce_safety_regression_count"] == 0
        and comparison["xunce_efficiency_regression_count"] == 0
        and model_inference["model_inference_mask_violation_count"] == 0
        and comparison["open_grid_fallback_count"] == 0
        and (
            comparison["incumbent_oracle_regret"] <= TOLERANCE
            or comparison["xunce_oracle_regret"] < comparison["incumbent_oracle_regret"] - TOLERANCE
        )
    )
    coverage_advantage_with_efficiency_regression = (
        not reasons
        and comparison["xunce_coverage_return_delta_vs_incumbent"] > TOLERANCE
        and comparison["xunce_efficiency_regression_count"] > 0
    )
    if reasons:
        status = "failed"
        if (
            "missing_xunce_candidate_checkpoint" in reasons
            or "invalid_xunce_candidate_checkpoint" in reasons
            or "xunce_checkpoint_state_dict_missing" in reasons
            or "xunce_checkpoint_model_config_missing" in reasons
        ):
            next_change = FIX_XUNCE_CHECKPOINT_NEXT_REQUIRED_CHANGE
        elif "missing_incumbent_policy_checkpoint" in reasons or "incumbent_checkpoint_format_unsupported" in reasons:
            next_change = FIX_INCUMBENT_CHECKPOINT_NEXT_REQUIRED_CHANGE
        elif "true_model_inference_not_executed" in reasons or "model_inference_failure" in reasons:
            next_change = FIX_RUNNER_NEXT_REQUIRED_CHANGE
        elif "model_inference_mask_violation" in reasons or any("boundary" in reason for reason in reasons):
            next_change = BOUNDARY_NEXT_REQUIRED_CHANGE
        else:
            next_change = FIX_SOURCE_NEXT_REQUIRED_CHANGE
    else:
        status = "passed"
        next_change = REVIEW_COMPARISON_NEXT_REQUIRED_CHANGE
    diagnostic_reasons: list[str] = list(config.get("diagnostic_reason_codes", []))
    if not coverage_advantage:
        diagnostic_reasons.append("xunce_coverage_advantage_not_established")
    if coverage_advantage_with_efficiency_regression:
        diagnostic_reasons.append("xunce_coverage_advantage_with_efficiency_regression")
    if comparison["coverage_gain_per_path_cost_delta_vs_incumbent"] < -TOLERANCE:
        diagnostic_reasons.append("coverage_gain_per_path_cost_regressive")
    if comparison["coverage_gain_per_risk_delta_vs_incumbent"] < -TOLERANCE:
        diagnostic_reasons.append("coverage_gain_per_risk_regressive")
    if comparison.get("dynamic_candidate_validation_missing_count", 0) > 0:
        diagnostic_reasons.append("dynamic_candidate_validation_missing")
    if config["candidate_refresh_mode"] == "dynamic_frontier_nbv_in_process" and comparison.get("dynamic_validation_success_count", 0) <= 0:
        diagnostic_reasons.append("dynamic_candidate_validation_no_success")
    if comparison.get("candidate_generation_exhausted_count", 0) > 0:
        diagnostic_reasons.append("candidate_generation_exhausted")
    if config["candidate_refresh_mode"] == "dynamic_frontier_nbv_in_process":
        if comparison.get("dynamic_path_length_preflight_failure_count", 0) > 0:
            diagnostic_reasons.append("dynamic_validation_path_length_preflight_failed")
        if comparison.get("dynamic_validation_full_adapter_evidence_passed") is not True:
            if comparison.get("sidecar_grid_astar_screening_count", 0) > 0 or comparison.get("dynamic_sidecar_grid_astar_fallback_count", 0) > 0:
                diagnostic_reasons.append("sidecar_screening_not_full_adapter_evidence")
            elif comparison.get("in_process_batch_astar_validation_count", 0) > 0:
                diagnostic_reasons.append("dynamic_batch_astar_screening_not_full_adapter_evidence")
            else:
                diagnostic_reasons.append("dynamic_validation_not_full_adapter_evidence")
    blocking_reasons = unique_sorted(reasons)
    diagnostic_reasons = unique_sorted(diagnostic_reasons)
    evidence_gate = not any(
        reason
        in {
            "missing_xunce_candidate_checkpoint",
            "invalid_xunce_candidate_checkpoint",
            "xunce_checkpoint_state_dict_missing",
            "xunce_checkpoint_model_config_missing",
            "missing_incumbent_policy_checkpoint",
            "incumbent_checkpoint_format_unsupported",
            "true_model_inference_not_executed",
            "proxy_selection_used",
        }
        for reason in blocking_reasons
    )
    candidate_gate = not any(
        reason in {"model_inference_mask_violation", "no_valid_candidates"}
        or "boundary" in reason
        or "fallback" in reason
        or reason.startswith("dynamic_")
        for reason in blocking_reasons
    )
    return {
        "schema_version": "xunce-exploration-coverage-decision-audit/v1",
        "status": status,
        "reason_codes": blocking_reasons,
        "blocking_reason_codes": blocking_reasons,
        "diagnostic_reason_codes": diagnostic_reasons,
        "diagnostic_recommended_change": _coverage_diagnostic_recommended_change(diagnostic_reasons),
        "evidence_authenticity_gate_passed": evidence_gate,
        "candidate_validity_gate_passed": candidate_gate,
        "comparison_allowed": status == "passed" and evidence_gate and candidate_gate,
        "xunce_coverage_advantage_established": coverage_advantage,
        "coverage_advantage_with_efficiency_regression": coverage_advantage_with_efficiency_regression,
        "next_required_change": next_change,
        **_boundary_fields(),
    }


def _coverage_diagnostic_recommended_change(diagnostic_reasons: list[str]) -> str:
    reason_set = set(diagnostic_reasons)
    if "xunce_coverage_advantage_with_efficiency_regression" in reason_set or "coverage_gain_per_path_cost_regressive" in reason_set:
        return EFFICIENCY_NEXT_REQUIRED_CHANGE
    if "xunce_coverage_advantage_not_established" in reason_set:
        return NO_ADVANTAGE_NEXT_REQUIRED_CHANGE
    return ""


def _summary(
    *,
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    config: dict[str, Any],
    source: dict[str, Any],
    boundary: dict[str, Any],
    source_match: dict[str, Any],
    model_inference: dict[str, Any],
    comparison: dict[str, Any],
    roi_breakdown: dict[str, Any],
    decision: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    xunce_latency = comparison["xunce_median_inference_latency_ms"]
    incumbent_latency = comparison["incumbent_median_inference_latency_ms"]
    latency_ratio = xunce_latency / max(incumbent_latency, 1.0e-9) if incumbent_latency else 0.0
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "blocking_reason_codes": decision["blocking_reason_codes"],
        "diagnostic_reason_codes": decision["diagnostic_reason_codes"],
        "diagnostic_recommended_change": decision["diagnostic_recommended_change"],
        "evidence_authenticity_gate_passed": decision["evidence_authenticity_gate_passed"],
        "candidate_validity_gate_passed": decision["candidate_validity_gate_passed"],
        "comparison_allowed": decision["comparison_allowed"],
        "true_model_inference_executed": model_inference["true_model_inference_executed"],
        "proxy_selection_used": model_inference["proxy_selection_used"],
        "xunce_checkpoint_loaded": model_inference["xunce_checkpoint_loaded"],
        "incumbent_checkpoint_loaded": model_inference["incumbent_checkpoint_loaded"],
        "scenario_count": comparison["scenario_count"],
        "normalized_config": _normalized_config_snapshot(config),
        "rollout_steps": config["rollout_steps"],
        "coverage_radius_cells": config["coverage_radius_cells"],
        "coverage_radius_sensitivity": config["coverage_radius_sensitivity"],
        "planning_backend": config["planning_backend"],
        "candidate_refresh_mode": config["candidate_refresh_mode"],
        "coverage_metric_mode": config["coverage_metric_mode"],
        "include_oracle_baselines": config["include_oracle_baselines"],
        "xunce_only_evaluation": bool(config.get("xunce_only_evaluation", False)),
        "hybrid_astar_candidate_eval_workers": int(config.get("hybrid_astar_candidate_eval_workers", 1)),
        "include_roi_weighted_coverage": config["include_roi_weighted_coverage"],
        "canonical_reward_profile": config["canonical_reward_profile"],
        "profile_id": config["profile_id"],
        "profile_version": config["profile_version"],
        "profile_hash": config["profile_hash"],
        "include_canonical_reward_rerank_oracle": config["include_canonical_reward_rerank_oracle"],
        "canonical_reward_rerank_profile": config["canonical_reward_rerank_profile"],
        "canonical_reward_rerank_profile_id": config["canonical_reward_rerank_profile_id"],
        "canonical_reward_rerank_profile_version": config["canonical_reward_rerank_profile_version"],
        "canonical_reward_rerank_profile_hash": config["canonical_reward_rerank_profile_hash"],
        "emit_on_policy_oracle_teacher_labels": config["emit_on_policy_oracle_teacher_labels"],
        "on_policy_oracle_teacher_label_audit": str(paths["on_policy_oracle_teacher_labels"])
        if config["emit_on_policy_oracle_teacher_labels"]
        else None,
        "on_policy_oracle_teacher_label_row_count": comparison["on_policy_oracle_teacher_label_row_count"],
        "on_policy_oracle_teacher_baseline_policy": config["on_policy_oracle_teacher_baseline_policy"],
        "on_policy_oracle_teacher_profile": config["on_policy_oracle_teacher_profile"],
        "on_policy_oracle_teacher_profile_id": config["on_policy_oracle_teacher_profile_id"],
        "on_policy_oracle_teacher_profile_version": config["on_policy_oracle_teacher_profile_version"],
        "on_policy_oracle_teacher_profile_hash": config["on_policy_oracle_teacher_profile_hash"],
        "canonical_guard_thresholds": config["canonical_guard_thresholds"],
        "comparison_utility_profiles_diagnostic_only": config["comparison_utility_profiles_diagnostic_only"],
        "stage19_readiness_source": config["stage19_readiness_source"],
        "xunce_coverage_advantage_established": decision["xunce_coverage_advantage_established"],
        "xunce_coverage_better_count": comparison["xunce_coverage_better_count"],
        "xunce_coverage_worse_count": comparison["xunce_coverage_worse_count"],
        "xunce_coverage_tie_count": comparison["xunce_coverage_tie_count"],
        "xunce_efficiency_regression_count": comparison["xunce_efficiency_regression_count"],
        "xunce_safety_regression_count": comparison["xunce_safety_regression_count"],
        "model_inference_mask_violation_count": model_inference["model_inference_mask_violation_count"],
        "model_inference_finite_output_count": model_inference["model_inference_finite_output_count"],
        "model_inference_failure_count": model_inference["model_inference_failure_count"],
        "candidate_generation_exhausted_count": comparison["candidate_generation_exhausted_count"],
        "coverage_rate_saturation_episode_count": comparison["coverage_rate_saturation_episode_count"],
        "max_final_coverage_rate_raw": comparison["max_final_coverage_rate_raw"],
        "max_coverage_rate_saturation_excess": comparison["max_coverage_rate_saturation_excess"],
        "open_grid_fallback_count": comparison["open_grid_fallback_count"],
        "unreachable_selected_count": comparison["unreachable_selected_count"],
        "path_planning_failure_count": comparison["path_planning_failure_count"],
        "risk_regression_count": comparison["risk_regression_count"],
        "path_cost_regression_count": comparison["path_cost_regression_count"],
        "xunce_final_coverage_rate_delta_vs_incumbent": comparison["xunce_final_coverage_rate_delta_vs_incumbent"],
        "xunce_coverage_return_delta_vs_incumbent": comparison["xunce_coverage_return_delta_vs_incumbent"],
        "xunce_coverage_curve_auc_delta_vs_incumbent": comparison["xunce_coverage_curve_auc_delta_vs_incumbent"],
        "xunce_new_covered_cell_delta_vs_incumbent": comparison["xunce_new_covered_cell_delta_vs_incumbent"],
        "xunce_min_roi_group_coverage_delta_vs_incumbent": comparison["xunce_min_roi_group_coverage_delta_vs_incumbent"],
        "coverage_gain_per_path_cost_delta_vs_incumbent": comparison["coverage_gain_per_path_cost_delta_vs_incumbent"],
        "coverage_gain_per_risk_delta_vs_incumbent": comparison["coverage_gain_per_risk_delta_vs_incumbent"],
        "xunce_path_cost_delta_vs_incumbent": comparison["xunce_path_cost_delta_vs_incumbent"],
        "xunce_risk_delta_vs_incumbent": comparison["xunce_risk_delta_vs_incumbent"],
        "risk_cost_weighted_delta_vs_incumbent": comparison["risk_cost_weighted_delta_vs_incumbent"],
        "coverage_per_100m_delta_vs_incumbent": comparison["coverage_per_100m_delta_vs_incumbent"],
        "risk_per_100m_delta_vs_incumbent": comparison["risk_per_100m_delta_vs_incumbent"],
        "comparison_aggregate": comparison["comparison_aggregate"],
        "comparison_utility_profiles": config["comparison_utility_profiles"],
        "dynamic_candidate_generation_executed": comparison["dynamic_candidate_generation_executed"],
        "dynamic_candidate_generation_source": comparison["dynamic_candidate_generation_source"],
        "dynamic_candidate_validation_mode": comparison["dynamic_candidate_validation_mode"],
        "dynamic_validation_work_root": comparison["dynamic_validation_work_root"],
        "dynamic_validation_work_root_path_length": comparison["dynamic_validation_work_root_path_length"],
        "dynamic_validation_max_path_length": comparison["dynamic_validation_max_path_length"],
        "dynamic_proposal_count": comparison["dynamic_proposal_count"],
        "dynamic_validation_attempt_count": comparison["dynamic_validation_attempt_count"],
        "dynamic_validation_success_count": comparison["dynamic_validation_success_count"],
        "dynamic_validation_failure_count": comparison["dynamic_validation_failure_count"],
        "dynamic_validation_cache_hit_count": comparison["dynamic_validation_cache_hit_count"],
        "dynamic_contract_sidecar_missing_count": comparison["dynamic_contract_sidecar_missing_count"],
        "dynamic_path_length_preflight_failure_count": comparison["dynamic_path_length_preflight_failure_count"],
        "in_process_batch_astar_validation_count": comparison["in_process_batch_astar_validation_count"],
        "path_planner_route_adapter_success_count": comparison["path_planner_route_adapter_success_count"],
        "path_planner_route_adapter_failure_count": comparison["path_planner_route_adapter_failure_count"],
        "path_planner_route_adapter_audit_sample_count": comparison["path_planner_route_adapter_audit_sample_count"],
        "path_planner_route_adapter_audit_failure_count": comparison["path_planner_route_adapter_audit_failure_count"],
        "adapter_batch_astar_mismatch_count": comparison["adapter_batch_astar_mismatch_count"],
        "adapter_audit_passed": comparison["adapter_audit_passed"],
        "sidecar_grid_astar_screening_count": comparison["sidecar_grid_astar_screening_count"],
        "sidecar_grid_astar_diagnostic_count": comparison["sidecar_grid_astar_diagnostic_count"],
        "adapter_error_type_counts": comparison["adapter_error_type_counts"],
        "adapter_error_message_samples": comparison["adapter_error_message_samples"],
        "planner_validation_backend_counts": comparison["planner_validation_backend_counts"],
        "validation_evidence_kind_counts": comparison["validation_evidence_kind_counts"],
        "dynamic_validation_full_adapter_evidence_passed": comparison["dynamic_validation_full_adapter_evidence_passed"],
        "dynamic_planner_validation_backend_counts": comparison["dynamic_planner_validation_backend_counts"],
        "dynamic_sidecar_grid_astar_fallback_count": comparison["dynamic_sidecar_grid_astar_fallback_count"],
        "coverage_frontier_candidate_count": comparison["coverage_frontier_candidate_count"],
        "undercovered_component_candidate_count": comparison["undercovered_component_candidate_count"],
        "candidate_generation_algorithm_source_counts": comparison["candidate_generation_algorithm_source_counts"],
        "low_cost_bridge_candidate_count": comparison["low_cost_bridge_candidate_count"],
        "conservative_local_candidate_count": comparison["conservative_local_candidate_count"],
        "validated_pareto_frontier_count": comparison["validated_pareto_frontier_count"],
        "validated_low_cost_candidate_count": comparison["validated_low_cost_candidate_count"],
        "validated_efficiency_candidate_count": comparison["validated_efficiency_candidate_count"],
        "prevalidation_proposal_drop_count": comparison["prevalidation_proposal_drop_count"],
        "postvalidation_candidate_drop_count": comparison["postvalidation_candidate_drop_count"],
        "risk_source_counts": comparison["risk_source_counts"],
        "formal_risk_source_counts": comparison["formal_risk_source_counts"],
        "selected_risk_source_counts": comparison["selected_risk_source_counts"],
        "route_derived_risk_count": comparison["route_derived_risk_count"],
        "roi_weight_source_counts": comparison["roi_weight_source_counts"],
        "candidate_selection_mode": comparison["candidate_selection_mode"],
        "dynamic_validation_source_root": comparison["dynamic_validation_source_root"],
        "dynamic_candidate_generation_missing_count": comparison["dynamic_candidate_generation_missing_count"],
        "dynamic_candidate_generation_exhausted_count": comparison["candidate_generation_exhausted_count"],
        "state_conditioned_candidate_generation": comparison["state_conditioned_candidate_generation"],
        "candidate_set_hash_mismatch_count": comparison["candidate_set_hash_mismatch_count"],
        "paired_decision_audit_row_count": comparison["paired_decision_audit_row_count"],
        "candidate_generation_effect_scope": comparison["candidate_generation_effect_scope"],
        "model_selection_evidence_scope": comparison["model_selection_evidence_scope"],
        "closed_loop_dynamic_rollout_summary": comparison["closed_loop_dynamic_rollout_summary"],
        "same_candidate_set_policy_selection_summary": comparison["same_candidate_set_policy_selection_summary"],
        "closed_loop_dynamic_rollout_advantage_established": comparison["closed_loop_dynamic_rollout_advantage_established"],
        "same_candidate_set_policy_selection_advantage_established": comparison["same_candidate_set_policy_selection_advantage_established"],
        "dynamic_candidate_validation_missing_count": comparison["dynamic_candidate_validation_missing_count"],
        "policy_disagreement_count": comparison["policy_disagreement_count"],
        "useful_disagreement_count": comparison["useful_disagreement_count"],
        "xunce_oracle_regret": comparison["xunce_oracle_regret"],
        "incumbent_oracle_regret": comparison["incumbent_oracle_regret"],
        "oracle_vs_incumbent_coverage_delta": comparison["oracle_vs_incumbent_coverage_delta"],
        "evaluation_task_discriminative": comparison["evaluation_task_discriminative"],
        "selected_probability_median": comparison["selected_probability_median"],
        "action_entropy_median": comparison["action_entropy_median"],
        "selected_rank_median": comparison["selected_rank_median"],
        "xunce_parameter_count": model_inference["xunce_parameter_count"],
        "incumbent_parameter_count": model_inference["incumbent_parameter_count"],
        "xunce_median_inference_latency_ms": xunce_latency,
        "incumbent_median_inference_latency_ms": incumbent_latency,
        "latency_ratio_vs_incumbent": latency_ratio,
        "source_match_audit_passed": source_match["passed"],
        "boundary_audit_passed": boundary["passed"],
        "roi_group_count": roi_breakdown["roi_group_count"],
        "next_required_change": decision["next_required_change"],
        "summary": str(paths["summary"]),
        "episodes": str(paths["episodes"]),
        "steps": str(paths["steps"]),
        "comparison_pairs": str(paths["comparison_pairs"]),
        "comparison_aggregate_path": str(paths["comparison_aggregate"]),
        "dynamic_proposals": str(paths["dynamic_proposals"]),
        "dynamic_validation_results": str(paths["dynamic_validation_results"]),
        "dynamic_validation_audit": str(paths["dynamic_validation_audit"]),
        "paired_decision_audit": str(paths["paired_decision_audit"]),
        "candidate_metric_audit": str(paths["candidate_metric_audit"]) if config["emit_candidate_metric_audit"] else None,
        "candidate_metric_audit_row_count": comparison["candidate_metric_audit_row_count"],
        "obstacle_source_audit": str(paths["obstacle_sources"]),
        "obstacle_source_count": source.get("obstacle_source_count"),
        "model_inference": str(paths["model_inference"]),
        "roi_breakdown": str(paths["roi_breakdown"]),
        "decision_audit": str(paths["decision_audit"]),
        "config": str(config_path),
        "output_root": str(output_root),
        "source_roi_expansion_root": str(source["root"]),
        "git_provenance": git_snapshot(repo_root),
        **_boundary_fields(),
    }


def _normalized_config_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in sorted(config.items()):
        output[key] = _json_safe_config_value(value)
    return output


def _json_safe_config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_json_safe_config_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_config_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_config_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    return str(value)


def _roi_breakdown(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        groups[str(episode["roi_group"])].append(episode)
    rows = []
    for roi_group, rows_for_group in sorted(groups.items()):
        xunce = [row for row in rows_for_group if row["policy"] == "xunce"]
        incumbent = [row for row in rows_for_group if row["policy"] == "incumbent"]
        rows.append(
            {
                "roi_group": roi_group,
                "episode_count": len(rows_for_group),
                "xunce_mean_coverage_return": _mean([row["coverage_return"] for row in xunce]),
                "incumbent_mean_coverage_return": _mean([row["coverage_return"] for row in incumbent]),
                "xunce_mean_final_coverage_rate": _mean([row["final_coverage_rate"] for row in xunce]),
                "incumbent_mean_final_coverage_rate": _mean([row["final_coverage_rate"] for row in incumbent]),
            }
        )
    return {"schema_version": "xunce-exploration-coverage-roi-breakdown/v1", "roi_group_count": len(rows), "families": rows}


def _public_episode(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"steps", "inference_rows", "action_indices", "dynamic_proposals", "dynamic_validation_rows", "paired_decision_rows"}
    }


def _disagreement_counts(steps: list[dict[str, Any]]) -> tuple[int, int]:
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in steps:
        by_key[(str(row["scenario_id"]), int(row["step_index"]))][str(row["policy"])] = row
    disagreement = 0
    useful = 0
    for policies in by_key.values():
        xunce = policies.get("xunce")
        incumbent = policies.get("incumbent")
        if not xunce or not incumbent:
            continue
        if xunce["selected_action_index"] != incumbent["selected_action_index"]:
            disagreement += 1
            if (
                float(xunce["coverage_rate_delta"]) > float(incumbent["coverage_rate_delta"]) + TOLERANCE
                and _none_to_zero(xunce.get("path_cost")) <= _none_to_zero(incumbent.get("path_cost")) + TOLERANCE
                and _none_to_zero(xunce.get("risk")) <= _none_to_zero(incumbent.get("risk")) + TOLERANCE
                and not xunce["model_inference_mask_violation"]
            ):
                useful += 1
    return disagreement, useful


def _candidate_rows_for_step(
    scenario: dict[str, Any],
    *,
    current_cell: tuple[int, int],
    current_theta_deg: float,
    covered_cells: set[tuple[int, int]],
    step_index: int,
    config: dict[str, Any],
    scenario_id: str,
    slice_row: dict[str, Any],
    repo_root: Path,
    output_root: Path,
    validation_cache: dict[str, list[dict[str, Any]]],
    hybrid_astar_executor: Any | None = None,
) -> dict[str, Any]:
    def with_theta(batch: dict[str, Any]) -> dict[str, Any]:
        if not theta_viewpoints_enabled(config):
            return batch
        base_candidates = [dict(candidate) for candidate in batch["candidates"]]
        base_hash = str(batch["candidate_set_hash"])
        expansion = expand_theta_aware_candidates(
            base_candidates,
            current_cell=current_cell,
            covered_cells=covered_cells,
            config=config,
            base_candidate_set_hash=base_hash,
        )
        viewpoint_candidates = expansion["candidates"]
        batch = dict(batch)
        batch["base_candidates"] = base_candidates
        batch["base_candidate_set_hash"] = base_hash
        batch["candidates"] = viewpoint_candidates
        batch["candidate_set_hash"] = expansion["candidate_set_hash"]
        batch["candidate_set_id"] = f"{scenario_id}:step-{step_index}:theta:{expansion['candidate_set_hash'][:16]}"
        batch["candidate_generation_source"] = f"{batch['candidate_generation_source']}+theta_viewpoint"
        batch["theta_aware_candidate_viewpoints_enabled"] = True
        batch["theta_values"] = expansion["theta_values"]
        batch["theta_value_count"] = expansion["theta_value_count"]
        batch["base_candidate_count"] = expansion["base_candidate_count"]
        batch["viewpoint_candidate_count"] = expansion["viewpoint_candidate_count"]
        batch["sensor_model_id"] = expansion["sensor_model_id"]
        batch["sensor_fov_deg"] = expansion["sensor_fov_deg"]
        batch["sensor_range_cells"] = expansion["sensor_range_cells"]
        return batch

    def finalize(batch: dict[str, Any]) -> dict[str, Any]:
        batch = with_theta(batch)
        if config.get("hybrid_astar_pose_path_cost_enabled") and not continuous_theta_enabled(config):
            _enrich_candidates_with_hybrid_astar_path_cost(
                batch["candidates"],
                current_cell=current_cell,
                current_theta_deg=current_theta_deg,
                candidate_set_hash_value=str(batch["candidate_set_hash"]),
                step_index=step_index,
                scenario_id=scenario_id,
                config=config,
                slice_row=slice_row,
                repo_root=repo_root,
                hybrid_astar_executor=hybrid_astar_executor,
            )
        return batch

    candidates = [dict(candidate) for candidate in _candidate_rows(scenario)]
    if config["candidate_refresh_mode"] == "dynamic_frontier_nbv_in_process":
        formal_candidates, proposal_rows, validation_rows = build_dynamic_frontier_nbv_candidates(
            scenario=scenario,
            slice_row=slice_row,
            current_cell=current_cell,
            covered_cells=covered_cells,
            step_index=step_index,
            config=config,
            repo_root=repo_root,
            output_work_root=Path(config["dynamic_validation_work_root"]),
            validation_cache=validation_cache,
        )
        batch_hash = candidate_set_hash(formal_candidates)
        return finalize({
            "candidates": formal_candidates,
            "dynamic_proposals": proposal_rows,
            "dynamic_validation_rows": validation_rows,
            "candidate_generation_source": DYNAMIC_FRONTIER_NBV_GENERATION_SOURCE,
            "candidate_set_id": f"{scenario_id}:step-{step_index}:dynamic:{batch_hash[:16]}",
            "candidate_set_hash": batch_hash,
            "dynamic_generation_executed": True,
            "dynamic_proposal_count": len(proposal_rows),
            "dynamic_validated_candidate_count": len(formal_candidates),
            "dynamic_validation_cache_hit": bool(validation_rows and all(row.get("dynamic_validation_cache_hit") is True for row in validation_rows)),
            "path_feedback_validation_source_counts": _count_by_field(validation_rows, "path_feedback_validation_source"),
            "frontier_candidate_source_counts": _count_by_field(formal_candidates, "frontier_candidate_source"),
        })
    if config["candidate_refresh_mode"] != "dynamic_validated_only":
        batch_hash = candidate_set_hash(candidates)
        return finalize({
            "candidates": candidates,
            "dynamic_proposals": [],
            "dynamic_validation_rows": [],
            "candidate_generation_source": "static_from_source",
            "candidate_set_id": f"{scenario_id}:step-{step_index}:static:{batch_hash[:16]}",
            "candidate_set_hash": batch_hash,
            "dynamic_generation_executed": False,
            "dynamic_proposal_count": 0,
            "dynamic_validated_candidate_count": 0,
            "dynamic_validation_cache_hit": False,
            "path_feedback_validation_source_counts": {},
            "frontier_candidate_source_counts": _count_by_field(candidates, "frontier_candidate_source"),
        })
    refreshed: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate = dict(candidate)
        validated_candidate = _validated_dynamic_candidate_for_step(candidate, step_index)
        if validated_candidate is None:
            candidate["dynamic_candidate_validation_missing"] = True
            refreshed.append(candidate)
            continue
        cell = _cell_tuple(validated_candidate["cell"])
        if cell is None:
            candidate["dynamic_candidate_validation_missing"] = True
            refreshed.append(candidate)
            continue
        candidate["base_cell"] = _candidate_cell(candidate)
        candidate["cell"] = [cell[0], cell[1]]
        candidate["reachable"] = bool(validated_candidate["reachable"])
        candidate["path_cost"] = float(validated_candidate["path_cost"])
        candidate["risk"] = float(validated_candidate["risk"])
        candidate["open_grid_fallback_used"] = bool(validated_candidate["open_grid_fallback_used"])
        candidate["proposal_validated_by_path_feedback"] = True
        candidate["dynamic_candidate_generated"] = True
        candidate["dynamic_candidate_validation_missing"] = False
        if _candidate_is_valid(candidate):
            candidate_cells = _coverage_cells(
                start=current_cell,
                end=cell,
                radius=int(config["coverage_radius_cells"]),
                mode=str(config["coverage_metric_mode"]),
            )
            new_count = len(candidate_cells - covered_cells)
            candidate["expected_new_coverage_area"] = float(new_count)
            candidate["expected_coverage_rate_delta"] = float(new_count) / float(config["coverage_denominator_cells"])
            candidate["coverage_overlap_count"] = len(candidate_cells & covered_cells)
            candidate["coverage_overlap_ratio"] = _safe_ratio(len(candidate_cells & covered_cells), len(candidate_cells)) or 0.0
        refreshed.append(candidate)
    batch_hash = candidate_set_hash(refreshed)
    return finalize({
        "candidates": refreshed,
        "dynamic_proposals": [],
        "dynamic_validation_rows": [],
        "candidate_generation_source": "dynamic_validated_only",
        "candidate_set_id": f"{scenario_id}:step-{step_index}:dynamic_validated_only:{batch_hash[:16]}",
        "candidate_set_hash": batch_hash,
        "dynamic_generation_executed": False,
        "dynamic_proposal_count": 0,
        "dynamic_validated_candidate_count": sum(1 for candidate in refreshed if candidate.get("dynamic_candidate_generated") is True),
        "dynamic_validation_cache_hit": False,
        "path_feedback_validation_source_counts": {},
        "frontier_candidate_source_counts": _count_by_field(refreshed, "frontier_candidate_source"),
    })


def _apply_synthetic_credit_feature_exposure_to_adapter(
    adapter: dict[str, Any],
    *,
    candidates: list[dict[str, Any]],
    current_cell: tuple[int, int],
    current_theta_deg: float,
    covered_cells: set[tuple[int, int]],
    candidate_set_hash_value: str,
    config: dict[str, Any],
    slice_row: dict[str, Any],
    obstacle_source_linkage: dict[str, Any] | None,
    repo_root: Path,
    hybrid_astar_executor: Any | None = None,
) -> dict[str, Any]:
    if not bool(config.get("synthetic_credit_feature_exposure_enabled", False)):
        return {"synthetic_credit_feature_exposure_enabled": False}
    xunce_batch = adapter.get("xunce_batch")
    if not isinstance(xunce_batch, dict) or "candidate_features" not in xunce_batch:
        return {
            "synthetic_credit_feature_exposure_enabled": True,
            "candidate_feature_signal_missing": True,
            "missing_reason": "xunce_batch_missing_candidate_features",
        }
    candidate_count = len(candidates)
    feature_candidates = _synthetic_credit_feature_probe_candidates(
        candidates,
        current_theta_deg=current_theta_deg,
        candidate_set_hash_value=candidate_set_hash_value,
        config=config,
        current_cell=current_cell,
        slice_row=slice_row,
        repo_root=repo_root,
        hybrid_astar_executor=hybrid_astar_executor,
    )
    hybrid_costs = [candidate.get("hybrid_astar_path_cost") for candidate in feature_candidates]
    hybrid_reachable = [candidate.get("hybrid_astar_reachable") is True for candidate in feature_candidates]
    coverage_counts: list[int | None] = []
    coverage_gain_per_cost: list[float | None] = []
    for candidate in feature_candidates:
        cell = _cell_tuple(_candidate_cell(candidate))
        if cell is None:
            coverage_counts.append(None)
            coverage_gain_per_cost.append(None)
            continue
        visible = _candidate_coverage_cells(
            start=current_cell,
            end=cell,
            candidate=candidate,
            config=config,
            obstacle_source_linkage=obstacle_source_linkage,
        )
        new_count = len(visible - covered_cells)
        coverage_counts.append(new_count)
        hybrid_cost = _finite_or_none(candidate.get("hybrid_astar_path_cost"))
        fallback_cost = _candidate_cost(candidate)
        denominator = hybrid_cost if hybrid_cost is not None and hybrid_cost > 0.0 else fallback_cost
        coverage_gain_per_cost.append(_safe_ratio(float(new_count), denominator))
    los_counts, hard_counts, pressure_source = _synthetic_pressure_counts_for_candidates(
        feature_candidates,
        config=config,
        slice_row=slice_row,
        repo_root=repo_root,
    )
    feature_rows = build_synthetic_credit_feature_rows(
        candidate_count=candidate_count,
        relative_distances=_relative_distances_from_current_cell(candidates, current_cell),
        hybrid_reachable_flags=hybrid_reachable,
        obstacle_aware_new_visible_cell_counts=coverage_counts,
        obstacle_aware_gain_per_hybrid_costs=coverage_gain_per_cost,
        hybrid_astar_path_costs=hybrid_costs,
        synthetic_los_blocker_candidate_counts=los_counts,
        synthetic_hard_obstacle_candidate_counts=hard_counts,
        risk_values=[_finite_or_none(candidate.get("risk")) for candidate in candidates],
    )
    adapter["xunce_batch"] = apply_feature_rows_to_xunce_batch(xunce_batch, feature_rows)
    semantic_map = feature_semantic_map()
    semantic_map["feature_contract_id"] = "synthetic_credit_candidate_features/v1"
    semantic_map["source"] = "high_fidelity_synthetic_credit_feature_exposure/v1"
    semantic_map["synthetic_pressure_source"] = pressure_source
    metadata = {
        "synthetic_credit_feature_exposure_enabled": True,
        "xunce_batch_feature_semantic_map": semantic_map,
        "synthetic_credit_feature_rows": feature_rows,
        "synthetic_los_blocker_candidate_counts": los_counts,
        "synthetic_hard_obstacle_candidate_counts": hard_counts,
        "candidate_feature_signal_missing": False,
    }
    adapter["xunce_batch_feature_semantic_map"] = semantic_map
    adapter["synthetic_credit_feature_rows"] = feature_rows
    return metadata


def _synthetic_credit_feature_probe_candidates(
    candidates: list[dict[str, Any]],
    *,
    current_theta_deg: float,
    candidate_set_hash_value: str,
    config: dict[str, Any],
    current_cell: tuple[int, int],
    slice_row: dict[str, Any],
    repo_root: Path,
    hybrid_astar_executor: Any | None,
) -> list[dict[str, Any]]:
    probes = [dict(candidate) for candidate in candidates]
    if continuous_theta_enabled(config):
        for index, candidate in enumerate(probes):
            cell = _cell_tuple(_candidate_cell(candidate))
            if cell is None:
                continue
            candidate["candidate_theta_deg"] = float(current_theta_deg)
            candidate["candidate_viewpoint"] = [int(cell[0]), int(cell[1]), float(current_theta_deg)]
            candidate["candidate_index"] = int(index)
            candidate["candidate_set_hash"] = candidate_set_hash_value
            candidate["continuous_theta_feature_probe"] = "current_heading_probe/v1"
    if bool(config.get("hybrid_astar_pose_path_cost_enabled", False)) and not _candidates_have_hybrid_path_cost(probes):
        _enrich_candidates_with_hybrid_astar_path_cost(
            probes,
            current_cell=current_cell,
            current_theta_deg=current_theta_deg,
            candidate_set_hash_value=candidate_set_hash_value,
            step_index=0,
            scenario_id="synthetic-credit-feature-probe",
            config=config,
            slice_row=slice_row,
            repo_root=repo_root,
            hybrid_astar_executor=hybrid_astar_executor,
        )
    return probes


def _candidates_have_hybrid_path_cost(candidates: list[dict[str, Any]]) -> bool:
    return bool(candidates) and all(candidate.get("hybrid_astar_path_cost") is not None for candidate in candidates)


def _relative_distances_from_current_cell(
    candidates: list[dict[str, Any]],
    current_cell: tuple[int, int],
) -> list[float | None]:
    values: list[float | None] = []
    for candidate in candidates:
        cell = _cell_tuple(_candidate_cell(candidate))
        if cell is None:
            values.append(None)
            continue
        values.append(math.hypot(float(cell[0] - current_cell[0]), float(cell[1] - current_cell[1])))
    return values


def _synthetic_pressure_counts_for_candidates(
    candidates: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    slice_row: dict[str, Any],
    repo_root: Path,
) -> tuple[list[float | None], list[float | None], str]:
    explicit_los = [candidate.get("synthetic_los_blocker_candidate_count") for candidate in candidates]
    explicit_hard = [candidate.get("synthetic_hard_obstacle_candidate_count") for candidate in candidates]
    if any(_finite_or_none(value) is not None for value in explicit_los + explicit_hard):
        return (
            candidate_pressure_values(explicit_los, len(candidates)),
            candidate_pressure_values(explicit_hard, len(candidates)),
            "candidate_explicit_fields/v1",
        )
    sidecar = _read_sidecar_payload(slice_row, repo_root)
    los_cells = _cell_set_from_payload(sidecar.get("synthetic_los_blocker_cells"))
    hard_cells = _cell_set_from_payload(sidecar.get("synthetic_hard_obstacle_cells"))
    if not los_cells and not hard_cells:
        return ([0.0 for _ in candidates], [0.0 for _ in candidates], "missing_candidate_pressure_fields")
    los_counts: list[float | None] = []
    hard_counts: list[float | None] = []
    for candidate in candidates:
        footprint = _synthetic_pressure_footprint(candidate, config)
        los_counts.append(float(len(footprint & los_cells)))
        hard_counts.append(float(len(footprint & hard_cells)))
    return los_counts, hard_counts, "sidecar_candidate_footprint_intersection/v1"


def _synthetic_pressure_footprint(candidate: dict[str, Any], config: dict[str, Any]) -> set[tuple[int, int]]:
    cell = _cell_tuple(_candidate_cell(candidate))
    if cell is None:
        return set()
    theta_deg = _finite_or_none(candidate.get("candidate_theta_deg"))
    if theta_deg is not None:
        return visible_cells_for_viewpoint(
            cell,
            theta_deg=float(theta_deg),
            sensor_range_cells=int(config.get("sensor_range_cells", config.get("coverage_radius_cells", 1))),
            sensor_fov_deg=float(config.get("sensor_fov_deg", 90.0)),
        )
    return _footprint(cell, radius=int(config.get("coverage_radius_cells", 1)))


def _cell_set_from_payload(value: Any) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if not isinstance(value, list):
        return cells
    for item in value:
        cell = _cell_tuple(item)
        if cell is not None:
            cells.add(cell)
    return cells


def _bind_continuous_theta_to_selected_candidate(
    candidate: dict[str, Any],
    *,
    selected_index: int,
    detail_payload: dict[str, Any],
    candidate_set_hash_value: str,
) -> None:
    cell = _cell_tuple(_candidate_cell(candidate))
    theta_rad = _finite_or_none(detail_payload.get("selected_theta_rad"))
    if cell is None or theta_rad is None:
        return
    theta_rad = float(normalize_theta_rad(float(theta_rad)))
    theta_deg = math.degrees(theta_rad)
    candidate["action_space_type"] = CONTINUOUS_THETA_ACTION_SPACE
    candidate["base_candidate_index"] = int(selected_index)
    candidate["selected_base_candidate_index"] = int(selected_index)
    candidate["candidate_theta_rad"] = theta_rad
    candidate["selected_theta_rad"] = theta_rad
    candidate["candidate_theta_deg"] = theta_deg
    candidate["selected_theta_deg"] = theta_deg
    candidate["candidate_viewpoint"] = [cell[0], cell[1], theta_deg]
    candidate["base_candidate_set_hash"] = str(candidate_set_hash_value)
    candidate["continuous_theta_eval_policy"] = "deterministic_theta_mu_for_selected_base_candidate/v1"


def _bind_continuous_theta_mu_to_candidate_set(
    candidates: list[dict[str, Any]],
    *,
    detail_payload: dict[str, Any],
    candidate_set_hash_value: str,
) -> None:
    theta_values = detail_payload.get("theta_mu_rad")
    if not isinstance(theta_values, list):
        return
    for index, candidate in enumerate(candidates):
        if index >= len(theta_values):
            continue
        theta_rad = _finite_or_none(theta_values[index])
        cell = _cell_tuple(_candidate_cell(candidate))
        if theta_rad is None or cell is None:
            continue
        theta_rad = float(normalize_theta_rad(float(theta_rad)))
        theta_deg = math.degrees(theta_rad)
        candidate["action_space_type"] = CONTINUOUS_THETA_ACTION_SPACE
        candidate["base_candidate_index"] = int(index)
        candidate["candidate_theta_rad"] = theta_rad
        candidate["candidate_theta_deg"] = theta_deg
        candidate["candidate_viewpoint"] = [cell[0], cell[1], theta_deg]
        candidate["base_candidate_set_hash"] = str(candidate_set_hash_value)
        candidate["continuous_theta_eval_policy"] = "deterministic_theta_mu_per_base_candidate/v1"


def _apply_continuous_theta_hybrid_reachable_eval_policy(
    detail_payload: dict[str, Any],
    *,
    candidates: list[dict[str, Any]],
    action_mask: Any,
    current_cell: tuple[int, int] | None = None,
    current_theta_deg: float | None = None,
    candidate_set_hash_value: str | None = None,
    step_index: int | None = None,
    scenario_id: str | None = None,
    config: dict[str, Any] | None = None,
    slice_row: dict[str, Any] | None = None,
    repo_root: Path | None = None,
    hybrid_astar_executor: Any | None = None,
) -> None:
    logits = detail_payload.get("logits") or detail_payload.get("masked_logits")
    if not isinstance(logits, list) or not logits:
        return
    usable_count = min(len(logits), len(candidates))
    mask_values = list(action_mask) if isinstance(action_mask, (list, tuple)) else []
    reachable_mask: list[bool] = []
    for index in range(usable_count):
        action_allowed = bool(mask_values[index]) if index < len(mask_values) else True
        reachable_mask.append(action_allowed and candidates[index].get("hybrid_astar_reachable") is True)
    if not any(reachable_mask):
        proposal = _continuous_theta_reachable_eval_proposal(
            detail_payload,
            candidates=candidates,
            action_mask=action_mask,
            current_cell=current_cell,
            current_theta_deg=current_theta_deg,
            candidate_set_hash_value=candidate_set_hash_value,
            step_index=step_index,
            scenario_id=scenario_id,
            config=config,
            slice_row=slice_row,
            repo_root=repo_root,
            hybrid_astar_executor=hybrid_astar_executor,
        )
        if proposal is None:
            return
        selected_index = int(proposal["selected_action_index"])
        reachable_mask = [
            bool(mask_values[index]) if index < len(mask_values) else True
            for index in range(usable_count)
        ]
        for index in range(usable_count):
            reachable_mask[index] = bool(reachable_mask[index] and index == selected_index)
    masked_logits = [
        float(logits[index]) if index < usable_count and reachable_mask[index] else -1.0e9
        for index in range(len(logits))
    ]
    finite_logits = [value for value in masked_logits if value > -1.0e8]
    if not finite_logits:
        return
    max_logit = max(finite_logits)
    exp_values = [math.exp(value - max_logit) if value > -1.0e8 else 0.0 for value in masked_logits]
    denom = sum(exp_values)
    if denom <= 0.0:
        return
    probs = [value / denom for value in exp_values]
    selected_index = max(range(len(masked_logits)), key=lambda idx: masked_logits[idx])
    selected_probability = probs[selected_index]
    selected_rank = 1 + sum(1 for value in probs if value > selected_probability)
    detail_payload["masked_logits"] = masked_logits
    detail_payload["action_probs"] = probs
    detail_payload["selected_action_index"] = int(selected_index)
    detail_payload["selected_probability"] = float(selected_probability)
    detail_payload["selected_rank"] = int(selected_rank)
    theta_values = detail_payload.get("theta_mu_rad")
    if isinstance(theta_values, list) and selected_index < len(theta_values):
        theta_rad = _finite_or_none(
            candidates[selected_index].get("candidate_theta_rad")
            if candidates[selected_index].get("continuous_theta_eval_policy") == "hybrid_astar_reachable_theta_proposal_argmax/v1"
            else theta_values[selected_index]
        )
        if theta_rad is not None:
            theta_rad = float(normalize_theta_rad(float(theta_rad)))
            detail_payload["selected_theta_rad"] = theta_rad
            detail_payload["selected_theta_deg"] = math.degrees(theta_rad)
    detail_payload["continuous_theta_eval_policy"] = str(
        candidates[selected_index].get("continuous_theta_eval_policy")
        or "hybrid_astar_reachable_theta_mu_argmax/v1"
    )


def _continuous_theta_reachable_eval_proposal(
    detail_payload: dict[str, Any],
    *,
    candidates: list[dict[str, Any]],
    action_mask: Any,
    current_cell: tuple[int, int] | None,
    current_theta_deg: float | None,
    candidate_set_hash_value: str | None,
    step_index: int | None,
    scenario_id: str | None,
    config: dict[str, Any] | None,
    slice_row: dict[str, Any] | None,
    repo_root: Path | None,
    hybrid_astar_executor: Any | None,
) -> dict[str, Any] | None:
    if current_cell is None or current_theta_deg is None or config is None or slice_row is None or repo_root is None:
        return None
    logits = detail_payload.get("logits") or detail_payload.get("masked_logits")
    theta_values = detail_payload.get("theta_mu_rad")
    if not isinstance(logits, list) or not isinstance(theta_values, list):
        return None
    theta_step_deg = float(config.get("theta_step_deg", 45.0) or 45.0)
    flat_probes: list[dict[str, Any]] = []
    candidate_probe_indices: list[list[int]] = []
    for index, candidate in enumerate(candidates):
        cell = _cell_tuple(_candidate_cell(candidate))
        if cell is None:
            candidate_probe_indices.append([])
            continue
        policy_theta_rad = _finite_or_none(theta_values[index]) if index < len(theta_values) else None
        policy_theta_deg = math.degrees(float(policy_theta_rad)) if policy_theta_rad is not None else None
        proposals = _unique_continuous_theta_eval_proposals(
            [
                policy_theta_deg,
                current_theta_deg,
                float(current_theta_deg) + theta_step_deg,
                float(current_theta_deg) - theta_step_deg,
            ]
        )
        local_indices: list[int] = []
        for theta_deg in proposals:
            probe = dict(candidate)
            probe["candidate_index"] = int(index)
            probe["candidate_set_hash"] = str(candidate_set_hash_value or "")
            probe["candidate_theta_deg"] = float(theta_deg)
            probe["candidate_theta_rad"] = float(normalize_theta_rad(math.radians(float(theta_deg))))
            probe["candidate_viewpoint"] = [int(cell[0]), int(cell[1]), float(theta_deg)]
            local_indices.append(len(flat_probes))
            flat_probes.append(probe)
        candidate_probe_indices.append(local_indices)
    if not flat_probes:
        return None
    _enrich_candidates_with_hybrid_astar_path_cost(
        flat_probes,
        current_cell=current_cell,
        current_theta_deg=float(current_theta_deg),
        candidate_set_hash_value=str(candidate_set_hash_value or ""),
        step_index=int(step_index or 0),
        scenario_id=str(scenario_id or "continuous-theta-eval-proposal"),
        config=config,
        slice_row=slice_row,
        repo_root=repo_root,
        hybrid_astar_executor=hybrid_astar_executor,
    )
    mask_values = list(action_mask) if isinstance(action_mask, (list, tuple)) else []
    usable_count = min(len(logits), len(candidates))
    best_score: tuple[float, float, int, int] | None = None
    best_selection: tuple[int, int] | None = None
    for candidate_index, probe_indices in enumerate(candidate_probe_indices):
        if candidate_index >= usable_count:
            continue
        action_allowed = bool(mask_values[candidate_index]) if candidate_index < len(mask_values) else True
        if not action_allowed:
            continue
        reachable_probes = [
            probe_index
            for probe_index in probe_indices
            if flat_probes[probe_index].get("hybrid_astar_reachable") is True
            and _finite_or_none(flat_probes[probe_index].get("hybrid_astar_path_cost")) is not None
            and str(flat_probes[probe_index].get("hybrid_astar_pose_path_hash") or "").strip()
        ]
        if not reachable_probes:
            continue
        best_probe = min(
            reachable_probes,
            key=lambda probe_index: (
                float(flat_probes[probe_index].get("hybrid_astar_path_cost")),
                probe_index,
            ),
        )
        logit = float(logits[candidate_index]) if candidate_index < len(logits) else -1.0e9
        path_cost = float(flat_probes[best_probe].get("hybrid_astar_path_cost"))
        score = (logit, -path_cost, -int(candidate_index), -int(best_probe))
        if best_score is None or score > best_score:
            best_score = score
            best_selection = (int(candidate_index), int(best_probe))
    if best_selection is None:
        return None
    selected_index, selected_probe_index = best_selection
    selected_probe = flat_probes[selected_probe_index]
    _copy_continuous_theta_eval_probe_to_candidate(
        candidates[selected_index],
        selected_probe,
        selected_index=selected_index,
        candidate_set_hash_value=str(candidate_set_hash_value or ""),
    )
    return {"selected_action_index": selected_index}


def _copy_continuous_theta_eval_probe_to_candidate(
    candidate: dict[str, Any],
    probe: dict[str, Any],
    *,
    selected_index: int,
    candidate_set_hash_value: str,
) -> None:
    for field in (
        "candidate_theta_deg",
        "candidate_theta_rad",
        "candidate_viewpoint",
        "path_cost_source",
        "hybrid_astar_path_cost",
        "hybrid_astar_pose_path_hash",
        "hybrid_astar_trajectory_kind",
        "hybrid_astar_reachable",
        "hybrid_astar_failure_reason",
        "legacy_grid_astar_path_cost",
        "hybrid_vs_grid_path_cost_delta",
        "point_grid_path_cost_fallback_used",
        "default_astar_replaced",
        "hybrid_astar_ackermann_feasible_claimed",
        "platform_contract_hash",
        "max_traversable_slope_deg",
        "hybrid_astar_current_pose",
        "hybrid_astar_current_pose_provenance",
        "path_cost",
        "reachable",
        "hybrid_astar_action_mask_allowed",
    ):
        if field in probe:
            candidate[field] = probe[field]
    candidate["action_space_type"] = CONTINUOUS_THETA_ACTION_SPACE
    candidate["base_candidate_index"] = int(selected_index)
    candidate["selected_base_candidate_index"] = int(selected_index)
    candidate["selected_theta_rad"] = candidate.get("candidate_theta_rad")
    candidate["selected_theta_deg"] = candidate.get("candidate_theta_deg")
    candidate["base_candidate_set_hash"] = str(candidate_set_hash_value)
    candidate["continuous_theta_eval_policy"] = "hybrid_astar_reachable_theta_proposal_argmax/v1"


def _unique_continuous_theta_eval_proposals(values: list[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        number = _finite_or_none(value)
        if number is None:
            continue
        normalized = float(number) % 360.0
        if not any(abs(((normalized - existing + 180.0) % 360.0) - 180.0) <= 1.0e-6 for existing in result):
            result.append(normalized)
    return result


def _evaluate_hybrid_astar_candidate_path_cost_worker(args: tuple[Any, ...]) -> tuple[int, dict[str, Any] | None, str | None]:
    (
        index,
        grid,
        current_pose,
        payload,
        platform_hash,
        max_slope,
        planner_options,
    ) = args
    try:
        row = evaluate_hybrid_astar_candidate_path_cost(
            grid=grid,
            current_pose=current_pose,
            candidate=payload,
            platform_contract_hash=platform_hash,
            max_traversable_slope_deg=float(max_slope),
            theta_bin_count=int(planner_options["theta_bin_count"]),
            goal_position_tolerance_m=planner_options["goal_position_tolerance_m"],
            goal_theta_tolerance_deg=float(planner_options["goal_theta_tolerance_deg"]),
            max_iterations=int(planner_options["max_iterations"]),
            primitive_duration_s=float(planner_options["primitive_duration_s"]),
            integration_dt_s=float(planner_options["integration_dt_s"]),
            max_speed_mps=float(planner_options["max_speed_mps"]),
            max_angular_speed_degps=float(planner_options["max_angular_speed_degps"]),
            rotation_cost_weight=float(planner_options["rotation_cost_weight"]),
            reverse_penalty_weight=float(planner_options["reverse_penalty_weight"]),
            turn_penalty_weight=float(planner_options["turn_penalty_weight"]),
        )
        return int(index), row, None
    except Exception as exc:  # pragma: no cover - worker failures are represented per candidate.
        return int(index), None, f"{type(exc).__name__}:{exc}"


def _enrich_candidates_with_hybrid_astar_path_cost(
    candidates: list[dict[str, Any]],
    *,
    current_cell: tuple[int, int],
    current_theta_deg: float,
    candidate_set_hash_value: str,
    step_index: int,
    scenario_id: str,
    config: dict[str, Any],
    slice_row: dict[str, Any],
    repo_root: Path,
    hybrid_astar_executor: Any | None = None,
) -> None:
    if not candidates:
        return
    sidecar_path = _resolved_file(slice_row.get("sidecar"), repo_root)
    platform_hash = str(config.get("platform_contract_hash") or slice_row.get("platform_contract_hash") or "")
    max_slope = _finite_or_none(config.get("max_traversable_slope_deg"))
    if max_slope is None:
        max_slope = 30.0
    if sidecar_path is None:
        for candidate in candidates:
            _apply_hybrid_astar_unavailable_candidate_fields(
                candidate,
                reason="sidecar_missing",
                platform_contract_hash=platform_hash,
                max_traversable_slope_deg=float(max_slope),
            )
        return
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        grid = build_cost_grid_from_sidecar(sidecar)
        current_world = _hybrid_cell_center_world(grid.spec, HYBRID_CELL(int(current_cell[0]), int(current_cell[1])))
    except Exception:
        for candidate in candidates:
            _apply_hybrid_astar_unavailable_candidate_fields(
                candidate,
                reason="sidecar_decode_or_grid_build_failed",
                platform_contract_hash=platform_hash,
                max_traversable_slope_deg=float(max_slope),
            )
        return

    platform_hash = str(platform_hash or sidecar.get("platform_contract_hash") or "")
    sidecar_max_slope = _finite_or_none(sidecar.get("max_traversable_slope_deg"))
    if sidecar_max_slope is not None:
        max_slope = sidecar_max_slope
    current_pose = [float(current_world.x), float(current_world.y), math.radians(float(current_theta_deg))]
    planner_options = {
        "theta_bin_count": int(config.get("hybrid_astar_theta_bin_count", 72)),
        "goal_position_tolerance_m": (
            float(config["hybrid_astar_goal_position_tolerance_m"])
            if config.get("hybrid_astar_goal_position_tolerance_m") is not None
            else None
        ),
        "goal_theta_tolerance_deg": float(config.get("hybrid_astar_goal_theta_tolerance_deg", 5.0)),
        "max_iterations": int(config.get("hybrid_astar_max_iterations", 100_000)),
        "primitive_duration_s": float(config.get("hybrid_astar_primitive_duration_s", 1.0)),
        "integration_dt_s": float(config.get("hybrid_astar_integration_dt_s", 0.25)),
        "max_speed_mps": float(config.get("hybrid_astar_max_speed_mps", 1.0)),
        "max_angular_speed_degps": float(config.get("hybrid_astar_max_angular_speed_degps", 45.0)),
        "rotation_cost_weight": float(config.get("hybrid_astar_rotation_cost_weight", 0.2)),
        "reverse_penalty_weight": float(config.get("hybrid_astar_reverse_penalty_weight", 0.5)),
        "turn_penalty_weight": float(config.get("hybrid_astar_turn_penalty_weight", 0.05)),
    }

    def payload_for(index: int, candidate: dict[str, Any]) -> dict[str, Any]:
        payload = dict(candidate)
        payload.setdefault("scenario_id", scenario_id)
        payload.setdefault("step_index", step_index)
        payload.setdefault("candidate_index", index)
        payload.setdefault("candidate_set_hash", candidate_set_hash_value)
        return payload

    rows_by_index: dict[int, dict[str, Any]] = {}
    failed_indices: set[int] = set()
    if hybrid_astar_executor is not None and len(candidates) > 1:
        futures = {
            hybrid_astar_executor.submit(
                _evaluate_hybrid_astar_candidate_path_cost_worker,
                (
                    index,
                    grid,
                    current_pose,
                    payload_for(index, candidate),
                    platform_hash,
                    float(max_slope),
                    planner_options,
                ),
            ): index
            for index, candidate in enumerate(candidates)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                row_index, row, error = future.result()
            except Exception:
                failed_indices.add(index)
                continue
            if error is not None or row is None:
                failed_indices.add(int(row_index))
            else:
                rows_by_index[int(row_index)] = row
    else:
        for index, candidate in enumerate(candidates):
            payload = payload_for(index, candidate)
            try:
                rows_by_index[index] = evaluate_hybrid_astar_candidate_path_cost(
                    grid=grid,
                    current_pose=current_pose,
                    candidate=payload,
                    platform_contract_hash=platform_hash,
                    max_traversable_slope_deg=float(max_slope),
                    theta_bin_count=int(planner_options["theta_bin_count"]),
                    goal_position_tolerance_m=planner_options["goal_position_tolerance_m"],
                    goal_theta_tolerance_deg=float(planner_options["goal_theta_tolerance_deg"]),
                    max_iterations=int(planner_options["max_iterations"]),
                    primitive_duration_s=float(planner_options["primitive_duration_s"]),
                    integration_dt_s=float(planner_options["integration_dt_s"]),
                    max_speed_mps=float(planner_options["max_speed_mps"]),
                    max_angular_speed_degps=float(planner_options["max_angular_speed_degps"]),
                    rotation_cost_weight=float(planner_options["rotation_cost_weight"]),
                    reverse_penalty_weight=float(planner_options["reverse_penalty_weight"]),
                    turn_penalty_weight=float(planner_options["turn_penalty_weight"]),
                )
            except Exception:
                failed_indices.add(index)

    for index, candidate in enumerate(candidates):
        row = rows_by_index.get(index)
        if row is None or index in failed_indices:
            _apply_hybrid_astar_unavailable_candidate_fields(
                candidate,
                reason="hybrid_astar_evaluation_failed",
                platform_contract_hash=platform_hash,
                max_traversable_slope_deg=float(max_slope),
            )
            continue
        _apply_hybrid_astar_candidate_fields(
            candidate,
            row=row,
            current_pose=current_pose,
            platform_contract_hash=platform_hash,
            max_traversable_slope_deg=float(max_slope),
        )


def _apply_hybrid_astar_candidate_fields(
    candidate: dict[str, Any],
    *,
    row: dict[str, Any],
    current_pose: list[float],
    platform_contract_hash: str,
    max_traversable_slope_deg: float,
) -> None:
    legacy_grid_cost = row.get("legacy_grid_astar_path_cost")
    hybrid_cost = row.get("hybrid_astar_path_cost")
    candidate["path_cost_source"] = HYBRID_ASTAR_PATH_COST_SOURCE
    candidate["hybrid_astar_path_cost"] = hybrid_cost
    candidate["hybrid_astar_pose_path_hash"] = row.get("hybrid_astar_pose_path_hash")
    candidate["hybrid_astar_trajectory_kind"] = row.get("hybrid_astar_trajectory_kind")
    candidate["hybrid_astar_reachable"] = row.get("hybrid_astar_reachable") is True
    candidate["hybrid_astar_failure_reason"] = row.get("hybrid_astar_failure_reason")
    candidate["legacy_grid_astar_path_cost"] = legacy_grid_cost
    candidate["hybrid_vs_grid_path_cost_delta"] = row.get("hybrid_vs_grid_path_cost_delta")
    candidate["point_grid_path_cost_fallback_used"] = False
    candidate["default_astar_replaced"] = False
    candidate["hybrid_astar_ackermann_feasible_claimed"] = False
    candidate["platform_contract_hash"] = platform_contract_hash
    candidate["max_traversable_slope_deg"] = max_traversable_slope_deg
    candidate["hybrid_astar_current_pose"] = list(current_pose)
    candidate["hybrid_astar_current_pose_provenance"] = "high_fidelity_current_cell_plus_previous_selected_theta/v1"
    if _finite_or_none(hybrid_cost) is not None:
        candidate["path_cost"] = float(hybrid_cost)
        candidate["reachable"] = candidate["hybrid_astar_reachable"]
    else:
        candidate["reachable"] = False
    candidate["hybrid_astar_action_mask_allowed"] = candidate["hybrid_astar_reachable"]


def _apply_hybrid_astar_unavailable_candidate_fields(
    candidate: dict[str, Any],
    *,
    reason: str,
    platform_contract_hash: str,
    max_traversable_slope_deg: float,
) -> None:
    candidate["path_cost_source"] = HYBRID_ASTAR_PATH_COST_SOURCE
    candidate["hybrid_astar_path_cost"] = None
    candidate["hybrid_astar_pose_path_hash"] = None
    candidate["hybrid_astar_trajectory_kind"] = "hybrid_astar_pose_path"
    candidate["hybrid_astar_reachable"] = False
    candidate["hybrid_astar_failure_reason"] = reason
    candidate["legacy_grid_astar_path_cost"] = _candidate_cost(candidate)
    candidate["hybrid_vs_grid_path_cost_delta"] = None
    candidate["point_grid_path_cost_fallback_used"] = False
    candidate["default_astar_replaced"] = False
    candidate["hybrid_astar_ackermann_feasible_claimed"] = False
    candidate["platform_contract_hash"] = platform_contract_hash
    candidate["max_traversable_slope_deg"] = max_traversable_slope_deg
    candidate["hybrid_astar_action_mask_allowed"] = False
    candidate["reachable"] = False


def _hybrid_cell_center_world(spec: Any, cell: HYBRID_CELL) -> HYBRID_WORLD_POINT:
    origin = getattr(spec, "origin", (0.0, 0.0))
    return HYBRID_WORLD_POINT(
        float(origin[0]) + (float(cell.x) + 0.5) * float(spec.resolution),
        float(origin[1]) + (float(cell.y) + 0.5) * float(spec.resolution),
    )


def _validated_dynamic_candidate_for_step(candidate: dict[str, Any], step_index: int) -> dict[str, Any] | None:
    dynamic_candidates = candidate.get("dynamic_validated_candidates")
    if not isinstance(dynamic_candidates, list):
        return None
    if step_index < len(dynamic_candidates):
        row = dynamic_candidates[step_index]
        if _dynamic_candidate_has_validation(row):
            return row
    for row in dynamic_candidates:
        if _dynamic_candidate_matches_step(row, step_index) and _dynamic_candidate_has_validation(row):
            return row
    return None


def _dynamic_candidate_matches_step(row: Any, step_index: int) -> bool:
    if not isinstance(row, dict):
        return False
    raw_step = row.get("step_index")
    if raw_step is None:
        return False
    try:
        return int(raw_step) == step_index
    except (TypeError, ValueError):
        return False


def _dynamic_candidate_has_validation(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    if row.get("proposal_validated_by_path_feedback") is not True:
        return False
    if _cell_tuple(row.get("cell")) is None:
        return False
    if not isinstance(row.get("reachable"), bool):
        return False
    if _finite_or_none(row.get("path_cost")) is None:
        return False
    if _finite_or_none(row.get("risk")) is None:
        return False
    if not isinstance(row.get("open_grid_fallback_used"), bool):
        return False
    return True


def _dynamic_artifact_row(
    row: dict[str, Any],
    *,
    scenario_id: str,
    policy: str,
    policy_step: int,
    current_cell: tuple[int, int],
    covered_cells_hash_value: str,
    candidate_set_id: str,
    candidate_set_hash_value: str,
) -> dict[str, Any]:
    payload = dict(row)
    payload.setdefault("scenario_id", scenario_id)
    payload["policy"] = policy
    payload.setdefault("step_index", policy_step)
    payload["current_cell_before"] = list(current_cell)
    payload["covered_cells_hash"] = covered_cells_hash_value
    payload["candidate_set_id"] = candidate_set_id
    payload["candidate_set_hash"] = candidate_set_hash_value
    return payload


def _candidate_metric_audit_rows(
    candidates: list[dict[str, Any]],
    *,
    scenario_id: str,
    roi_group: str,
    split: Any,
    policy_name: str,
    step_index: int,
    current_cell: tuple[int, int],
    covered_cells: set[tuple[int, int]],
    covered_cells_hash_value: str,
    candidate_set_id: str,
    candidate_set_hash_value: str,
    action_mask: list[bool],
    config: dict[str, Any],
    obstacle_source_linkage: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        cell = _cell_tuple(_candidate_cell(candidate))
        coverage_count = _finite_or_none(candidate.get("expected_new_coverage_cell_count"))
        if coverage_count is None and cell is not None:
            footprint = _candidate_coverage_cells(
                start=current_cell,
                end=cell,
                candidate=candidate,
                config=config,
                obstacle_source_linkage=obstacle_source_linkage,
            )
            coverage_count = float(len(footprint - covered_cells))
        path_cost = _candidate_cost(candidate)
        risk = _finite_or_none(candidate.get("risk"))
        roi_gain = _finite_or_none(candidate.get("roi_weighted_coverage_delta"))
        if roi_gain is None:
            roi_gain = coverage_count
        risk_cost_weighted = None
        if path_cost is not None and risk is not None:
            risk_cost_weighted = float(path_cost) * float(risk)
        soft_risk_exposure = _finite_or_none(candidate.get("soft_risk_exposure"))
        if soft_risk_exposure is None:
            soft_risk_exposure = _finite_or_none(candidate.get("path_risk_exposure"))
        if soft_risk_exposure is None:
            soft_risk_exposure = risk_cost_weighted
        obstacle_audit = _candidate_obstacle_audit(
            candidate=candidate,
            config=config,
            cell=cell,
            obstacle_source_linkage=obstacle_source_linkage,
        )
        rows.append(
            {
                "schema_version": "xunce-exploration-coverage-candidate-metric-audit-row/v1",
                "scenario_id": scenario_id,
                "split": split,
                "roi_group": roi_group,
                "policy": policy_name,
                "executing_policy": policy_name,
                "step_index": step_index,
                "current_cell": list(current_cell),
                "covered_cells_hash": covered_cells_hash_value,
                "candidate_set_id": candidate_set_id,
                "candidate_set_hash": candidate_set_hash_value,
                "candidate_index": candidate_index,
                "candidate_cell": list(cell) if cell is not None else None,
                "candidate_theta_deg": candidate.get("candidate_theta_deg"),
                "candidate_viewpoint": candidate.get("candidate_viewpoint"),
                "base_candidate_index": candidate.get("base_candidate_index"),
                "viewpoint_index": candidate.get("viewpoint_index"),
                "base_candidate_set_hash": candidate.get("base_candidate_set_hash"),
                "sensor_model_id": candidate.get("sensor_model_id"),
                "sensor_fov_deg": candidate.get("sensor_fov_deg"),
                "sensor_range_cells": candidate.get("sensor_range_cells"),
                "theta_visible_cell_count": candidate.get("theta_visible_cell_count"),
                "theta_new_visible_cell_count": candidate.get("theta_new_visible_cell_count"),
                "theta_coverage_hash": candidate.get("theta_coverage_hash"),
                "theta_coverage_gain_per_path_cost": candidate.get("theta_coverage_gain_per_path_cost"),
                "obstacle_occlusion_enabled": obstacle_audit["obstacle_occlusion_enabled"],
                "obstacle_source": obstacle_audit["obstacle_source"],
                "obstacle_source_id": obstacle_audit["obstacle_source_id"],
                "obstacle_source_hash": obstacle_audit["obstacle_source_hash"],
                "obstacle_source_kind": obstacle_audit["obstacle_source_kind"],
                "obstacle_source_is_proxy": obstacle_audit["obstacle_source_is_proxy"],
                "obstacle_source_missing": obstacle_audit["obstacle_source_missing"],
                "obstacle_cell_count": obstacle_audit["obstacle_cell_count"],
                "obstacle_occluded_cell_count": obstacle_audit["obstacle_occluded_cell_count"],
                "obstacle_blocked_cell_count": obstacle_audit["obstacle_blocked_cell_count"],
                "obstacle_aware_theta_coverage_hash": obstacle_audit["obstacle_aware_theta_coverage_hash"],
                "theta_feature_available": bool(candidate.get("theta_feature_available", False)),
                "sin_theta": candidate.get("sin_theta"),
                "cos_theta": candidate.get("cos_theta"),
                "theta_deg_norm": candidate.get("theta_deg_norm"),
                "action_mask_valid": bool(action_mask[candidate_index]) if candidate_index < len(action_mask) else False,
                "path_cost_source": candidate.get("path_cost_source"),
                "hybrid_astar_reachable": candidate.get("hybrid_astar_reachable"),
                "hybrid_astar_action_mask_allowed": candidate.get("hybrid_astar_action_mask_allowed"),
                "hybrid_astar_path_cost": candidate.get("hybrid_astar_path_cost"),
                "hybrid_astar_pose_path_hash": candidate.get("hybrid_astar_pose_path_hash"),
                "hybrid_astar_trajectory_kind": candidate.get("hybrid_astar_trajectory_kind"),
                "hybrid_astar_failure_reason": candidate.get("hybrid_astar_failure_reason"),
                "legacy_grid_astar_path_cost": candidate.get("legacy_grid_astar_path_cost"),
                "hybrid_vs_grid_path_cost_delta": candidate.get("hybrid_vs_grid_path_cost_delta"),
                "point_grid_path_cost_fallback_used": candidate.get("point_grid_path_cost_fallback_used"),
                "default_astar_replaced": candidate.get("default_astar_replaced"),
                "hybrid_astar_ackermann_feasible_claimed": candidate.get("hybrid_astar_ackermann_feasible_claimed"),
                "hybrid_astar_current_pose": candidate.get("hybrid_astar_current_pose"),
                "hybrid_astar_current_pose_provenance": candidate.get("hybrid_astar_current_pose_provenance"),
                "platform_contract_hash": candidate.get("platform_contract_hash"),
                "max_traversable_slope_deg": candidate.get("max_traversable_slope_deg"),
                "expected_new_coverage_cell_count": coverage_count,
                "roi_weighted_coverage_delta": roi_gain,
                "path_cost": path_cost,
                "risk": risk,
                "risk_proxy": risk,
                "risk_cost_weighted": risk_cost_weighted,
                "path_allowed_by_risk": candidate.get("path_allowed_by_risk"),
                "hard_risk_flags": list(candidate.get("hard_risk_flags") or []),
                "soft_risk_flags": list(candidate.get("soft_risk_flags") or []),
                "soft_risk_exposure": soft_risk_exposure,
                "path_risk_exposure": soft_risk_exposure,
                "path_risk_peak": _finite_or_none(candidate.get("path_risk_peak")),
                "high_risk_distance_m": _finite_or_none(candidate.get("high_risk_distance_m")),
                "recovery_margin_min": _finite_or_none(candidate.get("recovery_margin_min")),
                "risk_semantics_source": str(candidate.get("risk_semantics_source") or "legacy_candidate_risk_scalar"),
                "risk_proxy_is_physical_risk": bool(candidate.get("risk_proxy_is_physical_risk", False)),
                "risk_source": str(candidate.get("risk_source") or "offline_proxy"),
                "risk_proxy_source": str(candidate.get("risk_source") or "offline_proxy"),
                "coverage_gain_per_path_cost": _safe_ratio(coverage_count or 0.0, path_cost or 0.0),
                "profile_id": config["profile_id"],
                "profile_version": config["profile_version"],
                "profile_hash": config["profile_hash"],
            }
        )
    return rows


def _candidate_hard_risk_violation(candidate: dict[str, Any]) -> bool:
    return candidate.get("path_allowed_by_risk") is False or bool(candidate.get("hard_risk_flags") or [])


def _candidate_soft_risk_exposure(candidate: dict[str, Any]) -> float:
    value = _finite_or_none(candidate.get("soft_risk_exposure"))
    if value is None:
        value = _finite_or_none(candidate.get("path_risk_exposure"))
    if value is None:
        path_cost = _candidate_cost(candidate)
        risk = _finite_or_none(candidate.get("risk"))
        if path_cost is not None and risk is not None:
            value = float(path_cost) * float(risk)
    return 0.0 if value is None else float(value)


def _selected_candidate_metrics(candidates: list[dict[str, Any]], selected_index: Any) -> dict[str, Any]:
    candidate = _candidate_at(candidates, selected_index)
    if candidate is None:
        return {}
    return {
        "selected_cell": _candidate_cell(candidate),
        "selected_viewpoint": candidate.get("candidate_viewpoint"),
        "selected_theta_deg": candidate.get("candidate_theta_deg"),
        "selected_base_candidate_index": candidate.get("base_candidate_index"),
        "selected_path_cost": _candidate_cost(candidate),
        "selected_risk": _finite_or_none(candidate.get("risk")),
        "selected_expected_new_coverage_cell_count": _finite_or_none(candidate.get("expected_new_coverage_cell_count")),
        "selected_roi_weighted_coverage_delta": _finite_or_none(candidate.get("roi_weighted_coverage_delta")),
    }


def _teacher_label_candidate_metrics(
    candidate: dict[str, Any] | None,
    *,
    candidate_index: Any,
    current_cell: tuple[int, int],
    covered_cells: set[tuple[int, int]],
    config: dict[str, Any],
) -> dict[str, Any]:
    if candidate is None:
        return {}
    cell = _cell_tuple(_candidate_cell(candidate))
    coverage_count = 0
    if cell is not None:
        coverage_count = len(
            _candidate_coverage_cells(
                start=current_cell,
                end=cell,
                candidate=candidate,
                config=config,
            )
            - covered_cells
        )
    path_cost = _candidate_cost(candidate)
    risk = _finite_or_none(candidate.get("risk"))
    risk_cost_weighted = None
    if path_cost is not None and risk is not None:
        risk_cost_weighted = float(path_cost) * float(risk)
    return {
        "candidate_index": _int_value(candidate_index),
        "candidate_cell": list(cell) if cell is not None else None,
        "new_covered_cell_count": int(coverage_count),
        "expected_new_coverage_cell_count": int(coverage_count),
        "roi_weighted_coverage_delta": _finite_or_none(candidate.get("roi_weighted_coverage_delta")),
        "path_cost": path_cost,
        "risk": risk,
        "risk_proxy": risk,
        "risk_cost_weighted": risk_cost_weighted,
        "soft_risk_exposure": _candidate_soft_risk_exposure(candidate),
        "path_risk_exposure": _candidate_soft_risk_exposure(candidate),
        "path_risk_peak": _finite_or_none(candidate.get("path_risk_peak")),
        "path_allowed_by_risk": candidate.get("path_allowed_by_risk"),
        "hard_risk_flags": list(candidate.get("hard_risk_flags") or []),
        "hard_risk_violation": _candidate_hard_risk_violation(candidate),
        "risk_source": str(candidate.get("risk_source") or candidate.get("risk_proxy_source") or "offline_proxy"),
    }


def _on_policy_oracle_teacher_label_row(
    *,
    scenario_id: str,
    roi_group: str,
    split: Any,
    policy_name: str,
    step_index: int,
    current_cell: tuple[int, int],
    covered_cells: set[tuple[int, int]],
    covered_cells_hash_value: str,
    candidate_set_id: str,
    candidate_set_hash_value: str,
    candidates: list[dict[str, Any]],
    xunce_selected_index: Any,
    true_model_inference_executed: bool,
    model_inference_failure: bool,
    detail_payload: dict[str, Any],
    coverage_denominator: float,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    if not config.get("emit_on_policy_oracle_teacher_labels"):
        return None
    if policy_name != str(config.get("on_policy_oracle_teacher_baseline_policy", "xunce")):
        return None
    if policy_name != "xunce":
        return None
    if not true_model_inference_executed or model_inference_failure or not bool(detail_payload.get("finite_outputs")):
        return None
    teacher_config = dict(config)
    teacher_config["canonical_reward_rerank_profile"] = config["on_policy_oracle_teacher_profile"]
    teacher_detail = _oracle_policy_detail(
        CANONICAL_REWARD_RERANK_ORACLE,
        candidates,
        current_cell=current_cell,
        covered_cells=covered_cells,
        coverage_denominator=coverage_denominator,
        config=teacher_config,
    )
    teacher_index = teacher_detail.get("selected_action_index")
    xunce_candidate = _candidate_at(candidates, xunce_selected_index)
    teacher_candidate = _candidate_at(candidates, teacher_index)
    xunce_metrics = _teacher_label_candidate_metrics(
        xunce_candidate,
        candidate_index=xunce_selected_index,
        current_cell=current_cell,
        covered_cells=covered_cells,
        config=config,
    )
    teacher_metrics = _teacher_label_candidate_metrics(
        teacher_candidate,
        candidate_index=teacher_index,
        current_cell=current_cell,
        covered_cells=covered_cells,
        config=config,
    )
    xunce_coverage = _float_default(xunce_metrics.get("new_covered_cell_count"))
    teacher_coverage = _float_default(teacher_metrics.get("new_covered_cell_count"))
    xunce_cost = _finite_or_none(xunce_metrics.get("path_cost"))
    teacher_cost = _finite_or_none(teacher_metrics.get("path_cost"))
    teacher_higher_coverage = teacher_coverage > xunce_coverage + TOLERANCE
    equal_coverage = abs(teacher_coverage - xunce_coverage) <= TOLERANCE
    teacher_lower_cost = (
        teacher_cost is not None
        and xunce_cost is not None
        and teacher_cost < xunce_cost - TOLERANCE
    )
    teacher_acceptable_cost = (
        teacher_cost is not None
        and xunce_cost is not None
        and teacher_cost <= max(xunce_cost, TOLERANCE) * 1.25 + TOLERANCE
    )
    if teacher_higher_coverage and teacher_acceptable_cost:
        sample_weight = 1.0
    elif equal_coverage and teacher_lower_cost:
        sample_weight = 0.5
    else:
        sample_weight = 0.0
    hard_risk_clean_pair = bool(
        xunce_candidate is not None
        and teacher_candidate is not None
        and not _candidate_hard_risk_violation(xunce_candidate)
        and not _candidate_hard_risk_violation(teacher_candidate)
    )
    return {
        "schema_version": "xunce-on-policy-oracle-teacher-label/v1",
        "scenario_id": scenario_id,
        "roi_group": roi_group,
        "split": split,
        "step_index": int(step_index),
        "baseline_policy": "xunce",
        "teacher_policy": CANONICAL_REWARD_RERANK_ORACLE,
        "same_candidate_set": True,
        "candidate_set_id": candidate_set_id,
        "candidate_set_hash": candidate_set_hash_value,
        "current_cell": list(current_cell),
        "covered_cells_hash": covered_cells_hash_value,
        "teacher_action_index": _int_value(teacher_index),
        "xunce_action_index": _int_value(xunce_selected_index),
        "teacher_profile_id": config["on_policy_oracle_teacher_profile_id"],
        "teacher_profile_version": config["on_policy_oracle_teacher_profile_version"],
        "teacher_profile_hash": config["on_policy_oracle_teacher_profile_hash"],
        "teacher_reward_components": teacher_detail.get("reward_components"),
        "teacher_reward": teacher_detail.get("reward"),
        "xunce_candidate_metrics": xunce_metrics,
        "teacher_candidate_metrics": teacher_metrics,
        "oracle_new_covered_cell_count": teacher_metrics.get("new_covered_cell_count"),
        "xunce_new_covered_cell_count": xunce_metrics.get("new_covered_cell_count"),
        "oracle_path_cost": teacher_metrics.get("path_cost"),
        "xunce_path_cost": xunce_metrics.get("path_cost"),
        "oracle_soft_risk_exposure": teacher_metrics.get("soft_risk_exposure"),
        "xunce_soft_risk_exposure": xunce_metrics.get("soft_risk_exposure"),
        "hard_risk_clean_pair": hard_risk_clean_pair,
        "teacher_selected_higher_coverage": bool(teacher_higher_coverage),
        "teacher_selected_lower_or_acceptable_cost": bool(teacher_lower_cost or teacher_acceptable_cost),
        "sample_weight": sample_weight,
        "training_signal_type": "teacher_imitation_label",
        "counterfactual_type": "xunce_on_policy_same_candidate_set_teacher_label",
        "does_not_change_xunce_action": True,
    }


def _paired_disagreement_counts(rows: list[dict[str, Any]]) -> tuple[int, int]:
    disagreement = 0
    useful = 0
    seen: set[tuple[str, int, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("scenario_id")),
            int(row.get("step_index", 0) or 0),
            str(row.get("candidate_set_hash") or ""),
            str(row.get("covered_cells_hash") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        if row.get("policy_disagreement") is not True:
            continue
        disagreement += 1
        xunce_gain = _finite_or_none(row.get("xunce_selected_expected_new_coverage_cell_count"))
        incumbent_gain = _finite_or_none(row.get("incumbent_selected_expected_new_coverage_cell_count"))
        xunce_cost = _finite_or_none(row.get("xunce_selected_path_cost"))
        incumbent_cost = _finite_or_none(row.get("incumbent_selected_path_cost"))
        xunce_risk = _finite_or_none(row.get("xunce_selected_risk"))
        incumbent_risk = _finite_or_none(row.get("incumbent_selected_risk"))
        if (
            xunce_gain is not None
            and incumbent_gain is not None
            and xunce_cost is not None
            and incumbent_cost is not None
            and xunce_risk is not None
            and incumbent_risk is not None
            and xunce_gain > incumbent_gain + TOLERANCE
            and xunce_cost <= incumbent_cost + TOLERANCE
            and xunce_risk <= incumbent_risk + TOLERANCE
        ):
            useful += 1
    return disagreement, useful


def _count_by_field(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field) or "missing")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _paired_decision_audit_row(
    *,
    scenario_id: str,
    roi_group: str,
    split: Any,
    executing_policy: str,
    step_index: int,
    current_cell: tuple[int, int],
    candidate_set_id: str,
    candidate_set_hash_value: str,
    covered_cells_hash_value: str,
    candidates: list[dict[str, Any]],
    adapter: dict[str, Any],
    model_bundle: dict[str, Any],
) -> dict[str, Any] | None:
    if _is_oracle_policy(executing_policy):
        return None
    row = {
        "schema_version": "xunce-exploration-coverage-paired-decision-audit/v1",
        "scenario_id": scenario_id,
        "roi_group": roi_group,
        "split": split,
        "executing_policy": executing_policy,
        "step_index": step_index,
        "current_cell": list(current_cell),
        "candidate_set_id": candidate_set_id,
        "candidate_set_hash": candidate_set_hash_value,
        "covered_cells_hash": covered_cells_hash_value,
        "candidate_cells": candidate_observation_cells(candidates),
        **theta_metadata(candidates),
        "action_mask": list(adapter["action_mask"]),
        "same_state_same_candidate_set": True,
        "xunce_checkpoint_loaded": bool(model_bundle.get("xunce_checkpoint_loaded")),
        "incumbent_checkpoint_loaded": bool(model_bundle.get("incumbent_checkpoint_loaded")),
        "reason_codes": [],
    }
    if not adapter.get("has_valid_action"):
        row["reason_codes"] = ["no_valid_action"]
        return row
    if not model_bundle.get("xunce_checkpoint_loaded") or not model_bundle.get("incumbent_checkpoint_loaded"):
        row["reason_codes"] = ["checkpoint_not_loaded"]
        return row
    reasons: list[str] = []
    try:
        xunce_detail = _score_xunce_model(model_bundle["xunce_model"], adapter["xunce_batch"])
        row["xunce_selected_action_index"] = _selected_index(xunce_detail)
        row["xunce_selected_probability"] = xunce_detail.get("selected_probability")
        row["xunce_selected_rank"] = xunce_detail.get("selected_rank")
        row["xunce_finite_outputs"] = xunce_detail.get("finite_outputs")
        row.update(
            {
                f"xunce_{key}": value
                for key, value in _selected_candidate_metrics_for_detail(
                    candidates,
                    row["xunce_selected_action_index"],
                    detail_payload=xunce_detail,
                    candidate_set_hash_value=candidate_set_hash_value,
                ).items()
            }
        )
    except Exception as exc:  # pragma: no cover
        reasons.append(f"xunce_shadow_scoring_failed:{type(exc).__name__}")
    try:
        incumbent_detail = _policy_detail_to_dict(model_bundle["incumbent_scorer"].score_detail(adapter["incumbent_observation"]))
        row["incumbent_selected_action_index"] = _selected_index(incumbent_detail)
        row["incumbent_selected_probability"] = incumbent_detail.get("selected_probability")
        row["incumbent_selected_rank"] = incumbent_detail.get("selected_rank")
        row["incumbent_finite_outputs"] = incumbent_detail.get("finite_outputs")
        row.update({f"incumbent_{key}": value for key, value in _selected_candidate_metrics(candidates, row["incumbent_selected_action_index"]).items()})
    except Exception as exc:  # pragma: no cover
        reasons.append(f"incumbent_shadow_scoring_failed:{type(exc).__name__}")
    for prefix in ("xunce", "incumbent"):
        for key in (
            "selected_cell",
            "selected_viewpoint",
            "selected_theta_deg",
            "selected_base_candidate_index",
            "selected_path_cost",
            "selected_risk",
            "selected_expected_new_coverage_cell_count",
            "selected_roi_weighted_coverage_delta",
        ):
            row.setdefault(f"{prefix}_{key}", None)
    row["policy_disagreement"] = (
        row.get("xunce_selected_action_index") is not None
        and row.get("incumbent_selected_action_index") is not None
        and row.get("xunce_selected_action_index") != row.get("incumbent_selected_action_index")
    )
    row["reason_codes"] = unique_sorted(reasons)
    return row


def _selected_candidate_metrics_for_detail(
    candidates: list[dict[str, Any]],
    selected_index: Any,
    *,
    detail_payload: dict[str, Any],
    candidate_set_hash_value: str,
) -> dict[str, Any]:
    selected_int = _int_or_none(selected_index)
    if selected_int is None:
        return _selected_candidate_metrics(candidates, selected_index)
    selected_theta = _finite_or_none(detail_payload.get("selected_theta_rad"))
    if selected_theta is None:
        return _selected_candidate_metrics(candidates, selected_index)
    copied = [dict(candidate) for candidate in candidates]
    if selected_int < 0 or selected_int >= len(copied):
        return _selected_candidate_metrics(copied, selected_index)
    _bind_continuous_theta_to_selected_candidate(
        copied[selected_int],
        selected_index=selected_int,
        detail_payload=detail_payload,
        candidate_set_hash_value=candidate_set_hash_value,
    )
    return _selected_candidate_metrics(copied, selected_index)


def _is_oracle_policy(policy_name: str) -> bool:
    return policy_name in ORACLE_POLICIES or policy_name == CANONICAL_REWARD_RERANK_ORACLE


def _oracle_policy_detail(
    policy_name: str,
    candidates: list[dict[str, Any]],
    *,
    current_cell: tuple[int, int],
    covered_cells: set[tuple[int, int]],
    coverage_denominator: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    scored: list[tuple[float, float, float, int, dict[str, float] | None, float | None, str | None, str | None]] = []
    rerank_profile = None
    if policy_name == CANONICAL_REWARD_RERANK_ORACLE:
        rerank_profile = load_canonical_reward_profile(Path(config["canonical_reward_rerank_profile"]))
    for index, candidate in enumerate(candidates):
        if not _candidate_is_valid(candidate):
            continue
        if policy_name == CANONICAL_REWARD_RERANK_ORACLE and _candidate_hard_risk_violation(candidate):
            continue
        cell = _cell_tuple(_candidate_cell(candidate))
        if cell is None:
            continue
        coverage_cells = _candidate_coverage_cells(
            start=current_cell,
            end=cell,
            candidate=candidate,
            config=config,
        )
        new_count = len(coverage_cells - covered_cells)
        path_cost = _candidate_cost(candidate) or 0.0
        risk = _finite_or_none(candidate.get("risk")) or 0.0
        reward_components = None
        reward_value = None
        profile_id = None
        profile_hash = None
        if policy_name == CANONICAL_REWARD_RERANK_ORACLE and rerank_profile is not None:
            metrics = {
                "coverage_gain_rate": float(new_count) / max(float(coverage_denominator), TOLERANCE),
                "roi_coverage": _finite_or_none(candidate.get("roi_weighted_coverage_delta")) or float(new_count),
                "information_gain": 0.0,
                "path_cost_m": path_cost,
                "soft_risk_exposure": _candidate_soft_risk_exposure(candidate),
                "hard_risk_violation": False,
                "fallback_used": False,
                "failure": False,
            }
            result = compute_canonical_reward_components(metrics, rerank_profile)
            primary = float(result.reward)
            secondary = float(new_count)
            reward_components = dict(result.components)
            reward_value = float(result.reward)
            profile_id = result.profile_id
            profile_hash = result.profile_hash
        elif policy_name == "cost_aware_coverage_oracle":
            profile = config.get("comparison_utility_profiles", {}).get("risk_aware", DEFAULT_COMPARISON_UTILITY_PROFILES["risk_aware"])
            roi_gain = _finite_or_none(candidate.get("roi_weighted_coverage_delta"))
            if roi_gain is None:
                roi_gain = float(new_count)
            primary = (
                float(roi_gain)
                - float(profile.get("cost_weight", 0.0001)) * float(path_cost)
                - float(profile.get("risk_weight", 0.01)) * float(path_cost) * float(risk)
            )
            secondary = float(new_count)
        else:
            primary = float(new_count)
            secondary = -float(path_cost)
        scored.append((primary, secondary, -float(risk), index, reward_components, reward_value, profile_id, profile_hash))
    selected = max(scored) if scored else None
    selected_index = selected[3] if selected is not None else None
    return {
        "logits": [],
        "masked_logits": [],
        "action_probs": [],
        "value": 0.0,
        "selected_action_index": selected_index,
        "selected_probability": 1.0 if selected_index is not None else 0.0,
        "selected_rank": 1 if selected_index is not None else 0,
        "finite_outputs": True,
        "latency_ms": 0.0,
        "reward_components": selected[4] if selected is not None else None,
        "reward": selected[5] if selected is not None else None,
        "profile_id": selected[6] if selected is not None else None,
        "profile_hash": selected[7] if selected is not None else None,
    }


def _coverage_cells(
    *,
    start: tuple[int, int],
    end: tuple[int, int],
    radius: int,
    mode: str,
) -> set[tuple[int, int]]:
    cells = set(_footprint(end, radius=radius))
    if mode == "path_line_plus_endpoint":
        for cell in _line_cells(start, end):
            cells.update(_footprint(cell, radius=radius))
    return cells


def _candidate_coverage_cells(
    *,
    start: tuple[int, int],
    end: tuple[int, int],
    candidate: dict[str, Any],
    config: dict[str, Any],
    obstacle_source_linkage: dict[str, Any] | None = None,
) -> set[tuple[int, int]]:
    theta_coverage_enabled = theta_viewpoints_enabled(config) or continuous_theta_enabled(config)
    if theta_coverage_enabled and candidate.get("candidate_theta_deg") is not None:
        if bool(config.get("obstacle_occlusion_enabled", False)):
            obstacles = _candidate_obstacles(candidate, config, obstacle_source_linkage=obstacle_source_linkage)[0]
            return visible_cells_for_viewpoint_with_obstacles(
                end,
                theta_deg=float(candidate["candidate_theta_deg"]),
                sensor_range_cells=int(config.get("sensor_range_cells", config.get("coverage_radius_cells", 1))),
                sensor_fov_deg=float(config.get("sensor_fov_deg", 90.0)),
                obstacle_cells=obstacles,
            ).visible_cells
        return visible_cells_for_viewpoint(
            end,
            theta_deg=float(candidate["candidate_theta_deg"]),
            sensor_range_cells=int(config.get("sensor_range_cells", config.get("coverage_radius_cells", 1))),
            sensor_fov_deg=float(config.get("sensor_fov_deg", 90.0)),
        )
    return _coverage_cells(
        start=start,
        end=end,
        radius=int(config["coverage_radius_cells"]),
        mode=str(config["coverage_metric_mode"]),
    )


def _candidate_obstacle_audit(
    *,
    candidate: dict[str, Any],
    config: dict[str, Any],
    cell: tuple[int, int] | None,
    obstacle_source_linkage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    obstacles, source, source_id, source_hash, source_kind, source_is_proxy = _candidate_obstacles(
        candidate,
        config,
        obstacle_source_linkage=obstacle_source_linkage,
    )
    if not bool(config.get("obstacle_occlusion_enabled", False)) or cell is None or candidate.get("candidate_theta_deg") is None:
        return {
            "obstacle_occlusion_enabled": bool(config.get("obstacle_occlusion_enabled", False)),
            "obstacle_source": source,
            "obstacle_source_id": source_id,
            "obstacle_source_hash": source_hash,
            "obstacle_source_kind": source_kind,
            "obstacle_source_is_proxy": source_is_proxy,
            "obstacle_source_missing": False if source else bool(config.get("obstacle_occlusion_enabled", False)),
            "obstacle_cell_count": len(obstacles),
            "obstacle_occluded_cell_count": None,
            "obstacle_blocked_cell_count": None,
            "obstacle_aware_theta_coverage_hash": None,
        }
    if not source:
        return {
            "obstacle_occlusion_enabled": True,
            "obstacle_source": None,
            "obstacle_source_id": None,
            "obstacle_source_hash": None,
            "obstacle_source_kind": None,
            "obstacle_source_is_proxy": False,
            "obstacle_source_missing": True,
            "obstacle_cell_count": 0,
            "obstacle_occluded_cell_count": None,
            "obstacle_blocked_cell_count": None,
            "obstacle_aware_theta_coverage_hash": None,
        }
    result = visible_cells_for_viewpoint_with_obstacles(
        cell,
        theta_deg=float(candidate["candidate_theta_deg"]),
        sensor_range_cells=int(config.get("sensor_range_cells", config.get("coverage_radius_cells", 1))),
        sensor_fov_deg=float(config.get("sensor_fov_deg", 90.0)),
        obstacle_cells=obstacles,
    )
    return {
        "obstacle_occlusion_enabled": True,
        "obstacle_source": source,
        "obstacle_source_id": source_id,
        "obstacle_source_hash": source_hash,
        "obstacle_source_kind": source_kind,
        "obstacle_source_is_proxy": source_is_proxy,
        "obstacle_source_missing": False,
        "obstacle_cell_count": len(obstacles),
        "obstacle_occluded_cell_count": result.occluded_cell_count,
        "obstacle_blocked_cell_count": result.blocked_by_obstacle_count,
        "obstacle_aware_theta_coverage_hash": result.obstacle_aware_theta_coverage_hash,
    }


def _candidate_obstacles(
    candidate: dict[str, Any],
    config: dict[str, Any],
    *,
    obstacle_source_linkage: dict[str, Any] | None = None,
) -> tuple[set[tuple[int, int]], str | None, str | None, str | None, str | None, bool]:
    include_no_go = bool(config.get("no_go_blocks_los", False))
    effective_source = _effective_synthetic_los_obstacle_source(
        scenario_id=str(candidate.get("scenario_id")) if candidate.get("scenario_id") is not None else None,
        payloads=(("candidate", candidate), ("config", config)),
        include_no_go=include_no_go,
        config=config,
    )
    if effective_source is not None:
        cells = {_cell_tuple(cell) for cell in effective_source.get("obstacle_cells", [])}
        cells = {cell for cell in cells if cell is not None}
        return (
            cells,
            str(effective_source.get("obstacle_source_field") or "effective_los_blocker_cells"),
            effective_source.get("obstacle_source_id"),
            str(effective_source.get("obstacle_source_hash") or ""),
            str(effective_source.get("obstacle_source_kind") or ""),
            bool(effective_source.get("obstacle_source_is_proxy", True)),
        )
    obstacles, source = extract_obstacle_cells(candidate, include_no_go=include_no_go)
    if obstacles:
        source_kind, source_is_proxy = _obstacle_kind_for_source_field(source)
        source_hash = stable_obstacle_source_hash(
            source_kind=source_kind or str(source),
            obstacle_cells=obstacles,
            no_go_blocks_los=include_no_go,
        )
        return obstacles, source, None, source_hash, source_kind, source_is_proxy
    config_obstacles, config_source = extract_obstacle_cells(config, include_no_go=include_no_go)
    if config_obstacles:
        source_kind, source_is_proxy = _obstacle_kind_for_source_field(config_source)
        source_hash = stable_obstacle_source_hash(
            source_kind=source_kind or str(config_source),
            obstacle_cells=config_obstacles,
            no_go_blocks_los=include_no_go,
        )
        return config_obstacles, config_source, None, source_hash, source_kind, source_is_proxy
    if isinstance(obstacle_source_linkage, dict) and obstacle_source_linkage.get("obstacle_cells"):
        cells = {_cell_tuple(cell) for cell in obstacle_source_linkage.get("obstacle_cells", [])}
        cells = {cell for cell in cells if cell is not None}
        return (
            cells,
            str(obstacle_source_linkage.get("obstacle_source_field") or obstacle_source_linkage.get("obstacle_source_kind") or "obstacle_source_artifact"),
            str(obstacle_source_linkage.get("obstacle_source_id") or ""),
            str(obstacle_source_linkage.get("obstacle_source_hash") or ""),
            str(obstacle_source_linkage.get("obstacle_source_kind") or ""),
            bool(obstacle_source_linkage.get("obstacle_source_is_proxy", False)),
        )
    return set(), None, None, None, None, False


def _obstacle_kind_for_source_field(source: str | None) -> tuple[str | None, bool]:
    if not source:
        return None, False
    if str(source).startswith("obstacle"):
        return "physical_obstacle_cells", False
    if str(source).startswith("slope_blocked") or str(source).startswith("sidecar_dem_slope"):
        return "slope_blocked_as_obstacle_proxy", True
    if str(source).startswith("synthetic_"):
        return "synthetic_terrain_obstacle_proxy/v1", True
    if str(source).startswith("blocked"):
        return "blocked_as_obstacle_proxy", True
    if str(source).startswith("no_go"):
        return "no_go_as_obstacle_proxy", True
    return str(source), True


def _line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    cells: list[tuple[int, int]] = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        error2 = 2 * err
        if error2 > -dy:
            err -= dy
            x0 += sx
        if error2 < dx:
            err += dx
            y0 += sy
    return cells


def _footprint(cell: tuple[int, int], *, radius: int) -> set[tuple[int, int]]:
    x, y = cell
    return {(x + dx, y + dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)}


def _coverage_curve_auc(coverage_rates: list[float], rollout_steps: int) -> float:
    if len(coverage_rates) < 2 or rollout_steps <= 0:
        return coverage_rates[-1] if coverage_rates else 0.0
    area = 0.0
    for left, right in zip(coverage_rates, coverage_rates[1:]):
        area += (left + right) / 2.0
    return area / float(rollout_steps)


def _empty_model_detail() -> dict[str, Any]:
    return {
        "logits": [],
        "masked_logits": [],
        "action_probs": [],
        "value": 0.0,
        "selected_action_index": None,
        "selected_probability": 0.0,
        "selected_rank": 0,
        "finite_outputs": False,
        "latency_ms": 0.0,
    }


def _cell_tuple(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _entropy(values: Any) -> float:
    if not isinstance(values, (list, tuple)):
        return 0.0
    entropy = 0.0
    for value in values:
        probability = _float_default(value)
        if probability > 0.0:
            entropy -= probability * math.log(probability)
    return entropy


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    numeric = _finite_or_none(numerator)
    denom = _finite_or_none(denominator)
    if numeric is None or denom is None or abs(float(denom)) <= TOLERANCE:
        return None
    return float(numeric) / float(denom)


def _safe_ratio_v2(numerator: Any, denominator: Any, metric_name: str) -> dict[str, Any]:
    numeric = _finite_or_none(numerator)
    denom = _finite_or_none(denominator)
    if numeric is None:
        return {"value": None, "reason_codes": [f"{metric_name}:missing_numerator"]}
    if denom is None:
        return {"value": None, "reason_codes": [f"{metric_name}:missing_denominator"]}
    if abs(float(denom)) <= TOLERANCE:
        return {"value": None, "reason_codes": [f"{metric_name}:near_zero_denominator", "near_zero_denominator"]}
    return {"value": float(numeric) / float(denom), "reason_codes": []}


def _incremental_ratio(numerator: Any, coverage_delta_cells: Any, metric_name: str) -> dict[str, Any]:
    delta = _finite_or_none(coverage_delta_cells)
    if delta is None:
        return {"value": None, "reason_codes": [f"{metric_name}:missing_coverage_delta"]}
    if float(delta) <= TOLERANCE:
        return {"value": None, "reason_codes": [f"{metric_name}:no_positive_incremental_coverage", "no_positive_incremental_coverage"]}
    return _safe_ratio_v2(numerator, delta, metric_name)


def _mean(values: list[Any]) -> float:
    numeric = [float(value) for value in values if _finite_or_none(value) is not None]
    return statistics.mean(numeric) if numeric else 0.0


def _median(values: list[Any]) -> float:
    numeric = [float(value) for value in values if _finite_or_none(value) is not None]
    return statistics.median(numeric) if numeric else 0.0


def _iqr(values: list[Any]) -> float:
    numeric = sorted(float(value) for value in values if _finite_or_none(value) is not None)
    if not numeric:
        return 0.0
    return _percentile(numeric, 75.0) - _percentile(numeric, 25.0)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * (position - lower)


def _distribution_fields(prefix: str, values: list[Any]) -> dict[str, float]:
    numeric = [float(value) for value in values if _finite_or_none(value) is not None]
    if not numeric:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_iqr": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": statistics.mean(numeric),
        f"{prefix}_median": statistics.median(numeric),
        f"{prefix}_iqr": _iqr(numeric),
        f"{prefix}_min": min(numeric),
        f"{prefix}_max": max(numeric),
    }


def _episode_number(episode: dict[str, Any], field: str) -> float:
    aliases = {
        "total_new_cell_count": "new_covered_cell_count",
        "path_cost_total_m": "path_cost",
        "risk_total": "risk",
        "roi_weighted_coverage_total": "valuable_area_covered",
    }
    value = _finite_or_none(episode.get(aliases[field])) if field in aliases else None
    if value is None:
        value = _finite_or_none(episode.get(field))
    return 0.0 if value is None else float(value)


def _nullable_delta(left: Any, right: Any) -> float | None:
    left_number = _finite_or_none(left)
    right_number = _finite_or_none(right)
    if left_number is None or right_number is None:
        return None
    return float(left_number) - float(right_number)


def _oracle_coverage_regret(oracle: dict[str, Any] | None, episode: dict[str, Any]) -> float | None:
    if oracle is None:
        return None
    return max(0.0, _episode_number(oracle, "total_new_cell_count") - _episode_number(episode, "total_new_cell_count"))


def _episode_utility(episode: dict[str, Any], profile: dict[str, float]) -> float:
    return (
        _episode_number(episode, "roi_weighted_coverage_total")
        - float(profile.get("cost_weight", 0.0)) * _episode_number(episode, "path_cost_total_m")
        - float(profile.get("risk_weight", 0.0)) * _episode_number(episode, "risk_cost_weighted_total")
    )


def _oracle_utility_regret(oracle: dict[str, Any] | None, episode: dict[str, Any], profile: dict[str, float]) -> float | None:
    if oracle is None:
        return None
    return max(0.0, _episode_utility(oracle, profile) - _episode_utility(episode, profile))


def _utility_pair_outcome(xunce: dict[str, Any], incumbent: dict[str, Any], profile: dict[str, float]) -> str:
    delta = _episode_utility(xunce, profile) - _episode_utility(incumbent, profile)
    if delta > TOLERANCE:
        return "xunce_win"
    if delta < -TOLERANCE:
        return "incumbent_win"
    return "tie"


def _coverage_pairwise_outcome(coverage_delta_cells: float) -> str:
    if coverage_delta_cells > TOLERANCE:
        return "xunce_coverage_win"
    if coverage_delta_cells < -TOLERANCE:
        return "xunce_coverage_loss"
    return "coverage_tie"


def _missing_oracle_reasons(greedy_oracle: dict[str, Any] | None, cost_aware_oracle: dict[str, Any] | None) -> list[str]:
    reasons: list[str] = []
    if greedy_oracle is None:
        reasons.append("missing_greedy_coverage_oracle_episode")
    if cost_aware_oracle is None:
        reasons.append("missing_cost_aware_oracle_episode")
    return reasons


def _comparison_utility_profiles_from_canonical(profile: Any) -> dict[str, dict[str, float]]:
    weights = profile.soft_reward_components
    normalizers = profile.normalizers
    path_cost_weight = float(weights["path_cost_weight"]) / max(float(normalizers["path_cost_m"]), TOLERANCE)
    if profile.profile_version == "v3":
        risk_weight = float(weights["soft_risk_weight"]) / max(float(normalizers["soft_risk_exposure"]), TOLERANCE)
    else:
        risk_weight = float(weights["risk_weight"]) / max(float(normalizers["risk_proxy"]), TOLERANCE)
    return {
        "coverage_first": {"cost_weight": 0.0, "risk_weight": 0.0},
        "cost_aware": {"cost_weight": path_cost_weight, "risk_weight": 0.0},
        "risk_aware": {"cost_weight": path_cost_weight, "risk_weight": risk_weight},
    }


def _comparison_utility_profiles(payload: Any) -> dict[str, dict[str, float]]:
    if payload is None:
        payload = DEFAULT_COMPARISON_UTILITY_PROFILES
    if not isinstance(payload, dict) or not payload:
        raise ConfigError("comparison_utility_profiles must be a non-empty object")
    normalized: dict[str, dict[str, float]] = {}
    for name, profile in payload.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigError("comparison_utility_profiles keys must be non-empty strings")
        if not isinstance(profile, dict):
            raise ConfigError(f"comparison_utility_profiles.{name} must be an object")
        cost_weight = _nonnegative_float(profile.get("cost_weight", 0.0), f"comparison_utility_profiles.{name}.cost_weight")
        risk_weight = _nonnegative_float(profile.get("risk_weight", 0.0), f"comparison_utility_profiles.{name}.risk_weight")
        normalized[name] = {"cost_weight": cost_weight, "risk_weight": risk_weight}
    for required in ("coverage_first", "cost_aware", "risk_aware"):
        if required not in normalized:
            normalized[required] = dict(DEFAULT_COMPARISON_UTILITY_PROFILES[required])
    return normalized


def _none_to_zero(value: Any) -> float:
    numeric = _finite_or_none(value)
    return 0.0 if numeric is None else float(numeric)


def _boundary_fields() -> dict[str, Any]:
    fields = {field: False for field in BOUNDARY_FIELDS}
    fields.update(
        {
            "canary_traffic_fraction": 0.0,
            "runs_new_training_update": False,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
            "real_world_release_approved": False,
            "real_world_performance_claimed": False,
            "default_policy_replacement_approved": False,
            "real_executor_connection_approved": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
        }
    )
    return fields


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Exploration Coverage Comparison v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- true_model_inference_executed: `{summary['true_model_inference_executed']}`",
            f"- proxy_selection_used: `{summary['proxy_selection_used']}`",
            f"- scenario_count: `{summary['scenario_count']}`",
            f"- rollout_steps: `{summary['rollout_steps']}`",
            f"- xunce_coverage_advantage_established: `{summary['xunce_coverage_advantage_established']}`",
            f"- xunce_coverage_return_delta_vs_incumbent: `{summary['xunce_coverage_return_delta_vs_incumbent']}`",
            f"- xunce_coverage_curve_auc_delta_vs_incumbent: `{summary['xunce_coverage_curve_auc_delta_vs_incumbent']}`",
            f"- xunce_new_covered_cell_delta_vs_incumbent: `{summary['xunce_new_covered_cell_delta_vs_incumbent']}`",
            f"- xunce_path_cost_delta_vs_incumbent: `{summary['xunce_path_cost_delta_vs_incumbent']}`",
            f"- xunce_risk_delta_vs_incumbent: `{summary['xunce_risk_delta_vs_incumbent']}`",
            f"- coverage_per_100m_delta_vs_incumbent: `{summary['coverage_per_100m_delta_vs_incumbent']}`",
            f"- xunce_efficiency_regression_count: `{summary['xunce_efficiency_regression_count']}`",
            f"- xunce_safety_regression_count: `{summary['xunce_safety_regression_count']}`",
            f"- comparison_pairs: `{summary['comparison_pairs']}`",
            f"- comparison_aggregate: `{summary['comparison_aggregate_path']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            "",
            "This is an offline shadow rollout comparison. It does not approve default-policy replacement, real executor connection, checkpoint publication, PPO training, online canary, or real-world performance claims.",
            "",
        ]
    )


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0.0:
        raise ConfigError(f"{name} must be a positive number")
    return float(value)


def _nonnegative_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0.0:
        raise ConfigError(f"{name} must be a non-negative number")
    return float(value)


def _finite_float_or_default(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean")
    return bool(value)


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
