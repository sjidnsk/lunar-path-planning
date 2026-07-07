from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:  # pragma: no cover
    import run_xunce_stage26_8m_generalized_resumable_training_pipeline as stage26_8m
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage26_8m_generalized_resumable_training_pipeline as stage26_8m

import xunce_artifact_io as artifact_io

STAGE_ID = "xunce-stage26-8n-aggressive-sample-update-sweep"
CONFIG_SCHEMA_VERSION = "xunce-stage26-8n-aggressive-sample-update-sweep-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-8n-summary/v1"
UPDATE_ROW_SCHEMA_VERSION = "xunce-stage26-8n-update-sweep-row/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-8n-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-8n-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_8n_aggressive_sample_update_sweep_v1.json"
DEFAULT_OUTPUT_ROOT = "D:/xunce/out/s26_8n"
DEFAULT_STAGE26_8I_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair_v1"
)
DEFAULT_STAGE26_8G_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8g_repair_synthetic_scenario_diversity_v1"
)
DEFAULT_STAGE26_8Q_RECOMMENDED_CONFIG = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8q_derived_high_res_planning_proxy_alignment_v1/"
    "xunce-stage26-8q-recommended-stage26-8n-config.json"
)
DEFAULT_STAGE26_8R_RECOMMENDED_CONFIG = (
    "D:/xunce/out/s26_8r/xunce-stage26-8r-recommended-stage26-8n-config.json"
)
DEFAULT_STAGE26_8S_RECOMMENDED_CONFIG = (
    "D:/xunce/out/s26_8s/xunce-stage26-8s-recommended-stage26-8n-config.json"
)
STAGE26_8Q_RECOMMENDED_SCHEMA_VERSION = "xunce-stage26-8q-recommended-stage26-8n-config/v1"
STAGE26_8Q_RECOMMENDED_COMBO_ID = "aligned_1m_proxy"
STAGE26_8R_RECOMMENDED_SCHEMA_VERSION = "xunce-stage26-8r-recommended-stage26-8n-config/v1"
STAGE26_8S_RECOMMENDED_SCHEMA_VERSION = "xunce-stage26-8s-recommended-stage26-8n-config/v1"
STAGE26_8S_RECOMMENDED_STAGE_ID = "xunce-stage26-8s-terminal-aware-sample-expansion"

SUMMARY_FILE = "xunce-stage26-8n-summary.json"
AGGRESSIVE_CONFIG_FILE = "xunce-stage26-8n-aggressive-config.json"
SHARED_COLLECTOR_AUDIT_FILE = "xunce-stage26-8n-shared-collector-audit.json"
UPDATE_SWEEP_FILE = "xunce-stage26-8n-update-sweep-results.jsonl"
EVAL_SELECTION_AUDIT_FILE = "xunce-stage26-8n-eval-selection-audit.json"
POLICY_MARGIN_AUDIT_FILE = "xunce-stage26-8n-policy-margin-audit.json"
EFFICIENCY_AUDIT_FILE = "xunce-stage26-8n-efficiency-audit.json"
ROUTING_FILE = "xunce-stage26-8n-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-8n-report.md"
MANIFEST_FILE = "xunce-stage26-8n-manifest.json"

ROUTE_INPUTS = "rerun_stage26_8n_required_inputs"
ROUTE_EXPAND_SAMPLE = "expand_stage26_8n_aggressive_sample_budget"
ROUTE_NO_POSE_REACHABLE = "no_hybrid_astar_pose_reachable_candidate_for_action_mask"
ROUTE_COLLECTOR = "repair_stage26_8n_collector_binding_or_safety"
ROUTE_UPDATE = "repair_stage26_8n_update_stability"
ROUTE_REDUCE = "reduce_stage26_8n_aggressive_update_strength"
ROUTE_EVAL = "repair_stage26_8n_eval_binding_or_safety"
ROUTE_CONTINUE = "continue_stage26_8n_aggressive_jobs"
ROUTE_INCREASE = "increase_stage26_synthetic_update_strength_or_sample_count"
ROUTE_MARGIN = "calibrate_stage26_synthetic_discrete_margin_crossing_after_credit"
ROUTE_CREDIT = "repair_stage26_synthetic_credit_assignment"
ROUTE_RESUME = "resume_stage26_8d_seed_horizon_jobs_with_diverse_scenarios"
ROUTE_STAGE26_9 = "run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot"
ROUTE_BOUNDARY = "resolve_stage26_8n_boundary_rejections"

EFFICIENCY_ZERO_COUNT_FIELDS = (
    "synthetic_inference_required_field_missing_count",
    "hybrid_path_missing_provenance_count",
    "hybrid_path_contract_mismatch_count",
    "explicit_unreachable_selected_provenance_count",
    "pre_unreachable_selected_count",
    "post_unreachable_selected_count",
    "pre_selected_reachability_provenance_invalid_count",
    "post_selected_reachability_provenance_invalid_count",
)

STAGE26_8I_REQUIRED_ROUTE = "increase_stage26_synthetic_update_strength_or_sample_count"
BOUNDARY_FIELDS = (
    "release_or_training_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)
STAGE26_8Q_PLANNER_OVERRIDE_FIELDS = tuple(stage26_8m.PLANNER_OVERRIDE_FIELDS)
STAGE26_8Q_REQUIRED_PLANNER_OVERRIDE_FIELDS = (
    "hybrid_astar_planning_grid_source",
    "planner_grid_resolution_m",
    "hybrid_astar_closed_key_xy_resolution_m",
    "hybrid_astar_primitive_duration_s",
    "hybrid_astar_goal_position_tolerance_m",
    "hybrid_astar_goal_theta_tolerance_deg",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.8N aggressive sample/update sweep.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_8n_aggressive_sample_update_sweep(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_8n_aggressive_sample_update_sweep(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root)
    output_root = _resolve_path(output_root, repo_root)
    artifact_io.make_dirs(output_root)

    stage26_8i_summary = _read_json_if_exists(Path(config["stage26_8i_root"]) / "xunce-stage26-8i-summary.json")
    stage26_8q_recommended_config_path = Path(config["stage26_8q_recommended_config_path"])
    stage26_8q_recommended_config, stage26_8q_recommended_read_error = _read_stage26_8q_recommended_if_exists(
        stage26_8q_recommended_config_path
    )
    stage26_8q_recommended_rejections = _stage26_8q_recommended_input_rejections(
        stage26_8q_recommended_config,
        stage26_8q_recommended_config_path,
        int(config["min_trainable_transition_count"]),
        read_error=stage26_8q_recommended_read_error,
    )
    stage26_8r_recommended_config_path = Path(config["stage26_8r_recommended_config_path"])
    stage26_8s_recommended_config_path = Path(config["stage26_8s_recommended_config_path"])
    stage26_8s_recommended_config_exists = artifact_io.path_is_file(stage26_8s_recommended_config_path)
    stage26_8s_recommended_config, stage26_8s_recommended_read_error = _read_stage26_8s_recommended_if_exists(
        stage26_8s_recommended_config_path
    )
    stage26_8s_recommended_rejections = _stage26_8s_recommended_input_rejections(
        stage26_8s_recommended_config,
        stage26_8s_recommended_config_path,
        int(config["min_trainable_transition_count"]),
        read_error=stage26_8s_recommended_read_error,
    )
    stage26_8s_recommended_valid = stage26_8s_recommended_config_exists and not stage26_8s_recommended_rejections
    stage26_8r_recommended_config: dict[str, Any] = {}
    stage26_8r_recommended_rejections: list[str] = []
    if not stage26_8s_recommended_config_exists:
        stage26_8r_recommended_config, stage26_8r_recommended_read_error = _read_stage26_8r_recommended_if_exists(
            stage26_8r_recommended_config_path
        )
        stage26_8r_recommended_rejections = _stage26_8r_recommended_input_rejections(
            stage26_8r_recommended_config,
            stage26_8r_recommended_config_path,
            int(config["min_trainable_transition_count"]),
            read_error=stage26_8r_recommended_read_error,
        )
    pose_sample_recommendation = stage26_8s_recommended_config if stage26_8s_recommended_valid else stage26_8r_recommended_config
    stage26_8q_planner_overrides = _stage26_8q_planner_overrides(stage26_8q_recommended_config)
    boundary_rejections = _boundary_rejections(config)
    input_rejections = _input_rejections(stage26_8i_summary, Path(config["source_scenario_fixture_root"]))
    input_rejections.extend(stage26_8q_recommended_rejections)
    input_rejections.extend(stage26_8s_recommended_rejections)
    input_rejections.extend(stage26_8r_recommended_rejections)
    aggressive_config = _build_stage26_8m_config(config, stage26_8q_planner_overrides, pose_sample_recommendation)
    aggressive_config_path = output_root / AGGRESSIVE_CONFIG_FILE
    _write_json(aggressive_config_path, aggressive_config)

    stage26_8m_root = output_root / "m"
    stage26_8m_summary: dict[str, Any] = {}
    if not boundary_rejections and not input_rejections and config["run_stage26_8m"]:
        stage26_8m_summary = _run_controlled_stage26_8m(
            config,
            aggressive_config_path,
            stage26_8m_root,
            repo_root,
        )
    else:
        stage26_8m_summary = _read_json_if_exists(stage26_8m_root / stage26_8m.SUMMARY_FILE)

    job_rows = _read_jsonl_if_exists(stage26_8m_root / stage26_8m.JOB_STATE_FILE)
    collector_audit = _shared_collector_audit(job_rows)
    update_rows = _update_sweep_rows(job_rows, float(config["max_abs_approx_kl"]))
    selected_eval_job_ids = _selected_eval_job_ids(update_rows, float(config["max_abs_approx_kl"]), int(config["max_stable_eval_count"]))
    eval_selection_audit = {
        "schema_version": "xunce-stage26-8n-eval-selection-audit/v1",
        "max_stable_eval_count": int(config["max_stable_eval_count"]),
        "selected_eval_job_ids": selected_eval_job_ids,
        "stable_combo_count": sum(1 for row in update_rows if _update_row_is_stable(row, float(config["max_abs_approx_kl"]))),
    }
    policy_margin_audit = _policy_margin_audit(job_rows)
    efficiency_audit = _efficiency_audit(job_rows)

    route = _route(
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        collector_audit=collector_audit,
        update_rows=update_rows,
        selected_eval_job_ids=selected_eval_job_ids,
        job_rows=job_rows,
        stage26_8m_summary=stage26_8m_summary,
        min_trainable_transition_count=int(config["min_trainable_transition_count"]),
        max_abs_approx_kl=float(config["max_abs_approx_kl"]),
    )
    status = "passed" if route in {ROUTE_CONTINUE, ROUTE_RESUME, ROUTE_STAGE26_9} else "failed"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "stage26_8i_root": config["stage26_8i_root"],
        "stage26_8i_status": stage26_8i_summary.get("status"),
        "stage26_8i_next_required_change": stage26_8i_summary.get("next_required_change"),
        "stage26_8q_recommended_config_path": config["stage26_8q_recommended_config_path"],
        "stage26_8q_recommended_combo_id": stage26_8q_recommended_config.get("recommended_combo_id"),
        "stage26_8q_aligned_trainable_transition_count": _int_or_none(
            stage26_8q_recommended_config.get("aligned_trainable_transition_count")
        ),
        "stage26_8q_planning_proxy_source": stage26_8q_planner_overrides.get("hybrid_astar_planning_grid_source"),
        "stage26_8q_planner_overrides": stage26_8q_planner_overrides,
        "stage26_8r_recommended_config_path": config["stage26_8r_recommended_config_path"],
        "stage26_8r_recommended_combo_id": stage26_8r_recommended_config.get("recommended_combo_id"),
        "stage26_8r_trainable_transition_count": _int_or_none(
            stage26_8r_recommended_config.get("trainable_transition_count")
        ),
        "stage26_8s_recommended_config_path": config["stage26_8s_recommended_config_path"],
        "stage26_8s_recommended_scenario_count": _int_or_none(
            stage26_8s_recommended_config.get("recommended_scenario_count")
        ),
        "stage26_8s_trainable_transition_count": _int_or_none(
            stage26_8s_recommended_config.get("trainable_transition_count")
        ),
        "stage26_8s_no_hybrid_astar_pose_reachable_candidate_for_action_mask_count": _int_or_none(
            stage26_8s_recommended_config.get("no_hybrid_astar_pose_reachable_candidate_for_action_mask_count")
        ),
        "stage26_8s_selected_pose_unreachable_terminal_count": _int_or_none(
            stage26_8s_recommended_config.get("selected_pose_unreachable_terminal_count")
        ),
        "stage26_8m_root": str(stage26_8m_root),
        "stage26_8m_status": stage26_8m_summary.get("status"),
        "stage26_8m_next_required_change": stage26_8m_summary.get("next_required_change"),
        "run_stage26_8m": bool(config["run_stage26_8m"]),
        "stage26_8m_read_only_job_state_stale_risk": not bool(config["run_stage26_8m"]),
        "trainable_transition_count": collector_audit.get("trainable_transition_count", 0),
        "synthetic_credit_target_selected_count": collector_audit.get("synthetic_credit_target_selected_count", 0),
        "no_hybrid_astar_pose_reachable_candidate_for_action_mask_count": collector_audit.get(
            "no_hybrid_astar_pose_reachable_candidate_for_action_mask_count",
            0,
        ),
        "rejection_no_hybrid_astar_pose_reachable_candidate_for_action_mask_count": collector_audit.get(
            "rejection_no_hybrid_astar_pose_reachable_candidate_for_action_mask_count",
            0,
        ),
        "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
        "combo_count": len(update_rows),
        "stable_combo_count": eval_selection_audit["stable_combo_count"],
        "selected_eval_job_ids": selected_eval_job_ids,
        "selected_action_changed_count": efficiency_audit.get("selected_action_changed_count", 0),
        "selected_viewpoint_changed_count": efficiency_audit.get("selected_viewpoint_changed_count", 0),
        "selected_theta_changed_count": efficiency_audit.get("selected_theta_changed_count", 0),
        "mean_abs_probability_delta": policy_margin_audit.get("mean_abs_probability_delta", 0.0),
        "mean_top1_probability_margin": policy_margin_audit.get("mean_top1_probability_margin", 0.0),
        "mean_margin_closure_rate": policy_margin_audit.get("mean_margin_closure_rate"),
        "estimated_updates_to_cross_margin_mean": policy_margin_audit.get("estimated_updates_to_cross_margin_mean"),
        "main_final_coverage_delta": efficiency_audit.get("main_final_coverage_delta", 0.0),
        "main_coverage_auc_delta": efficiency_audit.get("main_coverage_auc_delta", 0.0),
        "main_coverage_per_100m_delta": efficiency_audit.get("main_coverage_per_100m_delta", 0.0),
        "hybrid_astar_path_cost_delta": efficiency_audit.get("hybrid_astar_path_cost_delta", 0.0),
        "hybrid_astar_path_cost_delta_is_diagnostic_only": True,
        "synthetic_inference_required_field_missing_count": efficiency_audit.get(
            "synthetic_inference_required_field_missing_count",
            0,
        ),
        "hybrid_path_missing_provenance_count": efficiency_audit.get(
            "hybrid_path_missing_provenance_count",
            0,
        ),
        "hybrid_path_contract_mismatch_count": efficiency_audit.get(
            "hybrid_path_contract_mismatch_count",
            0,
        ),
        "explicit_unreachable_selected_provenance_count": efficiency_audit.get(
            "explicit_unreachable_selected_provenance_count",
            0,
        ),
        "pre_unreachable_selected_count": efficiency_audit.get("pre_unreachable_selected_count", 0),
        "post_unreachable_selected_count": efficiency_audit.get("post_unreachable_selected_count", 0),
        "pre_selected_reachability_provenance_invalid_count": efficiency_audit.get(
            "pre_selected_reachability_provenance_invalid_count",
            0,
        ),
        "post_selected_reachability_provenance_invalid_count": efficiency_audit.get(
            "post_selected_reachability_provenance_invalid_count",
            0,
        ),
        "pre_selected_reachability_provenance_pass_count": efficiency_audit.get(
            "pre_selected_reachability_provenance_pass_count",
            0,
        ),
        "post_selected_reachability_provenance_pass_count": efficiency_audit.get(
            "post_selected_reachability_provenance_pass_count",
            0,
        ),
        "coverage_denominator_source": config["coverage_denominator_source"],
        "post_update_success_metric": config["post_update_success_metric"],
        "coverage_source": config["coverage_source"],
        "path_cost_source": config["path_cost_source"],
        "synthetic_source_kind": config["synthetic_source_kind"],
        "action_space_type": config["action_space_type"],
        "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
        "candidate_reachability_gate_source": aggressive_config.get("candidate_reachability_gate_source"),
        "hybrid_astar_max_iterations": aggressive_config.get("hybrid_astar_max_iterations"),
        "candidate_reachability_max_theta_proposals_per_candidate": aggressive_config.get(
            "candidate_reachability_max_theta_proposals_per_candidate"
        ),
        "candidate_reachability_theta_proposal_policy": aggressive_config.get(
            "candidate_reachability_theta_proposal_policy"
        ),
        "max_traversable_slope_deg": float(config["max_traversable_slope_deg"]),
        "max_abs_approx_kl": float(config["max_abs_approx_kl"]),
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
        "release_or_training_authorized": False,
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
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "summary": str(output_root / SUMMARY_FILE),
        "aggressive_config": str(aggressive_config_path),
        "stage26_8m_root": str(stage26_8m_root),
        "stage26_8q_recommended_config_path": config["stage26_8q_recommended_config_path"],
        "stage26_8q_recommended_combo_id": stage26_8q_recommended_config.get("recommended_combo_id"),
        "stage26_8q_aligned_trainable_transition_count": _int_or_none(
            stage26_8q_recommended_config.get("aligned_trainable_transition_count")
        ),
        "stage26_8q_planning_proxy_source": stage26_8q_planner_overrides.get("hybrid_astar_planning_grid_source"),
        "stage26_8q_planner_overrides": stage26_8q_planner_overrides,
        "stage26_8r_recommended_config_path": config["stage26_8r_recommended_config_path"],
        "stage26_8r_recommended_combo_id": stage26_8r_recommended_config.get("recommended_combo_id"),
        "stage26_8r_trainable_transition_count": _int_or_none(
            stage26_8r_recommended_config.get("trainable_transition_count")
        ),
        "stage26_8s_recommended_config_path": config["stage26_8s_recommended_config_path"],
        "stage26_8s_recommended_scenario_count": _int_or_none(
            stage26_8s_recommended_config.get("recommended_scenario_count")
        ),
        "stage26_8s_trainable_transition_count": _int_or_none(
            stage26_8s_recommended_config.get("trainable_transition_count")
        ),
        "stage26_8s_no_hybrid_astar_pose_reachable_candidate_for_action_mask_count": _int_or_none(
            stage26_8s_recommended_config.get("no_hybrid_astar_pose_reachable_candidate_for_action_mask_count")
        ),
        "stage26_8s_selected_pose_unreachable_terminal_count": _int_or_none(
            stage26_8s_recommended_config.get("selected_pose_unreachable_terminal_count")
        ),
        "candidate_reachability_gate_source": aggressive_config.get("candidate_reachability_gate_source"),
        "hybrid_astar_max_iterations": aggressive_config.get("hybrid_astar_max_iterations"),
        "candidate_reachability_max_theta_proposals_per_candidate": aggressive_config.get(
            "candidate_reachability_max_theta_proposals_per_candidate"
        ),
        "candidate_reachability_theta_proposal_policy": aggressive_config.get(
            "candidate_reachability_theta_proposal_policy"
        ),
        "shared_collector_audit": str(output_root / SHARED_COLLECTOR_AUDIT_FILE),
        "update_sweep_results": str(output_root / UPDATE_SWEEP_FILE),
        "eval_selection_audit": str(output_root / EVAL_SELECTION_AUDIT_FILE),
        "policy_margin_audit": str(output_root / POLICY_MARGIN_AUDIT_FILE),
        "efficiency_audit": str(output_root / EFFICIENCY_AUDIT_FILE),
        "routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }

    _write_json(output_root / SHARED_COLLECTOR_AUDIT_FILE, collector_audit)
    _write_jsonl(output_root / UPDATE_SWEEP_FILE, update_rows)
    _write_json(output_root / EVAL_SELECTION_AUDIT_FILE, eval_selection_audit)
    _write_json(output_root / POLICY_MARGIN_AUDIT_FILE, policy_margin_audit)
    _write_json(output_root / EFFICIENCY_AUDIT_FILE, efficiency_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    artifact_io.write_text(output_root / REPORT_FILE, _render_report(summary))
    return summary


def _run_controlled_stage26_8m(config: dict[str, Any], config_path: Path, output_root: Path, repo_root: Path) -> dict[str, Any]:
    aggregate = stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=config_path,
        output_root=output_root,
        repo_root=repo_root,
        run_mode_override="aggregate_only",
    )
    rows = _read_jsonl_if_exists(output_root / stage26_8m.JOB_STATE_FILE)
    collector_audit = _shared_collector_audit(rows)
    if not rows or _has_pending_phase(rows, "collector"):
        return stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
            config_path=config_path,
            output_root=output_root,
            repo_root=repo_root,
            run_mode_override="run_next",
            max_jobs_override=1,
        )
    if (
        collector_audit.get("collector_complete") is True
        and int(collector_audit.get("trainable_transition_count") or 0) < int(config["min_trainable_transition_count"])
    ):
        return aggregate
    update_pending = _first_pending_update(rows)
    if update_pending:
        return stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
            config_path=config_path,
            output_root=output_root,
            repo_root=repo_root,
            run_mode_override="run_phase",
            job_id_override=str(update_pending["job_id"]),
            phase_override="update",
            max_jobs_override=1,
        )
    update_rows = _update_sweep_rows(rows, float(config["max_abs_approx_kl"]))
    selected_eval_job_ids = _selected_eval_job_ids(update_rows, float(config["max_abs_approx_kl"]), int(config["max_stable_eval_count"]))
    for job_id in selected_eval_job_ids:
        next_phase = _next_eval_phase(rows, job_id)
        if next_phase:
            return stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
                config_path=config_path,
                output_root=output_root,
                repo_root=repo_root,
                run_mode_override="run_phase",
                job_id_override=job_id,
                phase_override=next_phase,
                max_jobs_override=1,
            )
    return aggregate


def _build_stage26_8m_config(
    config: dict[str, Any],
    planner_overrides: dict[str, Any] | None = None,
    pose_gate_recommendation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pose_gate_recommendation = pose_gate_recommendation or {}
    scenario_count = _int_or_none(pose_gate_recommendation.get("recommended_scenario_count"))
    payload = {
        "schema_version": stage26_8m.CONFIG_SCHEMA_VERSION,
        "stage_id": stage26_8m.STAGE_ID,
        "run_mode": "run_next",
        "max_jobs_per_invocation": 1,
        "collector_reuse_policy": stage26_8m.COLLECTOR_REUSE_BY_HORIZON_SEED_SCENARIO_ROLLOUT,
        "base_stage26_1_config": config["base_stage26_1_config"],
        "base_stage26_2_config": config["base_stage26_2_config"],
        "base_stage26_3_config": config["base_stage26_3_config"],
        "source_scenario_fixture_root": config["source_scenario_fixture_root"],
        "horizons": [16],
        "seeds": [260801],
        "scenario_counts": [scenario_count] if scenario_count is not None else [6],
        "collector_rollout_steps": [20],
        "eval_rollout_steps": [20],
        "update_combos": config["update_combos"],
        "clip_ratio": 0.2,
        "max_grad_norm": 1.0,
        "max_abs_approx_kl": float(config["max_abs_approx_kl"]),
        "coverage_denominator_source": config["coverage_denominator_source"],
        "post_update_success_metric": config["post_update_success_metric"],
        "coverage_source": config["coverage_source"],
        "path_cost_source": config["path_cost_source"],
        "synthetic_source_kind": config["synthetic_source_kind"],
        "action_space_type": config["action_space_type"],
        "candidate_reachability_gate_source": getattr(
            stage26_8m.stage21_5,
            "CANDIDATE_REACHABILITY_GATE_SOURCE",
            "hybrid_astar_pose_reachability/v1",
        ),
        "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
        "hybrid_astar_max_iterations": int(
            pose_gate_recommendation.get("hybrid_astar_max_iterations", config["hybrid_astar_max_iterations"])
        ),
        "candidate_reachability_max_theta_proposals_per_candidate": int(
            pose_gate_recommendation.get(
                "candidate_reachability_max_theta_proposals_per_candidate",
                config["candidate_reachability_max_theta_proposals_per_candidate"],
            )
        ),
        "candidate_reachability_theta_proposal_policy": str(
            pose_gate_recommendation.get(
                "candidate_reachability_theta_proposal_policy",
                config["candidate_reachability_theta_proposal_policy"],
            )
        ),
        "max_traversable_slope_deg": float(config["max_traversable_slope_deg"]),
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(planner_overrides or {})
    if pose_gate_recommendation:
        payload["hybrid_astar_max_iterations"] = int(
            pose_gate_recommendation.get("hybrid_astar_max_iterations", payload["hybrid_astar_max_iterations"])
        )
        payload["candidate_reachability_max_theta_proposals_per_candidate"] = int(
            pose_gate_recommendation.get(
                "candidate_reachability_max_theta_proposals_per_candidate",
                payload["candidate_reachability_max_theta_proposals_per_candidate"],
            )
        )
        payload["candidate_reachability_theta_proposal_policy"] = str(
            pose_gate_recommendation.get(
                "candidate_reachability_theta_proposal_policy",
                payload["candidate_reachability_theta_proposal_policy"],
            )
        )
    return payload


def _shared_collector_audit(job_rows: list[dict[str, Any]]) -> dict[str, Any]:
    collector_rows = [row for row in job_rows if row.get("phase") == "collector"]
    complete_rows = [row for row in collector_rows if row.get("status") == "complete"]
    failed_rows = [row for row in collector_rows if row.get("status") == "failed"]
    roots = sorted({str(row.get("output_root") or "") for row in collector_rows if row.get("output_root")})
    summary = _read_json_if_exists(Path(roots[0]) / stage26_8m.stage26_1.SUMMARY_FILE) if roots else {}
    return {
        "schema_version": "xunce-stage26-8n-shared-collector-audit/v1",
        "collector_root_count": len(roots),
        "collector_roots": roots,
        "collector_complete": bool(complete_rows),
        "collector_failed_count": len(failed_rows),
        "collector_blocking_reasons": _dedupe([str(row.get("blocking_reason")) for row in failed_rows if row.get("blocking_reason")]),
        "trainable_transition_count": int(summary.get("trainable_transition_count") or summary.get("batch_row_count") or 0),
        "synthetic_credit_target_selected_count": int(summary.get("synthetic_credit_target_selected_count") or 0),
        "no_hybrid_astar_pose_reachable_candidate_for_action_mask_count": int(
            summary.get("stage21_1_no_hybrid_astar_pose_reachable_candidate_for_action_mask_count")
            or summary.get("rejection_no_hybrid_astar_pose_reachable_candidate_for_action_mask_count")
            or 0
        ),
        "rejection_no_hybrid_astar_pose_reachable_candidate_for_action_mask_count": int(
            summary.get("rejection_no_hybrid_astar_pose_reachable_candidate_for_action_mask_count") or 0
        ),
        "collector_status": summary.get("status"),
        "binding_or_safety_failure": any(row.get("binding_or_safety_failure") for row in collector_rows),
    }


def _update_sweep_rows(job_rows: list[dict[str, Any]], max_abs_approx_kl: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in job_rows:
        if row.get("phase") != "update":
            continue
        summary = _read_json_if_exists(Path(str(row.get("summary_path") or "")))
        stage21_4_summary = _stage21_4_summary(summary)
        policy_kl = _first_float(summary.get("final_post_update_policy_approx_kl"), stage21_4_summary.get("final_post_update_policy_approx_kl"))
        sweep_row = {
            "schema_version": UPDATE_ROW_SCHEMA_VERSION,
            "job_id": row.get("job_id"),
            "combo_id": row.get("update_combo_id"),
            "status": row.get("status"),
            "stage26_2_status": summary.get("status"),
            "stage26_2_next_required_change": summary.get("next_required_change"),
            "stage21_4_status": summary.get("stage21_4_status"),
            "final_post_update_policy_approx_kl": policy_kl,
            "final_post_update_behavior_approx_kl": _first_float(
                summary.get("final_post_update_behavior_approx_kl"),
                stage21_4_summary.get("final_post_update_behavior_approx_kl"),
            ),
            "loss_finite": bool(summary.get("loss_finite")),
            "gradient_finite": bool(summary.get("gradient_finite")),
            "parameter_delta_l2": _first_float(summary.get("parameter_delta_l2"), stage21_4_summary.get("parameter_delta_l2"), default=0.0),
            "checkpoint_exists": bool(summary.get("checkpoint_exists")),
            "checkpoint_reload_passed": bool(summary.get("checkpoint_reload_passed")),
            "experimental_only": bool(summary.get("experimental_only")),
            "checkpoint_boundary_passed": bool(summary.get("checkpoint_boundary_passed")),
            "stage21_3_lineage_passed": bool(summary.get("stage21_3_lineage_passed")),
            "source_checkpoint_sha_consistent": bool(summary.get("source_checkpoint_sha_consistent")),
            "collector_source_checkpoint_match": bool(summary.get("collector_source_checkpoint_match")),
            "ppo_ratio_old_log_prob_source": stage21_4_summary.get("ppo_ratio_old_log_prob_source") or stage21_4_summary.get("old_log_prob_source"),
            "kl_gate_source": stage21_4_summary.get("kl_gate_source"),
            "behavior_policy_kl_diagnostic_only": stage21_4_summary.get("behavior_policy_kl_diagnostic_only"),
        }
        sweep_row["stable_for_eval"] = _update_row_is_stable(sweep_row, max_abs_approx_kl)
        rows.append(sweep_row)
    return rows


def _selected_eval_job_ids(rows: list[dict[str, Any]], max_abs_approx_kl: float, max_eval_count: int = 2) -> list[str]:
    stable = [row for row in rows if _update_row_is_stable(row, max_abs_approx_kl)]
    stable = sorted(
        stable,
        key=lambda row: (
            abs(float(row.get("final_post_update_policy_approx_kl") or 0.0)) <= 0.9 * max_abs_approx_kl,
            float(row.get("parameter_delta_l2") or 0.0),
            -abs(float(row.get("final_post_update_policy_approx_kl") or 0.0)),
        ),
        reverse=True,
    )
    return [str(row.get("job_id")) for row in stable[:max_eval_count]]


def _update_row_is_stable(row: dict[str, Any], max_abs_approx_kl: float) -> bool:
    kl = _finite(row.get("final_post_update_policy_approx_kl"))
    return (
        row.get("stage21_4_status") == "passed"
        and row.get("loss_finite") is True
        and row.get("gradient_finite") is True
        and kl is not None
        and abs(kl) <= max_abs_approx_kl
        and row.get("ppo_ratio_old_log_prob_source") == "behavior_policy_when_present"
        and row.get("kl_gate_source") == "policy_old_logprob_when_available/v1"
        and row.get("behavior_policy_kl_diagnostic_only") is True
        and row.get("checkpoint_exists") is True
        and row.get("checkpoint_reload_passed") is True
        and row.get("experimental_only") is True
        and row.get("checkpoint_boundary_passed") is True
        and row.get("stage21_3_lineage_passed") is True
        and row.get("source_checkpoint_sha_consistent") is True
        and row.get("collector_source_checkpoint_match") is True
    )


def _policy_margin_audit(job_rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate_rows = _aggregate_complete_rows(job_rows)
    if not aggregate_rows:
        return {"schema_version": "xunce-stage26-8n-policy-margin-audit/v1"}
    summaries = [_read_json_if_exists(Path(str(row["summary_path"]))) for row in aggregate_rows]
    best = max(summaries, key=lambda item: float(item.get("mean_abs_probability_delta") or 0.0))
    return {
        "schema_version": "xunce-stage26-8n-policy-margin-audit/v1",
        "mean_abs_probability_delta": float(best.get("mean_abs_probability_delta") or 0.0),
        "mean_top1_probability_margin": float(best.get("mean_top1_probability_margin") or 0.0),
        "mean_margin_closure_rate": best.get("mean_margin_closure_rate"),
        "estimated_updates_to_cross_margin_mean": best.get("estimated_updates_to_cross_margin_mean"),
    }


def _efficiency_audit(job_rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate_rows = _aggregate_complete_rows(job_rows)
    if not aggregate_rows:
        return {"schema_version": "xunce-stage26-8n-efficiency-audit/v1"}
    summaries = [_read_json_if_exists(Path(str(row["summary_path"]))) for row in aggregate_rows]
    best = max(
        summaries,
        key=lambda item: (
            int(item.get("selected_action_changed_count") or 0),
            float(item.get("main_coverage_per_100m_delta") or item.get("coverage_per_100m_delta") or 0.0),
        ),
    )
    return {
        "schema_version": "xunce-stage26-8n-efficiency-audit/v1",
        "selected_action_changed_count": int(best.get("selected_action_changed_count") or 0),
        "selected_viewpoint_changed_count": int(best.get("selected_viewpoint_changed_count") or 0),
        "selected_theta_changed_count": int(best.get("selected_theta_changed_count") or 0),
        "main_final_coverage_delta": _first_float(best.get("main_final_coverage_delta"), best.get("final_coverage_delta"), default=0.0),
        "main_coverage_auc_delta": _first_float(best.get("main_coverage_auc_delta"), best.get("coverage_auc_delta"), default=0.0),
        "main_coverage_per_100m_delta": _first_float(best.get("main_coverage_per_100m_delta"), best.get("coverage_per_100m_delta"), default=0.0),
        "hybrid_astar_path_cost_delta": _first_float(best.get("hybrid_astar_path_cost_delta"), default=0.0),
        **{field: int(best.get(field) or 0) for field in EFFICIENCY_ZERO_COUNT_FIELDS},
        "pre_selected_reachability_provenance_invalid_count": int(
            best.get("pre_selected_reachability_provenance_invalid_count") or 0
        ),
        "post_selected_reachability_provenance_invalid_count": int(
            best.get("post_selected_reachability_provenance_invalid_count") or 0
        ),
        "pre_selected_reachability_provenance_pass_count": int(
            best.get("pre_selected_reachability_provenance_pass_count") or 0
        ),
        "post_selected_reachability_provenance_pass_count": int(
            best.get("post_selected_reachability_provenance_pass_count") or 0
        ),
    }


def _route(
    *,
    boundary_rejections: list[str],
    input_rejections: list[str],
    collector_audit: dict[str, Any],
    update_rows: list[dict[str, Any]],
    selected_eval_job_ids: list[str],
    job_rows: list[dict[str, Any]],
    stage26_8m_summary: dict[str, Any],
    min_trainable_transition_count: int,
    max_abs_approx_kl: float,
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections:
        return ROUTE_INPUTS
    if collector_audit.get("binding_or_safety_failure"):
        return ROUTE_COLLECTOR
    if _has_failed_phase(job_rows, "collector"):
        return ROUTE_COLLECTOR
    if collector_audit.get("collector_complete") is True and int(collector_audit.get("trainable_transition_count") or 0) < min_trainable_transition_count:
        return ROUTE_EXPAND_SAMPLE
    if _has_failed_phase(job_rows, "update"):
        return ROUTE_UPDATE
    if _has_failed_eval_phase(job_rows):
        return ROUTE_EVAL
    if _has_dirty_complete_aggregate(job_rows):
        return ROUTE_EVAL
    if not job_rows and stage26_8m_summary.get("next_required_change") == stage26_8m.ROUTE_CONTINUE:
        return ROUTE_CONTINUE
    if _has_blocking_pending(job_rows, selected_eval_job_ids):
        return ROUTE_CONTINUE
    if update_rows and not any(_update_row_is_stable(row, max_abs_approx_kl) for row in update_rows):
        return ROUTE_REDUCE
    if update_rows and not selected_eval_job_ids:
        return ROUTE_REDUCE
    complete_aggregates = [row for row in _aggregate_complete_rows(job_rows) if _aggregate_row_clean(row)]
    if not complete_aggregates and selected_eval_job_ids:
        return ROUTE_CONTINUE
    positive = [
        row for row in complete_aggregates
        if int(row.get("selected_action_changed_count") or 0) > 0
        and float(row.get("main_coverage_per_100m_delta") or 0.0) >= 0.0
    ]
    negative = [
        row for row in complete_aggregates
        if int(row.get("selected_action_changed_count") or 0) > 0
        and float(row.get("main_coverage_per_100m_delta") or 0.0) < 0.0
    ]
    if len(positive) >= 2:
        return ROUTE_STAGE26_9
    if positive:
        return ROUTE_RESUME
    if negative:
        return ROUTE_CREDIT
    if _near_margin(job_rows):
        return ROUTE_MARGIN
    return ROUTE_INCREASE


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    if payload.get("stage_id") != STAGE_ID:
        raise ValueError(f"stage_id must be {STAGE_ID}")
    defaults = {
        "stage26_8i_root": DEFAULT_STAGE26_8I_ROOT,
        "source_scenario_fixture_root": DEFAULT_STAGE26_8G_ROOT,
        "stage26_8q_recommended_config_path": DEFAULT_STAGE26_8Q_RECOMMENDED_CONFIG,
        "stage26_8r_recommended_config_path": DEFAULT_STAGE26_8R_RECOMMENDED_CONFIG,
        "stage26_8s_recommended_config_path": DEFAULT_STAGE26_8S_RECOMMENDED_CONFIG,
        "run_stage26_8m": True,
        "min_trainable_transition_count": 100,
        "max_stable_eval_count": 2,
        "release_or_training_authorized": False,
        "base_stage26_1_config": "configs/xunce_stage26_1_synthetic_terrain_collector_smoke_v1.json",
        "base_stage26_2_config": "configs/xunce_stage26_2_synthetic_terrain_ppo_update_smoke_v1.json",
        "base_stage26_3_config": "configs/xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke_v1.json",
        "update_combos": _default_update_combos(),
        "max_abs_approx_kl": 1.5,
        "coverage_denominator_source": "main_coverable_cells/v1",
        "post_update_success_metric": "main_coverable_coverage_efficiency/v1",
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
        "hybrid_astar_candidate_eval_workers": 4,
        "hybrid_astar_max_iterations": 50,
        "candidate_reachability_max_theta_proposals_per_candidate": 1,
        "candidate_reachability_theta_proposal_policy": "candidate_viewpoint_current_step/v1",
        "max_traversable_slope_deg": 30.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    for field in (
        "stage26_8i_root",
        "source_scenario_fixture_root",
        "stage26_8q_recommended_config_path",
        "stage26_8r_recommended_config_path",
        "stage26_8s_recommended_config_path",
        "base_stage26_1_config",
        "base_stage26_2_config",
        "base_stage26_3_config",
    ):
        config[field] = str(_resolve_path(Path(str(config[field])), repo_root))
    config["min_trainable_transition_count"] = _positive_int(
        config["min_trainable_transition_count"],
        "min_trainable_transition_count",
    )
    config["max_stable_eval_count"] = _positive_int(config["max_stable_eval_count"], "max_stable_eval_count")
    config["max_abs_approx_kl"] = float(config["max_abs_approx_kl"])
    if not math.isfinite(config["max_abs_approx_kl"]) or config["max_abs_approx_kl"] > 1.5:
        raise ValueError("max_abs_approx_kl must be finite and must not exceed 1.5")
    config["update_combos"] = _update_combos(config["update_combos"])
    config["run_stage26_8m"] = bool(config["run_stage26_8m"])
    config["hybrid_astar_candidate_eval_workers"] = int(config["hybrid_astar_candidate_eval_workers"])
    config["hybrid_astar_max_iterations"] = int(config["hybrid_astar_max_iterations"])
    if config["hybrid_astar_max_iterations"] <= 0:
        raise ValueError("hybrid_astar_max_iterations must be positive")
    config["candidate_reachability_max_theta_proposals_per_candidate"] = int(
        config.get("candidate_reachability_max_theta_proposals_per_candidate", 1)
    )
    if config["candidate_reachability_max_theta_proposals_per_candidate"] <= 0:
        raise ValueError("candidate_reachability_max_theta_proposals_per_candidate must be positive")
    config["candidate_reachability_theta_proposal_policy"] = str(
        config.get("candidate_reachability_theta_proposal_policy") or "candidate_viewpoint_current_step/v1"
    )
    if config["candidate_reachability_theta_proposal_policy"] not in {
        "candidate_viewpoint_current_step/v1",
        "candidate_current_bearing_sweep/v1",
    }:
        raise ValueError("candidate_reachability_theta_proposal_policy is invalid")
    config["max_traversable_slope_deg"] = float(config["max_traversable_slope_deg"])
    for field in BOUNDARY_FIELDS:
        config[field] = bool(config.get(field, False))
    config["canary_traffic_fraction"] = float(config.get("canary_traffic_fraction", 0.0))
    return config


def _default_update_combos() -> list[dict[str, Any]]:
    return [
        {"combo_id": "policy_amp_repro", "epochs": 8, "learning_rate": 2.0e-5, "policy_loss_coefficient": 2.0, "value_loss_coefficient": 0.01, "entropy_coefficient": 0.005, "loss_scale": 0.5},
        {"combo_id": "depth16_lr3e5_policy3", "epochs": 16, "learning_rate": 3.0e-5, "policy_loss_coefficient": 3.0, "value_loss_coefficient": 0.01, "entropy_coefficient": 0.005, "loss_scale": 0.5},
        {"combo_id": "depth24_lr3e5_policy3", "epochs": 24, "learning_rate": 3.0e-5, "policy_loss_coefficient": 3.0, "value_loss_coefficient": 0.01, "entropy_coefficient": 0.003, "loss_scale": 0.5},
        {"combo_id": "lr5e5_policy3", "epochs": 16, "learning_rate": 5.0e-5, "policy_loss_coefficient": 3.0, "value_loss_coefficient": 0.01, "entropy_coefficient": 0.003, "loss_scale": 0.5},
        {"combo_id": "value_off_policy4_probe", "epochs": 16, "learning_rate": 3.0e-5, "policy_loss_coefficient": 4.0, "value_loss_coefficient": 0.0, "entropy_coefficient": 0.003, "loss_scale": 0.75},
    ]


def _update_combos(value: Any) -> list[dict[str, Any]]:
    combos: list[dict[str, Any]] = []
    for item in value:
        combos.append(
            {
                "combo_id": str(item["combo_id"]),
                "epochs": int(item["epochs"]),
                "learning_rate": float(item["learning_rate"]),
                "policy_loss_coefficient": float(item["policy_loss_coefficient"]),
                "value_loss_coefficient": float(item["value_loss_coefficient"]),
                "entropy_coefficient": float(item["entropy_coefficient"]),
                "loss_scale": float(item["loss_scale"]),
            }
        )
    return combos


def _input_rejections(stage26_8i_summary: dict[str, Any], stage26_8g_root: Path) -> list[str]:
    reasons: list[str] = []
    if not stage26_8i_summary:
        return ["stage26_8i_summary_missing"]
    if stage26_8i_summary.get("stage_id") != "xunce-stage26-8i-diverse-scenario-policy-signal-strength-repair":
        reasons.append("stage26_8i_wrong_stage_id")
    if stage26_8i_summary.get("next_required_change") != STAGE26_8I_REQUIRED_ROUTE:
        reasons.append("stage26_8i_route_mismatch")
    if not artifact_io.path_exists(stage26_8g_root):
        reasons.append("source_scenario_fixture_root_missing")
    return reasons


def _stage26_8q_recommended_input_rejections(
    recommended: dict[str, Any],
    recommended_path: Path,
    min_trainable_transition_count: int,
    *,
    read_error: str | None = None,
) -> list[str]:
    if not artifact_io.path_is_file(recommended_path):
        return ["stage26_8q_recommended_config_missing"]
    if read_error:
        return [read_error]

    reasons: list[str] = []
    if recommended.get("schema_version") != STAGE26_8Q_RECOMMENDED_SCHEMA_VERSION:
        reasons.append("stage26_8q_recommended_schema_version_mismatch")
    if recommended.get("recommended_combo_id") != STAGE26_8Q_RECOMMENDED_COMBO_ID:
        reasons.append("stage26_8q_recommended_combo_id_mismatch")

    recommended_min = _int_or_none(recommended.get("min_trainable_transition_count"))
    required_trainable_count = max(min_trainable_transition_count, recommended_min or 0)
    aligned_count = _int_or_none(recommended.get("aligned_trainable_transition_count"))
    if aligned_count is None or aligned_count < required_trainable_count:
        reasons.append("stage26_8q_aligned_trainable_transition_count_below_min")

    for field in BOUNDARY_FIELDS:
        if bool(recommended.get(field, False)):
            reasons.append(f"stage26_8q_recommended_{field}_not_false")

    canary_fraction = _finite(recommended.get("canary_traffic_fraction", 0.0))
    if canary_fraction is None or canary_fraction != 0.0:
        reasons.append("stage26_8q_recommended_canary_traffic_fraction_not_zero")

    if recommended.get("hybrid_astar_planning_grid_source") != stage26_8m.PLANNER_OVERRIDE_SOURCE:
        reasons.append("stage26_8q_recommended_planning_grid_source_mismatch")

    for field in STAGE26_8Q_REQUIRED_PLANNER_OVERRIDE_FIELDS:
        if field == "hybrid_astar_planning_grid_source":
            continue
        value = _finite(recommended.get(field))
        if value is None or value <= 0.0:
            reasons.append(f"stage26_8q_recommended_{field}_invalid")

    for field in STAGE26_8Q_PLANNER_OVERRIDE_FIELDS:
        if field in STAGE26_8Q_REQUIRED_PLANNER_OVERRIDE_FIELDS or field == "hybrid_astar_planning_grid_source":
            continue
        if recommended.get(field) is None:
            continue
        value = _finite(recommended.get(field))
        if value is None or value <= 0.0:
            reasons.append(f"stage26_8q_recommended_{field}_invalid")

    return _dedupe(reasons)


def _stage26_8q_planner_overrides(recommended: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for field in STAGE26_8Q_PLANNER_OVERRIDE_FIELDS:
        if recommended.get(field) is None:
            continue
        if field == "hybrid_astar_planning_grid_source":
            overrides[field] = str(recommended[field])
        else:
            parsed = _finite(recommended[field])
            overrides[field] = parsed if parsed is not None else recommended[field]
    return overrides


def _stage26_8r_recommended_input_rejections(
    recommended: dict[str, Any],
    recommended_path: Path,
    min_trainable_transition_count: int,
    *,
    read_error: str | None = None,
) -> list[str]:
    if not artifact_io.path_is_file(recommended_path):
        return ["stage26_8r_recommended_config_missing"]
    if read_error:
        return [read_error]

    reasons: list[str] = []
    if recommended.get("schema_version") != STAGE26_8R_RECOMMENDED_SCHEMA_VERSION:
        reasons.append("stage26_8r_recommended_schema_version_mismatch")
    if recommended.get("candidate_reachability_gate_source") != "hybrid_astar_pose_reachability/v1":
        reasons.append("stage26_8r_recommended_gate_source_mismatch")
    policy = str(recommended.get("candidate_reachability_theta_proposal_policy") or "")
    if policy not in {"candidate_viewpoint_current_step/v1", "candidate_current_bearing_sweep/v1"}:
        reasons.append("stage26_8r_recommended_theta_proposal_policy_invalid")
    cap = _int_or_none(recommended.get("candidate_reachability_max_theta_proposals_per_candidate"))
    if cap is None or cap <= 0:
        reasons.append("stage26_8r_recommended_theta_proposal_cap_invalid")
    iterations = _int_or_none(recommended.get("hybrid_astar_max_iterations"))
    if iterations is None or iterations <= 0:
        reasons.append("stage26_8r_recommended_hybrid_astar_max_iterations_invalid")
    recommended_min = _int_or_none(recommended.get("min_trainable_transition_count"))
    required_trainable_count = max(min_trainable_transition_count, recommended_min or 0)
    trainable = _int_or_none(recommended.get("trainable_transition_count"))
    if trainable is None or trainable < required_trainable_count:
        reasons.append("stage26_8r_recommended_trainable_transition_count_below_min")
    if _int_or_none(recommended.get("no_hybrid_astar_pose_reachable_candidate_for_action_mask_count")) != 0:
        reasons.append("stage26_8r_recommended_pose_gate_empty_count_nonzero")
    if _int_or_none(recommended.get("selected_pose_unreachable_terminal_count")) != 0:
        reasons.append("stage26_8r_recommended_selected_pose_unreachable_terminal_count_nonzero")
    for field in BOUNDARY_FIELDS:
        if bool(recommended.get(field, False)):
            reasons.append(f"stage26_8r_recommended_{field}_not_false")
    canary_fraction = _finite(recommended.get("canary_traffic_fraction", 0.0))
    if canary_fraction is None or canary_fraction != 0.0:
        reasons.append("stage26_8r_recommended_canary_traffic_fraction_not_zero")
    return _dedupe(reasons)


def _stage26_8s_recommended_input_rejections(
    recommended: dict[str, Any],
    recommended_path: Path,
    min_trainable_transition_count: int,
    *,
    read_error: str | None = None,
) -> list[str]:
    if not artifact_io.path_is_file(recommended_path):
        return []
    if read_error:
        return [read_error]

    reasons: list[str] = []
    if recommended.get("schema_version") != STAGE26_8S_RECOMMENDED_SCHEMA_VERSION:
        reasons.append("stage26_8s_recommended_schema_version_mismatch")
    if recommended.get("stage_id") != STAGE26_8S_RECOMMENDED_STAGE_ID:
        reasons.append("stage26_8s_recommended_stage_id_mismatch")

    scenario_count = _int_or_none(recommended.get("recommended_scenario_count"))
    if scenario_count is None or scenario_count <= 0:
        reasons.append("stage26_8s_recommended_scenario_count_invalid")

    if recommended.get("candidate_reachability_gate_source") != "hybrid_astar_pose_reachability/v1":
        reasons.append("stage26_8s_recommended_gate_source_mismatch")
    policy = str(recommended.get("candidate_reachability_theta_proposal_policy") or "")
    if policy not in {"candidate_viewpoint_current_step/v1", "candidate_current_bearing_sweep/v1"}:
        reasons.append("stage26_8s_recommended_theta_proposal_policy_invalid")
    cap = _int_or_none(recommended.get("candidate_reachability_max_theta_proposals_per_candidate"))
    if cap is None or cap <= 0:
        reasons.append("stage26_8s_recommended_theta_proposal_cap_invalid")
    iterations = _int_or_none(recommended.get("hybrid_astar_max_iterations"))
    if iterations is None or iterations <= 0:
        reasons.append("stage26_8s_recommended_hybrid_astar_max_iterations_invalid")

    recommended_min = _int_or_none(recommended.get("min_trainable_transition_count"))
    required_trainable_count = max(min_trainable_transition_count, recommended_min or 0)
    trainable = _int_or_none(recommended.get("trainable_transition_count"))
    if trainable is None or trainable < required_trainable_count:
        reasons.append("stage26_8s_recommended_trainable_transition_count_below_min")

    terminal_count = _int_or_none(
        recommended.get("no_hybrid_astar_pose_reachable_candidate_for_action_mask_count")
    )
    if terminal_count is None or terminal_count < 0:
        reasons.append("stage26_8s_recommended_pose_gate_terminal_count_invalid")

    if _int_or_none(recommended.get("selected_pose_unreachable_terminal_count")) != 0:
        reasons.append("stage26_8s_recommended_selected_pose_unreachable_terminal_count_nonzero")

    for field in BOUNDARY_FIELDS:
        if bool(recommended.get(field, False)):
            reasons.append(f"stage26_8s_recommended_{field}_not_false")
    canary_fraction = _finite(recommended.get("canary_traffic_fraction", 0.0))
    if canary_fraction is None or canary_fraction != 0.0:
        reasons.append("stage26_8s_recommended_canary_traffic_fraction_not_zero")
    return _dedupe(reasons)


def _read_stage26_8q_recommended_if_exists(path: Path) -> tuple[dict[str, Any], str | None]:
    if not artifact_io.path_is_file(path):
        return {}, None
    try:
        return _read_json(path), None
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
        return {}, "stage26_8q_recommended_config_unreadable"


def _read_stage26_8r_recommended_if_exists(path: Path) -> tuple[dict[str, Any], str | None]:
    if not artifact_io.path_is_file(path):
        return {}, None
    try:
        return _read_json(path), None
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
        return {}, "stage26_8r_recommended_config_unreadable"


def _read_stage26_8s_recommended_if_exists(path: Path) -> tuple[dict[str, Any], str | None]:
    if not artifact_io.path_is_file(path):
        return {}, None
    try:
        return _read_json(path), None
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
        return {}, "stage26_8s_recommended_config_unreadable"


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if bool(config.get(field))]
    if float(config.get("canary_traffic_fraction") or 0.0) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _has_pending_phase(rows: list[dict[str, Any]], phase: str) -> bool:
    return any(row.get("phase") == phase and row.get("status") == "pending" for row in rows)


def _first_pending_update(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pending = [row for row in rows if row.get("phase") == "update" and row.get("status") == "pending"]
    return sorted(pending, key=lambda row: str(row.get("job_id")))[0] if pending else {}


def _next_eval_phase(rows: list[dict[str, Any]], job_id: str) -> str | None:
    by_phase = {str(row.get("phase")): row for row in rows if row.get("job_id") == job_id}
    for phase in ("eval_pre", "eval_post", "aggregate"):
        if by_phase.get(phase, {}).get("status") == "pending":
            return phase
        if by_phase.get(phase, {}).get("status") == "failed":
            return None
    return None


def _has_failed_phase(rows: list[dict[str, Any]], phase: str) -> bool:
    return any(row.get("phase") == phase and row.get("status") == "failed" for row in rows)


def _has_failed_eval_phase(rows: list[dict[str, Any]]) -> bool:
    return any(row.get("phase") in {"eval_pre", "eval_post", "aggregate"} and row.get("status") == "failed" for row in rows)


def _has_dirty_complete_aggregate(rows: list[dict[str, Any]]) -> bool:
    return any(
        row.get("phase") == "aggregate"
        and row.get("status") == "complete"
        and not _aggregate_row_clean(row)
        for row in rows
    )


def _aggregate_row_clean(row: dict[str, Any]) -> bool:
    return not row.get("binding_or_safety_failure") and not row.get("lineage_mismatch")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _has_blocking_pending(rows: list[dict[str, Any]], selected_eval_job_ids: list[str]) -> bool:
    selected = set(selected_eval_job_ids)
    for row in rows:
        if row.get("status") != "pending":
            continue
        phase = row.get("phase")
        if phase in {"collector", "update"}:
            return True
        if phase in {"eval_pre", "eval_post", "aggregate"} and row.get("job_id") in selected:
            return True
    return False


def _aggregate_complete_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("phase") == "aggregate" and row.get("status") == "complete"]


def _near_margin(rows: list[dict[str, Any]]) -> bool:
    summaries = [_read_json_if_exists(Path(str(row.get("summary_path") or ""))) for row in _aggregate_complete_rows(rows)]
    return any(float(summary.get("mean_margin_closure_rate") or 0.0) > 0.5 for summary in summaries)


def _stage21_4_summary(stage26_2_summary: dict[str, Any]) -> dict[str, Any]:
    root = stage26_2_summary.get("stage21_4_root")
    if root:
        return _read_json_if_exists(Path(str(root)) / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json")
    return {}


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _positive_int(value: Any, name: str) -> int:
    number = _int_or_none(value)
    if number is None or number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _first_float(*values: Any, default: float | None = None) -> float | None:
    for value in values:
        parsed = _finite(value)
        if parsed is not None:
            return parsed
    return default


def _read_json(path: Path) -> dict[str, Any]:
    return artifact_io.read_json(path)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if artifact_io.path_is_file(path) else {}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not artifact_io.path_is_file(path):
        return []
    return artifact_io.read_jsonl(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    artifact_io.write_json(path, payload)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    artifact_io.write_jsonl(path, rows)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.8N Aggressive Sample + Update Sweep",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- trainable_transition_count: `{summary.get('trainable_transition_count')}`",
            f"- stage26_8q_recommended_config_path: `{summary.get('stage26_8q_recommended_config_path')}`",
            f"- stage26_8q_recommended_combo_id: `{summary.get('stage26_8q_recommended_combo_id')}`",
            f"- stage26_8q_aligned_trainable_transition_count: `{summary.get('stage26_8q_aligned_trainable_transition_count')}`",
            f"- stage26_8q_planning_proxy_source: `{summary.get('stage26_8q_planning_proxy_source')}`",
            f"- stage26_8q_planner_overrides: `{json.dumps(summary.get('stage26_8q_planner_overrides') or {}, sort_keys=True)}`",
            f"- stage26_8r_recommended_config_path: `{summary.get('stage26_8r_recommended_config_path')}`",
            f"- stage26_8r_recommended_combo_id: `{summary.get('stage26_8r_recommended_combo_id')}`",
            f"- stage26_8r_trainable_transition_count: `{summary.get('stage26_8r_trainable_transition_count')}`",
            f"- stage26_8s_recommended_config_path: `{summary.get('stage26_8s_recommended_config_path')}`",
            f"- stage26_8s_recommended_scenario_count: `{summary.get('stage26_8s_recommended_scenario_count')}`",
            f"- stage26_8s_trainable_transition_count: `{summary.get('stage26_8s_trainable_transition_count')}`",
            f"- stage26_8s_no_hybrid_astar_pose_reachable_candidate_for_action_mask_count: `{summary.get('stage26_8s_no_hybrid_astar_pose_reachable_candidate_for_action_mask_count')}`",
            f"- stage26_8s_selected_pose_unreachable_terminal_count: `{summary.get('stage26_8s_selected_pose_unreachable_terminal_count')}`",
            f"- candidate_reachability_gate_source: `{summary.get('candidate_reachability_gate_source')}`",
            f"- hybrid_astar_max_iterations: `{summary.get('hybrid_astar_max_iterations')}`",
            f"- candidate_reachability_max_theta_proposals_per_candidate: `{summary.get('candidate_reachability_max_theta_proposals_per_candidate')}`",
            f"- candidate_reachability_theta_proposal_policy: `{summary.get('candidate_reachability_theta_proposal_policy')}`",
            f"- no_hybrid_astar_pose_reachable_candidate_for_action_mask_count: `{summary.get('no_hybrid_astar_pose_reachable_candidate_for_action_mask_count')}`",
            f"- stable_combo_count: `{summary.get('stable_combo_count')}`",
            f"- selected_action_changed_count: `{summary.get('selected_action_changed_count')}`",
            f"- main_coverage_per_100m_delta: `{summary.get('main_coverage_per_100m_delta')}`",
            f"- pre_selected_reachability_provenance_invalid_count: `{summary.get('pre_selected_reachability_provenance_invalid_count')}`",
            f"- post_selected_reachability_provenance_invalid_count: `{summary.get('post_selected_reachability_provenance_invalid_count')}`",
            "",
            "This stage uses Stage26.8M for resumable orchestration. Hybrid A* path cost remains diagnostic only.",
        ]
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
