from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:  # pragma: no cover
    import run_xunce_stage26_8m_generalized_resumable_training_pipeline as stage26_8m
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage26_8m_generalized_resumable_training_pipeline as stage26_8m

import xunce_artifact_io as artifact_io

STAGE_ID = "xunce-stage26-9-synthetic-terrain-long-horizon-efficiency-pilot"
CONFIG_SCHEMA_VERSION = "xunce-stage26-9-synthetic-terrain-long-horizon-efficiency-pilot-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-9-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-9-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-9-manifest/v1"
MATRIX_ROW_SCHEMA_VERSION = "xunce-stage26-9-long-horizon-matrix-row/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot_v1.json"
DEFAULT_OUTPUT_ROOT = "D:/xunce/out/s26_9"
DEFAULT_STAGE26_8N_ROOT = "D:/xunce/out/s26_8n"

SUMMARY_FILE = "xunce-stage26-9-summary.json"
ROUTING_FILE = "xunce-stage26-9-routing.json"
MANIFEST_FILE = "xunce-stage26-9-manifest.json"
REPORT_FILE = "xunce-stage26-9-report.md"
MATRIX_FILE = "xunce-stage26-9-long-horizon-matrix.jsonl"
SUB_CONFIG_FILE = "xunce-stage26-9-stage26-8m-config.json"

ROUTE_BOUNDARY = "resolve_stage26_9_boundary_rejections"
ROUTE_INPUTS = "repair_stage26_9_required_inputs"
ROUTE_EVAL = "repair_stage26_9_eval_binding_or_safety"
ROUTE_CONTINUE = "continue_stage26_9_long_horizon_jobs"
ROUTE_REVIEW = "review_stage26_9_long_horizon_efficiency_readiness"
ROUTE_EXPAND = "expand_stage26_9_seed_or_horizon_budget"
ROUTE_CREDIT = "repair_stage26_synthetic_credit_assignment"

STAGE26_8N_REQUIRED_ROUTE = "run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot"
FIXED_LOCKED_TUPLES = (
    {"horizon": 20, "rollout_steps": 20},
    {"horizon": 24, "rollout_steps": 24},
    {"horizon": 32, "rollout_steps": 32},
)
FIXED_SEEDS = (260801, 260802, 260803)
REQUIRED_STAGE26_8Q_PLANNER_OVERRIDES = {
    "hybrid_astar_planning_grid_source": "derived_high_res_planning_proxy/v1",
    "planner_grid_resolution_m": 1.0,
    "hybrid_astar_closed_key_xy_resolution_m": 1.0,
    "hybrid_astar_primitive_duration_s": 1.0,
    "hybrid_astar_goal_position_tolerance_m": 1.0,
    "hybrid_astar_goal_theta_tolerance_deg": 45.0,
}
REQUIRED_ZERO_FIELDS = (
    "synthetic_inference_required_field_missing_count",
    "hybrid_path_missing_provenance_count",
    "hybrid_path_contract_mismatch_count",
    "explicit_unreachable_selected_provenance_count",
    "pre_selected_reachability_provenance_invalid_count",
    "post_selected_reachability_provenance_invalid_count",
    "pre_unreachable_selected_count",
    "post_unreachable_selected_count",
)
ALLOWED_STAGE26_8M_SUMMARY_STATUSES = {"passed", "failed"}
ALLOWED_STAGE26_8M_ROUTES = {
    stage26_8m.ROUTE_BOUNDARY,
    stage26_8m.ROUTE_INPUTS,
    stage26_8m.ROUTE_RESUME_STATE,
    stage26_8m.ROUTE_COLLECTOR,
    stage26_8m.ROUTE_UPDATE,
    stage26_8m.ROUTE_EVAL,
    stage26_8m.ROUTE_CONTINUE,
    stage26_8m.ROUTE_INCREASE,
    stage26_8m.ROUTE_CREDIT,
    stage26_8m.ROUTE_RESUME_DIVERSE,
    stage26_8m.ROUTE_STAGE26_9,
}
BOUNDARY_FIELDS = (
    "release_or_training_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)
SAFETY_COUNT_FIELDS = tuple(
    dict.fromkeys(
        tuple(getattr(stage26_8m, "COUNT_FIELDS_REQUIRING_ZERO", ()))
        + (
            "pre_selected_reachability_provenance_invalid_count",
            "post_selected_reachability_provenance_invalid_count",
            "pre_unreachable_selected_count",
            "post_unreachable_selected_count",
            "pre_explicit_unreachable_selected_count",
            "post_explicit_unreachable_selected_count",
            "mask_violation_count",
            "hard_risk_violation_count",
            "open_grid_fallback_count",
            "path_planning_failure_count",
        )
    )
)
DIAGNOSTIC_COUNT_FIELDS = (
    "pre_no_valid_action_terminal_count",
    "post_no_valid_action_terminal_count",
    "no_valid_action_terminal_count",
    "pre_model_inference_skipped_terminal_count",
    "post_model_inference_skipped_terminal_count",
    "model_inference_skipped_terminal_count",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.9 long-horizon efficiency pilot.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root)
    output_root = _resolve_path(output_root, repo_root)
    artifact_io.make_dirs(output_root)
    config["_output_root"] = str(output_root)

    stage26_8n_summary_path = Path(config["stage26_8n_root"]) / "xunce-stage26-8n-summary.json"
    stage26_8n_summary, stage26_8n_read_error = _read_json_if_exists_with_error(
        stage26_8n_summary_path,
        "stage26_8n_summary_unreadable",
    )
    boundary_rejections = _boundary_rejections(config, stage26_8n_summary)
    input_rejections = _stage26_8n_input_rejections(
        stage26_8n_summary,
        stage26_8n_summary_path,
        int(config["min_trainable_transition_count"]),
        read_error=stage26_8n_read_error,
    )
    selected_combos, combo_rejections = _selected_update_combos(stage26_8n_summary, config)
    input_rejections.extend(combo_rejections)

    subjobs = _subjobs(config)
    if not boundary_rejections and not input_rejections and selected_combos:
        for subjob in subjobs:
            sub_config = _build_stage26_8m_config(config, stage26_8n_summary, selected_combos, subjob)
            _write_json(Path(subjob["sub_config_path"]), sub_config)

    matrix_rows = _scan_matrix_rows(subjobs)
    if not boundary_rejections and not input_rejections and bool(config["run_stage26_8m"]):
        pending = next((row for row in matrix_rows if row["status"] == "pending"), None)
        if pending:
            stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
                config_path=Path(str(pending["sub_config_path"])),
                output_root=Path(str(pending["sub_output_root"])),
                repo_root=repo_root,
                run_mode_override="run_next",
                max_jobs_override=1,
            )
            matrix_rows = _scan_matrix_rows(subjobs)

    route = _route(boundary_rejections, input_rejections, matrix_rows)
    status = "passed" if route in {ROUTE_CONTINUE, ROUTE_REVIEW, ROUTE_EXPAND} else "failed"
    summary = _summary(
        config=config,
        stage26_8n_summary=stage26_8n_summary,
        route=route,
        status=status,
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        matrix_rows=matrix_rows,
        output_root=output_root,
    )
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
        "summary_status": status,
        "next_required_change": route,
        "source_scenario_fixture_root": config["source_scenario_fixture_root"],
        "source_scenario_fixture_root_is_legacy_readonly_input": True,
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "artifacts": {
            "summary": str(output_root / SUMMARY_FILE),
            "routing": str(output_root / ROUTING_FILE),
            "manifest": str(output_root / MANIFEST_FILE),
            "report": str(output_root / REPORT_FILE),
            "matrix": str(output_root / MATRIX_FILE),
        },
        "stage26_8n_summary": str(stage26_8n_summary_path),
        "subjob_count": len(matrix_rows),
        "selected_update_combo_ids": [combo["combo_id"] for combo in selected_combos],
    }

    _write_jsonl(output_root / MATRIX_FILE, matrix_rows)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    artifact_io.write_text(output_root / REPORT_FILE, _render_report(summary))
    return summary


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    if payload.get("stage_id") != STAGE_ID:
        raise ValueError(f"stage_id must be {STAGE_ID}")
    defaults = {
        "stage26_8n_root": DEFAULT_STAGE26_8N_ROOT,
        "run_stage26_8m": True,
        "min_trainable_transition_count": 100,
        "locked_tuples": [
            {"horizon": 20, "rollout_steps": 20},
            {"horizon": 24, "rollout_steps": 24},
            {"horizon": 32, "rollout_steps": 32},
        ],
        "seeds": [260801, 260802, 260803],
        "default_scenario_count": 11,
        "max_selected_update_combo_count": 2,
        "base_stage26_1_config": "configs/xunce_stage26_1_synthetic_terrain_collector_smoke_v1.json",
        "base_stage26_2_config": "configs/xunce_stage26_2_synthetic_terrain_ppo_update_smoke_v1.json",
        "base_stage26_3_config": "configs/xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke_v1.json",
        "source_scenario_fixture_root": "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/outputs/path_feedback_batch_xunce_stage26_8g_repair_synthetic_scenario_diversity_v1",
        "update_combos": _default_update_combos(),
        "max_abs_approx_kl": 1.5,
        "collector_reuse_policy": stage26_8m.COLLECTOR_REUSE_BY_HORIZON_SEED_SCENARIO_ROLLOUT,
        "coverage_denominator_source": "main_coverable_cells/v1",
        "post_update_success_metric": "main_coverable_coverage_efficiency/v1",
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
        "hybrid_astar_candidate_eval_workers": 4,
        "candidate_reachability_gate_source": "hybrid_astar_pose_reachability/v1",
        "candidate_reachability_max_theta_proposals_per_candidate": 5,
        "candidate_reachability_theta_proposal_policy": "candidate_current_bearing_sweep/v1",
        "hybrid_astar_max_iterations": 200,
        "max_traversable_slope_deg": 30.0,
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    for field in (
        "stage26_8n_root",
        "base_stage26_1_config",
        "base_stage26_2_config",
        "base_stage26_3_config",
        "source_scenario_fixture_root",
    ):
        config[field] = str(_resolve_path(Path(str(config[field])), repo_root))
    config["min_trainable_transition_count"] = _positive_int(
        config["min_trainable_transition_count"], "min_trainable_transition_count"
    )
    config["default_scenario_count"] = _positive_int(config["default_scenario_count"], "default_scenario_count")
    config["max_selected_update_combo_count"] = _positive_int(
        config["max_selected_update_combo_count"], "max_selected_update_combo_count"
    )
    if config["max_selected_update_combo_count"] != 2:
        raise ValueError("max_selected_update_combo_count must be 2")
    config["locked_tuples"] = _locked_tuples(config["locked_tuples"])
    config["seeds"] = _positive_int_list(config["seeds"], "seeds")
    if config["locked_tuples"] != list(FIXED_LOCKED_TUPLES):
        raise ValueError("Stage26.9 fixed matrix must be H20/H24/H32 with rollout_steps equal to horizon")
    if config["seeds"] != list(FIXED_SEEDS):
        raise ValueError("Stage26.9 fixed seeds must be [260801, 260802, 260803]")
    config["update_combos"] = _update_combos(config["update_combos"])
    config["max_abs_approx_kl"] = _finite_float(config["max_abs_approx_kl"], "max_abs_approx_kl")
    if config["max_abs_approx_kl"] > 1.5:
        raise ValueError("max_abs_approx_kl must not exceed 1.5")
    if not isinstance(config["run_stage26_8m"], bool):
        raise ValueError("run_stage26_8m must be a JSON boolean")
    config["hybrid_astar_candidate_eval_workers"] = _positive_int(
        config["hybrid_astar_candidate_eval_workers"], "hybrid_astar_candidate_eval_workers"
    )
    config["candidate_reachability_max_theta_proposals_per_candidate"] = _positive_int(
        config["candidate_reachability_max_theta_proposals_per_candidate"],
        "candidate_reachability_max_theta_proposals_per_candidate",
    )
    config["hybrid_astar_max_iterations"] = _positive_int(
        config["hybrid_astar_max_iterations"], "hybrid_astar_max_iterations"
    )
    config["max_traversable_slope_deg"] = _finite_float(config["max_traversable_slope_deg"], "max_traversable_slope_deg")
    for field in BOUNDARY_FIELDS:
        config[field] = bool(config.get(field, False))
    config["canary_traffic_fraction"] = _finite_float(config.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    return config


def _stage26_8n_input_rejections(
    summary: dict[str, Any],
    summary_path: Path,
    min_trainable_transition_count: int,
    *,
    read_error: str = "",
) -> list[str]:
    if not artifact_io.path_is_file(summary_path):
        return ["stage26_8n_summary_missing"]
    if read_error:
        return [read_error]
    reasons: list[str] = []
    if summary.get("stage_id") != "xunce-stage26-8n-aggressive-sample-update-sweep":
        reasons.append("stage26_8n_stage_id_mismatch")
    if summary.get("status") != "passed":
        reasons.append("stage26_8n_status_not_passed")
    if summary.get("next_required_change") != STAGE26_8N_REQUIRED_ROUTE:
        reasons.append("stage26_8n_next_required_change_mismatch")
    trainable = _int_or_none(summary.get("trainable_transition_count"))
    if trainable is None or trainable < min_trainable_transition_count:
        reasons.append("stage26_8n_trainable_transition_count_below_min")
    if summary.get("candidate_reachability_gate_source") != "hybrid_astar_pose_reachability/v1":
        reasons.append("stage26_8n_candidate_reachability_gate_source_mismatch")
    if summary.get("candidate_reachability_theta_proposal_policy") != "candidate_current_bearing_sweep/v1":
        reasons.append("stage26_8n_candidate_reachability_theta_proposal_policy_mismatch")
    theta_cap = _int_or_none(summary.get("candidate_reachability_max_theta_proposals_per_candidate"))
    if theta_cap is None or theta_cap < 5:
        reasons.append("stage26_8n_candidate_reachability_max_theta_proposals_per_candidate_below_min")
    max_iterations = _int_or_none(summary.get("hybrid_astar_max_iterations"))
    if max_iterations is None or max_iterations < 200:
        reasons.append("stage26_8n_hybrid_astar_max_iterations_below_min")
    planner_overrides = summary.get("stage26_8q_planner_overrides")
    if not isinstance(planner_overrides, dict):
        reasons.append("stage26_8n_stage26_8q_planner_overrides_missing")
    else:
        for field, expected in REQUIRED_STAGE26_8Q_PLANNER_OVERRIDES.items():
            actual = planner_overrides.get(field)
            if isinstance(expected, str):
                if actual != expected:
                    reasons.append(f"stage26_8n_stage26_8q_planner_overrides_{field}_mismatch")
            elif abs(_first_float(actual) - float(expected)) > 1.0e-9:
                reasons.append(f"stage26_8n_stage26_8q_planner_overrides_{field}_mismatch")
    for field in REQUIRED_ZERO_FIELDS:
        if field not in summary:
            reasons.append(f"stage26_8n_{field}_missing")
            continue
        parsed, valid = _safe_int(summary.get(field))
        if not valid or parsed != 0:
            reasons.append(f"stage26_8n_{field}_nonzero")
    for field in BOUNDARY_FIELDS:
        if field not in summary:
            reasons.append(f"stage26_8n_{field}_missing")
        elif summary.get(field) is not False:
            reasons.append(f"stage26_8n_{field}_not_false")
    if "canary_traffic_fraction" not in summary:
        reasons.append("stage26_8n_canary_traffic_fraction_missing")
    else:
        canary = _safe_float(summary.get("canary_traffic_fraction"))
        if canary is None or canary != 0.0:
            reasons.append("stage26_8n_canary_traffic_fraction_not_zero")
    return _dedupe(reasons)


def _selected_update_combos(summary: dict[str, Any], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    selected_ids = [str(item) for item in summary.get("selected_eval_job_ids") or []]
    if not selected_ids:
        return [], ["selected_eval_job_ids_missing"]
    combo_by_id = {str(combo["combo_id"]): combo for combo in _update_combos(summary.get("update_combos") or config["update_combos"])}
    selected_combos: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for selected_id in selected_ids:
        combo_id = _combo_id_from_selected_eval_job_id(selected_id, combo_by_id)
        if combo_id is None:
            unresolved.append(selected_id)
            continue
        if not any(combo["combo_id"] == combo_id for combo in selected_combos):
            selected_combos.append(dict(combo_by_id[combo_id]))
        if len(selected_combos) >= int(config["max_selected_update_combo_count"]):
            break
    if unresolved or not selected_combos:
        return selected_combos, ["selected_update_combo_ids_unresolved"]
    return selected_combos, []


def _combo_id_from_selected_eval_job_id(selected_id: str, combo_by_id: Mapping[str, dict[str, Any]]) -> str | None:
    if selected_id in combo_by_id:
        return selected_id
    marker = "_u_"
    if marker in selected_id:
        combo_id = selected_id.split(marker, 1)[1]
        return combo_id if combo_id in combo_by_id else None
    matches = [combo_id for combo_id in combo_by_id if selected_id.endswith(combo_id)]
    if len(matches) == 1:
        return matches[0]
    return None


def _subjobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    output_root = Path(str(config.get("_output_root", "")))
    for locked in config["locked_tuples"]:
        for seed in config["seeds"]:
            sub_root = output_root / f"h{locked['horizon']}_s{seed}_r{locked['rollout_steps']}"
            rows.append(
                {
                    "horizon": int(locked["horizon"]),
                    "rollout_steps": int(locked["rollout_steps"]),
                    "seed": int(seed),
                    "sub_output_root": str(sub_root),
                    "sub_config_path": str(sub_root / SUB_CONFIG_FILE),
                }
            )
    return rows


def _build_stage26_8m_config(
    config: dict[str, Any],
    stage26_8n_summary: dict[str, Any],
    selected_combos: list[dict[str, Any]],
    subjob: Mapping[str, Any],
) -> dict[str, Any]:
    scenario_count = _int_or_none(stage26_8n_summary.get("stage26_8s_recommended_scenario_count"))
    payload = {
        "schema_version": stage26_8m.CONFIG_SCHEMA_VERSION,
        "stage_id": stage26_8m.STAGE_ID,
        "run_mode": "run_next",
        "max_jobs_per_invocation": 1,
        "collector_reuse_policy": config["collector_reuse_policy"],
        "base_stage26_1_config": config["base_stage26_1_config"],
        "base_stage26_2_config": config["base_stage26_2_config"],
        "base_stage26_3_config": config["base_stage26_3_config"],
        "source_scenario_fixture_root": config["source_scenario_fixture_root"],
        "horizons": [int(subjob["horizon"])],
        "seeds": [int(subjob["seed"])],
        "scenario_counts": [scenario_count or int(config["default_scenario_count"])],
        "collector_rollout_steps": [int(subjob["rollout_steps"])],
        "eval_rollout_steps": [int(subjob["rollout_steps"])],
        "update_combos": selected_combos or config["update_combos"],
        "max_abs_approx_kl": float(_first_present(stage26_8n_summary.get("max_abs_approx_kl"), config["max_abs_approx_kl"])),
        "coverage_denominator_source": _first_present(
            stage26_8n_summary.get("coverage_denominator_source"), config["coverage_denominator_source"]
        ),
        "post_update_success_metric": _first_present(
            stage26_8n_summary.get("post_update_success_metric"), config["post_update_success_metric"]
        ),
        "coverage_source": _first_present(stage26_8n_summary.get("coverage_source"), config["coverage_source"]),
        "path_cost_source": _first_present(stage26_8n_summary.get("path_cost_source"), config["path_cost_source"]),
        "synthetic_source_kind": _first_present(
            stage26_8n_summary.get("synthetic_source_kind"), config["synthetic_source_kind"]
        ),
        "action_space_type": _first_present(stage26_8n_summary.get("action_space_type"), config["action_space_type"]),
        "hybrid_astar_candidate_eval_workers": int(
            _first_present(
                stage26_8n_summary.get("hybrid_astar_candidate_eval_workers"),
                config["hybrid_astar_candidate_eval_workers"],
            )
        ),
        "candidate_reachability_gate_source": _first_present(
            stage26_8n_summary.get("candidate_reachability_gate_source"),
            config["candidate_reachability_gate_source"],
        ),
        "candidate_reachability_max_theta_proposals_per_candidate": int(
            _first_present(
                stage26_8n_summary.get("candidate_reachability_max_theta_proposals_per_candidate"),
                config["candidate_reachability_max_theta_proposals_per_candidate"],
            )
        ),
        "candidate_reachability_theta_proposal_policy": _first_present(
            stage26_8n_summary.get("candidate_reachability_theta_proposal_policy"),
            config["candidate_reachability_theta_proposal_policy"],
        ),
        "hybrid_astar_max_iterations": int(
            _first_present(stage26_8n_summary.get("hybrid_astar_max_iterations"), config["hybrid_astar_max_iterations"])
        ),
        "max_traversable_slope_deg": float(config["max_traversable_slope_deg"]),
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    planner_overrides = stage26_8n_summary.get("stage26_8q_planner_overrides")
    if isinstance(planner_overrides, dict):
        for field in getattr(stage26_8m, "PLANNER_OVERRIDE_FIELDS", ()):
            if planner_overrides.get(field) is not None:
                payload[field] = planner_overrides[field]
    return payload


def _scan_matrix_rows(subjobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subjob in subjobs:
        sub_root = Path(str(subjob["sub_output_root"]))
        summary, summary_read_error = _read_json_if_exists_with_error(
            sub_root / stage26_8m.SUMMARY_FILE,
            "stage26_8m_summary_unreadable",
        )
        job_rows, job_state_read_error = _read_jsonl_if_exists_with_error(
            sub_root / stage26_8m.JOB_STATE_FILE,
            "stage26_8m_job_state_unreadable",
        )
        summary_contract_failure = (
            summary_read_error
            or job_state_read_error
            or _stage26_8m_summary_contract_failure(summary)
        )
        pending_job_count, pending_valid = _safe_int(summary.get("pending_job_count"))
        completed_job_count, completed_valid = _safe_int(summary.get("completed_job_count"))
        failed_job_count, failed_valid = _safe_int(summary.get("failed_job_count"))
        selected_action_changed_count, selected_valid = _safe_int(summary.get("selected_action_changed_count"))
        numeric_parse_failure = bool(summary) and not (pending_valid and completed_valid and failed_valid)
        numeric_parse_failure = numeric_parse_failure or (
            summary.get("selected_action_changed_count") is not None and not selected_valid
        )
        aggregate_selected_counts: list[int] = []
        for job_row in job_rows:
            if job_row.get("phase") != "aggregate":
                continue
            parsed, valid = _safe_int(job_row.get("selected_action_changed_count"))
            numeric_parse_failure = numeric_parse_failure or (
                job_row.get("selected_action_changed_count") is not None and not valid
            )
            aggregate_selected_counts.append(parsed)
        row = {
            "schema_version": MATRIX_ROW_SCHEMA_VERSION,
            **subjob,
            "status": _subjob_status(summary, summary_contract_failure, numeric_parse_failure),
            "stage26_8m_status": summary.get("status"),
            "stage26_8m_next_required_change": summary.get("next_required_change"),
            "stage26_8m_summary_path": str(sub_root / stage26_8m.SUMMARY_FILE),
            "pending_job_count": pending_job_count,
            "completed_job_count": completed_job_count,
            "failed_job_count": failed_job_count,
            "stage26_8m_clean_credit_route": (
                summary.get("next_required_change") == stage26_8m.ROUTE_CREDIT
                and failed_job_count == 0
                and not summary_contract_failure
            ),
            "main_coverage_per_100m_delta": _first_float(
                summary.get("main_coverage_per_100m_delta"),
                summary.get("mean_main_coverage_per_100m_delta"),
            ),
            "selected_action_changed_count": max(
                selected_action_changed_count,
                sum(aggregate_selected_counts),
            ),
            "binding_or_safety_failure": _binding_or_safety_failure(summary, job_rows)
            or bool(summary_contract_failure)
            or numeric_parse_failure,
            "lineage_mismatch": bool(summary.get("lineage_mismatch")) or any(bool(row.get("lineage_mismatch")) for row in job_rows),
            "summary_contract_failure": summary_contract_failure,
            "numeric_parse_failure": numeric_parse_failure,
            "release_or_training_authorized": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
        for field in SAFETY_COUNT_FIELDS:
            summary_count, valid = _safe_int(summary.get(field))
            numeric_parse_failure = numeric_parse_failure or (summary.get(field) is not None and not valid)
            row[field] = summary_count
            for job_row in job_rows:
                job_count, valid = _safe_int(job_row.get(field))
                numeric_parse_failure = numeric_parse_failure or (job_row.get(field) is not None and not valid)
                row[field] += job_count
        for field in DIAGNOSTIC_COUNT_FIELDS:
            row[field] = _diagnostic_count_from_summary_or_jobs(summary, job_rows, field)
        if numeric_parse_failure:
            row["numeric_parse_failure"] = True
            row["binding_or_safety_failure"] = True
            row["status"] = "failed"
        rows.append(row)
    return rows


def _stage26_8m_summary_contract_failure(summary: dict[str, Any]) -> str:
    if not summary:
        return ""
    if summary.get("schema_version") != stage26_8m.SUMMARY_SCHEMA_VERSION:
        return "stage26_8m_summary_schema_version_mismatch"
    if summary.get("stage_id") != stage26_8m.STAGE_ID:
        return "stage26_8m_summary_stage_id_mismatch"
    for field in ("status", "next_required_change", "pending_job_count", "completed_job_count", "failed_job_count"):
        if field not in summary:
            return f"stage26_8m_summary_{field}_missing"
    if summary.get("status") not in ALLOWED_STAGE26_8M_SUMMARY_STATUSES:
        return "stage26_8m_summary_status_invalid"
    if summary.get("next_required_change") not in ALLOWED_STAGE26_8M_ROUTES:
        return "stage26_8m_summary_next_required_change_invalid"
    if (
        summary.get("next_required_change") != stage26_8m.ROUTE_CONTINUE
        and "mean_main_coverage_per_100m_delta" not in summary
        and "main_coverage_per_100m_delta" not in summary
    ):
        return "stage26_8m_summary_main_coverage_per_100m_delta_missing"
    return ""


def _subjob_status(summary: dict[str, Any], summary_contract_failure: str = "", numeric_parse_failure: bool = False) -> str:
    if summary_contract_failure or numeric_parse_failure:
        return "failed"
    if not summary:
        return "pending"
    pending_job_count, _ = _safe_int(summary.get("pending_job_count"))
    failed_job_count, _ = _safe_int(summary.get("failed_job_count"))
    if summary.get("next_required_change") == stage26_8m.ROUTE_CONTINUE or pending_job_count > 0:
        return "pending"
    if summary.get("next_required_change") == stage26_8m.ROUTE_CREDIT and failed_job_count == 0:
        return "complete"
    if summary.get("status") == "failed" or failed_job_count > 0:
        return "failed"
    return "complete"


def _binding_or_safety_failure(summary: dict[str, Any], job_rows: list[dict[str, Any]]) -> bool:
    if summary.get("next_required_change") == stage26_8m.ROUTE_EVAL:
        return True
    if any(row.get("phase") in {"eval_pre", "eval_post", "aggregate"} and row.get("status") == "failed" for row in job_rows):
        return True
    if bool(summary.get("binding_or_safety_failure")) or bool(summary.get("lineage_mismatch")):
        return True
    if any(bool(row.get("binding_or_safety_failure")) or bool(row.get("lineage_mismatch")) for row in job_rows):
        return True
    return any(_safe_int(summary.get(field))[0] != 0 for field in SAFETY_COUNT_FIELDS)


def _route(boundary_rejections: list[str], input_rejections: list[str], matrix_rows: list[dict[str, Any]]) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections:
        return ROUTE_INPUTS
    if any(_matrix_row_has_eval_or_safety_risk(row) for row in matrix_rows):
        return ROUTE_EVAL
    if any(row.get("status") == "pending" for row in matrix_rows):
        return ROUTE_CONTINUE

    clean_rows = [row for row in matrix_rows if not _matrix_row_has_eval_or_safety_risk(row)]
    if not clean_rows:
        return ROUTE_EXPAND
    if any(row.get("stage26_8m_clean_credit_route") for row in clean_rows):
        return ROUTE_CREDIT
    horizon_count = len({int(row["horizon"]) for row in clean_rows})
    horizons_with_seed_majority_nonnegative = 0
    horizons_with_seed_majority_negative = 0
    has_positive = False
    has_negative = False
    has_action_changed_negative = False
    for horizon in sorted({int(row["horizon"]) for row in clean_rows}):
        rows = [row for row in clean_rows if int(row["horizon"]) == horizon]
        nonnegative = [row for row in rows if float(row.get("main_coverage_per_100m_delta") or 0.0) >= 0.0]
        negative = [row for row in rows if float(row.get("main_coverage_per_100m_delta") or 0.0) < 0.0]
        if len(nonnegative) > len(rows) / 2.0:
            horizons_with_seed_majority_nonnegative += 1
        if len(negative) > len(rows) / 2.0:
            horizons_with_seed_majority_negative += 1
        has_positive = has_positive or any(float(row.get("main_coverage_per_100m_delta") or 0.0) > 0.0 for row in rows)
        has_negative = has_negative or bool(negative)
        has_action_changed_negative = has_action_changed_negative or any(
            int(row.get("selected_action_changed_count") or 0) > 0
            and float(row.get("main_coverage_per_100m_delta") or 0.0) < 0.0
            for row in rows
        )
    if horizon_count and horizons_with_seed_majority_nonnegative >= math.ceil((2.0 * horizon_count) / 3.0):
        return ROUTE_REVIEW
    if horizons_with_seed_majority_negative > horizon_count / 2.0 or has_action_changed_negative:
        return ROUTE_CREDIT
    if has_positive and has_negative:
        return ROUTE_EXPAND
    return ROUTE_EXPAND


def _matrix_row_has_eval_or_safety_risk(row: Mapping[str, Any]) -> bool:
    if row.get("status") == "failed":
        return True
    if _safe_int(row.get("failed_job_count"))[0] > 0:
        return True
    if row.get("stage26_8m_status") == "failed" and row.get("stage26_8m_next_required_change") != stage26_8m.ROUTE_CREDIT:
        return True
    if row.get("stage26_8m_next_required_change") == stage26_8m.ROUTE_EVAL:
        return True
    if bool(row.get("binding_or_safety_failure")) or bool(row.get("lineage_mismatch")):
        return True
    return any(_safe_int(row.get(field))[0] != 0 for field in SAFETY_COUNT_FIELDS)


def _diagnostic_count_from_summary_or_jobs(
    summary: Mapping[str, Any],
    job_rows: list[dict[str, Any]],
    field: str,
) -> int:
    if field in summary:
        return _safe_int(summary.get(field))[0]
    return sum(_safe_int(row.get(field))[0] for row in job_rows)


def _summary(
    *,
    config: dict[str, Any],
    stage26_8n_summary: dict[str, Any],
    route: str,
    status: str,
    boundary_rejections: list[str],
    input_rejections: list[str],
    matrix_rows: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    completed = [row for row in matrix_rows if row["status"] == "complete"]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "stage26_8n_root": config["stage26_8n_root"],
        "stage26_8n_status": stage26_8n_summary.get("status"),
        "stage26_8n_next_required_change": stage26_8n_summary.get("next_required_change"),
        "trainable_transition_count": _int_or_none(stage26_8n_summary.get("trainable_transition_count")),
        "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
        "subjob_count": len(matrix_rows),
        "completed_subjob_count": len(completed),
        "pending_subjob_count": sum(1 for row in matrix_rows if row["status"] == "pending"),
        "failed_subjob_count": sum(1 for row in matrix_rows if row["status"] == "failed"),
        "mean_main_coverage_per_100m_delta": _mean([row.get("main_coverage_per_100m_delta") for row in completed]),
        "pre_no_valid_action_terminal_count": sum(
            _safe_int(row.get("pre_no_valid_action_terminal_count"))[0] for row in matrix_rows
        ),
        "post_no_valid_action_terminal_count": sum(
            _safe_int(row.get("post_no_valid_action_terminal_count"))[0] for row in matrix_rows
        ),
        "no_valid_action_terminal_count": sum(
            _safe_int(row.get("no_valid_action_terminal_count"))[0] for row in matrix_rows
        ),
        "pre_model_inference_skipped_terminal_count": sum(
            _safe_int(row.get("pre_model_inference_skipped_terminal_count"))[0] for row in matrix_rows
        ),
        "post_model_inference_skipped_terminal_count": sum(
            _safe_int(row.get("post_model_inference_skipped_terminal_count"))[0] for row in matrix_rows
        ),
        "model_inference_skipped_terminal_count": sum(
            _safe_int(row.get("model_inference_skipped_terminal_count"))[0] for row in matrix_rows
        ),
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
        "matrix": str(output_root / MATRIX_FILE),
        "source_scenario_fixture_root": config["source_scenario_fixture_root"],
        "source_scenario_fixture_root_is_legacy_readonly_input": True,
        "coverage_source": stage26_8n_summary.get("coverage_source", config["coverage_source"]),
        "path_cost_source": stage26_8n_summary.get("path_cost_source", config["path_cost_source"]),
        "synthetic_source_kind": stage26_8n_summary.get("synthetic_source_kind", config["synthetic_source_kind"]),
        "action_space_type": stage26_8n_summary.get("action_space_type", config["action_space_type"]),
        "hybrid_astar_candidate_eval_workers": int(
            stage26_8n_summary.get("hybrid_astar_candidate_eval_workers") or config["hybrid_astar_candidate_eval_workers"]
        ),
        "candidate_reachability_gate_source": stage26_8n_summary.get(
            "candidate_reachability_gate_source", config["candidate_reachability_gate_source"]
        ),
        "hybrid_astar_max_iterations": int(
            stage26_8n_summary.get("hybrid_astar_max_iterations") or config["hybrid_astar_max_iterations"]
        ),
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _boundary_rejections(config: Mapping[str, Any], stage26_8n_summary: Mapping[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if bool(config.get(field, False))]
    if float(config.get("canary_traffic_fraction") or 0.0) != 0.0:
        reasons.append("canary_traffic_fraction")
    return _dedupe(reasons)


def _locked_tuples(value: Any) -> list[dict[str, int]]:
    if not isinstance(value, list) or not value:
        raise ValueError("locked_tuples must be a non-empty array")
    rows: list[dict[str, int]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("locked_tuples entries must be objects")
        rows.append(
            {
                "horizon": _positive_int(item.get("horizon"), "horizon"),
                "rollout_steps": _positive_int(item.get("rollout_steps"), "rollout_steps"),
            }
        )
    return rows


def _default_update_combos() -> list[dict[str, Any]]:
    return [
        {"combo_id": "policy_amp_repro", "epochs": 8, "learning_rate": 2.0e-5, "policy_loss_coefficient": 2.0, "value_loss_coefficient": 0.01, "entropy_coefficient": 0.005, "loss_scale": 0.5},
        {"combo_id": "depth16_lr3e5_policy3", "epochs": 16, "learning_rate": 3.0e-5, "policy_loss_coefficient": 3.0, "value_loss_coefficient": 0.01, "entropy_coefficient": 0.005, "loss_scale": 0.5},
        {"combo_id": "depth24_lr3e5_policy3", "epochs": 24, "learning_rate": 3.0e-5, "policy_loss_coefficient": 3.0, "value_loss_coefficient": 0.01, "entropy_coefficient": 0.003, "loss_scale": 0.5},
        {"combo_id": "lr5e5_policy3", "epochs": 16, "learning_rate": 5.0e-5, "policy_loss_coefficient": 3.0, "value_loss_coefficient": 0.01, "entropy_coefficient": 0.003, "loss_scale": 0.5},
        {"combo_id": "value_off_policy4_probe", "epochs": 16, "learning_rate": 3.0e-5, "policy_loss_coefficient": 4.0, "value_loss_coefficient": 0.0, "entropy_coefficient": 0.003, "loss_scale": 0.75},
    ]


def _update_combos(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("update_combos must be a non-empty array")
    rows: list[dict[str, Any]] = []
    for item in value:
        rows.append(
            {
                "combo_id": str(item["combo_id"]),
                "epochs": _positive_int(item["epochs"], "epochs"),
                "learning_rate": _positive_float(item["learning_rate"], "learning_rate"),
                "policy_loss_coefficient": _nonnegative_float(
                    item["policy_loss_coefficient"], "policy_loss_coefficient"
                ),
                "value_loss_coefficient": _nonnegative_float(item["value_loss_coefficient"], "value_loss_coefficient"),
                "entropy_coefficient": _nonnegative_float(item["entropy_coefficient"], "entropy_coefficient"),
                "loss_scale": _positive_float(item["loss_scale"], "loss_scale"),
            }
        )
    return rows


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.9 Long-Horizon Efficiency Pilot",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- subjob_count: `{summary.get('subjob_count')}`",
            f"- completed_subjob_count: `{summary.get('completed_subjob_count')}`",
            f"- pending_subjob_count: `{summary.get('pending_subjob_count')}`",
            f"- failed_subjob_count: `{summary.get('failed_subjob_count')}`",
            f"- mean_main_coverage_per_100m_delta: `{summary.get('mean_main_coverage_per_100m_delta')}`",
            f"- no_valid_action_terminal_count: `{summary.get('no_valid_action_terminal_count')}`",
            f"- model_inference_skipped_terminal_count: `{summary.get('model_inference_skipped_terminal_count')}`",
            f"- release_or_training_authorized: `{summary.get('release_or_training_authorized')}`",
            f"- publishes_checkpoint: `{summary.get('publishes_checkpoint')}`",
            f"- replaces_default_policy: `{summary.get('replaces_default_policy')}`",
            f"- connects_real_executor: `{summary.get('connects_real_executor')}`",
            f"- starts_online_canary: `{summary.get('starts_online_canary')}`",
            f"- canary_traffic_fraction: `{summary.get('canary_traffic_fraction')}`",
            f"- source_scenario_fixture_root_is_legacy_readonly_input: `{summary.get('source_scenario_fixture_root_is_legacy_readonly_input')}`",
            "",
            "This stage only orchestrates Stage26.8M long-horizon runs and does not publish checkpoints.",
        ]
    ) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    return artifact_io.read_json(path)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if artifact_io.path_is_file(path) else {}


def _read_json_if_exists_with_error(path: Path, reason: str) -> tuple[dict[str, Any], str]:
    if not artifact_io.path_is_file(path):
        return {}, ""
    try:
        return _read_json(path), ""
    except Exception:
        return {}, reason


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return artifact_io.read_jsonl(path) if artifact_io.path_is_file(path) else []


def _read_jsonl_if_exists_with_error(path: Path, reason: str) -> tuple[list[dict[str, Any]], str]:
    if not artifact_io.path_is_file(path):
        return [], ""
    try:
        return artifact_io.read_jsonl(path), ""
    except Exception:
        return [], reason


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    artifact_io.write_json(path, payload)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    artifact_io.write_jsonl(path, rows)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be positive")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be positive") from None
    if parsed <= 0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _positive_int_list(value: Any, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty array")
    return [_positive_int(item, field) for item in value]


def _finite_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be finite") from None
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _positive_float(value: Any, field: str) -> float:
    parsed = _finite_float(value, field)
    if parsed <= 0.0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _nonnegative_float(value: Any, field: str) -> float:
    parsed = _finite_float(value, field)
    if parsed < 0.0:
        raise ValueError(f"{field} must be nonnegative")
    return parsed


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> tuple[int, bool]:
    if value is None:
        return 0, True
    if isinstance(value, bool):
        return int(value), True
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0, False
    return parsed, True


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _first_float(*values: Any) -> float:
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    return 0.0


def _mean(values: list[Any]) -> float:
    parsed = [_first_float(value) for value in values]
    return sum(parsed) / len(parsed) if parsed else 0.0


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
