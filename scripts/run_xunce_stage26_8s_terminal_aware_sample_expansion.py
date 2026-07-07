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
    import run_xunce_stage26_8n_aggressive_sample_update_sweep as stage26_8n
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage26_8m_generalized_resumable_training_pipeline as stage26_8m
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as stage26_8n

import xunce_artifact_io as artifact_io

STAGE_ID = "xunce-stage26-8s-terminal-aware-sample-expansion"
CONFIG_SCHEMA_VERSION = "xunce-stage26-8s-terminal-aware-sample-expansion-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-8s-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-8s-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-8s-manifest/v1"
RECOMMENDED_SCHEMA_VERSION = "xunce-stage26-8s-recommended-stage26-8n-config/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_8s_terminal_aware_sample_expansion_v1.json"
DEFAULT_OUTPUT_ROOT = "D:/xunce/out/s26_8s"
SUMMARY_FILE = "xunce-stage26-8s-summary.json"
ROUTING_FILE = "xunce-stage26-8s-routing.json"
MANIFEST_FILE = "xunce-stage26-8s-manifest.json"
REPORT_FILE = "xunce-stage26-8s-report.md"
MATRIX_FILE = "xunce-stage26-8s-combo-results.jsonl"
RECOMMENDED_FILE = "xunce-stage26-8s-recommended-stage26-8n-config.json"

ROUTE_USE_RECOMMENDED = "use_stage26_8s_recommended_terminal_aware_config"
ROUTE_EXPAND_SAMPLE = "expand_stage26_8s_terminal_aware_scenario_budget"
ROUTE_REPAIR = "repair_stage26_8s_terminal_or_reachability_evidence"
ROUTE_SCENARIO_SOURCE = "repair_stage26_8s_scenario_diversity_source"
ROUTE_INPUTS = "rerun_stage26_8s_required_inputs"

BOUNDARY_FIELDS = (
    "release_or_training_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)
SAFETY_COUNT_FIELDS = (
    "hard_risk_violation_count",
    "mask_violation_count",
    "path_planning_failure_count",
    "open_grid_fallback_count",
)

GATE_SOURCE = "hybrid_astar_pose_reachability/v1"
REPAIR_POLICY = "candidate_current_bearing_sweep/v1"
LEGACY_POLICY = "candidate_viewpoint_current_step/v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=STAGE_ID)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_8s_terminal_aware_sample_expansion(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_8s_terminal_aware_sample_expansion(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root)
    output_root = _resolve_path(output_root, repo_root)
    artifact_io.make_dirs(output_root)

    boundary_rejections = _boundary_rejections(config)
    base_8m_config, base_8m_config_read_error = _read_json_if_exists_with_error(
        Path(config["stage26_8n_aggressive_config_path"]),
        "stage26_8n_aggressive_config_unreadable",
    )
    input_rejections = _input_rejections(config, base_8m_config, read_error=base_8m_config_read_error)
    scenario_source_rejections = [] if input_rejections else _scenario_source_rejections(base_8m_config)
    scenario_results: list[dict[str, Any]] = []

    if not boundary_rejections and not input_rejections and not scenario_source_rejections:
        for scenario in _scenario_probes(config):
            result = _run_collector_scenario(
                scenario,
                base_8m_config=base_8m_config,
                output_root=output_root,
                repo_root=repo_root,
            )
            scenario_results.append(result)
            if _has_safety_or_evidence_rejections(result) or _scenario_passes(
                result,
                min_trainable_transition_count=int(config["min_trainable_transition_count"]),
            ):
                break

    recommendation = _select_recommended_scenario(
        scenario_results,
        min_trainable_transition_count=int(config["min_trainable_transition_count"]),
    )
    recommended_path = output_root / RECOMMENDED_FILE
    if recommendation:
        _write_json(recommended_path, _recommended_payload(config, recommendation))

    route = _route(
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        scenario_source_rejections=scenario_source_rejections,
        scenario_results=scenario_results,
        recommendation=recommendation,
    )
    status = "passed" if route == ROUTE_USE_RECOMMENDED else "failed"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "scenario_results": scenario_results,
        "combo_results": scenario_results,
        "recommended_scenario_count": recommendation.get("scenario_count") if recommendation else None,
        "recommended_config": str(recommended_path) if recommendation else None,
        "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
        "scenario_source_rejections": scenario_source_rejections,
        "stage26_8n_root": config["stage26_8n_root"],
        "stage26_8n_aggressive_config_path": config["stage26_8n_aggressive_config_path"],
        "candidate_reachability_gate_source": GATE_SOURCE,
        "candidate_reachability_theta_proposal_policy": config["candidate_reachability_theta_proposal_policy"],
        "candidate_reachability_max_theta_proposals_per_candidate": int(
            config["candidate_reachability_max_theta_proposals_per_candidate"]
        ),
        "hybrid_astar_max_iterations": int(config["hybrid_astar_max_iterations"]),
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
        "scenario_source_rejections": scenario_source_rejections,
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
        "routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
        "combo_results": str(output_root / MATRIX_FILE),
        "scenario_results": str(output_root / MATRIX_FILE),
        "recommended_config": str(recommended_path) if recommendation else None,
        "stage26_8n_aggressive_config_path": config["stage26_8n_aggressive_config_path"],
    }
    _write_jsonl(output_root / MATRIX_FILE, scenario_results)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    artifact_io.write_text(output_root / REPORT_FILE, _render_report(summary))
    return summary


def _run_collector_scenario(
    scenario: dict[str, Any],
    *,
    base_8m_config: dict[str, Any],
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    scenario_root = output_root / str(scenario["combo_id"])
    artifact_io.make_dirs(scenario_root)
    config_path = scenario_root / "xunce-stage26-8m-config.json"
    stage26_8m_root = scenario_root / "m"
    generated = _scenario_stage26_8m_config(base_8m_config, scenario)
    _write_json(config_path, generated)
    try:
        loaded = stage26_8m._load_config(config_path, repo_root)
        job = stage26_8m._expand_jobs(loaded, stage26_8m_root, repo_root)[0]
    except Exception as exc:
        return _scenario_result(
            scenario,
            stage26_8m_root,
            scenario_root / "collector",
            {},
            collector_row={
                "status": "failed",
                "blocking_reason": "stage26_8m_config_invalid",
            },
            generated_config=generated,
            execution_blocking_reason="stage26_8m_config_invalid",
            stage26_8m_input_rejections=[f"stage26_8m_config_invalid:{type(exc).__name__}"],
            stage26_8m_boundary_rejections=[],
        )
    collector_root = Path(str(job["collector_root"]))
    collector_row = _collector_scan_row(loaded, stage26_8m_root, repo_root, str(job["job_id"]))
    execution_blocking_reason = ""
    stage26_8m_input_rejections: list[str] = []
    stage26_8m_boundary_rejections: list[str] = []
    if collector_row.get("status") == "pending":
        execution_summary = stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
            config_path=config_path,
            output_root=stage26_8m_root,
            repo_root=repo_root,
            run_mode_override="run_phase",
            job_id_override=str(job["job_id"]),
            phase_override="collector",
            max_jobs_override=1,
        )
        stage26_8m_input_rejections = [str(item) for item in execution_summary.get("input_rejections") or []]
        stage26_8m_boundary_rejections = [str(item) for item in execution_summary.get("boundary_rejections") or []]
        execution_blocking_reason = _collector_only_execution_blocking_reason(execution_summary)
        collector_row = _collector_scan_row(loaded, stage26_8m_root, repo_root, str(job["job_id"]))
    collector_summary_path = collector_root / stage26_8m.stage26_1.SUMMARY_FILE
    collector_summary = _read_json_if_exists(collector_summary_path)
    return _scenario_result(
        scenario,
        stage26_8m_root,
        collector_root,
        collector_summary,
        collector_row=collector_row,
        generated_config=generated,
        execution_blocking_reason=execution_blocking_reason,
        stage26_8m_input_rejections=stage26_8m_input_rejections,
        stage26_8m_boundary_rejections=stage26_8m_boundary_rejections,
    )


def _scenario_stage26_8m_config(base: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    payload = dict(base)
    payload.update(
        {
            "schema_version": stage26_8m.CONFIG_SCHEMA_VERSION,
            "stage_id": stage26_8m.STAGE_ID,
            "run_mode": "run_next",
            "max_jobs_per_invocation": 1,
            "collector_reuse_policy": stage26_8m.COLLECTOR_REUSE_BY_HORIZON_SEED_SCENARIO_ROLLOUT,
            "scenario_counts": [int(scenario["scenario_count"])],
            "update_combos": [
                {
                    "combo_id": "collector_only_probe",
                    "epochs": 1,
                    "learning_rate": 1.0e-6,
                    "policy_loss_coefficient": 1.0,
                    "value_loss_coefficient": 0.01,
                    "entropy_coefficient": 0.005,
                    "loss_scale": 0.25,
                }
            ],
            "candidate_reachability_gate_source": GATE_SOURCE,
            "candidate_reachability_theta_proposal_policy": scenario["candidate_reachability_theta_proposal_policy"],
            "candidate_reachability_max_theta_proposals_per_candidate": int(
                scenario["candidate_reachability_max_theta_proposals_per_candidate"]
            ),
            "hybrid_astar_max_iterations": int(scenario["hybrid_astar_max_iterations"]),
            "stage26_8s_scenario_count": int(scenario["scenario_count"]),
            "release_or_training_authorized": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return payload


def _scenario_result(
    scenario: dict[str, Any],
    stage26_8m_root: Path,
    collector_root: Path,
    collector_summary: dict[str, Any],
    *,
    collector_row: dict[str, Any],
    generated_config: dict[str, Any],
    execution_blocking_reason: str = "",
    stage26_8m_input_rejections: list[str] | None = None,
    stage26_8m_boundary_rejections: list[str] | None = None,
) -> dict[str, Any]:
    validation_blocking_reason = (
        execution_blocking_reason
        or str(collector_row.get("blocking_reason") or "")
        or ("collector_phase_not_complete" if collector_row.get("status") != "complete" else "")
    )
    collector_valid = collector_row.get("status") == "complete" and not validation_blocking_reason
    planner_proxy_fields = _planner_proxy_fields(generated_config)
    result = {
        "schema_version": "xunce-stage26-8s-scenario-result/v1",
        "combo_id": scenario["combo_id"],
        "scenario_count": int(scenario["scenario_count"]),
        "status": collector_summary.get("status", "missing") if collector_valid else "failed",
        "stage26_8m_root": str(stage26_8m_root),
        "collector_root": str(collector_root),
        "collector_validation_status": "passed" if collector_valid else "failed",
        "collector_validation_blocking_reason": validation_blocking_reason,
        "stage26_8m_input_rejections": stage26_8m_input_rejections or [],
        "stage26_8m_boundary_rejections": stage26_8m_boundary_rejections or [],
        "collector_phase_config_hash": collector_row.get("config_hash"),
        "collector_input_hash": collector_row.get("input_hash"),
        "planner_proxy_fields": planner_proxy_fields,
        "hybrid_astar_planning_grid_source": planner_proxy_fields.get("hybrid_astar_planning_grid_source"),
        "planner_grid_resolution_m": planner_proxy_fields.get("planner_grid_resolution_m"),
        "hybrid_astar_closed_key_xy_resolution_m": planner_proxy_fields.get(
            "hybrid_astar_closed_key_xy_resolution_m"
        ),
        "candidate_reachability_gate_source": GATE_SOURCE,
        "candidate_reachability_theta_proposal_policy": scenario["candidate_reachability_theta_proposal_policy"],
        "candidate_reachability_max_theta_proposals_per_candidate": int(
            scenario["candidate_reachability_max_theta_proposals_per_candidate"]
        ),
        "hybrid_astar_max_iterations": int(scenario["hybrid_astar_max_iterations"]),
        "trainable_transition_count": _collector_trainable_transition_count(collector_summary),
        "transition_count": _int_value(collector_summary.get("transition_count")),
        "no_hybrid_astar_pose_reachable_candidate_for_action_mask_count": _int_value(
            collector_summary.get("stage21_1_no_hybrid_astar_pose_reachable_candidate_for_action_mask_count")
            or collector_summary.get("rejection_no_hybrid_astar_pose_reachable_candidate_for_action_mask_count")
        ),
        "selected_pose_unreachable_terminal_count": _int_value(
            collector_summary.get("stage21_1_selected_pose_unreachable_terminal_count")
        ),
        "scenario_early_terminal_step_histogram": collector_summary.get("stage21_1_scenario_early_terminal_step_histogram"),
        "trainable_transition_count_by_scenario": collector_summary.get("stage21_1_trainable_transition_count_by_scenario"),
    }
    for field in SAFETY_COUNT_FIELDS:
        result[field] = _int_value(collector_summary.get(field))
    result["safety_or_evidence_rejections"] = _safety_or_evidence_rejections(result)
    return result


def _collector_scan_row(config: dict[str, Any], output_root: Path, repo_root: Path, job_id: str) -> dict[str, Any]:
    rows = stage26_8m._scan_jobs(config=config, output_root=output_root, repo_root=repo_root)
    for row in rows:
        if str(row.get("job_id")) == job_id and row.get("phase") == "collector":
            return row
    return {"status": "missing", "blocking_reason": "collector_phase_state_missing"}


def _collector_only_execution_blocking_reason(summary: dict[str, Any]) -> str:
    if summary.get("input_rejections"):
        return "stage26_8m_input_rejections"
    if summary.get("boundary_rejections"):
        return "stage26_8m_boundary_rejections"
    executions = summary.get("phase_executions") or []
    if not executions:
        return "stage26_8m_executed_no_phases"
    if len(executions) > 1:
        return "stage26_8m_executed_multiple_phases"
    if any(str(row.get("phase") or "") != "collector" for row in executions):
        return "stage26_8m_executed_non_collector_phase"
    return ""


def _collector_trainable_transition_count(summary: dict[str, Any]) -> int:
    return _int_value(
        summary.get("trainable_transition_count")
        or summary.get("batch_row_count")
        or summary.get("transition_count")
    )


def _scenario_passes(result: dict[str, Any], *, min_trainable_transition_count: int) -> bool:
    return (
        result.get("status") == "passed"
        and int(result.get("trainable_transition_count") or 0) >= min_trainable_transition_count
        and not _has_safety_or_evidence_rejections(result)
    )


def _select_recommended_scenario(
    results: list[dict[str, Any]],
    *,
    min_trainable_transition_count: int,
) -> dict[str, Any]:
    for row in results:
        if _scenario_passes(row, min_trainable_transition_count=min_trainable_transition_count):
            return row
    return {}


def _recommended_payload(config: dict[str, Any], recommendation: dict[str, Any]) -> dict[str, Any]:
    planner_proxy_fields = dict(recommendation.get("planner_proxy_fields") or {})
    return {
        "schema_version": RECOMMENDED_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "recommended_scenario_count": int(recommendation["scenario_count"]),
        "planner_proxy_fields": planner_proxy_fields,
        "hybrid_astar_planning_grid_source": planner_proxy_fields.get("hybrid_astar_planning_grid_source"),
        "planner_grid_resolution_m": planner_proxy_fields.get("planner_grid_resolution_m"),
        "hybrid_astar_closed_key_xy_resolution_m": planner_proxy_fields.get(
            "hybrid_astar_closed_key_xy_resolution_m"
        ),
        "candidate_reachability_gate_source": GATE_SOURCE,
        "candidate_reachability_theta_proposal_policy": recommendation["candidate_reachability_theta_proposal_policy"],
        "candidate_reachability_max_theta_proposals_per_candidate": int(
            recommendation["candidate_reachability_max_theta_proposals_per_candidate"]
        ),
        "hybrid_astar_max_iterations": int(recommendation["hybrid_astar_max_iterations"]),
        "trainable_transition_count": int(recommendation["trainable_transition_count"]),
        "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
        "no_hybrid_astar_pose_reachable_candidate_for_action_mask_count": int(
            recommendation["no_hybrid_astar_pose_reachable_candidate_for_action_mask_count"]
        ),
        "selected_pose_unreachable_terminal_count": int(recommendation["selected_pose_unreachable_terminal_count"]),
        "collector_root": recommendation.get("collector_root"),
        "collector_phase_config_hash": recommendation.get("collector_phase_config_hash"),
        "collector_input_hash": recommendation.get("collector_input_hash"),
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _route(
    *,
    boundary_rejections: list[str],
    input_rejections: list[str],
    scenario_source_rejections: list[str],
    scenario_results: list[dict[str, Any]],
    recommendation: dict[str, Any],
) -> str:
    if boundary_rejections:
        return ROUTE_REPAIR
    if input_rejections:
        return ROUTE_INPUTS
    if scenario_source_rejections:
        return ROUTE_SCENARIO_SOURCE
    if any(row.get("stage26_8m_input_rejections") for row in scenario_results):
        return ROUTE_INPUTS
    if any(_has_safety_or_evidence_rejections(row) for row in scenario_results):
        return ROUTE_REPAIR
    if recommendation:
        return ROUTE_USE_RECOMMENDED
    return ROUTE_EXPAND_SAMPLE


def _safety_or_evidence_rejections(result: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if result.get("collector_validation_status") != "passed":
        reasons.append("collector_validation_failed")
    if result.get("status") != "passed":
        reasons.append("collector_summary_status_not_passed")
    if int(result.get("selected_pose_unreachable_terminal_count") or 0) > 0:
        reasons.append("selected_pose_unreachable_terminal_count")
    for field in SAFETY_COUNT_FIELDS:
        if int(result.get(field) or 0) > 0:
            reasons.append(field)
    return reasons


def _has_safety_or_evidence_rejections(result: dict[str, Any]) -> bool:
    return bool(result.get("safety_or_evidence_rejections") or _safety_or_evidence_rejections(result))


def _planner_proxy_fields(config: dict[str, Any]) -> dict[str, Any]:
    return {
        field: config[field]
        for field in stage26_8m.PLANNER_OVERRIDE_FIELDS
        if config.get(field) is not None
    }


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    if payload.get("stage_id") != STAGE_ID:
        raise ValueError(f"stage_id must be {STAGE_ID}")
    defaults = {
        "stage26_8n_root": stage26_8n.DEFAULT_OUTPUT_ROOT,
        "stage26_8n_aggressive_config_path": "",
        "min_trainable_transition_count": 100,
        "scenario_counts": [8, 10, 12],
        "candidate_reachability_theta_proposal_policy": REPAIR_POLICY,
        "candidate_reachability_max_theta_proposals_per_candidate": 5,
        "hybrid_astar_max_iterations": 200,
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    config["stage26_8n_root"] = str(_resolve_path(Path(str(config["stage26_8n_root"])), repo_root))
    if not config.get("stage26_8n_aggressive_config_path"):
        config["stage26_8n_aggressive_config_path"] = str(Path(config["stage26_8n_root"]) / stage26_8n.AGGRESSIVE_CONFIG_FILE)
    config["stage26_8n_aggressive_config_path"] = str(
        _resolve_path(Path(str(config["stage26_8n_aggressive_config_path"])), repo_root)
    )
    config["min_trainable_transition_count"] = _positive_int(
        config["min_trainable_transition_count"],
        "min_trainable_transition_count",
    )
    config["scenario_counts"] = _positive_int_list(config["scenario_counts"], "scenario_counts")
    policy = str(config.get("candidate_reachability_theta_proposal_policy") or REPAIR_POLICY)
    if policy not in {LEGACY_POLICY, REPAIR_POLICY}:
        raise ValueError("candidate_reachability_theta_proposal_policy is invalid")
    config["candidate_reachability_theta_proposal_policy"] = policy
    config["candidate_reachability_max_theta_proposals_per_candidate"] = _positive_int(
        config["candidate_reachability_max_theta_proposals_per_candidate"],
        "candidate_reachability_max_theta_proposals_per_candidate",
    )
    config["hybrid_astar_max_iterations"] = _positive_int(config["hybrid_astar_max_iterations"], "hybrid_astar_max_iterations")
    for field in BOUNDARY_FIELDS:
        config[field] = bool(config.get(field, False))
    config["canary_traffic_fraction"] = float(config.get("canary_traffic_fraction", 0.0))
    return config


def _scenario_probes(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _scenario_probe(
            count,
            policy=config["candidate_reachability_theta_proposal_policy"],
            cap=int(config["candidate_reachability_max_theta_proposals_per_candidate"]),
            iterations=int(config["hybrid_astar_max_iterations"]),
        )
        for count in config["scenario_counts"]
    ]


def _scenario_probe(
    scenario_count: int,
    *,
    policy: str = REPAIR_POLICY,
    cap: int = 5,
    iterations: int = 200,
) -> dict[str, Any]:
    count = _positive_int(scenario_count, "scenario_count")
    return {
        "combo_id": f"scenario_count_{count}",
        "scenario_count": count,
        "candidate_reachability_theta_proposal_policy": policy,
        "candidate_reachability_max_theta_proposals_per_candidate": _positive_int(
            cap,
            "candidate_reachability_max_theta_proposals_per_candidate",
        ),
        "hybrid_astar_max_iterations": _positive_int(iterations, "hybrid_astar_max_iterations"),
    }


def _input_rejections(config: dict[str, Any], base_8m_config: dict[str, Any], *, read_error: str | None) -> list[str]:
    reasons: list[str] = []
    path = Path(config["stage26_8n_aggressive_config_path"])
    if not artifact_io.path_is_file(path):
        return ["stage26_8n_aggressive_config_missing"]
    if read_error:
        return [read_error]
    if base_8m_config.get("schema_version") != stage26_8m.CONFIG_SCHEMA_VERSION:
        reasons.append("stage26_8n_aggressive_config_schema_version_mismatch")
    if base_8m_config.get("stage_id") != stage26_8m.STAGE_ID:
        reasons.append("stage26_8n_aggressive_config_stage_id_mismatch")
    return reasons


def _scenario_source_rejections(base_8m_config: dict[str, Any]) -> list[str]:
    if not base_8m_config:
        return []
    source_root = str(base_8m_config.get("source_scenario_fixture_root") or "").strip()
    if not source_root:
        return ["source_scenario_fixture_root_missing"]
    if not artifact_io.path_exists(Path(source_root)):
        return ["source_scenario_fixture_root_missing"]
    return []


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage26.8S Terminal-Aware Sample Expansion",
        "",
        f"- status: `{summary['status']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- recommended_scenario_count: `{summary.get('recommended_scenario_count')}`",
        f"- recommended_config: `{summary.get('recommended_config')}`",
        "",
        "## Scenario Results",
    ]
    for row in summary.get("scenario_results", []):
        lines.append(
            "- "
            f"scenario_count=`{row.get('scenario_count')}`: status=`{row.get('status')}`, "
            f"trainable=`{row.get('trainable_transition_count')}`, "
            f"no_hybrid_terminal=`{row.get('no_hybrid_astar_pose_reachable_candidate_for_action_mask_count')}`, "
            f"selected_pose_unreachable_terminal=`{row.get('selected_pose_unreachable_terminal_count')}`, "
            f"rejections=`{row.get('safety_or_evidence_rejections')}`"
        )
    lines.extend(
        [
            "",
            "This stage runs collector-only terminal-aware sample expansion probes. It does not run PPO update/eval, publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    return artifact_io.read_json(path)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return artifact_io.read_json(path) if artifact_io.path_is_file(path) else {}


def _read_json_if_exists_with_error(path: Path, reason: str) -> tuple[dict[str, Any], str | None]:
    if not artifact_io.path_is_file(path):
        return {}, None
    try:
        return artifact_io.read_json(path), None
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
        return {}, reason


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    artifact_io.write_json(path, payload)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    artifact_io.write_jsonl(path, rows)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _positive_int(value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive") from exc
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _positive_int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty array")
    return [_positive_int(item, name) for item in value]


def _int_value(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if math.isfinite(float(number)) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
