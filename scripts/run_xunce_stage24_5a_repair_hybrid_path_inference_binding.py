from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke as stage24_5
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke as stage24_5


CONFIG_SCHEMA_VERSION = "xunce-stage24-5a-repair-hybrid-path-inference-binding-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage24-5a-summary/v1"
AUDIT_SCHEMA_VERSION = "xunce-stage24-5a-inference-binding-repair-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage24-5a-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage24-5a-manifest/v1"

STAGE_ID = "xunce-stage24-5a-repair-hybrid-path-inference-binding"
DEFAULT_CONFIG = "configs/xunce_stage24_5a_repair_hybrid_path_inference_binding_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage24_hybrid_astar_pose_planner/"
    "outputs/path_feedback_batch_xunce_stage24_5a_repair_hybrid_path_inference_binding_v1"
)
DEFAULT_STAGE24_5_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage24_hybrid_astar_pose_planner/"
    "outputs/path_feedback_batch_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke_v1"
)

SUMMARY_FILE = "xunce-stage24-5a-summary.json"
AUDIT_FILE = "xunce-stage24-5a-inference-binding-repair-audit.json"
RERUN_SUMMARY_FILE = "xunce-stage24-5a-rerun-stage24-5-summary.json"
ROUTING_FILE = "xunce-stage24-5a-next-stage-routing.json"
REPORT_FILE = "xunce-stage24-5a-report.md"
MANIFEST_FILE = "xunce-stage24-5a-manifest.json"
REPAIRED_STAGE24_5_CONFIG_FILE = "xunce-stage24-5a-repaired-stage24-5-config.json"

HYBRID_ASTAR_PATH_COST_SOURCE = "hybrid_astar_pose_path/v1"
SLOPE_COVERAGE_SOURCE = "endpoint_theta_slope_obstacle_los/v1"
ROUTE_INPUTS = "rerun_stage24_5a_required_inputs"
ROUTE_CONTINUE_BINDING = "continue_stage24_5_hybrid_path_inference_binding_repair"
ROUTE_PLANNER = "repair_stage24_1_hybrid_astar_candidate_planner_contract"
ROUTE_BINDING = "repair_stage24_5_hybrid_path_inference_binding"
ROUTE_SIGNAL = "repair_stage24_hybrid_path_policy_update_signal_strength"
ROUTE_BOUNDARY = "resolve_stage24_5a_boundary_rejections"
ROUTE_FROM_STAGE24_5 = "repair_stage24_5_hybrid_path_inference_binding"

BOUNDARY_FIELDS = (
    "stage24_5a_authorized",
    "release_or_training_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair Stage24.5 Hybrid A* path-cost inference binding.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    summary = run_xunce_stage24_5a_repair_hybrid_path_inference_binding(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage24_5a_repair_hybrid_path_inference_binding(
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
    original_root = Path(config["stage24_5_root"])
    original_summary = _read_json_if_exists(original_root / stage24_5.SUMMARY_FILE)
    input_reasons = _input_rejections(original_summary)

    repaired_summary: dict[str, Any] = {}
    repaired_config_path = output_root / REPAIRED_STAGE24_5_CONFIG_FILE
    repaired_root = output_root / "s24_5_repaired"
    if not boundary_reasons and not input_reasons:
        repaired_config = _repaired_stage24_5_config(config, repo_root)
        _write_json(repaired_config_path, repaired_config)
        repaired_summary = stage24_5.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke(
            config_path=repaired_config_path,
            output_root=repaired_root,
            repo_root=repo_root,
        )

    audit = _binding_repair_audit(original_summary, repaired_summary, repaired_root)
    status, route, route_reason = _route(boundary_reasons, input_reasons, audit, repaired_summary)
    blocking = _unique(boundary_reasons + input_reasons + _route_blocking_reasons(route, audit))

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": blocking,
        "stage24_5_root": str(original_root),
        "stage24_5_status": original_summary.get("status"),
        "stage24_5_next_required_change": original_summary.get("next_required_change"),
        "repaired_stage24_5_root": str(repaired_root),
        "repaired_stage24_5_status": repaired_summary.get("status"),
        "repaired_stage24_5_next_required_change": repaired_summary.get("next_required_change"),
        "strong_state_join_available_count": audit["repaired_strong_state_join_available_count"],
        "hybrid_path_inference_required_field_missing_count": audit["repaired_hybrid_path_inference_required_field_missing_count"],
        "path_cost_source_mismatch_count": audit["repaired_path_cost_source_mismatch_count"],
        "grid_fallback_count": audit["repaired_grid_fallback_count"],
        "default_astar_replaced_count": audit["repaired_default_astar_replaced_count"],
        "ackermann_feasible_claimed_count": audit["repaired_ackermann_feasible_claimed_count"],
        "coverage_source": SLOPE_COVERAGE_SOURCE,
        "path_cost_source": HYBRID_ASTAR_PATH_COST_SOURCE,
        "default_astar_replaced": False,
        "ackermann_feasible_claimed": False,
        "release_or_training_authorized": False,
        "stage24_5a_authorized": False,
        "runs_new_ppo_update": False,
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
        "stage24_5a_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "summary": str(output_root / SUMMARY_FILE),
        "inference_binding_repair_audit": str(output_root / AUDIT_FILE),
        "rerun_stage24_5_summary": str(output_root / RERUN_SUMMARY_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
        "repaired_stage24_5_config": str(repaired_config_path),
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / AUDIT_FILE, audit)
    _write_json(output_root / RERUN_SUMMARY_FILE, repaired_summary)
    _write_json(output_root / ROUTING_FILE, routing)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    _write_json(output_root / MANIFEST_FILE, manifest)
    return summary


def _repaired_stage24_5_config(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(Path(config["stage24_5_base_config"]), repo_root))
    payload.update(
        {
            "hybrid_astar_pose_path_cost_enabled": True,
            "coverage_source": SLOPE_COVERAGE_SOURCE,
            "path_cost_source": HYBRID_ASTAR_PATH_COST_SOURCE,
            "max_traversable_slope_deg": 30.0,
            "hybrid_astar_theta_bin_count": 72,
            "hybrid_astar_goal_position_tolerance_m": 2.0,
            "hybrid_astar_goal_theta_tolerance_deg": 5.0,
            "hybrid_astar_max_iterations": 100_000,
            "hybrid_astar_primitive_duration_s": 1.5,
            "hybrid_astar_integration_dt_s": 0.25,
            "hybrid_astar_max_speed_mps": 3.0,
            "hybrid_astar_max_angular_speed_degps": 45.0,
            "hybrid_astar_rotation_cost_weight": 0.2,
            "hybrid_astar_reverse_penalty_weight": 0.5,
            "hybrid_astar_turn_penalty_weight": 0.05,
            "default_astar_replaced": False,
            "hybrid_astar_ackermann_feasible_claimed": False,
            "stage24_5_authorized": False,
            "release_or_training_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    for key in (
        "stage24_4_root",
        "stage21_5_base_config",
        "high_fidelity_config",
        "required_scenario_count",
        "rollout_steps",
        "dynamic_max_candidates_per_step",
        "dynamic_proposal_pool_limit_per_step",
        "theta_bin_count",
        "theta_step_deg",
        "sensor_model_id",
        "sensor_fov_deg",
        "sensor_range_cells",
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
        "min_mean_abs_probability_delta_for_signal",
    ):
        if key in config:
            payload[key] = config[key]
    return payload


def _binding_repair_audit(
    original_summary: dict[str, Any],
    repaired_summary: dict[str, Any],
    repaired_root: Path,
) -> dict[str, Any]:
    action_audit = _read_json_if_exists(repaired_root / stage24_5.ACTION_AUDIT_FILE)
    def value(name: str) -> int:
        return _int_value(repaired_summary.get(name, action_audit.get(name)))

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "original_status": original_summary.get("status"),
        "original_next_required_change": original_summary.get("next_required_change"),
        "original_hybrid_path_inference_required_field_missing_count": _int_value(
            original_summary.get("hybrid_path_inference_required_field_missing_count")
        ),
        "original_path_cost_source_mismatch_count": _int_value(original_summary.get("path_cost_source_mismatch_count")),
        "repaired_status": repaired_summary.get("status"),
        "repaired_next_required_change": repaired_summary.get("next_required_change"),
        "repaired_strong_state_join_available_count": value("strong_state_join_available_count"),
        "repaired_hybrid_path_inference_required_field_missing_count": value(
            "hybrid_path_inference_required_field_missing_count"
        ),
        "repaired_path_cost_source_mismatch_count": value("path_cost_source_mismatch_count"),
        "repaired_grid_fallback_count": value("grid_fallback_count"),
        "repaired_default_astar_replaced_count": value("default_astar_replaced_count"),
        "repaired_ackermann_feasible_claimed_count": value("ackermann_feasible_claimed_count"),
        "repaired_model_inference_root": str(repaired_root),
        "binding_repaired": bool(
            value("strong_state_join_available_count") > 0
            and value("hybrid_path_inference_required_field_missing_count") == 0
            and value("path_cost_source_mismatch_count") == 0
            and value("grid_fallback_count") == 0
            and value("default_astar_replaced_count") == 0
            and value("ackermann_feasible_claimed_count") == 0
        ),
    }


def _route(
    boundary_reasons: list[str],
    input_reasons: list[str],
    audit: dict[str, Any],
    repaired_summary: dict[str, Any],
) -> tuple[str, str, str]:
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "boundary flag opened"
    if input_reasons:
        return "failed", ROUTE_INPUTS, "required Stage24.5 input is missing or not on binding route"
    if _int_value(audit.get("repaired_strong_state_join_available_count")) <= 0:
        return "failed", ROUTE_BINDING, "repaired Stage24.5 strong join unavailable"
    if (
        _int_value(audit.get("repaired_hybrid_path_inference_required_field_missing_count")) > 0
        or _int_value(audit.get("repaired_path_cost_source_mismatch_count")) > 0
    ):
        return "failed", ROUTE_CONTINUE_BINDING, "Hybrid path fields are still missing or mismatched"
    if (
        _int_value(audit.get("repaired_grid_fallback_count")) > 0
        or _int_value(audit.get("repaired_default_astar_replaced_count")) > 0
        or _int_value(audit.get("repaired_ackermann_feasible_claimed_count")) > 0
    ):
        return "failed", ROUTE_PLANNER, "Hybrid A* provenance flags are not clean"
    route = str(repaired_summary.get("next_required_change") or ROUTE_SIGNAL)
    return "passed", route, "Hybrid path inference binding repaired; preserving repaired Stage24.5 route"


def _route_blocking_reasons(route: str, audit: dict[str, Any]) -> list[str]:
    if route == ROUTE_CONTINUE_BINDING:
        return ["hybrid_path_inference_binding_still_missing"]
    if route == ROUTE_BINDING:
        return ["repaired_strong_state_join_unavailable"]
    if route == ROUTE_PLANNER:
        return ["hybrid_path_provenance_flags_not_clean"]
    return []


def _input_rejections(stage24_5_summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not stage24_5_summary:
        reasons.append("stage24_5_summary_missing")
        return reasons
    if stage24_5_summary.get("next_required_change") != ROUTE_FROM_STAGE24_5:
        reasons.append("stage24_5_route_not_hybrid_path_inference_binding")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in BOUNDARY_FIELDS:
        value = config.get(field, False)
        if field == "canary_traffic_fraction":
            continue
        if value is True:
            reasons.append(field)
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    payload = dict(payload)
    payload.setdefault("stage24_5_root", DEFAULT_STAGE24_5_ROOT)
    payload.setdefault("stage24_5_base_config", stage24_5.DEFAULT_CONFIG)
    for key in ("stage24_5_root", "stage24_5_base_config"):
        payload[key] = str(_resolve_path(Path(str(payload[key])), repo_root))
    for key in BOUNDARY_FIELDS:
        payload.setdefault(key, False)
    payload.setdefault("canary_traffic_fraction", 0.0)
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = _read_json(path)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _unique(values: list[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage24.5A Hybrid A* Path-Cost Inference Binding Repair",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- repaired_stage24_5_root: `{summary['repaired_stage24_5_root']}`",
            f"- strong_state_join_available_count: `{summary['strong_state_join_available_count']}`",
            f"- hybrid_path_inference_required_field_missing_count: `{summary['hybrid_path_inference_required_field_missing_count']}`",
            f"- path_cost_source_mismatch_count: `{summary['path_cost_source_mismatch_count']}`",
            f"- grid_fallback_count: `{summary['grid_fallback_count']}`",
            "",
            "This stage repairs offline inference binding only. It does not run PPO training, publish checkpoints, replace the default policy, connect a real executor, or start a canary.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
