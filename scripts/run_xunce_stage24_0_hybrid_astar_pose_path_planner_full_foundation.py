from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PATH_PLANNER_SRC = REPO_ROOT / "path-planner" / "src"
if str(PATH_PLANNER_SRC) not in sys.path:
    sys.path.insert(0, str(PATH_PLANNER_SRC))

from path_planner.core import Cell, CostGrid, GridSpec, PlanRequest  # noqa: E402
from path_planner.search import AStarPlanner, HybridAStarPlanner, Pose2D, PosePlanRequest  # noqa: E402


STAGE_ID = "xunce-stage24-0-hybrid-astar-pose-path-planner-full-foundation"
CONFIG_SCHEMA_VERSION = "xunce-stage24-0-hybrid-astar-pose-path-planner-full-foundation-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage24-0-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage24-0-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage24-0-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage24_0_hybrid_astar_pose_path_planner_full_foundation_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage24_hybrid_astar_pose_planner/"
    "outputs/path_feedback_batch_xunce_stage24_0_hybrid_astar_pose_path_planner_full_foundation_v1"
)

SUMMARY_FILE = "xunce-stage24-0-summary.json"
SMOKE_FILE = "xunce-stage24-0-hybrid-planner-smoke.jsonl"
GRID_VS_HYBRID_FILE = "xunce-stage24-0-grid-vs-hybrid-audit.json"
FOOTPRINT_FILE = "xunce-stage24-0-footprint-collision-audit.json"
ROUTING_FILE = "xunce-stage24-0-next-stage-routing.json"
REPORT_FILE = "xunce-stage24-0-report.md"
MANIFEST_FILE = "xunce-stage24-0-manifest.json"

ROUTE_INPUTS = "rerun_stage24_0_required_inputs"
ROUTE_CONTRACT = "repair_stage24_0_hybrid_astar_contract"
ROUTE_PRIMITIVES = "repair_stage24_0_motion_primitives_or_heuristic"
ROUTE_STAGE24_1 = "run_stage24_1_hybrid_astar_candidate_path_cost_integration"
ROUTE_BOUNDARY = "resolve_stage24_0_boundary_rejections"

BOUNDARY_FIELDS = (
    "stage24_0_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage24.0 Hybrid A* pose planner foundation smoke.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage24_0_hybrid_astar_pose_path_planner_full_foundation(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage24_0_hybrid_astar_pose_path_planner_full_foundation(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(config_path, repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    platform_contract, platform_hash = _load_platform_contract(config["platform_contract"], repo_root)
    boundary_reasons = _boundary_rejections(config, platform_contract)
    input_reasons = _input_rejections(config, platform_contract)

    smoke_rows: list[dict[str, Any]] = []
    if not boundary_reasons and not input_reasons:
        smoke_rows = [_run_case(case, config, platform_contract) for case in _smoke_cases()]

    grid_vs_hybrid = _grid_vs_hybrid_audit(smoke_rows)
    footprint_audit = _footprint_audit(smoke_rows)
    status, route, route_reason = _route(boundary_reasons, input_reasons, smoke_rows, footprint_audit)

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": [*boundary_reasons, *input_reasons, *_smoke_blockers(smoke_rows, footprint_audit)],
        "stage23_6_root": config.get("stage23_6_root"),
        "platform_contract_id": platform_contract.get("platform_id"),
        "platform_contract_hash": platform_hash,
        "platform_model": "differential_skid_steer",
        "ackermann_feasible_claimed": False,
        "theta_bin_count": int(config["theta_bin_count"]),
        "theta_bin_width_deg": 360.0 / float(config["theta_bin_count"]),
        "goal_theta_tolerance_deg": float(config["goal_theta_tolerance_deg"]),
        "platform_max_climb_deg": float(platform_contract["base"]["platform_max_climb_deg"]),
        "max_traversable_slope_deg": float(platform_contract["terrain_policy"]["max_traversable_slope_deg"]),
        "smoke_case_count": len(smoke_rows),
        "successful_required_case_count": sum(1 for row in smoke_rows if row["expect_success"] and row["hybrid_success"]),
        "expected_blocked_case_count": sum(1 for row in smoke_rows if (not row["expect_success"]) and (not row["hybrid_success"])),
        "pose_path_case_count": sum(1 for row in smoke_rows if row["hybrid_success"] and row["hybrid_pose_path_count"] > 0),
        "turn_in_place_supported": any("turn_in_place" in name for row in smoke_rows for name in row["hybrid_primitive_names"]),
        "footprint_collision_audit_passed": footprint_audit["footprint_collision_audit_passed"],
        "obstacle_source_binding_audit_passed": footprint_audit["obstacle_source_binding_audit_passed"],
        "default_astar_replaced": False,
        "stage24_0_authorized": False,
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
        "stage24_0_authorized": False,
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
        "hybrid_planner_smoke": str(output_root / SMOKE_FILE),
        "grid_vs_hybrid_audit": str(output_root / GRID_VS_HYBRID_FILE),
        "footprint_collision_audit": str(output_root / FOOTPRINT_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }

    _write_json(output_root / SUMMARY_FILE, summary)
    _write_jsonl(output_root / SMOKE_FILE, smoke_rows)
    _write_json(output_root / GRID_VS_HYBRID_FILE, grid_vs_hybrid)
    _write_json(output_root / FOOTPRINT_FILE, footprint_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary, grid_vs_hybrid), encoding="utf-8")
    return summary


def _run_case(case: dict[str, Any], config: dict[str, Any], platform_contract: dict[str, Any]) -> dict[str, Any]:
    passable = np.asarray(case["passable_mask"], dtype=bool)
    declared_obstacles = _declared_obstacle_cells(case)
    for x, y in declared_obstacles:
        if 0 <= y < passable.shape[0] and 0 <= x < passable.shape[1]:
            passable[y, x] = False
    cost = np.ones(passable.shape, dtype=float)
    grid = CostGrid(GridSpec(width=passable.shape[1], height=passable.shape[0], resolution=1.0), cost, passable)
    planner = HybridAStarPlanner()
    start_pose = Pose2D(*case["start_pose"])
    goal_pose = Pose2D(*case["goal_pose"])
    pose_request = PosePlanRequest(
        start=start_pose,
        goal=goal_pose,
        theta_bin_count=int(config["theta_bin_count"]),
        theta_tolerance_rad=math.radians(float(config["goal_theta_tolerance_deg"])),
        max_iterations=int(config["max_iterations"]),
        primitive_duration_s=float(config["primitive_duration_s"]),
        integration_dt_s=float(config["integration_dt_s"]),
        max_speed_mps=float(config["max_speed_mps"]),
        max_angular_speed_radps=math.radians(float(config["max_angular_speed_degps"])),
        footprint_length_m=float(platform_contract["base"]["body_length_m"]),
        footprint_width_m=float(platform_contract["base"]["body_width_m"]),
        footprint_safety_margin_m=float(config["footprint_safety_margin_m"]),
        rotation_cost_weight=float(config["rotation_cost_weight"]),
        reverse_penalty_weight=float(config["reverse_penalty_weight"]),
        turn_penalty_weight=float(config["turn_penalty_weight"]),
    )
    hybrid = planner.plan(grid, pose_request)
    grid_start = grid.spec.world_to_cell(type("Point", (), {"x": start_pose.x_m, "y": start_pose.y_m})())
    grid_goal = grid.spec.world_to_cell(type("Point", (), {"x": goal_pose.x_m, "y": goal_pose.y_m})())
    grid_result = AStarPlanner().plan(grid, PlanRequest(start=grid_start, goal=grid_goal))
    route = hybrid.to_route_dict(grid.spec)
    legacy_path = [cell.to_list() for cell in hybrid.legacy_cell_path]
    legacy_intersections = sorted([cell for cell in legacy_path if tuple(cell) in declared_obstacles])
    pose_intersections = _pose_obstacle_intersections(grid.spec, hybrid.pose_path, declared_obstacles)
    return {
        "scenario_id": case["scenario_id"],
        "expect_success": bool(case["expect_success"]),
        "expected_failure_reason": case.get("expected_failure_reason"),
        "hybrid_success": bool(hybrid.success),
        "hybrid_failure_reason": hybrid.failure_reason.value if hybrid.failure_reason else None,
        "hybrid_trajectory_kind": route["trajectory_kind"],
        "hybrid_pose_path_count": len(hybrid.pose_path),
        "hybrid_control_count": len(hybrid.control_sequence),
        "hybrid_legacy_cell_path": legacy_path,
        "hybrid_primitive_names": [primitive.name for primitive in hybrid.control_sequence],
        "hybrid_total_cost": None if not hybrid.success else float(hybrid.total_cost),
        "hybrid_cost_breakdown": hybrid.cost_breakdown.to_dict(),
        "hybrid_expanded_pose_count": int(hybrid.expanded_count),
        "grid_astar_success": bool(grid_result.success),
        "grid_astar_total_cost": None if not grid_result.success else float(grid_result.total_cost),
        "grid_astar_path_cell_count": len(grid_result.path_cells),
        "grid_astar_trajectory_kind": grid_result.to_route_dict(grid.spec)["trajectory_kind"],
        "platform_model": hybrid.diagnostics.platform_model,
        "ackermann_feasible_claimed": hybrid.diagnostics.ackermann_feasible_claimed,
        "dominance_key_policy": hybrid.diagnostics.to_dict()["dominance_key_policy"],
        "footprint_length_m": hybrid.diagnostics.footprint_length_m,
        "footprint_width_m": hybrid.diagnostics.footprint_width_m,
        "hard_obstacle_sources": list(hybrid.diagnostics.hard_obstacle_sources),
        "case_obstacle_source_kind": case.get("obstacle_source_kind", "none"),
        "case_obstacle_cells": case.get("obstacle_cells", []),
        "case_slope_blocked_cells": case.get("slope_blocked_cells", []),
        "case_blocked_cells": case.get("blocked_cells", []),
        "declared_obstacle_cells": [list(cell) for cell in sorted(declared_obstacles)],
        "legacy_path_declared_obstacle_intersections": legacy_intersections,
        "pose_path_declared_obstacle_intersections": pose_intersections,
    }


def _smoke_cases() -> list[dict[str, Any]]:
    open5 = [[True for _ in range(5)] for _ in range(5)]
    detour = [[True for _ in range(7)] for _ in range(5)]
    detour[1][3] = False
    detour[2][3] = False
    detour[3][3] = False
    blocked_goal = [[True for _ in range(5)] for _ in range(5)]
    blocked_goal[2][2] = False
    return [
        {
            "scenario_id": "turn_in_place_to_goal_theta",
            "passable_mask": open5,
            "start_pose": [2.0, 2.0, 0.0],
            "goal_pose": [2.0, 2.0, math.pi / 2.0],
            "expect_success": True,
            "obstacle_source_kind": "none",
        },
        {
            "scenario_id": "straight_pose_path_to_viewpoint",
            "passable_mask": open5,
            "start_pose": [1.0, 1.0, 0.0],
            "goal_pose": [4.0, 1.0, 0.0],
            "expect_success": True,
            "obstacle_source_kind": "none",
        },
        {
            "scenario_id": "slope_obstacle_proxy_detour",
            "passable_mask": detour,
            "start_pose": [1.0, 2.0, 0.0],
            "goal_pose": [5.0, 2.0, 0.0],
            "expect_success": True,
            "obstacle_source_kind": "slope_blocked_as_obstacle_proxy",
            "slope_blocked_cells": [[3, 1], [3, 2], [3, 3]],
        },
        {
            "scenario_id": "footprint_rejects_blocked_goal",
            "passable_mask": blocked_goal,
            "start_pose": [1.0, 1.0, 0.0],
            "goal_pose": [2.0, 2.0, 0.0],
            "expect_success": False,
            "expected_failure_reason": "goal_blocked",
            "obstacle_source_kind": "blocked_as_obstacle_proxy",
            "blocked_cells": [[2, 2]],
        },
    ]


def _grid_vs_hybrid_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success_rows = [row for row in rows if row["hybrid_success"]]
    return {
        "schema_version": "xunce-stage24-0-grid-vs-hybrid-audit/v1",
        "case_count": len(rows),
        "hybrid_success_count": len(success_rows),
        "grid_astar_success_count": sum(1 for row in rows if row["grid_astar_success"]),
        "hybrid_pose_path_kind_count": sum(1 for row in rows if row["hybrid_success"] and row["hybrid_pose_path_count"] > 0),
        "grid_astar_trajectory_kind": "geometric_path",
        "hybrid_uses_pose_state": all(
            (not row["hybrid_success"]) or (row["hybrid_trajectory_kind"] == "hybrid_astar_pose_path" and row["hybrid_pose_path_count"] > 0)
            for row in rows
        ),
        "dominance_key_policy": sorted({row["dominance_key_policy"] for row in rows}),
        "cost_delta_by_case": [
            {
                "scenario_id": row["scenario_id"],
                "hybrid_total_cost": row["hybrid_total_cost"],
                "grid_astar_total_cost": row["grid_astar_total_cost"],
            }
            for row in rows
        ],
    }


def _footprint_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    blocked = [row for row in rows if row["scenario_id"] == "footprint_rejects_blocked_goal"]
    slope = [row for row in rows if row["scenario_id"] == "slope_obstacle_proxy_detour"]
    passed = bool(blocked and blocked[0]["hybrid_failure_reason"] == "goal_blocked")
    no_declared_intersections = all(
        not row["legacy_path_declared_obstacle_intersections"] and not row["pose_path_declared_obstacle_intersections"]
        for row in rows
        if row["hybrid_success"]
    )
    source_passed = bool(
        blocked
        and blocked[0]["case_blocked_cells"]
        and blocked[0]["case_obstacle_source_kind"] == "blocked_as_obstacle_proxy"
        and slope
        and slope[0]["case_slope_blocked_cells"]
        and slope[0]["case_obstacle_source_kind"] == "slope_blocked_as_obstacle_proxy"
    )
    return {
        "schema_version": "xunce-stage24-0-footprint-collision-audit/v1",
        "footprint_collision_audit_passed": passed,
        "declared_obstacle_hard_gate_passed": no_declared_intersections,
        "obstacle_source_binding_audit_passed": source_passed,
        "blocked_goal_failure_reason": blocked[0]["hybrid_failure_reason"] if blocked else None,
        "uses_rectangular_footprint": bool(blocked and blocked[0]["footprint_length_m"] > 0 and blocked[0]["footprint_width_m"] > 0),
        "hard_obstacle_sources": blocked[0]["hard_obstacle_sources"] if blocked else [],
        "blocked_source_kind": blocked[0]["case_obstacle_source_kind"] if blocked else None,
        "slope_source_kind": slope[0]["case_obstacle_source_kind"] if slope else None,
        "slope_blocked_cell_count": len(slope[0]["case_slope_blocked_cells"]) if slope else 0,
    }


def _route(
    boundary_reasons: list[str],
    input_reasons: list[str],
    rows: list[dict[str, Any]],
    footprint_audit: dict[str, Any],
) -> tuple[str, str, str]:
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "release/default-policy/executor/canary boundary is open"
    if input_reasons:
        return "failed", ROUTE_INPUTS, "required platform or Stage23 inputs are missing"
    if not footprint_audit.get("footprint_collision_audit_passed"):
        return "failed", ROUTE_CONTRACT, "rectangular footprint collision contract failed"
    if not footprint_audit.get("declared_obstacle_hard_gate_passed"):
        return "failed", ROUTE_CONTRACT, "declared obstacle hard gate contract failed"
    if not footprint_audit.get("obstacle_source_binding_audit_passed"):
        return "failed", ROUTE_CONTRACT, "obstacle source binding contract failed"
    failures = [row for row in rows if row["expect_success"] and not row["hybrid_success"]]
    if failures:
        return "failed", ROUTE_PRIMITIVES, "Hybrid A* could not solve required smoke scenarios"
    if any(row["hybrid_trajectory_kind"] != "hybrid_astar_pose_path" for row in rows):
        return "failed", ROUTE_CONTRACT, "Hybrid A* did not emit pose-path trajectory kind"
    return "passed", ROUTE_STAGE24_1, "Hybrid A* pose planner foundation smoke passed"


def _boundary_rejections(config: dict[str, Any], platform_contract: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for source_name, source in (("config", config), ("platform_contract", platform_contract.get("boundary", {}))):
        for field in BOUNDARY_FIELDS:
            value = source.get(field)
            if field == "stage24_0_authorized":
                continue
            if value not in (False, 0, 0.0, None):
                reasons.append(f"{source_name}_{field}_enabled")
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        reasons.append("config_canary_traffic_fraction_nonzero")
    return reasons


def _input_rejections(config: dict[str, Any], platform_contract: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if platform_contract.get("platform_id") != "agilex_scout_mini_piper":
        reasons.append("platform_contract_id_mismatch")
    if platform_contract.get("base", {}).get("steering_type") != "differential_skid_steer":
        reasons.append("platform_steering_type_not_differential_skid_steer")
    if float(platform_contract.get("base", {}).get("min_turning_radius_m", -1.0)) != 0.0:
        reasons.append("platform_min_turning_radius_not_zero")
    if float(platform_contract.get("terrain_policy", {}).get("max_traversable_slope_deg", 0.0)) != 30.0:
        reasons.append("platform_slope_threshold_not_30deg")
    stage23_6_root = _resolve_optional_path(config.get("stage23_6_root"))
    if stage23_6_root is not None:
        summary_path = stage23_6_root / "xunce-stage23-6-summary.json"
        if not summary_path.is_file():
            reasons.append("stage23_6_summary_missing")
        else:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("coverage_source") != "endpoint_theta_slope_obstacle_los/v1":
                reasons.append("stage23_6_coverage_source_mismatch")
            if float(summary.get("max_traversable_slope_deg", 0.0) or 0.0) != 30.0:
                reasons.append("stage23_6_slope_threshold_mismatch")
            if summary.get("next_required_change") != "calibrate_stage23_slope_theta_discrete_margin_crossing":
                reasons.append("stage23_6_route_not_discrete_margin")
            for field in ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary"):
                if summary.get(field) not in (False, 0, 0.0, None):
                    reasons.append(f"stage23_6_{field}_enabled")
    return reasons


def _smoke_blockers(rows: list[dict[str, Any]], footprint_audit: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not footprint_audit.get("footprint_collision_audit_passed"):
        reasons.append("rectangular_footprint_collision_audit_failed")
    if not footprint_audit.get("declared_obstacle_hard_gate_passed"):
        reasons.append("declared_obstacle_hard_gate_failed")
    if not footprint_audit.get("obstacle_source_binding_audit_passed"):
        reasons.append("obstacle_source_binding_audit_failed")
    for row in rows:
        if row["expect_success"] and not row["hybrid_success"]:
            reasons.append(f"{row['scenario_id']}_hybrid_astar_failed")
        if row["hybrid_trajectory_kind"] != "hybrid_astar_pose_path":
            reasons.append(f"{row['scenario_id']}_missing_hybrid_pose_path_kind")
        if row["ackermann_feasible_claimed"]:
            reasons.append(f"{row['scenario_id']}_ackermann_claimed")
    return reasons


def _declared_obstacle_cells(case: dict[str, Any]) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for field in ("obstacle_cells", "slope_blocked_cells", "blocked_cells"):
        for cell in case.get(field, []) or []:
            if isinstance(cell, (list, tuple)) and len(cell) >= 2:
                result.add((int(cell[0]), int(cell[1])))
    return result


def _pose_obstacle_intersections(spec: GridSpec, poses: tuple[Pose2D, ...], obstacles: set[tuple[int, int]]) -> list[list[int]]:
    result: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for pose in poses:
        cell = spec.world_to_cell(type("Point", (), {"x": pose.x_m, "y": pose.y_m})())
        key = (cell.x, cell.y)
        if key in obstacles and key not in seen:
            result.append([cell.x, cell.y])
            seen.add(key)
    return result


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    resolved = _resolve_path(path, repo_root)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"config schema_version must be {CONFIG_SCHEMA_VERSION}")
    return payload


def _load_platform_contract(path_value: str, repo_root: Path) -> tuple[dict[str, Any], str]:
    path = _resolve_path(Path(path_value), repo_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return payload, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _resolve_optional_path(value: Any) -> Path | None:
    if not value:
        return None
    return Path(str(value))


def _render_report(summary: dict[str, Any], audit: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage24.0 Hybrid A* Pose Planner Foundation",
            "",
            f"- status: {summary['status']}",
            f"- next_required_change: {summary['next_required_change']}",
            f"- platform_model: {summary['platform_model']}",
            f"- theta_bin_count: {summary['theta_bin_count']}",
            f"- max_traversable_slope_deg: {summary['max_traversable_slope_deg']}",
            f"- hybrid_success_count: {audit['hybrid_success_count']}/{audit['case_count']}",
            f"- footprint_collision_audit_passed: {summary['footprint_collision_audit_passed']}",
            f"- default_astar_replaced: {summary['default_astar_replaced']}",
            "",
            "Stage24.0 only proves the Hybrid A* pose planner foundation smoke; it does not publish checkpoints or replace the default policy.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
