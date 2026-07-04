from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PATH_PLANNER_SRC = REPO_ROOT / "path-planner" / "src"
if str(PATH_PLANNER_SRC) not in sys.path:
    sys.path.insert(0, str(PATH_PLANNER_SRC))

from path_planner.core import Cell, CostGrid, GridSpec, PlanRequest, WorldPoint  # noqa: E402
from path_planner.search import AStarPlanner, HybridAStarPlanner, Pose2D, PosePlanRequest  # noqa: E402


PATH_COST_SOURCE = "hybrid_astar_pose_path/v1"


def evaluate_hybrid_astar_candidate_path_cost(
    *,
    grid: CostGrid,
    current_pose: list[float] | tuple[float, float, float],
    candidate: dict[str, Any],
    platform_contract_hash: str,
    max_traversable_slope_deg: float,
    theta_bin_count: int = 72,
    goal_position_tolerance_m: float | None = None,
    goal_theta_tolerance_deg: float = 5.0,
    max_iterations: int = 100_000,
    primitive_duration_s: float = 1.0,
    integration_dt_s: float = 0.25,
    max_speed_mps: float = 1.0,
    max_angular_speed_degps: float = 45.0,
    footprint_length_m: float = 0.612,
    footprint_width_m: float = 0.580,
    footprint_safety_margin_m: float = 0.0,
    rotation_cost_weight: float = 0.2,
    reverse_penalty_weight: float = 0.5,
    turn_penalty_weight: float = 0.05,
    closed_key_xy_resolution_m: float | None = None,
) -> dict[str, Any]:
    base = _base_row(candidate, platform_contract_hash, max_traversable_slope_deg)
    pose = _parse_current_pose(current_pose)
    viewpoint = _parse_viewpoint(candidate.get("candidate_viewpoint"))
    goal_world_pose = _parse_goal_world_pose(candidate.get("candidate_goal_world_pose"))
    theta_deg = _finite_float(candidate.get("candidate_theta_deg"))
    if pose is None:
        return {**base, **_failed_contract("current_pose_missing")}
    if viewpoint is None or theta_deg is None:
        return {**base, **_failed_contract("candidate_viewpoint_missing")}
    if abs(_angle_delta_deg(float(viewpoint[2]), float(theta_deg))) > 1e-6:
        return {**base, **_failed_contract("candidate_theta_mismatch")}
    if len(str(candidate.get("candidate_set_hash") or "")) == 0:
        return {**base, **_failed_contract("candidate_set_hash_missing")}

    if goal_world_pose is None:
        goal_cell = Cell(int(viewpoint[0]), int(viewpoint[1]))
        goal_world = _cell_center_world(grid.spec, goal_cell)
    else:
        goal_world = WorldPoint(float(goal_world_pose[0]), float(goal_world_pose[1]))
        goal_cell = grid.spec.world_to_cell(goal_world)
    goal_pose = Pose2D(goal_world.x, goal_world.y, math.radians(float(theta_deg)))
    request = PosePlanRequest(
        start=Pose2D(float(pose[0]), float(pose[1]), float(pose[2])),
        goal=goal_pose,
        theta_bin_count=int(theta_bin_count),
        position_tolerance_m=(
            float(goal_position_tolerance_m)
            if goal_position_tolerance_m is not None and float(goal_position_tolerance_m) > 0.0
            else max(0.5, float(grid.spec.resolution) * 0.5)
        ),
        theta_tolerance_rad=math.radians(float(goal_theta_tolerance_deg)),
        max_iterations=int(max_iterations),
        primitive_duration_s=float(primitive_duration_s),
        integration_dt_s=float(integration_dt_s),
        max_speed_mps=float(max_speed_mps),
        max_angular_speed_radps=math.radians(float(max_angular_speed_degps)),
        footprint_length_m=float(footprint_length_m),
        footprint_width_m=float(footprint_width_m),
        footprint_safety_margin_m=float(footprint_safety_margin_m),
        rotation_cost_weight=float(rotation_cost_weight),
        reverse_penalty_weight=float(reverse_penalty_weight),
        turn_penalty_weight=float(turn_penalty_weight),
        closed_key_xy_resolution_m=(
            float(closed_key_xy_resolution_m)
            if closed_key_xy_resolution_m is not None and float(closed_key_xy_resolution_m) > 0.0
            else None
        ),
    )
    hybrid = HybridAStarPlanner().plan(grid, request)
    start_cell = grid.spec.world_to_cell(WorldPoint(float(pose[0]), float(pose[1])))
    legacy = AStarPlanner().plan(grid, PlanRequest(start=start_cell, goal=goal_cell))
    breakdown = hybrid.cost_breakdown.to_dict()
    hybrid_cost = float(hybrid.total_cost) if hybrid.success else None
    legacy_cost = float(legacy.total_cost) if legacy.success else None
    return {
        **base,
        "candidate_pose_contract_valid": True,
        "hybrid_astar_reachable": bool(hybrid.success),
        "hybrid_astar_trajectory_kind": hybrid.to_route_dict(grid.spec)["trajectory_kind"],
        "hybrid_astar_path_cost": hybrid_cost,
        "hybrid_astar_translation_cost": breakdown["translation_cost"],
        "hybrid_astar_rotation_cost": breakdown["rotation_cost"],
        "hybrid_astar_reverse_penalty": breakdown["reverse_penalty"],
        "hybrid_astar_turn_penalty": breakdown["turn_penalty"],
        "hybrid_astar_slope_cost": breakdown["slope_cost"],
        "hybrid_astar_clearance_cost": breakdown["clearance_cost"],
        "hybrid_astar_pose_path_hash": _pose_path_hash(hybrid.pose_path),
        "hybrid_astar_failure_reason": hybrid.failure_reason.value if hybrid.failure_reason else None,
        "hybrid_astar_expanded_pose_count": int(hybrid.expanded_count),
        "hybrid_astar_pose_path_count": len(hybrid.pose_path),
        "hybrid_astar_control_count": len(hybrid.control_sequence),
        "hybrid_astar_ackermann_feasible_claimed": bool(hybrid.diagnostics.ackermann_feasible_claimed),
        "hybrid_astar_dominance_key_policy": hybrid.diagnostics.to_dict()["dominance_key_policy"],
        "hybrid_astar_planning_grid_source": grid.metadata.get("planning_grid_source"),
        "planner_grid_resolution_m": grid.metadata.get("planner_grid_resolution_m"),
        "source_grid_resolution_m": grid.metadata.get("source_grid_resolution_m"),
        "planning_proxy_hash": grid.metadata.get("planning_proxy_hash"),
        "closed_key_xy_resolution_m": request.closed_key_xy_resolution_m,
        "candidate_goal_world_pose": list(goal_world_pose) if goal_world_pose is not None else None,
        "legacy_grid_astar_reachable": bool(legacy.success),
        "legacy_grid_astar_path_cost": legacy_cost,
        "legacy_grid_astar_failure_reason": legacy.failure_reason.value if legacy.failure_reason else None,
        "legacy_grid_astar_trajectory_kind": legacy.to_route_dict(grid.spec)["trajectory_kind"],
        "hybrid_vs_grid_path_cost_delta": None if hybrid_cost is None or legacy_cost is None else hybrid_cost - legacy_cost,
        "path_cost_source_recommendation": PATH_COST_SOURCE,
        "default_astar_replaced": False,
    }


def build_cost_grid_from_config(grid_config: dict[str, Any]) -> CostGrid:
    width = int(grid_config["width"])
    height = int(grid_config["height"])
    resolution = float(grid_config.get("resolution_m", grid_config.get("resolution", 1.0)))
    cost = [[1.0 for _ in range(width)] for _ in range(height)]
    for weighted in grid_config.get("weighted_cells", []) or []:
        if isinstance(weighted, (list, tuple)) and len(weighted) >= 3:
            x, y, value = int(weighted[0]), int(weighted[1]), float(weighted[2])
            if 0 <= x < width and 0 <= y < height:
                cost[y][x] = value
    passable = [[True for _ in range(width)] for _ in range(height)]
    for x, y in _declared_obstacles(grid_config):
        if 0 <= x < width and 0 <= y < height:
            passable[y][x] = False
    import numpy as np

    return CostGrid(
        GridSpec(width=width, height=height, resolution=resolution),
        np.asarray(cost, dtype=float),
        np.asarray(passable, dtype=bool),
        metadata={"planning_grid_source": "config_grid/v1", "planner_grid_resolution_m": resolution},
    )


def build_cost_grid_from_sidecar(sidecar: dict[str, Any]) -> CostGrid:
    cost = sidecar.get("cost")
    if not isinstance(cost, list) or not cost or not isinstance(cost[0], list):
        raise ValueError("sidecar cost must be a non-empty 2D list")
    height = len(cost)
    width = len(cost[0])
    if any(not isinstance(row, list) or len(row) != width for row in cost):
        raise ValueError("sidecar cost rows must have equal width")
    metadata = sidecar.get("metadata") if isinstance(sidecar.get("metadata"), dict) else {}
    map_source = metadata.get("map_source") if isinstance(metadata.get("map_source"), dict) else {}
    resolution = float(
        sidecar.get("resolution_m")
        or sidecar.get("resolution")
        or metadata.get("resolution_m")
        or metadata.get("resolution")
        or map_source.get("resolution_m")
        or 1.0
    )
    passable_mask = sidecar.get("passable_mask")
    if isinstance(passable_mask, list) and len(passable_mask) == height:
        passable = [[bool(value) for value in row] for row in passable_mask]
        if any(len(row) != width for row in passable):
            raise ValueError("sidecar passable_mask rows must match cost width")
    else:
        passable = [[True for _ in range(width)] for _ in range(height)]
    for x, y in _declared_obstacles(sidecar):
        if 0 <= x < width and 0 <= y < height:
            passable[y][x] = False
    import numpy as np

    return CostGrid(
        GridSpec(width=width, height=height, resolution=resolution),
        np.asarray(cost, dtype=float),
        np.asarray(passable, dtype=bool),
        metadata={"planning_grid_source": "sidecar_cost_grid/v1", "planner_grid_resolution_m": resolution},
    )


def build_derived_high_res_planning_proxy_grid(source_grid: CostGrid, target_resolution_m: float) -> CostGrid:
    target_resolution = float(target_resolution_m)
    if target_resolution <= 0.0 or not math.isfinite(target_resolution):
        raise ValueError("target_resolution_m must be a positive finite float")
    source_resolution = float(source_grid.spec.resolution)
    if target_resolution >= source_resolution:
        raise ValueError("target_resolution_m must be smaller than source grid resolution")

    world_width_m = float(source_grid.spec.width) * source_resolution
    world_height_m = float(source_grid.spec.height) * source_resolution
    width = int(math.ceil(world_width_m / target_resolution))
    height = int(math.ceil(world_height_m / target_resolution))
    import numpy as np

    cost = np.zeros((height, width), dtype=float)
    passable = np.zeros((height, width), dtype=bool)
    for y in range(height):
        for x in range(width):
            world = WorldPoint(
                source_grid.spec.origin[0] + (float(x) + 0.5) * target_resolution,
                source_grid.spec.origin[1] + (float(y) + 0.5) * target_resolution,
            )
            source_cell = source_grid.spec.world_to_cell(world)
            if source_grid.spec.in_bounds(source_cell):
                cost[y, x] = source_grid.cost_at(source_cell)
                passable[y, x] = source_grid.is_passable(source_cell)
            else:
                cost[y, x] = 1.0
                passable[y, x] = False
    spec = GridSpec(
        width=width,
        height=height,
        resolution=target_resolution,
        origin=source_grid.spec.origin,
        frame_id=source_grid.spec.frame_id,
    )
    source_grid_hash = _cost_grid_content_hash(source_grid)
    metadata = {
        "planning_grid_source": "derived_high_res_planning_proxy/v1",
        "source_grid_resolution_m": source_resolution,
        "planner_grid_resolution_m": target_resolution,
        "source_grid_width": source_grid.spec.width,
        "source_grid_height": source_grid.spec.height,
        "proxy_grid_width": width,
        "proxy_grid_height": height,
        "source_grid_hash": source_grid_hash,
        "planning_proxy_hash": _planning_proxy_hash(spec, cost, passable, source_grid_hash=source_grid_hash),
        "planning_proxy_semantics": "conservative_constant_inheritance/v1",
    }
    return CostGrid(spec, cost, passable, metadata=metadata)


def declared_obstacle_cells(grid_config: dict[str, Any]) -> set[tuple[int, int]]:
    return _declared_obstacles(grid_config)


def _base_row(candidate: dict[str, Any], platform_contract_hash: str, max_traversable_slope_deg: float) -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage24-1-candidate-hybrid-path-cost-audit-row/v1",
        "scenario_id": candidate.get("scenario_id"),
        "step_index": candidate.get("step_index"),
        "candidate_index": candidate.get("candidate_index"),
        "candidate_set_hash": candidate.get("candidate_set_hash"),
        "candidate_viewpoint": candidate.get("candidate_viewpoint"),
        "candidate_theta_deg": candidate.get("candidate_theta_deg"),
        "legacy_candidate_path_cost": candidate.get("path_cost"),
        "platform_contract_hash": platform_contract_hash,
        "max_traversable_slope_deg": float(max_traversable_slope_deg),
    }


def _failed_contract(reason: str) -> dict[str, Any]:
    return {
        "candidate_pose_contract_valid": False,
        "hybrid_astar_reachable": False,
        "hybrid_astar_trajectory_kind": "hybrid_astar_pose_path",
        "hybrid_astar_path_cost": None,
        "hybrid_astar_translation_cost": None,
        "hybrid_astar_rotation_cost": None,
        "hybrid_astar_reverse_penalty": None,
        "hybrid_astar_turn_penalty": None,
        "hybrid_astar_slope_cost": None,
        "hybrid_astar_clearance_cost": None,
        "hybrid_astar_pose_path_hash": None,
        "hybrid_astar_failure_reason": reason,
        "hybrid_astar_expanded_pose_count": 0,
        "hybrid_astar_pose_path_count": 0,
        "hybrid_astar_control_count": 0,
        "hybrid_astar_ackermann_feasible_claimed": False,
        "hybrid_astar_dominance_key_policy": None,
        "legacy_grid_astar_reachable": None,
        "legacy_grid_astar_path_cost": None,
        "legacy_grid_astar_failure_reason": None,
        "legacy_grid_astar_trajectory_kind": "geometric_path",
        "hybrid_vs_grid_path_cost_delta": None,
        "path_cost_source_recommendation": PATH_COST_SOURCE,
        "default_astar_replaced": False,
    }


def _parse_current_pose(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    parsed = tuple(_finite_float(item) for item in value[:3])
    if any(item is None for item in parsed):
        return None
    return (float(parsed[0]), float(parsed[1]), float(parsed[2]))


def _parse_viewpoint(value: Any) -> tuple[int, int, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    x = _finite_float(value[0])
    y = _finite_float(value[1])
    theta = _finite_float(value[2])
    if x is None or y is None or theta is None:
        return None
    return (int(x), int(y), float(theta))


def _parse_goal_world_pose(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = _finite_float(value[0])
    y = _finite_float(value[1])
    if x is None or y is None:
        return None
    return (float(x), float(y))


def _cell_center_world(spec: GridSpec, cell: Cell) -> WorldPoint:
    return WorldPoint(
        spec.origin[0] + (float(cell.x) + 0.5) * spec.resolution,
        spec.origin[1] + (float(cell.y) + 0.5) * spec.resolution,
    )


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _angle_delta_deg(left: float, right: float) -> float:
    return ((left - right + 180.0) % 360.0) - 180.0


def _pose_path_hash(poses: tuple[Pose2D, ...]) -> str | None:
    if not poses:
        return None
    payload = [
        [round(float(pose.x_m), 6), round(float(pose.y_m), 6), round(float(pose.theta_rad), 6)]
        for pose in poses
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cost_grid_content_hash(grid: CostGrid) -> str:
    import numpy as np

    cost_array = np.asarray(grid.cost, dtype=np.float64)
    passable_array = np.asarray(grid.passable_mask, dtype=bool)
    payload = {
        "spec": grid.spec.to_dict(),
        "cost_shape": list(cost_array.shape),
        "cost_hash": hashlib.sha256(cost_array.tobytes()).hexdigest(),
        "passable_shape": list(passable_array.shape),
        "passable_hash": hashlib.sha256(passable_array.astype("uint8").tobytes()).hexdigest(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _planning_proxy_hash(spec: GridSpec, cost: Any, passable: Any, *, source_grid_hash: str | None = None) -> str:
    import numpy as np

    cost_array = np.asarray(cost, dtype=np.float64)
    passable_array = np.asarray(passable, dtype=bool)
    payload = {
        "spec": spec.to_dict(),
        "cost_shape": list(cost_array.shape),
        "cost_hash": hashlib.sha256(cost_array.tobytes()).hexdigest(),
        "passable_shape": list(passable_array.shape),
        "passable_hash": hashlib.sha256(passable_array.astype("uint8").tobytes()).hexdigest(),
        "source_grid_hash": source_grid_hash,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _declared_obstacles(grid_config: dict[str, Any]) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for field in (
        "physical_obstacle_cells",
        "obstacle_cells",
        "slope_blocked_cells",
        "blocked_cells",
        "synthetic_hard_obstacle_cells",
    ):
        for cell in grid_config.get(field, []) or []:
            if isinstance(cell, (list, tuple)) and len(cell) >= 2:
                result.add((int(cell[0]), int(cell[1])))
    return result
