from __future__ import annotations

import math
from typing import Any

from .geometry import (
    map_clearance_mm,
    minimum_clearance_mm,
    plane_height_mm,
    plane_slope_cdeg,
    point_in_polygon_closed,
    sample_segment,
    signed_polygon_margin_mm,
)


def _decision(
    safe: bool,
    reason: str,
    *,
    witness: dict[str, Any],
    slacks: dict[str, int],
) -> dict[str, Any]:
    return {
        "numeric_witness": witness,
        "oracle_reason_code": reason,
        "oracle_safe": safe,
        "safety_slacks": slacks,
    }


def _legacy_geometry(case: dict[str, Any]) -> dict[str, Any]:
    target = case["target"]
    support = int(target.get("support_margin_mm", 75))
    body_clearance = int(target.get("body_sweep_clearance_mm", 20))
    foothold_slope = int(target.get("foothold_slope_cdeg", 0))
    body_slope = int(target.get("body_slope_cdeg", 0))
    target_xy = [int(value) for value in case["target_foot_mm"][:2]]
    obstacle = []
    if bool(target.get("foothold_obstacle", False)):
        obstacle = [
            [
                [target_xy[0] - 1, target_xy[1] - 1],
                [target_xy[0] + 1, target_xy[1] - 1],
                [target_xy[0] + 1, target_xy[1] + 1],
                [target_xy[0] - 1, target_xy[1] + 1],
            ]
        ]
    body_obstacle = []
    if body_clearance <= 0:
        body_obstacle = [[[-1, -1], [1, -1], [1, 1], [-1, 1]]]
    return {
        "map_bounds_mm": (
            [-10000, -10000, 10000, 10000]
            if bool(target.get("grid_valid", True))
            else [0, 0, 1, 1]
        ),
        "foothold_plane": {
            "origin_x_mm": 0,
            "origin_y_mm": 0,
            "origin_z_mm": int(case["target_foot_mm"][2]),
            "gradient_x_ppm": round(
                math.tan(math.radians(foothold_slope / 100.0)) * 1_000_000
            ),
            "gradient_y_ppm": 0,
        },
        "body_plane": {
            "origin_x_mm": 0,
            "origin_y_mm": 0,
            "origin_z_mm": 0,
            "gradient_x_ppm": round(
                math.tan(math.radians(body_slope / 100.0)) * 1_000_000
            ),
            "gradient_y_ppm": 0,
        },
        "unknown_polygons_mm": (
            [
                [
                    [target_xy[0] - 1, target_xy[1] - 1],
                    [target_xy[0] + 1, target_xy[1] - 1],
                    [target_xy[0] + 1, target_xy[1] + 1],
                    [target_xy[0] - 1, target_xy[1] + 1],
                ]
            ]
            if bool(target.get("foothold_unknown", False))
            else []
        ),
        "obstacle_polygons_mm": obstacle,
        "body_obstacle_polygons_mm": body_obstacle,
        "body_start_mm": [0, 0],
        "body_end_mm": [100, 0],
        "body_radius_mm": max(0, 20 - body_clearance),
        "support_polygon_mm": [
            [-support, -support],
            [support, -support],
            [support, support],
            [-support, support],
        ],
        "target_com_mm": [0, 0],
        "untraversable_polygons_mm": (
            [
                [
                    [target_xy[0] - 1, target_xy[1] - 1],
                    [target_xy[0] + 1, target_xy[1] - 1],
                    [target_xy[0] + 1, target_xy[1] + 1],
                    [target_xy[0] - 1, target_xy[1] + 1],
                ]
            ]
            if not bool(target.get("foothold_traversable", True))
            else []
        ),
    }


def evaluate_legged(case: dict[str, Any]) -> dict[str, Any]:
    if case.get("numeric_state") != "decided":
        return _decision(
            False,
            "G2I_NUMERIC_UNDECIDED",
            witness={"numeric_state": str(case.get("numeric_state"))},
            slacks={},
        )
    terrain = case.get("terrain")
    if terrain is None:
        terrain = _legacy_geometry(case)
    source_xyz = tuple(int(value) for value in case["source_foot_mm"])
    target_xyz = tuple(int(value) for value in case["target_foot_mm"])
    dx = target_xyz[0] - source_xyz[0]
    dy = target_xyz[1] - source_xyz[1]
    dz = target_xyz[2] - source_xyz[2]
    step_length_mm = round(math.sqrt(dx * dx + dy * dy + dz * dz))
    step_height_mm = abs(dz)
    target_xy = (target_xyz[0], target_xyz[1])
    map_slack = map_clearance_mm([target_xy], terrain["map_bounds_mm"], 0)
    unknown = any(
        point_in_polygon_closed(target_xy, polygon)
        for polygon in terrain.get("unknown_polygons_mm", [])
    )
    obstacle = any(
        point_in_polygon_closed(target_xy, polygon)
        for polygon in terrain.get("obstacle_polygons_mm", [])
    )
    untraversable = any(
        point_in_polygon_closed(target_xy, polygon)
        for polygon in terrain.get("untraversable_polygons_mm", [])
    )
    foothold_slope = plane_slope_cdeg(terrain["foothold_plane"])
    body_slope = plane_slope_cdeg(terrain["body_plane"])
    expected_target_z = plane_height_mm(
        terrain["foothold_plane"], target_xy[0], target_xy[1]
    )
    foothold_height_error = abs(target_xyz[2] - expected_target_z)
    support_margin = signed_polygon_margin_mm(
        tuple(int(value) for value in terrain["target_com_mm"]),
        terrain["support_polygon_mm"],
    )
    body_points = sample_segment(
        tuple(int(value) for value in terrain["body_start_mm"]),
        tuple(int(value) for value in terrain["body_end_mm"]),
    )
    body_clearance = minimum_clearance_mm(
        body_points,
        terrain.get("body_obstacle_polygons_mm", []),
        int(terrain["body_radius_mm"]),
    )
    slacks = {
        "body_clearance_mm": body_clearance,
        "body_slope_cdeg": 3000 - body_slope,
        "foothold_height_mm": 1 - foothold_height_error,
        "foothold_slope_cdeg": 2500 - foothold_slope,
        "map_mm": map_slack,
        "step_height_mm": 250 - step_height_mm,
        "step_length_mm": 500 - step_length_mm,
        "support_margin_mm": support_margin - 50,
    }
    witness = {
        "body_slope_cdeg": body_slope,
        "foothold_height_error_mm": foothold_height_error,
        "foothold_slope_cdeg": foothold_slope,
        "map_clearance_mm": map_slack,
        "minimum_body_clearance_mm": body_clearance,
        "moving_leg_id": int(case["moving_leg_id"]),
        "step_height_mm": step_height_mm,
        "step_length_mm": step_length_mm,
        "support_margin_mm": support_margin,
    }
    if int(case["phase"]) != int(case["required_phase"]):
        return _decision(False, "G2I_L_SEQUENCE", witness=witness, slacks=slacks)
    if map_slack < 0:
        return _decision(False, "G2I_L_GRID_DOMAIN", witness=witness, slacks=slacks)
    if unknown:
        return _decision(
            False, "G2I_L_FOOTHOLD_UNKNOWN", witness=witness, slacks=slacks
        )
    if obstacle:
        return _decision(
            False, "G2I_L_FOOTHOLD_OBSTACLE", witness=witness, slacks=slacks
        )
    if untraversable or foothold_slope > 2500 or foothold_height_error > 1:
        return _decision(
            False, "G2I_L_FOOTHOLD_SLOPE", witness=witness, slacks=slacks
        )
    if step_length_mm > 500:
        return _decision(False, "G2I_L_STEP_LENGTH", witness=witness, slacks=slacks)
    if step_height_mm > 250:
        return _decision(False, "G2I_L_STEP_HEIGHT", witness=witness, slacks=slacks)
    if support_margin < 50:
        return _decision(
            False, "G2I_L_SUPPORT_MARGIN", witness=witness, slacks=slacks
        )
    if body_clearance <= 0 or body_slope > 3000:
        return _decision(False, "G2I_L_BODY_SWEEP", witness=witness, slacks=slacks)
    return _decision(True, "G2I_L_SAFE", witness=witness, slacks=slacks)
