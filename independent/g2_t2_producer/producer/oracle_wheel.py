from __future__ import annotations

import math
from typing import Any

from .geometry import (
    map_clearance_mm,
    minimum_clearance_mm,
    plane_slope_cdeg,
    points_touch_polygons,
)


def wheel_endpoint_mm(case: dict[str, Any]) -> tuple[int, int, int]:
    start_x, start_y, start_theta_urad = case["start_pose_mm_urad"]
    primitive = case["primitive"]
    speed_mm_s = int(primitive["speed_um_s"]) / 1000.0
    omega_rad_s = int(primitive["angular_urad_s"]) / 1_000_000.0
    duration_s = int(primitive["duration_us"]) / 1_000_000.0
    theta0 = int(start_theta_urad) / 1_000_000.0
    theta1 = theta0 + omega_rad_s * duration_s
    if abs(omega_rad_s) < 1e-15:
        x = int(start_x) + speed_mm_s * duration_s * math.cos(theta0)
        y = int(start_y) + speed_mm_s * duration_s * math.sin(theta0)
    else:
        radius_mm = speed_mm_s / omega_rad_s
        x = int(start_x) + radius_mm * (math.sin(theta1) - math.sin(theta0))
        y = int(start_y) - radius_mm * (math.cos(theta1) - math.cos(theta0))
    return round(x), round(y), round(theta1 * 1_000_000.0)


def wheel_sweep_points(
    case: dict[str, Any], *, sample_count: int = 65
) -> list[tuple[int, int]]:
    if sample_count < 2:
        raise ValueError("sample_count must be at least two")
    start_x, start_y, start_theta_urad = [
        int(value) for value in case["start_pose_mm_urad"]
    ]
    primitive = case["primitive"]
    speed_mm_s = int(primitive["speed_um_s"]) / 1000.0
    omega = int(primitive["angular_urad_s"]) / 1_000_000.0
    duration = int(primitive["duration_us"]) / 1_000_000.0
    theta0 = start_theta_urad / 1_000_000.0
    points: list[tuple[int, int]] = []
    for index in range(sample_count):
        time_s = duration * index / (sample_count - 1)
        theta = theta0 + omega * time_s
        if abs(omega) < 1e-15:
            x = start_x + speed_mm_s * time_s * math.cos(theta0)
            y = start_y + speed_mm_s * time_s * math.sin(theta0)
        else:
            radius = speed_mm_s / omega if speed_mm_s else 0.0
            x = start_x + radius * (math.sin(theta) - math.sin(theta0))
            y = start_y - radius * (math.cos(theta) - math.cos(theta0))
        points.append((round(x), round(y)))
    return points


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


def _legacy_geometry(terrain: dict[str, Any]) -> dict[str, Any]:
    slope = int(terrain.get("slope_cdeg", 0))
    obstacle_slack = int(terrain.get("sweep_clearance_mm", 20))
    obstacle: list[list[list[int]]] = []
    if obstacle_slack <= 0:
        obstacle = [[[499, -1], [501, -1], [501, 1], [499, 1]]]
    return {
        "map_bounds_mm": (
            [-10000, -10000, 10000, 10000]
            if bool(terrain.get("inside_map", True))
            else [-10000, -10000, 100, 100]
        ),
        "height_plane": {
            "origin_x_mm": 0,
            "origin_y_mm": 0,
            "origin_z_mm": 0,
            "gradient_x_ppm": round(
                math.tan(math.radians(slope / 100.0)) * 1_000_000
            ),
            "gradient_y_ppm": 0,
        },
        "obstacle_polygons_mm": obstacle,
        "unknown_polygons_mm": (
            [[[-10000, -10000], [10000, -10000], [10000, 10000], [-10000, 10000]]]
            if not bool(terrain.get("known_sweep", True))
            else []
        ),
        "untraversable_polygons_mm": (
            [[[-10000, -10000], [10000, -10000], [10000, 10000], [-10000, 10000]]]
            if not bool(terrain.get("traversable", True))
            else []
        ),
        "wheel_footprint_radius_mm": max(0, 20 - obstacle_slack),
    }


def evaluate_wheel(case: dict[str, Any]) -> dict[str, Any]:
    if case.get("numeric_state") != "decided":
        return _decision(
            False,
            "G2I_NUMERIC_UNDECIDED",
            witness={"numeric_state": str(case.get("numeric_state"))},
            slacks={},
        )
    primitive = case["primitive"]
    limits = case["limits"]
    terrain = case["terrain"]
    if "inside_map" in terrain:
        terrain = _legacy_geometry(terrain)
    speed = int(primitive["speed_um_s"])
    angular = int(primitive["angular_urad_s"])
    duration = int(primitive["duration_us"])
    computed_endpoint = wheel_endpoint_mm(case)
    declared_endpoint = tuple(int(value) for value in primitive["endpoint_mm_urad"])
    endpoint_xy_error = round(
        math.hypot(
            computed_endpoint[0] - declared_endpoint[0],
            computed_endpoint[1] - declared_endpoint[1],
        )
    )
    endpoint_theta_error = abs(computed_endpoint[2] - declared_endpoint[2])
    speed_slack = int(limits["max_abs_speed_um_s"]) - abs(speed)
    angular_slack = int(limits["max_abs_angular_urad_s"]) - abs(angular)
    duration_slack = int(limits["max_duration_us"]) - duration
    turn_radius_mm = (
        round(abs((speed / 1000.0) / (angular / 1_000_000.0)))
        if speed and angular
        else 2**31 - 1
    )
    turn_radius_slack = turn_radius_mm - int(limits["min_turn_radius_mm"])
    samples = wheel_sweep_points(case)
    radius = int(terrain.get("wheel_footprint_radius_mm", 250))
    map_slack = map_clearance_mm(samples, terrain["map_bounds_mm"], radius)
    obstacle_slack = minimum_clearance_mm(
        samples, terrain.get("obstacle_polygons_mm", []), radius
    )
    unknown = points_touch_polygons(
        samples, terrain.get("unknown_polygons_mm", [])
    )
    untraversable = points_touch_polygons(
        samples, terrain.get("untraversable_polygons_mm", [])
    )
    slope = plane_slope_cdeg(terrain["height_plane"])
    slope_slack = 3000 - slope
    witness = {
        "computed_endpoint_mm_urad": list(computed_endpoint),
        "declared_endpoint_mm_urad": list(declared_endpoint),
        "duration_us": duration,
        "endpoint_theta_error_urad": endpoint_theta_error,
        "endpoint_xy_error_mm": endpoint_xy_error,
        "map_clearance_mm": map_slack,
        "minimum_obstacle_clearance_mm": obstacle_slack,
        "plane_slope_cdeg": slope,
        "sweep_sample_count": len(samples),
        "turn_radius_mm": turn_radius_mm,
    }
    slacks = {
        "angular_urad_s": angular_slack,
        "duration_us": duration_slack,
        "map_mm": map_slack,
        "obstacle_mm": obstacle_slack,
        "slope_cdeg": slope_slack,
        "speed_um_s": speed_slack,
        "turn_radius_mm": turn_radius_slack,
    }
    if duration <= 0 or duration_slack < 0:
        return _decision(False, "G2I_W_DURATION_DOMAIN", witness=witness, slacks=slacks)
    if speed_slack < 0 or (speed < 0 and not bool(limits["allow_reverse"])):
        return _decision(False, "G2I_W_SPEED_DOMAIN", witness=witness, slacks=slacks)
    if angular_slack < 0 or (angular != 0 and not bool(limits["allow_turn"])):
        return _decision(False, "G2I_W_ANGULAR_DOMAIN", witness=witness, slacks=slacks)
    if speed and angular and turn_radius_slack < 0:
        return _decision(
            False, "G2I_W_TURN_RADIUS_DOMAIN", witness=witness, slacks=slacks
        )
    if (
        endpoint_xy_error > int(limits["endpoint_tolerance_mm"])
        or endpoint_theta_error > int(limits.get("endpoint_tolerance_urad", 1000))
    ):
        return _decision(
            False, "G2I_W_ENDPOINT_INCONSISTENT", witness=witness, slacks=slacks
        )
    if unknown:
        return _decision(False, "G2I_W_UNKNOWN_SWEEP", witness=witness, slacks=slacks)
    if map_slack < 0:
        return _decision(False, "G2I_W_OUTSIDE_MAP", witness=witness, slacks=slacks)
    if obstacle_slack <= 0:
        return _decision(
            False, "G2I_W_CLOSED_OBSTACLE_CONTACT", witness=witness, slacks=slacks
        )
    if untraversable:
        return _decision(
            False, "G2I_W_NOT_TRAVERSABLE", witness=witness, slacks=slacks
        )
    if slope_slack < 0:
        return _decision(False, "G2I_W_SLOPE_LIMIT", witness=witness, slacks=slacks)
    return _decision(True, "G2I_W_SAFE", witness=witness, slacks=slacks)
