from __future__ import annotations

import math
from typing import Any

from .canonical import canonical_json_bytes, derive_seed, domain_hash
from .geometry import gradient_ppm_for_slope_cdeg, square_polygon
from .models import validate_truth_blind_case
from .oracle_hopper import (
    ballistic_witness,
    validate_hopper_parameter_record,
)
from .oracle_wheel import wheel_sweep_points


_CONTROL_FAMILIES = (
    "straight_forward",
    "straight_reverse",
    "rotate_left",
    "rotate_right",
    "arc_left",
    "arc_right",
    "hold",
)
_HOPPER_SPEEDS_MM_S = (1500, 2000, 2500, 3000)
_HOPPER_ELEVATIONS_MDEG = (30000, 45000, 60000)


def _terrain_sha(platform: str, terrain: dict[str, Any], case_index: int) -> str:
    return domain_hash(
        "g2-neutral-terrain-source/v2",
        platform.encode("ascii"),
        str(case_index).encode("ascii"),
        canonical_json_bytes(terrain),
    )


def _plane(slope_cdeg: int = 0, *, origin_z_mm: int = 0) -> dict[str, int]:
    return {
        "gradient_x_ppm": gradient_ppm_for_slope_cdeg(slope_cdeg),
        "gradient_y_ppm": 0,
        "origin_x_mm": 0,
        "origin_y_mm": 0,
        "origin_z_mm": int(origin_z_mm),
    }


def _wheel_proposal_endpoint(
    start_pose: list[int],
    *,
    speed_um_s: int,
    angular_urad_s: int,
    duration_us: int,
) -> list[int]:
    x0, y0, theta0_urad = start_pose
    speed_mm_s = speed_um_s / 1000.0
    omega = angular_urad_s / 1_000_000.0
    duration = duration_us / 1_000_000.0
    theta0 = theta0_urad / 1_000_000.0
    theta1 = theta0 + omega * duration
    if angular_urad_s == 0:
        x = x0 + speed_mm_s * duration * math.cos(theta0)
        y = y0 + speed_mm_s * duration * math.sin(theta0)
    else:
        radius = speed_mm_s / omega if speed_um_s else 0.0
        x = x0 + radius * (math.sin(theta1) - math.sin(theta0))
        y = y0 - radius * (math.cos(theta1) - math.cos(theta0))
    return [round(x), round(y), round(theta1 * 1_000_000)]


def _wheel_base(index: int) -> dict[str, Any]:
    heading_bin = index % 72
    theta_urad = round(heading_bin * 2.0 * math.pi / 72.0 * 1_000_000)
    family = _CONTROL_FAMILIES[index % len(_CONTROL_FAMILIES)]
    speed, angular = {
        "straight_forward": (400_000, 0),
        "straight_reverse": (-400_000, 0),
        "rotate_left": (0, 500_000),
        "rotate_right": (0, -500_000),
        "arc_left": (400_000, 500_000),
        "arc_right": (400_000, -500_000),
        "hold": (0, 0),
    }[family]
    start = [0, 0, theta_urad]
    duration = 1_000_000
    terrain = {
        "geometry_schema_version": "g2-wheel-neutral-geometry/v2",
        "height_plane": _plane((index % 21) * 10),
        "map_bounds_mm": [-5000, -5000, 5000, 5000],
        "obstacle_polygons_mm": [],
        "unknown_polygons_mm": [],
        "untraversable_polygons_mm": [],
        "wheel_footprint_radius_mm": 100,
    }
    case = {
        "case_index": index,
        "determinism_seed": derive_seed(20260727, "wheel", "labels", "case", index),
        "limits": {
            "allow_reverse": True,
            "allow_turn": True,
            "endpoint_tolerance_mm": 1,
            "endpoint_tolerance_urad": 1000,
            "max_abs_angular_urad_s": 2_000_000,
            "max_abs_speed_um_s": 1_000_000,
            "max_duration_us": 5_000_000,
            "min_turn_radius_mm": 250,
        },
        "numeric_state": "decided",
        "primitive": {
            "angular_urad_s": angular,
            "control_family": family,
            "duration_us": duration,
            "endpoint_mm_urad": _wheel_proposal_endpoint(
                start,
                speed_um_s=speed,
                angular_urad_s=angular,
                duration_us=duration,
            ),
            "heading_bin": heading_bin,
            "speed_um_s": speed,
        },
        "start_pose_mm_urad": start,
        "terrain": terrain,
    }
    case["terrain_sha256"] = _terrain_sha("wheel", terrain, index)
    return case


def _regenerate_wheel_endpoint(case: dict[str, Any]) -> None:
    primitive = case["primitive"]
    primitive["endpoint_mm_urad"] = _wheel_proposal_endpoint(
        case["start_pose_mm_urad"],
        speed_um_s=int(primitive["speed_um_s"]),
        angular_urad_s=int(primitive["angular_urad_s"]),
        duration_us=int(primitive["duration_us"]),
    )


def _force_wheel_motion(
    case: dict[str, Any], *, angular_urad_s: int = 0
) -> None:
    case["start_pose_mm_urad"] = [0, 0, 0]
    case["primitive"].update(
        {
            "angular_urad_s": angular_urad_s,
            "control_family": "arc_left" if angular_urad_s else "straight_forward",
            "duration_us": 1_000_000,
            "speed_um_s": 400_000,
        }
    )
    _regenerate_wheel_endpoint(case)


def _place_wheel_obstacle(case: dict[str, Any], desired_slack_mm: int) -> None:
    points = wheel_sweep_points(case)
    middle = points[len(points) // 2]
    radius = int(case["terrain"]["wheel_footprint_radius_mm"])
    center = (middle[0], middle[1] + radius + int(desired_slack_mm) + 1)
    case["terrain"]["obstacle_polygons_mm"] = [
        square_polygon(center[0], center[1], 1)
    ]


def _wheel_case(stratum: str, local: int, index: int) -> dict[str, Any]:
    case = _wheel_base(index)
    if stratum == "slope":
        case["terrain"]["height_plane"] = _plane((2999, 3000, 3001)[local // 150])
    elif stratum == "obstacle":
        _force_wheel_motion(case)
        _place_wheel_obstacle(case, (1, 0, -1)[local // 150])
    elif stratum == "knownness_boundary":
        _force_wheel_motion(case)
        if local < 150:
            case["terrain"]["unknown_polygons_mm"] = [
                square_polygon(200, 0, 300)
            ]
        else:
            case["terrain"]["map_bounds_mm"] = [-150, -150, 150, 150]
    elif stratum == "kinematic":
        group = local // 90
        if group == 0:
            case["primitive"]["duration_us"] = 5_000_001
        elif group == 1:
            case["primitive"]["speed_um_s"] = 1_000_001
        elif group == 2:
            case["primitive"]["angular_urad_s"] = 2_000_001
        elif group == 3:
            case["primitive"]["speed_um_s"] = 100_000
            case["primitive"]["angular_urad_s"] = 1_000_000
        elif group == 4:
            case["primitive"]["speed_um_s"] = -400_000
            case["limits"]["allow_reverse"] = False
        elif group == 5:
            case["primitive"]["angular_urad_s"] = 500_000
            case["limits"]["allow_turn"] = False
        else:
            case["primitive"]["speed_um_s"] = 1_000_000
            case["primitive"]["angular_urad_s"] = 0
        _regenerate_wheel_endpoint(case)
    elif stratum == "anti_alias":
        _force_wheel_motion(case, angular_urad_s=500_000)
        _place_wheel_obstacle(case, 1 if local < 202 else -1)
    elif stratum == "compound":
        _force_wheel_motion(case)
        case["primitive"]["speed_um_s"] = 1_000_001
        _regenerate_wheel_endpoint(case)
        _place_wheel_obstacle(case, -1)
    case["terrain_sha256"] = _terrain_sha("wheel", case["terrain"], index)
    return case


def _legged_base(index: int) -> dict[str, Any]:
    leg_id = index % 4
    terrain = {
        "body_end_mm": [100, 0],
        "body_obstacle_polygons_mm": [],
        "body_plane": _plane((index % 25) * 10),
        "body_radius_mm": 100,
        "body_start_mm": [0, 0],
        "foothold_plane": _plane((index % 20) * 10),
        "geometry_schema_version": "g2-legged-neutral-geometry/v2",
        "map_bounds_mm": [-5000, -5000, 5000, 5000],
        "obstacle_polygons_mm": [],
        "support_polygon_mm": square_polygon(0, 0, 75),
        "target_com_mm": [0, 0],
        "unknown_polygons_mm": [],
        "untraversable_polygons_mm": [],
    }
    case = {
        "case_index": index,
        "determinism_seed": derive_seed(20260727, "legged", "labels", "case", index),
        "moving_leg_id": leg_id,
        "numeric_state": "decided",
        "phase": leg_id,
        "required_phase": leg_id,
        "source_foot_mm": [0, 0, 0],
        "target_foot_mm": [300, 0, 0],
        "terrain": terrain,
    }
    case["target_foot_mm"][2] = round(
        300 * int(terrain["foothold_plane"]["gradient_x_ppm"]) / 1_000_000
    )
    case["terrain_sha256"] = _terrain_sha("legged", terrain, index)
    return case


def _target_square(case: dict[str, Any], halfwidth_mm: int = 2) -> list[list[int]]:
    x, y = [int(value) for value in case["target_foot_mm"][:2]]
    return square_polygon(x, y, halfwidth_mm)


def _place_body_obstacle(case: dict[str, Any], desired_slack_mm: int) -> None:
    start = case["terrain"]["body_start_mm"]
    end = case["terrain"]["body_end_mm"]
    middle_x = round((int(start[0]) + int(end[0])) / 2)
    middle_y = round((int(start[1]) + int(end[1])) / 2)
    radius = int(case["terrain"]["body_radius_mm"])
    center_y = middle_y + radius + int(desired_slack_mm) + 1
    case["terrain"]["body_obstacle_polygons_mm"] = [
        square_polygon(middle_x, center_y, 1)
    ]


def _legged_case(stratum: str, local: int, index: int) -> dict[str, Any]:
    case = _legged_base(index)
    if stratum == "foothold":
        group = local // 80
        if group <= 2:
            slope = (2499, 2500, 2501)[group]
            case["terrain"]["foothold_plane"] = _plane(slope)
            target_x = int(case["target_foot_mm"][0])
            case["target_foot_mm"][2] = round(
                target_x
                * int(case["terrain"]["foothold_plane"]["gradient_x_ppm"])
                / 1_000_000
            )
        elif group == 3:
            case["terrain"]["unknown_polygons_mm"] = [_target_square(case)]
        elif group == 4:
            case["terrain"]["obstacle_polygons_mm"] = [_target_square(case)]
        else:
            case["terrain"]["untraversable_polygons_mm"] = [_target_square(case)]
    elif stratum == "step_length":
        case["terrain"]["foothold_plane"] = _plane(0)
        case["target_foot_mm"] = [(499, 500, 501)[local // 120], 0, 0]
    elif stratum == "step_height":
        case["terrain"]["foothold_plane"] = _plane(0)
        case["target_foot_mm"] = [0, 0, (249, 250, 251)[local // 120]]
    elif stratum == "support":
        margin = (49, 50, 51)[local // 160]
        case["terrain"]["support_polygon_mm"] = square_polygon(0, 0, margin)
    elif stratum == "body_sweep":
        _place_body_obstacle(case, (1, 0, -1)[local // 140])
    elif stratum == "sequence_grid":
        if local < 71:
            case["phase"] = (int(case["required_phase"]) + 1) % 4
        elif local < 142:
            case["terrain"]["map_bounds_mm"] = [-100, -100, 100, 100]
    elif stratum == "compound":
        case["phase"] = (int(case["required_phase"]) + 1) % 4
        case["terrain"]["unknown_polygons_mm"] = [_target_square(case)]
    case["terrain_sha256"] = _terrain_sha("legged", case["terrain"], index)
    return case


def hopper_action_from_index(action_index: int) -> dict[str, int]:
    if not 0 <= action_index < 192:
        raise ValueError("hopper action index outside 192-action lattice")
    speed_index = action_index // (3 * 16)
    remainder = action_index % (3 * 16)
    elevation_index = remainder // 16
    azimuth_index = remainder % 16
    return {
        "azimuth_index": azimuth_index,
        "azimuth_mdeg": azimuth_index * 22500,
        "elevation_index": elevation_index,
        "elevation_mdeg": _HOPPER_ELEVATIONS_MDEG[elevation_index],
        "speed_index": speed_index,
        "speed_mm_s": _HOPPER_SPEEDS_MM_S[speed_index],
    }


def _hopper_base(
    index: int,
    action_index: int,
    parameter_record: dict[str, Any],
) -> dict[str, Any]:
    terrain = {
        "geometry_schema_version": "g2-hopper-neutral-geometry/v2",
        "height_plane": _plane(0),
        "landing_height_tolerance_mm": 50,
        "landing_pad_polygon_mm": square_polygon(0, 0, 2000),
        "landing_sigma_mm": 289,
        "landing_surface_plane": _plane(0),
        "map_bounds_mm": [-20000, -20000, 20000, 20000],
        "obstacle_polygons_mm": [],
        "obstacle_prisms": [],
        "unknown_polygons_mm": [],
    }
    case = {
        "action": hopper_action_from_index(action_index),
        "case_index": index,
        "determinism_seed": derive_seed(20260727, "hopper", "labels", "case", index),
        "launch_pose_mm": [0, 0, 0],
        "numeric_state": "decided",
        "terrain": terrain,
    }
    witness = ballistic_witness(case, parameter_record)
    terrain["landing_pad_polygon_mm"] = square_polygon(
        int(witness["landing_x_mm"]), int(witness["landing_y_mm"]), 2000
    )
    case["terrain_sha256"] = _terrain_sha("hopper", terrain, index)
    return case


def _hopper_trajectory_point(
    case: dict[str, Any],
    parameter_record: dict[str, Any],
    sample_index: int,
) -> tuple[int, int, int]:
    witness = ballistic_witness(case, parameter_record)
    action = case["action"]
    launch = case["launch_pose_mm"]
    time_s = witness["flight_time_us"] / 1_000_000.0 * sample_index / 64.0
    elevation = math.radians(int(action["elevation_mdeg"]) / 1000.0)
    azimuth = math.radians(int(action["azimuth_mdeg"]) / 1000.0)
    speed = int(action["speed_mm_s"]) / 1000.0
    horizontal = speed * math.cos(elevation)
    launch_z = round(
        float(parameter_record["launch_reference_height_m"]) * 1000.0
    )
    return (
        int(launch[0])
        + round(horizontal * math.cos(azimuth) * time_s * 1000.0),
        int(launch[1])
        + round(horizontal * math.sin(azimuth) * time_s * 1000.0),
        launch_z
        + round(
            (speed * math.sin(elevation) * time_s - 0.5 * 1.62 * time_s**2)
            * 1000.0
        ),
    )


def _place_hopper_launch_obstacle(
    case: dict[str, Any], desired_slack_mm: int
) -> None:
    body_radius = 375
    launch_x, launch_y = [int(value) for value in case["launch_pose_mm"][:2]]
    center_y = launch_y + body_radius + int(desired_slack_mm) + 1
    case["terrain"]["obstacle_polygons_mm"] = [
        square_polygon(launch_x, center_y, 1)
    ]


def _place_hopper_arc_prism(
    case: dict[str, Any],
    parameter_record: dict[str, Any],
    desired_slack_mm: int,
) -> None:
    x, y, z = _hopper_trajectory_point(case, parameter_record, 1)
    combined_radius = 375 + 125
    case["terrain"]["obstacle_prisms"] = [
        {
            "polygon_mm": square_polygon(x, y, 1),
            "top_z_mm": z - combined_radius - int(desired_slack_mm),
        }
    ]


def _place_hopper_landing_obstacle(
    case: dict[str, Any],
    parameter_record: dict[str, Any],
    desired_slack_mm: int,
) -> None:
    witness = ballistic_witness(case, parameter_record)
    radius = 625
    x = int(witness["landing_x_mm"])
    y = int(witness["landing_y_mm"]) + radius + int(desired_slack_mm) + 1
    case["terrain"]["obstacle_polygons_mm"] = [square_polygon(x, y, 1)]


def _set_landing_surface_slope(
    case: dict[str, Any],
    parameter_record: dict[str, Any],
    slope_cdeg: int,
) -> None:
    witness = ballistic_witness(case, parameter_record)
    gradient = gradient_ppm_for_slope_cdeg(slope_cdeg)
    x = int(witness["landing_x_mm"])
    case["terrain"]["landing_surface_plane"] = {
        "gradient_x_ppm": gradient,
        "gradient_y_ppm": 0,
        "origin_x_mm": 0,
        "origin_y_mm": 0,
        "origin_z_mm": -round(x * gradient / 1_000_000),
    }


def _hopper_case(
    stratum: str,
    local: int,
    index: int,
    parameter_record: dict[str, Any],
) -> dict[str, Any]:
    if stratum == "action_lattice":
        action_index = local % 192
    else:
        action_index = index % 144
    case = _hopper_base(index, action_index, parameter_record)
    if stratum == "action_lattice":
        if local >= 192:
            _place_hopper_launch_obstacle(case, 0)
    elif stratum == "launch":
        _place_hopper_launch_obstacle(case, (1, 0, -1)[local // 120])
    elif stratum == "arc":
        _place_hopper_arc_prism(
            case, parameter_record, (-1, 0, 1)[local // 200]
        )
    elif stratum == "unknown_boundary":
        if local < 150:
            midpoint = _hopper_trajectory_point(case, parameter_record, 32)
            case["terrain"]["unknown_polygons_mm"] = [
                square_polygon(midpoint[0], midpoint[1], 100)
            ]
        else:
            case["terrain"]["map_bounds_mm"] = [-500, -500, 500, 500]
    elif stratum == "landing_probability":
        witness = ballistic_witness(case, parameter_record)
        halfwidth = 625 + (810, 811, 812)[local // 100]
        case["terrain"]["landing_pad_polygon_mm"] = square_polygon(
            int(witness["landing_x_mm"]),
            int(witness["landing_y_mm"]),
            halfwidth,
        )
    elif stratum == "landing":
        group = local // 50
        if group in {0, 1, 2, 9, 10}:
            desired = {0: 1, 1: 0, 2: -1, 9: 2, 10: 0}[group]
            _place_hopper_landing_obstacle(case, parameter_record, desired)
        elif group in {3, 4, 5}:
            _set_landing_surface_slope(
                case,
                parameter_record,
                {3: 1499, 4: 1500, 5: 1501}[group],
            )
        elif group in {6, 7, 8}:
            case["terrain"]["landing_surface_plane"] = _plane(
                0, origin_z_mm={6: 49, 7: 50, 8: 51}[group]
            )
    elif stratum == "stop_model":
        case["action"] = hopper_action_from_index((local % 3) * 48 + local % 48)
        witness = ballistic_witness(case, parameter_record)
        case["terrain"]["landing_pad_polygon_mm"] = square_polygon(
            int(witness["landing_x_mm"]), int(witness["landing_y_mm"]), 2000
        )
    case["terrain_sha256"] = _terrain_sha("hopper", case["terrain"], index)
    return case


def _generate_platform(
    platform: str,
    quotas: dict[str, int],
    *,
    hopper_parameter_record: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_index = 0
    for stratum, count in quotas.items():
        for local in range(count):
            if platform == "wheel":
                case = _wheel_case(stratum, local, case_index)
            elif platform == "legged":
                case = _legged_case(stratum, local, case_index)
            else:
                case = _hopper_case(
                    stratum, local, case_index, hopper_parameter_record
                )
            row = {
                "case": case,
                "platform_kind": platform,
                "schema_version": "g2-truth-blind-case/v2",
                "stratum": stratum,
            }
            validate_truth_blind_case(row)
            rows.append(row)
            case_index += 1
    return rows


def generate_all_cases(
    specification: dict[str, Any],
    *,
    hopper_parameter_record: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    validate_hopper_parameter_record(hopper_parameter_record)
    quotas = specification["primitive_quotas"]
    generated = {
        platform: _generate_platform(
            platform,
            quotas[platform],
            hopper_parameter_record=hopper_parameter_record,
        )
        for platform in ("wheel", "legged", "hopper")
    }
    expected = {"wheel": 3334, "legged": 3334, "hopper": 3334}
    actual = {platform: len(rows) for platform, rows in generated.items()}
    if actual != expected:
        raise ValueError(f"primitive quota mismatch: {actual}")
    return generated
