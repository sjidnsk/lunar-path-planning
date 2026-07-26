from __future__ import annotations

import math
from typing import Any

from .canonical import canonical_json_bytes, derive_seed, domain_hash
from .models import validate_truth_blind_case
from .oracle_hopper import validate_hopper_parameter_record


_CONTROL_FAMILIES = (
    "straight_forward",
    "straight_reverse",
    "rotate_left",
    "rotate_right",
    "arc_left",
    "arc_right",
    "hold",
)


def _terrain_sha(platform: str, terrain: dict[str, Any], case_index: int) -> str:
    return domain_hash(
        "g2-terrain-source/v1",
        platform.encode("ascii"),
        str(case_index).encode("ascii"),
        canonical_json_bytes(terrain),
    )


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
        "endpoint_cells_clear": True,
        "inside_map": True,
        "known_sweep": True,
        "slope_cdeg": (index % 21) * 10,
        "sweep_clearance_mm": 25,
        "traversable": True,
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


def _wheel_case(stratum: str, local: int, index: int) -> dict[str, Any]:
    case = _wheel_base(index)
    if stratum == "slope":
        case["terrain"]["slope_cdeg"] = (2999, 3000, 3001)[local // 150]
    elif stratum == "obstacle":
        case["terrain"]["sweep_clearance_mm"] = (1, 0, -1)[local // 150]
    elif stratum == "knownness_boundary":
        if local < 150:
            case["terrain"]["known_sweep"] = False
        else:
            case["terrain"]["inside_map"] = False
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
        case["terrain"]["endpoint_cells_clear"] = True
        case["terrain"]["sweep_clearance_mm"] = 1 if local < 202 else -1
    elif stratum == "compound":
        case["primitive"]["speed_um_s"] = 1_000_001
        case["terrain"]["sweep_clearance_mm"] = -1
        _regenerate_wheel_endpoint(case)
    case["terrain_sha256"] = _terrain_sha("wheel", case["terrain"], index)
    return case


def _legged_base(index: int) -> dict[str, Any]:
    leg_id = index % 4
    terrain = {
        "body_slope_cdeg": (index % 25) * 10,
        "body_sweep_clearance_mm": 25,
        "foothold_obstacle": False,
        "foothold_slope_cdeg": (index % 20) * 10,
        "foothold_traversable": True,
        "foothold_unknown": False,
        "grid_valid": True,
        "support_margin_mm": 75,
    }
    case = {
        "case_index": index,
        "determinism_seed": derive_seed(20260727, "legged", "labels", "case", index),
        "moving_leg_id": leg_id,
        "numeric_state": "decided",
        "phase": leg_id,
        "required_phase": leg_id,
        "source_foot_mm": [0, 0, 0],
        "target": terrain,
        "target_foot_mm": [300, 0, 0],
    }
    case["terrain_sha256"] = _terrain_sha("legged", terrain, index)
    return case


def _legged_case(stratum: str, local: int, index: int) -> dict[str, Any]:
    case = _legged_base(index)
    if stratum == "foothold":
        group = local // 80
        if group == 0:
            case["target"]["foothold_slope_cdeg"] = 2499
        elif group == 1:
            case["target"]["foothold_slope_cdeg"] = 2500
        elif group == 2:
            case["target"]["foothold_slope_cdeg"] = 2501
        elif group == 3:
            case["target"]["foothold_unknown"] = True
        elif group == 4:
            case["target"]["foothold_obstacle"] = True
        else:
            case["target"]["foothold_traversable"] = False
    elif stratum == "step_length":
        case["target_foot_mm"] = [(499, 500, 501)[local // 120], 0, 0]
    elif stratum == "step_height":
        case["target_foot_mm"] = [0, 0, (249, 250, 251)[local // 120]]
    elif stratum == "support":
        case["target"]["support_margin_mm"] = (49, 50, 51)[local // 160]
    elif stratum == "body_sweep":
        case["target"]["body_sweep_clearance_mm"] = (1, 0, -1)[local // 140]
    elif stratum == "sequence_grid":
        if local < 71:
            case["phase"] = (int(case["required_phase"]) + 1) % 4
        elif local < 142:
            case["target"]["grid_valid"] = False
    elif stratum == "compound":
        case["phase"] = (int(case["required_phase"]) + 1) % 4
        case["target"]["foothold_unknown"] = True
    case["terrain_sha256"] = _terrain_sha("legged", case["target"], index)
    return case


def _action_from_index(action_index: int) -> dict[str, int]:
    speed_index = action_index // (3 * 16)
    remainder = action_index % (3 * 16)
    elevation_index = remainder // 16
    azimuth_index = remainder % 16
    return {
        "azimuth_index": azimuth_index,
        "azimuth_mdeg": azimuth_index * 22500,
        "elevation_index": elevation_index,
        "elevation_mdeg": (30000, 45000, 60000)[elevation_index],
        "speed_index": speed_index,
        "speed_mm_s": (1000, 1500, 2000, 2500)[speed_index],
    }


def _hopper_base(index: int, action_index: int) -> dict[str, Any]:
    terrain = {
        "arc_clearance_slack_mm": 25,
        "arc_inside_map": True,
        "arc_known": True,
        "landing_footprint_clearance_mm": 25,
        "landing_height_error_mm": 0,
        "landing_height_tolerance_mm": 50,
        "landing_halfwidth_mm": 5000,
        "landing_sigma_mm": 289,
        "landing_slope_cdeg": (index % 15) * 50,
        "launch_clearance_mm": 25,
    }
    action = _action_from_index(action_index)
    case = {
        "action": action,
        "case_index": index,
        "determinism_seed": derive_seed(20260727, "hopper", "labels", "case", index),
        "numeric_state": "decided",
        "terrain": terrain,
        "touchdown_speed_mm_s": min(int(action["speed_mm_s"]), 2500),
    }
    case["terrain_sha256"] = _terrain_sha("hopper", terrain, index)
    return case


def _hopper_case(stratum: str, local: int, index: int) -> dict[str, Any]:
    action_index = local % 192 if stratum == "action_lattice" else index % 192
    case = _hopper_base(index, action_index)
    if stratum == "action_lattice":
        if local >= 192:
            case["terrain"]["launch_clearance_mm"] = 0
    elif stratum == "launch":
        case["terrain"]["launch_clearance_mm"] = (1, 0, -1)[local // 120]
    elif stratum == "arc":
        case["terrain"]["arc_clearance_slack_mm"] = (-1, 0, 1)[local // 200]
    elif stratum == "unknown_boundary":
        if local < 150:
            case["terrain"]["arc_known"] = False
        else:
            case["terrain"]["arc_inside_map"] = False
    elif stratum == "landing_probability":
        case["terrain"]["landing_halfwidth_mm"] = (810, 811, 812)[local // 100]
    elif stratum == "landing":
        group = local // 50
        if group == 0:
            case["terrain"]["landing_footprint_clearance_mm"] = 1
        elif group == 1:
            case["terrain"]["landing_footprint_clearance_mm"] = 0
        elif group == 2:
            case["terrain"]["landing_footprint_clearance_mm"] = -1
        elif group == 3:
            case["terrain"]["landing_slope_cdeg"] = 1499
        elif group == 4:
            case["terrain"]["landing_slope_cdeg"] = 1500
        elif group == 5:
            case["terrain"]["landing_slope_cdeg"] = 1501
        elif group == 6:
            case["terrain"]["landing_height_error_mm"] = 49
        elif group == 7:
            case["terrain"]["landing_height_error_mm"] = 50
        elif group == 8:
            case["terrain"]["landing_height_error_mm"] = 51
        elif group == 9:
            case["terrain"]["landing_footprint_clearance_mm"] = 2
        else:
            case["terrain"]["landing_footprint_clearance_mm"] = 0
    elif stratum == "stop_model":
        group = local // 54
        if group == 0:
            case["touchdown_speed_mm_s"] = 2499
        elif group == 1:
            case["touchdown_speed_mm_s"] = 2500
        elif group == 2:
            case["touchdown_speed_mm_s"] = 2501
        elif group == 3:
            case["touchdown_speed_mm_s"] = -1
        elif group == 4:
            case["touchdown_speed_mm_s"] = 2000
        else:
            case["touchdown_speed_mm_s"] = 1000
    case["terrain_sha256"] = _terrain_sha("hopper", case["terrain"], index)
    return case


def _generate_platform(
    platform: str,
    quotas: dict[str, int],
) -> list[dict[str, Any]]:
    constructor = {
        "wheel": _wheel_case,
        "legged": _legged_case,
        "hopper": _hopper_case,
    }[platform]
    rows: list[dict[str, Any]] = []
    case_index = 0
    for stratum, count in quotas.items():
        for local in range(count):
            case = constructor(stratum, local, case_index)
            row = {
                "case": case,
                "platform_kind": platform,
                "schema_version": "g2-truth-blind-case/v1",
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
        platform: _generate_platform(platform, quotas[platform])
        for platform in ("wheel", "legged", "hopper")
    }
    expected = {"wheel": 3334, "legged": 3334, "hopper": 3334}
    actual = {platform: len(rows) for platform, rows in generated.items()}
    if actual != expected:
        raise ValueError(f"primitive quota mismatch: {actual}")
    return generated
