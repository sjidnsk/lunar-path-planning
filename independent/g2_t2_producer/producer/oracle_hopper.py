from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from .canonical import sha256_bytes
from .geometry import (
    map_clearance_mm,
    minimum_clearance_mm,
    plane_height_mm,
    plane_slope_cdeg,
    point_in_polygon_closed,
)
from .hopper_energy_evaluator import MODEL_ID as ENERGY_MODEL_ID
from .hopper_energy_evaluator import evaluate_energy
from .hopper_stop_evaluator import MODEL_ID as STOP_MODEL_ID
from .hopper_stop_evaluator import evaluate_stop


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_AUTHORIZED_DECIMALS = {
    "body_envelope_radius_m": "0.375",
    "launch_reference_height_m": "0.750",
    "arc_clearance_margin_m": "0.125",
    "landing_footprint_radius_m": "0.625",
}
_EVALUATOR_FILES = {
    "stop_condition": "producer/hopper_stop_evaluator.py",
    "energy_model": "producer/hopper_energy_evaluator.py",
}
_REQUIRED_FIELDS = (
    "schema_version",
    "parameter_set_id",
    *_AUTHORIZED_DECIMALS,
    "stop_condition",
    "energy_model",
    "evidence_class",
    "simulation_proxy",
    "physical_capability_claimed",
    "formal_evidence_eligible",
    "status",
)


def _producer_root(source_root: Path | None = None) -> Path:
    return (
        Path(source_root).resolve()
        if source_root is not None
        else Path(__file__).resolve().parents[1]
    )


def evaluator_source_bytes(
    source_root: Path | None = None,
) -> dict[str, bytes]:
    root = _producer_root(source_root)
    return {
        name: (root / relative).read_bytes()
        for name, relative in _EVALUATOR_FILES.items()
    }


def build_hopper_parameter_record(
    source_root: Path | None = None,
) -> dict[str, Any]:
    sources = evaluator_source_bytes(source_root)
    return {
        "schema_version": "g2-hopper-candidate/v2",
        "parameter_set_id": "hopper-generic-internal-proxy/v2",
        **_AUTHORIZED_DECIMALS,
        "stop_condition": {
            "model_id": STOP_MODEL_ID,
            "max_touchdown_speed_m_s": "2.500",
            "evaluator_relative_path": _EVALUATOR_FILES["stop_condition"],
            "evaluator_source_sha256": sha256_bytes(sources["stop_condition"]),
        },
        "energy_model": {
            "model_id": ENERGY_MODEL_ID,
            "reference_speed_m_s": "2.500",
            "max_energy_decimal": "1.000000",
            "evaluator_relative_path": _EVALUATOR_FILES["energy_model"],
            "evaluator_source_sha256": sha256_bytes(sources["energy_model"]),
        },
        "evidence_class": "candidate_engineering_proxy",
        "simulation_proxy": True,
        "physical_capability_claimed": False,
        "formal_evidence_eligible": False,
        "status": "pending_external_evidence",
    }


def _positive_decimal(record: Mapping[str, Any], field: str) -> Decimal:
    try:
        value = Decimal(str(record[field]))
    except (InvalidOperation, KeyError) as error:
        raise ValueError(f"invalid {field}") from error
    if not value.is_finite() or value <= 0:
        raise ValueError(f"invalid {field}")
    return value


def validate_hopper_parameter_record(
    record: dict[str, Any],
    *,
    source_root: Path | None = None,
    evaluator_sources: Mapping[str, bytes] | None = None,
) -> None:
    for field in _REQUIRED_FIELDS:
        if field not in record:
            raise ValueError(f"missing {field}")
    if record["schema_version"] != "g2-hopper-candidate/v2":
        raise ValueError("unsupported schema_version")
    if record["parameter_set_id"] != "hopper-generic-internal-proxy/v2":
        raise ValueError("unsupported parameter_set_id")
    for field, expected in _AUTHORIZED_DECIMALS.items():
        actual = format(_positive_decimal(record, field), "f")
        if actual != expected:
            raise ValueError(f"{field} must equal authorized value {expected}")
    if record["evidence_class"] != "candidate_engineering_proxy":
        raise ValueError("evidence_class must be candidate_engineering_proxy")
    if record["simulation_proxy"] is not True:
        raise ValueError("simulation_proxy must be true")
    if record["physical_capability_claimed"] is not False:
        raise ValueError("physical_capability_claimed must be false")
    if record["formal_evidence_eligible"] is not False:
        raise ValueError("formal_evidence_eligible must be false")
    if record["status"] != "pending_external_evidence":
        raise ValueError("status must be pending_external_evidence")

    stop = record["stop_condition"]
    energy = record["energy_model"]
    if stop.get("model_id") != STOP_MODEL_ID:
        raise ValueError("unsupported stop_condition model_id")
    if energy.get("model_id") != ENERGY_MODEL_ID:
        raise ValueError("unsupported energy_model model_id")
    if format(_positive_decimal(stop, "max_touchdown_speed_m_s"), "f") != "2.500":
        raise ValueError("max_touchdown_speed_m_s must equal 2.500")
    if format(_positive_decimal(energy, "reference_speed_m_s"), "f") != "2.500":
        raise ValueError("reference_speed_m_s must equal 2.500")
    if format(_positive_decimal(energy, "max_energy_decimal"), "f") != "1.000000":
        raise ValueError("max_energy_decimal must equal 1.000000")
    sources = (
        dict(evaluator_sources)
        if evaluator_sources is not None
        else evaluator_source_bytes(source_root)
    )
    if set(sources) != set(_EVALUATOR_FILES):
        raise ValueError("evaluator source set must cover stop_condition and energy_model")
    for container_name, container in (
        ("stop_condition", stop),
        ("energy_model", energy),
    ):
        digest = str(container.get("evaluator_source_sha256", ""))
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"invalid {container_name} evaluator source SHA-256")
        if container.get("evaluator_relative_path") != _EVALUATOR_FILES[container_name]:
            raise ValueError(f"{container_name} evaluator relative path mismatch")
        actual = sha256_bytes(sources[container_name])
        if digest != actual:
            raise ValueError(f"{container_name} evaluator source SHA-256 mismatch")


def gaussian_square_mass_ppm(*, halfwidth_mm: int, sigma_mm: int) -> int:
    if halfwidth_mm <= 0 or sigma_mm <= 0:
        raise ValueError("halfwidth_mm and sigma_mm must be positive")
    one_axis = math.erf(halfwidth_mm / (sigma_mm * math.sqrt(2.0)))
    return round(one_axis * one_axis * 1_000_000)


def ballistic_witness(
    case: dict[str, Any], parameter_record: dict[str, Any]
) -> dict[str, int]:
    action = case["action"]
    launch = case.get("launch_pose_mm", [0, 0, 0])
    terrain = case["terrain"]
    plane = terrain["height_plane"]
    speed_m_s = int(action["speed_mm_s"]) / 1000.0
    elevation_rad = math.radians(int(action["elevation_mdeg"]) / 1000.0)
    azimuth_rad = math.radians(int(action["azimuth_mdeg"]) / 1000.0)
    gravity_m_s2 = 1.62
    reference_height_m = float(
        Decimal(str(parameter_record["launch_reference_height_m"]))
    )
    ground_z_mm = plane_height_mm(plane, int(launch[0]), int(launch[1]))
    launch_z_m = (ground_z_mm + round(reference_height_m * 1000)) / 1000.0
    target_ground_m = plane_height_mm(
        plane, int(launch[0]), int(launch[1])
    ) / 1000.0
    vertical = speed_m_s * math.sin(elevation_rad)
    horizontal = speed_m_s * math.cos(elevation_rad)
    discriminant = vertical * vertical + 2.0 * gravity_m_s2 * (
        launch_z_m - target_ground_m
    )
    flight_time_s = (vertical + math.sqrt(max(0.0, discriminant))) / gravity_m_s2
    range_m = horizontal * flight_time_s
    landing_x = int(launch[0]) + round(
        range_m * 1000.0 * math.cos(azimuth_rad)
    )
    landing_y = int(launch[1]) + round(
        range_m * 1000.0 * math.sin(azimuth_rad)
    )
    apex_m = launch_z_m + vertical * vertical / (2.0 * gravity_m_s2)
    return {
        "apex_height_mm": round(apex_m * 1000.0),
        "flight_time_us": round(flight_time_s * 1_000_000.0),
        "landing_x_mm": landing_x,
        "landing_y_mm": landing_y,
        "launch_ground_z_mm": ground_z_mm,
        "range_mm": round(range_m * 1000.0),
    }


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


def _trajectory_points(
    case: dict[str, Any],
    parameter_record: dict[str, Any],
    witness: dict[str, int],
) -> list[tuple[int, int, int]]:
    action = case["action"]
    launch = [int(value) for value in case.get("launch_pose_mm", [0, 0, 0])]
    terrain = case["terrain"]
    plane = terrain["height_plane"]
    speed_m_s = int(action["speed_mm_s"]) / 1000.0
    elevation = math.radians(int(action["elevation_mdeg"]) / 1000.0)
    azimuth = math.radians(int(action["azimuth_mdeg"]) / 1000.0)
    horizontal_x = speed_m_s * math.cos(elevation) * math.cos(azimuth)
    horizontal_y = speed_m_s * math.cos(elevation) * math.sin(azimuth)
    vertical = speed_m_s * math.sin(elevation)
    launch_z_mm = plane_height_mm(plane, launch[0], launch[1]) + round(
        Decimal(str(parameter_record["launch_reference_height_m"])) * 1000
    )
    flight_s = witness["flight_time_us"] / 1_000_000.0
    points: list[tuple[int, int, int]] = []
    for index in range(65):
        time_s = flight_s * index / 64.0
        points.append(
            (
                launch[0] + round(horizontal_x * time_s * 1000.0),
                launch[1] + round(horizontal_y * time_s * 1000.0),
                launch_z_mm
                + round(
                    (vertical * time_s - 0.5 * 1.62 * time_s * time_s)
                    * 1000.0
                ),
            )
        )
    return points


def evaluate_hopper(
    case: dict[str, Any],
    parameter_record: dict[str, Any],
    *,
    source_root: Path | None = None,
    evaluator_sources: Mapping[str, bytes] | None = None,
) -> dict[str, Any]:
    try:
        validate_hopper_parameter_record(
            parameter_record,
            source_root=source_root,
            evaluator_sources=evaluator_sources,
        )
    except (OSError, ValueError) as error:
        return _decision(
            False,
            "G2I_H_PARAMETER_RECORD_UNAPPROVED",
            witness={"parameter_error": str(error)},
            slacks={},
        )
    if case.get("numeric_state") != "decided":
        return _decision(
            False,
            "G2I_NUMERIC_UNDECIDED",
            witness={"numeric_state": str(case.get("numeric_state"))},
            slacks={},
        )
    action = case["action"]
    expected_speeds = (1500, 2000, 2500, 3000)
    expected_elevations = (30000, 45000, 60000)
    indices = (
        int(action["speed_index"]),
        int(action["elevation_index"]),
        int(action["azimuth_index"]),
    )
    if not (
        0 <= indices[0] < 4
        and 0 <= indices[1] < 3
        and 0 <= indices[2] < 16
        and int(action["speed_mm_s"]) == expected_speeds[indices[0]]
        and int(action["elevation_mdeg"]) == expected_elevations[indices[1]]
        and int(action["azimuth_mdeg"]) == indices[2] * 22500
    ):
        return _decision(
            False,
            "G2I_H_MODEL_DOMAIN",
            witness={"action_indices": list(indices)},
            slacks={},
        )

    terrain = case["terrain"]
    witness = ballistic_witness(case, parameter_record)
    trajectory = _trajectory_points(case, parameter_record, witness)
    xy_points = [(x, y) for x, y, _ in trajectory]
    body_radius = round(
        Decimal(str(parameter_record["body_envelope_radius_m"])) * 1000
    )
    arc_margin = round(
        Decimal(str(parameter_record["arc_clearance_margin_m"])) * 1000
    )
    landing_radius = round(
        Decimal(str(parameter_record["landing_footprint_radius_m"])) * 1000
    )
    obstacle_polygons = terrain.get("obstacle_polygons_mm", [])
    unknown_polygons = terrain.get("unknown_polygons_mm", [])
    bounds = terrain["map_bounds_mm"]
    launch_xy = xy_points[0]
    landing_xy = (witness["landing_x_mm"], witness["landing_y_mm"])
    launch_clearance = min(
        map_clearance_mm([launch_xy], bounds, body_radius),
        minimum_clearance_mm(
            [launch_xy], obstacle_polygons, body_radius
        ),
    )
    arc_map_clearance = map_clearance_mm(
        xy_points, bounds, body_radius + arc_margin
    )
    arc_unknown = any(
        point_in_polygon_closed(point, polygon)
        for point in xy_points
        for polygon in unknown_polygons
    )
    obstacle_prisms = terrain.get("obstacle_prisms", [])
    arc_clearances: list[int] = []
    for x, y, z in trajectory[1:-1]:
        for prism in obstacle_prisms:
            horizontal = minimum_clearance_mm(
                [(x, y)], [prism["polygon_mm"]], body_radius + arc_margin
            )
            vertical = z - int(prism["top_z_mm"]) - body_radius - arc_margin
            arc_clearances.append(max(horizontal, vertical))
    arc_clearance = min(arc_clearances, default=2**30)
    landing_pad = terrain["landing_pad_polygon_mm"]
    pad_edge_distance = _signed_distance_to_pad_edge(landing_xy, landing_pad)
    landing_pad_clearance = min(
        pad_edge_distance - landing_radius,
        minimum_clearance_mm(
            [landing_xy], obstacle_polygons, landing_radius
        ),
    )
    landing_halfwidth = max(
        1,
        pad_edge_distance - landing_radius,
    )
    landing_mass = gaussian_square_mass_ppm(
        halfwidth_mm=landing_halfwidth,
        sigma_mm=int(terrain["landing_sigma_mm"]),
    )
    landing_plane = terrain.get("landing_surface_plane", terrain["height_plane"])
    slope = plane_slope_cdeg(landing_plane)
    landing_height_error = abs(
        trajectory[-1][2]
        - plane_height_mm(
            landing_plane, landing_xy[0], landing_xy[1]
        )
    )
    stop = evaluate_stop(
        action_speed_mm_s=int(action["speed_mm_s"]),
        max_touchdown_speed_mm_s=round(
            Decimal(
                str(parameter_record["stop_condition"]["max_touchdown_speed_m_s"])
            )
            * 1000
        ),
    )
    energy = evaluate_energy(
        action_speed_mm_s=int(action["speed_mm_s"]),
        reference_speed_mm_s=round(
            Decimal(
                str(parameter_record["energy_model"]["reference_speed_m_s"])
            )
            * 1000
        ),
        max_energy_decimal=str(
            parameter_record["energy_model"]["max_energy_decimal"]
        ),
    )
    witness.update(
        {
            "action_indices": list(indices),
            "arc_sample_count": len(trajectory),
            "body_envelope_radius_mm": body_radius,
            "arc_clearance_margin_mm": arc_margin,
            "landing_footprint_radius_mm": landing_radius,
            "landing_halfwidth_mm": landing_halfwidth,
            "landing_mass_ppm": landing_mass,
            "landing_slope_cdeg": slope,
            "energy_decimal": energy["energy_decimal"],
            "stop_evaluator_source_sha256": parameter_record["stop_condition"][
                "evaluator_source_sha256"
            ],
            "energy_evaluator_source_sha256": parameter_record["energy_model"][
                "evaluator_source_sha256"
            ],
        }
    )
    slacks = {
        "launch_clearance_mm": launch_clearance,
        "arc_map_clearance_mm": arc_map_clearance,
        "arc_clearance_mm": arc_clearance,
        "landing_mass_ppm": landing_mass - 990000,
        "landing_footprint_mm": landing_pad_clearance,
        "landing_slope_cdeg": 1500 - slope,
        "landing_height_mm": int(terrain.get("landing_height_tolerance_mm", 50))
        - landing_height_error,
        "stop_speed_mm_s": int(stop["slack_mm_s"]),
    }
    if launch_clearance <= 0:
        return _decision(
            False, "G2I_H_LAUNCH_FOOTPRINT", witness=witness, slacks=slacks
        )
    if arc_map_clearance < 0:
        return _decision(
            False, "G2I_H_ARC_OUTSIDE", witness=witness, slacks=slacks
        )
    if arc_unknown:
        return _decision(
            False, "G2I_H_ARC_UNKNOWN", witness=witness, slacks=slacks
        )
    if arc_clearance <= 0:
        return _decision(
            False, "G2I_H_ARC_CLEARANCE", witness=witness, slacks=slacks
        )
    if slacks["landing_mass_ppm"] < 0:
        return _decision(
            False, "G2I_H_LANDING_MASS", witness=witness, slacks=slacks
        )
    if landing_pad_clearance <= 0:
        return _decision(
            False, "G2I_H_LANDING_FOOTPRINT", witness=witness, slacks=slacks
        )
    if slacks["landing_slope_cdeg"] < 0:
        return _decision(
            False, "G2I_H_LANDING_SLOPE", witness=witness, slacks=slacks
        )
    if slacks["landing_height_mm"] < 0:
        return _decision(
            False, "G2I_H_LANDING_HEIGHT", witness=witness, slacks=slacks
        )
    if not bool(stop["accepted"]):
        return _decision(False, "G2I_H_STOP", witness=witness, slacks=slacks)
    if not bool(energy["accepted"]):
        return _decision(
            False, "G2I_H_MODEL_DOMAIN", witness=witness, slacks=slacks
        )
    return _decision(True, "G2I_H_SAFE", witness=witness, slacks=slacks)


def _signed_distance_to_pad_edge(
    point: tuple[int, int], polygon: list[list[int]]
) -> int:
    xs = [int(vertex[0]) for vertex in polygon]
    ys = [int(vertex[1]) for vertex in polygon]
    if not (
        min(xs) <= point[0] <= max(xs)
        and min(ys) <= point[1] <= max(ys)
    ):
        return -round(
            math.hypot(
                max(min(xs) - point[0], 0, point[0] - max(xs)),
                max(min(ys) - point[1], 0, point[1] - max(ys)),
            )
        )
    return min(
        point[0] - min(xs),
        max(xs) - point[0],
        point[1] - min(ys),
        max(ys) - point[1],
    )
