from __future__ import annotations

import copy
import math
import struct
from collections import defaultdict
from decimal import ROUND_CEILING, Decimal
from typing import Any, Iterable

import numpy as np

from .canonical import (
    canonical_json_bytes,
    canonical_npz_bytes,
    decode_canonical_npz,
    derive_seed,
    domain_hash,
    sha256_bytes,
)
from .finite_graph import reachability_certificate
from .geometry import (
    fine_cells_intersecting_capsule,
    fine_cells_intersecting_disk,
    fine_slope_cdeg,
    plane_height_mm,
    relative_relief_um,
    square_polygon,
)
from .models import (
    truth_request_identity,
    validate_provider_local_snapshot,
    validate_producer_specification,
    validate_raw_source_reject,
    validate_request_graph,
    validate_truth_blind_case,
    validate_truth_row,
)
from .oracle_hopper import (
    ballistic_witness,
    evaluate_hopper,
    validate_hopper_parameter_record,
)
from .oracle_legged import evaluate_legged
from .oracle_wheel import evaluate_wheel, wheel_endpoint_mm


_PLATFORMS = ("wheel", "legged", "hopper")
_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_STANDARD_TERRAIN_SIZE = 20
_KILOMETER_TERRAIN_SIZE = 5
_PROVIDER_FINE_RESOLUTION_MM = 500
_LOCAL_FRAME_ID = "g2-local-metric-frame-mm/v1"
_LOCAL_NODE_ORIGIN_MM = 1250
_LEGGED_NODE_ORIGIN_MM = 1250
_REQUEST_DIRECTIONS = ((1, 0),)
_ENDPOINT_ADMISSION_POLICY = (
    "actual-platform-endpoint-safe-then-frame-rank/v1"
)


def _graph_shape(platform: str) -> tuple[int, int]:
    if platform == "legged":
        return 9, 5
    if platform in {"wheel", "hopper"}:
        return 5, 5
    raise ValueError(f"unknown request platform: {platform}")


def _request_start_goal(platform: str) -> tuple[str, str]:
    goal_x = {"wheel": 2, "legged": 1, "hopper": 1}.get(platform)
    if goal_x is None:
        raise ValueError(f"unknown request platform: {platform}")
    return "n:0:2", f"n:{goal_x}:2"


def _frontier_cut_node(platform: str) -> str:
    goal_x = int(_request_start_goal(platform)[1].split(":")[1])
    return f"n:{goal_x // 2}:2"


def _require_hopper_parameter_record(
    record: dict[str, Any] | None,
) -> dict[str, Any]:
    if record is None:
        raise ValueError("G2I_BLOCKED_HOPPER_PARAMETER_RECORD_MISSING")
    try:
        validate_hopper_parameter_record(record)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"G2I_BLOCKED_HOPPER_PARAMETER_RECORD_INVALID: {error}"
        ) from error
    return record


def _legged_cycle_contract(specification: dict[str, Any]) -> dict[str, Any]:
    cycle = specification.get("legged_crawl_cycle")
    if not isinstance(cycle, dict):
        raise ValueError("independent legged crawl cycle is missing")
    if cycle.get("schema_version") != "g2-independent-static-crawl-cycle/v2":
        raise ValueError("independent legged crawl cycle schema mismatch")
    expected_order = [
        "front_left",
        "front_right",
        "rear_left",
        "rear_right",
    ]
    if cycle.get("foot_storage_order") != expected_order:
        raise ValueError("independent legged foot storage order mismatch")
    expected_steps = (
        (0, 3, "rear_right", [250, 0]),
        (1, 1, "front_right", [250, 0]),
        (2, 2, "rear_left", [250, 0]),
        (3, 0, "front_left", [250, 0]),
    )
    phase_steps = cycle.get("phase_steps")
    if not isinstance(phase_steps, list) or len(phase_steps) != 4:
        raise ValueError("independent legged crawl cycle must have four phases")
    for step, expected in zip(phase_steps, expected_steps, strict=True):
        phase, leg_id, leg_name, delta = expected
        if (
            step.get("phase") != phase
            or step.get("moving_leg_id") != leg_id
            or step.get("moving_leg") != leg_name
            or step.get("forward_lateral_delta_mm") != delta
        ):
            raise ValueError("independent legged crawl phase drifted")
    if (
        cycle.get("nominal_stance_offsets_mm")
        != [[350, 250], [350, -250], [-350, 250], [-350, -250]]
        or cycle.get("cycle_body_delta_xy_mm") != [250, 0]
        or cycle.get("maximum_step_length_mm") != 500
        or cycle.get("minimum_support_margin_mm") != 50
        or cycle.get("body_pose_rule")
        != "support-safe-staging-then-nominal-terminal/v1"
        or cycle.get("lift_body_pose_rule")
        != "three-support-foot-arithmetic-mean/v1"
        or cycle.get("vertical_body_delta_rule")
        != "four-foot-height-plane-mean/v1"
        or cycle.get("provider_phase_rotation")
        != {
            "producer_phase_zero_maps_to_provider_phase": 1,
            "schema_version": "g2-leg-phase-rotation/v1",
        }
    ):
        raise ValueError("independent legged crawl geometry drifted")
    return cycle


def _hopper_ballistic_contract(specification: dict[str, Any]) -> dict[str, Any]:
    contract = specification.get("hopper_request_ballistics")
    if not isinstance(contract, dict):
        raise ValueError("independent Hopper ballistic contract is missing")
    if (
        contract.get("schema_version")
        != "g2-independent-symmetric-ballistic-contract/v1"
        or contract.get("gravity_m_s2") != "1.62"
        or contract.get("launch_reference_height_semantics")
        != "same-support-body-reference-height/v1"
        or contract.get("endpoint_encoding")
        != "ieee754-binary64-hex-and-word/v1"
    ):
        raise ValueError("independent Hopper ballistic contract drifted")
    expected_order = [
        "horizontal_speed=speed*cos(elevation)",
        "vertical_speed=speed*sin(elevation)",
        "flight_time=2*(vertical_speed/gravity)",
        "horizontal_delta=(horizontal_speed*direction)*flight_time",
        "endpoint=start+horizontal_delta",
    ]
    if contract.get("calculation_order") != expected_order:
        raise ValueError("independent Hopper calculation order drifted")
    return contract


def _binary64_identity(value: float) -> dict[str, str]:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("binary64 identity requires a finite exact float")
    canonical = 0.0 if value == 0.0 else value
    word = struct.unpack(">Q", struct.pack(">d", canonical))[0]
    return {
        "hex": canonical.hex(),
        "word_hex": f"{word:016x}",
    }


def _hopper_exact_kinematics(
    action: dict[str, Any],
    *,
    direction: tuple[int, int],
    ballistic_contract: dict[str, Any],
) -> dict[str, Any]:
    speed_m_s = int(action["speed_mm_s"]) / 1000.0
    elevations = (math.pi / 6.0, math.pi / 4.0, math.pi / 3.0)
    elevation_rad = elevations[int(action["elevation_index"])]
    gravity_m_s2 = float(ballistic_contract["gravity_m_s2"])
    horizontal_speed = speed_m_s * math.cos(elevation_rad)
    vertical_speed = speed_m_s * math.sin(elevation_rad)
    flight_time_s = 2.0 * (vertical_speed / gravity_m_s2)
    direction_x, direction_y = direction
    delta_x_m = (horizontal_speed * float(direction_x)) * flight_time_s
    delta_y_m = (horizontal_speed * float(direction_y)) * flight_time_s
    horizontal_range_m = horizontal_speed * flight_time_s
    return {
        "delta_x_m_hex": _binary64_identity(delta_x_m)["hex"],
        "delta_x_m_word_hex": _binary64_identity(delta_x_m)["word_hex"],
        "delta_y_m_hex": _binary64_identity(delta_y_m)["hex"],
        "delta_y_m_word_hex": _binary64_identity(delta_y_m)["word_hex"],
        "elevation_rad_hex": _binary64_identity(elevation_rad)["hex"],
        "flight_time_s_hex": _binary64_identity(flight_time_s)["hex"],
        "flight_time_s_word_hex": _binary64_identity(flight_time_s)["word_hex"],
        "gravity_m_s2_hex": _binary64_identity(gravity_m_s2)["hex"],
        "horizontal_range_m_hex": _binary64_identity(horizontal_range_m)["hex"],
        "horizontal_range_m_word_hex": _binary64_identity(
            horizontal_range_m
        )["word_hex"],
        "launch_reference_height_semantics": ballistic_contract[
            "launch_reference_height_semantics"
        ],
        "schema_version": "g2-provider-exact-symmetric-ballistic-kinematics/v1",
        "speed_m_s_hex": _binary64_identity(speed_m_s)["hex"],
    }


def _hopper_support_envelope_max_residual_um(
    *,
    gradient_x_ppm: int,
    gradient_y_ppm: int,
    delta_x_m: float,
    delta_y_m: float,
    launch_radius_mm: int,
    landing_radius_mm: int,
) -> int:
    gradient_x = Decimal(int(gradient_x_ppm))
    gradient_y = Decimal(int(gradient_y_ppm))
    gradient_norm = (
        gradient_x * gradient_x + gradient_y * gradient_y
    ).sqrt()
    center_residual_um = abs(
        gradient_x * Decimal.from_float(float(delta_x_m))
        + gradient_y * Decimal.from_float(float(delta_y_m))
    )
    launch_edge_residual_um = (
        gradient_norm * Decimal(int(launch_radius_mm)) / Decimal(1000)
    )
    landing_edge_residual_um = center_residual_um + (
        gradient_norm * Decimal(int(landing_radius_mm)) / Decimal(1000)
    )
    maximum = max(launch_edge_residual_um, landing_edge_residual_um)
    return int(maximum.to_integral_value(rounding=ROUND_CEILING))


def _binary64_pose_identity(
    x_m: float, y_m: float, heading_rad: float
) -> dict[str, str]:
    x_identity = _binary64_identity(x_m)
    y_identity = _binary64_identity(y_m)
    heading_identity = _binary64_identity(heading_rad)
    return {
        "heading_rad_hex": heading_identity["hex"],
        "heading_rad_word_hex": heading_identity["word_hex"],
        "x_m_hex": x_identity["hex"],
        "x_m_word_hex": x_identity["word_hex"],
        "y_m_hex": y_identity["hex"],
        "y_m_word_hex": y_identity["word_hex"],
    }


def _hopper_exact_node_poses(
    ballistic_contract: dict[str, Any],
) -> dict[str, dict[str, str]]:
    action = _action_envelope("hopper")["actions"][0]
    kinematics = _hopper_exact_kinematics(
        action,
        direction=(1, 0),
        ballistic_contract=ballistic_contract,
    )
    spacing_m = float.fromhex(kinematics["horizontal_range_m_hex"])
    origin_m = _LOCAL_NODE_ORIGIN_MM / 1000.0
    axis_values = [origin_m]
    for _ in range(4):
        axis_values.append(axis_values[-1] + spacing_m)
    return {
        f"n:{x}:{y}": _binary64_pose_identity(
            axis_values[x],
            axis_values[y],
            0.0,
        )
        for x in range(5)
        for y in range(5)
    }


def _pose_binary64_from_display(
    pose_mm_urad: list[int],
) -> dict[str, str]:
    return _binary64_pose_identity(
        int(pose_mm_urad[0]) / 1000.0,
        int(pose_mm_urad[1]) / 1000.0,
        int(pose_mm_urad[2]) / 1_000_000.0,
    )


def _binary64_point_identity(
    x_m: float,
    y_m: float,
    z_m: float,
) -> dict[str, Any]:
    point: dict[str, Any] = {}
    for name, value in (("x_m", x_m), ("y_m", y_m), ("z_m", z_m)):
        identity = _binary64_identity(value)
        point[f"{name}_hex"] = identity["hex"]
        point[f"{name}_word_hex"] = identity["word_hex"]
    return point


def _snapshot_index(
    coordinate_mm: float,
    *,
    resolution_mm: int = _PROVIDER_FINE_RESOLUTION_MM,
    size: int = 20,
) -> int:
    return max(0, min(size - 1, int(math.floor(coordinate_mm / resolution_mm))))


def _snapshot_elevation_um(
    snapshot: dict[str, Any],
    x_mm: float,
    y_mm: float,
) -> int:
    column = _snapshot_index(x_mm)
    row = _snapshot_index(y_mm)
    return int(snapshot["arrays"]["elevation_um"][row][column])


def _snapshot_cell_record(
    snapshot: dict[str, Any],
    cell_xy: list[int],
) -> dict[str, Any]:
    column, row = (int(value) for value in cell_xy)
    height, width = (
        int(value) for value in snapshot["shape_height_width"]
    )
    if not 0 <= column < width or not 0 <= row < height:
        return {
            "cell_xy": [column, row],
            "hard_obstacle": True,
            "known": False,
            "out_of_bounds": True,
            "slope_cdeg": 90_00,
            "traversable": False,
        }
    arrays = snapshot["arrays"]
    return {
        "cell_xy": [column, row],
        "confidence_ppm": int(arrays["confidence_ppm"][row][column]),
        "elevation_um": int(arrays["elevation_um"][row][column]),
        "hard_obstacle": bool(arrays["hard_obstacle"][row][column]),
        "known": bool(arrays["known"][row][column]),
        "out_of_bounds": False,
        "slope_cdeg": int(arrays["slope_cdeg"][row][column]),
        "traversable": bool(arrays["traversable"][row][column]),
    }


def _snapshot_cells_safe(
    records: list[dict[str, Any]],
    *,
    maximum_slope_cdeg: int = 3000,
) -> bool:
    return bool(records) and all(
        record["known"]
        and not record["hard_obstacle"]
        and record["traversable"]
        and int(record["slope_cdeg"]) <= maximum_slope_cdeg
        for record in records
    )


def _snapshot_tangent_plane(
    snapshot: dict[str, Any],
    x_mm: int,
    y_mm: int,
) -> dict[str, int]:
    column = _snapshot_index(x_mm)
    row = _snapshot_index(y_mm)
    elevation = snapshot["arrays"]["elevation_um"]
    height = len(elevation)
    width = len(elevation[0])
    left = max(0, column - 1)
    right = min(width - 1, column + 1)
    lower = max(0, row - 1)
    upper = min(height - 1, row + 1)
    gradient_x_ppm = round(
        (
            int(elevation[row][right]) - int(elevation[row][left])
        )
        * 1000
        / (max(1, right - left) * _PROVIDER_FINE_RESOLUTION_MM)
    )
    gradient_y_ppm = round(
        (
            int(elevation[upper][column]) - int(elevation[lower][column])
        )
        * 1000
        / (max(1, upper - lower) * _PROVIDER_FINE_RESOLUTION_MM)
    )
    return {
        "gradient_x_ppm": gradient_x_ppm,
        "gradient_y_ppm": gradient_y_ppm,
        "origin_x_mm": int(x_mm),
        "origin_y_mm": int(y_mm),
        "origin_z_mm": round(int(elevation[row][column]) / 1000),
    }


def _provider_local_snapshot(
    *,
    platform: str,
    scale: str,
    source_terrain_arrays: dict[str, Any],
    terrain_provenance: dict[str, Any],
    local_height_plane: dict[str, int],
    node_metric_poses_mm_urad: dict[str, list[int]],
    hopper_parameter_record: dict[str, Any],
    base_index: int,
) -> dict[str, Any]:
    shape = (20, 20)
    if scale == "standard":
        source_height = source_terrain_arrays["height_mm"]
        if len(source_height) != 20 or any(len(row) != 20 for row in source_height):
            raise ValueError("standard local source must be exactly 20x20")
        elevation_um = [
            [int(value) * 1000 for value in row]
            for row in source_height
        ]
    elif scale == "kilometer":
        gradient_x = int(local_height_plane["gradient_x_ppm"])
        gradient_y = int(local_height_plane["gradient_y_ppm"])
        origin_um = int(local_height_plane["origin_z_mm"]) * 1000
        elevation_um = [
            [
                origin_um
                + round(
                    (
                        gradient_x * (column * _PROVIDER_FINE_RESOLUTION_MM)
                        + gradient_y * (row * _PROVIDER_FINE_RESOLUTION_MM)
                    )
                    / 1000
                )
                for column in range(shape[1])
            ]
            for row in range(shape[0])
        ]
    else:
        raise ValueError(f"unknown request scale: {scale}")

    mode = _local_proxy_difficulty_mode(base_index)
    start_node, goal_node = _request_start_goal(platform)
    start_pose = node_metric_poses_mm_urad[start_node]
    goal_pose = node_metric_poses_mm_urad[goal_node]
    start_column = _snapshot_index(int(start_pose[0]))
    start_row = _snapshot_index(int(start_pose[1]))
    goal_column = _snapshot_index(int(goal_pose[0]))
    goal_row = _snapshot_index(int(goal_pose[1]))
    proxy_operations: list[dict[str, Any]] = []
    if mode == "near_threshold":
        if platform == "hopper":
            required_landing_cells = fine_cells_intersecting_disk(
                center_x_mm=float(goal_pose[0]),
                center_y_mm=float(goal_pose[1]),
                radius_mm=_hopper_landing_required_radius_mm(
                    hopper_parameter_record
                ),
                resolution_mm=_PROVIDER_FINE_RESOLUTION_MM,
                shape_height_width=shape,
            )
            endpoint_authority_cells = {
                tuple(cell)
                for cell in (
                    fine_cells_intersecting_disk(
                        center_x_mm=float(start_pose[0]),
                        center_y_mm=float(start_pose[1]),
                        radius_mm=500,
                        resolution_mm=_PROVIDER_FINE_RESOLUTION_MM,
                        shape_height_width=shape,
                    )
                    + fine_cells_intersecting_disk(
                        center_x_mm=float(goal_pose[0]),
                        center_y_mm=float(goal_pose[1]),
                        radius_mm=625,
                        resolution_mm=_PROVIDER_FINE_RESOLUTION_MM,
                        shape_height_width=shape,
                    )
                )
            }
            internal_support_cells = sorted(
                (
                    tuple(cell)
                    for cell in required_landing_cells
                    if tuple(cell) not in endpoint_authority_cells
                ),
                key=lambda cell: (cell[1], cell[0]),
            )
            if not internal_support_cells:
                raise ValueError(
                    "Hopper near-threshold witness lacks an internal support cell"
                )
            affected_cells = [list(internal_support_cells[0])]
            affected_column, affected_row = affected_cells[0]
            before_um = int(elevation_um[affected_row][affected_column])
            elevation_um[affected_row][affected_column] = (
                int(elevation_um[start_row][start_column]) + 49_000
            )
            after_um = int(elevation_um[affected_row][affected_column])
        else:
            affected_cells = [[goal_column, goal_row]]
            affected_column, affected_row = affected_cells[0]
            before_um = int(elevation_um[affected_row][affected_column])
            elevation_um[affected_row][affected_column] = before_um + 49_000
            after_um = int(elevation_um[affected_row][affected_column])
        proxy_operations.append(
            {
                "after_elevation_um": after_um,
                "affected_cell_xy": affected_cells,
                "before_elevation_um": before_um,
                "cell_xy": [affected_column, affected_row],
                "operation_kind": (
                    "support-relief-near-threshold-elevation-override/v1"
                ),
                "reference_cell_xy": [start_column, start_row],
            }
        )

    input_elevation_um = [list(row) for row in elevation_um]
    input_anchor_elevation_um = int(
        input_elevation_um[start_row][start_column]
    )
    delta_z_um = -input_anchor_elevation_um
    elevation_um = [
        [int(value) + delta_z_um for value in row]
        for row in input_elevation_um
    ]
    input_elevation_sha = domain_hash(
        "g2-provider-local-elevation-um/v1",
        canonical_json_bytes(input_elevation_um),
    )
    output_elevation_sha = domain_hash(
        "g2-provider-local-elevation-um/v1",
        canonical_json_bytes(elevation_um),
    )
    input_relief = relative_relief_um(input_elevation_um)
    output_relief = relative_relief_um(elevation_um)
    if input_relief != output_relief:
        raise ValueError("uniform vertical translation changed relative relief")
    input_relief_sha = domain_hash(
        "g2-provider-local-relative-relief-um/v1",
        canonical_json_bytes(input_relief),
    )
    output_relief_sha = domain_hash(
        "g2-provider-local-relative-relief-um/v1",
        canonical_json_bytes(output_relief),
    )

    hard_obstacle = [[0 for _ in range(shape[1])] for _ in range(shape[0])]
    if mode == "frontier_cut":
        if platform == "wheel":
            cut_node = _frontier_cut_node(platform)
            cut_pose = node_metric_poses_mm_urad[cut_node]
            cut_column = _snapshot_index(int(cut_pose[0]))
            cut_row = _snapshot_index(int(cut_pose[1]))
            hard_obstacle[cut_row][cut_column] = 1
            proxy_operations.append(
                {
                    "cell_xy": [cut_column, cut_row],
                    "cut_node_id": cut_node,
                    "goal_node_id": goal_node,
                    "operation_kind": "interior-frontier-cut-proxy/v1",
                    "start_node_id": start_node,
                }
            )
        elif platform == "hopper":
            launch_radius_mm = round(
                (
                    float(
                        hopper_parameter_record["body_envelope_radius_m"]
                    )
                    + float(
                        hopper_parameter_record["arc_clearance_margin_m"]
                    )
                )
                * 1000.0
            )
            landing_footprint_radius_mm = round(
                float(
                    hopper_parameter_record["landing_footprint_radius_m"]
                )
                * 1000.0
            )
            arc_cells = fine_cells_intersecting_capsule(
                start_x_mm=float(start_pose[0]),
                start_y_mm=float(start_pose[1]),
                end_x_mm=float(goal_pose[0]),
                end_y_mm=float(goal_pose[1]),
                radius_mm=launch_radius_mm,
                resolution_mm=_PROVIDER_FINE_RESOLUTION_MM,
                shape_height_width=shape,
            )
            endpoint_cells = {
                tuple(cell)
                for cell in (
                    fine_cells_intersecting_disk(
                        center_x_mm=float(start_pose[0]),
                        center_y_mm=float(start_pose[1]),
                        radius_mm=launch_radius_mm,
                        resolution_mm=_PROVIDER_FINE_RESOLUTION_MM,
                        shape_height_width=shape,
                    )
                    + fine_cells_intersecting_disk(
                        center_x_mm=float(goal_pose[0]),
                        center_y_mm=float(goal_pose[1]),
                        radius_mm=landing_footprint_radius_mm,
                        resolution_mm=_PROVIDER_FINE_RESOLUTION_MM,
                        shape_height_width=shape,
                    )
                )
            }
            internal_arc_cells = sorted(
                (
                    tuple(cell)
                    for cell in arc_cells
                    if tuple(cell) not in endpoint_cells
                ),
                key=lambda cell: (cell[1], cell[0]),
            )
            if not internal_arc_cells:
                raise ValueError(
                    "Hopper single-hop frontier cut lacks an internal arc cell"
                )
            cut_column, cut_row = internal_arc_cells[0]
            hard_obstacle[cut_row][cut_column] = 1
            proxy_operations.append(
                {
                    "arc_capsule_radius_mm": launch_radius_mm,
                    "cell_xy": [cut_column, cut_row],
                    "endpoint_exclusion_radii_mm": [
                        launch_radius_mm,
                        landing_footprint_radius_mm,
                    ],
                    "goal_node_id": goal_node,
                    "operation_kind": (
                        "hopper-mid-arc-frontier-cut-proxy/v1"
                    ),
                    "start_node_id": start_node,
                }
            )
        elif platform == "legged":
            proxy_operations.append(
                {
                    "goal_cycle_phase": 1,
                    "goal_node_id": goal_node,
                    "incoming_complete_cycle_phase": 0,
                    "operation_kind": (
                        "legged-cycle-phase-frontier-cut/v1"
                    ),
                    "start_node_id": start_node,
                }
            )
        else:
            raise ValueError(f"unknown frontier-cut platform: {platform}")
    known = [[1 for _ in range(shape[1])] for _ in range(shape[0])]
    traversable = [
        [0 if hard_obstacle[row][column] else 1 for column in range(shape[1])]
        for row in range(shape[0])
    ]
    confidence_ppm = [
        [1_000_000 for _ in range(shape[1])] for _ in range(shape[0])
    ]
    slope_cdeg = fine_slope_cdeg(
        elevation_um,
        resolution_mm=_PROVIDER_FINE_RESOLUTION_MM,
    )
    relief_sha = output_relief_sha
    source_kind = terrain_provenance.get(
        "micro_source_kind",
        terrain_provenance.get("source_kind"),
    )
    modification_witness = {
        "operations": proxy_operations,
        "physical_obstacle_cells_written": False,
        "source_kind": "synthetic_terrain_obstacle_proxy/v1",
    }
    modification_witness["semantic_audit_sha256"] = domain_hash(
        "g2-provider-local-proxy-semantic-audit/v1",
        canonical_json_bytes(modification_witness),
        canonical_json_bytes(input_relief),
        canonical_json_bytes(hard_obstacle),
    )
    input_snapshot_identity = {
        "elevation_sha256": input_elevation_sha,
        "frame_id": _LOCAL_FRAME_ID,
        "hard_obstacle_sha256": domain_hash(
            "g2-provider-local-hard-obstacle/v1",
            canonical_json_bytes(hard_obstacle),
        ),
        "origin_mm": [0, 0],
        "platform_kind": platform,
        "proxy_modification_semantic_audit_sha256": (
            modification_witness["semantic_audit_sha256"]
        ),
        "relief_sha256": input_relief_sha,
        "resolution_mm": _PROVIDER_FINE_RESOLUTION_MM,
        "shape_height_width": [shape[0], shape[1]],
        "source_kind": source_kind,
    }
    input_snapshot_sha = domain_hash(
        "g2-provider-local-pretranslation-snapshot/v1",
        canonical_json_bytes(input_snapshot_identity),
    )
    vertical_translation = {
        "anchor_cell_xy": [start_column, start_row],
        "anchor_node_id": start_node,
        "delta_z_um": delta_z_um,
        "input_anchor_elevation_um": input_anchor_elevation_um,
        "input_elevation_sha256": input_elevation_sha,
        "input_relief_sha256": input_relief_sha,
        "input_snapshot_sha256": input_snapshot_sha,
        "output_elevation_sha256": output_elevation_sha,
        "output_relief_sha256": output_relief_sha,
        "semantic_kind": "uniform-vertical-translation/v1",
    }
    vertical_translation["translation_sha256"] = domain_hash(
        "g2-provider-uniform-vertical-translation/v1",
        canonical_json_bytes(vertical_translation),
    )
    core = {
        "arrays": {
            "confidence_ppm": confidence_ppm,
            "elevation_um": elevation_um,
            "hard_obstacle": hard_obstacle,
            "known": known,
            "slope_cdeg": slope_cdeg,
            "traversable": traversable,
        },
        "frame_id": _LOCAL_FRAME_ID,
        "origin_mm": [0, 0],
        "physical_obstacle_cells_written": False,
        "platform_kind": platform,
        "proxy_modification_witness": modification_witness,
        "relief_preservation_sha256": relief_sha,
        "resolution_mm": _PROVIDER_FINE_RESOLUTION_MM,
        "schema_version": "g2-provider-local-terrain-snapshot/v1",
        "shape_height_width": [shape[0], shape[1]],
        "source_kind": source_kind,
        "vertical_translation": vertical_translation,
    }
    snapshot = {
        **core,
        "snapshot_sha256": domain_hash(
            "g2-provider-local-terrain-snapshot/v1",
            canonical_json_bytes(core),
        ),
    }
    validate_provider_local_snapshot(snapshot)
    return snapshot


def _node_states(
    *,
    platform: str,
    node_metric_poses_mm_urad: dict[str, list[int]],
    exact_hopper_node_poses: dict[str, dict[str, str]] | None,
    snapshot: dict[str, Any],
    legged_cycle: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for node_id, display_pose in node_metric_poses_mm_urad.items():
        pose = (
            exact_hopper_node_poses[node_id]
            if platform == "hopper"
            else _pose_binary64_from_display(display_pose)
        )
        if platform == "wheel":
            state: dict[str, Any] = {
                "pose": pose,
                "schema_version": "g2-wheel-node-state/v1",
            }
        elif platform == "hopper":
            state = {
                "pose": pose,
                "schema_version": "g2-hopper-node-state/v1",
            }
        else:
            body_x_m = float.fromhex(pose["x_m_hex"])
            body_y_m = float.fromhex(pose["y_m_hex"])
            contacts: list[dict[str, Any]] = []
            for leg_id, (offset_x_mm, offset_y_mm) in zip(
                legged_cycle["foot_storage_order"],
                legged_cycle["nominal_stance_offsets_mm"],
                strict=True,
            ):
                foot_x_m = body_x_m + int(offset_x_mm) / 1000.0
                foot_y_m = body_y_m + int(offset_y_mm) / 1000.0
                foot_z_m = (
                    _snapshot_elevation_um(
                        snapshot,
                        foot_x_m * 1000.0,
                        foot_y_m * 1000.0,
                    )
                    / 1_000_000.0
                )
                contacts.append(
                    {
                        "leg_id": leg_id,
                        **_binary64_point_identity(
                            foot_x_m,
                            foot_y_m,
                            foot_z_m,
                        ),
                    }
                )
            state = {
                "body_pose": pose,
                "cycle_phase": 0,
                "foot_contacts": contacts,
                "schema_version": "g2-legged-node-state/v1",
            }
        states[node_id] = state
    return states


def _standard_height_mm(root_seed: int, base_index: int) -> list[list[int]]:
    seed = derive_seed(root_seed, "shared", "request-terrain", "standard", base_index)
    x_gradient = int(seed % 9) - 4
    y_gradient = int((seed // 11) % 9) - 4
    return [
        [
            int(seed % 1000) + x * x_gradient + y * y_gradient
            for x in range(_STANDARD_TERRAIN_SIZE)
        ]
        for y in range(_STANDARD_TERRAIN_SIZE)
    ]


def _fixture_macro_height_mm(root_seed: int, base_index: int) -> list[list[int]]:
    seed = derive_seed(root_seed, "shared", "request-terrain", "kilometer", base_index)
    return [
        [
            int(seed % 5000) + x * int(seed % 13) + y * int(seed % 17)
            for x in range(5)
        ]
        for y in range(5)
    ]


def _cell_class_for_pattern(
    base_index: int, *, terrain_size: int
) -> list[list[int]]:
    cell_class = [
        [0 for _ in range(terrain_size)] for _ in range(terrain_size)
    ]
    topology_selector = base_index % 10
    if topology_selector in {6, 7}:
        for x in (1, 2, 3):
            cell_class[2][x] = 2
    elif topology_selector in {8, 9}:
        for y in range(5):
            cell_class[y][2] = 2
    return cell_class


def _terrain_array_shape(terrain_arrays: dict[str, Any]) -> tuple[int, int]:
    height_mm = terrain_arrays["height_mm"]
    height = len(height_mm)
    width = len(height_mm[0]) if height else 0
    if height <= 0 or width <= 0:
        raise ValueError("request terrain arrays must be non-empty")
    for name in ("height_mm", "cell_class", "known", "confidence_ppm"):
        rows = terrain_arrays[name]
        if len(rows) != height or any(len(row) != width for row in rows):
            raise ValueError("request terrain array shape mismatch")
    return height, width


def _macro_cell_size_mm(scale: str) -> int:
    if scale == "standard":
        return 500
    if scale == "kilometer":
        return 20_000
    raise ValueError(f"unknown request scale: {scale}")


def _metric_height_plane(metric_problem: dict[str, Any]) -> dict[str, int]:
    plane = metric_problem["local_height_plane"]
    return {
        "gradient_x_ppm": int(plane["gradient_x_ppm"]),
        "gradient_y_ppm": int(plane["gradient_y_ppm"]),
        "origin_x_mm": int(plane["origin_x_mm"]),
        "origin_y_mm": int(plane["origin_y_mm"]),
        "origin_z_mm": int(plane["origin_z_mm"]),
    }


def _hopper_range_mm(ballistic_contract: dict[str, Any]) -> int:
    action = _action_envelope("hopper")["actions"][0]
    kinematics = _hopper_exact_kinematics(
        action,
        direction=(1, 0),
        ballistic_contract=ballistic_contract,
    )
    return round(
        float.fromhex(kinematics["horizontal_range_m_hex"]) * 1000.0
    )


def _node_spacing_mm(
    platform: str,
    *,
    hopper_parameter_record: dict[str, Any],
    hopper_ballistic_contract: dict[str, Any],
    legged_cycle: dict[str, Any] | None = None,
) -> int:
    if platform == "wheel":
        return 500
    if platform == "legged":
        if legged_cycle is None:
            raise ValueError("legged metric lattice requires crawl cycle")
        return int(legged_cycle["cycle_body_delta_xy_mm"][0])
    if platform == "hopper":
        return _hopper_range_mm(hopper_ballistic_contract)
    raise ValueError(f"unknown request platform: {platform}")


def _node_metric_poses(
    platform: str,
    *,
    hopper_parameter_record: dict[str, Any],
    hopper_ballistic_contract: dict[str, Any],
    legged_cycle: dict[str, Any] | None = None,
) -> dict[str, list[int]]:
    spacing_mm = _node_spacing_mm(
        platform,
        hopper_parameter_record=hopper_parameter_record,
        hopper_ballistic_contract=hopper_ballistic_contract,
        legged_cycle=legged_cycle,
    )
    node_origin_mm = (
        _LEGGED_NODE_ORIGIN_MM
        if platform == "legged"
        else _LOCAL_NODE_ORIGIN_MM
    )
    width, height = _graph_shape(platform)
    return {
        f"n:{x}:{y}": [
            node_origin_mm + x * spacing_mm,
            node_origin_mm + y * spacing_mm,
            0,
        ]
        for x in range(width)
        for y in range(height)
    }


def _endpoint_radius_mm(
    platform: str, *, hopper_parameter_record: dict[str, Any]
) -> int:
    if platform in {"wheel", "legged"}:
        return 100
    return round(
        float(hopper_parameter_record["landing_footprint_radius_m"]) * 1000
    )


def _point_map_margin_mm(
    pose: list[int], bounds_mm: list[int], radius_mm: int
) -> int:
    minimum_x, minimum_y, maximum_x, maximum_y = bounds_mm
    return min(
        int(pose[0]) - minimum_x,
        maximum_x - int(pose[0]),
        int(pose[1]) - minimum_y,
        maximum_y - int(pose[1]),
    ) - radius_mm


def _local_proxy_difficulty_mode(base_index: int) -> str:
    selector = int(base_index) % 10
    if selector in {6, 7}:
        return "near_threshold"
    if selector in {8, 9}:
        return "frontier_cut"
    return "nominal"


def _local_macro_frame_candidates(
    terrain_arrays: dict[str, Any],
    *,
    scale: str,
    macro_cell_size_mm: int,
    hopper_ballistic_contract: dict[str, Any],
    hopper_parameter_record: dict[str, Any],
) -> list[dict[str, Any]]:
    height_mm = terrain_arrays["height_mm"]
    if scale == "standard":
        candidates = [(0, 0, "source_x_then_y", (1, 0), (0, 1))]
        policy = "fixed-procedural-origin/v1"
    else:
        candidates = [
            (x, y, orientation, local_x, local_y)
            for y in range(len(height_mm) - 1)
            for x in range(len(height_mm[0]) - 1)
            for orientation, local_x, local_y in (
                ("source_x_then_y", (1, 0), (0, 1)),
                ("source_y_then_x", (0, 1), (1, 0)),
            )
        ]
        policy = (
            "min-abs-hopper-direct-residual-then-total-gradient-row-major/v1"
        )
    hopper_kinematics = _hopper_exact_kinematics(
        _action_envelope("hopper")["actions"][0],
        direction=(1, 0),
        ballistic_contract=hopper_ballistic_contract,
    )
    delta_x_m = float.fromhex(hopper_kinematics["delta_x_m_hex"])
    delta_y_m = float.fromhex(hopper_kinematics["delta_y_m_hex"])
    launch_radius_mm = round(
        (
            float(hopper_parameter_record["body_envelope_radius_m"])
            + float(hopper_parameter_record["arc_clearance_margin_m"])
        )
        * 1000.0
    )
    landing_radius_mm = round(
        float(hopper_parameter_record["landing_footprint_radius_m"])
        * 1000.0
    )
    ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for x, y, orientation, local_x, local_y in candidates:
        origin_height = int(height_mm[y][x])
        x_height = int(height_mm[y + local_x[1]][x + local_x[0]])
        y_height = int(height_mm[y + local_y[1]][x + local_y[0]])
        gradient_x_ppm = round(
            (x_height - origin_height) * 1_000_000 / macro_cell_size_mm
        )
        gradient_y_ppm = round(
            (y_height - origin_height) * 1_000_000 / macro_cell_size_mm
        )
        projected_residual_um = _hopper_support_envelope_max_residual_um(
            gradient_x_ppm=gradient_x_ppm,
            gradient_y_ppm=gradient_y_ppm,
            delta_x_m=delta_x_m,
            delta_y_m=delta_y_m,
            launch_radius_mm=launch_radius_mm,
            landing_radius_mm=landing_radius_mm,
        )
        frame = {
            "local_x_axis_source_delta_cell_xy": [
                local_x[0],
                local_x[1],
            ],
            "local_y_axis_source_delta_cell_xy": [
                local_y[0],
                local_y[1],
            ],
            "orientation": orientation,
            "origin_height_mm": origin_height,
            "hopper_full_support_envelope_max_residual_um": (
                projected_residual_um
            ),
            "hopper_landing_support_radius_mm": landing_radius_mm,
            "hopper_launch_support_radius_mm": launch_radius_mm,
            "selected_macro_cell_xy": [x, y],
            "selection_policy": policy,
            "selection_policy_sha256": domain_hash(
                "g2-local-macro-frame-selection-policy/v1",
                policy.encode("ascii"),
            ),
            "source_gradient_x_ppm_in_local_frame": gradient_x_ppm,
            "source_gradient_y_ppm_in_local_frame": gradient_y_ppm,
        }
        ranked.append(
            (
                (
                    projected_residual_um,
                    abs(gradient_x_ppm) + abs(gradient_y_ppm),
                    y,
                    x,
                    orientation,
                ),
                frame,
            )
        )
    return [frame for _, frame in sorted(ranked, key=lambda item: item[0])]


def _select_local_macro_frame(
    terrain_arrays: dict[str, Any],
    *,
    scale: str,
    macro_cell_size_mm: int,
    hopper_ballistic_contract: dict[str, Any],
    hopper_parameter_record: dict[str, Any],
) -> dict[str, Any]:
    return _local_macro_frame_candidates(
        terrain_arrays,
        scale=scale,
        macro_cell_size_mm=macro_cell_size_mm,
        hopper_ballistic_contract=hopper_ballistic_contract,
        hopper_parameter_record=hopper_parameter_record,
    )[0]


def _hopper_landing_required_radius_mm(
    hopper_parameter_record: dict[str, Any],
) -> int:
    landing_radius_mm = round(
        float(hopper_parameter_record["landing_footprint_radius_m"]) * 1000.0
    )
    landing_sigma_mm = 289
    landing_prefix_probability_ppm = 990_000
    landing_prefix_radius_mm = math.ceil(
        landing_sigma_mm
        * math.sqrt(
            -2.0
            * math.log(1.0 - landing_prefix_probability_ppm / 1_000_000.0)
        )
    )
    return landing_radius_mm + landing_prefix_radius_mm


def _endpoint_safety_witness(
    *,
    platform: str,
    endpoint_name: str,
    node_id: str,
    display_pose: list[int],
    state: dict[str, Any],
    snapshot: dict[str, Any],
    hopper_parameter_record: dict[str, Any],
    legged_cycle: dict[str, Any],
) -> dict[str, Any]:
    shape = tuple(int(value) for value in snapshot["shape_height_width"])
    resolution_mm = int(snapshot["resolution_mm"])
    component_geometry: list[tuple[str, float, float, int]] = []
    if platform == "wheel":
        component_geometry.append(
            ("body", float(display_pose[0]), float(display_pose[1]), 100)
        )
    elif platform == "legged":
        component_geometry.append(
            ("body", float(display_pose[0]), float(display_pose[1]), 100)
        )
        for contact in state["foot_contacts"]:
            component_geometry.append(
                (
                    str(contact["leg_id"]),
                    float.fromhex(contact["x_m_hex"]) * 1000.0,
                    float.fromhex(contact["y_m_hex"]) * 1000.0,
                    0,
                )
            )
    elif platform == "hopper":
        radius_mm = (
            round(
                (
                    float(hopper_parameter_record["body_envelope_radius_m"])
                    + float(hopper_parameter_record["arc_clearance_margin_m"])
                )
                * 1000.0
            )
            if endpoint_name == "start"
            else round(
                float(
                    hopper_parameter_record["landing_footprint_radius_m"]
                )
                * 1000.0
            )
        )
        component_geometry.append(
            (
                "launch-envelope"
                if endpoint_name == "start"
                else "landing-envelope",
                float(display_pose[0]),
                float(display_pose[1]),
                radius_mm,
            )
        )
    else:
        raise ValueError(f"unknown request platform: {platform}")

    components: list[dict[str, Any]] = []
    all_cells: set[tuple[int, int]] = set()
    all_records: list[dict[str, Any]] = []
    for component_id, center_x_mm, center_y_mm, radius_mm in component_geometry:
        cell_xy = fine_cells_intersecting_disk(
            center_x_mm=center_x_mm,
            center_y_mm=center_y_mm,
            radius_mm=radius_mm,
            resolution_mm=resolution_mm,
            shape_height_width=shape,
        )
        records = [_snapshot_cell_record(snapshot, cell) for cell in cell_xy]
        safe = _snapshot_cells_safe(records)
        components.append(
            {
                "cell_records": records,
                "cell_xy": cell_xy,
                "component_id": component_id,
                "geometry": {
                    "boundary_semantics": "closed-cell-intersection/v1",
                    "center_mm": [center_x_mm, center_y_mm],
                    "kind": "disk/v1",
                    "radius_mm": radius_mm,
                },
                "safe": safe,
            }
        )
        all_cells.update((int(cell[0]), int(cell[1])) for cell in cell_xy)
        all_records.extend(records)

    if platform == "legged":
        feet_xy = [
            [
                round(float.fromhex(contact["x_m_hex"]) * 1000.0),
                round(float.fromhex(contact["y_m_hex"]) * 1000.0),
            ]
            for contact in state["foot_contacts"]
        ]
        polygon = _convex_hull_xy(feet_xy)
        margin_mm = math.floor(
            _signed_convex_margin_mm(
                (float(display_pose[0]), float(display_pose[1])),
                polygon,
            )
        )
        support = {
            "cycle_phase": int(state["cycle_phase"]),
            "kind": "four-contact-convex-support/v1",
            "minimum_required_margin_mm": int(
                legged_cycle["minimum_support_margin_mm"]
            ),
            "support_margin_mm": margin_mm,
            "support_polygon_mm": polygon,
            "safe": (
                0 <= int(state["cycle_phase"]) < len(
                    legged_cycle["phase_steps"]
                )
                and margin_mm
                >= int(legged_cycle["minimum_support_margin_mm"])
            ),
        }
    elif platform == "hopper":
        center_elevation_um = _snapshot_elevation_um(
            snapshot,
            float(display_pose[0]),
            float(display_pose[1]),
        )
        maximum_abs_residual_um = max(
            (
                abs(int(record["elevation_um"]) - center_elevation_um)
                for record in all_records
            ),
            default=2**31,
        )
        support = {
            "kind": (
                "launch-support-envelope/v1"
                if endpoint_name == "start"
                else "landing-support-envelope/v1"
            ),
            "maximum_abs_residual_um": maximum_abs_residual_um,
            "maximum_tolerated_residual_um": 50_000,
            "safe": maximum_abs_residual_um <= 50_000,
        }
    else:
        support = {
            "kind": "single-body-footprint/v1",
            "safe": True,
        }

    failure_reasons: list[str] = []
    if any(not component["safe"] for component in components):
        failure_reasons.append("G2I_ENDPOINT_COMPONENT_UNSAFE")
    if not support["safe"]:
        failure_reasons.append("G2I_ENDPOINT_SUPPORT_UNSAFE")
    core = {
        "cell_xy": [list(cell) for cell in sorted(all_cells)],
        "components": components,
        "failure_reasons": failure_reasons,
        "maximum_slope_cdeg": 3000,
        "node_id": node_id,
        "platform_kind": platform,
        "safe": not failure_reasons,
        "schema_version": "g2-endpoint-safety/v2",
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "support": support,
    }
    return {
        **core,
        "endpoint_safety_sha256": domain_hash(
            "g2-endpoint-safety/v2",
            canonical_json_bytes(core),
        ),
    }


def _metric_problem(
    *,
    platform: str,
    scale: str,
    terrain_arrays: dict[str, Any],
    terrain_provenance: dict[str, Any],
    hopper_parameter_record: dict[str, Any],
    hopper_ballistic_contract: dict[str, Any],
    legged_cycle: dict[str, Any],
    base_index: int,
    local_macro_frame: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    height, width = _terrain_array_shape(terrain_arrays)
    macro_cell_size_mm = _macro_cell_size_mm(scale)
    if local_macro_frame is None:
        local_macro_frame = _select_local_macro_frame(
            terrain_arrays,
            scale=scale,
            macro_cell_size_mm=macro_cell_size_mm,
            hopper_ballistic_contract=hopper_ballistic_contract,
            hopper_parameter_record=hopper_parameter_record,
        )
    origin_height_mm = int(local_macro_frame["origin_height_mm"])
    local_height_plane = {
        "gradient_x_ppm": int(
            local_macro_frame["source_gradient_x_ppm_in_local_frame"]
        ),
        "gradient_y_ppm": int(
            local_macro_frame["source_gradient_y_ppm_in_local_frame"]
        ),
        "origin_x_mm": 0,
        "origin_y_mm": 0,
        "origin_z_mm": origin_height_mm,
    }
    source_extent_bounds_mm = [
        0,
        0,
        width * macro_cell_size_mm,
        height * macro_cell_size_mm,
    ]
    local_bounds_mm = [0, 0, 10_000, 10_000]
    poses = _node_metric_poses(
        platform,
        hopper_parameter_record=hopper_parameter_record,
        hopper_ballistic_contract=hopper_ballistic_contract,
        legged_cycle=legged_cycle,
    )
    exact_hopper_node_poses = (
        _hopper_exact_node_poses(hopper_ballistic_contract)
        if platform == "hopper"
        else None
    )
    provider_local_snapshot = _provider_local_snapshot(
        platform=platform,
        scale=scale,
        source_terrain_arrays=terrain_arrays,
        terrain_provenance=terrain_provenance,
        local_height_plane=local_height_plane,
        node_metric_poses_mm_urad=poses,
        hopper_parameter_record=hopper_parameter_record,
        base_index=base_index,
    )
    snapshot_sha = str(provider_local_snapshot["snapshot_sha256"])
    node_states = _node_states(
        platform=platform,
        node_metric_poses_mm_urad=poses,
        exact_hopper_node_poses=exact_hopper_node_poses,
        snapshot=provider_local_snapshot,
        legged_cycle=legged_cycle,
    )
    if (
        platform == "legged"
        and _local_proxy_difficulty_mode(base_index) == "frontier_cut"
    ):
        _, goal_node = _request_start_goal(platform)
        node_states[goal_node] = {
            **node_states[goal_node],
            "cycle_phase": 1,
        }
    terrain_arrays_sha256 = domain_hash(
        "g2-request-terrain-arrays/v1",
        canonical_json_bytes(terrain_arrays),
    )
    height_geometry_sha256 = domain_hash(
        "g2-request-height-geometry/v1",
        canonical_json_bytes(terrain_arrays["height_mm"]),
    )
    proxy_source_kind = terrain_provenance.get(
        "micro_source_kind",
        terrain_provenance.get("source_kind"),
    )
    local_proxy_geometry_sha256 = domain_hash(
        "g2-request-local-proxy-geometry/v1",
        platform.encode("ascii"),
        str(proxy_source_kind).encode("utf-8"),
        snapshot_sha.encode("ascii"),
        canonical_json_bytes(poses),
        canonical_json_bytes(exact_hopper_node_poses),
        canonical_json_bytes(local_height_plane),
        canonical_json_bytes(local_macro_frame),
        canonical_json_bytes(local_bounds_mm),
        _local_proxy_difficulty_mode(base_index).encode("ascii"),
    )
    endpoint_radius_mm = _endpoint_radius_mm(
        platform,
        hopper_parameter_record=hopper_parameter_record,
    )
    terrain_materialization = {
        "input_terrain_arrays_sha256": terrain_arrays_sha256,
        "output_snapshot_sha256": snapshot_sha,
        "proxy_modification_semantic_audit_sha256": (
            provider_local_snapshot["proxy_modification_witness"][
                "semantic_audit_sha256"
            ]
        ),
        "semantic_kind": (
            "procedural-fine-elevation-with-declared-synthetic-proxy-overlay/v1"
            if scale == "standard"
            else (
                "lola-macro-linear-fine-elevation-with-declared-"
                "synthetic-proxy-overlay/v1"
            )
        ),
        "transform_id": "g2-provider-local-snapshot-materialization/v1",
    }
    terrain_materialization["transform_sha256"] = domain_hash(
        "g2-provider-local-snapshot-materialization/v1",
        canonical_json_bytes(terrain_materialization),
    )
    provider_terrain_transform = {
        "delta_z_um": provider_local_snapshot["vertical_translation"][
            "delta_z_um"
        ],
        "input_elevation_sha256": provider_local_snapshot[
            "vertical_translation"
        ]["input_elevation_sha256"],
        "input_relief_sha256": provider_local_snapshot[
            "vertical_translation"
        ]["input_relief_sha256"],
        "input_snapshot_sha256": provider_local_snapshot[
            "vertical_translation"
        ]["input_snapshot_sha256"],
        "output_elevation_sha256": provider_local_snapshot[
            "vertical_translation"
        ]["output_elevation_sha256"],
        "output_relief_sha256": provider_local_snapshot[
            "vertical_translation"
        ]["output_relief_sha256"],
        "output_snapshot_sha256": snapshot_sha,
        "semantic_kind": "uniform-vertical-translation/v1",
        "transform_id": "g2-provider-terrain-transform/v1",
        "translation_sha256": provider_local_snapshot[
            "vertical_translation"
        ]["translation_sha256"],
    }
    provider_terrain_transform["transform_sha256"] = domain_hash(
        "g2-provider-terrain-transform/v1",
        canonical_json_bytes(provider_terrain_transform),
    )
    terrain_binding = {
        "materialization": terrain_materialization,
        "provider_transform": provider_terrain_transform,
        "relief_preservation_sha256": provider_local_snapshot[
            "relief_preservation_sha256"
        ],
        "snapshot_sha256": snapshot_sha,
    }
    terrain_binding["terrain_binding_sha256"] = domain_hash(
        "g2-provider-terrain-binding/v1",
        canonical_json_bytes(terrain_binding),
    )
    start_node, goal_node = _request_start_goal(platform)

    def endpoint_binding(
        node_id: str, *, endpoint_name: str
    ) -> dict[str, Any]:
        display = poses[node_id]
        state = node_states[node_id]
        pose = (
            state["body_pose"]
            if platform == "legged"
            else state["pose"]
        )
        endpoint_safety = _endpoint_safety_witness(
            platform=platform,
            endpoint_name=endpoint_name,
            node_id=node_id,
            display_pose=display,
            state=state,
            snapshot=provider_local_snapshot,
            hopper_parameter_record=hopper_parameter_record,
            legged_cycle=legged_cycle,
        )
        return {
            "endpoint_safety": endpoint_safety,
            "node_id": node_id,
            "pose_binary64_m_rad": pose,
            "pose_mm_urad": display,
        }

    vertical_reference = {
        "datum_id": "provider-launch-anchor-zero-datum/v1",
        "height_um": 0,
        "macro_cell_xy": local_macro_frame["selected_macro_cell_xy"],
        "source": (
            "lola-20m-macro-height/v1"
            if scale == "kilometer"
            else "procedural-height/v1"
        ),
        "source_datum_id": (
            "lola-20m-source-datum/v1"
            if scale == "kilometer"
            else "procedural-local-source-datum/v1"
        ),
        "terrain_transform_sha256": provider_terrain_transform[
            "transform_sha256"
        ],
        "vertical_unit": "um",
    }
    vertical_reference["reference_sha256"] = domain_hash(
        "g2-vertical-datum-reference/v1",
        canonical_json_bytes(vertical_reference),
    )
    metric_problem: dict[str, Any] = {
        "canonical_pose_unit": {
            "encoding": "ieee754-binary64-hex-and-word/v1",
            "heading": "rad",
            "horizontal": "m",
        },
        "endpoint_safety_margin_mm": {
            "goal": _point_map_margin_mm(
                poses[goal_node], local_bounds_mm, endpoint_radius_mm
            ),
            "start": _point_map_margin_mm(
                poses[start_node], local_bounds_mm, endpoint_radius_mm
            ),
        },
        "frame_id": _LOCAL_FRAME_ID,
        "goal": endpoint_binding(goal_node, endpoint_name="goal"),
        "heading_unit": "urad",
        "height_geometry_sha256": height_geometry_sha256,
        "horizontal_unit": "mm",
        "local_proxy_geometry_sha256": local_proxy_geometry_sha256,
        "local_proxy_difficulty_mode": _local_proxy_difficulty_mode(
            base_index
        ),
        "local_proxy_source_kind": proxy_source_kind,
        "local_height_plane": local_height_plane,
        "local_height_plane_sha256": domain_hash(
            "g2-request-local-height-plane/v1",
            canonical_json_bytes(local_height_plane),
        ),
        "macro_cell_size_mm": macro_cell_size_mm,
        "node_metric_poses_mm_urad": poses,
        "node_origin_contract": {
            "origin_mm": (
                [_LEGGED_NODE_ORIGIN_MM, _LEGGED_NODE_ORIGIN_MM]
                if platform == "legged"
                else [_LOCAL_NODE_ORIGIN_MM, _LOCAL_NODE_ORIGIN_MM]
            ),
            "policy_id": "platform-specific-selected-fine-cell-separation/v1",
        },
        "node_state_root_sha256": domain_hash(
            "g2-request-node-state-root/v1",
            canonical_json_bytes(node_states),
        ),
        "node_states": node_states,
        "origin_mm": [0, 0],
        "provider_local_snapshot_sha256": snapshot_sha,
        "provider_fine_resolution_mm": _PROVIDER_FINE_RESOLUTION_MM,
        "provider_local_planning_region": {
            "bounds_mm": local_bounds_mm,
            "fine_cell_size_mm": _PROVIDER_FINE_RESOLUTION_MM,
            "shape_height_width": [20, 20],
        },
        "scale": scale,
        "schema_version": "g2-metric-planning-problem/v3",
        "start": endpoint_binding(start_node, endpoint_name="start"),
        "terrain_arrays_sha256": terrain_arrays_sha256,
        "terrain_binding": terrain_binding,
        "source_extent_bounds_mm": source_extent_bounds_mm,
        "source_to_local_transform": local_macro_frame,
        "terrain_bounds_mm": local_bounds_mm,
        "terrain_shape_height_width": [height, width],
        "terrain_transform_semantics": {
            "macro_height": (
                "adjacent-macro-finite-difference-local-plane/v1"
            ),
            "sub_macro_geometry": (
                "materialized-fine-snapshot-with-declared-proxy-overlay/v1"
            ),
        },
        "terrain_transform": provider_terrain_transform,
        "vertical_datum": vertical_reference,
    }
    if platform == "hopper":
        metric_problem["hopper_ballistic_binding"] = {
            "action": {
                key: value
                for key, value in _action_envelope("hopper")["actions"][0].items()
                if key != "delta_cell_xy"
            },
            "parameter_record_sha256": sha256_bytes(
                canonical_json_bytes(hopper_parameter_record)
            ),
            "ballistic_contract": hopper_ballistic_contract,
            "range_mm_display": _hopper_range_mm(
                hopper_ballistic_contract
            ),
            "semantics": (
                "same-support-provider-binary64-discrete-ballistic-endpoint/v2"
            ),
        }
        metric_problem[
            "hopper_node_metric_poses_binary64_m_rad"
        ] = exact_hopper_node_poses
        start_pose = poses[start_node]
        h_ref_um = _snapshot_elevation_um(
            provider_local_snapshot,
            int(start_pose[0]),
            int(start_pose[1]),
        )
        h_ref = _binary64_identity(h_ref_um / 1_000_000.0)
        metric_problem["relief_preservation_sha256"] = (
            provider_local_snapshot["relief_preservation_sha256"]
        )
        support_plane = {
            "H_ref_m_hex": h_ref["hex"],
            "H_ref_m_word_hex": h_ref["word_hex"],
            "H_ref_um": h_ref_um,
            "anchor_node_id": start_node,
            "horizontal": True,
            "normal": [0, 0, 1],
            "schema_version": "g2-horizontal-support-plane/v1",
            "snapshot_sha256": snapshot_sha,
        }
        support_plane["support_plane_sha256"] = domain_hash(
            "g2-horizontal-support-plane/v1",
            canonical_json_bytes(support_plane),
        )
        metric_problem["support_plane"] = support_plane
    elif platform == "legged":
        metric_problem["legged_crawl_cycle"] = legged_cycle
        metric_problem["legged_crawl_cycle_sha256"] = domain_hash(
            "g2-independent-static-crawl-cycle/v2",
            canonical_json_bytes(legged_cycle),
        )
    return metric_problem, provider_local_snapshot


def _action_envelope(
    platform: str, *, legged_cycle: dict[str, Any] | None = None
) -> dict[str, Any]:
    if platform == "wheel":
        actions = [
            {
                "angular_urad_s": 0,
                "delta_cell_xy": [dx, dy],
                "duration_us": 1_000_000,
                "heading_urad": round(math.atan2(dy, dx) * 1_000_000),
                "speed_um_s": 500_000,
            }
            for dx, dy in _REQUEST_DIRECTIONS
        ]
    elif platform == "legged":
        if legged_cycle is None:
            actions = [
                {
                    "delta_cell_xy": [dx, dy],
                    "step_delta_mm": [dx * 400, dy * 400, 0],
                }
                for dx, dy in _REQUEST_DIRECTIONS
            ]
        else:
            actions = []
            for dx, dy in _REQUEST_DIRECTIONS:
                phase_steps = []
                for step in legged_cycle["phase_steps"]:
                    forward_mm, lateral_mm = step[
                        "forward_lateral_delta_mm"
                    ]
                    phase_steps.append(
                        {
                            "moving_foot_delta_mm": [
                                dx * int(forward_mm) - dy * int(lateral_mm),
                                dy * int(forward_mm) + dx * int(lateral_mm),
                                0,
                            ],
                            "moving_leg": step["moving_leg"],
                            "moving_leg_id": step["moving_leg_id"],
                            "phase": step["phase"],
                        }
                    )
                actions.append(
                    {
                        "cycle_body_delta_xy_mm": [
                            dx
                            * int(
                                legged_cycle["cycle_body_delta_xy_mm"][0]
                            ),
                            dy
                            * int(
                                legged_cycle["cycle_body_delta_xy_mm"][0]
                            ),
                        ],
                        "cycle_steps": phase_steps,
                        "delta_cell_xy": [dx, dy],
                    }
                )
    else:
        azimuth_indices = (0,)
        actions = [
            {
                "azimuth_index": azimuth_indices[index],
                "azimuth_mdeg": azimuth_indices[index] * 22500,
                "delta_cell_xy": [dx, dy],
                "elevation_index": 1,
                "elevation_mdeg": 45000,
                "speed_index": 0,
                "speed_mm_s": 1500,
            }
            for index, (dx, dy) in enumerate(_REQUEST_DIRECTIONS)
        ]
    actions = [
        {
            **action,
            "action_id": (
                f"g2i-action-{platform}-{index}-"
                f"{domain_hash('g2-request-action/v1', canonical_json_bytes(action))[:16]}"
            ),
        }
        for index, action in enumerate(actions)
    ]
    return {
        "actions": actions,
        "complete": True,
        "direction_set": "positive_x_only/v1",
        "platform_kind": platform,
        "schema_version": "g2-request-action-envelope/v2",
    }


def _lola_height_for_index(
    lola_provenance: dict[str, Any],
    *,
    root_seed: int,
    base_index: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    records = list(lola_provenance.get("roi_records", []))
    if records:
        record = records[base_index % len(records)]
        return (
            [[int(value) for value in row] for row in record["height_mm"]],
            {
                "roi_geometry_sha256": record["roi_geometry_sha256"],
                "roi_index": int(record["roi_index"]),
                "roi_root_sha256": lola_provenance["roi_root_sha256"],
                "window_col_row_width_height": record[
                    "window_col_row_width_height"
                ],
            },
        )
    if lola_provenance.get("fixture_only") is not True:
        raise ValueError("production kilometer source requires decoded LOLA ROI records")
    height = _fixture_macro_height_mm(root_seed, base_index)
    return (
        height,
        {
            "fixture_macro_geometry_sha256": domain_hash(
                "g2-fixture-macro-height/v1", canonical_json_bytes(height)
            ),
            "roi_index": base_index % 96,
        },
    )


def _frame_admission_witness(
    metric_problem: dict[str, Any],
    *,
    frame_rank: int,
) -> dict[str, Any]:
    start = metric_problem["start"]["endpoint_safety"]
    goal = metric_problem["goal"]["endpoint_safety"]
    return {
        "frame_rank": frame_rank,
        "goal_endpoint_safety_sha256": goal["endpoint_safety_sha256"],
        "goal_failure_reasons": list(goal["failure_reasons"]),
        "goal_safe": bool(goal["safe"]),
        "source_to_local_transform_sha256": domain_hash(
            "g2-local-macro-frame/v1",
            canonical_json_bytes(metric_problem["source_to_local_transform"]),
        ),
        "start_endpoint_safety_sha256": start["endpoint_safety_sha256"],
        "start_failure_reasons": list(start["failure_reasons"]),
        "start_safe": bool(start["safe"]),
    }


def _raw_source_reject(
    *,
    action_envelope: dict[str, Any],
    base_index: int,
    determinism_seed: int,
    frame_admission_witnesses: list[dict[str, Any]],
    platform: str,
    scale: str,
    terrain_arrays: dict[str, Any],
    terrain_provenance: dict[str, Any],
) -> dict[str, Any]:
    action_envelope_sha = domain_hash(
        "g2-request-action-envelope/v2",
        canonical_json_bytes(action_envelope),
    )
    terrain_arrays_sha = domain_hash(
        "g2-request-terrain-arrays/v1",
        canonical_json_bytes(terrain_arrays),
    )
    terrain_provenance_sha = domain_hash(
        "g2-raw-request-terrain-provenance/v1",
        canonical_json_bytes(terrain_provenance),
    )
    candidate_identity = {
        "action_envelope_sha256": action_envelope_sha,
        "base_index": base_index,
        "determinism_seed": determinism_seed,
        "platform_kind": platform,
        "scale": scale,
        "terrain_arrays_sha256": terrain_arrays_sha,
        "terrain_provenance_sha256": terrain_provenance_sha,
    }
    candidate_sha = domain_hash(
        "g2-raw-request-source-candidate/v1",
        canonical_json_bytes(candidate_identity),
    )
    policy_sha = domain_hash(
        "g2-raw-request-endpoint-admission-policy/v1",
        _ENDPOINT_ADMISSION_POLICY.encode("ascii"),
    )
    reject_core = {
        **candidate_identity,
        "artifact_kind": "raw_request_source_candidate",
        "artifact_sha256": candidate_sha,
        "endpoint_admission_policy": _ENDPOINT_ADMISSION_POLICY,
        "endpoint_admission_policy_sha256": policy_sha,
        "evaluated_frame_count": len(frame_admission_witnesses),
        "frame_admission_root_sha256": domain_hash(
            "g2-raw-request-frame-admission-root/v1",
            canonical_json_bytes(frame_admission_witnesses),
        ),
        "frame_admission_witnesses": frame_admission_witnesses,
        "reason_code": "G2I_RAW_ENDPOINT_UNSAFE_ALL_FRAMES",
        "schema_version": "g2-raw-request-source-reject/v1",
    }
    reject_sha = domain_hash(
        "g2-raw-request-source-reject/v1",
        canonical_json_bytes(reject_core),
    )
    reject = {
        **reject_core,
        "raw_source_reject_id": (
            f"g2i-reject-{platform}-{scale}-{reject_sha[:20]}"
        ),
        "raw_source_reject_sha256": reject_sha,
    }
    validate_raw_source_reject(reject)
    return reject


def generate_raw_request_source_admission(
    specification: dict[str, Any],
    *,
    lola_provenance: dict[str, Any],
    hopper_parameter_record: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_producer_specification(specification)
    if lola_provenance.get("physical_obstacle_cells_written") is not False:
        raise ValueError("kilometer synthetic obstacles cannot be physical")
    if (
        lola_provenance.get("micro_source_kind")
        != "synthetic_terrain_obstacle_proxy/v1"
    ):
        raise ValueError("kilometer microterrain source kind mismatch")
    record = _require_hopper_parameter_record(hopper_parameter_record)
    legged_cycle = _legged_cycle_contract(specification)
    hopper_ballistics = _hopper_ballistic_contract(specification)
    root_seed = int(specification["root_seed"])
    rows: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    endpoint_admission_policy_sha = domain_hash(
        "g2-raw-request-endpoint-admission-policy/v1",
        _ENDPOINT_ADMISSION_POLICY.encode("ascii"),
    )
    for scale, count in (
        ("standard", int(specification["request_pool"]["standard_base_terrains"])),
        ("kilometer", int(specification["request_pool"]["kilometer_base_terrains"])),
    ):
        for base_index in range(count):
            if scale == "standard":
                height_mm = _standard_height_mm(root_seed, base_index)
                terrain_size = _STANDARD_TERRAIN_SIZE
                provenance = {
                    "physical_obstacle_cells_written": False,
                    "source_kind": "procedural_simulation_proxy/v1",
                    "source_seed": derive_seed(
                        root_seed,
                        "shared",
                        "requests",
                        scale,
                        base_index,
                    ),
                }
            else:
                height_mm, roi_binding = _lola_height_for_index(
                    lola_provenance,
                    root_seed=root_seed,
                    base_index=base_index,
                )
                terrain_size = _KILOMETER_TERRAIN_SIZE
                if (
                    len(height_mm) != terrain_size
                    or any(len(row) != terrain_size for row in height_mm)
                ):
                    raise ValueError("kilometer LOLA ROI must be exactly 5x5")
                provenance = {
                    "interpolation_schema_version": lola_provenance.get(
                        "interpolation_schema_version",
                        "integer-bilinear-macro-only/v1",
                    ),
                    "jp2_sha256": lola_provenance["jp2_sha256"],
                    "lbl_sha256": lola_provenance["lbl_sha256"],
                    "macro_source_kind": lola_provenance["macro_source_kind"],
                    "micro_source_kind": lola_provenance["micro_source_kind"],
                    "physical_obstacle_cells_written": False,
                    "proxy_seed": derive_seed(
                        root_seed,
                        "shared",
                        "requests",
                        scale,
                        base_index,
                    ),
                    **roi_binding,
                }
                if lola_provenance.get("fixture_only") is True:
                    provenance["fixture_only"] = True
            terrain_arrays = {
                "cell_class": _cell_class_for_pattern(
                    base_index,
                    terrain_size=terrain_size,
                ),
                "confidence_ppm": [
                    [1_000_000 for _ in range(terrain_size)]
                    for _ in range(terrain_size)
                ],
                "height_mm": height_mm,
                "known": [
                    [1 for _ in range(terrain_size)]
                    for _ in range(terrain_size)
                ],
            }
            for platform in _PLATFORMS:
                start_node, goal_node = _request_start_goal(platform)
                start_x, start_y = (
                    int(value) for value in start_node.split(":")[1:]
                )
                goal_x, goal_y = (
                    int(value) for value in goal_node.split(":")[1:]
                )
                envelope = _action_envelope(
                    platform,
                    legged_cycle=legged_cycle if platform == "legged" else None,
                )
                determinism_seed = derive_seed(
                    root_seed, platform, "requests", scale, base_index
                )
                frame_admission_witnesses: list[dict[str, Any]] = []
                selected: tuple[dict[str, Any], dict[str, Any]] | None = None
                frames = _local_macro_frame_candidates(
                    terrain_arrays,
                    scale=scale,
                    macro_cell_size_mm=_macro_cell_size_mm(scale),
                    hopper_ballistic_contract=hopper_ballistics,
                    hopper_parameter_record=record,
                )
                for frame_rank, candidate_frame in enumerate(frames):
                    admitted_frame = {
                        **candidate_frame,
                        "endpoint_admission_policy": (
                            _ENDPOINT_ADMISSION_POLICY
                        ),
                        "endpoint_admission_policy_sha256": (
                            endpoint_admission_policy_sha
                        ),
                        "endpoint_admission_rank": frame_rank,
                    }
                    metric_problem, provider_local_snapshot = _metric_problem(
                        platform=platform,
                        scale=scale,
                        terrain_arrays=terrain_arrays,
                        terrain_provenance=provenance,
                        hopper_parameter_record=record,
                        hopper_ballistic_contract=hopper_ballistics,
                        legged_cycle=legged_cycle,
                        base_index=base_index,
                        local_macro_frame=admitted_frame,
                    )
                    witness = _frame_admission_witness(
                        metric_problem,
                        frame_rank=frame_rank,
                    )
                    frame_admission_witnesses.append(witness)
                    if (
                        selected is None
                        and witness["start_safe"] is True
                        and witness["goal_safe"] is True
                    ):
                        selected = (
                            metric_problem,
                            provider_local_snapshot,
                        )
                if selected is None:
                    rejects.append(
                        _raw_source_reject(
                            action_envelope=envelope,
                            base_index=base_index,
                            determinism_seed=determinism_seed,
                            frame_admission_witnesses=(
                                frame_admission_witnesses
                            ),
                            platform=platform,
                            scale=scale,
                            terrain_arrays=terrain_arrays,
                            terrain_provenance=provenance,
                        )
                    )
                    continue
                metric_problem, provider_local_snapshot = selected
                provider_provenance = {
                    **provenance,
                    "provider_local_proxy_semantic_audit_sha256": (
                        provider_local_snapshot["proxy_modification_witness"][
                            "semantic_audit_sha256"
                        ]
                    ),
                }
                source_core = {
                    "action_envelope": envelope,
                    "base_index": base_index,
                    "determinism_seed": determinism_seed,
                    "goal_cell_xy": [goal_x, goal_y],
                    "platform_kind": platform,
                    "scale": scale,
                    "start_cell_xy": [start_x, start_y],
                    "metric_problem": metric_problem,
                    "provider_local_snapshot": provider_local_snapshot,
                    "provider_local_snapshot_sha256": (
                        provider_local_snapshot["snapshot_sha256"]
                    ),
                    "terrain_arrays": terrain_arrays,
                    "terrain_provenance": provider_provenance,
                }
                source_sha = domain_hash(
                    "g2-raw-request-source/v4",
                    canonical_json_bytes(source_core),
                )
                row = {
                    **source_core,
                    "raw_source_id": (
                        f"g2i-raw-{platform}-{scale}-{source_sha[:20]}"
                    ),
                    "raw_source_sha256": source_sha,
                    "schema_version": "g2-truth-blind-request-source/v4",
                }
                validate_truth_blind_case(row)
                rows.append(row)
    expected_candidates = len(_PLATFORMS) * sum(
        int(specification["request_pool"][field])
        for field in ("standard_base_terrains", "kilometer_base_terrains")
    )
    if len(rows) + len(rejects) != expected_candidates:
        raise ValueError(
            "raw request source admission conservation mismatch: "
            f"{len(rows)} admitted + {len(rejects)} rejected "
            f"!= {expected_candidates}"
        )
    return rows, rejects


def generate_raw_request_sources(
    specification: dict[str, Any],
    *,
    lola_provenance: dict[str, Any],
    hopper_parameter_record: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows, _ = generate_raw_request_source_admission(
        specification,
        lola_provenance=lola_provenance,
        hopper_parameter_record=hopper_parameter_record,
    )
    return rows


def _array_payload(
    provenance: dict[str, Any],
    *,
    macro_height_mm: list[list[int]] | None = None,
    terrain_arrays: dict[str, Any] | None = None,
) -> dict[str, np.ndarray[Any, Any]]:
    if terrain_arrays is None:
        height_values = (
            macro_height_mm
            if macro_height_mm is not None
            else _fixture_macro_height_mm(
                int(provenance.get("source_seed", provenance.get("proxy_seed", 0))),
                int(provenance.get("roi_index", 0)),
            )
        )
        height_rows = len(height_values)
        height_columns = len(height_values[0]) if height_values else 0
        terrain_arrays = {
            "cell_class": [
                [0 for _ in range(height_columns)] for _ in range(height_rows)
            ],
            "confidence_ppm": [
                [1_000_000 for _ in range(height_columns)]
                for _ in range(height_rows)
            ],
            "height_mm": height_values,
            "known": [
                [1 for _ in range(height_columns)] for _ in range(height_rows)
            ],
        }
    return {
        "height_mm": np.asarray(terrain_arrays["height_mm"], dtype="<i4"),
        "cell_class": np.asarray(terrain_arrays["cell_class"], dtype="u1"),
        "known": np.asarray(terrain_arrays["known"], dtype="u1"),
        "confidence_ppm": np.asarray(
            terrain_arrays["confidence_ppm"], dtype="<u4"
        ),
    }


def terrain_npz_bytes(
    provenance: dict[str, Any],
    *,
    macro_height_mm: list[list[int]] | None = None,
    terrain_arrays: dict[str, Any] | None = None,
) -> bytes:
    arrays = _array_payload(
        provenance,
        macro_height_mm=macro_height_mm,
        terrain_arrays=terrain_arrays,
    )
    shapes = {tuple(array.shape) for array in arrays.values()}
    if len(shapes) != 1 or next(iter(shapes), (0, 0)) == (0, 0):
        raise ValueError(f"terrain arrays must share a non-empty shape: {shapes}")
    return canonical_npz_bytes(
        arrays,
        {
            "geometry_schema_version": "g2-neutral-terrain/v2",
            "macro_cell_size_mm": (
                20000
                if "macro_source_kind" in provenance
                else 500
            ),
            "provenance": provenance,
            "sub_20m_layer_source_kind": (
                provenance.get("micro_source_kind")
                if "macro_source_kind" in provenance
                else None
            ),
        },
    )


def terrain_geometry_sha256(payload: bytes) -> str:
    arrays, _ = decode_canonical_npz(payload)
    return domain_hash(
        "g2-terrain-geometry/v1",
        *[
            name.encode("ascii") + b"\0" + arrays[name].tobytes(order="C")
            for name in ("height_mm", "cell_class", "known", "confidence_ppm")
        ],
    )


def _metric_binding_sha256(raw_source: dict[str, Any]) -> str:
    return domain_hash(
        "g2-request-metric-problem/v3",
        canonical_json_bytes(raw_source["metric_problem"]),
    )


def _edge_obstacles(
    raw_source: dict[str, Any],
    *,
    source_xy: tuple[int, int],
    target_xy: tuple[int, int],
    source_pose_xy: tuple[int, int],
    target_pose_xy: tuple[int, int],
    halfwidth_mm: int,
) -> list[list[list[int]]]:
    del source_xy, target_xy, source_pose_xy, target_pose_xy, halfwidth_mm
    snapshot = raw_source["provider_local_snapshot"]
    resolution = int(snapshot["resolution_mm"])
    polygons: list[list[list[int]]] = []
    for row, values in enumerate(snapshot["arrays"]["hard_obstacle"]):
        for column, blocked in enumerate(values):
            if bool(blocked):
                polygons.append(
                    square_polygon(
                        column * resolution + resolution // 2,
                        row * resolution + resolution // 2,
                        resolution // 2,
                    )
                )
    return polygons


def _convex_hull_xy(points: list[list[int]]) -> list[list[int]]:
    unique = sorted({(int(point[0]), int(point[1])) for point in points})
    if len(unique) < 3:
        raise ValueError("support polygon requires three non-collinear points")

    def cross(
        origin: tuple[int, int],
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> int:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[int, int]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[int, int]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        raise ValueError("support polygon is degenerate")
    return [[x, y] for x, y in hull]


def _mean_pose_mm_rational(
    feet_mm: list[list[int]],
) -> dict[str, Any]:
    return {
        "denominator": len(feet_mm),
        "numerator_mm": [
            sum(int(foot[axis]) for foot in feet_mm) for axis in range(3)
        ],
    }


def _rational_pose_xy(pose: dict[str, Any]) -> tuple[float, float]:
    denominator = int(pose["denominator"])
    numerators = pose["numerator_mm"]
    return (
        int(numerators[0]) / denominator,
        int(numerators[1]) / denominator,
    )


def _signed_convex_margin_mm(
    point: tuple[float, float], polygon: list[list[int]]
) -> float:
    signs: set[int] = set()
    distances: list[float] = []
    for index, raw_end in enumerate(polygon):
        raw_start = polygon[index - 1]
        start = (float(raw_start[0]), float(raw_start[1]))
        end = (float(raw_end[0]), float(raw_end[1]))
        edge_x = end[0] - start[0]
        edge_y = end[1] - start[1]
        offset_x = point[0] - start[0]
        offset_y = point[1] - start[1]
        cross = edge_x * offset_y - edge_y * offset_x
        if cross > 0.0:
            signs.add(1)
        elif cross < 0.0:
            signs.add(-1)
        denominator = edge_x * edge_x + edge_y * edge_y
        projection = (
            0.0
            if denominator == 0.0
            else max(
                0.0,
                min(
                    1.0,
                    (offset_x * edge_x + offset_y * edge_y) / denominator,
                ),
            )
        )
        closest_x = start[0] + projection * edge_x
        closest_y = start[1] + projection * edge_y
        distances.append(math.hypot(point[0] - closest_x, point[1] - closest_y))
    distance = min(distances)
    return distance if len(signs) <= 1 else -distance


def _sweep_support_margin_um(
    start: dict[str, Any],
    end: dict[str, Any],
    polygon: list[list[int]],
) -> int:
    start_xy = _rational_pose_xy(start)
    end_xy = _rational_pose_xy(end)
    minimum_mm = min(
        _signed_convex_margin_mm(
            (
                start_xy[0] + (end_xy[0] - start_xy[0]) * index / 64.0,
                start_xy[1] + (end_xy[1] - start_xy[1]) * index / 64.0,
            ),
            polygon,
        )
        for index in range(65)
    )
    return math.floor(math.nextafter(minimum_mm * 1000.0, -math.inf))


def _rounded_pose_xy(pose: dict[str, Any]) -> list[int]:
    x_mm, y_mm = _rational_pose_xy(pose)
    return [round(x_mm), round(y_mm)]


def _wheel_edge_case(
    raw_source: dict[str, Any],
    source_xy: tuple[int, int],
    direction: tuple[int, int],
) -> dict[str, Any]:
    x, y = source_xy
    dx, dy = direction
    metric_problem = raw_source["metric_problem"]
    poses = metric_problem["node_metric_poses_mm_urad"]
    source_pose = poses[f"n:{x}:{y}"]
    requested_target_pose = [
        int(source_pose[0]) + dx * 500,
        int(source_pose[1]) + dy * 500,
        0,
    ]
    target_xy = (x + dx, y + dy)
    start = [
        int(source_pose[0]),
        int(source_pose[1]),
        round(math.atan2(dy, dx) * 1_000_000),
    ]
    case = {
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
            "angular_urad_s": 0,
            "control_family": "request_grid_translation",
            "duration_us": 1_000_000,
            "endpoint_mm_urad": [
                requested_target_pose[0],
                requested_target_pose[1],
                start[2],
            ],
            "heading_bin": 0,
            "speed_um_s": 500_000,
        },
        "metric_problem_sha256": _metric_binding_sha256(raw_source),
        "start_pose_mm_urad": start,
        "terrain_snapshot_sha256": raw_source[
            "provider_local_snapshot_sha256"
        ],
        "terrain": {
            "geometry_schema_version": "g2-wheel-request-geometry/v3",
            "height_plane": _snapshot_tangent_plane(
                raw_source["provider_local_snapshot"],
                int(source_pose[0]),
                int(source_pose[1]),
            ),
            "map_bounds_mm": metric_problem["terrain_bounds_mm"],
            "obstacle_polygons_mm": _edge_obstacles(
                raw_source,
                source_xy=source_xy,
                target_xy=target_xy,
                source_pose_xy=(int(source_pose[0]), int(source_pose[1])),
                target_pose_xy=(
                    requested_target_pose[0],
                    requested_target_pose[1],
                ),
                halfwidth_mm=166,
            ),
            "unknown_polygons_mm": [],
            "untraversable_polygons_mm": [],
            "wheel_footprint_radius_mm": 100,
        },
    }
    case["primitive"]["endpoint_mm_urad"] = list(wheel_endpoint_mm(case))
    snapshot = raw_source["provider_local_snapshot"]
    swept_cell_indices = fine_cells_intersecting_capsule(
        start_x_mm=float(start[0]),
        start_y_mm=float(start[1]),
        end_x_mm=float(case["primitive"]["endpoint_mm_urad"][0]),
        end_y_mm=float(case["primitive"]["endpoint_mm_urad"][1]),
        radius_mm=int(case["terrain"]["wheel_footprint_radius_mm"]),
        resolution_mm=int(snapshot["resolution_mm"]),
        shape_height_width=tuple(snapshot["shape_height_width"]),
    )
    swept_cells = [
        _snapshot_cell_record(snapshot, cell_xy)
        for cell_xy in swept_cell_indices
    ]
    snapshot_query_witness = {
        "boundary_semantics": "closed-cell-intersection/v1",
        "maximum_slope_cdeg": 3000,
        "query_kind": "wheel-swept-capsule/v1",
        "snapshot_sha256": raw_source["provider_local_snapshot_sha256"],
        "sweep_safe": _snapshot_cells_safe(swept_cells),
        "swept_cell_xy": swept_cell_indices,
    }
    snapshot_query_witness["query_sha256"] = domain_hash(
        "g2-wheel-snapshot-query/v1",
        canonical_json_bytes(snapshot_query_witness),
    )
    case["snapshot_query_witness"] = snapshot_query_witness
    return case


def _evaluate_wheel_request(
    case: dict[str, Any],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    decision = evaluate_wheel(case)
    query = case["snapshot_query_witness"]
    if query["snapshot_sha256"] != snapshot["snapshot_sha256"]:
        raise ValueError("wheel snapshot query/output snapshot mismatch")
    records = [
        _snapshot_cell_record(snapshot, cell_xy)
        for cell_xy in query["swept_cell_xy"]
    ]
    if bool(query["sweep_safe"]) != _snapshot_cells_safe(records):
        raise ValueError("wheel snapshot query safety summary mismatch")
    witness = {
        **decision.get("numeric_witness", {}),
        "snapshot_query_sha256": query["query_sha256"],
        "snapshot_swept_cell_count": len(records),
    }
    snapshot_slacks = {
        "snapshot_hard_obstacle": (
            -1 if any(record["hard_obstacle"] for record in records) else 1
        ),
        "snapshot_known": (
            -1 if any(not record["known"] for record in records) else 1
        ),
        "snapshot_slope_cdeg": min(
            (
                3000 - int(record["slope_cdeg"])
                for record in records
            ),
            default=-1,
        ),
        "snapshot_traversable": (
            -1 if any(not record["traversable"] for record in records) else 1
        ),
    }
    slacks = {
        **decision.get("safety_slacks", {}),
        **snapshot_slacks,
    }
    if not records or any(not record["known"] for record in records):
        return {
            "numeric_witness": witness,
            "oracle_reason_code": "G2I_W_UNKNOWN_SWEEP",
            "oracle_safe": False,
            "safety_slacks": slacks,
        }
    if any(record["hard_obstacle"] for record in records):
        return {
            "numeric_witness": witness,
            "oracle_reason_code": "G2I_W_CLOSED_OBSTACLE_CONTACT",
            "oracle_safe": False,
            "safety_slacks": slacks,
        }
    if any(not record["traversable"] for record in records):
        return {
            "numeric_witness": witness,
            "oracle_reason_code": "G2I_W_NOT_TRAVERSABLE",
            "oracle_safe": False,
            "safety_slacks": slacks,
        }
    if any(int(record["slope_cdeg"]) > 3000 for record in records):
        return {
            "numeric_witness": witness,
            "oracle_reason_code": "G2I_W_SLOPE_LIMIT",
            "oracle_safe": False,
            "safety_slacks": slacks,
        }
    return {
        **decision,
        "numeric_witness": witness,
        "safety_slacks": slacks,
    }


def _legged_edge_case(
    raw_source: dict[str, Any],
    source_xy: tuple[int, int],
    direction: tuple[int, int],
) -> dict[str, Any]:
    x, y = source_xy
    dx, dy = direction
    metric_problem = raw_source["metric_problem"]
    cycle_contract = metric_problem["legged_crawl_cycle"]
    source_body_xy = [
        int(value)
        for value in metric_problem["node_metric_poses_mm_urad"][f"n:{x}:{y}"][
            :2
        ]
    ]
    source_feet = []
    height_plane = _metric_height_plane(metric_problem)
    snapshot = raw_source["provider_local_snapshot"]
    for forward_mm, lateral_mm in cycle_contract[
        "nominal_stance_offsets_mm"
    ]:
        foot_x_mm = (
            source_body_xy[0]
            + dx * int(forward_mm)
            - dy * int(lateral_mm)
        )
        foot_y_mm = (
            source_body_xy[1]
            + dy * int(forward_mm)
            + dx * int(lateral_mm)
        )
        source_feet.append(
            [
                foot_x_mm,
                foot_y_mm,
                round(
                    _snapshot_elevation_um(
                        snapshot,
                        foot_x_mm,
                        foot_y_mm,
                    )
                    / 1000
                ),
            ]
        )
    target_xy = (x + dx, y + dy)
    shared_obstacle_polygons = _edge_obstacles(
        raw_source,
        source_xy=source_xy,
        target_xy=target_xy,
        source_pose_xy=(source_body_xy[0], source_body_xy[1]),
        target_pose_xy=(
            int(metric_problem["node_metric_poses_mm_urad"][f"n:{target_xy[0]}:{target_xy[1]}"][0]),
            int(metric_problem["node_metric_poses_mm_urad"][f"n:{target_xy[0]}:{target_xy[1]}"][1]),
        ) if f"n:{target_xy[0]}:{target_xy[1]}" in metric_problem["node_metric_poses_mm_urad"] else (source_body_xy[0], source_body_xy[1]),
        halfwidth_mm=250,
    )
    cycle_steps: list[dict[str, Any]] = []
    current_feet = source_feet
    current_body = _mean_pose_mm_rational(source_feet)
    for phase_contract in cycle_contract["phase_steps"]:
        moving_leg_id = int(phase_contract["moving_leg_id"])
        forward_mm, lateral_mm = phase_contract[
            "forward_lateral_delta_mm"
        ]
        moving_delta = [
            dx * int(forward_mm) - dy * int(lateral_mm),
            dy * int(forward_mm) + dx * int(lateral_mm),
            0,
        ]
        target_feet = [list(foot) for foot in current_feet]
        for axis in range(3):
            target_feet[moving_leg_id][axis] += moving_delta[axis]
        target_feet[moving_leg_id][2] = plane_height_mm(
            {
                "gradient_x_ppm": 0,
                "gradient_y_ppm": 0,
                "origin_x_mm": target_feet[moving_leg_id][0],
                "origin_y_mm": target_feet[moving_leg_id][1],
                "origin_z_mm": round(
                    _snapshot_elevation_um(
                        snapshot,
                        target_feet[moving_leg_id][0],
                        target_feet[moving_leg_id][1],
                    )
                    / 1000
                ),
            },
            target_feet[moving_leg_id][0],
            target_feet[moving_leg_id][1],
        )
        moving_delta[2] = (
            target_feet[moving_leg_id][2]
            - current_feet[moving_leg_id][2]
        )
        nonmoving = [
            foot
            for leg_id, foot in enumerate(current_feet)
            if leg_id != moving_leg_id
        ]
        source_body = current_body
        lift_body = _mean_pose_mm_rational(nonmoving)
        target_body = lift_body
        settled_body = (
            _mean_pose_mm_rational(target_feet)
            if int(phase_contract["phase"]) == 3
            else lift_body
        )
        four_support = _convex_hull_xy(current_feet)
        three_support = _convex_hull_xy(nonmoving)
        target_four_support = _convex_hull_xy(target_feet)
        minimum_margin_um = min(
            _sweep_support_margin_um(
                source_body,
                lift_body,
                four_support,
            ),
            _sweep_support_margin_um(
                lift_body,
                target_body,
                three_support,
            ),
            _sweep_support_margin_um(
                target_body,
                settled_body,
                target_four_support,
            ),
        )
        target_foot = target_feet[moving_leg_id]
        body_start_xy = _rounded_pose_xy(lift_body)
        body_end_xy = _rounded_pose_xy(target_body)
        foothold_cell_indices = fine_cells_intersecting_disk(
            center_x_mm=float(target_foot[0]),
            center_y_mm=float(target_foot[1]),
            radius_mm=0,
            resolution_mm=_PROVIDER_FINE_RESOLUTION_MM,
            shape_height_width=(20, 20),
        )
        body_cell_indices = fine_cells_intersecting_capsule(
            start_x_mm=float(body_start_xy[0]),
            start_y_mm=float(body_start_xy[1]),
            end_x_mm=float(body_end_xy[0]),
            end_y_mm=float(body_end_xy[1]),
            radius_mm=100,
            resolution_mm=_PROVIDER_FINE_RESOLUTION_MM,
            shape_height_width=(20, 20),
        )
        foothold_cells = [
            _snapshot_cell_record(snapshot, cell)
            for cell in foothold_cell_indices
        ]
        body_sweep_cells = [
            _snapshot_cell_record(snapshot, cell)
            for cell in body_cell_indices
        ]
        snapshot_query_witness = {
            "body_sweep_cell_xy": body_cell_indices,
            "body_sweep_safe": _snapshot_cells_safe(body_sweep_cells),
            "foothold_cell_xy": foothold_cell_indices,
            "foothold_safe": _snapshot_cells_safe(foothold_cells),
            "snapshot_sha256": raw_source[
                "provider_local_snapshot_sha256"
            ],
        }
        snapshot_query_witness["query_sha256"] = domain_hash(
            "g2-leg-snapshot-query/v1",
            canonical_json_bytes(snapshot_query_witness),
        )
        obstacle_polygons = shared_obstacle_polygons
        body_obstacle_polygons = shared_obstacle_polygons
        transition = {
            "four_foot_support_polygon_mm": four_support,
            "lift_body_pose_mm_rational": lift_body,
            "minimum_support_margin_um": minimum_margin_um,
            "moving_foot_delta_mm": moving_delta,
            "schema_version": "g2-leg-crawl-stance-transition/v2",
            "settled_body_pose_mm_rational": settled_body,
            "source_body_pose_mm_rational": source_body,
            "source_feet_mm": current_feet,
            "target_body_pose_mm_rational": target_body,
            "target_feet_mm": target_feet,
            "three_foot_support_polygon_mm": three_support,
        }
        cycle_steps.append(
            {
                "metric_problem_sha256": _metric_binding_sha256(raw_source),
                "moving_leg": phase_contract["moving_leg"],
                "moving_leg_id": moving_leg_id,
                "numeric_state": "decided",
                "phase": int(phase_contract["phase"]),
                "required_phase": int(phase_contract["phase"]),
                "source_foot_mm": current_feet[moving_leg_id],
                "snapshot_query_witness": snapshot_query_witness,
                "stance_transition": transition,
                "target_foot_mm": target_foot,
                "terrain_snapshot_sha256": raw_source[
                    "provider_local_snapshot_sha256"
                ],
                "terrain": {
                    "body_end_mm": body_end_xy,
                    "body_obstacle_polygons_mm": body_obstacle_polygons,
                    "body_plane": _snapshot_tangent_plane(
                        snapshot,
                        body_end_xy[0],
                        body_end_xy[1],
                    ),
                    "body_radius_mm": 100,
                    "body_start_mm": body_start_xy,
                    "foothold_plane": _snapshot_tangent_plane(
                        snapshot,
                        target_foot[0],
                        target_foot[1],
                    ),
                    "geometry_schema_version": (
                        "g2-legged-request-phase-geometry/v4"
                    ),
                    "map_bounds_mm": metric_problem["terrain_bounds_mm"],
                    "obstacle_polygons_mm": obstacle_polygons,
                    "support_polygon_mm": three_support,
                    "target_com_mm": _rounded_pose_xy(target_body),
                    "unknown_polygons_mm": [],
                    "untraversable_polygons_mm": [],
                },
            }
        )
        current_feet = target_feet
        current_body = settled_body
    cycle_minimum_support_margin_um = min(
        int(step["stance_transition"]["minimum_support_margin_um"])
        for step in cycle_steps
    )
    terminal_body_x_mm, terminal_body_y_mm = _rational_pose_xy(current_body)
    source_state = metric_problem["node_states"][f"n:{x}:{y}"]
    actual_terminal_state = {
        "body_pose": _binary64_pose_identity(
            terminal_body_x_mm / 1000.0,
            terminal_body_y_mm / 1000.0,
            float.fromhex(source_state["body_pose"]["heading_rad_hex"]),
        ),
        "cycle_phase": (
            int(source_state["cycle_phase"]) + len(cycle_steps)
        )
        % len(cycle_steps),
        "foot_contacts": [
            {
                "leg_id": leg_id,
                **_binary64_point_identity(
                    int(foot[0]) / 1000.0,
                    int(foot[1]) / 1000.0,
                    _snapshot_elevation_um(
                        snapshot,
                        int(foot[0]),
                        int(foot[1]),
                    )
                    / 1_000_000.0,
                ),
            }
            for leg_id, foot in zip(
                cycle_contract["foot_storage_order"],
                current_feet,
                strict=True,
            )
        ],
        "schema_version": "g2-legged-node-state/v1",
    }
    return {
        "actual_terminal_state": actual_terminal_state,
        "crawl_cycle": cycle_steps,
        "cycle_body_delta_mm_rational": {
            "denominator": 4,
            "numerator_mm": [
                current_feet_sum - source_feet_sum
                for current_feet_sum, source_feet_sum in zip(
                    (
                        sum(foot[axis] for foot in current_feet)
                        for axis in range(3)
                    ),
                    (
                        sum(foot[axis] for foot in source_feet)
                        for axis in range(3)
                    ),
                    strict=True,
                )
            ],
        },
        "cycle_minimum_support_margin_um": cycle_minimum_support_margin_um,
        "cycle_schema_version": "g2-independent-static-crawl-edge/v1",
        "metric_problem_sha256": _metric_binding_sha256(raw_source),
        "numeric_state": "decided",
        "terrain_snapshot_sha256": raw_source[
            "provider_local_snapshot_sha256"
        ],
    }


def _evaluate_legged_cycle(
    case: dict[str, Any],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    minimum_required_um = 50_000
    decisions: list[dict[str, Any]] = []
    for step in case["crawl_cycle"]:
        decision = evaluate_legged(step)
        query = step["snapshot_query_witness"]
        if query["snapshot_sha256"] != snapshot["snapshot_sha256"]:
            raise ValueError("legged snapshot query/output snapshot mismatch")
        foothold_cells = [
            _snapshot_cell_record(snapshot, cell_xy)
            for cell_xy in query["foothold_cell_xy"]
        ]
        body_sweep_cells = [
            _snapshot_cell_record(snapshot, cell_xy)
            for cell_xy in query["body_sweep_cell_xy"]
        ]
        foothold_safe = _snapshot_cells_safe(foothold_cells)
        body_sweep_safe = _snapshot_cells_safe(body_sweep_cells)
        if (
            bool(query["foothold_safe"]) != foothold_safe
            or bool(query["body_sweep_safe"]) != body_sweep_safe
        ):
            raise ValueError("legged snapshot query safety summary mismatch")
        if bool(decision["oracle_safe"]) and not foothold_safe:
            if any(not cell["known"] for cell in foothold_cells):
                reason = "G2I_L_FOOTHOLD_UNKNOWN"
            elif any(cell["hard_obstacle"] for cell in foothold_cells):
                reason = "G2I_L_FOOTHOLD_OBSTACLE"
            elif any(
                int(cell["slope_cdeg"]) > 3000 for cell in foothold_cells
            ):
                reason = "G2I_L_FOOTHOLD_SLOPE"
            else:
                reason = "G2I_L_FOOTHOLD_UNTRAVERSABLE"
            decision = {
                "numeric_witness": {
                    **decision["numeric_witness"],
                    "snapshot_query_sha256": query["query_sha256"],
                },
                "oracle_reason_code": reason,
                "oracle_safe": False,
                "safety_slacks": {
                    **decision["safety_slacks"],
                    "snapshot_foothold": -1,
                },
            }
        if bool(decision["oracle_safe"]) and not body_sweep_safe:
            decision = {
                "numeric_witness": {
                    **decision["numeric_witness"],
                    "snapshot_query_sha256": query["query_sha256"],
                },
                "oracle_reason_code": "G2I_L_BODY_SWEEP",
                "oracle_safe": False,
                "safety_slacks": {
                    **decision["safety_slacks"],
                    "snapshot_body_sweep": -1,
                },
            }
        decisions.append(decision)
        if (
            int(step["stance_transition"]["minimum_support_margin_um"])
            < minimum_required_um
            and bool(decision["oracle_safe"])
        ):
            decision = {
                "numeric_witness": decision["numeric_witness"],
                "oracle_reason_code": "G2I_L_SUPPORT_MARGIN",
                "oracle_safe": False,
                "safety_slacks": {
                    **decision["safety_slacks"],
                    "cycle_support_margin_um": (
                        int(
                            step["stance_transition"][
                                "minimum_support_margin_um"
                            ]
                        )
                        - minimum_required_um
                    ),
                },
            }
            decisions[-1] = decision
    first_failure = next(
        (decision for decision in decisions if not decision["oracle_safe"]),
        None,
    )
    all_slacks = [
        int(value)
        for decision in decisions
        for value in decision["safety_slacks"].values()
    ]
    cycle_slack_mm = (
        int(case["cycle_minimum_support_margin_um"]) - minimum_required_um
    ) // 1000
    return {
        "numeric_witness": {
            "actual_terminal_state": case["actual_terminal_state"],
            "cycle_body_delta_mm_rational": case[
                "cycle_body_delta_mm_rational"
            ],
            "cycle_minimum_support_margin_um": case[
                "cycle_minimum_support_margin_um"
            ],
            "phase_decision_sha256": [
                domain_hash(
                    "g2-leg-crawl-phase-decision/v1",
                    canonical_json_bytes(decision),
                )
                for decision in decisions
            ],
            "phase_count": 4,
        },
        "oracle_reason_code": (
            "G2I_L_SAFE"
            if first_failure is None
            else first_failure["oracle_reason_code"]
        ),
        "oracle_safe": first_failure is None,
        "safety_slacks": {
            "cycle_support_margin_mm": cycle_slack_mm,
            "minimum_phase_slack": min(all_slacks, default=cycle_slack_mm),
        },
    }


def _hopper_edge_case(
    raw_source: dict[str, Any],
    source_xy: tuple[int, int],
    direction_index: int,
    parameter_record: dict[str, Any],
) -> dict[str, Any]:
    x, y = source_xy
    dx, dy = _DIRECTIONS[direction_index]
    metric_problem = raw_source["metric_problem"]
    source_pose = metric_problem["node_metric_poses_mm_urad"][f"n:{x}:{y}"]
    action = raw_source["action_envelope"]["actions"][direction_index]
    exact_source = metric_problem[
        "hopper_node_metric_poses_binary64_m_rad"
    ][f"n:{x}:{y}"]
    exact_kinematics = _hopper_exact_kinematics(
        action,
        direction=(dx, dy),
        ballistic_contract=metric_problem["hopper_ballistic_binding"][
            "ballistic_contract"
        ],
    )
    exact_endpoint = _binary64_pose_identity(
        float.fromhex(exact_source["x_m_hex"])
        + float.fromhex(exact_kinematics["delta_x_m_hex"]),
        float.fromhex(exact_source["y_m_hex"])
        + float.fromhex(exact_kinematics["delta_y_m_hex"]),
        float.fromhex(exact_source["heading_rad_hex"]),
    )
    exact_kinematics["endpoint_pose_binary64_m_rad"] = exact_endpoint
    exact_kinematics["launch_pose_binary64_m_rad"] = exact_source
    snapshot = raw_source["provider_local_snapshot"]
    snapshot_sha = raw_source["provider_local_snapshot_sha256"]
    launch_radius_mm = round(
        (
            float(parameter_record["body_envelope_radius_m"])
            + float(parameter_record["arc_clearance_margin_m"])
        )
        * 1000.0
    )
    landing_radius_mm = round(
        float(parameter_record["landing_footprint_radius_m"]) * 1000.0
    )
    landing_sigma_mm = 289
    landing_prefix_probability_ppm = 990_000
    landing_prefix_radius_mm = math.ceil(
        landing_sigma_mm
        * math.sqrt(
            -2.0
            * math.log(1.0 - landing_prefix_probability_ppm / 1_000_000.0)
        )
    )
    landing_required_radius_mm = (
        landing_radius_mm + landing_prefix_radius_mm
    )
    launch_plane = _snapshot_tangent_plane(
        snapshot,
        int(source_pose[0]),
        int(source_pose[1]),
    )
    probe = {
        "action": {
            key: value for key, value in action.items() if key != "delta_cell_xy"
        },
        "launch_pose_mm": [int(source_pose[0]), int(source_pose[1]), 0],
        "metric_problem_sha256": _metric_binding_sha256(raw_source),
        "numeric_state": "decided",
        "terrain": {"height_plane": launch_plane},
    }
    map_bounds = metric_problem["terrain_bounds_mm"]
    shared_hard_polygons = _edge_obstacles(
        raw_source,
        source_xy=source_xy,
        target_xy=(x + dx, y + dy),
        source_pose_xy=(int(source_pose[0]), int(source_pose[1])),
        target_pose_xy=(
            int(source_pose[0]) + round(
                float.fromhex(exact_kinematics["delta_x_m_hex"]) * 1000.0
            ),
            int(source_pose[1]) + round(
                float.fromhex(exact_kinematics["delta_y_m_hex"]) * 1000.0
            ),
        ),
        halfwidth_mm=0,
    )
    case = {
        "action": probe["action"],
        "launch_pose_mm": probe["launch_pose_mm"],
        "metric_problem_sha256": probe["metric_problem_sha256"],
        "numeric_state": "decided",
        "provider_exact_kinematics": exact_kinematics,
        "relief_preservation_sha256": metric_problem[
            "relief_preservation_sha256"
        ],
        "terrain_snapshot_sha256": snapshot_sha,
        "terrain": {
            "geometry_schema_version": "g2-hopper-request-geometry/v3",
            "height_plane": launch_plane,
            "landing_height_tolerance_mm": 50,
            "landing_pad_polygon_mm": [
                [int(map_bounds[0]), int(map_bounds[1])],
                [int(map_bounds[2]), int(map_bounds[1])],
                [int(map_bounds[2]), int(map_bounds[3])],
                [int(map_bounds[0]), int(map_bounds[3])],
            ],
            "landing_sigma_mm": landing_sigma_mm,
            "landing_surface_plane": launch_plane,
            "map_bounds_mm": map_bounds,
            "obstacle_polygons_mm": shared_hard_polygons,
            "obstacle_prisms": [],
            "unknown_polygons_mm": [],
        },
    }
    witness = ballistic_witness(case, parameter_record)
    landing = (int(witness["landing_x_mm"]), int(witness["landing_y_mm"]))
    case["terrain"]["landing_surface_plane"] = _snapshot_tangent_plane(
        snapshot,
        landing[0],
        landing[1],
    )
    shape = tuple(int(value) for value in snapshot["shape_height_width"])
    resolution_mm = int(snapshot["resolution_mm"])
    launch_indices = fine_cells_intersecting_disk(
        center_x_mm=float(source_pose[0]),
        center_y_mm=float(source_pose[1]),
        radius_mm=launch_radius_mm,
        resolution_mm=resolution_mm,
        shape_height_width=shape,
    )
    landing_indices = fine_cells_intersecting_disk(
        center_x_mm=float(landing[0]),
        center_y_mm=float(landing[1]),
        radius_mm=landing_required_radius_mm,
        resolution_mm=resolution_mm,
        shape_height_width=shape,
    )
    arc_indices = fine_cells_intersecting_capsule(
        start_x_mm=float(source_pose[0]),
        start_y_mm=float(source_pose[1]),
        end_x_mm=float(landing[0]),
        end_y_mm=float(landing[1]),
        radius_mm=launch_radius_mm,
        resolution_mm=resolution_mm,
        shape_height_width=shape,
    )
    h_ref_um = int(metric_problem["support_plane"]["H_ref_um"])

    def support_record(cell_xy: list[int]) -> dict[str, Any]:
        record = _snapshot_cell_record(snapshot, cell_xy)
        residual_um = int(record.get("elevation_um", h_ref_um)) - h_ref_um
        return {
            **record,
            "support_residual_mm": residual_um / 1000.0,
            "support_residual_um": residual_um,
        }

    launch_records = [support_record(cell) for cell in launch_indices]
    landing_records = [support_record(cell) for cell in landing_indices]
    arc_records = [
        _snapshot_cell_record(snapshot, cell)
        for cell in arc_indices
    ]

    def cell_bbox(cells: list[list[int]]) -> list[int]:
        columns = [int(cell[0]) for cell in cells]
        rows = [int(cell[1]) for cell in cells]
        return [min(columns), min(rows), max(columns), max(rows)]

    required_cells = {
        "arc": arc_indices,
        "arc_bbox_cell_xy": cell_bbox(arc_indices),
        "arc_rule": {
            "capsule_radius_mm": launch_radius_mm,
            "semantics": "closed-fine-cell-intersection/v1",
        },
        "boundary_kind": "closed",
        "H_ref_um": h_ref_um,
        "landing": landing_indices,
        "landing_bbox_cell_xy": cell_bbox(landing_indices),
        "landing_rule": {
            "distribution_prefix_probability_ppm": (
                landing_prefix_probability_ppm
            ),
            "distribution_prefix_radius_mm": landing_prefix_radius_mm,
            "footprint_radius_mm": landing_radius_mm,
            "required_radius_mm": landing_required_radius_mm,
            "semantics": "closed-fine-cell-intersection/v1",
            "sigma_mm": landing_sigma_mm,
        },
        "launch": launch_indices,
        "launch_bbox_cell_xy": cell_bbox(launch_indices),
        "launch_rule": {
            "arc_clearance_margin_mm": round(
                float(parameter_record["arc_clearance_margin_m"]) * 1000.0
            ),
            "body_envelope_radius_mm": round(
                float(parameter_record["body_envelope_radius_m"]) * 1000.0
            ),
            "required_radius_mm": launch_radius_mm,
            "semantics": "closed-fine-cell-intersection/v1",
        },
        "maximum_slope_cdeg": 3000,
        "schema_version": "g2-hopper-required-cells/v1",
        "snapshot_sha256": snapshot_sha,
        "support_tolerance_mm": 50,
        "support_tolerance_um": 50_000,
    }
    required_cells["required_cells_sha256"] = domain_hash(
        "g2-hopper-required-cells/v1",
        canonical_json_bytes(required_cells),
    )
    support_residual_um = max(
        (
            abs(int(record["support_residual_um"]))
            for record in launch_records + landing_records
        ),
        default=2**63 - 1,
    )
    case["required_cells"] = required_cells
    case["support_height_envelope"] = {
        "landing_radius_mm": landing_required_radius_mm,
        "launch_radius_mm": launch_radius_mm,
        "maximum_residual_um": support_residual_um,
        "maximum_tolerated_residual_um": 50_000,
        "required_cells_sha256": required_cells["required_cells_sha256"],
        "schema_version": "g2-hopper-support-height-envelope/v2",
    }
    return case


def _hopper_required_cells_decision(
    required_cells: dict[str, Any],
    snapshot: dict[str, Any],
) -> tuple[bool, str, dict[str, int]]:
    if required_cells["snapshot_sha256"] != snapshot["snapshot_sha256"]:
        raise ValueError("Hopper required cells/output snapshot mismatch")
    tolerance_um = int(required_cells["support_tolerance_um"])
    maximum_slope_cdeg = int(required_cells["maximum_slope_cdeg"])
    h_ref_um = int(required_cells["H_ref_um"])

    def support_record(cell_xy: list[int]) -> dict[str, Any]:
        record = _snapshot_cell_record(snapshot, cell_xy)
        return {
            **record,
            "support_residual_um": (
                int(record.get("elevation_um", h_ref_um)) - h_ref_um
            ),
        }

    launch = [
        support_record(cell_xy) for cell_xy in required_cells["launch"]
    ]
    landing = [
        support_record(cell_xy) for cell_xy in required_cells["landing"]
    ]
    arc = [
        _snapshot_cell_record(snapshot, cell_xy)
        for cell_xy in required_cells["arc"]
    ]
    support_records = launch + landing
    maximum_residual_um = max(
        (
            abs(int(record["support_residual_um"]))
            for record in support_records
        ),
        default=2**63 - 1,
    )
    slacks = {
        "required_arc_hard_obstacle": (
            -1 if any(record["hard_obstacle"] for record in arc) else 1
        ),
        "required_arc_known": (
            -1 if any(not record["known"] for record in arc) else 1
        ),
        "required_arc_slope_cdeg": min(
            (
                maximum_slope_cdeg - int(record["slope_cdeg"])
                for record in arc
            ),
            default=-1,
        ),
        "required_arc_traversable": (
            -1 if any(not record["traversable"] for record in arc) else 1
        ),
        "required_support_hard_obstacle": (
            -1
            if any(record["hard_obstacle"] for record in support_records)
            else 1
        ),
        "required_support_height_um": tolerance_um - maximum_residual_um,
        "required_support_known": (
            -1
            if any(not record["known"] for record in support_records)
            else 1
        ),
        "required_support_slope_cdeg": min(
            (
                maximum_slope_cdeg - int(record["slope_cdeg"])
                for record in support_records
            ),
            default=-1,
        ),
        "required_support_traversable": (
            -1
            if any(not record["traversable"] for record in support_records)
            else 1
        ),
    }
    if not launch or not landing or not arc:
        return False, "G2I_H_REQUIRED_CELLS_EMPTY", slacks
    if any(not record["known"] for record in launch):
        return False, "G2I_H_LAUNCH_UNKNOWN", slacks
    if any(record["hard_obstacle"] for record in launch):
        return False, "G2I_H_LAUNCH_FOOTPRINT", slacks
    if any(not record["traversable"] for record in launch):
        return False, "G2I_H_LAUNCH_NOT_TRAVERSABLE", slacks
    if any(int(record["slope_cdeg"]) > maximum_slope_cdeg for record in launch):
        return False, "G2I_H_LAUNCH_SLOPE", slacks
    if any(not record["known"] for record in arc):
        return False, "G2I_H_ARC_UNKNOWN", slacks
    if any(record["hard_obstacle"] for record in arc):
        return False, "G2I_H_ARC_CLEARANCE", slacks
    if any(not record["traversable"] for record in arc):
        return False, "G2I_H_ARC_NOT_TRAVERSABLE", slacks
    if any(int(record["slope_cdeg"]) > maximum_slope_cdeg for record in arc):
        return False, "G2I_H_ARC_SLOPE", slacks
    if any(not record["known"] for record in landing):
        return False, "G2I_H_LANDING_UNKNOWN", slacks
    if any(record["hard_obstacle"] for record in landing):
        return False, "G2I_H_LANDING_FOOTPRINT", slacks
    if any(not record["traversable"] for record in landing):
        return False, "G2I_H_LANDING_NOT_TRAVERSABLE", slacks
    if any(
        int(record["slope_cdeg"]) > maximum_slope_cdeg
        for record in landing
    ):
        return False, "G2I_H_LANDING_SLOPE", slacks
    if maximum_residual_um > tolerance_um:
        return False, "G2I_H_LANDING_HEIGHT", slacks
    return True, "G2I_H_REQUIRED_CELLS_SAFE", slacks


def _evaluate_hopper_request(
    case: dict[str, Any],
    parameter_record: dict[str, Any],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    decision = evaluate_hopper(case, parameter_record)
    required_safe, required_reason, required_slacks = (
        _hopper_required_cells_decision(case["required_cells"], snapshot)
    )
    envelope = case["support_height_envelope"]
    residual_um = int(envelope["maximum_residual_um"])
    tolerance_um = int(envelope["maximum_tolerated_residual_um"])
    slacks = {
        **decision.get("safety_slacks", {}),
        **required_slacks,
        "support_height_envelope_mm": (tolerance_um - residual_um) // 1000,
    }
    witness = {
        **decision.get("numeric_witness", {}),
        "required_cells_sha256": case["required_cells"][
            "required_cells_sha256"
        ],
        "required_cells_safe": required_safe,
        "support_height_envelope_max_residual_um": residual_um,
        "support_height_envelope_tolerance_um": tolerance_um,
    }
    if not required_safe:
        return {
            "numeric_witness": witness,
            "oracle_reason_code": required_reason,
            "oracle_safe": False,
            "safety_slacks": slacks,
        }
    return {
        **decision,
        "numeric_witness": witness,
        "safety_slacks": slacks,
    }


def _decision_slack(decision: dict[str, Any]) -> int:
    slacks = [int(value) for value in decision.get("safety_slacks", {}).values()]
    non_domain = [value for value in slacks if value < 2**29]
    return min(non_domain, default=0)


def _canonical_state_xy_m(state: dict[str, Any]) -> tuple[float, float]:
    pose = state.get("body_pose", state.get("pose"))
    if not isinstance(pose, dict):
        raise ValueError("canonical node state pose missing")
    return float.fromhex(str(pose["x_m_hex"])), float.fromhex(
        str(pose["y_m_hex"])
    )


def _metric_l2_cost_milli(
    source_state: dict[str, Any],
    target_state: dict[str, Any],
) -> int:
    source_x_m, source_y_m = _canonical_state_xy_m(source_state)
    target_x_m, target_y_m = _canonical_state_xy_m(target_state)
    return round(
        math.hypot(target_x_m - source_x_m, target_y_m - source_y_m)
        * 1000.0
    )


def _metric_l2_witness(
    source_state: dict[str, Any],
    target_state: dict[str, Any],
    *,
    modeled_range_identity: dict[str, str] | None = None,
) -> dict[str, Any]:
    source_x_m, source_y_m = _canonical_state_xy_m(source_state)
    target_x_m, target_y_m = _canonical_state_xy_m(target_state)
    node_l2_m = math.hypot(
        target_x_m - source_x_m, target_y_m - source_y_m
    )
    node_identity = _binary64_identity(node_l2_m)
    witness: dict[str, Any] = {
        "cost_milli": round(node_l2_m * 1000.0),
        "node_l2_m_hex": node_identity["hex"],
        "node_l2_m_word_hex": node_identity["word_hex"],
        "rounding_policy": "python-binary64-round-times-1000/v1",
        "schema_version": "g2-edge-metric-l2-witness/v1",
    }
    if modeled_range_identity is not None:
        modeled_m = float.fromhex(modeled_range_identity["hex"])
        modeled_cost_milli = round(modeled_m * 1000.0)
        witness.update(
            {
                "modeled_range_cost_milli": modeled_cost_milli,
                "modeled_range_m_hex": modeled_range_identity["hex"],
                "modeled_range_m_word_hex": modeled_range_identity[
                    "word_hex"
                ],
                "node_minus_modeled_range_ulp": (
                    int(node_identity["word_hex"], 16)
                    - int(modeled_range_identity["word_hex"], 16)
                ),
                "rounding_equal_milli": (
                    modeled_cost_milli == witness["cost_milli"]
                ),
            }
        )
    return witness


def _graph_from_raw_source(
    raw_source: dict[str, Any],
    *,
    hopper_parameter_record: dict[str, Any],
    profile_record_sha256: str | None = None,
) -> tuple[dict[str, Any], str, str]:
    platform = str(raw_source["platform_kind"])
    profile_sha = profile_record_sha256 or (
        sha256_bytes(canonical_json_bytes(hopper_parameter_record))
        if platform == "hopper"
        else domain_hash("g2-independent-profile/v2", platform.encode("ascii"))
    )
    if (
        len(profile_sha) != 64
        or any(character not in "0123456789abcdef" for character in profile_sha)
    ):
        raise ValueError("invalid graph profile record SHA-256")
    metric_problem = raw_source.get("metric_problem")
    if (
        not isinstance(metric_problem, dict)
        or metric_problem.get("schema_version")
        != "g2-metric-planning-problem/v3"
    ):
        raise ValueError("request metric problem binding missing")
    width, height = _graph_shape(platform)
    nodes = [f"n:{x}:{y}" for x in range(width) for y in range(height)]
    if set(metric_problem["node_metric_poses_mm_urad"]) != set(nodes):
        raise ValueError("request metric node pose set mismatch")
    node_states = metric_problem.get("node_states")
    if not isinstance(node_states, dict) or set(node_states) != set(nodes):
        raise ValueError("request metric node state set mismatch")
    snapshot_sha = raw_source.get("provider_local_snapshot_sha256")
    if (
        not isinstance(snapshot_sha, str)
        or snapshot_sha
        != raw_source["provider_local_snapshot"].get("snapshot_sha256")
        or snapshot_sha != metric_problem.get("provider_local_snapshot_sha256")
    ):
        raise ValueError("request local terrain snapshot binding mismatch")
    metric_problem_sha = _metric_binding_sha256(raw_source)
    action_envelope_sha = domain_hash(
        "g2-request-action-envelope/v2",
        canonical_json_bytes(raw_source["action_envelope"]),
    )
    topology_binding = {
        "action_ids": [
            action["action_id"]
            for action in raw_source["action_envelope"]["actions"]
        ],
        "candidate_edge_count": len(nodes)
        * len(raw_source["action_envelope"]["actions"]),
        "node_count": len(nodes),
        "nodes": nodes,
    }
    semantic_binding_sha = domain_hash(
        "g2-request-graph-semantic-binding/v2",
        metric_problem_sha.encode("ascii"),
        action_envelope_sha.encode("ascii"),
        profile_sha.encode("ascii"),
        snapshot_sha.encode("ascii"),
        str(metric_problem["node_state_root_sha256"]).encode("ascii"),
        canonical_json_bytes(topology_binding),
    )
    edges: list[dict[str, Any]] = []
    for x in range(width):
        for y in range(height):
            source = f"n:{x}:{y}"
            for action_index, action in enumerate(
                raw_source["action_envelope"]["actions"]
            ):
                dx, dy = (int(value) for value in action["delta_cell_xy"])
                target_x, target_y = x + dx, y + dy
                inside = 0 <= target_x < width and 0 <= target_y < height
                target = (
                    f"n:{target_x}:{target_y}" if inside else source
                )
                if not inside:
                    oracle_input = {
                        "outside_grid_source_cell_xy": [x, y],
                        "requested_delta_cell_xy": [dx, dy],
                    }
                    decision = {
                        "numeric_witness": {
                            "outside_grid_target_cell_xy": [
                                target_x,
                                target_y,
                            ]
                        },
                        "oracle_reason_code": "G2I_GRAPH_DOMAIN_OUTSIDE",
                        "oracle_safe": False,
                        "safety_slacks": {"graph_domain": -1},
                    }
                elif platform == "wheel":
                    oracle_input = _wheel_edge_case(
                        raw_source, (x, y), (dx, dy)
                    )
                    decision = _evaluate_wheel_request(
                        oracle_input,
                        raw_source["provider_local_snapshot"],
                    )
                elif platform == "legged":
                    oracle_input = _legged_edge_case(
                        raw_source, (x, y), (dx, dy)
                    )
                    decision = _evaluate_legged_cycle(
                        oracle_input,
                        raw_source["provider_local_snapshot"],
                    )
                else:
                    oracle_input = _hopper_edge_case(
                        raw_source,
                        (x, y),
                        action_index,
                        hopper_parameter_record,
                    )
                    decision = _evaluate_hopper_request(
                        oracle_input,
                        hopper_parameter_record,
                        raw_source["provider_local_snapshot"],
                    )
                source_state = node_states[source]
                target_state = node_states[target]
                if not isinstance(oracle_input, dict):
                    raise ValueError("request edge oracle input must be an object")
                oracle_input = {
                    **oracle_input,
                    "source_state": source_state,
                    "target_state": target_state,
                    "terrain_snapshot_sha256": snapshot_sha,
                }
                if platform == "hopper" and inside:
                    endpoint = oracle_input["provider_exact_kinematics"][
                        "endpoint_pose_binary64_m_rad"
                    ]
                    if endpoint != target_state["pose"]:
                        decision = {
                            "numeric_witness": {
                                "actual_endpoint": endpoint,
                                "target_state": target_state,
                            },
                            "oracle_reason_code": "G2I_STATE_JOIN",
                            "oracle_safe": False,
                            "safety_slacks": {"binary64_ulp": -1},
                        }
                elif platform == "wheel" and inside:
                    endpoint = decision["numeric_witness"][
                        "computed_endpoint_mm_urad"
                    ]
                    if (
                        _pose_binary64_from_display(endpoint)
                        != target_state["pose"]
                    ):
                        decision = {
                            "numeric_witness": {
                                **decision["numeric_witness"],
                                "target_state": target_state,
                            },
                            "oracle_reason_code": "G2I_STATE_JOIN",
                            "oracle_safe": False,
                            "safety_slacks": {"binary64_ulp": -1},
                        }
                elif platform == "legged" and inside:
                    actual_terminal_state = decision["numeric_witness"][
                        "actual_terminal_state"
                    ]
                    if actual_terminal_state != target_state:
                        phase_only_frontier_cut = (
                            metric_problem["local_proxy_difficulty_mode"]
                            == "frontier_cut"
                            and target
                            == metric_problem["goal"]["node_id"]
                            and {
                                key: value
                                for key, value in actual_terminal_state.items()
                                if key != "cycle_phase"
                            }
                            == {
                                key: value
                                for key, value in target_state.items()
                                if key != "cycle_phase"
                            }
                            and actual_terminal_state["cycle_phase"] == 0
                            and target_state["cycle_phase"] == 1
                        )
                        decision = {
                            "numeric_witness": {
                                **decision["numeric_witness"],
                                "target_state": target_state,
                            },
                            "oracle_reason_code": (
                                "G2I_L_CYCLE_PHASE_FRONTIER_CUT"
                                if phase_only_frontier_cut
                                else "G2I_STATE_JOIN"
                            ),
                            "oracle_safe": False,
                            "safety_slacks": {
                                (
                                    "cycle_phase_frontier_cut"
                                    if phase_only_frontier_cut
                                    else "complete_state_join"
                                ): -1
                            },
                        }
                accepted = inside and bool(decision["oracle_safe"])
                modeled_range_identity = None
                if platform == "hopper" and inside:
                    modeled_range_identity = {
                        "hex": oracle_input["provider_exact_kinematics"][
                            "horizontal_range_m_hex"
                        ],
                        "word_hex": oracle_input[
                            "provider_exact_kinematics"
                        ]["horizontal_range_m_word_hex"],
                    }
                metric_l2_witness = _metric_l2_witness(
                    source_state,
                    target_state,
                    modeled_range_identity=modeled_range_identity,
                )
                decision_sha = domain_hash(
                    "g2-request-oracle-decision/v1",
                    canonical_json_bytes(decision),
                )
                edge = {
                    "accepted": accepted,
                    "action_id": action["action_id"],
                    "action_index": action_index,
                    "cost_milli": _metric_l2_cost_milli(
                        source_state, target_state
                    ),
                    "edge_id": f"{platform}-{x}-{y}-{action_index}",
                    "from_node": source,
                    "metric_l2_witness": metric_l2_witness,
                    "oracle_decision_sha256": decision_sha,
                    "oracle_input": oracle_input,
                    "oracle_numeric_witness": decision.get(
                        "numeric_witness", {}
                    ),
                    "oracle_safety_slacks": decision.get(
                        "safety_slacks", {}
                    ),
                    "reject_reason": (
                        None if accepted else decision["oracle_reason_code"]
                    ),
                    "safety_slack_mm": _decision_slack(decision),
                    "source_state": source_state,
                    "target_state": target_state,
                    "terrain_snapshot_sha256": snapshot_sha,
                    "to_node": target,
                }
                edges.append(edge)
    graph = {
        "action_envelope": raw_source["action_envelope"],
        "action_envelope_sha256": action_envelope_sha,
        "candidate_edges": edges,
        "expected_candidate_edge_count": len(nodes)
        * len(raw_source["action_envelope"]["actions"]),
        "expected_node_count": len(nodes),
        "graph_semantic_binding_sha256": semantic_binding_sha,
        "metric_problem": metric_problem,
        "metric_problem_sha256": metric_problem_sha,
        "node_state_root_sha256": metric_problem["node_state_root_sha256"],
        "node_states": node_states,
        "nodes": nodes,
        "platform_kind": platform,
        "profile_or_parameter_record_sha256": profile_sha,
        "provider_local_snapshot_sha256": snapshot_sha,
        "schema_version": "g2-request-finite-graph/v4",
    }
    validate_request_graph(graph)
    start = str(metric_problem["start"]["node_id"])
    goal = str(metric_problem["goal"]["node_id"])
    expected_start, expected_goal = _request_start_goal(platform)
    if (
        (start, goal) != (expected_start, expected_goal)
        or raw_source.get("start_cell_xy")
        != [int(value) for value in start.split(":")[1:]]
        or raw_source.get("goal_cell_xy")
        != [int(value) for value in goal.split(":")[1:]]
    ):
        raise ValueError("request graph endpoint binding mismatch")
    return graph, start, goal


def _difficulty(
    certificate: dict[str, Any],
    graph: dict[str, Any],
    start: str,
    goal: str,
) -> tuple[str, dict[str, Any]]:
    if not bool(certificate["oracle_reachable"]):
        return "unreachable", {
            "detour_ratio_milli": 0,
            "difficulty_basis": "exhaustive-local-frontier-cut/v1",
            "minimum_safety_slack_mm": -1,
            "path_safety_slacks": {},
            "path_primitive_count": 0,
            "platform_primitive_count": 0,
            "request_hop_count": 0,
        }
    coordinates = {
        node: tuple(int(value) for value in node.split(":")[1:])
        for node in graph["nodes"]
    }
    direct_steps = abs(coordinates[start][0] - coordinates[goal][0]) + abs(
        coordinates[start][1] - coordinates[goal][1]
    )
    path_steps = len(certificate["path_edge_ids"])
    detour_ratio = path_steps * 1000 // max(1, direct_steps)
    edge_by_id = {edge["edge_id"]: edge for edge in graph["candidate_edges"]}
    minimum_slack = min(
        int(edge_by_id[edge_id]["safety_slack_mm"])
        for edge_id in certificate["path_edge_ids"]
    )
    platform = str(graph["platform_kind"])
    difficulty_mode = str(
        graph["metric_problem"]["local_proxy_difficulty_mode"]
    )
    platform_primitive_count = path_steps * (4 if platform == "legged" else 1)
    path_slacks = {
        edge_id: edge_by_id[edge_id]["oracle_safety_slacks"]
        for edge_id in certificate["path_edge_ids"]
    }
    return (
        "hard_reachable"
        if difficulty_mode == "near_threshold" or detour_ratio >= 1500
        else "reachable"
    ), {
        "detour_ratio_milli": detour_ratio,
        "difficulty_basis": (
            f"{platform}-near-threshold-safety-witness/v1"
            if difficulty_mode == "near_threshold"
            else "direct-local-primitive/v1"
        ),
        "minimum_safety_slack_mm": minimum_slack,
        "path_safety_slacks": path_slacks,
        "path_primitive_count": path_steps,
        "platform_primitive_count": platform_primitive_count,
        "request_hop_count": path_steps,
    }


def request_graph_cache_key(
    raw_source: dict[str, Any],
    *,
    profile_record_sha256: str,
    hopper_parameter_record: dict[str, Any],
) -> str:
    platform = str(raw_source["platform_kind"])
    expected_profile = (
        sha256_bytes(canonical_json_bytes(hopper_parameter_record))
        if platform == "hopper"
        else profile_record_sha256
    )
    if platform == "hopper" and profile_record_sha256 != expected_profile:
        raise ValueError("Hopper profile hash must equal parameter record hash")
    metric = raw_source["metric_problem"]
    return domain_hash(
        "g2-request-graph-cache-key/v3",
        platform.encode("ascii"),
        _metric_binding_sha256(raw_source).encode("ascii"),
        str(raw_source["provider_local_snapshot_sha256"]).encode("ascii"),
        str(metric["terrain_transform"]["transform_sha256"]).encode("ascii"),
        str(metric["node_state_root_sha256"]).encode("ascii"),
        domain_hash(
            "g2-request-action-envelope/v2",
            canonical_json_bytes(raw_source["action_envelope"]),
        ).encode("ascii"),
        profile_record_sha256.encode("ascii"),
    )


def request_graph_from_cache(
    raw_source: dict[str, Any],
    *,
    profile_record_sha256: str,
    hopper_parameter_record: dict[str, Any],
    graph_cache: dict[str, tuple[dict[str, Any], str, str]],
    on_build: Any | None = None,
) -> tuple[dict[str, Any], str, str]:
    cache_key = request_graph_cache_key(
        raw_source,
        profile_record_sha256=profile_record_sha256,
        hopper_parameter_record=hopper_parameter_record,
    )
    cached = graph_cache.get(cache_key)
    if cached is not None:
        return cached
    if on_build is not None:
        on_build(cache_key)
    built = _graph_from_raw_source(
        raw_source,
        hopper_parameter_record=hopper_parameter_record,
        profile_record_sha256=profile_record_sha256,
    )
    graph_cache[cache_key] = built
    return built


def _request_endpoint_binding(
    graph: dict[str, Any], node_id: str
) -> dict[str, Any]:
    metric_problem = graph["metric_problem"]
    state = graph["node_states"][node_id]
    pose = (
        state["body_pose"]
        if graph["platform_kind"] == "legged"
        else state["pose"]
    )
    binding: dict[str, Any] = {
        "node_id": node_id,
        "node_state": state,
        "pose_binary64_m_rad": pose,
        "pose_mm_urad": metric_problem["node_metric_poses_mm_urad"][node_id],
    }
    for endpoint_name in ("start", "goal"):
        endpoint = metric_problem[endpoint_name]
        if endpoint["node_id"] == node_id:
            binding["endpoint_safety"] = endpoint["endpoint_safety"]
            break
    if "endpoint_safety" not in binding:
        raise ValueError("request endpoint lacks metric endpoint safety binding")
    return binding


def _request_objective() -> dict[str, Any]:
    return {
        "completion_norm": "L2",
        "distance_unit": "m",
        "distance_weight_milli": 1000,
        "kind": "distance-only-complete-L2/v1",
        "provider_goal_semantics": "direct-canonical-goal/v1",
        "resource_term_role": "reporting-only",
        "resource_weight_milli": 0,
    }


def provider_blind_request(
    truth_request: dict[str, Any],
    *,
    graph: dict[str, Any],
    hopper_parameter_record: dict[str, Any],
) -> dict[str, Any]:
    platform = str(truth_request["platform_kind"])
    if graph["platform_kind"] != platform:
        raise ValueError("provider request graph platform join mismatch")
    if (
        graph["metric_problem_sha256"]
        != truth_request["metric_problem_sha256"]
        or graph["action_envelope_sha256"]
        != truth_request["action_envelope_sha256"]
        or graph["provider_local_snapshot_sha256"]
        != truth_request["provider_local_snapshot_sha256"]
    ):
        raise ValueError("provider request graph execution join mismatch")
    core: dict[str, Any] = {
        "action_envelope": copy.deepcopy(graph["action_envelope"]),
        "action_envelope_sha256": graph["action_envelope_sha256"],
        "execution_graph": {
            "node_state_root_sha256": graph["node_state_root_sha256"],
            "node_states": copy.deepcopy(graph["node_states"]),
            "nodes": list(graph["nodes"]),
            "schema_version": "g2-provider-execution-topology/v1",
        },
        "frame_id": truth_request["frame_id"],
        "goal": copy.deepcopy(truth_request["goal"]),
        "metric_problem": copy.deepcopy(truth_request["metric_problem"]),
        "metric_problem_sha256": truth_request["metric_problem_sha256"],
        "objective": copy.deepcopy(truth_request["objective"]),
        "objective_sha256": truth_request["objective_sha256"],
        "platform_kind": platform,
        "producer_implementation_sha256": truth_request[
            "producer_implementation_sha256"
        ],
        "profile_or_parameter_record_sha256": truth_request[
            "profile_or_parameter_record_sha256"
        ],
        "provider_local_snapshot_payload_sha256": truth_request[
            "provider_local_snapshot_payload_sha256"
        ],
        "provider_local_snapshot_sha256": truth_request[
            "provider_local_snapshot_sha256"
        ],
        "provider_local_snapshot_ref": (
            "terrain/provider-local/"
            f"{truth_request['provider_local_snapshot_sha256']}.json"
        ),
        "resource_budget": copy.deepcopy(truth_request["resource_budget"]),
        "scale": truth_request["scale"],
        "schema_version": "g2-provider-blind-request/v1",
        "start": copy.deepcopy(truth_request["start"]),
        "terrain_geometry_sha256": truth_request["terrain_geometry_sha256"],
        "terrain_sha256": truth_request["terrain_sha256"],
        "vertical_datum": copy.deepcopy(
            truth_request["metric_problem"]["vertical_datum"]
        ),
    }
    if platform == "hopper":
        record = _require_hopper_parameter_record(hopper_parameter_record)
        if sha256_bytes(canonical_json_bytes(record)) != core[
            "profile_or_parameter_record_sha256"
        ]:
            raise ValueError("G2I_BLOCKED_HOPPER_PARAMETER_RECORD_MISMATCH")
        core["hopper_parameter_record"] = copy.deepcopy(record)
    provider_sha = domain_hash(
        "g2-provider-blind-request/v1",
        canonical_json_bytes(core),
    )
    return {
        **core,
        "provider_request_id": (
            f"g2i-provider-{platform}-{truth_request['scale']}-"
            f"{provider_sha[:20]}"
        ),
        "provider_request_sha256": provider_sha,
    }


def solve_raw_request_sources(
    raw_sources: list[dict[str, Any]],
    *,
    specification: dict[str, Any],
    profile_record_sha256: dict[str, str],
    producer_implementation_sha256: str,
    hopper_parameter_record: dict[str, Any] | None = None,
    graph_cache: (
        dict[str, tuple[dict[str, Any], str, str]] | None
    ) = None,
) -> list[dict[str, Any]]:
    validate_producer_specification(specification)
    if set(profile_record_sha256) != set(_PLATFORMS):
        raise ValueError("profile hash map must cover exactly three platforms")
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in profile_record_sha256.values()
    ):
        raise ValueError("invalid profile record SHA-256")
    if (
        len(producer_implementation_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in producer_implementation_sha256
        )
    ):
        raise ValueError(
            "producer implementation SHA-256 must be a lowercase SHA-256"
        )
    record = _require_hopper_parameter_record(hopper_parameter_record)
    expected_hopper_sha = sha256_bytes(canonical_json_bytes(record))
    if profile_record_sha256["hopper"] != expected_hopper_sha:
        raise ValueError("G2I_BLOCKED_HOPPER_PARAMETER_RECORD_MISMATCH")
    solved_rows: list[dict[str, Any]] = []
    active_graph_cache = {} if graph_cache is None else graph_cache
    for raw in raw_sources:
        platform = str(raw["platform_kind"])
        graph, start, goal = request_graph_from_cache(
            raw,
            profile_record_sha256=profile_record_sha256[platform],
            hopper_parameter_record=record,
            graph_cache=active_graph_cache,
        )
        terrain_geometry_source_sha = domain_hash(
            "g2-request-terrain-arrays/v1",
            canonical_json_bytes(raw["terrain_arrays"]),
        )
        certificate = reachability_certificate(graph, start, goal)
        difficulty, difficulty_witness = _difficulty(
            certificate, graph, start, goal
        )
        graph_template_sha = domain_hash(
            "g2-request-graph-template/v4",
            canonical_json_bytes(graph),
        )
        metric = graph["metric_problem"]
        snapshot_payload_sha = sha256_bytes(
            canonical_json_bytes(raw["provider_local_snapshot"])
        )
        objective = dict(specification["request_objective"])
        objective_sha = domain_hash(
            "g2-request-objective/v1",
            canonical_json_bytes(objective),
        )
        scale = str(raw["scale"])
        resource_budget = {
            "final_target_runtime_ms": 1000,
            "max_path_primitives": 128 if scale == "standard" else 512,
            "midterm_max_runtime_ms": 2000,
        }
        edge_by_id = {
            edge["edge_id"]: edge for edge in graph["candidate_edges"]
        }
        path_metric_l2_witness = [
            {
                "cost_milli": edge_by_id[edge_id]["cost_milli"],
                "edge_id": edge_id,
                "metric_l2_witness": edge_by_id[edge_id][
                    "metric_l2_witness"
                ],
            }
            for edge_id in certificate["path_edge_ids"]
        ]
        certificate_payload = {
            **certificate,
            "action_envelope_sha256": graph["action_envelope_sha256"],
            "difficulty_witness": difficulty_witness,
            "edge_source_root_sha256": domain_hash(
                "g2-request-edge-source/v2",
                canonical_json_bytes(graph["candidate_edges"]),
            ),
            "frame_id": metric["frame_id"],
            "graph_template_sha256": graph_template_sha,
            "graph_semantic_binding_sha256": graph[
                "graph_semantic_binding_sha256"
            ],
            "metric_problem_sha256": graph["metric_problem_sha256"],
            "node_state_root_sha256": graph["node_state_root_sha256"],
            "objective_sha256": objective_sha,
            "path_metric_l2_witness": path_metric_l2_witness,
            "platform_kind": raw["platform_kind"],
            "profile_or_parameter_record_sha256": profile_record_sha256[
                platform
            ],
            "provider_local_snapshot_payload_sha256": snapshot_payload_sha,
            "provider_local_snapshot_sha256": graph[
                "provider_local_snapshot_sha256"
            ],
            "raw_source_sha256": raw["raw_source_sha256"],
            "resource_budget": resource_budget,
            "scale": raw["scale"],
            "start": _request_endpoint_binding(graph, start),
            "goal": _request_endpoint_binding(graph, goal),
            "terrain_binding_sha256": metric["terrain_binding"][
                "terrain_binding_sha256"
            ],
            "terrain_geometry_source_sha256": terrain_geometry_source_sha,
            "terrain_transform_sha256": metric["terrain_transform"][
                "transform_sha256"
            ],
            "vertical_datum_reference_sha256": metric["vertical_datum"][
                "reference_sha256"
            ],
            "vertical_datum": metric["vertical_datum"],
        }
        certificate_sha = sha256_bytes(
            canonical_json_bytes(certificate_payload)
        )
        terrain_bytes = terrain_npz_bytes(
            raw["terrain_provenance"],
            terrain_arrays=raw["terrain_arrays"],
        )
        terrain_sha = sha256_bytes(terrain_bytes)
        terrain_geometry_sha = terrain_geometry_sha256(terrain_bytes)
        request_identity = {
            "action_envelope_sha256": graph["action_envelope_sha256"],
            "difficulty_class": difficulty,
            "frame_id": metric["frame_id"],
            "graph_template_sha256": graph_template_sha,
            "metric_problem_sha256": graph["metric_problem_sha256"],
            "node_state_root_sha256": graph["node_state_root_sha256"],
            "objective_sha256": objective_sha,
            "oracle_reachable": bool(certificate["oracle_reachable"]),
            "platform_primitive_count": int(
                difficulty_witness["platform_primitive_count"]
            ),
            "producer_implementation_sha256": (
                producer_implementation_sha256
            ),
            "profile_or_parameter_record_sha256": profile_record_sha256[
                platform
            ],
            "provider_local_snapshot_payload_sha256": snapshot_payload_sha,
            "provider_local_snapshot_sha256": graph[
                "provider_local_snapshot_sha256"
            ],
            "raw_source_sha256": raw["raw_source_sha256"],
            "request_hop_count": int(
                difficulty_witness["request_hop_count"]
            ),
            "resource_budget": resource_budget,
            "terrain_binding_sha256": metric["terrain_binding"][
                "terrain_binding_sha256"
            ],
            "terrain_geometry_sha256": terrain_geometry_sha,
            "terrain_sha256": terrain_sha,
            "terrain_transform_sha256": metric["terrain_transform"][
                "transform_sha256"
            ],
            "truth_certificate_sha256": certificate_sha,
            "vertical_datum_reference_sha256": metric["vertical_datum"][
                "reference_sha256"
            ],
        }
        request_sha = domain_hash(
            "g2-truth-request/v4",
            canonical_json_bytes(
                truth_request_identity(request_identity)
            ),
        )
        row = {
            "action_envelope_sha256": graph["action_envelope_sha256"],
            "determinism_seed": raw["determinism_seed"],
            "difficulty_class": difficulty,
            "frame_id": metric["frame_id"],
            "goal": _request_endpoint_binding(graph, goal),
            "graph_semantic_binding_sha256": graph[
                "graph_semantic_binding_sha256"
            ],
            "graph_template_sha256": graph_template_sha,
            "objective": objective,
            "objective_sha256": objective_sha,
            "oracle_reachable": bool(certificate["oracle_reachable"]),
            "platform_kind": platform,
            "producer_implementation_sha256": (
                producer_implementation_sha256
            ),
            "profile_or_parameter_record_sha256": profile_record_sha256[platform],
            "provider_local_snapshot_payload_sha256": snapshot_payload_sha,
            "provider_local_snapshot_sha256": graph[
                "provider_local_snapshot_sha256"
            ],
            "raw_source_sha256": raw["raw_source_sha256"],
            "request_id": f"g2i-req-{platform}-{scale}-{request_sha[:20]}",
            "resource_budget": resource_budget,
            "scale": scale,
            "schema_version": "g2-truth-request/v4",
            "selection_seed": derive_seed(
                int(specification["root_seed"]),
                platform,
                "request-selection",
                scale,
                0,
            ),
            "source_ids": [raw["raw_source_id"]],
            "start": _request_endpoint_binding(graph, start),
            "metric_problem": raw["metric_problem"],
            "metric_problem_sha256": graph["metric_problem_sha256"],
            "node_state_root_sha256": graph["node_state_root_sha256"],
            "request_hop_count": int(
                difficulty_witness["request_hop_count"]
            ),
            "platform_primitive_count": int(
                difficulty_witness["platform_primitive_count"]
            ),
            "terrain_arrays": raw["terrain_arrays"],
            "terrain_geometry_sha256": terrain_geometry_sha,
            "terrain_binding_sha256": metric["terrain_binding"][
                "terrain_binding_sha256"
            ],
            "terrain_provenance": raw["terrain_provenance"],
            "terrain_sha256": terrain_sha,
            "terrain_transform_sha256": metric["terrain_transform"][
                "transform_sha256"
            ],
            "truth_certificate": certificate_payload,
            "truth_certificate_sha256": certificate_sha,
            "truth_request_sha256": request_sha,
            "vertical_datum_reference_sha256": metric["vertical_datum"][
                "reference_sha256"
            ],
        }
        validate_truth_row(row)
        solved_rows.append(row)
    if len(solved_rows) != len(raw_sources):
        raise ValueError(
            "solved request source count mismatch: "
            f"{len(solved_rows)} != {len(raw_sources)}"
        )
    return solved_rows


def generate_request_pool(
    specification: dict[str, Any],
    *,
    profile_record_sha256: dict[str, str],
    lola_provenance: dict[str, Any],
    producer_implementation_sha256: str,
    hopper_parameter_record: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    raw_sources = generate_raw_request_sources(
        specification,
        lola_provenance=lola_provenance,
        hopper_parameter_record=hopper_parameter_record,
    )
    return solve_raw_request_sources(
        raw_sources,
        specification=specification,
        profile_record_sha256=profile_record_sha256,
        producer_implementation_sha256=producer_implementation_sha256,
        hopper_parameter_record=hopper_parameter_record,
    )


def select_requests(
    pool: list[dict[str, Any]],
    specification: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        grouped[
            (
                str(row["platform_kind"]),
                str(row["scale"]),
                str(row["difficulty_class"]),
            )
        ].append(row)
    selected: list[dict[str, Any]] = []
    for platform in _PLATFORMS:
        for scale in ("standard", "kilometer"):
            for difficulty in ("reachable", "hard_reachable", "unreachable"):
                quota = int(specification["request_mix"][scale][difficulty])
                candidates = grouped[(platform, scale, difficulty)]
                ranked = sorted(
                    candidates,
                    key=lambda row: domain_hash(
                        "g2-selection/v1",
                        str(row["selection_seed"]).encode("ascii"),
                        platform.encode("ascii"),
                        scale.encode("ascii"),
                        str(row["truth_request_sha256"]).encode("ascii"),
                    ),
                )
                if len(ranked) < quota:
                    raise ValueError(
                        f"request stratum shortfall: {platform}/{scale}/{difficulty}"
                    )
                for rank, row in enumerate(ranked[:quota]):
                    output = dict(row)
                    output["selection_rank"] = rank
                    output["selection_hash"] = domain_hash(
                        "g2-selection/v1",
                        str(row["selection_seed"]).encode("ascii"),
                        platform.encode("ascii"),
                        scale.encode("ascii"),
                        str(row["truth_request_sha256"]).encode("ascii"),
                    )
                    selected.append(output)
    return selected


def build_repeat_mapping(
    requests: Iterable[dict[str, Any]], *, input_set_id: str
) -> list[dict[str, Any]]:
    base = list(requests)
    provider_blind = bool(base) and all(
        row.get("schema_version") == "g2-provider-blind-request/v1"
        for row in base
    )
    mapping: list[dict[str, Any]] = []
    for repeat_index in range(5):
        ranked = sorted(
            base,
            key=lambda row: domain_hash(
                "g2-repeat-order/v1",
                input_set_id.encode("ascii"),
                str(row["platform_kind"]).encode("ascii"),
                str(repeat_index).encode("ascii"),
                str(
                    row[
                        "provider_request_sha256"
                        if provider_blind
                        else "truth_request_sha256"
                    ]
                ).encode("ascii"),
            ),
        )
        for submission_index, row in enumerate(ranked):
            mapping.append(
                {
                    "input_set_id": input_set_id,
                    "platform_kind": row["platform_kind"],
                    "repeat_index": repeat_index,
                    (
                        "provider_request_id"
                        if provider_blind
                        else "request_id"
                    ): row[
                        "provider_request_id"
                        if provider_blind
                        else "request_id"
                    ],
                    "submission_index": submission_index,
                    "terrain_sha256": row["terrain_sha256"],
                    (
                        "provider_request_sha256"
                        if provider_blind
                        else "truth_request_sha256"
                    ): row[
                        "provider_request_sha256"
                        if provider_blind
                        else "truth_request_sha256"
                    ],
                }
            )
    if len(mapping) != 645:
        raise ValueError(f"repeat mapping count mismatch: {len(mapping)}")
    return mapping
