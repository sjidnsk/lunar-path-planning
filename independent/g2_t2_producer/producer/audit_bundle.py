from __future__ import annotations

import ast
import heapq
import io
import math
import struct
import tarfile
from collections import Counter
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import PRODUCER_REVISION
from .canonical import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    canonical_loads,
    content_index,
    content_index_root,
    domain_hash,
    sha256_bytes,
)
from .finite_graph import build_all_optima, verify_optimum_record
from .generate_cases import generate_all_cases
from .generate_requests import (
    _graph_from_raw_source,
    build_repeat_mapping,
    generate_raw_request_source_admission,
    provider_blind_request,
    request_graph_from_cache,
    select_requests,
    solve_raw_request_sources,
    terrain_npz_bytes,
)
from .models import (
    make_case_identity,
    validate_manifest,
    validate_producer_specification,
    validate_provider_blind_request,
    validate_provider_local_snapshot,
    validate_published_truth_request,
    validate_raw_source_reject,
    validate_request_certificate,
    validate_request_graph,
    validate_source_attestation,
    validate_truth_blind_case,
    validate_truth_freeze,
)
from .oracle_hopper import evaluate_hopper, validate_hopper_parameter_record
from .oracle_legged import evaluate_legged
from .oracle_wheel import evaluate_wheel


_FORBIDDEN_IMPORT_PREFIXES = (
    "path_planner",
    "lunar_exploration_ppo",
    "xunce_mid_dual_g2_inputs",
)
_EXPECTED_CARDINALITIES = {
    "primitive_labels": 10002,
    "raw_request_candidates": 1056,
    "repeat_mapping": 645,
    "requests": 129,
    "small_map_optima": 3,
}


def audit_request_graph_semantics(
    graph: dict[str, Any],
    *,
    raw_source: dict[str, Any],
    start_node: str,
    goal_node: str,
    profile_record_sha256: str,
    hopper_parameter_record: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if not isinstance(graph.get("metric_problem"), dict):
        return ["request graph metric problem binding missing"]
    try:
        validate_truth_blind_case(raw_source)
    except (KeyError, TypeError, ValueError) as error:
        reasons.append(f"truth-blind raw source invalid: {error}")
    try:
        validate_request_graph(graph)
    except (KeyError, TypeError, ValueError) as error:
        reasons.append(f"request graph exact schema invalid: {error}")
    metric = graph["metric_problem"]
    if metric != raw_source.get("metric_problem"):
        reasons.append("request graph metric problem differs from raw source")
    if graph.get("profile_or_parameter_record_sha256") != profile_record_sha256:
        reasons.append("request graph profile binding mismatch")
    if (
        raw_source.get("platform_kind") == "hopper"
        and metric.get("hopper_node_metric_poses_binary64_m_rad")
        != raw_source.get("metric_problem", {}).get(
            "hopper_node_metric_poses_binary64_m_rad"
        )
    ):
        reasons.append("request graph hopper exact endpoint binding mismatch")
    try:
        expected, expected_start, expected_goal = _graph_from_raw_source(
            raw_source,
            hopper_parameter_record=hopper_parameter_record,
            profile_record_sha256=profile_record_sha256,
        )
        if (start_node, goal_node) != (expected_start, expected_goal):
            reasons.append("request graph endpoint node binding mismatch")
        if graph != expected:
            reasons.append("request graph archived bytes differ from regeneration")
    except (KeyError, TypeError, ValueError) as error:
        reasons.append(f"request graph regeneration failed: {error}")
    reasons.extend(
        audit_archived_graph_snapshot(
            graph,
            snapshot=raw_source["provider_local_snapshot"],
            hopper_parameter_record=hopper_parameter_record,
        )
    )
    return reasons


def _audit_binary64_identity(value: float) -> dict[str, str]:
    canonical = 0.0 if value == 0.0 else float(value)
    return {
        "hex": canonical.hex(),
        "word_hex": f"{struct.unpack('>Q', struct.pack('>d', canonical))[0]:016x}",
    }


def _audit_pose_identity(
    x_m: float,
    y_m: float,
    heading_rad: float,
) -> dict[str, str]:
    x = _audit_binary64_identity(x_m)
    y = _audit_binary64_identity(y_m)
    heading = _audit_binary64_identity(heading_rad)
    return {
        "heading_rad_hex": heading["hex"],
        "heading_rad_word_hex": heading["word_hex"],
        "x_m_hex": x["hex"],
        "x_m_word_hex": x["word_hex"],
        "y_m_hex": y["hex"],
        "y_m_word_hex": y["word_hex"],
    }


def _audit_point_identity(
    x_m: float,
    y_m: float,
    z_m: float,
) -> dict[str, str]:
    x = _audit_binary64_identity(x_m)
    y = _audit_binary64_identity(y_m)
    z = _audit_binary64_identity(z_m)
    return {
        "x_m_hex": x["hex"],
        "x_m_word_hex": x["word_hex"],
        "y_m_hex": y["hex"],
        "y_m_word_hex": y["word_hex"],
        "z_m_hex": z["hex"],
        "z_m_word_hex": z["word_hex"],
    }


def _audit_cell_bbox(cells: list[list[int]]) -> list[int]:
    columns = [int(cell[0]) for cell in cells]
    rows = [int(cell[1]) for cell in cells]
    return [min(columns), min(rows), max(columns), max(rows)]


def _audit_point_segment_distance(
    point_x: float,
    point_y: float,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> float:
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    squared_length = delta_x * delta_x + delta_y * delta_y
    if squared_length == 0.0:
        return math.hypot(point_x - start_x, point_y - start_y)
    fraction = (
        (point_x - start_x) * delta_x
        + (point_y - start_y) * delta_y
    ) / squared_length
    fraction = min(1.0, max(0.0, fraction))
    closest_x = start_x + fraction * delta_x
    closest_y = start_y + fraction * delta_y
    return math.hypot(point_x - closest_x, point_y - closest_y)


def _audit_snapshot_cell_record(
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
            "slope_cdeg": 9000,
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


def _audit_orientation(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _audit_on_segment_closed(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    px: float,
    py: float,
) -> bool:
    return (
        min(ax, bx) <= px <= max(ax, bx)
        and min(ay, by) <= py <= max(ay, by)
        and _audit_orientation(ax, ay, bx, by, px, py) == 0.0
    )


def _audit_segments_intersect_closed(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> bool:
    first = _audit_orientation(ax, ay, bx, by, cx, cy)
    second = _audit_orientation(ax, ay, bx, by, dx, dy)
    third = _audit_orientation(cx, cy, dx, dy, ax, ay)
    fourth = _audit_orientation(cx, cy, dx, dy, bx, by)
    if (
        ((first > 0.0 and second < 0.0) or (first < 0.0 and second > 0.0))
        and ((third > 0.0 and fourth < 0.0) or (third < 0.0 and fourth > 0.0))
    ):
        return True
    return (
        (first == 0.0 and _audit_on_segment_closed(ax, ay, bx, by, cx, cy))
        or (second == 0.0 and _audit_on_segment_closed(ax, ay, bx, by, dx, dy))
        or (third == 0.0 and _audit_on_segment_closed(cx, cy, dx, dy, ax, ay))
        or (fourth == 0.0 and _audit_on_segment_closed(cx, cy, dx, dy, bx, by))
    )


def _audit_segment_aabb_distance_squared(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    minimum_x: float,
    minimum_y: float,
    maximum_x: float,
    maximum_y: float,
) -> float:
    if (
        minimum_x <= start_x <= maximum_x
        and minimum_y <= start_y <= maximum_y
    ) or (
        minimum_x <= end_x <= maximum_x
        and minimum_y <= end_y <= maximum_y
    ):
        return 0.0
    rectangle_edges = (
        (minimum_x, minimum_y, maximum_x, minimum_y),
        (maximum_x, minimum_y, maximum_x, maximum_y),
        (maximum_x, maximum_y, minimum_x, maximum_y),
        (minimum_x, maximum_y, minimum_x, minimum_y),
    )
    if any(
        _audit_segments_intersect_closed(
            start_x,
            start_y,
            end_x,
            end_y,
            edge_start_x,
            edge_start_y,
            edge_end_x,
            edge_end_y,
        )
        for edge_start_x, edge_start_y, edge_end_x, edge_end_y in rectangle_edges
    ):
        return 0.0
    distances = [
        _audit_point_segment_distance(
            corner_x,
            corner_y,
            start_x,
            start_y,
            end_x,
            end_y,
        )
        ** 2
        for corner_x, corner_y in (
            (minimum_x, minimum_y),
            (minimum_x, maximum_y),
            (maximum_x, minimum_y),
            (maximum_x, maximum_y),
        )
    ]
    for point_x, point_y in ((start_x, start_y), (end_x, end_y)):
        closest_x = min(max(point_x, minimum_x), maximum_x)
        closest_y = min(max(point_y, minimum_y), maximum_y)
        distances.append(
            (point_x - closest_x) ** 2 + (point_y - closest_y) ** 2
        )
    return min(distances)


def _audit_closed_interval_candidate_indices(
    minimum_coordinate: float,
    maximum_coordinate: float,
    resolution_mm: int,
    cell_count: int,
) -> range:
    first = max(
        0,
        math.ceil(float(minimum_coordinate) / resolution_mm) - 1,
    )
    last = min(
        cell_count - 1,
        math.floor(float(maximum_coordinate) / resolution_mm),
    )
    if first > last:
        return range(0)
    return range(first, last + 1)


def _audit_cells_intersecting_disk(
    *,
    center_x_mm: float,
    center_y_mm: float,
    radius_mm: int,
    resolution_mm: int,
    shape_height_width: tuple[int, int],
) -> list[list[int]]:
    radius_squared = float(radius_mm) ** 2
    height, width = shape_height_width
    cells: list[list[int]] = []
    candidate_rows = _audit_closed_interval_candidate_indices(
        center_y_mm - radius_mm,
        center_y_mm + radius_mm,
        resolution_mm,
        height,
    )
    candidate_columns = _audit_closed_interval_candidate_indices(
        center_x_mm - radius_mm,
        center_x_mm + radius_mm,
        resolution_mm,
        width,
    )
    for row in candidate_rows:
        minimum_y = row * resolution_mm
        maximum_y = minimum_y + resolution_mm
        closest_y = min(max(center_y_mm, minimum_y), maximum_y)
        for column in candidate_columns:
            minimum_x = column * resolution_mm
            maximum_x = minimum_x + resolution_mm
            closest_x = min(max(center_x_mm, minimum_x), maximum_x)
            squared_distance = (
                (closest_x - center_x_mm) ** 2
                + (closest_y - center_y_mm) ** 2
            )
            if squared_distance <= radius_squared:
                cells.append([column, row])
    return cells


def _audit_cells_intersecting_capsule(
    *,
    start_x_mm: float,
    start_y_mm: float,
    end_x_mm: float,
    end_y_mm: float,
    radius_mm: int,
    resolution_mm: int,
    shape_height_width: tuple[int, int],
) -> list[list[int]]:
    height, width = shape_height_width
    radius_squared = float(radius_mm) ** 2
    cells: list[list[int]] = []
    candidate_rows = _audit_closed_interval_candidate_indices(
        min(start_y_mm, end_y_mm) - radius_mm,
        max(start_y_mm, end_y_mm) + radius_mm,
        resolution_mm,
        height,
    )
    candidate_columns = _audit_closed_interval_candidate_indices(
        min(start_x_mm, end_x_mm) - radius_mm,
        max(start_x_mm, end_x_mm) + radius_mm,
        resolution_mm,
        width,
    )
    for row in candidate_rows:
        minimum_y = row * resolution_mm
        maximum_y = minimum_y + resolution_mm
        for column in candidate_columns:
            minimum_x = column * resolution_mm
            maximum_x = minimum_x + resolution_mm
            if (
                _audit_segment_aabb_distance_squared(
                    start_x_mm,
                    start_y_mm,
                    end_x_mm,
                    end_y_mm,
                    minimum_x,
                    minimum_y,
                    maximum_x,
                    maximum_y,
                )
                <= radius_squared
            ):
                cells.append([column, row])
    return cells


def _audit_legged_cycle_phase_frontier_cut(
    edge: dict[str, Any],
    *,
    actual_terminal_state: dict[str, Any],
    snapshot: dict[str, Any],
) -> bool:
    operations = snapshot.get("proxy_modification_witness", {}).get(
        "operations", []
    )
    matching_operations = [
        operation
        for operation in operations
        if isinstance(operation, dict)
        and operation.get("operation_kind")
        == "legged-cycle-phase-frontier-cut/v1"
    ]
    if len(matching_operations) != 1:
        return False
    operation = matching_operations[0]
    target_state = edge.get("target_state")
    if not isinstance(target_state, dict):
        return False
    actual_without_phase = {
        key: value
        for key, value in actual_terminal_state.items()
        if key != "cycle_phase"
    }
    target_without_phase = {
        key: value
        for key, value in target_state.items()
        if key != "cycle_phase"
    }
    return (
        set(operation)
        == {
            "goal_cycle_phase",
            "goal_node_id",
            "incoming_complete_cycle_phase",
            "operation_kind",
            "start_node_id",
        }
        and edge.get("from_node") == operation["start_node_id"]
        and edge.get("to_node") == operation["goal_node_id"]
        and type(operation["incoming_complete_cycle_phase"]) is int
        and type(operation["goal_cycle_phase"]) is int
        and type(actual_terminal_state.get("cycle_phase")) is int
        and type(target_state.get("cycle_phase")) is int
        and operation["incoming_complete_cycle_phase"] == 0
        and operation["goal_cycle_phase"] == 1
        and actual_terminal_state["cycle_phase"]
        == operation["incoming_complete_cycle_phase"]
        and target_state["cycle_phase"] == operation["goal_cycle_phase"]
        and actual_without_phase == target_without_phase
    )


def _independent_archived_edge_contract(
    edge: dict[str, Any],
    *,
    graph: dict[str, Any],
    snapshot: dict[str, Any],
    hopper_parameter_record: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    edge_id = str(edge["edge_id"])
    platform = str(graph["platform_kind"])
    oracle_input = edge["oracle_input"]
    if "outside_grid_source_cell_xy" in oracle_input:
        source_x, source_y = (
            int(value)
            for value in oracle_input["outside_grid_source_cell_xy"]
        )
        delta_x, delta_y = (
            int(value)
            for value in oracle_input["requested_delta_cell_xy"]
        )
        if edge["oracle_numeric_witness"].get(
            "outside_grid_target_cell_xy"
        ) != [source_x + delta_x, source_y + delta_y]:
            reasons.append(
                f"archived edge {edge_id} independent outside-domain witness mismatch"
            )
        return reasons
    if platform == "wheel":
        start_x, start_y, start_theta_urad = (
            int(value) for value in oracle_input["start_pose_mm_urad"]
        )
        primitive = oracle_input["primitive"]
        speed_mm_s = int(primitive["speed_um_s"]) / 1000.0
        omega_rad_s = int(primitive["angular_urad_s"]) / 1_000_000.0
        duration_s = int(primitive["duration_us"]) / 1_000_000.0
        theta0 = start_theta_urad / 1_000_000.0
        theta1 = theta0 + omega_rad_s * duration_s
        if abs(omega_rad_s) < 1e-15:
            endpoint_x = (
                start_x + speed_mm_s * duration_s * math.cos(theta0)
            )
            endpoint_y = (
                start_y + speed_mm_s * duration_s * math.sin(theta0)
            )
        else:
            radius_mm = speed_mm_s / omega_rad_s
            endpoint_x = start_x + radius_mm * (
                math.sin(theta1) - math.sin(theta0)
            )
            endpoint_y = start_y - radius_mm * (
                math.cos(theta1) - math.cos(theta0)
            )
        actual_endpoint = [
            round(endpoint_x),
            round(endpoint_y),
            round(theta1 * 1_000_000.0),
        ]
        expected_pose = _audit_pose_identity(
            actual_endpoint[0] / 1000.0,
            actual_endpoint[1] / 1000.0,
            actual_endpoint[2] / 1_000_000.0,
        )
        if (
            edge["oracle_numeric_witness"].get(
                "computed_endpoint_mm_urad"
            )
            != actual_endpoint
            or (
                edge["from_node"] != edge["to_node"]
                and expected_pose != edge["target_state"]["pose"]
            )
        ):
            reasons.append(
                f"archived edge {edge_id} independent Wheel endpoint mismatch"
            )
        query = oracle_input["snapshot_query_witness"]
        start = oracle_input["start_pose_mm_urad"]
        endpoint = oracle_input["primitive"]["endpoint_mm_urad"]
        expected_cells = _audit_cells_intersecting_capsule(
            start_x_mm=float(start[0]),
            start_y_mm=float(start[1]),
            end_x_mm=float(endpoint[0]),
            end_y_mm=float(endpoint[1]),
            radius_mm=int(
                oracle_input["terrain"]["wheel_footprint_radius_mm"]
            ),
            resolution_mm=int(snapshot["resolution_mm"]),
            shape_height_width=tuple(snapshot["shape_height_width"]),
        )
        if query.get("swept_cell_xy") != expected_cells:
            reasons.append(
                f"archived edge {edge_id} independent Wheel sweep-cell mismatch"
            )
        query_core = {
            key: value
            for key, value in query.items()
            if key != "query_sha256"
        }
        if query.get("query_sha256") != domain_hash(
            "g2-wheel-snapshot-query/v1",
            canonical_json_bytes(query_core),
        ):
            reasons.append(
                f"archived edge {edge_id} independent Wheel query hash mismatch"
            )
        return reasons

    if platform == "legged":
        cycle = oracle_input["crawl_cycle"]
        if len(cycle) != 4:
            return [
                f"archived edge {edge_id} independent Legged phase-count mismatch"
            ]
        cycle_contract = graph["metric_problem"]["legged_crawl_cycle"]
        expected_leg_ids = [
            "front_left",
            "front_right",
            "rear_left",
            "rear_right",
        ]
        expected_moving_legs = [
            (3, "rear_right"),
            (1, "front_right"),
            (2, "rear_left"),
            (0, "front_left"),
        ]
        expected_contract_steps = [
            {
                "forward_lateral_delta_mm": [250, 0],
                "moving_leg": leg_name,
                "moving_leg_id": leg_id,
                "phase": phase,
            }
            for phase, (leg_id, leg_name) in enumerate(
                expected_moving_legs
            )
        ]
        if (
            cycle_contract.get("foot_storage_order") != expected_leg_ids
            or cycle_contract.get("phase_steps")
            != expected_contract_steps
            or cycle_contract.get("cycle_body_delta_xy_mm") != [250, 0]
            or graph["metric_problem"].get("legged_crawl_cycle_sha256")
            != domain_hash(
                "g2-independent-static-crawl-cycle/v2",
                canonical_json_bytes(cycle_contract),
            )
        ):
            return [
                f"archived edge {edge_id} independent Legged cycle-contract mismatch"
            ]
        source_state = edge["source_state"]
        source_contacts = source_state["foot_contacts"]
        if [
            contact.get("leg_id") for contact in source_contacts
        ] != expected_leg_ids:
            reasons.append(
                f"archived edge {edge_id} independent Legged source-contact order mismatch"
            )
        current_feet = [
            [
                round(float.fromhex(contact["x_m_hex"]) * 1000.0),
                round(float.fromhex(contact["y_m_hex"]) * 1000.0),
                round(float.fromhex(contact["z_m_hex"]) * 1000.0),
            ]
            for contact in source_contacts
        ]
        if (
            cycle[0]["stance_transition"]["source_feet_mm"]
            != current_feet
        ):
            reasons.append(
                f"archived edge {edge_id} independent Legged source-state join mismatch"
            )
        for phase, step in enumerate(cycle):
            transition = step["stance_transition"]
            expected_leg_id, expected_leg_name = expected_moving_legs[phase]
            if (
                int(step["phase"]) != phase
                or int(step["required_phase"]) != phase
                or int(step["moving_leg_id"]) != expected_leg_id
                or step["moving_leg"] != expected_leg_name
                or transition["source_feet_mm"] != current_feet
            ):
                reasons.append(
                    f"archived edge {edge_id} independent Legged phase chain mismatch"
                )
                break
            moving_leg = int(step["moving_leg_id"])
            expected_feet = [list(foot) for foot in current_feet]
            expected_target_xy = [
                int(current_feet[moving_leg][0]) + 250,
                int(current_feet[moving_leg][1]),
            ]
            target_cell = [
                coordinate // int(snapshot["resolution_mm"])
                for coordinate in expected_target_xy
            ]
            expected_target_z = round(
                int(
                    _audit_snapshot_cell_record(snapshot, target_cell)[
                        "elevation_um"
                    ]
                )
                / 1000.0
            )
            expected_delta = [
                250,
                0,
                expected_target_z - int(current_feet[moving_leg][2]),
            ]
            expected_feet[moving_leg] = [
                expected_target_xy[0],
                expected_target_xy[1],
                expected_target_z,
            ]
            if (
                transition["moving_foot_delta_mm"] != expected_delta
                or step["source_foot_mm"] != current_feet[moving_leg]
                or step["target_foot_mm"] != expected_feet[moving_leg]
            ):
                reasons.append(
                    f"archived edge {edge_id} independent Legged step delta mismatch"
                )
            if transition["target_feet_mm"] != expected_feet:
                reasons.append(
                    f"archived edge {edge_id} independent Legged target feet mismatch"
                )
            current_feet = expected_feet
            query = step["snapshot_query_witness"]
            target_foot = step["target_foot_mm"]
            expected_foothold_cells = _audit_cells_intersecting_disk(
                center_x_mm=float(target_foot[0]),
                center_y_mm=float(target_foot[1]),
                radius_mm=0,
                resolution_mm=int(snapshot["resolution_mm"]),
                shape_height_width=tuple(snapshot["shape_height_width"]),
            )
            terrain = step["terrain"]
            expected_body_cells = _audit_cells_intersecting_capsule(
                start_x_mm=float(terrain["body_start_mm"][0]),
                start_y_mm=float(terrain["body_start_mm"][1]),
                end_x_mm=float(terrain["body_end_mm"][0]),
                end_y_mm=float(terrain["body_end_mm"][1]),
                radius_mm=int(terrain["body_radius_mm"]),
                resolution_mm=int(snapshot["resolution_mm"]),
                shape_height_width=tuple(snapshot["shape_height_width"]),
            )
            query_core = {
                key: value
                for key, value in query.items()
                if key != "query_sha256"
            }
            if (
                query.get("foothold_cell_xy") != expected_foothold_cells
                or query.get("body_sweep_cell_xy") != expected_body_cells
                or query.get("query_sha256")
                != domain_hash(
                    "g2-leg-snapshot-query/v1",
                    canonical_json_bytes(query_core),
                )
            ):
                reasons.append(
                    f"archived edge {edge_id} independent Legged query-cell mismatch"
                )
        terminal_body_x = (
            sum(int(foot[0]) for foot in current_feet) / 4.0 / 1000.0
        )
        terminal_body_y = (
            sum(int(foot[1]) for foot in current_feet) / 4.0 / 1000.0
        )
        expected_terminal = {
            "body_pose": _audit_pose_identity(
                terminal_body_x,
                terminal_body_y,
                float.fromhex(
                    source_state["body_pose"]["heading_rad_hex"]
                ),
            ),
            "cycle_phase": (
                int(source_state["cycle_phase"]) + len(cycle)
            )
            % 4,
            "foot_contacts": [
                {
                    "leg_id": leg_id,
                    **_audit_point_identity(
                        int(foot[0]) / 1000.0,
                        int(foot[1]) / 1000.0,
                        int(
                            _audit_snapshot_cell_record(
                                snapshot,
                                [
                                    int(foot[0])
                                    // int(snapshot["resolution_mm"]),
                                    int(foot[1])
                                    // int(snapshot["resolution_mm"]),
                                ],
                            )["elevation_um"]
                        )
                        / 1_000_000.0,
                    ),
                }
                for leg_id, foot in zip(
                    expected_leg_ids,
                    current_feet,
                    strict=True,
                )
            ],
            "schema_version": "g2-legged-node-state/v1",
        }
        archived_actual_terminal = oracle_input.get("actual_terminal_state")
        if (
            archived_actual_terminal != expected_terminal
            or edge["oracle_numeric_witness"].get(
                "actual_terminal_state"
            )
            != expected_terminal
        ):
            reasons.append(
                f"archived edge {edge_id} independent Legged terminal-state mismatch"
            )
        return reasons

    if "outside_grid_source_cell_xy" in oracle_input:
        return reasons
    hopper_binding = graph["metric_problem"]["hopper_ballistic_binding"]
    ballistic_contract = hopper_binding["ballistic_contract"]
    required = oracle_input["required_cells"]
    landing_rule = required["landing_rule"]
    terrain = oracle_input["terrain"]
    action = oracle_input["action"]
    exact = oracle_input["provider_exact_kinematics"]
    if (
        ballistic_contract.get("gravity_m_s2") != "1.62"
        or hopper_binding.get("action") != action
        or action.get("azimuth_index") != 0
        or action.get("azimuth_mdeg") != 0
        or action.get("elevation_index") != 1
        or action.get("elevation_mdeg") != 45_000
        or action.get("speed_index") != 0
        or action.get("speed_mm_s") != 1_500
        or landing_rule.get("sigma_mm") != 289
        or terrain.get("landing_sigma_mm") != 289
        or landing_rule.get("distribution_prefix_probability_ppm")
        != 990_000
        or exact.get("gravity_m_s2_hex")
        != _audit_binary64_identity(1.62)["hex"]
    ):
        reasons.append(
            f"archived edge {edge_id} independent Hopper physical-constant binding mismatch"
        )
        return reasons
    gravity_m_s2 = float(ballistic_contract["gravity_m_s2"])
    source_pose = edge["source_state"]["pose"]
    speed_m_s = int(action["speed_mm_s"]) / 1000.0
    elevation_rad = (math.pi / 6.0, math.pi / 4.0, math.pi / 3.0)[
        int(action["elevation_index"])
    ]
    horizontal_speed = speed_m_s * math.cos(elevation_rad)
    vertical_speed = speed_m_s * math.sin(elevation_rad)
    flight_time = 2.0 * (vertical_speed / gravity_m_s2)
    horizontal_delta = horizontal_speed * flight_time
    expected_endpoint = _audit_pose_identity(
        float.fromhex(source_pose["x_m_hex"]) + horizontal_delta,
        float.fromhex(source_pose["y_m_hex"]),
        float.fromhex(source_pose["heading_rad_hex"]),
    )
    if (
        exact.get("endpoint_pose_binary64_m_rad") != expected_endpoint
        or expected_endpoint != edge["target_state"]["pose"]
    ):
        reasons.append(
            f"archived edge {edge_id} independent Hopper endpoint mismatch"
        )
    launch_x, launch_y = (
        int(value) for value in oracle_input["launch_pose_mm"][:2]
    )
    expected_launch_xy = [
        round(float.fromhex(source_pose["x_m_hex"]) * 1000.0),
        round(float.fromhex(source_pose["y_m_hex"]) * 1000.0),
    ]
    if [launch_x, launch_y] != expected_launch_xy:
        reasons.append(
            f"archived edge {edge_id} independent Hopper launch-state join mismatch"
        )
    landing_x = launch_x + round(horizontal_delta * 1000.0)
    landing_y = launch_y
    launch_radius = round(
        (
            float(hopper_parameter_record["body_envelope_radius_m"])
            + float(hopper_parameter_record["arc_clearance_margin_m"])
        )
        * 1000.0
    )
    landing_footprint_radius = round(
        float(hopper_parameter_record["landing_footprint_radius_m"])
        * 1000.0
    )
    sigma_mm = int(landing_rule["sigma_mm"])
    prefix_probability = int(
        landing_rule["distribution_prefix_probability_ppm"]
    )
    prefix_radius = math.ceil(
        sigma_mm
        * math.sqrt(
            -2.0 * math.log(1.0 - prefix_probability / 1_000_000.0)
        )
    )
    landing_radius = landing_footprint_radius + prefix_radius
    shape = tuple(int(value) for value in snapshot["shape_height_width"])
    resolution = int(snapshot["resolution_mm"])
    expected_launch = _audit_cells_intersecting_disk(
        center_x_mm=float(launch_x),
        center_y_mm=float(launch_y),
        radius_mm=launch_radius,
        resolution_mm=resolution,
        shape_height_width=shape,
    )
    expected_landing = _audit_cells_intersecting_disk(
        center_x_mm=float(landing_x),
        center_y_mm=float(landing_y),
        radius_mm=landing_radius,
        resolution_mm=resolution,
        shape_height_width=shape,
    )
    expected_arc = _audit_cells_intersecting_capsule(
        start_x_mm=float(launch_x),
        start_y_mm=float(launch_y),
        end_x_mm=float(landing_x),
        end_y_mm=float(landing_y),
        radius_mm=launch_radius,
        resolution_mm=resolution,
        shape_height_width=shape,
    )
    required_core = {
        key: value
        for key, value in required.items()
        if key != "required_cells_sha256"
    }
    required_sha = domain_hash(
        "g2-hopper-required-cells/v1",
        canonical_json_bytes(required_core),
    )
    if (
        required.get("launch") != expected_launch
        or required.get("landing") != expected_landing
        or required.get("arc") != expected_arc
        or required.get("launch_bbox_cell_xy")
        != _audit_cell_bbox(expected_launch)
        or required.get("landing_bbox_cell_xy")
        != _audit_cell_bbox(expected_landing)
        or required.get("arc_bbox_cell_xy") != _audit_cell_bbox(expected_arc)
        or required.get("required_cells_sha256") != required_sha
        or required.get("H_ref_um")
        != graph["metric_problem"]["support_plane"]["H_ref_um"]
    ):
        reasons.append(
            f"archived edge {edge_id} independent Hopper required-cell mismatch"
        )
    h_ref_um = int(required["H_ref_um"])
    for group in (expected_launch, expected_landing):
        for cell_xy in group:
            record = _audit_snapshot_cell_record(snapshot, cell_xy)
            if (
                not record["known"]
                or record["hard_obstacle"]
                or not record["traversable"]
                or int(record["slope_cdeg"]) > 3000
                or abs(int(record["elevation_um"]) - h_ref_um) > 50_000
            ) and edge["accepted"]:
                reasons.append(
                    f"archived edge {edge_id} independent Hopper support-cell unsafe"
                )
                break
    for cell_xy in expected_arc:
        record = _audit_snapshot_cell_record(snapshot, cell_xy)
        if (
            not record["known"]
            or record["hard_obstacle"]
            or not record["traversable"]
            or int(record["slope_cdeg"]) > 3000
        ) and edge["accepted"]:
            reasons.append(
                f"archived edge {edge_id} independent Hopper arc-cell unsafe"
            )
            break
    return reasons


def _audit_cells_safe(
    snapshot: dict[str, Any],
    cells: list[list[int]],
) -> tuple[bool, str | None]:
    records = [_audit_snapshot_cell_record(snapshot, cell) for cell in cells]
    if not records or any(not record["known"] for record in records):
        return False, "unknown"
    if any(record["hard_obstacle"] for record in records):
        return False, "hard_obstacle"
    if any(not record["traversable"] for record in records):
        return False, "untraversable"
    if any(int(record["slope_cdeg"]) > 3000 for record in records):
        return False, "slope"
    return True, None


def _independent_edge_acceptance(
    edge: dict[str, Any],
    *,
    platform: str,
    snapshot: dict[str, Any],
) -> tuple[bool, str | None]:
    oracle_input = edge["oracle_input"]
    if "outside_grid_source_cell_xy" in oracle_input:
        source_x, source_y = (
            int(value)
            for value in oracle_input["outside_grid_source_cell_xy"]
        )
        delta_x, delta_y = (
            int(value)
            for value in oracle_input["requested_delta_cell_xy"]
        )
        expected_target = [source_x + delta_x, source_y + delta_y]
        if edge["oracle_numeric_witness"].get(
            "outside_grid_target_cell_xy"
        ) != expected_target:
            raise ValueError("outside-grid target witness mismatch")
        return False, "G2I_GRAPH_DOMAIN_OUTSIDE"

    if platform == "wheel":
        cells = oracle_input["snapshot_query_witness"]["swept_cell_xy"]
        safe, category = _audit_cells_safe(snapshot, cells)
        reason_by_category = {
            "unknown": "G2I_W_UNKNOWN_SWEEP",
            "hard_obstacle": "G2I_W_CLOSED_OBSTACLE_CONTACT",
            "untraversable": "G2I_W_NOT_TRAVERSABLE",
            "slope": "G2I_W_SLOPE_LIMIT",
        }
        return (
            (True, None)
            if safe
            else (False, reason_by_category[str(category)])
        )

    if platform == "legged":
        for step in oracle_input["crawl_cycle"]:
            query = step["snapshot_query_witness"]
            foothold_safe, foothold_category = _audit_cells_safe(
                snapshot, query["foothold_cell_xy"]
            )
            if not foothold_safe:
                reason_by_category = {
                    "unknown": "G2I_L_FOOTHOLD_UNKNOWN",
                    "hard_obstacle": "G2I_L_FOOTHOLD_OBSTACLE",
                    "untraversable": "G2I_L_FOOTHOLD_UNTRAVERSABLE",
                    "slope": "G2I_L_FOOTHOLD_SLOPE",
                }
                return False, reason_by_category[str(foothold_category)]
            body_safe, _ = _audit_cells_safe(
                snapshot, query["body_sweep_cell_xy"]
            )
            if not body_safe:
                return False, "G2I_L_BODY_SWEEP"
            if int(
                step["stance_transition"]["minimum_support_margin_um"]
            ) < 50_000:
                return False, "G2I_L_SUPPORT_MARGIN"
        actual_terminal_state = oracle_input["actual_terminal_state"]
        if actual_terminal_state != edge["target_state"]:
            if _audit_legged_cycle_phase_frontier_cut(
                edge,
                actual_terminal_state=actual_terminal_state,
                snapshot=snapshot,
            ):
                return False, "G2I_L_CYCLE_PHASE_FRONTIER_CUT"
            return False, "G2I_STATE_JOIN"
        return True, None

    required = oracle_input["required_cells"]
    groups = (
        ("launch", "G2I_H_LAUNCH"),
        ("arc", "G2I_H_ARC"),
        ("landing", "G2I_H_LANDING"),
    )
    reason_suffix = {
        "unknown": "UNKNOWN",
        "hard_obstacle": "FOOTPRINT",
        "untraversable": "NOT_TRAVERSABLE",
        "slope": "SLOPE",
    }
    for group, prefix in groups:
        safe, category = _audit_cells_safe(snapshot, required[group])
        if not safe:
            if group == "arc" and category == "hard_obstacle":
                return False, "G2I_H_ARC_CLEARANCE"
            return False, f"{prefix}_{reason_suffix[str(category)]}"
    h_ref_um = int(required["H_ref_um"])
    support_records = [
        _audit_snapshot_cell_record(snapshot, cell)
        for group in ("launch", "landing")
        for cell in required[group]
    ]
    maximum_residual_um = max(
        (
            abs(int(record["elevation_um"]) - h_ref_um)
            for record in support_records
        ),
        default=2**31,
    )
    archived_residual = int(
        oracle_input["support_height_envelope"]["maximum_residual_um"]
    )
    if archived_residual != maximum_residual_um:
        raise ValueError("Hopper support residual witness mismatch")
    if maximum_residual_um > 50_000:
        return False, "G2I_H_LANDING_HEIGHT"
    terrain = oracle_input["terrain"]
    launch_x, launch_y = (
        int(value) for value in oracle_input["launch_pose_mm"][:2]
    )
    exact_endpoint = oracle_input["provider_exact_kinematics"][
        "endpoint_pose_binary64_m_rad"
    ]
    landing_x = round(float.fromhex(exact_endpoint["x_m_hex"]) * 1000.0)
    landing_y = round(float.fromhex(exact_endpoint["y_m_hex"]) * 1000.0)
    minimum_x, minimum_y, maximum_x, maximum_y = (
        int(value) for value in terrain["map_bounds_mm"]
    )
    launch_clearance = min(
        launch_x - minimum_x,
        maximum_x - launch_x,
        launch_y - minimum_y,
        maximum_y - launch_y,
    ) - 375
    if launch_clearance <= 0:
        return False, "G2I_H_LAUNCH_FOOTPRINT"
    arc_clearance = min(
        min(launch_x, landing_x) - minimum_x,
        maximum_x - max(launch_x, landing_x),
        min(launch_y, landing_y) - minimum_y,
        maximum_y - max(launch_y, landing_y),
    ) - 500
    if arc_clearance < 0:
        return False, "G2I_H_ARC_OUTSIDE"
    pad_edge_distance = min(
        landing_x - minimum_x,
        maximum_x - landing_x,
        landing_y - minimum_y,
        maximum_y - landing_y,
    )
    landing_halfwidth = max(1, pad_edge_distance - 625)
    one_axis_mass = math.erf(
        landing_halfwidth / (289 * math.sqrt(2.0))
    )
    landing_mass_ppm = round(one_axis_mass * one_axis_mass * 1_000_000)
    if landing_mass_ppm < 990_000:
        return False, "G2I_H_LANDING_MASS"
    if pad_edge_distance - 625 <= 0:
        return False, "G2I_H_LANDING_FOOTPRINT"
    landing_plane = terrain["landing_surface_plane"]
    gradient_x = int(landing_plane["gradient_x_ppm"])
    gradient_y = int(landing_plane["gradient_y_ppm"])
    landing_slope_cdeg = round(
        math.degrees(
            math.atan(math.hypot(gradient_x, gradient_y) / 1_000_000.0)
        )
        * 100.0
    )
    if landing_slope_cdeg > 1500:
        return False, "G2I_H_LANDING_SLOPE"
    return True, None


def audit_archived_graph_snapshot(
    graph: dict[str, Any],
    *,
    snapshot: dict[str, Any],
    hopper_parameter_record: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    try:
        validate_request_graph(graph)
        validate_provider_local_snapshot(snapshot)
    except (KeyError, TypeError, ValueError) as error:
        return [f"archived graph/snapshot exact schema invalid: {error}"]
    snapshot_sha = snapshot["snapshot_sha256"]
    if graph["provider_local_snapshot_sha256"] != snapshot_sha:
        return ["archived graph/output snapshot join mismatch"]

    def check_cell_records(value: Any, edge_id: str) -> None:
        if isinstance(value, dict):
            if {
                "cell_xy",
                "hard_obstacle",
                "known",
                "slope_cdeg",
                "traversable",
            }.issubset(value):
                expected = _audit_snapshot_cell_record(
                    snapshot, value["cell_xy"]
                )
                for field, expected_value in expected.items():
                    if value.get(field) != expected_value:
                        reasons.append(
                            f"archived edge {edge_id} snapshot cell {field} drift"
                        )
                        break
            for child in value.values():
                check_cell_records(child, edge_id)
        elif isinstance(value, list):
            for child in value:
                check_cell_records(child, edge_id)

    platform = graph["platform_kind"]
    for edge in graph["candidate_edges"]:
        edge_id = str(edge["edge_id"])
        oracle_input = edge["oracle_input"]
        check_cell_records(oracle_input, edge_id)
        reasons.extend(
            _independent_archived_edge_contract(
                edge,
                graph=graph,
                snapshot=snapshot,
                hopper_parameter_record=hopper_parameter_record,
            )
        )
        try:
            accepted, expected_reason = _independent_edge_acceptance(
                edge,
                platform=str(platform),
                snapshot=snapshot,
            )
        except (KeyError, TypeError, ValueError) as error:
            reasons.append(
                f"archived edge {edge_id} independent physics failed: {error}"
            )
            continue
        if edge["accepted"] is not accepted:
            reasons.append(
                f"archived edge {edge_id} independent acceptance mismatch"
            )
        if edge["reject_reason"] != expected_reason:
            reasons.append(
                f"archived edge {edge_id} independent reason mismatch"
            )
        archived_reason = (
            {
                "wheel": "G2I_W_SAFE",
                "legged": "G2I_L_SAFE",
                "hopper": "G2I_H_SAFE",
            }[str(platform)]
            if edge["accepted"]
            else edge["reject_reason"]
        )
        archived_decision = {
            "numeric_witness": edge["oracle_numeric_witness"],
            "oracle_reason_code": archived_reason,
            "oracle_safe": bool(edge["accepted"]),
            "safety_slacks": edge["oracle_safety_slacks"],
        }
        if edge["oracle_decision_sha256"] != domain_hash(
            "g2-request-oracle-decision/v1",
            canonical_json_bytes(archived_decision),
        ):
            reasons.append(
                f"archived edge {edge_id} decision content hash mismatch"
            )
    return reasons


def audit_archived_certificate_graph(
    certificate: dict[str, Any],
    *,
    graph: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    try:
        validate_request_certificate(certificate)
        validate_request_graph(graph)
    except (KeyError, TypeError, ValueError) as error:
        return [f"archived certificate/graph exact schema invalid: {error}"]
    start = str(certificate["start"]["node_id"])
    goal = str(certificate["goal"]["node_id"])
    nodes = sorted(str(node) for node in graph["nodes"])
    adjacency: dict[str, list[dict[str, Any]]] = {
        node: [] for node in nodes
    }
    for edge in graph["candidate_edges"]:
        if bool(edge["accepted"]):
            adjacency[str(edge["from_node"])].append(edge)
    for edges in adjacency.values():
        edges.sort(
            key=lambda edge: (
                str(edge["to_node"]),
                str(edge["edge_id"]),
            )
        )
    distances = {start: 0}
    predecessors: dict[str, tuple[str, str]] = {}
    settled: set[str] = set()
    settled_order: list[str] = []
    queue: list[tuple[int, str]] = [(0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if node in settled or distance != distances.get(node):
            continue
        settled.add(node)
        settled_order.append(node)
        for edge in adjacency[node]:
            target = str(edge["to_node"])
            candidate = distance + int(edge["cost_milli"])
            if candidate < distances.get(target, 10**30):
                distances[target] = candidate
                predecessors[target] = (node, str(edge["edge_id"]))
                heapq.heappush(queue, (candidate, target))
    path_nodes: list[str] = []
    path_edges: list[str] = []
    if goal in distances:
        cursor = goal
        path_nodes = [cursor]
        while cursor != start:
            parent, edge_id = predecessors[cursor]
            path_nodes.append(parent)
            path_edges.append(edge_id)
            cursor = parent
        path_nodes.reverse()
        path_edges.reverse()
    distance_labels = {
        node: distances.get(node)
        for node in nodes
    }
    reachable_nodes = sorted(distances)
    reachable_set = set(reachable_nodes)
    frontier_cut = sorted(
        (
            {
                "edge_id": str(edge["edge_id"]),
                "from_node": str(edge["from_node"]),
                "reject_reason": str(edge["reject_reason"]),
                "to_node": str(edge["to_node"]),
            }
            for edge in graph["candidate_edges"]
            if not bool(edge["accepted"])
            and edge["from_node"] in reachable_set
            and edge["to_node"] not in reachable_set
        ),
        key=lambda edge: edge["edge_id"],
    )
    expected = {
        "cost_milli": distances.get(goal),
        "distance_labels_sha256": sha256_bytes(
            canonical_json_bytes(distance_labels)
        ),
        "frontier_cut": frontier_cut,
        "open_set_exhausted": goal not in distances,
        "oracle_reachable": goal in distances,
        "path_edge_ids": path_edges,
        "path_nodes": path_nodes,
        "reachable_nodes": reachable_nodes,
        "reachable_set_sha256": domain_hash(
            "g2-reachable-set/v1",
            canonical_json_bytes(reachable_nodes),
        ),
        "settled_order": settled_order,
    }
    for field, expected_value in expected.items():
        if certificate.get(field) != expected_value:
            reasons.append(
                f"archived certificate independent {field} mismatch"
            )
    if certificate["graph_template_sha256"] != domain_hash(
        "g2-request-graph-template/v4",
        canonical_json_bytes(graph),
    ):
        reasons.append(
            "archived certificate independent graph template hash mismatch"
        )
    if certificate["edge_source_root_sha256"] != domain_hash(
        "g2-request-edge-source/v2",
        canonical_json_bytes(graph["candidate_edges"]),
    ):
        reasons.append(
            "archived certificate independent edge source root mismatch"
        )
    for field in (
        "action_envelope_sha256",
        "graph_semantic_binding_sha256",
        "metric_problem_sha256",
        "node_state_root_sha256",
        "profile_or_parameter_record_sha256",
        "provider_local_snapshot_sha256",
    ):
        graph_field = (
            field
            if field in graph
            else "profile_or_parameter_record_sha256"
        )
        if certificate[field] != graph[graph_field]:
            reasons.append(
                f"archived certificate independent {field} graph join mismatch"
            )
    return reasons


def audit_expected_cardinalities(
    counts: dict[str, int],
) -> list[str]:
    reasons: list[str] = []
    for field, expected in _EXPECTED_CARDINALITIES.items():
        actual = counts.get(field)
        if actual != expected:
            reasons.append(
                f"{field} cardinality mismatch: expected {expected}, got {actual}"
            )
    if (
        counts.get("raw_request_pool", 0)
        + counts.get("raw_request_rejects", 0)
        != counts.get("raw_request_candidates")
    ):
        reasons.append("raw request admission conservation mismatch")
    return reasons


def audit_request_implementation_bindings(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_implementation_sha256: str,
    artifact_label: str,
) -> list[str]:
    reasons: list[str] = []
    for index, row in enumerate(rows):
        if (
            row.get("producer_implementation_sha256")
            != expected_implementation_sha256
        ):
            identifier = str(
                row.get("request_id", f"row-{index}")
            )
            reasons.append(
                f"{artifact_label} producer implementation SHA-256 "
                f"mismatch at {identifier}"
            )
    return reasons


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_bytes().splitlines(), start=1):
        if not raw_line:
            continue
        value = canonical_loads(raw_line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_number} is not an object")
        rows.append(value)
    return rows


def _audit_python_sources(
    source_items: Iterable[tuple[str, bytes]],
) -> dict[str, Any]:
    forbidden_imports: list[str] = []
    forbidden_dynamic_calls: list[str] = []
    source_files: list[dict[str, Any]] = []
    for relative, data in sorted(source_items):
        if not relative.endswith(".py") or "__pycache__" in Path(relative).parts:
            continue
        source_files.append(
            {
                "relative_path": relative,
                "sha256": sha256_bytes(data),
            }
        )
        tree = ast.parse(data.decode("utf-8"), filename=relative)
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                if module.startswith(_FORBIDDEN_IMPORT_PREFIXES):
                    forbidden_imports.append(f"{relative}:{module}")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {
                    "__import__",
                    "exec",
                    "eval",
                }:
                    forbidden_dynamic_calls.append(f"{relative}:{node.func.id}")
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "importlib"
                ):
                    forbidden_dynamic_calls.append(
                        f"{relative}:importlib.{node.func.attr}"
                    )
    forbidden_imports.sort()
    forbidden_dynamic_calls.sort()
    return {
        "forbidden_dynamic_calls": forbidden_dynamic_calls,
        "forbidden_imports": forbidden_imports,
        "passed": not forbidden_imports and not forbidden_dynamic_calls,
        "source_files": source_files,
        "static_audit_schema_version": "g2-static-isolation-audit/v1",
    }


def audit_source_tree(source_root: Path) -> dict[str, Any]:
    items = [
        (path.relative_to(source_root).as_posix(), path.read_bytes())
        for path in source_root.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    return _audit_python_sources(items)


def _source_tar_items(payload: bytes) -> dict[str, bytes]:
    items: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or member.name in items
            ):
                raise ValueError(f"unsafe producer source tar member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"unreadable producer source tar member: {member.name}")
            items[member.name] = stream.read()
    return items


def _implementation_sha_from_items(items: Mapping[str, bytes]) -> str:
    parts = [
        relative.encode("utf-8") + b"\0" + items[relative]
        for relative in sorted(items)
    ]
    return domain_hash("g2-producer-implementation/v2", *parts)


def _expected_labels(
    cases: dict[str, list[dict[str, Any]]],
    *,
    specification_sha256: str,
    implementation_sha256: str,
    profile_hashes: dict[str, str],
    hopper_parameter_record: dict[str, Any],
) -> list[dict[str, Any]]:
    evaluator = {
        "wheel": evaluate_wheel,
        "legged": evaluate_legged,
        "hopper": lambda case: evaluate_hopper(case, hopper_parameter_record),
    }
    case_envelope_sha = domain_hash(
        "g2-case-envelope/v2", specification_sha256.encode("ascii")
    )
    rows: list[dict[str, Any]] = []
    for platform in ("wheel", "legged", "hopper"):
        profile_id = (
            hopper_parameter_record["parameter_set_id"]
            if platform == "hopper"
            else f"g2-independent-{platform}-profile/v2"
        )
        for source_row in cases[platform]:
            case = source_row["case"]
            decision = evaluator[platform](case)
            case_sha, label_id = make_case_identity(platform, case)
            rows.append(
                {
                    "case_envelope_sha256": case_envelope_sha,
                    "case_sha256": case_sha,
                    "label_id": label_id,
                    "numeric_witness": decision["numeric_witness"],
                    "oracle_reason_code": decision["oracle_reason_code"],
                    "oracle_safe": decision["oracle_safe"],
                    "oracle_spec_sha256": specification_sha256,
                    "platform_kind": platform,
                    "primitive_canonical": case,
                    "primitive_sha256": domain_hash(
                        "g2-primitive/v3", canonical_json_bytes(case)
                    ),
                    "producer_id": "g2-isolated-t2-producer",
                    "producer_implementation_sha256": implementation_sha256,
                    "producer_revision": PRODUCER_REVISION,
                    "profile_or_parameter_record_sha256": profile_hashes[platform],
                    "profile_or_parameter_set_id": profile_id,
                    "safety_slacks": decision["safety_slacks"],
                    "schema_version": "g2-primitive-label/v3",
                    "terrain_sha256": case["terrain_sha256"],
                }
            )
    return rows


def _compare_bytes(
    reasons: list[str],
    *,
    path: Path,
    expected: bytes,
    label: str,
) -> None:
    try:
        actual = path.read_bytes()
    except OSError as error:
        reasons.append(f"{label} missing: {error}")
        return
    if actual != expected:
        reasons.append(f"{label} byte mismatch")


def _strong_metadata_checks(
    *,
    freeze: dict[str, Any],
    manifest: dict[str, Any],
    attestation: dict[str, Any],
    attestation_bytes: bytes,
    reasons: list[str],
) -> None:
    try:
        validate_truth_freeze(freeze)
    except (KeyError, TypeError, ValueError) as error:
        reasons.append(f"truth-freeze exact schema invalid: {error}")
    try:
        validate_manifest(manifest)
    except (KeyError, TypeError, ValueError) as error:
        reasons.append(f"manifest exact schema invalid: {error}")
    try:
        validate_source_attestation(
            attestation,
            manifest=manifest,
            expected_payload_root_sha256=str(
                freeze.get("payload_root_sha256", "")
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        reasons.append(f"source-attestation exact schema invalid: {error}")
    for name, value in (
        ("truth-freeze", freeze),
        ("manifest", manifest),
        ("source-attestations", attestation),
    ):
        if value.get("formal_evidence_eligible") is not False:
            reasons.append(f"{name} formal_evidence_eligible must be false")
    if attestation.get("status") != "pending_o2_signature":
        reasons.append("source-attestations status must remain pending_o2_signature")
    fixture_only = bool(freeze.get("fixture_only"))
    expected_independence = "fixture_exercise" if fixture_only else "T2_candidate"
    if attestation.get("technical_independence") != expected_independence:
        reasons.append("source-attestations technical_independence mismatch")
    for field in (
        "input_set_id",
        "manifest_core_sha256",
        "payload_root_sha256",
    ):
        if attestation.get(field) != freeze.get(field):
            reasons.append(f"attestation/freeze {field} mismatch")
    if sha256_bytes(attestation_bytes) != freeze.get(
        "source_attestations_sha256"
    ):
        reasons.append("source-attestations SHA-256 drift")
    if manifest.get("input_set_id") != freeze.get("input_set_id"):
        reasons.append("manifest/freeze input_set_id mismatch")
    if manifest.get("manifest_core_sha256") != freeze.get(
        "manifest_core_sha256"
    ):
        reasons.append("manifest/freeze manifest_core_sha256 mismatch")
    if manifest.get("counts") != freeze.get("counts"):
        reasons.append("manifest/freeze counts mismatch")
    if manifest.get("preflight_sha256") != freeze.get("preflight_sha256"):
        reasons.append("manifest/freeze preflight mismatch")
    if (
        manifest.get("producer_revision") != PRODUCER_REVISION
        or freeze.get("producer_revision") != PRODUCER_REVISION
    ):
        reasons.append("producer revision mismatch")


def audit_bundle(bundle_root: Path) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        freeze = canonical_loads((bundle_root / "truth-freeze.json").read_bytes())
        manifest = canonical_loads((bundle_root / "manifest.json").read_bytes())
        attestation_bytes = (bundle_root / "source-attestations.json").read_bytes()
        attestation = canonical_loads(attestation_bytes)
    except (OSError, ValueError, KeyError) as error:
        return {
            "counts": {},
            "passed": False,
            "reasons": [f"bundle metadata unreadable: {error}"],
        }
    _strong_metadata_checks(
        freeze=freeze,
        manifest=manifest,
        attestation=attestation,
        attestation_bytes=attestation_bytes,
        reasons=reasons,
    )

    try:
        indexed_paths = [Path(row["relative_path"]) for row in freeze["payload_index"]]
        recomputed_index = content_index(bundle_root, indexed_paths)
    except (OSError, KeyError, TypeError, ValueError) as error:
        reasons.append(f"payload index unreadable: {error}")
        recomputed_index = []
    if recomputed_index != freeze.get("payload_index"):
        reasons.append("payload index mismatch")
    if recomputed_index and content_index_root(recomputed_index) != freeze.get(
        "payload_root_sha256"
    ):
        reasons.append("payload root mismatch")
    actual_payload_paths = sorted(
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file()
        and path.name not in {"truth-freeze.json", "source-attestations.json"}
    )
    indexed_path_names = sorted(
        row["relative_path"] for row in freeze.get("payload_index", [])
    )
    if actual_payload_paths != indexed_path_names:
        reasons.append("payload file set mismatch")

    try:
        source_tar = (bundle_root / "source" / "producer-source.tar").read_bytes()
        source_items = _source_tar_items(source_tar)
        implementation_sha = _implementation_sha_from_items(source_items)
        source_static = _audit_python_sources(source_items.items())
        specification_bytes = source_items["SPECIFICATION.json"]
        specification = canonical_loads(specification_bytes)
        validate_producer_specification(specification)
        specification_sha = sha256_bytes(specification_bytes)
        hopper_record = canonical_loads(
            (bundle_root / "raw" / "hopper-parameter-record.json").read_bytes()
        )
        evaluator_sources = {
            "stop_condition": source_items["producer/hopper_stop_evaluator.py"],
            "energy_model": source_items["producer/hopper_energy_evaluator.py"],
        }
        validate_hopper_parameter_record(
            hopper_record, evaluator_sources=evaluator_sources
        )
    except (OSError, KeyError, ValueError, tarfile.TarError) as error:
        reasons.append(f"trusted source snapshot invalid: {error}")
        source_items = {}
        source_static = {"passed": False}
        specification = {}
        specification_sha = ""
        implementation_sha = ""
        hopper_record = {}
    if not source_static.get("passed"):
        reasons.append("trusted source snapshot static isolation failed")
    if implementation_sha != manifest.get("producer_implementation_sha256"):
        reasons.append("producer implementation SHA-256 mismatch")
    if specification_sha != manifest.get("specification_sha256"):
        reasons.append("specification SHA-256 mismatch")
    if hopper_record and sha256_bytes(
        canonical_json_bytes(hopper_record)
    ) != manifest.get("hopper_parameter_record_sha256"):
        reasons.append("Hopper parameter record SHA-256 mismatch")

    raw_hashes = manifest.get("raw_source_sha256", {})
    for relative_name, expected_hash in sorted(raw_hashes.items()):
        raw_path = bundle_root / "raw" / Path(relative_name)
        if not raw_path.is_file():
            reasons.append(f"missing raw source: {relative_name}")
        elif sha256_bytes(raw_path.read_bytes()) != expected_hash:
            reasons.append(f"raw source hash mismatch: {relative_name}")
    actual_raw_contract_paths = {
        path.relative_to(bundle_root / "raw").as_posix()
        for path in (bundle_root / "raw").rglob("*")
        if path.is_file()
        and (
            path.name == "project-authorization.md"
            or path.suffix.lower() in {".jp2", ".lbl"}
            or path.parts[-2:-1] == ("hopper",)
        )
    }
    if actual_raw_contract_paths != set(raw_hashes):
        reasons.append("raw source exact file-set mismatch")

    if reasons:
        frozen_counts = freeze.get("counts")
        return {
            "counts": (
                dict(frozen_counts)
                if isinstance(frozen_counts, dict)
                else {}
            ),
            "passed": False,
            "reasons": reasons,
        }

    try:
        lola_provenance = canonical_loads(
            (bundle_root / "raw" / "lola-provenance.json").read_bytes()
        )
        if (bundle_root / "raw" / "lola-roi-records.jsonl").is_file():
            lola_provenance["roi_records"] = _jsonl(
                bundle_root / "raw" / "lola-roi-records.jsonl"
            )
    except (OSError, ValueError) as error:
        reasons.append(f"LOLA provenance unreadable: {error}")
        lola_provenance = {}

    expected_labels: list[dict[str, Any]] = []
    expected_optima: dict[str, dict[str, Any]] = {}
    expected_raw_sources: list[dict[str, Any]] = []
    expected_raw_rejects: list[dict[str, Any]] = []
    expected_pool: list[dict[str, Any]] = []
    expected_requests: list[dict[str, Any]] = []
    expected_repeat_mapping: list[dict[str, Any]] = []
    if specification and hopper_record and lola_provenance:
        hopper_sha = sha256_bytes(canonical_json_bytes(hopper_record))
        profile_hashes = {
            "wheel": domain_hash("g2-independent-profile/v2", b"wheel"),
            "legged": domain_hash("g2-independent-profile/v2", b"legged"),
            "hopper": hopper_sha,
        }
        if profile_hashes != manifest.get("profile_or_parameter_record_sha256"):
            reasons.append("profile/parameter record hash map mismatch")
        try:
            cases = generate_all_cases(
                specification, hopper_parameter_record=hopper_record
            )
            expected_labels = _expected_labels(
                cases,
                specification_sha256=specification_sha,
                implementation_sha256=implementation_sha,
                profile_hashes=profile_hashes,
                hopper_parameter_record=hopper_record,
            )
            _compare_bytes(
                reasons,
                path=bundle_root / "raw" / "case-pool.jsonl",
                expected=canonical_jsonl_bytes(
                    row
                    for platform in ("wheel", "legged", "hopper")
                    for row in cases[platform]
                ),
                label="raw case pool regeneration",
            )
            labels_bytes = canonical_jsonl_bytes(expected_labels)
            _compare_bytes(
                reasons,
                path=bundle_root / "primitive-labels.jsonl",
                expected=labels_bytes,
                label="primitive label regeneration",
            )
            _compare_bytes(
                reasons,
                path=bundle_root / "raw" / "oracle-rows.jsonl",
                expected=labels_bytes,
                label="raw oracle row regeneration",
            )

            expected_optima = build_all_optima(
                specification, hopper_parameter_record=hopper_record
            )
            _compare_bytes(
                reasons,
                path=bundle_root / "small-map-optima.jsonl",
                expected=canonical_jsonl_bytes(
                    expected_optima[platform]["row"]
                    for platform in ("wheel", "legged", "hopper")
                ),
                label="small-map optimum regeneration",
            )
            for artifact in expected_optima.values():
                certificate = artifact["certificate"]
                certificate_sha = artifact["row"]["certificate_sha256"]
                _compare_bytes(
                    reasons,
                    path=bundle_root
                    / "certificates"
                    / f"{certificate_sha}.json",
                    expected=canonical_json_bytes(certificate),
                    label=f"small-map certificate {certificate_sha}",
                )
                if not verify_optimum_record(artifact):
                    reasons.append("regenerated optimum certificate invalid")

            expected_raw_sources, expected_raw_rejects = (
                generate_raw_request_source_admission(
                    specification,
                    lola_provenance=lola_provenance,
                    hopper_parameter_record=hopper_record,
                )
            )
            _compare_bytes(
                reasons,
                path=bundle_root / "raw" / "request-source-pool.jsonl",
                expected=canonical_jsonl_bytes(expected_raw_sources),
                label="truth-blind request source regeneration",
            )
            for reject in expected_raw_rejects:
                validate_raw_source_reject(reject)
            _compare_bytes(
                reasons,
                path=(
                    bundle_root
                    / "raw"
                    / "request-source-rejects.jsonl"
                ),
                expected=canonical_jsonl_bytes(expected_raw_rejects),
                label="raw request source reject regeneration",
            )
            expected_snapshot_entries: list[dict[str, Any]] = []
            for raw in expected_raw_sources:
                validate_truth_blind_case(raw)
                snapshot = raw["provider_local_snapshot"]
                snapshot_sha = raw["provider_local_snapshot_sha256"]
                snapshot_payload = canonical_json_bytes(snapshot)
                relative_path = (
                    f"terrain-snapshots/{snapshot_sha}.json"
                )
                expected_snapshot_entries.append(
                    {
                        "payload_sha256": sha256_bytes(snapshot_payload),
                        "relative_path": relative_path,
                        "snapshot_sha256": snapshot_sha,
                    }
                )
                _compare_bytes(
                    reasons,
                    path=bundle_root / relative_path,
                    expected=snapshot_payload,
                    label=f"provider snapshot {snapshot_sha}",
                )
                _compare_bytes(
                    reasons,
                    path=(
                        bundle_root
                        / "terrain"
                        / "provider-local"
                        / f"{snapshot_sha}.json"
                    ),
                    expected=snapshot_payload,
                    label=f"provider snapshot alias {snapshot_sha}",
                )
            expected_snapshot_entries = sorted(
                {
                    row["snapshot_sha256"]: row
                    for row in expected_snapshot_entries
                }.values(),
                key=lambda row: row["snapshot_sha256"],
            )
            expected_snapshot_root = domain_hash(
                "g2-provider-local-snapshot-index/v1",
                canonical_json_bytes(expected_snapshot_entries),
            )
            expected_snapshot_index = {
                "entries": expected_snapshot_entries,
                "index_root_sha256": expected_snapshot_root,
                "schema_version": "g2-provider-local-snapshot-index/v1",
            }
            _compare_bytes(
                reasons,
                path=bundle_root / "terrain-snapshots" / "index.json",
                expected=canonical_json_bytes(expected_snapshot_index) + b"\n",
                label="provider snapshot index",
            )
            if (
                manifest.get("provider_local_snapshot_index_root_sha256")
                != expected_snapshot_root
            ):
                reasons.append("manifest provider snapshot index root mismatch")
            regeneration_graph_cache: dict[
                str, tuple[dict[str, Any], str, str]
            ] = {}
            expected_pool = solve_raw_request_sources(
                expected_raw_sources,
                specification=specification,
                profile_record_sha256=profile_hashes,
                producer_implementation_sha256=implementation_sha,
                hopper_parameter_record=hopper_record,
                graph_cache=regeneration_graph_cache,
            )
            reasons.extend(
                audit_request_implementation_bindings(
                    expected_pool,
                    expected_implementation_sha256=implementation_sha,
                    artifact_label="regenerated raw request",
                )
            )
            _compare_bytes(
                reasons,
                path=bundle_root / "raw" / "request-pool.jsonl",
                expected=canonical_jsonl_bytes(
                    {
                        key: value
                        for key, value in row.items()
                        if key != "truth_certificate"
                    }
                    for row in expected_pool
                ),
                label="raw request pool regeneration",
            )
            expected_reject_ledger = [
                {
                    "artifact_kind": reject["artifact_kind"],
                    "artifact_sha256": reject["artifact_sha256"],
                    "reason_code": reject["reason_code"],
                }
                for reject in expected_raw_rejects
            ] + [
                {
                    "artifact_kind": "primitive_label",
                    "artifact_sha256": row["case_sha256"],
                    "reason_code": row["oracle_reason_code"],
                }
                for row in expected_labels
                if not bool(row["oracle_safe"])
            ] + [
                {
                    "artifact_kind": "request_pool",
                    "artifact_sha256": row["truth_request_sha256"],
                    "reason_code": "G2I_REQUEST_UNREACHABLE",
                }
                for row in expected_pool
                if not bool(row["oracle_reachable"])
            ]
            _compare_bytes(
                reasons,
                path=bundle_root / "reject-ledger.jsonl",
                expected=canonical_jsonl_bytes(expected_reject_ledger),
                label="reject ledger regeneration",
            )
            expected_requests = select_requests(expected_pool, specification)
            reasons.extend(
                audit_request_implementation_bindings(
                    expected_requests,
                    expected_implementation_sha256=implementation_sha,
                    artifact_label="regenerated selected request",
                )
            )
            raw_by_sha = {
                row["raw_source_sha256"]: row
                for row in expected_raw_sources
            }
            expected_provider_requests: list[dict[str, Any]] = []
            expected_sidecar: list[dict[str, Any]] = []
            for request in expected_requests:
                raw = raw_by_sha[request["raw_source_sha256"]]
                graph, _, _ = request_graph_from_cache(
                    raw,
                    profile_record_sha256=profile_hashes[
                        request["platform_kind"]
                    ],
                    hopper_parameter_record=hopper_record,
                    graph_cache=regeneration_graph_cache,
                )
                blind = provider_blind_request(
                    request,
                    graph=graph,
                    hopper_parameter_record=hopper_record,
                )
                validate_provider_blind_request(blind)
                published = {
                    key: value
                    for key, value in request.items()
                    if key not in {"truth_certificate", "terrain_arrays"}
                }
                expected_provider_requests.append(blind)
                expected_sidecar.append(
                    {
                        "provider_request_id": blind[
                            "provider_request_id"
                        ],
                        "provider_request_sha256": blind[
                            "provider_request_sha256"
                        ],
                        "schema_version": "g2-truth-request-sidecar/v1",
                        "truth_request": published,
                    }
                )
            _compare_bytes(
                reasons,
                path=bundle_root / "requests.jsonl",
                expected=canonical_jsonl_bytes(expected_provider_requests),
                label="provider-blind selected request regeneration",
            )
            _compare_bytes(
                reasons,
                path=bundle_root / "truth" / "request-sidecar.jsonl",
                expected=canonical_jsonl_bytes(expected_sidecar),
                label="truth request sidecar regeneration",
            )
            expected_repeat_mapping = build_repeat_mapping(
                expected_provider_requests,
                input_set_id=str(freeze["input_set_id"]),
            )
            _compare_bytes(
                reasons,
                path=bundle_root / "repeat-mapping.jsonl",
                expected=canonical_jsonl_bytes(expected_repeat_mapping),
                label="repeat mapping regeneration",
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            reasons.append(f"raw-to-truth regeneration failed: {error}")

    for request in expected_pool:
        try:
            validate_request_certificate(request["truth_certificate"])
        except (KeyError, TypeError, ValueError) as error:
            reasons.append(f"request certificate exact schema invalid: {error}")
        certificate_sha = request["truth_certificate_sha256"]
        _compare_bytes(
            reasons,
            path=bundle_root / "certificates" / f"{certificate_sha}.json",
            expected=canonical_json_bytes(request["truth_certificate"]),
            label=f"request certificate {certificate_sha}",
        )
        terrain_payload = terrain_npz_bytes(
            request["terrain_provenance"],
            terrain_arrays=request["terrain_arrays"],
        )
        _compare_bytes(
            reasons,
            path=bundle_root / "terrain" / f"{request['terrain_sha256']}.npz",
            expected=terrain_payload,
            label=f"request terrain {request['terrain_sha256']}",
        )

    graph_template_payloads: dict[str, bytes] = {}
    graph_cache = regeneration_graph_cache
    graph_hashes_by_platform_scale: dict[tuple[str, str], set[str]] = {}
    for raw in expected_raw_sources:
        platform = str(raw["platform_kind"])
        graph, start_node, goal_node = request_graph_from_cache(
            raw,
            profile_record_sha256=profile_hashes[platform],
            hopper_parameter_record=hopper_record,
            graph_cache=graph_cache,
        )
        graph_sha = domain_hash(
            "g2-request-graph-template/v4", canonical_json_bytes(graph)
        )
        graph_hashes_by_platform_scale.setdefault(
            (platform, str(raw["scale"])), set()
        ).add(graph_sha)
        if (start_node, goal_node) != (
            graph["metric_problem"]["start"]["node_id"],
            graph["metric_problem"]["goal"]["node_id"],
        ):
            reasons.append("request graph endpoint node binding mismatch")
        graph_template_payloads[graph_sha] = canonical_json_bytes(graph)
    for platform in ("wheel", "legged", "hopper"):
        overlap = graph_hashes_by_platform_scale.get(
            (platform, "standard"), set()
        ).intersection(
            graph_hashes_by_platform_scale.get(
                (platform, "kilometer"), set()
            )
        )
        if overlap:
            reasons.append(
                f"request graph cross-scale alias detected: {platform}"
            )
    for graph_sha, payload in graph_template_payloads.items():
        graph_path = (
            bundle_root
            / "raw"
            / "graph-templates"
            / f"{graph_sha}.json"
        )
        _compare_bytes(
            reasons,
            path=graph_path,
            expected=payload,
            label=f"request graph template {graph_sha}",
        )
        try:
            archived_graph = canonical_loads(graph_path.read_bytes())
            archived_snapshot_sha = archived_graph[
                "provider_local_snapshot_sha256"
            ]
            archived_snapshot = canonical_loads(
                (
                    bundle_root
                    / "terrain-snapshots"
                    / f"{archived_snapshot_sha}.json"
                ).read_bytes()
            )
            reasons.extend(
                audit_archived_graph_snapshot(
                    archived_graph,
                    snapshot=archived_snapshot,
                    hopper_parameter_record=hopper_record,
                )
            )
        except (OSError, KeyError, TypeError, ValueError) as error:
            reasons.append(
                f"archived graph/snapshot audit unreadable {graph_sha}: {error}"
            )
    actual_graph_templates = {
        path.stem
        for path in (bundle_root / "raw" / "graph-templates").glob("*.json")
    }
    if actual_graph_templates != set(graph_template_payloads):
        reasons.append("request graph template file-set mismatch")
    expected_graph_archive_index = [
        {
            "graph_template_sha256": graph_sha,
            "relative_path": f"raw/graph-templates/{graph_sha}.json",
            "sha256": sha256_bytes(payload),
        }
        for graph_sha, payload in sorted(graph_template_payloads.items())
    ]
    expected_graph_archive_root = domain_hash(
        "g2-request-graph-archive-root/v1",
        canonical_json_bytes(expected_graph_archive_index),
    )
    if (
        manifest.get("request_graph_archive_root_sha256")
        != expected_graph_archive_root
    ):
        reasons.append("manifest request graph archive root mismatch")
    for request in expected_pool:
        try:
            archived_certificate = canonical_loads(
                (
                    bundle_root
                    / "certificates"
                    / f"{request['truth_certificate_sha256']}.json"
                ).read_bytes()
            )
            archived_graph = canonical_loads(
                graph_template_payloads[
                    request["graph_template_sha256"]
                ]
            )
            reasons.extend(
                audit_archived_certificate_graph(
                    archived_certificate,
                    graph=archived_graph,
                )
            )
        except (OSError, KeyError, TypeError, ValueError) as error:
            reasons.append(
                "archived certificate/graph independent audit unreadable: "
                f"{error}"
            )

    try:
        labels = _jsonl(bundle_root / "primitive-labels.jsonl")
        optima = _jsonl(bundle_root / "small-map-optima.jsonl")
        requests = _jsonl(bundle_root / "requests.jsonl")
        truth_sidecar = _jsonl(
            bundle_root / "truth" / "request-sidecar.jsonl"
        )
        raw_pool = _jsonl(bundle_root / "raw" / "request-pool.jsonl")
        raw_source_rejects = _jsonl(
            bundle_root / "raw" / "request-source-rejects.jsonl"
        )
        repeat_mapping = _jsonl(bundle_root / "repeat-mapping.jsonl")
    except (OSError, ValueError) as error:
        reasons.append(f"primary row artifact unreadable: {error}")
        (
            labels,
            optima,
            requests,
            truth_sidecar,
            raw_pool,
            raw_source_rejects,
            repeat_mapping,
        ) = [], [], [], [], [], [], []
    for reject in raw_source_rejects:
        try:
            validate_raw_source_reject(reject)
        except (KeyError, TypeError, ValueError) as error:
            reasons.append(f"raw request source reject invalid: {error}")
    for request in requests:
        try:
            validate_provider_blind_request(request)
        except (KeyError, TypeError, ValueError) as error:
            reasons.append(f"provider-blind request invalid: {error}")
    sidecar_by_provider_sha: dict[str, dict[str, Any]] = {}
    for sidecar in truth_sidecar:
        try:
            if set(sidecar) != {
                "provider_request_id",
                "provider_request_sha256",
                "schema_version",
                "truth_request",
            }:
                raise ValueError("truth sidecar exact schema mismatch")
            if sidecar["schema_version"] != "g2-truth-request-sidecar/v1":
                raise ValueError("truth sidecar schema mismatch")
            validate_published_truth_request(sidecar["truth_request"])
            sidecar_by_provider_sha[
                sidecar["provider_request_sha256"]
            ] = sidecar
        except (KeyError, TypeError, ValueError) as error:
            reasons.append(f"truth request sidecar invalid: {error}")
    if len(sidecar_by_provider_sha) != len(requests):
        reasons.append("provider request/truth sidecar count mismatch")
    for request in requests:
        sidecar = sidecar_by_provider_sha.get(
            request.get("provider_request_sha256")
        )
        if (
            sidecar is None
            or sidecar.get("provider_request_id")
            != request.get("provider_request_id")
        ):
            reasons.append("provider request/truth sidecar join mismatch")
    reasons.extend(
        audit_request_implementation_bindings(
            raw_pool,
            expected_implementation_sha256=str(
                manifest.get("producer_implementation_sha256", "")
            ),
            artifact_label="raw request pool",
        )
    )
    reasons.extend(
        audit_request_implementation_bindings(
            requests,
            expected_implementation_sha256=str(
                manifest.get("producer_implementation_sha256", "")
            ),
            artifact_label="published request",
        )
    )
    counts = {
        "primitive_labels": len(labels),
        "raw_request_candidates": len(raw_pool) + len(raw_source_rejects),
        "raw_request_pool": len(raw_pool),
        "raw_request_rejects": len(raw_source_rejects),
        "repeat_mapping": len(repeat_mapping),
        "requests": len(requests),
        "small_map_optima": len(optima),
    }
    reasons.extend(audit_expected_cardinalities(counts))
    expected_admitted, expected_rejected = (
        (1056, 0) if freeze.get("fixture_only") is True else (1007, 49)
    )
    if (
        counts["raw_request_pool"],
        counts["raw_request_rejects"],
    ) != (expected_admitted, expected_rejected):
        reasons.append(
            "raw request admission count mismatch: "
            f"expected {expected_admitted} admitted + "
            f"{expected_rejected} rejected, got "
            f"{counts['raw_request_pool']} admitted + "
            f"{counts['raw_request_rejects']} rejected"
        )
    if counts != freeze.get("counts"):
        reasons.append(f"freeze count mismatch: {counts}")
    if len({row.get("label_id") for row in labels}) != len(labels):
        reasons.append("duplicate label_id")
    if len({row.get("case_sha256") for row in labels}) != len(labels):
        reasons.append("duplicate case_sha256")
    if Counter(row.get("platform_kind") for row in labels) != Counter(
        {"wheel": 3334, "legged": 3334, "hopper": 3334}
    ):
        reasons.append("primitive platform quota mismatch")
    hopper_stop = sum(
        row.get("platform_kind") == "hopper"
        and row.get("oracle_reason_code") == "G2I_H_STOP"
        for row in labels
    )
    if hopper_stop != 48:
        reasons.append(f"Hopper 3.0 m/s stop reject count mismatch: {hopper_stop}")
    if len({row.get("truth_request_sha256") for row in raw_pool}) != len(raw_pool):
        reasons.append("duplicate raw truth_request_sha256")
    if len({row.get("provider_request_id") for row in requests}) != len(requests):
        reasons.append("duplicate provider_request_id")
    if len(
        {row.get("provider_request_sha256") for row in requests}
    ) != len(requests):
        reasons.append("duplicate selected provider_request_sha256")
    sidecar_truth = [row.get("truth_request", {}) for row in truth_sidecar]
    if len(
        {row.get("truth_request_sha256") for row in sidecar_truth}
    ) != len(sidecar_truth):
        reasons.append("duplicate selected truth_request_sha256")
    if Counter(row.get("platform_kind") for row in requests) != Counter(
        {"wheel": 43, "legged": 43, "hopper": 43}
    ):
        reasons.append("selected request platform quota mismatch")
    reachable = sum(
        bool(row.get("oracle_reachable")) for row in sidecar_truth
    )
    if (reachable, len(requests) - reachable) != (114, 15):
        reasons.append("selected request reachable/unreachable quota mismatch")
    mapping_keys = {
        (
            row.get("platform_kind"),
            row.get("provider_request_id"),
            row.get("repeat_index"),
        )
        for row in repeat_mapping
    }
    if len(mapping_keys) != 645:
        reasons.append("repeat mapping unique key mismatch")
    if Counter(row.get("repeat_index") for row in repeat_mapping) != Counter(
        {index: 129 for index in range(5)}
    ):
        reasons.append("repeat mapping per-repeat quota mismatch")
    if Counter(row.get("platform_kind") for row in repeat_mapping) != Counter(
        {"wheel": 215, "legged": 215, "hopper": 215}
    ):
        reasons.append("repeat mapping per-platform quota mismatch")
    if any(
        forbidden in request
        for request in requests
        for forbidden in (
            "oracle_reachable",
            "difficulty_class",
            "truth_certificate_sha256",
            "truth_request_sha256",
        )
    ):
        reasons.append("truth field leaked into provider request")

    isolation_path = bundle_root / "audits" / "technical-isolation.json"
    try:
        isolation = canonical_loads(isolation_path.read_bytes())
        if isolation.get("passed") is not True:
            reasons.append("technical static isolation audit failed")
        if not bool(freeze.get("fixture_only")):
            if isolation.get("production_process_isolation_verified") is not True:
                reasons.append("production process isolation was not verified")
            preflight = canonical_loads(
                (bundle_root / "audits" / "preflight.json").read_bytes()
            )
            if preflight.get("passed") is not True:
                reasons.append("production preflight did not pass")
            if preflight.get("preflight_sha256") != freeze.get(
                "preflight_sha256"
            ):
                reasons.append("production preflight hash binding mismatch")
    except (OSError, ValueError) as error:
        reasons.append(f"technical isolation audit unreadable: {error}")
    return {"counts": counts, "passed": not reasons, "reasons": reasons}


def compare_bundle_bytes(first_root: Path, second_root: Path) -> dict[str, Any]:
    first_paths = sorted(
        path.relative_to(first_root).as_posix()
        for path in first_root.rglob("*")
        if path.is_file()
    )
    second_paths = sorted(
        path.relative_to(second_root).as_posix()
        for path in second_root.rglob("*")
        if path.is_file()
    )
    mismatched = sorted(set(first_paths).symmetric_difference(second_paths))
    for relative_path in sorted(set(first_paths).intersection(second_paths)):
        if (first_root / relative_path).read_bytes() != (
            second_root / relative_path
        ).read_bytes():
            mismatched.append(relative_path)
    return {
        "matched": not mismatched,
        "mismatched_paths": sorted(set(mismatched)),
        "path_count": len(set(first_paths).union(second_paths)),
    }
