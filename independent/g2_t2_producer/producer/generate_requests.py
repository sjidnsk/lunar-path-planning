from __future__ import annotations

import math
from collections import defaultdict
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
from .geometry import square_polygon
from .models import validate_truth_blind_case, validate_truth_row
from .oracle_hopper import (
    ballistic_witness,
    build_hopper_parameter_record,
    evaluate_hopper,
)
from .oracle_legged import evaluate_legged
from .oracle_wheel import evaluate_wheel, wheel_endpoint_mm


_PLATFORMS = ("wheel", "legged", "hopper")
_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _standard_height_mm(root_seed: int, base_index: int) -> list[list[int]]:
    seed = derive_seed(root_seed, "shared", "request-terrain", "standard", base_index)
    x_gradient = int(seed % 9) - 4
    y_gradient = int((seed // 11) % 9) - 4
    return [
        [
            int(seed % 1000) + x * x_gradient + y * y_gradient
            for x in range(5)
        ]
        for y in range(5)
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


def _cell_class_for_pattern(base_index: int) -> list[list[int]]:
    cell_class = [[0 for _ in range(5)] for _ in range(5)]
    topology_selector = base_index % 10
    if topology_selector in {6, 7}:
        for x in (1, 2, 3):
            cell_class[2][x] = 2
    elif topology_selector in {8, 9}:
        for y in range(5):
            cell_class[y][2] = 2
    return cell_class


def _action_envelope(platform: str) -> dict[str, Any]:
    if platform == "wheel":
        actions = [
            {
                "angular_urad_s": 0,
                "delta_cell_xy": [dx, dy],
                "duration_us": 1_000_000,
                "heading_urad": round(math.atan2(dy, dx) * 1_000_000),
                "speed_um_s": 500_000,
            }
            for dx, dy in _DIRECTIONS
        ]
    elif platform == "legged":
        actions = [
            {
                "delta_cell_xy": [dx, dy],
                "step_delta_mm": [dx * 400, dy * 400, 0],
            }
            for dx, dy in _DIRECTIONS
        ]
    else:
        azimuth_indices = (0, 8, 4, 12)
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
            for index, (dx, dy) in enumerate(_DIRECTIONS)
        ]
    return {
        "actions": actions,
        "complete": True,
        "platform_kind": platform,
        "schema_version": "g2-request-action-envelope/v1",
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


def generate_raw_request_sources(
    specification: dict[str, Any],
    *,
    lola_provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    if lola_provenance.get("physical_obstacle_cells_written") is not False:
        raise ValueError("kilometer synthetic obstacles cannot be physical")
    if (
        lola_provenance.get("micro_source_kind")
        != "synthetic_terrain_obstacle_proxy/v1"
    ):
        raise ValueError("kilometer microterrain source kind mismatch")
    root_seed = int(specification["root_seed"])
    rows: list[dict[str, Any]] = []
    for scale, count in (
        ("standard", int(specification["request_pool"]["standard_base_terrains"])),
        ("kilometer", int(specification["request_pool"]["kilometer_base_terrains"])),
    ):
        for base_index in range(count):
            if scale == "standard":
                height_mm = _standard_height_mm(root_seed, base_index)
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
                "cell_class": _cell_class_for_pattern(base_index),
                "confidence_ppm": [[1_000_000 for _ in range(5)] for _ in range(5)],
                "height_mm": height_mm,
                "known": [[1 for _ in range(5)] for _ in range(5)],
            }
            for platform in _PLATFORMS:
                envelope = _action_envelope(platform)
                source_core = {
                    "action_envelope": envelope,
                    "base_index": base_index,
                    "determinism_seed": derive_seed(
                        root_seed, platform, "requests", scale, base_index
                    ),
                    "goal_cell_xy": [4, 2],
                    "platform_kind": platform,
                    "scale": scale,
                    "start_cell_xy": [0, 2],
                    "terrain_arrays": terrain_arrays,
                    "terrain_provenance": provenance,
                }
                source_sha = domain_hash(
                    "g2-raw-request-source/v1",
                    canonical_json_bytes(source_core),
                )
                row = {
                    **source_core,
                    "raw_source_id": (
                        f"g2i-raw-{platform}-{scale}-{source_sha[:20]}"
                    ),
                    "raw_source_sha256": source_sha,
                    "schema_version": "g2-truth-blind-request-source/v1",
                }
                validate_truth_blind_case(row)
                rows.append(row)
    if len(rows) != 1056:
        raise ValueError(f"raw request source count mismatch: {len(rows)}")
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


def _blocked_polygons(
    raw_source: dict[str, Any], *, cell_size_mm: int
) -> list[list[list[int]]]:
    polygons: list[list[list[int]]] = []
    classes = raw_source["terrain_arrays"]["cell_class"]
    halfwidth = max(1, cell_size_mm // 3)
    for y in range(5):
        for x in range(5):
            if int(classes[y][x]) == 2:
                polygons.append(
                    square_polygon(x * cell_size_mm, y * cell_size_mm, halfwidth)
                )
    return polygons


def _wheel_edge_case(
    raw_source: dict[str, Any],
    source_xy: tuple[int, int],
    direction: tuple[int, int],
) -> dict[str, Any]:
    cell_size = 500
    x, y = source_xy
    dx, dy = direction
    start = [x * cell_size, y * cell_size, round(math.atan2(dy, dx) * 1_000_000)]
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
                (x + dx) * cell_size,
                (y + dy) * cell_size,
                start[2],
            ],
            "heading_bin": 0,
            "speed_um_s": 500_000,
        },
        "start_pose_mm_urad": start,
        "terrain": {
            "geometry_schema_version": "g2-wheel-request-geometry/v1",
            "height_plane": {
                "gradient_x_ppm": 0,
                "gradient_y_ppm": 0,
                "origin_x_mm": 0,
                "origin_y_mm": 0,
                "origin_z_mm": 0,
            },
            "map_bounds_mm": [-100, -100, 2100, 2100],
            "obstacle_polygons_mm": _blocked_polygons(
                raw_source, cell_size_mm=cell_size
            ),
            "unknown_polygons_mm": [],
            "untraversable_polygons_mm": [],
            "wheel_footprint_radius_mm": 100,
        },
    }
    case["primitive"]["endpoint_mm_urad"] = list(wheel_endpoint_mm(case))
    return case


def _legged_edge_case(
    raw_source: dict[str, Any],
    source_xy: tuple[int, int],
    direction: tuple[int, int],
) -> dict[str, Any]:
    cell_size = 400
    x, y = source_xy
    dx, dy = direction
    target = [(x + dx) * cell_size, (y + dy) * cell_size, 0]
    return {
        "moving_leg_id": (x + y) % 4,
        "numeric_state": "decided",
        "phase": (x + y) % 4,
        "required_phase": (x + y) % 4,
        "source_foot_mm": [x * cell_size, y * cell_size, 0],
        "target_foot_mm": target,
        "terrain": {
            "body_end_mm": [x * cell_size + 100, y * cell_size],
            "body_obstacle_polygons_mm": [],
            "body_plane": {
                "gradient_x_ppm": 0,
                "gradient_y_ppm": 0,
                "origin_x_mm": 0,
                "origin_y_mm": 0,
                "origin_z_mm": 0,
            },
            "body_radius_mm": 100,
            "body_start_mm": [x * cell_size, y * cell_size],
            "foothold_plane": {
                "gradient_x_ppm": 0,
                "gradient_y_ppm": 0,
                "origin_x_mm": 0,
                "origin_y_mm": 0,
                "origin_z_mm": 0,
            },
            "geometry_schema_version": "g2-legged-request-geometry/v1",
            "map_bounds_mm": [-100, -100, 1700, 1700],
            "obstacle_polygons_mm": _blocked_polygons(
                raw_source, cell_size_mm=cell_size
            ),
            "support_polygon_mm": square_polygon(
                x * cell_size, y * cell_size, 75
            ),
            "target_com_mm": [x * cell_size, y * cell_size],
            "unknown_polygons_mm": [],
            "untraversable_polygons_mm": [],
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
    action = _action_envelope("hopper")["actions"][direction_index]
    probe = {
        "action": {
            key: value for key, value in action.items() if key != "delta_cell_xy"
        },
        "launch_pose_mm": [0, 0, 0],
        "numeric_state": "decided",
        "terrain": {
            "height_plane": {
                "gradient_x_ppm": 0,
                "gradient_y_ppm": 0,
                "origin_x_mm": 0,
                "origin_y_mm": 0,
                "origin_z_mm": 0,
            }
        },
    }
    range_mm = int(ballistic_witness(probe, parameter_record)["range_mm"])
    launch_x = x * range_mm
    launch_y = y * range_mm
    case = {
        "action": probe["action"],
        "launch_pose_mm": [launch_x, launch_y, 0],
        "numeric_state": "decided",
        "terrain": {
            "geometry_schema_version": "g2-hopper-request-geometry/v1",
            "height_plane": probe["terrain"]["height_plane"],
            "landing_height_tolerance_mm": 50,
            "landing_pad_polygon_mm": square_polygon(0, 0, 2000),
            "landing_sigma_mm": 289,
            "landing_surface_plane": probe["terrain"]["height_plane"],
            "map_bounds_mm": [
                -500,
                -500,
                4 * range_mm + 500,
                4 * range_mm + 500,
            ],
            "obstacle_polygons_mm": [],
            "obstacle_prisms": [],
            "unknown_polygons_mm": [],
        },
    }
    witness = ballistic_witness(case, parameter_record)
    landing = (int(witness["landing_x_mm"]), int(witness["landing_y_mm"]))
    case["terrain"]["landing_pad_polygon_mm"] = square_polygon(
        landing[0], landing[1], 2000
    )
    if int(raw_source["terrain_arrays"]["cell_class"][y + dy][x + dx]) == 2:
        case["terrain"]["obstacle_polygons_mm"] = [
            square_polygon(landing[0], landing[1], 100)
        ]
    return case


def _decision_slack(decision: dict[str, Any]) -> int:
    slacks = [int(value) for value in decision.get("safety_slacks", {}).values()]
    non_domain = [value for value in slacks if value < 2**29]
    return min(non_domain, default=0)


def _graph_from_raw_source(
    raw_source: dict[str, Any],
    *,
    hopper_parameter_record: dict[str, Any],
) -> tuple[dict[str, Any], str, str]:
    platform = str(raw_source["platform_kind"])
    nodes = [f"n:{x}:{y}" for x in range(5) for y in range(5)]
    edges: list[dict[str, Any]] = []
    for x in range(5):
        for y in range(5):
            source = f"n:{x}:{y}"
            for action_index, (dx, dy) in enumerate(_DIRECTIONS):
                target_x, target_y = x + dx, y + dy
                inside = 0 <= target_x < 5 and 0 <= target_y < 5
                target = (
                    f"n:{target_x}:{target_y}" if inside else source
                )
                if platform == "wheel":
                    oracle_input = _wheel_edge_case(
                        raw_source, (x, y), (dx, dy)
                    )
                    decision = evaluate_wheel(oracle_input)
                elif platform == "legged":
                    oracle_input = _legged_edge_case(
                        raw_source, (x, y), (dx, dy)
                    )
                    decision = evaluate_legged(oracle_input)
                else:
                    if inside:
                        oracle_input = _hopper_edge_case(
                            raw_source,
                            (x, y),
                            action_index,
                            hopper_parameter_record,
                        )
                        decision = evaluate_hopper(
                            oracle_input, hopper_parameter_record
                        )
                    else:
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
                            "oracle_reason_code": "G2I_H_ARC_OUTSIDE",
                            "oracle_safe": False,
                            "safety_slacks": {"map_mm": -1},
                        }
                accepted = inside and bool(decision["oracle_safe"])
                decision_sha = domain_hash(
                    "g2-request-oracle-decision/v1",
                    canonical_json_bytes(decision),
                )
                edge = {
                    "accepted": accepted,
                    "action_index": action_index,
                    "cost_milli": 1000,
                    "edge_id": f"{platform}-{x}-{y}-{action_index}",
                    "from_node": source,
                    "oracle_decision_sha256": decision_sha,
                    "oracle_input": oracle_input,
                    "reject_reason": (
                        None if accepted else decision["oracle_reason_code"]
                    ),
                    "safety_slack_mm": _decision_slack(decision),
                    "to_node": target,
                }
                edges.append(edge)
    graph = {
        "action_envelope": raw_source["action_envelope"],
        "candidate_edges": edges,
        "expected_candidate_edge_count": 100,
        "expected_node_count": 25,
        "nodes": nodes,
        "platform_kind": platform,
        "schema_version": "g2-request-finite-graph/v2",
    }
    return graph, "n:0:2", "n:4:2"


def _difficulty(
    certificate: dict[str, Any],
    graph: dict[str, Any],
    start: str,
    goal: str,
) -> tuple[str, dict[str, int]]:
    if not bool(certificate["oracle_reachable"]):
        return "unreachable", {
            "detour_ratio_milli": 0,
            "minimum_safety_slack_mm": -1,
            "path_primitive_count": 0,
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
    return (
        "hard_reachable" if detour_ratio >= 1500 else "reachable"
    ), {
        "detour_ratio_milli": detour_ratio,
        "minimum_safety_slack_mm": minimum_slack,
        "path_primitive_count": path_steps,
    }


def solve_raw_request_sources(
    raw_sources: list[dict[str, Any]],
    *,
    specification: dict[str, Any],
    profile_record_sha256: dict[str, str],
    hopper_parameter_record: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if set(profile_record_sha256) != set(_PLATFORMS):
        raise ValueError("profile hash map must cover exactly three platforms")
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in profile_record_sha256.values()
    ):
        raise ValueError("invalid profile record SHA-256")
    record = hopper_parameter_record or build_hopper_parameter_record()
    solved_rows: list[dict[str, Any]] = []
    graph_cache: dict[
        tuple[str, str],
        tuple[dict[str, Any], str, str],
    ] = {}
    for raw in raw_sources:
        topology_sha = domain_hash(
            "g2-request-topology/v1",
            canonical_json_bytes(raw["terrain_arrays"]["cell_class"]),
        )
        cache_key = (str(raw["platform_kind"]), topology_sha)
        if cache_key not in graph_cache:
            graph_cache[cache_key] = _graph_from_raw_source(
                raw, hopper_parameter_record=record
            )
        graph, start, goal = graph_cache[cache_key]
        terrain_geometry_source_sha = domain_hash(
            "g2-request-terrain-arrays/v1",
            canonical_json_bytes(raw["terrain_arrays"]),
        )
        certificate = reachability_certificate(graph, start, goal)
        difficulty, difficulty_witness = _difficulty(
            certificate, graph, start, goal
        )
        graph_template_sha = domain_hash(
            "g2-request-graph-template/v1",
            canonical_json_bytes(graph),
        )
        certificate_payload = {
            **certificate,
            "action_envelope_sha256": domain_hash(
                "g2-request-action-envelope/v1",
                canonical_json_bytes(graph["action_envelope"]),
            ),
            "difficulty_witness": difficulty_witness,
            "edge_source_root_sha256": domain_hash(
                "g2-request-edge-source/v1",
                canonical_json_bytes(graph["candidate_edges"]),
            ),
            "graph_template_sha256": graph_template_sha,
            "platform_kind": raw["platform_kind"],
            "raw_source_sha256": raw["raw_source_sha256"],
            "scale": raw["scale"],
            "terrain_geometry_source_sha256": terrain_geometry_source_sha,
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
        objective = {
            "kind": "minimum_resource_complete_l2/v1",
            "resource_weight_milli": 1000,
        }
        platform = str(raw["platform_kind"])
        scale = str(raw["scale"])
        request_sha = domain_hash(
            "g2-truth-request/v2",
            raw["raw_source_sha256"].encode("ascii"),
            certificate_sha.encode("ascii"),
            terrain_sha.encode("ascii"),
            terrain_geometry_sha.encode("ascii"),
            profile_record_sha256[platform].encode("ascii"),
        )
        row = {
            "determinism_seed": raw["determinism_seed"],
            "difficulty_class": difficulty,
            "goal": {"node_id": goal},
            "graph_template_sha256": graph_template_sha,
            "objective": objective,
            "oracle_reachable": bool(certificate["oracle_reachable"]),
            "platform_kind": platform,
            "producer_implementation_sha256": domain_hash(
                "g2-request-producer/v2",
                b"terrain-first-platform-oracle-complete-graph",
            ),
            "profile_or_parameter_record_sha256": profile_record_sha256[platform],
            "raw_source_sha256": raw["raw_source_sha256"],
            "request_id": f"g2i-req-{platform}-{scale}-{request_sha[:20]}",
            "resource_budget": {
                "max_path_primitives": 128 if scale == "standard" else 512
            },
            "scale": scale,
            "schema_version": "g2-truth-request/v2",
            "selection_seed": derive_seed(
                int(specification["root_seed"]),
                platform,
                "request-selection",
                scale,
                0,
            ),
            "source_ids": [raw["raw_source_id"]],
            "start": {"node_id": start},
            "terrain_arrays": raw["terrain_arrays"],
            "terrain_geometry_sha256": terrain_geometry_sha,
            "terrain_provenance": raw["terrain_provenance"],
            "terrain_sha256": terrain_sha,
            "truth_certificate": certificate_payload,
            "truth_certificate_sha256": certificate_sha,
            "truth_request_sha256": request_sha,
        }
        validate_truth_row(row)
        solved_rows.append(row)
    if len(solved_rows) != 1056:
        raise ValueError(f"solved request source count mismatch: {len(solved_rows)}")
    return solved_rows


def generate_request_pool(
    specification: dict[str, Any],
    *,
    profile_record_sha256: dict[str, str],
    lola_provenance: dict[str, Any],
    hopper_parameter_record: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    raw_sources = generate_raw_request_sources(
        specification, lola_provenance=lola_provenance
    )
    return solve_raw_request_sources(
        raw_sources,
        specification=specification,
        profile_record_sha256=profile_record_sha256,
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
    mapping: list[dict[str, Any]] = []
    for repeat_index in range(5):
        ranked = sorted(
            base,
            key=lambda row: domain_hash(
                "g2-repeat-order/v1",
                input_set_id.encode("ascii"),
                str(row["platform_kind"]).encode("ascii"),
                str(repeat_index).encode("ascii"),
                str(row["truth_request_sha256"]).encode("ascii"),
            ),
        )
        for submission_index, row in enumerate(ranked):
            mapping.append(
                {
                    "input_set_id": input_set_id,
                    "platform_kind": row["platform_kind"],
                    "repeat_index": repeat_index,
                    "request_id": row["request_id"],
                    "submission_index": submission_index,
                    "terrain_sha256": row["terrain_sha256"],
                    "truth_request_sha256": row["truth_request_sha256"],
                }
            )
    if len(mapping) != 645:
        raise ValueError(f"repeat mapping count mismatch: {len(mapping)}")
    return mapping
