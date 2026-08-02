from __future__ import annotations

import math
import struct
from typing import Any, Iterable, Mapping

from .canonical import canonical_json_bytes, domain_hash, sha256_bytes
from .geometry import relative_relief_um


_TRUTH_BLIND_FORBIDDEN = {
    "complete_l2",
    "expected_reason",
    "expected_safe",
    "oracle_reachable",
    "oracle_reason_code",
    "oracle_safe",
    "provider_safe",
    "provider_success",
    "runtime_ms",
    "target_reason",
}
_TRUTH_ROW_FORBIDDEN = {
    "cache_hit",
    "complete_l2",
    "expanded_states",
    "provider_safe",
    "provider_success",
    "runtime_ms",
    "timing_ms",
}

_SCHEMA_VERSIONS = {
    "manifest_core": "g2-truth-manifest-core/v4",
    "metric_problem": "g2-metric-planning-problem/v3",
    "producer_specification": "g2-t2-producer-spec/v3",
    "request_certificate": "g2-request-reachability-certificate/v2",
    "request_graph": "g2-request-finite-graph/v4",
    "truth_blind_request_source": "g2-truth-blind-request-source/v4",
    "truth_request": "g2-truth-request/v4",
}


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_keys(child)


def _reject_keys(value: Mapping[str, Any], forbidden: set[str]) -> None:
    for key in _walk_keys(value):
        normalized = key.casefold()
        if normalized in forbidden:
            raise ValueError(f"forbidden key: {key}")


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    name: str,
    allowed_extra: set[str] | None = None,
) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    unexpected = sorted(
        keys - required - (allowed_extra if allowed_extra is not None else set())
    )
    if missing or unexpected:
        raise ValueError(
            f"{name} exact schema mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )


def _require_sha256(value: Any, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def validate_producer_specification(value: Mapping[str, Any]) -> None:
    if (
        type(value) is not dict
        or value.get("schema_version")
        != _SCHEMA_VERSIONS["producer_specification"]
    ):
        raise ValueError("producer specification schema mismatch")
    required = {
        "authorization_sha256",
        "boundaries",
        "candidate_attestation",
        "diversity",
        "hopper_action_lattice",
        "hopper_request_ballistics",
        "legged_crawl_cycle",
        "lola_contract",
        "primitive_quotas",
        "reason_precedence",
        "request_action_envelopes",
        "request_mix",
        "request_objective",
        "request_pool",
        "root_seed",
        "schema_version",
        "small_map_envelopes",
        "source_kinds",
    }
    _require_exact_keys(value, required=required, name="producer specification")
    if value["request_objective"] != {
        "completion_norm": "L2",
        "distance_unit": "m",
        "distance_weight_milli": 1000,
        "kind": "distance-only-complete-L2/v1",
        "provider_goal_semantics": "direct-canonical-goal/v1",
        "resource_term_role": "reporting-only",
        "resource_weight_milli": 0,
    }:
        raise ValueError("producer specification request objective mismatch")
    action_envelopes = value["request_action_envelopes"]
    if any(
        action_envelopes[platform].get("action_count") != 1
        or action_envelopes[platform].get("direction_set") != ["+x"]
        for platform in ("wheel", "legged", "hopper")
    ):
        raise ValueError("producer specification request action envelope mismatch")
    canonical_json_bytes(value)


def validate_provider_local_snapshot(value: Mapping[str, Any]) -> None:
    if (
        type(value) is not dict
        or value.get("schema_version")
        != "g2-provider-local-terrain-snapshot/v1"
    ):
        raise ValueError("provider local snapshot schema mismatch")
    _require_exact_keys(
        value,
        required={
            "arrays",
            "frame_id",
            "origin_mm",
            "physical_obstacle_cells_written",
            "platform_kind",
            "proxy_modification_witness",
            "relief_preservation_sha256",
            "resolution_mm",
            "schema_version",
            "shape_height_width",
            "snapshot_sha256",
            "source_kind",
            "vertical_translation",
        },
        name="provider local snapshot",
    )
    translation = value.get("vertical_translation")
    arrays = value.get("arrays")
    if not isinstance(translation, dict) or not isinstance(arrays, dict):
        raise ValueError("provider local snapshot vertical translation missing")
    _require_exact_keys(
        arrays,
        required={
            "confidence_ppm",
            "elevation_um",
            "hard_obstacle",
            "known",
            "slope_cdeg",
            "traversable",
        },
        name="provider local snapshot arrays",
    )
    _require_exact_keys(
        translation,
        required={
            "anchor_cell_xy",
            "anchor_node_id",
            "delta_z_um",
            "input_anchor_elevation_um",
            "input_elevation_sha256",
            "input_relief_sha256",
            "input_snapshot_sha256",
            "output_elevation_sha256",
            "output_relief_sha256",
            "semantic_kind",
            "translation_sha256",
        },
        name="provider local snapshot vertical translation",
    )
    if (
        value["frame_id"] != "g2-local-metric-frame-mm/v1"
        or value["origin_mm"] != [0, 0]
        or value["resolution_mm"] != 500
        or value["shape_height_width"] != [20, 20]
        or value["platform_kind"] not in {"wheel", "legged", "hopper"}
        or value["physical_obstacle_cells_written"] is not False
    ):
        raise ValueError("provider local snapshot fixed contract mismatch")
    height, width = value["shape_height_width"]
    array_ranges = {
        "confidence_ppm": (0, 1_000_000),
        "elevation_um": (-10**12, 10**12),
        "hard_obstacle": (0, 1),
        "known": (0, 1),
        "slope_cdeg": (0, 9000),
        "traversable": (0, 1),
    }
    for name, (minimum, maximum) in array_ranges.items():
        rows = arrays[name]
        if (
            type(rows) is not list
            or len(rows) != height
            or any(type(row) is not list or len(row) != width for row in rows)
        ):
            raise ValueError(
                f"provider local snapshot {name} exact shape mismatch"
            )
        for row in rows:
            for cell in row:
                if type(cell) is not int or not minimum <= cell <= maximum:
                    raise ValueError(
                        f"provider local snapshot {name} type/range mismatch"
                    )
    witness = value["proxy_modification_witness"]
    _require_exact_keys(
        witness,
        required={
            "operations",
            "physical_obstacle_cells_written",
            "semantic_audit_sha256",
            "source_kind",
        },
        name="provider local snapshot proxy modification witness",
    )
    if (
        type(witness["operations"]) is not list
        or witness["physical_obstacle_cells_written"] is not False
        or witness["source_kind"]
        != "synthetic_terrain_obstacle_proxy/v1"
    ):
        raise ValueError("provider local snapshot proxy semantics mismatch")
    witness_core = {
        key: child
        for key, child in witness.items()
        if key != "semantic_audit_sha256"
    }
    if witness["semantic_audit_sha256"] != domain_hash(
        "g2-provider-local-proxy-semantic-audit/v1",
        canonical_json_bytes(witness_core),
        canonical_json_bytes(relative_relief_um(arrays["elevation_um"])),
        canonical_json_bytes(arrays["hard_obstacle"]),
    ):
        raise ValueError(
            "provider local snapshot proxy semantic audit mismatch"
        )
    if translation.get("semantic_kind") != (
        "uniform-vertical-translation/v1"
    ):
        raise ValueError("provider local snapshot vertical translation semantics")
    try:
        output_elevation = [
            [int(cell) for cell in row]
            for row in arrays["elevation_um"]
        ]
        delta_z_um = int(translation["delta_z_um"])
        anchor_column, anchor_row = (
            int(cell) for cell in translation["anchor_cell_xy"]
        )
        input_elevation = [
            [cell - delta_z_um for cell in row]
            for row in output_elevation
        ]
        input_anchor_um = int(input_elevation[anchor_row][anchor_column])
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "provider local snapshot vertical translation malformed"
        ) from error
    if (
        input_anchor_um != int(translation["input_anchor_elevation_um"])
        or delta_z_um != -input_anchor_um
        or int(output_elevation[anchor_row][anchor_column]) != 0
    ):
        raise ValueError(
            "provider local snapshot vertical translation anchor mismatch"
        )
    input_elevation_sha = domain_hash(
        "g2-provider-local-elevation-um/v1",
        canonical_json_bytes(input_elevation),
    )
    output_elevation_sha = domain_hash(
        "g2-provider-local-elevation-um/v1",
        canonical_json_bytes(output_elevation),
    )
    input_relief_sha = domain_hash(
        "g2-provider-local-relative-relief-um/v1",
        canonical_json_bytes(relative_relief_um(input_elevation)),
    )
    output_relief_sha = domain_hash(
        "g2-provider-local-relative-relief-um/v1",
        canonical_json_bytes(relative_relief_um(output_elevation)),
    )
    if (
        translation.get("input_elevation_sha256")
        != input_elevation_sha
        or translation.get("output_elevation_sha256")
        != output_elevation_sha
        or translation.get("input_relief_sha256") != input_relief_sha
        or translation.get("output_relief_sha256") != output_relief_sha
        or input_relief_sha != output_relief_sha
        or value.get("relief_preservation_sha256") != output_relief_sha
    ):
        raise ValueError(
            "provider local snapshot vertical translation hash mismatch"
        )
    translation_core = {
        key: child
        for key, child in translation.items()
        if key != "translation_sha256"
    }
    if translation.get("translation_sha256") != domain_hash(
        "g2-provider-uniform-vertical-translation/v1",
        canonical_json_bytes(translation_core),
    ):
        raise ValueError(
            "provider local snapshot vertical translation identity mismatch"
        )
    snapshot_core = {
        key: child for key, child in value.items() if key != "snapshot_sha256"
    }
    if value.get("snapshot_sha256") != domain_hash(
        "g2-provider-local-terrain-snapshot/v1",
        canonical_json_bytes(snapshot_core),
    ):
        raise ValueError("provider local snapshot content hash mismatch")
    canonical_json_bytes(value)


def _validate_endpoint_safety(
    value: Mapping[str, Any],
    *,
    platform: str,
    node_id: str,
    snapshot_sha256: str,
) -> None:
    _require_exact_keys(
        value,
        required={
            "cell_xy",
            "components",
            "endpoint_safety_sha256",
            "failure_reasons",
            "maximum_slope_cdeg",
            "node_id",
            "platform_kind",
            "safe",
            "schema_version",
            "snapshot_sha256",
            "support",
        },
        name="endpoint safety",
    )
    if (
        value["schema_version"] != "g2-endpoint-safety/v2"
        or value["platform_kind"] != platform
        or value["node_id"] != node_id
        or value["snapshot_sha256"] != snapshot_sha256
        or value["maximum_slope_cdeg"] != 3000
        or type(value["components"]) is not list
        or not value["components"]
        or type(value["failure_reasons"]) is not list
    ):
        raise ValueError("endpoint safety fixed contract mismatch")
    expected_component_ids = {
        "wheel": {"body"},
        "legged": {
            "body",
            "front_left",
            "front_right",
            "rear_left",
            "rear_right",
        },
        "hopper": {"launch-envelope", "landing-envelope"},
    }[platform]
    component_ids: set[str] = set()
    union_cells: set[tuple[int, int]] = set()
    for component in value["components"]:
        _require_exact_keys(
            component,
            required={
                "cell_records",
                "cell_xy",
                "component_id",
                "geometry",
                "safe",
            },
            name="endpoint safety component",
        )
        component_id = str(component["component_id"])
        component_ids.add(component_id)
        geometry = component["geometry"]
        _require_exact_keys(
            geometry,
            required={
                "boundary_semantics",
                "center_mm",
                "kind",
                "radius_mm",
            },
            name="endpoint safety component geometry",
        )
        if (
            geometry["boundary_semantics"]
            != "closed-cell-intersection/v1"
            or geometry["kind"] != "disk/v1"
            or type(geometry["center_mm"]) is not list
            or len(geometry["center_mm"]) != 2
            or type(geometry["radius_mm"]) is not int
            or geometry["radius_mm"] < 0
        ):
            raise ValueError("endpoint safety component geometry mismatch")
        if platform == "hopper":
            expected_radius_mm = {
                "launch-envelope": 500,
                "landing-envelope": 625,
            }.get(component_id)
            if geometry["radius_mm"] != expected_radius_mm:
                raise ValueError(
                    "endpoint safety Hopper authority footprint mismatch"
                )
        cell_xy = component["cell_xy"]
        records = component["cell_records"]
        if (
            type(cell_xy) is not list
            or not cell_xy
            or type(records) is not list
            or len(records) != len(cell_xy)
        ):
            raise ValueError("endpoint safety component cell list mismatch")
        independently_safe = True
        for cell, record in zip(cell_xy, records, strict=True):
            if (
                type(cell) is not list
                or len(cell) != 2
                or any(type(coordinate) is not int for coordinate in cell)
            ):
                raise ValueError("endpoint safety cell coordinate mismatch")
            _require_exact_keys(
                record,
                required={
                    "cell_xy",
                    "confidence_ppm",
                    "elevation_um",
                    "hard_obstacle",
                    "known",
                    "out_of_bounds",
                    "slope_cdeg",
                    "traversable",
                },
                name="endpoint safety cell record",
            )
            if record["cell_xy"] != cell or record["out_of_bounds"] is not False:
                raise ValueError("endpoint safety cell record join mismatch")
            independently_safe = independently_safe and (
                record["known"] is True
                and record["hard_obstacle"] is False
                and record["traversable"] is True
                and type(record["slope_cdeg"]) is int
                and record["slope_cdeg"] <= 3000
            )
            union_cells.add((cell[0], cell[1]))
        if component["safe"] is not independently_safe:
            raise ValueError("endpoint safety component summary mismatch")
    if platform == "hopper":
        if len(component_ids) != 1 or not component_ids <= expected_component_ids:
            raise ValueError("endpoint safety Hopper component mismatch")
    elif component_ids != expected_component_ids:
        raise ValueError("endpoint safety component set mismatch")
    if value["cell_xy"] != [list(cell) for cell in sorted(union_cells)]:
        raise ValueError("endpoint safety union cell mismatch")
    support = value["support"]
    if platform == "legged":
        _require_exact_keys(
            support,
            required={
                "cycle_phase",
                "kind",
                "minimum_required_margin_mm",
                "safe",
                "support_margin_mm",
                "support_polygon_mm",
            },
            name="endpoint safety legged support",
        )
        support_safe = (
            support["kind"] == "four-contact-convex-support/v1"
            and type(support["cycle_phase"]) is int
            and 0 <= support["cycle_phase"] < 4
            and support["support_margin_mm"]
            >= support["minimum_required_margin_mm"]
            and support["minimum_required_margin_mm"] == 50
        )
    elif platform == "hopper":
        _require_exact_keys(
            support,
            required={
                "kind",
                "maximum_abs_residual_um",
                "maximum_tolerated_residual_um",
                "safe",
            },
            name="endpoint safety Hopper support",
        )
        support_safe = (
            support["kind"]
            in {"launch-support-envelope/v1", "landing-support-envelope/v1"}
            and type(support["maximum_abs_residual_um"]) is int
            and support["maximum_tolerated_residual_um"] == 50_000
            and support["maximum_abs_residual_um"]
            <= support["maximum_tolerated_residual_um"]
        )
    else:
        _require_exact_keys(
            support,
            required={"kind", "safe"},
            name="endpoint safety wheel support",
        )
        support_safe = support["kind"] == "single-body-footprint/v1"
    if support["safe"] is not support_safe:
        raise ValueError("endpoint safety support summary mismatch")
    safe = all(component["safe"] for component in value["components"]) and bool(
        support["safe"]
    )
    expected_failures = [] if safe else [
        *(
            ["G2I_ENDPOINT_COMPONENT_UNSAFE"]
            if any(not component["safe"] for component in value["components"])
            else []
        ),
        *([] if support["safe"] else ["G2I_ENDPOINT_SUPPORT_UNSAFE"]),
    ]
    if (
        value["safe"] is not safe
        or value["failure_reasons"] != expected_failures
    ):
        raise ValueError("endpoint safety aggregate summary mismatch")
    core = {
        key: child
        for key, child in value.items()
        if key != "endpoint_safety_sha256"
    }
    if value["endpoint_safety_sha256"] != domain_hash(
        "g2-endpoint-safety/v2",
        canonical_json_bytes(core),
    ):
        raise ValueError("endpoint safety identity mismatch")


def validate_metric_problem(value: Mapping[str, Any]) -> None:
    if (
        type(value) is not dict
        or value.get("schema_version") != _SCHEMA_VERSIONS["metric_problem"]
    ):
        raise ValueError("metric problem schema mismatch")
    states = value.get("node_states")
    if not isinstance(states, dict) or not states:
        raise ValueError("metric problem node states missing")
    state_schemas = {
        str(state.get("schema_version"))
        for state in states.values()
        if isinstance(state, dict)
    }
    platform_extra: set[str]
    if state_schemas == {"g2-wheel-node-state/v1"}:
        platform_extra = set()
    elif state_schemas == {"g2-legged-node-state/v1"}:
        platform_extra = {
            "legged_crawl_cycle",
            "legged_crawl_cycle_sha256",
        }
    elif state_schemas == {"g2-hopper-node-state/v1"}:
        platform_extra = {
            "hopper_ballistic_binding",
            "hopper_node_metric_poses_binary64_m_rad",
            "relief_preservation_sha256",
            "support_plane",
        }
    else:
        raise ValueError("metric problem node state platform mismatch")
    if state_schemas == {"g2-legged-node-state/v1"}:
        for node_id, state in states.items():
            cycle_phase = state.get("cycle_phase")
            if type(cycle_phase) is not int or not 0 <= cycle_phase < 4:
                raise ValueError(
                    f"metric Legged node {node_id} cycle phase mismatch"
                )
    required = {
        "canonical_pose_unit",
        "endpoint_safety_margin_mm",
        "frame_id",
        "goal",
        "heading_unit",
        "height_geometry_sha256",
        "horizontal_unit",
        "local_height_plane",
        "local_height_plane_sha256",
        "local_proxy_difficulty_mode",
        "local_proxy_geometry_sha256",
        "local_proxy_source_kind",
        "macro_cell_size_mm",
        "node_metric_poses_mm_urad",
        "node_origin_contract",
        "node_state_root_sha256",
        "node_states",
        "origin_mm",
        "provider_fine_resolution_mm",
        "provider_local_planning_region",
        "provider_local_snapshot_sha256",
        "scale",
        "schema_version",
        "source_extent_bounds_mm",
        "source_to_local_transform",
        "start",
        "terrain_arrays_sha256",
        "terrain_binding",
        "terrain_bounds_mm",
        "terrain_shape_height_width",
        "terrain_transform",
        "terrain_transform_semantics",
        "vertical_datum",
        *platform_extra,
    }
    _require_exact_keys(value, required=required, name="metric problem")
    if value["horizontal_unit"] != "mm" or value["heading_unit"] != "urad":
        raise ValueError("metric problem horizontal or heading unit mismatch")
    if value["vertical_datum"].get("vertical_unit") != "um":
        raise ValueError("metric problem vertical datum unit mismatch")
    node_ids = set(value["node_metric_poses_mm_urad"])
    if node_ids != set(states):
        raise ValueError("metric problem node pose/state set mismatch")
    node_state_root = domain_hash(
        "g2-request-node-state-root/v1",
        canonical_json_bytes(states),
    )
    if value["node_state_root_sha256"] != node_state_root:
        raise ValueError("metric problem node state root mismatch")
    snapshot_sha = _require_sha256(
        value["provider_local_snapshot_sha256"],
        name="metric problem output snapshot",
    )
    transform = value["terrain_transform"]
    _require_exact_keys(
        transform,
        required={
            "delta_z_um",
            "input_elevation_sha256",
            "input_relief_sha256",
            "input_snapshot_sha256",
            "output_elevation_sha256",
            "output_relief_sha256",
            "output_snapshot_sha256",
            "semantic_kind",
            "transform_id",
            "transform_sha256",
            "translation_sha256",
        },
        name="metric terrain transform",
    )
    if (
        transform["semantic_kind"] != "uniform-vertical-translation/v1"
        or transform["output_snapshot_sha256"] != snapshot_sha
        or transform["input_relief_sha256"]
        != transform["output_relief_sha256"]
    ):
        raise ValueError("metric terrain transform join mismatch")
    transform_core = {
        key: child
        for key, child in transform.items()
        if key != "transform_sha256"
    }
    if transform["transform_sha256"] != domain_hash(
        "g2-provider-terrain-transform/v1",
        canonical_json_bytes(transform_core),
    ):
        raise ValueError("metric terrain transform hash mismatch")
    binding = value["terrain_binding"]
    if (
        binding.get("snapshot_sha256") != snapshot_sha
        or binding.get("provider_transform") != transform
        or binding.get("relief_preservation_sha256")
        != transform["output_relief_sha256"]
    ):
        raise ValueError("metric terrain binding join mismatch")
    binding_core = {
        key: child
        for key, child in binding.items()
        if key != "terrain_binding_sha256"
    }
    if binding.get("terrain_binding_sha256") != domain_hash(
        "g2-provider-terrain-binding/v1",
        canonical_json_bytes(binding_core),
    ):
        raise ValueError("metric terrain binding hash mismatch")
    for endpoint_name in ("start", "goal"):
        endpoint = value[endpoint_name]
        _require_exact_keys(
            endpoint,
            required={
                "endpoint_safety",
                "node_id",
                "pose_binary64_m_rad",
                "pose_mm_urad",
            },
            name=f"metric {endpoint_name} endpoint",
        )
        node_id = endpoint["node_id"]
        if (
            node_id not in states
            or endpoint["pose_mm_urad"]
            != value["node_metric_poses_mm_urad"][node_id]
            or endpoint["endpoint_safety"].get("snapshot_sha256")
            != snapshot_sha
        ):
            raise ValueError(f"metric {endpoint_name} endpoint join mismatch")
        platform = {
            "g2-wheel-node-state/v1": "wheel",
            "g2-legged-node-state/v1": "legged",
            "g2-hopper-node-state/v1": "hopper",
        }[next(iter(state_schemas))]
        _validate_endpoint_safety(
            endpoint["endpoint_safety"],
            platform=platform,
            node_id=node_id,
            snapshot_sha256=snapshot_sha,
        )
        state_pose = (
            states[node_id]["body_pose"]
            if state_schemas == {"g2-legged-node-state/v1"}
            else states[node_id]["pose"]
        )
        if endpoint["pose_binary64_m_rad"] != state_pose:
            raise ValueError(f"metric {endpoint_name} pose join mismatch")
    canonical_json_bytes(value)


def validate_request_graph(value: Mapping[str, Any]) -> None:
    if (
        type(value) is not dict
        or value.get("schema_version") != _SCHEMA_VERSIONS["request_graph"]
    ):
        raise ValueError("request graph schema mismatch")
    _require_exact_keys(
        value,
        required={
            "action_envelope",
            "action_envelope_sha256",
            "candidate_edges",
            "expected_candidate_edge_count",
            "expected_node_count",
            "graph_semantic_binding_sha256",
            "metric_problem",
            "metric_problem_sha256",
            "node_state_root_sha256",
            "node_states",
            "nodes",
            "platform_kind",
            "profile_or_parameter_record_sha256",
            "provider_local_snapshot_sha256",
            "schema_version",
        },
        name="request graph",
    )
    validate_metric_problem(value["metric_problem"])
    metric = value["metric_problem"]
    if (
        value["node_states"] != metric["node_states"]
        or value["node_state_root_sha256"] != metric["node_state_root_sha256"]
        or value["provider_local_snapshot_sha256"]
        != metric["provider_local_snapshot_sha256"]
    ):
        raise ValueError("request graph metric join mismatch")
    expected_action_sha = domain_hash(
        "g2-request-action-envelope/v2",
        canonical_json_bytes(value["action_envelope"]),
    )
    if value["action_envelope_sha256"] != expected_action_sha:
        raise ValueError("request graph action envelope hash mismatch")
    expected_metric_sha = domain_hash(
        "g2-request-metric-problem/v3",
        canonical_json_bytes(metric),
    )
    if value["metric_problem_sha256"] != expected_metric_sha:
        raise ValueError("request graph metric hash mismatch")
    nodes = set(value["nodes"])
    if nodes != set(value["node_states"]):
        raise ValueError("request graph node state set mismatch")
    if int(value["expected_node_count"]) != len(value["nodes"]):
        raise ValueError("request graph derived node count mismatch")
    derived_edge_count = len(value["nodes"]) * len(
        value["action_envelope"]["actions"]
    )
    if int(value["expected_candidate_edge_count"]) != derived_edge_count:
        raise ValueError("request graph derived candidate edge count mismatch")
    if len(value["candidate_edges"]) != int(
        value["expected_candidate_edge_count"]
    ):
        raise ValueError("request graph candidate edge count mismatch")
    edge_required = {
        "accepted",
        "action_id",
        "action_index",
        "cost_milli",
        "edge_id",
        "from_node",
        "metric_l2_witness",
        "oracle_decision_sha256",
        "oracle_input",
        "oracle_numeric_witness",
        "oracle_safety_slacks",
        "reject_reason",
        "safety_slack_mm",
        "source_state",
        "target_state",
        "terrain_snapshot_sha256",
        "to_node",
    }
    action_ids = [
        action.get("action_id")
        for action in value["action_envelope"]["actions"]
    ]
    if (
        any(not isinstance(action_id, str) for action_id in action_ids)
        or len(action_ids) != len(set(action_ids))
    ):
        raise ValueError("request graph action ids must be unique")
    edge_ids: set[str] = set()
    for edge in value["candidate_edges"]:
        _require_exact_keys(
            edge,
            required=edge_required,
            name="request graph edge",
        )
        if (
            edge["edge_id"] in edge_ids
            or not isinstance(edge["action_index"], int)
            or not 0 <= edge["action_index"] < len(action_ids)
            or edge["action_id"] != action_ids[edge["action_index"]]
            or
            edge["from_node"] not in nodes
            or edge["to_node"] not in nodes
            or edge["source_state"] != value["node_states"][edge["from_node"]]
            or edge["target_state"] != value["node_states"][edge["to_node"]]
            or edge["terrain_snapshot_sha256"]
            != value["provider_local_snapshot_sha256"]
            or edge["oracle_input"].get("terrain_snapshot_sha256")
            != value["provider_local_snapshot_sha256"]
        ):
            raise ValueError("request graph edge state/snapshot join mismatch")
        edge_ids.add(edge["edge_id"])
        source_pose = edge["source_state"].get(
            "body_pose", edge["source_state"].get("pose")
        )
        target_pose = edge["target_state"].get(
            "body_pose", edge["target_state"].get("pose")
        )
        if not isinstance(source_pose, dict) or not isinstance(target_pose, dict):
            raise ValueError("request graph canonical edge pose missing")
        source_x = float.fromhex(str(source_pose["x_m_hex"]))
        source_y = float.fromhex(str(source_pose["y_m_hex"]))
        target_x = float.fromhex(str(target_pose["x_m_hex"]))
        target_y = float.fromhex(str(target_pose["y_m_hex"]))
        expected_cost = round(
            math.hypot(target_x - source_x, target_y - source_y) * 1000.0
        )
        if edge["cost_milli"] != expected_cost:
            raise ValueError("request graph independent metric L2 cost mismatch")
        node_l2 = math.hypot(target_x - source_x, target_y - source_y)
        node_word = f"{struct.unpack('>Q', struct.pack('>d', node_l2))[0]:016x}"
        witness = edge["metric_l2_witness"]
        base_witness_keys = {
            "cost_milli",
            "node_l2_m_hex",
            "node_l2_m_word_hex",
            "rounding_policy",
            "schema_version",
        }
        hopper_modeled = (
            value["platform_kind"] == "hopper"
            and "outside_grid_source_cell_xy" not in edge["oracle_input"]
        )
        modeled_keys = {
            "modeled_range_cost_milli",
            "modeled_range_m_hex",
            "modeled_range_m_word_hex",
            "node_minus_modeled_range_ulp",
            "rounding_equal_milli",
        }
        _require_exact_keys(
            witness,
            required=(
                base_witness_keys | modeled_keys
                if hopper_modeled
                else base_witness_keys
            ),
            name="request graph metric L2 witness",
        )
        if (
            witness["schema_version"] != "g2-edge-metric-l2-witness/v1"
            or witness["rounding_policy"]
            != "python-binary64-round-times-1000/v1"
            or witness["node_l2_m_hex"] != node_l2.hex()
            or witness["node_l2_m_word_hex"] != node_word
            or witness["cost_milli"] != expected_cost
        ):
            raise ValueError("request graph metric L2 witness mismatch")
        if hopper_modeled:
            modeled = float.fromhex(witness["modeled_range_m_hex"])
            modeled_word = (
                f"{struct.unpack('>Q', struct.pack('>d', modeled))[0]:016x}"
            )
            actual_ulp = int(node_word, 16) - int(modeled_word, 16)
            if (
                witness["modeled_range_m_word_hex"] != modeled_word
                or witness["modeled_range_cost_milli"]
                != round(modeled * 1000.0)
                or witness["node_minus_modeled_range_ulp"] != actual_ulp
                or abs(actual_ulp) != 1
                or witness["rounding_equal_milli"] is not True
                or witness["modeled_range_cost_milli"] != expected_cost
            ):
                raise ValueError(
                    "request graph Hopper exact one-ULP rounding witness mismatch"
                )
    canonical_json_bytes(value)


def _platform_primitives_per_graph_hop(platform: str) -> int:
    if platform == "legged":
        return 4
    if platform in {"wheel", "hopper"}:
        return 1
    raise ValueError("hop/resource join platform mismatch")


def _validate_resource_budget(
    budget: Any,
    *,
    scale: str,
    required_primitives: int,
) -> None:
    expected_max_path_primitives = {
        "standard": 128,
        "kilometer": 512,
    }.get(scale)
    if (
        type(budget) is not dict
        or set(budget)
        != {
            "final_target_runtime_ms",
            "max_path_primitives",
            "midterm_max_runtime_ms",
        }
        or budget["final_target_runtime_ms"] != 1000
        or budget["midterm_max_runtime_ms"] != 2000
        or budget["max_path_primitives"] != expected_max_path_primitives
        or budget["max_path_primitives"] < required_primitives
    ):
        raise ValueError("hop/resource join mismatch")


def validate_request_certificate(value: Mapping[str, Any]) -> None:
    if (
        type(value) is not dict
        or value.get("schema_version")
        != _SCHEMA_VERSIONS["request_certificate"]
    ):
        raise ValueError("request certificate schema mismatch")
    _require_exact_keys(
        value,
        required={
            "action_envelope_sha256",
            "certificate_kind",
            "cost_milli",
            "difficulty_witness",
            "distance_labels_sha256",
            "edge_source_root_sha256",
            "frame_id",
            "frontier_cut",
            "goal",
            "graph_semantic_binding_sha256",
            "graph_template_sha256",
            "metric_problem_sha256",
            "node_state_root_sha256",
            "objective_sha256",
            "open_set_exhausted",
            "oracle_reachable",
            "path_edge_ids",
            "path_metric_l2_witness",
            "path_nodes",
            "platform_kind",
            "profile_or_parameter_record_sha256",
            "provider_local_snapshot_payload_sha256",
            "provider_local_snapshot_sha256",
            "raw_source_sha256",
            "reachable_nodes",
            "reachable_set_sha256",
            "resource_budget",
            "scale",
            "schema_version",
            "settled_order",
            "start",
            "terrain_binding_sha256",
            "terrain_geometry_source_sha256",
            "terrain_transform_sha256",
            "vertical_datum_reference_sha256",
            "vertical_datum",
        },
        name="request certificate",
    )
    endpoint_required = {
        "endpoint_safety",
        "node_id",
        "node_state",
        "pose_binary64_m_rad",
        "pose_mm_urad",
    }
    for endpoint_name in ("start", "goal"):
        endpoint = value[endpoint_name]
        _require_exact_keys(
            endpoint,
            required=endpoint_required,
            name=f"request certificate {endpoint_name}",
        )
        if (
            endpoint["endpoint_safety"].get("snapshot_sha256")
            != value["provider_local_snapshot_sha256"]
        ):
            raise ValueError(
                f"request certificate {endpoint_name} snapshot join mismatch"
            )
    for field in (
        "action_envelope_sha256",
        "edge_source_root_sha256",
        "graph_semantic_binding_sha256",
        "graph_template_sha256",
        "metric_problem_sha256",
        "node_state_root_sha256",
        "objective_sha256",
        "profile_or_parameter_record_sha256",
        "provider_local_snapshot_payload_sha256",
        "provider_local_snapshot_sha256",
        "raw_source_sha256",
        "terrain_binding_sha256",
        "terrain_geometry_source_sha256",
        "terrain_transform_sha256",
        "vertical_datum_reference_sha256",
    ):
        _require_sha256(value[field], name=f"request certificate {field}")
    path_witness = value["path_metric_l2_witness"]
    if (
        not isinstance(path_witness, list)
        or [entry.get("edge_id") for entry in path_witness]
        != value["path_edge_ids"]
        or sum(int(entry.get("cost_milli", -1)) for entry in path_witness)
        != (int(value["cost_milli"]) if value["cost_milli"] is not None else 0)
    ):
        raise ValueError("request certificate path metric L2 witness mismatch")
    if value["vertical_datum"].get("reference_sha256") != value[
        "vertical_datum_reference_sha256"
    ]:
        raise ValueError("request certificate vertical datum join mismatch")
    difficulty = value["difficulty_witness"]
    _require_exact_keys(
        difficulty,
        required={
            "detour_ratio_milli",
            "difficulty_basis",
            "minimum_safety_slack_mm",
            "path_primitive_count",
            "path_safety_slacks",
            "platform_primitive_count",
            "request_hop_count",
        },
        name="request certificate difficulty witness",
    )
    hop_count = difficulty["request_hop_count"]
    path_primitive_count = difficulty["path_primitive_count"]
    platform_primitive_count = difficulty["platform_primitive_count"]
    platform = str(value["platform_kind"])
    if (
        type(hop_count) is not int
        or type(path_primitive_count) is not int
        or type(platform_primitive_count) is not int
        or min(hop_count, path_primitive_count, platform_primitive_count) < 0
    ):
        raise ValueError("hop/resource join mismatch")
    if bool(value["oracle_reachable"]):
        expected_platform_primitives = (
            hop_count * _platform_primitives_per_graph_hop(platform)
        )
        if (
            hop_count != len(value["path_edge_ids"])
            or path_primitive_count != hop_count
            or platform_primitive_count != expected_platform_primitives
            or (platform in {"legged", "hopper"} and hop_count != 1)
        ):
            raise ValueError("hop/resource join mismatch")
    elif (
        hop_count != 0
        or path_primitive_count != 0
        or platform_primitive_count != 0
        or value["path_edge_ids"]
    ):
        raise ValueError("hop/resource join mismatch")
    _validate_resource_budget(
        value["resource_budget"],
        scale=str(value["scale"]),
        required_primitives=platform_primitive_count,
    )
    canonical_json_bytes(value)


def _truth_request_required_keys(*, published: bool) -> set[str]:
    required = {
        "action_envelope_sha256",
        "determinism_seed",
        "difficulty_class",
        "frame_id",
        "goal",
        "graph_semantic_binding_sha256",
        "graph_template_sha256",
        "metric_problem",
        "metric_problem_sha256",
        "node_state_root_sha256",
        "objective",
        "objective_sha256",
        "oracle_reachable",
        "platform_kind",
        "platform_primitive_count",
        "producer_implementation_sha256",
        "profile_or_parameter_record_sha256",
        "provider_local_snapshot_payload_sha256",
        "provider_local_snapshot_sha256",
        "raw_source_sha256",
        "request_hop_count",
        "request_id",
        "resource_budget",
        "scale",
        "schema_version",
        "selection_seed",
        "source_ids",
        "start",
        "terrain_binding_sha256",
        "terrain_geometry_sha256",
        "terrain_provenance",
        "terrain_sha256",
        "terrain_transform_sha256",
        "truth_certificate_sha256",
        "truth_request_sha256",
        "vertical_datum_reference_sha256",
    }
    if published:
        required.update({"selection_hash", "selection_rank"})
    else:
        required.update({"terrain_arrays", "truth_certificate"})
    return required


def truth_request_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action_envelope_sha256": row["action_envelope_sha256"],
        "frame_id": row["frame_id"],
        "graph_template_sha256": row["graph_template_sha256"],
        "metric_problem_sha256": row["metric_problem_sha256"],
        "node_state_root_sha256": row["node_state_root_sha256"],
        "objective_sha256": row["objective_sha256"],
        "oracle_reachable": row["oracle_reachable"],
        "difficulty_class": row["difficulty_class"],
        "platform_primitive_count": row["platform_primitive_count"],
        "producer_implementation_sha256": row[
            "producer_implementation_sha256"
        ],
        "profile_or_parameter_record_sha256": row[
            "profile_or_parameter_record_sha256"
        ],
        "provider_local_snapshot_payload_sha256": row[
            "provider_local_snapshot_payload_sha256"
        ],
        "provider_local_snapshot_sha256": row[
            "provider_local_snapshot_sha256"
        ],
        "raw_source_sha256": row["raw_source_sha256"],
        "request_hop_count": row["request_hop_count"],
        "resource_budget": row["resource_budget"],
        "terrain_binding_sha256": row["terrain_binding_sha256"],
        "terrain_geometry_sha256": row["terrain_geometry_sha256"],
        "terrain_sha256": row["terrain_sha256"],
        "terrain_transform_sha256": row["terrain_transform_sha256"],
        "truth_certificate_sha256": row["truth_certificate_sha256"],
        "vertical_datum_reference_sha256": row[
            "vertical_datum_reference_sha256"
        ],
    }


def _validate_truth_request_common(
    row: dict[str, Any],
    *,
    published: bool,
) -> None:
    _require_exact_keys(
        row,
        required=_truth_request_required_keys(published=published),
        name="published truth request" if published else "truth request",
    )
    if row.get("schema_version") != _SCHEMA_VERSIONS["truth_request"]:
        raise ValueError("truth request schema mismatch")
    _require_sha256(
        row["producer_implementation_sha256"],
        name="truth request producer implementation SHA-256",
    )
    metric = row.get("metric_problem")
    if not isinstance(metric, dict):
        raise ValueError("truth request metric problem missing")
    validate_metric_problem(metric)
    metric_sha = domain_hash(
        "g2-request-metric-problem/v3",
        canonical_json_bytes(metric),
    )
    if (
        row["metric_problem_sha256"] != metric_sha
        or row["provider_local_snapshot_sha256"]
        != metric["provider_local_snapshot_sha256"]
        or row["terrain_transform_sha256"]
        != metric["terrain_transform"]["transform_sha256"]
        or row["terrain_binding_sha256"]
        != metric["terrain_binding"]["terrain_binding_sha256"]
        or row["node_state_root_sha256"] != metric["node_state_root_sha256"]
        or row["frame_id"] != metric["frame_id"]
        or row["vertical_datum_reference_sha256"]
        != metric["vertical_datum"]["reference_sha256"]
    ):
        raise ValueError("truth request metric binding mismatch")
    objective_sha = domain_hash(
        "g2-request-objective/v1",
        canonical_json_bytes(row["objective"]),
    )
    if (
        row["objective_sha256"] != objective_sha
        or row["objective"].get("kind") != "distance-only-complete-L2/v1"
        or row["objective"].get("resource_weight_milli") != 0
    ):
        raise ValueError("truth request objective mismatch")
    for endpoint_name in ("start", "goal"):
        endpoint = row[endpoint_name]
        metric_endpoint = metric[endpoint_name]
        if (
            endpoint["node_id"] != metric_endpoint["node_id"]
            or endpoint["pose_binary64_m_rad"]
            != metric_endpoint["pose_binary64_m_rad"]
            or endpoint["pose_mm_urad"] != metric_endpoint["pose_mm_urad"]
            or endpoint["endpoint_safety"]
            != metric_endpoint["endpoint_safety"]
            or endpoint["node_state"]
            != metric["node_states"][endpoint["node_id"]]
        ):
            raise ValueError(
                f"truth request {endpoint_name} endpoint join mismatch"
            )
        if endpoint["endpoint_safety"].get("safe") is not True:
            raise ValueError("truth request endpoint safety mismatch")
    hop_count = row["request_hop_count"]
    platform_primitive_count = row["platform_primitive_count"]
    if (
        type(hop_count) is not int
        or type(platform_primitive_count) is not int
        or hop_count < 0
        or platform_primitive_count < 0
    ):
        raise ValueError("hop/resource join mismatch")
    if bool(row["oracle_reachable"]):
        expected_primitives = (
            hop_count
            * _platform_primitives_per_graph_hop(
                str(row["platform_kind"])
            )
        )
        if (
            row["difficulty_class"]
            not in {"reachable", "hard_reachable"}
            or platform_primitive_count != expected_primitives
            or (
                row["platform_kind"] in {"legged", "hopper"}
                and hop_count != 1
            )
        ):
            raise ValueError("hop/resource join mismatch")
    elif (
        row["difficulty_class"] != "unreachable"
        or hop_count != 0
        or platform_primitive_count != 0
    ):
        raise ValueError("hop/resource join mismatch")
    _validate_resource_budget(
        row["resource_budget"],
        scale=str(row["scale"]),
        required_primitives=platform_primitive_count,
    )
    if not published:
        certificate = row["truth_certificate"]
        validate_request_certificate(certificate)
        if (
            sha256_bytes(canonical_json_bytes(certificate))
            != row["truth_certificate_sha256"]
        ):
            raise ValueError("truth request certificate hash mismatch")
        for field in (
            "action_envelope_sha256",
            "frame_id",
            "graph_semantic_binding_sha256",
            "graph_template_sha256",
            "metric_problem_sha256",
            "node_state_root_sha256",
            "objective_sha256",
            "profile_or_parameter_record_sha256",
            "provider_local_snapshot_payload_sha256",
            "provider_local_snapshot_sha256",
            "raw_source_sha256",
            "terrain_binding_sha256",
            "terrain_transform_sha256",
            "vertical_datum_reference_sha256",
        ):
            if certificate[field] != row[field]:
                raise ValueError(
                    f"truth request certificate {field} join mismatch"
                )
        if (
            certificate["start"] != row["start"]
            or certificate["goal"] != row["goal"]
            or certificate["resource_budget"] != row["resource_budget"]
            or certificate["vertical_datum"] != metric["vertical_datum"]
            or certificate["platform_kind"] != row["platform_kind"]
            or certificate["scale"] != row["scale"]
        ):
            raise ValueError(
                "truth request certificate execution field exact join mismatch"
            )
        difficulty = certificate["difficulty_witness"]
        if (
            certificate["oracle_reachable"] is not row["oracle_reachable"]
            or difficulty["request_hop_count"] != hop_count
            or difficulty["platform_primitive_count"]
            != platform_primitive_count
        ):
            raise ValueError("hop/resource join mismatch")
    request_identity = truth_request_identity(row)
    expected_request_sha = domain_hash(
        "g2-truth-request/v4",
        canonical_json_bytes(request_identity),
    )
    if (
        row["truth_request_sha256"] != expected_request_sha
        or row["request_id"]
        != (
            f"g2i-req-{row['platform_kind']}-{row['scale']}-"
            f"{expected_request_sha[:20]}"
        )
    ):
        raise ValueError("truth request identity mismatch")
    canonical_json_bytes(row)


def validate_published_truth_request(row: dict[str, Any]) -> None:
    _reject_keys(row, _TRUTH_ROW_FORBIDDEN)
    _validate_truth_request_common(row, published=True)


def validate_manifest(value: Mapping[str, Any]) -> None:
    if (
        type(value) is not dict
        or value.get("schema_version") != _SCHEMA_VERSIONS["manifest_core"]
    ):
        raise ValueError("manifest schema mismatch")
    core_keys = {
        "authorization_sha256",
        "candidate_id",
        "counts",
        "fixture_only",
        "formal_evidence_eligible",
        "hopper_parameter_record_sha256",
        "input_contract_sha256",
        "lola_roi_root_sha256",
        "preflight_sha256",
        "producer_implementation_sha256",
        "producer_revision",
        "profile_or_parameter_record_sha256",
        "provider_local_snapshot_index_root_sha256",
        "raw_source_sha256",
        "request_graph_archive_root_sha256",
        "schema_version",
        "specification_sha256",
    }
    _require_exact_keys(
        value,
        required={
            *core_keys,
            "input_set_id",
            "manifest_core_sha256",
        },
        name="manifest",
    )
    _require_sha256(
        value["producer_implementation_sha256"],
        name="manifest producer implementation SHA-256",
    )
    manifest_core = {key: value[key] for key in core_keys}
    if value["manifest_core_sha256"] != domain_hash(
        "g2-manifest-core/v4",
        canonical_json_bytes(manifest_core),
    ):
        raise ValueError("manifest core SHA-256 mismatch")
    expected_input_contract = domain_hash(
        "g2-producer-input-contract/v4",
        str(value["specification_sha256"]).encode("ascii"),
        str(value["producer_implementation_sha256"]).encode("ascii"),
        str(value["hopper_parameter_record_sha256"]).encode("ascii"),
        canonical_json_bytes(value["raw_source_sha256"]),
        str(value["preflight_sha256"]).encode("ascii"),
    )
    if value["input_contract_sha256"] != expected_input_contract:
        raise ValueError("manifest input contract SHA-256 mismatch")
    prefix = (
        "g2t2-fixture"
        if value["fixture_only"] is True
        else "g2t2-candidate"
    )
    if value["input_set_id"] != (
        f"{prefix}-{expected_input_contract[:24]}"
    ):
        raise ValueError("manifest input-set identity mismatch")
    canonical_json_bytes(value)


def validate_source_attestation(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    expected_payload_root_sha256: str,
) -> None:
    _require_exact_keys(
        value,
        required={
            "attestation_sha256",
            "authorization_sha256",
            "candidate_id",
            "fixture_only",
            "formal_evidence_eligible",
            "input_set_id",
            "manifest_core_sha256",
            "organizational_independence",
            "payload_root_sha256",
            "producer_id",
            "producer_implementation_sha256",
            "producer_revision",
            "schema_version",
            "status",
            "technical_independence",
        },
        name="source attestation",
    )
    expected_independence = (
        "fixture_exercise"
        if manifest["fixture_only"] is True
        else "T2_candidate"
    )
    if (
        value["schema_version"] != "g2-source-attestation/v1"
        or value["producer_id"] != "g2_t2_producer"
        or value["organizational_independence"] != "project_internal"
        or value["technical_independence"] != expected_independence
        or value["status"] != "pending_o2_signature"
        or value["payload_root_sha256"] != expected_payload_root_sha256
    ):
        raise ValueError("source attestation fixed contract mismatch")
    for field in (
        "authorization_sha256",
        "candidate_id",
        "fixture_only",
        "formal_evidence_eligible",
        "input_set_id",
        "manifest_core_sha256",
        "producer_implementation_sha256",
        "producer_revision",
    ):
        if value[field] != manifest[field]:
            raise ValueError(
                f"source attestation manifest {field} binding mismatch"
            )
    core = {
        key: child
        for key, child in value.items()
        if key != "attestation_sha256"
    }
    if value["attestation_sha256"] != domain_hash(
        "g2-source-attestation/v1",
        canonical_json_bytes(core),
    ):
        raise ValueError("source attestation identity mismatch")
    canonical_json_bytes(value)


def validate_truth_freeze(value: Mapping[str, Any]) -> None:
    if (
        type(value) is not dict
        or value.get("schema_version") != "g2-truth-freeze/v4"
    ):
        raise ValueError("truth freeze schema mismatch")
    _require_exact_keys(
        value,
        required={
            "candidate_id",
            "counts",
            "fixture_only",
            "formal_evidence_eligible",
            "fresh_reproduction_payload_root_sha256",
            "input_set_id",
            "manifest_core_sha256",
            "payload_index",
            "payload_root_sha256",
            "preflight_sha256",
            "producer_revision",
            "publication_order",
            "schema_version",
            "source_attestations_sha256",
        },
        name="truth freeze",
    )
    if value["publication_order"] != (
        "staging-audit-before-atomic-publish/v1"
    ):
        raise ValueError("truth freeze publication order mismatch")
    expected_payload_root = domain_hash(
        "g2-payload-root/v1",
        canonical_json_bytes(value["payload_index"]),
    )
    if (
        value["payload_root_sha256"] != expected_payload_root
        or value["fresh_reproduction_payload_root_sha256"]
        != expected_payload_root
    ):
        raise ValueError("truth freeze payload root mismatch")
    canonical_json_bytes(value)


def validate_raw_source_reject(row: dict[str, Any]) -> None:
    if (
        type(row) is not dict
        or row.get("schema_version") != "g2-raw-request-source-reject/v1"
    ):
        raise ValueError("raw request source reject schema mismatch")
    required = {
        "action_envelope_sha256",
        "artifact_kind",
        "artifact_sha256",
        "base_index",
        "determinism_seed",
        "endpoint_admission_policy",
        "endpoint_admission_policy_sha256",
        "evaluated_frame_count",
        "frame_admission_root_sha256",
        "frame_admission_witnesses",
        "platform_kind",
        "raw_source_reject_id",
        "raw_source_reject_sha256",
        "reason_code",
        "scale",
        "schema_version",
        "terrain_arrays_sha256",
        "terrain_provenance_sha256",
    }
    _require_exact_keys(row, required=required, name="raw request source reject")
    if (
        row["artifact_kind"] != "raw_request_source_candidate"
        or row["reason_code"] != "G2I_RAW_ENDPOINT_UNSAFE_ALL_FRAMES"
        or row["platform_kind"] not in {"wheel", "legged", "hopper"}
        or row["scale"] not in {"standard", "kilometer"}
        or row["endpoint_admission_policy"]
        != "actual-platform-endpoint-safe-then-frame-rank/v1"
    ):
        raise ValueError("raw request source reject contract mismatch")
    for field in (
        "action_envelope_sha256",
        "artifact_sha256",
        "endpoint_admission_policy_sha256",
        "frame_admission_root_sha256",
        "raw_source_reject_sha256",
        "terrain_arrays_sha256",
        "terrain_provenance_sha256",
    ):
        _require_sha256(row[field], name=f"raw request source reject {field}")
    expected_policy_sha = domain_hash(
        "g2-raw-request-endpoint-admission-policy/v1",
        str(row["endpoint_admission_policy"]).encode("ascii"),
    )
    if row["endpoint_admission_policy_sha256"] != expected_policy_sha:
        raise ValueError("raw request source reject admission policy hash mismatch")
    candidate_identity = {
        "action_envelope_sha256": row["action_envelope_sha256"],
        "base_index": row["base_index"],
        "determinism_seed": row["determinism_seed"],
        "platform_kind": row["platform_kind"],
        "scale": row["scale"],
        "terrain_arrays_sha256": row["terrain_arrays_sha256"],
        "terrain_provenance_sha256": row["terrain_provenance_sha256"],
    }
    expected_artifact_sha = domain_hash(
        "g2-raw-request-source-candidate/v1",
        canonical_json_bytes(candidate_identity),
    )
    if row["artifact_sha256"] != expected_artifact_sha:
        raise ValueError("raw request source candidate identity mismatch")
    witnesses = row["frame_admission_witnesses"]
    if (
        not isinstance(witnesses, list)
        or len(witnesses) != int(row["evaluated_frame_count"])
        or [witness.get("frame_rank") for witness in witnesses]
        != list(range(len(witnesses)))
    ):
        raise ValueError("raw request source reject frame coverage mismatch")
    witness_required = {
        "frame_rank",
        "goal_endpoint_safety_sha256",
        "goal_failure_reasons",
        "goal_safe",
        "source_to_local_transform_sha256",
        "start_endpoint_safety_sha256",
        "start_failure_reasons",
        "start_safe",
    }
    for witness in witnesses:
        if not isinstance(witness, dict):
            raise ValueError("raw request source reject frame witness missing")
        _require_exact_keys(
            witness,
            required=witness_required,
            name="raw request source reject frame witness",
        )
        for field in (
            "goal_endpoint_safety_sha256",
            "source_to_local_transform_sha256",
            "start_endpoint_safety_sha256",
        ):
            _require_sha256(
                witness[field],
                name=f"raw request source reject frame witness {field}",
            )
        if (
            type(witness["start_safe"]) is not bool
            or type(witness["goal_safe"]) is not bool
            or (
                witness["start_safe"] is True
                and witness["goal_safe"] is True
            )
        ):
            raise ValueError("raw request source reject contains a safe frame")
        if (
            not isinstance(witness["start_failure_reasons"], list)
            or not isinstance(witness["goal_failure_reasons"], list)
        ):
            raise ValueError("raw request source reject reasons missing")
    expected_frame_root = domain_hash(
        "g2-raw-request-frame-admission-root/v1",
        canonical_json_bytes(witnesses),
    )
    if row["frame_admission_root_sha256"] != expected_frame_root:
        raise ValueError("raw request source reject frame root mismatch")
    reject_core = {
        key: value
        for key, value in row.items()
        if key not in {"raw_source_reject_id", "raw_source_reject_sha256"}
    }
    expected_reject_sha = domain_hash(
        "g2-raw-request-source-reject/v1",
        canonical_json_bytes(reject_core),
    )
    if (
        row["raw_source_reject_sha256"] != expected_reject_sha
        or row["raw_source_reject_id"]
        != (
            f"g2i-reject-{row['platform_kind']}-{row['scale']}-"
            f"{expected_reject_sha[:20]}"
        )
    ):
        raise ValueError("raw request source reject identity mismatch")
    canonical_json_bytes(row)


def validate_truth_blind_case(row: dict[str, Any]) -> None:
    _reject_keys(row, _TRUTH_BLIND_FORBIDDEN)
    schema = row.get("schema_version")
    if schema in {"g2-case/v1", "g2-truth-blind-case/v2"}:
        _require_exact_keys(
            row,
            required={
                "case",
                "platform_kind",
                "schema_version",
                "stratum",
            },
            name="truth-blind primitive case",
        )
        if (
            row["platform_kind"] not in {"wheel", "legged", "hopper"}
            or type(row["case"]) is not dict
            or not isinstance(row["stratum"], str)
        ):
            raise ValueError("truth-blind primitive case contract mismatch")
        canonical_json_bytes(row)
        return
    if schema != _SCHEMA_VERSIONS["truth_blind_request_source"]:
        raise ValueError("truth-blind request source schema mismatch")
    if schema == _SCHEMA_VERSIONS["truth_blind_request_source"]:
        metric = row.get("metric_problem")
        if (
            not isinstance(metric, dict)
            or metric.get("schema_version")
            != _SCHEMA_VERSIONS["metric_problem"]
        ):
            raise ValueError("truth-blind request metric problem missing")
        _require_exact_keys(
            row,
            required={
                "action_envelope",
                "base_index",
                "determinism_seed",
                "goal_cell_xy",
                "metric_problem",
                "platform_kind",
                "provider_local_snapshot",
                "provider_local_snapshot_sha256",
                "raw_source_id",
                "raw_source_sha256",
                "scale",
                "schema_version",
                "start_cell_xy",
                "terrain_arrays",
                "terrain_provenance",
            },
            name="truth-blind request source",
        )
        snapshot = row.get("provider_local_snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError("truth-blind provider snapshot missing")
        validate_provider_local_snapshot(snapshot)
        validate_metric_problem(metric)
        if (
            row["provider_local_snapshot_sha256"]
            != snapshot["snapshot_sha256"]
            or row["provider_local_snapshot_sha256"]
            != metric["provider_local_snapshot_sha256"]
        ):
            raise ValueError("truth-blind provider snapshot join mismatch")
        source_core = {
            key: value
            for key, value in row.items()
            if key not in {"raw_source_id", "raw_source_sha256", "schema_version"}
        }
        expected_raw_sha = domain_hash(
            "g2-raw-request-source/v4",
            canonical_json_bytes(source_core),
        )
        if row["raw_source_sha256"] != expected_raw_sha:
            raise ValueError("truth-blind raw source hash mismatch")
        if row["raw_source_id"] != (
            f"g2i-raw-{row['platform_kind']}-{row['scale']}-"
            f"{expected_raw_sha[:20]}"
        ):
            raise ValueError("truth-blind raw source id mismatch")
    canonical_json_bytes(row)


def validate_provider_blind_request(row: dict[str, Any]) -> None:
    if (
        type(row) is not dict
        or row.get("schema_version") != "g2-provider-blind-request/v1"
    ):
        raise ValueError("provider blind request schema mismatch")
    forbidden = {
        "oracle_reachable",
        "difficulty_class",
        "truth_certificate",
        "truth_certificate_sha256",
        "truth_request_sha256",
        "path_edge_ids",
        "cost_milli",
        "truth_sidecar",
    }
    _reject_keys(row, forbidden)
    required = {
        "action_envelope",
        "action_envelope_sha256",
        "execution_graph",
        "frame_id",
        "goal",
        "metric_problem",
        "metric_problem_sha256",
        "objective",
        "objective_sha256",
        "platform_kind",
        "producer_implementation_sha256",
        "profile_or_parameter_record_sha256",
        "provider_local_snapshot_payload_sha256",
        "provider_local_snapshot_ref",
        "provider_local_snapshot_sha256",
        "provider_request_id",
        "provider_request_sha256",
        "resource_budget",
        "scale",
        "schema_version",
        "start",
        "terrain_geometry_sha256",
        "terrain_sha256",
        "vertical_datum",
    }
    if row["platform_kind"] == "hopper":
        required.add("hopper_parameter_record")
    _require_exact_keys(row, required=required, name="provider blind request")
    if row["platform_kind"] not in {"wheel", "legged", "hopper"}:
        raise ValueError("provider blind request platform mismatch")
    validate_metric_problem(row["metric_problem"])
    metric = row["metric_problem"]
    execution = row["execution_graph"]
    _require_exact_keys(
        execution,
        required={
            "node_state_root_sha256",
            "node_states",
            "nodes",
            "schema_version",
        },
        name="provider execution topology",
    )
    if (
        execution["schema_version"] != "g2-provider-execution-topology/v1"
        or set(execution["nodes"]) != set(execution["node_states"])
        or execution["node_states"] != metric["node_states"]
        or execution["node_state_root_sha256"]
        != metric["node_state_root_sha256"]
    ):
        raise ValueError("provider execution topology join mismatch")
    action_sha = domain_hash(
        "g2-request-action-envelope/v2",
        canonical_json_bytes(row["action_envelope"]),
    )
    metric_sha = domain_hash(
        "g2-request-metric-problem/v3",
        canonical_json_bytes(metric),
    )
    objective_sha = domain_hash(
        "g2-request-objective/v1",
        canonical_json_bytes(row["objective"]),
    )
    if (
        row["action_envelope_sha256"] != action_sha
        or row["metric_problem_sha256"] != metric_sha
        or row["objective_sha256"] != objective_sha
        or row["frame_id"] != metric["frame_id"]
        or row["provider_local_snapshot_sha256"]
        != metric["provider_local_snapshot_sha256"]
        or row["vertical_datum"] != metric["vertical_datum"]
        or row["start"]["node_id"] != metric["start"]["node_id"]
        or row["goal"]["node_id"] != metric["goal"]["node_id"]
        or row["start"]["endpoint_safety"]
        != metric["start"]["endpoint_safety"]
        or row["goal"]["endpoint_safety"]
        != metric["goal"]["endpoint_safety"]
    ):
        raise ValueError("provider blind request execution join mismatch")
    expected_ref = (
        "terrain/provider-local/"
        f"{row['provider_local_snapshot_sha256']}.json"
    )
    if row["provider_local_snapshot_ref"] != expected_ref:
        raise ValueError("provider blind request snapshot reference mismatch")
    if set(row["resource_budget"]) != {
        "final_target_runtime_ms",
        "max_path_primitives",
        "midterm_max_runtime_ms",
    }:
        raise ValueError("provider blind request resource budget mismatch")
    if row["platform_kind"] == "hopper":
        record_sha = sha256_bytes(
            canonical_json_bytes(row["hopper_parameter_record"])
        )
        if record_sha != row["profile_or_parameter_record_sha256"]:
            raise ValueError("provider blind request Hopper record mismatch")
    core = {
        key: child
        for key, child in row.items()
        if key not in {"provider_request_id", "provider_request_sha256"}
    }
    expected_sha = domain_hash(
        "g2-provider-blind-request/v1",
        canonical_json_bytes(core),
    )
    if (
        row["provider_request_sha256"] != expected_sha
        or row["provider_request_id"]
        != (
            f"g2i-provider-{row['platform_kind']}-{row['scale']}-"
            f"{expected_sha[:20]}"
        )
    ):
        raise ValueError("provider blind request identity mismatch")
    canonical_json_bytes(row)


def validate_truth_row(row: dict[str, Any]) -> None:
    _reject_keys(row, _TRUTH_ROW_FORBIDDEN)
    schema = row.get("schema_version")
    if schema == "g2-primitive-oracle-decision/v1":
        _require_exact_keys(
            row,
            required={
                "numeric_witness",
                "oracle_reason_code",
                "oracle_safe",
                "schema_version",
            },
            name="primitive oracle decision",
        )
        canonical_json_bytes(row)
        return
    if schema != _SCHEMA_VERSIONS["truth_request"]:
        raise ValueError("truth request schema mismatch")
    _validate_truth_request_common(row, published=False)
    canonical_json_bytes(row)


def make_case_identity(platform: str, case: dict[str, Any]) -> tuple[str, str]:
    if platform not in {"wheel", "legged", "hopper"}:
        raise ValueError(f"unknown platform: {platform}")
    case_sha = domain_hash(
        "g2-case/v1",
        platform.encode("utf-8"),
        canonical_json_bytes(case),
    )
    return case_sha, f"g2i-label-{platform}-{case_sha[:20]}"


def assert_unique(values: Iterable[str], *, field_name: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {field_name}: {value}")
        seen.add(value)
