from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest

import run_producer
from producer import audit_bundle, generate_requests, models, package_bundle
from producer.canonical import canonical_json_bytes, domain_hash, sha256_bytes
from producer.finite_graph import reachability_certificate
from producer.geometry import fine_cells_intersecting_capsule
from producer.oracle_hopper import build_hopper_parameter_record


ROOT = Path(__file__).resolve().parents[1]
PLATFORMS = ("wheel", "legged", "hopper")


def _specification() -> dict[str, Any]:
    return json.loads((ROOT / "SPECIFICATION.json").read_text(encoding="utf-8"))


def _hopper_record() -> dict[str, Any]:
    return build_hopper_parameter_record(ROOT)


def _profile_hashes() -> dict[str, str]:
    record = _hopper_record()
    return {
        "wheel": domain_hash("g2-independent-profile/v2", b"wheel"),
        "legged": domain_hash("g2-independent-profile/v2", b"legged"),
        "hopper": sha256_bytes(canonical_json_bytes(record)),
    }


def _lola_provenance(*, relief_delta_mm: int = 0) -> dict[str, Any]:
    height_mm = [
        [1000 + x * 10 + y * 20 for x in range(5)]
        for y in range(5)
    ]
    height_mm[1][1] += relief_delta_mm
    roi_geometry_sha256 = domain_hash(
        "g2-r3-test-roi-geometry/v1",
        canonical_json_bytes(height_mm),
    )
    return {
        "interpolation_schema_version": "integer-bilinear-macro-only/v1",
        "jp2_sha256": "a" * 64,
        "lbl_sha256": "b" * 64,
        "macro_source_kind": "derived_lola_20m_macro_interpolation",
        "micro_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "physical_obstacle_cells_written": False,
        "roi_records": [
            {
                "height_mm": height_mm,
                "raw_sample_sha256": "c" * 64,
                "roi_geometry_sha256": roi_geometry_sha256,
                "roi_index": 0,
                "window_col_row_width_height": [900, 900, 5, 5],
            }
        ],
        "roi_root_sha256": domain_hash(
            "g2-r3-test-roi-root/v1",
            roi_geometry_sha256.encode("ascii"),
        ),
    }


@pytest.fixture(scope="module")
def raw_sources() -> list[dict[str, Any]]:
    return generate_requests.generate_raw_request_sources(
        _specification(),
        lola_provenance=_lola_provenance(),
        hopper_parameter_record=_hopper_record(),
    )


@pytest.fixture(scope="module")
def solved_rows(
    raw_sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return generate_requests.solve_raw_request_sources(
        raw_sources,
        specification=_specification(),
        profile_record_sha256=_profile_hashes(),
        producer_implementation_sha256="d" * 64,
        hopper_parameter_record=_hopper_record(),
    )


def _raw(
    rows: list[dict[str, Any]],
    *,
    platform: str,
    scale: str = "standard",
    base_index: int = 0,
) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["platform_kind"] == platform
        and row["scale"] == scale
        and row["base_index"] == base_index
    )


def _graph(raw: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    return generate_requests._graph_from_raw_source(
        raw,
        hopper_parameter_record=_hopper_record(),
        profile_record_sha256=_profile_hashes()[raw["platform_kind"]],
    )


def _pose_xy_m(state: dict[str, Any]) -> tuple[float, float]:
    pose = state.get("body_pose", state.get("pose"))
    assert isinstance(pose, dict)
    return float.fromhex(pose["x_m_hex"]), float.fromhex(pose["y_m_hex"])


def _independent_cost_milli(edge: dict[str, Any]) -> int:
    source_x, source_y = _pose_xy_m(edge["source_state"])
    target_x, target_y = _pose_xy_m(edge["target_state"])
    return round(math.hypot(target_x - source_x, target_y - source_y) * 1000)


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(_walk_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_walk_keys(child))
        return keys
    return set()


def _rehash_snapshot(snapshot: dict[str, Any]) -> None:
    translation = snapshot["vertical_translation"]
    translation_core = {
        key: value
        for key, value in translation.items()
        if key != "translation_sha256"
    }
    translation["translation_sha256"] = domain_hash(
        "g2-provider-uniform-vertical-translation/v1",
        canonical_json_bytes(translation_core),
    )
    snapshot_core = {
        key: value
        for key, value in snapshot.items()
        if key != "snapshot_sha256"
    }
    snapshot["snapshot_sha256"] = domain_hash(
        "g2-provider-local-terrain-snapshot/v1",
        canonical_json_bytes(snapshot_core),
    )


def test_every_selected_request_has_full_safe_start_and_goal(
    raw_sources: list[dict[str, Any]],
) -> None:
    for raw in raw_sources:
        metric = raw["metric_problem"]
        for endpoint_name in ("start", "goal"):
            witness = metric[endpoint_name]["endpoint_safety"]
            assert witness["schema_version"] == "g2-endpoint-safety/v2"
            assert witness["safe"] is True
            assert witness["platform_kind"] == raw["platform_kind"]
            assert witness["components"]
            assert all(component["safe"] for component in witness["components"])
            if raw["platform_kind"] == "legged":
                    assert {component["component_id"] for component in witness["components"]} == {
                        "body",
                        "front_left",
                        "front_right",
                        "rear_left",
                        "rear_right",
                    }
                    expected_phase = (
                        1
                        if (
                            endpoint_name == "goal"
                            and metric["local_proxy_difficulty_mode"]
                            == "frontier_cut"
                        )
                        else 0
                    )
                    assert (
                        witness["support"]["cycle_phase"]
                        == expected_phase
                    )
                    assert witness["support"]["safe"] is True
            if raw["platform_kind"] == "hopper":
                assert witness["support"]["safe"] is True
                assert witness["support"]["maximum_abs_residual_um"] <= 50_000


def test_unreachable_frontier_is_internal_between_safe_endpoints(
    raw_sources: list[dict[str, Any]],
) -> None:
    for platform in PLATFORMS:
        raw = _raw(raw_sources, platform=platform, base_index=8)
        graph, start, goal = _graph(raw)
        certificate = reachability_certificate(graph, start, goal)
        assert graph["metric_problem"]["start"]["endpoint_safety"]["safe"] is True
        assert graph["metric_problem"]["goal"]["endpoint_safety"]["safe"] is True
        assert certificate["oracle_reachable"] is False
        operations = raw["provider_local_snapshot"][
            "proxy_modification_witness"
        ]["operations"]
        if platform == "wheel":
            cut = next(
                operation
                for operation in operations
                if operation["operation_kind"]
                == "interior-frontier-cut-proxy/v1"
            )
            endpoint_cells = {
                tuple(cell)
                for endpoint_name in ("start", "goal")
                for cell in raw["metric_problem"][endpoint_name][
                    "endpoint_safety"
                ]["cell_xy"]
            }
            assert tuple(cut["cell_xy"]) not in endpoint_cells
            assert any(
                edge["reject_reason"] == "G2I_W_CLOSED_OBSTACLE_CONTACT"
                for edge in graph["candidate_edges"]
            )
        elif platform == "legged":
            direct = next(
                edge
                for edge in graph["candidate_edges"]
                if edge["from_node"] == start and edge["to_node"] == goal
            )
            assert direct["reject_reason"] == (
                "G2I_L_CYCLE_PHASE_FRONTIER_CUT"
            )
            cut = next(
                operation
                for operation in operations
                if operation["operation_kind"]
                == "legged-cycle-phase-frontier-cut/v1"
            )
            assert cut["incoming_complete_cycle_phase"] == 0
            assert cut["goal_cycle_phase"] == 1
            assert graph["node_states"][goal]["cycle_phase"] == 1
            assert not any(
                cell
                for row in raw["provider_local_snapshot"]["arrays"][
                    "hard_obstacle"
                ]
                for cell in row
            )
        else:
            direct = next(
                edge
                for edge in graph["candidate_edges"]
                if edge["from_node"] == start and edge["to_node"] == goal
            )
            assert direct["reject_reason"] == "G2I_H_ARC_CLEARANCE"
            cut = next(
                operation
                for operation in operations
                if operation["operation_kind"]
                == "hopper-mid-arc-frontier-cut-proxy/v1"
            )
            endpoint_cells = {
                tuple(cell)
                for endpoint_name in ("start", "goal")
                for cell in raw["metric_problem"][endpoint_name][
                    "endpoint_safety"
                ]["cell_xy"]
            }
            assert tuple(cut["cell_xy"]) not in endpoint_cells


def test_three_platform_edge_costs_are_metric_l2_milli(
    raw_sources: list[dict[str, Any]],
) -> None:
    expected = {"wheel": 500, "legged": 250, "hopper": 1389}
    for platform in PLATFORMS:
        graph, _, _ = _graph(_raw(raw_sources, platform=platform))
        interior = next(edge for edge in graph["candidate_edges"] if edge["accepted"])
        assert interior["cost_milli"] == expected[platform]
        assert interior["cost_milli"] == _independent_cost_milli(interior)


def test_self_consistent_wrong_l2_archive_is_rejected(
    raw_sources: list[dict[str, Any]],
) -> None:
    graph, _, _ = _graph(_raw(raw_sources, platform="wheel"))
    tampered = copy.deepcopy(graph)
    for edge in tampered["candidate_edges"]:
        if edge["accepted"]:
            edge["cost_milli"] = 1000
            edge["metric_l2_witness"]["cost_milli"] = 1000
    reasons = audit_bundle.audit_archived_graph_snapshot(
        tampered,
        snapshot=_raw(raw_sources, platform="wheel")["provider_local_snapshot"],
        hopper_parameter_record=_hopper_record(),
    )
    assert any("independent metric L2 cost mismatch" in reason for reason in reasons)


def test_archive_audit_does_not_import_production_evaluators() -> None:
    tree = ast.parse(
        (ROOT / "producer" / "audit_bundle.py").read_text(encoding="utf-8")
    )
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
    assert not imported.intersection(
        {
            "_evaluate_wheel_request",
            "_evaluate_legged_cycle",
            "_evaluate_hopper_request",
            "_snapshot_cell_record",
        }
    )


def test_systematic_production_evaluator_error_is_independently_rejected(
    raw_sources: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw(raw_sources, platform="wheel", base_index=8)
    original = generate_requests._evaluate_wheel_request

    def systematically_safe(
        case: dict[str, Any], snapshot: dict[str, Any]
    ) -> dict[str, Any]:
        decision = original(case, snapshot)
        decision["oracle_safe"] = True
        decision["oracle_reason_code"] = "G2I_W_SAFE"
        return decision

    monkeypatch.setattr(
        generate_requests, "_evaluate_wheel_request", systematically_safe
    )
    monkeypatch.setattr(
        audit_bundle,
        "_evaluate_wheel_request",
        systematically_safe,
        raising=False,
    )
    graph, _, _ = _graph(raw)
    reasons = audit_bundle.audit_archived_graph_snapshot(
        graph,
        snapshot=raw["provider_local_snapshot"],
        hopper_parameter_record=_hopper_record(),
    )
    assert any("independent acceptance mismatch" in reason for reason in reasons)


def test_certificate_request_metric_execution_fields_exact_join(
    raw_sources: list[dict[str, Any]],
) -> None:
    raw = _raw(raw_sources, platform="wheel")
    rows = generate_requests.solve_raw_request_sources(
        [raw] * 1056,
        specification=_specification(),
        profile_record_sha256=_profile_hashes(),
        producer_implementation_sha256="d" * 64,
        hopper_parameter_record=_hopper_record(),
    )
    row = rows[0]
    certificate = row["truth_certificate"]
    assert certificate["start"] == row["start"]
    assert certificate["goal"] == row["goal"]
    assert certificate["resource_budget"] == row["resource_budget"]
    assert certificate["vertical_datum"] == row["metric_problem"]["vertical_datum"]
    assert certificate["platform_kind"] == row["platform_kind"]
    assert certificate["scale"] == row["scale"]


def test_graph_semantic_binding_counts_and_ids_are_derived(
    raw_sources: list[dict[str, Any]],
) -> None:
    graph, _, _ = _graph(_raw(raw_sources, platform="wheel"))
    assert graph["expected_node_count"] == len(graph["nodes"])
    assert graph["expected_candidate_edge_count"] == (
        len(graph["nodes"]) * len(graph["action_envelope"]["actions"])
    )
    assert len(graph["nodes"]) == len(set(graph["nodes"]))
    assert len({edge["edge_id"] for edge in graph["candidate_edges"]}) == len(
        graph["candidate_edges"]
    )
    assert len(
        {action["action_id"] for action in graph["action_envelope"]["actions"]}
    ) == len(graph["action_envelope"]["actions"])

    tampered = copy.deepcopy(graph)
    tampered["expected_node_count"] += 1
    with pytest.raises(ValueError, match="derived node count"):
        models.validate_request_graph(tampered)


def test_unknown_missing_or_nonexact_schema_is_rejected() -> None:
    for row in (
        {},
        {"schema_version": "g2-unknown/v1"},
        {"schema_version": "g2-truth-blind-request-source/v3"},
    ):
        with pytest.raises(ValueError, match="schema"):
            models.validate_truth_blind_case(row)
    with pytest.raises(ValueError, match="schema"):
        models.validate_truth_row({"arbitrary": True})


def test_snapshot_exact_shape_types_ranges_and_proxy_flag(
    raw_sources: list[dict[str, Any]],
) -> None:
    snapshot = _raw(raw_sources, platform="wheel")["provider_local_snapshot"]
    models.validate_provider_local_snapshot(snapshot)
    mutations = []
    wrong_shape = copy.deepcopy(snapshot)
    wrong_shape["arrays"]["known"][0].pop()
    mutations.append(wrong_shape)
    wrong_type = copy.deepcopy(snapshot)
    wrong_type["arrays"]["confidence_ppm"][0][0] = True
    mutations.append(wrong_type)
    wrong_range = copy.deepcopy(snapshot)
    wrong_range["arrays"]["slope_cdeg"][0][0] = 9001
    mutations.append(wrong_range)
    physical = copy.deepcopy(snapshot)
    physical["physical_obstacle_cells_written"] = True
    mutations.append(physical)
    extra = copy.deepcopy(snapshot)
    extra["vertical_translation"]["unexpected"] = 1
    mutations.append(extra)
    for mutation in mutations:
        _rehash_snapshot(mutation)
        with pytest.raises(ValueError):
            models.validate_provider_local_snapshot(mutation)


def test_provider_requests_are_truth_blind(
    raw_sources: list[dict[str, Any]],
) -> None:
    projector = getattr(generate_requests, "provider_blind_request")
    raw = _raw(raw_sources, platform="wheel")
    rows = generate_requests.solve_raw_request_sources(
        [raw] * 1056,
        specification=_specification(),
        profile_record_sha256=_profile_hashes(),
        producer_implementation_sha256="d" * 64,
        hopper_parameter_record=_hopper_record(),
    )
    graph, _, _ = _graph(raw)
    blind = projector(
        rows[0],
        graph=graph,
        hopper_parameter_record=_hopper_record(),
    )
    models.validate_provider_blind_request(blind)
    forbidden = {
        "oracle_reachable",
        "difficulty_class",
        "truth_certificate",
        "truth_certificate_sha256",
        "truth_request_sha256",
        "path_edge_ids",
        "cost_milli",
    }
    assert not forbidden.intersection(_walk_keys(blind))


def test_provider_process_cannot_reach_truth_sidecar(
    raw_sources: list[dict[str, Any]],
) -> None:
    projector = getattr(generate_requests, "provider_blind_request")
    raw = _raw(raw_sources, platform="wheel")
    rows = generate_requests.solve_raw_request_sources(
        [raw] * 1056,
        specification=_specification(),
        profile_record_sha256=_profile_hashes(),
        producer_implementation_sha256="d" * 64,
        hopper_parameter_record=_hopper_record(),
    )
    graph, _, _ = _graph(raw)
    truth = rows[0]
    blind = projector(
        truth,
        graph=graph,
        hopper_parameter_record=_hopper_record(),
    )
    assert blind is not truth
    assert blind["start"] is not truth["start"]
    assert "truth_sidecar" not in _walk_keys(blind)
    assert "certificate" not in {
        key.casefold() for key in _walk_keys(blind)
    }
    entry = getattr(run_producer, "validate_provider_execution_rows")
    assert entry([blind]) == [blind]


def test_hopper_record_missing_invalid_or_mismatched_is_blocked() -> None:
    specification = _specification()
    provenance = _lola_provenance()
    with pytest.raises(ValueError, match="G2I_BLOCKED_HOPPER_PARAMETER_RECORD_MISSING"):
        generate_requests.generate_raw_request_sources(
            specification,
            lola_provenance=provenance,
            hopper_parameter_record=None,
        )
    invalid = copy.deepcopy(_hopper_record())
    invalid["schema_version"] = "g2-hopper-candidate/v1"
    with pytest.raises(ValueError, match="G2I_BLOCKED_HOPPER_PARAMETER_RECORD_INVALID"):
        generate_requests.generate_request_pool(
            specification,
            profile_record_sha256=_profile_hashes(),
            lola_provenance=provenance,
            producer_implementation_sha256="d" * 64,
            hopper_parameter_record=invalid,
        )
    mismatch = _profile_hashes()
    mismatch["hopper"] = "f" * 64
    with pytest.raises(ValueError, match="G2I_BLOCKED_HOPPER_PARAMETER_RECORD_MISMATCH"):
        generate_requests.solve_raw_request_sources(
            [],
            specification=specification,
            profile_record_sha256=mismatch,
            producer_implementation_sha256="d" * 64,
            hopper_parameter_record=_hopper_record(),
        )


def test_three_platform_outside_edges_have_explicit_domain_reason(
    raw_sources: list[dict[str, Any]],
) -> None:
    for platform in PLATFORMS:
        graph, _, _ = _graph(_raw(raw_sources, platform=platform))
        outside = [
            edge
            for edge in graph["candidate_edges"]
            if "outside_grid_source_cell_xy" in edge["oracle_input"]
        ]
        assert outside
        assert {edge["reject_reason"] for edge in outside} == {
            "G2I_GRAPH_DOMAIN_OUTSIDE"
        }
        for edge in outside:
            source = edge["oracle_input"]["outside_grid_source_cell_xy"]
            delta = edge["oracle_input"]["requested_delta_cell_xy"]
            assert edge["oracle_numeric_witness"][
                "outside_grid_target_cell_xy"
            ] == [source[0] + delta[0], source[1] + delta[1]]


def test_closed_capsule_aabb_nonintersection_and_tangent() -> None:
    nonintersecting = fine_cells_intersecting_capsule(
        start_x_mm=1366.7,
        start_y_mm=1833.3,
        end_x_mm=1366.7,
        end_y_mm=1833.3,
        radius_mm=100,
        resolution_mm=500,
        shape_height_width=(20, 20),
    )
    assert [3, 3] not in nonintersecting

    tangent = fine_cells_intersecting_capsule(
        start_x_mm=1400.0,
        start_y_mm=1750.0,
        end_x_mm=1400.0,
        end_y_mm=1750.0,
        radius_mm=100,
        resolution_mm=500,
        shape_height_width=(20, 20),
    )
    assert [3, 3] in tangent

    horizontal = fine_cells_intersecting_capsule(
        start_x_mm=1000.0,
        start_y_mm=1250.0,
        end_x_mm=2000.0,
        end_y_mm=1250.0,
        radius_mm=0,
        resolution_mm=500,
        shape_height_width=(20, 20),
    )
    assert [2, 2] in horizontal and [3, 2] in horizontal

    diagonal = fine_cells_intersecting_capsule(
        start_x_mm=1000.0,
        start_y_mm=1000.0,
        end_x_mm=2000.0,
        end_y_mm=2000.0,
        radius_mm=0,
        resolution_mm=500,
        shape_height_width=(20, 20),
    )
    assert [2, 2] in diagonal and [3, 3] in diagonal


def test_attestation_exact_schema_and_cross_layer_binding() -> None:
    builder = getattr(package_bundle, "build_source_attestation")
    validator = getattr(models, "validate_source_attestation")
    manifest = {
        "authorization_sha256": "a" * 64,
        "candidate_id": "fixture-r3",
        "fixture_only": True,
        "formal_evidence_eligible": False,
        "input_set_id": "g2t2-fixture-r3",
        "manifest_core_sha256": "b" * 64,
        "producer_implementation_sha256": "c" * 64,
        "producer_revision": "0.4.0",
    }
    attestation = builder(manifest=manifest, payload_root_sha256="d" * 64)
    validator(
        attestation,
        manifest=manifest,
        expected_payload_root_sha256="d" * 64,
    )
    tampered = copy.deepcopy(attestation)
    tampered["unexpected"] = True
    with pytest.raises(ValueError, match="exact schema"):
        validator(
            tampered,
            manifest=manifest,
            expected_payload_root_sha256="d" * 64,
        )


def test_shared_cache_has_real_cold_then_warm_hit(
    raw_sources: list[dict[str, Any]],
) -> None:
    builder = getattr(generate_requests, "request_graph_from_cache")
    raw = _raw(raw_sources, platform="wheel")
    cache: dict[str, tuple[dict[str, Any], str, str]] = {}
    builds: list[str] = []
    cold = builder(
        raw,
        profile_record_sha256=_profile_hashes()["wheel"],
        hopper_parameter_record=_hopper_record(),
        graph_cache=cache,
        on_build=builds.append,
    )
    warm = builder(
        raw,
        profile_record_sha256=_profile_hashes()["wheel"],
        hopper_parameter_record=_hopper_record(),
        graph_cache=cache,
        on_build=builds.append,
    )
    assert len(builds) == 1
    assert cold is warm


def test_hopper_node_l2_and_modeled_range_have_exact_one_ulp_rounding_witness(
    raw_sources: list[dict[str, Any]],
) -> None:
    graph, _, _ = _graph(_raw(raw_sources, platform="hopper"))
    inside = [
        edge
        for edge in graph["candidate_edges"]
        if "provider_exact_kinematics" in edge["oracle_input"]
    ]
    assert inside
    for edge in inside:
        witness = edge["metric_l2_witness"]
        assert abs(witness["node_minus_modeled_range_ulp"]) == 1
        assert witness["rounding_equal_milli"] is True
        assert witness["cost_milli"] == 1389
        assert witness["modeled_range_cost_milli"] == 1389


def test_cache_does_not_alias_scale_relief_profile_or_hopper_record(
    raw_sources: list[dict[str, Any]],
) -> None:
    wheel_standard = _raw(raw_sources, platform="wheel", scale="standard")
    wheel_kilometer = _raw(raw_sources, platform="wheel", scale="kilometer")
    key_standard = generate_requests.request_graph_cache_key(
        wheel_standard,
        profile_record_sha256=_profile_hashes()["wheel"],
        hopper_parameter_record=_hopper_record(),
    )
    key_kilometer = generate_requests.request_graph_cache_key(
        wheel_kilometer,
        profile_record_sha256=_profile_hashes()["wheel"],
        hopper_parameter_record=_hopper_record(),
    )
    key_profile = generate_requests.request_graph_cache_key(
        wheel_standard,
        profile_record_sha256="e" * 64,
        hopper_parameter_record=_hopper_record(),
    )
    assert len({key_standard, key_kilometer, key_profile}) == 3

    changed_sources = generate_requests.generate_raw_request_sources(
        _specification(),
        lola_provenance=_lola_provenance(relief_delta_mm=7),
        hopper_parameter_record=_hopper_record(),
    )
    changed = _raw(changed_sources, platform="wheel", scale="kilometer")
    key_relief = generate_requests.request_graph_cache_key(
        changed,
        profile_record_sha256=_profile_hashes()["wheel"],
        hopper_parameter_record=_hopper_record(),
    )
    assert key_relief not in {key_standard, key_kilometer, key_profile}

    hopper = _raw(raw_sources, platform="hopper")
    record = _hopper_record()
    mutated_record = copy.deepcopy(record)
    mutated_record["parameter_set_id"] = "hopper-generic-internal-proxy/r3-mutated"
    with pytest.raises(ValueError, match="Hopper profile hash"):
        generate_requests.request_graph_cache_key(
            hopper,
            profile_record_sha256=_profile_hashes()["hopper"],
            hopper_parameter_record=mutated_record,
        )


@pytest.mark.parametrize(
    ("platform", "expected_primitive_count"),
    [("legged", 4), ("hopper", 1)],
)
def test_reachable_and_hard_requests_use_one_graph_hop_with_resource_join(
    raw_sources: list[dict[str, Any]],
    solved_rows: list[dict[str, Any]],
    platform: str,
    expected_primitive_count: int,
) -> None:
    for base_index, expected_difficulty in (
        (0, "reachable"),
        (6, "hard_reachable"),
    ):
        raw = _raw(
            raw_sources,
            platform=platform,
            base_index=base_index,
        )
        row = next(
            candidate
            for candidate in solved_rows
            if candidate["raw_source_sha256"] == raw["raw_source_sha256"]
        )
        certificate = row["truth_certificate"]
        assert row["difficulty_class"] == expected_difficulty
        assert row["request_hop_count"] == 1
        assert row["platform_primitive_count"] == expected_primitive_count
        assert len(certificate["path_edge_ids"]) == 1
        assert certificate["difficulty_witness"]["request_hop_count"] == 1
        assert (
            certificate["difficulty_witness"]["platform_primitive_count"]
            == expected_primitive_count
        )
        assert (
            row["resource_budget"]["max_path_primitives"]
            >= expected_primitive_count
        )


@pytest.mark.parametrize(
    ("platform", "operation_kind", "reject_reason"),
    [
        (
            "legged",
            "legged-cycle-phase-frontier-cut/v1",
            "G2I_L_CYCLE_PHASE_FRONTIER_CUT",
        ),
        (
            "hopper",
            "hopper-mid-arc-frontier-cut-proxy/v1",
            "G2I_H_ARC_CLEARANCE",
        ),
    ],
)
def test_single_hop_frontier_cut_is_internal_with_safe_endpoints(
    raw_sources: list[dict[str, Any]],
    platform: str,
    operation_kind: str,
    reject_reason: str,
) -> None:
    raw = _raw(raw_sources, platform=platform, base_index=8)
    graph, start, goal = _graph(raw)
    assert (start, goal) == ("n:0:2", "n:1:2")
    assert graph["metric_problem"]["start"]["endpoint_safety"]["safe"] is True
    assert graph["metric_problem"]["goal"]["endpoint_safety"]["safe"] is True
    certificate = reachability_certificate(graph, start, goal)
    assert certificate["oracle_reachable"] is False
    assert certificate["open_set_exhausted"] is True
    direct = next(
        edge
        for edge in graph["candidate_edges"]
        if edge["from_node"] == start and edge["to_node"] == goal
    )
    assert direct["accepted"] is False
    assert direct["reject_reason"] == reject_reason
    operation = next(
        operation
        for operation in raw["provider_local_snapshot"][
            "proxy_modification_witness"
        ]["operations"]
        if operation["operation_kind"] == operation_kind
    )
    if platform == "hopper":
        endpoint_cells = {
            tuple(cell)
            for endpoint_name in ("start", "goal")
            for cell in raw["metric_problem"][endpoint_name][
                "endpoint_safety"
            ]["cell_xy"]
        }
        assert tuple(operation["cell_xy"]) not in endpoint_cells
        assert operation["cell_xy"] in direct["oracle_input"][
            "required_cells"
        ]["arc"]
    else:
        assert operation["incoming_complete_cycle_phase"] == 0
        assert operation["goal_cycle_phase"] == 1
        assert graph["node_states"][goal]["cycle_phase"] == 1
    assert (
        audit_bundle.audit_archived_graph_snapshot(
            graph,
            snapshot=raw["provider_local_snapshot"],
            hopper_parameter_record=_hopper_record(),
        )
        == []
    )


@pytest.mark.parametrize(
    ("platform", "expected_primitive_count"),
    [("legged", 4), ("hopper", 1)],
)
def test_hop_primitive_and_resource_join_fails_closed(
    raw_sources: list[dict[str, Any]],
    solved_rows: list[dict[str, Any]],
    platform: str,
    expected_primitive_count: int,
) -> None:
    raw = _raw(raw_sources, platform=platform, base_index=0)
    row = next(
        candidate
        for candidate in solved_rows
        if candidate["raw_source_sha256"] == raw["raw_source_sha256"]
    )
    assert row["request_hop_count"] == 1
    assert row["platform_primitive_count"] == expected_primitive_count

    certificate = copy.deepcopy(row["truth_certificate"])
    certificate["difficulty_witness"]["request_hop_count"] = 2
    with pytest.raises(ValueError, match="hop/resource join"):
        models.validate_request_certificate(certificate)

    certificate = copy.deepcopy(row["truth_certificate"])
    certificate["difficulty_witness"]["platform_primitive_count"] = (
        expected_primitive_count + 1
    )
    with pytest.raises(ValueError, match="hop/resource join"):
        models.validate_request_certificate(certificate)

    certificate = copy.deepcopy(row["truth_certificate"])
    certificate["resource_budget"]["max_path_primitives"] = (
        expected_primitive_count - 1
    )
    with pytest.raises(ValueError, match="hop/resource join"):
        models.validate_request_certificate(certificate)

    tampered = copy.deepcopy(row)
    tampered["request_hop_count"] = 2
    with pytest.raises(ValueError, match="hop/resource join"):
        models.validate_truth_row(tampered)

    tampered = copy.deepcopy(row)
    tampered["platform_primitive_count"] = expected_primitive_count + 1
    with pytest.raises(ValueError, match="hop/resource join"):
        models.validate_truth_row(tampered)
