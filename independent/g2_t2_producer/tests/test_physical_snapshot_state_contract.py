from __future__ import annotations

import copy
import json
import math
import struct
from pathlib import Path
from typing import Any

import pytest

from producer.canonical import canonical_json_bytes, domain_hash, sha256_bytes
from producer import generate_requests
from producer import models as producer_models
from producer import package_bundle
from producer import audit_bundle
from producer.geometry import relative_relief_um
from producer.finite_graph import reachability_certificate
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
        "g2-r2-test-roi-geometry/v1",
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
            "g2-r2-test-roi-root/v1",
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
    record = _hopper_record()
    profile_sha = (
        sha256_bytes(canonical_json_bytes(record))
        if raw["platform_kind"] == "hopper"
        else domain_hash(
            "g2-independent-profile/v2",
            str(raw["platform_kind"]).encode("ascii"),
        )
    )
    return generate_requests._graph_from_raw_source(
        raw,
        hopper_parameter_record=record,
        profile_record_sha256=profile_sha,
    )


def _float_word(value: float) -> str:
    canonical = 0.0 if value == 0.0 else value
    return f"{struct.unpack('>Q', struct.pack('>d', canonical))[0]:016x}"


def test_legacy_spec_is_rejected_before_raw_source_generation() -> None:
    legacy = copy.deepcopy(_specification())
    legacy["schema_version"] = "g2-t2-producer-spec/v1"

    with pytest.raises(ValueError, match="producer specification schema mismatch"):
        generate_requests.generate_raw_request_sources(
            legacy,
            lola_provenance=_lola_provenance(),
            hopper_parameter_record=_hopper_record(),
        )


def test_truth_request_generation_requires_real_implementation_sha() -> None:
    with pytest.raises(TypeError, match="producer_implementation_sha256"):
        generate_requests.solve_raw_request_sources(
            [],
            specification=_specification(),
            profile_record_sha256=_profile_hashes(),
            hopper_parameter_record=_hopper_record(),
        )
    with pytest.raises(ValueError, match="producer implementation SHA-256"):
        generate_requests.solve_raw_request_sources(
            [],
            specification=_specification(),
            profile_record_sha256=_profile_hashes(),
            hopper_parameter_record=_hopper_record(),
            producer_implementation_sha256="logical-revision",
        )
    with pytest.raises(TypeError, match="producer_implementation_sha256"):
        generate_requests.generate_request_pool(
            _specification(),
            profile_record_sha256=_profile_hashes(),
            lola_provenance=_lola_provenance(),
            hopper_parameter_record=_hopper_record(),
        )


def test_truth_request_implementation_sha_is_identity_bound(
    raw_sources: list[dict[str, Any]],
) -> None:
    implementation_sha = "d" * 64
    raw = _raw(raw_sources, platform="wheel")
    rows = generate_requests.solve_raw_request_sources(
        [raw] * 1056,
        specification=_specification(),
        profile_record_sha256=_profile_hashes(),
        hopper_parameter_record=_hopper_record(),
        producer_implementation_sha256=implementation_sha,
    )
    assert {
        row["producer_implementation_sha256"] for row in rows
    } == {implementation_sha}

    tampered = copy.deepcopy(rows[0])
    original_request_sha = tampered["truth_request_sha256"]
    tampered["producer_implementation_sha256"] = "e" * 64
    assert tampered["truth_request_sha256"] == original_request_sha
    with pytest.raises(ValueError, match="truth request identity mismatch"):
        producer_models.validate_truth_row(tampered)


def test_bundle_audit_rejects_request_manifest_implementation_mismatch() -> None:
    expected = "a" * 64
    reasons = audit_bundle.audit_request_implementation_bindings(
        [
            {
                "producer_implementation_sha256": "b" * 64,
                "request_id": "g2i-req-wheel-standard-tampered",
            }
        ],
        expected_implementation_sha256=expected,
        artifact_label="published request",
    )
    assert reasons == [
        "published request producer implementation SHA-256 mismatch at "
        "g2i-req-wheel-standard-tampered"
    ]


def test_every_raw_source_materializes_one_content_addressed_20x20_snapshot(
    raw_sources: list[dict[str, Any]],
) -> None:
    for platform in PLATFORMS:
        row = _raw(raw_sources, platform=platform, scale="kilometer")
        snapshot = row["provider_local_snapshot"]
        assert snapshot["schema_version"] == (
            "g2-provider-local-terrain-snapshot/v1"
        )
        assert snapshot["frame_id"] == "g2-local-metric-frame-mm/v1"
        assert snapshot["origin_mm"] == [0, 0]
        assert snapshot["shape_height_width"] == [20, 20]
        assert snapshot["resolution_mm"] == 500
        assert snapshot["physical_obstacle_cells_written"] is False
        assert set(snapshot["arrays"]) == {
            "confidence_ppm",
            "elevation_um",
            "hard_obstacle",
            "known",
            "slope_cdeg",
            "traversable",
        }
        assert all(
            len(values) == 20 and all(len(inner) == 20 for inner in values)
            for values in snapshot["arrays"].values()
        )
        core = {
            key: value
            for key, value in snapshot.items()
            if key != "snapshot_sha256"
        }
        assert snapshot["snapshot_sha256"] == domain_hash(
            "g2-provider-local-terrain-snapshot/v1",
            canonical_json_bytes(core),
        )
        assert row["provider_local_snapshot_sha256"] == (
            snapshot["snapshot_sha256"]
        )


def test_every_graph_edge_references_one_materialized_local_snapshot(
    raw_sources: list[dict[str, Any]],
) -> None:
    for platform in PLATFORMS:
        raw = _raw(raw_sources, platform=platform)
        graph, _, _ = _graph(raw)
        snapshot_sha = raw["provider_local_snapshot_sha256"]
        assert graph["provider_local_snapshot_sha256"] == snapshot_sha
        assert {
            edge["terrain_snapshot_sha256"]
            for edge in graph["candidate_edges"]
        } == {snapshot_sha}
        assert all(
            edge["oracle_input"]["terrain_snapshot_sha256"] == snapshot_sha
            for edge in graph["candidate_edges"]
        )


def test_snapshot_proxy_modifications_are_semantic_and_provenance_bound(
    raw_sources: list[dict[str, Any]],
) -> None:
    for mode_index in (6, 8):
        raw = _raw(
            raw_sources,
            platform="legged",
            base_index=mode_index,
        )
        snapshot = raw["provider_local_snapshot"]
        witness = snapshot["proxy_modification_witness"]
        assert witness["source_kind"] == "synthetic_terrain_obstacle_proxy/v1"
        assert witness["physical_obstacle_cells_written"] is False
        assert witness["operations"]
        assert len(witness["semantic_audit_sha256"]) == 64
        materialization = raw["metric_problem"]["terrain_binding"][
            "materialization"
        ]
        assert materialization["proxy_modification_semantic_audit_sha256"] == (
            witness["semantic_audit_sha256"]
        )
        assert raw["terrain_provenance"][
            "provider_local_proxy_semantic_audit_sha256"
        ] == witness["semantic_audit_sha256"]


def test_legged_truth_queries_snapshot_for_every_foothold_and_body_sweep(
    raw_sources: list[dict[str, Any]],
) -> None:
    raw = _raw(raw_sources, platform="legged", base_index=6)
    graph, start, _ = _graph(raw)
    edge = next(
        edge
        for edge in graph["candidate_edges"]
        if edge["accepted"]
        and edge["from_node"] == start
        and edge["to_node"] != start
    )
    for phase in edge["oracle_input"]["crawl_cycle"]:
        witness = phase["snapshot_query_witness"]
        assert witness["snapshot_sha256"] == (
            raw["provider_local_snapshot_sha256"]
        )
        assert witness["foothold_cell_xy"]
        assert witness["body_sweep_cell_xy"]
        foothold_cells = [
            generate_requests._snapshot_cell_record(
                raw["provider_local_snapshot"],
                cell_xy,
            )
            for cell_xy in witness["foothold_cell_xy"]
        ]
        assert all(cell["known"] for cell in foothold_cells)
        assert all(not cell["hard_obstacle"] for cell in foothold_cells)
        assert all(cell["traversable"] for cell in foothold_cells)
        assert all(cell["slope_cdeg"] <= 3000 for cell in foothold_cells)


def test_wheel_truth_queries_snapshot_for_every_sweep(
    raw_sources: list[dict[str, Any]],
) -> None:
    raw = _raw(raw_sources, platform="wheel", base_index=6)
    graph, start, _ = _graph(raw)
    edge = next(
        edge
        for edge in graph["candidate_edges"]
        if edge["accepted"]
        and edge["from_node"] == start
        and edge["to_node"] != start
    )
    witness = edge["oracle_input"]["snapshot_query_witness"]
    assert witness["snapshot_sha256"] == raw["provider_local_snapshot_sha256"]
    assert witness["swept_cell_xy"]
    assert witness["sweep_safe"] is True
    swept_cells = [
        generate_requests._snapshot_cell_record(
            raw["provider_local_snapshot"],
            cell_xy,
        )
        for cell_xy in witness["swept_cell_xy"]
    ]
    assert all(cell["known"] for cell in swept_cells)
    assert all(not cell["hard_obstacle"] for cell in swept_cells)
    assert all(cell["traversable"] for cell in swept_cells)
    assert all(cell["slope_cdeg"] <= 3000 for cell in swept_cells)


def test_frontier_cut_is_a_real_shared_snapshot_goal_failure(
    raw_sources: list[dict[str, Any]],
) -> None:
    for platform in PLATFORMS:
        raw = _raw(raw_sources, platform=platform, base_index=8)
        graph, start, goal = _graph(raw)
        operations = raw["provider_local_snapshot"][
            "proxy_modification_witness"
        ]["operations"]
        operation_kind = {
            "wheel": "interior-frontier-cut-proxy/v1",
            "legged": "legged-interior-frontier-cut-proxy/v1",
            "hopper": "hopper-mid-arc-frontier-cut-proxy/v1",
        }[platform]
        operation = next(
            candidate
            for candidate in operations
            if candidate["operation_kind"] == operation_kind
        )
        column, row = operation["cell_xy"]
        assert raw["provider_local_snapshot"]["arrays"]["hard_obstacle"][
            row
        ][column] == 1
        assert raw["metric_problem"]["start"]["endpoint_safety"]["safe"] is True
        assert raw["metric_problem"]["goal"]["endpoint_safety"]["safe"] is True
        assert reachability_certificate(graph, start, goal)[
            "oracle_reachable"
        ] is False
        selected = (
            next(
                edge
                for edge in graph["candidate_edges"]
                if edge["reject_reason"]
                == "G2I_W_CLOSED_OBSTACLE_CONTACT"
            )
            if platform == "wheel"
            else (
                next(
                    edge
                    for edge in graph["candidate_edges"]
                    if edge["from_node"] == start
                    and edge["to_node"] == goal
                )
                if platform == "hopper"
                else next(
                    edge
                    for edge in graph["candidate_edges"]
                    if edge["reject_reason"]
                    in {"G2I_L_BODY_SWEEP", "G2I_L_FOOTHOLD_OBSTACLE"}
                )
            )
        )
        assert selected["accepted"] is False
        if platform == "legged":
            assert selected["reject_reason"] in {
                "G2I_L_BODY_SWEEP",
                "G2I_L_FOOTHOLD_OBSTACLE",
            }
        else:
            assert selected["reject_reason"] == {
                "wheel": "G2I_W_CLOSED_OBSTACLE_CONTACT",
                "hopper": "G2I_H_ARC_CLEARANCE",
            }[platform]
        assert selected["terrain_snapshot_sha256"] == (
            raw["provider_local_snapshot_sha256"]
        )
        assert "local_proxy_edge_cuts" not in raw["metric_problem"]


def test_metric_problem_has_r2_units_datum_binary64_endpoints(
    raw_sources: list[dict[str, Any]],
) -> None:
    for platform in PLATFORMS:
        raw = _raw(raw_sources, platform=platform, scale="kilometer")
        metric = raw["metric_problem"]
        assert metric["schema_version"] == "g2-metric-planning-problem/v3"
        assert metric["horizontal_unit"] == "mm"
        assert metric["heading_unit"] == "urad"
        assert metric["vertical_datum"]["vertical_unit"] == "um"
        assert metric["canonical_pose_unit"] == {
            "encoding": "ieee754-binary64-hex-and-word/v1",
            "heading": "rad",
            "horizontal": "m",
        }
        assert metric["vertical_datum"]["datum_id"]
        assert len(metric["vertical_datum"]["reference_sha256"]) == 64
        for endpoint_name in ("start", "goal"):
            endpoint = metric[endpoint_name]
            assert endpoint["node_id"] in metric["node_states"]
            pose = endpoint["pose_binary64_m_rad"]
            for field in ("x_m", "y_m", "heading_rad"):
                value = float.fromhex(pose[f"{field}_hex"])
                assert pose[f"{field}_word_hex"] == _float_word(value)
            assert endpoint["pose_mm_urad"] == (
                metric["node_metric_poses_mm_urad"][endpoint["node_id"]]
            )
            assert endpoint["endpoint_safety"]["snapshot_sha256"] == (
                raw["provider_local_snapshot_sha256"]
            )


def test_selected_start_and_goal_use_distinct_fine_cells(
    raw_sources: list[dict[str, Any]],
) -> None:
    for platform in PLATFORMS:
        metric = _raw(raw_sources, platform=platform)["metric_problem"]
        assert (
            metric["start"]["endpoint_safety"]["cell_xy"]
            != metric["goal"]["endpoint_safety"]["cell_xy"]
        ), platform


@pytest.mark.parametrize("scale", ["standard", "kilometer"])
def test_legged_frontier_cut_is_a_provider_visible_geometric_barrier(
    raw_sources: list[dict[str, Any]],
    scale: str,
) -> None:
    raw = _raw(
        raw_sources,
        platform="legged",
        scale=scale,
        base_index=8,
    )
    graph, start, goal = _graph(raw)
    snapshot = raw["provider_local_snapshot"]
    operations = snapshot["proxy_modification_witness"]["operations"]
    obstacle_cells = [
        [column, row]
        for row, values in enumerate(snapshot["arrays"]["hard_obstacle"])
        for column, blocked in enumerate(values)
        if blocked
    ]
    endpoint_cells = {
        tuple(cell)
        for endpoint in ("start", "goal")
        for cell in raw["metric_problem"][endpoint]["endpoint_safety"][
            "cell_xy"
        ]
    }
    certificate = reachability_certificate(graph, start, goal)

    assert (start, goal) == ("n:0:2", "n:8:2")
    assert graph["node_states"][start]["cycle_phase"] == 0
    assert graph["node_states"][goal]["cycle_phase"] == 0
    assert obstacle_cells
    assert all(tuple(cell) not in endpoint_cells for cell in obstacle_cells)
    assert any(
        operation["operation_kind"]
        == "legged-interior-frontier-cut-proxy/v1"
        and operation["cell_xy"] in obstacle_cells
        for operation in operations
    )
    assert certificate["oracle_reachable"] is False
    assert any(
        edge["reject_reason"]
        in {"G2I_L_BODY_SWEEP", "G2I_L_FOOTHOLD_OBSTACLE"}
        for edge in certificate["frontier_cut"]
    )


@pytest.mark.parametrize("platform", PLATFORMS)
def test_all_accepted_edges_match_complete_node_state(
    raw_sources: list[dict[str, Any]],
    platform: str,
) -> None:
    raw = _raw(raw_sources, platform=platform)
    graph, _, _ = _graph(raw)
    node_states = graph["node_states"]
    assert set(node_states) == set(graph["nodes"])
    for edge in graph["candidate_edges"]:
        if not edge["accepted"]:
            continue
        assert edge["source_state"] == node_states[edge["from_node"]]
        assert edge["target_state"] == node_states[edge["to_node"]]
        assert edge["oracle_input"]["source_state"] == edge["source_state"]
        assert edge["oracle_input"]["target_state"] == edge["target_state"]


def test_all_accepted_wheel_actual_oracle_endpoints_match_target_node(
    raw_sources: list[dict[str, Any]],
) -> None:
    graph, _, _ = _graph(_raw(raw_sources, platform="wheel"))
    accepted = [edge for edge in graph["candidate_edges"] if edge["accepted"]]
    assert accepted
    for edge in accepted:
        witness = edge["oracle_numeric_witness"]
        target = graph["metric_problem"]["node_metric_poses_mm_urad"][
            edge["to_node"]
        ]
        assert witness["computed_endpoint_mm_urad"] == target
        actual_pose = generate_requests._pose_binary64_from_display(
            witness["computed_endpoint_mm_urad"]
        )
        assert actual_pose == graph["node_states"][edge["to_node"]]["pose"]


def test_all_accepted_legged_actual_crawl_terminal_state_matches_target_node(
    raw_sources: list[dict[str, Any]],
) -> None:
    graph, _, _ = _graph(_raw(raw_sources, platform="legged"))
    accepted = [edge for edge in graph["candidate_edges"] if edge["accepted"]]
    assert accepted
    for edge in accepted:
        actual = edge["oracle_numeric_witness"]["actual_terminal_state"]
        target = graph["node_states"][edge["to_node"]]
        assert actual["body_pose"] == target["body_pose"]
        assert actual["foot_contacts"] == target["foot_contacts"]
        assert actual["cycle_phase"] == target["cycle_phase"]


def test_legged_accepted_hop_finishes_translated_nominal_stance_and_phase(
    raw_sources: list[dict[str, Any]],
) -> None:
    graph, start, _ = _graph(_raw(raw_sources, platform="legged"))
    edge = next(
        edge
        for edge in graph["candidate_edges"]
        if edge["accepted"]
        and edge["from_node"] == start
        and edge["to_node"] != start
    )
    source = edge["source_state"]
    target = edge["target_state"]
    assert len(edge["oracle_input"]["crawl_cycle"]) == 4
    assert [step["moving_leg"] for step in edge["oracle_input"]["crawl_cycle"]] == [
        "rear_right",
        "front_right",
        "rear_left",
        "front_left",
    ]
    assert target["cycle_phase"] == source["cycle_phase"]
    assert target["body_pose"]["x_m_word_hex"] == _float_word(
        float.fromhex(source["body_pose"]["x_m_hex"]) + 0.25
    )
    assert target["body_pose"]["y_m_word_hex"] == (
        source["body_pose"]["y_m_word_hex"]
    )
    for source_contact, target_contact in zip(
        source["foot_contacts"],
        target["foot_contacts"],
        strict=True,
    ):
        assert target_contact["leg_id"] == source_contact["leg_id"]
        assert target_contact["x_m_word_hex"] == _float_word(
            float.fromhex(source_contact["x_m_hex"]) + 0.25
        )
        assert target_contact["y_m_word_hex"] == (
            source_contact["y_m_word_hex"]
        )


def test_hopper_support_plane_required_cells_and_relief_hash_are_bound(
    raw_sources: list[dict[str, Any]],
) -> None:
    raw = _raw(raw_sources, platform="hopper", scale="kilometer")
    graph, start, _ = _graph(raw)
    metric = graph["metric_problem"]
    support = metric["support_plane"]
    assert support["normal"] == [0, 0, 1]
    assert support["horizontal"] is True
    assert support["H_ref_m_word_hex"] == _float_word(
        float.fromhex(support["H_ref_m_hex"])
    )
    assert len(metric["relief_preservation_sha256"]) == 64
    edge = next(
        edge
        for edge in graph["candidate_edges"]
        if edge["accepted"]
        and edge["from_node"] == start
        and edge["to_node"] != start
    )
    required = edge["oracle_input"]["required_cells"]
    assert required["schema_version"] == "g2-hopper-required-cells/v1"
    assert required["support_tolerance_mm"] == 50
    assert required["boundary_kind"] == "closed"
    assert required["launch"]
    assert required["landing"]
    assert required["arc"]
    audited = [
        {
            **generate_requests._snapshot_cell_record(
                raw["provider_local_snapshot"],
                cell_xy,
            ),
            "support_residual_mm": (
                generate_requests._snapshot_cell_record(
                    raw["provider_local_snapshot"],
                    cell_xy,
                )["elevation_um"]
                - required["H_ref_um"]
            )
            / 1000.0,
        }
        for cell_xy in required["launch"] + required["landing"]
    ]
    assert all(cell["known"] is True for cell in audited)
    assert all(cell["hard_obstacle"] is False for cell in audited)
    assert all(cell["traversable"] is True for cell in audited)
    assert all(cell["slope_cdeg"] <= 3000 for cell in audited)
    assert all(abs(cell["support_residual_mm"]) <= 50 for cell in audited)
    assert edge["oracle_input"]["relief_preservation_sha256"] == (
        metric["relief_preservation_sha256"]
    )


@pytest.mark.parametrize("residual_um", [50_000, -50_000])
def test_hopper_support_height_boundary_is_inclusive(
    residual_um: int,
) -> None:
    snapshot = {
        "arrays": {
            "confidence_ppm": [[1_000_000]],
            "elevation_um": [[residual_um]],
            "hard_obstacle": [[0]],
            "known": [[1]],
            "slope_cdeg": [[3000]],
            "traversable": [[1]],
        },
        "shape_height_width": [1, 1],
        "snapshot_sha256": "a" * 64,
    }
    safe, reason, _ = generate_requests._hopper_required_cells_decision(
        {
            "H_ref_um": 0,
            "arc": [[0, 0]],
            "landing": [[0, 0]],
            "launch": [[0, 0]],
            "maximum_slope_cdeg": 3000,
            "snapshot_sha256": "a" * 64,
            "support_tolerance_um": 50_000,
        },
        snapshot,
    )
    assert safe is True
    assert reason == "G2I_H_REQUIRED_CELLS_SAFE"


@pytest.mark.parametrize("residual_um", [51_000, -51_000])
def test_hopper_support_height_outside_boundary_is_rejected(
    residual_um: int,
) -> None:
    snapshot = {
        "arrays": {
            "confidence_ppm": [[1_000_000]],
            "elevation_um": [[residual_um]],
            "hard_obstacle": [[0]],
            "known": [[1]],
            "slope_cdeg": [[0]],
            "traversable": [[1]],
        },
        "shape_height_width": [1, 1],
        "snapshot_sha256": "a" * 64,
    }
    safe, reason, slacks = (
        generate_requests._hopper_required_cells_decision(
            {
                "H_ref_um": 0,
                "arc": [[0, 0]],
                "landing": [[0, 0]],
                "launch": [[0, 0]],
                "maximum_slope_cdeg": 3000,
                "snapshot_sha256": "a" * 64,
                "support_tolerance_um": 50_000,
            },
            snapshot,
        )
    )
    assert safe is False
    assert reason == "G2I_H_LANDING_HEIGHT"
    assert slacks["required_support_height_um"] == -1000


def test_hopper_actual_edge_endpoint_one_ulp_drift_is_rejected(
    raw_sources: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw(raw_sources, platform="hopper")
    original = generate_requests._binary64_pose_identity

    def drifted_endpoint(
        x_m: float,
        y_m: float,
        heading_rad: float,
    ) -> dict[str, str]:
        return original(math.nextafter(x_m, math.inf), y_m, heading_rad)

    monkeypatch.setattr(
        generate_requests,
        "_binary64_pose_identity",
        drifted_endpoint,
    )
    graph, start, _ = _graph(raw)
    edge = next(
        edge
        for edge in graph["candidate_edges"]
        if edge["from_node"] == start and edge["to_node"] != start
    )
    assert edge["accepted"] is False
    assert edge["reject_reason"] == "G2I_STATE_JOIN"


def test_all_platform_source_relief_changes_snapshot_semantic_identity() -> None:
    baseline = generate_requests.generate_raw_request_sources(
        _specification(),
        lola_provenance=_lola_provenance(relief_delta_mm=0),
        hopper_parameter_record=_hopper_record(),
    )
    changed = generate_requests.generate_raw_request_sources(
        _specification(),
        lola_provenance=_lola_provenance(relief_delta_mm=1),
        hopper_parameter_record=_hopper_record(),
    )
    for platform in PLATFORMS:
        before = _raw(baseline, platform=platform, scale="kilometer")
        after = _raw(changed, platform=platform, scale="kilometer")
        assert before["provider_local_snapshot_sha256"] != (
            after["provider_local_snapshot_sha256"]
        )
        assert before["metric_problem"]["terrain_binding"] != (
            after["metric_problem"]["terrain_binding"]
        )
        assert before["metric_problem"]["local_proxy_geometry_sha256"] != (
            after["metric_problem"]["local_proxy_geometry_sha256"]
        )


def test_output_snapshot_is_launch_zeroed_uniform_vertical_translation(
    raw_sources: list[dict[str, Any]],
) -> None:
    for platform in PLATFORMS:
        raw = _raw(raw_sources, platform=platform, scale="kilometer")
        snapshot = raw["provider_local_snapshot"]
        translation = snapshot["vertical_translation"]
        assert translation["semantic_kind"] == (
            "uniform-vertical-translation/v1"
        )
        assert translation["input_anchor_elevation_um"] != 0
        assert translation["delta_z_um"] == (
            -translation["input_anchor_elevation_um"]
        )
        anchor_column, anchor_row = translation["anchor_cell_xy"]
        assert snapshot["arrays"]["elevation_um"][anchor_row][
            anchor_column
        ] == 0
        output_elevation = snapshot["arrays"]["elevation_um"]
        input_elevation = [
            [
                int(value) - int(translation["delta_z_um"])
                for value in row
            ]
            for row in output_elevation
        ]
        assert relative_relief_um(input_elevation) == (
            relative_relief_um(output_elevation)
        )
        producer_models.validate_provider_local_snapshot(snapshot)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("delta_z_um", 123),
        ("input_elevation_sha256", "0" * 64),
        ("output_elevation_sha256", "f" * 64),
    ],
)
def test_output_snapshot_translation_drift_is_rejected(
    raw_sources: list[dict[str, Any]],
    field: str,
    replacement: Any,
) -> None:
    snapshot = copy.deepcopy(
        _raw(
            raw_sources,
            platform="hopper",
            scale="kilometer",
        )["provider_local_snapshot"]
    )
    snapshot["vertical_translation"][field] = replacement
    with pytest.raises(ValueError, match="vertical translation"):
        producer_models.validate_provider_local_snapshot(snapshot)


def test_failed_staging_audit_does_not_publish_final_root(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "final"
    staging_root = tmp_path / ".final.staging"
    with pytest.raises(ValueError, match="staging bundle audit failed"):
        package_bundle._publish_payloads(
            output_root,
            staging_root,
            {
                "manifest.json": b"{}\n",
                "truth-freeze.json": b"{}\n",
            },
        )
    assert output_root.exists() is False


def test_archived_graph_snapshot_independently_recomputes_every_accepted_edge(
    raw_sources: list[dict[str, Any]],
) -> None:
    for platform in PLATFORMS:
        raw = _raw(raw_sources, platform=platform)
        graph, _, _ = _graph(raw)
        assert audit_bundle.audit_archived_graph_snapshot(
            graph,
            snapshot=raw["provider_local_snapshot"],
            hopper_parameter_record=_hopper_record(),
        ) == []
        mutated = copy.deepcopy(graph)
        accepted = next(
            edge
            for edge in mutated["candidate_edges"]
            if edge["accepted"]
        )
        accepted["oracle_decision_sha256"] = "0" * 64
        reasons = audit_bundle.audit_archived_graph_snapshot(
            mutated,
            snapshot=raw["provider_local_snapshot"],
            hopper_parameter_record=_hopper_record(),
        )
        assert any(
            "decision content hash mismatch" in reason for reason in reasons
        )

        witness_mutation = copy.deepcopy(graph)
        witness_edge = next(
            edge
            for edge in witness_mutation["candidate_edges"]
            if edge["accepted"]
        )
        if platform == "wheel":
            witness_edge["oracle_numeric_witness"][
                "computed_endpoint_mm_urad"
            ][0] += 1
        elif platform == "legged":
            witness_edge["oracle_input"]["actual_terminal_state"][
                "body_pose"
            ]["x_m_word_hex"] = "0" * 16
        else:
            witness_edge["oracle_input"]["required_cells"]["launch"][0] = [
                19,
                19,
            ]
        reasons = audit_bundle.audit_archived_graph_snapshot(
            witness_mutation,
            snapshot=raw["provider_local_snapshot"],
            hopper_parameter_record=_hopper_record(),
        )
        assert any("independent" in reason for reason in reasons)
        if platform == "hopper":
            outside_mutation = copy.deepcopy(graph)
            outside_edge = next(
                edge
                for edge in outside_mutation["candidate_edges"]
                if "outside_grid_source_cell_xy" in edge["oracle_input"]
            )
            outside_edge["oracle_numeric_witness"][
                "outside_grid_target_cell_xy"
            ] = [0, 0]
            reasons = audit_bundle.audit_archived_graph_snapshot(
                outside_mutation,
                snapshot=raw["provider_local_snapshot"],
                hopper_parameter_record=_hopper_record(),
            )
            assert any(
                "independent outside-domain witness mismatch" in reason
                or "independent physics failed" in reason
                for reason in reasons
            )

            gravity_mutation = copy.deepcopy(graph)
            gravity_mutation["metric_problem"]["hopper_ballistic_binding"][
                "ballistic_contract"
            ]["gravity_m_s2"] = "1.63"
            gravity_mutation["metric_problem_sha256"] = domain_hash(
                "g2-request-metric-problem/v3",
                canonical_json_bytes(gravity_mutation["metric_problem"]),
            )
            reasons = audit_bundle.audit_archived_graph_snapshot(
                gravity_mutation,
                snapshot=raw["provider_local_snapshot"],
                hopper_parameter_record=_hopper_record(),
            )
            assert any(
                "independent Hopper physical-constant binding mismatch"
                in reason
                for reason in reasons
            )

            distribution_mutation = copy.deepcopy(graph)
            distribution_edge = next(
                edge
                for edge in distribution_mutation["candidate_edges"]
                if edge["accepted"]
            )
            distribution_edge["oracle_input"]["required_cells"][
                "landing_rule"
            ]["sigma_mm"] = 288
            distribution_edge["oracle_input"]["required_cells"][
                "landing_rule"
            ]["distribution_prefix_probability_ppm"] = 989_999
            reasons = audit_bundle.audit_archived_graph_snapshot(
                distribution_mutation,
                snapshot=raw["provider_local_snapshot"],
                hopper_parameter_record=_hopper_record(),
            )
            assert any(
                "independent Hopper physical-constant binding mismatch"
                in reason
                for reason in reasons
            )


def test_archived_certificate_recomputes_accepted_adjacency_path_and_cost(
    raw_sources: list[dict[str, Any]],
) -> None:
    raw = _raw(raw_sources, platform="wheel")
    graph, start, goal = _graph(raw)
    base_certificate = reachability_certificate(graph, start, goal)
    edge_by_id = {
        edge["edge_id"]: edge for edge in graph["candidate_edges"]
    }
    certificate = {
        **base_certificate,
        "action_envelope_sha256": graph["action_envelope_sha256"],
        "difficulty_witness": generate_requests._difficulty(
            base_certificate,
            graph,
            start,
            goal,
        )[1],
        "edge_source_root_sha256": domain_hash(
            "g2-request-edge-source/v2",
            canonical_json_bytes(graph["candidate_edges"]),
        ),
        "frame_id": graph["metric_problem"]["frame_id"],
        "goal": generate_requests._request_endpoint_binding(graph, goal),
        "graph_semantic_binding_sha256": graph[
            "graph_semantic_binding_sha256"
        ],
        "graph_template_sha256": domain_hash(
            "g2-request-graph-template/v4",
            canonical_json_bytes(graph),
        ),
        "metric_problem_sha256": graph["metric_problem_sha256"],
        "node_state_root_sha256": graph["node_state_root_sha256"],
        "objective_sha256": domain_hash(
            "g2-request-objective/v1",
            canonical_json_bytes(generate_requests._request_objective()),
        ),
        "path_metric_l2_witness": [
            {
                "cost_milli": edge_by_id[edge_id]["cost_milli"],
                "edge_id": edge_id,
                "metric_l2_witness": edge_by_id[edge_id][
                    "metric_l2_witness"
                ],
            }
            for edge_id in base_certificate["path_edge_ids"]
        ],
        "platform_kind": "wheel",
        "profile_or_parameter_record_sha256": graph[
            "profile_or_parameter_record_sha256"
        ],
        "provider_local_snapshot_payload_sha256": sha256_bytes(
            canonical_json_bytes(raw["provider_local_snapshot"])
        ),
        "provider_local_snapshot_sha256": graph[
            "provider_local_snapshot_sha256"
        ],
        "raw_source_sha256": raw["raw_source_sha256"],
        "resource_budget": {
            "final_target_runtime_ms": 1000,
            "max_path_primitives": 128,
            "midterm_max_runtime_ms": 2000,
        },
        "scale": raw["scale"],
        "start": generate_requests._request_endpoint_binding(graph, start),
        "terrain_binding_sha256": graph["metric_problem"]["terrain_binding"][
            "terrain_binding_sha256"
        ],
        "terrain_geometry_source_sha256": domain_hash(
            "g2-request-terrain-arrays/v1",
            canonical_json_bytes(raw["terrain_arrays"]),
        ),
        "terrain_transform_sha256": graph["metric_problem"][
            "terrain_transform"
        ]["transform_sha256"],
        "vertical_datum_reference_sha256": graph["metric_problem"][
            "vertical_datum"
        ]["reference_sha256"],
        "vertical_datum": graph["metric_problem"]["vertical_datum"],
    }
    assert audit_bundle.audit_archived_certificate_graph(
        certificate,
        graph=graph,
    ) == []

    tampered_certificate = copy.deepcopy(certificate)
    tampered_certificate["cost_milli"] = None
    tampered_certificate["oracle_reachable"] = False
    tampered_certificate["path_edge_ids"] = []
    tampered_certificate["path_metric_l2_witness"] = []
    tampered_certificate["reachable_nodes"] = []
    tampered_certificate["difficulty_witness"] = {
        "detour_ratio_milli": 0,
        "difficulty_basis": "exhaustive-local-frontier-cut/v1",
        "minimum_safety_slack_mm": -1,
        "path_primitive_count": 0,
        "path_safety_slacks": {},
        "platform_primitive_count": 0,
        "request_hop_count": 0,
    }
    reasons = audit_bundle.audit_archived_certificate_graph(
        tampered_certificate,
        graph=graph,
    )
    assert {
        "archived certificate independent cost_milli mismatch",
        "archived certificate independent oracle_reachable mismatch",
        "archived certificate independent path_edge_ids mismatch",
        "archived certificate independent reachable_nodes mismatch",
    }.issubset(reasons)

    mutated = copy.deepcopy(graph)
    first_path_edge_id = certificate["path_edge_ids"][0]
    direct = next(
        edge
        for edge in mutated["candidate_edges"]
        if edge["edge_id"] == first_path_edge_id
    )
    direct["accepted"] = False
    direct["reject_reason"] = "G2I_AUDIT_TAMPER"
    reasons = audit_bundle.audit_archived_certificate_graph(
        certificate,
        graph=mutated,
    )
    assert any(
        "independent" in reason
        and any(
            field in reason
            for field in ("cost_milli", "path_edge_ids", "graph template")
        )
        for reason in reasons
    )


def test_removed_per_edge_proxy_injection_helpers_do_not_return() -> None:
    source = (
        ROOT / "producer" / "generate_requests.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "def _local_proxy_edge_cuts",
        "def _edge_is_cut",
        "def _near_sweep_obstacle",
        '"local_proxy_edge_cuts"',
        '"landing_pad_halfwidth_mm"',
    ):
        assert forbidden not in source
