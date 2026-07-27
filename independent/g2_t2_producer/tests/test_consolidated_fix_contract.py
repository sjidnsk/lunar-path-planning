from __future__ import annotations

import copy
import json
import math
import struct
from collections import Counter
from pathlib import Path

import pytest

import run_producer
from producer import audit_bundle, generate_requests, oracle_hopper
from producer.canonical import canonical_json_bytes, domain_hash, sha256_bytes
from producer.finite_graph import build_all_optima
from producer.generate_cases import generate_all_cases


ROOT = Path(__file__).resolve().parents[1]
_REAL_CONTAMINATED_SYS_PATH = (
    (
        "C:/Users/77634/.codex/worktrees/ca49/"
        "lunar-path-planning/src"
    ),
    (
        "D:/codex/worktrees/multiplatform-path-planner-v2/"
        "path-planner/src"
    ),
)


def _specification() -> dict[str, object]:
    return json.loads((ROOT / "SPECIFICATION.json").read_text(encoding="utf-8"))


def _record() -> dict[str, object]:
    builder = getattr(oracle_hopper, "build_hopper_parameter_record", None)
    assert callable(builder), "producer must build a record from real evaluator source bytes"
    return builder(ROOT)


def _request_lola(
    *,
    first_height_delta_mm: int = 0,
    x_gradient_delta_mm: int = 0,
) -> dict[str, object]:
    height_mm = [
        [1000 + x * 10 + y * 20 for x in range(5)]
        for y in range(5)
    ]
    height_mm[0][0] += first_height_delta_mm
    for y, row in enumerate(height_mm):
        for x in range(len(row)):
            height_mm[y][x] += x * x_gradient_delta_mm
    roi_geometry_sha256 = domain_hash(
        "g2-test-roi-geometry/v1",
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
            "g2-test-roi-root/v1",
            roi_geometry_sha256.encode("ascii"),
        ),
    }


def _request_raw_sources(
    *,
    first_height_delta_mm: int = 0,
    x_gradient_delta_mm: int = 0,
) -> list[dict[str, object]]:
    return generate_requests.generate_raw_request_sources(
        _specification(),
        lola_provenance=_request_lola(
            first_height_delta_mm=first_height_delta_mm,
            x_gradient_delta_mm=x_gradient_delta_mm,
        ),
        hopper_parameter_record=_record(),
    )


def _request_profiles(
    hopper_parameter_record: dict[str, object],
) -> dict[str, str]:
    return {
        "wheel": domain_hash("g2-independent-profile/v2", b"wheel"),
        "legged": domain_hash("g2-independent-profile/v2", b"legged"),
        "hopper": sha256_bytes(
            canonical_json_bytes(hopper_parameter_record)
        ),
    }


def _raw_request(
    rows: list[dict[str, object]],
    *,
    platform: str,
    scale: str,
    base_index: int = 0,
) -> dict[str, object]:
    return next(
        row
        for row in rows
        if row["platform_kind"] == platform
        and row["scale"] == scale
        and row["base_index"] == base_index
    )


def _request_graph(
    raw_source: dict[str, object],
    hopper_parameter_record: dict[str, object],
) -> tuple[dict[str, object], str, str]:
    return generate_requests._graph_from_raw_source(
        raw_source,
        hopper_parameter_record=hopper_parameter_record,
    )


def _all_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_all_keys(child))
    return keys


def test_production_builder_and_strict_preflight_are_exposed() -> None:
    strict = getattr(run_producer, "collect_production_isolation_evidence", None)
    assert callable(strict)

    from producer import package_bundle

    builder = getattr(package_bundle, "build_production_candidate", None)
    assert callable(builder)


def test_cli_request_identity_wiring_matches_package_contract() -> None:
    source = (ROOT / "run_producer.py").read_text(encoding="utf-8")
    package_source = (
        ROOT / "producer" / "package_bundle.py"
    ).read_text(encoding="utf-8")

    assert "g2-independent-profile/v1" not in source
    for platform in ("wheel", "legged"):
        expected = f'domain_hash("g2-independent-profile/v2", b"{platform}")'
        assert expected in source
        assert expected in package_source
    assert (
        source.count(
            "implementation_sha = producer_implementation_sha256(source_root)"
        )
        == 1
    )
    assert "producer_implementation_sha256=implementation_sha" in source


def test_preflight_fails_closed_for_missing_or_visible_project_roots(
    tmp_path: Path,
) -> None:
    strict = getattr(run_producer, "collect_production_isolation_evidence", None)
    assert callable(strict)

    missing = strict(
        source_root=Path("D:/definitely-missing-g2-source"),
        project_root=Path("D:/definitely-missing-g2-project"),
        path_planner_root=Path("D:/definitely-missing-path-planner"),
        input_root=Path("D:/definitely-missing-g2-input"),
        output_root=Path("D:/xunce/inputs/mid_dual/g2-truth/red-preflight"),
        environment={},
        sys_path_entries=[],
        loaded_modules={},
    )
    assert missing["passed"] is False
    assert any("missing" in reason for reason in missing["reasons"])

    d_root = tmp_path / "D-drive-fixture"
    source = d_root / "isolated-source"
    project = d_root / "project"
    planner = project / "path-planner"
    inputs = d_root / "inputs"
    for path in (source, project / "src", planner / "src", inputs):
        path.mkdir(parents=True, exist_ok=True)
    visible = strict(
        source_root=source,
        project_root=project,
        path_planner_root=planner,
        input_root=inputs,
        output_root=d_root / "output",
        environment={},
        sys_path_entries=[str(project / "src"), str(planner / "src")],
        loaded_modules={},
        require_d_drive=False,
    )
    assert visible["passed"] is False
    assert visible["project_path_entries"]


def test_preflight_rejects_real_hyphenated_repository_paths_but_keeps_producer_origin(
    tmp_path: Path,
) -> None:
    source = ROOT
    project = tmp_path / "configured-project"
    planner = tmp_path / "configured-provider"
    inputs = tmp_path / "inputs"
    for path in (project, planner, inputs):
        path.mkdir(parents=True, exist_ok=True)

    evidence = run_producer.collect_production_isolation_evidence(
        source_root=source,
        project_root=project,
        path_planner_root=planner,
        input_root=inputs,
        output_root=inputs / "candidate",
        environment={},
        sys_path_entries=list(_REAL_CONTAMINATED_SYS_PATH),
        loaded_modules={
            "producer.canonical": str(ROOT / "producer" / "canonical.py")
        },
        require_d_drive=False,
    )

    assert evidence["passed"] is False
    assert len(evidence["project_path_entries"]) == 2
    assert "project/provider path visible in sys.path" in evidence["reasons"]
    assert evidence["forbidden_loaded_modules"] == []
    assert any(
        row["module"] == "producer.canonical"
        and Path(row["origin"]).resolve()
        == (ROOT / "producer" / "canonical.py").resolve()
        for row in evidence["module_origins"]
    )


def test_preflight_rejects_hyphenated_repository_paths_from_any_pth_line(
    tmp_path: Path,
) -> None:
    source = tmp_path / "isolated-source"
    project = tmp_path / "configured-project"
    planner = tmp_path / "configured-provider"
    inputs = tmp_path / "inputs"
    site = tmp_path / "site-packages"
    for path in (source, project, planner, inputs, site):
        path.mkdir(parents=True, exist_ok=True)
    (site / "contaminated.pth").write_text(
        "\n".join(
            (
                _REAL_CONTAMINATED_SYS_PATH[0],
                (
                    "import site; site.addsitedir("
                    f"r'{_REAL_CONTAMINATED_SYS_PATH[1]}')"
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    evidence = run_producer.collect_production_isolation_evidence(
        source_root=source,
        project_root=project,
        path_planner_root=planner,
        input_root=inputs,
        output_root=inputs / "candidate",
        environment={},
        sys_path_entries=[str(site)],
        loaded_modules={},
        require_d_drive=False,
    )

    assert evidence["passed"] is False
    assert len(evidence["pth_injections"]) == 2
    assert "project/provider .pth injection visible" in evidence["reasons"]


def test_preflight_path_matching_does_not_reject_ordinary_word_fragments(
    tmp_path: Path,
) -> None:
    source = ROOT
    project = tmp_path / "configured-project"
    planner = tmp_path / "configured-provider"
    inputs = tmp_path / "inputs"
    harmless = tmp_path / "path_planner_notes"
    for path in (project, planner, inputs, harmless):
        path.mkdir(parents=True, exist_ok=True)

    evidence = run_producer.collect_production_isolation_evidence(
        source_root=source,
        project_root=project,
        path_planner_root=planner,
        input_root=inputs,
        output_root=inputs / "candidate",
        environment={},
        sys_path_entries=[str(harmless)],
        loaded_modules={
            "path_planner_notes": str(harmless / "notes.py"),
            "producer.canonical": str(ROOT / "producer" / "canonical.py"),
        },
        require_d_drive=False,
    )

    assert evidence["passed"] is True
    assert evidence["project_path_entries"] == []
    assert evidence["pth_injections"] == []
    assert evidence["forbidden_loaded_modules"] == []


@pytest.mark.parametrize("value", (None, "", "0", "true"))
def test_production_preflight_requires_python_no_user_site_exactly_one(
    value: str | None,
) -> None:
    environment = {} if value is None else {"PYTHONNOUSERSITE": value}
    evidence = run_producer.collect_production_isolation_evidence(
        source_root=Path("D:/missing-isolated-source"),
        project_root=Path("D:/missing-configured-project"),
        path_planner_root=Path("D:/missing-configured-provider"),
        input_root=Path("D:/missing-inputs"),
        output_root=Path("D:/missing-inputs/candidate"),
        environment=environment,
        sys_path_entries=[],
        loaded_modules={},
    )

    assert "PYTHONNOUSERSITE must equal '1'" in evidence["reasons"]


def test_production_preflight_audits_python_no_user_site_one() -> None:
    evidence = run_producer.collect_production_isolation_evidence(
        source_root=Path("D:/missing-isolated-source"),
        project_root=Path("D:/missing-configured-project"),
        path_planner_root=Path("D:/missing-configured-provider"),
        input_root=Path("D:/missing-inputs"),
        output_root=Path("D:/missing-inputs/candidate"),
        environment={"PYTHONNOUSERSITE": "1"},
        sys_path_entries=[],
        loaded_modules={},
    )

    assert "PYTHONNOUSERSITE must equal '1'" not in evidence["reasons"]
    assert evidence["environment"]["PYTHONNOUSERSITE"] == "1"


def test_generated_primitive_cases_contain_geometry_not_verdict_scalars() -> None:
    generated = generate_all_cases(
        _specification(), hopper_parameter_record=_record()
    )
    forbidden = {
        "inside_map",
        "known_sweep",
        "sweep_clearance_mm",
        "traversable",
        "endpoint_cells_clear",
        "grid_valid",
        "foothold_unknown",
        "foothold_obstacle",
        "foothold_traversable",
        "support_margin_mm",
        "body_sweep_clearance_mm",
        "arc_inside_map",
        "arc_known",
        "arc_clearance_slack_mm",
        "landing_footprint_clearance_mm",
        "launch_clearance_mm",
        "touchdown_speed_mm_s",
    }
    for rows in generated.values():
        for row in rows:
            assert not forbidden.intersection(_all_keys(row))

    wheel_case = copy.deepcopy(generated["wheel"][0]["case"])
    assert "obstacle_polygons_mm" in wheel_case["terrain"]
    from producer.oracle_wheel import evaluate_wheel

    before = evaluate_wheel(wheel_case)
    endpoint = before["numeric_witness"]["computed_endpoint_mm_urad"]
    wheel_case["terrain"]["obstacle_polygons_mm"] = [
        [
            [endpoint[0] - 1, endpoint[1] - 1],
            [endpoint[0] + 1, endpoint[1] - 1],
            [endpoint[0] + 1, endpoint[1] + 1],
            [endpoint[0] - 1, endpoint[1] + 1],
        ]
    ]
    after = evaluate_wheel(wheel_case)
    assert after["oracle_safe"] is False
    assert after["numeric_witness"] != before["numeric_witness"]


def test_hopper_lattice_and_evaluator_sources_are_exact_and_fail_closed() -> None:
    record = _record()
    generated = generate_all_cases(
        _specification(), hopper_parameter_record=record
    )["hopper"]
    lattice = [row for row in generated if row["stratum"] == "action_lattice"]
    tuples = Counter(
        (
            row["case"]["action"]["speed_mm_s"],
            row["case"]["action"]["elevation_index"],
            row["case"]["action"]["azimuth_index"],
        )
        for row in lattice
    )
    assert set(speed for speed, _, _ in tuples) == {1500, 2000, 2500, 3000}
    assert len(tuples) == 192
    assert set(tuples.values()) == {2}

    decisions = [
        oracle_hopper.evaluate_hopper(row["case"], record) for row in lattice
    ]
    stop_rejects = [
        decision
        for decision in decisions
        if decision["oracle_reason_code"] == "G2I_H_STOP"
    ]
    assert len(stop_rejects) == 48

    drifted = copy.deepcopy(record)
    drifted["stop_condition"]["evaluator_source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source SHA-256"):
        oracle_hopper.validate_hopper_parameter_record(drifted, source_root=ROOT)

    changed_radius = copy.deepcopy(record)
    changed_radius["landing_footprint_radius_m"] = "0.626"
    with pytest.raises(ValueError, match="landing_footprint_radius_m"):
        oracle_hopper.validate_hopper_parameter_record(
            changed_radius, source_root=ROOT
        )


def test_lola_decoder_uses_roi_samples_not_provenance_metadata() -> None:
    from producer import lola

    converter = getattr(lola, "macro_height_mm_from_raw_samples", None)
    assert callable(converter)
    first = converter([[1, 2], [3, 4]], scaling_factor_m="0.5")
    changed = converter([[1, 2], [3, 5]], scaling_factor_m="0.5")
    assert first != changed
    assert changed[1][1] - first[1][1] == 500

    metadata_a = {
        "jp2_sha256": "a" * 64,
        "lbl_sha256": "b" * 64,
        "note": "metadata-a",
    }
    metadata_b = {**metadata_a, "note": "metadata-b"}
    terrain_a = generate_requests.terrain_npz_bytes(
        metadata_a, macro_height_mm=first
    )
    terrain_b = generate_requests.terrain_npz_bytes(
        metadata_b, macro_height_mm=first
    )
    assert generate_requests.terrain_geometry_sha256(terrain_a) == (
        generate_requests.terrain_geometry_sha256(terrain_b)
    )


def test_small_map_identity_binds_specification_and_oracle_edge_source() -> None:
    specification = _specification()
    record = _record()
    first = build_all_optima(
        specification, hopper_parameter_record=record
    )
    changed = copy.deepcopy(specification)
    changed["boundaries"]["distance_epsilon_mm"] = 2
    second = build_all_optima(changed, hopper_parameter_record=record)

    for platform in ("wheel", "legged", "hopper"):
        assert first[platform]["row"]["certificate_sha256"] != (
            second[platform]["row"]["certificate_sha256"]
        )
        graph = first[platform]["certificate"]["graph"]
        assert all("oracle_input" in edge for edge in graph["candidate_edges"])
        assert all("oracle_decision_sha256" in edge for edge in graph["candidate_edges"])


def test_request_pool_has_truth_blind_raw_sources_before_oracle_solution() -> None:
    generator = getattr(generate_requests, "generate_raw_request_sources", None)
    solver = getattr(generate_requests, "solve_raw_request_sources", None)
    assert callable(generator)
    assert callable(solver)

    raw_sources = generator(
        _specification(),
        lola_provenance={
            "fixture_only": True,
            "jp2_sha256": "a" * 64,
            "lbl_sha256": "b" * 64,
            "macro_source_kind": "derived_lola_20m_macro_interpolation",
            "micro_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "physical_obstacle_cells_written": False,
        },
        hopper_parameter_record=_record(),
    )
    assert len(raw_sources) == 1056
    forbidden = {
        "difficulty_class",
        "oracle_reachable",
        "truth_certificate",
        "truth_certificate_sha256",
        "target_truth",
    }
    assert not forbidden.intersection(
        {key for row in raw_sources for key in _all_keys(row)}
    )


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        (
            {
                "primitive_labels": 10001,
                "raw_request_candidates": 1056,
                "raw_request_pool": 1056,
                "raw_request_rejects": 0,
                "requests": 129,
                "small_map_optima": 3,
                "repeat_mapping": 645,
            },
            "primitive_labels",
        ),
        (
            {
                "primitive_labels": 10002,
                "raw_request_candidates": 1056,
                "raw_request_pool": 1055,
                "raw_request_rejects": 0,
                "requests": 129,
                "small_map_optima": 3,
                "repeat_mapping": 645,
            },
            "raw request admission conservation",
        ),
    ],
)
def test_audit_cardinality_is_independent_of_self_reported_metadata(
    counts: dict[str, int],
    message: str,
) -> None:
    checker = getattr(audit_bundle, "audit_expected_cardinalities", None)
    assert callable(checker)
    reasons = checker(counts)
    assert any(message in reason for reason in reasons)


def test_standard_and_kilometer_graphs_do_not_alias_for_same_cell_class() -> None:
    record = _record()
    rows = _request_raw_sources()
    standard = _raw_request(rows, platform="wheel", scale="standard")
    kilometer = _raw_request(rows, platform="wheel", scale="kilometer")

    standard_graph, _, _ = _request_graph(standard, record)
    kilometer_graph, _, _ = _request_graph(kilometer, record)

    assert sha256_bytes(canonical_json_bytes(standard_graph)) != sha256_bytes(
        canonical_json_bytes(kilometer_graph)
    )


def test_kilometer_graph_is_sensitive_to_lola_height() -> None:
    record = _record()
    first = _raw_request(
        _request_raw_sources(x_gradient_delta_mm=0),
        platform="wheel",
        scale="kilometer",
    )
    changed = _raw_request(
        _request_raw_sources(x_gradient_delta_mm=200),
        platform="wheel",
        scale="kilometer",
    )

    first_graph, _, _ = _request_graph(first, record)
    changed_graph, _, _ = _request_graph(changed, record)
    first_edge = next(
        edge
        for edge in first_graph["candidate_edges"]
        if edge["from_node"] == "n:0:2" and edge["action_index"] == 0
    )
    changed_edge = next(
        edge
        for edge in changed_graph["candidate_edges"]
        if edge["from_node"] == "n:0:2" and edge["action_index"] == 0
    )

    assert sha256_bytes(canonical_json_bytes(first_graph)) != sha256_bytes(
        canonical_json_bytes(changed_graph)
    )
    assert first_graph["metric_problem"]["local_height_plane"] != (
        changed_graph["metric_problem"]["local_height_plane"]
    )
    assert first_edge["oracle_input"]["terrain"]["height_plane"] != (
        changed_edge["oracle_input"]["terrain"]["height_plane"]
    )
    assert first_edge["oracle_decision_sha256"] != (
        changed_edge["oracle_decision_sha256"]
    )
    assert first_edge["oracle_safety_slacks"] != (
        changed_edge["oracle_safety_slacks"]
    )
    assert first_graph["graph_semantic_binding_sha256"] != (
        changed_graph["graph_semantic_binding_sha256"]
    )


def test_graph_cache_is_order_invariant_across_scale_and_height(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record()
    profiles = _request_profiles(record)
    implementation_sha = "d" * 64
    rows = _request_raw_sources()
    real_builder = generate_requests._graph_from_raw_source

    def semantic_probe_builder(
        raw_source: dict[str, object],
        *,
        hopper_parameter_record: dict[str, object],
        profile_record_sha256: str | None = None,
    ) -> tuple[dict[str, object], str, str]:
        graph, start, goal = real_builder(
            raw_source,
            hopper_parameter_record=hopper_parameter_record,
            profile_record_sha256=profile_record_sha256,
        )
        graph = copy.deepcopy(graph)
        graph["semantic_probe"] = {
            "height_mm_0_0": raw_source["terrain_arrays"]["height_mm"][0][0],
            "scale": raw_source["scale"],
        }
        return graph, start, goal

    monkeypatch.setattr(
        generate_requests,
        "_graph_from_raw_source",
        semantic_probe_builder,
    )
    forward = generate_requests.solve_raw_request_sources(
        rows,
        specification=_specification(),
        profile_record_sha256=profiles,
        producer_implementation_sha256=implementation_sha,
        hopper_parameter_record=record,
    )
    reverse = generate_requests.solve_raw_request_sources(
        list(reversed(rows)),
        specification=_specification(),
        profile_record_sha256=profiles,
        producer_implementation_sha256=implementation_sha,
        hopper_parameter_record=record,
    )

    forward_by_raw = {
        row["raw_source_sha256"]: canonical_json_bytes(row)
        for row in forward
    }
    reverse_by_raw = {
        row["raw_source_sha256"]: canonical_json_bytes(row)
        for row in reverse
    }
    assert forward_by_raw == reverse_by_raw


def test_graph_binds_metric_node_poses_frame_resolution_and_terrain() -> None:
    record = _record()
    raw = _raw_request(
        _request_raw_sources(),
        platform="wheel",
        scale="kilometer",
    )
    graph, start, goal = _request_graph(raw, record)
    binding = graph.get("metric_problem")

    assert isinstance(binding, dict)
    assert binding["frame_id"] == "g2-local-metric-frame-mm/v1"
    assert binding["origin_mm"] == [0, 0]
    assert binding["macro_cell_size_mm"] == 20_000
    assert binding["provider_fine_resolution_mm"] == 500
    assert binding["scale"] == "kilometer"
    assert binding["terrain_arrays_sha256"] == domain_hash(
        "g2-request-terrain-arrays/v1",
        canonical_json_bytes(raw["terrain_arrays"]),
    )
    assert len(binding["height_geometry_sha256"]) == 64
    assert len(binding["local_proxy_geometry_sha256"]) == 64
    assert binding["terrain_transform_semantics"]
    assert binding["vertical_datum"]["height_um"] == 0
    assert binding["vertical_datum"]["datum_id"] == (
        "provider-launch-anchor-zero-datum/v1"
    )
    assert binding["terrain_transform"]["semantic_kind"] == (
        "uniform-vertical-translation/v1"
    )
    anchor_column, anchor_row = raw["provider_local_snapshot"][
        "vertical_translation"
    ]["anchor_cell_xy"]
    assert raw["provider_local_snapshot"]["arrays"]["elevation_um"][
        anchor_row
    ][anchor_column] == 0
    assert binding["source_extent_bounds_mm"] == [0, 0, 100_000, 100_000]
    assert binding["provider_local_planning_region"]["bounds_mm"] == [
        0,
        0,
        10_000,
        10_000,
    ]
    assert binding["source_to_local_transform"][
        "selection_policy_sha256"
    ]
    node_poses = binding["node_metric_poses_mm_urad"]
    assert set(node_poses) == set(graph["nodes"])
    assert len(node_poses[start]) == 3
    assert len(node_poses[goal]) == 3
    assert binding["endpoint_safety_margin_mm"]["start"] > 0
    assert binding["endpoint_safety_margin_mm"]["goal"] > 0


def test_legged_truth_edge_uses_body_pose_from_full_stance_transition() -> None:
    record = _record()
    cycle_contract = _specification()["legged_crawl_cycle"]
    raw = _raw_request(
        _request_raw_sources(),
        platform="legged",
        scale="standard",
    )
    graph, start_node, goal_node = _request_graph(raw, record)
    edge = next(
        row
        for row in graph["candidate_edges"]
        if row["from_node"] == "n:0:2" and row["action_index"] == 0
    )
    oracle_input = edge["oracle_input"]
    cycle = oracle_input.get("crawl_cycle")

    assert cycle_contract["foot_storage_order"] == [
        "front_left",
        "front_right",
        "rear_left",
        "rear_right",
    ]
    assert [
        step["moving_leg"] for step in cycle_contract["phase_steps"]
    ] == ["rear_right", "front_right", "rear_left", "front_left"]
    assert [
        step["forward_lateral_delta_mm"]
        for step in cycle_contract["phase_steps"]
    ] == [[250, 0], [250, 0], [250, 0], [250, 0]]
    assert cycle_contract["cycle_body_delta_xy_mm"] == [250, 0]
    assert isinstance(cycle, list)
    assert len(cycle) == 4

    previous_target_feet = None
    previous_settled_body = None
    for phase, step in enumerate(cycle):
        transition = step["stance_transition"]
        source_feet = transition["source_feet_mm"]
        target_feet = transition["target_feet_mm"]
        moving_leg_id = int(step["moving_leg_id"])
        expected_contract_step = cycle_contract["phase_steps"][phase]

        assert step["phase"] == phase
        assert step["required_phase"] == phase
        assert step["moving_leg"] == expected_contract_step["moving_leg"]
        assert moving_leg_id == expected_contract_step["moving_leg_id"]
        assert len(source_feet) == len(target_feet) == 4
        if previous_target_feet is not None:
            assert source_feet == previous_target_feet
        for leg_id, (source_foot, target_foot) in enumerate(
            zip(source_feet, target_feet, strict=True)
        ):
            delta = [
                target_foot[axis] - source_foot[axis] for axis in range(3)
            ]
            if leg_id == moving_leg_id:
                assert delta == transition["moving_foot_delta_mm"]
            else:
                assert delta == [0, 0, 0]

        nonmoving = [
            foot
            for leg_id, foot in enumerate(source_feet)
            if leg_id != moving_leg_id
        ]
        if previous_settled_body is None:
            assert transition["source_body_pose_mm_rational"] == {
                "denominator": 4,
                "numerator_mm": [
                    sum(foot[axis] for foot in source_feet)
                    for axis in range(3)
                ],
            }
        else:
            assert transition["source_body_pose_mm_rational"] == (
                previous_settled_body
            )
        assert transition["lift_body_pose_mm_rational"] == {
            "denominator": 3,
            "numerator_mm": [
                sum(foot[axis] for foot in nonmoving)
                for axis in range(3)
            ],
        }
        assert transition["target_body_pose_mm_rational"] == (
            transition["lift_body_pose_mm_rational"]
        )
        expected_settled = (
            {
                "denominator": 4,
                "numerator_mm": [
                    sum(foot[axis] for foot in target_feet)
                    for axis in range(3)
                ],
            }
            if phase == 3
            else transition["lift_body_pose_mm_rational"]
        )
        assert transition["settled_body_pose_mm_rational"] == (
            expected_settled
        )
        assert transition["minimum_support_margin_um"] >= 50_000
        previous_target_feet = target_feet
        previous_settled_body = transition[
            "settled_body_pose_mm_rational"
        ]

    assert oracle_input["cycle_minimum_support_margin_um"] == min(
        step["stance_transition"]["minimum_support_margin_um"]
        for step in cycle
    )
    initial = cycle[0]["stance_transition"]["source_body_pose_mm_rational"]
    terminal = cycle[-1]["stance_transition"][
        "settled_body_pose_mm_rational"
    ]
    assert [
        terminal["numerator_mm"][axis] * initial["denominator"]
        - initial["numerator_mm"][axis] * terminal["denominator"]
        for axis in range(2)
    ] == [
        value * initial["denominator"] * terminal["denominator"]
        for value in cycle_contract["cycle_body_delta_xy_mm"]
    ]
    assert oracle_input["cycle_body_delta_mm_rational"] == {
        "denominator": 4,
        "numerator_mm": [
            terminal["numerator_mm"][axis]
            - initial["numerator_mm"][axis]
            for axis in range(3)
        ],
    }


def test_hopper_truth_endpoint_uses_frozen_discrete_ballistic_action() -> None:
    record = _record()
    raw = _raw_request(
        _request_raw_sources(),
        platform="hopper",
        scale="standard",
    )
    graph, start_node, goal_node = _request_graph(raw, record)
    edge = next(
        row
        for row in graph["candidate_edges"]
        if row["from_node"] == "n:0:2" and row["action_index"] == 0
    )
    oracle_input = edge["oracle_input"]
    poses = graph["metric_problem"]["node_metric_poses_mm_urad"]
    exact_poses = graph["metric_problem"][
        "hopper_node_metric_poses_binary64_m_rad"
    ]
    speed_m_s = 1.5
    elevation_rad = math.pi / 4.0
    horizontal_speed = speed_m_s * math.cos(elevation_rad)
    vertical_speed = speed_m_s * math.sin(elevation_rad)
    flight_time_s = 2.0 * (vertical_speed / 1.62)
    expected_range_m = horizontal_speed * flight_time_s
    start_x_m = float.fromhex(exact_poses[edge["from_node"]]["x_m_hex"])
    goal_x_m = float.fromhex(exact_poses[edge["to_node"]]["x_m_hex"])

    assert oracle_input["action"] == {
        "action_id": raw["action_envelope"]["actions"][0]["action_id"],
        "azimuth_index": 0,
        "azimuth_mdeg": 0,
        "elevation_index": 1,
        "elevation_mdeg": 45000,
        "speed_index": 0,
        "speed_mm_s": 1500,
    }
    assert oracle_input["launch_pose_mm"][:2] == poses[edge["from_node"]][:2]
    assert oracle_input["provider_exact_kinematics"][
        "launch_reference_height_semantics"
    ] == "same-support-body-reference-height/v1"
    assert float.fromhex(
        oracle_input["provider_exact_kinematics"]["flight_time_s_hex"]
    ) == flight_time_s
    assert float.fromhex(
        oracle_input["provider_exact_kinematics"]["horizontal_range_m_hex"]
    ) == expected_range_m
    assert goal_x_m == start_x_m + expected_range_m
    assert (
        struct.unpack(
            ">Q",
            struct.pack(">d", goal_x_m),
        )[0]
        == int(exact_poses[edge["to_node"]]["x_m_word_hex"], 16)
    )
    assert (goal_x_m - start_x_m) * 1000.0 != (
        poses[edge["to_node"]][0] - poses[edge["from_node"]][0]
    )
    assert graph["metric_problem"]["hopper_ballistic_binding"][
        "parameter_record_sha256"
    ] == sha256_bytes(canonical_json_bytes(record))

    mutated = copy.deepcopy(graph)
    goal_binding = mutated["metric_problem"][
        "hopper_node_metric_poses_binary64_m_rad"
    ][edge["to_node"]]
    changed_word = int(goal_binding["x_m_word_hex"], 16) + 1
    goal_binding["x_m_word_hex"] = f"{changed_word:016x}"
    goal_binding["x_m_hex"] = struct.unpack(
        ">d", changed_word.to_bytes(8, "big")
    )[0].hex()
    assert sha256_bytes(canonical_json_bytes(mutated)) != sha256_bytes(
        canonical_json_bytes(graph)
    )
    reasons = audit_bundle.audit_request_graph_semantics(
        mutated,
        raw_source=raw,
        start_node=start_node,
        goal_node=goal_node,
        profile_record_sha256=_request_profiles(record)["hopper"],
        hopper_parameter_record=record,
    )
    assert "request graph hopper exact endpoint binding mismatch" in reasons


def test_hopper_ballistic_witness_uses_same_support_reference_height() -> None:
    record = _record()
    case = {
        "action": {
            "azimuth_index": 0,
            "azimuth_mdeg": 0,
            "elevation_index": 1,
            "elevation_mdeg": 45000,
            "speed_index": 0,
            "speed_mm_s": 1500,
        },
        "launch_pose_mm": [1000, 1000, 0],
        "numeric_state": "decided",
        "terrain": {
            "height_plane": {
                "gradient_x_ppm": 0,
                "gradient_y_ppm": 0,
                "origin_x_mm": 0,
                "origin_y_mm": 0,
                "origin_z_mm": 100,
            }
        },
    }
    vertical_speed = 1.5 * math.sin(math.pi / 4.0)
    horizontal_speed = 1.5 * math.cos(math.pi / 4.0)
    expected_flight_s = 2.0 * (vertical_speed / 1.62)
    expected_range_m = horizontal_speed * expected_flight_s

    witness = oracle_hopper.ballistic_witness(case, record)
    changed_reference = copy.deepcopy(record)
    changed_reference["launch_reference_height_m"] = "0.500"
    changed_witness = oracle_hopper.ballistic_witness(case, changed_reference)

    assert witness["flight_time_us"] == round(expected_flight_s * 1_000_000)
    assert witness["range_mm"] == round(expected_range_m * 1000)
    assert changed_witness["flight_time_us"] == witness["flight_time_us"]
    assert changed_witness["range_mm"] == witness["range_mm"]
    assert changed_witness["apex_height_mm"] != witness["apex_height_mm"]


def test_hopper_full_support_envelope_uses_50mm_closed_boundary() -> None:
    checker = getattr(
        generate_requests,
        "_hopper_support_envelope_max_residual_um",
        None,
    )
    assert callable(checker)
    exact_boundary = checker(
        gradient_x_ppm=0,
        gradient_y_ppm=80_000,
        delta_x_m=1.0,
        delta_y_m=0.0,
        launch_radius_mm=500,
        landing_radius_mm=625,
    )
    outside_boundary = checker(
        gradient_x_ppm=0,
        gradient_y_ppm=81_600,
        delta_x_m=1.0,
        delta_y_m=0.0,
        launch_radius_mm=500,
        landing_radius_mm=625,
    )

    assert exact_boundary == 50_000
    assert exact_boundary <= 50_000
    assert outside_boundary == 51_000
    assert outside_boundary > 50_000


def test_standard_request_terrain_contains_full_hopper_safety_envelope() -> None:
    record = _record()
    raw = _raw_request(
        _request_raw_sources(),
        platform="hopper",
        scale="standard",
    )
    graph, _, _ = _request_graph(raw, record)
    terrain = raw["terrain_arrays"]
    width_mm = len(terrain["height_mm"][0]) * 500
    height_mm = len(terrain["height_mm"]) * 500
    body_plus_arc_mm = round(
        (
            float(record["body_envelope_radius_m"])
            + float(record["arc_clearance_margin_m"])
        )
        * 1000
    )
    landing_radius_mm = round(
        float(record["landing_footprint_radius_m"]) * 1000
    )

    for edge in graph["candidate_edges"]:
        if not edge["accepted"]:
            continue
        oracle_input = edge["oracle_input"]
        witness = oracle_hopper.ballistic_witness(oracle_input, record)
        launch_x, launch_y = oracle_input["launch_pose_mm"][:2]
        landing_x = witness["landing_x_mm"]
        landing_y = witness["landing_y_mm"]
        assert min(launch_x, launch_y) - body_plus_arc_mm >= 0
        assert max(launch_x, landing_x) + landing_radius_mm <= width_mm
        assert max(launch_y, landing_y) + landing_radius_mm <= height_mm


def test_bundle_audit_rejects_graph_without_terrain_resolution_binding() -> None:
    checker = getattr(audit_bundle, "audit_request_graph_semantics", None)
    assert callable(checker)
    record = _record()
    raw = _raw_request(
        _request_raw_sources(),
        platform="wheel",
        scale="kilometer",
    )
    legacy_graph, start, goal = _request_graph(raw, record)
    legacy_graph.pop("metric_problem", None)

    reasons = checker(
        legacy_graph,
        raw_source=raw,
        start_node=start,
        goal_node=goal,
        profile_record_sha256=_request_profiles(record)["wheel"],
        hopper_parameter_record=record,
    )

    assert "request graph metric problem binding missing" in reasons
