from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

import run_producer
from producer import audit_bundle, generate_requests, oracle_hopper
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
                "raw_request_pool": 1056,
                "requests": 129,
                "small_map_optima": 3,
                "repeat_mapping": 645,
            },
            "primitive_labels",
        ),
        (
            {
                "primitive_labels": 10002,
                "raw_request_pool": 1055,
                "requests": 129,
                "small_map_optima": 3,
                "repeat_mapping": 645,
            },
            "raw_request_pool",
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
