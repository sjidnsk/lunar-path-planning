from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from producer.canonical import canonical_json_bytes, sha256_bytes
from producer.audit_bundle import audit_bundle, audit_source_tree
from producer.finite_graph import (
    build_all_optima,
    heap_dijkstra,
    quadratic_dijkstra,
    reachability_certificate,
    verify_optimum_record,
)
from producer.generate_requests import generate_request_pool, select_requests
from producer.oracle_hopper import build_hopper_parameter_record
from producer.package_bundle import build_fixture_bundle


ROOT = Path(__file__).resolve().parents[1]


def _hopper_record() -> dict[str, object]:
    return build_hopper_parameter_record(ROOT)


def _small_graph() -> dict[str, Any]:
    return {
        "schema_version": "g2-finite-graph/v1",
        "nodes": ["a", "b", "c", "d"],
        "candidate_edges": [
            {
                "accepted": True,
                "cost_milli": 1000,
                "edge_id": "ab",
                "from_node": "a",
                "reject_reason": None,
                "safety_slack_mm": 40,
                "to_node": "b",
            },
            {
                "accepted": True,
                "cost_milli": 1000,
                "edge_id": "bd",
                "from_node": "b",
                "reject_reason": None,
                "safety_slack_mm": 40,
                "to_node": "d",
            },
            {
                "accepted": True,
                "cost_milli": 3000,
                "edge_id": "ac",
                "from_node": "a",
                "reject_reason": None,
                "safety_slack_mm": 40,
                "to_node": "c",
            },
            {
                "accepted": True,
                "cost_milli": 1000,
                "edge_id": "cd",
                "from_node": "c",
                "reject_reason": None,
                "safety_slack_mm": 40,
                "to_node": "d",
            },
        ],
        "expected_node_count": 4,
        "expected_candidate_edge_count": 4,
    }


def test_two_structurally_different_solvers_agree_and_negative_cost_fails() -> None:
    graph = _small_graph()
    heap = heap_dijkstra(graph, "a", "d")
    quadratic = quadratic_dijkstra(graph, "a", "d")
    assert heap["cost_milli"] == quadratic["cost_milli"] == 2000
    assert heap["path_edge_ids"] == ["ab", "bd"]
    assert quadratic["path_edge_ids"] == ["ab", "bd"]

    bad = _small_graph()
    bad["candidate_edges"][0]["cost_milli"] = -1
    with pytest.raises(ValueError, match="negative edge cost"):
        heap_dijkstra(bad, "a", "d")


def test_unreachable_certificate_exhausts_open_set_and_keeps_frontier_cut() -> None:
    graph = _small_graph()
    graph["candidate_edges"] = [
        graph["candidate_edges"][0],
        {
            "accepted": False,
            "cost_milli": 1000,
            "edge_id": "bd-blocked",
            "from_node": "b",
            "reject_reason": "G2I_W_CLOSED_OBSTACLE_CONTACT",
            "safety_slack_mm": -1,
            "to_node": "d",
        },
    ]
    graph["expected_candidate_edge_count"] = 2
    certificate = reachability_certificate(graph, "a", "d")
    assert certificate["oracle_reachable"] is False
    assert certificate["open_set_exhausted"] is True
    assert certificate["reachable_nodes"] == ["a", "b"]
    assert certificate["frontier_cut"] == [
        {
            "edge_id": "bd-blocked",
            "from_node": "b",
            "reject_reason": "G2I_W_CLOSED_OBSTACLE_CONTACT",
            "to_node": "d",
        }
    ]


def test_three_small_map_optima_are_complete_and_reconstructable() -> None:
    spec = json.loads((ROOT / "SPECIFICATION.json").read_text(encoding="utf-8"))
    artifacts = build_all_optima(spec, hopper_parameter_record=_hopper_record())
    assert set(artifacts) == {"wheel", "legged", "hopper"}
    expected_envelopes = {
        "wheel": (288, 2016),
        "legged": (200, 1000),
        "hopper": (9, 1728),
    }
    for platform, artifact in artifacts.items():
        row = artifact["row"]
        certificate = artifact["certificate"]
        assert (row["node_count"], row["candidate_edge_count"]) == expected_envelopes[
            platform
        ]
        assert row["optimum_cost_decimal"] == certificate["quadratic_cost_decimal"]
        assert row["exhaustive_crosscheck"]["matched"] is True
        assert row["completeness"]["complete"] is True
        assert verify_optimum_record(artifact) is True


def test_request_pool_and_hash_ranked_selection_have_exact_provider_blind_mix() -> None:
    spec = json.loads((ROOT / "SPECIFICATION.json").read_text(encoding="utf-8"))
    profiles = {
        "wheel": "a" * 64,
        "legged": "b" * 64,
        "hopper": "c" * 64,
    }
    lola = {
        "fixture_only": True,
        "jp2_sha256": "fff7c2017a192788066a0867d78fd1ccf783216e26974b5e65287669aa472bac",
        "lbl_sha256": "9318f41503c7d02251ed643e6dd74b737d76378be49cd492c3abb9e05d431aaa",
        "macro_source_kind": "derived_lola_20m_macro_interpolation",
        "micro_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "physical_obstacle_cells_written": False,
    }
    pool = generate_request_pool(
        spec,
        profile_record_sha256=profiles,
        lola_provenance=lola,
    )
    assert len(pool) == 3 * (256 + 96)
    assert len({row["truth_request_sha256"] for row in pool}) == len(pool)
    forbidden = {
        "provider_success",
        "provider_safe",
        "runtime_ms",
        "complete_l2",
        "expanded_states",
        "cache_hit",
    }
    assert not forbidden.intersection({key for row in pool for key in row})

    selected = select_requests(pool, spec)
    reversed_selected = select_requests(list(reversed(pool)), spec)
    assert [row["truth_request_sha256"] for row in selected] == [
        row["truth_request_sha256"] for row in reversed_selected
    ]
    assert len(selected) == 129
    assert len({row["request_id"] for row in selected}) == 129
    assert sum(bool(row["oracle_reachable"]) for row in selected) == 114
    assert sum(not bool(row["oracle_reachable"]) for row in selected) == 15

    for platform in ("wheel", "legged", "hopper"):
        rows = [row for row in selected if row["platform_kind"] == platform]
        assert len(rows) == 43
        assert Counter(
            (row["scale"], row["difficulty_class"]) for row in rows
        ) == Counter(
            {
                ("standard", "reachable"): 23,
                ("standard", "hard_reachable"): 7,
                ("standard", "unreachable"): 3,
                ("kilometer", "reachable"): 6,
                ("kilometer", "hard_reachable"): 2,
                ("kilometer", "unreachable"): 2,
            }
        )
        assert all("provider_request_sha256" not in row for row in rows)

    kilometer = [row for row in selected if row["scale"] == "kilometer"]
    assert all(
        row["terrain_provenance"]["micro_source_kind"]
        == "synthetic_terrain_obstacle_proxy/v1"
        for row in kilometer
    )
    assert all(
        row["terrain_provenance"]["physical_obstacle_cells_written"] is False
        for row in kilometer
    )


def _fixture_inputs() -> tuple[dict[str, bytes], dict[str, Any]]:
    authorization = (
        ROOT.parents[1]
        / ".superpowers"
        / "sdd"
        / "2026-07-26-midterm-dual-gate-experiment"
        / "project-authorization.md"
    ).read_bytes()
    assert sha256_bytes(authorization) == (
        "720e11ef04ad2b57283421809a077ccf1f0b35167a9f482082241398ad0214d2"
    )
    jp2 = b"fixture-only-lola-jp2-bytes"
    label = b"fixture-only-lola-label-bytes"
    raw = {
        "lola/LDEM_FIXTURE.JP2": jp2,
        "lola/LDEM_FIXTURE.LBL": label,
        "project-authorization.md": authorization,
    }
    provenance = {
        "fixture_only": True,
        "jp2_sha256": sha256_bytes(jp2),
        "lbl_sha256": sha256_bytes(label),
        "macro_source_kind": "derived_lola_20m_macro_interpolation",
        "micro_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "physical_obstacle_cells_written": False,
    }
    return raw, provenance


def test_full_fixture_bundle_is_hash_bound_formally_ineligible_and_auditable(
    tmp_path: Path,
) -> None:
    raw, provenance = _fixture_inputs()
    bundle_root = tmp_path / "fixture-bundle"
    freeze = build_fixture_bundle(
        bundle_root,
        source_root=ROOT,
        hopper_parameter_record=_hopper_record(),
        raw_sources=raw,
        lola_provenance=provenance,
    )
    assert freeze["counts"] == {
        "primitive_labels": 10002,
        "raw_request_pool": 1056,
        "repeat_mapping": 645,
        "requests": 129,
        "small_map_optima": 3,
    }
    assert freeze["fixture_only"] is True
    assert freeze["formal_evidence_eligible"] is False
    assert freeze["input_set_id"].startswith("g2t2-fixture-")
    assert (bundle_root / "truth-freeze.json").is_file()
    assert max(
        path.stat().st_mtime_ns
        for path in bundle_root.rglob("*")
        if path.is_file() and path.name != "truth-freeze.json"
    ) <= (bundle_root / "truth-freeze.json").stat().st_mtime_ns

    audit = audit_bundle(bundle_root)
    assert audit["passed"] is True, audit["reasons"]
    assert audit["counts"] == freeze["counts"]


def test_bundle_audit_blocks_formal_injection_and_missing_raw_input(
    tmp_path: Path,
) -> None:
    raw, provenance = _fixture_inputs()
    with pytest.raises(ValueError, match="lola/LDEM_FIXTURE.LBL"):
        build_fixture_bundle(
            tmp_path / "missing-raw",
            source_root=ROOT,
            hopper_parameter_record=_hopper_record(),
            raw_sources={
                key: value
                for key, value in raw.items()
                if key != "lola/LDEM_FIXTURE.LBL"
            },
            lola_provenance=provenance,
        )
    assert not (tmp_path / "missing-raw" / "truth-freeze.json").exists()

    bundle_root = tmp_path / "formal-injection"
    build_fixture_bundle(
        bundle_root,
        source_root=ROOT,
        hopper_parameter_record=_hopper_record(),
        raw_sources=raw,
        lola_provenance=provenance,
    )
    attestation_path = bundle_root / "source-attestations.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["formal_evidence_eligible"] = True
    attestation_path.write_bytes(canonical_json_bytes(attestation) + b"\n")
    audit = audit_bundle(bundle_root)
    assert audit["passed"] is False
    assert any("formal_evidence_eligible" in reason for reason in audit["reasons"])


def test_static_source_audit_rejects_forbidden_import(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "safe.py").write_text("import hashlib\n", encoding="utf-8", newline="\n")
    assert audit_source_tree(clean)["passed"] is True

    contaminated = tmp_path / "contaminated"
    contaminated.mkdir()
    (contaminated / "bad.py").write_text(
        "import path_planner\n", encoding="utf-8", newline="\n"
    )
    result = audit_source_tree(contaminated)
    assert result["passed"] is False
    assert result["forbidden_imports"] == ["bad.py:path_planner"]
