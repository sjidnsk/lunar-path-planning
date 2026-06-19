import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    scripts_path = str(repo_root / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    return repo_root


def test_in_process_batch_astar_validates_multiple_proposals_without_route_adapter(tmp_path, monkeypatch) -> None:
    repo_root = _repo_root()
    import scripts.xunce_frontier_nbv_validation as validation

    def fail_if_legacy_route_adapter_path_is_used(*args, **kwargs):
        raise AssertionError("legacy evaluate_candidate_paths path should not be used by batch A* mode")

    monkeypatch.setattr(validation, "_evaluate_candidate_paths", fail_if_legacy_route_adapter_path_is_used)
    contract_path, sidecar_path = _write_tiny_contract_and_sidecar(tmp_path)

    rows = validation.validate_candidate_cells(
        scenario={"scenario_id": "batch-astar"},
        proposal_rows=[
            {"proposal_id": "p0", "cell": [2, 0], "frontier_candidate_source": "frontier_boundary"},
            {"proposal_id": "p1", "cell": [2, 1], "frontier_candidate_source": "roi_undercovered_boundary"},
        ],
        contract_path=contract_path,
        sidecar_path=sidecar_path,
        current_cell=(0, 0),
        repo_root=repo_root,
        output_work_root=tmp_path / "work",
        validation_mode="in_process_path_planner_astar_batch",
    )

    assert len(rows) == 2
    assert all(row["proposal_only"] is False for row in rows)
    assert all(row["proposal_validated_by_path_feedback"] is True for row in rows)
    assert {row["planner_validation_backend"] for row in rows} == {"in_process_path_planner_astar_batch"}
    assert {row["validation_evidence_kind"] for row in rows} == {"in_process_astar_screening"}
    assert all(row["reachable"] is True for row in rows)
    assert all(row["path_cost"] > 0 for row in rows)
    assert all(row["path_length"] > 0 for row in rows)
    assert all("risk_source" in row for row in rows)
    assert all(row["coverage_validated_by_path_feedback"] is False for row in rows)
    assert not list((tmp_path / "work").glob("path-planner-*.json"))
    assert not list((tmp_path / "work").rglob("path-planner-request.json"))


def test_in_process_batch_astar_keeps_blocked_goal_out_of_formal_candidates(tmp_path) -> None:
    repo_root = _repo_root()
    from scripts.xunce_frontier_nbv_validation import validate_candidate_cells

    contract_path, sidecar_path = _write_tiny_contract_and_sidecar(
        tmp_path,
        passable_mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, False],
        ],
    )

    rows = validate_candidate_cells(
        scenario={"scenario_id": "batch-blocked"},
        proposal_rows=[{"proposal_id": "blocked", "cell": [3, 3], "frontier_candidate_source": "frontier_boundary"}],
        contract_path=contract_path,
        sidecar_path=sidecar_path,
        current_cell=(0, 0),
        repo_root=repo_root,
        output_work_root=tmp_path / "work",
        validation_mode="in_process_path_planner_astar_batch",
    )

    row = rows[0]
    assert row["planner_validation_backend"] == "in_process_path_planner_astar_batch"
    assert row["validation_evidence_kind"] == "validation_failure"
    assert row["proposal_only"] is True
    assert row["proposal_validated_by_path_feedback"] is True
    assert row["reachable"] is False
    assert row["failure_reason"] in {"goal_blocked", "unreachable"}
    assert "proposal_unreachable" in row["validation_diagnostic_flags"]


def _write_tiny_contract_and_sidecar(tmp_path: Path, *, passable_mask: list[list[bool]] | None = None) -> tuple[Path, Path]:
    root = tmp_path / "tiny"
    root.mkdir()
    if passable_mask is None:
        passable_mask = [[True, True, True, True] for _ in range(4)]
    contract = {
        "schema_version": "model-explorer-contract/v1",
        "grid": {
            "width": 4,
            "height": 4,
            "resolution": 1.0,
            "frame_id": "test",
            "origin": [0.0, 0.0],
            "layers": ["cost"],
        },
        "constraints": {"violation_count": 0, "passable_ratio": 1.0, "reason_counts": {}},
        "top_goals": [{"cell": [1, 0], "utility": 1.0, "reachable": True}],
        "top_sequences": [],
        "observation_update": {"coverage_rate_delta": 0.0},
    }
    sidecar = {
        "schema_version": "path-planner-sidecar/v1",
        "cost": [[1.0, 2.0, 3.0, 4.0] for _ in range(4)],
        "passable_mask": passable_mask,
        "metadata": {"fixture": "batch-astar"},
    }
    contract_path = root / "contract.json"
    sidecar_path = root / "sidecar.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return contract_path, sidecar_path
