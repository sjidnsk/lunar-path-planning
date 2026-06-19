import json
import sys
from pathlib import Path


def _scripts_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    scripts_path = str(repo_root / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    return repo_root


def test_dynamic_frontier_nbv_builds_validated_state_conditioned_candidates(tmp_path) -> None:
    repo_root = _scripts_on_path()
    from scripts.xunce_dynamic_frontier_nbv import build_dynamic_frontier_nbv_candidates, candidate_set_hash, covered_cells_hash

    contract_path, sidecar_path = _write_contract_and_sidecar(tmp_path)
    scenario = {"scenario_id": "dynamic-tiny", "path_feedback": {"candidates": []}}
    slice_row = {
        "scenario_id": "dynamic-tiny",
        "contract": str(contract_path),
        "sidecar": str(sidecar_path),
        "map_source": {"roi": {"width": 8, "height": 8}},
    }
    config = _dynamic_config()
    covered = {(0, 0), (0, 1)}

    formal, proposals, validations = build_dynamic_frontier_nbv_candidates(
        scenario=scenario,
        slice_row=slice_row,
        current_cell=(0, 0),
        covered_cells=covered,
        step_index=0,
        config=config,
        repo_root=repo_root,
        output_work_root=tmp_path / "validation",
        validation_cache={},
    )

    assert proposals
    assert validations
    assert formal
    assert candidate_set_hash(formal)
    assert covered_cells_hash(covered)
    assert {row["frontier_candidate_source"] for row in proposals} >= {"frontier_boundary", "roi_undercovered_boundary"}
    for row in formal:
        assert row["proposal_only"] is False
        assert row["proposal_validated_by_path_feedback"] is True
        assert row["reachable"] is True
        assert row["open_grid_fallback_used"] is False
        assert row["path_cost"] >= 0
        assert row["risk"] >= 0
        assert row["path_length"] >= 0
        assert row["candidate_generation_source"] == "dynamic_frontier_nbv_in_process/v1"
        assert row["coverage_source"] == "geometric_counterfactual_from_dynamic_frontier_nbv/v1"
        assert row["coverage_validated_by_path_feedback"] is False
        assert row["coverage_validation_source"] == "offline_geometric_counterfactual_not_path_feedback"
        assert row["validation_batch_hash"]
        assert str(tmp_path / "validation") in row["validation_work_root"]


def test_dynamic_frontier_nbv_changes_candidates_when_state_changes(tmp_path) -> None:
    repo_root = _scripts_on_path()
    from scripts.xunce_dynamic_frontier_nbv import build_dynamic_frontier_nbv_candidates, candidate_set_hash

    contract_path, sidecar_path = _write_contract_and_sidecar(tmp_path)
    scenario = {"scenario_id": "dynamic-state", "path_feedback": {"candidates": []}}
    slice_row = {
        "scenario_id": "dynamic-state",
        "contract": str(contract_path),
        "sidecar": str(sidecar_path),
        "map_source": {"roi": {"width": 8, "height": 8}},
    }
    config = _dynamic_config()

    first, _first_proposals, _first_validations = build_dynamic_frontier_nbv_candidates(
        scenario=scenario,
        slice_row=slice_row,
        current_cell=(0, 0),
        covered_cells={(0, 0)},
        step_index=0,
        config=config,
        repo_root=repo_root,
        output_work_root=tmp_path / "validation",
        validation_cache={},
    )
    second, _second_proposals, _second_validations = build_dynamic_frontier_nbv_candidates(
        scenario=scenario,
        slice_row=slice_row,
        current_cell=(2, 0),
        covered_cells={(0, 0), (1, 0), (2, 0)},
        step_index=1,
        config=config,
        repo_root=repo_root,
        output_work_root=tmp_path / "validation",
        validation_cache={},
    )

    assert candidate_set_hash(first) != candidate_set_hash(second)
    assert [row["cell"] for row in first] != [row["cell"] for row in second]


def test_dynamic_frontier_nbv_missing_contract_keeps_proposals_out_of_formal_candidates(tmp_path) -> None:
    repo_root = _scripts_on_path()
    from scripts.xunce_dynamic_frontier_nbv import build_dynamic_frontier_nbv_candidates

    scenario = {"scenario_id": "missing-contract", "path_feedback": {"candidates": []}}
    slice_row = {
        "scenario_id": "missing-contract",
        "contract": str(tmp_path / "missing.contract.json"),
        "sidecar": str(tmp_path / "missing.sidecar.json"),
        "map_source": {"roi": {"width": 8, "height": 8}},
    }

    formal, proposals, validations = build_dynamic_frontier_nbv_candidates(
        scenario=scenario,
        slice_row=slice_row,
        current_cell=(0, 0),
        covered_cells={(0, 0)},
        step_index=0,
        config=_dynamic_config(),
        repo_root=repo_root,
        output_work_root=tmp_path / "validation",
        validation_cache={},
    )

    assert proposals
    assert not formal
    assert validations
    assert all(row["proposal_only"] is True for row in validations)
    assert all(row["proposal_validated_by_path_feedback"] is False for row in validations)
    assert {row["path_feedback_validation_source"] for row in validations} == {"dynamic_contract_or_sidecar_missing"}


def test_dynamic_frontier_nbv_roi_group_source_order() -> None:
    _scripts_on_path()
    from scripts.xunce_dynamic_frontier_nbv import _roi_group, resolve_roi_group

    assert resolve_roi_group({"roi_group": "scenario_roi"}, {"roi_group": "slice_roi"}) == "scenario_roi"
    assert resolve_roi_group({}, {"roi_group": "slice_roi"}) == "slice_roi"
    assert resolve_roi_group({}, {"roi_name": "slice_roi_name"}) == "slice_roi_name"
    assert resolve_roi_group({"scenario_group": "scenario_group_roi"}, {}) == "scenario_group_roi"
    assert resolve_roi_group({}, {"metadata": {"roi_group": "metadata_roi"}}) == "metadata_roi"
    assert resolve_roi_group({}, {}) == "unknown"
    assert _roi_group({"roi_group": "scenario_roi"}, {"roi_group": "slice_roi"}) == "scenario_roi"


def test_dynamic_frontier_nbv_roi_group_propagates_to_all_generated_rows(tmp_path) -> None:
    repo_root = _scripts_on_path()
    from scripts.xunce_dynamic_frontier_nbv import build_dynamic_frontier_nbv_candidates

    contract_path, sidecar_path = _write_contract_and_sidecar(tmp_path)
    variants = [
        ({"scenario_id": "scenario-roi", "roi_group": "scenario_roi"}, {"scenario_id": "scenario-roi"}, "scenario_roi"),
        ({}, {"scenario_id": "slice-roi", "roi_group": "slice_roi"}, "slice_roi"),
        ({}, {"scenario_id": "slice-name", "roi_name": "slice_roi_name"}, "slice_roi_name"),
        ({"scenario_id": "scenario-group", "scenario_group": "scenario_group_roi"}, {"scenario_id": "scenario-group"}, "scenario_group_roi"),
        ({}, {"scenario_id": "metadata-roi", "metadata": {"roi_group": "metadata_roi"}}, "metadata_roi"),
    ]
    for scenario_base, slice_base, expected_roi in variants:
        scenario = {"scenario_id": slice_base.get("scenario_id", "scenario"), "path_feedback": {"candidates": []}, **scenario_base}
        slice_row = {
            "contract": str(contract_path),
            "sidecar": str(sidecar_path),
            "map_source": {"roi": {"width": 8, "height": 8}},
            **slice_base,
        }
        formal, proposals, validations = build_dynamic_frontier_nbv_candidates(
            scenario=scenario,
            slice_row=slice_row,
            current_cell=(0, 0),
            covered_cells={(0, 0)},
            step_index=0,
            config=_dynamic_config(),
            repo_root=repo_root,
            output_work_root=tmp_path / "validation" / expected_roi,
            validation_cache={},
        )

        assert proposals
        assert validations
        assert formal
        for row in proposals + validations + formal:
            assert row["roi_group"] == expected_roi
            assert row["roi_group"] != "unknown"


def _dynamic_config() -> dict:
    return {
        "candidate_refresh_mode": "dynamic_frontier_nbv_in_process",
        "dynamic_candidate_validation_mode": "in_process_path_planner_astar_batch",
        "dynamic_frontier_radius_cells": [1, 2],
        "dynamic_frontier_direction_count": 8,
        "dynamic_proposal_pool_limit_per_step": 8,
        "dynamic_max_candidates_per_step": 3,
        "dynamic_validation_cache_enabled": True,
        "dynamic_validation_work_root": "outputs/_xunce_dynamic_validation_work",
        "dynamic_validation_max_path_length": 1000,
        "dynamic_sidecar_fallback_mode": "diagnostic_only",
        "debug_validation_artifacts": False,
        "coverage_radius_cells": 1,
        "coverage_metric_mode": "path_line_plus_endpoint",
        "coverage_denominator_cells": 1000,
        "allow_open_grid_fallback": False,
        "path_budget_m": 5000.0,
    }


def _write_contract_and_sidecar(tmp_path: Path) -> tuple[Path, Path]:
    contract = {
        "schema_version": "model-explorer-contract/v1",
        "grid": {
            "width": 8,
            "height": 8,
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
        "cost": [[1.0 for _ in range(8)] for _ in range(8)],
        "passable_mask": [[True for _ in range(8)] for _ in range(8)],
        "metadata": {"fixture": "dynamic"},
    }
    contract_path = tmp_path / "contract.json"
    sidecar_path = tmp_path / "sidecar.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return contract_path, sidecar_path
