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
    sources = {row["frontier_candidate_source"] for row in proposals}
    assert "coverage_frontier_boundary" in sources
    assert "undercovered_component_centroid" not in sources
    assert "low_cost_bridge_candidate" not in sources
    for row in formal:
        assert row["proposal_only"] is False
        assert row["proposal_validated_by_path_feedback"] is True
        assert row["reachable"] is True
        assert row["open_grid_fallback_used"] is False
        assert row["path_cost"] >= 0
        assert row["risk"] >= 0
        assert row["path_length"] >= 0
        assert row["candidate_generation_source"] == "dynamic_frontier_nbv_in_process/v1"
        assert row["candidate_generation_algorithm_source"] == "map_aware_coverage_frontier_nbv/v1"
        assert row["coverage_source"] == "geometric_counterfactual_from_dynamic_frontier_nbv/v1"
        assert row["coverage_validated_by_path_feedback"] is False
        assert row["coverage_validation_source"] == "offline_geometric_counterfactual_not_path_feedback"
        assert row["validation_batch_hash"]
        assert str(tmp_path / "validation") in row["validation_work_root"]
        assert row["candidate_selection_mode"] == "validated_pareto_diverse"


def test_dynamic_frontier_nbv_coverage_is_clipped_to_passable_roi_cells(tmp_path) -> None:
    repo_root = _scripts_on_path()
    from scripts.xunce_dynamic_frontier_nbv import build_dynamic_frontier_nbv_candidates

    passable_mask = [[True for _ in range(8)] for _ in range(8)]
    for y in range(8):
        passable_mask[y][7] = False
    contract_path, sidecar_path = _write_contract_and_sidecar(tmp_path, passable_mask=passable_mask)
    scenario = {"scenario_id": "dynamic-clipped", "path_feedback": {"candidates": []}}
    slice_row = {
        "scenario_id": "dynamic-clipped",
        "contract": str(contract_path),
        "sidecar": str(sidecar_path),
        "map_source": {"roi": {"width": 8, "height": 8}},
    }

    formal, proposals, validations = build_dynamic_frontier_nbv_candidates(
        scenario=scenario,
        slice_row=slice_row,
        current_cell=(5, 5),
        covered_cells={(5, 5)},
        step_index=0,
        config={**_dynamic_config(), "dynamic_proposal_pool_limit_per_step": 5},
        repo_root=repo_root,
        output_work_root=tmp_path / "validation",
        validation_cache={},
    )

    assert proposals
    assert validations
    assert all((row["cell"][0] != 7) for row in proposals)
    assert formal
    for row in proposals + validations + formal:
        assert row["coverage_denominator_mode"] == "roi_valid_cells"
        assert row["coverage_denominator_source"] == "dynamic_roi_valid_cells/v1"
        assert row["coverage_denominator_cells"] == 56.0


def test_dynamic_frontier_nbv_does_not_fallback_to_all_bounds_without_valid_cells_source(tmp_path) -> None:
    repo_root = _scripts_on_path()
    from scripts.xunce_dynamic_frontier_nbv import build_dynamic_frontier_nbv_candidates

    contract_path, sidecar_path = _write_contract_and_sidecar(tmp_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar.pop("passable_mask")
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    scenario = {"scenario_id": "dynamic-missing-valid-source", "path_feedback": {"candidates": []}}
    slice_row = {
        "scenario_id": "dynamic-missing-valid-source",
        "contract": str(contract_path),
        "sidecar": str(sidecar_path),
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

    assert formal == []
    assert proposals == []
    assert validations
    assert {row["candidate_prefilter_reject_reason"] for row in validations} == {"valid_cells_source_missing"}
    assert all(row["valid_cells_source"] == "missing" for row in validations)
    assert all(row["all_bounds_fallback_used"] is False for row in validations)
    assert all(row["path_validation_attempted"] is False for row in validations)


def test_dynamic_frontier_nbv_prefilter_rejects_blocked_endpoint(tmp_path) -> None:
    repo_root = _scripts_on_path()
    from scripts.xunce_dynamic_frontier_nbv import build_dynamic_frontier_nbv_candidates

    passable_mask = [[True for _ in range(8)] for _ in range(8)]
    passable_mask[0][1] = False
    contract_path, sidecar_path = _write_contract_and_sidecar(tmp_path, passable_mask=passable_mask)
    scenario = {"scenario_id": "dynamic-blocked-endpoint", "path_feedback": {"candidates": []}}
    slice_row = {
        "scenario_id": "dynamic-blocked-endpoint",
        "contract": str(contract_path),
        "sidecar": str(sidecar_path),
        "roi_valid_cells": [[x, y] for y in range(8) for x in range(8)],
        "map_source": {"roi": {"width": 8, "height": 8}},
    }

    _formal, proposals, validations = build_dynamic_frontier_nbv_candidates(
        scenario=scenario,
        slice_row=slice_row,
        current_cell=(0, 0),
        covered_cells={(0, 0)},
        step_index=0,
        config={**_dynamic_config(), "dynamic_path_validation_candidate_budget": 8},
        repo_root=repo_root,
        output_work_root=tmp_path / "validation",
        validation_cache={},
    )

    blocked = [row for row in proposals + validations if row.get("cell") == [1, 0]]
    assert blocked
    assert {row["candidate_prefilter_reject_reason"] for row in blocked} == {"candidate_endpoint_not_passable"}
    assert all(row["candidate_endpoint_passable"] is False for row in blocked)
    assert all(row["candidate_known_free"] is False for row in blocked)
    assert all(row["path_validation_attempted"] is False for row in blocked)


def test_dynamic_frontier_nbv_prefilter_rejects_disconnected_endpoint(tmp_path) -> None:
    repo_root = _scripts_on_path()
    import scripts.xunce_dynamic_frontier_nbv as module

    scenario = {"scenario_id": "dynamic-disconnected", "path_feedback": {"candidates": []}}
    slice_row = {
        "scenario_id": "dynamic-disconnected",
        "map_source": {"roi": {"width": 8, "height": 8}},
    }

    rows = module._proposal_rows(
        scenario=scenario,
        slice_row=slice_row,
        current_cell=(0, 0),
        covered_cells={(0, 0), (5, 5)},
        step_index=0,
        config={**_dynamic_config(), "dynamic_path_validation_candidate_budget": 8},
        repo_root=repo_root,
        sidecar_path=None,
        valid_cells={(0, 0), (0, 1), (6, 5)},
    )

    disconnected = [row for row in rows if row.get("cell") == [6, 5]]
    assert disconnected
    assert disconnected[0]["candidate_endpoint_passable"] is True
    assert disconnected[0]["candidate_known_free"] is True
    assert disconnected[0]["candidate_same_component_as_current"] is False
    assert disconnected[0]["candidate_prefilter_reject_reason"] == "candidate_not_same_component_as_current"
    assert disconnected[0]["path_validation_attempted"] is False


def test_dynamic_frontier_nbv_local_fallback_requires_new_coverage(tmp_path, monkeypatch) -> None:
    repo_root = _scripts_on_path()
    import scripts.xunce_dynamic_frontier_nbv as module

    monkeypatch.setattr(module, "_coverage_frontier_cells", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "_undercovered_component_proposals", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "_conservative_local_cells", lambda *args, **kwargs: [(1, 0)])
    scenario = {"scenario_id": "dynamic-local-zero-new", "path_feedback": {"candidates": []}}
    slice_row = {
        "scenario_id": "dynamic-local-zero-new",
        "map_source": {"roi": {"width": 3, "height": 3}},
    }

    rows = module._proposal_rows(
        scenario=scenario,
        slice_row=slice_row,
        current_cell=(0, 0),
        covered_cells={(0, 0), (1, 0)},
        step_index=0,
        config={**_dynamic_config(), "dynamic_path_validation_candidate_budget": 1, "coverage_radius_cells": 0},
        repo_root=repo_root,
        sidecar_path=None,
        valid_cells={(0, 0), (1, 0)},
    )

    assert len(rows) == 1
    assert rows[0]["frontier_candidate_source"] == "conservative_local_candidate"
    assert rows[0]["expected_new_coverage_cell_count"] == 0.0
    assert rows[0]["candidate_prefilter_reject_reason"] == "expected_new_coverage_cell_count_zero"
    assert rows[0]["path_validation_attempted"] is False


def test_dynamic_frontier_nbv_prefilter_limits_path_validation_budget(tmp_path, monkeypatch) -> None:
    repo_root = _scripts_on_path()
    import scripts.xunce_dynamic_frontier_nbv as module

    contract_path, sidecar_path = _write_contract_and_sidecar(tmp_path)
    scenario = {"scenario_id": "dynamic-budget", "path_feedback": {"candidates": []}}
    slice_row = {
        "scenario_id": "dynamic-budget",
        "contract": str(contract_path),
        "sidecar": str(sidecar_path),
        "map_source": {"roi": {"width": 8, "height": 8}},
    }
    captured: dict[str, list[dict]] = {}

    def fake_validate_candidate_cells(**kwargs):
        proposals = [dict(row) for row in kwargs["proposal_rows"]]
        captured["proposals"] = proposals
        return [
            {
                **row,
                "proposal_only": False,
                "proposal_validated_by_path_feedback": True,
                "reachable": True,
                "planner_reachable": True,
                "open_grid_fallback_used": False,
                "path_cost": float(index + 1),
                "path_length": float(index + 1),
                "risk": 0.0,
                "path_feedback_validation_source": "fake",
                "planner_validation_backend": "fake",
                "validation_evidence_kind": "fake",
            }
            for index, row in enumerate(proposals)
        ]

    monkeypatch.setattr(module, "validate_candidate_cells", fake_validate_candidate_cells)

    formal, proposals, validations = module.build_dynamic_frontier_nbv_candidates(
        scenario=scenario,
        slice_row=slice_row,
        current_cell=(0, 0),
        covered_cells={(0, 0)},
        step_index=0,
        config={**_dynamic_config(), "dynamic_proposal_pool_limit_per_step": 8, "dynamic_path_validation_candidate_budget": 3},
        repo_root=repo_root,
        output_work_root=tmp_path / "validation",
        validation_cache={},
    )

    assert len(proposals) > 3
    assert len(captured["proposals"]) == 3
    assert all(row["path_validation_attempted"] is True for row in captured["proposals"])
    assert all(row["candidate_prefilter_reject_reason"] is None for row in captured["proposals"])
    assert len(validations) == len(proposals)
    assert sum(1 for row in validations if row["path_validation_attempted"] is True) == 3
    assert sum(1 for row in validations if row["path_validation_attempted"] is False) == len(proposals) - 3
    assert formal


def test_dynamic_frontier_nbv_small_pool_uses_undercovered_boundary_as_supplement(tmp_path) -> None:
    repo_root = _scripts_on_path()
    from scripts.xunce_dynamic_frontier_nbv import build_dynamic_frontier_nbv_candidates

    contract_path, sidecar_path = _write_contract_and_sidecar(tmp_path)
    scenario = {"scenario_id": "dynamic-small-pool", "path_feedback": {"candidates": []}}
    slice_row = {
        "scenario_id": "dynamic-small-pool",
        "contract": str(contract_path),
        "sidecar": str(sidecar_path),
        "map_source": {"roi": {"width": 8, "height": 8}},
    }

    _formal, proposals, _validations = build_dynamic_frontier_nbv_candidates(
        scenario=scenario,
        slice_row=slice_row,
        current_cell=(0, 0),
        covered_cells={(0, 0), (0, 1)},
        step_index=0,
        config={**_dynamic_config(), "dynamic_proposal_pool_limit_per_step": 3},
        repo_root=repo_root,
        output_work_root=tmp_path / "validation",
        validation_cache={},
    )

    sources = {row["frontier_candidate_source"] for row in proposals}
    assert "coverage_frontier_boundary" in sources
    assert "undercovered_component_centroid" not in sources
    assert "low_cost_bridge_candidate" not in sources


def test_dynamic_frontier_nbv_prefers_frontier_and_disables_bridge_by_default(tmp_path, monkeypatch) -> None:
    repo_root = _scripts_on_path()
    import scripts.xunce_dynamic_frontier_nbv as module

    contract_path, sidecar_path = _write_contract_and_sidecar(tmp_path)
    scenario = {"scenario_id": "dynamic-frontier-first", "path_feedback": {"candidates": []}}
    slice_row = {
        "scenario_id": "dynamic-frontier-first",
        "contract": str(contract_path),
        "sidecar": str(sidecar_path),
        "map_source": {"roi": {"width": 8, "height": 8}},
    }
    captured: dict[str, list[dict]] = {}

    def fake_validate_candidate_cells(**kwargs):
        captured["proposals"] = [dict(row) for row in kwargs["proposal_rows"]]
        return [
            {
                **row,
                "proposal_only": False,
                "proposal_validated_by_path_feedback": True,
                "reachable": True,
                "planner_reachable": True,
                "open_grid_fallback_used": False,
                "path_cost": 1.0,
                "path_length": 1.0,
                "risk": 0.0,
                "path_feedback_validation_source": "fake",
                "planner_validation_backend": "fake",
                "validation_evidence_kind": "fake",
            }
            for row in kwargs["proposal_rows"]
        ]

    monkeypatch.setattr(module, "validate_candidate_cells", fake_validate_candidate_cells)

    _formal, proposals, _validations = module.build_dynamic_frontier_nbv_candidates(
        scenario=scenario,
        slice_row=slice_row,
        current_cell=(0, 0),
        covered_cells={(0, 0), (0, 1)},
        step_index=0,
        config={**_dynamic_config(), "dynamic_proposal_pool_limit_per_step": 8, "dynamic_path_validation_candidate_budget": 5},
        repo_root=repo_root,
        output_work_root=tmp_path / "validation",
        validation_cache={},
    )

    assert "low_cost_bridge_candidate" not in {row["frontier_candidate_source"] for row in proposals}
    assert captured["proposals"]
    assert captured["proposals"][0]["frontier_candidate_source"] == "coverage_frontier_boundary"
    assert all(row["candidate_quality"] is not None for row in captured["proposals"])


def test_dynamic_frontier_nbv_roi_weight_provenance_from_config_map(tmp_path) -> None:
    repo_root = _scripts_on_path()
    from scripts.xunce_dynamic_frontier_nbv import build_dynamic_frontier_nbv_candidates

    contract_path, sidecar_path = _write_contract_and_sidecar(tmp_path)
    scenario = {"scenario_id": "dynamic-roi-weight", "roi_group": "roi_0", "path_feedback": {"candidates": []}}
    slice_row = {
        "scenario_id": "dynamic-roi-weight",
        "contract": str(contract_path),
        "sidecar": str(sidecar_path),
        "map_source": {"roi": {"width": 8, "height": 8}},
    }
    config = {**_dynamic_config(), "roi_group_weight_map": {"roi_0": 2.5}}

    formal, proposals, validations = build_dynamic_frontier_nbv_candidates(
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

    assert formal
    for row in proposals + validations + formal:
        assert row["roi_weight"] == 2.5
        assert row["roi_weight_source"] == "config.roi_group_weight_map"
        assert row["roi_weighted_coverage_delta"] == row["expected_coverage_rate_delta"] * 2.5


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


def test_dynamic_frontier_nbv_missing_contract_without_valid_source_stops_before_proposal_pool(tmp_path) -> None:
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

    assert proposals == []
    assert not formal
    assert validations
    assert all(row["proposal_only"] is True for row in validations)
    assert all(row["proposal_validated_by_path_feedback"] is False for row in validations)
    assert {row["candidate_prefilter_reject_reason"] for row in validations} == {"valid_cells_source_missing"}


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
        "coverage_denominator_mode": "roi_valid_cells",
        "allow_open_grid_fallback": False,
        "path_budget_m": 5000.0,
    }


def _write_contract_and_sidecar(tmp_path: Path, *, passable_mask: list[list[bool]] | None = None) -> tuple[Path, Path]:
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
        "passable_mask": passable_mask or [[True for _ in range(8)] for _ in range(8)],
        "metadata": {"fixture": "dynamic"},
    }
    contract_path = tmp_path / "contract.json"
    sidecar_path = tmp_path / "sidecar.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return contract_path, sidecar_path
