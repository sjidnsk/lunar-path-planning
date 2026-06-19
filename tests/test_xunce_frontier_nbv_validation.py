import sys
import json
from pathlib import Path


def _scripts_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    scripts_path = str(repo_root / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    return repo_root


def test_exact_cell_match_formalizes_with_path_feedback_metrics(tmp_path) -> None:
    _scripts_on_path()
    from scripts.xunce_frontier_nbv_validation import validate_frontier_nbv_proposals

    scenario = {
        "scenario_id": "scenario_000",
        "path_feedback": {
            "candidates": [
                {
                    "action_index": 3,
                    "cell": [10, 5],
                    "reachable": True,
                    "path_cost": 12.5,
                    "risk": 0.07,
                    "open_grid_fallback_used": False,
                    "proposal_validated_by_path_feedback": True,
                }
            ]
        },
    }
    proposals = [{"proposal_id": "p-1", "cell": [10, 5], "proposal_only": True}]

    rows = validate_frontier_nbv_proposals(
        scenario=scenario,
        proposal_rows=proposals,
        repo_root=_scripts_on_path(),
        output_work_root=tmp_path,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["proposal_only"] is False
    assert row["proposal_validated_by_path_feedback"] is True
    assert row["path_feedback_validation_source"] == "deterministic_exact_cell_match/stage18a_path_feedback"
    assert row["source_action_index"] == 3
    assert row["reachable"] is True
    assert row["path_cost"] == 12.5
    assert row["risk"] == 0.07
    assert row["open_grid_fallback_used"] is False


def test_nonmatching_cell_stays_proposal_only_without_metric_fabrication(tmp_path) -> None:
    _scripts_on_path()
    from scripts.xunce_frontier_nbv_validation import validate_frontier_nbv_proposals

    scenario = {
        "scenario_id": "scenario_000",
        "path_feedback": {
            "candidates": [
                {
                    "action_index": 0,
                    "cell": [1, 1],
                    "reachable": True,
                    "path_cost": 3.0,
                    "risk": 0.02,
                }
            ]
        },
    }
    proposals = [{"proposal_id": "p-2", "cell": [9, 9], "proposal_only": True}]

    rows = validate_frontier_nbv_proposals(
        scenario=scenario,
        proposal_rows=proposals,
        repo_root=_scripts_on_path(),
        output_work_root=tmp_path,
    )

    row = rows[0]
    assert row["proposal_only"] is True
    assert row["proposal_validated_by_path_feedback"] is False
    assert row["path_feedback_validation_source"] == "missing_exact_cell_match"
    assert "path_cost" not in row
    assert "risk" not in row
    assert "reachable" not in row


def test_in_process_validation_formalizes_arbitrary_cell_with_planner_metrics(tmp_path) -> None:
    repo_root = _scripts_on_path()
    from scripts.xunce_frontier_nbv_validation import validate_candidate_cells

    contract_path, sidecar_path = _write_tiny_contract_and_sidecar(tmp_path)
    proposals = [
        {
            "proposal_id": "frontier-1",
            "cell": [2, 0],
            "frontier_candidate_source": "frontier_boundary",
            "roi_weighted_coverage_delta": 0.1,
            "coverage_source": "geometric_counterfactual_from_true_frontier_nbv_candidate/v1",
            "coverage_validated_by_path_feedback": True,
            "coverage_validation_source": "incorrect_planner_claim",
        }
    ]

    rows = validate_candidate_cells(
        scenario={"scenario_id": "tiny"},
        proposal_rows=proposals,
        contract_path=contract_path,
        sidecar_path=sidecar_path,
        current_cell=(0, 0),
        repo_root=repo_root,
        output_work_root=tmp_path / "work",
    )

    row = rows[0]
    assert row["proposal_only"] is False
    assert row["proposal_validated_by_path_feedback"] is True
    assert row["path_feedback_validation_source"] == "in_process_evaluate_candidate_paths/v1"
    assert row["validation_attempt_goal_reachable_seed"] is True
    assert row["planner_reachable"] is True
    assert row["reachable"] is True
    assert row["path_cost"] > 0
    assert row["path_length"] > 0
    assert "risk" in row
    assert row["risk_source"] in {
        "goal_experimental_fallback",
        "sidecar_cost_proxy_no_path_risk",
    }
    assert row["coverage_source"] == "geometric_counterfactual_from_true_frontier_nbv_candidate/v1"
    assert row["coverage_validated_by_path_feedback"] is False
    assert row["coverage_validation_source"] == "offline_geometric_counterfactual_not_path_feedback"
    assert not (tmp_path / "work" / "proposal-validation.contract.json").exists()
    assert not (tmp_path / "work" / "_proposal_validation_runtime").exists()
    assert not list((tmp_path / "work").glob("xunce-proposal-validation-*"))


def test_in_process_validation_keeps_unreachable_proposal_out_of_formal_set(tmp_path) -> None:
    repo_root = _scripts_on_path()
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
    proposals = [{"proposal_id": "blocked", "cell": [3, 3], "frontier_candidate_source": "frontier_boundary"}]

    rows = validate_candidate_cells(
        scenario={"scenario_id": "blocked"},
        proposal_rows=proposals,
        contract_path=contract_path,
        sidecar_path=sidecar_path,
        current_cell=(0, 0),
        repo_root=repo_root,
        output_work_root=tmp_path / "work",
    )

    row = rows[0]
    assert row["proposal_only"] is True
    assert row["proposal_validated_by_path_feedback"] is True
    assert row["planner_reachable"] is False
    assert row["reachable"] is False
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
        "cost": [[1.0, 1.0, 1.0, 1.0] for _ in range(4)],
        "passable_mask": passable_mask,
        "metadata": {"fixture": "tiny"},
    }
    contract_path = root / "contract.json"
    sidecar_path = root / "sidecar.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return contract_path, sidecar_path
