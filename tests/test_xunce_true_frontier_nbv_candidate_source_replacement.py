import json
import sys
from pathlib import Path


STAGE_ID = "xunce-true-frontier-nbv-candidate-source-replacement"
EXPANSION_SUMMARY = "xunce-high-fidelity-real-map-roi-expansion-summary.json"
SLICES = "xunce-high-fidelity-real-map-slices.jsonl"
PATH_FEEDBACK = "xunce-high-fidelity-path-feedback-audit.json"
BINDING_SUMMARY = "xunce-true-incumbent-selection-binding-summary.json"


def _scripts_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    scripts_path = str(repo_root / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    return repo_root


def test_runner_generates_true_frontier_candidates_and_preserves_source(tmp_path) -> None:
    repo_root = _scripts_on_path()
    stage18a_root = tmp_path / "stage18a"
    binding_root = tmp_path / "binding"
    output_root = tmp_path / "stage18i3"
    config_path = tmp_path / "config.json"
    _write_stage18a(stage18a_root)
    _write_binding(binding_root)
    _write_quantization(config_path.parent / "quantization")
    _write_config(config_path, stage18a_root, binding_root)
    original_stage18a = json.loads((stage18a_root / PATH_FEEDBACK).read_text(encoding="utf-8"))

    from scripts.run_xunce_true_frontier_nbv_candidate_source_replacement import (
        run_xunce_true_frontier_nbv_candidate_source_replacement,
    )

    summary = run_xunce_true_frontier_nbv_candidate_source_replacement(
        config_path=config_path,
        output_root=output_root,
        repo_root=repo_root,
    )

    assert summary["status"] == "passed"
    assert summary["scenario_count"] == 3
    assert summary["proposal_count"] >= summary["validated_candidate_count"]
    assert summary["candidate_count"] == summary["validated_candidate_count"]
    assert summary["candidate_validation_mode"] == "in_process_evaluate_candidate_paths"
    assert summary["valid_candidate_count"] == summary["candidate_count"]
    assert summary["proposal_validation_attempt_count"] >= summary["candidate_count"]
    assert summary["proposal_validation_success_count"] >= summary["candidate_count"]
    assert summary["insufficient_validated_candidate_scenario_count"] == 0
    assert summary["formal_candidate_missing_validation_count"] == 0
    assert summary["frontier_boundary_candidate_count"] > 0 or summary["roi_undercovered_boundary_candidate_count"] > 0
    assert summary["open_grid_fallback_count"] == 0
    assert summary["fallback_action_index_0_count"] == 0
    assert summary["comparison_allowed"] is True
    assert summary["canary_traffic_fraction"] == 0.0
    assert summary["publishes_checkpoint"] is False
    assert summary["starts_online_canary"] is False

    for filename in (
        "xunce-true-frontier-nbv-candidate-source-summary.json",
        "xunce-true-frontier-nbv-proposals.jsonl",
        "xunce-true-frontier-nbv-validated-candidates.jsonl",
        "xunce-true-frontier-nbv-validation-audit.json",
        "xunce-true-frontier-nbv-proposal-validation-summary.json",
        "xunce-true-frontier-nbv-proposal-validation-results.jsonl",
        "xunce-true-frontier-nbv-report.md",
        EXPANSION_SUMMARY,
        SLICES,
        PATH_FEEDBACK,
    ):
        assert (output_root / filename).is_file(), filename

    generated = json.loads((output_root / PATH_FEEDBACK).read_text(encoding="utf-8"))
    assert generated["scenario_count"] == 3
    first_candidates = generated["scenarios"][0]["path_feedback"]["candidates"]
    assert [row["action_index"] for row in first_candidates] == list(range(len(first_candidates)))
    assert {row["frontier_candidate_source"] for row in first_candidates} >= {
        "incumbent_neighborhood",
        "frontier_boundary",
    }
    for row in first_candidates:
        assert row["candidate_generation_source"] == "true_frontier_nbv_candidate_source_replacement/v1"
        assert row["coverage_source"] == "geometric_counterfactual_from_true_frontier_nbv_candidate/v1"
        assert row["coverage_cell_set_kind"] == "path_line_plus_endpoint_union"
        assert row["coverage_dedupe_scope"] == "scenario_step_new_cells"
        assert row["coverage_validated_by_path_feedback"] is False
        assert row["coverage_validation_source"] == "offline_geometric_counterfactual_not_path_feedback"
        assert row["proposal_only"] is False
        assert row["proposal_validated_by_path_feedback"] is True
        assert row["path_feedback_validation_source"] == "in_process_evaluate_candidate_paths/v1"
        assert row["validation_attempt_goal_reachable_seed"] is True
        assert row["planner_reachable"] is True
        assert "path_cost" in row
        assert "risk" in row
        assert "risk_source" in row

    assert json.loads((stage18a_root / PATH_FEEDBACK).read_text(encoding="utf-8")) == original_stage18a


def test_missing_true_binding_fails_to_binding_route(tmp_path) -> None:
    repo_root = _scripts_on_path()
    stage18a_root = tmp_path / "stage18a"
    config_path = tmp_path / "config.json"
    _write_stage18a(stage18a_root)
    _write_quantization(config_path.parent / "quantization")
    _write_config(config_path, stage18a_root, tmp_path / "missing-binding")

    from scripts.run_xunce_true_frontier_nbv_candidate_source_replacement import (
        run_xunce_true_frontier_nbv_candidate_source_replacement,
    )

    summary = run_xunce_true_frontier_nbv_candidate_source_replacement(
        config_path=config_path,
        output_root=tmp_path / "stage18i3",
        repo_root=repo_root,
    )

    assert summary["status"] == "failed"
    assert "missing_true_incumbent_binding" in summary["reason_codes"]
    assert summary["next_required_change"] == "run_true_incumbent_selection_binding"


def test_insufficient_validated_candidates_fail_without_formal_candidates(tmp_path) -> None:
    repo_root = _scripts_on_path()
    stage18a_root = tmp_path / "stage18a"
    binding_root = tmp_path / "binding"
    config_path = tmp_path / "config.json"
    _write_stage18a(stage18a_root, blocked_sidecar=True)
    _write_binding(binding_root)
    _write_quantization(config_path.parent / "quantization")
    _write_config(config_path, stage18a_root, binding_root)

    from scripts.run_xunce_true_frontier_nbv_candidate_source_replacement import (
        run_xunce_true_frontier_nbv_candidate_source_replacement,
    )

    summary = run_xunce_true_frontier_nbv_candidate_source_replacement(
        config_path=config_path,
        output_root=tmp_path / "stage18i3",
        repo_root=repo_root,
    )

    assert summary["status"] == "failed"
    assert summary["validated_candidate_count"] < 9
    assert summary["insufficient_validated_candidate_scenario_count"] == 3
    assert "insufficient_validated_candidates_per_scenario" in summary["reason_codes"]


def test_nonzero_canary_fraction_is_rejected(tmp_path) -> None:
    repo_root = _scripts_on_path()
    stage18a_root = tmp_path / "stage18a"
    binding_root = tmp_path / "binding"
    config_path = tmp_path / "config.json"
    _write_stage18a(stage18a_root)
    _write_binding(binding_root)
    _write_quantization(config_path.parent / "quantization")
    _write_config(config_path, stage18a_root, binding_root, canary_traffic_fraction=0.01)

    from global_99_coverage_contract import ConfigError
    from scripts.run_xunce_true_frontier_nbv_candidate_source_replacement import (
        run_xunce_true_frontier_nbv_candidate_source_replacement,
    )

    try:
        run_xunce_true_frontier_nbv_candidate_source_replacement(
            config_path=config_path,
            output_root=tmp_path / "stage18i3",
            repo_root=repo_root,
        )
    except ConfigError as exc:
        assert "canary_traffic_fraction must be 0.0" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected nonzero canary_traffic_fraction to be rejected")


def test_missing_quantization_summary_fails_to_proposal_repair_route(tmp_path) -> None:
    repo_root = _scripts_on_path()
    stage18a_root = tmp_path / "stage18a"
    binding_root = tmp_path / "binding"
    config_path = tmp_path / "config.json"
    _write_stage18a(stage18a_root)
    _write_binding(binding_root)
    _write_config(config_path, stage18a_root, binding_root)

    from scripts.run_xunce_true_frontier_nbv_candidate_source_replacement import (
        run_xunce_true_frontier_nbv_candidate_source_replacement,
    )

    summary = run_xunce_true_frontier_nbv_candidate_source_replacement(
        config_path=config_path,
        output_root=tmp_path / "stage18i3",
        repo_root=repo_root,
    )

    assert summary["status"] == "failed"
    assert "missing_stage18h0_quantization" in summary["reason_codes"]
    assert summary["next_required_change"] == "repair_true_frontier_nbv_proposal_generation"


def test_stage_registry_has_true_frontier_nbv_stage() -> None:
    repo_root = _scripts_on_path()
    registry = json.loads((repo_root / "configs" / "stage_registry.json").read_text(encoding="utf-8"))
    stage = registry["stages"][STAGE_ID]
    assert stage["script"] == "scripts/run_xunce_true_frontier_nbv_candidate_source_replacement.py"
    assert stage["default_config"] == "configs/xunce_true_frontier_nbv_candidate_source_replacement_v1.json"
    assert "--repo-root" in stage["args"]


def _write_config(config_path: Path, stage18a_root: Path, binding_root: Path, *, canary_traffic_fraction: float = 0.0) -> None:
    payload = {
        "schema_version": "xunce-true-frontier-nbv-candidate-source-replacement-config/v1",
        "source_roi_expansion_root": str(stage18a_root),
        "source_true_incumbent_binding_root": str(binding_root),
        "source_quantization_root": str(config_path.parent / "quantization"),
        "required_scenario_count": 3,
        "max_candidates_per_scenario": 3,
        "proposal_pool_limit_per_scenario": 3,
        "frontier_direction_count": 8,
        "coverage_denominator_cells": 1000,
        "risk_margin": 0.01,
        "cost_margin": 0.0,
        "path_budget_m": 5000.0,
        "candidate_validation_mode": "in_process_evaluate_candidate_paths",
        "min_validated_candidates_per_scenario": 3,
        "debug_validation_artifacts": False,
        "allow_open_grid_fallback": False,
        "canary_traffic_fraction": canary_traffic_fraction,
    }
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_quantization(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / "xunce-risk-coverage-cost-quantization-summary.json",
        {
            "schema_version": "xunce-risk-coverage-cost-quantization-summary/v1",
            "status": "passed",
            "candidate_count": 72,
            "canary_traffic_fraction": 0.0,
        },
    )


def _write_stage18a(root: Path, *, proposal_validated: bool = True, blocked_sidecar: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    scenarios = []
    slices = []
    for index in range(24):
        scenario_id = f"scenario_{index:03d}"
        roi_group = f"roi_{index % 8}"
        base = 0
        start_cell = [base, 0]
        cells = _source_cells_for_runner(base)
        candidates = []
        for action_index, cell in enumerate(cells):
            is_incumbent = cell == [base + 1, 0]
            is_safe_frontier = cell == [base + 2, 0]
            candidates.append(
                {
                    "action_index": action_index,
                    "cell": cell,
                    "reachable": True,
                    "path_cost": 10.0 if is_incumbent else 8.0 if is_safe_frontier else 12.0 + (action_index % 4),
                    "risk": 0.08 if is_incumbent else 0.07 if is_safe_frontier else 0.10 + (action_index % 3) * 0.02,
                    "open_grid_fallback_used": False,
                    "proposal_validated_by_path_feedback": proposal_validated,
                }
            )
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "roi_group": roi_group,
                "scenario_group": roi_group,
                "start_cell": start_cell,
                "roi_bounds": {"min_cell": [0, 0], "max_cell": [6, 6]},
                "open_grid_fallback_used": False,
                "path_feedback": {"candidates": candidates},
            }
        )
        contract_path, sidecar_path = _write_contract_sidecar(root, scenario_id, blocked=blocked_sidecar)
        slices.append(
            {
                "scenario_id": scenario_id,
                "scenario_group": roi_group,
                "roi_name": roi_group,
                "contract": str(contract_path),
                "sidecar": str(sidecar_path),
                "start_cell": start_cell,
            }
        )
    _write_json(root / EXPANSION_SUMMARY, {"schema_version": "xunce-high-fidelity-real-map-roi-expansion-summary/v1", "status": "passed", "slice_count": 24})
    _write_jsonl(root / SLICES, slices)
    _write_json(root / PATH_FEEDBACK, {"schema_version": "xunce-high-fidelity-path-feedback-audit/v1", "scenario_count": 24, "scenarios": scenarios})


def _write_binding(root: Path, *, proposal_validated: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    scenarios = []
    for index in range(24):
        scenario_id = f"scenario_{index:03d}"
        base = 0
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "roi_group": f"roi_{index % 8}",
                "start_cell": [base, 0],
                "incumbent_selected_action_index": 0,
                "incumbent_selection_source": "true_checkpoint_inference",
                "path_feedback": {
                    "candidates": [
                        {
                            "action_index": 0,
                            "cell": [base + 1, 0],
                            "reachable": True,
                            "path_cost": 10.0,
                            "risk": 0.08,
                            "proposal_validated_by_path_feedback": proposal_validated,
                        }
                    ]
                },
            }
        )
    _write_json(
        root / BINDING_SUMMARY,
        {
            "schema_version": "xunce-true-incumbent-selection-binding-summary/v1",
            "status": "passed",
            "true_incumbent_selection_bound": True,
            "fallback_action_index_0_count": 0,
            "candidate_cell_mismatch_count": 0,
        },
    )
    _write_json(root / PATH_FEEDBACK, {"schema_version": "xunce-high-fidelity-path-feedback-audit/v1", "scenario_count": 24, "scenarios": scenarios})


def _source_cells_for_runner(base: int) -> list[list[int]]:
    cells: list[list[int]] = [[base + 1, 0]]
    directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    for radius in (2, 4, 6):
        for dx, dy in directions:
            cells.append([base + dx * radius, dy * radius])
    roi_cells = [[base - 6, -6], [base - 6, 6], [base + 6, -6], [base + 6, 6], [base, 6], [base + 6, 0]]
    for cell in roi_cells:
        if cell not in cells:
            cells.append(cell)
    return cells


def _write_contract_sidecar(root: Path, scenario_id: str, *, blocked: bool = False) -> tuple[Path, Path]:
    sidecar_root = root / "path_planner_sidecars"
    sidecar_root.mkdir(parents=True, exist_ok=True)
    contract_path = sidecar_root / f"{scenario_id}.contract.json"
    sidecar_path = sidecar_root / f"{scenario_id}.path-planner-sidecar.json"
    contract = {
        "schema_version": "model-explorer-contract/v1",
        "grid": {
            "width": 16,
            "height": 16,
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
    passable_mask = [[True for _ in range(16)] for _ in range(16)]
    if blocked:
        passable_mask = [[False for _ in range(16)] for _ in range(16)]
        passable_mask[0][0] = True
    sidecar = {
        "schema_version": "path-planner-sidecar/v1",
        "cost": [[1.0 for _ in range(16)] for _ in range(16)],
        "passable_mask": passable_mask,
        "metadata": {"fixture": scenario_id},
    }
    _write_json(contract_path, contract)
    _write_json(sidecar_path, sidecar)
    return contract_path, sidecar_path


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
