from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage24_2_runner_passes_and_gates_hybrid_path_cost_contract(tmp_path: Path) -> None:
    import scripts.run_xunce_stage24_2_hybrid_astar_reward_path_cost_contract as s24_2

    stage24_1_root = _write_stage24_1_root(tmp_path)
    config = _write_config(tmp_path, stage24_1_root)

    summary = s24_2.run_xunce_stage24_2_hybrid_astar_reward_path_cost_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    rows = _read_jsonl(tmp_path / "out" / "xunce-stage24-2-reward-replay.jsonl")
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke"
    assert summary["reward_replay_row_count"] == 2
    assert summary["hybrid_astar_path_cost_reward_contract_count"] == 2
    assert summary["hybrid_astar_path_cost_contract_missing_count"] == 0
    assert summary["hybrid_grid_cost_material_row_count"] == 1
    assert summary["stage21_3_rejects_grid_only_path_cost_batch"] is True
    assert rows[0]["coverage_source"] == "endpoint_theta_slope_obstacle_los/v1"
    assert rows[0]["path_cost_source"] == "hybrid_astar_pose_path/v1"
    assert rows[0]["metrics"]["path_cost_m"] == 2.5
    assert rows[0]["legacy_grid_astar_path_cost"] == 10.0
    assert rows[0]["point_grid_path_cost_fallback_used"] is False
    assert rows[0]["default_astar_replaced"] is False
    assert rows[0]["hybrid_astar_ackermann_feasible_claimed"] is False
    assert (tmp_path / "out" / "xunce-stage24-2-stage21-3-batch-gate-audit.json").is_file()


def test_stage24_2_routes_missing_hybrid_path_provenance(tmp_path: Path) -> None:
    import scripts.run_xunce_stage24_2_hybrid_astar_reward_path_cost_contract as s24_2

    stage24_1_root = _write_stage24_1_root(tmp_path, omit_hybrid_path_hash=True)
    config = _write_config(tmp_path, stage24_1_root)

    summary = s24_2.run_xunce_stage24_2_hybrid_astar_reward_path_cost_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage24_2_reward_path_cost_source"
    assert "hybrid_astar_path_cost_contract_missing" in summary["blocking_reason_codes"]


def test_stage24_2_routes_stage24_1_input_mismatch(tmp_path: Path) -> None:
    import scripts.run_xunce_stage24_2_hybrid_astar_reward_path_cost_contract as s24_2

    stage24_1_root = _write_stage24_1_root(tmp_path, status="failed")
    config = _write_config(tmp_path, stage24_1_root)

    summary = s24_2.run_xunce_stage24_2_hybrid_astar_reward_path_cost_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage24_2_required_inputs"
    assert "stage24_1_not_passed" in summary["blocking_reason_codes"]


def test_stage24_2_writes_failed_summary_when_stage24_1_summary_missing(tmp_path: Path) -> None:
    import scripts.run_xunce_stage24_2_hybrid_astar_reward_path_cost_contract as s24_2

    stage24_1_root = tmp_path / "stage24_1_missing_summary"
    stage24_1_root.mkdir()
    _write_jsonl(stage24_1_root / "xunce-stage24-1-candidate-hybrid-path-cost-audit.jsonl", [])
    (stage24_1_root / "xunce-stage24-1-path-cost-source-recommendation.json").write_text(
        json.dumps(
            {
                "recommended_path_cost_source": "hybrid_astar_pose_path/v1",
                "default_astar_replaced": False,
                "ackermann_feasible_claimed": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = _write_config(tmp_path, stage24_1_root)

    summary = s24_2.run_xunce_stage24_2_hybrid_astar_reward_path_cost_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage24_2_required_inputs"
    assert "missing_stage24_1_summary" in summary["blocking_reason_codes"]
    assert (tmp_path / "out" / "xunce-stage24-2-summary.json").is_file()


def test_stage24_2_routes_boundary_rejection(tmp_path: Path) -> None:
    import scripts.run_xunce_stage24_2_hybrid_astar_reward_path_cost_contract as s24_2

    stage24_1_root = _write_stage24_1_root(tmp_path)
    config = _write_config(tmp_path, stage24_1_root, publishes_checkpoint=True)

    summary = s24_2.run_xunce_stage24_2_hybrid_astar_reward_path_cost_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage24_2_boundary_rejections"
    assert "publishes_checkpoint_enabled" in summary["blocking_reason_codes"]


def _write_stage24_1_root(
    tmp_path: Path,
    *,
    status: str = "passed",
    omit_hybrid_path_hash: bool = False,
) -> Path:
    root = tmp_path / "stage24_1"
    root.mkdir()
    summary = {
        "schema_version": "xunce-stage24-1-summary/v1",
        "stage_id": "xunce-stage24-1-hybrid-astar-candidate-path-cost-integration",
        "status": status,
        "next_required_change": "run_stage24_2_hybrid_astar_reward_path_cost_contract",
        "path_cost_source_recommendation": "hybrid_astar_pose_path/v1",
        "hybrid_vs_grid_cost_material": True,
        "material_cost_delta_count": 1,
        "default_astar_replaced": False,
        "ackermann_feasible_claimed": False,
        "platform_contract_hash": "platform-hash",
        "max_traversable_slope_deg": 30.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
    }
    (root / "xunce-stage24-1-summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    recommendation = {
        "schema_version": "xunce-stage24-1-path-cost-source-recommendation/v1",
        "recommended_path_cost_source": "hybrid_astar_pose_path/v1",
        "default_astar_replaced": False,
        "ackermann_feasible_claimed": False,
    }
    (root / "xunce-stage24-1-path-cost-source-recommendation.json").write_text(
        json.dumps(recommendation, ensure_ascii=False),
        encoding="utf-8",
    )
    rows = [
        _candidate_row(0, hybrid_path_hash=None if omit_hybrid_path_hash else "hybrid-hash-0", hybrid_cost=2.5, grid_cost=10.0),
        _candidate_row(1, hybrid_path_hash="hybrid-hash-1", hybrid_cost=3.0, grid_cost=3.0),
    ]
    _write_jsonl(root / "xunce-stage24-1-candidate-hybrid-path-cost-audit.jsonl", rows)
    return root


def _candidate_row(index: int, *, hybrid_path_hash: str | None, hybrid_cost: float, grid_cost: float) -> dict:
    return {
        "schema_version": "xunce-stage24-1-candidate-hybrid-path-cost-audit-row/v1",
        "scenario_id": "stage24-smoke",
        "step_index": 0,
        "candidate_index": index,
        "candidate_cell": [3 + index, 1],
        "candidate_viewpoint": [3 + index, 1, 90],
        "candidate_theta_deg": 90,
        "candidate_set_hash": "candidate-set-with-theta",
        "candidate_pose_contract_valid": True,
        "hybrid_astar_reachable": True,
        "hybrid_astar_trajectory_kind": "hybrid_astar_pose_path",
        "hybrid_astar_path_cost": hybrid_cost,
        "hybrid_astar_pose_path_hash": hybrid_path_hash,
        "hybrid_astar_failure_reason": None,
        "legacy_grid_astar_path_cost": grid_cost,
        "hybrid_vs_grid_path_cost_delta": hybrid_cost - grid_cost,
        "path_cost_source_recommendation": "hybrid_astar_pose_path/v1",
        "default_astar_replaced": False,
        "hybrid_astar_ackermann_feasible_claimed": False,
        "slope_obstacle_source_hash": "slope-source-hash",
        "platform_contract_hash": "platform-hash",
        "max_traversable_slope_deg": 30.0,
        "slope_blocked_source_kind": "slope_blocked_as_obstacle_proxy",
    }


def _write_config(tmp_path: Path, stage24_1_root: Path, **overrides) -> Path:
    payload = {
        "schema_version": "xunce-stage24-2-hybrid-astar-reward-path-cost-contract-config/v1",
        "stage24_1_root": str(stage24_1_root),
        "coverage_first_reward_profile": "configs/xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json",
        "coverage_denominator_cells": 100.0,
        "replay_obstacle_aware_new_visible_cell_count": 4,
        "stage24_2_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage24_2_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
