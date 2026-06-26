from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "scripts", REPO_ROOT / "path-planner" / "src"):
    value = str(_path)
    if value not in sys.path:
        sys.path.insert(0, value)


def test_candidate_helper_emits_pose_cost_without_replacing_grid_astar() -> None:
    from path_planner.core import CostGrid, GridSpec
    from scripts.xunce_hybrid_astar_candidate_path_cost import evaluate_hybrid_astar_candidate_path_cost

    passable = np.ones((5, 5), dtype=bool)
    grid = CostGrid(GridSpec(width=5, height=5, resolution=1.0), np.ones((5, 5)), passable)

    row = evaluate_hybrid_astar_candidate_path_cost(
        grid=grid,
        current_pose=[1.5, 1.5, 0.0],
        candidate={
            "scenario_id": "smoke",
            "step_index": 0,
            "candidate_index": 0,
            "candidate_set_hash": "hash-with-theta",
            "candidate_viewpoint": [3, 1, 90],
            "candidate_theta_deg": 90,
            "path_cost": 2.0,
        },
        platform_contract_hash="platform-hash",
        max_traversable_slope_deg=30.0,
    )

    assert row["hybrid_astar_reachable"] is True
    assert row["hybrid_astar_trajectory_kind"] == "hybrid_astar_pose_path"
    assert row["hybrid_astar_path_cost"] is not None
    assert row["hybrid_astar_rotation_cost"] > 0.0
    assert row["legacy_grid_astar_path_cost"] is not None
    assert row["hybrid_vs_grid_path_cost_delta"] is not None
    assert row["path_cost_source_recommendation"] == "hybrid_astar_pose_path/v1"
    assert row["hybrid_astar_ackermann_feasible_claimed"] is False
    assert row["default_astar_replaced"] is False
    assert row["platform_contract_hash"] == "platform-hash"
    assert row["max_traversable_slope_deg"] == 30.0


def test_candidate_helper_rejects_point_only_candidate_contract() -> None:
    from path_planner.core import CostGrid, GridSpec
    from scripts.xunce_hybrid_astar_candidate_path_cost import evaluate_hybrid_astar_candidate_path_cost

    grid = CostGrid(GridSpec(width=5, height=5, resolution=1.0), np.ones((5, 5)), np.ones((5, 5), dtype=bool))

    row = evaluate_hybrid_astar_candidate_path_cost(
        grid=grid,
        current_pose=[1.0, 1.0, 0.0],
        candidate={"scenario_id": "bad", "candidate_cell": [2, 2], "candidate_set_hash": "point-only"},
        platform_contract_hash="platform-hash",
        max_traversable_slope_deg=30.0,
    )

    assert row["hybrid_astar_reachable"] is False
    assert row["hybrid_astar_failure_reason"] == "candidate_viewpoint_missing"
    assert row["candidate_pose_contract_valid"] is False


def test_candidate_helper_rejects_theta_mismatch() -> None:
    from path_planner.core import CostGrid, GridSpec
    from scripts.xunce_hybrid_astar_candidate_path_cost import evaluate_hybrid_astar_candidate_path_cost

    grid = CostGrid(GridSpec(width=5, height=5, resolution=1.0), np.ones((5, 5)), np.ones((5, 5), dtype=bool))

    row = evaluate_hybrid_astar_candidate_path_cost(
        grid=grid,
        current_pose=[1.0, 1.0, 0.0],
        candidate={
            "scenario_id": "bad-theta",
            "candidate_set_hash": "hash-with-theta",
            "candidate_viewpoint": [3, 1, 90],
            "candidate_theta_deg": 0,
        },
        platform_contract_hash="platform-hash",
        max_traversable_slope_deg=30.0,
    )

    assert row["candidate_pose_contract_valid"] is False
    assert row["hybrid_astar_failure_reason"] == "candidate_theta_mismatch"


def test_stage24_1_runner_passes_and_writes_expected_artifacts(tmp_path: Path) -> None:
    import scripts.run_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration as s24_1

    config = _write_config(tmp_path)

    summary = s24_1.run_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage24_2_hybrid_astar_reward_path_cost_contract"
    assert summary["hybrid_astar_candidate_row_count"] == 3
    assert summary["hybrid_astar_reachable_count"] >= 2
    assert summary["hybrid_vs_grid_cost_material"] is True
    assert summary["default_astar_replaced"] is False
    assert summary["ackermann_feasible_claimed"] is False
    assert summary["publishes_checkpoint"] is False
    assert summary["max_traversable_slope_deg"] == 30.0
    audit_path = tmp_path / "out" / "xunce-stage24-1-candidate-hybrid-path-cost-audit.jsonl"
    assert audit_path.is_file()
    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert all(row["hybrid_astar_trajectory_kind"] == "hybrid_astar_pose_path" for row in rows if row["hybrid_astar_reachable"])
    assert all(row["hybrid_astar_ackermann_feasible_claimed"] is False for row in rows)
    assert (tmp_path / "out" / "xunce-stage24-1-grid-vs-hybrid-cost-delta-audit.json").is_file()
    assert (tmp_path / "out" / "xunce-stage24-1-path-cost-source-recommendation.json").is_file()


def test_stage24_1_routes_candidate_pose_contract_failure(tmp_path: Path) -> None:
    import scripts.run_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration as s24_1

    config = _write_config(tmp_path, candidate_overrides=[{"candidate_viewpoint": None, "candidate_theta_deg": None}])

    summary = s24_1.run_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage24_1_candidate_pose_contract"
    assert "candidate_pose_contract_invalid" in summary["blocking_reason_codes"]


def test_stage24_1_routes_boundary_rejection(tmp_path: Path) -> None:
    import scripts.run_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration as s24_1

    config = _write_config(tmp_path, config_overrides={"publishes_checkpoint": True})

    summary = s24_1.run_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage24_1_boundary_rejections"
    assert "config_publishes_checkpoint_enabled" in summary["blocking_reason_codes"]


def test_stage24_1_routes_authorization_boundary_rejection(tmp_path: Path) -> None:
    import scripts.run_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration as s24_1

    config = _write_config(tmp_path, config_overrides={"stage24_1_authorized": True})

    summary = s24_1.run_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage24_1_boundary_rejections"
    assert "config_stage24_1_authorized_enabled" in summary["blocking_reason_codes"]


def test_stage24_1_routes_platform_canary_boundary_rejection(tmp_path: Path) -> None:
    import scripts.run_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration as s24_1

    platform_path = _write_platform_contract(tmp_path, boundary_overrides={"canary_traffic_fraction": 0.1})
    config = _write_config(tmp_path, config_overrides={"platform_contract": str(platform_path)})

    summary = s24_1.run_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage24_1_boundary_rejections"
    assert "platform_contract_canary_traffic_fraction_nonzero" in summary["blocking_reason_codes"]


def test_stage24_1_routes_missing_pose_and_slope_lineage(tmp_path: Path) -> None:
    import scripts.run_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration as s24_1

    config = _write_config(tmp_path, config_overrides={"current_pose_provenance": ""})
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["grid"]["slope_obstacle_source_hash"] = ""
    config.write_text(json.dumps(payload), encoding="utf-8")

    summary = s24_1.run_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage24_1_required_inputs"
    assert "current_pose_provenance_missing" in summary["blocking_reason_codes"]
    assert "slope_obstacle_source_hash_missing" in summary["blocking_reason_codes"]


def test_stage24_1_routes_stage24_0_platform_lineage_mismatch(tmp_path: Path) -> None:
    import scripts.run_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration as s24_1

    config = _write_config(tmp_path, stage24_0_overrides={"platform_contract_hash": "other-hash"})

    summary = s24_1.run_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage24_1_required_inputs"
    assert "stage24_0_platform_contract_hash_mismatch" in summary["blocking_reason_codes"]


def test_stage24_1_routes_stage24_0_platform_lineage_missing(tmp_path: Path) -> None:
    import scripts.run_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration as s24_1

    config = _write_config(
        tmp_path,
        stage24_0_overrides={
            "platform_contract_hash": None,
            "max_traversable_slope_deg": None,
        },
    )

    summary = s24_1.run_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage24_1_required_inputs"
    assert "stage24_0_platform_contract_hash_missing" in summary["blocking_reason_codes"]
    assert "stage24_0_max_traversable_slope_missing" in summary["blocking_reason_codes"]


def _write_config(
    tmp_path: Path,
    *,
    candidate_overrides: list[dict] | None = None,
    config_overrides: dict | None = None,
    stage24_0_overrides: dict | None = None,
) -> Path:
    stage24_0 = tmp_path / "s24_0"
    stage24_0.mkdir()
    stage24_0_summary = {
        "status": "passed",
        "next_required_change": "run_stage24_1_hybrid_astar_candidate_path_cost_integration",
        "platform_contract_hash": _platform_contract_hash(),
        "max_traversable_slope_deg": 30.0,
        "default_astar_replaced": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
    }
    stage24_0_summary.update(stage24_0_overrides or {})
    (stage24_0 / "xunce-stage24-0-summary.json").write_text(
        json.dumps(stage24_0_summary),
        encoding="utf-8",
    )
    candidates = [
        {
            "scenario_id": "turning_candidate",
            "step_index": 0,
            "candidate_index": 0,
            "candidate_set_hash": "theta-hash-a",
            "candidate_viewpoint": [3, 1, 90],
            "candidate_theta_deg": 90,
            "path_cost": 2.0,
        },
        {
            "scenario_id": "straight_candidate",
            "step_index": 0,
            "candidate_index": 1,
            "candidate_set_hash": "theta-hash-a",
            "candidate_viewpoint": [4, 1, 0],
            "candidate_theta_deg": 0,
            "path_cost": 3.0,
        },
        {
            "scenario_id": "blocked_candidate",
            "step_index": 0,
            "candidate_index": 2,
            "candidate_set_hash": "theta-hash-a",
            "candidate_viewpoint": [2, 2, 0],
            "candidate_theta_deg": 0,
            "path_cost": None,
        },
    ]
    if candidate_overrides:
        for index, overrides in enumerate(candidate_overrides):
            candidates[index].update(overrides)
    payload = {
        "schema_version": "xunce-stage24-1-hybrid-astar-candidate-path-cost-integration-config/v1",
        "stage24_0_root": str(stage24_0),
        "platform_contract": "configs/platforms/agilex_scout_mini_piper_v1.json",
        "hybrid_astar_pose_path_cost_enabled": True,
        "current_pose": [1.5, 1.5, 0.0],
        "current_pose_provenance": "test_fixture",
        "grid": {
            "width": 5,
            "height": 5,
            "resolution_m": 1.0,
            "blocked_cells": [[2, 2]],
            "slope_blocked_cells": [[2, 2]],
            "slope_obstacle_source_hash": "slope-source-hash",
        },
        "candidate_rows": candidates,
        "theta_bin_count": 72,
        "goal_theta_tolerance_deg": 5.0,
        "max_iterations": 100000,
        "primitive_duration_s": 1.0,
        "integration_dt_s": 0.25,
        "max_speed_mps": 1.0,
        "max_angular_speed_degps": 45.0,
        "footprint_safety_margin_m": 0.0,
        "rotation_cost_weight": 0.2,
        "reverse_penalty_weight": 0.5,
        "turn_penalty_weight": 0.05,
        "material_cost_delta_threshold": 0.01,
        "stage24_1_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(config_overrides or {})
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_platform_contract(tmp_path: Path, *, boundary_overrides: dict | None = None) -> Path:
    source = REPO_ROOT / "configs" / "platforms" / "agilex_scout_mini_piper_v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload.setdefault("boundary", {}).update(boundary_overrides or {})
    path = tmp_path / "platform.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _platform_contract_hash() -> str:
    source = REPO_ROOT / "configs" / "platforms" / "agilex_scout_mini_piper_v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    import hashlib

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
