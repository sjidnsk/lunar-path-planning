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


def test_hybrid_astar_uses_pose_state_for_turn_in_place() -> None:
    from path_planner.core import CostGrid, GridSpec
    from path_planner.search import HybridAStarPlanner, Pose2D, PosePlanRequest

    passable = np.ones((5, 5), dtype=bool)
    grid = CostGrid(GridSpec(width=5, height=5, resolution=1.0), np.ones((5, 5)), passable)
    request = PosePlanRequest(
        start=Pose2D(2.0, 2.0, 0.0),
        goal=Pose2D(2.0, 2.0, math.pi / 2.0),
        max_speed_mps=1.0,
        max_angular_speed_radps=math.radians(45.0),
    )

    result = HybridAStarPlanner().plan(grid, request)

    assert result.success
    assert result.to_route_dict(grid.spec)["trajectory_kind"] == "hybrid_astar_pose_path"
    assert any(primitive.turn_in_place for primitive in result.control_sequence)
    assert abs((result.pose_path[-1].theta_rad - math.pi / 2.0)) <= math.radians(5.0)
    assert result.diagnostics.platform_model == "differential_skid_steer"
    assert result.diagnostics.ackermann_feasible_claimed is False


def test_hybrid_astar_rejects_blocked_goal_with_rectangular_footprint() -> None:
    from path_planner.core import CostGrid, FailureReason, GridSpec
    from path_planner.search import HybridAStarPlanner, Pose2D, PosePlanRequest

    passable = np.ones((5, 5), dtype=bool)
    passable[2, 2] = False
    grid = CostGrid(GridSpec(width=5, height=5, resolution=1.0), np.ones((5, 5)), passable)

    result = HybridAStarPlanner().plan(
        grid,
        PosePlanRequest(start=Pose2D(1.0, 1.0, 0.0), goal=Pose2D(2.0, 2.0, 0.0)),
    )

    assert not result.success
    assert result.failure_reason is FailureReason.GOAL_BLOCKED
    assert result.diagnostics.footprint_length_m == 0.612
    assert result.diagnostics.footprint_width_m == 0.580


def test_hybrid_astar_can_route_around_slope_blocked_proxy_cells() -> None:
    from path_planner.core import CostGrid, GridSpec
    from path_planner.search import HybridAStarPlanner, Pose2D, PosePlanRequest

    passable = np.ones((5, 7), dtype=bool)
    passable[1, 2] = False
    passable[2, 2] = False
    passable[3, 2] = False
    grid = CostGrid(GridSpec(width=7, height=5, resolution=1.0), np.ones((5, 7)), passable)

    result = HybridAStarPlanner().plan(
        grid,
        PosePlanRequest(
            start=Pose2D(1.0, 2.0, 0.0),
            goal=Pose2D(5.0, 2.0, 0.0),
            max_iterations=100000,
        ),
    )

    assert result.success
    assert [2, 1] not in [cell.to_list() for cell in result.legacy_cell_path]
    assert [2, 2] not in [cell.to_list() for cell in result.legacy_cell_path]
    assert [2, 3] not in [cell.to_list() for cell in result.legacy_cell_path]
    assert "slope_blocked_cells" in result.diagnostics.to_dict()["hard_obstacle_sources"]


def test_stage24_0_runner_passes_and_writes_expected_artifacts(tmp_path: Path) -> None:
    import scripts.run_xunce_stage24_0_hybrid_astar_pose_path_planner_full_foundation as s24

    config = _write_config(tmp_path)

    summary = s24.run_xunce_stage24_0_hybrid_astar_pose_path_planner_full_foundation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage24_1_hybrid_astar_candidate_path_cost_integration"
    assert summary["pose_path_case_count"] == 3
    assert summary["turn_in_place_supported"] is True
    assert summary["footprint_collision_audit_passed"] is True
    assert summary["obstacle_source_binding_audit_passed"] is True
    assert summary["default_astar_replaced"] is False
    assert summary["publishes_checkpoint"] is False
    assert (tmp_path / "out" / "xunce-stage24-0-hybrid-planner-smoke.jsonl").is_file()
    audit = json.loads((tmp_path / "out" / "xunce-stage24-0-grid-vs-hybrid-audit.json").read_text(encoding="utf-8"))
    assert audit["hybrid_uses_pose_state"] is True
    footprint = json.loads((tmp_path / "out" / "xunce-stage24-0-footprint-collision-audit.json").read_text(encoding="utf-8"))
    assert footprint["slope_source_kind"] == "slope_blocked_as_obstacle_proxy"
    assert footprint["blocked_source_kind"] == "blocked_as_obstacle_proxy"
    assert footprint["declared_obstacle_hard_gate_passed"] is True
    smoke_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "xunce-stage24-0-hybrid-planner-smoke.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in smoke_rows:
        if row["hybrid_success"]:
            assert row["legacy_path_declared_obstacle_intersections"] == []
            assert row["pose_path_declared_obstacle_intersections"] == []


def test_stage24_0_routes_boundary_rejection(tmp_path: Path) -> None:
    import scripts.run_xunce_stage24_0_hybrid_astar_pose_path_planner_full_foundation as s24

    config = _write_config(tmp_path, {"publishes_checkpoint": True})

    summary = s24.run_xunce_stage24_0_hybrid_astar_pose_path_planner_full_foundation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage24_0_boundary_rejections"
    assert "config_publishes_checkpoint_enabled" in summary["blocking_reason_codes"]


def test_stage24_0_routes_update_boundary_rejection(tmp_path: Path) -> None:
    import scripts.run_xunce_stage24_0_hybrid_astar_pose_path_planner_full_foundation as s24

    config = _write_config(tmp_path, {"runs_new_ppo_update": True})

    summary = s24.run_xunce_stage24_0_hybrid_astar_pose_path_planner_full_foundation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage24_0_boundary_rejections"
    assert "config_runs_new_ppo_update_enabled" in summary["blocking_reason_codes"]


def _write_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    payload = {
        "schema_version": "xunce-stage24-0-hybrid-astar-pose-path-planner-full-foundation-config/v1",
        "platform_contract": "configs/platforms/agilex_scout_mini_piper_v1.json",
        "stage23_6_root": "",
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
        "stage24_0_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides or {})
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
