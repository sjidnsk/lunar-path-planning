"""Stage 6 observed-only planning buffer、frontier 与 A* 合同。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunar_exploration_ppo.env.frontier import FrontierGenerator
from lunar_exploration_ppo.env.frontier_oracle import audit_frontier_opportunities
from lunar_exploration_ppo.env import map_state as map_state_module
from lunar_exploration_ppo.env.map_state import ObservedMapState
from lunar_exploration_ppo.env.scenario import LowResolutionPrior
from lunar_exploration_ppo.integrations.path_planner_adapter import (
    PathPlannerAdapter,
)
from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry, PoseXYTheta


def _physical_state(
    *,
    width: int,
    height: int,
    resolution_m: float,
    observed: np.ndarray,
) -> ObservedMapState:
    state = ObservedMapState.empty(
        GridGeometry(width=width, height=height, resolution_m=resolution_m)
    )
    state.observed_mask[...] = observed
    state.confidence[observed] = 1.0
    state.traversability[observed] = 1.0
    state.observed_safe_mask[...] = observed
    return state


def test_planning_buffer_constant_is_frozen_without_changing_physical_clearance() -> None:
    assert getattr(map_state_module, "PLANNING_UNKNOWN_BUFFER_M", None) == 0.75
    assert callable(getattr(map_state_module, "build_planning_safe_mask", None))


def _build_planning_safe_mask(**kwargs: object) -> np.ndarray:
    builder = getattr(map_state_module, "build_planning_safe_mask", None)
    if not callable(builder):
        pytest.skip("planning mask API is not implemented yet")
    return builder(**kwargs)


def test_planning_mask_excludes_half_meter_orthogonal_and_diagonal_unknown_neighbors() -> None:
    observed = np.ones((7, 7), dtype=bool)
    observed[3, 3] = False
    physical = np.ones_like(observed)

    planning = _build_planning_safe_mask(
        observed_safe_mask=physical,
        observed_mask=observed,
        resolution_m=0.5,
        unknown_buffer_m=0.75,
    )

    assert not planning[3, 3]
    assert not planning[3, 2]  # 0.5m orthogonal
    assert not planning[2, 2]  # sqrt(0.5^2 + 0.5^2)
    assert planning[3, 1]  # 1.0m
    assert planning[1, 3]  # 1.0m


def test_planning_mask_uses_strictly_less_than_boundary_on_non_half_meter_grid() -> None:
    observed = np.ones((9, 9), dtype=bool)
    observed[4, 4] = False
    physical = np.ones_like(observed)

    exact = _build_planning_safe_mask(
        observed_safe_mask=physical,
        observed_mask=observed,
        resolution_m=0.75,
        unknown_buffer_m=0.75,
    )
    fine = _build_planning_safe_mask(
        observed_safe_mask=physical,
        observed_mask=observed,
        resolution_m=0.3,
        unknown_buffer_m=0.75,
    )

    assert exact[4, 3]  # center distance exactly 0.75m is allowed
    assert not fine[4, 2]  # 0.6m
    assert not fine[3, 2]  # sqrt(0.6^2 + 0.3^2)
    assert fine[2, 2]  # sqrt(0.6^2 + 0.6^2)


def test_planning_mask_has_no_extra_roi_boundary_clearance_and_handles_extremes() -> None:
    physical = np.ones((4, 5), dtype=bool)

    fully_observed = _build_planning_safe_mask(
        observed_safe_mask=physical,
        observed_mask=np.ones_like(physical),
        resolution_m=0.5,
        unknown_buffer_m=0.75,
    )
    fully_unknown = _build_planning_safe_mask(
        observed_safe_mask=physical,
        observed_mask=np.zeros_like(physical),
        resolution_m=0.5,
        unknown_buffer_m=0.75,
    )

    assert np.array_equal(fully_observed, physical)
    assert fully_observed[0, 0]  # planning buffer does not inflate ROI boundary
    assert not np.any(fully_unknown)


def test_planning_mask_is_derived_only_from_observed_state_and_expands_after_reveal() -> None:
    observed = np.zeros((7, 7), dtype=bool)
    observed[2:5, 2:5] = True
    state = _physical_state(
        width=7,
        height=7,
        resolution_m=0.5,
        observed=observed,
    )
    physical_before = state.observed_safe_mask.copy()
    first = state.planning_safe_mask.copy()

    # Hidden truth arrays are irrelevant until observed_mask changes.
    state.obstacle[0, 0] = True
    state.slope_deg[0, 0] = 89.0
    state.traversability[0, 0] = 0.0
    assert np.array_equal(state.planning_safe_mask, first)

    state.observed_mask[1:6, 1:6] = True
    state.observed_safe_mask[1:6, 1:6] = True
    expanded = state.planning_safe_mask

    assert np.count_nonzero(expanded) > np.count_nonzero(first)
    assert np.array_equal(state.observed_safe_mask[2:5, 2:5], physical_before[2:5, 2:5])


def test_frontier_endpoints_and_reachability_use_planning_mask() -> None:
    observed = np.zeros((17, 17), dtype=bool)
    observed[2:15, 2:11] = True
    state = _physical_state(
        width=17,
        height=17,
        resolution_m=0.5,
        observed=observed,
    )
    prior = LowResolutionPrior(
        channels=np.zeros((7, 5, 5), dtype=np.float32),
        resolution_m=2.0,
        value_prior_source="constant_neutral/v1",
    )
    pose = PoseXYTheta(CellXY(4, 8), 0.0)

    actions = FrontierGenerator(top_m=64).extract(state, prior, pose)

    assert actions.candidate_count > 0
    assert all(state.planning_safe_mask[cell.y, cell.x] for cell in actions.cells)
    assert not any(cell.x == 10 for cell in actions.cells)
    assert actions.diagnostics["reachability_mask"] == "planning_safe_mask/v1"


def test_frontier_oracle_component_uses_caller_planning_safe_mask() -> None:
    observed = np.zeros((7, 7), dtype=bool)
    observed[1:6, 1:6] = True
    state = _physical_state(
        width=7,
        height=7,
        resolution_m=0.5,
        observed=observed,
    )
    pose = PoseXYTheta(CellXY(3, 3), 0.0)
    planning_safe = np.zeros_like(observed)
    planning_safe[3, 3] = True

    audit = audit_frontier_opportunities(
        state,
        pose,
        planning_safe_mask=planning_safe,
        sensor_range_m=20.0,
    )

    assert np.count_nonzero(state.observed_safe_mask) == 25
    assert audit.component_size == 1


def test_frontier_oracle_counts_safe_current_pose_with_positive_gain() -> None:
    observed = np.zeros((7, 7), dtype=bool)
    observed[1:6, 1:6] = True
    state = _physical_state(
        width=7,
        height=7,
        resolution_m=0.5,
        observed=observed,
    )
    pose = PoseXYTheta(CellXY(3, 3), 0.0)
    planning_safe = np.zeros_like(observed)
    planning_safe[3, 3] = True

    audit = audit_frontier_opportunities(
        state,
        pose,
        planning_safe_mask=planning_safe,
        sensor_range_m=20.0,
    )

    assert audit.unknown_count > 0
    assert audit.opportunity_cells == (pose.cell,)


def test_planner_distinguishes_physical_endpoint_unknown_buffer_and_no_path() -> None:
    geometry = GridGeometry(6, 5, 0.5)
    adapter = PathPlannerAdapter(geometry)
    observed_safe = np.ones(geometry.shape, dtype=bool)
    planning_safe = observed_safe.copy()
    target = CellXY(4, 2)

    physical = observed_safe.copy()
    physical[target.y, target.x] = False
    physical_failure = adapter.validate(
        physical,
        planning_safe,
        CellXY(1, 2),
        target,
        0.0,
    )
    planning_safe[target.y, target.x] = False
    unknown_failure = adapter.validate(
        observed_safe,
        planning_safe,
        CellXY(1, 2),
        target,
        0.0,
    )
    blocked = planning_safe.copy()
    blocked[:, 3] = False
    blocked[target.y, target.x] = True
    no_path = adapter.validate(
        observed_safe,
        blocked,
        CellXY(1, 2),
        target,
        math.pi / 2.0,
    )

    assert physical_failure.failure_reason == "endpoint_physical_unsafe"
    assert unknown_failure.failure_reason == "endpoint_unknown_buffer_unsafe"
    assert no_path.failure_reason == "planner_no_path"


def test_planner_passable_mask_and_returned_path_cells_are_planning_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.integrations import path_planner_adapter as module

    geometry = GridGeometry(5, 3, 0.5)
    observed_safe = np.ones(geometry.shape, dtype=bool)
    planning_safe = observed_safe.copy()
    planning_safe[1, 2] = False
    observed_passable: list[np.ndarray] = []

    class _Result:
        success = True
        failure_reason = None
        expanded_count = 3
        path_cells = (
            module.Cell(0, 1),
            module.Cell(1, 1),
            module.Cell(2, 1),
            module.Cell(3, 1),
            module.Cell(4, 1),
        )

    class _Planner:
        def plan(self, grid, request):
            del request
            observed_passable.append(np.asarray(grid.passable_mask, dtype=bool).copy())
            return _Result()

    monkeypatch.setattr(module, "AStarPlanner", _Planner)

    result = PathPlannerAdapter(geometry).validate(
        observed_safe,
        planning_safe,
        CellXY(0, 1),
        CellXY(4, 1),
        0.0,
    )

    assert np.array_equal(observed_passable[0], planning_safe)
    assert not result.valid
    assert result.failure_reason == "path_unknown_buffer_unsafe"


def test_planner_does_not_cut_diagonal_corner_through_planning_unsafe_cells() -> None:
    geometry = GridGeometry(3, 3, 0.5)
    observed_safe = np.ones(geometry.shape, dtype=bool)
    planning_safe = observed_safe.copy()
    planning_safe[0, 1] = False
    planning_safe[1, 0] = False

    result = PathPlannerAdapter(geometry).validate(
        observed_safe,
        planning_safe,
        CellXY(0, 0),
        CellXY(1, 1),
        0.0,
    )

    assert not result.valid
    assert result.failure_reason == "planner_no_path"
