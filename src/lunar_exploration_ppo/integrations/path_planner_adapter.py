"""The only Stage 1 integration boundary to the standalone path planner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
from path_planner.core import Cell, CostGrid, GridSpec, NeighborPolicy, PlanRequest
from path_planner.search import AStarPlanner

from lunar_exploration_ppo.env.reachability import reachable_component
from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry, WorldXY, normalize_theta


PlannerFailureClassification = Literal["none", "pre_execution_invalid_action"]


@dataclass(frozen=True, slots=True)
class PlannerResult:
    valid: bool
    failure_reason: str
    failure_classification: PlannerFailureClassification
    path_cells: tuple[CellXY, ...]
    path_world: tuple[WorldXY, ...]
    target_theta: float | None
    path_length_m: float
    diagnostics: Mapping[str, object]


class PathPlannerAdapter:
    def __init__(self, geometry: GridGeometry) -> None:
        self.geometry = geometry

    def validate(
        self,
        safe_mask: np.ndarray,
        start: CellXY,
        target: CellXY,
        theta: float,
    ) -> PlannerResult:
        try:
            normalized_theta = normalize_theta(theta)
        except ValueError:
            return self._failure("invalid_theta")
        mask = np.asarray(safe_mask, dtype=bool)
        if mask.shape != self.geometry.shape:
            return self._failure("invalid_safe_mask")
        if not self.geometry.in_bounds(start):
            return self._failure("start_out_of_bounds")
        if not self.geometry.in_bounds(target):
            return self._failure("target_out_of_bounds")
        if not mask[start.y, start.x]:
            return self._failure("start_unsafe")
        if not mask[target.y, target.x]:
            return self._failure("target_unsafe")
        component = reachable_component(mask, start)
        if not component[target.y, target.x]:
            return self._failure("target_unreachable")

        spec = GridSpec(
            width=self.geometry.width,
            height=self.geometry.height,
            resolution=self.geometry.resolution_m,
            origin=(self.geometry.origin.x, self.geometry.origin.y),
        )
        blocked_count = int(np.count_nonzero(~mask))
        grid = CostGrid(spec=spec, cost=np.ones(mask.shape, dtype=float), passable_mask=mask)
        result = AStarPlanner().plan(
            grid,
            PlanRequest(
                start=Cell(start.x, start.y),
                goal=Cell(target.x, target.y),
                neighbor_policy=NeighborPolicy.EIGHT,
                prevent_corner_cutting=True,
            ),
        )
        if not result.success:
            reason = result.failure_reason.value if result.failure_reason is not None else "unknown"
            return self._failure(f"planner_{reason}")
        path_cells = tuple(CellXY(cell.x, cell.y) for cell in result.path_cells)
        path_centers = tuple(self.geometry.cell_to_world_center(cell) for cell in path_cells)
        path_length_m = sum(
            math.hypot(right.x - left.x, right.y - left.y)
            for left, right in zip(path_centers[:-1], path_centers[1:], strict=True)
        )
        return PlannerResult(
            valid=True,
            failure_reason="none",
            failure_classification="none",
            path_cells=path_cells,
            path_world=path_centers,
            target_theta=normalized_theta,
            path_length_m=path_length_m,
            diagnostics={
                "neighbor_policy": "8-neighbor",
                "prevent_corner_cutting": True,
                "inflation_applied": False,
                "planner_footprint_radius_m": None,
                "original_blocked_count": blocked_count,
                "inflated_blocked_count": blocked_count,
                "expanded_count": result.expanded_count,
            },
        )

    @staticmethod
    def _failure(reason: str) -> PlannerResult:
        return PlannerResult(
            valid=False,
            failure_reason=reason,
            failure_classification="pre_execution_invalid_action",
            path_cells=(),
            path_world=(),
            target_theta=None,
            path_length_m=0.0,
            diagnostics={
                "inflation_applied": False,
                "planner_footprint_radius_m": None,
            },
        )
