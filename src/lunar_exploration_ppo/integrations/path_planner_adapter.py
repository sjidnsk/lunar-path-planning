"""The only Stage 1 integration boundary to the standalone path planner."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from time import perf_counter_ns
from typing import Literal, Mapping

import numpy as np
from path_planner.core import Cell, CostGrid, GridSpec, NeighborPolicy, PlanRequest
from path_planner.search import AStarPlanner

from lunar_exploration_ppo.env.reachability import reachable_component
from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry, WorldXY, normalize_theta


PlannerFailureClassification = Literal["none", "pre_execution_invalid_action"]
_TIMING_FIELDS = (
    "input_validation_ns",
    "platform_instantiation_ns",
    "search_ns",
    "complete_route_validation_ns",
    "result_assembly_ns",
)


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
        timings = {name: 0 for name in _TIMING_FIELDS}
        run_start_ns = perf_counter_ns()
        phase_started_ns = run_start_ns

        def finish_phase(name: str) -> int:
            nonlocal phase_started_ns
            ended_ns = perf_counter_ns()
            duration_ns = ended_ns - phase_started_ns
            if duration_ns < 0:
                raise RuntimeError("perf_counter_ns moved backwards")
            timings[name] = duration_ns
            phase_started_ns = ended_ns
            return ended_ns

        def assemble_failure(reason: str) -> PlannerResult:
            failure = self._failure(reason)
            finish_phase("result_assembly_ns")
            return self._with_timing(
                failure,
                timings,
                run_start_ns=run_start_ns,
                final_end_ns=phase_started_ns,
            )

        def fail(reason: str, phase_name: str) -> PlannerResult:
            finish_phase(phase_name)
            return assemble_failure(reason)

        try:
            normalized_theta = normalize_theta(theta)
        except ValueError:
            return fail("invalid_theta", "input_validation_ns")
        mask = np.asarray(safe_mask, dtype=bool)
        if mask.shape != self.geometry.shape:
            return fail("invalid_safe_mask", "input_validation_ns")
        if not self.geometry.in_bounds(start):
            return fail("start_out_of_bounds", "input_validation_ns")
        if not self.geometry.in_bounds(target):
            return fail("target_out_of_bounds", "input_validation_ns")
        if not mask[start.y, start.x]:
            return fail("start_unsafe", "input_validation_ns")
        if not mask[target.y, target.x]:
            return fail("target_unsafe", "input_validation_ns")
        component = reachable_component(mask, start)
        if not component[target.y, target.x]:
            return fail("target_unreachable", "input_validation_ns")
        finish_phase("input_validation_ns")

        spec = GridSpec(
            width=self.geometry.width,
            height=self.geometry.height,
            resolution=self.geometry.resolution_m,
            origin=(self.geometry.origin.x, self.geometry.origin.y),
        )
        blocked_count = int(np.count_nonzero(~mask))
        grid = CostGrid(
            spec=spec,
            cost=np.ones(mask.shape, dtype=float),
            passable_mask=mask,
        )
        finish_phase("platform_instantiation_ns")

        result = AStarPlanner().plan(
            grid,
            PlanRequest(
                start=Cell(start.x, start.y),
                goal=Cell(target.x, target.y),
                neighbor_policy=NeighborPolicy.EIGHT,
                prevent_corner_cutting=True,
            ),
        )
        finish_phase("search_ns")
        if not result.success:
            reason = (
                result.failure_reason.value
                if result.failure_reason is not None
                else "unknown"
            )
            return assemble_failure(f"planner_{reason}")

        path_cells = tuple(CellXY(cell.x, cell.y) for cell in result.path_cells)
        for cell in path_cells:
            if not self.geometry.in_bounds(cell):
                return fail(
                    "path_out_of_bounds",
                    "complete_route_validation_ns",
                )
            if not mask[cell.y, cell.x]:
                return fail("path_unsafe", "complete_route_validation_ns")
        finish_phase("complete_route_validation_ns")

        path_centers = tuple(
            self.geometry.cell_to_world_center(cell) for cell in path_cells
        )
        path_length_m = sum(
            math.hypot(right.x - left.x, right.y - left.y)
            for left, right in zip(
                path_centers[:-1],
                path_centers[1:],
                strict=True,
            )
        )
        diagnostics = {
            "neighbor_policy": "8-neighbor",
            "prevent_corner_cutting": True,
            "inflation_applied": False,
            "planner_footprint_radius_m": None,
            "original_blocked_count": blocked_count,
            "inflated_blocked_count": blocked_count,
            "expanded_count": result.expanded_count,
        }
        assembled = PlannerResult(
            valid=True,
            failure_reason="none",
            failure_classification="none",
            path_cells=path_cells,
            path_world=path_centers,
            target_theta=normalized_theta,
            path_length_m=path_length_m,
            diagnostics=diagnostics,
        )
        finish_phase("result_assembly_ns")
        return self._with_timing(
            assembled,
            timings,
            run_start_ns=run_start_ns,
            final_end_ns=phase_started_ns,
        )

    @staticmethod
    def _with_timing(
        result: PlannerResult,
        timings: Mapping[str, int],
        *,
        run_start_ns: int,
        final_end_ns: int,
    ) -> PlannerResult:
        timing_values = {
            name: int(timings[name]) for name in _TIMING_FIELDS
        }
        component_total_ns = sum(timing_values.values())
        total_ns = final_end_ns - run_start_ns
        if total_ns < 0 or total_ns != component_total_ns:
            raise RuntimeError("five-phase sequential timing invariant violated")
        return replace(
            result,
            diagnostics={
                **result.diagnostics,
                "run_start_ns": run_start_ns,
                "final_end_ns": final_end_ns,
                **timing_values,
                "total_ns": total_ns,
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
