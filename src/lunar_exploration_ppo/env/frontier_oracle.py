"""Independent observed-only audit for legal no-candidate termination."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from lunar_exploration_ppo.env.map_state import ObservedMapState
from lunar_exploration_ppo.env.sensor_model import ray_cells_from_world
from lunar_exploration_ppo.utils.geometry import CellXY, PoseXYTheta


@dataclass(frozen=True, slots=True)
class FrontierOpportunityAudit:
    opportunity_cells: tuple[CellXY, ...]
    component_size: int
    unknown_count: int

    @property
    def opportunity_count(self) -> int:
        return len(self.opportunity_cells)


def audit_frontier_opportunities(
    observed_state: ObservedMapState,
    pose: PoseXYTheta,
    *,
    sensor_range_m: float,
    max_slope_deg: float = 30.0,
) -> FrontierOpportunityAudit:
    component = _corner_safe_component(observed_state.observed_safe_mask, pose.cell)
    unknown = ~observed_state.observed_mask
    unknown_count = int(np.count_nonzero(unknown))
    if unknown_count == 0:
        return FrontierOpportunityAudit((), int(np.count_nonzero(component)), 0)
    radius_cells = sensor_range_m / observed_state.geometry.resolution_m
    opportunities: list[CellXY] = []
    for source_y, source_x in np.argwhere(component):
        min_x = max(0, math.floor(source_x - radius_cells))
        max_x = min(observed_state.geometry.width - 1, math.ceil(source_x + radius_cells))
        min_y = max(0, math.floor(source_y - radius_cells))
        max_y = min(observed_state.geometry.height - 1, math.ceil(source_y + radius_cells))
        local_y, local_x = np.nonzero(unknown[min_y : max_y + 1, min_x : max_x + 1])
        if local_y.size == 0:
            continue
        target_y = local_y + min_y
        target_x = local_x + min_x
        distances_sq = (target_x - source_x) ** 2 + (target_y - source_y) ** 2
        order = np.argsort(distances_sq, kind="stable")
        source = CellXY(int(source_x), int(source_y))
        for candidate_index in order:
            if math.sqrt(float(distances_sq[candidate_index])) > radius_cells:
                break
            target = CellXY(int(target_x[candidate_index]), int(target_y[candidate_index]))
            if _has_observed_gain_los(
                observed_state,
                source,
                target,
                sensor_range_m=sensor_range_m,
                max_slope_deg=max_slope_deg,
            ):
                opportunities.append(source)
                break
    return FrontierOpportunityAudit(
        tuple(opportunities),
        int(np.count_nonzero(component)),
        unknown_count,
    )


def _corner_safe_component(mask: np.ndarray, start: CellXY) -> np.ndarray:
    safe = np.asarray(mask, dtype=bool)
    result = np.zeros_like(safe)
    height, width = safe.shape
    if not (0 <= start.x < width and 0 <= start.y < height) or not safe[start.y, start.x]:
        return result
    result[start.y, start.x] = True
    queue: deque[CellXY] = deque((start,))
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1))
    while queue:
        current = queue.popleft()
        for dx, dy in directions:
            x, y = current.x + dx, current.y + dy
            if not (0 <= x < width and 0 <= y < height) or result[y, x] or not safe[y, x]:
                continue
            if dx and dy and (not safe[current.y, x] or not safe[y, current.x]):
                continue
            result[y, x] = True
            queue.append(CellXY(x, y))
    return result


def _has_observed_gain_los(
    state: ObservedMapState,
    source: CellXY,
    target: CellXY,
    *,
    sensor_range_m: float,
    max_slope_deg: float,
) -> bool:
    source_world = state.geometry.cell_to_world_center(source)
    target_world = state.geometry.cell_to_world_center(target)
    angle = math.atan2(target_world.y - source_world.y, target_world.x - source_world.x)
    distance = math.hypot(target_world.x - source_world.x, target_world.y - source_world.y)
    for cell in ray_cells_from_world(
        source_world,
        angle,
        state.geometry,
        range_m=min(sensor_range_m, distance),
    )[1:]:
        if not state.observed_mask[cell.y, cell.x]:
            return True
        if state.obstacle[cell.y, cell.x] or state.slope_deg[cell.y, cell.x] > max_slope_deg:
            return False
    return False
