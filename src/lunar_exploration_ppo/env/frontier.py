"""Observed-only Stage 1 frontier landing action generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lunar_exploration_ppo.env.map_state import ObservedMapState
from lunar_exploration_ppo.env.reachability import reachable_component
from lunar_exploration_ppo.env.scenario import LowResolutionPrior
from lunar_exploration_ppo.utils.geometry import CellXY, PoseXYTheta


@dataclass(frozen=True, slots=True)
class FrontierActionSet:
    cells: tuple[CellXY, ...]
    frontier_features: np.ndarray
    candidate_mask: np.ndarray

    def __post_init__(self) -> None:
        self.frontier_features.setflags(write=False)
        self.candidate_mask.setflags(write=False)

    @property
    def candidate_count(self) -> int:
        return len(self.cells)


class FrontierGenerator:
    def __init__(self, *, top_m: int = 512) -> None:
        if top_m <= 0:
            raise ValueError("frontier top_m must be positive")
        self.top_m = top_m

    def extract(
        self,
        observed_state: ObservedMapState,
        prior: LowResolutionPrior,
        pose: PoseXYTheta,
    ) -> FrontierActionSet:
        component = reachable_component(observed_state.observed_safe_mask, pose.cell)
        height, width = observed_state.geometry.shape
        candidates: list[CellXY] = []
        for y, x in np.argwhere(component):
            cell = CellXY(int(x), int(y))
            if _unknown_neighbor_count(observed_state.observed_mask, cell) > 0:
                candidates.append(cell)
        candidates.sort(key=lambda cell: (cell.y, cell.x))
        cells = tuple(candidates[: self.top_m])
        features = np.zeros((self.top_m, 22), dtype=np.float32)
        mask = np.zeros((self.top_m,), dtype=bool)
        diagonal = max(math.hypot(width - 1, height - 1), 1.0)
        for index, cell in enumerate(cells):
            dx = cell.x - pose.cell.x
            dy = cell.y - pose.cell.y
            distance = math.hypot(dx, dy)
            unknown_neighbors, unknown_dx, unknown_dy = _unknown_neighbor_statistics(
                observed_state.observed_mask,
                cell,
            )
            observed_blockers = _observed_blocker_neighbor_count(observed_state, cell)
            prior_x = min(prior.channels.shape[2] - 1, int(cell.x * prior.channels.shape[2] / width))
            prior_y = min(prior.channels.shape[1] - 1, int(cell.y * prior.channels.shape[1] / height))
            prior_values = prior.channels[:, prior_y, prior_x]
            row = np.zeros((22,), dtype=np.float32)
            row[0] = cell.x / max(width - 1, 1)
            row[1] = cell.y / max(height - 1, 1)
            row[2] = dx / max(width - 1, 1)
            row[3] = dy / max(height - 1, 1)
            row[4] = distance / diagonal
            if unknown_dx != 0 or unknown_dy != 0:
                recommended_theta = math.atan2(unknown_dy, unknown_dx)
            elif distance:
                # Symmetric unknown neighbours have no unique outward normal.
                # The observed-only travel bearing is the deterministic fallback.
                recommended_theta = math.atan2(dy, dx)
            else:
                recommended_theta = pose.theta
            row[5] = math.sin(recommended_theta)
            row[6] = math.cos(recommended_theta)
            row[7] = unknown_neighbors / 8.0
            row[8] = observed_blockers / 8.0
            row[9] = float(observed_state.confidence[cell.y, cell.x])
            row[10] = float(observed_state.height[cell.y, cell.x])
            row[11] = float(observed_state.slope_deg[cell.y, cell.x] / 30.0)
            row[12] = float(observed_state.traversability[cell.y, cell.x])
            row[13] = 1.0
            row[14:21] = prior_values
            row[21] = index / max(self.top_m - 1, 1)
            features[index] = row
            mask[index] = True
        return FrontierActionSet(cells=cells, frontier_features=features, candidate_mask=mask)


def _unknown_neighbor_count(observed_mask: np.ndarray, cell: CellXY) -> int:
    count, _, _ = _unknown_neighbor_statistics(observed_mask, cell)
    return count


def _unknown_neighbor_statistics(
    observed_mask: np.ndarray,
    cell: CellXY,
) -> tuple[int, int, int]:
    height, width = observed_mask.shape
    count = 0
    sum_dx = 0
    sum_dy = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            x, y = cell.x + dx, cell.y + dy
            if 0 <= x < width and 0 <= y < height and not observed_mask[y, x]:
                count += 1
                sum_dx += dx
                sum_dy += dy
    return count, sum_dx, sum_dy


def _observed_blocker_neighbor_count(state: ObservedMapState, cell: CellXY) -> int:
    height, width = state.observed_mask.shape
    count = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            x, y = cell.x + dx, cell.y + dy
            if 0 <= x < width and 0 <= y < height and state.observed_mask[y, x] and not state.observed_safe_mask[y, x]:
                count += 1
    return count
