"""Observed-only map state and the environment-owned final safety mask."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from lunar_exploration_ppo.env.scenario import TruthMap
from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry


@dataclass(slots=True)
class ObservedMapState:
    geometry: GridGeometry
    observed_mask: np.ndarray
    confidence: np.ndarray
    height: np.ndarray
    obstacle: np.ndarray
    slope_deg: np.ndarray
    traversability: np.ndarray
    observed_safe_mask: np.ndarray

    @classmethod
    def empty(cls, geometry: GridGeometry) -> ObservedMapState:
        shape = geometry.shape
        return cls(
            geometry=geometry,
            observed_mask=np.zeros(shape, dtype=bool),
            confidence=np.zeros(shape, dtype=np.float32),
            height=np.zeros(shape, dtype=np.float64),
            obstacle=np.zeros(shape, dtype=bool),
            slope_deg=np.zeros(shape, dtype=np.float64),
            traversability=np.zeros(shape, dtype=np.float64),
            observed_safe_mask=np.zeros(shape, dtype=bool),
        )

    def reveal_cells(
        self,
        truth: TruthMap,
        cells: Iterable[CellXY],
        *,
        min_clearance_m: float,
        max_slope_deg: float,
        traversability_threshold: float,
    ) -> tuple[CellXY, ...]:
        if truth.geometry != self.geometry:
            raise ValueError("truth and observed geometry mismatch")
        unique = sorted({cell for cell in cells if self.geometry.in_bounds(cell)}, key=lambda cell: (cell.y, cell.x))
        newly_observed: list[CellXY] = []
        for cell in unique:
            y, x = cell.y, cell.x
            if not self.observed_mask[y, x]:
                newly_observed.append(cell)
            self.observed_mask[y, x] = True
            self.confidence[y, x] = 1.0
            self.height[y, x] = truth.height[y, x]
            self.obstacle[y, x] = truth.hard_obstacle[y, x]
            self.slope_deg[y, x] = truth.slope_deg[y, x]
            self.traversability[y, x] = truth.traversability[y, x]
        self.recompute_observed_safe_mask(
            min_clearance_m=min_clearance_m,
            max_slope_deg=max_slope_deg,
            traversability_threshold=traversability_threshold,
        )
        return tuple(newly_observed)

    def recompute_observed_safe_mask(
        self,
        *,
        min_clearance_m: float,
        max_slope_deg: float,
        traversability_threshold: float,
    ) -> None:
        base_safe = (
            self.observed_mask
            & ~self.obstacle
            & np.isfinite(self.slope_deg)
            & (self.slope_deg <= max_slope_deg)
            & np.isfinite(self.traversability)
            & (self.traversability >= traversability_threshold)
        )
        known_blocker = self.observed_mask & ~base_safe
        self.observed_safe_mask[...] = apply_clearance_once(
            base_safe,
            known_blocker,
            resolution_m=self.geometry.resolution_m,
            min_clearance_m=min_clearance_m,
        )


def apply_clearance_once(
    base_safe_mask: np.ndarray,
    blocker_mask: np.ndarray,
    *,
    resolution_m: float,
    min_clearance_m: float,
) -> np.ndarray:
    """Inflate known blockers and ROI boundaries exactly once."""

    safe = np.asarray(base_safe_mask, dtype=bool).copy()
    blockers = np.asarray(blocker_mask, dtype=bool)
    if safe.shape != blockers.shape or safe.ndim != 2:
        raise ValueError("safety and blocker masks must share a 2D shape")
    if not math.isfinite(min_clearance_m) or min_clearance_m < 0.0:
        raise ValueError("minimum clearance must be finite and nonnegative")
    height, width = safe.shape
    yy, xx = np.mgrid[0:height, 0:width]
    left = (xx + 0.5) * resolution_m
    right = (width - xx - 0.5) * resolution_m
    bottom = (yy + 0.5) * resolution_m
    top = (height - yy - 0.5) * resolution_m
    safe &= np.minimum.reduce((left, right, bottom, top)) >= min_clearance_m
    radius_cells = math.ceil(min_clearance_m / resolution_m)
    blocked_y, blocked_x = np.nonzero(blockers)
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if math.hypot(dx * resolution_m, dy * resolution_m) >= min_clearance_m:
                continue
            target_y = blocked_y + dy
            target_x = blocked_x + dx
            inside = (target_y >= 0) & (target_y < height) & (target_x >= 0) & (target_x < width)
            safe[target_y[inside], target_x[inside]] = False
    return safe
