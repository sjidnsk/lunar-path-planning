"""Exact scenario-level safe, reachable, and coverable masks."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from lunar_exploration_ppo.env.map_state import apply_clearance_once
from lunar_exploration_ppo.env.reachability import reachable_component
from lunar_exploration_ppo.env.scenario import TruthMap
from lunar_exploration_ppo.env.sensor_model import ray_cells_from_world
from lunar_exploration_ppo.utils.geometry import CellXY


@dataclass(frozen=True, slots=True)
class CoverageMasks:
    safe_free_mask: np.ndarray
    reachable_safe_mask: np.ndarray
    coverable_mask: np.ndarray
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        for mask in (self.safe_free_mask, self.reachable_safe_mask, self.coverable_mask):
            mask.setflags(write=False)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def coverable_cell_count(self) -> int:
        return int(np.count_nonzero(self.coverable_mask))


def compute_coverage_masks(
    truth: TruthMap,
    start: CellXY,
    *,
    sensor_range_m: float,
    min_clearance_m: float,
    max_slope_deg: float,
    traversability_threshold: float,
) -> CoverageMasks:
    finite = np.isfinite(truth.height) & np.isfinite(truth.slope_deg) & np.isfinite(truth.traversability)
    free_mask = (
        finite
        & ~truth.hard_obstacle
        & (truth.slope_deg <= max_slope_deg)
        & (truth.traversability >= traversability_threshold)
    )
    if not np.any(free_mask):
        raise ValueError("exact coverable denominator is zero")
    blocker = ~free_mask
    safe_free = apply_clearance_once(
        free_mask,
        blocker,
        resolution_m=truth.geometry.resolution_m,
        min_clearance_m=min_clearance_m,
    )
    reachable = reachable_component(safe_free, start)
    if not reachable[start.y, start.x]:
        raise ValueError("start pose is not inside the exact safe mask")
    coverable = reachable.copy()
    radius_cells = sensor_range_m / truth.geometry.resolution_m
    unresolved = np.argwhere(free_mask & ~coverable)
    for target_y, target_x in unresolved:
        min_x = max(0, math.floor(target_x - radius_cells))
        max_x = min(truth.geometry.width - 1, math.ceil(target_x + radius_cells))
        min_y = max(0, math.floor(target_y - radius_cells))
        max_y = min(truth.geometry.height - 1, math.ceil(target_y + radius_cells))
        local_y, local_x = np.nonzero(reachable[min_y : max_y + 1, min_x : max_x + 1])
        candidates = [
            CellXY(int(x + min_x), int(y + min_y))
            for y, x in zip(local_y, local_x, strict=True)
            if math.hypot((x + min_x) - target_x, (y + min_y) - target_y) <= radius_cells
        ]
        target = CellXY(int(target_x), int(target_y))
        candidates.sort(key=lambda cell: ((cell.x - target.x) ** 2 + (cell.y - target.y) ** 2, cell.y, cell.x))
        if any(_has_los(truth, source, target, max_slope_deg=max_slope_deg) for source in candidates):
            coverable[target.y, target.x] = True
    count = int(np.count_nonzero(coverable))
    if count == 0:
        raise ValueError("exact coverable denominator is zero")
    mask_hash = hashlib.sha256(np.ascontiguousarray(coverable).tobytes()).hexdigest()
    return CoverageMasks(
        safe_free_mask=safe_free,
        reachable_safe_mask=reachable,
        coverable_mask=coverable,
        metadata={
            "algorithm_id": "exact_reachable_safe_pose_range_los/v1",
            "sha256": mask_hash,
            "exact": True,
            "precompute_scope": "scenario_reset/v1",
            "coverable_cell_count": count,
        },
    )


def _has_los(truth: TruthMap, source: CellXY, target: CellXY, *, max_slope_deg: float) -> bool:
    source_world = truth.geometry.cell_to_world_center(source)
    target_world = truth.geometry.cell_to_world_center(target)
    distance = math.hypot(target_world.x - source_world.x, target_world.y - source_world.y)
    angle = math.atan2(target_world.y - source_world.y, target_world.x - source_world.x)
    for cell in ray_cells_from_world(source_world, angle, truth.geometry, range_m=distance)[1:]:
        if cell == target:
            return True
        if truth.hard_obstacle[cell.y, cell.x] or truth.slope_deg[cell.y, cell.x] > max_slope_deg:
            return False
    return False
