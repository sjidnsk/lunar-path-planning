"""Deterministic path/endpoint 2D LOS sensor updates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal

from lunar_exploration_ppo.env.map_state import ObservedMapState
from lunar_exploration_ppo.env.scenario import TruthMap
from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry, WorldXY, normalize_theta


SensorSampleSource = Literal["reset", "path_tangent", "endpoint_theta"]


@dataclass(frozen=True, slots=True)
class SensorPose:
    world_xy: WorldXY
    heading: float
    source: SensorSampleSource

    def __post_init__(self) -> None:
        object.__setattr__(self, "heading", normalize_theta(self.heading))


@dataclass(frozen=True, slots=True)
class SensorDiagnostics:
    sample_count: int
    ray_count: int
    cell_visit_count: int
    unique_visible_cell_count: int
    duplicate_cell_visits: int
    sample_sources: tuple[str, ...]
    sample_headings: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ObservationDelta:
    newly_observed_cells: tuple[CellXY, ...]
    visible_cells: tuple[CellXY, ...]
    diagnostics: SensorDiagnostics

    @property
    def newly_observed_count(self) -> int:
        return len(self.newly_observed_cells)


class SensorUpdater:
    def __init__(
        self,
        *,
        range_m: float = 20.0,
        fov_deg: float = 90.0,
        ray_angle_step_deg: float = 1.0,
        min_clearance_m: float = 0.5215874761,
        max_slope_deg: float = 30.0,
        traversability_threshold: float = 0.5,
    ) -> None:
        for name, value in (
            ("range_m", range_m),
            ("fov_deg", fov_deg),
            ("ray_angle_step_deg", ray_angle_step_deg),
            ("min_clearance_m", min_clearance_m),
            ("max_slope_deg", max_slope_deg),
            ("traversability_threshold", traversability_threshold),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if range_m <= 0.0:
            raise ValueError("range_m must be positive")
        if not 0.0 <= fov_deg <= 360.0:
            raise ValueError("fov_deg must be within [0, 360]")
        if ray_angle_step_deg <= 0.0:
            raise ValueError("ray_angle_step_deg must be positive")
        if min_clearance_m < 0.0:
            raise ValueError("min_clearance_m must be nonnegative")
        if not 0.0 <= max_slope_deg <= 90.0:
            raise ValueError("max_slope_deg must be within [0, 90]")
        if not 0.0 <= traversability_threshold <= 1.0:
            raise ValueError("traversability_threshold must be within [0, 1]")
        self.range_m = float(range_m)
        self.fov_deg = float(fov_deg)
        self.ray_angle_step_deg = float(ray_angle_step_deg)
        self.min_clearance_m = float(min_clearance_m)
        self.max_slope_deg = float(max_slope_deg)
        self.traversability_threshold = float(traversability_threshold)

    def reveal(
        self,
        truth: TruthMap,
        observed_state: ObservedMapState,
        poses: Iterable[SensorPose],
    ) -> ObservationDelta:
        pose_tuple = tuple(poses)
        visible: set[CellXY] = set()
        visits = 0
        rays_per_sample = int(round(self.fov_deg / self.ray_angle_step_deg)) + 1
        half_fov = self.fov_deg / 2.0
        for pose in pose_tuple:
            for index in range(rays_per_sample):
                offset_deg = -half_fov + index * self.ray_angle_step_deg
                angle = pose.heading + math.radians(offset_deg)
                for ray_cell_index, cell in enumerate(
                    ray_cells_from_world(
                        pose.world_xy,
                        angle,
                        truth.geometry,
                        range_m=self.range_m,
                    )
                ):
                    center = truth.geometry.cell_to_world_center(cell)
                    if math.hypot(center.x - pose.world_xy.x, center.y - pose.world_xy.y) > self.range_m:
                        continue
                    visits += 1
                    visible.add(cell)
                    if ray_cell_index > 0 and (
                        truth.hard_obstacle[cell.y, cell.x]
                        or truth.slope_deg[cell.y, cell.x] > self.max_slope_deg
                    ):
                        break
        ordered_visible = tuple(sorted(visible, key=lambda cell: (cell.y, cell.x)))
        newly_observed = observed_state.reveal_cells(
            truth,
            ordered_visible,
            min_clearance_m=self.min_clearance_m,
            max_slope_deg=self.max_slope_deg,
            traversability_threshold=self.traversability_threshold,
        )
        diagnostics = SensorDiagnostics(
            sample_count=len(pose_tuple),
            ray_count=len(pose_tuple) * rays_per_sample,
            cell_visit_count=visits,
            unique_visible_cell_count=len(ordered_visible),
            duplicate_cell_visits=visits - len(ordered_visible),
            sample_sources=tuple(pose.source for pose in pose_tuple),
            sample_headings=tuple(pose.heading for pose in pose_tuple),
        )
        return ObservationDelta(newly_observed, ordered_visible, diagnostics)


def ray_cells_from_world(
    origin: WorldXY,
    angle: float,
    geometry: GridGeometry,
    *,
    range_m: float,
) -> tuple[CellXY, ...]:
    """Traverse intersected cells from the original world pose up to exact range."""

    current = geometry.world_to_cell(origin)
    direction_x = math.cos(angle)
    direction_y = math.sin(angle)
    step_x = 1 if direction_x > 0.0 else -1 if direction_x < 0.0 else 0
    step_y = 1 if direction_y > 0.0 else -1 if direction_y < 0.0 else 0
    resolution = geometry.resolution_m

    if step_x:
        boundary_x = geometry.origin.x + (current.x + (1 if step_x > 0 else 0)) * resolution
        t_max_x = (boundary_x - origin.x) / direction_x
        t_delta_x = resolution / abs(direction_x)
    else:
        t_max_x = math.inf
        t_delta_x = math.inf
    if step_y:
        boundary_y = geometry.origin.y + (current.y + (1 if step_y > 0 else 0)) * resolution
        t_max_y = (boundary_y - origin.y) / direction_y
        t_delta_y = resolution / abs(direction_y)
    else:
        t_max_y = math.inf
        t_delta_y = math.inf

    cells: list[CellXY] = []
    while geometry.in_bounds(current):
        cells.append(current)
        next_distance = min(t_max_x, t_max_y)
        if next_distance > range_m:
            break
        if math.isclose(t_max_x, t_max_y, rel_tol=0.0, abs_tol=1e-12):
            current = CellXY(current.x + step_x, current.y + step_y)
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        elif t_max_x < t_max_y:
            current = CellXY(current.x + step_x, current.y)
            t_max_x += t_delta_x
        else:
            current = CellXY(current.x, current.y + step_y)
            t_max_y += t_delta_y
    return tuple(cells)


def grid_line(start: CellXY, end: CellXY) -> tuple[CellXY, ...]:
    """Integer Bresenham line; callers apply bounds and blocker stopping."""

    x0, y0 = start.x, start.y
    x1, y1 = end.x, end.y
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    cells: list[CellXY] = []
    while True:
        cells.append(CellXY(x0, y0))
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy
    return tuple(cells)
