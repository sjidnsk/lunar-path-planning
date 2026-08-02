"""Stage 1 public grid and world coordinate types."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, order=True, slots=True)
class CellXY:
    """Grid cell in public ``(x, y)`` order."""

    x: int
    y: int


@dataclass(frozen=True, slots=True)
class WorldXY:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class PoseXYTheta:
    cell: CellXY
    theta: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta", normalize_theta(self.theta))


@dataclass(frozen=True, slots=True)
class GridGeometry:
    width: int
    height: int
    resolution_m: float
    origin: WorldXY = WorldXY(0.0, 0.0)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("grid width and height must be positive")
        if not math.isfinite(self.resolution_m) or self.resolution_m <= 0.0:
            raise ValueError("grid resolution must be finite and positive")

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def in_bounds(self, cell: CellXY) -> bool:
        return 0 <= cell.x < self.width and 0 <= cell.y < self.height

    def array_indices(self, cell: CellXY) -> tuple[int, int]:
        if not self.in_bounds(cell):
            raise ValueError(f"cell is outside grid: {cell}")
        return cell.y, cell.x

    def cell_to_world_center(self, cell: CellXY) -> WorldXY:
        if not self.in_bounds(cell):
            raise ValueError(f"cell is outside grid: {cell}")
        return WorldXY(
            self.origin.x + (cell.x + 0.5) * self.resolution_m,
            self.origin.y + (cell.y + 0.5) * self.resolution_m,
        )

    def world_to_cell(self, point: WorldXY) -> CellXY:
        if not math.isfinite(point.x) or not math.isfinite(point.y):
            raise ValueError("world coordinates must be finite")
        cell = CellXY(
            math.floor((point.x - self.origin.x) / self.resolution_m),
            math.floor((point.y - self.origin.y) / self.resolution_m),
        )
        if not self.in_bounds(cell):
            raise ValueError(f"world coordinate is outside grid: {point}")
        return cell

    def raster_row_to_cell_y(self, row: int) -> int:
        if not 0 <= row < self.height:
            raise ValueError("raster row is outside grid")
        return self.height - 1 - row

    def cell_y_to_raster_row(self, y: int) -> int:
        if not 0 <= y < self.height:
            raise ValueError("cell y is outside grid")
        return self.height - 1 - y


def normalize_theta(theta: float) -> float:
    if not math.isfinite(theta):
        raise ValueError("theta must be finite")
    normalized = (theta + math.pi) % (2.0 * math.pi) - math.pi
    return float(normalized)
