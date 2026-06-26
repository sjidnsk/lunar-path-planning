from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

try:
    from xunce_theta_sensor_coverage import visible_cells_for_viewpoint
except ModuleNotFoundError:  # pragma: no cover
    from scripts.xunce_theta_sensor_coverage import visible_cells_for_viewpoint


Cell = tuple[int, int]


@dataclass(frozen=True)
class ObstacleAwareVisibilityResult:
    visible_cells: set[Cell]
    occluded_cells: set[Cell]
    blocked_obstacle_cells: set[Cell]

    @property
    def occluded_cell_count(self) -> int:
        return len(self.occluded_cells)

    @property
    def blocked_by_obstacle_count(self) -> int:
        return len(self.blocked_obstacle_cells)

    @property
    def obstacle_aware_theta_coverage_hash(self) -> str:
        return obstacle_aware_theta_coverage_hash(self.visible_cells)


def visible_cells_for_viewpoint_with_obstacles(
    cell: Sequence[int],
    *,
    theta_deg: float,
    sensor_range_cells: int,
    sensor_fov_deg: float,
    obstacle_cells: Iterable[Sequence[int]] | None,
    bounds: tuple[int, int, int, int] | None = None,
    valid_cells: set[Cell] | None = None,
) -> ObstacleAwareVisibilityResult:
    center = _require_cell_tuple(cell)
    base_visible = visible_cells_for_viewpoint(
        center,
        theta_deg=theta_deg,
        sensor_range_cells=sensor_range_cells,
        sensor_fov_deg=sensor_fov_deg,
        bounds=bounds,
        valid_cells=valid_cells,
    )
    obstacles = {_require_cell_tuple(raw) for raw in obstacle_cells or []}
    if not obstacles:
        return ObstacleAwareVisibilityResult(
            visible_cells=set(base_visible),
            occluded_cells=set(),
            blocked_obstacle_cells=set(),
        )

    visible: set[Cell] = set()
    occluded: set[Cell] = set()
    blocked_obstacles: set[Cell] = set()
    for target in base_visible:
        if target == center:
            visible.add(target)
            continue
        if target in obstacles:
            blocked_obstacles.add(target)
            continue
        if _line_of_sight_blocked(center, target, obstacles):
            occluded.add(target)
            continue
        visible.add(target)

    return ObstacleAwareVisibilityResult(
        visible_cells=visible,
        occluded_cells=occluded,
        blocked_obstacle_cells=blocked_obstacles,
    )


def obstacle_aware_theta_coverage_hash(cells: Iterable[Sequence[int]]) -> str:
    normalized = sorted(list(_require_cell_tuple(cell)) for cell in cells)
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def stable_obstacle_source_hash(
    *,
    source_kind: str,
    obstacle_cells: Iterable[Sequence[int]],
    no_go_blocks_los: bool = False,
) -> str:
    payload = {
        "source_kind": str(source_kind),
        "no_go_blocks_los": bool(no_go_blocks_los),
        "obstacle_cells": sorted(list(_require_cell_tuple(cell)) for cell in obstacle_cells),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def extract_obstacle_cells(payload: dict[str, Any], *, include_no_go: bool = False) -> tuple[set[Cell], str | None]:
    for key in ("obstacle_cells",):
        cells = _cells_from_payload(payload.get(key))
        if cells:
            return cells, key
    for key in ("obstacle_rectangles",):
        rectangles = _cells_from_rectangles(payload.get(key))
        if rectangles:
            return rectangles, key
    for key in ("slope_blocked_cells",):
        cells = _cells_from_payload(payload.get(key))
        if cells:
            return cells, key
    for key in ("blocked_cells",):
        cells = _cells_from_payload(payload.get(key))
        if cells:
            return cells, key
    for key in ("blocked_rectangles",):
        rectangles = _cells_from_rectangles(payload.get(key))
        if rectangles:
            return rectangles, key
    for key in ("synthetic_los_blocker_cells",):
        cells = _cells_from_payload(payload.get(key))
        if cells:
            return cells, key
    for key in ("synthetic_hard_obstacle_cells",):
        cells = _cells_from_payload(payload.get(key))
        if cells:
            return cells, key
    if include_no_go:
        cells = _cells_from_payload(payload.get("no_go_cells"))
        if cells:
            return cells, "no_go_cells"
    if include_no_go:
        rectangles = _cells_from_rectangles(payload.get("no_go_rectangles"))
        if rectangles:
            return rectangles, "no_go_rectangles"

    return set(), None


def _line_of_sight_blocked(start: Cell, end: Cell, obstacle_cells: set[Cell]) -> bool:
    cells = _line_cells(start, end)
    for line_cell in cells[1:-1]:
        if line_cell in obstacle_cells:
            return True
    return False


def _line_cells(start: Cell, end: Cell) -> list[Cell]:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    cells: list[Cell] = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        error2 = 2 * err
        if error2 > -dy:
            err -= dy
            x0 += sx
        if error2 < dx:
            err += dx
            y0 += sy
    return cells


def _cells_from_payload(value: Any) -> set[Cell]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, dict)):
        return set()
    cells: set[Cell] = set()
    for raw in value:
        cell = _cell_tuple(raw)
        if cell is not None:
            cells.add(cell)
    return cells


def _cells_from_rectangles(value: Any) -> set[Cell]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, dict)):
        return set()
    cells: set[Cell] = set()
    for raw in value:
        bounds = _rectangle_bounds(raw)
        if bounds is None:
            continue
        min_x, min_y, max_x, max_y = bounds
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                cells.add((x, y))
    return cells


def _rectangle_bounds(value: Any) -> tuple[int, int, int, int] | None:
    if isinstance(value, dict):
        if all(key in value for key in ("min_x", "min_y", "max_x", "max_y")):
            raw = (value["min_x"], value["min_y"], value["max_x"], value["max_y"])
        elif all(key in value for key in ("x0", "y0", "x1", "y1")):
            raw = (value["x0"], value["y0"], value["x1"], value["y1"])
        else:
            return None
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 4:
        raw = (value[0], value[1], value[2], value[3])
    else:
        return None
    try:
        x0, y0, x1, y1 = (int(item) for item in raw)
    except (TypeError, ValueError):
        return None
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _cell_tuple(value: Any) -> Cell | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    try:
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return None


def _require_cell_tuple(value: Any) -> Cell:
    cell = _cell_tuple(value)
    if cell is None:
        raise ValueError("cell must be an (x, y) pair")
    return cell
