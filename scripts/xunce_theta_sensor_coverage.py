from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from typing import Any


Cell = tuple[int, int]


def theta_bins(*, theta_bin_count: int = 8, theta_step_deg: int = 45) -> list[int]:
    if theta_bin_count <= 0:
        raise ValueError("theta_bin_count must be positive")
    if theta_step_deg <= 0:
        raise ValueError("theta_step_deg must be positive")
    return [int((index * theta_step_deg) % 360) for index in range(theta_bin_count)]


def visible_cells_for_viewpoint(
    cell: Sequence[int],
    *,
    theta_deg: float,
    sensor_range_cells: int,
    sensor_fov_deg: float,
    bounds: tuple[int, int, int, int] | None = None,
    valid_cells: set[Cell] | None = None,
) -> set[Cell]:
    center = _cell_tuple(cell)
    if center is None:
        raise ValueError("cell must be an (x, y) pair")
    if sensor_range_cells < 0:
        raise ValueError("sensor_range_cells must be non-negative")
    if sensor_fov_deg <= 0:
        raise ValueError("sensor_fov_deg must be positive")

    x0, y0 = center
    theta = float(theta_deg) % 360.0
    half_fov = min(float(sensor_fov_deg), 360.0) / 2.0
    visible: set[Cell] = set()
    for dx in range(-sensor_range_cells, sensor_range_cells + 1):
        for dy in range(-sensor_range_cells, sensor_range_cells + 1):
            candidate = (x0 + dx, y0 + dy)
            if bounds is not None and not _inside_bounds(candidate, bounds):
                continue
            if valid_cells is not None and candidate not in valid_cells:
                continue
            if math.hypot(dx, dy) > float(sensor_range_cells) + 1.0e-9:
                continue
            if dx == 0 and dy == 0:
                visible.add(candidate)
                continue
            angle = math.degrees(math.atan2(dy, dx)) % 360.0
            if _angle_distance_deg(angle, theta) <= half_fov + 1.0e-9:
                visible.add(candidate)
    return visible


def theta_coverage_hash(cells: Iterable[Sequence[int]]) -> str:
    normalized = sorted(list(cell) for cell in {_require_cell_tuple(cell) for cell in cells})
    return _hash_payload(normalized)


def expand_candidate_viewpoints(
    candidate_cells: Iterable[Any],
    *,
    theta_values: Iterable[int],
    sensor_model_id: str,
    sensor_range_cells: int,
    sensor_fov_deg: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_index, raw_cell in enumerate(candidate_cells):
        cell = _cell_tuple(raw_cell)
        if cell is None:
            continue
        for theta in theta_values:
            rows.append(
                {
                    "candidate_index": int(candidate_index),
                    "candidate_cell": [int(cell[0]), int(cell[1])],
                    "candidate_theta_deg": int(theta),
                    "candidate_viewpoint": [int(cell[0]), int(cell[1]), int(theta)],
                    "sensor_model_id": sensor_model_id,
                    "sensor_range_cells": int(sensor_range_cells),
                    "sensor_fov_deg": float(sensor_fov_deg),
                }
            )
    return rows


def candidate_viewpoint_hash(viewpoints: Iterable[dict[str, Any]]) -> str:
    payload = [
        {
            "candidate_viewpoint": row.get("candidate_viewpoint"),
            "sensor_model_id": row.get("sensor_model_id"),
            "sensor_range_cells": row.get("sensor_range_cells"),
            "sensor_fov_deg": row.get("sensor_fov_deg"),
        }
        for row in viewpoints
    ]
    return _hash_payload(payload)


def _angle_distance_deg(left: float, right: float) -> float:
    delta = abs((left - right) % 360.0)
    return min(delta, 360.0 - delta)


def _inside_bounds(cell: Cell, bounds: tuple[int, int, int, int]) -> bool:
    min_x, min_y, max_x, max_y = bounds
    return min_x <= cell[0] <= max_x and min_y <= cell[1] <= max_y


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


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
