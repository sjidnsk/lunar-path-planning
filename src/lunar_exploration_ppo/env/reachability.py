"""Observed and truth grid reachability with corner-safe 8-connectivity."""

from __future__ import annotations

from collections import deque

import numpy as np

from lunar_exploration_ppo.utils.geometry import CellXY


_DIRECTIONS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
    (1, 1),
)


def reachable_component(passable_mask: np.ndarray, start: CellXY) -> np.ndarray:
    passable = np.asarray(passable_mask, dtype=bool)
    if passable.ndim != 2:
        raise ValueError("passable mask must be two-dimensional")
    height, width = passable.shape
    result = np.zeros_like(passable)
    if not (0 <= start.x < width and 0 <= start.y < height) or not passable[start.y, start.x]:
        return result
    result[start.y, start.x] = True
    queue: deque[CellXY] = deque((start,))
    while queue:
        current = queue.popleft()
        for dx, dy in _DIRECTIONS:
            x = current.x + dx
            y = current.y + dy
            if not (0 <= x < width and 0 <= y < height):
                continue
            if result[y, x] or not passable[y, x]:
                continue
            if dx and dy:
                if not passable[current.y, x] or not passable[y, current.x]:
                    continue
            result[y, x] = True
            queue.append(CellXY(x, y))
    return result
