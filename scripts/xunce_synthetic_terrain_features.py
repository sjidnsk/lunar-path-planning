from __future__ import annotations

import hashlib
import json
import math
import random
from collections import deque
from collections.abc import Iterable, Sequence
from typing import Any


Cell = tuple[int, int]

SYNTHETIC_TERRAIN_MODEL_ID = "synthetic_rock_pit_terrain/v1"
SYNTHETIC_SOURCE_KIND = "synthetic_terrain_obstacle_proxy/v1"


def generate_synthetic_terrain_augmentation(
    *,
    width: int,
    height: int,
    resolution_m: float,
    start_cell: Sequence[int],
    existing_hard_obstacle_cells: Iterable[Sequence[int]] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate deterministic synthetic rock and pit obstacle layers.

    The output intentionally labels every generated obstacle as synthetic proxy
    evidence. It never emits physical_obstacle_cells.
    """

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")

    seed = int(config.get("synthetic_terrain_seed", 260001))
    rng = random.Random(seed)
    start = _cell_tuple(start_cell)
    existing = _normalized_cells(existing_hard_obstacle_cells or [])
    max_blocked_fraction = float(config.get("max_blocked_fraction", 0.35))
    min_start_clearance_m = float(config.get("min_start_clearance_m", 3.0))
    min_start_clearance_cells = max(0.0, min_start_clearance_m / resolution_m)

    rocks: list[dict[str, Any]] = []
    pits: list[dict[str, Any]] = []
    synthetic_rock_cells: set[Cell] = set()
    synthetic_rock_hard: set[Cell] = set()
    synthetic_rock_los: set[Cell] = set()
    synthetic_pit_cells: set[Cell] = set()
    synthetic_pit_rim: set[Cell] = set()
    synthetic_pit_interior: set[Cell] = set()
    synthetic_pit_hard: set[Cell] = set()
    synthetic_pit_high_risk: set[Cell] = set()
    synthetic_pit_los: set[Cell] = set()

    hard_limit = max(0, int(math.floor(width * height * max_blocked_fraction)))

    if bool(config.get("rock_generation_enabled", True)):
        target_count = _randint_range(rng, config.get("rock_count_range", [20, 120]))
        for index in range(target_count):
            feature = _random_feature_base(
                rng,
                width=width,
                height=height,
                resolution_m=resolution_m,
                prefix="rock",
                index=index,
                radius_range=config.get("rock_radius_m_range", [0.5, 6.0]),
            )
            if feature is None:
                continue
            cells = _irregular_blob_cells(
                center=tuple(feature["center_cell"]),
                radius_m=float(feature["radius_m"]),
                resolution_m=resolution_m,
                width=width,
                height=height,
                rng=rng,
                irregularity=float(config.get("rock_irregularity", 0.25)),
            )
            cells = _filter_cells(cells, start=start, min_start_clearance_cells=min_start_clearance_cells)
            if not cells:
                continue
            height_m = _uniform_range(rng, config.get("rock_height_m_range", [0.2, 2.5]))
            los = set(cells) if height_m >= float(config.get("rock_los_blocker_height_threshold_m", 0.5)) else set()
            hard = set(cells)
            accepted_hard = _bounded_addition(existing, synthetic_rock_hard | synthetic_pit_hard, hard, hard_limit)
            if not accepted_hard:
                continue
            accepted_cells = set(accepted_hard)
            accepted_los = los & accepted_cells
            rock_row = {
                **feature,
                "feature_type": "synthetic_rock",
                "height_m": round(height_m, 6),
                "hard_obstacle": True,
                "los_blocker": bool(accepted_los),
                "cells": _sorted_cells(accepted_cells),
                "hard_obstacle_cells": _sorted_cells(accepted_hard),
                "los_blocker_cells": _sorted_cells(accepted_los),
                "source_kind": "synthetic_rock_obstacle/v1",
            }
            rocks.append(rock_row)
            synthetic_rock_cells.update(accepted_cells)
            synthetic_rock_hard.update(accepted_hard)
            synthetic_rock_los.update(accepted_los)

    if bool(config.get("pit_generation_enabled", True)):
        target_count = _randint_range(rng, config.get("pit_count_range", [5, 40]))
        for index in range(target_count):
            feature = _random_feature_base(
                rng,
                width=width,
                height=height,
                resolution_m=resolution_m,
                prefix="pit",
                index=index,
                radius_range=config.get("pit_radius_m_range", [1.5, 10.0]),
            )
            if feature is None:
                continue
            radius_m = float(feature["radius_m"])
            rim_width_m = _uniform_range(rng, config.get("pit_rim_width_m_range", [0.5, 2.0]))
            all_cells, interior, rim = _pit_cells(
                center=tuple(feature["center_cell"]),
                radius_m=radius_m,
                rim_width_m=rim_width_m,
                resolution_m=resolution_m,
                width=width,
                height=height,
            )
            all_cells = _filter_cells(all_cells, start=start, min_start_clearance_cells=min_start_clearance_cells)
            interior &= all_cells
            rim &= all_cells
            if not all_cells:
                continue
            depth_m = _uniform_range(rng, config.get("pit_depth_m_range", [0.3, 3.0]))
            hard = set(interior) if depth_m >= float(config.get("pit_hard_depth_threshold_m", 0.8)) else set()
            high_risk = set(all_cells)
            los = set(rim) if bool(config.get("pit_los_blocker_rim_enabled", True)) and hard else set()
            accepted_hard = _bounded_addition(existing, synthetic_rock_hard | synthetic_pit_hard, hard, hard_limit)
            accepted_cells = (set(all_cells) - hard) | accepted_hard
            if not accepted_cells:
                continue
            accepted_interior = interior & accepted_cells
            accepted_rim = rim & accepted_cells
            accepted_los = los & accepted_cells
            pit_row = {
                **feature,
                "feature_type": "synthetic_pit",
                "depth_m": round(depth_m, 6),
                "rim_width_m": round(rim_width_m, 6),
                "interior_hard_obstacle": bool(accepted_hard),
                "rim_los_blocker": bool(accepted_los),
                "cells": _sorted_cells(accepted_cells),
                "rim_cells": _sorted_cells(accepted_rim),
                "interior_cells": _sorted_cells(accepted_interior),
                "hard_obstacle_cells": _sorted_cells(accepted_hard),
                "high_risk_cells": _sorted_cells(high_risk & accepted_cells),
                "los_blocker_cells": _sorted_cells(accepted_los),
                "source_kind": "synthetic_pit_obstacle/v1",
            }
            pits.append(pit_row)
            synthetic_pit_cells.update(accepted_cells)
            synthetic_pit_rim.update(accepted_rim)
            synthetic_pit_interior.update(accepted_interior)
            synthetic_pit_hard.update(accepted_hard)
            synthetic_pit_high_risk.update(high_risk & accepted_cells)
            synthetic_pit_los.update(accepted_los)

    synthetic_hard = synthetic_rock_hard | synthetic_pit_hard
    synthetic_los = synthetic_rock_los | synthetic_pit_los
    synthetic_high_risk = synthetic_pit_high_risk
    synthetic_cost = synthetic_high_risk - synthetic_hard
    effective_hard = existing | synthetic_hard
    synthetic_connectivity = connectivity_audit(
        width=width,
        height=height,
        start_cell=start,
        blocked_cells=synthetic_hard,
        enabled=bool(config.get("connectivity_check_enabled", True)),
    )
    effective_connectivity = connectivity_audit(
        width=width,
        height=height,
        start_cell=start,
        blocked_cells=effective_hard,
        enabled=bool(config.get("connectivity_check_enabled", True)),
    )
    synthetic_blocked_fraction = len(synthetic_hard) / float(width * height)
    effective_blocked_fraction = len(effective_hard) / float(width * height)
    synthetic_start_clearance_passed = not _any_within_clearance(synthetic_hard, start, min_start_clearance_cells)
    effective_start_clearance_passed = not _any_within_clearance(effective_hard, start, min_start_clearance_cells)

    source = {
        "schema_version": "xunce-synthetic-terrain-obstacle-source/v1",
        "synthetic_terrain_model_id": SYNTHETIC_TERRAIN_MODEL_ID,
        "source_kind": SYNTHETIC_SOURCE_KIND,
        "synthetic_terrain_seed": seed,
        "synthetic_rock_count": len(rocks),
        "synthetic_pit_count": len(pits),
        "width": int(width),
        "height": int(height),
        "resolution_m": float(resolution_m),
        "synthetic_rock_cells": _sorted_cells(synthetic_rock_cells),
        "synthetic_rock_hard_obstacle_cells": _sorted_cells(synthetic_rock_hard),
        "synthetic_rock_los_blocker_cells": _sorted_cells(synthetic_rock_los),
        "synthetic_pit_cells": _sorted_cells(synthetic_pit_cells),
        "synthetic_pit_rim_cells": _sorted_cells(synthetic_pit_rim),
        "synthetic_pit_interior_cells": _sorted_cells(synthetic_pit_interior),
        "synthetic_pit_hard_obstacle_cells": _sorted_cells(synthetic_pit_hard),
        "synthetic_pit_high_risk_cells": _sorted_cells(synthetic_pit_high_risk),
        "synthetic_pit_los_blocker_cells": _sorted_cells(synthetic_pit_los),
        "synthetic_hard_obstacle_cells": _sorted_cells(synthetic_hard),
        "synthetic_los_blocker_cells": _sorted_cells(synthetic_los),
        "synthetic_high_risk_cells": _sorted_cells(synthetic_high_risk),
        "synthetic_cost_inflation_cells": _sorted_cells(synthetic_cost),
        "physical_obstacle_cells_written": False,
    }
    source["synthetic_terrain_hash"] = stable_synthetic_terrain_hash(source)
    catalog = [*rocks, *pits]
    for row in catalog:
        row["synthetic_terrain_hash"] = source["synthetic_terrain_hash"]
        row["synthetic_terrain_model_id"] = SYNTHETIC_TERRAIN_MODEL_ID

    audit = {
        "schema_version": "xunce-synthetic-terrain-map-augmentation-audit/v1",
        "synthetic_terrain_model_id": SYNTHETIC_TERRAIN_MODEL_ID,
        "synthetic_terrain_seed": seed,
        "synthetic_terrain_hash": source["synthetic_terrain_hash"],
        "synthetic_rock_count": len(rocks),
        "synthetic_pit_count": len(pits),
        "synthetic_hard_obstacle_cell_count": len(synthetic_hard),
        "synthetic_los_blocker_cell_count": len(synthetic_los),
        "synthetic_high_risk_cell_count": len(synthetic_high_risk),
        "synthetic_cost_inflation_cell_count": len(synthetic_cost),
        "existing_hard_obstacle_cell_count": len(existing),
        "effective_hard_obstacle_cell_count": len(effective_hard),
        "synthetic_blocked_fraction": synthetic_blocked_fraction,
        "effective_blocked_fraction": effective_blocked_fraction,
        "blocked_fraction": synthetic_blocked_fraction,
        "max_blocked_fraction": max_blocked_fraction,
        "blocked_fraction_passed": synthetic_blocked_fraction <= max_blocked_fraction,
        "effective_blocked_fraction_passed": effective_blocked_fraction <= max_blocked_fraction,
        "start_cell": [start[0], start[1]],
        "min_start_clearance_m": min_start_clearance_m,
        "start_clearance_passed": synthetic_start_clearance_passed,
        "effective_start_clearance_passed": effective_start_clearance_passed,
        "connectivity_passed": synthetic_connectivity["connectivity_passed"],
        "connectivity_audit": synthetic_connectivity,
        "effective_connectivity_passed": effective_connectivity["connectivity_passed"],
        "effective_connectivity_audit": effective_connectivity,
        "physical_obstacle_cells_written": False,
        "source_kind": SYNTHETIC_SOURCE_KIND,
    }
    return {"source": source, "catalog": catalog, "audit": audit}


def stable_synthetic_terrain_hash(source: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in source.items()
        if key not in {"synthetic_terrain_hash"}
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def augmented_sidecar_with_synthetic_source(sidecar: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    augmented = json.loads(json.dumps(sidecar, ensure_ascii=False))
    augmented["synthetic_terrain_augmentation_enabled"] = True
    augmented["synthetic_terrain_model_id"] = source["synthetic_terrain_model_id"]
    augmented["synthetic_terrain_seed"] = source["synthetic_terrain_seed"]
    augmented["synthetic_terrain_hash"] = source["synthetic_terrain_hash"]
    augmented["synthetic_source_kind"] = source["source_kind"]
    for key in (
        "synthetic_rock_cells",
        "synthetic_rock_hard_obstacle_cells",
        "synthetic_rock_los_blocker_cells",
        "synthetic_pit_cells",
        "synthetic_pit_rim_cells",
        "synthetic_pit_interior_cells",
        "synthetic_pit_hard_obstacle_cells",
        "synthetic_pit_high_risk_cells",
        "synthetic_pit_los_blocker_cells",
        "synthetic_hard_obstacle_cells",
        "synthetic_los_blocker_cells",
        "synthetic_high_risk_cells",
        "synthetic_cost_inflation_cells",
    ):
        augmented[key] = source.get(key, [])
    augmented["physical_obstacle_cells_written_by_synthetic"] = False
    return augmented


def connectivity_audit(
    *,
    width: int,
    height: int,
    start_cell: Cell,
    blocked_cells: Iterable[Sequence[int]],
    enabled: bool,
) -> dict[str, Any]:
    blocked = _normalized_cells(blocked_cells)
    if not enabled:
        return {
            "connectivity_check_enabled": False,
            "connectivity_passed": True,
            "reachable_cell_count": None,
            "reachable_fraction": None,
        }
    if start_cell in blocked or not _in_bounds(start_cell, width, height):
        return {
            "connectivity_check_enabled": True,
            "connectivity_passed": False,
            "reachable_cell_count": 0,
            "reachable_fraction": 0.0,
            "failure_reason": "start_blocked_or_out_of_bounds",
        }
    visited = {start_cell}
    queue: deque[Cell] = deque([start_cell])
    while queue:
        x, y = queue.popleft()
        for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if neighbor in visited or neighbor in blocked or not _in_bounds(neighbor, width, height):
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    passable_count = width * height - len(blocked)
    reachable_fraction = len(visited) / float(passable_count) if passable_count > 0 else 0.0
    return {
        "connectivity_check_enabled": True,
        "connectivity_passed": reachable_fraction >= 0.5,
        "reachable_cell_count": len(visited),
        "reachable_fraction": reachable_fraction,
        "passable_cell_count": passable_count,
    }


def _random_feature_base(
    rng: random.Random,
    *,
    width: int,
    height: int,
    resolution_m: float,
    prefix: str,
    index: int,
    radius_range: Any,
) -> dict[str, Any] | None:
    if width <= 0 or height <= 0:
        return None
    center = (rng.randrange(width), rng.randrange(height))
    radius_m = _uniform_range(rng, radius_range)
    return {
        "feature_id": f"{prefix}_{index:05d}",
        "center_cell": [center[0], center[1]],
        "radius_m": round(max(radius_m, resolution_m * 0.25), 6),
        "shape": "irregular_blob" if prefix == "rock" else "rimmed_depression",
    }


def _irregular_blob_cells(
    *,
    center: Cell,
    radius_m: float,
    resolution_m: float,
    width: int,
    height: int,
    rng: random.Random,
    irregularity: float,
) -> set[Cell]:
    radius_cells = max(1.0, radius_m / resolution_m)
    rx = max(1.0, radius_cells * rng.uniform(0.75, 1.35))
    ry = max(1.0, radius_cells * rng.uniform(0.75, 1.35))
    angle = rng.uniform(0.0, math.pi)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    max_r = int(math.ceil(max(rx, ry) + 1.0))
    result: set[Cell] = set()
    cx, cy = center
    for x in range(cx - max_r, cx + max_r + 1):
        for y in range(cy - max_r, cy + max_r + 1):
            if not _in_bounds((x, y), width, height):
                continue
            dx = x - cx
            dy = y - cy
            ux = cos_a * dx + sin_a * dy
            uy = -sin_a * dx + cos_a * dy
            norm = (ux / rx) ** 2 + (uy / ry) ** 2
            noise = rng.uniform(-irregularity, irregularity)
            if norm <= 1.0 + noise:
                result.add((x, y))
    return result


def _pit_cells(
    *,
    center: Cell,
    radius_m: float,
    rim_width_m: float,
    resolution_m: float,
    width: int,
    height: int,
) -> tuple[set[Cell], set[Cell], set[Cell]]:
    radius_cells = max(1.0, radius_m / resolution_m)
    rim_width_cells = max(1.0, rim_width_m / resolution_m)
    inner_radius = max(0.0, radius_cells - rim_width_cells)
    cx, cy = center
    max_r = int(math.ceil(radius_cells + 1.0))
    all_cells: set[Cell] = set()
    interior: set[Cell] = set()
    rim: set[Cell] = set()
    for x in range(cx - max_r, cx + max_r + 1):
        for y in range(cy - max_r, cy + max_r + 1):
            if not _in_bounds((x, y), width, height):
                continue
            dist = math.hypot(x - cx, y - cy)
            if dist <= radius_cells:
                all_cells.add((x, y))
                if dist <= inner_radius:
                    interior.add((x, y))
                else:
                    rim.add((x, y))
    if not interior and center in all_cells:
        interior.add(center)
        rim.discard(center)
    return all_cells, interior, rim


def _bounded_addition(existing: set[Cell], current: set[Cell], proposed: set[Cell], hard_limit: int) -> set[Cell]:
    if not proposed:
        return set()
    available = max(0, hard_limit - len(current))
    if available <= 0:
        return set()
    new_cells = sorted(cell for cell in proposed if cell not in existing and cell not in current)
    return set(new_cells[:available])


def _filter_cells(cells: set[Cell], *, start: Cell, min_start_clearance_cells: float) -> set[Cell]:
    return {cell for cell in cells if math.hypot(cell[0] - start[0], cell[1] - start[1]) > min_start_clearance_cells}


def _any_within_clearance(cells: set[Cell], start: Cell, clearance_cells: float) -> bool:
    return any(math.hypot(cell[0] - start[0], cell[1] - start[1]) <= clearance_cells for cell in cells)


def _randint_range(rng: random.Random, value: Any) -> int:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        left, right = int(value[0]), int(value[1])
        return rng.randint(min(left, right), max(left, right))
    return int(value)


def _uniform_range(rng: random.Random, value: Any) -> float:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        left, right = float(value[0]), float(value[1])
        return rng.uniform(min(left, right), max(left, right))
    return float(value)


def _normalized_cells(cells: Iterable[Sequence[int]]) -> set[Cell]:
    return {_cell_tuple(cell) for cell in cells}


def _sorted_cells(cells: Iterable[Cell]) -> list[list[int]]:
    return [[int(x), int(y)] for x, y in sorted(set(cells))]


def _cell_tuple(value: Sequence[int]) -> Cell:
    return (int(value[0]), int(value[1]))


def _in_bounds(cell: Cell, width: int, height: int) -> bool:
    return 0 <= cell[0] < width and 0 <= cell[1] < height
