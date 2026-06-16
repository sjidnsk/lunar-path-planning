from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GLOBAL_99_CONFIG_SCHEMA_VERSION = "global-99-coverage-benchmark-config/v1"
GLOBAL_99_LEDGER_ROW_SCHEMA_VERSION = "global-99-coverage-ledger-row/v1"
TOLERANCE = 1.0e-12

Cell = tuple[int, int]


class ConfigError(ValueError):
    pass


def load_global_99_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != GLOBAL_99_CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {GLOBAL_99_CONFIG_SCHEMA_VERSION!r}")
    target = float_value(payload.get("target_coverage_rate"), "target_coverage_rate")
    if target <= 0.0 or target > 1.0:
        raise ConfigError("target_coverage_rate must be > 0 and <= 1")
    path_budget = float_value(payload.get("path_budget_m"), "path_budget_m")
    if path_budget < 0.0:
        raise ConfigError("path_budget_m must be >= 0")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ConfigError("scenarios must be a non-empty list")
    normalized = dict(payload)
    normalized["target_coverage_rate"] = target
    normalized["path_budget_m"] = path_budget
    normalized["scenarios"] = [_validate_scenario_config(item, index) for index, item in enumerate(scenarios)]
    return normalized


def scenario_geometry(scenario: dict[str, Any]) -> dict[str, Any]:
    scenario_id = str(scenario["scenario_id"])
    grid = scenario["grid"]
    width = int(grid["width"])
    height = int(grid["height"])
    resolution_m = float(grid.get("resolution_m", 1.0))
    all_cells = {(x, y) for y in range(height) for x in range(width)}
    roi_cells = roi_cells_from_config(scenario["roi"], width, height)
    blocked_cells = rectangles_to_cells(scenario.get("blocked_rectangles", []), width, height)
    unsafe_cells = rectangles_to_cells(scenario.get("unsafe_rectangles", []), width, height)
    excluded_cells = blocked_cells | unsafe_cells
    start = cell_from_config(scenario["start_cell"], "start_cell")
    reachable_cells = reachable_cells_from(start, width, height, excluded_cells)
    safe_roi_cells = roi_cells - excluded_cells
    reachable_safe_cells = safe_roi_cells & reachable_cells
    unsafe_roi_cells = roi_cells & excluded_cells
    unreachable_roi_cells = safe_roi_cells - reachable_cells
    return {
        "scenario_id": scenario_id,
        "width": width,
        "height": height,
        "resolution_m": resolution_m,
        "all_cells": all_cells,
        "roi_cells": roi_cells,
        "blocked_cells": blocked_cells,
        "unsafe_cells": unsafe_cells,
        "excluded_cells": excluded_cells,
        "start": start,
        "navigation_cells": reachable_cells,
        "reachable_cells": reachable_cells,
        "safe_roi_cells": safe_roi_cells,
        "reachable_safe_cells": reachable_safe_cells,
        "unsafe_roi_cells": unsafe_roi_cells,
        "unreachable_roi_cells": unreachable_roi_cells,
    }


def evaluate_static_coverage_scenario(
    scenario: dict[str, Any],
    *,
    target_coverage_rate: float,
    path_budget_m: float,
) -> dict[str, Any]:
    geometry = scenario_geometry(scenario)
    scenario_id = geometry["scenario_id"]
    width = geometry["width"]
    height = geometry["height"]
    all_cells = geometry["all_cells"]
    reachable_safe_cells = geometry["reachable_safe_cells"]
    unsafe_roi_cells = geometry["unsafe_roi_cells"]
    unreachable_roi_cells = geometry["unreachable_roi_cells"]

    covered: set[Cell] = set()
    ledger_rows: list[dict[str, Any]] = []
    cumulative_path_cost_m = 0.0
    path_budget_exhausted = False
    coverage_ledger_complete = True
    seen_event_ids: set[str] = set()
    events = scenario.get("coverage_events", [])
    for event_index, event in enumerate(events):
        event_id = str(event.get("event_id", f"event-{event_index:04d}"))
        if event_id in seen_event_ids:
            coverage_ledger_complete = False
        seen_event_ids.add(event_id)
        path_cost_m = float(event.get("path_cost_m", 0.0))
        attempted_cumulative = cumulative_path_cost_m + path_cost_m
        counted = not path_budget_exhausted and attempted_cumulative <= path_budget_m + TOLERANCE
        event_cells = coverage_event_cells(event, width, height) & all_cells
        event_reachable_safe_cells = event_cells & reachable_safe_cells
        before_count = len(covered)
        if counted:
            cumulative_path_cost_m = attempted_cumulative
            covered |= event_reachable_safe_cells
        else:
            path_budget_exhausted = True
        new_count = len(covered) - before_count
        denominator_count = len(reachable_safe_cells)
        ledger_rows.append(
            {
                "schema_version": GLOBAL_99_LEDGER_ROW_SCHEMA_VERSION,
                "scenario_id": scenario_id,
                "event_index": event_index,
                "event_id": event_id,
                "path_cost_m": path_cost_m,
                "attempted_cumulative_path_cost_m": attempted_cumulative,
                "cumulative_path_cost_m": cumulative_path_cost_m,
                "path_budget_m": path_budget_m,
                "counted": counted,
                "event_cell_count": len(event_cells),
                "event_reachable_safe_cell_count": len(event_reachable_safe_cells),
                "new_covered_reachable_safe_cell_count": new_count,
                "cumulative_covered_reachable_safe_cell_count": len(covered),
                "reachable_safe_cell_count": denominator_count,
                "achieved_coverage_rate": (len(covered) / denominator_count) if denominator_count else 0.0,
            }
        )

    return summarize_scenario_coverage(
        scenario_id=scenario_id,
        target_coverage_rate=target_coverage_rate,
        reachable_safe_cells=reachable_safe_cells,
        covered_cells=covered,
        path_budget_exhausted=path_budget_exhausted,
        coverage_ledger_complete=coverage_ledger_complete,
        roi_cell_count=len(geometry["roi_cells"]),
        blocked_roi_cell_count=len(geometry["roi_cells"] & geometry["blocked_cells"]),
        unsafe_roi_cell_count=len(geometry["roi_cells"] & geometry["unsafe_cells"]),
        excluded_roi_cell_count=len(unsafe_roi_cells),
        unreachable_roi_cell_count=len(unreachable_roi_cells),
        unsafe_roi_cell_present=bool(unsafe_roi_cells),
        unreachable_roi_cell_present=bool(unreachable_roi_cells),
        grid={
            "width": geometry["width"],
            "height": geometry["height"],
            "resolution_m": geometry["resolution_m"],
            "cell_count": geometry["width"] * geometry["height"],
        },
        ledger_rows=ledger_rows,
    )


def summarize_scenario_coverage(
    *,
    scenario_id: str,
    target_coverage_rate: float,
    reachable_safe_cells: set[Cell],
    covered_cells: set[Cell],
    path_budget_exhausted: bool,
    coverage_ledger_complete: bool,
    roi_cell_count: int,
    blocked_roi_cell_count: int,
    unsafe_roi_cell_count: int,
    excluded_roi_cell_count: int,
    unreachable_roi_cell_count: int,
    unsafe_roi_cell_present: bool,
    unreachable_roi_cell_present: bool,
    grid: dict[str, Any],
    ledger_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reachable_safe_count = len(reachable_safe_cells)
    covered_count = len(covered_cells & reachable_safe_cells)
    achieved = (covered_count / reachable_safe_count) if reachable_safe_count else 0.0
    denominator_valid = reachable_safe_count > 0
    target_met = denominator_valid and coverage_ledger_complete and achieved + TOLERANCE >= target_coverage_rate
    infeasible_reason_codes: list[str] = []
    if unsafe_roi_cell_present:
        infeasible_reason_codes.append("unsafe_roi_cells")
    if unreachable_roi_cell_present:
        infeasible_reason_codes.append("unreachable_roi_cells")
    if path_budget_exhausted:
        infeasible_reason_codes.append("insufficient_budget")
    if not denominator_valid:
        infeasible_reason_codes.append("coverage_denominator_invalid")
    if not coverage_ledger_complete:
        infeasible_reason_codes.append("coverage_ledger_incomplete")
    if denominator_valid and not target_met:
        infeasible_reason_codes.append("coverage_target_not_met")

    failure_reasons: list[str] = []
    if not denominator_valid:
        failure_reasons.append("coverage_denominator_invalid")
    if not coverage_ledger_complete:
        failure_reasons.append("coverage_ledger_incomplete")
    if denominator_valid and not target_met:
        failure_reasons.append("coverage_target_not_met")
        if path_budget_exhausted:
            failure_reasons.append("insufficient_budget")

    return {
        "scenario_id": scenario_id,
        "status": "passed" if not failure_reasons else "failed",
        "reason_codes": unique_sorted(failure_reasons),
        "target_coverage_rate": target_coverage_rate,
        "achieved_coverage_rate": achieved,
        "coverage_target_met": target_met,
        "coverage_denominator_valid": denominator_valid,
        "coverage_ledger_complete": coverage_ledger_complete,
        "path_budget_exhausted": path_budget_exhausted,
        "grid": grid,
        "roi_cell_count": roi_cell_count,
        "blocked_roi_cell_count": blocked_roi_cell_count,
        "unsafe_roi_cell_count": unsafe_roi_cell_count,
        "excluded_roi_cell_count": excluded_roi_cell_count,
        "unreachable_roi_cell_count": unreachable_roi_cell_count,
        "reachable_safe_cell_count": reachable_safe_count,
        "covered_reachable_safe_cell_count": covered_count,
        "infeasible_reason_codes": unique_sorted(infeasible_reason_codes),
        "ledger_rows": ledger_rows,
    }


def coverage_footprint(center: Cell, width: int, height: int, radius_cells: int) -> set[Cell]:
    x, y = center
    radius = max(0, radius_cells)
    return {
        (cell_x, cell_y)
        for cell_y in range(max(0, y - radius), min(height, y + radius + 1))
        for cell_x in range(max(0, x - radius), min(width, x + radius + 1))
    }


def coverage_event_cells(event: dict[str, Any], width: int, height: int) -> set[Cell]:
    cells: set[Cell] = set()
    for rect in event.get("covered_rectangles", []) or []:
        cells |= rectangle_to_cells(rect, width, height)
    for cell in event.get("covered_cells", []) or []:
        parsed = cell_from_config(cell, "covered_cells[]")
        if cell_in_bounds(parsed, width, height):
            cells.add(parsed)
    return cells


def roi_cells_from_config(roi: dict[str, Any], width: int, height: int) -> set[Cell]:
    kind = roi.get("kind")
    if kind == "rect":
        return rectangle_to_cells(roi, width, height)
    if kind == "cells":
        cells = roi.get("cells")
        if not isinstance(cells, list):
            raise ConfigError("roi.cells must be a list for cells ROI")
        result: set[Cell] = set()
        for cell in cells:
            parsed = cell_from_config(cell, "roi.cells[]")
            if cell_in_bounds(parsed, width, height):
                result.add(parsed)
        return result
    raise ConfigError("roi.kind must be 'rect' or 'cells'")


def rectangles_to_cells(rectangles: list[Any], width: int, height: int) -> set[Cell]:
    cells: set[Cell] = set()
    for rect in rectangles:
        cells |= rectangle_to_cells(rect, width, height)
    return cells


def rectangle_to_cells(rect: Any, width: int, height: int) -> set[Cell]:
    if not isinstance(rect, dict):
        raise ConfigError("rectangle entries must be objects")
    x0 = int_value(rect.get("x0"), "rectangle.x0")
    y0 = int_value(rect.get("y0"), "rectangle.y0")
    x1 = int_value(rect.get("x1"), "rectangle.x1")
    y1 = int_value(rect.get("y1"), "rectangle.y1")
    if x1 < x0 or y1 < y0:
        raise ConfigError("rectangle x1/y1 must be >= x0/y0")
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    return {(x, y) for y in range(y0, y1) for x in range(x0, x1)}


def reachable_cells_from(start: Cell, width: int, height: int, excluded_cells: set[Cell]) -> set[Cell]:
    if not cell_in_bounds(start, width, height) or start in excluded_cells:
        return set()
    visited = {start}
    queue: deque[Cell] = deque([start])
    while queue:
        cell = queue.popleft()
        for neighbor in neighbors4(cell):
            if neighbor in visited or neighbor in excluded_cells or not cell_in_bounds(neighbor, width, height):
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return visited


def bfs_tree(start: Cell, allowed_cells: set[Cell]) -> tuple[dict[Cell, int], dict[Cell, Cell | None]]:
    if start not in allowed_cells:
        return {}, {}
    distance = {start: 0}
    parent: dict[Cell, Cell | None] = {start: None}
    queue: deque[Cell] = deque([start])
    while queue:
        cell = queue.popleft()
        for neighbor in neighbors4(cell):
            if neighbor in distance or neighbor not in allowed_cells:
                continue
            distance[neighbor] = distance[cell] + 1
            parent[neighbor] = cell
            queue.append(neighbor)
    return distance, parent


def reconstruct_path(parent: dict[Cell, Cell | None], target: Cell) -> list[Cell]:
    if target not in parent:
        return []
    path = [target]
    current = target
    while parent[current] is not None:
        current = parent[current]  # type: ignore[assignment]
        path.append(current)
    path.reverse()
    return path


def connected_components(cells: set[Cell]) -> list[set[Cell]]:
    remaining = set(cells)
    components: list[set[Cell]] = []
    while remaining:
        seed = min(remaining, key=lambda cell: (cell[1], cell[0]))
        component = {seed}
        remaining.remove(seed)
        queue: deque[Cell] = deque([seed])
        while queue:
            cell = queue.popleft()
            for neighbor in neighbors4(cell):
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                component.add(neighbor)
                queue.append(neighbor)
        components.append(component)
    components.sort(key=lambda component: min((cell[1], cell[0]) for cell in component))
    return components


def neighbors4(cell: Cell) -> tuple[Cell, Cell, Cell, Cell]:
    x, y = cell
    return ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))


def cell_from_config(value: Any, label: str) -> Cell:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ConfigError(f"{label} must be a two-item cell")
    return (int_value(value[0], f"{label}[0]"), int_value(value[1], f"{label}[1]"))


def cell_in_bounds(cell: Cell, width: int, height: int) -> bool:
    x, y = cell
    return 0 <= x < width and 0 <= y < height


def int_value(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    return value


def float_value(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{label} must be numeric")
    return float(value)


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def unique_sorted(values: Any) -> list[str]:
    return sorted({str(value) for value in values if value})


def resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _validate_scenario_config(scenario: Any, index: int) -> dict[str, Any]:
    if not isinstance(scenario, dict):
        raise ConfigError(f"scenarios[{index}] must be an object")
    if not nonempty_string(scenario.get("scenario_id")):
        raise ConfigError(f"scenarios[{index}].scenario_id must be a non-empty string")
    grid = scenario.get("grid")
    if not isinstance(grid, dict):
        raise ConfigError(f"scenarios[{index}].grid must be an object")
    width = int_value(grid.get("width"), f"scenarios[{index}].grid.width")
    height = int_value(grid.get("height"), f"scenarios[{index}].grid.height")
    if width <= 0 or height <= 0:
        raise ConfigError(f"scenarios[{index}].grid width/height must be positive")
    if "resolution_m" in grid and float_value(grid["resolution_m"], f"scenarios[{index}].grid.resolution_m") <= 0:
        raise ConfigError(f"scenarios[{index}].grid.resolution_m must be positive")
    cell_from_config(scenario.get("start_cell"), f"scenarios[{index}].start_cell")
    if not isinstance(scenario.get("roi"), dict):
        raise ConfigError(f"scenarios[{index}].roi must be an object")
    for key in ("blocked_rectangles", "unsafe_rectangles", "coverage_events"):
        if key in scenario and not isinstance(scenario[key], list):
            raise ConfigError(f"scenarios[{index}].{key} must be a list")
    return scenario
