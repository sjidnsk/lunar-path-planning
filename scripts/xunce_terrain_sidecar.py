"""Shared terrain-sidecar construction for retained Stage23 ingestion."""

from __future__ import annotations

import hashlib
import json
from math import atan, degrees, isfinite, sqrt
from typing import Any


DEFAULT_MAX_TRAVERSABLE_SLOPE_DEG = 30.0


def sidecar_from_roi(
    dem_values: list[list[float]] | tuple[tuple[float, ...], ...],
    count_values: list[list[float]] | tuple[tuple[float, ...], ...],
    *,
    contract: dict[str, Any], scenario_id: str, data_manifest: dict[str, Any], roi: Any,
    resolution: float, max_traversable_slope_deg: float = DEFAULT_MAX_TRAVERSABLE_SLOPE_DEG,
    slope_deg_values: list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
    slope_source: str = "derived_from_dem", platform_contract_lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dem = [[_finite(value) for value in row] for row in dem_values]
    counts = [[_finite(value) for value in row] for row in count_values]
    height, width = len(dem), len(dem[0])
    max_count = max((value for row in counts for value in row), default=1.0) or 1.0
    slope = _slope_grid(dem)
    if slope_deg_values is None:
        slope_deg, slope_source_value = _slope_deg_grid(dem, resolution_m=float(resolution)), "derived_from_dem"
    else:
        slope_deg = _coerce_slope_deg_grid(slope_deg_values, width=width, height=height)
        slope_source_value = str(slope_source or "provided_slope_map")
    max_slope = max((value for row in slope for value in row), default=1.0) or 1.0
    risk_grid: list[list[float]] = []
    confidence_grid: list[list[float]] = []
    cost_grid: list[list[float]] = []
    passable_mask: list[list[bool]] = []
    reachable_goal_cells = {tuple(goal["cell"]) for goal in contract.get("top_goals", []) if goal.get("reachable") is True}
    for y in range(height):
        risk_row: list[float] = []
        confidence_row: list[float] = []
        cost_row: list[float] = []
        mask_row: list[bool] = []
        for x in range(width):
            confidence = min(max(counts[y][x] / max_count, 0.0), 1.0)
            slope_value = min(max(slope[y][x] / max_slope, 0.0), 1.0)
            risk = min(max(0.65 * slope_value + 0.35 * (1.0 - confidence), 0.0), 1.0)
            passable = (confidence > 0.0 and risk < 0.92) or (x, y) in reachable_goal_cells or (x, y) == (0, 0)
            risk_row.append(risk)
            confidence_row.append(confidence)
            cost_row.append(float(1.0 + risk + 0.25 * slope_value))
            mask_row.append(bool(passable))
        risk_grid.append(risk_row)
        confidence_grid.append(confidence_row)
        cost_grid.append(cost_row)
        passable_mask.append(mask_row)
    passable_count = sum(1 for row in passable_mask for value in row if value)
    total_count = max(width * height, 1)
    blocked_cells = _blocked_cells_from_passable_mask(passable_mask)
    slope_blocked_cells = _slope_blocked_cells_from_grid(slope_deg, max_traversable_slope_deg=max_traversable_slope_deg)
    platform = dict(platform_contract_lineage or {})
    platform_id = str(platform.get("platform_contract_id") or "agilex_scout_mini_piper")
    return {
        "schema_version": "path-planner-sidecar/v1", "cost": cost_grid, "passable_mask": passable_mask,
        "blocked_cells": blocked_cells, "blocked_source_kind": "passable_mask_false", "slope_source": slope_source_value,
        "slope_blocked_cells": slope_blocked_cells, "slope_blocked_source_kind": "slope_gt_max_traversable_deg",
        "slope_blocked_obstacle_source_kind": "slope_blocked_as_obstacle_proxy",
        "platform_contract_id": platform.get("platform_contract_id"), "platform_contract_hash": platform.get("platform_contract_hash"),
        "platform_max_climb_deg": platform.get("platform_max_climb_deg"), "max_traversable_slope_deg": float(max_traversable_slope_deg),
        "slope_blocked_cell_count": len(slope_blocked_cells),
        "terrain_layers": {"risk": risk_grid, "confidence": confidence_grid, "dem": dem, "slope_deg": slope_deg, "observation_count": counts},
        "metadata": {
            "source": "quasi-real-lola-map-bridge", "scenario_id": scenario_id,
            "map_source": {"kind": "lola_quasi_real_roi", "dataset_id": str(data_manifest.get("dataset_id", "unknown")),
                           "region": str(data_manifest.get("region", "unknown")), "roi_name": roi.name, "split": roi.split,
                           "roi": roi.bounds, "resolution_m": float(resolution)},
            "platform": platform_id, "platform_contract_id": platform.get("platform_contract_id"),
            "platform_contract_hash": platform.get("platform_contract_hash"), "platform_max_climb_deg": platform.get("platform_max_climb_deg"),
            "blocked_count": total_count - passable_count, "passable_ratio": passable_count / total_count,
            "slope_source": slope_source_value, "slope_blocked_cell_count": len(slope_blocked_cells),
            "max_traversable_slope_deg": float(max_traversable_slope_deg),
        },
    }


def slice_context_id(*, scenario_id: str, scenario_group: str, scenario_seed: int, scenario_variant_id: str,
                     top_k: int, goal_cell: Any, start_cell: tuple[int, int]) -> str | None:
    if not isinstance(goal_cell, list) or len(goal_cell) != 2:
        return None
    fields = {"scenario_id": scenario_id, "scenario_group": scenario_group, "scenario_seed": int(scenario_seed),
              "scenario_variant_id": scenario_variant_id, "diagnostic_profile": "execution", "planning_backend": "path_planner_route",
              "top_k": int(top_k), "start_cell": [int(start_cell[0]), int(start_cell[1])], "sample_type": "quasi_real_map_slice",
              "candidate_role": "quasi_real_map_slice", "source_action_index": 0,
              "policy_target_cell": [int(goal_cell[0]), int(goal_cell[1])], "execution_goal_cell": [int(goal_cell[0]), int(goal_cell[1])],
              "target_binding_mode": "source_selected_policy_target"}
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def first_reachable_goal(contract: dict[str, Any]) -> dict[str, Any] | None:
    for goal in contract.get("top_goals", []):
        if isinstance(goal, dict) and goal.get("reachable") is True:
            return goal
    goals = contract.get("top_goals", [])
    return goals[0] if goals and isinstance(goals[0], dict) else None


def _coerce_slope_deg_grid(values: list[list[float]] | tuple[tuple[float, ...], ...], *, width: int, height: int) -> list[list[float]]:
    rows = [[_finite(value) for value in row] for row in values]
    if len(rows) != height or any(len(row) != width for row in rows):
        raise ValueError("slope_deg_values shape must match dem_values")
    return rows


def _blocked_cells_from_passable_mask(passable_mask: list[list[bool]]) -> list[list[int]]:
    return [[int(x), int(y)] for y, row in enumerate(passable_mask) for x, value in enumerate(row) if value is False]


def _slope_blocked_cells_from_grid(slope_deg: list[list[float]], *, max_traversable_slope_deg: float) -> list[list[int]]:
    return [[int(x), int(y)] for y, row in enumerate(slope_deg) for x, value in enumerate(row) if value > max_traversable_slope_deg]


def _slope_deg_grid(values: list[list[float]], *, resolution_m: float) -> list[list[float]]:
    resolution = float(resolution_m) if isfinite(float(resolution_m)) and float(resolution_m) > 0 else 1.0
    rows: list[list[float]] = []
    for y, row in enumerate(values):
        out: list[float] = []
        for x, value in enumerate(row):
            max_angle = 0.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < len(values) and 0 <= xx < len(values[yy]):
                        distance = resolution * sqrt(float(dx * dx + dy * dy))
                        if distance > 0:
                            max_angle = max(max_angle, degrees(atan(abs(float(value) - float(values[yy][xx])) / distance)))
            out.append(float(max_angle))
        rows.append(out)
    return rows


def _slope_grid(values: list[list[float]]) -> list[list[float]]:
    rows: list[list[float]] = []
    for y, row in enumerate(values):
        out: list[float] = []
        for x, value in enumerate(row):
            neighbors = [abs(value - values[yy][xx]) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                         if not (dx == 0 and dy == 0) for yy, xx in [(y + dy, x + dx)]
                         if 0 <= yy < len(values) and 0 <= xx < len(values[yy])]
            out.append(max(neighbors) if neighbors else 0.0)
        rows.append(out)
    return rows


def _finite(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if isfinite(numeric) else 0.0
