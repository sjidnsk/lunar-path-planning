from __future__ import annotations

import math
from typing import Any


RISK_SEMANTICS_SOURCE = "path_cost_proxy_risk_semantics/v1"
DEFAULT_SOFT_HIGH_RISK_THRESHOLD = 8.0
DEFAULT_HARD_PATH_RISK_PEAK_LIMIT = 10.0

_NO_GO_REASONS = {
    "blocked",
    "goal_blocked",
    "impassable",
    "no_go",
    "path_blocked",
    "start_blocked",
}


def classify_path_risk(
    route: Any,
    sidecar_cost: Any = None,
    path_cells: Any = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(config or {})
    soft_threshold = _finite_float(config.get("soft_high_risk_threshold"))
    hard_peak_limit = _finite_float(config.get("hard_path_risk_peak_limit"))
    resolution_m = _finite_float(config.get("resolution_m"))
    if soft_threshold is None:
        soft_threshold = DEFAULT_SOFT_HIGH_RISK_THRESHOLD
    if hard_peak_limit is None:
        hard_peak_limit = DEFAULT_HARD_PATH_RISK_PEAK_LIMIT
    if resolution_m is None or resolution_m <= 0.0:
        resolution_m = 1.0

    hard_flags: list[str] = []
    soft_flags: list[str] = []
    route_flags = _route_flags(route)
    failure_reason = _text_or_none(_value(route, "failure_reason"))
    reachable_value = _value(route, "reachable")
    success_value = _value(route, "success")

    if success_value is False and reachable_value is not True:
        _append_unique(hard_flags, "planning_failed")
    if reachable_value is False or failure_reason == "unreachable":
        _append_unique(hard_flags, "unreachable")
    if _truthy(_value(route, "open_grid_fallback_used")) or _truthy(_value(route, "open_grid_fallback")):
        _append_unique(hard_flags, "open_grid_fallback")
    for flag in route_flags:
        _classify_hard_flag(flag, hard_flags)
    if failure_reason:
        _classify_hard_flag(failure_reason, hard_flags)

    cells = list(_iter_cells(path_cells if path_cells is not None else _value(route, "path_cells")))
    bounds = _bounds_from_sidecar(sidecar_cost)
    in_bounds_cells: list[tuple[int, int]] = []
    for cell in cells:
        if bounds is not None and not _cell_in_bounds(cell, bounds):
            _append_unique(hard_flags, "out_of_bounds")
            continue
        in_bounds_cells.append(cell)

    values = _path_cost_values(sidecar_cost, in_bounds_cells)
    if not values:
        values = _proxy_values_from_route(route)

    peak = _max_or_none(values)
    exposure = float(sum(values)) if values else None
    p95 = _percentile_nearest(values, 0.95)
    high_risk_distance = float(sum(1 for value in values if value >= soft_threshold) * resolution_m) if values else 0.0
    recovery_margin = float(hard_peak_limit - peak) if peak is not None else None

    if peak is not None and peak >= soft_threshold:
        _append_unique(soft_flags, "path_risk_peak_above_soft_threshold")
    if p95 is not None and p95 >= soft_threshold:
        _append_unique(soft_flags, "path_risk_p95_above_soft_threshold")
    if peak is not None and peak >= hard_peak_limit:
        _append_unique(hard_flags, "path_risk_peak_above_hard_limit")

    return {
        "path_allowed_by_risk": not hard_flags,
        "hard_risk_flags": hard_flags,
        "soft_risk_flags": soft_flags,
        "path_risk_peak": peak,
        "path_risk_p95": p95,
        "path_risk_exposure": exposure,
        "high_risk_distance_m": high_risk_distance,
        "recovery_margin_min": recovery_margin,
        "risk_semantics_source": RISK_SEMANTICS_SOURCE,
        "risk_proxy_is_physical_risk": False,
    }


def _classify_hard_flag(flag: str, hard_flags: list[str]) -> None:
    text = flag.strip().lower()
    if not text:
        return
    if text in {"planning_failed", "planner_failed", "invalid_candidate_cell"}:
        _append_unique(hard_flags, "planning_failed")
    if text in {"unreachable", "proposal_unreachable"}:
        _append_unique(hard_flags, "unreachable")
    if text in {"open_grid_fallback", "open_grid_fallback_used", "proposal_open_grid_fallback"}:
        _append_unique(hard_flags, "open_grid_fallback")
    if text in _NO_GO_REASONS or "no_go" in text or "blocked" in text or "impassable" in text:
        _append_unique(hard_flags, "no_go")
    if "slope" in text:
        _append_unique(hard_flags, "slope_capability")
    if "terrain_confidence_too_low" in text or "low_terrain_confidence" in text or "confidence_too_low" in text:
        _append_unique(hard_flags, "terrain_confidence_too_low")
    if "unrecoverable" in text:
        _append_unique(hard_flags, "unrecoverable_segment")
    if "out_of_bounds" in text:
        _append_unique(hard_flags, "out_of_bounds")


def _path_cost_values(sidecar_cost: Any, cells: list[tuple[int, int]]) -> list[float]:
    if sidecar_cost is None or not cells:
        return []
    values: list[float] = []
    for x, y in cells:
        try:
            raw = sidecar_cost[y][x]
        except (IndexError, KeyError, TypeError):
            continue
        value = _finite_float(raw)
        if value is not None:
            values.append(value)
    return values


def _proxy_values_from_route(route: Any) -> list[float]:
    values = []
    for key in ("path_cost_proxy_mean", "path_cost_proxy_peak", "path_cost_proxy_p95"):
        value = _finite_float(_value(route, key))
        if value is not None:
            values.append(value)
    risk = _finite_float(_value(route, "risk"))
    if risk is not None and not values:
        values.append(risk)
    return values


def _route_flags(route: Any) -> list[str]:
    flags: list[str] = []
    for key in ("hard_risk_flags", "validation_diagnostic_flags", "risk_proxy_reason_codes", "flags"):
        raw = _value(route, key)
        if isinstance(raw, str):
            flags.append(raw)
        elif isinstance(raw, (list, tuple, set)):
            flags.extend(str(item) for item in raw if item is not None)
    return flags


def _iter_cells(raw_cells: Any):
    for raw in raw_cells or ():
        cell = _cell_tuple(raw)
        if cell is not None:
            yield cell


def _cell_tuple(raw: Any) -> tuple[int, int] | None:
    x = _value(raw, "x")
    y = _value(raw, "y")
    if x is None or y is None:
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            return None
        x, y = raw[0], raw[1]
    try:
        return int(x), int(y)
    except (TypeError, ValueError):
        return None


def _bounds_from_sidecar(sidecar_cost: Any) -> tuple[int, int] | None:
    shape = getattr(sidecar_cost, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    try:
        height = len(sidecar_cost)
        width = len(sidecar_cost[0]) if height else 0
    except (TypeError, IndexError):
        return None
    return width, height


def _cell_in_bounds(cell: tuple[int, int], bounds: tuple[int, int]) -> bool:
    width, height = bounds
    x, y = cell
    return 0 <= x < width and 0 <= y < height


def _percentile_nearest(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(math.ceil(percentile * len(ordered))) - 1)
    return float(ordered[index])


def _max_or_none(values: list[float]) -> float | None:
    return float(max(values)) if values else None


def _value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _truthy(value: Any) -> bool:
    return bool(value) if not isinstance(value, str) else value.strip().lower() in {"1", "true", "yes"}


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)
