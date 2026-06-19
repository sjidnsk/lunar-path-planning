from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from global_99_coverage_contract import resolve_path
    from xunce_frontier_nbv_validation import BATCH_ASTAR_VALIDATION_MODE, validate_candidate_cells
except ModuleNotFoundError:  # pragma: no cover
    from scripts.global_99_coverage_contract import resolve_path
    from scripts.xunce_frontier_nbv_validation import BATCH_ASTAR_VALIDATION_MODE, validate_candidate_cells


GENERATION_SOURCE = "dynamic_frontier_nbv_in_process/v1"
COVERAGE_SOURCE = "geometric_counterfactual_from_dynamic_frontier_nbv/v1"
COVERAGE_CELL_SET_KIND = "path_line_plus_endpoint_union"
COVERAGE_DEDUPE_SCOPE = "scenario_step_new_cells"
DIRECTIONS_8 = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
SOURCE_PRIORITY = {"frontier_boundary": 0, "roi_undercovered_boundary": 1, "incumbent_neighborhood": 2}
TOLERANCE = 1.0e-12
ROI_WEIGHTS = {
    "smooth_high_confidence": 1.0,
    "mixed_risk": 1.2,
    "rim_or_steep_slope": 1.4,
    "low_sun_roughness": 1.4,
    "shadowed_transition": 1.5,
    "crater_rim_fragmented": 1.5,
    "mixed_passability_edge": 1.5,
    "low_observation_count": 1.6,
}


def build_dynamic_frontier_nbv_candidates(
    *,
    scenario: dict[str, Any],
    slice_row: dict[str, Any],
    current_cell: tuple[int, int],
    covered_cells: set[tuple[int, int]],
    step_index: int,
    config: dict[str, Any],
    repo_root: Path,
    output_work_root: Path,
    validation_cache: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build and validate one rollout step's state-conditioned frontier/NBV candidates."""

    proposal_rows = _proposal_rows(
        scenario=scenario,
        slice_row=slice_row,
        current_cell=current_cell,
        covered_cells=covered_cells,
        step_index=step_index,
        config=config,
    )
    if not proposal_rows:
        return [], [], []

    contract_path = _resolve_existing_path(slice_row.get("contract"), repo_root)
    sidecar_path = _resolve_existing_path(slice_row.get("sidecar"), repo_root)
    if contract_path is None or sidecar_path is None:
        validation_rows = [
            _validation_failure(
                proposal,
                "dynamic_contract_or_sidecar_missing",
                contract_missing=contract_path is None,
                sidecar_missing=sidecar_path is None,
            )
            for proposal in proposal_rows
        ]
        return [], proposal_rows, validation_rows

    cache_key = _validation_cache_key(
        scenario_id=str(scenario.get("scenario_id", "scenario")),
        current_cell=current_cell,
        proposals=proposal_rows,
        contract_path=contract_path,
        sidecar_path=sidecar_path,
        config=config,
    )
    cache_enabled = bool(config.get("dynamic_validation_cache_enabled", True))
    cache_hit = cache_enabled and cache_key in validation_cache
    if cache_hit:
        validation_rows = [dict(row, dynamic_validation_cache_hit=True) for row in validation_cache[cache_key]]
    else:
        validation_batch_hash = cache_key[:16]
        validation_work_root = output_work_root / validation_batch_hash
        route_cache = validation_cache.setdefault("__route_cache__", {})
        if not isinstance(route_cache, dict):
            route_cache = {}
            validation_cache["__route_cache__"] = route_cache
        validation_rows = validate_candidate_cells(
            scenario=scenario,
            proposal_rows=proposal_rows,
            contract_path=contract_path,
            sidecar_path=sidecar_path,
            current_cell=current_cell,
            repo_root=repo_root,
            output_work_root=validation_work_root,
            validation_mode=str(config.get("dynamic_candidate_validation_mode", BATCH_ASTAR_VALIDATION_MODE)),
            top_k=len(proposal_rows),
            allow_open_grid_fallback=bool(config.get("allow_open_grid_fallback", False)),
            debug_validation_artifacts=bool(config.get("debug_validation_artifacts", False)),
            max_validation_path_length=int(config.get("dynamic_validation_max_path_length", 180)),
            sidecar_fallback_mode=str(config.get("dynamic_sidecar_fallback_mode", "diagnostic_only")),
            validation_batch_hash=validation_batch_hash,
            route_cache=route_cache,
        )
        validation_rows = [dict(row, dynamic_validation_cache_hit=False) for row in validation_rows]
        if cache_enabled:
            validation_cache[cache_key] = [dict(row, dynamic_validation_cache_hit=False) for row in validation_rows]

    enriched = [
        _enrich_validated_candidate(
            row,
            current_cell=current_cell,
            covered_cells=covered_cells,
            config=config,
            step_index=step_index,
            roi_group=_roi_group(scenario, slice_row),
        )
        for row in validation_rows
    ]
    formal = [row for row in enriched if _is_formal_candidate(row, allow_open_grid_fallback=bool(config.get("allow_open_grid_fallback", False)))]
    selected = _select_formal_candidates(formal, int(config.get("dynamic_max_candidates_per_step", 6)))
    for action_index, candidate in enumerate(selected):
        candidate["action_index"] = action_index
    return selected, proposal_rows, enriched


def covered_cells_hash(covered_cells: set[tuple[int, int]]) -> str:
    return _hash_payload(sorted([list(cell) for cell in covered_cells]))


def candidate_set_hash(candidates: list[dict[str, Any]]) -> str:
    payload = [
        {
            "cell": _cell_tuple(candidate.get("cell")),
            "path_cost": _finite_float(candidate.get("path_cost")),
            "risk": _finite_float(candidate.get("risk")),
            "reachable": candidate.get("reachable"),
            "source": candidate.get("frontier_candidate_source"),
        }
        for candidate in candidates
    ]
    return _hash_payload(payload)


def _proposal_rows(
    *,
    scenario: dict[str, Any],
    slice_row: dict[str, Any],
    current_cell: tuple[int, int],
    covered_cells: set[tuple[int, int]],
    step_index: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    bounds = _grid_bounds(scenario, slice_row, current_cell)
    roi_group = resolve_roi_group(scenario, slice_row)
    rows: list[dict[str, Any]] = []
    proposal_index = 0

    for cell in _frontier_cells(current_cell, covered_cells, config, bounds):
        rows.append(_proposal("frontier_boundary", cell, current_cell, covered_cells, step_index, proposal_index, config, slice_row, roi_group))
        proposal_index += 1

    for cell in _roi_undercovered_cells(current_cell, covered_cells, config, bounds):
        rows.append(_proposal("roi_undercovered_boundary", cell, current_cell, covered_cells, step_index, proposal_index, config, slice_row, roi_group))
        proposal_index += 1

    for cell in _incumbent_neighborhood_cells(current_cell, covered_cells, bounds):
        rows.append(_proposal("incumbent_neighborhood", cell, current_cell, covered_cells, step_index, proposal_index, config, slice_row, roi_group))
        proposal_index += 1

    rows = _dedupe_rows(rows)
    return _limit_rows_by_family(rows, int(config.get("dynamic_proposal_pool_limit_per_step", 24)))


def _frontier_cells(
    current_cell: tuple[int, int],
    covered_cells: set[tuple[int, int]],
    config: dict[str, Any],
    bounds: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    radii = config.get("dynamic_frontier_radius_cells", [2, 4, 6])
    if not isinstance(radii, list) or not radii:
        radii = [2, 4, 6]
    direction_count = max(1, min(int(config.get("dynamic_frontier_direction_count", 8)), len(DIRECTIONS_8)))
    rows: list[tuple[int, int]] = []
    anchors = [current_cell]
    if covered_cells:
        xs = [cell[0] for cell in covered_cells]
        ys = [cell[1] for cell in covered_cells]
        anchors.extend(
            [
                (min(xs), current_cell[1]),
                (max(xs), current_cell[1]),
                (current_cell[0], min(ys)),
                (current_cell[0], max(ys)),
            ]
        )
    for anchor in anchors:
        for radius in radii:
            try:
                radius_int = max(1, int(radius))
            except (TypeError, ValueError):
                continue
            for dx, dy in DIRECTIONS_8[:direction_count]:
                cell = (anchor[0] + dx * radius_int, anchor[1] + dy * radius_int)
                if _in_bounds(cell, bounds) and cell not in covered_cells:
                    rows.append(cell)
    return rows


def _roi_undercovered_cells(
    current_cell: tuple[int, int],
    covered_cells: set[tuple[int, int]],
    config: dict[str, Any],
    bounds: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    min_x, min_y, max_x, max_y = bounds
    mid_x = int(round((min_x + max_x) / 2))
    mid_y = int(round((min_y + max_y) / 2))
    candidates = [
        (min_x, min_y),
        (min_x, max_y),
        (max_x, min_y),
        (max_x, max_y),
        (mid_x, min_y),
        (mid_x, max_y),
        (min_x, mid_y),
        (max_x, mid_y),
    ]
    radius = int(config.get("coverage_radius_cells", 1))
    return [
        cell
        for cell in sorted(candidates, key=lambda item: (_manhattan(current_cell, item), item[0], item[1]))
        if cell not in covered_cells and len(_coverage_cells(current_cell, cell, radius, "path_line_plus_endpoint") - covered_cells) > 0
    ]


def _incumbent_neighborhood_cells(
    current_cell: tuple[int, int],
    covered_cells: set[tuple[int, int]],
    bounds: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    rows = []
    for dx, dy in DIRECTIONS_8[:4]:
        cell = (current_cell[0] + dx, current_cell[1] + dy)
        if _in_bounds(cell, bounds) and cell not in covered_cells:
            rows.append(cell)
    return rows


def _proposal(
    source: str,
    cell: tuple[int, int],
    current_cell: tuple[int, int],
    covered_cells: set[tuple[int, int]],
    step_index: int,
    proposal_index: int,
    config: dict[str, Any],
    slice_row: dict[str, Any],
    roi_group: str,
) -> dict[str, Any]:
    metrics = _coverage_metrics(current_cell, cell, covered_cells, config, roi_group=roi_group)
    scenario_id = str(slice_row.get("scenario_id") or "scenario")
    return {
        "schema_version": "xunce-dynamic-frontier-nbv-proposal/v1",
        "scenario_id": scenario_id,
        "step_index": int(step_index),
        "proposal_id": f"{scenario_id}-step-{step_index:03d}-{source}-{proposal_index:03d}-{cell[0]}-{cell[1]}",
        "cell": [int(cell[0]), int(cell[1])],
        "frontier_candidate_source": source,
        "roi_group": roi_group,
        "candidate_generation_source": GENERATION_SOURCE,
        "proposal_only": True,
        "proposal_validated_by_path_feedback": False,
        "coverage_source": COVERAGE_SOURCE,
        "coverage_cell_set_kind": COVERAGE_CELL_SET_KIND,
        "coverage_dedupe_scope": COVERAGE_DEDUPE_SCOPE,
        "coverage_validated_by_path_feedback": False,
        "coverage_validation_source": "offline_geometric_counterfactual_not_path_feedback",
        **metrics,
    }


def _coverage_metrics(
    current_cell: tuple[int, int],
    cell: tuple[int, int],
    covered_cells: set[tuple[int, int]],
    config: dict[str, Any],
    *,
    roi_group: str,
) -> dict[str, Any]:
    radius = int(config.get("coverage_radius_cells", 1))
    mode = str(config.get("coverage_metric_mode", "path_line_plus_endpoint"))
    denominator = float(config.get("coverage_denominator_cells", 1000))
    endpoint_cells = set(_footprint(cell, radius))
    path_cells = _coverage_cells(current_cell, cell, radius, mode)
    union_cells = endpoint_cells | path_cells
    new_cells = union_cells - covered_cells
    overlap = union_cells & covered_cells
    new_count = len(new_cells)
    roi_weight = _roi_weight(roi_group)
    return {
        "endpoint_new_cell_count": float(len(endpoint_cells - covered_cells)),
        "path_line_new_cell_count": float(len(path_cells - covered_cells)),
        "total_new_cell_count": float(new_count),
        "expected_new_coverage_cell_count": float(new_count),
        "expected_new_coverage_area": float(new_count),
        "endpoint_coverage_delta": float(len(endpoint_cells - covered_cells)) / denominator,
        "path_line_coverage_delta": float(len(path_cells - covered_cells)) / denominator,
        "expected_coverage_rate_delta": float(new_count) / denominator,
        "roi_weighted_coverage_delta": float(new_count) * roi_weight / denominator,
        "coverage_overlap_count": float(len(overlap)),
        "coverage_overlap_ratio": float(len(overlap)) / max(float(len(union_cells)), 1.0),
        "revisit_penalty": float(len(overlap)) / max(float(len(union_cells)), 1.0),
        "information_gain": float(new_count) / denominator,
        "value": float(new_count) * roi_weight,
    }


def _enrich_validated_candidate(
    row: dict[str, Any],
    *,
    current_cell: tuple[int, int],
    covered_cells: set[tuple[int, int]],
    config: dict[str, Any],
    step_index: int,
    roi_group: str,
) -> dict[str, Any]:
    cell = _cell_tuple(row.get("cell"))
    enriched = dict(row)
    enriched["step_index"] = int(step_index)
    enriched["candidate_generation_source"] = GENERATION_SOURCE
    enriched["coverage_source"] = COVERAGE_SOURCE
    enriched["coverage_validated_by_path_feedback"] = False
    enriched["coverage_validation_source"] = "offline_geometric_counterfactual_not_path_feedback"
    enriched["coverage_cell_set_kind"] = COVERAGE_CELL_SET_KIND
    enriched["coverage_dedupe_scope"] = COVERAGE_DEDUPE_SCOPE
    enriched.setdefault("roi_group", roi_group)
    if cell is not None:
        enriched.update(_coverage_metrics(current_cell, cell, covered_cells, config, roi_group=roi_group))
    return enriched


def _is_formal_candidate(row: dict[str, Any], *, allow_open_grid_fallback: bool) -> bool:
    if row.get("proposal_validated_by_path_feedback") is not True:
        return False
    if row.get("proposal_only") is True:
        return False
    if row.get("reachable") is not True:
        return False
    if row.get("open_grid_fallback_used") is True and not allow_open_grid_fallback:
        return False
    if _cell_tuple(row.get("cell")) is None:
        return False
    for key in ("path_cost", "risk", "path_length"):
        if _finite_float(row.get(key)) is None:
            return False
    return True


def _select_formal_candidates(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row.get("expected_new_coverage_cell_count") or 0.0),
            float(row.get("path_cost") or 1.0e12),
            float(row.get("risk") or 1.0e12),
            SOURCE_PRIORITY.get(str(row.get("frontier_candidate_source")), 99),
            _cell_tuple(row.get("cell")) or (1_000_000, 1_000_000),
        ),
    )
    return [dict(row) for row in ordered[: max(0, limit)]]


def _limit_rows_by_family(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    selected: list[dict[str, Any]] = []
    for source in ("frontier_boundary", "roi_undercovered_boundary", "incumbent_neighborhood"):
        for row in rows:
            if row.get("frontier_candidate_source") == source and row not in selected:
                selected.append(row)
                break
    for row in rows:
        if len(selected) >= limit:
            break
        if row not in selected:
            selected.append(row)
    return selected[:limit]


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        cell = _cell_tuple(row.get("cell"))
        if cell is None or cell in seen:
            continue
        seen.add(cell)
        deduped.append(row)
    return deduped


def _grid_bounds(scenario: dict[str, Any], slice_row: dict[str, Any], current_cell: tuple[int, int]) -> tuple[int, int, int, int]:
    map_source = slice_row.get("map_source") if isinstance(slice_row.get("map_source"), dict) else {}
    roi = map_source.get("roi") if isinstance(map_source.get("roi"), dict) else {}
    width = _positive_int(roi.get("width")) or _positive_int(slice_row.get("width")) or _candidate_extent(scenario, axis=0) or 32
    height = _positive_int(roi.get("height")) or _positive_int(slice_row.get("height")) or _candidate_extent(scenario, axis=1) or 32
    max_x = max(int(width) - 1, current_cell[0] + 8)
    max_y = max(int(height) - 1, current_cell[1] + 8)
    return (0, 0, max_x, max_y)


def _candidate_extent(scenario: dict[str, Any], *, axis: int) -> int | None:
    candidates = []
    path_feedback = scenario.get("path_feedback")
    if isinstance(path_feedback, dict) and isinstance(path_feedback.get("candidates"), list):
        candidates = path_feedback["candidates"]
    values = []
    for candidate in candidates:
        if isinstance(candidate, dict):
            cell = _cell_tuple(candidate.get("cell"))
            if cell is not None:
                values.append(cell[axis])
    return max(values) + 8 if values else None


def _validation_failure(
    proposal: dict[str, Any],
    reason: str,
    *,
    contract_missing: bool,
    sidecar_missing: bool,
) -> dict[str, Any]:
    row = dict(proposal)
    row.update(
        {
            "proposal_only": True,
            "proposal_validated_by_path_feedback": False,
            "path_feedback_validation_source": reason,
            "failure_reason": reason,
            "planner_reachable": False,
            "reachable": False,
            "open_grid_fallback_used": False,
            "validation_diagnostic_flags": [reason],
            "dynamic_contract_missing": bool(contract_missing),
            "dynamic_sidecar_missing": bool(sidecar_missing),
        }
    )
    return row


def _resolve_existing_path(value: Any, repo_root: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = resolve_path(Path(value), repo_root)
    return path if path.is_file() else None


def _validation_cache_key(
    *,
    scenario_id: str,
    current_cell: tuple[int, int],
    proposals: list[dict[str, Any]],
    contract_path: Path,
    sidecar_path: Path,
    config: dict[str, Any],
) -> str:
    payload = {
        "scenario_id": scenario_id,
        "current_cell": list(current_cell),
        "proposal_cells": [_cell_tuple(row.get("cell")) for row in proposals],
        "contract_path": str(contract_path),
        "sidecar_path": str(sidecar_path),
        "validation_mode": str(config.get("dynamic_candidate_validation_mode", BATCH_ASTAR_VALIDATION_MODE)),
        "top_k": len(proposals),
        "allow_open_grid_fallback": bool(config.get("allow_open_grid_fallback", False)),
        "dynamic_validation_max_path_length": int(config.get("dynamic_validation_max_path_length", 180)),
        "dynamic_sidecar_fallback_mode": str(config.get("dynamic_sidecar_fallback_mode", "diagnostic_only")),
    }
    return _hash_payload(payload)


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _coverage_cells(start: tuple[int, int], end: tuple[int, int], radius: int, mode: str) -> set[tuple[int, int]]:
    cells = set(_footprint(end, radius))
    if mode == "path_line_plus_endpoint":
        for cell in _line_cells(start, end):
            cells.update(_footprint(cell, radius))
    return cells


def _line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    cells: list[tuple[int, int]] = []
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


def _footprint(cell: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    x, y = cell
    return {(x + dx, y + dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)}


def _in_bounds(cell: tuple[int, int], bounds: tuple[int, int, int, int]) -> bool:
    min_x, min_y, max_x, max_y = bounds
    return min_x <= cell[0] <= max_x and min_y <= cell[1] <= max_y


def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _cell_tuple(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return (int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return None
    return None


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def resolve_roi_group(scenario: dict[str, Any], slice_row: dict[str, Any]) -> str:
    for source, key in (
        (scenario, "roi_group"),
        (slice_row, "roi_group"),
        (slice_row, "roi_name"),
        (scenario, "scenario_group"),
    ):
        value = source.get(key) if isinstance(source, dict) else None
        if isinstance(value, str) and value:
            return value
    metadata = slice_row.get("metadata") if isinstance(slice_row.get("metadata"), dict) else {}
    value = metadata.get("roi_group")
    return str(value) if isinstance(value, str) and value else "unknown"


def _roi_group(scenario: dict[str, Any], slice_row: dict[str, Any]) -> str:
    return resolve_roi_group(scenario, slice_row)


def _roi_weight(roi_group: str) -> float:
    return float(ROI_WEIGHTS.get(roi_group, 1.0))
