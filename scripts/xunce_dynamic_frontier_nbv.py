from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from global_99_coverage_contract import resolve_path
    from xunce_frontier_nbv_validation import BATCH_ASTAR_VALIDATION_MODE, validate_candidate_cells
except ModuleNotFoundError:  # pragma: no cover
    from scripts.global_99_coverage_contract import resolve_path
    from scripts.xunce_frontier_nbv_validation import BATCH_ASTAR_VALIDATION_MODE, validate_candidate_cells


GENERATION_SOURCE = "dynamic_frontier_nbv_in_process/v1"
ALGORITHM_SOURCE = "map_aware_coverage_frontier_nbv/v1"
COVERAGE_SOURCE = "geometric_counterfactual_from_dynamic_frontier_nbv/v1"
COVERAGE_CELL_SET_KIND = "path_line_plus_endpoint_union"
COVERAGE_DEDUPE_SCOPE = "scenario_step_new_cells"
DIRECTIONS_8 = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
SOURCE_COVERAGE_FRONTIER = "coverage_frontier_boundary"
SOURCE_UNDERCOVERED_CENTROID = "undercovered_component_centroid"
SOURCE_UNDERCOVERED_BOUNDARY = "undercovered_component_boundary"
SOURCE_LOW_COST_BRIDGE = "low_cost_bridge_candidate"
SOURCE_CONSERVATIVE_LOCAL = "conservative_local_candidate"
SOURCE_PRIORITY = {
    SOURCE_COVERAGE_FRONTIER: 0,
    SOURCE_UNDERCOVERED_CENTROID: 1,
    SOURCE_UNDERCOVERED_BOUNDARY: 2,
    SOURCE_LOW_COST_BRIDGE: 3,
    SOURCE_CONSERVATIVE_LOCAL: 4,
    # Legacy labels kept below so historical fixtures still sort deterministically.
    "frontier_boundary": 5,
    "roi_undercovered_boundary": 6,
    "incumbent_neighborhood": 7,
}
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


@dataclass
class ValidCellsContext:
    valid_cells: set[tuple[int, int]]
    proposal_cells: set[tuple[int, int]]
    passable_cells: set[tuple[int, int]] | None
    valid_cells_source: str
    valid_cells_count: int
    all_bounds_fallback_used: bool = False


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

    contract_path = _resolve_existing_path(slice_row.get("contract"), repo_root)
    sidecar_path = _resolve_existing_path(slice_row.get("sidecar"), repo_root)
    bounds = _grid_bounds(scenario, slice_row, current_cell)
    valid_context = _valid_cells_context(
        scenario=scenario,
        slice_row=slice_row,
        bounds=bounds,
        repo_root=repo_root,
        sidecar_path=sidecar_path,
    )
    if not valid_context.valid_cells:
        return [], [], [
            _prefilter_failure_row(
                scenario=scenario,
                slice_row=slice_row,
                current_cell=current_cell,
                step_index=step_index,
                reason="valid_cells_source_missing",
                valid_context=valid_context,
            )
        ]
    proposal_rows = _proposal_rows(
        scenario=scenario,
        slice_row=slice_row,
        current_cell=current_cell,
        covered_cells=covered_cells,
        step_index=step_index,
        config=config,
        repo_root=repo_root,
        sidecar_path=sidecar_path,
        bounds=bounds,
        valid_context=valid_context,
    )
    if not proposal_rows:
        return [], [], []
    validation_candidates = [row for row in proposal_rows if row.get("path_validation_attempted") is True]

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
    if not validation_candidates:
        enriched_failures = [
            _enrich_validated_candidate(
                row,
                current_cell=current_cell,
                covered_cells=covered_cells,
                config=config,
                scenario=scenario,
                slice_row=slice_row,
                valid_cells=valid_context.valid_cells,
                step_index=step_index,
                roi_group=_roi_group(scenario, slice_row),
            )
            for row in proposal_rows
        ]
        return [], proposal_rows, enriched_failures

    cache_key = _validation_cache_key(
        scenario_id=str(scenario.get("scenario_id", "scenario")),
        current_cell=current_cell,
        proposals=validation_candidates,
        contract_path=contract_path,
        sidecar_path=sidecar_path,
        config=config,
        valid_context=valid_context,
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
            proposal_rows=validation_candidates,
            contract_path=contract_path,
            sidecar_path=sidecar_path,
            current_cell=current_cell,
            repo_root=repo_root,
            output_work_root=validation_work_root,
            validation_mode=str(config.get("dynamic_candidate_validation_mode", BATCH_ASTAR_VALIDATION_MODE)),
            top_k=len(validation_candidates),
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
    validation_rows = _merge_validation_rows_with_prefilter_rows(proposal_rows, validation_rows)

    enriched = [
        _enrich_validated_candidate(
            row,
            current_cell=current_cell,
            covered_cells=covered_cells,
            config=config,
            scenario=scenario,
            slice_row=slice_row,
            valid_cells=valid_context.valid_cells,
            step_index=step_index,
            roi_group=_roi_group(scenario, slice_row),
        )
        for row in validation_rows
    ]
    formal = [row for row in enriched if _is_formal_candidate(row, allow_open_grid_fallback=bool(config.get("allow_open_grid_fallback", False)))]
    selected = _select_formal_candidates(formal, int(config.get("dynamic_max_candidates_per_step", 6)))
    for action_index, candidate in enumerate(selected):
        candidate["action_index"] = action_index
    _mark_selected_validation_rows(enriched, selected)
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
            "algorithm_source": candidate.get("candidate_generation_algorithm_source"),
            "expected_new_coverage_cell_count": _finite_float(candidate.get("expected_new_coverage_cell_count")),
            "roi_weighted_coverage_delta": _finite_float(candidate.get("roi_weighted_coverage_delta")),
            "proposal_validated_by_path_feedback": candidate.get("proposal_validated_by_path_feedback"),
            "proposal_only": candidate.get("proposal_only"),
            "open_grid_fallback_used": candidate.get("open_grid_fallback_used"),
            "validation_evidence_kind": candidate.get("validation_evidence_kind"),
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
    repo_root: Path,
    sidecar_path: Path | None,
    bounds: tuple[int, int, int, int] | None = None,
    valid_cells: set[tuple[int, int]] | None = None,
    valid_context: ValidCellsContext | None = None,
) -> list[dict[str, Any]]:
    bounds = bounds or _grid_bounds(scenario, slice_row, current_cell)
    roi_group = resolve_roi_group(scenario, slice_row)
    if valid_context is None:
        valid_context = _valid_cells_context(
            scenario=scenario,
            slice_row=slice_row,
            bounds=bounds,
            repo_root=repo_root,
            sidecar_path=sidecar_path,
        )
    if valid_cells is not None:
        valid_context = ValidCellsContext(
            valid_cells=set(valid_cells),
            proposal_cells=set(valid_cells),
            passable_cells=set(valid_cells),
            valid_cells_source="caller_valid_cells/v1",
            valid_cells_count=len(valid_cells),
            all_bounds_fallback_used=False,
        )
    valid_cells = set(valid_context.valid_cells)
    if not valid_cells:
        return []
    validation_budget = _path_validation_candidate_budget(config)
    raw_budget = int(config.get("dynamic_proposal_pool_limit_per_step", 48))
    distances, current_anchor, current_anchor_reason = _cheap_distance_map(current_cell, valid_cells, bounds)
    rows: list[dict[str, Any]] = []
    proposal_index = 0
    seen_cells: set[tuple[int, int]] = set()

    def append_cells(source: str, cells: list[tuple[int, int]]) -> None:
        nonlocal proposal_index
        for cell in cells:
            if len(rows) >= raw_budget:
                return
            if cell in seen_cells:
                continue
            seen_cells.add(cell)
            row = _proposal(
                source,
                cell,
                current_cell,
                covered_cells,
                step_index,
                proposal_index,
                config,
                scenario,
                slice_row,
                valid_cells,
                roi_group,
            )
            rows.append(
                _prefilter_proposal_row(
                    row,
                    current_cell=current_cell,
                    valid_context=valid_context,
                    distances=distances,
                    current_anchor=current_anchor,
                    current_anchor_reason=current_anchor_reason,
                )
            )
            proposal_index += 1

    append_cells(
        SOURCE_COVERAGE_FRONTIER,
        _coverage_frontier_cells(current_cell, covered_cells, config, bounds, valid_context.proposal_cells),
    )
    if _accepted_prefilter_count(rows) < validation_budget and len(rows) < raw_budget:
        for source, cell in _undercovered_component_proposals(
            current_cell,
            covered_cells,
            config,
            bounds,
            valid_cells,
            include_centroid=bool(config.get("dynamic_undercovered_centroid_enabled", False)),
        ):
            append_cells(source, [cell])
            if len(rows) >= raw_budget:
                break
            if _accepted_prefilter_count(rows) >= validation_budget:
                break
    if (
        bool(config.get("dynamic_low_cost_bridge_enabled", False))
        and _accepted_prefilter_count(rows) < validation_budget
        and len(rows) < raw_budget
    ):
        append_cells(SOURCE_LOW_COST_BRIDGE, _low_cost_bridge_cells(current_cell, covered_cells, config, bounds, valid_cells, distances))
    if _accepted_prefilter_count(rows) < validation_budget and len(rows) < raw_budget:
        append_cells(SOURCE_CONSERVATIVE_LOCAL, _conservative_local_cells(current_cell, covered_cells, bounds, valid_cells))

    return _mark_path_validation_budget(rows, validation_budget)


def _coverage_frontier_cells(
    current_cell: tuple[int, int],
    covered_cells: set[tuple[int, int]],
    config: dict[str, Any],
    bounds: tuple[int, int, int, int],
    valid_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Coverage frontier: uncovered valid cells adjacent to currently covered cells."""

    uncovered = valid_cells - covered_cells
    if not uncovered:
        return []
    anchors = set(covered_cells)
    anchors.add(current_cell)
    frontier: set[tuple[int, int]] = set()
    for anchor in anchors:
        for neighbor in _neighbors8(anchor):
            if neighbor in uncovered and _in_bounds(neighbor, bounds):
                frontier.add(neighbor)
    if not frontier:
        # If the current coverage memory is sparse, seed from nearest uncovered ROI cells.
        frontier = set(sorted(uncovered, key=lambda cell: (_manhattan(current_cell, cell), cell[0], cell[1]))[:32])
    target = max(1, int(config.get("dynamic_proposal_pool_limit_per_step", 48)))
    if len(frontier) < target:
        for cell in sorted(uncovered - frontier, key=lambda item: (_manhattan(current_cell, item), item[0], item[1])):
            frontier.add(cell)
            if len(frontier) >= target:
                break
    radius = int(config.get("coverage_radius_cells", 1))
    return sorted(
        frontier,
        key=lambda cell: (
            -len((_coverage_cells(current_cell, cell, radius, str(config.get("coverage_metric_mode", "path_line_plus_endpoint"))) & valid_cells) - covered_cells),
            _manhattan(current_cell, cell),
            cell[0],
            cell[1],
        ),
    )


def _accepted_prefilter_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("candidate_prefilter_reject_reason") is None)


def _path_validation_candidate_budget(config: dict[str, Any]) -> int:
    explicit = _positive_int(config.get("dynamic_path_validation_candidate_budget"))
    if explicit is not None:
        return explicit
    final_count = _positive_int(config.get("dynamic_max_candidates_per_step")) or 6
    if final_count >= 36:
        return 48
    if final_count >= 6:
        return 12
    return max(final_count * 2, final_count)


def _mark_path_validation_budget(rows: list[dict[str, Any]], validation_budget: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=_prefilter_sort_key)
    attempted_cells: set[tuple[int, int]] = set()
    for row in ordered:
        if len(attempted_cells) >= validation_budget:
            break
        if row.get("candidate_prefilter_reject_reason") is not None:
            continue
        cell = _cell_tuple(row.get("cell"))
        if cell is None:
            continue
        attempted_cells.add(cell)
    result: list[dict[str, Any]] = []
    for row in rows:
        cell = _cell_tuple(row.get("cell"))
        payload = dict(row)
        attempted = cell in attempted_cells if cell is not None else False
        payload["path_validation_attempted"] = bool(attempted)
        if not attempted and payload.get("candidate_prefilter_reject_reason") is None:
            payload["candidate_prefilter_reject_reason"] = "path_validation_budget_exceeded"
            payload["path_feedback_validation_source"] = "path_validation_budget_exceeded"
            payload["failure_reason"] = "path_validation_budget_exceeded"
            payload["planner_reachable"] = False
            payload["reachable"] = False
            payload["validation_diagnostic_flags"] = ["path_validation_budget_exceeded"]
        result.append(payload)
    return result


def _prefilter_sort_key(row: dict[str, Any]) -> tuple[int, float, int, float, float, tuple[int, int]]:
    rejected = 1 if row.get("candidate_prefilter_reject_reason") is not None else 0
    return (
        rejected,
        -_float_default(row.get("candidate_quality")),
        SOURCE_PRIORITY.get(str(row.get("frontier_candidate_source")), 99),
        _float_or_large(row.get("cheap_distance")),
        _float_default(row.get("coverage_overlap_ratio")),
        _cell_tuple(row.get("cell")) or (1_000_000, 1_000_000),
    )


def _undercovered_component_proposals(
    current_cell: tuple[int, int],
    covered_cells: set[tuple[int, int]],
    config: dict[str, Any],
    bounds: tuple[int, int, int, int],
    valid_cells: set[tuple[int, int]],
    *,
    include_centroid: bool = False,
) -> list[tuple[str, tuple[int, int]]]:
    uncovered = valid_cells - covered_cells
    if not uncovered:
        return []
    min_size = int(config.get("dynamic_frontier_min_uncovered_component_cells", 4))
    max_components = int(config.get("dynamic_frontier_max_components", 8))
    components = [component for component in _connected_components(uncovered) if len(component) >= max(1, min_size)]
    components = sorted(
        components,
        key=lambda component: (
            -len(component),
            min(_manhattan(current_cell, cell) for cell in component),
        ),
    )[:max(0, max_components)]
    rows: list[tuple[str, tuple[int, int]]] = []
    boundary_limit = max(1, int(config.get("dynamic_undercovered_boundary_candidates_per_component", 16)))
    for component in components:
        centroid = _nearest_component_cell_to_centroid(component)
        if include_centroid and centroid is not None:
            rows.append((SOURCE_UNDERCOVERED_CENTROID, centroid))
        for boundary in _component_boundary_cells(component, current_cell, bounds)[:boundary_limit]:
            if include_centroid and boundary == centroid:
                continue
            rows.append((SOURCE_UNDERCOVERED_BOUNDARY, boundary))
    return rows


def _low_cost_bridge_cells(
    current_cell: tuple[int, int],
    covered_cells: set[tuple[int, int]],
    config: dict[str, Any],
    bounds: tuple[int, int, int, int],
    valid_cells: set[tuple[int, int]],
    cheap_distances: dict[tuple[int, int], int] | None = None,
) -> list[tuple[int, int]]:
    radius = int(config.get("coverage_radius_cells", 1))
    uncovered = valid_cells - covered_cells
    candidates = []
    for cell in uncovered:
        coverage_gain = len((_coverage_cells(current_cell, cell, radius, str(config.get("coverage_metric_mode", "path_line_plus_endpoint"))) & valid_cells) - covered_cells)
        if coverage_gain <= 0:
            continue
        cheap_distance = (cheap_distances or {}).get(cell)
        distance = max(int(cheap_distance), 1) if cheap_distance is not None else max(_manhattan(current_cell, cell), 1)
        candidates.append((-(coverage_gain / distance), distance, cell[0], cell[1], cell))
    candidates.sort()
    return [row[-1] for row in candidates[: max(1, int(config.get("dynamic_min_efficiency_candidates_per_step", 1)) * 4)] if _in_bounds(row[-1], bounds)]


def _conservative_local_cells(
    current_cell: tuple[int, int],
    covered_cells: set[tuple[int, int]],
    bounds: tuple[int, int, int, int],
    valid_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    rows = []
    for dx, dy in DIRECTIONS_8:
        cell = (current_cell[0] + dx, current_cell[1] + dy)
        if _in_bounds(cell, bounds) and cell in valid_cells and cell not in covered_cells:
            rows.append(cell)
    return rows


def _connected_components(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(cells)
    components: list[set[tuple[int, int]]] = []
    while remaining:
        seed = min(remaining)
        stack = [seed]
        remaining.remove(seed)
        component = {seed}
        while stack:
            cell = stack.pop()
            for neighbor in _neighbors4(cell):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _nearest_component_cell_to_centroid(component: set[tuple[int, int]]) -> tuple[int, int] | None:
    if not component:
        return None
    cx = sum(cell[0] for cell in component) / len(component)
    cy = sum(cell[1] for cell in component) / len(component)
    return min(component, key=lambda cell: ((cell[0] - cx) ** 2 + (cell[1] - cy) ** 2, cell[0], cell[1]))


def _nearest_component_boundary_cell(
    component: set[tuple[int, int]],
    current_cell: tuple[int, int],
    bounds: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    if not component:
        return None
    boundary = [
        cell
        for cell in component
        if any(neighbor not in component or not _in_bounds(neighbor, bounds) for neighbor in _neighbors4(cell))
    ]
    if not boundary:
        boundary = list(component)
    return min(boundary, key=lambda cell: (_manhattan(current_cell, cell), cell[0], cell[1]))


def _component_boundary_cells(
    component: set[tuple[int, int]],
    current_cell: tuple[int, int],
    bounds: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    if not component:
        return []
    boundary = [
        cell
        for cell in component
        if any(neighbor not in component or not _in_bounds(neighbor, bounds) for neighbor in _neighbors4(cell))
    ]
    if not boundary:
        boundary = list(component)
    return sorted(boundary, key=lambda cell: (_manhattan(current_cell, cell), cell[0], cell[1]))


def _roi_valid_cells(
    *,
    scenario: dict[str, Any],
    slice_row: dict[str, Any],
    bounds: tuple[int, int, int, int],
    repo_root: Path,
    sidecar_path: Path | None,
) -> set[tuple[int, int]]:
    return _valid_cells_context(
        scenario=scenario,
        slice_row=slice_row,
        bounds=bounds,
        repo_root=repo_root,
        sidecar_path=sidecar_path,
    ).valid_cells


def _valid_cells_context(
    *,
    scenario: dict[str, Any],
    slice_row: dict[str, Any],
    bounds: tuple[int, int, int, int],
    repo_root: Path,
    sidecar_path: Path | None,
) -> ValidCellsContext:
    del scenario, repo_root
    passable_cells = _sidecar_passable_cells(sidecar_path, bounds)
    roi_cells = _roi_cells_from_slice(slice_row, bounds)
    if passable_cells is not None and roi_cells:
        valid = passable_cells & roi_cells
        return ValidCellsContext(
            valid_cells=valid,
            proposal_cells=set(roi_cells),
            passable_cells=passable_cells,
            valid_cells_source="sidecar_passable_mask_and_roi_valid_cells/v1",
            valid_cells_count=len(valid),
        )
    if passable_cells is not None:
        return ValidCellsContext(
            valid_cells=set(passable_cells),
            proposal_cells=set(passable_cells),
            passable_cells=passable_cells,
            valid_cells_source="sidecar_passable_mask/v1",
            valid_cells_count=len(passable_cells),
        )
    if roi_cells:
        return ValidCellsContext(
            valid_cells=set(roi_cells),
            proposal_cells=set(roi_cells),
            passable_cells=set(roi_cells),
            valid_cells_source="roi_valid_cells/v1",
            valid_cells_count=len(roi_cells),
        )
    return ValidCellsContext(
        valid_cells=set(),
        proposal_cells=set(),
        passable_cells=None,
        valid_cells_source="missing",
        valid_cells_count=0,
    )


def _sidecar_passable_cells(sidecar_path: Path | None, bounds: tuple[int, int, int, int]) -> set[tuple[int, int]] | None:
    if sidecar_path is not None and sidecar_path.is_file():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            mask = sidecar.get("passable_mask")
            if isinstance(mask, list):
                cells: set[tuple[int, int]] = set()
                for y, row in enumerate(mask):
                    if not isinstance(row, list):
                        continue
                    for x, value in enumerate(row):
                        cell = (int(x), int(y))
                        if bool(value) and _in_bounds(cell, bounds):
                            cells.add(cell)
                return cells
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _roi_cells_from_slice(slice_row: dict[str, Any], bounds: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    roi_cells = slice_row.get("roi_valid_cells")
    if isinstance(roi_cells, list):
        parsed = {_cell_tuple(cell) for cell in roi_cells}
        parsed.discard(None)
        return {cell for cell in parsed if cell is not None and _in_bounds(cell, bounds)}
    return set()


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
    scenario: dict[str, Any],
    slice_row: dict[str, Any],
    valid_cells: set[tuple[int, int]],
    roi_group: str,
) -> dict[str, Any]:
    metrics = _coverage_metrics(
        current_cell,
        cell,
        covered_cells,
        config,
        scenario=scenario,
        slice_row=slice_row,
        valid_cells=valid_cells,
        roi_group=roi_group,
    )
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
        "candidate_generation_algorithm_source": ALGORITHM_SOURCE,
        "proposal_only": True,
        "proposal_validated_by_path_feedback": False,
        "coverage_source": COVERAGE_SOURCE,
        "coverage_cell_set_kind": COVERAGE_CELL_SET_KIND,
        "coverage_dedupe_scope": COVERAGE_DEDUPE_SCOPE,
        "coverage_validated_by_path_feedback": False,
        "coverage_validation_source": "offline_geometric_counterfactual_not_path_feedback",
        **metrics,
    }


def _prefilter_proposal_row(
    row: dict[str, Any],
    *,
    current_cell: tuple[int, int],
    valid_context: ValidCellsContext,
    distances: dict[tuple[int, int], int],
    current_anchor: tuple[int, int] | None,
    current_anchor_reason: str | None,
) -> dict[str, Any]:
    cell = _cell_tuple(row.get("cell"))
    endpoint_passable = bool(cell is not None and _endpoint_passable(cell, valid_context))
    known_free = bool(cell is not None and cell in valid_context.valid_cells)
    same_component = bool(cell is not None and cell in distances)
    cheap_distance = distances.get(cell) if cell is not None else None
    expected_new = _finite_float(row.get("expected_new_coverage_cell_count"))
    reason = None
    if current_anchor is None:
        reason = current_anchor_reason or "current_cell_not_known_free"
    elif cell is None:
        reason = "invalid_candidate_cell"
    elif not endpoint_passable:
        reason = "candidate_endpoint_not_passable"
    elif not known_free:
        reason = "candidate_endpoint_not_known_free"
    elif not same_component:
        reason = "candidate_not_same_component_as_current"
    elif cheap_distance is None:
        reason = "cheap_distance_missing"
    elif expected_new is None or expected_new <= 0.0:
        reason = "expected_new_coverage_cell_count_zero"
    quality = _candidate_quality(row, cheap_distance=cheap_distance)
    payload = dict(row)
    payload.update(
        {
            "valid_cells_source": valid_context.valid_cells_source,
            "valid_cells_count": int(valid_context.valid_cells_count),
            "all_bounds_fallback_used": False,
            "candidate_endpoint_passable": endpoint_passable,
            "candidate_known_free": known_free,
            "candidate_same_component_as_current": same_component,
            "cheap_distance": float(cheap_distance) if cheap_distance is not None else None,
            "cheap_distance_source": "sidecar_passable_mask_bfs_8_neighbor_no_corner_cutting/v1",
            "current_cell_known_free": current_cell in valid_context.valid_cells,
            "current_cell_anchor": list(current_anchor) if current_anchor is not None else None,
            "current_cell_anchor_reason": current_anchor_reason,
            "candidate_prefilter_reject_reason": reason,
            "candidate_quality": quality,
            "path_validation_attempted": False,
        }
    )
    if reason is not None:
        payload.update(
            {
                "proposal_only": True,
                "proposal_validated_by_path_feedback": False,
                "path_feedback_validation_source": reason,
                "failure_reason": reason,
                "planner_reachable": False,
                "reachable": False,
                "open_grid_fallback_used": False,
                "validation_diagnostic_flags": [reason],
            }
        )
    return payload


def _candidate_quality(row: dict[str, Any], *, cheap_distance: int | None) -> float:
    coverage = _float_default(row.get("expected_new_coverage_cell_count"))
    distance = max(float(cheap_distance if cheap_distance is not None else _float_or_large(row.get("relative_distance"))), 1.0)
    roi_value = _float_default(row.get("value"))
    roi_delta = _float_default(row.get("roi_weighted_coverage_delta"))
    revisit_penalty = _float_default(row.get("revisit_penalty"))
    local_risk = _float_default(row.get("risk"), 0.0)
    source = str(row.get("frontier_candidate_source"))
    frontier_priority = 2.0 if source == SOURCE_COVERAGE_FRONTIER else 0.75 if source == SOURCE_UNDERCOVERED_BOUNDARY else 0.25
    return float(coverage + (coverage / distance) + roi_value + roi_delta + frontier_priority - revisit_penalty - (0.01 * distance) - (0.05 * local_risk))


def _endpoint_passable(cell: tuple[int, int], valid_context: ValidCellsContext) -> bool:
    if valid_context.passable_cells is None:
        return cell in valid_context.valid_cells
    return cell in valid_context.passable_cells


def _cheap_distance_map(
    current_cell: tuple[int, int],
    valid_cells: set[tuple[int, int]],
    bounds: tuple[int, int, int, int],
) -> tuple[dict[tuple[int, int], int], tuple[int, int] | None, str | None]:
    if current_cell in valid_cells:
        anchor = current_cell
        reason = "current_cell_known_free"
    elif valid_cells:
        anchor = min(valid_cells, key=lambda cell: (_manhattan(current_cell, cell), cell[0], cell[1]))
        reason = "nearest_known_free_anchor"
    else:
        return {}, None, "current_cell_not_known_free"
    distances = {anchor: 0}
    queue = [anchor]
    index = 0
    while index < len(queue):
        cell = queue[index]
        index += 1
        base_distance = distances[cell]
        for neighbor in _neighbors8_no_corner_cutting(cell, valid_cells, bounds):
            if neighbor in distances:
                continue
            distances[neighbor] = base_distance + 1
            queue.append(neighbor)
    return distances, anchor, reason


def _neighbors8_no_corner_cutting(
    cell: tuple[int, int],
    valid_cells: set[tuple[int, int]],
    bounds: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    x, y = cell
    for dx, dy in DIRECTIONS_8:
        neighbor = (x + dx, y + dy)
        if neighbor not in valid_cells or not _in_bounds(neighbor, bounds):
            continue
        if dx != 0 and dy != 0 and ((x + dx, y) not in valid_cells or (x, y + dy) not in valid_cells):
            continue
        rows.append(neighbor)
    return rows


def _merge_validation_rows_with_prefilter_rows(
    proposal_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (str(row.get("proposal_id") or ""), _cell_tuple(row.get("cell"))): row
        for row in validation_rows
    }
    merged: list[dict[str, Any]] = []
    for proposal in proposal_rows:
        key = (str(proposal.get("proposal_id") or ""), _cell_tuple(proposal.get("cell")))
        validation = by_key.get(key)
        if validation is not None:
            merged.append({**proposal, **validation, "path_validation_attempted": True})
            continue
        reason = str(proposal.get("candidate_prefilter_reject_reason") or "path_validation_not_attempted")
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
                "path_validation_attempted": False,
            }
        )
        merged.append(row)
    return merged


def _prefilter_failure_row(
    *,
    scenario: dict[str, Any],
    slice_row: dict[str, Any],
    current_cell: tuple[int, int],
    step_index: int,
    reason: str,
    valid_context: ValidCellsContext,
) -> dict[str, Any]:
    scenario_id = str(slice_row.get("scenario_id") or scenario.get("scenario_id") or "scenario")
    return {
        "schema_version": "xunce-dynamic-frontier-nbv-prefilter-audit/v1",
        "scenario_id": scenario_id,
        "step_index": int(step_index),
        "proposal_id": f"{scenario_id}-step-{step_index:03d}-{reason}",
        "cell": [int(current_cell[0]), int(current_cell[1])],
        "frontier_candidate_source": "none",
        "candidate_generation_source": GENERATION_SOURCE,
        "candidate_generation_algorithm_source": ALGORITHM_SOURCE,
        "proposal_only": True,
        "proposal_validated_by_path_feedback": False,
        "path_feedback_validation_source": reason,
        "failure_reason": reason,
        "planner_reachable": False,
        "reachable": False,
        "open_grid_fallback_used": False,
        "validation_diagnostic_flags": [reason],
        "valid_cells_source": valid_context.valid_cells_source,
        "valid_cells_count": int(valid_context.valid_cells_count),
        "all_bounds_fallback_used": False,
        "candidate_endpoint_passable": False,
        "candidate_known_free": False,
        "candidate_same_component_as_current": False,
        "candidate_prefilter_reject_reason": reason,
        "cheap_distance": None,
        "candidate_quality": None,
        "path_validation_attempted": False,
    }


def _coverage_metrics(
    current_cell: tuple[int, int],
    cell: tuple[int, int],
    covered_cells: set[tuple[int, int]],
    config: dict[str, Any],
    *,
    scenario: dict[str, Any],
    slice_row: dict[str, Any],
    valid_cells: set[tuple[int, int]],
    roi_group: str,
) -> dict[str, Any]:
    radius = int(config.get("coverage_radius_cells", 1))
    mode = str(config.get("coverage_metric_mode", "path_line_plus_endpoint"))
    denominator_context = _coverage_denominator_context(config, valid_cells)
    denominator = float(denominator_context["coverage_denominator_cells"])
    endpoint_cells = set(_footprint(cell, radius)) & valid_cells
    path_cells = _coverage_cells(current_cell, cell, radius, mode) & valid_cells
    union_cells = endpoint_cells | path_cells
    new_cells = union_cells - covered_cells
    overlap = union_cells & covered_cells
    new_count = len(new_cells)
    roi_weight, roi_weight_source, roi_weight_reason_codes = _roi_weight_context(
        roi_group=roi_group,
        scenario=scenario,
        slice_row=slice_row,
        config=config,
    )
    return {
        **denominator_context,
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
        "roi_weight": roi_weight,
        "roi_weight_source": roi_weight_source,
        "roi_weight_reason_codes": roi_weight_reason_codes,
    }


def _enrich_validated_candidate(
    row: dict[str, Any],
    *,
    current_cell: tuple[int, int],
    covered_cells: set[tuple[int, int]],
    config: dict[str, Any],
    scenario: dict[str, Any],
    slice_row: dict[str, Any],
    valid_cells: set[tuple[int, int]],
    step_index: int,
    roi_group: str,
) -> dict[str, Any]:
    cell = _cell_tuple(row.get("cell"))
    enriched = dict(row)
    enriched["step_index"] = int(step_index)
    enriched["candidate_generation_source"] = GENERATION_SOURCE
    enriched["candidate_generation_algorithm_source"] = ALGORITHM_SOURCE
    enriched["coverage_source"] = COVERAGE_SOURCE
    enriched["coverage_validated_by_path_feedback"] = False
    enriched["coverage_validation_source"] = "offline_geometric_counterfactual_not_path_feedback"
    enriched["coverage_cell_set_kind"] = COVERAGE_CELL_SET_KIND
    enriched["coverage_dedupe_scope"] = COVERAGE_DEDUPE_SCOPE
    enriched.setdefault("roi_group", roi_group)
    if cell is not None:
        enriched.update(
            _coverage_metrics(
                current_cell,
                cell,
                covered_cells,
                config,
                scenario=scenario,
                slice_row=slice_row,
                valid_cells=valid_cells,
                roi_group=roi_group,
            )
        )
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


def _mark_selected_validation_rows(rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> None:
    selected_by_key = {
        (str(row.get("proposal_id") or ""), _cell_tuple(row.get("cell"))): row
        for row in selected
    }
    for row in rows:
        match = selected_by_key.get((str(row.get("proposal_id") or ""), _cell_tuple(row.get("cell"))))
        if match is None:
            continue
        row["candidate_selected_for_action_set"] = True
        row["candidate_selection_mode"] = match.get("candidate_selection_mode")
        row["candidate_selection_role"] = match.get("candidate_selection_role")
        row["action_index"] = match.get("action_index")


def _select_formal_candidates(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    limit = max(0, limit)
    if limit <= 0:
        return []
    if not rows:
        return []
    selected: list[dict[str, Any]] = []

    def add(row: dict[str, Any], role: str) -> None:
        if len(selected) >= limit:
            return
        cell = _cell_tuple(row.get("cell"))
        if cell is None or any(_cell_tuple(existing.get("cell")) == cell for existing in selected):
            return
        payload = dict(row)
        payload.setdefault("candidate_selection_mode", "validated_pareto_diverse")
        payload.setdefault("candidate_selection_role", role)
        selected.append(payload)

    frontier_rows = [
        row
        for row in rows
        if str(row.get("frontier_candidate_source"))
        in {SOURCE_COVERAGE_FRONTIER, SOURCE_UNDERCOVERED_CENTROID, SOURCE_UNDERCOVERED_BOUNDARY}
    ]
    if frontier_rows:
        add(min(frontier_rows, key=_frontier_selection_key), "frontier_or_undercovered")
    add(min(rows, key=_low_cost_key), "low_cost")
    add(max(rows, key=_coverage_efficiency_score), "coverage_efficiency")

    for row in _pareto_ordered(rows):
        add(row, "pareto_frontier")
    for row in sorted(rows, key=_formal_candidate_sort_key):
        add(row, "stable_fill")
    return selected[:limit]


def _frontier_selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, int, tuple[int, int]]:
    return (
        -_float_default(row.get("candidate_quality")),
        -_coverage_score(row),
        _float_or_large(row.get("cheap_distance")),
        _float_or_large(row.get("path_cost")),
        SOURCE_PRIORITY.get(str(row.get("frontier_candidate_source")), 99),
        _cell_tuple(row.get("cell")) or (1_000_000, 1_000_000),
    )


def _low_cost_key(row: dict[str, Any]) -> tuple[float, float, float, int, tuple[int, int]]:
    return (
        _float_or_large(row.get("path_cost")),
        _float_or_large(row.get("cheap_distance")),
        _float_or_large(row.get("risk")),
        SOURCE_PRIORITY.get(str(row.get("frontier_candidate_source")), 99),
        _cell_tuple(row.get("cell")) or (1_000_000, 1_000_000),
    )


def _coverage_efficiency_score(row: dict[str, Any]) -> tuple[float, float, float]:
    coverage = _coverage_score(row)
    cost = max(_float_or_large(row.get("path_cost")), TOLERANCE)
    risk = max(_float_default(row.get("risk")), 0.0)
    return (coverage / cost, coverage, -risk)


def _pareto_ordered(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = [row for row in rows if _pareto_rank(row, rows) == 0]
    remainder = [row for row in rows if row not in frontier]
    return sorted(frontier, key=_formal_candidate_sort_key) + sorted(remainder, key=_formal_candidate_sort_key)


def _pareto_rank(row: dict[str, Any], rows: list[dict[str, Any]]) -> int:
    rank = 0
    for other in rows:
        if other is row:
            continue
        if _dominates(other, row):
            rank += 1
    return rank


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_tuple = (_coverage_score(left), -_float_or_large(left.get("path_cost")), -_float_or_large(left.get("risk")))
    right_tuple = (_coverage_score(right), -_float_or_large(right.get("path_cost")), -_float_or_large(right.get("risk")))
    return all(l >= r - TOLERANCE for l, r in zip(left_tuple, right_tuple)) and any(l > r + TOLERANCE for l, r in zip(left_tuple, right_tuple))


def _formal_candidate_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, int, tuple[int, int]]:
    return (
        -_float_default(row.get("candidate_quality")),
        -_coverage_score(row),
        _float_or_large(row.get("cheap_distance")),
        _float_or_large(row.get("path_cost")),
        _float_or_large(row.get("risk")),
        SOURCE_PRIORITY.get(str(row.get("frontier_candidate_source")), 99),
        _cell_tuple(row.get("cell")) or (1_000_000, 1_000_000),
    )


def _coverage_score(row: dict[str, Any]) -> float:
    return _float_default(row.get("expected_new_coverage_cell_count"))


def _float_or_large(value: Any) -> float:
    parsed = _finite_float(value)
    return parsed if parsed is not None else 1.0e12


def _float_default(value: Any, default: float = 0.0) -> float:
    parsed = _finite_float(value)
    return parsed if parsed is not None else float(default)


def _limit_rows_by_family(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    selected: list[dict[str, Any]] = []
    sources = (
        SOURCE_COVERAGE_FRONTIER,
        SOURCE_UNDERCOVERED_CENTROID,
        SOURCE_UNDERCOVERED_BOUNDARY,
        SOURCE_LOW_COST_BRIDGE,
        SOURCE_CONSERVATIVE_LOCAL,
        "frontier_boundary",
        "roi_undercovered_boundary",
        "incumbent_neighborhood",
    )
    for source in sources:
        if len(selected) >= limit:
            break
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
    valid_context: ValidCellsContext,
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
        "dynamic_path_validation_candidate_budget": _path_validation_candidate_budget(config),
        "valid_cells_source": valid_context.valid_cells_source,
        "valid_cells_hash": _hash_payload(sorted([list(cell) for cell in valid_context.valid_cells])),
        "prefilter_source": "frontier_first_passable_connected_new_coverage/v1",
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


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = cell
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _neighbors8(cell: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = cell
    return [(x + dx, y + dy) for dx, dy in DIRECTIONS_8]


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


def _coverage_denominator_context(config: dict[str, Any], valid_cells: set[tuple[int, int]]) -> dict[str, Any]:
    mode = str(config.get("coverage_denominator_mode", "fixed_config_cells"))
    configured = _finite_float(config.get("coverage_denominator_cells")) or 1000.0
    if mode == "roi_valid_cells" and valid_cells:
        return {
            "coverage_denominator_cells": float(len(valid_cells)),
            "coverage_denominator_mode": mode,
            "coverage_denominator_source": "dynamic_roi_valid_cells/v1",
            "coverage_denominator_reason_codes": [],
        }
    if mode == "raw_cells_only":
        return {
            "coverage_denominator_cells": 1.0,
            "coverage_denominator_mode": mode,
            "coverage_denominator_source": "raw_cells_only_no_rate_denominator/v1",
            "coverage_denominator_reason_codes": ["raw_cells_only_dynamic_candidate_rate_for_sorting_only"],
        }
    return {
        "coverage_denominator_cells": float(max(configured, TOLERANCE)),
        "coverage_denominator_mode": "fixed_config_cells" if mode not in {"roi_valid_cells", "fixed_config_cells"} else mode,
        "coverage_denominator_source": "config_coverage_denominator_cells/v1",
        "coverage_denominator_reason_codes": [] if mode == "fixed_config_cells" else ["roi_valid_cells_unavailable_fallback_to_config"],
    }


def _roi_weight_context(
    *,
    roi_group: str,
    scenario: dict[str, Any],
    slice_row: dict[str, Any],
    config: dict[str, Any],
) -> tuple[float, str, list[str]]:
    for source_name, source in (("scenario", scenario), ("slice_row", slice_row)):
        value = source.get("roi_weight") if isinstance(source, dict) else None
        parsed = _finite_float(value)
        if parsed is not None:
            return parsed, f"{source_name}.roi_weight", []
    metadata = slice_row.get("metadata") if isinstance(slice_row.get("metadata"), dict) else {}
    parsed = _finite_float(metadata.get("roi_weight"))
    if parsed is not None:
        return parsed, "slice_row.metadata.roi_weight", []
    weight_map = config.get("roi_group_weight_map")
    if isinstance(weight_map, dict):
        parsed = _finite_float(weight_map.get(roi_group))
        if parsed is not None:
            return parsed, "config.roi_group_weight_map", []
    if roi_group in ROI_WEIGHTS:
        return float(ROI_WEIGHTS[roi_group]), "built_in_roi_group_weight_map", []
    return 1.0, "default_unweighted_roi", ["roi_weight_defaulted"]


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
