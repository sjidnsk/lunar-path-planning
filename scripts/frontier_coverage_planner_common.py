from __future__ import annotations

from typing import Any

try:
    from global_99_coverage_contract import (
        Cell,
        bfs_tree,
        connected_components,
        coverage_footprint,
        neighbors4,
        reconstruct_path,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.global_99_coverage_contract import (
        Cell,
        bfs_tree,
        connected_components,
        coverage_footprint,
        neighbors4,
        reconstruct_path,
    )


def coverage_rate(covered_cells: set[Cell], target_cells: set[Cell]) -> float:
    return (len(covered_cells) / len(target_cells)) if target_cells else 0.0


def score_frontier_candidate(
    *,
    path_cost_m: float,
    revisited_path_cell_count: int,
    new_covered_cell_count: int,
    revisit_penalty_weight: float,
    new_coverage_weight: float,
) -> float:
    return (
        path_cost_m
        + revisit_penalty_weight * revisited_path_cell_count
        - new_coverage_weight * new_covered_cell_count
    )


def frontier_cells(coverage_map_cells: set[Cell], navigation_cells: set[Cell]) -> set[Cell]:
    uncovered_navigation = navigation_cells - coverage_map_cells
    return {
        cell
        for cell in uncovered_navigation
        if any(neighbor in coverage_map_cells for neighbor in neighbors4(cell))
    }


def select_frontier_candidate(
    *,
    current_cell: Cell,
    frontier: set[Cell],
    navigation_cells: set[Cell],
    target_cells: set[Cell],
    covered_target_cells: set[Cell],
    coverage_map_cells: set[Cell],
    width: int,
    height: int,
    resolution_m: float,
    coverage_radius_cells: int,
    revisit_penalty_weight: float,
    new_coverage_weight: float,
) -> dict[str, Any] | None:
    candidates = enumerate_frontier_candidates(
        current_cell=current_cell,
        frontier=frontier,
        navigation_cells=navigation_cells,
        target_cells=target_cells,
        covered_target_cells=covered_target_cells,
        coverage_map_cells=coverage_map_cells,
        width=width,
        height=height,
        resolution_m=resolution_m,
        coverage_radius_cells=coverage_radius_cells,
        revisit_penalty_weight=revisit_penalty_weight,
        new_coverage_weight=new_coverage_weight,
        component_best_only=True,
    )
    return candidates[0] if candidates else None


def enumerate_frontier_candidates(
    *,
    current_cell: Cell,
    frontier: set[Cell],
    navigation_cells: set[Cell],
    target_cells: set[Cell],
    covered_target_cells: set[Cell],
    coverage_map_cells: set[Cell],
    width: int,
    height: int,
    resolution_m: float,
    coverage_radius_cells: int,
    revisit_penalty_weight: float,
    new_coverage_weight: float,
    component_best_only: bool = False,
) -> list[dict[str, Any]]:
    distance, parent = bfs_tree(current_cell, navigation_cells)
    components = connected_components(frontier)
    candidates: list[dict[str, Any]] = []
    for cluster_index, component in enumerate(components):
        component_candidates: list[dict[str, Any]] = []
        for cell in sorted(component, key=lambda item: (item[1], item[0])):
            if cell not in distance:
                continue
            path = reconstruct_path(parent, cell)
            path_cells = set(path)
            footprint = coverage_footprint(cell, width, height, coverage_radius_cells) & navigation_cells
            event_cells = footprint | path_cells
            new_target_cells = (event_cells & target_cells) - covered_target_cells
            revisited_path_cell_count = sum(1 for path_cell in path if path_cell in coverage_map_cells)
            path_cost_m = distance[cell] * resolution_m
            score = score_frontier_candidate(
                path_cost_m=path_cost_m,
                revisited_path_cell_count=revisited_path_cell_count,
                new_covered_cell_count=len(new_target_cells),
                revisit_penalty_weight=revisit_penalty_weight,
                new_coverage_weight=new_coverage_weight,
            )
            component_candidates.append(
                {
                    "cluster_index": cluster_index,
                    "selected_waypoint": cell,
                    "path": path,
                    "event_cells": event_cells,
                    "new_target_cell_count": len(new_target_cells),
                    "revisited_path_cell_count": revisited_path_cell_count,
                    "path_cost_m": path_cost_m,
                    "score": score,
                    "frontier_cluster_count": len(components),
                }
            )
        if component_candidates:
            component_candidates.sort(
                key=lambda candidate: (
                    -candidate["new_target_cell_count"],
                    candidate["path_cost_m"],
                    candidate["selected_waypoint"][1],
                    candidate["selected_waypoint"][0],
                )
            )
            if component_best_only:
                component_candidates = component_candidates[:1]
            candidates.extend(component_candidates)
    candidates.sort(
        key=lambda candidate: (
            candidate["score"],
            -candidate["new_target_cell_count"],
            candidate["path_cost_m"],
            candidate["selected_waypoint"][1],
            candidate["selected_waypoint"][0],
        )
    )
    for rank, candidate in enumerate(candidates):
        candidate["baseline_rank"] = rank
    return candidates
