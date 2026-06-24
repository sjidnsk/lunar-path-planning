from __future__ import annotations

from scripts.xunce_obstacle_aware_theta_sensor_coverage import (
    obstacle_aware_theta_coverage_hash,
    visible_cells_for_viewpoint_with_obstacles,
)
from scripts.xunce_theta_sensor_coverage import visible_cells_for_viewpoint


def test_no_obstacles_matches_existing_theta_fov() -> None:
    current = visible_cells_for_viewpoint((0, 0), theta_deg=0, sensor_range_cells=3, sensor_fov_deg=90)
    obstacle_aware = visible_cells_for_viewpoint_with_obstacles(
        (0, 0),
        theta_deg=0,
        sensor_range_cells=3,
        sensor_fov_deg=90,
        obstacle_cells=set(),
    )

    assert obstacle_aware.visible_cells == current
    assert obstacle_aware.occluded_cell_count == 0
    assert obstacle_aware.blocked_by_obstacle_count == 0


def test_obstacle_blocks_cells_behind_it_on_line_of_sight() -> None:
    result = visible_cells_for_viewpoint_with_obstacles(
        (0, 0),
        theta_deg=0,
        sensor_range_cells=4,
        sensor_fov_deg=30,
        obstacle_cells={(1, 0)},
    )

    assert (0, 0) in result.visible_cells
    assert (1, 0) not in result.visible_cells
    assert (2, 0) not in result.visible_cells
    assert (3, 0) not in result.visible_cells
    assert result.blocked_by_obstacle_count == 1
    assert result.occluded_cell_count >= 2
    assert (2, 0) in result.occluded_cells


def test_target_obstacle_is_blocked_but_sensor_cell_is_not_self_occluded() -> None:
    result = visible_cells_for_viewpoint_with_obstacles(
        (0, 0),
        theta_deg=0,
        sensor_range_cells=2,
        sensor_fov_deg=90,
        obstacle_cells={(0, 0), (1, 0)},
    )

    assert (0, 0) in result.visible_cells
    assert (1, 0) not in result.visible_cells
    assert result.blocked_by_obstacle_count == 1


def test_obstacle_aware_hash_changes_when_occlusion_removes_visible_cells() -> None:
    clear = visible_cells_for_viewpoint_with_obstacles(
        (0, 0),
        theta_deg=0,
        sensor_range_cells=4,
        sensor_fov_deg=30,
        obstacle_cells=set(),
    )
    blocked = visible_cells_for_viewpoint_with_obstacles(
        (0, 0),
        theta_deg=0,
        sensor_range_cells=4,
        sensor_fov_deg=30,
        obstacle_cells={(1, 0)},
    )

    assert obstacle_aware_theta_coverage_hash(clear.visible_cells) != obstacle_aware_theta_coverage_hash(blocked.visible_cells)
