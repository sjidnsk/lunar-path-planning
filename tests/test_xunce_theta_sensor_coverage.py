from __future__ import annotations

from scripts.xunce_theta_sensor_coverage import (
    candidate_viewpoint_hash,
    expand_candidate_viewpoints,
    theta_coverage_hash,
    theta_bins,
    visible_cells_for_viewpoint,
)


def test_theta_bins_default_eight_directions() -> None:
    assert theta_bins(theta_bin_count=8, theta_step_deg=45) == [0, 45, 90, 135, 180, 225, 270, 315]


def test_visible_cells_respects_heading_and_90_degree_fov() -> None:
    east = visible_cells_for_viewpoint((0, 0), theta_deg=0, sensor_range_cells=2, sensor_fov_deg=90)
    north = visible_cells_for_viewpoint((0, 0), theta_deg=90, sensor_range_cells=2, sensor_fov_deg=90)

    assert (2, 0) in east
    assert (0, 2) not in east
    assert (0, 2) in north
    assert (2, 0) not in north
    assert (0, 0) in east


def test_visible_cells_wraps_fov_across_zero_degrees() -> None:
    cells = visible_cells_for_viewpoint((0, 0), theta_deg=315, sensor_range_cells=2, sensor_fov_deg=90)

    assert (1, 0) in cells
    assert (1, -1) in cells
    assert (0, 1) not in cells


def test_theta_coverage_hash_is_stable_and_order_independent() -> None:
    cells = [(2, 0), (0, 0), (1, 1)]

    assert theta_coverage_hash(cells) == theta_coverage_hash(reversed(cells))


def test_expand_candidate_viewpoints_and_hash_include_theta() -> None:
    viewpoints = expand_candidate_viewpoints(
        [[1, 2], [3, 4]],
        theta_values=[0, 90],
        sensor_model_id="theta-fov/v1",
        sensor_range_cells=3,
        sensor_fov_deg=90,
    )

    assert [row["candidate_viewpoint"] for row in viewpoints] == [[1, 2, 0], [1, 2, 90], [3, 4, 0], [3, 4, 90]]
    assert candidate_viewpoint_hash(viewpoints[:1]) != candidate_viewpoint_hash(viewpoints[1:2])
