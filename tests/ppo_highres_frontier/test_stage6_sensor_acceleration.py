from __future__ import annotations

import ast
import inspect
import math
import textwrap
from collections.abc import Iterable

import numpy as np
import pytest

from lunar_exploration_ppo.env.action_execution import (
    ActionExecutionDiagnostics,
    build_sensor_poses,
)
from lunar_exploration_ppo.env.map_state import ObservedMapState
from lunar_exploration_ppo.env.scenario import TruthMap
from lunar_exploration_ppo.env.sensor_model import (
    ObservationDelta,
    SensorDiagnostics,
    SensorPose,
    SensorUpdater,
    ray_cells_from_world,
)
from lunar_exploration_ppo.utils.geometry import (
    CellXY,
    GridGeometry,
    WorldXY,
    normalize_theta,
)


_STATE_ARRAY_FIELDS = (
    "observed_mask",
    "confidence",
    "height",
    "obstacle",
    "slope_deg",
    "traversability",
    "observed_safe_mask",
)


def _legacy_ray_cells_from_world(
    origin: WorldXY,
    angle: float,
    geometry: GridGeometry,
    *,
    range_m: float,
) -> tuple[CellXY, ...]:
    """Frozen pre-acceleration DDA oracle; intentionally independent of production."""

    current = geometry.world_to_cell(origin)
    direction_x = math.cos(angle)
    direction_y = math.sin(angle)
    step_x = 1 if direction_x > 0.0 else -1 if direction_x < 0.0 else 0
    step_y = 1 if direction_y > 0.0 else -1 if direction_y < 0.0 else 0
    resolution = geometry.resolution_m

    if step_x:
        boundary_x = geometry.origin.x + (current.x + (1 if step_x > 0 else 0)) * resolution
        t_max_x = (boundary_x - origin.x) / direction_x
        t_delta_x = resolution / abs(direction_x)
    else:
        t_max_x = math.inf
        t_delta_x = math.inf
    if step_y:
        boundary_y = geometry.origin.y + (current.y + (1 if step_y > 0 else 0)) * resolution
        t_max_y = (boundary_y - origin.y) / direction_y
        t_delta_y = resolution / abs(direction_y)
    else:
        t_max_y = math.inf
        t_delta_y = math.inf

    cells: list[CellXY] = []
    while geometry.in_bounds(current):
        cells.append(current)
        next_distance = min(t_max_x, t_max_y)
        if next_distance > range_m:
            break
        if math.isclose(t_max_x, t_max_y, rel_tol=0.0, abs_tol=1e-12):
            current = CellXY(current.x + step_x, current.y + step_y)
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        elif t_max_x < t_max_y:
            current = CellXY(current.x + step_x, current.y)
            t_max_x += t_delta_x
        else:
            current = CellXY(current.x, current.y + step_y)
            t_max_y += t_delta_y
    return tuple(cells)


def _legacy_reveal(
    updater: SensorUpdater,
    truth: TruthMap,
    observed_state: ObservedMapState,
    poses: Iterable[SensorPose],
) -> ObservationDelta:
    """Frozen pre-acceleration reveal oracle, including its scalar range gate."""

    pose_tuple = tuple(poses)
    visible: set[CellXY] = set()
    visits = 0
    rays_per_sample = int(round(updater.fov_deg / updater.ray_angle_step_deg)) + 1
    half_fov = updater.fov_deg / 2.0
    for pose in pose_tuple:
        for index in range(rays_per_sample):
            offset_deg = -half_fov + index * updater.ray_angle_step_deg
            angle = pose.heading + math.radians(offset_deg)
            for ray_cell_index, cell in enumerate(
                _legacy_ray_cells_from_world(
                    pose.world_xy,
                    angle,
                    truth.geometry,
                    range_m=updater.range_m,
                )
            ):
                center = truth.geometry.cell_to_world_center(cell)
                if math.hypot(
                    center.x - pose.world_xy.x,
                    center.y - pose.world_xy.y,
                ) > updater.range_m:
                    continue
                visits += 1
                visible.add(cell)
                if ray_cell_index > 0 and (
                    truth.hard_obstacle[cell.y, cell.x]
                    or truth.slope_deg[cell.y, cell.x] > updater.max_slope_deg
                ):
                    break
    ordered_visible = tuple(sorted(visible, key=lambda cell: (cell.y, cell.x)))
    newly_observed = observed_state.reveal_cells(
        truth,
        ordered_visible,
        min_clearance_m=updater.min_clearance_m,
        max_slope_deg=updater.max_slope_deg,
        traversability_threshold=updater.traversability_threshold,
    )
    diagnostics = SensorDiagnostics(
        sample_count=len(pose_tuple),
        ray_count=len(pose_tuple) * rays_per_sample,
        cell_visit_count=visits,
        unique_visible_cell_count=len(ordered_visible),
        duplicate_cell_visits=visits - len(ordered_visible),
        sample_sources=tuple(pose.source for pose in pose_tuple),
        sample_headings=tuple(pose.heading for pose in pose_tuple),
    )
    return ObservationDelta(newly_observed, ordered_visible, diagnostics)


def _legacy_build_sensor_poses(
    path_cells: tuple[CellXY, ...],
    geometry: GridGeometry,
    *,
    target_theta: float,
    step_m: float = 1.0,
) -> tuple[tuple[SensorPose, ...], ActionExecutionDiagnostics]:
    """Frozen pre-acceleration path sampler with the original repeated scan."""

    if not path_cells:
        raise ValueError("validated path must contain at least one cell")
    if not math.isfinite(step_m) or step_m <= 0.0:
        raise ValueError("path observation step must be finite and positive")
    endpoint_theta = normalize_theta(target_theta)
    points = tuple(geometry.cell_to_world_center(cell) for cell in path_cells)
    segment_lengths = tuple(
        math.hypot(right.x - left.x, right.y - left.y)
        for left, right in zip(points[:-1], points[1:], strict=True)
    )
    total_length = sum(segment_lengths)
    cumulative = [0.0]
    for length in segment_lengths:
        cumulative.append(cumulative[-1] + length)
    path_poses: list[SensorPose] = []
    distance = 0.0
    while distance < total_length - 1e-12:
        segment_index = next(
            index
            for index, end_distance in enumerate(cumulative[1:])
            if distance < end_distance - 1e-12 or index == len(segment_lengths) - 1
        )
        segment_start = points[segment_index]
        segment_end = points[segment_index + 1]
        segment_length = segment_lengths[segment_index]
        fraction = (distance - cumulative[segment_index]) / segment_length
        world = WorldXY(
            segment_start.x + fraction * (segment_end.x - segment_start.x),
            segment_start.y + fraction * (segment_end.y - segment_start.y),
        )
        tangent = math.atan2(
            segment_end.y - segment_start.y,
            segment_end.x - segment_start.x,
        )
        path_poses.append(SensorPose(world, tangent, "path_tangent"))
        distance += step_m
    endpoint_pose = SensorPose(points[-1], endpoint_theta, "endpoint_theta")
    poses = (*path_poses, endpoint_pose)
    return poses, ActionExecutionDiagnostics(
        path_sample_count=len(path_poses),
        endpoint_sample_count=1,
        path_length_m=total_length,
        observation_step_m=step_m,
    )


def _truth_map(
    geometry: GridGeometry,
    *,
    hard_obstacle: np.ndarray | None = None,
    slope_deg: np.ndarray | None = None,
    height: np.ndarray | None = None,
    traversability: np.ndarray | None = None,
) -> TruthMap:
    shape = geometry.shape
    return TruthMap(
        geometry=geometry,
        height=(
            np.arange(shape[0] * shape[1], dtype=np.float64).reshape(shape) / 1000.0
            if height is None
            else np.asarray(height, dtype=np.float64)
        ),
        hard_obstacle=(
            np.zeros(shape, dtype=bool)
            if hard_obstacle is None
            else np.asarray(hard_obstacle, dtype=bool)
        ),
        slope_deg=(
            np.zeros(shape, dtype=np.float64)
            if slope_deg is None
            else np.asarray(slope_deg, dtype=np.float64)
        ),
        traversability=(
            np.ones(shape, dtype=np.float64)
            if traversability is None
            else np.asarray(traversability, dtype=np.float64)
        ),
        provenance={
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "physical_obstacle_cells_written": False,
        },
    )


def _state_pair(
    truth: TruthMap,
    initially_observed: Iterable[CellXY] = (),
    *,
    min_clearance_m: float = 0.5215874761,
    max_slope_deg: float = 30.0,
    traversability_threshold: float = 0.5,
) -> tuple[ObservedMapState, ObservedMapState]:
    reference = ObservedMapState.empty(truth.geometry)
    actual = ObservedMapState.empty(truth.geometry)
    cells = tuple(initially_observed)
    for state in (reference, actual):
        state.reveal_cells(
            truth,
            cells,
            min_clearance_m=min_clearance_m,
            max_slope_deg=max_slope_deg,
            traversability_threshold=traversability_threshold,
        )
    return reference, actual


def _assert_state_exact(
    actual: ObservedMapState,
    expected: ObservedMapState,
    *,
    case_id: object,
) -> None:
    assert actual.geometry == expected.geometry, case_id
    assert actual.remaining_step_budget_norm == expected.remaining_step_budget_norm, case_id
    for field_name in _STATE_ARRAY_FIELDS:
        actual_array = getattr(actual, field_name)
        expected_array = getattr(expected, field_name)
        assert actual_array.dtype == expected_array.dtype, (case_id, field_name)
        assert np.array_equal(actual_array, expected_array), (case_id, field_name)


def _assert_reveal_matches_legacy(
    updater: SensorUpdater,
    truth: TruthMap,
    poses: tuple[SensorPose, ...],
    initially_observed: Iterable[CellXY] = (),
    *,
    case_id: object,
) -> ObservationDelta:
    reference_state, actual_state = _state_pair(
        truth,
        initially_observed,
        min_clearance_m=updater.min_clearance_m,
        max_slope_deg=updater.max_slope_deg,
        traversability_threshold=updater.traversability_threshold,
    )
    expected = _legacy_reveal(updater, truth, reference_state, poses)
    actual = updater.reveal(truth, actual_state, poses)

    assert actual == expected, case_id
    _assert_state_exact(actual_state, reference_state, case_id=case_id)
    return actual


def test_public_ray_cells_from_world_api_retains_axis_diagonal_and_corner_tie_behavior() -> None:
    geometry = GridGeometry(8, 8, 1.0)
    origin = geometry.cell_to_world_center(CellXY(2, 2))

    assert ray_cells_from_world(origin, 0.0, geometry, range_m=2.0) == (
        CellXY(2, 2),
        CellXY(3, 2),
        CellXY(4, 2),
    )
    assert ray_cells_from_world(origin, math.pi / 4.0, geometry, range_m=3.0) == (
        CellXY(2, 2),
        CellXY(3, 3),
        CellXY(4, 4),
    )
    assert ray_cells_from_world(origin, math.pi, geometry, range_m=20.0) == (
        CellXY(2, 2),
        CellXY(1, 2),
        CellXY(0, 2),
    )


def test_lazy_ray_iterator_matches_public_tuple_api_on_deterministic_edge_cases() -> None:
    from lunar_exploration_ppo.env import sensor_model as sensor_model_module

    iterator = getattr(sensor_model_module, "iter_ray_cells_from_world", None)
    assert callable(iterator)
    geometry = GridGeometry(8, 8, 1.0)
    centered = geometry.cell_to_world_center(CellXY(2, 2))
    cases = (
        ("axial", centered, 0.0, 2.0),
        ("diagonal", centered, math.atan2(2.0, 1.0), 3.0),
        ("corner-tie", centered, math.pi / 4.0, 3.0),
        ("map-edge", centered, math.pi, 20.0),
        ("finite-range", centered, 0.0, 0.49),
    )

    for case_id, origin, angle, range_m in cases:
        expected = _legacy_ray_cells_from_world(
            origin,
            angle,
            geometry,
            range_m=range_m,
        )
        assert tuple(iterator(origin, angle, geometry, range_m=range_m)) == expected, case_id
        assert ray_cells_from_world(origin, angle, geometry, range_m=range_m) == expected, case_id


def test_lazy_ray_iterator_defers_dda_and_can_stop_after_first_cell() -> None:
    from lunar_exploration_ppo.env import sensor_model as sensor_model_module

    iterator = getattr(sensor_model_module, "iter_ray_cells_from_world", None)
    assert callable(iterator)
    geometry = GridGeometry(128, 128, 0.5)
    origin = geometry.cell_to_world_center(CellXY(64, 64))
    in_bounds_calls = 0

    class CountingGeometry:
        width = geometry.width
        height = geometry.height
        resolution_m = geometry.resolution_m
        origin = geometry.origin

        @staticmethod
        def world_to_cell(world: WorldXY) -> CellXY:
            return geometry.world_to_cell(world)

        @staticmethod
        def in_bounds(cell: CellXY) -> bool:
            nonlocal in_bounds_calls
            in_bounds_calls += 1
            return geometry.in_bounds(cell)

    ray = iterator(origin, 0.0, CountingGeometry(), range_m=20.0)

    assert in_bounds_calls == 0
    assert next(ray) == CellXY(64, 64)
    assert in_bounds_calls == 1


def test_sensor_reveal_matches_legacy_on_deterministic_edge_cases() -> None:
    standard_geometry = GridGeometry(128, 128, 0.5)
    open_truth = _truth_map(standard_geometry)
    center = standard_geometry.cell_to_world_center(CellXY(48, 48))
    four_quadrants = tuple(
        SensorPose(center, heading, "path_tangent")
        for heading in (0.0, math.pi / 2.0, math.pi, -math.pi / 2.0)
    )
    _assert_reveal_matches_legacy(
        SensorUpdater(),
        open_truth,
        (*four_quadrants, SensorPose(center, math.pi / 4.0, "endpoint_theta")),
        case_id="standard-four-quadrants",
    )

    blocker_geometry = GridGeometry(18, 18, 1.0)
    obstacle = np.zeros(blocker_geometry.shape, dtype=bool)
    slope = np.zeros(blocker_geometry.shape, dtype=np.float64)
    obstacle[9, 3] = True  # The origin cell must not stop its own ray.
    obstacle[9, 7] = True
    slope[13, 7] = 31.0
    blocker_truth = _truth_map(
        blocker_geometry,
        hard_obstacle=obstacle,
        slope_deg=slope,
    )
    _assert_reveal_matches_legacy(
        SensorUpdater(range_m=20.0, fov_deg=0.0),
        blocker_truth,
        (
            SensorPose(
                blocker_geometry.cell_to_world_center(CellXY(3, 9)),
                0.0,
                "endpoint_theta",
            ),
            SensorPose(
                blocker_geometry.cell_to_world_center(CellXY(7, 9)),
                math.pi / 2.0,
                "path_tangent",
            ),
        ),
        case_id="origin-hard-blocker-and-later-steep-blocker",
    )

    edge_geometry = GridGeometry(16, 16, 0.5, WorldXY(-3.2, 7.1))
    edge_truth = _truth_map(edge_geometry)
    subcell_origin = WorldXY(
        edge_geometry.origin.x + 14.03 * edge_geometry.resolution_m,
        edge_geometry.origin.y + 14.71 * edge_geometry.resolution_m,
    )
    edge_pose = SensorPose(subcell_origin, math.pi / 4.0, "endpoint_theta")
    _assert_reveal_matches_legacy(
        SensorUpdater(range_m=20.0, fov_deg=90.0, ray_angle_step_deg=1.0),
        edge_truth,
        (edge_pose, edge_pose),
        initially_observed=(CellXY(14, 14), CellXY(15, 15)),
        case_id="subcell-edge-corner-tie-duplicate-pose",
    )


def test_sensor_cell_center_range_gate_is_scalar_hypot_at_exact_20m_boundary() -> None:
    from lunar_exploration_ppo.env import sensor_model as sensor_model_module

    geometry = GridGeometry(128, 128, 0.5)
    truth = _truth_map(geometry)
    origin_cell = CellXY(40, 40)
    origin = geometry.cell_to_world_center(origin_cell)
    target = CellXY(origin_cell.x + 24, origin_cell.y + 32)
    heading = math.atan2(16.0, 12.0)
    updater = SensorUpdater(range_m=20.0, fov_deg=0.0)

    delta = _assert_reveal_matches_legacy(
        updater,
        truth,
        (SensorPose(origin, heading, "endpoint_theta"),),
        case_id="exact-3-4-5-20m-boundary",
    )
    assert math.hypot(
        geometry.cell_to_world_center(target).x - origin.x,
        geometry.cell_to_world_center(target).y - origin.y,
    ) == 20.0
    assert target in delta.visible_cells

    source = textwrap.dedent(inspect.getsource(sensor_model_module.SensorUpdater.reveal))
    tree = ast.parse(source)

    def is_range_value(node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Name)
            and node.id == "range_m"
            or isinstance(node, ast.Attribute)
            and node.attr == "range_m"
        )

    scalar_hypot_gates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Call)
        and isinstance(node.left.func, ast.Attribute)
        and isinstance(node.left.func.value, ast.Name)
        and node.left.func.value.id == "math"
        and node.left.func.attr == "hypot"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Gt)
        and len(node.comparators) == 1
        and is_range_value(node.comparators[0])
    ]
    assert len(scalar_hypot_gates) == 1


def test_sensor_reveal_matches_legacy_for_128_fixed_seed_randomized_cases() -> None:
    rng = np.random.default_rng(20260722)
    sources = ("reset", "path_tangent", "endpoint_theta")
    for case_index in range(128):
        width = int(rng.integers(6, 19))
        height = int(rng.integers(6, 19))
        resolution = float(rng.choice((0.25, 0.5, 0.75, 1.0)))
        geometry = GridGeometry(
            width,
            height,
            resolution,
            WorldXY(float(rng.uniform(-9.0, -1.0)), float(rng.uniform(2.0, 11.0))),
        )
        shape = geometry.shape
        hard_obstacle = rng.random(shape) < 0.10
        slope_deg = rng.uniform(0.0, 45.0, size=shape).astype(np.float64)
        traversability = rng.uniform(0.0, 1.0, size=shape).astype(np.float64)
        height_layer = rng.normal(0.0, 2.0, size=shape).astype(np.float64)
        truth = _truth_map(
            geometry,
            hard_obstacle=hard_obstacle,
            slope_deg=slope_deg,
            height=height_layer,
            traversability=traversability,
        )
        updater = SensorUpdater(
            range_m=float(rng.choice((0.5, 1.0, 2.5, 4.0, 20.0))),
            fov_deg=float(rng.choice((0.0, 30.0, 45.0, 90.0))),
            ray_angle_step_deg=float(rng.choice((1.0, 5.0, 15.0))),
            min_clearance_m=float(rng.choice((0.0, 0.5215874761))),
            max_slope_deg=30.0,
            traversability_threshold=0.5,
        )
        pose_count = int(rng.integers(1, 4))
        poses: list[SensorPose] = []
        for pose_index in range(pose_count):
            cell_x = int(rng.integers(0, width))
            cell_y = int(rng.integers(0, height))
            fraction_x = float(rng.uniform(0.03, 0.97))
            fraction_y = float(rng.uniform(0.03, 0.97))
            world = WorldXY(
                geometry.origin.x + (cell_x + fraction_x) * resolution,
                geometry.origin.y + (cell_y + fraction_y) * resolution,
            )
            poses.append(
                SensorPose(
                    world,
                    float(rng.uniform(-4.0 * math.pi, 4.0 * math.pi)),
                    sources[(case_index + pose_index) % len(sources)],
                )
            )
        if case_index % 11 == 0:
            poses.append(poses[-1])
        initial_count = int(rng.integers(0, min(width * height, 12) + 1))
        initial_flat = rng.choice(width * height, size=initial_count, replace=False)
        initially_observed = tuple(
            CellXY(int(flat % width), int(flat // width)) for flat in initial_flat
        )
        _assert_reveal_matches_legacy(
            updater,
            truth,
            tuple(poses),
            initially_observed,
            case_id=case_index,
        )


def test_reveal_uses_fused_traversal_without_public_ray_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.env import sensor_model as sensor_model_module

    geometry = GridGeometry(32, 32, 0.5)
    truth = _truth_map(geometry)
    state = ObservedMapState.empty(geometry)
    pose = SensorPose(
        geometry.cell_to_world_center(CellXY(16, 16)),
        0.0,
        "endpoint_theta",
    )

    def forbidden_public_ray_materialization(*args: object, **kwargs: object) -> tuple[CellXY, ...]:
        raise AssertionError("reveal must use its fused DDA hot path")

    monkeypatch.setattr(
        sensor_model_module,
        "ray_cells_from_world",
        forbidden_public_ray_materialization,
    )
    delta = SensorUpdater().reveal(truth, state, (pose,))

    assert delta.visible_cells
    assert delta.diagnostics.ray_count == 91


def _serpentine_path(width: int, height: int) -> tuple[CellXY, ...]:
    cells: list[CellXY] = []
    for y in range(height):
        columns = range(width) if y % 2 == 0 else range(width - 1, -1, -1)
        cells.extend(CellXY(x, y) for x in columns)
    return tuple(cells)


@pytest.mark.parametrize(
    ("geometry", "path", "target_theta", "step_m"),
    (
        (GridGeometry(8, 8, 0.5), (CellXY(4, 2),), -math.pi / 2.0, 1.0),
        (
            GridGeometry(16, 16, 0.5),
            (CellXY(1, 1), CellXY(2, 2), CellXY(3, 2), CellXY(3, 3)),
            3.0 * math.pi,
            0.75,
        ),
        (GridGeometry(48, 48, 0.5), _serpentine_path(48, 48), 0.123, 1.0),
    ),
    ids=("single-cell", "diagonal-corners", "long-serpentine"),
)
def test_build_sensor_poses_matches_frozen_legacy_exactly_on_long_polylines(
    geometry: GridGeometry,
    path: tuple[CellXY, ...],
    target_theta: float,
    step_m: float,
) -> None:
    expected = _legacy_build_sensor_poses(
        path,
        geometry,
        target_theta=target_theta,
        step_m=step_m,
    )
    actual = build_sensor_poses(
        path,
        geometry,
        target_theta=target_theta,
        step_m=step_m,
    )

    assert actual == expected


def test_build_sensor_poses_uses_forward_only_segment_cursor() -> None:
    source = textwrap.dedent(inspect.getsource(build_sensor_poses))
    tree = ast.parse(source)
    repeated_next_scans = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "next"
    ]

    assert repeated_next_scans == []
