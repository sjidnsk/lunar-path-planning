"""Convert validated A* cell paths into deterministic sensor samples."""

from __future__ import annotations

import math
from dataclasses import dataclass

from lunar_exploration_ppo.env.sensor_model import SensorPose
from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry, WorldXY, normalize_theta


@dataclass(frozen=True, slots=True)
class ActionExecutionDiagnostics:
    path_sample_count: int
    endpoint_sample_count: int
    path_length_m: float
    observation_step_m: float


def build_sensor_poses(
    path_cells: tuple[CellXY, ...],
    geometry: GridGeometry,
    *,
    target_theta: float,
    step_m: float = 1.0,
) -> tuple[tuple[SensorPose, ...], ActionExecutionDiagnostics]:
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
    segment_index = 0
    last_segment_index = len(segment_lengths) - 1
    while distance < total_length - 1e-12:
        while segment_index < last_segment_index and not (
            distance < cumulative[segment_index + 1] - 1e-12
        ):
            segment_index += 1
        segment_start = points[segment_index]
        segment_end = points[segment_index + 1]
        segment_length = segment_lengths[segment_index]
        fraction = (distance - cumulative[segment_index]) / segment_length
        world = WorldXY(
            segment_start.x + fraction * (segment_end.x - segment_start.x),
            segment_start.y + fraction * (segment_end.y - segment_start.y),
        )
        tangent = math.atan2(segment_end.y - segment_start.y, segment_end.x - segment_start.x)
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
