"""Benchmark the Stage 6 exact-semantic fused sensor hot path."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from lunar_exploration_ppo.env.map_state import ObservedMapState
from lunar_exploration_ppo.env.scenario import TruthMap
from lunar_exploration_ppo.env.sensor_model import (
    ObservationDelta,
    SensorDiagnostics,
    SensorPose,
    SensorUpdater,
)
from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry, WorldXY


EXPECTED_POSE_COUNT = 128
EXPECTED_RAY_COUNT = 11_648
EXPECTED_CELL_VISIT_COUNT = 593_536
HISTORICAL_BASELINE_MEDIAN_SECONDS = 0.9124088999815285
MINIMUM_SPEEDUP = 1.5
STATE_ARRAY_FIELDS = (
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
    """冻结优化前的公开 DDA 实现，作为独立性能基线。"""

    current = geometry.world_to_cell(origin)
    direction_x = math.cos(angle)
    direction_y = math.sin(angle)
    step_x = 1 if direction_x > 0.0 else -1 if direction_x < 0.0 else 0
    step_y = 1 if direction_y > 0.0 else -1 if direction_y < 0.0 else 0
    resolution = geometry.resolution_m

    if step_x:
        boundary_x = geometry.origin.x + (
            current.x + (1 if step_x > 0 else 0)
        ) * resolution
        t_max_x = (boundary_x - origin.x) / direction_x
        t_delta_x = resolution / abs(direction_x)
    else:
        t_max_x = math.inf
        t_delta_x = math.inf
    if step_y:
        boundary_y = geometry.origin.y + (
            current.y + (1 if step_y > 0 else 0)
        ) * resolution
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
    """冻结优化前的 reveal，实现不得调用生产 fused 路径。"""

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


def _build_workload() -> tuple[SensorUpdater, TruthMap, tuple[SensorPose, ...]]:
    geometry = GridGeometry(256, 256, 0.5)
    shape = geometry.shape
    truth = TruthMap(
        geometry=geometry,
        height=np.zeros(shape, dtype=np.float64),
        hard_obstacle=np.zeros(shape, dtype=bool),
        slope_deg=np.zeros(shape, dtype=np.float64),
        traversability=np.ones(shape, dtype=np.float64),
        provenance={"source_kind": "stage6_sensor_hotpath_benchmark/v1"},
    )
    origin = geometry.cell_to_world_center(CellXY(128, 128))
    poses = tuple(
        SensorPose(
            origin,
            0.0 if index % 2 == 0 else math.pi / 4.0,
            "path_tangent",
        )
        for index in range(EXPECTED_POSE_COUNT)
    )
    return SensorUpdater(), truth, poses


def _run_legacy(
    updater: SensorUpdater,
    truth: TruthMap,
    poses: tuple[SensorPose, ...],
) -> tuple[ObservationDelta, ObservedMapState]:
    state = ObservedMapState.empty(truth.geometry)
    return _legacy_reveal(updater, truth, state, poses), state


def _run_optimized(
    updater: SensorUpdater,
    truth: TruthMap,
    poses: tuple[SensorPose, ...],
) -> tuple[ObservationDelta, ObservedMapState]:
    state = ObservedMapState.empty(truth.geometry)
    return updater.reveal(truth, state, poses), state


def _semantic_comparison(
    legacy: tuple[ObservationDelta, ObservedMapState],
    optimized: tuple[ObservationDelta, ObservedMapState],
) -> dict[str, object]:
    legacy_delta, legacy_state = legacy
    optimized_delta, optimized_state = optimized
    delta_fields = {
        "newly_observed_cells": (
            legacy_delta.newly_observed_cells == optimized_delta.newly_observed_cells
        ),
        "visible_cells": legacy_delta.visible_cells == optimized_delta.visible_cells,
        "diagnostics": legacy_delta.diagnostics == optimized_delta.diagnostics,
    }
    state_arrays = {
        name: bool(
            np.array_equal(
                getattr(legacy_state, name),
                getattr(optimized_state, name),
                equal_nan=True,
            )
        )
        for name in STATE_ARRAY_FIELDS
    }
    remaining_budget_equal = (
        legacy_state.remaining_step_budget_norm
        == optimized_state.remaining_step_budget_norm
    )
    passed = all(delta_fields.values()) and all(state_arrays.values()) and remaining_budget_equal
    return {
        "passed": passed,
        "delta_fields": delta_fields,
        "state_arrays": state_arrays,
        "remaining_step_budget_norm": remaining_budget_equal,
        "legacy_diagnostics": asdict(legacy_delta.diagnostics),
        "optimized_diagnostics": asdict(optimized_delta.diagnostics),
    }


BenchmarkCallable = Callable[[], tuple[ObservationDelta, ObservedMapState]]


def _timed(run: BenchmarkCallable) -> float:
    gc.collect()
    started = time.perf_counter()
    run()
    return time.perf_counter() - started


def _benchmark_pair(
    legacy_run: BenchmarkCallable,
    optimized_run: BenchmarkCallable,
    *,
    warmups: int,
    repeats: int,
) -> tuple[list[float], list[float]]:
    for _ in range(warmups):
        legacy_run()
        optimized_run()

    legacy_samples: list[float] = []
    optimized_samples: list[float] = []
    for repeat_index in range(repeats):
        if repeat_index % 2 == 0:
            legacy_samples.append(_timed(legacy_run))
            optimized_samples.append(_timed(optimized_run))
        else:
            optimized_samples.append(_timed(optimized_run))
            legacy_samples.append(_timed(legacy_run))
    return legacy_samples, optimized_samples


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Explicit absolute JSON output path on the D: drive.",
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.warmups < 3:
        parser.error("--warmups must be at least 3")
    if args.repeats < 7:
        parser.error("--repeats must be at least 7")
    if not args.output.is_absolute() or args.output.drive.upper() != "D:":
        parser.error("--output must be an explicit absolute D: path")
    return args


def main() -> int:
    args = _parse_args()
    output_path = args.output.resolve()
    updater, truth, poses = _build_workload()
    legacy_run = lambda: _run_legacy(updater, truth, poses)
    optimized_run = lambda: _run_optimized(updater, truth, poses)

    legacy_result = legacy_run()
    optimized_result = optimized_run()
    semantic = _semantic_comparison(legacy_result, optimized_result)
    diagnostics = legacy_result[0].diagnostics
    workload_matches_frozen = (
        diagnostics.sample_count == EXPECTED_POSE_COUNT
        and diagnostics.ray_count == EXPECTED_RAY_COUNT
        and diagnostics.cell_visit_count == EXPECTED_CELL_VISIT_COUNT
    )

    legacy_samples: list[float] = []
    optimized_samples: list[float] = []
    if semantic["passed"] and workload_matches_frozen:
        legacy_samples, optimized_samples = _benchmark_pair(
            legacy_run,
            optimized_run,
            warmups=args.warmups,
            repeats=args.repeats,
        )

    legacy_median = statistics.median(legacy_samples) if legacy_samples else None
    optimized_median = statistics.median(optimized_samples) if optimized_samples else None
    speedup = (
        legacy_median / optimized_median
        if legacy_median is not None and optimized_median is not None
        else None
    )
    acceptance_passed = bool(
        semantic["passed"]
        and workload_matches_frozen
        and speedup is not None
        and speedup >= MINIMUM_SPEEDUP
    )
    payload = {
        "schema": "stage6_sensor_hotpath_benchmark/v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "numpy_version": np.__version__,
        },
        "workload": {
            "grid_width": truth.geometry.width,
            "grid_height": truth.geometry.height,
            "resolution_m": truth.geometry.resolution_m,
            "pose_count": diagnostics.sample_count,
            "ray_count": diagnostics.ray_count,
            "cell_visit_count": diagnostics.cell_visit_count,
            "expected_pose_count": EXPECTED_POSE_COUNT,
            "expected_ray_count": EXPECTED_RAY_COUNT,
            "expected_cell_visit_count": EXPECTED_CELL_VISIT_COUNT,
            "matches_frozen_workload": workload_matches_frozen,
        },
        "settings": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "minimum_speedup": MINIMUM_SPEEDUP,
            "historical_baseline_median_seconds": HISTORICAL_BASELINE_MEDIAN_SECONDS,
        },
        "semantic_comparison": semantic,
        "timings": {
            "legacy_seconds": legacy_samples,
            "optimized_seconds": optimized_samples,
            "legacy_median_seconds": legacy_median,
            "optimized_median_seconds": optimized_median,
            "median_speedup": speedup,
        },
        "acceptance": {
            "passed": acceptance_passed,
            "reason": (
                "exact semantics and median speedup gate passed"
                if acceptance_passed
                else "semantic, workload, or median speedup gate failed"
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if acceptance_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
