"""从 G1 正式轨迹重建学术风格探索过程动画。

本脚本只读取正式结果中已经保存的决策、规划路径和覆盖增量，不重新执行
PPO，不改写任何门槛结果。动画中的整数步覆盖值会逐步与正式记录核对。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
import numpy as np

import xunce_artifact_io as artifact_io


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lunar_exploration_ppo.configs.stage6 import (  # noqa: E402
    SafetyContract,
    parse_stage6_config_bytes,
)
from lunar_exploration_ppo.env.action_execution import build_sensor_poses  # noqa: E402
from lunar_exploration_ppo.env.map_state import ObservedMapState  # noqa: E402
from lunar_exploration_ppo.env.scenario_catalog import StandardScenarioFactory  # noqa: E402
from lunar_exploration_ppo.env.sensor_model import SensorPose, SensorUpdater  # noqa: E402
from lunar_exploration_ppo.env.standard_training import (  # noqa: E402
    StandardEnvSettings,
    build_standard_catalog,
)
from lunar_exploration_ppo.utils.geometry import CellXY  # noqa: E402


RENDERER_ID = "render_xunce_mid_dual_g1_animation/v1"
AUDIT_SCHEMA = "xunce-mid-dual-g1-animation-audit/v1"
SOURCE_DATA_SCHEMA = "xunce-mid-dual-g1-animation-source-data/v1"
SELECTION_RULE = (
    "minimum robust normalized L1 distance to split medians of final coverage "
    "and steps; ties by episode_index"
)
FORMAL_RESULT_RELATIVE_PATHS = {
    "test_q24": Path("phases") / "p03" / "a01" / "results.jsonl",
    "unseen24": Path("phases") / "p04" / "a01" / "results.jsonl",
}
DISPLAY_LABELS = {
    "test_q24": "Test-Q24",
    "unseen24": "Unseen-24",
}
PANEL_LABELS = {
    "test_q24": "a",
    "unseen24": "b",
}
SPLIT_COLORS = {
    "test_q24": "#6F7FB3",
    "unseen24": "#C78FA6",
}
UNKNOWN_COLOR = "#17212B"
OBSERVED_COLOR = "#AEB9C2"
NEWLY_OBSERVED_COLOR = "#39B9AE"
OBSTACLE_COLOR = "#542A2A"
EXECUTED_PATH_COLOR = "#08A6C4"
PLANNED_PATH_COLOR = "#D65A8E"
ROBOT_COLOR = "#E98B2A"
TARGET_COLOR = "#F2C14E"
INK_COLOR = "#24272B"
GRID_COLOR = "#D9DEE3"
COVERAGE_THRESHOLD_PERCENT = 80.0
DEFAULT_FPS = 12
DEFAULT_SUBFRAMES = 4


class AnimationDataError(RuntimeError):
    """正式轨迹、场景身份或覆盖复算不一致。"""


@dataclass(frozen=True, slots=True)
class FormalEpisode:
    split: str
    result_path: Path
    coverage_row: Mapping[str, Any]
    decisions: tuple[Mapping[str, Any], ...]
    planners: tuple[Mapping[str, Any], ...]
    selection_audit: Mapping[str, Any]
    selected_rows_sha256: str


@dataclass(frozen=True, slots=True)
class SceneFrame:
    observed_mask: np.ndarray
    newly_observed_mask: np.ndarray
    robot_xy: tuple[float, float]
    robot_theta: float
    executed_xy: np.ndarray
    planned_xy: np.ndarray
    target_xy: tuple[float, float] | None
    progress_step: float
    formal_step: int
    coverage_count: int
    coverage_percent: float
    cumulative_path_m: float
    candidate_count: int | None


@dataclass(frozen=True, slots=True)
class SceneReplay:
    split: str
    display_label: str
    short_scenario_id: str
    episode_id: str
    scenario_id: str
    scenario_hash: str
    denominator_count: int
    denominator_sha256: str
    final_coverage_percent: float
    steps_executed: int
    safety_violation_count: int
    extent: tuple[float, float, float, float]
    terrain_rgb: np.ndarray
    hard_obstacle: np.ndarray
    frames: tuple[SceneFrame, ...]
    source_result_sha256: str
    selected_rows_sha256: str
    selection_audit: Mapping[str, Any]
    verification: Mapping[str, Any]


@dataclass(slots=True)
class SceneArtists:
    image: Any
    executed_line: Line2D
    planned_line: Line2D
    target_marker: Line2D
    robot_arrow: FancyArrowPatch
    hud: Any
    coverage_line: Line2D
    coverage_point: Line2D
    coverage_text: Any


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise AnimationDataError(reason)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(artifact_io.read_bytes(path))


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _median_absolute_deviation(values: Sequence[float], median: float) -> float:
    return float(statistics.median(abs(float(value) - median) for value in values))


def _select_representative_episode(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    result_path: Path,
) -> FormalEpisode:
    coverage_rows = [
        row for row in rows if row.get("row_kind") == "coverage_episode"
    ]
    _require(len(coverage_rows) == 24, f"{split}:expected_24_coverage_rows")
    coverages = [float(row["coverage"]) for row in coverage_rows]
    steps = [float(row["steps_executed"]) for row in coverage_rows]
    coverage_median = float(statistics.median(coverages))
    steps_median = float(statistics.median(steps))
    coverage_mad = _median_absolute_deviation(coverages, coverage_median)
    steps_mad = _median_absolute_deviation(steps, steps_median)
    coverage_scale = max(coverage_mad, 1.0e-12)
    steps_scale = max(steps_mad, 1.0)

    def rank_key(row: Mapping[str, Any]) -> tuple[float, int]:
        distance = (
            abs(float(row["coverage"]) - coverage_median) / coverage_scale
            + abs(float(row["steps_executed"]) - steps_median) / steps_scale
        )
        return distance, int(row["episode_index"])

    selected = min(coverage_rows, key=rank_key)
    episode_id = str(selected["episode_id"])
    selected_rows = [row for row in rows if row.get("episode_id") == episode_id]
    decisions = sorted(
        (row for row in selected_rows if row.get("row_kind") == "decision"),
        key=lambda row: int(row["trace"]["step_index"]),
    )
    planners = sorted(
        (row for row in selected_rows if row.get("row_kind") == "planner_call"),
        key=lambda row: int(row["trace"]["step_index"]),
    )
    expected_steps = int(selected["steps_executed"])
    _require(len(decisions) == expected_steps, f"{split}:decision_count_mismatch")
    _require(len(planners) == expected_steps, f"{split}:planner_count_mismatch")
    for step_index, (decision, planner) in enumerate(
        zip(decisions, planners, strict=True)
    ):
        decision_trace = decision["trace"]
        planner_trace = planner["trace"]
        _require(
            int(decision_trace["step_index"]) == step_index
            and int(planner_trace["step_index"]) == step_index,
            f"{split}:noncontiguous_step_{step_index}",
        )
        _require(
            decision_trace["join_key"] == planner_trace["join_key"],
            f"{split}:join_key_mismatch_{step_index}",
        )
        _require(
            int(decision_trace["selected_index"])
            == int(planner_trace["selected_candidate_index"]),
            f"{split}:selected_candidate_mismatch_{step_index}",
        )
    selection_audit = {
        "rule": SELECTION_RULE,
        "universe_episode_count": len(coverage_rows),
        "coverage_median_percent": coverage_median * 100.0,
        "steps_median": steps_median,
        "coverage_mad_percent": coverage_mad * 100.0,
        "steps_mad": steps_mad,
        "selected_distance": rank_key(selected)[0],
        "selected_episode_index": int(selected["episode_index"]),
        "selected_episode_id": episode_id,
    }
    return FormalEpisode(
        split=split,
        result_path=result_path,
        coverage_row=selected,
        decisions=tuple(decisions),
        planners=tuple(planners),
        selection_audit=selection_audit,
        selected_rows_sha256=_sha256_bytes(_canonical_json_bytes(selected_rows)),
    )


def _load_formal_episode(formal_root: Path, *, split: str) -> FormalEpisode:
    result_path = formal_root / FORMAL_RESULT_RELATIVE_PATHS[split]
    _require(artifact_io.path_is_file(result_path), f"{split}:missing_formal_results")
    rows = artifact_io.read_jsonl(result_path)
    return _select_representative_episode(
        rows,
        split=split,
        result_path=result_path,
    )


def _load_reconstruction_index(package_root: Path) -> dict[str, Mapping[str, Any]]:
    index_path = package_root / "reconstruction-index.jsonl"
    _require(
        artifact_io.path_is_file(index_path),
        "missing_scenario_reconstruction_index",
    )
    rows = artifact_io.read_jsonl(index_path)
    by_scenario = {str(row["scenario_id"]): row for row in rows}
    _require(len(by_scenario) == len(rows), "duplicate_reconstruction_scenario_id")
    return by_scenario


def _load_coverable_mask(
    index_row: Mapping[str, Any],
    *,
    expected_denominator_sha256: str,
) -> np.ndarray:
    mask_path = Path(str(index_row["mask_path"]))
    payload = artifact_io.read_bytes(mask_path)
    _require(
        _sha256_bytes(payload) == index_row["mask_file_sha256"],
        "coverable_mask_file_sha256_mismatch",
    )
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        _require(
            "coverable_mask" in archive.files,
            "coverable_mask_array_missing",
        )
        mask = np.asarray(archive["coverable_mask"], dtype=bool)
    content_sha256 = _sha256_bytes(np.ascontiguousarray(mask).tobytes())
    _require(
        content_sha256 == expected_denominator_sha256,
        "coverable_mask_content_sha256_mismatch",
    )
    return mask


def _cell_path_to_world(path_cells: Sequence[CellXY], geometry: Any) -> np.ndarray:
    if not path_cells:
        return np.empty((0, 2), dtype=np.float64)
    points = [geometry.cell_to_world_center(cell) for cell in path_cells]
    return np.asarray([(point.x, point.y) for point in points], dtype=np.float64)


def _terrain_rgb(height: np.ndarray) -> np.ndarray:
    finite = np.asarray(height[np.isfinite(height)], dtype=np.float64)
    _require(finite.size > 0, "scenario_height_has_no_finite_values")
    lower, upper = np.percentile(finite, (2.0, 98.0))
    if not math.isfinite(float(lower)) or not math.isfinite(float(upper)) or upper <= lower:
        lower = float(np.min(finite))
        upper = float(np.max(finite))
    scale = max(float(upper - lower), 1.0e-12)
    normalized = np.clip((height - lower) / scale, 0.0, 1.0)
    gray = 0.24 + 0.64 * normalized
    return np.stack((gray * 0.92, gray * 0.97, gray), axis=-1)


def _frame_coverage_count(state: ObservedMapState, coverable_mask: np.ndarray) -> int:
    return int(np.count_nonzero(state.observed_mask & coverable_mask))


def _split_pose_batch(
    poses: Sequence[SensorPose],
    *,
    part_index: int,
    part_count: int,
) -> tuple[SensorPose, ...]:
    start = math.ceil(len(poses) * part_index / part_count)
    stop = math.ceil(len(poses) * (part_index + 1) / part_count)
    return tuple(poses[start:stop])


def _reconstruct_scene(
    episode: FormalEpisode,
    *,
    catalog: Any,
    scenario_factory: StandardScenarioFactory,
    reconstruction_index: Mapping[str, Mapping[str, Any]],
    stage6_config: Any,
    formal_config_sha256: str,
    subframes_per_step: int,
) -> SceneReplay:
    coverage = episode.coverage_row
    scenario_id = str(coverage["scenario_id"])
    _require(
        scenario_id.endswith("/standard-proxy/v1"),
        f"{episode.split}:unexpected_scenario_id",
    )
    base_scenario_id = scenario_id.removesuffix("/standard-proxy/v1")
    matching_records = [
        record for record in catalog.records if record.scenario_id == base_scenario_id
    ]
    _require(
        len(matching_records) == 1,
        f"{episode.split}:catalog_record_not_unique",
    )
    bundle = scenario_factory.build(matching_records[0])
    index_row = reconstruction_index.get(scenario_id)
    _require(index_row is not None, f"{episode.split}:missing_reconstruction_row")
    scenario_hash = str(index_row["scenario_hash"])
    _require(
        bundle.scenario_hash == scenario_hash,
        f"{episode.split}:scenario_hash_mismatch",
    )
    coverable_mask = _load_coverable_mask(
        index_row,
        expected_denominator_sha256=str(coverage["denominator_sha256"]),
    )
    _require(
        coverable_mask.shape == bundle.truth.geometry.shape,
        f"{episode.split}:coverable_mask_shape_mismatch",
    )
    denominator_count = int(np.count_nonzero(coverable_mask))
    _require(
        denominator_count == int(coverage["denominator_cell_count"]),
        f"{episode.split}:denominator_count_mismatch",
    )

    safety_contract = SafetyContract.from_stage6_config(stage6_config)
    settings = StandardEnvSettings(
        scenario_key=base_scenario_id,
        safety_contract=safety_contract,
        config_sha256=formal_config_sha256,
    )
    sensor = SensorUpdater(
        range_m=settings.sensor_range_m,
        fov_deg=settings.sensor_fov_deg,
        ray_angle_step_deg=settings.sensor_ray_angle_step_deg,
        min_clearance_m=settings.min_clearance_m,
        max_slope_deg=settings.max_traversable_slope_deg,
        traversability_threshold=settings.traversability_threshold,
    )
    local_sensor = SensorUpdater(
        range_m=settings.reset_local_safety_scan_range_m,
        fov_deg=settings.reset_local_safety_scan_fov_deg,
        ray_angle_step_deg=settings.reset_local_safety_scan_ray_angle_step_deg,
        min_clearance_m=settings.min_clearance_m,
        max_slope_deg=settings.max_traversable_slope_deg,
        traversability_threshold=settings.traversability_threshold,
    )
    geometry = bundle.truth.geometry
    state = ObservedMapState.empty(geometry)
    start_world = geometry.cell_to_world_center(bundle.start_pose.cell)
    local_sensor.reveal(
        bundle.truth,
        state,
        (
            SensorPose(
                start_world,
                bundle.start_pose.theta,
                "reset_local_safety",
            ),
        ),
    )
    sensor.reveal(
        bundle.truth,
        state,
        (SensorPose(start_world, bundle.start_pose.theta, "reset"),),
    )
    initial_count = _frame_coverage_count(state, coverable_mask)
    _require(
        initial_count == int(coverage["initial_covered_cell_count"]),
        f"{episode.split}:initial_coverage_mismatch",
    )

    initial_frame = SceneFrame(
        observed_mask=state.observed_mask.copy(),
        newly_observed_mask=state.observed_mask.copy(),
        robot_xy=(start_world.x, start_world.y),
        robot_theta=bundle.start_pose.theta,
        executed_xy=np.asarray([(start_world.x, start_world.y)], dtype=np.float64),
        planned_xy=np.empty((0, 2), dtype=np.float64),
        target_xy=None,
        progress_step=0.0,
        formal_step=0,
        coverage_count=initial_count,
        coverage_percent=100.0 * initial_count / denominator_count,
        cumulative_path_m=0.0,
        candidate_count=None,
    )
    frames: list[SceneFrame] = [initial_frame]
    executed_points: list[tuple[float, float]] = [(start_world.x, start_world.y)]
    current_robot_xy = (start_world.x, start_world.y)
    current_robot_theta = bundle.start_pose.theta
    previous_step_count = initial_count
    previous_cumulative_path_m = 0.0
    verified_steps = 0

    for step_index, (decision_row, planner_row) in enumerate(
        zip(episode.decisions, episode.planners, strict=True)
    ):
        decision = decision_row["trace"]
        trace = planner_row["trace"]
        raw_path = trace["planned_path_cells"]
        _require(
            isinstance(raw_path, list) and raw_path,
            f"{episode.split}:empty_planned_path_{step_index}",
        )
        path_cells = tuple(CellXY(int(cell[0]), int(cell[1])) for cell in raw_path)
        selected_theta = float(trace["selected_theta"])
        poses, execution = build_sensor_poses(
            path_cells,
            geometry,
            target_theta=selected_theta,
            step_m=settings.path_observation_step_m,
        )
        _require(
            math.isclose(
                execution.path_length_m,
                float(trace["path_length_m"]),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ),
            f"{episode.split}:path_length_mismatch_{step_index}",
        )
        planned_xy = _cell_path_to_world(path_cells, geometry)
        target_world = geometry.cell_to_world_center(
            CellXY(
                int(trace["selected_candidate_cell_xy"][0]),
                int(trace["selected_candidate_cell_xy"][1]),
            )
        )
        candidate_count = int(decision["candidate_mask"]["valid_count"])

        for part_index in range(subframes_per_step):
            observed_before = state.observed_mask.copy()
            pose_batch = _split_pose_batch(
                poses,
                part_index=part_index,
                part_count=subframes_per_step,
            )
            if pose_batch:
                sensor.reveal(bundle.truth, state, pose_batch)
                for pose in pose_batch:
                    executed_points.append((pose.world_xy.x, pose.world_xy.y))
                current_robot_xy = (
                    pose_batch[-1].world_xy.x,
                    pose_batch[-1].world_xy.y,
                )
                current_robot_theta = pose_batch[-1].heading
            newly_observed = state.observed_mask & ~observed_before
            coverage_count = _frame_coverage_count(state, coverable_mask)
            fraction = float(part_index + 1) / float(subframes_per_step)
            cumulative_path_m = (
                previous_cumulative_path_m
                + float(trace["path_length_m"]) * fraction
            )
            if part_index == subframes_per_step - 1:
                cumulative_path_m = float(trace["cumulative_path_length_m"])
            frames.append(
                SceneFrame(
                    observed_mask=state.observed_mask.copy(),
                    newly_observed_mask=newly_observed.copy(),
                    robot_xy=current_robot_xy,
                    robot_theta=current_robot_theta,
                    executed_xy=np.asarray(executed_points, dtype=np.float64),
                    planned_xy=planned_xy.copy(),
                    target_xy=(target_world.x, target_world.y),
                    progress_step=step_index + fraction,
                    formal_step=step_index + 1,
                    coverage_count=coverage_count,
                    coverage_percent=100.0 * coverage_count / denominator_count,
                    cumulative_path_m=cumulative_path_m,
                    candidate_count=candidate_count,
                )
            )

        actual_count = _frame_coverage_count(state, coverable_mask)
        actual_gain = actual_count - previous_step_count
        expected_gain = int(trace["coverage_gain_cells"])
        expected_rate = float(trace["coverage_rate"])
        _require(
            actual_gain == expected_gain,
            f"{episode.split}:coverage_gain_mismatch_{step_index}",
        )
        _require(
            math.isclose(
                actual_count / denominator_count,
                expected_rate,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            f"{episode.split}:coverage_rate_mismatch_{step_index}",
        )
        _require(
            bool(trace["safety_violation"]) is False,
            f"{episode.split}:unexpected_safety_violation_{step_index}",
        )
        previous_step_count = actual_count
        previous_cumulative_path_m = float(trace["cumulative_path_length_m"])
        verified_steps += 1

    final_count = _frame_coverage_count(state, coverable_mask)
    _require(
        final_count == int(coverage["final_covered_cell_count"]),
        f"{episode.split}:final_coverage_count_mismatch",
    )
    _require(
        math.isclose(
            final_count / denominator_count,
            float(coverage["coverage"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        f"{episode.split}:final_coverage_rate_mismatch",
    )
    _require(
        verified_steps == int(coverage["steps_executed"]),
        f"{episode.split}:verified_step_count_mismatch",
    )
    short_scenario_id = base_scenario_id.split("/")[-1]
    resolution = float(geometry.resolution_m)
    extent = (
        float(geometry.origin.x),
        float(geometry.origin.x + geometry.width * resolution),
        float(geometry.origin.y),
        float(geometry.origin.y + geometry.height * resolution),
    )
    verification = {
        "scenario_hash_match": True,
        "denominator_sha256_match": True,
        "denominator_count_match": True,
        "initial_coverage_exact": True,
        "per_step_coverage_exact_count": verified_steps,
        "per_step_coverage_expected_count": int(coverage["steps_executed"]),
        "final_coverage_exact": True,
        "path_length_exact_count": verified_steps,
        "join_key_exact_count": verified_steps,
        "safety_violation_count": int(coverage["safety_violation_count"]),
    }
    return SceneReplay(
        split=episode.split,
        display_label=DISPLAY_LABELS[episode.split],
        short_scenario_id=short_scenario_id,
        episode_id=str(coverage["episode_id"]),
        scenario_id=scenario_id,
        scenario_hash=scenario_hash,
        denominator_count=denominator_count,
        denominator_sha256=str(coverage["denominator_sha256"]),
        final_coverage_percent=100.0 * float(coverage["coverage"]),
        steps_executed=int(coverage["steps_executed"]),
        safety_violation_count=int(coverage["safety_violation_count"]),
        extent=extent,
        terrain_rgb=_terrain_rgb(bundle.truth.height),
        hard_obstacle=np.asarray(bundle.truth.hard_obstacle, dtype=bool),
        frames=tuple(frames),
        source_result_sha256=_sha256_file(episode.result_path),
        selected_rows_sha256=episode.selected_rows_sha256,
        selection_audit=episode.selection_audit,
        verification=verification,
    )


def _composite_map(scene: SceneReplay, frame: SceneFrame) -> np.ndarray:
    unknown = np.asarray(mcolors.to_rgb(UNKNOWN_COLOR), dtype=np.float64)
    image = np.empty((*frame.observed_mask.shape, 3), dtype=np.float64)
    image[...] = unknown
    image[frame.observed_mask] = scene.terrain_rgb[frame.observed_mask]
    observed_obstacle = frame.observed_mask & scene.hard_obstacle
    image[observed_obstacle] = np.asarray(
        mcolors.to_rgb(OBSTACLE_COLOR),
        dtype=np.float64,
    )
    newly = frame.newly_observed_mask
    if np.any(newly):
        highlight = np.asarray(
            mcolors.to_rgb(NEWLY_OBSERVED_COLOR),
            dtype=np.float64,
        )
        image[newly] = 0.42 * image[newly] + 0.58 * highlight
    return np.clip(image, 0.0, 1.0)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": 10.0,
            "axes.labelsize": 10.5,
            "axes.titlesize": 12.0,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _scene_title(scene: SceneReplay) -> str:
    return (
        f"{scene.display_label} representative · {scene.short_scenario_id} · "
        f"formal {scene.final_coverage_percent:.2f}% / {scene.steps_executed} steps"
    )


def _build_figure(
    scenes: Sequence[SceneReplay],
) -> tuple[plt.Figure, tuple[SceneArtists, ...]]:
    _configure_style()
    fig = plt.figure(figsize=(16.0, 9.0), dpi=120)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(4.15, 1.2),
        hspace=0.20,
        wspace=0.14,
        left=0.055,
        right=0.985,
        bottom=0.09,
        top=0.82,
    )
    map_axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]))
    coverage_axes = (fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]))

    fig.suptitle(
        "G1 autonomous exploration process",
        x=0.5,
        y=0.975,
        fontsize=18.0,
        fontweight="bold",
        color=INK_COLOR,
    )
    fig.text(
        0.5,
        0.938,
        "Formal-trace reconstruction · representative episodes · "
        "not used for gate recomputation",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#59636D",
    )
    legend_handles = (
        Patch(facecolor=UNKNOWN_COLOR, edgecolor="none", label="Unknown"),
        Patch(facecolor=OBSERVED_COLOR, edgecolor="none", label="Observed terrain"),
        Patch(
            facecolor=NEWLY_OBSERVED_COLOR,
            edgecolor="none",
            label="Newly observed",
        ),
        Line2D(
            (0,),
            (0,),
            color=EXECUTED_PATH_COLOR,
            linewidth=2.2,
            label="Executed trajectory",
        ),
        Line2D(
            (0,),
            (0,),
            color=PLANNED_PATH_COLOR,
            linewidth=1.8,
            linestyle=(0, (4, 2)),
            label="Active planned path",
        ),
        Line2D(
            (0,),
            (0,),
            color=ROBOT_COLOR,
            marker=">",
            linestyle="none",
            markersize=8,
            label="Platform pose",
        ),
        Line2D(
            (0,),
            (0,),
            color=TARGET_COLOR,
            marker="*",
            markeredgecolor=INK_COLOR,
            linestyle="none",
            markersize=10,
            label="Selected target",
        ),
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.910),
        ncol=7,
        columnspacing=1.35,
        handlelength=2.3,
    )
    fig.text(
        0.5,
        0.025,
        "Dark cells remain unobserved; terrain values are shown only after "
        "sensor revelation. Integer-step coverage exactly matches the formal trace.",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#59636D",
    )

    artists: list[SceneArtists] = []
    for scene, map_ax, coverage_ax in zip(
        scenes,
        map_axes,
        coverage_axes,
        strict=True,
    ):
        initial = scene.frames[0]
        map_image = map_ax.imshow(
            _composite_map(scene, initial),
            origin="lower",
            extent=scene.extent,
            interpolation="nearest",
            zorder=1,
        )
        executed_line, = map_ax.plot(
            initial.executed_xy[:, 0],
            initial.executed_xy[:, 1],
            color=EXECUTED_PATH_COLOR,
            linewidth=2.2,
            solid_capstyle="round",
            alpha=0.96,
            zorder=4,
        )
        planned_line, = map_ax.plot(
            (),
            (),
            color=PLANNED_PATH_COLOR,
            linewidth=1.7,
            linestyle=(0, (4, 2)),
            alpha=0.95,
            zorder=5,
        )
        target_marker, = map_ax.plot(
            (),
            (),
            marker="*",
            markersize=12.0,
            markerfacecolor=TARGET_COLOR,
            markeredgecolor=INK_COLOR,
            markeredgewidth=0.8,
            linestyle="none",
            zorder=7,
        )
        arrow_length = 4.2
        arrow = FancyArrowPatch(
            initial.robot_xy,
            (
                initial.robot_xy[0] + arrow_length * math.cos(initial.robot_theta),
                initial.robot_xy[1] + arrow_length * math.sin(initial.robot_theta),
            ),
            arrowstyle="-|>",
            mutation_scale=17.0,
            linewidth=2.0,
            edgecolor="white",
            facecolor=ROBOT_COLOR,
            zorder=8,
        )
        map_ax.add_patch(arrow)
        hud = map_ax.set_title(
            "",
            loc="left",
            pad=10.0,
            ha="left",
            va="bottom",
            fontsize=10.5,
            color=INK_COLOR,
        )
        map_ax.set_xlabel("East (m)")
        map_ax.set_ylabel("North (m)")
        map_ax.set_aspect("equal", adjustable="box")
        map_ax.set_xlim(scene.extent[0], scene.extent[1])
        map_ax.set_ylim(scene.extent[2], scene.extent[3])
        map_ax.tick_params(direction="out", length=3.5, width=0.8)
        map_ax.text(
            -0.075,
            1.035,
            PANEL_LABELS[scene.split],
            transform=map_ax.transAxes,
            fontsize=17.0,
            fontweight="bold",
            ha="left",
            va="bottom",
            color=INK_COLOR,
        )

        coverage_ax.axhline(
            COVERAGE_THRESHOLD_PERCENT,
            color="#555B61",
            linewidth=1.1,
            linestyle=(0, (5, 3)),
            zorder=1,
        )
        coverage_ax.text(
            scene.steps_executed,
            COVERAGE_THRESHOLD_PERCENT + 2.0,
            "80% threshold",
            ha="right",
            va="bottom",
            fontsize=8.7,
            color="#555B61",
        )
        coverage_line, = coverage_ax.plot(
            (0.0,),
            (initial.coverage_percent,),
            color=SPLIT_COLORS[scene.split],
            linewidth=2.2,
            zorder=3,
        )
        coverage_point, = coverage_ax.plot(
            (0.0,),
            (initial.coverage_percent,),
            marker="o",
            markersize=6.5,
            markerfacecolor=SPLIT_COLORS[scene.split],
            markeredgecolor="white",
            markeredgewidth=0.8,
            linestyle="none",
            zorder=4,
        )
        coverage_text = coverage_ax.text(
            0.99,
            0.10,
            "",
            transform=coverage_ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.5,
            fontweight="semibold",
            color=SPLIT_COLORS[scene.split],
        )
        coverage_ax.set_xlim(0.0, float(scene.steps_executed))
        coverage_ax.set_ylim(0.0, 102.0)
        coverage_ax.set_xticks((0, 10, 20, 30, 40))
        coverage_ax.set_yticks((0, 20, 40, 60, 80, 100))
        coverage_ax.set_xlabel("Formal decision step")
        coverage_ax.set_ylabel("Coverable area observed (%)")
        coverage_ax.grid(
            axis="y",
            color=GRID_COLOR,
            linewidth=0.7,
            alpha=0.85,
            zorder=0,
        )
        coverage_ax.tick_params(direction="out", length=3.5, width=0.8)
        artists.append(
            SceneArtists(
                image=map_image,
                executed_line=executed_line,
                planned_line=planned_line,
                target_marker=target_marker,
                robot_arrow=arrow,
                hud=hud,
                coverage_line=coverage_line,
                coverage_point=coverage_point,
                coverage_text=coverage_text,
            )
        )
    return fig, tuple(artists)


def _update_scene_artists(
    scene: SceneReplay,
    artists: SceneArtists,
    *,
    replay_index: int,
) -> None:
    frame = scene.frames[replay_index]
    artists.image.set_data(_composite_map(scene, frame))
    artists.executed_line.set_data(
        frame.executed_xy[:, 0],
        frame.executed_xy[:, 1],
    )
    if frame.planned_xy.size:
        artists.planned_line.set_data(
            frame.planned_xy[:, 0],
            frame.planned_xy[:, 1],
        )
    else:
        artists.planned_line.set_data((), ())
    if frame.target_xy is None:
        artists.target_marker.set_data((), ())
    else:
        artists.target_marker.set_data(
            (frame.target_xy[0],),
            (frame.target_xy[1],),
        )
    arrow_length = 4.2
    artists.robot_arrow.set_positions(
        frame.robot_xy,
        (
            frame.robot_xy[0] + arrow_length * math.cos(frame.robot_theta),
            frame.robot_xy[1] + arrow_length * math.sin(frame.robot_theta),
        ),
    )
    candidate_text = "—" if frame.candidate_count is None else str(frame.candidate_count)
    artists.hud.set_text(
        f"{_scene_title(scene)}\n"
        f"Step {frame.formal_step:02d}/{scene.steps_executed:02d} · "
        f"current {frame.coverage_percent:.2f}% · "
        f"path {frame.cumulative_path_m:.1f} m · "
        f"candidates {candidate_text} · "
        f"safety {scene.safety_violation_count}"
    )
    progress = np.asarray(
        [value.progress_step for value in scene.frames[: replay_index + 1]],
        dtype=np.float64,
    )
    coverage = np.asarray(
        [value.coverage_percent for value in scene.frames[: replay_index + 1]],
        dtype=np.float64,
    )
    artists.coverage_line.set_data(progress, coverage)
    artists.coverage_point.set_data(
        (frame.progress_step,),
        (frame.coverage_percent,),
    )
    artists.coverage_text.set_text(
        f"{frame.coverage_percent:.2f}%  ·  "
        f"{frame.coverage_count:,}/{scene.denominator_count:,} cells"
    )


def _write_source_data(
    scenes: Sequence[SceneReplay],
    *,
    subframes_per_step: int,
    output_path: Path,
) -> None:
    handle = io.StringIO()
    fieldnames = (
        "schema_version",
        "split",
        "episode_id",
        "scenario_id",
        "formal_step",
        "coverage_count",
        "denominator_count",
        "coverage_percent",
        "cumulative_path_m",
        "candidate_count",
        "robot_east_m",
        "robot_north_m",
        "robot_heading_rad",
        "selected_target_east_m",
        "selected_target_north_m",
    )
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for scene in scenes:
        for formal_step in range(scene.steps_executed + 1):
            frame = scene.frames[formal_step * subframes_per_step]
            writer.writerow(
                {
                    "schema_version": SOURCE_DATA_SCHEMA,
                    "split": scene.split,
                    "episode_id": scene.episode_id,
                    "scenario_id": scene.scenario_id,
                    "formal_step": formal_step,
                    "coverage_count": frame.coverage_count,
                    "denominator_count": scene.denominator_count,
                    "coverage_percent": f"{frame.coverage_percent:.12f}",
                    "cumulative_path_m": f"{frame.cumulative_path_m:.12f}",
                    "candidate_count": (
                        "" if frame.candidate_count is None else frame.candidate_count
                    ),
                    "robot_east_m": f"{frame.robot_xy[0]:.12f}",
                    "robot_north_m": f"{frame.robot_xy[1]:.12f}",
                    "robot_heading_rad": f"{frame.robot_theta:.12f}",
                    "selected_target_east_m": (
                        "" if frame.target_xy is None else f"{frame.target_xy[0]:.12f}"
                    ),
                    "selected_target_north_m": (
                        "" if frame.target_xy is None else f"{frame.target_xy[1]:.12f}"
                    ),
                }
            )
    artifact_io.write_text(output_path, handle.getvalue())


def _render_outputs(
    scenes: Sequence[SceneReplay],
    *,
    output_root: Path,
    ffmpeg_path: Path,
    fps: int,
    subframes_per_step: int,
) -> dict[str, Path]:
    _require(len(scenes) == 2, "expected_two_g1_scenes")
    frame_counts = {len(scene.frames) for scene in scenes}
    _require(len(frame_counts) == 1, "scene_frame_count_mismatch")
    replay_frame_count = frame_counts.pop()
    _require(replay_frame_count > 1, "replay_has_no_dynamic_frames")
    _require(artifact_io.path_is_file(ffmpeg_path), "ffmpeg_binary_missing")
    artifact_io.make_dirs(output_root)
    stem = "g1_exploration_formal_trace"
    mp4_path = output_root / f"{stem}.mp4"
    gif_path = output_root / f"{stem}.gif"
    poster_path = output_root / f"{stem}_poster.png"
    midpoint_path = output_root / f"{stem}_midpoint.png"
    source_data_path = output_root / f"{stem}_source_data.csv"
    caption_path = output_root / f"{stem}_caption.md"

    fig, scene_artists = _build_figure(scenes)
    for scene, artists in zip(scenes, scene_artists, strict=True):
        _update_scene_artists(scene, artists, replay_index=0)

    start_hold_frames = int(round(1.5 * fps))
    end_hold_frames = int(round(2.5 * fps))
    render_indices = (
        [0] * start_hold_frames
        + list(range(replay_frame_count))
        + [replay_frame_count - 1] * end_hold_frames
    )
    matplotlib.rcParams["animation.ffmpeg_path"] = str(ffmpeg_path)
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=5200,
        metadata={
            "title": "G1 autonomous exploration process",
            "artist": "Xunce midterm dual-gate experiment",
            "comment": "Formal-trace reconstruction; visualization only",
        },
        extra_args=("-pix_fmt", "yuv420p", "-movflags", "+faststart"),
    )
    with writer.saving(fig, str(mp4_path), dpi=120):
        for replay_index in render_indices:
            for scene, artists in zip(scenes, scene_artists, strict=True):
                _update_scene_artists(
                    scene,
                    artists,
                    replay_index=replay_index,
                )
            writer.grab_frame(facecolor="white")

    midpoint_index = replay_frame_count // 2
    for scene, artists in zip(scenes, scene_artists, strict=True):
        _update_scene_artists(scene, artists, replay_index=midpoint_index)
    fig.savefig(
        midpoint_path,
        dpi=120,
        facecolor="white",
        bbox_inches=None,
    )
    for scene, artists in zip(scenes, scene_artists, strict=True):
        _update_scene_artists(
            scene,
            artists,
            replay_index=replay_frame_count - 1,
        )
    fig.savefig(
        poster_path,
        dpi=200,
        facecolor="white",
        bbox_inches=None,
    )
    plt.close(fig)

    gif_filter = (
        "[0:v]fps=8,scale=960:-2:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=96[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=3"
    )
    completed = subprocess.run(
        (
            str(ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(mp4_path),
            "-filter_complex",
            gif_filter,
            "-loop",
            "0",
            str(gif_path),
        ),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    _require(
        completed.returncode == 0,
        f"gif_export_failed:{completed.stderr.strip()}",
    )
    _write_source_data(
        scenes,
        subframes_per_step=subframes_per_step,
        output_path=source_data_path,
    )
    caption = (
        "# 图 6-5｜G1 自主探索过程\n\n"
        "左、右分别展示 Test-Q24 与 Unseen-24 中按预先固定的稳健中位数"
        "距离规则选出的代表场景。深色区域表示尚未观测，灰色表示已观测"
        "地形，青绿色表示当前帧新增观测区域；青色实线为累计执行轨迹，"
        "粉色虚线为当前规划路径，橙色箭头与黄色星标分别表示平台位姿和"
        "选定目标。下方曲线给出可覆盖区域的累计观测率，虚线为 80% "
        "中期门槛。\n\n"
        "动画由正式实验保存的逐步决策与规划轨迹重建；每个整数决策步的"
        "覆盖格数均与正式结果逐步一致。该动画仅用于结果展示，不重新执行"
        "策略，也不参与门槛复算。\n"
    )
    artifact_io.write_text(caption_path, caption)
    return {
        "mp4": mp4_path,
        "gif": gif_path,
        "poster_png": poster_path,
        "midpoint_png": midpoint_path,
        "source_data_csv": source_data_path,
        "caption_md": caption_path,
    }


def _output_file_entry(path: Path) -> dict[str, Any]:
    return {
        "name": path.name,
        "size_bytes": artifact_io.file_size(path),
        "sha256": _sha256_file(path),
    }


def _write_audit(
    scenes: Sequence[SceneReplay],
    *,
    outputs: Mapping[str, Path],
    output_root: Path,
    formal_root: Path,
    package_root: Path,
    stage6_config_path: Path,
    fps: int,
    subframes_per_step: int,
) -> Path:
    audit_path = output_root / "audit.json"
    payload = {
        "schema_version": AUDIT_SCHEMA,
        "renderer_id": RENDERER_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "visualization_only": True,
        "used_for_gate_recomputation": False,
        "policy_reexecuted": False,
        "source_kind": "formal_step_trace_reconstruction/v1",
        "selection_rule": SELECTION_RULE,
        "render": {
            "backend": "Python/matplotlib",
            "canvas_pixels": [1920, 1080],
            "fps": fps,
            "subframes_per_formal_step": subframes_per_step,
            "coverage_threshold_percent": COVERAGE_THRESHOLD_PERCENT,
            "start_hold_seconds": 1.5,
            "end_hold_seconds": 2.5,
        },
        "figure_contract": {
            "core_conclusion": (
                "Representative formal G1 trajectories progressively expand the "
                "exact observed coverable area to approximately 99% with zero "
                "safety violations in both evaluation splits."
            ),
            "archetype": "image plate + quant",
            "target_output": "academic report animation and high-resolution poster",
            "backend": "Python",
            "panel_map": {
                "a": "Test-Q24 representative map replay and coverage curve",
                "b": "Unseen-24 representative map replay and coverage curve",
            },
            "reviewer_risk_controls": [
                "representative selection rule recorded before rendering",
                "unknown terrain is never shown as observed",
                "integer-step coverage is checked against every formal trace row",
                "animation is explicitly excluded from gate recomputation",
            ],
        },
        "source_identity": {
            "formal_config_sha256": _sha256_file(formal_root / "config.json"),
            "scenario_manifest_sha256": _sha256_file(package_root / "manifest.json"),
            "stage6_config_sha256": _sha256_file(stage6_config_path),
            "renderer_source_sha256": _sha256_file(Path(__file__)),
        },
        "scenes": [
            {
                "split": scene.split,
                "episode_id": scene.episode_id,
                "scenario_id": scene.scenario_id,
                "scenario_hash": scene.scenario_hash,
                "denominator_count": scene.denominator_count,
                "denominator_sha256": scene.denominator_sha256,
                "final_coverage_percent": scene.final_coverage_percent,
                "steps_executed": scene.steps_executed,
                "safety_violation_count": scene.safety_violation_count,
                "source_result_sha256": scene.source_result_sha256,
                "selected_rows_sha256": scene.selected_rows_sha256,
                "selection": dict(scene.selection_audit),
                "verification": dict(scene.verification),
            }
            for scene in scenes
        ],
        "outputs": {
            key: _output_file_entry(path) for key, path in outputs.items()
        },
    }
    artifact_io.write_json(audit_path, payload)
    return audit_path


def _copy_report_assets(
    outputs: Mapping[str, Path],
    *,
    audit_path: Path,
    report_asset_root: Path,
) -> dict[str, Path]:
    artifact_io.make_dirs(report_asset_root)
    destinations = {
        "mp4": report_asset_root / "图6-5_G1探索过程动画.mp4",
        "gif": report_asset_root / "图6-5_G1探索过程动画.gif",
        "poster_png": report_asset_root / "图6-5_G1探索过程动画关键帧.png",
        "caption_md": report_asset_root / "图6-5_G1探索过程动画说明.md",
        "audit_json": report_asset_root / "图6-5_G1探索过程动画审计.json",
    }
    artifact_io.copy_file(outputs["mp4"], destinations["mp4"])
    artifact_io.copy_file(outputs["gif"], destinations["gif"])
    artifact_io.copy_file(outputs["poster_png"], destinations["poster_png"])
    artifact_io.copy_file(outputs["caption_md"], destinations["caption_md"])
    artifact_io.copy_file(audit_path, destinations["audit_json"])
    for key, destination in destinations.items():
        source_key = "poster_png" if key == "poster_png" else key
        if key == "audit_json":
            source_path = audit_path
        else:
            source_path = outputs[source_key]
        _require(
            _sha256_file(source_path) == _sha256_file(destination),
            f"report_asset_copy_sha256_mismatch:{key}",
        )
    return destinations


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 G1 正式逐步轨迹重建学术风格探索动画。"
    )
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--scenario-package-root", type=Path, required=True)
    parser.add_argument(
        "--stage6-config",
        type=Path,
        default=REPO_ROOT / "configs" / "ppo_highres_frontier_stage6_v1.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-asset-root", type=Path)
    parser.add_argument("--ffmpeg-path", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--subframes-per-step",
        type=int,
        default=DEFAULT_SUBFRAMES,
    )
    args = parser.parse_args(argv)
    _require(args.fps > 0, "fps_must_be_positive")
    _require(
        args.subframes_per_step > 0,
        "subframes_per_step_must_be_positive",
    )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    formal_config_path = args.formal_root / "config.json"
    _require(
        artifact_io.path_is_file(formal_config_path),
        "missing_formal_config",
    )
    formal_config = artifact_io.read_json(formal_config_path)
    formal_config_sha256 = str(formal_config["config_sha256"])
    _require(
        len(formal_config_sha256) == 64,
        "invalid_formal_config_sha256",
    )
    stage6_config = parse_stage6_config_bytes(
        artifact_io.read_bytes(args.stage6_config)
    )
    reconstruction_index = _load_reconstruction_index(
        args.scenario_package_root
    )
    episodes = tuple(
        _load_formal_episode(args.formal_root, split=split)
        for split in ("test_q24", "unseen24")
    )
    catalog = build_standard_catalog(verify_hashes=False)
    scenario_factory = StandardScenarioFactory(catalog)
    scenes = tuple(
        _reconstruct_scene(
            episode,
            catalog=catalog,
            scenario_factory=scenario_factory,
            reconstruction_index=reconstruction_index,
            stage6_config=stage6_config,
            formal_config_sha256=formal_config_sha256,
            subframes_per_step=args.subframes_per_step,
        )
        for episode in episodes
    )
    outputs = _render_outputs(
        scenes,
        output_root=args.output_root,
        ffmpeg_path=args.ffmpeg_path,
        fps=args.fps,
        subframes_per_step=args.subframes_per_step,
    )
    audit_path = _write_audit(
        scenes,
        outputs=outputs,
        output_root=args.output_root,
        formal_root=args.formal_root,
        package_root=args.scenario_package_root,
        stage6_config_path=args.stage6_config,
        fps=args.fps,
        subframes_per_step=args.subframes_per_step,
    )
    copied: dict[str, Path] = {}
    if args.report_asset_root is not None:
        copied = _copy_report_assets(
            outputs,
            audit_path=audit_path,
            report_asset_root=args.report_asset_root,
        )
    summary = {
        "status": "complete",
        "renderer_id": RENDERER_ID,
        "scenes": [
            {
                "split": scene.split,
                "episode_id": scene.episode_id,
                "scenario_id": scene.scenario_id,
                "final_coverage_percent": scene.final_coverage_percent,
                "steps_executed": scene.steps_executed,
                "per_step_coverage_exact": True,
            }
            for scene in scenes
        ],
        "outputs": {key: str(path) for key, path in outputs.items()},
        "audit": str(audit_path),
        "report_assets": {key: str(path) for key, path in copied.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
