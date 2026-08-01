"""Observed-only Stage 2 frontier segment and landing action generation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np

from lunar_exploration_ppo.env.map_state import ObservedMapState
from lunar_exploration_ppo.env.reachability import reachable_component
from lunar_exploration_ppo.env.scenario import LowResolutionPrior
from lunar_exploration_ppo.env.sensor_model import (
    iter_ray_cells_from_world,
    ray_cells_from_world,
)
from lunar_exploration_ppo.utils.geometry import CellXY, PoseXYTheta


SEGMENT_POLICY = "binary_regular_normal_or_irregular_gain_sampling/v1"
POTENTIAL_GAIN_SOURCE = "endpoint_fov_los_visible_unknown_gain/v1"
TOP_M_SELECTION_POLICY = "score_first_top_m/v1"
CANDIDATE_PRIORITY_SOURCE = "score_first_gain_value_cost_priority/v1"
_IRREGULAR_NEIGHBORHOOD_RADIUS_CELLS = 4
_IRREGULAR_MAX_ANCHORS_PER_SEGMENT = 12
_IRREGULAR_COMPASS_DIRECTIONS = (
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
)
_IRREGULAR_MAX_GAIN_EVALUATIONS_PER_SEGMENT = (
    _IRREGULAR_MAX_ANCHORS_PER_SEGMENT * (1 + len(_IRREGULAR_COMPASS_DIRECTIONS))
)
_IRREGULAR_GAIN_RANGE_M = 20.0
_IRREGULAR_GAIN_FOV_DEG = 90.0
_IRREGULAR_GAIN_RAY_ANGLE_STEP_DEG = 1.0
_START_FALLBACK_SOURCE = "safe_current_pose_best_of_8_absolute_headings/v1"
_START_FALLBACK_HEADINGS = tuple(
    index * math.pi / 4.0 for index in range(8)
)


@dataclass(frozen=True, slots=True)
class FrontierActionSet:
    cells: tuple[CellXY, ...]
    frontier_features: np.ndarray
    candidate_mask: np.ndarray
    frontier_mask: np.ndarray | None = None
    observed_safe_frontier_mask: np.ndarray | None = None
    reachable_frontier_mask: np.ndarray | None = None
    diagnostics: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        self.frontier_features.setflags(write=False)
        self.candidate_mask.setflags(write=False)
        for value in (self.frontier_mask, self.observed_safe_frontier_mask, self.reachable_frontier_mask):
            if value is not None:
                value.setflags(write=False)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics or {})))

    @property
    def candidate_count(self) -> int:
        return len(self.cells)


@dataclass(frozen=True, slots=True)
class GainEstimate:
    visible_unknown: tuple[CellXY, ...]
    visible_footprint: tuple[CellXY, ...]
    potential_gain_cells: int
    value_gain: float


@dataclass(frozen=True, slots=True)
class _FootprintCount:
    count: int

    def __len__(self) -> int:
        return self.count


@dataclass(frozen=True, slots=True)
class _CompactGainEstimate:
    visible_unknown: tuple[CellXY, ...]
    visible_footprint: _FootprintCount
    potential_gain_cells: int
    value_gain: float


@dataclass(frozen=True, slots=True)
class _IrregularScoredCandidate:
    cell: CellXY
    recommended_theta: float
    gain: GainEstimate | _CompactGainEstimate
    redirected: bool


@dataclass(frozen=True, slots=True)
class _RawCandidate:
    cell: CellXY
    segment_id: int
    segment_length_m: float
    normal_x: float
    normal_y: float
    normal_confidence: float
    generation_mode: int
    recommended_theta: float
    gain: GainEstimate | _CompactGainEstimate
    distance_norm: float
    clearance_norm: float
    reachability_cost_norm: float
    region_coverage_ratio: float
    region_unknown_ratio: float
    original_rank: int


@dataclass(frozen=True, slots=True)
class _FrontierSegmentPath:
    cells: tuple[CellXY, ...]
    closed: bool = False


@dataclass(frozen=True, slots=True)
class _ObservedClearanceSupport:
    clearance_m: np.ndarray


def score_first_top_m(priorities: Sequence[float], top_m: int) -> tuple[int, ...]:
    """Return stable descending-score indices; input order is the tie-break."""

    if top_m < 0:
        raise ValueError("top_m must be nonnegative")
    if any(not math.isfinite(float(value)) for value in priorities):
        raise ValueError("candidate priorities must be finite")
    return tuple(sorted(range(len(priorities)), key=lambda index: -float(priorities[index]))[:top_m])


def estimate_observed_only_gain(
    observed_state: ObservedMapState,
    prior: LowResolutionPrior,
    candidate: CellXY,
    recommended_theta: float,
    *,
    range_m: float = 20.0,
    fov_deg: float = 90.0,
    ray_angle_step_deg: float = 1.0,
) -> GainEstimate:
    """Estimate FOV gain, stopping only at blockers already present in observed state."""

    geometry = observed_state.geometry
    if not geometry.in_bounds(candidate):
        raise ValueError("gain candidate is outside the observed map")
    if range_m <= 0.0 or ray_angle_step_deg <= 0.0 or not 0.0 <= fov_deg <= 360.0:
        raise ValueError("invalid gain sensor geometry")
    return _estimate_observed_only_gain_with_blockers(
        observed_state,
        prior,
        candidate,
        recommended_theta,
        _build_observed_gain_blocker_mask(observed_state),
        range_m=range_m,
        fov_deg=fov_deg,
        ray_angle_step_deg=ray_angle_step_deg,
    )


def _build_observed_gain_blocker_mask(observed_state: ObservedMapState) -> np.ndarray:
    return observed_state.observed_mask & (
        observed_state.obstacle | (observed_state.slope_deg > 30.0)
    )


def _estimate_observed_only_gain_with_blockers(
    observed_state: ObservedMapState,
    prior: LowResolutionPrior,
    candidate: CellXY,
    recommended_theta: float,
    observed_blocker: np.ndarray,
    *,
    range_m: float = 20.0,
    fov_deg: float = 90.0,
    ray_angle_step_deg: float = 1.0,
) -> GainEstimate:
    """Estimate observed-only gain using a precomputed observed blocker mask."""

    geometry = observed_state.geometry
    if not geometry.in_bounds(candidate):
        raise ValueError("gain candidate is outside the observed map")
    if range_m <= 0.0 or ray_angle_step_deg <= 0.0 or not 0.0 <= fov_deg <= 360.0:
        raise ValueError("invalid gain sensor geometry")
    if (
        not isinstance(observed_blocker, np.ndarray)
        or observed_blocker.dtype != np.bool_
        or observed_blocker.shape != observed_state.observed_mask.shape
    ):
        raise ValueError("gain blocker mask must be a boolean observed-map-shaped array")
    ray_count = int(round(fov_deg / ray_angle_step_deg)) + 1
    footprint: set[CellXY] = set()
    origin = geometry.cell_to_world_center(candidate)
    for ray_index in range(ray_count):
        offset = -fov_deg / 2.0 + ray_index * ray_angle_step_deg
        angle = recommended_theta + math.radians(offset)
        for ray_cell_index, cell in enumerate(
            ray_cells_from_world(origin, angle, geometry, range_m=range_m)
        ):
            footprint.add(cell)
            if ray_cell_index > 0 and observed_blocker[cell.y, cell.x]:
                break
    visible_unknown = tuple(
        sorted((cell for cell in footprint if not observed_state.observed_mask[cell.y, cell.x]), key=lambda c: (c.y, c.x))
    )
    low_height, low_width = prior.channels.shape[1:]
    value_gain = 0.0
    for cell in visible_unknown:
        low_x = min(low_width - 1, int(cell.x * low_width / geometry.width))
        low_y = min(low_height - 1, int(cell.y * low_height / geometry.height))
        value_gain += float(prior.channels[1, low_y, low_x])
    return GainEstimate(
        visible_unknown=visible_unknown,
        visible_footprint=tuple(sorted(footprint, key=lambda c: (c.y, c.x))),
        potential_gain_cells=len(visible_unknown),
        value_gain=value_gain,
    )


def _estimate_observed_only_gain_batch_with_blockers(
    observed_state: ObservedMapState,
    prior: LowResolutionPrior,
    cells: Sequence[CellXY],
    thetas: Sequence[float],
    observed_blocker: np.ndarray,
    *,
    range_m: float = 20.0,
    fov_deg: float = 90.0,
    ray_angle_step_deg: float = 1.0,
) -> tuple[_CompactGainEstimate, ...]:
    """Evaluate one segment's gain candidates with vectorized DDA steps."""

    geometry = observed_state.geometry
    cell_tuple = tuple(cells)
    theta_tuple = tuple(float(theta) for theta in thetas)
    if len(cell_tuple) != len(theta_tuple):
        raise ValueError("batch gain cells and thetas must have equal length")
    if not cell_tuple:
        return ()
    if any(not geometry.in_bounds(cell) for cell in cell_tuple):
        raise ValueError("gain candidate is outside the observed map")
    if any(not math.isfinite(theta) for theta in theta_tuple):
        raise ValueError("batch gain theta must be finite")
    if range_m <= 0.0 or ray_angle_step_deg <= 0.0 or not 0.0 <= fov_deg <= 360.0:
        raise ValueError("invalid gain sensor geometry")
    if (
        not isinstance(observed_blocker, np.ndarray)
        or observed_blocker.dtype != np.bool_
        or observed_blocker.shape != observed_state.observed_mask.shape
        or not observed_blocker.flags.c_contiguous
    ):
        raise ValueError("gain blocker mask must be a C-contiguous boolean observed-map-shaped array")
    if not observed_state.observed_mask.flags.c_contiguous:
        raise ValueError("observed mask must be C-contiguous")

    candidate_count = len(cell_tuple)
    ray_count = int(round(fov_deg / ray_angle_step_deg)) + 1
    shape = (candidate_count, ray_count)
    resolution = geometry.resolution_m
    height = geometry.height
    width = geometry.width
    map_cell_count = height * width
    grid_origin_x = geometry.origin.x
    grid_origin_y = geometry.origin.y

    cell_x = np.asarray([cell.x for cell in cell_tuple], dtype=np.int32)
    cell_y = np.asarray([cell.y for cell in cell_tuple], dtype=np.int32)
    current_x = np.broadcast_to(cell_x[:, None], shape).copy()
    current_y = np.broadcast_to(cell_y[:, None], shape).copy()
    origin_world_x = grid_origin_x + (cell_x.astype(np.float64) + 0.5) * resolution
    origin_world_y = grid_origin_y + (cell_y.astype(np.float64) + 0.5) * resolution

    direction_x = np.empty(shape, dtype=np.float64)
    direction_y = np.empty(shape, dtype=np.float64)
    half_fov = fov_deg / 2.0
    offsets = tuple(
        math.radians(-half_fov + ray_index * ray_angle_step_deg)
        for ray_index in range(ray_count)
    )
    for candidate_index, theta in enumerate(theta_tuple):
        for ray_index, offset in enumerate(offsets):
            angle = theta + offset
            direction_x[candidate_index, ray_index] = math.cos(angle)
            direction_y[candidate_index, ray_index] = math.sin(angle)

    step_x = np.where(direction_x > 0.0, 1, np.where(direction_x < 0.0, -1, 0)).astype(np.int8)
    step_y = np.where(direction_y > 0.0, 1, np.where(direction_y < 0.0, -1, 0)).astype(np.int8)
    t_max_x = np.full(shape, math.inf, dtype=np.float64)
    t_max_y = np.full(shape, math.inf, dtype=np.float64)
    t_delta_x = np.full(shape, math.inf, dtype=np.float64)
    t_delta_y = np.full(shape, math.inf, dtype=np.float64)

    moving_x = step_x != 0
    moving_y = step_y != 0
    boundary_x = grid_origin_x + (
        current_x.astype(np.float64) + (step_x > 0).astype(np.float64)
    ) * resolution
    boundary_y = grid_origin_y + (
        current_y.astype(np.float64) + (step_y > 0).astype(np.float64)
    ) * resolution
    np.divide(
        boundary_x - origin_world_x[:, None],
        direction_x,
        out=t_max_x,
        where=moving_x,
    )
    np.divide(resolution, np.abs(direction_x), out=t_delta_x, where=moving_x)
    np.divide(
        boundary_y - origin_world_y[:, None],
        direction_y,
        out=t_max_y,
        where=moving_y,
    )
    np.divide(resolution, np.abs(direction_y), out=t_delta_y, where=moving_y)

    active = np.ones(shape, dtype=np.bool_)
    candidate_ids = np.broadcast_to(
        np.arange(candidate_count, dtype=np.int64)[:, None],
        shape,
    )
    blocker_flat = observed_blocker.ravel(order="C")
    visible_key_parts: list[np.ndarray] = []
    ray_cell_index = 0
    while bool(np.any(active)):
        in_bounds = (
            active
            & (current_x >= 0)
            & (current_x < width)
            & (current_y >= 0)
            & (current_y < height)
        )
        if not bool(np.any(in_bounds)):
            break
        safe_x = np.where(in_bounds, current_x, 0)
        safe_y = np.where(in_bounds, current_y, 0)
        flat = safe_y.astype(np.int64) * width + safe_x.astype(np.int64)
        visible_key_parts.append(
            candidate_ids[in_bounds] * map_cell_count + flat[in_bounds]
        )

        blocked = np.zeros(shape, dtype=np.bool_)
        if ray_cell_index > 0:
            blocked[in_bounds] = blocker_flat[flat[in_bounds]]
        active = in_bounds & ~blocked
        if not bool(np.any(active)):
            break

        next_distance = np.minimum(t_max_x, t_max_y)
        active &= next_distance <= range_m
        if not bool(np.any(active)):
            break

        equal = t_max_x == t_max_y
        finite_close = (
            np.isfinite(t_max_x)
            & np.isfinite(t_max_y)
            & (np.abs(t_max_x - t_max_y) <= 1.0e-12)
        )
        tie = active & (equal | finite_close)
        x_first = active & ~tie & (t_max_x < t_max_y)
        y_first = active & ~tie & ~x_first

        current_x[tie] += step_x[tie]
        current_y[tie] += step_y[tie]
        t_max_x[tie] += t_delta_x[tie]
        t_max_y[tie] += t_delta_y[tie]
        current_x[x_first] += step_x[x_first]
        t_max_x[x_first] += t_delta_x[x_first]
        current_y[y_first] += step_y[y_first]
        t_max_y[y_first] += t_delta_y[y_first]
        ray_cell_index += 1

    if visible_key_parts:
        unique_keys = np.unique(np.concatenate(visible_key_parts))
    else:
        unique_keys = np.empty((0,), dtype=np.int64)
    unique_candidate = unique_keys // map_cell_count
    unique_flat = unique_keys % map_cell_count
    footprint_counts = np.bincount(unique_candidate, minlength=candidate_count)
    observed_flat = observed_state.observed_mask.ravel(order="C")
    unknown_mask = ~observed_flat[unique_flat]
    unknown_candidate = unique_candidate[unknown_mask]
    unknown_flat = unique_flat[unknown_mask]
    unknown_counts = np.bincount(unknown_candidate, minlength=candidate_count)

    low_height, low_width = prior.channels.shape[1:]
    results: list[_CompactGainEstimate] = []
    for candidate_index in range(candidate_count):
        candidate_unknown = unknown_flat[unknown_candidate == candidate_index]
        value_gain = 0.0
        for flat_index_value in candidate_unknown:
            flat_index = int(flat_index_value)
            visible_x = flat_index % width
            visible_y = flat_index // width
            low_x = min(low_width - 1, int(visible_x * low_width / geometry.width))
            low_y = min(low_height - 1, int(visible_y * low_height / geometry.height))
            value_gain += float(prior.channels[1, low_y, low_x])
        results.append(
            _CompactGainEstimate(
                visible_unknown=(),
                visible_footprint=_FootprintCount(int(footprint_counts[candidate_index])),
                potential_gain_cells=int(unknown_counts[candidate_index]),
                value_gain=value_gain,
            )
        )
    return tuple(results)


class FrontierGenerator:
    def __init__(
        self,
        *,
        top_m: int = 512,
        normal_confidence_threshold: float = 0.6,
        standoff_distance_m: float = 5.0,
        max_candidates_per_segment: int = 3,
        min_clearance_m: float = 0.5215874761,
        traversability_threshold: float = 0.50,
    ) -> None:
        if top_m <= 0:
            raise ValueError("frontier top_m must be positive")
        if normal_confidence_threshold != 0.6:
            raise ValueError("normal_confidence_threshold is frozen at 0.6")
        if standoff_distance_m != 5.0:
            raise ValueError("standoff_distance_m is frozen at 5.0")
        if max_candidates_per_segment != 3:
            raise ValueError("max_candidates_per_segment is frozen at 3")
        if (
            not math.isfinite(min_clearance_m)
            or min_clearance_m <= 0.0
            or min_clearance_m != 0.5215874761
        ):
            raise ValueError("min_clearance_m is frozen at 0.5215874761")
        if (
            not math.isfinite(traversability_threshold)
            or not 0.0 <= traversability_threshold <= 1.0
            or traversability_threshold != 0.50
        ):
            raise ValueError("traversability_threshold is frozen at 0.50")
        self.top_m = top_m
        self.normal_confidence_threshold = normal_confidence_threshold
        self.standoff_distance_m = standoff_distance_m
        self.max_candidates_per_segment = max_candidates_per_segment
        self.min_clearance_m = min_clearance_m
        self.traversability_threshold = traversability_threshold

    def extract(
        self,
        observed_state: ObservedMapState,
        prior: LowResolutionPrior,
        pose: PoseXYTheta,
    ) -> FrontierActionSet:
        geometry = observed_state.geometry
        planning_safe_mask = observed_state.planning_safe_mask
        component = reachable_component(planning_safe_mask, pose.cell)
        frontier_mask = _frontier_mask(observed_state.observed_mask)
        observed_safe_cells = _observed_safe_cells(observed_state, self.traversability_threshold)
        safe_frontier = frontier_mask & observed_safe_cells
        reachable_frontier = safe_frontier & component
        landing_mask = observed_safe_cells & component
        observed_gain_blocker_mask = _build_observed_gain_blocker_mask(observed_state)
        observed_clearance_support = _build_observed_clearance_support(observed_state)
        contour_frontier = reachable_frontier & _frontier_contour_mask(observed_state.observed_mask)
        primary_segments = _frontier_segment_paths(contour_frontier)
        diagonal_m = max(math.hypot(geometry.width - 1, geometry.height - 1) * geometry.resolution_m, 1.0)
        def collect_raw_candidates(
            candidate_segments: tuple[_FrontierSegmentPath, ...],
        ) -> list[_RawCandidate]:
            raw: list[_RawCandidate] = []
            for segment_id, segment_path in enumerate(candidate_segments):
                segment = _unique_segment_cells(segment_path)
                normal_x, normal_y, confidence = _segment_normal(observed_state.observed_mask, segment)
                segment_length_m = _segment_arc_length_m(segment_path, geometry.resolution_m)
                if confidence >= self.normal_confidence_threshold:
                    candidate_records = [
                        (cell, theta, None)
                        for cell, theta in self._regular_candidates(
                            observed_state,
                            landing_mask,
                            prior,
                            segment_path,
                            normal_x,
                            normal_y,
                            segment_length_m,
                            observed_gain_blocker_mask,
                            observed_clearance_support,
                        )
                    ]
                    mode = 0
                else:
                    candidate_records = [
                        (candidate.cell, candidate.recommended_theta, candidate.gain)
                        for candidate in self._irregular_scored_candidates(
                            observed_state,
                            landing_mask,
                            prior,
                            segment_path,
                            observed_gain_blocker_mask,
                        )
                    ]
                    mode = 1
                segment_raw: list[_RawCandidate] = []
                for cell, theta, bound_gain in candidate_records:
                    if not geometry.in_bounds(cell) or not landing_mask[cell.y, cell.x]:
                        continue
                    clearance_m = _observed_clearance_m_with_support(
                        observed_state,
                        cell,
                        observed_clearance_support,
                    )
                    if clearance_m < self.min_clearance_m:
                        continue
                    gain = (
                        bound_gain
                        if bound_gain is not None
                        else _estimate_observed_only_gain_with_blockers(
                            observed_state,
                            prior,
                            cell,
                            theta,
                            observed_gain_blocker_mask,
                        )
                    )
                    if gain.potential_gain_cells <= 0:
                        continue
                    dx = (cell.x - pose.cell.x) * geometry.resolution_m
                    dy = (cell.y - pose.cell.y) * geometry.resolution_m
                    distance_norm = math.hypot(dx, dy) / diagonal_m
                    low_y, low_x, coverage_ratio, unknown_ratio = _region_ratios(observed_state, prior, cell)
                    del low_y, low_x
                    segment_raw.append(
                        _RawCandidate(
                            cell=cell,
                            segment_id=segment_id,
                            segment_length_m=segment_length_m,
                            normal_x=normal_x,
                            normal_y=normal_y,
                            normal_confidence=confidence,
                            generation_mode=mode,
                            recommended_theta=theta,
                            gain=gain,
                            distance_norm=distance_norm,
                            clearance_norm=min(clearance_m / diagonal_m, 1.0),
                            reachability_cost_norm=distance_norm,
                            region_coverage_ratio=coverage_ratio,
                            region_unknown_ratio=unknown_ratio,
                            original_rank=0,
                        )
                    )
                if mode == 1:
                    _, segment_priorities = _feature_rows(
                        segment_raw,
                        pose,
                        observed_state,
                        len(candidate_segments),
                    )
                    segment_selected = score_first_top_m(
                        segment_priorities,
                        self.max_candidates_per_segment,
                    )
                else:
                    segment_selected = tuple(
                        range(min(len(segment_raw), self.max_candidates_per_segment))
                    )
                for index in segment_selected:
                    raw.append(replace(segment_raw[index], original_rank=len(raw)))
            return raw

        segments = primary_segments
        raw = collect_raw_candidates(segments)
        fallback_activated = not raw
        fallback_segments: tuple[_FrontierSegmentPath, ...] = ()
        if fallback_activated:
            fallback_contour = frontier_mask & _frontier_contour_mask(observed_state.observed_mask)
            fallback_segments = _frontier_segment_paths(fallback_contour)
            segments = fallback_segments
            raw = collect_raw_candidates(segments)
        start_fallback_activated = False
        start_fallback_heading_index: int | None = None
        start_fallback_theta: float | None = None
        start_fallback_gain_cells = 0
        if not raw and geometry.in_bounds(pose.cell) and landing_mask[pose.cell.y, pose.cell.x]:
            clearance_m = _observed_clearance_m_with_support(
                observed_state,
                pose.cell,
                observed_clearance_support,
            )
            if clearance_m >= self.min_clearance_m:
                heading_gains = tuple(
                    _estimate_observed_only_gain_with_blockers(
                        observed_state,
                        prior,
                        pose.cell,
                        theta,
                        observed_gain_blocker_mask,
                        range_m=_IRREGULAR_GAIN_RANGE_M,
                        fov_deg=_IRREGULAR_GAIN_FOV_DEG,
                        ray_angle_step_deg=_IRREGULAR_GAIN_RAY_ANGLE_STEP_DEG,
                    )
                    for theta in _START_FALLBACK_HEADINGS
                )
                best_heading_index = max(
                    range(len(_START_FALLBACK_HEADINGS)),
                    key=lambda index: (
                        heading_gains[index].potential_gain_cells,
                        heading_gains[index].value_gain,
                        -index,
                    ),
                )
                gain = heading_gains[best_heading_index]
                if gain.potential_gain_cells > 0:
                    start_fallback_heading_index = best_heading_index
                    start_fallback_theta = math.atan2(
                        math.sin(_START_FALLBACK_HEADINGS[start_fallback_heading_index]),
                        math.cos(_START_FALLBACK_HEADINGS[start_fallback_heading_index]),
                    )
                    _, _, coverage_ratio, unknown_ratio = _region_ratios(
                        observed_state,
                        prior,
                        pose.cell,
                    )
                    raw = [
                        _RawCandidate(
                            cell=pose.cell,
                            segment_id=0,
                            segment_length_m=0.0,
                            normal_x=0.0,
                            normal_y=0.0,
                            normal_confidence=0.0,
                            generation_mode=1,
                            recommended_theta=start_fallback_theta,
                            gain=gain,
                            distance_norm=0.0,
                            clearance_norm=min(clearance_m / diagonal_m, 1.0),
                            reachability_cost_norm=0.0,
                            region_coverage_ratio=coverage_ratio,
                            region_unknown_ratio=unknown_ratio,
                            original_rank=0,
                        )
                    ]
                    start_fallback_activated = True
                    start_fallback_gain_cells = gain.potential_gain_cells
        features, priorities = _feature_rows(raw, pose, observed_state, len(segments))
        selected = score_first_top_m(priorities, self.top_m)
        cells = tuple(raw[index].cell for index in selected)
        padded = np.zeros((self.top_m, 22), dtype=np.float32)
        mask = np.zeros((self.top_m,), dtype=bool)
        for output_index, raw_index in enumerate(selected):
            padded[output_index] = features[raw_index]
            mask[output_index] = True
        pruned = tuple(index for index in range(len(raw)) if index not in set(selected))
        diagnostics: dict[str, object] = {
            "frontier_segment_candidate_policy": SEGMENT_POLICY,
            "potential_gain_source": POTENTIAL_GAIN_SOURCE,
            "top_m_selection_policy": TOP_M_SELECTION_POLICY,
            "candidate_priority_source": CANDIDATE_PRIORITY_SOURCE,
            "candidate_count_before_top_m": len(raw),
            "candidate_count_after_top_m": len(selected),
            "candidate_overflow_count": max(0, len(raw) - self.top_m),
            "kept_min_priority": min((priorities[index] for index in selected), default=0.0),
            "pruned_max_priority": max((priorities[index] for index in pruned), default=0.0),
            "selected_candidate_original_rank": [raw[index].original_rank for index in selected],
            "segment_count": len(segments),
            "fallback_activated": fallback_activated,
            "fallback_segment_count": len(fallback_segments),
            "start_fallback_activated": start_fallback_activated,
            "start_fallback_source": _START_FALLBACK_SOURCE,
            "start_fallback_heading_index": start_fallback_heading_index,
            "start_fallback_recommended_theta": start_fallback_theta,
            "start_fallback_gain_cells": start_fallback_gain_cells,
            "regular_segment_count": sum(
                _segment_normal(observed_state.observed_mask, _unique_segment_cells(segment))[2]
                >= self.normal_confidence_threshold
                for segment in segments
            ),
            "reachability_mask": "planning_safe_mask/v1",
        }
        return FrontierActionSet(
            cells=cells,
            frontier_features=padded,
            candidate_mask=mask,
            frontier_mask=frontier_mask,
            observed_safe_frontier_mask=safe_frontier,
            reachable_frontier_mask=reachable_frontier,
            diagnostics=diagnostics,
        )

    def _regular_candidates(
        self,
        state: ObservedMapState,
        landing_mask: np.ndarray,
        prior: LowResolutionPrior,
        segment: _FrontierSegmentPath,
        normal_x: float,
        normal_y: float,
        segment_length_m: float,
        observed_gain_blocker_mask: np.ndarray,
        observed_clearance_support: _ObservedClearanceSupport,
    ) -> list[tuple[CellXY, float]]:
        anchor_count = _regular_anchor_count(segment_length_m)
        anchors = _arc_length_anchor_cells(
            segment,
            anchor_count,
            resolution_m=state.geometry.resolution_m,
        )
        result: list[tuple[CellXY, float]] = []
        theta = math.atan2(normal_y, normal_x)
        standoff_cells = self.standoff_distance_m / state.geometry.resolution_m
        correction_radius_cells = max(8, math.ceil(standoff_cells))
        for anchor in anchors:
            ideal = CellXY(
                round(anchor.x - normal_x * standoff_cells),
                round(anchor.y - normal_y * standoff_cells),
            )
            corrected = _nearest_valid_gain_cell(
                state,
                landing_mask,
                ideal,
                radius_cells=correction_radius_cells,
                min_clearance_m=self.min_clearance_m,
                prior=prior,
                recommended_theta=theta,
                observed_gain_blocker_mask=observed_gain_blocker_mask,
                observed_clearance_support=observed_clearance_support,
            )
            if corrected is not None and corrected not in (cell for cell, _ in result):
                result.append((corrected, theta))
        return result

    def _irregular_candidates(
        self,
        state: ObservedMapState,
        landing_mask: np.ndarray,
        prior: LowResolutionPrior,
        segment: _FrontierSegmentPath,
        observed_gain_blocker_mask: np.ndarray,
    ) -> list[tuple[CellXY, float]]:
        return [
            (candidate.cell, candidate.recommended_theta)
            for candidate in self._irregular_scored_candidates(
                state,
                landing_mask,
                prior,
                segment,
                observed_gain_blocker_mask,
            )[: self.max_candidates_per_segment]
        ]

    def _irregular_scored_candidates(
        self,
        state: ObservedMapState,
        landing_mask: np.ndarray,
        prior: LowResolutionPrior,
        segment: _FrontierSegmentPath,
        observed_gain_blocker_mask: np.ndarray,
    ) -> list[_IrregularScoredCandidate]:
        sample_count = min(_IRREGULAR_MAX_ANCHORS_PER_SEGMENT, len(segment.cells))
        anchors = _arc_length_anchor_cells(
            segment,
            sample_count,
            resolution_m=state.geometry.resolution_m,
        )
        pool: list[CellXY] = []
        pooled_cells: set[CellXY] = set()

        def add_to_pool(cell: CellXY | None) -> None:
            if cell is not None and cell not in pooled_cells:
                pooled_cells.add(cell)
                pool.append(cell)

        for anchor in anchors:
            nearest = _nearest_valid_cell(
                state,
                landing_mask,
                anchor,
                radius_cells=_IRREGULAR_NEIGHBORHOOD_RADIUS_CELLS,
            )
            if nearest is None:
                continue
            add_to_pool(nearest)
            for direction_x, direction_y in _IRREGULAR_COMPASS_DIRECTIONS:
                extremal: CellXY | None = None
                for distance in range(1, _IRREGULAR_NEIGHBORHOOD_RADIUS_CELLS + 1):
                    cell = CellXY(
                        anchor.x + direction_x * distance,
                        anchor.y + direction_y * distance,
                    )
                    if state.geometry.in_bounds(cell) and landing_mask[cell.y, cell.x]:
                        extremal = cell
                add_to_pool(extremal)

        heading_targets = _irregular_anchor_unknown_targets(state, anchors)
        thetas: list[float] = []
        redirected: list[bool] = []
        for cell in pool:
            unknown_x, unknown_y = _local_unknown_vector(
                state.observed_mask,
                cell,
                radius_cells=_IRREGULAR_NEIGHBORHOOD_RADIUS_CELLS,
            )
            local_theta = math.atan2(unknown_y, unknown_x) if unknown_x or unknown_y else 0.0
            visible_target = _first_visible_irregular_heading_target(
                state,
                cell,
                heading_targets,
                observed_gain_blocker_mask,
                local_theta,
            )
            if visible_target is None:
                theta = local_theta
            else:
                theta = math.atan2(
                    visible_target.y - cell.y,
                    visible_target.x - cell.x,
                )
            thetas.append(theta)
            redirected.append(visible_target is not None)

        gains = _estimate_observed_only_gain_batch_with_blockers(
            state,
            prior,
            pool,
            thetas,
            observed_gain_blocker_mask,
            range_m=_IRREGULAR_GAIN_RANGE_M,
            fov_deg=_IRREGULAR_GAIN_FOV_DEG,
            ray_angle_step_deg=_IRREGULAR_GAIN_RAY_ANGLE_STEP_DEG,
        )
        sampled: list[_IrregularScoredCandidate] = []
        for cell, theta, is_redirected, gain in zip(
            pool,
            thetas,
            redirected,
            gains,
            strict=True,
        ):
            if gain.potential_gain_cells > 0:
                sampled.append(
                    _IrregularScoredCandidate(
                        cell=cell,
                        recommended_theta=theta,
                        gain=gain,
                        redirected=is_redirected,
                    )
                )
        sampled.sort(
            key=lambda item: (
                -item.gain.potential_gain_cells,
                item.cell.y,
                item.cell.x,
            )
        )
        local_candidates = [
            item for item in sampled if not item.redirected
        ][: self.max_candidates_per_segment]
        redirected_candidates = [item for item in sampled if item.redirected]
        return [
            *local_candidates,
            *redirected_candidates,
        ]


def _irregular_anchor_unknown_targets(
    state: ObservedMapState,
    anchors: Sequence[CellXY],
) -> tuple[CellXY, ...]:
    targets: list[CellXY] = []
    seen: set[CellXY] = set()
    for anchor in anchors:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                target = CellXY(anchor.x + dx, anchor.y + dy)
                if (
                    state.geometry.in_bounds(target)
                    and not state.observed_mask[target.y, target.x]
                    and target not in seen
                ):
                    seen.add(target)
                    targets.append(target)
    return tuple(targets)


def _first_visible_irregular_heading_target(
    state: ObservedMapState,
    landing: CellXY,
    targets: Sequence[CellXY],
    observed_gain_blocker_mask: np.ndarray,
    local_theta: float,
) -> CellXY | None:
    origin = state.geometry.cell_to_world_center(landing)
    visible_target: CellXY | None = None
    for target in targets:
        delta_x = target.x - landing.x
        delta_y = target.y - landing.y
        theta = math.atan2(delta_y, delta_x)
        range_m = math.hypot(delta_x, delta_y) * state.geometry.resolution_m
        if range_m > _IRREGULAR_GAIN_RANGE_M:
            continue
        for ray_cell_index, ray_cell in enumerate(
            iter_ray_cells_from_world(
                origin,
                theta,
                state.geometry,
                range_m=range_m,
            )
        ):
            if ray_cell_index == 0:
                continue
            if observed_gain_blocker_mask[ray_cell.y, ray_cell.x]:
                break
            if ray_cell == target:
                visible_target = target
                break
        if visible_target is not None:
            break
    if visible_target is None:
        return None

    ray_count = int(
        round(_IRREGULAR_GAIN_FOV_DEG / _IRREGULAR_GAIN_RAY_ANGLE_STEP_DEG)
    ) + 1
    for ray_index in range(ray_count):
        sampled_theta = local_theta + math.radians(
            -_IRREGULAR_GAIN_FOV_DEG / 2.0
            + ray_index * _IRREGULAR_GAIN_RAY_ANGLE_STEP_DEG
        )
        for ray_cell_index, ray_cell in enumerate(
            iter_ray_cells_from_world(
                origin,
                sampled_theta,
                state.geometry,
                range_m=_IRREGULAR_GAIN_RANGE_M,
            )
        ):
            if ray_cell_index == 0:
                continue
            if observed_gain_blocker_mask[ray_cell.y, ray_cell.x]:
                break
            if not state.observed_mask[ray_cell.y, ray_cell.x]:
                return None
    return visible_target


def _observed_safe_cells(state: ObservedMapState, threshold: float) -> np.ndarray:
    return (
        state.observed_mask
        & ~state.obstacle
        & (state.slope_deg <= 30.0)
        & (state.traversability >= threshold)
        & state.planning_safe_mask
    )


def _valid_landing_cell(state: ObservedMapState, component: np.ndarray, cell: CellXY, threshold: float) -> bool:
    return state.geometry.in_bounds(cell) and bool(component[cell.y, cell.x]) and bool(
        _observed_safe_cells(state, threshold)[cell.y, cell.x]
    )


def _frontier_mask(observed_mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(observed_mask, dtype=bool)
    height, width = observed_mask.shape
    for y, x in np.argwhere(observed_mask):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and not observed_mask[ny, nx]:
                    result[y, x] = True
                    break
            if result[y, x]:
                break
    return result


def _frontier_contour_mask(observed_mask: np.ndarray) -> np.ndarray:
    """Return edge-sharing observed/unknown boundary cells for contour traversal."""

    result = np.zeros_like(observed_mask, dtype=bool)
    height, width = observed_mask.shape
    for y, x in np.argwhere(observed_mask):
        for dx, dy in ((0, -1), (-1, 0), (1, 0), (0, 1)):
            neighbor_x = x + dx
            neighbor_y = y + dy
            if (
                0 <= neighbor_x < width
                and 0 <= neighbor_y < height
                and not observed_mask[neighbor_y, neighbor_x]
            ):
                result[y, x] = True
                break
    return result


def _frontier_segment_paths(mask: np.ndarray) -> tuple[_FrontierSegmentPath, ...]:
    """Split a contour graph at junctions and freeze a deterministic traversal."""

    nodes = {CellXY(int(x), int(y)) for y, x in np.argwhere(np.asarray(mask, dtype=bool))}
    if not nodes:
        return ()
    adjacency_lists: dict[CellXY, list[CellXY]] = {cell: [] for cell in nodes}
    for cell in sorted(nodes, key=_cell_sort_key):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = CellXY(cell.x + dx, cell.y + dy)
                if neighbor not in nodes or _cell_sort_key(neighbor) <= _cell_sort_key(cell):
                    continue
                if dx != 0 and dy != 0:
                    horizontal_bridge = CellXY(cell.x + dx, cell.y)
                    vertical_bridge = CellXY(cell.x, cell.y + dy)
                    if horizontal_bridge in nodes or vertical_bridge in nodes:
                        continue
                adjacency_lists[cell].append(neighbor)
                adjacency_lists[neighbor].append(cell)
    adjacency = {
        cell: tuple(sorted(neighbors, key=_cell_sort_key))
        for cell, neighbors in adjacency_lists.items()
    }
    paths: list[_FrontierSegmentPath] = []
    remaining = set(nodes)
    for seed in sorted(nodes, key=_cell_sort_key):
        if seed not in remaining:
            continue
        component: list[CellXY] = []
        queue = [seed]
        remaining.remove(seed)
        for cell in queue:
            component.append(cell)
            for neighbor in adjacency[cell]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        paths.append(_traverse_frontier_component(adjacency, tuple(component)))
    return tuple(paths)


def _cell_sort_key(cell: CellXY) -> tuple[int, int]:
    return cell.y, cell.x


def _traverse_frontier_component(
    adjacency: Mapping[CellXY, tuple[CellXY, ...]],
    component: tuple[CellXY, ...],
) -> _FrontierSegmentPath:
    members = tuple(sorted(component, key=_cell_sort_key))
    if len(members) == 1:
        return _FrontierSegmentPath(members)
    terminals = tuple(cell for cell in members if len(adjacency[cell]) == 1)
    if all(len(adjacency[cell]) <= 2 for cell in members):
        if terminals:
            start = min(terminals, key=_cell_sort_key)
            ordered = [start]
            previous: CellXY | None = None
            current = start
            while True:
                next_cells = [neighbor for neighbor in adjacency[current] if neighbor != previous]
                if not next_cells:
                    break
                next_cell = next_cells[0]
                ordered.append(next_cell)
                previous, current = current, next_cell
            return _FrontierSegmentPath(tuple(ordered))
        start = members[0]
        previous = start
        current = adjacency[start][0]
        ordered = [start]
        while current != start:
            ordered.append(current)
            next_cell = next(neighbor for neighbor in adjacency[current] if neighbor != previous)
            previous, current = current, next_cell
        return _FrontierSegmentPath(tuple(ordered), closed=True)

    root = min(terminals or members, key=_cell_sort_key)
    parent: dict[CellXY, CellXY | None] = {root: None}
    distance = {root: 0.0}
    breadth_first = [root]
    for cell in breadth_first:
        for neighbor in adjacency[cell]:
            if neighbor in parent:
                continue
            parent[neighbor] = cell
            distance[neighbor] = distance[cell] + _grid_step_length_cells(cell, neighbor)
            breadth_first.append(neighbor)
    child_lists: dict[CellXY, list[CellXY]] = {cell: [] for cell in members}
    for candidate, candidate_parent in parent.items():
        if candidate_parent is not None:
            child_lists[candidate_parent].append(candidate)
    children = {
        cell: tuple(sorted(candidates, key=_cell_sort_key))
        for cell, candidates in child_lists.items()
    }
    tree_leaves = tuple(cell for cell in members if not children[cell])
    max_distance = max(distance[cell] for cell in tree_leaves)
    finish = min(
        (cell for cell in tree_leaves if distance[cell] == max_distance),
        key=_cell_sort_key,
    )
    next_to_finish: dict[CellXY, CellXY] = {}
    current = finish
    while parent[current] is not None:
        previous = parent[current]
        assert previous is not None
        next_to_finish[previous] = current
        current = previous

    ordered = [root]
    operations: list[tuple[str, CellXY]] = [("finish", root)]
    while operations:
        operation, cell = operations.pop()
        if operation == "append":
            ordered.append(cell)
            continue
        path_child = next_to_finish.get(cell) if operation == "finish" else None
        off_path_children = [child for child in children[cell] if child != path_child]
        sequence: list[tuple[str, CellXY]] = []
        for child in off_path_children:
            sequence.extend((("append", child), ("return", child), ("append", cell)))
        if path_child is not None:
            sequence.extend((("append", path_child), ("finish", path_child)))
        operations.extend(reversed(sequence))
    return _FrontierSegmentPath(tuple(ordered))


def _grid_step_length_cells(first: CellXY, second: CellXY) -> float:
    dx = abs(second.x - first.x)
    dy = abs(second.y - first.y)
    if max(dx, dy) != 1 or dx + dy == 0:
        raise RuntimeError("frontier contour traversal contains a non-neighbor jump")
    return math.sqrt(2.0) if dx == 1 and dy == 1 else 1.0


def _unique_segment_cells(segment: _FrontierSegmentPath) -> tuple[CellXY, ...]:
    return tuple(dict.fromkeys(segment.cells))


def _segment_arc_positions_m(
    segment: _FrontierSegmentPath,
    resolution_m: float,
) -> tuple[tuple[float, ...], float]:
    if not math.isfinite(resolution_m) or resolution_m <= 0.0:
        raise ValueError("frontier resolution must be finite and positive")
    positions = [0.0]
    for first, second in zip(segment.cells, segment.cells[1:]):
        positions.append(positions[-1] + _grid_step_length_cells(first, second) * resolution_m)
    total = positions[-1]
    if segment.closed and len(segment.cells) > 1:
        total += _grid_step_length_cells(segment.cells[-1], segment.cells[0]) * resolution_m
    return tuple(positions), total


def _segment_arc_length_m(segment: _FrontierSegmentPath, resolution_m: float) -> float:
    return _segment_arc_positions_m(segment, resolution_m)[1]


def _regular_anchor_count(segment_length_m: float) -> int:
    if not math.isfinite(segment_length_m) or segment_length_m < 0.0:
        raise ValueError("segment arc length must be finite and nonnegative")
    return 1 if segment_length_m <= 10.0 else 2 if segment_length_m <= 30.0 else 3


def _arc_length_anchor_cells(
    segment: _FrontierSegmentPath,
    count: int,
    *,
    resolution_m: float,
) -> tuple[CellXY, ...]:
    if count < 0:
        raise ValueError("arc-length anchor count must be nonnegative")
    if count == 0 or not segment.cells:
        return ()
    positions, total = _segment_arc_positions_m(segment, resolution_m)
    if total == 0.0:
        return tuple(segment.cells[0] for _ in range(count))
    searchable = list(zip(positions, segment.cells))
    if segment.closed:
        searchable.append((total, segment.cells[0]))
    anchors: list[CellXY] = []
    for index in range(count):
        target = total * (index + 1) / (count + 1)
        anchors.append(next(cell for position, cell in searchable if position >= target))
    return tuple(anchors)


def _local_unknown_vector(mask: np.ndarray, cell: CellXY, *, radius_cells: int = 1) -> tuple[float, float]:
    sum_x = 0.0
    sum_y = 0.0
    count = 0
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            x, y = cell.x + dx, cell.y + dy
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and not mask[y, x]:
                distance = math.hypot(dx, dy)
                if distance > 0.0:
                    sum_x += dx / distance
                    sum_y += dy / distance
                    count += 1
    return (sum_x / count, sum_y / count) if count else (0.0, 0.0)


def _segment_normal(mask: np.ndarray, segment: Iterable[CellXY]) -> tuple[float, float, float]:
    units: list[tuple[float, float]] = []
    for cell in segment:
        dx, dy = _local_unknown_vector(mask, cell)
        magnitude = math.hypot(dx, dy)
        if magnitude > 0.0:
            units.append((dx / magnitude, dy / magnitude))
    if not units:
        return 1.0, 0.0, 0.0
    mean_x = sum(item[0] for item in units) / len(units)
    mean_y = sum(item[1] for item in units) / len(units)
    confidence = min(math.hypot(mean_x, mean_y), 1.0)
    if confidence == 0.0:
        return 1.0, 0.0, 0.0
    return mean_x / confidence, mean_y / confidence, confidence


def _nearest_valid_cell(
    state: ObservedMapState,
    landing_mask: np.ndarray,
    center: CellXY,
    *,
    radius_cells: int,
) -> CellXY | None:
    offsets = sorted(
        ((dx, dy) for dy in range(-radius_cells, radius_cells + 1) for dx in range(-radius_cells, radius_cells + 1)),
        key=lambda item: (item[0] ** 2 + item[1] ** 2, item[1], item[0]),
    )
    for dx, dy in offsets:
        cell = CellXY(center.x + dx, center.y + dy)
        if state.geometry.in_bounds(cell) and landing_mask[cell.y, cell.x]:
            return cell
    return None


def _nearest_valid_gain_cell(
    state: ObservedMapState,
    landing_mask: np.ndarray,
    center: CellXY,
    *,
    radius_cells: int,
    min_clearance_m: float,
    prior: LowResolutionPrior,
    recommended_theta: float,
    observed_gain_blocker_mask: np.ndarray,
    observed_clearance_support: _ObservedClearanceSupport | None = None,
) -> CellXY | None:
    clearance_support = observed_clearance_support or _build_observed_clearance_support(state)
    offsets = sorted(
        ((dx, dy) for dy in range(-radius_cells, radius_cells + 1) for dx in range(-radius_cells, radius_cells + 1)),
        key=lambda item: (item[0] ** 2 + item[1] ** 2, item[1], item[0]),
    )
    for dx, dy in offsets:
        cell = CellXY(center.x + dx, center.y + dy)
        if not state.geometry.in_bounds(cell) or not landing_mask[cell.y, cell.x]:
            continue
        if not _has_minimum_observed_clearance_with_support(
            state,
            cell,
            min_clearance_m,
            clearance_support,
        ):
            continue
        if _estimate_observed_only_gain_with_blockers(
            state,
            prior,
            cell,
            recommended_theta,
            observed_gain_blocker_mask,
        ).potential_gain_cells <= 0:
            continue
        return cell
    return None


def _has_minimum_observed_clearance(
    state: ObservedMapState,
    cell: CellXY,
    min_clearance_m: float,
) -> bool:
    return _has_minimum_observed_clearance_with_support(
        state,
        cell,
        min_clearance_m,
        _build_observed_clearance_support(state),
    )


def _has_minimum_observed_clearance_with_support(
    state: ObservedMapState,
    cell: CellXY,
    min_clearance_m: float,
    clearance_support: _ObservedClearanceSupport,
) -> bool:
    if not state.geometry.in_bounds(cell):
        return False
    return bool(clearance_support.clearance_m[cell.y, cell.x] >= min_clearance_m)


def _squared_distance_transform_1d(costs: np.ndarray) -> np.ndarray:
    size = len(costs)
    finite_sites = np.flatnonzero(np.isfinite(costs))
    if len(finite_sites) == 0:
        return np.full((size,), np.inf, dtype=np.float64)

    sites = np.empty((len(finite_sites),), dtype=np.intp)
    boundaries = np.empty((len(finite_sites) + 1,), dtype=np.float64)
    envelope_index = 0
    sites[0] = finite_sites[0]
    boundaries[0] = -np.inf
    boundaries[1] = np.inf
    for finite_site in finite_sites[1:]:
        site = int(finite_site)
        while True:
            previous = int(sites[envelope_index])
            separation = (
                (float(costs[site]) + site * site)
                - (float(costs[previous]) + previous * previous)
            ) / (2.0 * (site - previous))
            if separation > boundaries[envelope_index]:
                break
            envelope_index -= 1
        envelope_index += 1
        sites[envelope_index] = site
        boundaries[envelope_index] = separation
        boundaries[envelope_index + 1] = np.inf

    transformed = np.empty((size,), dtype=np.float64)
    envelope_index = 0
    for coordinate in range(size):
        while boundaries[envelope_index + 1] < coordinate:
            envelope_index += 1
        site = int(sites[envelope_index])
        delta = coordinate - site
        transformed[coordinate] = delta * delta + float(costs[site])
    return transformed


def _squared_euclidean_distance_transform(blocker_mask: np.ndarray) -> np.ndarray:
    initial = np.where(blocker_mask, 0.0, np.inf)
    horizontal = np.empty(blocker_mask.shape, dtype=np.float64)
    for y in range(blocker_mask.shape[0]):
        horizontal[y] = _squared_distance_transform_1d(initial[y])
    transformed = np.empty(blocker_mask.shape, dtype=np.float64)
    for x in range(blocker_mask.shape[1]):
        transformed[:, x] = _squared_distance_transform_1d(horizontal[:, x])
    return transformed


def _build_observed_clearance_support(state: ObservedMapState) -> _ObservedClearanceSupport:
    blocker_mask = state.observed_mask & (
        state.obstacle | (state.slope_deg > 30.0) | (state.traversability < 0.5)
    )
    geometry = state.geometry
    blocker_distance_m = (
        np.sqrt(_squared_euclidean_distance_transform(blocker_mask))
        * geometry.resolution_m
    )
    x_coordinates = np.arange(geometry.width, dtype=np.float64)
    y_coordinates = np.arange(geometry.height, dtype=np.float64)
    x_boundary_m = np.minimum(
        x_coordinates + 0.5,
        geometry.width - x_coordinates - 0.5,
    ) * geometry.resolution_m
    y_boundary_m = np.minimum(
        y_coordinates + 0.5,
        geometry.height - y_coordinates - 0.5,
    ) * geometry.resolution_m
    boundary_m = np.minimum(y_boundary_m[:, None], x_boundary_m[None, :])
    clearance_m = np.minimum(boundary_m, blocker_distance_m)
    clearance_m.setflags(write=False)
    return _ObservedClearanceSupport(clearance_m=clearance_m)


def _observed_clearance_m(state: ObservedMapState, cell: CellXY) -> float:
    return _observed_clearance_m_with_support(
        state,
        cell,
        _build_observed_clearance_support(state),
    )


def _observed_clearance_m_with_support(
    state: ObservedMapState,
    cell: CellXY,
    clearance_support: _ObservedClearanceSupport,
) -> float:
    return float(clearance_support.clearance_m[cell.y, cell.x])


def _region_ratios(
    state: ObservedMapState, prior: LowResolutionPrior, cell: CellXY
) -> tuple[int, int, float, float]:
    low_height, low_width = prior.channels.shape[1:]
    low_x = min(low_width - 1, int(cell.x * low_width / state.geometry.width))
    low_y = min(low_height - 1, int(cell.y * low_height / state.geometry.height))
    x0 = int(low_x * state.geometry.width / low_width)
    x1 = int((low_x + 1) * state.geometry.width / low_width)
    y0 = int(low_y * state.geometry.height / low_height)
    y1 = int((low_y + 1) * state.geometry.height / low_height)
    observed = state.observed_mask[y0:y1, x0:x1]
    coverage = float(np.mean(observed, dtype=np.float64)) if observed.size else 0.0
    return low_y, low_x, coverage, 1.0 - coverage


def _feature_rows(
    candidates: list[_RawCandidate],
    pose: PoseXYTheta,
    state: ObservedMapState,
    segment_count: int,
) -> tuple[np.ndarray, list[float]]:
    rows = np.zeros((len(candidates), 22), dtype=np.float32)
    max_gain = max((candidate.gain.potential_gain_cells for candidate in candidates), default=1)
    max_value = max((candidate.gain.value_gain for candidate in candidates), default=1.0e-6)
    max_reachability_cost = max(
        (candidate.reachability_cost_norm for candidate in candidates),
        default=0.0,
    )
    priorities: list[float] = []
    for index, candidate in enumerate(candidates):
        dx = candidate.cell.x - pose.cell.x
        dy = candidate.cell.y - pose.cell.y
        bearing = math.atan2(dy, dx) if dx or dy else pose.theta
        gain_norm = candidate.gain.potential_gain_cells / max(1, max_gain)
        value_norm = candidate.gain.value_gain / max(1.0e-6, max_value)
        coverage_gain_norm = candidate.gain.potential_gain_cells / max(1, len(candidate.gain.visible_footprint))
        reachability_cost_norm = (
            candidate.reachability_cost_norm / max_reachability_cost
            if max_reachability_cost > 0.0
            else 0.0
        )
        row = rows[index]
        row[:] = (
            candidate.cell.x / max(state.geometry.width - 1, 1),
            candidate.cell.y / max(state.geometry.height - 1, 1),
            candidate.distance_norm,
            math.sin(bearing),
            math.cos(bearing),
            coverage_gain_norm,
            gain_norm,
            value_norm,
            candidate.segment_id / max(segment_count - 1, 1),
            min(candidate.segment_length_m / max(state.geometry.width * state.geometry.resolution_m, 1.0), 1.0),
            candidate.normal_y,
            candidate.normal_x,
            candidate.normal_confidence,
            float(candidate.generation_mode),
            math.sin(candidate.recommended_theta),
            math.cos(candidate.recommended_theta),
            float(state.traversability[candidate.cell.y, candidate.cell.x]),
            candidate.clearance_norm,
            reachability_cost_norm,
            candidate.region_coverage_ratio,
            candidate.region_unknown_ratio,
            1.0,
        )
        priorities.append(
            0.5 * coverage_gain_norm
            + 0.3 * value_norm
            - 0.1 * candidate.distance_norm
            - 0.1 * reachability_cost_norm
        )
    return rows, priorities
