"""Observed-only Stage 2 frontier segment and landing action generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np

from lunar_exploration_ppo.env.map_state import ObservedMapState
from lunar_exploration_ppo.env.reachability import reachable_component
from lunar_exploration_ppo.env.scenario import LowResolutionPrior
from lunar_exploration_ppo.env.sensor_model import ray_cells_from_world
from lunar_exploration_ppo.utils.geometry import CellXY, PoseXYTheta


SEGMENT_POLICY = "binary_regular_normal_or_irregular_gain_sampling/v1"
POTENTIAL_GAIN_SOURCE = "endpoint_fov_los_visible_unknown_gain/v1"
TOP_M_SELECTION_POLICY = "score_first_top_m/v1"
CANDIDATE_PRIORITY_SOURCE = "score_first_gain_value_cost_priority/v1"


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
class _RawCandidate:
    cell: CellXY
    segment_id: int
    segment_length_m: float
    normal_x: float
    normal_y: float
    normal_confidence: float
    generation_mode: int
    recommended_theta: float
    gain: GainEstimate
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
    ray_count = int(round(fov_deg / ray_angle_step_deg)) + 1
    footprint: set[CellXY] = set()
    origin = geometry.cell_to_world_center(candidate)
    observed_blocker = observed_state.observed_mask & (
        observed_state.obstacle | (observed_state.slope_deg > 30.0)
    )
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
        component = reachable_component(observed_state.observed_safe_mask, pose.cell)
        frontier_mask = _frontier_mask(observed_state.observed_mask)
        safe_frontier = frontier_mask & _observed_safe_cells(observed_state, self.traversability_threshold)
        reachable_frontier = safe_frontier & component
        contour_frontier = reachable_frontier & _frontier_contour_mask(observed_state.observed_mask)
        segments = _frontier_segment_paths(contour_frontier)
        raw: list[_RawCandidate] = []
        diagonal_m = max(math.hypot(geometry.width - 1, geometry.height - 1) * geometry.resolution_m, 1.0)
        for segment_id, segment_path in enumerate(segments):
            segment = _unique_segment_cells(segment_path)
            normal_x, normal_y, confidence = _segment_normal(observed_state.observed_mask, segment)
            segment_length_m = _segment_arc_length_m(segment_path, geometry.resolution_m)
            if confidence >= self.normal_confidence_threshold:
                candidates = self._regular_candidates(
                    observed_state,
                    component,
                    segment_path,
                    normal_x,
                    normal_y,
                    segment_length_m,
                )
                mode = 0
            else:
                candidates = self._irregular_candidates(
                    observed_state,
                    component,
                    prior,
                    segment_path,
                )
                mode = 1
            for cell, theta in candidates[: self.max_candidates_per_segment]:
                if not _valid_landing_cell(observed_state, component, cell, self.traversability_threshold):
                    continue
                clearance_m = _observed_clearance_m(observed_state, cell)
                if clearance_m < self.min_clearance_m:
                    continue
                gain = estimate_observed_only_gain(observed_state, prior, cell, theta)
                if gain.potential_gain_cells <= 0:
                    continue
                dx = (cell.x - pose.cell.x) * geometry.resolution_m
                dy = (cell.y - pose.cell.y) * geometry.resolution_m
                distance_norm = math.hypot(dx, dy) / diagonal_m
                low_y, low_x, coverage_ratio, unknown_ratio = _region_ratios(observed_state, prior, cell)
                del low_y, low_x
                raw.append(
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
                        original_rank=len(raw),
                    )
                )
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
            "regular_segment_count": sum(
                _segment_normal(observed_state.observed_mask, _unique_segment_cells(segment))[2]
                >= self.normal_confidence_threshold
                for segment in segments
            ),
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
        component: np.ndarray,
        segment: _FrontierSegmentPath,
        normal_x: float,
        normal_y: float,
        segment_length_m: float,
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
        for anchor in anchors:
            ideal = CellXY(
                round(anchor.x - normal_x * standoff_cells),
                round(anchor.y - normal_y * standoff_cells),
            )
            corrected = _nearest_valid_cell(state, component, ideal, radius_cells=8, threshold=self.traversability_threshold)
            if corrected is not None and corrected not in (cell for cell, _ in result):
                result.append((corrected, theta))
        return result

    def _irregular_candidates(
        self,
        state: ObservedMapState,
        component: np.ndarray,
        prior: LowResolutionPrior,
        segment: _FrontierSegmentPath,
    ) -> list[tuple[CellXY, float]]:
        sample_count = min(12, len(segment.cells))
        anchors = _arc_length_anchor_cells(
            segment,
            sample_count,
            resolution_m=state.geometry.resolution_m,
        )
        sampled: list[tuple[int, CellXY, float]] = []
        for anchor in anchors:
            corrected = _nearest_valid_cell(state, component, anchor, radius_cells=4, threshold=self.traversability_threshold)
            if corrected is None:
                continue
            dx, dy = _local_unknown_vector(state.observed_mask, corrected, radius_cells=4)
            theta = math.atan2(dy, dx) if dx or dy else 0.0
            gain = estimate_observed_only_gain(state, prior, corrected, theta)
            sampled.append((gain.potential_gain_cells, corrected, theta))
        sampled.sort(key=lambda item: -item[0])
        result: list[tuple[CellXY, float]] = []
        for gain, cell, theta in sampled:
            if gain <= 0 or any(existing == cell for existing, _ in result):
                continue
            result.append((cell, theta))
            if len(result) == self.max_candidates_per_segment:
                break
        return result


def _observed_safe_cells(state: ObservedMapState, threshold: float) -> np.ndarray:
    return (
        state.observed_mask
        & ~state.obstacle
        & (state.slope_deg <= 30.0)
        & (state.traversability >= threshold)
        & state.observed_safe_mask
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
    component: np.ndarray,
    center: CellXY,
    *,
    radius_cells: int,
    threshold: float,
) -> CellXY | None:
    offsets = sorted(
        ((dx, dy) for dy in range(-radius_cells, radius_cells + 1) for dx in range(-radius_cells, radius_cells + 1)),
        key=lambda item: (item[0] ** 2 + item[1] ** 2, item[1], item[0]),
    )
    for dx, dy in offsets:
        cell = CellXY(center.x + dx, center.y + dy)
        if _valid_landing_cell(state, component, cell, threshold):
            return cell
    return None


def _observed_clearance_m(state: ObservedMapState, cell: CellXY) -> float:
    geometry = state.geometry
    boundary = min(
        (cell.x + 0.5) * geometry.resolution_m,
        (geometry.width - cell.x - 0.5) * geometry.resolution_m,
        (cell.y + 0.5) * geometry.resolution_m,
        (geometry.height - cell.y - 0.5) * geometry.resolution_m,
    )
    blockers = state.observed_mask & (state.obstacle | (state.slope_deg > 30.0) | (state.traversability < 0.5))
    ys, xs = np.nonzero(blockers)
    if len(xs) == 0:
        return boundary
    distance = float(np.min(np.hypot(xs - cell.x, ys - cell.y))) * geometry.resolution_m
    return min(boundary, distance)


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
