from __future__ import annotations

from collections import Counter
import inspect
import math

import numpy as np
import pytest

from lunar_exploration_ppo.env import frontier as frontier_module
from lunar_exploration_ppo.env.frontier import FrontierGenerator
from lunar_exploration_ppo.env.env import select_conservative_frontier_candidate_index, select_conservative_rule_action
from lunar_exploration_ppo.env.map_state import ObservedMapState
from lunar_exploration_ppo.env.scenario import LowResolutionPrior
from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry, PoseXYTheta
from lunar_exploration_ppo.policy.observation import PolicyObservation


def _prior(low_size: int = 8) -> LowResolutionPrior:
    channels = np.zeros((7, low_size, low_size), dtype=np.float32)
    channels[1] = np.linspace(0.1, 1.0, low_size, dtype=np.float32)[None, :]
    channels[3] = 1.0
    return LowResolutionPrior(channels, 4.0, "deployment_prior/v1")


def _half_plane_state(size: int = 64) -> ObservedMapState:
    state = ObservedMapState.empty(GridGeometry(size, size, 0.5))
    state.observed_mask[:, : size // 2] = True
    state.confidence[:, : size // 2] = 1.0
    state.traversability[:, : size // 2] = 1.0
    state.observed_safe_mask[2:-2, 2 : size // 2] = True
    return state


def _legacy_exact_observed_clearance_map(state: ObservedMapState) -> np.ndarray:
    geometry = state.geometry
    blockers = state.observed_mask & (
        state.obstacle | (state.slope_deg > 30.0) | (state.traversability < 0.5)
    )
    blocker_ys, blocker_xs = np.nonzero(blockers)
    clearance_m = np.empty(blockers.shape, dtype=np.float64)
    for y in range(geometry.height):
        for x in range(geometry.width):
            boundary_m = min(
                (x + 0.5) * geometry.resolution_m,
                (geometry.width - x - 0.5) * geometry.resolution_m,
                (y + 0.5) * geometry.resolution_m,
                (geometry.height - y - 0.5) * geometry.resolution_m,
            )
            if len(blocker_xs) == 0:
                clearance_m[y, x] = boundary_m
                continue
            blocker_distance_m = float(
                np.min(np.hypot(blocker_xs - x, blocker_ys - y))
            ) * geometry.resolution_m
            clearance_m[y, x] = min(boundary_m, blocker_distance_m)
    return clearance_m


def _raw_candidate_for_reachability_cost(
    *,
    x: int,
    cost: float,
    distance_norm: float = 0.2,
) -> object:
    footprint = tuple(CellXY(index, 0) for index in range(10))
    return frontier_module._RawCandidate(
        cell=CellXY(x, 1),
        segment_id=0,
        segment_length_m=1.0,
        normal_x=1.0,
        normal_y=0.0,
        normal_confidence=1.0,
        generation_mode=0,
        recommended_theta=0.0,
        gain=frontier_module.GainEstimate(
            visible_unknown=footprint[:5],
            visible_footprint=footprint,
            potential_gain_cells=5,
            value_gain=10.0,
        ),
        distance_norm=distance_norm,
        clearance_norm=1.0,
        reachability_cost_norm=cost,
        region_coverage_ratio=0.5,
        region_unknown_ratio=0.5,
        original_rank=x,
    )


def test_frontier_signature_and_candidates_use_observed_deployment_evidence_only() -> None:
    signature = inspect.signature(FrontierGenerator.extract)
    text = repr(FrontierGenerator.extract.__annotations__).lower()
    assert tuple(signature.parameters) == ("self", "observed_state", "prior", "pose")
    assert "truth" not in text and "coverable" not in text
    state = _half_plane_state()
    generator = FrontierGenerator(top_m=16)
    pose = PoseXYTheta(CellXY(10, 32), 0.0)
    first = generator.extract(state, _prior(), pose)
    state.height[~state.observed_mask] = 1.0e30
    state.obstacle[~state.observed_mask] = True
    state.slope_deg[~state.observed_mask] = 89.0
    state.traversability[~state.observed_mask] = 0.0
    repeated = generator.extract(state, _prior(), pose)
    assert first.cells == repeated.cells
    assert np.array_equal(first.frontier_features, repeated.frontier_features)
    assert first.diagnostics == repeated.diagnostics
    assert all(state.observed_mask[cell.y, cell.x] for cell in first.cells)
    assert all(state.observed_safe_mask[cell.y, cell.x] for cell in first.cells)


def _safe_standoff_frontier_fixture(
    *,
    occluded: bool = False,
    no_safe_landing: bool = False,
) -> tuple[ObservedMapState, LowResolutionPrior, PoseXYTheta]:
    state = ObservedMapState.empty(GridGeometry(48, 48, 0.5))
    state.observed_mask[:, :41] = True
    state.confidence[:, :41] = 1.0
    state.traversability[:, :41] = 1.0
    state.traversability[:, 39] = 0.0
    if not no_safe_landing:
        state.observed_safe_mask[2:-2, 2:39] = True
    if occluded:
        state.obstacle[:, 35] = True
        state.observed_safe_mask[:, 35:] = False
    return state, _prior(), PoseXYTheta(CellXY(30, 24), 0.0)


def test_frontier_fallback_recovers_safe_standoff_across_non_occluding_buffer() -> None:
    state, prior, pose = _safe_standoff_frontier_fixture()
    generator = FrontierGenerator(top_m=16)

    actions = generator.extract(state, prior, pose)

    assert actions.candidate_count >= 1
    component = frontier_module.reachable_component(state.observed_safe_mask, pose.cell)
    for index, cell in enumerate(actions.cells):
        theta = math.atan2(
            float(actions.frontier_features[index, 14]),
            float(actions.frontier_features[index, 15]),
        )
        assert state.observed_mask[cell.y, cell.x]
        assert state.observed_safe_mask[cell.y, cell.x]
        assert state.traversability[cell.y, cell.x] >= generator.traversability_threshold
        assert component[cell.y, cell.x]
        assert frontier_module._observed_clearance_m(state, cell) >= generator.min_clearance_m
        assert frontier_module.estimate_observed_only_gain(state, prior, cell, theta).potential_gain_cells > 0
    assert actions.diagnostics["fallback_activated"] is True
    assert actions.diagnostics["fallback_segment_count"] == 1
    assert math.isfinite(float(actions.diagnostics["fallback_segment_count"]))

    occluded_state, _, _ = _safe_standoff_frontier_fixture(occluded=True)
    occluded = generator.extract(occluded_state, prior, pose)
    assert occluded.cells == ()
    assert occluded.diagnostics["fallback_activated"] is True
    assert occluded.diagnostics["fallback_segment_count"] == 1

    no_safe_state, _, _ = _safe_standoff_frontier_fixture(no_safe_landing=True)
    no_safe = generator.extract(no_safe_state, prior, pose)
    assert no_safe.cells == ()
    assert no_safe.diagnostics["fallback_activated"] is True
    assert no_safe.diagnostics["fallback_segment_count"] == 1


def test_regular_segment_uses_standoff_anchors_and_outward_recommended_theta() -> None:
    action_set = FrontierGenerator(top_m=16).extract(
        _half_plane_state(), _prior(), PoseXYTheta(CellXY(10, 32), 0.0)
    )
    valid = action_set.frontier_features[action_set.candidate_mask]
    assert action_set.candidate_count == 3
    assert np.all(valid[:, 13] == 0.0)
    assert np.all(valid[:, 12] >= 0.6)
    assert np.all(valid[:, 14] == 0.0)
    assert np.allclose(valid[:, 15], 1.0, atol=1e-6)
    assert all(cell.x < 31 for cell in action_set.cells)
    assert all(valid[:, 5] > 0.0)


def _regular_frontier_correction_fixture() -> tuple[ObservedMapState, LowResolutionPrior, PoseXYTheta]:
    rows = (
        "...........................",
        "...........................",
        "...........................",
        "...........................",
        "...........................",
        ".......bbbbbb..............",
        "......bnnnnnnb.............",
        ".....bnssssssnb............",
        ".....bnssssssnb............",
        ".....bnsssssssnb...........",
        ".....bnsssssssnbb..........",
        ".....bnssssssssnn..........",
        ".....bnsssssssssssssnb.....",
        ".....bnsssssssssssssnb.....",
        ".....bnssssssssnnnnnnb.....",
        ".....bnsssssssnbbbbbb......",
        ".....bnsssssssnb...........",
        ".....bnsssssssnb...........",
        "......bnssssssnb...........",
        "......bnsssssnnb...........",
        ".......bnnnnnbb............",
        "........bbbbb..............",
        "...........................",
        "...........................",
        "...........................",
        "...........................",
        "...........................",
        "...........................",
        "...........................",
        "...........................",
    )
    assert len(rows) == 30
    assert all(len(row) == 27 for row in rows)
    state = ObservedMapState.empty(GridGeometry(27, 30, 0.5))
    for y, row in enumerate(rows):
        for x, marker in enumerate(row):
            if marker == ".":
                continue
            state.observed_mask[y, x] = True
            state.confidence[y, x] = 1.0
            if marker == "s":
                state.observed_safe_mask[y, x] = True
                state.traversability[y, x] = 1.0
            elif marker == "b":
                state.obstacle[y, x] = True

    prior = LowResolutionPrior(
        channels=np.zeros((7, 8, 8), dtype=np.float32),
        resolution_m=4.0,
        value_prior_source="deployment_neutral/v1",
    )
    pose = PoseXYTheta(CellXY(12, 19), -1.9389368295669556)
    return state, prior, pose


def test_regular_frontier_correction_skips_zero_gain_safe_cell() -> None:
    state, prior, pose = _regular_frontier_correction_fixture()
    actions = FrontierGenerator(top_m=16).extract(state, prior, pose)

    assert actions.candidate_count >= 1
    component = frontier_module.reachable_component(state.observed_safe_mask, pose.cell)
    for index, cell in enumerate(actions.cells):
        assert state.observed_safe_mask[cell.y, cell.x]
        assert component[cell.y, cell.x]
        theta = math.atan2(
            float(actions.frontier_features[index, 14]),
            float(actions.frontier_features[index, 15]),
        )
        assert frontier_module.estimate_observed_only_gain(state, prior, cell, theta).potential_gain_cells > 0


def test_regular_frontier_extraction_builds_whole_map_safety_and_clearance_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, prior, pose = _regular_frontier_correction_fixture()
    safe_calls = 0
    clearance_calls = 0
    real_observed_safe_cells = frontier_module._observed_safe_cells
    real_observed_clearance_m = frontier_module._observed_clearance_m

    def count_observed_safe_cells(
        counted_state: ObservedMapState,
        threshold: float,
    ) -> np.ndarray:
        nonlocal safe_calls
        safe_calls += 1
        return real_observed_safe_cells(counted_state, threshold)

    def count_observed_clearance_m(counted_state: ObservedMapState, cell: CellXY) -> float:
        nonlocal clearance_calls
        clearance_calls += 1
        return real_observed_clearance_m(counted_state, cell)

    monkeypatch.setattr(frontier_module, "_observed_safe_cells", count_observed_safe_cells)
    monkeypatch.setattr(frontier_module, "_observed_clearance_m", count_observed_clearance_m)

    actions = FrontierGenerator(top_m=16).extract(state, prior, pose)

    assert actions.candidate_count >= 1
    assert safe_calls == 1
    assert clearance_calls == 0


def test_regular_frontier_correction_reuses_observed_gain_blockers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, prior, pose = _regular_frontier_correction_fixture()
    blocker_build_calls = 0
    gain_core_calls = 0
    gain_core_results: list[int] = []
    real_builder = frontier_module._build_observed_gain_blocker_mask
    real_gain_core = frontier_module._estimate_observed_only_gain_with_blockers

    def count_builder(counted_state: ObservedMapState) -> np.ndarray:
        nonlocal blocker_build_calls
        blocker_build_calls += 1
        return real_builder(counted_state)

    def count_gain_core(*args: object, **kwargs: object) -> frontier_module.GainEstimate:
        nonlocal gain_core_calls
        gain_core_calls += 1
        result = real_gain_core(*args, **kwargs)
        gain_core_results.append(result.potential_gain_cells)
        return result

    monkeypatch.setattr(frontier_module, "_build_observed_gain_blocker_mask", count_builder)
    monkeypatch.setattr(frontier_module, "_estimate_observed_only_gain_with_blockers", count_gain_core)

    actions = FrontierGenerator(top_m=16).extract(state, prior, pose)

    assert actions.candidate_count >= 1
    assert blocker_build_calls == 1

    blocker_build_calls = 0
    gain_core_calls = 0
    gain_core_results.clear()
    landing_mask = frontier_module._observed_safe_cells(state, 0.50) & frontier_module.reachable_component(
        state.observed_safe_mask,
        pose.cell,
    )
    blocker_mask = frontier_module._build_observed_gain_blocker_mask(state)
    corrected = frontier_module._nearest_valid_gain_cell(
        state,
        landing_mask,
        CellXY(18, 4),
        radius_cells=10,
        min_clearance_m=0.5215874761,
        prior=prior,
        recommended_theta=-1.4406540010227697,
        observed_gain_blocker_mask=blocker_mask,
    )

    assert corrected is not None
    final_gain = frontier_module.estimate_observed_only_gain(
        state,
        prior,
        corrected,
        -1.4406540010227697,
    )
    correction_gain_results = gain_core_results[:-1]
    assert correction_gain_results[0] == 0
    assert correction_gain_results[-1] == final_gain.potential_gain_cells
    assert final_gain.potential_gain_cells > 0
    assert gain_core_calls > 1
    assert blocker_build_calls == 2


def test_observed_gain_private_core_matches_public_estimator() -> None:
    state, prior, pose = _regular_frontier_correction_fixture()
    blocker_mask = frontier_module._build_observed_gain_blocker_mask(state)

    for candidate, theta in (
        (pose.cell, pose.theta),
        (CellXY(11, 18), -math.pi / 2.0),
        (CellXY(15, 14), 0.0),
    ):
        assert frontier_module._estimate_observed_only_gain_with_blockers(
            state,
            prior,
            candidate,
            theta,
            blocker_mask,
        ) == frontier_module.estimate_observed_only_gain(state, prior, candidate, theta)

    with pytest.raises(ValueError, match="gain blocker mask"):
        frontier_module._estimate_observed_only_gain_with_blockers(
            state,
            prior,
            pose.cell,
            pose.theta,
            np.zeros(state.observed_mask.shape, dtype=np.uint8),
        )
    with pytest.raises(ValueError, match="gain blocker mask"):
        frontier_module._estimate_observed_only_gain_with_blockers(
            state,
            prior,
            pose.cell,
            pose.theta,
            np.zeros((1, 1), dtype=bool),
        )


def test_observed_gain_batch_matches_materialized_scalar_reference() -> None:
    state, prior, pose = _regular_frontier_correction_fixture()
    blocker_mask = frontier_module._build_observed_gain_blocker_mask(state)
    cells = (
        pose.cell,
        CellXY(11, 18),
        CellXY(15, 14),
        CellXY(19, 11),
    )
    thetas = (pose.theta, -math.pi / 2.0, 0.0, math.pi / 4.0)
    reference = tuple(
        frontier_module._estimate_observed_only_gain_with_blockers(
            state,
            prior,
            cell,
            theta,
            blocker_mask,
        )
        for cell, theta in zip(cells, thetas, strict=True)
    )

    actual = frontier_module._estimate_observed_only_gain_batch_with_blockers(
        state,
        prior,
        cells,
        thetas,
        blocker_mask,
    )

    assert len(actual) == len(reference)
    for compact, materialized in zip(actual, reference, strict=True):
        assert compact.potential_gain_cells == materialized.potential_gain_cells
        assert len(compact.visible_footprint) == len(materialized.visible_footprint)
        assert compact.value_gain == materialized.value_gain
    assert (
        frontier_module._estimate_observed_only_gain_batch_with_blockers(
            state,
            prior,
            (),
            (),
            blocker_mask,
        )
        == ()
    )


def test_local_minimum_clearance_predicate_matches_observed_clearance() -> None:
    def observed_state() -> ObservedMapState:
        state = ObservedMapState.empty(GridGeometry(7, 7, 0.5))
        state.observed_mask[:] = True
        state.observed_safe_mask[:] = True
        state.traversability[:] = 1.0
        return state

    cases: list[tuple[ObservedMapState, CellXY, float]] = []

    boundary = observed_state()
    cases.append((boundary, CellXY(0, 3), 0.25))

    obstacle = observed_state()
    obstacle.obstacle[3, 4] = True
    cases.extend(((obstacle, CellXY(3, 3), 0.5), (obstacle, CellXY(3, 3), 0.500001)))

    steep_slope = observed_state()
    steep_slope.slope_deg[5, 3] = math.nextafter(30.0, math.inf)
    cases.append((steep_slope, CellXY(3, 4), 0.5))

    low_traversability = observed_state()
    low_traversability.traversability[3, 5] = math.nextafter(0.5, -math.inf)
    cases.append((low_traversability, CellXY(4, 3), 0.5))

    no_blockers = observed_state()
    cases.append((no_blockers, CellXY(3, 3), 1.75))

    for state, cell, min_clearance_m in cases:
        expected = frontier_module._observed_clearance_m(state, cell) >= min_clearance_m
        assert frontier_module._has_minimum_observed_clearance(state, cell, min_clearance_m) is expected


def test_precomputed_clearance_map_matches_legacy_exact_semantics() -> None:
    no_blockers = ObservedMapState.empty(GridGeometry(5, 4, 0.5))
    no_blockers.observed_mask[:] = True
    no_blockers.traversability[:] = 1.0

    mixed = ObservedMapState.empty(GridGeometry(7, 6, 0.3))
    mixed.observed_mask[:] = True
    mixed.traversability[:] = 1.0
    mixed.obstacle[1, 1] = True
    mixed.slope_deg[4, 5] = math.nextafter(30.0, math.inf)
    mixed.traversability[2, 4] = math.nextafter(0.5, -math.inf)
    mixed.slope_deg[3, 2] = 30.0
    mixed.traversability[4, 2] = 0.5
    mixed.observed_mask[0, 6] = False
    mixed.obstacle[0, 6] = True

    dense = ObservedMapState.empty(GridGeometry(9, 8, 0.7))
    dense.observed_mask[:] = True
    dense.traversability[:] = 1.0
    yy, xx = np.mgrid[:8, :9]
    dense.obstacle[(3 * xx + 5 * yy) % 7 < 3] = True

    for state in (no_blockers, mixed, dense):
        expected = _legacy_exact_observed_clearance_map(state)
        support = frontier_module._build_observed_clearance_support(state)
        clearance_m = getattr(support, "clearance_m", None)

        assert isinstance(clearance_m, np.ndarray)
        assert clearance_m.dtype == np.float64
        assert clearance_m.shape == state.observed_mask.shape
        np.testing.assert_allclose(clearance_m, expected, rtol=0.0, atol=1.0e-12)
        for y in range(state.geometry.height):
            for x in range(state.geometry.width):
                assert math.isclose(
                    frontier_module._observed_clearance_m_with_support(
                        state,
                        CellXY(x, y),
                        support,
                    ),
                    expected[y, x],
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )


def test_cached_candidate_clearance_queries_do_not_scan_blocker_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = ObservedMapState.empty(GridGeometry(17, 17, 0.5))
    state.observed_mask[:] = True
    state.traversability[:] = 1.0
    yy, xx = np.mgrid[:17, :17]
    state.obstacle[(xx + 2 * yy) % 3 == 0] = True
    support = frontier_module._build_observed_clearance_support(state)
    hypot_calls = 0
    real_hypot = np.hypot

    def count_hypot(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal hypot_calls
        hypot_calls += 1
        return real_hypot(*args, **kwargs)

    monkeypatch.setattr(frontier_module.np, "hypot", count_hypot)

    for cell in (CellXY(1, 1), CellXY(8, 8), CellXY(15, 15)):
        frontier_module._observed_clearance_m_with_support(state, cell, support)
        frontier_module._has_minimum_observed_clearance_with_support(
            state,
            cell,
            0.5215874761,
            support,
        )

    assert hypot_calls == 0


def test_frontier_graph_arc_length_uses_axial_and_diagonal_step_metrics() -> None:
    axial_mask = np.zeros((7, 9), dtype=bool)
    axial_mask[3, 1:7] = True
    axial = frontier_module._frontier_segment_paths(axial_mask)
    assert len(axial) == 1
    assert axial[0].cells == tuple(CellXY(x, 3) for x in range(1, 7))
    assert math.isclose(frontier_module._segment_arc_length_m(axial[0], 0.5), 2.5)

    diagonal_mask = np.zeros((8, 8), dtype=bool)
    for index in range(1, 6):
        diagonal_mask[index, index] = True
    diagonal = frontier_module._frontier_segment_paths(diagonal_mask)
    assert len(diagonal) == 1
    assert diagonal[0].cells == tuple(CellXY(index, index) for index in range(1, 6))
    assert math.isclose(
        frontier_module._segment_arc_length_m(diagonal[0], 0.5),
        4.0 * math.sqrt(2.0) * 0.5,
    )


def test_frontier_graph_traversal_follows_l_and_backtrack_without_coordinate_jumps() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    expected = tuple(
        CellXY(x, y)
        for x, y in ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2))
    )
    for cell in expected:
        mask[cell.y, cell.x] = True
    paths = frontier_module._frontier_segment_paths(mask)
    assert len(paths) == 1
    assert paths[0].cells == expected
    assert all(
        math.hypot(second.x - first.x, second.y - first.y) in (1.0, math.sqrt(2.0))
        for first, second in zip(paths[0].cells, paths[0].cells[1:])
    )
    assert math.isclose(frontier_module._segment_arc_length_m(paths[0], 1.0), 6.0)


def test_frontier_graph_traverses_branches_with_deterministic_backtracking() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    junction = CellXY(1, 1)
    leaves = (CellXY(1, 0), CellXY(0, 1), CellXY(2, 1))
    for cell in (*leaves, junction):
        mask[cell.y, cell.x] = True
    paths = frontier_module._frontier_segment_paths(mask)
    assert len(paths) == 1
    assert not paths[0].closed
    assert paths[0].cells == (
        CellXY(1, 0),
        junction,
        CellXY(2, 1),
        junction,
        CellXY(0, 1),
    )
    assert math.isclose(frontier_module._segment_arc_length_m(paths[0], 1.0), 4.0)


def test_arc_length_anchor_count_freezes_threshold_sides_and_cumulative_positions() -> None:
    assert frontier_module._regular_anchor_count(math.nextafter(10.0, -math.inf)) == 1
    assert frontier_module._regular_anchor_count(10.0) == 1
    assert frontier_module._regular_anchor_count(math.nextafter(10.0, math.inf)) == 2
    assert frontier_module._regular_anchor_count(math.nextafter(30.0, -math.inf)) == 2
    assert frontier_module._regular_anchor_count(30.0) == 2
    assert frontier_module._regular_anchor_count(math.nextafter(30.0, math.inf)) == 3

    mask = np.zeros((5, 5), dtype=bool)
    ordered = tuple(
        CellXY(x, y)
        for x, y in ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2))
    )
    for cell in ordered:
        mask[cell.y, cell.x] = True
    path = frontier_module._frontier_segment_paths(mask)[0]
    assert frontier_module._arc_length_anchor_cells(path, 3, resolution_m=1.0) == (
        CellXY(2, 0),
        CellXY(2, 1),
        CellXY(1, 2),
    )


def test_landing_threshold_equality_is_valid_and_just_below_is_rejected() -> None:
    clearance_threshold = 0.5215874761
    traversability_threshold = 0.50
    state = ObservedMapState.empty(GridGeometry(3, 3, 2.0 * clearance_threshold))
    state.observed_mask[:] = True
    state.observed_safe_mask[:] = True
    state.traversability[:] = traversability_threshold
    component = np.ones(state.geometry.shape, dtype=bool)
    cell = CellXY(0, 1)
    assert frontier_module._valid_landing_cell(state, component, cell, traversability_threshold)
    assert frontier_module._observed_clearance_m(state, cell) == clearance_threshold

    state.traversability[cell.y, cell.x] = math.nextafter(
        traversability_threshold,
        -math.inf,
    )
    assert not frontier_module._valid_landing_cell(
        state,
        component,
        cell,
        traversability_threshold,
    )
    below = ObservedMapState.empty(
        GridGeometry(3, 3, 2.0 * math.nextafter(clearance_threshold, -math.inf))
    )
    below.observed_mask[:] = True
    below.observed_safe_mask[:] = True
    below.traversability[:] = traversability_threshold
    assert frontier_module._observed_clearance_m(below, cell) < clearance_threshold


def test_frontier_generator_accepts_only_exact_frozen_landing_thresholds() -> None:
    generator = FrontierGenerator(
        min_clearance_m=0.5215874761,
        traversability_threshold=0.50,
    )
    assert generator.min_clearance_m == 0.5215874761
    assert generator.traversability_threshold == 0.50


@pytest.mark.parametrize(
    "overrides",
    (
        {"min_clearance_m": math.nextafter(0.5215874761, -math.inf)},
        {"min_clearance_m": math.nextafter(0.5215874761, math.inf)},
        {"min_clearance_m": 0.0},
        {"min_clearance_m": math.nan},
        {"min_clearance_m": math.inf},
        {"traversability_threshold": math.nextafter(0.50, -math.inf)},
        {"traversability_threshold": math.nextafter(0.50, math.inf)},
        {"traversability_threshold": -0.1},
        {"traversability_threshold": 1.1},
        {"traversability_threshold": math.nan},
        {"traversability_threshold": math.inf},
    ),
)
def test_frontier_generator_rejects_landing_threshold_override(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="frozen"):
        FrontierGenerator(**overrides)


def test_irregular_segment_uses_local_gain_sampling() -> None:
    size = 64
    state = ObservedMapState.empty(GridGeometry(size, size, 0.5))
    yy, xx = np.mgrid[:size, :size]
    observed = (xx - 32) ** 2 + (yy - 32) ** 2 <= 18 ** 2
    state.observed_mask[:] = observed
    state.confidence[observed] = 1.0
    state.traversability[observed] = 1.0
    state.observed_safe_mask[:] = observed
    actions = FrontierGenerator(top_m=16).extract(state, _prior(), PoseXYTheta(CellXY(32, 32), 0.0))
    valid = actions.frontier_features[actions.candidate_mask]
    assert 1 <= actions.candidate_count <= 3
    assert np.all(valid[:, 13] == 1.0)
    assert np.all(valid[:, 12] < 0.6)
    assert np.isfinite(valid[:, 14:16]).all()


def test_irregular_extract_evaluates_each_unique_pool_landing_at_most_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    size = 64
    state = ObservedMapState.empty(GridGeometry(size, size, 0.5))
    yy, xx = np.mgrid[:size, :size]
    observed = (xx - 32) ** 2 + (yy - 32) ** 2 <= 18 ** 2
    state.observed_mask[:] = observed
    state.confidence[observed] = 1.0
    state.traversability[observed] = 1.0
    state.observed_safe_mask[:] = observed
    evaluated: list[CellXY] = []
    batch_calls = 0
    real_batch_gain = frontier_module._estimate_observed_only_gain_batch_with_blockers

    def count_batch_gain(
        counted_state: ObservedMapState,
        prior: LowResolutionPrior,
        cells: list[CellXY],
        thetas: list[float],
        blockers: np.ndarray,
        **kwargs: float,
    ) -> tuple[frontier_module._CompactGainEstimate, ...]:
        nonlocal batch_calls
        batch_calls += 1
        evaluated.extend(cells)
        return real_batch_gain(
            counted_state,
            prior,
            cells,
            thetas,
            blockers,
            **kwargs,
        )

    monkeypatch.setattr(
        frontier_module,
        "_estimate_observed_only_gain_batch_with_blockers",
        count_batch_gain,
    )
    monkeypatch.setattr(
        frontier_module,
        "_estimate_observed_only_gain_with_blockers",
        lambda *_args, **_kwargs: pytest.fail(
            "irregular default path must not call the scalar gain evaluator"
        ),
    )

    actions = FrontierGenerator(top_m=16).extract(
        state,
        _prior(),
        PoseXYTheta(CellXY(32, 32), 0.0),
    )
    counts = Counter(evaluated)

    assert actions.candidate_count >= 1
    assert actions.diagnostics["segment_count"] == 1
    assert actions.diagnostics["regular_segment_count"] == 0
    assert batch_calls == 1
    assert len(evaluated) <= 108
    assert counts and set(counts.values()) == {1}


def test_irregular_extract_does_not_drop_high_gain_redirected_landing_before_outer_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    size = 64
    state = ObservedMapState.empty(GridGeometry(size, size, 0.5))
    yy, xx = np.mgrid[:size, :size]
    observed = (xx - 32) ** 2 + (yy - 32) ** 2 <= 18 ** 2
    state.observed_mask[:] = observed
    state.confidence[observed] = 1.0
    state.traversability[observed] = 1.0
    state.observed_safe_mask[:] = observed

    redirected = CellXY(24, 18)
    local_gains = {
        CellXY(36, 16): 9,
        CellXY(27, 17): 8,
        CellXY(39, 17): 7,
    }

    def mixed_heading_target(
        _state: ObservedMapState,
        landing: CellXY,
        _targets: tuple[CellXY, ...],
        _blockers: np.ndarray,
        _local_theta: float,
    ) -> CellXY | None:
        if landing == redirected:
            return CellXY(redirected.x, redirected.y - 1)
        return None

    def mixed_gain_batch(
        _state: ObservedMapState,
        _prior_value: LowResolutionPrior,
        cells: list[CellXY],
        _thetas: list[float],
        _blockers: np.ndarray,
        **_kwargs: float,
    ) -> tuple[frontier_module.GainEstimate, ...]:
        return tuple(
            frontier_module.GainEstimate(
                visible_unknown=(),
                visible_footprint=(),
                potential_gain_cells=(gain := 100 if cell == redirected else local_gains.get(cell, 1)),
                value_gain=float(gain),
            )
            for cell in cells
        )

    monkeypatch.setattr(
        frontier_module,
        "_first_visible_irregular_heading_target",
        mixed_heading_target,
    )
    monkeypatch.setattr(
        frontier_module,
        "_estimate_observed_only_gain_batch_with_blockers",
        mixed_gain_batch,
    )

    actions = FrontierGenerator(top_m=16).extract(
        state,
        _prior(),
        PoseXYTheta(CellXY(32, 32), 0.0),
    )

    assert actions.candidate_count == 3
    assert redirected in actions.cells


def test_irregular_extract_does_not_let_low_gain_locals_evict_a_higher_gain_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    size = 64
    state = ObservedMapState.empty(GridGeometry(size, size, 0.5))
    yy, xx = np.mgrid[:size, :size]
    observed = (xx - 32) ** 2 + (yy - 32) ** 2 <= 18 ** 2
    state.observed_mask[:] = observed
    state.confidence[observed] = 1.0
    state.traversability[observed] = 1.0
    state.observed_safe_mask[:] = observed

    redirected = CellXY(24, 18)
    local_gains = {
        CellXY(36, 16): 100,
        CellXY(27, 17): 1,
        CellXY(39, 17): 1,
    }

    def mixed_heading_target(
        _state: ObservedMapState,
        landing: CellXY,
        _targets: tuple[CellXY, ...],
        _blockers: np.ndarray,
        _local_theta: float,
    ) -> CellXY | None:
        if landing == redirected:
            return CellXY(redirected.x, redirected.y - 1)
        return None

    def mixed_gain_batch(
        _state: ObservedMapState,
        _prior_value: LowResolutionPrior,
        cells: list[CellXY],
        _thetas: list[float],
        _blockers: np.ndarray,
        **_kwargs: float,
    ) -> tuple[frontier_module.GainEstimate, ...]:
        return tuple(
            frontier_module.GainEstimate(
                visible_unknown=(),
                visible_footprint=(),
                potential_gain_cells=(gain := 99 if cell == redirected else local_gains.get(cell, 1)),
                value_gain=float(gain),
            )
            for cell in cells
        )

    monkeypatch.setattr(
        frontier_module,
        "_first_visible_irregular_heading_target",
        mixed_heading_target,
    )
    monkeypatch.setattr(
        frontier_module,
        "_estimate_observed_only_gain_batch_with_blockers",
        mixed_gain_batch,
    )

    actions = FrontierGenerator(top_m=16).extract(
        state,
        _prior(),
        PoseXYTheta(CellXY(32, 32), 0.0),
    )

    assert actions.candidate_count == 3
    assert redirected in actions.cells


def test_irregular_extract_keeps_multiple_redirects_that_dominate_every_local_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    size = 64
    state = ObservedMapState.empty(GridGeometry(size, size, 0.5))
    yy, xx = np.mgrid[:size, :size]
    observed = (xx - 32) ** 2 + (yy - 32) ** 2 <= 18 ** 2
    state.observed_mask[:] = observed
    state.confidence[observed] = 1.0
    state.traversability[observed] = 1.0
    state.observed_safe_mask[:] = observed

    redirected_gains = {
        CellXY(24, 18): 100,
        CellXY(18, 24): 90,
    }
    local_gains = {
        CellXY(36, 16): 9,
        CellXY(27, 17): 8,
        CellXY(39, 17): 7,
    }

    def mixed_heading_target(
        _state: ObservedMapState,
        landing: CellXY,
        _targets: tuple[CellXY, ...],
        _blockers: np.ndarray,
        _local_theta: float,
    ) -> CellXY | None:
        if landing in redirected_gains:
            return CellXY(landing.x, landing.y - 1)
        return None

    def mixed_gain_batch(
        _state: ObservedMapState,
        _prior_value: LowResolutionPrior,
        cells: list[CellXY],
        _thetas: list[float],
        _blockers: np.ndarray,
        **_kwargs: float,
    ) -> tuple[frontier_module.GainEstimate, ...]:
        return tuple(
            frontier_module.GainEstimate(
                visible_unknown=(),
                visible_footprint=(),
                potential_gain_cells=(gain := redirected_gains.get(cell, local_gains.get(cell, 1))),
                value_gain=float(gain),
            )
            for cell in cells
        )

    monkeypatch.setattr(
        frontier_module,
        "_first_visible_irregular_heading_target",
        mixed_heading_target,
    )
    monkeypatch.setattr(
        frontier_module,
        "_estimate_observed_only_gain_batch_with_blockers",
        mixed_gain_batch,
    )

    actions = FrontierGenerator(top_m=16).extract(
        state,
        _prior(),
        PoseXYTheta(CellXY(32, 32), 0.0),
    )

    assert actions.candidate_count == 3
    assert set(redirected_gains) <= set(actions.cells)


def test_irregular_segment_evaluates_true_108_landing_pool_once_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = ObservedMapState.empty(GridGeometry(160, 40, 1.0))
    state.observed_mask[:] = True
    state.confidence[:] = 1.0
    state.traversability[:] = 1.0
    state.observed_safe_mask[:] = True
    segment = frontier_module._FrontierSegmentPath(
        cells=tuple(CellXY(x, 20) for x in range(20, 140)),
    )
    landing_mask = np.ones(state.geometry.shape, dtype=bool)
    evaluated: list[CellXY] = []

    def count_gain_batch(
        _state: ObservedMapState,
        _prior_value: LowResolutionPrior,
        cells: list[CellXY],
        _thetas: list[float],
        _blockers: np.ndarray,
        **_kwargs: float,
    ) -> tuple[frontier_module.GainEstimate, ...]:
        evaluated.extend(cells)
        return tuple(
            frontier_module.GainEstimate(
                visible_unknown=(),
                visible_footprint=(),
                potential_gain_cells=1,
                value_gain=1.0,
            )
            for _ in cells
        )

    monkeypatch.setattr(
        frontier_module,
        "_estimate_observed_only_gain_batch_with_blockers",
        count_gain_batch,
    )

    candidates = FrontierGenerator()._irregular_candidates(
        state,
        landing_mask,
        _prior(),
        segment,
        np.zeros(state.geometry.shape, dtype=bool),
    )

    counts = Counter(evaluated)
    assert len(candidates) == 3
    assert len(evaluated) == len(counts) == 108
    assert set(counts.values()) == {1}


def test_irregular_extract_binds_explicit_gain_and_theta_across_fallback_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scored_type = getattr(frontier_module, "_IrregularScoredCandidate", None)
    assert scored_type is not None
    state = ObservedMapState.empty(GridGeometry(24, 24, 1.0))
    state.observed_mask[:] = True
    state.confidence[:] = 1.0
    state.traversability[:] = 1.0
    state.observed_safe_mask[:] = True
    paths = (
        frontier_module._FrontierSegmentPath(cells=(CellXY(5, 5),)),
        frontier_module._FrontierSegmentPath(cells=(CellXY(18, 18),)),
    )
    bound = (
        (
            CellXY(5, 5),
            0.25,
            frontier_module.GainEstimate(
                visible_unknown=(),
                visible_footprint=tuple(CellXY(x, 6) for x in range(6)),
                potential_gain_cells=3,
                value_gain=7.0,
            ),
        ),
        (
            CellXY(18, 18),
            -1.0,
            frontier_module.GainEstimate(
                visible_unknown=(),
                visible_footprint=tuple(CellXY(x, 17) for x in range(5)),
                potential_gain_cells=5,
                value_gain=11.0,
            ),
        ),
    )
    scored_calls = 0

    def fixed_paths(_mask: np.ndarray) -> tuple[object, ...]:
        return paths

    def scored_candidates(
        _self: FrontierGenerator,
        _state: ObservedMapState,
        _landing_mask: np.ndarray,
        _prior_value: LowResolutionPrior,
        _segment: object,
        _blockers: np.ndarray,
    ) -> list[object]:
        nonlocal scored_calls
        call_index = scored_calls
        scored_calls += 1
        if call_index < len(paths):
            return []
        cell, theta, gain = bound[call_index - len(paths)]
        return [
            scored_type(
                cell=cell,
                recommended_theta=theta,
                gain=gain,
                redirected=False,
            )
        ]

    def forbid_recomputed_gain(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("irregular outer assembly must reuse the bound GainEstimate")

    monkeypatch.setattr(frontier_module, "_frontier_segment_paths", fixed_paths)
    monkeypatch.setattr(
        FrontierGenerator,
        "_irregular_scored_candidates",
        scored_candidates,
    )
    monkeypatch.setattr(
        frontier_module,
        "_estimate_observed_only_gain_with_blockers",
        forbid_recomputed_gain,
    )

    actions = FrontierGenerator(top_m=16).extract(
        state,
        _prior(),
        PoseXYTheta(CellXY(12, 12), 0.0),
    )

    assert scored_calls == 4
    assert actions.diagnostics["fallback_activated"] is True
    assert actions.diagnostics["segment_count"] == 2
    assert set(actions.cells) == {item[0] for item in bound}
    by_cell = {
        cell: actions.frontier_features[index]
        for index, cell in enumerate(actions.cells)
    }
    for cell, theta, gain in bound:
        row = by_cell[cell]
        assert math.isclose(math.atan2(float(row[14]), float(row[15])), theta, abs_tol=1.0e-7)
        assert math.isclose(
            float(row[5]),
            gain.potential_gain_cells / len(gain.visible_footprint),
            abs_tol=1.0e-7,
        )


def _blocked_irregular_heading_fixture(
) -> tuple[
    ObservedMapState,
    LowResolutionPrior,
    CellXY,
    CellXY,
    np.ndarray,
    np.ndarray,
]:
    state = ObservedMapState.empty(GridGeometry(15, 15, 1.0))
    state.observed_mask[:] = True
    state.confidence[:] = 1.0
    state.traversability[:] = 1.0
    state.observed_safe_mask[:] = True
    state.observed_mask[3:12, 11:15] = False
    state.observed_mask[6, 7] = False
    state.observed_mask[8, 7] = False
    state.obstacle[:, 10] = True
    landing = CellXY(7, 7)
    target = CellXY(7, 6)
    landing_mask = np.zeros(state.geometry.shape, dtype=bool)
    landing_mask[landing.y, landing.x] = state.planning_safe_mask[landing.y, landing.x]
    blocker_mask = frontier_module._build_observed_gain_blocker_mask(state)
    return state, _prior(), landing, target, landing_mask, blocker_mask


def test_irregular_segment_uses_visible_anchor_adjacent_unknown_heading() -> None:
    state, prior, landing, target, landing_mask, blocker_mask = (
        _blocked_irregular_heading_fixture()
    )
    unknown_x, unknown_y = frontier_module._local_unknown_vector(
        state.observed_mask,
        landing,
        radius_cells=4,
    )
    local_theta = math.atan2(unknown_y, unknown_x)
    target_theta = math.atan2(target.y - landing.y, target.x - landing.x)

    assert landing_mask[landing.y, landing.x]
    assert frontier_module._observed_clearance_m(state, landing) >= 2.0
    heading_targets = frontier_module._irregular_anchor_unknown_targets(
        state,
        (landing,),
    )
    assert heading_targets == (target, CellXY(7, 8))
    assert len(heading_targets) <= 12 * 8
    assert frontier_module._estimate_observed_only_gain_with_blockers(
        state,
        prior,
        landing,
        local_theta,
        blocker_mask,
    ).potential_gain_cells == 0
    target_ray = frontier_module._estimate_observed_only_gain_with_blockers(
        state,
        prior,
        landing,
        target_theta,
        blocker_mask,
        fov_deg=0.0,
    )
    assert target in target_ray.visible_unknown
    assert target_ray.potential_gain_cells > 0

    candidates = FrontierGenerator()._irregular_candidates(
        state,
        landing_mask,
        prior,
        frontier_module._FrontierSegmentPath(cells=(landing,)),
        blocker_mask,
    )

    assert candidates == [(landing, target_theta)]
    assert all(
        frontier_module._estimate_observed_only_gain_with_blockers(
            state,
            prior,
            candidate,
            theta,
            blocker_mask,
        ).potential_gain_cells
        > 0
        for candidate, theta in candidates
    )


def test_irregular_segment_uses_exact_target_when_sampled_local_fov_misses_narrow_los() -> None:
    state = ObservedMapState.empty(GridGeometry(17, 17, 1.0))
    state.observed_mask[:] = True
    state.confidence[:] = 1.0
    state.traversability[:] = 1.0
    state.observed_safe_mask[:] = True
    state.observed_mask[7, 9:12] = False
    state.observed_mask[11, 11] = False
    state.obstacle[7, 8] = True
    state.obstacle[8, 7] = True
    landing = CellXY(7, 7)
    anchor = CellXY(10, 10)
    target = CellXY(11, 11)
    landing_mask = np.zeros(state.geometry.shape, dtype=bool)
    landing_mask[landing.y, landing.x] = state.planning_safe_mask[landing.y, landing.x]
    blocker_mask = frontier_module._build_observed_gain_blocker_mask(state)
    unknown_x, unknown_y = frontier_module._local_unknown_vector(
        state.observed_mask,
        landing,
        radius_cells=4,
    )
    local_theta = math.atan2(unknown_y, unknown_x)
    target_theta = math.atan2(target.y - landing.y, target.x - landing.x)
    target_offset_deg = math.degrees(target_theta - local_theta)

    assert landing_mask[landing.y, landing.x]
    assert frontier_module._observed_clearance_m(state, landing) >= 1.0
    assert frontier_module._irregular_anchor_unknown_targets(state, (anchor,)) == (target,)
    assert 0.0 < target_offset_deg < 45.0
    assert not math.isclose(target_offset_deg, round(target_offset_deg), abs_tol=1.0e-6)
    assert frontier_module._estimate_observed_only_gain_with_blockers(
        state,
        _prior(),
        landing,
        local_theta,
        blocker_mask,
    ).potential_gain_cells == 0
    exact_target_ray = frontier_module._estimate_observed_only_gain_with_blockers(
        state,
        _prior(),
        landing,
        target_theta,
        blocker_mask,
        fov_deg=0.0,
    )
    assert target in exact_target_ray.visible_unknown
    assert exact_target_ray.potential_gain_cells > 0
    assert frontier_module._estimate_observed_only_gain_with_blockers(
        state,
        _prior(),
        landing,
        target_theta,
        blocker_mask,
    ).potential_gain_cells > 0

    candidates = FrontierGenerator()._irregular_candidates(
        state,
        landing_mask,
        _prior(),
        frontier_module._FrontierSegmentPath(cells=(anchor,)),
        blocker_mask,
    )

    assert candidates == [(landing, target_theta)]


def test_irregular_segment_keeps_positive_local_heading_when_anchor_target_is_missed() -> None:
    state = ObservedMapState.empty(GridGeometry(17, 17, 1.0))
    state.observed_mask[:] = True
    state.confidence[:] = 1.0
    state.traversability[:] = 1.0
    state.observed_safe_mask[:] = True
    local_visible_unknown = CellXY(8, 7)
    target = CellXY(11, 11)
    state.observed_mask[local_visible_unknown.y, local_visible_unknown.x] = False
    state.observed_mask[target.y, target.x] = False
    state.obstacle[8, 9] = True
    state.obstacle[9, 8] = True
    landing = CellXY(7, 7)
    anchor = CellXY(10, 10)
    landing_mask = np.zeros(state.geometry.shape, dtype=bool)
    landing_mask[landing.y, landing.x] = state.planning_safe_mask[landing.y, landing.x]
    blocker_mask = frontier_module._build_observed_gain_blocker_mask(state)
    unknown_x, unknown_y = frontier_module._local_unknown_vector(
        state.observed_mask,
        landing,
        radius_cells=4,
    )
    local_theta = math.atan2(unknown_y, unknown_x)
    target_theta = math.atan2(target.y - landing.y, target.x - landing.x)
    local_gain = frontier_module._estimate_observed_only_gain_with_blockers(
        state,
        _prior(),
        landing,
        local_theta,
        blocker_mask,
    )

    assert landing_mask[landing.y, landing.x]
    assert frontier_module._irregular_anchor_unknown_targets(state, (anchor,)) == (target,)
    assert local_gain.potential_gain_cells > 0
    assert local_visible_unknown in local_gain.visible_unknown
    assert target not in local_gain.visible_unknown
    assert frontier_module._estimate_observed_only_gain_with_blockers(
        state,
        _prior(),
        landing,
        target_theta,
        blocker_mask,
        fov_deg=0.0,
    ).potential_gain_cells > 0

    candidates = FrontierGenerator()._irregular_candidates(
        state,
        landing_mask,
        _prior(),
        frontier_module._FrontierSegmentPath(cells=(anchor,)),
        blocker_mask,
    )

    assert candidates == [(landing, local_theta)]


def test_irregular_heading_probe_stops_after_first_ordered_visible_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = ObservedMapState.empty(GridGeometry(16, 16, 1.0))
    state.observed_mask[:] = True
    state.confidence[:] = 1.0
    state.traversability[:] = 1.0
    state.observed_safe_mask[:] = True
    landing = CellXY(5, 5)
    first_target = CellXY(7, 5)
    second_target = CellXY(8, 5)
    blocker_mask = np.zeros(state.geometry.shape, dtype=bool)
    ray_name = (
        "iter_ray_cells_from_world"
        if hasattr(frontier_module, "iter_ray_cells_from_world")
        else "ray_cells_from_world"
    )
    real_ray = getattr(frontier_module, ray_name)
    queried_ranges: list[float] = []

    def count_ray(
        *args: object,
        **kwargs: float,
    ) -> object:
        queried_ranges.append(float(kwargs["range_m"]))
        return real_ray(*args, **kwargs)

    monkeypatch.setattr(frontier_module, ray_name, count_ray)

    result = frontier_module._first_visible_irregular_heading_target(
        state,
        landing,
        (first_target, second_target),
        blocker_mask,
        math.pi / 2.0,
    )

    assert result == first_target
    assert queried_ranges[0] == 2.0
    assert 3.0 not in queried_ranges
    assert queried_ranges.count(20.0) == 91
    assert len(queried_ranges) == 92


def test_irregular_heading_probe_excludes_targets_beyond_sensor_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = ObservedMapState.empty(GridGeometry(48, 16, 1.0))
    state.observed_mask[:] = True
    state.confidence[:] = 1.0
    state.traversability[:] = 1.0
    state.observed_safe_mask[:] = True
    landing = CellXY(10, 8)
    target = CellXY(31, 8)
    state.observed_mask[target.y, target.x] = False
    blocker_mask = np.zeros(state.geometry.shape, dtype=bool)
    ray_name = (
        "iter_ray_cells_from_world"
        if hasattr(frontier_module, "iter_ray_cells_from_world")
        else "ray_cells_from_world"
    )
    real_ray = getattr(frontier_module, ray_name)
    queried_ranges: list[float] = []

    def count_ray(
        *args: object,
        **kwargs: float,
    ) -> object:
        queried_ranges.append(float(kwargs["range_m"]))
        return real_ray(*args, **kwargs)

    monkeypatch.setattr(frontier_module, ray_name, count_ray)

    result = frontier_module._first_visible_irregular_heading_target(
        state,
        landing,
        (target,),
        blocker_mask,
        math.pi,
    )

    assert result is None
    assert queried_ranges == []


def test_irregular_local_presence_probe_stops_on_first_visible_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = ObservedMapState.empty(GridGeometry(64, 64, 1.0))
    state.observed_mask[:] = True
    state.confidence[:] = 1.0
    state.traversability[:] = 1.0
    state.observed_safe_mask[:] = True
    landing = CellXY(32, 32)
    first_unknown = CellXY(33, 32)
    target = CellXY(34, 32)
    state.observed_mask[first_unknown.y, first_unknown.x] = False
    state.observed_mask[target.y, target.x] = False
    blocker_mask = np.zeros(state.geometry.shape, dtype=bool)
    real_iterator = frontier_module.iter_ray_cells_from_world
    queried_ranges: list[float] = []
    yielded_cells = 0

    def count_iterator(
        *args: object,
        **kwargs: float,
    ) -> object:
        nonlocal yielded_cells
        queried_ranges.append(float(kwargs["range_m"]))
        for cell in real_iterator(*args, **kwargs):
            yielded_cells += 1
            yield cell

    monkeypatch.setattr(
        frontier_module,
        "iter_ray_cells_from_world",
        count_iterator,
    )

    result = frontier_module._first_visible_irregular_heading_target(
        state,
        landing,
        (target,),
        blocker_mask,
        math.pi / 4.0,
    )

    assert result is None
    assert queried_ranges == [2.0, 20.0]
    assert yielded_cells == 5


def test_irregular_visible_target_selection_is_deterministic() -> None:
    state, prior, landing, _, landing_mask, blocker_mask = (
        _blocked_irregular_heading_fixture()
    )
    generator = FrontierGenerator()
    segment = frontier_module._FrontierSegmentPath(cells=(landing,))

    first = generator._irregular_candidates(
        state,
        landing_mask,
        prior,
        segment,
        blocker_mask,
    )
    repeated = generator._irregular_candidates(
        state,
        landing_mask,
        prior,
        segment,
        blocker_mask,
    )

    assert first == repeated


def test_irregular_visible_target_selection_uses_observed_only_state() -> None:
    state, prior, landing, _, landing_mask, blocker_mask = (
        _blocked_irregular_heading_fixture()
    )
    generator = FrontierGenerator()
    segment = frontier_module._FrontierSegmentPath(cells=(landing,))
    first = generator._irregular_candidates(
        state,
        landing_mask,
        prior,
        segment,
        blocker_mask,
    )

    unknown = ~state.observed_mask
    state.height[unknown] = 1.0e30
    state.obstacle[unknown] = True
    state.slope_deg[unknown] = 89.0
    state.traversability[unknown] = 0.0
    repeated = generator._irregular_candidates(
        state,
        landing_mask,
        prior,
        segment,
        frontier_module._build_observed_gain_blocker_mask(state),
    )

    assert first == repeated
    assert "frontier_oracle" not in inspect.getsource(frontier_module)


def test_irregular_segment_samples_positive_gain_cell_beyond_nearest_safe_landing() -> None:
    state = ObservedMapState.empty(GridGeometry(15, 15, 1.0))
    state.observed_mask[:] = True
    state.confidence[:] = 1.0
    state.traversability[:] = 1.0
    state.observed_safe_mask[:] = True
    state.observed_mask[7, 10] = False
    state.obstacle[7, 7] = True
    landing_mask = np.zeros(state.geometry.shape, dtype=bool)
    nearest = CellXY(5, 7)
    positive_gain = CellXY(6, 8)
    landing_mask[nearest.y, nearest.x] = True
    landing_mask[positive_gain.y, positive_gain.x] = True
    blocker_mask = frontier_module._build_observed_gain_blocker_mask(state)
    segment = frontier_module._FrontierSegmentPath(cells=(nearest,))

    assert frontier_module._nearest_valid_cell(
        state,
        landing_mask,
        nearest,
        radius_cells=4,
    ) == nearest
    nearest_unknown_x, nearest_unknown_y = frontier_module._local_unknown_vector(
        state.observed_mask,
        nearest,
        radius_cells=4,
    )
    nearest_theta = math.atan2(nearest_unknown_y, nearest_unknown_x)
    assert frontier_module._estimate_observed_only_gain_with_blockers(
        state,
        _prior(),
        nearest,
        nearest_theta,
        blocker_mask,
    ).potential_gain_cells == 0
    positive_unknown_x, positive_unknown_y = frontier_module._local_unknown_vector(
        state.observed_mask,
        positive_gain,
        radius_cells=4,
    )
    positive_theta = math.atan2(positive_unknown_y, positive_unknown_x)
    assert frontier_module._estimate_observed_only_gain_with_blockers(
        state,
        _prior(),
        positive_gain,
        positive_theta,
        blocker_mask,
    ).potential_gain_cells > 0

    candidates = FrontierGenerator()._irregular_candidates(
        state,
        landing_mask,
        _prior(),
        segment,
        blocker_mask,
    )

    assert candidates
    assert positive_gain in {cell for cell, _ in candidates}
    assert frontier_module._irregular_anchor_unknown_targets(state, (nearest,)) == ()
    assert dict(candidates)[positive_gain] == positive_theta


def test_irregular_segment_scores_bounded_compass_pool_and_keeps_stable_best_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = ObservedMapState.empty(GridGeometry(15, 15, 1.0))
    state.observed_mask[:] = True
    state.confidence[:] = 1.0
    state.traversability[:] = 1.0
    state.observed_safe_mask[:] = True
    anchor = CellXY(7, 7)
    nearest = anchor
    compass_extrema = (
        CellXY(11, 7),
        CellXY(10, 10),
        CellXY(7, 11),
        CellXY(4, 10),
        CellXY(3, 7),
        CellXY(4, 4),
        CellXY(7, 3),
        CellXY(10, 4),
    )
    landing_mask = np.zeros(state.geometry.shape, dtype=bool)
    for cell in (nearest, *compass_extrema):
        landing_mask[cell.y, cell.x] = True
    gains = {
        nearest: 1,
        CellXY(7, 3): 10,
        CellXY(3, 7): 10,
        CellXY(11, 7): 10,
        CellXY(7, 11): 9,
        CellXY(10, 10): 8,
        CellXY(4, 10): 7,
        CellXY(4, 4): 6,
        CellXY(10, 4): 5,
    }
    evaluated: list[CellXY] = []

    def fake_gain_batch(
        _state: ObservedMapState,
        _prior_value: LowResolutionPrior,
        cells: list[CellXY],
        _thetas: list[float],
        _blockers: np.ndarray,
        **_kwargs: float,
    ) -> tuple[frontier_module.GainEstimate, ...]:
        evaluated.extend(cells)
        return tuple(
            frontier_module.GainEstimate(
                visible_unknown=(),
                visible_footprint=(),
                potential_gain_cells=(gain := gains[cell]),
                value_gain=float(gain),
            )
            for cell in cells
        )

    monkeypatch.setattr(
        frontier_module,
        "_estimate_observed_only_gain_batch_with_blockers",
        fake_gain_batch,
    )

    generator = FrontierGenerator()
    segment = frontier_module._FrontierSegmentPath(cells=(anchor,))
    candidates = generator._irregular_candidates(
        state,
        landing_mask,
        _prior(),
        segment,
        np.zeros(state.geometry.shape, dtype=bool),
    )

    assert (
        getattr(frontier_module, "_IRREGULAR_MAX_GAIN_EVALUATIONS_PER_SEGMENT", None)
        == getattr(frontier_module, "_IRREGULAR_MAX_ANCHORS_PER_SEGMENT", None)
        * (1 + len(getattr(frontier_module, "_IRREGULAR_COMPASS_DIRECTIONS", ())))
        == 108
    )
    assert set(evaluated) == {nearest, *compass_extrema}
    assert len(evaluated) == len(set(evaluated)) == 9
    expected = [
        (CellXY(7, 3), 0.0),
        (CellXY(3, 7), 0.0),
        (CellXY(11, 7), 0.0),
    ]
    assert candidates == expected

    evaluated.clear()
    repeated = generator._irregular_candidates(
        state,
        landing_mask,
        _prior(),
        segment,
        np.zeros(state.geometry.shape, dtype=bool),
    )

    assert len(evaluated) == len(set(evaluated)) == 9
    assert repeated == expected


def test_irregular_frontier_extraction_builds_whole_map_state_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    size = 64
    state = ObservedMapState.empty(GridGeometry(size, size, 0.5))
    yy, xx = np.mgrid[:size, :size]
    observed = (xx - 32) ** 2 + (yy - 32) ** 2 <= 18 ** 2
    state.observed_mask[:] = observed
    state.confidence[observed] = 1.0
    state.traversability[observed] = 1.0
    state.observed_safe_mask[:] = observed
    safe_calls = 0
    blocker_build_calls = 0
    clearance_calls = 0
    real_observed_safe_cells = frontier_module._observed_safe_cells
    real_builder = frontier_module._build_observed_gain_blocker_mask
    real_observed_clearance_m = frontier_module._observed_clearance_m

    def count_observed_safe_cells(
        counted_state: ObservedMapState,
        threshold: float,
    ) -> np.ndarray:
        nonlocal safe_calls
        safe_calls += 1
        return real_observed_safe_cells(counted_state, threshold)

    def count_builder(counted_state: ObservedMapState) -> np.ndarray:
        nonlocal blocker_build_calls
        blocker_build_calls += 1
        return real_builder(counted_state)

    def count_observed_clearance_m(counted_state: ObservedMapState, cell: CellXY) -> float:
        nonlocal clearance_calls
        clearance_calls += 1
        return real_observed_clearance_m(counted_state, cell)

    monkeypatch.setattr(frontier_module, "_observed_safe_cells", count_observed_safe_cells)
    monkeypatch.setattr(frontier_module, "_build_observed_gain_blocker_mask", count_builder)
    monkeypatch.setattr(frontier_module, "_observed_clearance_m", count_observed_clearance_m)

    actions = FrontierGenerator(top_m=16).extract(
        state,
        _prior(),
        PoseXYTheta(CellXY(32, 32), 0.0),
    )

    assert actions.candidate_count >= 1
    assert safe_calls == 1
    assert blocker_build_calls == 1
    assert clearance_calls == 0


def test_gain_los_stops_only_at_currently_observed_blockers() -> None:
    state = ObservedMapState.empty(GridGeometry(21, 21, 1.0))
    state.observed_mask[10, :8] = True
    state.traversability[state.observed_mask] = 1.0
    candidate = CellXY(5, 10)
    state.observed_mask[10, 8] = True
    state.obstacle[10, 8] = True
    blocked = frontier_module.estimate_observed_only_gain(
        state, _prior(6), candidate, 0.0, range_m=10.0, fov_deg=0.0
    )
    assert CellXY(8, 10) in blocked.visible_footprint
    assert CellXY(9, 10) not in blocked.visible_footprint
    state.observed_mask[10, 8] = False
    unknown_truth_mutation = frontier_module.estimate_observed_only_gain(
        state, _prior(6), candidate, 0.0, range_m=10.0, fov_deg=0.0
    )
    assert CellXY(9, 10) in unknown_truth_mutation.visible_unknown
    assert unknown_truth_mutation.potential_gain_cells > blocked.potential_gain_cells


def test_empty_and_overflow_sets_are_finite_and_score_first_top_m_is_stable() -> None:
    assert frontier_module.score_first_top_m([0.4, 0.4, 0.3], 2) == (0, 1)
    empty_state = ObservedMapState.empty(GridGeometry(64, 64, 0.5))
    empty = FrontierGenerator(top_m=2).extract(empty_state, _prior(), PoseXYTheta(CellXY(10, 10), 0.0))
    assert empty.cells == ()
    assert not np.any(empty.candidate_mask)
    assert np.isfinite(empty.frontier_features).all()
    assert empty.diagnostics["candidate_count_before_top_m"] == 0

    state = _half_plane_state()
    first = FrontierGenerator(top_m=1).extract(state, _prior(), PoseXYTheta(CellXY(10, 32), 0.0))
    second = FrontierGenerator(top_m=1).extract(state, _prior(), PoseXYTheta(CellXY(10, 32), 0.0))
    assert first.cells == second.cells
    assert first.diagnostics["candidate_count_before_top_m"] == 3
    assert first.diagnostics["candidate_count_after_top_m"] == 1
    assert first.diagnostics["candidate_overflow_count"] == 2
    assert first.diagnostics["kept_min_priority"] >= first.diagnostics["pruned_max_priority"]
    assert first.diagnostics["selected_candidate_original_rank"] == second.diagnostics[
        "selected_candidate_original_rank"
    ]


def test_reachability_cost_is_normalized_within_single_and_two_candidate_sets() -> None:
    state = ObservedMapState.empty(GridGeometry(32, 32, 1.0))
    state.traversability[:] = 1.0
    pose = PoseXYTheta(CellXY(0, 0), 0.0)
    single_rows, _ = frontier_module._feature_rows(
        [_raw_candidate_for_reachability_cost(x=1, cost=0.25)],
        pose,
        state,
        1,
    )
    assert single_rows[0, 18] == 1.0

    pair_rows, _ = frontier_module._feature_rows(
        [
            _raw_candidate_for_reachability_cost(x=1, cost=0.25),
            _raw_candidate_for_reachability_cost(x=2, cost=0.50),
        ],
        pose,
        state,
        1,
    )
    np.testing.assert_allclose(pair_rows[:, 18], (0.5, 1.0), rtol=0.0, atol=0.0)


def test_reachability_cost_normalization_is_finite_for_empty_and_all_zero_sets() -> None:
    state = ObservedMapState.empty(GridGeometry(32, 32, 1.0))
    pose = PoseXYTheta(CellXY(0, 0), 0.0)
    empty_rows, empty_priorities = frontier_module._feature_rows([], pose, state, 0)
    assert empty_rows.shape == (0, 22)
    assert empty_priorities == []
    zero_rows, zero_priorities = frontier_module._feature_rows(
        [
            _raw_candidate_for_reachability_cost(x=1, cost=0.0),
            _raw_candidate_for_reachability_cost(x=2, cost=0.0),
        ],
        pose,
        state,
        1,
    )
    assert np.array_equal(zero_rows[:, 18], np.zeros(2, dtype=np.float32))
    assert np.isfinite(zero_rows).all()
    assert all(math.isfinite(priority) for priority in zero_priorities)


def test_priority_and_top_m_use_candidate_set_normalized_reachability_cost() -> None:
    state = ObservedMapState.empty(GridGeometry(32, 32, 1.0))
    state.traversability[:] = 1.0
    pose = PoseXYTheta(CellXY(0, 0), 0.0)
    rows, priorities = frontier_module._feature_rows(
        [
            _raw_candidate_for_reachability_cost(x=1, cost=0.25),
            _raw_candidate_for_reachability_cost(x=2, cost=0.50),
        ],
        pose,
        state,
        1,
    )
    np.testing.assert_allclose(rows[:, 18], (0.5, 1.0), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(priorities, (0.48, 0.43), rtol=0.0, atol=1.0e-7)
    assert frontier_module.score_first_top_m(priorities, 1) == (0,)


def test_default_rule_selector_preserves_stage1_risk_order_and_theta_fields() -> None:
    features = np.zeros((2, 22), dtype=np.float32)
    features[0, [4, 5, 6, 7, 8]] = (0.1, 0.1, 1.0, 0.9, 0.9)
    features[1, [4, 5, 6, 7, 8]] = (0.9, 1.0, 0.0, 0.1, 0.1)
    features[1, 14:16] = (0.0, -1.0)
    mask = np.ones(2, dtype=bool)
    assert select_conservative_frontier_candidate_index(features, mask) == 1
    observation = PolicyObservation(
        prior_channels=np.zeros((7, 2, 2), dtype=np.float32),
        coverage_summary=np.zeros((8, 2, 2), dtype=np.float32),
        local_crop=np.zeros((8, 4, 4), dtype=np.float32),
        frontier_features=features,
        pose_features=np.asarray((0, 0, 0, 1, 0, 1), dtype=np.float32),
        candidate_mask=mask,
    )
    action = select_conservative_rule_action(observation)
    assert action.candidate_index == 1
    assert math.isclose(action.target_theta, math.pi / 2.0)


def test_opt_in_stage2_diagnostic_selector_uses_frozen_stage2_priority() -> None:
    from lunar_exploration_ppo.env.env import select_stage2_diagnostic_rule_action

    features = np.zeros((2, 22), dtype=np.float32)
    features[0, [2, 5, 7, 14, 15, 18]] = (0.9, 0.8, 0.8, 1.0, 0.0, 0.9)
    features[1, [2, 5, 7, 14, 15, 18]] = (0.0, 0.1, 0.0, 0.0, 1.0, 0.0)
    observation = PolicyObservation(
        prior_channels=np.zeros((7, 2, 2), dtype=np.float32),
        coverage_summary=np.zeros((8, 2, 2), dtype=np.float32),
        local_crop=np.zeros((8, 4, 4), dtype=np.float32),
        frontier_features=features,
        pose_features=np.asarray((0, 0, 0, 1, 0, 1), dtype=np.float32),
        candidate_mask=np.ones(2, dtype=bool),
    )
    action = select_stage2_diagnostic_rule_action(observation)
    assert action.candidate_index == 0
    assert math.isclose(action.target_theta, math.pi / 2.0)


def _expected_stage1_rule_features(env: object) -> np.ndarray:
    state = env.observed_state
    action_set = env.current_action_set
    prior = env._scenario.prior
    pose = env.pose
    height, width = state.geometry.shape
    diagonal = max(math.hypot(width - 1, height - 1), 1.0)
    features = np.zeros_like(action_set.frontier_features)
    for index, cell in enumerate(action_set.cells):
        dx = cell.x - pose.cell.x
        dy = cell.y - pose.cell.y
        unknown_count = 0
        unknown_dx = 0
        unknown_dy = 0
        observed_blockers = 0
        for offset_y in (-1, 0, 1):
            for offset_x in (-1, 0, 1):
                if offset_x == 0 and offset_y == 0:
                    continue
                x = cell.x + offset_x
                y = cell.y + offset_y
                if not (0 <= x < width and 0 <= y < height):
                    continue
                if not state.observed_mask[y, x]:
                    unknown_count += 1
                    unknown_dx += offset_x
                    unknown_dy += offset_y
                elif not state.observed_safe_mask[y, x]:
                    observed_blockers += 1
        if unknown_dx != 0 or unknown_dy != 0:
            recommended_theta = math.atan2(unknown_dy, unknown_dx)
        elif math.hypot(dx, dy) > 0.0:
            recommended_theta = math.atan2(dy, dx)
        else:
            recommended_theta = pose.theta
        prior_x = min(prior.channels.shape[2] - 1, int(cell.x * prior.channels.shape[2] / width))
        prior_y = min(prior.channels.shape[1] - 1, int(cell.y * prior.channels.shape[1] / height))
        features[index, 0:9] = (
            cell.x / max(width - 1, 1),
            cell.y / max(height - 1, 1),
            dx / max(width - 1, 1),
            dy / max(height - 1, 1),
            math.hypot(dx, dy) / diagonal,
            math.sin(recommended_theta),
            math.cos(recommended_theta),
            unknown_count / 8.0,
            observed_blockers / 8.0,
        )
        features[index, 9:14] = (
            state.confidence[cell.y, cell.x],
            state.height[cell.y, cell.x],
            state.slope_deg[cell.y, cell.x] / 30.0,
            state.traversability[cell.y, cell.x],
            1.0,
        )
        features[index, 14:21] = prior.channels[:, prior_y, prior_x]
        features[index, 21] = index / max(action_set.candidate_mask.size - 1, 1)
    return features


def test_default_env_adapts_real_stage2_candidates_to_stage1_rule_semantics(monkeypatch) -> None:
    from pathlib import Path

    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env import env as env_module
    from lunar_exploration_ppo.env.env import EnvAction

    root = Path(__file__).resolve().parents[2]
    env = env_module.LunarExplorationEnv(
        load_stage1_config(root / "configs/ppo_highres_frontier_smoke_v1.json")
    )
    public_observation = env.reset()
    expected = _expected_stage1_rule_features(env)
    captured: dict[str, object] = {}

    def capture_rule_input(rule_input: object) -> EnvAction:
        captured["input"] = rule_input
        return EnvAction(candidate_index=0, target_theta=0.0)

    monkeypatch.setattr(env_module, "select_conservative_rule_action", capture_rule_input)
    env.select_rule_action(public_observation)

    rule_input = captured["input"]
    assert rule_input is not public_observation
    assert not isinstance(rule_input, PolicyObservation)
    np.testing.assert_array_equal(rule_input.frontier_features, expected)
    np.testing.assert_array_equal(rule_input.candidate_mask, env.current_action_set.candidate_mask)


def test_default_env_legacy_rule_path_preserves_smoke_success() -> None:
    from pathlib import Path

    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import LunarExplorationEnv

    root = Path(__file__).resolve().parents[2]
    env = LunarExplorationEnv(load_stage1_config(root / "configs/ppo_highres_frontier_smoke_v1.json"))
    observation = env.reset()
    final = None
    while not env.is_done:
        final = env.step(env.select_rule_action(observation))
        observation = final.observation
    assert final is not None
    assert env.terminal_reason == "success_done"
    assert final.coverage_rate >= 0.99


def test_legacy_smoke_retains_first_rule_action_at_pure_gain_divergence() -> None:
    from pathlib import Path

    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import LunarExplorationEnv

    root = Path(__file__).resolve().parents[2]
    env = LunarExplorationEnv(load_stage1_config(root / "configs/ppo_highres_frontier_smoke_v1.json"))
    observation = env.reset()
    for _ in range(5):
        result = env.step(env.select_rule_action(observation))
        observation = result.observation

    action = env.select_rule_action(observation)
    selected = env.current_action_set.cells[action.candidate_index]

    assert selected == CellXY(17, 29)
    assert math.isclose(action.target_theta, -2.081619187040, abs_tol=1.0e-12)


def test_opt_in_stage2_diagnostic_selector_preserves_smoke_demo_success() -> None:
    from pathlib import Path

    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env import env as env_module

    assert hasattr(env_module, "select_stage2_diagnostic_rule_action")
    selector = env_module.select_stage2_diagnostic_rule_action

    root = Path(__file__).resolve().parents[2]
    env = env_module.LunarExplorationEnv(
        load_stage1_config(root / "configs/ppo_highres_frontier_smoke_v1.json")
    )
    observation = env.reset()
    final = None
    while not env.is_done:
        final = env.step(selector(observation))
        observation = final.observation
    assert final is not None
    assert env.terminal_reason == "success_done"
    assert final.coverage_rate >= 0.99
