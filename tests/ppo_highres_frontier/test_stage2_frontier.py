from __future__ import annotations

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


def test_regular_segment_uses_standoff_anchors_and_outward_recommended_theta() -> None:
    action_set = FrontierGenerator(top_m=16).extract(
        _half_plane_state(), _prior(), PoseXYTheta(CellXY(10, 32), 0.0)
    )
    valid = action_set.frontier_features[action_set.candidate_mask]
    assert action_set.candidate_count == 2
    assert np.all(valid[:, 13] == 0.0)
    assert np.all(valid[:, 12] >= 0.6)
    assert np.all(valid[:, 14] == 0.0)
    assert np.allclose(valid[:, 15], 1.0, atol=1e-6)
    assert all(cell.x < 31 for cell in action_set.cells)
    assert all(valid[:, 5] > 0.0)


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
    assert first.diagnostics["candidate_count_before_top_m"] == 2
    assert first.diagnostics["candidate_count_after_top_m"] == 1
    assert first.diagnostics["candidate_overflow_count"] == 1
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
