from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from lunar_exploration_ppo.configs.stage1 import load_stage1_config
from lunar_exploration_ppo.env import frontier as frontier_module
from lunar_exploration_ppo.env.env import LunarExplorationEnv
from lunar_exploration_ppo.env.frontier import FrontierGenerator
from lunar_exploration_ppo.env.map_state import ObservedMapState
from lunar_exploration_ppo.env.reachability import reachable_component
from lunar_exploration_ppo.env.scenario import LowResolutionPrior
from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry, PoseXYTheta


ROOT = Path(__file__).resolve().parents[2]
STAGE1_CONFIG = ROOT / "configs/ppo_highres_frontier_smoke_v1.json"


def test_regular_boundary_frontier_correction_preserves_high_recall_candidate() -> None:
    state = ObservedMapState.empty(GridGeometry(32, 32, 0.5))
    state.observed_mask[:] = True
    state.confidence[:] = 1.0
    state.traversability[:] = 1.0
    unknown_pocket = (
        (14, 26),
        (15, 26),
        (16, 26),
        (14, 27),
        (15, 27),
        (16, 27),
        (15, 28),
    )
    for x, y in unknown_pocket:
        state.observed_mask[y, x] = False
        state.confidence[y, x] = 0.0
        state.traversability[y, x] = 0.0

    state.observed_safe_mask[5:30, 5] = True
    state.observed_safe_mask[29, 5:17] = True
    state.observed_safe_mask[28, 14] = True
    state.observed_safe_mask[28, 16] = True
    pose = PoseXYTheta(CellXY(5, 5), 0.0)
    prior_channels = np.zeros((7, 8, 8), dtype=np.float32)
    prior_channels[1] = 1.0
    prior_channels[3] = 1.0
    prior = LowResolutionPrior(
        prior_channels,
        resolution_m=4.0,
        value_prior_source="constant_neutral/v1",
        provenance={"fixture": "regular_boundary_frontier/v1"},
    )

    action_set = FrontierGenerator(top_m=16).extract(state, prior, pose)

    assert action_set.diagnostics["segment_count"] == 1
    assert action_set.diagnostics["regular_segment_count"] == 1
    assert action_set.candidate_count >= 1
    component = reachable_component(state.observed_safe_mask, pose.cell)
    valid_features = action_set.frontier_features[action_set.candidate_mask]
    assert np.all(valid_features[:, 5] > 0.0)
    assert np.all(valid_features[:, 14] < -0.9)
    assert np.all(np.abs(valid_features[:, 15]) < 0.1)
    for cell in action_set.cells:
        assert state.observed_mask[cell.y, cell.x]
        assert state.observed_safe_mask[cell.y, cell.x]
        assert component[cell.y, cell.x]
        assert frontier_module._observed_clearance_m(state, cell) >= 0.5215874761


def test_stage1_step_diagnostics_preserve_exact_planner_path_without_replanning() -> None:
    env = LunarExplorationEnv(load_stage1_config(STAGE1_CONFIG))
    observation = env.reset()
    action = env.select_rule_action(observation)
    original_validate = env.planner.validate
    captured: dict[str, object] = {}

    def capture_validate(*args, **kwargs):
        plan = original_validate(*args, **kwargs)
        captured["path_cells"] = plan.path_cells
        captured["path_length_m"] = plan.path_length_m
        return plan

    env.planner.validate = capture_validate  # type: ignore[method-assign]

    result = env.step(action)

    assert result.diagnostics.planned_path_cells == captured["path_cells"]
    assert result.diagnostics.path_length_m == pytest.approx(captured["path_length_m"])
    assert result.diagnostics.path_observation_step_m == env.config.path_observation_step_m
    assert result.diagnostics.newly_observed_cell_count >= 0
    assert result.diagnostics.execution is not None
    assert result.diagnostics.path_length_m == pytest.approx(
        result.diagnostics.execution.path_length_m
    )


def test_episode_state_export_import_is_exact_and_future_equivalent() -> None:
    config = load_stage1_config(STAGE1_CONFIG)
    original = LunarExplorationEnv(config)
    observation = original.reset()
    first = original.step(original.select_rule_action(observation))
    assert not first.done
    state = original.export_episode_state()

    restored = LunarExplorationEnv(config)
    restored.import_episode_state(state)

    assert restored.export_episode_state() == state
    assert restored.step_count == original.step_count
    assert restored.consecutive_no_gain_steps == original.consecutive_no_gain_steps
    assert restored.pose == original.pose
    assert restored.is_done == original.is_done
    assert restored.terminal_reason == original.terminal_reason
    assert restored.needs_policy == original.needs_policy
    assert restored.current_action_set.cells == original.current_action_set.cells
    assert np.array_equal(
        restored.current_action_set.frontier_features,
        original.current_action_set.frontier_features,
    )
    assert restored.current_observation is not None
    assert original.current_observation is not None
    for name in (
        "prior_channels",
        "coverage_summary",
        "local_crop",
        "frontier_features",
        "pose_features",
        "candidate_mask",
    ):
        assert np.array_equal(
            getattr(restored.current_observation, name),
            getattr(original.current_observation, name),
        )

    next_original = original.step(
        original.select_rule_action(original.current_observation)
    )
    next_restored = restored.step(
        restored.select_rule_action(restored.current_observation)
    )
    assert next_restored.reward == next_original.reward
    assert next_restored.done == next_original.done
    assert next_restored.reason == next_original.reason
    assert next_restored.coverage_gain_cells == next_original.coverage_gain_cells
    assert next_restored.diagnostics.planned_path_cells == (
        next_original.diagnostics.planned_path_cells
    )


def test_episode_state_import_rejects_tamper_without_mutating_env() -> None:
    config = load_stage1_config(STAGE1_CONFIG)
    source = LunarExplorationEnv(config)
    source.reset()
    state = source.export_episode_state()
    tampered = copy.deepcopy(state)
    tampered["step_count"] = 99

    target = LunarExplorationEnv(config)
    before = target.export_episode_state()
    with pytest.raises(ValueError, match="state hash mismatch"):
        target.import_episode_state(tampered)
    assert target.export_episode_state() == before
