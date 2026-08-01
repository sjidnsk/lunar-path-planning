from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from lunar_exploration_ppo.configs.stage1 import load_stage1_config
from lunar_exploration_ppo.env.env import LunarExplorationEnv


ROOT = Path(__file__).resolve().parents[2]
STAGE1_CONFIG = ROOT / "configs/ppo_highres_frontier_smoke_v1.json"


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
