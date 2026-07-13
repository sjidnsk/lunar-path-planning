"""Serialization of Stage 1 reset and action metrics."""

from __future__ import annotations

from datetime import datetime, timezone

from lunar_exploration_ppo.env.env import LunarExplorationEnv, StepResult


def reset_metric(env: LunarExplorationEnv, episode_index: int) -> dict[str, object]:
    diagnostics = env.last_reset_diagnostics
    assert diagnostics is not None
    return {
        "progress_type": "reset",
        "timestamp": _timestamp(),
        "scale_profile": env.config.scale_profile,
        "episode": episode_index,
        "step": 0,
        "coverage_rate": diagnostics.coverage_rate,
        "coverage_gain_cells": 0,
        "reward": 0.0,
        "done": env.is_done,
        "reason": env.terminal_reason,
        "trainable": False,
        "terminal": env.is_done,
        "bootstrap_value": 0.0 if env.is_done else 1.0,
        "coverage_progress": diagnostics.progress.coverage_progress,
        "step_progress": diagnostics.progress.step_progress,
        "stagnation_progress": diagnostics.progress.stagnation_progress,
        "path_length_m": 0.0,
        "path_sample_count": 0,
        "endpoint_sample_count": 0,
        "sensor_sample_count": diagnostics.sensor.sample_count,
        "sensor_ray_count": diagnostics.sensor.ray_count,
        "sensor_cell_visit_count": diagnostics.sensor.cell_visit_count,
        "sensor_unique_visible_cell_count": diagnostics.sensor.unique_visible_cell_count,
        "candidate_count": diagnostics.frontier.candidate_count,
        "observed_component_size": diagnostics.frontier.component_size,
        "unknown_count": diagnostics.frontier.unknown_count,
        "remaining_unobserved_coverable_count": (
            diagnostics.frontier.remaining_unobserved_coverable_count
        ),
        "frontier_oracle_opportunity_count": diagnostics.frontier.oracle_opportunity_count,
    }


def step_metric(
    env: LunarExplorationEnv,
    episode_index: int,
    result: StepResult,
) -> dict[str, object]:
    sensor = result.diagnostics.sensor
    execution = result.diagnostics.execution
    return {
        "progress_type": "action",
        "timestamp": _timestamp(),
        "scale_profile": env.config.scale_profile,
        "episode": episode_index,
        "step": env.step_count,
        "coverage_rate": result.coverage_rate,
        "coverage_gain_cells": result.coverage_gain_cells,
        "reward": result.reward,
        "done": result.done,
        "reason": result.reason,
        "trainable": result.trainable,
        "terminal": result.terminal,
        "bootstrap_value": result.bootstrap_value,
        "coverage_progress": result.diagnostics.progress.coverage_progress,
        "step_progress": result.diagnostics.progress.step_progress,
        "stagnation_progress": result.diagnostics.progress.stagnation_progress,
        "path_length_m": execution.path_length_m if execution is not None else 0.0,
        "path_sample_count": execution.path_sample_count if execution is not None else 0,
        "endpoint_sample_count": execution.endpoint_sample_count if execution is not None else 0,
        "sensor_sample_count": sensor.sample_count if sensor is not None else 0,
        "sensor_ray_count": sensor.ray_count if sensor is not None else 0,
        "sensor_cell_visit_count": sensor.cell_visit_count if sensor is not None else 0,
        "sensor_unique_visible_cell_count": (
            sensor.unique_visible_cell_count if sensor is not None else 0
        ),
        "planner_failure_reason": result.diagnostics.planner.get("failure_reason", "none"),
        "candidate_count": result.diagnostics.frontier.candidate_count,
        "observed_component_size": result.diagnostics.frontier.component_size,
        "unknown_count": result.diagnostics.frontier.unknown_count,
        "remaining_unobserved_coverable_count": (
            result.diagnostics.frontier.remaining_unobserved_coverable_count
        ),
        "frontier_oracle_opportunity_count": (
            result.diagnostics.frontier.oracle_opportunity_count
        ),
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
