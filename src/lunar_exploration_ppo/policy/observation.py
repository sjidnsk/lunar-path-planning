"""Future-compatible Stage 1 policy observation built from deployed evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lunar_exploration_ppo.env.frontier import FrontierActionSet
from lunar_exploration_ppo.env.map_state import ObservedMapState
from lunar_exploration_ppo.env.scenario import LowResolutionPrior
from lunar_exploration_ppo.utils.geometry import PoseXYTheta


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    prior_channels: np.ndarray
    coverage_summary: np.ndarray
    local_crop: np.ndarray
    frontier_features: np.ndarray
    pose_features: np.ndarray
    candidate_mask: np.ndarray

    def __post_init__(self) -> None:
        for array in self.array_fields():
            array.setflags(write=False)

    def array_fields(self) -> tuple[np.ndarray, ...]:
        return (
            self.prior_channels,
            self.coverage_summary,
            self.local_crop,
            self.frontier_features,
            self.pose_features,
            self.candidate_mask,
        )


class ObservationBuilder:
    def build(
        self,
        prior: LowResolutionPrior,
        observed_state: ObservedMapState,
        pose: PoseXYTheta,
        action_set: FrontierActionSet,
    ) -> PolicyObservation:
        summary = _coverage_summary(observed_state, prior.channels.shape[1:])
        crop = _local_crop(observed_state, pose, size=64)
        height, width = observed_state.geometry.shape
        pose_features = np.asarray(
            (
                pose.cell.x / max(width - 1, 1),
                pose.cell.y / max(height - 1, 1),
                math.sin(pose.theta),
                math.cos(pose.theta),
                action_set.candidate_count / max(action_set.candidate_mask.size, 1),
                float(np.mean(observed_state.observed_mask)),
            ),
            dtype=np.float32,
        )
        return PolicyObservation(
            prior_channels=np.asarray(prior.channels, dtype=np.float32).copy(),
            coverage_summary=summary,
            local_crop=crop,
            frontier_features=np.asarray(action_set.frontier_features, dtype=np.float32).copy(),
            pose_features=pose_features,
            candidate_mask=np.asarray(action_set.candidate_mask, dtype=bool).copy(),
        )


def _coverage_summary(state: ObservedMapState, output_shape: tuple[int, int]) -> np.ndarray:
    out_height, out_width = output_shape
    height, width = state.geometry.shape
    if height % out_height or width % out_width:
        raise ValueError("observed grid must divide evenly into the low-resolution summary")
    block_y, block_x = height // out_height, width // out_width
    observed = state.observed_mask.astype(np.float32)
    channels = np.stack(
        (
            observed,
            state.confidence.astype(np.float32),
            np.where(state.observed_mask, state.height, 0.0).astype(np.float32),
            state.obstacle.astype(np.float32),
            np.where(state.observed_mask, state.slope_deg / 30.0, 0.0).astype(np.float32),
            np.where(state.observed_mask, state.traversability, 0.0).astype(np.float32),
            state.observed_safe_mask.astype(np.float32),
            (~state.observed_mask).astype(np.float32),
        ),
        axis=0,
    )
    reshaped = channels.reshape(8, out_height, block_y, out_width, block_x)
    return reshaped.mean(axis=(2, 4), dtype=np.float64).astype(np.float32)


def _local_crop(state: ObservedMapState, pose: PoseXYTheta, *, size: int) -> np.ndarray:
    crop = np.zeros((8, size, size), dtype=np.float32)
    half = size // 2
    requested_x0 = pose.cell.x - half
    requested_y0 = pose.cell.y - half
    source_x0 = max(0, requested_x0)
    source_y0 = max(0, requested_y0)
    source_x1 = min(state.geometry.width, requested_x0 + size)
    source_y1 = min(state.geometry.height, requested_y0 + size)
    destination_x0 = source_x0 - requested_x0
    destination_y0 = source_y0 - requested_y0
    destination_x1 = destination_x0 + source_x1 - source_x0
    destination_y1 = destination_y0 + source_y1 - source_y0
    source = np.stack(
        (
            state.observed_mask.astype(np.float32),
            state.confidence.astype(np.float32),
            np.where(state.observed_mask, state.height, 0.0).astype(np.float32),
            state.obstacle.astype(np.float32),
            np.where(state.observed_mask, state.slope_deg / 30.0, 0.0).astype(np.float32),
            np.where(state.observed_mask, state.traversability, 0.0).astype(np.float32),
            state.observed_safe_mask.astype(np.float32),
            (~state.observed_mask).astype(np.float32),
        ),
        axis=0,
    )
    crop[:, destination_y0:destination_y1, destination_x0:destination_x1] = source[
        :, source_y0:source_y1, source_x0:source_x1
    ]
    return crop
