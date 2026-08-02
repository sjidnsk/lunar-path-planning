"""Four-context Stage 4 tiny-overfit acceptance through the real PPO path."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
import torch

from lunar_exploration_ppo.policy.cross_attention import (
    CrossAttentionFrontierPolicy,
    batch_policy_observations,
    masked_frontier_probabilities,
    recompute_action_log_probs,
)
from lunar_exploration_ppo.policy.observation import PolicyObservation
from lunar_exploration_ppo.ppo.rollout import (
    ROLLOUT_ENV_COUNT,
    ROLLOUT_STEPS_PER_ENV,
    CandidateSnapshot,
    RolloutBuffer,
    RolloutTransition,
)
from lunar_exploration_ppo.ppo.trainer import (
    PPOTrainer,
    policy_state_sha256,
)
from lunar_exploration_ppo.utils.geometry import CellXY


TINY_OVERFIT_SEEDS: Final = (17, 29, 43)
TINY_CONTEXT_COUNT: Final = 4
TINY_MAX_UPDATES: Final = 200
_CONFIG_SHA256: Final = "d" * 64
_LINEAGE: Final = MappingProxyType(
    {
        "acceptance": "stage4_tiny_overfit_four_context/v1",
        "stage3_commit": "4df7be92cd6517e77c648e890bf7d32c9fcd560b",
    }
)
_TARGET_THETA: Final = (0.2, 0.2, 0.2, 0.2)
_VALUE_TARGET: Final = (-0.2, -0.05, 0.05, 0.2)


@dataclass(frozen=True, slots=True)
class TinyOverfitCurvePoint:
    update: int
    trainable_transition_count: int
    correct_frontier_probability_min: float
    theta_error_rad_max: float
    value_mse: float
    value_mse_reduction: float
    initial_ratio_max_abs_error: float
    approx_kl: float
    post_clip_grad_norm: float
    optimizer_steps: int


@dataclass(frozen=True, slots=True)
class TinyOverfitSeedResult:
    seed: int
    passed: bool
    stop_update: int
    correct_frontier_probability_min: float
    theta_error_rad_max: float
    initial_value_mse: float
    final_value_mse: float
    value_mse_reduction: float
    ppo_update_count: int
    rollout_batch_count: int
    curve: tuple[TinyOverfitCurvePoint, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["curve"] = [asdict(point) for point in self.curve]
        return value


@dataclass(frozen=True, slots=True)
class TinyOverfitResult:
    context_count: int
    seeds: tuple[TinyOverfitSeedResult, ...]
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "stage4_tiny_overfit_acceptance/v1",
            "context_count": self.context_count,
            "passed": self.passed,
            "seeds": [seed.to_dict() for seed in self.seeds],
        }


def run_tiny_overfit_acceptance(
    *,
    device: torch.device | str,
) -> TinyOverfitResult:
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("tiny-overfit requested CUDA but CUDA is unavailable")
    contexts, snapshots, correct_indices = _tiny_contexts()
    results = tuple(
        _run_seed(
            seed=seed,
            device=target_device,
            contexts=contexts,
            snapshots=snapshots,
            correct_indices=correct_indices,
        )
        for seed in TINY_OVERFIT_SEEDS
    )
    return TinyOverfitResult(
        context_count=TINY_CONTEXT_COUNT,
        seeds=results,
        passed=all(result.passed for result in results),
    )


def _run_seed(
    *,
    seed: int,
    device: torch.device,
    contexts: tuple[PolicyObservation, ...],
    snapshots: tuple[CandidateSnapshot, ...],
    correct_indices: tuple[int, ...],
) -> TinyOverfitSeedResult:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    policy = CrossAttentionFrontierPolicy().to(device)
    trainer = PPOTrainer(
        policy,
        device=device,
        shuffle_seed=seed * 1000,
    )
    initial_probability, initial_theta_error, initial_value_mse = _metrics(
        policy,
        contexts=contexts,
        correct_indices=correct_indices,
        device=device,
    )
    del initial_probability, initial_theta_error
    curve: list[TinyOverfitCurvePoint] = []
    final_probability = 0.0
    final_theta_error = math.inf
    final_value_mse = math.inf
    final_reduction = -math.inf
    for update in range(1, TINY_MAX_UPDATES + 1):
        batch = _build_batch(
            policy,
            contexts=contexts,
            snapshots=snapshots,
            correct_indices=correct_indices,
            device=device,
        )
        metrics = trainer.update(batch)
        final_probability, final_theta_error, final_value_mse = _metrics(
            policy,
            contexts=contexts,
            correct_indices=correct_indices,
            device=device,
        )
        final_reduction = (initial_value_mse - final_value_mse) / initial_value_mse
        curve.append(
            TinyOverfitCurvePoint(
                update=update,
                trainable_transition_count=1024,
                correct_frontier_probability_min=final_probability,
                theta_error_rad_max=final_theta_error,
                value_mse=final_value_mse,
                value_mse_reduction=final_reduction,
                initial_ratio_max_abs_error=metrics.initial_ratio_max_abs_error,
                approx_kl=metrics.approx_kl,
                post_clip_grad_norm=metrics.grad_post_clip_norm_max,
                optimizer_steps=metrics.optimizer_steps,
            )
        )
        if (
            final_probability >= 0.95
            and final_theta_error <= 0.1
            and final_reduction >= 0.90
        ):
            break
    passed = (
        final_probability >= 0.95
        and final_theta_error <= 0.1
        and final_reduction >= 0.90
    )
    return TinyOverfitSeedResult(
        seed=seed,
        passed=passed,
        stop_update=len(curve),
        correct_frontier_probability_min=final_probability,
        theta_error_rad_max=final_theta_error,
        initial_value_mse=initial_value_mse,
        final_value_mse=final_value_mse,
        value_mse_reduction=final_reduction,
        ppo_update_count=trainer.update_step,
        rollout_batch_count=len(curve),
        curve=tuple(curve),
    )


def _build_batch(
    policy: CrossAttentionFrontierPolicy,
    *,
    contexts: tuple[PolicyObservation, ...],
    snapshots: tuple[CandidateSnapshot, ...],
    correct_indices: tuple[int, ...],
    device: torch.device,
):
    sample_count = ROLLOUT_STEPS_PER_ENV * ROLLOUT_ENV_COUNT
    context_indices = np.arange(sample_count, dtype=np.int64) % TINY_CONTEXT_COUNT
    groups = (np.arange(sample_count, dtype=np.int64) // TINY_CONTEXT_COUNT) % 4
    selected_indices = np.empty((sample_count,), dtype=np.int64)
    selected_thetas = np.empty((sample_count,), dtype=np.float32)
    rewards = np.empty((sample_count,), dtype=np.float32)
    for row in range(sample_count):
        context_index = int(context_indices[row])
        group = int(groups[row])
        correct = correct_indices[context_index]
        selected_indices[row] = correct if group < 2 else 1 - correct
        theta = _TARGET_THETA[context_index]
        selected_thetas[row] = np.float32(
            theta
            if group % 2 == 0
            else _normalize_angle(theta + math.pi - 1.0e-4)
        )
        frontier_bonus = 1.0 if group < 2 else -1.0
        theta_bonus = 1.0 if group % 2 == 0 else -1.0
        rewards[row] = np.float32(
            _VALUE_TARGET[context_index] + frontier_bonus + theta_bonus
        )

    old_frontier = np.empty((sample_count,), dtype=np.float32)
    old_theta = np.empty((sample_count,), dtype=np.float32)
    old_total = np.empty((sample_count,), dtype=np.float32)
    old_value = np.empty((sample_count,), dtype=np.float32)
    policy.eval()
    with torch.no_grad():
        for start in range(0, sample_count, 32):
            stop = min(start + 32, sample_count)
            rows = context_indices[start:stop]
            observations = tuple(contexts[int(index)] for index in rows)
            policy_batch = batch_policy_observations(observations, device=device)
            output = policy(policy_batch)
            evaluation = recompute_action_log_probs(
                output,
                policy_batch.candidate_mask,
                torch.as_tensor(
                    selected_indices[start:stop],
                    dtype=torch.int64,
                    device=device,
                ),
                torch.as_tensor(
                    selected_thetas[start:stop],
                    dtype=torch.float32,
                    device=device,
                ),
            )
            old_frontier[start:stop] = (
                evaluation.log_prob_frontier.cpu().numpy()
            )
            old_theta[start:stop] = evaluation.log_prob_theta.cpu().numpy()
            old_total[start:stop] = evaluation.log_prob_total.cpu().numpy()
            old_value[start:stop] = output.value.cpu().numpy()

    policy_hash = policy_state_sha256(policy)
    buffer = RolloutBuffer()
    for time_index in range(ROLLOUT_STEPS_PER_ENV):
        for env_index in range(ROLLOUT_ENV_COUNT):
            row = time_index * ROLLOUT_ENV_COUNT + env_index
            context_index = int(context_indices[row])
            selected_index = int(selected_indices[row])
            snapshot = snapshots[context_index]
            selected_rank = tuple(
                int(index) for index in np.flatnonzero(snapshot.candidate_mask)
            ).index(selected_index)
            selected_cell = snapshot.frontier_cells[selected_rank]
            transition = RolloutTransition.create(
                snapshot=snapshot,
                policy_state_sha256=policy_hash,
                config_sha256=_CONFIG_SHA256,
                lineage=_LINEAGE,
                selected_frontier_index=selected_index,
                selected_cell_xy=selected_cell,
                selected_theta=float(selected_thetas[row]),
                old_log_prob_frontier=float(old_frontier[row]),
                old_log_prob_theta=float(old_theta[row]),
                old_log_prob_total=float(old_total[row]),
                old_value=float(old_value[row]),
                reward=float(rewards[row]),
                done=True,
                done_reason="success_done",
                planned_path_cells=(selected_cell,),
                path_length_m=1.0,
                path_observation_step_m=1.0,
                coverage_gain_cells=0,
                coverage_gain_per_meter=0.0,
                observation_sample_count=1,
                ray_count_per_sample=1,
                ray_cell_visit_count=1,
                newly_observed_cell_count=0,
                planner_validation_summary={"valid": True},
                execution_observation_summary={"tiny_overfit": True},
                frontier_extractor_version="tiny_overfit_fixture/v1",
                observation_schema_version="policy_observation/v1",
                action_space_version="frontier_index_continuous_theta/v1",
                reward_version="tiny_overfit_balanced_return/v1",
                planner_version="tiny_overfit_one_step/v1",
            )
            buffer.add(env_index, transition)
    return buffer.finalize(np.zeros((ROLLOUT_ENV_COUNT,), dtype=np.float32))


def _metrics(
    policy: CrossAttentionFrontierPolicy,
    *,
    contexts: tuple[PolicyObservation, ...],
    correct_indices: tuple[int, ...],
    device: torch.device,
) -> tuple[float, float, float]:
    policy.eval()
    with torch.no_grad():
        policy_batch = batch_policy_observations(contexts, device=device)
        output = policy(policy_batch)
        probabilities = masked_frontier_probabilities(
            output.frontier_logits,
            policy_batch.candidate_mask,
        )
        correct = torch.as_tensor(
            correct_indices,
            dtype=torch.int64,
            device=device,
        )
        correct_probability = probabilities.gather(
            1,
            correct.unsqueeze(1),
        ).squeeze(1)
        predicted_theta = output.theta_mu.gather(
            1,
            correct.unsqueeze(1),
        ).squeeze(1)
        target_theta = torch.as_tensor(
            _TARGET_THETA,
            dtype=torch.float32,
            device=device,
        )
        theta_error = torch.abs(
            torch.remainder(predicted_theta - target_theta + torch.pi, 2 * torch.pi)
            - torch.pi
        )
        value_target = torch.as_tensor(
            _VALUE_TARGET,
            dtype=torch.float32,
            device=device,
        )
        value_mse = (output.value - value_target).square().mean()
    return (
        float(correct_probability.min().cpu()),
        float(theta_error.max().cpu()),
        float(value_mse.cpu()),
    )


def _tiny_contexts() -> tuple[
    tuple[PolicyObservation, ...],
    tuple[CandidateSnapshot, ...],
    tuple[int, ...],
]:
    observations: list[PolicyObservation] = []
    snapshots: list[CandidateSnapshot] = []
    correct_indices: list[int] = []
    yy, xx = np.mgrid[0:8, 0:8]
    for context_index in range(TINY_CONTEXT_COUNT):
        correct_index = context_index % 2
        correct_indices.append(correct_index)
        prior = np.zeros((7, 8, 8), dtype=np.float32)
        prior[0] = np.float32(context_index / 3.0)
        prior[1] = np.asarray(xx / 7.0, dtype=np.float32)
        prior[2] = np.asarray(yy / 7.0, dtype=np.float32)
        coverage = np.zeros((8, 8, 8), dtype=np.float32)
        coverage[context_index] = 1.0
        local = np.zeros((8, 8, 8), dtype=np.float32)
        local[(context_index + 1) % 8] = np.asarray(
            (xx + yy) / 14.0,
            dtype=np.float32,
        )
        frontier = np.zeros((2, 22), dtype=np.float32)
        frontier[:, 0] = (0.25, 0.75)
        frontier[:, 1] = (0.25, 0.75)
        frontier[correct_index, 5] = 10.0
        frontier[1 - correct_index, 5] = -10.0
        frontier[:, 14] = math.sin(_TARGET_THETA[context_index])
        frontier[:, 15] = math.cos(_TARGET_THETA[context_index])
        frontier[:, 16] = 1.0
        frontier[:, 17] = 1.0
        frontier[:, 21] = 1.0
        pose = np.asarray(
            (
                context_index / 3.0,
                0.5,
                0.0,
                1.0,
                0.25,
                1.0,
            ),
            dtype=np.float32,
        )
        observation = PolicyObservation(
            prior_channels=prior,
            coverage_summary=coverage,
            local_crop=local,
            frontier_features=frontier,
            pose_features=pose,
            candidate_mask=np.ones((2,), dtype=bool),
        )
        cells = (
            CellXY(context_index * 2 + 1, 1),
            CellXY(context_index * 2 + 2, 1),
        )
        observations.append(observation)
        snapshots.append(
            CandidateSnapshot.capture(
                observation=observation,
                frontier_cells=cells,
                top_m_selection_summary={
                    "fixture": "four_context_tiny_overfit/v1",
                    "context_index": context_index,
                    "correct_index": correct_index,
                },
            )
        )
    return tuple(observations), tuple(snapshots), tuple(correct_indices)


def _normalize_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


__all__ = [
    "TINY_CONTEXT_COUNT",
    "TINY_MAX_UPDATES",
    "TINY_OVERFIT_SEEDS",
    "TinyOverfitCurvePoint",
    "TinyOverfitResult",
    "TinyOverfitSeedResult",
    "run_tiny_overfit_acceptance",
]
