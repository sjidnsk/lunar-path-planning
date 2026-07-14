from __future__ import annotations

import copy
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from lunar_exploration_ppo.policy.cross_attention import (
    INVALID_LOGIT_VALUE,
    PolicyForwardOutput,
    batch_policy_observations,
    recompute_action_log_probs,
)
from lunar_exploration_ppo.policy.observation import PolicyObservation
from lunar_exploration_ppo.ppo.rollout import (
    CandidateSnapshot,
    RolloutBuffer,
    RolloutContractError,
    RolloutTransition,
)
from lunar_exploration_ppo.utils.geometry import CellXY


def test_ppo_loss_terms_use_joint_ratio_clipped_value_and_frontier_entropy_only() -> None:
    from lunar_exploration_ppo.ppo.trainer import compute_ppo_loss_terms

    terms = compute_ppo_loss_terms(
        new_log_prob_total=torch.tensor([math.log(1.5), math.log(0.5)], dtype=torch.float32),
        old_log_prob_total=torch.zeros(2, dtype=torch.float32),
        normalized_advantage=torch.tensor([1.0, -1.0], dtype=torch.float32),
        new_value=torch.tensor([1.0, -1.0], dtype=torch.float32),
        old_value=torch.zeros(2, dtype=torch.float32),
        returns=torch.tensor([2.0, -2.0], dtype=torch.float32),
        frontier_entropy=torch.tensor([0.4, 0.6], dtype=torch.float32),
    )

    assert torch.allclose(terms.ratio, torch.tensor([1.5, 0.5]))
    assert terms.policy_loss.item() == pytest.approx(-0.2)
    assert terms.value_loss.item() == pytest.approx(1.62)
    assert terms.frontier_entropy.item() == pytest.approx(0.5)
    assert terms.total_loss.item() == pytest.approx(-0.2 + 0.5 * 1.62 - 0.01 * 0.5)
    assert terms.approx_kl.item() == pytest.approx(
        -(math.log(1.5) + math.log(0.5)) / 2.0
    )


@pytest.mark.parametrize("field", ["new_log_prob_total", "frontier_entropy", "returns"])
def test_ppo_loss_terms_reject_nonfinite_or_non_fp32(field: str) -> None:
    from lunar_exploration_ppo.ppo.trainer import PPOTrainingError, compute_ppo_loss_terms

    values = {
        "new_log_prob_total": torch.zeros(2, dtype=torch.float32),
        "old_log_prob_total": torch.zeros(2, dtype=torch.float32),
        "normalized_advantage": torch.tensor([-1.0, 1.0], dtype=torch.float32),
        "new_value": torch.zeros(2, dtype=torch.float32),
        "old_value": torch.zeros(2, dtype=torch.float32),
        "returns": torch.ones(2, dtype=torch.float32),
        "frontier_entropy": torch.ones(2, dtype=torch.float32),
    }
    values[field] = torch.tensor([float("nan"), 0.0], dtype=torch.float32)
    with pytest.raises(PPOTrainingError, match="finite"):
        compute_ppo_loss_terms(**values)

    values[field] = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(PPOTrainingError, match="FP32"):
        compute_ppo_loss_terms(**values)


class _ToyPolicy(nn.Module):
    def __init__(self, *, logit_scale: float = 1.0) -> None:
        super().__init__()
        self.preference = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.theta_mu_parameter = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.value_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.logit_scale = float(logit_scale)

    def forward(self, observation) -> PolicyForwardOutput:
        batch_size, candidate_count = observation.candidate_mask.shape
        signs = torch.zeros(candidate_count, dtype=torch.float32, device=self.preference.device)
        signs[0] = 1.0
        signs[1] = -1.0
        raw_logits = self.preference * self.logit_scale * signs.unsqueeze(0).expand(batch_size, -1)
        logits = torch.where(
            observation.candidate_mask,
            raw_logits,
            torch.full_like(raw_logits, INVALID_LOGIT_VALUE),
        )
        theta_mu = self.theta_mu_parameter.expand(batch_size, candidate_count)
        theta_kappa = torch.ones_like(theta_mu)
        zeros = torch.zeros_like(theta_mu)
        ones = torch.ones_like(theta_mu)
        token = torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=logits.device)
        action_hidden = torch.zeros(
            (batch_size, candidate_count, 1), dtype=torch.float32, device=logits.device
        )
        return PolicyForwardOutput(
            frontier_logits=logits,
            theta_mu_sin_raw=zeros,
            theta_mu_cos_raw=ones,
            theta_kappa_raw=zeros,
            theta_mu=theta_mu,
            theta_kappa=theta_kappa,
            value=self.value_bias.expand(batch_size),
            global_map_tokens=token,
            local_map_tokens=token,
            pose_token=token,
            context_tokens=token,
            refined_frontier_tokens=action_hidden,
            action_hidden=action_hidden,
        )


def _policy_observation() -> PolicyObservation:
    return PolicyObservation(
        prior_channels=np.zeros((7, 2, 2), dtype=np.float32),
        coverage_summary=np.zeros((8, 2, 2), dtype=np.float32),
        local_crop=np.zeros((8, 3, 3), dtype=np.float32),
        frontier_features=np.zeros((3, 22), dtype=np.float32),
        pose_features=np.zeros((6,), dtype=np.float32),
        candidate_mask=np.asarray([True, True, False], dtype=bool),
    )


def _trainer_batch(
    policy: _ToyPolicy,
    *,
    old_frontier_offset: float = 0.0,
):
    from lunar_exploration_ppo.ppo.trainer import policy_state_sha256

    observation = _policy_observation()
    snapshot = CandidateSnapshot.capture(
        observation=observation,
        frontier_cells=(CellXY(1, 1), CellXY(2, 2)),
        top_m_selection_summary={"candidate_count_after_top_m": 2},
    )
    policy_hash = policy_state_sha256(policy)
    with torch.no_grad():
        policy_batch = batch_policy_observations([observation, observation])
        output = policy(policy_batch)
        selected = torch.tensor([0, 1], dtype=torch.int64)
        theta = torch.zeros(2, dtype=torch.float32)
        evaluation = recompute_action_log_probs(
            output,
            policy_batch.candidate_mask,
            selected,
            theta,
        )
    transitions: list[RolloutTransition] = []
    for index in (0, 1):
        frontier_log_prob = np.float32(
            evaluation.log_prob_frontier[index].item() + old_frontier_offset
        )
        theta_log_prob = np.float32(evaluation.log_prob_theta[index].item())
        transitions.append(
            RolloutTransition.create(
                snapshot=snapshot,
                policy_state_sha256=policy_hash,
                config_sha256="2" * 64,
                lineage={"stage3_gate_sha256": "6" * 64},
                selected_frontier_index=index,
                selected_cell_xy=(CellXY(1, 1), CellXY(2, 2))[index],
                selected_theta=np.float32(0.0),
                old_log_prob_frontier=frontier_log_prob,
                old_log_prob_theta=theta_log_prob,
                old_log_prob_total=np.float32(frontier_log_prob + theta_log_prob),
                old_value=np.float32(output.value[index].item()),
                reward=np.float32(1.0 if index == 0 else -1.0),
                done=True,
                done_reason="failure_done",
                planned_path_cells=(),
                path_length_m=np.float32(0.0),
                path_observation_step_m=np.float32(1.0),
                coverage_gain_cells=0,
                coverage_gain_per_meter=np.float32(0.0),
                observation_sample_count=0,
                ray_count_per_sample=0,
                ray_cell_visit_count=0,
                newly_observed_cell_count=0,
                planner_validation_summary={"valid": False},
                execution_observation_summary={},
                frontier_extractor_version="observed_frontier_segment_landing/v1",
                observation_schema_version="policy_observation/v1",
                action_space_version="frontier_index_continuous_theta/v1",
                reward_version="coverage_gain_terminal_reward/v1",
                planner_version="observed_safe_astar/v1",
            )
        )
    buffer = RolloutBuffer()
    for _time_index in range(128):
        for env_index in range(8):
            buffer.add(env_index, transitions[env_index % 2])
    return buffer.finalize(np.zeros((8,), dtype=np.float32))


def test_policy_state_hash_is_deterministic_and_parameter_sensitive() -> None:
    from lunar_exploration_ppo.ppo.trainer import policy_state_sha256

    policy = _ToyPolicy()
    first = policy_state_sha256(policy)
    assert first == policy_state_sha256(policy)
    with torch.no_grad():
        policy.value_bias.add_(1.0)
    assert policy_state_sha256(policy) != first


def test_physical_microbatch_slices_weight_to_the_exact_effective_mean() -> None:
    from lunar_exploration_ppo.ppo.trainer import physical_microbatch_slices

    parameter = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
    samples = torch.arange(70, dtype=torch.float32)
    full_loss = (parameter - samples).square().mean()
    full_loss.backward()
    expected_gradient = parameter.grad.detach().clone()

    parameter.grad = None
    slices = physical_microbatch_slices(70)
    assert slices == ((0, 32), (32, 64), (64, 70))
    for start, stop in slices:
        ((parameter - samples[start:stop]).square().sum() / 70).backward()
    assert torch.allclose(parameter.grad, expected_gradient, rtol=0.0, atol=1e-5)


def test_ppo_trainer_updates_once_from_saved_snapshots_with_exact_accumulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.env.frontier import FrontierGenerator
    from lunar_exploration_ppo.ppo.trainer import PPOTrainer, policy_state_sha256

    policy = _ToyPolicy()
    batch = _trainer_batch(policy)
    before = policy_state_sha256(policy)

    def forbidden_extract(*_args, **_kwargs):
        raise AssertionError("frontier extractor must not run during PPO update")

    monkeypatch.setattr(FrontierGenerator, "extract", forbidden_extract)
    metrics = PPOTrainer(policy, device="cpu", shuffle_seed=17).update(batch)

    assert metrics.initial_ratio_max_abs_error <= 1e-6
    assert metrics.optimizer_steps == 16
    assert metrics.early_stopped is False
    assert metrics.effective_minibatch_sizes == (256,) * 16
    assert metrics.physical_microbatch_sizes == (32,) * 128
    assert metrics.grad_pre_clip_norm_max > 0.0
    assert metrics.grad_post_clip_norm_max <= 0.500001
    assert metrics.parameter_change_l2 > 0.0
    assert metrics.frontier_entropy_mean >= metrics.frontier_entropy_min >= 0.0
    assert metrics.theta_kappa_mean == pytest.approx(1.0)
    assert metrics.theta_kappa_max == pytest.approx(1.0)
    assert metrics.policy_state_sha256_before == before
    assert metrics.policy_state_sha256_after == policy_state_sha256(policy)
    assert metrics.policy_state_sha256_after != before
    assert batch.consumed


def test_ppo_trainer_initial_ratio_mismatch_fails_before_optimizer_step() -> None:
    from lunar_exploration_ppo.ppo.trainer import (
        PPOTrainer,
        PPOTrainingError,
        policy_state_sha256,
    )

    policy = _ToyPolicy()
    batch = _trainer_batch(policy, old_frontier_offset=0.01)
    before = policy_state_sha256(policy)
    with pytest.raises(PPOTrainingError, match="initial ratio"):
        PPOTrainer(policy, device="cpu", shuffle_seed=5).update(batch)
    assert policy_state_sha256(policy) == before
    with pytest.raises(RolloutContractError, match="invalid"):
        batch.claim_for_update(before)


def test_ppo_trainer_stops_all_remaining_minibatches_after_target_kl() -> None:
    from lunar_exploration_ppo.ppo.trainer import PPOTrainer

    policy = _ToyPolicy(logit_scale=100_000.0)
    batch = _trainer_batch(policy)
    metrics = PPOTrainer(policy, device="cpu", shuffle_seed=11).update(batch)

    assert metrics.early_stopped
    assert metrics.approx_kl > 0.03
    assert metrics.optimizer_steps == 1
    assert metrics.early_stop_epoch == 0
    assert metrics.early_stop_minibatch == 1
    assert metrics.skipped_minibatch_count == 14
    assert metrics.grad_post_clip_norm_max <= 0.500001
    assert batch.consumed


class _JsonStateComponent:
    def __init__(self, state: dict[str, object]) -> None:
        self._state = copy.deepcopy(state)
        self.restore_count = 0

    def restore_state(self, state: dict[str, object]) -> None:
        self._state = copy.deepcopy(state)
        self.restore_count += 1

    def capture_state(self) -> dict[str, object]:
        return copy.deepcopy(self._state)


class _VectorStateComponent:
    def __init__(self) -> None:
        self._states = tuple(
            {"worker": index, "episode": -1} for index in range(8)
        )
        self.restore_count = 0

    def restore_states(self, states) -> None:
        self._states = tuple(copy.deepcopy(dict(state)) for state in states)
        self.restore_count += 1

    def capture_states(self) -> tuple[dict[str, object], ...]:
        return tuple(copy.deepcopy(state) for state in self._states)


def test_training_resume_restores_fresh_runtime_and_matches_uninterrupted_update(
    tmp_path: Path,
) -> None:
    import lunar_exploration_ppo.ppo as ppo
    from lunar_exploration_ppo.ppo.checkpoint import CheckpointManager
    from lunar_exploration_ppo.ppo.trainer import PPOTrainer, policy_state_sha256

    random.seed(73)
    np.random.seed(73)
    torch.manual_seed(73)
    source_policy = _ToyPolicy()
    source_trainer = PPOTrainer(
        source_policy,
        device="cpu",
        shuffle_seed=41,
    )
    first_metrics = source_trainer.update(_trainer_batch(source_policy))
    assert first_metrics.update_step == 1

    normalization_state = {"count": 19, "mean": [1.5, -2.0]}
    sampler_state = {"seed": 73, "cursor": 11}
    vector_states = tuple(
        {"worker": index, "episode": index + 20} for index in range(8)
    )
    config_sha256 = "a" * 64
    lineage = {
        "stage3_commit": "4df7be92cd6517e77c648e890bf7d32c9fcd560b",
        "stage3_gate_sha256": (
            "6ea3dc5bc199dd9370fab4b1a5aac1ebab08d9e42f49c0d01b3aa2b1e9aae0f0"
        ),
        "stage3_approval_sha256": (
            "8e901e04d587110de681da302ad24025dfebed945ff646ffa1e85bc5f5ecb3f2"
        ),
    }
    manager = CheckpointManager(tmp_path / "resume-checkpoints")
    manager.save_complete(
        policy=source_policy,
        optimizer=source_trainer.optimizer,
        update_step=1,
        normalization_stats=normalization_state,
        scenario_sampler_state=sampler_state,
        vector_env_states=vector_states,
        best_record={"update_step": 1},
        versions={
            "observation_schema_version": "policy_observation/v1",
            "action_space_version": "frontier_index_continuous_theta/v1",
            "network_architecture_version": "toy_resume_policy/v1",
            "reward_version": "toy_resume_reward/v1",
            "frontier_version": "toy_resume_frontier/v1",
            "planner_version": "toy_resume_planner/v1",
        },
        top_m_config={"top_m": 2},
        scale_profile="resume_test/v1",
        training_config={"shuffle_seed": 41},
        config_sha256=config_sha256,
        lineage=lineage,
        eval_metrics={},
    )
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)

    uninterrupted = source_trainer.update(_trainer_batch(source_policy))
    uninterrupted_policy_hash = policy_state_sha256(source_policy)

    fresh_policy = _ToyPolicy()
    fresh_trainer = PPOTrainer(
        fresh_policy,
        device="cpu",
        shuffle_seed=41,
    )
    assert fresh_trainer.update_step == 0
    normalizer = _JsonStateComponent({"count": 0})
    sampler = _JsonStateComponent({"seed": -1, "cursor": -1})
    vector_env = _VectorStateComponent()

    receipt = ppo.resume_training_from_last_checkpoint(
        checkpoint_manager=manager,
        trainer=fresh_trainer,
        normalizer=normalizer,
        scenario_sampler=sampler,
        vector_env=vector_env,
        expected_config_sha256=config_sha256,
        expected_lineage=lineage,
    )

    assert receipt.update_step == 1
    assert receipt.next_update_step == 2
    assert fresh_trainer.update_step == 1
    assert normalizer.restore_count == 1
    assert sampler.restore_count == 1
    assert vector_env.restore_count == 1
    assert normalizer.capture_state() == normalization_state
    assert sampler.capture_state() == sampler_state
    assert vector_env.capture_states() == vector_states
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)

    uninterrupted_optimizer_hash = ppo.optimizer_state_sha256(
        source_trainer.optimizer
    )
    resumed = fresh_trainer.update(_trainer_batch(fresh_policy))
    assert uninterrupted.update_step == resumed.update_step == 2
    assert policy_state_sha256(fresh_policy) == uninterrupted_policy_hash
    assert ppo.optimizer_state_sha256(
        fresh_trainer.optimizer
    ) == uninterrupted_optimizer_hash
