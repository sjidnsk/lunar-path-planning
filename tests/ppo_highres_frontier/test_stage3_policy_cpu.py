from __future__ import annotations

import importlib
import math
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from lunar_exploration_ppo.policy.observation import PolicyObservation


ROOT = Path(__file__).resolve().parents[2]


def _policy_module():
    try:
        return importlib.import_module(
            "lunar_exploration_ppo.policy.cross_attention"
        )
    except ModuleNotFoundError:
        pytest.fail("Stage 3 cross-attention policy module is missing")


def _observation(
    candidate_count: int,
    valid_count: int,
    *,
    seed: int = 0,
    low_shape: tuple[int, int] = (32, 32),
    local_shape: tuple[int, int] = (64, 64),
) -> PolicyObservation:
    rng = np.random.default_rng(seed)
    candidate_mask = np.zeros(candidate_count, dtype=bool)
    candidate_mask[:valid_count] = True
    return PolicyObservation(
        prior_channels=rng.normal(size=(7, *low_shape)).astype(np.float32),
        coverage_summary=rng.normal(size=(8, *low_shape)).astype(np.float32),
        local_crop=rng.normal(size=(8, *local_shape)).astype(np.float32),
        frontier_features=rng.normal(size=(candidate_count, 22)).astype(np.float32),
        pose_features=rng.normal(size=(6,)).astype(np.float32),
        candidate_mask=candidate_mask,
    )


def test_stage3_architecture_metadata_and_forbidden_modules_are_exact() -> None:
    policy = _policy_module()
    model = policy.CrossAttentionFrontierPolicy()

    assert model.architecture_metadata() == {
        "network_architecture_version": "cross_attention_frontier_policy/v1",
        "network_memory_mode": "stateless_observation_only/v1",
        "prior_channels": 7,
        "coverage_summary_channels": 8,
        "global_encoder_input_channels": 15,
        "local_crop_channels": 8,
        "frontier_feature_dim": 22,
        "pose_feature_dim": 6,
        "map_pool_shape": [16, 16],
        "global_token_count": 256,
        "local_token_count": 256,
        "pose_token_count": 1,
        "context_token_count": 513,
        "token_dim": 128,
        "cross_attention_layers": 2,
        "attention_heads": 4,
        "ffn_hidden_dim": 512,
        "action_hidden_dim": 128,
        "dropout": 0.0,
        "cross_attention_direction": (
            "frontier_queries_to_context_keys_values/v1"
        ),
    }
    assert len(model.cross_attention_blocks) == 2
    assert all(
        isinstance(block.attention, nn.MultiheadAttention)
        and block.attention.num_heads == 4
        and block.attention.dropout == 0.0
        for block in model.cross_attention_blocks
    )
    modules = tuple(model.modules())
    assert any(isinstance(module, nn.GroupNorm) for module in modules)
    assert any(isinstance(module, nn.LayerNorm) for module in modules)
    assert not any(
        isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.RNNBase))
        for module in modules
    )
    assert not any(isinstance(module, nn.Dropout) for module in modules)
    assert not hasattr(model, "candidate_self_attention")
    assert not hasattr(model, "recurrent_state")
    assert torch.float32 == next(model.parameters()).dtype


def test_batch_policy_observations_pads_variable_candidates_with_typed_tensors() -> None:
    policy = _policy_module()
    batch = policy.batch_policy_observations(
        (_observation(5, 4, seed=1), _observation(3, 2, seed=2))
    )

    assert isinstance(batch, policy.PolicyBatch)
    assert batch.prior_channels.shape == (2, 7, 32, 32)
    assert batch.coverage_summary.shape == (2, 8, 32, 32)
    assert batch.local_crop.shape == (2, 8, 64, 64)
    assert batch.frontier_features.shape == (2, 5, 22)
    assert batch.pose_features.shape == (2, 6)
    assert batch.candidate_mask.shape == (2, 5)
    assert all(
        tensor.dtype == torch.float32
        for tensor in (
            batch.prior_channels,
            batch.coverage_summary,
            batch.local_crop,
            batch.frontier_features,
            batch.pose_features,
        )
    )
    assert batch.candidate_mask.dtype == torch.bool
    assert torch.equal(
        batch.candidate_mask,
        torch.tensor(
            [[True, True, True, True, False], [True, True, False, False, False]]
        ),
    )
    assert torch.count_nonzero(batch.frontier_features[1, 3:]) == 0
    assert not hasattr(batch, "truth")
    assert not hasattr(batch, "coverable_mask")


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda obs: replace(obs, prior_channels=np.zeros((6, 32, 32), np.float32)),
            "prior_channels",
        ),
        (
            lambda obs: replace(
                obs, coverage_summary=np.zeros((8, 31, 32), np.float32)
            ),
            "coverage_summary",
        ),
        (
            lambda obs: replace(obs, local_crop=np.zeros((7, 64, 64), np.float32)),
            "local_crop",
        ),
        (
            lambda obs: replace(obs, frontier_features=np.zeros((4, 21), np.float32)),
            "frontier_features",
        ),
        (
            lambda obs: replace(obs, pose_features=np.zeros((5,), np.float32)),
            "pose_features",
        ),
        (
            lambda obs: replace(obs, candidate_mask=np.ones((3,), dtype=bool)),
            "candidate_mask",
        ),
        (
            lambda obs: replace(obs, candidate_mask=np.ones((4,), dtype=np.int8)),
            "boolean",
        ),
        (
            lambda obs: replace(
                obs,
                frontier_features=np.full((4, 22), np.nan, dtype=np.float32),
            ),
            "finite",
        ),
        (
            lambda obs: replace(obs, candidate_mask=np.zeros((4,), dtype=bool)),
            "valid candidate",
        ),
    ],
)
def test_batch_policy_observations_rejects_malformed_rows_before_network(
    mutate,
    message: str,
) -> None:
    policy = _policy_module()
    with pytest.raises(policy.PolicyInputError, match=message):
        policy.batch_policy_observations((mutate(_observation(4, 3)),))


def test_batch_policy_observations_rejects_empty_or_mismatched_batch() -> None:
    policy = _policy_module()
    with pytest.raises(policy.PolicyInputError, match="at least one"):
        policy.batch_policy_observations(())
    with pytest.raises(policy.PolicyInputError, match="spatial shapes"):
        policy.batch_policy_observations(
            (_observation(4, 3), _observation(4, 2, low_shape=(16, 16)))
        )


@pytest.mark.parametrize(
    ("local_shape", "candidate_counts"),
    [
        ((64, 64), ((5, 4), (3, 2))),
        ((96, 96), ((7, 7), (6, 3))),
    ],
    ids=("smoke", "standard"),
)
def test_forward_smoke_and_standard_shapes_are_finite_fp32(
    local_shape: tuple[int, int],
    candidate_counts: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    policy = _policy_module()
    observations = tuple(
        _observation(total, valid, seed=index + 10, local_shape=local_shape)
        for index, (total, valid) in enumerate(candidate_counts)
    )
    batch = policy.batch_policy_observations(observations)
    model = policy.CrossAttentionFrontierPolicy().eval()

    output = model(batch)

    batch_size, max_candidates = batch.candidate_mask.shape
    assert isinstance(output, policy.PolicyForwardOutput)
    assert output.global_map_tokens.shape == (batch_size, 256, 128)
    assert output.local_map_tokens.shape == (batch_size, 256, 128)
    assert output.pose_token.shape == (batch_size, 1, 128)
    assert output.context_tokens.shape == (batch_size, 513, 128)
    assert output.refined_frontier_tokens.shape == (
        batch_size,
        max_candidates,
        128,
    )
    assert output.action_hidden.shape == (batch_size, max_candidates, 128)
    expected_candidate_shape = (batch_size, max_candidates)
    for tensor in (
        output.frontier_logits,
        output.theta_mu_sin_raw,
        output.theta_mu_cos_raw,
        output.theta_kappa_raw,
        output.theta_mu,
        output.theta_kappa,
    ):
        assert tensor.shape == expected_candidate_shape
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()
    assert output.value.shape == (batch_size,)
    assert output.value.dtype == torch.float32
    assert torch.isfinite(output.value).all()
    assert torch.all(output.theta_mu >= -torch.pi)
    assert torch.all(output.theta_mu < torch.pi)
    assert torch.all(output.theta_kappa >= 1e-3)
    assert torch.all(output.theta_kappa <= 20.0)


def test_padding_feature_and_count_do_not_change_valid_outputs_or_value() -> None:
    policy = _policy_module()
    torch.manual_seed(17)
    model = policy.CrossAttentionFrontierPolicy().eval()
    baseline = _observation(5, 3, seed=31)
    mutated_features = baseline.frontier_features.copy()
    mutated_features[3:] = np.float32(1.0e6)
    mutated = replace(baseline, frontier_features=mutated_features)

    baseline_output = model(policy.batch_policy_observations((baseline,)))
    mutated_output = model(policy.batch_policy_observations((mutated,)))

    for name in (
        "frontier_logits",
        "theta_mu_sin_raw",
        "theta_mu_cos_raw",
        "theta_kappa_raw",
        "theta_mu",
        "theta_kappa",
    ):
        assert torch.equal(
            getattr(baseline_output, name)[:, :3],
            getattr(mutated_output, name)[:, :3],
        )
    assert torch.equal(baseline_output.value, mutated_output.value)

    short = replace(
        baseline,
        frontier_features=baseline.frontier_features[:3].copy(),
        candidate_mask=np.ones(3, dtype=bool),
    )
    long_features = np.zeros((8, 22), dtype=np.float32)
    long_features[:3] = baseline.frontier_features[:3]
    long_features[3:] = np.float32(-1.0e6)
    long = replace(
        baseline,
        frontier_features=long_features,
        candidate_mask=np.asarray([True, True, True, False, False, False, False, False]),
    )
    short_output = model(policy.batch_policy_observations((short,)))
    long_output = model(policy.batch_policy_observations((long,)))
    torch.testing.assert_close(
        short_output.frontier_logits,
        long_output.frontier_logits[:, :3],
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        short_output.theta_mu,
        long_output.theta_mu[:, :3],
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        short_output.theta_kappa,
        long_output.theta_kappa[:, :3],
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        short_output.value,
        long_output.value,
        rtol=0.0,
        atol=1e-6,
    )


def test_value_mlp_receives_mask_aware_refined_mean_and_max() -> None:
    policy = _policy_module()
    torch.manual_seed(23)
    model = policy.CrossAttentionFrontierPolicy().eval()
    batch = policy.batch_policy_observations((_observation(6, 3, seed=41),))
    captured: list[torch.Tensor] = []

    def capture_value_input(_module, arguments) -> None:
        captured.append(arguments[0].detach().clone())

    handle = model.value_mlp[0].register_forward_pre_hook(capture_value_input)
    try:
        output = model(batch)
    finally:
        handle.remove()

    valid = output.refined_frontier_tokens[:, :3]
    expected = torch.cat(
        (
            valid.mean(dim=1),
            valid.amax(dim=1),
            output.context_tokens.mean(dim=1),
        ),
        dim=-1,
    )
    assert len(captured) == 1
    assert torch.equal(captured[0], expected)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda batch: replace(
                batch,
                candidate_mask=torch.zeros_like(batch.candidate_mask),
            ),
            "valid candidate",
        ),
        (
            lambda batch: replace(
                batch,
                candidate_mask=batch.candidate_mask.to(torch.int8),
            ),
            "boolean",
        ),
        (
            lambda batch: replace(
                batch,
                frontier_features=batch.frontier_features[..., :21],
            ),
            "frontier_features",
        ),
        (
            lambda batch: replace(
                batch,
                prior_channels=batch.prior_channels.clone().index_put_(
                    (torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0])),
                    torch.tensor([float("nan")]),
                ),
            ),
            "finite",
        ),
    ],
)
def test_forward_rejects_invalid_policy_batch_before_any_encoder(
    mutate,
    message: str,
) -> None:
    policy = _policy_module()
    model = policy.CrossAttentionFrontierPolicy().eval()
    batch = policy.batch_policy_observations((_observation(4, 3, seed=51),))
    encoder_calls = 0

    def mark_encoder_call(_module, _arguments) -> None:
        nonlocal encoder_calls
        encoder_calls += 1

    handle = model.global_encoder.register_forward_pre_hook(mark_encoder_call)
    try:
        with pytest.raises(policy.PolicyInputError, match=message):
            model(mutate(batch))
    finally:
        handle.remove()
    assert encoder_calls == 0


def test_theta_and_frontier_head_initialization_is_frozen() -> None:
    policy = _policy_module()
    model = policy.CrossAttentionFrontierPolicy()
    expected_kappa_raw = math.log(math.expm1(0.1 - 1e-3))

    assert torch.count_nonzero(model.theta_sin_head.weight) == 0
    assert torch.count_nonzero(model.theta_cos_head.weight) == 0
    assert torch.count_nonzero(model.theta_kappa_head.weight) == 0
    assert torch.equal(model.theta_sin_head.bias, torch.zeros_like(model.theta_sin_head.bias))
    assert torch.equal(model.theta_cos_head.bias, torch.ones_like(model.theta_cos_head.bias))
    assert model.theta_kappa_head.bias.item() == pytest.approx(
        expected_kappa_raw,
        rel=0.0,
        abs=1e-7,
    )
    assert torch.equal(
        model.frontier_logit_head.bias,
        torch.zeros_like(model.frontier_logit_head.bias),
    )
    assert torch.max(torch.abs(model.frontier_logit_head.weight)).item() <= 0.01

    output = model(
        policy.batch_policy_observations((_observation(4, 3, seed=61),))
    )
    assert torch.equal(output.theta_mu_sin_raw, torch.zeros_like(output.theta_mu_sin_raw))
    assert torch.equal(output.theta_mu_cos_raw, torch.ones_like(output.theta_mu_cos_raw))
    torch.testing.assert_close(
        output.theta_kappa,
        torch.full_like(output.theta_kappa, 0.1),
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize("raw_bias", (0.0, 1e-12), ids=("zero", "near-zero"))
def test_zero_or_near_zero_theta_direction_uses_finite_recommended_fallback(
    raw_bias: float,
) -> None:
    policy = _policy_module()
    model = policy.CrossAttentionFrontierPolicy()
    with torch.no_grad():
        model.theta_sin_head.weight.zero_()
        model.theta_cos_head.weight.zero_()
        model.theta_sin_head.bias.fill_(raw_bias)
        model.theta_cos_head.bias.fill_(-raw_bias)
    observation = _observation(3, 3, seed=71)
    features = observation.frontier_features.copy()
    features[:, 14:16] = np.asarray(
        [[1.0, 0.0], [0.0, 0.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    observation = replace(observation, frontier_features=features)
    batch = policy.batch_policy_observations((observation,))
    batch = replace(
        batch,
        frontier_features=batch.frontier_features.detach().clone().requires_grad_(True),
    )

    output = model(batch)
    expected = torch.tensor([[torch.pi / 2.0, 0.0, -torch.pi]], dtype=torch.float32)
    torch.testing.assert_close(output.theta_mu, expected, rtol=0.0, atol=1e-6)
    loss = output.theta_mu.sum() + output.theta_kappa.sum() + output.value.sum()
    loss.backward()

    assert torch.isfinite(batch.frontier_features.grad).all()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_masked_categorical_has_exact_zero_probability_and_padding_gradient() -> None:
    policy = _policy_module()
    torch.manual_seed(83)
    model = policy.CrossAttentionFrontierPolicy()
    batch = policy.batch_policy_observations((_observation(5, 3, seed=81),))
    batch = replace(
        batch,
        frontier_features=batch.frontier_features.detach().clone().requires_grad_(True),
    )

    output = model(batch)
    probabilities = policy.masked_frontier_probabilities(
        output.frontier_logits,
        batch.candidate_mask,
    )

    assert torch.equal(
        output.frontier_logits[:, 3:],
        torch.full_like(output.frontier_logits[:, 3:], -1.0e9),
    )
    assert torch.equal(probabilities[:, 3:], torch.zeros_like(probabilities[:, 3:]))
    assert torch.equal(probabilities.sum(dim=-1), torch.ones(1, dtype=torch.float32))
    loss = -torch.log(probabilities[:, 0]).sum() + output.value.sum()
    loss.backward()
    assert torch.count_nonzero(batch.frontier_features.grad[:, 3:]) == 0
    assert torch.isfinite(batch.frontier_features.grad).all()


def _extreme_masked_distribution_fixture():
    policy = _policy_module()
    model = policy.CrossAttentionFrontierPolicy().eval()
    batch = policy.batch_policy_observations((_observation(3, 1, seed=87),))
    output = model(batch)
    logits = torch.tensor(
        [[-1.0e20, 7.0, 8.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    return policy, batch, replace(output, frontier_logits=logits), logits


def test_extreme_valid_logit_keeps_masked_probability_exactly_zero() -> None:
    policy, batch, output, _logits = _extreme_masked_distribution_fixture()

    probabilities = policy.masked_frontier_probabilities(
        output.frontier_logits,
        batch.candidate_mask,
    )

    assert torch.equal(probabilities, torch.tensor([[1.0, 0.0, 0.0]]))


def test_extreme_valid_logit_deterministic_action_remains_legal() -> None:
    policy, batch, output, _logits = _extreme_masked_distribution_fixture()

    action = policy.sample_action(output, batch.candidate_mask, deterministic=True)

    assert torch.equal(action.selected_frontier_index, torch.tensor([0]))
    assert batch.candidate_mask.gather(
        1,
        action.selected_frontier_index.unsqueeze(1),
    ).all()


def test_extreme_valid_logit_stochastic_sampling_never_selects_masked_candidate() -> None:
    policy, batch, output, _logits = _extreme_masked_distribution_fixture()
    torch.manual_seed(88)

    selected = [
        policy.sample_action(
            output,
            batch.candidate_mask,
            deterministic=False,
        ).selected_frontier_index.item()
        for _ in range(64)
    ]

    assert selected == [0] * 64


def test_extreme_valid_logit_distribution_has_zero_masked_gradient() -> None:
    policy, batch, output, logits = _extreme_masked_distribution_fixture()
    selected_index = torch.tensor([0], dtype=torch.int64)
    selected_theta = output.theta_mu[:, 0].detach()

    evaluation = policy.recompute_action_log_probs(
        output,
        batch.candidate_mask,
        selected_index,
        selected_theta,
    )
    assert torch.equal(evaluation.log_prob_frontier, torch.zeros(1))
    (-evaluation.log_prob_total.sum()).backward()

    assert logits.grad is not None
    assert torch.equal(logits.grad[:, 1:], torch.zeros_like(logits.grad[:, 1:]))
    assert torch.isfinite(logits.grad).all()


def test_masked_distribution_still_rejects_nonfinite_raw_logits() -> None:
    policy, batch, output, _logits = _extreme_masked_distribution_fixture()
    nonfinite = output.frontier_logits.detach().clone()
    nonfinite[0, 0] = float("-inf")

    with pytest.raises(policy.PolicyActionError, match="finite"):
        policy.masked_frontier_probabilities(nonfinite, batch.candidate_mask)


def test_deterministic_action_is_bit_exact_and_logprob_recomputes_within_cpu_tolerance() -> None:
    policy = _policy_module()
    torch.manual_seed(89)
    model = policy.CrossAttentionFrontierPolicy().eval()
    batch = policy.batch_policy_observations(
        (_observation(6, 4, seed=91), _observation(5, 2, seed=92))
    )
    output = model(batch)

    first = policy.sample_action(output, batch.candidate_mask, deterministic=True)
    second = policy.sample_action(output, batch.candidate_mask, deterministic=True)

    for name in (
        "selected_frontier_index",
        "selected_theta",
        "log_prob_frontier",
        "log_prob_theta",
        "log_prob_total",
        "frontier_entropy",
        "value",
    ):
        assert torch.equal(getattr(first, name), getattr(second, name))
    expected_index = output.frontier_logits.argmax(dim=-1)
    assert torch.equal(first.selected_frontier_index, expected_index)
    expected_theta = output.theta_mu.gather(1, expected_index.unsqueeze(1)).squeeze(1)
    assert torch.equal(first.selected_theta, expected_theta)
    assert torch.equal(
        first.log_prob_total,
        first.log_prob_frontier + first.log_prob_theta,
    )
    assert torch.all(first.selected_theta >= -torch.pi)
    assert torch.all(first.selected_theta < torch.pi)

    recomputed = policy.recompute_action_log_probs(
        output,
        batch.candidate_mask,
        first.selected_frontier_index,
        first.selected_theta,
    )
    assert torch.max(
        torch.abs(recomputed.log_prob_total - first.log_prob_total)
    ).item() <= 1e-6
    assert torch.max(
        torch.abs(recomputed.log_prob_frontier - first.log_prob_frontier)
    ).item() <= 1e-6
    assert torch.max(
        torch.abs(recomputed.log_prob_theta - first.log_prob_theta)
    ).item() <= 1e-6


def test_stochastic_sampling_never_selects_masked_candidate() -> None:
    policy = _policy_module()
    torch.manual_seed(97)
    model = policy.CrossAttentionFrontierPolicy().eval()
    batch = policy.batch_policy_observations((_observation(7, 2, seed=101),))
    output = model(batch)

    for _ in range(128):
        action = policy.sample_action(output, batch.candidate_mask, deterministic=False)
        assert batch.candidate_mask[0, action.selected_frontier_index.item()]
        assert torch.isfinite(action.selected_theta).all()
        assert torch.isfinite(action.log_prob_total).all()


@pytest.mark.parametrize(
    ("selected_index", "message"),
    [(-1, "negative"), (5, "out of range"), (3, "masked")],
)
def test_logprob_recomputation_rejects_illegal_frontier_indices(
    selected_index: int,
    message: str,
) -> None:
    policy = _policy_module()
    model = policy.CrossAttentionFrontierPolicy().eval()
    batch = policy.batch_policy_observations((_observation(5, 3, seed=103),))
    output = model(batch)
    with pytest.raises(policy.PolicyActionError, match=message):
        policy.recompute_action_log_probs(
            output,
            batch.candidate_mask,
            torch.tensor([selected_index], dtype=torch.int64),
            torch.tensor([0.0], dtype=torch.float32),
        )


def test_theta_periodicity_and_minus_pi_inclusive_boundary() -> None:
    policy = _policy_module()
    values = torch.tensor(
        [
            -3.0 * torch.pi,
            -torch.pi,
            torch.pi,
            3.0 * torch.pi,
            torch.pi - 1e-5,
        ],
        dtype=torch.float32,
    )
    normalized = policy.normalize_theta(values)
    expected = torch.tensor(
        [-torch.pi, -torch.pi, -torch.pi, -torch.pi, torch.pi - 1e-5],
        dtype=torch.float32,
    )
    torch.testing.assert_close(normalized, expected, rtol=0.0, atol=1e-6)
    assert torch.all(normalized >= -torch.pi)
    assert torch.all(normalized < torch.pi)

    model = policy.CrossAttentionFrontierPolicy().eval()
    batch = policy.batch_policy_observations((_observation(4, 4, seed=107),))
    output = model(batch)
    index = torch.tensor([1], dtype=torch.int64)
    theta = torch.tensor([0.37], dtype=torch.float32)
    baseline = policy.recompute_action_log_probs(
        output,
        batch.candidate_mask,
        index,
        theta,
    )
    periodic = policy.recompute_action_log_probs(
        output,
        batch.candidate_mask,
        index,
        theta + 2.0 * torch.pi,
    )
    torch.testing.assert_close(
        baseline.log_prob_theta,
        periodic.log_prob_theta,
        rtol=0.0,
        atol=1e-6,
    )


def test_candidate_permutation_is_equivariant_and_maps_deterministic_cell() -> None:
    policy = _policy_module()
    torch.manual_seed(109)
    model = policy.CrossAttentionFrontierPolicy().eval()
    observation = _observation(4, 4, seed=113)
    permutation = np.asarray([2, 0, 3, 1])
    permuted = replace(
        observation,
        frontier_features=observation.frontier_features[permutation].copy(),
        candidate_mask=observation.candidate_mask[permutation].copy(),
    )
    original_batch = policy.batch_policy_observations((observation,))
    permuted_batch = policy.batch_policy_observations((permuted,))
    original_output = model(original_batch)
    permuted_output = model(permuted_batch)

    for name in (
        "frontier_logits",
        "theta_mu",
        "theta_kappa",
        "refined_frontier_tokens",
        "action_hidden",
    ):
        torch.testing.assert_close(
            getattr(permuted_output, name),
            getattr(original_output, name)[:, permutation],
            rtol=0.0,
            atol=1e-6,
        )
    torch.testing.assert_close(
        permuted_output.value,
        original_output.value,
        rtol=0.0,
        atol=1e-6,
    )
    original_action = policy.sample_action(
        original_output,
        original_batch.candidate_mask,
        deterministic=True,
    )
    permuted_action = policy.sample_action(
        permuted_output,
        permuted_batch.candidate_mask,
        deterministic=True,
    )
    mapped_original_index = permutation[permuted_action.selected_frontier_index.item()]
    assert mapped_original_index == original_action.selected_frontier_index.item()
    assert torch.equal(permuted_action.selected_theta, original_action.selected_theta)


def test_changing_one_valid_candidate_does_not_mix_into_another_action_row() -> None:
    policy = _policy_module()
    torch.manual_seed(127)
    model = policy.CrossAttentionFrontierPolicy().eval()
    observation = _observation(4, 4, seed=131)
    changed_features = observation.frontier_features.copy()
    changed_features[1] += np.float32(100.0)
    changed = replace(observation, frontier_features=changed_features)

    baseline = model(policy.batch_policy_observations((observation,)))
    mutated = model(policy.batch_policy_observations((changed,)))

    assert torch.equal(
        baseline.refined_frontier_tokens[:, 0],
        mutated.refined_frontier_tokens[:, 0],
    )
    assert torch.equal(baseline.action_hidden[:, 0], mutated.action_hidden[:, 0])
    assert torch.equal(baseline.frontier_logits[:, 0], mutated.frontier_logits[:, 0])
    assert not torch.equal(baseline.value, mutated.value)


def test_forward_fails_closed_under_autocast_or_non_fp32_parameters() -> None:
    policy = _policy_module()
    batch = policy.batch_policy_observations((_observation(4, 3, seed=137),))
    model = policy.CrossAttentionFrontierPolicy().eval()
    encoder_calls = 0

    def mark_encoder_call(_module, _arguments) -> None:
        nonlocal encoder_calls
        encoder_calls += 1

    handle = model.global_encoder.register_forward_pre_hook(mark_encoder_call)
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            with pytest.raises(policy.PolicyInputError, match="autocast"):
                model(batch)
        model.half()
        with pytest.raises(policy.PolicyInputError, match="parameters.*FP32"):
            model(batch)
    finally:
        handle.remove()
    assert encoder_calls == 0


def test_stage3_policy_public_api_and_torch_dependency_are_frozen() -> None:
    public_api = importlib.import_module("lunar_exploration_ppo.policy")
    expected = {
        "ActionEvaluation",
        "ActionSample",
        "CrossAttentionFrontierPolicy",
        "PolicyActionError",
        "PolicyBatch",
        "PolicyForwardOutput",
        "PolicyInputError",
        "batch_policy_observations",
        "masked_frontier_probabilities",
        "normalize_theta",
        "recompute_action_log_probs",
        "sample_action",
    }
    assert expected <= set(public_api.__all__)
    assert all(hasattr(public_api, name) for name in expected)

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "torch>=2.12.1,<2.13" in metadata["project"]["dependencies"]
