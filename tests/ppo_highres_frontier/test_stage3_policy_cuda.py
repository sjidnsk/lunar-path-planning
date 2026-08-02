"""CUDA-only Stage 3 policy numeric and masking contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from lunar_exploration_ppo.policy import (
    CrossAttentionFrontierPolicy,
    PolicyInputError,
    batch_policy_observations,
    masked_frontier_probabilities,
    recompute_action_log_probs,
    sample_action,
)
from lunar_exploration_ppo.policy.observation import PolicyObservation


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is genuinely unavailable",
)


def _observation(
    candidate_count: int,
    valid_count: int,
    *,
    seed: int,
    local_shape: tuple[int, int] = (64, 64),
) -> PolicyObservation:
    rng = np.random.default_rng(seed)
    mask = np.zeros(candidate_count, dtype=bool)
    mask[:valid_count] = True
    return PolicyObservation(
        prior_channels=rng.normal(size=(7, 32, 32)).astype(np.float32),
        coverage_summary=rng.normal(size=(8, 32, 32)).astype(np.float32),
        local_crop=rng.normal(size=(8, *local_shape)).astype(np.float32),
        frontier_features=rng.normal(size=(candidate_count, 22)).astype(np.float32),
        pose_features=rng.normal(size=(6,)).astype(np.float32),
        candidate_mask=mask,
    )


def test_cuda_forward_backward_is_finite_fp32_and_mask_gradient_is_exact_zero() -> None:
    device = torch.device("cuda")
    torch.manual_seed(20260714)
    torch.cuda.manual_seed_all(20260714)
    model = CrossAttentionFrontierPolicy().to(device)
    batch = batch_policy_observations(
        (
            _observation(32, 23, seed=1),
            _observation(32, 17, seed=2),
        ),
        device=device,
    )
    batch = replace(
        batch,
        frontier_features=batch.frontier_features.detach().clone().requires_grad_(True),
    )

    output = model(batch)
    probabilities = masked_frontier_probabilities(
        output.frontier_logits,
        batch.candidate_mask,
    )
    loss = (
        -torch.log(probabilities[:, 0]).mean()
        + output.value.square().mean()
        + output.theta_kappa[batch.candidate_mask].mean()
    )
    loss.backward()
    torch.cuda.synchronize()

    assert output.frontier_logits.device.type == "cuda"
    assert all(
        tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all())
        for tensor in (
            output.frontier_logits,
            output.theta_mu_sin_raw,
            output.theta_mu_cos_raw,
            output.theta_kappa_raw,
            output.theta_mu,
            output.theta_kappa,
            output.value,
            probabilities,
            batch.frontier_features.grad,
        )
    )
    assert torch.equal(
        probabilities[~batch.candidate_mask],
        torch.zeros_like(probabilities[~batch.candidate_mask]),
    )
    assert torch.count_nonzero(
        batch.frontier_features.grad[~batch.candidate_mask]
    ).item() == 0
    assert all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_cuda_deterministic_action_and_saved_logprobs_recompute_within_tolerance() -> None:
    device = torch.device("cuda")
    torch.manual_seed(47)
    model = CrossAttentionFrontierPolicy().to(device).eval()
    batch = batch_policy_observations(
        (
            _observation(40, 31, seed=11, local_shape=(96, 96)),
            _observation(36, 19, seed=12, local_shape=(96, 96)),
        ),
        device=device,
    )

    with torch.no_grad():
        output = model(batch)
        first = sample_action(output, batch.candidate_mask, deterministic=True)
        second = sample_action(output, batch.candidate_mask, deterministic=True)
        recomputed = recompute_action_log_probs(
            output,
            batch.candidate_mask,
            first.selected_frontier_index,
            first.selected_theta,
        )
    torch.cuda.synchronize()

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
    assert bool(
        batch.candidate_mask.gather(
            1,
            first.selected_frontier_index.unsqueeze(1),
        ).all()
    )
    assert bool((first.selected_theta >= -torch.pi).all())
    assert bool((first.selected_theta < torch.pi).all())
    for name in ("log_prob_frontier", "log_prob_theta", "log_prob_total"):
        error = torch.max(torch.abs(getattr(recomputed, name) - getattr(first, name)))
        assert error.item() <= 1e-5
    assert torch.equal(
        first.log_prob_total,
        first.log_prob_frontier + first.log_prob_theta,
    )


def test_cuda_autocast_is_rejected_before_encoder_and_never_falls_back_to_cpu() -> None:
    device = torch.device("cuda")
    model = CrossAttentionFrontierPolicy().to(device).eval()
    batch = batch_policy_observations(
        (_observation(8, 6, seed=21),),
        device=device,
    )
    encoder_calls = 0

    def mark_encoder_call(_module, _arguments) -> None:
        nonlocal encoder_calls
        encoder_calls += 1

    handle = model.global_encoder.register_forward_pre_hook(mark_encoder_call)
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with pytest.raises(PolicyInputError, match="autocast"):
                model(batch)
    finally:
        handle.remove()
    assert encoder_calls == 0
    assert next(model.parameters()).device.type == "cuda"
    assert all(tensor.device.type == "cuda" for tensor in (
        batch.prior_channels,
        batch.coverage_summary,
        batch.local_crop,
        batch.frontier_features,
        batch.pose_features,
        batch.candidate_mask,
    ))
