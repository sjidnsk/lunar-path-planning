from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch
import torch.nn.functional as F


CONTINUOUS_THETA_ACTION_SPACE = "hybrid_discrete_xy_continuous_theta/v1"


def normalize_theta_rad(theta_rad: float | torch.Tensor) -> float | torch.Tensor:
    if isinstance(theta_rad, torch.Tensor):
        return torch.remainder(theta_rad + math.pi, 2.0 * math.pi) - math.pi
    return float((float(theta_rad) + math.pi) % (2.0 * math.pi) - math.pi)


def theta_distribution_parameters(
    theta_mu_sin: torch.Tensor,
    theta_mu_cos: torch.Tensor,
    theta_kappa_raw: torch.Tensor,
    *,
    max_kappa: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    theta_mu_rad = normalize_theta_rad(torch.atan2(theta_mu_sin, theta_mu_cos))
    theta_kappa = torch.clamp(F.softplus(theta_kappa_raw) + 1.0e-3, min=1.0e-3, max=float(max_kappa))
    return theta_mu_rad, theta_kappa


def continuous_theta_log_prob(
    *,
    point_logits: torch.Tensor,
    theta_mu_rad: torch.Tensor,
    theta_kappa: torch.Tensor,
    action_index: int,
    theta_rad: float | torch.Tensor,
) -> dict[str, Any]:
    action_index = int(action_index)
    if action_index < 0 or action_index >= int(point_logits.shape[0]):
        raise ValueError("continuous_theta_action_index_out_of_range")
    theta_tensor = (
        theta_rad.detach().clone().to(dtype=point_logits.dtype, device=point_logits.device)
        if isinstance(theta_rad, torch.Tensor)
        else torch.tensor(float(theta_rad), dtype=point_logits.dtype, device=point_logits.device)
    )
    theta_tensor = normalize_theta_rad(theta_tensor)
    point_distribution = torch.distributions.Categorical(logits=point_logits)
    selected_action = torch.tensor(action_index, dtype=torch.long, device=point_logits.device)
    theta_distribution = torch.distributions.VonMises(theta_mu_rad[action_index], theta_kappa[action_index])
    point_log_prob = point_distribution.log_prob(selected_action)
    theta_log_prob = theta_distribution.log_prob(theta_tensor)
    total = point_log_prob + theta_log_prob
    return {
        "action_space_type": CONTINUOUS_THETA_ACTION_SPACE,
        "old_point_log_prob": float(point_log_prob.detach().cpu()),
        "old_theta_log_prob": float(theta_log_prob.detach().cpu()),
        "old_log_prob": float(total.detach().cpu()),
        "selected_theta_rad": float(theta_tensor.detach().cpu()),
        "selected_theta_deg": math.degrees(float(theta_tensor.detach().cpu())),
    }


def continuous_theta_torch_log_prob(
    *,
    point_logits: torch.Tensor,
    theta_mu_rad: torch.Tensor,
    theta_kappa: torch.Tensor,
    action_index: int,
    theta_rad: float,
) -> dict[str, torch.Tensor]:
    action_index = int(action_index)
    action = torch.tensor(action_index, dtype=torch.long, device=point_logits.device)
    theta_tensor = torch.tensor(float(theta_rad), dtype=point_logits.dtype, device=point_logits.device)
    theta_tensor = normalize_theta_rad(theta_tensor)
    point_distribution = torch.distributions.Categorical(logits=point_logits)
    theta_distribution = torch.distributions.VonMises(theta_mu_rad[action_index], theta_kappa[action_index])
    point_log_prob = point_distribution.log_prob(action)
    theta_log_prob = theta_distribution.log_prob(theta_tensor)
    try:
        theta_entropy = theta_distribution.entropy()
    except NotImplementedError:
        theta_entropy = torch.zeros((), dtype=point_logits.dtype, device=point_logits.device)
    point_entropy = point_distribution.entropy()
    return {
        "point_log_prob": point_log_prob,
        "theta_log_prob": theta_log_prob,
        "total_log_prob": point_log_prob + theta_log_prob,
        "point_entropy": point_entropy,
        "theta_entropy": theta_entropy,
        "total_entropy": point_entropy + theta_entropy,
    }


def sample_continuous_theta_action(
    *,
    point_logits: torch.Tensor,
    theta_mu_rad: torch.Tensor,
    theta_kappa: torch.Tensor,
    sampling_mask: torch.Tensor,
    temperature: float,
) -> dict[str, Any]:
    if temperature <= 0.0 or not math.isfinite(float(temperature)):
        raise ValueError("sampling_temperature must be finite and > 0")
    masked_logits = (point_logits / float(temperature)).masked_fill(~sampling_mask.bool(), -1.0e9)
    point_distribution = torch.distributions.Categorical(logits=masked_logits)
    action = point_distribution.sample()
    action_index = int(action.item())
    theta_distribution = torch.distributions.VonMises(theta_mu_rad[action_index], theta_kappa[action_index])
    theta_sample = normalize_theta_rad(theta_distribution.sample())
    point_log_prob = point_distribution.log_prob(action)
    theta_log_prob = theta_distribution.log_prob(theta_sample)
    probs = torch.softmax(masked_logits, dim=-1)
    try:
        theta_entropy_value: float | None = float(theta_distribution.entropy().detach().cpu())
    except NotImplementedError:
        theta_entropy_value = None
    point_entropy = point_distribution.entropy()
    total_entropy = point_entropy if theta_entropy_value is None else point_entropy + theta_distribution.entropy()
    return {
        "action_space_type": CONTINUOUS_THETA_ACTION_SPACE,
        "action_index": action_index,
        "selected_base_candidate_index": action_index,
        "selected_theta_rad": float(theta_sample.detach().cpu()),
        "selected_theta_deg": math.degrees(float(theta_sample.detach().cpu())),
        "old_point_log_prob": float(point_log_prob.detach().cpu()),
        "old_theta_log_prob": float(theta_log_prob.detach().cpu()),
        "old_log_prob": float((point_log_prob + theta_log_prob).detach().cpu()),
        "old_action_probs": [float(item) for item in probs.detach().cpu()],
        "old_sampling_logits": [float(item) for item in masked_logits.detach().cpu()],
        "argmax_action_index": int(torch.argmax(probs).item()),
        "selected_probability": float(probs[action_index]),
        "point_action_entropy": float(point_entropy.detach().cpu()),
        "theta_action_entropy": theta_entropy_value,
        "action_entropy": float(total_entropy.detach().cpu()),
    }


def action_sample_hash(
    *,
    base_candidate_set_hash: str,
    action_index: int,
    theta_rad: float,
    sampling_seed: int,
) -> str:
    payload = {
        "base_candidate_set_hash": str(base_candidate_set_hash),
        "action_index": int(action_index),
        "theta_rad": round(float(normalize_theta_rad(theta_rad)), 12),
        "sampling_seed": int(sampling_seed),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def continuous_theta_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("continuous_theta_action_space_enabled", False)) or (
        config.get("action_space_type") == CONTINUOUS_THETA_ACTION_SPACE
    )
