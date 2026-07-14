"""Stage 4 严格 FP32 PPO update。"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch
from torch import nn

from lunar_exploration_ppo.policy.cross_attention import (
    PolicyForwardOutput,
    batch_policy_observations,
    recompute_action_log_probs,
)
from lunar_exploration_ppo.ppo.rollout import RolloutBatch, RolloutContractError


class PPOTrainingError(RuntimeError):
    """PPO 数学、batch 或 optimizer 状态不满足冻结合同。"""


@dataclass(frozen=True, slots=True)
class PPOLossTerms:
    ratio: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    frontier_entropy: torch.Tensor
    total_loss: torch.Tensor
    approx_kl: torch.Tensor


@dataclass(frozen=True, slots=True)
class PPOUpdateMetrics:
    update_step: int
    initial_ratio_max_abs_error: float
    policy_loss: float
    value_loss: float
    frontier_entropy_mean: float
    frontier_entropy_min: float
    frontier_entropy_by_valid_candidate_count: Mapping[int, float]
    theta_kappa_mean: float
    theta_kappa_max: float
    approx_kl: float
    optimizer_steps: int
    planned_optimizer_steps: int
    effective_minibatch_sizes: tuple[int, ...]
    physical_microbatch_sizes: tuple[int, ...]
    early_stopped: bool
    early_stop_epoch: int | None
    early_stop_minibatch: int | None
    skipped_minibatch_count: int
    grad_pre_clip_norm_max: float
    grad_post_clip_norm_max: float
    parameter_change_l2: float
    policy_state_sha256_before: str
    policy_state_sha256_after: str


def compute_ppo_loss_terms(
    *,
    new_log_prob_total: torch.Tensor,
    old_log_prob_total: torch.Tensor,
    normalized_advantage: torch.Tensor,
    new_value: torch.Tensor,
    old_value: torch.Tensor,
    returns: torch.Tensor,
    frontier_entropy: torch.Tensor,
) -> PPOLossTerms:
    """计算 joint-action clipped PPO 与 clipped value 的 batch mean。"""

    tensors = (
        new_log_prob_total,
        old_log_prob_total,
        normalized_advantage,
        new_value,
        old_value,
        returns,
        frontier_entropy,
    )
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32 for value in tensors):
        raise PPOTrainingError("PPO loss inputs must be FP32 tensors")
    shape = new_log_prob_total.shape
    if len(shape) != 1 or shape[0] == 0 or any(value.shape != shape for value in tensors):
        raise PPOTrainingError("PPO loss input shapes must match [N]")
    devices = {value.device for value in tensors}
    if len(devices) != 1:
        raise PPOTrainingError("PPO loss inputs must share one device")
    device = new_log_prob_total.device
    try:
        autocast_enabled = torch.is_autocast_enabled(device.type)
    except TypeError:  # pragma: no cover - older supported torch compatibility
        autocast_enabled = torch.is_autocast_enabled()
    if autocast_enabled:
        raise PPOTrainingError("AMP/autocast is forbidden for PPO")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise PPOTrainingError("PPO loss inputs must be finite")

    ratio = torch.exp(new_log_prob_total - old_log_prob_total)
    unclipped = ratio * normalized_advantage
    clipped = torch.clamp(ratio, 0.8, 1.2) * normalized_advantage
    policy_loss = -torch.minimum(unclipped, clipped).mean(dtype=torch.float32)
    value_clipped = old_value + torch.clamp(new_value - old_value, -0.2, 0.2)
    value_loss = 0.5 * torch.maximum(
        (new_value - returns).square(),
        (value_clipped - returns).square(),
    ).mean(dtype=torch.float32)
    entropy_mean = frontier_entropy.mean(dtype=torch.float32)
    total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_mean
    approx_kl = (old_log_prob_total - new_log_prob_total).mean(dtype=torch.float32)
    outputs = (ratio, policy_loss, value_loss, entropy_mean, total_loss, approx_kl)
    if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in outputs):
        raise PPOTrainingError("PPO loss outputs must be finite FP32")
    return PPOLossTerms(
        ratio=ratio,
        policy_loss=policy_loss,
        value_loss=value_loss,
        frontier_entropy=entropy_mean,
        total_loss=total_loss,
        approx_kl=approx_kl,
    )


def policy_state_sha256(policy: nn.Module) -> str:
    """对 state_dict 的名称、dtype、shape 和原始 bytes 做稳定 SHA-256。"""

    if not isinstance(policy, nn.Module):
        raise PPOTrainingError("policy hash requires a torch module")
    digest = hashlib.sha256()
    state = policy.state_dict()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise PPOTrainingError("policy state contains a non-tensor value")
        contiguous = tensor.detach().contiguous().cpu()
        raw = contiguous.numpy().tobytes(order="C")
        for value in (
            name.encode("utf-8"),
            str(contiguous.dtype).encode("ascii"),
            np.asarray(tuple(contiguous.shape), dtype="<i8").tobytes(),
            raw,
        ):
            digest.update(struct.pack("<Q", len(value)))
            digest.update(value)
    return digest.hexdigest()


def physical_microbatch_slices(sample_count: int) -> tuple[tuple[int, int], ...]:
    if type(sample_count) is not int or sample_count <= 0:
        raise PPOTrainingError("physical microbatch sample count must be positive")
    return tuple(
        (start, min(start + 32, sample_count))
        for start in range(0, sample_count, 32)
    )


class PPOTrainer:
    """固定 v1 超参数、只消费保存 snapshot 的单次 PPO updater。"""

    PPO_EPOCHS = 4
    EFFECTIVE_MINIBATCH_SIZE = 256
    PHYSICAL_MICROBATCH_SIZE = 32
    MAX_GRAD_NORM = 0.5
    TARGET_KL = 0.03

    def __init__(
        self,
        policy: nn.Module,
        *,
        device: torch.device | str,
        optimizer: torch.optim.Optimizer | None = None,
        shuffle_seed: int = 0,
    ) -> None:
        if not isinstance(policy, nn.Module):
            raise PPOTrainingError("PPOTrainer requires a torch policy module")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise PPOTrainingError("CUDA was requested but unavailable; CPU fallback is forbidden")
        if self.device.type not in {"cpu", "cuda"}:
            raise PPOTrainingError("PPOTrainer supports only CPU or CUDA")
        self.policy = policy.to(self.device)
        if any(parameter.dtype != torch.float32 for parameter in self.policy.parameters()):
            raise PPOTrainingError("PPO policy parameters must remain FP32")
        if type(shuffle_seed) is not int:
            raise PPOTrainingError("shuffle_seed must be an integer")
        self.shuffle_seed = shuffle_seed
        self.optimizer = optimizer or torch.optim.AdamW(
            self.policy.parameters(),
            lr=3e-4,
            eps=1e-5,
            weight_decay=1e-4,
        )
        self._validate_optimizer()
        self.update_step = 0

    def restore_update_step(self, update_step: int) -> None:
        """只允许把 fresh trainer 绑定到已验证 checkpoint step。"""

        if self.update_step != 0:
            raise PPOTrainingError("trainer update step can only be restored once")
        if type(update_step) is not int or update_step <= 0:
            raise PPOTrainingError(
                "restored trainer update step must be a positive integer"
            )
        self.update_step = update_step

    def update(self, batch: RolloutBatch) -> PPOUpdateMetrics:
        if not isinstance(batch, RolloutBatch):
            raise PPOTrainingError("PPOTrainer.update requires a RolloutBatch")
        policy_hash_before = policy_state_sha256(self.policy)
        try:
            batch.claim_for_update(policy_hash_before)
        except RolloutContractError:
            raise
        parameters_before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.policy.named_parameters()
        }
        try:
            self.policy.train()
            initial_error = self._initial_ratio_error(batch)
            tolerance = 1e-5 if self.device.type == "cuda" else 1e-6
            if initial_error > tolerance:
                raise PPOTrainingError(
                    f"initial ratio mismatch: max_abs_error={initial_error:.9g}"
                )

            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.shuffle_seed + self.update_step)
            total_size = batch.size
            planned_steps = self.PPO_EPOCHS * math.ceil(
                total_size / self.EFFECTIVE_MINIBATCH_SIZE
            )
            optimizer_steps = 0
            effective_sizes: list[int] = []
            physical_sizes: list[int] = []
            policy_losses: list[float] = []
            value_losses: list[float] = []
            entropy_sum = 0.0
            entropy_count = 0
            entropy_min = math.inf
            entropy_by_count: dict[int, list[float]] = {}
            kappa_sum = 0.0
            kappa_count = 0
            kappa_max = -math.inf
            grad_pre_max = 0.0
            grad_post_max = 0.0
            approx_kl = 0.0
            early_stopped = False
            early_stop_epoch: int | None = None
            early_stop_minibatch: int | None = None
            skipped_minibatches = 0

            for epoch in range(self.PPO_EPOCHS):
                permutation = torch.randperm(total_size, generator=generator).numpy()
                minibatch_count = math.ceil(
                    total_size / self.EFFECTIVE_MINIBATCH_SIZE
                )
                for minibatch_index in range(minibatch_count):
                    start = minibatch_index * self.EFFECTIVE_MINIBATCH_SIZE
                    stop = min(start + self.EFFECTIVE_MINIBATCH_SIZE, total_size)
                    effective_indices = np.asarray(
                        permutation[start:stop], dtype=np.int64
                    )
                    effective_size = int(effective_indices.size)
                    self.optimizer.zero_grad(set_to_none=True)
                    kl_sum = 0.0
                    local_policy_sum = 0.0
                    local_value_sum = 0.0
                    local_entropy_values: list[tuple[int, float]] = []
                    local_kappas: list[float] = []

                    for micro_start, micro_stop in physical_microbatch_slices(
                        effective_size
                    ):
                        indices = effective_indices[micro_start:micro_stop]
                        physical_size = int(indices.size)
                        physical_sizes.append(physical_size)
                        output, evaluation, policy_batch = self._evaluate(
                            batch,
                            indices,
                        )
                        old_total = self._batch_tensor(
                            batch.old_log_prob_total,
                            indices,
                            dtype=torch.float32,
                        )
                        terms = compute_ppo_loss_terms(
                            new_log_prob_total=evaluation.log_prob_total,
                            old_log_prob_total=old_total,
                            normalized_advantage=self._batch_tensor(
                                batch.normalized_advantages,
                                indices,
                                dtype=torch.float32,
                            ),
                            new_value=output.value,
                            old_value=self._batch_tensor(
                                batch.old_values,
                                indices,
                                dtype=torch.float32,
                            ),
                            returns=self._batch_tensor(
                                batch.returns,
                                indices,
                                dtype=torch.float32,
                            ),
                            frontier_entropy=evaluation.frontier_entropy,
                        )
                        weight = physical_size / effective_size
                        (terms.total_loss * weight).backward()
                        kl_sum += float(
                            (old_total - evaluation.log_prob_total)
                            .detach()
                            .sum(dtype=torch.float32)
                            .cpu()
                        )
                        local_policy_sum += float(terms.policy_loss.detach().cpu()) * physical_size
                        local_value_sum += float(terms.value_loss.detach().cpu()) * physical_size
                        valid_counts = (
                            policy_batch.candidate_mask.sum(dim=1).detach().cpu().tolist()
                        )
                        entropies = evaluation.frontier_entropy.detach().cpu().tolist()
                        local_entropy_values.extend(
                            (int(count), float(entropy))
                            for count, entropy in zip(valid_counts, entropies, strict=True)
                        )
                        selected = self._batch_tensor(
                            batch.selected_frontier_indices,
                            indices,
                            dtype=torch.int64,
                        )
                        selected_kappa = output.theta_kappa.gather(
                            1, selected.unsqueeze(1)
                        ).squeeze(1)
                        local_kappas.extend(
                            float(value) for value in selected_kappa.detach().cpu().tolist()
                        )

                    approx_kl = kl_sum / effective_size
                    if not math.isfinite(approx_kl):
                        raise PPOTrainingError("approx_kl is non-finite")
                    linear_minibatch = epoch * minibatch_count + minibatch_index
                    if approx_kl > self.TARGET_KL:
                        self.optimizer.zero_grad(set_to_none=True)
                        early_stopped = True
                        early_stop_epoch = epoch
                        early_stop_minibatch = minibatch_index
                        skipped_minibatches = planned_steps - linear_minibatch - 1
                        break

                    pre_clip = _gradient_norm(self.policy.parameters())
                    if not math.isfinite(pre_clip):
                        raise PPOTrainingError("pre-clip gradient norm is non-finite")
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.MAX_GRAD_NORM,
                        error_if_nonfinite=True,
                    )
                    post_clip = _gradient_norm(self.policy.parameters())
                    if not math.isfinite(post_clip) or post_clip > 0.500001:
                        raise PPOTrainingError("post-clip gradient norm exceeds 0.500001")
                    self.optimizer.step()
                    optimizer_steps += 1
                    effective_sizes.append(effective_size)
                    policy_losses.append(local_policy_sum / effective_size)
                    value_losses.append(local_value_sum / effective_size)
                    grad_pre_max = max(grad_pre_max, pre_clip)
                    grad_post_max = max(grad_post_max, post_clip)
                    for count, entropy in local_entropy_values:
                        entropy_sum += entropy
                        entropy_count += 1
                        entropy_min = min(entropy_min, entropy)
                        aggregate = entropy_by_count.setdefault(count, [0.0, 0.0])
                        aggregate[0] += entropy
                        aggregate[1] += 1.0
                    for value in local_kappas:
                        kappa_sum += value
                        kappa_count += 1
                        kappa_max = max(kappa_max, value)
                if early_stopped:
                    break

            parameter_change_l2 = _parameter_change_l2(
                self.policy,
                parameters_before,
            )
            if optimizer_steps == 0 or parameter_change_l2 <= 0.0:
                raise PPOTrainingError("PPO update changed no trainable parameter")
            if not all(
                math.isfinite(value)
                for value in (
                    parameter_change_l2,
                    grad_pre_max,
                    grad_post_max,
                    entropy_sum,
                    kappa_sum,
                )
            ):
                raise PPOTrainingError("PPO update metrics are non-finite")
            policy_hash_after = policy_state_sha256(self.policy)
            if policy_hash_after == policy_hash_before:
                raise PPOTrainingError("PPO policy hash did not change")
            self.update_step += 1
            metrics = PPOUpdateMetrics(
                update_step=self.update_step,
                initial_ratio_max_abs_error=initial_error,
                policy_loss=sum(policy_losses) / len(policy_losses),
                value_loss=sum(value_losses) / len(value_losses),
                frontier_entropy_mean=entropy_sum / entropy_count,
                frontier_entropy_min=entropy_min,
                frontier_entropy_by_valid_candidate_count=MappingProxyType(
                    {
                        count: values[0] / values[1]
                        for count, values in sorted(entropy_by_count.items())
                    }
                ),
                theta_kappa_mean=kappa_sum / kappa_count,
                theta_kappa_max=kappa_max,
                approx_kl=approx_kl,
                optimizer_steps=optimizer_steps,
                planned_optimizer_steps=planned_steps,
                effective_minibatch_sizes=tuple(effective_sizes),
                physical_microbatch_sizes=tuple(physical_sizes),
                early_stopped=early_stopped,
                early_stop_epoch=early_stop_epoch,
                early_stop_minibatch=early_stop_minibatch,
                skipped_minibatch_count=skipped_minibatches,
                grad_pre_clip_norm_max=grad_pre_max,
                grad_post_clip_norm_max=grad_post_max,
                parameter_change_l2=parameter_change_l2,
                policy_state_sha256_before=policy_hash_before,
                policy_state_sha256_after=policy_hash_after,
            )
            batch.mark_consumed()
            return metrics
        except BaseException as exc:
            if batch.claimed:
                batch.invalidate()
            if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
                raise PPOTrainingError("CUDA OOM during PPO update") from exc
            raise

    def _initial_ratio_error(self, batch: RolloutBatch) -> float:
        max_error = 0.0
        with torch.no_grad():
            for start, stop in physical_microbatch_slices(batch.size):
                indices = np.arange(start, stop, dtype=np.int64)
                _, evaluation, _ = self._evaluate(batch, indices)
                old_frontier = self._batch_tensor(
                    batch.old_log_prob_frontier,
                    indices,
                    dtype=torch.float32,
                )
                old_theta = self._batch_tensor(
                    batch.old_log_prob_theta,
                    indices,
                    dtype=torch.float32,
                )
                old_total = self._batch_tensor(
                    batch.old_log_prob_total,
                    indices,
                    dtype=torch.float32,
                )
                differences = (
                    (evaluation.log_prob_frontier - old_frontier).abs(),
                    (evaluation.log_prob_theta - old_theta).abs(),
                    (evaluation.log_prob_total - old_total).abs(),
                    (torch.exp(evaluation.log_prob_total - old_total) - 1.0).abs(),
                )
                max_error = max(
                    max_error,
                    *(float(value.max().cpu()) for value in differences),
                )
        if not math.isfinite(max_error):
            raise PPOTrainingError("initial ratio is non-finite")
        return max_error

    def _evaluate(
        self,
        batch: RolloutBatch,
        indices: np.ndarray,
    ) -> tuple[PolicyForwardOutput, object, object]:
        observations = batch.policy_observations(indices)
        policy_batch = batch_policy_observations(observations, device=self.device)
        output = self.policy(policy_batch)
        if not isinstance(output, PolicyForwardOutput):
            raise PPOTrainingError("policy must return PolicyForwardOutput")
        selected = self._batch_tensor(
            batch.selected_frontier_indices,
            indices,
            dtype=torch.int64,
        )
        theta = self._batch_tensor(
            batch.selected_thetas,
            indices,
            dtype=torch.float32,
        )
        evaluation = recompute_action_log_probs(
            output,
            policy_batch.candidate_mask,
            selected,
            theta,
        )
        return output, evaluation, policy_batch

    def _batch_tensor(
        self,
        matrix: np.ndarray,
        indices: np.ndarray,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        flat = matrix.reshape(-1)
        selected = np.asarray(flat[indices]).copy()
        return torch.as_tensor(selected, dtype=dtype, device=self.device)

    def _validate_optimizer(self) -> None:
        if not isinstance(self.optimizer, torch.optim.AdamW):
            raise PPOTrainingError("Stage 4 optimizer must be AdamW")
        for group in self.optimizer.param_groups:
            if (
                group.get("lr") != 3e-4
                or group.get("eps") != 1e-5
                or group.get("weight_decay") != 1e-4
            ):
                raise PPOTrainingError("Stage 4 AdamW hyperparameters drifted")


def _gradient_norm(parameters) -> float:
    total = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found = True
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all()):
            return math.inf
        total += float(gradient.double().square().sum().cpu())
    return math.sqrt(total) if found else 0.0


def _parameter_change_l2(
    policy: nn.Module,
    before: Mapping[str, torch.Tensor],
) -> float:
    total = 0.0
    for name, parameter in policy.named_parameters():
        if name not in before:
            raise PPOTrainingError("policy parameter set changed during update")
        difference = parameter.detach().cpu().double() - before[name].double()
        total += float(difference.square().sum())
    if set(before) != {name for name, _ in policy.named_parameters()}:
        raise PPOTrainingError("policy parameter set changed during update")
    return math.sqrt(total)


__all__ = [
    "PPOLossTerms",
    "PPOTrainer",
    "PPOTrainingError",
    "PPOUpdateMetrics",
    "compute_ppo_loss_terms",
    "physical_microbatch_slices",
    "policy_state_sha256",
]
