"""Fail-closed Stage 3 CUDA latency and microbatch gate primitives."""

from __future__ import annotations

import gc
import io
import math
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np
import torch

from lunar_exploration_ppo.configs.stage3 import Stage3Config
from lunar_exploration_ppo.policy.cross_attention import (
    CrossAttentionFrontierPolicy,
    batch_policy_observations,
    masked_frontier_probabilities,
    recompute_action_log_probs,
    sample_action,
)
from lunar_exploration_ppo.policy.observation import PolicyObservation


class Stage3GpuGateError(RuntimeError):
    """CUDA audit evidence is incomplete or violates a Stage 3 hard gate."""


@dataclass(frozen=True, slots=True)
class MicrobatchTrial:
    batch_size: int
    status: Literal["success", "oom", "error"]
    peak_memory_gib: float | None
    error_kind: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MicrobatchGateResult:
    selected_microbatch_size: int
    warning: bool
    hard_gate_passed: bool
    attempted_sizes: tuple[int, ...]
    trials: tuple[MicrobatchTrial, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_microbatch_size": self.selected_microbatch_size,
            "warning": self.warning,
            "hard_gate_passed": self.hard_gate_passed,
            "attempted_sizes": list(self.attempted_sizes),
            "trials": [trial.to_dict() for trial in self.trials],
        }


@dataclass(frozen=True, slots=True)
class LatencyGateResult:
    smoke_p95_ms: float
    standard_p95_ms: float
    smoke_limit_ms: float
    standard_limit_ms: float
    smoke_sample_count: int
    standard_sample_count: int
    gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Stage3CudaAuditBundle:
    network_shape_audit: dict[str, object]
    logprob_recompute_audit: dict[str, object]
    mask_handling_audit: dict[str, object]
    cuda_latency_audit: dict[str, object]
    cuda_microbatch_audit: dict[str, object]
    sample_policy_outputs_npz: bytes


def evaluate_microbatch_gate(
    trials: Sequence[MicrobatchTrial],
    *,
    candidates: Sequence[int],
    warning_gib: float,
    hard_stop_gib: float,
) -> MicrobatchGateResult:
    """Validate a complete ascending sweep and select its largest safe success."""

    frozen_trials = tuple(trials)
    frozen_candidates = tuple(candidates)
    if (
        not frozen_candidates
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in frozen_candidates)
        or tuple(sorted(set(frozen_candidates))) != frozen_candidates
    ):
        raise Stage3GpuGateError("microbatch candidates must be unique ascending positive integers")
    if tuple(trial.batch_size for trial in frozen_trials) != frozen_candidates:
        raise Stage3GpuGateError("microbatch sweep must record every candidate in ascending order")
    if (
        not math.isfinite(warning_gib)
        or not math.isfinite(hard_stop_gib)
        or warning_gib <= 0.0
        or hard_stop_gib <= warning_gib
    ):
        raise Stage3GpuGateError("memory thresholds are invalid")

    safe_successes: list[MicrobatchTrial] = []
    warned = False
    for trial in frozen_trials:
        if trial.status == "success":
            peak = trial.peak_memory_gib
            if (
                peak is None
                or not math.isfinite(peak)
                or peak < 0.0
                or trial.error_kind is not None
            ):
                raise Stage3GpuGateError("successful microbatch trial has invalid evidence")
            if peak > hard_stop_gib:
                raise Stage3GpuGateError("microbatch peak memory crossed the hard stop")
            warned = warned or peak > warning_gib
            safe_successes.append(trial)
        elif trial.status == "oom":
            peak = trial.peak_memory_gib
            if (
                (peak is not None and (not math.isfinite(peak) or peak < 0.0))
                or trial.error_kind != "cuda_out_of_memory"
            ):
                raise Stage3GpuGateError("OOM microbatch trial has invalid classification")
            if peak is not None:
                if peak > hard_stop_gib:
                    raise Stage3GpuGateError("microbatch peak memory crossed the hard stop")
                warned = warned or peak > warning_gib
        elif trial.status == "error":
            raise Stage3GpuGateError("non-OOM microbatch failure is a hard gate failure")
        else:
            raise Stage3GpuGateError("microbatch trial status is invalid")
    if not safe_successes:
        raise Stage3GpuGateError("microbatch sweep has no successful safe size")
    selected = max(trial.batch_size for trial in safe_successes)
    return MicrobatchGateResult(
        selected_microbatch_size=selected,
        warning=warned,
        hard_gate_passed=True,
        attempted_sizes=frozen_candidates,
        trials=frozen_trials,
    )


def evaluate_latency_gate(
    *,
    smoke_samples_ms: Sequence[float],
    standard_samples_ms: Sequence[float],
    smoke_limit_ms: float,
    standard_limit_ms: float,
) -> LatencyGateResult:
    """Calculate linear p95 values and enforce both batch-one hard limits."""

    smoke = _validated_latency_samples(smoke_samples_ms, "Smoke")
    standard = _validated_latency_samples(standard_samples_ms, "Standard")
    if (
        not math.isfinite(smoke_limit_ms)
        or not math.isfinite(standard_limit_ms)
        or smoke_limit_ms <= 0.0
        or standard_limit_ms <= 0.0
    ):
        raise Stage3GpuGateError("latency thresholds are invalid")
    smoke_p95 = float(np.percentile(smoke, 95.0))
    standard_p95 = float(np.percentile(standard, 95.0))
    if smoke_p95 > smoke_limit_ms or standard_p95 > standard_limit_ms:
        raise Stage3GpuGateError("Stage 3 batch-one latency hard gate failed")
    return LatencyGateResult(
        smoke_p95_ms=smoke_p95,
        standard_p95_ms=standard_p95,
        smoke_limit_ms=float(smoke_limit_ms),
        standard_limit_ms=float(standard_limit_ms),
        smoke_sample_count=len(smoke),
        standard_sample_count=len(standard),
        gate_passed=True,
    )


def _validated_latency_samples(
    values: Sequence[float],
    label: str,
) -> tuple[float, ...]:
    samples = tuple(float(value) for value in values)
    if not samples or any(not math.isfinite(value) or value < 0.0 for value in samples):
        raise Stage3GpuGateError(f"{label} latency samples are incomplete or non-finite")
    return samples


def run_stage3_cuda_audit(config: Stage3Config) -> Stage3CudaAuditBundle:
    """Run the bounded real-CUDA Stage 3 policy and resource audit."""

    if not isinstance(config, Stage3Config):
        raise Stage3GpuGateError("CUDA audit requires a validated Stage3Config")
    if config.gpu.device != "cuda" or config.distribution.amp_enabled:
        raise Stage3GpuGateError("Stage 3 CUDA audit contract drift")
    if not torch.cuda.is_available():
        raise Stage3GpuGateError("CUDA is unavailable; CPU fallback is forbidden")
    device = torch.device("cuda")
    if torch.is_autocast_enabled(device.type):
        raise Stage3GpuGateError("AMP/autocast is forbidden during the CUDA audit")

    torch.manual_seed(20260714)
    torch.cuda.manual_seed_all(20260714)
    model = CrossAttentionFrontierPolicy().to(device).eval()
    network_shape = _network_shape_audit(model, config)
    logprob, mask, sample_npz = _numeric_cuda_audit(model, config, device)

    smoke_observation = _audit_observation(
        candidate_count=512,
        valid_count=512,
        local_shape=(64, 64),
        seed=3101,
    )
    standard_observation = _audit_observation(
        candidate_count=config.gpu.standard_frontier_top_m,
        valid_count=config.gpu.standard_frontier_top_m,
        local_shape=(config.gpu.standard_local_shape[1], config.gpu.standard_local_shape[2]),
        seed=3102,
    )
    smoke_samples, smoke_peak = _measure_forward_latency(
        model=model,
        observation=smoke_observation,
        device=device,
        warmups=config.gpu.timing_warmup_iterations,
        iterations=config.gpu.timing_measure_iterations,
    )
    standard_samples, standard_peak = _measure_forward_latency(
        model=model,
        observation=standard_observation,
        device=device,
        warmups=config.gpu.timing_warmup_iterations,
        iterations=config.gpu.timing_measure_iterations,
    )
    latency_gate = evaluate_latency_gate(
        smoke_samples_ms=smoke_samples,
        standard_samples_ms=standard_samples,
        smoke_limit_ms=config.gpu.smoke_batch1_forward_p95_ms,
        standard_limit_ms=config.gpu.standard_batch1_forward_p95_ms,
    )
    for label, peak in (("Smoke", smoke_peak), ("Standard", standard_peak)):
        if peak > config.gpu.memory_hard_stop_gib:
            raise Stage3GpuGateError(f"{label} latency audit peak crossed the memory hard stop")
    latency_payload = {
        "schema_version": "ppo_highres_frontier_stage3_cuda_latency_audit/v1",
        "device": torch.cuda.get_device_name(device),
        "synchronized_cuda_events": True,
        "batch_size": 1,
        "warmup_iterations": config.gpu.timing_warmup_iterations,
        "measure_iterations": config.gpu.timing_measure_iterations,
        "smoke_samples_ms": list(smoke_samples),
        "standard_samples_ms": list(standard_samples),
        "smoke_peak_memory_gib": smoke_peak,
        "standard_peak_memory_gib": standard_peak,
        **latency_gate.to_dict(),
    }

    del model
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    trials = tuple(
        _run_microbatch_trial(
            batch_size=batch_size,
            observation=standard_observation,
            device=device,
        )
        for batch_size in config.gpu.microbatch_candidates
    )
    microbatch_gate = evaluate_microbatch_gate(
        trials,
        candidates=config.gpu.microbatch_candidates,
        warning_gib=config.gpu.memory_warning_gib,
        hard_stop_gib=config.gpu.memory_hard_stop_gib,
    )
    if microbatch_gate.selected_microbatch_size != config.gpu.frozen_microbatch_size:
        raise Stage3GpuGateError(
            "measured safe microbatch does not equal the frozen Stage 3 value"
        )
    microbatch_payload = {
        "schema_version": "ppo_highres_frontier_stage3_cuda_microbatch_audit/v1",
        "device": torch.cuda.get_device_name(device),
        "compute_dtype": "float32",
        "amp_enabled": False,
        "shape_profile": "standard_worst_case_top_m/v1",
        "forward": True,
        "representative_backward": True,
        "memory_warning_gib": config.gpu.memory_warning_gib,
        "memory_hard_stop_gib": config.gpu.memory_hard_stop_gib,
        "frozen_microbatch_size": config.gpu.frozen_microbatch_size,
        **microbatch_gate.to_dict(),
    }
    if not finite_audit_tree(
        {
            "network": network_shape,
            "logprob": logprob,
            "mask": mask,
            "latency": latency_payload,
            "microbatch": microbatch_payload,
        }
    ):
        raise Stage3GpuGateError("CUDA audit produced non-finite evidence")
    return Stage3CudaAuditBundle(
        network_shape_audit=network_shape,
        logprob_recompute_audit=logprob,
        mask_handling_audit=mask,
        cuda_latency_audit=latency_payload,
        cuda_microbatch_audit=microbatch_payload,
        sample_policy_outputs_npz=sample_npz,
    )


def _network_shape_audit(
    model: CrossAttentionFrontierPolicy,
    config: Stage3Config,
) -> dict[str, object]:
    metadata = model.architecture_metadata()
    modules = tuple(model.modules())
    forbidden = {
        "batch_norm": any(
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            for module in modules
        ),
        "recurrent": any(isinstance(module, torch.nn.RNNBase) for module in modules),
        "dropout_module": any(isinstance(module, torch.nn.Dropout) for module in modules),
        "candidate_self_attention_attribute": hasattr(model, "candidate_self_attention"),
    }
    expected = config.architecture.model_dump(mode="json")
    comparable = {
        key: metadata[key]
        for key in (
            "network_architecture_version",
            "network_memory_mode",
            "global_encoder_input_channels",
            "map_pool_shape",
            "global_token_count",
            "local_token_count",
            "pose_token_count",
            "context_token_count",
            "token_dim",
            "cross_attention_layers",
            "attention_heads",
            "ffn_hidden_dim",
            "action_hidden_dim",
            "dropout",
            "cross_attention_direction",
        )
    }
    if any(comparable[key] != expected[key] for key in comparable) or any(forbidden.values()):
        raise Stage3GpuGateError("network architecture audit failed")
    if not any(isinstance(module, torch.nn.GroupNorm) for module in modules) or not any(
        isinstance(module, torch.nn.LayerNorm) for module in modules
    ):
        raise Stage3GpuGateError("network normalization contract failed")
    return {
        "schema_version": "ppo_highres_frontier_stage3_network_shape_audit/v1",
        "passed": True,
        "metadata": metadata,
        "forbidden_modules_present": forbidden,
        "groupnorm_present": True,
        "layernorm_present": True,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "all_parameters_fp32": all(
            parameter.dtype == torch.float32 for parameter in model.parameters()
        ),
    }


def _numeric_cuda_audit(
    model: CrossAttentionFrontierPolicy,
    config: Stage3Config,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object], bytes]:
    observation = _audit_observation(
        candidate_count=64,
        valid_count=47,
        local_shape=(64, 64),
        seed=3201,
    )
    cpu_model = CrossAttentionFrontierPolicy().eval()
    cpu_model.load_state_dict({key: value.detach().cpu() for key, value in model.state_dict().items()})
    cpu_batch = batch_policy_observations((observation,), device="cpu")
    with torch.no_grad():
        cpu_output = cpu_model(cpu_batch)
        cpu_action = sample_action(cpu_output, cpu_batch.candidate_mask, deterministic=True)
        cpu_recomputed = recompute_action_log_probs(
            cpu_output,
            cpu_batch.candidate_mask,
            cpu_action.selected_frontier_index,
            cpu_action.selected_theta,
        )
    cpu_errors = _logprob_errors(cpu_action, cpu_recomputed)
    if max(cpu_errors.values()) > config.distribution.cpu_logprob_tolerance:
        raise Stage3GpuGateError("CPU logprob recomputation tolerance failed")

    cuda_batch = batch_policy_observations((observation,), device=device)
    cuda_batch = type(cuda_batch)(
        prior_channels=cuda_batch.prior_channels,
        coverage_summary=cuda_batch.coverage_summary,
        local_crop=cuda_batch.local_crop,
        frontier_features=cuda_batch.frontier_features.detach().clone().requires_grad_(True),
        pose_features=cuda_batch.pose_features,
        candidate_mask=cuda_batch.candidate_mask,
    )
    model.zero_grad(set_to_none=True)
    cuda_output = model(cuda_batch)
    probabilities = masked_frontier_probabilities(
        cuda_output.frontier_logits,
        cuda_batch.candidate_mask,
    )
    cuda_action = sample_action(cuda_output, cuda_batch.candidate_mask, deterministic=True)
    cuda_repeat = sample_action(cuda_output, cuda_batch.candidate_mask, deterministic=True)
    cuda_recomputed = recompute_action_log_probs(
        cuda_output,
        cuda_batch.candidate_mask,
        cuda_action.selected_frontier_index,
        cuda_action.selected_theta,
    )
    cuda_errors = _logprob_errors(cuda_action, cuda_recomputed)
    if max(cuda_errors.values()) > config.distribution.cuda_logprob_tolerance:
        raise Stage3GpuGateError("CUDA logprob recomputation tolerance failed")
    loss = (
        -cuda_action.log_prob_total.mean()
        + cuda_output.value.square().mean()
        + cuda_output.theta_kappa[cuda_batch.candidate_mask].mean()
    )
    loss.backward()
    torch.cuda.synchronize(device)
    invalid_probability_zero = torch.equal(
        probabilities[~cuda_batch.candidate_mask],
        torch.zeros_like(probabilities[~cuda_batch.candidate_mask]),
    )
    invalid_gradient_zero = (
        torch.count_nonzero(
            cuda_batch.frontier_features.grad[~cuda_batch.candidate_mask]
        ).item()
        == 0
    )
    deterministic_exact = all(
        torch.equal(getattr(cuda_action, name), getattr(cuda_repeat, name))
        for name in (
            "selected_frontier_index",
            "selected_theta",
            "log_prob_frontier",
            "log_prob_theta",
            "log_prob_total",
            "frontier_entropy",
            "value",
        )
    )
    selected_valid = bool(
        cuda_batch.candidate_mask.gather(
            1,
            cuda_action.selected_frontier_index.unsqueeze(1),
        ).all()
    )
    theta_in_range = bool(
        (cuda_action.selected_theta >= -torch.pi).all()
        and (cuda_action.selected_theta < torch.pi).all()
    )
    kappa_in_bounds = bool(
        (cuda_output.theta_kappa >= config.distribution.kappa_min).all()
        and (cuda_output.theta_kappa <= config.distribution.kappa_max).all()
    )
    all_finite = all(
        bool(torch.isfinite(tensor).all())
        for tensor in (
            cuda_output.frontier_logits,
            cuda_output.theta_mu_sin_raw,
            cuda_output.theta_mu_cos_raw,
            cuda_output.theta_kappa_raw,
            cuda_output.theta_mu,
            cuda_output.theta_kappa,
            cuda_output.value,
            cuda_action.selected_theta,
            cuda_action.log_prob_frontier,
            cuda_action.log_prob_theta,
            cuda_action.log_prob_total,
            cuda_batch.frontier_features.grad,
        )
    )
    if not all(
        (
            invalid_probability_zero,
            invalid_gradient_zero,
            deterministic_exact,
            selected_valid,
            theta_in_range,
            kappa_in_bounds,
            all_finite,
        )
    ):
        raise Stage3GpuGateError("CUDA mask or finite-output audit failed")
    logprob = {
        "schema_version": "ppo_highres_frontier_stage3_logprob_recompute_audit/v1",
        "passed": True,
        "cpu_tolerance": config.distribution.cpu_logprob_tolerance,
        "cuda_tolerance": config.distribution.cuda_logprob_tolerance,
        "cpu_max_abs_errors": cpu_errors,
        "cuda_max_abs_errors": cuda_errors,
        "factorization_exact": torch.equal(
            cuda_action.log_prob_total,
            cuda_action.log_prob_frontier + cuda_action.log_prob_theta,
        ),
    }
    mask = {
        "schema_version": "ppo_highres_frontier_stage3_mask_handling_audit/v1",
        "passed": True,
        "candidate_count": 64,
        "valid_candidate_count": 47,
        "invalid_probability_exact_zero": invalid_probability_zero,
        "invalid_frontier_feature_gradient_exact_zero": invalid_gradient_zero,
        "deterministic_action_bit_exact": deterministic_exact,
        "selected_candidate_valid": selected_valid,
        "selected_theta_in_half_open_interval": theta_in_range,
        "theta_kappa_in_bounds": kappa_in_bounds,
        "all_outputs_and_gradients_finite": all_finite,
    }
    archive = io.BytesIO()
    np.savez_compressed(
        archive,
        schema_version=np.asarray("ppo_highres_frontier_stage3_sample_policy_outputs/v1"),
        candidate_mask=cuda_batch.candidate_mask.detach().cpu().numpy(),
        frontier_logits=cuda_output.frontier_logits.detach().cpu().numpy(),
        theta_mu=cuda_output.theta_mu.detach().cpu().numpy(),
        theta_kappa=cuda_output.theta_kappa.detach().cpu().numpy(),
        value=cuda_output.value.detach().cpu().numpy(),
        selected_frontier_index=cuda_action.selected_frontier_index.detach().cpu().numpy(),
        selected_theta=cuda_action.selected_theta.detach().cpu().numpy(),
        log_prob_frontier=cuda_action.log_prob_frontier.detach().cpu().numpy(),
        log_prob_theta=cuda_action.log_prob_theta.detach().cpu().numpy(),
        log_prob_total=cuda_action.log_prob_total.detach().cpu().numpy(),
    )
    return logprob, mask, archive.getvalue()


def _logprob_errors(action, recomputed) -> dict[str, float]:
    return {
        name: float(
            torch.max(torch.abs(getattr(recomputed, name) - getattr(action, name)))
            .detach()
            .cpu()
            .item()
        )
        for name in ("log_prob_frontier", "log_prob_theta", "log_prob_total")
    }


def _measure_forward_latency(
    *,
    model: CrossAttentionFrontierPolicy,
    observation: PolicyObservation,
    device: torch.device,
    warmups: int,
    iterations: int,
) -> tuple[tuple[float, ...], float]:
    batch = batch_policy_observations((observation,), device=device)
    with torch.no_grad():
        for _ in range(warmups):
            model(batch)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        samples: list[float] = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(batch)
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
    peak = _peak_memory_gib(device)
    return tuple(samples), peak


def _run_microbatch_trial(
    *,
    batch_size: int,
    observation: PolicyObservation,
    device: torch.device,
) -> MicrobatchTrial:
    gc.collect()
    torch.cuda.empty_cache()
    try:
        peak = _execute_microbatch_trial(
            batch_size=batch_size,
            observation=observation,
            device=device,
        )
        trial = MicrobatchTrial(batch_size, "success", peak, None)
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        message = str(exc).lower()
        if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in message:
            peak = _peak_memory_gib(device)
            trial = MicrobatchTrial(
                batch_size,
                "oom",
                peak if math.isfinite(peak) else None,
                "cuda_out_of_memory",
            )
        else:
            trial = MicrobatchTrial(
                batch_size,
                "error",
                None,
                f"{type(exc).__name__}:{str(exc)[:160]}",
            )
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    return trial


def _execute_microbatch_trial(
    *,
    batch_size: int,
    observation: PolicyObservation,
    device: torch.device,
) -> float:
    torch.cuda.reset_peak_memory_stats(device)
    model = CrossAttentionFrontierPolicy().to(device).train()
    batch = batch_policy_observations((observation,) * batch_size, device=device)
    model.zero_grad(set_to_none=True)
    output = model(batch)
    action = sample_action(output, batch.candidate_mask, deterministic=True)
    loss = (
        -action.log_prob_total.mean()
        - 0.01 * action.frontier_entropy.mean()
        + 0.5 * output.value.square().mean()
    )
    loss.backward()
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(loss)) or any(
        not bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    ):
        raise RuntimeError("non-finite representative backward")
    return _peak_memory_gib(device)


def _peak_memory_gib(device: torch.device) -> float:
    return float(torch.cuda.max_memory_allocated(device) / (1024.0**3))


def _audit_observation(
    *,
    candidate_count: int,
    valid_count: int,
    local_shape: tuple[int, int],
    seed: int,
) -> PolicyObservation:
    rng = np.random.default_rng(seed)
    mask = np.zeros(candidate_count, dtype=bool)
    mask[:valid_count] = True
    features = rng.normal(size=(candidate_count, 22)).astype(np.float32)
    recommended = rng.normal(size=(candidate_count, 2)).astype(np.float32)
    recommended_norm = np.linalg.norm(recommended, axis=1, keepdims=True)
    features[:, 14:16] = recommended / np.maximum(recommended_norm, np.float32(1e-6))
    return PolicyObservation(
        prior_channels=rng.normal(size=(7, 32, 32)).astype(np.float32),
        coverage_summary=rng.normal(size=(8, 32, 32)).astype(np.float32),
        local_crop=rng.normal(size=(8, *local_shape)).astype(np.float32),
        frontier_features=features,
        pose_features=rng.normal(size=(6,)).astype(np.float32),
        candidate_mask=mask,
    )


def finite_audit_tree(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_audit_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_audit_tree(item) for item in value)
    return False


__all__ = [
    "LatencyGateResult",
    "MicrobatchGateResult",
    "MicrobatchTrial",
    "Stage3CudaAuditBundle",
    "Stage3GpuGateError",
    "evaluate_latency_gate",
    "evaluate_microbatch_gate",
    "run_stage3_cuda_audit",
]
