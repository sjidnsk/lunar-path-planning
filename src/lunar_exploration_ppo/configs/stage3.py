"""Frozen Stage 3 cross-attention policy and CUDA audit configuration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class Stage3ObservationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    prior_channels: int
    coverage_summary_channels: int
    local_crop_channels: int
    frontier_feature_dim: int
    pose_feature_dim: int
    candidate_mask_dtype: str
    forbidden_policy_inputs: tuple[str, ...]


class Stage3ArchitectureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    network_architecture_version: str
    network_memory_mode: str
    global_encoder_input_channels: int
    map_pool_shape: tuple[int, int]
    global_token_count: int
    local_token_count: int
    pose_token_count: int
    context_token_count: int
    token_dim: int
    cross_attention_layers: int
    attention_heads: int
    ffn_hidden_dim: int
    action_hidden_dim: int
    dropout: float
    normalization: str
    cross_attention_direction: str
    candidate_self_attention: bool


class Stage3DistributionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_space_type: str
    factorization: str
    frontier_distribution: str
    theta_distribution: str
    theta_parameterization: str
    invalid_logit_value: float
    theta_min_inclusive: float
    theta_max_exclusive: float
    kappa_epsilon: float
    kappa_min: float
    kappa_max: float
    initial_kappa: float
    theta_sin_bias: float
    theta_cos_bias: float
    compute_dtype: str
    amp_enabled: bool
    cpu_logprob_tolerance: float
    cuda_logprob_tolerance: float


class Stage3GpuConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    device: str
    memory_warning_gib: float
    memory_hard_stop_gib: float
    smoke_batch1_forward_p95_ms: float
    standard_batch1_forward_p95_ms: float
    microbatch_candidates: tuple[int, ...]
    frozen_microbatch_size: int
    timing_warmup_iterations: int
    timing_measure_iterations: int
    standard_prior_shape: tuple[int, int, int]
    standard_coverage_shape: tuple[int, int, int]
    standard_local_shape: tuple[int, int, int]
    standard_frontier_top_m: int
    representative_backward: bool


class Stage3Config(BaseModel):
    """Exact Stage 3 implementation and machine-evidence contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["ppo_highres_frontier_stage3_config/v1"]
    goal_id: Literal["ppo-highres-frontier-map-exploration"]
    stage_id: Literal["ppo_highres_frontier_stage3_cross_attention_policy/v1"]
    authorized_next_stage: Literal[
        "ppo_highres_frontier_stage4_rollout_ppo_update/v1"
    ]
    stage2_approval_commit: Literal["898911559ccc9ae8ef701b69a58a93b1d8a4d8d8"]
    observation: Stage3ObservationConfig
    architecture: Stage3ArchitectureConfig
    distribution: Stage3DistributionConfig
    gpu: Stage3GpuConfig
    output_root: Literal["D:/xunce/out/ppo_frontier"]
    run_id: str | None = None

    @model_validator(mode="after")
    def validate_frozen_contracts(self) -> "Stage3Config":
        expected_observation = {
            "schema_version": "policy_observation/v1",
            "prior_channels": 7,
            "coverage_summary_channels": 8,
            "local_crop_channels": 8,
            "frontier_feature_dim": 22,
            "pose_feature_dim": 6,
            "candidate_mask_dtype": "bool",
            "forbidden_policy_inputs": (
                "truth",
                "coverable_mask",
                "history",
                "previous_action",
            ),
        }
        expected_architecture = {
            "network_architecture_version": "cross_attention_frontier_policy/v1",
            "network_memory_mode": "stateless_observation_only/v1",
            "global_encoder_input_channels": 15,
            "map_pool_shape": (16, 16),
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
            "normalization": "groupnorm_cnn_layernorm_transformer_mlp/v1",
            "cross_attention_direction": (
                "frontier_queries_to_context_keys_values/v1"
            ),
            "candidate_self_attention": False,
        }
        expected_distribution = {
            "action_space_type": "frontier_index_continuous_theta/v1",
            "factorization": (
                "masked_categorical_times_selected_frontier_von_mises/v1"
            ),
            "frontier_distribution": "masked_categorical/v1",
            "theta_distribution": "von_mises/v1",
            "theta_parameterization": "clamped_softplus_kappa/v1",
            "invalid_logit_value": -1.0e9,
            "theta_min_inclusive": -math.pi,
            "theta_max_exclusive": math.pi,
            "kappa_epsilon": 1e-3,
            "kappa_min": 1e-3,
            "kappa_max": 20.0,
            "initial_kappa": 0.1,
            "theta_sin_bias": 0.0,
            "theta_cos_bias": 1.0,
            "compute_dtype": "float32",
            "amp_enabled": False,
            "cpu_logprob_tolerance": 1e-6,
            "cuda_logprob_tolerance": 1e-5,
        }
        expected_gpu = {
            "device": "cuda",
            "memory_warning_gib": 9.0,
            "memory_hard_stop_gib": 10.1,
            "smoke_batch1_forward_p95_ms": 50.0,
            "standard_batch1_forward_p95_ms": 100.0,
            "microbatch_candidates": (4, 8, 16, 32),
            "frozen_microbatch_size": 32,
            "timing_warmup_iterations": 10,
            "timing_measure_iterations": 30,
            "standard_prior_shape": (7, 32, 32),
            "standard_coverage_shape": (8, 32, 32),
            "standard_local_shape": (8, 96, 96),
            "standard_frontier_top_m": 1024,
            "representative_backward": True,
        }
        actuals = (
            ("observation", self.observation.model_dump()),
            ("architecture", self.architecture.model_dump()),
            ("distribution", self.distribution.model_dump()),
            ("gpu", self.gpu.model_dump()),
        )
        expected = {
            "observation": expected_observation,
            "architecture": expected_architecture,
            "distribution": expected_distribution,
            "gpu": expected_gpu,
        }
        for name, actual in actuals:
            if actual != expected[name]:
                raise ValueError(f"Stage 3 {name} contract drifted")
        return self


def load_stage3_config(path: str | Path) -> Stage3Config:
    return Stage3Config.model_validate_json(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "Stage3ArchitectureConfig",
    "Stage3Config",
    "Stage3DistributionConfig",
    "Stage3GpuConfig",
    "Stage3ObservationConfig",
    "load_stage3_config",
]
