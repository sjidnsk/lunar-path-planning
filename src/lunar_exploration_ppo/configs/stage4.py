"""Frozen Stage 4 rollout、PPO、checkpoint 与 machine acceptance 配置。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class Stage3AuthorityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    commit_sha256: str
    commit_parent: str
    commit_tree: str
    commit_subject: str
    gate_path: str
    gate_sha256: str
    approval_sha256: str
    review_sha256: str
    manifest_sha256: str
    config_sha256: str
    source_set_sha256: str
    environment_sha256: str
    stage2_gate_sha256: str
    reviewed_path_set_sha256: str
    authorized_stage: str


class Stage4RolloutConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    layout: str
    env_count: int
    trainable_steps_per_env: int
    batch_size: int
    multiprocessing_start_method: str
    diagnostic_empty_reset_counts_toward_quota: bool
    fake_logprob_allowed: bool
    all_done_are_terminal: bool
    gamma: float
    gae_lambda: float
    advantage_normalization: str


class Stage4PPOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ppo_epochs: int
    effective_minibatch_size: int
    physical_microbatch_size: int
    shuffle: bool
    optimizer: str
    learning_rate: float
    adam_eps: float
    weight_decay: float
    clip_eps: float
    value_clip_eps: float
    value_loss_coef: float
    frontier_entropy_coef: float
    theta_entropy_enabled: bool
    max_grad_norm: float
    target_kl: float
    compute_dtype: str
    amp_enabled: bool


class Stage4CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    save_every_update: bool
    periodic_every_updates: int
    periodic_keep_count: int
    latest_keep_count: int
    complete_marker_required: bool
    sha256_manifest_required: bool
    overwrite_complete_update: bool
    restore_last_complete_only: bool


class Stage4AcceptanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    smoke_updates: int
    trainable_transitions_per_update: int
    tiny_overfit_contexts: int
    tiny_overfit_seeds: tuple[int, ...]
    tiny_overfit_max_updates: int
    tiny_frontier_probability_min: float
    tiny_theta_error_rad_max: float
    tiny_value_mse_reduction_min: float


class Stage4Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["ppo_highres_frontier_stage4_config/v1"]
    goal_id: Literal["ppo-highres-frontier-map-exploration"]
    stage_id: Literal["ppo_highres_frontier_stage4_rollout_ppo_update/v1"]
    authorized_next_stage: Literal[
        "ppo_highres_frontier_stage5_fair_baseline_evaluator/v1"
    ]
    stage3_authority: Stage3AuthorityConfig
    stage1_config_path: Literal["configs/ppo_highres_frontier_smoke_v1.json"]
    stage1_config_sha256: Literal[
        "7220d8fc0b039872fa278a0227f1dc58c7a42709c3deab592e9ec31ce9d1e3b2"
    ]
    observation_schema_version: Literal["policy_observation/v1"]
    action_space_version: Literal["frontier_index_continuous_theta/v1"]
    network_architecture_version: Literal["cross_attention_frontier_policy/v1"]
    reward_version: Literal["stage1_coverage_gain_reward/v1"]
    frontier_extractor_version: Literal["stage2_observed_frontier_top_m/v1"]
    planner_version: Literal["stage1_path_planner_adapter/v1"]
    frontier_top_m: Literal[512]
    scale_profile: Literal["Smoke v1"]
    device: Literal["cuda"]
    allow_cpu_fallback: Literal[False]
    worker_timeout_seconds: Literal[120.0]
    training_seed: Literal[20260714]
    rollout: Stage4RolloutConfig
    ppo: Stage4PPOConfig
    checkpoint: Stage4CheckpointConfig
    acceptance: Stage4AcceptanceConfig
    output_root: Literal["D:/xunce/out/ppo_frontier"]
    run_id: str | None = None

    @model_validator(mode="after")
    def validate_frozen_contracts(self) -> "Stage4Config":
        expected_authority = {
            "run_id": "s3-task4-cifix-final-20260713T213739Z",
            "commit_sha256": "4df7be92cd6517e77c648e890bf7d32c9fcd560b",
            "commit_parent": "898911559ccc9ae8ef701b69a58a93b1d8a4d8d8",
            "commit_tree": "689876d5031eac53290397c534be3a155d017b10",
            "commit_subject": "feat: add cross-attention frontier policy",
            "gate_path": (
                "D:/xunce/out/ppo_frontier/"
                "s3-task4-cifix-final-20260713T213739Z/s3/gate.json"
            ),
            "gate_sha256": (
                "6ea3dc5bc199dd9370fab4b1a5aac1ebab08d9e42f49c0d01b3aa2b1e9aae0f0"
            ),
            "approval_sha256": (
                "8e901e04d587110de681da302ad24025dfebed945ff646ffa1e85bc5f5ecb3f2"
            ),
            "review_sha256": (
                "62bf0869ba66f22bfec07464b7d17478d27fa618c3ae645039c98333262b2afc"
            ),
            "manifest_sha256": (
                "2715e790d843ac9eb873b9123a569f5254e603a734f86aa9629ac8abce142a90"
            ),
            "config_sha256": (
                "89b1a4b860d6732143b3ff7ddca7f9b02573e268a1454f709ab3957602998947"
            ),
            "source_set_sha256": (
                "26ef6026b40107347ea35b6d2545f47fee1297f7a181a5d4f517183609d9759c"
            ),
            "environment_sha256": (
                "dac42c3bc7d28e4acb78264ab7b5955c89f5716b7a841eae93a4658704bdd7c0"
            ),
            "stage2_gate_sha256": (
                "997203efa6bedd0d0e7d124f8b2362568983c1eac48a4b9e588c80d41b33dcb2"
            ),
            "reviewed_path_set_sha256": (
                "dc75183d3be7993893fdbc4b169b00e3d24d27986a2c8e85442c2a8ba6a0754d"
            ),
            "authorized_stage": (
                "ppo_highres_frontier_stage4_rollout_ppo_update/v1"
            ),
        }
        expected_rollout = {
            "layout": "time_major_t_e/v1",
            "env_count": 8,
            "trainable_steps_per_env": 128,
            "batch_size": 1024,
            "multiprocessing_start_method": "spawn",
            "diagnostic_empty_reset_counts_toward_quota": False,
            "fake_logprob_allowed": False,
            "all_done_are_terminal": True,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "advantage_normalization": "full_batch_population_std/v1",
        }
        expected_ppo = {
            "ppo_epochs": 4,
            "effective_minibatch_size": 256,
            "physical_microbatch_size": 32,
            "shuffle": True,
            "optimizer": "AdamW",
            "learning_rate": 3.0e-4,
            "adam_eps": 1.0e-5,
            "weight_decay": 1.0e-4,
            "clip_eps": 0.2,
            "value_clip_eps": 0.2,
            "value_loss_coef": 0.5,
            "frontier_entropy_coef": 0.01,
            "theta_entropy_enabled": False,
            "max_grad_norm": 0.5,
            "target_kl": 0.03,
            "compute_dtype": "float32",
            "amp_enabled": False,
        }
        expected_checkpoint = {
            "schema_version": "stage4_complete_checkpoint/v1",
            "save_every_update": True,
            "periodic_every_updates": 50,
            "periodic_keep_count": 5,
            "latest_keep_count": 1,
            "complete_marker_required": True,
            "sha256_manifest_required": True,
            "overwrite_complete_update": False,
            "restore_last_complete_only": True,
        }
        expected_acceptance = {
            "smoke_updates": 3,
            "trainable_transitions_per_update": 1024,
            "tiny_overfit_contexts": 4,
            "tiny_overfit_seeds": (17, 29, 43),
            "tiny_overfit_max_updates": 200,
            "tiny_frontier_probability_min": 0.95,
            "tiny_theta_error_rad_max": 0.1,
            "tiny_value_mse_reduction_min": 0.90,
        }
        actual_expected = (
            (self.stage3_authority.model_dump(), expected_authority),
            (self.rollout.model_dump(), expected_rollout),
            (self.ppo.model_dump(), expected_ppo),
            (self.checkpoint.model_dump(), expected_checkpoint),
            (self.acceptance.model_dump(), expected_acceptance),
        )
        if any(actual != expected for actual, expected in actual_expected):
            raise ValueError("Stage 4 frozen contract drifted")
        return self


def load_stage4_config(path: str | Path) -> Stage4Config:
    return Stage4Config.model_validate_json(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "Stage3AuthorityConfig",
    "Stage4AcceptanceConfig",
    "Stage4CheckpointConfig",
    "Stage4Config",
    "Stage4PPOConfig",
    "Stage4RolloutConfig",
    "load_stage4_config",
]
