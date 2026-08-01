"""Stage 6 Standard v1 正式训练与评估冻结配置。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from collections.abc import Mapping
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, model_validator


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Stage5AuthorityConfig(_FrozenModel):
    commit_sha256: str
    commit_tree: str
    gate_path: str
    gate_sha256: str
    review_sha256: str
    manifest_sha256: str
    checkpoint_sha256: str
    policy_state_sha256: str
    authorized_stage: str
    performance_advantage_established: bool


class Stage6DataConfig(_FrozenModel):
    dem_path: str
    dem_bytes: int
    dem_sha256: str
    slope_provenance_path: str
    slope_bytes: int
    slope_sha256: str
    slope_product_semantics: str
    physical_slope_source: str


class Stage6ScaleConfig(_FrozenModel):
    profile: str
    roi_size_m: int
    highres_shape: tuple[int, int]
    highres_cell_size_m: float
    lowres_shape: tuple[int, int]
    lowres_cell_size_m: float
    local_crop_shape: tuple[int, int]
    frontier_top_m: int
    max_steps: int
    stagnation_no_gain_steps: int
    success_threshold: float
    coverage_denominator: str
    coverable_mask_exact: bool
    truth_source_kind: str


class Stage6SafetyConfig(_FrozenModel):
    vehicle_radius_m: float
    safety_margin_m: float
    min_clearance_m: float
    traversability_threshold: float
    max_traversable_slope_deg: float

    @model_validator(mode="after")
    def validate_safety_contract(self) -> "Stage6SafetyConfig":
        expected = {
            "vehicle_radius_m": 0.4215874761,
            "safety_margin_m": 0.10,
            "min_clearance_m": 0.5215874761,
            "traversability_threshold": 0.50,
            "max_traversable_slope_deg": 30.0,
        }
        if (
            self.model_dump() != expected
            or abs(
                self.vehicle_radius_m
                + self.safety_margin_m
                - self.min_clearance_m
            )
            > 1.0e-12
        ):
            raise ValueError("Stage 6 safety contract drifted")
        return self


class Stage6RolloutConfig(_FrozenModel):
    num_envs: int
    steps_per_env: int
    batch_size: int
    multiprocessing_start_method: str
    diagnostic_empty_reset_counts_toward_quota: bool


class Stage6PPOConfig(_FrozenModel):
    ppo_epochs: int
    effective_minibatch_size: int
    physical_microbatch_size: int
    gamma: float
    gae_lambda: float
    clip_eps: float
    value_clip_eps: float
    value_loss_coef: float
    frontier_entropy_coef: float
    theta_entropy_enabled: bool
    optimizer: str
    learning_rate: float
    adam_eps: float
    weight_decay: float
    max_grad_norm: float
    target_kl: float
    compute_dtype: str
    amp_enabled: bool


class Stage6TrainingConfig(_FrozenModel):
    initialization_source: str
    seeds: tuple[int, ...]
    updates_per_seed: int
    validation_every_updates: int
    validation_episodes: int
    validation_policy: str
    periodic_updates: tuple[int, ...]
    periodic_keep_count: int


class Stage6EvaluationConfig(_FrozenModel):
    test_episodes: int
    unseen_episodes: int
    baseline_methods: tuple[str, ...]
    baseline_episodes_per_method_split: int
    bootstrap_resamples: int
    bootstrap_seed: int
    ppo_theta_source: str
    baseline_theta_source: str


class Stage6CheckpointConfig(_FrozenModel):
    schema_version: str
    save_every_update: bool
    latest_keep_count: int
    periodic_keep_count: int
    complete_marker_required: bool
    restore_last_complete_only: bool
    vector_env_state_count: int


class Stage6ResourceConfig(_FrozenModel):
    preflight_d_free_gib_min: float
    runtime_d_free_gib_hard_stop_below: float
    rss_warning_gib: float
    rss_hard_stop_at_or_above_gib: float
    vram_warning_gib: float
    vram_hard_stop_at_or_above_gib: float


class Stage6Config(_FrozenModel):
    schema_version: Literal["ppo_highres_frontier_stage6_config/v1"]
    goal_id: Literal["ppo-highres-frontier-map-exploration"]
    stage_id: Literal["ppo_highres_frontier_stage6_standard_training_eval/v1"]
    stage5_authority: Stage5AuthorityConfig
    data: Stage6DataConfig
    scale: Stage6ScaleConfig
    safety: Stage6SafetyConfig
    rollout: Stage6RolloutConfig
    ppo: Stage6PPOConfig
    training: Stage6TrainingConfig
    evaluation: Stage6EvaluationConfig
    checkpoint: Stage6CheckpointConfig
    resources: Stage6ResourceConfig
    catalog_split_counts: dict[str, int]
    audit_updates: tuple[int, ...]
    observation_schema_version: str
    action_space_version: str
    network_architecture_version: str
    reward_version: str
    planner_version: str
    candidate_snapshot_contract: str
    terminal_bootstrap: float
    device: Literal["cuda"]
    allow_cpu_fallback: Literal[False]
    output_root: Literal["D:/xunce/out/ppo_frontier"]
    run_id: str | None = None

    @property
    def planning_unknown_buffer_m(self) -> float:
        return 0.75

    @property
    def reset_local_safety_scan_range_m(self) -> float:
        return 0.75

    @property
    def reset_local_safety_scan_fov_deg(self) -> float:
        return 360.0

    @property
    def reset_local_safety_scan_ray_angle_step_deg(self) -> float:
        return 1.0

    @model_validator(mode="after")
    def validate_frozen_contracts(self) -> "Stage6Config":
        actual = self.model_dump(exclude={"run_id"})
        if actual != _EXPECTED_CONFIG:
            raise ValueError("Stage 6 frozen contract drifted")
        return self


_EXPECTED_CONFIG: dict[str, object] = {
    "schema_version": "ppo_highres_frontier_stage6_config/v1",
    "goal_id": "ppo-highres-frontier-map-exploration",
    "stage_id": "ppo_highres_frontier_stage6_standard_training_eval/v1",
    "stage5_authority": {
        "commit_sha256": "b635740ee021258ef31811ec87c60add839fc5f9",
        "commit_tree": "edec4c4dbed7f91efa9856a9df899c594bbcf04c",
        "gate_path": "D:/xunce/out/ppo_frontier/s5-task6-fix-r1-20260714T223702Z/s5/gate.json",
        "gate_sha256": "5fc93a7fb0d1d23c1c2ee99db14e15f6880c422d1ce85a83946542ebbe238bdf",
        "review_sha256": "3c69d565f27bdabf64cf1ca4728be5da420ffa5057444f29af9b717bd2d9d9db",
        "manifest_sha256": "edee94cf2a0f39f89cf040d850af154e060b07c6a66d0bfc47386d5b05ee5672",
        "checkpoint_sha256": "d1d80e6478262a68d01208ddc1e8721768b56109e8dd009e862dd668cb39dc4c",
        "policy_state_sha256": "51fecca55af92838f302951ae1272063cd945ef481579af5f6600469d9a7fe86",
        "authorized_stage": "ppo_highres_frontier_stage6_standard_training_eval/v1",
        "performance_advantage_established": False,
    },
    "data": {
        "dem_path": "D:/CodexDownloads/lunar-path-planning/data/raw/high_resolution_lunar_terrain/lunar_south_pole_usgs_lro_dem_slope_4m/MOON_LRO_NAC_DEM_89S210E_4mp.tif",
        "dem_bytes": 310140246,
        "dem_sha256": "7c431d32d977cd6b66572a0fae874ec5eae91be16aa21c4149ac897c677add22",
        "slope_provenance_path": "D:/CodexDownloads/lunar-path-planning/data/raw/high_resolution_lunar_terrain/lunar_south_pole_usgs_lro_dem_slope_4m/MOON_LRO_NAC_Slope_89S210E_4mp.tif",
        "slope_bytes": 70131711,
        "slope_sha256": "df741123daec2d935b12481a579ce4443ce6c2e8916b13906ecd7801961112dd",
        "slope_product_semantics": "rgb_visualization_provenance_only/v1",
        "physical_slope_source": "dem_float32_metric_gradient_4m/v1",
    },
    "scale": {
        "profile": "Standard v1",
        "roi_size_m": 128,
        "highres_shape": (256, 256),
        "highres_cell_size_m": 0.5,
        "lowres_shape": (32, 32),
        "lowres_cell_size_m": 4.0,
        "local_crop_shape": (96, 96),
        "frontier_top_m": 1024,
        "max_steps": 128,
        "stagnation_no_gain_steps": 16,
        "success_threshold": 0.99,
        "coverage_denominator": "reachable_observable_free_highres_cells/v1",
        "coverable_mask_exact": True,
        "truth_source_kind": "procedural_lunar_rock_crater_proxy/v1",
    },
    "safety": {
        "vehicle_radius_m": 0.4215874761,
        "safety_margin_m": 0.10,
        "min_clearance_m": 0.5215874761,
        "traversability_threshold": 0.50,
        "max_traversable_slope_deg": 30.0,
    },
    "rollout": {
        "num_envs": 8,
        "steps_per_env": 128,
        "batch_size": 1024,
        "multiprocessing_start_method": "spawn",
        "diagnostic_empty_reset_counts_toward_quota": False,
    },
    "ppo": {
        "ppo_epochs": 4,
        "effective_minibatch_size": 256,
        "physical_microbatch_size": 32,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "value_clip_eps": 0.2,
        "value_loss_coef": 0.5,
        "frontier_entropy_coef": 0.01,
        "theta_entropy_enabled": False,
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "adam_eps": 1e-5,
        "weight_decay": 1e-4,
        "max_grad_norm": 0.5,
        "target_kl": 0.03,
        "compute_dtype": "float32",
        "amp_enabled": False,
    },
    "training": {
        "initialization_source": "stage4_smoke_policy_weights_fresh_adamw/v1",
        "seeds": (20260716,),
        "updates_per_seed": 100,
        "validation_every_updates": 10,
        "validation_episodes": 16,
        "validation_policy": "deterministic_argmax_frontier_mean_theta/v1",
        "periodic_updates": (50, 100),
        "periodic_keep_count": 5,
    },
    "evaluation": {
        "test_episodes": 64,
        "unseen_episodes": 64,
        "baseline_methods": (
            "random_valid_frontier",
            "nearest_frontier",
            "max_potential_gain_frontier",
            "gain_over_cost_frontier",
        ),
        "baseline_episodes_per_method_split": 64,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20260716,
        "ppo_theta_source": "policy_theta_mu/v1",
        "baseline_theta_source": "candidate_recommended_theta/v1",
    },
    "checkpoint": {
        "schema_version": "stage6_complete_checkpoint/v1",
        "save_every_update": True,
        "latest_keep_count": 1,
        "periodic_keep_count": 5,
        "complete_marker_required": True,
        "restore_last_complete_only": True,
        "vector_env_state_count": 8,
    },
    "resources": {
        "preflight_d_free_gib_min": 100.0,
        "runtime_d_free_gib_hard_stop_below": 50.0,
        "rss_warning_gib": 16.0,
        "rss_hard_stop_at_or_above_gib": 20.0,
        "vram_warning_gib": 9.0,
        "vram_hard_stop_at_or_above_gib": 10.1,
    },
    "catalog_split_counts": {
        "train": 700,
        "validation": 150,
        "test": 150,
        "unseen": 64,
    },
    "audit_updates": (1, 10, 50, 100),
    "observation_schema_version": "policy_observation/v1",
    "action_space_version": "frontier_index_continuous_theta/v1",
    "network_architecture_version": "cross_attention_frontier_policy/v1",
    "reward_version": "stage1_coverage_gain_reward/v1",
    "planner_version": "stage1_path_planner_adapter/v1",
    "candidate_snapshot_contract": "rollout_time_candidate_snapshot_no_update_reextract/v1",
    "terminal_bootstrap": 0.0,
    "device": "cuda",
    "allow_cpu_fallback": False,
    "output_root": "D:/xunce/out/ppo_frontier",
}


SAFETY_CONTRACT_SOURCE: Final = "Stage6Config.safety/v1"
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_SAFETY_FIELDS: Final = (
    "vehicle_radius_m",
    "safety_margin_m",
    "min_clearance_m",
    "traversability_threshold",
    "max_traversable_slope_deg",
)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True, init=False)
class SafetyContract:
    """由已验证 ``Stage6Config.safety`` 派生的五字段运行时合同。"""

    vehicle_radius_m: float
    safety_margin_m: float
    min_clearance_m: float
    traversability_threshold: float
    max_traversable_slope_deg: float

    def __init__(self, safety: Stage6SafetyConfig) -> None:
        if not isinstance(safety, Stage6SafetyConfig):
            raise TypeError("SafetyContract requires validated Stage6Config.safety")
        validated = Stage6SafetyConfig.model_validate(
            safety.model_dump(mode="python")
        )
        for name in _SAFETY_FIELDS:
            object.__setattr__(self, name, float(getattr(validated, name)))

    @classmethod
    def from_stage6_config(cls, config: Stage6Config) -> "SafetyContract":
        if not isinstance(config, Stage6Config):
            raise TypeError("SafetyContract requires validated Stage6Config")
        return cls(
            Stage6SafetyConfig.model_validate(
                config.safety.model_dump(mode="python")
            )
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SafetyContract":
        if not isinstance(value, Mapping) or set(value) != set(_SAFETY_FIELDS):
            raise ValueError("Stage 6 safety contract field set drifted")
        return cls(Stage6SafetyConfig.model_validate(dict(value)))

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in _SAFETY_FIELDS}

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_dict())).hexdigest()

    def binding(self, *, config_sha256: str) -> dict[str, object]:
        _require_sha256(config_sha256, "safety contract config")
        return {
            "safety_contract": self.to_dict(),
            "safety_contract_sha256": self.sha256,
            "safety_contract_source": SAFETY_CONTRACT_SOURCE,
            "safety_contract_config_sha256": config_sha256,
        }

    @classmethod
    def from_binding(
        cls,
        value: Mapping[str, object],
        *,
        expected_config_sha256: str | None = None,
    ) -> "SafetyContract":
        required = {
            "safety_contract",
            "safety_contract_sha256",
            "safety_contract_source",
            "safety_contract_config_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("Stage 6 safety contract binding field set drifted")
        if value["safety_contract_source"] != SAFETY_CONTRACT_SOURCE:
            raise ValueError("Stage 6 safety contract source drifted")
        config_sha256 = value["safety_contract_config_sha256"]
        _require_sha256(config_sha256, "safety contract config")
        if expected_config_sha256 is not None:
            _require_sha256(expected_config_sha256, "expected safety config")
            if config_sha256 != expected_config_sha256:
                raise ValueError("Stage 6 safety contract config SHA drifted")
        contract_value = value["safety_contract"]
        if not isinstance(contract_value, Mapping):
            raise ValueError("Stage 6 safety contract payload drifted")
        contract = cls.from_dict(contract_value)
        if value["safety_contract_sha256"] != contract.sha256:
            raise ValueError("Stage 6 safety contract SHA drifted")
        return contract


def _require_sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} SHA256 is invalid")


def parse_stage6_config_bytes(payload: bytes) -> Stage6Config:
    """Parse one exact Stage 6 config snapshot without reopening a path."""

    if type(payload) is not bytes or not payload:
        raise ValueError("Stage 6 config snapshot bytes are invalid")
    return Stage6Config.model_validate_json(payload)


def load_stage6_config(path: str | Path) -> Stage6Config:
    return parse_stage6_config_bytes(Path(path).read_bytes())


__all__ = [
    "SAFETY_CONTRACT_SOURCE",
    "SafetyContract",
    "Stage5AuthorityConfig",
    "Stage6CheckpointConfig",
    "Stage6Config",
    "Stage6DataConfig",
    "Stage6EvaluationConfig",
    "Stage6PPOConfig",
    "Stage6ResourceConfig",
    "Stage6RolloutConfig",
    "Stage6SafetyConfig",
    "Stage6ScaleConfig",
    "Stage6TrainingConfig",
    "load_stage6_config",
    "parse_stage6_config_bytes",
]
