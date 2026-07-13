"""Frozen Smoke v1 Stage 1 configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Stage1Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["ppo_highres_frontier_stage1_smoke_config/v1"]
    goal_id: Literal["ppo-highres-frontier-map-exploration"]
    stage_id: Literal["ppo_highres_frontier_stage1_smoke_environment/v1"]
    authorized_next_stage: Literal["ppo_highres_frontier_stage2_observation_frontier/v1"]
    foundation_base_commit: Literal["7378737a0d18ce6e4779e30605b557bd1ab25e6e"]
    scale_profile: Literal["Smoke v1"]
    scenario_key: Literal["smoke-v1"]
    roi_width_m: Literal[64.0]
    roi_height_m: Literal[64.0]
    highres_width: Literal[128]
    highres_height: Literal[128]
    highres_resolution_m: Literal[0.5]
    lowres_width: Literal[32]
    lowres_height: Literal[32]
    lowres_resolution_m: Literal[2.0]
    local_crop_size: Literal[64]
    frontier_top_m: Literal[512]
    max_steps: Literal[64]
    stagnation_no_gain_steps: Literal[8]
    synthetic_source_kind: Literal["synthetic_terrain_obstacle_proxy/v1"]
    physical_obstacle_cells_written: Literal[False]
    proxy_generator_version: Literal["procedural_lunar_rock_crater_proxy/v1"]
    proxy_density_profile: Literal["medium"]
    proxy_base_seed: Literal[20260710]
    proxy_start_protection_m: Literal[6.0]
    proxy_max_scene_attempts: Literal[64]
    proxy_max_object_attempts: Literal[256]
    value_prior_source: Literal["constant_neutral/v1"]
    vehicle_radius_m: Literal[0.4215874761]
    safety_margin_m: Literal[0.10]
    min_clearance_m: float = Field(strict=True)
    traversability_threshold: Literal[0.50]
    max_traversable_slope_deg: Literal[30.0]
    sensor_model_id: Literal["path-tangent-plus-endpoint-theta-fov-90-range-20m-los/v1"]
    sensor_range_m: Literal[20.0]
    sensor_fov_deg: Literal[90.0]
    sensor_ray_angle_step_deg: Literal[1.0]
    path_observation_step_m: Literal[1.0]
    connectivity: Literal[8]
    prevent_diagonal_corner_cutting: Literal[True]
    hard_path_budget_enabled: Literal[False]
    coverage_gain_reward_scale: Literal[100.0]
    success_coverage_rate: Literal[0.99]
    success_bonus: Literal[100.0]
    invalid_action_penalty: Literal[2.0]
    safety_violation_penalty: Literal[20.0]
    deterministic_episodes: Literal[10]
    output_root: Literal["D:/xunce/out/ppo_frontier"]
    run_id: str | None = None

    @property
    def sensor_range_cells(self) -> int:
        return int(self.sensor_range_m / self.highres_resolution_m)

    @model_validator(mode="after")
    def validate_derived_contracts(self) -> Stage1Config:
        if self.min_clearance_m != self.vehicle_radius_m + self.safety_margin_m:
            raise ValueError("min_clearance_m must equal vehicle_radius_m + safety_margin_m")
        if self.roi_width_m != self.highres_width * self.highres_resolution_m:
            raise ValueError("highres width does not span the ROI")
        if self.roi_height_m != self.highres_height * self.highres_resolution_m:
            raise ValueError("highres height does not span the ROI")
        if self.roi_width_m != self.lowres_width * self.lowres_resolution_m:
            raise ValueError("lowres width does not span the ROI")
        if self.roi_height_m != self.lowres_height * self.lowres_resolution_m:
            raise ValueError("lowres height does not span the ROI")
        return self


def load_stage1_config(path: str | Path) -> Stage1Config:
    return Stage1Config.model_validate_json(Path(path).read_text(encoding="utf-8"))
