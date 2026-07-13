"""Frozen Stage 2 observation/frontier/catalog workflow configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from lunar_exploration_ppo.configs.schema import DEM_PATH, DEM_SHA256, DEM_SIZE_BYTES, SLOPE_PATH, SLOPE_SHA256, SLOPE_SIZE_BYTES


class ScaleProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    highres_shape: tuple[int, int]
    highres_resolution_m: float
    lowres_shape: tuple[int, int]
    lowres_resolution_m: float
    local_crop_shape: tuple[int, int]
    frontier_top_m: int


class Stage2Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["ppo_highres_frontier_stage2_config/v1"]
    goal_id: Literal["ppo-highres-frontier-map-exploration"]
    stage_id: Literal["ppo_highres_frontier_stage2_observation_frontier/v1"]
    authorized_next_stage: Literal["ppo_highres_frontier_stage3_cross_attention_policy/v1"]
    stage1_approval_commit: Literal["deb876032054f300c68a4215d597f607df2a356b"]
    split_policy: Literal["fixed_seed_disjoint_split/v1"]
    catalog_counts: dict[str, int]
    catalog_base_seed: Literal[20260712]
    smoke: ScaleProfileConfig
    standard: ScaleProfileConfig
    kilometer: ScaleProfileConfig
    dem_path: Literal[DEM_PATH]
    dem_size_bytes: Literal[310140246]
    dem_sha256: Literal[DEM_SHA256]
    slope_path: Literal[SLOPE_PATH]
    slope_size_bytes: Literal[70131711]
    slope_sha256: Literal[SLOPE_SHA256]
    slope_product_semantics: Literal["rgb_visualization_provenance_only/v1"]
    physical_slope_source: Literal["dem_float32_metric_gradient_4m/v1"]
    slope_pixels_used_for_traversability: Literal[False]
    proxy_generator_version: Literal["procedural_lunar_rock_crater_proxy/v1"]
    synthetic_source_kind: Literal["synthetic_terrain_obstacle_proxy/v1"]
    physical_obstacle_cells_written: Literal[False]
    output_root: Literal["D:/xunce/out/ppo_frontier"]
    run_id: str | None = None

    @model_validator(mode="after")
    def validate_frozen_contracts(self) -> "Stage2Config":
        if self.catalog_counts != {"train": 700, "validation": 150, "test": 150, "unseen": 64}:
            raise ValueError("Stage 2 catalog counts drifted")
        expected = {
            "smoke": ((128, 128), 0.5, (32, 32), 2.0, (64, 64), 512),
            "standard": ((256, 256), 0.5, (32, 32), 4.0, (96, 96), 1024),
            "kilometer": ((2048, 2048), 0.5, (128, 128), 8.0, (192, 192), 2048),
        }
        for name, frozen in expected.items():
            profile = getattr(self, name)
            actual = (
                profile.highres_shape, profile.highres_resolution_m, profile.lowres_shape,
                profile.lowres_resolution_m, profile.local_crop_shape, profile.frontier_top_m,
            )
            if actual != frozen:
                raise ValueError(f"{name} scale profile drifted")
        if (self.dem_size_bytes, self.slope_size_bytes) != (DEM_SIZE_BYTES, SLOPE_SIZE_BYTES):
            raise ValueError("frozen source sizes drifted")
        return self


def load_stage2_config(path: str | Path) -> Stage2Config:
    return Stage2Config.model_validate_json(Path(path).read_text(encoding="utf-8"))
