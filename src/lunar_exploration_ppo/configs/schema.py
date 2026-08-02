from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DEM_PATH: Final = "D:/CodexDownloads/lunar-path-planning/data/raw/high_resolution_lunar_terrain/lunar_south_pole_usgs_lro_dem_slope_4m/MOON_LRO_NAC_DEM_89S210E_4mp.tif"
DEM_SIZE_BYTES: Final = 310140246
DEM_SHA256: Final = "7c431d32d977cd6b66572a0fae874ec5eae91be16aa21c4149ac897c677add22"
SLOPE_PATH: Final = "D:/CodexDownloads/lunar-path-planning/data/raw/high_resolution_lunar_terrain/lunar_south_pole_usgs_lro_dem_slope_4m/MOON_LRO_NAC_Slope_89S210E_4mp.tif"
SLOPE_SIZE_BYTES: Final = 70131711
SLOPE_SHA256: Final = "df741123daec2d935b12481a579ce4443ce6c2e8916b13906ecd7801961112dd"
FOUNDATION_WORKTREE_ROOT: Final = "C:/Users/77634/.codex/worktrees/ca49/lunar-path-planning"
FOUNDATION_BRANCH: Final = "codex/ppo-highres-frontier-map-exploration"
FOUNDATION_GIT_DIR: Final = "D:/codex/project/lunar-path-planning/.git/worktrees/lunar-path-planning1"
FOUNDATION_GIT_COMMON_DIR: Final = "D:/codex/project/lunar-path-planning/.git"
FOUNDATION_REPOSITORY_IDENTITY_SOURCE: Final = "foundation_linked_worktree_identity/v1"


class RepositoryIdentityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["foundation_linked_worktree_identity/v1"]
    worktree_root: Literal["C:/Users/77634/.codex/worktrees/ca49/lunar-path-planning"]
    branch: Literal["codex/ppo-highres-frontier-map-exploration"]
    git_dir: Literal["D:/codex/project/lunar-path-planning/.git/worktrees/lunar-path-planning1"]
    git_common_dir: Literal["D:/codex/project/lunar-path-planning/.git"]


class DemSourceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Literal[DEM_PATH]
    size_bytes: Literal[310140246]
    sha256: Literal["7c431d32d977cd6b66572a0fae874ec5eae91be16aa21c4149ac897c677add22"]


class SlopeSourceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Literal[SLOPE_PATH]
    size_bytes: Literal[70131711]
    sha256: Literal["df741123daec2d935b12481a579ce4443ce6c2e8916b13906ecd7801961112dd"]


class VerifiedDataSourcesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_access: Literal["metadata_only_no_policy_read/v1"]
    dem: DemSourceEvidence
    slope: SlopeSourceEvidence


class InitialObservationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["reset_start_pose_los_scan/v1"]
    range_m: Literal[20.0]
    fov_deg: Literal[90.0]
    counts_reward: Literal[False]
    counts_step: Literal[False]


class FoundationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["ppo_highres_frontier_foundation/v1"]
    goal_id: Literal["ppo-highres-frontier-map-exploration"]
    stage_id: Literal["foundation"]
    repository_identity: RepositoryIdentityConfig
    synthetic_source_kind: Literal["synthetic_terrain_obstacle_proxy/v1"]
    physical_obstacle_cells_written: Literal[False]
    value_prior_source: Literal["constant_neutral/v1"]
    coordinate_convention: Literal["world_xy_grid_col_row_cell_center/v1"]
    theta_convention: Literal["radians_world_x_ccw_normalized_minus_pi_to_pi/v1"]
    grid_resolution_m: Literal[0.5]
    world_origin_xy_m: tuple[float, float]
    cell_00_world_center_xy_m: tuple[float, float]
    vehicle_radius_m: Literal[0.4215874761]
    safety_margin_m: Literal[0.10]
    min_clearance_m: float = Field(strict=True)
    traversability_threshold: Literal[0.50]
    max_traversable_slope_deg: Literal[30.0]
    sensor_model_id: Literal["path-tangent-plus-endpoint-theta-fov-90-range-20m-los/v1"]
    initial_observation: InitialObservationConfig
    verified_data_sources: VerifiedDataSourcesConfig
    device: Literal["cpu", "cuda"]
    output_root: Literal["D:/xunce/out/ppo_frontier"]
    run_id: str | None = None

    @model_validator(mode="after")
    def validate_derived_contracts(self) -> FoundationConfig:
        expected_clearance = self.vehicle_radius_m + self.safety_margin_m
        if self.min_clearance_m != expected_clearance:
            raise ValueError("min_clearance_m must exactly match vehicle_radius_m + safety_margin_m")
        if self.world_origin_xy_m != (0.0, 0.0):
            raise ValueError("world_origin_xy_m is frozen at (0.0, 0.0)")
        if self.cell_00_world_center_xy_m != (0.25, 0.25):
            raise ValueError("cell_00_world_center_xy_m is frozen at (0.25, 0.25)")
        return self

    @staticmethod
    def cell_xy_to_array_indices(x: int, y: int) -> tuple[int, int]:
        return y, x

    def cell_center_world_xy(self, x: int, y: int) -> tuple[float, float]:
        origin_x, origin_y = self.world_origin_xy_m
        return (
            origin_x + (x + 0.5) * self.grid_resolution_m,
            origin_y + (y + 0.5) * self.grid_resolution_m,
        )


def load_foundation_config(path: str | Path) -> FoundationConfig:
    return FoundationConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def resolve_device(requested: Literal["cpu", "cuda"], *, cuda_available: bool | None = None) -> Literal["cpu", "cuda"]:
    if requested == "cpu":
        return "cpu"
    if cuda_available is None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("CUDA was requested but PyTorch is not installed") from exc
        cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        raise RuntimeError("CUDA was requested but is unavailable; refusing CPU fallback")
    return "cuda"
