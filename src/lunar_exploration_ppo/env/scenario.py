"""Deterministic procedural Smoke v1 scenario source."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from lunar_exploration_ppo.env.terrain_proxy import (
    GENERATOR_VERSION,
    ProceduralTerrainProxyGenerator,
    TerrainProxyCatalog,
    TerrainProxySettings,
)
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry, PoseXYTheta


DEFAULT_SMOKE_PROXY_SETTINGS = TerrainProxySettings(
    generator_version=GENERATOR_VERSION,
    density_profile="medium",
    base_seed=20260710,
    start_protection_m=6.0,
    max_scene_attempts=64,
    max_object_attempts=256,
)


@dataclass(frozen=True, slots=True)
class TruthMap:
    geometry: GridGeometry
    height: np.ndarray
    hard_obstacle: np.ndarray
    slope_deg: np.ndarray
    traversability: np.ndarray
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        for layer in (self.height, self.hard_obstacle, self.slope_deg, self.traversability):
            layer.setflags(write=False)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


@dataclass(frozen=True, slots=True)
class LowResolutionPrior:
    channels: np.ndarray
    resolution_m: float
    value_prior_source: str

    def __post_init__(self) -> None:
        self.channels.setflags(write=False)


@dataclass(frozen=True, slots=True)
class ScenarioBundle:
    scenario_id: str
    scenario_hash: str
    truth: TruthMap
    prior: LowResolutionPrior
    start_pose: PoseXYTheta
    proxy_catalog: TerrainProxyCatalog
    proxy_layer_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "proxy_layer_hashes",
            MappingProxyType(dict(self.proxy_layer_hashes)),
        )


class ScenarioSource:
    """Load programmatic scenarios without claiming native lunar resolution."""

    def __init__(self, settings: TerrainProxySettings | None = None) -> None:
        self._settings = settings or DEFAULT_SMOKE_PROXY_SETTINGS

    def load(self, key: str) -> ScenarioBundle:
        if key != "smoke-v1":
            raise KeyError(f"unknown Stage 1 scenario: {key}")
        geometry = GridGeometry(width=128, height=128, resolution_m=0.5)
        yy, xx = np.mgrid[0:128, 0:128]
        base_height = (
            0.01 * xx
            + 0.006 * yy
            + 0.05 * np.sin(xx / 11.0) * np.cos(yy / 13.0)
        ).astype(np.float64)
        start_pose = PoseXYTheta(CellXY(16, 64), 0.0)
        proxy = ProceduralTerrainProxyGenerator(self._settings).generate(
            base_height,
            geometry,
            start_pose,
            scenario_key=key,
            split="stage1",
            parent_roi="smoke-parent-roi",
        )
        provenance = MappingProxyType(
            {
                "source": "procedural_smoke_rock_crater_fixture/v1",
                "proxy_generator_version": proxy.catalog.generator_version,
                "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
                "physical_obstacle_cells_written": False,
                "density_profile": proxy.catalog.density_profile,
                "seed_derivation_version": "canonical_utf8_sha256_truncated_128bit/v1",
                "seed_hex": proxy.catalog.seed_hex,
                "generation_attempt": proxy.catalog.generation_attempt,
                "rock_count": len(proxy.catalog.rocks),
                "crater_count": len(proxy.catalog.craters),
                "object_catalog_sha256": proxy.catalog.sha256,
                "height_sha256": proxy.layer_hashes["height"],
                "hard_obstacle_sha256": proxy.layer_hashes["hard_obstacle"],
                "slope_sha256": proxy.layer_hashes["slope"],
                "traversability_sha256": proxy.layer_hashes["traversability"],
                "layer_hashes": MappingProxyType(dict(proxy.layer_hashes)),
            }
        )
        truth = TruthMap(
            geometry=geometry,
            height=proxy.height,
            hard_obstacle=proxy.hard_obstacle,
            slope_deg=proxy.slope_deg,
            traversability=proxy.traversability,
            provenance=provenance,
        )
        low_y, low_x = np.mgrid[0:32, 0:32]
        prior_channels = np.stack(
            (
                np.full((32, 32), 0.5),
                low_x / 31.0,
                low_y / 31.0,
                np.full((32, 32), 0.9),
                np.zeros((32, 32)),
                np.full((32, 32), 0.5),
                np.ones((32, 32)),
            ),
            axis=0,
        ).astype(np.float32)
        prior_channels.setflags(write=False)
        prior = LowResolutionPrior(
            channels=prior_channels,
            resolution_m=2.0,
            value_prior_source="constant_neutral/v1",
        )
        scenario_id = "smoke-v1/procedural-rock-crater/v1"
        digest = hashlib.sha256()
        digest.update(b"stage1-rock-crater-scenario-identity/v1\0")
        digest.update(ArtifactStore.canonical_json_bytes({
            "scenario_id": scenario_id,
            "generator_version": proxy.catalog.generator_version,
        }))
        digest.update(ArtifactStore.canonical_json_bytes({
            "width": geometry.width,
            "height": geometry.height,
            "resolution_m": geometry.resolution_m,
            "origin_x": geometry.origin.x,
            "origin_y": geometry.origin.y,
        }))
        digest.update(ArtifactStore.canonical_json_bytes({
            "start_x": start_pose.cell.x,
            "start_y": start_pose.cell.y,
            "start_theta": start_pose.theta,
        }))
        digest.update(proxy.catalog.sha256.encode("ascii"))
        digest.update(b"\0")
        for layer in (
            proxy.height,
            proxy.hard_obstacle,
            proxy.slope_deg,
            proxy.traversability,
            prior_channels,
        ):
            digest.update(np.ascontiguousarray(layer).tobytes())
        return ScenarioBundle(
            scenario_id=scenario_id,
            scenario_hash=digest.hexdigest(),
            truth=truth,
            prior=prior,
            start_pose=start_pose,
            proxy_catalog=proxy.catalog,
            proxy_layer_hashes=proxy.layer_hashes,
        )
