"""Standard v1 production catalog sampler 与 spawn-safe 训练环境。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
import time
from typing import Literal, Mapping, Sequence

import numpy as np

from lunar_exploration_ppo.configs.stage6 import (
    SAFETY_CONTRACT_SOURCE,
    SafetyContract,
)
from lunar_exploration_ppo.env.env import LunarExplorationEnv
from lunar_exploration_ppo.env.coverage_cache import Stage6CoverageManifest
from lunar_exploration_ppo.env.scenario import ScenarioBundle
from lunar_exploration_ppo.env.scenario_catalog import (
    DEM_PATH,
    DEM_SHA256,
    DEM_SIZE_BYTES,
    SLOPE_PATH,
    SLOPE_SHA256,
    SLOPE_SIZE_BYTES,
    FrozenRasterSources,
    RasterSourceProvenance,
    RasterWindow,
    ScenarioCatalogRecord,
    StandardScenarioCatalog,
    StandardScenarioCatalogBuilder,
    StandardScenarioFactory,
)
from lunar_exploration_ppo.ppo.collector import SpawnEnvSpec


Split = Literal["train", "validation", "test", "unseen"]
SPLIT_COUNTS = {"train": 700, "validation": 150, "test": 150, "unseen": 64}


@dataclass(frozen=True, slots=True)
class StandardEnvSettings:
    scenario_key: str
    safety_contract: SafetyContract
    config_sha256: str
    frontier_top_m: int = 1024
    max_steps: int = 128
    stagnation_no_gain_steps: int = 16
    success_coverage_rate: float = 0.99
    sensor_range_m: float = 20.0
    sensor_fov_deg: float = 90.0
    sensor_ray_angle_step_deg: float = 1.0
    planning_unknown_buffer_m: float = 0.75
    reset_local_safety_scan_range_m: float = 0.75
    reset_local_safety_scan_fov_deg: float = 360.0
    reset_local_safety_scan_ray_angle_step_deg: float = 1.0
    path_observation_step_m: float = 1.0
    coverage_gain_reward_scale: float = 100.0
    success_bonus: float = 100.0
    invalid_action_penalty: float = 2.0
    safety_violation_penalty: float = 20.0

    def __post_init__(self) -> None:
        _validated_typed_safety_contract(
            self.safety_contract,
            config_sha256=self.config_sha256,
        )

    @property
    def safety_contract_source(self) -> str:
        return SAFETY_CONTRACT_SOURCE

    @property
    def vehicle_radius_m(self) -> float:
        return self.safety_contract.vehicle_radius_m

    @property
    def safety_margin_m(self) -> float:
        return self.safety_contract.safety_margin_m

    @property
    def min_clearance_m(self) -> float:
        return self.safety_contract.min_clearance_m

    @property
    def max_traversable_slope_deg(self) -> float:
        return self.safety_contract.max_traversable_slope_deg

    @property
    def traversability_threshold(self) -> float:
        return self.safety_contract.traversability_threshold


@dataclass(frozen=True, slots=True)
class _BoundScenarioSource:
    key: str
    bundle: ScenarioBundle

    def load(self, key: str) -> ScenarioBundle:
        if key != self.key:
            raise KeyError(key)
        return self.bundle


class StandardScenarioSampler:
    """按 split 隔离、可由 epoch/cursor 精确恢复的确定性 sampler。"""

    def __init__(self, catalog: StandardScenarioCatalog, *, split: Split, seed: int) -> None:
        if split not in SPLIT_COUNTS:
            raise ValueError(f"unknown Standard split: {split}")
        records = tuple(record for record in catalog.records if record.split == split)
        if len(records) != SPLIT_COUNTS[split]:
            raise ValueError("Standard sampler split count drift")
        self.catalog = catalog
        self.split = split
        self.seed = int(seed)
        self.records = records
        self.epoch = 0
        self.cursor = 0
        self._order = self._order_for_epoch(0)

    def next_record(self) -> ScenarioCatalogRecord:
        if self.cursor == len(self.records):
            self.epoch += 1
            self.cursor = 0
            self._order = self._order_for_epoch(self.epoch)
        record = self.records[self._order[self.cursor]]
        self.cursor += 1
        return record

    def capture_state(self) -> dict[str, object]:
        return {
            "schema_version": "stage6_standard_scenario_sampler/v1",
            "catalog_sha256": self.catalog.sha256,
            "split": self.split,
            "seed": self.seed,
            "epoch": self.epoch,
            "cursor": self.cursor,
        }

    def restore_state(self, state: Mapping[str, object]) -> None:
        expected_keys = {
            "schema_version",
            "catalog_sha256",
            "split",
            "seed",
            "epoch",
            "cursor",
        }
        if set(state) != expected_keys:
            raise ValueError("Standard sampler state schema drift")
        if (
            state["schema_version"] != "stage6_standard_scenario_sampler/v1"
            or state["catalog_sha256"] != self.catalog.sha256
            or state["split"] != self.split
            or state["seed"] != self.seed
            or type(state["epoch"]) is not int
            or type(state["cursor"]) is not int
            or int(state["epoch"]) < 0
            or not 0 <= int(state["cursor"]) <= len(self.records)
        ):
            raise ValueError("Standard sampler state binding drift")
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        self._order = self._order_for_epoch(self.epoch)

    def _order_for_epoch(self, epoch: int) -> tuple[int, ...]:
        generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence((self.seed, epoch))))
        return tuple(int(index) for index in generator.permutation(len(self.records)))


class StandardTrainingEnv:
    """每次 reset 从绑定 split 取新场景，并复用冻结 LunarExplorationEnv 语义。"""

    def __init__(
        self,
        catalog: StandardScenarioCatalog,
        *,
        split: Split,
        sampler_seed: int,
        safety_contract: SafetyContract,
        config_sha256: str,
        coverage_cache_manifest_path: str | None = None,
        coverage_cache_manifest_sha256: str | None = None,
        initial_sampler_state: Mapping[str, object] | None = None,
    ) -> None:
        if (coverage_cache_manifest_path is None) != (coverage_cache_manifest_sha256 is None):
            raise ValueError("coverage cache manifest path and SHA-256 must be provided together")
        self._coverage_cache_manifest = (
            None
            if coverage_cache_manifest_path is None
            else Stage6CoverageManifest.load(
                coverage_cache_manifest_path,
                expected_sha256=coverage_cache_manifest_sha256,
            )
        )
        _validate_production_catalog(catalog)
        if self._coverage_cache_manifest is not None:
            _validate_coverage_cache_manifest_binding(
                self._coverage_cache_manifest,
                catalog,
            )
        self.safety_contract = _validated_typed_safety_contract(
            safety_contract,
            config_sha256=config_sha256,
        )
        self.config_sha256 = config_sha256
        self.safety_contract_source = SAFETY_CONTRACT_SOURCE
        self.catalog = catalog
        self.split = split
        self.sampler = StandardScenarioSampler(catalog, split=split, seed=sampler_seed)
        if initial_sampler_state is not None:
            self.sampler.restore_state(initial_sampler_state)
        self.factory = StandardScenarioFactory(catalog)
        self._record: ScenarioCatalogRecord | None = None
        self._bundle: ScenarioBundle | None = None
        self._inner: LunarExplorationEnv | None = None
        self._last_timing_seconds: dict[str, float] | None = None

    @property
    def scenario_record(self) -> ScenarioCatalogRecord:
        if self._record is None:
            raise RuntimeError("Standard env has not been reset")
        return self._record

    @property
    def scenario_bundle(self) -> ScenarioBundle:
        if self._bundle is None:
            raise RuntimeError("Standard env has not been reset")
        return self._bundle

    @property
    def last_timing_seconds(self) -> dict[str, float]:
        if self._last_timing_seconds is None:
            raise RuntimeError("Standard env has no completed reset timing")
        return dict(self._last_timing_seconds)

    def reset(self):
        timing = self._bind_record(self.sampler.next_record())
        started = time.perf_counter()
        observation = self._require_inner().reset()
        timing["reset"] = time.perf_counter() - started
        self._last_timing_seconds = timing
        return observation

    def step(self, action):
        return self._require_inner().step(action)

    def select_rule_action(self, observation):
        return self._require_inner().select_rule_action(observation)

    def export_sampler_state(self) -> dict[str, object]:
        return self.sampler.capture_state()

    def import_sampler_state(self, state: Mapping[str, object]) -> None:
        self.sampler.restore_state(state)

    def export_episode_state(self) -> dict[str, object]:
        return {
            "schema_version": "stage6_standard_vector_env_episode_state/v1",
            "catalog_sha256": self.catalog.sha256,
            "split": self.split,
            "scenario_id": self.scenario_record.scenario_id,
            "inner_state": self._require_inner().export_episode_state(),
        }

    def import_episode_state(self, state: Mapping[str, object]) -> None:
        if set(state) != {
            "schema_version",
            "catalog_sha256",
            "split",
            "scenario_id",
            "inner_state",
        }:
            raise ValueError("Standard episode state schema drift")
        if (
            state["schema_version"] != "stage6_standard_vector_env_episode_state/v1"
            or state["catalog_sha256"] != self.catalog.sha256
            or state["split"] != self.split
            or not isinstance(state["scenario_id"], str)
            or not isinstance(state["inner_state"], Mapping)
        ):
            raise ValueError("Standard episode state binding drift")
        matches = tuple(
            record
            for record in self.catalog.records
            if record.scenario_id == state["scenario_id"] and record.split == self.split
        )
        if len(matches) != 1:
            raise ValueError("Standard episode scenario is outside bound split")
        self._bind_record(matches[0])
        self._require_inner().import_episode_state(state["inner_state"])

    def close(self) -> None:
        self._inner = None
        self._bundle = None
        self._record = None
        self._last_timing_seconds = None

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._require_inner(), name)

    def _bind_record(self, record: ScenarioCatalogRecord) -> dict[str, float]:
        if record.split != self.split:
            raise ValueError("Standard env record crossed split")
        scenario_started = time.perf_counter()
        bundle = self.factory.build(record)
        scenario_seconds = time.perf_counter() - scenario_started
        settings = StandardEnvSettings(
            scenario_key=record.scenario_id,
            safety_contract=self.safety_contract,
            config_sha256=self.config_sha256,
        )
        source = _BoundScenarioSource(key=record.scenario_id, bundle=bundle)
        precomputed_coverage_masks = (
            None
            if self._coverage_cache_manifest is None
            else self._coverage_cache_manifest.load_masks(
                bundle,
                sensor_range_m=settings.sensor_range_m,
                min_clearance_m=settings.min_clearance_m,
                max_slope_deg=settings.max_traversable_slope_deg,
                traversability_threshold=settings.traversability_threshold,
            )
        )
        self._record = record
        self._bundle = bundle
        env_started = time.perf_counter()
        self._inner = LunarExplorationEnv(  # type: ignore[arg-type]
            settings,
            scenario_source=source,
            precomputed_coverage_masks=precomputed_coverage_masks,
        )
        env_seconds = time.perf_counter() - env_started
        return {
            "scenario_build": float(scenario_seconds),
            "coverable_env_init": float(env_seconds),
        }

    def _require_inner(self) -> LunarExplorationEnv:
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise RuntimeError("Standard env has not been reset or restored")
        return inner


def _validate_coverage_cache_manifest_binding(
    manifest: Stage6CoverageManifest,
    catalog: StandardScenarioCatalog,
) -> None:
    expected_split_counts = {
        split: sum(record.split == split for record in catalog.records)
        for split in SPLIT_COUNTS
    }
    if (
        manifest.catalog_sha256 != catalog.sha256
        or dict(manifest.split_counts) != expected_split_counts
    ):
        raise ValueError("coverage cache manifest catalog binding drifted")


class StandardEvaluationEnv(StandardTrainingEnv):
    """按固定 scenario ID lane 顺序执行只读 Standard evaluation。"""

    def __init__(
        self,
        catalog: StandardScenarioCatalog,
        *,
        split: Split,
        scenario_ids: tuple[str, ...],
        safety_contract: SafetyContract,
        config_sha256: str,
    ) -> None:
        super().__init__(
            catalog,
            split=split,
            sampler_seed=0,
            safety_contract=safety_contract,
            config_sha256=config_sha256,
        )
        by_id = {
            record.scenario_id: record
            for record in catalog.records
            if record.split == split
        }
        if (
            not scenario_ids
            or len(scenario_ids) > 8
            or len(set(scenario_ids)) != len(scenario_ids)
            or any(scenario_id not in by_id for scenario_id in scenario_ids)
        ):
            raise ValueError("Standard evaluation scenario lane drifted")
        self._evaluation_records = tuple(by_id[value] for value in scenario_ids)
        self._evaluation_cursor = 0

    def reset(self):
        if self._evaluation_cursor >= len(self._evaluation_records):
            raise RuntimeError("Standard evaluation scenario lane exhausted")
        record = self._evaluation_records[self._evaluation_cursor]
        self._evaluation_cursor += 1
        timing = self._bind_record(record)
        started = time.perf_counter()
        observation = self._require_inner().reset()
        timing["reset"] = time.perf_counter() - started
        self._last_timing_seconds = timing
        return observation


@lru_cache(maxsize=2)
def build_standard_catalog(*, verify_hashes: bool = True) -> StandardScenarioCatalog:
    catalog = StandardScenarioCatalogBuilder(verify_hashes=verify_hashes).build()
    _validate_production_catalog(catalog)
    return catalog


def standard_env_specs(
    catalog: StandardScenarioCatalog,
    *,
    split: Split,
    sampler_seeds: tuple[int, ...],
    safety_contract: SafetyContract,
    config_sha256: str,
    initial_sampler_states: Sequence[Mapping[str, object]] | None = None,
) -> tuple[SpawnEnvSpec, ...]:
    _validate_production_catalog(catalog)
    if len(sampler_seeds) != 8 or len(set(sampler_seeds)) != 8:
        raise ValueError("Standard vector env requires eight distinct sampler seeds")
    if initial_sampler_states is not None and (
        len(initial_sampler_states) != 8
        or any(not isinstance(state, Mapping) for state in initial_sampler_states)
    ):
        raise ValueError("Standard vector env requires eight sampler states")
    payload = catalog.to_dict()
    binding = _validated_typed_safety_contract(
        safety_contract,
        config_sha256=config_sha256,
    ).binding(config_sha256=config_sha256)
    return tuple(
        SpawnEnvSpec(
            factory=build_standard_training_env,
            kwargs={
                "catalog_payload": payload,
                "split": split,
                "sampler_seed": int(seed),
                "initial_sampler_state": (
                    None
                    if initial_sampler_states is None
                    else dict(initial_sampler_states[index])
                ),
                "production": True,
                **binding,
            },
        )
        for index, seed in enumerate(sampler_seeds)
    )


def build_standard_training_env(
    *,
    catalog_payload: Mapping[str, object],
    split: str,
    sampler_seed: int,
    initial_sampler_state: Mapping[str, object] | None = None,
    production: bool,
    safety_contract: Mapping[str, object],
    safety_contract_sha256: str,
    safety_contract_source: str,
    safety_contract_config_sha256: str,
) -> StandardTrainingEnv:
    """Spawn worker top-level factory；拒绝 fixture/fake production adapter。"""

    if production is not True:
        raise ValueError("Stage 6 production env rejects fixture/fake adapter")
    if split not in SPLIT_COUNTS:
        raise ValueError("Stage 6 production env split is invalid")
    catalog = _catalog_from_payload(catalog_payload)
    _validate_production_catalog(catalog)
    contract = SafetyContract.from_binding(
        {
            "safety_contract": safety_contract,
            "safety_contract_sha256": safety_contract_sha256,
            "safety_contract_source": safety_contract_source,
            "safety_contract_config_sha256": safety_contract_config_sha256,
        },
        expected_config_sha256=safety_contract_config_sha256,
    )
    return StandardTrainingEnv(
        catalog,
        split=split,
        sampler_seed=int(sampler_seed),
        safety_contract=contract,
        config_sha256=safety_contract_config_sha256,
        initial_sampler_state=initial_sampler_state,
    )  # type: ignore[arg-type]


def build_standard_evaluation_env(
    *,
    catalog_payload: Mapping[str, object],
    split: str,
    scenario_ids: tuple[str, ...] | list[str],
    production: bool,
    safety_contract: Mapping[str, object],
    safety_contract_sha256: str,
    safety_contract_source: str,
    safety_contract_config_sha256: str,
) -> StandardEvaluationEnv:
    """Spawn worker top-level Standard evaluation factory。"""

    if production is not True or split not in {"validation", "test", "unseen"}:
        raise ValueError("Stage 6 production evaluation env binding is invalid")
    if not isinstance(scenario_ids, (tuple, list)) or any(
        not isinstance(value, str) for value in scenario_ids
    ):
        raise ValueError("Stage 6 evaluation scenario IDs are invalid")
    catalog = _catalog_from_payload(catalog_payload)
    _validate_production_catalog(catalog)
    contract = SafetyContract.from_binding(
        {
            "safety_contract": safety_contract,
            "safety_contract_sha256": safety_contract_sha256,
            "safety_contract_source": safety_contract_source,
            "safety_contract_config_sha256": safety_contract_config_sha256,
        },
        expected_config_sha256=safety_contract_config_sha256,
    )
    return StandardEvaluationEnv(
        catalog,
        split=split,  # type: ignore[arg-type]
        scenario_ids=tuple(scenario_ids),
        safety_contract=contract,
        config_sha256=safety_contract_config_sha256,
    )


def _validated_typed_safety_contract(
    value: object,
    *,
    config_sha256: str,
) -> SafetyContract:
    if not isinstance(value, SafetyContract):
        raise ValueError("Standard environment requires SafetyContract")
    return SafetyContract.from_binding(
        value.binding(config_sha256=config_sha256),
        expected_config_sha256=config_sha256,
    )


def _catalog_from_payload(payload: Mapping[str, object]) -> StandardScenarioCatalog:
    if set(payload) != {
        "schema_version",
        "split_policy",
        "catalog_sha256",
        "sources",
        "records",
    }:
        raise ValueError("Standard catalog payload schema drift")
    if payload["schema_version"] != "standard_unseen_scenario_catalog/v1":
        raise ValueError("Standard catalog payload version drift")
    sources_value = payload["sources"]
    records_value = payload["records"]
    if not isinstance(sources_value, Mapping) or not isinstance(records_value, list):
        raise ValueError("Standard catalog payload types drifted")
    dem = _raster_source_from_payload(sources_value.get("dem"))
    slope = _raster_source_from_payload(sources_value.get("slope"))
    sources = FrozenRasterSources(
        dem=dem,
        slope=slope,
        slope_product_semantics=str(sources_value.get("slope_product_semantics")),
        physical_slope_source=str(sources_value.get("physical_slope_source")),
        slope_pixels_used_for_traversability=sources_value.get(
            "slope_pixels_used_for_traversability"
        )
        is True,
    )
    records: list[ScenarioCatalogRecord] = []
    for item in records_value:
        if not isinstance(item, Mapping):
            raise ValueError("Standard catalog record payload drift")
        value = dict(item)
        value["parent_window"] = RasterWindow(**dict(value["parent_window"]))
        value["source_window"] = RasterWindow(**dict(value["source_window"]))
        value["source_window_transform"] = tuple(value["source_window_transform"])
        value["source_window_bounds"] = tuple(value["source_window_bounds"])
        records.append(ScenarioCatalogRecord(**value))  # type: ignore[arg-type]
    catalog = StandardScenarioCatalog(
        records=tuple(records),
        sources=sources,
        split_policy=str(payload["split_policy"]),
    )
    if catalog.sha256 != payload["catalog_sha256"]:
        raise ValueError("Standard catalog payload hash drift")
    return catalog


def _raster_source_from_payload(value: object) -> RasterSourceProvenance:
    if not isinstance(value, Mapping):
        raise ValueError("Standard raster source payload drift")
    payload = dict(value)
    payload["dtypes"] = tuple(payload["dtypes"])
    payload["colorinterp"] = tuple(payload["colorinterp"])
    payload["transform"] = tuple(payload["transform"])
    return RasterSourceProvenance(**payload)  # type: ignore[arg-type]


def _validate_production_catalog(catalog: StandardScenarioCatalog) -> None:
    counts = {split: 0 for split in SPLIT_COUNTS}
    for record in catalog.records:
        counts[record.split] += 1
        if (
            record.highres_truth_source != "procedural_lunar_rock_crater_proxy/v1"
            or record.synthetic_source_kind != "synthetic_terrain_obstacle_proxy/v1"
            or record.physical_obstacle_cells_written is not False
        ):
            raise ValueError("Standard catalog proxy truth labels drifted")
    if counts != SPLIT_COUNTS or catalog.spatial_audit()["parent_cross_split_count"] != 0:
        raise ValueError("Standard catalog split isolation drifted")
    if catalog.spatial_audit()["child_overlap_pair_count"] != 0:
        raise ValueError("Standard catalog child overlap drifted")
    expected_sources = (
        (catalog.sources.dem, DEM_PATH, DEM_SIZE_BYTES, DEM_SHA256),
        (catalog.sources.slope, SLOPE_PATH, SLOPE_SIZE_BYTES, SLOPE_SHA256),
    )
    if any(
        source.path != path
        or source.size_bytes != size
        or source.sha256 != digest
        for source, path, size, digest in expected_sources
    ):
        raise ValueError("Standard catalog real 4m source binding drifted")
    if (
        catalog.sources.slope_product_semantics
        != "rgb_visualization_provenance_only/v1"
        or catalog.sources.physical_slope_source
        != "dem_float32_metric_gradient_4m/v1"
        or catalog.sources.slope_pixels_used_for_traversability is not False
    ):
        raise ValueError("Standard catalog slope semantics drifted")


__all__ = [
    "SPLIT_COUNTS",
    "StandardEnvSettings",
    "StandardScenarioSampler",
    "StandardEvaluationEnv",
    "StandardTrainingEnv",
    "build_standard_catalog",
    "build_standard_evaluation_env",
    "build_standard_training_env",
    "standard_env_specs",
]
