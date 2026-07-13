"""Deterministic Standard/Unseen catalog from frozen USGS LRO raster sources."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, Mapping

import numpy as np
import rasterio
from rasterio.windows import Window

from lunar_exploration_ppo.configs.schema import (
    DEM_PATH,
    DEM_SHA256,
    DEM_SIZE_BYTES,
    SLOPE_PATH,
    SLOPE_SHA256,
    SLOPE_SIZE_BYTES,
)
from lunar_exploration_ppo.env.scenario import LowResolutionPrior, ScenarioBundle, TruthMap
from lunar_exploration_ppo.env.terrain_proxy import (
    ProceduralTerrainProxyGenerator,
    TerrainProxySettings,
)
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry, PoseXYTheta


SPLIT_ORDER: Final = ("train", "validation", "test", "unseen")
SPLIT_POLICY: Final = "fixed_seed_disjoint_split/v1"
PROXY_GENERATOR_VERSION: Final = "procedural_lunar_rock_crater_proxy/v1"
SEED_DERIVATION_VERSION: Final = "post_parent_split_canonical_sha256_128bit/v1"
_SPLIT_COUNTS: Final = {"train": 700, "validation": 150, "test": 150, "unseen": 64}
_PARENT_COUNTS: Final = {"train": 44, "validation": 10, "test": 10, "unseen": 4}


@dataclass(frozen=True, slots=True)
class RasterWindow:
    col_off: int
    row_off: int
    width: int
    height: int

    def as_rasterio(self) -> Window:
        return Window(self.col_off, self.row_off, self.width, self.height)


@dataclass(frozen=True, slots=True)
class RasterSourceProvenance:
    path: str
    size_bytes: int
    sha256: str
    width: int
    height: int
    count: int
    dtypes: tuple[str, ...]
    colorinterp: tuple[str, ...]
    transform: tuple[float, ...]
    crs_wkt: str | None
    nodata: float | None
    tags: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", MappingProxyType(dict(self.tags)))


@dataclass(frozen=True, slots=True)
class FrozenRasterSources:
    dem: RasterSourceProvenance
    slope: RasterSourceProvenance
    slope_product_semantics: str = "rgb_visualization_provenance_only/v1"
    physical_slope_source: str = "dem_float32_metric_gradient_4m/v1"
    slope_pixels_used_for_traversability: bool = False


@dataclass(frozen=True, slots=True)
class ScenarioCatalogRecord:
    scenario_id: str
    split: Literal["train", "validation", "test", "unseen"]
    density_profile: Literal["low", "medium", "high"]
    parent_roi: str
    parent_window: RasterWindow
    source_window: RasterWindow
    source_window_transform: tuple[float, ...]
    source_window_bounds: tuple[float, float, float, float]
    scenario_seed_hex: str
    start_pose_seed_hex: str
    terrain_seed_hex: str
    proxy_seed_hex: str
    seed_derivation_version: str
    seed_derivation_input: str
    real_lowres_prior_source: str = "usgs_lro_dem_native_4m/v1"
    highres_truth_source: str = PROXY_GENERATOR_VERSION
    observed_state_source: str = "sensor_revealed_highres_state/v1"
    synthetic_source_kind: str = "synthetic_terrain_obstacle_proxy/v1"
    physical_obstacle_cells_written: bool = False


class StandardScenarioCatalog:
    def __init__(
        self,
        *,
        records: tuple[ScenarioCatalogRecord, ...],
        sources: FrozenRasterSources,
        split_policy: str,
    ) -> None:
        self.records = records
        self.sources = sources
        self.split_policy = split_policy
        payload = {
            "schema_version": "standard_unseen_scenario_catalog/v1",
            "split_policy": split_policy,
            "sources": _sources_dict(sources),
            "records": [asdict(record) for record in records],
        }
        self.sha256 = hashlib.sha256(ArtifactStore.canonical_json_bytes(payload)).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "standard_unseen_scenario_catalog/v1",
            "split_policy": self.split_policy,
            "catalog_sha256": self.sha256,
            "sources": _sources_dict(self.sources),
            "records": [asdict(record) for record in self.records],
        }

    def spatial_audit(self) -> dict[str, int | str]:
        parent_splits: dict[str, set[str]] = defaultdict(set)
        for record in self.records:
            parent_splits[record.parent_roi].add(record.split)
        overlap_pairs = 0
        ordered = sorted(self.records, key=lambda record: (record.source_window.row_off, record.source_window.col_off))
        for index, first in enumerate(ordered):
            first_window = first.source_window
            for second in ordered[index + 1 :]:
                second_window = second.source_window
                if second_window.row_off >= first_window.row_off + first_window.height:
                    break
                if _windows_overlap(first_window, second_window):
                    overlap_pairs += 1
        return {
            "schema_version": "catalog_spatial_leakage_audit/v1",
            "parent_cross_split_count": sum(len(splits) > 1 for splits in parent_splits.values()),
            "child_overlap_pair_count": overlap_pairs,
        }

    def read_standard_prior(self, record: ScenarioCatalogRecord) -> LowResolutionPrior:
        height = self._read_dem_window(record)
        return _prior_from_dem(
            height,
            spacing_m=4.0,
            provenance=self._window_provenance(record, profile="Standard v1"),
            aggregation_source="native_4m_dem_window/v1",
        )

    def read_aggregated_8m_prior(self, record: ScenarioCatalogRecord) -> LowResolutionPrior:
        height = aggregate_dem_2x2(self._read_dem_window(record, source_window=record.parent_window))
        provenance = self._window_provenance(
            record,
            profile="Kilometer v1 metadata sample",
            source_window=record.parent_window,
            derived_child_window=False,
        )
        provenance["standard_child_source_window"] = asdict(record.source_window)
        provenance["aggregation_source"] = "native_4m_dem_2x2_mean/v1"
        return _prior_from_dem(
            height,
            spacing_m=8.0,
            provenance=provenance,
            aggregation_source="native_4m_dem_2x2_mean/v1",
        )

    def _read_dem_window(
        self,
        record: ScenarioCatalogRecord,
        *,
        source_window: RasterWindow | None = None,
    ) -> np.ndarray:
        window = record.source_window if source_window is None else source_window
        with rasterio.open(self.sources.dem.path) as dataset:
            data = dataset.read(1, window=window.as_rasterio(), masked=True)
        if np.any(np.ma.getmaskarray(data)):
            raise ValueError(f"DEM source window contains nodata: {record.scenario_id}")
        height = np.flipud(np.asarray(data, dtype=np.float64))
        if not np.isfinite(height).all():
            raise ValueError(f"DEM source window contains non-finite values: {record.scenario_id}")
        return height

    def _window_provenance(
        self,
        record: ScenarioCatalogRecord,
        *,
        profile: str,
        source_window: RasterWindow | None = None,
        derived_child_window: bool = True,
    ) -> dict[str, object]:
        window = record.source_window if source_window is None else source_window
        source_transform = rasterio.Affine(*self.sources.dem.transform[:6])
        transform = rasterio.windows.transform(window.as_rasterio(), source_transform)
        bounds = rasterio.windows.bounds(window.as_rasterio(), source_transform)
        return {
            "schema_version": "real_lowres_prior_provenance/v1",
            "scale_profile": profile,
            "dem_source": _raster_source_dict(self.sources.dem),
            "slope_product_source": _raster_source_dict(self.sources.slope),
            "slope_product_semantics": self.sources.slope_product_semantics,
            "slope_product_pixels_used": False,
            "source_window": asdict(window),
            "parent_roi": record.parent_roi,
            "parent_window": asdict(record.parent_window),
            "derived_child_window": derived_child_window,
            "source_window_transform": [float(value) for value in transform],
            "source_window_bounds": [float(value) for value in bounds],
            "array_y_axis": "south_to_north_increasing/v1",
        }


class StandardScenarioFactory:
    """Construct deterministic Standard proxy truth from one bound catalog record."""

    def __init__(self, catalog: StandardScenarioCatalog) -> None:
        if not isinstance(catalog, StandardScenarioCatalog):
            raise TypeError("catalog must be a StandardScenarioCatalog")
        self.catalog = catalog

    def build(self, record: ScenarioCatalogRecord) -> ScenarioBundle:
        if record not in self.catalog.records:
            raise ValueError("Standard scenario record is not bound to this catalog")
        prior = self.catalog.read_standard_prior(record)
        native_height = self.catalog._read_dem_window(record)
        base_height = _upsample_standard_dem(native_height)
        geometry = GridGeometry(width=256, height=256, resolution_m=0.5)
        start_pose = _standard_start_pose(record, prior)
        settings = TerrainProxySettings(
            generator_version=PROXY_GENERATOR_VERSION,
            density_profile=record.density_profile,
            base_seed=int(record.proxy_seed_hex, 16),
            start_protection_m=6.0,
            max_scene_attempts=64,
            max_object_attempts=256,
        )
        proxy = ProceduralTerrainProxyGenerator(settings).generate(
            base_height,
            geometry,
            start_pose,
            scenario_key=record.scenario_id,
            split=record.split,
            parent_roi=record.parent_roi,
            seed_hex=record.proxy_seed_hex,
        )
        record_sha256 = hashlib.sha256(
            ArtifactStore.canonical_json_bytes(asdict(record))
        ).hexdigest()
        prior_sha256 = _array_content_sha256(prior.channels, domain="standard_prior/v1")
        provenance = {
            "source": record.highres_truth_source,
            "synthetic_source_kind": record.synthetic_source_kind,
            "physical_obstacle_cells_written": record.physical_obstacle_cells_written,
            "catalog_sha256": self.catalog.sha256,
            "catalog_record_sha256": record_sha256,
            "scenario_id": record.scenario_id,
            "split": record.split,
            "density_profile": record.density_profile,
            "base_height_source": record.real_lowres_prior_source,
            "base_height_upsampling": "bilinear_32x32_to_256x256_cell_center/v1",
            "prior_channels_sha256": prior_sha256,
            "proxy_generator_version": proxy.catalog.generator_version,
            "proxy_seed_hex": proxy.catalog.seed_hex,
            "proxy_catalog_sha256": proxy.catalog.sha256,
            "layer_hashes": dict(proxy.layer_hashes),
        }
        truth = TruthMap(
            geometry=geometry,
            height=proxy.height,
            hard_obstacle=proxy.hard_obstacle,
            slope_deg=proxy.slope_deg,
            traversability=proxy.traversability,
            provenance=provenance,
        )
        scenario_id = f"{record.scenario_id}/standard-proxy/v1"
        scenario_hash = hashlib.sha256(ArtifactStore.canonical_json_bytes({
            "schema_version": "standard_proxy_scenario_identity/v1",
            "scenario_id": scenario_id,
            "catalog_sha256": self.catalog.sha256,
            "catalog_record_sha256": record_sha256,
            "prior_channels_sha256": prior_sha256,
            "start_pose": {
                "x": start_pose.cell.x,
                "y": start_pose.cell.y,
                "theta": start_pose.theta,
            },
            "proxy_catalog_sha256": proxy.catalog.sha256,
            "proxy_layer_hashes": dict(proxy.layer_hashes),
        })).hexdigest()
        return ScenarioBundle(
            scenario_id=scenario_id,
            scenario_hash=scenario_hash,
            truth=truth,
            prior=prior,
            start_pose=start_pose,
            proxy_catalog=proxy.catalog,
            proxy_layer_hashes=proxy.layer_hashes,
        )


class StandardScenarioCatalogBuilder:
    def __init__(
        self,
        *,
        dem_path: str | Path = DEM_PATH,
        slope_path: str | Path = SLOPE_PATH,
        base_seed: int = 20260712,
        verify_hashes: bool = True,
    ) -> None:
        self.dem_path = Path(dem_path)
        self.slope_path = Path(slope_path)
        self.base_seed = int(base_seed)
        self.verify_hashes = verify_hashes
        self._sources: FrozenRasterSources | None = None

    def inspect_sources(self) -> FrozenRasterSources:
        if self._sources is not None:
            return self._sources
        dem = _inspect_raster(
            self.dem_path,
            expected_size=DEM_SIZE_BYTES if self.dem_path.as_posix() == DEM_PATH else None,
            expected_sha256=DEM_SHA256 if self.dem_path.as_posix() == DEM_PATH else None,
            verify_hash=self.verify_hashes,
        )
        slope = _inspect_raster(
            self.slope_path,
            expected_size=SLOPE_SIZE_BYTES if self.slope_path.as_posix() == SLOPE_PATH else None,
            expected_sha256=SLOPE_SHA256 if self.slope_path.as_posix() == SLOPE_PATH else None,
            verify_hash=self.verify_hashes,
        )
        if dem.count != 1 or dem.dtypes != ("float32",):
            raise ValueError("frozen DEM must be one-band float32")
        if slope.count != 3 or slope.dtypes != ("uint8", "uint8", "uint8") or slope.colorinterp != ("red", "green", "blue"):
            raise ValueError("frozen slope product must remain a three-band uint8 RGB visualization")
        if dem.width != slope.width or dem.height != slope.height or dem.transform != slope.transform:
            raise ValueError("DEM and slope visualization grids do not align")
        self._sources = FrozenRasterSources(dem=dem, slope=slope)
        return self._sources

    def build(self) -> StandardScenarioCatalog:
        sources = self.inspect_sources()
        parent_candidates = [
            RasterWindow(column, row, 256, 256)
            for row in range(0, sources.dem.height - 255, 256)
            for column in range(0, sources.dem.width - 255, 256)
        ]
        if any(window.col_off + window.width > sources.dem.width or window.row_off + window.height > sources.dem.height for window in parent_candidates):
            raise ValueError("catalog parent grid is outside the frozen DEM")
        parents: list[RasterWindow] = []
        with rasterio.open(self.dem_path) as dataset:
            for window in parent_candidates:
                mask = dataset.read_masks(1, window=window.as_rasterio())
                if mask.shape == (window.height, window.width) and np.all(mask == 255):
                    parents.append(window)
        required_parent_count = sum(_PARENT_COUNTS.values())
        if len(parents) < required_parent_count:
            raise ValueError("frozen DEM does not contain enough fully valid disjoint parent ROIs")
        permutation = np.random.Generator(np.random.PCG64(self.base_seed)).permutation(len(parents))
        parents = [parents[int(index)] for index in permutation[:required_parent_count]]
        records: list[ScenarioCatalogRecord] = []
        source_transform = rasterio.Affine(*sources.dem.transform[:6])
        parent_cursor = 0
        for split in SPLIT_ORDER:
            split_parents = parents[parent_cursor : parent_cursor + _PARENT_COUNTS[split]]
            parent_cursor += _PARENT_COUNTS[split]
            children: list[tuple[RasterWindow, RasterWindow, str]] = []
            for parent in split_parents:
                parent_roi = f"r{parent.row_off:05d}-c{parent.col_off:05d}"
                for row_delta in (0, 64, 128, 192):
                    for col_delta in (0, 64, 128, 192):
                        child = RasterWindow(parent.col_off + col_delta, parent.row_off + row_delta, 32, 32)
                        children.append((parent, child, parent_roi))
            children = children[: _SPLIT_COUNTS[split]]
            for index, (parent, child, parent_roi) in enumerate(children):
                density = ("low", "medium", "high")[index % 3]
                scenario_id = f"{split}/scenario-{index:04d}"
                seed_input = _seed_input(scenario_id, split, parent_roi, self.base_seed, density)
                transform = rasterio.windows.transform(child.as_rasterio(), source_transform)
                bounds = rasterio.windows.bounds(child.as_rasterio(), source_transform)
                records.append(
                    ScenarioCatalogRecord(
                        scenario_id=scenario_id,
                        split=split,  # type: ignore[arg-type]
                        density_profile=density,  # type: ignore[arg-type]
                        parent_roi=parent_roi,
                        parent_window=parent,
                        source_window=child,
                        source_window_transform=tuple(float(value) for value in transform),
                        source_window_bounds=tuple(float(value) for value in bounds),
                        scenario_seed_hex=derive_catalog_seed(
                            kind="scenario", scenario_key=scenario_id, split=split, parent_roi=parent_roi,
                            base_seed=self.base_seed, density_profile=density,
                        ),
                        start_pose_seed_hex=derive_catalog_seed(
                            kind="start_pose", scenario_key=scenario_id, split=split, parent_roi=parent_roi,
                            base_seed=self.base_seed, density_profile=density,
                        ),
                        terrain_seed_hex=derive_catalog_seed(
                            kind="terrain", scenario_key=scenario_id, split=split, parent_roi=parent_roi,
                            base_seed=self.base_seed, density_profile=density,
                        ),
                        proxy_seed_hex=derive_catalog_seed(
                            kind="proxy", scenario_key=scenario_id, split=split, parent_roi=parent_roi,
                            base_seed=self.base_seed, density_profile=density,
                        ),
                        seed_derivation_version=SEED_DERIVATION_VERSION,
                        seed_derivation_input=seed_input,
                    )
                )
        records.sort(key=lambda record: (SPLIT_ORDER.index(record.split), record.scenario_id))
        catalog = StandardScenarioCatalog(records=tuple(records), sources=sources, split_policy=SPLIT_POLICY)
        audit = catalog.spatial_audit()
        if audit["parent_cross_split_count"] or audit["child_overlap_pair_count"]:
            raise ValueError("catalog spatial isolation failed closed")
        return catalog


def derive_catalog_seed(
    *,
    kind: str,
    scenario_key: str,
    split: str,
    parent_roi: str,
    base_seed: int,
    density_profile: str,
) -> str:
    payload = f"{_seed_input(scenario_key, split, parent_roi, base_seed, density_profile)} | {kind}"
    return hashlib.sha256(payload.encode("utf-8")).digest()[:16].hex()


def _seed_input(scenario_key: str, split: str, parent_roi: str, base_seed: int, density_profile: str) -> str:
    return f"{PROXY_GENERATOR_VERSION} | {scenario_key} | {split} | {parent_roi} | {base_seed} | {density_profile}"


def _upsample_standard_dem(height: np.ndarray) -> np.ndarray:
    values = np.asarray(height, dtype=np.float64)
    if values.shape != (32, 32) or not np.isfinite(values).all():
        raise ValueError("Standard base DEM must be a finite 32x32 window")
    source_axis = np.arange(32, dtype=np.float64)
    target_axis = np.clip((np.arange(256, dtype=np.float64) + 0.5) / 8.0 - 0.5, 0.0, 31.0)
    horizontal = np.empty((32, 256), dtype=np.float64)
    for row in range(32):
        horizontal[row] = np.interp(target_axis, source_axis, values[row])
    result = np.empty((256, 256), dtype=np.float64)
    for column in range(256):
        result[:, column] = np.interp(target_axis, source_axis, horizontal[:, column])
    return result


def _standard_start_pose(
    record: ScenarioCatalogRecord,
    prior: LowResolutionPrior,
) -> PoseXYTheta:
    rng = np.random.Generator(np.random.PCG64(int(record.start_pose_seed_hex, 16)))
    traversability = np.asarray(prior.channels[3], dtype=np.float64)
    candidates = np.argwhere(traversability[2:-2, 2:-2] >= 0.75) + 2
    if not len(candidates):
        candidates = np.argwhere(np.ones((28, 28), dtype=bool)) + 2
    low_y, low_x = candidates[int(rng.integers(0, len(candidates)))]
    cell = CellXY(int(low_x) * 8 + 4, int(low_y) * 8 + 4)
    return PoseXYTheta(cell, float(rng.uniform(-math.pi, math.pi)))


def _array_content_sha256(array: np.ndarray, *, domain: str) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(b"\0")
    for dimension in values.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def derive_physical_slope_deg(height: np.ndarray, *, spacing_m: float) -> np.ndarray:
    values = np.asarray(height, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("DEM height must be a finite 2D array with both dimensions >= 2")
    if not math.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("metric spacing must be finite and positive")
    dz_dy, dz_dx = np.gradient(values, spacing_m, spacing_m, edge_order=1)
    return np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))


def aggregate_dem_2x2(height: np.ndarray) -> np.ndarray:
    values = np.asarray(height, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] % 2 or values.shape[1] % 2:
        raise ValueError("4m DEM aggregation requires an even 2D shape")
    return values.reshape(values.shape[0] // 2, 2, values.shape[1] // 2, 2).mean(axis=(1, 3))


def _prior_from_dem(
    height: np.ndarray,
    *,
    spacing_m: float,
    provenance: dict[str, object],
    aggregation_source: str,
) -> LowResolutionPrior:
    slope = derive_physical_slope_deg(height, spacing_m=spacing_m)
    traversability = np.clip(1.0 - slope / 60.0, 0.0, 1.0)
    obstacle = slope > 30.0
    centered = height - float(np.mean(height))
    scale = max(float(np.std(centered)), 1.0e-6)
    height_prior = np.clip(centered / scale, -5.0, 5.0)
    channels = np.stack(
        (
            height_prior,
            np.ones_like(height),
            obstacle.astype(np.float64),
            traversability,
            np.zeros_like(height),
            np.zeros_like(height),
            np.zeros_like(height),
        ),
        axis=0,
    ).astype(np.float32)
    provenance.update(
        {
            "aggregation_source": aggregation_source,
            "physical_slope_derivation": f"numpy_gradient_metric_spacing_{int(spacing_m)}m/v1",
            "physical_slope_spacing_m": spacing_m,
            "max_traversable_slope_deg": 30.0,
            "traversability_derivation": "clip_1_minus_slope_deg_over_60/v1",
            "value_prior_source": "constant_neutral/v1",
        }
    )
    return LowResolutionPrior(
        channels=channels,
        resolution_m=spacing_m,
        value_prior_source="constant_neutral/v1",
        provenance=provenance,
    )


def _inspect_raster(
    path: Path,
    *,
    expected_size: int | None,
    expected_sha256: str | None,
    verify_hash: bool,
) -> RasterSourceProvenance:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise ValueError(f"frozen raster size drift: {path}")
    if verify_hash or expected_sha256 is None:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        sha256 = digest.hexdigest()
    else:
        sha256 = expected_sha256
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ValueError(f"frozen raster SHA-256 drift: {path}")
    with rasterio.open(path) as dataset:
        return RasterSourceProvenance(
            path=path.as_posix(),
            size_bytes=size,
            sha256=sha256,
            width=dataset.width,
            height=dataset.height,
            count=dataset.count,
            dtypes=tuple(dataset.dtypes),
            colorinterp=tuple(value.name for value in dataset.colorinterp),
            transform=tuple(float(value) for value in dataset.transform),
            crs_wkt=dataset.crs.to_wkt() if dataset.crs is not None else None,
            nodata=float(dataset.nodata) if dataset.nodata is not None else None,
            tags=dataset.tags(),
        )


def _windows_overlap(first: RasterWindow, second: RasterWindow) -> bool:
    return not (
        first.col_off + first.width <= second.col_off
        or second.col_off + second.width <= first.col_off
        or first.row_off + first.height <= second.row_off
        or second.row_off + second.height <= first.row_off
    )


def _sources_dict(sources: FrozenRasterSources) -> dict[str, object]:
    return {
        "dem": _raster_source_dict(sources.dem),
        "slope": _raster_source_dict(sources.slope),
        "slope_product_semantics": sources.slope_product_semantics,
        "physical_slope_source": sources.physical_slope_source,
        "slope_pixels_used_for_traversability": sources.slope_pixels_used_for_traversability,
    }


def _raster_source_dict(source: RasterSourceProvenance) -> dict[str, object]:
    return {
        "path": source.path,
        "size_bytes": source.size_bytes,
        "sha256": source.sha256,
        "width": source.width,
        "height": source.height,
        "count": source.count,
        "dtypes": list(source.dtypes),
        "colorinterp": list(source.colorinterp),
        "transform": list(source.transform),
        "crs_wkt": source.crs_wkt,
        "nodata": source.nodata,
        "tags": dict(source.tags),
    }
