from __future__ import annotations

from collections import Counter, defaultdict
import importlib
from pathlib import Path

import numpy as np
import pytest

from lunar_exploration_ppo.configs.schema import DEM_PATH, SLOPE_PATH


def _catalog_module():
    try:
        return importlib.import_module("lunar_exploration_ppo.env.scenario_catalog")
    except ModuleNotFoundError:
        pytest.fail("Stage 2 scenario catalog module is missing")


def test_physical_slope_is_derived_from_dem_with_metric_spacing() -> None:
    catalog_module = _catalog_module()
    xx = np.arange(6, dtype=np.float64)[None, :]
    height = np.repeat(xx * 4.0, 5, axis=0)
    slope = catalog_module.derive_physical_slope_deg(height, spacing_m=4.0)
    assert np.allclose(slope, 45.0, atol=1e-10)
    aggregated = catalog_module.aggregate_dem_2x2(height[:4, :6])
    assert aggregated.shape == (2, 3)
    slope_8m = catalog_module.derive_physical_slope_deg(aggregated, spacing_m=8.0)
    assert np.allclose(slope_8m, 45.0, atol=1e-10)


def test_real_rasterio_provenance_preserves_affine_crs_and_rgb_slope_semantics() -> None:
    catalog_module = _catalog_module()
    builder = catalog_module.StandardScenarioCatalogBuilder(
        dem_path=DEM_PATH,
        slope_path=SLOPE_PATH,
        verify_hashes=True,
    )
    sources = builder.inspect_sources()
    assert sources.dem.count == 1
    assert sources.dem.dtypes == ("float32",)
    assert sources.dem.transform[:6] == (4.0, 0.0, -33386.0, 0.0, -4.0, 27978.0)
    assert sources.dem.crs_wkt and "Moon" in sources.dem.crs_wkt
    assert sources.slope.count == 3
    assert sources.slope.dtypes == ("uint8", "uint8", "uint8")
    assert sources.slope.colorinterp == ("red", "green", "blue")
    assert sources.slope_product_semantics == "rgb_visualization_provenance_only/v1"
    assert sources.physical_slope_source == "dem_float32_metric_gradient_4m/v1"
    assert sources.slope_pixels_used_for_traversability is False


def test_standard_catalog_counts_density_parent_and_child_spatial_isolation() -> None:
    catalog_module = _catalog_module()
    catalog = catalog_module.StandardScenarioCatalogBuilder(
        dem_path=DEM_PATH,
        slope_path=SLOPE_PATH,
        verify_hashes=False,
    ).build()
    counts = Counter(record.split for record in catalog.records)
    assert counts == {"train": 700, "validation": 150, "test": 150, "unseen": 64}
    density = defaultdict(Counter)
    parent_splits: dict[str, set[str]] = defaultdict(set)
    for record in catalog.records:
        density[record.split][record.density_profile] += 1
        parent_splits[record.parent_roi].add(record.split)
        assert record.synthetic_source_kind == "synthetic_terrain_obstacle_proxy/v1"
        assert record.physical_obstacle_cells_written is False
        assert record.real_lowres_prior_source == "usgs_lro_dem_native_4m/v1"
        assert record.highres_truth_source == "procedural_lunar_rock_crater_proxy/v1"
        assert record.observed_state_source == "sensor_revealed_highres_state/v1"
    assert sorted(density["train"].values()) == [233, 233, 234]
    for split in ("validation", "test", "unseen"):
        values = density[split].values()
        assert max(values) - min(values) <= 1
    assert all(len(splits) == 1 for splits in parent_splits.values())
    audit = catalog.spatial_audit()
    assert audit["parent_cross_split_count"] == 0
    assert audit["child_overlap_pair_count"] == 0


def test_catalog_is_deterministic_and_proxy_seeds_bind_assigned_split() -> None:
    catalog_module = _catalog_module()
    builder = catalog_module.StandardScenarioCatalogBuilder(
        dem_path=DEM_PATH,
        slope_path=SLOPE_PATH,
        verify_hashes=False,
    )
    first = builder.build()
    second = builder.build()
    assert first.sha256 == second.sha256
    assert [record.scenario_id for record in first.records] == sorted(
        (record.scenario_id for record in first.records),
        key=lambda value: (catalog_module.SPLIT_ORDER.index(value.split("/")[0]), value),
    )
    record = first.records[0]
    assert record.seed_derivation_version == "post_parent_split_canonical_sha256_128bit/v1"
    assert record.split in record.seed_derivation_input
    changed = catalog_module.derive_catalog_seed(
        kind="proxy", scenario_key=record.scenario_id, split="test",
        parent_roi=record.parent_roi, base_seed=builder.base_seed,
        density_profile=record.density_profile,
    )
    assert changed != record.proxy_seed_hex


def test_native_4m_prior_and_kilometer_8m_have_distinct_frozen_extents() -> None:
    catalog_module = _catalog_module()
    catalog = catalog_module.StandardScenarioCatalogBuilder(
        dem_path=DEM_PATH,
        slope_path=SLOPE_PATH,
        verify_hashes=False,
    ).build()
    record = catalog.records[0]
    prior = catalog.read_standard_prior(record)
    assert prior.channels.shape == (7, 32, 32)
    assert prior.resolution_m == 4.0
    assert prior.provenance["array_y_axis"] == "south_to_north_increasing/v1"
    assert prior.provenance["physical_slope_derivation"] == "numpy_gradient_metric_spacing_4m/v1"
    assert prior.provenance["slope_product_pixels_used"] is False
    assert np.isfinite(prior.channels).all()
    kilometer = catalog.read_aggregated_8m_prior(record)
    assert kilometer.channels.shape == (7, 128, 128)
    assert kilometer.resolution_m == 8.0
    assert kilometer.channels.shape[-1] * kilometer.resolution_m == 1024.0
    assert kilometer.provenance["aggregation_source"] == "native_4m_dem_2x2_mean/v1"
    assert kilometer.provenance["physical_slope_derivation"] == "numpy_gradient_metric_spacing_8m/v1"
    assert record.parent_window.width == record.parent_window.height == 256
    assert record.source_window.width == record.source_window.height == 32
    assert prior.provenance["source_window"] != kilometer.provenance["source_window"]
    assert kilometer.provenance["source_window"] == kilometer.provenance["parent_window"]
    assert kilometer.provenance["derived_child_window"] is False


def test_standard_and_kilometer_value_prior_is_constant_neutral() -> None:
    catalog_module = _catalog_module()
    catalog = catalog_module.StandardScenarioCatalogBuilder(
        dem_path=DEM_PATH,
        slope_path=SLOPE_PATH,
        verify_hashes=False,
    ).build()
    record = catalog.records[0]

    for prior in (
        catalog.read_standard_prior(record),
        catalog.read_aggregated_8m_prior(record),
    ):
        assert np.array_equal(prior.channels[1], np.ones_like(prior.channels[1]))
        assert prior.value_prior_source == "constant_neutral/v1"
        assert prior.provenance["value_prior_source"] == "constant_neutral/v1"


def test_catalog_record_builds_deterministic_standard_proxy_truth_and_sensor_state() -> None:
    catalog_module = _catalog_module()
    catalog = catalog_module.StandardScenarioCatalogBuilder(
        dem_path=DEM_PATH,
        slope_path=SLOPE_PATH,
        verify_hashes=False,
    ).build()
    record = catalog.records[0]
    factory = catalog_module.StandardScenarioFactory(catalog)

    first = factory.build(record)
    second = factory.build(record)
    assert first.scenario_hash == second.scenario_hash
    assert first.truth.geometry.shape == (256, 256)
    assert first.truth.geometry.resolution_m == 0.5
    assert first.proxy_catalog.seed_hex == record.proxy_seed_hex
    assert first.proxy_catalog.sha256 == second.proxy_catalog.sha256
    assert dict(first.proxy_layer_hashes) == dict(second.proxy_layer_hashes)
    assert first.truth.provenance["synthetic_source_kind"] == "synthetic_terrain_obstacle_proxy/v1"
    assert first.truth.provenance["physical_obstacle_cells_written"] is False

    from lunar_exploration_ppo.env.map_state import ObservedMapState
    from lunar_exploration_ppo.env.sensor_model import SensorPose, SensorUpdater

    observed = ObservedMapState.empty(first.truth.geometry)
    reset_pose = SensorPose(
        first.truth.geometry.cell_to_world_center(first.start_pose.cell),
        first.start_pose.theta,
        "reset",
    )
    delta = SensorUpdater().reveal(first.truth, observed, (reset_pose,))
    assert delta.newly_observed_count > 0
    assert np.array_equal(observed.height[observed.observed_mask], first.truth.height[observed.observed_mask])
    assert np.array_equal(
        observed.obstacle[observed.observed_mask],
        first.truth.hard_obstacle[observed.observed_mask],
    )
