from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import FrozenInstanceError, asdict, replace
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from lunar_exploration_ppo.configs.stage1 import load_stage1_config
from lunar_exploration_ppo.env.terrain_proxy import (
    DENSITY_COUNTS,
    GENERATOR_VERSION,
    CraterProxy,
    ProceduralTerrainProxyGenerator,
    RockProxy,
    TerrainProxyBundle,
    TerrainProxyCatalog,
    TerrainProxyGenerationError,
    TerrainProxySettings,
    crater_height_at,
    derive_proxy_seed,
    rock_height_at,
    rock_is_core,
    scaled_count_bounds,
)
from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry, PoseXYTheta
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE1_CONFIG = REPO_ROOT / "configs" / "ppo_highres_frontier_smoke_v1.json"


def _expected_layer_content_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    dtype_bytes = contiguous.dtype.str.encode("ascii")
    digest = hashlib.sha256()
    digest.update(b"terrain_proxy_layer_content/v1\0")
    digest.update(len(dtype_bytes).to_bytes(8, "big"))
    digest.update(dtype_bytes)
    digest.update(contiguous.ndim.to_bytes(8, "big"))
    for dimension in contiguous.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def test_stage1_config_freezes_medium_rock_crater_proxy() -> None:
    config = load_stage1_config(STAGE1_CONFIG)

    assert config.proxy_generator_version == "procedural_lunar_rock_crater_proxy/v1"
    assert config.proxy_density_profile == "medium"
    assert config.proxy_base_seed == 20260710
    assert config.proxy_start_protection_m == 6.0
    assert config.proxy_max_scene_attempts == 64
    assert config.proxy_max_object_attempts == 256


@pytest.mark.parametrize(
    ("profile", "rock", "crater"),
    [
        ("low", (10, 18), (2, 3)),
        ("medium", (24, 36), (4, 6)),
        ("high", (45, 64), (7, 10)),
    ],
)
def test_density_bounds_at_smoke_area(
    profile: str,
    rock: tuple[int, int],
    crater: tuple[int, int],
) -> None:
    assert scaled_count_bounds(profile, "rock", 4096.0) == rock
    assert scaled_count_bounds(profile, "crater", 4096.0) == crater


def test_proxy_seed_uses_canonical_utf8_sha256_and_changes_with_identity() -> None:
    source = "|".join(
        (
            "procedural_lunar_rock_crater_proxy/v1",
            "smoke-v1",
            "stage1",
            "smoke-parent-roi",
            "20260710",
            "medium",
        )
    )
    expected = hashlib.sha256(source.encode("utf-8")).hexdigest()[:32]

    actual = derive_proxy_seed(
        scenario_key="smoke-v1",
        split="stage1",
        parent_roi="smoke-parent-roi",
        base_seed=20260710,
        density_profile="medium",
    )
    changed = derive_proxy_seed(
        scenario_key="smoke-v1-changed",
        split="stage1",
        parent_roi="smoke-parent-roi",
        base_seed=20260710,
        density_profile="medium",
    )

    assert actual == expected
    assert len(actual) == 32
    assert changed != actual


def test_proxy_internal_types_are_frozen_and_validate_finite_ranges() -> None:
    settings = TerrainProxySettings(
        generator_version=GENERATOR_VERSION,
        density_profile="medium",
        base_seed=20260710,
        start_protection_m=6.0,
        max_scene_attempts=64,
        max_object_attempts=256,
    )
    rock = RockProxy(10.0, 12.0, 1.0, 1.2, 1.0 / 1.2, 0.0, 0.8)
    crater = CraterProxy(20.0, 20.0, 4.0, 0.8, 0.32, 0.2)

    with pytest.raises(FrozenInstanceError):
        settings.base_seed = 1
    with pytest.raises(ValueError, match="finite"):
        RockProxy(10.0, 12.0, 1.0, 1.2, 1.0 / 1.2, float("nan"), 0.8)
    with pytest.raises(ValueError, match="depth"):
        CraterProxy(20.0, 20.0, 4.0, 0.1, 0.32, 0.2)
    with pytest.raises(ValueError, match="attempt"):
        TerrainProxySettings(GENERATOR_VERSION, "medium", 20260710, 6.0, 0, 256)

    assert rock.semi_major_m * rock.semi_minor_m == pytest.approx(
        rock.equivalent_radius_m**2
    )
    assert crater.rim_width_ratio == pytest.approx(0.2)
    assert issubclass(TerrainProxyGenerationError, RuntimeError)


def test_proxy_bundle_freezes_layers_and_copies_hash_mapping() -> None:
    generated = _generated_proxy("medium")
    layer_hashes = dict(generated.layer_hashes)
    bundle = TerrainProxyBundle(
        height=generated.height.copy(),
        hard_obstacle=generated.hard_obstacle.copy(),
        slope_deg=generated.slope_deg.copy(),
        traversability=generated.traversability.copy(),
        catalog=generated.catalog,
        layer_hashes=layer_hashes,
    )

    assert all(
        not layer.flags.writeable
        for layer in (
            bundle.height,
            bundle.hard_obstacle,
            bundle.slope_deg,
            bundle.traversability,
        )
    )
    original_height_hash = bundle.layer_hashes["height"]
    layer_hashes["height"] = "f" * 64
    assert bundle.layer_hashes["height"] == original_height_hash
    with pytest.raises(TypeError):
        bundle.layer_hashes["height"] = "0" * 64


def test_catalog_and_bundle_recompute_content_hashes_and_reject_forgery() -> None:
    generated = _generated_proxy("medium")

    with pytest.raises(ValueError, match="catalog.*hash|sha256.*content"):
        replace(generated.catalog, sha256="0" * 64)

    forged_layer_hashes = dict(generated.layer_hashes)
    forged_layer_hashes["height"] = "0" * 64
    with pytest.raises(ValueError, match="layer.*hash|height.*hash"):
        TerrainProxyBundle(
            height=generated.height.copy(),
            hard_obstacle=generated.hard_obstacle.copy(),
            slope_deg=generated.slope_deg.copy(),
            traversability=generated.traversability.copy(),
            catalog=generated.catalog,
            layer_hashes=forged_layer_hashes,
        )


def test_density_counts_are_runtime_immutable_at_both_mapping_levels() -> None:
    original_medium = DENSITY_COUNTS["medium"]
    original_rock = original_medium["rock"]
    try:
        with pytest.raises(TypeError):
            DENSITY_COUNTS["medium"] = {"rock": (1, 1), "crater": (1, 1)}
    finally:
        if isinstance(DENSITY_COUNTS, dict):
            DENSITY_COUNTS["medium"] = original_medium
    try:
        with pytest.raises(TypeError):
            DENSITY_COUNTS["medium"]["rock"] = (1, 1)
    finally:
        if isinstance(DENSITY_COUNTS["medium"], dict):
            DENSITY_COUNTS["medium"]["rock"] = original_rock


@pytest.mark.parametrize("area_m2", [0.0, -1.0, float("nan"), float("inf")])
def test_density_bounds_reject_nonpositive_or_nonfinite_area(area_m2: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        scaled_count_bounds("medium", "rock", area_m2)


def test_analytic_rock_profile_and_core() -> None:
    rock = RockProxy(10.0, 12.0, 1.0, 1.2, 1.0 / 1.2, 0.0, 0.8)

    assert rock_height_at(rock, 10.0, 12.0) == pytest.approx(0.8)
    assert rock_height_at(rock, 11.2, 12.0) == pytest.approx(0.0)
    assert rock_is_core(rock, 10.0, 12.0)
    assert not rock_is_core(rock, 11.1, 12.0)


def test_analytic_crater_has_bowl_rim_and_no_hard_obstacle_semantics() -> None:
    crater = CraterProxy(20.0, 20.0, 4.0, 0.8, 0.32, 0.2)

    center = crater_height_at(crater, 0.0)
    rim = crater_height_at(crater, 4.0)
    outside = crater_height_at(crater, 5.6)

    assert center < 0.0
    assert rim > 0.0
    assert outside == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("x_m", "y_m"),
    [
        (float("nan"), 12.0),
        (float("inf"), 12.0),
        (10.0, float("-inf")),
    ],
)
def test_rock_helpers_reject_nonfinite_world_coordinates(x_m: float, y_m: float) -> None:
    rock = RockProxy(10.0, 12.0, 1.0, 1.2, 1.0 / 1.2, 0.0, 0.8)

    with pytest.raises(ValueError, match="world coordinates must be finite"):
        rock_height_at(rock, x_m, y_m)
    with pytest.raises(ValueError, match="world coordinates must be finite"):
        rock_is_core(rock, x_m, y_m)


@pytest.mark.parametrize("distance_m", [-1.0, float("nan"), float("inf")])
def test_crater_helper_rejects_negative_or_nonfinite_distance(distance_m: float) -> None:
    crater = CraterProxy(20.0, 20.0, 4.0, 0.8, 0.32, 0.2)

    with pytest.raises(ValueError, match="distance must be finite and nonnegative"):
        crater_height_at(crater, distance_m)


def _proxy_settings(
    density_profile: str,
    *,
    base_seed: int = 20260710,
    max_scene_attempts: int = 64,
    max_object_attempts: int = 256,
) -> TerrainProxySettings:
    return TerrainProxySettings(
        generator_version=GENERATOR_VERSION,
        density_profile=density_profile,
        base_seed=base_seed,
        start_protection_m=6.0,
        max_scene_attempts=max_scene_attempts,
        max_object_attempts=max_object_attempts,
    )


@lru_cache(maxsize=None)
def _generated_proxy(
    density_profile: str,
    base_seed: int = 20260710,
) -> TerrainProxyBundle:
    geometry = GridGeometry(width=128, height=128, resolution_m=0.5)
    return ProceduralTerrainProxyGenerator(
        _proxy_settings(density_profile, base_seed=base_seed)
    ).generate(
        np.zeros(geometry.shape, dtype=np.float64),
        geometry,
        PoseXYTheta(CellXY(16, 64), 0.0),
        scenario_key="smoke-v1",
        split="stage1",
        parent_roi="smoke-parent-roi",
    )


@pytest.mark.parametrize(
    ("density_profile", "rock_bounds", "crater_bounds"),
    [
        ("low", (10, 18), (2, 3)),
        ("medium", (24, 36), (4, 6)),
        ("high", (45, 64), (7, 10)),
    ],
)
def test_generator_counts_are_in_frozen_density_ranges(
    density_profile: str,
    rock_bounds: tuple[int, int],
    crater_bounds: tuple[int, int],
) -> None:
    catalog = _generated_proxy(density_profile).catalog

    assert rock_bounds[0] <= len(catalog.rocks) <= rock_bounds[1]
    assert crater_bounds[0] <= len(catalog.craters) <= crater_bounds[1]
    assert catalog.density_profile == density_profile
    assert 0 <= catalog.generation_attempt < 64


def test_generator_catalog_and_layers_are_byte_deterministic_and_seed_sensitive() -> None:
    first = _generated_proxy("medium")
    repeated = ProceduralTerrainProxyGenerator(_proxy_settings("medium")).generate(
        np.zeros(first.height.shape, dtype=np.float64),
        GridGeometry(width=128, height=128, resolution_m=0.5),
        PoseXYTheta(CellXY(16, 64), 0.0),
        scenario_key="smoke-v1",
        split="stage1",
        parent_roi="smoke-parent-roi",
    )
    changed = _generated_proxy("medium", 20260711)

    assert first.catalog == repeated.catalog
    assert first.layer_hashes == repeated.layer_hashes
    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            (first.height, first.hard_obstacle, first.slope_deg, first.traversability),
            (
                repeated.height,
                repeated.hard_obstacle,
                repeated.slope_deg,
                repeated.traversability,
            ),
            strict=True,
        )
    )
    assert changed.catalog.sha256 != first.catalog.sha256


@pytest.mark.parametrize("density_profile", ["low", "medium", "high"])
def test_generator_independently_satisfies_protection_boundary_spacing_and_quadrants(
    density_profile: str,
) -> None:
    bundle = _generated_proxy(density_profile)
    catalog = bundle.catalog
    start_x, start_y = 8.25, 32.25
    map_width_m = map_height_m = 64.0

    objects: list[tuple[str, float, float, float]] = []
    for rock in catalog.rocks:
        objects.append(("rock", rock.center_x_m, rock.center_y_m, rock.semi_major_m))
    for crater in catalog.craters:
        objects.append(("crater", crater.center_x_m, crater.center_y_m, 1.35 * crater.radius_m))

    for _, center_x, center_y, outer_radius in objects:
        assert outer_radius <= center_x <= map_width_m - outer_radius
        assert outer_radius <= center_y <= map_height_m - outer_radius
        assert math.hypot(center_x - start_x, center_y - start_y) >= 6.0 + outer_radius

    for left, right in combinations(catalog.rocks, 2):
        distance = math.hypot(left.center_x_m - right.center_x_m, left.center_y_m - right.center_y_m)
        assert distance >= 1.10 * (left.semi_major_m + right.semi_major_m)
    for left, right in combinations(catalog.craters, 2):
        distance = math.hypot(left.center_x_m - right.center_x_m, left.center_y_m - right.center_y_m)
        assert distance >= 1.05 * (1.35 * left.radius_m + 1.35 * right.radius_m)
    for rock in catalog.rocks:
        for crater in catalog.craters:
            distance = math.hypot(rock.center_x_m - crater.center_x_m, rock.center_y_m - crater.center_y_m)
            assert distance >= rock.semi_major_m + 1.35 * crater.radius_m

    quadrant_counts = [0, 0, 0, 0]
    for _, center_x, center_y, _ in objects:
        quadrant = int(center_x >= 32.0) + 2 * int(center_y >= 32.0)
        quadrant_counts[quadrant] += 1
    occupied = sum(count > 0 for count in quadrant_counts)
    assert occupied >= (3 if density_profile == "low" else 4)
    if density_profile in {"medium", "high"}:
        assert max(quadrant_counts) / len(objects) <= 0.40


def test_impossible_placement_fails_closed_at_configured_scene_bound() -> None:
    geometry = GridGeometry(width=8, height=8, resolution_m=0.5)
    generator = ProceduralTerrainProxyGenerator(
        _proxy_settings(
            "medium",
            max_scene_attempts=3,
            max_object_attempts=4,
        )
    )

    with pytest.raises(TerrainProxyGenerationError, match="placement_exhausted") as error:
        generator.generate(
            np.zeros(geometry.shape, dtype=np.float64),
            geometry,
            PoseXYTheta(CellXY(4, 4), 0.0),
            scenario_key="impossible",
            split="stage1-test",
            parent_roi="tiny-roi",
        )

    assert error.value.reason == "placement_exhausted"
    assert error.value.scene_attempts == 3


def _independent_rock_rho2(rock: RockProxy, x_m: float, y_m: float) -> float:
    dx = x_m - rock.center_x_m
    dy = y_m - rock.center_y_m
    cosine = math.cos(rock.orientation_rad)
    sine = math.sin(rock.orientation_rad)
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    return (local_x / rock.semi_major_m) ** 2 + (local_y / rock.semi_minor_m) ** 2


def _independent_crater_delta(crater: CraterProxy, distance_m: float) -> float:
    u = distance_m / crater.radius_m
    bowl = -crater.depth_m * (1.0 - u * u) ** 2 if u <= 1.0 else 0.0
    rim = (
        crater.rim_height_m
        * math.exp(-((u - 1.0) / crater.rim_width_ratio) ** 2)
        if u <= 1.35
        else 0.0
    )
    return bowl + rim


def test_rasterized_morphology_matches_independent_catalog_oracle_and_craters_write_no_hard_mask() -> None:
    bundle = _generated_proxy("medium")
    catalog = bundle.catalog
    geometry = GridGeometry(width=128, height=128, resolution_m=0.5)
    expected_height = np.zeros(geometry.shape, dtype=np.float64)
    expected_rock_core = np.zeros(geometry.shape, dtype=bool)

    for y in range(geometry.height):
        for x in range(geometry.width):
            center = geometry.cell_to_world_center(CellXY(x, y))
            for crater in catalog.craters:
                expected_height[y, x] += _independent_crater_delta(
                    crater,
                    math.hypot(center.x - crater.center_x_m, center.y - crater.center_y_m),
                )
            for rock in catalog.rocks:
                rho2 = _independent_rock_rho2(rock, center.x, center.y)
                if rho2 <= 1.0:
                    expected_height[y, x] += rock.height_m * (1.0 - rho2) ** 2
                if rho2 <= 0.85**2:
                    expected_rock_core[y, x] = True

    assert np.count_nonzero(expected_rock_core) > 0
    assert np.array_equal(bundle.hard_obstacle, expected_rock_core)
    assert np.allclose(bundle.height, expected_height, rtol=0.0, atol=1e-12)
    assert all(
        np.any(
            [
                expected_rock_core[y, x]
                for y in range(geometry.height)
                for x in range(geometry.width)
                if _independent_rock_rho2(
                    rock,
                    geometry.cell_to_world_center(CellXY(x, y)).x,
                    geometry.cell_to_world_center(CellXY(x, y)).y,
                )
                <= 0.85**2
            ]
        )
        for rock in catalog.rocks
    )
    for crater in catalog.craters:
        center_cell = geometry.world_to_cell(
            type(geometry.origin)(crater.center_x_m, crater.center_y_m)
        )
        assert not bundle.hard_obstacle[center_cell.y, center_cell.x]
        assert bundle.height[center_cell.y, center_cell.x] < 0.0


def test_derived_layers_and_hashes_are_independently_recomputable_and_mutation_sensitive() -> None:
    bundle = _generated_proxy("medium")
    dz_dy, dz_dx = np.gradient(bundle.height, 0.5, 0.5, edge_order=1)
    expected_slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    expected_traversability = np.clip(1.0 - expected_slope / 60.0, 0.0, 1.0)

    assert np.allclose(bundle.slope_deg, expected_slope, rtol=0.0, atol=1e-12)
    assert np.allclose(bundle.traversability, expected_traversability, rtol=0.0, atol=1e-12)
    assert np.any(bundle.slope_deg > 30.0)
    assert np.any((bundle.slope_deg <= 30.0) & (bundle.traversability >= 0.5))

    catalog_record = {
        "schema_version": bundle.catalog.schema_version,
        "generator_version": bundle.catalog.generator_version,
        "density_profile": bundle.catalog.density_profile,
        "seed_hex": bundle.catalog.seed_hex,
        "generation_attempt": bundle.catalog.generation_attempt,
        "rocks": [asdict(rock) for rock in bundle.catalog.rocks],
        "craters": [asdict(crater) for crater in bundle.catalog.craters],
    }
    assert bundle.catalog.sha256 == hashlib.sha256(
        ArtifactStore.canonical_json_bytes(catalog_record)
    ).hexdigest()
    for key, layer in {
        "height": bundle.height,
        "hard_obstacle": bundle.hard_obstacle,
        "slope": bundle.slope_deg,
        "traversability": bundle.traversability,
    }.items():
        assert bundle.layer_hashes[key] == _expected_layer_content_sha256(layer)

    mutated_height = bundle.height.copy()
    mutated_height[0, 0] += 1e-6
    assert _expected_layer_content_sha256(mutated_height) != bundle.layer_hashes["height"]
    mutated_catalog = json.loads(
        ArtifactStore.canonical_json_bytes(catalog_record).decode("utf-8")
    )
    mutated_catalog["rocks"][0]["height_m"] += 1e-6
    assert hashlib.sha256(ArtifactStore.canonical_json_bytes(mutated_catalog)).hexdigest() != bundle.catalog.sha256


def test_smoke_scenario_contains_auditable_rock_crater_proxy() -> None:
    from lunar_exploration_ppo.env.scenario import ScenarioSource

    scenario = ScenarioSource().load("smoke-v1")

    assert scenario.scenario_id == "smoke-v1/procedural-rock-crater/v1"
    assert 24 <= len(scenario.proxy_catalog.rocks) <= 36
    assert 4 <= len(scenario.proxy_catalog.craters) <= 6
    assert scenario.truth.provenance["proxy_generator_version"] == GENERATOR_VERSION
    assert scenario.truth.provenance["synthetic_source_kind"] == "synthetic_terrain_obstacle_proxy/v1"
    assert scenario.truth.provenance["physical_obstacle_cells_written"] is False
    assert scenario.truth.provenance["object_catalog_sha256"] == scenario.proxy_catalog.sha256
    assert dict(scenario.proxy_layer_hashes) == dict(scenario.truth.provenance["layer_hashes"])
    assert np.count_nonzero(scenario.truth.hard_obstacle) > 0


def test_scenario_prior_and_policy_api_never_expose_hidden_proxy_catalog() -> None:
    from lunar_exploration_ppo.env.frontier import FrontierGenerator
    from lunar_exploration_ppo.env.scenario import ScenarioSource
    from lunar_exploration_ppo.policy.observation import ObservationBuilder, PolicyObservation

    first = ScenarioSource(settings=_proxy_settings("medium", base_seed=20260710)).load("smoke-v1")
    changed = ScenarioSource(settings=_proxy_settings("medium", base_seed=20260711)).load("smoke-v1")

    assert first.proxy_catalog.sha256 != changed.proxy_catalog.sha256
    assert first.scenario_hash != changed.scenario_hash
    assert first.prior.channels.tobytes() == changed.prior.channels.tobytes()
    for api in (ObservationBuilder.build, FrontierGenerator.extract):
        signature = inspect.signature(api)
        annotations = repr(api.__annotations__).lower()
        assert "truth" not in signature.parameters
        assert "catalog" not in signature.parameters
        assert "truth" not in annotations
        assert "catalog" not in annotations
    assert not any(
        "truth" in name or "catalog" in name or "coverable" in name
        for name in PolicyObservation.__dataclass_fields__
    )


@pytest.mark.parametrize(
    "constructor_parameter",
    [
        pytest.param("range_m", id="max_range_m(range_m)"),
        pytest.param("fov_deg", id="horizontal_fov_deg(fov_deg)"),
        pytest.param("ray_angle_step_deg", id="ray_angle_step_deg"),
        pytest.param("min_clearance_m", id="min_clearance_m"),
        pytest.param(
            "max_slope_deg",
            id="max_traversable_slope_deg(max_slope_deg)",
        ),
        pytest.param(
            "traversability_threshold",
            id="traversability_threshold",
        ),
    ],
)
@pytest.mark.parametrize(
    "nonfinite_value",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_sensor_updater_constructor_rejects_nonfinite_parameters(
    constructor_parameter: str,
    nonfinite_value: float,
) -> None:
    from lunar_exploration_ppo.env.sensor_model import SensorUpdater

    with pytest.raises(
        ValueError,
        match=rf"{constructor_parameter} must be finite",
    ):
        SensorUpdater(**{constructor_parameter: nonfinite_value})


@pytest.mark.parametrize(
    ("constructor_parameter", "invalid_value"),
    [
        pytest.param("range_m", 0.0, id="range-zero"),
        pytest.param("range_m", -1.0, id="range-negative"),
        pytest.param("fov_deg", -1.0, id="fov-below-zero"),
        pytest.param("fov_deg", 360.000001, id="fov-above-360"),
        pytest.param("ray_angle_step_deg", 0.0, id="ray-step-zero"),
        pytest.param("ray_angle_step_deg", -1.0, id="ray-step-negative"),
        pytest.param("min_clearance_m", -1.0, id="clearance-below-zero"),
        pytest.param("max_slope_deg", -1.0, id="slope-below-zero"),
        pytest.param("max_slope_deg", 90.000001, id="slope-above-90"),
        pytest.param(
            "traversability_threshold",
            -0.000001,
            id="traversability-below-zero",
        ),
        pytest.param(
            "traversability_threshold",
            1.000001,
            id="traversability-above-one",
        ),
    ],
)
def test_sensor_updater_constructor_rejects_values_outside_existing_contracts(
    constructor_parameter: str,
    invalid_value: float,
) -> None:
    from lunar_exploration_ppo.env.sensor_model import SensorUpdater

    with pytest.raises(ValueError):
        SensorUpdater(**{constructor_parameter: invalid_value})


def test_sensor_updater_constructor_accepts_closed_interval_boundaries() -> None:
    from lunar_exploration_ppo.env.sensor_model import SensorUpdater

    lower = SensorUpdater(
        range_m=0.000001,
        fov_deg=0.0,
        ray_angle_step_deg=0.000001,
        min_clearance_m=0.0,
        max_slope_deg=0.0,
        traversability_threshold=0.0,
    )
    upper = SensorUpdater(
        fov_deg=360.0,
        max_slope_deg=90.0,
        traversability_threshold=1.0,
    )

    assert lower.fov_deg == 0.0
    assert lower.min_clearance_m == 0.0
    assert lower.max_slope_deg == 0.0
    assert lower.traversability_threshold == 0.0
    assert upper.fov_deg == 360.0
    assert upper.max_slope_deg == 90.0
    assert upper.traversability_threshold == 1.0


def _independent_collinear_truth(
    blocker_kind: str,
    *,
    blocker_cell: CellXY = CellXY(5, 4),
):
    from lunar_exploration_ppo.env.scenario import TruthMap

    geometry = GridGeometry(width=9, height=9, resolution_m=1.0)
    height = np.zeros(geometry.shape, dtype=np.float64)
    hard_obstacle = np.zeros(geometry.shape, dtype=bool)
    slope_deg = np.zeros(geometry.shape, dtype=np.float64)
    traversability = np.ones(geometry.shape, dtype=np.float64)
    if blocker_kind == "rock_hard":
        hard_obstacle[blocker_cell.y, blocker_cell.x] = True
        height[blocker_cell.y, blocker_cell.x] = 1.0
    elif blocker_kind == "crater_steep":
        slope_deg[blocker_cell.y, blocker_cell.x] = 31.0
        traversability[blocker_cell.y, blocker_cell.x] = 1.0 - 31.0 / 60.0
        height[blocker_cell.y, blocker_cell.x] = -0.5
    elif blocker_kind == "crater_gentle":
        slope_deg[blocker_cell.y, blocker_cell.x] = 29.0
        traversability[blocker_cell.y, blocker_cell.x] = 1.0 - 29.0 / 60.0
        height[blocker_cell.y, blocker_cell.x] = -0.5
    else:
        raise AssertionError(f"unsupported blocker fixture: {blocker_kind}")
    return TruthMap(
        geometry=geometry,
        height=height,
        hard_obstacle=hard_obstacle,
        slope_deg=slope_deg,
        traversability=traversability,
        provenance={
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "physical_obstacle_cells_written": False,
        },
    )


@pytest.mark.parametrize("blocker_kind", ["rock_hard", "crater_steep"])
def test_origin_blocker_is_visible_without_occluding_next_ray_cell(blocker_kind: str) -> None:
    from lunar_exploration_ppo.env.map_state import ObservedMapState
    from lunar_exploration_ppo.env.sensor_model import SensorPose, SensorUpdater

    origin_cell = CellXY(2, 4)
    truth = _independent_collinear_truth(blocker_kind, blocker_cell=origin_cell)
    state = ObservedMapState.empty(truth.geometry)
    delta = SensorUpdater(range_m=10.0, fov_deg=0.0).reveal(
        truth,
        state,
        (
            SensorPose(
                truth.geometry.cell_to_world_center(origin_cell),
                0.0,
                "endpoint_theta",
            ),
        ),
    )
    expected_visible = tuple(CellXY(x, 4) for x in range(2, 9))

    assert delta.visible_cells == expected_visible
    assert state.observed_mask[origin_cell.y, origin_cell.x]
    assert state.observed_mask[4, 3]
    assert delta.diagnostics.cell_visit_count == len(expected_visible)
    assert delta.diagnostics.unique_visible_cell_count == len(expected_visible)


@pytest.mark.parametrize("blocker_kind", ["rock_hard", "crater_steep"])
def test_proxy_blocker_is_visible_and_cell_behind_remains_unknown(blocker_kind: str) -> None:
    from lunar_exploration_ppo.env.map_state import ObservedMapState
    from lunar_exploration_ppo.env.sensor_model import SensorPose, SensorUpdater

    truth = _independent_collinear_truth(blocker_kind)
    origin = truth.geometry.cell_to_world_center(CellXY(2, 4))
    updater = SensorUpdater(range_m=10.0, fov_deg=0.0)
    blocked_state = ObservedMapState.empty(truth.geometry)
    blocked_delta = updater.reveal(
        truth,
        blocked_state,
        (SensorPose(origin, 0.0, "endpoint_theta"),),
    )

    assert CellXY(5, 4) in blocked_delta.visible_cells
    assert CellXY(6, 4) not in blocked_delta.visible_cells
    assert blocked_state.observed_mask[4, 5]
    assert not blocked_state.observed_mask[4, 6]

    adjacent_state = ObservedMapState.empty(truth.geometry)
    updater.reveal(
        truth,
        adjacent_state,
        (SensorPose(origin, math.atan2(1.0, 4.0), "endpoint_theta"),),
    )
    assert adjacent_state.observed_mask[5, 6]


def test_gentle_crater_does_not_block_2d_los() -> None:
    from lunar_exploration_ppo.env.map_state import ObservedMapState
    from lunar_exploration_ppo.env.sensor_model import SensorPose, SensorUpdater

    truth = _independent_collinear_truth("crater_gentle")
    state = ObservedMapState.empty(truth.geometry)
    delta = SensorUpdater(range_m=10.0, fov_deg=0.0).reveal(
        truth,
        state,
        (
            SensorPose(
                truth.geometry.cell_to_world_center(CellXY(2, 4)),
                0.0,
                "endpoint_theta",
            ),
        ),
    )

    assert CellXY(5, 4) in delta.visible_cells
    assert CellXY(6, 4) in delta.visible_cells
    assert state.observed_mask[4, 6]


def _minimal_valid_stage1_summary_with_proxy_morphology() -> dict[str, object]:
    return {
        "scenario_id": "smoke-v1/procedural-rock-crater/v1",
        "scenario_hash": "1" * 64,
        "coverage_mask": {"sha256": "2" * 64},
        "proxy_morphology": {
            "proxy_generator_version": GENERATOR_VERSION,
            "density_profile": "medium",
            "generation_attempt": 0,
            "rock_count": 24,
            "crater_count": 4,
            "object_catalog_sha256": "3" * 64,
            "layer_hashes": {
                "height": "4" * 64,
                "hard_obstacle": "5" * 64,
                "slope": "6" * 64,
                "traversability": "7" * 64,
            },
        },
    }


def test_gate_scenario_identity_hash_binds_proxy_catalog() -> None:
    from lunar_exploration_ppo.workflows.stage1_gate import Stage1GateBindingVerifier

    config = load_stage1_config(STAGE1_CONFIG)
    summary = _minimal_valid_stage1_summary_with_proxy_morphology()
    first = Stage1GateBindingVerifier._scenario_identity_hash(summary, config)
    changed = json.loads(json.dumps(summary))
    changed["proxy_morphology"]["object_catalog_sha256"] = "0" * 64
    second = Stage1GateBindingVerifier._scenario_identity_hash(changed, config)

    assert first != second


def test_environment_exposes_only_aggregate_proxy_morphology_metadata() -> None:
    from lunar_exploration_ppo.env.env import LunarExplorationEnv

    env = LunarExplorationEnv(load_stage1_config(STAGE1_CONFIG))
    metadata = env.proxy_morphology_metadata

    assert set(metadata) == {
        "proxy_generator_version",
        "density_profile",
        "generation_attempt",
        "rock_count",
        "crater_count",
        "object_catalog_sha256",
        "layer_hashes",
    }
    assert metadata["proxy_generator_version"] == GENERATOR_VERSION
    assert metadata["density_profile"] == "medium"
    assert 24 <= metadata["rock_count"] <= 36
    assert 4 <= metadata["crater_count"] <= 6
    assert "rocks" not in metadata and "craters" not in metadata
    metadata["layer_hashes"]["height"] = "0" * 64
    assert env.proxy_morphology_metadata["layer_hashes"]["height"] != "0" * 64


def test_reviewed_stage1_source_set_binds_all_proxy_contract_paths() -> None:
    from lunar_exploration_ppo.workflows.stage1_source import STAGE1_REVIEWED_PATHS

    assert {
        "docs/superpowers/plans/2026-07-10-ppo-highres-frontier-map-exploration.md",
        "docs/superpowers/specs/2026-07-10-rock-crater-proxy-fixture-design-addendum.md",
        "docs/superpowers/plans/2026-07-10-rock-crater-proxy-fixture.md",
        "src/lunar_exploration_ppo/env/terrain_proxy.py",
        "tests/ppo_highres_frontier/test_stage1_terrain_proxy.py",
    } <= STAGE1_REVIEWED_PATHS
    assert "tests/ppo_highres_frontier/test_stage1_smoke_env_r1.py" not in STAGE1_REVIEWED_PATHS
    assert len(STAGE1_REVIEWED_PATHS) == 31


def test_fixed_smoke_seed_passes_all_structural_gates_without_seed_search() -> None:
    from lunar_exploration_ppo.env.env import LunarExplorationEnv

    config = load_stage1_config(STAGE1_CONFIG)
    assert config.proxy_base_seed == 20260710
    assert config.proxy_density_profile == "medium"
    env = LunarExplorationEnv(config)
    observation = env.reset()
    masks = env._coverage_masks
    start = env._scenario.start_pose.cell
    safe_free_count = int(np.count_nonzero(masks.safe_free_mask))
    reachable_safe_count = int(np.count_nonzero(masks.reachable_safe_mask))

    assert masks.safe_free_mask[start.y, start.x]
    assert safe_free_count > 0
    assert reachable_safe_count / safe_free_count >= 0.70
    assert masks.coverable_cell_count > 0
    assert env.last_reset_diagnostics is not None
    assert env.last_reset_diagnostics.coverage_rate < 0.99
    assert env.current_action_set.candidate_count > 0
    assert np.any(observation.candidate_mask)


def test_fixed_smoke_structural_gate_enforcement_fails_closed_on_reachability_drift() -> None:
    from lunar_exploration_ppo.env.coverage import CoverageMasks
    from lunar_exploration_ppo.env.env import LunarExplorationEnv

    env = LunarExplorationEnv(load_stage1_config(STAGE1_CONFIG))
    current = env._coverage_masks
    reachable = np.zeros_like(current.reachable_safe_mask)
    start = env._scenario.start_pose.cell
    reachable[start.y, start.x] = True
    env._coverage_masks = CoverageMasks(
        safe_free_mask=current.safe_free_mask.copy(),
        reachable_safe_mask=reachable,
        coverable_mask=current.coverable_mask.copy(),
        metadata=dict(current.metadata),
    )

    with pytest.raises(RuntimeError, match="reachable safe ratio"):
        env.reset()
