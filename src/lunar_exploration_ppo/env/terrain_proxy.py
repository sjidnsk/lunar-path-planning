"""确定性岩石/陨石坑高分辨率 terrain proxy。"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Final, Literal, Mapping

import numpy as np

from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.geometry import GridGeometry, PoseXYTheta


GENERATOR_VERSION: Final = "procedural_lunar_rock_crater_proxy/v1"

DensityProfile = Literal["low", "medium", "high"]
ObjectKind = Literal["rock", "crater"]

DENSITY_COUNTS: Final[Mapping[str, Mapping[str, tuple[int, int]]]] = MappingProxyType(
    {
        "low": MappingProxyType({"rock": (10, 18), "crater": (2, 3)}),
        "medium": MappingProxyType({"rock": (24, 36), "crater": (4, 6)}),
        "high": MappingProxyType({"rock": (45, 64), "crater": (7, 10)}),
    }
)
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_SEED_PATTERN: Final = re.compile(r"^[0-9a-f]{32}$")
_LAYER_HASH_KEYS: Final = frozenset(
    {"height", "hard_obstacle", "slope", "traversability"}
)


class TerrainProxyGenerationError(RuntimeError):
    """在有限次数内无法生成满足合同的 terrain proxy。"""

    def __init__(self, reason: str, *, scene_attempts: int) -> None:
        self.reason = reason
        self.scene_attempts = scene_attempts
        super().__init__(f"{reason}: scene_attempts={scene_attempts}")


@dataclass(frozen=True, slots=True)
class TerrainProxySettings:
    generator_version: str
    density_profile: DensityProfile
    base_seed: int
    start_protection_m: float
    max_scene_attempts: int
    max_object_attempts: int

    def __post_init__(self) -> None:
        if self.generator_version != GENERATOR_VERSION:
            raise ValueError("unsupported terrain proxy generator version")
        if self.density_profile not in DENSITY_COUNTS:
            raise ValueError("unsupported terrain proxy density profile")
        if type(self.base_seed) is not int:
            raise ValueError("base_seed must be an integer")
        if not math.isfinite(self.start_protection_m) or self.start_protection_m <= 0.0:
            raise ValueError("start protection must be positive and finite")
        if (
            type(self.max_scene_attempts) is not int
            or type(self.max_object_attempts) is not int
            or self.max_scene_attempts <= 0
            or self.max_object_attempts <= 0
        ):
            raise ValueError("attempt limits must be positive integers")


@dataclass(frozen=True, slots=True)
class RockProxy:
    center_x_m: float
    center_y_m: float
    equivalent_radius_m: float
    semi_major_m: float
    semi_minor_m: float
    orientation_rad: float
    height_m: float

    def __post_init__(self) -> None:
        values = (
            self.center_x_m,
            self.center_y_m,
            self.equivalent_radius_m,
            self.semi_major_m,
            self.semi_minor_m,
            self.orientation_rad,
            self.height_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("rock parameters must be finite")
        if not 0.5 <= self.equivalent_radius_m <= 1.5:
            raise ValueError("rock equivalent radius is outside [0.5, 1.5]")
        if self.semi_minor_m <= 0.0 or self.semi_major_m < self.semi_minor_m:
            raise ValueError("rock semi axes are invalid")
        axis_ratio = self.semi_major_m / self.semi_minor_m
        if not 1.0 <= axis_ratio <= 1.8:
            raise ValueError("rock axis ratio is outside [1.0, 1.8]")
        if not math.isclose(
            self.semi_major_m * self.semi_minor_m,
            self.equivalent_radius_m**2,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("rock semi axes do not preserve equivalent radius")
        if not -math.pi <= self.orientation_rad < math.pi:
            raise ValueError("rock orientation is outside [-pi, pi)")
        if not 0.2 <= self.height_m <= 1.0:
            raise ValueError("rock height is outside [0.2, 1.0]")


@dataclass(frozen=True, slots=True)
class CraterProxy:
    center_x_m: float
    center_y_m: float
    radius_m: float
    depth_m: float
    rim_height_m: float
    rim_width_ratio: float

    def __post_init__(self) -> None:
        values = (
            self.center_x_m,
            self.center_y_m,
            self.radius_m,
            self.depth_m,
            self.rim_height_m,
            self.rim_width_ratio,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("crater parameters must be finite")
        if not 2.0 <= self.radius_m <= 6.0:
            raise ValueError("crater radius is outside [2.0, 6.0]")
        if not 0.15 * self.radius_m <= self.depth_m <= 0.30 * self.radius_m:
            raise ValueError("crater depth is outside [0.15R, 0.30R]")
        if not 0.05 * self.radius_m <= self.rim_height_m <= 0.12 * self.radius_m:
            raise ValueError("crater rim height is outside [0.05R, 0.12R]")
        if not 0.12 <= self.rim_width_ratio <= 0.25:
            raise ValueError("crater rim width ratio is outside [0.12, 0.25]")


def _catalog_content_record(
    *,
    schema_version: str,
    generator_version: str,
    density_profile: DensityProfile,
    seed_hex: str,
    generation_attempt: int,
    rocks: tuple[RockProxy, ...],
    craters: tuple[CraterProxy, ...],
) -> dict[str, object]:
    """Return the single canonical catalog representation used for hashing."""

    return {
        "schema_version": schema_version,
        "generator_version": generator_version,
        "density_profile": density_profile,
        "seed_hex": seed_hex,
        "generation_attempt": generation_attempt,
        "rocks": [asdict(rock) for rock in rocks],
        "craters": [asdict(crater) for crater in craters],
    }


def _catalog_content_sha256(
    *,
    schema_version: str,
    generator_version: str,
    density_profile: DensityProfile,
    seed_hex: str,
    generation_attempt: int,
    rocks: tuple[RockProxy, ...],
    craters: tuple[CraterProxy, ...],
) -> str:
    record = _catalog_content_record(
        schema_version=schema_version,
        generator_version=generator_version,
        density_profile=density_profile,
        seed_hex=seed_hex,
        generation_attempt=generation_attempt,
        rocks=rocks,
        craters=craters,
    )
    return hashlib.sha256(ArtifactStore.canonical_json_bytes(record)).hexdigest()


@dataclass(frozen=True, slots=True)
class TerrainProxyCatalog:
    schema_version: str
    generator_version: str
    density_profile: DensityProfile
    seed_hex: str
    generation_attempt: int
    rocks: tuple[RockProxy, ...]
    craters: tuple[CraterProxy, ...]
    sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "terrain_proxy_catalog/v1":
            raise ValueError("terrain proxy catalog schema is invalid")
        if self.generator_version != GENERATOR_VERSION:
            raise ValueError("terrain proxy catalog generator version is invalid")
        if self.density_profile not in DENSITY_COUNTS:
            raise ValueError("terrain proxy catalog density profile is invalid")
        if _SEED_PATTERN.fullmatch(self.seed_hex) is None:
            raise ValueError("terrain proxy catalog seed_hex is invalid")
        if type(self.generation_attempt) is not int or self.generation_attempt < 0:
            raise ValueError("terrain proxy catalog generation attempt is invalid")
        if not isinstance(self.rocks, tuple) or not all(
            isinstance(item, RockProxy) for item in self.rocks
        ):
            raise ValueError("terrain proxy catalog rocks must be an immutable tuple")
        if not isinstance(self.craters, tuple) or not all(
            isinstance(item, CraterProxy) for item in self.craters
        ):
            raise ValueError("terrain proxy catalog craters must be an immutable tuple")
        if _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("terrain proxy catalog sha256 is invalid")
        actual_hash = _catalog_content_sha256(
            schema_version=self.schema_version,
            generator_version=self.generator_version,
            density_profile=self.density_profile,
            seed_hex=self.seed_hex,
            generation_attempt=self.generation_attempt,
            rocks=self.rocks,
            craters=self.craters,
        )
        if not hmac.compare_digest(self.sha256, actual_hash):
            raise ValueError("terrain proxy catalog sha256 does not match content")


@dataclass(frozen=True, slots=True)
class TerrainProxyBundle:
    height: np.ndarray
    hard_obstacle: np.ndarray
    slope_deg: np.ndarray
    traversability: np.ndarray
    catalog: TerrainProxyCatalog
    layer_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        layers = (self.height, self.hard_obstacle, self.slope_deg, self.traversability)
        if not all(isinstance(layer, np.ndarray) and layer.ndim == 2 for layer in layers):
            raise ValueError("terrain proxy layers must be two-dimensional numpy arrays")
        if len({layer.shape for layer in layers}) != 1:
            raise ValueError("terrain proxy layer shapes must match")
        if self.hard_obstacle.dtype != np.dtype(bool):
            raise ValueError("terrain proxy hard obstacle layer must be boolean")
        if not all(
            np.all(np.isfinite(layer))
            for layer in (self.height, self.slope_deg, self.traversability)
        ):
            raise ValueError("terrain proxy layers must be finite")
        if np.any(self.slope_deg < 0.0):
            raise ValueError("terrain proxy slope layer must be nonnegative")
        if np.any((self.traversability < 0.0) | (self.traversability > 1.0)):
            raise ValueError("terrain proxy traversability must be inside [0, 1]")
        hashes = dict(self.layer_hashes)
        if set(hashes) != _LAYER_HASH_KEYS or any(
            _SHA256_PATTERN.fullmatch(value) is None for value in hashes.values()
        ):
            raise ValueError("terrain proxy layer hashes are invalid")
        named_layers = {
            "height": self.height,
            "hard_obstacle": self.hard_obstacle,
            "slope": self.slope_deg,
            "traversability": self.traversability,
        }
        for name, layer in named_layers.items():
            if not hmac.compare_digest(hashes[name], _array_sha256(layer)):
                raise ValueError(f"terrain proxy {name} layer hash does not match content")
        for layer in layers:
            layer.setflags(write=False)
        object.__setattr__(self, "layer_hashes", MappingProxyType(hashes))


def derive_proxy_seed(
    *,
    scenario_key: str,
    split: str,
    parent_roi: str,
    base_seed: int,
    density_profile: DensityProfile,
) -> str:
    """从 canonical UTF-8 场景身份派生 PCG64 所需的 128-bit seed。"""

    if density_profile not in DENSITY_COUNTS:
        raise ValueError("unsupported terrain proxy density profile")
    if type(base_seed) is not int:
        raise ValueError("base_seed must be an integer")
    source = "|".join(
        (
            GENERATOR_VERSION,
            scenario_key,
            split,
            parent_roi,
            str(base_seed),
            density_profile,
        )
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:32]


def rock_height_at(rock: RockProxy, x_m: float, y_m: float) -> float:
    """返回旋转椭圆岩石在给定 world 点的解析高度增量。"""

    _validate_world_coordinates(x_m, y_m)
    dx, dy = x_m - rock.center_x_m, y_m - rock.center_y_m
    cosine, sine = math.cos(rock.orientation_rad), math.sin(rock.orientation_rad)
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    rho2 = (local_x / rock.semi_major_m) ** 2 + (
        local_y / rock.semi_minor_m
    ) ** 2
    return rock.height_m * (1.0 - rho2) ** 2 if rho2 <= 1.0 else 0.0


def rock_is_core(rock: RockProxy, x_m: float, y_m: float) -> bool:
    """判断 world 点是否落在岩石 synthetic hard core 内。"""

    _validate_world_coordinates(x_m, y_m)
    dx, dy = x_m - rock.center_x_m, y_m - rock.center_y_m
    cosine, sine = math.cos(rock.orientation_rad), math.sin(rock.orientation_rad)
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    return math.hypot(
        local_x / rock.semi_major_m,
        local_y / rock.semi_minor_m,
    ) <= 0.85


def crater_height_at(crater: CraterProxy, distance_m: float) -> float:
    """返回圆形陨石坑 bowl 与 rim 叠加后的解析高度增量。"""

    if not math.isfinite(distance_m) or distance_m < 0.0:
        raise ValueError("crater distance must be finite and nonnegative")
    u = distance_m / crater.radius_m
    bowl = -crater.depth_m * (1.0 - u * u) ** 2 if u <= 1.0 else 0.0
    rim = (
        crater.rim_height_m
        * math.exp(-((u - 1.0) / crater.rim_width_ratio) ** 2)
        if u <= 1.35
        else 0.0
    )
    return bowl + rim


def _validate_world_coordinates(x_m: float, y_m: float) -> None:
    if not math.isfinite(x_m) or not math.isfinite(y_m):
        raise ValueError("rock world coordinates must be finite")


class ProceduralTerrainProxyGenerator:
    """使用 PCG64 与有限拒绝采样生成可审计对象 catalog。"""

    def __init__(self, settings: TerrainProxySettings) -> None:
        if not isinstance(settings, TerrainProxySettings):
            raise TypeError("settings must be TerrainProxySettings")
        self.settings = settings

    def generate(
        self,
        base_height: np.ndarray,
        geometry: GridGeometry,
        start_pose: PoseXYTheta,
        *,
        scenario_key: str,
        split: str,
        parent_roi: str,
    ) -> TerrainProxyBundle:
        base = np.asarray(base_height, dtype=np.float64)
        if base.shape != geometry.shape or base.ndim != 2:
            raise ValueError("base_height shape must match geometry")
        if not np.all(np.isfinite(base)):
            raise ValueError("base_height must be finite")
        if not geometry.in_bounds(start_pose.cell):
            raise ValueError("start pose is outside geometry")

        seed_hex = derive_proxy_seed(
            scenario_key=scenario_key,
            split=split,
            parent_roi=parent_roi,
            base_seed=self.settings.base_seed,
            density_profile=self.settings.density_profile,
        )
        area_m2 = (
            geometry.width
            * geometry.resolution_m
            * geometry.height
            * geometry.resolution_m
        )
        count_rng = np.random.Generator(np.random.PCG64(int(seed_hex, 16)))
        rock_low, rock_high = scaled_count_bounds(
            self.settings.density_profile,
            "rock",
            area_m2,
        )
        crater_low, crater_high = scaled_count_bounds(
            self.settings.density_profile,
            "crater",
            area_m2,
        )
        target_rock_count = int(count_rng.integers(rock_low, rock_high + 1))
        target_crater_count = int(count_rng.integers(crater_low, crater_high + 1))

        for scene_attempt in range(self.settings.max_scene_attempts):
            attempt_source = f"{seed_hex}|scene_attempt|{scene_attempt}".encode("utf-8")
            attempt_seed = int(hashlib.sha256(attempt_source).hexdigest()[:32], 16)
            rng = np.random.Generator(np.random.PCG64(attempt_seed))
            placed = self._place_catalog(
                rng,
                geometry,
                start_pose,
                rock_count=target_rock_count,
                crater_count=target_crater_count,
            )
            if placed is None:
                continue
            rocks, craters = placed
            catalog = _build_catalog(
                density_profile=self.settings.density_profile,
                seed_hex=seed_hex,
                generation_attempt=scene_attempt,
                rocks=rocks,
                craters=craters,
            )
            return _rasterize_bundle(base, geometry, catalog)
        raise TerrainProxyGenerationError(
            "placement_exhausted",
            scene_attempts=self.settings.max_scene_attempts,
        )

    def _place_catalog(
        self,
        rng: np.random.Generator,
        geometry: GridGeometry,
        start_pose: PoseXYTheta,
        *,
        rock_count: int,
        crater_count: int,
    ) -> tuple[tuple[RockProxy, ...], tuple[CraterProxy, ...]] | None:
        rocks: list[RockProxy] = []
        craters: list[CraterProxy] = []
        quadrant_counts = [0, 0, 0, 0]
        start = geometry.cell_to_world_center(start_pose.cell)

        for _ in range(crater_count):
            radius = float(rng.uniform(2.0, 6.0))
            crater_parameters = (
                radius,
                float(rng.uniform(0.15, 0.30) * radius),
                float(rng.uniform(0.05, 0.12) * radius),
                float(rng.uniform(0.12, 0.25)),
            )
            accepted: CraterProxy | None = None
            for candidate_index in range(self.settings.max_object_attempts):
                quadrant = _candidate_quadrant(rng, quadrant_counts, candidate_index)
                center = _sample_center(rng, geometry, 1.35 * radius, quadrant)
                if center is None:
                    continue
                candidate = CraterProxy(center[0], center[1], *crater_parameters)
                if _crater_position_is_valid(
                    candidate,
                    craters,
                    start_x=start.x,
                    start_y=start.y,
                    start_protection_m=self.settings.start_protection_m,
                ):
                    accepted = candidate
                    break
            if accepted is None:
                return None
            craters.append(accepted)
            quadrant_counts[_quadrant_for(accepted.center_x_m, accepted.center_y_m, geometry)] += 1

        for _ in range(rock_count):
            equivalent_radius = float(rng.uniform(0.5, 1.5))
            axis_ratio = float(rng.uniform(1.0, 1.8))
            semi_major = equivalent_radius * math.sqrt(axis_ratio)
            semi_minor = equivalent_radius / math.sqrt(axis_ratio)
            rock_parameters = (
                equivalent_radius,
                semi_major,
                semi_minor,
                float(rng.uniform(-math.pi, math.pi)),
                float(rng.uniform(0.2, 1.0)),
            )
            accepted_rock: RockProxy | None = None
            for candidate_index in range(self.settings.max_object_attempts):
                quadrant = _candidate_quadrant(rng, quadrant_counts, candidate_index)
                center = _sample_center(rng, geometry, semi_major, quadrant)
                if center is None:
                    continue
                candidate = RockProxy(center[0], center[1], *rock_parameters)
                if _rock_position_is_valid(
                    candidate,
                    rocks,
                    craters,
                    start_x=start.x,
                    start_y=start.y,
                    start_protection_m=self.settings.start_protection_m,
                ) and _rock_core_has_cell(candidate, geometry):
                    accepted_rock = candidate
                    break
            if accepted_rock is None:
                return None
            rocks.append(accepted_rock)
            quadrant_counts[
                _quadrant_for(accepted_rock.center_x_m, accepted_rock.center_y_m, geometry)
            ] += 1

        minimum_occupied = 3 if self.settings.density_profile == "low" else 4
        if sum(count > 0 for count in quadrant_counts) < minimum_occupied:
            return None
        total = sum(quadrant_counts)
        if self.settings.density_profile in {"medium", "high"} and max(quadrant_counts) / total > 0.40:
            return None
        return tuple(rocks), tuple(craters)


def _candidate_quadrant(
    rng: np.random.Generator,
    counts: list[int],
    candidate_index: int,
) -> int:
    jitter = rng.random(4)
    ordered = sorted(range(4), key=lambda index: (counts[index], float(jitter[index])))
    return ordered[candidate_index % 4]


def _quadrant_for(x_m: float, y_m: float, geometry: GridGeometry) -> int:
    middle_x = geometry.origin.x + geometry.width * geometry.resolution_m / 2.0
    middle_y = geometry.origin.y + geometry.height * geometry.resolution_m / 2.0
    return int(x_m >= middle_x) + 2 * int(y_m >= middle_y)


def _sample_center(
    rng: np.random.Generator,
    geometry: GridGeometry,
    outer_radius_m: float,
    quadrant: int,
) -> tuple[float, float] | None:
    map_min_x = geometry.origin.x
    map_min_y = geometry.origin.y
    map_max_x = map_min_x + geometry.width * geometry.resolution_m
    map_max_y = map_min_y + geometry.height * geometry.resolution_m
    middle_x = (map_min_x + map_max_x) / 2.0
    middle_y = (map_min_y + map_max_y) / 2.0
    quadrant_min_x = middle_x if quadrant & 1 else map_min_x
    quadrant_max_x = map_max_x if quadrant & 1 else middle_x
    quadrant_min_y = middle_y if quadrant & 2 else map_min_y
    quadrant_max_y = map_max_y if quadrant & 2 else middle_y
    low_x = max(quadrant_min_x, map_min_x + outer_radius_m)
    high_x = min(quadrant_max_x, map_max_x - outer_radius_m)
    low_y = max(quadrant_min_y, map_min_y + outer_radius_m)
    high_y = min(quadrant_max_y, map_max_y - outer_radius_m)
    if low_x >= high_x or low_y >= high_y:
        return None
    return float(rng.uniform(low_x, high_x)), float(rng.uniform(low_y, high_y))


def _crater_position_is_valid(
    candidate: CraterProxy,
    craters: list[CraterProxy],
    *,
    start_x: float,
    start_y: float,
    start_protection_m: float,
) -> bool:
    outer = 1.35 * candidate.radius_m
    if math.hypot(candidate.center_x_m - start_x, candidate.center_y_m - start_y) < start_protection_m + outer:
        return False
    return all(
        math.hypot(
            candidate.center_x_m - existing.center_x_m,
            candidate.center_y_m - existing.center_y_m,
        )
        >= 1.05 * (outer + 1.35 * existing.radius_m)
        for existing in craters
    )


def _rock_position_is_valid(
    candidate: RockProxy,
    rocks: list[RockProxy],
    craters: list[CraterProxy],
    *,
    start_x: float,
    start_y: float,
    start_protection_m: float,
) -> bool:
    if math.hypot(candidate.center_x_m - start_x, candidate.center_y_m - start_y) < (
        start_protection_m + candidate.semi_major_m
    ):
        return False
    if any(
        math.hypot(
            candidate.center_x_m - existing.center_x_m,
            candidate.center_y_m - existing.center_y_m,
        )
        < 1.10 * (candidate.semi_major_m + existing.semi_major_m)
        for existing in rocks
    ):
        return False
    return not any(
        math.hypot(
            candidate.center_x_m - crater.center_x_m,
            candidate.center_y_m - crater.center_y_m,
        )
        < candidate.semi_major_m + 1.35 * crater.radius_m
        for crater in craters
    )


def _rock_core_has_cell(rock: RockProxy, geometry: GridGeometry) -> bool:
    core_radius = 0.85 * rock.semi_major_m
    min_x = max(
        0,
        math.floor((rock.center_x_m - core_radius - geometry.origin.x) / geometry.resolution_m),
    )
    max_x = min(
        geometry.width - 1,
        math.floor((rock.center_x_m + core_radius - geometry.origin.x) / geometry.resolution_m),
    )
    min_y = max(
        0,
        math.floor((rock.center_y_m - core_radius - geometry.origin.y) / geometry.resolution_m),
    )
    max_y = min(
        geometry.height - 1,
        math.floor((rock.center_y_m + core_radius - geometry.origin.y) / geometry.resolution_m),
    )
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            center_x = geometry.origin.x + (x + 0.5) * geometry.resolution_m
            center_y = geometry.origin.y + (y + 0.5) * geometry.resolution_m
            if rock_is_core(rock, center_x, center_y):
                return True
    return False


def _build_catalog(
    *,
    density_profile: DensityProfile,
    seed_hex: str,
    generation_attempt: int,
    rocks: tuple[RockProxy, ...],
    craters: tuple[CraterProxy, ...],
) -> TerrainProxyCatalog:
    catalog_hash = _catalog_content_sha256(
        schema_version="terrain_proxy_catalog/v1",
        generator_version=GENERATOR_VERSION,
        density_profile=density_profile,
        seed_hex=seed_hex,
        generation_attempt=generation_attempt,
        rocks=rocks,
        craters=craters,
    )
    return TerrainProxyCatalog(
        schema_version="terrain_proxy_catalog/v1",
        generator_version=GENERATOR_VERSION,
        density_profile=density_profile,
        seed_hex=seed_hex,
        generation_attempt=generation_attempt,
        rocks=rocks,
        craters=craters,
        sha256=catalog_hash,
    )


def _rasterize_bundle(
    base_height: np.ndarray,
    geometry: GridGeometry,
    catalog: TerrainProxyCatalog,
) -> TerrainProxyBundle:
    height = np.asarray(base_height, dtype=np.float64).copy()
    hard_obstacle = np.zeros(height.shape, dtype=bool)
    grid_y, grid_x = np.mgrid[0 : geometry.height, 0 : geometry.width]
    world_x = geometry.origin.x + (grid_x + 0.5) * geometry.resolution_m
    world_y = geometry.origin.y + (grid_y + 0.5) * geometry.resolution_m

    for crater in catalog.craters:
        distance = np.hypot(world_x - crater.center_x_m, world_y - crater.center_y_m)
        u = distance / crater.radius_m
        bowl = np.where(
            u <= 1.0,
            -crater.depth_m * (1.0 - u * u) ** 2,
            0.0,
        )
        rim = np.where(
            u <= 1.35,
            crater.rim_height_m
            * np.exp(-((u - 1.0) / crater.rim_width_ratio) ** 2),
            0.0,
        )
        height += bowl + rim

    for rock in catalog.rocks:
        dx = world_x - rock.center_x_m
        dy = world_y - rock.center_y_m
        cosine, sine = math.cos(rock.orientation_rad), math.sin(rock.orientation_rad)
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        rho2 = (local_x / rock.semi_major_m) ** 2 + (
            local_y / rock.semi_minor_m
        ) ** 2
        height += np.where(rho2 <= 1.0, rock.height_m * (1.0 - rho2) ** 2, 0.0)
        hard_obstacle |= rho2 <= 0.85**2

    dz_dy, dz_dx = np.gradient(
        height,
        geometry.resolution_m,
        geometry.resolution_m,
        edge_order=1,
    )
    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    traversability = np.clip(1.0 - slope_deg / 60.0, 0.0, 1.0)
    return TerrainProxyBundle(
        height=height,
        hard_obstacle=hard_obstacle,
        slope_deg=slope_deg,
        traversability=traversability,
        catalog=catalog,
        layer_hashes={
            "height": _array_sha256(height),
            "hard_obstacle": _array_sha256(hard_obstacle),
            "slope": _array_sha256(slope_deg),
            "traversability": _array_sha256(traversability),
        },
    )


def _array_sha256(array: np.ndarray) -> str:
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


def scaled_count_bounds(
    profile: DensityProfile,
    kind: ObjectKind,
    area_m2: float,
) -> tuple[int, int]:
    """按冻结的 4096 平方米基准面积缩放对象数量边界。"""

    if not math.isfinite(area_m2) or area_m2 <= 0.0:
        raise ValueError("area_m2 must be positive and finite")
    base_low, base_high = DENSITY_COUNTS[profile][kind]
    scale = area_m2 / 4096.0
    low = max(1, math.ceil(base_low * scale))
    high = max(low, math.floor(base_high * scale + 0.5))
    return low, high
