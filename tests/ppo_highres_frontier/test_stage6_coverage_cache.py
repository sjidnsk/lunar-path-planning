"""Task 1 contracts for exact, read-only Stage 6 coverable-mask cache entries."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import io
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
import warnings
import zipfile

import numpy as np
import pytest

from lunar_exploration_ppo.configs.stage1 import load_stage1_config
from lunar_exploration_ppo.env.coverage import CoverageMasks, compute_coverage_masks
from lunar_exploration_ppo.env.env import LunarExplorationEnv
from lunar_exploration_ppo.env.scenario import LowResolutionPrior, ScenarioBundle, TruthMap
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry, PoseXYTheta, WorldXY


ROOT = Path(__file__).resolve().parents[2]
STAGE1_CONFIG = ROOT / "configs/ppo_highres_frontier_smoke_v1.json"


class _ScenarioSource:
    def __init__(self, bundle: ScenarioBundle) -> None:
        self.bundle = bundle

    def load(self, key: str) -> ScenarioBundle:
        assert key == "smoke-v1"
        return self.bundle


def _scenario(
    *,
    scenario_hash: str = "a" * 64,
    start: CellXY = CellXY(3, 3),
    width: int = 8,
    origin_x_m: float = -2.0,
    prior_size: int = 32,
) -> ScenarioBundle:
    geometry = GridGeometry(width, 8, 1.0, origin=WorldXY(origin_x_m, 1.5))
    shape = geometry.shape
    hard_obstacle = np.zeros(shape, dtype=bool)
    hard_obstacle[0, 0] = True
    truth = TruthMap(
        geometry=geometry,
        height=np.zeros(shape, dtype=np.float64),
        hard_obstacle=hard_obstacle,
        slope_deg=np.zeros(shape, dtype=np.float64),
        traversability=np.ones(shape, dtype=np.float64),
        provenance={"synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1"},
    )
    prior = LowResolutionPrior(
        channels=np.zeros((7, prior_size, prior_size), dtype=np.float32),
        resolution_m=2.0,
        value_prior_source="constant_neutral/v1",
    )
    return ScenarioBundle(
        scenario_id="cache-test/scenario-0",
        scenario_hash=scenario_hash,
        truth=truth,
        prior=prior,
        start_pose=PoseXYTheta(start, 0.0),
        proxy_catalog=cast(object, None),
        proxy_layer_hashes={},
    )


def _key(module, scenario: ScenarioBundle):
    return module.CoverageCacheKey.build(
        scenario,
        sensor_range_m=3.0,
        min_clearance_m=0.0,
        max_slope_deg=30.0,
        traversability_threshold=0.5,
    )


def _masks(scenario: ScenarioBundle) -> CoverageMasks:
    return compute_coverage_masks(
        scenario.truth,
        scenario.start_pose.cell,
        sensor_range_m=3.0,
        min_clearance_m=0.0,
        max_slope_deg=30.0,
        traversability_threshold=0.5,
    )


def _replace_npz_member(payload: bytes, member: str, value: np.ndarray) -> bytes:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        fields = {name: archive[name] for name in archive.files}
    fields[member] = value
    destination = io.BytesIO()
    np.savez(destination, **fields)
    return destination.getvalue()


def _replace_entry_metadata(payload: bytes, mutate) -> bytes:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        fields = {name: archive[name] for name in archive.files}
    metadata = json.loads(bytes(fields["metadata_utf8"]).decode("utf-8"))
    mutate(metadata)
    fields["metadata_utf8"] = np.frombuffer(
        ArtifactStore.canonical_json_bytes(metadata), dtype=np.uint8
    )
    destination = io.BytesIO()
    np.savez(destination, **fields)
    return destination.getvalue()


def _add_duplicate_npz_member(payload: bytes, member: str) -> bytes:
    destination = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(payload), "r") as source:
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as target:
            for info in source.infolist():
                target.writestr(info, source.read(info.filename))
            duplicate = source.getinfo(f"{member}.npy")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                target.writestr(duplicate, source.read(duplicate.filename))
    return destination.getvalue()


def _manifest_fixture(tmp_path: Path, module, scenario: ScenarioBundle):
    key = _key(module, scenario)
    entry = module.serialize_coverage_entry(key, _masks(scenario))
    cache_root = tmp_path / "cache"
    entry_path = cache_root / "entries" / key.sha256[:2] / f"{key.sha256}.npz"
    entry_path.parent.mkdir(parents=True)
    entry_path.write_bytes(entry)
    relative_path = entry_path.relative_to(cache_root).as_posix()
    entries = [
        {
            "scenario_id": scenario.scenario_id,
            "scenario_hash": scenario.scenario_hash,
            "split": "train",
            "key_sha256": key.sha256,
            "path": relative_path,
            "sha256": hashlib.sha256(entry).hexdigest(),
            "size_bytes": len(entry),
        }
    ]
    manifest = {
        "schema_version": "stage6_exact_coverable_cache_manifest/v1",
        "cache_root": str(cache_root),
        "catalog_sha256": "b" * 64,
        "split_counts": {"train": 1, "validation": 0, "test": 0, "unseen": 0},
        "generation_environment": {
            "producer": "stage6_coverage_cache_prewarm/v1",
            "python_version": "3.12.0",
            "numpy_version": "2.0.0",
            "platform": "unit-test",
        },
        "integrity": {
            "complete": True,
            "entry_count": 1,
            "valid_entry_count": 1,
            "missing_entry_count": 0,
            "duplicate_entry_count": 0,
            "corrupt_entry_count": 0,
        },
        "entry_set_sha256": hashlib.sha256(
            ArtifactStore.canonical_json_bytes(entries)
        ).hexdigest(),
        "entries": entries,
    }
    manifest_path = tmp_path / "coverage-cache-manifest.json"
    manifest_bytes = ArtifactStore.canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    return manifest_path, hashlib.sha256(manifest_bytes).hexdigest(), key, entry_path


def _write_manifest(tmp_path: Path, payload: dict[str, object], *, name: str = "manifest.json") -> tuple[Path, str]:
    raw = ArtifactStore.canonical_json_bytes(payload)
    path = tmp_path / name
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _refresh_manifest_aggregates(payload: dict[str, object]) -> None:
    entries = payload["entries"]
    assert isinstance(entries, list)
    payload["split_counts"] = {
        split: sum(entry["split"] == split for entry in entries)
        for split in ("train", "validation", "test", "unseen")
    }
    integrity = payload["integrity"]
    assert isinstance(integrity, dict)
    integrity["entry_count"] = len(entries)
    integrity["valid_entry_count"] = len(entries)
    payload["entry_set_sha256"] = hashlib.sha256(
        ArtifactStore.canonical_json_bytes(entries)
    ).hexdigest()


def _second_complete_manifest_entry() -> dict[str, object]:
    key_sha256 = "d" * 64
    return {
        "scenario_id": "cache-test/scenario-1",
        "scenario_hash": "c" * 64,
        "split": "validation",
        "key_sha256": key_sha256,
        "path": f"entries/{key_sha256[:2]}/{key_sha256}.npz",
        "sha256": "e" * 64,
        "size_bytes": 1,
    }


def test_cache_key_is_stable_and_binds_scenario_geometry_start_and_safety() -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    scenario = _scenario()

    key = _key(module, scenario)

    assert key == _key(module, scenario)
    assert key.payload == {
        "schema_version": "exact_coverable_mask_cache_key/v1",
        "scenario_hash": scenario.scenario_hash,
        "start_cell_xy": [3, 3],
        "geometry": {
            "width": 8,
            "height": 8,
            "resolution_m": 1.0,
            "origin_x_m": -2.0,
            "origin_y_m": 1.5,
        },
        "algorithm_id": "exact_reachable_safe_pose_range_los/v1",
        "los_model": "two_dimensional_grid_line_of_sight/v1",
        "sensor_range_m": 3.0,
        "min_clearance_m": 0.0,
        "max_slope_deg": 30.0,
        "traversability_threshold": 0.5,
        "cache_format": "exact_coverable_mask_npz/v1",
    }
    assert len(key.sha256) == 64
    assert key != _key(module, _scenario(start=CellXY(4, 3)))
    assert key != _key(module, _scenario(width=9))
    assert key != module.CoverageCacheKey.build(
        scenario,
        sensor_range_m=3.0,
        min_clearance_m=0.0,
        max_slope_deg=29.0,
        traversability_threshold=0.5,
    )


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ((), {}),
        ((object(),), {}),
        ((), {"payload": {}, "sha256": "0" * 64}),
    ],
    ids=["zero-arguments", "positional", "forged-payload"],
)
def test_cache_key_rejects_all_direct_public_construction(args: tuple[object, ...], kwargs: dict[str, object]) -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")

    with pytest.raises(TypeError, match="CoverageCacheKey|construct"):
        module.CoverageCacheKey(*args, **kwargs)


def test_cache_key_payload_is_deeply_immutable_and_identity_bound() -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    key = _key(module, _scenario())
    expected_payload = ArtifactStore.canonical_json_bytes(dict(key.payload))

    exposed_payload = key.payload
    exposed_payload["geometry"]["width"] = 999  # type: ignore[index]

    assert ArtifactStore.canonical_json_bytes(dict(key.payload)) == expected_payload
    assert key.sha256 == hashlib.sha256(expected_payload).hexdigest()


def test_uncompressed_npz_round_trip_is_exact_and_preserves_algorithm_metadata() -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    scenario = _scenario()
    masks = _masks(scenario)
    payload = module.serialize_coverage_entry(_key(module, scenario), masks)

    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        assert set(archive.files) == {
            "safe_free_mask",
            "reachable_safe_mask",
            "coverable_mask",
            "metadata_utf8",
        }
        assert all(not item.compress_type for item in archive.zip.infolist())
        assert archive["metadata_utf8"].dtype == np.dtype("uint8")
    loaded = module.load_coverage_entry(payload, expected_key=_key(module, scenario))

    for field in ("safe_free_mask", "reachable_safe_mask", "coverable_mask"):
        assert np.array_equal(getattr(loaded, field), getattr(masks, field))
        assert getattr(loaded, field).dtype == np.dtype(bool)
    assert loaded.metadata == masks.metadata
    assert loaded.coverable_cell_count == masks.coverable_cell_count


@pytest.mark.parametrize(
    "payload_mutation",
    [
        lambda payload: _add_duplicate_npz_member(payload, "safe_free_mask"),
        lambda payload: _replace_entry_metadata(
            payload,
            lambda metadata: (
                metadata["arrays"]["safe_free_mask"]["shape"].__setitem__(0, 8.0),
                metadata["algorithm_metadata"].__setitem__("exact", 1),
            ),
        ),
    ],
    ids=["duplicate-member", "json-type-drift"],
)
def test_entry_rejects_duplicate_members_and_strict_json_type_drift(payload_mutation) -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    scenario = _scenario()
    payload = module.serialize_coverage_entry(_key(module, scenario), _masks(scenario))

    with pytest.raises(module.CoverageCacheError):
        module.load_coverage_entry(payload_mutation(payload), expected_key=_key(module, scenario))


def test_entry_rejects_nonfinite_json_constant_as_coverage_cache_error() -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    scenario = _scenario()
    payload = module.serialize_coverage_entry(_key(module, scenario), _masks(scenario))
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        fields = {name: archive[name] for name in archive.files}
    metadata = json.loads(bytes(fields["metadata_utf8"]).decode("utf-8"))
    metadata["algorithm_metadata"]["coverable_cell_count"] = float("nan")
    metadata_json = json.dumps(metadata, allow_nan=True, separators=(",", ":"), sort_keys=True)
    fields["metadata_utf8"] = np.frombuffer(metadata_json.encode("utf-8"), dtype=np.uint8)
    destination = io.BytesIO()
    np.savez(destination, **fields)

    with pytest.raises(module.CoverageCacheError, match="metadata.*invalid"):
        module.load_coverage_entry(destination.getvalue(), expected_key=_key(module, scenario))


def test_codec_rejects_zero_coverable_denominator_on_serialize_and_load() -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    scenario = _scenario()
    key = _key(module, scenario)
    zeros = np.zeros(scenario.truth.geometry.shape, dtype=bool)
    zero_metadata = {
        "algorithm_id": "exact_reachable_safe_pose_range_los/v1",
        "sha256": hashlib.sha256(zeros.tobytes()).hexdigest(),
        "exact": True,
        "precompute_scope": "scenario_reset/v1",
        "coverable_cell_count": 0,
    }
    zero_masks = CoverageMasks(
        safe_free_mask=zeros.copy(),
        reachable_safe_mask=zeros.copy(),
        coverable_mask=zeros.copy(),
        metadata=zero_metadata,
    )

    with pytest.raises(module.CoverageCacheError, match="coverable"):
        module.serialize_coverage_entry(key, zero_masks)

    metadata = {
        "schema_version": "exact_coverable_mask_npz/v1",
        "key": dict(key.payload),
        "key_sha256": key.sha256,
        "arrays": {
            name: {
                "dtype": "bool",
                "shape": list(zeros.shape),
                "sha256": hashlib.sha256(zeros.tobytes()).hexdigest(),
            }
            for name in ("safe_free_mask", "reachable_safe_mask", "coverable_mask")
        },
        "coverable_cell_count": 0,
        "algorithm_metadata": zero_metadata,
    }
    destination = io.BytesIO()
    np.savez(
        destination,
        safe_free_mask=zeros,
        reachable_safe_mask=zeros,
        coverable_mask=zeros,
        metadata_utf8=np.frombuffer(ArtifactStore.canonical_json_bytes(metadata), dtype=np.uint8),
    )
    with pytest.raises(module.CoverageCacheError, match="coverable"):
        module.load_coverage_entry(destination.getvalue(), expected_key=key)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda payload: payload[:17], "truncated"),
        (
            lambda payload: _replace_npz_member(
                payload, "safe_free_mask", np.zeros((8, 8), dtype=np.uint8)
            ),
            "wrong dtype",
        ),
        (
            lambda payload: _replace_npz_member(
                payload, "reachable_safe_mask", np.zeros((7, 8), dtype=bool)
            ),
            "wrong shape",
        ),
        (
            lambda payload: _replace_npz_member(
                payload, "coverable_mask", np.zeros((8, 8), dtype=bool)
            ),
            "array hash or count",
        ),
        (
            lambda payload: _replace_npz_member(
                payload, "metadata_utf8", np.array([123], dtype=np.uint8)
            ),
            "metadata",
        ),
    ],
)
def test_entry_load_fails_closed_for_corruption_and_contract_drift(mutation, reason: str) -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    scenario = _scenario()
    payload = module.serialize_coverage_entry(_key(module, scenario), _masks(scenario))

    with pytest.raises(module.CoverageCacheError):
        module.load_coverage_entry(mutation(payload), expected_key=_key(module, scenario))


def test_entry_load_rejects_a_different_exact_key() -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    scenario = _scenario()
    payload = module.serialize_coverage_entry(_key(module, scenario), _masks(scenario))

    with pytest.raises(module.CoverageCacheError, match="key"):
        module.load_coverage_entry(payload, expected_key=_key(module, _scenario(start=CellXY(4, 3))))


def test_manifest_verifies_its_hash_entry_size_hash_and_containment(tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    scenario = _scenario()
    manifest_path, manifest_sha256, _, entry_path = _manifest_fixture(tmp_path, module, scenario)

    manifest = module.Stage6CoverageManifest.load(manifest_path, expected_sha256=manifest_sha256)
    loaded = manifest.load_masks(
        scenario,
        sensor_range_m=3.0,
        min_clearance_m=0.0,
        max_slope_deg=30.0,
        traversability_threshold=0.5,
    )
    assert loaded.metadata == _masks(scenario).metadata
    with pytest.raises(module.CoverageCacheError):
        module.Stage6CoverageManifest.load(manifest_path, expected_sha256="0" * 64)

    entry_path.write_bytes(entry_path.read_bytes() + b"drift")
    with pytest.raises(module.CoverageCacheError):
        manifest.load_masks(
            scenario,
            sensor_range_m=3.0,
            min_clearance_m=0.0,
            max_slope_deg=30.0,
            traversability_threshold=0.5,
        )

    malformed = {
        "schema_version": "stage6_exact_coverable_cache_manifest/v1",
        "cache_root": str(tmp_path / "cache"),
        "catalog_sha256": "b" * 64,
        "split_counts": {"train": 1, "validation": 0, "test": 0, "unseen": 0},
        "entry_set_sha256": "c" * 64,
        "entries": [
            {
                "scenario_id": scenario.scenario_id,
                "scenario_hash": scenario.scenario_hash,
                "key_sha256": "d" * 64,
                "path": "../escape.npz",
                "sha256": "e" * 64,
                "size_bytes": 1,
            }
        ],
    }
    malformed_bytes = ArtifactStore.canonical_json_bytes(malformed)
    malformed_path = tmp_path / "escape-manifest.json"
    malformed_path.write_bytes(malformed_bytes)
    with pytest.raises(module.CoverageCacheError):
        module.Stage6CoverageManifest.load(
            malformed_path,
            expected_sha256=hashlib.sha256(malformed_bytes).hexdigest(),
        )


def test_manifest_rejects_missing_exact_entry(tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    scenario = _scenario()
    manifest_path, manifest_sha256, _, _ = _manifest_fixture(tmp_path, module, scenario)
    manifest = module.Stage6CoverageManifest.load(manifest_path, expected_sha256=manifest_sha256)

    with pytest.raises(module.CoverageCacheError, match="missing"):
        manifest.load_masks(
            _scenario(scenario_hash="f" * 64),
            sensor_range_m=3.0,
            min_clearance_m=0.0,
            max_slope_deg=30.0,
            traversability_threshold=0.5,
        )


def test_manifest_loads_and_exposes_complete_v1_identity(tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    scenario = _scenario()
    manifest_path, manifest_sha256, key, _ = _manifest_fixture(tmp_path, module, scenario)

    manifest = module.Stage6CoverageManifest.load(manifest_path, expected_sha256=manifest_sha256)

    assert manifest.catalog_sha256 == "b" * 64
    assert manifest.split_counts == {"train": 1, "validation": 0, "test": 0, "unseen": 0}
    assert manifest.generation_environment == {
        "producer": "stage6_coverage_cache_prewarm/v1",
        "python_version": "3.12.0",
        "numpy_version": "2.0.0",
        "platform": "unit-test",
    }
    assert manifest.integrity == {
        "complete": True,
        "entry_count": 1,
        "valid_entry_count": 1,
        "missing_entry_count": 0,
        "duplicate_entry_count": 0,
        "corrupt_entry_count": 0,
    }
    assert manifest.entry_set_sha256 == hashlib.sha256(
        ArtifactStore.canonical_json_bytes(
            [
                {
                    "scenario_id": scenario.scenario_id,
                    "scenario_hash": scenario.scenario_hash,
                    "split": "train",
                    "key_sha256": key.sha256,
                    "path": f"entries/{key.sha256[:2]}/{key.sha256}.npz",
                    "sha256": manifest.entries[0].sha256,
                    "size_bytes": manifest.entries[0].size_bytes,
                }
            ]
        )
    ).hexdigest()
    assert manifest.entries[0].split == "train"


def test_complete_manifest_rejects_noncanonical_path_at_path_validation(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    scenario = _scenario()
    manifest_path, manifest_sha256, key, _ = _manifest_fixture(tmp_path, module, scenario)
    assert module.Stage6CoverageManifest.load(manifest_path, expected_sha256=manifest_sha256)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["entries"][0]["path"] = f"entries/{key.sha256[:2]}/../{key.sha256}.npz"
    _refresh_manifest_aggregates(payload)
    mutated_path, mutated_sha256 = _write_manifest(tmp_path, payload, name="noncanonical-path.json")

    with pytest.raises(module.CoverageCacheError, match="path is not canonical"):
        module.Stage6CoverageManifest.load(mutated_path, expected_sha256=mutated_sha256)


def test_complete_manifest_rejects_unsorted_entries_at_sort_validation(tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    scenario = _scenario()
    manifest_path, manifest_sha256, _, _ = _manifest_fixture(tmp_path, module, scenario)
    assert module.Stage6CoverageManifest.load(manifest_path, expected_sha256=manifest_sha256)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = payload["entries"][0]
    payload["entries"] = [_second_complete_manifest_entry(), first]
    _refresh_manifest_aggregates(payload)
    mutated_path, mutated_sha256 = _write_manifest(tmp_path, payload, name="unsorted-entries.json")

    with pytest.raises(module.CoverageCacheError, match="scenario-id sorted"):
        module.Stage6CoverageManifest.load(mutated_path, expected_sha256=mutated_sha256)


@pytest.mark.parametrize("duplicate_field", ["scenario_id", "scenario_hash", "key_sha256", "path"])
def test_complete_manifest_rejects_duplicate_identity_at_identity_validation(
    tmp_path: Path,
    duplicate_field: str,
) -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    scenario = _scenario()
    manifest_path, manifest_sha256, _, _ = _manifest_fixture(tmp_path, module, scenario)
    assert module.Stage6CoverageManifest.load(manifest_path, expected_sha256=manifest_sha256)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = payload["entries"][0]
    second = _second_complete_manifest_entry()
    second[duplicate_field] = first[duplicate_field]
    if duplicate_field in {"key_sha256", "path"}:
        second["key_sha256"] = first["key_sha256"]
        second["path"] = first["path"]
    payload["entries"] = [first, second]
    _refresh_manifest_aggregates(payload)
    mutated_path, mutated_sha256 = _write_manifest(tmp_path, payload, name=f"duplicate-{duplicate_field}.json")

    with pytest.raises(module.CoverageCacheError, match="duplicate identity"):
        module.Stage6CoverageManifest.load(mutated_path, expected_sha256=mutated_sha256)


def test_manifest_rejects_nonfinite_json_constant_as_coverage_cache_error(tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    scenario = _scenario()
    manifest_path, _, _, _ = _manifest_fixture(tmp_path, module, scenario)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["integrity"]["entry_count"] = float("nan")
    raw = json.dumps(payload, allow_nan=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    nonfinite_path = tmp_path / "nonfinite-manifest.json"
    nonfinite_path.write_bytes(raw)

    with pytest.raises(module.CoverageCacheError, match="manifest JSON.*invalid"):
        module.Stage6CoverageManifest.load(
            nonfinite_path,
            expected_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_manifest_rejects_unhashable_split_as_coverage_cache_error(tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    scenario = _scenario()
    manifest_path, manifest_sha256, _, _ = _manifest_fixture(tmp_path, module, scenario)
    assert module.Stage6CoverageManifest.load(manifest_path, expected_sha256=manifest_sha256)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["entries"][0]["split"] = []
    payload["entry_set_sha256"] = hashlib.sha256(
        ArtifactStore.canonical_json_bytes(payload["entries"])
    ).hexdigest()
    malformed_path, malformed_sha256 = _write_manifest(tmp_path, payload, name="list-split.json")

    with pytest.raises(module.CoverageCacheError, match="entry value"):
        module.Stage6CoverageManifest.load(malformed_path, expected_sha256=malformed_sha256)


def test_complete_manifest_rejects_duplicate_identity_type_drift_and_entry_set_drift(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    scenario = _scenario()
    manifest_path, _, _, _ = _manifest_fixture(tmp_path, module, scenario)
    base = json.loads(manifest_path.read_text(encoding="utf-8"))

    duplicate = json.loads(json.dumps(base))
    second = dict(duplicate["entries"][0])
    second.update({"scenario_id": "cache-test/scenario-1", "scenario_hash": "c" * 64})
    duplicate["entries"] = [duplicate["entries"][0], second]
    duplicate["split_counts"]["train"] = 2
    duplicate["integrity"]["entry_count"] = 2
    duplicate["integrity"]["valid_entry_count"] = 2
    duplicate["entry_set_sha256"] = hashlib.sha256(
        ArtifactStore.canonical_json_bytes(duplicate["entries"])
    ).hexdigest()
    duplicate_path, duplicate_sha256 = _write_manifest(tmp_path, duplicate, name="complete-duplicate.json")

    with pytest.raises(module.CoverageCacheError, match="duplicate"):
        module.Stage6CoverageManifest.load(duplicate_path, expected_sha256=duplicate_sha256)

    type_drift = json.loads(json.dumps(base))
    type_drift["integrity"]["complete"] = 1
    type_path, type_sha256 = _write_manifest(tmp_path, type_drift, name="integrity-type-drift.json")
    with pytest.raises(module.CoverageCacheError, match="integrity"):
        module.Stage6CoverageManifest.load(type_path, expected_sha256=type_sha256)

    entry_set_drift = json.loads(json.dumps(base))
    entry_set_drift["entries"][0]["sha256"] = "f" * 64
    entry_path, entry_sha256 = _write_manifest(tmp_path, entry_set_drift, name="entry-set-drift.json")
    with pytest.raises(module.CoverageCacheError, match="entry-set"):
        module.Stage6CoverageManifest.load(entry_path, expected_sha256=entry_sha256)


def test_manifest_rejects_relative_cache_root_before_path_resolution(tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    scenario = _scenario()
    manifest_path, _, _, _ = _manifest_fixture(tmp_path, module, scenario)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["cache_root"] = "."
    manifest_path, manifest_sha256 = _write_manifest(tmp_path, payload, name="relative-root.json")

    with pytest.raises(module.CoverageCacheError, match="absolute"):
        module.Stage6CoverageManifest.load(manifest_path, expected_sha256=manifest_sha256)


def test_manifest_rejects_absolute_entry_path(tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    scenario = _scenario()
    manifest_path, _, _, entry_path = _manifest_fixture(tmp_path, module, scenario)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["entries"][0]["path"] = str(entry_path)
    payload["entry_set_sha256"] = hashlib.sha256(
        ArtifactStore.canonical_json_bytes(payload["entries"])
    ).hexdigest()
    manifest_path, manifest_sha256 = _write_manifest(tmp_path, payload, name="absolute-entry.json")

    with pytest.raises(module.CoverageCacheError, match="path"):
        module.Stage6CoverageManifest.load(manifest_path, expected_sha256=manifest_sha256)


def test_manifest_load_rejects_hard_linked_or_missing_entry(tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    scenario = _scenario()
    manifest_path, manifest_sha256, _, entry_path = _manifest_fixture(tmp_path, module, scenario)
    manifest = module.Stage6CoverageManifest.load(manifest_path, expected_sha256=manifest_sha256)
    os.link(entry_path, tmp_path / "entry-hardlink-alias.npz")

    with pytest.raises(module.CoverageCacheError, match="secure read"):
        manifest.load_masks(
            scenario,
            sensor_range_m=3.0,
            min_clearance_m=0.0,
            max_slope_deg=30.0,
            traversability_threshold=0.5,
        )


def test_manifest_load_rejects_symlink_or_reparse_entry(tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    scenario = _scenario()
    manifest_path, manifest_sha256, _, entry_path = _manifest_fixture(tmp_path, module, scenario)
    external = tmp_path / "external-entry.npz"
    external.write_bytes(entry_path.read_bytes())
    entry_path.unlink()
    try:
        entry_path.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlink/reparse creation is unavailable: {exc}")
    manifest = module.Stage6CoverageManifest.load(manifest_path, expected_sha256=manifest_sha256)

    with pytest.raises(module.CoverageCacheError, match="secure read"):
        manifest.load_masks(
            scenario,
            sensor_range_m=3.0,
            min_clearance_m=0.0,
            max_slope_deg=30.0,
            traversability_threshold=0.5,
        )


def test_manifest_load_rejects_missing_entry_file(tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.coverage_cache")
    scenario = _scenario()
    manifest_path, manifest_sha256, _, entry_path = _manifest_fixture(tmp_path, module, scenario)
    manifest = module.Stage6CoverageManifest.load(manifest_path, expected_sha256=manifest_sha256)
    entry_path.unlink()

    with pytest.raises(module.CoverageCacheError, match="secure read"):
        manifest.load_masks(
            scenario,
            sensor_range_m=3.0,
            min_clearance_m=0.0,
            max_slope_deg=30.0,
            traversability_threshold=0.5,
        )


def test_standard_constructor_rejects_manifest_catalog_binding_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.standard_training")
    manifest = SimpleNamespace(
        catalog_sha256="b" * 64,
        split_counts={"train": 700, "validation": 150, "test": 150, "unseen": 64},
    )
    catalog = SimpleNamespace(sha256="c" * 64, records=())
    monkeypatch.setattr(module.Stage6CoverageManifest, "load", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(module, "_validate_production_catalog", lambda value: None)
    monkeypatch.setattr(module, "_validated_typed_safety_contract", lambda value, **kwargs: object())
    monkeypatch.setattr(module, "StandardScenarioSampler", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "StandardScenarioFactory", lambda *args, **kwargs: object())

    with pytest.raises(ValueError, match="coverage cache manifest.*catalog"):
        module.StandardTrainingEnv(
            catalog,
            split="train",
            sampler_seed=1,
            safety_contract=object(),
            config_sha256="a" * 64,
            coverage_cache_manifest_path="D:/xunce/cache/manifest.json",
            coverage_cache_manifest_sha256="a" * 64,
        )


def test_env_keeps_legacy_compute_without_precomputed_masks_and_never_computes_with_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_module = importlib.import_module("lunar_exploration_ppo.env.env")
    scenario = _scenario(scenario_hash="e" * 64)
    config = load_stage1_config(STAGE1_CONFIG)
    source = _ScenarioSource(scenario)
    original_compute = env_module.compute_coverage_masks
    calls = 0

    def counted_compute(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_compute(*args, **kwargs)

    env_module._COVERAGE_CACHE.clear()
    monkeypatch.setattr(env_module, "compute_coverage_masks", counted_compute)
    legacy = LunarExplorationEnv(config, scenario_source=source)
    assert calls == 1

    precomputed = original_compute(
        scenario.truth,
        scenario.start_pose.cell,
        sensor_range_m=config.sensor_range_m,
        min_clearance_m=config.min_clearance_m,
        max_slope_deg=config.max_traversable_slope_deg,
        traversability_threshold=config.traversability_threshold,
    )

    def forbidden_compute(*args, **kwargs):
        raise AssertionError("precomputed path must not call compute_coverage_masks")

    monkeypatch.setattr(env_module, "compute_coverage_masks", forbidden_compute)
    injected = LunarExplorationEnv(
        config,
        scenario_source=source,
        precomputed_coverage_masks=precomputed,
    )
    assert injected._coverage_masks.metadata == precomputed.metadata
    assert not np.shares_memory(injected._coverage_masks.coverable_mask, precomputed.coverable_mask)
    assert np.array_equal(legacy._coverage_masks.coverable_mask, injected._coverage_masks.coverable_mask)


def test_env_rejects_zero_coverable_precomputed_masks() -> None:
    scenario = _scenario(scenario_hash="e" * 64)
    config = load_stage1_config(STAGE1_CONFIG)
    zeros = np.zeros(scenario.truth.geometry.shape, dtype=bool)
    masks = CoverageMasks(
        safe_free_mask=zeros.copy(),
        reachable_safe_mask=zeros.copy(),
        coverable_mask=zeros.copy(),
        metadata={
            "algorithm_id": "exact_reachable_safe_pose_range_los/v1",
            "sha256": hashlib.sha256(zeros.tobytes()).hexdigest(),
            "exact": True,
            "precompute_scope": "scenario_reset/v1",
            "coverable_cell_count": 0,
        },
    )

    with pytest.raises(ValueError, match="coverable"):
        LunarExplorationEnv(
            config,
            scenario_source=_ScenarioSource(scenario),
            precomputed_coverage_masks=masks,
        )


@pytest.mark.parametrize(
    ("field", "drift"),
    [
        ("exact", lambda masks: 1),
        ("coverable_cell_count", lambda masks: float(masks.coverable_cell_count)),
    ],
    ids=["bool-to-int", "int-to-float"],
)
def test_env_rejects_precomputed_metadata_json_type_drift(field: str, drift) -> None:
    scenario = _scenario(scenario_hash="e" * 64)
    config = load_stage1_config(STAGE1_CONFIG)
    masks = _masks(scenario)
    metadata = dict(masks.metadata)
    metadata[field] = drift(masks)
    drifted_masks = CoverageMasks(
        safe_free_mask=masks.safe_free_mask.copy(),
        reachable_safe_mask=masks.reachable_safe_mask.copy(),
        coverable_mask=masks.coverable_mask.copy(),
        metadata=metadata,
    )

    with pytest.raises(ValueError, match="metadata"):
        LunarExplorationEnv(
            config,
            scenario_source=_ScenarioSource(scenario),
            precomputed_coverage_masks=drifted_masks,
        )


def _assert_observation_exact(left, right) -> None:
    assert type(left) is type(right)
    for left_array, right_array in zip(left.array_fields(), right.array_fields(), strict=True):
        assert left_array.dtype == right_array.dtype
        assert np.array_equal(left_array, right_array)


def _without_planner_timings(diagnostics):
    timing_fields = {
        "run_start_ns",
        "final_end_ns",
        "input_validation_ns",
        "platform_instantiation_ns",
        "search_ns",
        "complete_route_validation_ns",
        "result_assembly_ns",
        "total_ns",
    }
    planner = {
        key: value
        for key, value in diagnostics.planner.items()
        if key not in timing_fields
    }
    return replace(diagnostics, planner=planner)


def test_programmatic_scenario_legacy_and_cache_paths_are_exactly_equivalent() -> None:
    scenario = _scenario(scenario_hash="f" * 64, prior_size=8)
    config = load_stage1_config(STAGE1_CONFIG)
    source = _ScenarioSource(scenario)
    cached_masks = compute_coverage_masks(
        scenario.truth,
        scenario.start_pose.cell,
        sensor_range_m=config.sensor_range_m,
        min_clearance_m=config.min_clearance_m,
        max_slope_deg=config.max_traversable_slope_deg,
        traversability_threshold=config.traversability_threshold,
    )
    legacy = LunarExplorationEnv(config, scenario_source=source)
    cached = LunarExplorationEnv(
        config,
        scenario_source=source,
        precomputed_coverage_masks=cached_masks,
    )
    legacy.observation_builder.local_crop_size = 8
    cached.observation_builder.local_crop_size = 8

    for field in ("safe_free_mask", "reachable_safe_mask", "coverable_mask"):
        assert np.array_equal(getattr(legacy._coverage_masks, field), getattr(cached._coverage_masks, field))
    assert legacy.coverage_metadata == cached.coverage_metadata
    legacy_observation = legacy.reset()
    cached_observation = cached.reset()
    _assert_observation_exact(legacy_observation, cached_observation)
    assert legacy.current_action_set.cells == cached.current_action_set.cells
    assert np.array_equal(legacy.current_action_set.frontier_features, cached.current_action_set.frontier_features)
    assert np.array_equal(legacy.current_action_set.candidate_mask, cached.current_action_set.candidate_mask)

    legacy_action = legacy.select_rule_action(legacy_observation)
    cached_action = cached.select_rule_action(cached_observation)
    assert legacy_action == cached_action
    legacy_step = legacy.step(legacy_action)
    cached_step = cached.step(cached_action)
    _assert_observation_exact(legacy_step.observation, cached_step.observation)
    assert (
        legacy_step.reward,
        legacy_step.done,
        legacy_step.reason,
        legacy_step.coverage_gain_cells,
        legacy_step.coverage_rate,
        legacy_step.trainable,
        legacy_step.terminal,
        legacy_step.bootstrap_value,
        _without_planner_timings(legacy_step.diagnostics),
    ) == (
        cached_step.reward,
        cached_step.done,
        cached_step.reason,
        cached_step.coverage_gain_cells,
        cached_step.coverage_rate,
        cached_step.trainable,
        cached_step.terminal,
        cached_step.bootstrap_value,
        _without_planner_timings(cached_step.diagnostics),
    )


def test_standard_constructor_requires_paired_manifest_parameters_before_other_work() -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.standard_training")

    with pytest.raises(ValueError, match="together"):
        module.StandardTrainingEnv(
            None,
            split="train",
            sampler_seed=1,
            safety_contract=None,
            config_sha256="not-reached",
            coverage_cache_manifest_path="D:/xunce/cache/manifest.json",
        )


def test_standard_bind_loads_manifest_masks_and_injects_them_without_legacy_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("lunar_exploration_ppo.env.standard_training")
    config_module = importlib.import_module("lunar_exploration_ppo.configs.stage6")
    stage6_config = config_module.load_stage6_config(ROOT / "configs/ppo_highres_frontier_stage6_v1.json")
    safety_contract = config_module.SafetyContract.from_stage6_config(stage6_config)
    scenario = _scenario()
    masks = _masks(scenario)
    captured: dict[str, object] = {}

    class FakeManifest:
        def load_masks(self, bundle, **kwargs):
            captured["bundle"] = bundle
            captured["load_kwargs"] = kwargs
            return masks

    class FakeEnv:
        def __init__(self, settings, *, scenario_source, **kwargs) -> None:
            captured["settings"] = settings
            captured["scenario_source"] = scenario_source
            captured.update(kwargs)

    env = object.__new__(module.StandardTrainingEnv)
    env.split = "train"
    env.factory = SimpleNamespace(build=lambda record: scenario)
    env.safety_contract = safety_contract
    env.config_sha256 = "a" * 64
    env._coverage_cache_manifest = FakeManifest()
    env._record = None
    env._bundle = None
    monkeypatch.setattr(module, "LunarExplorationEnv", FakeEnv)

    env._bind_record(SimpleNamespace(split="train", scenario_id=scenario.scenario_id))

    assert captured["bundle"] is scenario
    assert captured["precomputed_coverage_masks"] is masks
    assert captured["load_kwargs"] == {
        "sensor_range_m": 20.0,
        "min_clearance_m": 0.5215874761,
        "max_slope_deg": 30.0,
        "traversability_threshold": 0.5,
    }


def test_policy_and_frontier_public_contracts_have_no_cache_parameter() -> None:
    observation = importlib.import_module("lunar_exploration_ppo.policy.observation")
    frontier = importlib.import_module("lunar_exploration_ppo.env.frontier")

    assert "coverage" not in inspect.signature(observation.ObservationBuilder.build).parameters
    assert "coverage" not in inspect.signature(frontier.FrontierGenerator.extract).parameters
    assert "coverage_cache" not in inspect.getsource(observation.ObservationBuilder)
    assert "coverage_cache" not in inspect.getsource(frontier.FrontierGenerator)
