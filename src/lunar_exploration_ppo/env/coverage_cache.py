"""Fail-closed, exact coverable-mask cache entry codec and manifest reader."""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from lunar_exploration_ppo.env.coverage import CoverageMasks
from lunar_exploration_ppo.env.scenario import ScenarioBundle
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.path_security import PathSecurityError, require_plain_path, secure_read_bytes


_KEY_SCHEMA = "exact_coverable_mask_cache_key/v1"
_ENTRY_SCHEMA = "exact_coverable_mask_npz/v1"
_MANIFEST_SCHEMA = "stage6_exact_coverable_cache_manifest/v1"
_MASK_FIELDS = ("safe_free_mask", "reachable_safe_mask", "coverable_mask")
_ENTRY_FIELDS = frozenset((*_MASK_FIELDS, "metadata_utf8"))


class CoverageCacheError(RuntimeError):
    """An exact coverage-cache contract or integrity check failed."""


@dataclass(frozen=True, slots=True, init=False)
class CoverageCacheKey:
    _payload_bytes: bytes
    sha256: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("CoverageCacheKey instances must be constructed with build()")

    @property
    def payload(self) -> Mapping[str, object]:
        """Return an isolated JSON value; callers cannot mutate key identity."""

        value = json.loads(self._payload_bytes.decode("utf-8"))
        assert isinstance(value, dict)
        return value

    @classmethod
    def _from_payload(cls, payload: Mapping[str, object]) -> CoverageCacheKey:
        canonical = ArtifactStore.canonical_json_bytes(payload)
        key = object.__new__(cls)
        object.__setattr__(key, "_payload_bytes", canonical)
        object.__setattr__(key, "sha256", hashlib.sha256(canonical).hexdigest())
        return key

    @classmethod
    def build(
        cls,
        scenario: ScenarioBundle,
        *,
        sensor_range_m: float,
        min_clearance_m: float,
        max_slope_deg: float,
        traversability_threshold: float,
    ) -> CoverageCacheKey:
        if not _is_sha256(scenario.scenario_hash):
            raise ValueError("scenario hash is not a SHA-256 digest")
        values = (
            sensor_range_m,
            min_clearance_m,
            max_slope_deg,
            traversability_threshold,
        )
        if any(type(value) not in {int, float} or not math.isfinite(float(value)) for value in values):
            raise ValueError("coverage cache safety values must be finite numbers")
        geometry = scenario.truth.geometry
        payload: dict[str, object] = {
            "schema_version": _KEY_SCHEMA,
            "scenario_hash": scenario.scenario_hash,
            "start_cell_xy": [scenario.start_pose.cell.x, scenario.start_pose.cell.y],
            "geometry": {
                "width": geometry.width,
                "height": geometry.height,
                "resolution_m": geometry.resolution_m,
                "origin_x_m": geometry.origin.x,
                "origin_y_m": geometry.origin.y,
            },
            "algorithm_id": "exact_reachable_safe_pose_range_los/v1",
            "los_model": "two_dimensional_grid_line_of_sight/v1",
            "sensor_range_m": float(sensor_range_m),
            "min_clearance_m": float(min_clearance_m),
            "max_slope_deg": float(max_slope_deg),
            "traversability_threshold": float(traversability_threshold),
            "cache_format": _ENTRY_SCHEMA,
        }
        return cls._from_payload(payload)


def serialize_coverage_entry(key: CoverageCacheKey, masks: CoverageMasks) -> bytes:
    """Encode one exact mask triple as a self-validating, uncompressed NPZ."""

    arrays = _validated_mask_arrays(masks, key=key)
    algorithm_metadata = dict(masks.metadata)
    _validate_algorithm_metadata(algorithm_metadata, arrays["coverable_mask"])
    metadata = {
        "schema_version": _ENTRY_SCHEMA,
        "key": dict(key.payload),
        "key_sha256": key.sha256,
        "arrays": {
            name: {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": _array_sha256(array),
            }
            for name, array in arrays.items()
        },
        "coverable_cell_count": int(np.count_nonzero(arrays["coverable_mask"])),
        "algorithm_metadata": algorithm_metadata,
    }
    metadata_bytes = ArtifactStore.canonical_json_bytes(metadata)
    destination = io.BytesIO()
    np.savez(
        destination,
        **arrays,
        metadata_utf8=np.frombuffer(metadata_bytes, dtype=np.uint8),
    )
    return destination.getvalue()


def load_coverage_entry(payload: bytes, *, expected_key: CoverageCacheKey) -> CoverageMasks:
    """Load only an entry whose bytes, key, arrays, and metadata are exact."""

    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            members = archive.infolist()
            member_names = tuple(member.filename for member in members)
            expected_names = tuple(f"{field}.npy" for field in (*_MASK_FIELDS, "metadata_utf8"))
            if (
                len(members) != len(expected_names)
                or set(member_names) != set(expected_names)
                or len(set(member_names)) != len(member_names)
                or any(member.compress_type != zipfile.ZIP_STORED for member in members)
            ):
                raise CoverageCacheError("coverage cache entry must be an uncompressed NPZ")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if tuple(archive.files) != _MASK_FIELDS + ("metadata_utf8",):
                raise CoverageCacheError("coverage cache entry field set drifted")
            arrays = {name: np.array(archive[name], copy=True) for name in _MASK_FIELDS}
            metadata_utf8 = np.array(archive["metadata_utf8"], copy=True)
    except CoverageCacheError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise CoverageCacheError("coverage cache entry is unreadable") from exc

    if metadata_utf8.dtype != np.dtype(np.uint8) or metadata_utf8.ndim != 1:
        raise CoverageCacheError("coverage cache entry metadata dtype or shape drifted")
    try:
        metadata_bytes = metadata_utf8.tobytes()
        metadata = json.loads(
            metadata_bytes.decode("utf-8"),
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise CoverageCacheError("coverage cache entry metadata is invalid") from exc
    if not isinstance(metadata, dict) or _canonical_json_bytes(
        metadata,
        error="coverage cache entry metadata is invalid",
    ) != metadata_bytes:
        raise CoverageCacheError("coverage cache entry metadata is not canonical")
    _validate_entry_metadata(metadata, expected_key, arrays)
    algorithm_metadata = metadata["algorithm_metadata"]
    assert isinstance(algorithm_metadata, dict)
    return CoverageMasks(
        safe_free_mask=arrays["safe_free_mask"],
        reachable_safe_mask=arrays["reachable_safe_mask"],
        coverable_mask=arrays["coverable_mask"],
        metadata=algorithm_metadata,
    )


@dataclass(frozen=True, slots=True)
class _ManifestEntry:
    scenario_id: str
    scenario_hash: str
    split: str
    key_sha256: str
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class Stage6CoverageManifest:
    path: Path
    cache_root: Path
    catalog_sha256: str
    split_counts: Mapping[str, int]
    generation_environment: Mapping[str, str]
    integrity: Mapping[str, object]
    entry_set_sha256: str
    entries: tuple[_ManifestEntry, ...]

    @classmethod
    def load(cls, path: str | Path, *, expected_sha256: str) -> Stage6CoverageManifest:
        if not _is_sha256(expected_sha256):
            raise CoverageCacheError("coverage cache manifest SHA-256 is invalid")
        try:
            result = secure_read_bytes(path, label="coverage cache manifest")
        except (OSError, PathSecurityError) as exc:
            raise CoverageCacheError("coverage cache manifest secure read failed") from exc
        if hashlib.sha256(result.payload).hexdigest() != expected_sha256:
            raise CoverageCacheError("coverage cache manifest SHA-256 drifted")
        try:
            value = json.loads(
                result.payload.decode("utf-8"),
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (UnicodeDecodeError, ValueError) as exc:
            raise CoverageCacheError("coverage cache manifest JSON is invalid") from exc
        if not isinstance(value, dict) or _canonical_json_bytes(
            value,
            error="coverage cache manifest JSON is invalid",
        ) != result.payload:
            raise CoverageCacheError("coverage cache manifest is not canonical JSON")
        return cls._from_payload(Path(path), value)

    @classmethod
    def _from_payload(cls, path: Path, value: dict[str, object]) -> Stage6CoverageManifest:
        required = {
            "schema_version",
            "cache_root",
            "catalog_sha256",
            "split_counts",
            "generation_environment",
            "integrity",
            "entry_set_sha256",
            "entries",
        }
        if set(value) != required or value["schema_version"] != _MANIFEST_SCHEMA:
            raise CoverageCacheError("coverage cache manifest schema drifted")
        cache_root_value = value["cache_root"]
        catalog_sha256 = value["catalog_sha256"]
        split_counts = value["split_counts"]
        generation_environment = value["generation_environment"]
        integrity = value["integrity"]
        entry_set_sha256 = value["entry_set_sha256"]
        raw_entries = value["entries"]
        if (
            type(cache_root_value) is not str
            or not _is_sha256(catalog_sha256)
            or not _is_sha256(entry_set_sha256)
            or type(split_counts) is not dict
            or set(split_counts) != {"train", "validation", "test", "unseen"}
            or any(type(count) is not int or count < 0 for count in split_counts.values())
            or not isinstance(raw_entries, list)
        ):
            raise CoverageCacheError("coverage cache manifest value contract drifted")
        if not Path(cache_root_value).is_absolute():
            raise CoverageCacheError("coverage cache root must be an absolute lexical path")
        _validate_generation_environment(generation_environment)
        _validate_manifest_integrity(integrity, entry_count=len(raw_entries))
        try:
            cache_root = require_plain_path(
                cache_root_value,
                leaf_kind="directory",
                label="coverage cache root",
            )
        except (OSError, PathSecurityError) as exc:
            raise CoverageCacheError("coverage cache root is unsafe") from exc
        entries = tuple(_parse_manifest_entry(item, cache_root) for item in raw_entries)
        if tuple(entry.scenario_id for entry in entries) != tuple(sorted(entry.scenario_id for entry in entries)):
            raise CoverageCacheError("coverage cache manifest entries are not scenario-id sorted")
        if any(
            len({getattr(entry, field) for entry in entries}) != len(entries)
            for field in ("scenario_id", "scenario_hash", "key_sha256", "path")
        ):
            raise CoverageCacheError("coverage cache manifest has duplicate identity")
        derived_split_counts = {
            split: sum(entry.split == split for entry in entries)
            for split in ("train", "validation", "test", "unseen")
        }
        if derived_split_counts != split_counts:
            raise CoverageCacheError("coverage cache manifest split counts drifted")
        if hashlib.sha256(
            ArtifactStore.canonical_json_bytes(raw_entries)
        ).hexdigest() != entry_set_sha256:
            raise CoverageCacheError("coverage cache manifest entry-set SHA-256 drifted")
        return cls(
            path=Path(path),
            cache_root=cache_root,
            catalog_sha256=catalog_sha256,
            split_counts=MappingProxyType(dict(split_counts)),
            generation_environment=MappingProxyType(dict(generation_environment)),
            integrity=MappingProxyType(dict(integrity)),
            entry_set_sha256=entry_set_sha256,
            entries=entries,
        )

    def load_masks(
        self,
        scenario: ScenarioBundle,
        *,
        sensor_range_m: float,
        min_clearance_m: float,
        max_slope_deg: float,
        traversability_threshold: float,
    ) -> CoverageMasks:
        key = CoverageCacheKey.build(
            scenario,
            sensor_range_m=sensor_range_m,
            min_clearance_m=min_clearance_m,
            max_slope_deg=max_slope_deg,
            traversability_threshold=traversability_threshold,
        )
        matches = tuple(
            entry
            for entry in self.entries
            if entry.scenario_id == scenario.scenario_id
            and entry.scenario_hash == scenario.scenario_hash
            and entry.key_sha256 == key.sha256
        )
        if len(matches) != 1:
            raise CoverageCacheError("coverage cache manifest has missing or ambiguous exact entry")
        entry = matches[0]
        entry_path = self.cache_root / Path(entry.path)
        try:
            result = secure_read_bytes(
                entry_path,
                base=self.cache_root,
                label="coverage cache entry",
            )
        except (OSError, PathSecurityError) as exc:
            raise CoverageCacheError("coverage cache entry secure read failed") from exc
        if len(result.payload) != entry.size_bytes or hashlib.sha256(result.payload).hexdigest() != entry.sha256:
            raise CoverageCacheError("coverage cache entry size or SHA-256 drifted")
        return load_coverage_entry(result.payload, expected_key=key)

    def scenario_audit(self, scenario_id: str) -> Mapping[str, object]:
        """Return read-only exact-denominator evidence for one manifest scenario."""

        if not isinstance(scenario_id, str) or not scenario_id:
            raise CoverageCacheError("coverage cache scenario audit ID is invalid")
        matches = tuple(entry for entry in self.entries if entry.scenario_id == scenario_id)
        if len(matches) != 1:
            raise CoverageCacheError("coverage cache scenario audit has missing or ambiguous entry")
        entry = matches[0]
        entry_path = self.cache_root / Path(entry.path)
        try:
            result = secure_read_bytes(entry_path, base=self.cache_root, label="coverage cache entry")
        except (OSError, PathSecurityError) as exc:
            raise CoverageCacheError("coverage cache scenario audit secure read failed") from exc
        if len(result.payload) != entry.size_bytes or hashlib.sha256(result.payload).hexdigest() != entry.sha256:
            raise CoverageCacheError("coverage cache scenario audit entry size or SHA-256 drifted")
        try:
            with np.load(io.BytesIO(result.payload), allow_pickle=False) as archive:
                metadata_bytes = np.array(archive["metadata_utf8"], copy=True).tobytes()
            metadata = json.loads(metadata_bytes.decode("utf-8"), parse_constant=_reject_nonfinite_json_constant)
        except (OSError, ValueError, UnicodeDecodeError, KeyError) as exc:
            raise CoverageCacheError("coverage cache scenario audit entry metadata is invalid") from exc
        if not isinstance(metadata, dict) or not isinstance(metadata.get("key"), dict):
            raise CoverageCacheError("coverage cache scenario audit entry metadata drifted")
        key = CoverageCacheKey._from_payload(metadata["key"])
        if key.sha256 != entry.key_sha256:
            raise CoverageCacheError("coverage cache scenario audit key drifted")
        if key.payload.get("scenario_hash") != entry.scenario_hash:
            raise CoverageCacheError("coverage cache scenario audit scenario hash is not bound to the entry key")
        masks = load_coverage_entry(result.payload, expected_key=key)
        algorithm = dict(masks.metadata)
        audit = {
            "scenario_id": entry.scenario_id,
            "scenario_hash": entry.scenario_hash,
            "split": entry.split,
            "entry_path": entry.path,
            "entry_sha256": entry.sha256,
            "entry_size_bytes": entry.size_bytes,
            "key_sha256": key.sha256,
            "key": dict(key.payload),
            "coverable_mask_sha256": _array_sha256(masks.coverable_mask),
            "coverable_cell_count": int(np.count_nonzero(masks.coverable_mask)),
            "geometry": dict(key.payload["geometry"]),
            "algorithm_id": algorithm.get("algorithm_id"),
            "exact": algorithm.get("exact"),
        }
        return MappingProxyType(audit)


def _parse_manifest_entry(value: object, cache_root: Path) -> _ManifestEntry:
    if not isinstance(value, dict) or set(value) != {
        "scenario_id",
        "scenario_hash",
        "split",
        "key_sha256",
        "path",
        "sha256",
        "size_bytes",
    }:
        raise CoverageCacheError("coverage cache manifest entry schema drifted")
    scenario_id = value["scenario_id"]
    scenario_hash = value["scenario_hash"]
    split = value["split"]
    key_sha256 = value["key_sha256"]
    relative_path = value["path"]
    payload_sha256 = value["sha256"]
    size_bytes = value["size_bytes"]
    if (
        type(scenario_id) is not str
        or not scenario_id
        or not _is_sha256(scenario_hash)
            or type(split) is not str
            or split not in {"train", "validation", "test", "unseen"}
        or not _is_sha256(key_sha256)
        or not isinstance(relative_path, str)
        or not _is_sha256(payload_sha256)
        or type(size_bytes) is not int
        or size_bytes < 1
    ):
        raise CoverageCacheError("coverage cache manifest entry value drifted")
    candidate = Path(relative_path)
    expected = Path("entries") / key_sha256[:2] / f"{key_sha256}.npz"
    if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != expected.as_posix():
        raise CoverageCacheError("coverage cache manifest entry path is not canonical")
    try:
        require_plain_path(
            cache_root / candidate,
            base=cache_root,
            allow_missing=True,
            label="coverage cache manifest entry",
        )
    except (OSError, PathSecurityError) as exc:
        raise CoverageCacheError("coverage cache manifest entry path escapes cache root") from exc
    return _ManifestEntry(
        scenario_id=scenario_id,
        scenario_hash=scenario_hash,
        split=split,
        key_sha256=key_sha256,
        path=candidate.as_posix(),
        sha256=payload_sha256,
        size_bytes=size_bytes,
    )


def _validate_generation_environment(value: object) -> None:
    required = {"producer", "python_version", "numpy_version", "platform"}
    if (
        type(value) is not dict
        or set(value) != required
        or value.get("producer") != "stage6_coverage_cache_prewarm/v1"
        or any(type(item) is not str or not item for item in value.values())
    ):
        raise CoverageCacheError("coverage cache manifest generation environment drifted")


def _validate_manifest_integrity(value: object, *, entry_count: int) -> None:
    required = {
        "complete",
        "entry_count",
        "valid_entry_count",
        "missing_entry_count",
        "duplicate_entry_count",
        "corrupt_entry_count",
    }
    if type(value) is not dict or set(value) != required:
        raise CoverageCacheError("coverage cache manifest integrity schema drifted")
    if (
        value["complete"] is not True
        or any(type(value[field]) is not int for field in required - {"complete"})
        or value["entry_count"] <= 0
        or value["entry_count"] != entry_count
        or value["valid_entry_count"] != entry_count
        or any(value[field] != 0 for field in ("missing_entry_count", "duplicate_entry_count", "corrupt_entry_count"))
    ):
        raise CoverageCacheError("coverage cache manifest integrity drifted")


def _validated_mask_arrays(masks: CoverageMasks, *, key: CoverageCacheKey) -> dict[str, np.ndarray]:
    geometry = key.payload["geometry"]
    assert isinstance(geometry, dict)
    expected_shape = (geometry["height"], geometry["width"])
    assert isinstance(expected_shape[0], int) and isinstance(expected_shape[1], int)
    arrays = {name: np.asarray(getattr(masks, name)) for name in _MASK_FIELDS}
    for name, array in arrays.items():
        if array.dtype != np.dtype(bool) or array.shape != expected_shape or array.ndim != 2:
            raise CoverageCacheError(f"coverage cache {name} dtype or shape drifted")
    if int(np.count_nonzero(arrays["coverable_mask"])) <= 0:
        raise CoverageCacheError("coverage cache coverable denominator must be positive")
    return arrays


def _validate_entry_metadata(
    metadata: dict[str, object], expected_key: CoverageCacheKey, arrays: dict[str, np.ndarray]
) -> None:
    expected_fields = {
        "schema_version",
        "key",
        "key_sha256",
        "arrays",
        "coverable_cell_count",
        "algorithm_metadata",
    }
    if set(metadata) != expected_fields or metadata["schema_version"] != _ENTRY_SCHEMA:
        raise CoverageCacheError("coverage cache entry metadata schema drifted")
    entry_key = metadata["key"]
    entry_key_sha256 = metadata["key_sha256"]
    if (
        type(entry_key) is not dict
        or not _is_sha256(entry_key_sha256)
        or hashlib.sha256(ArtifactStore.canonical_json_bytes(entry_key)).hexdigest()
        != entry_key_sha256
        or not _strict_json_equal(entry_key, dict(expected_key.payload))
        or entry_key_sha256 != expected_key.sha256
    ):
        raise CoverageCacheError("coverage cache entry exact key drifted")
    _validated_mask_arrays(
        CoverageMasks(
            safe_free_mask=arrays["safe_free_mask"],
            reachable_safe_mask=arrays["reachable_safe_mask"],
            coverable_mask=arrays["coverable_mask"],
            metadata={},
        ),
        key=expected_key,
    )
    metadata_arrays = metadata["arrays"]
    if not isinstance(metadata_arrays, dict) or set(metadata_arrays) != set(_MASK_FIELDS):
        raise CoverageCacheError("coverage cache entry array metadata drifted")
    for name, array in arrays.items():
        descriptor = metadata_arrays[name]
        expected_descriptor = {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": _array_sha256(array),
        }
        if not _strict_json_equal(descriptor, expected_descriptor):
            raise CoverageCacheError("coverage cache entry array integrity drifted")
    count = metadata["coverable_cell_count"]
    if (
        type(count) is not int
        or count <= 0
        or count != int(np.count_nonzero(arrays["coverable_mask"]))
    ):
        raise CoverageCacheError("coverage cache entry coverable count drifted")
    algorithm_metadata = metadata["algorithm_metadata"]
    if not isinstance(algorithm_metadata, dict):
        raise CoverageCacheError("coverage cache entry algorithm metadata drifted")
    _validate_algorithm_metadata(algorithm_metadata, arrays["coverable_mask"])


def _validate_algorithm_metadata(metadata: Mapping[str, object], coverable_mask: np.ndarray) -> None:
    expected = {
        "algorithm_id": "exact_reachable_safe_pose_range_los/v1",
        "sha256": _array_sha256(coverable_mask),
        "exact": True,
        "precompute_scope": "scenario_reset/v1",
        "coverable_cell_count": int(np.count_nonzero(coverable_mask)),
    }
    if expected["coverable_cell_count"] <= 0 or not _strict_json_equal(dict(metadata), expected):
        raise CoverageCacheError("coverage cache algorithm metadata drifted")


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _reject_nonfinite_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _canonical_json_bytes(value: object, *, error: str) -> bytes:
    try:
        return ArtifactStore.canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise CoverageCacheError(error) from exc


def _strict_json_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return set(actual) == set(expected) and all(
            _strict_json_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _strict_json_equal(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
