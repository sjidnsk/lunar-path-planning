"""Freeze policy-blind midterm scenario cohorts with exact source evidence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import xunce_artifact_io as artifact_io
from lunar_exploration_ppo.env.coverage_cache import Stage6CoverageManifest


SCHEMA_VERSION = "mid-dual-scenario-freeze/v1"
SOURCE_MANIFEST_SCHEMA = "mid-dual-scenario-source-manifest/v1"
RECONSTRUCTION_SCHEMA = "mid-dual-denominator-reconstruction-index/v1"
DENOMINATOR_SOURCE = "reachable_observable_free_highres_cells/v1"
DENOMINATOR_ALGORITHM = "exact_reachable_safe_pose_range_los/v1"
DESCRIPTOR_FIELDS = (
    "scenario_id",
    "scenario_hash",
    "source_split",
    "source_pool_sha256",
    "slope_p90_deg",
    "hard_obstacle_fraction",
    "start_pose_bin",
    "initial_observed_coverable_fraction",
    "initial_valid_frontier_count",
    "coverable_cell_count",
    "parent_roi",
    "density_profile",
    "start_to_farthest_candidate_distance_bin",
)
_POOL_HASH_FIELDS = tuple(field for field in DESCRIPTOR_FIELDS if field != "source_pool_sha256")
NUMERIC_FIELDS = (
    "slope_p90_deg",
    "hard_obstacle_fraction",
    "initial_observed_coverable_fraction",
    "initial_valid_frontier_count",
    "coverable_cell_count",
)
FORBIDDEN_FIELD_FRAGMENTS = (
    "coverage_result",
    "final_coverage",
    "policy",
    "checkpoint",
    "planner_success",
    "runtime",
    "reward",
)
_SOURCE_SPLITS = ("test", "unseen", "validation")
_COHORT_SIZES = {
    "test_q24": 24,
    "test_c24": 24,
    "unseen24": 24,
    "g3_test_q5": 5,
    "g3_unseen5": 5,
    "validation3": 3,
    "replay3": 3,
}
_SOURCE_MANIFEST_FIELDS = {
    "schema_version",
    "descriptor_catalog",
    "standard_catalog",
    "standard_source",
    "static_truth_cache",
    "reset_state",
    "descriptor_generator",
    "stage6_coverage_manifest",
    "policy_blind_attestation",
    "source_pools",
}
_ARTIFACT_REF_FIELDS = {"artifact_id", "path", "sha256"}
_RECONSTRUCTION_FIELDS = {
    "scenario_id",
    "scenario_hash",
    "key_sha256",
    "mask_path",
    "mask_file_sha256",
    "mask_size_bytes",
}
_AUDIT_FIELDS = {
    "scenario_id",
    "scenario_hash",
    "split",
    "entry_path",
    "entry_sha256",
    "entry_size_bytes",
    "key_sha256",
    "key",
    "coverable_mask_sha256",
    "coverable_cell_count",
    "geometry",
    "algorithm_id",
    "exact",
}
_DATA_FILES = (
    "config.json",
    "descriptors.jsonl",
    "source-manifest.json",
    "reconstruction-index.jsonl",
    "denominator-proofs.jsonl",
)
_BUNDLE_FILES = (*_DATA_FILES, "manifest.json")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("canonical JSON serialization failed") from exc


def _canonical_jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def _stored_text_bytes(payload: bytes) -> bytes:
    """Mirror xunce_artifact_io.write_text newline semantics on Windows."""

    return payload.replace(b"\n", b"\r\n") if sys.platform == "win32" else payload


def _text_for_artifact_write(payload: bytes) -> str:
    text = payload.decode("utf-8")
    return text.replace("\r\n", "\n") if sys.platform == "win32" else text


def _read_json_bytes(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=_reject_nonfinite)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _read_jsonl_bytes(payload: bytes, *, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        for line in payload.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line, parse_constant=_reject_nonfinite)
            if not isinstance(row, dict):
                raise ValueError
            rows.append(row)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid JSONL") from exc
    return rows


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite value {value!r}")


def _require_finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite number")
    return float(value)


def _factor_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _same_path(left: object, right: str | Path) -> bool:
    return isinstance(left, str) and Path(left).resolve() == Path(right).resolve()


def extract_policy_blind_descriptor(candidate: Mapping[str, object]) -> dict[str, object]:
    """Normalize a strict static/reset-time descriptor with source identity."""

    if not isinstance(candidate, Mapping):
        raise ValueError("scenario descriptor candidate must be a mapping")
    if set(candidate) != set(DESCRIPTOR_FIELDS):
        missing = sorted(set(DESCRIPTOR_FIELDS) - set(candidate))
        extra = sorted(set(candidate) - set(DESCRIPTOR_FIELDS))
        for key in extra:
            if any(fragment in key.lower() for fragment in FORBIDDEN_FIELD_FRAGMENTS):
                raise ValueError(f"forbidden policy or outcome descriptor field: {key}")
        raise ValueError(f"scenario descriptor schema drifted: missing={missing}, extra={extra}")
    for key in candidate:
        if not isinstance(key, str):
            raise ValueError("scenario descriptor key must be a string")
        if any(fragment in key.lower() for fragment in FORBIDDEN_FIELD_FRAGMENTS):
            raise ValueError(f"forbidden policy or outcome descriptor field: {key}")
    scenario_id = candidate["scenario_id"]
    scenario_hash = candidate["scenario_hash"]
    source_split = candidate["source_split"]
    source_pool_sha256 = candidate["source_pool_sha256"]
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario_id must be non-empty")
    if not _is_sha256(scenario_hash):
        raise ValueError("scenario_hash must be a lowercase SHA-256 digest")
    if source_split not in _SOURCE_SPLITS:
        raise ValueError("source_split must be test, unseen, or validation")
    if not _is_sha256(source_pool_sha256):
        raise ValueError("source_pool_sha256 must be a lowercase SHA-256 digest")
    pose = candidate["start_pose_bin"]
    if (
        not isinstance(pose, (tuple, list))
        or len(pose) != 3
        or any(type(value) is not int for value in pose)
        or pose[0] not in (0, 1)
        or pose[1] not in (0, 1)
        or pose[2] not in (0, 1, 2, 3)
    ):
        raise ValueError("start_pose_bin must be (x_half in {0,1}, y_half in {0,1}, heading_quadrant in {0,1,2,3})")
    frontier_count = candidate["initial_valid_frontier_count"]
    coverable_count = candidate["coverable_cell_count"]
    distance_bin = candidate["start_to_farthest_candidate_distance_bin"]
    if type(frontier_count) is not int or type(coverable_count) is not int or type(distance_bin) is not int:
        raise ValueError("static count and distance fields must be exact integers")
    parent_roi = candidate["parent_roi"]
    density_profile = candidate["density_profile"]
    if not isinstance(parent_roi, str) or not parent_roi or not isinstance(density_profile, str) or not density_profile:
        raise ValueError("parent_roi and density_profile must be non-empty")
    descriptor: dict[str, object] = {
        "scenario_id": scenario_id,
        "scenario_hash": scenario_hash,
        "source_split": source_split,
        "source_pool_sha256": source_pool_sha256,
        "slope_p90_deg": _require_finite_number(candidate["slope_p90_deg"], "slope_p90_deg"),
        "hard_obstacle_fraction": _require_finite_number(candidate["hard_obstacle_fraction"], "hard_obstacle_fraction"),
        "start_pose_bin": list(pose),
        "initial_observed_coverable_fraction": _require_finite_number(
            candidate["initial_observed_coverable_fraction"], "initial_observed_coverable_fraction"
        ),
        "initial_valid_frontier_count": frontier_count,
        "coverable_cell_count": coverable_count,
        "parent_roi": parent_roi,
        "density_profile": density_profile,
        "start_to_farthest_candidate_distance_bin": distance_bin,
    }
    if (
        descriptor["slope_p90_deg"] < 0.0
        or not 0.0 <= float(descriptor["hard_obstacle_fraction"]) <= 1.0
        or not 0.0 <= float(descriptor["initial_observed_coverable_fraction"]) <= 1.0
        or frontier_count < 0
        or coverable_count <= 0
        or distance_bin not in (0, 1, 2)
    ):
        raise ValueError("static descriptor value is outside its exact domain")
    return descriptor


def _pool_hash(rows: Sequence[Mapping[str, object]]) -> str:
    basis = [
        {field: row[field] for field in _POOL_HASH_FIELDS}
        for row in sorted(rows, key=lambda row: str(row["scenario_id"]))
    ]
    return _sha256_bytes(_canonical_jsonl_bytes(basis))


def _validated_descriptors(rows: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], dict[str, str]]:
    descriptors = [extract_policy_blind_descriptor(row) for row in rows]
    scenario_ids = [str(row["scenario_id"]) for row in descriptors]
    scenario_hashes = [str(row["scenario_hash"]) for row in descriptors]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("scenario IDs must be unique across all source pools")
    if len(scenario_hashes) != len(set(scenario_hashes)):
        raise ValueError("scenario hash must be unique across all source pools")
    pools = {
        split: [row for row in descriptors if row["source_split"] == split]
        for split in _SOURCE_SPLITS
    }
    if len(pools["test"]) < 48 or len(pools["unseen"]) < 24 or len(pools["validation"]) < 3:
        raise ValueError("source pools do not contain the required formal cohorts")
    hashes: dict[str, str] = {}
    for split, pool in pools.items():
        digest = _pool_hash(pool)
        if any(row["source_pool_sha256"] != digest for row in pool):
            raise ValueError(f"{split} source pool SHA-256 drifted")
        hashes[split] = digest
    return sorted(descriptors, key=lambda row: str(row["scenario_id"])), hashes


def assign_stable_rank_tertiles(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    """Assign rank tertiles after sorting each numeric field by value then ID."""

    descriptors = [extract_policy_blind_descriptor(row) for row in rows]
    if not descriptors:
        raise ValueError("rank tertiles require descriptors")
    bins = {str(row["scenario_id"]): {} for row in descriptors}
    for field in NUMERIC_FIELDS:
        ordered = sorted(descriptors, key=lambda row: (float(row[field]), str(row["scenario_id"])))
        for index, row in enumerate(ordered):
            bins[str(row["scenario_id"])][field] = min(2, (3 * index) // len(ordered))
    return bins


def _selection_score(
    candidate: Mapping[str, object],
    selected: Sequence[Mapping[str, object]],
    rank_bins: Mapping[str, Mapping[str, int]],
    selection_seed: str,
    *,
    include_distance_bin: bool,
) -> tuple[int, int, int, int, int, str]:
    factor_counts: dict[tuple[str, str], int] = {}
    stratum_counts: dict[tuple[str, ...], int] = {}
    factor_names = [*NUMERIC_FIELDS, "start_pose_bin"]
    if include_distance_bin:
        factor_names.append("start_to_farthest_candidate_distance_bin")
    all_rows = [*selected, candidate]
    for row in all_rows:
        scenario_id = str(row["scenario_id"])
        stratum_values: list[str] = []
        for field in factor_names:
            value: object = rank_bins[scenario_id][field] if field in NUMERIC_FIELDS else row[field]
            encoded = _factor_value(value)
            factor_counts[(field, encoded)] = factor_counts.get((field, encoded), 0) + 1
            stratum_values.append(encoded)
        stratum = tuple(stratum_values)
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
    return (
        max(factor_counts.values()),
        sum(count * count for count in factor_counts.values()),
        sum(count - 1 for count in stratum_counts.values() if count > 1),
        sum(row["parent_roi"] == candidate["parent_roi"] for row in all_rows),
        sum(row["density_profile"] == candidate["density_profile"] for row in all_rows),
        _sha256_text(f"{selection_seed}:{candidate['scenario_id']}"),
    )


def _select(
    candidates: Sequence[Mapping[str, object]],
    *,
    count: int,
    selection_seed: str,
    include_distance_bin: bool = False,
) -> list[str]:
    normalized = [extract_policy_blind_descriptor(row) for row in candidates]
    if len(normalized) < count:
        raise ValueError("not enough policy-blind scenarios to freeze cohort")
    ranks = assign_stable_rank_tertiles(normalized)
    selected: list[dict[str, object]] = []
    remaining = {str(row["scenario_id"]): row for row in normalized}
    while len(selected) < count:
        chosen = min(
            remaining.values(),
            key=lambda row: _selection_score(
                row,
                selected,
                ranks,
                selection_seed,
                include_distance_bin=include_distance_bin,
            ),
        )
        selected.append(chosen)
        del remaining[str(chosen["scenario_id"])]
    return [str(row["scenario_id"]) for row in selected]


def freeze_selection(rows: Sequence[Mapping[str, object]], *, selection_seed: str) -> dict[str, list[str]]:
    """Choose each cohort only from its independently bound source pool."""

    descriptors, _ = _validated_descriptors(rows)
    pools = {
        split: [row for row in descriptors if row["source_split"] == split]
        for split in _SOURCE_SPLITS
    }
    by_id = {str(row["scenario_id"]): row for row in descriptors}
    test_q24 = _select(pools["test"], count=24, selection_seed=f"{selection_seed}:test-q")
    q_ids = set(test_q24)
    remaining_test = [row for row in pools["test"] if str(row["scenario_id"]) not in q_ids]
    test_c24 = _select(remaining_test, count=24, selection_seed=f"{selection_seed}:test-c")
    unseen24 = _select(pools["unseen"], count=24, selection_seed=f"{selection_seed}:unseen")
    validation3 = _select(pools["validation"], count=3, selection_seed=f"{selection_seed}:validation")
    replay3 = _select(
        [by_id[scenario_id] for scenario_id in test_q24],
        count=3,
        selection_seed=f"{selection_seed}:replay",
    )
    g3_test_q5 = _select(
        [by_id[scenario_id] for scenario_id in test_q24],
        count=5,
        selection_seed=f"{selection_seed}:g3-test",
        include_distance_bin=True,
    )
    g3_unseen5 = _select(
        [by_id[scenario_id] for scenario_id in unseen24],
        count=5,
        selection_seed=f"{selection_seed}:g3-unseen",
        include_distance_bin=True,
    )
    return {
        "test_q24": test_q24,
        "test_c24": test_c24,
        "unseen24": unseen24,
        "g3_test_q5": g3_test_q5,
        "g3_unseen5": g3_unseen5,
        "validation3": validation3,
        "replay3": replay3,
    }


def _validate_config(config_bytes: bytes) -> tuple[dict[str, object], bytes]:
    config = _read_json_bytes(config_bytes, label="scenario freeze config")
    required = {
        "schema_version",
        "selection_seed",
        "coverage_denominator_source",
        "coverage_denominator_algorithm",
        "source_manifest_schema",
        "reconstruction_index_schema",
        "publication_mode",
        "cohort_sizes",
    }
    if (
        set(config) != required
        or config.get("schema_version") != SCHEMA_VERSION
        or not isinstance(config.get("selection_seed"), str)
        or not config["selection_seed"]
        or config.get("coverage_denominator_source") != DENOMINATOR_SOURCE
        or config.get("coverage_denominator_algorithm") != DENOMINATOR_ALGORITHM
        or config.get("source_manifest_schema") != SOURCE_MANIFEST_SCHEMA
        or config.get("reconstruction_index_schema") != RECONSTRUCTION_SCHEMA
        or config.get("publication_mode") != "data_files_then_completion_manifest/v1"
        or config.get("cohort_sizes") != _COHORT_SIZES
    ):
        raise ValueError("scenario freeze config contract drifted")
    canonical = _canonical_json_bytes(config)
    return config, canonical


def _validate_file_hash(path: object, digest: object, *, label: str) -> bytes:
    if not isinstance(path, str) or not path or not _is_sha256(digest):
        raise ValueError(f"source manifest {label} artifact reference is invalid")
    try:
        payload = artifact_io.read_bytes(path)
    except OSError as exc:
        raise ValueError(f"source manifest {label} artifact is unreadable") from exc
    if _sha256_bytes(payload) != digest:
        raise ValueError(f"source manifest {label} artifact SHA-256 drifted")
    return payload


def _validate_artifact_ref(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _ARTIFACT_REF_FIELDS:
        raise ValueError(f"source manifest {label} artifact schema drifted")
    if not isinstance(value.get("artifact_id"), str) or not value["artifact_id"]:
        raise ValueError(f"source manifest {label} artifact ID is invalid")
    _validate_file_hash(value.get("path"), value.get("sha256"), label=label)
    return dict(value)


def _validate_source_manifest(
    *,
    source_manifest_bytes: bytes,
    descriptor_catalog_path: str | Path,
    descriptor_catalog_bytes: bytes,
    descriptors: Sequence[Mapping[str, object]],
    source_pool_hashes: Mapping[str, str],
    coverage_manifest_path: str | Path,
    coverage_manifest_sha256: str,
    coverage_catalog_sha256: str,
) -> tuple[dict[str, object], bytes]:
    source = _read_json_bytes(source_manifest_bytes, label="source manifest")
    canonical = _canonical_json_bytes(source)
    if canonical != source_manifest_bytes or set(source) != _SOURCE_MANIFEST_FIELDS or source.get("schema_version") != SOURCE_MANIFEST_SCHEMA:
        raise ValueError("source manifest schema or canonical bytes drifted")
    descriptor_ref = source.get("descriptor_catalog")
    if not isinstance(descriptor_ref, dict) or set(descriptor_ref) != _ARTIFACT_REF_FIELDS:
        raise ValueError("source manifest descriptor catalog schema drifted")
    if (
        not isinstance(descriptor_ref.get("artifact_id"), str)
        or not descriptor_ref["artifact_id"]
        or not _same_path(descriptor_ref.get("path"), descriptor_catalog_path)
        or descriptor_ref.get("sha256") != _sha256_bytes(descriptor_catalog_bytes)
    ):
        raise ValueError("source manifest descriptor catalog SHA-256 drifted")
    for field in ("standard_catalog", "standard_source", "static_truth_cache", "reset_state"):
        _validate_artifact_ref(source.get(field), label=field)
    standard_catalog = source["standard_catalog"]
    assert isinstance(standard_catalog, dict)
    if standard_catalog["sha256"] != coverage_catalog_sha256:
        raise ValueError("source manifest Standard catalog does not bind the Stage6 catalog")
    generator = source.get("descriptor_generator")
    if not isinstance(generator, dict) or set(generator) != {
        "id",
        "version",
        "implementation_path",
        "implementation_sha256",
    }:
        raise ValueError("source manifest descriptor generator schema drifted")
    if (
        not isinstance(generator.get("id"), str)
        or not generator["id"]
        or not isinstance(generator.get("version"), str)
        or not generator["version"]
    ):
        raise ValueError("source manifest descriptor generator identity is invalid")
    _validate_file_hash(
        generator.get("implementation_path"),
        generator.get("implementation_sha256"),
        label="descriptor generator implementation",
    )
    stage6 = source.get("stage6_coverage_manifest")
    if not isinstance(stage6, dict) or set(stage6) != {"path", "sha256", "catalog_sha256"}:
        raise ValueError("source manifest Stage6 coverage binding schema drifted")
    if (
        not _same_path(stage6.get("path"), coverage_manifest_path)
        or stage6.get("sha256") != coverage_manifest_sha256
        or stage6.get("catalog_sha256") != coverage_catalog_sha256
    ):
        raise ValueError("source manifest Stage6 coverage manifest binding drifted")
    attestation = source.get("policy_blind_attestation")
    if not isinstance(attestation, dict) or set(attestation) != {
        "attestation_id",
        "approval_id",
        "approval_artifact_path",
        "approval_artifact_sha256",
        "forbidden_inputs_used",
    }:
        raise ValueError("source manifest policy-blind attestation schema drifted")
    if (
        not isinstance(attestation.get("attestation_id"), str)
        or not attestation["attestation_id"]
        or not isinstance(attestation.get("approval_id"), str)
        or not attestation["approval_id"]
    ):
        raise ValueError("source manifest policy-blind attestation or approval ID is invalid")
    _validate_file_hash(
        attestation.get("approval_artifact_path"),
        attestation.get("approval_artifact_sha256"),
        label="approval",
    )
    forbidden = attestation.get("forbidden_inputs_used")
    if (
        not isinstance(forbidden, dict)
        or set(forbidden) != {"policy", "checkpoint", "reward", "runtime", "result"}
        or any(value is not False for value in forbidden.values())
    ):
        raise ValueError("source manifest policy-blind attestation admits forbidden provenance")
    pools = source.get("source_pools")
    if not isinstance(pools, dict) or set(pools) != set(_SOURCE_SPLITS):
        raise ValueError("source manifest source pool schema drifted")
    for split in _SOURCE_SPLITS:
        value = pools[split]
        if not isinstance(value, dict) or set(value) != {"sha256"} or value["sha256"] != source_pool_hashes[split]:
            raise ValueError(f"source manifest {split} source pool SHA-256 drifted")
    if descriptor_ref["sha256"] != _sha256_bytes(
        _canonical_jsonl_bytes(sorted(descriptors, key=lambda row: str(row["scenario_id"])))
    ):
        raise ValueError("source manifest descriptor catalog content drifted")
    return source, canonical


def prove_denominator_identity(
    audit: Mapping[str, object],
    descriptor: Mapping[str, object],
    reconstruction: Mapping[str, object],
    reconstructed_mask: np.ndarray,
) -> dict[str, object]:
    """Bind one reconstructed mask to a verified Stage6 manifest entry."""

    if set(audit) != _AUDIT_FIELDS:
        raise ValueError("denominator Stage6 audit evidence is incomplete")
    if set(reconstruction) != _RECONSTRUCTION_FIELDS:
        raise ValueError("denominator reconstruction row schema drifted")
    for field in ("scenario_id", "scenario_hash"):
        if descriptor.get(field) != audit.get(field) or reconstruction.get(field) != audit.get(field):
            raise ValueError(f"denominator {field} drift")
    if descriptor.get("source_split") != audit.get("split"):
        raise ValueError("denominator source split drift")
    if descriptor.get("coverable_cell_count") != audit.get("coverable_cell_count"):
        raise ValueError("denominator descriptor count drift")
    if reconstruction.get("key_sha256") != audit.get("key_sha256"):
        raise ValueError("denominator key SHA-256 drift")
    mask = np.asarray(reconstructed_mask)
    geometry = audit.get("geometry")
    if (
        mask.dtype != np.dtype(bool)
        or mask.ndim != 2
        or not isinstance(geometry, Mapping)
        or type(geometry.get("width")) is not int
        or type(geometry.get("height")) is not int
        or mask.shape != (geometry["height"], geometry["width"])
    ):
        raise ValueError("denominator mask shape or geometry drift")
    bytes_hash = _sha256_bytes(np.ascontiguousarray(mask).tobytes())
    count = int(np.count_nonzero(mask))
    if (
        count <= 0
        or count != audit.get("coverable_cell_count")
        or bytes_hash != audit.get("coverable_mask_sha256")
        or audit.get("algorithm_id") != DENOMINATOR_ALGORITHM
        or audit.get("exact") is not True
    ):
        raise ValueError("denominator mask hash, count, or algorithm drift")
    proof = dict(audit)
    proof.update(
        {
            "reconstruction_mask_path": reconstruction["mask_path"],
            "reconstruction_mask_file_sha256": reconstruction["mask_file_sha256"],
            "reconstruction_mask_size_bytes": reconstruction["mask_size_bytes"],
            "coverage_denominator_source": DENOMINATOR_SOURCE,
            "coverage_denominator_algorithm": DENOMINATOR_ALGORITHM,
            "semantic_alias_proven": True,
        }
    )
    return proof


def _validated_reconstruction_rows(
    reconstruction_bytes: bytes,
    descriptors: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], bytes]:
    rows = _read_jsonl_bytes(reconstruction_bytes, label="reconstruction index")
    if not rows:
        raise ValueError("reconstruction index must be non-empty")
    if any(set(row) != _RECONSTRUCTION_FIELDS for row in rows):
        raise ValueError("reconstruction index row schema drifted")
    ids = [row.get("scenario_id") for row in rows]
    descriptor_ids = {row["scenario_id"] for row in descriptors}
    if (
        any(not isinstance(value, str) or not value for value in ids)
        or len(ids) != len(set(ids))
        or set(ids) != descriptor_ids
    ):
        raise ValueError("reconstruction index must bind descriptors one-to-one")
    canonical_rows = sorted((dict(row) for row in rows), key=lambda row: str(row["scenario_id"]))
    for row in canonical_rows:
        if (
            not _is_sha256(row.get("scenario_hash"))
            or not _is_sha256(row.get("key_sha256"))
            or not isinstance(row.get("mask_path"), str)
            or not row["mask_path"]
            or not _is_sha256(row.get("mask_file_sha256"))
            or type(row.get("mask_size_bytes")) is not int
            or row["mask_size_bytes"] <= 0
        ):
            raise ValueError("reconstruction index row value drifted")
    return canonical_rows, _canonical_jsonl_bytes(canonical_rows)


def _load_denominator_proofs(
    *,
    descriptors: Sequence[Mapping[str, object]],
    coverage_manifest_path: str | Path,
    coverage_manifest_sha256: str,
    reconstruction_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    manifest = Stage6CoverageManifest.load(
        coverage_manifest_path,
        expected_sha256=coverage_manifest_sha256,
    )
    reconstruction_by_id = {str(row["scenario_id"]): row for row in reconstruction_rows}
    proofs: list[dict[str, object]] = []
    for descriptor in descriptors:
        scenario_id = str(descriptor["scenario_id"])
        reconstruction = reconstruction_by_id[scenario_id]
        try:
            payload = artifact_io.read_bytes(str(reconstruction["mask_path"]))
        except OSError as exc:
            raise ValueError("reconstruction mask is unreadable") from exc
        if (
            len(payload) != reconstruction["mask_size_bytes"]
            or _sha256_bytes(payload) != reconstruction["mask_file_sha256"]
        ):
            raise ValueError("reconstruction mask file size or SHA-256 drifted")
        try:
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                if tuple(archive.files) != ("coverable_mask",):
                    raise ValueError
                mask = np.array(archive["coverable_mask"], copy=True)
        except (OSError, ValueError) as exc:
            raise ValueError("reconstruction mask archive schema drifted") from exc
        audit = manifest.scenario_audit(scenario_id)
        proofs.append(prove_denominator_identity(audit, descriptor, reconstruction, mask))
    return sorted(proofs, key=lambda row: str(row["scenario_id"]))


@dataclass(frozen=True, slots=True)
class _ValidatedFreezeEvidence:
    config: Mapping[str, object]
    config_bytes: bytes
    descriptors: tuple[Mapping[str, object], ...]
    descriptor_bytes: bytes
    source_manifest: Mapping[str, object]
    source_manifest_bytes: bytes
    coverage_manifest_sha256: str
    coverage_catalog_sha256: str
    reconstruction_rows: tuple[Mapping[str, object], ...]
    reconstruction_bytes: bytes
    denominator_proofs: tuple[Mapping[str, object], ...]
    proof_bytes: bytes
    source_pool_hashes: Mapping[str, str]


def build_frozen_manifest(evidence: _ValidatedFreezeEvidence) -> bytes:
    """Build semantic completion bytes only from fully validated evidence."""

    if not isinstance(evidence, _ValidatedFreezeEvidence) or not evidence.denominator_proofs:
        raise ValueError("validated denominator evidence is required")
    selection = freeze_selection(
        evidence.descriptors,
        selection_seed=str(evidence.config["selection_seed"]),
    )
    source_generator = evidence.source_manifest["descriptor_generator"]
    attestation = evidence.source_manifest["policy_blind_attestation"]
    bundle_payloads = {
        "config.json": evidence.config_bytes,
        "descriptors.jsonl": _stored_text_bytes(evidence.descriptor_bytes),
        "source-manifest.json": evidence.source_manifest_bytes,
        "reconstruction-index.jsonl": _stored_text_bytes(evidence.reconstruction_bytes),
        "denominator-proofs.jsonl": _stored_text_bytes(evidence.proof_bytes),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "completion_status": "complete",
        "config_sha256": _sha256_bytes(evidence.config_bytes),
        "descriptor_catalog_sha256": _sha256_bytes(evidence.descriptor_bytes),
        "source_manifest_sha256": _sha256_bytes(evidence.source_manifest_bytes),
        "stage6_coverage_manifest_sha256": evidence.coverage_manifest_sha256,
        "stage6_catalog_sha256": evidence.coverage_catalog_sha256,
        "reconstruction_index_sha256": _sha256_bytes(bundle_payloads["reconstruction-index.jsonl"]),
        "denominator_proofs_sha256": _sha256_bytes(bundle_payloads["denominator-proofs.jsonl"]),
        "descriptor_generator": source_generator,
        "policy_blind_attestation": attestation,
        "source_pools": dict(evidence.source_pool_hashes),
        "cohort_sizes": dict(_COHORT_SIZES),
        "cohorts": selection,
        "denominator_proofs": list(evidence.denominator_proofs),
        "bundle_files": [
            {"path": name, "sha256": _sha256_bytes(payload), "size_bytes": len(payload)}
            for name, payload in bundle_payloads.items()
        ],
    }
    return _canonical_json_bytes(manifest)


def build_freeze_bundle_from_paths(
    *,
    config_path: str | Path,
    descriptor_catalog_path: str | Path,
    source_manifest_path: str | Path,
    coverage_manifest_path: str | Path,
    coverage_manifest_sha256: str,
    reconstruction_index_path: str | Path,
) -> dict[str, bytes]:
    """Read, validate, and retain the complete reproducible freeze input bundle."""

    if not _is_sha256(coverage_manifest_sha256):
        raise ValueError("Stage6 coverage manifest expected SHA-256 is invalid")
    config, config_bytes = _validate_config(artifact_io.read_bytes(config_path))
    raw_descriptors = _read_jsonl_bytes(
        artifact_io.read_bytes(descriptor_catalog_path),
        label="descriptor catalog",
    )
    descriptors, pool_hashes = _validated_descriptors(raw_descriptors)
    descriptor_bytes = _canonical_jsonl_bytes(descriptors)
    coverage_manifest = Stage6CoverageManifest.load(
        coverage_manifest_path,
        expected_sha256=coverage_manifest_sha256,
    )
    source_manifest, source_manifest_bytes = _validate_source_manifest(
        source_manifest_bytes=artifact_io.read_bytes(source_manifest_path),
        descriptor_catalog_path=descriptor_catalog_path,
        descriptor_catalog_bytes=descriptor_bytes,
        descriptors=descriptors,
        source_pool_hashes=pool_hashes,
        coverage_manifest_path=coverage_manifest_path,
        coverage_manifest_sha256=coverage_manifest_sha256,
        coverage_catalog_sha256=coverage_manifest.catalog_sha256,
    )
    reconstruction_rows, reconstruction_bytes = _validated_reconstruction_rows(
        artifact_io.read_bytes(reconstruction_index_path),
        descriptors,
    )
    proofs = _load_denominator_proofs(
        descriptors=descriptors,
        coverage_manifest_path=coverage_manifest_path,
        coverage_manifest_sha256=coverage_manifest_sha256,
        reconstruction_rows=reconstruction_rows,
    )
    if len(proofs) != len(descriptors):
        raise ValueError("denominator proof set does not bind every descriptor")
    proof_bytes = _canonical_jsonl_bytes(proofs)
    evidence = _ValidatedFreezeEvidence(
        config=config,
        config_bytes=config_bytes,
        descriptors=tuple(descriptors),
        descriptor_bytes=descriptor_bytes,
        source_manifest=source_manifest,
        source_manifest_bytes=source_manifest_bytes,
        coverage_manifest_sha256=coverage_manifest_sha256,
        coverage_catalog_sha256=coverage_manifest.catalog_sha256,
        reconstruction_rows=tuple(reconstruction_rows),
        reconstruction_bytes=reconstruction_bytes,
        denominator_proofs=tuple(proofs),
        proof_bytes=proof_bytes,
        source_pool_hashes=pool_hashes,
    )
    manifest_bytes = build_frozen_manifest(evidence)
    return {
        "config.json": config_bytes,
        "descriptors.jsonl": _stored_text_bytes(descriptor_bytes),
        "source-manifest.json": source_manifest_bytes,
        "reconstruction-index.jsonl": _stored_text_bytes(reconstruction_bytes),
        "denominator-proofs.jsonl": _stored_text_bytes(proof_bytes),
        "manifest.json": manifest_bytes,
    }


def freeze_id(manifest_bytes: bytes) -> str:
    return _sha256_bytes(manifest_bytes)[:16]


def _write_or_resume(path: Path, expected: bytes) -> None:
    if artifact_io.path_is_file(path):
        actual = artifact_io.read_bytes(path)
        if actual == expected:
            return
        if expected.startswith(actual):
            artifact_io.write_text(path, _text_for_artifact_write(expected))
            return
        raise ValueError(f"incomplete freeze artifact drifted: {path.name}")
    artifact_io.write_text(path, _text_for_artifact_write(expected))


def publish_frozen_bundle(bundle: Mapping[str, bytes], *, output_root: str | Path) -> Path:
    """Publish data first and completion manifest last, with strict resumability."""

    if set(bundle) != set(_BUNDLE_FILES) or any(not isinstance(bundle[name], bytes) for name in _BUNDLE_FILES):
        raise ValueError("frozen bundle file set is incomplete")
    root = Path(output_root) / freeze_id(bundle["manifest.json"])
    manifest_path = root / "manifest.json"
    if artifact_io.path_is_file(manifest_path):
        if artifact_io.read_bytes(manifest_path) == bundle["manifest.json"] and verify_frozen_bundle(root):
            raise FileExistsError(f"complete freeze root already exists: {root}")
        if not bundle["manifest.json"].startswith(artifact_io.read_bytes(manifest_path)):
            raise ValueError("incomplete freeze completion evidence drifted")
    artifact_io.make_dirs(root)
    for name in _DATA_FILES:
        _write_or_resume(root / name, bundle[name])
    for name in _DATA_FILES:
        if artifact_io.read_bytes(root / name) != bundle[name]:
            raise ValueError("frozen bundle data verification failed")
    _write_or_resume(manifest_path, bundle["manifest.json"])
    if artifact_io.read_bytes(manifest_path) != bundle["manifest.json"] or not verify_frozen_bundle(root):
        raise ValueError("frozen bundle completion verification failed")
    return root


def verify_frozen_bundle(root: str | Path) -> bool:
    """Independently recalculate source, selection, denominator, and file hashes."""

    base = Path(root)
    manifest_path = base / "manifest.json"
    if not artifact_io.path_is_file(manifest_path):
        raise FileNotFoundError("frozen bundle completion manifest is missing")
    manifest_bytes = artifact_io.read_bytes(manifest_path)
    manifest = _read_json_bytes(manifest_bytes, label="frozen bundle manifest")
    if _canonical_json_bytes(manifest) != manifest_bytes:
        raise ValueError("frozen bundle completion manifest is not canonical")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("completion_status") != "complete":
        raise ValueError("frozen bundle completion manifest is invalid")
    file_rows = manifest.get("bundle_files")
    if not isinstance(file_rows, list) or len(file_rows) != len(_DATA_FILES):
        raise ValueError("frozen bundle file index is incomplete")
    indexed: dict[str, dict[str, object]] = {}
    for row in file_rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "sha256", "size_bytes"}
            or row.get("path") not in _DATA_FILES
            or not _is_sha256(row.get("sha256"))
            or type(row.get("size_bytes")) is not int
            or row["size_bytes"] < 0
            or row["path"] in indexed
        ):
            raise ValueError("frozen bundle file index drifted")
        indexed[str(row["path"])] = row
    if set(indexed) != set(_DATA_FILES):
        raise ValueError("frozen bundle file index is incomplete")
    payloads: dict[str, bytes] = {}
    for name in _DATA_FILES:
        path = base / name
        if not artifact_io.path_is_file(path):
            raise ValueError("frozen bundle data file is missing")
        payload = artifact_io.read_bytes(path)
        row = indexed[name]
        if len(payload) != row["size_bytes"] or _sha256_bytes(payload) != row["sha256"]:
            raise ValueError("frozen bundle data file hash drifted")
        payloads[name] = payload
    config, config_bytes = _validate_config(payloads["config.json"])
    descriptors, pool_hashes = _validated_descriptors(
        _read_jsonl_bytes(payloads["descriptors.jsonl"], label="descriptor bundle")
    )
    descriptor_bytes = _canonical_jsonl_bytes(descriptors)
    if _stored_text_bytes(descriptor_bytes) != payloads["descriptors.jsonl"]:
        raise ValueError("descriptor bundle is not canonical")
    source = _read_json_bytes(payloads["source-manifest.json"], label="source manifest bundle")
    if _canonical_json_bytes(source) != payloads["source-manifest.json"]:
        raise ValueError("source manifest bundle is not canonical")
    stage6 = source.get("stage6_coverage_manifest")
    if not isinstance(stage6, dict):
        raise ValueError("source manifest Stage6 binding is missing")
    coverage_path = stage6.get("path")
    coverage_sha256 = stage6.get("sha256")
    if not isinstance(coverage_path, str) or not _is_sha256(coverage_sha256):
        raise ValueError("source manifest Stage6 binding is invalid")
    coverage_manifest = Stage6CoverageManifest.load(coverage_path, expected_sha256=coverage_sha256)
    source_validated, source_bytes = _validate_source_manifest(
        source_manifest_bytes=payloads["source-manifest.json"],
        descriptor_catalog_path=str(source["descriptor_catalog"]["path"]),
        descriptor_catalog_bytes=descriptor_bytes,
        descriptors=descriptors,
        source_pool_hashes=pool_hashes,
        coverage_manifest_path=coverage_path,
        coverage_manifest_sha256=coverage_sha256,
        coverage_catalog_sha256=coverage_manifest.catalog_sha256,
    )
    reconstruction_rows, reconstruction_bytes = _validated_reconstruction_rows(
        payloads["reconstruction-index.jsonl"],
        descriptors,
    )
    if _stored_text_bytes(reconstruction_bytes) != payloads["reconstruction-index.jsonl"]:
        raise ValueError("reconstruction index bundle is not canonical")
    proofs = _load_denominator_proofs(
        descriptors=descriptors,
        coverage_manifest_path=coverage_path,
        coverage_manifest_sha256=coverage_sha256,
        reconstruction_rows=reconstruction_rows,
    )
    proof_bytes = _canonical_jsonl_bytes(proofs)
    if _stored_text_bytes(proof_bytes) != payloads["denominator-proofs.jsonl"]:
        raise ValueError("denominator proof bundle does not independently recalculate")
    evidence = _ValidatedFreezeEvidence(
        config=config,
        config_bytes=config_bytes,
        descriptors=tuple(descriptors),
        descriptor_bytes=descriptor_bytes,
        source_manifest=source_validated,
        source_manifest_bytes=source_bytes,
        coverage_manifest_sha256=coverage_sha256,
        coverage_catalog_sha256=coverage_manifest.catalog_sha256,
        reconstruction_rows=tuple(reconstruction_rows),
        reconstruction_bytes=reconstruction_bytes,
        denominator_proofs=tuple(proofs),
        proof_bytes=proof_bytes,
        source_pool_hashes=pool_hashes,
    )
    if build_frozen_manifest(evidence) != manifest_bytes:
        raise ValueError("frozen bundle semantic manifest does not independently recalculate")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--execute" not in raw_argv:
        raise SystemExit("refusing to read inputs or write scenario freeze without --execute")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--descriptor-catalog", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--coverage-manifest", required=True)
    parser.add_argument("--coverage-manifest-sha256", required=True)
    parser.add_argument("--reconstruction-index", required=True)
    parser.add_argument("--output-root", default="D:/xunce/inputs/mid_dual/scenarios")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(raw_argv)
    bundle = build_freeze_bundle_from_paths(
        config_path=args.config,
        descriptor_catalog_path=args.descriptor_catalog,
        source_manifest_path=args.source_manifest,
        coverage_manifest_path=args.coverage_manifest,
        coverage_manifest_sha256=args.coverage_manifest_sha256,
        reconstruction_index_path=args.reconstruction_index,
    )
    publish_frozen_bundle(bundle, output_root=args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
