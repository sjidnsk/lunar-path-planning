"""Materialize policy-blind Standard scenario evidence for the midterm freeze."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Iterable, Mapping, Sequence
import zipfile

import numpy as np

import xunce_artifact_io as artifact_io


SCHEMA_VERSION = "mid-dual-scenario-source-materialization/v1"
SOURCE_MANIFEST_SCHEMA = "mid-dual-scenario-source-manifest/v1"
APPROVAL_ID = "mid-dual-policy-blind-source-approval-20260727/v1"
ATTESTATION_ID = "mid-dual-policy-blind-source-attestation/v1"
SAFETY_CONTRACT_SOURCE = "mid-dual-policy-blind-safety-contract/v1"
AUTHORIZATION_SHA256 = (
    "720e11ef04ad2b57283421809a077ccf1f0b35167a9f482082241398ad0214d2"
)
DENOMINATOR_SOURCE = "reachable_observable_free_highres_cells/v1"
DENOMINATOR_ALGORITHM = "exact_reachable_safe_pose_range_los/v1"
SPLIT_COUNTS = {"test": 150, "unseen": 64, "validation": 150}
SOURCE_SPLITS = ("test", "unseen", "validation")
SOURCE_SCENARIO_COUNT = sum(SPLIT_COUNTS.values())
SENSOR_RANGE_M = 20.0
SAFETY_CONTRACT = {
    "vehicle_radius_m": 0.4215874761,
    "safety_margin_m": 0.10,
    "min_clearance_m": 0.5215874761,
    "traversability_threshold": 0.50,
    "max_traversable_slope_deg": 30.0,
}
OUTPUT_BASE = Path("D:/xunce/inputs/mid_dual/scenario-sources")
XUNCE_D_ROOT = Path("D:/xunce")
_MASK_PATH_RE = re.compile(r"^masks/([0-9a-f]{64})\.npz$")
_SOURCE_ROOT_TOKEN = "${SOURCE_ROOT}"
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
POOL_HASH_FIELDS = tuple(
    field for field in DESCRIPTOR_FIELDS if field != "source_pool_sha256"
)
FORBIDDEN_DESCRIPTOR_FRAGMENTS = (
    "coverage_result",
    "final_coverage",
    "policy",
    "checkpoint",
    "planner_success",
    "runtime",
    "reward",
)
DATA_FILE_NAMES = (
    "standard-catalog.json",
    "standard-source.json",
    "static-truth-index.jsonl",
    "reset-state-index.jsonl",
    "descriptors.jsonl",
    "reconstruction-index.jsonl",
    "policy-blind-approval.json",
    "source-manifest.json",
)
_RECONSTRUCTION_FIELDS = {
    "scenario_id",
    "scenario_hash",
    "key_sha256",
    "mask_path",
    "mask_file_sha256",
    "mask_size_bytes",
}


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
    return b"".join(_canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_relative_artifact_path(value: str) -> None:
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("scenario source artifact path is not safe and relative")


def _same_resolved_path(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(
        str(Path(right).resolve())
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _reject_existing_reparse_components(path: Path) -> None:
    candidates = [path, *path.parents]
    for candidate in reversed(candidates):
        if not os.path.lexists(artifact_io.windows_safe_path(candidate)):
            continue
        result = os.lstat(artifact_io.windows_safe_path(candidate))
        attributes = int(getattr(result, "st_file_attributes", 0))
        reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
        if os.path.islink(artifact_io.windows_safe_path(candidate)) or (
            reparse_flag and attributes & reparse_flag
        ):
            raise ValueError("execution paths must be absolute D paths without reparse points")


def validate_execution_paths(
    coverage_manifest_path: str | Path,
    output_root: str | Path,
) -> tuple[Path, Path]:
    raw_coverage = Path(coverage_manifest_path)
    raw_output = Path(output_root)
    if (
        not raw_coverage.is_absolute()
        or not raw_output.is_absolute()
        or raw_coverage.drive.casefold() != "d:"
        or raw_output.drive.casefold() != "d:"
        or ".." in raw_coverage.parts
        or ".." in raw_output.parts
    ):
        raise ValueError("execution paths must be absolute D paths inside approved boundaries")
    coverage = raw_coverage.resolve()
    output = raw_output.resolve()
    xunce_root = XUNCE_D_ROOT.resolve()
    approved_output = OUTPUT_BASE.resolve()
    if (
        not _is_within(coverage, xunce_root)
        or not _same_resolved_path(output, approved_output)
    ):
        raise ValueError("execution paths must be absolute D paths inside approved boundaries")
    _reject_existing_reparse_components(coverage)
    _reject_existing_reparse_components(output)
    return coverage, output


def _validate_source_config(payload: bytes) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=_reject_nonfinite)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("scenario source config is invalid JSON") from exc
    expected = {
        "schema_version": SCHEMA_VERSION,
        "catalog_split_counts": SPLIT_COUNTS,
        "excluded_splits": ["train"],
        "coverage_denominator_source": DENOMINATOR_SOURCE,
        "coverage_denominator_algorithm": DENOMINATOR_ALGORITHM,
        "distance_bin_algorithm": "within_split_stable_rank_tertiles/v1",
        "policy_blind_approval_id": APPROVAL_ID,
        "project_authorization_sha256": AUTHORIZATION_SHA256,
        "publication_mode": "data_files_then_completion_manifest/v1",
        "sensor_range_m": SENSOR_RANGE_M,
        "safety_contract_source": SAFETY_CONTRACT_SOURCE,
        "safety_contract": SAFETY_CONTRACT,
    }
    if not isinstance(value, dict) or value != expected:
        raise ValueError("scenario source config contract drifted")
    return value


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite value {value!r}")


def validate_project_authorization(path: str | Path) -> bytes:
    try:
        payload = artifact_io.read_bytes(path)
    except OSError as exc:
        raise ValueError("project authorization is unreadable") from exc
    if _sha256(payload) != AUTHORIZATION_SHA256:
        raise ValueError("project authorization SHA-256 drifted")
    return payload


def bind_source_pool_hashes(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    normalized = [dict(row) for row in rows]
    if any(set(row) != set(DESCRIPTOR_FIELDS) for row in normalized):
        raise ValueError("descriptor field set drifted")
    scenario_ids = [row["scenario_id"] for row in normalized]
    scenario_hashes = [row["scenario_hash"] for row in normalized]
    if (
        any(not isinstance(value, str) or not value for value in scenario_ids)
        or len(scenario_ids) != len(set(scenario_ids))
        or any(not _is_sha256(value) for value in scenario_hashes)
        or len(scenario_hashes) != len(set(scenario_hashes))
        or any(row["source_split"] not in SOURCE_SPLITS for row in normalized)
        or any(
            any(fragment in str(key).lower() for fragment in FORBIDDEN_DESCRIPTOR_FRAGMENTS)
            for row in normalized
            for key in row
        )
    ):
        raise ValueError("descriptor identity or source split drifted")
    pool_hashes: dict[str, str] = {}
    for split in SOURCE_SPLITS:
        split_rows = sorted(
            (row for row in normalized if row["source_split"] == split),
            key=lambda row: str(row["scenario_id"]),
        )
        if not split_rows:
            raise ValueError(f"{split} descriptor source pool is empty")
        basis = [
            {field: row[field] for field in POOL_HASH_FIELDS}
            for row in split_rows
        ]
        pool_hashes[split] = _sha256(_canonical_jsonl_bytes(basis))
    for row in normalized:
        row["source_pool_sha256"] = pool_hashes[str(row["source_split"])]
    return sorted(normalized, key=lambda row: str(row["scenario_id"]))


def assign_distance_bins(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    normalized = [dict(row) for row in rows]
    ids = [row.get("scenario_id") for row in normalized]
    if (
        len(ids) != len(set(ids))
        or any(not isinstance(value, str) or not value for value in ids)
        or any(row.get("source_split") not in SOURCE_SPLITS for row in normalized)
    ):
        raise ValueError("reset distance rows have invalid identity")
    result: dict[str, int] = {}
    for split in SOURCE_SPLITS:
        pool = [row for row in normalized if row["source_split"] == split]
        if not pool:
            continue
        for row in pool:
            value = row.get("max_reset_candidate_distance_m")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError("reset candidate distance is invalid")
        ordered = sorted(
            pool,
            key=lambda row: (
                float(row["max_reset_candidate_distance_m"]),
                str(row["scenario_id"]),
            ),
        )
        for index, row in enumerate(ordered):
            result[str(row["scenario_id"])] = min(2, (3 * index) // len(ordered))
    return result


def serialize_reconstruction_mask(mask: np.ndarray) -> bytes:
    value = np.asarray(mask)
    if value.dtype != np.dtype(bool) or value.ndim != 2 or not np.any(value):
        raise ValueError("coverable reconstruction mask must be a nonempty 2D bool array")
    npy = io.BytesIO()
    np.save(npy, np.ascontiguousarray(value), allow_pickle=False)
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_STORED,
        strict_timestamps=True,
    ) as archive:
        info = zipfile.ZipInfo("coverable_mask.npy", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o600 << 16
        archive.writestr(info, npy.getvalue())
    return output.getvalue()


def build_source_manifest(
    *,
    artifact_refs: Mapping[str, Mapping[str, object]],
    descriptor_generator_path: str,
    descriptor_generator_sha256: str,
    coverage_manifest_path: str,
    coverage_manifest_sha256: str,
    coverage_catalog_sha256: str,
    approval_path: str,
    approval_sha256: str,
    source_pool_hashes: Mapping[str, str],
) -> dict[str, object]:
    expected_refs = {
        "descriptor_catalog",
        "standard_catalog",
        "standard_source",
        "static_truth_cache",
        "reset_state",
    }
    if set(artifact_refs) != expected_refs:
        raise ValueError("source manifest artifact reference set drifted")
    refs: dict[str, dict[str, object]] = {}
    for name in sorted(expected_refs):
        row = dict(artifact_refs[name])
        if (
            set(row) != {"artifact_id", "path", "sha256"}
            or not isinstance(row["artifact_id"], str)
            or not row["artifact_id"]
            or not isinstance(row["path"], str)
            or not row["path"]
            or not _is_sha256(row["sha256"])
        ):
            raise ValueError(f"source manifest {name} artifact reference drifted")
        refs[name] = row
    if (
        not descriptor_generator_path
        or not _is_sha256(descriptor_generator_sha256)
        or not coverage_manifest_path
        or not _is_sha256(coverage_manifest_sha256)
        or not _is_sha256(coverage_catalog_sha256)
        or not approval_path
        or not _is_sha256(approval_sha256)
        or set(source_pool_hashes) != set(SOURCE_SPLITS)
        or any(not _is_sha256(value) for value in source_pool_hashes.values())
        or refs["standard_catalog"]["sha256"] != coverage_catalog_sha256
    ):
        raise ValueError("source manifest binding drifted")
    return {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        **refs,
        "descriptor_generator": {
            "id": "mid-dual-policy-blind-standard-descriptor-generator/v1",
            "version": "1",
            "implementation_path": descriptor_generator_path,
            "implementation_sha256": descriptor_generator_sha256,
        },
        "stage6_coverage_manifest": {
            "path": coverage_manifest_path,
            "sha256": coverage_manifest_sha256,
            "catalog_sha256": coverage_catalog_sha256,
        },
        "policy_blind_attestation": {
            "attestation_id": ATTESTATION_ID,
            "approval_id": APPROVAL_ID,
            "approval_artifact_path": approval_path,
            "approval_artifact_sha256": approval_sha256,
            "forbidden_inputs_used": {
                "policy": False,
                "checkpoint": False,
                "reward": False,
                "runtime": False,
                "result": False,
            },
        },
        "source_pools": {
            split: {"sha256": source_pool_hashes[split]}
            for split in SOURCE_SPLITS
        },
    }


def _read_json_payload(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=_reject_nonfinite)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict) or _canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} is not a canonical JSON object")
    return value


def _read_jsonl_payload(payload: bytes, *, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8 JSONL") from exc
    for line in lines:
        if not line:
            raise ValueError(f"{label} contains a blank JSONL row")
        try:
            value = json.loads(line, parse_constant=_reject_nonfinite)
        except ValueError as exc:
            raise ValueError(f"{label} contains invalid JSONL") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} row must be an object")
        rows.append(value)
    if _canonical_jsonl_bytes(rows) != payload:
        raise ValueError(f"{label} is not canonical JSONL")
    return rows


def _validate_exact_file_sets(
    data_files: Mapping[str, bytes],
    mask_files: Mapping[str, bytes],
) -> None:
    if set(data_files) != set(DATA_FILE_NAMES):
        raise ValueError("exact source data file set is required")
    if len(mask_files) != SOURCE_SCENARIO_COUNT:
        raise ValueError("exact source mask file set must contain 364 files")
    for name, payload in {**data_files, **mask_files}.items():
        _validate_relative_artifact_path(name)
        if type(payload) is not bytes:
            raise TypeError("scenario source completion payload must be exact bytes")
    mask_hashes = []
    for name in mask_files:
        match = _MASK_PATH_RE.fullmatch(name)
        if match is None:
            raise ValueError("exact source mask path set is invalid")
        mask_hashes.append(match.group(1))
    if len(mask_hashes) != len(set(mask_hashes)):
        raise ValueError("exact source mask scenario hashes are not unique")


def _normalized_mask_path(
    value: object,
    *,
    expected_relative: str,
    source_root: Path | None,
) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("reconstruction mask path is invalid")
    if source_root is None:
        if value.replace("\\", "/") != expected_relative:
            raise ValueError("relative reconstruction mask path drifted")
    elif not _same_resolved_path(value, source_root / expected_relative):
        raise ValueError("absolute reconstruction mask path drifted")
    return expected_relative


def _normalize_source_manifest(
    payload: bytes,
    *,
    data_files: Mapping[str, bytes],
    source_root: Path | None,
) -> bytes:
    value = _read_json_payload(payload, label="source manifest")
    artifact_files = {
        "descriptor_catalog": "descriptors.jsonl",
        "standard_catalog": "standard-catalog.json",
        "standard_source": "standard-source.json",
        "static_truth_cache": "static-truth-index.jsonl",
        "reset_state": "reset-state-index.jsonl",
    }
    for key, relative in artifact_files.items():
        ref = value.get(key)
        if (
            not isinstance(ref, dict)
            or set(ref) != {"artifact_id", "path", "sha256"}
            or ref.get("sha256") != _sha256(data_files[relative])
        ):
            raise ValueError(f"source manifest {key} reference drifted")
        expected_path = (
            relative
            if source_root is None
            else str((source_root / relative).resolve())
        )
        if source_root is None:
            path_matches = str(ref["path"]).replace("\\", "/") == relative
        else:
            path_matches = _same_resolved_path(ref["path"], expected_path)
        if not path_matches:
            raise ValueError(f"source manifest {key} path drifted")
        ref["path"] = f"{_SOURCE_ROOT_TOKEN}/{relative}"
    attestation = value.get("policy_blind_attestation")
    if not isinstance(attestation, dict):
        raise ValueError("source manifest policy-blind attestation drifted")
    expected_approval = (
        "policy-blind-approval.json"
        if source_root is None
        else str((source_root / "policy-blind-approval.json").resolve())
    )
    approval_path = attestation.get("approval_artifact_path")
    if source_root is None:
        approval_matches = str(approval_path).replace("\\", "/") == expected_approval
    else:
        approval_matches = _same_resolved_path(str(approval_path), expected_approval)
    if (
        not approval_matches
        or attestation.get("approval_artifact_sha256")
        != _sha256(data_files["policy-blind-approval.json"])
    ):
        raise ValueError("source manifest approval binding drifted")
    attestation["approval_artifact_path"] = (
        f"{_SOURCE_ROOT_TOKEN}/policy-blind-approval.json"
    )
    return _canonical_json_bytes(value)


def _semantic_identity_payloads(
    data_files: Mapping[str, bytes],
    mask_files: Mapping[str, bytes],
    *,
    source_root: Path | None,
) -> dict[str, bytes]:
    _validate_exact_file_sets(data_files, mask_files)
    descriptors = _read_jsonl_payload(
        data_files["descriptors.jsonl"],
        label="descriptor catalog",
    )
    if (
        len(descriptors) != SOURCE_SCENARIO_COUNT
        or any(set(row) != set(DESCRIPTOR_FIELDS) for row in descriptors)
        or {
            split: sum(row.get("source_split") == split for row in descriptors)
            for split in SOURCE_SPLITS
        }
        != SPLIT_COUNTS
        or bind_source_pool_hashes(descriptors) != descriptors
    ):
        raise ValueError("descriptor source set is not the exact 364-row contract")
    descriptor_ids = [str(row["scenario_id"]) for row in descriptors]
    descriptor_hashes = [str(row["scenario_hash"]) for row in descriptors]
    if (
        len(descriptor_ids) != len(set(descriptor_ids))
        or len(descriptor_hashes) != len(set(descriptor_hashes))
    ):
        raise ValueError("descriptor identity set is not unique")

    reconstruction = _read_jsonl_payload(
        data_files["reconstruction-index.jsonl"],
        label="reconstruction index",
    )
    if (
        len(reconstruction) != SOURCE_SCENARIO_COUNT
        or any(set(row) != _RECONSTRUCTION_FIELDS for row in reconstruction)
        or {str(row["scenario_id"]) for row in reconstruction}
        != set(descriptor_ids)
    ):
        raise ValueError("reconstruction rows do not bind descriptors one-to-one")
    descriptor_by_id = {str(row["scenario_id"]): row for row in descriptors}
    normalized_reconstruction: list[dict[str, object]] = []
    seen_masks: set[str] = set()
    for raw_row in reconstruction:
        row = dict(raw_row)
        scenario_id = str(row["scenario_id"])
        descriptor = descriptor_by_id[scenario_id]
        scenario_hash = descriptor["scenario_hash"]
        relative = f"masks/{scenario_hash}.npz"
        if (
            row.get("scenario_hash") != scenario_hash
            or not _is_sha256(row.get("key_sha256"))
            or relative not in mask_files
            or row.get("mask_file_sha256") != _sha256(mask_files[relative])
            or row.get("mask_size_bytes") != len(mask_files[relative])
            or relative in seen_masks
        ):
            raise ValueError("reconstruction mask binding drifted")
        row["mask_path"] = _normalized_mask_path(
            row.get("mask_path"),
            expected_relative=relative,
            source_root=source_root,
        )
        seen_masks.add(relative)
        normalized_reconstruction.append(row)
    if seen_masks != set(mask_files):
        raise ValueError("reconstruction mask set is not exact")

    approval = _read_json_payload(
        data_files["policy-blind-approval.json"],
        label="policy-blind approval",
    )
    if approval.get("formal_gate_pass_approved") is not False:
        raise ValueError("policy-blind approval cannot approve a gate pass")
    standard_source = _read_json_payload(
        data_files["standard-source.json"],
        label="standard source provenance",
    )
    forbidden = standard_source.get("forbidden_inputs_used")
    if (
        not isinstance(forbidden, dict)
        or set(forbidden) != {"policy", "checkpoint", "reward", "runtime", "result"}
        or any(value is not False for value in forbidden.values())
        or "stage6_config" in standard_source
    ):
        raise ValueError("standard source forbidden-input attestation drifted")

    normalized = dict(data_files)
    normalized["reconstruction-index.jsonl"] = _canonical_jsonl_bytes(
        sorted(normalized_reconstruction, key=lambda row: str(row["scenario_id"]))
    )
    normalized["source-manifest.json"] = _normalize_source_manifest(
        data_files["source-manifest.json"],
        data_files=data_files,
        source_root=source_root,
    )
    return normalized


def _file_rows(payloads: Mapping[str, bytes]) -> list[dict[str, object]]:
    return [
        {
            "path": name,
            "sha256": _sha256(payload),
            "size_bytes": len(payload),
        }
        for name, payload in sorted(payloads.items())
    ]


def compute_source_id(
    data_files: Mapping[str, bytes],
    mask_files: Mapping[str, bytes],
    *,
    source_root: Path | None,
) -> str:
    normalized = _semantic_identity_payloads(
        data_files,
        mask_files,
        source_root=source_root,
    )
    semantic_rows = _file_rows({**normalized, **mask_files})
    return _sha256(_canonical_jsonl_bytes(semantic_rows))[:16]


def build_completion_manifest(
    data_files: Mapping[str, bytes],
    mask_files: Mapping[str, bytes],
    *,
    source_root: Path | None = None,
) -> bytes:
    normalized = _semantic_identity_payloads(
        data_files,
        mask_files,
        source_root=source_root,
    )
    semantic_rows = _file_rows({**normalized, **mask_files})
    rows = _file_rows({**data_files, **mask_files})
    semantic_root = _sha256(_canonical_jsonl_bytes(semantic_rows))
    return _canonical_json_bytes(
        {
            "schema_version": SCHEMA_VERSION,
            "completion_status": "complete",
            "publication_mode": "data_files_then_completion_manifest/v1",
            "source_id": semantic_root[:16],
            "semantic_payload_root_sha256": semantic_root,
            "file_count": len(rows),
            "payload_root_sha256": _sha256(_canonical_jsonl_bytes(rows)),
            "files": rows,
        }
    )


def _source_id(completion_manifest: bytes) -> str:
    value = _read_json_payload(
        completion_manifest,
        label="scenario source completion manifest",
    )
    source_id = value.get("source_id")
    if (
        not isinstance(source_id, str)
        or len(source_id) != 16
        or any(character not in "0123456789abcdef" for character in source_id)
    ):
        raise ValueError("scenario source ID is invalid")
    return source_id


def publication_order(
    data_files: Mapping[str, bytes],
    mask_files: Mapping[str, bytes],
) -> tuple[str, ...]:
    _validate_exact_file_sets(data_files, mask_files)
    return (*sorted(data_files), *sorted(mask_files), "manifest.json")


def _write_or_complete_prefix(path: Path, payload: bytes) -> None:
    if artifact_io.path_is_file(path):
        existing = artifact_io.read_bytes(path)
        if existing == payload:
            return
        if not payload.startswith(existing):
            raise ValueError(f"scenario source partial artifact drifted: {path.name}")
    artifact_io.write_bytes(path, payload)
    if artifact_io.read_bytes(path) != payload:
        raise ValueError(f"scenario source artifact write verification failed: {path.name}")


def verify_source_bundle(root: str | Path) -> bool:
    base = Path(root).resolve()
    manifest_path = base / "manifest.json"
    if not artifact_io.path_is_file(manifest_path):
        raise FileNotFoundError("scenario source completion manifest is missing")
    raw = artifact_io.read_bytes(manifest_path)
    value = _read_json_payload(raw, label="scenario source completion manifest")
    required = {
        "schema_version",
        "completion_status",
        "publication_mode",
        "source_id",
        "semantic_payload_root_sha256",
        "file_count",
        "payload_root_sha256",
        "files",
    }
    if (
        set(value) != required
        or value["schema_version"] != SCHEMA_VERSION
        or value["completion_status"] != "complete"
        or value["publication_mode"] != "data_files_then_completion_manifest/v1"
        or not isinstance(value["files"], list)
        or value["source_id"] != base.name
    ):
        raise ValueError("scenario source completion manifest drifted")
    rows = value["files"]
    if value["file_count"] != len(rows):
        raise ValueError("scenario source completion count drifted")
    expected_names = [str(row.get("path")) for row in rows if isinstance(row, dict)]
    if (
        len(expected_names) != len(rows)
        or len(expected_names) != len(set(expected_names))
        or expected_names != sorted(expected_names)
    ):
        raise ValueError("scenario source completion file identity drifted")
    actual_names = artifact_io.list_relative_files(base)
    if set(actual_names) != {*expected_names, "manifest.json"}:
        raise ValueError("scenario source root is not an exact closed file set")
    data_files: dict[str, bytes] = {}
    mask_files: dict[str, bytes] = {}
    for item in rows:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256", "size_bytes"}
            or not isinstance(item["path"], str)
            or not _is_sha256(item["sha256"])
            or type(item["size_bytes"]) is not int
            or item["size_bytes"] < 0
        ):
            raise ValueError("scenario source completion file row drifted")
        _validate_relative_artifact_path(item["path"])
        payload = artifact_io.read_bytes(base / item["path"])
        if len(payload) != item["size_bytes"] or _sha256(payload) != item["sha256"]:
            raise ValueError("scenario source artifact bytes drifted")
        if item["path"] in DATA_FILE_NAMES:
            data_files[item["path"]] = payload
        else:
            mask_files[item["path"]] = payload
    expected = build_completion_manifest(
        data_files,
        mask_files,
        source_root=base,
    )
    if expected != raw:
        raise ValueError("scenario source completion or identity evidence drifted")
    return True


def publish_source_bundle(
    *,
    data_files: Mapping[str, bytes],
    mask_files: Mapping[str, bytes],
    completion_manifest: bytes,
    output_root: str | Path,
    source_id: str | None = None,
) -> Path:
    declared = _source_id(completion_manifest)
    identity = declared if source_id is None else source_id
    if identity != declared:
        raise ValueError("scenario source ID does not match semantic content")
    root = Path(output_root).resolve() / identity
    expected_completion = build_completion_manifest(
        data_files,
        mask_files,
        source_root=root,
    )
    if expected_completion != completion_manifest:
        raise ValueError("scenario source completion manifest does not match payloads")
    order = publication_order(data_files, mask_files)
    expected_payloads = {**data_files, **mask_files, "manifest.json": completion_manifest}
    existing = set(artifact_io.list_relative_files(root))
    if "manifest.json" in existing and artifact_io.read_bytes(root / "manifest.json") == completion_manifest:
        if verify_source_bundle(root):
            raise FileExistsError(f"complete scenario source root already exists: {root}")
    prefix_length = next(
        (
            length
            for length in range(len(order) + 1)
            if existing == set(order[:length])
        ),
        None,
    )
    if prefix_length is None:
        raise ValueError("scenario source root is not a contiguous publication prefix")
    artifact_io.make_dirs(root)
    for name in order:
        _write_or_complete_prefix(root / name, expected_payloads[name])
    if not verify_source_bundle(root):
        raise ValueError("scenario source completion verification failed")
    return root


@dataclass(frozen=True, slots=True)
class _BoundScenarioSource:
    key: str
    bundle: object

    def load(self, key: str) -> object:
        if key != self.key:
            raise KeyError(key)
        return self.bundle


def _catalog_artifact_bytes(catalog: object) -> bytes:
    payload = dict(catalog.to_dict())
    catalog_sha256 = payload.pop("catalog_sha256", None)
    raw = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if catalog_sha256 != getattr(catalog, "sha256", None) or _sha256(raw) != catalog_sha256:
        raise ValueError("Standard catalog canonical artifact identity drifted")
    return raw


def _pose_bin(bundle: object) -> list[int]:
    pose = bundle.start_pose
    geometry = bundle.truth.geometry
    theta = float(pose.theta) % (2.0 * math.pi)
    quadrant = min(3, int(theta // (math.pi / 2.0)))
    return [
        int(pose.cell.x >= geometry.width / 2),
        int(pose.cell.y >= geometry.height / 2),
        quadrant,
    ]


def _reset_probe(
    *,
    bundle: object,
    masks: object,
    safety_contract: object,
    config_sha256: str,
) -> tuple[dict[str, object], object]:
    from lunar_exploration_ppo.env.env import LunarExplorationEnv
    from lunar_exploration_ppo.env.standard_training import StandardEnvSettings

    settings = StandardEnvSettings(
        scenario_key=bundle.scenario_id,
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )
    env = LunarExplorationEnv(
        settings,
        scenario_source=_BoundScenarioSource(bundle.scenario_id, bundle),
        precomputed_coverage_masks=masks,
    )
    try:
        env.reset()
        cells = tuple(env.current_action_set.cells)
        resolution = float(bundle.truth.geometry.resolution_m)
        distances = [
            math.hypot(cell.x - bundle.start_pose.cell.x, cell.y - bundle.start_pose.cell.y)
            * resolution
            for cell in cells
        ]
        endpoints = [
            {"x": int(cell.x), "y": int(cell.y)}
            for cell in cells
        ]
        observed_coverable = int(
            np.count_nonzero(env.observed_state.observed_mask & masks.coverable_mask)
        )
        probe = {
            "scenario_id": bundle.scenario_id,
            "scenario_hash": bundle.scenario_hash,
            "start_pose": {
                "x": int(bundle.start_pose.cell.x),
                "y": int(bundle.start_pose.cell.y),
                "theta": float(bundle.start_pose.theta),
            },
            "start_pose_bin": _pose_bin(bundle),
            "observed_mask_sha256": _sha256(
                np.ascontiguousarray(env.observed_state.observed_mask).tobytes()
            ),
            "initial_observed_coverable_count": observed_coverable,
            "initial_observed_coverable_fraction": (
                observed_coverable / int(masks.coverable_cell_count)
            ),
            "initial_valid_frontier_count": int(env.current_action_set.candidate_count),
            "candidate_endpoints_sha256": _sha256(_canonical_json_bytes(endpoints)),
            "max_reset_candidate_distance_m": max(distances, default=0.0),
            "terminal_reason": str(env.terminal_reason),
        }
        return probe, env.current_observation
    finally:
        env = None


def build_source_bundle_from_paths(
    *,
    config_path: str | Path,
    coverage_manifest_path: str | Path,
    coverage_manifest_sha256: str,
    output_root: str | Path,
) -> tuple[dict[str, bytes], dict[str, bytes], bytes, str]:
    if not _is_sha256(coverage_manifest_sha256):
        raise ValueError("coverage manifest expected SHA-256 is invalid")
    coverage_manifest_path, output_root = validate_execution_paths(
        coverage_manifest_path,
        output_root,
    )
    config_bytes = artifact_io.read_bytes(config_path)
    source_config = _validate_source_config(config_bytes)
    config_sha256 = _sha256(config_bytes)

    from lunar_exploration_ppo.configs.stage6 import SafetyContract
    from lunar_exploration_ppo.env.coverage_cache import Stage6CoverageManifest
    from lunar_exploration_ppo.env.scenario_catalog import StandardScenarioFactory
    from lunar_exploration_ppo.env.standard_training import build_standard_catalog

    repo_root = Path(__file__).resolve().parents[1]
    authorization_path = (
        repo_root
        / ".superpowers"
        / "sdd"
        / "2026-07-26-midterm-dual-gate-experiment"
        / "project-authorization.md"
    ).resolve()
    validate_project_authorization(authorization_path)
    safety_payload = source_config["safety_contract"]
    if not isinstance(safety_payload, Mapping):
        raise ValueError("source safety contract is invalid")
    safety = SafetyContract.from_dict(safety_payload)
    sensor_range_m = source_config["sensor_range_m"]
    if type(sensor_range_m) is not float or sensor_range_m != SENSOR_RANGE_M:
        raise ValueError("source sensor range drifted")
    catalog = build_standard_catalog()
    coverage = Stage6CoverageManifest.load(
        coverage_manifest_path,
        expected_sha256=coverage_manifest_sha256,
    )
    if (
        coverage.catalog_sha256 != catalog.sha256
        or dict(coverage.split_counts)
        != {"train": 700, "validation": 150, "test": 150, "unseen": 64}
    ):
        raise ValueError("coverage manifest does not bind the canonical Standard catalog")

    factory = StandardScenarioFactory(catalog)
    static_rows: list[dict[str, object]] = []
    reset_rows: list[dict[str, object]] = []
    provisional: list[dict[str, object]] = []
    reconstruction_rows: list[dict[str, object]] = []
    mask_files: dict[str, bytes] = {}
    selected_records = sorted(
        (record for record in catalog.records if record.split in SOURCE_SPLITS),
        key=lambda record: record.scenario_id,
    )
    if {
        split: sum(record.split == split for record in selected_records)
        for split in SOURCE_SPLITS
    } != SPLIT_COUNTS:
        raise ValueError("Standard source split counts drifted")

    for record in selected_records:
        bundle = factory.build(record)
        audit = dict(coverage.scenario_audit(bundle.scenario_id))
        if (
            audit["scenario_hash"] != bundle.scenario_hash
            or audit["split"] != record.split
            or audit["algorithm_id"] != DENOMINATOR_ALGORITHM
            or audit["exact"] is not True
        ):
            raise ValueError("Stage6 denominator audit does not bind rebuilt scenario")
        settings = {
            "sensor_range_m": sensor_range_m,
            "min_clearance_m": safety.min_clearance_m,
            "max_slope_deg": safety.max_traversable_slope_deg,
            "traversability_threshold": safety.traversability_threshold,
        }
        masks = coverage.load_masks(bundle, **settings)
        mask_payload = serialize_reconstruction_mask(masks.coverable_mask)
        mask_relative = f"masks/{bundle.scenario_hash}.npz"
        if mask_relative in mask_files:
            raise ValueError("scenario reconstruction mask identity was reused")
        mask_files[mask_relative] = mask_payload
        reconstruction_rows.append(
            {
                "scenario_id": bundle.scenario_id,
                "scenario_hash": bundle.scenario_hash,
                "key_sha256": audit["key_sha256"],
                "mask_path": "",
                "mask_file_sha256": _sha256(mask_payload),
                "mask_size_bytes": len(mask_payload),
                "_mask_relative": mask_relative,
            }
        )
        probe, _observation = _reset_probe(
            bundle=bundle,
            masks=masks,
            safety_contract=safety,
            config_sha256=config_sha256,
        )
        probe["record_scenario_id"] = record.scenario_id
        probe["source_split"] = record.split
        reset_rows.append(probe)
        slope = np.asarray(bundle.truth.slope_deg, dtype=np.float64)
        slope_p90 = float(np.percentile(slope, 90.0, method="linear"))
        obstacle_fraction = float(
            np.mean(np.asarray(bundle.truth.hard_obstacle, dtype=np.float64))
        )
        static_rows.append(
            {
                "scenario_id": bundle.scenario_id,
                "scenario_hash": bundle.scenario_hash,
                "record_scenario_id": record.scenario_id,
                "source_split": record.split,
                "parent_roi": record.parent_roi,
                "density_profile": record.density_profile,
                "catalog_record_sha256": hashlib.sha256(
                    _canonical_json_bytes(asdict(record))
                ).hexdigest(),
                "proxy_catalog_sha256": bundle.proxy_catalog.sha256,
                "proxy_layer_hashes": dict(bundle.proxy_layer_hashes),
                "slope_p90_deg": slope_p90,
                "hard_obstacle_fraction": obstacle_fraction,
                "coverable_mask_sha256": audit["coverable_mask_sha256"],
                "coverable_cell_count": audit["coverable_cell_count"],
            }
        )
        provisional.append(
            {
                "scenario_id": bundle.scenario_id,
                "scenario_hash": bundle.scenario_hash,
                "source_split": record.split,
                "source_pool_sha256": "0" * 64,
                "slope_p90_deg": slope_p90,
                "hard_obstacle_fraction": obstacle_fraction,
                "start_pose_bin": list(probe["start_pose_bin"]),
                "initial_observed_coverable_fraction": float(
                    probe["initial_observed_coverable_fraction"]
                ),
                "initial_valid_frontier_count": int(
                    probe["initial_valid_frontier_count"]
                ),
                "coverable_cell_count": int(audit["coverable_cell_count"]),
                "parent_roi": record.parent_roi,
                "density_profile": record.density_profile,
                "start_to_farthest_candidate_distance_bin": 0,
            }
        )

    distance_bins = assign_distance_bins(reset_rows)
    for row in provisional:
        row["start_to_farthest_candidate_distance_bin"] = distance_bins[
            str(row["scenario_id"])
        ]
    for row in reset_rows:
        row["start_to_farthest_candidate_distance_bin"] = distance_bins[
            str(row["scenario_id"])
        ]
    descriptors = bind_source_pool_hashes(provisional)
    source_pool_hashes = {
        split: next(
            str(row["source_pool_sha256"])
            for row in descriptors
            if row["source_split"] == split
        )
        for split in SOURCE_SPLITS
    }

    catalog_bytes = _catalog_artifact_bytes(catalog)
    standard_source_bytes = _canonical_json_bytes(
        {
            "schema_version": "mid-dual-standard-source-provenance/v1",
            "catalog_sha256": catalog.sha256,
            "catalog_sources": catalog.to_dict()["sources"],
            "source_config": {
                "path": str(Path(config_path).resolve()),
                "sha256": config_sha256,
            },
            "safety_contract_source": SAFETY_CONTRACT_SOURCE,
            "safety_contract": safety.to_dict(),
            "safety_contract_sha256": safety.sha256,
            "sensor_range_m": sensor_range_m,
            "forbidden_inputs_used": {
                "policy": False,
                "checkpoint": False,
                "reward": False,
                "runtime": False,
                "result": False,
            },
        }
    )
    static_bytes = _canonical_jsonl_bytes(
        sorted(static_rows, key=lambda row: str(row["scenario_id"]))
    )
    reset_bytes = _canonical_jsonl_bytes(
        sorted(reset_rows, key=lambda row: str(row["scenario_id"]))
    )
    descriptor_bytes = _canonical_jsonl_bytes(descriptors)

    generator_path = str(Path(__file__).resolve())
    generator_bytes = artifact_io.read_bytes(generator_path)
    generator_sha256 = _sha256(generator_bytes)
    preliminary_reconstruction = [
        {
            key: value
            for key, value in row.items()
            if key not in {"mask_path", "_mask_relative"}
        }
        | {"mask_path": str(row["_mask_relative"])}
        for row in reconstruction_rows
    ]
    preliminary_reconstruction_bytes = _canonical_jsonl_bytes(
        sorted(preliminary_reconstruction, key=lambda row: str(row["scenario_id"]))
    )
    approval = {
        "schema_version": "mid-dual-policy-blind-source-approval/v1",
        "approval_id": APPROVAL_ID,
        "scope": "midterm-reduced-g1-scenario-source-generation/v1",
        "status": "approved_for_policy_blind_source_generation",
        "formal_gate_pass_approved": False,
        "project_authorization": {
            "path": str(authorization_path),
            "sha256": AUTHORIZATION_SHA256,
        },
        "descriptor_generator": {
            "path": generator_path,
            "sha256": generator_sha256,
        },
        "coverage_manifest": {
            "path": str(coverage_manifest_path),
            "sha256": coverage_manifest_sha256,
            "catalog_sha256": catalog.sha256,
        },
        "forbidden_inputs_used": {
            "policy": False,
            "checkpoint": False,
            "reward": False,
            "runtime": False,
            "result": False,
        },
    }
    approval_bytes = _canonical_json_bytes(approval)

    artifact_payloads = {
        "descriptor_catalog": ("descriptors.jsonl", descriptor_bytes),
        "standard_catalog": ("standard-catalog.json", catalog_bytes),
        "standard_source": ("standard-source.json", standard_source_bytes),
        "static_truth_cache": ("static-truth-index.jsonl", static_bytes),
        "reset_state": ("reset-state-index.jsonl", reset_bytes),
    }

    preliminary_refs = {
        name: {
            "artifact_id": f"mid-dual-{name.replace('_', '-')}/v1",
            "path": relative,
            "sha256": _sha256(payload),
        }
        for name, (relative, payload) in artifact_payloads.items()
    }
    preliminary_source_manifest = build_source_manifest(
        artifact_refs=preliminary_refs,
        descriptor_generator_path=generator_path,
        descriptor_generator_sha256=generator_sha256,
        coverage_manifest_path=str(coverage_manifest_path),
        coverage_manifest_sha256=coverage_manifest_sha256,
        coverage_catalog_sha256=catalog.sha256,
        approval_path="policy-blind-approval.json",
        approval_sha256=_sha256(approval_bytes),
        source_pool_hashes=source_pool_hashes,
    )
    preliminary_data_files = {
        "standard-catalog.json": catalog_bytes,
        "standard-source.json": standard_source_bytes,
        "static-truth-index.jsonl": static_bytes,
        "reset-state-index.jsonl": reset_bytes,
        "descriptors.jsonl": descriptor_bytes,
        "reconstruction-index.jsonl": preliminary_reconstruction_bytes,
        "policy-blind-approval.json": approval_bytes,
        "source-manifest.json": _canonical_json_bytes(preliminary_source_manifest),
    }
    source_id = compute_source_id(
        preliminary_data_files,
        mask_files,
        source_root=None,
    )
    root = Path(output_root) / source_id
    reconstruction = [
        {
            key: value
            for key, value in row.items()
            if key != "_mask_relative"
        }
        | {"mask_path": str((root / str(row["_mask_relative"])).resolve())}
        for row in reconstruction_rows
    ]
    reconstruction_bytes = _canonical_jsonl_bytes(
        sorted(reconstruction, key=lambda row: str(row["scenario_id"]))
    )
    refs = {
        name: {
            "artifact_id": f"mid-dual-{name.replace('_', '-')}/v1",
            "path": str((root / relative).resolve()),
            "sha256": _sha256(payload),
        }
        for name, (relative, payload) in artifact_payloads.items()
    }
    source_manifest = build_source_manifest(
        artifact_refs=refs,
        descriptor_generator_path=generator_path,
        descriptor_generator_sha256=generator_sha256,
        coverage_manifest_path=str(coverage_manifest_path),
        coverage_manifest_sha256=coverage_manifest_sha256,
        coverage_catalog_sha256=catalog.sha256,
        approval_path=str((root / "policy-blind-approval.json").resolve()),
        approval_sha256=_sha256(approval_bytes),
        source_pool_hashes=source_pool_hashes,
    )
    data_files = {
        "standard-catalog.json": catalog_bytes,
        "standard-source.json": standard_source_bytes,
        "static-truth-index.jsonl": static_bytes,
        "reset-state-index.jsonl": reset_bytes,
        "descriptors.jsonl": descriptor_bytes,
        "reconstruction-index.jsonl": reconstruction_bytes,
        "policy-blind-approval.json": approval_bytes,
        "source-manifest.json": _canonical_json_bytes(source_manifest),
    }
    if compute_source_id(data_files, mask_files, source_root=root) != source_id:
        raise ValueError("scenario source identity changed after root binding")
    completion = build_completion_manifest(
        data_files,
        mask_files,
        source_root=root,
    )
    return data_files, mask_files, completion, source_id


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--execute" not in raw_argv:
        raise SystemExit("refusing scenario source materialization without --execute")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--coverage-manifest", required=True)
    parser.add_argument("--coverage-manifest-sha256", required=True)
    parser.add_argument(
        "--output-root",
        default="D:/xunce/inputs/mid_dual/scenario-sources",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(raw_argv)
    data_files, mask_files, completion, source_id = build_source_bundle_from_paths(
        config_path=args.config,
        coverage_manifest_path=args.coverage_manifest,
        coverage_manifest_sha256=args.coverage_manifest_sha256,
        output_root=args.output_root,
    )
    publish_source_bundle(
        data_files=data_files,
        mask_files=mask_files,
        completion_manifest=completion,
        output_root=args.output_root,
        source_id=source_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
