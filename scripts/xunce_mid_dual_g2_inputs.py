"""G2 independent-input readiness, one-way adaptation, and approval audit.

This module consumes a completed immutable truth bundle.  It deliberately does
not import the truth producer and never generates labels, optima, reachability
truth, or provider outcomes.  Formal eligibility is resolved only from the
current truth bytes, a separate provider identity, and an artifact-bound
approval record.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np

import xunce_artifact_io as artifact_io


REPO_ROOT = Path(__file__).resolve().parents[1]
PATH_PLANNER_SRC = REPO_ROOT / "path-planner" / "src"
if str(PATH_PLANNER_SRC) not in sys.path:
    sys.path.insert(0, str(PATH_PLANNER_SRC))

SCALE_PROFILE = "midterm_reduced_w8x3_update80/v1"
CONFIG_SCHEMA_VERSION = "xunce-mid-dual-g2-planning-time-config/v1"
INPUT_AUDIT_SCHEMA_VERSION = "xunce-mid-dual-g2-input-audit/v1"
EXECUTION_BUNDLE_SCHEMA_VERSION = "xunce-mid-dual-g2-execution-bundle/v1"
PROVIDER_REQUEST_SCHEMA_VERSION = "xunce-mid-dual-g2-provider-request/v1"
CROSSWALK_SCHEMA_VERSION = "xunce-mid-dual-g2-truth-provider-crosswalk/v1"
G3_REPLAY_COHORT_SCHEMA_VERSION = "xunce-mid-dual-g3-replay-cohort/v1"
APPROVAL_SCHEMA_VERSION = "xunce-mid-dual-g2-artifact-bound-approval/v1"
APPROVAL_ARTIFACT_BASE = "D:/xunce/inputs/mid_dual/g2-approvals"
APPROVAL_ARTIFACT_FILE_NAME = "artifact-bound-o2-approval.json"
APPROVAL_ARTIFACT_CONTRACT = {
    "base": APPROVAL_ARTIFACT_BASE,
    "relative_contract": (
        "<input_set_id>/artifact-bound-o2-approval.json"
    ),
    "file_name": APPROVAL_ARTIFACT_FILE_NAME,
    "publication_mode": "separate-no-candidate-mutation/v1",
}
AUTHORIZATION_SHA256 = (
    "720e11ef04ad2b57283421809a077ccf1f0b35167a9f482082241398ad0214d2"
)
G2_PLATFORMS = ("wheel", "legged", "hopper")
EXPECTED_LABELS_PER_PLATFORM = 3334
EXPECTED_OPTIMA_PER_PLATFORM = 1
EXPECTED_REQUEST_COUNTS = {
    ("standard", "reachable"): 23,
    ("standard", "hard_reachable"): 7,
    ("standard", "unreachable"): 3,
    ("kilometer", "reachable"): 6,
    ("kilometer", "hard_reachable"): 2,
    ("kilometer", "unreachable"): 2,
}
EXPECTED_SCALE_COUNTS = {"standard": 33, "kilometer": 10}
EXPECTED_BUNDLE_COUNTS = {
    "primitive_labels": 10002,
    "raw_request_pool": 1056,
    "repeat_mapping": 645,
    "requests": 129,
    "small_map_optima": 3,
}
G3_REPLAY_SELECTION_SEED = 20260727
G3_REPLAY_STRATA = (
    ("standard", "reachable"),
    ("standard", "hard_reachable"),
    ("kilometer", "hard_reachable"),
)
G3_REPLAY_SELECTION_CONTRACT = (
    "input_side_stratified_sha256_rank_seed20260727/v1"
)
GENERIC_HOPPER_CANDIDATE_PARAMETER_SET_ID = "hopper-generic-internal-proxy/v2"
GENERIC_HOPPER_IMPLEMENTATION_PARAMETER_SET_ID = (
    "hopper_generic_internal_computational_simulation_proxy_midterm_g2g3/v1"
)
GATE5B_HOPPER_PARAMETER_SET_ID = "hopper_gate5b_algorithm_fixture/v1"
PATH_PLANNER_RUNTIME_SOURCE_CLOSURE_SCHEMA_VERSION = (
    "xunce-mid-dual-path-planner-runtime-source-closure/v1"
)
ACTIVE_PLATFORMS = ("wheel", "legged", "hopper")
PLATFORM_INVARIANTS = {
    platform: {"max_traversable_slope_deg": 30.0}
    for platform in ACTIVE_PLATFORMS
}

PLATFORM_STACKS: dict[str, dict[str, object]] = {
    "wheel": {
        "profile_id": "scout-mini-wheel-kinematic-sqp/v1",
        "profile_api": "path_planner.v2.profiles.WheelKinematicSQPProfileV2",
        "provider_api": "path_planner.v2.providers.WheelKinematicSQPProviderV2",
        "l2_recheck_api": "path_planner.v2.validation.validate_route_l2",
        "capability_id": "wheel_kinematic_corridor_sqp/v2",
        "simulation_proxy": False,
        "max_traversable_slope_deg": 30.0,
    },
    "legged": {
        "profile_id": "legged-static-crawl-simulation-proxy-midterm/v1",
        "profile_api": "path_planner.v2.profiles.LeggedProfileV2",
        "provider_api": "path_planner.v2.providers.LeggedPrimitiveProviderV2",
        "l2_recheck_api": "path_planner.v2.validation.validate_legged_route_l2",
        "capability_id": "legged_static_crawl_simulation_proxy/v2",
        "simulation_proxy": True,
        "max_traversable_slope_deg": 30.0,
    },
    "hopper": {
        "profile_id": (
            "hopper-generic-internal-computational-simulation-proxy-midterm/v1"
        ),
        "profile_api": (
            "path_planner.v2.hopper_authority."
            "hopper_generic_internal_simulation_proxy_midterm_v1"
        ),
        "provider_api": "path_planner.v2.providers.HopperPrimitiveProviderV2",
        "l2_recheck_api": "path_planner.v2.validation.validate_hopper_route_l2",
        "capability_id": (
            "simulation_proxy_generic_internal_lunar_ballistic/v2"
        ),
        "simulation_proxy": True,
        "max_traversable_slope_deg": 30.0,
    },
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INPUT_SET_ID_RE = re.compile(r"^g2t2-candidate-[0-9a-f]{24}$")
_NODE_ID_RE = re.compile(r"^n:(?P<x>[0-9]+):(?P<y>[0-9]+)$")
_TERRAIN_ARRAY_SPECS = (
    ("height_mm", np.dtype("<i4")),
    ("cell_class", np.dtype("u1")),
    ("known", np.dtype("u1")),
    ("confidence_ppm", np.dtype("<u4")),
)
_PROVIDER_FORBIDDEN_KEYS = {
    "difficulty_class",
    "expected_reason",
    "expected_safe",
    "oracle_reachable",
    "oracle_reason_code",
    "oracle_safe",
    "provider_success",
    "truth_certificate",
    "truth_certificate_sha256",
}


class G2InputContractError(ValueError):
    """A stable fail-closed G2 input-readiness violation."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        self.detail = detail
        message = code if detail is None else f"{code}: {detail}"
        super().__init__(message)


def _fail(code: str, detail: object | None = None) -> None:
    raise G2InputContractError(
        code,
        None if detail is None else str(detail),
    )


def _is_sha256(value: object) -> bool:
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None


def _require_sha256(value: object, code: str) -> str:
    if not _is_sha256(value):
        _fail(code)
    return str(value)


def _require_nonempty(value: object, code: str) -> str:
    if type(value) is not str or not value:
        _fail(code)
    return value


def _require_exact_bool(value: object, code: str) -> bool:
    if type(value) is not bool:
        _fail(code)
    return value


def approval_artifact_path(input_set_id: str) -> Path:
    """Return the only accepted O2 approval path for an immutable input set."""

    if _INPUT_SET_ID_RE.fullmatch(input_set_id) is None:
        _fail("approval_input_set_id_invalid")
    return (
        Path(APPROVAL_ARTIFACT_BASE)
        / input_set_id
        / APPROVAL_ARTIFACT_FILE_NAME
    )


def _validate_canonical_value(value: object, *, path: str = "$") -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value) or (
            value == 0.0 and math.copysign(1.0, value) < 0.0
        ):
            _fail("noncanonical_json_value", path)
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _validate_canonical_value(child, path=f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                _fail("noncanonical_json_key", path)
            _validate_canonical_value(child, path=f"{path}.{key}")
        return
    _fail("noncanonical_json_type", f"{path}:{type(value).__name__}")


def _canonical_json_bytes(value: object) -> bytes:
    _validate_canonical_value(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise G2InputContractError("canonical_json_encoding_failed") from exc


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key", key)
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> object:
    _fail("noncanonical_json_value", token)


def _canonical_loads(payload: bytes) -> object:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except UnicodeDecodeError as exc:
        raise G2InputContractError("artifact_not_utf8") from exc
    except json.JSONDecodeError as exc:
        raise G2InputContractError("artifact_invalid_json") from exc
    _validate_canonical_value(value)
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _domain_hash(domain: str, *parts: bytes) -> str:
    payload = bytearray()
    for part in (domain.encode("utf-8"), *parts):
        payload.extend(struct.pack(">Q", len(part)))
        payload.extend(part)
    return _sha256(bytes(payload))


def build_path_planner_runtime_source_closure(
    *,
    submodule_commit: str,
    dirty_inventory: Sequence[Mapping[str, object]],
    required_sources: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build a canonical exact-byte closure for every Path Planner v2 module."""
    if (
        type(submodule_commit) is not str
        or re.fullmatch(r"[0-9a-f]{40}", submodule_commit) is None
    ):
        _fail("path_planner_runtime_source_commit_invalid")
    inventory: list[dict[str, str]] = []
    seen_inventory: set[str] = set()
    for raw in dirty_inventory:
        if type(raw) is not dict or set(raw) != {"path", "status"}:
            _fail("path_planner_runtime_source_dirty_inventory_invalid")
        path = _safe_relative_path(raw["path"])
        status = raw["status"]
        if (
            type(status) is not str
            or len(status) != 2
            or path in seen_inventory
        ):
            _fail("path_planner_runtime_source_dirty_inventory_invalid")
        seen_inventory.add(path)
        inventory.append({"path": path, "status": status})
    if inventory != sorted(inventory, key=lambda row: row["path"]):
        _fail("path_planner_runtime_source_dirty_inventory_invalid")

    sources: list[dict[str, object]] = []
    seen_sources: set[str] = set()
    for raw in required_sources:
        if (
            type(raw) is not dict
            or set(raw) != {"logical_path", "size_bytes", "sha256"}
        ):
            _fail("path_planner_runtime_source_rows_invalid")
        logical_path = _safe_relative_path(raw["logical_path"])
        size_bytes = raw["size_bytes"]
        if (
            not logical_path.startswith("src/path_planner/v2/")
            or not logical_path.endswith(".py")
            or type(size_bytes) is not int
            or size_bytes < 0
            or logical_path in seen_sources
        ):
            _fail("path_planner_runtime_source_rows_invalid")
        seen_sources.add(logical_path)
        sources.append(
            {
                "logical_path": logical_path,
                "size_bytes": size_bytes,
                "sha256": _require_sha256(
                    raw["sha256"],
                    "path_planner_runtime_source_rows_invalid",
                ),
            }
        )
    if not sources or sources != sorted(
        sources,
        key=lambda row: str(row["logical_path"]),
    ):
        _fail("path_planner_runtime_source_rows_invalid")
    required_layers = {
        "src/path_planner/v2/api.py",
        "src/path_planner/v2/formal_request_codec.py",
        "src/path_planner/v2/profiles.py",
        "src/path_planner/v2/terrain.py",
        "src/path_planner/v2/hopper_authority.py",
        "src/path_planner/v2/hopper_api.py",
        "src/path_planner/v2/validation.py",
        "src/path_planner/v2/hopper_route_validation.py",
    }
    logical_paths = {str(row["logical_path"]) for row in sources}
    if not required_layers.issubset(logical_paths) or not any(
        path.startswith("src/path_planner/v2/providers/")
        for path in logical_paths
    ):
        _fail("path_planner_runtime_source_layers_incomplete")
    core = {
        "schema_version": (
            PATH_PLANNER_RUNTIME_SOURCE_CLOSURE_SCHEMA_VERSION
        ),
        "submodule_commit": submodule_commit,
        "dirty_inventory": inventory,
        "required_sources": sources,
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": PLATFORM_INVARIANTS,
    }
    return {
        **core,
        "path_planner_runtime_source_closure_sha256": _domain_hash(
            PATH_PLANNER_RUNTIME_SOURCE_CLOSURE_SCHEMA_VERSION,
            _canonical_json_bytes(core),
        ),
    }


def _validated_path_planner_runtime_source_closure(
    value: Mapping[str, object],
) -> dict[str, object]:
    if type(value) is not dict:
        _fail("path_planner_runtime_source_closure_invalid")
    try:
        rebuilt = build_path_planner_runtime_source_closure(
            submodule_commit=value["submodule_commit"],
            dirty_inventory=value["dirty_inventory"],
            required_sources=value["required_sources"],
        )
    except (KeyError, TypeError) as exc:
        raise G2InputContractError(
            "path_planner_runtime_source_closure_invalid"
        ) from exc
    if dict(value) != rebuilt:
        _fail("path_planner_runtime_source_closure_invalid")
    return rebuilt


def validate_path_planner_runtime_source_closure(
    approved: Mapping[str, object],
    current: Mapping[str, object],
) -> dict[str, object]:
    """Reject both commit switches and same-commit working-byte drift."""
    approved_closure = _validated_path_planner_runtime_source_closure(
        approved
    )
    current_closure = _validated_path_planner_runtime_source_closure(
        current
    )
    if (
        approved_closure["submodule_commit"]
        != current_closure["submodule_commit"]
    ):
        _fail("path_planner_runtime_source_commit_drift")
    if approved_closure != current_closure:
        _fail("path_planner_runtime_source_dirty_drift")
    return current_closure


def _path_planner_git_output(
    root: Path,
    *args: str,
    binary: bool = False,
) -> bytes | str:
    result = subprocess.run(
        ("git", *args),
        cwd=root,
        check=False,
        capture_output=True,
        text=not binary,
        encoding=None if binary else "utf-8",
    )
    if result.returncode != 0:
        _fail("path_planner_runtime_source_git_probe_failed")
    return result.stdout


def capture_path_planner_runtime_source_closure(
    path_planner_root: str | Path | None = None,
) -> dict[str, object]:
    """Capture current commit, full dirty inventory, and all v2 Python bytes."""
    root = Path(path_planner_root or (REPO_ROOT / "path-planner")).resolve()
    source_root = root / "src" / "path_planner" / "v2"
    if not artifact_io.path_is_dir(source_root):
        _fail("path_planner_runtime_source_missing")
    commit = str(
        _path_planner_git_output(root, "rev-parse", "HEAD")
    ).strip()
    raw_status = _path_planner_git_output(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        binary=True,
    )
    if type(raw_status) is not bytes:
        _fail("path_planner_runtime_source_git_probe_failed")
    records = raw_status.split(b"\0")
    inventory: list[dict[str, str]] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 4:
            _fail("path_planner_runtime_source_dirty_inventory_invalid")
        status = record[:2].decode("ascii")
        path = record[3:].decode(
            "utf-8",
            errors="surrogateescape",
        ).replace("\\", "/")
        inventory.append({"path": path, "status": status})
        if "R" in status or "C" in status:
            index += 1
    inventory.sort(key=lambda row: row["path"])
    sources: list[dict[str, object]] = []
    for relative in artifact_io.list_relative_files(root):
        if (
            not relative.startswith("src/path_planner/v2/")
            or not relative.endswith(".py")
        ):
            continue
        payload = artifact_io.read_bytes(root / PurePosixPath(relative))
        sources.append(
            {
                "logical_path": relative,
                "size_bytes": len(payload),
                "sha256": _sha256(payload),
            }
        )
    sources.sort(key=lambda row: str(row["logical_path"]))
    return build_path_planner_runtime_source_closure(
        submodule_commit=commit,
        dirty_inventory=inventory,
        required_sources=sources,
    )


def _safe_relative_path(value: object) -> str:
    text = _require_nonempty(value, "payload_index_path_invalid")
    if "\\" in text:
        _fail("payload_index_path_invalid", text)
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        _fail("payload_index_path_invalid", text)
    return text


def _read_canonical_json(path: str | Path) -> tuple[dict[str, object], bytes]:
    if not artifact_io.path_is_file(path):
        _fail("required_bundle_artifact_missing", path)
    payload = artifact_io.read_bytes(path)
    value = _canonical_loads(payload)
    if type(value) is not dict:
        _fail("bundle_json_root_not_object", path)
    if payload != _canonical_json_bytes(value) + b"\n":
        _fail("bundle_json_not_canonical", path)
    return value, payload


def _read_canonical_jsonl(path: str | Path) -> list[dict[str, object]]:
    if not artifact_io.path_is_file(path):
        _fail("required_bundle_artifact_missing", path)
    payload = artifact_io.read_bytes(path)
    if not payload or not payload.endswith(b"\n"):
        _fail("bundle_jsonl_not_canonical", path)
    rows: list[dict[str, object]] = []
    for index, line in enumerate(payload.splitlines()):
        value = _canonical_loads(line)
        if type(value) is not dict or line != _canonical_json_bytes(value):
            _fail("bundle_jsonl_not_canonical", f"{path}:{index + 1}")
        rows.append(value)
    return rows


def _walk_keys(value: object) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_keys(child)


def _reject_provider_truth_leak(value: Mapping[str, object]) -> None:
    leaked = sorted(
        key
        for key in _walk_keys(value)
        if key.casefold() in _PROVIDER_FORBIDDEN_KEYS
    )
    if leaked:
        _fail("provider_request_contains_truth", leaked[0])


def validate_primitive_label_identity(row: Mapping[str, object]) -> None:
    if type(row) is not dict:
        _fail("primitive_label_schema_invalid")
    platform = row.get("platform_kind")
    if platform not in G2_PLATFORMS:
        _fail("primitive_label_platform_invalid")
    if row.get("schema_version") != "g2-primitive-label/v2":
        _fail("primitive_label_schema_invalid")
    primitive = row.get("primitive_canonical")
    if type(primitive) is not dict:
        _fail("primitive_label_schema_invalid")
    case_sha256 = _domain_hash(
        "g2-case/v1",
        str(platform).encode("utf-8"),
        _canonical_json_bytes(primitive),
    )
    if row.get("case_sha256") != case_sha256:
        _fail("primitive_case_hash_mismatch")
    if row.get("label_id") != f"g2i-label-{platform}-{case_sha256[:20]}":
        _fail("primitive_label_join_mismatch")
    if row.get("primitive_sha256") != _domain_hash(
        "g2-primitive/v2",
        _canonical_json_bytes(primitive),
    ):
        _fail("primitive_hash_mismatch")
    terrain_sha256 = _require_sha256(
        row.get("terrain_sha256"),
        "primitive_terrain_hash_invalid",
    )
    if primitive.get("terrain_sha256") != terrain_sha256:
        _fail("primitive_terrain_join_mismatch")
    for field in (
        "case_envelope_sha256",
        "oracle_spec_sha256",
        "producer_implementation_sha256",
        "profile_or_parameter_record_sha256",
    ):
        _require_sha256(row.get(field), f"primitive_{field}_invalid")
    _require_nonempty(row.get("producer_id"), "primitive_producer_id_invalid")
    _require_nonempty(
        row.get("producer_revision"),
        "primitive_producer_revision_invalid",
    )
    _require_nonempty(
        row.get("profile_or_parameter_set_id"),
        "primitive_profile_id_invalid",
    )
    _require_nonempty(
        row.get("oracle_reason_code"),
        "primitive_reason_invalid",
    )
    _require_exact_bool(row.get("oracle_safe"), "primitive_oracle_safe_invalid")


def audit_primitive_labels(
    rows: Sequence[Mapping[str, object]],
    *,
    manifest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    counts = Counter(row.get("platform_kind") for row in rows)
    expected = {
        platform: EXPECTED_LABELS_PER_PLATFORM for platform in G2_PLATFORMS
    }
    if len(rows) != EXPECTED_LABELS_PER_PLATFORM * len(G2_PLATFORMS) or dict(
        counts
    ) != expected:
        _fail("primitive_label_count_mismatch", dict(counts))

    label_ids: set[str] = set()
    case_hashes: set[str] = set()
    primitive_hashes: set[str] = set()
    identities: dict[str, set[str]] = {
        "case_envelope_sha256": set(),
        "oracle_spec_sha256": set(),
        "producer_implementation_sha256": set(),
    }
    platform_profile_hashes: dict[str, set[str]] = {
        platform: set() for platform in G2_PLATFORMS
    }
    safe_counts = Counter()
    terrain_hashes: dict[str, set[str]] = {
        platform: set() for platform in G2_PLATFORMS
    }
    for row in rows:
        validate_primitive_label_identity(row)
        platform = str(row["platform_kind"])
        label_ids.add(str(row["label_id"]))
        case_hashes.add(str(row["case_sha256"]))
        primitive_hashes.add(str(row["primitive_sha256"]))
        for field in identities:
            identities[field].add(str(row[field]))
        platform_profile_hashes[platform].add(
            str(row["profile_or_parameter_record_sha256"])
        )
        terrain_hashes[platform].add(str(row["terrain_sha256"]))
        if row["oracle_safe"] is True:
            safe_counts[platform] += 1

    if (
        len(label_ids) != len(rows)
        or len(case_hashes) != len(rows)
        or len(primitive_hashes) != len(rows)
    ):
        _fail("primitive_label_identity_not_one_to_one")
    if any(len(values) != 1 for values in identities.values()):
        _fail("primitive_label_source_identity_drift")
    if any(len(values) != 1 for values in platform_profile_hashes.values()):
        _fail("primitive_label_profile_identity_drift")
    for platform in G2_PLATFORMS:
        safe_ratio = safe_counts[platform] / EXPECTED_LABELS_PER_PLATFORM
        if not 0.35 <= safe_ratio <= 0.65:
            _fail("primitive_label_safe_ratio_invalid", platform)
        if len(terrain_hashes[platform]) < 64:
            _fail("primitive_label_terrain_diversity_invalid", platform)

    if manifest is not None:
        producer_sha256 = next(iter(identities["producer_implementation_sha256"]))
        specification_sha256 = next(iter(identities["oracle_spec_sha256"]))
        if manifest.get("producer_implementation_sha256") != producer_sha256:
            _fail("primitive_label_producer_manifest_mismatch")
        if manifest.get("specification_sha256") != specification_sha256:
            _fail("primitive_label_specification_manifest_mismatch")
        profile_hashes = manifest.get("profile_or_parameter_record_sha256")
        if type(profile_hashes) is not dict:
            _fail("primitive_label_profile_manifest_mismatch")
        for platform in G2_PLATFORMS:
            if profile_hashes.get(platform) != next(
                iter(platform_profile_hashes[platform])
            ):
                _fail("primitive_label_profile_manifest_mismatch", platform)

    return {
        "schema_version": "xunce-mid-dual-g2-primitive-label-audit/v1",
        "platform_counts": expected,
        "label_count": len(rows),
        "label_ids_unique": True,
        "case_hashes_unique": True,
        "primitive_hashes_unique": True,
        "safe_ratio": {
            platform: safe_counts[platform] / EXPECTED_LABELS_PER_PLATFORM
            for platform in G2_PLATFORMS
        },
        "terrain_hash_count": {
            platform: len(terrain_hashes[platform])
            for platform in G2_PLATFORMS
        },
        "producer_implementation_sha256": next(
            iter(identities["producer_implementation_sha256"])
        ),
    }


def audit_small_map_optima(
    rows: Sequence[Mapping[str, object]],
    *,
    bundle_root: str | Path | None = None,
) -> dict[str, object]:
    counts = Counter(row.get("platform_kind") for row in rows)
    expected = {
        platform: EXPECTED_OPTIMA_PER_PLATFORM for platform in G2_PLATFORMS
    }
    if len(rows) != len(G2_PLATFORMS) or dict(counts) != expected:
        _fail("small_map_optimum_count_mismatch", dict(counts))
    case_ids: set[str] = set()
    certificate_hashes: set[str] = set()
    for row in rows:
        if (
            type(row) is not dict
            or row.get("schema_version") != "g2-small-map-optimum/v1"
        ):
            _fail("small_map_optimum_schema_invalid")
        platform = str(row["platform_kind"])
        case_ids.add(
            _require_nonempty(
                row.get("case_id"),
                "small_map_optimum_case_id_invalid",
            )
        )
        certificate_sha256 = _require_sha256(
            row.get("certificate_sha256"),
            "small_map_optimum_certificate_invalid",
        )
        certificate_hashes.add(certificate_sha256)
        _require_sha256(
            row.get("map_sha256"),
            "small_map_optimum_map_hash_invalid",
        )
        _require_sha256(
            row.get("solver_implementation_sha256"),
            "small_map_optimum_solver_hash_invalid",
        )
        completeness = row.get("completeness")
        if (
            type(completeness) is not dict
            or completeness.get("complete") is not True
            or completeness.get("actual_candidate_edge_count")
            != completeness.get("expected_candidate_edge_count")
            or completeness.get("actual_node_count")
            != completeness.get("expected_node_count")
            or row.get("candidate_edge_count")
            != completeness.get("actual_candidate_edge_count")
        ):
            _fail("small_map_optimum_completeness_invalid", platform)
        if bundle_root is not None:
            certificate_path = (
                Path(bundle_root)
                / "certificates"
                / f"{certificate_sha256}.json"
            )
            if not artifact_io.path_is_file(certificate_path):
                _fail("small_map_optimum_certificate_missing", platform)
            if _sha256(artifact_io.read_bytes(certificate_path)) != certificate_sha256:
                _fail("small_map_optimum_certificate_hash_mismatch", platform)
    if (
        len(case_ids) != len(G2_PLATFORMS)
        or len(certificate_hashes) != len(G2_PLATFORMS)
    ):
        _fail("small_map_optimum_identity_not_one_to_one")
    return {
        "schema_version": "xunce-mid-dual-g2-small-map-optimum-audit/v1",
        "platform_counts": expected,
        "case_ids_unique": True,
        "certificate_hashes_unique": True,
    }


def _validate_truth_request_identity(row: Mapping[str, object]) -> None:
    if (
        type(row) is not dict
        or row.get("schema_version") != "g2-truth-request/v2"
    ):
        _fail("request_schema_invalid")
    platform = row.get("platform_kind")
    scale = row.get("scale")
    difficulty = row.get("difficulty_class")
    if platform not in G2_PLATFORMS:
        _fail("request_platform_invalid")
    if scale not in EXPECTED_SCALE_COUNTS:
        _fail("request_scale_invalid")
    if difficulty not in {"reachable", "hard_reachable", "unreachable"}:
        _fail("request_difficulty_invalid")
    expected_reachable = difficulty != "unreachable"
    if (
        _require_exact_bool(
            row.get("oracle_reachable"),
            "request_oracle_reachable_invalid",
        )
        is not expected_reachable
    ):
        _fail("request_reachability_taxonomy_mismatch")
    raw_source_sha256 = _require_sha256(
        row.get("raw_source_sha256"),
        "request_raw_source_hash_invalid",
    )
    certificate_sha256 = _require_sha256(
        row.get("truth_certificate_sha256"),
        "request_certificate_hash_invalid",
    )
    terrain_sha256 = _require_sha256(
        row.get("terrain_sha256"),
        "request_terrain_hash_invalid",
    )
    terrain_geometry_sha256 = _require_sha256(
        row.get("terrain_geometry_sha256"),
        "request_terrain_geometry_hash_invalid",
    )
    profile_sha256 = _require_sha256(
        row.get("profile_or_parameter_record_sha256"),
        "request_profile_hash_invalid",
    )
    truth_request_sha256 = _domain_hash(
        "g2-truth-request/v2",
        raw_source_sha256.encode("ascii"),
        certificate_sha256.encode("ascii"),
        terrain_sha256.encode("ascii"),
        terrain_geometry_sha256.encode("ascii"),
        profile_sha256.encode("ascii"),
    )
    if row.get("truth_request_sha256") != truth_request_sha256:
        _fail("truth_request_hash_mismatch")
    expected_request_id = (
        f"g2i-req-{platform}-{scale}-{truth_request_sha256[:20]}"
    )
    if row.get("request_id") != expected_request_id:
        _fail("truth_request_id_hash_join_mismatch")
    budget = row.get("resource_budget")
    expected_budget = 128 if scale == "standard" else 512
    if (
        type(budget) is not dict
        or budget.get("max_path_primitives") != expected_budget
    ):
        _fail("request_resource_budget_invalid")
    source_ids = row.get("source_ids")
    if (
        type(source_ids) is not list
        or len(source_ids) != 1
        or type(source_ids[0]) is not str
        or not source_ids[0]
    ):
        _fail("request_source_ids_invalid")
    provenance = row.get("terrain_provenance")
    if (
        type(provenance) is not dict
        or provenance.get("physical_obstacle_cells_written") is not False
    ):
        _fail("request_terrain_provenance_invalid")
    if scale == "standard":
        if provenance.get("source_kind") != "procedural_simulation_proxy/v1":
            _fail("request_standard_terrain_source_invalid")
    else:
        micro_kind = provenance.get(
            "micro_source_kind",
            provenance.get("source_kind"),
        )
        if micro_kind != "synthetic_terrain_obstacle_proxy/v1":
            _fail("request_kilometer_terrain_source_invalid")
    _require_sha256(
        row.get("producer_implementation_sha256"),
        "request_producer_hash_invalid",
    )
    _require_sha256(
        row.get("graph_template_sha256"),
        "request_graph_hash_invalid",
    )
    _require_sha256(
        row.get("selection_hash"),
        "request_selection_hash_invalid",
    )


def audit_request_matrix(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(rows) != 129:
        _fail("request_matrix_count_mismatch", len(rows))
    platform_scale = Counter(
        (row.get("platform_kind"), row.get("scale")) for row in rows
    )
    expected_platform_scale = {
        (platform, scale): count
        for platform in G2_PLATFORMS
        for scale, count in EXPECTED_SCALE_COUNTS.items()
    }
    if dict(platform_scale) != expected_platform_scale:
        _fail("request_matrix_count_mismatch", dict(platform_scale))
    platform_scale_difficulty = Counter(
        (
            row.get("platform_kind"),
            row.get("scale"),
            row.get("difficulty_class"),
        )
        for row in rows
    )
    expected_difficulty = {
        (platform, scale, difficulty): count
        for platform in G2_PLATFORMS
        for (scale, difficulty), count in EXPECTED_REQUEST_COUNTS.items()
    }
    if dict(platform_scale_difficulty) != expected_difficulty:
        _fail(
            "request_difficulty_count_mismatch",
            dict(platform_scale_difficulty),
        )
    ids: set[str] = set()
    truth_hashes: set[str] = set()
    raw_hashes: set[str] = set()
    certificate_hashes: set[str] = set()
    source_ids: set[str] = set()
    producer_hashes: set[str] = set()
    for row in rows:
        _validate_truth_request_identity(row)
        ids.add(str(row["request_id"]))
        truth_hashes.add(str(row["truth_request_sha256"]))
        raw_hashes.add(str(row["raw_source_sha256"]))
        certificate_hashes.add(str(row["truth_certificate_sha256"]))
        source_ids.add(str(row["source_ids"][0]))
        producer_hashes.add(str(row["producer_implementation_sha256"]))
    if any(
        len(values) != len(rows)
        for values in (
            ids,
            truth_hashes,
            raw_hashes,
            certificate_hashes,
            source_ids,
        )
    ):
        _fail("request_identity_not_one_to_one")
    if len(producer_hashes) != 1:
        _fail("request_producer_identity_drift")
    return {
        "schema_version": "xunce-mid-dual-g2-request-matrix-audit/v1",
        "request_count": len(rows),
        "platform_scale_counts": {
            f"{platform}/{scale}": expected_platform_scale[(platform, scale)]
            for platform in G2_PLATFORMS
            for scale in ("standard", "kilometer")
        },
        "platform_scale_difficulty_counts": {
            f"{platform}/{scale}": {
                difficulty: expected_difficulty[
                    (platform, scale, difficulty)
                ]
                for difficulty in (
                    "reachable",
                    "hard_reachable",
                    "unreachable",
                )
            }
            for platform in G2_PLATFORMS
            for scale in ("standard", "kilometer")
        },
        "request_ids_unique": True,
        "truth_request_hashes_unique": True,
        "raw_source_hashes_unique": True,
        "certificate_hashes_unique": True,
        "producer_implementation_sha256": next(iter(producer_hashes)),
    }


def _validate_source(value: Mapping[str, object], code: str) -> dict[str, str]:
    if type(value) is not dict:
        _fail(code)
    return {
        "identity": _require_nonempty(value.get("identity"), code),
        "source_bytes_sha256": _require_sha256(
            value.get("source_bytes_sha256"),
            code,
        ),
        "implementation_sha256": _require_sha256(
            value.get("implementation_sha256"),
            code,
        ),
    }


def audit_source_separation(
    *,
    oracle_source: Mapping[str, object],
    provider_source: Mapping[str, object],
) -> dict[str, object]:
    oracle = _validate_source(oracle_source, "oracle_source_invalid")
    provider = _validate_source(provider_source, "provider_source_invalid")
    if any(
        oracle[field] == provider[field]
        for field in (
            "identity",
            "source_bytes_sha256",
            "implementation_sha256",
        )
    ):
        _fail("provider_oracle_source_not_separated")
    return {
        "schema_version": "xunce-mid-dual-g2-source-separation-audit/v1",
        "separated": True,
        "oracle_source": oracle,
        "provider_source": provider,
    }


def _canonical_npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(
        output,
        np.ascontiguousarray(array),
        version=(2, 0),
        allow_pickle=False,
    )
    return output.getvalue()


def _decode_truth_terrain(
    payload: bytes,
) -> tuple[dict[str, np.ndarray], dict[str, object], str]:
    expected_names = [
        *(f"{name}.npy" for name, _ in _TERRAIN_ARRAY_SPECS),
        "metadata.json",
    ]
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload), mode="r")
    except zipfile.BadZipFile as exc:
        raise G2InputContractError("terrain_blob_invalid_npz") from exc
    arrays: dict[str, np.ndarray] = {}
    with archive:
        infos = archive.infolist()
        if [info.filename for info in infos] != expected_names:
            _fail("terrain_blob_member_contract_mismatch")
        for info in infos:
            if (
                info.compress_type != zipfile.ZIP_STORED
                or info.date_time != (1980, 1, 1, 0, 0, 0)
                or info.create_system != 3
                or info.external_attr != 0o100644 << 16
                or info.flag_bits != 0
                or info.comment != b""
                or info.extra != b""
            ):
                _fail("terrain_blob_noncanonical_zip")
        shapes: set[tuple[int, ...]] = set()
        for name, dtype in _TERRAIN_ARRAY_SPECS:
            data = archive.read(f"{name}.npy")
            try:
                array = np.lib.format.read_array(
                    io.BytesIO(data),
                    allow_pickle=False,
                )
            except (TypeError, ValueError) as exc:
                raise G2InputContractError("terrain_blob_array_invalid") from exc
            if (
                array.dtype != dtype
                or array.ndim != 2
                or _canonical_npy_bytes(array) != data
            ):
                _fail("terrain_blob_array_contract_mismatch", name)
            arrays[name] = np.ascontiguousarray(array)
            shapes.add(tuple(array.shape))
        if len(shapes) != 1 or next(iter(shapes), (0, 0)) == (0, 0):
            _fail("terrain_blob_shape_mismatch")
        metadata_bytes = archive.read("metadata.json")
        metadata = _canonical_loads(metadata_bytes)
        if (
            type(metadata) is not dict
            or metadata_bytes != _canonical_json_bytes(metadata)
            or metadata.get("geometry_schema_version")
            != "g2-neutral-terrain/v2"
        ):
            _fail("terrain_blob_metadata_invalid")
    if not set(np.unique(arrays["known"])).issubset({0, 1}):
        _fail("terrain_blob_known_mask_invalid")
    if not set(np.unique(arrays["cell_class"])).issubset({0, 2}):
        _fail("terrain_blob_cell_class_invalid")
    if np.any(arrays["confidence_ppm"] > 1_000_000):
        _fail("terrain_blob_confidence_invalid")
    geometry_sha256 = _domain_hash(
        "g2-terrain-geometry/v1",
        *(
            name.encode("ascii") + b"\0" + arrays[name].tobytes(order="C")
            for name, _ in _TERRAIN_ARRAY_SPECS
        ),
    )
    return arrays, metadata, geometry_sha256


def _validate_truth_request_join(
    row: Mapping[str, object],
    terrain_payload: bytes,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    _validate_truth_request_identity(row)
    if _sha256(terrain_payload) != row["terrain_sha256"]:
        _fail("terrain_blob_sha256_mismatch")
    arrays, metadata, geometry_sha256 = _decode_truth_terrain(terrain_payload)
    if geometry_sha256 != row["terrain_geometry_sha256"]:
        _fail("terrain_geometry_sha256_mismatch")
    if metadata.get("provenance") != row.get("terrain_provenance"):
        _fail("terrain_provenance_join_mismatch")
    if metadata.get("provenance", {}).get(
        "physical_obstacle_cells_written"
    ) is not False:
        _fail("terrain_physical_claim_invalid")
    return arrays, metadata


def _node_xy(node: object, *, width: int, height: int) -> tuple[int, int]:
    value = _require_nonempty(node, "request_node_id_invalid")
    match = _NODE_ID_RE.fullmatch(value)
    if match is None:
        _fail("request_node_id_invalid", value)
    x = int(match.group("x"))
    y = int(match.group("y"))
    if x >= width or y >= height:
        _fail("request_node_out_of_terrain", value)
    return x, y


def _provider_request_payload(
    row: Mapping[str, object],
    terrain_payload: bytes,
) -> dict[str, object]:
    arrays, metadata = _validate_truth_request_join(row, terrain_payload)
    platform = str(row["platform_kind"])
    stack = PLATFORM_STACKS[platform]
    height, width = arrays["height_mm"].shape
    resolution_m = int(metadata["macro_cell_size_mm"]) / 1000.0
    start = row.get("start")
    goal = row.get("goal")
    if type(start) is not dict or type(goal) is not dict:
        _fail("request_node_id_invalid")
    start_x, start_y = _node_xy(
        start.get("node_id"),
        width=width,
        height=height,
    )
    goal_x, goal_y = _node_xy(
        goal.get("node_id"),
        width=width,
        height=height,
    )
    max_path_primitives = int(row["resource_budget"]["max_path_primitives"])
    envelope: dict[str, object] = {
        "schema_version": PROVIDER_REQUEST_SCHEMA_VERSION,
        "request_id": row["request_id"],
        "platform": platform,
        "scale": row["scale"],
        "platform_stack": stack,
        "canonical_request_metadata": {
            "accelerator_policy": "disabled",
            "determinism_seed": row["determinism_seed"],
            "goal_state": {
                "heading_rad": 0.0,
                "x_m": (goal_x + 0.5) * resolution_m,
                "y_m": (goal_y + 0.5) * resolution_m,
            },
            "objective_profile": {
                "distance_weight": 1.0,
                "energy_weight": 0.0,
                "risk_weight": 0.0,
                "time_weight": 0.0,
            },
            "platform_profile_id": stack["profile_id"],
            "request_id": row["request_id"],
            "resource_budget": {
                "max_expanded_states": max_path_primitives * 64,
                "max_memory_bytes": (
                    536_870_912
                    if platform == "hopper"
                    else max_path_primitives * 262144
                ),
                "max_route_states": max_path_primitives + 1,
            },
            "start_state": {
                "heading_rad": 0.0,
                "x_m": (start_x + 0.5) * resolution_m,
                "y_m": (start_y + 0.5) * resolution_m,
            },
            "timeout_s": 2.0,
        },
        "terrain_binding": {
            "geometry_schema_version": metadata["geometry_schema_version"],
            "height": height,
            "macro_cell_size_mm": metadata["macro_cell_size_mm"],
            "physical_obstacle_cells_written": False,
            "sub_20m_layer_source_kind": metadata.get(
                "sub_20m_layer_source_kind"
            ),
            "terrain_geometry_sha256": row["terrain_geometry_sha256"],
            "terrain_sha256": row["terrain_sha256"],
            "width": width,
        },
    }
    _reject_provider_truth_leak(envelope)
    provider_request_sha256 = _domain_hash(
        "xunce-mid-dual-g2-provider-request/v1",
        _canonical_json_bytes(envelope),
        terrain_payload,
    )
    return {
        **envelope,
        "provider_request_sha256": provider_request_sha256,
    }


def _crosswalk_row(
    truth_row: Mapping[str, object],
    provider_row: Mapping[str, object],
) -> dict[str, object]:
    difficulty = str(truth_row["difficulty_class"])
    return {
        "schema_version": CROSSWALK_SCHEMA_VERSION,
        "request_id": truth_row["request_id"],
        "platform": truth_row["platform_kind"],
        "scale": truth_row["scale"],
        "request_class": (
            "normal_reachable" if difficulty == "reachable" else difficulty
        ),
        "outcome_kind": (
            "unreachable" if difficulty == "unreachable" else "reachable"
        ),
        "truth_request_sha256": truth_row["truth_request_sha256"],
        "truth_certificate_sha256": truth_row[
            "truth_certificate_sha256"
        ],
        "terrain_sha256": truth_row["terrain_sha256"],
        "provider_request_sha256": provider_row[
            "provider_request_sha256"
        ],
    }


def _select_g3_replay_cohort(
    crosswalk: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if len(crosswalk) != 129:
        return None
    rows: list[dict[str, object]] = []
    for platform in ("legged", "hopper"):
        for replay_index, (scale, difficulty) in enumerate(G3_REPLAY_STRATA):
            request_class = (
                "normal_reachable"
                if difficulty == "reachable"
                else difficulty
            )
            candidates = [
                row
                for row in crosswalk
                if row["platform"] == platform
                and row["scale"] == scale
                and row["request_class"] == request_class
            ]
            if not candidates:
                _fail("g3_replay_cohort_stratum_missing")
            selected = min(
                candidates,
                key=lambda row: _domain_hash(
                    "xunce-mid-dual-g3-replay-selection/v1",
                    str(G3_REPLAY_SELECTION_SEED).encode("ascii"),
                    platform.encode("ascii"),
                    scale.encode("ascii"),
                    difficulty.encode("ascii"),
                    str(row["truth_request_sha256"]).encode("ascii"),
                ),
            )
            rows.append(
                {
                    "replay_id": f"g3-replay-{platform}-{replay_index:02d}",
                    "platform": platform,
                    "scale": scale,
                    "difficulty_class": difficulty,
                    "request_id": selected["request_id"],
                    "truth_request_sha256": selected[
                        "truth_request_sha256"
                    ],
                    "provider_request_sha256": selected[
                        "provider_request_sha256"
                    ],
                }
            )
    cohort_core = {
        "schema_version": G3_REPLAY_COHORT_SCHEMA_VERSION,
        "selection_contract": G3_REPLAY_SELECTION_CONTRACT,
        "selection_seed": G3_REPLAY_SELECTION_SEED,
        "result_fields_used": False,
        "rows": rows,
    }
    return {
        **cohort_core,
        "cohort_sha256": _domain_hash(
            "xunce-mid-dual-g3-replay-cohort/v1",
            _canonical_json_bytes(cohort_core),
        ),
    }


def build_execution_crosswalk(
    truth_requests: Sequence[Mapping[str, object]],
    terrain_payload_by_sha256: Mapping[str, bytes],
) -> dict[str, object]:
    provider_rows: list[dict[str, object]] = []
    crosswalk: list[dict[str, object]] = []
    truth_hashes: set[str] = set()
    provider_hashes: set[str] = set()
    request_ids: set[str] = set()
    ordered = sorted(
        truth_requests,
        key=lambda row: (
            str(row.get("platform_kind")),
            str(row.get("scale")),
            int(row.get("selection_rank", 0)),
            str(row.get("request_id")),
        ),
    )
    for truth_row in ordered:
        terrain_sha256 = _require_sha256(
            truth_row.get("terrain_sha256"),
            "request_terrain_hash_invalid",
        )
        terrain_payload = terrain_payload_by_sha256.get(terrain_sha256)
        if type(terrain_payload) is not bytes:
            _fail("request_terrain_blob_missing", terrain_sha256)
        provider_row = _provider_request_payload(truth_row, terrain_payload)
        crosswalk_row = _crosswalk_row(truth_row, provider_row)
        truth_sha256 = str(crosswalk_row["truth_request_sha256"])
        provider_sha256 = str(crosswalk_row["provider_request_sha256"])
        request_id = str(crosswalk_row["request_id"])
        if (
            truth_sha256 in truth_hashes
            or provider_sha256 in provider_hashes
            or request_id in request_ids
        ):
            _fail("truth_provider_crosswalk_not_one_to_one")
        truth_hashes.add(truth_sha256)
        provider_hashes.add(provider_sha256)
        request_ids.add(request_id)
        provider_rows.append(provider_row)
        crosswalk.append(crosswalk_row)
    return {
        "schema_version": "xunce-mid-dual-g2-execution-crosswalk-build/v1",
        "provider_requests": provider_rows,
        "crosswalk": crosswalk,
        "g3_replay_cohort": _select_g3_replay_cohort(crosswalk),
    }


def audit_ppo_targets(
    *,
    mode: str,
    request_sha256: Sequence[str],
    ppo_target_rows: Sequence[Mapping[str, object]] | None,
) -> dict[str, object]:
    if mode not in {"reduced", "gate6"}:
        _fail("g2_mode_invalid")
    request_hashes = tuple(
        _require_sha256(value, "ppo_target_request_hash_invalid")
        for value in request_sha256
    )
    if len(request_hashes) != len(set(request_hashes)):
        _fail("ppo_target_request_hash_not_unique")
    if ppo_target_rows is None:
        if mode == "gate6":
            _fail("gate6_ppo_targets_missing")
        return {
            "mode": "reduced",
            "ppo_targets_required": False,
            "ppo_target_count": 0,
        }
    seen: set[str] = set()
    for row in ppo_target_rows:
        if (
            type(row) is not dict
            or row.get("schema_version") != "g2-gate6-ppo-target/v1"
        ):
            _fail("ppo_target_schema_invalid")
        request_sha = _require_sha256(
            row.get("truth_request_sha256"),
            "ppo_target_request_hash_invalid",
        )
        _require_sha256(
            row.get("target_sha256"),
            "ppo_target_hash_invalid",
        )
        if request_sha in seen:
            _fail("ppo_target_duplicate_request")
        seen.add(request_sha)
    if set(request_hashes) != seen:
        _fail(
            "gate6_ppo_target_join_mismatch"
            if mode == "gate6"
            else "ppo_target_join_mismatch"
        )
    return {
        "mode": mode,
        "ppo_targets_required": mode == "gate6",
        "ppo_target_count": len(seen),
    }


def resolve_hopper_formal_eligibility(
    *,
    candidate_record: Mapping[str, object] | None,
    approval_record: Mapping[str, object] | None,
    input_binding: Mapping[str, object],
    provider_source: Mapping[str, object],
    oracle_source: Mapping[str, object],
    current_runtime_source_closure: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if candidate_record is None:
        return {
            "formal_evidence_eligible": False,
            "blockers": [
                "approve_midterm_hopper_simulation_proxy_parameter_set"
            ],
        }
    if (
        candidate_record.get("evidence_class") == "test_fixture"
        or candidate_record.get("parameter_set_id")
        == GATE5B_HOPPER_PARAMETER_SET_ID
    ):
        return {
            "formal_evidence_eligible": False,
            "blockers": ["hopper_gate5b_test_fixture_not_formal"],
        }
    blockers: list[str] = []
    if (
        candidate_record.get("schema_version") != "g2-hopper-candidate/v2"
        or candidate_record.get("parameter_set_id")
        != GENERIC_HOPPER_CANDIDATE_PARAMETER_SET_ID
        or candidate_record.get("evidence_class")
        != "candidate_engineering_proxy"
        or candidate_record.get("simulation_proxy") is not True
        or candidate_record.get("formal_evidence_eligible") is not False
        or candidate_record.get("physical_capability_claimed") is not False
    ):
        blockers.append("hopper_candidate_record_invalid")
    candidate_sha256 = _sha256(_canonical_json_bytes(candidate_record))
    if approval_record is None:
        blockers.extend(
            (
                "missing_artifact_bound_o2_approval",
                "approve_midterm_hopper_simulation_proxy_parameter_set",
            )
        )
        return {
            "formal_evidence_eligible": False,
            "candidate_record_sha256": candidate_sha256,
            "blockers": sorted(set(blockers)),
        }
    try:
        provider = _validate_source(
            provider_source,
            "provider_source_invalid",
        )
        oracle = _validate_source(
            oracle_source,
            "oracle_source_invalid",
        )
        audit_source_separation(
            provider_source=provider,
            oracle_source=oracle,
        )
        current_closure = _validated_path_planner_runtime_source_closure(
            dict(current_runtime_source_closure)
            if current_runtime_source_closure is not None
            else capture_path_planner_runtime_source_closure()
        )
        approved_closure = approval_record.get(
            "path_planner_runtime_source_closure"
        )
        approved_closure_sha256 = approval_record.get(
            "path_planner_runtime_source_closure_sha256"
        )
        if (
            type(approved_closure) is not dict
            or not _is_sha256(approved_closure_sha256)
            or approved_closure_sha256
            != approved_closure.get(
                "path_planner_runtime_source_closure_sha256"
            )
        ):
            _fail("artifact_bound_approval_runtime_source_missing")
        validate_path_planner_runtime_source_closure(
            approved_closure,
            current_closure,
        )
        required_input = {
            "authorization_sha256": AUTHORIZATION_SHA256,
            "input_set_id": _require_nonempty(
                input_binding.get("input_set_id"),
                "approval_input_binding_invalid",
            ),
            "manifest_core_sha256": _require_sha256(
                input_binding.get("manifest_core_sha256"),
                "approval_input_binding_invalid",
            ),
            "payload_root_sha256": _require_sha256(
                input_binding.get("payload_root_sha256"),
                "approval_input_binding_invalid",
            ),
            "source_attestations_sha256": _require_sha256(
                input_binding.get("source_attestations_sha256"),
                "approval_input_binding_invalid",
            ),
            "producer_implementation_sha256": _require_sha256(
                input_binding.get("producer_implementation_sha256"),
                "approval_input_binding_invalid",
            ),
            "producer_source_bytes_sha256": _require_sha256(
                input_binding.get("producer_source_bytes_sha256"),
                "approval_input_binding_invalid",
            ),
            "reproduction_audit_sha256": _require_sha256(
                input_binding.get("reproduction_audit_sha256"),
                "approval_input_binding_invalid",
            ),
            "technical_isolation_audit_sha256": _require_sha256(
                input_binding.get("technical_isolation_audit_sha256"),
                "approval_input_binding_invalid",
            ),
        }
        if (
            input_binding.get("authorization_sha256")
            != AUTHORIZATION_SHA256
            or input_binding.get("technical_independence") != "T2_candidate"
            or input_binding.get("organizational_independence")
            != "project_internal"
            or input_binding.get("structural_audit_passed") is not True
        ):
            _fail("approval_input_binding_invalid")
        if (
            approval_record.get("schema_version")
            != APPROVAL_SCHEMA_VERSION
            or type(approval_record.get("approval_id")) is not str
            or not approval_record.get("approval_id")
            or approval_record.get("formal_evidence_eligible") is not True
            or approval_record.get("authorization_sha256")
            != required_input["authorization_sha256"]
            or approval_record.get("input_set_id")
            != required_input["input_set_id"]
            or approval_record.get("manifest_core_sha256")
            != required_input["manifest_core_sha256"]
            or approval_record.get("payload_root_sha256")
            != required_input["payload_root_sha256"]
            or approval_record.get("source_attestations_sha256")
            != required_input["source_attestations_sha256"]
            or approval_record.get("producer_implementation_sha256")
            != required_input["producer_implementation_sha256"]
            or approval_record.get("producer_source_bytes_sha256")
            != required_input["producer_source_bytes_sha256"]
            or approval_record.get("reproduction_audit_sha256")
            != required_input["reproduction_audit_sha256"]
            or approval_record.get("technical_isolation_audit_sha256")
            != required_input["technical_isolation_audit_sha256"]
            or approval_record.get("hopper_candidate_record_sha256")
            != candidate_sha256
            or approval_record.get("hopper_candidate_parameter_set_id")
            != GENERIC_HOPPER_CANDIDATE_PARAMETER_SET_ID
            or approval_record.get(
                "hopper_implementation_parameter_set_id"
            )
            != GENERIC_HOPPER_IMPLEMENTATION_PARAMETER_SET_ID
            or approval_record.get("provider_source") != provider
            or approval_record.get("oracle_source") != oracle
            or approval_record.get("g2_scope_authorized") is not True
            or approval_record.get("g3_scope_authorized") is not True
            or approval_record.get("physical_capability_claimed") is not False
            or approval_record.get("hardware_certification_claimed") is not False
        ):
            _fail("artifact_bound_approval_mismatch")
        for field in (
            "hopper_implementation_record_sha256",
            "regression_evidence_sha256",
            "independent_technical_review_sha256",
        ):
            _require_sha256(
                approval_record.get(field),
                "artifact_bound_approval_mismatch",
            )
        cohort_sha256 = input_binding.get("g3_replay_cohort_sha256")
        if cohort_sha256 is not None:
            _require_sha256(
                cohort_sha256,
                "artifact_bound_approval_g3_cohort_mismatch",
            )
            truth_hashes = input_binding.get(
                "g3_replay_truth_request_sha256"
            )
            provider_hashes = input_binding.get(
                "g3_replay_provider_request_sha256"
            )
            if (
                type(truth_hashes) is not list
                or type(provider_hashes) is not list
                or len(truth_hashes) != 6
                or len(provider_hashes) != 6
                or any(not _is_sha256(value) for value in truth_hashes)
                or any(not _is_sha256(value) for value in provider_hashes)
                or approval_record.get("g3_replay_cohort_sha256")
                != cohort_sha256
                or approval_record.get(
                    "g3_replay_truth_request_sha256"
                )
                != truth_hashes
                or approval_record.get(
                    "g3_replay_provider_request_sha256"
                )
                != provider_hashes
            ):
                _fail("artifact_bound_approval_g3_cohort_mismatch")
    except G2InputContractError as exc:
        blockers.append(exc.code)
    approval_sha256 = _sha256(_canonical_json_bytes(approval_record))
    return {
        "formal_evidence_eligible": not blockers,
        "candidate_record_sha256": candidate_sha256,
        "approval_record_sha256": approval_sha256,
        "resolved_parameter_set_id": (
            GENERIC_HOPPER_IMPLEMENTATION_PARAMETER_SET_ID
            if not blockers
            else None
        ),
        "simulation_proxy": True,
        "physical_capability_claimed": False,
        "hardware_certification_claimed": False,
        "path_planner_runtime_source_closure_sha256": (
            None
            if blockers
            else approval_record.get(
                "path_planner_runtime_source_closure_sha256"
            )
        ),
        "blockers": sorted(set(blockers)),
    }


def _verify_payload_index(
    root: Path,
    freeze: Mapping[str, object],
) -> dict[str, object]:
    raw_index = freeze.get("payload_index")
    if type(raw_index) is not list or not raw_index:
        _fail("payload_index_invalid")
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_row in raw_index:
        if (
            type(raw_row) is not dict
            or set(raw_row)
            != {"byte_length", "relative_path", "sha256"}
            or type(raw_row.get("byte_length")) is not int
            or raw_row["byte_length"] < 0
        ):
            _fail("payload_index_invalid")
        relative_path = _safe_relative_path(raw_row["relative_path"])
        if relative_path in seen:
            _fail("payload_index_duplicate_path", relative_path)
        seen.add(relative_path)
        expected_sha256 = _require_sha256(
            raw_row["sha256"],
            "payload_index_hash_invalid",
        )
        path = root / PurePosixPath(relative_path)
        if not artifact_io.path_is_file(path):
            _fail("payload_index_file_missing", relative_path)
        payload = artifact_io.read_bytes(path)
        actual = {
            "byte_length": len(payload),
            "relative_path": relative_path,
            "sha256": _sha256(payload),
        }
        if actual["byte_length"] != raw_row["byte_length"] or (
            actual["sha256"] != expected_sha256
        ):
            _fail("payload_index_file_drift", relative_path)
        normalized.append(actual)
    if normalized != sorted(
        normalized,
        key=lambda row: str(row["relative_path"]),
    ):
        _fail("payload_index_order_invalid")
    actual_files = sorted(
        relative_path
        for relative_path in artifact_io.list_relative_files(root)
        if relative_path
        not in {"truth-freeze.json", "source-attestations.json"}
    )
    if actual_files != sorted(seen):
        _fail("payload_file_set_mismatch")
    payload_root_sha256 = _domain_hash(
        "g2-payload-root/v1",
        _canonical_json_bytes(normalized),
    )
    if freeze.get("payload_root_sha256") != payload_root_sha256:
        _fail("payload_root_hash_mismatch")
    return {
        "payload_index": normalized,
        "payload_root_sha256": payload_root_sha256,
    }


def _manifest_core_sha256(manifest: Mapping[str, object]) -> str:
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"input_set_id", "manifest_core_sha256"}
    }
    return _domain_hash(
        "g2-manifest-core/v2",
        _canonical_json_bytes(core),
    )


def _read_terrain_payloads(
    root: Path,
    requests: Sequence[Mapping[str, object]],
) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for row in requests:
        terrain_sha256 = str(row["terrain_sha256"])
        if terrain_sha256 in payloads:
            continue
        path = root / "terrain" / f"{terrain_sha256}.npz"
        if not artifact_io.path_is_file(path):
            _fail("request_terrain_blob_missing", terrain_sha256)
        payloads[terrain_sha256] = artifact_io.read_bytes(path)
    return payloads


def _verify_request_certificates(
    root: Path,
    requests: Sequence[Mapping[str, object]],
) -> None:
    for row in requests:
        certificate_sha256 = str(row["truth_certificate_sha256"])
        path = root / "certificates" / f"{certificate_sha256}.json"
        if not artifact_io.path_is_file(path):
            _fail("request_certificate_missing", row["request_id"])
        if _sha256(artifact_io.read_bytes(path)) != certificate_sha256:
            _fail("request_certificate_hash_mismatch", row["request_id"])


def audit_truth_bundle(
    bundle_root: str | Path,
    *,
    approval_path: str | Path | None = None,
    mode: str = "reduced",
) -> dict[str, object]:
    root = Path(bundle_root).resolve()
    if not artifact_io.path_is_dir(root):
        _fail("independent_g2_input_bundle_missing", root)
    freeze, freeze_bytes = _read_canonical_json(root / "truth-freeze.json")
    manifest, manifest_bytes = _read_canonical_json(root / "manifest.json")
    attestation, attestation_bytes = _read_canonical_json(
        root / "source-attestations.json"
    )
    if (
        freeze.get("schema_version") != "g2-truth-freeze/v2"
        or manifest.get("schema_version") != "g2-truth-manifest-core/v2"
        or freeze.get("publication_order") != "data-first-freeze-last"
    ):
        _fail("truth_bundle_metadata_schema_invalid")
    if (
        freeze.get("fixture_only") is not False
        or manifest.get("fixture_only") is not False
        or attestation.get("fixture_only") is not False
        or freeze.get("formal_evidence_eligible") is not False
        or manifest.get("formal_evidence_eligible") is not False
        or attestation.get("formal_evidence_eligible") is not False
        or attestation.get("technical_independence") != "T2_candidate"
        or attestation.get("organizational_independence") != "project_internal"
        or attestation.get("status") != "pending_o2_signature"
    ):
        _fail("truth_bundle_candidate_boundary_invalid")
    if (
        manifest.get("authorization_sha256") != AUTHORIZATION_SHA256
        or attestation.get("authorization_sha256") != AUTHORIZATION_SHA256
    ):
        _fail("truth_bundle_authorization_mismatch")
    if manifest.get("counts") != EXPECTED_BUNDLE_COUNTS or (
        freeze.get("counts") != EXPECTED_BUNDLE_COUNTS
    ):
        _fail("truth_bundle_counts_mismatch")
    manifest_core_sha256 = _manifest_core_sha256(manifest)
    if (
        manifest.get("manifest_core_sha256") != manifest_core_sha256
        or freeze.get("manifest_core_sha256") != manifest_core_sha256
        or attestation.get("manifest_core_sha256") != manifest_core_sha256
    ):
        _fail("truth_bundle_manifest_core_mismatch")
    input_contract_sha256 = _require_sha256(
        manifest.get("input_contract_sha256"),
        "truth_bundle_input_contract_invalid",
    )
    input_set_id = _require_nonempty(
        manifest.get("input_set_id"),
        "truth_bundle_input_set_id_invalid",
    )
    if input_set_id != f"g2t2-candidate-{input_contract_sha256[:24]}" or (
        freeze.get("input_set_id") != input_set_id
        or attestation.get("input_set_id") != input_set_id
    ):
        _fail("truth_bundle_input_set_id_mismatch")
    expected_approval_path = approval_artifact_path(input_set_id).resolve()
    selected_approval_path: Path | None = None
    if approval_path is not None:
        supplied_approval_path = Path(approval_path)
        if (
            not supplied_approval_path.is_absolute()
            or os.path.normcase(str(supplied_approval_path.resolve()))
            != os.path.normcase(str(expected_approval_path))
        ):
            _fail("artifact_bound_approval_path_mismatch")
        selected_approval_path = expected_approval_path
    elif artifact_io.path_is_file(expected_approval_path):
        selected_approval_path = expected_approval_path
    approval_record: dict[str, object] | None = None
    approval_bytes: bytes | None = None
    if selected_approval_path is not None:
        approval_record, approval_bytes = _read_canonical_json(
            selected_approval_path
        )
    payload_audit = _verify_payload_index(root, freeze)
    if (
        freeze.get("fresh_reproduction_payload_root_sha256")
        != payload_audit["payload_root_sha256"]
        or attestation.get("payload_root_sha256")
        != payload_audit["payload_root_sha256"]
        or freeze.get("source_attestations_sha256")
        != _sha256(attestation_bytes)
    ):
        _fail("truth_bundle_attestation_or_reproduction_mismatch")

    technical, technical_bytes = _read_canonical_json(
        root / "audits" / "technical-isolation.json"
    )
    reproduction, reproduction_bytes = _read_canonical_json(
        root / "audits" / "reproduction.json"
    )
    raw_to_truth, _ = _read_canonical_json(
        root / "audits" / "raw-to-truth.json"
    )
    if (
        technical.get("passed") is not True
        or technical.get("production_process_isolation_verified") is not True
        or technical.get("forbidden_imports") != []
        or technical.get("forbidden_dynamic_calls") != []
        or reproduction.get("fresh_run_comparison_complete") is not True
        or reproduction.get("matched") is not True
        or raw_to_truth.get("input_contract_sha256")
        != input_contract_sha256
    ):
        _fail("truth_bundle_independence_audit_invalid")

    labels = _read_canonical_jsonl(root / "primitive-labels.jsonl")
    optima = _read_canonical_jsonl(root / "small-map-optima.jsonl")
    requests = _read_canonical_jsonl(root / "requests.jsonl")
    label_audit = audit_primitive_labels(labels, manifest=manifest)
    optimum_audit = audit_small_map_optima(
        optima,
        bundle_root=root,
    )
    request_audit = audit_request_matrix(requests)
    _verify_request_certificates(root, requests)
    terrain_payloads = _read_terrain_payloads(root, requests)
    execution = build_execution_crosswalk(requests, terrain_payloads)
    cohort = execution["g3_replay_cohort"]
    if type(cohort) is not dict or len(cohort.get("rows", [])) != 6:
        _fail("g3_replay_cohort_invalid")

    ppo_targets_path = root / "ppo-targets.jsonl"
    ppo_targets = (
        _read_canonical_jsonl(ppo_targets_path)
        if artifact_io.path_is_file(ppo_targets_path)
        else None
    )
    ppo_audit = audit_ppo_targets(
        mode=mode,
        request_sha256=[
            str(row["truth_request_sha256"]) for row in requests
        ],
        ppo_target_rows=ppo_targets,
    )

    source_tar_path = root / "source" / "producer-source.tar"
    source_tar_sha256 = _sha256(artifact_io.read_bytes(source_tar_path))
    oracle_source = {
        "identity": "g2-isolated-t2-producer/v0.2.0",
        "source_bytes_sha256": source_tar_sha256,
        "implementation_sha256": manifest[
            "producer_implementation_sha256"
        ],
    }
    provider_source = (
        approval_record.get("provider_source")
        if type(approval_record) is dict
        else None
    )
    source_separation: dict[str, object] | None = None
    source_blockers: list[str] = []
    if type(provider_source) is dict:
        source_separation = audit_source_separation(
            oracle_source=oracle_source,
            provider_source=provider_source,
        )
    else:
        source_blockers.append("missing_provider_source_approval")
        provider_source = {
            "identity": "unapproved",
            "source_bytes_sha256": "0" * 64,
            "implementation_sha256": "0" * 64,
        }

    candidate_path = root / "raw" / "hopper-parameter-record.json"
    candidate_record, _ = _read_canonical_json(
        candidate_path
    )
    candidate_sha256 = _sha256(_canonical_json_bytes(candidate_record))
    if (
        manifest.get("hopper_parameter_record_sha256") != candidate_sha256
        or manifest.get("profile_or_parameter_record_sha256", {}).get(
            "hopper"
        )
        != candidate_sha256
    ):
        _fail("hopper_candidate_manifest_binding_mismatch")

    cohort_rows = list(cohort["rows"])
    input_binding = {
        "authorization_sha256": AUTHORIZATION_SHA256,
        "input_set_id": input_set_id,
        "manifest_core_sha256": manifest_core_sha256,
        "payload_root_sha256": payload_audit["payload_root_sha256"],
        "source_attestations_sha256": _sha256(attestation_bytes),
        "producer_implementation_sha256": manifest[
            "producer_implementation_sha256"
        ],
        "producer_source_bytes_sha256": source_tar_sha256,
        "reproduction_audit_sha256": _sha256(reproduction_bytes),
        "technical_isolation_audit_sha256": _sha256(technical_bytes),
        "technical_independence": attestation["technical_independence"],
        "organizational_independence": attestation[
            "organizational_independence"
        ],
        "structural_audit_passed": True,
        "g3_replay_cohort_sha256": cohort["cohort_sha256"],
        "g3_replay_truth_request_sha256": [
            row["truth_request_sha256"] for row in cohort_rows
        ],
        "g3_replay_provider_request_sha256": [
            row["provider_request_sha256"] for row in cohort_rows
        ],
    }
    runtime_source_closure = (
        capture_path_planner_runtime_source_closure()
    )
    hopper_resolution = resolve_hopper_formal_eligibility(
        candidate_record=candidate_record,
        approval_record=approval_record,
        input_binding=input_binding,
        provider_source=provider_source,
        oracle_source=oracle_source,
        current_runtime_source_closure=runtime_source_closure,
    )
    blockers = sorted(
        set(
            [
                *source_blockers,
                *hopper_resolution.get("blockers", []),
            ]
        )
    )
    formal_evidence_eligible = (
        not blockers
        and hopper_resolution.get("formal_evidence_eligible") is True
        and source_separation is not None
    )

    approval_projection = (
        {
            **dict(approval_record),
            "approval_record_sha256": _sha256(
                _canonical_json_bytes(approval_record)
            ),
            "approval_artifact_sha256": _sha256(approval_bytes),
            "approval_artifact_path": expected_approval_path.as_posix(),
        }
        if type(approval_record) is dict and approval_bytes is not None
        else None
    )
    platform_indices = {platform: 0 for platform in G2_PLATFORMS}
    request_rows: list[dict[str, object]] = []
    for crosswalk_row in execution["crosswalk"]:
        platform = str(crosswalk_row["platform"])
        request_rows.append(
            {
                **crosswalk_row,
                "request_index": platform_indices[platform],
                "expected_success": (
                    crosswalk_row["outcome_kind"] == "reachable"
                ),
                "expected_route_l2_valid": (
                    crosswalk_row["outcome_kind"] == "reachable"
                ),
            }
        )
        platform_indices[platform] += 1
    return {
        "schema_version": INPUT_AUDIT_SCHEMA_VERSION,
        "gate_id": "g2",
        "scale_profile": SCALE_PROFILE,
        "status": "ready" if formal_evidence_eligible else "blocked",
        "formal_evidence_eligible": formal_evidence_eligible,
        "blockers": blockers,
        "truth_bundle_root": str(root),
        "truth_manifest_sha256": _sha256(manifest_bytes),
        "truth_freeze_sha256": _sha256(freeze_bytes),
        "truth_payload_root_sha256": payload_audit[
            "payload_root_sha256"
        ],
        "truth_source_attestations_sha256": _sha256(attestation_bytes),
        "input_set_id": input_set_id,
        "manifest_core_sha256": manifest_core_sha256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": PLATFORM_INVARIANTS,
        "path_planner_runtime_source_closure": runtime_source_closure,
        "path_planner_runtime_source_closure_sha256": (
            runtime_source_closure[
                "path_planner_runtime_source_closure_sha256"
            ]
        ),
        "candidate_boundary": {
            "technical_independence": "T2_candidate",
            "organizational_independence": "project_internal",
            "source_attestation_status": "pending_o2_signature",
            "candidate_formal_evidence_eligible": False,
        },
        "provider_source": (
            source_separation["provider_source"]
            if source_separation is not None
            else None
        ),
        "oracle_source": oracle_source,
        "expected_approval_artifact_path": (
            expected_approval_path.as_posix()
        ),
        "approval": approval_projection,
        "hopper_resolution": hopper_resolution,
        "primitive_label_audit": label_audit,
        "small_map_optimum_audit": optimum_audit,
        "request_matrix_audit": request_audit,
        "ppo_target_audit": ppo_audit,
        "g3_replay_cohort": cohort,
        "g3_replay_cohort_sha256": cohort["cohort_sha256"],
        "requests": request_rows,
        "provider_requests": execution["provider_requests"],
        "terrain_sha256": sorted(terrain_payloads),
        "formal_row_count": 0,
    }


def _validate_config(config: Mapping[str, object]) -> None:
    if (
        type(config) is not dict
        or config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or config.get("gate_id") != "g2"
        or config.get("runner_id")
        != "run_xunce_mid_dual_g2_planning_time/v1"
        or config.get("scale_profile") != SCALE_PROFILE
        or config.get("platform_stacks") != PLATFORM_STACKS
        or config.get("approval_artifact") != APPROVAL_ARTIFACT_CONTRACT
    ):
        _fail("g2_config_contract_invalid")
    formal_inputs = config.get("formal_inputs")
    if type(formal_inputs) is not dict or set(formal_inputs) != {
        "truth_bundle",
        "artifact_bound_approval",
        "hopper_candidate_record",
        "execution_bundle",
    }:
        _fail("g2_config_formal_inputs_invalid")


def preflight(
    config: Mapping[str, object],
    *,
    truth_bundle: str | Path | None = None,
    approval_path: str | Path | None = None,
    mode: str | None = None,
) -> dict[str, object]:
    _validate_config(config)
    formal_inputs = config["formal_inputs"]
    selected_truth = truth_bundle or formal_inputs["truth_bundle"]
    selected_approval = (
        approval_path or formal_inputs["artifact_bound_approval"]
    )
    selected_mode = mode or str(config.get("evaluation_mode", "reduced"))
    if selected_truth is None:
        return {
            "schema_version": "xunce-mid-dual-g2-preflight/v1",
            "gate_id": "g2",
            "scale_profile": SCALE_PROFILE,
            "status": "blocked",
            "formal_evidence_eligible": False,
            "formal_row_count": 0,
            "blockers": [
                "missing_independent_g2_input_bundle",
                "missing_artifact_bound_o2_approval",
                "approve_midterm_hopper_simulation_proxy_parameter_set",
            ],
        }
    try:
        audit = audit_truth_bundle(
            selected_truth,
            approval_path=selected_approval,
            mode=selected_mode,
        )
    except G2InputContractError as exc:
        return {
            "schema_version": "xunce-mid-dual-g2-preflight/v1",
            "gate_id": "g2",
            "scale_profile": SCALE_PROFILE,
            "status": "blocked",
            "formal_evidence_eligible": False,
            "formal_row_count": 0,
            "blockers": [f"g2_input_invalid:{exc.code}"],
        }
    return {
        "schema_version": "xunce-mid-dual-g2-preflight/v1",
        "gate_id": "g2",
        "scale_profile": SCALE_PROFILE,
        "status": audit["status"],
        "formal_evidence_eligible": audit[
            "formal_evidence_eligible"
        ],
        "formal_row_count": 0,
        "blockers": audit["blockers"],
        "input_audit": audit,
    }


def _assert_execution_root(
    execution_root: str | Path,
    truth_root: str | Path,
) -> Path:
    raw = Path(execution_root)
    if not raw.is_absolute() or raw.drive.casefold() != "d:":
        _fail("execution_bundle_root_not_absolute_d_path")
    resolved = raw.resolve()
    approved = Path("D:/xunce/inputs/mid_dual/g2").resolve()
    try:
        resolved.relative_to(approved)
    except ValueError:
        _fail("execution_bundle_root_outside_approved_boundary")
    if os.path.normcase(str(resolved)) == os.path.normcase(
        str(Path(truth_root).resolve())
    ):
        _fail("execution_bundle_cannot_overwrite_truth")
    if artifact_io.path_exists(resolved):
        _fail("execution_bundle_root_already_exists")
    return resolved


def materialize_execution_bundle(
    bundle_root: str | Path,
    execution_root: str | Path,
    *,
    approval_path: str | Path | None = None,
    mode: str = "reduced",
    require_formal_eligibility: bool = False,
) -> dict[str, object]:
    audit = audit_truth_bundle(
        bundle_root,
        approval_path=approval_path,
        mode=mode,
    )
    if (
        require_formal_eligibility
        and audit["formal_evidence_eligible"] is not True
    ):
        _fail("formal_execution_bundle_input_not_eligible")
    output = _assert_execution_root(execution_root, bundle_root)
    artifact_io.make_dirs(output)
    artifact_io.write_jsonl(
        output / "provider-requests.jsonl",
        audit["provider_requests"],
    )
    crosswalk = [
        {
            key: row[key]
            for key in (
                "schema_version",
                "request_id",
                "platform",
                "scale",
                "request_class",
                "outcome_kind",
                "truth_request_sha256",
                "truth_certificate_sha256",
                "terrain_sha256",
                "provider_request_sha256",
            )
        }
        for row in audit["requests"]
    ]
    artifact_io.write_jsonl(
        output / "truth-provider-crosswalk.jsonl",
        crosswalk,
    )
    artifact_io.write_json(
        output / "input-audit.json",
        {
            key: value
            for key, value in audit.items()
            if key != "provider_requests"
        },
    )
    artifact_io.write_json(
        output / "g3-replay-cohort.json",
        audit["g3_replay_cohort"],
    )
    truth = Path(bundle_root).resolve()
    for terrain_sha256 in audit["terrain_sha256"]:
        artifact_io.copy_file(
            truth / "terrain" / f"{terrain_sha256}.npz",
            output / "terrain" / f"{terrain_sha256}.npz",
        )
    data_paths = [
        relative_path
        for relative_path in artifact_io.list_relative_files(output)
        if relative_path != "manifest.json"
    ]
    payload_index = [
        {
            "relative_path": relative_path,
            "byte_length": len(
                artifact_io.read_bytes(output / PurePosixPath(relative_path))
            ),
            "sha256": _sha256(
                artifact_io.read_bytes(output / PurePosixPath(relative_path))
            ),
        }
        for relative_path in data_paths
    ]
    manifest = {
        "schema_version": EXECUTION_BUNDLE_SCHEMA_VERSION,
        "scale_profile": SCALE_PROFILE,
        "input_set_id": audit["input_set_id"],
        "truth_payload_root_sha256": audit[
            "truth_payload_root_sha256"
        ],
        "truth_manifest_sha256": audit["truth_manifest_sha256"],
        "formal_evidence_eligible": audit["formal_evidence_eligible"],
        "blockers": audit["blockers"],
        "request_count": 129,
        "crosswalk_count": 129,
        "g3_replay_cohort_sha256": audit[
            "g3_replay_cohort_sha256"
        ],
        "payload_index": payload_index,
        "payload_root_sha256": _domain_hash(
            "xunce-mid-dual-g2-execution-payload-root/v1",
            _canonical_json_bytes(payload_index),
        ),
        "publication_order": "data-first-manifest-last",
    }
    artifact_io.write_json(output / "manifest.json", manifest)
    return manifest


def _provider_fine_snapshot(
    provider_request: Mapping[str, object],
    terrain_payload: bytes,
):
    """Convert one audited neutral terrain blob to the fixed 0.5 m v2 grid.

    The source representation remains immutable.  Kilometer-scale 20 m macro
    elevations are bilinearly projected to the v2 fine grid, while categorical
    obstacle/known/confidence layers are expanded without inventing physical
    obstacle claims.
    """

    from path_planner.v2.terrain import (
        FineGridGeometryV2,
        TerrainProvenanceV2,
        TerrainSnapshotV2,
    )

    terrain_binding = provider_request.get("terrain_binding")
    if type(terrain_binding) is not dict:
        _fail("provider_terrain_binding_invalid")
    if _sha256(terrain_payload) != terrain_binding.get("terrain_sha256"):
        _fail("provider_terrain_sha256_mismatch")
    arrays, metadata, geometry_sha256 = _decode_truth_terrain(terrain_payload)
    if (
        geometry_sha256 != terrain_binding.get("terrain_geometry_sha256")
        or metadata.get("geometry_schema_version")
        != terrain_binding.get("geometry_schema_version")
        or metadata.get("macro_cell_size_mm")
        != terrain_binding.get("macro_cell_size_mm")
        or metadata.get("sub_20m_layer_source_kind")
        != terrain_binding.get("sub_20m_layer_source_kind")
        or metadata.get("provenance", {}).get(
            "physical_obstacle_cells_written"
        )
        is not False
        or terrain_binding.get("physical_obstacle_cells_written") is not False
    ):
        _fail("provider_terrain_binding_mismatch")
    height, width = arrays["height_mm"].shape
    if (
        terrain_binding.get("height") != height
        or terrain_binding.get("width") != width
    ):
        _fail("provider_terrain_shape_mismatch")
    macro_mm = metadata.get("macro_cell_size_mm")
    if type(macro_mm) is not int or macro_mm <= 0 or macro_mm % 500 != 0:
        _fail("provider_terrain_resolution_invalid")
    expansion = macro_mm // 500
    fine_height = height * expansion
    fine_width = width * expansion
    macro_resolution_m = macro_mm / 1000.0
    fine_resolution_m = 0.5
    macro_x = (np.arange(width, dtype="<f8") + 0.5) * macro_resolution_m
    macro_y = (np.arange(height, dtype="<f8") + 0.5) * macro_resolution_m
    fine_x = (np.arange(fine_width, dtype="<f8") + 0.5) * fine_resolution_m
    fine_y = (np.arange(fine_height, dtype="<f8") + 0.5) * fine_resolution_m
    macro_elevation = arrays["height_mm"].astype("<f8") / 1000.0
    interpolated_x = np.vstack(
        [
            np.interp(
                fine_x,
                macro_x,
                macro_elevation[row_index],
            )
            for row_index in range(height)
        ]
    )
    elevation_m = np.ascontiguousarray(
        np.vstack(
            [
                np.interp(
                    fine_y,
                    macro_y,
                    interpolated_x[:, column_index],
                )
                for column_index in range(fine_width)
            ]
        ).T,
        dtype="<f8",
    )
    gradient_y, gradient_x = np.gradient(
        elevation_m,
        fine_resolution_m,
        fine_resolution_m,
    )
    slope_deg = np.ascontiguousarray(
        np.degrees(np.arctan(np.hypot(gradient_x, gradient_y))),
        dtype="<f8",
    )
    cell_class = np.repeat(
        np.repeat(arrays["cell_class"], expansion, axis=0),
        expansion,
        axis=1,
    )
    known = np.repeat(
        np.repeat(arrays["known"], expansion, axis=0),
        expansion,
        axis=1,
    ).astype(bool)
    confidence = np.ascontiguousarray(
        np.repeat(
            np.repeat(arrays["confidence_ppm"], expansion, axis=0),
            expansion,
            axis=1,
        ).astype("<f8")
        / 1_000_000.0,
        dtype="<f8",
    )
    hard_obstacle = np.ascontiguousarray(cell_class == 2, dtype=bool)
    traversable = np.ascontiguousarray(
        known & ~hard_obstacle & (slope_deg <= 30.0),
        dtype=bool,
    )
    source_kind = metadata.get("provenance", {}).get("source_kind")
    if type(source_kind) is not str or not source_kind:
        macro_kind = metadata.get("provenance", {}).get("macro_source_kind")
        micro_kind = metadata.get("provenance", {}).get("micro_source_kind")
        if type(macro_kind) is not str or type(micro_kind) is not str:
            _fail("provider_terrain_provenance_invalid")
        source_kind = f"{macro_kind}+{micro_kind}"
    terrain_sha256 = str(terrain_binding["terrain_sha256"])
    return TerrainSnapshotV2(
        geometry=FineGridGeometryV2(
            width=fine_width,
            height=fine_height,
            resolution_m=0.5,
        ),
        elevation_m=elevation_m,
        slope_deg=slope_deg,
        traversable_mask=traversable,
        hard_obstacle_mask=hard_obstacle,
        observed_mask=np.ascontiguousarray(known, dtype=bool),
        confidence=confidence,
        provenance=TerrainProvenanceV2(
            source_kind=source_kind,
            source_id=f"g2-neutral-terrain-{terrain_sha256[:24]}",
            source_hash=terrain_sha256,
            physical_obstacle_cells_written=False,
            details=(
                ("macro_cell_size_mm", macro_mm),
                (
                    "sub_20m_layer_source_kind",
                    str(
                        metadata.get("sub_20m_layer_source_kind")
                        or "not-applicable"
                    ),
                ),
            ),
        ),
    )


def decode_provider_execution_request(
    provider_request: Mapping[str, object],
    terrain_payload: bytes,
):
    """Validate and decode exactly one Task 7 provider request.

    This is intentionally the timed worker's input-validation boundary.  It
    accepts data, never import paths, and rechecks both the one-way provider
    request hash and the formal v2 request codec.
    """

    from path_planner.v2.contracts import (
        AcceleratorPolicyV2,
        ObjectiveProfileV2,
        PlanningRequestV2,
        PoseStateV2,
        ResourceBudgetV2,
    )
    from path_planner.v2.formal_request_codec import (
        decode_formal_request_v2,
        encode_formal_request_v2,
    )

    if type(provider_request) is not dict:
        _fail("provider_request_not_object")
    _reject_provider_truth_leak(provider_request)
    required = {
        "schema_version",
        "request_id",
        "platform",
        "scale",
        "platform_stack",
        "canonical_request_metadata",
        "terrain_binding",
        "provider_request_sha256",
    }
    if set(provider_request) != required:
        _fail("provider_request_schema_invalid")
    platform = provider_request.get("platform")
    if (
        provider_request.get("schema_version")
        != PROVIDER_REQUEST_SCHEMA_VERSION
        or platform not in G2_PLATFORMS
        or provider_request.get("platform_stack") != PLATFORM_STACKS[platform]
    ):
        _fail("provider_request_stack_invalid")
    request_sha256 = _require_sha256(
        provider_request.get("provider_request_sha256"),
        "provider_request_sha256_invalid",
    )
    envelope = {
        key: value
        for key, value in provider_request.items()
        if key != "provider_request_sha256"
    }
    expected_sha256 = _domain_hash(
        "xunce-mid-dual-g2-provider-request/v1",
        _canonical_json_bytes(envelope),
        terrain_payload,
    )
    if request_sha256 != expected_sha256:
        _fail("provider_request_sha256_mismatch")
    snapshot = _provider_fine_snapshot(provider_request, terrain_payload)
    metadata = provider_request.get("canonical_request_metadata")
    if type(metadata) is not dict:
        _fail("provider_request_metadata_invalid")
    expected_metadata_keys = {
        "accelerator_policy",
        "determinism_seed",
        "goal_state",
        "objective_profile",
        "platform_profile_id",
        "request_id",
        "resource_budget",
        "start_state",
        "timeout_s",
    }
    if (
        set(metadata) != expected_metadata_keys
        or metadata.get("request_id") != provider_request.get("request_id")
        or metadata.get("platform_profile_id")
        != PLATFORM_STACKS[platform]["profile_id"]
    ):
        _fail("provider_request_metadata_invalid")
    try:
        request = PlanningRequestV2(
            request_id=str(metadata["request_id"]),
            platform_profile_id=str(metadata["platform_profile_id"]),
            start_state=PoseStateV2(**metadata["start_state"]),
            goal_state=PoseStateV2(**metadata["goal_state"]),
            terrain_snapshot=snapshot,
            objective_profile=ObjectiveProfileV2(
                **metadata["objective_profile"]
            ),
            resource_budget=ResourceBudgetV2(
                **metadata["resource_budget"]
            ),
            timeout_s=metadata["timeout_s"],
            accelerator_policy=AcceleratorPolicyV2(
                metadata["accelerator_policy"]
            ),
            determinism_seed=metadata["determinism_seed"],
        )
        artifact = encode_formal_request_v2(request)
        decoded = decode_formal_request_v2(artifact)
    except (KeyError, TypeError, ValueError) as exc:
        raise G2InputContractError(
            "provider_request_codec_invalid",
            type(exc).__name__,
        ) from exc
    if (
        decoded.request_id != provider_request["request_id"]
        or decoded.platform_profile_id
        != PLATFORM_STACKS[platform]["profile_id"]
    ):
        _fail("provider_request_codec_binding_mismatch")
    return decoded


def build_approved_platform_execution_stack(
    platform: str,
    request: object,
) -> dict[str, object]:
    """Instantiate only the three frozen provider/profile/L2 stacks."""

    from time import monotonic

    from path_planner.v2.contracts import PlanningRequestV2, PlatformKindV2
    from path_planner.v2.hopper_api import dispatch_hopper_provider_v2
    from path_planner.v2.hopper_authority import (
        HOPPER_GENERIC_INTERNAL_SIMULATION_PROXY_PARAMETER_SET_ID_V1,
        HopperProviderAuthorityV2,
        hopper_generic_internal_simulation_proxy_midterm_v1,
    )
    from path_planner.v2.profiles import (
        LEGGED_STATIC_CRAWL_CAPABILITY_REVISION_V2,
        WHEEL_KINEMATIC_CORRIDOR_SQP_CAPABILITY_V2,
        LeggedProfileV2,
        PlatformProfileV2,
        WheelKinematicSQPProfileV2,
    )
    from path_planner.v2.providers import (
        HopperPrimitiveProviderV2,
        LeggedPrimitiveProviderV2,
        WheelKinematicSQPProviderV2,
    )
    from path_planner.v2.runtime import PlanningDeadlineV2
    from path_planner.v2.terrain import FineSafetyAnchorV2
    from path_planner.v2.validation import validate_legged_route_l2
    from path_planner.v2.hopper_route_validation import (
        validate_hopper_route_l2,
    )
    from path_planner.v2.wheel_sqp_api import dispatch_wheel_sqp_provider_v2

    if platform not in G2_PLATFORMS or type(request) is not PlanningRequestV2:
        _fail("provider_stack_request_invalid")
    if request.platform_profile_id != PLATFORM_STACKS[platform]["profile_id"]:
        _fail("provider_stack_profile_identity_mismatch")
    if platform == "wheel":
        profile = PlatformProfileV2(
            profile_id="scout-mini-wheel-kinematic-sqp/v1",
            platform_kind=PlatformKindV2.WHEEL,
            capability_revision=(
                WHEEL_KINEMATIC_CORRIDOR_SQP_CAPABILITY_V2
            ),
            simulation_proxy=False,
            max_traversable_slope_deg=30.0,
            goal_position_tolerance_m=0.25,
            goal_heading_tolerance_rad=0.08726646259971647,
        )
        execution_profile = WheelKinematicSQPProfileV2(profile=profile)
        profile = execution_profile.profile
        provider = WheelKinematicSQPProviderV2(execution_profile)
        l2_validator = None

        def plan():
            return dispatch_wheel_sqp_provider_v2(
                request,
                profile,
                provider,
                anchor,
                deadline,
            )

    elif platform == "legged":
        profile = PlatformProfileV2(
            profile_id=(
                "legged-static-crawl-simulation-proxy-midterm/v1"
            ),
            platform_kind=PlatformKindV2.LEGGED,
            capability_revision=LEGGED_STATIC_CRAWL_CAPABILITY_REVISION_V2,
            simulation_proxy=True,
            max_traversable_slope_deg=30.0,
            goal_position_tolerance_m=0.0,
            goal_heading_tolerance_rad=0.0,
        )
        execution_profile = LeggedProfileV2(profile=profile)
        provider = LeggedPrimitiveProviderV2(execution_profile)
        l2_validator = validate_legged_route_l2

        def plan():
            return provider.plan(request, anchor, deadline)

    else:
        execution_profile = (
            hopper_generic_internal_simulation_proxy_midterm_v1()
        )
        profile = execution_profile.profile
        authority = HopperProviderAuthorityV2(
            hopper_profile=execution_profile,
            parameter_set_id=(
                HOPPER_GENERIC_INTERNAL_SIMULATION_PROXY_PARAMETER_SET_ID_V1
            ),
            authority_schema_version="hopper-provider-authority/v1",
        )
        provider = HopperPrimitiveProviderV2(authority)
        l2_validator = validate_hopper_route_l2

        def plan():
            return dispatch_hopper_provider_v2(
                request,
                profile,
                provider,
                anchor,
                deadline,
            )

    anchor = FineSafetyAnchorV2(request.terrain_snapshot)
    started = monotonic()
    deadline = PlanningDeadlineV2(
        started,
        started + request.timeout_s,
        monotonic,
    )
    if (
        profile.profile_id != PLATFORM_STACKS[platform]["profile_id"]
        or profile.max_traversable_slope_deg != 30.0
    ):
        _fail("provider_stack_contract_drift")
    return {
        "platform": platform,
        "profile": profile,
        "execution_profile": execution_profile,
        "provider": provider,
        "anchor": anchor,
        "deadline": deadline,
        "l2_validator": l2_validator,
        "plan": plan,
    }


def _load_config(path: str | Path) -> dict[str, object]:
    if not artifact_io.path_is_file(path):
        _fail("g2_config_missing")
    value = artifact_io.read_json(path)
    _validate_config(value)
    return value


def _compact_preflight(result: Mapping[str, object]) -> dict[str, object]:
    compact = {
        key: value
        for key, value in result.items()
        if key != "input_audit"
    }
    audit = result.get("input_audit")
    if type(audit) is dict:
        cohort = audit.get("g3_replay_cohort")
        compact["input_audit"] = {
            "schema_version": audit.get("schema_version"),
            "status": audit.get("status"),
            "formal_evidence_eligible": audit.get(
                "formal_evidence_eligible"
            ),
            "blockers": audit.get("blockers"),
            "input_set_id": audit.get("input_set_id"),
            "truth_manifest_sha256": audit.get("truth_manifest_sha256"),
            "truth_payload_root_sha256": audit.get(
                "truth_payload_root_sha256"
            ),
            "request_count": len(audit.get("requests", [])),
            "primitive_label_count": audit.get(
                "primitive_label_audit",
                {},
            ).get("label_count"),
            "small_map_optimum_count": sum(
                audit.get("small_map_optimum_audit", {})
                .get("platform_counts", {})
                .values()
            ),
            "g3_replay_cohort_sha256": audit.get(
                "g3_replay_cohort_sha256"
            ),
            "g3_replay_request_ids": (
                [row.get("request_id") for row in cohort.get("rows", [])]
                if type(cohort) is dict
                else []
            ),
        }
    return compact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit immutable reduced-G2 truth inputs.",
    )
    parser.add_argument(
        "--config",
        default="configs/xunce_mid_dual_g2_planning_time_v1.json",
    )
    parser.add_argument("--input-bundle")
    parser.add_argument("--approval")
    parser.add_argument("--mode", choices=("reduced", "gate6"), default="reduced")
    parser.add_argument("--execution-root")
    parser.add_argument(
        "--require-formal-eligibility",
        action="store_true",
    )
    args = parser.parse_args(argv)
    try:
        config = _load_config(args.config)
        result = preflight(
            config,
            truth_bundle=args.input_bundle,
            approval_path=args.approval,
            mode=args.mode,
        )
        if args.execution_root is not None and args.input_bundle is not None:
            result = {
                **result,
                "execution_manifest": materialize_execution_bundle(
                    args.input_bundle,
                    args.execution_root,
                    approval_path=args.approval,
                    mode=args.mode,
                    require_formal_eligibility=(
                        args.require_formal_eligibility
                    ),
                ),
            }
        print(
            json.dumps(
                _compact_preflight(result),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 0
    except G2InputContractError as exc:
        print(
            json.dumps(
                {
                    "schema_version": "xunce-mid-dual-g2-preflight/v1",
                    "status": "blocked",
                    "formal_evidence_eligible": False,
                    "formal_row_count": 0,
                    "blockers": [exc.code],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "APPROVAL_ARTIFACT_BASE",
    "APPROVAL_ARTIFACT_CONTRACT",
    "APPROVAL_ARTIFACT_FILE_NAME",
    "APPROVAL_SCHEMA_VERSION",
    "CONFIG_SCHEMA_VERSION",
    "CROSSWALK_SCHEMA_VERSION",
    "G2InputContractError",
    "G3_REPLAY_COHORT_SCHEMA_VERSION",
    "INPUT_AUDIT_SCHEMA_VERSION",
    "PLATFORM_STACKS",
    "approval_artifact_path",
    "audit_ppo_targets",
    "audit_primitive_labels",
    "audit_request_matrix",
    "audit_small_map_optima",
    "audit_source_separation",
    "audit_truth_bundle",
    "build_approved_platform_execution_stack",
    "build_execution_crosswalk",
    "decode_provider_execution_request",
    "materialize_execution_bundle",
    "preflight",
    "resolve_hopper_formal_eligibility",
    "validate_primitive_label_identity",
]
