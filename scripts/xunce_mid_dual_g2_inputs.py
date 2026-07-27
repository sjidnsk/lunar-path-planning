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
from typing import Any, Callable, Iterable, Mapping, Sequence
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
_R3_PROVIDER_TRUTH_TOKENS = frozenset(
    {
        "certificate",
        "difficulty",
        "frontier",
        "optimal",
        "oracle",
        "outcome",
        "sidecar",
        "truth",
    }
)
_BINARY64_WORD_RE = re.compile(r"^[0-9a-f]{16}$")
R3_PRODUCER_SCHEMA_CONTRACT_VERSION = (
    "xunce-mid-dual-g2-producer-schema-contract/v1"
)
R3_PRODUCER_BINDING_SCHEMA_VERSION = (
    "xunce-mid-dual-g2-producer-binding/v1"
)
R3_PROVIDER_BLIND_SCHEMA_VERSION = "g2-provider-blind-request/v1"
R3_PROVIDER_EXECUTION_REQUEST_SCHEMA_VERSION = (
    "xunce-mid-dual-g2-provider-execution-request/v3"
)
R3_CONFIG_SCHEMA_VERSION = "xunce-mid-dual-g2-planning-time-config/v2"
R3_RUNNER_ID = "run_xunce_mid_dual_g2_planning_time/v2"
R3_EXECUTION_BUNDLE_SCHEMA_VERSION = (
    "xunce-mid-dual-g2-execution-bundle/v2"
)
R3_APPROVAL_SCHEMA_VERSION = (
    "xunce-mid-dual-g2-artifact-bound-approval/v2"
)
R3_APPROVAL_TARGET_SCHEMA_VERSION = (
    "xunce-mid-dual-g2-r3-approval-target/v1"
)
R3_FINAL_CANDIDATE_SCHEMA_VERSION = (
    "xunce-mid-dual-g2-r3-final-candidate/v1"
)
R3_P03_SELECTION_CONTRACT = (
    "xunce-mid-dual-g2-r3-p03-input-side-sha256-rank/v1"
)
R3_LOCAL_SNAPSHOT_SCHEMA_VERSION = (
    "xunce-mid-dual-g2-local-snapshot-normalized/v1"
)
R3_PROVIDER_LOCAL_SNAPSHOT_SCHEMA_VERSION = (
    "g2-provider-local-terrain-snapshot/v1"
)
R3_RESOURCE_POLICY_SCHEMA_VERSION = "g2-provider-resource-policy/v2"
R3_HOPPER_SUPPORT_PLANE_MODEL_ID = (
    "hopper_horizontal_same_support_full_envelope_50mm/v1"
)
R3_HOPPER_CAPABILITY_ID = (
    "simulation_proxy_generic_internal_lunar_ballistic/v3"
)
_R3_PROVIDER_BLIND_BASE_KEYS = frozenset(
    {
        "action_envelope",
        "action_envelope_sha256",
        "execution_graph",
        "frame_id",
        "goal",
        "metric_problem",
        "metric_problem_sha256",
        "objective",
        "objective_sha256",
        "platform_kind",
        "producer_implementation_sha256",
        "profile_or_parameter_record_sha256",
        "provider_local_snapshot_payload_sha256",
        "provider_local_snapshot_ref",
        "provider_local_snapshot_sha256",
        "provider_request_id",
        "provider_request_sha256",
        "resource_budget",
        "scale",
        "schema_version",
        "start",
        "terrain_geometry_sha256",
        "terrain_sha256",
        "vertical_datum",
    }
)
_R3_PROVIDER_FORBIDDEN_EXACT_KEYS = frozenset(
    {
        "cost_milli",
        "difficulty_class",
        "oracle_reachable",
        "path_edge_ids",
        "truth_certificate",
        "truth_certificate_sha256",
        "truth_request_sha256",
        "truth_sidecar",
    }
)


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


def _r3_truth_token_present(value: object) -> bool:
    """Return whether a provider-reachable object exposes truth semantics."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            folded = str(key).casefold()
            if any(token in folded for token in _R3_PROVIDER_TRUTH_TOKENS):
                return True
            if _r3_truth_token_present(child):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_r3_truth_token_present(child) for child in value)
    return False


def _r3_forbidden_provider_key_present(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _R3_PROVIDER_FORBIDDEN_EXACT_KEYS:
                return True
            if _r3_forbidden_provider_key_present(child):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(
            _r3_forbidden_provider_key_present(child) for child in value
        )
    return False


def validate_r3_producer_binding(
    binding: Mapping[str, object],
) -> dict[str, object]:
    """Validate caller-supplied final Producer identity without freezing it."""

    required = {
        "binding_sha256",
        "bundle_root",
        "hopper_parameter_record_sha256",
        "input_contract_sha256",
        "producer_implementation_sha256",
        "provider_blind_schema_version",
        "schema_version",
    }
    if type(binding) is not dict or set(binding) != required:
        _fail("r3_producer_binding")
    bundle_root = binding.get("bundle_root")
    if (
        binding.get("schema_version")
        != R3_PRODUCER_BINDING_SCHEMA_VERSION
        or binding.get("provider_blind_schema_version")
        != R3_PROVIDER_BLIND_SCHEMA_VERSION
        or type(bundle_root) is not str
        or re.fullmatch(r"[A-Za-z]:/[^\x00]+", bundle_root) is None
        or "\\" in bundle_root
        or bundle_root.endswith("/")
        or ".." in PurePosixPath(bundle_root).parts
    ):
        _fail("r3_producer_binding")
    for field in (
        "hopper_parameter_record_sha256",
        "input_contract_sha256",
        "producer_implementation_sha256",
    ):
        _require_sha256(binding.get(field), "r3_producer_binding")
    core = {
        key: binding[key] for key in required if key != "binding_sha256"
    }
    if binding.get("binding_sha256") != _domain_hash(
        R3_PRODUCER_BINDING_SCHEMA_VERSION,
        _canonical_json_bytes(core),
    ):
        _fail("r3_producer_binding")
    return dict(binding)


def _validate_r3_exact_provider_endpoint(
    endpoint: object,
    *,
    metric_endpoint: object,
    node_states: Mapping[str, object],
    snapshot_sha256: str,
) -> None:
    exact_keys = {
        "endpoint_safety",
        "node_id",
        "node_state",
        "pose_binary64_m_rad",
        "pose_mm_urad",
    }
    metric_keys = exact_keys - {"node_state"}
    if (
        type(endpoint) is not dict
        or set(endpoint) != exact_keys
        or type(metric_endpoint) is not dict
        or set(metric_endpoint) != metric_keys
        or endpoint["node_id"] not in node_states
        or endpoint["node_state"] != node_states[endpoint["node_id"]]
        or {
            key: value
            for key, value in endpoint.items()
            if key != "node_state"
        }
        != metric_endpoint
        or type(endpoint["endpoint_safety"]) is not dict
        or endpoint["endpoint_safety"].get("snapshot_sha256")
        != snapshot_sha256
        or type(endpoint["pose_mm_urad"]) is not list
        or len(endpoint["pose_mm_urad"]) != 3
        or any(type(value) is not int for value in endpoint["pose_mm_urad"])
    ):
        _fail("r3_blind_execution_join")
    projected = project_r3_canonical_pose(endpoint["pose_binary64_m_rad"])
    state = endpoint["node_state"]
    state_pose = state.get("body_pose", state.get("pose"))
    if state_pose != endpoint["pose_binary64_m_rad"]:
        _fail("r3_blind_execution_join")
    pose = projected["pose"]
    expected_display = [
        round(float(pose["x_m"]) * 1000.0),
        round(float(pose["y_m"]) * 1000.0),
        round(float(pose["heading_rad"]) * 1_000_000.0),
    ]
    if endpoint["pose_mm_urad"] != expected_display:
        _fail("r3_blind_pose_display_join")


def _validate_r3_exact_provider_blind_request(
    row: Mapping[str, object],
    producer_binding: Mapping[str, object],
) -> dict[str, object]:
    binding = validate_r3_producer_binding(producer_binding)
    if type(row) is not dict:
        _fail("r3_blind_schema")
    platform = row.get("platform_kind")
    expected_keys = set(_R3_PROVIDER_BLIND_BASE_KEYS)
    if platform == "hopper":
        expected_keys.add("hopper_parameter_record")
    if (
        platform not in G2_PLATFORMS
        or set(row) != expected_keys
        or row.get("schema_version") != R3_PROVIDER_BLIND_SCHEMA_VERSION
        or row.get("scale") not in {"standard", "kilometer"}
    ):
        _fail("r3_blind_schema")
    if (
        row.get("producer_implementation_sha256")
        != binding["producer_implementation_sha256"]
    ):
        _fail("r3_producer_binding_mismatch")
    if _r3_forbidden_provider_key_present(row):
        _fail("g2_provider_truth_leak_v3")
    for field in (
        "action_envelope_sha256",
        "metric_problem_sha256",
        "objective_sha256",
        "profile_or_parameter_record_sha256",
        "provider_local_snapshot_payload_sha256",
        "provider_local_snapshot_sha256",
        "provider_request_sha256",
        "terrain_geometry_sha256",
        "terrain_sha256",
    ):
        _require_sha256(row.get(field), "r3_blind_schema")

    execution = row.get("execution_graph")
    metric = row.get("metric_problem")
    if (
        type(execution) is not dict
        or set(execution)
        != {
            "node_state_root_sha256",
            "node_states",
            "nodes",
            "schema_version",
        }
        or execution.get("schema_version")
        != "g2-provider-execution-topology/v1"
        or type(execution.get("nodes")) is not list
        or len(execution["nodes"]) != len(set(execution["nodes"]))
        or type(execution.get("node_states")) is not dict
        or set(execution["nodes"]) != set(execution["node_states"])
        or type(metric) is not dict
        or metric.get("schema_version") != "g2-metric-planning-problem/v3"
        or metric.get("node_states") != execution["node_states"]
        or metric.get("node_state_root_sha256")
        != execution["node_state_root_sha256"]
        or metric.get("provider_local_snapshot_sha256")
        != row["provider_local_snapshot_sha256"]
        or row.get("frame_id") != metric.get("frame_id")
        or row.get("vertical_datum") != metric.get("vertical_datum")
    ):
        _fail("r3_blind_execution_join")
    expected_node_root = _domain_hash(
        "g2-request-node-state-root/v1",
        _canonical_json_bytes(execution["node_states"]),
    )
    if execution["node_state_root_sha256"] != expected_node_root:
        _fail("r3_blind_execution_join")
    _validate_r3_exact_provider_endpoint(
        row.get("start"),
        metric_endpoint=metric.get("start"),
        node_states=execution["node_states"],
        snapshot_sha256=str(row["provider_local_snapshot_sha256"]),
    )
    _validate_r3_exact_provider_endpoint(
        row.get("goal"),
        metric_endpoint=metric.get("goal"),
        node_states=execution["node_states"],
        snapshot_sha256=str(row["provider_local_snapshot_sha256"]),
    )
    if (
        row["action_envelope_sha256"]
        != _domain_hash(
            "g2-request-action-envelope/v2",
            _canonical_json_bytes(row["action_envelope"]),
        )
        or row["metric_problem_sha256"]
        != _domain_hash(
            "g2-request-metric-problem/v3",
            _canonical_json_bytes(metric),
        )
        or row["objective_sha256"]
        != _domain_hash(
            "g2-request-objective/v1",
            _canonical_json_bytes(row["objective"]),
        )
    ):
        _fail("r3_blind_execution_join")
    resource_budget = row.get("resource_budget")
    if (
        type(resource_budget) is not dict
        or set(resource_budget)
        != {
            "final_target_runtime_ms",
            "max_path_primitives",
            "midterm_max_runtime_ms",
        }
        or resource_budget.get("final_target_runtime_ms") != 1000
        or resource_budget.get("midterm_max_runtime_ms") != 2000
        or type(resource_budget.get("max_path_primitives")) is not int
        or resource_budget["max_path_primitives"] <= 0
    ):
        _fail("r3_blind_resource_budget")
    snapshot_ref = (
        "terrain/provider-local/"
        f"{row['provider_local_snapshot_sha256']}.json"
    )
    if row.get("provider_local_snapshot_ref") != snapshot_ref:
        _fail("r3_blind_snapshot_ref")

    if platform == "hopper":
        record = _validate_r3_exact_hopper_parameter_record(
            row.get("hopper_parameter_record")
        )
        record_sha256 = _sha256(_canonical_json_bytes(record))
        if (
            record_sha256 != row["profile_or_parameter_record_sha256"]
            or record_sha256
            != binding["hopper_parameter_record_sha256"]
        ):
            _fail(
                "G2I_BLOCKED_HOPPER_PARAMETER_RECORD_MISMATCH",
                "g2_hopper_parameter_record_mismatch",
            )

    core = {
        key: value
        for key, value in row.items()
        if key not in {"provider_request_id", "provider_request_sha256"}
    }
    expected_sha256 = _domain_hash(
        R3_PROVIDER_BLIND_SCHEMA_VERSION,
        _canonical_json_bytes(core),
    )
    expected_id = (
        f"g2i-provider-{platform}-{row['scale']}-"
        f"{expected_sha256[:20]}"
    )
    if (
        row.get("provider_request_sha256") != expected_sha256
        or row.get("provider_request_id") != expected_id
    ):
        _fail("r3_blind_identity")
    _canonical_json_bytes(row)
    return dict(row)


def _validated_r3_producer_schema_contract(
    contract: Mapping[str, object],
) -> dict[str, object]:
    required = {
        "schema_version",
        "provider_blind_schema_version",
        "provider_blind_exact_keys",
        "contract_sha256",
    }
    if type(contract) is not dict or set(contract) != required:
        _fail("r3_producer_schema_contract")
    exact_keys = contract.get("provider_blind_exact_keys")
    if (
        contract.get("schema_version")
        != R3_PRODUCER_SCHEMA_CONTRACT_VERSION
        or type(contract.get("provider_blind_schema_version")) is not str
        or not contract["provider_blind_schema_version"]
        or type(exact_keys) is not list
        or not exact_keys
        or any(type(key) is not str or not key for key in exact_keys)
        or exact_keys != sorted(set(exact_keys))
    ):
        _fail("r3_producer_schema_contract")
    core = {
        key: contract[key]
        for key in (
            "schema_version",
            "provider_blind_schema_version",
            "provider_blind_exact_keys",
        )
    }
    if contract.get("contract_sha256") != _domain_hash(
        R3_PRODUCER_SCHEMA_CONTRACT_VERSION,
        _canonical_json_bytes(core),
    ):
        _fail("r3_producer_schema_contract")
    return dict(contract)


def validate_r3_provider_blind_request(
    row: Mapping[str, object],
    producer_schema_contract: Mapping[str, object],
) -> dict[str, object]:
    """Validate a frozen exact row, with legacy test-fixture compatibility."""

    if (
        isinstance(producer_schema_contract, Mapping)
        and producer_schema_contract.get("schema_version")
        == R3_PRODUCER_BINDING_SCHEMA_VERSION
    ):
        return _validate_r3_exact_provider_blind_request(
            row,
            producer_schema_contract,
        )

    contract = _validated_r3_producer_schema_contract(
        producer_schema_contract
    )
    exact_keys = set(contract["provider_blind_exact_keys"])
    if (
        type(row) is not dict
        or set(row) != exact_keys
        or row.get("schema_version")
        != contract["provider_blind_schema_version"]
    ):
        _fail("r3_blind_schema")
    if _r3_truth_token_present(row):
        _fail("g2_provider_truth_leak_v3")
    if (
        row.get("platform") not in G2_PLATFORMS
        or row.get("scale") not in {"standard", "kilometer"}
        or type(row.get("request_id")) is not str
        or not row["request_id"]
        or not _is_sha256(row.get("provider_request_identity_sha256"))
    ):
        _fail("r3_blind_schema")
    _canonical_json_bytes(row)
    return dict(row)


def _r3_reachable_strings(value: object) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _r3_reachable_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _r3_reachable_strings(child)
    elif type(value) is str:
        yield value
    elif type(value) is bytes:
        yield value.decode("utf-8", errors="ignore")


def validate_r3_provider_worker_isolation(
    spec: Mapping[str, object],
    *,
    truth_sidecar_path: str,
) -> dict[str, object]:
    """Fail closed when any spawn-reachable worker object exposes truth."""

    required = {
        "schema_version",
        "provider_request",
        "terrain_payload",
        "terrain_payload_sha256",
        "environment",
        "accessible_paths",
        "working_directory",
    }
    if type(spec) is not dict or set(spec) != required:
        _fail("g2_provider_truth_leak_v3")
    environment = spec.get("environment")
    if (
        spec.get("schema_version")
        != "xunce-mid-dual-g2-provider-worker-spec/v3"
        or type(spec.get("provider_request")) is not dict
        or type(spec.get("terrain_payload")) is not bytes
        or _sha256(spec["terrain_payload"])
        != spec.get("terrain_payload_sha256")
        or type(environment) is not dict
        or any(
            type(key) is not str or type(value) is not str
            for key, value in environment.items()
        )
        or spec.get("accessible_paths") != []
        or spec.get("working_directory") is not None
    ):
        _fail("g2_provider_truth_leak_v3")
    if _r3_forbidden_provider_key_present(spec):
        _fail("g2_provider_truth_leak_v3")
    normalized_truth = truth_sidecar_path.replace("\\", "/").casefold()
    for value in _r3_reachable_strings(spec):
        if normalized_truth and normalized_truth in value.replace(
            "\\",
            "/",
        ).casefold():
            _fail("g2_provider_truth_leak_v3")
    return dict(spec)


def build_r3_provider_worker_spec(
    provider_blind_request: Mapping[str, object],
    *,
    producer_schema_contract: Mapping[str, object],
    terrain_payload: bytes,
    environment: Mapping[str, str],
) -> dict[str, object]:
    """Build the spawn payload; truth sidecars have no parameter slot."""

    request = validate_r3_provider_blind_request(
        provider_blind_request,
        producer_schema_contract,
    )
    provider_request = (
        build_r3_provider_execution_request(
            request,
            producer_binding=producer_schema_contract,
        )
        if producer_schema_contract.get("schema_version")
        == R3_PRODUCER_BINDING_SCHEMA_VERSION
        else request
    )
    if type(terrain_payload) is not bytes or not terrain_payload:
        _fail("r3_provider_terrain_payload")
    if (
        type(environment) is not dict
        or any(
            type(key) is not str
            or not key
            or type(value) is not str
            or any(
                token in key.casefold()
                for token in _R3_PROVIDER_TRUTH_TOKENS
            )
            for key, value in environment.items()
        )
    ):
        _fail("g2_provider_truth_leak_v3")
    return {
        "schema_version": "xunce-mid-dual-g2-provider-worker-spec/v3",
        "provider_request": provider_request,
        "terrain_payload": terrain_payload,
        "terrain_payload_sha256": _sha256(terrain_payload),
        "environment": dict(sorted(environment.items())),
        "accessible_paths": [],
        "working_directory": None,
    }


def _decode_r3_binary64_word(value: object, code: str) -> float:
    if type(value) is not str or _BINARY64_WORD_RE.fullmatch(value) is None:
        _fail(code)
    number = struct.unpack(">d", bytes.fromhex(value))[0]
    if (
        not math.isfinite(number)
        or (number == 0.0 and math.copysign(1.0, number) < 0.0)
        or struct.pack(">d", number).hex() != value
    ):
        _fail(code)
    return number


def project_r3_canonical_pose(
    binary64_words: Mapping[str, object],
) -> dict[str, object]:
    """Decode only canonical words; node IDs and macro scaling have no path."""

    legacy_keys = {"x_m", "y_m", "heading_rad"}
    exact_keys = {
        "heading_rad_hex",
        "heading_rad_word_hex",
        "x_m_hex",
        "x_m_word_hex",
        "y_m_hex",
        "y_m_word_hex",
    }
    if type(binary64_words) is not dict:
        _fail("r3_canonical_pose_schema")
    if set(binary64_words) == exact_keys:
        words = {
            "x_m": binary64_words["x_m_word_hex"],
            "y_m": binary64_words["y_m_word_hex"],
            "heading_rad": binary64_words["heading_rad_word_hex"],
        }
        hex_values = {
            "x_m": binary64_words["x_m_hex"],
            "y_m": binary64_words["y_m_hex"],
            "heading_rad": binary64_words["heading_rad_hex"],
        }
    elif set(binary64_words) == legacy_keys:
        words = {
            key: str(binary64_words[key]) for key in sorted(legacy_keys)
        }
        hex_values = None
    else:
        _fail("r3_canonical_pose_schema")
    pose = {
        key: _decode_r3_binary64_word(
            words[key],
            "r3_canonical_pose_binary64",
        )
        for key in ("x_m", "y_m", "heading_rad")
    }
    if any(
        struct.pack(">d", pose[key]).hex() != words[key]
        for key in legacy_keys
    ):
        _fail("r3_canonical_pose_roundtrip")
    if hex_values is not None and any(
        type(hex_values[key]) is not str
        or pose[key].hex() != hex_values[key]
        for key in legacy_keys
    ):
        _fail("r3_canonical_pose_roundtrip")
    return {
        "binary64_words": (
            dict(binary64_words)
            if set(binary64_words) == legacy_keys
            else dict(words)
        ),
        "canonical_identity": (
            None if hex_values is None else dict(binary64_words)
        ),
        "pose": pose,
    }


def validate_r3_hopper_one_ulp_witness(
    witness: Mapping[str, object],
) -> dict[str, object]:
    required = {
        "start_x_binary64",
        "goal_x_binary64",
        "modeled_range_binary64",
        "node_l2_binary64",
        "ulp_distance",
        "cost_milli",
    }
    if type(witness) is not dict or set(witness) != required:
        _fail("r3_hopper_ulp_witness")
    start = _decode_r3_binary64_word(
        witness["start_x_binary64"],
        "r3_hopper_ulp_witness",
    )
    goal = _decode_r3_binary64_word(
        witness["goal_x_binary64"],
        "r3_hopper_ulp_witness",
    )
    modeled_range = _decode_r3_binary64_word(
        witness["modeled_range_binary64"],
        "r3_hopper_ulp_witness",
    )
    node_l2 = _decode_r3_binary64_word(
        witness["node_l2_binary64"],
        "r3_hopper_ulp_witness",
    )
    observed = goal - start
    if (
        type(witness.get("ulp_distance")) is not int
        or witness["ulp_distance"] != 1
        or type(witness.get("cost_milli")) is not int
        or witness["cost_milli"] != 1389
        or struct.pack(">d", observed).hex()
        != witness["node_l2_binary64"]
        or observed != node_l2
        or math.nextafter(modeled_range, math.inf) != node_l2
        or math.nextafter(node_l2, -math.inf) != modeled_range
        or round(modeled_range * 1000.0) != witness["cost_milli"]
    ):
        _fail("r3_hopper_ulp_witness")
    return dict(witness)


def _r3_relative_relief_um(
    elevation_um: Sequence[Sequence[int]],
) -> list[list[int]]:
    reference = int(elevation_um[0][0])
    return [
        [int(value) - reference for value in row]
        for row in elevation_um
    ]


def _r3_exact_int_grid(
    value: object,
    *,
    minimum: int,
    maximum: int,
) -> list[list[int]]:
    if (
        type(value) is not list
        or len(value) != 20
        or any(type(row) is not list or len(row) != 20 for row in value)
        or any(
            type(cell) is not int or not minimum <= cell <= maximum
            for row in value
            for cell in row
        )
    ):
        _fail("r3_local_snapshot_arrays")
    return [list(row) for row in value]


def _project_r3_materialized_local_snapshot(
    snapshot: Mapping[str, object],
) -> dict[str, object]:
    required = {
        "arrays",
        "frame_id",
        "origin_mm",
        "physical_obstacle_cells_written",
        "platform_kind",
        "proxy_modification_witness",
        "relief_preservation_sha256",
        "resolution_mm",
        "schema_version",
        "shape_height_width",
        "snapshot_sha256",
        "source_kind",
        "vertical_translation",
    }
    if (
        type(snapshot) is not dict
        or set(snapshot) != required
        or snapshot.get("schema_version")
        != R3_PROVIDER_LOCAL_SNAPSHOT_SCHEMA_VERSION
        or snapshot.get("frame_id") != "g2-local-metric-frame-mm/v1"
        or snapshot.get("origin_mm") != [0, 0]
        or snapshot.get("resolution_mm") != 500
        or snapshot.get("shape_height_width") != [20, 20]
        or snapshot.get("platform_kind") not in G2_PLATFORMS
        or snapshot.get("physical_obstacle_cells_written") is not False
        or type(snapshot.get("source_kind")) is not str
        or not snapshot["source_kind"]
    ):
        _fail("r3_local_snapshot_schema")
    arrays = snapshot.get("arrays")
    if (
        type(arrays) is not dict
        or set(arrays)
        != {
            "confidence_ppm",
            "elevation_um",
            "hard_obstacle",
            "known",
            "slope_cdeg",
            "traversable",
        }
    ):
        _fail("r3_local_snapshot_arrays")
    confidence = _r3_exact_int_grid(
        arrays["confidence_ppm"],
        minimum=0,
        maximum=1_000_000,
    )
    elevation = _r3_exact_int_grid(
        arrays["elevation_um"],
        minimum=-(10**12),
        maximum=10**12,
    )
    hard_obstacle = _r3_exact_int_grid(
        arrays["hard_obstacle"],
        minimum=0,
        maximum=1,
    )
    known = _r3_exact_int_grid(
        arrays["known"],
        minimum=0,
        maximum=1,
    )
    slope = _r3_exact_int_grid(
        arrays["slope_cdeg"],
        minimum=0,
        maximum=9000,
    )
    traversable = _r3_exact_int_grid(
        arrays["traversable"],
        minimum=0,
        maximum=1,
    )

    translation = snapshot.get("vertical_translation")
    translation_keys = {
        "anchor_cell_xy",
        "anchor_node_id",
        "delta_z_um",
        "input_anchor_elevation_um",
        "input_elevation_sha256",
        "input_relief_sha256",
        "input_snapshot_sha256",
        "output_elevation_sha256",
        "output_relief_sha256",
        "semantic_kind",
        "translation_sha256",
    }
    if (
        type(translation) is not dict
        or set(translation) != translation_keys
        or translation.get("semantic_kind")
        != "uniform-vertical-translation/v1"
        or type(translation.get("anchor_cell_xy")) is not list
        or len(translation["anchor_cell_xy"]) != 2
        or any(
            type(value) is not int or not 0 <= value < 20
            for value in translation["anchor_cell_xy"]
        )
        or type(translation.get("anchor_node_id")) is not str
        or not translation["anchor_node_id"]
        or type(translation.get("delta_z_um")) is not int
        or type(translation.get("input_anchor_elevation_um")) is not int
    ):
        _fail("r3_local_snapshot_vertical_translation")
    for field in (
        "input_elevation_sha256",
        "input_relief_sha256",
        "input_snapshot_sha256",
        "output_elevation_sha256",
        "output_relief_sha256",
        "translation_sha256",
    ):
        _require_sha256(
            translation.get(field),
            "r3_local_snapshot_vertical_translation",
        )
    delta_z_um = int(translation["delta_z_um"])
    input_elevation = [
        [cell - delta_z_um for cell in row] for row in elevation
    ]
    anchor_x, anchor_y = translation["anchor_cell_xy"]
    input_anchor = input_elevation[anchor_y][anchor_x]
    if (
        input_anchor != translation["input_anchor_elevation_um"]
        or delta_z_um != -input_anchor
        or elevation[anchor_y][anchor_x] != 0
    ):
        _fail("r3_local_snapshot_vertical_translation")
    input_relief = _r3_relative_relief_um(input_elevation)
    output_relief = _r3_relative_relief_um(elevation)
    input_elevation_sha = _domain_hash(
        "g2-provider-local-elevation-um/v1",
        _canonical_json_bytes(input_elevation),
    )
    output_elevation_sha = _domain_hash(
        "g2-provider-local-elevation-um/v1",
        _canonical_json_bytes(elevation),
    )
    input_relief_sha = _domain_hash(
        "g2-provider-local-relative-relief-um/v1",
        _canonical_json_bytes(input_relief),
    )
    output_relief_sha = _domain_hash(
        "g2-provider-local-relative-relief-um/v1",
        _canonical_json_bytes(output_relief),
    )
    if (
        input_relief != output_relief
        or translation["input_elevation_sha256"] != input_elevation_sha
        or translation["output_elevation_sha256"] != output_elevation_sha
        or translation["input_relief_sha256"] != input_relief_sha
        or translation["output_relief_sha256"] != output_relief_sha
        or snapshot.get("relief_preservation_sha256") != output_relief_sha
    ):
        _fail("r3_local_snapshot_relief")
    translation_core = {
        key: value
        for key, value in translation.items()
        if key != "translation_sha256"
    }
    if translation["translation_sha256"] != _domain_hash(
        "g2-provider-uniform-vertical-translation/v1",
        _canonical_json_bytes(translation_core),
    ):
        _fail("r3_local_snapshot_vertical_translation")

    witness = snapshot.get("proxy_modification_witness")
    if (
        type(witness) is not dict
        or set(witness)
        != {
            "operations",
            "physical_obstacle_cells_written",
            "semantic_audit_sha256",
            "source_kind",
        }
        or type(witness.get("operations")) is not list
        or witness.get("physical_obstacle_cells_written") is not False
        or witness.get("source_kind")
        != "synthetic_terrain_obstacle_proxy/v1"
    ):
        _fail("r3_local_snapshot_proxy_semantics")
    witness_core = {
        key: value
        for key, value in witness.items()
        if key != "semantic_audit_sha256"
    }
    if witness.get("semantic_audit_sha256") != _domain_hash(
        "g2-provider-local-proxy-semantic-audit/v1",
        _canonical_json_bytes(witness_core),
        _canonical_json_bytes(input_relief),
        _canonical_json_bytes(hard_obstacle),
    ):
        _fail("r3_local_snapshot_proxy_semantics")
    snapshot_core = {
        key: value
        for key, value in snapshot.items()
        if key != "snapshot_sha256"
    }
    snapshot_sha256 = _domain_hash(
        R3_PROVIDER_LOCAL_SNAPSHOT_SCHEMA_VERSION,
        _canonical_json_bytes(snapshot_core),
    )
    if snapshot.get("snapshot_sha256") != snapshot_sha256:
        _fail("r3_local_snapshot_hash")
    return {
        "shape": (20, 20),
        "resolution_m": 0.5,
        "origin_m": (0.0, 0.0),
        "frame_id": snapshot["frame_id"],
        "platform_kind": snapshot["platform_kind"],
        "elevation_m": (
            np.asarray(elevation, dtype="<f8") / 1_000_000.0
        ),
        "slope_deg": np.asarray(slope, dtype="<f8") / 100.0,
        "hard_obstacle": np.asarray(hard_obstacle, dtype="u1"),
        "known": np.asarray(known, dtype="u1"),
        "traversable": np.asarray(traversable, dtype="u1"),
        "confidence_ppm": np.asarray(confidence, dtype="<u4"),
        "source_snapshot_sha256": snapshot_sha256,
        "source_payload_sha256": _sha256(
            _canonical_json_bytes(snapshot)
        ),
        "relief_preservation_sha256": output_relief_sha,
        "projection_sha256": _domain_hash(
            "xunce-mid-dual-g2-local-snapshot-projection/v1",
            snapshot_sha256.encode("ascii"),
        ),
    }


def project_r3_local_snapshot(
    snapshot: Mapping[str, object],
    *,
    scale: str,
) -> dict[str, object]:
    """Validate the normalized 10 m snapshot without any resampling branch."""

    if scale not in {"standard", "kilometer"}:
        _fail("r3_local_snapshot_scale")
    if (
        isinstance(snapshot, Mapping)
        and snapshot.get("schema_version")
        == R3_PROVIDER_LOCAL_SNAPSHOT_SCHEMA_VERSION
    ):
        return _project_r3_materialized_local_snapshot(snapshot)
    core_keys = {
        "schema_version",
        "width",
        "height",
        "resolution_m_binary64",
        "origin_x_m_binary64",
        "origin_y_m_binary64",
        "frame_id",
        "vertical_datum_id",
        "elevation_m_binary64",
        "slope_deg_binary64",
        "cell_class",
        "known",
        "confidence_ppm",
        "physical_obstacle_cells_written",
    }
    if (
        type(snapshot) is not dict
        or set(snapshot) != {*core_keys, "snapshot_sha256"}
        or snapshot.get("schema_version")
        != R3_LOCAL_SNAPSHOT_SCHEMA_VERSION
        or snapshot.get("width") != 20
        or snapshot.get("height") != 20
        or snapshot.get("physical_obstacle_cells_written") is not False
        or type(snapshot.get("frame_id")) is not str
        or not snapshot["frame_id"]
        or type(snapshot.get("vertical_datum_id")) is not str
        or not snapshot["vertical_datum_id"]
    ):
        _fail("r3_local_snapshot_schema")
    core = {key: snapshot[key] for key in core_keys}
    expected_hash = _domain_hash(
        R3_LOCAL_SNAPSHOT_SCHEMA_VERSION,
        _canonical_json_bytes(core),
    )
    if snapshot.get("snapshot_sha256") != expected_hash:
        _fail("r3_local_snapshot_hash")
    resolution = _decode_r3_binary64_word(
        snapshot["resolution_m_binary64"],
        "r3_local_snapshot_binary64",
    )
    origin_x = _decode_r3_binary64_word(
        snapshot["origin_x_m_binary64"],
        "r3_local_snapshot_binary64",
    )
    origin_y = _decode_r3_binary64_word(
        snapshot["origin_y_m_binary64"],
        "r3_local_snapshot_binary64",
    )
    if resolution != 0.5:
        _fail("r3_local_snapshot_resolution")
    count = 400
    word_fields = ("elevation_m_binary64", "slope_deg_binary64")
    for field in word_fields:
        values = snapshot.get(field)
        if type(values) is not list or len(values) != count:
            _fail("r3_local_snapshot_shape")
    elevation = np.asarray(
        [
            _decode_r3_binary64_word(
                value,
                "r3_local_snapshot_binary64",
            )
            for value in snapshot["elevation_m_binary64"]
        ],
        dtype="<f8",
    ).reshape((20, 20))
    slope = np.asarray(
        [
            _decode_r3_binary64_word(
                value,
                "r3_local_snapshot_binary64",
            )
            for value in snapshot["slope_deg_binary64"]
        ],
        dtype="<f8",
    ).reshape((20, 20))
    cell_class = snapshot.get("cell_class")
    known = snapshot.get("known")
    confidence = snapshot.get("confidence_ppm")
    if (
        type(cell_class) is not list
        or len(cell_class) != count
        or any(type(value) is not int or value not in {0, 2} for value in cell_class)
        or type(known) is not list
        or len(known) != count
        or any(type(value) is not int or value not in {0, 1} for value in known)
        or type(confidence) is not list
        or len(confidence) != count
        or any(
            type(value) is not int or not 0 <= value <= 1_000_000
            for value in confidence
        )
    ):
        _fail("r3_local_snapshot_arrays")
    return {
        "shape": (20, 20),
        "resolution_m": resolution,
        "origin_m": (origin_x, origin_y),
        "frame_id": snapshot["frame_id"],
        "vertical_datum_id": snapshot["vertical_datum_id"],
        "elevation_m": elevation,
        "slope_deg": slope,
        "cell_class": np.asarray(cell_class, dtype="u1").reshape((20, 20)),
        "known": np.asarray(known, dtype="u1").reshape((20, 20)),
        "confidence_ppm": np.asarray(
            confidence,
            dtype="<u4",
        ).reshape((20, 20)),
        "source_snapshot_sha256": expected_hash,
        "projection_sha256": _domain_hash(
            "xunce-mid-dual-g2-local-snapshot-projection/v1",
            expected_hash.encode("ascii"),
        ),
    }


def _validate_r3_exact_hopper_parameter_record(
    parameter_record: object,
) -> dict[str, object]:
    required = {
        "arc_clearance_margin_m",
        "body_envelope_radius_m",
        "energy_model",
        "evidence_class",
        "formal_evidence_eligible",
        "landing_footprint_radius_m",
        "launch_reference_height_m",
        "parameter_set_id",
        "physical_capability_claimed",
        "schema_version",
        "simulation_proxy",
        "status",
        "stop_condition",
    }
    if type(parameter_record) is not dict or set(parameter_record) != required:
        _fail(
            "G2I_BLOCKED_HOPPER_PARAMETER_RECORD_INVALID",
            "g2_hopper_parameter_record_mismatch",
        )
    energy = parameter_record.get("energy_model")
    stop = parameter_record.get("stop_condition")
    if (
        parameter_record.get("schema_version") != "g2-hopper-candidate/v2"
        or parameter_record.get("parameter_set_id")
        != "hopper-generic-internal-proxy/v2"
        or parameter_record.get("evidence_class")
        != "candidate_engineering_proxy"
        or parameter_record.get("formal_evidence_eligible") is not False
        or parameter_record.get("physical_capability_claimed") is not False
        or parameter_record.get("simulation_proxy") is not True
        or parameter_record.get("status") != "pending_external_evidence"
        or parameter_record.get("arc_clearance_margin_m") != "0.125"
        or parameter_record.get("body_envelope_radius_m") != "0.375"
        or parameter_record.get("landing_footprint_radius_m") != "0.625"
        or parameter_record.get("launch_reference_height_m") != "0.750"
        or type(energy) is not dict
        or set(energy)
        != {
            "evaluator_relative_path",
            "evaluator_source_sha256",
            "max_energy_decimal",
            "model_id",
            "reference_speed_m_s",
        }
        or energy.get("evaluator_relative_path")
        != "producer/hopper_energy_evaluator.py"
        or energy.get("max_energy_decimal") != "1.000000"
        or energy.get("model_id") != "quadratic-normalized-speed/v1"
        or energy.get("reference_speed_m_s") != "2.500"
        or type(stop) is not dict
        or set(stop)
        != {
            "evaluator_relative_path",
            "evaluator_source_sha256",
            "max_touchdown_speed_m_s",
            "model_id",
        }
        or stop.get("evaluator_relative_path")
        != "producer/hopper_stop_evaluator.py"
        or stop.get("max_touchdown_speed_m_s") != "2.500"
        or stop.get("model_id") != "touchdown-speed-upper-bound/v1"
    ):
        _fail(
            "G2I_BLOCKED_HOPPER_PARAMETER_RECORD_INVALID",
            "g2_hopper_parameter_record_mismatch",
        )
    _require_sha256(
        energy.get("evaluator_source_sha256"),
        "G2I_BLOCKED_HOPPER_PARAMETER_RECORD_INVALID",
    )
    _require_sha256(
        stop.get("evaluator_source_sha256"),
        "G2I_BLOCKED_HOPPER_PARAMETER_RECORD_INVALID",
    )
    _canonical_json_bytes(parameter_record)
    return dict(parameter_record)


def _validate_r3_exact_hopper_execution_binding(
    parameter_record: Mapping[str, object],
    execution_binding: Mapping[str, object],
) -> dict[str, object]:
    record = _validate_r3_exact_hopper_parameter_record(parameter_record)
    required = {
        "parameter_record_sha256",
        "provider_local_snapshot_sha256",
        "relief_preservation_sha256",
        "support_plane",
    }
    if type(execution_binding) is not dict or set(execution_binding) != required:
        _fail(
            "G2I_BLOCKED_HOPPER_PARAMETER_RECORD_MISMATCH",
            "g2_hopper_parameter_record_mismatch",
        )
    for field in (
        "parameter_record_sha256",
        "provider_local_snapshot_sha256",
        "relief_preservation_sha256",
    ):
        _require_sha256(
            execution_binding.get(field),
            "G2I_BLOCKED_HOPPER_PARAMETER_RECORD_MISMATCH",
        )
    record_sha256 = _sha256(_canonical_json_bytes(record))
    support = execution_binding.get("support_plane")
    support_keys = {
        "H_ref_m_hex",
        "H_ref_m_word_hex",
        "H_ref_um",
        "anchor_node_id",
        "horizontal",
        "normal",
        "schema_version",
        "snapshot_sha256",
        "support_plane_sha256",
    }
    if (
        execution_binding.get("parameter_record_sha256") != record_sha256
        or type(support) is not dict
        or set(support) != support_keys
        or support.get("schema_version")
        != "g2-horizontal-support-plane/v1"
        or support.get("horizontal") is not True
        or support.get("normal") != [0, 0, 1]
        or type(support.get("anchor_node_id")) is not str
        or not support["anchor_node_id"]
        or support.get("snapshot_sha256")
        != execution_binding["provider_local_snapshot_sha256"]
    ):
        _fail(
            "G2I_BLOCKED_HOPPER_PARAMETER_RECORD_MISMATCH",
            "g2_hopper_parameter_record_mismatch",
        )
    reference_height = _decode_r3_binary64_word(
        support.get("H_ref_m_word_hex"),
        "G2I_BLOCKED_HOPPER_PARAMETER_RECORD_MISMATCH",
    )
    if (
        type(support.get("H_ref_m_hex")) is not str
        or reference_height.hex() != support["H_ref_m_hex"]
        or type(support.get("H_ref_um")) is not int
        or round(reference_height * 1_000_000.0) != support["H_ref_um"]
    ):
        _fail(
            "G2I_BLOCKED_HOPPER_PARAMETER_RECORD_MISMATCH",
            "g2_hopper_parameter_record_mismatch",
        )
    support_core = {
        key: value
        for key, value in support.items()
        if key != "support_plane_sha256"
    }
    if support.get("support_plane_sha256") != _domain_hash(
        "g2-horizontal-support-plane/v1",
        _canonical_json_bytes(support_core),
    ):
        _fail(
            "G2I_BLOCKED_HOPPER_PARAMETER_RECORD_MISMATCH",
            "g2_hopper_parameter_record_mismatch",
        )
    return dict(execution_binding)


def validate_r3_hopper_execution_binding(
    parameter_record: Mapping[str, object] | None,
    execution_binding: Mapping[str, object],
) -> dict[str, object]:
    """Bind the provider-visible support datum to one exact Hopper record."""

    if parameter_record is None:
        _fail(
            "G2I_BLOCKED_HOPPER_PARAMETER_RECORD_MISSING",
            "g2_hopper_parameter_record_missing",
        )
    if parameter_record.get("schema_version") == "g2-hopper-candidate/v2":
        return _validate_r3_exact_hopper_execution_binding(
            parameter_record,
            execution_binding,
        )
    record_required = {
        "schema_version",
        "parameter_set_id",
        "capability_revision",
        "support_plane_model_id",
        "support_height_tolerance_m_binary64",
        "relief_preservation_required",
        "record_sha256",
    }
    binding_required = {
        "parameter_record_sha256",
        "support_plane_model_id",
        "support_reference_height_m_binary64",
        "required_cells_sha256",
        "relief_preservation_required",
    }
    if (
        type(parameter_record) is not dict
        or not record_required.issubset(parameter_record)
        or type(execution_binding) is not dict
        or set(execution_binding) != binding_required
    ):
        _fail("g2_hopper_parameter_record_mismatch")
    record_core = {
        key: value
        for key, value in parameter_record.items()
        if key != "record_sha256"
    }
    record_sha256 = _domain_hash(
        "xunce-mid-dual-g2-hopper-record-binding/v1",
        _canonical_json_bytes(record_core),
    )
    _decode_r3_binary64_word(
        parameter_record["support_height_tolerance_m_binary64"],
        "g2_hopper_parameter_record_mismatch",
    )
    _decode_r3_binary64_word(
        execution_binding["support_reference_height_m_binary64"],
        "g2_hopper_parameter_record_mismatch",
    )
    if (
        parameter_record.get("schema_version")
        != "hopper-parameter-set-record/v1"
        or parameter_record.get("capability_revision")
        != R3_HOPPER_CAPABILITY_ID
        or parameter_record.get("support_plane_model_id")
        != R3_HOPPER_SUPPORT_PLANE_MODEL_ID
        or parameter_record.get("support_height_tolerance_m_binary64")
        != struct.pack(">d", 0.05).hex()
        or parameter_record.get("relief_preservation_required") is not True
        or parameter_record.get("record_sha256") != record_sha256
        or execution_binding.get("parameter_record_sha256")
        != record_sha256
        or execution_binding.get("support_plane_model_id")
        != parameter_record["support_plane_model_id"]
        or execution_binding.get("relief_preservation_required") is not True
        or not _is_sha256(execution_binding.get("required_cells_sha256"))
    ):
        _fail("g2_hopper_parameter_record_mismatch")
    return dict(execution_binding)


_R3_RESOURCE_POLICY_CORES: dict[str, dict[str, object]] = {
    "wheel": {
        "schema_version": R3_RESOURCE_POLICY_SCHEMA_VERSION,
        "platform": "wheel",
        "capability_id": "wheel_kinematic_corridor_sqp/v1",
        "work_budget_basis": "wheel_explicit_corridor_sqp/v1",
        "graph_hops": 1,
        "provider_primitives_per_graph_hop": 1,
        "max_expanded_states": 8192,
        "max_route_states": 129,
        "max_memory_bytes": 33_554_432,
    },
    "legged": {
        "schema_version": R3_RESOURCE_POLICY_SCHEMA_VERSION,
        "platform": "legged",
        "capability_id": "simulation_proxy_static_crawl/v2",
        "work_budget_basis": "legged_four_phase_static_crawl/v2",
        "graph_hops": 1,
        "provider_primitives_per_graph_hop": 4,
        "max_expanded_states": 10_000,
        "max_route_states": 5,
        "max_memory_bytes": 0,
    },
    "hopper": {
        "schema_version": R3_RESOURCE_POLICY_SCHEMA_VERSION,
        "platform": "hopper",
        "capability_id": R3_HOPPER_CAPABILITY_ID,
        "work_budget_basis": "hopper_exact_goal_one_ballistic/v1",
        "graph_hops": 1,
        "provider_primitives_per_graph_hop": 1,
        "max_expanded_states": 1,
        "max_route_states": 2,
        "max_memory_bytes": 536_870_912,
    },
}


def r3_provider_resource_policy(platform: str) -> dict[str, object]:
    core = _R3_RESOURCE_POLICY_CORES.get(platform)
    if core is None:
        _fail("g2_provider_resource_policy")
    return {
        **core,
        "resource_policy_sha256": _domain_hash(
            R3_RESOURCE_POLICY_SCHEMA_VERSION,
            _canonical_json_bytes(core),
        ),
    }


def validate_r3_provider_resource_policy(
    policy: Mapping[str, object],
) -> dict[str, object]:
    if type(policy) is not dict:
        _fail("g2_provider_resource_policy")
    platform = policy.get("platform")
    expected = (
        r3_provider_resource_policy(platform)
        if type(platform) is str and platform in G2_PLATFORMS
        else None
    )
    if expected is None or dict(policy) != expected:
        _fail("g2_provider_resource_policy")
    return dict(policy)


def build_r3_provider_execution_request(
    provider_blind_request: Mapping[str, object],
    *,
    producer_binding: Mapping[str, object],
) -> dict[str, object]:
    """Bind one blind request to its immutable platform resource policy."""

    binding = validate_r3_producer_binding(producer_binding)
    request = _validate_r3_exact_provider_blind_request(
        provider_blind_request,
        binding,
    )
    policy = r3_provider_resource_policy(str(request["platform_kind"]))
    core = {
        "producer_binding_sha256": binding["binding_sha256"],
        "provider_blind_request": request,
        "provider_request_sha256": request["provider_request_sha256"],
        "resource_policy": policy,
        "resource_policy_sha256": policy["resource_policy_sha256"],
        "schema_version": R3_PROVIDER_EXECUTION_REQUEST_SCHEMA_VERSION,
    }
    return {
        **core,
        "execution_request_sha256": _domain_hash(
            R3_PROVIDER_EXECUTION_REQUEST_SCHEMA_VERSION,
            _canonical_json_bytes(core),
        ),
    }


def validate_r3_parent_static_hop_resource_join(
    provider_blind_requests: Sequence[Mapping[str, object]],
    truth_sidecars: Sequence[Mapping[str, object]],
    *,
    producer_binding: Mapping[str, object],
) -> dict[str, object]:
    """Join truth only in p01 parent memory and emit a truth-free crosswalk."""

    binding = validate_r3_producer_binding(producer_binding)
    if (
        not isinstance(provider_blind_requests, Sequence)
        or isinstance(provider_blind_requests, (str, bytes))
        or not provider_blind_requests
        or not isinstance(truth_sidecars, Sequence)
        or isinstance(truth_sidecars, (str, bytes))
        or len(truth_sidecars) != len(provider_blind_requests)
    ):
        _fail("g2_r3_parent_static_join")
    requests = [
        _validate_r3_exact_provider_blind_request(row, binding)
        for row in provider_blind_requests
    ]
    requests_by_sha = {
        str(row["provider_request_sha256"]): row for row in requests
    }
    if len(requests_by_sha) != len(requests):
        _fail("g2_r3_parent_static_join")

    exact_sidecar_keys = {
        "provider_request_id",
        "provider_request_sha256",
        "schema_version",
        "truth_request",
    }
    truth_required = {
        "action_envelope_sha256",
        "difficulty_class",
        "goal",
        "metric_problem_sha256",
        "objective_sha256",
        "oracle_reachable",
        "platform_kind",
        "producer_implementation_sha256",
        "profile_or_parameter_record_sha256",
        "provider_local_snapshot_payload_sha256",
        "provider_local_snapshot_sha256",
        "request_hop_count",
        "request_id",
        "resource_budget",
        "scale",
        "schema_version",
        "start",
        "terrain_geometry_sha256",
        "terrain_sha256",
        "truth_request_sha256",
    }
    join_fields = (
        "action_envelope_sha256",
        "goal",
        "metric_problem_sha256",
        "objective_sha256",
        "platform_kind",
        "producer_implementation_sha256",
        "profile_or_parameter_record_sha256",
        "provider_local_snapshot_payload_sha256",
        "provider_local_snapshot_sha256",
        "resource_budget",
        "scale",
        "start",
        "terrain_geometry_sha256",
        "terrain_sha256",
    )
    seen_request_sha256: set[str] = set()
    crosswalk: list[dict[str, object]] = []
    reachable_hop_counts: dict[str, list[int]] = {}
    for sidecar in truth_sidecars:
        if (
            type(sidecar) is not dict
            or set(sidecar) != exact_sidecar_keys
            or sidecar.get("schema_version")
            != "g2-truth-request-sidecar/v1"
        ):
            _fail("g2_r3_truth_sidecar_schema")
        provider_sha256 = sidecar.get("provider_request_sha256")
        _require_sha256(
            provider_sha256,
            "g2_r3_truth_sidecar_schema",
        )
        request = requests_by_sha.get(str(provider_sha256))
        if (
            request is None
            or provider_sha256 in seen_request_sha256
            or sidecar.get("provider_request_id")
            != request["provider_request_id"]
        ):
            _fail("g2_r3_truth_sidecar_join")
        truth = sidecar.get("truth_request")
        if (
            type(truth) is not dict
            or not truth_required.issubset(truth)
            or truth.get("schema_version") != "g2-truth-request/v4"
            or any(truth.get(field) != request[field] for field in join_fields)
        ):
            _fail("g2_r3_truth_sidecar_join")
        _require_sha256(
            truth.get("truth_request_sha256"),
            "g2_r3_truth_sidecar_join",
        )
        difficulty = truth.get("difficulty_class")
        oracle_reachable = truth.get("oracle_reachable")
        hop_count = truth.get("request_hop_count")
        if (
            difficulty
            not in {"reachable", "hard_reachable", "unreachable"}
            or type(oracle_reachable) is not bool
            or type(hop_count) is not int
            or hop_count < 0
            or (
                difficulty in {"reachable", "hard_reachable"}
                and (oracle_reachable is not True or hop_count <= 0)
            )
            or (
                difficulty == "unreachable"
                and (oracle_reachable is not False or hop_count != 0)
            )
        ):
            _fail("g2_r3_truth_sidecar_semantics")

        execution_request = build_r3_provider_execution_request(
            request,
            producer_binding=binding,
        )
        policy = execution_request["resource_policy"]
        platform = str(request["platform_kind"])
        if difficulty in {"reachable", "hard_reachable"}:
            reachable_hop_counts.setdefault(platform, []).append(hop_count)
            if platform in {"legged", "hopper"}:
                required_primitives = (
                    hop_count
                    * int(policy["provider_primitives_per_graph_hop"])
                )
                required_route_states = required_primitives + 1
                if (
                    hop_count != policy["graph_hops"]
                    or required_primitives
                    != policy["provider_primitives_per_graph_hop"]
                    or required_route_states != policy["max_route_states"]
                ):
                    _fail(
                        "g2_r3_hop_resource_mismatch",
                        (
                            f"platform={platform};hop_count={hop_count};"
                            f"required_primitives={required_primitives};"
                            "required_route_states="
                            f"{required_route_states}"
                        ),
                    )
        crosswalk.append(
            {
                "execution_request_sha256": execution_request[
                    "execution_request_sha256"
                ],
                "platform_kind": platform,
                "provider_request_id": request["provider_request_id"],
                "provider_request_sha256": provider_sha256,
                "resource_policy": policy,
                "resource_policy_sha256": execution_request[
                    "resource_policy_sha256"
                ],
                "scale": request["scale"],
            }
        )
        seen_request_sha256.add(str(provider_sha256))
    if seen_request_sha256 != set(requests_by_sha):
        _fail("g2_r3_truth_sidecar_join")
    if _r3_forbidden_provider_key_present(crosswalk):
        _fail("g2_provider_truth_leak_v3")
    return {
        "schema_version": (
            "xunce-mid-dual-g2-parent-static-hop-resource-audit/v1"
        ),
        "status": "passed",
        "producer_binding_sha256": binding["binding_sha256"],
        "request_count": len(requests),
        "reachable_hop_counts": {
            platform: sorted(values)
            for platform, values in sorted(reachable_hop_counts.items())
        },
        "execution_crosswalk": sorted(
            crosswalk,
            key=lambda row: str(row["provider_request_sha256"]),
        ),
    }


_R3_FORMAL_INPUT_KEYS = {
    "producer_candidate_bundle",
    "producer_manifest_sha256",
    "provider_blind_requests_sha256",
    "truth_request_sidecar_sha256",
    "hopper_parameter_record_sha256",
    "execution_bundle",
    "execution_manifest_sha256",
    "artifact_bound_approval",
    "artifact_bound_approval_sha256",
}
_R3_PRODUCER_BINDING_KEYS = {
    "schema_version",
    "bundle_root",
    "producer_implementation_sha256",
    "input_contract_sha256",
    "hopper_parameter_record_sha256",
    "provider_blind_schema_version",
    "binding_sha256",
}
_R3_ACTIVATION_BINDING_KEYS = {
    "candidate_id",
    "input_set_id",
    "manifest_core_sha256",
    "producer_payload_root_sha256",
    "producer_repeatability_audit_sha256",
    "provider_runtime_source_closure_sha256",
    "consumer_source_closure_sha256",
    "consumer_activation_contract_sha256",
    "resource_policy_root_sha256",
    "execution_request_root_sha256",
    "p03_probe_selection_sha256",
    "formal_schedule_sha256",
    "execution_data_root_sha256",
    "approval_target_sha256",
}
_R3_FINAL_CANDIDATE_KEYS = {
    "schema_version",
    "fixture_only",
    "formal_evidence_eligible",
    "formal_ineligibility_reason",
    "candidate_id",
    "input_set_id",
    "producer_binding_sha256",
    "producer_manifest_file_sha256",
    "manifest_core_sha256",
    "producer_payload_root_sha256",
    "provider_blind_requests_file_sha256",
    "truth_request_sidecar_file_sha256",
    "source_attestations_file_sha256",
    "source_attestation_sha256",
    "provider_local_snapshot_index_root_sha256",
    "request_graph_archive_root_sha256",
    "hopper_parameter_record_sha256",
    "producer_repeatability_audit_sha256",
    "producer_repeatability_passed",
    "request_count",
}
_R3_APPROVAL_TARGET_KEYS = {
    "schema_version",
    "candidate_id",
    "input_set_id",
    "producer_binding_sha256",
    "producer_manifest_file_sha256",
    "manifest_core_sha256",
    "producer_payload_root_sha256",
    "producer_repeatability_audit_sha256",
    "provider_blind_requests_file_sha256",
    "truth_request_sidecar_file_sha256",
    "hopper_parameter_record_sha256",
    "provider_runtime_source_closure_sha256",
    "consumer_source_closure_sha256",
    "consumer_activation_contract_sha256",
    "resource_policy_root_sha256",
    "execution_request_root_sha256",
    "p03_probe_selection_sha256",
    "formal_schedule_sha256",
    "execution_data_root_sha256",
    "request_count",
    "formal_call_count",
}
R3_CONSUMER_SOURCE_RELATIVE_PATHS = (
    "scripts/run_xunce_mid_dual_g2_planning_time.py",
    "scripts/xunce_artifact_io.py",
    "scripts/xunce_artifact_paths.py",
    "scripts/xunce_mid_dual_artifacts.py",
    "scripts/xunce_mid_dual_contracts.py",
    "scripts/xunce_mid_dual_g2_inputs.py",
)


def _r3_contains_placeholder(value: object) -> bool:
    if type(value) is str:
        return "${" in value
    if isinstance(value, Mapping):
        return any(
            _r3_contains_placeholder(key)
            or _r3_contains_placeholder(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_r3_contains_placeholder(child) for child in value)
    return False


def _r3_activation_contract_core(
    config: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": (
            "xunce-mid-dual-g2-r3-consumer-activation-contract/v1"
        ),
        "gate_id": config["gate_id"],
        "scale_profile": config["scale_profile"],
        "request_contract": config["request_contract"],
        "phase_contract": config["phase_contract"],
        "execution": {
            key: value
            for key, value in config["execution"].items()
            if key != "formal_environment_gate"
        },
        "thresholds_ms": config["thresholds_ms"],
        "platform_stacks": config["platform_stacks"],
        "schemas": config["schemas"],
    }


def r3_consumer_activation_contract_sha256(
    config: Mapping[str, object],
) -> str:
    return _domain_hash(
        "xunce-mid-dual-g2-r3-consumer-activation-contract/v1",
        _canonical_json_bytes(_r3_activation_contract_core(config)),
    )


def capture_r3_consumer_source_closure(
    *,
    source_reader: Callable[[Path], bytes] | None = None,
) -> dict[str, object]:
    """Hash the exact noncyclic Consumer runtime sources at activation time."""

    reader = artifact_io.read_bytes if source_reader is None else source_reader
    if not callable(reader):
        _fail("g2_r3_source_closure")
    rows: list[dict[str, object]] = []
    for relative_path in R3_CONSUMER_SOURCE_RELATIVE_PATHS:
        path = REPO_ROOT / relative_path
        if not artifact_io.path_is_file(path):
            _fail("g2_r3_source_closure")
        payload = reader(path)
        if type(payload) is not bytes:
            _fail("g2_r3_source_closure")
        rows.append(
            {
                "relative_path": relative_path,
                "size_bytes": len(payload),
                "sha256": _sha256(payload),
            }
        )
    return {
        "schema_version": (
            "xunce-mid-dual-g2-r3-consumer-source-closure/v1"
        ),
        "files": rows,
        "consumer_source_closure_sha256": _domain_hash(
            "xunce-mid-dual-g2-r3-consumer-source-closure/v1",
            _canonical_json_bytes(rows),
        ),
    }


def validate_r3_activation_config(
    config: Mapping[str, object],
    *,
    require_resolved: bool,
) -> dict[str, object]:
    """Validate the v2 shape while keeping the checked-in template blocked."""

    expected_top_keys = {
        "schema_version",
        "gate_id",
        "runner_id",
        "scale_profile",
        "evaluation_mode",
        "output_base",
        "formal_inputs",
        "producer_binding",
        "activation_binding",
        "request_contract",
        "phase_contract",
        "execution",
        "thresholds_ms",
        "platform_stacks",
        "schemas",
        "default_readiness",
    }
    if (
        type(config) is not dict
        or set(config) != expected_top_keys
        or config.get("schema_version") != R3_CONFIG_SCHEMA_VERSION
        or config.get("gate_id") != "g2"
        or config.get("runner_id") != R3_RUNNER_ID
        or config.get("scale_profile") != SCALE_PROFILE
        or config.get("evaluation_mode") != "reduced"
        or config.get("output_base") != "D:/xunce/out/mid_dual/g2"
        or _r3_contains_placeholder(config)
    ):
        if _r3_contains_placeholder(config):
            _fail("g2_r3_config_placeholder_unresolved")
        _fail("g2_r3_config_invalid")

    formal_inputs = config["formal_inputs"]
    producer_binding = config["producer_binding"]
    activation_binding = config["activation_binding"]
    if (
        type(formal_inputs) is not dict
        or set(formal_inputs) != _R3_FORMAL_INPUT_KEYS
        or type(producer_binding) is not dict
        or set(producer_binding) != _R3_PRODUCER_BINDING_KEYS
        or producer_binding.get("schema_version")
        != R3_PRODUCER_BINDING_SCHEMA_VERSION
        or producer_binding.get("provider_blind_schema_version")
        != R3_PROVIDER_BLIND_SCHEMA_VERSION
        or type(activation_binding) is not dict
        or set(activation_binding) != _R3_ACTIVATION_BINDING_KEYS
    ):
        _fail("g2_r3_config_invalid")

    expected_request_contract = {
        "requests_per_platform": 43,
        "standard": {
            "reachable": 23,
            "hard_reachable": 7,
            "unreachable": 3,
        },
        "kilometer": {
            "reachable": 6,
            "hard_reachable": 2,
            "unreachable": 2,
        },
        "repeat_count": 5,
        "formal_call_count": 645,
    }
    expected_phase_contract = {
        "p01": {
            "name": "r3_parent_static_preflight",
            "provider_call_count": 0,
            "blind_request_count": 129,
            "truth_sidecar_count": 129,
        },
        "p02": {
            "name": "r3_cold_start_and_warmup",
            "cold_start_count": 3,
            "warmup_count": 30,
            "formal_sample": False,
        },
        "p03": {
            "name": "r3_worker_semantic_probe",
            "worker_one_count": 12,
            "worker_four_count": 12,
            "formal_sample": False,
            "requires": ["p01", "p02"],
        },
        "p04": {
            "name": "r3_formal_worker_four",
            "formal_call_count": 645,
            "formal_sample": True,
            "requires": ["p01", "p02", "p03"],
        },
    }
    expected_thresholds = {
        "midterm_mean": 2000.0,
        "midterm_p95": 2000.0,
        "absolute_max": 2000.0,
        "final_mean": 1000.0,
        "final_p95": 1000.0,
        "final_proportion_at_or_below": 0.95,
        "engineering_standard_p95": 250.0,
        "engineering_kilometer_p95": 750.0,
    }
    expected_schemas = {
        "provider_blind_request": R3_PROVIDER_BLIND_SCHEMA_VERSION,
        "truth_sidecar": "g2-truth-request-sidecar/v1",
        "provider_execution_request": (
            R3_PROVIDER_EXECUTION_REQUEST_SCHEMA_VERSION
        ),
        "resource_policy": R3_RESOURCE_POLICY_SCHEMA_VERSION,
        "execution_bundle": R3_EXECUTION_BUNDLE_SCHEMA_VERSION,
        "artifact_bound_approval": R3_APPROVAL_SCHEMA_VERSION,
        "formal_schedule": "xunce-mid-dual-g2-formal-schedule/v2",
    }
    execution = config["execution"]
    platform_stacks = config["platform_stacks"]
    if (
        config["request_contract"] != expected_request_contract
        or config["phase_contract"] != expected_phase_contract
        or config["thresholds_ms"] != expected_thresholds
        or config["schemas"] != expected_schemas
        or type(execution) is not dict
        or execution.get("formal_worker_count") != 4
        or execution.get("diagnostic_worker_count") != 1
        or execution.get("warmup_requests_per_platform") != 10
        or execution.get("cold_start_calls_per_platform") != 1
        or execution.get("formal_timing_contract")
        != "five-phase-sequential-ns/v1"
        or execution.get("formal_cache_contract")
        != "immutable-terrain-static-validation-only/v1"
        or execution.get("p03_selection_contract")
        != R3_P03_SELECTION_CONTRACT
        or type(execution.get("formal_environment_gate")) is not dict
        or type(platform_stacks) is not dict
        or set(platform_stacks) != set(G2_PLATFORMS)
        or any(
            type(platform_stacks[platform]) is not dict
            or platform_stacks[platform].get(
                "max_traversable_slope_deg"
            )
            != 30.0
            for platform in G2_PLATFORMS
        )
    ):
        _fail("g2_r3_config_invalid")

    unresolved = (
        any(value is None for value in formal_inputs.values())
        or any(
            value is None
            for key, value in producer_binding.items()
            if key
            not in {"schema_version", "provider_blind_schema_version"}
        )
        or any(value is None for value in activation_binding.values())
    )
    blocked_readiness = {
        "status": "blocked",
        "formal_evidence_eligible": False,
        "blockers": ["missing_final_g2_r3_activation_handoff"],
    }
    ready_readiness = {
        "status": "ready",
        "formal_evidence_eligible": True,
        "blockers": [],
    }
    if unresolved:
        if require_resolved:
            _fail("g2_r3_config_unresolved")
        if config["default_readiness"] != blocked_readiness:
            _fail("g2_r3_config_invalid")
    else:
        validate_r3_producer_binding(producer_binding)
        for key, value in formal_inputs.items():
            if key.endswith("_sha256"):
                _require_sha256(value, "g2_r3_config_invalid")
            elif type(value) is not str or not value.startswith("D:/"):
                _fail("g2_r3_config_invalid")
        for key, value in activation_binding.items():
            if key in {"candidate_id", "input_set_id"}:
                _require_nonempty(value, "g2_r3_config_invalid")
            else:
                _require_sha256(value, "g2_r3_config_invalid")
        if (
            activation_binding["consumer_activation_contract_sha256"]
            != r3_consumer_activation_contract_sha256(config)
            or config["default_readiness"] != ready_readiness
        ):
            _fail("g2_r3_config_invalid")
    return dict(config)


def validate_r3_final_candidate_metadata(
    candidate: Mapping[str, object],
) -> dict[str, object]:
    if (
        type(candidate) is not dict
        or set(candidate) != _R3_FINAL_CANDIDATE_KEYS
        or candidate.get("schema_version")
        != R3_FINAL_CANDIDATE_SCHEMA_VERSION
        or candidate.get("fixture_only") is not False
        or candidate.get("formal_evidence_eligible") is not False
        or candidate.get("formal_ineligibility_reason")
        != "awaiting_artifact_bound_approval"
        or type(candidate.get("candidate_id")) is not str
        or not candidate["candidate_id"]
        or type(candidate.get("input_set_id")) is not str
        or not candidate["input_set_id"]
        or candidate.get("producer_repeatability_passed") is not True
        or candidate.get("request_count") != 129
    ):
        _fail("g2_r3_final_candidate_ineligible")
    for field in _R3_FINAL_CANDIDATE_KEYS:
        if field.endswith("_sha256"):
            _require_sha256(
                candidate.get(field),
                "g2_r3_final_candidate_ineligible",
            )
    return dict(candidate)


def _r3_parse_canonical_jsonl_payload(
    payload: bytes,
    *,
    code: str,
) -> list[dict[str, object]]:
    if type(payload) is not bytes or not payload or not payload.endswith(b"\n"):
        _fail(code)
    rows: list[dict[str, object]] = []
    for raw_line in payload.splitlines(keepends=True):
        if not raw_line.endswith(b"\n") or raw_line == b"\n":
            _fail(code)
        try:
            value = json.loads(raw_line[:-1].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise G2InputContractError(code, type(exc).__name__) from exc
        if type(value) is not dict or raw_line != _canonical_json_bytes(value) + b"\n":
            _fail(code)
        rows.append(value)
    return rows


def _r3_parse_canonical_json_payload(
    payload: bytes,
    *,
    code: str,
) -> dict[str, object]:
    if type(payload) is not bytes or not payload:
        _fail(code)
    body = payload[:-1] if payload.endswith(b"\n") else payload
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G2InputContractError(code, type(exc).__name__) from exc
    if type(value) is not dict or body != _canonical_json_bytes(value):
        _fail(code)
    return value


def _r3_execution_signature(
    request: Mapping[str, object],
    snapshot: Mapping[str, object],
    policy: Mapping[str, object],
) -> str:
    actions = request.get("action_envelope", {}).get("actions")
    if not isinstance(actions, list) or not actions:
        _fail("g2_r3_execution_signature")
    action_shapes = sorted(
        sorted(
            key
            for key in action
            if not str(key).endswith("_id")
        )
        for action in actions
        if type(action) is dict
    )
    translation = snapshot.get("vertical_translation")
    core = {
        "schema_version": "xunce-mid-dual-g2-r3-execution-signature/v1",
        "platform": request["platform_kind"],
        "scale": request["scale"],
        "profile_or_parameter_record_sha256": request[
            "profile_or_parameter_record_sha256"
        ],
        "action_envelope_schema_version": request[
            "action_envelope"
        ].get("schema_version"),
        "action_count": len(actions),
        "action_shapes": action_shapes,
        "snapshot_schema_version": snapshot.get("schema_version"),
        "snapshot_shape_height_width": snapshot.get(
            "shape_height_width"
        ),
        "snapshot_resolution_mm": snapshot.get("resolution_mm"),
        "snapshot_source_kind": snapshot.get("source_kind"),
        "vertical_translation_semantic_kind": (
            translation.get("semantic_kind")
            if type(translation) is dict
            else None
        ),
        "resource_policy_sha256": policy["resource_policy_sha256"],
    }
    return _domain_hash(
        "xunce-mid-dual-g2-r3-execution-signature/v1",
        _canonical_json_bytes(core),
    )


def _r3_payload_index(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative in artifact_io.list_relative_files(root):
        if relative == "manifest.json":
            continue
        payload = artifact_io.read_bytes(root / PurePosixPath(relative))
        rows.append(
            {
                "relative_path": relative,
                "byte_length": len(payload),
                "sha256": _sha256(payload),
            }
        )
    return rows


def _r3_validate_source_closure_hash(
    closure: Mapping[str, object],
    *,
    field: str,
) -> str:
    if type(closure) is not dict:
        _fail("g2_r3_source_closure")
    return _require_sha256(
        closure.get(field),
        "g2_r3_source_closure",
    )


def prepare_r3_execution_data(
    *,
    output_root: str | Path,
    binding: Mapping[str, object],
    candidate: Mapping[str, object],
    blind_payload: bytes,
    sidecar_payload: bytes,
    snapshot_payloads: Mapping[str, bytes],
    hopper_parameter_record_payload: bytes,
    provider_source_closure: Mapping[str, object],
    consumer_source_closure: Mapping[str, object],
    requests: Sequence[Mapping[str, object]] | None = None,
    sidecars: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Prepare immutable R3 execution data without approval or manifest."""

    output = Path(output_root)
    if artifact_io.path_exists(output):
        _fail("g2_r3_execution_root_exists")
    producer_binding = validate_r3_producer_binding(binding)
    final_candidate = validate_r3_final_candidate_metadata(candidate)
    if (
        final_candidate["producer_binding_sha256"]
        != producer_binding["binding_sha256"]
        or final_candidate["hopper_parameter_record_sha256"]
        != producer_binding["hopper_parameter_record_sha256"]
        or _sha256(blind_payload)
        != final_candidate["provider_blind_requests_file_sha256"]
        or _sha256(sidecar_payload)
        != final_candidate["truth_request_sidecar_file_sha256"]
    ):
        _fail("g2_r3_candidate_payload_drift")

    blind_rows = _r3_parse_canonical_jsonl_payload(
        blind_payload,
        code="g2_r3_blind_payload",
    )
    truth_rows = _r3_parse_canonical_jsonl_payload(
        sidecar_payload,
        code="g2_r3_truth_sidecar_payload",
    )
    if (
        len(blind_rows) != 129
        or len(truth_rows) != 129
        or (
            requests is not None
            and [dict(row) for row in requests] != blind_rows
        )
        or (
            sidecars is not None
            and [dict(row) for row in sidecars] != truth_rows
        )
    ):
        _fail("g2_r3_candidate_payload_drift")
    validated_requests = [
        _validate_r3_exact_provider_blind_request(
            row,
            producer_binding,
        )
        for row in blind_rows
    ]
    parent_audit = validate_r3_parent_static_hop_resource_join(
        validated_requests,
        truth_rows,
        producer_binding=producer_binding,
    )

    record = _r3_parse_canonical_json_payload(
        hopper_parameter_record_payload,
        code="g2_r3_hopper_parameter_record",
    )
    record_sha256 = _sha256(_canonical_json_bytes(record))
    if record_sha256 != final_candidate["hopper_parameter_record_sha256"]:
        _fail(
            "G2I_BLOCKED_HOPPER_PARAMETER_RECORD_MISMATCH",
            "g2_hopper_parameter_record_mismatch",
        )

    if type(snapshot_payloads) is not dict:
        _fail("g2_r3_snapshot_payloads")
    snapshots_by_sha: dict[str, dict[str, object]] = {}
    for request in validated_requests:
        snapshot_sha256 = str(
            request["provider_local_snapshot_sha256"]
        )
        payload = snapshot_payloads.get(snapshot_sha256)
        if type(payload) is not bytes:
            _fail("g2_r3_snapshot_payloads")
        if _sha256(payload) != request[
            "provider_local_snapshot_payload_sha256"
        ]:
            _fail("g2_r3_snapshot_payload_drift")
        snapshot = _r3_parse_canonical_json_payload(
            payload,
            code="g2_r3_snapshot_payloads",
        )
        projected = project_r3_local_snapshot(
            snapshot,
            scale=str(request["scale"]),
        )
        if (
            snapshot.get("snapshot_sha256") != snapshot_sha256
            or projected["source_snapshot_sha256"] != snapshot_sha256
            or snapshot.get("platform_kind")
            != request["platform_kind"]
        ):
            _fail("g2_r3_snapshot_payload_drift")
        snapshots_by_sha[snapshot_sha256] = snapshot
    if (
        len(snapshots_by_sha) != 129
        or set(snapshots_by_sha) != set(snapshot_payloads)
    ):
        _fail("g2_r3_snapshot_payloads")

    execution_requests = [
        build_r3_provider_execution_request(
            request,
            producer_binding=producer_binding,
        )
        for request in validated_requests
    ]
    policies = [
        r3_provider_resource_policy(platform)
        for platform in G2_PLATFORMS
    ]
    resource_policy_root_sha256 = _domain_hash(
        "xunce-mid-dual-g2-r3-resource-policy-root/v1",
        _canonical_json_bytes(policies),
    )
    execution_request_root_sha256 = _domain_hash(
        "xunce-mid-dual-g2-r3-execution-request-root/v1",
        _canonical_json_bytes(execution_requests),
    )

    sidecar_by_sha = {
        str(row["provider_request_sha256"]): row for row in truth_rows
    }
    execution_by_sha = {
        str(row["provider_request_sha256"]): row
        for row in execution_requests
    }
    signatures: dict[
        tuple[str, str, str],
        dict[str, list[dict[str, object]]],
    ] = {}
    for request in validated_requests:
        sidecar = sidecar_by_sha[str(request["provider_request_sha256"])]
        truth = sidecar["truth_request"]
        difficulty = str(truth["difficulty_class"])
        if difficulty == "unreachable":
            continue
        probe_class = (
            "normal_reachable"
            if difficulty == "reachable"
            else "hard_reachable"
        )
        execution_request = execution_by_sha[
            str(request["provider_request_sha256"])
        ]
        signature = _r3_execution_signature(
            request,
            snapshots_by_sha[
                str(request["provider_local_snapshot_sha256"])
            ],
            execution_request["resource_policy"],
        )
        key = (
            str(request["platform_kind"]),
            str(request["scale"]),
            probe_class,
        )
        signatures.setdefault(key, {}).setdefault(signature, []).append(
            request
        )
    expected_probe_keys = {
        (platform, scale, probe_class)
        for platform in G2_PLATFORMS
        for scale in ("standard", "kilometer")
        for probe_class in ("normal_reachable", "hard_reachable")
    }
    if set(signatures) != expected_probe_keys or any(
        len(by_signature) != 1
        for by_signature in signatures.values()
    ):
        _fail("g2_r3_p03_signature_uncovered")
    probe_selection: list[dict[str, object]] = []
    for platform, scale, probe_class in sorted(expected_probe_keys):
        by_signature = signatures[(platform, scale, probe_class)]
        signature, candidates = next(iter(by_signature.items()))
        selected = min(
            candidates,
            key=lambda request: _domain_hash(
                R3_P03_SELECTION_CONTRACT,
                str(final_candidate["input_set_id"]).encode("utf-8"),
                platform.encode("ascii"),
                scale.encode("ascii"),
                probe_class.encode("ascii"),
                str(request["provider_request_sha256"]).encode("ascii"),
            ),
        )
        probe_selection.append(
            {
                "request_id": selected["provider_request_id"],
                "platform": platform,
                "scale": scale,
                "probe_class": probe_class,
                "execution_signature_sha256": signature,
            }
        )
    p03_probe_selection_sha256 = _domain_hash(
        R3_P03_SELECTION_CONTRACT,
        _canonical_json_bytes(probe_selection),
    )

    try:
        import run_xunce_mid_dual_g2_planning_time as r3_runner
    except ImportError as exc:
        raise G2InputContractError(
            "g2_r3_runner_import",
            type(exc).__name__,
        ) from exc
    formal_schedule = r3_runner.build_r3_formal_schedule(
        str(final_candidate["input_set_id"]),
        validated_requests,
        producer_schema_contract=producer_binding,
    )
    formal_schedule_sha256 = _require_sha256(
        formal_schedule.get("schedule_sha256"),
        "g2_r3_formal_schedule",
    )

    provider_closure_sha256 = _r3_validate_source_closure_hash(
        provider_source_closure,
        field="path_planner_runtime_source_closure_sha256",
    )
    consumer_closure_sha256 = _r3_validate_source_closure_hash(
        consumer_source_closure,
        field="consumer_source_closure_sha256",
    )
    activation_config = artifact_io.read_json(
        REPO_ROOT / "configs" / "xunce_mid_dual_g2_planning_time_v2.json"
    )
    validate_r3_activation_config(
        activation_config,
        require_resolved=False,
    )
    consumer_activation_contract_sha256 = (
        r3_consumer_activation_contract_sha256(activation_config)
    )

    artifact_io.write_bytes(
        output / "producer-binding.json",
        _canonical_json_bytes(producer_binding) + b"\n",
    )
    artifact_io.write_bytes(
        output / "provider-blind-requests.jsonl",
        blind_payload,
    )
    artifact_io.write_bytes(
        output / "truth" / "request-sidecar.jsonl",
        sidecar_payload,
    )
    artifact_io.write_bytes(
        output / "provider-execution-requests.jsonl",
        b"".join(
            _canonical_json_bytes(row) + b"\n"
            for row in execution_requests
        ),
    )
    crosswalk = list(parent_audit["execution_crosswalk"])
    artifact_io.write_bytes(
        output / "truth-free-execution-crosswalk.jsonl",
        b"".join(
            _canonical_json_bytes(row) + b"\n" for row in crosswalk
        ),
    )
    for snapshot_sha256, payload in snapshot_payloads.items():
        artifact_io.write_bytes(
            output
            / "terrain"
            / "provider-local"
            / f"{snapshot_sha256}.json",
            payload,
        )
    artifact_io.write_bytes(
        output / "hopper-parameter-record.json",
        _canonical_json_bytes(record) + b"\n",
    )
    artifact_io.write_bytes(
        output / "p03-probe-selection.json",
        _canonical_json_bytes(
            {
                "schema_version": R3_P03_SELECTION_CONTRACT,
                "rows": probe_selection,
                "selection_sha256": p03_probe_selection_sha256,
            }
        )
        + b"\n",
    )
    artifact_io.write_bytes(
        output / "parent-static-audit.json",
        _canonical_json_bytes(parent_audit) + b"\n",
    )
    artifact_io.write_bytes(
        output / "source" / "provider-runtime-source-closure.json",
        _canonical_json_bytes(provider_source_closure) + b"\n",
    )
    artifact_io.write_bytes(
        output / "source" / "consumer-source-closure.json",
        _canonical_json_bytes(consumer_source_closure) + b"\n",
    )
    input_audit = {
        "schema_version": "xunce-mid-dual-g2-r3-input-audit/v1",
        "status": "awaiting_artifact_bound_approval",
        "formal_evidence_eligible": False,
        "provider_called": False,
        "candidate_id": final_candidate["candidate_id"],
        "input_set_id": final_candidate["input_set_id"],
        "producer_binding_sha256": producer_binding["binding_sha256"],
        "request_count": 129,
        "truth_sidecar_count": 129,
        "hopper_request_count": sum(
            row["platform_kind"] == "hopper"
            for row in validated_requests
        ),
        "resource_policy_root_sha256": resource_policy_root_sha256,
        "execution_request_root_sha256": (
            execution_request_root_sha256
        ),
        "p03_probe_selection_sha256": p03_probe_selection_sha256,
        "formal_schedule_sha256": formal_schedule_sha256,
    }
    artifact_io.write_bytes(
        output / "input-audit.json",
        _canonical_json_bytes(input_audit) + b"\n",
    )
    payload_index = _r3_payload_index(output)
    execution_data_root_sha256 = _domain_hash(
        "xunce-mid-dual-g2-r3-execution-data-root/v1",
        _canonical_json_bytes(payload_index),
    )
    return {
        "schema_version": (
            "xunce-mid-dual-g2-r3-prepared-execution-data/v1"
        ),
        "status": "awaiting_artifact_bound_approval",
        "formal_evidence_eligible": False,
        "provider_called": False,
        "output_root": output.as_posix(),
        **{
            key: final_candidate[key]
            for key in (
                "candidate_id",
                "input_set_id",
                "producer_manifest_file_sha256",
                "manifest_core_sha256",
                "producer_payload_root_sha256",
                "producer_repeatability_audit_sha256",
                "provider_blind_requests_file_sha256",
                "truth_request_sidecar_file_sha256",
                "hopper_parameter_record_sha256",
            )
        },
        "producer_binding_sha256": producer_binding["binding_sha256"],
        "provider_runtime_source_closure_sha256": (
            provider_closure_sha256
        ),
        "consumer_source_closure_sha256": consumer_closure_sha256,
        "consumer_activation_contract_sha256": (
            consumer_activation_contract_sha256
        ),
        "resource_policy_root_sha256": resource_policy_root_sha256,
        "execution_request_root_sha256": execution_request_root_sha256,
        "p03_probe_selection_sha256": p03_probe_selection_sha256,
        "formal_schedule_sha256": formal_schedule_sha256,
        "execution_data_root_sha256": execution_data_root_sha256,
        "request_count": 129,
        "truth_sidecar_count": 129,
        "hopper_request_count": 43,
        "resource_policy_count": 3,
        "formal_call_count": 645,
        "payload_index": payload_index,
    }


def build_r3_approval_target(
    prepared: Mapping[str, object],
) -> dict[str, object]:
    if (
        type(prepared) is not dict
        or prepared.get("schema_version")
        != "xunce-mid-dual-g2-r3-prepared-execution-data/v1"
        or prepared.get("status") != "awaiting_artifact_bound_approval"
        or prepared.get("provider_called") is not False
        or prepared.get("request_count") != 129
        or prepared.get("formal_call_count") != 645
    ):
        _fail("g2_r3_prepared_execution_data")
    field_map = {
        "candidate_id": "candidate_id",
        "input_set_id": "input_set_id",
        "producer_binding_sha256": "producer_binding_sha256",
        "producer_manifest_file_sha256": (
            "producer_manifest_file_sha256"
        ),
        "manifest_core_sha256": "manifest_core_sha256",
        "producer_payload_root_sha256": "producer_payload_root_sha256",
        "producer_repeatability_audit_sha256": (
            "producer_repeatability_audit_sha256"
        ),
        "provider_blind_requests_file_sha256": (
            "provider_blind_requests_file_sha256"
        ),
        "truth_request_sidecar_file_sha256": (
            "truth_request_sidecar_file_sha256"
        ),
        "hopper_parameter_record_sha256": (
            "hopper_parameter_record_sha256"
        ),
        "provider_runtime_source_closure_sha256": (
            "provider_runtime_source_closure_sha256"
        ),
        "consumer_source_closure_sha256": (
            "consumer_source_closure_sha256"
        ),
        "consumer_activation_contract_sha256": (
            "consumer_activation_contract_sha256"
        ),
        "resource_policy_root_sha256": "resource_policy_root_sha256",
        "execution_request_root_sha256": (
            "execution_request_root_sha256"
        ),
        "p03_probe_selection_sha256": "p03_probe_selection_sha256",
        "formal_schedule_sha256": "formal_schedule_sha256",
        "execution_data_root_sha256": "execution_data_root_sha256",
    }
    core: dict[str, object] = {
        "schema_version": R3_APPROVAL_TARGET_SCHEMA_VERSION,
        **{
            target_field: prepared[source_field]
            for target_field, source_field in field_map.items()
        },
        "request_count": 129,
        "formal_call_count": 645,
    }
    for field in _R3_APPROVAL_TARGET_KEYS:
        if field.endswith("_sha256"):
            _require_sha256(core.get(field), "g2_r3_approval_target")
    return {
        **core,
        "approval_target_sha256": _domain_hash(
            R3_APPROVAL_TARGET_SCHEMA_VERSION,
            _canonical_json_bytes(core),
        ),
    }


def validate_r3_artifact_bound_approval_v2(
    approval: Mapping[str, object],
    target: Mapping[str, object],
) -> dict[str, object]:
    if (
        type(approval) is not dict
        or approval.get("schema_version") != R3_APPROVAL_SCHEMA_VERSION
    ):
        _fail("g2_r3_legacy_approval_ineligible")
    required = {
        "schema_version",
        "approval_id",
        "decision",
        "formal_evidence_eligible",
        "blockers",
        "approval_target",
        "approval_target_sha256",
    }
    target_core = {
        key: value
        for key, value in target.items()
        if key != "approval_target_sha256"
    }
    expected_target_sha256 = _domain_hash(
        R3_APPROVAL_TARGET_SCHEMA_VERSION,
        _canonical_json_bytes(target_core),
    )
    if (
        set(approval) != required
        or type(approval.get("approval_id")) is not str
        or not approval["approval_id"]
        or approval.get("decision") != "approved"
        or approval.get("formal_evidence_eligible") is not True
        or approval.get("blockers") != []
        or set(target_core) != _R3_APPROVAL_TARGET_KEYS
        or target.get("approval_target_sha256")
        != expected_target_sha256
        or approval.get("approval_target") != target_core
        or approval.get("approval_target_sha256")
        != expected_target_sha256
    ):
        _fail("g2_r3_approval_target_drift")
    return dict(approval)


def _r3_revalidate_prepared_payloads(
    output: Path,
    prepared: Mapping[str, object],
) -> list[dict[str, object]]:
    expected_index = prepared.get("payload_index")
    if type(expected_index) is not list:
        _fail("g2_r3_execution_payload_drift")
    actual_index = _r3_payload_index(output)
    if (
        actual_index != expected_index
        or prepared.get("execution_data_root_sha256")
        != _domain_hash(
            "xunce-mid-dual-g2-r3-execution-data-root/v1",
            _canonical_json_bytes(actual_index),
        )
    ):
        _fail("g2_r3_execution_payload_drift")
    return actual_index


def seal_r3_execution_bundle(
    *,
    output_root: str | Path,
    prepared: Mapping[str, object],
    approval_path: str | Path,
) -> dict[str, object]:
    output = Path(output_root)
    if (
        output.as_posix() != prepared.get("output_root")
        or artifact_io.path_is_file(output / "manifest.json")
        or not artifact_io.path_is_file(approval_path)
    ):
        _fail("g2_r3_execution_seal")
    payload_index = _r3_revalidate_prepared_payloads(output, prepared)
    approval_bytes = artifact_io.read_bytes(approval_path)
    approval = _r3_parse_canonical_json_payload(
        approval_bytes,
        code="g2_r3_approval_artifact",
    )
    target = build_r3_approval_target(prepared)
    validate_r3_artifact_bound_approval_v2(approval, target)
    manifest = {
        "schema_version": R3_EXECUTION_BUNDLE_SCHEMA_VERSION,
        "publication_order": "data-first-manifest-last",
        "formal_evidence_eligible": True,
        "blockers": [],
        **{
            key: target[key]
            for key in (
                "candidate_id",
                "input_set_id",
                "producer_binding_sha256",
                "producer_manifest_file_sha256",
                "manifest_core_sha256",
                "producer_payload_root_sha256",
                "producer_repeatability_audit_sha256",
                "provider_blind_requests_file_sha256",
                "truth_request_sidecar_file_sha256",
                "hopper_parameter_record_sha256",
                "provider_runtime_source_closure_sha256",
                "consumer_source_closure_sha256",
                "consumer_activation_contract_sha256",
                "resource_policy_root_sha256",
                "execution_request_root_sha256",
                "p03_probe_selection_sha256",
                "formal_schedule_sha256",
                "execution_data_root_sha256",
            )
        },
        "approval_schema_version": R3_APPROVAL_SCHEMA_VERSION,
        "approval_target_schema_version": (
            R3_APPROVAL_TARGET_SCHEMA_VERSION
        ),
        "approval_artifact_path": Path(approval_path).as_posix(),
        "approval_artifact_sha256": _sha256(approval_bytes),
        "approval_target_sha256": target["approval_target_sha256"],
        "provider_blind_payload_sha256": prepared[
            "provider_blind_requests_file_sha256"
        ],
        "truth_sidecar_payload_sha256": prepared[
            "truth_request_sidecar_file_sha256"
        ],
        "request_count": 129,
        "crosswalk_count": 129,
        "formal_call_count": 645,
        "payload_index": payload_index,
        "payload_root_sha256": prepared["execution_data_root_sha256"],
    }
    artifact_io.write_bytes(
        output / "manifest.json",
        _canonical_json_bytes(manifest) + b"\n",
    )
    return manifest


def validate_r3_sealed_execution_bundle(
    output_root: str | Path,
    *,
    approval_path: str | Path,
) -> dict[str, object]:
    output = Path(output_root)
    manifest_path = output / "manifest.json"
    if not artifact_io.path_is_file(manifest_path):
        _fail("g2_r3_execution_manifest_missing")
    manifest = _r3_parse_canonical_json_payload(
        artifact_io.read_bytes(manifest_path),
        code="g2_r3_execution_manifest",
    )
    if (
        manifest.get("schema_version")
        != R3_EXECUTION_BUNDLE_SCHEMA_VERSION
        or manifest.get("publication_order")
        != "data-first-manifest-last"
        or manifest.get("formal_evidence_eligible") is not True
        or manifest.get("blockers") != []
        or manifest.get("request_count") != 129
        or manifest.get("crosswalk_count") != 129
        or manifest.get("formal_call_count") != 645
        or manifest.get("approval_schema_version")
        != R3_APPROVAL_SCHEMA_VERSION
        or manifest.get("approval_target_schema_version")
        != R3_APPROVAL_TARGET_SCHEMA_VERSION
    ):
        _fail("g2_r3_execution_manifest")
    payload_index = _r3_payload_index(output)
    expected_index = manifest.get("payload_index")
    if (
        type(expected_index) is not list
        or payload_index != expected_index
        or manifest.get("payload_root_sha256")
        != _domain_hash(
            "xunce-mid-dual-g2-r3-execution-data-root/v1",
            _canonical_json_bytes(payload_index),
        )
    ):
        _fail("g2_r3_execution_payload_drift")
    if not artifact_io.path_is_file(approval_path):
        _fail("g2_r3_approval_artifact")
    approval_bytes = artifact_io.read_bytes(approval_path)
    if (
        Path(approval_path).as_posix()
        != manifest.get("approval_artifact_path")
        or _sha256(approval_bytes)
        != manifest.get("approval_artifact_sha256")
    ):
        _fail("g2_r3_approval_artifact")
    approval = _r3_parse_canonical_json_payload(
        approval_bytes,
        code="g2_r3_approval_artifact",
    )
    target_core = {
        key: (
            manifest["approval_target_schema_version"]
            if key == "schema_version"
            else manifest[key]
        )
        for key in _R3_APPROVAL_TARGET_KEYS
    }
    target = {
        **target_core,
        "approval_target_sha256": manifest["approval_target_sha256"],
    }
    validate_r3_artifact_bound_approval_v2(approval, target)
    return {
        "manifest": manifest,
        "manifest_file_sha256": _sha256(
            artifact_io.read_bytes(manifest_path)
        ),
        "approval": approval,
        "payload_index": payload_index,
    }


def decode_r3_provider_execution_request(
    execution_request: Mapping[str, object],
    snapshot_payload: bytes,
    *,
    producer_binding: Mapping[str, object],
) -> dict[str, object]:
    """Decode one resource-bound R3 request without macro-grid resampling."""

    binding = validate_r3_producer_binding(producer_binding)
    if (
        type(execution_request) is not dict
        or set(execution_request)
        != {
            "schema_version",
            "producer_binding_sha256",
            "provider_blind_request",
            "provider_request_sha256",
            "resource_policy",
            "resource_policy_sha256",
            "execution_request_sha256",
        }
        or execution_request.get("schema_version")
        != R3_PROVIDER_EXECUTION_REQUEST_SCHEMA_VERSION
        or execution_request.get("producer_binding_sha256")
        != binding["binding_sha256"]
    ):
        _fail("g2_r3_execution_request")
    request = _validate_r3_exact_provider_blind_request(
        execution_request["provider_blind_request"],
        binding,
    )
    policy = validate_r3_provider_resource_policy(
        execution_request["resource_policy"]
    )
    core = {
        key: value
        for key, value in execution_request.items()
        if key != "execution_request_sha256"
    }
    if (
        execution_request.get("provider_request_sha256")
        != request["provider_request_sha256"]
        or execution_request.get("resource_policy_sha256")
        != policy["resource_policy_sha256"]
        or execution_request.get("execution_request_sha256")
        != _domain_hash(
            R3_PROVIDER_EXECUTION_REQUEST_SCHEMA_VERSION,
            _canonical_json_bytes(core),
        )
    ):
        _fail("g2_r3_execution_request")
    if (
        type(snapshot_payload) is not bytes
        or _sha256(snapshot_payload)
        != request["provider_local_snapshot_payload_sha256"]
    ):
        _fail("g2_r3_snapshot_payload_drift")
    snapshot = _r3_parse_canonical_json_payload(
        snapshot_payload,
        code="g2_r3_snapshot_payloads",
    )
    projected = project_r3_local_snapshot(
        snapshot,
        scale=str(request["scale"]),
    )
    if (
        projected["source_snapshot_sha256"]
        != request["provider_local_snapshot_sha256"]
        or projected["platform_kind"] != request["platform_kind"]
    ):
        _fail("g2_r3_snapshot_payload_drift")
    start = project_r3_canonical_pose(
        request["start"]["pose_binary64_m_rad"]
    )
    goal = project_r3_canonical_pose(
        request["goal"]["pose_binary64_m_rad"]
    )

    from path_planner.v2.contracts import (
        AcceleratorPolicyV2,
        ObjectiveProfileV2,
        PlanningRequestV2,
        PoseStateV2,
        ResourceBudgetV2,
    )
    from path_planner.v2.terrain import (
        FineGridGeometryV2,
        TerrainProvenanceV2,
        TerrainSnapshotV2,
    )

    terrain = TerrainSnapshotV2(
        geometry=FineGridGeometryV2(
            width=20,
            height=20,
            resolution_m=0.5,
        ),
        elevation_m=np.ascontiguousarray(
            projected["elevation_m"],
            dtype="<f8",
        ),
        slope_deg=np.ascontiguousarray(
            projected["slope_deg"],
            dtype="<f8",
        ),
        traversable_mask=np.ascontiguousarray(
            projected["traversable"],
            dtype=bool,
        ),
        hard_obstacle_mask=np.ascontiguousarray(
            projected["hard_obstacle"],
            dtype=bool,
        ),
        observed_mask=np.ascontiguousarray(
            projected["known"],
            dtype=bool,
        ),
        confidence=np.ascontiguousarray(
            projected["confidence_ppm"].astype("<f8") / 1_000_000.0,
            dtype="<f8",
        ),
        provenance=TerrainProvenanceV2(
            source_kind=str(snapshot["source_kind"]),
            source_id=(
                "g2-r3-provider-local-"
                f"{request['provider_local_snapshot_sha256'][:24]}"
            ),
            source_hash=str(
                request["provider_local_snapshot_sha256"]
            ),
            physical_obstacle_cells_written=False,
            details=(
                ("frame_id", str(snapshot["frame_id"])),
                ("resolution_mm", 500),
            ),
        ),
    )
    platform = str(request["platform_kind"])
    planning_request = PlanningRequestV2(
        request_id=str(request["provider_request_id"]),
        platform_profile_id=str(PLATFORM_STACKS[platform]["profile_id"]),
        start_state=PoseStateV2(**start["pose"]),
        goal_state=PoseStateV2(**goal["pose"]),
        terrain_snapshot=terrain,
        objective_profile=ObjectiveProfileV2(
            distance_weight=1.0,
            energy_weight=0.0,
            risk_weight=0.0,
            time_weight=0.0,
        ),
        resource_budget=ResourceBudgetV2(
            max_expanded_states=int(policy["max_expanded_states"]),
            max_memory_bytes=int(policy["max_memory_bytes"]),
            max_route_states=int(policy["max_route_states"]),
        ),
        timeout_s=2.0,
        accelerator_policy=AcceleratorPolicyV2.DISABLED,
        determinism_seed=(
            int(str(request["provider_request_sha256"])[:16], 16)
            % (2**31)
        ),
    )
    return {
        "planning_request": planning_request,
        "platform": platform,
        "scale": request["scale"],
        "provider_request_id": request["provider_request_id"],
        "provider_request_sha256": request["provider_request_sha256"],
        "execution_request_sha256": execution_request[
            "execution_request_sha256"
        ],
        "resource_policy": policy,
        "resource_policy_sha256": policy["resource_policy_sha256"],
        "projection_sha256": projected["projection_sha256"],
        "provider_local_snapshot_sha256": request[
            "provider_local_snapshot_sha256"
        ],
        "terrain_geometry_sha256": request["terrain_geometry_sha256"],
        "start_pose": start,
        "goal_pose": goal,
        "execution_signature_sha256": _r3_execution_signature(
            request,
            snapshot,
            policy,
        ),
    }


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
    "R3_PRODUCER_BINDING_SCHEMA_VERSION",
    "approval_artifact_path",
    "audit_ppo_targets",
    "audit_primitive_labels",
    "audit_request_matrix",
    "audit_small_map_optima",
    "audit_source_separation",
    "audit_truth_bundle",
    "build_approved_platform_execution_stack",
    "build_execution_crosswalk",
    "build_r3_provider_execution_request",
    "capture_r3_consumer_source_closure",
    "decode_provider_execution_request",
    "materialize_execution_bundle",
    "preflight",
    "resolve_hopper_formal_eligibility",
    "validate_r3_hopper_execution_binding",
    "validate_r3_parent_static_hop_resource_join",
    "validate_r3_producer_binding",
    "validate_r3_provider_blind_request",
    "validate_r3_provider_resource_policy",
    "validate_primitive_label_identity",
]
