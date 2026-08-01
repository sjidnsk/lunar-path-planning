"""Independently recalculate the reduced midterm dual-gate from raw rows.

The three source summaries are never used as metric inputs.  Each source
manifest is verified first; its manifest-bound config then provides the
expected scale/input/code lineage for every raw row.  Stored summaries are
read only after recalculation and act as fail-closed consistency checks.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import statistics
import struct
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import xunce_artifact_io as artifact_io
from xunce_artifact_paths import (
    MID_DUAL_CONFIG,
    MID_DUAL_MANIFEST,
    MID_DUAL_RESULTS,
    MID_DUAL_SUMMARY,
    artifact_path,
)
from xunce_mid_dual_artifacts import MidDualRunStore
from xunce_mid_dual_contracts import (
    ClosedLoopStepRow,
    CoverageEpisodeRow,
    FINAL_COVERAGE_THRESHOLD,
    FINAL_TIME_MS,
    G1_EPISODES_PER_FORMAL_SPLIT,
    G2_FORMAL_CALLS,
    G2_KILOMETER_REQUESTS,
    G2_PLATFORMS,
    G2_REPEATS,
    G2_STANDARD_REQUESTS,
    InterfaceReplayRow,
    MID_COVERAGE_THRESHOLD,
    MID_TIME_MS,
    PlanningCallRow,
    SCALE_PROFILE,
    evaluate_formal_g2,
    evaluate_g1_split,
    nearest_rank,
)


AGGREGATE_SCHEMA_VERSION = "xunce-mid-dual-independent-aggregate/v1"
AGGREGATE_CONFIG_SCHEMA_VERSION = "xunce-mid-dual-aggregate-config/v1"
G1_NATIVE_EVIDENCE_KIND = "native_completed_run_v1"
G1_ASSESSMENT_EVIDENCE_KIND = "existing_run_assessment_v1"
_SOURCE_GATES = ("g1", "g2", "g3")
_SHA256_LENGTH = 64
_UPDATE80_CHECKPOINT_SHA256 = (
    "35e04c86f9f973af028fb08f0175d42ab45378d09e1f2b96ee6aad6e4c12b5b5"
)
_UPDATE80_POLICY_STATE_SHA256 = (
    "3123e6adde99be41e3bd5cc2f3843068e5416892395926d759c2c6a243ffd381"
)
_DENOMINATOR_SOURCE = "reachable_observable_free_highres_cells/v1"
_DENOMINATOR_ALGORITHM = "exact_reachable_safe_pose_range_los/v1"
_FORMAL_AGGREGATE_BASE = "D:/xunce/out/mid_dual/aggregate"
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")
_TIMING_CONTRACT_ID = "five-phase-sequential-ns/v1"
_TIMING_COMPONENT_FIELDS = (
    "input_validation_ns",
    "platform_instantiation_ns",
    "search_ns",
    "complete_route_validation_ns",
    "result_assembly_ns",
)
_G2_ACTIVE_PLATFORMS = ("wheel", "legged", "hopper")
_G2_PLATFORM_INVARIANTS = {
    platform_name: {"max_traversable_slope_deg": 30.0}
    for platform_name in _G2_ACTIVE_PLATFORMS
}
_G2_RUNTIME_SOURCE_SCHEMA_VERSION = (
    "xunce-mid-dual-path-planner-runtime-source-closure/v1"
)
_G2_FORMAL_ENVIRONMENT_POLICY_SCHEMA_VERSION = (
    "xunce-mid-dual-g2-formal-environment-policy/v1"
)
_G2_FORMAL_ENVIRONMENT_OBSERVATION_SCHEMA_VERSION = (
    "xunce-mid-dual-g2-formal-environment-observation/v1"
)
_G2_FORMAL_ENVIRONMENT_AUDIT_SCHEMA_VERSION = (
    "xunce-mid-dual-g2-formal-environment-audit/v1"
)
_G2_FORMAL_LEASE_SCHEMA_VERSION = "xunce-mid-dual-g2-formal-lease/v1"
_G2_FORMAL_ENVIRONMENT_POLICY = {
    "schema_version": _G2_FORMAL_ENVIRONMENT_POLICY_SCHEMA_VERSION,
    "lease_path": "D:/xunce/out/mid_dual/g2/.formal-exclusive-lease.json",
    "allowed_power_scheme_guids": [
        "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
    ],
    "required_thread_variables": {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    },
    "sample_window_seconds": 2.0,
    "limits": {
        "cpu_percent_max": 20.0,
        "memory_percent_max": 85.0,
        "memory_available_bytes_min": 4_294_967_296,
        "disk_busy_percent_max": 20.0,
        "disk_free_bytes_min": 10_737_418_240,
    },
    "competing_process_patterns": [
        "run_xunce_mid_dual_g1_coverage.py",
        "run_xunce_mid_dual_g2_planning_time.py",
        "run_xunce_mid_dual_g3_closed_loop.py",
        "run_ppo_stage6_standard.py",
        "pytest",
        "training",
        "formal",
    ],
    "exclude_current_process_tree": True,
}

_G1_ROW_KEYS = frozenset(
    {
        "row_kind",
        "schema_version",
        "gate_id",
        "runner_id",
        "phase_id",
        "phase_name",
        "scale_profile",
        "run_id",
        "split",
        "episode_id",
        "episode_index",
        "scenario_id",
        "lane_id",
        "source_sha256",
        "config_sha256",
        "input_sha256",
        "code_sha256",
        "scenario_manifest_sha256",
        "checkpoint_sha256",
        "policy_state_sha256",
        "denominator_source",
        "denominator_algorithm",
        "denominator_sha256",
        "denominator_cell_count",
        "initial_covered_cell_count",
        "final_covered_cell_count",
        "coverage",
        "elapsed_ms",
        "steps_executed",
        "termination_reason",
        "safety_violation_count",
        "masked_action_count",
    }
)
_G2_ROW_KEYS = frozenset(
    {
        "row_kind",
        "schema_version",
        "scale_profile",
        "run_id",
        "episode_id",
        "request_id",
        "call_id",
        "platform",
        "scale",
        "request_class",
        "outcome_kind",
        "source_sha256",
        "config_sha256",
        "input_sha256",
        "code_sha256",
        "request_sha256",
        "truth_sha256",
        "provider_sha256",
        "provider_source_bytes_sha256",
        "oracle_sha256",
        "oracle_source_bytes_sha256",
        "provider_result_sha256",
        "oracle_result_sha256",
        *_TIMING_COMPONENT_FIELDS,
        "total_ns",
        "elapsed_ms",
        "input_validation_ms",
        "platform_instantiation_ms",
        "search_ms",
        "complete_route_validation_ms",
        "result_assembly_ms",
        "provider_success",
        "route_l2_valid",
        "semantic_digest",
        "timing_contract_id",
        "formal_sample",
        "repeat_index",
    }
)
_G3_WHEEL_ROW_KEYS = frozenset(
    {
        "row_kind",
        "schema_version",
        "scale_profile",
        "run_id",
        "split",
        "episode_index",
        "episode_id",
        "scenario_id",
        "lane_id",
        "step_id",
        "step_index",
        "is_terminal",
        "termination_reason",
        "decision_sha256",
        "candidate_id",
        "candidate_sha256",
        "request_candidate_sha256",
        "selected_candidate_cell_xy",
        "planned_path_cells",
        "planner_path_length_m",
        "route_endpoint_cell_xy",
        "feedback_pose_cell_xy",
        "selected_theta",
        "route_endpoint_theta",
        "feedback_theta",
        "pre_snapshot_sha256",
        "request_id",
        "request_sha256",
        "route_request_id",
        "route_request_sha256",
        "route_result_sha256",
        "feedback_request_id",
        "feedback_route_result_sha256",
        "feedback_sha256",
        "post_snapshot_parent_sha256",
        "post_snapshot_sha256",
        "coverage",
        "paired_g1_coverage",
        "safety_violation_count",
        "masked_action_count",
        "candidate_route_mismatch_count",
        "planner_success",
        "planner_failure_reason",
        *_TIMING_COMPONENT_FIELDS,
        "total_ns",
        "execution_class",
        "formal_sample",
        "timing_contract_id",
        "wheel_platform",
        "wheel_profile",
        "wheel_capability_revision",
        "checkpoint_sha256",
        "policy_state_sha256",
        "config_sha256",
        "input_sha256",
        "code_sha256",
        "source_sha256",
        "frozen_manifest_sha256",
        "g1_source_manifest_sha256",
        "g2_source_manifest_sha256",
        "decision_record_id",
        "observation_record_id",
        "post_observation_record_sha256",
    }
)
_G3_INTERFACE_ROW_KEYS = frozenset(
    {
        "row_kind",
        "schema_version",
        "scale_profile",
        "run_id",
        "replay_id",
        "platform",
        "request_id",
        "request_sha256",
        "g2_reference_call_ids",
        "g2_semantic_digest",
        "replay_semantic_digest",
        "timing_contract_id",
        "formal_input_eligible",
        *_TIMING_COMPONENT_FIELDS,
        "total_ns",
        "execution_class",
        "formal_sample",
        "config_sha256",
        "input_sha256",
        "code_sha256",
        "source_sha256",
        "g1_source_manifest_sha256",
        "g2_source_manifest_sha256",
        "truth_request_sha256",
        "provider_request_sha256",
        "provider_result_sha256",
        "approval_sha256",
        "cohort_sha256",
        "provider_identity_sha256",
        "oracle_identity_sha256",
        "hopper_resolution_sha256",
    }
)

_G3_UPSTREAM_BINDING_KEYS = frozenset(
    {
        "g1_source_manifest_sha256",
        "g2_source_manifest_sha256",
        "freeze_manifest_sha256",
        "g1_config_sha256",
        "g1_input_sha256",
        "g1_code_sha256",
        "g2_config_sha256",
        "g2_input_sha256",
        "g2_code_sha256",
        "g2_input_set_id",
        "g2_approval_sha256",
        "g2_cohort_sha256",
        "g2_provider_identity_sha256",
        "g2_oracle_identity_sha256",
        "g2_hopper_resolution_sha256",
        "active_platforms",
        "platform_invariants",
        "g1_evidence_binding",
    }
)


class AggregateBlocked(ValueError):
    """A stable evidence blocker, distinct from a measured gate failure."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(char in "0123456789abcdef" for char in value)
    )


def _exact_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise AggregateBlocked(f"{field_name}_invalid")
    return value


def _exact_nonnegative_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise AggregateBlocked(f"{field_name}_invalid")
    return value


def _finite_number(value: object, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise AggregateBlocked(f"{field_name}_nonfinite_or_invalid")
    return float(value)


def _finite_nonnegative(value: object, field_name: str) -> float:
    number = _finite_number(value, field_name)
    if number < 0.0:
        raise AggregateBlocked(f"{field_name}_negative")
    return number


def _validate_g1_evidence_binding(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AggregateBlocked("g3_g1_evidence_binding_invalid")
    copied = dict(value)
    kind = copied.get("g1_evidence_kind")
    if kind == G1_NATIVE_EVIDENCE_KIND:
        if set(copied) != {
            "g1_evidence_kind",
            "g1_source_manifest_sha256",
        } or not _is_sha256(copied.get("g1_source_manifest_sha256")):
            raise AggregateBlocked("g3_g1_evidence_binding_invalid")
        return copied
    hash_fields = {
        "g1_source_manifest_sha256",
        "assessment_envelope_sha256",
        "assessment_manifest_sha256",
        "phase_snapshot_sha256",
        "stop_evidence_sha256",
        "review_manifest_sha256",
        "review_result_sha256",
    }
    expected = {
        "g1_evidence_kind",
        *hash_fields,
        "assessment_use_status",
        "strict_prestart_lineage_status",
        "lineage_limitation_acknowledged",
        "numerical_gate_status",
        "numerical_result_status",
    }
    if (
        kind != G1_ASSESSMENT_EVIDENCE_KIND
        or set(copied) != expected
        or any(not _is_sha256(copied.get(field)) for field in hash_fields)
        or copied["g1_source_manifest_sha256"]
        != copied["assessment_manifest_sha256"]
        or copied["assessment_use_status"] != "authorized_existing_run"
        or copied["strict_prestart_lineage_status"] != "not_satisfied"
        or copied["lineage_limitation_acknowledged"] is not True
        or copied["numerical_gate_status"] != "recomputed_from_raw_rows"
        or copied["numerical_result_status"] not in {"passed", "failed"}
    ):
        raise AggregateBlocked("g3_g1_evidence_binding_invalid")
    return copied


def _g2_elapsed_ms_from_raw_ns(row: Mapping[str, Any]) -> float:
    """Validate Task8's five raw nanosecond phases and derived projections."""
    components = [
        _exact_nonnegative_int(row.get(field), f"g2_{field}")
        for field in _TIMING_COMPONENT_FIELDS
    ]
    total_ns = _exact_nonnegative_int(row.get("total_ns"), "g2_total_ns")
    if total_ns != sum(components):
        raise AggregateBlocked("g2_timing_component_sum_mismatch")
    expected_ms = {
        "input_validation_ms": components[0] / 1_000_000.0,
        "platform_instantiation_ms": components[1] / 1_000_000.0,
        "search_ms": components[2] / 1_000_000.0,
        "complete_route_validation_ms": components[3] / 1_000_000.0,
        "result_assembly_ms": components[4] / 1_000_000.0,
        "elapsed_ms": total_ns / 1_000_000.0,
    }
    for field, expected in expected_ms.items():
        actual = _finite_nonnegative(row.get(field), f"g2_{field}")
        if actual != expected:
            raise AggregateBlocked("g2_timing_ms_projection_mismatch")
    return expected_ms["elapsed_ms"]


def _g2_expected_semantic_digest(
    request_sha256: object,
    provider_success: object,
    route_l2_valid: object,
) -> str:
    request_hash = _nonempty_string(request_sha256, "g2_semantic_request_hash")
    if not _is_sha256(request_hash):
        raise AggregateBlocked("g2_semantic_request_hash")
    success = _exact_bool(provider_success, "g2_semantic_success")
    valid = _exact_bool(route_l2_valid, "g2_semantic_l2")
    payload = json.dumps(
        {
            "request_sha256": request_hash,
            "provider_success": success,
            "route_l2_valid": valid,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    parts = (b"xunce-mid-dual-g2-provider-semantics/v1", payload)
    framed = b"".join(
        struct.pack(">Q", len(part)) + part for part in parts
    )
    return _sha256_bytes(framed)


def _exact_positive_int(value: object, field_name: str) -> int:
    result = _exact_nonnegative_int(value, field_name)
    if result <= 0:
        raise AggregateBlocked(f"{field_name}_invalid")
    return result


def _nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AggregateBlocked(f"{field_name}_invalid")
    return value


def _exact_keys(
    value: object,
    expected: set[str] | frozenset[str],
    reason: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise AggregateBlocked(reason)
    return dict(value)


def _canonical_json_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AggregateBlocked("canonical_json_invalid") from exc
    return _sha256_bytes(payload)


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
        raise AggregateBlocked("canonical_json_invalid") from exc


def _framed_domain_sha256(domain: str, *parts: bytes) -> str:
    framed = bytearray()
    for part in (domain.encode("utf-8"), *parts):
        framed.extend(struct.pack(">Q", len(part)))
        framed.extend(part)
    return _sha256_bytes(bytes(framed))


def _validate_g2_platform_invariants(
    active_platforms: object,
    platform_invariants: object,
    reason: str = "g2_platform_invariant_drift",
) -> dict[str, object]:
    if (
        active_platforms != list(_G2_ACTIVE_PLATFORMS)
        or platform_invariants != _G2_PLATFORM_INVARIANTS
    ):
        raise AggregateBlocked(reason)
    return {
        "active_platforms": list(_G2_ACTIVE_PLATFORMS),
        "platform_invariants": _G2_PLATFORM_INVARIANTS,
    }


def _validate_g2_runtime_source_closure(
    value: object,
    reason: str = "g2_runtime_source_closure_invalid",
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "submodule_commit",
        "dirty_inventory",
        "required_sources",
        "active_platforms",
        "platform_invariants",
        "path_planner_runtime_source_closure_sha256",
    }
    closure = _exact_keys(value, expected_keys, reason)
    if (
        closure["schema_version"] != _G2_RUNTIME_SOURCE_SCHEMA_VERSION
        or not isinstance(closure["submodule_commit"], str)
        or re.fullmatch(
            r"[0-9a-f]{40}",
            closure["submodule_commit"],
        )
        is None
    ):
        raise AggregateBlocked(reason)
    _validate_g2_platform_invariants(
        closure["active_platforms"],
        closure["platform_invariants"],
        reason,
    )
    inventory = closure["dirty_inventory"]
    if not isinstance(inventory, list):
        raise AggregateBlocked(reason)
    normalized_inventory: list[dict[str, str]] = []
    seen_inventory: set[str] = set()
    for raw in inventory:
        row = _exact_keys(raw, {"path", "status"}, reason)
        path = row["path"]
        status = row["status"]
        if (
            not isinstance(path, str)
            or not path
            or "\\" in path
            or path.startswith("/")
            or ".." in path.split("/")
            or not isinstance(status, str)
            or len(status) != 2
            or path in seen_inventory
        ):
            raise AggregateBlocked(reason)
        seen_inventory.add(path)
        normalized_inventory.append({"path": path, "status": status})
    if normalized_inventory != sorted(
        normalized_inventory,
        key=lambda row: row["path"],
    ):
        raise AggregateBlocked(reason)
    sources = closure["required_sources"]
    if not isinstance(sources, list) or not sources:
        raise AggregateBlocked(reason)
    normalized_sources: list[dict[str, object]] = []
    seen_sources: set[str] = set()
    for raw in sources:
        row = _exact_keys(
            raw,
            {"logical_path", "size_bytes", "sha256"},
            reason,
        )
        logical_path = row["logical_path"]
        size_bytes = row["size_bytes"]
        digest = row["sha256"]
        if (
            not isinstance(logical_path, str)
            or not logical_path.startswith("src/path_planner/v2/")
            or not logical_path.endswith(".py")
            or "\\" in logical_path
            or ".." in logical_path.split("/")
            or logical_path in seen_sources
            or type(size_bytes) is not int
            or size_bytes < 0
            or not _is_sha256(digest)
        ):
            raise AggregateBlocked(reason)
        seen_sources.add(logical_path)
        normalized_sources.append(
            {
                "logical_path": logical_path,
                "size_bytes": size_bytes,
                "sha256": digest,
            }
        )
    if normalized_sources != sorted(
        normalized_sources,
        key=lambda row: str(row["logical_path"]),
    ):
        raise AggregateBlocked(reason)
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
    logical_paths = {
        str(row["logical_path"]) for row in normalized_sources
    }
    if not required_layers.issubset(logical_paths) or not any(
        path.startswith("src/path_planner/v2/providers/")
        for path in logical_paths
    ):
        raise AggregateBlocked(reason)
    core = {
        "schema_version": _G2_RUNTIME_SOURCE_SCHEMA_VERSION,
        "submodule_commit": closure["submodule_commit"],
        "dirty_inventory": normalized_inventory,
        "required_sources": normalized_sources,
        "active_platforms": list(_G2_ACTIVE_PLATFORMS),
        "platform_invariants": _G2_PLATFORM_INVARIANTS,
    }
    expected_sha256 = _framed_domain_sha256(
        _G2_RUNTIME_SOURCE_SCHEMA_VERSION,
        _canonical_json_bytes(core),
    )
    if (
        closure["path_planner_runtime_source_closure_sha256"]
        != expected_sha256
    ):
        raise AggregateBlocked(reason)
    return {**core, "path_planner_runtime_source_closure_sha256": expected_sha256}


def _g2_code_sha256_with_runtime_source(
    local_source_sha256: str,
    runtime_source_closure_sha256: str,
) -> str:
    return _framed_domain_sha256(
        "xunce-mid-dual-g2-code-with-runtime-source/v1",
        _canonical_json_bytes(
            {
                "local_source_sha256": local_source_sha256,
                "path_planner_runtime_source_closure_sha256": (
                    runtime_source_closure_sha256
                ),
            }
        ),
    )


def _artifact_text_bytes(text: str) -> bytes:
    return text.replace("\n", os.linesep).encode("utf-8")


def _canonical_json_artifact_text(value: Mapping[str, Any]) -> str:
    try:
        return (
            json.dumps(
                dict(value),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise AggregateBlocked("canonical_json_invalid") from exc


def _parse_json_bytes(payload: bytes, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AggregateBlocked(reason) from exc
    if not isinstance(value, dict):
        raise AggregateBlocked(reason)
    return value


def _parse_jsonl_bytes(payload: bytes, reason: str) -> list[object]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise AggregateBlocked(reason) from exc
    rows: list[object] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AggregateBlocked(reason) from exc
    return rows


def _import_local_script(module_name: str) -> Any:
    scripts = str(Path(__file__).resolve().parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module(module_name)


def _safe_manifest_path(value: object, gate_id: str) -> str:
    path = _nonempty_string(value, f"{gate_id}_manifest_path")
    normalized = path.replace("\\", "/")
    candidate = Path(normalized)
    if (
        candidate.is_absolute()
        or normalized.startswith("/")
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
    ):
        raise AggregateBlocked(f"{gate_id}_manifest_path_invalid")
    return normalized


def _dataclass_kwargs(row: Mapping[str, Any], row_type: type[object]) -> dict[str, Any]:
    names = tuple(item.name for item in fields(row_type))
    missing = [name for name in names if name not in row]
    if missing:
        raise AggregateBlocked(f"row_schema_missing:{','.join(missing)}")
    return {name: row[name] for name in names}


def _status_for(midterm_passed: bool) -> str:
    return "passed" if midterm_passed else "failed"


def midterm_reduced_truth_table(g1: bool, g2: bool, g3: bool) -> bool:
    """The reduced midterm gate requires all three independently recalculated gates."""
    return type(g1) is bool and type(g2) is bool and type(g3) is bool and g1 and g2 and g3


def final_threshold_reduced_truth_table(g1: bool, g2: bool, g3: bool) -> bool:
    """The reduced final-threshold gate also requires all three gates."""
    return type(g1) is bool and type(g2) is bool and type(g3) is bool and g1 and g2 and g3


def blocked_summary(blockers: Iterable[str]) -> dict[str, Any]:
    """Build terminal evidence-not-ready state without a failed/pass alias."""
    unique_blockers = sorted({str(reason) for reason in blockers if str(reason)})
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "scale_profile": SCALE_PROFILE,
        "status": "blocked",
        "formal_evidence_eligible": False,
        "midterm_reduced_gate_passed": False,
        "final_threshold_reduced_gate_passed": False,
        "blockers": unique_blockers,
        "gates": {},
        "sample_counts": {
            "g1_total_episodes": 0,
            "g1_test_q24": 0,
            "g1_unseen24": 0,
            "g2_formal_calls": 0,
            "g3_wheel_episodes": 0,
            "g3_interface_replays": 0,
        },
    }


def _contract_value(
    contract: Mapping[str, Any],
    key: str,
    gate_id: str,
) -> Any:
    if key not in contract:
        raise AggregateBlocked(f"{gate_id}_source_contract_invalid")
    return contract[key]


def _snapshot_verified_manifest(
    source_root: Path,
    gate_id: str,
) -> tuple[dict[str, Any], dict[str, bytes], str]:
    try:
        # The common verifier is invoked exactly once.  All later consumers use
        # the immutable byte snapshot below instead of reopening canonical files.
        verified = MidDualRunStore.verify_manifest(source_root)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise AggregateBlocked(
            f"{gate_id}_manifest_invalid:{type(exc).__name__}"
        ) from exc
    if verified is not True:
        raise AggregateBlocked(f"{gate_id}_manifest_invalid")
    manifest_path = artifact_path(source_root, MID_DUAL_MANIFEST)
    try:
        manifest_bytes = artifact_io.read_bytes(manifest_path)
    except OSError as exc:
        raise AggregateBlocked(f"{gate_id}_manifest_snapshot_missing") from exc
    manifest = _parse_json_bytes(
        manifest_bytes,
        f"{gate_id}_manifest_snapshot_invalid",
    )
    if set(manifest) != {
        "schema_version",
        "config_sha256",
        "formal_evidence_eligible",
        "artifacts",
    }:
        raise AggregateBlocked(f"{gate_id}_manifest_schema_invalid")
    if (
        manifest.get("schema_version") != "mid-dual-manifest/v1"
        or manifest.get("formal_evidence_eligible") is not True
        or not _is_sha256(manifest.get("config_sha256"))
    ):
        raise AggregateBlocked(f"{gate_id}_source_blocked")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        raise AggregateBlocked(f"{gate_id}_manifest_entries_invalid")
    snapshot: dict[str, bytes] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            raise AggregateBlocked(f"{gate_id}_manifest_entry_invalid")
        relative = _safe_manifest_path(entry.get("path"), gate_id)
        digest = entry.get("sha256")
        if not _is_sha256(digest) or relative in snapshot:
            raise AggregateBlocked(f"{gate_id}_manifest_entry_invalid")
        try:
            payload = artifact_io.read_bytes(source_root / relative)
        except OSError as exc:
            raise AggregateBlocked(
                f"{gate_id}_immutable_bytes_missing:{relative}"
            ) from exc
        if _sha256_bytes(payload) != digest:
            raise AggregateBlocked(
                f"{gate_id}_immutable_bytes_drift:{relative}"
            )
        snapshot[relative] = payload
    required = {
        "config.json",
        "results.jsonl",
        "summary.json",
        "report.md",
        "phase-state.jsonl",
        "lineage_audit.json",
    }
    if not required.issubset(snapshot):
        raise AggregateBlocked(f"{gate_id}_manifest_artifact_set_incomplete")
    return manifest, snapshot, _sha256_bytes(manifest_bytes)


def _lineage_code_sha256(
    *,
    audit: Mapping[str, Any],
    snapshot: Mapping[str, bytes],
    expected_sources: Sequence[object],
    gate_id: str,
) -> str:
    if (
        audit.get("schema_version") != "mid-dual-lineage-audit/v1"
        or audit.get("status") != "captured"
        or audit.get("formal_evidence_eligible") is not True
    ):
        raise AggregateBlocked(f"{gate_id}_lineage_invalid")
    rows = audit.get("required_sources")
    if not isinstance(rows, list):
        raise AggregateBlocked(f"{gate_id}_lineage_invalid")
    logical: list[dict[str, object]] = []
    observed_paths: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise AggregateBlocked(f"{gate_id}_lineage_invalid")
        path = _nonempty_string(
            row.get("original_relative_path"),
            f"{gate_id}_lineage_path",
        ).replace("\\", "/")
        digest = row.get("sha256")
        size = row.get("size_bytes")
        if not _is_sha256(digest) or type(size) is not int or size < 0:
            raise AggregateBlocked(f"{gate_id}_lineage_invalid")
        status = _nonempty_string(
            row.get("status"),
            f"{gate_id}_lineage_status",
        )
        expected_keys = {
            "original_relative_path",
            "status",
            "size_bytes",
            "sha256",
        }
        if status != "clean":
            expected_keys.add("snapshot_path")
        if set(row) != expected_keys:
            raise AggregateBlocked(f"{gate_id}_lineage_invalid")
        if status != "clean":
            snapshot_path = _safe_manifest_path(
                row.get("snapshot_path"),
                gate_id,
            )
            source_bytes = snapshot.get(snapshot_path)
            if (
                source_bytes is None
                or len(source_bytes) != size
                or _sha256_bytes(source_bytes) != digest
            ):
                raise AggregateBlocked(f"{gate_id}_lineage_snapshot_invalid")
        observed_paths.append(path)
        logical.append(
            {
                "logical_path": path,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    if observed_paths != list(expected_sources):
        raise AggregateBlocked(f"{gate_id}_lineage_required_sources_mismatch")
    canonical = json.dumps(
        logical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(b"xunce-mid-dual-code/v1\0" + canonical)


def _native_g1_code_lineage(
    *,
    config: Mapping[str, Any],
    snapshot: Mapping[str, bytes],
    expected_sources: Sequence[object],
) -> dict[str, bytes]:
    source_lineage = config.get("source_lineage")
    lineage = _parse_json_bytes(
        snapshot["lineage_audit.json"], "g1_lineage_invalid"
    )
    if (
        not isinstance(source_lineage, Mapping)
        or set(source_lineage)
        != {
            "schema_version",
            "hash_algorithm",
            "required_sources",
        }
        or source_lineage.get("schema_version")
        != "xunce-mid-dual-g1-code-lineage/v1"
        or source_lineage.get("hash_algorithm")
        != "sha256-domain-separated-path-length-bytes/v1"
        or set(lineage)
        != {
            "schema_version",
            "status",
            "formal_evidence_eligible",
            "root_commit",
            "submodule_commit",
            "branch",
            "required_sources",
            "status_inventory",
        }
        or lineage.get("schema_version") != "mid-dual-lineage-audit/v1"
        or lineage.get("status") != "captured"
        or lineage.get("formal_evidence_eligible") is not True
    ):
        raise AggregateBlocked("g1_lineage_invalid")
    config_rows = source_lineage.get("required_sources")
    audit_rows = lineage.get("required_sources")
    expected = [str(value) for value in expected_sources]
    if (
        not isinstance(config_rows, list)
        or not isinstance(audit_rows, list)
        or any(not isinstance(row, Mapping) for row in config_rows)
        or any(not isinstance(row, Mapping) for row in audit_rows)
        or [row.get("path") for row in config_rows] != expected
        or [row.get("original_relative_path") for row in audit_rows]
        != expected
    ):
        raise AggregateBlocked("g1_lineage_required_sources_mismatch")
    digest = hashlib.sha256()
    digest.update(b"xunce-mid-dual-g1-code-lineage/v1\0")
    source_payloads: dict[str, bytes] = {}
    repository_root = Path(__file__).resolve().parents[1]
    for relative, config_row, audit_row in zip(
        expected, config_rows, audit_rows, strict=True
    ):
        if not isinstance(config_row, Mapping) or not isinstance(
            audit_row, Mapping
        ):
            raise AggregateBlocked("g1_lineage_invalid")
        size = config_row.get("size_bytes")
        sha256 = config_row.get("sha256")
        if (
            set(config_row) != {"path", "size_bytes", "sha256"}
            or type(size) is not int
            or size < 0
            or not _is_sha256(sha256)
            or audit_row.get("size_bytes") != size
            or audit_row.get("sha256") != sha256
        ):
            raise AggregateBlocked("g1_lineage_invalid")
        status = audit_row.get("status")
        snapshot_path = audit_row.get("snapshot_path")
        expected_audit_keys = {
            "original_relative_path",
            "status",
            "size_bytes",
            "sha256",
        }
        if status != "clean":
            expected_audit_keys.add("snapshot_path")
        if (
            set(audit_row) != expected_audit_keys
            or not isinstance(status, str)
            or not status
            or status == "query_failed"
        ):
            raise AggregateBlocked("g1_lineage_invalid")
        if snapshot_path is None:
            try:
                payload = artifact_io.read_bytes(repository_root / relative)
            except OSError as exc:
                raise AggregateBlocked("g1_lineage_clean_source_missing") from exc
        else:
            safe_snapshot = _safe_manifest_path(snapshot_path, "g1")
            payload = snapshot.get(safe_snapshot)
            if payload is None:
                raise AggregateBlocked("g1_lineage_snapshot_invalid")
        if len(payload) != size or _sha256_bytes(payload) != sha256:
            raise AggregateBlocked("g1_lineage_bytes_drift")
        path_bytes = relative.encode("utf-8")
        digest.update(len(path_bytes).to_bytes(8, "little"))
        digest.update(path_bytes)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
        source_payloads[relative] = payload
    if digest.hexdigest() != config.get("code_sha256"):
        raise AggregateBlocked("g1_code_sha256_mismatch")
    return source_payloads


def _native_g1_phase_snapshot(
    snapshot: Mapping[str, bytes],
    expected_phase_ids: Sequence[object],
) -> dict[str, dict[str, Any]]:
    states = _parse_jsonl_bytes(
        snapshot["phase-state.jsonl"], "g1_phase_state_invalid"
    )
    attempts = _parse_jsonl_bytes(
        snapshot["phase-attempts.jsonl"], "g1_phase_attempts_invalid"
    )
    expected = [str(value) for value in expected_phase_ids]
    if [row.get("phase_id") for row in states if isinstance(row, Mapping)] != expected:
        raise AggregateBlocked("g1_required_phase_sequence_mismatch")
    phases: dict[str, dict[str, Any]] = {}
    for phase_id, raw_state in zip(expected, states, strict=True):
        state = _exact_keys(
            raw_state,
            {"phase_id", "attempt_id", "row_sha256", "rows_path"},
            "g1_phase_state_invalid",
        )
        accepted = [
            row
            for row in attempts
            if isinstance(row, Mapping)
            and row.get("phase_id") == phase_id
            and row.get("status") == "accepted"
        ]
        if len(accepted) != 1:
            raise AggregateBlocked("g1_phase_attempts_invalid")
        attempt = accepted[0]
        if (
            state["attempt_id"] != attempt.get("attempt_id")
            or state["row_sha256"] != attempt.get("row_sha256")
            or state["rows_path"] != attempt.get("rows_path")
            or not _is_sha256(state["row_sha256"])
        ):
            raise AggregateBlocked("g1_phase_attempt_binding_invalid")
        rows_path = _safe_manifest_path(state["rows_path"], "g1")
        audit_path = _safe_manifest_path(attempt.get("audit_path"), "g1")
        rows_bytes = snapshot.get(rows_path)
        audit_bytes = snapshot.get(audit_path)
        if (
            rows_bytes is None
            or audit_bytes is None
            or _sha256_bytes(rows_bytes) != state["row_sha256"]
        ):
            raise AggregateBlocked("g1_phase_bytes_invalid")
        phases[phase_id] = {
            "rows": _parse_jsonl_bytes(
                rows_bytes, "g1_phase_results_invalid"
            ),
            "audit": _parse_json_bytes(
                audit_bytes, "g1_phase_audit_invalid"
            ),
            "rows_bytes": rows_bytes,
        }
    if snapshot["results.jsonl"] != b"".join(
        phases[phase_id]["rows_bytes"] for phase_id in expected
    ):
        raise AggregateBlocked("g1_results_phase_concat_invalid")
    return phases


def _validate_phase_sequence(
    payload: bytes,
    expected: Sequence[object],
    gate_id: str,
) -> None:
    rows = _parse_jsonl_bytes(payload, f"{gate_id}_phase_state_invalid")
    if any(not isinstance(row, Mapping) for row in rows):
        raise AggregateBlocked(f"{gate_id}_phase_state_invalid")
    if [row.get("phase_id") for row in rows] != list(expected):
        raise AggregateBlocked(f"{gate_id}_required_phase_sequence_mismatch")


def _validate_g2_p04_results_snapshot(snapshot: Mapping[str, bytes]) -> None:
    """Require canonical G2 results to be exactly the formal p04 payload."""
    phase_rows = _parse_jsonl_bytes(
        snapshot["phase-state.jsonl"], "g2_phase_state_invalid"
    )
    if len(phase_rows) != 4 or any(
        not isinstance(row, Mapping) for row in phase_rows
    ):
        raise AggregateBlocked("g2_phase_state_invalid")
    p04 = phase_rows[-1]
    if (
        set(p04) != {"phase_id", "attempt_id", "row_sha256", "rows_path"}
        or p04.get("phase_id") != "p04"
        or not _is_sha256(p04.get("row_sha256"))
    ):
        raise AggregateBlocked("g2_phase_state_invalid")
    p04_path = _safe_manifest_path(p04.get("rows_path"), "g2")
    p04_bytes = snapshot.get(p04_path)
    if (
        p04_bytes is None
        or _sha256_bytes(p04_bytes) != p04["row_sha256"]
        or snapshot["results.jsonl"] != p04_bytes
    ):
        raise AggregateBlocked("g2_results_not_canonical_p04")


def _validate_g2_formal_environment_observation(
    value: object,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    reason = "g2_formal_environment_observation_invalid"
    observation = _exact_keys(
        value,
        {
            "schema_version",
            "phase",
            "captured_utc",
            "host",
            "pid",
            "power_scheme_guid",
            "power_probe_sha256",
            "thread_variables",
            "sample_window_seconds",
            "cpu_percent",
            "memory_percent",
            "memory_available_bytes",
            "disk_busy_percent",
            "disk_free_bytes",
            "disk_root",
            "competing_processes",
            "excluded_process_ids",
            "process_inventory_sha256",
        },
        reason,
    )
    if (
        observation["schema_version"]
        != _G2_FORMAL_ENVIRONMENT_OBSERVATION_SCHEMA_VERSION
        or observation["phase"] not in {"start", "end"}
        or not isinstance(observation["captured_utc"], str)
        or not observation["captured_utc"]
        or not isinstance(observation["host"], str)
        or not observation["host"]
        or type(observation["pid"]) is not int
        or observation["pid"] <= 0
        or not _is_sha256(observation["power_probe_sha256"])
        or not _is_sha256(observation["process_inventory_sha256"])
        or not isinstance(observation["disk_root"], str)
        or not observation["disk_root"]
        or observation["sample_window_seconds"]
        != policy["sample_window_seconds"]
    ):
        raise AggregateBlocked(reason)
    power = observation["power_scheme_guid"]
    if power == "not-recorded":
        raise AggregateBlocked("g2_formal_environment_not_recorded")
    if (
        not isinstance(power, str)
        or power.casefold()
        not in {
            str(item).casefold()
            for item in policy["allowed_power_scheme_guids"]
        }
    ):
        raise AggregateBlocked("g2_formal_power_scheme_not_allowed")
    if observation["thread_variables"] != policy["required_thread_variables"]:
        raise AggregateBlocked("g2_formal_thread_settings_invalid")
    competing = observation["competing_processes"]
    if not isinstance(competing, list):
        raise AggregateBlocked(reason)
    for row in competing:
        checked = _exact_keys(
            row,
            {
                "pid",
                "parent_pid",
                "name",
                "matched_pattern",
                "command_sha256",
            },
            reason,
        )
        if (
            type(checked["pid"]) is not int
            or type(checked["parent_pid"]) is not int
            or not isinstance(checked["name"], str)
            or not isinstance(checked["matched_pattern"], str)
            or not _is_sha256(checked["command_sha256"])
        ):
            raise AggregateBlocked(reason)
    if competing:
        raise AggregateBlocked("g2_formal_competing_process")
    excluded = observation["excluded_process_ids"]
    if (
        not isinstance(excluded, list)
        or not excluded
        or any(type(item) is not int or item <= 0 for item in excluded)
    ):
        raise AggregateBlocked(reason)
    for name in (
        "cpu_percent",
        "memory_percent",
        "memory_available_bytes",
        "disk_busy_percent",
        "disk_free_bytes",
    ):
        measured = observation[name]
        if (
            isinstance(measured, bool)
            or not isinstance(measured, (int, float))
            or not math.isfinite(float(measured))
            or float(measured) < 0.0
        ):
            raise AggregateBlocked(reason)
    limits = policy["limits"]
    limit_checks = (
        float(observation["cpu_percent"])
        <= float(limits["cpu_percent_max"]),
        float(observation["memory_percent"])
        <= float(limits["memory_percent_max"]),
        int(observation["memory_available_bytes"])
        >= int(limits["memory_available_bytes_min"]),
        float(observation["disk_busy_percent"])
        <= float(limits["disk_busy_percent_max"]),
        int(observation["disk_free_bytes"])
        >= int(limits["disk_free_bytes_min"]),
    )
    if not all(limit_checks):
        raise AggregateBlocked("g2_formal_environment_load_invalid")
    return observation


def _validate_g2_formal_environment_audit(
    value: object,
    *,
    policy: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    reason = "g2_formal_environment_audit_invalid"
    audit = _exact_keys(
        value,
        {
            "schema_version",
            "gate_id",
            "scale_profile",
            "run_id",
            "status",
            "formal_evidence_eligible",
            "formal_environment_policy",
            "formal_environment_policy_sha256",
            "lease",
            "lease_sha256",
            "start_observation",
            "start_observation_sha256",
            "end_observation",
            "end_observation_sha256",
            "same_process",
            "formal_row_count",
            "lease_released",
            "blockers",
        },
        reason,
    )
    if audit["formal_environment_policy"] != policy:
        raise AggregateBlocked(reason)
    lease = _exact_keys(
        audit["lease"],
        {
            "schema_version",
            "host",
            "pid",
            "process_start_utc",
            "run_id",
            "run_root",
            "nonce",
            "acquired_utc",
            "lease_sha256",
        },
        reason,
    )
    lease_core = {
        key: lease[key]
        for key in (
            "schema_version",
            "host",
            "pid",
            "process_start_utc",
            "run_id",
            "run_root",
            "nonce",
            "acquired_utc",
        )
    }
    lease_sha256 = _framed_domain_sha256(
        _G2_FORMAL_LEASE_SCHEMA_VERSION,
        _canonical_json_bytes(lease_core),
    )
    start = _validate_g2_formal_environment_observation(
        audit["start_observation"],
        policy,
    )
    end = _validate_g2_formal_environment_observation(
        audit["end_observation"],
        policy,
    )
    if (
        audit["schema_version"]
        != _G2_FORMAL_ENVIRONMENT_AUDIT_SCHEMA_VERSION
        or audit["gate_id"] != "g2"
        or audit["scale_profile"] != SCALE_PROFILE
        or audit["run_id"] != run_id
        or audit["status"] != "passed"
        or audit["formal_evidence_eligible"] is not True
        or audit["formal_environment_policy_sha256"]
        != _canonical_json_sha256(policy)
        or lease["schema_version"] != _G2_FORMAL_LEASE_SCHEMA_VERSION
        or lease["lease_sha256"] != lease_sha256
        or audit["lease_sha256"] != lease_sha256
        or lease["run_id"] != run_id
        or start["phase"] != "start"
        or end["phase"] != "end"
        or start["host"] != lease["host"]
        or end["host"] != lease["host"]
        or start["pid"] != lease["pid"]
        or end["pid"] != lease["pid"]
        or start["pid"] not in start["excluded_process_ids"]
        or end["pid"] not in end["excluded_process_ids"]
        or audit["start_observation_sha256"]
        != _canonical_json_sha256(start)
        or audit["end_observation_sha256"] != _canonical_json_sha256(end)
        or audit["same_process"] is not True
        or audit["formal_row_count"] != G2_FORMAL_CALLS
        or audit["lease_released"] is not True
        or audit["blockers"] != []
    ):
        raise AggregateBlocked(reason)
    return audit


def _validate_g2_runtime_binding_artifacts(
    *,
    snapshot: Mapping[str, bytes],
    config: Mapping[str, Any],
    input_audit: Mapping[str, Any],
    binding: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    policy = _exact_keys(
        config.get("formal_environment_gate"),
        set(_G2_FORMAL_ENVIRONMENT_POLICY),
        "g2_formal_environment_policy_invalid",
    )
    if policy != _G2_FORMAL_ENVIRONMENT_POLICY:
        raise AggregateBlocked("g2_formal_environment_policy_invalid")
    input_projection = _validate_g2_platform_invariants(
        input_audit.get("active_platforms"),
        input_audit.get("platform_invariants"),
    )
    config_projection = _validate_g2_platform_invariants(
        config.get("active_platforms"),
        config.get("platform_invariants"),
    )
    if config_projection != input_projection:
        raise AggregateBlocked("g2_platform_invariant_drift")
    closure = _validate_g2_runtime_source_closure(
        input_audit.get("path_planner_runtime_source_closure")
    )
    closure_sha256 = closure[
        "path_planner_runtime_source_closure_sha256"
    ]
    if (
        input_audit.get(
            "path_planner_runtime_source_closure_sha256"
        )
        != closure_sha256
        or config.get(
            "path_planner_runtime_source_closure_sha256"
        )
        != closure_sha256
    ):
        raise AggregateBlocked("g2_runtime_source_closure_binding")
    approval = input_audit.get("approval")
    if (
        not isinstance(approval, Mapping)
        or approval.get("path_planner_runtime_source_closure") != closure
        or approval.get(
            "path_planner_runtime_source_closure_sha256"
        )
        != closure_sha256
    ):
        raise AggregateBlocked("g2_runtime_source_approval_binding")

    closure_path = _safe_manifest_path(
        binding.get("runtime_source_closure_audit_path"),
        "g2",
    )
    invariants_path = _safe_manifest_path(
        binding.get("platform_invariants_audit_path"),
        "g2",
    )
    formal_environment_path = _safe_manifest_path(
        binding.get("formal_environment_audit_path"),
        "g2",
    )
    if (
        closure_path not in snapshot
        or invariants_path not in snapshot
        or formal_environment_path not in snapshot
    ):
        raise AggregateBlocked("g2_runtime_source_audit_missing")
    stored_closure = _parse_json_bytes(
        snapshot[closure_path],
        "g2_runtime_source_closure_audit_invalid",
    )
    if (
        _validate_g2_runtime_source_closure(stored_closure) != closure
        or snapshot[closure_path]
        != _artifact_text_bytes(
            _canonical_json_artifact_text(stored_closure)
        )
    ):
        raise AggregateBlocked("g2_runtime_source_closure_audit_invalid")
    expected_invariants_audit = {
        "schema_version": (
            "xunce-mid-dual-g2-platform-invariants-audit/v1"
        ),
        "gate_id": "g2",
        "scale_profile": SCALE_PROFILE,
        "run_id": run_id,
        "status": "passed",
        "formal_evidence_eligible": True,
        **input_projection,
        "path_planner_runtime_source_closure_sha256": closure_sha256,
        "config_sha256": config.get("config_sha256"),
        "input_sha256": config.get("input_sha256"),
        "code_sha256": config.get("code_sha256"),
    }
    stored_invariants = _parse_json_bytes(
        snapshot[invariants_path],
        "g2_platform_invariant_audit_invalid",
    )
    if (
        stored_invariants != expected_invariants_audit
        or snapshot[invariants_path]
        != _artifact_text_bytes(
            _canonical_json_artifact_text(stored_invariants)
        )
    ):
        raise AggregateBlocked("g2_platform_invariant_audit_invalid")

    state_rows = _parse_jsonl_bytes(
        snapshot["phase-state.jsonl"],
        "g2_phase_state_invalid",
    )
    attempts_bytes = snapshot.get("phase-attempts.jsonl")
    if attempts_bytes is None:
        raise AggregateBlocked("g2_phase_runtime_binding_missing")
    attempt_rows = _parse_jsonl_bytes(
        attempts_bytes,
        "g2_phase_attempts_invalid",
    )
    p04_environment_audit: dict[str, Any] | None = None
    for state in state_rows:
        if not isinstance(state, Mapping):
            raise AggregateBlocked("g2_phase_state_invalid")
        phase_id = state.get("phase_id")
        matching = [
            row
            for row in attempt_rows
            if isinstance(row, Mapping)
            and row.get("phase_id") == phase_id
            and row.get("status") == "accepted"
            and row.get("attempt_id") == state.get("attempt_id")
        ]
        if len(matching) != 1:
            raise AggregateBlocked("g2_phase_attempt_binding_invalid")
        audit_path = _safe_manifest_path(
            matching[0].get("audit_path"),
            "g2",
        )
        audit_bytes = snapshot.get(audit_path)
        if audit_bytes is None:
            raise AggregateBlocked("g2_phase_runtime_binding_missing")
        phase_audit = _parse_json_bytes(
            audit_bytes,
            "g2_phase_runtime_binding_invalid",
        )
        try:
            _validate_g2_platform_invariants(
                phase_audit.get("active_platforms"),
                phase_audit.get("platform_invariants"),
                "g2_phase_runtime_binding_invalid",
            )
        except AggregateBlocked as exc:
            raise AggregateBlocked(
                "g2_phase_runtime_binding_invalid"
            ) from exc
        if (
            phase_audit.get("gate_id") != "g2"
            or phase_audit.get("phase_id") != phase_id
            or phase_audit.get(
                "path_planner_runtime_source_closure_sha256"
            )
            != closure_sha256
        ):
            raise AggregateBlocked("g2_phase_runtime_binding_invalid")
        if phase_id == "p04":
            p04_environment_audit = _validate_g2_formal_environment_audit(
                phase_audit.get("formal_environment_audit"),
                policy=policy,
                run_id=run_id,
            )
            expected_binding = {
                "formal_environment_audit_schema_version": (
                    _G2_FORMAL_ENVIRONMENT_AUDIT_SCHEMA_VERSION
                ),
                "formal_environment_audit_sha256": (
                    _canonical_json_sha256(p04_environment_audit)
                ),
                "formal_lease_sha256": p04_environment_audit["lease_sha256"],
            }
            if any(
                phase_audit.get(key) != expected
                for key, expected in expected_binding.items()
            ):
                raise AggregateBlocked(
                    "g2_formal_environment_phase_binding_invalid"
                )
    if p04_environment_audit is None:
        raise AggregateBlocked("g2_formal_environment_phase_binding_invalid")
    final_environment_audit = _parse_json_bytes(
        snapshot[formal_environment_path],
        "g2_formal_environment_audit_invalid",
    )
    final_environment_audit = _validate_g2_formal_environment_audit(
        final_environment_audit,
        policy=policy,
        run_id=run_id,
    )
    if snapshot[formal_environment_path] != _artifact_text_bytes(
        _canonical_json_artifact_text(final_environment_audit)
    ):
        raise AggregateBlocked("g2_formal_environment_audit_invalid")
    if (
        p04_environment_audit["lease_sha256"]
        == final_environment_audit["lease_sha256"]
        and p04_environment_audit != final_environment_audit
    ):
        raise AggregateBlocked("g2_formal_environment_audit_invalid")
    return closure


def _reconstruct_existing_g1_input_audit(
    *,
    config: Mapping[str, Any],
    manifest_bytes: bytes,
) -> dict[str, Any]:
    """Rebuild the exact G1 input audit without trusting a stored audit file."""

    manifest = _parse_json_bytes(
        manifest_bytes,
        "g1_existing_run_assessment_input_invalid",
    )
    scenario = config.get("scenario_manifest")
    checkpoint = config.get("checkpoint")
    input_pointer = config.get("input_audit")
    cohorts = manifest.get("cohorts")
    proofs_value = manifest.get("denominator_proofs")
    attestation = manifest.get("policy_blind_attestation")
    if (
        manifest.get("schema_version") != "mid-dual-scenario-freeze/v1"
        or manifest.get("completion_status") != "complete"
        or not isinstance(scenario, Mapping)
        or not isinstance(checkpoint, Mapping)
        or not isinstance(input_pointer, Mapping)
        or not isinstance(cohorts, Mapping)
        or not isinstance(proofs_value, list)
        or not isinstance(attestation, Mapping)
        or scenario.get("schema_version") != "mid-dual-scenario-freeze/v1"
        or scenario.get("sha256") != _sha256_bytes(manifest_bytes)
        or checkpoint.get("update") != 80
        or checkpoint.get("sha256") != _UPDATE80_CHECKPOINT_SHA256
        or checkpoint.get("policy_state_sha256")
        != _UPDATE80_POLICY_STATE_SHA256
    ):
        raise AggregateBlocked("g1_existing_run_assessment_input_invalid")
    forbidden = attestation.get("forbidden_inputs_used")
    if (
        not isinstance(forbidden, Mapping)
        or set(forbidden)
        != {"checkpoint", "policy", "result", "reward", "runtime"}
        or any(value is not False for value in forbidden.values())
    ):
        raise AggregateBlocked("g1_existing_run_assessment_input_invalid")

    proofs: dict[str, Mapping[str, Any]] = {}
    for raw in proofs_value:
        if not isinstance(raw, Mapping):
            raise AggregateBlocked("g1_existing_run_assessment_input_invalid")
        scenario_id = raw.get("scenario_id")
        key = raw.get("key")
        if (
            not isinstance(scenario_id, str)
            or not scenario_id
            or scenario_id in proofs
            or raw.get("coverage_denominator_source") != _DENOMINATOR_SOURCE
            or raw.get("coverage_denominator_algorithm")
            != _DENOMINATOR_ALGORITHM
            or raw.get("exact") is not True
            or not _is_sha256(raw.get("coverable_mask_sha256"))
            or type(raw.get("coverable_cell_count")) is not int
            or int(raw["coverable_cell_count"]) <= 0
            or not isinstance(key, Mapping)
            or key.get("max_slope_deg") != 30.0
        ):
            raise AggregateBlocked("g1_existing_run_assessment_input_invalid")
        proofs[scenario_id] = raw

    expected_sizes = {
        "test_q24": 24,
        "unseen24": 24,
        "test_c24": 24,
        "validation3": 3,
        "replay3": 3,
    }
    normalized: dict[str, list[str]] = {}
    for cohort, expected_size in expected_sizes.items():
        values = cohorts.get(cohort)
        if (
            not isinstance(values, list)
            or len(values) != expected_size
            or len(set(values)) != expected_size
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise AggregateBlocked("g1_existing_run_assessment_input_invalid")
        normalized[cohort] = list(values)
    formal_scenarios = [
        *normalized["test_q24"],
        *normalized["unseen24"],
    ]
    if (
        len(set(formal_scenarios)) != 48
        or any(scenario_id not in proofs for scenario_id in formal_scenarios)
    ):
        raise AggregateBlocked("g1_existing_run_assessment_input_invalid")

    formal_jobs: list[dict[str, Any]] = []
    for split in ("test_q24", "unseen24"):
        for index, scenario_id in enumerate(normalized[split]):
            proof = proofs[scenario_id]
            formal_jobs.append(
                {
                    "split": split,
                    "scenario_id": scenario_id,
                    "episode_id": f"{split}-episode-{index:02d}",
                    "episode_index": index,
                    "lane_id": f"lane-{index % 8}",
                    "denominator_sha256": proof["coverable_mask_sha256"],
                    "denominator_cell_count": proof["coverable_cell_count"],
                    "denominator_source": _DENOMINATOR_SOURCE,
                    "denominator_algorithm": _DENOMINATOR_ALGORITHM,
                }
            )
    scenario_path = scenario.get("path")
    if not isinstance(scenario_path, str) or not scenario_path:
        raise AggregateBlocked("g1_existing_run_assessment_input_invalid")
    reconstructed = {
        "schema_version": "xunce-mid-dual-g1-input-audit/v1",
        "gate_id": "g1",
        "runner_id": "run_xunce_mid_dual_g1_coverage/v1",
        "scale_profile": SCALE_PROFILE,
        "status": "verified",
        "scenario_manifest_path": scenario_path,
        "scenario_manifest_schema_version": (
            "mid-dual-scenario-freeze/v1"
        ),
        "scenario_manifest_sha256": _sha256_bytes(manifest_bytes),
        "scenario_manifest_canonical_sha256": _canonical_json_sha256(
            manifest
        ),
        "policy_blind_attestation_sha256": _canonical_json_sha256(
            attestation
        ),
        "checkpoint_sha256": _UPDATE80_CHECKPOINT_SHA256,
        "policy_state_sha256": _UPDATE80_POLICY_STATE_SHA256,
        "denominator_source": _DENOMINATOR_SOURCE,
        "denominator_algorithm": _DENOMINATOR_ALGORITHM,
        "formal_split_order": ["test_q24", "unseen24"],
        "formal_jobs": formal_jobs,
        "test_c24_scenario_ids": normalized["test_c24"],
        "validation3_scenario_ids": normalized["validation3"],
        "replay3_scenario_ids": normalized["replay3"],
    }
    reconstructed_sha256 = _canonical_json_sha256(reconstructed)
    if (
        input_pointer.get("path") != "g1_input_audit.json"
        or input_pointer.get("schema_version")
        != "xunce-mid-dual-g1-input-audit/v1"
        or input_pointer.get("sha256") != reconstructed_sha256
        or config.get("input_sha256") != reconstructed_sha256
    ):
        raise AggregateBlocked(
            "g1_existing_run_assessment_input_reconstruction_mismatch"
        )
    return reconstructed


def _assessment_review_matches_projection(
    *,
    review_metrics: Mapping[str, Any],
    projection: Mapping[str, Any],
    input_audit: Mapping[str, Any],
    verified: object,
) -> bool:
    splits = review_metrics.get("splits")
    if (
        review_metrics.get("schema_version")
        != "xunce-mid-dual-g1-existing-run-independent-recompute/v1"
        or review_metrics.get("formal_episode_count") != 48
        or review_metrics.get("split_order") != ["test_q24", "unseen24"]
        or not isinstance(splits, Mapping)
        or set(splits) != {"test_q24", "unseen24"}
        or not _is_sha256(review_metrics.get("coverage_projection_sha256"))
        or not _is_sha256(review_metrics.get("phase_state_sha256"))
        or review_metrics.get("input_audit_canonical_sha256")
        != _canonical_json_sha256(input_audit)
        or review_metrics.get("scenario_manifest_sha256")
        != input_audit.get("scenario_manifest_sha256")
        or _finite_number(
            review_metrics.get("max_traversable_slope_deg"),
            "g1_existing_run_assessment_review_mismatch",
        )
        != 30.0
        or review_metrics.get("safety_clean") is not True
        or review_metrics.get("masked_action_clean") is not True
        or review_metrics.get("replay_required") is not False
        or review_metrics.get("replay_status") != "not_run_by_ruling"
        or review_metrics.get("midterm_reduced_passed")
        is not projection.get("g1_coverage_80_passed")
        or review_metrics.get("final_threshold_reduced_passed")
        is not projection.get("g1_coverage_99_passed")
        or review_metrics.get("review_result_status")
        != projection.get("status")
        or review_metrics.get("review_result_status")
        != getattr(verified, "numerical_result_status", None)
    ):
        return False
    for split in ("test_q24", "unseen24"):
        reviewed = splits.get(split)
        recomputed = projection.get("splits", {}).get(split)
        if not isinstance(reviewed, Mapping) or not isinstance(
            recomputed,
            Mapping,
        ):
            return False
        if any(
            reviewed.get(review_field) != recomputed.get(recomputed_field)
            for review_field, recomputed_field in (
                ("sample_count", "sample_count"),
                ("coverage_80_count", "coverage_80_count"),
                ("coverage_99_count", "coverage_99_count"),
                ("g1_coverage_80_passed", "midterm_reduced_passed"),
                (
                    "g1_coverage_99_passed",
                    "final_threshold_reduced_passed",
                ),
            )
        ):
            return False
        try:
            reviewed_mean = _finite_number(
                reviewed.get("mean"),
                "g1_existing_run_assessment_review_mismatch",
            )
            recomputed_mean = _finite_number(
                recomputed.get("mean"),
                "g1_existing_run_assessment_review_mismatch",
            )
        except AggregateBlocked:
            return False
        if (
            not math.isclose(
                reviewed_mean,
                recomputed_mean,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            or reviewed.get("safety_violation_count") != 0
            or reviewed.get("masked_action_count") != 0
            or reviewed.get("integer_denominator_check") != "passed"
            or reviewed.get("phase_audit_equality") is not True
        ):
            return False
    return True


def _load_verified_existing_g1_assessment_source(
    *,
    source_root: Path,
    envelope_root: Path,
    contract: Mapping[str, Any],
    assessment_module: object,
) -> dict[str, Any]:
    try:
        verified = assessment_module.verify_existing_run_assessment(
            envelope_root=envelope_root,
            expected_source_root=source_root,
        )
    except Exception as exc:
        raise AggregateBlocked(
            "g1_existing_run_assessment_invalid"
        ) from exc
    hash_fields = (
        "phase_snapshot_sha256",
        "stop_evidence_sha256",
        "review_manifest_sha256",
        "review_result_sha256",
        "envelope_sha256",
        "envelope_manifest_sha256",
        "verifier_source_sha256",
    )
    if (
        _normalized_absolute(getattr(verified, "source_root", ""))
        != _normalized_absolute(source_root)
        or _normalized_absolute(getattr(verified, "envelope_root", ""))
        != _normalized_absolute(envelope_root)
        or getattr(verified, "assessment_use_status", None)
        != "authorized_existing_run"
        or getattr(verified, "strict_prestart_lineage_status", None)
        != "not_satisfied"
        or getattr(verified, "lineage_limitation_acknowledged", None)
        is not True
        or getattr(verified, "numerical_gate_status", None)
        != "recomputed_from_raw_rows"
        or getattr(verified, "numerical_result_status", None)
        not in {"passed", "failed"}
        or any(
            not _is_sha256(getattr(verified, field, None))
            for field in hash_fields
        )
    ):
        raise AggregateBlocked("g1_existing_run_assessment_invalid")
    snapshot = getattr(verified, "snapshot", None)
    review_metrics = getattr(verified, "review_metrics", None)
    if not isinstance(snapshot, Mapping) or not isinstance(
        review_metrics,
        Mapping,
    ):
        raise AggregateBlocked("g1_existing_run_assessment_invalid")
    required_snapshot = (
        "config.json",
        "external/scenario-manifest.json",
        "phases/p03/a01/results.jsonl",
        "phases/p04/a01/results.jsonl",
    )
    if any(type(snapshot.get(path)) is not bytes for path in required_snapshot):
        raise AggregateBlocked("g1_existing_run_assessment_snapshot_invalid")
    config = _parse_json_bytes(
        snapshot["config.json"],
        "g1_existing_run_assessment_config_invalid",
    )
    expected_config_keys = {
        "schema_version",
        "gate_id",
        "runner_id",
        "scale_profile",
        "run_id",
        "mode",
        "output_root",
        "required_phase_ids",
        "required_phases",
        "checkpoint",
        "scenario_manifest",
        "denominator",
        "execution",
        "bootstrap",
        "schemas",
        "base_config_sha256",
        "input_audit",
        "input_sha256",
        "source_lineage",
        "code_sha256",
        "repair_lineage",
        "config_sha256",
    }
    config_without_sha256 = dict(config)
    supplied_config_sha256 = config_without_sha256.pop(
        "config_sha256",
        None,
    )
    if (
        set(config) != expected_config_keys
        or config.get("schema_version") != contract["config_schema_version"]
        or config.get("gate_id") != "g1"
        or config.get("runner_id") != contract["runner_id"]
        or config.get("scale_profile") != SCALE_PROFILE
        or config.get("mode") != "formal"
        or config.get("output_root")
        != str(source_root).replace("\\", "/")
        or config.get("required_phase_ids") != contract["required_phase_ids"]
        or config.get("repair_lineage") is not None
        or not _is_sha256(config.get("input_sha256"))
        or not _is_sha256(config.get("code_sha256"))
        or supplied_config_sha256
        != _canonical_json_sha256(config_without_sha256)
    ):
        raise AggregateBlocked("g1_existing_run_assessment_config_invalid")
    input_audit = _reconstruct_existing_g1_input_audit(
        config=config,
        manifest_bytes=snapshot["external/scenario-manifest.json"],
    )
    rows: list[dict[str, Any]] = []
    trace_row_count = 0
    for phase_id, split in (("p03", "test_q24"), ("p04", "unseen24")):
        phase_rows = _parse_jsonl_bytes(
            snapshot[f"phases/{phase_id}/a01/results.jsonl"],
            "g1_existing_run_assessment_rows_invalid",
        )
        coverage = [
            dict(row)
            for row in phase_rows
            if isinstance(row, Mapping)
            and row.get("row_kind") == "coverage_episode"
        ]
        if (
            len(coverage) != 24
            or any(
                row.get("phase_id") != phase_id
                or row.get("split") != split
                for row in coverage
            )
            or any(
                not isinstance(row, Mapping)
                or row.get("row_kind")
                not in {"coverage_episode", "decision", "planner_call"}
                for row in phase_rows
            )
        ):
            raise AggregateBlocked("g1_existing_run_assessment_rows_invalid")
        rows.extend(coverage)
        trace_row_count += len(phase_rows) - len(coverage)
    projection = recompute_g1(rows, config, input_audit)
    if not _assessment_review_matches_projection(
        review_metrics=review_metrics,
        projection=projection,
        input_audit=input_audit,
        verified=verified,
    ):
        raise AggregateBlocked(
            "g1_existing_run_assessment_review_mismatch"
        )
    assessment_binding = _validate_g1_evidence_binding(
        {
            "g1_evidence_kind": G1_ASSESSMENT_EVIDENCE_KIND,
            "g1_source_manifest_sha256": (
                verified.envelope_manifest_sha256
            ),
            "assessment_envelope_sha256": verified.envelope_sha256,
            "assessment_manifest_sha256": (
                verified.envelope_manifest_sha256
            ),
            "phase_snapshot_sha256": verified.phase_snapshot_sha256,
            "stop_evidence_sha256": verified.stop_evidence_sha256,
            "review_manifest_sha256": verified.review_manifest_sha256,
            "review_result_sha256": verified.review_result_sha256,
            "assessment_use_status": verified.assessment_use_status,
            "strict_prestart_lineage_status": (
                verified.strict_prestart_lineage_status
            ),
            "lineage_limitation_acknowledged": (
                verified.lineage_limitation_acknowledged
            ),
            "numerical_gate_status": verified.numerical_gate_status,
            "numerical_result_status": verified.numerical_result_status,
        }
    )
    use_mapping = {
        "schema_version": (
            "xunce-mid-dual-g1-existing-run-assessment-binding/v1"
        ),
        "source_root": str(source_root).replace("\\", "/"),
        "envelope_root": str(envelope_root).replace("\\", "/"),
        **assessment_binding,
    }
    return {
        "source_kind": "g1_existing_run_assessment",
        "root": source_root,
        "evidence_root": envelope_root,
        "run_id": config["run_id"],
        "config": config,
        "rows": rows,
        "input_audit": input_audit,
        "metric_projection": projection,
        "manifest_sha256": verified.envelope_manifest_sha256,
        "trace_row_count": trace_row_count,
        "review_metrics": dict(review_metrics),
        "g1_evidence_kind": G1_ASSESSMENT_EVIDENCE_KIND,
        "assessment_binding": assessment_binding,
        "g1_existing_run_assessment": use_mapping,
        "platform_invariants": {
            "wheel": {"max_traversable_slope_deg": 30.0}
        },
        "platform_invariants_source_sha256": (
            verified.verifier_source_sha256
        ),
    }


def _load_verified_native_g1_source(
    *,
    source_root: Path,
    manifest: Mapping[str, Any],
    snapshot: Mapping[str, bytes],
    manifest_sha256: str,
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Load a completed G1 root through its native seven-phase contract."""
    g1 = _import_local_script("run_xunce_mid_dual_g1_coverage")
    required_artifacts = {
        "routing.json",
        "phase-attempts.jsonl",
        "environment_audit.json",
        "g1_input_audit.json",
    }
    if not required_artifacts.issubset(snapshot):
        raise AggregateBlocked("g1_native_artifact_set_incomplete")
    expected_config_keys = {
        "schema_version", "gate_id", "runner_id", "scale_profile",
        "run_id", "mode", "output_root", "required_phase_ids",
        "required_phases", "checkpoint", "scenario_manifest",
        "denominator", "execution", "bootstrap", "schemas",
        "base_config_sha256", "input_audit", "input_sha256",
        "source_lineage", "code_sha256", "repair_lineage",
        "config_sha256",
    }
    if set(config) != expected_config_keys:
        raise AggregateBlocked("g1_native_config_exact_schema_mismatch")
    run_id = _nonempty_string(config.get("run_id"), "g1_run_id")
    expected_output_root = str(source_root).replace("\\", "/")
    checkpoint = config.get("checkpoint")
    scenario = config.get("scenario_manifest")
    input_pointer = config.get("input_audit")
    if (
        config.get("schema_version") != contract["config_schema_version"]
        or config.get("gate_id") != "g1"
        or config.get("runner_id") != contract["runner_id"]
        or config.get("scale_profile") != SCALE_PROFILE
        or config.get("mode") != "formal"
        or config.get("output_root") != expected_output_root
        or config.get("required_phase_ids") != contract["required_phase_ids"]
        or config.get("required_phases") != list(g1.G1_REQUIRED_PHASES)
        or config.get("repair_lineage") is not None
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("update") != 80
        or checkpoint.get("sha256") != _UPDATE80_CHECKPOINT_SHA256
        or checkpoint.get("policy_state_sha256")
        != _UPDATE80_POLICY_STATE_SHA256
        or not isinstance(scenario, Mapping)
        or set(scenario) != {"path", "schema_version", "sha256"}
        or scenario.get("schema_version") != "mid-dual-scenario-freeze/v1"
        or not isinstance(scenario.get("path"), str)
        or not scenario["path"].replace("\\", "/").startswith("D:/")
        or not _is_sha256(scenario.get("sha256"))
        or not isinstance(input_pointer, Mapping)
        or set(input_pointer) != {"path", "schema_version", "sha256"}
        or input_pointer.get("path") != "g1_input_audit.json"
        or input_pointer.get("schema_version")
        != contract["input_audit_schema_version"]
        or input_pointer.get("sha256") != config.get("input_sha256")
        or not _is_sha256(config.get("input_sha256"))
        or not _is_sha256(config.get("code_sha256"))
        or config.get("config_sha256") != manifest.get("config_sha256")
    ):
        raise AggregateBlocked("g1_native_config_invalid")
    config_without_sha = dict(config)
    supplied_config_sha = config_without_sha.pop("config_sha256")
    if supplied_config_sha != _canonical_json_sha256(config_without_sha):
        raise AggregateBlocked("g1_config_sha256_drift")

    input_audit = _parse_json_bytes(
        snapshot["g1_input_audit.json"], "g1_input_audit_invalid"
    )
    input_audit_text = _canonical_json_artifact_text(input_audit)
    if (
        snapshot["g1_input_audit.json"]
        != _artifact_text_bytes(input_audit_text)
        or _canonical_json_sha256(input_audit) != config["input_sha256"]
        or input_audit.get("scenario_manifest_sha256")
        != scenario.get("sha256")
        or input_audit.get("scenario_manifest_path")
        != scenario.get("path")
        or input_audit.get("scenario_manifest_schema_version")
        != scenario.get("schema_version")
    ):
        raise AggregateBlocked("g1_input_sha256_mismatch")
    source_payloads = _native_g1_code_lineage(
        config=config,
        snapshot=snapshot,
        expected_sources=contract["required_lineage_sources"],
    )
    base_config_bytes = source_payloads.get(
        "configs/xunce_mid_dual_g1_coverage_v1.json"
    )
    stage6_config_bytes = source_payloads.get(
        "configs/ppo_highres_frontier_stage6_v1.json"
    )
    if base_config_bytes is None or stage6_config_bytes is None:
        raise AggregateBlocked("g1_base_config_missing")
    base_config = _parse_json_bytes(
        base_config_bytes, "g1_base_config_invalid"
    )
    try:
        validated_base = g1.validate_g1_config_payload(base_config)
    except Exception as exc:
        raise AggregateBlocked("g1_base_config_invalid") from exc
    if (
        config.get("base_config_sha256") != _sha256_bytes(base_config_bytes)
        or config.get("checkpoint") != validated_base.get("checkpoint")
        or config.get("denominator") != validated_base.get("denominator")
        or config.get("execution") != validated_base.get("execution")
        or config.get("bootstrap") != validated_base.get("bootstrap")
        or config.get("schemas") != validated_base.get("schemas")
    ):
        raise AggregateBlocked("g1_base_config_binding_invalid")
    stage6_config = _parse_json_bytes(
        stage6_config_bytes,
        "g1_platform_slope_invariant",
    )
    stage6_safety = stage6_config.get("safety")
    if (
        not isinstance(stage6_safety, Mapping)
        or stage6_safety.get("max_traversable_slope_deg") != 30.0
    ):
        raise AggregateBlocked("g1_platform_slope_invariant")
    environment = _parse_json_bytes(
        snapshot["environment_audit.json"], "g1_environment_invalid"
    )
    if (
        environment.get("schema_version") != "mid-dual-environment-audit/v1"
        or environment.get("status") != "captured"
        or environment.get("formal_evidence_eligible") is not True
        or not isinstance(environment.get("probe"), Mapping)
    ):
        raise AggregateBlocked("g1_environment_invalid")

    phases = _native_g1_phase_snapshot(
        snapshot, contract["required_phase_ids"]
    )
    for phase_id in contract["required_phase_ids"]:
        audit = phases[phase_id]["audit"]
        if (
            audit.get("schema_version")
            != "xunce-mid-dual-g1-phase-audit/v1"
            or audit.get("gate_id") != "g1"
            or audit.get("runner_id") != contract["runner_id"]
            or audit.get("phase_id") != phase_id
            or audit.get("phase_name") != g1.G1_PHASE_NAMES[phase_id]
        ):
            raise AggregateBlocked("g1_phase_audit_invalid")
    p01 = phases["p01"]["audit"]
    if (
        phases["p01"]["rows"]
        or p01.get("schema_version") != "xunce-mid-dual-g1-phase-audit/v1"
        or p01.get("phase_name") != "preflight"
        or p01.get("status") != "passed"
        or p01.get("scenario_manifest_sha256") != scenario.get("sha256")
        or p01.get("input_sha256") != config["input_sha256"]
        or p01.get("code_sha256") != config["code_sha256"]
        or p01.get("checkpoint_sha256") != _UPDATE80_CHECKPOINT_SHA256
        or p01.get("policy_state_sha256")
        != _UPDATE80_POLICY_STATE_SHA256
    ):
        raise AggregateBlocked("g1_p01_invalid")
    p02 = phases["p02"]["audit"]
    dry_run = p02.get("dry_run")
    dry_run_results = (
        dry_run.get("results") if isinstance(dry_run, Mapping) else None
    )
    if (
        phases["p02"]["rows"]
        or p02.get("phase_name") != "validation_dry_run"
        or p02.get("status") != "passed"
        or not isinstance(dry_run, Mapping)
        or dry_run.get("schema_version")
        != "xunce-mid-dual-g1-validation-dry-run-audit/v1"
        or dry_run.get("status") != "passed"
        or dry_run.get("scenario_count") != 3
        or dry_run.get("scenario_ids")
        != input_audit.get("validation3_scenario_ids")
        or not isinstance(dry_run_results, list)
        or len(dry_run_results) != 3
        or any(
            not isinstance(row, Mapping)
            for row in dry_run_results
        )
        or [
            row.get("scenario_id")
            for row in dry_run_results
        ]
        != input_audit.get("validation3_scenario_ids")
        or dry_run.get("trace_sha256")
        != _canonical_json_sha256(dry_run_results)
    ):
        raise AggregateBlocked("g1_p02_invalid")

    coverage_rows: list[dict[str, Any]] = []
    trace_keys: set[tuple[str, str, int]] = set()
    for phase_id, split in (("p03", "test_q24"), ("p04", "unseen24")):
        raw_rows = phases[phase_id]["rows"]
        audit = phases[phase_id]["audit"]
        coverage = [
            dict(row)
            for row in raw_rows
            if isinstance(row, Mapping)
            and row.get("row_kind") == "coverage_episode"
        ]
        traces = [
            dict(row)
            for row in raw_rows
            if isinstance(row, Mapping)
            and row.get("row_kind") in {"decision", "planner_call"}
        ]
        if (
            len(coverage) != 24
            or len(coverage) + len(traces) != len(raw_rows)
            or audit.get("schema_version")
            != "xunce-mid-dual-g1-phase-audit/v1"
            or audit.get("phase_name") != g1.G1_PHASE_NAMES[phase_id]
            or audit.get("status") != "complete"
            or audit.get("cohort") != split
            or audit.get("coverage_episode_count") != 24
            or audit.get("decision_count")
            != sum(row["row_kind"] == "decision" for row in traces)
            or audit.get("planner_call_count")
            != sum(row["row_kind"] == "planner_call" for row in traces)
        ):
            raise AggregateBlocked("g1_native_phase_rows_invalid")
        for row in traces:
            key = (
                str(row.get("row_kind")),
                str(row.get("episode_id")),
                _exact_nonnegative_int(
                    row.get("step_index"), "g1_trace_step_index"
                ),
            )
            if (
                set(row) != set(g1.G1_TRACE_ROW_KEYS)
                or row.get("schema_version")
                != {
                    "decision": g1.G1_DECISION_SCHEMA_VERSION,
                    "planner_call": g1.G1_PLANNER_CALL_SCHEMA_VERSION,
                }[str(row.get("row_kind"))]
                or row.get("phase_id") != phase_id
                or row.get("phase_name") != split
                or row.get("config_sha256") != config["config_sha256"]
                or row.get("input_sha256") != config["input_sha256"]
                or row.get("code_sha256") != config["code_sha256"]
                or row.get("source_sha256") != config["code_sha256"]
                or row.get("scenario_manifest_sha256")
                != scenario.get("sha256")
                or not isinstance(row.get("trace"), Mapping)
                or key in trace_keys
            ):
                raise AggregateBlocked("g1_native_trace_invalid")
            _canonical_json_sha256(row)
            trace_keys.add(key)
        coverage_rows.extend(coverage)
    metric_projection = recompute_g1(coverage_rows, config, input_audit)
    replay = phases["p05"]["audit"].get("replay")
    if (
        phases["p05"]["rows"]
        or phases["p05"]["audit"].get("phase_name") != "replay3"
        or phases["p05"]["audit"].get("status") != "passed"
        or not isinstance(replay, Mapping)
    ):
        raise AggregateBlocked("g1_replay_invalid")
    try:
        native_summary = g1.recompute_g1_summary(
            test_q24_rows=[
                row for row in coverage_rows if row["split"] == "test_q24"
            ],
            unseen24_rows=[
                row for row in coverage_rows if row["split"] == "unseen24"
            ],
            mode="formal",
            replay=replay,
        )
        g1.validate_canonical_g1_summary(native_summary)
    except Exception as exc:
        raise AggregateBlocked("g1_native_summary_recompute_invalid") from exc
    stored_summary = _parse_json_bytes(
        snapshot["summary.json"], "g1_stored_summary_invalid"
    )
    p06 = phases["p06"]["audit"]
    p07 = phases["p07"]["audit"]
    if (
        stored_summary != native_summary
        or phases["p06"]["rows"]
        or p06.get("phase_name") != "recompute"
        or p06.get("status") != "complete"
        or p06.get("canonical_summary") != native_summary
        or p06.get("canonical_summary_sha256")
        != _canonical_json_sha256(native_summary)
        or phases["p07"]["rows"]
        or p07.get("phase_name") != "finalize"
        or p07.get("status") != "ready"
        or p07.get("accepted_phase_ids") != contract["required_phase_ids"]
    ):
        raise AggregateBlocked("g1_native_summary_mismatch")
    expected_report = _artifact_text_bytes(
        g1.render_g1_report(native_summary)
    )
    if snapshot["report.md"] != expected_report:
        raise AggregateBlocked("g1_native_report_mismatch")
    routing = _parse_json_bytes(snapshot["routing.json"], "g1_routing_invalid")
    if routing != g1._routing(native_summary):  # noqa: SLF001
        raise AggregateBlocked("g1_routing_mismatch")
    return {
        "source_kind": "g1_native",
        "root": source_root,
        "evidence_root": source_root,
        "run_id": run_id,
        "config": dict(config),
        "rows": coverage_rows,
        "input_audit": input_audit,
        "stored_summary": stored_summary,
        "native_summary": native_summary,
        "metric_projection": metric_projection,
        "report_bytes": snapshot["report.md"],
        "summary_bytes": snapshot["summary.json"],
        "manifest_sha256": manifest_sha256,
        "trace_row_count": len(trace_keys),
        "g1_evidence_kind": G1_NATIVE_EVIDENCE_KIND,
        "assessment_binding": {
            "g1_evidence_kind": G1_NATIVE_EVIDENCE_KIND,
            "g1_source_manifest_sha256": manifest_sha256,
        },
        "platform_invariants": {
            "wheel": {"max_traversable_slope_deg": 30.0}
        },
        "platform_invariants_source_sha256": _sha256_bytes(
            stage6_config_bytes
        ),
    }


def _validate_g3_evidence_snapshot(
    *,
    snapshot: Mapping[str, bytes],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    g3 = _import_local_script("run_xunce_mid_dual_g3_closed_loop")
    g1_evidence_binding = _validate_g1_evidence_binding(
        config.get("g1_evidence_binding")
    )
    upstream_binding = config.get("upstream_binding")
    if (
        not isinstance(upstream_binding, Mapping)
        or _validate_g1_evidence_binding(
            upstream_binding.get("g1_evidence_binding")
        )
        != g1_evidence_binding
    ):
        raise AggregateBlocked("g3_g1_evidence_binding_invalid")
    required = {
        "phase-attempts.jsonl",
        "upstream_lineage_audit.json",
        "g3_decisions_audit.json",
        "g3_observations_audit.json",
        "environment_audit.json",
    }
    if not required.issubset(snapshot):
        raise AggregateBlocked("g3_evidence_audit_set_incomplete")
    environment = _parse_json_bytes(
        snapshot["environment_audit.json"], "g3_environment_invalid"
    )
    if (
        environment.get("schema_version") != "mid-dual-environment-audit/v1"
        or environment.get("status") != "captured"
        or environment.get("formal_evidence_eligible") is not True
    ):
        raise AggregateBlocked("g3_environment_invalid")
    states = _parse_jsonl_bytes(
        snapshot["phase-state.jsonl"], "g3_phase_state_invalid"
    )
    attempts = _parse_jsonl_bytes(
        snapshot["phase-attempts.jsonl"], "g3_phase_attempts_invalid"
    )
    if [row.get("phase_id") for row in states if isinstance(row, Mapping)] != [
        "p01", "p02"
    ]:
        raise AggregateBlocked("g3_required_phase_sequence_mismatch")
    phase_rows: dict[str, list[dict[str, Any]]] = {}
    phase_audits: dict[str, dict[str, Any]] = {}
    phase_bytes: list[bytes] = []
    accepted_prefix: list[str] = []
    upstream_bytes = snapshot["upstream_lineage_audit.json"]
    upstream = _parse_json_bytes(
        upstream_bytes, "g3_upstream_lineage_invalid"
    )
    upstream_text = _canonical_json_artifact_text(upstream)
    if upstream_bytes != _artifact_text_bytes(upstream_text):
        raise AggregateBlocked("g3_upstream_lineage_bytes_noncanonical")
    upstream_sha256 = _sha256_bytes(upstream_text.encode("utf-8"))
    invariant_projection = _validate_g2_platform_invariants(
        config.get("active_platforms"),
        config.get("platform_invariants"),
        "g3_platform_slope_invariant",
    )
    platform_audit_path = _safe_manifest_path(
        config["evidence_binding"].get(
            "platform_invariants_audit_path"
        ),
        "g3",
    )
    platform_audit_bytes = snapshot.get(platform_audit_path)
    if platform_audit_bytes is None:
        raise AggregateBlocked("g3_platform_invariants_audit_invalid")
    platform_audit = _parse_json_bytes(
        platform_audit_bytes,
        "g3_platform_invariants_audit_invalid",
    )
    expected_platform_audit = {
        "schema_version": (
            "xunce-mid-dual-g3-platform-invariants-audit/v1"
        ),
        "gate_id": "g3",
        "scale_profile": SCALE_PROFILE,
        "run_id": config.get("run_id"),
        "status": "passed",
        "formal_evidence_eligible": True,
        **invariant_projection,
        "full_scale_acceptance": False,
        "config_sha256": config.get("config_sha256"),
        "input_sha256": config.get("input_sha256"),
        "code_sha256": config.get("code_sha256"),
        "upstream_lineage_audit_sha256": upstream_sha256,
    }
    if (
        platform_audit != expected_platform_audit
        or platform_audit_bytes
        != _artifact_text_bytes(
            _canonical_json_artifact_text(platform_audit)
        )
    ):
        raise AggregateBlocked("g3_platform_invariants_audit_invalid")
    platform_audit_sha256 = _sha256_bytes(
        _canonical_json_artifact_text(platform_audit).encode("utf-8")
    )
    for phase_id, raw_state in zip(("p01", "p02"), states, strict=True):
        state = _exact_keys(
            raw_state,
            {"phase_id", "attempt_id", "row_sha256", "rows_path"},
            "g3_phase_state_invalid",
        )
        accepted = [
            row
            for row in attempts
            if isinstance(row, Mapping)
            and row.get("phase_id") == phase_id
            and row.get("status") == "accepted"
        ]
        if len(accepted) != 1:
            raise AggregateBlocked("g3_phase_attempts_invalid")
        attempt = accepted[0]
        if any(
            state[field] != attempt.get(field)
            for field in ("attempt_id", "row_sha256", "rows_path")
        ):
            raise AggregateBlocked("g3_phase_attempt_binding_invalid")
        rows_path = _safe_manifest_path(state["rows_path"], "g3")
        audit_path = _safe_manifest_path(attempt.get("audit_path"), "g3")
        rows_bytes = snapshot.get(rows_path)
        audit_bytes = snapshot.get(audit_path)
        if (
            rows_bytes is None
            or audit_bytes is None
            or _sha256_bytes(rows_bytes) != state["row_sha256"]
        ):
            raise AggregateBlocked("g3_phase_bytes_invalid")
        rows = [
            dict(row)
            for row in _parse_jsonl_bytes(
                rows_bytes, "g3_phase_results_invalid"
            )
            if isinstance(row, Mapping)
        ]
        try:
            projection = g3.validate_g3_phase_rows(
                phase_id=phase_id,
                rows=rows,
                accepted_phase_ids=accepted_prefix,
            )
        except Exception as exc:
            raise AggregateBlocked("g3_phase_semantics_invalid") from exc
        audit = _parse_json_bytes(audit_bytes, "g3_phase_audit_invalid")
        first = rows[0]
        if (
            audit.get("schema_version")
            != "xunce-mid-dual-g3-phase-audit/v1"
            or audit.get("gate_id") != "g3"
            or audit.get("runner_id")
            != "run_xunce_mid_dual_g3_closed_loop/v1"
            or audit.get("phase_id") != phase_id
            or audit.get("phase_name")
            != {"p01": "wheel_closed_loop", "p02": "interface_replay"}[
                phase_id
            ]
            or audit.get("status") != "complete"
            or audit.get("formal_sample") is not True
            or audit.get("formal_evidence_eligible") is not True
            or _validate_g2_platform_invariants(
                audit.get("active_platforms"),
                audit.get("platform_invariants"),
                "g3_phase_audit_invalid",
            )
            != invariant_projection
            or audit.get("platform_invariants_audit_sha256")
            != platform_audit_sha256
            or audit.get("upstream_lineage_audit_sha256") != upstream_sha256
            or _validate_g1_evidence_binding(
                audit.get("g1_evidence_binding")
            )
            != g1_evidence_binding
            or audit.get("phase_projection") != projection
            or any(
                audit.get(field) != first.get(field)
                for field in (
                    "config_sha256",
                    "input_sha256",
                    "code_sha256",
                    "g1_source_manifest_sha256",
                    "g2_source_manifest_sha256",
                )
            )
            or audit.get("config_sha256") != config.get("config_sha256")
            or audit.get("input_sha256") != config.get("input_sha256")
            or audit.get("code_sha256") != config.get("code_sha256")
            or audit.get("frozen_manifest_sha256")
            != config["upstream_binding"].get(
                "freeze_manifest_sha256"
            )
            or (
                phase_id == "p01"
                and audit.get("frozen_manifest_sha256")
                != first.get("frozen_manifest_sha256")
            )
        ):
            raise AggregateBlocked("g3_phase_audit_invalid")
        if phase_id == "p02" and (
            audit.get("g2_cohort_sha256")
            != config["upstream_binding"].get("g2_cohort_sha256")
            or audit.get("g2_approval_sha256")
            != config["upstream_binding"].get("g2_approval_sha256")
            or audit.get("g2_hopper_resolution_sha256")
            != config["upstream_binding"].get(
                "g2_hopper_resolution_sha256"
            )
        ):
            raise AggregateBlocked("g3_phase_audit_invalid")
        phase_rows[phase_id] = rows
        phase_audits[phase_id] = audit
        phase_bytes.append(rows_bytes)
        accepted_prefix.append(phase_id)
    if snapshot["results.jsonl"] != b"".join(phase_bytes):
        raise AggregateBlocked("g3_results_phase_concat_invalid")
    wheel_rows = phase_rows["p01"]
    try:
        decisions, observations = g3._wheel_authority_audits(  # noqa: SLF001
            wheel_rows
        )
    except Exception as exc:
        raise AggregateBlocked("g3_wheel_authority_invalid") from exc
    stored_decisions = _parse_json_bytes(
        snapshot["g3_decisions_audit.json"], "g3_decision_audit_invalid"
    )
    stored_observations = _parse_json_bytes(
        snapshot["g3_observations_audit.json"],
        "g3_observation_audit_invalid",
    )
    if stored_decisions != decisions or stored_observations != observations:
        raise AggregateBlocked("g3_wheel_authority_audit_mismatch")
    decision_text = _canonical_json_artifact_text(stored_decisions)
    observation_text = _canonical_json_artifact_text(stored_observations)
    if (
        snapshot["g3_decisions_audit.json"]
        != _artifact_text_bytes(decision_text)
        or snapshot["g3_observations_audit.json"]
        != _artifact_text_bytes(observation_text)
    ):
        raise AggregateBlocked("g3_wheel_authority_audit_noncanonical")
    if (
        phase_audits["p01"].get("decision_audit_sha256")
        != _sha256_bytes(decision_text.encode("utf-8"))
        or phase_audits["p01"].get("observation_audit_sha256")
        != _sha256_bytes(observation_text.encode("utf-8"))
    ):
        raise AggregateBlocked("g3_wheel_authority_audit_binding_invalid")
    if (
        set(upstream)
        != {
            "schema_version",
            "gate_id",
            "formal_evidence_eligible",
            "active_platforms",
            "platform_invariants",
            "upstream_binding",
            "freeze",
            "g1",
            "g2",
        }
        or _validate_g2_platform_invariants(
            upstream.get("active_platforms"),
            upstream.get("platform_invariants"),
            "g3_upstream_lineage_invalid",
        )
        != invariant_projection
        or upstream.get("schema_version")
        != "xunce-mid-dual-g3-upstream-lineage/v1"
        or upstream.get("gate_id") != "g3"
        or upstream.get("formal_evidence_eligible") is not True
        or upstream.get("upstream_binding") != config.get("upstream_binding")
    ):
        raise AggregateBlocked("g3_upstream_lineage_invalid")
    return {
        "upstream_lineage": upstream,
        "upstream_lineage_audit_sha256": upstream_sha256,
        "decision_audit": stored_decisions,
        "observation_audit": stored_observations,
    }


def _load_verified_source(
    root: str | Path,
    gate_id: str,
    contract: Mapping[str, Any],
    *,
    g1_assessment_envelope: str | Path | None = None,
    assessment_module: object | None = None,
) -> dict[str, Any]:
    contract = _exact_keys(
        contract,
        {
            "schema_version",
            "gate_id",
            "config_schema_version",
            "runner_id",
            "required_phase_ids",
            "output_base",
            "input_audit_schema_version",
            "report_audit_schema_version",
            "report_renderer_id",
            "required_lineage_sources",
        },
        f"{gate_id}_source_contract_invalid",
    )
    if (
        contract["schema_version"] != "xunce-mid-dual-source-contract/v1"
        or contract["gate_id"] != gate_id
        or not isinstance(contract["required_phase_ids"], list)
        or not contract["required_phase_ids"]
        or not isinstance(contract["required_lineage_sources"], list)
        or not contract["required_lineage_sources"]
    ):
        raise AggregateBlocked(f"{gate_id}_source_contract_invalid")
    source_root = Path(root)
    if not source_root.is_absolute():
        raise AggregateBlocked(f"{gate_id}_root_not_absolute")
    if gate_id == "g1":
        assessment = assessment_module or _import_local_script(
            "xunce_mid_dual_g1_existing_run_assessment"
        )
        authorized_root = Path(str(assessment.AUTHORIZED_SOURCE_ROOT))
        is_authorized_stopped_root = (
            _normalized_absolute(source_root)
            == _normalized_absolute(authorized_root)
        )
        if is_authorized_stopped_root:
            if g1_assessment_envelope is None:
                raise AggregateBlocked(
                    "g1_existing_run_assessment_required"
                )
            envelope_root = Path(g1_assessment_envelope)
            if not envelope_root.is_absolute():
                raise AggregateBlocked(
                    "g1_existing_run_assessment_scope_mismatch"
                )
            return _load_verified_existing_g1_assessment_source(
                source_root=source_root,
                envelope_root=envelope_root,
                contract=contract,
                assessment_module=assessment,
            )
        if g1_assessment_envelope is not None:
            raise AggregateBlocked(
                "g1_existing_run_assessment_scope_mismatch"
            )
    manifest, snapshot, manifest_sha256 = _snapshot_verified_manifest(
        source_root,
        gate_id,
    )
    config = _parse_json_bytes(
        snapshot["config.json"],
        f"{gate_id}_config_schema_invalid",
    )
    if gate_id == "g1" and "evidence_binding" not in config:
        return _load_verified_native_g1_source(
            source_root=source_root,
            manifest=manifest,
            snapshot=snapshot,
            manifest_sha256=manifest_sha256,
            contract=contract,
            config=config,
        )
    expected_config_keys = {
        "schema_version",
        "gate_id",
        "runner_id",
        "run_id",
        "output_root",
        "scale_profile",
        "input_sha256",
        "code_sha256",
        "required_phase_ids",
        "source_contract_sha256",
        "evidence_binding",
        "config_sha256",
    }
    if gate_id == "g3":
        expected_config_keys.update(
            {
                "active_platforms",
                "platform_invariants",
                "g1_evidence_binding",
                "upstream_binding",
            }
        )
    elif gate_id == "g2":
        expected_config_keys.update(
            {
                "active_platforms",
                "platform_invariants",
                "path_planner_runtime_source_closure_sha256",
                "formal_environment_gate",
            }
        )
    if config.get("schema_version") != _contract_value(
        contract,
        "config_schema_version",
        gate_id,
    ):
        raise AggregateBlocked(f"{gate_id}_config_schema_version_drift")
    if set(config) != expected_config_keys:
        raise AggregateBlocked(f"{gate_id}_config_exact_schema_mismatch")
    if (
        config.get("gate_id") != gate_id
        or config.get("runner_id")
        != _contract_value(contract, "runner_id", gate_id)
        or config.get("scale_profile") != SCALE_PROFILE
        or config.get("output_root")
        != _contract_value(contract, "output_base", gate_id)
        or config.get("required_phase_ids")
        != _contract_value(contract, "required_phase_ids", gate_id)
        or config.get("source_contract_sha256")
        != _canonical_json_sha256(contract)
    ):
        raise AggregateBlocked(f"{gate_id}_config_contract_drift")
    run_id = _nonempty_string(config.get("run_id"), f"{gate_id}_run_id")
    if config.get("config_sha256") != manifest.get("config_sha256"):
        raise AggregateBlocked(f"{gate_id}_config_sha256_drift")
    if not _is_sha256(config.get("input_sha256")) or not _is_sha256(
        config.get("code_sha256")
    ):
        raise AggregateBlocked(f"{gate_id}_config_lineage_invalid")
    binding_keys = {
        "schema_version",
        "input_audit_path",
        "lineage_audit_path",
        "report_audit_path",
    }
    if gate_id == "g2":
        binding_keys.update(
            {
                "runtime_source_closure_audit_path",
                "platform_invariants_audit_path",
                "formal_environment_audit_path",
            }
        )
    elif gate_id == "g3":
        binding_keys.add("platform_invariants_audit_path")
    binding = _exact_keys(
        config.get("evidence_binding"),
        binding_keys,
        f"{gate_id}_evidence_binding_invalid",
    )
    if binding["schema_version"] != f"xunce-mid-dual-{gate_id}-evidence-binding/v1":
        raise AggregateBlocked(f"{gate_id}_evidence_binding_invalid")
    input_path = _safe_manifest_path(binding["input_audit_path"], gate_id)
    lineage_path = _safe_manifest_path(binding["lineage_audit_path"], gate_id)
    report_audit_path = _safe_manifest_path(
        binding["report_audit_path"],
        gate_id,
    )
    if input_path not in snapshot or lineage_path not in snapshot or report_audit_path not in snapshot:
        raise AggregateBlocked(f"{gate_id}_evidence_binding_missing")
    input_audit = _parse_json_bytes(
        snapshot[input_path],
        f"{gate_id}_input_audit_invalid",
    )
    canonical_input_text = _canonical_json_artifact_text(input_audit)
    canonical_input_bytes = canonical_input_text.encode("utf-8")
    if snapshot[input_path] != _artifact_text_bytes(canonical_input_text):
        raise AggregateBlocked(f"{gate_id}_input_audit_bytes_noncanonical")
    if _sha256_bytes(canonical_input_bytes) != config["input_sha256"]:
        raise AggregateBlocked(f"{gate_id}_input_sha256_mismatch")
    if gate_id == "g3":
        g1_binding = _validate_g1_evidence_binding(
            config.get("g1_evidence_binding")
        )
        upstream_binding = config.get("upstream_binding")
        if (
            _validate_g1_evidence_binding(
                input_audit.get("g1_evidence_binding")
            )
            != g1_binding
            or not isinstance(upstream_binding, Mapping)
            or _validate_g1_evidence_binding(
                upstream_binding.get("g1_evidence_binding")
            )
            != g1_binding
        ):
            raise AggregateBlocked("g3_g1_evidence_binding_invalid")
    lineage_audit = _parse_json_bytes(
        snapshot[lineage_path],
        f"{gate_id}_lineage_invalid",
    )
    recomputed_code = _lineage_code_sha256(
        audit=lineage_audit,
        snapshot=snapshot,
        expected_sources=_contract_value(
            contract,
            "required_lineage_sources",
            gate_id,
        ),
        gate_id=gate_id,
    )
    if gate_id == "g2":
        closure_sha256 = input_audit.get(
            "path_planner_runtime_source_closure_sha256"
        )
        if not _is_sha256(closure_sha256):
            raise AggregateBlocked(
                "g2_runtime_source_closure_binding"
            )
        recomputed_code = _g2_code_sha256_with_runtime_source(
            recomputed_code,
            str(closure_sha256),
        )
    if config["code_sha256"] != recomputed_code:
        raise AggregateBlocked(f"{gate_id}_code_sha256_mismatch")
    _validate_phase_sequence(
        snapshot["phase-state.jsonl"],
        _contract_value(contract, "required_phase_ids", gate_id),
        gate_id,
    )
    if gate_id == "g2":
        _validate_g2_runtime_binding_artifacts(
            snapshot=snapshot,
            config=config,
            input_audit=input_audit,
            binding=binding,
            run_id=run_id,
        )
        _validate_g2_p04_results_snapshot(snapshot)
    stored_summary = _parse_json_bytes(
        snapshot["summary.json"],
        f"{gate_id}_stored_summary_invalid",
    )
    if (
        stored_summary.get("status") == "blocked"
        or stored_summary.get("formal_evidence_eligible") is not True
    ):
        raise AggregateBlocked(f"{gate_id}_source_blocked")
    report_audit = _parse_json_bytes(
        snapshot[report_audit_path],
        f"{gate_id}_report_audit_invalid",
    )
    rows = _parse_jsonl_bytes(
        snapshot["results.jsonl"],
        f"{gate_id}_results_jsonl_invalid",
    )
    source = {
        "root": source_root,
        "run_id": run_id,
        "config": config,
        "rows": rows,
        "input_audit": input_audit,
        "stored_summary": stored_summary,
        "report_audit": report_audit,
        "report_bytes": snapshot["report.md"],
        "summary_bytes": snapshot["summary.json"],
        "manifest_sha256": manifest_sha256,
    }
    if gate_id == "g1":
        source.update(
            {
                "source_kind": "g1_manifest_completed",
                "evidence_root": source_root,
                "trace_row_count": sum(
                    isinstance(row, Mapping)
                    and row.get("row_kind") in {"decision", "planner_call"}
                    for row in rows
                ),
                "native_summary": stored_summary,
                "g1_evidence_kind": G1_NATIVE_EVIDENCE_KIND,
                "assessment_binding": {
                    "g1_evidence_kind": G1_NATIVE_EVIDENCE_KIND,
                    "g1_source_manifest_sha256": manifest_sha256,
                },
                "platform_invariants": {
                    "wheel": {"max_traversable_slope_deg": 30.0}
                },
                "platform_invariants_source_sha256": config[
                    "code_sha256"
                ],
            }
        )
    if gate_id == "g3":
        source.update(
            _validate_g3_evidence_snapshot(snapshot=snapshot, config=config)
        )
    return source


def _validate_row_lineage(
    rows: Sequence[object],
    config: Mapping[str, Any],
    gate_id: str,
) -> None:
    expected_config = config["config_sha256"]
    expected_input = config["input_sha256"]
    expected_code = config["code_sha256"]
    for index, row in enumerate(rows):
        prefix = f"{gate_id}_row_{index}"
        if not isinstance(row, Mapping):
            raise AggregateBlocked(f"{gate_id}_row_schema_invalid")
        if row.get("scale_profile") != SCALE_PROFILE:
            raise AggregateBlocked(f"{prefix}_scale_profile_drift")
        if row.get("config_sha256") != expected_config:
            raise AggregateBlocked(f"{prefix}_config_sha256_drift")
        if row.get("input_sha256") != expected_input:
            raise AggregateBlocked(f"{prefix}_input_sha256_drift")
        if row.get("code_sha256") != expected_code:
            raise AggregateBlocked(f"{prefix}_code_sha256_drift")
        if row.get("source_sha256") != expected_code:
            raise AggregateBlocked(f"{prefix}_source_sha256_drift")


def _timing_statistics(values: Sequence[object]) -> dict[str, Any]:
    materialized = tuple(_finite_number(value, "timing") for value in values)
    if not materialized:
        raise AggregateBlocked("timing_partition_empty")
    mean_value = statistics.mean(materialized)
    p95 = nearest_rank(materialized, 0.95)
    maximum = max(materialized)
    midterm = mean_value <= MID_TIME_MS and p95 <= MID_TIME_MS and maximum <= MID_TIME_MS
    final = (
        mean_value <= FINAL_TIME_MS
        and p95 <= FINAL_TIME_MS
        and sum(value <= FINAL_TIME_MS for value in materialized) / len(materialized) >= 0.95
        and maximum <= MID_TIME_MS
    )
    return {
        "sample_count": len(materialized),
        "mean_ms": mean_value,
        "p50_ms": nearest_rank(materialized, 0.50),
        "p95_ms": p95,
        "p99_ms": nearest_rank(materialized, 0.99),
        "sample_stddev_ms": statistics.stdev(materialized) if len(materialized) > 1 else 0.0,
        "min_ms": min(materialized),
        "max_ms": maximum,
        "at_or_below_1000_count": sum(value <= FINAL_TIME_MS for value in materialized),
        "over_2000_count": sum(value > MID_TIME_MS for value in materialized),
        "midterm_reduced_passed": midterm,
        "final_threshold_reduced_passed": final,
    }


def _coverage_statistics(values: Sequence[object]) -> dict[str, Any]:
    materialized = tuple(_finite_number(value, "coverage") for value in values)
    if not materialized:
        raise AggregateBlocked("coverage_partition_empty")
    if any(value < 0.0 or value > 1.0 for value in materialized):
        raise AggregateBlocked("g1_coverage_out_of_range")
    mean_value = statistics.mean(materialized)
    count80 = sum(value >= MID_COVERAGE_THRESHOLD for value in materialized)
    count99 = sum(value >= FINAL_COVERAGE_THRESHOLD for value in materialized)
    midterm = mean_value >= MID_COVERAGE_THRESHOLD and count80 >= 23
    final = mean_value >= FINAL_COVERAGE_THRESHOLD and count99 >= 23
    return {
        "status": "passed" if midterm else "failed",
        "sample_count": len(materialized),
        "mean": mean_value,
        "p50": nearest_rank(materialized, 0.50),
        "p95": nearest_rank(materialized, 0.95),
        "p99": nearest_rank(materialized, 0.99),
        "sample_stddev": (
            statistics.stdev(materialized) if len(materialized) > 1 else 0.0
        ),
        "min": min(materialized),
        "max": max(materialized),
        "coverage_80_count": count80,
        "coverage_99_count": count99,
        "midterm_reduced_passed": midterm,
        "final_threshold_reduced_passed": final,
    }


def recompute_g1(
    rows: Sequence[object],
    config: Mapping[str, Any],
    input_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Recalculate G1 from exact frozen jobs and integer cell counts."""
    expected_audit_keys = {
        "schema_version",
        "gate_id",
        "runner_id",
        "scale_profile",
        "status",
        "scenario_manifest_path",
        "scenario_manifest_schema_version",
        "scenario_manifest_sha256",
        "scenario_manifest_canonical_sha256",
        "policy_blind_attestation_sha256",
        "checkpoint_sha256",
        "policy_state_sha256",
        "denominator_source",
        "denominator_algorithm",
        "formal_split_order",
        "formal_jobs",
        "test_c24_scenario_ids",
        "validation3_scenario_ids",
        "replay3_scenario_ids",
    }
    audit = _exact_keys(
        input_audit,
        expected_audit_keys,
        "g1_input_audit_schema_invalid",
    )
    if (
        audit["schema_version"] != "xunce-mid-dual-g1-input-audit/v1"
        or audit["gate_id"] != "g1"
        or audit["runner_id"] != config["runner_id"]
        or audit["scale_profile"] != SCALE_PROFILE
        or audit["status"] != "verified"
        or audit["checkpoint_sha256"] != _UPDATE80_CHECKPOINT_SHA256
        or audit["policy_state_sha256"] != _UPDATE80_POLICY_STATE_SHA256
        or audit["denominator_source"] != _DENOMINATOR_SOURCE
        or audit["denominator_algorithm"] != _DENOMINATOR_ALGORITHM
        or audit["formal_split_order"] != ["test_q24", "unseen24"]
    ):
        raise AggregateBlocked("g1_input_audit_binding_invalid")
    for field_name in (
        "scenario_manifest_sha256",
        "scenario_manifest_canonical_sha256",
        "policy_blind_attestation_sha256",
        "checkpoint_sha256",
        "policy_state_sha256",
    ):
        if not _is_sha256(audit[field_name]):
            raise AggregateBlocked("g1_input_audit_binding_invalid")
    jobs = audit["formal_jobs"]
    if not isinstance(jobs, list) or len(jobs) != 48:
        raise AggregateBlocked("g1_frozen_episode_binding_mismatch")
    expected_job_keys = {
        "split",
        "scenario_id",
        "episode_id",
        "episode_index",
        "lane_id",
        "denominator_mask_sha256",
        "denominator_cell_count",
        "denominator_source",
        "denominator_algorithm",
    }
    # Task 5 names the frozen mask field denominator_sha256.  The fixture
    # contract retains denominator_mask_sha256 in the audit, so accept either
    # exact spelling but never both.
    frozen_index: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    for raw_job in jobs:
        if not isinstance(raw_job, Mapping):
            raise AggregateBlocked("g1_frozen_episode_binding_mismatch")
        keys = set(raw_job)
        if keys == expected_job_keys:
            mask_key = "denominator_mask_sha256"
        elif keys == (
            expected_job_keys - {"denominator_mask_sha256"}
        ) | {"denominator_sha256"}:
            mask_key = "denominator_sha256"
        else:
            raise AggregateBlocked("g1_frozen_episode_binding_mismatch")
        split = raw_job.get("split")
        index = raw_job.get("episode_index")
        scenario_id = raw_job.get("scenario_id")
        episode_id = raw_job.get("episode_id")
        lane_id = raw_job.get("lane_id")
        if (
            split not in {"test_q24", "unseen24"}
            or type(index) is not int
            or index not in range(24)
            or lane_id != f"lane-{index % 8}"
            or not isinstance(scenario_id, str)
            or not scenario_id
            or not isinstance(episode_id, str)
            or not episode_id
            or not _is_sha256(raw_job.get(mask_key))
            or _exact_positive_int(
                raw_job.get("denominator_cell_count"),
                "g1_denominator_count",
            )
            <= 0
            or raw_job.get("denominator_source") != _DENOMINATOR_SOURCE
            or raw_job.get("denominator_algorithm") != _DENOMINATOR_ALGORITHM
        ):
            raise AggregateBlocked("g1_frozen_episode_binding_mismatch")
        key = (str(split), scenario_id, episode_id, index, str(lane_id))
        if key in frozen_index:
            raise AggregateBlocked("g1_frozen_episode_binding_mismatch")
        normalized = dict(raw_job)
        normalized["denominator_sha256"] = raw_job[mask_key]
        frozen_index[key] = normalized
    if {
        (split, index)
        for split, _, _, index, _ in frozen_index
    } != {
        (split, index)
        for split in ("test_q24", "unseen24")
        for index in range(24)
    }:
        raise AggregateBlocked("g1_frozen_episode_binding_mismatch")

    if len(rows) != 48:
        raise AggregateBlocked("g1_formal_episode_count_mismatch")
    _validate_row_lineage(rows, config, "g1")
    materialized: list[dict[str, Any]] = []
    for raw in rows:
        row = _exact_keys(raw, _G1_ROW_KEYS, "g1_row_schema_invalid")
        if (
            row["row_kind"] != "coverage_episode"
            or row["schema_version"]
            != "xunce-mid-dual-g1-coverage-episode/v1"
            or row["gate_id"] != "g1"
            or row["runner_id"] != config["runner_id"]
            or row["run_id"] != config["run_id"]
            or row["checkpoint_sha256"] != _UPDATE80_CHECKPOINT_SHA256
            or row["policy_state_sha256"] != _UPDATE80_POLICY_STATE_SHA256
            or row["scenario_manifest_sha256"]
            != audit["scenario_manifest_sha256"]
            or row["denominator_source"] != _DENOMINATOR_SOURCE
            or row["denominator_algorithm"] != _DENOMINATOR_ALGORITHM
        ):
            raise AggregateBlocked("g1_checkpoint_binding_mismatch")
        split = row["split"]
        index = _exact_nonnegative_int(row["episode_index"], "g1_episode_index")
        if index >= 24:
            raise AggregateBlocked("g1_frozen_episode_binding_mismatch")
        expected_phase = "p03" if split == "test_q24" else "p04"
        key = (
            str(split),
            _nonempty_string(row["scenario_id"], "g1_scenario_id"),
            _nonempty_string(row["episode_id"], "g1_episode_id"),
            index,
            _nonempty_string(row["lane_id"], "g1_lane_id"),
        )
        frozen = frozen_index.get(key)
        if (
            frozen is None
            or row["phase_id"] != expected_phase
            or row["phase_name"] != split
            or row["lane_id"] != f"lane-{index % 8}"
        ):
            raise AggregateBlocked("g1_frozen_episode_binding_mismatch")
        denominator = _exact_positive_int(
            row["denominator_cell_count"],
            "g1_denominator_count",
        )
        initial = _exact_nonnegative_int(
            row["initial_covered_cell_count"],
            "g1_initial_covered_count",
        )
        final = _exact_nonnegative_int(
            row["final_covered_cell_count"],
            "g1_final_covered_count",
        )
        coverage = _finite_number(row["coverage"], "g1_coverage")
        if coverage < 0.0 or coverage > 1.0:
            raise AggregateBlocked("g1_coverage_out_of_range")
        if (
            row["denominator_sha256"] != frozen["denominator_sha256"]
            or denominator != frozen["denominator_cell_count"]
            or not 0 <= initial <= final <= denominator
            or not math.isclose(
                coverage,
                final / denominator,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        ):
            raise AggregateBlocked("g1_denominator_binding_mismatch")
        _finite_nonnegative(row["elapsed_ms"], "g1_elapsed_ms")
        _exact_nonnegative_int(row["steps_executed"], "g1_steps_executed")
        _nonempty_string(row["termination_reason"], "g1_termination_reason")
        if (
            _exact_nonnegative_int(
                row["safety_violation_count"],
                "g1_safety_violation_count",
            )
            != 0
            or _exact_nonnegative_int(
                row["masked_action_count"],
                "g1_masked_action_count",
            )
            != 0
        ):
            raise AggregateBlocked("g1_integrity_not_clean")
        materialized.append(row)
    if len({row["episode_id"] for row in materialized}) != 48:
        raise AggregateBlocked("g1_duplicate_episode_id")
    if len({row["scenario_id"] for row in materialized}) != 48:
        raise AggregateBlocked("g1_duplicate_scenario_id")
    split_results = {
        split: _coverage_statistics(
            [row["coverage"] for row in materialized if row["split"] == split]
        )
        for split in ("test_q24", "unseen24")
    }
    if any(result["sample_count"] != 24 for result in split_results.values()):
        raise AggregateBlocked("g1_frozen_episode_binding_mismatch")
    midterm = all(
        result["midterm_reduced_passed"] is True
        for result in split_results.values()
    )
    final = all(
        result["final_threshold_reduced_passed"] is True
        for result in split_results.values()
    )
    return {
        "status": _status_for(midterm),
        "sample_count": 48,
        "g1_coverage_80_passed": midterm,
        "g1_coverage_99_passed": final,
        "safety_clean": True,
        "masked_action_clean": True,
        "checkpoint_sha256": _UPDATE80_CHECKPOINT_SHA256,
        "policy_state_sha256": _UPDATE80_POLICY_STATE_SHA256,
        "denominator_integrity_passed": True,
        "splits": split_results,
    }


def recompute_g2(
    rows: Sequence[object],
    config: Mapping[str, Any],
    input_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Recalculate G2 against the manifest-bound 129-request truth crosswalk."""
    audit_keys = {
        "schema_version", "gate_id", "scale_profile", "status",
        "formal_evidence_eligible", "blockers", "truth_bundle_root",
        "truth_manifest_sha256", "truth_freeze_sha256",
        "truth_payload_root_sha256", "truth_source_attestations_sha256",
        "input_set_id", "manifest_core_sha256", "authorization_sha256",
        "active_platforms", "platform_invariants",
        "path_planner_runtime_source_closure",
        "path_planner_runtime_source_closure_sha256",
        "candidate_boundary", "provider_source", "oracle_source",
        "expected_approval_artifact_path", "approval", "hopper_resolution",
        "primitive_label_audit", "small_map_optimum_audit",
        "request_matrix_audit", "ppo_target_audit", "g3_replay_cohort",
        "g3_replay_cohort_sha256", "requests", "provider_requests",
        "terrain_sha256", "formal_row_count",
    }
    audit = _exact_keys(
        input_audit,
        audit_keys,
        "g2_input_audit_schema_invalid",
    )
    if (
        audit["schema_version"] != "xunce-mid-dual-g2-input-audit/v1"
        or audit["gate_id"] != "g2"
        or audit["scale_profile"] != SCALE_PROFILE
        or audit["status"] != "ready"
        or audit["formal_evidence_eligible"] is not True
        or audit["blockers"] != []
        or audit["formal_row_count"] != 0
        or not _is_sha256(audit["truth_manifest_sha256"])
    ):
        raise AggregateBlocked("g2_input_audit_binding_invalid")
    invariant_projection = _validate_g2_platform_invariants(
        audit["active_platforms"],
        audit["platform_invariants"],
    )
    if _validate_g2_platform_invariants(
        config.get("active_platforms"),
        config.get("platform_invariants"),
    ) != invariant_projection:
        raise AggregateBlocked("g2_platform_invariant_drift")
    runtime_source_closure = _validate_g2_runtime_source_closure(
        audit["path_planner_runtime_source_closure"]
    )
    runtime_source_closure_sha256 = runtime_source_closure[
        "path_planner_runtime_source_closure_sha256"
    ]
    if (
        audit["path_planner_runtime_source_closure_sha256"]
        != runtime_source_closure_sha256
        or config.get(
            "path_planner_runtime_source_closure_sha256"
        )
        != runtime_source_closure_sha256
    ):
        raise AggregateBlocked("g2_runtime_source_closure_binding")
    source_keys = {
        "identity",
        "source_bytes_sha256",
        "implementation_sha256",
    }
    provider = _exact_keys(
        audit["provider_source"],
        source_keys,
        "g2_provider_source_invalid",
    )
    oracle = _exact_keys(
        audit["oracle_source"],
        source_keys,
        "g2_oracle_source_invalid",
    )
    for source in (provider, oracle):
        _nonempty_string(source["identity"], "g2_source_identity")
        if not _is_sha256(source["source_bytes_sha256"]) or not _is_sha256(
            source["implementation_sha256"]
        ):
            raise AggregateBlocked("g2_source_identity_invalid")
    if (
        provider["identity"] == oracle["identity"]
        or provider["source_bytes_sha256"] == oracle["source_bytes_sha256"]
        or provider["implementation_sha256"] == oracle["implementation_sha256"]
    ):
        raise AggregateBlocked("g2_provider_oracle_identity_mismatch")
    approval = audit["approval"]
    if not isinstance(approval, Mapping) or (
        approval.get("schema_version")
        != "xunce-mid-dual-g2-artifact-bound-approval/v1"
    ):
        raise AggregateBlocked("g2_approval_schema_invalid")
    required_approval = (
        "approval_id", "approval_record_sha256", "approval_artifact_sha256",
        "approval_artifact_path", "provider_source", "oracle_source",
        "g2_scope_authorized", "g3_scope_authorized",
        "physical_capability_claimed", "hardware_certification_claimed",
        "formal_evidence_eligible",
        "path_planner_runtime_source_closure",
        "path_planner_runtime_source_closure_sha256",
    )
    if (
        any(field not in approval for field in required_approval)
        or not _nonempty_string(approval.get("approval_id"), "g2_approval_id")
        or any(not _is_sha256(approval[field]) for field in (
            "approval_record_sha256", "approval_artifact_sha256"
        ))
        or approval.get("provider_source") != provider
        or approval.get("oracle_source") != oracle
        or approval.get("path_planner_runtime_source_closure")
        != runtime_source_closure
        or approval.get(
            "path_planner_runtime_source_closure_sha256"
        )
        != runtime_source_closure_sha256
        or any(_exact_bool(approval[field], f"g2_approval_{field}") is not True
               for field in ("g2_scope_authorized", "g3_scope_authorized", "formal_evidence_eligible"))
        or any(_exact_bool(approval[field], f"g2_approval_{field}") is not False
               for field in ("physical_capability_claimed", "hardware_certification_claimed"))
    ):
        raise AggregateBlocked("g2_approval_invalid")
    approval_path = _nonempty_string(
        approval.get("approval_artifact_path"), "g2_approval_path"
    ).replace("\\", "/")
    expected_approval_path = _nonempty_string(
        audit["expected_approval_artifact_path"], "g2_expected_approval_path"
    ).replace("\\", "/")
    if approval_path != expected_approval_path or not approval_path.lower().startswith(
        "d:/xunce/inputs/mid_dual/"
    ):
        raise AggregateBlocked("g2_approval_payload_root_invalid")

    provider_requests = audit["provider_requests"]
    if not isinstance(provider_requests, list) or len(provider_requests) != 129:
        raise AggregateBlocked("g2_provider_request_matrix_invalid")
    provider_request_keys: set[tuple[str, str, str]] = set()
    for raw_provider_request in provider_requests:
        if not isinstance(raw_provider_request, Mapping):
            raise AggregateBlocked("g2_provider_request_matrix_invalid")
        platform_name = raw_provider_request.get("platform")
        request_id = raw_provider_request.get("request_id")
        request_sha256 = raw_provider_request.get(
            "provider_request_sha256"
        )
        platform_stack = raw_provider_request.get("platform_stack")
        if (
            platform_name not in _G2_ACTIVE_PLATFORMS
            or not isinstance(request_id, str)
            or not request_id
            or not _is_sha256(request_sha256)
            or not isinstance(platform_stack, Mapping)
            or platform_stack.get("max_traversable_slope_deg") != 30.0
        ):
            raise AggregateBlocked("g2_platform_invariant_drift")
        provider_request_keys.add(
            (
                str(platform_name),
                request_id,
                str(request_sha256),
            )
        )
    if len(provider_request_keys) != 129:
        raise AggregateBlocked("g2_provider_request_matrix_invalid")

    request_keys = {
        "platform",
        "scale",
        "request_index",
        "request_id",
        "schema_version",
        "request_class",
        "outcome_kind",
        "truth_request_sha256",
        "truth_certificate_sha256",
        "terrain_sha256",
        "provider_request_sha256",
        "expected_success",
        "expected_route_l2_valid",
    }
    request_rows = audit["requests"]
    if not isinstance(request_rows, list) or len(request_rows) != 129:
        raise AggregateBlocked("g2_request_matrix_mismatch")
    request_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    request_hashes: set[str] = set()
    class_counts: dict[str, dict[str, int]] = {}
    expected_class_counts = {
        "standard": {
            "normal_reachable": 23,
            "hard_reachable": 7,
            "unreachable": 3,
        },
        "kilometer": {
            "normal_reachable": 6,
            "hard_reachable": 2,
            "unreachable": 2,
        },
    }
    for raw_request in request_rows:
        request = _exact_keys(
            raw_request,
            request_keys,
            "g2_request_schema_invalid",
        )
        if request["schema_version"] != "xunce-mid-dual-g2-truth-provider-crosswalk/v1":
            raise AggregateBlocked("g2_request_schema_invalid")
        platform_name = request["platform"]
        scale = request["scale"]
        request_class = request["request_class"]
        outcome = request["outcome_kind"]
        if platform_name not in G2_PLATFORMS or scale not in {
            "standard",
            "kilometer",
        }:
            raise AggregateBlocked("g2_request_matrix_mismatch")
        if request_class not in {
            "normal_reachable",
            "hard_reachable",
            "unreachable",
        } or outcome not in {"reachable", "unreachable"}:
            raise AggregateBlocked("g2_outcome_taxonomy_invalid")
        expected_outcome = (
            "unreachable" if request_class == "unreachable" else "reachable"
        )
        expected_success = request_class != "unreachable"
        if (
            outcome != expected_outcome
            or _exact_bool(
                request["expected_success"],
                "g2_expected_success",
            )
            is not expected_success
            or _exact_bool(
                request["expected_route_l2_valid"],
                "g2_expected_route_l2_valid",
            )
            is not expected_success
        ):
            raise AggregateBlocked("g2_outcome_taxonomy_invalid")
        request_id = _nonempty_string(request["request_id"], "g2_request_id")
        request_sha = request["provider_request_sha256"]
        for field_name in (
            "provider_request_sha256", "truth_request_sha256",
            "truth_certificate_sha256", "terrain_sha256",
        ):
            if not _is_sha256(request[field_name]):
                raise AggregateBlocked("g2_request_hash_invalid")
        key = (str(platform_name), request_id, str(request_sha))
        if key in request_index:
            raise AggregateBlocked("g2_request_matrix_mismatch")
        request_index[key] = request
        request_hashes.add(str(request_sha))
    if len(request_hashes) != 129:
        raise AggregateBlocked("g2_request_hash_unique_mismatch")
    if set(request_index) != provider_request_keys:
        raise AggregateBlocked("g2_provider_request_matrix_invalid")
    for platform_name in G2_PLATFORMS:
        for scale in ("standard", "kilometer"):
            key = f"{platform_name}/{scale}"
            counts = {
                request_class: sum(
                    request["platform"] == platform_name
                    and request["scale"] == scale
                    and request["request_class"] == request_class
                    for request in request_index.values()
                )
                for request_class in (
                    "normal_reachable",
                    "hard_reachable",
                    "unreachable",
                )
            }
            if counts != expected_class_counts[scale]:
                raise AggregateBlocked("g2_request_matrix_mismatch")
            class_counts[key] = counts

    if len(rows) != G2_FORMAL_CALLS:
        raise AggregateBlocked("g2_formal_call_count_mismatch")
    _validate_row_lineage(rows, config, "g2")
    materialized: list[dict[str, Any]] = []
    repeat_indices: dict[tuple[str, str, str], list[int]] = {}
    call_ids: set[str] = set()
    semantic_by_request: dict[tuple[str, str, str], set[str]] = {}
    provider_results_by_request: dict[tuple[str, str, str], set[str]] = {}
    for raw in rows:
        row = _exact_keys(raw, _G2_ROW_KEYS, "g2_row_schema_invalid")
        if (
            row["row_kind"] != "g2_planning_call"
            or row["schema_version"]
            != "xunce-mid-dual-g2-planning-call-row/v1"
            or row["run_id"] != config["run_id"]
            or row["formal_sample"] is not True
            or row["platform"] not in G2_PLATFORMS
            or row["timing_contract_id"] != _TIMING_CONTRACT_ID
        ):
            raise AggregateBlocked("g2_row_schema_invalid")
        request_id = _nonempty_string(row["request_id"], "g2_row_schema")
        request_sha = row["request_sha256"]
        if not _is_sha256(request_sha):
            raise AggregateBlocked("g2_request_hash_invalid")
        key = (str(row["platform"]), request_id, str(request_sha))
        request = request_index.get(key)
        if request is None:
            raise AggregateBlocked("g2_truth_provider_crosswalk_mismatch")
        bindings = {
            "scale": request["scale"],
            "request_class": request["request_class"],
            "outcome_kind": request["outcome_kind"],
            "truth_sha256": request["truth_request_sha256"],
            "oracle_result_sha256": request["truth_certificate_sha256"],
            "provider_sha256": provider["implementation_sha256"],
            "provider_source_bytes_sha256": provider["source_bytes_sha256"],
            "oracle_sha256": oracle["implementation_sha256"],
            "oracle_source_bytes_sha256": oracle["source_bytes_sha256"],
        }
        if any(row[field] != value for field, value in bindings.items()):
            raise AggregateBlocked("g2_truth_provider_crosswalk_mismatch")
        if (
            not _is_sha256(row["provider_result_sha256"])
            or row["semantic_digest"] != _g2_expected_semantic_digest(
                row["request_sha256"],
                row["provider_success"],
                row["route_l2_valid"],
            )
        ):
            raise AggregateBlocked("g2_provider_result_or_semantic_invalid")
        call_id = _nonempty_string(row["call_id"], "g2_call_id")
        if call_id in call_ids:
            raise AggregateBlocked("g2_duplicate_call_id")
        call_ids.add(call_id)
        repeat = _exact_nonnegative_int(row["repeat_index"], "g2_repeat_index")
        if repeat >= G2_REPEATS:
            raise AggregateBlocked("g2_repeat_index_invalid")
        repeat_indices.setdefault(key, []).append(repeat)
        semantic_by_request.setdefault(key, set()).add(str(row["semantic_digest"]))
        provider_results_by_request.setdefault(key, set()).add(
            str(row["provider_result_sha256"])
        )
        _g2_elapsed_ms_from_raw_ns(row)
        materialized.append(row)
    if any(
        sorted(indices) != list(range(G2_REPEATS))
        for indices in repeat_indices.values()
    ) or len(repeat_indices) != 129:
        raise AggregateBlocked("g2_duplicate_or_missing_repeat_index")
    if any(len(values) != 1 for values in semantic_by_request.values()):
        raise AggregateBlocked("g2_semantic_consensus_mismatch")
    if any(len(values) != 1 for values in provider_results_by_request.values()):
        raise AggregateBlocked("g2_provider_result_consensus_mismatch")

    try:
        canonical = evaluate_formal_g2(
            [
                PlanningCallRow(
                    **_dataclass_kwargs(row, PlanningCallRow)
                )
                for row in materialized
            ]
        )
    except (TypeError, ValueError) as exc:
        raise AggregateBlocked("g2_canonical_evaluator_input") from exc
    if canonical.get("status") == "blocked":
        raise AggregateBlocked(
            str(
                canonical.get(
                    "blocking_reason",
                    "g2_canonical_evaluator_blocked",
                )
            )
        )
    correctness = canonical["reachable_correctness"]
    return {
        "status": canonical["status"],
        "formal_call_count": canonical["formal_call_count"],
        "unique_request_count": canonical["unique_request_count"],
        "active_platforms": canonical["active_platforms"],
        "platform_invariants": canonical["platform_invariants"],
        "g2_all_platforms_2s_passed": canonical[
            "g2_all_platforms_2s_passed"
        ],
        "g2_all_platforms_1s_passed": canonical[
            "g2_all_platforms_1s_passed"
        ],
        "correctness_passed": canonical["correctness_passed"],
        "request_class_counts_by_platform_scale": canonical[
            "request_class_counts_by_platform_scale"
        ],
        "timing_by_platform_scale": canonical["timing_by_platform_scale"],
        "timing_by_platform_scale_outcome": canonical[
            "timing_by_platform_scale_outcome"
        ],
        "timing_by_platform_scale_class": canonical[
            "timing_by_platform_scale_class"
        ],
        "reachable_correctness": {
            "reachable_unique_request_count_by_platform": correctness[
                "reachable_unique_request_count_by_platform"
            ],
            "reachable_unique_success_count_by_platform": correctness[
                "reachable_unique_success_count_by_platform"
            ],
            "unreachable_correct": correctness["unreachable_correct"],
            "semantic_consensus": correctness["semantic_consensus"],
            "truth_provider_crosswalk_valid": True,
        },
    }


def _g3_cell(value: object, reason: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        raise AggregateBlocked(reason)
    return int(value[0]), int(value[1])


def _g3_theta(value: object, reason: str) -> float:
    return _finite_number(value, reason)


def _g3_timing_ms(row: Mapping[str, Any]) -> float:
    components = [
        _exact_nonnegative_int(row.get(field), f"g3_{field}")
        for field in _TIMING_COMPONENT_FIELDS
    ]
    total = _exact_nonnegative_int(row.get("total_ns"), "g3_total_ns")
    if total != sum(components):
        raise AggregateBlocked("g3_timing_contract")
    return total / 1_000_000.0


def _g3_request_id(
    *,
    scenario_id: str,
    step_index: int,
    pre_snapshot_sha256: str,
) -> str:
    encoded = (
        "mid-dual-g3-request/v1\0"
        f"{scenario_id}\0{step_index}\0{pre_snapshot_sha256}"
    ).encode("utf-8")
    return f"g3-wheel-{_sha256_bytes(encoded)}"


def _g3_candidate_sha256(row: Mapping[str, Any]) -> str:
    return _canonical_json_sha256(
        {
            "schema_version": "mid-dual-g3-candidate-binding/v1",
            "scenario_id": _nonempty_string(
                row.get("scenario_id"),
                "g3_wheel_hash_chain",
            ),
            "step_index": _exact_nonnegative_int(
                row.get("step_index"),
                "g3_wheel_hash_chain",
            ),
            "pre_snapshot_sha256": row["pre_snapshot_sha256"],
            "decision_sha256": row["decision_sha256"],
            "candidate_id": _nonempty_string(
                row.get("candidate_id"),
                "g3_wheel_hash_chain",
            ),
            "selected_candidate_cell_xy": list(
                _g3_cell(
                    row.get("selected_candidate_cell_xy"),
                    "g3_wheel_hash_chain",
                )
            ),
            "selected_theta": _g3_theta(
                row.get("selected_theta"),
                "g3_wheel_hash_chain",
            ),
        }
    )


def _g3_planner_request_sha256(row: Mapping[str, Any]) -> str:
    return _canonical_json_sha256(
        {
            "schema_version": "mid-dual-g3-planner-request-binding/v1",
            "request_id": _nonempty_string(
                row.get("request_id"),
                "g3_wheel_hash_chain",
            ),
            "candidate_sha256": row["candidate_sha256"],
            "pre_snapshot_sha256": row["pre_snapshot_sha256"],
        }
    )


def _g3_route_result_sha256(row: Mapping[str, Any]) -> str:
    path_value = row.get("planned_path_cells")
    if not isinstance(path_value, (list, tuple)):
        raise AggregateBlocked("g3_wheel_hash_chain")
    path = [
        list(_g3_cell(value, "g3_wheel_hash_chain"))
        for value in path_value
    ]
    return _canonical_json_sha256(
        {
            "schema_version": "mid-dual-g3-route-result-binding/v1",
            "request_sha256": row["request_sha256"],
            "planner_success": _exact_bool(
                row.get("planner_success"),
                "g3_wheel_hash_chain",
            ),
            "planner_failure_reason": _nonempty_string(
                row.get("planner_failure_reason"),
                "g3_wheel_hash_chain",
            ),
            "planned_path_cells": path,
            "planner_path_length_m": _finite_nonnegative(
                row.get("planner_path_length_m"),
                "g3_wheel_hash_chain",
            ),
            "route_endpoint_cell_xy": list(
                _g3_cell(
                    row.get("route_endpoint_cell_xy"),
                    "g3_wheel_hash_chain",
                )
            ),
            "route_endpoint_theta": _g3_theta(
                row.get("route_endpoint_theta"),
                "g3_wheel_hash_chain",
            ),
        }
    )


def _g3_feedback_sha256(row: Mapping[str, Any]) -> str:
    return _canonical_json_sha256(
        {
            "schema_version": "mid-dual-g3-feedback-binding/v2",
            "route_result_sha256": row["route_result_sha256"],
            "feedback_pose_cell_xy": list(
                _g3_cell(
                    row.get("feedback_pose_cell_xy"),
                    "g3_wheel_hash_chain",
                )
            ),
            "feedback_theta": _g3_theta(
                row.get("feedback_theta"),
                "g3_wheel_hash_chain",
            ),
            "coverage": _finite_number(
                row.get("coverage"),
                "g3_wheel_hash_chain",
            ),
            "safety_violation_count": _exact_nonnegative_int(
                row.get("safety_violation_count"),
                "g3_wheel_hash_chain",
            ),
            "masked_action_count": _exact_nonnegative_int(
                row.get("masked_action_count"),
                "g3_wheel_hash_chain",
            ),
            "post_snapshot_sha256": row["post_snapshot_sha256"],
        }
    )


def _validate_g3_g1_lineage(
    *,
    lineage_value: object,
    evidence_binding: Mapping[str, Any],
    manifest_sha256: str,
    config: Mapping[str, Any],
    trace_row_count: int,
    native_summary: Mapping[str, Any] | None,
    review_metrics: Mapping[str, Any] | None,
    platform_invariants: Mapping[str, Any],
    platform_invariants_source_sha256: str,
) -> None:
    binding = _validate_g1_evidence_binding(evidence_binding)
    common_keys = {
        "root",
        "evidence_root",
        "manifest_sha256",
        "config_sha256",
        "input_sha256",
        "code_sha256",
        "coverage_row_count",
        "trace_row_count",
        "g1_evidence_binding",
        "platform_invariants",
        "platform_invariants_source_sha256",
    }
    summary_key = (
        "native_summary_sha256"
        if binding["g1_evidence_kind"] == G1_NATIVE_EVIDENCE_KIND
        else "review_metrics_sha256"
    )
    lineage = _exact_keys(
        lineage_value,
        {*common_keys, summary_key},
        "g3_upstream_lineage_invalid",
    )
    if (
        not isinstance(lineage.get("root"), str)
        or not lineage["root"]
        or not isinstance(lineage.get("evidence_root"), str)
        or not lineage["evidence_root"]
        or lineage.get("manifest_sha256") != manifest_sha256
        or lineage.get("config_sha256") != config.get("config_sha256")
        or lineage.get("input_sha256") != config.get("input_sha256")
        or lineage.get("code_sha256") != config.get("code_sha256")
        or lineage.get("coverage_row_count") != 48
        or lineage.get("trace_row_count") != trace_row_count
        or _validate_g1_evidence_binding(
            lineage.get("g1_evidence_binding")
        )
        != binding
        or lineage.get("platform_invariants") != dict(platform_invariants)
        or platform_invariants
        != {"wheel": {"max_traversable_slope_deg": 30.0}}
        or not _is_sha256(platform_invariants_source_sha256)
        or lineage.get("platform_invariants_source_sha256")
        != platform_invariants_source_sha256
    ):
        raise AggregateBlocked("g3_upstream_lineage_invalid")
    if binding["g1_evidence_kind"] == G1_NATIVE_EVIDENCE_KIND:
        if (
            native_summary is None
            or lineage.get("native_summary_sha256")
            != _canonical_json_sha256(native_summary)
        ):
            raise AggregateBlocked("g3_upstream_lineage_invalid")
    elif (
        review_metrics is None
        or lineage.get("review_metrics_sha256")
        != _canonical_json_sha256(review_metrics)
    ):
        raise AggregateBlocked("g3_upstream_lineage_invalid")


def recompute_g3(
    rows: Sequence[object],
    config: Mapping[str, Any],
    input_audit: Mapping[str, Any],
    *,
    g1_rows: Sequence[object],
    g2_rows: Sequence[object],
    g1_manifest_sha256: str,
    g2_manifest_sha256: str,
    g1_config: Mapping[str, Any],
    g1_input_audit: Mapping[str, Any],
    g1_native_summary: Mapping[str, Any] | None,
    g1_trace_row_count: int,
    g1_evidence_binding: Mapping[str, Any],
    g1_review_metrics: Mapping[str, Any] | None,
    g1_platform_invariants: Mapping[str, Any],
    g1_platform_invariants_source_sha256: str,
    g2_config: Mapping[str, Any],
    g2_input_audit: Mapping[str, Any],
    upstream_lineage: Mapping[str, Any],
    upstream_lineage_audit_sha256: str,
) -> dict[str, Any]:
    """Independently recalculate G3 from its raw rows and G1/G2 raw roots."""

    audit = _exact_keys(
        input_audit,
        {
            "schema_version",
            "gate_id",
            "run_id",
            "scale_profile",
            "active_platforms",
            "platform_invariants",
            "formal_evidence_eligible",
            "g1_source_manifest_sha256",
            "g2_source_manifest_sha256",
            "g1_evidence_binding",
            "wheel_selections",
            "interface_selections",
        },
        "g3_input_audit_schema_invalid",
    )
    upstream = _exact_keys(
        config.get("upstream_binding"),
        _G3_UPSTREAM_BINDING_KEYS,
        "g3_upstream_binding_invalid",
    )
    for field in _G3_UPSTREAM_BINDING_KEYS - {
        "g2_input_set_id",
        "active_platforms",
        "platform_invariants",
        "g1_evidence_binding",
    }:
        if not _is_sha256(upstream[field]):
            raise AggregateBlocked("g3_upstream_binding_invalid")
    _validate_g2_platform_invariants(
        audit["active_platforms"],
        audit["platform_invariants"],
        "g3_platform_slope_invariant",
    )
    _validate_g2_platform_invariants(
        config.get("active_platforms"),
        config.get("platform_invariants"),
        "g3_platform_slope_invariant",
    )
    _validate_g2_platform_invariants(
        upstream["active_platforms"],
        upstream["platform_invariants"],
        "g3_platform_slope_invariant",
    )
    if (
        not isinstance(upstream["g2_input_set_id"], str)
        or not upstream["g2_input_set_id"]
        or upstream["g2_provider_identity_sha256"]
        == upstream["g2_oracle_identity_sha256"]
    ):
        raise AggregateBlocked("g3_upstream_binding_invalid")
    expected_g1_evidence_binding = _validate_g1_evidence_binding(
        g1_evidence_binding
    )
    if (
        expected_g1_evidence_binding["g1_source_manifest_sha256"]
        != g1_manifest_sha256
    ):
        raise AggregateBlocked("g3_g1_evidence_binding_invalid")
    expected_upstream = {
        "g1_source_manifest_sha256": g1_manifest_sha256,
        "g2_source_manifest_sha256": g2_manifest_sha256,
        "freeze_manifest_sha256": g1_input_audit.get(
            "scenario_manifest_sha256"
        ),
        "g1_config_sha256": g1_config.get("config_sha256"),
        "g1_input_sha256": g1_config.get("input_sha256"),
        "g1_code_sha256": g1_config.get("code_sha256"),
        "g2_config_sha256": g2_config.get("config_sha256"),
        "g2_input_sha256": g2_config.get("input_sha256"),
        "g2_code_sha256": g2_config.get("code_sha256"),
        "g2_input_set_id": g2_input_audit.get("input_set_id"),
        "g2_approval_sha256": _canonical_json_sha256(
            g2_input_audit.get("approval")
        ),
        "g2_cohort_sha256": g2_input_audit.get(
            "g3_replay_cohort_sha256"
        ),
        "g2_provider_identity_sha256": _canonical_json_sha256(
            g2_input_audit.get("provider_source")
        ),
        "g2_oracle_identity_sha256": _canonical_json_sha256(
            g2_input_audit.get("oracle_source")
        ),
        "g2_hopper_resolution_sha256": _canonical_json_sha256(
            g2_input_audit.get("hopper_resolution")
        ),
        "active_platforms": list(_G2_ACTIVE_PLATFORMS),
        "platform_invariants": _G2_PLATFORM_INVARIANTS,
        "g1_evidence_binding": expected_g1_evidence_binding,
    }
    if (
        audit["schema_version"] != "xunce-mid-dual-g3-input-audit/v1"
        or audit["gate_id"] != "g3"
        or audit["run_id"] != config["run_id"]
        or audit["scale_profile"] != SCALE_PROFILE
        or audit["formal_evidence_eligible"] is not True
        or audit["g1_source_manifest_sha256"] != g1_manifest_sha256
        or audit["g2_source_manifest_sha256"] != g2_manifest_sha256
        or _validate_g1_evidence_binding(
            audit["g1_evidence_binding"]
        )
        != expected_g1_evidence_binding
        or upstream != expected_upstream
    ):
        raise AggregateBlocked("g3_cross_root_manifest_binding")
    lineage_envelope = _exact_keys(
        upstream_lineage,
        {
            "schema_version",
            "gate_id",
            "formal_evidence_eligible",
            "active_platforms",
            "platform_invariants",
            "upstream_binding",
            "freeze",
            "g1",
            "g2",
        },
        "g3_upstream_lineage_invalid",
    )
    if (
        lineage_envelope["schema_version"]
        != "xunce-mid-dual-g3-upstream-lineage/v1"
        or lineage_envelope["gate_id"] != "g3"
        or lineage_envelope["formal_evidence_eligible"] is not True
        or _validate_g2_platform_invariants(
            lineage_envelope["active_platforms"],
            lineage_envelope["platform_invariants"],
            "g3_upstream_lineage_invalid",
        )
        != {
            "active_platforms": list(_G2_ACTIVE_PLATFORMS),
            "platform_invariants": _G2_PLATFORM_INVARIANTS,
        }
        or upstream_lineage.get("upstream_binding") != upstream
        or _sha256_bytes(
            (
                json.dumps(
                    dict(upstream_lineage),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
        )
        != upstream_lineage_audit_sha256
    ):
        raise AggregateBlocked("g3_upstream_lineage_invalid")
    _validate_g3_g1_lineage(
        lineage_value=lineage_envelope["g1"],
        evidence_binding=expected_g1_evidence_binding,
        manifest_sha256=g1_manifest_sha256,
        config=g1_config,
        trace_row_count=g1_trace_row_count,
        native_summary=g1_native_summary,
        review_metrics=g1_review_metrics,
        platform_invariants=g1_platform_invariants,
        platform_invariants_source_sha256=(
            g1_platform_invariants_source_sha256
        ),
    )
    lineage_g2 = _exact_keys(
        lineage_envelope["g2"],
        {
            "root",
            "execution_root",
            "manifest_sha256",
            "config_sha256",
            "input_sha256",
            "code_sha256",
            "formal_row_count",
            "input_set_id",
            "approval",
            "cohort",
            "provider_source",
            "oracle_source",
            "hopper_resolution",
        },
        "g3_upstream_lineage_invalid",
    )
    lineage_freeze = _exact_keys(
        lineage_envelope["freeze"],
        {"root", "manifest_sha256", "selected"},
        "g3_upstream_lineage_invalid",
    )
    if (
        lineage_freeze.get("manifest_sha256")
        != upstream["freeze_manifest_sha256"]
        or lineage_g2.get("manifest_sha256") != g2_manifest_sha256
        or lineage_g2.get("config_sha256")
        != g2_config.get("config_sha256")
        or lineage_g2.get("input_sha256") != g2_config.get("input_sha256")
        or lineage_g2.get("code_sha256") != g2_config.get("code_sha256")
        or lineage_g2.get("formal_row_count") != G2_FORMAL_CALLS
        or lineage_g2.get("input_set_id")
        != g2_input_audit.get("input_set_id")
        or lineage_g2.get("approval") != g2_input_audit.get("approval")
        or lineage_g2.get("cohort")
        != g2_input_audit.get("g3_replay_cohort")
        or lineage_g2.get("provider_source")
        != g2_input_audit.get("provider_source")
        or lineage_g2.get("oracle_source")
        != g2_input_audit.get("oracle_source")
        or lineage_g2.get("hopper_resolution")
        != g2_input_audit.get("hopper_resolution")
    ):
        raise AggregateBlocked("g3_upstream_lineage_invalid")

    g1_index: dict[tuple[str, str], dict[str, Any]] = {}
    if len(g1_rows) != 48:
        raise AggregateBlocked("g3_g1_cross_root_join")
    for raw in g1_rows:
        row = _exact_keys(raw, _G1_ROW_KEYS, "g3_g1_cross_root_join")
        split = row["split"]
        scenario_id = row["scenario_id"]
        if split not in {"test_q24", "unseen24"} or not isinstance(
            scenario_id,
            str,
        ):
            raise AggregateBlocked("g3_g1_cross_root_join")
        key = (str(split), scenario_id)
        if key in g1_index:
            raise AggregateBlocked("g3_g1_cross_root_join")
        g1_index[key] = row

    wheel_selections = audit["wheel_selections"]
    if not isinstance(wheel_selections, list) or len(wheel_selections) != 10:
        raise AggregateBlocked("g3_g1_cross_root_join")
    selection_by_episode: dict[str, dict[str, Any]] = {}
    selected_scenarios: set[tuple[str, str]] = set()
    selection_split_counts = {"test_q24": 0, "unseen24": 0}
    wheel_selection_keys = {
        "split",
        "scenario_id",
        "g1_episode_id",
        "g1_episode_index",
        "lane_id",
        "wheel_episode_id",
    }
    for raw in wheel_selections:
        selection = _exact_keys(
            raw,
            wheel_selection_keys,
            "g3_g1_cross_root_join",
        )
        split = selection["split"]
        scenario_id = selection["scenario_id"]
        if split not in selection_split_counts or not isinstance(
            scenario_id,
            str,
        ):
            raise AggregateBlocked("g3_g1_cross_root_join")
        g1 = g1_index.get((str(split), scenario_id))
        episode_id = _nonempty_string(
            selection["wheel_episode_id"],
            "g3_g1_cross_root_join",
        )
        if (
            g1 is None
            or selection["g1_episode_id"] != g1["episode_id"]
            or selection["g1_episode_index"] != g1["episode_index"]
            or selection["lane_id"] != g1["lane_id"]
            or episode_id in selection_by_episode
            or (str(split), scenario_id) in selected_scenarios
        ):
            raise AggregateBlocked("g3_g1_cross_root_join")
        selection_by_episode[episode_id] = selection
        selected_scenarios.add((str(split), scenario_id))
        selection_split_counts[str(split)] += 1
    if selection_split_counts != {"test_q24": 5, "unseen24": 5}:
        raise AggregateBlocked("g3_g1_cross_root_join")
    selected_from_lineage = lineage_freeze.get("selected")
    if (
        not isinstance(selected_from_lineage, Mapping)
        or set(selected_from_lineage) != {"test_q24", "unseen24"}
        or {
            split: list(selected_from_lineage[split])
            if isinstance(selected_from_lineage[split], list)
            else None
            for split in ("test_q24", "unseen24")
        }
        != {
            split: [
                row["scenario_id"]
                for row in wheel_selections
                if row["split"] == split
            ]
            for split in ("test_q24", "unseen24")
        }
    ):
        raise AggregateBlocked("g3_upstream_lineage_invalid")

    if any(not isinstance(row, Mapping) for row in rows):
        raise AggregateBlocked("g3_row_schema_invalid")
    wheel_raw = [
        row for row in rows if row.get("row_kind") == "g3_wheel_step"
    ]
    interface_raw = [
        row for row in rows if row.get("row_kind") == "g3_interface_replay"
    ]
    if len(wheel_raw) + len(interface_raw) != len(rows):
        raise AggregateBlocked("g3_row_kind_mismatch")
    if len(interface_raw) != 6:
        raise AggregateBlocked("g3_interface_replay_count_mismatch")

    sequence_precheck: dict[str, list[dict[str, Any]]] = {}
    for raw in wheel_raw:
        row = _exact_keys(raw, _G3_WHEEL_ROW_KEYS, "g3_wheel_row_schema")
        episode_id = _nonempty_string(
            row["episode_id"],
            "g3_step_sequence",
        )
        _exact_nonnegative_int(row["step_index"], "g3_step_sequence")
        _exact_bool(row["is_terminal"], "g3_terminal_position")
        sequence_precheck.setdefault(episode_id, []).append(row)
    if set(sequence_precheck) != set(selection_by_episode):
        raise AggregateBlocked("g3_g1_cross_root_join")
    for episode_rows in sequence_precheck.values():
        ordered = sorted(episode_rows, key=lambda row: int(row["step_index"]))
        if [row["step_index"] for row in ordered] != list(range(len(ordered))):
            raise AggregateBlocked("g3_step_sequence")
        terminals = [
            row for row in ordered if row["is_terminal"] is True
        ]
        if len(terminals) != 1 or terminals[0] is not ordered[-1]:
            raise AggregateBlocked("g3_terminal_position")

    episodes: dict[str, list[dict[str, Any]]] = {}
    seen_steps: set[str] = set()
    wheel_timings: list[float] = []
    for raw in wheel_raw:
        row = _exact_keys(raw, _G3_WHEEL_ROW_KEYS, "g3_wheel_row_schema")
        if (
            row["schema_version"] != "mid-dual-g3-wheel-step/v1"
            or row["scale_profile"] != SCALE_PROFILE
            or row["run_id"] != config["run_id"]
            or row["execution_class"] != "formal"
            or row["formal_sample"] is not True
            or row["timing_contract_id"] != _TIMING_CONTRACT_ID
            or row["wheel_platform"] != "wheel"
            or row["wheel_profile"] != "ppo-standard-wheel-grid/v1"
            or row["wheel_capability_revision"]
            != "ppo-path-planner-adapter/v1"
            or row["checkpoint_sha256"] != _UPDATE80_CHECKPOINT_SHA256
            or row["policy_state_sha256"]
            != _UPDATE80_POLICY_STATE_SHA256
            or row["config_sha256"] != config["config_sha256"]
            or row["input_sha256"] != config["input_sha256"]
            or row["code_sha256"] != config["code_sha256"]
            or row["source_sha256"] != config["code_sha256"]
            or row["frozen_manifest_sha256"]
            != upstream["freeze_manifest_sha256"]
            or row["g1_source_manifest_sha256"] != g1_manifest_sha256
            or row["g2_source_manifest_sha256"] != g2_manifest_sha256
        ):
            raise AggregateBlocked("g3_wheel_row_schema")
        for identity_field in (
            "decision_record_id",
            "observation_record_id",
        ):
            _nonempty_string(row[identity_field], "g3_wheel_authority")
        if not _is_sha256(row["post_observation_record_sha256"]):
            raise AggregateBlocked("g3_wheel_authority")
        episode_id = _nonempty_string(
            row["episode_id"],
            "g3_g1_cross_root_join",
        )
        selection = selection_by_episode.get(episode_id)
        g1 = g1_index.get((str(row["split"]), str(row["scenario_id"])))
        if (
            selection is None
            or g1 is None
            or row["split"] != selection["split"]
            or row["scenario_id"] != selection["scenario_id"]
            or row["episode_index"] != selection["g1_episode_index"]
            or row["lane_id"] != selection["lane_id"]
        ):
            raise AggregateBlocked("g3_g1_cross_root_join")
        step_id = _nonempty_string(row["step_id"], "g3_step_sequence")
        if step_id in seen_steps:
            raise AggregateBlocked("g3_step_sequence")
        seen_steps.add(step_id)
        step_index = _exact_nonnegative_int(
            row["step_index"],
            "g3_step_sequence",
        )
        _exact_bool(row["is_terminal"], "g3_terminal_position")
        _nonempty_string(row["termination_reason"], "g3_terminal_position")
        for field_name in (
            "pre_snapshot_sha256",
            "decision_sha256",
            "candidate_sha256",
            "request_candidate_sha256",
            "request_sha256",
            "route_request_sha256",
            "route_result_sha256",
            "feedback_route_result_sha256",
            "feedback_sha256",
            "post_snapshot_parent_sha256",
            "post_snapshot_sha256",
        ):
            if not _is_sha256(row[field_name]):
                raise AggregateBlocked("g3_wheel_hash_chain")
        expected_request_id = _g3_request_id(
            scenario_id=str(row["scenario_id"]),
            step_index=step_index,
            pre_snapshot_sha256=str(row["pre_snapshot_sha256"]),
        )
        request_id = _nonempty_string(
            row["request_id"],
            "g3_wheel_hash_chain",
        )
        if request_id != expected_request_id:
            raise AggregateBlocked("g3_wheel_hash_chain")
        if row["candidate_sha256"] != _g3_candidate_sha256(row):
            raise AggregateBlocked("g3_wheel_hash_chain")
        if row["request_sha256"] != _g3_planner_request_sha256(row):
            raise AggregateBlocked("g3_wheel_hash_chain")
        if row["route_result_sha256"] != _g3_route_result_sha256(row):
            raise AggregateBlocked("g3_wheel_hash_chain")
        if row["feedback_sha256"] != _g3_feedback_sha256(row):
            raise AggregateBlocked("g3_wheel_hash_chain")
        if (
            row["request_candidate_sha256"] != row["candidate_sha256"]
            or row["route_request_id"] != request_id
            or row["route_request_sha256"] != row["request_sha256"]
            or row["feedback_request_id"] != request_id
            or row["feedback_route_result_sha256"]
            != row["route_result_sha256"]
            or row["post_snapshot_parent_sha256"] != row["feedback_sha256"]
        ):
            raise AggregateBlocked("g3_wheel_hash_chain")
        selected_cell = _g3_cell(
            row["selected_candidate_cell_xy"],
            "g3_wheel_endpoint_mismatch",
        )
        if (
            _g3_cell(
                row["route_endpoint_cell_xy"],
                "g3_wheel_endpoint_mismatch",
            )
            != selected_cell
            or _g3_cell(
                row["feedback_pose_cell_xy"],
                "g3_wheel_endpoint_mismatch",
            )
            != selected_cell
            or not math.isclose(
                _g3_theta(
                    row["selected_theta"],
                    "g3_wheel_endpoint_mismatch",
                ),
                _g3_theta(
                    row["route_endpoint_theta"],
                    "g3_wheel_endpoint_mismatch",
                ),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                _g3_theta(
                    row["selected_theta"],
                    "g3_wheel_endpoint_mismatch",
                ),
                _g3_theta(
                    row["feedback_theta"],
                    "g3_wheel_endpoint_mismatch",
                ),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise AggregateBlocked("g3_wheel_endpoint_mismatch")
        if (
            _exact_bool(row["planner_success"], "g3_planner_failure")
            is not True
            or row["planner_failure_reason"] != "none"
            or _exact_nonnegative_int(
                row["safety_violation_count"],
                "g3_wheel_integrity",
            )
            != 0
            or _exact_nonnegative_int(
                row["masked_action_count"],
                "g3_wheel_integrity",
            )
            != 0
            or _exact_nonnegative_int(
                row["candidate_route_mismatch_count"],
                "g3_wheel_integrity",
            )
            != 0
        ):
            raise AggregateBlocked("g3_wheel_integrity")
        coverage = _finite_number(row["coverage"], "g3_wheel_coverage")
        paired = _finite_number(
            row["paired_g1_coverage"],
            "g3_g1_cross_root_join",
        )
        if (
            not 0.0 <= coverage <= 1.0
            or not 0.0 <= paired <= 1.0
            or not math.isclose(
                paired,
                float(g1["coverage"]),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        ):
            raise AggregateBlocked("g3_g1_cross_root_join")
        wheel_timings.append(_g3_timing_ms(row))
        episodes.setdefault(episode_id, []).append(row)
    if set(episodes) != set(selection_by_episode):
        raise AggregateBlocked("g3_g1_cross_root_join")

    terminal_rows: list[dict[str, Any]] = []
    for episode_rows in episodes.values():
        ordered = sorted(episode_rows, key=lambda row: int(row["step_index"]))
        if [row["step_index"] for row in ordered] != list(range(len(ordered))):
            raise AggregateBlocked("g3_step_sequence")
        terminals = [
            row for row in ordered if row["is_terminal"] is True
        ]
        if len(terminals) != 1 or terminals[0] is not ordered[-1]:
            raise AggregateBlocked("g3_terminal_position")
        for left, right in zip(ordered[:-1], ordered[1:], strict=True):
            if left["post_snapshot_sha256"] != right["pre_snapshot_sha256"]:
                raise AggregateBlocked("g3_step_sequence")
        terminal_rows.append(ordered[-1])

    g2_calls: dict[str, dict[str, Any]] = {}
    for raw in g2_rows:
        row = _exact_keys(raw, _G2_ROW_KEYS, "g3_g2_cross_root_join")
        call_id = _nonempty_string(row["call_id"], "g3_g2_cross_root_join")
        if call_id in g2_calls:
            raise AggregateBlocked("g3_g2_cross_root_join")
        g2_calls[call_id] = row

    selections = audit["interface_selections"]
    if not isinstance(selections, list) or len(selections) != 6:
        raise AggregateBlocked("g3_g2_cross_root_join")
    interface_selection_keys = {
        "platform",
        "replay_id",
        "g2_reference_call_ids",
        "request_id",
        "request_sha256",
        "g2_semantic_digest",
        "timing_contract_id",
    }
    selection_by_replay: dict[str, dict[str, Any]] = {}
    for raw in selections:
        selection = _exact_keys(
            raw,
            interface_selection_keys,
            "g3_g2_cross_root_join",
        )
        replay_id = _nonempty_string(
            selection["replay_id"],
            "g3_g2_cross_root_join",
        )
        if replay_id in selection_by_replay:
            raise AggregateBlocked("g3_g2_cross_root_join")
        selection_by_replay[replay_id] = selection

    interface_timings: list[float] = []
    replay_ids: set[str] = set()
    request_keys: set[tuple[str, str, str]] = set()
    platform_counts = {"legged": 0, "hopper": 0}
    for raw in interface_raw:
        row = _exact_keys(
            raw,
            _G3_INTERFACE_ROW_KEYS,
            "g3_interface_row_schema",
        )
        if (
            row["schema_version"] != "mid-dual-g3-interface-replay/v1"
            or row["scale_profile"] != SCALE_PROFILE
            or row["run_id"] != config["run_id"]
            or row["execution_class"] != "formal"
            or row["formal_sample"] is not True
            or row["config_sha256"] != config["config_sha256"]
            or row["input_sha256"] != config["input_sha256"]
            or row["code_sha256"] != config["code_sha256"]
            or row["source_sha256"] != config["code_sha256"]
            or row["g1_source_manifest_sha256"] != g1_manifest_sha256
            or row["g2_source_manifest_sha256"] != g2_manifest_sha256
            or row["approval_sha256"] != upstream["g2_approval_sha256"]
            or row["cohort_sha256"] != upstream["g2_cohort_sha256"]
            or row["provider_identity_sha256"]
            != upstream["g2_provider_identity_sha256"]
            or row["oracle_identity_sha256"]
            != upstream["g2_oracle_identity_sha256"]
            or row["hopper_resolution_sha256"]
            != upstream["g2_hopper_resolution_sha256"]
        ):
            raise AggregateBlocked("g3_interface_row_schema")
        replay_id = _nonempty_string(
            row["replay_id"],
            "g3_interface_request_unique",
        )
        platform_name = row["platform"]
        request_id = _nonempty_string(
            row["request_id"],
            "g3_interface_request_unique",
        )
        request_sha = row["request_sha256"]
        if platform_name not in platform_counts or not _is_sha256(request_sha):
            raise AggregateBlocked("g3_interface_row_schema")
        request_key = (str(platform_name), request_id, str(request_sha))
        if replay_id in replay_ids or request_key in request_keys:
            raise AggregateBlocked("g3_interface_request_unique")
        replay_ids.add(replay_id)
        request_keys.add(request_key)
        selection = selection_by_replay.get(replay_id)
        if (
            selection is None
            or selection["platform"] != platform_name
            or selection["request_id"] != request_id
            or selection["request_sha256"] != request_sha
        ):
            raise AggregateBlocked("g3_g2_cross_root_join")
        references = row["g2_reference_call_ids"]
        if (
            not isinstance(references, list)
            or len(references) != G2_REPEATS
            or len(set(references)) != G2_REPEATS
            or references != selection["g2_reference_call_ids"]
        ):
            raise AggregateBlocked("g3_g2_cross_root_join")
        reference_rows = [g2_calls.get(str(call_id)) for call_id in references]
        if any(reference is None for reference in reference_rows):
            raise AggregateBlocked("g3_g2_cross_root_join")
        typed_references = [
            reference for reference in reference_rows if reference is not None
        ]
        semantics = {
            str(reference["semantic_digest"])
            for reference in typed_references
        }
        if (
            {reference["platform"] for reference in typed_references}
            != {platform_name}
            or {reference["request_id"] for reference in typed_references}
            != {request_id}
            or {
                reference["request_sha256"]
                for reference in typed_references
            }
            != {request_sha}
            or {reference["truth_sha256"] for reference in typed_references}
            != {row["truth_request_sha256"]}
            or {
                reference["provider_result_sha256"]
                for reference in typed_references
            }
            != {row["provider_result_sha256"]}
            or {
                reference["repeat_index"]
                for reference in typed_references
            }
            != set(range(G2_REPEATS))
            or len(semantics) != 1
            or any(
                reference["provider_success"] is not True
                or reference["route_l2_valid"] is not True
                or reference["timing_contract_id"] != _TIMING_CONTRACT_ID
                for reference in typed_references
            )
        ):
            raise AggregateBlocked("g3_g2_cross_root_join")
        semantic = next(iter(semantics))
        if (
            row["provider_request_sha256"] != request_sha
            or row["g2_semantic_digest"] != semantic
            or row["replay_semantic_digest"] != semantic
            or selection["g2_semantic_digest"] != semantic
            or row["timing_contract_id"] != _TIMING_CONTRACT_ID
            or selection["timing_contract_id"] != _TIMING_CONTRACT_ID
            or _exact_bool(
                row["formal_input_eligible"],
                "g3_interface_formal_input",
            )
            is not True
        ):
            raise AggregateBlocked("g3_g2_cross_root_join")
        interface_timings.append(_g3_timing_ms(row))
        platform_counts[str(platform_name)] += 1
    if (
        replay_ids != set(selection_by_replay)
        or platform_counts != {"legged": 3, "hopper": 3}
    ):
        raise AggregateBlocked("g3_g2_cross_root_join")

    final_coverages = [
        float(row["coverage"]) for row in terminal_rows
    ]
    paired_coverages = [
        float(row["paired_g1_coverage"]) for row in terminal_rows
    ]
    coverage_mean = statistics.mean(final_coverages)
    paired_deltas = [
        actual - reference
        for actual, reference in zip(
            final_coverages,
            paired_coverages,
            strict=True,
        )
    ]
    paired_delta_mean = statistics.mean(paired_deltas)
    wheel_timing = _timing_statistics(wheel_timings)
    interface_timing = _timing_statistics(interface_timings)
    coverage_mid = (
        coverage_mean >= MID_COVERAGE_THRESHOLD
        and all(value >= MID_COVERAGE_THRESHOLD for value in final_coverages)
    )
    coverage_final = (
        coverage_mean >= FINAL_COVERAGE_THRESHOLD
        and all(value >= FINAL_COVERAGE_THRESHOLD for value in final_coverages)
    )
    shared = paired_delta_mean >= -0.01
    midterm = (
        shared
        and coverage_mid
        and wheel_timing["midterm_reduced_passed"] is True
        and interface_timing["midterm_reduced_passed"] is True
    )
    final = (
        shared
        and coverage_final
        and wheel_timing["final_threshold_reduced_passed"] is True
        and interface_timing["final_threshold_reduced_passed"] is True
    )
    wheel_coverage_80_count = sum(
        value >= MID_COVERAGE_THRESHOLD for value in final_coverages
    )
    return {
        "status": _status_for(midterm),
        "scale_profile": SCALE_PROFILE,
        "active_platforms": list(_G2_ACTIVE_PLATFORMS),
        "platform_invariants": _G2_PLATFORM_INVARIANTS,
        "full_scale_acceptance": False,
        "thresholds": {
            "midterm_coverage": MID_COVERAGE_THRESHOLD,
            "final_coverage": FINAL_COVERAGE_THRESHOLD,
            "paired_g1_delta_min": -0.01,
            "midterm_planner_ms": MID_TIME_MS,
            "final_planner_ms": FINAL_TIME_MS,
            "final_proportion_at_or_below_1000": 0.95,
            "absolute_max_planner_ms": MID_TIME_MS,
        },
        "g1_source_manifest_sha256": g1_manifest_sha256,
        "g2_source_manifest_sha256": g2_manifest_sha256,
        "upstream_lineage_audit_sha256": upstream_lineage_audit_sha256,
        "wheel_episode_count": len(episodes),
        "wheel_step_count": len(wheel_raw),
        "interface_replay_count": len(interface_raw),
        "split_episode_counts": selection_split_counts,
        "g3_midterm_crosscheck_passed": midterm,
        "g3_final_crosscheck_passed": final,
        "wheel_coverage_mean": coverage_mean,
        "wheel_coverage_80_count": wheel_coverage_80_count,
        "wheel_coverage_99_count": sum(
            value >= FINAL_COVERAGE_THRESHOLD for value in final_coverages
        ),
        "paired_g1_coverage_delta_mean": paired_delta_mean,
        "paired_g1_coverage_delta_sample_stddev": (
            statistics.stdev(paired_deltas)
            if len(paired_deltas) > 1
            else 0.0
        ),
        "wheel_timing": wheel_timing,
        "interface_timing": interface_timing,
        "wheel_coverage_all_episodes_passed": (
            wheel_coverage_80_count == 10
        ),
        "wheel_coverage_mean_passed": (
            coverage_mean >= MID_COVERAGE_THRESHOLD
        ),
        "paired_g1_delta_passed": paired_delta_mean >= -0.01,
        "wheel_timing_midterm_passed": wheel_timing[
            "midterm_reduced_passed"
        ],
        "wheel_timing_final_passed": wheel_timing[
            "final_threshold_reduced_passed"
        ],
        "interface_timing_midterm_passed": interface_timing[
            "midterm_reduced_passed"
        ],
        "interface_timing_final_passed": interface_timing[
            "final_threshold_reduced_passed"
        ],
        "wheel_integrity_passed": True,
        "interface_correctness_passed": True,
        "required_replay_identities_passed": True,
    }


def _stored_summary_matches(
    gate_id: str,
    recomputed: Mapping[str, Any],
    stored: Mapping[str, Any],
    *,
    run_id: str,
    config: Mapping[str, Any] | None = None,
) -> bool:
    expected = {
        "schema_version": f"xunce-mid-dual-{gate_id}-canonical-summary/v1",
        "scale_profile": SCALE_PROFILE,
        "gate_id": gate_id,
        "run_id": run_id,
        "status": recomputed.get("status"),
        "formal_evidence_eligible": True,
        "recomputed": dict(recomputed),
    }
    if gate_id == "g2":
        if config is None:
            return False
        expected.update(
            {
                **_validate_g2_platform_invariants(
                    config.get("active_platforms"),
                    config.get("platform_invariants"),
                ),
                "path_planner_runtime_source_closure_sha256": (
                    config.get(
                        "path_planner_runtime_source_closure_sha256"
                    )
                ),
            }
        )
    elif gate_id == "g3":
        expected.update(
            {
                **_validate_g2_platform_invariants(
                    recomputed.get("active_platforms"),
                    recomputed.get("platform_invariants"),
                    "g3_platform_slope_invariant",
                ),
                "full_scale_acceptance": False,
            }
        )
    return dict(stored) == expected


def _report_audit_matches(
    *,
    gate_id: str,
    source: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> bool:
    audit = source["report_audit"]
    if not isinstance(audit, Mapping):
        return False
    expected_summary = {
        "schema_version": (
            f"xunce-mid-dual-{gate_id}-canonical-summary/v1"
        ),
        "scale_profile": SCALE_PROFILE,
        "gate_id": gate_id,
        "run_id": source["run_id"],
        "status": recomputed.get("status"),
        "formal_evidence_eligible": True,
        "recomputed": dict(recomputed),
    }
    if gate_id == "g2":
        expected_summary.update(
            {
                **_validate_g2_platform_invariants(
                    source["config"].get("active_platforms"),
                    source["config"].get("platform_invariants"),
                ),
                "path_planner_runtime_source_closure_sha256": (
                    source["config"].get(
                        "path_planner_runtime_source_closure_sha256"
                    )
                ),
            }
        )
    elif gate_id == "g3":
        expected_summary.update(
            {
                **_validate_g2_platform_invariants(
                    recomputed.get("active_platforms"),
                    recomputed.get("platform_invariants"),
                    "g3_platform_slope_invariant",
                ),
                "full_scale_acceptance": False,
            }
        )
    if gate_id in {"g2", "g3"}:
        producer = _import_local_script(
            {
                "g2": "run_xunce_mid_dual_g2_planning_time",
                "g3": "run_xunce_mid_dual_g3_closed_loop",
            }[gate_id]
        )
        report = (
            producer._render_report(expected_summary)  # noqa: SLF001
            if gate_id == "g2"
            else producer._render_g3_report(  # noqa: SLF001
                expected_summary
            )
        )
        expected_audit = (
            producer._source_report_audit(  # noqa: SLF001
                summary=expected_summary,
                report=report,
            )
            if gate_id == "g2"
            else producer._g3_report_audit(  # noqa: SLF001
                summary=expected_summary,
                report=report,
            )
        )
        return (
            source["summary_bytes"]
            == _artifact_text_bytes(
                _canonical_json_artifact_text(expected_summary)
            )
            and source["report_bytes"] == _artifact_text_bytes(report)
            and dict(audit) == expected_audit
        )
    expected = {
        "schema_version": contract["report_audit_schema_version"],
        "gate_id": gate_id,
        "run_id": source["run_id"],
        "renderer_id": contract["report_renderer_id"],
        "summary_projection_sha256": _canonical_json_sha256(recomputed),
        "stored_summary_bytes_sha256": _sha256_bytes(
            source["summary_bytes"]
        ),
        "report_bytes_sha256": _sha256_bytes(source["report_bytes"]),
        "formal_evidence_eligible": True,
    }
    return dict(audit) == expected


def aggregate_completed_roots(
    g1_root: str | Path,
    g2_root: str | Path,
    g3_root: str | Path,
    *,
    source_contracts: Mapping[str, Mapping[str, Any]] | None = None,
    g1_assessment_envelope: str | Path | None = None,
    assessment_module: object | None = None,
    _verified_sources_out: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify all source manifests, then independently recalculate the three gates."""
    if source_contracts is None:
        raise AggregateBlocked("source_contracts_required")
    if set(source_contracts) != set(_SOURCE_GATES):
        return blocked_summary(["source_contracts_invalid"])

    gates: dict[str, dict[str, Any]] = {}
    source_manifests: dict[str, str] = {}
    blockers: list[str] = []
    roots = {"g1": g1_root, "g2": g2_root, "g3": g3_root}
    sources: dict[str, dict[str, Any]] = {}
    for gate_id in _SOURCE_GATES:
        try:
            source = _load_verified_source(
                roots[gate_id],
                gate_id,
                source_contracts[gate_id],
                g1_assessment_envelope=(
                    g1_assessment_envelope if gate_id == "g1" else None
                ),
                assessment_module=(
                    assessment_module if gate_id == "g1" else None
                ),
            )
            sources[gate_id] = source
            source_manifests[gate_id] = source["manifest_sha256"]
        except AggregateBlocked as exc:
            blockers.append(str(exc))
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            blockers.append(
                f"{gate_id}_malformed_evidence:{type(exc).__name__}"
            )
    if _verified_sources_out is not None:
        _verified_sources_out.update(sources)

    for gate_id in ("g1", "g2"):
        source = sources.get(gate_id)
        if source is None:
            gates[gate_id] = {
                "status": "blocked",
                "midterm_reduced_passed": False,
                "final_threshold_reduced_passed": False,
            }
            continue
        try:
            recomputed = (
                dict(source["metric_projection"])
                if gate_id == "g1"
                and source.get("source_kind") == "g1_native"
                else recompute_g1(
                    source["rows"], source["config"], source["input_audit"]
                )
                if gate_id == "g1"
                else recompute_g2(
                    source["rows"],
                    source["config"],
                    source["input_audit"],
                )
            )
            gates[gate_id] = recomputed
            native_g1 = (
                gate_id == "g1"
                and source.get("source_kind") == "g1_native"
            )
            assessment_g1 = (
                gate_id == "g1"
                and source.get("source_kind")
                == "g1_existing_run_assessment"
            )
            if (
                assessment_g1
                and recomputed != source.get("metric_projection")
            ):
                blockers.append(
                    "g1_existing_run_assessment_review_mismatch"
                )
            if not native_g1 and not assessment_g1 and not _stored_summary_matches(
                gate_id,
                recomputed,
                source["stored_summary"],
                run_id=source["run_id"],
                config=source["config"],
            ):
                blockers.append(f"{gate_id}_stored_summary_mismatch")
            if not native_g1 and not assessment_g1 and not _report_audit_matches(
                gate_id=gate_id,
                source=source,
                recomputed=recomputed,
                contract=source_contracts[gate_id],
            ):
                blockers.append(f"{gate_id}_report_audit_mismatch")
        except AggregateBlocked as exc:
            blockers.append(str(exc))
            gates.setdefault(
                gate_id,
                {
                    "status": "blocked",
                    "midterm_reduced_passed": False,
                    "final_threshold_reduced_passed": False,
                },
            )
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            blockers.append(
                f"{gate_id}_malformed_evidence:{type(exc).__name__}"
            )
            gates.setdefault(
                gate_id,
                {
                    "status": "blocked",
                    "midterm_reduced_passed": False,
                    "final_threshold_reduced_passed": False,
                },
            )

    g3_source = sources.get("g3")
    if (
        g3_source is None
        or "g1" not in sources
        or "g2" not in sources
        or gates.get("g1", {}).get("status") == "blocked"
        or gates.get("g2", {}).get("status") == "blocked"
    ):
        gates["g3"] = {
            "status": "blocked",
            "midterm_reduced_passed": False,
            "final_threshold_reduced_passed": False,
        }
        if g3_source is not None:
            blockers.append("g3_upstream_source_unavailable")
    else:
        try:
            recomputed_g3 = recompute_g3(
                g3_source["rows"],
                g3_source["config"],
                g3_source["input_audit"],
                g1_rows=sources["g1"]["rows"],
                g2_rows=sources["g2"]["rows"],
                g1_manifest_sha256=sources["g1"]["manifest_sha256"],
                g2_manifest_sha256=sources["g2"]["manifest_sha256"],
                g1_config=sources["g1"]["config"],
                g1_input_audit=sources["g1"]["input_audit"],
                g1_native_summary=sources["g1"].get(
                    "native_summary"
                ),
                g1_trace_row_count=sources["g1"]["trace_row_count"],
                g1_evidence_binding=sources["g1"][
                    "assessment_binding"
                ],
                g1_review_metrics=sources["g1"].get("review_metrics"),
                g1_platform_invariants=sources["g1"][
                    "platform_invariants"
                ],
                g1_platform_invariants_source_sha256=sources["g1"][
                    "platform_invariants_source_sha256"
                ],
                g2_config=sources["g2"]["config"],
                g2_input_audit=sources["g2"]["input_audit"],
                upstream_lineage=g3_source["upstream_lineage"],
                upstream_lineage_audit_sha256=g3_source[
                    "upstream_lineage_audit_sha256"
                ],
            )
            gates["g3"] = recomputed_g3
            if not _stored_summary_matches(
                "g3",
                recomputed_g3,
                g3_source["stored_summary"],
                run_id=g3_source["run_id"],
                config=g3_source["config"],
            ):
                blockers.append("g3_stored_summary_mismatch")
            if not _report_audit_matches(
                gate_id="g3",
                source=g3_source,
                recomputed=recomputed_g3,
                contract=source_contracts["g3"],
            ):
                blockers.append("g3_report_audit_mismatch")
        except AggregateBlocked as exc:
            blockers.append(str(exc))
            gates["g3"] = {
                "status": "blocked",
                "midterm_reduced_passed": False,
                "final_threshold_reduced_passed": False,
            }
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            blockers.append(f"g3_malformed_evidence:{type(exc).__name__}")
            gates["g3"] = {
                "status": "blocked",
                "midterm_reduced_passed": False,
                "final_threshold_reduced_passed": False,
            }

    sample_counts = {
        "g1_total_episodes": int(gates.get("g1", {}).get("sample_count", 0)),
        "g1_test_q24": int(
            gates.get("g1", {}).get("splits", {}).get("test_q24", {}).get("sample_count", 0)
        ),
        "g1_unseen24": int(
            gates.get("g1", {}).get("splits", {}).get("unseen24", {}).get("sample_count", 0)
        ),
        "g2_formal_calls": int(gates.get("g2", {}).get("formal_call_count", 0)),
        "g3_wheel_episodes": int(gates.get("g3", {}).get("wheel_episode_count", 0)),
        "g3_interface_replays": int(
            gates.get("g3", {}).get("interface_replay_count", 0)
        ),
    }
    if blockers:
        return {
            **blocked_summary(blockers),
            "gates": gates,
            "sample_counts": sample_counts,
            "source_manifest_sha256": source_manifests,
        }

    midterm = midterm_reduced_truth_table(
        gates["g1"]["g1_coverage_80_passed"],
        gates["g2"]["g2_all_platforms_2s_passed"],
        gates["g3"]["g3_midterm_crosscheck_passed"],
    )
    final = final_threshold_reduced_truth_table(
        gates["g1"]["g1_coverage_99_passed"],
        gates["g2"]["g2_all_platforms_1s_passed"],
        gates["g3"]["g3_final_crosscheck_passed"],
    )
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "scale_profile": SCALE_PROFILE,
        "status": _status_for(midterm),
        "formal_evidence_eligible": True,
        "midterm_reduced_gate_passed": midterm,
        "final_threshold_reduced_gate_passed": final,
        "blockers": [],
        "gates": gates,
        "sample_counts": sample_counts,
        "source_manifest_sha256": source_manifests,
    }


def render_report(summary: Mapping[str, Any]) -> str:
    """Render the bounded one-page reduced-scale conclusion table."""
    counts = summary.get("sample_counts", {})
    gates = summary.get("gates", {})
    status = summary.get("status")
    if status == "blocked":
        conclusion = "证据未就绪；以下阻塞项关闭前不得描述为指标失败或通过。"
    elif status == "passed":
        conclusion = "缩减规模中期双门槛通过。"
    else:
        conclusion = "完整证据已复算，指标未通过。"
    blockers = summary.get("blockers", [])
    blocker_text = "、".join(str(item) for item in blockers) if blockers else "无"
    lines = [
        "# 缩减规模中期实验（G1 24 场景/split）独立复算报告",
        "",
        f"- 规模合同：`{SCALE_PROFILE}`",
        f"- 结论：{conclusion}",
        f"- 阻塞项：{blocker_text}",
        "",
        "| 证据组 | 精确样本数 |",
        "|---|---:|",
        f"| G1 Test-Q24 | {counts.get('g1_test_q24', 0)} |",
        f"| G1 Unseen-24 | {counts.get('g1_unseen24', 0)} |",
        f"| G2 正式调用 | {counts.get('g2_formal_calls', 0)} |",
        f"| G3 轮式闭环 | {counts.get('g3_wheel_episodes', 0)} |",
        f"| G3 接口回放 | {counts.get('g3_interface_replays', 0)} |",
    ]
    g1_splits = gates.get("g1", {}).get("splits", {})
    if g1_splits:
        lines.extend(("", "| G1 split | mean | >=80% | >=99% |", "|---|---:|---:|---:|"))
        for split, label in (("test_q24", "G1 Test-Q24"), ("unseen24", "G1 Unseen-24")):
            row = g1_splits.get(split, {})
            lines.append(
                f"| {label} | {float(row.get('mean', 0.0)):.6f} | "
                f"{row.get('coverage_80_count', 0)}/24 | "
                f"{row.get('coverage_99_count', 0)}/24 |"
            )
    g2_groups = gates.get("g2", {}).get("timing_by_platform_scale", {})
    if g2_groups:
        lines.extend(
            (
                "",
                "| G2 platform/scale | n | mean ms | P95 ms | P99 ms | 2 s | 1 s |",
                "|---|---:|---:|---:|---:|---|---|",
            )
        )
        for platform in G2_PLATFORMS:
            for scale in ("standard", "kilometer"):
                key = f"{platform}/{scale}"
                row = g2_groups.get(key, {})
                lines.append(
                    f"| {key} | {row.get('sample_count', 0)} | "
                    f"{float(row.get('mean_ms', 0.0)):.3f} | "
                    f"{float(row.get('p95_ms', 0.0)):.3f} | "
                    f"{float(row.get('p99_ms', 0.0)):.3f} | "
                    f"{bool(row.get('midterm_reduced_passed'))} | "
                    f"{bool(row.get('final_threshold_reduced_passed'))} |"
                )
    g3 = gates.get("g3", {})
    if g3:
        lines.extend(
            (
                "",
                "| G3 cross-check | value |",
                "|---|---:|",
                f"| G3 轮式覆盖均值 | {float(g3.get('wheel_coverage_mean', 0.0)):.6f} |",
                f"| G3 轮式规划 P95 ms | {float(g3.get('wheel_timing', {}).get('p95_ms', 0.0)):.3f} |",
                f"| G3 G1 配对覆盖差均值 | {float(g3.get('paired_g1_coverage_delta_mean', 0.0)):.6f} |",
                f"| G3 接口对应正确 | {bool(g3.get('interface_correctness_passed'))} |",
            )
        )
    lines.extend(
        (
            "",
            "| 路由 | 结果 |",
            "|---|---|",
            f"| 中期缩减门槛（80% / 2 s） | {bool(summary.get('midterm_reduced_gate_passed'))} |",
            f"| 最终阈值缩减门槛（99% / 1 s） | {bool(summary.get('final_threshold_reduced_gate_passed'))} |",
            "",
            "本报告仅代表上述缩减规模样本，不等同于项目全规模完成验收。",
            "",
        )
    )
    return "\n".join(lines)


def _environment_probe() -> dict[str, Any]:
    memory_bytes = 1
    try:
        import psutil

        memory_bytes = int(psutil.virtual_memory().total)
    except (ImportError, OSError, ValueError):
        pass
    gpu_model = gpu_driver = gpu_cuda = "unavailable"
    try:
        probe = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            gpu_model, gpu_driver = (
                item.strip() for item in probe.stdout.splitlines()[0].split(",", 1)
            )
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    try:
        import torch

        gpu_cuda = str(torch.version.cuda or "unavailable")
    except ImportError:
        pass
    thread_names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "windows_version": platform.platform() or "unavailable",
        "cpu_model": platform.processor() or platform.machine() or "unavailable",
        "cpu_logical_count": int(os.cpu_count() or 1),
        "memory_bytes": max(1, memory_bytes),
        "gpu": {"model": gpu_model, "driver": gpu_driver, "cuda": gpu_cuda},
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "frozen_dependencies": [
            f"python=={platform.python_version()}",
            f"pytest-environment={os.environ.get('PYTEST_VERSION', 'not-running')}",
        ],
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unset"),
        "thread_variables": {name: os.environ.get(name, "unset") for name in thread_names},
        "worker_start_method": "spawn",
        "power_mode": os.environ.get("XUNCE_POWER_MODE", "not-recorded"),
    }


def _git_output(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=cwd or Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _load_aggregate_config(path: str | Path) -> dict[str, Any]:
    config = artifact_io.read_json(path)
    expected_keys = {
        "schema_version",
        "scale_profile",
        "output_root",
        "required_phase_ids",
        "actual_sample_counts",
        "reference_sample_counts",
        "source_contracts",
    }
    if set(config) != expected_keys:
        raise ValueError("aggregate config exact schema drift")
    if config.get("schema_version") != AGGREGATE_CONFIG_SCHEMA_VERSION:
        raise ValueError("aggregate config schema_version is unsupported")
    if config.get("scale_profile") != SCALE_PROFILE:
        raise ValueError("aggregate config scale_profile drift")
    output_root = config.get("output_root")
    if not isinstance(output_root, str) or not Path(output_root).is_absolute():
        raise ValueError("aggregate output_root must be absolute")
    if output_root.replace("\\", "/") != _FORMAL_AGGREGATE_BASE:
        raise ValueError(
            f"aggregate output_root must be {_FORMAL_AGGREGATE_BASE}"
        )
    if config.get("required_phase_ids") != ["p01"]:
        raise ValueError("aggregate required_phase_ids must be ['p01']")
    expected_actual = {
        "g1_episodes_per_split": 24,
        "g1_split_count": 2,
        "g2_formal_calls": 645,
        "g3_interface_replays": 6,
        "g3_wheel_episodes": 10,
    }
    if config.get("actual_sample_counts") != expected_actual:
        raise ValueError("aggregate actual_sample_counts drift")
    expected_reference = {
        "g1_episodes_per_split": 64,
        "g1_split_count": 2,
        "g2_formal_calls": 1950,
        "g3_interface_replays": 18,
        "g3_wheel_episodes": 30,
    }
    if config.get("reference_sample_counts") != expected_reference:
        raise ValueError("aggregate reference_sample_counts drift")
    contracts = config.get("source_contracts")
    if not isinstance(contracts, Mapping) or set(contracts) != set(
        _SOURCE_GATES
    ):
        raise ValueError("aggregate source_contracts drift")
    for gate_id in _SOURCE_GATES:
        contract = contracts[gate_id]
        if (
            not isinstance(contract, Mapping)
            or contract.get("schema_version")
            != "xunce-mid-dual-source-contract/v1"
            or contract.get("gate_id") != gate_id
            or contract.get("output_base")
            != f"D:/xunce/out/mid_dual/{gate_id}"
        ):
            raise ValueError(f"aggregate {gate_id} source_contract drift")
    return config


def _validate_run_id(run_id: object) -> str:
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError("run_id is invalid")
    if run_id.upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError("run_id is a Windows reserved name")
    return run_id


def _effective_config_sha256(config: Mapping[str, Any]) -> str:
    payload = dict(config)
    supplied = payload.pop("config_sha256", None)
    digest = _canonical_json_sha256(payload)
    if supplied is not None and supplied != digest:
        raise ValueError("effective config_sha256 mismatch")
    return digest


def _open_aggregate_store(
    run_root: str | Path,
    effective_config: Mapping[str, Any],
    *,
    mode: str,
) -> MidDualRunStore:
    root = Path(run_root)
    if mode == "create":
        return MidDualRunStore.create_new(root, effective_config)
    if mode != "resume":
        raise ValueError("mode must be create or resume")
    if artifact_io.path_is_file(artifact_path(root, MID_DUAL_MANIFEST)):
        raise ValueError("aggregate run is already finalized")
    expected_sha256 = _effective_config_sha256(effective_config)
    try:
        return MidDualRunStore.load_for_resume(root, expected_sha256)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError("resume_config_mismatch") from exc


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = getattr(
            os.stat(path, follow_symlinks=False),
            "st_file_attributes",
            0,
        )
    except OSError as exc:
        raise ValueError(f"source root is not readable: {path}") from exc
    return bool(
        path.is_symlink()
        or attributes
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _normalized_absolute(path: str | Path) -> str:
    return os.path.normcase(
        os.path.normpath(str(Path(path).resolve()))
    )


def _validate_formal_roots(
    *,
    roots: Mapping[str, str | Path],
    source_contracts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    normalized: dict[str, str] = {}
    for gate_id in _SOURCE_GATES:
        root = Path(roots[gate_id])
        if not root.is_absolute():
            raise ValueError(f"{gate_id}_root must be absolute")
        if not artifact_io.path_exists(root):
            raise ValueError(f"{gate_id}_root is missing")
        expected_parent = Path(source_contracts[gate_id]["output_base"])
        if _normalized_absolute(root.parent) != _normalized_absolute(
            expected_parent
        ):
            raise ValueError(
                f"{gate_id}_root must be a direct child of "
                f"{source_contracts[gate_id]['output_base']}"
            )
        if _is_reparse_point(root):
            raise ValueError(f"{gate_id}_root must not be a reparse point")
        resolved[gate_id] = root
        normalized[gate_id] = _normalized_absolute(root)
    if len(set(normalized.values())) != len(_SOURCE_GATES):
        raise ValueError("source roots must be distinct")
    for left_id, left in normalized.items():
        for right_id, right in normalized.items():
            if left_id == right_id:
                continue
            try:
                common = os.path.commonpath((left, right))
            except ValueError:
                continue
            if common in {left, right}:
                raise ValueError("source roots must not be nested")
    return resolved


def run_aggregate(
    *,
    config_path: str | Path,
    g1_root: str | Path,
    g1_assessment_envelope: str | Path | None = None,
    g2_root: str | Path,
    g3_root: str | Path,
    run_id: str,
    mode: str = "create",
) -> Path:
    """Run one non-overwriting aggregate attempt and write canonical artifacts."""
    run_id = _validate_run_id(run_id)
    config = _load_aggregate_config(config_path)
    roots = _validate_formal_roots(
        roots={"g1": g1_root, "g2": g2_root, "g3": g3_root},
        source_contracts=config["source_contracts"],
    )
    verified_sources: dict[str, dict[str, Any]] = {}
    summary = aggregate_completed_roots(
        roots["g1"],
        roots["g2"],
        roots["g3"],
        source_contracts=config["source_contracts"],
        g1_assessment_envelope=g1_assessment_envelope,
        _verified_sources_out=verified_sources,
    )
    assessment_use = verified_sources.get("g1", {}).get(
        "g1_existing_run_assessment"
    )
    output_root = Path(config["output_root"]) / run_id
    effective_config = {
        **config,
        "run_id": run_id,
        "g1_root": str(roots["g1"]).replace("\\", "/"),
        "g2_root": str(roots["g2"]).replace("\\", "/"),
        "g3_root": str(roots["g3"]).replace("\\", "/"),
        "g1_existing_run_assessment": assessment_use,
    }
    store = _open_aggregate_store(
        output_root,
        effective_config,
        mode=mode,
    )
    source_rows = [
        {
            "row_kind": "independent_gate_recalculation",
            "gate_id": gate_id,
            "recalculated": summary.get("gates", {}).get(gate_id, {"status": "blocked"}),
        }
        for gate_id in _SOURCE_GATES
    ]
    if summary["status"] != "blocked" and "p01" not in store.accepted_phase_ids:
        attempt = store.write_phase_attempt(
            "p01", source_rows, {"kind": "independent_aggregate"}
        )
        store.accept_phase(
            "p01", attempt, store.phase_attempt_row_sha256("p01", attempt)
        )
    repository_root = Path(_git_output("rev-parse", "--show-toplevel") or Path.cwd())
    required_sources = [
        Path(config_path),
        Path(__file__),
        repository_root / "scripts" / "xunce_mid_dual_contracts.py",
        repository_root / "scripts" / "xunce_mid_dual_artifacts.py",
        repository_root
        / "scripts"
        / "xunce_mid_dual_g1_existing_run_assessment.py",
        repository_root / "scripts" / "xunce_artifact_io.py",
        repository_root / "scripts" / "xunce_artifact_paths.py",
    ]
    store.capture_lineage(
        required_sources,
        _git_output("rev-parse", "HEAD"),
        _git_output("-C", "path-planner", "rev-parse", "HEAD"),
    )
    store.capture_environment(_environment_probe)
    routing = {
        "schema_version": "xunce-mid-dual-aggregate-routing/v1",
        "scale_profile": SCALE_PROFILE,
        "status": summary["status"],
        "midterm_reduced_gate_passed": summary["midterm_reduced_gate_passed"],
        "final_threshold_reduced_gate_passed": summary[
            "final_threshold_reduced_gate_passed"
        ],
        "blockers": summary["blockers"],
    }
    store.finalize(
        summary,
        routing,
        render_report(summary),
        {
            "source_roots": {
                "g1_root": str(Path(g1_root)),
                "g2_root": str(Path(g2_root)),
                "g3_root": str(Path(g3_root)),
                "source_manifest_sha256": summary.get("source_manifest_sha256", {}),
                "g1_existing_run_assessment": assessment_use,
            }
        },
    )
    MidDualRunStore.verify_manifest(output_root)
    return output_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently recalculate the reduced midterm dual-gate."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--g1-root", required=True)
    parser.add_argument("--g1-assessment-envelope")
    parser.add_argument("--g2-root", required=True)
    parser.add_argument("--g3-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("create", "resume"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = run_aggregate(
        config_path=args.config,
        g1_root=args.g1_root,
        g1_assessment_envelope=args.g1_assessment_envelope,
        g2_root=args.g2_root,
        g3_root=args.g3_root,
        run_id=args.run_id,
        mode=args.mode,
    )
    summary = artifact_io.read_json(
        artifact_path(output_root, MID_DUAL_SUMMARY)
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "execution_status": "complete",
                "gate_status": summary["status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
