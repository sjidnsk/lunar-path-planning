"""中期双门槛 G3 闭环计时与跨门强联接 runner。

本模块把 G3 的判定保持为纯 raw-row 复算：轮式闭环行必须逐步形成
candidate -> request -> route -> feedback -> next-state 哈希链，并与 G1 的
整数覆盖分母结果强联接；Legged/Hopper 回放必须与 G2 的五次重复结果强联接。
任何输入、联接或计时合同缺口均 fail closed 为 ``blocked``。
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
from statistics import mean, median, stdev
import struct
import subprocess
import sys
from typing import Any, Mapping, Sequence

import xunce_artifact_io as artifact_io
from xunce_artifact_paths import (
    MID_DUAL_MANIFEST,
    MID_DUAL_PHASE_STATE,
    artifact_path,
)
from xunce_mid_dual_artifacts import MidDualRunStore


SCALE_PROFILE = "midterm_reduced_w8x3_update80/v1"
RUNNER_ID = "run_xunce_mid_dual_g3_closed_loop/v1"
UPDATE80_CHECKPOINT_SHA256 = (
    "35e04c86f9f973af028fb08f0175d42ab45378d09e1f2b96ee6aad6e4c12b5b5"
)
UPDATE80_POLICY_STATE_SHA256 = (
    "3123e6adde99be41e3bd5cc2f3843068e5416892395926d759c2c6a243ffd381"
)
MID_COVERAGE_THRESHOLD = 0.80
FINAL_COVERAGE_THRESHOLD = 0.99
MID_TIME_MS = 2000.0
FINAL_TIME_MS = 1000.0
PAIRED_G1_DELTA_MIN = -0.01
TIMING_CONTRACT_ID = "five-phase-sequential-ns/v1"
MAX_TRAVERSABLE_SLOPE_DEG = 30.0
ACTIVE_PLATFORMS = ("wheel", "legged", "hopper")
PLATFORM_INVARIANTS = {
    platform: {
        "max_traversable_slope_deg": MAX_TRAVERSABLE_SLOPE_DEG,
    }
    for platform in ACTIVE_PLATFORMS
}
G1_NATIVE_EVIDENCE_KIND = "native_completed_run_v1"
G1_ASSESSMENT_EVIDENCE_KIND = "existing_run_assessment_v1"
G3_OUTPUT_BASE = "D:/xunce/out/mid_dual/g3"
G3_FROZEN_BASE = "D:/xunce/inputs/mid_dual/scenarios"
G1_OUTPUT_BASE = "D:/xunce/out/mid_dual/g1"
G2_OUTPUT_BASE = "D:/xunce/out/mid_dual/g2"
G2_EXECUTION_INPUT_BASE = "D:/xunce/inputs/mid_dual/g2"
G3_EFFECTIVE_CONFIG_SCHEMA_VERSION = "xunce-mid-dual-g3-effective-config/v1"
G3_INPUT_AUDIT_SCHEMA_VERSION = "xunce-mid-dual-g3-input-audit/v1"
G3_SOURCE_CONTRACT_SCHEMA_VERSION = "xunce-mid-dual-source-contract/v1"
G3_REQUIRED_PHASE_IDS = ("p01", "p02")
G3_PHASE_NAMES = {
    "p01": "wheel_closed_loop",
    "p02": "interface_replay",
}
WHEEL_PLATFORM = "wheel"
WHEEL_PROFILE = "ppo-standard-wheel-grid/v1"
WHEEL_CAPABILITY_REVISION = "ppo-path-planner-adapter/v1"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CANONICAL_CONFIG_PATH = (
    _REPO_ROOT / "configs" / "xunce_mid_dual_g3_closed_loop_v1.json"
)
G3_REQUIRED_SOURCE_RELATIVE_PATHS = (
    "configs/xunce_mid_dual_g3_closed_loop_v1.json",
    "scripts/run_xunce_mid_dual_g3_closed_loop.py",
    "scripts/freeze_xunce_mid_dual_scenarios.py",
    "scripts/run_xunce_mid_dual_g1_coverage.py",
    "scripts/xunce_mid_dual_g1_existing_run_assessment.py",
    "scripts/run_xunce_mid_dual_g2_planning_time.py",
    "scripts/xunce_mid_dual_g2_inputs.py",
    "scripts/xunce_mid_dual_contracts.py",
    "scripts/xunce_mid_dual_artifacts.py",
    "scripts/xunce_artifact_io.py",
    "scripts/xunce_artifact_paths.py",
    "src/lunar_exploration_ppo/eval/midterm_reduced.py",
    "src/lunar_exploration_ppo/eval/standard.py",
    "src/lunar_exploration_ppo/env/standard_training.py",
    "src/lunar_exploration_ppo/integrations/path_planner_adapter.py",
)
G3_SOURCE_CONTRACT = {
    "schema_version": G3_SOURCE_CONTRACT_SCHEMA_VERSION,
    "gate_id": "g3",
    "config_schema_version": G3_EFFECTIVE_CONFIG_SCHEMA_VERSION,
    "runner_id": RUNNER_ID,
    "required_phase_ids": list(G3_REQUIRED_PHASE_IDS),
    "output_base": G3_OUTPUT_BASE,
    "active_platforms": list(ACTIVE_PLATFORMS),
    "platform_invariants": PLATFORM_INVARIANTS,
    "input_audit_schema_version": G3_INPUT_AUDIT_SCHEMA_VERSION,
    "report_audit_schema_version": "xunce-mid-dual-source-report-audit/v1",
    "report_renderer_id": "xunce-mid-dual-g3-canonical-report/v1",
    "required_lineage_sources": list(G3_REQUIRED_SOURCE_RELATIVE_PATHS),
}
TIMING_COMPONENT_FIELDS = (
    "input_validation_ns",
    "platform_instantiation_ns",
    "search_ns",
    "complete_route_validation_ns",
    "result_assembly_ns",
)
_G3_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "gate_id",
        "runner_id",
        "scale_profile",
        "output_base",
        "active_platforms",
        "platform_invariants",
        "required_phase_ids",
        "required_phases",
        "phase_contract",
        "formal_counts",
        "thresholds",
        "checkpoint_sha256",
        "policy_state_sha256",
        "timing_contract_id",
        "wheel_identity",
    }
)

_G1_ROW_FIELDS = frozenset(
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
        "episode_index",
        "episode_id",
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
_WHEEL_ROW_FIELDS = frozenset(
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
        *TIMING_COMPONENT_FIELDS,
        "total_ns",
    }
)
_INTERFACE_ROW_FIELDS = frozenset(
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
        *TIMING_COMPONENT_FIELDS,
        "total_ns",
    }
)
_FORMAL_WHEEL_EXTRA_FIELDS = frozenset(
    {
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
_FORMAL_INTERFACE_EXTRA_FIELDS = frozenset(
    {
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
_G2_REFERENCE_FIELDS = frozenset(
    {
        "row_kind",
        "platform",
        "request_id",
        "request_sha256",
        "call_id",
        "repeat_index",
        "semantic_digest",
        "provider_success",
        "route_l2_valid",
        "timing_contract_id",
    }
)


class G3Blocked(ValueError):
    """G3 证据不能形成正式判定。"""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def validate_platform_invariants(
    active_platforms: object,
    platform_invariants: object,
    *,
    expected_platforms: Sequence[str] = ACTIVE_PLATFORMS,
) -> dict[str, object]:
    """验证每个 active platform 都严格绑定 30°，不接受诊断放宽。"""

    expected = tuple(expected_platforms)
    if (
        not isinstance(active_platforms, (list, tuple))
        or tuple(active_platforms) != expected
        or not isinstance(platform_invariants, Mapping)
        or set(platform_invariants) != set(expected)
    ):
        raise G3Blocked("g3_platform_slope_invariant")
    copied: dict[str, dict[str, float]] = {}
    for platform in expected:
        invariant = platform_invariants.get(platform)
        if (
            not isinstance(invariant, Mapping)
            or set(invariant) != {"max_traversable_slope_deg"}
            or type(invariant.get("max_traversable_slope_deg")) is not float
            or invariant.get("max_traversable_slope_deg")
            != MAX_TRAVERSABLE_SLOPE_DEG
        ):
            raise G3Blocked("g3_platform_slope_invariant")
        copied[platform] = {
            "max_traversable_slope_deg": MAX_TRAVERSABLE_SLOPE_DEG,
        }
    return {
        "active_platforms": list(expected),
        "platform_invariants": copied,
    }


def validate_g3_config_payload(
    payload: object,
) -> dict[str, object]:
    """验证仓库内 G3 canonical config，不接受运行时放宽或 override。"""

    if not isinstance(payload, Mapping) or set(payload) != set(
        _G3_CONFIG_FIELDS
    ):
        raise G3Blocked("g3_config_invalid")
    validate_platform_invariants(
        payload.get("active_platforms"),
        payload.get("platform_invariants"),
    )
    expected = {
        "schema_version": "mid-dual-g3-config/v1",
        "gate_id": "g3",
        "runner_id": RUNNER_ID,
        "scale_profile": SCALE_PROFILE,
        "output_base": "D:/xunce/out/mid_dual/g3",
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": PLATFORM_INVARIANTS,
        "required_phase_ids": ["p01", "p02"],
        "required_phases": ["wheel_closed_loop", "interface_replay"],
        "phase_contract": {
            "p01": {
                "name": "wheel_closed_loop",
                "row_kind": "g3_wheel_step",
                "episodes": 10,
                "formal_only": True,
            },
            "p02": {
                "name": "interface_replay",
                "row_kind": "g3_interface_replay",
                "platform_counts": {"legged": 3, "hopper": 3},
                "formal_only": True,
                "requires": "p01",
            },
        },
        "formal_counts": {
            "wheel_episodes": 10,
            "test_q24": 5,
            "unseen24": 5,
            "legged_replays": 3,
            "hopper_replays": 3,
        },
        "thresholds": {
            "midterm_coverage": MID_COVERAGE_THRESHOLD,
            "final_coverage": FINAL_COVERAGE_THRESHOLD,
            "paired_g1_delta_min": PAIRED_G1_DELTA_MIN,
            "midterm_planner_ms": MID_TIME_MS,
            "final_planner_ms": FINAL_TIME_MS,
            "absolute_max_planner_ms": MID_TIME_MS,
        },
        "checkpoint_sha256": UPDATE80_CHECKPOINT_SHA256,
        "policy_state_sha256": UPDATE80_POLICY_STATE_SHA256,
        "timing_contract_id": TIMING_CONTRACT_ID,
        "wheel_identity": {
            "platform": WHEEL_PLATFORM,
            "profile": WHEEL_PROFILE,
            "capability_revision": WHEEL_CAPABILITY_REVISION,
        },
    }
    copied = json.loads(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    if copied != expected:
        raise G3Blocked("g3_config_invalid")
    return copied


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, reason: str) -> str:
    if not _is_sha256(value):
        raise G3Blocked(reason)
    return str(value)


def _require_nonempty(value: object, reason: str) -> str:
    if not isinstance(value, str) or not value:
        raise G3Blocked(reason)
    return value


def _require_exact_int(value: object, reason: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise G3Blocked(reason)
    return value


def _require_exact_bool(value: object, reason: str) -> bool:
    if type(value) is not bool:
        raise G3Blocked(reason)
    return value


def _finite_number(
    value: object,
    reason: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise G3Blocked(reason)
    number = float(value)
    if minimum is not None and number < minimum:
        raise G3Blocked(reason)
    if maximum is not None and number > maximum:
        raise G3Blocked(reason)
    return number


def _require_exact_keys(
    row: object,
    expected: frozenset[str],
    reason: str,
) -> Mapping[str, object]:
    if not isinstance(row, Mapping) or set(row) != set(expected):
        raise G3Blocked(reason)
    return row


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise G3Blocked("g3_timing_contract")
    index = max(1, math.ceil(quantile * len(values))) - 1
    return sorted(values)[index]


def _timing_statistics(values: Sequence[float]) -> dict[str, object]:
    rows = tuple(values)
    if not rows:
        raise G3Blocked("g3_timing_contract")
    p95 = _nearest_rank(rows, 0.95)
    p99 = _nearest_rank(rows, 0.99)
    average = mean(rows)
    count_le_1000ms = sum(value <= FINAL_TIME_MS for value in rows)
    return {
        "sample_count": len(rows),
        "mean_ms": average,
        "median_ms": median(rows),
        "sample_stddev_ms": stdev(rows) if len(rows) > 1 else 0.0,
        "min_ms": min(rows),
        "max_ms": max(rows),
        "p95_ms": p95,
        "p99_ms": p99,
        "count_le_1000ms": count_le_1000ms,
        "at_or_below_1000_fraction": count_le_1000ms / len(rows),
        "count_le_2000ms": sum(value <= MID_TIME_MS for value in rows),
        "midterm_reduced_passed": (
            average <= MID_TIME_MS
            and p95 <= MID_TIME_MS
            and max(rows) <= MID_TIME_MS
        ),
        "final_threshold_reduced_passed": (
            average <= FINAL_TIME_MS
            and p95 <= FINAL_TIME_MS
            and count_le_1000ms >= math.ceil(0.95 * len(rows))
            and max(rows) <= MID_TIME_MS
        ),
    }


def _timing_ms(row: Mapping[str, object]) -> float:
    components = tuple(
        _require_exact_int(row.get(field), "g3_timing_contract")
        for field in TIMING_COMPONENT_FIELDS
    )
    total = _require_exact_int(row.get("total_ns"), "g3_timing_contract")
    if total != sum(components):
        raise G3Blocked("g3_timing_contract")
    return total / 1_000_000.0


def deterministic_request_id(
    *,
    scenario_id: str,
    step_index: int,
    pre_snapshot_sha256: str,
) -> str:
    """从冻结三元组域分隔生成 G3 planner request ID。"""

    scenario = _require_nonempty(scenario_id, "g3_request_id")
    step = _require_exact_int(step_index, "g3_request_id")
    snapshot = _require_sha256(pre_snapshot_sha256, "g3_request_id")
    encoded = (
        "mid-dual-g3-request/v1\0"
        f"{scenario}\0{step}\0{snapshot}"
    ).encode("utf-8")
    return f"g3-wheel-{hashlib.sha256(encoded).hexdigest()}"


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    try:
        encoded = json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise G3Blocked("g3_wheel_hash_chain") from exc
    return hashlib.sha256(encoded).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise G3Blocked("g3_noncanonical_json") from exc


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_artifact_bytes(path: str | Path) -> bytes:
    reader = getattr(artifact_io, "read_bytes")
    return reader(path)


def _import_local_script(name: str) -> Any:
    scripts_path = str(Path(__file__).resolve().parent)
    added = scripts_path not in sys.path
    if added:
        sys.path.insert(0, scripts_path)
    try:
        return importlib.import_module(name)
    finally:
        if added:
            if not sys.path or sys.path[0] != scripts_path:
                raise G3Blocked("g3_script_import_path_drift")
            sys.path.pop(0)


def _json_artifact_sha256(value: Mapping[str, object]) -> str:
    payload = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return _bytes_sha256(payload)


def _domain_hash(domain: str, *parts: bytes) -> str:
    framed = bytearray()
    for part in (domain.encode("utf-8"), *parts):
        framed.extend(struct.pack(">Q", len(part)))
        framed.extend(part)
    return _bytes_sha256(bytes(framed))


def validate_g3_run_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or _RUN_ID_RE.fullmatch(value) is None
        or value in {".", ".."}
        or ".." in value
        or value.endswith((".", " "))
        or "/" in value
        or "\\" in value
        or value.split(".", 1)[0].upper() in _WINDOWS_RESERVED
    ):
        raise G3Blocked("g3_run_id_invalid")
    return value


def _normalized_windows_path(value: str | Path, reason: str) -> tuple[Path, str]:
    raw = str(value)
    windows = PureWindowsPath(raw)
    if (
        not windows.is_absolute()
        or windows.drive.upper() != "D:"
        or any(part in {"", ".", ".."} for part in windows.parts[1:])
    ):
        raise G3Blocked(reason)
    path = Path(raw).resolve()
    normalized = str(path).replace("\\", "/").rstrip("/").casefold()
    return path, normalized


def _direct_child_of(
    value: str | Path,
    base: str,
    reason: str,
    *,
    require_existing: bool,
) -> Path:
    path, normalized = _normalized_windows_path(value, reason)
    base_path, base_normalized = _normalized_windows_path(base, reason)
    parent = str(path.parent.resolve()).replace("\\", "/").rstrip("/").casefold()
    if parent != base_normalized or normalized == base_normalized:
        raise G3Blocked(reason)
    if require_existing and not artifact_io.path_is_dir(path):
        raise G3Blocked(reason)
    if require_existing:
        try:
            file_attributes = getattr(path.stat(), "st_file_attributes", 0)
        except OSError as exc:
            raise G3Blocked(reason) from exc
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if path.is_symlink() or bool(file_attributes & reparse_flag):
            raise G3Blocked(reason)
    if str(base_path).replace("\\", "/").rstrip("/").casefold() != base_normalized:
        raise G3Blocked(reason)
    return path


def validate_g3_root_contract(
    *,
    frozen_bundle_root: str | Path,
    g1_root: str | Path,
    g2_root: str | Path,
    run_id: object,
    require_existing: bool = True,
) -> dict[str, Path]:
    validated_run_id = validate_g3_run_id(run_id)
    frozen = _direct_child_of(
        frozen_bundle_root,
        G3_FROZEN_BASE,
        "g3_frozen_bundle_root_invalid",
        require_existing=require_existing,
    )
    g1 = _direct_child_of(
        g1_root,
        G1_OUTPUT_BASE,
        "g3_g1_root_invalid",
        require_existing=require_existing,
    )
    g2 = _direct_child_of(
        g2_root,
        G2_OUTPUT_BASE,
        "g3_g2_root_invalid",
        require_existing=require_existing,
    )
    output = Path(G3_OUTPUT_BASE) / validated_run_id
    normalized = {
        str(item.resolve()).replace("\\", "/").rstrip("/").casefold()
        for item in (frozen, g1, g2, output)
    }
    if len(normalized) != 4:
        raise G3Blocked("g3_input_roots_not_disjoint")
    values = tuple(normalized)
    if any(
        left != right
        and (left.startswith(f"{right}/") or right.startswith(f"{left}/"))
        for left in values
        for right in values
    ):
        raise G3Blocked("g3_input_roots_not_disjoint")
    return {
        "frozen_bundle_root": frozen,
        "g1_root": g1,
        "g2_root": g2,
        "output_root": output,
    }


def _g3_required_source_paths() -> tuple[Path, ...]:
    paths = tuple(_REPO_ROOT / item for item in G3_REQUIRED_SOURCE_RELATIVE_PATHS)
    if any(not artifact_io.path_is_file(path) for path in paths):
        raise G3Blocked("g3_required_source_missing")
    return paths


def compute_g3_code_lineage() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for relative, path in zip(
        G3_REQUIRED_SOURCE_RELATIVE_PATHS,
        _g3_required_source_paths(),
        strict=True,
    ):
        payload = _read_artifact_bytes(path)
        rows.append(
            {
                "logical_path": relative,
                "size_bytes": len(payload),
                "sha256": _bytes_sha256(payload),
            }
        )
    return {
        "schema_version": "xunce-mid-dual-code/v1",
        "required_sources": rows,
        "code_sha256": _bytes_sha256(
            b"xunce-mid-dual-code/v1\0" + _canonical_bytes(rows)
        ),
    }


def _validate_g1_evidence_binding(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise G3Blocked("g3_g1_evidence_binding_invalid")
    copied = dict(value)
    kind = copied.get("g1_evidence_kind")
    if kind == G1_NATIVE_EVIDENCE_KIND:
        if set(copied) != {
            "g1_evidence_kind",
            "g1_source_manifest_sha256",
        }:
            raise G3Blocked("g3_g1_evidence_binding_invalid")
        _require_sha256(
            copied["g1_source_manifest_sha256"],
            "g3_g1_evidence_binding_invalid",
        )
        return copied
    if kind != G1_ASSESSMENT_EVIDENCE_KIND:
        raise G3Blocked("g3_g1_evidence_binding_invalid")
    required_hashes = {
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
        *required_hashes,
        "assessment_use_status",
        "strict_prestart_lineage_status",
        "lineage_limitation_acknowledged",
        "numerical_gate_status",
        "numerical_result_status",
    }
    if set(copied) != expected:
        raise G3Blocked("g3_g1_evidence_binding_invalid")
    for field in required_hashes:
        _require_sha256(copied[field], "g3_g1_evidence_binding_invalid")
    if (
        copied["g1_source_manifest_sha256"]
        != copied["assessment_manifest_sha256"]
        or copied["assessment_use_status"] != "authorized_existing_run"
        or copied["strict_prestart_lineage_status"] != "not_satisfied"
        or copied["lineage_limitation_acknowledged"] is not True
        or copied["numerical_gate_status"] != "recomputed_from_raw_rows"
        or copied["numerical_result_status"] not in {"passed", "failed"}
    ):
        raise G3Blocked("g3_g1_evidence_binding_invalid")
    return copied


def _g1_source_evidence_binding(
    source: Mapping[str, object],
) -> dict[str, object]:
    binding = _validate_g1_evidence_binding(source.get("assessment_binding"))
    if (
        binding["g1_source_manifest_sha256"]
        != source.get("manifest_sha256")
        or binding["g1_evidence_kind"] != source.get("g1_evidence_kind")
    ):
        raise G3Blocked("g3_g1_evidence_binding_invalid")
    return binding


def _validate_upstream_binding(value: object) -> dict[str, object]:
    required_hashes = {
        "g1_source_manifest_sha256",
        "g2_source_manifest_sha256",
        "freeze_manifest_sha256",
        "g1_config_sha256",
        "g1_input_sha256",
        "g1_code_sha256",
        "g2_config_sha256",
        "g2_input_sha256",
        "g2_code_sha256",
        "g2_approval_sha256",
        "g2_cohort_sha256",
        "g2_provider_identity_sha256",
        "g2_oracle_identity_sha256",
        "g2_hopper_resolution_sha256",
    }
    expected = {
        *required_hashes,
        "g2_input_set_id",
        "active_platforms",
        "platform_invariants",
        "g1_evidence_binding",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise G3Blocked("g3_upstream_binding_invalid")
    copied = dict(value)
    invariant_projection = validate_platform_invariants(
        copied["active_platforms"],
        copied["platform_invariants"],
    )
    copied.update(invariant_projection)
    copied["g1_evidence_binding"] = _validate_g1_evidence_binding(
        copied["g1_evidence_binding"]
    )
    for field in required_hashes:
        _require_sha256(copied[field], "g3_upstream_binding_invalid")
    _require_nonempty(copied["g2_input_set_id"], "g3_upstream_binding_invalid")
    if (
        copied["g2_provider_identity_sha256"]
        == copied["g2_oracle_identity_sha256"]
        or copied["g1_evidence_binding"]["g1_source_manifest_sha256"]
        != copied["g1_source_manifest_sha256"]
    ):
        raise G3Blocked("g3_provider_oracle_identity_not_separated")
    return copied


def build_g3_effective_config(
    *,
    base_config: Mapping[str, object],
    run_id: object,
    input_audit: Mapping[str, object],
    code_lineage: Mapping[str, object],
    upstream_binding: Mapping[str, object],
) -> dict[str, object]:
    validate_g3_config_payload(base_config)
    validated_run_id = validate_g3_run_id(run_id)
    upstream = _validate_upstream_binding(upstream_binding)
    input_invariants = validate_platform_invariants(
        input_audit.get("active_platforms"),
        input_audit.get("platform_invariants"),
    )
    if (
        input_audit.get("schema_version") != G3_INPUT_AUDIT_SCHEMA_VERSION
        or input_audit.get("gate_id") != "g3"
        or input_audit.get("run_id") != validated_run_id
        or input_audit.get("formal_evidence_eligible") is not True
        or _validate_g1_evidence_binding(
            input_audit.get("g1_evidence_binding")
        )
        != upstream["g1_evidence_binding"]
        or code_lineage.get("schema_version") != "xunce-mid-dual-code/v1"
        or not _is_sha256(code_lineage.get("code_sha256"))
    ):
        raise G3Blocked("g3_effective_config_input_invalid")
    return {
        "schema_version": G3_EFFECTIVE_CONFIG_SCHEMA_VERSION,
        "gate_id": "g3",
        "runner_id": RUNNER_ID,
        "run_id": validated_run_id,
        "output_root": G3_OUTPUT_BASE,
        "scale_profile": SCALE_PROFILE,
        **input_invariants,
        "input_sha256": _json_artifact_sha256(input_audit),
        "code_sha256": str(code_lineage["code_sha256"]),
        "required_phase_ids": list(G3_REQUIRED_PHASE_IDS),
        "source_contract_sha256": _canonical_sha256(G3_SOURCE_CONTRACT),
        "evidence_binding": {
            "schema_version": "xunce-mid-dual-g3-evidence-binding/v1",
            "input_audit_path": "g3_input_audit.json",
            "lineage_audit_path": "lineage_audit.json",
            "report_audit_path": "g3_report_audit.json",
            "platform_invariants_audit_path": (
                "platform_invariants_audit.json"
            ),
        },
        "g1_evidence_binding": dict(upstream["g1_evidence_binding"]),
        "upstream_binding": upstream,
    }


def validate_g1_native_effective_config_projection(
    payload: object,
    *,
    expected_root: Path,
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise G3Blocked("g3_g1_native_config_invalid")
    copied = dict(payload)
    expected_keys = {
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
    checkpoint = copied.get("checkpoint")
    scenario = copied.get("scenario_manifest")
    input_audit = copied.get("input_audit")
    if (
        set(copied) != expected_keys
        or copied.get("schema_version")
        != "xunce-mid-dual-g1-effective-config/v1"
        or copied.get("gate_id") != "g1"
        or copied.get("runner_id") != "run_xunce_mid_dual_g1_coverage/v1"
        or copied.get("mode") != "formal"
        or copied.get("output_root")
        != str(expected_root).replace("\\", "/")
        or copied.get("required_phase_ids")
        != ["p01", "p02", "p03", "p04", "p05", "p06", "p07"]
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("update") != 80
        or checkpoint.get("sha256") != UPDATE80_CHECKPOINT_SHA256
        or checkpoint.get("policy_state_sha256")
        != UPDATE80_POLICY_STATE_SHA256
        or not isinstance(scenario, Mapping)
        or not _is_sha256(scenario.get("sha256"))
        or not isinstance(input_audit, Mapping)
        or input_audit.get("path") != "g1_input_audit.json"
        or input_audit.get("schema_version")
        != "xunce-mid-dual-g1-input-audit/v1"
        or input_audit.get("sha256") != copied.get("input_sha256")
        or not _is_sha256(copied.get("input_sha256"))
        or not _is_sha256(copied.get("code_sha256"))
    ):
        raise G3Blocked("g3_g1_native_config_invalid")
    return copied


def _parse_json_bytes(payload: bytes, reason: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite:{token}")
            ),
        )
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise G3Blocked(reason) from exc
    if not isinstance(value, dict):
        raise G3Blocked(reason)
    return value


def _parse_jsonl_bytes(payload: bytes, reason: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        for raw_line in payload.decode("utf-8").splitlines():
            if not raw_line.strip():
                continue
            value = json.loads(
                raw_line,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"nonfinite:{token}")
                ),
            )
            if not isinstance(value, dict):
                raise ValueError("row-not-object")
            rows.append(value)
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise G3Blocked(reason) from exc
    return rows


def _canonical_report_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _safe_manifest_relative_path(value: object, reason: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise G3Blocked(reason)
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise G3Blocked(reason)
    return value


def _verified_manifest_snapshot(
    root: Path,
    *,
    reason: str,
) -> tuple[dict[str, object], dict[str, bytes], str]:
    try:
        MidDualRunStore.verify_manifest(root)
    except (OSError, ValueError, TypeError) as exc:
        raise G3Blocked(reason) from exc
    manifest_path = artifact_path(root, MID_DUAL_MANIFEST)
    manifest_bytes = _read_artifact_bytes(manifest_path)
    manifest = _parse_json_bytes(manifest_bytes, reason)
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        raise G3Blocked(reason)
    snapshot: dict[str, bytes] = {}
    for raw in entries:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
            raise G3Blocked(reason)
        relative = _safe_manifest_relative_path(raw.get("path"), reason)
        digest = _require_sha256(raw.get("sha256"), reason)
        if relative in snapshot:
            raise G3Blocked(reason)
        payload = _read_artifact_bytes(root / PurePosixPath(relative))
        if _bytes_sha256(payload) != digest:
            raise G3Blocked(reason)
        snapshot[relative] = payload
    required = {
        "config.json",
        "results.jsonl",
        "phase-state.jsonl",
        "summary.json",
        "routing.json",
        "report.md",
        "phase-attempts.jsonl",
        "lineage_audit.json",
        "environment_audit.json",
    }
    if not required.issubset(snapshot):
        raise G3Blocked(reason)
    return manifest, snapshot, _bytes_sha256(manifest_bytes)


def _accepted_phase_snapshot(
    snapshot: Mapping[str, bytes],
    expected_phase_ids: Sequence[str],
    *,
    gate_id: str,
) -> dict[str, dict[str, object]]:
    reason = f"g3_{gate_id}_phase_chain_invalid"
    states = _parse_jsonl_bytes(snapshot["phase-state.jsonl"], reason)
    attempts = _parse_jsonl_bytes(snapshot["phase-attempts.jsonl"], reason)
    if [row.get("phase_id") for row in states] != list(expected_phase_ids):
        raise G3Blocked(reason)
    accepted_by_phase: dict[str, dict[str, object]] = {}
    for phase_id, state in zip(expected_phase_ids, states, strict=True):
        if set(state) != {
            "phase_id",
            "attempt_id",
            "row_sha256",
            "rows_path",
        }:
            raise G3Blocked(reason)
        accepted = [
            row
            for row in attempts
            if row.get("phase_id") == phase_id
            and row.get("status") == "accepted"
        ]
        if len(accepted) != 1:
            raise G3Blocked(reason)
        attempt = accepted[0]
        if (
            state.get("attempt_id") != attempt.get("attempt_id")
            or state.get("row_sha256") != attempt.get("row_sha256")
            or state.get("rows_path") != attempt.get("rows_path")
        ):
            raise G3Blocked(reason)
        rows_path = _safe_manifest_relative_path(state.get("rows_path"), reason)
        audit_path = _safe_manifest_relative_path(attempt.get("audit_path"), reason)
        rows_bytes = snapshot.get(rows_path)
        audit_bytes = snapshot.get(audit_path)
        if (
            rows_bytes is None
            or audit_bytes is None
            or _bytes_sha256(rows_bytes) != state.get("row_sha256")
        ):
            raise G3Blocked(reason)
        accepted_by_phase[phase_id] = {
            "state": dict(state),
            "attempt": dict(attempt),
            "rows": _parse_jsonl_bytes(rows_bytes, reason),
            "audit": _parse_json_bytes(audit_bytes, reason),
            "rows_bytes": rows_bytes,
        }
    concatenated = b"".join(
        accepted_by_phase[phase_id]["rows_bytes"]
        for phase_id in expected_phase_ids
    )
    if snapshot["results.jsonl"] != concatenated:
        raise G3Blocked(f"g3_{gate_id}_results_phase_concat_invalid")
    return accepted_by_phase


def _common_lineage_code_sha256(
    *,
    snapshot: Mapping[str, bytes],
    expected_sources: Sequence[str],
    reason: str,
) -> str:
    audit = _parse_json_bytes(snapshot["lineage_audit.json"], reason)
    rows = audit.get("required_sources")
    if (
        audit.get("schema_version") != "mid-dual-lineage-audit/v1"
        or audit.get("formal_evidence_eligible") is not True
        or not isinstance(rows, list)
        or [row.get("original_relative_path") for row in rows]
        != list(expected_sources)
    ):
        raise G3Blocked(reason)
    logical: list[dict[str, object]] = []
    for relative, row in zip(expected_sources, rows, strict=True):
        if not isinstance(row, Mapping):
            raise G3Blocked(reason)
        size = _require_exact_int(row.get("size_bytes"), reason)
        digest = _require_sha256(row.get("sha256"), reason)
        snapshot_path = row.get("snapshot_path")
        payload = (
            snapshot.get(str(snapshot_path))
            if isinstance(snapshot_path, str)
            else _read_artifact_bytes(_REPO_ROOT / relative)
        )
        if (
            payload is None
            or len(payload) != size
            or _bytes_sha256(payload) != digest
        ):
            raise G3Blocked(reason)
        logical.append(
            {
                "logical_path": relative,
                "size_bytes": size,
                "sha256": digest,
            }
        )
    return _bytes_sha256(
        b"xunce-mid-dual-code/v1\0" + _canonical_bytes(logical)
    )


def load_verified_frozen_bundle(root: Path) -> dict[str, object]:
    freezer = _import_local_script("freeze_xunce_mid_dual_scenarios")
    from lunar_exploration_ppo.eval.midterm_reduced import (
        load_frozen_scenario_manifest,
    )

    try:
        if freezer.verify_frozen_bundle(root) is not True:
            raise ValueError("verify-returned-false")
        before = _read_artifact_bytes(root / "manifest.json")
        raw = _parse_json_bytes(before, "g3_frozen_bundle_invalid")
        selected = validate_g3_manifest(raw)
        frozen = load_frozen_scenario_manifest(
            bundle_root=root,
            expected_manifest_sha256=_bytes_sha256(before),
        )
        after = _read_artifact_bytes(root / "manifest.json")
    except G3Blocked:
        raise
    except Exception as exc:
        raise G3Blocked("g3_frozen_bundle_invalid") from exc
    if before != after:
        raise G3Blocked("g3_frozen_bundle_bytes_drift")
    return {
        "root": root,
        "raw_manifest": raw,
        "frozen_manifest": frozen,
        "manifest_sha256": _bytes_sha256(before),
        "selected": selected,
    }


def _verify_g1_code_lineage(
    config: Mapping[str, object],
    snapshot: Mapping[str, bytes],
) -> dict[str, bytes]:
    g1 = _import_local_script("run_xunce_mid_dual_g1_coverage")

    source_lineage = config.get("source_lineage")
    lineage = _parse_json_bytes(
        snapshot["lineage_audit.json"],
        "g3_g1_lineage_invalid",
    )
    if (
        not isinstance(source_lineage, Mapping)
        or source_lineage.get("schema_version")
        != "xunce-mid-dual-g1-code-lineage/v1"
        or source_lineage.get("hash_algorithm")
        != "sha256-domain-separated-path-length-bytes/v1"
        or lineage.get("formal_evidence_eligible") is not True
    ):
        raise G3Blocked("g3_g1_lineage_invalid")
    config_rows = source_lineage.get("required_sources")
    lineage_rows = lineage.get("required_sources")
    if (
        not isinstance(config_rows, list)
        or not isinstance(lineage_rows, list)
        or [row.get("path") for row in config_rows]
        != list(g1.G1_REQUIRED_SOURCE_RELATIVE_PATHS)
        or [row.get("original_relative_path") for row in lineage_rows]
        != list(g1.G1_REQUIRED_SOURCE_RELATIVE_PATHS)
    ):
        raise G3Blocked("g3_g1_lineage_invalid")
    digest = hashlib.sha256()
    digest.update(b"xunce-mid-dual-g1-code-lineage/v1\0")
    source_payloads: dict[str, bytes] = {}
    for config_row, lineage_row in zip(
        config_rows,
        lineage_rows,
        strict=True,
    ):
        if (
            not isinstance(config_row, Mapping)
            or not isinstance(lineage_row, Mapping)
            or config_row.get("size_bytes") != lineage_row.get("size_bytes")
            or config_row.get("sha256") != lineage_row.get("sha256")
        ):
            raise G3Blocked("g3_g1_lineage_invalid")
        relative = str(config_row["path"])
        snapshot_path = lineage_row.get("snapshot_path")
        payload = (
            snapshot.get(str(snapshot_path))
            if isinstance(snapshot_path, str)
            else _read_artifact_bytes(_REPO_ROOT / relative)
        )
        if (
            payload is None
            or len(payload) != config_row.get("size_bytes")
            or _bytes_sha256(payload) != config_row.get("sha256")
        ):
            raise G3Blocked("g3_g1_lineage_invalid")
        path_bytes = relative.encode("utf-8")
        source_payloads[relative] = payload
        digest.update(len(path_bytes).to_bytes(8, "little"))
        digest.update(path_bytes)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    if digest.hexdigest() != config.get("code_sha256"):
        raise G3Blocked("g3_g1_code_sha256_invalid")
    return source_payloads


def load_verified_g1_root(
    root: Path,
    *,
    frozen_bundle: Mapping[str, object],
) -> dict[str, object]:
    g1 = _import_local_script("run_xunce_mid_dual_g1_coverage")

    manifest, snapshot, manifest_sha256 = _verified_manifest_snapshot(
        root,
        reason="g3_g1_manifest_invalid",
    )
    if manifest.get("formal_evidence_eligible") is not True:
        raise G3Blocked("g3_g1_root_not_formal")
    config = _parse_json_bytes(snapshot["config.json"], "g3_g1_config_invalid")
    validate_g1_native_effective_config_projection(config, expected_root=root)
    config_sha256 = config.get("config_sha256")
    if (
        not _is_sha256(config_sha256)
        or config_sha256 != manifest.get("config_sha256")
        or config.get("repair_lineage") is not None
        or config.get("scenario_manifest", {}).get("sha256")
        != frozen_bundle["manifest_sha256"]
    ):
        raise G3Blocked("g3_g1_config_invalid")
    input_audit = _parse_json_bytes(
        snapshot.get("g1_input_audit.json", b""),
        "g3_g1_input_audit_invalid",
    )
    raw_frozen = frozen_bundle["raw_manifest"]
    frozen = frozen_bundle["frozen_manifest"]
    if not isinstance(raw_frozen, Mapping):
        raise G3Blocked("g3_g1_input_audit_invalid")
    expected_input = g1._build_input_audit(  # noqa: SLF001
        manifest_path=Path(frozen_bundle["root"]) / "manifest.json",
        frozen=frozen,
        raw_manifest=raw_frozen,
        manifest_sha256=str(frozen_bundle["manifest_sha256"]),
    )
    if (
        input_audit != expected_input
        or _canonical_sha256(input_audit) != config.get("input_sha256")
        or config.get("input_audit", {}).get("sha256")
        != config.get("input_sha256")
    ):
        raise G3Blocked("g3_g1_input_audit_invalid")
    source_payloads = _verify_g1_code_lineage(config, snapshot)
    base_config_bytes = source_payloads.get(
        "configs/xunce_mid_dual_g1_coverage_v1.json"
    )
    stage6_config_bytes = source_payloads.get(
        "configs/ppo_highres_frontier_stage6_v1.json"
    )
    if base_config_bytes is None or stage6_config_bytes is None:
        raise G3Blocked("g3_g1_base_config_invalid")
    stage6_config = _parse_json_bytes(
        stage6_config_bytes,
        "g3_g1_platform_slope_invariant",
    )
    stage6_safety = stage6_config.get("safety")
    if not isinstance(stage6_safety, Mapping):
        raise G3Blocked("g3_g1_platform_slope_invariant")
    g1_platform_projection = validate_platform_invariants(
        ["wheel"],
        {
            "wheel": {
                "max_traversable_slope_deg": stage6_safety.get(
                    "max_traversable_slope_deg"
                ),
            }
        },
        expected_platforms=("wheel",),
    )
    try:
        base_config = g1.validate_g1_config_payload(
            _parse_json_bytes(
                base_config_bytes,
                "g3_g1_base_config_invalid",
            )
        )
    except Exception as exc:
        raise G3Blocked("g3_g1_base_config_invalid") from exc
    if (
        config.get("base_config_sha256")
        != _bytes_sha256(base_config_bytes)
        or config.get("checkpoint") != base_config.get("checkpoint")
        or config.get("denominator") != base_config.get("denominator")
        or config.get("execution") != base_config.get("execution")
        or config.get("bootstrap") != base_config.get("bootstrap")
        or config.get("schemas") != base_config.get("schemas")
        or config.get("required_phases") != list(g1.G1_REQUIRED_PHASES)
        or config.get("scenario_manifest", {}).get("path")
        != str(
            Path(frozen_bundle["root"]) / "manifest.json"
        ).replace("\\", "/")
    ):
        raise G3Blocked("g3_g1_base_config_invalid")
    environment = _parse_json_bytes(
        snapshot["environment_audit.json"],
        "g3_g1_environment_invalid",
    )
    if (
        environment.get("schema_version") != "mid-dual-environment-audit/v1"
        or environment.get("status") != "captured"
        or environment.get("formal_evidence_eligible") is not True
    ):
        raise G3Blocked("g3_g1_environment_invalid")
    phases = _accepted_phase_snapshot(
        snapshot,
        g1.G1_REQUIRED_PHASE_IDS,
        gate_id="g1",
    )
    p01 = phases["p01"]["audit"]
    if (
        p01.get("schema_version") != "xunce-mid-dual-g1-phase-audit/v1"
        or p01.get("phase_name") != "preflight"
        or p01.get("status") != "passed"
        or p01.get("scenario_manifest_sha256")
        != frozen_bundle["manifest_sha256"]
        or p01.get("input_sha256") != config.get("input_sha256")
        or p01.get("code_sha256") != config.get("code_sha256")
        or p01.get("checkpoint_sha256") != UPDATE80_CHECKPOINT_SHA256
        or p01.get("policy_state_sha256")
        != UPDATE80_POLICY_STATE_SHA256
    ):
        raise G3Blocked("g3_g1_p01_invalid")
    p02 = phases["p02"]["audit"]
    dry_run = p02.get("dry_run")
    dry_results = (
        dry_run.get("results") if isinstance(dry_run, Mapping) else None
    )
    if (
        phases["p02"]["rows"]
        or p02.get("schema_version")
        != "xunce-mid-dual-g1-phase-audit/v1"
        or p02.get("phase_name") != "validation_dry_run"
        or p02.get("status") != "passed"
        or not isinstance(dry_run, Mapping)
        or dry_run.get("schema_version")
        != "xunce-mid-dual-g1-validation-dry-run-audit/v1"
        or dry_run.get("status") != "passed"
        or dry_run.get("scenario_count") != 3
        or dry_run.get("scenario_ids")
        != input_audit.get("validation3_scenario_ids")
        or not isinstance(dry_results, list)
        or len(dry_results) != 3
        or any(not isinstance(row, Mapping) for row in dry_results)
        or [row.get("scenario_id") for row in dry_results]
        != input_audit.get("validation3_scenario_ids")
        or dry_run.get("trace_sha256")
        != _canonical_sha256(dry_results)
    ):
        raise G3Blocked("g3_g1_p02_invalid")
    coverage_rows: list[dict[str, object]] = []
    trace_ids: set[tuple[str, str, int]] = set()
    for phase_id, split in (("p03", "test_q24"), ("p04", "unseen24")):
        phase_rows = phases[phase_id]["rows"]
        audit = phases[phase_id]["audit"]
        coverage = [
            dict(row) for row in phase_rows if row.get("row_kind") == "coverage_episode"
        ]
        traces = [
            dict(row)
            for row in phase_rows
            if row.get("row_kind") in {"decision", "planner_call"}
        ]
        if (
            len(coverage) != 24
            or len(coverage) + len(traces) != len(phase_rows)
            or audit.get("status") != "complete"
            or audit.get("cohort") != split
            or audit.get("coverage_episode_count") != 24
            or audit.get("decision_count")
            != sum(row.get("row_kind") == "decision" for row in traces)
            or audit.get("planner_call_count")
            != sum(row.get("row_kind") == "planner_call" for row in traces)
        ):
            raise G3Blocked("g3_g1_native_phase_rows_invalid")
        for row in traces:
            key = (
                str(row.get("row_kind")),
                str(row.get("episode_id")),
                _require_exact_int(
                    row.get("step_index"),
                    "g3_g1_native_trace_invalid",
                ),
            )
            if (
                set(row) != set(g1.G1_TRACE_ROW_KEYS)
                or row.get("schema_version")
                not in {
                    g1.G1_DECISION_SCHEMA_VERSION,
                    g1.G1_PLANNER_CALL_SCHEMA_VERSION,
                }
                or row.get("phase_id") != phase_id
                or row.get("phase_name") != split
                or row.get("config_sha256") != config_sha256
                or row.get("input_sha256") != config.get("input_sha256")
                or row.get("code_sha256") != config.get("code_sha256")
                or row.get("scenario_manifest_sha256")
                != frozen_bundle["manifest_sha256"]
                or not isinstance(row.get("trace"), Mapping)
                or key in trace_ids
            ):
                raise G3Blocked("g3_g1_native_trace_invalid")
            _canonical_bytes(row)
            trace_ids.add(key)
        coverage_rows.extend(coverage)
    jobs = {
        (
            str(row["split"]),
            str(row["scenario_id"]),
            int(row["episode_index"]),
        ): row
        for row in input_audit["formal_jobs"]
    }
    if len(jobs) != 48:
        raise G3Blocked("g3_g1_input_audit_invalid")
    for row in coverage_rows:
        job = jobs.get(
            (
                str(row.get("split")),
                str(row.get("scenario_id")),
                int(row.get("episode_index", -1)),
            )
        )
        if (
            job is None
            or row.get("scenario_manifest_sha256")
            != frozen_bundle["manifest_sha256"]
            or row.get("config_sha256") != config_sha256
            or row.get("input_sha256") != config.get("input_sha256")
            or row.get("code_sha256") != config.get("code_sha256")
            or row.get("denominator_sha256") != job.get("denominator_sha256")
            or row.get("denominator_cell_count")
            != job.get("denominator_cell_count")
            or row.get("safety_violation_count") != 0
            or row.get("masked_action_count") != 0
        ):
            raise G3Blocked("g3_g1_cross_root_join")
    replay = phases["p05"]["audit"].get("replay")
    if (
        phases["p05"]["audit"].get("status") != "passed"
        or not isinstance(replay, Mapping)
    ):
        raise G3Blocked("g3_g1_replay_invalid")
    recomputed = g1.recompute_g1_summary(
        test_q24_rows=[
            row for row in coverage_rows if row.get("split") == "test_q24"
        ],
        unseen24_rows=[
            row for row in coverage_rows if row.get("split") == "unseen24"
        ],
        mode="formal",
        replay=replay,
    )
    stored = _parse_json_bytes(snapshot["summary.json"], "g3_g1_summary_invalid")
    g1.validate_canonical_g1_summary(stored)
    p06 = phases["p06"]["audit"]
    p07 = phases["p07"]["audit"]
    if (
        stored != recomputed
        or p06.get("status") != "complete"
        or p06.get("canonical_summary") != recomputed
        or p06.get("canonical_summary_sha256")
        != _canonical_sha256(recomputed)
        or p07.get("status") != "ready"
        or p07.get("accepted_phase_ids") != list(g1.G1_REQUIRED_PHASE_IDS)
        or snapshot["report.md"].decode("utf-8") != g1.render_g1_report(recomputed)
    ):
        raise G3Blocked("g3_g1_native_summary_report_invalid")
    assessment_binding = _validate_g1_evidence_binding(
        {
            "g1_evidence_kind": G1_NATIVE_EVIDENCE_KIND,
            "g1_source_manifest_sha256": manifest_sha256,
        }
    )
    return {
        "root": root,
        "evidence_root": root,
        "manifest_sha256": manifest_sha256,
        "config": config,
        "input_audit": input_audit,
        "coverage_rows": coverage_rows,
        "native_summary": recomputed,
        "trace_row_count": len(trace_ids),
        "g1_evidence_kind": G1_NATIVE_EVIDENCE_KIND,
        "assessment_binding": assessment_binding,
        "platform_invariants": g1_platform_projection[
            "platform_invariants"
        ],
        "platform_invariants_source_sha256": _bytes_sha256(
            stage6_config_bytes
        ),
    }


def _assessment_split_projection(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    values = [
        _finite_number(
            row.get("coverage"),
            "g3_g1_existing_run_assessment_invalid",
            minimum=0.0,
            maximum=1.0,
        )
        for row in rows
    ]
    return {
        "sample_count": len(rows),
        "mean": sum(values) / len(values),
        "coverage_80_count": sum(
            value >= MID_COVERAGE_THRESHOLD for value in values
        ),
        "coverage_99_count": sum(
            value >= FINAL_COVERAGE_THRESHOLD for value in values
        ),
        "safety_violation_count": sum(
            _require_exact_int(
                row.get("safety_violation_count"),
                "g3_g1_existing_run_assessment_invalid",
            )
            for row in rows
        ),
        "masked_action_count": sum(
            _require_exact_int(
                row.get("masked_action_count"),
                "g3_g1_existing_run_assessment_invalid",
            )
            for row in rows
        ),
    }


def load_verified_existing_g1_assessment_source(
    *,
    root: Path,
    frozen_bundle: Mapping[str, object],
    envelope_root: Path,
    assessment_module: object | None = None,
) -> dict[str, object]:
    assessment = assessment_module or _import_local_script(
        "xunce_mid_dual_g1_existing_run_assessment"
    )
    try:
        verified = assessment.verify_existing_run_assessment(
            envelope_root=envelope_root,
            expected_source_root=root,
        )
    except Exception as exc:
        raise G3Blocked("g3_g1_existing_run_assessment_invalid") from exc

    required_hash_fields = (
        "phase_snapshot_sha256",
        "stop_evidence_sha256",
        "review_manifest_sha256",
        "review_result_sha256",
        "envelope_sha256",
        "envelope_manifest_sha256",
        "verifier_source_sha256",
    )
    if (
        getattr(verified, "source_root", None) != root
        or getattr(verified, "envelope_root", None) != envelope_root
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
            for field in required_hash_fields
        )
    ):
        raise G3Blocked("g3_g1_existing_run_assessment_invalid")

    snapshot = getattr(verified, "snapshot", None)
    review_metrics = getattr(verified, "review_metrics", None)
    if not isinstance(snapshot, Mapping) or not isinstance(
        review_metrics,
        Mapping,
    ):
        raise G3Blocked("g3_g1_existing_run_assessment_invalid")
    config_bytes = snapshot.get("config.json")
    if not isinstance(config_bytes, bytes):
        raise G3Blocked("g3_g1_existing_run_assessment_invalid")
    config = _parse_json_bytes(
        config_bytes,
        "g3_g1_existing_run_assessment_invalid",
    )
    checkpoint = config.get("checkpoint")
    scenario = config.get("scenario_manifest")
    if (
        config.get("schema_version")
        != "xunce-mid-dual-g1-effective-config/v1"
        or config.get("gate_id") != "g1"
        or config.get("runner_id")
        != "run_xunce_mid_dual_g1_coverage/v1"
        or config.get("scale_profile") != SCALE_PROFILE
        or config.get("run_id") != "g1-formal-20260727-0650-cst"
        or config.get("mode") != "formal"
        or config.get("output_root")
        != str(root).replace("\\", "/")
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("update") != 80
        or checkpoint.get("sha256") != UPDATE80_CHECKPOINT_SHA256
        or checkpoint.get("policy_state_sha256")
        != UPDATE80_POLICY_STATE_SHA256
        or not isinstance(scenario, Mapping)
        or scenario.get("sha256") != frozen_bundle.get("manifest_sha256")
        or any(
            not _is_sha256(config.get(field))
            for field in ("config_sha256", "input_sha256", "code_sha256")
        )
    ):
        raise G3Blocked("g3_g1_existing_run_assessment_invalid")

    all_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for phase_id, split in (("p03", "test_q24"), ("p04", "unseen24")):
        payload = snapshot.get(f"phases/{phase_id}/a01/results.jsonl")
        if not isinstance(payload, bytes):
            raise G3Blocked("g3_g1_existing_run_assessment_invalid")
        phase_rows = _parse_jsonl_bytes(
            payload,
            "g3_g1_existing_run_assessment_invalid",
        )
        selected = [
            dict(row)
            for row in phase_rows
            if row.get("row_kind") == "coverage_episode"
        ]
        if len(selected) != 24 or any(
            row.get("phase_id") != phase_id or row.get("split") != split
            for row in selected
        ):
            raise G3Blocked("g3_g1_existing_run_assessment_invalid")
        all_rows.extend(dict(row) for row in phase_rows)
        coverage_rows.extend(selected)

    _g1_index(coverage_rows)
    for row in coverage_rows:
        if (
            row.get("run_id") != config["run_id"]
            or row.get("config_sha256") != config["config_sha256"]
            or row.get("input_sha256") != config["input_sha256"]
            or row.get("code_sha256") != config["code_sha256"]
            or row.get("scenario_manifest_sha256")
            != frozen_bundle.get("manifest_sha256")
        ):
            raise G3Blocked("g3_g1_existing_run_assessment_invalid")

    metrics_splits = review_metrics.get("splits")
    if (
        review_metrics.get("schema_version")
        != "xunce-mid-dual-g1-existing-run-independent-recompute/v1"
        or review_metrics.get("formal_episode_count") != 48
        or review_metrics.get("split_order") != ["test_q24", "unseen24"]
        or not isinstance(metrics_splits, Mapping)
        or set(metrics_splits) != {"test_q24", "unseen24"}
        or not _is_sha256(review_metrics.get("phase_state_sha256"))
        or review_metrics.get("input_audit_canonical_sha256")
        != config["input_sha256"]
        or review_metrics.get("scenario_manifest_sha256")
        != frozen_bundle.get("manifest_sha256")
        or type(review_metrics.get("max_traversable_slope_deg")) is not float
        or review_metrics.get("max_traversable_slope_deg")
        != MAX_TRAVERSABLE_SLOPE_DEG
        or review_metrics.get("safety_clean") is not True
        or review_metrics.get("masked_action_clean") is not True
        or review_metrics.get("review_result_status")
        != getattr(verified, "numerical_result_status")
        or review_metrics.get("replay_required") is not False
        or review_metrics.get("replay_status") != "not_run_by_ruling"
    ):
        raise G3Blocked("g3_g1_existing_run_assessment_invalid")
    for split in ("test_q24", "unseen24"):
        split_rows = [
            row for row in coverage_rows if row.get("split") == split
        ]
        recomputed = _assessment_split_projection(split_rows)
        stored_split = metrics_splits.get(split)
        if not isinstance(stored_split, Mapping):
            raise G3Blocked("g3_g1_existing_run_assessment_invalid")
        for field in (
            "sample_count",
            "coverage_80_count",
            "coverage_99_count",
            "safety_violation_count",
            "masked_action_count",
        ):
            if stored_split.get(field) != recomputed[field]:
                raise G3Blocked("g3_g1_existing_run_assessment_invalid")
        if (
            not math.isclose(
                _finite_number(
                    stored_split.get("mean"),
                    "g3_g1_existing_run_assessment_invalid",
                ),
                float(recomputed["mean"]),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            or stored_split.get("integer_denominator_check") != "passed"
            or stored_split.get("phase_audit_equality") is not True
        ):
            raise G3Blocked("g3_g1_existing_run_assessment_invalid")

    binding = _validate_g1_evidence_binding(
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
    return {
        "root": root,
        "evidence_root": envelope_root,
        "manifest_sha256": verified.envelope_manifest_sha256,
        "config": config,
        "coverage_rows": coverage_rows,
        "review_metrics": dict(review_metrics),
        "trace_row_count": len(all_rows) - len(coverage_rows),
        "g1_evidence_kind": G1_ASSESSMENT_EVIDENCE_KIND,
        "assessment_binding": binding,
        "platform_invariants": {
            "wheel": {
                "max_traversable_slope_deg": MAX_TRAVERSABLE_SLOPE_DEG,
            }
        },
        "platform_invariants_source_sha256": (
            verified.verifier_source_sha256
        ),
    }


def load_verified_g1_source(
    *,
    root: Path,
    frozen_bundle: Mapping[str, object],
    g1_assessment_envelope: Path | None,
) -> dict[str, object]:
    assessment = _import_local_script(
        "xunce_mid_dual_g1_existing_run_assessment"
    )
    authorized_root = Path(str(assessment.AUTHORIZED_SOURCE_ROOT))
    _, normalized_root = _normalized_windows_path(
        root,
        "g3_g1_existing_run_assessment_scope_mismatch",
    )
    _, normalized_authorized = _normalized_windows_path(
        authorized_root,
        "g3_g1_existing_run_assessment_scope_mismatch",
    )
    if normalized_root == normalized_authorized:
        if g1_assessment_envelope is None:
            raise G3Blocked("g3_g1_existing_run_assessment_required")
        return load_verified_existing_g1_assessment_source(
            root=root,
            frozen_bundle=frozen_bundle,
            envelope_root=g1_assessment_envelope,
            assessment_module=assessment,
        )
    if g1_assessment_envelope is not None:
        raise G3Blocked("g3_g1_existing_run_assessment_scope_mismatch")
    return load_verified_g1_root(root, frozen_bundle=frozen_bundle)


def _resolve_g2_execution_bundle(
    *,
    execution_manifest_sha256: str,
    input_set_id: str,
) -> Path:
    base = Path(G2_EXECUTION_INPUT_BASE)
    matches: list[Path] = []
    if artifact_io.path_is_dir(base):
        for relative in artifact_io.list_relative_files(base):
            posix = PurePosixPath(relative)
            if len(posix.parts) != 2 or posix.name != "manifest.json":
                continue
            candidate = base / posix
            payload = _read_artifact_bytes(candidate)
            if _bytes_sha256(payload) != execution_manifest_sha256:
                continue
            manifest = _parse_json_bytes(
                payload,
                "g3_g2_execution_manifest_invalid",
            )
            if manifest.get("input_set_id") == input_set_id:
                matches.append(candidate.parent)
    if len(matches) != 1:
        raise G3Blocked("g3_g2_execution_bundle_not_unique")
    return matches[0]


def _validate_g2_runtime_source_closure_audit(
    value: object,
) -> dict[str, object]:
    schema = "xunce-mid-dual-path-planner-runtime-source-closure/v1"
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "submodule_commit",
        "dirty_inventory",
        "required_sources",
        "active_platforms",
        "platform_invariants",
        "path_planner_runtime_source_closure_sha256",
    }:
        raise G3Blocked("g3_g2_runtime_source_closure_invalid")
    commit = value.get("submodule_commit")
    dirty_inventory = value.get("dirty_inventory")
    required_sources = value.get("required_sources")
    if (
        value.get("schema_version") != schema
        or not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or not isinstance(dirty_inventory, list)
        or not isinstance(required_sources, list)
    ):
        raise G3Blocked("g3_g2_runtime_source_closure_invalid")
    validated_inventory: list[dict[str, str]] = []
    inventory_paths: set[str] = set()
    for row in dirty_inventory:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "status"}
            or not isinstance(row.get("path"), str)
            or "\\" in str(row.get("path"))
            or not isinstance(row.get("status"), str)
            or len(str(row["status"])) != 2
            or row["path"] in inventory_paths
            or PurePosixPath(str(row["path"])).is_absolute()
            or any(
                part in {"", ".", ".."}
                for part in PurePosixPath(str(row["path"])).parts
            )
        ):
            raise G3Blocked("g3_g2_runtime_source_closure_invalid")
        inventory_paths.add(str(row["path"]))
        validated_inventory.append(
            {"path": str(row["path"]), "status": str(row["status"])}
        )
    if validated_inventory != sorted(
        validated_inventory,
        key=lambda row: row["path"],
    ):
        raise G3Blocked("g3_g2_runtime_source_closure_invalid")

    validated_sources: list[dict[str, object]] = []
    source_paths: set[str] = set()
    for row in required_sources:
        if not isinstance(row, Mapping) or set(row) != {
            "logical_path",
            "size_bytes",
            "sha256",
        }:
            raise G3Blocked("g3_g2_runtime_source_closure_invalid")
        logical_path = row.get("logical_path")
        size_bytes = row.get("size_bytes")
        if (
            not isinstance(logical_path, str)
            or "\\" in str(logical_path)
            or not logical_path.startswith("src/path_planner/v2/")
            or not logical_path.endswith(".py")
            or PurePosixPath(logical_path).is_absolute()
            or any(
                part in {"", ".", ".."}
                for part in PurePosixPath(logical_path).parts
            )
            or type(size_bytes) is not int
            or size_bytes < 0
            or logical_path in source_paths
        ):
            raise G3Blocked("g3_g2_runtime_source_closure_invalid")
        source_paths.add(logical_path)
        validated_sources.append(
            {
                "logical_path": logical_path,
                "size_bytes": size_bytes,
                "sha256": _require_sha256(
                    row.get("sha256"),
                    "g3_g2_runtime_source_closure_invalid",
                ),
            }
        )
    if not validated_sources or validated_sources != sorted(
        validated_sources,
        key=lambda row: str(row["logical_path"]),
    ):
        raise G3Blocked("g3_g2_runtime_source_closure_invalid")
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
    if not required_layers.issubset(source_paths) or not any(
        path.startswith("src/path_planner/v2/providers/")
        for path in source_paths
    ):
        raise G3Blocked("g3_g2_runtime_source_closure_invalid")
    invariant_projection = validate_platform_invariants(
        value.get("active_platforms"),
        value.get("platform_invariants"),
    )
    core = {
        "schema_version": schema,
        "submodule_commit": commit,
        "dirty_inventory": validated_inventory,
        "required_sources": validated_sources,
        **invariant_projection,
    }
    rebuilt = {
        **core,
        "path_planner_runtime_source_closure_sha256": _domain_hash(
            schema,
            _canonical_bytes(core),
        ),
    }
    if dict(value) != rebuilt:
        raise G3Blocked("g3_g2_runtime_source_closure_invalid")
    return rebuilt


def validate_g2_r6_runtime_evidence(
    *,
    config: Mapping[str, object],
    input_audit: Mapping[str, object],
    approval: Mapping[str, object],
    hopper_resolution: Mapping[str, object],
    phase_audits: Mapping[str, Mapping[str, object]],
    summary: Mapping[str, object],
    runtime_closure_audit: Mapping[str, object],
    platform_invariants_audit: Mapping[str, object],
    local_source_sha256: object,
) -> dict[str, object]:
    expected_evidence_binding = {
        "schema_version": "xunce-mid-dual-g2-evidence-binding/v1",
        "input_audit_path": "g2_input_audit.json",
        "lineage_audit_path": "lineage_audit.json",
        "report_audit_path": "g2_report_audit.json",
        "runtime_source_closure_audit_path": (
            "g2_runtime_source_closure_audit.json"
        ),
        "platform_invariants_audit_path": (
            "g2_platform_invariants_audit.json"
        ),
        "formal_environment_audit_path": (
            "g2_formal_environment_audit.json"
        ),
    }
    if config.get("evidence_binding") != expected_evidence_binding:
        raise G3Blocked("g3_g2_config_invalid")
    validate_platform_invariants(
        config.get("active_platforms"),
        config.get("platform_invariants"),
    )
    runtime_sha256 = _require_sha256(
        config.get("path_planner_runtime_source_closure_sha256"),
        "g3_g2_runtime_source_closure_invalid",
    )
    validated_closure = _validate_g2_runtime_source_closure_audit(
        runtime_closure_audit
    )
    if (
        validated_closure.get(
            "path_planner_runtime_source_closure_sha256"
        )
        != runtime_sha256
        or input_audit.get("path_planner_runtime_source_closure")
        != validated_closure
        or input_audit.get(
            "path_planner_runtime_source_closure_sha256"
        )
        != runtime_sha256
        or approval.get("path_planner_runtime_source_closure")
        != validated_closure
        or approval.get(
            "path_planner_runtime_source_closure_sha256"
        )
        != runtime_sha256
        or hopper_resolution.get(
            "path_planner_runtime_source_closure_sha256"
        )
        != runtime_sha256
        or set(phase_audits) != {"p01", "p02", "p03", "p04"}
    ):
        raise G3Blocked("g3_g2_runtime_source_closure_invalid")
    for phase_id in ("p01", "p02", "p03", "p04"):
        phase_audit = phase_audits[phase_id]
        validate_platform_invariants(
            phase_audit.get("active_platforms"),
            phase_audit.get("platform_invariants"),
        )
        if (
            phase_audit.get(
                "path_planner_runtime_source_closure_sha256"
            )
            != runtime_sha256
        ):
            raise G3Blocked("g3_g2_runtime_source_closure_invalid")
    validate_platform_invariants(
        input_audit.get("active_platforms"),
        input_audit.get("platform_invariants"),
    )
    validate_platform_invariants(
        summary.get("active_platforms"),
        summary.get("platform_invariants"),
    )
    if (
        summary.get("path_planner_runtime_source_closure_sha256")
        != runtime_sha256
    ):
        raise G3Blocked("g3_g2_runtime_source_closure_invalid")

    local_sha256 = _require_sha256(
        local_source_sha256,
        "g3_g2_lineage_invalid",
    )
    expected_code_sha256 = _domain_hash(
        "xunce-mid-dual-g2-code-with-runtime-source/v1",
        _canonical_bytes(
            {
                "local_source_sha256": local_sha256,
                "path_planner_runtime_source_closure_sha256": (
                    runtime_sha256
                ),
            }
        ),
    )
    if config.get("code_sha256") != expected_code_sha256:
        raise G3Blocked("g3_g2_summary_lineage_invalid")

    expected_platform_audit_keys = {
        "schema_version",
        "gate_id",
        "scale_profile",
        "run_id",
        "status",
        "formal_evidence_eligible",
        "active_platforms",
        "platform_invariants",
        "config_sha256",
        "input_sha256",
        "code_sha256",
        "path_planner_runtime_source_closure_sha256",
    }
    if set(platform_invariants_audit) != expected_platform_audit_keys:
        raise G3Blocked("g3_g2_platform_invariants_audit_invalid")
    validate_platform_invariants(
        platform_invariants_audit.get("active_platforms"),
        platform_invariants_audit.get("platform_invariants"),
    )
    if (
        platform_invariants_audit.get("schema_version")
        != "xunce-mid-dual-g2-platform-invariants-audit/v1"
        or platform_invariants_audit.get("gate_id") != "g2"
        or platform_invariants_audit.get("scale_profile") != SCALE_PROFILE
        or platform_invariants_audit.get("run_id") != config.get("run_id")
        or platform_invariants_audit.get("status") != "passed"
        or platform_invariants_audit.get("formal_evidence_eligible")
        is not True
        or platform_invariants_audit.get("config_sha256")
        != config.get("config_sha256")
        or platform_invariants_audit.get("input_sha256")
        != config.get("input_sha256")
        or platform_invariants_audit.get("code_sha256")
        != config.get("code_sha256")
        or platform_invariants_audit.get(
            "path_planner_runtime_source_closure_sha256"
        )
        != runtime_sha256
    ):
        raise G3Blocked("g3_g2_platform_invariants_audit_invalid")
    return {
        "runtime_source_closure": dict(validated_closure),
        "path_planner_runtime_source_closure_sha256": runtime_sha256,
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": PLATFORM_INVARIANTS,
        "expected_code_sha256": expected_code_sha256,
    }


def validate_g2_completed_result_status(status: object) -> str:
    if status not in {"passed", "failed"}:
        raise G3Blocked("g3_g2_summary_status_invalid")
    return str(status)


def validate_g2_p03_worker_count(audit: Mapping[str, object]) -> None:
    present = [
        field
        for field in ("worker_count", "worker_one_count")
        if field in audit
    ]
    if not present:
        raise G3Blocked("g3_g2_nonformal_phase_invalid")
    for field in present:
        if _require_exact_int(
            audit[field],
            "g3_g2_nonformal_phase_invalid",
        ) != 1:
            raise G3Blocked("g3_g2_nonformal_phase_invalid")


def validate_g2_formal_environment_evidence(
    *,
    g2: object,
    snapshot: Mapping[str, bytes],
    p04_audit: Mapping[str, object],
) -> dict[str, object]:
    try:
        embedded = g2.validate_formal_environment_phase_audit(p04_audit)
    except Exception as exc:
        raise G3Blocked("g3_g2_formal_environment_p04_invalid") from exc
    final_bytes = snapshot.get("g2_formal_environment_audit.json")
    if not isinstance(final_bytes, bytes):
        raise G3Blocked("g3_g2_formal_environment_final_invalid")
    final_audit = _parse_json_bytes(
        final_bytes,
        "g3_g2_formal_environment_final_invalid",
    )
    try:
        validated_final = g2._validate_complete_formal_environment_audit(  # noqa: SLF001
            final_audit
        )
    except Exception as exc:
        raise G3Blocked("g3_g2_formal_environment_final_invalid") from exc
    if embedded != validated_final:
        raise G3Blocked("g3_g2_formal_environment_audit_mismatch")
    return dict(validated_final)


def validate_g2_summary_report(
    *,
    g2: object,
    summary: Mapping[str, object],
    report: str,
    report_audit: Mapping[str, object],
    formal_environment_audit: Mapping[str, object],
) -> None:
    if (
        report
        != g2._render_report(  # noqa: SLF001
            summary,
            formal_environment_audit=formal_environment_audit,
        )
        or report_audit
        != g2._source_report_audit(  # noqa: SLF001
            summary=summary,
            report=report,
        )
    ):
        raise G3Blocked("g3_g2_summary_lineage_invalid")


def load_verified_g2_root(root: Path) -> dict[str, object]:
    g2 = _import_local_script("run_xunce_mid_dual_g2_planning_time")

    manifest, snapshot, manifest_sha256 = _verified_manifest_snapshot(
        root,
        reason="g3_g2_manifest_invalid",
    )
    if manifest.get("formal_evidence_eligible") is not True:
        raise G3Blocked("g3_g2_root_not_formal")
    config = _parse_json_bytes(snapshot["config.json"], "g3_g2_config_invalid")
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
        "active_platforms",
        "platform_invariants",
        "path_planner_runtime_source_closure_sha256",
        "formal_environment_gate",
        "config_sha256",
    }
    if (
        set(config) != expected_config_keys
        or config.get("schema_version") != g2.G2_CONFIG_SCHEMA_VERSION
        or config.get("gate_id") != "g2"
        or config.get("runner_id") != g2.G2_RUNNER_ID
        or config.get("output_root") != g2.G2_OUTPUT_BASE
        or config.get("scale_profile") != SCALE_PROFILE
        or config.get("required_phase_ids")
        != list(g2.G2_REQUIRED_PHASE_IDS)
        or validate_platform_invariants(
            config.get("active_platforms"),
            config.get("platform_invariants"),
        )
        != {
            "active_platforms": list(ACTIVE_PLATFORMS),
            "platform_invariants": PLATFORM_INVARIANTS,
        }
        or config.get("source_contract_sha256")
        != _canonical_sha256(g2.G2_SOURCE_CONTRACT)
        or config.get("config_sha256") != manifest.get("config_sha256")
    ):
        raise G3Blocked("g3_g2_config_invalid")
    try:
        formal_environment_policy = g2.validate_formal_environment_policy(
            config.get("formal_environment_gate")
        )
    except Exception as exc:
        raise G3Blocked("g3_g2_config_invalid") from exc
    phases = _accepted_phase_snapshot(
        snapshot,
        g2.G2_REQUIRED_PHASE_IDS,
        gate_id="g2",
    )
    if any(phases[phase_id]["rows"] for phase_id in ("p01", "p02", "p03")):
        raise G3Blocked("g3_g2_phase_rows_invalid")
    p01 = phases["p01"]["audit"]
    p02 = phases["p02"]["audit"]
    p03 = phases["p03"]["audit"]
    for phase_id in g2.G2_REQUIRED_PHASE_IDS:
        phase_audit = phases[phase_id]["audit"]
        validate_platform_invariants(
            phase_audit.get("active_platforms"),
            phase_audit.get("platform_invariants"),
        )
    validate_g2_p03_worker_count(p03)
    if (
        p01.get("schema_version") != "xunce-mid-dual-g2-phase-audit/v1"
        or p01.get("phase_name") != "preflight"
        or p01.get("status") != "passed"
        or p01.get("blocking_reason") is not None
        or p01.get("provider_called") is not False
        or p01.get("formal_row_count") != 0
        or p02.get("schema_version")
        != "xunce-mid-dual-g2-phase-audit/v1"
        or p02.get("phase_name") != "cold_start_and_warmup"
        or p02.get("status") != "complete"
        or p02.get("formal_sample") is not False
        or _require_exact_int(
            p02.get("cold_start_count"),
            "g3_g2_p02_invalid",
        )
        <= 0
        or _require_exact_int(
            p02.get("warmup_count"),
            "g3_g2_p02_invalid",
        )
        <= 0
        or not isinstance(p02.get("cold_start_results"), list)
        or not isinstance(p02.get("warmup_results"), list)
        or p03.get("schema_version")
        != "xunce-mid-dual-g2-phase-audit/v1"
        or p03.get("phase_name") != "worker_one_semantic_diagnostic"
        or p03.get("status") != "complete"
        or p03.get("formal_sample") is not False
        or not isinstance(p03.get("worker_one_results"), list)
        or p03.get("request_count") != len(p03["worker_one_results"])
    ):
        raise G3Blocked("g3_g2_nonformal_phase_invalid")
    p04_rows = phases["p04"]["rows"]
    p04 = phases["p04"]["audit"]
    if (
        p04.get("schema_version") != "xunce-mid-dual-g2-phase-audit/v1"
        or p04.get("gate_id") != "g2"
        or p04.get("runner_id") != g2.G2_RUNNER_ID
        or p04.get("status") != "complete"
        or p04.get("phase_name") != "formal_worker_four"
        or p04.get("formal_sample") is not True
        or p04.get("formal_worker_count") != 4
        or p04.get("formal_call_count") != 645
        or p04.get("timing_contract_id") != TIMING_CONTRACT_ID
        or len(p04_rows) != 645
    ):
        raise G3Blocked("g3_g2_p04_invalid")
    formal_environment_audit = validate_g2_formal_environment_evidence(
        g2=g2,
        snapshot=snapshot,
        p04_audit=p04,
    )
    if formal_environment_audit.get("formal_environment_policy") != (
        formal_environment_policy
    ):
        raise G3Blocked("g3_g2_formal_environment_policy_invalid")
    recomputed = g2.recompute_g2_summary(p04_rows)
    if recomputed.get("formal_call_count") != 645:
        raise G3Blocked("g3_g2_p04_invalid")
    stored = _parse_json_bytes(snapshot["summary.json"], "g3_g2_summary_invalid")
    validate_g2_completed_result_status(stored.get("status"))
    report = _canonical_report_text(snapshot["report.md"].decode("utf-8"))
    report_audit = _parse_json_bytes(
        snapshot.get("g2_report_audit.json", b""),
        "g3_g2_report_audit_invalid",
    )
    if (
        stored
        != {
            "schema_version": g2.G2_SUMMARY_SCHEMA_VERSION,
            "scale_profile": SCALE_PROFILE,
            "gate_id": "g2",
            "run_id": config.get("run_id"),
            "status": recomputed["status"],
            "formal_evidence_eligible": True,
            "active_platforms": list(ACTIVE_PLATFORMS),
            "platform_invariants": PLATFORM_INVARIANTS,
            "path_planner_runtime_source_closure_sha256": (
                config.get("path_planner_runtime_source_closure_sha256")
            ),
            "recomputed": recomputed,
        }
    ):
        raise G3Blocked("g3_g2_summary_lineage_invalid")
    validate_g2_summary_report(
        g2=g2,
        summary=stored,
        report=report,
        report_audit=report_audit,
        formal_environment_audit=formal_environment_audit,
    )
    local_source_sha256 = _common_lineage_code_sha256(
        snapshot=snapshot,
        expected_sources=g2.G2_REQUIRED_SOURCE_RELATIVE_PATHS,
        reason="g3_g2_lineage_invalid",
    )
    input_audit = _parse_json_bytes(
        snapshot.get("g2_input_audit.json", b""),
        "g3_g2_input_audit_invalid",
    )
    if (
        _json_artifact_sha256(input_audit) != config.get("input_sha256")
        or input_audit.get("formal_evidence_eligible") is not True
        or input_audit.get("status") != "ready"
        or input_audit.get("blockers") != []
    ):
        raise G3Blocked("g3_g2_input_audit_invalid")
    validate_platform_invariants(
        input_audit.get("active_platforms"),
        input_audit.get("platform_invariants"),
    )
    p01 = phases["p01"]["audit"]
    execution_manifest_sha256 = _require_sha256(
        p01.get("input_manifest_sha256"),
        "g3_g2_execution_manifest_invalid",
    )
    input_set_id = _require_nonempty(
        input_audit.get("input_set_id"),
        "g3_g2_input_audit_invalid",
    )
    execution_root = _resolve_g2_execution_bundle(
        execution_manifest_sha256=execution_manifest_sha256,
        input_set_id=input_set_id,
    )
    runtime_closure_audit = _parse_json_bytes(
        snapshot.get("g2_runtime_source_closure_audit.json", b""),
        "g3_g2_runtime_source_closure_invalid",
    )
    scripts_path = str(Path(__file__).resolve().parent)
    added_scripts_path = scripts_path not in sys.path
    if added_scripts_path:
        sys.path.insert(0, scripts_path)
    try:
        execution = g2._read_execution_bundle(  # noqa: SLF001
            execution_root,
            runtime_source_closure=runtime_closure_audit,
        )
    except Exception as exc:
        raise G3Blocked("g3_g2_execution_reaudit_invalid") from exc
    finally:
        if added_scripts_path:
            sys.path.remove(scripts_path)
    if (
        execution.get("input_audit") != input_audit
        or execution.get("manifest_sha256") != execution_manifest_sha256
        or execution.get("input_sha256") != config.get("input_sha256")
        or p01.get("input_sha256") != config.get("input_sha256")
        or p01.get("code_sha256") != config.get("code_sha256")
    ):
        raise G3Blocked("g3_g2_execution_reaudit_invalid")
    cohort = input_audit.get("g3_replay_cohort")
    approval = input_audit.get("approval")
    hopper = input_audit.get("hopper_resolution")
    provider = input_audit.get("provider_source")
    oracle = input_audit.get("oracle_source")
    if (
        not isinstance(cohort, Mapping)
        or not isinstance(cohort.get("rows"), list)
        or len(cohort["rows"]) != 6
        or cohort.get("cohort_sha256")
        != input_audit.get("g3_replay_cohort_sha256")
        or not isinstance(approval, Mapping)
        or approval.get("formal_evidence_eligible") is not True
        or approval.get("g3_scope_authorized") is not True
        or approval.get("g3_replay_cohort_sha256")
        != cohort.get("cohort_sha256")
        or approval.get("g3_replay_truth_request_sha256")
        != [row.get("truth_request_sha256") for row in cohort["rows"]]
        or approval.get("g3_replay_provider_request_sha256")
        != [row.get("provider_request_sha256") for row in cohort["rows"]]
        or not isinstance(hopper, Mapping)
        or hopper.get("formal_evidence_eligible") is not True
        or hopper.get("blockers") != []
        or not isinstance(provider, Mapping)
        or not isinstance(oracle, Mapping)
        or any(
            provider.get(field) == oracle.get(field)
            for field in (
                "identity",
                "implementation_sha256",
                "source_bytes_sha256",
            )
        )
    ):
        raise G3Blocked("g3_g2_o2_cohort_identity_invalid")
    runtime_closure_bytes = snapshot.get(
        "g2_runtime_source_closure_audit.json",
        b"",
    )
    platform_audit_bytes = snapshot.get(
        "g2_platform_invariants_audit.json",
        b"",
    )
    platform_audit = _parse_json_bytes(
        platform_audit_bytes,
        "g3_g2_platform_invariants_audit_invalid",
    )
    runtime_evidence = validate_g2_r6_runtime_evidence(
        config=config,
        input_audit=input_audit,
        approval=approval,
        hopper_resolution=hopper,
        phase_audits={
            phase_id: phases[phase_id]["audit"]
            for phase_id in g2.G2_REQUIRED_PHASE_IDS
        },
        summary=stored,
        runtime_closure_audit=runtime_closure_audit,
        platform_invariants_audit=platform_audit,
        local_source_sha256=local_source_sha256,
    )
    calls_by_request: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in p04_rows:
        key = (
            str(row.get("platform")),
            str(row.get("request_id")),
            str(row.get("request_sha256")),
        )
        calls_by_request.setdefault(key, []).append(dict(row))
    cohort_calls: list[dict[str, object]] = []
    for row in cohort["rows"]:
        key = (
            str(row.get("platform")),
            str(row.get("request_id")),
            str(row.get("provider_request_sha256")),
        )
        selected = sorted(
            calls_by_request.get(key, []),
            key=lambda call: int(call["repeat_index"]),
        )
        if (
            len(selected) != 5
            or [call["repeat_index"] for call in selected] != list(range(5))
            or len({call["semantic_digest"] for call in selected}) != 1
        ):
            raise G3Blocked("g3_g2_cohort_call_join_invalid")
        cohort_calls.extend(selected)
    return {
        "root": root,
        "manifest_sha256": manifest_sha256,
        "config": config,
        "input_audit": input_audit,
        "formal_rows": p04_rows,
        "native_summary": stored,
        "formal_environment_audit": formal_environment_audit,
        "formal_environment_audit_sha256": g2.formal_environment_audit_binding(
            formal_environment_audit
        )["formal_environment_audit_sha256"],
        "formal_environment_audit_file_sha256": _bytes_sha256(
            snapshot["g2_formal_environment_audit.json"]
        ),
        "execution_bundle": execution,
        "execution_root": execution_root,
        "cohort": cohort,
        "cohort_calls": cohort_calls,
        "approval": approval,
        "hopper_resolution": hopper,
        "platform_invariants": PLATFORM_INVARIANTS,
        "platform_invariants_audit_sha256": _bytes_sha256(
            platform_audit_bytes
        ),
        "runtime_source_closure": runtime_evidence[
            "runtime_source_closure"
        ],
        "path_planner_runtime_source_closure_sha256": runtime_evidence[
            "path_planner_runtime_source_closure_sha256"
        ],
        "runtime_source_closure_audit_sha256": _bytes_sha256(
            runtime_closure_bytes
        ),
        "provider_source": provider,
        "oracle_source": oracle,
    }


def validate_g3_phase_rows(
    *,
    phase_id: str,
    rows: Sequence[Mapping[str, object]],
    accepted_phase_ids: Sequence[str],
) -> dict[str, object]:
    materialized = [dict(row) for row in rows]
    if phase_id not in G3_REQUIRED_PHASE_IDS:
        raise G3Blocked("g3_phase_id_invalid")
    expected_prefix = tuple(G3_REQUIRED_PHASE_IDS[: len(accepted_phase_ids)])
    if tuple(accepted_phase_ids) != expected_prefix:
        raise G3Blocked("g3_phase_prefix_invalid")
    if phase_id == "p02" and tuple(accepted_phase_ids) != ("p01",):
        raise G3Blocked("g3_phase_requires_p01")
    if phase_id == "p01" and accepted_phase_ids:
        raise G3Blocked("g3_phase_prefix_invalid")
    expected_kind = (
        "g3_wheel_step" if phase_id == "p01" else "g3_interface_replay"
    )
    if not materialized or any(
        row.get("row_kind") != expected_kind for row in materialized
    ):
        raise G3Blocked("g3_phase_row_kind")
    if any(
        row.get("execution_class") != "formal"
        or row.get("formal_sample") is not True
        for row in materialized
    ):
        raise G3Blocked("g3_formal_fixture_partition")
    if any(
        row.get("timing_contract_id") != TIMING_CONTRACT_ID
        for row in materialized
    ):
        raise G3Blocked("g3_timing_contract")
    if phase_id == "p01":
        if any(
            set(row) != set(_WHEEL_ROW_FIELDS | _FORMAL_WHEEL_EXTRA_FIELDS)
            for row in materialized
        ):
            raise G3Blocked("g3_wheel_row_schema")
        episodes: dict[str, list[dict[str, object]]] = {}
        step_ids: set[str] = set()
        for row in materialized:
            episode_id = _require_nonempty(
                row.get("episode_id"),
                "g3_wheel_episode_matrix",
            )
            step_id = _require_nonempty(
                row.get("step_id"),
                "g3_wheel_step_chain",
            )
            if step_id in step_ids:
                raise G3Blocked("g3_wheel_step_chain")
            step_ids.add(step_id)
            episodes.setdefault(episode_id, []).append(row)
        if len(episodes) != 10:
            raise G3Blocked("g3_wheel_episode_matrix")
        for episode_rows in episodes.values():
            ordered = sorted(
                episode_rows,
                key=lambda row: _require_exact_int(
                    row.get("step_index"),
                    "g3_wheel_step_chain",
                ),
            )
            if [row["step_index"] for row in ordered] != list(
                range(len(ordered))
            ):
                raise G3Blocked("g3_wheel_step_chain")
            terminals = [row for row in ordered if row.get("is_terminal") is True]
            if len(terminals) != 1 or terminals[0] is not ordered[-1]:
                raise G3Blocked("g3_wheel_step_chain")
        return {
            "phase_id": "p01",
            "phase_name": G3_PHASE_NAMES["p01"],
            "row_kind": expected_kind,
            "row_count": len(materialized),
            "episode_count": 10,
            "fixture_ids": [],
            "active_platforms": list(ACTIVE_PLATFORMS),
            "platform_invariants": PLATFORM_INVARIANTS,
        }
    if any(
        set(row) != set(_INTERFACE_ROW_FIELDS | _FORMAL_INTERFACE_EXTRA_FIELDS)
        for row in materialized
    ):
        raise G3Blocked("g3_interface_row_schema")
    counts = Counter(str(row.get("platform")) for row in materialized)
    if len(materialized) != 6 or dict(counts) != {"legged": 3, "hopper": 3}:
        raise G3Blocked("g3_interface_request_matrix")
    if len({str(row.get("replay_id")) for row in materialized}) != 6:
        raise G3Blocked("g3_interface_request_matrix")
    return {
        "phase_id": "p02",
        "phase_name": G3_PHASE_NAMES["p02"],
        "row_kind": expected_kind,
        "row_count": 6,
        "platform_counts": {"legged": 3, "hopper": 3},
        "fixture_ids": [],
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": PLATFORM_INVARIANTS,
    }


def materialize_wheel_authority(
    rows: Sequence[Mapping[str, object]],
    *,
    config_sha256: str,
    input_sha256: str,
    code_sha256: str,
    frozen_manifest_sha256: str,
    g1_source_manifest_sha256: str,
    g2_source_manifest_sha256: str,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    identity = {
        "wheel_platform": WHEEL_PLATFORM,
        "wheel_profile": WHEEL_PROFILE,
        "wheel_capability_revision": WHEEL_CAPABILITY_REVISION,
        "checkpoint_sha256": UPDATE80_CHECKPOINT_SHA256,
        "policy_state_sha256": UPDATE80_POLICY_STATE_SHA256,
        "config_sha256": _require_sha256(
            config_sha256,
            "g3_wheel_identity_invalid",
        ),
        "input_sha256": _require_sha256(
            input_sha256,
            "g3_wheel_identity_invalid",
        ),
        "code_sha256": _require_sha256(
            code_sha256,
            "g3_wheel_identity_invalid",
        ),
        "source_sha256": code_sha256,
        "frozen_manifest_sha256": _require_sha256(
            frozen_manifest_sha256,
            "g3_wheel_identity_invalid",
        ),
        "g1_source_manifest_sha256": _require_sha256(
            g1_source_manifest_sha256,
            "g3_wheel_identity_invalid",
        ),
        "g2_source_manifest_sha256": _require_sha256(
            g2_source_manifest_sha256,
            "g3_wheel_identity_invalid",
        ),
    }
    materialized: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    for raw in rows:
        if set(raw) != set(_WHEEL_ROW_FIELDS):
            raise G3Blocked("g3_wheel_row_schema")
        row = dict(raw)
        decision_record_id = f"decision:{row['step_id']}"
        decision_record = {
            "schema_version": "xunce-mid-dual-g3-decision-record/v1",
            "decision_record_id": decision_record_id,
            "scenario_id": row["scenario_id"],
            "split": row["split"],
            "episode_id": row["episode_id"],
            "episode_index": row["episode_index"],
            "step_id": row["step_id"],
            "step_index": row["step_index"],
            "pre_observation_sha256": row["pre_snapshot_sha256"],
            "candidate_id": row["candidate_id"],
            "selected_candidate_cell_xy": row["selected_candidate_cell_xy"],
            "selected_theta": row["selected_theta"],
            "platform_invariant": PLATFORM_INVARIANTS[WHEEL_PLATFORM],
            "policy_identity": {
                key: value
                for key, value in identity.items()
                if key
                in {
                    "wheel_platform",
                    "wheel_profile",
                    "wheel_capability_revision",
                    "checkpoint_sha256",
                    "policy_state_sha256",
                    "config_sha256",
                    "input_sha256",
                    "code_sha256",
                    "frozen_manifest_sha256",
                }
            },
        }
        decision_sha256 = _canonical_sha256(decision_record)
        row["decision_sha256"] = decision_sha256
        row["candidate_sha256"] = deterministic_candidate_sha256(
            scenario_id=str(row["scenario_id"]),
            step_index=int(row["step_index"]),
            pre_snapshot_sha256=str(row["pre_snapshot_sha256"]),
            decision_sha256=decision_sha256,
            candidate_id=str(row["candidate_id"]),
            selected_candidate_cell_xy=row["selected_candidate_cell_xy"],
            selected_theta=float(row["selected_theta"]),
        )
        row["request_candidate_sha256"] = row["candidate_sha256"]
        row["request_sha256"] = deterministic_request_sha256(
            request_id=str(row["request_id"]),
            candidate_sha256=str(row["candidate_sha256"]),
            pre_snapshot_sha256=str(row["pre_snapshot_sha256"]),
        )
        row["route_request_sha256"] = row["request_sha256"]
        row["route_result_sha256"] = deterministic_route_result_sha256(
            request_sha256=str(row["request_sha256"]),
            planner_success=bool(row["planner_success"]),
            planner_failure_reason=str(row["planner_failure_reason"]),
            planned_path_cells=row["planned_path_cells"],
            planner_path_length_m=float(row["planner_path_length_m"]),
            route_endpoint_cell_xy=row["route_endpoint_cell_xy"],
            route_endpoint_theta=float(row["route_endpoint_theta"]),
        )
        row["feedback_route_result_sha256"] = row["route_result_sha256"]
        row["feedback_sha256"] = deterministic_feedback_sha256(
            route_result_sha256=str(row["route_result_sha256"]),
            feedback_pose_cell_xy=row["feedback_pose_cell_xy"],
            feedback_theta=float(row["feedback_theta"]),
            coverage=float(row["coverage"]),
            safety_violation_count=int(row["safety_violation_count"]),
            masked_action_count=int(row["masked_action_count"]),
            post_snapshot_sha256=str(row["post_snapshot_sha256"]),
        )
        row["post_snapshot_parent_sha256"] = row["feedback_sha256"]
        observation_record_id = f"observation:{row['step_id']}:post"
        observation_record = {
            "schema_version": "xunce-mid-dual-g3-observation-record/v1",
            "observation_record_id": observation_record_id,
            "scenario_id": row["scenario_id"],
            "episode_id": row["episode_id"],
            "step_id": row["step_id"],
            "step_index": row["step_index"],
            "feedback_sha256": row["feedback_sha256"],
            "observation_sha256": row["post_snapshot_sha256"],
        }
        row.update(identity)
        row.update(
            {
                "execution_class": "formal",
                "formal_sample": True,
                "timing_contract_id": TIMING_CONTRACT_ID,
                "decision_record_id": decision_record_id,
                "observation_record_id": observation_record_id,
                "post_observation_record_sha256": _canonical_sha256(
                    observation_record
                ),
            }
        )
        decisions.append(
            {**decision_record, "decision_sha256": decision_sha256}
        )
        observations.append(
            {
                **observation_record,
                "record_sha256": row["post_observation_record_sha256"],
            }
        )
        materialized.append(row)
    return materialized, decisions, observations


def validate_wheel_authority_records(
    rows: Sequence[Mapping[str, object]],
    *,
    decision_records: Sequence[Mapping[str, object]],
    observation_records: Sequence[Mapping[str, object]],
) -> None:
    decisions: dict[str, Mapping[str, object]] = {}
    for raw in decision_records:
        record = dict(raw)
        digest = record.pop("decision_sha256", None)
        record_id = record.get("decision_record_id")
        if (
            not isinstance(record_id, str)
            or record_id in decisions
            or digest != _canonical_sha256(record)
        ):
            raise G3Blocked("g3_decision_artifact_invalid")
        decisions[record_id] = raw
    observations: dict[str, Mapping[str, object]] = {}
    for raw in observation_records:
        record = dict(raw)
        digest = record.pop("record_sha256", None)
        record_id = record.get("observation_record_id")
        if (
            not isinstance(record_id, str)
            or record_id in observations
            or digest != _canonical_sha256(record)
        ):
            raise G3Blocked("g3_observation_artifact_invalid")
        observations[record_id] = raw
    if len(decisions) != len(rows) or len(observations) != len(rows):
        raise G3Blocked("g3_wheel_authority_count_invalid")
    for row in rows:
        decision = decisions.get(str(row.get("decision_record_id")))
        observation = observations.get(str(row.get("observation_record_id")))
        if (
            decision is None
            or observation is None
            or decision.get("decision_sha256") != row.get("decision_sha256")
            or decision.get("scenario_id") != row.get("scenario_id")
            or decision.get("episode_id") != row.get("episode_id")
            or decision.get("step_index") != row.get("step_index")
            or decision.get("pre_observation_sha256")
            != row.get("pre_snapshot_sha256")
            or decision.get("candidate_id") != row.get("candidate_id")
            or decision.get("selected_candidate_cell_xy")
            != row.get("selected_candidate_cell_xy")
            or decision.get("selected_theta") != row.get("selected_theta")
            or decision.get("platform_invariant")
            != PLATFORM_INVARIANTS[WHEEL_PLATFORM]
            or observation.get("record_sha256")
            != row.get("post_observation_record_sha256")
            or observation.get("observation_sha256")
            != row.get("post_snapshot_sha256")
            or observation.get("feedback_sha256") != row.get("feedback_sha256")
        ):
            raise G3Blocked("g3_wheel_authority_binding_invalid")


def execute_interface_replays(
    *,
    g2_source: Mapping[str, object],
    run_id: str,
    config_sha256: str,
    input_sha256: str,
    code_sha256: str,
    g1_source_manifest_sha256: str,
    g2_source_manifest_sha256: str,
) -> list[dict[str, object]]:
    g2 = _import_local_script("run_xunce_mid_dual_g2_planning_time")

    execution = g2_source.get("execution_bundle")
    cohort = g2_source.get("cohort")
    formal_rows = g2_source.get("formal_rows")
    if (
        not isinstance(execution, Mapping)
        or not isinstance(cohort, Mapping)
        or not isinstance(cohort.get("rows"), list)
        or not isinstance(formal_rows, list)
    ):
        raise G3Blocked("g3_interface_source_invalid")
    schedule = g2.build_formal_schedule(
        str(execution["manifest"]["input_set_id"]),
        execution["requests"],
    )
    scheduled: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for call in schedule["calls"]:
        if call.get("repeat_index") != 0:
            continue
        key = (
            str(call.get("platform")),
            str(call.get("request_id")),
            str(call.get("request_sha256")),
        )
        scheduled[key] = call
    references: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for raw in formal_rows:
        key = (
            str(raw.get("platform")),
            str(raw.get("request_id")),
            str(raw.get("request_sha256")),
        )
        references.setdefault(key, []).append(dict(raw))
    provider_identity_sha256 = _canonical_sha256(
        g2_source["provider_source"]
    )
    oracle_identity_sha256 = _canonical_sha256(g2_source["oracle_source"])
    approval_sha256 = _canonical_sha256(g2_source["approval"])
    hopper_resolution_sha256 = _canonical_sha256(
        g2_source["hopper_resolution"]
    )
    cohort_sha256 = _require_sha256(
        cohort.get("cohort_sha256"),
        "g3_interface_source_invalid",
    )
    output: list[dict[str, object]] = []
    for selection in cohort["rows"]:
        key = (
            str(selection.get("platform")),
            str(selection.get("request_id")),
            str(selection.get("provider_request_sha256")),
        )
        call = scheduled.get(key)
        reference_rows = sorted(
            references.get(key, []),
            key=lambda row: int(row["repeat_index"]),
        )
        if (
            call is None
            or len(reference_rows) != 5
            or [row["repeat_index"] for row in reference_rows]
            != list(range(5))
        ):
            raise G3Blocked("g3_g2_cohort_call_join_invalid")
        hydrated = g2._hydrate_calls(  # noqa: SLF001
            [call],
            bundle=execution,
            run_id=run_id,
            config_sha256=config_sha256,
            code_sha256=code_sha256,
        )
        replay = g2.execute_provider_timed_call(hydrated[0])
        semantic = str(reference_rows[0]["semantic_digest"])
        provider_result_sha256 = str(
            reference_rows[0]["provider_result_sha256"]
        )
        if (
            replay.get("semantic_digest") != semantic
            or replay.get("provider_result_sha256") != provider_result_sha256
            or replay.get("provider_success") is not True
            or replay.get("route_l2_valid") is not True
            or replay.get("timing_contract_id") != TIMING_CONTRACT_ID
            or any(
                row.get("semantic_digest") != semantic
                or row.get("provider_result_sha256")
                != provider_result_sha256
                or row.get("provider_success") is not True
                or row.get("route_l2_valid") is not True
                for row in reference_rows
            )
        ):
            raise G3Blocked("g3_interface_replay_semantic_drift")
        timing = {
            field: _require_exact_int(
                replay.get(field),
                "g3_timing_contract",
            )
            for field in TIMING_COMPONENT_FIELDS
        }
        timing["total_ns"] = _require_exact_int(
            replay.get("total_ns"),
            "g3_timing_contract",
        )
        output.append(
            {
                "row_kind": "g3_interface_replay",
                "schema_version": "mid-dual-g3-interface-replay/v1",
                "scale_profile": SCALE_PROFILE,
                "run_id": run_id,
                "replay_id": selection["replay_id"],
                "platform": selection["platform"],
                "request_id": selection["request_id"],
                "request_sha256": selection["provider_request_sha256"],
                "g2_reference_call_ids": [
                    row["call_id"] for row in reference_rows
                ],
                "g2_semantic_digest": semantic,
                "replay_semantic_digest": replay["semantic_digest"],
                "timing_contract_id": TIMING_CONTRACT_ID,
                "formal_input_eligible": True,
                **timing,
                "execution_class": "formal",
                "formal_sample": True,
                "config_sha256": config_sha256,
                "input_sha256": input_sha256,
                "code_sha256": code_sha256,
                "source_sha256": code_sha256,
                "g1_source_manifest_sha256": g1_source_manifest_sha256,
                "g2_source_manifest_sha256": g2_source_manifest_sha256,
                "truth_request_sha256": selection[
                    "truth_request_sha256"
                ],
                "provider_request_sha256": selection[
                    "provider_request_sha256"
                ],
                "provider_result_sha256": provider_result_sha256,
                "approval_sha256": approval_sha256,
                "cohort_sha256": cohort_sha256,
                "provider_identity_sha256": provider_identity_sha256,
                "oracle_identity_sha256": oracle_identity_sha256,
                "hopper_resolution_sha256": hopper_resolution_sha256,
            }
        )
    validate_g3_phase_rows(
        phase_id="p02",
        rows=output,
        accepted_phase_ids=("p01",),
    )
    return output


def deterministic_candidate_sha256(
    *,
    scenario_id: str,
    step_index: int,
    pre_snapshot_sha256: str,
    decision_sha256: str,
    candidate_id: str,
    selected_candidate_cell_xy: Sequence[int],
    selected_theta: float,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "mid-dual-g3-candidate-binding/v1",
            "scenario_id": _require_nonempty(
                scenario_id,
                "g3_wheel_hash_chain",
            ),
            "step_index": _require_exact_int(
                step_index,
                "g3_wheel_hash_chain",
            ),
            "pre_snapshot_sha256": _require_sha256(
                pre_snapshot_sha256,
                "g3_wheel_hash_chain",
            ),
            "decision_sha256": _require_sha256(
                decision_sha256,
                "g3_wheel_hash_chain",
            ),
            "candidate_id": _require_nonempty(
                candidate_id,
                "g3_wheel_hash_chain",
            ),
            "selected_candidate_cell_xy": list(
                _cell(
                    selected_candidate_cell_xy,
                    "g3_wheel_hash_chain",
                )
            ),
            "selected_theta": _theta(
                selected_theta,
                "g3_wheel_hash_chain",
            ),
        }
    )


def deterministic_request_sha256(
    *,
    request_id: str,
    candidate_sha256: str,
    pre_snapshot_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "mid-dual-g3-planner-request-binding/v1",
            "request_id": _require_nonempty(
                request_id,
                "g3_wheel_hash_chain",
            ),
            "candidate_sha256": _require_sha256(
                candidate_sha256,
                "g3_wheel_hash_chain",
            ),
            "pre_snapshot_sha256": _require_sha256(
                pre_snapshot_sha256,
                "g3_wheel_hash_chain",
            ),
        }
    )


def deterministic_route_result_sha256(
    *,
    request_sha256: str,
    planner_success: bool,
    planner_failure_reason: str,
    planned_path_cells: Sequence[Sequence[int]],
    planner_path_length_m: float,
    route_endpoint_cell_xy: Sequence[int],
    route_endpoint_theta: float,
) -> str:
    if not isinstance(planned_path_cells, (list, tuple)):
        raise G3Blocked("g3_wheel_hash_chain")
    path = [
        list(_cell(value, "g3_wheel_hash_chain"))
        for value in planned_path_cells
    ]
    return _canonical_sha256(
        {
            "schema_version": "mid-dual-g3-route-result-binding/v1",
            "request_sha256": _require_sha256(
                request_sha256,
                "g3_wheel_hash_chain",
            ),
            "planner_success": _require_exact_bool(
                planner_success,
                "g3_wheel_hash_chain",
            ),
            "planner_failure_reason": _require_nonempty(
                planner_failure_reason,
                "g3_wheel_hash_chain",
            ),
            "planned_path_cells": path,
            "planner_path_length_m": _finite_number(
                planner_path_length_m,
                "g3_wheel_hash_chain",
                minimum=0.0,
            ),
            "route_endpoint_cell_xy": list(
                _cell(
                    route_endpoint_cell_xy,
                    "g3_wheel_hash_chain",
                )
            ),
            "route_endpoint_theta": _theta(
                route_endpoint_theta,
                "g3_wheel_hash_chain",
            ),
        }
    )


def deterministic_feedback_sha256(
    *,
    route_result_sha256: str,
    feedback_pose_cell_xy: Sequence[int],
    feedback_theta: float,
    coverage: float,
    safety_violation_count: int,
    masked_action_count: int,
    post_snapshot_sha256: str | None = None,
) -> str:
    payload: dict[str, object] = {
            "schema_version": "mid-dual-g3-feedback-binding/v1",
            "route_result_sha256": _require_sha256(
                route_result_sha256,
                "g3_wheel_hash_chain",
            ),
            "feedback_pose_cell_xy": list(
                _cell(
                    feedback_pose_cell_xy,
                    "g3_wheel_hash_chain",
                )
            ),
            "feedback_theta": _theta(
                feedback_theta,
                "g3_wheel_hash_chain",
            ),
            "coverage": _finite_number(
                coverage,
                "g3_wheel_hash_chain",
                minimum=0.0,
                maximum=1.0,
            ),
            "safety_violation_count": _require_exact_int(
                safety_violation_count,
                "g3_wheel_hash_chain",
            ),
            "masked_action_count": _require_exact_int(
                masked_action_count,
                "g3_wheel_hash_chain",
            ),
        }
    if post_snapshot_sha256 is not None:
        payload["schema_version"] = "mid-dual-g3-feedback-binding/v2"
        payload["post_snapshot_sha256"] = _require_sha256(
            post_snapshot_sha256,
            "g3_wheel_hash_chain",
        )
    return _canonical_sha256(payload)


def validate_g3_manifest(
    frozen_manifest: Mapping[str, object],
) -> dict[str, tuple[str, ...]]:
    """只从冻结清单取得 5+5；禁止从 G1/G2 结果选择场景。"""

    if (
        not isinstance(frozen_manifest, Mapping)
        or frozen_manifest.get("schema_version")
        != "mid-dual-scenario-freeze/v1"
        or frozen_manifest.get("completion_status") != "complete"
    ):
        raise G3Blocked("g3_frozen_manifest")
    cohorts = frozen_manifest.get("cohorts")
    if not isinstance(cohorts, Mapping):
        raise G3Blocked("g3_frozen_manifest")
    required = {
        "test_q24": (24, "test/"),
        "unseen24": (24, "unseen/"),
        "g3_test_q5": (5, "test/"),
        "g3_unseen5": (5, "unseen/"),
    }
    parsed: dict[str, tuple[str, ...]] = {}
    for name, (count, prefix) in required.items():
        values = cohorts.get(name)
        if (
            not isinstance(values, (list, tuple))
            or len(values) != count
            or len(set(values)) != count
            or any(
                not isinstance(value, str)
                or not value.startswith(prefix)
                for value in values
            )
        ):
            raise G3Blocked("g3_frozen_manifest")
        parsed[name] = tuple(values)
    if (
        not set(parsed["g3_test_q5"]).issubset(parsed["test_q24"])
        or not set(parsed["g3_unseen5"]).issubset(parsed["unseen24"])
        or set(parsed["g3_test_q5"]).intersection(parsed["g3_unseen5"])
    ):
        raise G3Blocked("g3_frozen_manifest")
    return {
        "test_q24": parsed["g3_test_q5"],
        "unseen24": parsed["g3_unseen5"],
    }


def _g1_index(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], Mapping[str, object]]:
    if len(rows) != 48:
        raise G3Blocked("g3_g1_cross_root_join")
    result: dict[tuple[str, str], Mapping[str, object]] = {}
    split_counts = {"test_q24": 0, "unseen24": 0}
    for raw in rows:
        row = _require_exact_keys(raw, _G1_ROW_FIELDS, "g3_g1_cross_root_join")
        if (
            row.get("row_kind") != "coverage_episode"
            or row.get("schema_version")
            != "xunce-mid-dual-g1-coverage-episode/v1"
            or row.get("gate_id") != "g1"
            or row.get("runner_id")
            != "run_xunce_mid_dual_g1_coverage/v1"
            or row.get("scale_profile") != SCALE_PROFILE
        ):
            raise G3Blocked("g3_g1_cross_root_join")
        split = row.get("split")
        if split not in split_counts:
            raise G3Blocked("g3_g1_cross_root_join")
        episode_index = _require_exact_int(
            row.get("episode_index"),
            "g3_g1_cross_root_join",
        )
        if episode_index >= 24 or row.get("lane_id") != f"lane-{episode_index % 8}":
            raise G3Blocked("g3_g1_cross_root_join")
        expected_phase = "p03" if split == "test_q24" else "p04"
        if (
            row.get("phase_id") != expected_phase
            or row.get("phase_name") != split
            or row.get("episode_id")
            != f"{split}-episode-{episode_index:02d}"
        ):
            raise G3Blocked("g3_g1_cross_root_join")
        scenario_id = _require_nonempty(
            row.get("scenario_id"),
            "g3_g1_cross_root_join",
        )
        expected_prefix = "test/" if split == "test_q24" else "unseen/"
        if not scenario_id.startswith(expected_prefix):
            raise G3Blocked("g3_g1_cross_root_join")
        _require_nonempty(row.get("run_id"), "g3_g1_cross_root_join")
        _require_nonempty(row.get("episode_id"), "g3_g1_cross_root_join")
        if (
            row.get("checkpoint_sha256") != UPDATE80_CHECKPOINT_SHA256
            or row.get("policy_state_sha256") != UPDATE80_POLICY_STATE_SHA256
        ):
            raise G3Blocked("g3_g1_update80_binding")
        for name in (
            "source_sha256",
            "config_sha256",
            "input_sha256",
            "code_sha256",
            "scenario_manifest_sha256",
        ):
            _require_sha256(row.get(name), "g3_g1_cross_root_join")
        if row.get("source_sha256") != row.get("code_sha256"):
            raise G3Blocked("g3_g1_cross_root_join")
        if (
            row.get("denominator_source")
            != "reachable_observable_free_highres_cells/v1"
            or row.get("denominator_algorithm")
            != "exact_reachable_safe_pose_range_los/v1"
        ):
            raise G3Blocked("g3_g1_cross_root_join")
        _require_sha256(
            row.get("denominator_sha256"),
            "g3_g1_cross_root_join",
        )
        denominator = _require_exact_int(
            row.get("denominator_cell_count"),
            "g3_g1_cross_root_join",
            minimum=1,
        )
        initial = _require_exact_int(
            row.get("initial_covered_cell_count"),
            "g3_g1_cross_root_join",
        )
        covered = _require_exact_int(
            row.get("final_covered_cell_count"),
            "g3_g1_cross_root_join",
        )
        if initial > covered or covered > denominator:
            raise G3Blocked("g3_g1_cross_root_join")
        _finite_number(
            row.get("elapsed_ms"),
            "g3_g1_cross_root_join",
            minimum=0.0,
        )
        _require_exact_int(row.get("steps_executed"), "g3_g1_cross_root_join")
        _require_nonempty(
            row.get("termination_reason"),
            "g3_g1_cross_root_join",
        )
        if (
            _require_exact_int(
                row.get("safety_violation_count"),
                "g3_g1_cross_root_join",
            )
            != 0
            or _require_exact_int(
                row.get("masked_action_count"),
                "g3_g1_cross_root_join",
            )
            != 0
        ):
            raise G3Blocked("g3_g1_cross_root_join")
        coverage = _finite_number(
            row.get("coverage"),
            "g3_g1_cross_root_join",
            minimum=0.0,
            maximum=1.0,
        )
        if not math.isclose(
            coverage,
            covered / denominator,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise G3Blocked("g3_g1_cross_root_join")
        key = (str(split), scenario_id)
        if key in result:
            raise G3Blocked("g3_g1_cross_root_join")
        result[key] = row
        split_counts[str(split)] += 1
    if split_counts != {"test_q24": 24, "unseen24": 24}:
        raise G3Blocked("g3_g1_cross_root_join")
    return result


def _cell(value: object, reason: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        raise G3Blocked(reason)
    return int(value[0]), int(value[1])


def _theta(value: object, reason: str) -> float:
    return _finite_number(value, reason)


def _evaluate_wheel(
    *,
    selected: Mapping[str, tuple[str, ...]],
    wheel_rows: Sequence[Mapping[str, object]],
    g1_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    g1 = _g1_index(g1_rows)
    if not wheel_rows:
        raise G3Blocked("g3_wheel_episode_matrix")
    episodes: dict[str, list[Mapping[str, object]]] = {}
    timings: list[float] = []
    expected_scenarios = {
        (split, scenario_id)
        for split, scenario_ids in selected.items()
        for scenario_id in scenario_ids
    }
    observed_scenarios: set[tuple[str, str]] = set()
    seen_steps: set[str] = set()
    for raw in wheel_rows:
        expected_fields = (
            _WHEEL_ROW_FIELDS | _FORMAL_WHEEL_EXTRA_FIELDS
            if isinstance(raw, Mapping) and "execution_class" in raw
            else _WHEEL_ROW_FIELDS
        )
        row = _require_exact_keys(raw, expected_fields, "g3_wheel_row_schema")
        if (
            row.get("row_kind") != "g3_wheel_step"
            or row.get("schema_version") != "mid-dual-g3-wheel-step/v1"
            or row.get("scale_profile") != SCALE_PROFILE
        ):
            raise G3Blocked("g3_wheel_row_schema")
        if expected_fields != _WHEEL_ROW_FIELDS and (
            row.get("execution_class") != "formal"
            or row.get("formal_sample") is not True
            or row.get("timing_contract_id") != TIMING_CONTRACT_ID
            or row.get("wheel_platform") != WHEEL_PLATFORM
            or row.get("wheel_profile") != WHEEL_PROFILE
            or row.get("wheel_capability_revision")
            != WHEEL_CAPABILITY_REVISION
            or row.get("checkpoint_sha256") != UPDATE80_CHECKPOINT_SHA256
            or row.get("policy_state_sha256")
            != UPDATE80_POLICY_STATE_SHA256
            or row.get("source_sha256") != row.get("code_sha256")
        ):
            raise G3Blocked("g3_wheel_identity_invalid")
        _require_nonempty(row.get("run_id"), "g3_wheel_row_schema")
        split = row.get("split")
        scenario_id = row.get("scenario_id")
        if (
            split not in selected
            or not isinstance(scenario_id, str)
            or (str(split), scenario_id) not in expected_scenarios
        ):
            raise G3Blocked("g3_wheel_episode_matrix")
        episode_id = _require_nonempty(
            row.get("episode_id"),
            "g3_wheel_episode_matrix",
        )
        episode_index = _require_exact_int(
            row.get("episode_index"),
            "g3_wheel_episode_matrix",
        )
        g1_row = g1.get((str(split), scenario_id))
        if (
            g1_row is None
            or episode_index != g1_row.get("episode_index")
            or row.get("lane_id") != g1_row.get("lane_id")
        ):
            raise G3Blocked("g3_g1_cross_root_join")
        step_id = _require_nonempty(row.get("step_id"), "g3_wheel_step_chain")
        if step_id in seen_steps:
            raise G3Blocked("g3_wheel_step_chain")
        seen_steps.add(step_id)
        _require_exact_int(row.get("step_index"), "g3_wheel_step_chain")
        _require_exact_bool(row.get("is_terminal"), "g3_wheel_step_chain")
        _require_nonempty(row.get("termination_reason"), "g3_wheel_step_chain")
        pre = _require_sha256(row.get("pre_snapshot_sha256"), "g3_request_id")
        expected_request_id = deterministic_request_id(
            scenario_id=scenario_id,
            step_index=int(row["step_index"]),
            pre_snapshot_sha256=pre,
        )
        request_id = _require_nonempty(row.get("request_id"), "g3_request_id")
        if request_id != expected_request_id:
            raise G3Blocked("g3_request_id")
        candidate_sha = _require_sha256(
            row.get("candidate_sha256"),
            "g3_wheel_hash_chain",
        )
        decision_sha = _require_sha256(
            row.get("decision_sha256"),
            "g3_wheel_hash_chain",
        )
        candidate_id = _require_nonempty(
            row.get("candidate_id"),
            "g3_wheel_hash_chain",
        )
        request_sha = _require_sha256(
            row.get("request_sha256"),
            "g3_wheel_hash_chain",
        )
        route_sha = _require_sha256(
            row.get("route_result_sha256"),
            "g3_wheel_hash_chain",
        )
        feedback_sha = _require_sha256(
            row.get("feedback_sha256"),
            "g3_wheel_hash_chain",
        )
        _require_sha256(
            row.get("post_snapshot_sha256"),
            "g3_wheel_hash_chain",
        )
        selected_cell = _cell(
            row.get("selected_candidate_cell_xy"),
            "g3_wheel_endpoint_mismatch",
        )
        planned_path_value = row.get("planned_path_cells")
        if not isinstance(planned_path_value, (list, tuple)):
            raise G3Blocked("g3_wheel_hash_chain")
        planned_path = [
            list(_cell(value, "g3_wheel_hash_chain"))
            for value in planned_path_value
        ]
        planner_length = _finite_number(
            row.get("planner_path_length_m"),
            "g3_wheel_hash_chain",
            minimum=0.0,
        )
        route_endpoint = _cell(
            row.get("route_endpoint_cell_xy"),
            "g3_wheel_endpoint_mismatch",
        )
        feedback_pose = _cell(
            row.get("feedback_pose_cell_xy"),
            "g3_wheel_endpoint_mismatch",
        )
        selected_theta = _theta(
            row.get("selected_theta"),
            "g3_wheel_endpoint_mismatch",
        )
        route_theta = _theta(
            row.get("route_endpoint_theta"),
            "g3_wheel_endpoint_mismatch",
        )
        feedback_theta = _theta(
            row.get("feedback_theta"),
            "g3_wheel_endpoint_mismatch",
        )
        planner_success = _require_exact_bool(
            row.get("planner_success"),
            "g3_planner_failure",
        )
        planner_failure_reason = _require_nonempty(
            row.get("planner_failure_reason"),
            "g3_planner_failure",
        )
        safety_count = _require_exact_int(
            row.get("safety_violation_count"),
            "g3_wheel_integrity",
        )
        masked_count = _require_exact_int(
            row.get("masked_action_count"),
            "g3_wheel_integrity",
        )
        coverage = _finite_number(
            row.get("coverage"),
            "g3_wheel_coverage",
            minimum=0.0,
            maximum=1.0,
        )
        if candidate_sha != deterministic_candidate_sha256(
            scenario_id=scenario_id,
            step_index=int(row["step_index"]),
            pre_snapshot_sha256=pre,
            decision_sha256=decision_sha,
            candidate_id=candidate_id,
            selected_candidate_cell_xy=selected_cell,
            selected_theta=selected_theta,
        ):
            raise G3Blocked("g3_wheel_hash_chain")
        if request_sha != deterministic_request_sha256(
            request_id=request_id,
            candidate_sha256=candidate_sha,
            pre_snapshot_sha256=pre,
        ):
            raise G3Blocked("g3_wheel_hash_chain")
        if route_sha != deterministic_route_result_sha256(
            request_sha256=request_sha,
            planner_success=planner_success,
            planner_failure_reason=planner_failure_reason,
            planned_path_cells=planned_path,
            planner_path_length_m=planner_length,
            route_endpoint_cell_xy=route_endpoint,
            route_endpoint_theta=route_theta,
        ):
            raise G3Blocked("g3_wheel_hash_chain")
        if feedback_sha != deterministic_feedback_sha256(
            route_result_sha256=route_sha,
            feedback_pose_cell_xy=feedback_pose,
            feedback_theta=feedback_theta,
            coverage=coverage,
            safety_violation_count=safety_count,
            masked_action_count=masked_count,
            post_snapshot_sha256=(
                str(row["post_snapshot_sha256"])
                if expected_fields != _WHEEL_ROW_FIELDS
                else None
            ),
        ):
            raise G3Blocked("g3_wheel_hash_chain")
        if (
            row.get("request_candidate_sha256") != candidate_sha
            or row.get("route_request_id") != request_id
            or row.get("route_request_sha256") != request_sha
            or row.get("feedback_request_id") != request_id
            or row.get("feedback_route_result_sha256") != route_sha
            or row.get("post_snapshot_parent_sha256") != feedback_sha
        ):
            raise G3Blocked("g3_wheel_hash_chain")
        if (
            route_endpoint != selected_cell
            or feedback_pose != selected_cell
            or not math.isclose(
                selected_theta,
                route_theta,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                selected_theta,
                feedback_theta,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise G3Blocked("g3_wheel_endpoint_mismatch")
        if (
            planner_success is not True
            or planner_failure_reason != "none"
        ):
            raise G3Blocked("g3_planner_failure")
        mismatch_count = _require_exact_int(
            row.get("candidate_route_mismatch_count"),
            "g3_wheel_integrity",
        )
        if safety_count != 0 or masked_count != 0 or mismatch_count != 0:
            raise G3Blocked("g3_wheel_integrity")
        paired = _finite_number(
            row.get("paired_g1_coverage"),
            "g3_g1_cross_root_join",
            minimum=0.0,
            maximum=1.0,
        )
        g1_coverage = float(g1_row["coverage"])
        if not math.isclose(
            paired,
            g1_coverage,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise G3Blocked("g3_g1_cross_root_join")
        timings.append(_timing_ms(row))
        episodes.setdefault(episode_id, []).append(row)
        observed_scenarios.add((str(split), scenario_id))

    if len(episodes) != 10 or observed_scenarios != expected_scenarios:
        raise G3Blocked("g3_wheel_episode_matrix")
    terminal_rows: list[Mapping[str, object]] = []
    for episode_rows in episodes.values():
        ordered = sorted(
            episode_rows,
            key=lambda row: int(row["step_index"]),
        )
        if [row["step_index"] for row in ordered] != list(range(len(ordered))):
            raise G3Blocked("g3_wheel_step_chain")
        stable = {
            (
                row["scenario_id"],
                row["split"],
                row["episode_index"],
                row["episode_id"],
                row["lane_id"],
            )
            for row in ordered
        }
        terminals = [
            row for row in ordered if row.get("is_terminal") is True
        ]
        if (
            len(stable) != 1
            or len(terminals) != 1
            or terminals[0] is not ordered[-1]
        ):
            raise G3Blocked("g3_wheel_step_chain")
        for left, right in zip(ordered[:-1], ordered[1:], strict=True):
            if left["post_snapshot_sha256"] != right["pre_snapshot_sha256"]:
                raise G3Blocked("g3_wheel_step_chain")
        terminal_rows.append(terminals[0])

    final_coverages = [float(row["coverage"]) for row in terminal_rows]
    deltas = [
        float(row["coverage"]) - float(row["paired_g1_coverage"])
        for row in terminal_rows
    ]
    paired_delta_mean = round(mean(deltas), 15)
    timing = _timing_statistics(timings)
    coverage_mean = mean(final_coverages)
    coverage_80_count = sum(
        value >= MID_COVERAGE_THRESHOLD for value in final_coverages
    )
    coverage_99_count = sum(
        value >= FINAL_COVERAGE_THRESHOLD for value in final_coverages
    )
    coverage_all_episodes_passed = coverage_80_count == 10
    coverage_mean_passed = coverage_mean >= MID_COVERAGE_THRESHOLD
    paired_g1_delta_passed = paired_delta_mean >= PAIRED_G1_DELTA_MIN
    return {
        "wheel_episode_count": 10,
        "wheel_step_count": len(wheel_rows),
        "coverage_mean": coverage_mean,
        "coverage_80_count": coverage_80_count,
        "coverage_99_count": coverage_99_count,
        "paired_g1_coverage_delta_mean": paired_delta_mean,
        "coverage_all_episodes_passed": coverage_all_episodes_passed,
        "coverage_mean_passed": coverage_mean_passed,
        "paired_g1_delta_passed": paired_g1_delta_passed,
        "timing": timing,
        "midterm_reduced_passed": (
            coverage_mean_passed
            and coverage_all_episodes_passed
            and paired_g1_delta_passed
            and timing["midterm_reduced_passed"] is True
        ),
        "final_threshold_reduced_passed": (
            coverage_mean >= FINAL_COVERAGE_THRESHOLD
            and coverage_99_count == 10
            and paired_delta_mean >= PAIRED_G1_DELTA_MIN
            and timing["final_threshold_reduced_passed"] is True
        ),
    }


def execute_update80_wheel_rows(
    *,
    catalog: object,
    frozen_manifest: object,
    raw_frozen_manifest: Mapping[str, object],
    g1_rows: Sequence[Mapping[str, object]],
    safety_contract: object,
    config_sha256: str,
    run_id: str,
    evaluation_seed_start: int,
    policy: object | None = None,
) -> list[dict[str, object]]:
    """顺序重跑冻结 5+5 update80 episode，并保留每个 planner step。

    该路径刻意不调用 Standard 的 16/24/64 调度器；G3 的正式样本恰为十个
    episode，不通过补跑额外场景来凑 worker 数。
    """

    import numpy as np
    import torch
    from torch import nn

    from lunar_exploration_ppo.env.standard_training import (
        StandardEvaluationEnv,
    )
    from lunar_exploration_ppo.eval import standard
    from lunar_exploration_ppo.eval.midterm_reduced import (
        load_midterm_update80_policy,
    )
    from lunar_exploration_ppo.policy.cross_attention import (
        batch_policy_observations,
    )

    selected = validate_g3_manifest(raw_frozen_manifest)
    g1 = _g1_index(g1_rows)
    records = {
        f"{record.scenario_id}/standard-proxy/v1": record
        for record in getattr(catalog, "records", ())
    }
    if set(selected["test_q24"] + selected["unseen24"]) - set(records):
        raise G3Blocked("g3_frozen_manifest_catalog_join")
    frozen_cohorts = getattr(frozen_manifest, "cohorts", None)
    if (
        not isinstance(frozen_cohorts, Mapping)
        or tuple(frozen_cohorts.get("g3_test_q5", ()))
        != selected["test_q24"]
        or tuple(frozen_cohorts.get("g3_unseen5", ()))
        != selected["unseen24"]
    ):
        raise G3Blocked("g3_frozen_manifest_catalog_join")
    if policy is None:
        policy = load_midterm_update80_policy()
    if not isinstance(policy, nn.Module):
        raise G3Blocked("g3_update80_policy_invalid")

    sequence_index = 0
    output_rows: list[dict[str, object]] = []
    original_training = policy.training
    policy.eval()
    try:
        for split in ("test_q24", "unseen24"):
            for scenario_id in selected[split]:
                g1_row = g1.get((split, scenario_id))
                if g1_row is None:
                    raise G3Blocked("g3_g1_cross_root_join")
                episode_index = int(g1_row["episode_index"])
                record = records[scenario_id]
                expected_source_split = (
                    "test" if split == "test_q24" else "unseen"
                )
                if record.split != expected_source_split:
                    raise G3Blocked("g3_frozen_manifest_catalog_join")
                environment = StandardEvaluationEnv(
                    catalog,
                    split=record.split,
                    scenario_ids=(record.scenario_id,),
                    safety_contract=safety_contract,
                    config_sha256=config_sha256,
                )
                try:
                    observation = environment.reset()
                    if environment.is_done or not environment.needs_policy:
                        raise G3Blocked("g3_wheel_reset_terminal")
                    episode_id = (
                        f"g3-{split}-episode-{episode_index:02d}"
                    )
                    step_index = 0
                    while not environment.is_done and step_index < 128:
                        batch = batch_policy_observations(
                            (observation,),
                            device="cuda",
                        )
                        with torch.inference_mode():
                            policy_output = policy(batch)
                        action = standard.select_standard_evaluation_actions(
                            method="ppo_policy",
                            observations=(observation,),
                            evaluation_seeds=(
                                evaluation_seed_start
                                + episode_index * 128
                                + step_index,
                            ),
                            policy_output=policy_output,
                        )[0]
                        decision = standard.build_standard_decision_record(
                            method="ppo_policy",
                            sequence_index=sequence_index,
                            episode_index=episode_index,
                            step_index=step_index,
                            worker_index=episode_index % 8,
                            observation=observation,
                            action=action,
                            selector_inputs={
                                "observation_batch": batch,
                                "policy_output": policy_output,
                            },
                            policy_batch_row_index=0,
                        )
                        sequence_index += 1
                        valid_indices = tuple(
                            int(value)
                            for value in np.flatnonzero(
                                np.asarray(
                                    observation.candidate_mask,
                                    dtype=bool,
                                )
                            )
                        )
                        try:
                            selected_rank = valid_indices.index(
                                action.candidate_index
                            )
                            selected_cell = environment.frontier_cells[
                                selected_rank
                            ]
                        except (IndexError, ValueError) as exc:
                            raise G3Blocked(
                                "g3_selected_candidate_invalid"
                            ) from exc
                        pre_snapshot_sha256 = _require_sha256(
                            decision.get("policy_observation_sha256"),
                            "g3_wheel_hash_chain",
                        )
                        decision_sha256 = _require_sha256(
                            decision.get("decision_sha256"),
                            "g3_wheel_hash_chain",
                        )
                        candidate_id = (
                            f"g3-candidate:{split}:{episode_index:02d}:"
                            f"{step_index:03d}:{selected_cell.x}:"
                            f"{selected_cell.y}"
                        )
                        selected_cell_xy = [
                            int(selected_cell.x),
                            int(selected_cell.y),
                        ]
                        candidate_sha256 = deterministic_candidate_sha256(
                            scenario_id=scenario_id,
                            step_index=step_index,
                            pre_snapshot_sha256=pre_snapshot_sha256,
                            decision_sha256=decision_sha256,
                            candidate_id=candidate_id,
                            selected_candidate_cell_xy=selected_cell_xy,
                            selected_theta=float(action.target_theta),
                        )
                        request_id = deterministic_request_id(
                            scenario_id=scenario_id,
                            step_index=step_index,
                            pre_snapshot_sha256=pre_snapshot_sha256,
                        )
                        request_sha256 = deterministic_request_sha256(
                            request_id=request_id,
                            candidate_sha256=candidate_sha256,
                            pre_snapshot_sha256=pre_snapshot_sha256,
                        )

                        step = environment.step(action)
                        diagnostics = step.diagnostics.planner
                        if not isinstance(diagnostics, Mapping):
                            raise G3Blocked("g3_timing_contract")
                        timing = {
                            name: _require_exact_int(
                                diagnostics.get(name),
                                "g3_timing_contract",
                            )
                            for name in TIMING_COMPONENT_FIELDS
                        }
                        timing["total_ns"] = _require_exact_int(
                            diagnostics.get("total_ns"),
                            "g3_timing_contract",
                        )
                        if timing["total_ns"] != sum(
                            timing[name]
                            for name in TIMING_COMPONENT_FIELDS
                        ):
                            raise G3Blocked("g3_timing_contract")
                        planned_path_cells = [
                            [int(cell.x), int(cell.y)]
                            for cell in step.diagnostics.planned_path_cells
                        ]
                        planner_failure_reason = str(
                            diagnostics.get("failure_reason", "unknown")
                        )
                        planner_success = (
                            planner_failure_reason == "none"
                            and not step.diagnostics.invalid_action
                            and bool(planned_path_cells)
                        )
                        route_endpoint = (
                            planned_path_cells[-1]
                            if planned_path_cells
                            else selected_cell_xy
                        )
                        route_theta = float(action.target_theta)
                        route_result_sha256 = (
                            deterministic_route_result_sha256(
                                request_sha256=request_sha256,
                                planner_success=planner_success,
                                planner_failure_reason=(
                                    planner_failure_reason
                                ),
                                planned_path_cells=planned_path_cells,
                                planner_path_length_m=float(
                                    step.diagnostics.path_length_m
                                ),
                                route_endpoint_cell_xy=route_endpoint,
                                route_endpoint_theta=route_theta,
                            )
                        )
                        feedback_pose = [
                            int(environment.pose.cell.x),
                            int(environment.pose.cell.y),
                        ]
                        feedback_theta = float(environment.pose.theta)
                        safety_count = int(
                            step.diagnostics.safety_violation
                        )
                        masked_count = int(
                            step.diagnostics.invalid_action
                        )
                        mismatch_count = int(
                            route_endpoint != selected_cell_xy
                            or feedback_pose != selected_cell_xy
                            or not math.isclose(
                                route_theta,
                                float(action.target_theta),
                                rel_tol=0.0,
                                abs_tol=1.0e-12,
                            )
                            or not math.isclose(
                                feedback_theta,
                                float(action.target_theta),
                                rel_tol=0.0,
                                abs_tol=1.0e-12,
                            )
                        )
                        coverage = float(step.coverage_rate)
                        feedback_sha256 = deterministic_feedback_sha256(
                            route_result_sha256=route_result_sha256,
                            feedback_pose_cell_xy=feedback_pose,
                            feedback_theta=feedback_theta,
                            coverage=coverage,
                            safety_violation_count=safety_count,
                            masked_action_count=masked_count,
                        )
                        post_snapshot_sha256 = _require_sha256(
                            standard._policy_observation_sha256(  # noqa: SLF001
                                step.observation
                            ),
                            "g3_wheel_hash_chain",
                        )
                        output_rows.append(
                            {
                                "row_kind": "g3_wheel_step",
                                "schema_version": (
                                    "mid-dual-g3-wheel-step/v1"
                                ),
                                "scale_profile": SCALE_PROFILE,
                                "run_id": run_id,
                                "split": split,
                                "episode_index": episode_index,
                                "episode_id": episode_id,
                                "scenario_id": scenario_id,
                                "lane_id": g1_row["lane_id"],
                                "step_id": (
                                    f"{episode_id}:step:{step_index:03d}"
                                ),
                                "step_index": step_index,
                                "is_terminal": bool(step.done),
                                "termination_reason": (
                                    str(step.reason)
                                    if step.done
                                    else "none"
                                ),
                                "decision_sha256": decision_sha256,
                                "candidate_id": candidate_id,
                                "candidate_sha256": candidate_sha256,
                                "request_candidate_sha256": (
                                    candidate_sha256
                                ),
                                "selected_candidate_cell_xy": (
                                    selected_cell_xy
                                ),
                                "planned_path_cells": planned_path_cells,
                                "planner_path_length_m": float(
                                    step.diagnostics.path_length_m
                                ),
                                "route_endpoint_cell_xy": route_endpoint,
                                "feedback_pose_cell_xy": feedback_pose,
                                "selected_theta": float(
                                    action.target_theta
                                ),
                                "route_endpoint_theta": route_theta,
                                "feedback_theta": feedback_theta,
                                "pre_snapshot_sha256": (
                                    pre_snapshot_sha256
                                ),
                                "request_id": request_id,
                                "request_sha256": request_sha256,
                                "route_request_id": request_id,
                                "route_request_sha256": request_sha256,
                                "route_result_sha256": (
                                    route_result_sha256
                                ),
                                "feedback_request_id": request_id,
                                "feedback_route_result_sha256": (
                                    route_result_sha256
                                ),
                                "feedback_sha256": feedback_sha256,
                                "post_snapshot_parent_sha256": (
                                    feedback_sha256
                                ),
                                "post_snapshot_sha256": (
                                    post_snapshot_sha256
                                ),
                                "coverage": coverage,
                                "paired_g1_coverage": float(
                                    g1_row["coverage"]
                                ),
                                "safety_violation_count": safety_count,
                                "masked_action_count": masked_count,
                                "candidate_route_mismatch_count": (
                                    mismatch_count
                                ),
                                "planner_success": planner_success,
                                "planner_failure_reason": (
                                    planner_failure_reason
                                ),
                                **timing,
                            }
                        )
                        observation = step.observation
                        step_index += 1
                    if not environment.is_done:
                        raise G3Blocked(
                            "g3_wheel_episode_did_not_terminate"
                        )
                    episode_rows = [
                        row
                        for row in output_rows
                        if row["episode_id"] == episode_id
                    ]
                    if (
                        not episode_rows
                        or episode_rows[-1]["is_terminal"] is not True
                    ):
                        raise G3Blocked("g3_wheel_step_chain")
                finally:
                    environment.close()
    finally:
        policy.train(original_training)

    _evaluate_wheel(
        selected=selected,
        wheel_rows=output_rows,
        g1_rows=g1_rows,
    )
    return output_rows


def _g2_call_index(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise G3Blocked("g3_g2_cross_root_join")
        if set(raw) == set(_G2_REFERENCE_FIELDS):
            row = dict(raw)
        else:
            g2 = _import_local_script(
                "run_xunce_mid_dual_g2_planning_time"
            )

            if set(raw) != set(g2.G2_ROW_KEYS):
                raise G3Blocked("g3_g2_cross_root_join")
            row = {field: raw[field] for field in _G2_REFERENCE_FIELDS}
        if (
            row.get("row_kind") != "g2_planning_call"
            or row.get("platform") not in {"wheel", "legged", "hopper"}
        ):
            raise G3Blocked("g3_g2_cross_root_join")
        call_id = _require_nonempty(
            row.get("call_id"),
            "g3_g2_cross_root_join",
        )
        if call_id in result:
            raise G3Blocked("g3_g2_cross_root_join")
        _require_nonempty(row.get("request_id"), "g3_g2_cross_root_join")
        _require_sha256(row.get("request_sha256"), "g3_g2_cross_root_join")
        _require_sha256(row.get("semantic_digest"), "g3_g2_cross_root_join")
        repeat = _require_exact_int(
            row.get("repeat_index"),
            "g3_g2_cross_root_join",
        )
        if repeat >= 5:
            raise G3Blocked("g3_g2_cross_root_join")
        if (
            _require_exact_bool(
                row.get("provider_success"),
                "g3_g2_cross_root_join",
            )
            is not True
            or _require_exact_bool(
                row.get("route_l2_valid"),
                "g3_g2_cross_root_join",
            )
            is not True
            or row.get("timing_contract_id") != TIMING_CONTRACT_ID
        ):
            raise G3Blocked("g3_g2_cross_root_join")
        result[call_id] = row
    return result


def _evaluate_interface(
    *,
    interface_rows: Sequence[Mapping[str, object]],
    g2_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(interface_rows) != 6:
        raise G3Blocked("g3_interface_request_matrix")
    calls = _g2_call_index(g2_rows)
    replay_ids: set[str] = set()
    request_keys: set[tuple[str, str, str]] = set()
    platform_counts = {"legged": 0, "hopper": 0}
    timings: list[float] = []
    for raw in interface_rows:
        expected_fields = (
            _INTERFACE_ROW_FIELDS | _FORMAL_INTERFACE_EXTRA_FIELDS
            if isinstance(raw, Mapping) and "execution_class" in raw
            else _INTERFACE_ROW_FIELDS
        )
        row = _require_exact_keys(
            raw,
            expected_fields,
            "g3_interface_row_schema",
        )
        if (
            row.get("row_kind") != "g3_interface_replay"
            or row.get("schema_version")
            != "mid-dual-g3-interface-replay/v1"
            or row.get("scale_profile") != SCALE_PROFILE
        ):
            raise G3Blocked("g3_interface_row_schema")
        if expected_fields != _INTERFACE_ROW_FIELDS and (
            row.get("execution_class") != "formal"
            or row.get("formal_sample") is not True
            or row.get("source_sha256") != row.get("code_sha256")
        ):
            raise G3Blocked("g3_formal_fixture_partition")
        _require_nonempty(row.get("run_id"), "g3_interface_row_schema")
        replay_id = _require_nonempty(
            row.get("replay_id"),
            "g3_interface_request_matrix",
        )
        if replay_id in replay_ids:
            raise G3Blocked("g3_interface_request_matrix")
        replay_ids.add(replay_id)
        platform = row.get("platform")
        if platform not in platform_counts:
            raise G3Blocked("g3_interface_request_matrix")
        request_id = _require_nonempty(
            row.get("request_id"),
            "g3_interface_request_matrix",
        )
        request_sha = _require_sha256(
            row.get("request_sha256"),
            "g3_interface_request_matrix",
        )
        key = (str(platform), request_id, request_sha)
        if key in request_keys:
            raise G3Blocked("g3_interface_request_matrix")
        request_keys.add(key)
        references = row.get("g2_reference_call_ids")
        if (
            not isinstance(references, (list, tuple))
            or len(references) != 5
            or len(set(references)) != 5
            or any(not isinstance(item, str) or not item for item in references)
        ):
            raise G3Blocked("g3_g2_cross_root_join")
        reference_rows = [calls.get(str(call_id)) for call_id in references]
        if any(reference is None for reference in reference_rows):
            raise G3Blocked("g3_g2_cross_root_join")
        typed_references = [
            reference for reference in reference_rows if reference is not None
        ]
        if (
            {reference["platform"] for reference in typed_references}
            != {platform}
            or {reference["request_id"] for reference in typed_references}
            != {request_id}
            or {reference["request_sha256"] for reference in typed_references}
            != {request_sha}
            or {reference["repeat_index"] for reference in typed_references}
            != set(range(5))
            or len(
                {
                    reference["semantic_digest"]
                    for reference in typed_references
                }
            )
            != 1
        ):
            raise G3Blocked("g3_g2_cross_root_join")
        semantic = typed_references[0]["semantic_digest"]
        if (
            row.get("g2_semantic_digest") != semantic
            or row.get("replay_semantic_digest") != semantic
            or row.get("timing_contract_id") != TIMING_CONTRACT_ID
        ):
            raise G3Blocked("g3_g2_cross_root_join")
        eligible = _require_exact_bool(
            row.get("formal_input_eligible"),
            "g3_interface_formal_input",
        )
        if not eligible:
            if platform == "hopper":
                raise G3Blocked("g3_hopper_formal_input_ineligible")
            raise G3Blocked("g3_interface_formal_input_ineligible")
        timings.append(_timing_ms(row))
        platform_counts[str(platform)] += 1
    if platform_counts != {"legged": 3, "hopper": 3}:
        raise G3Blocked("g3_interface_request_matrix")
    timing = _timing_statistics(timings)
    return {
        "interface_replay_count": 6,
        "platform_counts": platform_counts,
        "cross_root_join_passed": True,
        "interface_correctness_passed": True,
        "required_replay_identities_passed": True,
        "timing": timing,
        "midterm_reduced_passed": timing["midterm_reduced_passed"],
        "final_threshold_reduced_passed": timing[
            "final_threshold_reduced_passed"
        ],
    }


def evaluate_g3_evidence(
    *,
    frozen_manifest: Mapping[str, object],
    wheel_rows: Sequence[Mapping[str, object]],
    g1_rows: Sequence[Mapping[str, object]],
    interface_rows: Sequence[Mapping[str, object]],
    g2_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """从冻结清单和三组 raw rows 独立复算 G3；不信任 stored pass 字段。"""

    try:
        selected = validate_g3_manifest(frozen_manifest)
        wheel = _evaluate_wheel(
            selected=selected,
            wheel_rows=wheel_rows,
            g1_rows=g1_rows,
        )
        interface = _evaluate_interface(
            interface_rows=interface_rows,
            g2_rows=g2_rows,
        )
    except G3Blocked as exc:
        return {
            "schema_version": "mid-dual-g3-summary/v1",
            "scale_profile": SCALE_PROFILE,
            "active_platforms": list(ACTIVE_PLATFORMS),
            "platform_invariants": PLATFORM_INVARIANTS,
            "full_scale_acceptance": False,
            "status": "blocked",
            "formal_evidence_eligible": False,
            "blockers": [exc.reason],
            "g3_midterm_crosscheck_passed": False,
            "g3_final_crosscheck_passed": False,
        }
    midterm = (
        wheel["midterm_reduced_passed"] is True
        and interface["midterm_reduced_passed"] is True
    )
    final = (
        wheel["final_threshold_reduced_passed"] is True
        and interface["final_threshold_reduced_passed"] is True
    )
    return {
        "schema_version": "mid-dual-g3-summary/v1",
        "scale_profile": SCALE_PROFILE,
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": PLATFORM_INVARIANTS,
        "full_scale_acceptance": False,
        "status": "passed" if midterm else "failed",
        "formal_evidence_eligible": True,
        "blockers": [],
        "g3_midterm_crosscheck_passed": midterm,
        "g3_final_crosscheck_passed": final,
        "selected_scenarios": {
            split: list(values) for split, values in selected.items()
        },
        "wheel": wheel,
        "interface": interface,
    }


def _build_g3_input_audit(
    *,
    run_id: str,
    frozen: Mapping[str, object],
    g1_source: Mapping[str, object],
    g2_source: Mapping[str, object],
) -> dict[str, object]:
    selected = frozen.get("selected")
    g1_rows = g1_source.get("coverage_rows")
    cohort = g2_source.get("cohort")
    cohort_calls = g2_source.get("cohort_calls")
    g1_invariants = g1_source.get("platform_invariants")
    g2_invariants = g2_source.get("platform_invariants")
    g1_evidence_binding = _g1_source_evidence_binding(g1_source)
    if (
        not isinstance(selected, Mapping)
        or not isinstance(g1_rows, list)
        or not isinstance(cohort, Mapping)
        or not isinstance(cohort.get("rows"), list)
        or not isinstance(cohort_calls, list)
    ):
        raise G3Blocked("g3_input_audit_source_invalid")
    validate_platform_invariants(
        ["wheel"],
        g1_invariants,
        expected_platforms=("wheel",),
    )
    validate_platform_invariants(
        ACTIVE_PLATFORMS,
        g2_invariants,
    )
    g1_index = _g1_index(g1_rows)
    wheel_selections: list[dict[str, object]] = []
    for split in ("test_q24", "unseen24"):
        scenario_ids = selected.get(split)
        if not isinstance(scenario_ids, tuple) or len(scenario_ids) != 5:
            raise G3Blocked("g3_input_audit_source_invalid")
        for scenario_id in scenario_ids:
            source = g1_index.get((split, str(scenario_id)))
            if source is None:
                raise G3Blocked("g3_g1_cross_root_join")
            episode_index = _require_exact_int(
                source.get("episode_index"),
                "g3_g1_cross_root_join",
            )
            wheel_selections.append(
                {
                    "split": split,
                    "scenario_id": scenario_id,
                    "g1_episode_id": source["episode_id"],
                    "g1_episode_index": episode_index,
                    "lane_id": source["lane_id"],
                    "wheel_episode_id": (
                        f"g3-{split}-episode-{episode_index:02d}"
                    ),
                }
            )
    calls_by_key: dict[
        tuple[str, str, str],
        list[Mapping[str, object]],
    ] = {}
    for raw in cohort_calls:
        if not isinstance(raw, Mapping):
            raise G3Blocked("g3_g2_cohort_call_join_invalid")
        key = (
            str(raw.get("platform")),
            str(raw.get("request_id")),
            str(raw.get("request_sha256")),
        )
        calls_by_key.setdefault(key, []).append(raw)
    interface_selections: list[dict[str, object]] = []
    for raw in cohort["rows"]:
        if not isinstance(raw, Mapping):
            raise G3Blocked("g3_g2_cohort_call_join_invalid")
        key = (
            str(raw.get("platform")),
            str(raw.get("request_id")),
            str(raw.get("provider_request_sha256")),
        )
        references = sorted(
            calls_by_key.get(key, []),
            key=lambda row: int(row["repeat_index"]),
        )
        if (
            len(references) != 5
            or [row.get("repeat_index") for row in references]
            != list(range(5))
            or len({row.get("semantic_digest") for row in references}) != 1
        ):
            raise G3Blocked("g3_g2_cohort_call_join_invalid")
        interface_selections.append(
            {
                "platform": raw["platform"],
                "replay_id": raw["replay_id"],
                "g2_reference_call_ids": [
                    row["call_id"] for row in references
                ],
                "request_id": raw["request_id"],
                "request_sha256": raw["provider_request_sha256"],
                "g2_semantic_digest": references[0]["semantic_digest"],
                "timing_contract_id": TIMING_CONTRACT_ID,
            }
        )
    if (
        len(wheel_selections) != 10
        or len(interface_selections) != 6
        or Counter(
            str(row["platform"]) for row in interface_selections
        )
        != Counter({"legged": 3, "hopper": 3})
    ):
        raise G3Blocked("g3_input_audit_count_invalid")
    return {
        "schema_version": G3_INPUT_AUDIT_SCHEMA_VERSION,
        "gate_id": "g3",
        "run_id": run_id,
        "scale_profile": SCALE_PROFILE,
        "formal_evidence_eligible": True,
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": PLATFORM_INVARIANTS,
        "g1_source_manifest_sha256": g1_source["manifest_sha256"],
        "g2_source_manifest_sha256": g2_source["manifest_sha256"],
        "g1_evidence_binding": g1_evidence_binding,
        "wheel_selections": wheel_selections,
        "interface_selections": interface_selections,
    }


def _build_upstream_binding(
    *,
    frozen: Mapping[str, object],
    g1_source: Mapping[str, object],
    g2_source: Mapping[str, object],
) -> dict[str, object]:
    g1_config = g1_source.get("config")
    g2_config = g2_source.get("config")
    g2_input = g2_source.get("input_audit")
    cohort = g2_source.get("cohort")
    if (
        not isinstance(g1_config, Mapping)
        or not isinstance(g2_config, Mapping)
        or not isinstance(g2_input, Mapping)
        or not isinstance(cohort, Mapping)
    ):
        raise G3Blocked("g3_upstream_binding_invalid")
    return _validate_upstream_binding(
        {
            "g1_source_manifest_sha256": g1_source["manifest_sha256"],
            "g2_source_manifest_sha256": g2_source["manifest_sha256"],
            "freeze_manifest_sha256": frozen["manifest_sha256"],
            "g1_config_sha256": g1_config["config_sha256"],
            "g1_input_sha256": g1_config["input_sha256"],
            "g1_code_sha256": g1_config["code_sha256"],
            "g2_config_sha256": g2_config["config_sha256"],
            "g2_input_sha256": g2_config["input_sha256"],
            "g2_code_sha256": g2_config["code_sha256"],
            "g2_input_set_id": g2_input["input_set_id"],
            "g2_approval_sha256": _canonical_sha256(
                g2_source["approval"]
            ),
            "g2_cohort_sha256": cohort["cohort_sha256"],
            "g2_provider_identity_sha256": _canonical_sha256(
                g2_source["provider_source"]
            ),
            "g2_oracle_identity_sha256": _canonical_sha256(
                g2_source["oracle_source"]
            ),
            "g2_hopper_resolution_sha256": _canonical_sha256(
                g2_source["hopper_resolution"]
            ),
            "active_platforms": list(ACTIVE_PLATFORMS),
            "platform_invariants": PLATFORM_INVARIANTS,
            "g1_evidence_binding": _g1_source_evidence_binding(g1_source),
        }
    )


def _build_upstream_lineage_audit(
    *,
    upstream_binding: Mapping[str, object],
    frozen: Mapping[str, object],
    g1_source: Mapping[str, object],
    g2_source: Mapping[str, object],
) -> dict[str, object]:
    g1_binding = _g1_source_evidence_binding(g1_source)
    g1_projection = {
        "root": str(g1_source["root"]).replace("\\", "/"),
        "evidence_root": str(
            g1_source.get("evidence_root", g1_source["root"])
        ).replace("\\", "/"),
        "manifest_sha256": g1_source["manifest_sha256"],
        "config_sha256": g1_source["config"]["config_sha256"],
        "input_sha256": g1_source["config"]["input_sha256"],
        "code_sha256": g1_source["config"]["code_sha256"],
        "coverage_row_count": len(g1_source["coverage_rows"]),
        "trace_row_count": g1_source["trace_row_count"],
        "g1_evidence_binding": g1_binding,
        "platform_invariants": dict(
            g1_source["platform_invariants"]
        ),
        "platform_invariants_source_sha256": g1_source[
            "platform_invariants_source_sha256"
        ],
    }
    if g1_binding["g1_evidence_kind"] == G1_NATIVE_EVIDENCE_KIND:
        g1_projection["native_summary_sha256"] = _canonical_sha256(
            g1_source["native_summary"]
        )
    else:
        review_metrics = g1_source.get("review_metrics")
        if not isinstance(review_metrics, Mapping):
            raise G3Blocked("g3_g1_existing_run_assessment_invalid")
        g1_projection["review_metrics_sha256"] = _canonical_sha256(
            review_metrics
        )
    return {
        "schema_version": "xunce-mid-dual-g3-upstream-lineage/v1",
        "gate_id": "g3",
        "formal_evidence_eligible": True,
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": PLATFORM_INVARIANTS,
        "upstream_binding": dict(upstream_binding),
        "freeze": {
            "root": str(frozen["root"]).replace("\\", "/"),
            "manifest_sha256": frozen["manifest_sha256"],
            "selected": {
                name: list(values)
                for name, values in frozen["selected"].items()
            },
        },
        "g1": g1_projection,
        "g2": {
            "root": str(g2_source["root"]).replace("\\", "/"),
            "execution_root": str(
                g2_source["execution_root"]
            ).replace("\\", "/"),
            "manifest_sha256": g2_source["manifest_sha256"],
            "config_sha256": g2_source["config"]["config_sha256"],
            "input_sha256": g2_source["config"]["input_sha256"],
            "code_sha256": g2_source["config"]["code_sha256"],
            "formal_row_count": len(g2_source["formal_rows"]),
            "input_set_id": g2_source["input_audit"]["input_set_id"],
            "approval": dict(g2_source["approval"]),
            "cohort": dict(g2_source["cohort"]),
            "provider_source": dict(g2_source["provider_source"]),
            "oracle_source": dict(g2_source["oracle_source"]),
            "hopper_resolution": dict(g2_source["hopper_resolution"]),
            "platform_invariants": dict(
                g2_source["platform_invariants"]
            ),
            "platform_invariants_audit_sha256": g2_source[
                "platform_invariants_audit_sha256"
            ],
            "path_planner_runtime_source_closure_sha256": g2_source[
                "path_planner_runtime_source_closure_sha256"
            ],
            "runtime_source_closure_audit_sha256": g2_source[
                "runtime_source_closure_audit_sha256"
            ],
        },
    }


def _accepted_g3_phase_rows(
    store: MidDualRunStore,
    phase_id: str,
) -> list[dict[str, object]]:
    states = artifact_io.read_jsonl(
        artifact_path(store.run_root, MID_DUAL_PHASE_STATE)
    )
    selected = next(
        (row for row in states if row.get("phase_id") == phase_id),
        None,
    )
    if selected is None or type(selected.get("rows_path")) is not str:
        raise G3Blocked("g3_accepted_phase_missing")
    rows_path = _safe_manifest_relative_path(
        selected["rows_path"],
        "g3_accepted_phase_missing",
    )
    return artifact_io.read_jsonl(store.run_root / rows_path)


def _accepted_g3_phase_audit(
    store: MidDualRunStore,
    phase_id: str,
) -> dict[str, object]:
    attempts = artifact_io.read_jsonl(
        store.run_root / "phase-attempts.jsonl"
    )
    selected = next(
        (
            row
            for row in attempts
            if row.get("phase_id") == phase_id
            and row.get("status") == "accepted"
        ),
        None,
    )
    if selected is None or type(selected.get("audit_path")) is not str:
        raise G3Blocked("g3_accepted_phase_audit_missing")
    audit_path = _safe_manifest_relative_path(
        selected["audit_path"],
        "g3_accepted_phase_audit_missing",
    )
    value = artifact_io.read_json(store.run_root / audit_path)
    if not isinstance(value, dict):
        raise G3Blocked("g3_accepted_phase_audit_missing")
    return value


def _accept_g3_phase(
    store: MidDualRunStore,
    *,
    phase_id: str,
    rows: Sequence[Mapping[str, object]],
    upstream_lineage_sha256: str,
    audit_fields: Mapping[str, object],
) -> None:
    if phase_id in store.accepted_phase_ids:
        return
    phase_projection = validate_g3_phase_rows(
        phase_id=phase_id,
        rows=rows,
        accepted_phase_ids=store.accepted_phase_ids,
    )
    audit = {
        "schema_version": "xunce-mid-dual-g3-phase-audit/v1",
        "gate_id": "g3",
        "runner_id": RUNNER_ID,
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": PLATFORM_INVARIANTS,
        "phase_id": phase_id,
        "phase_name": G3_PHASE_NAMES[phase_id],
        "status": "complete",
        "formal_sample": True,
        "formal_evidence_eligible": True,
        "upstream_lineage_audit_sha256": _require_sha256(
            upstream_lineage_sha256,
            "g3_upstream_lineage_invalid",
        ),
        "phase_projection": phase_projection,
        **dict(audit_fields),
    }
    attempt_id = store.write_phase_attempt(phase_id, rows, audit)
    row_sha256 = store.phase_attempt_row_sha256(phase_id, attempt_id)
    store.accept_phase(phase_id, attempt_id, row_sha256)


def _revalidate_g3_accepted_prefix(
    store: MidDualRunStore,
    *,
    upstream_lineage_sha256: str,
    platform_invariants_audit_sha256: str,
    g1_evidence_binding: Mapping[str, object],
) -> None:
    expected_g1_binding = _validate_g1_evidence_binding(
        g1_evidence_binding
    )
    previous: list[str] = []
    for phase_id in store.accepted_phase_ids:
        rows = _accepted_g3_phase_rows(store, phase_id)
        expected_projection = validate_g3_phase_rows(
            phase_id=phase_id,
            rows=rows,
            accepted_phase_ids=previous,
        )
        audit = _accepted_g3_phase_audit(store, phase_id)
        first = rows[0]
        if (
            audit.get("schema_version")
            != "xunce-mid-dual-g3-phase-audit/v1"
            or audit.get("gate_id") != "g3"
            or audit.get("runner_id") != RUNNER_ID
            or validate_platform_invariants(
                audit.get("active_platforms"),
                audit.get("platform_invariants"),
            )
            != {
                "active_platforms": list(ACTIVE_PLATFORMS),
                "platform_invariants": PLATFORM_INVARIANTS,
            }
            or audit.get("phase_id") != phase_id
            or audit.get("phase_name") != G3_PHASE_NAMES[phase_id]
            or audit.get("status") != "complete"
            or audit.get("formal_sample") is not True
            or audit.get("formal_evidence_eligible") is not True
            or audit.get("upstream_lineage_audit_sha256")
            != upstream_lineage_sha256
            or audit.get("platform_invariants_audit_sha256")
            != platform_invariants_audit_sha256
            or _validate_g1_evidence_binding(
                audit.get("g1_evidence_binding")
            )
            != expected_g1_binding
            or audit.get("phase_projection") != expected_projection
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
            or (
                phase_id == "p01"
                and audit.get("frozen_manifest_sha256")
                != first.get("frozen_manifest_sha256")
            )
        ):
            raise G3Blocked("g3_accepted_phase_audit_invalid")
        previous.append(phase_id)


def _wheel_authority_audits(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    decisions: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    for row in rows:
        policy_identity = {
            field: row[field]
            for field in (
                "wheel_platform",
                "wheel_profile",
                "wheel_capability_revision",
                "checkpoint_sha256",
                "policy_state_sha256",
                "config_sha256",
                "input_sha256",
                "code_sha256",
                "frozen_manifest_sha256",
            )
        }
        decision = {
            "schema_version": "xunce-mid-dual-g3-decision-record/v1",
            "decision_record_id": row["decision_record_id"],
            "scenario_id": row["scenario_id"],
            "split": row["split"],
            "episode_id": row["episode_id"],
            "episode_index": row["episode_index"],
            "step_id": row["step_id"],
            "step_index": row["step_index"],
            "pre_observation_sha256": row["pre_snapshot_sha256"],
            "candidate_id": row["candidate_id"],
            "selected_candidate_cell_xy": row[
                "selected_candidate_cell_xy"
            ],
            "selected_theta": row["selected_theta"],
            "platform_invariant": PLATFORM_INVARIANTS[WHEEL_PLATFORM],
            "policy_identity": policy_identity,
        }
        observation = {
            "schema_version": "xunce-mid-dual-g3-observation-record/v1",
            "observation_record_id": row["observation_record_id"],
            "scenario_id": row["scenario_id"],
            "episode_id": row["episode_id"],
            "step_id": row["step_id"],
            "step_index": row["step_index"],
            "feedback_sha256": row["feedback_sha256"],
            "observation_sha256": row["post_snapshot_sha256"],
        }
        decisions.append(
            {
                **decision,
                "decision_sha256": _canonical_sha256(decision),
            }
        )
        observations.append(
            {
                **observation,
                "record_sha256": _canonical_sha256(observation),
            }
        )
    validate_wheel_authority_records(
        rows,
        decision_records=decisions,
        observation_records=observations,
    )
    return (
        {
            "schema_version": "xunce-mid-dual-g3-decision-audit/v1",
            "gate_id": "g3",
            "formal_evidence_eligible": True,
            "active_platforms": list(ACTIVE_PLATFORMS),
            "platform_invariants": PLATFORM_INVARIANTS,
            "record_count": len(decisions),
            "records": decisions,
        },
        {
            "schema_version": "xunce-mid-dual-g3-observation-audit/v1",
            "gate_id": "g3",
            "formal_evidence_eligible": True,
            "active_platforms": list(ACTIVE_PLATFORMS),
            "platform_invariants": PLATFORM_INVARIANTS,
            "record_count": len(observations),
            "records": observations,
        },
    )


def _validate_formal_row_lineage(
    *,
    wheel_rows: Sequence[Mapping[str, object]],
    interface_rows: Sequence[Mapping[str, object]],
    run_id: str,
    config_sha256: str,
    input_sha256: str,
    code_sha256: str,
    frozen_manifest_sha256: str,
    g1_manifest_sha256: str,
    g2_manifest_sha256: str,
) -> None:
    shared = {
        "run_id": run_id,
        "config_sha256": config_sha256,
        "input_sha256": input_sha256,
        "code_sha256": code_sha256,
        "source_sha256": code_sha256,
        "g1_source_manifest_sha256": g1_manifest_sha256,
        "g2_source_manifest_sha256": g2_manifest_sha256,
    }
    for row in wheel_rows:
        if (
            any(row.get(field) != value for field, value in shared.items())
            or row.get("frozen_manifest_sha256")
            != frozen_manifest_sha256
            or row.get("checkpoint_sha256") != UPDATE80_CHECKPOINT_SHA256
            or row.get("policy_state_sha256")
            != UPDATE80_POLICY_STATE_SHA256
        ):
            raise G3Blocked("g3_wheel_identity_invalid")
    for row in interface_rows:
        if any(row.get(field) != value for field, value in shared.items()):
            raise G3Blocked("g3_interface_identity_invalid")


def _validate_g3_input_selections(
    *,
    input_audit: Mapping[str, object],
    wheel_rows: Sequence[Mapping[str, object]],
    interface_rows: Sequence[Mapping[str, object]],
) -> None:
    wheel_selections = input_audit.get("wheel_selections")
    interface_selections = input_audit.get("interface_selections")
    if (
        not isinstance(wheel_selections, list)
        or len(wheel_selections) != 10
        or not isinstance(interface_selections, list)
        or len(interface_selections) != 6
    ):
        raise G3Blocked("g3_input_selection_invalid")
    selected_wheel = {
        str(row["wheel_episode_id"]): row for row in wheel_selections
    }
    episodes: dict[str, list[Mapping[str, object]]] = {}
    for row in wheel_rows:
        episodes.setdefault(str(row.get("episode_id")), []).append(row)
    if len(selected_wheel) != 10 or set(episodes) != set(selected_wheel):
        raise G3Blocked("g3_input_selection_invalid")
    for episode_id, rows in episodes.items():
        first = min(rows, key=lambda row: int(row["step_index"]))
        selection = selected_wheel[episode_id]
        if (
            first.get("split") != selection.get("split")
            or first.get("scenario_id") != selection.get("scenario_id")
            or first.get("episode_index")
            != selection.get("g1_episode_index")
            or first.get("lane_id") != selection.get("lane_id")
        ):
            raise G3Blocked("g3_input_selection_invalid")
    selected_interface = {
        str(row["replay_id"]): row for row in interface_selections
    }
    if len(selected_interface) != 6 or {
        str(row.get("replay_id")) for row in interface_rows
    } != set(selected_interface):
        raise G3Blocked("g3_input_selection_invalid")
    for row in interface_rows:
        selection = selected_interface[str(row["replay_id"])]
        for field in (
            "platform",
            "request_id",
            "request_sha256",
            "g2_reference_call_ids",
            "g2_semantic_digest",
            "timing_contract_id",
        ):
            if row.get(field) != selection.get(field):
                raise G3Blocked("g3_input_selection_invalid")


def _aggregate_style_timing(
    values: Sequence[float],
) -> dict[str, object]:
    materialized = tuple(values)
    if not materialized:
        raise G3Blocked("g3_timing_contract")
    average = mean(materialized)
    p95 = _nearest_rank(materialized, 0.95)
    maximum = max(materialized)
    at_or_below_1000_count = sum(
        value <= FINAL_TIME_MS for value in materialized
    )
    return {
        "sample_count": len(materialized),
        "mean_ms": average,
        "p50_ms": _nearest_rank(materialized, 0.50),
        "p95_ms": p95,
        "p99_ms": _nearest_rank(materialized, 0.99),
        "sample_stddev_ms": (
            stdev(materialized) if len(materialized) > 1 else 0.0
        ),
        "min_ms": min(materialized),
        "max_ms": maximum,
        "at_or_below_1000_count": at_or_below_1000_count,
        "at_or_below_1000_fraction": (
            at_or_below_1000_count / len(materialized)
        ),
        "over_2000_count": sum(
            value > MID_TIME_MS for value in materialized
        ),
        "midterm_reduced_passed": (
            average <= MID_TIME_MS
            and p95 <= MID_TIME_MS
            and maximum <= MID_TIME_MS
        ),
        "final_threshold_reduced_passed": (
            average <= FINAL_TIME_MS
            and p95 <= FINAL_TIME_MS
            and at_or_below_1000_count / len(materialized) >= 0.95
            and maximum <= MID_TIME_MS
        ),
    }


def recompute_g3_canonical_projection(
    *,
    frozen_manifest: Mapping[str, object],
    wheel_rows: Sequence[Mapping[str, object]],
    g1_rows: Sequence[Mapping[str, object]],
    interface_rows: Sequence[Mapping[str, object]],
    g2_rows: Sequence[Mapping[str, object]],
    g1_source_manifest_sha256: str,
    g2_source_manifest_sha256: str,
    upstream_lineage_audit_sha256: str,
) -> dict[str, object]:
    evaluated = evaluate_g3_evidence(
        frozen_manifest=frozen_manifest,
        wheel_rows=wheel_rows,
        g1_rows=g1_rows,
        interface_rows=interface_rows,
        g2_rows=g2_rows,
    )
    if evaluated.get("formal_evidence_eligible") is not True:
        blockers = evaluated.get("blockers")
        reason = (
            str(blockers[0])
            if isinstance(blockers, list) and blockers
            else "g3_recompute_blocked"
        )
        raise G3Blocked(reason)
    terminal_rows = [
        row for row in wheel_rows if row.get("is_terminal") is True
    ]
    if len(terminal_rows) != 10:
        raise G3Blocked("g3_wheel_episode_matrix")
    coverages = [float(row["coverage"]) for row in terminal_rows]
    paired_deltas = [
        float(row["coverage"]) - float(row["paired_g1_coverage"])
        for row in terminal_rows
    ]
    wheel_timing = _aggregate_style_timing(
        [_timing_ms(row) for row in wheel_rows]
    )
    interface_timing = _aggregate_style_timing(
        [_timing_ms(row) for row in interface_rows]
    )
    selected = validate_g3_manifest(frozen_manifest)
    wheel_coverage_80_count = sum(
        value >= MID_COVERAGE_THRESHOLD for value in coverages
    )
    wheel_coverage_mean = mean(coverages)
    paired_delta_mean = round(mean(paired_deltas), 15)
    return {
        "status": evaluated["status"],
        "scale_profile": SCALE_PROFILE,
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": PLATFORM_INVARIANTS,
        "full_scale_acceptance": False,
        "thresholds": {
            "midterm_coverage": MID_COVERAGE_THRESHOLD,
            "final_coverage": FINAL_COVERAGE_THRESHOLD,
            "paired_g1_delta_min": PAIRED_G1_DELTA_MIN,
            "midterm_planner_ms": MID_TIME_MS,
            "final_planner_ms": FINAL_TIME_MS,
            "final_proportion_at_or_below_1000": 0.95,
            "absolute_max_planner_ms": MID_TIME_MS,
        },
        "g1_source_manifest_sha256": _require_sha256(
            g1_source_manifest_sha256,
            "g3_summary_upstream_binding",
        ),
        "g2_source_manifest_sha256": _require_sha256(
            g2_source_manifest_sha256,
            "g3_summary_upstream_binding",
        ),
        "upstream_lineage_audit_sha256": _require_sha256(
            upstream_lineage_audit_sha256,
            "g3_summary_upstream_binding",
        ),
        "wheel_episode_count": 10,
        "wheel_step_count": len(wheel_rows),
        "interface_replay_count": 6,
        "split_episode_counts": {
            split: len(scenario_ids)
            for split, scenario_ids in selected.items()
        },
        "g3_midterm_crosscheck_passed": evaluated[
            "g3_midterm_crosscheck_passed"
        ],
        "g3_final_crosscheck_passed": evaluated[
            "g3_final_crosscheck_passed"
        ],
        "wheel_coverage_mean": wheel_coverage_mean,
        "wheel_coverage_80_count": wheel_coverage_80_count,
        "wheel_coverage_99_count": sum(
            value >= FINAL_COVERAGE_THRESHOLD for value in coverages
        ),
        "paired_g1_coverage_delta_mean": paired_delta_mean,
        "paired_g1_coverage_delta_sample_stddev": (
            stdev(paired_deltas) if len(paired_deltas) > 1 else 0.0
        ),
        "wheel_timing": wheel_timing,
        "interface_timing": interface_timing,
        "wheel_coverage_all_episodes_passed": (
            wheel_coverage_80_count == 10
        ),
        "wheel_coverage_mean_passed": (
            wheel_coverage_mean >= MID_COVERAGE_THRESHOLD
        ),
        "paired_g1_delta_passed": (
            paired_delta_mean >= PAIRED_G1_DELTA_MIN
        ),
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
        "interface_correctness_passed": evaluated["interface"][
            "interface_correctness_passed"
        ],
        "required_replay_identities_passed": evaluated["interface"][
            "required_replay_identities_passed"
        ],
    }


def _g3_summary(
    *,
    run_id: str,
    recomputed: Mapping[str, object],
) -> dict[str, object]:
    invariant_projection = validate_platform_invariants(
        recomputed.get("active_platforms"),
        recomputed.get("platform_invariants"),
    )
    if (
        recomputed.get("scale_profile") != SCALE_PROFILE
        or recomputed.get("full_scale_acceptance") is not False
        or recomputed.get("status") not in {"passed", "failed"}
    ):
        raise G3Blocked("g3_summary_projection_invalid")
    return {
        "schema_version": "xunce-mid-dual-g3-canonical-summary/v1",
        "scale_profile": SCALE_PROFILE,
        **invariant_projection,
        "full_scale_acceptance": False,
        "gate_id": "g3",
        "run_id": run_id,
        "status": recomputed["status"],
        "formal_evidence_eligible": True,
        "recomputed": dict(recomputed),
    }


def _g3_routing(summary: Mapping[str, object]) -> dict[str, object]:
    recomputed = summary.get("recomputed")
    if not isinstance(recomputed, Mapping):
        raise G3Blocked("g3_summary_projection_invalid")
    invariant_projection = validate_platform_invariants(
        summary.get("active_platforms"),
        summary.get("platform_invariants"),
    )
    if (
        summary.get("scale_profile") != SCALE_PROFILE
        or summary.get("full_scale_acceptance") is not False
    ):
        raise G3Blocked("g3_summary_projection_invalid")
    status = str(summary.get("status"))
    return {
        "schema_version": "xunce-mid-dual-g3-routing/v1",
        "gate_id": "g3",
        "run_id": summary.get("run_id"),
        "status": status,
        "scale_profile": SCALE_PROFILE,
        **invariant_projection,
        "full_scale_acceptance": False,
        "formal_evidence_eligible": True,
        "midterm_reduced_passed": (
            recomputed.get("g3_midterm_crosscheck_passed") is True
        ),
        "final_threshold_reduced_passed": (
            recomputed.get("g3_final_crosscheck_passed") is True
        ),
        "next_route": (
            "task10_aggregate" if status == "passed" else "repair_g3"
        ),
        "blocking_reasons": [],
    }


def _render_g3_report(summary: Mapping[str, object]) -> str:
    recomputed = summary.get("recomputed")
    if not isinstance(recomputed, Mapping):
        raise G3Blocked("g3_summary_projection_invalid")
    validate_platform_invariants(
        summary.get("active_platforms"),
        summary.get("platform_invariants"),
    )
    if (
        summary.get("scale_profile") != SCALE_PROFILE
        or summary.get("full_scale_acceptance") is not False
        or summary.get("status") not in {"passed", "failed"}
    ):
        raise G3Blocked("g3_summary_projection_invalid")
    status = str(summary["status"])
    conclusion = (
        "通过既定中期双门槛交叉检查。"
        if status == "passed"
        else "未通过既定中期双门槛交叉检查，保留真实失败结果。"
    )
    wheel_timing = recomputed.get("wheel_timing")
    interface_timing = recomputed.get("interface_timing")
    if not isinstance(wheel_timing, Mapping) or not isinstance(
        interface_timing,
        Mapping,
    ):
        raise G3Blocked("g3_summary_projection_invalid")
    lines = [
        "# G3 缩减规模闭环覆盖与接口回放正式实验报告",
        "",
        f"- 运行标识：`{summary.get('run_id')}`",
        f"- 缩减规模标识：`{SCALE_PROFILE}`",
        f"- 缩减规模判定状态：`{status}`",
        "- 坡度硬约束：wheel、legged、hopper 均为 "
        "`max_traversable_slope_deg == 30.0`",
        "- wheel 覆盖率："
        f"{recomputed.get('wheel_coverage_80_count')}/10 达到 80%，"
        f"均值={recomputed.get('wheel_coverage_mean')}，"
        "要求 10/10 wheel episode 单独达到 80% 且均值达到 80%",
        "- wheel 与 G1 配对覆盖率差均值："
        f"{recomputed.get('paired_g1_coverage_delta_mean')}，要求 ≥ -0.01",
        "- wheel 中期计时公式：mean / P95 / max ≤ 2000 ms；"
        f"实测 mean={wheel_timing.get('mean_ms')} ms，"
        f"P95={wheel_timing.get('p95_ms')} ms，"
        f"max={wheel_timing.get('max_ms')} ms",
        "- wheel 最终计时公式：mean / P95 ≤ 1000 ms，"
        "至少 95% ≤ 1000 ms，max ≤ 2000 ms；"
        f"实测比例={wheel_timing.get('at_or_below_1000_fraction')}",
        "- interface 合并计时（3 条 legged + 3 条 Hopper）："
        "中期 mean / P95 / max ≤ 2000 ms；最终 mean / P95 ≤ 1000 ms，"
        "至少 95% ≤ 1000 ms，max ≤ 2000 ms；"
        f"实测 mean={interface_timing.get('mean_ms')} ms，"
        f"P95={interface_timing.get('p95_ms')} ms，"
        f"max={interface_timing.get('max_ms')} ms，"
        f"比例={interface_timing.get('at_or_below_1000_fraction')}",
        "- 接口正确性与回放身份："
        f"correctness={recomputed.get('interface_correctness_passed')}，"
        "exact 3+3="
        f"{recomputed.get('required_replay_identities_passed')}",
        f"- 中期双门交叉检查：{recomputed.get('g3_midterm_crosscheck_passed')}",
        f"- 最终双门交叉检查：{recomputed.get('g3_final_crosscheck_passed')}",
        "- 验收范围：本结果仅适用于上述缩减规模与 update80，"
        "不构成全尺度验收。",
        f"- 缩减规模结论：{conclusion}",
        "",
        "所有正式样本仅来自已验证的 Task3 freeze、G1 独立复算数值证据和 "
        "Task8 G2 正式根；p01 为十个 wheel episode 的逐步闭环记录，"
        "p02 为 3 条 legged 与 3 条 Hopper 生产接口回放。",
    ]
    return "\n".join(lines) + "\n"


def _g3_report_audit(
    *,
    summary: Mapping[str, object],
    report: str,
) -> dict[str, object]:
    recomputed = summary.get("recomputed")
    if not isinstance(recomputed, Mapping):
        raise G3Blocked("g3_summary_projection_invalid")
    invariant_projection = validate_platform_invariants(
        summary.get("active_platforms"),
        summary.get("platform_invariants"),
    )
    if (
        summary.get("scale_profile") != SCALE_PROFILE
        or summary.get("full_scale_acceptance") is not False
    ):
        raise G3Blocked("g3_summary_projection_invalid")
    summary_bytes = (
        json.dumps(
            dict(summary),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return {
        "schema_version": "xunce-mid-dual-source-report-audit/v1",
        "gate_id": "g3",
        "run_id": summary["run_id"],
        "scale_profile": SCALE_PROFILE,
        **invariant_projection,
        "full_scale_acceptance": False,
        "renderer_id": "xunce-mid-dual-g3-canonical-report/v1",
        "summary_projection_sha256": _canonical_sha256(recomputed),
        "stored_summary_bytes_sha256": _bytes_sha256(summary_bytes),
        "report_bytes_sha256": _bytes_sha256(report.encode("utf-8")),
        "formal_evidence_eligible": True,
    }


def _g3_platform_invariants_audit(
    *,
    run_id: str,
    config_sha256: str,
    input_sha256: str,
    code_sha256: str,
    upstream_lineage_audit_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": (
            "xunce-mid-dual-g3-platform-invariants-audit/v1"
        ),
        "gate_id": "g3",
        "scale_profile": SCALE_PROFILE,
        "run_id": validate_g3_run_id(run_id),
        "status": "passed",
        "formal_evidence_eligible": True,
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": PLATFORM_INVARIANTS,
        "full_scale_acceptance": False,
        "config_sha256": _require_sha256(
            config_sha256,
            "g3_platform_invariants_audit_invalid",
        ),
        "input_sha256": _require_sha256(
            input_sha256,
            "g3_platform_invariants_audit_invalid",
        ),
        "code_sha256": _require_sha256(
            code_sha256,
            "g3_platform_invariants_audit_invalid",
        ),
        "upstream_lineage_audit_sha256": _require_sha256(
            upstream_lineage_audit_sha256,
            "g3_platform_invariants_audit_invalid",
        ),
    }


def _git_output(*args: str, cwd: Path = _REPO_ROOT) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _root_and_submodule_commits() -> tuple[str, str]:
    root_commit = _git_output("rev-parse", "HEAD")
    submodule = _REPO_ROOT / "path-planner"
    submodule_commit = (
        _git_output("rev-parse", "HEAD", cwd=submodule)
        if artifact_io.path_exists(submodule)
        else ""
    )
    return (
        root_commit or "unavailable-root-commit",
        submodule_commit or "not-present",
    )


def _open_g3_store(
    *,
    run_root: Path,
    effective_config: Mapping[str, object],
    mode: str,
) -> tuple[MidDualRunStore, bool]:
    expected_sha256 = _canonical_sha256(effective_config)
    root_exists = artifact_io.path_exists(run_root)
    if mode == "formal":
        if root_exists:
            raise G3Blocked("g3_formal_run_root_exists")
        return MidDualRunStore.create_new(run_root, effective_config), True
    if mode != "resume":
        raise G3Blocked("g3_mode_invalid")
    if not root_exists:
        raise G3Blocked("g3_resume_root_missing")
    if artifact_io.path_is_file(artifact_path(run_root, MID_DUAL_MANIFEST)):
        try:
            MidDualRunStore.verify_manifest(run_root)
        except (OSError, ValueError, TypeError) as exc:
            raise G3Blocked("g3_finalized_root_invalid") from exc
        raise G3Blocked("g3_run_already_finalized")
    try:
        return (
            MidDualRunStore.load_for_resume(run_root, expected_sha256),
            False,
        )
    except (OSError, ValueError, TypeError) as exc:
        raise G3Blocked("g3_resume_state_invalid") from exc


def _finalize_input_blocker(
    *,
    run_root: Path,
    run_id: str,
    mode: str,
    code_lineage: Mapping[str, object],
    reason: str,
) -> dict[str, object]:
    if mode == "resume" or artifact_io.path_exists(run_root):
        raise G3Blocked(reason)
    input_audit = {
        "schema_version": G3_INPUT_AUDIT_SCHEMA_VERSION,
        "gate_id": "g3",
        "run_id": run_id,
        "scale_profile": SCALE_PROFILE,
        "formal_evidence_eligible": False,
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": PLATFORM_INVARIANTS,
        "g1_source_manifest_sha256": None,
        "g2_source_manifest_sha256": None,
        "wheel_selections": [],
        "interface_selections": [],
        "blockers": [reason],
    }
    effective_config = {
        "schema_version": G3_EFFECTIVE_CONFIG_SCHEMA_VERSION,
        "gate_id": "g3",
        "runner_id": RUNNER_ID,
        "run_id": run_id,
        "output_root": G3_OUTPUT_BASE,
        "scale_profile": SCALE_PROFILE,
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": PLATFORM_INVARIANTS,
        "input_sha256": _json_artifact_sha256(input_audit),
        "code_sha256": code_lineage["code_sha256"],
        "required_phase_ids": list(G3_REQUIRED_PHASE_IDS),
        "source_contract_sha256": _canonical_sha256(G3_SOURCE_CONTRACT),
        "evidence_binding": {
            "schema_version": "xunce-mid-dual-g3-evidence-binding/v1",
            "input_audit_path": "g3_input_audit.json",
            "lineage_audit_path": "lineage_audit.json",
            "report_audit_path": "g3_report_audit.json",
            "platform_invariants_audit_path": (
                "platform_invariants_audit.json"
            ),
        },
        "upstream_binding": {
            "status": "blocked",
            "blocking_reason": reason,
            "active_platforms": list(ACTIVE_PLATFORMS),
            "platform_invariants": PLATFORM_INVARIANTS,
        },
    }
    store = MidDualRunStore.create_new(run_root, effective_config)
    root_commit, submodule_commit = _root_and_submodule_commits()
    store.capture_lineage(
        _g3_required_source_paths(),
        root_commit,
        submodule_commit,
    )
    summary = {
        "schema_version": "xunce-mid-dual-g3-canonical-summary/v1",
        "scale_profile": SCALE_PROFILE,
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": PLATFORM_INVARIANTS,
        "full_scale_acceptance": False,
        "gate_id": "g3",
        "run_id": run_id,
        "status": "blocked",
        "scale_profile": SCALE_PROFILE,
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": PLATFORM_INVARIANTS,
        "full_scale_acceptance": False,
        "formal_evidence_eligible": False,
        "blockers": [reason],
        "recomputed": None,
    }
    routing = {
        "schema_version": "xunce-mid-dual-g3-routing/v1",
        "gate_id": "g3",
        "run_id": run_id,
        "status": "blocked",
        "formal_evidence_eligible": False,
        "midterm_reduced_passed": False,
        "final_threshold_reduced_passed": False,
        "next_route": "resolve_g3_input_blocker",
        "blocking_reasons": [reason],
    }
    report = (
        "# G3 缩减规模正式实验阻塞报告\n\n"
        f"- 运行标识：`{run_id}`\n"
        f"- 缩减规模标识：`{SCALE_PROFILE}`\n"
        "- 坡度硬约束：wheel、legged、hopper 均为 "
        "`max_traversable_slope_deg == 30.0`\n"
        f"- 阻塞原因：`{reason}`\n"
        "- 未产生或接受任何正式 p01/p02 样本。\n"
        "- 本状态不构成全尺度验收。\n"
    )
    report_audit = {
        "schema_version": "xunce-mid-dual-source-report-audit/v1",
        "gate_id": "g3",
        "run_id": run_id,
        "scale_profile": SCALE_PROFILE,
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": PLATFORM_INVARIANTS,
        "full_scale_acceptance": False,
        "renderer_id": "xunce-mid-dual-g3-canonical-report/v1",
        "formal_evidence_eligible": False,
        "blocking_reason": reason,
        "report_bytes_sha256": _bytes_sha256(report.encode("utf-8")),
    }
    store.finalize(
        summary,
        routing,
        report,
        {
            "g3_input": input_audit,
            "g3_report": report_audit,
            "upstream_lineage": {
                "schema_version": (
                    "xunce-mid-dual-g3-upstream-lineage/v1"
                ),
                "gate_id": "g3",
                "formal_evidence_eligible": False,
                "active_platforms": list(ACTIVE_PLATFORMS),
                "platform_invariants": PLATFORM_INVARIANTS,
                "blockers": [reason],
            },
            "platform_invariants": {
                "schema_version": (
                    "xunce-mid-dual-g3-platform-invariants-audit/v1"
                ),
                "gate_id": "g3",
                "scale_profile": SCALE_PROFILE,
                "run_id": run_id,
                "status": "blocked",
                "formal_evidence_eligible": False,
                "active_platforms": list(ACTIVE_PLATFORMS),
                "platform_invariants": PLATFORM_INVARIANTS,
                "full_scale_acceptance": False,
                "blocking_reason": reason,
            },
        },
    )
    MidDualRunStore.verify_manifest(store.run_root)
    return {
        "execution_status": "blocked",
        "gate_status": "blocked",
        "formal_evidence_eligible": False,
        "blockers": [reason],
        "run_id": run_id,
        "run_root": str(store.run_root).replace("\\", "/"),
        "summary": artifact_io.read_json(store.run_root / "summary.json"),
    }


def run_g3(
    *,
    config_path: str | Path,
    frozen_bundle_root: str | Path,
    g1_root: str | Path,
    g2_root: str | Path,
    run_id: str,
    mode: str,
    g1_assessment_envelope: str | Path | None = None,
) -> dict[str, object]:
    """创建或恢复唯一的 G3 正式执行链。"""

    validated_run_id = validate_g3_run_id(run_id)
    if mode not in {"preflight", "formal", "resume"}:
        raise G3Blocked("g3_mode_invalid")
    supplied_config = Path(config_path).resolve()
    if supplied_config != _CANONICAL_CONFIG_PATH.resolve():
        raise G3Blocked("g3_config_path_not_canonical")
    try:
        base_config = artifact_io.read_json(supplied_config)
        validate_g3_config_payload(base_config)
    except (OSError, ValueError, TypeError) as exc:
        raise G3Blocked("g3_config_missing_or_invalid") from exc
    roots = validate_g3_root_contract(
        frozen_bundle_root=frozen_bundle_root,
        g1_root=g1_root,
        g2_root=g2_root,
        run_id=validated_run_id,
        require_existing=True,
    )
    code_lineage = compute_g3_code_lineage()
    try:
        frozen = load_verified_frozen_bundle(roots["frozen_bundle_root"])
        g1_source = load_verified_g1_source(
            root=roots["g1_root"],
            frozen_bundle=frozen,
            g1_assessment_envelope=(
                None
                if g1_assessment_envelope is None
                else Path(g1_assessment_envelope)
            ),
        )
        g2_source = load_verified_g2_root(roots["g2_root"])
    except G3Blocked as exc:
        return _finalize_input_blocker(
            run_root=roots["output_root"],
            run_id=validated_run_id,
            mode=mode,
            code_lineage=code_lineage,
            reason=exc.reason,
        )
    except (OSError, ValueError, TypeError, UnicodeError, KeyError):
        return _finalize_input_blocker(
            run_root=roots["output_root"],
            run_id=validated_run_id,
            mode=mode,
            code_lineage=code_lineage,
            reason="g3_input_verification_failed",
        )
    input_audit = _build_g3_input_audit(
        run_id=validated_run_id,
        frozen=frozen,
        g1_source=g1_source,
        g2_source=g2_source,
    )
    upstream_binding = _build_upstream_binding(
        frozen=frozen,
        g1_source=g1_source,
        g2_source=g2_source,
    )
    upstream_lineage = _build_upstream_lineage_audit(
        upstream_binding=upstream_binding,
        frozen=frozen,
        g1_source=g1_source,
        g2_source=g2_source,
    )
    upstream_lineage_sha256 = _json_artifact_sha256(upstream_lineage)
    g1_evidence_binding = _g1_source_evidence_binding(g1_source)
    effective_config = build_g3_effective_config(
        base_config=base_config,
        run_id=validated_run_id,
        input_audit=input_audit,
        code_lineage=code_lineage,
        upstream_binding=upstream_binding,
    )
    if mode == "preflight":
        return {
            "execution_status": "not_started",
            "gate_status": "preflight_passed",
            "formal_evidence_eligible": False,
            "blockers": [],
            "run_id": validated_run_id,
            "active_platforms": list(ACTIVE_PLATFORMS),
            "platform_invariants": PLATFORM_INVARIANTS,
            "validated_inputs": {
                "freeze_manifest_sha256": frozen["manifest_sha256"],
                "g1_source_manifest_sha256": g1_source["manifest_sha256"],
                "g2_source_manifest_sha256": g2_source["manifest_sha256"],
                "g1_evidence_binding": g1_evidence_binding,
            },
        }
    store, newly_created = _open_g3_store(
        run_root=roots["output_root"],
        effective_config=effective_config,
        mode=mode,
    )
    if newly_created:
        root_commit, submodule_commit = _root_and_submodule_commits()
        lineage = store.capture_lineage(
            _g3_required_source_paths(),
            root_commit,
            submodule_commit,
        )
        if (
            lineage.get("formal_evidence_eligible") is not True
            or compute_g3_code_lineage()["code_sha256"]
            != code_lineage["code_sha256"]
        ):
            raise G3Blocked("g3_lineage_capture_blocked")
        g1 = _import_local_script("run_xunce_mid_dual_g1_coverage")

        environment = store.capture_environment(g1._environment_probe)  # noqa: SLF001
        if environment.get("formal_evidence_eligible") is not True:
            raise G3Blocked("g3_environment_capture_blocked")
    input_sha256 = str(effective_config["input_sha256"])
    code_sha256 = str(effective_config["code_sha256"])
    platform_invariants_audit = _g3_platform_invariants_audit(
        run_id=validated_run_id,
        config_sha256=store.config_sha256,
        input_sha256=input_sha256,
        code_sha256=code_sha256,
        upstream_lineage_audit_sha256=upstream_lineage_sha256,
    )
    platform_invariants_audit_sha256 = _json_artifact_sha256(
        platform_invariants_audit
    )
    _revalidate_g3_accepted_prefix(
        store,
        upstream_lineage_sha256=upstream_lineage_sha256,
        platform_invariants_audit_sha256=(
            platform_invariants_audit_sha256
        ),
        g1_evidence_binding=g1_evidence_binding,
    )
    if "p01" not in store.accepted_phase_ids:
        g1 = _import_local_script("run_xunce_mid_dual_g1_coverage")

        catalog, safety_contract = g1._load_production_context()  # noqa: SLF001
        raw_wheel_rows = execute_update80_wheel_rows(
            catalog=catalog,
            frozen_manifest=frozen["frozen_manifest"],
            raw_frozen_manifest=frozen["raw_manifest"],
            g1_rows=g1_source["coverage_rows"],
            safety_contract=safety_contract,
            config_sha256=store.config_sha256,
            run_id=validated_run_id,
            evaluation_seed_start=2026073100,
        )
        wheel_rows, decisions, observations = materialize_wheel_authority(
            raw_wheel_rows,
            config_sha256=store.config_sha256,
            input_sha256=input_sha256,
            code_sha256=code_sha256,
            frozen_manifest_sha256=str(frozen["manifest_sha256"]),
            g1_source_manifest_sha256=str(
                g1_source["manifest_sha256"]
            ),
            g2_source_manifest_sha256=str(
                g2_source["manifest_sha256"]
            ),
        )
        decision_audit = {
            "schema_version": "xunce-mid-dual-g3-decision-audit/v1",
            "gate_id": "g3",
            "formal_evidence_eligible": True,
            "active_platforms": list(ACTIVE_PLATFORMS),
            "platform_invariants": PLATFORM_INVARIANTS,
            "record_count": len(decisions),
            "records": decisions,
        }
        observation_audit = {
            "schema_version": "xunce-mid-dual-g3-observation-audit/v1",
            "gate_id": "g3",
            "formal_evidence_eligible": True,
            "active_platforms": list(ACTIVE_PLATFORMS),
            "platform_invariants": PLATFORM_INVARIANTS,
            "record_count": len(observations),
            "records": observations,
        }
        validate_wheel_authority_records(
            wheel_rows,
            decision_records=decisions,
            observation_records=observations,
        )
        _evaluate_wheel(
            selected=frozen["selected"],
            wheel_rows=wheel_rows,
            g1_rows=g1_source["coverage_rows"],
        )
        _validate_formal_row_lineage(
            wheel_rows=wheel_rows,
            interface_rows=(),
            run_id=validated_run_id,
            config_sha256=store.config_sha256,
            input_sha256=input_sha256,
            code_sha256=code_sha256,
            frozen_manifest_sha256=str(frozen["manifest_sha256"]),
            g1_manifest_sha256=str(g1_source["manifest_sha256"]),
            g2_manifest_sha256=str(g2_source["manifest_sha256"]),
        )
        _accept_g3_phase(
            store,
            phase_id="p01",
            rows=wheel_rows,
            upstream_lineage_sha256=upstream_lineage_sha256,
            audit_fields={
                "config_sha256": store.config_sha256,
                "input_sha256": input_sha256,
                "code_sha256": code_sha256,
                "decision_audit_sha256": _json_artifact_sha256(
                    decision_audit
                ),
                "observation_audit_sha256": _json_artifact_sha256(
                    observation_audit
                ),
                "frozen_manifest_sha256": frozen["manifest_sha256"],
                "g1_source_manifest_sha256": g1_source["manifest_sha256"],
                "g2_source_manifest_sha256": g2_source["manifest_sha256"],
                "g1_evidence_binding": g1_evidence_binding,
                "platform_invariants_audit_sha256": (
                    platform_invariants_audit_sha256
                ),
            },
        )
    else:
        wheel_rows = _accepted_g3_phase_rows(store, "p01")
        decision_audit, observation_audit = _wheel_authority_audits(
            wheel_rows
        )
        _evaluate_wheel(
            selected=frozen["selected"],
            wheel_rows=wheel_rows,
            g1_rows=g1_source["coverage_rows"],
        )
        _validate_formal_row_lineage(
            wheel_rows=wheel_rows,
            interface_rows=(),
            run_id=validated_run_id,
            config_sha256=store.config_sha256,
            input_sha256=input_sha256,
            code_sha256=code_sha256,
            frozen_manifest_sha256=str(frozen["manifest_sha256"]),
            g1_manifest_sha256=str(g1_source["manifest_sha256"]),
            g2_manifest_sha256=str(g2_source["manifest_sha256"]),
        )
    if "p02" not in store.accepted_phase_ids:
        interface_rows = execute_interface_replays(
            g2_source=g2_source,
            run_id=validated_run_id,
            config_sha256=store.config_sha256,
            input_sha256=input_sha256,
            code_sha256=code_sha256,
            g1_source_manifest_sha256=str(
                g1_source["manifest_sha256"]
            ),
            g2_source_manifest_sha256=str(
                g2_source["manifest_sha256"]
            ),
        )
        _evaluate_interface(
            interface_rows=interface_rows,
            g2_rows=g2_source["formal_rows"],
        )
        _validate_formal_row_lineage(
            wheel_rows=wheel_rows,
            interface_rows=interface_rows,
            run_id=validated_run_id,
            config_sha256=store.config_sha256,
            input_sha256=input_sha256,
            code_sha256=code_sha256,
            frozen_manifest_sha256=str(frozen["manifest_sha256"]),
            g1_manifest_sha256=str(g1_source["manifest_sha256"]),
            g2_manifest_sha256=str(g2_source["manifest_sha256"]),
        )
        _accept_g3_phase(
            store,
            phase_id="p02",
            rows=interface_rows,
            upstream_lineage_sha256=upstream_lineage_sha256,
            audit_fields={
                "config_sha256": store.config_sha256,
                "input_sha256": input_sha256,
                "code_sha256": code_sha256,
                "frozen_manifest_sha256": frozen["manifest_sha256"],
                "g1_source_manifest_sha256": g1_source["manifest_sha256"],
                "g2_source_manifest_sha256": g2_source["manifest_sha256"],
                "g2_cohort_sha256": upstream_binding[
                    "g2_cohort_sha256"
                ],
                "g2_approval_sha256": upstream_binding[
                    "g2_approval_sha256"
                ],
                "g2_hopper_resolution_sha256": upstream_binding[
                    "g2_hopper_resolution_sha256"
                ],
                "g1_evidence_binding": g1_evidence_binding,
                "platform_invariants_audit_sha256": (
                    platform_invariants_audit_sha256
                ),
            },
        )
    else:
        interface_rows = _accepted_g3_phase_rows(store, "p02")
        _evaluate_interface(
            interface_rows=interface_rows,
            g2_rows=g2_source["formal_rows"],
        )
    _revalidate_g3_accepted_prefix(
        store,
        upstream_lineage_sha256=upstream_lineage_sha256,
        platform_invariants_audit_sha256=(
            platform_invariants_audit_sha256
        ),
        g1_evidence_binding=g1_evidence_binding,
    )
    _validate_formal_row_lineage(
        wheel_rows=wheel_rows,
        interface_rows=interface_rows,
        run_id=validated_run_id,
        config_sha256=store.config_sha256,
        input_sha256=input_sha256,
        code_sha256=code_sha256,
        frozen_manifest_sha256=str(frozen["manifest_sha256"]),
        g1_manifest_sha256=str(g1_source["manifest_sha256"]),
        g2_manifest_sha256=str(g2_source["manifest_sha256"]),
    )
    _validate_g3_input_selections(
        input_audit=input_audit,
        wheel_rows=wheel_rows,
        interface_rows=interface_rows,
    )
    decision_audit, observation_audit = _wheel_authority_audits(wheel_rows)
    p01_audit = _accepted_g3_phase_audit(store, "p01")
    if (
        p01_audit.get("decision_audit_sha256")
        != _json_artifact_sha256(decision_audit)
        or p01_audit.get("observation_audit_sha256")
        != _json_artifact_sha256(observation_audit)
    ):
        raise G3Blocked("g3_wheel_authority_audit_drift")
    recomputed = recompute_g3_canonical_projection(
        frozen_manifest=frozen["raw_manifest"],
        wheel_rows=wheel_rows,
        g1_rows=g1_source["coverage_rows"],
        interface_rows=interface_rows,
        g2_rows=g2_source["formal_rows"],
        g1_source_manifest_sha256=str(g1_source["manifest_sha256"]),
        g2_source_manifest_sha256=str(g2_source["manifest_sha256"]),
        upstream_lineage_audit_sha256=upstream_lineage_sha256,
    )
    summary = _g3_summary(
        run_id=validated_run_id,
        recomputed=recomputed,
    )
    report = _render_g3_report(summary)
    report_audit = _g3_report_audit(summary=summary, report=report)
    store.finalize(
        summary,
        _g3_routing(summary),
        report,
        {
            "g3_input": input_audit,
            "g3_report": report_audit,
            "upstream_lineage": upstream_lineage,
            "g3_decisions": decision_audit,
            "g3_observations": observation_audit,
            "platform_invariants": platform_invariants_audit,
        },
    )
    try:
        MidDualRunStore.verify_manifest(store.run_root)
    except (OSError, ValueError, TypeError) as exc:
        raise G3Blocked("g3_final_manifest_invalid") from exc
    persisted_summary = artifact_io.read_json(
        store.run_root / "summary.json"
    )
    if persisted_summary != summary:
        raise G3Blocked("g3_persisted_summary_drift")
    return {
        "execution_status": "complete",
        "gate_status": summary["status"],
        "formal_evidence_eligible": True,
        "blockers": [],
        "run_id": validated_run_id,
        "run_root": str(store.run_root).replace("\\", "/"),
        "summary": persisted_summary,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the reduced midterm G3 closed-loop cross-check",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--frozen-bundle-root")
    parser.add_argument("--g1-root")
    parser.add_argument("--g1-assessment-envelope")
    parser.add_argument("--g2-root")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--mode",
        choices=("preflight", "formal", "resume"),
        default="preflight",
    )
    args = parser.parse_args(argv)
    missing = [
        reason
        for value, reason in (
            (
                args.frozen_bundle_root,
                "g3_frozen_bundle_root_missing",
            ),
            (args.g1_root, "g3_g1_root_missing"),
            (args.g2_root, "g3_g2_root_missing"),
        )
        if value is None
    ]
    if missing:
        result = {
            "execution_status": "not_started",
            "gate_status": "blocked",
            "formal_evidence_eligible": False,
            "blockers": missing,
            "run_id": args.run_id,
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 1
    try:
        result = run_g3(
            config_path=args.config,
            frozen_bundle_root=args.frozen_bundle_root,
            g1_root=args.g1_root,
            g2_root=args.g2_root,
            run_id=args.run_id,
            mode=args.mode,
            g1_assessment_envelope=args.g1_assessment_envelope,
        )
    except G3Blocked as exc:
        result = {
            "execution_status": "not_started",
            "gate_status": "blocked",
            "formal_evidence_eligible": False,
            "blockers": [exc.reason],
            "run_id": args.run_id,
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return int(result.get("gate_status") == "blocked")


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "G3Blocked",
    "RUNNER_ID",
    "SCALE_PROFILE",
    "TIMING_COMPONENT_FIELDS",
    "TIMING_CONTRACT_ID",
    "deterministic_candidate_sha256",
    "deterministic_feedback_sha256",
    "deterministic_request_id",
    "deterministic_request_sha256",
    "deterministic_route_result_sha256",
    "evaluate_g3_evidence",
    "execute_update80_wheel_rows",
    "build_g3_effective_config",
    "load_verified_g1_root",
    "load_verified_g2_root",
    "recompute_g3_canonical_projection",
    "run_g3",
    "validate_g3_config_payload",
    "validate_g3_phase_rows",
    "validate_g3_root_contract",
    "validate_g3_run_id",
    "validate_g3_manifest",
    "validate_g1_native_effective_config_projection",
]
