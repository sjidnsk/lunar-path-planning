"""Run the reduced midterm G2 planning-time qualification experiment.

Formal evidence is limited to the frozen 3 x 43 x 5 request matrix.  Every
stored timing row retains the five raw integer nanosecond phases and their
exact sum; millisecond fields are derived projections used only for reporting
and threshold evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform as host_platform
import random
import re
import secrets
import shutil
import statistics
import struct
import subprocess
import sys
import threading
from typing import Any

import xunce_artifact_io as artifact_io
from xunce_artifact_io import read_bytes as artifact_read_bytes
from xunce_artifact_paths import MID_DUAL_MANIFEST, MID_DUAL_PHASE_STATE, artifact_path
from xunce_mid_dual_artifacts import MidDualRunStore
from xunce_mid_dual_contracts import PlanningCallRow, evaluate_formal_g2


SCALE_PROFILE = "midterm_reduced_w8x3_update80/v1"
G2_CONFIG_SCHEMA_VERSION = "xunce-mid-dual-g2-planning-time-config/v1"
G2_ROW_SCHEMA_VERSION = "xunce-mid-dual-g2-planning-call-row/v1"
G2_SUMMARY_SCHEMA_VERSION = "xunce-mid-dual-g2-canonical-summary/v1"
G2_RUNNER_ID = "xunce-mid-dual-g2-planning-time-runner/v1"
G2_GATE_ID = "g2"
G2_OUTPUT_BASE = "D:/xunce/out/mid_dual/g2"
G2_PLATFORMS = ("wheel", "legged", "hopper")
G2_SCALES = ("standard", "kilometer")
G2_REPEATS = 5
G2_FORMAL_CALLS = 645
G2_REQUIRED_PHASE_IDS = ("p01", "p02", "p03", "p04")
G2_PHASE_NAMES = {
    "p01": "preflight",
    "p02": "cold_start_and_warmup",
    "p03": "worker_one_semantic_diagnostic",
    "p04": "formal_worker_four",
}
G2_MODES = ("preflight", "diagnostic", "formal")
TIMING_CONTRACT_ID = "five-phase-sequential-ns/v1"
STATIC_CACHE_CONTRACT_ID = "immutable-terrain-static-validation-only/v1"
FORMAL_ENVIRONMENT_POLICY_SCHEMA_VERSION = (
    "xunce-mid-dual-g2-formal-environment-policy/v1"
)
FORMAL_ENVIRONMENT_OBSERVATION_SCHEMA_VERSION = (
    "xunce-mid-dual-g2-formal-environment-observation/v1"
)
FORMAL_ENVIRONMENT_AUDIT_SCHEMA_VERSION = (
    "xunce-mid-dual-g2-formal-environment-audit/v1"
)
FORMAL_LEASE_SCHEMA_VERSION = "xunce-mid-dual-g2-formal-lease/v1"
FORMAL_LEASE_PATH = (
    "D:/xunce/out/mid_dual/g2/.formal-exclusive-lease.json"
)
HIGH_PERFORMANCE_POWER_SCHEME_GUID = (
    "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
)
FORMAL_ENVIRONMENT_POLICY = {
    "schema_version": FORMAL_ENVIRONMENT_POLICY_SCHEMA_VERSION,
    "lease_path": FORMAL_LEASE_PATH,
    "allowed_power_scheme_guids": [
        HIGH_PERFORMANCE_POWER_SCHEME_GUID
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
TIMING_FIELDS = (
    "input_validation_ns",
    "platform_instantiation_ns",
    "search_ns",
    "complete_route_validation_ns",
    "result_assembly_ns",
)
TIMING_MS_FIELDS = (
    "input_validation_ms",
    "platform_instantiation_ms",
    "search_ms",
    "complete_route_validation_ms",
    "result_assembly_ms",
)
EXPECTED_CLASS_COUNTS = {
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
ACTIVE_PLATFORMS = ("wheel", "legged", "hopper")
PLATFORM_INVARIANTS = {
    platform: {"max_traversable_slope_deg": 30.0}
    for platform in ACTIVE_PLATFORMS
}
G2_REQUIRED_SOURCE_RELATIVE_PATHS = (
    "configs/xunce_mid_dual_g2_planning_time_v1.json",
    "scripts/run_xunce_mid_dual_g2_planning_time.py",
    "scripts/xunce_mid_dual_g2_inputs.py",
    "scripts/xunce_mid_dual_contracts.py",
    "scripts/xunce_mid_dual_artifacts.py",
    "scripts/xunce_artifact_io.py",
    "scripts/xunce_artifact_paths.py",
)
G2_SOURCE_CONTRACT = {
    "schema_version": "xunce-mid-dual-source-contract/v1",
    "gate_id": "g2",
    "config_schema_version": G2_CONFIG_SCHEMA_VERSION,
    "runner_id": G2_RUNNER_ID,
    "required_phase_ids": list(G2_REQUIRED_PHASE_IDS),
    "output_base": G2_OUTPUT_BASE,
    "input_audit_schema_version": "xunce-mid-dual-g2-input-audit/v1",
    "report_audit_schema_version": (
        "xunce-mid-dual-source-report-audit/v1"
    ),
    "report_renderer_id": "xunce-mid-dual-g2-canonical-report/v1",
    "required_lineage_sources": list(G2_REQUIRED_SOURCE_RELATIVE_PATHS),
}

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG_PATH = (
    REPO_ROOT / "configs" / "xunce_mid_dual_g2_planning_time_v1.json"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}

G2_ROW_KEYS = frozenset(
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
        *TIMING_FIELDS,
        "total_ns",
        *TIMING_MS_FIELDS,
        "elapsed_ms",
        "provider_success",
        "route_l2_valid",
        "semantic_digest",
        "timing_contract_id",
        "formal_sample",
        "repeat_index",
    }
)


class G2Blocked(ValueError):
    """Stable fail-closed reason for unusable G2 formal evidence."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class G2Interrupted(RuntimeError):
    """A recoverable stop after completed call rows were journalled."""

    def __init__(self, reason: str = "g2_execution_interrupted") -> None:
        self.reason = reason
        super().__init__(reason)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


class _FileTime(ctypes.Structure):
    _fields_ = (
        ("low", wintypes.DWORD),
        ("high", wintypes.DWORD),
    )


def _windows_kernel32():
    try:
        return ctypes.WinDLL("kernel32", use_last_error=True)
    except (AttributeError, OSError) as exc:
        raise G2Blocked("g2_formal_windows_probe_unavailable") from exc


def _file_time_ticks(value: _FileTime) -> int:
    return (int(value.high) << 32) | int(value.low)


def _current_process_start_utc() -> str:
    if os.name != "nt":
        return _utc_now()

    kernel32 = _windows_kernel32()
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = ctypes.c_void_p
    get_process_times = kernel32.GetProcessTimes
    get_process_times.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
    ]
    get_process_times.restype = wintypes.BOOL
    creation = _FileTime()
    exit_time = _FileTime()
    kernel = _FileTime()
    user = _FileTime()
    handle = get_current_process()
    ok = get_process_times(
        handle,
        ctypes.byref(creation),
        ctypes.byref(exit_time),
        ctypes.byref(kernel),
        ctypes.byref(user),
    )
    if not ok:
        raise G2Blocked("g2_formal_process_start_probe_failed")
    ticks = _file_time_ticks(creation)
    unix_seconds = ticks / 10_000_000.0 - 11_644_473_600.0
    return (
        datetime.fromtimestamp(unix_seconds, timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _run_read_only_windows_probe(
    arguments: Sequence[str],
    *,
    reason: str,
    timeout_seconds: float = 30.0,
) -> tuple[bytes, str]:
    try:
        completed = subprocess.run(
            tuple(arguments),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
            creationflags=int(
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
            ),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise G2Blocked(reason) from exc
    probe_sha256 = _domain_hash(
        "xunce-mid-dual-g2-read-only-windows-probe/v1",
        _canonical_bytes(list(arguments)),
        str(completed.returncode).encode("ascii"),
        completed.stdout,
        completed.stderr,
    )
    if completed.returncode != 0:
        raise G2Blocked(reason)
    return completed.stdout, probe_sha256


def _parse_active_power_scheme(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise G2Blocked("g2_formal_power_probe_failed")
    matches = {
        item.casefold()
        for item in re.findall(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{12}\b",
            payload.decode("ascii", errors="ignore"),
        )
    }
    if len(matches) != 1:
        raise G2Blocked("g2_formal_power_probe_failed")
    return next(iter(matches))


def _probe_active_power_scheme() -> tuple[str, str]:
    payload, probe_sha256 = _run_read_only_windows_probe(
        ("powercfg.exe", "/getactivescheme"),
        reason="g2_formal_power_probe_failed",
    )
    return _parse_active_power_scheme(payload), probe_sha256


def _powershell_executable() -> str:
    executable = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
    if not executable:
        raise G2Blocked("g2_formal_powershell_probe_unavailable")
    return executable


def _powershell_json_probe(
    script: str,
    *,
    reason: str,
) -> tuple[object, str]:
    payload, probe_sha256 = _run_read_only_windows_probe(
        (
            _powershell_executable(),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                "$ErrorActionPreference='Stop';"
                "$OutputEncoding=[Console]::OutputEncoding="
                "[Text.UTF8Encoding]::new($false);"
                + script
            ),
        ),
        reason=reason,
    )
    try:
        parsed = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G2Blocked(reason) from exc
    return parsed, probe_sha256


def _classify_process_inventory(
    records: Sequence[Mapping[str, object]],
    *,
    patterns: Sequence[str],
    current_pid: int,
) -> tuple[list[dict[str, object]], list[int]]:
    if (
        type(current_pid) is not int
        or current_pid <= 0
        or not patterns
        or any(type(item) is not str or not item for item in patterns)
    ):
        raise G2Blocked("g2_formal_process_inventory_invalid")
    by_pid: dict[int, dict[str, object]] = {}
    for source in records:
        if not isinstance(source, Mapping):
            raise G2Blocked("g2_formal_process_inventory_invalid")
        pid = source.get("pid")
        parent_pid = source.get("parent_pid")
        name = source.get("name")
        command_line = source.get("command_line")
        if type(pid) is int and pid == 0:
            continue
        if (
            type(pid) is not int
            or pid < 0
            or type(parent_pid) is not int
            or parent_pid < 0
            or type(name) is not str
            or not name
            or type(command_line) is not str
            or pid in by_pid
        ):
            raise G2Blocked("g2_formal_process_inventory_invalid")
        by_pid[pid] = {
            "pid": pid,
            "parent_pid": parent_pid,
            "name": name,
            "command_line": command_line,
        }
    if current_pid not in by_pid:
        raise G2Blocked("g2_formal_current_process_missing")

    ancestors = {current_pid}
    cursor = current_pid
    while True:
        parent_pid = int(by_pid[cursor]["parent_pid"])
        if parent_pid <= 0 or parent_pid not in by_pid:
            break
        if parent_pid in ancestors:
            raise G2Blocked("g2_formal_process_inventory_cycle")
        ancestors.add(parent_pid)
        cursor = parent_pid

    descendants = {current_pid}
    changed = True
    while changed:
        changed = False
        for pid, row in by_pid.items():
            if pid not in descendants and row["parent_pid"] in descendants:
                descendants.add(pid)
                changed = True
    excluded = ancestors | descendants

    normalized_patterns = [
        (pattern, pattern.casefold()) for pattern in patterns
    ]
    competing: list[dict[str, object]] = []
    for pid in sorted(by_pid):
        if pid in excluded:
            continue
        row = by_pid[pid]
        searchable = (
            f"{row['name']}\n{row['command_line']}".casefold()
        )
        matched = next(
            (
                original
                for original, normalized in normalized_patterns
                if normalized in searchable
            ),
            None,
        )
        if matched is None:
            continue
        competing.append(
            {
                "pid": pid,
                "parent_pid": row["parent_pid"],
                "name": row["name"],
                "matched_pattern": matched,
                "command_sha256": _domain_hash(
                    "xunce-mid-dual-g2-process-command/v1",
                    str(row["command_line"]).encode(
                        "utf-8",
                        errors="strict",
                    ),
                ),
            }
        )
    return competing, sorted(excluded)


def _probe_process_inventory(
    patterns: Sequence[str],
    current_pid: int,
) -> tuple[list[dict[str, object]], list[int], str]:
    parsed, inventory_sha256 = _powershell_json_probe(
        (
            "$rows=@(Get-CimInstance -ClassName Win32_Process |"
            "Select-Object ProcessId,ParentProcessId,Name,CommandLine |"
            "Sort-Object ProcessId);"
            "ConvertTo-Json -InputObject $rows -Compress"
        ),
        reason="g2_formal_process_probe_failed",
    )
    if type(parsed) is not list:
        raise G2Blocked("g2_formal_process_probe_failed")
    records: list[dict[str, object]] = []
    for row in parsed:
        if not isinstance(row, Mapping):
            raise G2Blocked("g2_formal_process_probe_failed")
        try:
            pid = int(row["ProcessId"])
            parent_pid = int(row["ParentProcessId"])
        except (KeyError, TypeError, ValueError) as exc:
            raise G2Blocked("g2_formal_process_probe_failed") from exc
        name = row.get("Name")
        command_line = row.get("CommandLine")
        records.append(
            {
                "pid": pid,
                "parent_pid": parent_pid,
                "name": name if type(name) is str else "",
                "command_line": (
                    command_line
                    if type(command_line) is str
                    else ""
                ),
            }
        )
    competing, excluded = _classify_process_inventory(
        records,
        patterns=patterns,
        current_pid=current_pid,
    )
    return competing, excluded, inventory_sha256


def _read_system_times() -> tuple[int, int, int]:
    kernel32 = _windows_kernel32()
    get_system_times = kernel32.GetSystemTimes
    get_system_times.argtypes = [
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
    ]
    get_system_times.restype = wintypes.BOOL
    idle = _FileTime()
    kernel = _FileTime()
    user = _FileTime()
    if not get_system_times(
        ctypes.byref(idle),
        ctypes.byref(kernel),
        ctypes.byref(user),
    ):
        raise G2Blocked("g2_formal_cpu_probe_failed")
    return (
        _file_time_ticks(idle),
        _file_time_ticks(kernel),
        _file_time_ticks(user),
    )


def _read_disk_busy_counter() -> tuple[int, int]:
    parsed, _probe_sha256 = _powershell_json_probe(
        (
            "$row=Get-CimInstance -ClassName "
            "Win32_PerfRawData_PerfDisk_LogicalDisk "
            "-Filter \"Name='D:'\" |"
            "Select-Object -First 1 PercentDiskTime,Timestamp_Sys100NS;"
            "if($null -eq $row){throw 'D disk counter missing'};"
            "ConvertTo-Json -InputObject $row -Compress"
        ),
        reason="g2_formal_disk_busy_probe_failed",
    )
    if not isinstance(parsed, Mapping):
        raise G2Blocked("g2_formal_disk_busy_probe_failed")
    try:
        busy_ticks = int(parsed["PercentDiskTime"])
        timestamp_ticks = int(parsed["Timestamp_Sys100NS"])
    except (KeyError, TypeError, ValueError) as exc:
        raise G2Blocked("g2_formal_disk_busy_probe_failed") from exc
    if busy_ticks < 0 or timestamp_ticks <= 0:
        raise G2Blocked("g2_formal_disk_busy_probe_failed")
    return busy_ticks, timestamp_ticks


def _sample_cpu_and_disk_busy(
    sample_window_seconds: float,
) -> tuple[float, float]:
    if sample_window_seconds != 2.0:
        raise G2Blocked("g2_formal_sample_window_invalid")
    idle_before, kernel_before, user_before = _read_system_times()
    disk_busy_before, disk_time_before = _read_disk_busy_counter()
    threading.Event().wait(sample_window_seconds)
    idle_after, kernel_after, user_after = _read_system_times()
    disk_busy_after, disk_time_after = _read_disk_busy_counter()

    idle_delta = idle_after - idle_before
    total_delta = (
        kernel_after - kernel_before + user_after - user_before
    )
    disk_time_delta = disk_time_after - disk_time_before
    disk_busy_delta = disk_busy_after - disk_busy_before
    if (
        idle_delta < 0
        or total_delta <= 0
        or idle_delta > total_delta
        or disk_time_delta <= 0
        or disk_busy_delta < 0
    ):
        raise G2Blocked("g2_formal_load_probe_failed")
    cpu_percent = 100.0 * (total_delta - idle_delta) / total_delta
    disk_busy_percent = 100.0 * disk_busy_delta / disk_time_delta
    if (
        not math.isfinite(cpu_percent)
        or not math.isfinite(disk_busy_percent)
        or cpu_percent < 0.0
        or disk_busy_percent < 0.0
    ):
        raise G2Blocked("g2_formal_load_probe_failed")
    return cpu_percent, disk_busy_percent


def _probe_memory_status() -> tuple[float, int]:
    class MemoryStatus(ctypes.Structure):
        _fields_ = (
            ("length", wintypes.DWORD),
            ("memory_load", wintypes.DWORD),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        )

    kernel32 = _windows_kernel32()
    global_memory_status = kernel32.GlobalMemoryStatusEx
    global_memory_status.argtypes = [ctypes.POINTER(MemoryStatus)]
    global_memory_status.restype = wintypes.BOOL
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(MemoryStatus)
    if not global_memory_status(ctypes.byref(memory)):
        raise G2Blocked("g2_formal_memory_probe_failed")
    return float(memory.memory_load), int(memory.available_physical)


def _probe_disk_free() -> tuple[str, int]:
    try:
        free_bytes = int(shutil.disk_usage("D:/").free)
    except OSError as exc:
        raise G2Blocked("g2_formal_disk_free_probe_failed") from exc
    if free_bytes < 0:
        raise G2Blocked("g2_formal_disk_free_probe_failed")
    return "D:/", free_bytes


def capture_windows_formal_environment(
    phase: str,
    policy: Mapping[str, object],
    *,
    platform_name: str | None = None,
) -> dict[str, object]:
    validated_policy = validate_formal_environment_policy(policy)
    if phase not in {"start", "end"}:
        raise G2Blocked("g2_formal_environment_phase_invalid")
    actual_platform = os.name if platform_name is None else platform_name
    if actual_platform != "nt":
        raise G2Blocked("g2_formal_windows_required")

    power_scheme_guid, power_probe_sha256 = (
        _probe_active_power_scheme()
    )
    competing, excluded, process_inventory_sha256 = (
        _probe_process_inventory(
            validated_policy["competing_process_patterns"],
            os.getpid(),
        )
    )
    cpu_percent, disk_busy_percent = _sample_cpu_and_disk_busy(
        float(validated_policy["sample_window_seconds"])
    )
    memory_percent, memory_available_bytes = _probe_memory_status()
    disk_root, disk_free_bytes = _probe_disk_free()
    thread_names = tuple(
        validated_policy["required_thread_variables"]
    )
    return {
        "schema_version": (
            FORMAL_ENVIRONMENT_OBSERVATION_SCHEMA_VERSION
        ),
        "phase": phase,
        "captured_utc": _utc_now(),
        "host": host_platform.node() or "unknown-host",
        "pid": os.getpid(),
        "power_scheme_guid": power_scheme_guid,
        "power_probe_sha256": power_probe_sha256,
        "thread_variables": {
            name: os.environ.get(name, "not-recorded")
            for name in thread_names
        },
        "sample_window_seconds": validated_policy[
            "sample_window_seconds"
        ],
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "memory_available_bytes": memory_available_bytes,
        "disk_busy_percent": disk_busy_percent,
        "disk_free_bytes": disk_free_bytes,
        "disk_root": disk_root,
        "competing_processes": competing,
        "excluded_process_ids": excluded,
        "process_inventory_sha256": process_inventory_sha256,
    }


def validate_formal_environment_policy(
    value: object,
) -> dict[str, object]:
    required = {
        "schema_version",
        "lease_path",
        "allowed_power_scheme_guids",
        "required_thread_variables",
        "sample_window_seconds",
        "limits",
        "competing_process_patterns",
        "exclude_current_process_tree",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise G2Blocked("g2_formal_environment_policy_invalid")
    policy = dict(value)
    lease_path = policy["lease_path"]
    if (
        policy["schema_version"]
        != FORMAL_ENVIRONMENT_POLICY_SCHEMA_VERSION
        or type(lease_path) is not str
        or not lease_path
        or not PureWindowsPath(lease_path).is_absolute()
        or PureWindowsPath(lease_path).drive.upper() != "D:"
    ):
        raise G2Blocked("g2_formal_environment_policy_invalid")
    allowed = policy["allowed_power_scheme_guids"]
    if (
        type(allowed) is not list
        or not allowed
        or len(allowed) != len(set(allowed))
        or any(
            type(item) is not str
            or re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                r"[0-9a-f]{4}-[0-9a-f]{12}",
                item,
            )
            is None
            for item in allowed
        )
    ):
        raise G2Blocked("g2_formal_environment_policy_invalid")
    required_threads = {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    }
    thread_variables = policy["required_thread_variables"]
    if (
        not isinstance(thread_variables, Mapping)
        or set(thread_variables) != required_threads
        or any(
            type(thread_variables[name]) is not str
            or not thread_variables[name]
            for name in required_threads
        )
    ):
        raise G2Blocked("g2_formal_environment_policy_invalid")
    limits = policy["limits"]
    limit_names = {
        "cpu_percent_max",
        "memory_percent_max",
        "memory_available_bytes_min",
        "disk_busy_percent_max",
        "disk_free_bytes_min",
    }
    if (
        not isinstance(limits, Mapping)
        or set(limits) != limit_names
        or any(
            isinstance(limits[name], bool)
            or not isinstance(limits[name], (int, float))
            or not math.isfinite(float(limits[name]))
            or float(limits[name]) < 0.0
            for name in limit_names
        )
        or float(limits["cpu_percent_max"]) > 100.0
        or float(limits["memory_percent_max"]) > 100.0
        or float(limits["disk_busy_percent_max"]) > 100.0
    ):
        raise G2Blocked("g2_formal_environment_policy_invalid")
    patterns = policy["competing_process_patterns"]
    if (
        type(patterns) is not list
        or not patterns
        or len(patterns) != len(set(patterns))
        or any(type(item) is not str or not item for item in patterns)
        or policy["sample_window_seconds"] != 2.0
        or policy["exclude_current_process_tree"] is not True
    ):
        raise G2Blocked("g2_formal_environment_policy_invalid")
    return {
        "schema_version": FORMAL_ENVIRONMENT_POLICY_SCHEMA_VERSION,
        "lease_path": str(lease_path).replace("\\", "/"),
        "allowed_power_scheme_guids": list(allowed),
        "required_thread_variables": {
            name: str(thread_variables[name])
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "sample_window_seconds": 2.0,
        "limits": {
            name: limits[name]
            for name in (
                "cpu_percent_max",
                "memory_percent_max",
                "memory_available_bytes_min",
                "disk_busy_percent_max",
                "disk_free_bytes_min",
            )
        },
        "competing_process_patterns": list(patterns),
        "exclude_current_process_tree": True,
    }


class G2FormalLease:
    """One exact atomic file owned by one process and one nonce."""

    def __init__(
        self,
        *,
        lease_path: str | Path,
        run_id: str,
        run_root: str | Path,
    ) -> None:
        self.lease_path = Path(lease_path)
        self.run_id = _validated_run_id(run_id)
        self.run_root = str(Path(run_root)).replace("\\", "/")
        self._payload: dict[str, object] | None = None

    @property
    def payload(self) -> dict[str, object]:
        if self._payload is None:
            raise G2Blocked("g2_formal_lease_not_acquired")
        return dict(self._payload)

    def acquire(self) -> dict[str, object]:
        if self._payload is not None:
            raise G2Blocked("g2_formal_lease_already_acquired")
        core = {
            "schema_version": FORMAL_LEASE_SCHEMA_VERSION,
            "host": host_platform.node() or "unknown-host",
            "pid": os.getpid(),
            "process_start_utc": _current_process_start_utc(),
            "run_id": self.run_id,
            "run_root": self.run_root,
            "nonce": secrets.token_hex(16),
            "acquired_utc": _utc_now(),
        }
        payload = {
            **core,
            "lease_sha256": _domain_hash(
                FORMAL_LEASE_SCHEMA_VERSION,
                _canonical_bytes(core),
            ),
        }
        artifact_io.ensure_parent(self.lease_path)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= int(getattr(os, "O_BINARY", 0))
        open_descriptor = getattr(os, "open")
        try:
            descriptor = open_descriptor(
                artifact_io.windows_safe_path(self.lease_path),
                flags,
                0o600,
            )
        except FileExistsError as exc:
            raise G2Blocked("g2_formal_lease_contended") from exc
        try:
            os.write(
                descriptor,
                (
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8"),
            )
            os.fsync(descriptor)
        except BaseException:
            os.close(descriptor)
            self._payload = payload
            self.release()
            raise
        else:
            os.close(descriptor)
        self._payload = payload
        return dict(payload)

    def release(self) -> bool:
        if self._payload is None or not artifact_io.path_is_file(
            self.lease_path
        ):
            return False
        try:
            stored = artifact_io.read_json(self.lease_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if stored != self._payload:
            return False
        try:
            getattr(os, "unlink")(
                artifact_io.windows_safe_path(self.lease_path)
            )
        except OSError:
            return False
        self._payload = None
        return True


def validate_g2_formal_environment(
    observation: object,
    policy: object,
) -> dict[str, object]:
    validated_policy = validate_formal_environment_policy(policy)
    required = {
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
    }
    if not isinstance(observation, Mapping) or set(observation) != required:
        raise G2Blocked("g2_formal_environment_observation_invalid")
    result = dict(observation)
    if (
        result["schema_version"]
        != FORMAL_ENVIRONMENT_OBSERVATION_SCHEMA_VERSION
        or result["phase"] not in {"start", "end"}
        or type(result["captured_utc"]) is not str
        or not result["captured_utc"]
        or type(result["host"]) is not str
        or not result["host"]
        or type(result["pid"]) is not int
        or result["pid"] <= 0
        or not _is_sha256(result["power_probe_sha256"])
        or not _is_sha256(result["process_inventory_sha256"])
        or type(result["disk_root"]) is not str
        or not result["disk_root"]
        or result["sample_window_seconds"]
        != validated_policy["sample_window_seconds"]
    ):
        raise G2Blocked("g2_formal_environment_observation_invalid")
    power = result["power_scheme_guid"]
    if power == "not-recorded":
        raise G2Blocked("g2_formal_environment_not_recorded")
    if (
        type(power) is not str
        or power.casefold()
        not in {
            str(item).casefold()
            for item in validated_policy[
                "allowed_power_scheme_guids"
            ]
        }
    ):
        raise G2Blocked("g2_formal_power_scheme_not_allowed")
    if result["thread_variables"] != validated_policy[
        "required_thread_variables"
    ]:
        raise G2Blocked("g2_formal_thread_settings_invalid")
    competing = result["competing_processes"]
    if type(competing) is not list:
        raise G2Blocked("g2_formal_environment_observation_invalid")
    for row in competing:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "pid",
                "parent_pid",
                "name",
                "matched_pattern",
                "command_sha256",
            }
            or type(row["pid"]) is not int
            or type(row["parent_pid"]) is not int
            or type(row["name"]) is not str
            or type(row["matched_pattern"]) is not str
            or not _is_sha256(row["command_sha256"])
        ):
            raise G2Blocked("g2_formal_environment_observation_invalid")
    if competing:
        raise G2Blocked("g2_formal_competing_process")
    excluded = result["excluded_process_ids"]
    if (
        type(excluded) is not list
        or not excluded
        or any(type(item) is not int or item <= 0 for item in excluded)
    ):
        raise G2Blocked("g2_formal_environment_observation_invalid")
    numeric_fields = (
        "cpu_percent",
        "memory_percent",
        "memory_available_bytes",
        "disk_busy_percent",
        "disk_free_bytes",
    )
    for name in numeric_fields:
        value = result[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise G2Blocked("g2_formal_environment_observation_invalid")
    limits = validated_policy["limits"]
    if float(result["cpu_percent"]) > float(limits["cpu_percent_max"]):
        raise G2Blocked("g2_formal_cpu_load_exceeded")
    if float(result["memory_percent"]) > float(
        limits["memory_percent_max"]
    ):
        raise G2Blocked("g2_formal_memory_load_exceeded")
    if int(result["memory_available_bytes"]) < int(
        limits["memory_available_bytes_min"]
    ):
        raise G2Blocked("g2_formal_memory_available_below_min")
    if float(result["disk_busy_percent"]) > float(
        limits["disk_busy_percent_max"]
    ):
        raise G2Blocked("g2_formal_disk_busy_exceeded")
    if int(result["disk_free_bytes"]) < int(
        limits["disk_free_bytes_min"]
    ):
        raise G2Blocked("g2_formal_disk_free_below_min")
    return result


class G2FormalEnvironmentGuard:
    """Acquire, validate start/end observations, then release exactly once."""

    def __init__(
        self,
        *,
        run_id: str,
        policy: Mapping[str, object],
        lease: G2FormalLease,
        capture_environment: Callable[
            [str, Mapping[str, object]],
            Mapping[str, object],
        ],
    ) -> None:
        self.run_id = _validated_run_id(run_id)
        self.policy = validate_formal_environment_policy(policy)
        self.lease = lease
        if (
            str(self.lease.lease_path).replace("\\", "/").casefold()
            != str(self.policy["lease_path"]).casefold()
        ):
            raise G2Blocked("g2_formal_lease_policy_mismatch")
        self.capture_environment = capture_environment
        self.formal_row_count = 0
        self.audit: dict[str, object] = {}
        self._lease_payload: dict[str, object] | None = None
        self._start: dict[str, object] | None = None
        self._end: dict[str, object] | None = None
        self._blockers: list[str] = []
        self._closed = False

    def set_formal_row_count(self, value: int) -> None:
        self.formal_row_count = _exact_nonnegative_int(
            value,
            "g2_formal_environment_row_count",
        )

    @staticmethod
    def _reason(exc: BaseException) -> str:
        if isinstance(exc, G2Blocked):
            return exc.reason
        if isinstance(exc, G2Interrupted):
            return exc.reason
        return f"g2_formal_body_exception:{type(exc).__name__}"

    def _capture(self, phase: str) -> dict[str, object]:
        try:
            raw = self.capture_environment(phase, self.policy)
        except BaseException as exc:
            failure = {
                "schema_version": (
                    "xunce-mid-dual-g2-formal-environment-probe-error/v1"
                ),
                "phase": phase,
                "captured_utc": _utc_now(),
                "reason": self._reason(exc),
            }
            if phase == "start":
                self._start = failure
            else:
                self._end = failure
            raise
        captured = (
            dict(raw)
            if isinstance(raw, Mapping)
            else {
                "schema_version": (
                    "xunce-mid-dual-g2-formal-environment-probe-error/v1"
                ),
                "phase": phase,
                "captured_utc": _utc_now(),
                "reason": "g2_formal_environment_observation_invalid",
            }
        )
        if phase == "start":
            self._start = captured
        else:
            self._end = captured
        validated = validate_g2_formal_environment(raw, self.policy)
        if self._lease_payload is None:
            raise G2Blocked("g2_formal_lease_not_acquired")
        if (
            validated["host"] != self._lease_payload["host"]
            or validated["pid"] != self._lease_payload["pid"]
            or validated["pid"] not in validated["excluded_process_ids"]
        ):
            raise G2Blocked("g2_formal_environment_process_mismatch")
        if phase == "start":
            self._start = validated
        else:
            self._end = validated
        return validated

    def _build_audit(self, *, lease_released: bool) -> None:
        if self._lease_payload is None:
            return
        start = self._start or {
            "schema_version": (
                "xunce-mid-dual-g2-formal-environment-probe-error/v1"
            ),
            "phase": "start",
            "captured_utc": _utc_now(),
            "reason": "g2_formal_environment_not_recorded",
        }
        end = self._end or {
            "schema_version": (
                "xunce-mid-dual-g2-formal-environment-probe-error/v1"
            ),
            "phase": "end",
            "captured_utc": _utc_now(),
            "reason": "g2_formal_environment_not_recorded",
        }
        passed = not self._blockers and lease_released
        self.audit = {
            "schema_version": FORMAL_ENVIRONMENT_AUDIT_SCHEMA_VERSION,
            "gate_id": G2_GATE_ID,
            "scale_profile": SCALE_PROFILE,
            "run_id": self.run_id,
            "status": "passed" if passed else "blocked",
            "formal_evidence_eligible": passed,
            "formal_environment_policy": dict(self.policy),
            "formal_environment_policy_sha256": _canonical_sha256(
                self.policy
            ),
            "lease": dict(self._lease_payload),
            "lease_sha256": self._lease_payload["lease_sha256"],
            "start_observation": start,
            "start_observation_sha256": _canonical_sha256(start),
            "end_observation": end,
            "end_observation_sha256": _canonical_sha256(end),
            "same_process": (
                start.get("pid") == self._lease_payload["pid"]
                and end.get("pid") == self._lease_payload["pid"]
            ),
            "formal_row_count": self.formal_row_count if passed else 0,
            "lease_released": lease_released,
            "blockers": list(dict.fromkeys(self._blockers)),
        }

    def _finish(self, original: BaseException | None) -> None:
        if self._closed:
            return
        deferred: BaseException | None = None
        try:
            self._capture("end")
        except BaseException as exc:
            self._blockers.append(self._reason(exc))
            deferred = exc
        if original is not None:
            self._blockers.append(self._reason(original))
        elif self.formal_row_count != G2_FORMAL_CALLS:
            row_count_error = G2Blocked(
                "g2_formal_row_count_invalid"
            )
            self._blockers.append(row_count_error.reason)
            if deferred is None:
                deferred = row_count_error
        released = self.lease.release()
        if not released:
            self._blockers.append("g2_formal_lease_release_failed")
            if deferred is None:
                deferred = G2Blocked("g2_formal_lease_release_failed")
        self._closed = True
        self._build_audit(lease_released=released)
        if original is None and deferred is not None:
            raise deferred

    def __enter__(self) -> "G2FormalEnvironmentGuard":
        self._lease_payload = self.lease.acquire()
        try:
            self._capture("start")
        except BaseException as exc:
            self._blockers.append(self._reason(exc))
            self._finish(exc)
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        del exc_type, traceback
        self._finish(exc)
        return False


def formal_environment_audit_binding(
    audit: object,
) -> dict[str, object]:
    if (
        not isinstance(audit, Mapping)
        or audit.get("schema_version")
        != FORMAL_ENVIRONMENT_AUDIT_SCHEMA_VERSION
        or not _is_sha256(audit.get("lease_sha256"))
    ):
        raise G2Blocked("g2_formal_environment_audit_invalid")
    return {
        "formal_environment_audit_schema_version": (
            FORMAL_ENVIRONMENT_AUDIT_SCHEMA_VERSION
        ),
        "formal_environment_audit_sha256": _canonical_sha256(audit),
        "formal_lease_sha256": audit["lease_sha256"],
    }


def _validate_complete_formal_environment_audit(
    audit: object,
) -> dict[str, object]:
    required = {
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
    }
    if not isinstance(audit, Mapping) or set(audit) != required:
        raise G2Blocked("g2_formal_environment_audit_invalid")
    result = dict(audit)
    policy = validate_formal_environment_policy(
        result["formal_environment_policy"]
    )
    lease = result["lease"]
    lease_keys = {
        "schema_version",
        "host",
        "pid",
        "process_start_utc",
        "run_id",
        "run_root",
        "nonce",
        "acquired_utc",
        "lease_sha256",
    }
    if not isinstance(lease, Mapping) or set(lease) != lease_keys:
        raise G2Blocked("g2_formal_environment_audit_invalid")
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
    expected_lease_sha256 = _domain_hash(
        FORMAL_LEASE_SCHEMA_VERSION,
        _canonical_bytes(lease_core),
    )
    start = validate_g2_formal_environment(
        result["start_observation"],
        policy,
    )
    end = validate_g2_formal_environment(
        result["end_observation"],
        policy,
    )
    if (
        result["schema_version"]
        != FORMAL_ENVIRONMENT_AUDIT_SCHEMA_VERSION
        or result["gate_id"] != G2_GATE_ID
        or result["scale_profile"] != SCALE_PROFILE
        or result["status"] != "passed"
        or result["formal_evidence_eligible"] is not True
        or result["formal_environment_policy_sha256"]
        != _canonical_sha256(policy)
        or lease["schema_version"] != FORMAL_LEASE_SCHEMA_VERSION
        or lease["lease_sha256"] != expected_lease_sha256
        or result["lease_sha256"] != expected_lease_sha256
        or result["run_id"] != lease["run_id"]
        or start["phase"] != "start"
        or end["phase"] != "end"
        or start["host"] != lease["host"]
        or end["host"] != lease["host"]
        or start["pid"] != lease["pid"]
        or end["pid"] != lease["pid"]
        or start["pid"] not in start["excluded_process_ids"]
        or end["pid"] not in end["excluded_process_ids"]
        or result["start_observation_sha256"]
        != _canonical_sha256(start)
        or result["end_observation_sha256"]
        != _canonical_sha256(end)
        or result["same_process"] is not True
        or result["formal_row_count"] != G2_FORMAL_CALLS
        or result["lease_released"] is not True
        or result["blockers"] != []
    ):
        raise G2Blocked("g2_formal_environment_audit_invalid")
    return result


def formal_environment_phase_audit_fields(
    audit: object,
) -> dict[str, object]:
    validated = _validate_complete_formal_environment_audit(audit)
    return {
        "formal_environment_audit": validated,
        **formal_environment_audit_binding(validated),
    }


def validate_formal_environment_phase_audit(
    phase_audit: object,
) -> dict[str, object]:
    if (
        not isinstance(phase_audit, Mapping)
        or phase_audit.get("phase_id") != "p04"
        or not isinstance(
            phase_audit.get("formal_environment_audit"),
            Mapping,
        )
    ):
        raise G2Blocked("g2_formal_environment_phase_binding_invalid")
    embedded = _validate_complete_formal_environment_audit(
        phase_audit["formal_environment_audit"]
    )
    binding = formal_environment_audit_binding(embedded)
    if any(phase_audit.get(key) != value for key, value in binding.items()):
        raise G2Blocked("g2_formal_environment_phase_binding_invalid")
    return embedded


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise G2Blocked("g2_noncanonical_json") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _domain_hash(domain: str, *parts: bytes) -> str:
    framed = bytearray()
    for part in (domain.encode("utf-8"), *parts):
        framed.extend(struct.pack(">Q", len(part)))
        framed.extend(part)
    return _sha256(bytes(framed))


def _canonical_sha256(value: object) -> str:
    return _sha256(_canonical_bytes(value))


def _json_artifact_sha256(value: Mapping[str, object]) -> str:
    payload = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return _sha256(payload)


def _is_sha256(value: object) -> bool:
    return type(value) is str and SHA256_RE.fullmatch(value) is not None


def _require_sha256(value: object, reason: str) -> str:
    if not _is_sha256(value):
        raise G2Blocked(reason)
    return str(value)


def _exact_bool(value: object, reason: str) -> bool:
    if type(value) is not bool:
        raise G2Blocked(reason)
    return bool(value)


def _exact_nonnegative_int(value: object, reason: str) -> int:
    if type(value) is not int or value < 0:
        raise G2Blocked(reason)
    return int(value)


def _finite_nonnegative(value: object, reason: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise G2Blocked(reason)
    return float(value)


def _safe_relative_path(value: object) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise G2Blocked("g2_execution_payload_path_invalid")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise G2Blocked("g2_execution_payload_path_invalid")
    return value


def _validated_run_id(value: object) -> str:
    if type(value) is not str or RUN_ID_RE.fullmatch(value) is None:
        raise G2Blocked("g2_run_id_invalid")
    if value.split(".", 1)[0].upper() in WINDOWS_RESERVED:
        raise G2Blocked("g2_run_id_windows_reserved")
    return value


def _normalized_request(row: Mapping[str, object]) -> dict[str, object]:
    platform = row.get("platform", row.get("platform_kind"))
    scale = row.get("scale")
    request_id = row.get("request_id")
    request_sha256 = row.get(
        "provider_request_sha256",
        row.get("request_sha256"),
    )
    truth_sha256 = row.get(
        "truth_request_sha256",
        row.get("truth_sha256"),
    )
    oracle_result_sha256 = row.get(
        "truth_certificate_sha256",
        row.get("oracle_result_sha256"),
    )
    request_class = row.get("request_class")
    outcome_kind = row.get("outcome_kind")
    if (
        platform not in G2_PLATFORMS
        or scale not in G2_SCALES
        or type(request_id) is not str
        or not request_id
        or request_class
        not in {"normal_reachable", "hard_reachable", "unreachable"}
        or outcome_kind not in {"reachable", "unreachable"}
    ):
        raise G2Blocked("g2_request_matrix")
    expected_outcome = (
        "unreachable" if request_class == "unreachable" else "reachable"
    )
    if outcome_kind != expected_outcome:
        raise G2Blocked("g2_request_matrix")
    return {
        "platform": platform,
        "scale": scale,
        "request_index": _exact_nonnegative_int(
            row.get("request_index"),
            "g2_request_matrix",
        ),
        "request_id": request_id,
        "request_sha256": _require_sha256(
            request_sha256,
            "g2_request_hash",
        ),
        "provider_request_sha256": _require_sha256(
            request_sha256,
            "g2_request_hash",
        ),
        "truth_sha256": _require_sha256(
            truth_sha256,
            "g2_truth_hash",
        ),
        "truth_request_sha256": _require_sha256(
            truth_sha256,
            "g2_truth_hash",
        ),
        "truth_certificate_sha256": _require_sha256(
            oracle_result_sha256,
            "g2_oracle_result_hash",
        ),
        "terrain_sha256": _require_sha256(
            row.get("terrain_sha256"),
            "g2_terrain_hash",
        ),
        "request_class": request_class,
        "outcome_kind": outcome_kind,
        "expected_success": outcome_kind == "reachable",
        "expected_route_l2_valid": outcome_kind == "reachable",
    }


def _validated_request_matrix(
    requests: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if not isinstance(requests, Sequence) or len(requests) != 129:
        raise G2Blocked("g2_request_matrix")
    normalized = [_normalized_request(row) for row in requests]
    if (
        len({row["request_id"] for row in normalized}) != 129
        or len({row["request_sha256"] for row in normalized}) != 129
        or len({row["truth_sha256"] for row in normalized}) != 129
    ):
        raise G2Blocked("g2_request_matrix")
    for platform in G2_PLATFORMS:
        selected = [row for row in normalized if row["platform"] == platform]
        if len(selected) != 43 or sorted(
            int(row["request_index"]) for row in selected
        ) != list(range(43)):
            raise G2Blocked("g2_request_matrix")
        for scale in G2_SCALES:
            counts = Counter(
                str(row["request_class"])
                for row in selected
                if row["scale"] == scale
            )
            if dict(counts) != EXPECTED_CLASS_COUNTS[scale]:
                raise G2Blocked("g2_request_matrix")
    return normalized


def _scheduled_call(
    request: Mapping[str, object],
    *,
    repeat_index: int,
    formal_sample: bool,
    schedule_kind: str,
) -> dict[str, object]:
    request_sha256 = str(request["request_sha256"])
    repeat_tag = (
        f"r{repeat_index}"
        if repeat_index >= 0
        else f"d{schedule_kind}"
    )
    call_id = (
        f"g2-call-{request_sha256[:24]}-{repeat_tag}"
        if formal_sample
        else (
            f"g2-diagnostic-{schedule_kind}-"
            f"{request_sha256[:20]}-{max(repeat_index, 0)}"
        )
    )
    return {
        **dict(request),
        "call_id": call_id,
        "repeat_index": repeat_index,
        "formal_sample": formal_sample,
        "schedule_kind": schedule_kind,
    }


def build_formal_schedule(
    input_set_id: str,
    requests: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if type(input_set_id) is not str or not input_set_id:
        raise G2Blocked("g2_input_set_id")
    normalized = _validated_request_matrix(requests)
    calls: list[dict[str, object]] = []
    for platform in G2_PLATFORMS:
        platform_rows = [
            row for row in normalized if row["platform"] == platform
        ]
        for repeat_index in range(G2_REPEATS):
            ordered = sorted(
                platform_rows,
                key=lambda row: _domain_hash(
                    "xunce-mid-dual-g2-formal-order/v1",
                    input_set_id.encode("utf-8"),
                    platform.encode("ascii"),
                    str(repeat_index).encode("ascii"),
                    str(row["request_sha256"]).encode("ascii"),
                ),
            )
            calls.extend(
                _scheduled_call(
                    row,
                    repeat_index=repeat_index,
                    formal_sample=True,
                    schedule_kind="formal",
                )
                for row in ordered
            )
    for submission_index, call in enumerate(calls):
        call["submission_index"] = submission_index
    if (
        len(calls) != G2_FORMAL_CALLS
        or len({call["call_id"] for call in calls}) != G2_FORMAL_CALLS
    ):
        raise G2Blocked("g2_formal_schedule")
    core = {
        "schema_version": "xunce-mid-dual-g2-formal-schedule/v1",
        "input_set_id": input_set_id,
        "repeat_count": G2_REPEATS,
        "formal_worker_count": 4,
        "calls": calls,
    }
    return {
        **core,
        "schedule_sha256": _domain_hash(
            "xunce-mid-dual-g2-formal-schedule/v1",
            _canonical_bytes(core),
        ),
    }


def build_nonformal_schedules(
    requests: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    normalized = _validated_request_matrix(requests)
    cold: list[dict[str, object]] = []
    warmup: list[dict[str, object]] = []
    worker_one: list[dict[str, object]] = []
    worker_four: list[dict[str, object]] = []
    for platform in G2_PLATFORMS:
        selected = sorted(
            (row for row in normalized if row["platform"] == platform),
            key=lambda row: (
                str(row["scale"]),
                int(row["request_index"]),
                str(row["request_sha256"]),
            ),
        )
        cold.append(
            _scheduled_call(
                selected[0],
                repeat_index=-1,
                formal_sample=False,
                schedule_kind="cold",
            )
        )
        warmup.extend(
            _scheduled_call(
                row,
                repeat_index=-1,
                formal_sample=False,
                schedule_kind=f"warmup-{index:02d}",
            )
            for index, row in enumerate(selected[:10])
        )
        worker_one.extend(
            _scheduled_call(
                row,
                repeat_index=-1,
                formal_sample=False,
                schedule_kind="worker-one",
            )
            for row in selected
        )
        worker_four.extend(
            _scheduled_call(
                row,
                repeat_index=-1,
                formal_sample=False,
                schedule_kind="worker-four",
            )
            for row in selected
        )
    return {
        "cold_start": cold,
        "warmup": warmup,
        "worker_one": worker_one,
        "worker_four": worker_four,
    }


def execute_preloaded_batch(
    tasks: Sequence[Mapping[str, object]],
    *,
    prepare_task: Callable[[Mapping[str, object]], object],
    execute_task: Callable[[object], object],
    max_workers: int,
) -> list[object]:
    if type(max_workers) is not int or max_workers <= 0:
        raise G2Blocked("g2_worker_count")
    prepared = [prepare_task(task) for task in tasks]
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="g2-planner",
    ) as executor:
        return list(executor.map(execute_task, prepared))


def validate_timing_payload(
    payload: Mapping[str, object],
) -> dict[str, int]:
    if type(payload) is not dict or set(payload) != {
        *TIMING_FIELDS,
        "total_ns",
    }:
        raise G2Blocked("g2_timing_contract_schema")
    values = {
        field: _exact_nonnegative_int(
            payload[field],
            "g2_timing_contract_value",
        )
        for field in (*TIMING_FIELDS, "total_ns")
    }
    if values["total_ns"] != sum(values[field] for field in TIMING_FIELDS):
        raise G2Blocked("g2_timing_contract_sum")
    return values


def _timing_ms_projection(timing: Mapping[str, int]) -> dict[str, float]:
    return {
        **{
            milliseconds: timing[nanoseconds] / 1_000_000.0
            for nanoseconds, milliseconds in zip(
                TIMING_FIELDS,
                TIMING_MS_FIELDS,
                strict=True,
            )
        },
        "elapsed_ms": timing["total_ns"] / 1_000_000.0,
    }


def expected_semantic_digest(
    *,
    request_sha256: object,
    provider_success: object,
    route_l2_valid: object,
) -> str:
    request_hash = _require_sha256(
        request_sha256,
        "g2_semantic_request_hash",
    )
    success = _exact_bool(provider_success, "g2_semantic_success")
    valid = _exact_bool(route_l2_valid, "g2_semantic_l2")
    return _domain_hash(
        "xunce-mid-dual-g2-provider-semantics/v1",
        _canonical_bytes(
            {
                "request_sha256": request_hash,
                "provider_success": success,
                "route_l2_valid": valid,
            }
        ),
    )


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(1, math.ceil(quantile * len(ordered))) - 1
    return ordered[index]


def _request_bootstrap_ci(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["request_id"]), []).append(
            float(row["elapsed_ms"])
        )
    request_means = [
        statistics.mean(values)
        for _request_id, values in sorted(grouped.items())
    ]
    generator = random.Random(20260726)
    samples = sorted(
        statistics.mean(
            generator.choice(request_means) for _ in request_means
        )
        for _ in range(2000)
    )
    return {
        "unit": "unique_request",
        "seed": 20260726,
        "resamples": 2000,
        "confidence_level": 0.95,
        "lower_mean_ms": _nearest_rank(samples, 0.025),
        "upper_mean_ms": _nearest_rank(samples, 0.975),
    }


def _timing_statistics(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not rows:
        raise G2Blocked("g2_timing_partition_empty")
    values = [float(row["elapsed_ms"]) for row in rows]
    mean_ms = statistics.mean(values)
    p95_ms = _nearest_rank(values, 0.95)
    maximum = max(values)
    at_or_below = sum(value <= 1000.0 for value in values)
    midterm = mean_ms <= 2000.0 and p95_ms <= 2000.0 and maximum <= 2000.0
    final = (
        mean_ms <= 1000.0
        and p95_ms <= 1000.0
        and at_or_below / len(values) >= 0.95
        and maximum <= 2000.0
    )
    return {
        "sample_count": len(values),
        "unique_request_count": len(
            {str(row["request_id"]) for row in rows}
        ),
        "mean_ms": mean_ms,
        "p50_ms": _nearest_rank(values, 0.50),
        "p95_ms": p95_ms,
        "p99_ms": _nearest_rank(values, 0.99),
        "sample_stddev_ms": (
            statistics.stdev(values) if len(values) > 1 else 0.0
        ),
        "min_ms": min(values),
        "max_ms": maximum,
        "at_or_below_1000_count": at_or_below,
        "proportion_at_or_below_1000ms": at_or_below / len(values),
        "over_2000_count": sum(value > 2000.0 for value in values),
        "midterm_reduced_passed": midterm,
        "final_threshold_reduced_passed": final,
        "bootstrap_ci": _request_bootstrap_ci(rows),
    }


def _validated_formal_row(raw: object) -> dict[str, object]:
    if type(raw) is not dict or set(raw) != G2_ROW_KEYS:
        raise G2Blocked("g2_row_schema")
    row = dict(raw)
    if (
        row["row_kind"] != "g2_planning_call"
        or row["schema_version"] != G2_ROW_SCHEMA_VERSION
        or row["scale_profile"] != SCALE_PROFILE
        or row["platform"] not in G2_PLATFORMS
        or row["scale"] not in G2_SCALES
        or row["request_class"]
        not in {"normal_reachable", "hard_reachable", "unreachable"}
        or row["outcome_kind"] not in {"reachable", "unreachable"}
        or row["formal_sample"] is not True
        or row["timing_contract_id"] != TIMING_CONTRACT_ID
        or type(row["run_id"]) is not str
        or not row["run_id"]
        or type(row["request_id"]) is not str
        or not row["request_id"]
        or row["episode_id"] != row["request_id"]
        or type(row["call_id"]) is not str
        or not row["call_id"]
    ):
        raise G2Blocked("g2_row_schema")
    for field in (
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
        "semantic_digest",
    ):
        _require_sha256(row[field], "g2_row_hash")
    repeat = _exact_nonnegative_int(
        row["repeat_index"],
        "g2_repeat_matrix",
    )
    if repeat >= G2_REPEATS:
        raise G2Blocked("g2_repeat_matrix")
    timing = validate_timing_payload(
        {
            field: row[field]
            for field in (*TIMING_FIELDS, "total_ns")
        }
    )
    expected_ms = _timing_ms_projection(timing)
    for field, expected in expected_ms.items():
        actual = _finite_nonnegative(row[field], "g2_timing_contract_ms")
        if actual != expected:
            raise G2Blocked("g2_timing_contract_ms")
    success = _exact_bool(row["provider_success"], "g2_row_success")
    route_valid = _exact_bool(row["route_l2_valid"], "g2_row_l2")
    expected_semantic = expected_semantic_digest(
        request_sha256=row["request_sha256"],
        provider_success=success,
        route_l2_valid=route_valid,
    )
    if row["semantic_digest"] != expected_semantic:
        raise G2Blocked("g2_semantic_digest")
    return row


def recompute_g2_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(rows) != G2_FORMAL_CALLS:
        raise G2Blocked("g2_formal_call_count")
    materialized = [_validated_formal_row(row) for row in rows]
    if len({str(row["call_id"]) for row in materialized}) != G2_FORMAL_CALLS:
        raise G2Blocked("g2_repeat_matrix")
    lineage_fields = (
        "run_id",
        "source_sha256",
        "config_sha256",
        "input_sha256",
        "code_sha256",
        "provider_sha256",
        "provider_source_bytes_sha256",
        "oracle_sha256",
        "oracle_source_bytes_sha256",
    )
    if any(
        len({row[field] for row in materialized}) != 1
        for field in lineage_fields
    ):
        raise G2Blocked("g2_row_lineage")
    grouped: dict[
        tuple[str, str, str],
        list[dict[str, object]],
    ] = {}
    for row in materialized:
        key = (
            str(row["platform"]),
            str(row["request_id"]),
            str(row["request_sha256"]),
        )
        grouped.setdefault(key, []).append(row)
    if len(grouped) != 129:
        raise G2Blocked("g2_repeat_matrix")
    for request_rows in grouped.values():
        if (
            sorted(int(row["repeat_index"]) for row in request_rows)
            != list(range(G2_REPEATS))
            or len(
                {
                    (
                        row["scale"],
                        row["request_class"],
                        row["outcome_kind"],
                        row["truth_sha256"],
                        row["oracle_result_sha256"],
                    )
                    for row in request_rows
                }
            )
            != 1
        ):
            raise G2Blocked("g2_repeat_matrix")
    canonical_rows = [
        PlanningCallRow(
            schema_version=str(row["schema_version"]),
            scale_profile=str(row["scale_profile"]),
            run_id=str(row["run_id"]),
            episode_id=str(row["episode_id"]),
            request_id=str(row["request_id"]),
            call_id=str(row["call_id"]),
            platform=str(row["platform"]),
            scale=str(row["scale"]),
            request_class=str(row["request_class"]),
            outcome_kind=str(row["outcome_kind"]),
            source_sha256=str(row["source_sha256"]),
            config_sha256=str(row["config_sha256"]),
            request_sha256=str(row["request_sha256"]),
            provider_sha256=str(row["provider_sha256"]),
            oracle_sha256=str(row["oracle_sha256"]),
            elapsed_ms=float(row["elapsed_ms"]),
            input_validation_ms=float(row["input_validation_ms"]),
            platform_instantiation_ms=float(
                row["platform_instantiation_ms"]
            ),
            search_ms=float(row["search_ms"]),
            complete_route_validation_ms=float(
                row["complete_route_validation_ms"]
            ),
            result_assembly_ms=float(row["result_assembly_ms"]),
            provider_success=bool(row["provider_success"]),
            route_l2_valid=bool(row["route_l2_valid"]),
            semantic_digest=str(row["semantic_digest"]),
        )
        for row in materialized
    ]
    canonical = evaluate_formal_g2(canonical_rows)
    if canonical.get("status") == "blocked":
        raise G2Blocked(
            str(canonical.get("blocking_reason", "g2_canonical_evaluator"))
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


def compare_worker_semantics(
    formal_rows: Sequence[Mapping[str, object]],
    diagnostic_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(formal_rows) != G2_FORMAL_CALLS or len(diagnostic_rows) != 129:
        raise G2Blocked("g2_worker_semantic_drift")
    formal: dict[str, list[Mapping[str, object]]] = {}
    for row in formal_rows:
        formal.setdefault(str(row.get("request_id")), []).append(row)
    diagnostics: dict[str, Mapping[str, object]] = {}
    for row in diagnostic_rows:
        request_id = row.get("request_id")
        if (
            type(request_id) is not str
            or request_id in diagnostics
            or row.get("formal_sample") is not False
        ):
            raise G2Blocked("g2_worker_semantic_drift")
        diagnostics[request_id] = row
    if set(formal) != set(diagnostics):
        raise G2Blocked("g2_worker_semantic_drift")
    for request_id, rows_for_request in formal.items():
        diagnostic = diagnostics[request_id]
        reference = {
            (
                row.get("request_sha256"),
                row.get("semantic_digest"),
                row.get("provider_success"),
                row.get("route_l2_valid"),
            )
            for row in rows_for_request
        }
        observed = {
            (
                diagnostic.get("request_sha256"),
                diagnostic.get("semantic_digest"),
                diagnostic.get("provider_success"),
                diagnostic.get("route_l2_valid"),
            )
        }
        if len(rows_for_request) != G2_REPEATS or reference != observed:
            raise G2Blocked("g2_worker_semantic_drift")
    return {
        "schema_version": "xunce-mid-dual-g2-worker-semantic-audit/v1",
        "matched": True,
        "request_count": 129,
        "formal_repeat_count": G2_REPEATS,
    }


def compare_diagnostic_worker_semantics(
    worker_one_rows: Sequence[Mapping[str, object]],
    worker_four_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Require exact 129-request semantic equality before formal timing."""

    def indexed(
        rows: Sequence[Mapping[str, object]],
    ) -> dict[str, tuple[object, ...]]:
        if len(rows) != 129:
            raise G2Blocked("g2_preformal_worker_semantic_drift")
        result: dict[str, tuple[object, ...]] = {}
        for row in rows:
            request_id = row.get("request_id")
            if (
                type(row) is not dict
                or type(request_id) is not str
                or not request_id
                or request_id in result
                or row.get("formal_sample") is not False
            ):
                raise G2Blocked(
                    "g2_preformal_worker_semantic_drift"
                )
            request_sha256 = row.get("request_sha256")
            semantic_digest = row.get("semantic_digest")
            if (
                not _is_sha256(request_sha256)
                or not _is_sha256(semantic_digest)
                or type(row.get("provider_success")) is not bool
                or type(row.get("route_l2_valid")) is not bool
            ):
                raise G2Blocked(
                    "g2_preformal_worker_semantic_drift"
                )
            result[request_id] = (
                request_sha256,
                semantic_digest,
                row["provider_success"],
                row["route_l2_valid"],
            )
        return result

    worker_one = indexed(worker_one_rows)
    worker_four = indexed(worker_four_rows)
    if worker_one != worker_four:
        raise G2Blocked("g2_preformal_worker_semantic_drift")
    return {
        "schema_version": (
            "xunce-mid-dual-g2-preformal-worker-equivalence/v1"
        ),
        "matched": True,
        "request_count": 129,
        "worker_one_count": 1,
        "worker_four_count": 4,
        "worker_one_results_sha256": _canonical_sha256(
            worker_one_rows
        ),
        "worker_four_results_sha256": _canonical_sha256(
            worker_four_rows
        ),
    }


def validate_static_cache_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    required = {
        "cache_contract",
        "terrain_sha256",
        "static_validation_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise G2Blocked("g2_route_answer_cache")
    if (
        payload["cache_contract"] != STATIC_CACHE_CONTRACT_ID
        or not _is_sha256(payload["terrain_sha256"])
        or not _is_sha256(payload["static_validation_sha256"])
    ):
        raise G2Blocked("g2_static_cache_contract")
    return dict(payload)


def execute_read_only_static_cache_audit(
    terrain_payloads: Mapping[str, bytes],
) -> dict[str, object]:
    """Execute canonical terrain validation without caching route answers."""
    import xunce_mid_dual_g2_inputs as inputs

    if type(terrain_payloads) is not dict or not terrain_payloads:
        raise G2Blocked("g2_static_cache_payload_missing")
    entries: list[dict[str, object]] = []
    for terrain_sha256, payload in sorted(terrain_payloads.items()):
        if (
            not _is_sha256(terrain_sha256)
            or type(payload) is not bytes
            or _sha256(payload) != terrain_sha256
        ):
            raise G2Blocked("g2_static_cache_payload_hash")
        before = _sha256(payload)
        try:
            arrays, metadata, geometry_sha256 = (
                inputs._decode_truth_terrain(payload)  # noqa: SLF001
            )
        except inputs.G2InputContractError as exc:
            raise G2Blocked(
                f"g2_static_cache_payload_invalid:{exc.code}"
            ) from exc
        static_projection = {
            "terrain_sha256": terrain_sha256,
            "geometry_sha256": geometry_sha256,
            "metadata_sha256": _canonical_sha256(metadata),
            "arrays": {
                name: {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "sha256": _sha256(array.tobytes(order="C")),
                }
                for name, array in sorted(arrays.items())
            },
        }
        cache_payload = validate_static_cache_payload(
            {
                "cache_contract": STATIC_CACHE_CONTRACT_ID,
                "terrain_sha256": terrain_sha256,
                "static_validation_sha256": _domain_hash(
                    "xunce-mid-dual-g2-static-validation/v1",
                    _canonical_bytes(static_projection),
                ),
            }
        )
        after = _sha256(payload)
        if after != before:
            raise G2Blocked("g2_static_cache_payload_mutated")
        entries.append(
            {
                **cache_payload,
                "payload_sha256_before": before,
                "payload_sha256_after": after,
            }
        )
    return {
        "schema_version": "xunce-mid-dual-g2-static-cache-audit/v1",
        "status": "passed",
        "formal_evidence_eligible": True,
        "cache_contract": STATIC_CACHE_CONTRACT_ID,
        "read_only": True,
        "terrain_count": len(entries),
        "entries": entries,
        "entries_sha256": _canonical_sha256(entries),
    }


def validate_preformal_diagnostic_evidence(
    audit: Mapping[str, object],
    terrain_payloads: Mapping[str, bytes],
) -> dict[str, object]:
    """Recompute all four preformal artifacts before every formal attempt."""
    required = {
        "worker_one_results",
        "worker_one_results_sha256",
        "worker_four_results",
        "worker_four_results_sha256",
        "worker_equivalence",
        "worker_equivalence_sha256",
        "static_cache_audit",
        "static_cache_audit_sha256",
    }
    if (
        not isinstance(audit, Mapping)
        or audit.get("status") != "complete"
        or audit.get("formal_sample") is not False
        or audit.get("worker_one_count") != 1
        or audit.get("worker_four_count") != 4
        or audit.get("request_count") != 129
        or not required.issubset(audit)
    ):
        raise G2Blocked("g2_preformal_diagnostic_evidence")
    worker_one = audit["worker_one_results"]
    worker_four = audit["worker_four_results"]
    equivalence = audit["worker_equivalence"]
    cache_audit = audit["static_cache_audit"]
    if (
        not isinstance(worker_one, list)
        or not isinstance(worker_four, list)
        or not isinstance(equivalence, Mapping)
        or not isinstance(cache_audit, Mapping)
        or audit["worker_one_results_sha256"]
        != _canonical_sha256(worker_one)
        or audit["worker_four_results_sha256"]
        != _canonical_sha256(worker_four)
        or audit["worker_equivalence_sha256"]
        != _canonical_sha256(equivalence)
        or audit["static_cache_audit_sha256"]
        != _canonical_sha256(cache_audit)
    ):
        raise G2Blocked("g2_preformal_diagnostic_evidence")
    fresh_equivalence = compare_diagnostic_worker_semantics(
        worker_one,
        worker_four,
    )
    fresh_cache_audit = execute_read_only_static_cache_audit(
        terrain_payloads
    )
    if (
        dict(equivalence) != fresh_equivalence
        or dict(cache_audit) != fresh_cache_audit
    ):
        raise G2Blocked("g2_preformal_diagnostic_evidence")
    return dict(audit)


def _journal_row(
    result: Mapping[str, object],
    schedule_sha256: str,
) -> dict[str, object]:
    call_id = result.get("call_id")
    if type(call_id) is not str or not call_id:
        raise G2Blocked("g2_job_state_result")
    materialized = dict(result)
    return {
        "schema_version": "xunce-mid-dual-g2-job-state/v1",
        "schedule_sha256": schedule_sha256,
        "call_id": call_id,
        "result_sha256": _domain_hash(
            "xunce-mid-dual-g2-job-result/v1",
            _canonical_bytes(materialized),
        ),
        "result": materialized,
    }


def _load_job_state(
    state_path: str | Path,
    *,
    schedule_sha256: str,
    expected_call_ids: set[str],
) -> dict[str, dict[str, object]]:
    if not artifact_io.path_is_file(state_path):
        return {}
    completed: dict[str, dict[str, object]] = {}
    for raw in artifact_io.read_jsonl(state_path):
        if (
            set(raw)
            != {
                "schema_version",
                "schedule_sha256",
                "call_id",
                "result_sha256",
                "result",
            }
            or raw["schema_version"] != "xunce-mid-dual-g2-job-state/v1"
            or raw["schedule_sha256"] != schedule_sha256
            or raw["call_id"] not in expected_call_ids
            or raw["call_id"] in completed
            or type(raw["result"]) is not dict
            or raw["result"].get("call_id") != raw["call_id"]
            or raw["result_sha256"]
            != _domain_hash(
                "xunce-mid-dual-g2-job-result/v1",
                _canonical_bytes(raw["result"]),
            )
        ):
            raise G2Blocked("g2_job_state_drift")
        completed[str(raw["call_id"])] = dict(raw)
    return completed


def execute_recoverable_batch(
    calls: Sequence[Mapping[str, object]],
    *,
    execute_task: Callable[[Mapping[str, object]], Mapping[str, object]],
    state_path: str | Path,
    schedule_sha256: str,
    max_workers: int,
) -> list[dict[str, object]]:
    _require_sha256(schedule_sha256, "g2_schedule_hash")
    if type(max_workers) is not int or max_workers <= 0:
        raise G2Blocked("g2_worker_count")
    call_ids = [call.get("call_id") for call in calls]
    if (
        any(type(call_id) is not str or not call_id for call_id in call_ids)
        or len(set(call_ids)) != len(call_ids)
    ):
        raise G2Blocked("g2_schedule_call_id")
    expected = {str(call_id) for call_id in call_ids}
    completed = _load_job_state(
        state_path,
        schedule_sha256=schedule_sha256,
        expected_call_ids=expected,
    )
    ordered_pending = [
        call for call in calls if str(call["call_id"]) not in completed
    ]
    journal = [
        completed[str(call_id)]
        for call_id in call_ids
        if str(call_id) in completed
    ]

    executor = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="g2-recoverable",
    )
    pending: dict[Future[Mapping[str, object]], Mapping[str, object]] = {}
    next_index = 0

    def submit_until_full() -> None:
        nonlocal next_index
        while (
            next_index < len(ordered_pending)
            and len(pending) < max_workers
        ):
            task = ordered_pending[next_index]
            next_index += 1
            pending[executor.submit(execute_task, task)] = task

    def persist(result: Mapping[str, object]) -> None:
        entry = _journal_row(result, schedule_sha256)
        call_id = str(entry["call_id"])
        if call_id in completed:
            raise G2Blocked("g2_job_state_duplicate")
        completed[call_id] = entry
        journal.append(entry)
        artifact_io.write_jsonl(state_path, journal)

    submit_until_full()
    interrupted: BaseException | None = None
    try:
        while pending:
            done, _not_done = wait(
                tuple(pending),
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                task = pending.pop(future)
                try:
                    result = future.result()
                except (KeyboardInterrupt, SystemExit, MemoryError) as exc:
                    interrupted = exc
                    continue
                if type(result) is not dict:
                    raise G2Blocked("g2_job_state_result")
                if result.get("call_id") != task.get("call_id"):
                    raise G2Blocked("g2_job_state_result")
                persist(result)
            if interrupted is not None:
                break
            submit_until_full()
        if interrupted is not None:
            for future in pending:
                future.cancel()
            for future, task in tuple(pending.items()):
                if future.cancelled():
                    continue
                try:
                    result = future.result()
                except (KeyboardInterrupt, SystemExit, MemoryError):
                    continue
                if type(result) is dict and result.get("call_id") == task.get(
                    "call_id"
                ):
                    persist(result)
            raise G2Interrupted() from interrupted
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return [
        dict(completed[str(call_id)]["result"])
        for call_id in call_ids
    ]


def validate_approval_snapshot(
    binding: Mapping[str, object],
    approval_bytes: bytes,
) -> dict[str, object]:
    if type(binding) is not dict or type(approval_bytes) is not bytes:
        raise G2Blocked("g2_approval_drift")
    expected = binding.get("approval_artifact_sha256")
    if (
        not _is_sha256(expected)
        or _sha256(approval_bytes) != expected
        or binding.get("formal_evidence_eligible") is not True
    ):
        raise G2Blocked("g2_approval_drift")
    return dict(binding)


def _route_projection(route: object) -> object:
    primitives = getattr(route, "primitives", None)
    if not isinstance(primitives, tuple):
        return None
    projected: list[dict[str, object]] = []
    for primitive in primitives:
        start = getattr(primitive, "start_state", None)
        end = getattr(primitive, "end_state", None)
        projected.append(
            {
                "kind": getattr(getattr(primitive, "kind", None), "value", None),
                "start": [
                    getattr(start, "x_m", None),
                    getattr(start, "y_m", None),
                    getattr(start, "heading_rad", None),
                ],
                "end": [
                    getattr(end, "x_m", None),
                    getattr(end, "y_m", None),
                    getattr(end, "heading_rad", None),
                ],
                "duration_s": getattr(primitive, "duration_s", None),
                "distance_m": getattr(primitive, "distance_m", None),
                "energy_cost": getattr(primitive, "energy_cost", None),
                "observation_contribution": getattr(
                    primitive,
                    "observation_contribution",
                    None,
                ),
            }
        )
    return projected


def _normalize_provider_outcome(outcome: object) -> dict[str, object]:
    from path_planner.v2.contracts import PlanningFailureV2, PlanningSuccessV2

    if type(outcome) is PlanningSuccessV2:
        return {
            "outcome_type": "success",
            "request_id": outcome.request_id,
            "platform_kind": outcome.platform_kind.value,
            "route": _route_projection(outcome.route),
            "validation": {
                "validator_id": outcome.validation_evidence.validator_id,
                "level": outcome.validation_evidence.level.value,
                "passed": outcome.validation_evidence.passed,
                "checks": list(outcome.validation_evidence.checks),
            },
        }
    if type(outcome) is PlanningFailureV2:
        return {
            "outcome_type": "failure",
            "request_id": outcome.request_id,
            "platform_kind": (
                None
                if outcome.platform_kind is None
                else outcome.platform_kind.value
            ),
            "category": outcome.category.value,
            "reason_code": outcome.reason_code,
            "stage": outcome.evidence.stage,
            "checks": list(outcome.evidence.checks),
        }
    if isinstance(outcome, Mapping):
        category = outcome.get("category")
        return {
            "outcome_type": "internal_failure",
            "category": getattr(category, "value", category),
            "reason_code": outcome.get("reason_code"),
            "exception_stage": outcome.get("exception_stage"),
            "exception_type": outcome.get("exception_type"),
        }
    return {
        "outcome_type": "invalid",
        "python_type": type(outcome).__name__,
    }


def execute_provider_timed_call(
    task: Mapping[str, object],
) -> dict[str, object]:
    """Execute one already-loaded request with the actual approved provider."""

    from path_planner.v2.contracts import (
        PlanningFailureV2,
        PlanningSuccessV2,
        ValidationEvidenceV2,
    )
    from path_planner.v2.timing import execute_timed_request_v2
    import xunce_mid_dual_g2_inputs as inputs

    provider_request = task.get("_provider_request")
    terrain_payload = task.get("_terrain_payload")
    if type(provider_request) is not dict or type(terrain_payload) is not bytes:
        raise G2Blocked("g2_preloaded_request_invalid")
    platform = task.get("platform")

    def decode(encoded: object) -> object:
        if type(encoded) is not dict:
            raise G2Blocked("g2_provider_request_invalid")
        return inputs.decode_provider_execution_request(
            encoded,
            terrain_payload,
        )

    def build(request: object) -> object:
        return inputs.build_approved_platform_execution_stack(
            str(platform),
            request,
        )

    def plan(_request: object, stack: object) -> object:
        if type(stack) is not dict or not callable(stack.get("plan")):
            raise G2Blocked("g2_provider_stack_invalid")
        return stack["plan"]()

    def revalidate(
        request: object,
        outcome: object,
        stack: object,
    ) -> object:
        if type(outcome) is PlanningFailureV2:
            return None
        if type(outcome) is not PlanningSuccessV2 or type(stack) is not dict:
            return None
        if platform == "wheel":
            evidence = outcome.validation_evidence
            return ValidationEvidenceV2(
                validator_id=evidence.validator_id,
                level=evidence.level,
                passed=evidence.passed,
                checks=evidence.checks,
            )
        validator = stack.get("l2_validator")
        if not callable(validator):
            return None
        authority = (
            stack["execution_profile"]
            if platform == "legged"
            else stack["provider"].hopper_authority
        )
        result = validator(
            outcome.route,
            request,
            stack["anchor"],
            authority,
            stack["deadline"],
        )
        return getattr(result, "evidence", None)

    timed = execute_timed_request_v2(
        provider_request,
        decode_request=decode,
        build_platform_stack=build,
        plan_request=plan,
        revalidate_success_route=revalidate,
        assemble_result=lambda outcome, _validation, _timing: (
            _normalize_provider_outcome(outcome)
        ),
    )
    if timed.timing.timing_measurement_valid is not True:
        raise G2Blocked("g2_timing_measurement_invalid")
    timing = validate_timing_payload(timed.timing.as_dict())
    provider_success = type(timed.outcome) is PlanningSuccessV2
    route_l2_valid = (
        provider_success
        and timed.outcome.validation_evidence.passed is True
        and timed.outcome.validation_evidence.level.value == "L2"
    )
    request_sha256 = str(task["request_sha256"])
    semantic = expected_semantic_digest(
        request_sha256=request_sha256,
        provider_success=provider_success,
        route_l2_valid=route_l2_valid,
    )
    normalized_outcome = _normalize_provider_outcome(timed.outcome)
    provider_result_sha256 = _domain_hash(
        "xunce-mid-dual-g2-provider-result/v1",
        _canonical_bytes(normalized_outcome),
    )
    formal = task.get("formal_sample") is True
    if not formal:
        return {
            "call_id": task["call_id"],
            "request_id": task["request_id"],
            "request_sha256": request_sha256,
            "platform": platform,
            "provider_success": provider_success,
            "route_l2_valid": route_l2_valid,
            "semantic_digest": semantic,
            "provider_result_sha256": provider_result_sha256,
            "timing": timing,
            "formal_sample": False,
        }
    row = {
        "row_kind": "g2_planning_call",
        "schema_version": G2_ROW_SCHEMA_VERSION,
        "scale_profile": SCALE_PROFILE,
        "run_id": task["_run_id"],
        "episode_id": task["request_id"],
        "request_id": task["request_id"],
        "call_id": task["call_id"],
        "platform": platform,
        "scale": task["scale"],
        "request_class": task["request_class"],
        "outcome_kind": task["outcome_kind"],
        "source_sha256": task["_code_sha256"],
        "config_sha256": task["_config_sha256"],
        "input_sha256": task["_input_sha256"],
        "code_sha256": task["_code_sha256"],
        "request_sha256": request_sha256,
        "truth_sha256": task["truth_sha256"],
        "provider_sha256": task["_provider_sha256"],
        "provider_source_bytes_sha256": task[
            "_provider_source_bytes_sha256"
        ],
        "oracle_sha256": task["_oracle_sha256"],
        "oracle_source_bytes_sha256": task[
            "_oracle_source_bytes_sha256"
        ],
        "provider_result_sha256": provider_result_sha256,
        "oracle_result_sha256": task["truth_certificate_sha256"],
        **timing,
        **_timing_ms_projection(timing),
        "provider_success": provider_success,
        "route_l2_valid": route_l2_valid,
        "semantic_digest": semantic,
        "timing_contract_id": TIMING_CONTRACT_ID,
        "formal_sample": True,
        "repeat_index": task["repeat_index"],
    }
    if set(row) != G2_ROW_KEYS:
        raise AssertionError("internal G2 formal row schema drifted")
    return row


def _read_execution_bundle(
    root: str | Path,
    *,
    runtime_source_closure: Mapping[str, object],
) -> dict[str, object]:
    import xunce_mid_dual_g2_inputs as inputs

    bundle_root = Path(root)
    manifest_path = bundle_root / "manifest.json"
    if not artifact_io.path_is_file(manifest_path):
        raise G2Blocked("missing_independent_g2_input_bundle")
    try:
        manifest = artifact_io.read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise G2Blocked("g2_execution_manifest_invalid") from exc
    if (
        manifest.get("schema_version")
        != "xunce-mid-dual-g2-execution-bundle/v1"
        or manifest.get("scale_profile") != SCALE_PROFILE
        or manifest.get("formal_evidence_eligible") is not True
        or manifest.get("blockers") != []
        or manifest.get("request_count") != 129
        or manifest.get("crosswalk_count") != 129
        or manifest.get("publication_order") != "data-first-manifest-last"
    ):
        raise G2Blocked("g2_execution_manifest_invalid")
    payload_index = manifest.get("payload_index")
    if not isinstance(payload_index, list) or not payload_index:
        raise G2Blocked("g2_execution_payload_index")
    recorded_paths: list[str] = []
    for raw in payload_index:
        if (
            type(raw) is not dict
            or set(raw) != {"relative_path", "byte_length", "sha256"}
        ):
            raise G2Blocked("g2_execution_payload_index")
        relative = _safe_relative_path(raw["relative_path"])
        payload_path = bundle_root / PurePosixPath(relative)
        if (
            not artifact_io.path_is_file(payload_path)
            or _exact_nonnegative_int(
                raw["byte_length"],
                "g2_execution_payload_index",
            )
            != artifact_io.file_size(payload_path)
            or _require_sha256(
                raw["sha256"],
                "g2_execution_payload_index",
            )
            != _sha256(artifact_read_bytes(payload_path))
        ):
            raise G2Blocked("g2_execution_payload_hash_drift")
        recorded_paths.append(relative)
    actual_paths = [
        path
        for path in artifact_io.list_relative_files(bundle_root)
        if path != "manifest.json"
    ]
    if (
        recorded_paths != sorted(recorded_paths)
        or len(recorded_paths) != len(set(recorded_paths))
        or actual_paths != recorded_paths
        or manifest.get("payload_root_sha256")
        != _domain_hash(
            "xunce-mid-dual-g2-execution-payload-root/v1",
            _canonical_bytes(payload_index),
        )
    ):
        raise G2Blocked("g2_execution_payload_index")
    try:
        provider_requests = artifact_io.read_jsonl(
            bundle_root / "provider-requests.jsonl"
        )
        crosswalk = artifact_io.read_jsonl(
            bundle_root / "truth-provider-crosswalk.jsonl"
        )
        input_audit = artifact_io.read_json(
            bundle_root / "input-audit.json"
        )
        cohort = artifact_io.read_json(
            bundle_root / "g3-replay-cohort.json"
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise G2Blocked("g2_execution_payload_parse") from exc
    if (
        len(provider_requests) != 129
        or len(crosswalk) != 129
        or input_audit.get("formal_evidence_eligible") is not True
        or input_audit.get("status") != "ready"
        or input_audit.get("blockers") != []
        or cohort.get("cohort_sha256")
        != manifest.get("g3_replay_cohort_sha256")
        or input_audit.get("g3_replay_cohort_sha256")
        != manifest.get("g3_replay_cohort_sha256")
    ):
        raise G2Blocked("g2_execution_payload_binding")
    providers_by_hash: dict[str, dict[str, object]] = {}
    for row in provider_requests:
        request_hash = _require_sha256(
            row.get("provider_request_sha256"),
            "g2_provider_request_hash",
        )
        if request_hash in providers_by_hash:
            raise G2Blocked("g2_provider_request_duplicate")
        providers_by_hash[request_hash] = row
    requests: list[dict[str, object]] = []
    for index_by_platform in G2_PLATFORMS:
        selected = [
            row
            for row in crosswalk
            if row.get("platform") == index_by_platform
        ]
        for request_index, row in enumerate(selected):
            provider_hash = _require_sha256(
                row.get("provider_request_sha256"),
                "g2_crosswalk_hash",
            )
            provider = providers_by_hash.get(provider_hash)
            if (
                provider is None
                or provider.get("request_id") != row.get("request_id")
                or provider.get("platform") != row.get("platform")
                or provider.get("scale") != row.get("scale")
                or provider.get("terrain_binding", {}).get("terrain_sha256")
                != row.get("terrain_sha256")
            ):
                raise G2Blocked("g2_truth_provider_crosswalk")
            requests.append({**row, "request_index": request_index})
    normalized = _validated_request_matrix(requests)
    approval = input_audit.get("approval")
    if type(approval) is not dict:
        raise G2Blocked("g2_approval_missing")
    approval_path = approval.get("approval_artifact_path")
    if type(approval_path) is not str or not artifact_io.path_is_file(
        approval_path
    ):
        raise G2Blocked("g2_approval_missing")
    approval_bytes = artifact_read_bytes(approval_path)
    validate_approval_snapshot(approval, approval_bytes)
    truth_bundle_root = input_audit.get("truth_bundle_root")
    if type(truth_bundle_root) is not str or not truth_bundle_root:
        raise G2Blocked("g2_truth_reaudit_root")
    try:
        fresh_audit = inputs.audit_truth_bundle(
            truth_bundle_root,
            approval_path=approval_path,
            mode="reduced",
        )
    except inputs.G2InputContractError as exc:
        raise G2Blocked(f"g2_truth_reaudit:{exc.code}") from exc
    stored_audit_without_provider_rows = {
        key: value
        for key, value in fresh_audit.items()
        if key != "provider_requests"
    }
    if (
        fresh_audit.get("formal_evidence_eligible") is not True
        or fresh_audit.get("status") != "ready"
        or fresh_audit.get("blockers") != []
        or fresh_audit.get("provider_requests") != provider_requests
        or stored_audit_without_provider_rows != input_audit
        or fresh_audit.get("path_planner_runtime_source_closure")
        != runtime_source_closure
    ):
        raise G2Blocked("g2_truth_reaudit_drift")
    terrain_payloads: dict[str, bytes] = {}
    for request in normalized:
        terrain_sha256 = str(request["terrain_sha256"])
        if terrain_sha256 not in terrain_payloads:
            terrain_path = (
                bundle_root / "terrain" / f"{terrain_sha256}.npz"
            )
            if not artifact_io.path_is_file(terrain_path):
                raise G2Blocked("g2_terrain_missing")
            terrain_payload = artifact_read_bytes(terrain_path)
            if _sha256(terrain_payload) != terrain_sha256:
                raise G2Blocked("g2_terrain_hash_drift")
            terrain_payloads[terrain_sha256] = terrain_payload
    return {
        "root": bundle_root,
        "manifest": manifest,
        "manifest_sha256": _sha256(artifact_read_bytes(manifest_path)),
        "input_audit": input_audit,
        "input_sha256": _json_artifact_sha256(input_audit),
        "requests": normalized,
        "providers_by_hash": providers_by_hash,
        "terrain_payloads": terrain_payloads,
        "approval_bytes": approval_bytes,
    }


def _hydrate_calls(
    calls: Sequence[Mapping[str, object]],
    *,
    bundle: Mapping[str, object],
    run_id: str,
    config_sha256: str,
    code_sha256: str,
) -> list[dict[str, object]]:
    audit = bundle["input_audit"]
    if type(audit) is not dict:
        raise G2Blocked("g2_input_audit_invalid")
    provider_source = audit.get("provider_source")
    oracle_source = audit.get("oracle_source")
    if type(provider_source) is not dict or type(oracle_source) is not dict:
        raise G2Blocked("g2_source_identity_invalid")
    for source in (provider_source, oracle_source):
        _require_sha256(
            source.get("implementation_sha256"),
            "g2_source_identity_invalid",
        )
        _require_sha256(
            source.get("source_bytes_sha256"),
            "g2_source_identity_invalid",
        )
    if (
        provider_source.get("identity") == oracle_source.get("identity")
        or provider_source.get("implementation_sha256")
        == oracle_source.get("implementation_sha256")
        or provider_source.get("source_bytes_sha256")
        == oracle_source.get("source_bytes_sha256")
    ):
        raise G2Blocked("g2_provider_oracle_source_not_separated")
    providers = bundle["providers_by_hash"]
    terrains = bundle["terrain_payloads"]
    if type(providers) is not dict or type(terrains) is not dict:
        raise G2Blocked("g2_preload_invalid")
    hydrated: list[dict[str, object]] = []
    for call in calls:
        provider_request = providers.get(call["request_sha256"])
        terrain_payload = terrains.get(call["terrain_sha256"])
        if type(provider_request) is not dict or type(terrain_payload) is not bytes:
            raise G2Blocked("g2_preload_invalid")
        hydrated.append(
            {
                **dict(call),
                "_provider_request": provider_request,
                "_terrain_payload": terrain_payload,
                "_run_id": run_id,
                "_config_sha256": config_sha256,
                "_input_sha256": bundle["input_sha256"],
                "_code_sha256": code_sha256,
                "_provider_sha256": provider_source[
                    "implementation_sha256"
                ],
                "_provider_source_bytes_sha256": provider_source[
                    "source_bytes_sha256"
                ],
                "_oracle_sha256": oracle_source[
                    "implementation_sha256"
                ],
                "_oracle_source_bytes_sha256": oracle_source[
                    "source_bytes_sha256"
                ],
            }
        )
    return hydrated


def _git_output(*args: str, cwd: Path = REPO_ROOT) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def validate_platform_invariants(
    active_platforms: object,
    platform_invariants: object,
) -> dict[str, object]:
    if (
        active_platforms != list(ACTIVE_PLATFORMS)
        or platform_invariants != PLATFORM_INVARIANTS
    ):
        raise G2Blocked("g2_platform_invariant_drift")
    return {
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": PLATFORM_INVARIANTS,
    }


def _runtime_binding(
    runtime_source_closure_sha256: str,
) -> dict[str, object]:
    return {
        **validate_platform_invariants(
            list(ACTIVE_PLATFORMS),
            PLATFORM_INVARIANTS,
        ),
        "path_planner_runtime_source_closure_sha256": _require_sha256(
            runtime_source_closure_sha256,
            "g2_runtime_source_closure_sha256",
        ),
    }


def _code_lineage_sha256(
    runtime_source_closure: Mapping[str, object],
) -> str:
    import xunce_mid_dual_g2_inputs as inputs

    closure = inputs.validate_path_planner_runtime_source_closure(
        runtime_source_closure,
        runtime_source_closure,
    )
    rows: list[dict[str, object]] = []
    for relative in G2_REQUIRED_SOURCE_RELATIVE_PATHS:
        path = REPO_ROOT / relative
        if not artifact_io.path_is_file(path):
            raise G2Blocked("g2_required_source_missing")
        payload = artifact_read_bytes(path)
        rows.append(
            {
                "logical_path": relative,
                "size_bytes": len(payload),
                "sha256": _sha256(payload),
            }
        )
    local_source_sha256 = _sha256(
        b"xunce-mid-dual-code/v1\0" + _canonical_bytes(rows)
    )
    return _domain_hash(
        "xunce-mid-dual-g2-code-with-runtime-source/v1",
        _canonical_bytes(
            {
                "local_source_sha256": local_source_sha256,
                "path_planner_runtime_source_closure_sha256": closure[
                    "path_planner_runtime_source_closure_sha256"
                ],
            }
        ),
    )


def _environment_probe() -> dict[str, object]:
    class MemoryStatus(ctypes.Structure):
        _fields_ = (
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        )

    memory = MemoryStatus()
    memory.length = ctypes.sizeof(MemoryStatus)
    memory_bytes = 1
    if os.name == "nt":
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
        if not ok:
            raise G2Blocked("g2_memory_probe_failed")
        memory_bytes = int(memory.total_physical)
    thread_names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "windows_version": host_platform.platform(),
        "cpu_model": host_platform.processor() or "unknown-cpu",
        "cpu_logical_count": os.cpu_count() or 1,
        "memory_bytes": memory_bytes,
        "gpu": {
            "model": "not-used-cpu-planner",
            "driver": "not-applicable",
            "cuda": "not-used",
        },
        "python_executable": sys.executable,
        "python_version": host_platform.python_version(),
        "frozen_dependencies": [
            f"python=={host_platform.python_version()}",
        ],
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "not-set"),
        "thread_variables": {
            name: os.environ.get(name, "not-set") for name in thread_names
        },
        "worker_start_method": "thread-pool-fixed-four/v1",
        "power_mode": "formal-live-gate-separate/v1",
    }


def _load_config(path: str | Path) -> tuple[dict[str, object], str]:
    supplied = Path(path).resolve()
    if supplied != CANONICAL_CONFIG_PATH.resolve():
        raise G2Blocked("g2_config_path_not_canonical")
    if not artifact_io.path_is_file(supplied):
        raise G2Blocked("g2_config_missing")
    payload_bytes = artifact_read_bytes(supplied)
    try:
        config = json.loads(payload_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G2Blocked("g2_config_invalid") from exc
    platform_stacks = (
        config.get("platform_stacks")
        if type(config) is dict
        else None
    )
    try:
        formal_environment_gate = validate_formal_environment_policy(
            (
                config.get("execution", {}).get(
                    "formal_environment_gate"
                )
                if type(config) is dict
                else None
            )
        )
    except G2Blocked as exc:
        raise G2Blocked("g2_config_invalid") from exc
    if (
        type(config) is not dict
        or config.get("schema_version") != G2_CONFIG_SCHEMA_VERSION
        or config.get("gate_id") != G2_GATE_ID
        or config.get("runner_id")
        != "run_xunce_mid_dual_g2_planning_time/v1"
        or config.get("scale_profile") != SCALE_PROFILE
        or config.get("execution", {}).get("formal_worker_count") != 4
        or config.get("execution", {}).get("diagnostic_worker_count") != 1
        or config.get("execution", {}).get("formal_timing_contract")
        != TIMING_CONTRACT_ID
        or config.get("execution", {}).get("formal_cache_contract")
        != STATIC_CACHE_CONTRACT_ID
        or formal_environment_gate != FORMAL_ENVIRONMENT_POLICY
        or config.get("request_contract", {}).get("formal_call_count")
        != G2_FORMAL_CALLS
        or config.get("request_contract", {}).get("repeat_count")
        != G2_REPEATS
        or config.get("request_contract", {}).get("requests_per_platform")
        != 43
        or config.get("request_contract", {}).get("standard")
        != {"reachable": 23, "hard_reachable": 7, "unreachable": 3}
        or config.get("request_contract", {}).get("kilometer")
        != {"reachable": 6, "hard_reachable": 2, "unreachable": 2}
        or config.get("thresholds_ms")
        != {
            "midterm_mean": 2000.0,
            "midterm_p95": 2000.0,
            "absolute_max": 2000.0,
            "final_mean": 1000.0,
            "final_p95": 1000.0,
            "final_proportion_at_or_below": 0.95,
            "engineering_standard_p95": 250.0,
            "engineering_kilometer_p95": 750.0,
        }
        or type(platform_stacks) is not dict
        or list(platform_stacks) != list(ACTIVE_PLATFORMS)
        or any(
            type(platform_stacks.get(platform)) is not dict
            or platform_stacks[platform].get(
                "max_traversable_slope_deg"
            )
            != 30.0
            for platform in ACTIVE_PLATFORMS
        )
    ):
        raise G2Blocked("g2_config_invalid")
    return config, _sha256(payload_bytes)


def _effective_config(
    *,
    run_id: str,
    input_sha256: str,
    code_sha256: str,
    runtime_source_closure_sha256: str,
    formal_environment_gate: Mapping[str, object],
) -> dict[str, object]:
    _require_sha256(
        runtime_source_closure_sha256,
        "g2_runtime_source_closure_sha256",
    )
    validated_formal_environment_gate = (
        validate_formal_environment_policy(formal_environment_gate)
    )
    return {
        "schema_version": G2_CONFIG_SCHEMA_VERSION,
        "gate_id": G2_GATE_ID,
        "runner_id": G2_RUNNER_ID,
        "run_id": run_id,
        "output_root": G2_OUTPUT_BASE,
        "scale_profile": SCALE_PROFILE,
        "input_sha256": input_sha256,
        "code_sha256": code_sha256,
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": PLATFORM_INVARIANTS,
        "path_planner_runtime_source_closure_sha256": (
            runtime_source_closure_sha256
        ),
        "formal_environment_gate": validated_formal_environment_gate,
        "required_phase_ids": list(G2_REQUIRED_PHASE_IDS),
        "source_contract_sha256": _canonical_sha256(G2_SOURCE_CONTRACT),
        "evidence_binding": {
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
        },
    }


def _open_store(
    run_root: Path,
    effective_config: Mapping[str, object],
) -> tuple[MidDualRunStore, bool]:
    expected_sha256 = _canonical_sha256(effective_config)
    if artifact_io.path_exists(run_root):
        if artifact_io.path_is_file(artifact_path(run_root, MID_DUAL_MANIFEST)):
            MidDualRunStore.verify_manifest(run_root)
            raise G2Blocked("g2_run_already_finalized")
        return (
            MidDualRunStore.load_for_resume(run_root, expected_sha256),
            False,
        )
    return MidDualRunStore.create_new(run_root, effective_config), True


def _accept_phase(
    store: MidDualRunStore,
    *,
    phase_id: str,
    rows: Sequence[Mapping[str, object]],
    audit: Mapping[str, object],
) -> None:
    if phase_id in store.accepted_phase_ids:
        return
    expected = f"p{len(store.accepted_phase_ids) + 1:02d}"
    if phase_id != expected:
        raise G2Blocked("g2_resume_phase_prefix")
    phase_audit = {
        "schema_version": "xunce-mid-dual-g2-phase-audit/v1",
        "gate_id": G2_GATE_ID,
        "runner_id": G2_RUNNER_ID,
        "phase_id": phase_id,
        "phase_name": G2_PHASE_NAMES[phase_id],
        **dict(audit),
    }
    attempt_id = store.write_phase_attempt(phase_id, rows, phase_audit)
    digest = store.phase_attempt_row_sha256(phase_id, attempt_id)
    store.accept_phase(phase_id, attempt_id, digest)


def _accepted_phase_rows(
    store: MidDualRunStore,
    phase_id: str,
) -> list[dict[str, Any]]:
    state_path = artifact_path(store.run_root, MID_DUAL_PHASE_STATE)
    states = artifact_io.read_jsonl(state_path)
    selected = next(
        (row for row in states if row.get("phase_id") == phase_id),
        None,
    )
    if selected is None or type(selected.get("rows_path")) is not str:
        raise G2Blocked("g2_accepted_phase_missing")
    return artifact_io.read_jsonl(store.run_root / selected["rows_path"])


def _accepted_phase_audit(
    store: MidDualRunStore,
    phase_id: str,
) -> dict[str, Any]:
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
        raise G2Blocked("g2_accepted_phase_audit_missing")
    return artifact_io.read_json(store.run_root / selected["audit_path"])


def _root_and_submodule_commits() -> tuple[str, str]:
    root_commit = _git_output("rev-parse", "HEAD") or "unavailable-root-commit"
    submodule = REPO_ROOT / "path-planner"
    submodule_commit = (
        _git_output("rev-parse", "HEAD", cwd=submodule)
        if artifact_io.path_exists(submodule)
        else ""
    )
    return root_commit, submodule_commit or "not-present"


def _capture_preflight(
    store: MidDualRunStore,
    *,
    newly_created: bool,
    status: str,
    reason: str | None,
    input_manifest_sha256: str | None,
    input_sha256: str,
    code_sha256: str,
    runtime_source_closure_sha256: str,
) -> None:
    if "p01" in store.accepted_phase_ids:
        return
    if newly_created:
        root_commit, submodule_commit = _root_and_submodule_commits()
        lineage = store.capture_lineage(
            [REPO_ROOT / path for path in G2_REQUIRED_SOURCE_RELATIVE_PATHS],
            root_commit,
            submodule_commit,
        )
        if status == "passed":
            if lineage.get("formal_evidence_eligible") is not True:
                raise G2Blocked("g2_lineage_capture_blocked")
            environment = store.capture_environment(_environment_probe)
            if environment.get("formal_evidence_eligible") is not True:
                raise G2Blocked("g2_environment_capture_blocked")
    _accept_phase(
        store,
        phase_id="p01",
        rows=(),
        audit={
            "status": status,
            "blocking_reason": reason,
            "input_manifest_sha256": input_manifest_sha256,
            "input_sha256": input_sha256,
            "code_sha256": code_sha256,
            **_runtime_binding(runtime_source_closure_sha256),
            "provider_called": False,
            "formal_row_count": 0,
        },
    )


def _blocked_summary(
    *,
    run_id: str,
    mode: str,
    config_sha256: str,
    input_sha256: str,
    code_sha256: str,
    runtime_source_closure_sha256: str,
    reason: str,
) -> dict[str, object]:
    return {
        "schema_version": G2_SUMMARY_SCHEMA_VERSION,
        "gate_id": G2_GATE_ID,
        "runner_id": G2_RUNNER_ID,
        "scale_profile": SCALE_PROFILE,
        "run_id": run_id,
        "mode": mode,
        "config_sha256": config_sha256,
        "input_sha256": input_sha256,
        "code_sha256": code_sha256,
        **_runtime_binding(runtime_source_closure_sha256),
        "status": "blocked",
        "formal_evidence_eligible": False,
        "formal_call_count": 0,
        "g2_all_platforms_2s_passed": False,
        "g2_all_platforms_1s_passed": False,
        "correctness_passed": False,
        "blockers": [reason],
        "recomputed": None,
    }


def _routing(summary: Mapping[str, object]) -> dict[str, object]:
    status = summary.get("status")
    recomputed = summary.get("recomputed")
    projection = recomputed if isinstance(recomputed, Mapping) else summary
    return {
        "schema_version": "xunce-mid-dual-g2-routing/v1",
        "gate_id": G2_GATE_ID,
        "run_id": summary.get("run_id"),
        "status": status,
        "formal_evidence_eligible": (
            summary.get("formal_evidence_eligible") is True
        ),
        "midterm_reduced_passed": (
            projection.get("g2_all_platforms_2s_passed") is True
        ),
        "final_threshold_reduced_passed": (
            projection.get("g2_all_platforms_1s_passed") is True
        ),
        "next_route": (
            "g3"
            if status == "passed"
            else "repair_g2" if status == "failed" else "resolve_g2_blocker"
        ),
        "blocking_reasons": list(summary.get("blockers", [])),
    }


def _render_report(
    summary: Mapping[str, object],
    *,
    formal_environment_audit: Mapping[str, object] | None = None,
) -> str:
    recomputed = summary.get("recomputed")
    projection = recomputed if isinstance(recomputed, Mapping) else summary
    lines = [
        "# G2 多平台规划时间正式实验报告",
        "",
        f"- 运行标识：`{summary.get('run_id')}`",
        f"- 证据状态：`{summary.get('status')}`",
        f"- 正式调用数：{projection.get('formal_call_count', 0)}",
        f"- 中期 2 s 门槛：{projection.get('g2_all_platforms_2s_passed', False)}",
        f"- 最终 1 s 门槛：{projection.get('g2_all_platforms_1s_passed', False)}",
        f"- 正确性门槛：{projection.get('correctness_passed', False)}",
        "",
        "正式计时采用 `five-phase-sequential-ns/v1`，每条原始记录保留"
        "五段整数纳秒与精确总和；毫秒值仅由原始整数唯一换算。",
    ]
    if formal_environment_audit is not None:
        lines.extend(("", "## 正式环境核验", ""))
        if formal_environment_audit.get("status") == "passed":
            validated_environment = (
                _validate_complete_formal_environment_audit(
                    formal_environment_audit
                )
            )
            start = validated_environment["start_observation"]
            end = validated_environment["end_observation"]
            thread_order = (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
            start_threads = "，".join(
                f"{name}={start['thread_variables'][name]}"
                for name in thread_order
            )
            end_threads = "，".join(
                f"{name}={end['thread_variables'][name]}"
                for name in thread_order
            )
            lines.extend(
                (
                    "- 环境门状态：`passed`",
                    "- 电源方案（起始/结束）："
                    f"`{start['power_scheme_guid']}` / "
                    f"`{end['power_scheme_guid']}`",
                    f"- 线程变量（起始）：{start_threads}",
                    f"- 线程变量（结束）：{end_threads}",
                    f"- 起始 CPU：{float(start['cpu_percent']):.3f}%",
                    "- 起始内存："
                    f"{float(start['memory_percent']):.3f}%，"
                    f"可用 {int(start['memory_available_bytes'])} B",
                    "- 起始磁盘忙碌率："
                    f"{float(start['disk_busy_percent']):.3f}%，"
                    f"可用 {int(start['disk_free_bytes'])} B",
                    f"- 结束 CPU：{float(end['cpu_percent']):.3f}%",
                    "- 结束内存："
                    f"{float(end['memory_percent']):.3f}%，"
                    f"可用 {int(end['memory_available_bytes'])} B",
                    "- 结束磁盘忙碌率："
                    f"{float(end['disk_busy_percent']):.3f}%，"
                    f"可用 {int(end['disk_free_bytes'])} B",
                )
            )
        else:
            lines.append(
                "- 环境门状态："
                f"`{formal_environment_audit.get('status', 'blocked')}`"
            )
    blockers = summary.get("blockers")
    if isinstance(blockers, list) and blockers:
        lines.extend(("", "## 阻塞原因", ""))
        lines.extend(f"- `{reason}`" for reason in blockers)
    return "\n".join(lines) + "\n"


def _source_report_audit(
    *,
    summary: Mapping[str, object],
    report: str,
) -> dict[str, object]:
    recomputed = summary.get("recomputed")
    if not isinstance(recomputed, Mapping):
        raise G2Blocked("g2_report_summary_projection")
    stored_summary_bytes = (
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
        "gate_id": G2_GATE_ID,
        "run_id": summary.get("run_id"),
        "renderer_id": "xunce-mid-dual-g2-canonical-report/v1",
        "summary_projection_sha256": _canonical_sha256(recomputed),
        "stored_summary_bytes_sha256": _sha256(stored_summary_bytes),
        "report_bytes_sha256": _sha256(report.encode("utf-8")),
        "formal_evidence_eligible": True,
    }


def _platform_invariants_audit(
    *,
    run_id: str,
    config_sha256: str,
    input_sha256: str,
    code_sha256: str,
    runtime_source_closure_sha256: str,
    status: str,
    formal_evidence_eligible: bool,
) -> dict[str, object]:
    if status not in {"passed", "blocked"}:
        raise G2Blocked("g2_platform_invariant_audit_status")
    if type(formal_evidence_eligible) is not bool:
        raise G2Blocked("g2_platform_invariant_audit_status")
    return {
        "schema_version": (
            "xunce-mid-dual-g2-platform-invariants-audit/v1"
        ),
        "gate_id": G2_GATE_ID,
        "scale_profile": SCALE_PROFILE,
        "run_id": run_id,
        "status": status,
        "formal_evidence_eligible": formal_evidence_eligible,
        **_runtime_binding(runtime_source_closure_sha256),
        "config_sha256": _require_sha256(
            config_sha256,
            "g2_platform_invariant_config_sha256",
        ),
        "input_sha256": _require_sha256(
            input_sha256,
            "g2_platform_invariant_input_sha256",
        ),
        "code_sha256": _require_sha256(
            code_sha256,
            "g2_platform_invariant_code_sha256",
        ),
    }


def _finalize_blocked(
    *,
    store: MidDualRunStore,
    run_id: str,
    mode: str,
    input_audit: Mapping[str, object],
    code_sha256: str,
    runtime_source_closure: Mapping[str, object],
    reason: str,
    formal_environment_audit: Mapping[str, object] | None = None,
    formal_environment_gate_attempted: bool = False,
) -> dict[str, object]:
    runtime_source_closure_sha256 = str(
        runtime_source_closure[
            "path_planner_runtime_source_closure_sha256"
        ]
    )
    summary = _blocked_summary(
        run_id=run_id,
        mode=mode,
        config_sha256=store.config_sha256,
        input_sha256=_json_artifact_sha256(input_audit),
        code_sha256=code_sha256,
        runtime_source_closure_sha256=(
            runtime_source_closure_sha256
        ),
        reason=reason,
    )
    extra_audits: dict[str, Mapping[str, object]] = {
        "g2_input": input_audit,
        "g2_runtime_source_closure": runtime_source_closure,
        "g2_platform_invariants": _platform_invariants_audit(
            run_id=run_id,
            config_sha256=store.config_sha256,
            input_sha256=_json_artifact_sha256(input_audit),
            code_sha256=code_sha256,
            runtime_source_closure_sha256=(
                runtime_source_closure_sha256
            ),
            status="blocked",
            formal_evidence_eligible=False,
        ),
    }
    if formal_environment_audit is not None:
        extra_audits["g2_formal_environment"] = dict(
            formal_environment_audit
        )
    store.finalize(
        summary,
        _routing(summary),
        _render_report(
            summary,
            formal_environment_audit=formal_environment_audit,
        ),
        extra_audits,
    )
    return {
        "execution_status": "complete",
        "gate_status": "blocked",
        "formal_row_count": 0,
        "formal_environment_gate_status": (
            (
                "blocked"
                if formal_environment_gate_attempted
                else "not_run"
            )
            if formal_environment_audit is None
            else str(
                formal_environment_audit.get("status", "blocked")
            )
        ),
        "blocking_reason": reason,
        "run_root": str(store.run_root).replace("\\", "/"),
        "summary": summary,
    }


def _missing_input_audit(
    input_bundle: str | Path,
    reason: str,
) -> dict[str, object]:
    return {
        "schema_version": "xunce-mid-dual-g2-run-input-audit/v1",
        "gate_id": G2_GATE_ID,
        "scale_profile": SCALE_PROFILE,
        "status": "blocked",
        "formal_evidence_eligible": False,
        "input_bundle": str(input_bundle).replace("\\", "/"),
        "formal_row_count": 0,
        "provider_called": False,
        "blockers": [reason],
    }


def _diagnostic_projection(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "request_id": row["request_id"],
            "request_sha256": row["request_sha256"],
            "semantic_digest": row["semantic_digest"],
            "provider_success": row["provider_success"],
            "route_l2_valid": row["route_l2_valid"],
            "formal_sample": False,
        }
        for row in rows
    ]


def run_g2(
    *,
    config_path: str | Path,
    input_bundle: str | Path,
    run_id: str,
    mode: str,
) -> dict[str, object]:
    run_id = _validated_run_id(run_id)
    if mode not in G2_MODES:
        raise G2Blocked("g2_mode_invalid")
    base_config, _base_config_sha256 = _load_config(config_path)
    formal_environment_gate = validate_formal_environment_policy(
        base_config["execution"]["formal_environment_gate"]
    )
    import xunce_mid_dual_g2_inputs as inputs

    runtime_source_closure = (
        inputs.capture_path_planner_runtime_source_closure()
    )
    runtime_source_closure_sha256 = str(
        runtime_source_closure[
            "path_planner_runtime_source_closure_sha256"
        ]
    )
    code_sha256 = _code_lineage_sha256(runtime_source_closure)
    run_root = Path(G2_OUTPUT_BASE) / run_id
    try:
        bundle = _read_execution_bundle(
            input_bundle,
            runtime_source_closure=runtime_source_closure,
        )
    except G2Blocked as exc:
        input_audit = _missing_input_audit(input_bundle, exc.reason)
        effective = _effective_config(
            run_id=run_id,
            input_sha256=_json_artifact_sha256(input_audit),
            code_sha256=code_sha256,
            runtime_source_closure_sha256=(
                runtime_source_closure_sha256
            ),
            formal_environment_gate=formal_environment_gate,
        )
        store, created = _open_store(run_root, effective)
        _capture_preflight(
            store,
            newly_created=created,
            status="blocked",
            reason=exc.reason,
            input_manifest_sha256=None,
            input_sha256=_json_artifact_sha256(input_audit),
            code_sha256=code_sha256,
            runtime_source_closure_sha256=(
                runtime_source_closure_sha256
            ),
        )
        return _finalize_blocked(
            store=store,
            run_id=run_id,
            mode=mode,
            input_audit=input_audit,
            code_sha256=code_sha256,
            runtime_source_closure=runtime_source_closure,
            reason=exc.reason,
        )

    input_audit = dict(bundle["input_audit"])
    effective = _effective_config(
        run_id=run_id,
        input_sha256=str(bundle["input_sha256"]),
        code_sha256=code_sha256,
        runtime_source_closure_sha256=runtime_source_closure_sha256,
        formal_environment_gate=formal_environment_gate,
    )
    store: MidDualRunStore | None = None
    formal_guard: G2FormalEnvironmentGuard | None = None
    formal_guard_entered = False
    try:
        store, created = _open_store(run_root, effective)
        _capture_preflight(
            store,
            newly_created=created,
            status="passed",
            reason=None,
            input_manifest_sha256=str(bundle["manifest_sha256"]),
            input_sha256=str(bundle["input_sha256"]),
            code_sha256=code_sha256,
            runtime_source_closure_sha256=(
                runtime_source_closure_sha256
            ),
        )
        if mode == "preflight":
            return {
                "execution_status": "incomplete",
                "gate_status": "preflight_complete",
                "formal_row_count": 0,
                "formal_environment_gate_status": "not_run",
                "run_root": str(run_root).replace("\\", "/"),
                "accepted_phase_ids": list(store.accepted_phase_ids),
            }

        if mode == "formal":
            formal_lease = G2FormalLease(
                lease_path=str(formal_environment_gate["lease_path"]),
                run_id=run_id,
                run_root=run_root,
            )
            formal_guard = G2FormalEnvironmentGuard(
                run_id=run_id,
                policy=formal_environment_gate,
                lease=formal_lease,
                capture_environment=(
                    capture_windows_formal_environment
                ),
            )
            formal_guard.__enter__()
            formal_guard_entered = True

        nonformal = build_nonformal_schedules(bundle["requests"])
        if "p02" not in store.accepted_phase_ids:
            cold_calls = _hydrate_calls(
                nonformal["cold_start"],
                bundle=bundle,
                run_id=run_id,
                config_sha256=store.config_sha256,
                code_sha256=code_sha256,
            )
            warmup_calls = _hydrate_calls(
                nonformal["warmup"],
                bundle=bundle,
                run_id=run_id,
                config_sha256=store.config_sha256,
                code_sha256=code_sha256,
            )
            cold_rows = execute_preloaded_batch(
                cold_calls,
                prepare_task=lambda task: task,
                execute_task=execute_provider_timed_call,
                max_workers=1,
            )
            warmup_rows = execute_preloaded_batch(
                warmup_calls,
                prepare_task=lambda task: task,
                execute_task=execute_provider_timed_call,
                max_workers=4,
            )
            _accept_phase(
                store,
                phase_id="p02",
                rows=(),
                audit={
                    "status": "complete",
                    "formal_sample": False,
                    **_runtime_binding(
                        runtime_source_closure_sha256
                    ),
                    "cold_start_count": len(cold_rows),
                    "warmup_count": len(warmup_rows),
                    "cold_start_results_sha256": _canonical_sha256(
                        cold_rows
                    ),
                    "warmup_results_sha256": _canonical_sha256(warmup_rows),
                    "cold_start_results": cold_rows,
                    "warmup_results": warmup_rows,
                },
            )
        if "p03" not in store.accepted_phase_ids:
            worker_one_calls = _hydrate_calls(
                nonformal["worker_one"],
                bundle=bundle,
                run_id=run_id,
                config_sha256=store.config_sha256,
                code_sha256=code_sha256,
            )
            worker_one_full = execute_preloaded_batch(
                worker_one_calls,
                prepare_task=lambda task: task,
                execute_task=execute_provider_timed_call,
                max_workers=1,
            )
            worker_four_calls = _hydrate_calls(
                nonformal["worker_four"],
                bundle=bundle,
                run_id=run_id,
                config_sha256=store.config_sha256,
                code_sha256=code_sha256,
            )
            worker_four_full = execute_preloaded_batch(
                worker_four_calls,
                prepare_task=lambda task: task,
                execute_task=execute_provider_timed_call,
                max_workers=4,
            )
            worker_one_rows = _diagnostic_projection(worker_one_full)
            worker_four_rows = _diagnostic_projection(worker_four_full)
            worker_equivalence = compare_diagnostic_worker_semantics(
                worker_one_rows,
                worker_four_rows,
            )
            static_cache_audit = execute_read_only_static_cache_audit(
                bundle["terrain_payloads"]
            )
            _accept_phase(
                store,
                phase_id="p03",
                rows=(),
                audit={
                    "status": "complete",
                    "formal_sample": False,
                    **_runtime_binding(
                        runtime_source_closure_sha256
                    ),
                    "worker_one_count": 1,
                    "worker_four_count": 4,
                    "request_count": len(worker_one_rows),
                    "worker_one_results": worker_one_rows,
                    "worker_one_results_sha256": _canonical_sha256(
                        worker_one_rows
                    ),
                    "worker_four_results": worker_four_rows,
                    "worker_four_results_sha256": _canonical_sha256(
                        worker_four_rows
                    ),
                    "worker_equivalence": worker_equivalence,
                    "worker_equivalence_sha256": _canonical_sha256(
                        worker_equivalence
                    ),
                    "static_cache_audit": static_cache_audit,
                    "static_cache_audit_sha256": _canonical_sha256(
                        static_cache_audit
                    ),
                },
            )
        diagnostic_evidence = validate_preformal_diagnostic_evidence(
            _accepted_phase_audit(store, "p03"),
            bundle["terrain_payloads"],
        )
        recorded_worker_one = diagnostic_evidence[
            "worker_one_results"
        ]
        if not isinstance(recorded_worker_one, list):
            raise G2Blocked("g2_worker_one_results_missing")
        worker_one_rows = [
            dict(row)
            for row in recorded_worker_one
            if isinstance(row, Mapping)
        ]
        if len(worker_one_rows) != len(recorded_worker_one):
            raise G2Blocked("g2_worker_one_results_missing")
        if mode == "diagnostic":
            return {
                "execution_status": "incomplete",
                "gate_status": "diagnostic_complete",
                "formal_row_count": 0,
                "formal_environment_gate_status": "not_run",
                "run_root": str(run_root).replace("\\", "/"),
                "accepted_phase_ids": list(store.accepted_phase_ids),
            }

        schedule = build_formal_schedule(
            str(bundle["manifest"]["input_set_id"]),
            bundle["requests"],
        )
        pending_p04: tuple[
            list[dict[str, object]],
            dict[str, object],
        ] | None = None
        if "p04" not in store.accepted_phase_ids:
            formal_calls = _hydrate_calls(
                schedule["calls"],
                bundle=bundle,
                run_id=run_id,
                config_sha256=store.config_sha256,
                code_sha256=code_sha256,
            )
            formal_rows = execute_recoverable_batch(
                formal_calls,
                execute_task=execute_provider_timed_call,
                state_path=store.run_root / "job-state.jsonl",
                schedule_sha256=str(schedule["schedule_sha256"]),
                max_workers=4,
            )
            recomputed = recompute_g2_summary(formal_rows)
            worker_audit = compare_worker_semantics(
                formal_rows,
                worker_one_rows,
            )
            pending_p04 = (
                list(formal_rows),
                {
                    "status": "complete",
                    "formal_sample": True,
                    **_runtime_binding(
                        runtime_source_closure_sha256
                    ),
                    "formal_worker_count": 4,
                    "formal_call_count": len(formal_rows),
                    "schedule_sha256": schedule["schedule_sha256"],
                    "timing_contract_id": TIMING_CONTRACT_ID,
                    "worker_semantic_audit": worker_audit,
                    "independent_recomputed_sha256": _canonical_sha256(
                        recomputed
                    ),
                },
            )
        else:
            validate_formal_environment_phase_audit(
                _accepted_phase_audit(store, "p04")
            )
            formal_rows = _accepted_phase_rows(store, "p04")
            recomputed = recompute_g2_summary(formal_rows)
            worker_audit = compare_worker_semantics(
                formal_rows,
                worker_one_rows,
            )
        if formal_guard is None or not formal_guard_entered:
            raise G2Blocked("g2_formal_environment_guard_not_active")
        formal_guard.set_formal_row_count(len(formal_rows))
        formal_guard.__exit__(None, None, None)
        formal_guard_entered = False
        formal_environment_audit = dict(formal_guard.audit)
        if pending_p04 is not None:
            pending_rows, pending_audit = pending_p04
            _accept_phase(
                store,
                phase_id="p04",
                rows=pending_rows,
                audit={
                    **pending_audit,
                    **formal_environment_phase_audit_fields(
                        formal_environment_audit
                    ),
                },
            )
        summary = {
            "schema_version": G2_SUMMARY_SCHEMA_VERSION,
            "scale_profile": SCALE_PROFILE,
            "gate_id": G2_GATE_ID,
            "run_id": run_id,
            "status": recomputed["status"],
            "formal_evidence_eligible": True,
            **_runtime_binding(runtime_source_closure_sha256),
            "recomputed": recomputed,
        }
        report = _render_report(
            summary,
            formal_environment_audit=formal_environment_audit,
        )
        report_audit = _source_report_audit(
            summary=summary,
            report=report,
        )
        store.finalize(
            summary,
            _routing(summary),
            report,
            {
                "g2_input": input_audit,
                "g2_runtime_source_closure": runtime_source_closure,
                "g2_formal_environment": formal_environment_audit,
                "g2_platform_invariants": _platform_invariants_audit(
                    run_id=run_id,
                    config_sha256=store.config_sha256,
                    input_sha256=str(bundle["input_sha256"]),
                    code_sha256=code_sha256,
                    runtime_source_closure_sha256=(
                        runtime_source_closure_sha256
                    ),
                    status="passed",
                    formal_evidence_eligible=True,
                ),
                "g2_report": report_audit,
                "g2_worker_semantics": worker_audit,
                "g2_schedule": {
                    "schema_version": schedule["schema_version"],
                    "input_set_id": schedule["input_set_id"],
                    "repeat_count": schedule["repeat_count"],
                    "formal_worker_count": schedule[
                        "formal_worker_count"
                    ],
                    "formal_call_count": len(schedule["calls"]),
                    "schedule_sha256": schedule["schedule_sha256"],
                },
            },
        )
        MidDualRunStore.verify_manifest(store.run_root)
        return {
            "execution_status": "complete",
            "gate_status": summary["status"],
            "formal_row_count": len(formal_rows),
            "formal_environment_gate_status": "passed",
            "run_root": str(run_root).replace("\\", "/"),
            "summary": summary,
        }
    except G2Interrupted as exc:
        if formal_guard is not None and formal_guard_entered:
            formal_guard.__exit__(
                G2Interrupted,
                exc,
                exc.__traceback__,
            )
            formal_guard_entered = False
        return {
            "execution_status": "interrupted",
            "gate_status": "incomplete",
            "formal_row_count": 0,
            "formal_environment_gate_status": (
                "not_run"
                if formal_guard is None or not formal_guard.audit
                else str(formal_guard.audit.get("status", "blocked"))
            ),
            "run_root": str(run_root).replace("\\", "/"),
            "accepted_phase_ids": (
                [] if store is None else list(store.accepted_phase_ids)
            ),
        }
    except G2Blocked as exc:
        if formal_guard is not None and formal_guard_entered:
            formal_guard.__exit__(
                G2Blocked,
                exc,
                exc.__traceback__,
            )
            formal_guard_entered = False
        if store is None:
            raise
        return _finalize_blocked(
            store=store,
            run_id=run_id,
            mode=mode,
            input_audit=input_audit,
            code_sha256=code_sha256,
            runtime_source_closure=runtime_source_closure,
            reason=exc.reason,
            formal_environment_audit=(
                None
                if formal_guard is None or not formal_guard.audit
                else formal_guard.audit
            ),
            formal_environment_gate_attempted=formal_guard is not None,
        )
    except Exception as exc:
        if formal_guard is not None and formal_guard_entered:
            formal_guard.__exit__(
                type(exc),
                exc,
                exc.__traceback__,
            )
            formal_guard_entered = False
        if store is None:
            raise G2Blocked(
                f"g2_runner_exception:{type(exc).__name__}"
            ) from exc
        return _finalize_blocked(
            store=store,
            run_id=run_id,
            mode=mode,
            input_audit=input_audit,
            code_sha256=code_sha256,
            runtime_source_closure=runtime_source_closure,
            reason=f"g2_runner_exception:{type(exc).__name__}",
            formal_environment_audit=(
                None
                if formal_guard is None or not formal_guard.audit
                else formal_guard.audit
            ),
            formal_environment_gate_attempted=formal_guard is not None,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed reduced G2 multi-platform timing gate."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-bundle", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", required=True, choices=G2_MODES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_g2(
            config_path=args.config,
            input_bundle=args.input_bundle,
            run_id=args.run_id,
            mode=args.mode,
        )
    except G2Blocked as exc:
        result = {
            "execution_status": "not_started",
            "gate_status": "blocked",
            "formal_row_count": 0,
            "formal_environment_gate_status": "not_run",
            "blocking_reason": exc.reason,
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
