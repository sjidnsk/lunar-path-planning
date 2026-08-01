"""Stage 6 跨平台进程树 RSS 采样、峰值监控与资源门禁。"""

from __future__ import annotations

import math
import os
import shutil
import sys
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import TracebackType
from typing import Callable, Mapping, Protocol


GIB = 1024**3


@dataclass(frozen=True, slots=True)
class ProcessMemorySample:
    """单个进程的一次 current RSS 与操作系统历史 peak RSS 样本。"""

    pid: int
    current_rss_bytes: int
    peak_rss_bytes: int | None

    def __post_init__(self) -> None:
        if type(self.pid) is not int or self.pid <= 0:
            raise ValueError("pid must be a positive integer")
        if type(self.current_rss_bytes) is not int or self.current_rss_bytes < 0:
            raise ValueError("current RSS must be a non-negative integer")
        if self.peak_rss_bytes is not None and (
            type(self.peak_rss_bytes) is not int or self.peak_rss_bytes < 0
        ):
            raise ValueError("peak RSS must be a non-negative integer or None")


@dataclass(frozen=True, slots=True)
class ProcessTreeSample:
    """同一采样时刻的 root 进程树内存样本。"""

    root_pid: int
    processes: tuple[ProcessMemorySample, ...]

    def __post_init__(self) -> None:
        if type(self.root_pid) is not int or self.root_pid <= 0:
            raise ValueError("root_pid must be a positive integer")
        if type(self.processes) is not tuple:
            raise TypeError("processes must be a tuple")
        if not self.processes:
            raise ValueError("processes must be non-empty")
        if any(
            not isinstance(process, ProcessMemorySample)
            for process in self.processes
        ):
            raise TypeError("processes must contain only ProcessMemorySample values")
        process_pids = tuple(process.pid for process in self.processes)
        if len(set(process_pids)) != len(process_pids):
            raise ValueError("process PIDs must be unique")
        if self.root_pid not in process_pids:
            raise ValueError("processes must contain root_pid")

    @property
    def aggregate_current_rss_bytes(self) -> int:
        return sum(process.current_rss_bytes for process in self.processes)


class ProcessTreeBackend(Protocol):
    """进程父子关系与进程内存读取的跨平台 backend 合同。"""

    def process_parent_map(self) -> Mapping[int, int]: ...

    def read_process_memory(self, pid: int) -> ProcessMemorySample | None: ...


class _WindowsProcessTreeBackend:
    _TH32CS_SNAPPROCESS = 0x00000002
    _PROCESS_VM_READ = 0x0010
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _ERROR_ACCESS_DENIED = 5
    _ERROR_INVALID_HANDLE = 6
    _ERROR_NO_MORE_FILES = 18
    _ERROR_INVALID_PARAMETER = 87
    _ERROR_NOT_FOUND = 1168

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class ProcessEntry32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", wintypes.WCHAR * 260),
            ]

        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.CreateToolhelp32Snapshot.argtypes = [
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel32.Process32FirstW.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessEntry32W),
        ]
        kernel32.Process32FirstW.restype = wintypes.BOOL
        kernel32.Process32NextW.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessEntry32W),
        ]
        kernel32.Process32NextW.restype = wintypes.BOOL
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCountersEx),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        self._ctypes = ctypes
        self._kernel32 = kernel32
        self._psapi = psapi
        self._process_entry_type = ProcessEntry32W
        self._memory_counters_type = ProcessMemoryCountersEx

    def process_parent_map(self) -> dict[int, int]:
        ctypes = self._ctypes
        snapshot = self._kernel32.CreateToolhelp32Snapshot(
            self._TH32CS_SNAPPROCESS,
            0,
        )
        if snapshot == ctypes.c_void_p(-1).value:
            error = ctypes.get_last_error()
            raise OSError(error, "CreateToolhelp32Snapshot failed")

        parents: dict[int, int] = {}
        try:
            entry = self._process_entry_type()
            entry.dwSize = ctypes.sizeof(entry)
            ctypes.set_last_error(0)
            if not self._kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
                error = ctypes.get_last_error()
                if error == self._ERROR_NO_MORE_FILES:
                    return parents
                raise OSError(error, "Process32FirstW failed")
            while True:
                pid = int(entry.th32ProcessID)
                if pid > 0:
                    parents[pid] = int(entry.th32ParentProcessID)
                ctypes.set_last_error(0)
                if not self._kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                    error = ctypes.get_last_error()
                    if error != self._ERROR_NO_MORE_FILES:
                        raise OSError(error, "Process32NextW failed")
                    break
        finally:
            self._kernel32.CloseHandle(snapshot)
        return parents

    def read_process_memory(self, pid: int) -> ProcessMemorySample | None:
        ctypes = self._ctypes
        ctypes.set_last_error(0)
        process = self._kernel32.OpenProcess(
            self._PROCESS_QUERY_LIMITED_INFORMATION | self._PROCESS_VM_READ,
            False,
            pid,
        )
        if not process:
            error = ctypes.get_last_error()
            if error in {
                self._ERROR_INVALID_HANDLE,
                self._ERROR_INVALID_PARAMETER,
                self._ERROR_NOT_FOUND,
            }:
                return None
            if error == self._ERROR_ACCESS_DENIED:
                raise PermissionError(error, f"cannot read RSS for PID {pid}")
            raise OSError(error, f"OpenProcess failed for PID {pid}")

        try:
            counters = self._memory_counters_type()
            counters.cb = ctypes.sizeof(counters)
            ctypes.set_last_error(0)
            if not self._psapi.GetProcessMemoryInfo(
                process,
                ctypes.byref(counters),
                counters.cb,
            ):
                error = ctypes.get_last_error()
                if error in {
                    self._ERROR_INVALID_HANDLE,
                    self._ERROR_INVALID_PARAMETER,
                    self._ERROR_NOT_FOUND,
                }:
                    return None
                raise OSError(error, f"GetProcessMemoryInfo failed for PID {pid}")
            return ProcessMemorySample(
                pid=pid,
                current_rss_bytes=int(counters.WorkingSetSize),
                peak_rss_bytes=int(counters.PeakWorkingSetSize),
            )
        finally:
            self._kernel32.CloseHandle(process)


class _LinuxProcessTreeBackend:
    def __init__(
        self,
        *,
        proc_root: Path = Path("/proc"),
        fail_on_parent_snapshot_permission_error: bool = False,
    ) -> None:
        self._proc_root = proc_root
        self._fail_on_parent_snapshot_permission_error = (
            fail_on_parent_snapshot_permission_error
        )

    def process_parent_map(self) -> dict[int, int]:
        parents: dict[int, int] = {}
        for process_root in self._proc_root.iterdir():
            if not process_root.name.isdecimal():
                continue
            try:
                status = (process_root / "status").read_text(
                    encoding="ascii",
                    errors="replace",
                )
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError:
                if self._fail_on_parent_snapshot_permission_error:
                    raise
                continue
            parent_pid = self._status_integer(status, "PPid")
            if parent_pid is not None:
                parents[int(process_root.name)] = parent_pid
        return parents

    def read_process_memory(self, pid: int) -> ProcessMemorySample | None:
        try:
            status = (self._proc_root / str(pid) / "status").read_text(
                encoding="ascii",
                errors="replace",
            )
        except (FileNotFoundError, ProcessLookupError):
            return None
        current_kib = self._status_kib(status, "VmRSS")
        if current_kib is None:
            return None
        peak_kib = self._status_kib(status, "VmHWM")
        return ProcessMemorySample(
            pid=pid,
            current_rss_bytes=current_kib * 1024,
            peak_rss_bytes=None if peak_kib is None else peak_kib * 1024,
        )

    @staticmethod
    def _status_integer(status: str, field: str) -> int | None:
        prefix = f"{field}:"
        for line in status.splitlines():
            if line.startswith(prefix):
                fields = line[len(prefix) :].split()
                if not fields:
                    raise ValueError(f"missing {field} value in /proc status")
                return int(fields[0])
        return None

    @classmethod
    def _status_kib(cls, status: str, field: str) -> int | None:
        prefix = f"{field}:"
        for line in status.splitlines():
            if line.startswith(prefix):
                fields = line[len(prefix) :].split()
                if len(fields) != 2 or fields[1] != "kB":
                    raise ValueError(f"invalid {field} value in /proc status")
                return int(fields[0])
        return None


@lru_cache(maxsize=1)
def _native_process_tree_backend() -> ProcessTreeBackend:
    if os.name == "nt":
        return _WindowsProcessTreeBackend()
    if sys.platform.startswith("linux"):
        return _LinuxProcessTreeBackend()
    raise NotImplementedError("native process-tree backend is not implemented")


def sample_process_tree(
    *,
    root_pid: int | None = None,
    backend: ProcessTreeBackend | None = None,
) -> ProcessTreeSample:
    """枚举 root 及全部 descendants，并读取仍存活进程的内存样本。"""

    actual_root_pid = os.getpid() if root_pid is None else root_pid
    if type(actual_root_pid) is not int or actual_root_pid <= 0:
        raise ValueError("root_pid must be a positive integer")
    if backend is None:
        backend = _native_process_tree_backend()

    children_by_parent: dict[int, list[int]] = {}
    for pid, parent_pid in backend.process_parent_map().items():
        children_by_parent.setdefault(parent_pid, []).append(pid)

    pending = [actual_root_pid]
    selected_pids: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in selected_pids:
            continue
        selected_pids.add(pid)
        pending.extend(children_by_parent.get(pid, ()))

    processes: list[ProcessMemorySample] = []
    for pid in sorted(selected_pids):
        try:
            process = backend.read_process_memory(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        if process is not None:
            processes.append(process)
    return ProcessTreeSample(
        root_pid=actual_root_pid,
        processes=tuple(processes),
    )


ProcessTreeSampler = Callable[[int], ProcessTreeSample]


class ProcessTreeSamplingError(RuntimeError):
    """进程树 RSS 监控无法继续提供可信样本。"""


def snapshot_process_parent_map(
    *,
    backend: ProcessTreeBackend | None = None,
) -> dict[int, int]:
    """获取可验证的全局 PID -> parent PID 快照；不确定时 fail closed。"""

    if backend is None:
        if os.name == "nt":
            actual_backend: ProcessTreeBackend = _WindowsProcessTreeBackend()
        elif sys.platform.startswith("linux"):
            actual_backend = _LinuxProcessTreeBackend(
                fail_on_parent_snapshot_permission_error=True,
            )
        else:
            raise ProcessTreeSamplingError(
                "process parent map snapshot failed: unsupported platform"
            )
    else:
        actual_backend = backend
    try:
        raw_snapshot = actual_backend.process_parent_map()
        if not isinstance(raw_snapshot, Mapping):
            raise TypeError("backend parent map must be a mapping")
        snapshot: dict[int, int] = {}
        for pid, parent_pid in raw_snapshot.items():
            if type(pid) is not int or pid <= 0:
                raise ValueError("process parent map PID must be a positive integer")
            if type(parent_pid) is not int or parent_pid < 0:
                raise ValueError(
                    "process parent map parent PID must be a non-negative integer"
                )
            snapshot[pid] = parent_pid
    except Exception as exc:
        raise ProcessTreeSamplingError(
            "process parent map snapshot failed"
        ) from exc
    return snapshot


def _sample_process_tree_for_monitor(root_pid: int) -> ProcessTreeSample:
    return sample_process_tree(root_pid=root_pid)


class ProcessTreeRSSMonitor:
    """周期采样进程树，并保留各次 current RSS 聚合值的生命周期峰值。

    每次采样只聚合同一时刻各进程的 ``current_rss_bytes``。单进程
    ``peak_rss_bytes`` 不参与求和，因此不会伪造并不同时发生的树峰值。
    公开计数证据可在停止或失败后读取；停止等待使用有界 join timeout，
    超时会显式 fail closed，不会把仍存活的监控线程报告为已停止。
    """

    def __init__(
        self,
        *,
        root_pid: int | None = None,
        interval_seconds: float = 0.1,
        join_timeout_seconds: float = 5.0,
        sampler: ProcessTreeSampler = _sample_process_tree_for_monitor,
    ) -> None:
        actual_root_pid = os.getpid() if root_pid is None else root_pid
        if type(actual_root_pid) is not int or actual_root_pid <= 0:
            raise ValueError("root_pid must be a positive integer")
        if (
            type(interval_seconds) not in (int, float)
            or not math.isfinite(interval_seconds)
            or interval_seconds <= 0
        ):
            raise ValueError("interval_seconds must be a finite positive number")
        if (
            type(join_timeout_seconds) not in (int, float)
            or not math.isfinite(join_timeout_seconds)
            or join_timeout_seconds <= 0
        ):
            raise ValueError(
                "join_timeout_seconds must be a finite positive number"
            )
        if not callable(sampler):
            raise TypeError("sampler must be callable")

        self._root_pid = actual_root_pid
        self._interval_seconds = float(interval_seconds)
        self._join_timeout_seconds = float(join_timeout_seconds)
        self._sampler = sampler
        self._lock = threading.RLock()
        self._sample_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = "new"
        self._latest_sample: ProcessTreeSample | None = None
        self._peak_aggregate_rss_bytes = 0
        self._sample_count = 0
        self._latest_process_count = 0
        self._peak_process_count = 0
        self._sampling_error: Exception | None = None

    def __enter__(self) -> ProcessTreeRSSMonitor:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del traceback
        try:
            self.stop()
        except ProcessTreeSamplingError as monitor_error:
            if exc_type is None or exc is None:
                raise
            exc.add_note(str(monitor_error))
        return False

    def start(self) -> ProcessTreeRSSMonitor:
        with self._lock:
            if self._state == "running":
                return self
            if self._state == "failed":
                self._raise_sampling_error()
            if self._state != "new":
                raise RuntimeError("process-tree RSS monitor cannot be restarted")
            self._sample_once(expected_state="new")
            self._state = "running"
            thread = threading.Thread(
                target=self._run,
                name=f"process-tree-rss-{self._root_pid}",
                daemon=True,
            )
            self._thread = thread
            try:
                thread.start()
            except Exception as exc:
                self._thread = None
                self._mark_failed(exc)
                self._raise_sampling_error()
        return self

    def stop(self) -> None:
        with self._lock:
            if self._state == "failed":
                self._stop_event.set()
                thread = self._thread
            elif self._state == "new":
                self._state = "stopped"
                return
            elif self._state == "stopped":
                return
            elif self._state == "running":
                self._state = "stopping"
                self._stop_event.set()
                thread = self._thread
            else:
                thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self._join_timeout_seconds)
            if thread.is_alive():
                timeout_error = TimeoutError(
                    "process-tree RSS monitor thread did not stop within "
                    f"{self._join_timeout_seconds:.3f} seconds"
                )
                with self._lock:
                    self._mark_failed(timeout_error)
                    self._raise_sampling_error()
        with self._lock:
            if self._state == "failed":
                self._raise_sampling_error()
            self._state = "stopped"

    def sample_now(self) -> ProcessTreeSample:
        return self._sample_once(expected_state="running")

    @property
    def running(self) -> bool:
        with self._lock:
            return self._state == "running"

    @property
    def root_pid(self) -> int:
        with self._lock:
            return self._root_pid

    @property
    def sample_count(self) -> int:
        with self._lock:
            return self._sample_count

    @property
    def latest_process_count(self) -> int:
        with self._lock:
            return self._latest_process_count

    @property
    def peak_process_count(self) -> int:
        with self._lock:
            return self._peak_process_count

    @property
    def latest_aggregate_rss_bytes(self) -> int:
        with self._lock:
            if self._state == "failed":
                self._raise_sampling_error()
            if self._latest_sample is None:
                raise RuntimeError("process-tree RSS monitor has no samples")
            return self._latest_sample.aggregate_current_rss_bytes

    @property
    def peak_aggregate_rss_bytes(self) -> int:
        with self._lock:
            if self._state == "failed":
                self._raise_sampling_error()
            if self._latest_sample is None:
                raise RuntimeError("process-tree RSS monitor has no samples")
            return self._peak_aggregate_rss_bytes

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            try:
                self.sample_now()
            except ProcessTreeSamplingError:
                return
            except RuntimeError:
                if self._stop_event.is_set():
                    return
                raise

    def _sample_once(self, *, expected_state: str) -> ProcessTreeSample:
        with self._sample_lock:
            with self._lock:
                if self._state == "failed":
                    self._raise_sampling_error()
                if self._state != expected_state:
                    raise RuntimeError("process-tree RSS monitor is not running")
            try:
                sample = self._sampler(self._root_pid)
            except Exception as exc:
                with self._lock:
                    self._mark_failed(exc)
                    self._raise_sampling_error()
            with self._lock:
                if self._state == "failed":
                    self._raise_sampling_error()
                if self._state != expected_state:
                    raise RuntimeError("process-tree RSS monitor is not running")
                try:
                    self._record_sample(sample)
                except Exception as exc:
                    self._mark_failed(exc)
                    self._raise_sampling_error()
                return sample

    def _mark_failed(self, error: Exception) -> None:
        if self._sampling_error is None:
            self._sampling_error = error
        self._state = "failed"
        self._stop_event.set()

    def _raise_sampling_error(self) -> None:
        error = self._sampling_error
        if error is None:
            raise ProcessTreeSamplingError("process-tree RSS sampling failed")
        raise ProcessTreeSamplingError(
            f"process-tree RSS sampling failed: {error}"
        ) from error

    def _record_sample(self, sample: ProcessTreeSample) -> None:
        if not isinstance(sample, ProcessTreeSample):
            raise TypeError("sampler must return ProcessTreeSample")
        if sample.root_pid != self._root_pid:
            raise ValueError("sampler returned a different root PID")
        aggregate = sample.aggregate_current_rss_bytes
        process_count = len(sample.processes)
        if self._sample_count == 0 or aggregate > self._peak_aggregate_rss_bytes:
            self._peak_aggregate_rss_bytes = aggregate
            self._peak_process_count = process_count
        self._latest_sample = sample
        self._latest_process_count = process_count
        self._sample_count += 1


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    d_free_bytes: int
    rss_bytes: int
    peak_vram_bytes: int
    rss_source: str = "process_tree_instantaneous_current_sum/v1"
    rss_root_pid: int = field(default_factory=os.getpid)
    rss_sample_count: int = 1
    rss_latest_process_count: int = 1
    rss_peak_process_count: int = 1

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (self.d_free_bytes, self.rss_bytes, self.peak_vram_bytes)
        ):
            raise ValueError("resource snapshot values must be non-negative integers")
        if self.rss_source not in {
            "process_tree_instantaneous_current_sum/v1",
            "process_tree_lifecycle_peak_current_sum/v1",
        }:
            raise ValueError("resource snapshot RSS source drifted")
        if (
            type(self.rss_root_pid) is not int
            or self.rss_root_pid <= 0
            or type(self.rss_sample_count) is not int
            or self.rss_sample_count <= 0
            or type(self.rss_latest_process_count) is not int
            or self.rss_latest_process_count <= 0
            or type(self.rss_peak_process_count) is not int
            or self.rss_peak_process_count <= 0
        ):
            raise ValueError("resource snapshot RSS provenance drifted")
        if (
            self.rss_source == "process_tree_instantaneous_current_sum/v1"
            and (
                self.rss_sample_count != 1
                or self.rss_latest_process_count != self.rss_peak_process_count
            )
        ):
            raise ValueError("instantaneous RSS provenance drifted")


@dataclass(frozen=True, slots=True)
class ResourceDecision:
    warnings: tuple[str, ...]
    hard_stops: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.hard_stops


class ResourceHardStopError(RuntimeError):
    """Raised after a runtime resource gate latches for one process attempt."""


class ResourceHardStopLatch:
    """Fail-closed runtime resource latch owned by one process/run attempt."""

    def __init__(
        self,
        *,
        attempt_id: str,
        snapshot_provider: Callable[[], ResourceSnapshot],
    ) -> None:
        if not isinstance(attempt_id, str) or not attempt_id.strip():
            raise ValueError("resource latch attempt_id must be non-empty")
        if not callable(snapshot_provider):
            raise TypeError("resource latch snapshot_provider must be callable")
        self._attempt_id = attempt_id
        self._snapshot_provider = snapshot_provider
        self._owner_pid = os.getpid()
        self._lock = threading.Lock()
        self._first_failure_boundary: str | None = None
        self._hard_stops: tuple[str, ...] = ()
        self._warnings: list[str] = []
        self._latest_snapshot: ResourceSnapshot | None = None

    @property
    def attempt_id(self) -> str:
        return self._attempt_id

    @property
    def owner_pid(self) -> int:
        return self._owner_pid

    @property
    def latched(self) -> bool:
        return bool(self._hard_stops)

    @property
    def first_failure_boundary(self) -> str | None:
        return self._first_failure_boundary

    @property
    def hard_stops(self) -> tuple[str, ...]:
        return self._hard_stops

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    @property
    def latest_snapshot(self) -> ResourceSnapshot | None:
        return self._latest_snapshot

    def poll(self, boundary: str) -> tuple[ResourceSnapshot, ResourceDecision]:
        if not isinstance(boundary, str) or not boundary.strip():
            raise ValueError("resource latch boundary must be non-empty")
        with self._lock:
            if self._hard_stops:
                self._raise_latched()
            if os.getpid() != self._owner_pid:
                self._latch(
                    boundary=boundary,
                    hard_stops=("resource latch owner process changed",),
                )
                self._raise_latched()
            try:
                snapshot = self._snapshot_provider()
                if not isinstance(snapshot, ResourceSnapshot):
                    raise TypeError("resource sampler returned an invalid snapshot")
                decision = evaluate_resource_gates(snapshot, preflight=False)
            except Exception as exc:
                self._latch(
                    boundary=boundary,
                    hard_stops=("resource sampler failure",),
                )
                raise ResourceHardStopError(
                    self._failure_message()
                ) from exc
            self._latest_snapshot = snapshot
            for warning in decision.warnings:
                if warning not in self._warnings:
                    self._warnings.append(warning)
            if decision.hard_stops:
                self._latch(
                    boundary=boundary,
                    hard_stops=decision.hard_stops,
                )
                self._raise_latched()
            return snapshot, decision

    def _latch(
        self,
        *,
        boundary: str,
        hard_stops: tuple[str, ...],
    ) -> None:
        if not self._hard_stops:
            self._first_failure_boundary = boundary
            self._hard_stops = tuple(hard_stops)

    def _failure_message(self) -> str:
        reasons = "; ".join(self._hard_stops)
        return (
            f"{reasons} [attempt={self._attempt_id}, "
            f"owner_pid={self._owner_pid}, "
            f"boundary={self._first_failure_boundary}]"
        )

    def _raise_latched(self) -> None:
        raise ResourceHardStopError(self._failure_message())


def evaluate_resource_gates(
    snapshot: ResourceSnapshot,
    *,
    preflight: bool,
) -> ResourceDecision:
    warnings: list[str] = []
    hard_stops: list[str] = []
    disk_threshold = 100 * GIB if preflight else 50 * GIB
    if snapshot.d_free_bytes < disk_threshold:
        hard_stops.append(
            "D free space below 100 GiB preflight"
            if preflight
            else "D free space below 50 GiB runtime"
        )
    if snapshot.rss_bytes >= 16 * GIB:
        warnings.append("process RSS reached 16 GiB")
    if snapshot.rss_bytes >= 20 * GIB:
        hard_stops.append("process RSS reached 20 GiB hard stop")
    if snapshot.peak_vram_bytes > 9 * GIB:
        warnings.append("peak VRAM exceeded 9 GiB")
    if snapshot.peak_vram_bytes >= int(10.1 * GIB):
        hard_stops.append(
            "peak VRAM reached 10.1 GiB hard stop"
            if preflight
            else "peak VRAM exceeded 10.1 GiB hard stop"
        )
    return ResourceDecision(tuple(warnings), tuple(hard_stops))


def capture_resource_snapshot(
    *,
    peak_vram_bytes: int,
    process_tree_monitor: ProcessTreeRSSMonitor | None = None,
) -> ResourceSnapshot:
    """捕获资源快照；传入 monitor 时将已观测树峰值绑定到 ``rss_bytes``。"""

    if process_tree_monitor is None:
        sample = sample_process_tree()
        rss_bytes = sample.aggregate_current_rss_bytes
        rss_source = "process_tree_instantaneous_current_sum/v1"
        rss_root_pid = sample.root_pid
        rss_sample_count = 1
        rss_latest_process_count = len(sample.processes)
        rss_peak_process_count = len(sample.processes)
    else:
        if process_tree_monitor.running:
            process_tree_monitor.sample_now()
        rss_bytes = process_tree_monitor.peak_aggregate_rss_bytes
        rss_source = "process_tree_lifecycle_peak_current_sum/v1"
        rss_root_pid = process_tree_monitor.root_pid
        rss_sample_count = process_tree_monitor.sample_count
        rss_latest_process_count = process_tree_monitor.latest_process_count
        rss_peak_process_count = process_tree_monitor.peak_process_count
    return ResourceSnapshot(
        d_free_bytes=int(shutil.disk_usage(Path("D:/")).free),
        rss_bytes=rss_bytes,
        peak_vram_bytes=int(peak_vram_bytes),
        rss_source=rss_source,
        rss_root_pid=rss_root_pid,
        rss_sample_count=rss_sample_count,
        rss_latest_process_count=rss_latest_process_count,
        rss_peak_process_count=rss_peak_process_count,
    )


def process_rss_bytes(
    pid: int | None = None,
    *,
    backend: ProcessTreeBackend | None = None,
) -> int:
    """读取指定 PID（默认当前进程）的 current RSS。"""

    actual_pid = os.getpid() if pid is None else pid
    if type(actual_pid) is not int or actual_pid <= 0:
        raise ValueError("pid must be a positive integer")
    actual_backend = _native_process_tree_backend() if backend is None else backend
    try:
        sample = actual_backend.read_process_memory(actual_pid)
    except (FileNotFoundError, ProcessLookupError):
        sample = None
    if sample is None:
        raise ProcessLookupError(actual_pid, "process exited before RSS sampling")
    if sample.pid != actual_pid:
        raise ValueError("backend returned a different PID")
    return sample.current_rss_bytes


__all__ = [
    "GIB",
    "ProcessMemorySample",
    "ProcessTreeBackend",
    "ProcessTreeRSSMonitor",
    "ProcessTreeSample",
    "ProcessTreeSampler",
    "ProcessTreeSamplingError",
    "ResourceDecision",
    "ResourceHardStopError",
    "ResourceHardStopLatch",
    "ResourceSnapshot",
    "capture_resource_snapshot",
    "evaluate_resource_gates",
    "process_rss_bytes",
    "sample_process_tree",
    "snapshot_process_parent_map",
]
