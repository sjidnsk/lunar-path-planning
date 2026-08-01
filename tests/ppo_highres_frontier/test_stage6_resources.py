"""Stage 6 进程树 RSS 采样与资源门禁测试。"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lunar_exploration_ppo.utils import resources


class _FakeProcessTreeBackend:
    def __init__(
        self,
        *,
        parent_pids: dict[int, int],
        memory_by_pid: dict[int, resources.ProcessMemorySample],
    ) -> None:
        self._parent_pids = parent_pids
        self._memory_by_pid = memory_by_pid

    def process_parent_map(self) -> dict[int, int]:
        return dict(self._parent_pids)

    def read_process_memory(
        self,
        pid: int,
    ) -> resources.ProcessMemorySample | None:
        return self._memory_by_pid.get(pid)


@pytest.mark.parametrize(
    ("root_pid", "processes", "error_type", "message"),
    (
        (
            0,
            (resources.ProcessMemorySample(1, 10, 20),),
            ValueError,
            "root_pid",
        ),
        (
            1,
            [resources.ProcessMemorySample(1, 10, 20)],
            TypeError,
            "tuple",
        ),
        (1, (), ValueError, "non-empty"),
        (1, (object(),), TypeError, "ProcessMemorySample"),
        (
            1,
            (
                resources.ProcessMemorySample(1, 10, 20),
                resources.ProcessMemorySample(1, 30, 40),
            ),
            ValueError,
            "unique",
        ),
        (
            1,
            (resources.ProcessMemorySample(2, 10, 20),),
            ValueError,
            "contain root_pid",
        ),
    ),
)
def test_process_tree_sample_rejects_invalid_contracts(
    root_pid: int,
    processes: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        resources.ProcessTreeSample(  # type: ignore[arg-type]
            root_pid=root_pid,
            processes=processes,
        )


def test_process_tree_sampler_fails_closed_when_root_exits_before_memory_read() -> None:
    backend = _FakeProcessTreeBackend(
        parent_pids={101: 100},
        memory_by_pid={101: resources.ProcessMemorySample(101, 20, 30)},
    )

    with pytest.raises(ValueError, match="contain root_pid"):
        resources.sample_process_tree(root_pid=100, backend=backend)


def test_process_tree_sample_aggregates_root_and_all_descendant_current_rss() -> None:
    backend = _FakeProcessTreeBackend(
        parent_pids={101: 100, 102: 101, 200: 1},
        memory_by_pid={
            100: resources.ProcessMemorySample(100, 10, 100),
            101: resources.ProcessMemorySample(101, 20, 200),
            102: resources.ProcessMemorySample(102, 30, 300),
            200: resources.ProcessMemorySample(200, 1_000, 2_000),
        },
    )

    sample = resources.sample_process_tree(root_pid=100, backend=backend)

    assert tuple(process.pid for process in sample.processes) == (100, 101, 102)
    assert sample.aggregate_current_rss_bytes == 60
    assert tuple(process.peak_rss_bytes for process in sample.processes) == (
        100,
        200,
        300,
    )


def test_process_parent_map_snapshot_returns_validated_copy() -> None:
    backend = _FakeProcessTreeBackend(
        parent_pids={100: 1, 101: 100},
        memory_by_pid={},
    )

    snapshot = resources.snapshot_process_parent_map(backend=backend)

    assert snapshot == {100: 1, 101: 100}
    assert snapshot is not backend._parent_pids


@pytest.mark.parametrize("uncertainty_kind", ("provider_error", "invalid_pid"))
def test_process_parent_map_snapshot_fails_closed_on_uncertainty(
    uncertainty_kind: str,
) -> None:
    class UncertainBackend(_FakeProcessTreeBackend):
        def process_parent_map(self) -> dict[int, int]:
            if uncertainty_kind == "provider_error":
                raise PermissionError("snapshot access denied")
            return {100: "unknown"}  # type: ignore[dict-item]

    backend = UncertainBackend(parent_pids={}, memory_by_pid={})

    with pytest.raises(
        resources.ProcessTreeSamplingError,
        match="process parent map snapshot failed",
    ):
        resources.snapshot_process_parent_map(backend=backend)


def test_native_process_tree_sampler_reads_parent_and_child_rss() -> None:
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import os, time; "
                "payload = bytearray(4 * 1024 * 1024); "
                "print(os.getpid(), flush=True); "
                "time.sleep(30)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert int(child.stdout.readline().strip()) == child.pid

        sample = resources.sample_process_tree(root_pid=os.getpid())
        by_pid = {process.pid: process for process in sample.processes}

        assert os.getpid() in by_pid
        assert child.pid in by_pid
        assert by_pid[os.getpid()].current_rss_bytes > 0
        assert by_pid[child.pid].current_rss_bytes > 0
        assert by_pid[os.getpid()].peak_rss_bytes is not None
        assert by_pid[child.pid].peak_rss_bytes is not None
        assert sample.aggregate_current_rss_bytes >= (
            by_pid[os.getpid()].current_rss_bytes
            + by_pid[child.pid].current_rss_bytes
        )
    finally:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def _write_proc_status(
    proc_root: Path,
    *,
    pid: int,
    parent_pid: int,
    current_kib: int,
    peak_kib: int,
) -> None:
    process_root = proc_root / str(pid)
    process_root.mkdir(parents=True)
    (process_root / "status").write_text(
        (
            "Name:\tpython\n"
            f"Pid:\t{pid}\n"
            f"PPid:\t{parent_pid}\n"
            f"VmHWM:\t{peak_kib} kB\n"
            f"VmRSS:\t{current_kib} kB\n"
        ),
        encoding="ascii",
    )


def test_linux_proc_backend_reads_descendants_current_and_peak_rss(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    _write_proc_status(
        proc_root,
        pid=100,
        parent_pid=1,
        current_kib=10,
        peak_kib=100,
    )
    _write_proc_status(
        proc_root,
        pid=101,
        parent_pid=100,
        current_kib=20,
        peak_kib=200,
    )
    _write_proc_status(
        proc_root,
        pid=102,
        parent_pid=101,
        current_kib=30,
        peak_kib=300,
    )
    _write_proc_status(
        proc_root,
        pid=200,
        parent_pid=1,
        current_kib=1_000,
        peak_kib=2_000,
    )
    backend = resources._LinuxProcessTreeBackend(proc_root=proc_root)

    sample = resources.sample_process_tree(root_pid=100, backend=backend)

    assert tuple(process.pid for process in sample.processes) == (100, 101, 102)
    assert sample.aggregate_current_rss_bytes == 60 * 1024
    assert tuple(process.peak_rss_bytes for process in sample.processes) == (
        100 * 1024,
        200 * 1024,
        300 * 1024,
    )
    assert backend.read_process_memory(999) is None


def test_process_tree_sampler_ignores_process_exit_race() -> None:
    class ExitingChildBackend(_FakeProcessTreeBackend):
        def read_process_memory(
            self,
            pid: int,
        ) -> resources.ProcessMemorySample | None:
            if pid == 101:
                raise ProcessLookupError(pid, "process exited")
            return super().read_process_memory(pid)

    backend = ExitingChildBackend(
        parent_pids={101: 100},
        memory_by_pid={100: resources.ProcessMemorySample(100, 10, 20)},
    )

    sample = resources.sample_process_tree(root_pid=100, backend=backend)

    assert tuple(process.pid for process in sample.processes) == (100,)
    assert sample.aggregate_current_rss_bytes == 10


def _tree_sample(
    *processes: tuple[int, int, int | None],
) -> resources.ProcessTreeSample:
    return resources.ProcessTreeSample(
        root_pid=100,
        processes=tuple(
            resources.ProcessMemorySample(pid, current, peak)
            for pid, current, peak in processes
        ),
    )


def test_monitor_exposes_read_only_sample_and_process_count_evidence() -> None:
    samples = iter(
        (
            _tree_sample((100, 10, 100), (101, 20, 200)),
            _tree_sample((100, 20, 100)),
            _tree_sample((100, 10, 100), (101, 10, 200), (102, 20, 300)),
            _tree_sample(
                (100, 10, 100),
                (101, 10, 200),
                (102, 10, 300),
                (103, 10, 400),
            ),
        )
    )
    monitor = resources.ProcessTreeRSSMonitor(
        root_pid=100,
        interval_seconds=60.0,
        sampler=lambda root_pid: next(samples),
    )

    assert monitor.root_pid == 100
    assert monitor.sample_count == 0
    assert monitor.latest_process_count == 0
    assert monitor.peak_process_count == 0

    monitor.start()
    assert (
        monitor.sample_count,
        monitor.latest_process_count,
        monitor.peak_process_count,
    ) == (1, 2, 2)
    monitor.sample_now()
    assert (
        monitor.sample_count,
        monitor.latest_process_count,
        monitor.peak_process_count,
    ) == (2, 1, 2)
    monitor.sample_now()
    assert (
        monitor.sample_count,
        monitor.latest_process_count,
        monitor.peak_process_count,
    ) == (3, 3, 3)
    monitor.sample_now()
    assert (
        monitor.sample_count,
        monitor.latest_process_count,
        monitor.peak_process_count,
    ) == (4, 4, 3)
    monitor.stop()

    assert monitor.root_pid == 100
    assert monitor.sample_count == 4
    assert monitor.latest_process_count == 4
    assert monitor.peak_process_count == 3
    for attribute in (
        "root_pid",
        "sample_count",
        "latest_process_count",
        "peak_process_count",
    ):
        with pytest.raises(AttributeError):
            setattr(monitor, attribute, 999)


def test_monitor_failed_sample_does_not_advance_public_evidence() -> None:
    calls = 0

    def sampler(root_pid: int) -> resources.ProcessTreeSample:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _tree_sample((root_pid, 10, 20))
        raise OSError("failed evidence sample")

    monitor = resources.ProcessTreeRSSMonitor(
        root_pid=100,
        interval_seconds=60.0,
        sampler=sampler,
    ).start()

    with pytest.raises(resources.ProcessTreeSamplingError):
        monitor.sample_now()

    assert monitor.sample_count == 1
    assert monitor.latest_process_count == 1
    assert monitor.peak_process_count == 1
    with pytest.raises(resources.ProcessTreeSamplingError):
        monitor.stop()


def test_monitor_retains_peak_aggregate_current_rss_after_short_lived_child() -> None:
    samples = iter(
        (
            _tree_sample((100, 10, 1_000)),
            _tree_sample((100, 10, 1_000), (101, 50, 5_000)),
            _tree_sample((100, 12, 1_000)),
        )
    )
    monitor = resources.ProcessTreeRSSMonitor(
        root_pid=100,
        interval_seconds=60.0,
        sampler=lambda root_pid: next(samples),
    )

    monitor.start()
    assert monitor.peak_aggregate_rss_bytes == 10
    monitor.sample_now()
    assert monitor.peak_aggregate_rss_bytes == 60
    monitor.sample_now()
    assert monitor.latest_aggregate_rss_bytes == 12
    assert monitor.peak_aggregate_rss_bytes == 60
    monitor.stop()

    assert monitor.peak_aggregate_rss_bytes == 60


def test_monitor_fails_closed_after_sampling_error() -> None:
    calls = 0

    def sampler(root_pid: int) -> resources.ProcessTreeSample:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _tree_sample((root_pid, 10, 20))
        raise OSError("backend failed")

    monitor = resources.ProcessTreeRSSMonitor(
        root_pid=100,
        interval_seconds=60.0,
        sampler=sampler,
    ).start()

    with pytest.raises(resources.ProcessTreeSamplingError, match="backend failed"):
        monitor.sample_now()
    with pytest.raises(resources.ProcessTreeSamplingError, match="backend failed"):
        _ = monitor.peak_aggregate_rss_bytes
    with pytest.raises(resources.ProcessTreeSamplingError, match="backend failed"):
        monitor.stop()
    assert monitor.running is False


def test_monitor_context_periodically_samples_and_stops() -> None:
    calls = 0
    second_sampled = threading.Event()

    def sampler(root_pid: int) -> resources.ProcessTreeSample:
        nonlocal calls
        calls += 1
        if calls >= 2:
            second_sampled.set()
        return _tree_sample((root_pid, 10 * calls, 1_000))

    monitor = resources.ProcessTreeRSSMonitor(
        root_pid=100,
        interval_seconds=0.01,
        sampler=sampler,
    )

    with monitor:
        assert second_sampled.wait(timeout=1.0)

    assert monitor.running is False
    assert monitor.peak_aggregate_rss_bytes >= 20


def test_monitor_start_and_stop_are_idempotent_and_restart_fails_closed() -> None:
    calls = 0

    def sampler(root_pid: int) -> resources.ProcessTreeSample:
        nonlocal calls
        calls += 1
        return _tree_sample((root_pid, 10, 20))

    monitor = resources.ProcessTreeRSSMonitor(
        root_pid=100,
        interval_seconds=60.0,
        sampler=sampler,
    )
    assert monitor.start() is monitor
    assert monitor.start() is monitor
    assert calls == 1
    monitor.stop()
    monitor.stop()
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        monitor.start()

    stopped_before_start = resources.ProcessTreeRSSMonitor(
        root_pid=100,
        interval_seconds=60.0,
        sampler=sampler,
    )
    stopped_before_start.stop()
    stopped_before_start.stop()
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        stopped_before_start.start()


@pytest.mark.parametrize(
    ("tree_peak_bytes", "warning", "hard_stop"),
    (
        (16 * resources.GIB - 1, False, False),
        (16 * resources.GIB, True, False),
        (20 * resources.GIB, True, True),
        (20 * resources.GIB + 1, True, True),
    ),
)
def test_capture_snapshot_binds_monitor_tree_peak_to_rss_gates(
    monkeypatch: pytest.MonkeyPatch,
    tree_peak_bytes: int,
    warning: bool,
    hard_stop: bool,
) -> None:
    parent_rss = tree_peak_bytes // 2
    child_rss = tree_peak_bytes - parent_rss
    monitor = resources.ProcessTreeRSSMonitor(
        root_pid=100,
        interval_seconds=60.0,
        sampler=lambda root_pid: _tree_sample(
            (root_pid, parent_rss, parent_rss),
            (101, child_rss, child_rss),
        ),
    ).start()
    monitor.stop()
    monkeypatch.setattr(
        resources.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=200 * resources.GIB),
    )

    snapshot = resources.capture_resource_snapshot(
        peak_vram_bytes=0,
        process_tree_monitor=monitor,
    )
    decision = resources.evaluate_resource_gates(snapshot, preflight=False)

    assert snapshot.rss_bytes == tree_peak_bytes
    assert snapshot.rss_source == "process_tree_lifecycle_peak_current_sum/v1"
    assert snapshot.rss_root_pid == 100
    assert snapshot.rss_sample_count == 1
    assert snapshot.rss_latest_process_count == 2
    assert snapshot.rss_peak_process_count == 2
    assert bool(decision.warnings) is warning
    assert bool(decision.hard_stops) is hard_stop


def test_capture_snapshot_compatible_call_uses_one_shot_tree_current_rss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resources,
        "sample_process_tree",
        lambda: _tree_sample((100, 10, 100), (101, 20, 200)),
    )
    monkeypatch.setattr(
        resources.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=200 * resources.GIB),
    )

    snapshot = resources.capture_resource_snapshot(peak_vram_bytes=0)

    assert snapshot.rss_bytes == 30
    assert snapshot.rss_source == "process_tree_instantaneous_current_sum/v1"
    assert snapshot.rss_root_pid == 100
    assert snapshot.rss_sample_count == 1
    assert snapshot.rss_latest_process_count == 2
    assert snapshot.rss_peak_process_count == 2


def test_resource_gate_hard_stops_at_exact_rss_and_vram_limits() -> None:
    from lunar_exploration_ppo.utils.resources import (
        GIB,
        ResourceSnapshot,
        evaluate_resource_gates,
    )

    rss_decision = evaluate_resource_gates(
        ResourceSnapshot(
            d_free_bytes=100 * GIB,
            rss_bytes=20 * GIB,
            peak_vram_bytes=0,
        ),
        preflight=True,
    )
    vram_decision = evaluate_resource_gates(
        ResourceSnapshot(
            d_free_bytes=100 * GIB,
            rss_bytes=0,
            peak_vram_bytes=int(10.1 * GIB),
        ),
        preflight=True,
    )

    assert rss_decision.passed is False
    assert rss_decision.hard_stops == ("process RSS reached 20 GiB hard stop",)
    assert vram_decision.passed is False
    assert vram_decision.hard_stops == ("peak VRAM reached 10.1 GiB hard stop",)


def test_runtime_resource_latch_hard_stops_at_exact_vram_limit() -> None:
    snapshot = resources.ResourceSnapshot(
        d_free_bytes=50 * resources.GIB,
        rss_bytes=0,
        peak_vram_bytes=int(10.1 * resources.GIB),
    )
    latch = resources.ResourceHardStopLatch(
        attempt_id="update:seed-20260716:update-001:attempt-exact-vram-limit",
        snapshot_provider=lambda: snapshot,
    )

    with pytest.raises(
        resources.ResourceHardStopError,
        match="peak VRAM exceeded 10.1 GiB hard stop",
    ):
        latch.poll("rollout:step:1")

    assert latch.latched is True
    assert latch.first_failure_boundary == "rollout:step:1"
    assert latch.hard_stops == ("peak VRAM exceeded 10.1 GiB hard stop",)


@pytest.mark.parametrize(
    ("snapshot", "expected_reason"),
    (
        (
            resources.ResourceSnapshot(
                d_free_bytes=50 * resources.GIB - 1,
                rss_bytes=0,
                peak_vram_bytes=0,
            ),
            "D free space below 50 GiB runtime",
        ),
        (
            resources.ResourceSnapshot(
                d_free_bytes=50 * resources.GIB,
                rss_bytes=20 * resources.GIB,
                peak_vram_bytes=0,
            ),
            "process RSS reached 20 GiB hard stop",
        ),
        (
            resources.ResourceSnapshot(
                d_free_bytes=50 * resources.GIB,
                rss_bytes=0,
                peak_vram_bytes=int(10.1 * resources.GIB) + 1,
            ),
            "peak VRAM exceeded 10.1 GiB hard stop",
        ),
    ),
)
def test_runtime_resource_latch_records_first_hard_stop_for_process_attempt(
    snapshot: resources.ResourceSnapshot,
    expected_reason: str,
) -> None:
    latch = resources.ResourceHardStopLatch(
        attempt_id="update:seed-20260716:update-001:attempt-1",
        snapshot_provider=lambda: snapshot,
    )

    with pytest.raises(resources.ResourceHardStopError, match=expected_reason):
        latch.poll("rollout:step:1")

    assert latch.owner_pid == os.getpid()
    assert latch.attempt_id == "update:seed-20260716:update-001:attempt-1"
    assert latch.latched is True
    assert latch.first_failure_boundary == "rollout:step:1"
    assert latch.hard_stops == (expected_reason,)
    with pytest.raises(resources.ResourceHardStopError, match=expected_reason):
        latch.poll("checkpoint:before-publication")


def test_runtime_resource_latch_allows_one_byte_below_vram_limit_with_warnings() -> None:
    snapshot = resources.ResourceSnapshot(
        d_free_bytes=50 * resources.GIB,
        rss_bytes=16 * resources.GIB,
        peak_vram_bytes=int(10.1 * resources.GIB) - 1,
    )
    latch = resources.ResourceHardStopLatch(
        attempt_id="final:test:ppo_policy:attempt-1",
        snapshot_provider=lambda: snapshot,
    )

    observed, decision = latch.poll("evaluation:episode:1")

    assert observed == snapshot
    assert decision.passed is True
    assert decision.warnings == (
        "process RSS reached 16 GiB",
        "peak VRAM exceeded 9 GiB",
    )
    assert latch.latched is False


def test_runtime_resource_latch_latches_sampler_failure() -> None:
    calls = 0

    def fail_sample() -> resources.ResourceSnapshot:
        nonlocal calls
        calls += 1
        raise OSError("sampler unavailable")

    latch = resources.ResourceHardStopLatch(
        attempt_id="update:seed-20260716:update-001:attempt-2",
        snapshot_provider=fail_sample,
    )

    with pytest.raises(
        resources.ResourceHardStopError,
        match="resource sampler failure",
    ):
        latch.poll("ppo:minibatch:1")
    with pytest.raises(
        resources.ResourceHardStopError,
        match="resource sampler failure",
    ):
        latch.poll("ppo:minibatch:2")

    assert calls == 1
    assert latch.latched is True
    assert latch.first_failure_boundary == "ppo:minibatch:1"


def test_process_rss_bytes_reads_current_rss_for_requested_pid() -> None:
    backend = _FakeProcessTreeBackend(
        parent_pids={},
        memory_by_pid={100: resources.ProcessMemorySample(100, 10, 1_000)},
    )

    assert resources.process_rss_bytes(pid=100, backend=backend) == 10


def test_process_tree_monitor_api_is_public() -> None:
    assert {
        "ProcessMemorySample",
        "ProcessTreeBackend",
        "ProcessTreeRSSMonitor",
        "ProcessTreeSample",
        "ProcessTreeSampler",
        "ProcessTreeSamplingError",
        "sample_process_tree",
    }.issubset(resources.__all__)


def test_resources_module_docstring_describes_process_tree_monitoring() -> None:
    assert resources.__doc__ == (
        "Stage 6 跨平台进程树 RSS 采样、峰值监控与资源门禁。"
    )


@pytest.mark.parametrize(
    "interval_seconds",
    (True, 0.0, -1.0, float("nan"), float("inf")),
)
def test_monitor_rejects_non_finite_or_non_positive_interval(
    interval_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        resources.ProcessTreeRSSMonitor(interval_seconds=interval_seconds)


def test_background_sampling_error_is_reported_by_stop() -> None:
    calls = 0
    failed_sample_entered = threading.Event()

    def sampler(root_pid: int) -> resources.ProcessTreeSample:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _tree_sample((root_pid, 10, 20))
        failed_sample_entered.set()
        raise OSError("periodic backend failed")

    monitor = resources.ProcessTreeRSSMonitor(
        root_pid=100,
        interval_seconds=0.01,
        sampler=sampler,
    ).start()

    assert failed_sample_entered.wait(timeout=1.0)
    with pytest.raises(
        resources.ProcessTreeSamplingError,
        match="periodic backend failed",
    ):
        monitor.stop()
    assert monitor.running is False


def test_monitor_serializes_concurrent_sample_calls() -> None:
    activity_lock = threading.Lock()
    active_calls = 0
    max_active_calls = 0
    errors: list[Exception] = []

    def sampler(root_pid: int) -> resources.ProcessTreeSample:
        nonlocal active_calls, max_active_calls
        with activity_lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
        time.sleep(0.01)
        with activity_lock:
            active_calls -= 1
        return _tree_sample((root_pid, 10, 20))

    monitor = resources.ProcessTreeRSSMonitor(
        root_pid=100,
        interval_seconds=60.0,
        sampler=sampler,
    ).start()

    def sample_once() -> None:
        try:
            monitor.sample_now()
        except Exception as exc:  # pragma: no cover - assertion reports details
            errors.append(exc)

    threads = [threading.Thread(target=sample_once) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)
    monitor.stop()

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert max_active_calls == 1


def test_monitor_stop_race_does_not_leak_background_thread_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PausingMonitor(resources.ProcessTreeRSSMonitor):
        def __init__(self) -> None:
            self.about_to_sample = threading.Event()
            self.release_sample = threading.Event()
            super().__init__(
                root_pid=100,
                interval_seconds=0.01,
                sampler=lambda root_pid: _tree_sample((root_pid, 10, 20)),
            )

        def sample_now(self) -> resources.ProcessTreeSample:
            self.about_to_sample.set()
            if not self.release_sample.wait(timeout=1.0):
                raise TimeoutError("test did not release periodic sample")
            return super().sample_now()

    thread_errors: list[BaseException] = []
    monkeypatch.setattr(
        threading,
        "excepthook",
        lambda args: thread_errors.append(args.exc_value),
    )
    monitor = PausingMonitor().start()
    assert monitor.about_to_sample.wait(timeout=1.0)

    stop_thread = threading.Thread(target=monitor.stop)
    stop_thread.start()
    deadline = time.monotonic() + 1.0
    while monitor.running and time.monotonic() < deadline:
        time.sleep(0.001)
    assert monitor.running is False
    monitor.release_sample.set()
    stop_thread.join(timeout=1.0)

    assert stop_thread.is_alive() is False
    assert thread_errors == []


@pytest.mark.parametrize(
    "join_timeout_seconds",
    (True, 0.0, -1.0, float("nan"), float("inf")),
)
def test_monitor_rejects_invalid_join_timeout(join_timeout_seconds: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        resources.ProcessTreeRSSMonitor(
            join_timeout_seconds=join_timeout_seconds,
        )


def test_stop_fails_closed_with_bounded_join_for_stuck_sampler() -> None:
    calls = 0
    stuck_sample_entered = threading.Event()
    release_stuck_sample = threading.Event()

    def sampler(root_pid: int) -> resources.ProcessTreeSample:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _tree_sample((root_pid, 10, 20))
        stuck_sample_entered.set()
        release_stuck_sample.wait(timeout=5.0)
        return _tree_sample((root_pid, 20, 30))

    monitor = resources.ProcessTreeRSSMonitor(
        root_pid=100,
        interval_seconds=0.01,
        join_timeout_seconds=0.02,
        sampler=sampler,
    ).start()
    assert stuck_sample_entered.wait(timeout=1.0)

    started_at = time.monotonic()
    try:
        with pytest.raises(
            resources.ProcessTreeSamplingError,
            match="did not stop within",
        ):
            monitor.stop()
        assert time.monotonic() - started_at < 0.5
        assert monitor.running is False
        assert monitor.sample_count == 1
        assert monitor.latest_process_count == 1
        assert monitor.peak_process_count == 1
        with pytest.raises(
            resources.ProcessTreeSamplingError,
            match="did not stop within",
        ):
            _ = monitor.peak_aggregate_rss_bytes
    finally:
        release_stuck_sample.set()

    thread_name = f"process-tree-rss-{monitor.root_pid}"
    deadline = time.monotonic() + 1.0
    while (
        any(thread.name == thread_name for thread in threading.enumerate())
        and time.monotonic() < deadline
    ):
        time.sleep(0.001)
    assert all(thread.name != thread_name for thread in threading.enumerate())
