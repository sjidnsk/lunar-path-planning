from __future__ import annotations

import math
import os
import time
from dataclasses import asdict

import numpy as np
import pytest
import torch
from torch import nn

import lunar_exploration_ppo.ppo.collector as collector_module
from lunar_exploration_ppo.env.action_execution import ActionExecutionDiagnostics
from lunar_exploration_ppo.env.env import (
    EnvAction,
    FrontierDiagnostics,
    ProgressSnapshot,
    ResetDiagnostics,
    StepDiagnostics,
    StepResult,
)
from lunar_exploration_ppo.env.frontier import FrontierActionSet
from lunar_exploration_ppo.env.sensor_model import SensorDiagnostics
from lunar_exploration_ppo.policy.cross_attention import (
    CrossAttentionFrontierPolicy,
    PolicyBatch,
    PolicyForwardOutput,
    batch_policy_observations,
    sample_action,
)
from lunar_exploration_ppo.policy.observation import PolicyObservation
from lunar_exploration_ppo.ppo.collector import (
    CollectorContract,
    RolloutCollector,
    SpawnEnvSpec,
    SpawnVectorEnv,
    WorkerProcessError,
    lunar_env_specs,
)
from lunar_exploration_ppo.ppo.rollout import CandidateSnapshot, ROLLOUT_BATCH_SIZE
from lunar_exploration_ppo.ppo.trainer import policy_state_sha256
from lunar_exploration_ppo.utils.geometry import CellXY


CONFIG_SHA = "c" * 64
LINEAGE = {
    "stage3_commit": "4df7be92cd6517e77c648e890bf7d32c9fcd560b",
    "stage3_gate_sha256": (
        "6ea3dc5bc199dd9370fab4b1a5aac1ebab08d9e42f49c0d01b3aa2b1e9aae0f0"
    ),
}


class _FastMainProcessPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.preference = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))
        self.value_bias = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, batch: PolicyBatch) -> PolicyForwardOutput:
        assert os.getpid() == _PARENT_PID
        batch_size, candidate_count = batch.candidate_mask.shape
        logits = self.preference * batch.frontier_features[..., 0]
        theta_mu = torch.zeros(
            (batch_size, candidate_count),
            dtype=torch.float32,
            device=logits.device,
        )
        theta_kappa = torch.full_like(theta_mu, 0.5)
        value = self.value_bias.expand(batch_size)
        empty = torch.empty((batch_size, 0, 1), dtype=torch.float32, device=logits.device)
        return PolicyForwardOutput(
            frontier_logits=logits,
            theta_mu_sin_raw=theta_mu,
            theta_mu_cos_raw=torch.ones_like(theta_mu),
            theta_kappa_raw=torch.zeros_like(theta_mu),
            theta_mu=theta_mu,
            theta_kappa=theta_kappa,
            value=value,
            global_map_tokens=empty,
            local_map_tokens=empty,
            pose_token=empty,
            context_tokens=empty,
            refined_frontier_tokens=empty,
            action_hidden=empty,
        )


_PARENT_PID = os.getpid()


class _SpawnFixtureEnv:
    def __init__(self, *, worker_index: int, fail_mode: str = "none") -> None:
        self.worker_index = worker_index
        self.fail_mode = fail_mode
        self.reset_count = 0
        self.episode_step = 0
        self.total_steps = 0
        self.needs_policy = False
        self.current_action_set = self._action_set(False)
        self.current_observation: PolicyObservation | None = None
        self.last_reset_diagnostics: ResetDiagnostics | None = None

    def _observation(self, has_candidate: bool) -> PolicyObservation:
        prior = np.full((7, 4, 4), self.worker_index / 10.0, dtype=np.float32)
        coverage = np.full(
            (8, 4, 4),
            self.total_steps / 1000.0,
            dtype=np.float32,
        )
        local = np.zeros((8, 4, 4), dtype=np.float32)
        frontier = np.zeros((2, 22), dtype=np.float32)
        frontier[:, 0] = (0.0, 1.0)
        frontier[:, 14] = 0.0
        frontier[:, 15] = 1.0
        pose = np.asarray(
            (
                self.worker_index / 8.0,
                self.episode_step / 4.0,
                0.0,
                1.0,
                self.total_steps / 128.0,
                1.0,
            ),
            dtype=np.float32,
        )
        return PolicyObservation(
            prior_channels=prior,
            coverage_summary=coverage,
            local_crop=local,
            frontier_features=frontier,
            pose_features=pose,
            candidate_mask=np.asarray((has_candidate, has_candidate), dtype=bool),
        )

    def _action_set(self, has_candidate: bool) -> FrontierActionSet:
        observation = self._observation(has_candidate)
        cells = (
            (CellXY(self.worker_index + 1, 1), CellXY(self.worker_index + 1, 2))
            if has_candidate
            else ()
        )
        return FrontierActionSet(
            cells=cells,
            frontier_features=observation.frontier_features.copy(),
            candidate_mask=observation.candidate_mask.copy(),
            diagnostics={
                "selection_policy": "spawn_fixture_top_m/v1",
                "worker_index": self.worker_index,
            },
        )

    def reset(self) -> PolicyObservation:
        self.reset_count += 1
        self.episode_step = 0
        has_candidate = self.reset_count > 1
        self.needs_policy = has_candidate
        self.current_action_set = self._action_set(has_candidate)
        self.current_observation = self._observation(has_candidate)
        sensor = SensorDiagnostics(1, 91, 4, 4, 0, ("reset",), (0.0,))
        local_sensor = SensorDiagnostics(
            1,
            361,
            4,
            4,
            0,
            ("reset_local_safety",),
            (0.0,),
        )
        self.last_reset_diagnostics = ResetDiagnostics(
            reward=0.0,
            trainable=False,
            fake_logprob_created=False,
            sensor=sensor,
            local_safety_sensor=local_sensor,
            exploration_sensor=sensor,
            scan_order=(
                ("reset_exploration",)
                if self.fail_mode == "legacy_reset"
                else ("reset_local_safety", "reset_exploration")
            ),
            coverage_rate=0.0,
            progress=ProgressSnapshot(0.0, 0.0, 0.0),
            reason="none" if has_candidate else "no_candidate_done",
            frontier=FrontierDiagnostics(
                candidate_count=2 if has_candidate else 0,
                component_size=0,
                unknown_count=10,
                remaining_unobserved_coverable_count=10,
                oracle_opportunity_count=0,
            ),
        )
        return self.current_observation

    def step(self, action: EnvAction) -> StepResult:
        if self.fail_mode == "exception":
            raise RuntimeError(f"fixture worker {self.worker_index} exploded")
        if self.fail_mode == "eof":
            os._exit(17)
        if self.fail_mode == "timeout":
            time.sleep(2.0)
        if not self.needs_policy or action.candidate_index not in (0, 1):
            raise RuntimeError("invalid fixture action")
        self.episode_step += 1
        self.total_steps += 1
        done = self.episode_step == 4
        self.needs_policy = not done
        self.current_action_set = self._action_set(True)
        self.current_observation = self._observation(True)
        selected_cell = self.current_action_set.cells[action.candidate_index]
        sensor = SensorDiagnostics(
            sample_count=1,
            ray_count=3,
            cell_visit_count=4,
            unique_visible_cell_count=4,
            duplicate_cell_visits=0,
            sample_sources=("endpoint_theta",),
            sample_headings=(float(action.target_theta),),
        )
        execution = ActionExecutionDiagnostics(0, 1, 1.0, 1.0)
        reward = float(
            1.0
            if action.candidate_index == (self.worker_index + self.total_steps) % 2
            else -0.25
        )
        return StepResult(
            observation=self.current_observation,
            reward=reward,
            done=done,
            reason="stagnation_done" if done else "none",
            coverage_gain_cells=1 if reward > 0.0 else 0,
            coverage_rate=min(self.total_steps / 128.0, 1.0),
            trainable=True,
            terminal=done,
            bootstrap_value=0.0 if done else 1.0,
            diagnostics=StepDiagnostics(
                planner={
                    "valid": True,
                    "worker_index": self.worker_index,
                    "failure_reason": (
                        "unexpected_planner_reason"
                        if self.fail_mode == "unknown_reason"
                        else "none"
                    ),
                },
                sensor=sensor,
                execution=execution,
                invalid_action=False,
                safety_violation=False,
                progress=ProgressSnapshot(
                    min(self.total_steps / 128.0, 1.0),
                    self.episode_step / 4.0,
                    0.0,
                ),
                frontier=FrontierDiagnostics(2, 0, 10, 10, 0),
                planned_path_cells=(selected_cell,),
                path_length_m=1.0,
                path_observation_step_m=1.0,
                newly_observed_cell_count=1,
            ),
        )

    def export_episode_state(self) -> dict[str, object]:
        return {
            "worker_index": self.worker_index,
            "reset_count": self.reset_count,
            "episode_step": self.episode_step,
            "total_steps": self.total_steps,
            "needs_policy": self.needs_policy,
        }

    def import_episode_state(self, state: dict[str, object]) -> None:
        if state["worker_index"] != self.worker_index:
            raise ValueError("worker state mismatch")
        self.reset_count = int(state["reset_count"])
        self.episode_step = int(state["episode_step"])
        self.total_steps = int(state["total_steps"])
        self.needs_policy = bool(state["needs_policy"])
        self.current_action_set = self._action_set(self.needs_policy)
        self.current_observation = (
            self._observation(self.needs_policy) if self.reset_count else None
        )

    def export_sampler_state(self) -> dict[str, object]:
        return {"reset_count": self.reset_count}

    def import_sampler_state(self, state: dict[str, object]) -> None:
        if int(state["reset_count"]) != self.reset_count:
            raise ValueError("sampler/episode reset count mismatch")


def _make_spawn_fixture_env(
    *,
    worker_index: int,
    fail_mode: str = "none",
) -> _SpawnFixtureEnv:
    return _SpawnFixtureEnv(worker_index=worker_index, fail_mode=fail_mode)


def _specs(*, failure_worker: int | None = None, fail_mode: str = "none"):
    return tuple(
        SpawnEnvSpec(
            factory=_make_spawn_fixture_env,
            kwargs={
                "worker_index": worker_index,
                "fail_mode": fail_mode if worker_index == failure_worker else "none",
            },
        )
        for worker_index in range(8)
    )


class _LifecycleConnection:
    def __init__(
        self,
        context: _LifecycleContext,
        *,
        worker_index: int,
        endpoint: str,
    ) -> None:
        self.context = context
        self.worker_index = worker_index
        self.endpoint = endpoint
        self.close_calls = 0
        self.closed = False
        self._close_failures_remaining = int(
            endpoint == "child" and context.child_close_fail_at == worker_index
        )

    def send(self, payload: object) -> None:
        if self.endpoint != "parent":
            raise AssertionError("only the parent endpoint sends commands")
        if payload == ("close", None):
            process = self.context.process_for(self.worker_index)
            if process is not None:
                process.close_requested = True

    def poll(self, _timeout: float) -> bool:
        return True

    def recv(self) -> dict[str, object]:
        process = self.context.process_for(self.worker_index)
        if process is None:
            raise AssertionError("ready response requires a process")
        return {
            "kind": "ready",
            "worker_index": self.worker_index,
            "pid": process.pid,
            "start_method": "spawn",
        }

    def close(self) -> None:
        self.close_calls += 1
        if self._close_failures_remaining:
            self._close_failures_remaining -= 1
            raise RuntimeError(f"child-close-primary-{self.worker_index}")
        self.closed = True


class _LifecycleProcess:
    def __init__(self, context: _LifecycleContext, worker_index: int) -> None:
        self.context = context
        self.worker_index = worker_index
        self.pid = 50_000 + worker_index
        self.started = False
        self.alive = False
        self.close_requested = False
        self.join_calls = 0
        self.terminate_calls = 0
        self.kill_calls = 0
        self.process_close_calls = 0
        self.handle_closed = False

    def start(self) -> None:
        self.started = True
        self.alive = True
        if self.context.fail_start_at == self.worker_index:
            raise RuntimeError(f"start-primary-{self.worker_index}")

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        assert timeout is not None
        self.join_calls += 1
        remaining = self.context.join_failures.get(self.worker_index, 0)
        if remaining:
            self.context.join_failures[self.worker_index] = remaining - 1
            raise RuntimeError(f"join-cleanup-{self.worker_index}")
        if (
            self.close_requested
            and self.worker_index not in self.context.ignore_graceful_exit
        ):
            self.alive = False

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.worker_index in self.context.terminate_failures:
            raise RuntimeError(f"terminate-cleanup-{self.worker_index}")
        if self.worker_index not in self.context.stay_alive_after_terminate:
            self.alive = False

    def kill(self) -> None:
        self.kill_calls += 1
        if self.worker_index in self.context.kill_failures:
            raise RuntimeError(f"kill-cleanup-{self.worker_index}")
        self.alive = False

    def close(self) -> None:
        self.process_close_calls += 1
        if self.alive:
            raise RuntimeError(f"close-live-process-{self.worker_index}")
        self.handle_closed = True

    @property
    def exitcode(self) -> int | None:
        return None if self.alive or not self.started else 0


class _LifecycleContext:
    def __init__(
        self,
        *,
        fail_process_at: int | None = None,
        fail_start_at: int | None = None,
        child_close_fail_at: int | None = None,
        join_failures: dict[int, int] | None = None,
        terminate_failures: set[int] | None = None,
        kill_failures: set[int] | None = None,
        stay_alive_after_terminate: set[int] | None = None,
        ignore_graceful_exit: set[int] | None = None,
    ) -> None:
        self.fail_process_at = fail_process_at
        self.fail_start_at = fail_start_at
        self.child_close_fail_at = child_close_fail_at
        self.join_failures = dict(join_failures or {})
        self.terminate_failures = set(terminate_failures or ())
        self.kill_failures = set(kill_failures or ())
        self.stay_alive_after_terminate = set(stay_alive_after_terminate or ())
        self.ignore_graceful_exit = set(ignore_graceful_exit or ())
        self.pipes: list[tuple[_LifecycleConnection, _LifecycleConnection]] = []
        self.processes: list[_LifecycleProcess] = []

    def Pipe(
        self,
        *,
        duplex: bool,
    ) -> tuple[_LifecycleConnection, _LifecycleConnection]:
        assert duplex is True
        worker_index = len(self.pipes)
        parent = _LifecycleConnection(
            self,
            worker_index=worker_index,
            endpoint="parent",
        )
        child = _LifecycleConnection(
            self,
            worker_index=worker_index,
            endpoint="child",
        )
        self.pipes.append((parent, child))
        return parent, child

    def Process(
        self,
        *,
        target: object,
        args: tuple[object, ...],
        name: str,
        daemon: bool,
    ) -> _LifecycleProcess:
        assert callable(target)
        assert name.startswith("stage4-spawn-env-")
        assert daemon is False
        worker_index = int(args[0])
        if self.fail_process_at == worker_index:
            raise RuntimeError(f"process-primary-{worker_index}")
        process = _LifecycleProcess(self, worker_index)
        self.processes.append(process)
        return process

    def process_for(self, worker_index: int) -> _LifecycleProcess | None:
        return next(
            (
                process
                for process in self.processes
                if process.worker_index == worker_index
            ),
            None,
        )

    def force_cleanup(self) -> None:
        for parent, child in self.pipes:
            parent.closed = True
            child.closed = True
        for process in self.processes:
            process.alive = False
            process.handle_closed = True


def _reset_to_trainable(vector: SpawnVectorEnv) -> None:
    first = vector.reset()
    assert all(not value.needs_policy for value in first.values())
    second = vector.reset()
    assert all(value.needs_policy for value in second.values())


@pytest.mark.parametrize("failure_phase", ("process", "start"))
def test_spawn_constructor_owns_current_resources_before_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    context = _LifecycleContext(
        fail_process_at=2 if failure_phase == "process" else None,
        fail_start_at=2 if failure_phase == "start" else None,
        ignore_graceful_exit={0, 1, 2},
    )
    monkeypatch.setattr(collector_module.mp, "get_context", lambda method: context)

    try:
        with pytest.raises(RuntimeError, match=rf"{failure_phase}-primary-2"):
            SpawnVectorEnv(_specs(), timeout_seconds=0.1)

        assert len(context.pipes) == 3
        assert all(parent.closed and child.closed for parent, child in context.pipes)
        assert not any(process.alive for process in context.processes)
        assert all(process.handle_closed for process in context.processes)
    finally:
        context.force_cleanup()

    assert not any(process.alive for process in context.processes)


def test_spawn_constructor_retries_child_endpoint_cleanup_after_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _LifecycleContext(
        child_close_fail_at=2,
        ignore_graceful_exit={0, 1, 2},
    )
    monkeypatch.setattr(collector_module.mp, "get_context", lambda method: context)

    try:
        with pytest.raises(RuntimeError, match="child-close-primary-2"):
            SpawnVectorEnv(_specs(), timeout_seconds=0.1)

        failed_child = context.pipes[2][1]
        assert failed_child.close_calls == 2
        assert all(parent.closed and child.closed for parent, child in context.pipes)
        assert not any(process.alive for process in context.processes)
        assert all(process.handle_closed for process in context.processes)
    finally:
        context.force_cleanup()

    assert not any(process.alive for process in context.processes)


def test_spawn_constructor_cleanup_continues_kills_survivors_and_preserves_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _LifecycleContext(
        fail_start_at=3,
        join_failures={0: 2, 2: 1},
        terminate_failures={0},
        stay_alive_after_terminate={1},
        ignore_graceful_exit={0, 1, 2, 3},
    )
    monkeypatch.setattr(collector_module.mp, "get_context", lambda method: context)

    try:
        with pytest.raises(RuntimeError, match="start-primary-3"):
            SpawnVectorEnv(_specs(), timeout_seconds=0.1)

        assert len(context.processes) == 4
        assert context.processes[0].terminate_calls == 1
        assert context.processes[0].kill_calls == 1
        assert context.processes[1].terminate_calls == 1
        assert context.processes[1].kill_calls == 1
        assert context.processes[2].terminate_calls == 1
        assert all(parent.closed and child.closed for parent, child in context.pipes)
        assert not any(process.alive for process in context.processes)
        assert all(process.handle_closed for process in context.processes)
    finally:
        context.force_cleanup()

    assert not any(process.alive for process in context.processes)


def test_spawn_runtime_cleanup_errors_do_not_mask_worker_failure_or_skip_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _LifecycleContext(
        join_failures={0: 2},
        terminate_failures={0},
        stay_alive_after_terminate={1},
        ignore_graceful_exit=set(range(8)),
    )
    monkeypatch.setattr(collector_module.mp, "get_context", lambda method: context)
    vector: SpawnVectorEnv | None = None

    try:
        vector = SpawnVectorEnv(_specs(), timeout_seconds=0.1)
        with pytest.raises(WorkerProcessError, match="runtime-primary"):
            try:
                vector._fail("runtime-primary")
            finally:
                vector.close()

        assert vector.closed
        assert len(context.processes) == 8
        assert context.processes[0].kill_calls == 1
        assert context.processes[1].kill_calls == 1
        assert all(parent.closed and child.closed for parent, child in context.pipes)
        assert not any(process.alive for process in context.processes)
        assert all(process.handle_closed for process in context.processes)
    finally:
        if vector is not None:
            vector.close()
        context.force_cleanup()

    assert not any(process.alive for process in context.processes)


def test_spawn_close_reports_residual_worker_and_retries_after_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _LifecycleContext(
        kill_failures={0},
        stay_alive_after_terminate={0},
        ignore_graceful_exit={0},
    )
    monkeypatch.setattr(collector_module.mp, "get_context", lambda method: context)
    vector: SpawnVectorEnv | None = None

    try:
        vector = SpawnVectorEnv(_specs(), timeout_seconds=0.1)
        with pytest.raises(RuntimeError, match="SpawnVectorEnv cleanup incomplete") as exc:
            vector.close()

        assert isinstance(exc.value, collector_module.SpawnCleanupError)
        assert vector.closed is False
        assert vector.worker_alive == (True,) + (False,) * 7
        assert "worker=0" in str(exc.value)
        assert "pid=50000" in str(exc.value)
        assert "kill-cleanup-0" in str(exc.value)
        assert "open_process_handles=[0]" in str(exc.value)
        assert context.processes[0].process_close_calls == 0
        assert all(process.close_requested for process in context.processes)
        assert all(process.join_calls >= 1 for process in context.processes)
        assert all(parent.closed and child.closed for parent, child in context.pipes)
        assert all(process.handle_closed for process in context.processes[1:])

        context.kill_failures.clear()
        vector.close()

        assert vector.closed is True
        assert vector.worker_alive == (False,) * 8
        assert context.processes[0].kill_calls == 2
        assert context.processes[0].handle_closed is True
        assert all(process.handle_closed for process in context.processes)
    finally:
        context.kill_failures.clear()
        context.stay_alive_after_terminate.clear()
        if vector is not None and not vector.closed:
            try:
                vector.close()
            except BaseException:
                pass
        context.force_cleanup()

    assert not any(process.alive for process in context.processes)


def test_spawn_close_attaches_cleanup_diagnostics_without_masking_runtime_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _LifecycleContext(
        kill_failures={0},
        stay_alive_after_terminate={0},
        ignore_graceful_exit={0},
    )
    monkeypatch.setattr(collector_module.mp, "get_context", lambda method: context)
    vector: SpawnVectorEnv | None = None

    try:
        vector = SpawnVectorEnv(_specs(), timeout_seconds=0.1)
        with pytest.raises(WorkerProcessError) as exc:
            try:
                vector._fail("runtime-primary")
            finally:
                vector.close()

        assert type(exc.value) is WorkerProcessError
        assert str(exc.value) == "runtime-primary"
        notes = tuple(getattr(exc.value, "__notes__", ()))
        assert len(notes) == 1
        assert "SpawnVectorEnv cleanup incomplete" in notes[0]
        assert "worker=0" in notes[0]
        assert "pid=50000" in notes[0]
        assert "kill-cleanup-0" in notes[0]
        assert vector.closed is False
        assert vector.worker_alive == (True,) + (False,) * 7
        assert context.processes[0].process_close_calls == 0
        assert all(process.close_requested for process in context.processes)
        assert all(process.join_calls >= 1 for process in context.processes)
        assert all(parent.closed and child.closed for parent, child in context.pipes)

        context.kill_failures.clear()
        vector.close()
        assert vector.closed is True
        assert vector.worker_alive == (False,) * 8
    finally:
        context.kill_failures.clear()
        context.stay_alive_after_terminate.clear()
        if vector is not None and not vector.closed:
            try:
                vector.close()
            except BaseException:
                pass
        context.force_cleanup()

    assert not any(process.alive for process in context.processes)


def test_true_spawn_vector_env_has_eight_distinct_workers_state_restore_and_clean_close() -> None:
    vector = SpawnVectorEnv(_specs(), timeout_seconds=30.0)
    try:
        assert len(set(vector.worker_pids)) == 8
        assert all(pid != os.getpid() for pid in vector.worker_pids)
        assert vector.worker_start_methods == ("spawn",) * 8
        _reset_to_trainable(vector)
        before = vector.capture_states()
        vector.step(
            {
                index: EnvAction(candidate_index=index % 2, target_theta=0.0)
                for index in range(8)
            }
        )
        assert vector.capture_states() != before
        vector.restore_states(before)
        assert vector.capture_states() == before
    finally:
        vector.close()
    assert vector.closed
    assert not any(vector.worker_alive)
    assert vector.worker_exitcodes == (0,) * 8


def test_restored_spawn_current_state_continues_without_initial_reset() -> None:
    source = SpawnVectorEnv(_specs(), timeout_seconds=30.0)
    try:
        _reset_to_trainable(source)
        source.step(
            {
                index: EnvAction(candidate_index=index % 2, target_theta=0.0)
                for index in range(8)
            }
        )
        checkpoint_states = source.capture_states()
    finally:
        source.close()

    policy = _FastMainProcessPolicy()
    contract = CollectorContract(
        config_sha256=CONFIG_SHA,
        lineage=LINEAGE,
        frontier_extractor_version="spawn_fixture_frontier/v1",
        observation_schema_version="policy_observation/v1",
        action_space_version="frontier_index_continuous_theta/v1",
        reward_version="spawn_fixture_reward/v1",
        planner_version="spawn_fixture_planner/v1",
    )
    resumed = SpawnVectorEnv(_specs(), timeout_seconds=30.0)
    try:
        resumed.restore_states(checkpoint_states)
        current = resumed.current_observations()
        assert resumed.capture_states() == checkpoint_states
        assert all(value.needs_policy for value in current.values())
        assert all(
            value.observation.pose_features[1] == np.float32(0.25)
            for value in current.values()
        )

        result = RolloutCollector(
            policy=policy,
            vector_env=resumed,
            device="cpu",
            contract=contract,
        ).collect(continue_from_current_state=True)
    finally:
        resumed.close()

    first_time_step = result.batch.transitions[:8]
    assert all(
        transition.snapshot.pose_features[1] == np.float32(0.25)
        for transition in first_time_step
    )
    assert all(
        state["episode_state"]["total_steps"] == 129
        for state in result.vector_env_states
    )


@pytest.mark.parametrize("fail_mode", ("exception", "eof", "timeout"))
def test_spawn_worker_exception_eof_and_timeout_fail_closed(fail_mode: str) -> None:
    timeout = 0.1 if fail_mode == "timeout" else 30.0
    vector = SpawnVectorEnv(
        _specs(failure_worker=3, fail_mode=fail_mode),
        timeout_seconds=timeout,
    )
    try:
        _reset_to_trainable(vector)
        with pytest.raises(WorkerProcessError, match="worker 3"):
            vector.step(
                {
                    index: EnvAction(candidate_index=0, target_theta=0.0)
                    for index in range(8)
                }
            )
    finally:
        vector.close()
    assert vector.closed
    assert not any(vector.worker_alive)


def test_collector_counts_only_trainable_rows_and_infers_only_in_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.ppo.collector as collector_module

    torch.manual_seed(11)
    policy = _FastMainProcessPolicy()
    expected_policy_hash = policy_state_sha256(policy)
    hash_call_count = 0
    original_policy_hash = collector_module.policy_state_sha256

    def _counted_policy_hash(module: nn.Module) -> str:
        nonlocal hash_call_count
        hash_call_count += 1
        return original_policy_hash(module)

    monkeypatch.setattr(
        collector_module,
        "policy_state_sha256",
        _counted_policy_hash,
    )
    contract = CollectorContract(
        config_sha256=CONFIG_SHA,
        lineage=LINEAGE,
        frontier_extractor_version="spawn_fixture_frontier/v1",
        observation_schema_version="policy_observation/v1",
        action_space_version="frontier_index_continuous_theta/v1",
        reward_version="spawn_fixture_reward/v1",
        planner_version="spawn_fixture_planner/v1",
    )
    vector = SpawnVectorEnv(_specs(), timeout_seconds=30.0)
    try:
        result = RolloutCollector(
            policy=policy,
            vector_env=vector,
            device="cpu",
            contract=contract,
        ).collect()
    finally:
        vector.close()

    assert result.batch.size == ROLLOUT_BATCH_SIZE
    assert result.batch.layout_shape == (128, 8)
    assert result.audit.trainable_transition_count == 1024
    assert result.audit.per_env_trainable_counts == (128,) * 8
    assert result.audit.diagnostic_reset_counts == (1,) * 8
    assert result.audit.worker_pids == vector.worker_pids
    assert result.audit.worker_start_methods == ("spawn",) * 8
    assert result.audit.inference_pids == (os.getpid(),)
    assert result.audit.policy_device == "cpu"
    assert result.audit.policy_state_sha256 == expected_policy_hash
    assert hash_call_count == 2
    assert policy_state_sha256(policy) == expected_policy_hash
    assert len(result.audit.snapshot_sha256) == 1024
    assert set(result.audit.snapshot_sha256) == set(result.batch.snapshot_hashes)
    assert result.audit.planner_failure_counts == {
        "endpoint_physical_unsafe": 0,
        "endpoint_unknown_buffer_unsafe": 0,
        "path_physical_unsafe": 0,
        "path_unknown_buffer_unsafe": 0,
        "planner_no_path": 0,
    }
    assert result.audit.reset_diagnostics
    assert {
        tuple(row["scan_order"])
        for row in result.audit.reset_diagnostics
    } == {("reset_local_safety", "reset_exploration")}
    assert {
        tuple(row["local_safety_sensor"]["sample_sources"])
        for row in result.audit.reset_diagnostics
    } == {("reset_local_safety",)}
    assert {
        tuple(row["exploration_sensor"]["sample_sources"])
        for row in result.audit.reset_diagnostics
    } == {("reset",)}
    assert len(result.vector_env_states) == 8
    assert all(
        set(state) == {"episode_state", "sampler_state"}
        for state in result.vector_env_states
    )
    assert np.isfinite(result.batch.raw_advantages).all()
    assert np.isfinite(result.batch.returns).all()
    assert math.isclose(
        float(result.batch.normalized_advantages.mean()),
        0.0,
        abs_tol=1.0e-5,
    )
    assert math.isclose(
        float(result.batch.normalized_advantages.std(ddof=0)),
        1.0,
        abs_tol=1.0e-5,
    )
    assert all(
        transition.old_log_prob_total.tobytes()
        == np.float32(
            transition.old_log_prob_frontier + transition.old_log_prob_theta
        ).tobytes()
        for transition in result.batch.transitions
    )


def test_planning_warm_collector_rejects_legacy_single_scan_reset() -> None:
    policy = _FastMainProcessPolicy()
    contract = CollectorContract(
        config_sha256=CONFIG_SHA,
        lineage=LINEAGE,
        frontier_extractor_version="spawn_fixture_frontier/v1",
        observation_schema_version="policy_observation/v1",
        action_space_version="frontier_index_continuous_theta/v1",
        reward_version="spawn_fixture_reward/v1",
        planner_version="spawn_fixture_planner/v1",
    )
    vector = SpawnVectorEnv(
        _specs(failure_worker=0, fail_mode="legacy_reset"),
        timeout_seconds=30.0,
    )
    try:
        with pytest.raises(
            collector_module.CollectorError,
            match="reset diagnostics",
        ):
            RolloutCollector(
                policy=policy,
                vector_env=vector,
                device="cpu",
                contract=contract,
                require_dual_scan=True,
            ).preflight_one_step()
    finally:
        vector.close()


def test_collector_one_step_preflight_covers_all_workers_without_changing_collect_contract() -> None:
    torch.manual_seed(23)
    policy = _FastMainProcessPolicy()
    expected_policy_hash = policy_state_sha256(policy)
    contract = CollectorContract(
        config_sha256=CONFIG_SHA,
        lineage=LINEAGE,
        frontier_extractor_version="spawn_fixture_frontier/v1",
        observation_schema_version="policy_observation/v1",
        action_space_version="frontier_index_continuous_theta/v1",
        reward_version="spawn_fixture_reward/v1",
        planner_version="spawn_fixture_planner/v1",
    )
    vector = SpawnVectorEnv(_specs(), timeout_seconds=30.0)
    try:
        result = RolloutCollector(
            policy=policy,
            vector_env=vector,
            device="cpu",
            contract=contract,
        ).preflight_one_step()
    finally:
        vector.close()

    assert result.schema_version == "stage6_collector_one_step_preflight/v1"
    assert len(result.transitions) == 8
    assert result.trainable_transition_count == 8
    assert all(transition.done in {True, False} for transition in result.transitions)
    assert result.worker_pids == vector.worker_pids
    assert len(set(result.worker_pids)) == 8
    assert result.worker_start_methods == ("spawn",) * 8
    assert result.inference_pids == (os.getpid(),)
    assert result.inference_batch_count == 1
    assert result.policy_state_sha256 == expected_policy_hash
    assert result.diagnostic_reset_counts == (1,) * 8
    assert result.planner_failure_counts == {
        "endpoint_physical_unsafe": 0,
        "endpoint_unknown_buffer_unsafe": 0,
        "path_physical_unsafe": 0,
        "path_unknown_buffer_unsafe": 0,
        "planner_no_path": 0,
    }
    assert len(result.reset_diagnostics) == 16
    assert len(set(result.snapshot_sha256)) == 8
    assert set(result.timings_seconds) == {"reset", "inference", "step"}
    assert all(value >= 0.0 for value in result.timings_seconds.values())


def test_collector_rejects_unknown_planner_failure_reason() -> None:
    policy = _FastMainProcessPolicy()
    contract = CollectorContract(
        config_sha256=CONFIG_SHA,
        lineage=LINEAGE,
        frontier_extractor_version="spawn_fixture_frontier/v1",
        observation_schema_version="policy_observation/v1",
        action_space_version="frontier_index_continuous_theta/v1",
        reward_version="spawn_fixture_reward/v1",
        planner_version="spawn_fixture_planner/v1",
    )
    vector = SpawnVectorEnv(
        _specs(failure_worker=0, fail_mode="unknown_reason"),
        timeout_seconds=30.0,
    )
    try:
        with pytest.raises(
            collector_module.CollectorError,
            match="planner failure reason",
        ):
            RolloutCollector(
                policy=policy,
                vector_env=vector,
                device="cpu",
                contract=contract,
            ).preflight_one_step()
    finally:
        vector.close()
    assert vector.closed
    assert not any(vector.worker_alive)


def test_collector_inference_audit_is_scoped_to_each_call_and_recovers_after_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(29)
    policy = _FastMainProcessPolicy()
    contract = CollectorContract(
        config_sha256=CONFIG_SHA,
        lineage=LINEAGE,
        frontier_extractor_version="spawn_fixture_frontier/v1",
        observation_schema_version="policy_observation/v1",
        action_space_version="frontier_index_continuous_theta/v1",
        reward_version="spawn_fixture_reward/v1",
        planner_version="spawn_fixture_planner/v1",
    )
    vector = SpawnVectorEnv(_specs(), timeout_seconds=30.0)
    collector = RolloutCollector(
        policy=policy,
        vector_env=vector,
        device="cpu",
        contract=contract,
    )
    try:
        first = collector.preflight_one_step()
        second = collector.preflight_one_step()
        assert first.inference_batch_count == 1
        assert second.inference_batch_count == 1
        assert first.inference_pids == second.inference_pids == (os.getpid(),)

        original_forward = collector._forward
        fail_once = True

        def _forward_then_fail(observations):
            nonlocal fail_once
            output = original_forward(observations)
            if fail_once:
                fail_once = False
                raise RuntimeError("injected post-inference failure")
            return output

        monkeypatch.setattr(collector, "_forward", _forward_then_fail)
        with pytest.raises(RuntimeError, match="post-inference failure"):
            collector.preflight_one_step()
        recovered = collector.preflight_one_step()
        assert recovered.inference_batch_count == 1
        assert recovered.inference_pids == (os.getpid(),)

        collected = collector.collect()
        assert collected.audit.inference_batch_count == 128
        assert collected.audit.inference_pids == (os.getpid(),)
    finally:
        vector.close()


def test_collector_resource_guard_aborts_before_mid_rollout_worker_step() -> None:
    policy = _FastMainProcessPolicy()
    contract = CollectorContract(
        config_sha256=CONFIG_SHA,
        lineage=LINEAGE,
        frontier_extractor_version="spawn_fixture_frontier/v1",
        observation_schema_version="policy_observation/v1",
        action_space_version="frontier_index_continuous_theta/v1",
        reward_version="spawn_fixture_reward/v1",
        planner_version="spawn_fixture_planner/v1",
    )
    vector = SpawnVectorEnv(_specs(), timeout_seconds=30.0)
    collector = RolloutCollector(
        policy=policy,
        vector_env=vector,
        device="cpu",
        contract=contract,
    )
    boundaries: list[str] = []

    def resource_guard(boundary: str) -> None:
        boundaries.append(boundary)
        if boundary == "rollout:before-step":
            raise RuntimeError("latched rollout hard stop")

    try:
        with pytest.raises(RuntimeError, match="latched rollout hard stop"):
            collector.collect(resource_guard=resource_guard)
    finally:
        vector.close()

    assert boundaries[-1] == "rollout:before-step"
    assert "rollout:after-inference" in boundaries


def test_real_lunar_eight_spawn_cuda_policy_single_step_preflight() -> None:
    from pathlib import Path

    from lunar_exploration_ppo.configs.stage1 import load_stage1_config

    root = Path(__file__).resolve().parents[2]
    config = load_stage1_config(root / "configs/ppo_highres_frontier_smoke_v1.json")
    policy = CrossAttentionFrontierPolicy().to(device="cuda", dtype=torch.float32)
    vector = SpawnVectorEnv(
        lunar_env_specs(config.model_dump(mode="json")),
        timeout_seconds=120.0,
    )
    try:
        observations = vector.reset()
        assert len(observations) == 8
        assert all(value.needs_policy for value in observations.values())
        batch = batch_policy_observations(
            tuple(observations[index].observation for index in range(8)),
            device=torch.device("cuda"),
        )
        policy.eval()
        with torch.no_grad():
            output = policy(batch)
            sampled = sample_action(
                output,
                batch.candidate_mask,
                deterministic=True,
            )
        steps = vector.step(
            {
                index: EnvAction(
                    candidate_index=int(
                        sampled.selected_frontier_index[index].item()
                    ),
                    target_theta=float(sampled.selected_theta[index].item()),
                )
                for index in range(8)
            }
        )
        assert all(step.result.trainable for step in steps.values())
        assert all(step.result.done == step.result.terminal for step in steps.values())
        contract = CollectorContract(
            config_sha256=CONFIG_SHA,
            lineage=LINEAGE,
            frontier_extractor_version="stage2_observed_frontier_top_m/v1",
            observation_schema_version="policy_observation/v1",
            action_space_version="frontier_index_continuous_theta/v1",
            reward_version="stage1_coverage_gain_reward/v1",
            planner_version="stage1_path_planner_adapter/v1",
        )
        collector = RolloutCollector(
            policy=policy,
            vector_env=vector,
            device="cuda",
            contract=contract,
        )
        first = observations[0]
        snapshot = CandidateSnapshot.capture(
            observation=first.observation,
            frontier_cells=first.frontier_cells,
            top_m_selection_summary=first.top_m_selection_summary,
        )
        transition = collector._transition_from_step(
            snapshot=snapshot,
            policy_hash=policy_state_sha256(policy),
            selected_index=int(sampled.selected_frontier_index[0].item()),
            selected_theta=float(sampled.selected_theta[0].item()),
            old_log_prob_frontier=float(sampled.log_prob_frontier[0].item()),
            old_log_prob_theta=float(sampled.log_prob_theta[0].item()),
            old_log_prob_total=float(sampled.log_prob_total[0].item()),
            old_value=float(output.value[0].item()),
            result=steps[0].result,
        )
        assert transition.planned_path_cells == steps[0].result.diagnostics.planned_path_cells
        assert transition.candidate_snapshot_sha256 == snapshot.sha256
        states = vector.capture_states()
        assert len(states) == 8
        vector.restore_states(states)
        assert vector.capture_states() == states
        assert vector.worker_start_methods == ("spawn",) * 8
        assert len(set(vector.worker_pids)) == 8
    finally:
        vector.close()
    assert vector.closed
    assert not any(vector.worker_alive)
