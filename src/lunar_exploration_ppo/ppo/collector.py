"""Stage 4 真正 spawn 的 8-env rollout collector。"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import pickle
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import torch
from torch import nn

from lunar_exploration_ppo.env.env import EnvAction, StepResult
from lunar_exploration_ppo.policy.cross_attention import (
    PolicyForwardOutput,
    batch_policy_observations,
    sample_action,
)
from lunar_exploration_ppo.policy.observation import PolicyObservation
from lunar_exploration_ppo.ppo.rollout import (
    ROLLOUT_ENV_COUNT,
    ROLLOUT_STEPS_PER_ENV,
    CandidateSnapshot,
    RolloutBatch,
    RolloutBuffer,
    RolloutContractError,
    RolloutTransition,
)
from lunar_exploration_ppo.ppo.trainer import policy_state_sha256
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.geometry import CellXY


COLLECTOR_SCHEMA_VERSION: Final = "stage4_spawn_collector/v1"
VECTOR_STATE_SCHEMA_VERSION: Final = "stage4_spawn_vector_worker_state/v1"
_MAX_DIAGNOSTIC_RESETS_PER_ENV: Final = 1024


class WorkerProcessError(RuntimeError):
    """Spawn worker exception、EOF、timeout 或协议漂移。"""


class CollectorError(RuntimeError):
    """Rollout collection 未满足固定 Stage 4 合同。"""


@dataclass(frozen=True, slots=True)
class SpawnEnvSpec:
    """可 pickle 的 top-level factory 与 JSON kwargs。"""

    factory: Callable[..., object]
    kwargs: Mapping[str, object]

    def __post_init__(self) -> None:
        if not callable(self.factory):
            raise ValueError("spawn env factory must be callable")
        try:
            kwargs = json.loads(
                json.dumps(
                    self.kwargs,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("spawn env kwargs must be finite JSON") from exc
        if not isinstance(kwargs, dict):
            raise ValueError("spawn env kwargs must be a JSON object")
        object.__setattr__(self, "kwargs", MappingProxyType(kwargs))
        try:
            pickle.dumps(self)
        except Exception as exc:
            raise ValueError("spawn env spec must be serializable") from exc

    def __reduce__(self) -> tuple[object, tuple[object, dict[str, object]]]:
        return type(self), (self.factory, dict(self.kwargs))


@dataclass(frozen=True, slots=True)
class WorkerObservation:
    observation: PolicyObservation
    frontier_cells: tuple[CellXY, ...]
    top_m_selection_summary: Mapping[str, object]
    needs_policy: bool
    reset_trainable: bool = False
    fake_logprob_created: bool = False


@dataclass(frozen=True, slots=True)
class WorkerStep:
    result: StepResult
    next_observation: WorkerObservation


@dataclass(frozen=True, slots=True)
class CollectorContract:
    config_sha256: str
    lineage: Mapping[str, object]
    frontier_extractor_version: str
    observation_schema_version: str
    action_space_version: str
    reward_version: str
    planner_version: str

    def __post_init__(self) -> None:
        if not _is_sha256(self.config_sha256):
            raise ValueError("collector config hash is invalid")
        lineage = _copy_json_mapping(self.lineage, "collector lineage")
        object.__setattr__(self, "lineage", MappingProxyType(lineage))
        for name in (
            "frontier_extractor_version",
            "observation_schema_version",
            "action_space_version",
            "reward_version",
            "planner_version",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"collector {name} is invalid")


@dataclass(frozen=True, slots=True)
class CollectorAudit:
    schema_version: str
    worker_pids: tuple[int, ...]
    worker_start_methods: tuple[str, ...]
    per_env_trainable_counts: tuple[int, ...]
    diagnostic_reset_counts: tuple[int, ...]
    trainable_transition_count: int
    inference_pids: tuple[int, ...]
    inference_batch_count: int
    policy_device: str
    policy_state_sha256: str
    snapshot_sha256: tuple[str, ...]
    terminal_transition_count: int


@dataclass(frozen=True, slots=True)
class CollectionResult:
    batch: RolloutBatch
    audit: CollectorAudit
    vector_env_states: tuple[dict[str, object], ...]


class SpawnVectorEnv:
    """八个显式 spawn worker；环境、planner、sensor 全在 child。"""

    def __init__(
        self,
        specs: Sequence[SpawnEnvSpec],
        *,
        timeout_seconds: float = 60.0,
    ) -> None:
        if len(specs) != ROLLOUT_ENV_COUNT or any(
            not isinstance(spec, SpawnEnvSpec) for spec in specs
        ):
            raise ValueError("Stage 4 vector env requires exactly 8 spawn specs")
        if (
            not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0.0
        ):
            raise ValueError("worker timeout must be finite and positive")
        self.timeout_seconds = float(timeout_seconds)
        self._context = mp.get_context("spawn")
        self._connections: list[Connection] = []
        self._processes: list[mp.Process] = []
        self._worker_pids: tuple[int, ...] = ()
        self._worker_start_methods: tuple[str, ...] = ()
        self._closed = False
        self._broken = False
        try:
            for worker_index, spec in enumerate(specs):
                parent, child = self._context.Pipe(duplex=True)
                process = self._context.Process(
                    target=_worker_main,
                    args=(worker_index, child, spec),
                    name=f"stage4-spawn-env-{worker_index}",
                    daemon=False,
                )
                process.start()
                child.close()
                self._connections.append(parent)
                self._processes.append(process)
            ready = tuple(
                self._receive(
                    worker_index,
                    timeout_seconds=max(30.0, self.timeout_seconds),
                )
                for worker_index in range(ROLLOUT_ENV_COUNT)
            )
            if any(
                not isinstance(value, dict)
                or set(value) != {"kind", "worker_index", "pid", "start_method"}
                or value["kind"] != "ready"
                or value["worker_index"] != index
                or value["start_method"] != "spawn"
                for index, value in enumerate(ready)
            ):
                raise WorkerProcessError("spawn worker ready protocol mismatch")
            self._worker_pids = tuple(int(value["pid"]) for value in ready)
            self._worker_start_methods = tuple(
                str(value["start_method"]) for value in ready
            )
            if len(set(self._worker_pids)) != ROLLOUT_ENV_COUNT:
                raise WorkerProcessError("spawn workers do not have distinct PIDs")
        except BaseException:
            self.close()
            raise

    @property
    def worker_pids(self) -> tuple[int, ...]:
        return self._worker_pids

    @property
    def worker_start_methods(self) -> tuple[str, ...]:
        return self._worker_start_methods

    @property
    def worker_alive(self) -> tuple[bool, ...]:
        return tuple(process.is_alive() for process in self._processes)

    @property
    def worker_exitcodes(self) -> tuple[int | None, ...]:
        return tuple(process.exitcode for process in self._processes)

    @property
    def closed(self) -> bool:
        return self._closed

    def reset(
        self,
        worker_indices: Sequence[int] | None = None,
    ) -> dict[int, WorkerObservation]:
        indices = self._validated_indices(worker_indices)
        values = self._round_trip(
            {index: ("reset", None) for index in indices}
        )
        if any(not isinstance(value, WorkerObservation) for value in values.values()):
            self._fail("worker reset response type mismatch")
        return values  # type: ignore[return-value]

    def step(self, actions: Mapping[int, EnvAction]) -> dict[int, WorkerStep]:
        if not isinstance(actions, Mapping) or not actions:
            raise ValueError("vector step requires worker actions")
        indices = self._validated_indices(tuple(actions))
        if any(not isinstance(actions[index], EnvAction) for index in indices):
            raise ValueError("vector step actions must be EnvAction values")
        values = self._round_trip(
            {index: ("step", actions[index]) for index in indices}
        )
        if any(not isinstance(value, WorkerStep) for value in values.values()):
            self._fail("worker step response type mismatch")
        return values  # type: ignore[return-value]

    def current_observations(self) -> dict[int, WorkerObservation]:
        """读取已恢复 worker 的当前 policy observation，不触发 reset。"""

        values = self._round_trip(
            {
                index: ("current_observation", None)
                for index in range(ROLLOUT_ENV_COUNT)
            }
        )
        if any(not isinstance(value, WorkerObservation) for value in values.values()):
            self._fail("worker current observation response type mismatch")
        return values  # type: ignore[return-value]

    def capture_states(self) -> tuple[dict[str, object], ...]:
        values = self._round_trip(
            {index: ("capture_state", None) for index in range(ROLLOUT_ENV_COUNT)}
        )
        ordered: list[dict[str, object]] = []
        for index in range(ROLLOUT_ENV_COUNT):
            value = values[index]
            if not isinstance(value, dict) or set(value) != {
                "episode_state",
                "sampler_state",
            }:
                self._fail(f"worker {index} state response mismatch")
            ordered.append(_copy_json_mapping(value, "worker vector state"))
        return tuple(ordered)

    def restore_states(
        self,
        states: Sequence[Mapping[str, object]],
    ) -> None:
        if len(states) != ROLLOUT_ENV_COUNT:
            raise ValueError("vector restore requires exactly 8 worker states")
        payloads = {
            index: (
                "restore_state",
                _copy_json_mapping(states[index], "worker vector state"),
            )
            for index in range(ROLLOUT_ENV_COUNT)
        }
        values = self._round_trip(payloads)
        if any(value != "restored" for value in values.values()):
            self._fail("worker restore acknowledgement mismatch")

    def close(self) -> None:
        if self._closed:
            return
        for connection, process in zip(
            self._connections,
            self._processes,
            strict=True,
        ):
            if process.is_alive():
                try:
                    connection.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for connection, process in zip(
            self._connections,
            self._processes,
            strict=True,
        ):
            if process.is_alive():
                try:
                    if connection.poll(min(max(self.timeout_seconds, 0.1), 1.0)):
                        connection.recv()
                except (BrokenPipeError, EOFError, OSError):
                    pass
                process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            try:
                connection.close()
            except OSError:
                pass
        self._closed = True

    def __enter__(self) -> "SpawnVectorEnv":
        if self._closed:
            raise WorkerProcessError("vector env is already closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _round_trip(
        self,
        commands: Mapping[int, tuple[str, object]],
    ) -> dict[int, object]:
        if self._closed or self._broken:
            raise WorkerProcessError("vector env is closed or broken")
        for worker_index, command in commands.items():
            try:
                self._connections[worker_index].send(command)
            except (BrokenPipeError, EOFError, OSError) as exc:
                self._fail(
                    f"worker {worker_index} command send failed",
                    cause=exc,
                )
        values: dict[int, object] = {}
        for worker_index in commands:
            values[worker_index] = self._receive(
                worker_index,
                timeout_seconds=self.timeout_seconds,
            )
        return values

    def _receive(self, worker_index: int, *, timeout_seconds: float) -> object:
        connection = self._connections[worker_index]
        process = self._processes[worker_index]
        try:
            if not connection.poll(timeout_seconds):
                self._fail(
                    f"worker {worker_index} timed out after {timeout_seconds:g}s"
                )
            response = connection.recv()
        except EOFError as exc:
            self._fail(
                f"worker {worker_index} reached EOF (exitcode={process.exitcode})",
                cause=exc,
            )
        except OSError as exc:
            self._fail(f"worker {worker_index} receive failed", cause=exc)
        if (
            isinstance(response, dict)
            and response.get("kind") == "remote_error"
        ):
            self._fail(
                f"worker {worker_index} remote exception: "
                f"{response.get('error_type')}: {response.get('message')}\n"
                f"{response.get('traceback')}"
            )
        return response

    def _validated_indices(
        self,
        values: Sequence[int] | None,
    ) -> tuple[int, ...]:
        indices = (
            tuple(range(ROLLOUT_ENV_COUNT))
            if values is None
            else tuple(values)
        )
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(
                type(index) is not int
                or not 0 <= index < ROLLOUT_ENV_COUNT
                for index in indices
            )
        ):
            raise ValueError("worker index set is invalid")
        return indices

    def _fail(
        self,
        message: str,
        *,
        cause: BaseException | None = None,
    ) -> None:
        self._broken = True
        if cause is None:
            raise WorkerProcessError(message)
        raise WorkerProcessError(message) from cause


class RolloutCollector:
    """主进程批量 policy inference，worker 只执行环境语义。"""

    def __init__(
        self,
        *,
        policy: nn.Module,
        vector_env: SpawnVectorEnv,
        device: torch.device | str,
        contract: CollectorContract,
    ) -> None:
        if not isinstance(policy, nn.Module):
            raise CollectorError("collector policy must be a torch module")
        if not isinstance(vector_env, SpawnVectorEnv):
            raise CollectorError("collector requires SpawnVectorEnv")
        if not isinstance(contract, CollectorContract):
            raise CollectorError("collector contract is invalid")
        self.policy = policy
        self.vector_env = vector_env
        self.device = torch.device(device)
        self.contract = contract
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise CollectorError("CUDA was requested but is unavailable; no CPU fallback")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        try:
            model_device = next(policy.parameters()).device
        except StopIteration as exc:
            raise CollectorError("collector policy has no trainable parameters") from exc
        if model_device != self.device:
            raise CollectorError("collector policy and requested device differ")
        if any(parameter.dtype != torch.float32 for parameter in policy.parameters()):
            raise CollectorError("collector policy parameters must be FP32")
        self._inference_pids: set[int] = set()
        self._inference_batch_count = 0

    def collect(
        self,
        *,
        continue_from_current_state: bool = False,
    ) -> CollectionResult:
        if type(continue_from_current_state) is not bool:
            raise CollectorError("continue_from_current_state must be boolean")
        self.policy.eval()
        initial_policy_hash = policy_state_sha256(self.policy)
        buffer = RolloutBuffer()
        counts = [0] * ROLLOUT_ENV_COUNT
        diagnostic_resets = [0] * ROLLOUT_ENV_COUNT
        current: dict[int, WorkerObservation] = {}
        if continue_from_current_state:
            current.update(self.vector_env.current_observations())
        else:
            first_resets = self.vector_env.reset()
            for env_index, envelope in first_resets.items():
                current[env_index] = self._accept_reset(
                    env_index,
                    envelope,
                    diagnostic_resets,
                )
        pending = [
            index
            for index, envelope in current.items()
            if not envelope.needs_policy
        ]
        self._reset_until_trainable(current, pending, diagnostic_resets)

        last_done = [False] * ROLLOUT_ENV_COUNT
        snapshot_hashes: list[str] = []
        terminal_count = 0
        while any(count < ROLLOUT_STEPS_PER_ENV for count in counts):
            active = tuple(
                index
                for index, count in enumerate(counts)
                if count < ROLLOUT_STEPS_PER_ENV
            )
            if any(not current[index].needs_policy for index in active):
                raise CollectorError("active env lacks a trainable policy observation")
            snapshots = tuple(
                CandidateSnapshot.capture(
                    observation=current[index].observation,
                    frontier_cells=current[index].frontier_cells,
                    top_m_selection_summary=current[
                        index
                    ].top_m_selection_summary,
                )
                for index in active
            )
            output, action_sample = self._infer_actions(
                tuple(current[index].observation for index in active)
            )
            actions = {
                env_index: EnvAction(
                    candidate_index=int(
                        action_sample.selected_frontier_index[row].item()
                    ),
                    target_theta=float(action_sample.selected_theta[row].item()),
                )
                for row, env_index in enumerate(active)
            }
            step_values = self.vector_env.step(actions)
            terminal_to_reset: list[int] = []
            for row, env_index in enumerate(active):
                worker_step = step_values[env_index]
                result = worker_step.result
                if not result.trainable:
                    raise CollectorError("sampled action produced non-trainable transition")
                if result.done != result.terminal:
                    raise CollectorError("Stage 4 treats every done as terminal")
                if result.done and result.bootstrap_value != 0.0:
                    raise CollectorError("terminal transition attempted to bootstrap")
                if not result.done and not worker_step.next_observation.needs_policy:
                    raise CollectorError(
                        "next-empty sampled transition was not closed as terminal"
                    )
                transition = self._transition_from_step(
                    snapshot=snapshots[row],
                    policy_hash=initial_policy_hash,
                    selected_index=actions[env_index].candidate_index,
                    selected_theta=actions[env_index].target_theta,
                    old_log_prob_frontier=float(
                        action_sample.log_prob_frontier[row].item()
                    ),
                    old_log_prob_theta=float(
                        action_sample.log_prob_theta[row].item()
                    ),
                    old_log_prob_total=float(
                        action_sample.log_prob_total[row].item()
                    ),
                    old_value=float(output.value[row].item()),
                    result=result,
                )
                buffer.add(env_index, transition)
                counts[env_index] += 1
                snapshot_hashes.append(transition.candidate_snapshot_sha256)
                current[env_index] = worker_step.next_observation
                last_done[env_index] = result.done
                if result.done:
                    terminal_count += 1
                    if counts[env_index] < ROLLOUT_STEPS_PER_ENV:
                        terminal_to_reset.append(env_index)
            self._reset_until_trainable(
                current,
                terminal_to_reset,
                diagnostic_resets,
            )

        last_values = np.zeros((ROLLOUT_ENV_COUNT,), dtype=np.float32)
        nonterminal = tuple(
            index for index, done in enumerate(last_done) if not done
        )
        if nonterminal:
            output = self._forward(
                tuple(current[index].observation for index in nonterminal)
            )
            last_values[np.asarray(nonterminal, dtype=np.int64)] = (
                output.value.detach().cpu().numpy().astype(np.float32, copy=False)
            )
        vector_states = self.vector_env.capture_states()
        if policy_state_sha256(self.policy) != initial_policy_hash:
            raise CollectorError("policy changed during rollout collection")
        try:
            batch = buffer.finalize(last_values)
        except RolloutContractError as exc:
            raise CollectorError("collected rollout failed buffer finalization") from exc
        audit = CollectorAudit(
            schema_version=COLLECTOR_SCHEMA_VERSION,
            worker_pids=self.vector_env.worker_pids,
            worker_start_methods=self.vector_env.worker_start_methods,
            per_env_trainable_counts=tuple(counts),
            diagnostic_reset_counts=tuple(diagnostic_resets),
            trainable_transition_count=sum(counts),
            inference_pids=tuple(sorted(self._inference_pids)),
            inference_batch_count=self._inference_batch_count,
            policy_device=str(self.device),
            policy_state_sha256=initial_policy_hash,
            snapshot_sha256=tuple(snapshot_hashes),
            terminal_transition_count=terminal_count,
        )
        return CollectionResult(
            batch=batch,
            audit=audit,
            vector_env_states=vector_states,
        )

    def _accept_reset(
        self,
        env_index: int,
        envelope: WorkerObservation,
        diagnostic_resets: list[int],
    ) -> WorkerObservation:
        if envelope.reset_trainable or envelope.fake_logprob_created:
            raise CollectorError(
                f"worker {env_index} reset created trainable/fake policy data"
            )
        if not envelope.needs_policy:
            diagnostic_resets[env_index] += 1
            if diagnostic_resets[env_index] > _MAX_DIAGNOSTIC_RESETS_PER_ENV:
                raise CollectorError("diagnostic reset retry limit exceeded")
        return envelope

    def _reset_until_trainable(
        self,
        current: dict[int, WorkerObservation],
        indices: Sequence[int],
        diagnostic_resets: list[int],
    ) -> None:
        pending = tuple(indices)
        while pending:
            reset_values = self.vector_env.reset(pending)
            next_pending: list[int] = []
            for env_index in pending:
                envelope = self._accept_reset(
                    env_index,
                    reset_values[env_index],
                    diagnostic_resets,
                )
                current[env_index] = envelope
                if not envelope.needs_policy:
                    next_pending.append(env_index)
            pending = tuple(next_pending)

    def _infer_actions(
        self,
        observations: tuple[PolicyObservation, ...],
    ) -> tuple[PolicyForwardOutput, Any]:
        policy_batch = batch_policy_observations(observations, device=self.device)
        output = self._forward_batch(policy_batch)
        with torch.no_grad():
            action = sample_action(
                output,
                policy_batch.candidate_mask,
                deterministic=False,
            )
        return output, action

    def _forward(
        self,
        observations: tuple[PolicyObservation, ...],
    ) -> PolicyForwardOutput:
        policy_batch = batch_policy_observations(observations, device=self.device)
        return self._forward_batch(policy_batch)

    def _forward_batch(self, policy_batch: Any) -> PolicyForwardOutput:
        if os.getpid() in self.vector_env.worker_pids:
            raise CollectorError("policy inference attempted inside a worker")
        with torch.no_grad():
            output = self.policy(policy_batch)
        if not isinstance(output, PolicyForwardOutput):
            raise CollectorError("policy returned an invalid forward output")
        self._inference_pids.add(os.getpid())
        self._inference_batch_count += 1
        return output

    def _transition_from_step(
        self,
        *,
        snapshot: CandidateSnapshot,
        policy_hash: str,
        selected_index: int,
        selected_theta: float,
        old_log_prob_frontier: float,
        old_log_prob_theta: float,
        old_log_prob_total: float,
        old_value: float,
        result: StepResult,
    ) -> RolloutTransition:
        valid_indices = tuple(
            int(value) for value in np.flatnonzero(snapshot.candidate_mask)
        )
        try:
            selected_rank = valid_indices.index(selected_index)
        except ValueError as exc:
            raise CollectorError("sampled frontier index is not valid in snapshot") from exc
        selected_cell = snapshot.frontier_cells[selected_rank]
        diagnostics = result.diagnostics
        sensor = diagnostics.sensor
        execution = diagnostics.execution
        if sensor is None:
            observation_sample_count = 0
            ray_count_per_sample = 0
            ray_cell_visit_count = 0
            sensor_summary: dict[str, object] = {}
        else:
            observation_sample_count = sensor.sample_count
            if (
                observation_sample_count <= 0
                or sensor.ray_count % observation_sample_count != 0
            ):
                raise CollectorError("sensor ray count is not per-sample auditable")
            ray_count_per_sample = sensor.ray_count // observation_sample_count
            ray_cell_visit_count = sensor.cell_visit_count
            sensor_summary = asdict(sensor)
        execution_summary = asdict(execution) if execution is not None else {}
        execution_summary["sensor"] = sensor_summary
        path_length = float(diagnostics.path_length_m)
        gain_per_meter = (
            result.coverage_gain_cells / path_length if path_length > 0.0 else 0.0
        )
        return RolloutTransition.create(
            snapshot=snapshot,
            policy_state_sha256=policy_hash,
            config_sha256=self.contract.config_sha256,
            lineage=self.contract.lineage,
            selected_frontier_index=selected_index,
            selected_cell_xy=selected_cell,
            selected_theta=selected_theta,
            old_log_prob_frontier=old_log_prob_frontier,
            old_log_prob_theta=old_log_prob_theta,
            old_log_prob_total=old_log_prob_total,
            old_value=old_value,
            reward=result.reward,
            done=result.done,
            done_reason=result.reason,
            planned_path_cells=diagnostics.planned_path_cells,
            path_length_m=path_length,
            path_observation_step_m=diagnostics.path_observation_step_m,
            coverage_gain_cells=result.coverage_gain_cells,
            coverage_gain_per_meter=gain_per_meter,
            observation_sample_count=observation_sample_count,
            ray_count_per_sample=ray_count_per_sample,
            ray_cell_visit_count=ray_cell_visit_count,
            newly_observed_cell_count=diagnostics.newly_observed_cell_count,
            planner_validation_summary=diagnostics.planner,
            execution_observation_summary=execution_summary,
            frontier_extractor_version=self.contract.frontier_extractor_version,
            observation_schema_version=self.contract.observation_schema_version,
            action_space_version=self.contract.action_space_version,
            reward_version=self.contract.reward_version,
            planner_version=self.contract.planner_version,
        )


def build_lunar_exploration_env(*, stage1_config: Mapping[str, object]) -> object:
    """Spawn-safe production factory；延迟导入，worker 内构造 env/planner/sensor。"""

    from lunar_exploration_ppo.configs.stage1 import Stage1Config
    from lunar_exploration_ppo.env.env import LunarExplorationEnv

    return LunarExplorationEnv(Stage1Config.model_validate(dict(stage1_config)))


def lunar_env_specs(
    stage1_config: Mapping[str, object],
) -> tuple[SpawnEnvSpec, ...]:
    payload = _copy_json_mapping(stage1_config, "Stage 1 config")
    return tuple(
        SpawnEnvSpec(
            factory=build_lunar_exploration_env,
            kwargs={"stage1_config": payload},
        )
        for _ in range(ROLLOUT_ENV_COUNT)
    )


def _worker_main(
    worker_index: int,
    connection: Connection,
    spec: SpawnEnvSpec,
) -> None:
    env: object | None = None
    try:
        env = spec.factory(**dict(spec.kwargs))
        connection.send(
            {
                "kind": "ready",
                "worker_index": worker_index,
                "pid": os.getpid(),
                "start_method": mp.get_start_method(),
            }
        )
        while True:
            command, payload = connection.recv()
            if command == "close":
                connection.send("closed")
                break
            try:
                if command == "reset":
                    observation = env.reset()  # type: ignore[attr-defined]
                    connection.send(
                        _worker_observation(env, observation, is_reset=True)
                    )
                elif command == "step":
                    result = env.step(payload)  # type: ignore[attr-defined]
                    if not isinstance(result, StepResult):
                        raise TypeError("worker env step must return StepResult")
                    connection.send(
                        WorkerStep(
                            result=result,
                            next_observation=_worker_observation(
                                env,
                                result.observation,
                                is_reset=False,
                            ),
                        )
                    )
                elif command == "current_observation":
                    current_observation = env.current_observation  # type: ignore[attr-defined]
                    if current_observation is None:
                        raise RuntimeError("worker current observation is unavailable")
                    connection.send(
                        _worker_observation(
                            env,
                            current_observation,
                            is_reset=False,
                        )
                    )
                elif command == "capture_state":
                    episode_state = env.export_episode_state()  # type: ignore[attr-defined]
                    sampler_state = (
                        env.export_sampler_state()  # type: ignore[attr-defined]
                        if hasattr(env, "export_sampler_state")
                        else {}
                    )
                    connection.send(
                        {
                            "episode_state": episode_state,
                            "sampler_state": sampler_state,
                        }
                    )
                elif command == "restore_state":
                    if not isinstance(payload, dict) or set(payload) != {
                        "episode_state",
                        "sampler_state",
                    }:
                        raise ValueError("worker restore payload mismatch")
                    env.import_episode_state(payload["episode_state"])  # type: ignore[attr-defined]
                    if hasattr(env, "import_sampler_state"):
                        env.import_sampler_state(payload["sampler_state"])  # type: ignore[attr-defined]
                    elif payload["sampler_state"] != {}:
                        raise ValueError("worker env has no sampler state importer")
                    connection.send("restored")
                else:
                    raise ValueError(f"unknown worker command: {command}")
            except BaseException as exc:
                connection.send(
                    {
                        "kind": "remote_error",
                        "worker_index": worker_index,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                break
    except (EOFError, BrokenPipeError, OSError):
        pass
    except BaseException as exc:
        try:
            connection.send(
                {
                    "kind": "remote_error",
                    "worker_index": worker_index,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if env is not None and hasattr(env, "close"):
            try:
                env.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        connection.close()


def _worker_observation(
    env: object,
    observation: object,
    *,
    is_reset: bool,
) -> WorkerObservation:
    if not isinstance(observation, PolicyObservation):
        raise TypeError("worker env reset/step observation must be PolicyObservation")
    action_set = env.current_action_set  # type: ignore[attr-defined]
    cells = tuple(action_set.cells)
    summary = dict(action_set.diagnostics or {})
    summary["candidate_count"] = len(cells)
    needs_policy = env.needs_policy  # type: ignore[attr-defined]
    if type(needs_policy) is not bool:
        raise TypeError("worker env needs_policy must be boolean")
    reset_trainable = False
    fake_logprob_created = False
    if is_reset:
        reset_diagnostics = env.last_reset_diagnostics  # type: ignore[attr-defined]
        if reset_diagnostics is None:
            raise RuntimeError("worker env reset diagnostics are missing")
        reset_trainable = reset_diagnostics.trainable
        fake_logprob_created = reset_diagnostics.fake_logprob_created
    return WorkerObservation(
        observation=observation,
        frontier_cells=cells,
        top_m_selection_summary=summary,
        needs_policy=needs_policy,
        reset_trainable=reset_trainable,
        fake_logprob_created=fake_logprob_created,
    )


def _copy_json_mapping(
    value: Mapping[str, object],
    name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        copied = json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON") from exc
    if not isinstance(copied, dict):
        raise ValueError(f"{name} must be a JSON object")
    ArtifactStore.canonical_json_bytes(copied)
    return copied


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "COLLECTOR_SCHEMA_VERSION",
    "CollectionResult",
    "CollectorAudit",
    "CollectorContract",
    "CollectorError",
    "RolloutCollector",
    "SpawnEnvSpec",
    "SpawnVectorEnv",
    "VECTOR_STATE_SCHEMA_VERSION",
    "WorkerObservation",
    "WorkerProcessError",
    "WorkerStep",
    "build_lunar_exploration_env",
    "lunar_env_specs",
]
