"""Stage 4 端到端训练恢复编排。"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final, Protocol, runtime_checkable

import torch

from lunar_exploration_ppo.ppo.checkpoint import CheckpointManager
from lunar_exploration_ppo.ppo.trainer import (
    PPOTrainer,
    policy_state_sha256,
)
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore


TRAINING_RESUME_SCHEMA_VERSION: Final = "stage4_training_resume/v1"


class TrainingResumeError(RuntimeError):
    """Checkpoint 无法完整应用到 fresh training runtime。"""


@runtime_checkable
class JsonStateComponent(Protocol):
    """可恢复并回读 canonical JSON 状态的 runtime 组件。"""

    def restore_state(self, state: dict[str, Any]) -> None: ...

    def capture_state(self) -> Mapping[str, Any]: ...


@runtime_checkable
class VectorEnvStateComponent(Protocol):
    """可恢复并回读八个 worker 状态的 vector runtime。"""

    def restore_states(self, states: Sequence[Mapping[str, Any]]) -> None: ...

    def capture_states(self) -> Sequence[Mapping[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class TrainingResumeReceipt:
    schema_version: str
    update_step: int
    next_update_step: int
    checkpoint_sha256: str
    manifest_sha256: str
    policy_state_sha256: str
    optimizer_state_sha256: str
    normalization_state_sha256: str
    scenario_sampler_state_sha256: str
    vector_env_states_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resume_training_from_last_checkpoint(
    *,
    checkpoint_manager: CheckpointManager,
    trainer: PPOTrainer,
    normalizer: JsonStateComponent,
    scenario_sampler: JsonStateComponent,
    vector_env: VectorEnvStateComponent,
    expected_config_sha256: str,
    expected_lineage: Mapping[str, Any],
) -> TrainingResumeReceipt:
    """把最后完整 checkpoint 原地应用到 fresh trainer 与全部 runtime 组件。"""

    if not isinstance(checkpoint_manager, CheckpointManager):
        raise TrainingResumeError("checkpoint_manager must be CheckpointManager")
    if not isinstance(trainer, PPOTrainer):
        raise TrainingResumeError("trainer must be PPOTrainer")
    if trainer.update_step != 0:
        raise TrainingResumeError("resume requires a fresh trainer at update_step 0")
    if not isinstance(normalizer, JsonStateComponent):
        raise TrainingResumeError("normalizer does not implement JSON state restore")
    if not isinstance(scenario_sampler, JsonStateComponent):
        raise TrainingResumeError(
            "scenario sampler does not implement JSON state restore"
        )
    if not isinstance(vector_env, VectorEnvStateComponent):
        raise TrainingResumeError("vector env does not implement state restore")

    loaded = checkpoint_manager.load_last_complete(
        policy=trainer.policy,
        optimizer=trainer.optimizer,
        expected_config_sha256=expected_config_sha256,
        expected_lineage=expected_lineage,
    )
    normalization_state = _canonical_mapping(
        loaded.normalization_stats,
        "normalization state",
    )
    sampler_state = _canonical_mapping(
        loaded.scenario_sampler_state,
        "scenario sampler state",
    )
    vector_states = _canonical_vector_states(loaded.vector_env_states)

    normalizer.restore_state(normalization_state)
    scenario_sampler.restore_state(sampler_state)
    vector_env.restore_states(vector_states)
    _require_canonical_equal(
        normalizer.capture_state(),
        normalization_state,
        "normalization state",
    )
    _require_canonical_equal(
        scenario_sampler.capture_state(),
        sampler_state,
        "scenario sampler state",
    )
    captured_vector_states = _canonical_vector_states(vector_env.capture_states())
    _require_canonical_equal(
        captured_vector_states,
        vector_states,
        "vector env states",
    )
    trainer.restore_update_step(loaded.update_step)

    restored_policy_hash = policy_state_sha256(trainer.policy)
    if restored_policy_hash != loaded.policy_state_sha256:
        raise TrainingResumeError("resumed policy hash does not match checkpoint")
    return TrainingResumeReceipt(
        schema_version=TRAINING_RESUME_SCHEMA_VERSION,
        update_step=loaded.update_step,
        next_update_step=loaded.update_step + 1,
        checkpoint_sha256=loaded.checkpoint_sha256,
        manifest_sha256=loaded.manifest_sha256,
        policy_state_sha256=restored_policy_hash,
        optimizer_state_sha256=optimizer_state_sha256(trainer.optimizer),
        normalization_state_sha256=_canonical_sha256(normalization_state),
        scenario_sampler_state_sha256=_canonical_sha256(sampler_state),
        vector_env_states_sha256=_canonical_sha256(vector_states),
    )


def optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
    """稳定哈希 optimizer 类型及完整 state_dict。"""

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TrainingResumeError("optimizer hash requires a torch optimizer")
    digest = hashlib.sha256()
    _hash_value(
        digest,
        {
            "optimizer_type": (
                f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
            ),
            "state_dict": optimizer.state_dict(),
        },
    )
    return digest.hexdigest()


def _canonical_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise TrainingResumeError(f"{label} must be a string-keyed mapping")
    try:
        payload = ArtifactStore.canonical_json_bytes(dict(value))
        copied = json.loads(payload.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise TrainingResumeError(f"{label} must be canonical JSON") from exc
    if not isinstance(copied, dict):
        raise TrainingResumeError(f"{label} must be a JSON object")
    return copied


def _canonical_vector_states(
    value: object,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TrainingResumeError("vector env states must be a sequence")
    if len(value) != 8:
        raise TrainingResumeError("resume requires exactly 8 vector env states")
    return tuple(
        _canonical_mapping(state, f"vector env state {index}")
        for index, state in enumerate(value)
    )


def _require_canonical_equal(actual: object, expected: object, label: str) -> None:
    try:
        actual_bytes = ArtifactStore.canonical_json_bytes(actual)
        expected_bytes = ArtifactStore.canonical_json_bytes(expected)
    except (TypeError, ValueError) as exc:
        raise TrainingResumeError(f"captured {label} is not canonical JSON") from exc
    if actual_bytes != expected_bytes:
        raise TrainingResumeError(f"{label} was not applied exactly")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(ArtifactStore.canonical_json_bytes(value)).hexdigest()


def _hash_value(digest: Any, value: object) -> None:
    if value is None:
        digest.update(b"N")
    elif type(value) is bool:
        digest.update(b"B1" if value else b"B0")
    elif type(value) is int:
        encoded = str(value).encode("ascii")
        digest.update(b"I" + struct.pack("<Q", len(encoded)) + encoded)
    elif type(value) is float:
        if not math.isfinite(value):
            raise TrainingResumeError("optimizer state contains non-finite float")
        digest.update(b"F" + struct.pack("<d", value))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S" + struct.pack("<Q", len(encoded)) + encoded)
    elif isinstance(value, bytes):
        digest.update(b"Y" + struct.pack("<Q", len(value)) + value)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().cpu()
        if not bool(torch.isfinite(tensor).all()):
            raise TrainingResumeError("optimizer state contains non-finite tensor")
        dtype = str(tensor.dtype).encode("ascii")
        shape = tuple(int(size) for size in tensor.shape)
        raw = tensor.numpy().tobytes(order="C")
        digest.update(b"T")
        _hash_value(digest, dtype)
        _hash_value(digest, shape)
        _hash_value(digest, raw)
    elif isinstance(value, Mapping):
        digest.update(b"M")
        items = sorted(
            value.items(),
            key=lambda item: (type(item[0]).__qualname__, repr(item[0])),
        )
        _hash_value(digest, len(items))
        for key, item in items:
            _hash_value(digest, key)
            _hash_value(digest, item)
    elif isinstance(value, Sequence):
        digest.update(b"Q")
        _hash_value(digest, len(value))
        for item in value:
            _hash_value(digest, item)
    else:
        raise TrainingResumeError(
            f"unsupported optimizer state value: {type(value).__qualname__}"
        )


__all__ = [
    "JsonStateComponent",
    "TRAINING_RESUME_SCHEMA_VERSION",
    "TrainingResumeError",
    "TrainingResumeReceipt",
    "VectorEnvStateComponent",
    "optimizer_state_sha256",
    "resume_training_from_last_checkpoint",
]
