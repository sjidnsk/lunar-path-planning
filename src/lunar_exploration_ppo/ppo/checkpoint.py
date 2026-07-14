"""Stage 4 完整、原子且可恢复的 PPO checkpoint。"""

from __future__ import annotations

import hashlib
import io
import os
import random
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn

from lunar_exploration_ppo.ppo.trainer import policy_state_sha256
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore


CHECKPOINT_SCHEMA_VERSION: Final = "stage4_complete_checkpoint/v1"
CHECKPOINT_MANIFEST_SCHEMA_VERSION: Final = "stage4_checkpoint_manifest/v1"
CHECKPOINT_COMPLETE_SCHEMA_VERSION: Final = "stage4_checkpoint_complete/v1"
CHECKPOINT_LATEST_SCHEMA_VERSION: Final = "stage4_checkpoint_latest/v1"
_CHECKPOINT_NAME = "checkpoint.pt"
_MANIFEST_NAME = "manifest.json"
_COMPLETE_NAME = "complete.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UPDATE_DIRECTORY_RE = re.compile(r"^update-(?P<step>[0-9]{8})$")
_REQUIRED_VERSIONS = frozenset(
    {
        "observation_schema_version",
        "action_space_version",
        "network_architecture_version",
        "reward_version",
        "frontier_version",
        "planner_version",
    }
)
_REQUIRED_LINEAGE = frozenset(
    {
        "stage3_commit",
        "stage3_gate_sha256",
        "stage3_approval_sha256",
    }
)


class CheckpointError(RuntimeError):
    """Checkpoint 不完整、损坏或与当前训练 lineage 不兼容。"""


@dataclass(frozen=True, slots=True)
class CheckpointReceipt:
    update_step: int
    directory: Path
    checkpoint_path: Path
    manifest_path: Path
    complete_marker_path: Path
    checkpoint_sha256: str
    manifest_sha256: str
    policy_state_sha256: str


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    update_step: int
    directory: Path
    checkpoint_sha256: str
    manifest_sha256: str
    policy_state_sha256: str
    normalization_stats: dict[str, Any]
    scenario_sampler_state: dict[str, Any]
    vector_env_states: tuple[dict[str, Any], ...]
    best_record: dict[str, Any]
    versions: dict[str, str]
    top_m_config: dict[str, Any]
    scale_profile: str
    training_config: dict[str, Any]
    config_sha256: str
    lineage: dict[str, Any]
    eval_metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _ValidatedCandidate:
    directory: Path
    checkpoint_sha256: str
    manifest_sha256: str
    payload: dict[str, Any]


class CheckpointManager:
    """只暴露带 manifest 与 complete marker 的不可覆盖 update。"""

    def __init__(
        self,
        root: str | Path,
        *,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if os.name == "nt" and self.root.drive.upper() != "D:":
            raise CheckpointError("Stage 4 checkpoints must be stored on the D drive")
        self._store = ArtifactStore(self.root)
        self._store.resolve(".").mkdir(parents=True, exist_ok=True)
        self._fault_injector = fault_injector

    def save_complete(
        self,
        *,
        policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        update_step: int,
        normalization_stats: Mapping[str, Any],
        scenario_sampler_state: Mapping[str, Any],
        vector_env_states: Sequence[Mapping[str, Any]],
        best_record: Mapping[str, Any],
        versions: Mapping[str, str],
        top_m_config: Mapping[str, Any],
        scale_profile: str,
        training_config: Mapping[str, Any],
        config_sha256: str,
        lineage: Mapping[str, Any],
        eval_metrics: Mapping[str, Any],
    ) -> CheckpointReceipt:
        metadata = _validated_metadata(
            policy=policy,
            optimizer=optimizer,
            update_step=update_step,
            normalization_stats=normalization_stats,
            scenario_sampler_state=scenario_sampler_state,
            vector_env_states=vector_env_states,
            best_record=best_record,
            versions=versions,
            top_m_config=top_m_config,
            scale_profile=scale_profile,
            training_config=training_config,
            config_sha256=config_sha256,
            lineage=lineage,
            eval_metrics=eval_metrics,
        )
        directory_name = _update_directory_name(update_step)
        destination = self._store.resolve(directory_name)
        if destination.exists():
            raise CheckpointError(f"checkpoint update {update_step} already exists")

        policy_hash = policy_state_sha256(policy)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "update_step": update_step,
            "policy_state_sha256": policy_hash,
            "rng_state": _capture_rng_state(),
            **metadata,
        }
        payload_buffer = io.BytesIO()
        torch.save(payload, payload_buffer)
        payload_bytes = payload_buffer.getvalue()
        checkpoint_hash = _sha256(payload_bytes)
        lineage_hash = _canonical_sha256(metadata["lineage"])
        manifest = {
            "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
            "update_step": update_step,
            "checkpoint": {
                "path": _CHECKPOINT_NAME,
                "sha256": checkpoint_hash,
                "size_bytes": len(payload_bytes),
            },
            "policy_state_sha256": policy_hash,
            "config_sha256": config_sha256,
            "lineage_sha256": lineage_hash,
        }
        manifest_bytes = ArtifactStore.canonical_json_bytes(manifest)
        manifest_hash = _sha256(manifest_bytes)
        complete = {
            "schema_version": CHECKPOINT_COMPLETE_SCHEMA_VERSION,
            "update_step": update_step,
            "manifest_sha256": manifest_hash,
            "checkpoint_sha256": checkpoint_hash,
        }
        complete_bytes = ArtifactStore.canonical_json_bytes(complete)

        pending = self._store.resolve(
            f".pending-{directory_name}-{uuid.uuid4().hex}"
        )
        pending.mkdir(parents=False, exist_ok=False)
        published = False
        try:
            _write_new_fsynced(pending / _CHECKPOINT_NAME, payload_bytes)
            self._inject("after_payload_write")
            _write_new_fsynced(pending / _MANIFEST_NAME, manifest_bytes)
            self._inject("after_manifest_write")
            _write_new_fsynced(pending / _COMPLETE_NAME, complete_bytes)
            self._inject("before_publish_rename")
            os.replace(pending, destination)
            published = True
            self._inject("after_publish_rename")

            latest = {
                "schema_version": CHECKPOINT_LATEST_SCHEMA_VERSION,
                "update_step": update_step,
                "directory": directory_name,
                "manifest_sha256": manifest_hash,
                "checkpoint_sha256": checkpoint_hash,
            }
            self._store.write_json("latest.json", latest)
        except BaseException:
            if not published:
                _remove_known_pending_files(pending)
            raise

        return CheckpointReceipt(
            update_step=update_step,
            directory=destination,
            checkpoint_path=destination / _CHECKPOINT_NAME,
            manifest_path=destination / _MANIFEST_NAME,
            complete_marker_path=destination / _COMPLETE_NAME,
            checkpoint_sha256=checkpoint_hash,
            manifest_sha256=manifest_hash,
            policy_state_sha256=policy_hash,
        )

    def load_last_complete(
        self,
        *,
        policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        expected_config_sha256: str,
        expected_lineage: Mapping[str, Any],
    ) -> LoadedCheckpoint:
        if not isinstance(policy, nn.Module):
            raise CheckpointError("policy must be a torch module")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise CheckpointError("optimizer must be a torch optimizer")
        _require_sha256(expected_config_sha256, "expected config")
        expected_lineage_dict = _validated_json_mapping(
            expected_lineage,
            "expected lineage",
        )
        if not _REQUIRED_LINEAGE.issubset(expected_lineage_dict):
            raise CheckpointError("expected lineage is incomplete")

        rejection_reasons: list[str] = []
        candidate_directories = sorted(
            (
                path
                for path in self.root.iterdir()
                if path.is_dir() and _UPDATE_DIRECTORY_RE.fullmatch(path.name)
            ),
            key=lambda path: int(
                _UPDATE_DIRECTORY_RE.fullmatch(path.name).group("step")  # type: ignore[union-attr]
            ),
            reverse=True,
        )
        selected: _ValidatedCandidate | None = None
        for directory in candidate_directories:
            try:
                candidate = self._validate_candidate(directory)
                payload = candidate.payload
                if payload["config_sha256"] != expected_config_sha256:
                    raise CheckpointError("config hash mismatch")
                if payload["lineage"] != expected_lineage_dict:
                    raise CheckpointError("lineage mismatch")
                selected = candidate
                break
            except Exception as exc:
                rejection_reasons.append(f"{directory.name}: {exc}")
        if selected is None:
            detail = "; ".join(rejection_reasons) or "no completed update directories"
            raise CheckpointError(
                f"no compatible complete checkpoint: {detail}"
            )

        payload = selected.payload
        _validate_model_state(policy, payload["model_state_dict"])
        policy.load_state_dict(payload["model_state_dict"], strict=True)
        restored_hash = policy_state_sha256(policy)
        if restored_hash != payload["policy_state_sha256"]:
            raise CheckpointError("restored policy hash mismatch")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _restore_rng_state(payload["rng_state"])
        return LoadedCheckpoint(
            update_step=payload["update_step"],
            directory=selected.directory,
            checkpoint_sha256=selected.checkpoint_sha256,
            manifest_sha256=selected.manifest_sha256,
            policy_state_sha256=restored_hash,
            normalization_stats=dict(payload["normalization_stats"]),
            scenario_sampler_state=dict(payload["scenario_sampler_state"]),
            vector_env_states=tuple(
                dict(state) for state in payload["vector_env_states"]
            ),
            best_record=dict(payload["best_record"]),
            versions=dict(payload["versions"]),
            top_m_config=dict(payload["top_m_config"]),
            scale_profile=payload["scale_profile"],
            training_config=dict(payload["training_config"]),
            config_sha256=payload["config_sha256"],
            lineage=dict(payload["lineage"]),
            eval_metrics=dict(payload["eval_metrics"]),
        )

    def _validate_candidate(self, directory: Path) -> _ValidatedCandidate:
        match = _UPDATE_DIRECTORY_RE.fullmatch(directory.name)
        if match is None:
            raise CheckpointError("invalid checkpoint directory name")
        expected_step = int(match.group("step"))
        checkpoint_path = directory / _CHECKPOINT_NAME
        manifest_path = directory / _MANIFEST_NAME
        complete_path = directory / _COMPLETE_NAME
        if not (
            checkpoint_path.is_file()
            and manifest_path.is_file()
            and complete_path.is_file()
        ):
            raise CheckpointError("checkpoint, manifest, or complete marker missing")

        manifest, manifest_bytes = _read_canonical_json(manifest_path)
        complete, _ = _read_canonical_json(complete_path)
        _validate_manifest(manifest, expected_step=expected_step)
        _validate_complete(complete, expected_step=expected_step)
        manifest_hash = _sha256(manifest_bytes)
        if complete["manifest_sha256"] != manifest_hash:
            raise CheckpointError("complete marker manifest hash mismatch")

        checkpoint_bytes = checkpoint_path.read_bytes()
        checkpoint_hash = _sha256(checkpoint_bytes)
        checkpoint_record = manifest["checkpoint"]
        if (
            checkpoint_record["sha256"] != checkpoint_hash
            or complete["checkpoint_sha256"] != checkpoint_hash
            or checkpoint_record["size_bytes"] != len(checkpoint_bytes)
        ):
            raise CheckpointError("checkpoint payload hash or size mismatch")
        try:
            payload = torch.load(
                io.BytesIO(checkpoint_bytes),
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            raise CheckpointError("checkpoint payload cannot be decoded") from exc
        _validate_loaded_payload(
            payload,
            expected_step=expected_step,
            manifest=manifest,
        )
        return _ValidatedCandidate(
            directory=directory,
            checkpoint_sha256=checkpoint_hash,
            manifest_sha256=manifest_hash,
            payload=payload,
        )

    def _inject(self, phase: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(phase)


def _validated_metadata(
    *,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    update_step: int,
    normalization_stats: Mapping[str, Any],
    scenario_sampler_state: Mapping[str, Any],
    vector_env_states: Sequence[Mapping[str, Any]],
    best_record: Mapping[str, Any],
    versions: Mapping[str, str],
    top_m_config: Mapping[str, Any],
    scale_profile: str,
    training_config: Mapping[str, Any],
    config_sha256: str,
    lineage: Mapping[str, Any],
    eval_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(policy, nn.Module):
        raise CheckpointError("policy must be a torch module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise CheckpointError("optimizer must be a torch optimizer")
    if type(update_step) is not int or update_step <= 0:
        raise CheckpointError("update_step must be a positive integer")
    if not isinstance(vector_env_states, Sequence) or len(vector_env_states) != 8:
        raise CheckpointError("checkpoint requires exactly 8 vector env states")
    vector_states = tuple(
        _validated_json_mapping(state, "vector env state")
        for state in vector_env_states
    )
    version_dict = _validated_string_mapping(versions, "versions")
    if set(version_dict) != _REQUIRED_VERSIONS:
        raise CheckpointError("checkpoint versions must contain the exact frozen fields")
    if not isinstance(scale_profile, str) or not scale_profile:
        raise CheckpointError("scale_profile must be non-empty")
    _require_sha256(config_sha256, "config")
    lineage_dict = _validated_json_mapping(lineage, "lineage")
    if not _REQUIRED_LINEAGE.issubset(lineage_dict):
        raise CheckpointError("checkpoint lineage is incomplete")
    metadata = {
        "normalization_stats": _validated_json_mapping(
            normalization_stats,
            "normalization stats",
        ),
        "scenario_sampler_state": _validated_json_mapping(
            scenario_sampler_state,
            "scenario sampler state",
        ),
        "vector_env_states": vector_states,
        "best_record": _validated_json_mapping(best_record, "best record"),
        "versions": version_dict,
        "top_m_config": _validated_json_mapping(top_m_config, "top-M config"),
        "scale_profile": scale_profile,
        "training_config": _validated_json_mapping(
            training_config,
            "training config",
        ),
        "config_sha256": config_sha256,
        "lineage": lineage_dict,
        "eval_metrics": _validated_json_mapping(eval_metrics, "eval metrics"),
    }
    try:
        ArtifactStore.canonical_json_bytes(metadata)
    except (TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint metadata must be canonical JSON data") from exc
    return metadata


def _validated_json_mapping(
    value: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise CheckpointError(f"{name} must be a string-keyed mapping")
    result = dict(value)
    try:
        ArtifactStore.canonical_json_bytes(result)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"{name} must contain canonical JSON values") from exc
    return result


def _validated_string_mapping(
    value: Mapping[str, str],
    name: str,
) -> dict[str, str]:
    result = _validated_json_mapping(value, name)
    if any(not isinstance(item, str) or not item for item in result.values()):
        raise CheckpointError(f"{name} values must be non-empty strings")
    return result


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()
        ),
    }


def _restore_rng_state(state: object) -> None:
    if not isinstance(state, Mapping) or set(state) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise CheckpointError("checkpoint RNG state is incomplete")
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        cuda_states = tuple(state["torch_cuda"])
        if cuda_states:
            if not torch.cuda.is_available():
                raise CheckpointError(
                    "checkpoint contains CUDA RNG state but CUDA is unavailable"
                )
            if len(cuda_states) != torch.cuda.device_count():
                raise CheckpointError("CUDA RNG device count mismatch")
            torch.cuda.set_rng_state_all(list(cuda_states))
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError("checkpoint RNG state cannot be restored") from exc


def _validate_loaded_payload(
    payload: object,
    *,
    expected_step: int,
    manifest: dict[str, Any],
) -> None:
    required = {
        "schema_version",
        "model_state_dict",
        "optimizer_state_dict",
        "update_step",
        "policy_state_sha256",
        "rng_state",
        "normalization_stats",
        "scenario_sampler_state",
        "vector_env_states",
        "best_record",
        "versions",
        "top_m_config",
        "scale_profile",
        "training_config",
        "config_sha256",
        "lineage",
        "eval_metrics",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise CheckpointError("checkpoint payload fields are not exact")
    if (
        payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION
        or payload["update_step"] != expected_step
    ):
        raise CheckpointError("checkpoint schema or update step mismatch")
    _require_sha256(payload["policy_state_sha256"], "policy state")
    _require_sha256(payload["config_sha256"], "config")
    if (
        payload["policy_state_sha256"] != manifest["policy_state_sha256"]
        or payload["config_sha256"] != manifest["config_sha256"]
        or _canonical_sha256(payload["lineage"]) != manifest["lineage_sha256"]
    ):
        raise CheckpointError("checkpoint metadata does not match manifest")
    if not isinstance(payload["model_state_dict"], Mapping):
        raise CheckpointError("model state dict is invalid")
    if not isinstance(payload["optimizer_state_dict"], Mapping):
        raise CheckpointError("optimizer state dict is invalid")
    if not isinstance(payload["vector_env_states"], (tuple, list)) or len(
        payload["vector_env_states"]
    ) != 8:
        raise CheckpointError("checkpoint vector env state count is invalid")
    version_dict = _validated_string_mapping(payload["versions"], "versions")
    if set(version_dict) != _REQUIRED_VERSIONS:
        raise CheckpointError("checkpoint versions are incomplete")
    for name in (
        "normalization_stats",
        "scenario_sampler_state",
        "best_record",
        "top_m_config",
        "training_config",
        "lineage",
        "eval_metrics",
    ):
        _validated_json_mapping(payload[name], name)
    for state in payload["vector_env_states"]:
        _validated_json_mapping(state, "vector env state")


def _validate_model_state(
    policy: nn.Module,
    saved_state: Mapping[str, Any],
) -> None:
    current = policy.state_dict()
    if set(current) != set(saved_state):
        raise CheckpointError("checkpoint model keys do not match policy")
    for name, current_tensor in current.items():
        saved_tensor = saved_state[name]
        if not isinstance(saved_tensor, torch.Tensor):
            raise CheckpointError("checkpoint model state contains non-tensor value")
        if (
            current_tensor.shape != saved_tensor.shape
            or current_tensor.dtype != saved_tensor.dtype
        ):
            raise CheckpointError(f"checkpoint model tensor mismatch: {name}")


def _validate_manifest(manifest: object, *, expected_step: int) -> None:
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "update_step",
        "checkpoint",
        "policy_state_sha256",
        "config_sha256",
        "lineage_sha256",
    }:
        raise CheckpointError("checkpoint manifest fields are not exact")
    if (
        manifest["schema_version"] != CHECKPOINT_MANIFEST_SCHEMA_VERSION
        or manifest["update_step"] != expected_step
    ):
        raise CheckpointError("checkpoint manifest schema or step mismatch")
    checkpoint = manifest["checkpoint"]
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise CheckpointError("checkpoint manifest payload record is invalid")
    if (
        checkpoint["path"] != _CHECKPOINT_NAME
        or type(checkpoint["size_bytes"]) is not int
        or checkpoint["size_bytes"] <= 0
    ):
        raise CheckpointError("checkpoint manifest path or size is invalid")
    for key in ("policy_state_sha256", "config_sha256", "lineage_sha256"):
        _require_sha256(manifest[key], key)
    _require_sha256(checkpoint["sha256"], "checkpoint")


def _validate_complete(complete: object, *, expected_step: int) -> None:
    if not isinstance(complete, dict) or set(complete) != {
        "schema_version",
        "update_step",
        "manifest_sha256",
        "checkpoint_sha256",
    }:
        raise CheckpointError("checkpoint complete marker fields are not exact")
    if (
        complete["schema_version"] != CHECKPOINT_COMPLETE_SCHEMA_VERSION
        or complete["update_step"] != expected_step
    ):
        raise CheckpointError("checkpoint complete marker schema or step mismatch")
    _require_sha256(complete["manifest_sha256"], "manifest")
    _require_sha256(complete["checkpoint_sha256"], "checkpoint")


def _read_canonical_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        import json

        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CheckpointError(f"invalid JSON checkpoint artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"checkpoint JSON artifact is not an object: {path.name}")
    if raw != ArtifactStore.canonical_json_bytes(value):
        raise CheckpointError(f"checkpoint JSON artifact is not canonical: {path.name}")
    return value, raw


def _write_new_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _remove_known_pending_files(directory: Path) -> None:
    for name in (_CHECKPOINT_NAME, _MANIFEST_NAME, _COMPLETE_NAME):
        path = directory / name
        if path.is_file():
            path.unlink()
    if directory.is_dir():
        directory.rmdir()


def _update_directory_name(update_step: int) -> str:
    return f"update-{update_step:08d}"


def _canonical_sha256(value: object) -> str:
    return _sha256(ArtifactStore.canonical_json_bytes(value))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CheckpointError(f"{name} SHA-256 is invalid")


__all__ = [
    "CHECKPOINT_COMPLETE_SCHEMA_VERSION",
    "CHECKPOINT_LATEST_SCHEMA_VERSION",
    "CHECKPOINT_MANIFEST_SCHEMA_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointError",
    "CheckpointManager",
    "CheckpointReceipt",
    "LoadedCheckpoint",
]
