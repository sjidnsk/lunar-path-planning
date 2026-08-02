"""Standard v1 PPO 正式训练的确定性控制面与 hard gate。"""

from __future__ import annotations

import csv
import hashlib
import io
import inspect
import json
import math
import os
import random
import stat
import struct
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import nn

from lunar_exploration_ppo.configs.stage6 import SafetyContract, Stage6Config
from lunar_exploration_ppo.ppo.trainer import policy_state_sha256
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl, DurableJsonlError
from lunar_exploration_ppo.utils.path_security import (
    PathSecurityError,
    durable_file_identity,
    durable_unlink,
    lexical_absolute,
    require_plain_path,
    secure_read_bytes,
)

if TYPE_CHECKING:
    from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
        PlanningChildRecoveryCapability,
    )


FROZEN_SEEDS = (20260716,)
SAMPLER_SEED_DERIVATION_VERSION = "stage6_seed_times_16_plus_lane/v1"
AUDIT_UPDATES = frozenset({1, 10, 50, 100})
_INITIAL_RECORD_HASH = (
    "3057252298292ae9a47b0879812581277456d1e9d9c0995178967da7c929236d"
)
_REVIEW_IMMUTABLE_BINDING_KEYS = frozenset(
    {
        "formal_run_id",
        "changed_path_set_sha256",
        "review_authorization_record_sha256",
        "authorization_file_sha256",
        "review_identity_sha256",
        "reviewed_prospective_git_tree",
        "frozen_diff_sha256",
        "spec_review_sha256",
        "quality_review_sha256",
    }
)
_BASE_STAGE6_IMMUTABLE_BINDING_KEYS = frozenset(
    {
        "config_sha256",
        "source_set_sha256",
        "prospective_tree_sha256",
        "data_sha256",
        "environment_identity",
        "environment_sha256",
        "stage5_gate_sha256",
        *_REVIEW_IMMUTABLE_BINDING_KEYS,
    }
)
_COVERAGE_CACHE_IMMUTABLE_BINDING_KEYS = frozenset(
    {
        "coverage_cache_manifest_path",
        "coverage_cache_manifest_sha256",
        "coverage_cache_manifest_size_bytes",
        "coverage_cache_root",
        "coverage_cache_entry_set_sha256",
        "coverage_cache_runtime_mode",
        "coverage_cache_formal_audit_sha256",
        "coverage_cache_formal_audit_size_bytes",
    }
)
STAGE6_IMMUTABLE_BINDING_KEYS = frozenset(
    {*_BASE_STAGE6_IMMUTABLE_BINDING_KEYS, *_COVERAGE_CACHE_IMMUTABLE_BINDING_KEYS}
)
_IMMUTABLE_JOURNAL_BINDING_KEYS = _BASE_STAGE6_IMMUTABLE_BINDING_KEYS
_JOURNAL_BINDING_KEYS = frozenset(
    {*_IMMUTABLE_JOURNAL_BINDING_KEYS, "checkpoint_sha256"}
)
_PHASE_STATE_SEQUENCE = (
    "preflight",
    "global_best_frozen",
    "final_test_running",
    "final_unseen_running",
    "baselines_running",
    "machine_passed",
    "awaiting_independent_review",
)
_CHECKPOINT_RECEIPT_FIELDS = frozenset(
    {
        "transaction_key",
        "seed",
        "update",
        "checkpoint_sha256",
        "complete_marker_sha256",
        "policy_state_sha256",
        "previous_record_hash",
        "record_hash",
    }
)
_VALIDATION_TRACE_BINDING_SCHEMA_VERSION = "stage6_validation_trace_binding/v1"


class StandardTrainingError(RuntimeError):
    """Standard 训练状态、指标或 checkpoint 合同无效。"""


def _require_standard_execution_capability(
    execution_capability: object,
    *,
    label: str,
    rehash_inputs: bool = False,
    config: Stage6Config,
    run_root: str | Path,
    repo_root: str | Path,
    stage5_authority: Mapping[str, object],
    verified_review_authorization: Mapping[str, object] | None = None,
    execution_identity: Mapping[str, object] | None = None,
) -> None:
    from lunar_exploration_ppo.workflows.stage6 import (
        Stage6WorkflowError,
        _require_stage6_execution_capability,
    )

    run = lexical_absolute(run_root)
    repo = lexical_absolute(repo_root)
    try:
        _require_stage6_execution_capability(
            execution_capability,
            label=label,
            rehash_inputs=rehash_inputs,
            formal_run_id=run.name,
            run_root=run,
            stage_root=run / "s6",
            repo_root=repo,
            config_path=repo / "configs/ppo_highres_frontier_stage6_v1.json",
            config=config,
            execution_identity=execution_identity,
            stage5_authority=stage5_authority,
            verified_review_authorization=verified_review_authorization,
        )
    except Stage6WorkflowError as exc:
        raise StandardTrainingError(
            f"Stage 6 execution capability rejected at {label}"
        ) from exc


@contextmanager
def _standard_execution_operation(
    execution_capability: object,
    *,
    label: str,
    config: Stage6Config,
    run_root: str | Path,
    repo_root: str | Path,
    stage5_authority: Mapping[str, object],
    verified_review_authorization: Mapping[str, object] | None = None,
    execution_identity: Mapping[str, object] | None = None,
):
    from lunar_exploration_ppo.workflows.stage6 import (
        Stage6WorkflowError,
        _stage6_execution_operation,
    )

    run = lexical_absolute(run_root)
    repo = lexical_absolute(repo_root)
    try:
        with _stage6_execution_operation(
            execution_capability,
            label=label,
            formal_run_id=run.name,
            run_root=run,
            stage_root=run / "s6",
            repo_root=repo,
            config_path=repo / "configs/ppo_highres_frontier_stage6_v1.json",
            config=config,
            execution_identity=execution_identity,
            stage5_authority=stage5_authority,
            verified_review_authorization=verified_review_authorization,
        ):
            yield
    except Stage6WorkflowError as exc:
        raise StandardTrainingError(
            f"Stage 6 execution capability rejected at {label}"
        ) from exc


def _guarded_standard_training_mutation(function):
    def guarded(*args: object, **kwargs: object):
        with _standard_execution_operation(
            kwargs.get("execution_capability"),
            label=function.__name__,
            config=kwargs.get("config"),  # type: ignore[arg-type]
            run_root=kwargs.get("run_root"),  # type: ignore[arg-type]
            repo_root=kwargs.get("repo_root"),  # type: ignore[arg-type]
            stage5_authority=kwargs.get("stage5_authority"),  # type: ignore[arg-type]
            verified_review_authorization=kwargs.get(
                "verified_review_authorization"
            ),  # type: ignore[arg-type]
        ):
            return function(*args, **kwargs)

    guarded.__name__ = function.__name__
    guarded.__qualname__ = function.__qualname__
    guarded.__doc__ = function.__doc__
    guarded.__annotations__ = dict(function.__annotations__)
    guarded.__signature__ = inspect.signature(function)
    return guarded


def _guarded_backend_constructor(function):
    def guarded(self, *args: object, **kwargs: object):
        with _standard_execution_operation(
            kwargs.get("execution_capability"),
            label="StandardProductionBackend.__init__",
            config=kwargs.get("config"),  # type: ignore[arg-type]
            run_root=kwargs.get("run_root"),  # type: ignore[arg-type]
            repo_root=kwargs.get("repo_root"),  # type: ignore[arg-type]
            stage5_authority=kwargs.get("stage5_authority"),  # type: ignore[arg-type]
        ):
            return function(self, *args, **kwargs)

    guarded.__name__ = function.__name__
    guarded.__qualname__ = function.__qualname__
    guarded.__doc__ = function.__doc__
    guarded.__annotations__ = dict(function.__annotations__)
    guarded.__signature__ = inspect.signature(function)
    return guarded


def _guarded_backend_mutation(function):
    def guarded(self, *args: object, **kwargs: object):
        with self._execution_operation(function.__name__):
            return function(self, *args, **kwargs)

    guarded.__name__ = function.__name__
    guarded.__qualname__ = function.__qualname__
    guarded.__doc__ = function.__doc__
    guarded.__annotations__ = dict(function.__annotations__)
    guarded.__signature__ = inspect.signature(function)
    return guarded


def derive_standard_sampler_seeds(seed: int) -> tuple[int, ...]:
    if seed not in FROZEN_SEEDS:
        raise StandardTrainingError("sampler seed requires a frozen training seed")
    return tuple(seed * 16 + lane for lane in range(8))


def _seed_standard_training_rng(seed: int) -> None:
    if seed not in FROZEN_SEEDS:
        raise StandardTrainingError("RNG initialization seed drifted")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True, slots=True)
class TrainingUnit:
    sequence: int
    seed: int
    seed_index: int
    kind: Literal["initialize", "train", "validate", "complete"]
    update: int
    episode_count: int = 0

    @property
    def key(self) -> str:
        return f"{self.sequence:04d}:{self.seed}:{self.kind}:{self.update:03d}"

    @property
    def state(self) -> str:
        if self.kind == "initialize":
            return f"seed_{self.seed}_initializing"
        if self.kind == "train":
            return f"seed_{self.seed}_training_update_{self.update}"
        if self.kind == "validate":
            return f"seed_{self.seed}_validating_update_{self.update}"
        return f"seed_{self.seed}_complete"


@dataclass(frozen=True, slots=True)
class StandardTrainingTransaction:
    sequence: int
    seed: int
    seed_index: int
    update: int
    validation_episodes: int
    commit_states: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.sequence) is not int
            or self.sequence < 0
            or self.seed not in FROZEN_SEEDS
            or type(self.seed_index) is not int
            or not 0 <= self.seed_index < len(FROZEN_SEEDS)
            or FROZEN_SEEDS[self.seed_index] != self.seed
            or type(self.update) is not int
            or not 1 <= self.update <= 100
            or self.validation_episodes not in {0, 16}
            or not self.commit_states
        ):
            raise StandardTrainingError("Stage 6 training transaction drifted")

    @property
    def key(self) -> str:
        return f"{self.sequence:04d}:{self.seed}:update:{self.update:03d}"


@dataclass(frozen=True, slots=True)
class StandardTransactionCheckpoint:
    transaction_key: str
    seed: int
    update: int
    checkpoint_sha256: str
    complete_marker_sha256: str
    policy_state_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.transaction_key, str)
            or not self.transaction_key
            or self.seed not in FROZEN_SEEDS
            or type(self.update) is not int
            or not 1 <= self.update <= 100
            or any(
                not _is_sha256(value)
                for value in (
                    self.checkpoint_sha256,
                    self.complete_marker_sha256,
                    self.policy_state_sha256,
                )
            )
        ):
            raise StandardTrainingError("Stage 6 transaction checkpoint drifted")


def _checkpoint_acceptance_identity(
    checkpoint: StandardTransactionCheckpoint,
) -> dict[str, object]:
    if not isinstance(checkpoint, StandardTransactionCheckpoint):
        raise StandardTrainingError("resource acceptance checkpoint drifted")
    return {
        "transaction_key": checkpoint.transaction_key,
        "seed": checkpoint.seed,
        "update": checkpoint.update,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "complete_marker_sha256": checkpoint.complete_marker_sha256,
        "policy_state_sha256": checkpoint.policy_state_sha256,
    }


class CheckpointReceiptIndex:
    """Append-only durable identities for complete Stage 6 checkpoints."""

    def __init__(self, path: str | Path) -> None:
        self.path = lexical_absolute(path)
        self._durable = DurableJsonl(self.path)

    def append_once(self, checkpoint: StandardTransactionCheckpoint) -> bool:
        if not isinstance(checkpoint, StandardTransactionCheckpoint):
            raise StandardTrainingError("checkpoint receipt is invalid")
        rows = self.verify()
        payload = {
            "transaction_key": checkpoint.transaction_key,
            "seed": checkpoint.seed,
            "update": checkpoint.update,
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "complete_marker_sha256": checkpoint.complete_marker_sha256,
            "policy_state_sha256": checkpoint.policy_state_sha256,
        }
        for row in rows:
            if row["transaction_key"] != checkpoint.transaction_key:
                continue
            existing = {key: row[key] for key in payload}
            if existing != payload:
                raise StandardTrainingError("checkpoint receipt drifted for existing key")
            return False
        previous = rows[-1]["record_hash"] if rows else _INITIAL_RECORD_HASH
        event = {**payload, "previous_record_hash": previous}
        row = {**event, "record_hash": _receipt_record_hash(event)}
        try:
            self._durable.append(row)
        except (DurableJsonlError, OSError) as exc:
            raise StandardTrainingError(
                "checkpoint receipt durable append failed"
            ) from exc
        return True

    def verify(self) -> tuple[dict[str, object], ...]:
        try:
            payload = self._durable.recover_and_snapshot()
        except (DurableJsonlError, OSError) as exc:
            raise StandardTrainingError(
                "checkpoint receipt committed snapshot failed"
            ) from exc
        return self.verify_snapshot_bytes(payload)

    @classmethod
    def verify_snapshot_bytes(
        cls,
        payload: bytes,
    ) -> tuple[dict[str, object], ...]:
        """Verify exact receipt-index bytes without recovery or path access."""

        if type(payload) is not bytes:
            raise StandardTrainingError("checkpoint receipt snapshot bytes drifted")
        if not payload:
            return ()
        rows: list[dict[str, object]] = []
        try:
            for line in payload.splitlines(keepends=True):
                value = json.loads(line.decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("receipt row is not an object")
                canonical = (
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                if line != canonical:
                    raise ValueError("receipt row is not canonical")
                rows.append(value)
            return cls._validate_rows(rows)
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            raise StandardTrainingError("checkpoint receipt index drifted") from exc

    @staticmethod
    def _validate_rows(
        rows: Sequence[Mapping[str, object]],
    ) -> tuple[dict[str, object], ...]:
        previous = _INITIAL_RECORD_HASH
        seen: set[str] = set()
        verified: list[dict[str, object]] = []
        for raw in rows:
            if not isinstance(raw, Mapping) or set(raw) != _CHECKPOINT_RECEIPT_FIELDS:
                raise StandardTrainingError("checkpoint receipt fields drifted")
            row = dict(raw)
            transaction_key = row["transaction_key"]
            if (
                not isinstance(transaction_key, str)
                or not transaction_key
                or transaction_key in seen
                or row["seed"] not in FROZEN_SEEDS
                or type(row["update"]) is not int
                or not 1 <= row["update"] <= 100
                or any(
                    not _is_sha256(row[key])
                    for key in (
                        "checkpoint_sha256",
                        "complete_marker_sha256",
                        "policy_state_sha256",
                        "previous_record_hash",
                        "record_hash",
                    )
                )
            ):
                raise StandardTrainingError("checkpoint receipt value drifted")
            event = {key: row[key] for key in row if key != "record_hash"}
            if (
                row["previous_record_hash"] != previous
                or row["record_hash"] != _receipt_record_hash(event)
            ):
                raise StandardTrainingError("checkpoint receipt hash chain drifted")
            seen.add(transaction_key)
            verified.append(row)
            previous = str(row["record_hash"])
        return tuple(verified)


@dataclass(frozen=True, slots=True)
class EvalIsolationSnapshot:
    policy_sha256: str
    policy_training: bool
    optimizer_sha256: str
    rng_sha256: str
    normalizer_sha256: str
    runtime_state_sha256: str
    binding_sha256: str


@dataclass(frozen=True, slots=True)
class StandardUpdateResult:
    transaction: StandardTrainingTransaction
    update_metrics: object
    validation_result: object | None
    eval_isolation_audit: Mapping[str, object] | None
    best_record: Mapping[str, object]
    checkpoint: StandardTransactionCheckpoint
    journal_records: tuple[dict[str, object], ...]
    collection_audit: Mapping[str, object] = field(default_factory=dict)


def run_standard_update_transaction(
    *,
    transaction: StandardTrainingTransaction,
    transactions: Sequence[StandardTrainingTransaction],
    collector: object,
    trainer: object,
    checkpoint_manager: object,
    checkpoint_receipt_index: object,
    journal: object,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Stage6Config,
    normalization_stats: Mapping[str, object],
    checkpoint_metadata: Mapping[str, object],
    journal_bindings: Mapping[str, object],
    validation_evaluator: Callable[[], object] | None,
    eval_binding_state_provider: Callable[[], Mapping[str, bytes]] | None,
    continue_from_current_state: bool,
    historical_immutable_bindings: Mapping[str, object] | None = None,
    current_binding_first_transaction_key: str | None = None,
    immutable_binding_epochs: (
        Sequence[tuple[str, Mapping[str, object]]] | None
    ) = None,
    resource_guard: Callable[[str], None] | None = None,
    capability_guard: Callable[[str], None] | None = None,
    persist_post_resource: Callable[[StandardTransactionCheckpoint], None]
    | None = None,
    accept_post_resource: Callable[[StandardTransactionCheckpoint], None]
    | None = None,
    require_dual_scan: bool = False,
) -> StandardUpdateResult:
    """执行一个 collect/update/eval/checkpoint/journal 原子事务。"""

    expected_metadata = {
        "best_record",
        "versions",
        "top_m_config",
        "scale_profile",
        "training_config",
        "config_sha256",
        "lineage",
        "safety_contract",
    }
    if (
        not isinstance(transaction, StandardTrainingTransaction)
        or not isinstance(config, Stage6Config)
        or set(checkpoint_metadata) != expected_metadata
        or type(continue_from_current_state) is not bool
        or type(require_dual_scan) is not bool
        or len(
            {
                resource_guard is None,
                persist_post_resource is None,
                accept_post_resource is None,
            }
        )
        != 1
        or (resource_guard is not None and not callable(resource_guard))
        or ((capability_guard is None) != (resource_guard is None))
        or (
            capability_guard is not None
            and not callable(capability_guard)
        )
        or (
            persist_post_resource is not None
            and not callable(persist_post_resource)
        )
        or (
            accept_post_resource is not None
            and not callable(accept_post_resource)
        )
    ):
        raise StandardTrainingError("Standard update transaction binding drifted")
    expected_contract = SafetyContract.from_stage6_config(config)
    expected_safety = expected_contract.to_dict()
    metadata_contract = checkpoint_metadata["safety_contract"]
    training_config = checkpoint_metadata["training_config"]
    lineage = checkpoint_metadata["lineage"]
    config_sha256 = checkpoint_metadata["config_sha256"]
    preserve_prepared = False
    try:
        safety_binding_valid = (
            isinstance(metadata_contract, SafetyContract)
            and metadata_contract == expected_contract
            and SafetyContract.from_binding(
                metadata_contract.binding(config_sha256=config_sha256),
                expected_config_sha256=config_sha256,
            )
            == expected_contract
        )
    except (TypeError, ValueError):
        safety_binding_valid = False
    if (
        not safety_binding_valid
        or not isinstance(training_config, Mapping)
        or training_config.get("safety") != expected_safety
        or not isinstance(lineage, Mapping)
        or lineage.get("safety_contract_sha256") != expected_contract.sha256
    ):
        raise StandardTrainingError("checkpoint safety metadata drifted")
    collect = getattr(collector, "collect", None)
    update = getattr(trainer, "update", None)
    prepare_complete = getattr(checkpoint_manager, "prepare_complete", None)
    publish_prepared = getattr(checkpoint_manager, "publish_prepared", None)
    abort_prepared = getattr(checkpoint_manager, "abort_prepared", None)
    append_receipt = getattr(checkpoint_receipt_index, "append_once", None)
    verify_receipts = getattr(checkpoint_receipt_index, "verify", None)
    vector_env = getattr(collector, "vector_env", None)
    capture_states = getattr(vector_env, "capture_states", None)
    if not all(
        callable(value)
        for value in (
            collect,
            update,
            prepare_complete,
            publish_prepared,
            abort_prepared,
            append_receipt,
            verify_receipts,
            capture_states,
        )
    ):
        raise StandardTrainingError("Standard update collaborator interface drifted")

    policy_sha256_before_collection = policy_state_sha256(policy)
    collection_kwargs: dict[str, object] = {
        "continue_from_current_state": continue_from_current_state,
    }
    if resource_guard is not None:
        collection_kwargs["resource_guard"] = resource_guard
    if capability_guard is not None:
        capability_guard("collection:before")
    collection = collect(**collection_kwargs)
    if capability_guard is not None:
        capability_guard("collection:after")
    batch = getattr(collection, "batch", None)
    vector_states = tuple(getattr(collection, "vector_env_states", ()))
    collection_audit = validate_standard_collection_audit(
        getattr(collection, "audit", None),
        expected_device=config.device,
        expected_policy_sha256=policy_sha256_before_collection,
        expected_inference_pid=os.getpid(),
        require_dual_scan=require_dual_scan,
    )
    trainable_count = collection_audit["trainable_transition_count"]
    if (
        getattr(batch, "size", None) != config.rollout.batch_size
        or trainable_count != config.rollout.batch_size
        or len(vector_states) != config.rollout.num_envs
        or any(not isinstance(state, Mapping) for state in vector_states)
    ):
        raise StandardTrainingError("collector did not produce true 8x128 transitions")
    if capability_guard is not None:
        capability_guard("update:before")
    update_metrics = (
        update(batch, resource_guard=resource_guard)
        if resource_guard is not None
        else update(batch)
    )
    if capability_guard is not None:
        capability_guard("update:after")
    update_metric_record = _jsonable_runtime_value(update_metrics)
    if not isinstance(update_metric_record, Mapping):
        raise StandardTrainingError("PPOTrainer update metrics are not auditable")
    update_step = (
        update_metric_record.get("update_step")
    )
    if update_step != transaction.update:
        raise StandardTrainingError("PPOTrainer update step drifted")
    policy_sha256_after_update = policy_state_sha256(policy)
    if (
        update_metric_record.get("policy_state_sha256_before")
        != collection_audit["policy_state_sha256"]
        or update_metric_record.get("policy_state_sha256_after")
        != policy_sha256_after_update
    ):
        raise StandardTrainingError("collector/trainer policy lineage drifted")

    validation_result: object | None = None
    isolation: dict[str, object] | None = None
    selected_best = dict(checkpoint_metadata["best_record"])  # type: ignore[arg-type]
    if transaction.validation_episodes:
        if not callable(validation_evaluator) or not callable(eval_binding_state_provider):
            raise StandardTrainingError("validation transaction lacks read-only evaluator")
        if capability_guard is not None:
            capability_guard("validation:before")
        validation_result, isolation = run_eval_only_transaction(
            evaluator=validation_evaluator,
            policy=policy,
            optimizer=optimizer,
            normalization_stats=normalization_stats,
            runtime_state_provider=capture_states,
            binding_state_provider=eval_binding_state_provider,
        )
        if capability_guard is not None:
            capability_guard("validation:after")
        episode_count = (
            validation_result.get("episode_count")
            if isinstance(validation_result, Mapping)
            else getattr(validation_result, "metrics", {}).get("episode_count")
        )
        if episode_count != transaction.validation_episodes:
            raise StandardTrainingError("validation episode count drifted")
        selected_best = select_validation_checkpoint_best(
            seed=transaction.seed,
            update=transaction.update,
            validation_metrics=_evaluation_metrics_mapping(validation_result),
            previous_best=selected_best,
        )
    elif validation_evaluator is not None or eval_binding_state_provider is not None:
        raise StandardTrainingError("non-validation update received evaluator")

    eval_metrics = _evaluation_metrics_mapping(validation_result)
    if resource_guard is not None:
        resource_guard("checkpoint:before-publication")
    if capability_guard is not None:
        capability_guard("checkpoint:before-prepare")
    prepared = prepare_complete(
        policy=policy,
        optimizer=optimizer,
        update_step=transaction.update,
        normalization_stats=normalization_stats,
        scenario_sampler_state={
            "workers": [dict(state).get("sampler_state", {}) for state in vector_states]
        },
        vector_env_states=vector_states,
        best_record=selected_best,
        versions=checkpoint_metadata["versions"],
        top_m_config=checkpoint_metadata["top_m_config"],
        scale_profile=checkpoint_metadata["scale_profile"],
        training_config=checkpoint_metadata["training_config"],
        config_sha256=checkpoint_metadata["config_sha256"],
        lineage=checkpoint_metadata["lineage"],
        eval_metrics={
            "validation": eval_metrics,
            "update": dict(update_metric_record),
            "eval_isolation": isolation or {},
            "collection_audit": dict(collection_audit),
        },
        safety_contract=metadata_contract,
    )
    try:
        checkpoint_sha256 = getattr(prepared, "checkpoint_sha256", None)
        complete_marker_sha256 = getattr(
            prepared,
            "complete_marker_sha256",
            None,
        )
        policy_sha256 = getattr(prepared, "policy_state_sha256", None)
        if (
            not _is_sha256(checkpoint_sha256)
            or not _is_sha256(complete_marker_sha256)
            or not _is_sha256(policy_sha256)
            or policy_sha256 != policy_sha256_after_update
        ):
            raise StandardTrainingError("prepared checkpoint identity drifted")
        checkpoint = StandardTransactionCheckpoint(
            transaction_key=transaction.key,
            seed=transaction.seed,
            update=transaction.update,
            checkpoint_sha256=checkpoint_sha256,
            complete_marker_sha256=complete_marker_sha256,
            policy_state_sha256=policy_sha256,
        )
        bindings = dict(journal_bindings)
        bindings["checkpoint_sha256"] = checkpoint.checkpoint_sha256
        if persist_post_resource is not None:
            persist_post_resource(checkpoint)
            if resource_guard is not None:
                resource_guard("checkpoint:before-resource-acceptance")
            preserve_prepared = True
            assert accept_post_resource is not None
            accept_post_resource(checkpoint)
        if capability_guard is not None:
            capability_guard("checkpoint:before-publication")
        receipt = publish_prepared(prepared)
        marker_path = Path(getattr(receipt, "complete_marker_path", ""))
        if (
            getattr(receipt, "checkpoint_sha256", None)
            != checkpoint.checkpoint_sha256
            or getattr(receipt, "policy_state_sha256", None)
            != checkpoint.policy_state_sha256
            or not marker_path.is_file()
            or hashlib.sha256(marker_path.read_bytes()).hexdigest()
            != checkpoint.complete_marker_sha256
        ):
            raise StandardTrainingError("published checkpoint receipt drifted")
        if capability_guard is not None:
            capability_guard("checkpoint:before-receipt")
        append_receipt(checkpoint)
        receipt_rows = tuple(verify_receipts())
        if capability_guard is not None:
            capability_guard("checkpoint:before-journal")
        journal_records = commit_checkpointed_transaction(
            journal=journal,
            transactions=transactions,
            transaction=transaction,
            checkpoint=checkpoint,
            bindings=bindings,
            receipt_rows=receipt_rows,
            historical_immutable_bindings=historical_immutable_bindings,
            current_binding_first_transaction_key=(
                current_binding_first_transaction_key
            ),
            immutable_binding_epochs=immutable_binding_epochs,
        )
    except BaseException:
        if not preserve_prepared:
            abort_prepared(prepared)
        raise
    return StandardUpdateResult(
        transaction=transaction,
        update_metrics=update_metrics,
        validation_result=validation_result,
        eval_isolation_audit=isolation,
        best_record=selected_best,
        checkpoint=checkpoint,
        journal_records=journal_records,
        collection_audit=collection_audit,
    )


def select_validation_checkpoint_best(
    *,
    seed: int,
    update: int,
    validation_metrics: Mapping[str, object],
    previous_best: Mapping[str, object],
) -> dict[str, object]:
    if validation_metrics.get("episode_count") != 16:
        raise StandardTrainingError("validation best requires 16 episodes")
    success = validation_metrics.get("success_rate_under_fixed_step_budget")
    coverage = validation_metrics.get("mean_final_coverage")
    if not isinstance(success, (int, float)) or not isinstance(coverage, (int, float)):
        raise StandardTrainingError("validation best metrics are missing")
    current = ValidationRecord(
        seed=seed,
        update=update,
        success_rate_under_fixed_step_budget=float(success),
        mean_final_coverage=float(coverage),
        checkpoint_ref=f"update-{update:08d}",
    )
    if not previous_best:
        return current.to_dict()
    try:
        prior = ValidationRecord(
            seed=int(previous_best["seed"]),
            update=int(previous_best["update"]),
            success_rate_under_fixed_step_budget=float(
                previous_best["success_rate_under_fixed_step_budget"]
            ),
            mean_final_coverage=float(previous_best["mean_final_coverage"]),
            checkpoint_ref=str(previous_best["checkpoint_ref"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StandardTrainingError("previous validation best record drifted") from exc
    if prior.seed != seed:
        raise StandardTrainingError("validation best crossed seed boundary")
    return select_seed_best((prior, current)).to_dict()


def _evaluation_metrics_mapping(value: object | None) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): _jsonable_runtime_value(item) for key, item in value.items()}
    metrics = getattr(value, "metrics", None)
    if isinstance(metrics, Mapping):
        return {str(key): _jsonable_runtime_value(item) for key, item in metrics.items()}
    raise StandardTrainingError("evaluation result metrics drifted")


def _validation_trace_line_count(payload: bytes) -> int:
    return len(payload.splitlines())


def _validation_trace_binding_from_bytes(payload: bytes) -> dict[str, object]:
    return {
        "schema_version": _VALIDATION_TRACE_BINDING_SCHEMA_VERSION,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "line_count": _validation_trace_line_count(payload),
    }


def _validated_validation_trace_binding(
    value: object,
    *,
    episode_count: int,
    payload: bytes | None = None,
) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {"schema_version", "sha256", "size_bytes", "line_count"}
        or value.get("schema_version") != _VALIDATION_TRACE_BINDING_SCHEMA_VERSION
        or not _is_sha256(value.get("sha256"))
        or type(value.get("size_bytes")) is not int
        or int(value["size_bytes"]) < 0
        or type(value.get("line_count")) is not int
        or int(value["line_count"]) != episode_count
    ):
        raise StandardTrainingError("validation trace binding drifted")
    binding = {
        "schema_version": str(value["schema_version"]),
        "sha256": str(value["sha256"]),
        "size_bytes": int(value["size_bytes"]),
        "line_count": int(value["line_count"]),
    }
    if payload is not None:
        actual = _validation_trace_binding_from_bytes(payload)
        if actual != binding:
            raise StandardTrainingError("validation trace binding drifted")
    return binding


def _validation_trace_binding_from_metrics(
    loaded: object,
    *,
    episode_count: int,
) -> dict[str, object]:
    metrics = getattr(loaded, "eval_metrics", None)
    validation = metrics.get("validation") if isinstance(metrics, Mapping) else None
    if not isinstance(validation, Mapping):
        raise StandardTrainingError("validation trace binding is missing")
    return _validated_validation_trace_binding(
        validation.get("validation_trace_binding"),
        episode_count=episode_count,
    )


def _durable_identity_matches_secure_read(
    secure_read: object,
    durable_identity: object,
) -> bool:
    stat_identity = getattr(secure_read, "stat_identity", None)
    link_count = getattr(secure_read, "link_count", None)
    return (
        isinstance(stat_identity, tuple)
        and len(stat_identity) == 5
        and isinstance(durable_identity, tuple)
        and len(durable_identity) == 6
        and durable_identity[0] == stat_identity[0]
        and durable_identity[1] == stat_identity[1]
        and durable_identity[2] == stat.S_IFMT(int(stat_identity[2]))
        and durable_identity[3] == stat_identity[3]
        and durable_identity[4] == stat_identity[4]
        and durable_identity[5] == link_count
    )


FINAL_EVALUATION_METRICS = (
    "success_rate_under_fixed_step_budget",
    "mean_final_coverage",
    "steps_to_99_success_only",
    "path_length_to_99_success_only",
    "coverage_auc_over_steps",
    "coverage_per_meter",
    "invalid_action_count_mean",
    "planner_failure_count_mean",
    "safety_violation_count",
)


def verify_standard_final_evaluation_artifacts(
    *,
    trace_path: str | Path,
    summary_path: str | Path,
    commit_path: str | Path,
    split: str,
    method: str,
    config_sha256: str,
    safety_contract: SafetyContract,
    checkpoint_sha256: str,
    policy_state_sha256: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Recompute one final evaluation from its canonical 64-episode trace."""

    try:
        trace_candidate = lexical_absolute(trace_path)
        summary_candidate = lexical_absolute(summary_path)
        commit_candidate = lexical_absolute(commit_path)
        artifact_parent = require_plain_path(
            trace_candidate.parent,
            leaf_kind="directory",
            label="final evaluation artifact parent",
        )
        trace = require_plain_path(
            trace_candidate,
            base=artifact_parent,
            leaf_kind="file",
            label="final evaluation trace",
        )
        summary = require_plain_path(
            summary_candidate,
            base=artifact_parent,
            leaf_kind="file",
            label="final evaluation summary",
        )
        commit = require_plain_path(
            commit_candidate,
            base=artifact_parent,
            leaf_kind="file",
            label="final evaluation commit",
        )
    except (OSError, PathSecurityError) as exc:
        raise StandardTrainingError(
            "final evaluation artifact contains a link or reparse point"
        ) from exc
    if (
        not trace.is_file()
        or not summary.is_file()
        or not commit.is_file()
    ):
        raise StandardTrainingError("final evaluation artifact identity drifted")
    return verify_standard_final_evaluation_artifacts_from_bytes(
        trace_bytes=trace.read_bytes(),
        trace_name=trace.name,
        summary_bytes=summary.read_bytes(),
        summary_name=summary.name,
        commit_bytes=commit.read_bytes(),
        commit_name=commit.name,
        split=split,
        method=method,
        config_sha256=config_sha256,
        safety_contract=safety_contract,
        checkpoint_sha256=checkpoint_sha256,
        policy_state_sha256=policy_state_sha256,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def verify_standard_final_evaluation_artifacts_from_bytes(
    *,
    trace_bytes: bytes,
    trace_name: str,
    summary_bytes: bytes,
    summary_name: str,
    commit_bytes: bytes,
    commit_name: str,
    split: str,
    method: str,
    config_sha256: str,
    safety_contract: SafetyContract,
    checkpoint_sha256: str,
    policy_state_sha256: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Replay one final evaluation from an already-bound exact byte snapshot."""

    from lunar_exploration_ppo.eval.metrics import (
        EPISODE_FIELDS,
        EpisodeResult,
        summarize_episodes,
    )
    from lunar_exploration_ppo.eval.standard import (
        STANDARD_EVALUATION_METHODS,
        validate_standard_fairness_audit,
    )
    from lunar_exploration_ppo.ppo.collector import (
        PLANNER_FAILURE_REASONS,
        validate_reset_diagnostics_payload,
    )

    names = (trace_name, summary_name, commit_name)
    if (
        any(
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or ":" in name
            for name in names
        )
        or any(type(payload) is not bytes or not payload for payload in (
            trace_bytes,
            summary_bytes,
            commit_bytes,
        ))
        or split not in {"test", "unseen"}
        or method not in STANDARD_EVALUATION_METHODS
        or not all(
            _is_sha256(value)
            for value in (config_sha256, checkpoint_sha256, policy_state_sha256)
        )
        or type(bootstrap_resamples) is not int
        or bootstrap_resamples <= 0
        or type(bootstrap_seed) is not int
    ):
        raise StandardTrainingError("final evaluation artifact identity drifted")
    lines = trace_bytes.splitlines(keepends=True)
    if len(lines) != 64:
        raise StandardTrainingError("final evaluation trace count drifted")
    episodes = []
    episode_action_counts: list[int] = []
    diagnostic_trace_mode: bool | None = None
    planner_failure_counts = {
        reason: 0 for reason in PLANNER_FAILURE_REASONS
    }
    dual_scan_episode_count = 0
    try:
        for line in lines:
            row = json.loads(line.decode("utf-8"))
            canonical = (
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            row_fields = set(row) if isinstance(row, dict) else set()
            legacy_fields = {*EPISODE_FIELDS, "steps_executed"}
            diagnostics_fields = {
                "reset_diagnostics",
                "planner_failure_counts",
            }
            has_diagnostics = row_fields == legacy_fields | diagnostics_fields
            if diagnostic_trace_mode is None:
                diagnostic_trace_mode = has_diagnostics
            if (
                line != canonical
                or not isinstance(row, dict)
                or row_fields not in (
                    legacy_fields,
                    legacy_fields | diagnostics_fields,
                )
                or diagnostic_trace_mode is not has_diagnostics
                or row.get("method") != method
                or row.get("scale_profile") != "Standard v1"
                or not isinstance(row.get("coverage_curve"), list)
                or len(row["coverage_curve"]) != 129
                or type(row.get("steps_executed")) is not int
                or not 0 <= row["steps_executed"] <= 128
            ):
                raise ValueError("episode row schema drifted")
            if has_diagnostics:
                reset_diagnostics = validate_reset_diagnostics_payload(
                    row["reset_diagnostics"],
                    require_dual_scan=True,
                )
                counts = row["planner_failure_counts"]
                if (
                    not isinstance(counts, Mapping)
                    or set(counts) != set(PLANNER_FAILURE_REASONS)
                    or any(
                        type(counts[reason]) is not int
                        or int(counts[reason]) < 0
                        for reason in PLANNER_FAILURE_REASONS
                    )
                    or sum(int(counts[reason]) for reason in PLANNER_FAILURE_REASONS)
                    != row.get("planner_failure_count")
                ):
                    raise ValueError("planner failure classification drifted")
                for reason in PLANNER_FAILURE_REASONS:
                    planner_failure_counts[reason] += int(counts[reason])
                if reset_diagnostics["scan_order"] == [
                    "reset_local_safety",
                    "reset_exploration",
                ]:
                    dual_scan_episode_count += 1
            episode_action_counts.append(int(row["steps_executed"]))
            episodes.append(
                EpisodeResult(
                    **{
                        **{name: row[name] for name in EPISODE_FIELDS},
                        "coverage_curve": tuple(row["coverage_curve"]),
                    }
                )
            )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise StandardTrainingError("final evaluation episode trace drifted") from exc
    schedule = tuple(
        (
            episode.scenario_key,
            episode.scenario_seed,
            episode.terrain_seed,
            episode.start_pose_seed,
            episode.evaluation_seed,
        )
        for episode in episodes
    )
    if len(set(schedule)) != 64:
        raise StandardTrainingError("final evaluation schedule identity drifted")

    try:
        completion = json.loads(summary_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StandardTrainingError("final evaluation completion drifted") from exc
    expected_completion_fields = {
        "schema_version",
        "split",
        "method",
        "episode_count",
        "trace",
        "config_sha256",
        "checkpoint_sha256",
        "policy_state_sha256",
        "result",
    }
    if (
        not isinstance(completion, dict)
        or ArtifactStore.canonical_json_bytes(completion) != summary_bytes
        or set(completion) != expected_completion_fields
        or completion.get("schema_version") != "stage6_final_eval_completion/v1"
        or completion.get("split") != split
        or completion.get("method") != method
        or completion.get("episode_count") != 64
        or completion.get("trace")
        != {
            "path": trace_name,
            "sha256": hashlib.sha256(trace_bytes).hexdigest(),
            "size_bytes": len(trace_bytes),
        }
        or completion.get("config_sha256") != config_sha256
        or completion.get("checkpoint_sha256") != checkpoint_sha256
        or completion.get("policy_state_sha256") != policy_state_sha256
        or not isinstance(completion.get("result"), dict)
    ):
        raise StandardTrainingError("final evaluation completion binding drifted")
    try:
        commit_record = json.loads(commit_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StandardTrainingError("final evaluation commit drifted") from exc
    if (
        not isinstance(commit_record, dict)
        or ArtifactStore.canonical_json_bytes(commit_record) != commit_bytes
        or set(commit_record)
        != {
            "schema_version",
            "split",
            "method",
            "attempt",
            "config_sha256",
            "checkpoint_sha256",
            "policy_state_sha256",
            "trace",
            "summary",
        }
        or commit_record.get("schema_version") != "stage6_final_eval_commit/v1"
        or commit_record.get("split") != split
        or commit_record.get("method") != method
        or type(commit_record.get("attempt")) is not int
        or int(commit_record["attempt"]) <= 0
        or commit_record.get("config_sha256") != config_sha256
        or commit_record.get("checkpoint_sha256") != checkpoint_sha256
        or commit_record.get("policy_state_sha256") != policy_state_sha256
        or commit_record.get("trace")
        != {
            "path": trace_name,
            "sha256": hashlib.sha256(trace_bytes).hexdigest(),
            "size_bytes": len(trace_bytes),
        }
        or commit_record.get("summary")
        != {
            "path": summary_name,
            "sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "size_bytes": len(summary_bytes),
        }
    ):
        raise StandardTrainingError("final evaluation commit binding drifted")
    result = dict(completion["result"])
    recomputed_metrics, recomputed_bootstrap = summarize_episodes(
        tuple(episodes),
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    fairness = result.get("fairness_audit")
    try:
        validated_fairness = validate_standard_fairness_audit(
            fairness,  # type: ignore[arg-type]
            episode_count=64,
            parent_pid=None,
            safety_contract=safety_contract,
            config_sha256=config_sha256,
        )
    except Exception as exc:
        raise StandardTrainingError("final evaluation fairness drifted") from exc
    shared_environment = validated_fairness.get("shared_environment_contract")
    fairness_seed_schedule = (
        shared_environment.get("scenario_seed_schedule")
        if isinstance(shared_environment, Mapping)
        else None
    )
    expected_seed_schedule: list[tuple[str, int, int, int, int]] = []
    seed_schedule_fields = {
        "episode_index",
        "scenario_id",
        "scenario_seed",
        "terrain_seed",
        "start_pose_seed",
        "evaluation_seed",
    }
    if not isinstance(fairness_seed_schedule, list) or len(fairness_seed_schedule) != 64:
        raise StandardTrainingError("final evaluation trace seed schedule drifted")
    for episode_index, row in enumerate(fairness_seed_schedule):
        if (
            not isinstance(row, Mapping)
            or set(row) != seed_schedule_fields
            or type(row.get("episode_index")) is not int
            or row.get("episode_index") != episode_index
            or not isinstance(row.get("scenario_id"), str)
            or any(
                type(row.get(field_name)) is not int
                for field_name in (
                    "scenario_seed",
                    "terrain_seed",
                    "start_pose_seed",
                    "evaluation_seed",
                )
            )
        ):
            raise StandardTrainingError("final evaluation trace seed schedule drifted")
        expected_seed_schedule.append(
            (
                str(row["scenario_id"]),
                int(row["scenario_seed"]),
                int(row["terrain_seed"]),
                int(row["start_pose_seed"]),
                int(row["evaluation_seed"]),
            )
        )
    if tuple(expected_seed_schedule) != schedule:
        raise StandardTrainingError("final evaluation trace seed schedule drifted")
    expected_fairness_policy_sha256 = (
        policy_state_sha256 if method == "ppo_policy" else None
    )
    if (
        result.get("episode_count") != 64
        or result.get("metrics") != recomputed_metrics
        or result.get("bootstrap_audit") != recomputed_bootstrap
        or fairness != validated_fairness
        or fairness.get("scenario_schedule")
        != [episode.scenario_key for episode in episodes]
        or fairness.get("evaluation_seeds")
        != [episode.evaluation_seed for episode in episodes]
        or fairness.get("episode_action_counts") != episode_action_counts
        or fairness.get("selected_action_count") != sum(episode_action_counts)
        or fairness.get("policy_state_sha256_before")
        != expected_fairness_policy_sha256
        or fairness.get("policy_state_sha256_after")
        != expected_fairness_policy_sha256
    ):
        raise StandardTrainingError("final evaluation result did not match trace")
    coverage_curve = tuple(
        float(
            np.mean(
                [episode.coverage_curve[step] for episode in episodes],
                dtype=np.float64,
            )
        )
        for step in range(129)
    )
    termination_classification: dict[str, int] = {}
    for episode in episodes:
        termination_classification[episode.termination_reason] = (
            termination_classification.get(episode.termination_reason, 0) + 1
        )
    return {
        "result": result,
        "schedule": schedule,
        "coverage_curve": coverage_curve,
        "termination_classification": dict(sorted(termination_classification.items())),
        "safety_violation_count": sum(
            episode.safety_violation_count for episode in episodes
        ),
        "planner_failure_counts": (
            dict(planner_failure_counts)
            if diagnostic_trace_mode is True
            else None
        ),
        "reset_scan_audit": (
            {
                "schema_version": "stage6_reset_scan_replay/v1",
                "episode_count": len(episodes),
                "scan_order": [
                    "reset_local_safety",
                    "reset_exploration",
                ],
                "dual_scan_episode_count": dual_scan_episode_count,
            }
            if diagnostic_trace_mode is True
            else None
        ),
        "episode_count": len(episodes),
        "commit_attempt": int(commit_record["attempt"]),
    }


def build_standard_final_aggregate_artifacts(
    verified: Mapping[str, Mapping[str, object]],
    isolation: Mapping[str, Mapping[str, object]],
    *,
    safety_contract: SafetyContract,
    config_sha256: str,
) -> dict[str, object]:
    """Build canonical final tables and hard-gate audits from trace replay."""

    from lunar_exploration_ppo.eval.standard import (
        STANDARD_EVALUATION_METHODS,
        StandardEvaluationError,
        validate_standard_fairness_cohort,
    )
    from lunar_exploration_ppo.ppo.collector import PLANNER_FAILURE_REASONS

    expected_keys = {
        f"{split}:{method}"
        for split in ("test", "unseen")
        for method in STANDARD_EVALUATION_METHODS
    }
    if set(verified) != expected_keys or set(isolation) != expected_keys:
        raise StandardTrainingError("final evaluation aggregate key set drifted")
    if any(
        not isinstance(value, Mapping) or value.get("passed") is not True
        for value in isolation.values()
    ):
        raise StandardTrainingError("final evaluation isolation audit drifted")

    comparison_stream = io.StringIO(newline="")
    comparison_writer = csv.writer(comparison_stream, lineterminator="\n")
    comparison_writer.writerow(
        [
            "split",
            "method",
            "episode_count",
            *FINAL_EVALUATION_METRICS,
            *(f"{name}_ci95_low" for name in FINAL_EVALUATION_METRICS),
            *(f"{name}_ci95_high" for name in FINAL_EVALUATION_METRICS),
        ]
    )
    curves_stream = io.StringIO(newline="")
    curves_writer = csv.writer(curves_stream, lineterminator="\n")
    curves_writer.writerow(["split", "method", "step", "mean_coverage"])
    fairness_rows: list[dict[str, object]] = []
    fairness_by_split: dict[str, list[Mapping[str, object]]] = {
        "test": [],
        "unseen": [],
    }
    per_evaluation_failures: list[dict[str, object]] = []
    total_terminations: dict[str, int] = {}
    total_safety = 0
    total_planner_failures = {
        reason: 0 for reason in PLANNER_FAILURE_REASONS
    }
    diagnostics_mode: bool | None = None
    dual_scan_episode_count = 0
    split_schedules: dict[str, object] = {}

    for split in ("test", "unseen"):
        for method in STANDARD_EVALUATION_METHODS:
            key = f"{split}:{method}"
            replay = verified[key]
            result = replay.get("result")
            if (
                not isinstance(result, Mapping)
                or replay.get("episode_count") != 64
                or not isinstance(replay.get("schedule"), tuple)
                or not isinstance(replay.get("coverage_curve"), tuple)
                or len(replay["coverage_curve"]) != 129  # type: ignore[arg-type]
            ):
                raise StandardTrainingError("final evaluation replay aggregate drifted")
            if split not in split_schedules:
                split_schedules[split] = replay["schedule"]
            elif replay["schedule"] != split_schedules[split]:
                raise StandardTrainingError("final evaluation shared schedule drifted")
            metrics = result.get("metrics")
            bootstrap = result.get("bootstrap_audit")
            intervals = bootstrap.get("metrics") if isinstance(bootstrap, Mapping) else None
            if (
                not isinstance(metrics, Mapping)
                or set(FINAL_EVALUATION_METRICS) - set(metrics)
                or not isinstance(intervals, Mapping)
                or set(FINAL_EVALUATION_METRICS) - set(intervals)
            ):
                raise StandardTrainingError("final evaluation metric schema drifted")
            metric_values: list[object] = []
            lows: list[object] = []
            highs: list[object] = []
            success_rate = metrics["success_rate_under_fixed_step_budget"]
            for name in FINAL_EVALUATION_METRICS:
                value = metrics[name]
                interval = intervals[name]
                if not isinstance(interval, Mapping) or interval.get("estimate") != value:
                    raise StandardTrainingError("final bootstrap metric binding drifted")
                low = interval.get("ci95_low")
                high = interval.get("ci95_high")
                allows_none = name in {
                    "steps_to_99_success_only",
                    "path_length_to_99_success_only",
                } and success_rate == 0.0
                if value is None:
                    if not allows_none or low is not None or high is not None:
                        raise StandardTrainingError("final metric missing unexpectedly")
                elif (
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not isinstance(low, (int, float))
                    or not math.isfinite(float(low))
                    or not isinstance(high, (int, float))
                    or not math.isfinite(float(high))
                ):
                    raise StandardTrainingError("final metric is nonfinite")
                metric_values.append("" if value is None else value)
                lows.append("" if low is None else low)
                highs.append("" if high is None else high)
            comparison_writer.writerow(
                [split, method, 64, *metric_values, *lows, *highs]
            )
            coverage_curve = tuple(float(value) for value in replay["coverage_curve"])
            if (
                not all(math.isfinite(value) for value in coverage_curve)
                or any(
                    right < left
                    for left, right in zip(coverage_curve, coverage_curve[1:])
                )
                or coverage_curve[-1] != metrics["mean_final_coverage"]
            ):
                raise StandardTrainingError("final coverage curve drifted")
            for step, value in enumerate(coverage_curve):
                curves_writer.writerow([split, method, step, value])
            fairness = result.get("fairness_audit")
            if not isinstance(fairness, Mapping) or fairness.get("passed") is not True:
                raise StandardTrainingError("final fairness result drifted")
            fairness_rows.append({"split": split, "method": method, **dict(fairness)})
            fairness_by_split[split].append(fairness)
            classifications = replay.get("termination_classification")
            safety = replay.get("safety_violation_count")
            if (
                not isinstance(classifications, Mapping)
                or any(type(value) is not int or value < 0 for value in classifications.values())
                or sum(classifications.values()) != 64
                or type(safety) is not int
                or safety < 0
                or metrics["safety_violation_count"] != safety
            ):
                raise StandardTrainingError("final failure classification drifted")
            planner_failures = replay.get("planner_failure_counts")
            reset_scan = replay.get("reset_scan_audit")
            has_diagnostics = (
                isinstance(planner_failures, Mapping)
                and isinstance(reset_scan, Mapping)
            )
            if diagnostics_mode is None:
                diagnostics_mode = has_diagnostics
            if diagnostics_mode is not has_diagnostics:
                raise StandardTrainingError(
                    "final diagnostics evidence mode drifted"
                )
            if has_diagnostics:
                assert isinstance(planner_failures, Mapping)
                assert isinstance(reset_scan, Mapping)
                if (
                    set(planner_failures) != set(PLANNER_FAILURE_REASONS)
                    or any(
                        type(planner_failures[reason]) is not int
                        or int(planner_failures[reason]) < 0
                        for reason in PLANNER_FAILURE_REASONS
                    )
                    or not math.isclose(
                        sum(
                            int(planner_failures[reason])
                            for reason in PLANNER_FAILURE_REASONS
                        )
                        / 64.0,
                        float(metrics["planner_failure_count_mean"]),
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                ):
                    raise StandardTrainingError(
                        "final planner failure classification drifted"
                    )
                if dict(reset_scan) != {
                    "schema_version": "stage6_reset_scan_replay/v1",
                    "episode_count": 64,
                    "scan_order": [
                        "reset_local_safety",
                        "reset_exploration",
                    ],
                    "dual_scan_episode_count": 64,
                }:
                    raise StandardTrainingError(
                        "final reset scan classification drifted"
                    )
                for reason in PLANNER_FAILURE_REASONS:
                    total_planner_failures[reason] += int(
                        planner_failures[reason]
                    )
                dual_scan_episode_count += int(
                    reset_scan["dual_scan_episode_count"]
                )
            for reason, count in classifications.items():
                total_terminations[str(reason)] = (
                    total_terminations.get(str(reason), 0) + int(count)
                )
            total_safety += safety
            failure_row = {
                "split": split,
                "method": method,
                "episode_count": 64,
                "safety_violation_count": safety,
                "termination_classification": dict(classifications),
            }
            if has_diagnostics:
                failure_row["planner_failure_counts"] = dict(
                    planner_failures
                )
                failure_row["reset_scan_audit"] = dict(reset_scan)
            per_evaluation_failures.append(failure_row)

    try:
        fairness_cohorts = {
            split: validate_standard_fairness_cohort(
                fairness_by_split[split],
                split=split,
                episode_count=64,
                safety_contract=safety_contract,
                config_sha256=config_sha256,
            )
            for split in ("test", "unseen")
        }
    except StandardEvaluationError as exc:
        raise StandardTrainingError("final fairness cohort drifted") from exc
    fairness_audit = {
        "schema_version": "stage6_final_fairness_audit/v2",
        "passed": (
            all(row.get("passed") is True for row in fairness_rows)
            and all(
                cohort.get("passed") is True
                for cohort in fairness_cohorts.values()
            )
        ),
        "method_count": len(fairness_rows),
        "methods": fairness_rows,
        "cohorts": fairness_cohorts,
        "eval_only_isolation": {key: dict(isolation[key]) for key in sorted(isolation)},
    }
    runtime_truth_like = 0
    frozen_truth_like = 0
    for row in fairness_rows:
        decision_audit = row.get("decision_audit")
        decision_schema = row.get("decision_input_schema")
        frozen_scan = (
            decision_schema.get("frozen_field_scan")
            if isinstance(decision_schema, Mapping)
            else None
        )
        runtime_count = (
            decision_audit.get("runtime_truth_like_selector_field_count")
            if isinstance(decision_audit, Mapping)
            else None
        )
        truth_like_fields = (
            frozen_scan.get("truth_like_fields")
            if isinstance(frozen_scan, Mapping)
            else None
        )
        if (
            type(runtime_count) is not int
            or runtime_count < 0
            or not isinstance(truth_like_fields, list)
            or any(not isinstance(field, str) for field in truth_like_fields)
        ):
            raise StandardTrainingError("final leakage provenance drifted")
        runtime_truth_like += runtime_count
        frozen_truth_like += len(truth_like_fields)
    leakage_audit = {
        "schema_version": "stage6_final_leakage_audit/v2",
        "passed": (
            fairness_audit["passed"] is True
            and runtime_truth_like == 0
            and frozen_truth_like == 0
        ),
        "runtime_truth_like_selector_field_count": runtime_truth_like,
        "frozen_truth_like_observation_field_count": frozen_truth_like,
    }
    episode_count = sum(
        int(row["episode_count"]) for row in per_evaluation_failures
    )
    safety_by_split = {
        split: sum(
            int(row["safety_violation_count"])
            for row in per_evaluation_failures
            if row["split"] == split
        )
        for split in ("test", "unseen")
    }
    safety_by_split_method = {
        f"{row['split']}:{row['method']}": int(row["safety_violation_count"])
        for row in per_evaluation_failures
    }
    machine_blocking_findings: list[dict[str, object]] = []
    adverse_findings = (
        []
        if total_safety == 0
        else [
            {
                "code": "synthetic_proxy_safety_termination_observed",
                "count": total_safety,
                "machine_blocking": False,
            }
        ]
    )
    reset_scan_audit = (
        {
            "schema_version": "stage6_final_reset_scan_audit/v1",
            "episode_count": episode_count,
            "scan_order": [
                "reset_local_safety",
                "reset_exploration",
            ],
            "dual_scan_episode_count": dual_scan_episode_count,
            "passed": dual_scan_episode_count == episode_count,
        }
        if diagnostics_mode is True
        else None
    )
    failure_audit = {
        "schema_version": (
            "stage6_failure_audit/v3"
            if diagnostics_mode is True
            else "stage6_failure_audit/v2"
        ),
        "passed": (
            leakage_audit["passed"] is True
            and episode_count == 640
            and (
                reset_scan_audit is None
                or reset_scan_audit["passed"] is True
            )
            and not machine_blocking_findings
        ),
        "episode_count": episode_count,
        "safety_violation_count": total_safety,
        "safety_violation_count_by_split": safety_by_split,
        "safety_violation_count_by_split_method": safety_by_split_method,
        "termination_classification": dict(sorted(total_terminations.items())),
        "evaluations": per_evaluation_failures,
        "machine_blocking_findings": machine_blocking_findings,
        "adverse_findings": adverse_findings,
        "hard_findings": machine_blocking_findings,
        "safety_interpretation": {
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "physical_obstacle_cells_written": False,
            "physical_safety_claim": False,
            "claim_scope": "no_physical_safety_claim/v1",
        },
    }
    if diagnostics_mode is True:
        failure_audit["planner_failure_counts"] = dict(
            total_planner_failures
        )
        failure_audit["reset_scan_audit"] = reset_scan_audit
    if (
        fairness_audit["passed"] is not True
        or leakage_audit["passed"] is not True
        or failure_audit["passed"] is not True
        or failure_audit["episode_count"] != 640
    ):
        raise StandardTrainingError("final machine acceptance failed")
    return {
        "comparison_csv": comparison_stream.getvalue().encode("utf-8"),
        "coverage_curves_csv": curves_stream.getvalue().encode("utf-8"),
        "fairness_audit": fairness_audit,
        "leakage_audit": leakage_audit,
        "failure_audit": failure_audit,
    }


def validate_stage6_resource_audit(
    rows: Sequence[Mapping[str, object]],
    transactions: Sequence[StandardTrainingTransaction],
    *,
    require_terminal: bool = True,
) -> dict[str, object]:
    """Validate attempt evidence and one accepted pre/post gate per atomic unit."""

    from lunar_exploration_ppo.eval.standard import STANDARD_EVALUATION_METHODS
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        ResourceLifecycleError,
        validate_resource_lifecycle_rows,
    )
    from lunar_exploration_ppo.utils.resources import (
        ResourceSnapshot,
        evaluate_resource_gates,
    )

    if type(require_terminal) is not bool:
        raise StandardTrainingError("resource terminal verification mode drifted")
    try:
        lifecycle = validate_resource_lifecycle_rows(
            rows,
            require_terminal=require_terminal,
        )
    except ResourceLifecycleError as exc:
        raise StandardTrainingError(
            f"resource lifecycle replay failed: {exc}"
        ) from exc
    transaction_rows = tuple(transactions)
    expected_updates = {transaction.key: transaction for transaction in transaction_rows}
    expected_finals = {
        f"final:{split}:{method}": (split, method)
        for split in ("test", "unseen")
        for method in STANDARD_EVALUATION_METHODS
    }
    expected_keys = {*expected_updates, *expected_finals}
    grouped: dict[str, list[tuple[int, Mapping[str, object]]]] = {
        key: [] for key in expected_keys
    }
    terminal_evidence: dict[str, object] | None = None

    def validate_resource(value: object) -> dict[str, object]:
        if not isinstance(value, Mapping) or set(value) != {
            "d_free_bytes",
            "rss_bytes",
            "peak_vram_bytes",
            "rss_source",
            "rss_root_pid",
            "rss_sample_count",
            "rss_latest_process_count",
            "rss_peak_process_count",
            "warnings",
            "hard_stops",
            "passed",
        }:
            raise StandardTrainingError("resource snapshot schema drifted")
        try:
            snapshot = ResourceSnapshot(
                d_free_bytes=value["d_free_bytes"],  # type: ignore[arg-type]
                rss_bytes=value["rss_bytes"],  # type: ignore[arg-type]
                peak_vram_bytes=value["peak_vram_bytes"],  # type: ignore[arg-type]
                rss_source=value["rss_source"],  # type: ignore[arg-type]
                rss_root_pid=value["rss_root_pid"],  # type: ignore[arg-type]
                rss_sample_count=value["rss_sample_count"],  # type: ignore[arg-type]
                rss_latest_process_count=value[  # type: ignore[arg-type]
                    "rss_latest_process_count"
                ],
                rss_peak_process_count=value[  # type: ignore[arg-type]
                    "rss_peak_process_count"
                ],
            )
        except (TypeError, ValueError) as exc:
            raise StandardTrainingError("resource snapshot value drifted") from exc
        decision = evaluate_resource_gates(snapshot, preflight=False)
        if snapshot.rss_source != "process_tree_lifecycle_peak_current_sum/v1":
            raise StandardTrainingError("resource RSS lifecycle provenance drifted")
        expected = {
            "d_free_bytes": snapshot.d_free_bytes,
            "rss_bytes": snapshot.rss_bytes,
            "peak_vram_bytes": snapshot.peak_vram_bytes,
            "rss_source": snapshot.rss_source,
            "rss_root_pid": snapshot.rss_root_pid,
            "rss_sample_count": snapshot.rss_sample_count,
            "rss_latest_process_count": snapshot.rss_latest_process_count,
            "rss_peak_process_count": snapshot.rss_peak_process_count,
            "warnings": list(decision.warnings),
            "hard_stops": list(decision.hard_stops),
            "passed": decision.passed,
        }
        if dict(value) != expected:
            raise StandardTrainingError("resource gate decision drifted")
        return expected

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise StandardTrainingError("resource audit row drifted")
        if row.get("phase") == "segment_start":
            continue
        key = row.get("transaction_key")
        if key == "terminal:formal_backend":
            if terminal_evidence is not None or index != len(rows) - 1:
                raise StandardTrainingError("terminal resource evidence order drifted")
            resource = validate_resource(row.get("resource"))
            if resource["passed"] is not True:
                raise StandardTrainingError("terminal resource hard stop")
            terminal_evidence = dict(row)
            continue
        if key not in expected_keys:
            raise StandardTrainingError("resource audit contains unknown atomic unit")
        identity = (
            {
                "kind": "update",
                "transaction_key": key,
                "seed": expected_updates[str(key)].seed,
                "update": expected_updates[str(key)].update,
            }
            if key in expected_updates
            else {
                "kind": "final_evaluation",
                "transaction_key": key,
                "split": expected_finals[str(key)][0],
                "method": expected_finals[str(key)][1],
            }
        )
        phase = row.get("phase")
        attempt = row.get("attempt")
        common = {
            "schema_version",
            *identity,
            "attempt",
            "phase",
            "accepted",
            "segment_id",
            "segment_index",
        }
        if type(attempt) is not int or attempt <= 0 or phase not in {
            "pre",
            "post",
            "accepted",
        }:
            raise StandardTrainingError("resource audit attempt/phase drifted")
        if any(row.get(name) != value for name, value in identity.items()):
            raise StandardTrainingError("resource audit identity drifted")
        if phase in {"pre", "post"}:
            if (
                set(row) != {*common, "resource"}
                or row.get("schema_version") != "stage6_resource_attempt/v1"
                or row.get("accepted") is not False
            ):
                raise StandardTrainingError("resource attempt schema drifted")
            validate_resource(row.get("resource"))
        else:
            is_update = key in expected_updates
            acceptance_fields = {*common, "pre", "post"}
            if is_update:
                acceptance_fields.add("checkpoint")
            if (
                set(row) != acceptance_fields
                or row.get("schema_version")
                != (
                    "stage6_resource_acceptance/v2"
                    if is_update
                    else "stage6_resource_acceptance/v1"
                )
                or row.get("accepted") is not True
            ):
                raise StandardTrainingError("resource acceptance schema drifted")
            pre = validate_resource(row.get("pre"))
            post = validate_resource(row.get("post"))
            if pre["passed"] is not True or post["passed"] is not True:
                raise StandardTrainingError("accepted resource gate did not pass")
            if is_update:
                try:
                    checkpoint = StandardTransactionCheckpoint(
                        **row["checkpoint"]  # type: ignore[arg-type]
                    )
                except (KeyError, TypeError, StandardTrainingError) as exc:
                    raise StandardTrainingError(
                        "resource acceptance checkpoint drifted"
                    ) from exc
                transaction = expected_updates[str(key)]
                if (
                    checkpoint.transaction_key != transaction.key
                    or checkpoint.seed != transaction.seed
                    or checkpoint.update != transaction.update
                ):
                    raise StandardTrainingError(
                        "resource acceptance checkpoint drifted"
                    )
        grouped[str(key)].append((index, row))

    attempt_count = 0
    for key, keyed_rows in grouped.items():
        if not keyed_rows:
            raise StandardTrainingError("resource audit missing atomic unit")
        attempts = sorted({int(row["attempt"]) for _, row in keyed_rows})
        if attempts != list(range(1, max(attempts) + 1)):
            raise StandardTrainingError("resource audit attempt sequence drifted")
        attempt_count += len(attempts)
        accepted = [(index, row) for index, row in keyed_rows if row["phase"] == "accepted"]
        if len(accepted) != 1:
            raise StandardTrainingError("resource audit accepted unit drifted")
        accepted_index, accepted_row = accepted[0]
        accepted_attempt = int(accepted_row["attempt"])
        matching_pre = [
            (index, row)
            for index, row in keyed_rows
            if row["attempt"] == accepted_attempt and row["phase"] == "pre"
        ]
        matching_post = [
            (index, row)
            for index, row in keyed_rows
            if row["attempt"] == accepted_attempt and row["phase"] == "post"
        ]
        if (
            len(matching_pre) != 1
            or len(matching_post) != 1
            or not matching_pre[0][0] < matching_post[0][0] < accepted_index
            or accepted_row["pre"] != matching_pre[0][1]["resource"]
            or accepted_row["post"] != matching_post[0][1]["resource"]
        ):
            raise StandardTrainingError("resource acceptance pre/post binding drifted")
        for attempt in attempts:
            attempt_rows = [row for _, row in keyed_rows if row["attempt"] == attempt]
            if sum(row["phase"] == "pre" for row in attempt_rows) != 1:
                raise StandardTrainingError("resource attempt pre gate drifted")
            if sum(row["phase"] == "post" for row in attempt_rows) > 1:
                raise StandardTrainingError("resource attempt post gate drifted")
    if require_terminal and terminal_evidence is None:
        raise StandardTrainingError("terminal resource evidence is missing")
    segment_chain = lifecycle.get("segment_chain")
    if not isinstance(segment_chain, list) or not segment_chain:
        raise StandardTrainingError("resource lifecycle segment summary drifted")
    active_segment = segment_chain[-1]
    if not isinstance(active_segment, Mapping):
        raise StandardTrainingError("resource lifecycle active segment drifted")
    return {
        "schema_version": "stage6_resource_audit_validation/v1",
        "passed": True,
        "update_accepted_count": len(expected_updates),
        "final_accepted_count": len(expected_finals),
        "attempt_count": attempt_count,
        "rss_root_pid": lifecycle["active_root_pid"],
        "rss_final_sample_count": active_segment["local_final_sample_count"],
        "rss_lifecycle_peak_bytes": lifecycle[
            "run_rss_lifecycle_peak_bytes"
        ],
        "resource_segment_count": lifecycle["segment_count"],
        "resource_active_segment_id": lifecycle["active_segment_id"],
        "resource_active_segment_index": lifecycle["active_segment_index"],
        "terminal_evidence_present": terminal_evidence is not None,
    }


def _jsonable_runtime_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable_runtime_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable_runtime_value(item) for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable_runtime_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise StandardTrainingError(
        f"runtime artifact cannot encode {type(value).__name__}"
    )


def validate_standard_collection_audit(
    value: object,
    *,
    expected_device: str,
    expected_policy_sha256: str | None = None,
    expected_inference_pid: int | None = None,
    require_dual_scan: bool = False,
) -> dict[str, object]:
    """Validate one complete 8x128 on-policy collector audit."""

    if type(require_dual_scan) is not bool:
        raise StandardTrainingError(
            "collector reset diagnostics profile is invalid"
        )
    audit = _jsonable_runtime_value(value)
    legacy_fields = {
        "schema_version",
        "worker_pids",
        "worker_start_methods",
        "per_env_trainable_counts",
        "diagnostic_reset_counts",
        "trainable_transition_count",
        "inference_pids",
        "inference_batch_count",
        "policy_device",
        "policy_state_sha256",
        "snapshot_sha256",
        "terminal_transition_count",
    }
    diagnostics_fields = {
        "reset_diagnostics",
        "planner_failure_counts",
    }
    audit_fields = set(audit) if isinstance(audit, Mapping) else set()
    if (
        not isinstance(audit, Mapping)
        or (
            audit_fields != legacy_fields
            and audit_fields != legacy_fields | diagnostics_fields
        )
    ):
        raise StandardTrainingError("collector audit field set drifted")
    from lunar_exploration_ppo.ppo.collector import (
        PLANNER_FAILURE_REASONS,
        validate_reset_diagnostics_payload,
    )

    schema_version = audit["schema_version"]
    has_diagnostics = set(audit) == legacy_fields | diagnostics_fields
    if schema_version == "stage4_spawn_collector/v2":
        if not has_diagnostics:
            raise StandardTrainingError(
                "collector diagnostics audit is missing"
            )
    elif schema_version != "stage4_spawn_collector/v1":
        raise StandardTrainingError("collector audit schema drifted")
    if require_dual_scan and not has_diagnostics:
        raise StandardTrainingError(
            "collector reset diagnostics audit is missing"
        )
    if has_diagnostics:
        reset_rows = audit["reset_diagnostics"]
        failure_counts = audit["planner_failure_counts"]
        if (
            not isinstance(reset_rows, list)
            or not isinstance(failure_counts, Mapping)
            or set(failure_counts) != set(PLANNER_FAILURE_REASONS)
            or any(
                type(failure_counts[reason]) is not int
                or int(failure_counts[reason]) < 0
                for reason in PLANNER_FAILURE_REASONS
            )
        ):
            raise StandardTrainingError(
                "collector diagnostics audit drifted"
            )
        for reset_index, row in enumerate(reset_rows):
            if (
                not isinstance(row, Mapping)
                or set(row)
                != {
                    "worker_index",
                    "reset_index",
                    "schema_version",
                    "scan_order",
                    "local_safety_sensor",
                    "exploration_sensor",
                }
                or type(row.get("worker_index")) is not int
                or not 0 <= int(row["worker_index"]) < 8
                or row.get("reset_index") != reset_index
            ):
                raise StandardTrainingError(
                    "collector reset diagnostics audit drifted"
                )
            try:
                validate_reset_diagnostics_payload(
                    {
                        key: row[key]
                        for key in (
                            "schema_version",
                            "scan_order",
                            "local_safety_sensor",
                            "exploration_sensor",
                        )
                    },
                    require_dual_scan=require_dual_scan,
                )
            except ValueError as exc:
                raise StandardTrainingError(
                    "collector reset diagnostics audit drifted"
                ) from exc
    workers = audit["worker_pids"]
    starts = audit["worker_start_methods"]
    counts = audit["per_env_trainable_counts"]
    resets = audit["diagnostic_reset_counts"]
    inference_pids = audit["inference_pids"]
    snapshots = audit["snapshot_sha256"]
    if (
        not isinstance(workers, list)
        or len(workers) != 8
        or any(type(pid) is not int or pid <= 0 for pid in workers)
        or len(set(workers)) != 8
        or starts != ["spawn"] * 8
        or counts != [128] * 8
        or not isinstance(resets, list)
        or len(resets) != 8
        or any(type(count) is not int or count < 0 for count in resets)
        or audit["trainable_transition_count"] != 1024
        or not isinstance(inference_pids, list)
        or len(inference_pids) != 1
        or type(inference_pids[0]) is not int
        or inference_pids[0] <= 0
        or inference_pids[0] in workers
        or type(audit["inference_batch_count"]) is not int
        or audit["inference_batch_count"] not in {128, 129}
        or not isinstance(snapshots, list)
        or len(snapshots) != 1024
        or any(not _is_sha256(item) for item in snapshots)
        or type(audit["terminal_transition_count"]) is not int
        or not 0 <= audit["terminal_transition_count"] <= 1024
        or not _is_sha256(audit["policy_state_sha256"])
    ):
        raise StandardTrainingError("collector 8x128 audit drifted")
    policy_device = audit["policy_device"]
    if not isinstance(expected_device, str) or not expected_device:
        raise StandardTrainingError("collector expected device drifted")
    try:
        parsed_expected_device = torch.device(expected_device)
        if (
            parsed_expected_device.type == "cuda"
            and parsed_expected_device.index is None
        ):
            parsed_expected_device = torch.device(
                "cuda", torch.cuda.current_device()
            )
    except (AssertionError, RuntimeError, TypeError, ValueError):
        raise StandardTrainingError("collector expected device drifted") from None
    if not isinstance(policy_device, str) or not policy_device:
        raise StandardTrainingError("collector policy device drifted")
    try:
        parsed_policy_device = torch.device(policy_device)
        if (
            parsed_policy_device.type == "cuda"
            and parsed_policy_device.index is None
        ):
            parsed_policy_device = torch.device(
                "cuda", torch.cuda.current_device()
            )
    except (AssertionError, RuntimeError, TypeError, ValueError):
        raise StandardTrainingError("collector policy device drifted") from None
    if parsed_policy_device != parsed_expected_device:
        raise StandardTrainingError("collector policy device drifted")
    if (
        expected_policy_sha256 is not None
        and audit["policy_state_sha256"] != expected_policy_sha256
    ):
        raise StandardTrainingError("collector stale policy hash drifted")
    if (
        expected_inference_pid is not None
        and inference_pids != [expected_inference_pid]
    ):
        raise StandardTrainingError("collector inference escaped the parent process")
    return dict(audit)


def validate_standard_training_validation_rows(
    *,
    config: Stage6Config,
    transactions: Sequence[StandardTrainingTransaction],
    receipt_rows: Sequence[Mapping[str, object]],
    training_rows: Sequence[Mapping[str, object]],
    validation_rows: Sequence[Mapping[str, object]],
    planning_warm_start: bool,
) -> tuple[
    dict[str, Mapping[str, object]],
    dict[int, dict[str, object]],
]:
    """Validate one complete accepted training prefix using production semantics."""

    if (
        type(config) is not Stage6Config
        or type(planning_warm_start) is not bool
        or not isinstance(transactions, Sequence)
        or isinstance(transactions, (str, bytes, bytearray))
        or any(
            not isinstance(item, StandardTrainingTransaction)
            for item in transactions
        )
        or not isinstance(receipt_rows, Sequence)
        or isinstance(receipt_rows, (str, bytes, bytearray))
        or any(not isinstance(row, Mapping) for row in receipt_rows)
        or not isinstance(training_rows, Sequence)
        or isinstance(training_rows, (str, bytes, bytearray))
        or not isinstance(validation_rows, Sequence)
        or isinstance(validation_rows, (str, bytes, bytearray))
    ):
        raise StandardTrainingError(
            "training/validation prefix validator input drifted"
        )
    receipt_by_key = {
        str(row.get("transaction_key")): row for row in receipt_rows
    }
    training_by_key: dict[str, Mapping[str, object]] = {}
    previous_policy_by_seed: dict[int, str] = {}
    if planning_warm_start:
        from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
            PARENT_POLICY_STATE_SHA256,
        )

        initial_policy_state_sha256 = PARENT_POLICY_STATE_SHA256
    else:
        initial_policy_state_sha256 = (
            config.stage5_authority.policy_state_sha256
        )
    if len(training_rows) != len(transactions):
        raise StandardTrainingError("Stage 6 training metric count drifted")
    for transaction, row in zip(transactions, training_rows, strict=True):
        if not isinstance(row, Mapping) or set(row) != {
            "transaction_key",
            "seed",
            "update",
            "checkpoint_sha256",
            "policy_state_sha256",
            "update_metrics",
            "validation",
            "collection_audit",
        }:
            raise StandardTrainingError("training metric row schema drifted")
        if (
            row["transaction_key"] != transaction.key
            or row["seed"] != transaction.seed
            or row["update"] != transaction.update
            or not isinstance(row["update_metrics"], Mapping)
            or not isinstance(row["validation"], Mapping)
        ):
            raise StandardTrainingError("training metric schedule drifted")
        receipt = receipt_by_key.get(transaction.key)
        if receipt is None:
            raise StandardTrainingError("training metric receipt is missing")
        policy_before = previous_policy_by_seed.get(
            transaction.seed,
            initial_policy_state_sha256,
        )
        policy_after = str(receipt["policy_state_sha256"])
        update_metrics = row["update_metrics"]
        collection = validate_standard_collection_audit(
            row["collection_audit"],
            expected_device=config.device,
            expected_policy_sha256=policy_before,
            require_dual_scan=planning_warm_start,
        )
        if (
            row["checkpoint_sha256"] != receipt["checkpoint_sha256"]
            or row["policy_state_sha256"] != policy_after
            or update_metrics.get("update_step") != transaction.update
            or update_metrics.get("policy_state_sha256_before")
            != policy_before
            or update_metrics.get("policy_state_sha256_after")
            != policy_after
            or collection["policy_state_sha256"] != policy_before
        ):
            raise StandardTrainingError("training on-policy lineage drifted")
        training_by_key[transaction.key] = row
        previous_policy_by_seed[transaction.seed] = policy_after

    validation_transactions = tuple(
        item for item in transactions if item.validation_episodes == 16
    )
    seed_best_records: dict[int, dict[str, object]] = {}
    if len(validation_rows) != len(validation_transactions):
        raise StandardTrainingError("validation metric count drifted")
    for transaction, row in zip(
        validation_transactions,
        validation_rows,
        strict=True,
    ):
        if not isinstance(row, Mapping) or set(row) != {
            "transaction_key",
            "seed",
            "update",
            "episode_count",
            "result",
            "eval_isolation",
            "best_record",
        }:
            raise StandardTrainingError("validation metric row schema drifted")
        result = row["result"]
        isolation = row["eval_isolation"]
        if (
            row["transaction_key"] != transaction.key
            or row["seed"] != transaction.seed
            or row["update"] != transaction.update
            or row["episode_count"] != 16
            or not isinstance(result, Mapping)
            or result.get("episode_count") != 16
            or not isinstance(isolation, Mapping)
            or isolation.get("passed") is not True
            or training_by_key[transaction.key]["validation"] != result
        ):
            raise StandardTrainingError("validation metric binding drifted")
        selected = select_validation_checkpoint_best(
            seed=transaction.seed,
            update=transaction.update,
            validation_metrics=result,
            previous_best=seed_best_records.get(transaction.seed, {}),
        )
        if row["best_record"] != selected:
            raise StandardTrainingError("validation seed best drifted")
        seed_best_records[transaction.seed] = selected
    return training_by_key, seed_best_records


def run_eval_only_transaction(
    *,
    evaluator: Callable[[], object],
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    normalization_stats: Mapping[str, object],
    runtime_state_provider: Callable[[], Sequence[Mapping[str, object]]],
    binding_state_provider: Callable[[], Mapping[str, bytes]],
) -> tuple[object, dict[str, object]]:
    """执行只读 eval，并用实际状态内容证明训练事务未被修改。"""

    if (
        not callable(evaluator)
        or not isinstance(policy, nn.Module)
        or not isinstance(optimizer, torch.optim.Optimizer)
        or not isinstance(normalization_stats, Mapping)
        or not callable(runtime_state_provider)
        or not callable(binding_state_provider)
    ):
        raise StandardTrainingError("eval-only transaction binding drifted")
    before = _capture_eval_isolation(
        policy=policy,
        optimizer=optimizer,
        normalization_stats=normalization_stats,
        runtime_states=runtime_state_provider(),
        binding_state=binding_state_provider(),
    )
    result = evaluator()
    after = _capture_eval_isolation(
        policy=policy,
        optimizer=optimizer,
        normalization_stats=normalization_stats,
        runtime_states=runtime_state_provider(),
        binding_state=binding_state_provider(),
    )
    comparisons = (
        (
            "policy",
            (before.policy_sha256, before.policy_training),
            (after.policy_sha256, after.policy_training),
        ),
        ("optimizer", before.optimizer_sha256, after.optimizer_sha256),
        ("RNG", before.rng_sha256, after.rng_sha256),
        ("normalizer", before.normalizer_sha256, after.normalizer_sha256),
        (
            "scenario sampler/vector state",
            before.runtime_state_sha256,
            after.runtime_state_sha256,
        ),
        ("config/checkpoint binding", before.binding_sha256, after.binding_sha256),
    )
    for label, left, right in comparisons:
        if left != right:
            raise StandardTrainingError(f"eval-only transaction mutated {label}")
    return result, {
        "schema_version": "stage6_eval_only_isolation_audit/v1",
        "passed": True,
        "policy_unchanged": True,
        "optimizer_unchanged": True,
        "rng_unchanged": True,
        "normalizer_unchanged": True,
        "scenario_sampler_vector_state_unchanged": True,
        "config_checkpoint_binding_unchanged": True,
        "policy_sha256": before.policy_sha256,
        "optimizer_sha256": before.optimizer_sha256,
        "rng_sha256": before.rng_sha256,
        "runtime_state_sha256": before.runtime_state_sha256,
        "binding_sha256": before.binding_sha256,
    }


def _capture_eval_isolation(
    *,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    normalization_stats: Mapping[str, object],
    runtime_states: Sequence[Mapping[str, object]],
    binding_state: Mapping[str, bytes],
) -> EvalIsolationSnapshot:
    states = tuple(runtime_states)
    if len(states) != 8 or any(not isinstance(value, Mapping) for value in states):
        raise StandardTrainingError("eval-only runtime state requires eight workers")
    if (
        not isinstance(binding_state, Mapping)
        or set(binding_state) != {"config_bytes", "checkpoint_bytes"}
        or any(
            not isinstance(binding_state[name], bytes) or not binding_state[name]
            for name in ("config_bytes", "checkpoint_bytes")
        )
    ):
        raise StandardTrainingError("eval-only backing-file binding drifted")
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()
        ),
    }
    return EvalIsolationSnapshot(
        policy_sha256=policy_state_sha256(policy),
        policy_training=policy.training,
        optimizer_sha256=_runtime_value_sha256(optimizer.state_dict()),
        rng_sha256=_runtime_value_sha256(rng_state),
        normalizer_sha256=_runtime_value_sha256(dict(normalization_stats)),
        runtime_state_sha256=_runtime_value_sha256(states),
        binding_sha256=_runtime_value_sha256(dict(binding_state)),
    )


def _runtime_value_sha256(value: object) -> str:
    digest = hashlib.sha256()

    def update(item: object) -> None:
        if item is None:
            digest.update(b"none")
        elif isinstance(item, bool):
            digest.update(b"bool1" if item else b"bool0")
        elif isinstance(item, int):
            digest.update(b"int" + str(item).encode("ascii") + b";")
        elif isinstance(item, float):
            digest.update(b"float" + struct.pack("<d", item))
        elif isinstance(item, str):
            payload = item.encode("utf-8")
            digest.update(b"str" + struct.pack("<Q", len(payload)) + payload)
        elif isinstance(item, bytes):
            digest.update(b"bytes" + struct.pack("<Q", len(item)) + item)
        elif isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous().cpu()
            update(str(tensor.dtype))
            update(tuple(tensor.shape))
            update(tensor.numpy().tobytes(order="C"))
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            update(array.dtype.str)
            update(tuple(array.shape))
            update(array.tobytes(order="C"))
        elif isinstance(item, Mapping):
            digest.update(b"map")
            for key in sorted(item, key=lambda value: (type(value).__name__, repr(value))):
                update(key)
                update(item[key])
            digest.update(b"map-end")
        elif isinstance(item, (tuple, list)):
            digest.update(b"tuple" if isinstance(item, tuple) else b"list")
            for value in item:
                update(value)
            digest.update(b"seq-end")
        else:
            raise StandardTrainingError(
                f"runtime hash does not support {type(item).__name__}"
            )

    update(value)
    return digest.hexdigest()


def _immutable_bindings_valid(bindings: Mapping[str, object]) -> bool:
    binding_keys = set(bindings)
    if (
        binding_keys != _BASE_STAGE6_IMMUTABLE_BINDING_KEYS
        and binding_keys != STAGE6_IMMUTABLE_BINDING_KEYS
    ):
        return False
    environment = bindings.get("environment_identity")
    if not isinstance(environment, Mapping):
        return False
    environment_sha256 = hashlib.sha256(
        ArtifactStore.canonical_json_bytes(dict(environment))
    ).hexdigest()
    tree = bindings.get("reviewed_prospective_git_tree")
    try:
        from lunar_exploration_ppo.workflows.stage6 import (
            validate_stage6_formal_run_id,
        )

        formal_run_id_valid = (
            validate_stage6_formal_run_id(str(bindings.get("formal_run_id")))
            == bindings.get("formal_run_id")
        )
    except (TypeError, ValueError, RuntimeError):
        formal_run_id_valid = False
    hash_fields = _IMMUTABLE_JOURNAL_BINDING_KEYS - {
        "environment_identity",
        "formal_run_id",
        "reviewed_prospective_git_tree",
    }
    base_valid = (
        all(_is_sha256(bindings.get(key)) for key in hash_fields)
        and bindings.get("environment_sha256") == environment_sha256
        and formal_run_id_valid
        and isinstance(tree, str)
        and len(tree) == 40
        and all(character in "0123456789abcdef" for character in tree)
        and bindings.get("prospective_tree_sha256")
        == hashlib.sha256(tree.encode("ascii")).hexdigest()
    )
    if not base_valid or binding_keys == _BASE_STAGE6_IMMUTABLE_BINDING_KEYS:
        return base_valid
    return (
        isinstance(bindings.get("coverage_cache_manifest_path"), str)
        and bool(bindings.get("coverage_cache_manifest_path"))
        and _is_sha256(bindings.get("coverage_cache_manifest_sha256"))
        and type(bindings.get("coverage_cache_manifest_size_bytes")) is int
        and int(bindings["coverage_cache_manifest_size_bytes"]) > 0
        and isinstance(bindings.get("coverage_cache_root"), str)
        and bool(bindings.get("coverage_cache_root"))
        and _is_sha256(bindings.get("coverage_cache_entry_set_sha256"))
        and bindings.get("coverage_cache_runtime_mode")
        == "persistent_exact_manifest_read_only/v1"
        and _is_sha256(bindings.get("coverage_cache_formal_audit_sha256"))
        and type(bindings.get("coverage_cache_formal_audit_size_bytes")) is int
        and int(bindings["coverage_cache_formal_audit_size_bytes"]) > 0
    )


def _review_authorization_immutable_bindings(
    verified: Mapping[str, object],
) -> dict[str, object]:
    return {
        "formal_run_id": verified["formal_run_id"],
        "changed_path_set_sha256": verified["changed_path_set_sha256"],
        "review_authorization_record_sha256": hashlib.sha256(
            ArtifactStore.canonical_json_bytes(dict(verified))
        ).hexdigest(),
        "authorization_file_sha256": verified["authorization_file_sha256"],
        "review_identity_sha256": verified["review_identity_sha256"],
        "reviewed_prospective_git_tree": verified[
            "reviewed_prospective_git_tree"
        ],
        "frozen_diff_sha256": verified["frozen_diff_sha256"],
        "spec_review_sha256": verified["spec_review_sha256"],
        "quality_review_sha256": verified["quality_review_sha256"],
    }


def _journal_bindings_valid(bindings: Mapping[str, object]) -> bool:
    immutable_keys = frozenset(set(bindings) - {"checkpoint_sha256"})
    return (
        immutable_keys
        in {
            _BASE_STAGE6_IMMUTABLE_BINDING_KEYS,
            STAGE6_IMMUTABLE_BINDING_KEYS,
        }
        and _immutable_bindings_valid(
            {key: bindings.get(key) for key in immutable_keys}
        )
        and _is_sha256(bindings.get("checkpoint_sha256"))
    )


def _journal_immutable_bindings(
    bindings: Mapping[str, object],
) -> dict[str, object] | None:
    if not _journal_bindings_valid(bindings):
        return None
    return {
        key: value
        for key, value in bindings.items()
        if key != "checkpoint_sha256"
    }


def build_standard_training_transactions(
    config: Stage6Config,
) -> tuple[StandardTrainingTransaction, ...]:
    if not isinstance(config, Stage6Config):
        raise StandardTrainingError("Stage 6 transactions require frozen config")
    training = config.training
    if (
        tuple(training.seeds) != FROZEN_SEEDS
        or training.updates_per_seed != 100
        or training.validation_every_updates != 10
        or training.validation_episodes != 16
    ):
        raise StandardTrainingError("Stage 6 frozen transaction schedule drifted")
    transactions: list[StandardTrainingTransaction] = []
    sequence = 0
    for seed_index, seed in enumerate(training.seeds):
        for update in range(1, training.updates_per_seed + 1):
            states: list[str] = []
            if update == 1:
                states.append(f"seed_{seed}_initializing")
            states.append(f"seed_{seed}_training_update_{update}")
            validation_episodes = 0
            if update % training.validation_every_updates == 0:
                validation_episodes = training.validation_episodes
                states.append(f"seed_{seed}_validating_update_{update}")
            if update == training.updates_per_seed:
                states.append(f"seed_{seed}_complete")
            transactions.append(
                StandardTrainingTransaction(
                    sequence=sequence,
                    seed=seed,
                    seed_index=seed_index,
                    update=update,
                    validation_episodes=validation_episodes,
                    commit_states=tuple(states),
                )
            )
            sequence += 1
    return tuple(transactions)


def run_standard_training_schedule(
    config: Stage6Config,
    backend: object,
) -> object:
    """以可审计 backend 执行固定 1×100 调度；不提供缩短入口。"""

    if not isinstance(config, Stage6Config):
        raise StandardTrainingError("Standard execution schedule requires frozen config")
    required_methods = (
        "prepare_seed",
        "run_update",
        "finish_seed",
        "freeze_global_best",
        "checkpoint_write_count",
        "run_final_evaluation",
        "finalize",
    )
    methods = {name: getattr(backend, name, None) for name in required_methods}
    if any(not callable(value) for value in methods.values()):
        raise StandardTrainingError("Standard execution backend interface drifted")
    frozen_transactions = build_standard_training_transactions(config)
    backend_transactions = getattr(backend, "transactions", None)
    if backend_transactions is None:
        transactions = frozen_transactions
    else:
        transactions = tuple(backend_transactions)
        from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
            build_planning_child_transactions,
        )

        child_transactions = build_planning_child_transactions(config)
        if transactions not in (frozen_transactions, child_transactions):
            raise StandardTrainingError(
                "backend transaction schedule is outside frozen contracts"
            )
    seed_bests: list[ValidationRecord] = []
    runtime_tokens: set[object] = set()
    sampler_seeds_seen: set[int] = set()
    for seed_index, seed in enumerate(config.training.seeds):
        seed_transactions = tuple(
            transaction for transaction in transactions if transaction.seed == seed
        )
        runtime: Mapping[str, object] | None = None
        prepared: object | None = None
        try:
            prepared = methods["prepare_seed"](seed, seed_transactions)
            if (
                not isinstance(prepared, tuple)
                or len(prepared) != 2
                or not isinstance(prepared[0], Mapping)
            ):
                raise StandardTrainingError("seed runtime preparation drifted")
            runtime = prepared[0]
            remaining = tuple(prepared[1])
            resumed = runtime.get("continue_from_current_state") is True
            planning_warm_start = runtime.get("planning_warm_start") is True
            sampler_seeds = runtime.get("sampler_seeds")
            common_invalid = (
                runtime.get("seed") != seed
                or runtime.get("sampler_seed_derivation_version")
                != SAMPLER_SEED_DERIVATION_VERSION
                or sampler_seeds != derive_standard_sampler_seeds(seed)
                or "runtime_token" not in runtime
                or "policy" not in runtime
                or "optimizer" not in runtime
                or remaining
                != seed_transactions[len(seed_transactions) - len(remaining) :]
            )
            if planning_warm_start:
                from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
                    PARENT_POLICY_STATE_SHA256,
                    PARENT_UPDATE,
                    build_planning_child_transactions,
                )

                warm_invalid = (
                    transactions != build_planning_child_transactions(config)
                    or runtime.get("initial_policy_sha256")
                    != PARENT_POLICY_STATE_SHA256
                    or runtime.get("optimizer_fresh") is not False
                    or runtime.get("independent_seed_origin") is not False
                    or runtime.get("parent_update") != PARENT_UPDATE
                    or runtime.get("fresh_vector_env_reset_required")
                    is not (not resumed)
                )
            else:
                warm_invalid = (
                    runtime.get("initial_policy_sha256")
                    != config.stage5_authority.policy_state_sha256
                    or runtime.get("optimizer_fresh") is not (not resumed)
                    or runtime.get("independent_seed_origin") is not True
                )
            if common_invalid or warm_invalid:
                raise StandardTrainingError("seed fresh initialization or resume drifted")
            assert isinstance(sampler_seeds, tuple)
            if any(value in sampler_seeds_seen for value in sampler_seeds):
                raise StandardTrainingError("training seed sampler streams overlap")
            sampler_seeds_seen.update(sampler_seeds)
            runtime_token = runtime["runtime_token"]
            if runtime_token in runtime_tokens:
                raise StandardTrainingError("seed reused policy or optimizer runtime")
            runtime_tokens.add(runtime_token)
            for transaction in remaining:
                if not isinstance(transaction, StandardTrainingTransaction):
                    raise StandardTrainingError("remaining transaction type drifted")
                methods["run_update"](runtime, transaction)
            best = methods["finish_seed"](runtime)
            if not isinstance(best, ValidationRecord) or best.seed != seed:
                raise StandardTrainingError("seed completion best record drifted")
            seed_bests.append(best)
        finally:
            if isinstance(runtime, Mapping):
                vector_env = runtime.get("vector_env")
                close = getattr(vector_env, "close", None)
                if callable(close) and getattr(vector_env, "closed", False) is not True:
                    close()
            runtime = None
            prepared = None
    global_best = select_global_best(
        seed_bests,
        configured_seeds=config.training.seeds,
    )
    frozen_best = methods["freeze_global_best"](global_best)
    if frozen_best != global_best:
        raise StandardTrainingError("global best freeze drifted")
    checkpoint_writes_before = methods["checkpoint_write_count"]()
    evaluations: dict[str, object] = {}
    from lunar_exploration_ppo.eval.standard import STANDARD_EVALUATION_METHODS

    evaluation_order = (
        ("test", "ppo_policy"),
        ("unseen", "ppo_policy"),
        *tuple(
            (split, method)
            for method in STANDARD_EVALUATION_METHODS
            if method != "ppo_policy"
            for split in ("test", "unseen")
        ),
    )
    for split, method in evaluation_order:
        result = methods["run_final_evaluation"](global_best, split, method)
        episode_count = (
            result.get("episode_count")
            if isinstance(result, Mapping)
            else getattr(result, "metrics", {}).get("episode_count")
        )
        if episode_count != 64:
            raise StandardTrainingError("final evaluation episode count drifted")
        evaluations[f"{split}:{method}"] = result
    if methods["checkpoint_write_count"]() != checkpoint_writes_before:
        raise StandardTrainingError("test/unseen evaluation wrote checkpoint")
    return methods["finalize"](global_best, evaluations)


def resume_standard_training_transactions(
    transactions: Sequence[StandardTrainingTransaction],
    *,
    completed_keys: Sequence[str],
    checkpoint: StandardTransactionCheckpoint | None,
) -> tuple[StandardTrainingTransaction, ...]:
    schedule = tuple(transactions)
    if not schedule or any(
        not isinstance(item, StandardTrainingTransaction) for item in schedule
    ):
        raise StandardTrainingError("resume transaction schedule is invalid")
    completed = tuple(completed_keys)
    expected = tuple(item.key for item in schedule[: len(completed)])
    if completed != expected:
        raise StandardTrainingError("completed transactions must be an exact prefix")
    if not completed:
        if checkpoint is not None:
            raise StandardTrainingError("fresh transaction prefix cannot bind checkpoint")
        return schedule
    last = schedule[len(completed) - 1]
    if (
        not isinstance(checkpoint, StandardTransactionCheckpoint)
        or checkpoint.transaction_key != last.key
        or checkpoint.seed != last.seed
        or checkpoint.update != last.update
    ):
        raise StandardTrainingError("resume checkpoint does not match transaction prefix")
    return schedule[len(completed) :]


def completed_transaction_keys_from_journal(
    records: Sequence[Mapping[str, object]],
    transactions: Sequence[StandardTrainingTransaction],
) -> tuple[str, ...]:
    schedule = tuple(transactions)
    rows = tuple(records)
    expected_states = tuple(
        state for transaction in schedule for state in transaction.commit_states
    )
    actual_states = tuple(row.get("state") for row in rows)
    if any(not isinstance(row, Mapping) for row in rows) or actual_states != expected_states[
        : len(actual_states)
    ]:
        raise StandardTrainingError("transaction journal is not an exact state prefix")
    completed: list[str] = []
    state_count = 0
    for transaction in schedule:
        state_count += len(transaction.commit_states)
        if state_count > len(actual_states):
            break
        completed.append(transaction.key)
    return tuple(completed)


def verify_journal_checkpoint_bindings(
    records: Sequence[Mapping[str, object]],
    transactions: Sequence[StandardTrainingTransaction],
    receipt_rows: Sequence[Mapping[str, object]],
    immutable_bindings: Mapping[str, object],
    *,
    historical_immutable_bindings: Mapping[str, object] | None = None,
    current_binding_first_transaction_key: str | None = None,
    immutable_binding_epochs: (
        Sequence[tuple[str, Mapping[str, object]]] | None
    ) = None,
) -> tuple[str, ...]:
    schedule = tuple(transactions)
    rows = tuple(records)
    split_requested = (
        historical_immutable_bindings is not None
        or current_binding_first_transaction_key is not None
    )
    epochs_requested = immutable_binding_epochs is not None
    if (
        not schedule
        or any(not isinstance(item, StandardTrainingTransaction) for item in schedule)
        or not _immutable_bindings_valid(immutable_bindings)
        or (split_requested and epochs_requested)
        or (
            split_requested
            and (
                historical_immutable_bindings is None
                or not isinstance(current_binding_first_transaction_key, str)
                or not current_binding_first_transaction_key
                or set(historical_immutable_bindings)
                not in (
                    _BASE_STAGE6_IMMUTABLE_BINDING_KEYS,
                    STAGE6_IMMUTABLE_BINDING_KEYS,
                )
                or set(immutable_bindings) != STAGE6_IMMUTABLE_BINDING_KEYS
                or not _immutable_bindings_valid(historical_immutable_bindings)
            )
        )
    ):
        raise StandardTrainingError("immutable journal binding drifted")
    current_binding_first_index: int | None = None
    historical = (
        dict(historical_immutable_bindings)
        if historical_immutable_bindings is not None
        else None
    )
    if split_requested:
        matching_indexes = tuple(
            index
            for index, transaction in enumerate(schedule)
            if transaction.key == current_binding_first_transaction_key
        )
        if len(matching_indexes) != 1:
            raise StandardTrainingError("immutable journal cutover drifted")
        current_binding_first_index = matching_indexes[0]
    epochs: list[tuple[int, dict[str, object]]] = []
    if epochs_requested:
        raw_epochs = tuple(immutable_binding_epochs or ())
        if not raw_epochs:
            raise StandardTrainingError(
                "immutable journal binding epochs drifted"
            )
        schedule_indexes = {
            transaction.key: index
            for index, transaction in enumerate(schedule)
        }
        for value in raw_epochs:
            if (
                not isinstance(value, tuple)
                or len(value) != 2
                or not isinstance(value[0], str)
                or not value[0]
                or not isinstance(value[1], Mapping)
                or value[0] not in schedule_indexes
                or not _immutable_bindings_valid(value[1])
            ):
                raise StandardTrainingError(
                    "immutable journal binding epochs drifted"
                )
            epochs.append(
                (schedule_indexes[value[0]], dict(value[1]))
            )
        if (
            epochs[0][0] != 0
            or any(
                later[0] <= earlier[0]
                for earlier, later in zip(epochs, epochs[1:])
            )
            or epochs[-1][1] != dict(immutable_bindings)
        ):
            raise StandardTrainingError(
                "immutable journal binding epochs drifted"
            )
    if any(not isinstance(row, Mapping) for row in rows):
        raise StandardTrainingError("transaction journal row is invalid")
    completed = completed_transaction_keys_from_journal(rows, schedule)
    receipts = CheckpointReceiptIndex._validate_rows(tuple(receipt_rows))
    if len(receipts) < len(completed):
        raise StandardTrainingError("checkpoint receipt prefix is incomplete")
    if len(receipts) > len(completed) + 1:
        raise StandardTrainingError(
            "checkpoint receipt ahead exceeds one transaction"
        )
    for index, receipt in enumerate(receipts):
        transaction = schedule[index]
        if (
            receipt["transaction_key"] != transaction.key
            or receipt["seed"] != transaction.seed
            or receipt["update"] != transaction.update
        ):
            raise StandardTrainingError("checkpoint receipt is not an exact prefix")

    offset = 0
    for transaction_index, transaction in enumerate(schedule):
        group_end = min(offset + len(transaction.commit_states), len(rows))
        group = rows[offset:group_end]
        if group:
            if transaction_index >= len(receipts):
                raise StandardTrainingError(
                    "transaction journal is missing checkpoint receipt"
                )
            receipt = receipts[transaction_index]
            if epochs:
                expected_immutable = epochs[0][1]
                for first_index, epoch_bindings in epochs:
                    if first_index > transaction_index:
                        break
                    expected_immutable = epoch_bindings
            else:
                expected_immutable = (
                    historical
                    if (
                        historical is not None
                        and current_binding_first_index is not None
                        and transaction_index < current_binding_first_index
                    )
                    else dict(immutable_bindings)
                )
            for row in group:
                bindings = row.get("bindings")
                recorded_immutable = (
                    _journal_immutable_bindings(bindings)
                    if isinstance(bindings, Mapping)
                    else None
                )
                if recorded_immutable is None:
                    raise StandardTrainingError("transaction journal binding drifted")
                if recorded_immutable != expected_immutable:
                    raise StandardTrainingError(
                        "transaction journal immutable binding drifted"
                    )
                if bindings["checkpoint_sha256"] != receipt["checkpoint_sha256"]:
                    raise StandardTrainingError(
                        "transaction journal checkpoint receipt binding drifted"
                    )
        offset += len(transaction.commit_states)
        if offset >= len(rows):
            break
    return completed


def commit_checkpointed_transaction(
    *,
    journal: object,
    transactions: Sequence[StandardTrainingTransaction],
    transaction: StandardTrainingTransaction,
    checkpoint: StandardTransactionCheckpoint,
    bindings: Mapping[str, object],
    receipt_rows: Sequence[Mapping[str, object]],
    historical_immutable_bindings: Mapping[str, object] | None = None,
    current_binding_first_transaction_key: str | None = None,
    immutable_binding_epochs: (
        Sequence[tuple[str, Mapping[str, object]]] | None
    ) = None,
) -> tuple[dict[str, object], ...]:
    schedule = tuple(transactions)
    if (
        not isinstance(transaction, StandardTrainingTransaction)
        or transaction.sequence >= len(schedule)
        or schedule[transaction.sequence] != transaction
        or not isinstance(checkpoint, StandardTransactionCheckpoint)
        or checkpoint.transaction_key != transaction.key
        or checkpoint.seed != transaction.seed
        or checkpoint.update != transaction.update
        or bindings.get("checkpoint_sha256") != checkpoint.checkpoint_sha256
    ):
        raise StandardTrainingError("transaction checkpoint binding mismatch")
    verify = getattr(journal, "verify", None)
    append = getattr(journal, "append", None)
    if not callable(verify) or not callable(append):
        raise StandardTrainingError("transaction journal interface is invalid")
    records = tuple(verify())
    immutable_bindings = _journal_immutable_bindings(bindings)
    if immutable_bindings is None:
        raise StandardTrainingError("transaction journal binding drifted")
    completed = verify_journal_checkpoint_bindings(
        records,
        schedule,
        receipt_rows,
        immutable_bindings,
        historical_immutable_bindings=historical_immutable_bindings,
        current_binding_first_transaction_key=(
            current_binding_first_transaction_key
        ),
        immutable_binding_epochs=immutable_binding_epochs,
    )
    if transaction.key in completed:
        return ()
    expected_previous = tuple(item.key for item in schedule[: transaction.sequence])
    if completed != expected_previous:
        raise StandardTrainingError("transaction journal previous prefix is incomplete")
    flattened_before = sum(
        len(item.commit_states) for item in schedule[: transaction.sequence]
    )
    states_already = len(records) - flattened_before
    if not 0 <= states_already < len(transaction.commit_states):
        raise StandardTrainingError("transaction journal partial state is invalid")
    appended: list[dict[str, object]] = []
    for state in transaction.commit_states[states_already:]:
        record = append(state, bindings)
        if not isinstance(record, dict):
            raise StandardTrainingError("transaction journal append receipt is invalid")
        appended.append(record)
    if verify_journal_checkpoint_bindings(
        verify(),
        schedule,
        receipt_rows,
        immutable_bindings,
        historical_immutable_bindings=historical_immutable_bindings,
        current_binding_first_transaction_key=(
            current_binding_first_transaction_key
        ),
        immutable_binding_epochs=immutable_binding_epochs,
    )[-1:] != (
        transaction.key,
    ):
        raise StandardTrainingError("transaction journal commit did not become durable")
    return tuple(appended)


def reconcile_checkpointed_resume(
    *,
    transactions: Sequence[StandardTrainingTransaction],
    journal: object,
    checkpoint: StandardTransactionCheckpoint | None,
    bindings: Mapping[str, object],
    receipt_rows: Sequence[Mapping[str, object]],
    historical_immutable_bindings: Mapping[str, object] | None = None,
    current_binding_first_transaction_key: str | None = None,
    immutable_binding_epochs: (
        Sequence[tuple[str, Mapping[str, object]]] | None
    ) = None,
) -> tuple[tuple[StandardTrainingTransaction, ...], tuple[dict[str, object], ...]]:
    schedule = tuple(transactions)
    verify = getattr(journal, "verify", None)
    if not schedule or not callable(verify):
        raise StandardTrainingError("resume reconciliation interface drifted")
    records = tuple(verify())
    immutable_bindings = _journal_immutable_bindings(bindings)
    if immutable_bindings is None:
        raise StandardTrainingError("transaction journal binding drifted")
    completed = verify_journal_checkpoint_bindings(
        records,
        schedule,
        receipt_rows,
        immutable_bindings,
        historical_immutable_bindings=historical_immutable_bindings,
        current_binding_first_transaction_key=(
            current_binding_first_transaction_key
        ),
        immutable_binding_epochs=immutable_binding_epochs,
    )
    if checkpoint is None:
        return (
            resume_standard_training_transactions(
                schedule,
                completed_keys=completed,
                checkpoint=None,
            ),
            (),
        )
    if completed:
        last = schedule[len(completed) - 1]
        if (
            checkpoint.transaction_key == last.key
            and checkpoint.seed == last.seed
            and checkpoint.update == last.update
        ):
            return (
                resume_standard_training_transactions(
                    schedule,
                    completed_keys=completed,
                    checkpoint=checkpoint,
                ),
                (),
            )
    next_index = len(completed)
    if (
        next_index >= len(schedule)
        or checkpoint.transaction_key != schedule[next_index].key
        or checkpoint.seed != schedule[next_index].seed
        or checkpoint.update != schedule[next_index].update
    ):
        raise StandardTrainingError(
            "complete checkpoint may be ahead of journal by exactly one transaction"
        )
    appended = commit_checkpointed_transaction(
        journal=journal,
        transactions=schedule,
        transaction=schedule[next_index],
        checkpoint=checkpoint,
        bindings=bindings,
        receipt_rows=receipt_rows,
        historical_immutable_bindings=historical_immutable_bindings,
        current_binding_first_transaction_key=(
            current_binding_first_transaction_key
        ),
        immutable_binding_epochs=immutable_binding_epochs,
    )
    repaired_completed = verify_journal_checkpoint_bindings(
        verify(),
        schedule,
        receipt_rows,
        immutable_bindings,
        historical_immutable_bindings=historical_immutable_bindings,
        current_binding_first_transaction_key=(
            current_binding_first_transaction_key
        ),
        immutable_binding_epochs=immutable_binding_epochs,
    )
    return (
        resume_standard_training_transactions(
            schedule,
            completed_keys=repaired_completed,
            checkpoint=checkpoint,
        ),
        appended,
    )


def append_transaction_metric_once(
    path: str | Path,
    record: Mapping[str, object],
) -> bool:
    destination = lexical_absolute(path)
    durable = DurableJsonl(destination)
    try:
        prior_payload = durable.recover_and_snapshot()
    except (DurableJsonlError, OSError) as exc:
        raise StandardTrainingError(
            "transaction metric committed snapshot failed"
        ) from exc
    value = dict(record)
    transaction_key = value.get("transaction_key")
    if not isinstance(transaction_key, str) or not transaction_key:
        raise StandardTrainingError("transaction metric key is invalid")
    rows: list[dict[str, object]] = []
    if prior_payload:
        try:
            for raw_line in prior_payload.splitlines(keepends=True):
                row = json.loads(raw_line.decode("utf-8"))
                canonical = (
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                if not isinstance(row, dict) or raw_line != canonical:
                    raise ValueError("noncanonical metric row")
                rows.append(row)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise StandardTrainingError("transaction metric JSONL drifted") from exc
    matching = [row for row in rows if row.get("transaction_key") == transaction_key]
    if len(matching) > 1 or (matching and matching[0] != value):
        raise StandardTrainingError("transaction metric duplicate content drifted")
    if matching:
        return False
    try:
        row_bytes = (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        durable.append(value)
        committed_payload = durable.recover_and_snapshot()
    except (DurableJsonlError, OSError, TypeError, ValueError) as exc:
        raise StandardTrainingError(
            "transaction metric durable append or snapshot failed"
        ) from exc
    if committed_payload != prior_payload + row_bytes:
        raise StandardTrainingError(
            "transaction metric exact append prefix drifted"
        )
    return True


class StandardTrainingStateMachine:
    def __init__(
        self,
        *,
        seeds: Sequence[int],
        updates_per_seed: int,
        validation_every_updates: int,
        validation_episodes: int,
    ) -> None:
        if (
            tuple(seeds) != FROZEN_SEEDS
            or updates_per_seed != 100
            or validation_every_updates != 10
            or validation_episodes != 16
        ):
            raise StandardTrainingError("Stage 6 frozen training schedule drifted")
        units: list[TrainingUnit] = []
        sequence = 0
        for seed_index, seed in enumerate(seeds):
            units.append(TrainingUnit(sequence, seed, seed_index, "initialize", 0))
            sequence += 1
            for update in range(1, updates_per_seed + 1):
                units.append(TrainingUnit(sequence, seed, seed_index, "train", update))
                sequence += 1
                if update % validation_every_updates == 0:
                    units.append(
                        TrainingUnit(
                            sequence,
                            seed,
                            seed_index,
                            "validate",
                            update,
                            validation_episodes,
                        )
                    )
                    sequence += 1
            units.append(
                TrainingUnit(sequence, seed, seed_index, "complete", updates_per_seed)
            )
            sequence += 1
        self.units = tuple(units)

    def remaining_after(self, completed_keys: Sequence[str]) -> tuple[TrainingUnit, ...]:
        completed = tuple(completed_keys)
        expected = tuple(unit.key for unit in self.units[: len(completed)])
        if completed != expected:
            raise StandardTrainingError(
                "completed atomic units must be an exact schedule prefix"
            )
        return self.units[len(completed) :]


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    seed: int
    update: int
    success_rate_under_fixed_step_budget: float
    mean_final_coverage: float
    checkpoint_ref: str

    def __post_init__(self) -> None:
        if (
            self.seed not in FROZEN_SEEDS
            or self.update not in range(10, 101, 10)
            or not 0.0 <= self.success_rate_under_fixed_step_budget <= 1.0
            or not 0.0 <= self.mean_final_coverage <= 1.0
            or not self.checkpoint_ref
        ):
            raise StandardTrainingError("validation record contract drifted")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def select_seed_best(records: Sequence[ValidationRecord]) -> ValidationRecord:
    if not records or len({record.seed for record in records}) != 1:
        raise StandardTrainingError("seed best requires one non-empty seed record set")
    return max(
        records,
        key=lambda record: (
            record.success_rate_under_fixed_step_budget,
            record.mean_final_coverage,
            -record.update,
        ),
    )


def select_global_best(
    records: Sequence[ValidationRecord],
    *,
    configured_seeds: Sequence[int],
) -> ValidationRecord:
    configured = tuple(configured_seeds)
    if configured != FROZEN_SEEDS or tuple(record.seed for record in records) != configured:
        raise StandardTrainingError(
            "global best requires exactly configured completed seeds in order"
        )
    seed_order = {seed: index for index, seed in enumerate(configured)}
    return max(
        records,
        key=lambda record: (
            record.success_rate_under_fixed_step_budget,
            record.mean_final_coverage,
            -seed_order[record.seed],
            -record.update,
        ),
    )


def _validated_planning_child_recovery_evidence_binding(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate the canonical evidence view exported by the issuer."""

    from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
        PLANNING_CHILD_RECOVERY_EVIDENCE_SCHEMA,
    )

    expected_fields = {
        "schema_version",
        "capability_sha256",
        "input_snapshot_sha256",
        "parent_artifact_sha256",
        "continuation_artifact_sha256",
        "acceptance_binding",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise StandardTrainingError(
            "Stage 6 planning child recovery evidence binding drifted"
        )
    result = dict(value)
    acceptance = result.get("acceptance_binding")
    hash_fields = (
        "capability_sha256",
        "input_snapshot_sha256",
        "parent_artifact_sha256",
        "continuation_artifact_sha256",
    )
    if (
        result.get("schema_version")
        != PLANNING_CHILD_RECOVERY_EVIDENCE_SCHEMA
        or any(not _is_sha256(result.get(field)) for field in hash_fields)
        or not isinstance(acceptance, Mapping)
        or acceptance.get("schema_version")
        != "stage6_planning_child_recovery_capability_acceptance/v1"
        or acceptance.get("input_snapshot_sha256")
        != result["input_snapshot_sha256"]
        or acceptance.get("parent_artifact_sha256")
        != result["parent_artifact_sha256"]
        or acceptance.get("continuation_artifact_sha256")
        != result["continuation_artifact_sha256"]
        or not isinstance(
            acceptance.get("current_verified_review_authorization"),
            Mapping,
        )
        or type(acceptance.get("terminal_complete")) is not bool
        or (
            acceptance.get("terminal_complete") is True
            and acceptance.get("resume_cursor") is not None
        )
        or (
            acceptance.get("terminal_complete") is False
            and acceptance.get("resume_cursor") is None
            and (
                not isinstance(
                    acceptance.get("accepted_anchor"),
                    Mapping,
                )
                or (
                    acceptance.get("crash_suffix") is not None
                    and not isinstance(
                        acceptance.get("crash_suffix"),
                        Mapping,
                    )
                )
                or (
                    (
                        acceptance["crash_suffix"].get("update")
                        if isinstance(
                            acceptance.get("crash_suffix"),
                            Mapping,
                        )
                        else acceptance["accepted_anchor"].get(
                            "last_accepted_update"
                        )
                    )
                    != 100
                )
            )
        )
        or (
            acceptance.get("terminal_complete") is False
            and acceptance.get("resume_cursor") is not None
            and not isinstance(
                acceptance.get("resume_cursor"),
                Mapping,
            )
        )
        or not isinstance(acceptance.get("lineage_epochs"), (list, tuple))
        or not acceptance.get("lineage_epochs")
        or not isinstance(
            acceptance.get("journal_binding_epochs"),
            (list, tuple),
        )
        or not acceptance.get("journal_binding_epochs")
        or not isinstance(
            acceptance.get("journal_prefixes"),
            Mapping,
        )
    ):
        raise StandardTrainingError(
            "Stage 6 planning child recovery evidence binding drifted"
        )
    result["acceptance_binding"] = json.loads(
        ArtifactStore.canonical_json_bytes(acceptance).decode("utf-8")
    )
    return result


def _validated_stage6_classic_source_repair_profile(
    value: Mapping[str, object] | None,
) -> str | None:
    """Validate the classic summary shape before it selects immutable keys."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise StandardTrainingError(
            "Stage 6 source-repair summary drifted"
        )
    common_fields = {
        "schema_version",
        "repair_id",
        "amendment_sha256",
        "seed",
        "last_origin_update",
        "first_repaired_update",
        "origin_execution_identity_sha256",
        "current_execution_identity_sha256",
        "origin_source_set_sha256",
        "current_source_set_sha256",
    }
    continuation_fields = {
        "continuation_id",
        "continuation_sha256",
        "repair_ordinal",
        "parent_execution_identity_sha256",
        "parent_source_set_sha256",
    }
    supplement_fields = {
        "supplement_id",
        "supplement_sha256",
        "last_continuation_update",
        "first_supplement_update",
        "continuation_execution_identity_sha256",
        "continuation_source_set_sha256",
        "ordinal_artifacts",
    }
    closure_fields = {
        "closure_id",
        "closure_sha256",
        "first_closure_update",
        "supplement_execution_identity_sha256",
        "supplement_source_set_sha256",
    }
    frontier_fields = {
        "frontier_recovery_id",
        "frontier_recovery_sha256",
        "last_closure_update",
        "first_frontier_recovery_update",
        "closure_execution_identity_sha256",
        "closure_source_set_sha256",
    }
    sensor_fields = {
        "sensor_acceleration_id",
        "sensor_acceleration_sha256",
        "last_frontier_recovery_update",
        "first_sensor_acceleration_update",
        "frontier_recovery_execution_identity_sha256",
        "frontier_recovery_source_set_sha256",
        "coverage_cache_manifest",
    }
    expected_by_schema = {
        "stage6_source_repair_summary/v1": (common_fields, None),
        "stage6_source_repair_summary/v2": (
            common_fields | continuation_fields,
            2,
        ),
        "stage6_source_repair_summary/v3": (
            common_fields | continuation_fields | supplement_fields,
            3,
        ),
        "stage6_source_repair_summary/v4": (
            common_fields
            | continuation_fields
            | supplement_fields
            | closure_fields,
            4,
        ),
        "stage6_source_repair_summary/v5": (
            common_fields
            | continuation_fields
            | supplement_fields
            | closure_fields
            | frontier_fields,
            5,
        ),
        "stage6_source_repair_summary/v6": (
            common_fields
            | continuation_fields
            | supplement_fields
            | closure_fields
            | frontier_fields
            | sensor_fields,
            6,
        ),
    }
    schema = value.get("schema_version")
    profile = expected_by_schema.get(schema)
    if profile is None:
        raise StandardTrainingError(
            "Stage 6 source-repair summary drifted"
        )
    expected_fields, expected_ordinal = profile
    if (
        set(value) != expected_fields
        or (
            expected_ordinal is not None
            and value.get("repair_ordinal") != expected_ordinal
        )
    ):
        raise StandardTrainingError(
            "Stage 6 source-repair summary drifted"
        )
    if expected_ordinal is not None and expected_ordinal >= 3:
        ordinal_artifacts = value.get("ordinal_artifacts")
        if (
            not isinstance(ordinal_artifacts, list)
            or len(ordinal_artifacts) != expected_ordinal
            or any(
                not isinstance(row, Mapping)
                or row.get("repair_ordinal") != index
                for index, row in enumerate(
                    ordinal_artifacts,
                    start=1,
                )
            )
        ):
            raise StandardTrainingError(
                "Stage 6 source-repair summary drifted"
            )
    return str(schema)


def build_stage6_acceptance_artifacts(
    *,
    global_best: Mapping[str, object],
    performance_advantage_established: bool,
    performance_claim: str,
    ppo_ci95_low: float,
    gain_over_cost_ci95_high: float,
    final_evaluation_count: int,
    final_episode_count: int,
    checkpoint_receipt_count: int,
    immutable_bindings: Mapping[str, object],
    source_repair_binding: Mapping[str, object] | None = None,
    acceptance_profile: Mapping[str, object] | None = None,
    planning_child_source_repair_binding: (
        Mapping[str, object] | None
    ) = None,
) -> dict[str, object]:
    """Build the one canonical Stage 6 success payload used by writer and replay."""

    source_repair_schema: str | None = None
    planning_profile: dict[str, object] | None = None
    planning_child_recovery: dict[str, object] | None = None
    if (
        planning_child_source_repair_binding is not None
        and source_repair_binding is not None
    ):
        raise StandardTrainingError(
            "Stage 6 planning child source-repair cannot use classic source repair"
        )
    source_repair_schema = (
        _validated_stage6_classic_source_repair_profile(
            source_repair_binding
        )
    )
    if acceptance_profile is not None:
        from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
            Stage6PlanningWarmStartError,
            validate_planning_child_acceptance_profile,
        )

        try:
            planning_profile = validate_planning_child_acceptance_profile(
                acceptance_profile
            )
        except Stage6PlanningWarmStartError as exc:
            raise StandardTrainingError(
                "Stage 6 acceptance profile drifted"
            ) from exc
        if source_repair_binding is not None:
            raise StandardTrainingError(
                "Stage 6 warm-start acceptance cannot use source repair"
            )
    if planning_child_source_repair_binding is not None:
        if planning_profile is None:
            raise StandardTrainingError(
                "Stage 6 planning child source-repair requires warm-start acceptance"
            )
        planning_child_recovery = (
            _validated_planning_child_recovery_evidence_binding(
                planning_child_source_repair_binding
            )
        )
    immutable_keys = set(immutable_bindings)
    expected_immutable_keys = (
        STAGE6_IMMUTABLE_BINDING_KEYS
        if (
            source_repair_schema == "stage6_source_repair_summary/v6"
            or planning_profile is not None
        )
        else _BASE_STAGE6_IMMUTABLE_BINDING_KEYS
    )
    immutable_key_set_valid = immutable_keys == expected_immutable_keys
    expected_receipt_count = (
        int(planning_profile["child_receipt_count"])
        if planning_profile is not None
        else 100
    )
    allowed_best_updates = (
        set(planning_profile["child_validation_updates"])
        if planning_profile is not None
        else None
    )
    try:
        record = ValidationRecord(
            seed=int(global_best["seed"]),
            update=int(global_best["update"]),
            success_rate_under_fixed_step_budget=float(
                global_best["success_rate_under_fixed_step_budget"]
            ),
            mean_final_coverage=float(global_best["mean_final_coverage"]),
            checkpoint_ref=str(global_best["checkpoint_ref"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StandardTrainingError("Stage 6 acceptance global best drifted") from exc
    if (
        set(global_best) != set(record.to_dict())
        or dict(global_best) != record.to_dict()
        or type(performance_advantage_established) is not bool
        or not isinstance(performance_claim, str)
        or not performance_claim
        or not isinstance(ppo_ci95_low, (int, float))
        or not math.isfinite(float(ppo_ci95_low))
        or not isinstance(gain_over_cost_ci95_high, (int, float))
        or not math.isfinite(float(gain_over_cost_ci95_high))
        or final_evaluation_count != 10
        or final_episode_count != 640
        or checkpoint_receipt_count != expected_receipt_count
        or (
            allowed_best_updates is not None
            and record.update not in allowed_best_updates
        )
        or not immutable_key_set_valid
        or not _immutable_bindings_valid(immutable_bindings)
    ):
        raise StandardTrainingError("Stage 6 acceptance payload drifted")
    immutable = dict(immutable_bindings)
    repaired: dict[str, object] | None = None
    if source_repair_binding is not None:
        from lunar_exploration_ppo.workflows.stage6_source_repair import (
            FIRST_CLOSURE_UPDATE,
            FIRST_FRONTIER_RECOVERY_UPDATE,
            FIRST_SENSOR_ACCELERATION_UPDATE,
            FIRST_REPAIRED_UPDATE,
            FIRST_SUPPLEMENT_UPDATE,
            FIXED_SEED,
            LAST_CONTINUATION_UPDATE,
            LAST_FRONTIER_RECOVERY_PARENT_UPDATE,
            LAST_SENSOR_ACCELERATION_PARENT_UPDATE,
            LAST_ORIGIN_UPDATE,
            SOURCE_REPAIR_AMENDMENT_NAME,
            SOURCE_REPAIR_CLOSURE_ID,
            SOURCE_REPAIR_CLOSURE_NAME,
            SOURCE_REPAIR_CONTINUATION_ID,
            SOURCE_REPAIR_CONTINUATION_NAME,
            SOURCE_REPAIR_FRONTIER_RECOVERY_ID,
            SOURCE_REPAIR_FRONTIER_RECOVERY_NAME,
            SOURCE_REPAIR_SENSOR_ACCELERATION_ID,
            SOURCE_REPAIR_SENSOR_ACCELERATION_NAME,
            SOURCE_REPAIR_SENSOR_ACCELERATION_SUMMARY_SCHEMA,
            SOURCE_REPAIR_ID,
            SOURCE_REPAIR_SUPPLEMENT_ID,
            SOURCE_REPAIR_SUPPLEMENT_NAME,
            _FIXED_CLOSURE_BINDING,
            _FIXED_CLOSURE_EXECUTION_IDENTITY_SHA256,
            _FIXED_CONTINUATION_BINDING,
            _FIXED_CONTINUATION_EXECUTION_IDENTITY_SHA256,
            _FIXED_COVERAGE_CACHE_MANIFEST_BINDING,
            _FIXED_FRONTIER_RECOVERY_BINDING,
            _FIXED_FRONTIER_RECOVERY_EXECUTION_IDENTITY_SHA256,
            _FIXED_PRIMARY_AMENDMENT_BINDING,
            _FIXED_SUPPLEMENT_BINDING,
            _FIXED_SUPPLEMENT_EXECUTION_IDENTITY_SHA256,
        )

        repaired = dict(source_repair_binding)
        common_fields = {
            "schema_version",
            "repair_id",
            "amendment_sha256",
            "seed",
            "last_origin_update",
            "first_repaired_update",
            "origin_execution_identity_sha256",
            "current_execution_identity_sha256",
            "origin_source_set_sha256",
            "current_source_set_sha256",
        }
        hash_fields = {
            "amendment_sha256",
            "origin_execution_identity_sha256",
            "current_execution_identity_sha256",
            "origin_source_set_sha256",
            "current_source_set_sha256",
        }
        schema = repaired.get("schema_version")
        if schema == "stage6_source_repair_summary/v1":
            expected_fields = common_fields
            continuation_valid = True
        elif schema == "stage6_source_repair_summary/v2":
            expected_fields = common_fields | {
                "continuation_id",
                "continuation_sha256",
                "repair_ordinal",
                "parent_execution_identity_sha256",
                "parent_source_set_sha256",
            }
            hash_fields |= {
                "continuation_sha256",
                "parent_execution_identity_sha256",
                "parent_source_set_sha256",
            }
            continuation_valid = (
                repaired.get("continuation_id") == SOURCE_REPAIR_CONTINUATION_ID
                and repaired.get("repair_ordinal") == 2
                and len(
                    {
                        repaired.get("origin_source_set_sha256"),
                        repaired.get("parent_source_set_sha256"),
                        repaired.get("current_source_set_sha256"),
                    }
                )
                == 3
            )
        elif schema == "stage6_source_repair_summary/v3":
            expected_fields = common_fields | {
                "continuation_id",
                "continuation_sha256",
                "repair_ordinal",
                "parent_execution_identity_sha256",
                "parent_source_set_sha256",
                "supplement_id",
                "supplement_sha256",
                "last_continuation_update",
                "first_supplement_update",
                "continuation_execution_identity_sha256",
                "continuation_source_set_sha256",
                "ordinal_artifacts",
            }
            hash_fields |= {
                "continuation_sha256",
                "parent_execution_identity_sha256",
                "parent_source_set_sha256",
                "supplement_sha256",
                "continuation_execution_identity_sha256",
                "continuation_source_set_sha256",
            }
            expected_ordinal_artifacts = [
                {
                    "repair_ordinal": 1,
                    "path": SOURCE_REPAIR_AMENDMENT_NAME,
                    "sha256": repaired.get("amendment_sha256"),
                },
                {
                    "repair_ordinal": 2,
                    "path": SOURCE_REPAIR_CONTINUATION_NAME,
                    "sha256": repaired.get("continuation_sha256"),
                },
                {
                    "repair_ordinal": 3,
                    "path": SOURCE_REPAIR_SUPPLEMENT_NAME,
                    "sha256": repaired.get("supplement_sha256"),
                },
            ]
            continuation_valid = (
                repaired.get("continuation_id") == SOURCE_REPAIR_CONTINUATION_ID
                and repaired.get("supplement_id") == SOURCE_REPAIR_SUPPLEMENT_ID
                and repaired.get("repair_ordinal") == 3
                and repaired.get("last_continuation_update")
                == LAST_CONTINUATION_UPDATE
                and repaired.get("first_supplement_update")
                == FIRST_SUPPLEMENT_UPDATE
                and repaired.get("amendment_sha256")
                == _FIXED_PRIMARY_AMENDMENT_BINDING.get("sha256")
                and repaired.get("continuation_sha256")
                == _FIXED_CONTINUATION_BINDING.get("sha256")
                and repaired.get("continuation_execution_identity_sha256")
                == _FIXED_CONTINUATION_EXECUTION_IDENTITY_SHA256
                and repaired.get("ordinal_artifacts")
                == expected_ordinal_artifacts
                and len(
                    {
                        repaired.get("origin_execution_identity_sha256"),
                        repaired.get("parent_execution_identity_sha256"),
                        repaired.get("continuation_execution_identity_sha256"),
                        repaired.get("current_execution_identity_sha256"),
                    }
                )
                == 4
                and len(
                    {
                        repaired.get("origin_source_set_sha256"),
                        repaired.get("parent_source_set_sha256"),
                        repaired.get("continuation_source_set_sha256"),
                        repaired.get("current_source_set_sha256"),
                    }
                )
                == 4
            )
        elif schema == "stage6_source_repair_summary/v4":
            expected_fields = common_fields | {
                "continuation_id",
                "continuation_sha256",
                "repair_ordinal",
                "parent_execution_identity_sha256",
                "parent_source_set_sha256",
                "supplement_id",
                "supplement_sha256",
                "last_continuation_update",
                "first_supplement_update",
                "continuation_execution_identity_sha256",
                "continuation_source_set_sha256",
                "closure_id",
                "closure_sha256",
                "first_closure_update",
                "supplement_execution_identity_sha256",
                "supplement_source_set_sha256",
                "ordinal_artifacts",
            }
            hash_fields |= {
                "continuation_sha256",
                "parent_execution_identity_sha256",
                "parent_source_set_sha256",
                "supplement_sha256",
                "continuation_execution_identity_sha256",
                "continuation_source_set_sha256",
                "closure_sha256",
                "supplement_execution_identity_sha256",
                "supplement_source_set_sha256",
            }
            expected_ordinal_artifacts = [
                {
                    "repair_ordinal": 1,
                    "path": SOURCE_REPAIR_AMENDMENT_NAME,
                    "sha256": repaired.get("amendment_sha256"),
                },
                {
                    "repair_ordinal": 2,
                    "path": SOURCE_REPAIR_CONTINUATION_NAME,
                    "sha256": repaired.get("continuation_sha256"),
                },
                {
                    "repair_ordinal": 3,
                    "path": SOURCE_REPAIR_SUPPLEMENT_NAME,
                    "sha256": repaired.get("supplement_sha256"),
                },
                {
                    "repair_ordinal": 4,
                    "path": SOURCE_REPAIR_CLOSURE_NAME,
                    "sha256": repaired.get("closure_sha256"),
                },
            ]
            continuation_valid = (
                repaired.get("continuation_id") == SOURCE_REPAIR_CONTINUATION_ID
                and repaired.get("supplement_id") == SOURCE_REPAIR_SUPPLEMENT_ID
                and repaired.get("closure_id") == SOURCE_REPAIR_CLOSURE_ID
                and repaired.get("repair_ordinal") == 4
                and repaired.get("last_continuation_update")
                == LAST_CONTINUATION_UPDATE
                and repaired.get("first_supplement_update")
                == FIRST_SUPPLEMENT_UPDATE
                and repaired.get("first_closure_update") == FIRST_CLOSURE_UPDATE
                and repaired.get("amendment_sha256")
                == _FIXED_PRIMARY_AMENDMENT_BINDING.get("sha256")
                and repaired.get("continuation_sha256")
                == _FIXED_CONTINUATION_BINDING.get("sha256")
                and repaired.get("continuation_execution_identity_sha256")
                == _FIXED_CONTINUATION_EXECUTION_IDENTITY_SHA256
                and repaired.get("supplement_sha256")
                == _FIXED_SUPPLEMENT_BINDING.get("sha256")
                and repaired.get("supplement_execution_identity_sha256")
                == _FIXED_SUPPLEMENT_EXECUTION_IDENTITY_SHA256
                and repaired.get("ordinal_artifacts")
                == expected_ordinal_artifacts
                and len(
                    {
                        repaired.get("origin_execution_identity_sha256"),
                        repaired.get("parent_execution_identity_sha256"),
                        repaired.get("continuation_execution_identity_sha256"),
                        repaired.get("supplement_execution_identity_sha256"),
                        repaired.get("current_execution_identity_sha256"),
                    }
                )
                == 5
                and len(
                    {
                        repaired.get("origin_source_set_sha256"),
                        repaired.get("parent_source_set_sha256"),
                        repaired.get("continuation_source_set_sha256"),
                        repaired.get("supplement_source_set_sha256"),
                        repaired.get("current_source_set_sha256"),
                    }
                )
                == 5
            )
        elif schema == "stage6_source_repair_summary/v5":
            expected_fields = common_fields | {
                "continuation_id",
                "continuation_sha256",
                "repair_ordinal",
                "parent_execution_identity_sha256",
                "parent_source_set_sha256",
                "supplement_id",
                "supplement_sha256",
                "last_continuation_update",
                "first_supplement_update",
                "continuation_execution_identity_sha256",
                "continuation_source_set_sha256",
                "closure_id",
                "closure_sha256",
                "first_closure_update",
                "supplement_execution_identity_sha256",
                "supplement_source_set_sha256",
                "frontier_recovery_id",
                "frontier_recovery_sha256",
                "last_closure_update",
                "first_frontier_recovery_update",
                "closure_execution_identity_sha256",
                "closure_source_set_sha256",
                "ordinal_artifacts",
            }
            hash_fields |= {
                "continuation_sha256",
                "parent_execution_identity_sha256",
                "parent_source_set_sha256",
                "supplement_sha256",
                "continuation_execution_identity_sha256",
                "continuation_source_set_sha256",
                "closure_sha256",
                "supplement_execution_identity_sha256",
                "supplement_source_set_sha256",
                "frontier_recovery_sha256",
                "closure_execution_identity_sha256",
                "closure_source_set_sha256",
            }
            expected_ordinal_artifacts = [
                {
                    "repair_ordinal": 1,
                    "path": SOURCE_REPAIR_AMENDMENT_NAME,
                    "sha256": repaired.get("amendment_sha256"),
                },
                {
                    "repair_ordinal": 2,
                    "path": SOURCE_REPAIR_CONTINUATION_NAME,
                    "sha256": repaired.get("continuation_sha256"),
                },
                {
                    "repair_ordinal": 3,
                    "path": SOURCE_REPAIR_SUPPLEMENT_NAME,
                    "sha256": repaired.get("supplement_sha256"),
                },
                {
                    "repair_ordinal": 4,
                    "path": SOURCE_REPAIR_CLOSURE_NAME,
                    "sha256": repaired.get("closure_sha256"),
                },
                {
                    "repair_ordinal": 5,
                    "path": SOURCE_REPAIR_FRONTIER_RECOVERY_NAME,
                    "sha256": repaired.get("frontier_recovery_sha256"),
                },
            ]
            continuation_valid = (
                repaired.get("continuation_id") == SOURCE_REPAIR_CONTINUATION_ID
                and repaired.get("supplement_id") == SOURCE_REPAIR_SUPPLEMENT_ID
                and repaired.get("closure_id") == SOURCE_REPAIR_CLOSURE_ID
                and repaired.get("frontier_recovery_id")
                == SOURCE_REPAIR_FRONTIER_RECOVERY_ID
                and repaired.get("repair_ordinal") == 5
                and repaired.get("last_continuation_update")
                == LAST_CONTINUATION_UPDATE
                and repaired.get("first_supplement_update")
                == FIRST_SUPPLEMENT_UPDATE
                and repaired.get("first_closure_update") == FIRST_CLOSURE_UPDATE
                and repaired.get("last_closure_update")
                == LAST_FRONTIER_RECOVERY_PARENT_UPDATE
                and repaired.get("first_frontier_recovery_update")
                == FIRST_FRONTIER_RECOVERY_UPDATE
                and repaired.get("amendment_sha256")
                == _FIXED_PRIMARY_AMENDMENT_BINDING.get("sha256")
                and repaired.get("continuation_sha256")
                == _FIXED_CONTINUATION_BINDING.get("sha256")
                and repaired.get("continuation_execution_identity_sha256")
                == _FIXED_CONTINUATION_EXECUTION_IDENTITY_SHA256
                and repaired.get("supplement_sha256")
                == _FIXED_SUPPLEMENT_BINDING.get("sha256")
                and repaired.get("supplement_execution_identity_sha256")
                == _FIXED_SUPPLEMENT_EXECUTION_IDENTITY_SHA256
                and repaired.get("closure_sha256")
                == _FIXED_CLOSURE_BINDING.get("sha256")
                and repaired.get("closure_execution_identity_sha256")
                == _FIXED_CLOSURE_EXECUTION_IDENTITY_SHA256
                and repaired.get("ordinal_artifacts")
                == expected_ordinal_artifacts
                and len(
                    {
                        repaired.get("origin_execution_identity_sha256"),
                        repaired.get("parent_execution_identity_sha256"),
                        repaired.get("continuation_execution_identity_sha256"),
                        repaired.get("supplement_execution_identity_sha256"),
                        repaired.get("closure_execution_identity_sha256"),
                        repaired.get("current_execution_identity_sha256"),
                    }
                )
                == 6
                and len(
                    {
                        repaired.get("origin_source_set_sha256"),
                        repaired.get("parent_source_set_sha256"),
                        repaired.get("continuation_source_set_sha256"),
                        repaired.get("supplement_source_set_sha256"),
                        repaired.get("closure_source_set_sha256"),
                        repaired.get("current_source_set_sha256"),
                    }
                )
                == 6
            )
        elif schema == SOURCE_REPAIR_SENSOR_ACCELERATION_SUMMARY_SCHEMA:
            expected_fields = common_fields | {
                "continuation_id",
                "continuation_sha256",
                "repair_ordinal",
                "parent_execution_identity_sha256",
                "parent_source_set_sha256",
                "supplement_id",
                "supplement_sha256",
                "last_continuation_update",
                "first_supplement_update",
                "continuation_execution_identity_sha256",
                "continuation_source_set_sha256",
                "closure_id",
                "closure_sha256",
                "first_closure_update",
                "supplement_execution_identity_sha256",
                "supplement_source_set_sha256",
                "frontier_recovery_id",
                "frontier_recovery_sha256",
                "last_closure_update",
                "first_frontier_recovery_update",
                "closure_execution_identity_sha256",
                "closure_source_set_sha256",
                "sensor_acceleration_id",
                "sensor_acceleration_sha256",
                "last_frontier_recovery_update",
                "first_sensor_acceleration_update",
                "frontier_recovery_execution_identity_sha256",
                "frontier_recovery_source_set_sha256",
                "coverage_cache_manifest",
                "ordinal_artifacts",
            }
            hash_fields |= {
                "continuation_sha256",
                "parent_execution_identity_sha256",
                "parent_source_set_sha256",
                "supplement_sha256",
                "continuation_execution_identity_sha256",
                "continuation_source_set_sha256",
                "closure_sha256",
                "supplement_execution_identity_sha256",
                "supplement_source_set_sha256",
                "frontier_recovery_sha256",
                "closure_execution_identity_sha256",
                "closure_source_set_sha256",
                "sensor_acceleration_sha256",
                "frontier_recovery_execution_identity_sha256",
                "frontier_recovery_source_set_sha256",
            }
            expected_ordinal_artifacts = [
                {
                    "repair_ordinal": 1,
                    "path": SOURCE_REPAIR_AMENDMENT_NAME,
                    "sha256": repaired.get("amendment_sha256"),
                },
                {
                    "repair_ordinal": 2,
                    "path": SOURCE_REPAIR_CONTINUATION_NAME,
                    "sha256": repaired.get("continuation_sha256"),
                },
                {
                    "repair_ordinal": 3,
                    "path": SOURCE_REPAIR_SUPPLEMENT_NAME,
                    "sha256": repaired.get("supplement_sha256"),
                },
                {
                    "repair_ordinal": 4,
                    "path": SOURCE_REPAIR_CLOSURE_NAME,
                    "sha256": repaired.get("closure_sha256"),
                },
                {
                    "repair_ordinal": 5,
                    "path": SOURCE_REPAIR_FRONTIER_RECOVERY_NAME,
                    "sha256": repaired.get("frontier_recovery_sha256"),
                },
                {
                    "repair_ordinal": 6,
                    "path": SOURCE_REPAIR_SENSOR_ACCELERATION_NAME,
                    "sha256": repaired.get("sensor_acceleration_sha256"),
                },
            ]
            cache = repaired.get("coverage_cache_manifest")
            expected_cache = {
                **dict(_FIXED_COVERAGE_CACHE_MANIFEST_BINDING),
                "runtime_mode": "persistent_exact_manifest_read_only/v1",
            }
            continuation_valid = (
                repaired.get("continuation_id") == SOURCE_REPAIR_CONTINUATION_ID
                and repaired.get("supplement_id") == SOURCE_REPAIR_SUPPLEMENT_ID
                and repaired.get("closure_id") == SOURCE_REPAIR_CLOSURE_ID
                and repaired.get("frontier_recovery_id")
                == SOURCE_REPAIR_FRONTIER_RECOVERY_ID
                and repaired.get("sensor_acceleration_id")
                == SOURCE_REPAIR_SENSOR_ACCELERATION_ID
                and repaired.get("repair_ordinal") == 6
                and repaired.get("last_continuation_update")
                == LAST_CONTINUATION_UPDATE
                and repaired.get("first_supplement_update")
                == FIRST_SUPPLEMENT_UPDATE
                and repaired.get("first_closure_update") == FIRST_CLOSURE_UPDATE
                and repaired.get("last_closure_update")
                == LAST_FRONTIER_RECOVERY_PARENT_UPDATE
                and repaired.get("first_frontier_recovery_update")
                == FIRST_FRONTIER_RECOVERY_UPDATE
                and repaired.get("last_frontier_recovery_update")
                == LAST_SENSOR_ACCELERATION_PARENT_UPDATE
                and repaired.get("first_sensor_acceleration_update")
                == FIRST_SENSOR_ACCELERATION_UPDATE
                and repaired.get("amendment_sha256")
                == _FIXED_PRIMARY_AMENDMENT_BINDING.get("sha256")
                and repaired.get("continuation_sha256")
                == _FIXED_CONTINUATION_BINDING.get("sha256")
                and repaired.get("continuation_execution_identity_sha256")
                == _FIXED_CONTINUATION_EXECUTION_IDENTITY_SHA256
                and repaired.get("supplement_sha256")
                == _FIXED_SUPPLEMENT_BINDING.get("sha256")
                and repaired.get("supplement_execution_identity_sha256")
                == _FIXED_SUPPLEMENT_EXECUTION_IDENTITY_SHA256
                and repaired.get("closure_sha256")
                == _FIXED_CLOSURE_BINDING.get("sha256")
                and repaired.get("closure_execution_identity_sha256")
                == _FIXED_CLOSURE_EXECUTION_IDENTITY_SHA256
                and repaired.get("frontier_recovery_sha256")
                == _FIXED_FRONTIER_RECOVERY_BINDING.get("sha256")
                and repaired.get("frontier_recovery_execution_identity_sha256")
                == _FIXED_FRONTIER_RECOVERY_EXECUTION_IDENTITY_SHA256
                and repaired.get("ordinal_artifacts")
                == expected_ordinal_artifacts
                and isinstance(cache, Mapping)
                and dict(cache) == expected_cache
                and len(
                    {
                        repaired.get("origin_execution_identity_sha256"),
                        repaired.get("parent_execution_identity_sha256"),
                        repaired.get("continuation_execution_identity_sha256"),
                        repaired.get("supplement_execution_identity_sha256"),
                        repaired.get("closure_execution_identity_sha256"),
                        repaired.get("frontier_recovery_execution_identity_sha256"),
                        repaired.get("current_execution_identity_sha256"),
                    }
                )
                == 7
                and len(
                    {
                        repaired.get("origin_source_set_sha256"),
                        repaired.get("parent_source_set_sha256"),
                        repaired.get("continuation_source_set_sha256"),
                        repaired.get("supplement_source_set_sha256"),
                        repaired.get("closure_source_set_sha256"),
                        repaired.get("frontier_recovery_source_set_sha256"),
                        repaired.get("current_source_set_sha256"),
                    }
                )
                == 7
            )
        else:
            expected_fields = set()
            continuation_valid = False
        if (
            set(repaired) != expected_fields
            or repaired.get("repair_id") != SOURCE_REPAIR_ID
            or repaired.get("seed") != FIXED_SEED
            or repaired.get("last_origin_update") != LAST_ORIGIN_UPDATE
            or repaired.get("first_repaired_update") != FIRST_REPAIRED_UPDATE
            or any(not _is_sha256(repaired.get(name)) for name in hash_fields)
            or repaired["origin_source_set_sha256"]
            == repaired["current_source_set_sha256"]
            or not continuation_valid
        ):
            raise StandardTrainingError("Stage 6 source-repair summary drifted")
    summary = {
        "schema_version": "ppo_highres_frontier_stage6_summary/v1",
        "state": "awaiting_independent_review",
        "machine_passed": True,
        "acceptance_scope": "single_seed_system_closure/v1",
        "cross_seed_performance_conclusion": False,
        "optional_seed_extension_blocks_next_stage": False,
        "optional_seed_extension_trigger": "explicit_user_request_only/v1",
        "performance_advantage_established": performance_advantage_established,
        "performance_claim": performance_claim,
        "ppo_ci95_low": float(ppo_ci95_low),
        "gain_over_cost_ci95_high": float(gain_over_cost_ci95_high),
        "global_best": record.to_dict(),
        "final_evaluation_count": final_evaluation_count,
        "final_episode_count": final_episode_count,
        "checkpoint_receipt_count": checkpoint_receipt_count,
        **immutable,
    }
    if planning_profile is not None:
        summary.update(
            {
                "acceptance_scope": (
                    "single_seed_planning_u74_warm_start_closure/v1"
                ),
                "acceptance_profile": dict(planning_profile),
                "parent_semantics_update_count": planning_profile[
                    "parent_semantics_update_count"
                ],
                "child_new_semantics_update_count": planning_profile[
                    "child_new_semantics_update_count"
                ],
                "update_semantics": planning_profile["update_semantics"],
            }
        )
    if repaired is not None:
        summary["source_repair"] = repaired
    if planning_child_recovery is not None:
        summary["planning_child_recovery"] = dict(
            planning_child_recovery
        )
    routing = {
        "schema_version": "ppo_highres_frontier_stage6_routing/v1",
        "state": "awaiting_independent_review",
        "route": "awaiting_independent_review",
        "machine_passed": True,
        "performance_advantage_established": performance_advantage_established,
        "next_stage_after_human_approval": (
            "ppo_highres_frontier_stage7_kilometer_stress/v1"
        ),
    }
    if planning_profile is not None:
        routing.update(
            {
                "acceptance_profile": dict(planning_profile),
                "parent_semantics_update_count": planning_profile[
                    "parent_semantics_update_count"
                ],
                "child_new_semantics_update_count": planning_profile[
                    "child_new_semantics_update_count"
                ],
                "update_semantics": planning_profile["update_semantics"],
            }
        )
    if planning_child_recovery is not None:
        routing["planning_child_recovery"] = dict(
            planning_child_recovery
        )
    semantics_line = (
        (
            "\n"
            f"{planning_profile['update_semantics']}.\n"
        )
        if planning_profile is not None
        else ""
    )
    reports = {
        "standard_training_report.md": (
            "# Standard v1 正式训练\n\n"
            f"全局最佳：seed {record.seed}，update {record.update}。\n\n"
            f"完整 checkpoint receipt：{checkpoint_receipt_count}。\n"
            f"{semantics_line}"
        ).encode("utf-8"),
        "standard_eval_report.md": (
            "# Standard v1 最终评估\n\n"
            f"完成 {final_evaluation_count} 个方法/数据集组合，共 "
            f"{final_episode_count} 个 episode。\n\n"
            f"性能结论：{performance_claim}。\n"
            f"{semantics_line}"
        ).encode("utf-8"),
        "report.md": (
            "# Stage 6 Standard v1\n\n"
            "机器验收已完成，状态为 awaiting_independent_review。\n\n"
            "单 seed 只证明系统闭环，不形成跨 seed 性能结论；"
            "可选追加不阻塞 Stage 7/8。\n\n"
            "未写入 review、approval 或 gate authority artifact。\n"
            f"{semantics_line}"
        ).encode("utf-8"),
    }
    return {"summary": summary, "routing": routing, "reports": reports}


def audit_required(update: int) -> bool:
    if type(update) is not int or not 1 <= update <= 100:
        raise StandardTrainingError("audit update must be within 1..100")
    return update in AUDIT_UPDATES


def validate_math_audit(value: Mapping[str, object]) -> dict[str, object]:
    finite_fields = (
        "observation_finite",
        "action_finite",
        "logprob_finite",
        "value_finite",
        "advantage_finite",
        "return_finite",
        "ratio_finite",
        "loss_finite",
        "kl_finite",
        "grad_finite",
    )
    count_fields = (
        "mask_violation_count",
        "snapshot_mismatch_count",
        "stale_policy_transition_count",
    )
    expected_keys = {
        *finite_fields,
        *count_fields,
        "initial_ratio_max_abs_error",
        "device",
        "allowed_initial_ratio_tolerance",
        "compute_dtype",
        "joint_logprob",
        "joint_logprob_factorization_max_abs_error",
        "grad_post_clip_norm_max",
        "evidence_source",
        "evidence_sha256",
        "sample_count",
        "snapshot_list_sha256",
        "initial_forward_sample_count",
        "forward_sample_count",
        "loss_sample_count",
        "gradient_step_count",
    }
    if set(value) != expected_keys:
        raise StandardTrainingError("math audit field set drifted")
    if any(value[field] is not True for field in finite_fields):
        raise StandardTrainingError("math audit detected nonfinite values")
    for field in count_fields:
        if type(value[field]) is not int or value[field] != 0:
            label = "stale" if field == "stale_policy_transition_count" else field
            raise StandardTrainingError(f"math audit {label} violation")
    ratio_error = value["initial_ratio_max_abs_error"]
    device = value["device"]
    tolerance = value["allowed_initial_ratio_tolerance"]
    expected_tolerance = {"cpu": 1e-6, "cuda": 1e-5}.get(device)
    if expected_tolerance is None or tolerance != expected_tolerance:
        raise StandardTrainingError("initial ratio device tolerance drifted")
    grad_norm = value["grad_post_clip_norm_max"]
    if (
        isinstance(ratio_error, bool)
        or not isinstance(ratio_error, (int, float))
        or not math.isfinite(float(ratio_error))
        or float(ratio_error) > expected_tolerance
    ):
        raise StandardTrainingError("initial ratio audit tolerance exceeded")
    joint_error = value["joint_logprob_factorization_max_abs_error"]
    if (
        value["compute_dtype"] != "float32"
        or value["joint_logprob"] is not True
        or isinstance(joint_error, bool)
        or not isinstance(joint_error, (int, float))
        or not math.isfinite(float(joint_error))
        or not 0.0 <= float(joint_error) <= 1.0e-6
    ):
        raise StandardTrainingError("FP32 joint logprob audit failed")
    if (
        isinstance(grad_norm, bool)
        or not isinstance(grad_norm, (int, float))
        or not math.isfinite(float(grad_norm))
        or float(grad_norm) > 0.500001
    ):
        raise StandardTrainingError("post-clip grad norm audit failed")
    if (
        value["evidence_source"] != "ppo_update_math_evidence/v1"
        or not _is_sha256(value["evidence_sha256"])
        or not _is_sha256(value["snapshot_list_sha256"])
        or type(value["sample_count"]) is not int
        or value["sample_count"] != 1024
        or value["initial_forward_sample_count"] != value["sample_count"]
        or type(value["forward_sample_count"]) is not int
        or value["forward_sample_count"] < value["sample_count"]
        or type(value["loss_sample_count"]) is not int
        or value["loss_sample_count"] <= 0
        or type(value["gradient_step_count"]) is not int
        or value["gradient_step_count"] <= 0
    ):
        raise StandardTrainingError("math audit runtime evidence drifted")
    return {"schema_version": "stage6_math_audit/v1", "passed": True, **dict(value)}


def _math_audit_from_completed_update(
    update_metrics: object,
    *,
    collection_audit: object,
    device: str,
    compute_dtype: str,
    require_dual_scan: bool = False,
) -> dict[str, object]:
    from lunar_exploration_ppo.ppo.trainer import (
        PPOTrainingError,
        validate_ppo_math_evidence,
    )

    metrics = _jsonable_runtime_value(update_metrics)
    if not isinstance(metrics, Mapping):
        raise StandardTrainingError("PPO update metrics are not auditable")
    required_numbers = (
        "initial_ratio_max_abs_error",
        "policy_loss",
        "value_loss",
        "approx_kl",
        "grad_pre_clip_norm_max",
        "grad_post_clip_norm_max",
        "optimizer_steps",
    )
    if any(
        isinstance(metrics.get(name), bool)
        or not isinstance(metrics.get(name), (int, float))
        or not math.isfinite(float(metrics[name]))
        for name in required_numbers
    ):
        raise StandardTrainingError("PPO update math audit detected nonfinite metrics")
    policy_before = metrics.get("policy_state_sha256_before")
    policy_after = metrics.get("policy_state_sha256_after")
    if not _is_sha256(policy_before) or not _is_sha256(policy_after):
        raise StandardTrainingError("PPO update policy lineage is missing")
    collection = validate_standard_collection_audit(
        collection_audit,
        expected_device=device,
        expected_policy_sha256=policy_before,
        require_dual_scan=require_dual_scan,
    )
    optimizer_steps = metrics.get("optimizer_steps")
    if type(optimizer_steps) is not int or optimizer_steps <= 0:
        raise StandardTrainingError("PPO optimizer step evidence is missing")
    snapshot_hashes = collection.get("snapshot_sha256")
    if not isinstance(snapshot_hashes, list):
        raise StandardTrainingError("collection snapshot evidence is missing")
    try:
        evidence = validate_ppo_math_evidence(
            metrics.get("math_evidence"),
            expected_policy_state_sha256=str(policy_before),
            expected_snapshot_hashes=snapshot_hashes,
            expected_sample_count=int(collection["trainable_transition_count"]),
            expected_optimizer_steps=optimizer_steps,
        )
    except PPOTrainingError as exc:
        raise StandardTrainingError("PPO update math evidence drifted") from exc
    observed_dtypes = evidence["observed_compute_dtypes"]
    joint_error = float(evidence["joint_logprob_factorization_max_abs_error"])
    math_audit = validate_math_audit(
        {
            "observation_finite": evidence["observation_finite"] is True,
            "action_finite": evidence["action_finite"] is True,
            "logprob_finite": (
                evidence["old_logprob_finite"] is True
                and evidence["new_logprob_finite"] is True
            ),
            "value_finite": (
                evidence["old_value_finite"] is True
                and evidence["new_value_finite"] is True
            ),
            "advantage_finite": evidence["advantage_finite"] is True,
            "return_finite": evidence["return_finite"] is True,
            "ratio_finite": evidence["ratio_finite"] is True,
            "loss_finite": evidence["loss_finite"] is True,
            "kl_finite": evidence["kl_finite"] is True,
            "grad_finite": evidence["grad_finite"] is True,
            "mask_violation_count": evidence["mask_violation_count"],
            "snapshot_mismatch_count": evidence["snapshot_mismatch_count"],
            "stale_policy_transition_count": evidence[
                "stale_policy_transition_count"
            ],
            "initial_ratio_max_abs_error": float(
                metrics["initial_ratio_max_abs_error"]
            ),
            "device": device,
            "allowed_initial_ratio_tolerance": (
                1e-5 if device == "cuda" else 1e-6
            ),
            "compute_dtype": (
                compute_dtype if observed_dtypes == ["float32"] else "drifted"
            ),
            "joint_logprob": joint_error <= 1.0e-6,
            "joint_logprob_factorization_max_abs_error": joint_error,
            "grad_post_clip_norm_max": float(
                metrics["grad_post_clip_norm_max"]
            ),
            "evidence_source": evidence["schema_version"],
            "evidence_sha256": evidence["evidence_sha256"],
            "sample_count": evidence["sample_count"],
            "snapshot_list_sha256": evidence["snapshot_list_sha256"],
            "initial_forward_sample_count": evidence[
                "initial_forward_sample_count"
            ],
            "forward_sample_count": evidence["forward_sample_count"],
            "loss_sample_count": evidence["loss_sample_count"],
            "gradient_step_count": evidence["gradient_step_count"],
        }
    )
    return {
        **math_audit,
        "collection_audit": collection,
        "policy_state_sha256_before": policy_before,
        "policy_state_sha256_after": policy_after,
        "ppo_math_evidence": evidence,
    }


def validate_checkpoint_runtime_payload(value: Mapping[str, object]) -> dict[str, object]:
    expected_keys = {
        "normalization_stats",
        "scenario_sampler_state",
        "vector_env_states",
        "best_record",
        "config_sha256",
        "lineage",
        "eval_metrics",
    }
    if set(value) != expected_keys:
        raise StandardTrainingError("Stage 6 checkpoint runtime field set drifted")
    mappings = (
        "normalization_stats",
        "scenario_sampler_state",
        "best_record",
        "lineage",
        "eval_metrics",
    )
    if any(not isinstance(value[field], Mapping) for field in mappings):
        raise StandardTrainingError("Stage 6 checkpoint runtime mapping drifted")
    vector_states = value["vector_env_states"]
    if (
        not isinstance(vector_states, Sequence)
        or isinstance(vector_states, (str, bytes))
        or len(vector_states) != 8
        or any(not isinstance(state, Mapping) for state in vector_states)
    ):
        raise StandardTrainingError("checkpoint requires eight vector-env episode states")
    config_hash = value["config_sha256"]
    if not _is_sha256(config_hash):
        raise StandardTrainingError("checkpoint config SHA-256 drifted")
    lineage = value["lineage"]
    if "stage5_gate_sha256" not in lineage or not _is_sha256(lineage["stage5_gate_sha256"]):
        raise StandardTrainingError("checkpoint Stage 5 lineage drifted")
    return {
        "schema_version": "stage6_checkpoint_runtime_contract/v1",
        "passed": True,
        "vector_env_state_count": len(vector_states),
        "checkpoint_manager_captures_policy_optimizer_rng": True,
    }


def validate_checkpoint_replay_audit(
    value: Mapping[str, object],
    *,
    transaction_key: str,
    seed: int,
    update: int,
    checkpoint_sha256: str,
    complete_marker_sha256: str,
    policy_state_sha256: str,
    lineage_sha256: str,
) -> dict[str, object]:
    boolean_fields = {
        "passed",
        "last_complete_loaded",
        "policy_state_unchanged",
        "optimizer_state_unchanged",
        "rng_state_unchanged",
        "normalizer_restored",
        "scenario_sampler_state_restored",
        "vector_env_state_round_trip_bit_exact",
        "observation_round_trip_bit_exact",
        "deterministic_action_bit_exact",
        "lineage_unchanged",
    }
    expected_fields = {
        "transaction_key",
        "seed",
        "update",
        "schema_version",
        "vector_env_state_count",
        "checkpoint_sha256",
        "complete_marker_sha256",
        "policy_state_sha256",
        "lineage_sha256",
        *boolean_fields,
    }
    expected_hashes = {
        "checkpoint_sha256": checkpoint_sha256,
        "complete_marker_sha256": complete_marker_sha256,
        "policy_state_sha256": policy_state_sha256,
        "lineage_sha256": lineage_sha256,
    }
    if (
        set(value) != expected_fields
        or value.get("schema_version")
        != "stage6_checkpoint_replay_audit/v1"
        or value.get("transaction_key") != transaction_key
        or value.get("seed") != seed
        or value.get("update") != update
        or value.get("vector_env_state_count") != 8
        or any(value.get(field) is not True for field in boolean_fields)
        or any(not _is_sha256(digest) for digest in expected_hashes.values())
        or any(value.get(name) != digest for name, digest in expected_hashes.items())
    ):
        raise StandardTrainingError("checkpoint replay audit binding drifted")
    return dict(value)


def _default_production_components() -> dict[str, object]:
    from lunar_exploration_ppo.env.standard_training import (
        build_standard_catalog,
        standard_env_specs,
    )
    from lunar_exploration_ppo.ppo.checkpoint import CheckpointManager
    from lunar_exploration_ppo.ppo.checkpoint_retention import (
        CheckpointRetentionManager,
    )
    from lunar_exploration_ppo.ppo.collector import (
        CollectorContract,
        RolloutCollector,
        SpawnVectorEnv,
    )
    from lunar_exploration_ppo.ppo.trainer import PPOTrainer
    from lunar_exploration_ppo.eval.standard import run_standard_evaluation
    from lunar_exploration_ppo.utils.resources import (
        capture_resource_snapshot,
        evaluate_resource_gates,
    )
    from lunar_exploration_ppo.workflows.stage6 import (
        CANONICAL_STAGE4_CHECKPOINT,
        Stage6StateJournal,
        load_stage4_policy_for_standard,
    )
    from lunar_exploration_ppo.workflows.stage4_machine import (
        deterministic_action_record,
    )

    return {
        "build_standard_catalog": build_standard_catalog,
        "load_stage4_policy_for_standard": load_stage4_policy_for_standard,
        "stage4_checkpoint_path": CANONICAL_STAGE4_CHECKPOINT,
        "standard_env_specs": standard_env_specs,
        "SpawnVectorEnv": SpawnVectorEnv,
        "RolloutCollector": RolloutCollector,
        "CollectorContract": CollectorContract,
        "PPOTrainer": PPOTrainer,
        "CheckpointManager": CheckpointManager,
        "CheckpointRetentionManager": CheckpointRetentionManager,
        "CheckpointReceiptIndex": CheckpointReceiptIndex,
        "Stage6StateJournal": Stage6StateJournal,
        "run_standard_update_transaction": run_standard_update_transaction,
        "run_standard_evaluation": _run_cached_standard_evaluation,
        "verify_standard_final_evaluation_artifacts": (
            verify_standard_final_evaluation_artifacts
        ),
        "run_eval_only_transaction": run_eval_only_transaction,
        "capture_resource_snapshot": capture_resource_snapshot,
        "evaluate_resource_gates": evaluate_resource_gates,
        "append_transaction_metric_once": append_transaction_metric_once,
        "deterministic_action_record": deterministic_action_record,
    }


def _build_cached_standard_training_env(
    *,
    coverage_cache_manifest_path: str,
    coverage_cache_manifest_sha256: str,
    **kwargs: object,
):
    """Spawn-safe factory that installs the strict manifest before first reset."""

    from lunar_exploration_ppo.env.coverage_cache import Stage6CoverageManifest
    from lunar_exploration_ppo.env.standard_training import (
        _validate_coverage_cache_manifest_binding,
        build_standard_training_env,
    )

    if (
        not isinstance(coverage_cache_manifest_path, str)
        or not coverage_cache_manifest_path
        or not _is_sha256(coverage_cache_manifest_sha256)
    ):
        raise StandardTrainingError(
            "production coverage cache manifest binding is missing"
        )
    environment = build_standard_training_env(**kwargs)  # type: ignore[arg-type]
    manifest = Stage6CoverageManifest.load(
        coverage_cache_manifest_path,
        expected_sha256=coverage_cache_manifest_sha256,
    )
    _validate_coverage_cache_manifest_binding(manifest, environment.catalog)
    environment._coverage_cache_manifest = manifest
    return environment


def _build_cached_standard_evaluation_env(
    *,
    coverage_cache_manifest_path: str,
    coverage_cache_manifest_sha256: str,
    **kwargs: object,
):
    """Spawn-safe evaluation factory with the same strict pre-reset cache load."""

    from lunar_exploration_ppo.env.coverage_cache import Stage6CoverageManifest
    from lunar_exploration_ppo.env.standard_training import (
        _validate_coverage_cache_manifest_binding,
        build_standard_evaluation_env,
    )

    if (
        not isinstance(coverage_cache_manifest_path, str)
        or not coverage_cache_manifest_path
        or not _is_sha256(coverage_cache_manifest_sha256)
    ):
        raise StandardTrainingError(
            "production evaluation coverage cache manifest binding is missing"
        )
    environment = build_standard_evaluation_env(**kwargs)  # type: ignore[arg-type]
    manifest = Stage6CoverageManifest.load(
        coverage_cache_manifest_path,
        expected_sha256=coverage_cache_manifest_sha256,
    )
    _validate_coverage_cache_manifest_binding(manifest, environment.catalog)
    environment._coverage_cache_manifest = manifest
    return environment


def _cache_bound_standard_evaluation_specs(
    legacy_specs: Sequence[object],
    *,
    coverage_cache_manifest_path: str,
    coverage_cache_manifest_sha256: str,
) -> tuple[object, ...]:
    from lunar_exploration_ppo.ppo.collector import SpawnEnvSpec

    specs = tuple(legacy_specs)
    if (
        len(specs) != 8
        or not isinstance(coverage_cache_manifest_path, str)
        or not coverage_cache_manifest_path
        or not _is_sha256(coverage_cache_manifest_sha256)
        or any(not isinstance(getattr(spec, "kwargs", None), Mapping) for spec in specs)
    ):
        raise StandardTrainingError(
            "production evaluation coverage cache spawn spec drifted"
        )
    return tuple(
        SpawnEnvSpec(
            factory=_build_cached_standard_evaluation_env,
            kwargs={
                **dict(spec.kwargs),
                "coverage_cache_manifest_path": coverage_cache_manifest_path,
                "coverage_cache_manifest_sha256": coverage_cache_manifest_sha256,
            },
        )
        for spec in specs
    )


def _run_cached_standard_evaluation(
    *,
    catalog: object,
    split: str,
    method: str,
    episode_count: int,
    evaluation_seed_start: int,
    policy: object,
    policy_device: object,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    trace_path: str | Path,
    safety_contract: object,
    config_sha256: str,
    coverage_cache_manifest_path: str,
    coverage_cache_manifest_sha256: str,
    resource_guard: Callable[[str], None] | None = None,
):
    """Run the frozen evaluation schedule with cache-only spawned envs."""

    from lunar_exploration_ppo.eval import standard as evaluation_module
    from lunar_exploration_ppo.ppo.collector import SpawnVectorEnv

    theta_source = (
        "policy_theta_mu/v1"
        if method == "ppo_policy"
        else "candidate_recommended_theta/v1"
    )
    schedule = evaluation_module.build_standard_evaluation_schedule(
        catalog,
        split=split,
        episode_count=episode_count,
        evaluation_seed_start=evaluation_seed_start,
        theta_source=theta_source,
    )
    if method not in evaluation_module.ALL_METHODS:
        raise evaluation_module.StandardEvaluationError(
            "Standard evaluation method drifted"
        )
    legacy_specs = evaluation_module.standard_evaluation_env_specs(
        catalog,
        schedule,
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )
    shared_environment_contract = (
        evaluation_module.build_standard_environment_contract(
            catalog=catalog,
            env_specs=legacy_specs,
            jobs=schedule,
            max_steps=128,
            success_threshold=0.99,
            safety_contract=safety_contract,
            config_sha256=config_sha256,
        )
    )
    specs = _cache_bound_standard_evaluation_specs(
        legacy_specs,
        coverage_cache_manifest_path=coverage_cache_manifest_path,
        coverage_cache_manifest_sha256=coverage_cache_manifest_sha256,
    )
    vector = SpawnVectorEnv(specs, timeout_seconds=900.0)  # type: ignore[arg-type]
    try:
        result = evaluation_module._run_parallel_episode_batch(
            vector_env=vector,
            jobs=schedule,
            method=method,
            policy=policy,
            policy_device=policy_device,
            max_steps=128,
            success_threshold=0.99,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            trace_path=trace_path,
            shared_environment_contract=shared_environment_contract,
            safety_contract=safety_contract,
            config_sha256=config_sha256,
            resource_guard=resource_guard,
        )
    except BaseException as primary:
        try:
            vector.close()
        except BaseException as secondary:
            evaluation_module._attach_cleanup_secondary_note(
                primary,
                operation="vector.close",
                secondary=secondary,
            )
        raise
    vector.close()
    return result


def _resource_gate_record(snapshot: object, decision: object) -> dict[str, object]:
    try:
        return {
            "d_free_bytes": int(snapshot.d_free_bytes),
            "rss_bytes": int(snapshot.rss_bytes),
            "peak_vram_bytes": int(snapshot.peak_vram_bytes),
            "rss_source": str(snapshot.rss_source),
            "rss_root_pid": int(snapshot.rss_root_pid),
            "rss_sample_count": int(snapshot.rss_sample_count),
            "rss_latest_process_count": int(snapshot.rss_latest_process_count),
            "rss_peak_process_count": int(snapshot.rss_peak_process_count),
            "warnings": list(decision.warnings),
            "hard_stops": list(decision.hard_stops),
            "passed": decision.passed is True,
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise StandardTrainingError("resource gate result drifted") from exc


class StandardProductionBackend:
    """Production-only backend for the fixed Standard v1 schedule."""

    @_guarded_backend_constructor
    def __init__(
        self,
        *,
        config: Stage6Config,
        run_root: str | Path,
        repo_root: str | Path,
        stage5_authority: Mapping[str, object],
        execution_capability: object,
        _components: Mapping[str, object] | None = None,
        _immutable_bindings: Mapping[str, object] | None = None,
        _source_repair: object | None = None,
        _planning_warm_start: object | None = None,
        _planning_warm_start_sha256: str | None = None,
        _planning_child_recovery_capability: (
            PlanningChildRecoveryCapability | None
        ) = None,
        _verified_parent_u74: object | None = None,
        _process_tree_monitor: object | None = None,
        _resource_segment_start: Mapping[str, object] | None = None,
    ) -> None:
        _require_standard_execution_capability(
            execution_capability,
            label="StandardProductionBackend constructor entry",
            config=config,
            run_root=run_root,
            repo_root=repo_root,
            stage5_authority=stage5_authority,
        )
        self._execution_capability = execution_capability
        if not isinstance(config, Stage6Config) or not isinstance(
            stage5_authority, Mapping
        ):
            raise StandardTrainingError("production backend identity drifted")
        self.config = config
        try:
            run_candidate = lexical_absolute(run_root)
            require_plain_path(
                run_candidate,
                allow_missing=True,
                label="Stage 6 run root",
            )
            run_candidate.mkdir(parents=True, exist_ok=True)
            self.run_root = require_plain_path(
                run_candidate,
                leaf_kind="directory",
                label="Stage 6 run root",
            )
            stage_candidate = self.run_root / "s6"
            require_plain_path(
                stage_candidate,
                base=self.run_root,
                allow_missing=True,
                label="Stage 6 stage root",
            )
            stage_candidate.mkdir(parents=False, exist_ok=True)
            self.stage_root = require_plain_path(
                stage_candidate,
                base=self.run_root,
                leaf_kind="directory",
                label="Stage 6 stage root",
            )
        except (OSError, PathSecurityError) as exc:
            raise StandardTrainingError(
                "Stage 6 backend path contains a link or reparse point"
            ) from exc
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.stage5_authority = dict(stage5_authority)
        for relative in (
            "checkpoints/index.jsonl",
            "job-state.jsonl",
            "phase-state.jsonl",
            "metrics.jsonl",
            "resource_audit.jsonl",
            "math_audit.jsonl",
            "checkpoint_audit.jsonl",
        ):
            try:
                journal_path = self.stage_root / relative
                journal_path.parent.mkdir(parents=True, exist_ok=True)
                DurableJsonl(journal_path).recover()
            except (DurableJsonlError, OSError) as exc:
                raise StandardTrainingError(
                    f"Stage 6 durable JSONL recovery failed: {relative}"
                ) from exc
        self._components = dict(
            _components if _components is not None else _default_production_components()
        )
        required = {
            "build_standard_catalog",
            "load_stage4_policy_for_standard",
            "stage4_checkpoint_path",
            "standard_env_specs",
            "SpawnVectorEnv",
            "RolloutCollector",
            "CollectorContract",
            "PPOTrainer",
            "CheckpointManager",
            "CheckpointReceiptIndex",
            "Stage6StateJournal",
        }
        if not required.issubset(self._components):
            raise StandardTrainingError("production component contract drifted")
        self._immutable_bindings = (
            dict(_immutable_bindings) if _immutable_bindings is not None else None
        )
        self._source_repair = _source_repair
        self._planning_warm_start = _planning_warm_start
        self._planning_warm_start_sha256 = _planning_warm_start_sha256
        self._planning_child_recovery_capability = (
            _planning_child_recovery_capability
        )
        self._planning_child_resume_boundary_consumed = False
        self._verified_parent_u74 = _verified_parent_u74
        self._process_tree_monitor = _process_tree_monitor
        self._resource_segment_start = (
            dict(_resource_segment_start)
            if _resource_segment_start is not None
            else None
        )
        if self._immutable_bindings is not None and (
            not _immutable_bindings_valid(self._immutable_bindings)
        ):
            raise StandardTrainingError("production immutable binding drifted")
        if self._planning_warm_start is not None:
            from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
                PlanningWarmStartContext,
                Stage6PlanningWarmStartError,
                VerifiedParentU74,
            )

            if (
                self._source_repair is not None
                or not isinstance(
                    self._planning_warm_start,
                    PlanningWarmStartContext,
                )
                or not _is_sha256(self._planning_warm_start_sha256)
                or self._planning_warm_start.artifact_sha256
                != self._planning_warm_start_sha256
                or self._planning_warm_start.child_run_id
                != self.run_root.name
                or not isinstance(self._verified_parent_u74, VerifiedParentU74)
                or self._verified_parent_u74.resource_accepted is not True
                or self._verified_parent_u74.u75_attempt1_discarded is not True
            ):
                raise StandardTrainingError(
                    "production planning warm-start binding is required"
                )
            try:
                self._planning_warm_start.require_current(
                    "backend constructor"
                )
                self._coverage_cache_binding()
            except Stage6PlanningWarmStartError as exc:
                raise StandardTrainingError(
                    "production planning warm-start binding drifted"
                ) from exc
            recovery_capability = (
                self._planning_child_recovery_capability
            )
            if recovery_capability is not None:
                from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
                    PlanningChildRecoveryCapability,
                )

                immutable_sha256 = (
                    hashlib.sha256(
                        ArtifactStore.canonical_json_bytes(
                            self._immutable_bindings
                        )
                    ).hexdigest()
                    if self._immutable_bindings is not None
                    else None
                )
                if (
                    not isinstance(
                        recovery_capability,
                        PlanningChildRecoveryCapability,
                    )
                    or recovery_capability.formal_run_id
                    != self.run_root.name
                    or recovery_capability.seed
                    != self.config.training.seeds[0]
                    or recovery_capability.stage_root != self.stage_root
                    or not _is_sha256(
                        recovery_capability.parent_artifact_sha256
                    )
                    or not _is_sha256(
                        recovery_capability.continuation_artifact_sha256
                    )
                    or not _is_sha256(
                        recovery_capability.input_snapshot_sha256
                    )
                    or not _is_sha256(
                        recovery_capability.capability_sha256
                    )
                    or recovery_capability.acceptance_binding.get(
                        "current_immutable_bindings_sha256"
                    )
                    != immutable_sha256
                ):
                    raise StandardTrainingError(
                        "production planning child recovery capability "
                        "binding is required"
                    )
        else:
            from lunar_exploration_ppo.workflows.stage6_source_repair import (
                Stage6SourceRepairContext,
                Stage6SourceRepairError,
            )

            if (
                self._planning_child_recovery_capability is not None
                or not isinstance(
                    self._source_repair,
                    Stage6SourceRepairContext,
                )
                or self._source_repair.sensor_acceleration_sha256 is None
                or self._immutable_bindings
                != dict(self._source_repair.current_immutable_bindings)
            ):
                raise StandardTrainingError(
                    "production ordinal6 source-repair binding is required"
                )
            try:
                self._source_repair.require_current("backend constructor")
                self._coverage_cache_binding()
            except Stage6SourceRepairError as exc:
                raise StandardTrainingError(
                    "production source-repair binding drifted"
                ) from exc
        base_config_bytes = ArtifactStore.canonical_json_bytes(
            config.model_dump(mode="json")
        )
        self.base_config_sha256 = hashlib.sha256(
            base_config_bytes
        ).hexdigest()
        if self._planning_warm_start is not None:
            from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
                Stage6PlanningWarmStartError,
                parse_planning_effective_config_bytes,
            )

            try:
                effective = parse_planning_effective_config_bytes(
                    self._planning_warm_start.child_effective_config_bytes
                )
            except Stage6PlanningWarmStartError as exc:
                raise StandardTrainingError(
                    "planning warm-start effective config is invalid"
                ) from exc
            if (
                effective.base_config_sha256 != self.base_config_sha256
                or effective.base_config != config
            ):
                raise StandardTrainingError(
                    "planning warm-start base config drifted"
                )
            self.config_bytes = effective.effective_config_bytes
            self.config_sha256 = effective.effective_config_sha256
            if (
                self.config_sha256
                != self._planning_warm_start.child_effective_config_sha256
            ):
                raise StandardTrainingError(
                    "planning warm-start effective config drifted"
                )
        else:
            self.config_bytes = base_config_bytes
            self.config_sha256 = self.base_config_sha256
        self.safety_contract = SafetyContract.from_stage6_config(config)
        self.catalog = self._call("build_standard_catalog")(
            verify_hashes=True
        )
        receipt_type = self._components["CheckpointReceiptIndex"]
        journal_type = self._components["Stage6StateJournal"]
        self.receipt_index = receipt_type(  # type: ignore[operator]
            self.stage_root / "checkpoints/index.jsonl"
        )
        self.journal = journal_type(  # type: ignore[operator]
            self.stage_root / "job-state.jsonl"
        )
        self.phase_journal = journal_type(  # type: ignore[operator]
            self.stage_root / "phase-state.jsonl"
        )
        if self._planning_warm_start is not None:
            from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
                build_planning_child_transactions,
            )

            self.transactions = build_planning_child_transactions(config)
        else:
            self.transactions = build_standard_training_transactions(config)
        self._checkpoint_writes = len(self.receipt_index.verify())

        self._require_execution_capability_current(
            "StandardProductionBackend constructor exit"
        )

    def _require_execution_capability_current(
        self,
        label: str,
        *,
        rehash_inputs: bool = False,
    ) -> None:
        _require_standard_execution_capability(
            self._execution_capability,
            label=label,
            rehash_inputs=rehash_inputs,
            config=self.config,
            run_root=self.run_root,
            repo_root=self.repo_root,
            stage5_authority=self.stage5_authority,
        )

    def _require_source_repair_current(self, label: str) -> None:
        warm_start = getattr(self, "_planning_warm_start", None)
        if warm_start is not None:
            from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
                Stage6PlanningWarmStartError,
            )

            try:
                warm_start.require_current(label)
            except Stage6PlanningWarmStartError as exc:
                raise StandardTrainingError(
                    "production planning warm-start drifted"
                ) from exc
            return
        context = getattr(self, "_source_repair", None)
        if context is None:
            return
        from lunar_exploration_ppo.workflows.stage6_source_repair import (
            Stage6SourceRepairError,
        )

        try:
            context.require_current(label)
        except Stage6SourceRepairError as exc:
            raise StandardTrainingError("production source-repair drifted") from exc

    def _execution_operation(self, label: str):
        return _standard_execution_operation(
            self._execution_capability,
            label=f"StandardProductionBackend.{label}",
            config=self.config,
            run_root=self.run_root,
            repo_root=self.repo_root,
            stage5_authority=self.stage5_authority,
        )

    def _call(self, name: str) -> Callable[..., object]:
        value = self._components.get(name)
        if not callable(value):
            raise StandardTrainingError(f"production component is not callable: {name}")
        return value

    def _journal_binding_split(
        self,
    ) -> tuple[dict[str, object] | None, str | None]:
        if (
            getattr(
                self,
                "_planning_child_recovery_capability",
                None,
            )
            is not None
        ):
            return None, None
        context = getattr(self, "_source_repair", None)
        if getattr(context, "sensor_acceleration_sha256", None) is None:
            return None, None
        from lunar_exploration_ppo.workflows.stage6_source_repair import (
            SENSOR_ACCELERATION_NEXT_TRANSACTION_KEY,
        )
        return (
            dict(context.origin_immutable_bindings),
            SENSOR_ACCELERATION_NEXT_TRANSACTION_KEY,
        )

    def _journal_binding_epochs(
        self,
    ) -> tuple[tuple[str, Mapping[str, object]], ...] | None:
        recovery_capability = getattr(
            self,
            "_planning_child_recovery_capability",
            None,
        )
        if recovery_capability is not None:
            from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
                _plain_json,
            )

            raw_epochs = recovery_capability.acceptance_binding.get(
                "journal_binding_epochs"
            )
            if not isinstance(raw_epochs, tuple):
                raise StandardTrainingError(
                    "planning child recovery journal epochs drifted"
                )
            epochs: list[tuple[str, Mapping[str, object]]] = []
            for value in raw_epochs:
                if (
                    not isinstance(value, tuple)
                    or len(value) != 2
                    or not isinstance(value[0], str)
                    or not isinstance(value[1], Mapping)
                ):
                    raise StandardTrainingError(
                        "planning child recovery journal epochs drifted"
                    )
                normalized_bindings = _plain_json(value[1])
                if not isinstance(normalized_bindings, dict):
                    raise StandardTrainingError(
                        "planning child recovery journal epochs drifted"
                    )
                epochs.append((value[0], normalized_bindings))
            if len(epochs) != 3:
                raise StandardTrainingError(
                    "planning child recovery journal epochs drifted"
                )
            return tuple(epochs)
        return None

    def _require_planning_child_resume_boundary(
        self,
        remaining: Sequence[StandardTrainingTransaction],
    ) -> None:
        recovery_capability = getattr(
            self,
            "_planning_child_recovery_capability",
            None,
        )
        if (
            recovery_capability is None
            or getattr(
                self,
                "_planning_child_resume_boundary_consumed",
                False,
            )
        ):
            return
        if not remaining:
            return
        cursor = recovery_capability.resume_cursor
        first = remaining[0]
        if (
            not isinstance(first, StandardTrainingTransaction)
            or first.update != cursor.next_update
            or first.key != cursor.next_transaction_key
            or self._next_resource_attempt(first.key)
            != cursor.next_attempt
        ):
            raise StandardTrainingError(
                "planning child resume boundary drifted"
            )
        segment = getattr(self, "_resource_segment_start", None)
        if (
            not isinstance(segment, Mapping)
            or segment.get("segment_index")
            != cursor.next_resource_segment_index
        ):
            raise StandardTrainingError(
                "planning child resume resource segment drifted"
            )
        self._planning_child_resume_boundary_consumed = True

    def _coverage_cache_binding(self) -> dict[str, str]:
        if getattr(self, "_planning_warm_start", None) is not None:
            from lunar_exploration_ppo.workflows.stage6_source_repair import (
                _FIXED_COVERAGE_CACHE_MANIFEST_BINDING,
            )

            path = _FIXED_COVERAGE_CACHE_MANIFEST_BINDING.get("path")
            sha256 = _FIXED_COVERAGE_CACHE_MANIFEST_BINDING.get("sha256")
            if not isinstance(path, str) or not _is_sha256(sha256):
                raise StandardTrainingError(
                    "production planning warm-start cache binding drifted"
                )
            return {
                "coverage_cache_manifest_path": path,
                "coverage_cache_manifest_sha256": str(sha256),
            }
        context = getattr(self, "_source_repair", None)
        if (
            context is None
            or getattr(context, "sensor_acceleration_sha256", None) is None
            or not isinstance(
                getattr(context, "coverage_cache_manifest_path", None),
                str,
            )
            or not _is_sha256(
                getattr(context, "coverage_cache_manifest_sha256", None)
            )
            or getattr(context, "coverage_cache_runtime_mode", None)
            != "persistent_exact_manifest_read_only/v1"
        ):
            raise StandardTrainingError(
                "production coverage cache ordinal6 binding is missing"
            )
        path = context.coverage_cache_manifest_path
        sha256 = context.coverage_cache_manifest_sha256
        immutable = dict(context.current_immutable_bindings)
        if (
            immutable.get("coverage_cache_manifest_path") != path
            or immutable.get("coverage_cache_manifest_sha256") != sha256
            or immutable.get("coverage_cache_runtime_mode")
            != "persistent_exact_manifest_read_only/v1"
        ):
            raise StandardTrainingError(
                "production coverage cache ordinal6 binding drifted"
            )
        return {
            "coverage_cache_manifest_path": path,
            "coverage_cache_manifest_sha256": sha256,
        }

    def _production_training_env_specs(
        self,
        *,
        split: str,
        sampler_seeds: tuple[int, ...],
        safety_contract: object,
        initial_sampler_states: Sequence[Mapping[str, object]] | None = None,
    ) -> tuple[object, ...]:
        spec_kwargs: dict[str, object] = {
            "split": split,
            "sampler_seeds": sampler_seeds,
            "safety_contract": safety_contract,
            "config_sha256": self.config_sha256,
        }
        if initial_sampler_states is not None:
            spec_kwargs["initial_sampler_states"] = initial_sampler_states
        legacy_specs = self._call("standard_env_specs")(
            self.catalog,
            **spec_kwargs,
        )
        if not isinstance(legacy_specs, tuple) or len(legacy_specs) != 8:
            raise StandardTrainingError(
                "production Standard env spawn spec contract drifted"
            )
        from lunar_exploration_ppo.ppo.collector import SpawnEnvSpec

        cache_binding = self._coverage_cache_binding()
        result: list[SpawnEnvSpec] = []
        for spec in legacy_specs:
            kwargs = getattr(spec, "kwargs", None)
            if not isinstance(kwargs, Mapping):
                raise StandardTrainingError(
                    "production Standard env spawn spec contract drifted"
                )
            result.append(
                SpawnEnvSpec(
                    factory=_build_cached_standard_training_env,
                    kwargs={**dict(kwargs), **cache_binding},
                )
            )
        return tuple(result)

    @_guarded_backend_mutation
    def _resource_audit_rows(self) -> tuple[dict[str, object], ...]:
        path = self.stage_root / "resource_audit.jsonl"
        try:
            payload = DurableJsonl(path).recover_and_snapshot()
        except (DurableJsonlError, OSError) as exc:
            raise StandardTrainingError(
                "resource audit committed snapshot failed"
            ) from exc
        if not payload:
            return ()
        rows: list[dict[str, object]] = []
        try:
            for line in payload.splitlines(keepends=True):
                row = json.loads(line.decode("utf-8"))
                canonical = (
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                if not isinstance(row, dict) or line != canonical:
                    raise ValueError("resource row is not canonical")
                rows.append(row)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise StandardTrainingError("resource audit JSONL drifted") from exc
        return tuple(rows)

    def _next_resource_attempt(self, transaction_key: str) -> int:
        attempts = [
            row.get("attempt")
            for row in self._resource_audit_rows()
            if row.get("transaction_key") == transaction_key
        ]
        if any(type(value) is not int or value <= 0 for value in attempts):
            raise StandardTrainingError("resource audit attempt drifted")
        return max(attempts, default=0) + 1

    def _capture_resource_snapshot(self) -> object:
        from lunar_exploration_ppo.utils.resources import ResourceSnapshot

        peak_vram = (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else 0
        )
        snapshot = self._call("capture_resource_snapshot")(
            peak_vram_bytes=peak_vram,
            process_tree_monitor=self._process_tree_monitor,
        )
        if not isinstance(snapshot, ResourceSnapshot):
            try:
                snapshot = ResourceSnapshot(
                    d_free_bytes=int(snapshot.d_free_bytes),
                    rss_bytes=int(snapshot.rss_bytes),
                    peak_vram_bytes=int(snapshot.peak_vram_bytes),
                    rss_source=str(snapshot.rss_source),
                    rss_root_pid=int(snapshot.rss_root_pid),
                    rss_sample_count=int(snapshot.rss_sample_count),
                    rss_latest_process_count=int(snapshot.rss_latest_process_count),
                    rss_peak_process_count=int(snapshot.rss_peak_process_count),
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise StandardTrainingError(
                    "resource sampler returned an invalid snapshot"
                ) from exc
        return snapshot

    @staticmethod
    def _resource_gate_record(snapshot: object, decision: object) -> dict[str, object]:
        return _resource_gate_record(snapshot, decision)

    def _capture_resource_gate(self) -> dict[str, object]:
        snapshot = self._capture_resource_snapshot()
        decision = self._call("evaluate_resource_gates")(
            snapshot,
            preflight=False,
        )
        return self._resource_gate_record(snapshot, decision)

    def _new_runtime_resource_latch(
        self,
        *,
        transaction_key: str,
        attempt: int,
    ) -> object:
        from lunar_exploration_ppo.utils.resources import ResourceHardStopLatch

        return ResourceHardStopLatch(
            attempt_id=f"{transaction_key}:attempt-{attempt}",
            snapshot_provider=self._capture_resource_snapshot,
        )

    def _poll_runtime_resource_latch(
        self,
        latch: object,
        boundary: str,
    ) -> dict[str, object]:
        from lunar_exploration_ppo.utils.resources import ResourceHardStopError

        try:
            snapshot, decision = latch.poll(boundary)
        except ResourceHardStopError as exc:
            raise StandardTrainingError(
                f"runtime resource hard stop at {boundary}"
            ) from exc
        return self._resource_gate_record(snapshot, decision)

    @_guarded_backend_mutation
    def _append_resource_phase(
        self,
        *,
        identity: Mapping[str, object],
        attempt: int,
        phase: Literal["pre", "post"],
        resource: Mapping[str, object],
    ) -> None:
        from lunar_exploration_ppo.utils.resource_lifecycle import (
            ResourceLifecycleError,
            bind_resource_row_to_segment,
        )

        segment = getattr(self, "_resource_segment_start", None)
        if not isinstance(segment, Mapping):
            raise StandardTrainingError("active resource segment is missing")
        row = {
            "schema_version": "stage6_resource_attempt/v1",
            **dict(identity),
            "attempt": attempt,
            "phase": phase,
            "accepted": False,
            "resource": dict(resource),
        }
        try:
            DurableJsonl(self.stage_root / "resource_audit.jsonl").append(
                bind_resource_row_to_segment(row, segment)
            )
        except (DurableJsonlError, OSError, ResourceLifecycleError) as exc:
            raise StandardTrainingError(
                "resource audit durable append failed"
            ) from exc

    @_guarded_backend_mutation
    def _append_resource_acceptance(
        self,
        *,
        identity: Mapping[str, object],
        attempt: int,
        pre: Mapping[str, object],
        post: Mapping[str, object],
        checkpoint: StandardTransactionCheckpoint | None = None,
    ) -> None:
        from lunar_exploration_ppo.utils.resource_lifecycle import (
            ResourceLifecycleError,
            bind_resource_row_to_segment,
        )

        kind = identity.get("kind")
        if kind == "update":
            checkpoint_identity = _checkpoint_acceptance_identity(checkpoint)  # type: ignore[arg-type]
            if any(
                checkpoint_identity.get(name) != identity.get(name)
                for name in ("transaction_key", "seed", "update")
            ):
                raise StandardTrainingError(
                    "resource acceptance checkpoint transaction drifted"
                )
            schema_version = "stage6_resource_acceptance/v2"
        elif kind == "final_evaluation" and checkpoint is None:
            checkpoint_identity = None
            schema_version = "stage6_resource_acceptance/v1"
        else:
            raise StandardTrainingError("resource acceptance kind drifted")
        row = {
            "schema_version": schema_version,
            **dict(identity),
            "attempt": attempt,
            "phase": "accepted",
            "accepted": True,
            "pre": dict(pre),
            "post": dict(post),
        }
        if checkpoint_identity is not None:
            row["checkpoint"] = checkpoint_identity
        keyed = tuple(
            existing
            for existing in self._resource_audit_rows()
            if existing.get("transaction_key") == identity.get("transaction_key")
            and existing.get("attempt") == attempt
        )
        pre_rows = tuple(
            existing for existing in keyed if existing.get("phase") == "pre"
        )
        post_rows = tuple(
            existing for existing in keyed if existing.get("phase") == "post"
        )
        if (
            len(pre_rows) != 1
            or len(post_rows) != 1
            or pre_rows[0].get("resource") != dict(pre)
            or post_rows[0].get("resource") != dict(post)
            or pre_rows[0].get("segment_id") != post_rows[0].get("segment_id")
            or pre_rows[0].get("segment_index")
            != post_rows[0].get("segment_index")
        ):
            raise StandardTrainingError(
                "resource acceptance pre/post segment drifted"
            )
        segment = next(
            (
                existing
                for existing in self._resource_audit_rows()
                if existing.get("phase") == "segment_start"
                and existing.get("segment_id") == pre_rows[0].get("segment_id")
                and existing.get("segment_index")
                == pre_rows[0].get("segment_index")
            ),
            None,
        )
        if not isinstance(segment, Mapping):
            raise StandardTrainingError(
                "resource acceptance segment start is missing"
            )
        try:
            DurableJsonl(self.stage_root / "resource_audit.jsonl").append(
                bind_resource_row_to_segment(row, segment)
            )
        except (DurableJsonlError, OSError, ResourceLifecycleError) as exc:
            raise StandardTrainingError(
                "resource audit durable append failed"
            ) from exc

    @_guarded_backend_mutation
    def capture_terminal_resource_sample(self) -> dict[str, object]:
        """Take the final explicit sample while the lifecycle monitor is running."""

        monitor = self._process_tree_monitor
        if monitor is None or monitor.running is not True:
            raise StandardTrainingError(
                "terminal resource sample requires a running lifecycle monitor"
            )
        before = monitor.sample_count
        try:
            monitor.sample_now()
        except Exception as exc:
            raise StandardTrainingError("terminal resource sample failed") from exc
        after = monitor.sample_count
        if monitor.running is not True or after <= before:
            raise StandardTrainingError("terminal resource sample count drifted")
        return {
            "schema_version": "stage6_terminal_resource_sample/v1",
            "sample_count_before": before,
            "sample_count_after": after,
        }

    @_guarded_backend_mutation
    def record_terminal_resource_evidence(self) -> dict[str, object]:
        """Persist the receipt-bound v4 terminal after monitor stop."""

        existing = getattr(self, "_terminal_resource_evidence", None)
        if isinstance(existing, Mapping):
            return dict(existing)
        monitor = self._process_tree_monitor
        if monitor is None or monitor.running is not False:
            raise StandardTrainingError(
                "terminal resource evidence requires a stopped lifecycle monitor"
            )
        receipt_identity = getattr(
            self,
            "_preterminal_acceptance_identity",
            None,
        )
        if (
            not isinstance(receipt_identity, Mapping)
            or set(receipt_identity) != {"sha256", "size_bytes"}
            or not _is_sha256(receipt_identity.get("sha256"))
            or type(receipt_identity.get("size_bytes")) is not int
            or int(receipt_identity["size_bytes"]) < 0
        ):
            raise StandardTrainingError(
                "terminal resource preterminal receipt is missing"
            )
        terminal_resource = self._capture_resource_gate()
        if (
            terminal_resource.get("passed") is not True
            or terminal_resource.get("rss_source")
            != "process_tree_lifecycle_peak_current_sum/v1"
        ):
            raise StandardTrainingError("terminal resource lifecycle hard gate failed")
        from lunar_exploration_ppo.workflows.stage6_terminal_recovery import (
            TerminalRecoveryError,
            append_stage6_recovery_resource_terminal,
        )

        try:
            terminal = append_stage6_recovery_resource_terminal(
                stage_root=self.stage_root,
                terminal_resource=terminal_resource,
                execution_capability=self._execution_capability,
            )
        except TerminalRecoveryError as exc:
            raise StandardTrainingError(
                "bound terminal resource durable append failed"
            ) from exc
        if (
            not isinstance(terminal, Mapping)
            or terminal.get("schema_version")
            != "stage6_terminal_resource_evidence/v4"
            or terminal.get("preterminal_acceptance") != dict(receipt_identity)
            or terminal.get("resource") != dict(terminal_resource)
        ):
            raise StandardTrainingError("bound terminal resource replay drifted")
        self._terminal_resource_evidence = dict(terminal)
        return dict(terminal)

    def _accepted_resource_attempt(
        self,
        identity: Mapping[str, object],
    ) -> dict[str, object] | None:
        keyed = tuple(
            (index, row)
            for index, row in enumerate(self._resource_audit_rows())
            if row.get("transaction_key") == identity.get("transaction_key")
        )
        accepted = tuple(
            (index, row)
            for index, row in keyed
            if row.get("phase") == "accepted" and row.get("accepted") is True
        )
        if not accepted:
            return None
        if len(accepted) != 1:
            raise StandardTrainingError("resource acceptance drifted")
        accepted_index, accepted_row = accepted[0]
        attempt = accepted_row.get("attempt")
        if type(attempt) is not int or attempt <= 0:
            raise StandardTrainingError("resource acceptance attempt drifted")
        pre_rows = tuple(
            (index, row)
            for index, row in keyed
            if row.get("attempt") == attempt and row.get("phase") == "pre"
        )
        post_rows = tuple(
            (index, row)
            for index, row in keyed
            if row.get("attempt") == attempt and row.get("phase") == "post"
        )
        common_identity = {
            name: value for name, value in identity.items()
        }
        attempt_fields = {
            "schema_version",
            *common_identity,
            "attempt",
            "phase",
            "accepted",
            "resource",
            "segment_id",
            "segment_index",
        }
        acceptance_fields = {
            "schema_version",
            *common_identity,
            "attempt",
            "phase",
            "accepted",
            "pre",
            "post",
            "segment_id",
            "segment_index",
        }
        is_update = identity.get("kind") == "update"
        if is_update:
            acceptance_fields.add("checkpoint")
        if (
            len(pre_rows) != 1
            or len(post_rows) != 1
            or not pre_rows[0][0] < post_rows[0][0] < accepted_index
            or set(pre_rows[0][1]) != attempt_fields
            or set(post_rows[0][1]) != attempt_fields
            or set(accepted_row) != acceptance_fields
            or pre_rows[0][1].get("schema_version")
            != "stage6_resource_attempt/v1"
            or post_rows[0][1].get("schema_version")
            != "stage6_resource_attempt/v1"
            or accepted_row.get("schema_version")
            != (
                "stage6_resource_acceptance/v2"
                if is_update
                else "stage6_resource_acceptance/v1"
            )
            or any(
                accepted_row.get(name) != value
                for name, value in common_identity.items()
            )
            or accepted_row.get("segment_id") != pre_rows[0][1].get("segment_id")
            or accepted_row.get("segment_index")
            != pre_rows[0][1].get("segment_index")
            or post_rows[0][1].get("segment_id")
            != pre_rows[0][1].get("segment_id")
            or post_rows[0][1].get("segment_index")
            != pre_rows[0][1].get("segment_index")
            or accepted_row.get("pre") != pre_rows[0][1].get("resource")
            or accepted_row.get("post") != post_rows[0][1].get("resource")
            or not isinstance(accepted_row.get("pre"), Mapping)
            or not isinstance(accepted_row.get("post"), Mapping)
            or accepted_row["pre"].get("passed") is not True
            or accepted_row["post"].get("passed") is not True
        ):
            raise StandardTrainingError("resource acceptance binding drifted")
        if is_update:
            checkpoint_value = accepted_row.get("checkpoint")
            try:
                checkpoint = StandardTransactionCheckpoint(**checkpoint_value)  # type: ignore[arg-type]
            except (TypeError, StandardTrainingError) as exc:
                raise StandardTrainingError(
                    "resource acceptance checkpoint binding drifted"
                ) from exc
            if any(
                getattr(checkpoint, name) != identity.get(name)
                for name in ("transaction_key", "seed", "update")
            ):
                raise StandardTrainingError(
                    "resource acceptance checkpoint binding drifted"
                )
        return dict(accepted_row)

    @_guarded_backend_mutation
    def _apply_checkpoint_retention(
        self,
        *,
        checkpoint_root: Path,
        update: int,
        best_record: Mapping[str, object],
    ) -> object:
        self._require_execution_capability_current(
            "checkpoint retention mutation",
            rehash_inputs=True,
        )
        best_update = int(best_record.get("update", update))
        if getattr(self, "_planning_warm_start", None) is not None:
            from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
                CHILD_RETENTION_PERIODIC_UPDATES,
            )

            periodic = tuple(
                value
                for value in CHILD_RETENTION_PERIODIC_UPDATES
                if value <= update
            )
            periodic_keep_count = self.config.training.periodic_keep_count
        else:
            periodic = tuple(
                value
                for value in self.config.training.periodic_updates
                if value <= update
            )
            periodic_keep_count = self.config.training.periodic_keep_count
        retention = self._call("CheckpointRetentionManager")(
            checkpoint_root,
            schema_version=self.config.checkpoint.schema_version,
        )
        retention_kwargs: dict[str, object] = {
            "latest_update": update,
            "periodic_updates": periodic,
            "best_update": best_update,
            "periodic_keep_count": periodic_keep_count,
        }
        source_repair = getattr(self, "_source_repair", None)
        recovery_capability = getattr(
            self,
            "_planning_child_recovery_capability",
            None,
        )
        if recovery_capability is not None:
            retention_kwargs["protected_updates"] = (
                recovery_capability.protected_checkpoint_updates
            )
        if recovery_capability is None and source_repair is not None:
            retention_kwargs["protected_updates"] = (
                source_repair.protected_checkpoint_updates
            )
        return retention.apply(
            **retention_kwargs,
        )

    @_guarded_backend_mutation
    def _publish_validation_trace(
        self,
        *,
        pending_path: Path,
        canonical_path: Path,
        episode_count: int,
        expected_binding: Mapping[str, object],
    ) -> None:
        expected = _validated_validation_trace_binding(
            expected_binding,
            episode_count=episode_count,
        )
        try:
            canonical_parent = require_plain_path(
                canonical_path.parent,
                base=self.stage_root,
                allow_missing=True,
                leaf_kind="directory",
                label="validation trace canonical parent",
            )
            canonical_stage_root = require_plain_path(
                self.stage_root,
                base=self.run_root,
                leaf_kind="directory",
                label="validation trace stage root",
            )
            canonical = require_plain_path(
                canonical_path,
                base=self.stage_root,
                allow_missing=True,
                leaf_kind="file",
                label="validation trace canonical path",
            )
            pending = require_plain_path(
                pending_path,
                base=self.run_root,
                leaf_kind="file",
                label="validation trace pending path",
            )
            pending_read = secure_read_bytes(
                pending,
                base=self.run_root,
                label="validation trace pending",
            )
            _validated_validation_trace_binding(
                expected,
                episode_count=episode_count,
                payload=pending_read.payload,
            )
            pending_delete_identity = durable_file_identity(pending)
            if not _durable_identity_matches_secure_read(
                pending_read,
                pending_delete_identity,
            ):
                raise PathSecurityError("validation trace pending identity drifted")
            if canonical.parent != canonical_parent:
                raise PathSecurityError("validation trace canonical parent drifted")
            ArtifactStore(canonical_stage_root).write_bytes_exclusive(
                canonical.relative_to(canonical_stage_root),
                pending_read.payload,
            )
            canonical_read = secure_read_bytes(
                canonical,
                base=self.stage_root,
                label="validation trace canonical",
            )
            if canonical_read.payload != pending_read.payload:
                raise StandardTrainingError("validation trace publication drifted")
            _validated_validation_trace_binding(
                expected,
                episode_count=episode_count,
                payload=canonical_read.payload,
            )
            durable_unlink(pending, expected_identity=pending_delete_identity)
        except StandardTrainingError:
            raise
        except (FileExistsError, OSError, PathSecurityError, ValueError) as exc:
            raise StandardTrainingError(
                "validation trace publication drifted"
            ) from exc

    @_guarded_backend_mutation
    def _recover_validation_trace(
        self,
        transaction: StandardTrainingTransaction,
        *,
        loaded: object,
    ) -> None:
        if transaction.validation_episodes == 0:
            return
        expected_binding = _validation_trace_binding_from_metrics(
            loaded,
            episode_count=transaction.validation_episodes,
        )
        canonical = self.stage_root / (
            "episode-traces/"
            f"validation-seed-{transaction.seed}-update-{transaction.update:03d}.jsonl"
        )
        canonical_present = False
        try:
            canonical_path = require_plain_path(
                canonical,
                base=self.stage_root,
                allow_missing=True,
                leaf_kind="file",
                label="validation trace canonical path",
            )
            canonical_present = os.path.lexists(canonical_path)
            if canonical_present:
                canonical_read = secure_read_bytes(
                    canonical_path,
                    base=self.stage_root,
                    label="validation trace canonical",
                )
                _validated_validation_trace_binding(
                    expected_binding,
                    episode_count=transaction.validation_episodes,
                    payload=canonical_read.payload,
                )
        except StandardTrainingError:
            raise
        except (OSError, PathSecurityError) as exc:
            raise StandardTrainingError(
                "validation trace recovery binding drifted"
            ) from exc
        if not canonical_present and os.path.lexists(canonical):
            raise StandardTrainingError("validation trace recovery binding drifted")
        pre_attempts = [
            int(row["attempt"])
            for row in self._resource_audit_rows()
            if row.get("transaction_key") == transaction.key
            and row.get("phase") == "pre"
            and type(row.get("attempt")) is int
        ]
        if not pre_attempts:
            raise StandardTrainingError("validation checkpoint lacks attempt evidence")
        attempt = max(pre_attempts)
        pending = self.run_root / (
            "stage6-attempts/"
            f"validation-seed-{transaction.seed}-update-{transaction.update:03d}-"
            f"attempt-{attempt:03d}.pending.jsonl"
        )
        if canonical_present:
            try:
                pending_path = require_plain_path(
                    pending,
                    base=self.run_root,
                    allow_missing=True,
                    leaf_kind="file",
                    label="validation trace recovery pending path",
                )
                pending_present = os.path.lexists(pending_path)
                if not pending_present:
                    if os.path.lexists(pending_path):
                        raise PathSecurityError(
                            "validation trace recovery pending appeared"
                        )
                    return
                pending_read = secure_read_bytes(
                    pending_path,
                    base=self.run_root,
                    label="validation trace recovery pending",
                )
                _validated_validation_trace_binding(
                    expected_binding,
                    episode_count=transaction.validation_episodes,
                    payload=pending_read.payload,
                )
                pending_delete_identity = durable_file_identity(pending_path)
                if not _durable_identity_matches_secure_read(
                    pending_read,
                    pending_delete_identity,
                ):
                    raise PathSecurityError(
                        "validation trace recovery pending identity drifted"
                    )
                durable_unlink(
                    pending_path,
                    expected_identity=pending_delete_identity,
                )
            except StandardTrainingError:
                raise
            except (OSError, PathSecurityError) as exc:
                raise StandardTrainingError(
                    "validation trace recovery binding drifted"
                ) from exc
            return
        self._publish_validation_trace(
            pending_path=pending,
            canonical_path=canonical,
            episode_count=transaction.validation_episodes,
            expected_binding=expected_binding,
        )

    @_guarded_backend_mutation
    def _recover_update_resource_acceptance(
        self,
        transaction: StandardTrainingTransaction,
        checkpoint: StandardTransactionCheckpoint,
    ) -> dict[str, object]:
        if (
            not isinstance(checkpoint, StandardTransactionCheckpoint)
            or checkpoint.transaction_key != transaction.key
            or checkpoint.seed != transaction.seed
            or checkpoint.update != transaction.update
        ):
            raise StandardTrainingError(
                "resume resource checkpoint transaction drifted"
            )
        keyed = tuple(
            row
            for row in self._resource_audit_rows()
            if row.get("transaction_key") == transaction.key
        )
        accepted = tuple(
            row
            for row in keyed
            if row.get("phase") == "accepted" and row.get("accepted") is True
        )
        if len(accepted) > 1 or not keyed:
            raise StandardTrainingError("resume resource acceptance drifted")
        attempts = [
            int(row["attempt"])
            for row in keyed
            if type(row.get("attempt")) is int
        ]
        if not attempts:
            raise StandardTrainingError("resume resource attempt is missing")
        attempt = (
            int(accepted[0]["attempt"])
            if accepted and type(accepted[0].get("attempt")) is int
            else max(attempts)
        )
        pre_rows = tuple(
            row
            for row in keyed
            if row.get("attempt") == attempt and row.get("phase") == "pre"
        )
        post_rows = tuple(
            row
            for row in keyed
            if row.get("attempt") == attempt and row.get("phase") == "post"
        )
        if len(pre_rows) != 1 or len(post_rows) > 1:
            raise StandardTrainingError("resume resource pre/post drifted")
        pre = pre_rows[0].get("resource")
        if not isinstance(pre, Mapping) or pre.get("passed") is not True:
            raise StandardTrainingError("resume resource pre gate did not pass")
        identity = {
            "kind": "update",
            "transaction_key": transaction.key,
            "seed": transaction.seed,
            "update": transaction.update,
        }
        if not post_rows:
            raise StandardTrainingError(
                "resume checkpoint lacks original post-resource evidence"
            )
        post = post_rows[0].get("resource")
        if not isinstance(post, Mapping):
            raise StandardTrainingError("resume resource post gate drifted")
        if post.get("passed") is not True:
            raise StandardTrainingError("resume post-resource hard stop")
        if (
            pre.get("rss_root_pid") != post.get("rss_root_pid")
            or pre.get("rss_source")
            != "process_tree_lifecycle_peak_current_sum/v1"
            or post.get("rss_source")
            != "process_tree_lifecycle_peak_current_sum/v1"
            or type(pre.get("rss_sample_count")) is not int
            or type(post.get("rss_sample_count")) is not int
            or int(post["rss_sample_count"]) < int(pre["rss_sample_count"])
            or type(pre.get("rss_bytes")) is not int
            or type(post.get("rss_bytes")) is not int
            or int(post["rss_bytes"]) < int(pre["rss_bytes"])
        ):
            raise StandardTrainingError(
                "resume original post-resource process binding drifted"
            )
        if accepted:
            acceptance = accepted[0]
            if (
                acceptance.get("pre") != pre
                or acceptance.get("post") != post
                or acceptance.get("schema_version")
                != "stage6_resource_acceptance/v2"
                or acceptance.get("checkpoint")
                != _checkpoint_acceptance_identity(checkpoint)
            ):
                raise StandardTrainingError(
                    "resume resource acceptance pre/post drifted"
                )
            return dict(acceptance)
        self._append_resource_acceptance(
            identity=identity,
            attempt=attempt,
            pre=pre,
            post=post,
            checkpoint=checkpoint,
        )
        accepted_row = self._accepted_resource_attempt(identity)
        if accepted_row is None:
            raise StandardTrainingError("resource acceptance did not become durable")
        return accepted_row

    @_guarded_backend_mutation
    def prepare_seed(self, seed, transactions):
        if (
            getattr(
                self,
                "_planning_child_recovery_capability",
                None,
            )
            is None
        ):
            self._require_source_repair_current("prepare_seed")
        seed_transactions = tuple(transactions)
        expected = tuple(item for item in self.transactions if item.seed == seed)
        if seed_transactions != expected:
            raise StandardTrainingError("production seed transaction schedule drifted")
        checkpoint_root = (
            self.stage_root / f"checkpoints/seed-{seed}"
        ).resolve()
        manager = self._call("CheckpointManager")(
            checkpoint_root,
            schema_version=self.config.checkpoint.schema_version,
        )
        receipt_rows = self.receipt_index.verify()
        (
            historical_immutable_bindings,
            current_binding_first_transaction_key,
        ) = self._journal_binding_split()
        immutable_binding_epochs = self._journal_binding_epochs()

        def accepted_checkpoint(
            transaction: StandardTrainingTransaction,
        ) -> StandardTransactionCheckpoint | None:
            acceptance = self._accepted_resource_attempt(
                {
                    "kind": "update",
                    "transaction_key": transaction.key,
                    "seed": transaction.seed,
                    "update": transaction.update,
                }
            )
            if acceptance is None:
                return None
            try:
                checkpoint = StandardTransactionCheckpoint(
                    **acceptance["checkpoint"]  # type: ignore[arg-type]
                )
            except (KeyError, TypeError, StandardTrainingError) as exc:
                raise StandardTrainingError(
                    "accepted checkpoint recovery identity drifted"
                ) from exc
            if (
                checkpoint.transaction_key != transaction.key
                or checkpoint.seed != transaction.seed
                or checkpoint.update != transaction.update
            ):
                raise StandardTrainingError(
                    "accepted checkpoint recovery transaction drifted"
                )
            return checkpoint

        recovered_keys: set[str] = set()

        def recover_accepted(
            transaction: StandardTrainingTransaction,
            checkpoint: StandardTransactionCheckpoint,
        ) -> None:
            nonlocal receipt_rows
            if self._immutable_bindings is None:
                raise StandardTrainingError(
                    "accepted checkpoint recovery requires immutable bindings"
                )
            recover = getattr(manager, "recover_accepted_checkpoint", None)
            if not callable(recover):
                raise StandardTrainingError(
                    "checkpoint manager lacks accepted recovery"
                )
            receipt = recover(
                update_step=checkpoint.update,
                checkpoint_sha256=checkpoint.checkpoint_sha256,
                complete_marker_sha256=checkpoint.complete_marker_sha256,
                policy_state_sha256=checkpoint.policy_state_sha256,
            )
            if (
                getattr(receipt, "update_step", None) != checkpoint.update
                or getattr(receipt, "checkpoint_sha256", None)
                != checkpoint.checkpoint_sha256
                or getattr(receipt, "policy_state_sha256", None)
                != checkpoint.policy_state_sha256
            ):
                raise StandardTrainingError(
                    "accepted checkpoint recovery receipt drifted"
                )
            self.receipt_index.append_once(checkpoint)
            receipt_rows = self.receipt_index.verify()
            matching_receipts = tuple(
                row
                for row in receipt_rows
                if row.get("transaction_key") == checkpoint.transaction_key
            )
            if len(matching_receipts) != 1 or any(
                matching_receipts[0].get(name) != value
                for name, value in _checkpoint_acceptance_identity(checkpoint).items()
            ):
                raise StandardTrainingError(
                    "accepted checkpoint receipt persistence drifted"
                )
            bindings = {
                **self._immutable_bindings,
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
            }
            remaining, _ = reconcile_checkpointed_resume(
                transactions=self.transactions,
                journal=self.journal,
                checkpoint=checkpoint,
                bindings=bindings,
                receipt_rows=receipt_rows,
                historical_immutable_bindings=historical_immutable_bindings,
                current_binding_first_transaction_key=(
                    current_binding_first_transaction_key
                ),
                immutable_binding_epochs=immutable_binding_epochs,
            )
            if any(item.key == checkpoint.transaction_key for item in remaining):
                raise StandardTrainingError(
                    "accepted checkpoint journal reconciliation drifted"
                )
            recovered_keys.add(transaction.key)

        next_transaction = (
            self.transactions[len(receipt_rows)]
            if len(receipt_rows) < len(self.transactions)
            else None
        )
        if next_transaction is not None and next_transaction.seed == seed:
            next_accepted = accepted_checkpoint(next_transaction)
            if next_accepted is not None:
                recover_accepted(next_transaction, next_accepted)

        complete_directories = tuple(
            path
            for path in checkpoint_root.iterdir()
            if path.is_dir()
            and path.name.startswith("update-")
            and (path / "complete.json").is_file()
        ) if checkpoint_root.is_dir() else ()
        if complete_directories:
            highest_update = max(
                int(path.name.removeprefix("update-"))
                for path in complete_directories
            )
            highest_transaction = next(
                (
                    item
                    for item in seed_transactions
                    if item.update == highest_update
                ),
                None,
            )
            if highest_transaction is None:
                raise StandardTrainingError(
                    "resume checkpoint transaction identity drifted"
                )
            highest_accepted = accepted_checkpoint(highest_transaction)
            if highest_accepted is None:
                raise StandardTrainingError(
                    "complete checkpoint lacks durable acceptance"
                )
            if highest_transaction.key not in recovered_keys:
                recover_accepted(highest_transaction, highest_accepted)
        if not complete_directories:
            _seed_standard_training_rng(seed)
        policy = self._call("load_stage4_policy_for_standard")(
            checkpoint_path=self._components["stage4_checkpoint_path"],
            checkpoint_sha256=self.config.stage5_authority.checkpoint_sha256,
            policy_state_sha256=self.config.stage5_authority.policy_state_sha256,
            device=self.config.device,
        )
        if not isinstance(policy, nn.Module):
            raise StandardTrainingError("production Stage 4 policy load drifted")
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=self.config.ppo.learning_rate,
            eps=self.config.ppo.adam_eps,
            weight_decay=self.config.ppo.weight_decay,
        )
        trainer = self._call("PPOTrainer")(
            policy,
            device=self.config.device,
            optimizer=optimizer,
            shuffle_seed=seed,
        )
        loaded = None
        if complete_directories:
            if self._immutable_bindings is None:
                raise StandardTrainingError("resume requires immutable execution bindings")
            loaded = manager.load_last_complete(
                policy=policy,
                optimizer=optimizer,
                expected_config_sha256=self.config_sha256,
                expected_lineage=self._checkpoint_lineage_for_update(
                    highest_update
                ),
                expected_safety_contract=self._safety_contract(),
            )
            loaded_transaction = next(
                (
                    item
                    for item in self.transactions
                    if item.seed == seed and item.update == loaded.update_step
                ),
                None,
            )
            if loaded_transaction is None:
                raise StandardTrainingError(
                    "resume checkpoint transaction identity drifted"
                )
            loaded_checkpoint = StandardTransactionCheckpoint(
                transaction_key=loaded_transaction.key,
                seed=loaded_transaction.seed,
                update=loaded_transaction.update,
                checkpoint_sha256=str(loaded.checkpoint_sha256),
                complete_marker_sha256=hashlib.sha256(
                    (
                        Path(
                            getattr(
                                loaded,
                                "directory",
                                checkpoint_root
                                / f"update-{int(loaded.update_step):08d}",
                            )
                        )
                        / "complete.json"
                    ).read_bytes()
                ).hexdigest(),
                policy_state_sha256=str(loaded.policy_state_sha256),
            )
            self._recover_update_resource_acceptance(
                loaded_transaction,
                loaded_checkpoint,
            )
            receipt = next(
                (
                    row
                    for row in receipt_rows
                    if row["seed"] == seed and row["update"] == loaded.update_step
                ),
                None,
            )
            if receipt is None:
                next_transaction = (
                    self.transactions[len(receipt_rows)]
                    if len(receipt_rows) < len(self.transactions)
                    else None
                )
                loaded_directory = Path(
                    getattr(
                        loaded,
                        "directory",
                        checkpoint_root / f"update-{int(loaded.update_step):08d}",
                    )
                ).resolve()
                complete_marker = loaded_directory / "complete.json"
                if (
                    next_transaction is None
                    or next_transaction.seed != seed
                    or next_transaction.update != loaded.update_step
                    or not complete_marker.is_file()
                    or loaded_directory.parent != checkpoint_root
                ):
                    raise StandardTrainingError(
                        "resume checkpoint is more than one receipt ahead"
                    )
                reconstructed = StandardTransactionCheckpoint(
                    transaction_key=next_transaction.key,
                    seed=seed,
                    update=int(loaded.update_step),
                    checkpoint_sha256=str(loaded.checkpoint_sha256),
                    complete_marker_sha256=hashlib.sha256(
                        complete_marker.read_bytes()
                    ).hexdigest(),
                    policy_state_sha256=str(loaded.policy_state_sha256),
                )
                self.receipt_index.append_once(reconstructed)
                receipt_rows = self.receipt_index.verify()
                receipt = receipt_rows[-1]
            if (
                not isinstance(receipt, Mapping)
                or receipt["checkpoint_sha256"] != loaded.checkpoint_sha256
                or receipt["policy_state_sha256"] != loaded.policy_state_sha256
            ):
                raise StandardTrainingError("resume complete checkpoint receipt drifted")
            checkpoint = StandardTransactionCheckpoint(
                transaction_key=str(receipt["transaction_key"]),
                seed=int(receipt["seed"]),
                update=int(receipt["update"]),
                checkpoint_sha256=str(receipt["checkpoint_sha256"]),
                complete_marker_sha256=str(receipt["complete_marker_sha256"]),
                policy_state_sha256=str(receipt["policy_state_sha256"]),
            )
            bindings = {
                **self._immutable_bindings,
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
            }
            remaining_all, _ = reconcile_checkpointed_resume(
                transactions=self.transactions,
                journal=self.journal,
                checkpoint=checkpoint,
                bindings=bindings,
                receipt_rows=receipt_rows,
                historical_immutable_bindings=historical_immutable_bindings,
                current_binding_first_transaction_key=(
                    current_binding_first_transaction_key
                ),
                immutable_binding_epochs=immutable_binding_epochs,
            )
            trainer.restore_update_step(loaded.update_step)
            self._recover_validation_trace(loaded_transaction, loaded=loaded)
            self._backfill_loaded_checkpoint_metrics(
                loaded=loaded,
                receipt=receipt,
            )
            self._apply_checkpoint_retention(
                checkpoint_root=checkpoint_root,
                update=int(loaded.update_step),
                best_record=dict(loaded.best_record),
            )
        else:
            if any(row["seed"] == seed for row in receipt_rows):
                raise StandardTrainingError("seed checkpoint files are missing")
            if self._immutable_bindings is None:
                if receipt_rows or self.journal.verify():
                    raise StandardTrainingError("resume identity is unavailable")
                completed = ()
            else:
                completed = verify_journal_checkpoint_bindings(
                    self.journal.verify(),
                    self.transactions,
                    receipt_rows,
                    self._immutable_bindings,
                    historical_immutable_bindings=historical_immutable_bindings,
                    current_binding_first_transaction_key=(
                        current_binding_first_transaction_key
                    ),
                    immutable_binding_epochs=immutable_binding_epochs,
                )
            completed_set = set(completed)
            remaining_all = tuple(
                item for item in self.transactions if item.key not in completed_set
            )

        parent_warm_runtime = None
        if (
            loaded is None
            and getattr(self, "_planning_warm_start", None) is not None
        ):
            if receipt_rows or self.journal.verify():
                raise StandardTrainingError(
                    "fresh planning warm-start child root is not empty"
                )
            from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
                PARENT_SEED,
                PARENT_UPDATE,
                restore_parent_u74_runtime,
            )

            parent_warm_runtime = restore_parent_u74_runtime(
                verified_parent=self._verified_parent_u74,
                policy=policy,
                optimizer=optimizer,
            )
            if (
                seed != PARENT_SEED
                or parent_warm_runtime.update_step != PARENT_UPDATE
            ):
                raise StandardTrainingError(
                    "parent planning warm-start runtime drifted"
                )
            trainer.restore_update_step(PARENT_UPDATE)

        remaining = tuple(item for item in remaining_all if item.seed == seed)
        if remaining != seed_transactions[len(seed_transactions) - len(remaining) :]:
            raise StandardTrainingError("seed resume is not an exact suffix")
        self._require_planning_child_resume_boundary(remaining)
        collector_update = (
            remaining[0].update if remaining else seed_transactions[-1].update
        )

        sampler_seeds = derive_standard_sampler_seeds(seed)
        initial_sampler_states = None
        if parent_warm_runtime is not None:
            workers = parent_warm_runtime.scenario_sampler_state.get("workers")
            if (
                not isinstance(workers, list)
                or len(workers) != 8
                or any(not isinstance(state, Mapping) for state in workers)
            ):
                raise StandardTrainingError(
                    "parent planning sampler state drifted"
                )
            initial_sampler_states = tuple(dict(state) for state in workers)
        specs = self._production_training_env_specs(
            split="train",
            sampler_seeds=sampler_seeds,
            safety_contract=self._safety_contract(),
            initial_sampler_states=initial_sampler_states,
        )
        vector_env = self._call("SpawnVectorEnv")(specs, timeout_seconds=900.0)
        prepared_successfully = False
        try:
            if loaded is not None:
                vector_env.restore_states(loaded.vector_env_states)
            lineage = (
                self._checkpoint_lineage(collector_update)
                if self._immutable_bindings
                else self.stage5_authority
            )
            contract = self._call("CollectorContract")(
                config_sha256=self.config_sha256,
                lineage=lineage,
                frontier_extractor_version="stage2_observed_frontier_top_m/v1",
                observation_schema_version=self.config.observation_schema_version,
                action_space_version=self.config.action_space_version,
                reward_version=self.config.reward_version,
                planner_version=self.config.planner_version,
            )
            collector = self._call("RolloutCollector")(
                policy=policy,
                vector_env=vector_env,
                device=self.config.device,
                contract=contract,
                require_dual_scan=self._planning_warm_start is not None,
            )
            runtime = {
                "seed": seed,
                "policy": policy,
                "optimizer": optimizer,
                "trainer": trainer,
                "collector": collector,
                "vector_env": vector_env,
                "checkpoint_manager": manager,
                "checkpoint_root": checkpoint_root,
                "normalization_stats": (
                    dict(loaded.normalization_stats)
                    if loaded is not None
                    else (
                        dict(parent_warm_runtime.normalization_stats)
                        if parent_warm_runtime is not None
                        else {}
                    )
                ),
                "best_record": dict(loaded.best_record) if loaded is not None else {},
                "initial_policy_sha256": (
                    self._planning_warm_start.artifact["parent"][
                        "policy_state_sha256"
                    ]
                    if self._planning_warm_start is not None
                    else self.config.stage5_authority.policy_state_sha256
                ),
                "optimizer_fresh": (
                    loaded is None and parent_warm_runtime is None
                ),
                "independent_seed_origin": self._planning_warm_start is None,
                "rng_initialization": (
                    "fresh_seed_all_rng/v1"
                    if loaded is None and parent_warm_runtime is None
                    else (
                        "parent_u74_checkpoint_restore_only/v1"
                        if parent_warm_runtime is not None
                        else "checkpoint_restore_only/v1"
                    )
                ),
                "sampler_seed_derivation_version": SAMPLER_SEED_DERIVATION_VERSION,
                "sampler_seeds": sampler_seeds,
                "runtime_token": f"stage6-seed-{seed}",
                "continue_from_current_state": loaded is not None,
                "planning_warm_start": self._planning_warm_start is not None,
                "parent_update": (
                    74 if self._planning_warm_start is not None else None
                ),
                "fresh_vector_env_reset_required": (
                    self._planning_warm_start is not None
                    and loaded is None
                ),
            }
            if loaded is not None and audit_required(int(loaded.update_step)):
                loaded_eval_metrics = getattr(loaded, "eval_metrics", None)
                if not isinstance(loaded_eval_metrics, Mapping):
                    raise StandardTrainingError("resume checkpoint audit metrics drifted")
                replay_result = StandardUpdateResult(
                    transaction=next(
                        item
                        for item in self.transactions
                        if item.key == checkpoint.transaction_key
                    ),
                    update_metrics=loaded_eval_metrics.get("update", {}),
                    validation_result=loaded_eval_metrics.get("validation", {}),
                    eval_isolation_audit=loaded_eval_metrics.get("eval_isolation", {}),
                    best_record=dict(getattr(loaded, "best_record", {})),
                    checkpoint=checkpoint,
                    journal_records=(),
                    collection_audit=loaded_eval_metrics.get(
                        "collection_audit", {}
                    ),
                )
                checkpoint_audit = self._checkpoint_replay_audit(
                    runtime=runtime,
                    transaction=replay_result.transaction,
                    result=replay_result,
                )
                self._require_execution_capability_current(
                    "resume checkpoint audit publication",
                    rehash_inputs=True,
                )
                append_transaction_metric_once(
                    self.stage_root / "checkpoint_audit.jsonl",
                    checkpoint_audit,
                )
            prepared_successfully = True
            return runtime, remaining
        finally:
            if not prepared_successfully:
                vector_env.close()

    @_guarded_backend_mutation
    def _backfill_loaded_checkpoint_metrics(
        self,
        *,
        loaded: object,
        receipt: Mapping[str, object],
    ) -> None:
        transaction = next(
            (
                item
                for item in self.transactions
                if item.key == receipt.get("transaction_key")
            ),
            None,
        )
        eval_metrics = getattr(loaded, "eval_metrics", None)
        if (
            transaction is None
            or not isinstance(eval_metrics, Mapping)
            or set(eval_metrics)
            != {"validation", "update", "eval_isolation", "collection_audit"}
            or not isinstance(eval_metrics["update"], Mapping)
            or not isinstance(eval_metrics["validation"], Mapping)
            or not isinstance(eval_metrics["eval_isolation"], Mapping)
            or not isinstance(eval_metrics["collection_audit"], Mapping)
        ):
            raise StandardTrainingError("resume checkpoint metrics drifted")
        self._require_execution_capability_current(
            "resume training metrics publication",
            rehash_inputs=True,
        )
        append_transaction_metric_once(
            self.stage_root / "training_metrics.jsonl",
            {
                "transaction_key": transaction.key,
                "seed": transaction.seed,
                "update": transaction.update,
                "checkpoint_sha256": receipt["checkpoint_sha256"],
                "policy_state_sha256": receipt["policy_state_sha256"],
                "update_metrics": dict(eval_metrics["update"]),
                "validation": dict(eval_metrics["validation"]),
                "collection_audit": dict(eval_metrics["collection_audit"]),
            },
        )
        if transaction.validation_episodes:
            self._require_execution_capability_current(
                "resume validation metrics publication",
                rehash_inputs=True,
            )
            append_transaction_metric_once(
                self.stage_root / "validation_metrics.jsonl",
                {
                    "transaction_key": transaction.key,
                    "seed": transaction.seed,
                    "update": transaction.update,
                    "episode_count": transaction.validation_episodes,
                    "result": dict(eval_metrics["validation"]),
                    "eval_isolation": dict(eval_metrics["eval_isolation"]),
                    "best_record": dict(getattr(loaded, "best_record", {})),
                },
            )
        if audit_required(transaction.update):
            math_audit = _math_audit_from_completed_update(
                eval_metrics["update"],
                collection_audit=eval_metrics["collection_audit"],
                device=self.config.device,
                compute_dtype=self.config.ppo.compute_dtype,
                require_dual_scan=self._planning_warm_start is not None,
            )
            self._require_execution_capability_current(
                "resume math audit publication",
                rehash_inputs=True,
            )
            append_transaction_metric_once(
                self.stage_root / "math_audit.jsonl",
                {
                    "transaction_key": transaction.key,
                    "seed": transaction.seed,
                    "update": transaction.update,
                    **math_audit,
                },
            )

    def _safety_contract(self) -> SafetyContract:
        expected = SafetyContract.from_stage6_config(self.config)
        value = getattr(self, "safety_contract", None)
        if not isinstance(value, SafetyContract) or value != expected:
            raise StandardTrainingError("Stage 6 safety contract drifted")
        try:
            restored = SafetyContract.from_binding(
                value.binding(config_sha256=self.config_sha256),
                expected_config_sha256=self.config_sha256,
            )
        except (TypeError, ValueError) as exc:
            raise StandardTrainingError(
                "Stage 6 safety/config binding drifted"
            ) from exc
        if restored != expected:
            raise StandardTrainingError("Stage 6 safety contract drifted")
        return value

    def _validated_evaluation_safety_contract(
        self,
        result: object,
    ) -> SafetyContract:
        if isinstance(result, Mapping):
            fairness = result.get("fairness_audit")
        else:
            fairness = getattr(result, "fairness_audit", None)
        shared = (
            fairness.get("shared_environment_contract")
            if isinstance(fairness, Mapping)
            else None
        )
        environment = (
            shared.get("environment_contract")
            if isinstance(shared, Mapping)
            else None
        )
        expected = self._safety_contract()
        if not isinstance(environment, Mapping):
            raise StandardTrainingError("evaluation safety contract drifted")
        binding = {
            name: environment.get(name)
            for name in expected.binding(config_sha256=self.config_sha256)
        }
        try:
            restored = SafetyContract.from_binding(
                binding,
                expected_config_sha256=self.config_sha256,
            )
        except (TypeError, ValueError) as exc:
            raise StandardTrainingError(
                "evaluation safety/config binding drifted"
            ) from exc
        if restored != expected:
            raise StandardTrainingError("evaluation safety contract drifted")
        return restored

    def _checkpoint_lineage(self, update: int) -> dict[str, object]:
        return self._checkpoint_lineage_for_update(update)

    def _checkpoint_lineage_for_update(self, update: int) -> dict[str, object]:
        if type(update) is not int or update <= 0:
            raise StandardTrainingError("checkpoint update identity drifted")
        recovery_capability = getattr(
            self,
            "_planning_child_recovery_capability",
            None,
        )
        if recovery_capability is not None:
            from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
                Stage6PlanningChildRecoveryError,
                _plain_json,
            )

            try:
                lineage = _plain_json(
                    recovery_capability.checkpoint_lineage_for_update(
                        update,
                    )
                )
            except Stage6PlanningChildRecoveryError as exc:
                raise StandardTrainingError(
                    "planning child recovery checkpoint lineage drifted"
                ) from exc
            if not isinstance(lineage, dict):
                raise StandardTrainingError(
                    "planning child recovery checkpoint lineage drifted"
                )
            return lineage
        warm_start = getattr(self, "_planning_warm_start", None)
        if warm_start is not None:
            try:
                return {
                    **self._origin_checkpoint_lineage(),
                    **warm_start.checkpoint_lineage_for_update(
                        update,
                        warm_start_artifact_sha256=(
                            self._planning_warm_start_sha256
                        ),
                    ),
                }
            except Exception as exc:
                raise StandardTrainingError(
                    "planning warm-start checkpoint lineage drifted"
                ) from exc
        source_repair = getattr(self, "_source_repair", None)
        if source_repair is not None:
            from lunar_exploration_ppo.workflows.stage6_source_repair import (
                Stage6SourceRepairError,
            )

            try:
                return source_repair.checkpoint_lineage_for_update(update)
            except Stage6SourceRepairError as exc:
                raise StandardTrainingError(
                    "source-repair checkpoint lineage drifted"
                ) from exc
        return self._origin_checkpoint_lineage()

    def _origin_checkpoint_lineage(
        self,
        immutable_bindings: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        bindings = (
            self._immutable_bindings
            if immutable_bindings is None
            else immutable_bindings
        )
        if bindings is None:
            raise StandardTrainingError("checkpoint lineage identity is unavailable")
        safety_contract_sha256 = self._safety_contract().sha256
        return {
            "stage5_commit": self.config.stage5_authority.commit_sha256,
            "stage5_gate_sha256": self.config.stage5_authority.gate_sha256,
            "stage5_manifest_sha256": self.config.stage5_authority.manifest_sha256,
            "stage4_checkpoint_sha256": self.config.stage5_authority.checkpoint_sha256,
            "stage4_policy_state_sha256": self.config.stage5_authority.policy_state_sha256,
            "source_set_sha256": bindings["source_set_sha256"],
            "prospective_tree_sha256": bindings[
                "prospective_tree_sha256"
            ],
            "data_sha256": bindings["data_sha256"],
            "environment_identity": bindings[
                "environment_identity"
            ],
            "environment_sha256": bindings[
                "environment_sha256"
            ],
            "formal_run_id": bindings["formal_run_id"],
            "changed_path_set_sha256": bindings[
                "changed_path_set_sha256"
            ],
            "review_authorization_record_sha256": bindings[
                "review_authorization_record_sha256"
            ],
            "authorization_file_sha256": bindings[
                "authorization_file_sha256"
            ],
            "review_identity_sha256": bindings[
                "review_identity_sha256"
            ],
            "reviewed_prospective_git_tree": bindings[
                "reviewed_prospective_git_tree"
            ],
            "frozen_diff_sha256": bindings[
                "frozen_diff_sha256"
            ],
            "spec_review_sha256": bindings[
                "spec_review_sha256"
            ],
            "quality_review_sha256": bindings[
                "quality_review_sha256"
            ],
            "safety_contract_sha256": safety_contract_sha256,
        }

    @_guarded_backend_mutation
    def run_update(self, runtime, transaction):
        recovery_capability = getattr(
            self,
            "_planning_child_recovery_capability",
            None,
        )
        if recovery_capability is None:
            self._require_source_repair_current("run_update")
        if (
            not isinstance(runtime, Mapping)
            or not isinstance(transaction, StandardTrainingTransaction)
            or runtime.get("seed") != transaction.seed
            or self._immutable_bindings is None
        ):
            raise StandardTrainingError("production update runtime drifted")
        resource_identity = {
            "kind": "update",
            "transaction_key": transaction.key,
            "seed": transaction.seed,
            "update": transaction.update,
        }
        resource_attempt = self._next_resource_attempt(transaction.key)
        resource_latch = self._new_runtime_resource_latch(
            transaction_key=transaction.key,
            attempt=resource_attempt,
        )

        def assert_capability(
            boundary: str,
            *,
            rehash_inputs: bool = False,
        ) -> None:
            self._require_execution_capability_current(
                f"run_update:{boundary}",
                rehash_inputs=rehash_inputs,
            )

        def assert_lifecycle_boundary(
            boundary: str,
            *,
            rehash_inputs: bool = False,
        ) -> None:
            if recovery_capability is None:
                self._require_source_repair_current(
                    f"run_update:{boundary}"
                )
            assert_capability(
                boundary,
                rehash_inputs=(
                    rehash_inputs
                    or boundary.startswith("checkpoint:")
                ),
            )

        def assert_resource(boundary: str) -> None:
            assert_capability(boundary)
            self._poll_runtime_resource_latch(resource_latch, boundary)

        pre_resource = self._poll_runtime_resource_latch(
            resource_latch,
            "update:pre",
        )
        self._append_resource_phase(
            identity=resource_identity,
            attempt=resource_attempt,
            phase="pre",
            resource=pre_resource,
        )
        if pre_resource["passed"] is not True:
            raise StandardTrainingError("runtime resource hard stop")

        validation_evaluator = None
        eval_binding_state_provider = None
        validation_trace_path: Path | None = None
        validation_pending_path: Path | None = None
        if transaction.validation_episodes:
            validation_trace_path = self.stage_root / (
                "episode-traces/"
                f"validation-seed-{transaction.seed}-update-{transaction.update:03d}.jsonl"
            )
            validation_pending_path = self.run_root / (
                "stage6-attempts/"
                f"validation-seed-{transaction.seed}-update-{transaction.update:03d}-"
                f"attempt-{resource_attempt:03d}.pending.jsonl"
            )

            def validation_evaluator():
                evaluation = self._call("run_standard_evaluation")(
                    catalog=self.catalog,
                    split="validation",
                    method="ppo_policy",
                    episode_count=transaction.validation_episodes,
                    evaluation_seed_start=self.config.evaluation.bootstrap_seed,
                    policy=runtime["policy"],
                    policy_device=self.config.device,
                    bootstrap_resamples=self.config.evaluation.bootstrap_resamples,
                    bootstrap_seed=self.config.evaluation.bootstrap_seed,
                    trace_path=validation_pending_path,
                    safety_contract=self._safety_contract(),
                    config_sha256=self.config_sha256,
                    **self._coverage_cache_binding(),
                    resource_guard=assert_resource,
                )
                self._validated_evaluation_safety_contract(evaluation)
                try:
                    pending_read = secure_read_bytes(
                        validation_pending_path,
                        base=self.run_root,
                        label="validation trace pending",
                    )
                    trace_binding = _validation_trace_binding_from_bytes(
                        pending_read.payload
                    )
                    _validated_validation_trace_binding(
                        trace_binding,
                        episode_count=transaction.validation_episodes,
                        payload=pending_read.payload,
                    )
                except StandardTrainingError:
                    raise
                except (OSError, PathSecurityError) as exc:
                    raise StandardTrainingError(
                        "validation trace binding drifted"
                    ) from exc
                if isinstance(evaluation, Mapping):
                    evaluation = {
                        **dict(evaluation),
                        "validation_trace_binding": trace_binding,
                    }
                else:
                    metrics = getattr(evaluation, "metrics", None)
                    if not isinstance(metrics, dict):
                        raise StandardTrainingError(
                            "validation trace metrics are not mutable"
                        )
                    metrics["validation_trace_binding"] = trace_binding
                return evaluation

            def eval_binding_state_provider():
                return {
                    "config_bytes": (self.stage_root / "config.json").read_bytes(),
                    "checkpoint_bytes": self._checkpoint_binding_bytes(
                        Path(runtime["checkpoint_root"])
                    ),
                }

        versions = {
            "observation_schema_version": self.config.observation_schema_version,
            "action_space_version": self.config.action_space_version,
            "network_architecture_version": self.config.network_architecture_version,
            "reward_version": self.config.reward_version,
            "frontier_version": "stage2_observed_frontier_top_m/v1",
            "planner_version": self.config.planner_version,
        }
        def persist_post_resource(
            checkpoint: StandardTransactionCheckpoint,
        ) -> None:
            if checkpoint.transaction_key != transaction.key:
                raise StandardTrainingError(
                    "post-resource checkpoint transaction drifted"
                )
            self._require_execution_capability_current(
                "run_update:post-resource-write",
                rehash_inputs=True,
            )
            post_resource = self._poll_runtime_resource_latch(
                resource_latch,
                "update:post-checkpoint-prepare",
            )
            self._append_resource_phase(
                identity=resource_identity,
                attempt=resource_attempt,
                phase="post",
                resource=post_resource,
            )

        def accept_post_resource(
            checkpoint: StandardTransactionCheckpoint,
        ) -> None:
            if checkpoint.transaction_key != transaction.key:
                raise StandardTrainingError(
                    "resource acceptance checkpoint transaction drifted"
                )
            self._require_execution_capability_current(
                "run_update:accepted-resource-write",
                rehash_inputs=True,
            )
            self._recover_update_resource_acceptance(transaction, checkpoint)

        (
            historical_immutable_bindings,
            current_binding_first_transaction_key,
        ) = self._journal_binding_split()
        immutable_binding_epochs = self._journal_binding_epochs()
        result = self._call("run_standard_update_transaction")(
            transaction=transaction,
            transactions=self.transactions,
            collector=runtime["collector"],
            trainer=runtime["trainer"],
            checkpoint_manager=runtime["checkpoint_manager"],
            checkpoint_receipt_index=self.receipt_index,
            journal=self.journal,
            policy=runtime["policy"],
            optimizer=runtime["optimizer"],
            config=self.config,
            normalization_stats=runtime["normalization_stats"],
            checkpoint_metadata={
                "best_record": runtime["best_record"],
                "versions": versions,
                "top_m_config": {
                    "frontier_top_m": self.config.scale.frontier_top_m,
                    "selection": "stage2_observed_frontier_top_m/v1",
                },
                "scale_profile": self.config.scale.profile,
                "training_config": {
                    "rollout": self.config.rollout.model_dump(mode="json"),
                    "ppo": self.config.ppo.model_dump(mode="json"),
                    "training": self.config.training.model_dump(mode="json"),
                    "safety": self._safety_contract().to_dict(),
                },
                "config_sha256": self.config_sha256,
                "lineage": self._checkpoint_lineage(transaction.update),
                "safety_contract": self._safety_contract(),
            },
            journal_bindings={
                **self._immutable_bindings,
                "checkpoint_sha256": "0" * 64,
            },
            validation_evaluator=validation_evaluator,
            eval_binding_state_provider=eval_binding_state_provider,
            continue_from_current_state=bool(
                runtime["continue_from_current_state"]
            ),
            historical_immutable_bindings=historical_immutable_bindings,
            current_binding_first_transaction_key=(
                current_binding_first_transaction_key
            ),
            immutable_binding_epochs=immutable_binding_epochs,
            resource_guard=assert_resource,
            capability_guard=assert_lifecycle_boundary,
            persist_post_resource=persist_post_resource,
            accept_post_resource=accept_post_resource,
            require_dual_scan=self._planning_warm_start is not None,
        )
        if not isinstance(result, StandardUpdateResult):
            raise StandardTrainingError("production update result drifted")
        update_metric_record = _jsonable_runtime_value(result.update_metrics)
        if not isinstance(update_metric_record, Mapping):
            raise StandardTrainingError("production update metrics drifted")
        policy_before = update_metric_record.get("policy_state_sha256_before")
        policy_after = update_metric_record.get("policy_state_sha256_after")
        collection_audit = validate_standard_collection_audit(
            result.collection_audit,
            expected_device=self.config.device,
            expected_policy_sha256=(
                str(policy_before) if _is_sha256(policy_before) else None
            ),
            expected_inference_pid=os.getpid(),
            require_dual_scan=self._planning_warm_start is not None,
        )
        if (
            not _is_sha256(policy_before)
            or not _is_sha256(policy_after)
            or collection_audit["policy_state_sha256"] != policy_before
            or result.checkpoint.policy_state_sha256 != policy_after
        ):
            raise StandardTrainingError("production on-policy lineage drifted")
        if validation_pending_path is not None and validation_trace_path is not None:
            assert_lifecycle_boundary(
                "before validation trace publication",
                rehash_inputs=True,
            )
            validation_metrics = _evaluation_metrics_mapping(
                result.validation_result
            )
            trace_binding = _validated_validation_trace_binding(
                validation_metrics.get("validation_trace_binding"),
                episode_count=transaction.validation_episodes,
            )
            self._publish_validation_trace(
                pending_path=validation_pending_path,
                canonical_path=validation_trace_path,
                episode_count=transaction.validation_episodes,
                expected_binding=trace_binding,
            )
        runtime["best_record"] = dict(result.best_record)  # type: ignore[index]
        runtime["continue_from_current_state"] = True  # type: ignore[index]
        metric = {
            "transaction_key": transaction.key,
            "seed": transaction.seed,
            "update": transaction.update,
            "checkpoint_sha256": result.checkpoint.checkpoint_sha256,
            "policy_state_sha256": result.checkpoint.policy_state_sha256,
            "update_metrics": dict(update_metric_record),
            "validation": _evaluation_metrics_mapping(result.validation_result),
            "collection_audit": collection_audit,
        }
        assert_lifecycle_boundary(
            "before training metrics publication",
            rehash_inputs=True,
        )
        self._call("append_transaction_metric_once")(
            self.stage_root / "training_metrics.jsonl",
            metric,
        )
        if transaction.validation_episodes:
            assert_lifecycle_boundary(
                "before validation metrics publication",
                rehash_inputs=True,
            )
            self._call("append_transaction_metric_once")(
                self.stage_root / "validation_metrics.jsonl",
                {
                    "transaction_key": transaction.key,
                    "seed": transaction.seed,
                    "update": transaction.update,
                    "episode_count": transaction.validation_episodes,
                    "result": _evaluation_metrics_mapping(
                        result.validation_result
                    ),
                    "eval_isolation": _jsonable_runtime_value(
                        result.eval_isolation_audit
                    ),
                    "best_record": dict(result.best_record),
                },
            )
        if audit_required(transaction.update):
            assert_lifecycle_boundary(
                "before math audit publication",
                rehash_inputs=True,
            )
            math_audit = _math_audit_from_completed_update(
                update_metric_record,
                collection_audit=collection_audit,
                device=self.config.device,
                compute_dtype=self.config.ppo.compute_dtype,
                require_dual_scan=self._planning_warm_start is not None,
            )
            append_transaction_metric_once(
                self.stage_root / "math_audit.jsonl",
                {
                    "transaction_key": transaction.key,
                    "seed": transaction.seed,
                    "update": transaction.update,
                    **math_audit,
                },
            )
            checkpoint_audit = self._checkpoint_replay_audit(
                runtime=runtime,
                transaction=transaction,
                result=result,
            )
            assert_lifecycle_boundary(
                "before checkpoint audit publication",
                rehash_inputs=True,
            )
            append_transaction_metric_once(
                self.stage_root / "checkpoint_audit.jsonl",
                checkpoint_audit,
            )
        self._checkpoint_writes += 1
        assert_lifecycle_boundary(
            "before checkpoint retention",
            rehash_inputs=True,
        )
        self._apply_checkpoint_retention(
            checkpoint_root=Path(runtime["checkpoint_root"]),
            update=transaction.update,
            best_record=runtime["best_record"],
        )
        assert_lifecycle_boundary("after checkpoint retention")
        return result

    def _checkpoint_replay_audit(
        self,
        *,
        runtime: Mapping[str, object],
        transaction: StandardTrainingTransaction,
        result: StandardUpdateResult,
    ) -> dict[str, object]:
        policy = runtime.get("policy")
        optimizer = runtime.get("optimizer")
        vector_env = runtime.get("vector_env")
        manager = runtime.get("checkpoint_manager")
        capture_states = getattr(vector_env, "capture_states", None)
        restore_states = getattr(vector_env, "restore_states", None)
        current_observations = getattr(vector_env, "current_observations", None)
        load_last_complete = getattr(manager, "load_last_complete", None)
        if (
            not isinstance(policy, nn.Module)
            or not isinstance(optimizer, torch.optim.Optimizer)
            or not callable(capture_states)
            or not callable(restore_states)
            or not callable(current_observations)
            or not callable(load_last_complete)
        ):
            raise StandardTrainingError("checkpoint replay collaborator drifted")
        vector_states = tuple(capture_states())
        observations = current_observations()
        if (
            len(vector_states) != 8
            or any(not isinstance(value, Mapping) for value in vector_states)
            or not isinstance(observations, Mapping)
            or set(observations) != set(range(8))
        ):
            raise StandardTrainingError("checkpoint replay worker state drifted")
        replay_worker = next(
            (
                index
                for index in range(8)
                if getattr(observations[index], "needs_policy", True)
            ),
            None,
        )
        if replay_worker is None or not hasattr(observations[replay_worker], "observation"):
            raise StandardTrainingError("checkpoint replay observation is unavailable")

        def runtime_hashes() -> tuple[str, str, str]:
            rng_state = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": (
                    tuple(torch.cuda.get_rng_state_all())
                    if torch.cuda.is_available()
                    else ()
                ),
            }
            return (
                policy_state_sha256(policy),
                _runtime_value_sha256(optimizer.state_dict()),
                _runtime_value_sha256(rng_state),
            )

        original_training = policy.training
        action = self._call("deterministic_action_record")
        before_hashes = runtime_hashes()
        before_observation_sha256 = _runtime_value_sha256(
            getattr(
                observations[replay_worker].observation,
                "array_fields",
                lambda: observations[replay_worker].observation,
            )()
            if callable(
                getattr(observations[replay_worker].observation, "array_fields", None)
            )
            else observations[replay_worker].observation
        )
        before_action = action(
            policy,
            observations[replay_worker].observation,
            device=self.config.device,
        )
        policy.train(original_training)
        loaded = load_last_complete(
            policy=policy,
            optimizer=optimizer,
            expected_config_sha256=self.config_sha256,
            expected_lineage=self._checkpoint_lineage(transaction.update),
            expected_safety_contract=self._safety_contract(),
        )
        loaded_vector_states = tuple(getattr(loaded, "vector_env_states", ()))
        try:
            restore_states(loaded_vector_states)
            restored_vector_states = tuple(capture_states())
            restored_observations = current_observations()
        except Exception as exc:
            raise StandardTrainingError(
                "checkpoint vector state restore failed"
            ) from exc
        if (
            len(restored_vector_states) != 8
            or any(not isinstance(value, Mapping) for value in restored_vector_states)
            or not isinstance(restored_observations, Mapping)
            or set(restored_observations) != set(range(8))
            or not hasattr(restored_observations[replay_worker], "observation")
        ):
            raise StandardTrainingError("checkpoint vector state round-trip drifted")
        restored_observation = restored_observations[replay_worker].observation
        restored_observation_sha256 = _runtime_value_sha256(
            restored_observation.array_fields()
            if callable(getattr(restored_observation, "array_fields", None))
            else restored_observation
        )
        after_action = action(
            policy,
            restored_observation,
            device=self.config.device,
        )
        policy.train(original_training)
        after_hashes = runtime_hashes()
        expected_sampler = {
            "workers": [dict(state).get("sampler_state", {}) for state in vector_states]
        }
        if (
            before_hashes != after_hashes
            or before_action != after_action
            or getattr(loaded, "update_step", None) != transaction.update
            or getattr(loaded, "checkpoint_sha256", None)
            != result.checkpoint.checkpoint_sha256
            or getattr(loaded, "policy_state_sha256", None)
            != result.checkpoint.policy_state_sha256
            or getattr(loaded, "config_sha256", None) != self.config_sha256
            or getattr(loaded, "lineage", None)
            != self._checkpoint_lineage(transaction.update)
            or getattr(loaded, "normalization_stats", None)
            != runtime.get("normalization_stats")
            or getattr(loaded, "scenario_sampler_state", None) != expected_sampler
            or _runtime_value_sha256(loaded_vector_states)
            != _runtime_value_sha256(vector_states)
            or _runtime_value_sha256(restored_vector_states)
            != _runtime_value_sha256(loaded_vector_states)
            or restored_observation_sha256 != before_observation_sha256
            or getattr(loaded, "best_record", None) != dict(result.best_record)
        ):
            raise StandardTrainingError("checkpoint deterministic replay failed")
        audit = {
            "transaction_key": transaction.key,
            "seed": transaction.seed,
            "update": transaction.update,
            "schema_version": "stage6_checkpoint_replay_audit/v1",
            "passed": True,
            "last_complete_loaded": True,
            "policy_state_unchanged": True,
            "optimizer_state_unchanged": True,
            "rng_state_unchanged": True,
            "normalizer_restored": True,
            "scenario_sampler_state_restored": True,
            "vector_env_state_count": len(vector_states),
            "vector_env_state_round_trip_bit_exact": True,
            "observation_round_trip_bit_exact": True,
            "deterministic_action_bit_exact": True,
            "checkpoint_sha256": result.checkpoint.checkpoint_sha256,
            "complete_marker_sha256": (
                result.checkpoint.complete_marker_sha256
            ),
            "policy_state_sha256": result.checkpoint.policy_state_sha256,
            "lineage_sha256": _runtime_value_sha256(
                self._checkpoint_lineage(transaction.update)
            ),
            "lineage_unchanged": True,
        }
        return validate_checkpoint_replay_audit(
            audit,
            transaction_key=transaction.key,
            seed=transaction.seed,
            update=transaction.update,
            checkpoint_sha256=result.checkpoint.checkpoint_sha256,
            complete_marker_sha256=result.checkpoint.complete_marker_sha256,
            policy_state_sha256=result.checkpoint.policy_state_sha256,
            lineage_sha256=_runtime_value_sha256(
                self._checkpoint_lineage(transaction.update)
            ),
        )

    @staticmethod
    def _checkpoint_binding_bytes(
        checkpoint_root: Path,
        update_step: int | None = None,
    ) -> bytes:
        directories = sorted(
            path
            for path in checkpoint_root.iterdir()
            if path.is_dir() and path.name.startswith("update-")
        )
        if not directories:
            raise StandardTrainingError("eval binding lacks complete checkpoint")
        directory = (
            checkpoint_root / f"update-{update_step:08d}"
            if update_step is not None
            else directories[-1]
        )
        if directory not in directories:
            raise StandardTrainingError("eval binding checkpoint was not retained")
        chunks: list[bytes] = []
        for name in ("checkpoint.pt", "manifest.json", "complete.json"):
            path = directory / name
            if not path.is_file():
                raise StandardTrainingError("eval binding checkpoint bundle drifted")
            payload = path.read_bytes()
            chunks.extend((name.encode("ascii"), b"\0", payload, b"\0"))
        return b"".join(chunks)

    @_guarded_backend_mutation
    def finish_seed(self, runtime):
        if not isinstance(runtime, Mapping):
            raise StandardTrainingError("production seed runtime drifted")
        close = getattr(runtime.get("vector_env"), "close", None)
        if not callable(close):
            raise StandardTrainingError("production seed vector close is unavailable")
        close()
        value = runtime.get("best_record")
        if not isinstance(value, Mapping):
            raise StandardTrainingError("production seed has no validation best")
        try:
            best = ValidationRecord(
                seed=int(value["seed"]),
                update=int(value["update"]),
                success_rate_under_fixed_step_budget=float(
                    value["success_rate_under_fixed_step_budget"]
                ),
                mean_final_coverage=float(value["mean_final_coverage"]),
                checkpoint_ref=str(value["checkpoint_ref"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StandardTrainingError(
                "production seed validation best drifted"
            ) from exc
        if best.seed != runtime.get("seed"):
            raise StandardTrainingError("production seed validation best mismatch")
        return best

    @_guarded_backend_mutation
    def freeze_global_best(self, record):
        if not isinstance(record, ValidationRecord):
            raise StandardTrainingError("global best record drifted")
        matches = tuple(
            row
            for row in self.receipt_index.verify()
            if row["seed"] == record.seed and row["update"] == record.update
        )
        if len(matches) != 1:
            raise StandardTrainingError("global best checkpoint receipt is missing")
        receipt = matches[0]
        value = {
            "schema_version": "stage6_global_best/v1",
            "record": record.to_dict(),
            "transaction_key": receipt["transaction_key"],
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "complete_marker_sha256": receipt["complete_marker_sha256"],
            "policy_state_sha256": receipt["policy_state_sha256"],
        }
        path = self.stage_root / "global-best.json"
        if path.is_file():
            try:
                payload = path.read_bytes()
                current = json.loads(payload.decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise StandardTrainingError("global best artifact drifted") from exc
            if (
                not isinstance(current, dict)
                or ArtifactStore.canonical_json_bytes(current) != payload
                or current != value
            ):
                raise StandardTrainingError("global best artifact drifted")
        else:
            ArtifactStore(self.stage_root).write_json_exclusive(
                "global-best.json", value
            )
        self._global_best_identity = dict(value)
        self._append_phase_state(
            "global_best_frozen",
            checkpoint_sha256=str(value["checkpoint_sha256"]),
        )
        return record

    @_guarded_backend_mutation
    def record_preflight_phase(self) -> None:
        self._append_phase_state(
            "preflight",
            checkpoint_sha256=self.config.stage5_authority.checkpoint_sha256,
        )

    @_guarded_backend_mutation
    def record_final_evaluation_phase(self, split: str, method: str) -> None:
        from lunar_exploration_ppo.eval.standard import STANDARD_EVALUATION_METHODS

        if split not in {"test", "unseen"} or method not in STANDARD_EVALUATION_METHODS:
            raise StandardTrainingError("final evaluation phase identity drifted")
        if method == "ppo_policy":
            state = "final_test_running" if split == "test" else "final_unseen_running"
        else:
            state = "baselines_running"
        checkpoint_sha256 = getattr(self, "_global_best_identity", {}).get(
            "checkpoint_sha256"
        )
        if not _is_sha256(checkpoint_sha256):
            raise StandardTrainingError("final evaluation phase lacks global best")
        self._append_phase_state(
            state,
            checkpoint_sha256=str(checkpoint_sha256),
        )

    @_guarded_backend_mutation
    def _append_phase_state(self, state: str, *, checkpoint_sha256: str) -> None:
        if self._immutable_bindings is None or state not in _PHASE_STATE_SEQUENCE:
            raise StandardTrainingError("phase state binding is unavailable")
        rows = self.phase_journal.verify()
        states = tuple(row.get("state") for row in rows)
        expected_prefix = _PHASE_STATE_SEQUENCE[: len(states)]
        if states != expected_prefix:
            raise StandardTrainingError("phase state sequence drifted")
        target_index = _PHASE_STATE_SEQUENCE.index(state)
        phase_immutable = self._immutable_bindings
        recovery_capability = getattr(
            self,
            "_planning_child_recovery_capability",
            None,
        )
        source_repair = getattr(self, "_source_repair", None)
        if (
            state == "preflight"
            and recovery_capability is not None
        ):
            epochs = self._journal_binding_epochs()
            if epochs is None:
                raise StandardTrainingError(
                    "planning child recovery phase epoch is missing"
                )
            phase_immutable = dict(epochs[0][1])
        elif (
            state == "preflight"
            and getattr(source_repair, "sensor_acceleration_sha256", None)
            is not None
        ):
            phase_immutable = dict(source_repair.origin_immutable_bindings)
        bindings = {
            **phase_immutable,
            "checkpoint_sha256": checkpoint_sha256,
        }
        if len(states) > target_index:
            if rows[target_index].get("bindings") != bindings:
                raise StandardTrainingError("phase state resume binding drifted")
            return
        if len(states) != target_index:
            raise StandardTrainingError("phase state cannot skip required predecessor")
        self.phase_journal.append(state, bindings)

    def checkpoint_write_count(self):
        return self._checkpoint_writes

    @_guarded_backend_mutation
    def _final_eval_paths(
        self,
        *,
        split: str,
        method: str,
        attempt: int | None = None,
    ) -> tuple[Path, Path, Path]:
        stem = f"final-{split}-{method}"
        if attempt is None:
            root = self.stage_root / "episode-traces"
        else:
            if type(attempt) is not int or attempt <= 0:
                raise StandardTrainingError("final eval attempt identity drifted")
            attempts_root = self.run_root / "stage6-attempts"
            method_slug = {
                "ppo_policy": "ppo",
                "random_valid_frontier": "rnd",
                "nearest_frontier": "near",
                "max_potential_gain_frontier": "gain",
                "gain_over_cost_frontier": "goc",
            }.get(method)
            split_slug = {"test": "t", "unseen": "u"}.get(split)
            if method_slug is None or split_slug is None:
                raise StandardTrainingError("final eval attempt identity drifted")
            attempt_root = attempts_root / (
                f"fe-{split_slug}-{method_slug}-a{attempt:03d}"
            )
            try:
                require_plain_path(
                    attempts_root,
                    base=self.run_root,
                    allow_missing=True,
                    label="Stage 6 attempts root",
                )
                attempts_root.mkdir(parents=False, exist_ok=True)
                require_plain_path(
                    attempts_root,
                    base=self.run_root,
                    leaf_kind="directory",
                    label="Stage 6 attempts root",
                )
                require_plain_path(
                    attempt_root,
                    base=attempts_root,
                    allow_missing=True,
                    label="final eval attempt root",
                )
                attempt_root.mkdir(parents=False, exist_ok=True)
                root = require_plain_path(
                    attempt_root,
                    base=attempts_root,
                    leaf_kind="directory",
                    label="final eval attempt root",
                )
            except (OSError, PathSecurityError) as exc:
                raise StandardTrainingError(
                    "final eval attempt path contains a link or reparse point"
                ) from exc
        trace = root / f"{stem}.jsonl"
        return trace, trace.with_suffix(".summary.json"), trace.with_suffix(".commit.json")

    @staticmethod
    def _artifact_identity(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "path": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    def _final_eval_commit_value(
        self,
        *,
        trace_path: Path,
        summary_path: Path,
        split: str,
        method: str,
        attempt: int,
    ) -> dict[str, object]:
        if not trace_path.is_file() or not summary_path.is_file():
            raise StandardTrainingError("final eval publish pair is incomplete")
        return {
            "schema_version": "stage6_final_eval_commit/v1",
            "split": split,
            "method": method,
            "attempt": attempt,
            "config_sha256": self.config_sha256,
            "checkpoint_sha256": self._global_best_identity["checkpoint_sha256"],
            "policy_state_sha256": self._global_best_identity["policy_state_sha256"],
            "trace": self._artifact_identity(trace_path),
            "summary": self._artifact_identity(summary_path),
        }

    def _verify_final_eval_commit(
        self,
        *,
        trace_path: Path,
        summary_path: Path,
        commit_path: Path,
        record: ValidationRecord,
        split: str,
        method: str,
        expected_attempt: int | None = None,
    ) -> dict[str, object]:
        try:
            payload = commit_path.read_bytes()
            value = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StandardTrainingError("final eval commit drifted") from exc
        if (
            not isinstance(value, dict)
            or ArtifactStore.canonical_json_bytes(value) != payload
            or set(value)
            != {
                "schema_version",
                "split",
                "method",
                "attempt",
                "config_sha256",
                "checkpoint_sha256",
                "policy_state_sha256",
                "trace",
                "summary",
            }
            or value.get("schema_version") != "stage6_final_eval_commit/v1"
            or value.get("split") != split
            or value.get("method") != method
            or type(value.get("attempt")) is not int
            or int(value["attempt"]) <= 0
            or (
                expected_attempt is not None
                and value.get("attempt") != expected_attempt
            )
        ):
            raise StandardTrainingError("final eval commit identity drifted")
        expected = self._final_eval_commit_value(
            trace_path=trace_path,
            summary_path=summary_path,
            split=split,
            method=method,
            attempt=int(value["attempt"]),
        )
        if value != expected:
            raise StandardTrainingError("final eval commit binding drifted")
        return self._verify_final_eval_summary(
            summary_path=summary_path,
            trace_path=trace_path,
            record=record,
            split=split,
            method=method,
        )

    @_guarded_backend_mutation
    def _publish_final_eval_attempt(
        self,
        *,
        attempt_trace: Path,
        attempt_summary: Path,
        attempt_commit: Path,
        canonical_trace: Path,
        canonical_summary: Path,
        canonical_commit: Path,
        record: ValidationRecord,
        split: str,
        method: str,
        attempt: int,
    ) -> dict[str, object]:
        replay = self._call("verify_standard_final_evaluation_artifacts")(
            trace_path=attempt_trace,
            summary_path=attempt_summary,
            commit_path=attempt_commit,
            split=split,
            method=method,
            config_sha256=self.config_sha256,
            checkpoint_sha256=str(
                self._global_best_identity["checkpoint_sha256"]
            ),
            policy_state_sha256=str(
                self._global_best_identity["policy_state_sha256"]
            ),
            bootstrap_resamples=self.config.evaluation.bootstrap_resamples,
            bootstrap_seed=self.config.evaluation.bootstrap_seed,
        )
        if not isinstance(replay, Mapping) or not isinstance(
            replay.get("result"), Mapping
        ):
            raise StandardTrainingError("final eval attempt replay drifted")
        result = dict(replay["result"])
        trace_bytes = attempt_trace.read_bytes()
        summary_bytes = attempt_summary.read_bytes()
        try:
            canonical_parent = require_plain_path(
                canonical_trace.parent,
                base=self.stage_root,
                allow_missing=True,
                label="final eval canonical parent",
            )
            canonical_parent.mkdir(parents=False, exist_ok=True)
            canonical_parent = require_plain_path(
                canonical_parent,
                base=self.stage_root,
                leaf_kind="directory",
                label="final eval canonical parent",
            )
            for candidate, label in (
                (canonical_trace, "final eval canonical trace"),
                (canonical_summary, "final eval canonical summary"),
                (canonical_commit, "final eval canonical commit"),
            ):
                require_plain_path(
                    candidate,
                    base=canonical_parent,
                    allow_missing=True,
                    label=label,
                )
        except (OSError, PathSecurityError) as exc:
            raise StandardTrainingError(
                "final eval canonical path contains a link or reparse point"
            ) from exc
        store = ArtifactStore(canonical_parent)

        def publish_exact(path: Path, payload: bytes) -> None:
            if path.exists():
                if not path.is_file() or path.read_bytes() != payload:
                    raise StandardTrainingError("final eval canonical artifact drifted")
                return
            store.write_bytes_exclusive(path.name, payload)

        publish_exact(canonical_trace, trace_bytes)
        publish_exact(canonical_summary, summary_bytes)
        commit_value = self._final_eval_commit_value(
            trace_path=canonical_trace,
            summary_path=canonical_summary,
            split=split,
            method=method,
            attempt=attempt,
        )
        publish_exact(
            canonical_commit,
            ArtifactStore.canonical_json_bytes(commit_value),
        )
        committed = self._verify_final_eval_commit(
            trace_path=canonical_trace,
            summary_path=canonical_summary,
            commit_path=canonical_commit,
            record=record,
            split=split,
            method=method,
            expected_attempt=attempt,
        )
        canonical_replay = self._call(
            "verify_standard_final_evaluation_artifacts"
        )(
            trace_path=canonical_trace,
            summary_path=canonical_summary,
            commit_path=canonical_commit,
            split=split,
            method=method,
            config_sha256=self.config_sha256,
            checkpoint_sha256=str(
                self._global_best_identity["checkpoint_sha256"]
            ),
            policy_state_sha256=str(
                self._global_best_identity["policy_state_sha256"]
            ),
            bootstrap_resamples=self.config.evaluation.bootstrap_resamples,
            bootstrap_seed=self.config.evaluation.bootstrap_seed,
        )
        if (
            committed != result
            or not isinstance(canonical_replay, Mapping)
            or canonical_replay.get("result") != result
        ):
            raise StandardTrainingError("final eval publication result drifted")
        return committed

    @_guarded_backend_mutation
    def run_final_evaluation(self, record, split, method):
        from lunar_exploration_ppo.eval.standard import STANDARD_EVALUATION_METHODS

        if (
            not isinstance(record, ValidationRecord)
            or split not in {"test", "unseen"}
            or method not in STANDARD_EVALUATION_METHODS
            or getattr(self, "_global_best_identity", {}).get("record")
            != record.to_dict()
        ):
            raise StandardTrainingError("production final eval identity drifted")

        def assert_capability(boundary: str) -> None:
            self._require_execution_capability_current(
                f"run_final_evaluation:{boundary}"
            )

        self.record_final_evaluation_phase(split, method)
        final_runtime = self._load_final_runtime(record)
        trace, summary_path, commit_path = self._final_eval_paths(
            split=split,
            method=method,
        )
        resource_key = f"final:{split}:{method}"
        resource_identity = {
            "kind": "final_evaluation",
            "transaction_key": resource_key,
            "split": split,
            "method": method,
        }
        accepted_resource = self._accepted_resource_attempt(resource_identity)
        accepted_resources = (() if accepted_resource is None else (accepted_resource,))
        accepted_attempt = (
            accepted_resources[0].get("attempt") if accepted_resources else None
        )
        if accepted_resources and (
            type(accepted_attempt) is not int or accepted_attempt <= 0
        ):
            raise StandardTrainingError("final eval resource attempt drifted")
        committed = commit_path.is_file()
        if committed and len(accepted_resources) != 1:
            raise StandardTrainingError("committed final eval lacks resource acceptance")
        if not committed and any(path.exists() for path in (trace, summary_path)):
            if len(accepted_resources) != 1:
                raise StandardTrainingError("unbound final eval canonical artifact drifted")

        resource_attempt: int | None = None
        pre_resource: dict[str, object] | None = None
        resource_latch: object | None = None
        assert_resource: Callable[[str], None] | None = None
        attempt_trace: Path | None = None
        attempt_summary: Path | None = None
        attempt_commit: Path | None = None
        if committed:
            assert isinstance(accepted_attempt, int)
            resource_attempt = accepted_attempt
        elif accepted_resources:
            assert isinstance(accepted_attempt, int)
            resource_attempt = accepted_attempt
            attempt_trace, attempt_summary, attempt_commit = self._final_eval_paths(
                split=split,
                method=method,
                attempt=resource_attempt,
            )
        else:
            resource_attempt = self._next_resource_attempt(resource_key)
            attempt_trace, attempt_summary, attempt_commit = self._final_eval_paths(
                split=split,
                method=method,
                attempt=resource_attempt,
            )
            resource_latch = self._new_runtime_resource_latch(
                transaction_key=resource_key,
                attempt=resource_attempt,
            )

            def assert_resource(boundary: str) -> None:
                assert resource_latch is not None
                assert_capability(boundary)
                self._poll_runtime_resource_latch(resource_latch, boundary)

            pre_resource = self._poll_runtime_resource_latch(
                resource_latch,
                "final-evaluation:pre",
            )
            self._append_resource_phase(
                identity=resource_identity,
                attempt=resource_attempt,
                phase="pre",
                resource=pre_resource,
            )
            if pre_resource["passed"] is not True:
                raise StandardTrainingError("final eval pre-resource hard stop")

        def evaluator():
            assert_capability("evaluator entry")
            if committed:
                self._verify_final_eval_commit(
                    trace_path=trace,
                    summary_path=summary_path,
                    commit_path=commit_path,
                    record=record,
                    split=split,
                    method=method,
                    expected_attempt=resource_attempt,
                )
                replay = self._call(
                    "verify_standard_final_evaluation_artifacts"
                )(
                    trace_path=trace,
                    summary_path=summary_path,
                    commit_path=commit_path,
                    split=split,
                    method=method,
                    config_sha256=self.config_sha256,
                    checkpoint_sha256=str(
                        self._global_best_identity["checkpoint_sha256"]
                    ),
                    policy_state_sha256=str(
                        self._global_best_identity["policy_state_sha256"]
                    ),
                    bootstrap_resamples=self.config.evaluation.bootstrap_resamples,
                    bootstrap_seed=self.config.evaluation.bootstrap_seed,
                )
                if not isinstance(replay, Mapping) or not isinstance(
                    replay.get("result"), Mapping
                ):
                    raise StandardTrainingError("final eval commit replay drifted")
                assert_capability("committed replay return")
                return dict(replay["result"])
            if accepted_resources:
                assert (
                    attempt_trace is not None
                    and attempt_summary is not None
                    and attempt_commit is not None
                )
                replay = self._call(
                    "verify_standard_final_evaluation_artifacts"
                )(
                    trace_path=attempt_trace,
                    summary_path=attempt_summary,
                    commit_path=attempt_commit,
                    split=split,
                    method=method,
                    config_sha256=self.config_sha256,
                    checkpoint_sha256=str(
                        self._global_best_identity["checkpoint_sha256"]
                    ),
                    policy_state_sha256=str(
                        self._global_best_identity["policy_state_sha256"]
                    ),
                    bootstrap_resamples=self.config.evaluation.bootstrap_resamples,
                    bootstrap_seed=self.config.evaluation.bootstrap_seed,
                )
                if not isinstance(replay, Mapping) or not isinstance(
                    replay.get("result"), Mapping
                ):
                    raise StandardTrainingError("final eval attempt replay drifted")
                assert_capability("accepted attempt replay return")
                return dict(replay["result"])
            assert attempt_trace is not None
            assert assert_resource is not None
            result = self._call("run_standard_evaluation")(
                catalog=self.catalog,
                split=split,
                method=method,
                episode_count=64,
                evaluation_seed_start=self.config.evaluation.bootstrap_seed,
                policy=(
                    final_runtime["policy"] if method == "ppo_policy" else None
                ),
                policy_device=self.config.device,
                bootstrap_resamples=self.config.evaluation.bootstrap_resamples,
                bootstrap_seed=self.config.evaluation.bootstrap_seed,
                trace_path=attempt_trace,
                safety_contract=self._safety_contract(),
                config_sha256=self.config_sha256,
                **self._coverage_cache_binding(),
                resource_guard=assert_resource,
            )
            assert_capability("evaluation result conversion")
            return self._evaluation_result_record(result)

        result, isolation = self._call("run_eval_only_transaction")(
            evaluator=evaluator,
            policy=final_runtime["policy"],
            optimizer=final_runtime["optimizer"],
            normalization_stats=final_runtime["loaded"].normalization_stats,
            runtime_state_provider=lambda: tuple(
                final_runtime["loaded"].vector_env_states
            ),
            binding_state_provider=lambda: {
                "config_bytes": (self.stage_root / "config.json").read_bytes(),
                "checkpoint_bytes": self._checkpoint_binding_bytes(
                    Path(final_runtime["checkpoint_root"]),
                    record.update,
                ),
            },
        )
        if not isinstance(result, Mapping) or result.get("episode_count") != 64:
            raise StandardTrainingError("final eval result drifted")
        if not committed and not accepted_resources:
            assert (
                attempt_trace is not None
                and attempt_summary is not None
                and attempt_commit is not None
            )
            if not attempt_trace.is_file():
                raise StandardTrainingError("final eval trace is missing")
            trace_bytes = attempt_trace.read_bytes()
            if len(trace_bytes.splitlines()) != 64:
                raise StandardTrainingError("final eval trace count drifted")
            assert resource_attempt is not None and pre_resource is not None
            value = {
                "schema_version": "stage6_final_eval_completion/v1",
                "split": split,
                "method": method,
                "episode_count": 64,
                "trace": {
                    "path": attempt_trace.name,
                    "sha256": hashlib.sha256(trace_bytes).hexdigest(),
                    "size_bytes": len(trace_bytes),
                },
                "config_sha256": self.config_sha256,
                "checkpoint_sha256": self._global_best_identity[
                    "checkpoint_sha256"
                ],
                "policy_state_sha256": self._global_best_identity[
                    "policy_state_sha256"
                ],
                "result": dict(result),
            }
            assert_capability("before attempt summary publication")
            ArtifactStore(attempt_summary.parent).write_json_exclusive(
                attempt_summary.name,
                value,
            )
            verified_attempt = self._verify_final_eval_summary(
                summary_path=attempt_summary,
                trace_path=attempt_trace,
                record=record,
                split=split,
                method=method,
            )
            if verified_attempt != dict(result):
                raise StandardTrainingError("final eval attempt verification drifted")
            assert_capability("before attempt commit publication")
            ArtifactStore(attempt_commit.parent).write_json_exclusive(
                attempt_commit.name,
                self._final_eval_commit_value(
                    trace_path=attempt_trace,
                    summary_path=attempt_summary,
                    split=split,
                    method=method,
                    attempt=resource_attempt,
                ),
            )
            replay = self._call("verify_standard_final_evaluation_artifacts")(
                trace_path=attempt_trace,
                summary_path=attempt_summary,
                commit_path=attempt_commit,
                split=split,
                method=method,
                config_sha256=self.config_sha256,
                checkpoint_sha256=str(
                    self._global_best_identity["checkpoint_sha256"]
                ),
                policy_state_sha256=str(
                    self._global_best_identity["policy_state_sha256"]
                ),
                bootstrap_resamples=self.config.evaluation.bootstrap_resamples,
                bootstrap_seed=self.config.evaluation.bootstrap_seed,
            )
            if (
                not isinstance(replay, Mapping)
                or replay.get("result") != dict(result)
            ):
                raise StandardTrainingError("final eval attempt replay drifted")
            assert resource_latch is not None
            post_resource = self._poll_runtime_resource_latch(
                resource_latch,
                "final-evaluation:post-attempt-publication",
            )
            self._append_resource_phase(
                identity=resource_identity,
                attempt=resource_attempt,
                phase="post",
                resource=post_resource,
            )
            if post_resource["passed"] is not True:
                raise StandardTrainingError("final eval post-resource hard stop")
            self._append_resource_acceptance(
                identity=resource_identity,
                attempt=resource_attempt,
                pre=pre_resource,
                post=post_resource,
            )
        if not committed:
            assert (
                resource_attempt is not None
                and attempt_trace is not None
                and attempt_summary is not None
                and attempt_commit is not None
            )
            assert_capability("before canonical publication")
            result = self._publish_final_eval_attempt(
                attempt_trace=attempt_trace,
                attempt_summary=attempt_summary,
                attempt_commit=attempt_commit,
                canonical_trace=trace,
                canonical_summary=summary_path,
                canonical_commit=commit_path,
                record=record,
                split=split,
                method=method,
                attempt=resource_attempt,
            )
            assert_capability("after canonical publication")
        if not hasattr(self, "_final_eval_audits"):
            self._final_eval_audits = {}
        self._final_eval_audits[f"{split}:{method}"] = dict(isolation)
        return dict(result)

    @_guarded_backend_mutation
    def _load_final_runtime(self, record: ValidationRecord) -> dict[str, object]:
        existing = getattr(self, "_final_runtime", None)
        if isinstance(existing, dict):
            if existing.get("record") != record:
                raise StandardTrainingError("global best runtime record drifted")
            return existing
        policy = self._call("load_stage4_policy_for_standard")(
            checkpoint_path=self._components["stage4_checkpoint_path"],
            checkpoint_sha256=self.config.stage5_authority.checkpoint_sha256,
            policy_state_sha256=self.config.stage5_authority.policy_state_sha256,
            device=self.config.device,
        )
        if not isinstance(policy, nn.Module):
            raise StandardTrainingError("global best policy load drifted")
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=self.config.ppo.learning_rate,
            eps=self.config.ppo.adam_eps,
            weight_decay=self.config.ppo.weight_decay,
        )
        checkpoint_root = (
            self.stage_root / f"checkpoints/seed-{record.seed}"
        ).resolve()
        manager = self._call("CheckpointManager")(
            checkpoint_root,
            schema_version=self.config.checkpoint.schema_version,
        )
        loaded = manager.load_complete(
            update_step=record.update,
            policy=policy,
            optimizer=optimizer,
            expected_config_sha256=self.config_sha256,
            expected_lineage=self._checkpoint_lineage(record.update),
            expected_safety_contract=self._safety_contract(),
        )
        if (
            loaded.checkpoint_sha256
            != self._global_best_identity["checkpoint_sha256"]
            or loaded.policy_state_sha256
            != self._global_best_identity["policy_state_sha256"]
        ):
            raise StandardTrainingError("global best checkpoint load drifted")
        self._final_runtime = {
            "record": record,
            "policy": policy,
            "optimizer": optimizer,
            "manager": manager,
            "loaded": loaded,
            "checkpoint_root": checkpoint_root,
        }
        return self._final_runtime

    def _verify_final_eval_summary(
        self,
        *,
        summary_path: Path,
        trace_path: Path,
        record: ValidationRecord,
        split: str,
        method: str,
    ) -> dict[str, object]:
        try:
            payload = summary_path.read_bytes()
            value = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StandardTrainingError("final eval summary drifted") from exc
        expected = {
            "schema_version",
            "split",
            "method",
            "episode_count",
            "trace",
            "config_sha256",
            "checkpoint_sha256",
            "policy_state_sha256",
            "result",
        }
        trace_bytes = trace_path.read_bytes()
        trace_record = value.get("trace") if isinstance(value, dict) else None
        if (
            not isinstance(value, dict)
            or ArtifactStore.canonical_json_bytes(value) != payload
            or set(value) != expected
            or value["schema_version"] != "stage6_final_eval_completion/v1"
            or value["split"] != split
            or value["method"] != method
            or value["episode_count"] != 64
            or not isinstance(trace_record, dict)
            or trace_record
            != {
                "path": trace_path.name,
                "sha256": hashlib.sha256(trace_bytes).hexdigest(),
                "size_bytes": len(trace_bytes),
            }
            or len(trace_bytes.splitlines()) != 64
            or value["config_sha256"] != self.config_sha256
            or value["checkpoint_sha256"]
            != self._global_best_identity["checkpoint_sha256"]
            or value["policy_state_sha256"]
            != self._global_best_identity["policy_state_sha256"]
            or not isinstance(value["result"], dict)
            or value["result"].get("episode_count") != 64
            or record.to_dict() != self._global_best_identity["record"]
        ):
            raise StandardTrainingError("final eval completion drifted")
        return dict(value["result"])

    def _evaluation_result_record(self, result: object) -> dict[str, object]:
        if isinstance(result, Mapping):
            value = _jsonable_runtime_value(result)
            if isinstance(value, dict):
                count = value.get("episode_count")
                if count is None and isinstance(value.get("metrics"), dict):
                    count = value["metrics"].get("episode_count")
                value["episode_count"] = count
                if count == 64:
                    self._validated_evaluation_safety_contract(value)
                    return value
        metrics = getattr(result, "metrics", None)
        if isinstance(metrics, Mapping) and metrics.get("episode_count") == 64:
            value = {
                "episode_count": 64,
                "metrics": _jsonable_runtime_value(metrics),
                "bootstrap_audit": _jsonable_runtime_value(
                    getattr(result, "bootstrap_audit", {})
                ),
                "fairness_audit": _jsonable_runtime_value(
                    getattr(result, "fairness_audit", {})
                ),
            }
            self._validated_evaluation_safety_contract(value)
            return value
        raise StandardTrainingError("final eval summary result drifted")

    @_guarded_backend_mutation
    def finalize(self, record, evaluations):
        if (
            getattr(
                self,
                "_planning_child_recovery_capability",
                None,
            )
            is None
        ):
            self._require_source_repair_current("finalize")
        from lunar_exploration_ppo.eval.standard import STANDARD_EVALUATION_METHODS
        from lunar_exploration_ppo.workflows.stage6 import (
            Stage6WorkflowError,
            decide_performance_advantage,
            verify_stage6_machine_acceptance,
            verify_stage6_pending_acceptance,
        )

        expected_keys = {
            f"{split}:{method}"
            for split in ("test", "unseen")
            for method in STANDARD_EVALUATION_METHODS
        }
        if (
            not isinstance(record, ValidationRecord)
            or getattr(self, "_global_best_identity", {}).get("record")
            != record.to_dict()
            or not isinstance(evaluations, Mapping)
            or set(evaluations) != expected_keys
            or self._immutable_bindings is None
        ):
            raise StandardTrainingError("production finalization identity drifted")
        if any(
            (self.stage_root / name).exists()
            for name in ("review.json", "approval.json", "gate.json")
        ):
            raise StandardTrainingError("production root contains authority artifact")

        ordered_evaluations: list[dict[str, object]] = []
        verified_evaluations: dict[str, dict[str, object]] = {}
        for split in ("test", "unseen"):
            for method in STANDARD_EVALUATION_METHODS:
                key = f"{split}:{method}"
                value = self._evaluation_result_record(evaluations[key])
                trace = self.stage_root / f"episode-traces/final-{split}-{method}.jsonl"
                replay = verify_standard_final_evaluation_artifacts(
                    trace_path=trace,
                    summary_path=trace.with_suffix(".summary.json"),
                    commit_path=trace.with_suffix(".commit.json"),
                    split=split,
                    method=method,
                    config_sha256=self.config_sha256,
                    safety_contract=self._safety_contract(),
                    checkpoint_sha256=str(
                        self._global_best_identity["checkpoint_sha256"]
                    ),
                    policy_state_sha256=str(
                        self._global_best_identity["policy_state_sha256"]
                    ),
                    bootstrap_resamples=self.config.evaluation.bootstrap_resamples,
                    bootstrap_seed=self.config.evaluation.bootstrap_seed,
                )
                if replay["result"] != value:
                    raise StandardTrainingError("final evaluation summary/trace drifted")
                verified_evaluations[key] = replay
                ordered_evaluations.append(
                    {
                        "transaction_key": f"final:{key}",
                        "kind": "final_evaluation",
                        "split": split,
                        "method": method,
                        "result": value,
                    }
                )
        isolation = getattr(self, "_final_eval_audits", None)
        if (
            not isinstance(isolation, Mapping)
            or set(isolation) != expected_keys
            or any(
                not isinstance(value, Mapping) or value.get("passed") is not True
                for value in isolation.values()
            )
        ):
            raise StandardTrainingError("final evaluation isolation audit drifted")
        aggregate = build_standard_final_aggregate_artifacts(
            verified_evaluations,
            isolation,  # type: ignore[arg-type]
            safety_contract=self._safety_contract(),
            config_sha256=self.config_sha256,
        )

        def primary_ci(key: str, bound: str) -> float:
            value = next(
                row["result"]
                for row in ordered_evaluations
                if f"{row['split']}:{row['method']}" == key
            )
            if not isinstance(value, Mapping):
                raise StandardTrainingError("final bootstrap result drifted")
            bootstrap = value.get("bootstrap_audit")
            metric = (
                bootstrap.get("metrics", {}).get(
                    "success_rate_under_fixed_step_budget", {}
                )
                if isinstance(bootstrap, Mapping)
                else {}
            )
            candidate = metric.get(bound) if isinstance(metric, Mapping) else None
            if (
                not isinstance(candidate, (int, float))
                or not math.isfinite(float(candidate))
            ):
                raise StandardTrainingError("final bootstrap CI drifted")
            return float(candidate)

        try:
            performance = decide_performance_advantage(
                ppo_ci95_low=min(
                    primary_ci(f"{split}:ppo_policy", "ci95_low")
                    for split in ("test", "unseen")
                ),
                gain_over_cost_ci95_high=max(
                    primary_ci(
                        f"{split}:gain_over_cost_frontier", "ci95_high"
                    )
                    for split in ("test", "unseen")
                ),
            )
        except Stage6WorkflowError as exc:
            raise StandardTrainingError("final performance CI decision failed") from exc

        receipts = self.receipt_index.verify()
        global_identity = dict(self._global_best_identity)
        matching_receipts = tuple(
            row
            for row in receipts
            if row["transaction_key"] == global_identity["transaction_key"]
            and row["checkpoint_sha256"] == global_identity["checkpoint_sha256"]
        )
        if len(matching_receipts) != 1:
            raise StandardTrainingError("global best receipt finalization drifted")

        store = ArtifactStore(self.stage_root)

        def write_exact(relative_path: str, payload: bytes) -> None:
            path = store.resolve(relative_path)
            if path.exists():
                if not path.is_file() or path.read_bytes() != payload:
                    raise StandardTrainingError(
                        f"final artifact drifted: {relative_path}"
                    )
                return
            store.write_bytes_exclusive(relative_path, payload)

        def write_json(relative_path: str, value: object) -> None:
            write_exact(relative_path, ArtifactStore.canonical_json_bytes(value))

        write_exact("config.json", self.config_bytes)
        lineage_path = self.stage_root / "lineage_audit.json"
        if not lineage_path.exists():
            raise StandardTrainingError(
                "Stage 6 verified review lineage is missing"
            )
        for name in (
            "job-state.jsonl",
            "training_metrics.jsonl",
            "validation_metrics.jsonl",
            "resource_audit.jsonl",
            "math_audit.jsonl",
            "checkpoint_audit.jsonl",
        ):
            path = self.stage_root / name
            if not path.exists():
                store.write_bytes_exclusive(name, b"")

        for row in ordered_evaluations:
            append_transaction_metric_once(self.stage_root / "metrics.jsonl", row)

        scenario_audit = {
            "schema_version": "stage6_scenario_split_audit/v1",
            "catalog_sha256": str(self.catalog.sha256),
            "spatial_audit": _jsonable_runtime_value(self.catalog.spatial_audit()),
        }
        checkpoint_index_path = self.receipt_index.path
        checkpoint_index_bytes = checkpoint_index_path.read_bytes()
        acceptance_profile = None
        if self._planning_warm_start is not None:
            from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
                build_planning_child_acceptance_profile,
            )

            acceptance_profile = build_planning_child_acceptance_profile()
        checkpoint_manifest = {
            "schema_version": (
                "stage6_checkpoint_manifest/v2"
                if acceptance_profile is not None
                else "stage6_checkpoint_manifest/v1"
            ),
            "receipt_count": len(receipts),
            "receipt_index": {
                "path": "checkpoints/index.jsonl",
                "sha256": hashlib.sha256(checkpoint_index_bytes).hexdigest(),
                "size_bytes": len(checkpoint_index_bytes),
            },
            "global_best": global_identity,
            "receipts": list(receipts),
        }
        if acceptance_profile is not None:
            checkpoint_manifest["acceptance_profile"] = acceptance_profile
        write_json("scenario_split_audit.json", scenario_audit)
        write_json("standard_checkpoint_manifest.json", checkpoint_manifest)
        write_json("fairness_audit.json", aggregate["fairness_audit"])
        write_json("leakage_audit.json", aggregate["leakage_audit"])
        write_json("failure_audit.json", aggregate["failure_audit"])
        write_exact(
            "standard_baseline_comparison.csv",
            aggregate["comparison_csv"],  # type: ignore[arg-type]
        )
        write_exact(
            "standard_coverage_curves.csv",
            aggregate["coverage_curves_csv"],  # type: ignore[arg-type]
        )

        acceptance_artifacts = build_stage6_acceptance_artifacts(
            global_best=record.to_dict(),
            performance_advantage_established=performance.established,
            performance_claim=performance.claim,
            ppo_ci95_low=performance.ppo_ci95_low,
            gain_over_cost_ci95_high=performance.gain_over_cost_ci95_high,
            final_evaluation_count=len(ordered_evaluations),
            final_episode_count=sum(
                int(row["result"]["episode_count"])  # type: ignore[index]
                for row in ordered_evaluations
            ),
            checkpoint_receipt_count=len(receipts),
            immutable_bindings=self._immutable_bindings,
            source_repair_binding=(
                getattr(self, "_source_repair").summary_binding
                if getattr(self, "_source_repair", None) is not None
                else None
            ),
            acceptance_profile=acceptance_profile,
            planning_child_source_repair_binding=(
                getattr(
                    self,
                    "_planning_child_recovery_capability",
                ).evidence_binding()
                if getattr(
                    self,
                    "_planning_child_recovery_capability",
                    None,
                )
                is not None
                else None
            ),
        )
        summary = dict(acceptance_artifacts["summary"])  # type: ignore[arg-type]
        routing = dict(acceptance_artifacts["routing"])  # type: ignore[arg-type]
        reports = acceptance_artifacts["reports"]
        if not isinstance(reports, Mapping):
            raise StandardTrainingError("Stage 6 acceptance reports drifted")
        terminal_artifacts = {
            "summary.json": ArtifactStore.canonical_json_bytes(summary),
            "routing.json": ArtifactStore.canonical_json_bytes(routing),
            **dict(reports),
        }
        phase_states = tuple(
            row.get("state") for row in self.phase_journal.verify()
        )
        try:
            if (self.stage_root / "manifest.json").exists():
                completed = verify_stage6_machine_acceptance(
                    stage_root=self.stage_root,
                    repo_root=self.repo_root,
                    planning_child_recovery_capability=getattr(
                        self,
                        "_planning_child_recovery_capability",
                        None,
                    ),
                )
                if completed.get("passed") is not True:
                    raise Stage6WorkflowError(
                        "Stage 6 completed machine acceptance did not pass"
                    )
            elif phase_states == _PHASE_STATE_SEQUENCE[:5]:
                pending_acceptance = verify_stage6_pending_acceptance(
                    stage_root=self.stage_root,
                    repo_root=self.repo_root,
                    summary=summary,
                    routing=routing,
                    reports=reports,  # type: ignore[arg-type]
                    planning_child_recovery_capability=getattr(
                        self,
                        "_planning_child_recovery_capability",
                        None,
                    ),
                )
                if pending_acceptance.get("passed") is not True:
                    raise Stage6WorkflowError(
                        "Stage 6 pending machine acceptance did not pass"
                    )
                from lunar_exploration_ppo.workflows.stage6_terminal_recovery import (
                    TerminalRecoveryError,
                    write_stage6_preterminal_acceptance,
                )

                try:
                    self._preterminal_acceptance_identity = (
                        write_stage6_preterminal_acceptance(
                            stage_root=self.stage_root,
                            semantic_result=pending_acceptance,
                            immutable_bindings=self._immutable_bindings,
                            global_checkpoint_identity=global_identity,
                            terminal_artifacts=terminal_artifacts,
                            execution_capability=self._execution_capability,
                        )
                    )
                except TerminalRecoveryError as exc:
                    raise Stage6WorkflowError(
                        "Stage 6 preterminal receipt commit failed"
                    ) from exc
            elif phase_states in (
                _PHASE_STATE_SEQUENCE[:6],
                _PHASE_STATE_SEQUENCE[:7],
            ):
                for relative_path, payload in terminal_artifacts.items():
                    path = self.stage_root / relative_path
                    if not path.is_file() or path.read_bytes() != payload:
                        raise Stage6WorkflowError(
                            "Stage 6 partial terminal artifact drifted"
                        )
            else:
                raise Stage6WorkflowError("Stage 6 finalization phase prefix drifted")
        except Stage6WorkflowError as exc:
            raise StandardTrainingError(
                "pending Stage 6 machine verification failed"
            ) from exc
        self._pending_terminal_artifacts = terminal_artifacts
        return {
            "stage_root": str(self.stage_root.resolve()),
            "summary": summary,
            "routing": routing,
        }

    @_guarded_backend_mutation
    def commit_terminal_acceptance(self, pending: Mapping[str, object]) -> dict[str, object]:
        """Commit success artifacts only after durable terminal resource acceptance."""

        if (
            getattr(
                self,
                "_planning_child_recovery_capability",
                None,
            )
            is None
        ):
            self._require_source_repair_current(
                "commit_terminal_acceptance"
            )
        from lunar_exploration_ppo.workflows.stage6 import (
            Stage6WorkflowError,
            write_or_verify_stage6_manifest,
        )
        from lunar_exploration_ppo.workflows.stage6_terminal_recovery import (
            TerminalRecoveryError,
            recover_stage6_terminal_commit,
        )

        artifacts = getattr(self, "_pending_terminal_artifacts", None)
        if (
            not isinstance(pending, Mapping)
            or set(pending) != {"stage_root", "summary", "routing"}
            or pending.get("stage_root") != str(self.stage_root.resolve())
            or not isinstance(pending.get("summary"), Mapping)
            or not isinstance(pending.get("routing"), Mapping)
            or not isinstance(artifacts, Mapping)
            or set(artifacts)
            != {
                "summary.json",
                "routing.json",
                "standard_training_report.md",
                "standard_eval_report.md",
                "report.md",
            }
            or artifacts.get("summary.json")
            != ArtifactStore.canonical_json_bytes(dict(pending["summary"]))
            or artifacts.get("routing.json")
            != ArtifactStore.canonical_json_bytes(dict(pending["routing"]))
        ):
            raise StandardTrainingError("terminal acceptance payload drifted")
        try:
            resource_acceptance = validate_stage6_resource_audit(
                self._resource_audit_rows(),
                self.transactions,
            )
        except StandardTrainingError:
            raise
        if resource_acceptance.get("terminal_evidence_present") is not True:
            raise StandardTrainingError("terminal resource acceptance is missing")
        try:
            recovery = recover_stage6_terminal_commit(
                stage_root=self.stage_root,
                manifest_committer=lambda stage: write_or_verify_stage6_manifest(
                    stage,
                    repo_root=self.repo_root,
                    planning_child_recovery_capability=getattr(
                        self,
                        "_planning_child_recovery_capability",
                        None,
                    ),
                ),
                execution_capability=self._execution_capability,
                planning_child_recovery_capability=getattr(
                    self,
                    "_planning_child_recovery_capability",
                    None,
                ),
            )
        except (Stage6WorkflowError, TerminalRecoveryError) as exc:
            raise StandardTrainingError(
                "terminal Stage 6 recovery commit failed"
            ) from exc
        receipt_identity = getattr(self, "_preterminal_acceptance_identity", None)
        if (
            isinstance(receipt_identity, Mapping)
            and dict(recovery["receipt_identity"]) != dict(receipt_identity)
        ):
            raise StandardTrainingError("terminal receipt identity drifted")
        for relative_path, payload in artifacts.items():
            if not isinstance(relative_path, str) or not isinstance(payload, bytes):
                raise StandardTrainingError("terminal artifact payload drifted")
            path = self.stage_root / relative_path
            if not path.is_file() or path.read_bytes() != payload:
                raise StandardTrainingError(
                    f"terminal artifact drifted: {relative_path}"
                )
        if hasattr(self, "_final_runtime"):
            del self._final_runtime
        return {
            "stage_root": str(self.stage_root.resolve()),
            "summary": dict(pending["summary"]),
            "routing": dict(pending["routing"]),
        }


@_guarded_standard_training_mutation
def execute_standard_training(
    *,
    config: Stage6Config,
    run_root: str | Path,
    repo_root: str | Path,
    stage5_authority: Mapping[str, object],
    verified_review_authorization: Mapping[str, object],
    execution_capability: object,
    planning_warm_start_context: object | None = None,
    planning_warm_start_sha256: str | None = None,
    planning_child_source_repair_context: object | None = None,
    planning_child_source_repair_sha256: str | None = None,
    planning_child_recovery_capability: (
        PlanningChildRecoveryCapability | None
    ) = None,
    verified_parent_u74: object | None = None,
) -> dict[str, object]:
    """执行冻结的 Standard v1 正式训练、验证与最终评估。"""

    _require_standard_execution_capability(
        execution_capability,
        label="execute_standard_training public entry",
        config=config,
        run_root=run_root,
        repo_root=repo_root,
        stage5_authority=stage5_authority,
        verified_review_authorization=verified_review_authorization,
    )
    result = _execute_standard_training(
        config=config,
        run_root=run_root,
        repo_root=repo_root,
        stage5_authority=stage5_authority,
        verified_review_authorization=verified_review_authorization,
        execution_capability=execution_capability,
        planning_warm_start_context=planning_warm_start_context,
        planning_warm_start_sha256=planning_warm_start_sha256,
        planning_child_source_repair_context=(
            planning_child_source_repair_context
        ),
        planning_child_source_repair_sha256=(
            planning_child_source_repair_sha256
        ),
        planning_child_recovery_capability=(
            planning_child_recovery_capability
        ),
        verified_parent_u74=verified_parent_u74,
    )
    _require_standard_execution_capability(
        execution_capability,
        label="execute_standard_training public exit",
        config=config,
        run_root=run_root,
        repo_root=repo_root,
        stage5_authority=stage5_authority,
        verified_review_authorization=verified_review_authorization,
    )
    return result


def _require_machine_preflight(
    *,
    stage_root: str | Path,
    config_sha256: str,
    stage5_gate_sha256: str,
) -> dict[str, object]:
    """Validate the one canonical machine preflight bound to this execution."""

    from lunar_exploration_ppo.workflows.stage6 import (
        Stage6WorkflowError,
        validate_stage6_machine_preflight_audit,
    )

    try:
        stage = require_plain_path(
            lexical_absolute(stage_root),
            leaf_kind="directory",
            label="Stage 6 stage root",
        )
        root = require_plain_path(
            stage / "preflight",
            base=stage,
            leaf_kind="directory",
            label="Stage 6 preflight root",
        )
        audit_path = require_plain_path(
            root / "audit.json",
            base=stage,
            leaf_kind="file",
            label="Stage 6 preflight audit",
        )
    except (OSError, PathSecurityError) as exc:
        raise StandardTrainingError(
            "Stage 6 machine preflight path contains a link or reparse point"
        ) from exc
    if (
        {path.name for path in root.iterdir()} != {"audit.json"}
    ):
        raise StandardTrainingError("Stage 6 machine preflight artifact drifted")
    payload = audit_path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StandardTrainingError("Stage 6 machine preflight is invalid JSON") from exc
    if not isinstance(value, dict) or ArtifactStore.canonical_json_bytes(value) != payload:
        raise StandardTrainingError("Stage 6 machine preflight is not canonical JSON")
    try:
        audit = validate_stage6_machine_preflight_audit(value)
    except Stage6WorkflowError as exc:
        raise StandardTrainingError("Stage 6 machine preflight validation failed") from exc
    if (
        audit["config_sha256"] != config_sha256
        or audit["stage5_gate_sha256"] != stage5_gate_sha256
    ):
        raise StandardTrainingError("Stage 6 machine preflight binding drifted")
    return audit


def _persist_execution_identity(
    *,
    stage_root: Path,
    config: Stage6Config,
    run_root: str | Path,
    repo_root: str | Path,
    stage5_authority: Mapping[str, object],
    execution_capability: object,
    config_bytes: bytes,
    identity: Mapping[str, object],
    immutable_bindings: Mapping[str, object],
    verified_review_authorization: Mapping[str, object],
    planning_warm_start_context: object | None = None,
    planning_child_recovery_capability: (
        PlanningChildRecoveryCapability | None
    ) = None,
) -> None:
    with _standard_execution_operation(
        execution_capability,
        label="persist execution identity",
        config=config,
        run_root=run_root,
        repo_root=repo_root,
        stage5_authority=stage5_authority,
        verified_review_authorization=verified_review_authorization,
        execution_identity=identity,
    ):
        if lexical_absolute(stage_root) != lexical_absolute(run_root) / "s6":
            raise StandardTrainingError(
                "Stage 6 persisted execution identity root drifted"
            )
        store = ArtifactStore(stage_root)
        if planning_child_recovery_capability is not None:
            from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
                PlanningChildRecoveryCapability,
            )

            if (
                planning_warm_start_context is None
                or not isinstance(
                    planning_child_recovery_capability,
                    PlanningChildRecoveryCapability,
                )
                or planning_child_recovery_capability.stage_root
                != lexical_absolute(stage_root)
                or planning_child_recovery_capability.formal_run_id
                != lexical_absolute(run_root).name
                or planning_child_recovery_capability.acceptance_binding.get(
                    "current_execution_identity_sha256"
                )
                != hashlib.sha256(
                    ArtifactStore.canonical_json_bytes(identity)
                ).hexdigest()
                or planning_child_recovery_capability.acceptance_binding.get(
                    "current_immutable_bindings_sha256"
                )
                != hashlib.sha256(
                    ArtifactStore.canonical_json_bytes(
                        immutable_bindings
                    )
                ).hexdigest()
                or planning_child_recovery_capability.acceptance_binding.get(
                    "current_verified_review_authorization"
                )
                != verified_review_authorization
            ):
                raise StandardTrainingError(
                    "Stage 6 planning child recovery lineage binding is incomplete"
                )
            return
        lineage: dict[str, object] = {
            "schema_version": "stage6_lineage_audit/v2",
            "execution_identity": dict(identity),
            "immutable_bindings": dict(immutable_bindings),
            "verified_review_authorization": dict(
                verified_review_authorization
            ),
        }
        if planning_warm_start_context is not None:
            from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
                PlanningWarmStartContext,
                Stage6PlanningWarmStartError,
            )

            if (
                not isinstance(
                    planning_warm_start_context,
                    PlanningWarmStartContext,
                )
                or planning_warm_start_context.artifact_path is None
                or not _is_sha256(
                    planning_warm_start_context.artifact_sha256
                )
            ):
                raise StandardTrainingError(
                    "Stage 6 persisted warm-start lineage is incomplete"
                )
            try:
                planning_warm_start_context.require_current(
                    "persist execution identity"
                )
            except Stage6PlanningWarmStartError as exc:
                raise StandardTrainingError(
                    "Stage 6 persisted warm-start artifact drifted"
                ) from exc
            artifact_payload = (
                json.dumps(
                    _jsonable_runtime_value(
                        planning_warm_start_context.artifact
                    ),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            if (
                hashlib.sha256(artifact_payload).hexdigest()
                != planning_warm_start_context.artifact_sha256
            ):
                raise StandardTrainingError(
                    "Stage 6 persisted warm-start artifact identity drifted"
                )
            lineage.update(
                {
                    "schema_version": "stage6_lineage_audit/v3",
                    "planning_warm_start": {
                        "artifact": json.loads(
                            artifact_payload.decode("utf-8")
                        ),
                        "artifact_sha256": (
                            planning_warm_start_context.artifact_sha256
                        ),
                        "artifact_size_bytes": len(artifact_payload),
                    },
                }
            )
        lineage_payload = ArtifactStore.canonical_json_bytes(lineage)
        for relative_path, payload in (
            ("config.json", config_bytes),
            ("lineage_audit.json", lineage_payload),
        ):
            path = store.resolve(relative_path)
            if path.exists():
                if not path.is_file() or path.read_bytes() != payload:
                    raise StandardTrainingError(
                        "Stage 6 persisted execution identity drifted"
                    )
                continue
            try:
                store.write_bytes_exclusive(relative_path, payload)
            except FileExistsError:
                if not path.is_file() or path.read_bytes() != payload:
                    raise StandardTrainingError(
                        "Stage 6 persisted execution identity drifted"
                    )


def _append_initial_resource_segment_start(
    *,
    stage_root: Path,
    first_sample: Mapping[str, object],
    planning_child_recovery_capability: (
        PlanningChildRecoveryCapability | None
    ),
) -> dict[str, object]:
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        ResourceLifecycleError,
        append_resource_segment_start,
        validate_resource_lifecycle_ledger,
    )

    resource_path = stage_root / "resource_audit.jsonl"
    if planning_child_recovery_capability is not None:
        try:
            lifecycle = validate_resource_lifecycle_ledger(
                resource_path,
                require_terminal=False,
            )
        except ResourceLifecycleError as exc:
            raise StandardTrainingError(
                "Stage 6 planning child recovery resource lifecycle drifted"
            ) from exc
        active_segment_index = lifecycle.get("active_segment_index")
        cursor = planning_child_recovery_capability.resume_cursor
        if cursor is None:
            acceptance = (
                planning_child_recovery_capability.acceptance_binding
            )
            crash_suffix = acceptance.get("crash_suffix")
            accepted_anchor = acceptance.get("accepted_anchor")
            effective_update = (
                crash_suffix.get("update")
                if isinstance(crash_suffix, Mapping)
                else (
                    accepted_anchor.get("last_accepted_update")
                    if isinstance(accepted_anchor, Mapping)
                    else None
                )
            )
            if (
                planning_child_recovery_capability.terminal_complete
                or acceptance.get("terminal_complete") is not False
                or acceptance.get("resume_cursor") is not None
                or effective_update != 100
            ):
                raise StandardTrainingError(
                    "Stage 6 planning child final transaction "
                    "capability drifted"
                )
            try:
                payload = DurableJsonl(
                    resource_path
                ).recover_and_snapshot()
                rows = tuple(
                    json.loads(line.decode("utf-8"))
                    for line in payload.splitlines(keepends=True)
                )
            except (
                DurableJsonlError,
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                raise StandardTrainingError(
                    "Stage 6 planning child active resource segment "
                    "is unreadable"
                ) from exc
            segment_rows = tuple(
                row
                for row in rows
                if isinstance(row, Mapping)
                and row.get("phase") == "segment_start"
            )
            if (
                not segment_rows
                or segment_rows[-1].get("segment_index")
                != active_segment_index
                or segment_rows[-1].get("segment_id")
                != lifecycle.get("active_segment_id")
                or segment_rows[-1].get("root_pid")
                != lifecycle.get("active_root_pid")
            ):
                raise StandardTrainingError(
                    "Stage 6 planning child active resource segment "
                    "drifted"
                )
            return dict(segment_rows[-1])
        expected_segment_index = (
            cursor.next_resource_segment_index
        )
        if (
            type(active_segment_index) is not int
            or active_segment_index + 1 != expected_segment_index
        ):
            raise StandardTrainingError(
                "Stage 6 planning child recovery resource segment drifted"
            )
    return append_resource_segment_start(
        resource_path,
        first_sample=first_sample,
    )


@_guarded_standard_training_mutation
def _execute_standard_training(
    *,
    config: Stage6Config,
    run_root: str | Path,
    repo_root: str | Path,
    stage5_authority: Mapping[str, object],
    verified_review_authorization: Mapping[str, object],
    execution_capability: object,
    planning_warm_start_context: object | None = None,
    planning_warm_start_sha256: str | None = None,
    planning_child_source_repair_context: object | None = None,
    planning_child_source_repair_sha256: str | None = None,
    planning_child_recovery_capability: (
        PlanningChildRecoveryCapability | None
    ) = None,
    verified_parent_u74: object | None = None,
) -> dict[str, object]:
    _require_standard_execution_capability(
        execution_capability,
        label="_execute_standard_training internal entry before backend imports",
        config=config,
        run_root=run_root,
        repo_root=repo_root,
        stage5_authority=stage5_authority,
        verified_review_authorization=verified_review_authorization,
    )
    from lunar_exploration_ppo.env.standard_training import (
        build_standard_catalog,
        standard_env_specs,
    )
    from lunar_exploration_ppo.eval.standard import (
        STANDARD_EVALUATION_METHODS,
        run_standard_evaluation,
    )
    from lunar_exploration_ppo.ppo.checkpoint import CheckpointManager
    from lunar_exploration_ppo.ppo.collector import RolloutCollector, SpawnVectorEnv
    from lunar_exploration_ppo.ppo.trainer import PPOTrainer
    from lunar_exploration_ppo.utils.resources import (
        ProcessTreeRSSMonitor,
        capture_resource_snapshot,
        evaluate_resource_gates,
    )
    from lunar_exploration_ppo.workflows.stage6 import (
        Stage6WorkflowError,
        load_stage4_policy_for_standard,
        stage6_execution_identity,
        stage6_environment_sha256,
        validate_stage6_environment_identity,
        validate_stage6_verified_review_authorization,
        verify_stage6_machine_acceptance,
        write_stage6_manifest,
    )
    from lunar_exploration_ppo.workflows.stage6_source_repair import (
        Stage6SourceRepairError,
        _FIXED_COVERAGE_CACHE_MANIFEST_BINDING,
        _SENSOR_ACCELERATION_EVIDENCE_BINDINGS,
        load_stage6_source_repair_context,
    )

    if not isinstance(config, Stage6Config):
        raise StandardTrainingError("Stage 6 execution requires frozen config")
    if not isinstance(stage5_authority, Mapping) or not isinstance(
        verified_review_authorization, Mapping
    ):
        raise StandardTrainingError("Stage 6 authority binding is invalid")
    if (
        planning_child_source_repair_context is not None
        or planning_child_source_repair_sha256 is not None
    ):
        raise StandardTrainingError(
            "legacy planning child source-repair context is unsupported"
        )
    if (
        planning_child_recovery_capability is not None
        and planning_warm_start_context is None
    ):
        raise StandardTrainingError(
            "Stage 6 planning child recovery requires planning warm-start"
        )
    try:
        run_candidate = lexical_absolute(run_root)
        require_plain_path(
            run_candidate,
            allow_missing=True,
            label="Stage 6 run root",
        )
    except PathSecurityError as exc:
        raise StandardTrainingError(
            "Stage 6 runtime path contains a link or reparse point"
        ) from exc
    resolved_repo_root = Path(repo_root).expanduser().resolve()
    config_path = (
        resolved_repo_root / "configs/ppo_highres_frontier_stage6_v1.json"
    ).resolve()
    try:
        base_config_bytes = config_path.read_bytes()
    except OSError as exc:
        raise StandardTrainingError("Stage 6 canonical config is unreadable") from exc
    try:
        config_value = json.loads(base_config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StandardTrainingError("Stage 6 canonical config is invalid JSON") from exc
    if ArtifactStore.canonical_json_bytes(config_value) != base_config_bytes:
        raise StandardTrainingError("Stage 6 canonical config is not canonical JSON")
    config_bytes = base_config_bytes
    if planning_warm_start_context is not None:
        from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
            PlanningWarmStartContext,
            Stage6PlanningWarmStartError,
            parse_planning_effective_config_bytes,
        )

        if not isinstance(
            planning_warm_start_context,
            PlanningWarmStartContext,
        ):
            raise StandardTrainingError(
                "Stage 6 planning warm-start context is invalid"
            )
        try:
            effective = parse_planning_effective_config_bytes(
                planning_warm_start_context.child_effective_config_bytes
            )
        except Stage6PlanningWarmStartError as exc:
            raise StandardTrainingError(
                "Stage 6 planning effective config is invalid"
            ) from exc
        if (
            effective.base_config_sha256
            != hashlib.sha256(base_config_bytes).hexdigest()
            or effective.base_config != config
        ):
            raise StandardTrainingError(
                "Stage 6 planning effective/base config drifted"
            )
        config_bytes = effective.effective_config_bytes
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    gate_sha256 = stage5_authority.get("gate_sha256")
    if gate_sha256 != config.stage5_authority.gate_sha256 or not _is_sha256(
        gate_sha256
    ):
        raise StandardTrainingError("Stage 6 authority binding drifted")
    identity = stage6_execution_identity(
        repo_root=resolved_repo_root,
        config_path=config_path,
        effective_config_bytes=(
            config_bytes
            if planning_warm_start_context is not None
            else None
        ),
    )
    _require_standard_execution_capability(
        execution_capability,
        label="_execute_standard_training after execution identity recomputation",
        config=config,
        run_root=run_candidate,
        repo_root=resolved_repo_root,
        stage5_authority=stage5_authority,
        verified_review_authorization=verified_review_authorization,
        execution_identity=identity,
    )
    identity_hashes = {
        key: identity.get(key)
        for key in (
            "config_sha256",
            "source_set_sha256",
            "prospective_tree_sha256",
            "data_sha256",
            "environment_sha256",
        )
    }
    try:
        validated_review_authorization = (
            validate_stage6_verified_review_authorization(
                verified_review_authorization,
                execution_identity=identity,
                formal_run_id=run_candidate.name,
            )
        )
    except Stage6WorkflowError as exc:
        raise StandardTrainingError(
            "Stage 6 verified review authorization drifted"
        ) from exc
    _require_standard_execution_capability(
        execution_capability,
        label="_execute_standard_training before runtime mkdir",
        config=config,
        run_root=run_candidate,
        repo_root=resolved_repo_root,
        stage5_authority=stage5_authority,
        verified_review_authorization=validated_review_authorization,
        execution_identity=identity,
    )
    try:
        run_candidate.mkdir(parents=True, exist_ok=True)
        resolved_run_root = require_plain_path(
            run_candidate,
            leaf_kind="directory",
            label="Stage 6 run root",
        )
        stage_candidate = resolved_run_root / "s6"
        require_plain_path(
            stage_candidate,
            base=resolved_run_root,
            allow_missing=True,
            label="Stage 6 stage root",
        )
        stage_candidate.mkdir(parents=False, exist_ok=True)
        stage_root = require_plain_path(
            stage_candidate,
            base=resolved_run_root,
            leaf_kind="directory",
            label="Stage 6 stage root",
        )
    except (OSError, PathSecurityError) as exc:
        raise StandardTrainingError(
            "Stage 6 runtime path contains a link or reparse point"
        ) from exc
    _require_standard_execution_capability(
        execution_capability,
        label="_execute_standard_training after runtime mkdir",
        config=config,
        run_root=resolved_run_root,
        repo_root=resolved_repo_root,
        stage5_authority=stage5_authority,
        verified_review_authorization=validated_review_authorization,
        execution_identity=identity,
    )
    preflight_audit = _require_machine_preflight(
        stage_root=stage_root,
        config_sha256=config_sha256,
        stage5_gate_sha256=gate_sha256,
    )
    environment_identity = identity.get("environment_identity")
    try:
        validated_environment = validate_stage6_environment_identity(
            environment_identity  # type: ignore[arg-type]
        )
        environment_binding_valid = (
            identity_hashes["environment_sha256"]
            == stage6_environment_sha256(validated_environment)
            and isinstance(preflight_audit, Mapping)
            and preflight_audit.get("environment_identity") == validated_environment
            and preflight_audit.get("environment_sha256")
            == identity_hashes["environment_sha256"]
        )
    except (Stage6WorkflowError, TypeError, ValueError):
        environment_binding_valid = False
    if (
        identity.get("schema_version") != "stage6_execution_identity/v1"
        or identity_hashes["config_sha256"] != config_sha256
        or any(not _is_sha256(value) for value in identity_hashes.values())
        or not environment_binding_valid
    ):
        raise StandardTrainingError("Stage 6 execution identity drifted")
    origin_immutable_bindings = {
        **identity_hashes,
        "environment_identity": validated_environment,
        "stage5_gate_sha256": gate_sha256,
        **_review_authorization_immutable_bindings(
            validated_review_authorization
        ),
    }
    manifest_binding = _FIXED_COVERAGE_CACHE_MANIFEST_BINDING
    audit_binding = _SENSOR_ACCELERATION_EVIDENCE_BINDINGS.get(
        "coverage_cache_formal_audit",
        {},
    )
    immutable_bindings = {
        **origin_immutable_bindings,
        "coverage_cache_manifest_path": manifest_binding.get("path"),
        "coverage_cache_manifest_sha256": manifest_binding.get("sha256"),
        "coverage_cache_manifest_size_bytes": manifest_binding.get("size_bytes"),
        "coverage_cache_root": manifest_binding.get("cache_root"),
        "coverage_cache_entry_set_sha256": manifest_binding.get(
            "entry_set_sha256"
        ),
        "coverage_cache_runtime_mode": "persistent_exact_manifest_read_only/v1",
        "coverage_cache_formal_audit_sha256": audit_binding.get("sha256"),
        "coverage_cache_formal_audit_size_bytes": audit_binding.get("size_bytes"),
    }
    if not _immutable_bindings_valid(immutable_bindings):
        raise StandardTrainingError(
            "Stage 6 verified review authorization binding drifted"
        )
    source_repair = None
    if planning_warm_start_context is not None:
        from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
            PlanningWarmStartContext,
            Stage6PlanningWarmStartError,
            VerifiedParentU74,
        )

        if (
            not isinstance(
                planning_warm_start_context,
                PlanningWarmStartContext,
            )
            or not isinstance(verified_parent_u74, VerifiedParentU74)
            or not _is_sha256(planning_warm_start_sha256)
            or planning_warm_start_context.artifact_sha256
            != planning_warm_start_sha256
            or planning_warm_start_context.child_run_id
            != resolved_run_root.name
            or planning_warm_start_context.child_effective_config_sha256
            != config_sha256
            or planning_warm_start_context.child_effective_config_bytes
            != config_bytes
        ):
            raise StandardTrainingError(
                "Stage 6 planning warm-start binding drifted"
            )
        try:
            planning_warm_start_context.require_current("execute startup")
        except Stage6PlanningWarmStartError as exc:
            raise StandardTrainingError(
                "Stage 6 planning warm-start artifact drifted"
            ) from exc
        if planning_child_recovery_capability is not None:
            from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
                PlanningChildRecoveryCapability,
            )

            recovery_capability = planning_child_recovery_capability
            immutable_sha256 = hashlib.sha256(
                ArtifactStore.canonical_json_bytes(immutable_bindings)
            ).hexdigest()
            if (
                not isinstance(
                    recovery_capability,
                    PlanningChildRecoveryCapability,
                )
                or recovery_capability.formal_run_id
                != resolved_run_root.name
                or recovery_capability.seed
                != config.training.seeds[0]
                or recovery_capability.stage_root != stage_root
                or recovery_capability.acceptance_binding.get(
                    "current_execution_identity_sha256"
                )
                != hashlib.sha256(
                    ArtifactStore.canonical_json_bytes(identity)
                ).hexdigest()
                or recovery_capability.acceptance_binding.get(
                    "current_immutable_bindings_sha256"
                )
                != immutable_sha256
            ):
                raise StandardTrainingError(
                    "Stage 6 planning child recovery capability drifted"
                )
            if recovery_capability.acceptance_binding.get(
                "current_verified_review_authorization"
            ) != validated_review_authorization:
                raise StandardTrainingError(
                    "Stage 6 planning child review authorization drifted"
                )
            runtime_immutable_bindings = immutable_bindings
        else:
            runtime_immutable_bindings = immutable_bindings
    else:
        try:
            source_repair = load_stage6_source_repair_context(
                stage_root=stage_root,
                current_execution_identity=identity,
                current_verified_review_authorization=(
                    validated_review_authorization
                ),
                current_immutable_bindings=immutable_bindings,
            )
        except Stage6SourceRepairError as exc:
            raise StandardTrainingError(
                "Stage 6 source-repair amendment drifted"
            ) from exc
        if source_repair is None or source_repair.sensor_acceleration_sha256 is None:
            raise StandardTrainingError(
                "Stage 6 production requires the complete ordinal6 cache binding"
            )
        runtime_immutable_bindings = dict(source_repair.current_immutable_bindings)
        try:
            source_repair.require_current("execute startup")
        except Stage6SourceRepairError as exc:
            raise StandardTrainingError(
                "Stage 6 source-repair amendment drifted"
            ) from exc
    _persist_execution_identity(
        stage_root=stage_root,
        config=config,
        run_root=resolved_run_root,
        repo_root=resolved_repo_root,
        stage5_authority=stage5_authority,
        execution_capability=execution_capability,
        config_bytes=config_bytes,
        identity=identity,
        immutable_bindings=runtime_immutable_bindings,
        verified_review_authorization=validated_review_authorization,
        planning_warm_start_context=planning_warm_start_context,
        planning_child_recovery_capability=(
            planning_child_recovery_capability
        ),
    )
    runtime_contract = (
        build_standard_catalog,
        standard_env_specs,
        load_stage4_policy_for_standard,
        SpawnVectorEnv,
        RolloutCollector,
        PPOTrainer,
        CheckpointManager,
        CheckpointManager.load_last_complete,
        SpawnVectorEnv.restore_states,
        run_standard_update_transaction,
        select_global_best,
        run_standard_evaluation,
        STANDARD_EVALUATION_METHODS,
        write_stage6_manifest,
        verify_stage6_machine_acceptance,
        ProcessTreeRSSMonitor,
    )
    if not runtime_contract:
        raise StandardTrainingError("production runtime contract is unavailable")
    if (
        planning_child_recovery_capability is not None
        and planning_child_recovery_capability.terminal_complete
    ):
        if (
            planning_child_recovery_capability.resume_cursor is not None
            or planning_child_recovery_capability.acceptance_binding.get(
                "terminal_complete"
            )
            not in {False, True}
            or (
                planning_child_recovery_capability.acceptance_binding.get(
                    "resume_cursor"
                )
                is not None
                and not isinstance(
                    planning_child_recovery_capability.acceptance_binding.get(
                        "resume_cursor"
                    ),
                    Mapping,
                )
            )
        ):
            raise StandardTrainingError(
                "Stage 6 terminal-complete recovery capability drifted"
            )
        acceptance = verify_stage6_machine_acceptance(
            stage_root=stage_root,
            repo_root=resolved_repo_root,
            planning_child_recovery_capability=(
                planning_child_recovery_capability
            ),
        )
        terminal_result: dict[str, object] = {
            "stage_root": str(stage_root.resolve())
        }
        for relative_path, key in (
            ("summary.json", "summary"),
            ("routing.json", "routing"),
        ):
            try:
                bound = secure_read_bytes(
                    stage_root / relative_path,
                    base=stage_root,
                    label=(
                        "Stage 6 terminal-complete "
                        f"{relative_path}"
                    ),
                )
                value = json.loads(bound.payload.decode("utf-8"))
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                PathSecurityError,
            ) as exc:
                raise StandardTrainingError(
                    "Stage 6 terminal-complete artifacts are unreadable"
                ) from exc
            if (
                not isinstance(value, Mapping)
                or ArtifactStore.canonical_json_bytes(value)
                != bound.payload
            ):
                raise StandardTrainingError(
                    "Stage 6 terminal-complete artifacts drifted"
                )
            terminal_result[key] = dict(value)
        if (
            not isinstance(acceptance, Mapping)
            or acceptance.get("passed") is not True
        ):
            raise StandardTrainingError(
                "Stage 6 terminal-complete acceptance drifted"
            )
        _require_standard_execution_capability(
            execution_capability,
            label="_execute_standard_training terminal-complete exit",
            config=config,
            run_root=resolved_run_root,
            repo_root=resolved_repo_root,
            stage5_authority=stage5_authority,
            verified_review_authorization=(
                validated_review_authorization
            ),
            execution_identity=identity,
        )
        return terminal_result
    with ProcessTreeRSSMonitor() as process_tree_monitor:
        peak_vram_bytes = (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else 0
        )
        first_snapshot = capture_resource_snapshot(
            peak_vram_bytes=peak_vram_bytes,
            process_tree_monitor=process_tree_monitor,
        )
        first_decision = evaluate_resource_gates(
            first_snapshot,
            preflight=False,
        )
        first_sample = _resource_gate_record(
            first_snapshot,
            first_decision,
        )
        if first_sample.get("passed") is not True:
            raise StandardTrainingError("resource segment start hard stop")
        with _standard_execution_operation(
            execution_capability,
            label="initial resource segment publication",
            config=config,
            run_root=resolved_run_root,
            repo_root=resolved_repo_root,
            stage5_authority=stage5_authority,
            verified_review_authorization=validated_review_authorization,
            execution_identity=identity,
        ):
            resource_segment_start = _append_initial_resource_segment_start(
                stage_root=stage_root,
                first_sample=first_sample,
                planning_child_recovery_capability=(
                    planning_child_recovery_capability
                ),
            )
        if planning_child_recovery_capability is not None:
            cursor = planning_child_recovery_capability.resume_cursor
            if (
                cursor is not None
                and
                resource_segment_start.get("segment_index")
                != cursor.next_resource_segment_index
            ):
                raise StandardTrainingError(
                    "Stage 6 planning child recovery resource segment drifted"
                )
        backend_kwargs: dict[str, object] = {
            "config": config,
            "run_root": resolved_run_root,
            "repo_root": resolved_repo_root,
            "stage5_authority": stage5_authority,
            "execution_capability": execution_capability,
            "_immutable_bindings": runtime_immutable_bindings,
            "_process_tree_monitor": process_tree_monitor,
            "_resource_segment_start": resource_segment_start,
        }
        if source_repair is not None:
            backend_kwargs["_source_repair"] = source_repair
        if planning_warm_start_context is not None:
            backend_kwargs.update(
                {
                    "_planning_warm_start": planning_warm_start_context,
                    "_planning_warm_start_sha256": planning_warm_start_sha256,
                    "_verified_parent_u74": verified_parent_u74,
                }
            )
        if planning_child_recovery_capability is not None:
            backend_kwargs.update(
                {
                    "_planning_child_recovery_capability": (
                        planning_child_recovery_capability
                    ),
                }
            )
        backend = StandardProductionBackend(
            **backend_kwargs,
        )
        backend.record_preflight_phase()
        result = run_standard_training_schedule(config, backend)
        backend.capture_terminal_resource_sample()
        process_tree_monitor.stop()
        backend.record_terminal_resource_evidence()
        result = backend.commit_terminal_acceptance(result)
        terminal_verifier_kwargs: dict[str, object] = {
            "stage_root": stage_root,
            "repo_root": resolved_repo_root,
        }
        if planning_child_recovery_capability is not None:
            terminal_verifier_kwargs[
                "planning_child_recovery_capability"
            ] = planning_child_recovery_capability
        acceptance = verify_stage6_machine_acceptance(
            **terminal_verifier_kwargs
        )
        if (
            not isinstance(result, Mapping)
            or set(result) != {"stage_root", "summary", "routing"}
            or not isinstance(result["summary"], Mapping)
            or not isinstance(result["routing"], Mapping)
            or not isinstance(acceptance, Mapping)
            or acceptance.get("passed") is not True
        ):
            raise StandardTrainingError("production schedule result drifted")
        normalized_result = {
            "stage_root": str(Path(result["stage_root"]).expanduser().resolve()),
            "summary": dict(result["summary"]),
            "routing": dict(result["routing"]),
        }
    _require_standard_execution_capability(
        execution_capability,
        label="_execute_standard_training internal exit",
        config=config,
        run_root=resolved_run_root,
        repo_root=resolved_repo_root,
        stage5_authority=stage5_authority,
        verified_review_authorization=validated_review_authorization,
        execution_identity=identity,
    )
    return normalized_result


def _receipt_record_hash(value: Mapping[str, object]) -> str:
    return hashlib.sha256(ArtifactStore.canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "AUDIT_UPDATES",
    "CheckpointReceiptIndex",
    "EvalIsolationSnapshot",
    "FROZEN_SEEDS",
    "SAMPLER_SEED_DERIVATION_VERSION",
    "StandardUpdateResult",
    "StandardTrainingTransaction",
    "StandardTransactionCheckpoint",
    "StandardProductionBackend",
    "StandardTrainingError",
    "StandardTrainingStateMachine",
    "TrainingUnit",
    "ValidationRecord",
    "audit_required",
    "append_transaction_metric_once",
    "build_stage6_acceptance_artifacts",
    "build_standard_final_aggregate_artifacts",
    "build_standard_training_transactions",
    "commit_checkpointed_transaction",
    "completed_transaction_keys_from_journal",
    "derive_standard_sampler_seeds",
    "execute_standard_training",
    "reconcile_checkpointed_resume",
    "resume_standard_training_transactions",
    "run_eval_only_transaction",
    "run_standard_training_schedule",
    "run_standard_update_transaction",
    "select_global_best",
    "select_seed_best",
    "select_validation_checkpoint_best",
    "validate_checkpoint_replay_audit",
    "validate_checkpoint_runtime_payload",
    "validate_math_audit",
    "validate_stage6_resource_audit",
    "validate_standard_collection_audit",
    "validate_standard_training_validation_rows",
    "verify_standard_final_evaluation_artifacts",
    "verify_standard_final_evaluation_artifacts_from_bytes",
    "verify_journal_checkpoint_bindings",
]
