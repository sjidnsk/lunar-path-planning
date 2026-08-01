"""One-shot immutable recovery capability for a Stage 6 planning child."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from lunar_exploration_ppo.configs.stage6 import SafetyContract
from lunar_exploration_ppo.ppo.checkpoint import (
    CheckpointError,
    inspect_complete_checkpoint_snapshot,
)
from lunar_exploration_ppo.ppo.standard_training import (
    CheckpointReceiptIndex,
    StandardTrainingError,
    completed_transaction_keys_from_journal,
    validate_standard_training_validation_rows,
    verify_journal_checkpoint_bindings,
)
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.path_security import (
    PathSecurityError,
    SecureReadResult,
    lexical_absolute,
    require_plain_path,
    secure_read_bytes,
)
from lunar_exploration_ppo.utils.resource_lifecycle import (
    ResourceLifecycleError,
    validate_resource_lifecycle_rows,
)
from lunar_exploration_ppo.workflows.stage6_planning_child_source_repair import (
    PLANNING_CHILD_SOURCE_REPAIR_NAME,
    Stage6PlanningChildSourceRepairError,
    ValidatedPlanningChildParent,
    _canonical_sha256,
    _json_mapping,
    _jsonl_rows,
    _require_int,
    _require_mapping,
    _require_sha256,
    _require_string,
    _sha256,
    validate_planning_child_source_repair_bytes,
)
from lunar_exploration_ppo.workflows.stage6_planning_child_source_repair_continuation import (
    PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_LINEAGE_SCHEMA,
    PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_NAME,
    Stage6PlanningChildSourceRepairContinuationError,
    ValidatedPlanningChildContinuation,
    _v2_canonical_record,
    _v2_file_binding,
    validate_planning_child_source_repair_continuation_bytes,
)
from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
    Stage6PlanningWarmStartError,
    build_planning_child_transactions,
    parse_planning_effective_config_bytes,
    validate_planning_warm_start_artifact,
)


_JOURNAL_PATHS = (
    "checkpoints/index.jsonl",
    "job-state.jsonl",
    "resource_audit.jsonl",
    "training_metrics.jsonl",
    "validation_metrics.jsonl",
)
_CHECKPOINT_MEMBERS = (
    "checkpoint.pt",
    "complete.json",
    "manifest.json",
)
_ANCHOR_SCHEMA = "stage6_planning_child_recovery_accepted_anchor/v1"
_CAPABILITY_ACCEPTANCE_SCHEMA = (
    "stage6_planning_child_recovery_capability_acceptance/v1"
)
_CRASH_SUFFIX_SCHEMA = (
    "stage6_planning_child_recovery_crash_suffix/v1"
)
PLANNING_CHILD_RECOVERY_EVIDENCE_SCHEMA = (
    "stage6_planning_child_recovery_semantic_binding/v1"
)
_CHECKPOINT_LINEAGE_SCHEMA = (
    "stage6_planning_child_recovery_checkpoint_lineage/v1"
)


class Stage6PlanningChildRecoveryError(RuntimeError):
    """The planning-child recovery graph is incomplete or has drifted."""


@dataclass(frozen=True, slots=True)
class PlanningChildResumeCursor:
    last_accepted_update: int
    next_update: int
    next_attempt: int
    next_transaction_key: str
    next_resource_segment_index: int
    latest_complete_checkpoint_update: int
    pending_pre_attempt: int | None


@dataclass(frozen=True, slots=True)
class PlanningChildLineageEpoch:
    first_update: int
    last_update: int | None
    artifact_sha256: str
    execution_identity_sha256: str


@dataclass(frozen=True, slots=True)
class PlanningChildRecoveryAnchor:
    formal_run_id: str
    seed: int
    stage_root: Path
    parent_artifact_sha256: str
    input_snapshot_sha256: str
    accepted_anchor: Mapping[str, object]
    journal_prefixes: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class PlanningChildRecoveryCapability:
    formal_run_id: str
    seed: int
    stage_root: Path
    parent_artifact_sha256: str
    continuation_artifact_sha256: str | None
    input_snapshot_sha256: str
    capability_sha256: str
    resume_cursor: PlanningChildResumeCursor | None
    lineage_epochs: tuple[PlanningChildLineageEpoch, ...]
    protected_checkpoint_updates: tuple[int, ...]
    input_pin_requests: tuple[tuple[str, Path], ...]
    acceptance_binding: Mapping[str, object]
    terminal_complete: bool = False

    def evidence_binding(self) -> Mapping[str, object]:
        """Return the one canonical machine/terminal acceptance binding."""

        return {
            "schema_version": (
                PLANNING_CHILD_RECOVERY_EVIDENCE_SCHEMA
            ),
            "capability_sha256": self.capability_sha256,
            "input_snapshot_sha256": self.input_snapshot_sha256,
            "parent_artifact_sha256": (
                self.parent_artifact_sha256
            ),
            "continuation_artifact_sha256": (
                self.continuation_artifact_sha256
            ),
            "acceptance_binding": _plain_json(
                self.acceptance_binding
            ),
        }

    def checkpoint_lineage_for_update(
        self,
        update: int,
    ) -> Mapping[str, object]:
        if type(update) is not int:
            raise Stage6PlanningChildRecoveryError(
                "planning child checkpoint update is invalid"
            )
        selected = next(
            (
                (index, epoch)
                for index, epoch in enumerate(self.lineage_epochs)
                if update >= epoch.first_update
                and (
                    epoch.last_update is None
                    or update <= epoch.last_update
                )
            ),
            None,
        )
        if selected is None:
            raise Stage6PlanningChildRecoveryError(
                "planning child checkpoint update is outside lineage epochs"
            )
        templates = self.acceptance_binding.get(
            "checkpoint_lineage_templates"
        )
        if not isinstance(templates, Mapping):
            raise Stage6PlanningChildRecoveryError(
                "planning child checkpoint lineage templates are missing"
            )
        template_name = ("origin", "parent", "continuation")[
            selected[0]
        ]
        template = templates.get(template_name)
        if not isinstance(template, Mapping):
            raise Stage6PlanningChildRecoveryError(
                "planning child checkpoint lineage template is missing"
            )
        result = _plain_json(template)
        if not isinstance(result, dict):
            raise Stage6PlanningChildRecoveryError(
                "planning child checkpoint lineage template drifted"
            )
        result["checkpoint_update"] = update
        return _deep_freeze(result)


@dataclass(frozen=True, slots=True)
class _BoundInput:
    label: str
    path: Path
    payload: bytes
    size_bytes: int
    sha256: str
    secure_identity: tuple[int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class _LaunchBindings:
    current_execution_identity: Mapping[str, object]
    current_verified_review_authorization: Mapping[str, object]
    current_immutable_bindings: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _BoundRecoverySnapshot:
    stage_root: Path
    parent_path: Path
    continuation_path: Path | None
    inputs: Mapping[str, _BoundInput]
    parent: ValidatedPlanningChildParent
    continuation_record: Mapping[str, object] | None
    ahead_checkpoint_directory: Path | None
    ahead_checkpoint_update: int | None
    ahead_checkpoint_published: bool


@dataclass(frozen=True, slots=True)
class _ValidatedRecoverySnapshot:
    parent: ValidatedPlanningChildParent
    continuation: ValidatedPlanningChildContinuation | None
    input_snapshot_sha256: str
    accepted_anchor: Mapping[str, object]
    crash_suffix: Mapping[str, object] | None
    journal_prefixes: Mapping[str, Mapping[str, object]]
    resume_cursor: PlanningChildResumeCursor | None
    lineage_epochs: tuple[PlanningChildLineageEpoch, ...]
    checkpoint_lineage_templates: Mapping[str, Mapping[str, object]]
    journal_binding_epochs: tuple[
        tuple[str, Mapping[str, object]], ...
    ]
    protected_checkpoint_updates: tuple[int, ...]
    input_pin_requests: tuple[tuple[str, Path], ...]


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _deep_freeze(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _plain_json(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def _frozen_mapping(value: object, label: str) -> Mapping[str, object]:
    mapping = _require_mapping(value, label)
    frozen = _deep_freeze(_plain_json(mapping))
    if not isinstance(frozen, Mapping):
        raise Stage6PlanningChildRecoveryError(f"{label} drifted")
    return frozen


def _domain_error(
    label: str,
    exc: BaseException,
) -> Stage6PlanningChildRecoveryError:
    return Stage6PlanningChildRecoveryError(label)


def _capture_bound_input(
    *,
    label: str,
    path: str | Path,
    base: Path | None,
    by_label: dict[str, _BoundInput],
    by_path: dict[Path, _BoundInput],
) -> _BoundInput:
    candidate = lexical_absolute(path)
    existing = by_path.get(candidate)
    if existing is not None:
        by_label[label] = existing
        return existing
    try:
        result = secure_read_bytes(
            candidate,
            base=base,
            label=label,
        )
    except PathSecurityError as exc:
        raise _domain_error(
            f"planning child recovery input {label} is unsafe",
            exc,
        ) from exc
    bound = _BoundInput(
        label=label,
        path=candidate,
        payload=result.payload,
        size_bytes=len(result.payload),
        sha256=_sha256(result.payload),
        secure_identity=result.stat_identity,
    )
    by_path[candidate] = bound
    by_label[label] = bound
    return bound


def _binding_path(
    value: object,
    *,
    label: str,
) -> Path:
    binding = _require_mapping(value, label)
    return lexical_absolute(
        _require_string(binding.get("path"), f"{label} path")
    )


def _capture_recovery_snapshot(
    *,
    stage_root: Path,
    parent_artifact_path: Path,
    continuation_artifact_path: Path | None,
) -> _BoundRecoverySnapshot:
    root = lexical_absolute(stage_root)
    parent_path = lexical_absolute(parent_artifact_path)
    continuation_path = (
        lexical_absolute(continuation_artifact_path)
        if continuation_artifact_path is not None
        else None
    )
    try:
        require_plain_path(
            root,
            leaf_kind="directory",
            label="planning child recovery stage root",
        )
    except PathSecurityError as exc:
        raise _domain_error(
            "planning child recovery stage root is unsafe",
            exc,
        ) from exc
    if parent_path != root / PLANNING_CHILD_SOURCE_REPAIR_NAME:
        raise Stage6PlanningChildRecoveryError(
            "planning child recovery parent path is not canonical"
        )
    if continuation_path is not None and (
        continuation_path
        != root / PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_NAME
    ):
        raise Stage6PlanningChildRecoveryError(
            "planning child recovery continuation path is not canonical"
        )

    by_label: dict[str, _BoundInput] = {}
    by_path: dict[Path, _BoundInput] = {}
    parent_bound = _capture_bound_input(
        label="parent-artifact",
        path=parent_path,
        base=root,
        by_label=by_label,
        by_path=by_path,
    )
    try:
        parent = validate_planning_child_source_repair_bytes(
            parent_bound.payload,
            stage_root=root,
            artifact_path=parent_path,
        )
    except Stage6PlanningChildSourceRepairError as exc:
        raise _domain_error(
            "planning child recovery parent artifact drifted",
            exc,
        ) from exc

    for label, binding in parent.immutable_inputs.items():
        _capture_bound_input(
            label=f"parent-input:{label}",
            path=_binding_path(
                binding,
                label=f"parent immutable input {label}",
            ),
            base=None,
            by_label=by_label,
            by_path=by_path,
        )
    for relative in _JOURNAL_PATHS:
        _capture_bound_input(
            label=f"journal:{relative}",
            path=root / relative,
            base=root,
            by_label=by_label,
            by_path=by_path,
        )

    continuation_record: Mapping[str, object] | None = None
    if continuation_path is not None:
        continuation_bound = _capture_bound_input(
            label="continuation-artifact",
            path=continuation_path,
            base=root,
            by_label=by_label,
            by_path=by_path,
        )
        try:
            continuation_record = MappingProxyType(
                _v2_canonical_record(
                    _json_mapping(
                        continuation_bound.payload,
                        "planning child recovery continuation",
                    )
                )
            )
            current = _require_mapping(
                continuation_record.get("current"),
                "planning child recovery continuation current",
            )
            authorization_binding = _v2_file_binding(
                current.get("authorization"),
                label="planning child recovery current authorization",
            )
            _capture_bound_input(
                label="continuation-input:current-authorization",
                path=authorization_binding["path"],
                base=None,
                by_label=by_label,
                by_path=by_path,
            )
            reviews = _require_mapping(
                continuation_record.get("review_evidence"),
                "planning child recovery continuation reviews",
            )
            for label in (
                "implementation_report",
                "spec_rereview",
                "quality_rereview",
            ):
                binding = _v2_file_binding(
                    reviews.get(label),
                    label=f"planning child recovery {label}",
                )
                _capture_bound_input(
                    label=f"continuation-input:{label}",
                    path=binding["path"],
                    base=None,
                    by_label=by_label,
                    by_path=by_path,
                )
            anchor = _require_mapping(
                continuation_record.get("accepted_anchor"),
                "planning child recovery continuation anchor",
            )
            latest = _require_mapping(
                anchor.get("latest_complete_checkpoint"),
                "planning child recovery anchor checkpoint",
            )
            members = _require_mapping(
                latest.get("members"),
                "planning child recovery anchor checkpoint members",
            )
            for name in _CHECKPOINT_MEMBERS:
                _capture_bound_input(
                    label=f"anchor-checkpoint:{name}",
                    path=_binding_path(
                        members.get(name),
                        label=f"anchor checkpoint {name}",
                    ),
                    base=root,
                    by_label=by_label,
                    by_path=by_path,
                )
        except (
            Stage6PlanningChildSourceRepairContinuationError,
            Stage6PlanningChildSourceRepairError,
        ) as exc:
            raise _domain_error(
                "planning child recovery continuation artifact drifted",
                exc,
            ) from exc

    try:
        receipt_rows = _jsonl_rows(
            by_label["journal:checkpoints/index.jsonl"].payload,
            "planning child recovery checkpoint receipts",
        )
        training_rows = _jsonl_rows(
            by_label["journal:training_metrics.jsonl"].payload,
            "planning child recovery training metrics",
        )
        if not receipt_rows or not training_rows:
            raise Stage6PlanningChildRecoveryError(
                "planning child recovery stable checkpoint prefix is empty"
            )
        latest_update = _require_int(
            training_rows[-1].get("update"),
            "planning child recovery latest stable update",
        )
        config_path = _binding_path(
            parent.immutable_inputs["config"],
            label="planning child recovery config",
        )
        config_bound = by_path.get(config_path)
        if config_bound is None:
            raise Stage6PlanningChildRecoveryError(
                "planning child recovery config binding is missing"
            )
        transactions = build_planning_child_transactions(
            parse_planning_effective_config_bytes(
                config_bound.payload
            ).base_config
        )
        final_update = transactions[-1].update
    except Stage6PlanningChildSourceRepairError as exc:
        raise _domain_error(
            "planning child recovery stable checkpoint prefix drifted",
            exc,
        ) from exc
    seed_root = (
        root
        / "checkpoints"
        / f"seed-{parent.seed}"
    )
    try:
        seed_root = require_plain_path(
            seed_root,
            base=root,
            leaf_kind="directory",
            label="planning child recovery checkpoint seed root",
        )
        checkpoint_inventory = tuple(seed_root.iterdir())
    except (OSError, PathSecurityError) as exc:
        raise _domain_error(
            "planning child recovery checkpoint inventory drifted",
            exc,
        ) from exc

    canonical_by_update: dict[int, Path] = {}
    pending_by_update: dict[int, list[Path]] = {}
    for candidate in checkpoint_inventory:
        name = candidate.name
        update_text: str | None = None
        pending_nonce: str | None = None
        if name.startswith("update-"):
            update_text = name.removeprefix("update-")
            if len(update_text) != 8 or not update_text.isdigit():
                continue
        elif name.startswith(".pending-update-"):
            tail = name.removeprefix(".pending-update-")
            update_text, separator, pending_nonce = tail.partition("-")
            if (
                separator != "-"
                or len(update_text) != 8
                or not update_text.isdigit()
                or len(pending_nonce) != 32
                or any(
                    character not in "0123456789abcdef"
                    for character in pending_nonce
                )
            ):
                continue
        else:
            continue
        update = int(update_text)
        if update > final_update:
            raise Stage6PlanningChildRecoveryError(
                "planning child recovery checkpoint inventory contains "
                "an update after the final transaction"
            )
        if pending_nonce is None:
            if update in canonical_by_update:
                raise Stage6PlanningChildRecoveryError(
                    "planning child recovery canonical checkpoint inventory "
                    "is not exact"
                )
            canonical_by_update[update] = candidate
        else:
            pending_by_update.setdefault(update, []).append(candidate)

    latest_root = (
        canonical_by_update.get(latest_update)
    )
    if latest_root is None:
        raise Stage6PlanningChildRecoveryError(
            "planning child recovery latest stable checkpoint is missing"
        )
    for name in _CHECKPOINT_MEMBERS:
        _capture_bound_input(
            label=f"latest-checkpoint:{latest_update}:{name}",
            path=latest_root / name,
            base=root,
            by_label=by_label,
            by_path=by_path,
        )

    ahead_checkpoint_directory: Path | None = None
    ahead_checkpoint_update: int | None = None
    ahead_checkpoint_published = False
    try:
        resource_rows = _jsonl_rows(
            by_label["journal:resource_audit.jsonl"].payload,
            "planning child recovery resource audit",
        )
        accepted_rows = [
            row
            for row in resource_rows
            if row.get("phase") == "accepted"
            and row.get("kind") == "update"
        ]
        if len(accepted_rows) == len(training_rows) + 1:
            ahead = accepted_rows[-1]
            ahead_seed = _require_int(
                ahead.get("seed"),
                "planning child recovery ahead checkpoint seed",
            )
            ahead_checkpoint_update = _require_int(
                ahead.get("update"),
                "planning child recovery ahead checkpoint update",
            )
            if ahead_seed != parent.seed:
                raise Stage6PlanningChildRecoveryError(
                    "planning child recovery ahead checkpoint seed drifted"
                )
            canonical = (
                seed_root
                / f"update-{ahead_checkpoint_update:08d}"
            )
            pending_prefix = (
                f".pending-update-{ahead_checkpoint_update:08d}-"
            )
            pending = tuple(
                pending_by_update.get(ahead_checkpoint_update, ())
            )
            candidates = (
                (
                    (canonical,)
                    if canonical_by_update.get(ahead_checkpoint_update)
                    == canonical
                    else ()
                )
                + pending
            )
            if len(candidates) != 1:
                raise Stage6PlanningChildRecoveryError(
                    "planning child recovery ahead checkpoint bundle is not exact"
                )
            ahead_checkpoint_directory = require_plain_path(
                candidates[0],
                base=root,
                leaf_kind="directory",
                label="planning child recovery ahead checkpoint directory",
            )
            ahead_checkpoint_published = (
                ahead_checkpoint_directory == canonical
            )
            members = tuple(ahead_checkpoint_directory.iterdir())
            if {member.name for member in members} != set(
                _CHECKPOINT_MEMBERS
            ):
                raise Stage6PlanningChildRecoveryError(
                    "planning child recovery ahead checkpoint members drifted"
                )
            for name in _CHECKPOINT_MEMBERS:
                _capture_bound_input(
                    label=(
                        "ahead-checkpoint:"
                        f"{ahead_checkpoint_update}:{name}"
                    ),
                    path=ahead_checkpoint_directory / name,
                    base=root,
                    by_label=by_label,
                    by_path=by_path,
                )
    except (
        OSError,
        PathSecurityError,
        Stage6PlanningChildSourceRepairError,
    ) as exc:
        raise _domain_error(
            "planning child recovery ahead checkpoint drifted",
            exc,
        ) from exc

    return _BoundRecoverySnapshot(
        stage_root=root,
        parent_path=parent_path,
        continuation_path=continuation_path,
        inputs=MappingProxyType(by_label),
        parent=parent,
        continuation_record=continuation_record,
        ahead_checkpoint_directory=ahead_checkpoint_directory,
        ahead_checkpoint_update=ahead_checkpoint_update,
        ahead_checkpoint_published=ahead_checkpoint_published,
    )


def _bound_for_path(
    snapshot: _BoundRecoverySnapshot,
    path: str | Path,
    *,
    label: str,
) -> _BoundInput:
    candidate = lexical_absolute(path)
    matching = {
        id(bound): bound
        for bound in snapshot.inputs.values()
        if bound.path == candidate
    }
    if len(matching) != 1:
        raise Stage6PlanningChildRecoveryError(
            f"planning child recovery bound input {label} is missing"
        )
    return next(iter(matching.values()))


def _require_bound_binding(
    snapshot: _BoundRecoverySnapshot,
    value: Mapping[str, object],
    *,
    label: str,
) -> _BoundInput:
    path = _binding_path(value, label=label)
    bound = _bound_for_path(snapshot, path, label=label)
    if (
        bound.size_bytes
        != _require_int(value.get("size_bytes"), f"{label} size")
        or bound.sha256
        != _require_sha256(value.get("sha256"), f"{label} SHA")
    ):
        raise Stage6PlanningChildRecoveryError(
            f"planning child recovery {label} bytes drifted"
        )
    return bound


def _require_prefix(
    payload: bytes,
    binding: Mapping[str, object],
    *,
    label: str,
    size_key: str,
    sha_key: str,
) -> bytes:
    size_bytes = _require_int(binding.get(size_key), f"{label} size")
    expected_sha256 = _require_sha256(
        binding.get(sha_key),
        f"{label} SHA",
    )
    line_count = _require_int(
        binding.get("line_count"),
        f"{label} line count",
    )
    if (
        size_bytes <= 0
        or len(payload) < size_bytes
        or _sha256(payload[:size_bytes]) != expected_sha256
        or payload[:size_bytes].count(b"\n") != line_count
    ):
        raise Stage6PlanningChildRecoveryError(
            f"planning child recovery {label} prefix drifted"
        )
    return payload[:size_bytes]


def _parent_checkpoint_lineage(
    parent: ValidatedPlanningChildParent,
) -> dict[str, object]:
    record = parent.record
    origin = _require_mapping(record.get("origin"), "parent origin")
    checkpoint = _require_mapping(
        origin.get("checkpoint"),
        "parent origin checkpoint",
    )
    accepted = _require_mapping(
        record.get("accepted_prefix"),
        "parent accepted prefix",
    )
    resume = _require_mapping(
        record.get("resume_point"),
        "parent resume point",
    )
    return {
        "schema_version": record["schema_version"],
        "artifact_sha256": parent.artifact_sha256,
        "origin_lineage_sha256": _require_sha256(
            _require_mapping(
                origin.get("lineage"),
                "parent origin lineage",
            ).get("sha256"),
            "parent origin lineage SHA",
        ),
        "last_origin_update": _require_int(
            accepted.get("last_update"),
            "parent last origin update",
        ),
        "first_repaired_update": _require_int(
            resume.get("next_update"),
            "parent first repaired update",
        ),
        "current_execution_identity_sha256": (
            parent.current_execution_identity_sha256
        ),
        "_origin_checkpoint_lineage": dict(
            _require_mapping(
                checkpoint.get("lineage"),
                "parent origin checkpoint lineage",
            )
        ),
    }


def _lineage_epochs(
    parent: ValidatedPlanningChildParent,
    continuation: ValidatedPlanningChildContinuation | None,
) -> tuple[PlanningChildLineageEpoch, ...]:
    record = parent.record
    origin = _require_mapping(record.get("origin"), "parent origin")
    accepted = _require_mapping(
        record.get("accepted_prefix"),
        "parent accepted prefix",
    )
    resume = _require_mapping(
        record.get("resume_point"),
        "parent resume point",
    )
    warm = _require_mapping(
        origin.get("planning_warm_start"),
        "parent planning warm-start",
    )
    origin_end = _require_int(
        accepted.get("last_update"),
        "parent origin final update",
    )
    parent_update = _require_int(
        resume.get("next_update"),
        "parent repaired update",
    )
    if parent_update != origin_end + 1:
        raise Stage6PlanningChildRecoveryError(
            "planning child recovery parent epoch gap drifted"
        )
    origin_identity_sha256 = _require_sha256(
        origin.get("execution_identity_sha256"),
        "parent origin execution identity SHA",
    )
    warm_child = _require_mapping(
        _require_mapping(
            warm.get("artifact"),
            "parent planning warm-start artifact",
        ).get("child"),
        "parent planning warm-start child",
    )
    epochs = [
        PlanningChildLineageEpoch(
            first_update=_require_int(
                warm_child.get("first_update"),
                "parent planning warm-start first update",
            ),
            last_update=origin_end,
            artifact_sha256=_require_sha256(
                warm.get("sha256"),
                "parent planning warm-start SHA",
            ),
            execution_identity_sha256=origin_identity_sha256,
        ),
        PlanningChildLineageEpoch(
            first_update=parent_update,
            last_update=parent_update,
            artifact_sha256=parent.artifact_sha256,
            execution_identity_sha256=(
                parent.current_execution_identity_sha256
            ),
        ),
    ]
    if continuation is not None:
        first_update = _require_int(
            continuation.lineage_epoch.get("first_update"),
            "continuation epoch first update",
        )
        if first_update != parent_update + 1:
            raise Stage6PlanningChildRecoveryError(
                "planning child recovery continuation epoch gap drifted"
            )
        epochs.append(
            PlanningChildLineageEpoch(
                first_update=first_update,
                last_update=None,
                artifact_sha256=continuation.artifact_sha256,
                execution_identity_sha256=(
                    continuation.current_execution_identity_sha256
                ),
            )
        )
    return tuple(epochs)


def _checkpoint_lineage_templates(
    parent: ValidatedPlanningChildParent,
    *,
    continuation: ValidatedPlanningChildContinuation | None,
    launch_bindings: _LaunchBindings | None,
    config: object,
    effective_config_bytes: bytes,
) -> dict[str, Mapping[str, object]]:
    from lunar_exploration_ppo.workflows.stage6 import (
        Stage6WorkflowError,
        _stage6_checkpoint_lineage_base,
    )

    epochs = _lineage_epochs(parent, continuation)
    parent_lineage = _parent_checkpoint_lineage(parent)
    origin_checkpoint_lineage = {
        **dict(
            _stage6_checkpoint_lineage_base(
                config=config,
                bindings=parent.origin_immutable_bindings,
                safety_contract=SafetyContract.from_stage6_config(config),
            )
        )
    }
    parent_lineage.pop("_origin_checkpoint_lineage")
    origin = _require_mapping(parent.record.get("origin"), "parent origin")
    warm = _require_mapping(
        origin.get("planning_warm_start"),
        "parent planning warm-start",
    )
    warm_artifact = _require_mapping(
        warm.get("artifact"),
        "parent planning warm-start artifact",
    )
    warm_sha256 = _require_sha256(
        warm.get("sha256"),
        "parent planning warm-start SHA",
    )
    try:
        warm_context = validate_planning_warm_start_artifact(
            warm_artifact,
            expected_child_run_id=parent.formal_run_id,
            expected_child_config_bytes=effective_config_bytes,
        )
        origin_checkpoint_lineage.update(
            warm_context.checkpoint_lineage_for_update(
                epochs[0].first_update,
                warm_start_artifact_sha256=warm_sha256,
            )
        )
        parent_checkpoint_lineage = dict(
            _stage6_checkpoint_lineage_base(
                config=config,
                bindings=parent.current_immutable_bindings,
                safety_contract=SafetyContract.from_stage6_config(config),
            )
        )
        parent_checkpoint_lineage.update(
            warm_context.checkpoint_lineage_for_update(
                epochs[1].first_update,
                warm_start_artifact_sha256=warm_sha256,
            )
        )
    except (Stage6PlanningWarmStartError, Stage6WorkflowError) as exc:
        raise _domain_error(
            "planning child recovery checkpoint lineage base drifted",
            exc,
        ) from exc
    parent_checkpoint_lineage["planning_child_source_repair"] = (
        parent_lineage
    )
    templates: dict[str, Mapping[str, object]] = {
        "origin": MappingProxyType(origin_checkpoint_lineage),
        "parent": MappingProxyType(parent_checkpoint_lineage),
    }
    if continuation is not None:
        if launch_bindings is None or len(epochs) != 3:
            raise Stage6PlanningChildRecoveryError(
                "planning child recovery continuation lineage bindings are missing"
            )
        try:
            continuation_checkpoint_lineage = dict(
                _stage6_checkpoint_lineage_base(
                    config=config,
                    bindings=launch_bindings.current_immutable_bindings,
                    safety_contract=SafetyContract.from_stage6_config(config),
                )
            )
            continuation_checkpoint_lineage.update(
                warm_context.checkpoint_lineage_for_update(
                    epochs[2].first_update,
                    warm_start_artifact_sha256=warm_sha256,
                )
            )
        except (
            Stage6PlanningWarmStartError,
            Stage6WorkflowError,
        ) as exc:
            raise _domain_error(
                "planning child recovery continuation lineage drifted",
                exc,
            ) from exc
        continuation_checkpoint_lineage[
            "planning_child_source_repair"
        ] = {
            "schema_version": (
                PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_LINEAGE_SCHEMA
            ),
            "artifact_sha256": continuation.artifact_sha256,
            "parent": parent_lineage,
            "first_continuation_update": epochs[2].first_update,
            "current_execution_identity_sha256": (
                continuation.current_execution_identity_sha256
            ),
        }
        templates["continuation"] = MappingProxyType(
            continuation_checkpoint_lineage
        )
    return templates


def _expected_checkpoint_lineage(
    epochs: Sequence[PlanningChildLineageEpoch],
    templates: Mapping[str, Mapping[str, object]],
    *,
    update: int,
) -> dict[str, object]:
    selected = next(
        (
            (index, epoch)
            for index, epoch in enumerate(epochs)
            if update >= epoch.first_update
            and (
                epoch.last_update is None
                or update <= epoch.last_update
            )
        ),
        None,
    )
    if selected is None:
        raise Stage6PlanningChildRecoveryError(
            "planning child recovery checkpoint lineage update drifted"
        )
    template_name = ("origin", "parent", "continuation")[
        selected[0]
    ]
    template = templates.get(template_name)
    if not isinstance(template, Mapping):
        raise Stage6PlanningChildRecoveryError(
            "planning child recovery checkpoint lineage template drifted"
        )
    result = _plain_json(template)
    if not isinstance(result, dict):
        raise Stage6PlanningChildRecoveryError(
            "planning child recovery checkpoint lineage template drifted"
        )
    result["checkpoint_update"] = update
    return result


def _checkpoint_bundle(
    snapshot: _BoundRecoverySnapshot,
    *,
    update: int,
) -> dict[str, _BoundInput]:
    root = (
        snapshot.ahead_checkpoint_directory
        if snapshot.ahead_checkpoint_update == update
        else (
            snapshot.stage_root
            / "checkpoints"
            / f"seed-{snapshot.parent.seed}"
            / f"update-{update:08d}"
        )
    )
    if root is None:
        raise Stage6PlanningChildRecoveryError(
            "planning child recovery checkpoint directory is missing"
        )
    return {
        name: _bound_for_path(
            snapshot,
            root / name,
            label=f"checkpoint U{update} {name}",
        )
        for name in _CHECKPOINT_MEMBERS
    }


def _journal_rows(
    payloads: Mapping[str, bytes],
) -> dict[str, list[Mapping[str, object]]]:
    try:
        return {
            relative: _jsonl_rows(
                payloads[relative],
                f"planning child recovery {relative}",
            )
            for relative in _JOURNAL_PATHS
        }
    except Stage6PlanningChildSourceRepairError as exc:
        raise _domain_error(
            "planning child recovery journal JSONL drifted",
            exc,
        ) from exc


def _validate_journal_snapshot(
    snapshot: _BoundRecoverySnapshot,
    *,
    payloads: Mapping[str, bytes],
    continuation: ValidatedPlanningChildContinuation | None,
    launch_bindings: _LaunchBindings | None,
) -> tuple[
    Mapping[str, object],
    PlanningChildResumeCursor | None,
    Mapping[str, object] | None,
    Mapping[str, Mapping[str, object]],
    Mapping[str, Mapping[str, object]],
    tuple[tuple[str, Mapping[str, object]], ...],
]:
    rows = _journal_rows(payloads)
    parent = snapshot.parent
    config_bound = _require_bound_binding(
        snapshot,
        parent.immutable_inputs["config"],
        label="parent config",
    )
    try:
        effective_config = parse_planning_effective_config_bytes(
            config_bound.payload
        )
        config = effective_config.base_config
        transactions = build_planning_child_transactions(config)
    except (Stage6PlanningWarmStartError, TypeError, ValueError) as exc:
        raise _domain_error(
            "planning child recovery config drifted",
            exc,
        ) from exc
    from lunar_exploration_ppo.workflows.stage6 import (
        Stage6StateJournal,
        Stage6WorkflowError,
    )

    try:
        receipts = list(
            CheckpointReceiptIndex.verify_snapshot_bytes(
                payloads["checkpoints/index.jsonl"]
            )
        )
        job_rows = list(
            Stage6StateJournal.verify_snapshot_bytes(
                payloads["job-state.jsonl"]
            )
        )
    except (StandardTrainingError, Stage6WorkflowError) as exc:
        raise _domain_error(
            "planning child recovery durable journal drifted",
            exc,
        ) from exc
    training_rows = rows["training_metrics.jsonl"]
    validation_rows = rows["validation_metrics.jsonl"]
    resource_rows = rows["resource_audit.jsonl"]
    if not receipts or len(receipts) > len(transactions):
        raise Stage6PlanningChildRecoveryError(
            "planning child recovery receipt count drifted"
        )

    epochs = _lineage_epochs(parent, continuation)
    checkpoint_lineage_templates = _checkpoint_lineage_templates(
        parent,
        continuation=continuation,
        launch_bindings=launch_bindings,
        config=config,
        effective_config_bytes=config_bound.payload,
    )
    origin_immutable = _require_mapping(
        _plain_json(parent.origin_immutable_bindings),
        "planning child recovery origin immutable bindings",
    )
    parent_current_immutable = _require_mapping(
        _plain_json(parent.current_immutable_bindings),
        "planning child recovery parent immutable bindings",
    )
    immutable_epochs: list[tuple[str, Mapping[str, object]]] = [
        (
            transactions[0].key,
            origin_immutable,
        ),
        (
            transactions[
                epochs[1].first_update - epochs[0].first_update
            ].key,
            parent_current_immutable,
        ),
    ]
    current_immutable: Mapping[str, object] = (
        parent_current_immutable
    )
    if continuation is not None and launch_bindings is not None:
        current_immutable = _require_mapping(
            _plain_json(launch_bindings.current_immutable_bindings),
            "planning child recovery continuation immutable bindings",
        )
        continuation_index = (
            epochs[2].first_update - epochs[0].first_update
        )
        immutable_epochs.append(
            (
                transactions[continuation_index].key,
                current_immutable,
            )
        )
    stable_count = len(training_rows)
    try:
        completed = verify_journal_checkpoint_bindings(
            job_rows,
            transactions,
            receipts,
            current_immutable,
            immutable_binding_epochs=tuple(immutable_epochs),
        )
        independently_completed = (
            completed_transaction_keys_from_journal(
                job_rows,
                transactions,
            )
        )
        if (
            completed != independently_completed
            or not 0 < stable_count <= len(transactions)
            or not (
                stable_count
                <= len(completed)
                <= stable_count + 1
            )
            or not (
                stable_count
                <= len(receipts)
                <= stable_count + 1
            )
            or len(completed) > len(receipts)
        ):
            raise StandardTrainingError(
                "bounded accepted journal joins are not exact"
            )
        validate_standard_training_validation_rows(
            config=config,
            transactions=transactions[:stable_count],
            receipt_rows=receipts[:stable_count],
            training_rows=training_rows,
            validation_rows=validation_rows,
            planning_warm_start=True,
        )
        resource_validation = validate_resource_lifecycle_rows(
            resource_rows,
            require_terminal=False,
        )
    except (
        StandardTrainingError,
        ResourceLifecycleError,
    ) as exc:
        raise _domain_error(
            "planning child recovery production journal validation failed",
            exc,
        ) from exc

    accepted_resource_rows = [
        row
        for row in resource_rows
        if row.get("phase") == "accepted"
        and row.get("kind") == "update"
    ]
    accepted_count = len(accepted_resource_rows)
    if (
        accepted_count not in {stable_count, stable_count + 1}
        or len(receipts) > accepted_count
        or accepted_count > len(transactions)
    ):
        raise Stage6PlanningChildRecoveryError(
            "planning child recovery resource acceptance join drifted"
        )
    for index, accepted in enumerate(accepted_resource_rows):
        transaction = transactions[index]
        checkpoint = _require_mapping(
            accepted.get("checkpoint"),
            "planning child recovery accepted checkpoint",
        )
        if (
            accepted.get("transaction_key") != transaction.key
            or accepted.get("seed") != transaction.seed
            or accepted.get("update") != transaction.update
            or type(accepted.get("attempt")) is not int
            or int(accepted["attempt"]) <= 0
            or checkpoint.get("transaction_key") != transaction.key
            or checkpoint.get("seed") != transaction.seed
            or checkpoint.get("update") != transaction.update
        ):
            raise Stage6PlanningChildRecoveryError(
                "planning child recovery accepted schedule drifted"
            )
        if index < len(receipts):
            receipt = receipts[index]
            if any(
                checkpoint.get(name) != receipt.get(name)
                for name in (
                    "transaction_key",
                    "seed",
                    "update",
                    "checkpoint_sha256",
                    "complete_marker_sha256",
                    "policy_state_sha256",
                )
            ):
                raise Stage6PlanningChildRecoveryError(
                    "planning child recovery receipt/resource identity drifted"
                )

    accepted_ahead = accepted_count == stable_count + 1
    receipt_ahead = len(receipts) == stable_count + 1
    stable_job_row_count = sum(
        len(transaction.commit_states)
        for transaction in transactions[:stable_count]
    )
    job_suffix = job_rows[stable_job_row_count:]
    journal_ahead = bool(job_suffix)
    if (
        (receipt_ahead and not accepted_ahead)
        or (journal_ahead and not receipt_ahead)
        or (
            len(completed) == stable_count + 1
            and not journal_ahead
        )
    ):
        raise Stage6PlanningChildRecoveryError(
            "planning child recovery crash suffix ordering drifted"
        )

    current_transaction = (
        transactions[stable_count]
        if stable_count < len(transactions)
        else None
    )
    stable_keys = {
        transaction.key
        for transaction in transactions[:stable_count]
    }
    current_resource_rows = [
        row
        for row in resource_rows
        if row.get("phase") in {"pre", "post", "accepted"}
        and row.get("transaction_key") not in stable_keys
    ]
    if current_transaction is None:
        if current_resource_rows or accepted_ahead or receipt_ahead:
            raise Stage6PlanningChildRecoveryError(
                "planning child recovery data exists after terminal prefix"
            )
        current_attempt = None
        current_phases: tuple[object, ...] = ()
    else:
        attempts = {
            row.get("attempt") for row in current_resource_rows
        }
        current_phases = tuple(
            row.get("phase") for row in current_resource_rows
        )
        if (
            any(
                row.get("transaction_key")
                != current_transaction.key
                or row.get("seed") != current_transaction.seed
                or row.get("update") != current_transaction.update
                or row.get("kind") != "update"
                for row in current_resource_rows
            )
            or current_phases
            not in {
                (),
                ("pre",),
                ("pre", "post"),
                ("pre", "post", "accepted"),
            }
            or (
                current_resource_rows
                and (
                    len(attempts) != 1
                    or type(next(iter(attempts))) is not int
                    or int(next(iter(attempts))) <= 0
                )
            )
            or accepted_ahead
            != (current_phases == ("pre", "post", "accepted"))
        ):
            raise Stage6PlanningChildRecoveryError(
                "planning child recovery transaction suffix drifted"
            )
        current_attempt = (
            int(next(iter(attempts)))
            if current_resource_rows
            else None
        )

    last_receipt = receipts[stable_count - 1]
    last_resource = accepted_resource_rows[stable_count - 1]
    last_update = _require_int(
        last_receipt.get("update"),
        "planning child recovery last accepted update",
    )
    latest_bundle = _checkpoint_bundle(
        snapshot,
        update=last_update,
    )
    expected_lineage = _expected_checkpoint_lineage(
        epochs,
        checkpoint_lineage_templates,
        update=last_update,
    )
    try:
        inspected = inspect_complete_checkpoint_snapshot(
            directory=(
                snapshot.stage_root
                / "checkpoints"
                / f"seed-{parent.seed}"
                / f"update-{last_update:08d}"
            ),
            update_step=last_update,
            schema_version=config.checkpoint.schema_version,
            member_names=tuple(sorted(_CHECKPOINT_MEMBERS)),
            checkpoint_bytes=latest_bundle["checkpoint.pt"].payload,
            manifest_bytes=latest_bundle["manifest.json"].payload,
            complete_bytes=latest_bundle["complete.json"].payload,
            expected_config_sha256=_sha256(config_bound.payload),
            expected_lineage=expected_lineage,
            expected_safety_contract=SafetyContract.from_stage6_config(
                config
            ),
        )
    except CheckpointError as exc:
        raise _domain_error(
            "planning child recovery latest checkpoint drifted",
            exc,
        ) from exc
    checkpoint_identity = _require_mapping(
        last_resource.get("checkpoint"),
        "planning child recovery resource checkpoint",
    )
    if (
        inspected.checkpoint_sha256
        != last_receipt.get("checkpoint_sha256")
        or inspected.policy_state_sha256
        != last_receipt.get("policy_state_sha256")
        or latest_bundle["complete.json"].sha256
        != last_receipt.get("complete_marker_sha256")
        or checkpoint_identity.get("checkpoint_sha256")
        != inspected.checkpoint_sha256
        or checkpoint_identity.get("complete_marker_sha256")
        != latest_bundle["complete.json"].sha256
        or checkpoint_identity.get("policy_state_sha256")
        != inspected.policy_state_sha256
    ):
        raise Stage6PlanningChildRecoveryError(
            "planning child recovery checkpoint acceptance drifted"
        )

    last_accepted_segment = _require_int(
        last_resource.get("segment_index"),
        "planning child recovery last accepted segment",
    )
    active_segment = _require_int(
        resource_validation.get("active_segment_index"),
        "planning child recovery active segment",
    )
    if active_segment not in {
        last_accepted_segment,
        last_accepted_segment + 1,
    }:
        raise Stage6PlanningChildRecoveryError(
            "planning child recovery resource segment tail drifted"
        )
    final_transaction = transactions[-1]
    expected_current_segment = (
        last_accepted_segment
        if (
            current_transaction is not None
            and current_transaction.key == final_transaction.key
        )
        else last_accepted_segment + 1
    )
    if current_resource_rows and (
        any(
            row.get("segment_index") != active_segment
            for row in current_resource_rows
        )
        or active_segment != expected_current_segment
    ):
        raise Stage6PlanningChildRecoveryError(
            "planning child recovery pending segment drifted"
        )

    crash_suffix: dict[str, object] | None = None
    crash_checkpoint: Mapping[str, object] | None = None
    if accepted_ahead:
        if current_transaction is None or current_attempt is None:
            raise Stage6PlanningChildRecoveryError(
                "planning child recovery accepted suffix is incomplete"
            )
        crash_resource = accepted_resource_rows[stable_count]
        crash_checkpoint = _require_mapping(
            crash_resource.get("checkpoint"),
            "planning child recovery crash checkpoint",
        )
        crash_bundle = _checkpoint_bundle(
            snapshot,
            update=current_transaction.update,
        )
        crash_lineage = _expected_checkpoint_lineage(
            epochs,
            checkpoint_lineage_templates,
            update=current_transaction.update,
        )
        try:
            crash_inspected = inspect_complete_checkpoint_snapshot(
                directory=(
                    snapshot.stage_root
                    / "checkpoints"
                    / f"seed-{parent.seed}"
                    / f"update-{current_transaction.update:08d}"
                ),
                update_step=current_transaction.update,
                schema_version=config.checkpoint.schema_version,
                member_names=tuple(sorted(_CHECKPOINT_MEMBERS)),
                checkpoint_bytes=crash_bundle["checkpoint.pt"].payload,
                manifest_bytes=crash_bundle["manifest.json"].payload,
                complete_bytes=crash_bundle["complete.json"].payload,
                expected_config_sha256=_sha256(config_bound.payload),
                expected_lineage=crash_lineage,
                expected_safety_contract=(
                    SafetyContract.from_stage6_config(config)
                ),
            )
        except CheckpointError as exc:
            raise _domain_error(
                "planning child recovery crash checkpoint drifted",
                exc,
            ) from exc
        if (
            crash_checkpoint.get("checkpoint_sha256")
            != crash_inspected.checkpoint_sha256
            or crash_checkpoint.get("complete_marker_sha256")
            != crash_bundle["complete.json"].sha256
            or crash_checkpoint.get("policy_state_sha256")
            != crash_inspected.policy_state_sha256
        ):
            raise Stage6PlanningChildRecoveryError(
                "planning child recovery crash checkpoint acceptance drifted"
            )
        if receipt_ahead:
            crash_receipt = receipts[stable_count]
            if any(
                crash_receipt.get(name)
                != crash_checkpoint.get(name)
                for name in (
                    "transaction_key",
                    "seed",
                    "update",
                    "checkpoint_sha256",
                    "complete_marker_sha256",
                    "policy_state_sha256",
                )
            ):
                raise Stage6PlanningChildRecoveryError(
                    "planning child recovery crash receipt drifted"
                )
        if journal_ahead:
            crash_boundary = "journal"
        elif receipt_ahead:
            crash_boundary = "receipt"
        elif snapshot.ahead_checkpoint_published:
            crash_boundary = "checkpoint"
        else:
            crash_boundary = "accepted"
        crash_suffix = {
            "schema_version": _CRASH_SUFFIX_SCHEMA,
            "boundary": crash_boundary,
            "transaction_key": current_transaction.key,
            "seed": current_transaction.seed,
            "update": current_transaction.update,
            "attempt": current_attempt,
            "checkpoint": _plain_json(crash_checkpoint),
        }
    elif current_phases == ("pre", "post"):
        if current_transaction is None or current_attempt is None:
            raise Stage6PlanningChildRecoveryError(
                "planning child recovery post suffix is incomplete"
            )
        crash_suffix = {
            "schema_version": _CRASH_SUFFIX_SCHEMA,
            "boundary": "post",
            "transaction_key": current_transaction.key,
            "seed": current_transaction.seed,
            "update": current_transaction.update,
            "attempt": current_attempt,
            "checkpoint": None,
        }

    effective_index = stable_count + (1 if accepted_ahead else 0)
    has_remaining_transaction = effective_index < len(transactions)
    next_transaction = (
        transactions[effective_index]
        if has_remaining_transaction
        else None
    )
    pending_pre_attempt = (
        current_attempt
        if current_phases in {("pre",), ("pre", "post")}
        else None
    )
    effective_last_update = (
        current_transaction.update
        if accepted_ahead and current_transaction is not None
        else last_update
    )
    cursor = (
        PlanningChildResumeCursor(
            last_accepted_update=effective_last_update,
            next_update=next_transaction.update,
            next_attempt=(
                pending_pre_attempt + 1
                if pending_pre_attempt is not None
                else 1
            ),
            next_transaction_key=next_transaction.key,
            next_resource_segment_index=active_segment + 1,
            latest_complete_checkpoint_update=effective_last_update,
            pending_pre_attempt=pending_pre_attempt,
        )
        if next_transaction is not None
        else None
    )
    latest_members = {
        name: {
            "path": bound.path.as_posix(),
            "size_bytes": bound.size_bytes,
            "sha256": bound.sha256,
        }
        for name, bound in sorted(latest_bundle.items())
    }
    accepted_anchor = {
        "schema_version": _ANCHOR_SCHEMA,
        "last_accepted_update": last_update,
        "last_accepted_transaction_key": _require_string(
            last_receipt.get("transaction_key"),
            "planning child recovery accepted transaction key",
        ),
        "accepted_attempt": _require_int(
            last_resource.get("attempt"),
            "planning child recovery accepted attempt",
        ),
        "accepted_resource_segment_index": (
            last_accepted_segment
        ),
        "latest_complete_checkpoint": {
            "update": last_update,
            "checkpoint_sha256": inspected.checkpoint_sha256,
            "manifest_sha256": inspected.manifest_sha256,
            "complete_marker_sha256": (
                latest_bundle["complete.json"].sha256
            ),
            "policy_state_sha256": inspected.policy_state_sha256,
            "members": latest_members,
        },
    }
    journal_prefixes = {
        relative: MappingProxyType(
            {
                "path": relative,
                "size_bytes": len(payloads[relative]),
                "sha256": _sha256(payloads[relative]),
                "line_count": payloads[relative].count(b"\n"),
            }
        )
        for relative in _JOURNAL_PATHS
    }
    return (
        MappingProxyType(accepted_anchor),
        cursor,
        (
            None
            if crash_suffix is None
            else MappingProxyType(crash_suffix)
        ),
        MappingProxyType(journal_prefixes),
        MappingProxyType(checkpoint_lineage_templates),
        tuple(immutable_epochs),
    )


def _snapshot_sha256(
    snapshot: _BoundRecoverySnapshot,
    *,
    launch_bindings: _LaunchBindings | None,
) -> str:
    unique = {
        bound.path: bound for bound in snapshot.inputs.values()
    }
    payload: dict[str, object] = {
        "files": [
            {
                "path": path.as_posix(),
                "size_bytes": bound.size_bytes,
                "sha256": bound.sha256,
            }
            for path, bound in sorted(
                unique.items(),
                key=lambda item: item[0].as_posix(),
            )
        ]
    }
    if launch_bindings is not None:
        payload["launch_bindings"] = {
            "execution_identity_sha256": _canonical_sha256(
                _plain_json(
                    launch_bindings.current_execution_identity
                )
            ),
            "verified_authorization_sha256": _canonical_sha256(
                _plain_json(
                    launch_bindings
                    .current_verified_review_authorization
                )
            ),
            "immutable_bindings_sha256": _canonical_sha256(
                _plain_json(
                    launch_bindings.current_immutable_bindings
                )
            ),
        }
    return _canonical_sha256(payload)


def _protected_updates(
    parent: ValidatedPlanningChildParent,
    continuation: ValidatedPlanningChildContinuation | None,
) -> tuple[int, ...]:
    updates: set[int] = set()
    for binding in parent.immutable_inputs.values():
        path = lexical_absolute(
            _require_string(
                binding.get("path"),
                "planning child recovery parent checkpoint path",
            )
        )
        if path.parent.name.startswith("update-"):
            try:
                updates.add(int(path.parent.name.removeprefix("update-")))
            except ValueError as exc:
                raise _domain_error(
                    "planning child recovery checkpoint reference drifted",
                    exc,
                ) from exc
    if continuation is not None:
        latest = _require_mapping(
            continuation.accepted_anchor.get(
                "latest_complete_checkpoint"
            ),
            "planning child recovery continuation checkpoint",
        )
        updates.add(
            _require_int(
                latest.get("update"),
                "planning child recovery continuation checkpoint update",
            )
        )
    return tuple(sorted(updates))


def _immutable_input_pin_requests(
    snapshot: _BoundRecoverySnapshot,
    *,
    continuation: ValidatedPlanningChildContinuation | None,
) -> tuple[tuple[str, Path], ...]:
    requests = list(snapshot.parent.input_pin_requests)
    if continuation is not None:
        requests.extend(continuation.input_pin_requests)
    requested: set[Path] = set()
    result: list[tuple[str, Path]] = []
    mutable = {
        snapshot.stage_root / relative
        for relative in _JOURNAL_PATHS
    }
    for label, path in requests:
        normalized = lexical_absolute(path)
        if normalized not in requested and normalized not in mutable:
            result.append((label, normalized))
            requested.add(normalized)
    return tuple(result)


def _validate_bound_recovery_snapshot(
    snapshot: _BoundRecoverySnapshot,
    *,
    launch_bindings: _LaunchBindings | None,
) -> _ValidatedRecoverySnapshot:
    parent = snapshot.parent
    for label, binding in parent.immutable_inputs.items():
        _require_bound_binding(
            snapshot,
            binding,
            label=f"parent immutable input {label}",
        )
    full_journals = {
        relative: snapshot.inputs[f"journal:{relative}"].payload
        for relative in _JOURNAL_PATHS
    }
    for relative, binding in parent.journal_prefixes.items():
        _require_prefix(
            full_journals[relative],
            binding,
            label=f"parent journal {relative}",
            size_key="prefix_size_bytes",
            sha_key="prefix_sha256",
        )

    continuation: ValidatedPlanningChildContinuation | None = None
    sealed_validation: tuple[
        Mapping[str, object],
        PlanningChildResumeCursor | None,
        Mapping[str, object] | None,
        Mapping[str, Mapping[str, object]],
        Mapping[str, Mapping[str, object]],
        tuple[tuple[str, Mapping[str, object]], ...],
    ] | None = None
    if snapshot.continuation_record is not None:
        if launch_bindings is None or snapshot.continuation_path is None:
            raise Stage6PlanningChildRecoveryError(
                "planning child recovery launch bindings are missing"
            )
        continuation_payload = snapshot.inputs[
            "continuation-artifact"
        ].payload
        parent_payload = snapshot.inputs["parent-artifact"].payload
        authorization_payload = snapshot.inputs[
            "continuation-input:current-authorization"
        ].payload
        review_payloads = {
            label: snapshot.inputs[
                f"continuation-input:{label}"
            ].payload
            for label in (
                "implementation_report",
                "spec_rereview",
                "quality_rereview",
            )
        }
        try:
            continuation = (
                validate_planning_child_source_repair_continuation_bytes(
                    continuation_payload,
                    stage_root=snapshot.stage_root,
                    artifact_path=snapshot.continuation_path,
                    parent_payload=parent_payload,
                    current_authorization_payload=authorization_payload,
                    review_evidence_payloads=review_payloads,
                    current_execution_identity=(
                        _plain_json(
                            launch_bindings.current_execution_identity
                        )
                    ),
                    current_verified_review_authorization=(
                        _plain_json(
                            launch_bindings
                            .current_verified_review_authorization
                        )
                    ),
                    current_immutable_bindings=(
                        _plain_json(
                            launch_bindings.current_immutable_bindings
                        )
                    ),
                )
            )
        except Stage6PlanningChildSourceRepairContinuationError as exc:
            raise _domain_error(
                "planning child recovery continuation validation failed",
                exc,
            ) from exc
        sealed_journals = {
            relative: _require_prefix(
                full_journals[relative],
                continuation.journal_prefixes[relative],
                label=f"continuation journal {relative}",
                size_key="size_bytes",
                sha_key="sha256",
            )
            for relative in _JOURNAL_PATHS
        }
        sealed_validation = _validate_journal_snapshot(
            snapshot,
            payloads=sealed_journals,
            continuation=continuation,
            launch_bindings=launch_bindings,
        )
        sealed_anchor, _, _, sealed_prefixes, _, _ = sealed_validation
        if (
            _plain_json(sealed_anchor)
            != _plain_json(continuation.accepted_anchor)
            or _plain_json(sealed_prefixes)
            != _plain_json(continuation.journal_prefixes)
        ):
            raise Stage6PlanningChildRecoveryError(
                "planning child recovery continuation anchor drifted"
            )
    elif launch_bindings is not None:
        raise Stage6PlanningChildRecoveryError(
            "planning child recovery continuation is required for launch"
        )

    if (
        sealed_validation is not None
        and all(
            full_journals[relative] == sealed_journals[relative]
            for relative in _JOURNAL_PATHS
        )
    ):
        current_validation = sealed_validation
    else:
        current_validation = _validate_journal_snapshot(
            snapshot,
            payloads=full_journals,
            continuation=continuation,
            launch_bindings=launch_bindings,
        )
    (
        accepted_anchor,
        cursor,
        crash_suffix,
        journal_prefixes,
        checkpoint_lineage_templates,
        journal_binding_epochs,
    ) = current_validation
    epochs = _lineage_epochs(parent, continuation)
    return _ValidatedRecoverySnapshot(
        parent=parent,
        continuation=continuation,
        input_snapshot_sha256=_snapshot_sha256(
            snapshot,
            launch_bindings=launch_bindings,
        ),
        accepted_anchor=accepted_anchor,
        crash_suffix=crash_suffix,
        journal_prefixes=journal_prefixes,
        resume_cursor=cursor,
        lineage_epochs=epochs,
        checkpoint_lineage_templates=checkpoint_lineage_templates,
        journal_binding_epochs=journal_binding_epochs,
        protected_checkpoint_updates=_protected_updates(
            parent,
            continuation,
        ),
        input_pin_requests=_immutable_input_pin_requests(
            snapshot,
            continuation=continuation,
        ),
    )


def _require_snapshot_current(
    snapshot: _BoundRecoverySnapshot,
) -> None:
    unique = {
        bound.path: bound for bound in snapshot.inputs.values()
    }
    for path, bound in sorted(
        unique.items(),
        key=lambda item: item[0].as_posix(),
    ):
        try:
            current: SecureReadResult = secure_read_bytes(
                path,
                base=(
                    snapshot.stage_root
                    if (
                        path == snapshot.stage_root
                        or snapshot.stage_root in path.parents
                    )
                    else None
                ),
                label=f"planning child recovery currentness {path.name}",
            )
        except PathSecurityError as exc:
            raise _domain_error(
                "planning child recovery currentness check failed",
                exc,
            ) from exc
        if (
            current.payload != bound.payload
            or current.stat_identity != bound.secure_identity
        ):
            raise Stage6PlanningChildRecoveryError(
                "planning child recovery input changed before return"
            )


def _terminal_capability_from_receipt(
    *,
    evidence: object,
    snapshot: _BoundRecoverySnapshot,
    validated: _ValidatedRecoverySnapshot,
    launch_bindings: _LaunchBindings,
) -> PlanningChildRecoveryCapability:
    from lunar_exploration_ppo.workflows.stage6_terminal_recovery import (
        _validate_planning_child_recovery_semantic_binding,
        _validate_receipt_payload,
    )

    if validated.continuation is None:
        raise Stage6PlanningChildRecoveryError(
            "planning child terminal recovery continuation is missing"
        )
    read_bytes = getattr(evidence, "read_bytes", None)
    if not callable(read_bytes):
        raise Stage6PlanningChildRecoveryError(
            "planning child terminal evidence handle is missing"
        )
    receipt, _, _ = _validate_receipt_payload(
        read_bytes(
            "preterminal_acceptance.json",
            label="planning child terminal receipt",
        )
    )
    semantic = _require_mapping(
        receipt.get("semantic_verification"),
        "planning child terminal semantic verification",
    )
    binding = _validate_planning_child_recovery_semantic_binding(
        semantic.get("planning_child_recovery"),
        expected_schema=PLANNING_CHILD_RECOVERY_EVIDENCE_SCHEMA,
    )
    acceptance = _require_mapping(
        binding.get("acceptance_binding"),
        "planning child terminal capability acceptance",
    )
    historical_prefixes = _require_mapping(
        acceptance.get("journal_prefixes"),
        "planning child terminal historical journal prefixes",
    )
    if set(historical_prefixes) != set(_JOURNAL_PATHS):
        raise Stage6PlanningChildRecoveryError(
            "planning child terminal historical journal set drifted"
        )
    historical_payloads = {
        relative: _require_prefix(
            snapshot.inputs[f"journal:{relative}"].payload,
            _require_mapping(
                historical_prefixes.get(relative),
                f"planning child terminal historical {relative}",
            ),
            label=f"terminal historical journal {relative}",
            size_key="size_bytes",
            sha_key="sha256",
        )
        for relative in _JOURNAL_PATHS
    }
    (
        historical_anchor,
        historical_cursor,
        historical_crash_suffix,
        validated_historical_prefixes,
        historical_lineage_templates,
        historical_journal_epochs,
    ) = _validate_journal_snapshot(
        snapshot,
        payloads=historical_payloads,
        continuation=validated.continuation,
        launch_bindings=launch_bindings,
    )
    expected_epochs = [
        asdict(epoch) for epoch in validated.lineage_epochs
    ]
    expected_historical_cursor = (
        None
        if historical_cursor is None
        else asdict(historical_cursor)
    )
    protected_values = acceptance.get(
        "protected_checkpoint_updates"
    )
    if (
        not isinstance(protected_values, Sequence)
        or isinstance(protected_values, (str, bytes, bytearray))
    ):
        raise Stage6PlanningChildRecoveryError(
            "planning child terminal historical protected updates drifted"
        )
    historical_protected_updates = tuple(
        _require_int(
            update,
            "planning child terminal historical protected update",
        )
        for update in protected_values
    )
    if (
        not historical_protected_updates
        or historical_protected_updates
        != tuple(sorted(set(historical_protected_updates)))
    ):
        raise Stage6PlanningChildRecoveryError(
            "planning child terminal historical protected updates drifted"
        )
    expected_capability_payload = {
        **_plain_json(acceptance),
        "stage_root": validated.parent.stage_root.as_posix(),
        "input_pin_requests": [
            [label, path.as_posix()]
            for label, path in validated.input_pin_requests
        ],
    }
    if (
        binding.get("parent_artifact_sha256")
        != validated.parent.artifact_sha256
        or binding.get("continuation_artifact_sha256")
        != validated.continuation.artifact_sha256
        or acceptance.get("formal_run_id")
        != validated.parent.formal_run_id
        or acceptance.get("seed") != validated.parent.seed
        or acceptance.get("parent_artifact_sha256")
        != validated.parent.artifact_sha256
        or acceptance.get("continuation_artifact_sha256")
        != validated.continuation.artifact_sha256
        or _plain_json(acceptance.get("accepted_anchor"))
        != _plain_json(historical_anchor)
        or _plain_json(acceptance.get("lineage_epochs"))
        != expected_epochs
        or _plain_json(acceptance.get("resume_cursor"))
        != _plain_json(expected_historical_cursor)
        or _plain_json(acceptance.get("crash_suffix"))
        != _plain_json(historical_crash_suffix)
        or _plain_json(historical_prefixes)
        != _plain_json(validated_historical_prefixes)
        or _plain_json(
            acceptance.get("checkpoint_lineage_templates")
        )
        != _plain_json(historical_lineage_templates)
        or _plain_json(acceptance.get("journal_binding_epochs"))
        != _plain_json(historical_journal_epochs)
        or acceptance.get("current_execution_identity_sha256")
        != validated.continuation.current_execution_identity_sha256
        or acceptance.get("current_immutable_bindings_sha256")
        != validated.continuation.current_immutable_bindings_sha256
        or _plain_json(
            acceptance.get("current_verified_review_authorization")
        )
        != _plain_json(
            launch_bindings.current_verified_review_authorization
        )
        or binding.get("input_snapshot_sha256")
        != acceptance.get("input_snapshot_sha256")
        or _canonical_sha256(expected_capability_payload)
        != binding.get("capability_sha256")
    ):
        raise Stage6PlanningChildRecoveryError(
            "planning child terminal capability binding drifted"
        )
    return PlanningChildRecoveryCapability(
        formal_run_id=validated.parent.formal_run_id,
        seed=validated.parent.seed,
        stage_root=validated.parent.stage_root,
        parent_artifact_sha256=validated.parent.artifact_sha256,
        continuation_artifact_sha256=(
            validated.continuation.artifact_sha256
        ),
        input_snapshot_sha256=_require_sha256(
            binding.get("input_snapshot_sha256"),
            "planning child terminal input snapshot SHA",
        ),
        capability_sha256=_require_sha256(
            binding.get("capability_sha256"),
            "planning child terminal capability SHA",
        ),
        resume_cursor=None,
        lineage_epochs=validated.lineage_epochs,
        protected_checkpoint_updates=historical_protected_updates,
        input_pin_requests=validated.input_pin_requests,
        acceptance_binding=_deep_freeze(acceptance),
        terminal_complete=True,
    )


def _resolve_terminal_recovery_capability(
    *,
    snapshot: _BoundRecoverySnapshot,
    validated: _ValidatedRecoverySnapshot,
    base_capability: PlanningChildRecoveryCapability,
    launch_bindings: _LaunchBindings,
) -> PlanningChildRecoveryCapability:
    from lunar_exploration_ppo.workflows.stage6 import (
        Stage6WorkflowError,
        _verify_stage6_terminal_manifest_evidence,
        stage6_source_identity,
    )
    from lunar_exploration_ppo.workflows.stage6_terminal_recovery import (
        COMPLETE_STATES,
        MANIFEST_NAME,
        RECEIPT_NAME,
        TERMINAL_ARTIFACT_NAMES,
        TerminalRecoveryError,
        _capture_terminal_evidence_handle,
        detect_stage6_terminal_recovery,
    )

    repo_root = Path(__file__).resolve().parents[3]
    source_identity = stage6_source_identity(repo_root)
    try:
        with _capture_terminal_evidence_handle(
            snapshot.stage_root
        ) as evidence:
            paths = set(evidence.file_paths)
            receipt_present = RECEIPT_NAME in paths
            terminal_capability = (
                _terminal_capability_from_receipt(
                    evidence=evidence,
                    snapshot=snapshot,
                    validated=validated,
                    launch_bindings=launch_bindings,
                )
                if receipt_present
                else base_capability
            )

            def manifest_verifier(candidate: Path) -> Mapping[str, object]:
                return _verify_stage6_terminal_manifest_evidence(
                    candidate,
                    source_identity=source_identity,
                    repo_root=repo_root,
                    evidence_handle=evidence,
                    planning_child_recovery_capability=(
                        terminal_capability
                    ),
                )

            detection = detect_stage6_terminal_recovery(
                stage_root=snapshot.stage_root,
                manifest_verifier=manifest_verifier,
                evidence_handle=evidence,
                planning_child_recovery_capability=terminal_capability,
            )
            status = detection.get("status")
            if status == "invalid":
                raise Stage6PlanningChildRecoveryError(
                    "planning child terminal evidence is invalid: "
                    f"{detection.get('reason', '')}"
                )
            if status == "no_terminal":
                unexpected = paths.intersection(
                    {MANIFEST_NAME, *TERMINAL_ARTIFACT_NAMES}
                )
                if receipt_present or unexpected:
                    raise Stage6PlanningChildRecoveryError(
                        "planning child terminal evidence is partial"
                    )
                result = base_capability
            elif (
                status == "valid_terminal_recovery"
                and receipt_present
                and MANIFEST_NAME in paths
                and tuple(detection.get("phase_states", ()))
                == COMPLETE_STATES
            ):
                result = terminal_capability
            elif status == "valid_terminal_recovery" and receipt_present:
                result = replace(
                    terminal_capability,
                    terminal_complete=False,
                )
            elif (
                status == "valid_preterminal_recovery"
                and receipt_present
                and MANIFEST_NAME not in paths
            ):
                result = base_capability
            else:
                raise Stage6PlanningChildRecoveryError(
                    "planning child terminal evidence is partial"
                )
            _require_snapshot_current(snapshot)
            evidence.require_current(
                "planning child terminal evidence before capability return"
            )
            return result
    except Stage6PlanningChildRecoveryError:
        raise
    except (TerminalRecoveryError, Stage6WorkflowError) as exc:
        raise _domain_error(
            "planning child terminal recovery validation failed",
            exc,
        ) from exc


def inspect_planning_child_recovery_anchor(
    *,
    stage_root: Path,
    parent_artifact_path: Path,
) -> PlanningChildRecoveryAnchor:
    snapshot = _capture_recovery_snapshot(
        stage_root=stage_root,
        parent_artifact_path=parent_artifact_path,
        continuation_artifact_path=None,
    )
    validated = _validate_bound_recovery_snapshot(
        snapshot,
        launch_bindings=None,
    )
    _require_snapshot_current(snapshot)
    return PlanningChildRecoveryAnchor(
        formal_run_id=validated.parent.formal_run_id,
        seed=validated.parent.seed,
        stage_root=validated.parent.stage_root,
        parent_artifact_sha256=(
            validated.parent.artifact_sha256
        ),
        input_snapshot_sha256=validated.input_snapshot_sha256,
        accepted_anchor=_deep_freeze(validated.accepted_anchor),
        journal_prefixes=_deep_freeze(validated.journal_prefixes),
    )


def issue_planning_child_recovery_capability(
    *,
    stage_root: Path,
    parent_artifact_path: Path,
    continuation_artifact_path: Path | None,
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, Mapping[str, object]],
) -> PlanningChildRecoveryCapability:
    if continuation_artifact_path is None:
        raise Stage6PlanningChildRecoveryError(
            "planning child recovery continuation is required for launch"
        )
    launch = _LaunchBindings(
        current_execution_identity=_frozen_mapping(
            current_execution_identity,
            "planning child recovery execution identity",
        ),
        current_verified_review_authorization=_frozen_mapping(
            current_verified_review_authorization,
            "planning child recovery verified authorization",
        ),
        current_immutable_bindings=_frozen_mapping(
            current_immutable_bindings,
            "planning child recovery immutable bindings",
        ),
    )
    snapshot = _capture_recovery_snapshot(
        stage_root=stage_root,
        parent_artifact_path=parent_artifact_path,
        continuation_artifact_path=continuation_artifact_path,
    )
    validated = _validate_bound_recovery_snapshot(
        snapshot,
        launch_bindings=launch,
    )
    if validated.continuation is None:
        raise Stage6PlanningChildRecoveryError(
            "planning child recovery continuation was not validated"
        )
    acceptance = {
        "schema_version": _CAPABILITY_ACCEPTANCE_SCHEMA,
        "formal_run_id": validated.parent.formal_run_id,
        "seed": validated.parent.seed,
        "parent_artifact_sha256": (
            validated.parent.artifact_sha256
        ),
        "continuation_artifact_sha256": (
            validated.continuation.artifact_sha256
        ),
        "input_snapshot_sha256": validated.input_snapshot_sha256,
        "terminal_complete": False,
        "crash_suffix": (
            None
            if validated.crash_suffix is None
            else _plain_json(validated.crash_suffix)
        ),
        "resume_cursor": (
            None
            if validated.resume_cursor is None
            else asdict(validated.resume_cursor)
        ),
        "lineage_epochs": [
            asdict(epoch) for epoch in validated.lineage_epochs
        ],
        "checkpoint_lineage_templates": _plain_json(
            validated.checkpoint_lineage_templates
        ),
        "journal_binding_epochs": [
            [transaction_key, _plain_json(bindings)]
            for transaction_key, bindings
            in validated.journal_binding_epochs
        ],
        "current_execution_identity_sha256": (
            validated.continuation
            .current_execution_identity_sha256
        ),
        "current_immutable_bindings_sha256": (
            validated.continuation
            .current_immutable_bindings_sha256
        ),
        "current_verified_review_authorization": _plain_json(
            launch.current_verified_review_authorization
        ),
        "origin_verified_review_authorization": _plain_json(
            _require_mapping(
                _require_mapping(
                    validated.parent.record.get("origin"),
                    "planning child recovery parent origin",
                ).get("verified_review_authorization"),
                "planning child recovery origin review authorization",
            )
        ),
        "protected_checkpoint_updates": list(
            validated.protected_checkpoint_updates
        ),
        "accepted_anchor": _plain_json(validated.accepted_anchor),
        "journal_prefixes": _plain_json(validated.journal_prefixes),
    }
    capability_payload = {
        **acceptance,
        "stage_root": validated.parent.stage_root.as_posix(),
        "input_pin_requests": [
            [label, path.as_posix()]
            for label, path in validated.input_pin_requests
        ],
    }
    capability_sha256 = _canonical_sha256(capability_payload)
    base_capability = PlanningChildRecoveryCapability(
        formal_run_id=validated.parent.formal_run_id,
        seed=validated.parent.seed,
        stage_root=validated.parent.stage_root,
        parent_artifact_sha256=(
            validated.parent.artifact_sha256
        ),
        continuation_artifact_sha256=(
            validated.continuation.artifact_sha256
        ),
        input_snapshot_sha256=validated.input_snapshot_sha256,
        capability_sha256=capability_sha256,
        resume_cursor=validated.resume_cursor,
        lineage_epochs=validated.lineage_epochs,
        protected_checkpoint_updates=(
            validated.protected_checkpoint_updates
        ),
        input_pin_requests=validated.input_pin_requests,
        acceptance_binding=_deep_freeze(acceptance),
        terminal_complete=False,
    )
    return _resolve_terminal_recovery_capability(
        snapshot=snapshot,
        validated=validated,
        base_capability=base_capability,
        launch_bindings=launch,
    )


__all__ = [
    "PLANNING_CHILD_RECOVERY_EVIDENCE_SCHEMA",
    "PlanningChildLineageEpoch",
    "PlanningChildRecoveryAnchor",
    "PlanningChildRecoveryCapability",
    "PlanningChildResumeCursor",
    "Stage6PlanningChildRecoveryError",
    "inspect_planning_child_recovery_anchor",
    "issue_planning_child_recovery_capability",
]
