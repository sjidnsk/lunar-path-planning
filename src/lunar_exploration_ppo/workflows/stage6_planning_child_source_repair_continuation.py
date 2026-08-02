"""Static write-once successor codec for an accepted Stage 6 planning child."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Mapping

from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.durable_jsonl import RunLease, RunLeaseError
from lunar_exploration_ppo.utils.path_security import (
    PathSecurityError,
    lexical_absolute,
    require_plain_path,
    secure_read_bytes,
)
from lunar_exploration_ppo.workflows.stage6_planning_child_source_repair import (
    PLANNING_CHILD_SOURCE_REPAIR_NAME,
    Stage6PlanningChildSourceRepairError,
    _canonical_sha256,
    _file_binding,
    _json_mapping,
    _require_int,
    _require_mapping,
    _require_sha256,
    _require_string,
    _secure_payload,
    _sha256,
    _validate_current_authorization,
    validate_planning_child_source_repair_bytes,
)


PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_NAME: Final = (
    "planning-child-source-repair-continuation.json"
)
PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_SCHEMA: Final = (
    "stage6_planning_child_source_repair_continuation/v2"
)
PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_MODE: Final = (
    "write_once_reviewed_source_epoch_successor/v1"
)
PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_LINEAGE_SCHEMA: Final = (
    "stage6_planning_child_source_repair_continuation_lineage/v1"
)

FIRST_CONTINUATION_UPDATE: Final = 86

_EXPECTED_FIELDS: Final = frozenset(
    {
        "schema_version",
        "mode",
        "formal_run_id",
        "seed",
        "parent",
        "current",
        "review_evidence",
        "accepted_anchor",
        "journal_prefixes",
        "lineage_epoch",
        "created_at_utc",
        "canonical_sha256",
    }
)
_REVIEW_LABELS: Final = (
    "implementation_report",
    "spec_rereview",
    "quality_rereview",
)
_JOURNAL_PATHS: Final = frozenset(
    {
        "checkpoints/index.jsonl",
        "job-state.jsonl",
        "resource_audit.jsonl",
        "training_metrics.jsonl",
        "validation_metrics.jsonl",
    }
)


class Stage6PlanningChildSourceRepairContinuationError(RuntimeError):
    """The planning-child continuation artifact is invalid or drifted."""


@dataclass(frozen=True, slots=True)
class _ContinuationPublishInput:
    label: str
    path: Path
    base: Path | None
    payload: bytes
    size_bytes: int
    sha256: str
    stat_identity: tuple[int, int, int, int, int]
    link_count: int


_PUBLISH_CAPABILITY_ISSUER: Final = object()


@dataclass(frozen=True, slots=True)
class PlanningChildContinuationPublishCapability(Mapping[str, object]):
    """Immutable preview snapshot authorized for one leased publication."""

    _record: Mapping[str, object]
    _stage_root: Path
    _run_lease: RunLease
    _authorization_currentness: object
    _stage5_authority_currentness: object
    _bound_inputs: tuple[_ContinuationPublishInput, ...]
    _issuer: object

    def __getitem__(self, key: str) -> object:
        return self._record[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._record)

    def __len__(self) -> int:
        return len(self._record)

    def canonical_record(self) -> dict[str, object]:
        value = _plain_json_value(self._record)
        assert isinstance(value, dict)
        return value


def _as_continuation_error(
    label: str,
    exc: BaseException,
) -> Stage6PlanningChildSourceRepairContinuationError:
    del exc
    return Stage6PlanningChildSourceRepairContinuationError(label)


def _require_continuation_run_lease(
    *,
    stage_root: str | Path,
    run_lease: object,
    label: str,
) -> RunLease:
    root = lexical_absolute(stage_root)
    expected_path = lexical_absolute(root.parent / ".stage6.lease")
    if (
        type(run_lease) is not RunLease
        or run_lease.path != expected_path
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            f"{label} requires the exact current RunLease"
        )
    try:
        run_lease.require_current()
    except RunLeaseError as exc:
        raise _as_continuation_error(
            f"{label} requires the exact current RunLease",
            exc,
        ) from exc
    return run_lease


def _plain_mapping(value: object, label: str) -> dict[str, object]:
    try:
        return dict(_require_mapping(value, label))
    except Stage6PlanningChildSourceRepairError as exc:
        raise _as_continuation_error(f"{label} drifted", exc) from exc


def _plain_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _plain_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_plain_json_value(item) for item in value]
    return value


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


def _capture_publish_input(
    *,
    path: str | Path,
    label: str,
    size_bytes: int,
    sha256: str,
    base: Path | None,
) -> _ContinuationPublishInput:
    lexical = lexical_absolute(path)
    try:
        result = secure_read_bytes(
            lexical,
            base=base,
            label=label,
        )
    except (OSError, PathSecurityError) as exc:
        raise _as_continuation_error(
            f"{label} could not be captured for publish capability",
            exc,
        ) from exc
    if (
        len(result.payload) != size_bytes
        or _sha256(result.payload) != sha256
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            f"{label} drifted before publish capability issuance"
        )
    return _ContinuationPublishInput(
        label=label,
        path=lexical,
        base=base,
        payload=result.payload,
        size_bytes=size_bytes,
        sha256=sha256,
        stat_identity=result.stat_identity,
        link_count=result.link_count,
    )


def _require_publish_inputs_current(
    capability: PlanningChildContinuationPublishCapability,
) -> None:
    for bound in capability._bound_inputs:
        try:
            current = secure_read_bytes(
                bound.path,
                base=bound.base,
                label=bound.label,
            )
        except (OSError, PathSecurityError) as exc:
            raise _as_continuation_error(
                f"{bound.label} changed before continuation publication",
                exc,
            ) from exc
        if (
            current.payload != bound.payload
            or len(current.payload) != bound.size_bytes
            or _sha256(current.payload) != bound.sha256
            or current.stat_identity != bound.stat_identity
            or current.link_count != bound.link_count
        ):
            raise Stage6PlanningChildSourceRepairContinuationError(
                f"{bound.label} drifted before continuation publication"
            )


def _require_publish_currentness(
    owner: object,
    *,
    label: str,
) -> None:
    require_current = getattr(owner, "require_current", None)
    if not callable(require_current):
        raise Stage6PlanningChildSourceRepairContinuationError(
            f"{label} currentness owner is missing"
        )
    try:
        require_current(label)
    except RuntimeError as exc:
        raise _as_continuation_error(
            f"{label} changed before continuation publication",
            exc,
        ) from exc


def _v2_canonical_record(
    value: Mapping[str, object],
) -> dict[str, object]:
    record = dict(value)
    try:
        stored_sha256 = _require_sha256(
            record.pop("canonical_sha256", None),
            "planning child continuation canonical SHA",
        )
        if (
            set(record) != (_EXPECTED_FIELDS - {"canonical_sha256"})
            or record.get("schema_version")
            != PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_SCHEMA
            or record.get("mode")
            != PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_MODE
            or _canonical_sha256(record) != stored_sha256
        ):
            raise Stage6PlanningChildSourceRepairContinuationError(
                "planning child continuation v2 schema or canonical binding "
                "drifted"
            )
        record["canonical_sha256"] = stored_sha256
        return record
    except Stage6PlanningChildSourceRepairContinuationError:
        raise
    except (Stage6PlanningChildSourceRepairError, TypeError, ValueError) as exc:
        raise _as_continuation_error(
            "planning child continuation v2 canonical binding drifted",
            exc,
        ) from exc


def _v2_file_binding(
    value: object,
    *,
    label: str,
    extra_fields: tuple[str, ...] = (),
) -> dict[str, object]:
    binding = _plain_mapping(value, label)
    expected_fields = {"path", "sha256", "size_bytes"}
    expected_fields.update(extra_fields)
    if set(binding) != expected_fields:
        raise Stage6PlanningChildSourceRepairContinuationError(
            f"{label} binding schema drifted"
        )
    try:
        path = lexical_absolute(
            _require_string(binding.get("path"), f"{label} path")
        )
        size_bytes = _require_int(
            binding.get("size_bytes"),
            f"{label} size",
        )
        digest = _require_sha256(
            binding.get("sha256"),
            f"{label} SHA",
        )
    except Stage6PlanningChildSourceRepairError as exc:
        raise _as_continuation_error(f"{label} binding drifted", exc) from exc
    if size_bytes < 0:
        raise Stage6PlanningChildSourceRepairContinuationError(
            f"{label} size drifted"
        )
    result: dict[str, object] = {
        "path": path.as_posix(),
        "sha256": digest,
        "size_bytes": size_bytes,
    }
    for field in extra_fields:
        result[field] = binding[field]
    return result


def _v2_read_binding(
    value: object,
    *,
    label: str,
    base: Path | None = None,
) -> tuple[dict[str, object], Path, bytes]:
    binding = _v2_file_binding(value, label=label)
    try:
        path, payload = _secure_payload(
            binding["path"],
            label=label,
            base=base,
        )
    except Stage6PlanningChildSourceRepairError as exc:
        raise _as_continuation_error(f"{label} is unavailable", exc) from exc
    if (
        len(payload) != binding["size_bytes"]
        or _sha256(payload) != binding["sha256"]
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            f"{label} bytes drifted"
        )
    return binding, path, payload


def _bound_review_file(path: str | Path, *, label: str) -> dict[str, object]:
    try:
        plain_path, payload = _secure_payload(path, label=label)
    except Stage6PlanningChildSourceRepairError as exc:
        raise _as_continuation_error(f"{label} is unavailable", exc) from exc
    return _file_binding(plain_path, payload)


def _validate_review_evidence_authorization_bindings(
    review_evidence: Mapping[str, object],
    *,
    verified_authorization: Mapping[str, object],
    immutable_bindings: Mapping[str, object],
) -> None:
    for evidence_label, authorization_key in (
        ("spec_rereview", "spec_review_sha256"),
        ("quality_rereview", "quality_review_sha256"),
    ):
        binding = _plain_mapping(
            review_evidence.get(evidence_label),
            f"planning child continuation {evidence_label}",
        )
        try:
            evidence_sha256 = _require_sha256(
                binding.get("sha256"),
                f"planning child continuation {evidence_label} SHA",
            )
            authorization_sha256 = _require_sha256(
                verified_authorization.get(authorization_key),
                f"current authorization {authorization_key}",
            )
            immutable_sha256 = _require_sha256(
                immutable_bindings.get(authorization_key),
                f"current immutable {authorization_key}",
            )
        except Stage6PlanningChildSourceRepairError as exc:
            raise _as_continuation_error(
                f"planning child continuation {evidence_label} "
                "review binding is invalid",
                exc,
            ) from exc
        if not (
            evidence_sha256
            == authorization_sha256
            == immutable_sha256
        ):
            raise Stage6PlanningChildSourceRepairContinuationError(
                f"planning child continuation {evidence_label} review "
                "evidence authorization binding drifted"
            )


@dataclass(frozen=True, slots=True)
class ValidatedPlanningChildContinuation:
    formal_run_id: str
    seed: int
    artifact_path: Path | None
    artifact_sha256: str
    parent_artifact_sha256: str
    current_execution_identity_sha256: str
    current_authorization_binding: Mapping[str, object]
    current_immutable_bindings_sha256: str
    review_evidence: Mapping[str, Mapping[str, object]]
    accepted_anchor: Mapping[str, object]
    journal_prefixes: Mapping[str, Mapping[str, object]]
    lineage_epoch: Mapping[str, object]
    input_pin_requests: tuple[tuple[str, Path], ...]


def _v2_review_evidence(
    *,
    implementation_report_path: str | Path,
    spec_review_path: str | Path,
    quality_review_path: str | Path,
    verified_authorization: Mapping[str, object],
    immutable_bindings: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    evidence = {
        "implementation_report": _bound_review_file(
            implementation_report_path,
            label="planning child continuation implementation report",
        ),
        "spec_rereview": _bound_review_file(
            spec_review_path,
            label="planning child continuation spec rereview",
        ),
        "quality_rereview": _bound_review_file(
            quality_review_path,
            label="planning child continuation quality rereview",
        ),
    }
    _validate_review_evidence_authorization_bindings(
        evidence,
        verified_authorization=verified_authorization,
        immutable_bindings=immutable_bindings,
    )
    return evidence


def _v2_static_sections(
    record: Mapping[str, object],
    *,
    parent_artifact_sha256: str,
    current_execution_identity_sha256: str,
) -> tuple[
    dict[str, object],
    dict[str, Mapping[str, object]],
    dict[str, object],
]:
    accepted_anchor = _plain_mapping(
        record.get("accepted_anchor"),
        "planning child continuation accepted anchor",
    )
    journal_raw = _plain_mapping(
        record.get("journal_prefixes"),
        "planning child continuation journal prefixes",
    )
    if set(journal_raw) != _JOURNAL_PATHS:
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation journal prefix set drifted"
        )
    journal_prefixes: dict[str, Mapping[str, object]] = {}
    for relative, value in sorted(journal_raw.items()):
        binding = _plain_mapping(
            value,
            f"planning child continuation journal prefix {relative}",
        )
        try:
            size_bytes = _require_int(
                binding.get("size_bytes"),
                f"planning child continuation journal {relative} size",
            )
            line_count = _require_int(
                binding.get("line_count"),
                f"planning child continuation journal {relative} lines",
            )
            _require_sha256(
                binding.get("sha256"),
                f"planning child continuation journal {relative} SHA",
            )
        except Stage6PlanningChildSourceRepairError as exc:
            raise _as_continuation_error(
                f"planning child continuation journal {relative} drifted",
                exc,
            ) from exc
        if (
            set(binding)
            != {"path", "size_bytes", "sha256", "line_count"}
            or binding.get("path") != relative
            or size_bytes <= 0
            or line_count <= 0
        ):
            raise Stage6PlanningChildSourceRepairContinuationError(
                f"planning child continuation journal {relative} drifted"
            )
        journal_prefixes[relative] = binding

    lineage_epoch = _plain_mapping(
        record.get("lineage_epoch"),
        "planning child continuation lineage epoch",
    )
    if (
        set(lineage_epoch)
        != {
            "schema_version",
            "first_update",
            "parent_artifact_sha256",
            "current_execution_identity_sha256",
        }
        or lineage_epoch.get("schema_version")
        != PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_LINEAGE_SCHEMA
        or lineage_epoch.get("first_update")
        != FIRST_CONTINUATION_UPDATE
        or lineage_epoch.get("parent_artifact_sha256")
        != parent_artifact_sha256
        or lineage_epoch.get("current_execution_identity_sha256")
        != current_execution_identity_sha256
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation lineage epoch drifted"
        )
    return accepted_anchor, journal_prefixes, lineage_epoch


def build_planning_child_source_repair_continuation_artifact(
    *,
    stage_root: str | Path,
    parent_path: str | Path,
    anchor: object,
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
    current_review_authorization_path: str | Path,
    implementation_report_path: str | Path,
    spec_review_path: str | Path,
    quality_review_path: str | Path,
    created_at_utc: str,
) -> dict[str, object]:
    """Build v2 only from a validated recovery anchor and static inputs."""

    from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
        PlanningChildRecoveryAnchor,
    )

    root = lexical_absolute(stage_root)
    parent = lexical_absolute(parent_path)
    if (
        not isinstance(anchor, PlanningChildRecoveryAnchor)
        or anchor.stage_root != root
        or parent != root / PLANNING_CHILD_SOURCE_REPAIR_NAME
        or anchor.formal_run_id != root.parent.name
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation recovery anchor drifted"
        )
    try:
        parent, parent_payload = _secure_payload(
            parent,
            label="planning child continuation parent",
            base=root,
        )
        validated_parent = validate_planning_child_source_repair_bytes(
            parent_payload,
            stage_root=root,
            artifact_path=parent,
        )
        authorization_path, authorization_payload = _secure_payload(
            current_review_authorization_path,
            label="planning child continuation current authorization",
        )
        current_identity_sha256, _ = _validate_current_authorization(
            formal_run_id=validated_parent.formal_run_id,
            execution_identity=current_execution_identity,
            verified_authorization=current_verified_review_authorization,
            immutable_bindings=current_immutable_bindings,
            authorization_path=authorization_path,
            authorization_payload=authorization_payload,
        )
    except Stage6PlanningChildSourceRepairError as exc:
        raise _as_continuation_error(
            "planning child continuation v2 immutable binding drifted",
            exc,
        ) from exc
    if (
        anchor.parent_artifact_sha256
        != validated_parent.artifact_sha256
        or anchor.formal_run_id != validated_parent.formal_run_id
        or anchor.seed != validated_parent.seed
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation anchor parent drifted"
        )

    evidence = _v2_review_evidence(
        implementation_report_path=implementation_report_path,
        spec_review_path=spec_review_path,
        quality_review_path=quality_review_path,
        verified_authorization=current_verified_review_authorization,
        immutable_bindings=current_immutable_bindings,
    )
    record: dict[str, object] = {
        "schema_version": (
            PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_SCHEMA
        ),
        "mode": PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_MODE,
        "formal_run_id": validated_parent.formal_run_id,
        "seed": validated_parent.seed,
        "parent": {
            **_file_binding(parent, parent_payload),
            "canonical_sha256": validated_parent.canonical_sha256,
        },
        "current": {
            "execution_identity_sha256": current_identity_sha256,
            "authorization": _file_binding(
                authorization_path,
                authorization_payload,
            ),
            "immutable_bindings_sha256": _canonical_sha256(
                current_immutable_bindings
            ),
        },
        "review_evidence": evidence,
        "accepted_anchor": _plain_json_value(anchor.accepted_anchor),
        "journal_prefixes": _plain_json_value(anchor.journal_prefixes),
        "lineage_epoch": {
            "schema_version": (
                PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_LINEAGE_SCHEMA
            ),
            "first_update": FIRST_CONTINUATION_UPDATE,
            "parent_artifact_sha256": validated_parent.artifact_sha256,
            "current_execution_identity_sha256": current_identity_sha256,
        },
        "created_at_utc": _require_string(
            created_at_utc,
            "planning child continuation created_at_utc",
        ),
    }
    record["canonical_sha256"] = _canonical_sha256(record)
    validate_planning_child_source_repair_continuation_artifact(
        record,
        stage_root=root,
        current_execution_identity=current_execution_identity,
        current_verified_review_authorization=(
            current_verified_review_authorization
        ),
        current_immutable_bindings=current_immutable_bindings,
    )
    return record


def issue_planning_child_source_repair_continuation_publish_capability(
    artifact: Mapping[str, object],
    *,
    stage_root: str | Path,
    anchor: object,
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
    authorization_currentness: object,
    stage5_authority_currentness: object,
    effective_config_bytes: bytes,
    run_lease: RunLease,
) -> PlanningChildContinuationPublishCapability:
    """Bind one immutable preview to its exact lease and physical inputs."""

    from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
        PlanningChildRecoveryAnchor,
    )

    root = lexical_absolute(stage_root)
    lease = _require_continuation_run_lease(
        stage_root=root,
        run_lease=run_lease,
        label="planning child continuation publish capability",
    )
    if (
        type(anchor) is not PlanningChildRecoveryAnchor
        or anchor.stage_root != root
        or anchor.formal_run_id != root.parent.name
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation publish capability anchor drifted"
        )
    from lunar_exploration_ppo.workflows.stage6 import (
        FrozenStage5AuthorityHandle,
    )
    from lunar_exploration_ppo.workflows.stage6_review_authorization import (
        Stage6ReviewAuthorizationHandle,
    )

    if (
        type(authorization_currentness)
        is not Stage6ReviewAuthorizationHandle
        or type(stage5_authority_currentness)
        is not FrozenStage5AuthorityHandle
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation exact authorization and Stage5 "
            "owners are required"
        )
    if type(effective_config_bytes) is not bytes or not effective_config_bytes:
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation effective config bytes are missing"
        )
    try:
        authorization_record = authorization_currentness.canonical_record()
        stage5_identity = stage5_authority_currentness.identity
        authorization_repo_root = lexical_absolute(
            authorization_currentness._repo_root
        )
        authorization_config_path = lexical_absolute(
            authorization_currentness._config_path
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise _as_continuation_error(
            "planning child continuation owner binding drifted",
            exc,
        ) from exc
    if (
        _plain_json_value(authorization_record)
        != _plain_json_value(
            current_verified_review_authorization
        )
        or authorization_currentness._formal_run_id
        != root.parent.name
        or authorization_config_path
        != authorization_repo_root
        / "configs"
        / "ppo_highres_frontier_stage6_v1.json"
        or authorization_currentness._effective_config_bytes
        != effective_config_bytes
        or not isinstance(stage5_identity, Mapping)
        or stage5_identity.get("gate_sha256")
        != current_immutable_bindings.get("stage5_gate_sha256")
        or _sha256(effective_config_bytes)
        != current_execution_identity.get("config_sha256")
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation exact owner binding drifted"
        )

    record = _v2_canonical_record(artifact)
    validated = validate_planning_child_source_repair_continuation_artifact(
        record,
        stage_root=root,
        current_execution_identity=current_execution_identity,
        current_verified_review_authorization=(
            current_verified_review_authorization
        ),
        current_immutable_bindings=current_immutable_bindings,
    )
    if (
        validated.parent_artifact_sha256
        != anchor.parent_artifact_sha256
        or _plain_json_value(validated.accepted_anchor)
        != _plain_json_value(anchor.accepted_anchor)
        or _plain_json_value(validated.journal_prefixes)
        != _plain_json_value(anchor.journal_prefixes)
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation publish capability snapshot drifted"
        )

    inputs: list[_ContinuationPublishInput] = [
        _capture_publish_input(
            path=root / "config.json",
            label="planning child continuation publish effective config",
            size_bytes=len(effective_config_bytes),
            sha256=_sha256(effective_config_bytes),
            base=root,
        )
    ]
    parent_binding = _v2_file_binding(
        record.get("parent"),
        label="planning child continuation parent",
        extra_fields=("canonical_sha256",),
    )
    parent_path = lexical_absolute(str(parent_binding["path"]))
    if parent_path != root / PLANNING_CHILD_SOURCE_REPAIR_NAME:
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation publish parent path drifted"
        )
    inputs.append(
        _capture_publish_input(
            path=parent_path,
            label="planning child continuation publish parent",
            size_bytes=int(parent_binding["size_bytes"]),
            sha256=str(parent_binding["sha256"]),
            base=root,
        )
    )

    current = _plain_mapping(
        record.get("current"),
        "planning child continuation publish current",
    )
    authorization_binding = _v2_file_binding(
        current.get("authorization"),
        label="planning child continuation publish authorization",
    )
    inputs.append(
        _capture_publish_input(
            path=str(authorization_binding["path"]),
            label="planning child continuation publish authorization",
            size_bytes=int(authorization_binding["size_bytes"]),
            sha256=str(authorization_binding["sha256"]),
            base=None,
        )
    )

    reviews = _plain_mapping(
        record.get("review_evidence"),
        "planning child continuation publish review evidence",
    )
    for label in _REVIEW_LABELS:
        binding = _v2_file_binding(
            reviews.get(label),
            label=f"planning child continuation publish {label}",
        )
        inputs.append(
            _capture_publish_input(
                path=str(binding["path"]),
                label=f"planning child continuation publish {label}",
                size_bytes=int(binding["size_bytes"]),
                sha256=str(binding["sha256"]),
                base=None,
            )
        )

    latest_checkpoint = _plain_mapping(
        validated.accepted_anchor.get("latest_complete_checkpoint"),
        "planning child continuation publish accepted checkpoint",
    )
    checkpoint_members = _plain_mapping(
        latest_checkpoint.get("members"),
        "planning child continuation publish accepted checkpoint members",
    )
    if not checkpoint_members:
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation publish checkpoint members drifted"
        )
    for member_name, value in sorted(checkpoint_members.items()):
        binding = _v2_file_binding(
            value,
            label=(
                "planning child continuation publish checkpoint "
                f"{member_name}"
            ),
        )
        inputs.append(
            _capture_publish_input(
                path=str(binding["path"]),
                label=(
                    "planning child continuation publish checkpoint "
                    f"{member_name}"
                ),
                size_bytes=int(binding["size_bytes"]),
                sha256=str(binding["sha256"]),
                base=root,
            )
        )

    for relative in sorted(_JOURNAL_PATHS):
        binding = validated.journal_prefixes[relative]
        inputs.append(
            _capture_publish_input(
                path=root / relative,
                label=(
                    "planning child continuation publish journal "
                    f"{relative}"
                ),
                size_bytes=int(binding["size_bytes"]),
                sha256=str(binding["sha256"]),
                base=root,
            )
        )

    _require_publish_currentness(
        authorization_currentness,
        label="planning child continuation authorization",
    )
    _require_publish_currentness(
        stage5_authority_currentness,
        label="planning child continuation Stage5 authority",
    )
    lease.require_current()
    frozen_record = _deep_freeze(record)
    assert isinstance(frozen_record, Mapping)
    return PlanningChildContinuationPublishCapability(
        _record=frozen_record,
        _stage_root=root,
        _run_lease=lease,
        _authorization_currentness=authorization_currentness,
        _stage5_authority_currentness=stage5_authority_currentness,
        _bound_inputs=tuple(inputs),
        _issuer=_PUBLISH_CAPABILITY_ISSUER,
    )


def validate_planning_child_source_repair_continuation_bytes(
    payload: bytes,
    *,
    stage_root: str | Path,
    artifact_path: str | Path | None,
    parent_payload: bytes,
    current_authorization_payload: bytes,
    review_evidence_payloads: Mapping[str, bytes],
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
) -> ValidatedPlanningChildContinuation:
    """Validate v2 entirely from one caller-bound immutable snapshot."""

    if (
        type(payload) is not bytes
        or not payload
        or type(parent_payload) is not bytes
        or not parent_payload
        or type(current_authorization_payload) is not bytes
        or not current_authorization_payload
        or not isinstance(review_evidence_payloads, Mapping)
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation bound snapshot drifted"
        )
    root = lexical_absolute(stage_root)
    try:
        record = _v2_canonical_record(
            _json_mapping(
                payload,
                "planning child continuation bound artifact",
            )
        )
    except Stage6PlanningChildSourceRepairError as exc:
        raise _as_continuation_error(
            "planning child continuation bound artifact drifted",
            exc,
        ) from exc
    if ArtifactStore.canonical_json_bytes(record) != payload:
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation bound artifact is not canonical"
        )
    try:
        formal_run_id = _require_string(
            record.get("formal_run_id"),
            "planning child continuation formal run id",
        )
        seed = _require_int(
            record.get("seed"),
            "planning child continuation seed",
        )
    except Stage6PlanningChildSourceRepairError as exc:
        raise _as_continuation_error(
            "planning child continuation run identity drifted",
            exc,
        ) from exc
    if root.parent.name != formal_run_id or seed <= 0:
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation run identity drifted"
        )

    parent_binding = _v2_file_binding(
        record.get("parent"),
        label="planning child continuation parent",
        extra_fields=("canonical_sha256",),
    )
    parent_path = lexical_absolute(
        _require_string(
            parent_binding.get("path"),
            "planning child continuation parent path",
        )
    )
    if (
        parent_path != root / PLANNING_CHILD_SOURCE_REPAIR_NAME
        or len(parent_payload) != parent_binding["size_bytes"]
        or _sha256(parent_payload) != parent_binding["sha256"]
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation parent bytes drifted"
        )
    try:
        validated_parent = validate_planning_child_source_repair_bytes(
            parent_payload,
            stage_root=root,
            artifact_path=parent_path,
        )
    except Stage6PlanningChildSourceRepairError as exc:
        raise _as_continuation_error(
            "planning child continuation parent drifted",
            exc,
        ) from exc
    if (
        validated_parent.canonical_sha256
        != parent_binding["canonical_sha256"]
        or validated_parent.formal_run_id != formal_run_id
        or validated_parent.seed != seed
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation parent identity drifted"
        )

    current = _plain_mapping(
        record.get("current"),
        "planning child continuation current",
    )
    if set(current) != {
        "execution_identity_sha256",
        "authorization",
        "immutable_bindings_sha256",
    }:
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation current schema drifted"
        )
    authorization_binding = _v2_file_binding(
        current.get("authorization"),
        label="planning child continuation current authorization",
    )
    authorization_path = lexical_absolute(
        _require_string(
            authorization_binding.get("path"),
            "planning child continuation current authorization path",
        )
    )
    if (
        len(current_authorization_payload)
        != authorization_binding["size_bytes"]
        or _sha256(current_authorization_payload)
        != authorization_binding["sha256"]
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation authorization bytes drifted"
        )
    try:
        identity_sha256, _ = _validate_current_authorization(
            formal_run_id=formal_run_id,
            execution_identity=current_execution_identity,
            verified_authorization=current_verified_review_authorization,
            immutable_bindings=current_immutable_bindings,
            authorization_path=authorization_path,
            authorization_payload=current_authorization_payload,
        )
    except Stage6PlanningChildSourceRepairError as exc:
        raise _as_continuation_error(
            "planning child continuation authorization drifted",
            exc,
        ) from exc
    immutable_sha256 = _canonical_sha256(current_immutable_bindings)
    if (
        current.get("execution_identity_sha256") != identity_sha256
        or current.get("immutable_bindings_sha256") != immutable_sha256
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation current identity drifted"
        )

    review_raw = _plain_mapping(
        record.get("review_evidence"),
        "planning child continuation review evidence",
    )
    if (
        set(review_raw) != set(_REVIEW_LABELS)
        or set(review_evidence_payloads) != set(_REVIEW_LABELS)
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation review evidence set drifted"
        )
    review_evidence: dict[str, Mapping[str, object]] = {}
    review_paths: list[tuple[str, Path]] = []
    for label in _REVIEW_LABELS:
        binding = _v2_file_binding(
            review_raw.get(label),
            label=f"planning child continuation {label}",
        )
        review_payload = review_evidence_payloads[label]
        if (
            type(review_payload) is not bytes
            or len(review_payload) != binding["size_bytes"]
            or _sha256(review_payload) != binding["sha256"]
        ):
            raise Stage6PlanningChildSourceRepairContinuationError(
                f"planning child continuation {label} bytes drifted"
            )
        review_evidence[label] = binding
        review_paths.append(
            (
                label,
                lexical_absolute(
                    _require_string(
                        binding.get("path"),
                        f"planning child continuation {label} path",
                    )
                ),
            )
        )
    _validate_review_evidence_authorization_bindings(
        review_evidence,
        verified_authorization=current_verified_review_authorization,
        immutable_bindings=current_immutable_bindings,
    )
    accepted_anchor, journal_prefixes, lineage_epoch = (
        _v2_static_sections(
            record,
            parent_artifact_sha256=validated_parent.artifact_sha256,
            current_execution_identity_sha256=identity_sha256,
        )
    )

    resolved_path = (
        lexical_absolute(artifact_path)
        if artifact_path is not None
        else None
    )
    if (
        resolved_path is not None
        and resolved_path
        != root / PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_NAME
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation artifact path drifted"
        )

    requests = list(validated_parent.input_pin_requests)
    requested = {path for _label, path in requests}
    for label, path in (
        ("planning-child-recovery:continuation", resolved_path),
        (
            "planning-child-recovery:current-authorization",
            authorization_path,
        ),
    ):
        if path is not None and path not in requested:
            requests.append((label, path))
            requested.add(path)
    for label, path in review_paths:
        if path not in requested:
            requests.append(
                (f"planning-child-recovery:review:{label}", path)
            )
            requested.add(path)

    frozen_review = _deep_freeze(review_evidence)
    frozen_anchor = _deep_freeze(accepted_anchor)
    frozen_journals = _deep_freeze(journal_prefixes)
    frozen_lineage = _deep_freeze(lineage_epoch)
    frozen_authorization = _deep_freeze(authorization_binding)
    assert isinstance(frozen_review, Mapping)
    assert isinstance(frozen_anchor, Mapping)
    assert isinstance(frozen_journals, Mapping)
    assert isinstance(frozen_lineage, Mapping)
    assert isinstance(frozen_authorization, Mapping)
    return ValidatedPlanningChildContinuation(
        formal_run_id=formal_run_id,
        seed=seed,
        artifact_path=resolved_path,
        artifact_sha256=_sha256(payload),
        parent_artifact_sha256=validated_parent.artifact_sha256,
        current_execution_identity_sha256=identity_sha256,
        current_authorization_binding=frozen_authorization,
        current_immutable_bindings_sha256=immutable_sha256,
        review_evidence=frozen_review,
        accepted_anchor=frozen_anchor,
        journal_prefixes=frozen_journals,
        lineage_epoch=frozen_lineage,
        input_pin_requests=tuple(requests),
    )


def validate_planning_child_source_repair_continuation_artifact(
    artifact: Mapping[str, object],
    *,
    stage_root: str | Path,
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
    artifact_path: str | Path | None = None,
    artifact_sha256: str | None = None,
) -> ValidatedPlanningChildContinuation:
    """Read static bindings, then delegate to the bytes validator."""

    root = lexical_absolute(stage_root)
    record = _v2_canonical_record(artifact)
    payload = ArtifactStore.canonical_json_bytes(record)
    if (
        artifact_sha256 is not None
        and _require_sha256(
            artifact_sha256,
            "planning child continuation artifact SHA",
        )
        != _sha256(payload)
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation artifact file SHA drifted"
        )
    parent_binding = _v2_file_binding(
        record.get("parent"),
        label="planning child continuation parent",
        extra_fields=("canonical_sha256",),
    )
    parent_path = lexical_absolute(
        _require_string(
            parent_binding.get("path"),
            "planning child continuation parent path",
        )
    )
    try:
        _, parent_payload = _secure_payload(
            parent_path,
            label="planning child continuation parent",
            base=root,
        )
        current = _plain_mapping(
            record.get("current"),
            "planning child continuation current",
        )
        authorization_binding = _v2_file_binding(
            current.get("authorization"),
            label="planning child continuation current authorization",
        )
        _, authorization_payload = _secure_payload(
            authorization_binding["path"],
            label="planning child continuation current authorization",
        )
        review_raw = _plain_mapping(
            record.get("review_evidence"),
            "planning child continuation review evidence",
        )
        review_payloads: dict[str, bytes] = {}
        for label in _REVIEW_LABELS:
            binding = _v2_file_binding(
                review_raw.get(label),
                label=f"planning child continuation {label}",
            )
            _, review_payloads[label] = _secure_payload(
                binding["path"],
                label=f"planning child continuation {label}",
            )
    except Stage6PlanningChildSourceRepairError as exc:
        raise _as_continuation_error(
            "planning child continuation immutable input drifted",
            exc,
        ) from exc
    return validate_planning_child_source_repair_continuation_bytes(
        payload,
        stage_root=root,
        artifact_path=artifact_path,
        parent_payload=parent_payload,
        current_authorization_payload=authorization_payload,
        review_evidence_payloads=review_payloads,
        current_execution_identity=current_execution_identity,
        current_verified_review_authorization=(
            current_verified_review_authorization
        ),
        current_immutable_bindings=current_immutable_bindings,
    )


def load_planning_child_source_repair_continuation_artifact(
    path: str | Path,
    *,
    stage_root: str | Path,
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
) -> tuple[ValidatedPlanningChildContinuation, str]:
    """Load the canonical published v2 artifact."""

    root = lexical_absolute(stage_root)
    continuation = lexical_absolute(path)
    if (
        continuation
        != root / PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_NAME
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation path is not canonical"
        )
    try:
        continuation, payload = _secure_payload(
            continuation,
            label="planning child continuation artifact",
            base=root,
        )
        artifact = _json_mapping(
            payload,
            "planning child continuation artifact",
        )
    except Stage6PlanningChildSourceRepairError as exc:
        raise _as_continuation_error(
            "planning child continuation artifact is unavailable",
            exc,
        ) from exc
    if ArtifactStore.canonical_json_bytes(dict(artifact)) != payload:
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation file is not canonical JSON"
        )
    digest = _sha256(payload)
    return (
        validate_planning_child_source_repair_continuation_artifact(
            artifact,
            stage_root=root,
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=(
                current_verified_review_authorization
            ),
            current_immutable_bindings=current_immutable_bindings,
            artifact_path=continuation,
            artifact_sha256=digest,
        ),
        digest,
    )


def publish_planning_child_source_repair_continuation_artifact(
    capability: PlanningChildContinuationPublishCapability,
    *,
    stage_root: str | Path,
    output_path: str | Path,
    run_lease: RunLease,
) -> Path:
    """Atomically publish one canonical v2 successor; never overwrite."""

    if (
        type(capability) is not PlanningChildContinuationPublishCapability
        or capability._issuer is not _PUBLISH_CAPABILITY_ISSUER
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation exact publish capability is required"
        )
    if run_lease is not capability._run_lease:
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation publication requires the same "
            "RunLease as preview"
        )
    root = lexical_absolute(stage_root)
    if root != capability._stage_root:
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation publish capability root drifted"
        )
    lease = _require_continuation_run_lease(
        stage_root=root,
        run_lease=run_lease,
        label="planning child continuation publication",
    )
    output = lexical_absolute(output_path)
    if (
        output
        != root / PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_NAME
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation output path is not canonical"
        )
    record = _v2_canonical_record(capability.canonical_record())
    try:
        plain_root = require_plain_path(
            root,
            leaf_kind="directory",
            label="planning child continuation stage root",
        )
        require_plain_path(
            output,
            base=plain_root,
            allow_missing=True,
            leaf_kind="file",
            label="planning child continuation output",
        )
        payload = ArtifactStore.canonical_json_bytes(record)
        _require_publish_currentness(
            capability._authorization_currentness,
            label="planning child continuation authorization",
        )
        _require_publish_currentness(
            capability._stage5_authority_currentness,
            label="planning child continuation Stage5 authority",
        )
        _require_publish_inputs_current(capability)
        lease.require_current()
        published = ArtifactStore(plain_root).write_bytes_exclusive(
            PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_NAME,
            payload,
        )
        lease.require_current()
        published_path, observed = _secure_payload(
            published,
            label="published planning child continuation",
            base=plain_root,
        )
    except (
        OSError,
        PathSecurityError,
        FileExistsError,
        RunLeaseError,
        Stage6PlanningChildSourceRepairError,
    ) as exc:
        raise _as_continuation_error(
            "planning child continuation exclusive publication failed",
            exc,
        ) from exc
    if published_path != output or observed != payload:
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation publication verification failed"
        )
    return published_path


def planning_child_source_repair_chain_input_pin_requests(
    parent_path: str | Path,
    continuation_path: str | Path,
) -> tuple[tuple[str, Path], ...]:
    """Return pins from static parent/continuation bindings only."""

    parent = lexical_absolute(parent_path)
    continuation = lexical_absolute(continuation_path)
    if (
        parent.name != PLANNING_CHILD_SOURCE_REPAIR_NAME
        or continuation.name
        != PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_NAME
        or parent.parent != continuation.parent
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation input pin path drifted"
        )
    try:
        _, parent_payload = _secure_payload(
            parent,
            label="planning child continuation parent input pin",
            base=parent.parent,
        )
        validated_parent = validate_planning_child_source_repair_bytes(
            parent_payload,
            stage_root=parent.parent,
            artifact_path=parent,
        )
        _, continuation_payload = _secure_payload(
            continuation,
            label="planning child continuation input pin",
            base=parent.parent,
        )
        record = _v2_canonical_record(
            _json_mapping(
                continuation_payload,
                "planning child continuation input pin",
            )
        )
    except Stage6PlanningChildSourceRepairError as exc:
        raise _as_continuation_error(
            "planning child continuation input pin graph drifted",
            exc,
        ) from exc
    parent_binding = _v2_file_binding(
        record.get("parent"),
        label="planning child continuation parent",
        extra_fields=("canonical_sha256",),
    )
    if (
        parent_binding.get("sha256")
        != validated_parent.artifact_sha256
        or parent_binding.get("canonical_sha256")
        != validated_parent.canonical_sha256
    ):
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation parent input pin drifted"
        )

    requests = list(validated_parent.input_pin_requests)
    requested = {path for _label, path in requests}
    if continuation not in requested:
        requests.append(
            ("planning-child-recovery:continuation", continuation)
        )
        requested.add(continuation)
    current = _plain_mapping(
        record.get("current"),
        "planning child continuation current",
    )
    _, authorization_path, _ = _v2_read_binding(
        current.get("authorization"),
        label="planning child continuation current authorization",
    )
    if authorization_path not in requested:
        requests.append(
            (
                "planning-child-recovery:current-authorization",
                authorization_path,
            )
        )
        requested.add(authorization_path)
    review = _plain_mapping(
        record.get("review_evidence"),
        "planning child continuation review evidence",
    )
    if set(review) != set(_REVIEW_LABELS):
        raise Stage6PlanningChildSourceRepairContinuationError(
            "planning child continuation review evidence set drifted"
        )
    for label in _REVIEW_LABELS:
        _, path, _ = _v2_read_binding(
            review[label],
            label=f"planning child continuation review {label}",
        )
        if path not in requested:
            requests.append(
                (f"planning-child-recovery:review:{label}", path)
            )
            requested.add(path)
    return tuple(requests)


__all__ = [
    "FIRST_CONTINUATION_UPDATE",
    "PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_LINEAGE_SCHEMA",
    "PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_MODE",
    "PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_NAME",
    "PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_SCHEMA",
    "PlanningChildContinuationPublishCapability",
    "Stage6PlanningChildSourceRepairContinuationError",
    "ValidatedPlanningChildContinuation",
    "build_planning_child_source_repair_continuation_artifact",
    "issue_planning_child_source_repair_continuation_publish_capability",
    "load_planning_child_source_repair_continuation_artifact",
    "planning_child_source_repair_chain_input_pin_requests",
    "publish_planning_child_source_repair_continuation_artifact",
    "validate_planning_child_source_repair_continuation_artifact",
    "validate_planning_child_source_repair_continuation_bytes",
]
