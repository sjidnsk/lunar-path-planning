"""Stage 6 preterminal receipt and workload-free terminal commit recovery.

The module deliberately depends only on byte/path primitives.  In particular,
terminal detection and recovery never replay semantic verification or construct
runtime policy, optimizer, checkpoint, worker, collector, trainer, or evaluator
objects.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final

from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl, DurableJsonlError
from lunar_exploration_ppo.utils.path_security import (
    PathSecurityError,
    is_link_or_reparse,
    require_plain_path,
    secure_read_bytes,
)
from lunar_exploration_ppo.utils.resource_lifecycle import (
    ResourceLifecycleError,
    append_bound_resource_lifecycle_terminal,
    append_resource_segment_start,
    build_bound_terminal_resource_evidence,
    validate_resource_lifecycle_rows,
)
from lunar_exploration_ppo.utils.stage6_input_pinning import (
    Stage6InputPin,
    Stage6InputPinError,
    acquire_stage6_input_pin,
)


class TerminalRecoveryError(RuntimeError):
    """A preterminal receipt or terminal commit prefix failed closed."""


def _require_recovery_execution_capability(
    execution_capability: object,
    *,
    stage_root: str | Path,
    label: str,
) -> None:
    from lunar_exploration_ppo.workflows.stage6 import (
        Stage6WorkflowError,
        _require_stage6_execution_capability,
    )

    try:
        _require_stage6_execution_capability(
            execution_capability,
            label=label,
            stage_root=stage_root,
        )
    except Stage6WorkflowError as exc:
        raise TerminalRecoveryError(
            f"Stage 6 execution capability rejected {label}"
        ) from exc


def _guarded_formal_recovery_mutation(function):
    def guarded(*args: object, **kwargs: object):
        from lunar_exploration_ppo.workflows.stage6 import (
            Stage6WorkflowError,
            _stage6_execution_operation,
        )

        execution_capability = kwargs.get("execution_capability")
        stage_root = kwargs.get("stage_root")
        try:
            with _stage6_execution_operation(
                execution_capability,
                label=function.__name__,
                stage_root=stage_root,  # type: ignore[arg-type]
            ):
                return function(*args, **kwargs)
        except Stage6WorkflowError as exc:
            raise TerminalRecoveryError(
                "Stage 6 execution capability rejected "
                f"{function.__name__} operation"
            ) from exc

    guarded.__name__ = function.__name__
    guarded.__qualname__ = function.__qualname__
    guarded.__doc__ = function.__doc__
    guarded.__annotations__ = dict(function.__annotations__)
    guarded.__signature__ = inspect.signature(function)
    return guarded


RECEIPT_NAME: Final = "preterminal_acceptance.json"
MANIFEST_NAME: Final = "manifest.json"
RESOURCE_NAME: Final = "resource_audit.jsonl"
PHASE_NAME: Final = "phase-state.jsonl"
PLANNING_CHILD_SOURCE_REPAIR_NAME: Final = (
    "planning-child-source-repair.json"
)
PLANNING_CHILD_CONTINUATION_NAME: Final = (
    "planning-child-source-repair-continuation.json"
)
CLASSIC_SOURCE_REPAIR_NAMES: Final = frozenset(
    {
        "source-repair-amendment.json",
        "source-repair-continuation.json",
        "source-repair-supplement.json",
        "source-repair-closure.json",
        "source-repair-frontier-recovery.json",
        "source-repair-sensor-acceleration.json",
    }
)
RECEIPT_SCHEMA: Final = "stage6_preterminal_acceptance/v2"
SEMANTIC_SCHEMA: Final = "stage6_machine_acceptance_verification/v2"
EVIDENCE_BINDING_SCHEMA: Final = "stage6_preterminal_evidence_binding/v1"
TERMINAL_SCHEMA: Final = "stage6_terminal_resource_evidence/v4"
TERMINAL_RECEIPT_FIELD: Final = "preterminal_acceptance"
ABSENT_AUTHORITY_SHA256: Final = (
    "3057252298292ae9a47b0879812581277456d1e9d9c0995178967da7c929236d"
)
TERMINAL_ARTIFACT_NAMES: Final = (
    "summary.json",
    "routing.json",
    "standard_training_report.md",
    "standard_eval_report.md",
    "report.md",
)
PRETERMINAL_STATES: Final = (
    "preflight",
    "global_best_frozen",
    "final_test_running",
    "final_unseen_running",
    "baselines_running",
)
COMPLETE_STATES: Final = (
    *PRETERMINAL_STATES,
    "machine_passed",
    "awaiting_independent_review",
)
_SEMANTIC_FIELDS: Final = frozenset(
    {
        "schema_version",
        "passed",
        "state",
        "final_evaluation_count",
        "final_episode_count",
        "checkpoint_receipt_count",
        "global_best_policy_state_sha256",
        "evidence_binding",
    }
)
_PLANNING_CHILD_SEMANTIC_FIELDS: Final = (
    _SEMANTIC_FIELDS | {"planning_child_recovery"}
)
_IMMUTABLE_FIELDS: Final = frozenset(
    {
        "config_sha256",
        "source_set_sha256",
        "prospective_tree_sha256",
        "data_sha256",
        "environment_identity",
        "environment_sha256",
        "stage5_gate_sha256",
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
_COVERAGE_CACHE_IMMUTABLE_FIELDS: Final = frozenset(
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
_CURRENT_IMMUTABLE_FIELDS: Final = (
    _IMMUTABLE_FIELDS | _COVERAGE_CACHE_IMMUTABLE_FIELDS
)
_FORMAL_RUN_ID_PATTERN: Final = re.compile(
    r"s6-standard-single-r1-(?P<utc>\d{8}T\d{6}Z)"
)
_GIT_TREE_PATTERN: Final = re.compile(r"[0-9a-f]{40}")
_GLOBAL_FIELDS: Final = frozenset(
    {
        "schema_version",
        "record",
        "transaction_key",
        "checkpoint_sha256",
        "complete_marker_sha256",
        "policy_state_sha256",
    }
)
_GLOBAL_RECORD_FIELDS: Final = frozenset(
    {
        "seed",
        "update",
        "success_rate_under_fixed_step_budget",
        "mean_final_coverage",
        "checkpoint_ref",
    }
)
_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "semantic_verification",
        "preterminal_evidence_binding",
        "immutable_bindings",
        "global_checkpoint_identity",
        "terminal_artifacts",
        "preterminal_artifact_graph",
        "preterminal_directory_graph",
        "resource_audit_prefix",
        "phase_state_prefix",
    }
)
_EVIDENCE_BINDING_FIELDS: Final = frozenset(
    {"schema_version", "graph_sha256", "files", "directories"}
)
_EVIDENCE_FILE_FIELDS: Final = frozenset(
    {"path", "sha256", "size_bytes", "identity"}
)
_EVIDENCE_FILE_IDENTITY_FIELDS: Final = frozenset(
    {
        "device",
        "inode",
        "file_type",
        "size_bytes",
        "file_attributes",
        "link_count",
    }
)
_SUCCESS_PHASE_BINDING_FIELDS: Final = frozenset(
    {"preterminal_acceptance_sha256", "preterminal_evidence_graph_sha256"}
)
_TERMINAL_FIELDS: Final = frozenset(
    {
        "schema_version",
        "kind",
        "transaction_key",
        "phase",
        "accepted",
        "measurement_boundary",
        "segment_id",
        "segment_index",
        "prior_resource_audit",
        "segment_chain",
        "run_rss_lifecycle_peak_bytes",
        "resource",
        TERMINAL_RECEIPT_FIELD,
    }
)
_RESOURCE_FIELDS: Final = frozenset(
    {
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
    }
)
_DIRECTORY_IDENTITY_FIELDS: Final = frozenset(
    {
        "device",
        "inode",
        "mode",
        "file_attributes",
        "link_count",
    }
)


def _terminal_recovery_event(event: str) -> None:
    """Private deterministic crash seam; production behavior is a no-op."""

    del event


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _canonical_json(value: object) -> bytes:
    try:
        return ArtifactStore.canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise TerminalRecoveryError("value is not canonical JSON data") from exc


def _canonical_row(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TerminalRecoveryError("value is not canonical JSONL data") from exc


def _identity(payload: bytes) -> dict[str, object]:
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _validate_identity(value: object, *, label: str) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"sha256", "size_bytes"}
        or not _is_sha256(value.get("sha256"))
        or type(value.get("size_bytes")) is not int
        or int(value["size_bytes"]) < 0
    ):
        raise TerminalRecoveryError(f"{label} identity drifted")
    return {"sha256": value["sha256"], "size_bytes": value["size_bytes"]}


def _strict_json(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TerminalRecoveryError(f"{label} is not canonical UTF-8 JSON") from exc
    if not isinstance(value, dict) or _canonical_json(value) != payload:
        raise TerminalRecoveryError(f"{label} is not canonical UTF-8 JSON")
    return value


def _strict_jsonl(payload: bytes, *, label: str) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    try:
        lines = payload.splitlines(keepends=True)
        if b"".join(lines) != payload:
            raise ValueError("trailing JSONL bytes")
        for line in lines:
            value = json.loads(
                line.decode("utf-8"), parse_constant=_reject_json_constant
            )
            if not isinstance(value, dict) or _canonical_row(value) != line:
                raise ValueError("noncanonical JSONL row")
            rows.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TerminalRecoveryError(f"{label} is not canonical JSONL") from exc
    return tuple(rows)


def _require_stage_root(stage_root: str | Path) -> Path:
    try:
        return require_plain_path(
            stage_root,
            leaf_kind="directory",
            label="Stage 6 terminal recovery root",
        )
    except (OSError, PathSecurityError, TypeError, ValueError) as exc:
        raise TerminalRecoveryError("Stage 6 terminal recovery root is unsafe") from exc


def _read_plain(stage: Path, relative: str, *, label: str) -> bytes:
    path = stage / relative
    try:
        return secure_read_bytes(path, base=stage, label=label).payload
    except (OSError, PathSecurityError) as exc:
        raise TerminalRecoveryError(f"{label} is unsafe or unreadable") from exc


def _validate_evidence_repair_profile(
    stage: Path,
    *,
    file_paths: set[str],
    payload_reader: Callable[[str], bytes] | None = None,
) -> None:
    child_present = PLANNING_CHILD_SOURCE_REPAIR_NAME in file_paths
    classic = CLASSIC_SOURCE_REPAIR_NAMES.intersection(file_paths)
    if child_present and classic:
        raise TerminalRecoveryError(
            "Stage 6 planning child and classic source-repair evidence "
            "profiles are mutually exclusive"
        )
    if not child_present:
        return
    if "lineage_audit.json" not in file_paths:
        raise TerminalRecoveryError(
            "Stage 6 planning child evidence requires planning warm-start "
            "lineage"
        )
    payload = (
        payload_reader("lineage_audit.json")
        if payload_reader is not None
        else _read_plain(
            stage,
            "lineage_audit.json",
            label="Stage 6 planning child evidence lineage",
        )
    )
    lineage = _strict_json(
        payload,
        label="Stage 6 planning child evidence lineage",
    )
    if (
        lineage.get("schema_version") != "stage6_lineage_audit/v3"
        or not isinstance(lineage.get("planning_warm_start"), Mapping)
    ):
        raise TerminalRecoveryError(
            "Stage 6 planning child evidence requires planning warm-start "
            "lineage"
        )


def _validate_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise TerminalRecoveryError(f"{label} path drifted")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise TerminalRecoveryError(f"{label} path drifted")
    if relative.as_posix() != value:
        raise TerminalRecoveryError(f"{label} path drifted")
    return value


def _capture_graph(stage: Path, *, excluded: frozenset[str] = frozenset()) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    seen_casefold: set[str] = set()
    try:
        for root_value, directory_names, file_names in os.walk(
            stage, topdown=True, followlinks=False
        ):
            root = Path(root_value)
            require_plain_path(
                root,
                base=stage,
                leaf_kind="directory",
                label="Stage 6 preterminal graph directory",
            )
            for name in sorted(directory_names):
                child = root / name
                if is_link_or_reparse(child):
                    raise TerminalRecoveryError(
                        "preterminal artifact graph contains link or reparse directory"
                    )
                metadata = child.lstat()
                if not stat.S_ISDIR(metadata.st_mode):
                    raise TerminalRecoveryError(
                        "preterminal artifact graph contains special directory member"
                    )
            for name in sorted(file_names):
                path = root / name
                relative = path.relative_to(stage).as_posix()
                if relative in excluded:
                    continue
                _validate_relative_path(relative, label="preterminal artifact graph")
                if relative.casefold() in seen_casefold:
                    raise TerminalRecoveryError(
                        "preterminal artifact graph has a case-fold alias"
                    )
                seen_casefold.add(relative.casefold())
                if is_link_or_reparse(path):
                    raise TerminalRecoveryError(
                        "preterminal artifact graph contains link or reparse file"
                    )
                read = secure_read_bytes(
                    path,
                    base=stage,
                    label="Stage 6 preterminal artifact graph file",
                )
                artifacts.append({"path": relative, **_identity(read.payload)})
    except TerminalRecoveryError:
        raise
    except (OSError, PathSecurityError) as exc:
        raise TerminalRecoveryError(
            "preterminal artifact graph is unsafe or drifted"
        ) from exc
    artifacts.sort(key=lambda row: str(row["path"]))
    return artifacts


def _directory_identity(metadata: os.stat_result) -> dict[str, int]:
    return {
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "mode": int(metadata.st_mode),
        "file_attributes": int(getattr(metadata, "st_file_attributes", 0)),
        "link_count": int(metadata.st_nlink),
    }


def _validate_member_name(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or ":" in value
    ):
        raise TerminalRecoveryError(f"{label} member name drifted")
    return value


def _capture_directory_graph(
    stage: Path,
    *,
    excluded: frozenset[str] = frozenset(),
) -> list[dict[str, object]]:
    directories: list[dict[str, object]] = []
    seen_casefold: set[str] = set()
    try:
        for root_value, directory_names, file_names in os.walk(
            stage, topdown=True, followlinks=False
        ):
            directory_names.sort()
            file_names.sort()
            root = Path(root_value)
            require_plain_path(
                root,
                base=stage,
                leaf_kind="directory",
                label="Stage 6 preterminal directory graph member",
            )
            root_metadata_before = root.lstat()
            if is_link_or_reparse(root) or not stat.S_ISDIR(root_metadata_before.st_mode):
                raise TerminalRecoveryError(
                    "preterminal directory graph contains an unsafe directory"
                )
            root_relative = root.relative_to(stage).as_posix()
            directory_path = "." if root_relative == "." else _validate_relative_path(
                root_relative,
                label="preterminal directory graph",
            )
            members: list[dict[str, str]] = []
            member_casefold: set[str] = set()
            walked_members = [
                *((name, "directory") for name in directory_names),
                *((name, "file") for name in file_names),
            ]
            for name, walked_kind in sorted(walked_members):
                member_name = _validate_member_name(
                    name,
                    label="preterminal directory graph",
                )
                child = root / member_name
                relative = child.relative_to(stage).as_posix()
                _validate_relative_path(relative, label="preterminal directory graph")
                if is_link_or_reparse(child):
                    raise TerminalRecoveryError(
                        "preterminal directory graph contains link or reparse member"
                    )
                metadata = child.lstat()
                if stat.S_ISDIR(metadata.st_mode):
                    kind = "directory"
                elif stat.S_ISREG(metadata.st_mode):
                    kind = "file"
                else:
                    raise TerminalRecoveryError(
                        "preterminal directory graph contains special member"
                    )
                if kind != walked_kind:
                    raise TerminalRecoveryError(
                        "preterminal directory graph changed during traversal"
                    )
                if relative in excluded:
                    if kind != "file":
                        raise TerminalRecoveryError(
                            "reserved terminal artifact path is not a regular file"
                        )
                    continue
                if (
                    member_name.casefold() in member_casefold
                    or relative.casefold() in seen_casefold
                ):
                    raise TerminalRecoveryError(
                        "preterminal directory graph has a case-fold alias"
                    )
                member_casefold.add(member_name.casefold())
                seen_casefold.add(relative.casefold())
                members.append({"name": member_name, "kind": kind})
            root_metadata_after = root.lstat()
            if _directory_identity(root_metadata_before) != _directory_identity(
                root_metadata_after
            ):
                raise TerminalRecoveryError(
                    "preterminal directory identity changed during capture"
                )
            directories.append(
                {
                    "path": directory_path,
                    "identity": _directory_identity(root_metadata_after),
                    "members": members,
                }
            )
    except TerminalRecoveryError:
        raise
    except (OSError, PathSecurityError) as exc:
        raise TerminalRecoveryError(
            "preterminal directory graph is unsafe or drifted"
        ) from exc
    directories.sort(key=lambda row: str(row["path"]))
    return directories


def _validate_semantic_result(value: object) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or frozenset(value)
        not in {_SEMANTIC_FIELDS, _PLANNING_CHILD_SEMANTIC_FIELDS}
    ):
        raise TerminalRecoveryError("semantic verification result schema drifted")
    result = dict(value)
    if (
        result.get("schema_version")
        != SEMANTIC_SCHEMA
        or result.get("passed") is not True
        or result.get("state") != "awaiting_independent_review"
        or any(
            type(result.get(key)) is not int or int(result[key]) <= 0
            for key in (
                "final_evaluation_count",
                "final_episode_count",
                "checkpoint_receipt_count",
            )
        )
        or not _is_sha256(result.get("global_best_policy_state_sha256"))
    ):
        raise TerminalRecoveryError("semantic verification did not pass exactly")
    result["evidence_binding"] = _validate_evidence_binding(
        result.get("evidence_binding")
    )
    if "planning_child_recovery" in result:
        result["planning_child_recovery"] = (
            _validate_planning_child_recovery_semantic_binding(
                result["planning_child_recovery"]
            )
        )
    _canonical_json(result)
    return result


def _validate_immutable_bindings(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or frozenset(value) not in {
        _IMMUTABLE_FIELDS,
        _CURRENT_IMMUTABLE_FIELDS,
    }:
        raise TerminalRecoveryError("immutable binding schema drifted")
    bindings = dict(value)
    environment = bindings.get("environment_identity")
    if not isinstance(environment, Mapping):
        raise TerminalRecoveryError("immutable environment identity drifted")
    environment_value = dict(environment)
    environment_sha256 = hashlib.sha256(_canonical_json(environment_value)).hexdigest()
    formal_run_id = bindings.get("formal_run_id")
    formal_match = (
        _FORMAL_RUN_ID_PATTERN.fullmatch(formal_run_id)
        if isinstance(formal_run_id, str)
        else None
    )
    try:
        if formal_match is None:
            raise ValueError("formal run id pattern drifted")
        datetime.strptime(formal_match.group("utc"), "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise TerminalRecoveryError("immutable formal run id drifted") from exc
    reviewed_tree = bindings.get("reviewed_prospective_git_tree")
    if not isinstance(reviewed_tree, str) or _GIT_TREE_PATTERN.fullmatch(
        reviewed_tree
    ) is None:
        raise TerminalRecoveryError("immutable prospective git tree drifted")
    prospective_tree_sha256 = hashlib.sha256(
        reviewed_tree.encode("ascii")
    ).hexdigest()
    hash_fields = _IMMUTABLE_FIELDS - {
        "environment_identity",
        "formal_run_id",
        "reviewed_prospective_git_tree",
    }
    if (
        any(not _is_sha256(bindings.get(key)) for key in hash_fields)
        or bindings.get("environment_sha256") != environment_sha256
        or bindings.get("prospective_tree_sha256") != prospective_tree_sha256
    ):
        raise TerminalRecoveryError("immutable binding hash drifted")
    bindings["environment_identity"] = environment_value
    if set(bindings) == _CURRENT_IMMUTABLE_FIELDS and (
        not isinstance(bindings.get("coverage_cache_manifest_path"), str)
        or not bindings["coverage_cache_manifest_path"]
        or not _is_sha256(bindings.get("coverage_cache_manifest_sha256"))
        or type(bindings.get("coverage_cache_manifest_size_bytes")) is not int
        or int(bindings["coverage_cache_manifest_size_bytes"]) <= 0
        or not isinstance(bindings.get("coverage_cache_root"), str)
        or not bindings["coverage_cache_root"]
        or not _is_sha256(bindings.get("coverage_cache_entry_set_sha256"))
        or bindings.get("coverage_cache_runtime_mode")
        != "persistent_exact_manifest_read_only/v1"
        or not _is_sha256(bindings.get("coverage_cache_formal_audit_sha256"))
        or type(bindings.get("coverage_cache_formal_audit_size_bytes")) is not int
        or int(bindings["coverage_cache_formal_audit_size_bytes"]) <= 0
    ):
        raise TerminalRecoveryError("immutable coverage cache binding drifted")
    _canonical_json(bindings)
    return bindings


def _validate_global_checkpoint(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _GLOBAL_FIELDS:
        raise TerminalRecoveryError("global checkpoint identity schema drifted")
    result = dict(value)
    record = result.get("record")
    if not isinstance(record, Mapping) or set(record) != _GLOBAL_RECORD_FIELDS:
        raise TerminalRecoveryError("global checkpoint record schema drifted")
    record_value = dict(record)
    numeric = (
        record_value.get("success_rate_under_fixed_step_budget"),
        record_value.get("mean_final_coverage"),
    )
    if (
        result.get("schema_version") != "stage6_global_best/v1"
        or not isinstance(result.get("transaction_key"), str)
        or not result["transaction_key"]
        or any(
            not _is_sha256(result.get(key))
            for key in (
                "checkpoint_sha256",
                "complete_marker_sha256",
                "policy_state_sha256",
            )
        )
        or type(record_value.get("seed")) is not int
        or int(record_value["seed"]) <= 0
        or type(record_value.get("update")) is not int
        or int(record_value["update"]) <= 0
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            for item in numeric
        )
        or not isinstance(record_value.get("checkpoint_ref"), str)
        or not record_value["checkpoint_ref"]
    ):
        raise TerminalRecoveryError("global checkpoint identity drifted")
    result["record"] = record_value
    _canonical_json(result)
    return result


def _validate_terminal_artifacts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Mapping) or set(value) != set(TERMINAL_ARTIFACT_NAMES):
        raise TerminalRecoveryError("terminal artifact path set drifted")
    rows: list[dict[str, object]] = []
    for relative in TERMINAL_ARTIFACT_NAMES:
        payload = value[relative]
        if type(payload) is not bytes:
            raise TerminalRecoveryError("terminal artifact payload must be exact bytes")
        try:
            utf8 = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TerminalRecoveryError("terminal artifact is not UTF-8") from exc
        if relative in {"summary.json", "routing.json"}:
            _strict_json(payload, label=relative)
        rows.append({"path": relative, "utf8": utf8, **_identity(payload)})
    return rows


def _validate_artifact_rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != len(TERMINAL_ARTIFACT_NAMES):
        raise TerminalRecoveryError("receipt terminal artifact list drifted")
    rows: list[dict[str, object]] = []
    for expected_path, row_value in zip(TERMINAL_ARTIFACT_NAMES, value, strict=True):
        if (
            not isinstance(row_value, Mapping)
            or set(row_value) != {"path", "utf8", "sha256", "size_bytes"}
            or row_value.get("path") != expected_path
            or not isinstance(row_value.get("utf8"), str)
        ):
            raise TerminalRecoveryError("receipt terminal artifact row drifted")
        row = dict(row_value)
        identity = _validate_identity(
            {key: row[key] for key in ("sha256", "size_bytes")},
            label=expected_path,
        )
        payload = row["utf8"].encode("utf-8")
        if identity != _identity(payload):
            raise TerminalRecoveryError("receipt terminal artifact bytes drifted")
        if expected_path in {"summary.json", "routing.json"}:
            _strict_json(payload, label=expected_path)
        rows.append(row)
    return rows


def _validate_graph_rows(
    value: object,
    *,
    allow_reserved: bool = False,
    require_mutable_prefix: bool = True,
) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise TerminalRecoveryError("preterminal artifact graph drifted")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    reserved = {RECEIPT_NAME, MANIFEST_NAME, *TERMINAL_ARTIFACT_NAMES}
    for row_value in value:
        if not isinstance(row_value, Mapping) or set(row_value) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise TerminalRecoveryError("preterminal artifact graph row drifted")
        path = _validate_relative_path(
            row_value.get("path"), label="preterminal artifact graph"
        )
        identity = _validate_identity(
            {key: row_value[key] for key in ("sha256", "size_bytes")},
            label=path,
        )
        if path in seen or (not allow_reserved and path in reserved):
            raise TerminalRecoveryError("preterminal artifact graph path drifted")
        seen.add(path)
        rows.append({"path": path, **identity})
    if [row["path"] for row in rows] != sorted(seen):
        raise TerminalRecoveryError("preterminal artifact graph order drifted")
    if require_mutable_prefix and not {RESOURCE_NAME, PHASE_NAME}.issubset(seen):
        raise TerminalRecoveryError("preterminal mutable prefix artifacts are missing")
    return rows


def _validate_directory_identity(value: object, *, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _DIRECTORY_IDENTITY_FIELDS:
        raise TerminalRecoveryError(f"{label} identity schema drifted")
    identity = {key: value[key] for key in _DIRECTORY_IDENTITY_FIELDS}
    if (
        any(type(identity[key]) is not int for key in _DIRECTORY_IDENTITY_FIELDS)
        or int(identity["device"]) < 0
        or int(identity["inode"]) < 0
        or int(identity["file_attributes"]) < 0
        or int(identity["link_count"]) <= 0
        or not stat.S_ISDIR(int(identity["mode"]))
    ):
        raise TerminalRecoveryError(f"{label} identity drifted")
    return {key: int(identity[key]) for key in sorted(_DIRECTORY_IDENTITY_FIELDS)}


def _validate_directory_rows(
    value: object,
    *,
    allow_reserved: bool = False,
) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise TerminalRecoveryError("preterminal directory graph drifted")
    rows: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    seen_casefold: set[str] = set()
    directory_members: set[str] = set()
    file_members: set[str] = set()
    reserved = {RECEIPT_NAME, MANIFEST_NAME, *TERMINAL_ARTIFACT_NAMES}
    for row_value in value:
        if not isinstance(row_value, Mapping) or set(row_value) != {
            "path",
            "identity",
            "members",
        }:
            raise TerminalRecoveryError("preterminal directory graph row drifted")
        raw_path = row_value.get("path")
        path = (
            "."
            if raw_path == "."
            else _validate_relative_path(raw_path, label="preterminal directory graph")
        )
        if (
            path in seen_paths
            or path.casefold() in seen_casefold
            or (not allow_reserved and path in reserved)
        ):
            raise TerminalRecoveryError("preterminal directory graph path drifted")
        seen_paths.add(path)
        seen_casefold.add(path.casefold())
        identity = _validate_directory_identity(
            row_value.get("identity"),
            label=f"preterminal directory {path}",
        )
        member_values = row_value.get("members")
        if not isinstance(member_values, list):
            raise TerminalRecoveryError("preterminal directory member list drifted")
        members: list[dict[str, str]] = []
        member_names: set[str] = set()
        member_casefold: set[str] = set()
        for member_value in member_values:
            if not isinstance(member_value, Mapping) or set(member_value) != {
                "name",
                "kind",
            }:
                raise TerminalRecoveryError("preterminal directory member row drifted")
            name = _validate_member_name(
                member_value.get("name"),
                label="preterminal directory graph",
            )
            kind = member_value.get("kind")
            if (
                kind not in {"directory", "file"}
                or name in member_names
                or name.casefold() in member_casefold
                or (not allow_reserved and path == "." and name in reserved)
            ):
                raise TerminalRecoveryError("preterminal directory member drifted")
            member_path = name if path == "." else f"{path}/{name}"
            _validate_relative_path(
                member_path,
                label="preterminal directory graph member",
            )
            if member_path in directory_members or member_path in file_members:
                raise TerminalRecoveryError("preterminal directory member path drifted")
            member_names.add(name)
            member_casefold.add(name.casefold())
            if kind == "directory":
                directory_members.add(member_path)
            else:
                file_members.add(member_path)
            members.append({"name": name, "kind": str(kind)})
        if [member["name"] for member in members] != sorted(member_names):
            raise TerminalRecoveryError("preterminal directory member order drifted")
        rows.append({"path": path, "identity": identity, "members": members})
    if [str(row["path"]) for row in rows] != sorted(seen_paths) or "." not in seen_paths:
        raise TerminalRecoveryError("preterminal directory graph order drifted")
    if directory_members != seen_paths - {"."} or file_members & seen_paths:
        raise TerminalRecoveryError("preterminal directory membership graph drifted")
    return rows


def _directory_file_paths(rows: list[dict[str, object]]) -> set[str]:
    paths: set[str] = set()
    for row in rows:
        parent = str(row["path"])
        members = row["members"]
        assert isinstance(members, list)
        for member in members:
            assert isinstance(member, Mapping)
            if member["kind"] == "file":
                name = str(member["name"])
                paths.add(name if parent == "." else f"{parent}/{name}")
    return paths


def _validate_evidence_file_rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise TerminalRecoveryError("preterminal evidence file binding drifted")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for row_value in value:
        if not isinstance(row_value, Mapping) or set(row_value) != _EVIDENCE_FILE_FIELDS:
            raise TerminalRecoveryError("preterminal evidence file row drifted")
        path = _validate_relative_path(
            row_value.get("path"),
            label="preterminal evidence file",
        )
        identity = row_value.get("identity")
        if (
            not isinstance(identity, Mapping)
            or set(identity) != _EVIDENCE_FILE_IDENTITY_FIELDS
            or any(type(identity.get(key)) is not int for key in _EVIDENCE_FILE_IDENTITY_FIELDS)
            or int(identity["device"]) < 0
            or int(identity["inode"]) < 0
            or int(identity["file_type"]) != stat.S_IFREG
            or int(identity["size_bytes"]) < 0
            or int(identity["file_attributes"]) < 0
            or int(identity["link_count"]) != 1
        ):
            raise TerminalRecoveryError("preterminal evidence file identity drifted")
        file_identity = _validate_identity(
            {
                "sha256": row_value.get("sha256"),
                "size_bytes": row_value.get("size_bytes"),
            },
            label=f"preterminal evidence file {path}",
        )
        if file_identity["size_bytes"] != identity["size_bytes"] or path in seen:
            raise TerminalRecoveryError("preterminal evidence file binding drifted")
        seen.add(path)
        rows.append(
            {
                "path": path,
                **file_identity,
                "identity": {
                    key: int(identity[key])
                    for key in sorted(_EVIDENCE_FILE_IDENTITY_FIELDS)
                },
            }
        )
    if [str(row["path"]) for row in rows] != sorted(seen):
        raise TerminalRecoveryError("preterminal evidence file order drifted")
    return rows


def _validate_evidence_binding(
    value: object,
    *,
    allow_reserved: bool = False,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _EVIDENCE_BINDING_FIELDS:
        raise TerminalRecoveryError("preterminal evidence binding schema drifted")
    if value.get("schema_version") != EVIDENCE_BINDING_SCHEMA:
        raise TerminalRecoveryError("preterminal evidence binding version drifted")
    files = _validate_evidence_file_rows(value.get("files"))
    directories = _validate_directory_rows(
        value.get("directories"),
        allow_reserved=allow_reserved,
    )
    if _directory_file_paths(directories) != {str(row["path"]) for row in files}:
        raise TerminalRecoveryError(
            "preterminal evidence file and directory graphs drifted"
        )
    graph = {"files": files, "directories": directories}
    graph_sha256 = hashlib.sha256(_canonical_json(graph)).hexdigest()
    if value.get("graph_sha256") != graph_sha256:
        raise TerminalRecoveryError("preterminal evidence graph digest drifted")
    result = {
        "schema_version": EVIDENCE_BINDING_SCHEMA,
        "graph_sha256": graph_sha256,
        **graph,
    }
    _canonical_json(result)
    return result


def _evidence_file_rows(pin: Stage6InputPin, stage: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in pin.records:
        try:
            relative = record.path.relative_to(stage).as_posix()
        except ValueError as exc:
            raise TerminalRecoveryError(
                "preterminal evidence pin escaped the Stage 6 root"
            ) from exc
        identity = record.identity
        rows.append(
            {
                "path": _validate_relative_path(
                    relative,
                    label="preterminal evidence pin",
                ),
                "sha256": record.sha256,
                "size_bytes": record.size_bytes,
                "identity": {
                    "device": int(identity[0]),
                    "inode": int(identity[1]),
                    "file_type": int(identity[2]),
                    "size_bytes": int(identity[3]),
                    "file_attributes": int(identity[4]),
                    "link_count": int(identity[5]),
                },
            }
        )
    rows.sort(key=lambda row: str(row["path"]))
    return _validate_evidence_file_rows(rows)


def _require_absent_evidence_paths(
    stage: Path,
    relatives: frozenset[str],
    *,
    label: str,
) -> None:
    for relative_value in sorted(relatives):
        relative = _validate_relative_path(
            relative_value,
            label="Stage 6 absent evidence path",
        )
        if os.path.lexists(stage / relative):
            raise TerminalRecoveryError(
                f"{label} reserved artifact appeared: {relative}"
            )


@dataclass(slots=True)
class _Stage6EvidenceHandle:
    stage: Path
    excluded: frozenset[str]
    required_absent: frozenset[str]
    allow_reserved: bool
    pin: Stage6InputPin
    _binding: dict[str, object]
    graph_rows: list[dict[str, object]]
    directory_rows: list[dict[str, object]]
    closed: bool = False

    def __enter__(self) -> "_Stage6EvidenceHandle":
        if self.closed:
            raise TerminalRecoveryError("Stage 6 evidence handle is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exc_type, traceback
        try:
            self.close()
        except BaseException as close_error:
            if exc_value is None:
                raise
            exc_value.add_note(f"suppressed Stage 6 evidence close failure: {close_error}")

    @property
    def binding(self) -> dict[str, object]:
        if self.closed:
            raise TerminalRecoveryError("Stage 6 evidence handle is closed")
        return json.loads(_canonical_json(self._binding).decode("utf-8"))

    @property
    def file_paths(self) -> tuple[str, ...]:
        if self.closed:
            raise TerminalRecoveryError("Stage 6 evidence handle is closed")
        return tuple(str(row["path"]) for row in self.graph_rows)

    def graph_snapshot(self) -> list[dict[str, object]]:
        if self.closed:
            raise TerminalRecoveryError("Stage 6 evidence handle is closed")
        return json.loads(_canonical_json(self.graph_rows).decode("utf-8"))

    def directory_snapshot(self) -> list[dict[str, object]]:
        if self.closed:
            raise TerminalRecoveryError("Stage 6 evidence handle is closed")
        return json.loads(_canonical_json(self.directory_rows).decode("utf-8"))

    def read_bytes(self, relative: str, *, label: str) -> bytes:
        if self.closed:
            raise TerminalRecoveryError("Stage 6 evidence handle is closed")
        path = _validate_relative_path(relative, label=label)
        expected = next(
            (row for row in self._binding["files"] if row["path"] == path),  # type: ignore[index]
            None,
        )
        if not isinstance(expected, Mapping):
            raise TerminalRecoveryError(f"{label} is outside the captured evidence graph")
        try:
            payload = self.pin.read_bytes(
                self.stage / path,
                label=label,
            )
        except Stage6InputPinError as exc:
            raise TerminalRecoveryError(
                f"{label} pinned evidence is unreadable"
            ) from exc
        if (
            hashlib.sha256(payload).hexdigest() != expected["sha256"]
            or len(payload) != expected["size_bytes"]
        ):
            raise TerminalRecoveryError(f"{label} changed from captured evidence")
        return payload

    def require_current(self, label: str) -> None:
        if self.closed:
            raise TerminalRecoveryError("Stage 6 evidence handle is closed")
        _require_absent_evidence_paths(
            self.stage,
            self.required_absent,
            label=label,
        )
        try:
            self.pin.require_current(label, rehash=True)
        except Stage6InputPinError as exc:
            raise TerminalRecoveryError(f"{label} file binding changed") from exc
        current_directories = _validate_directory_rows(
            _capture_directory_graph(self.stage, excluded=self.excluded),
            allow_reserved=self.allow_reserved,
        )
        if current_directories != self.directory_rows:
            raise TerminalRecoveryError(f"{label} directory graph changed")
        current_files = _evidence_file_rows(self.pin, self.stage)
        current_graph = [
            {
                "path": row["path"],
                "sha256": row["sha256"],
                "size_bytes": row["size_bytes"],
            }
            for row in current_files
        ]
        if current_graph != self.graph_rows:
            raise TerminalRecoveryError(f"{label} artifact graph changed")
        current_binding = _validate_evidence_binding(
            {
                "schema_version": EVIDENCE_BINDING_SCHEMA,
                "graph_sha256": hashlib.sha256(
                    _canonical_json(
                        {"files": current_files, "directories": current_directories}
                    )
                ).hexdigest(),
                "files": current_files,
                "directories": current_directories,
            },
            allow_reserved=self.allow_reserved,
        )
        if current_binding != self._binding:
            raise TerminalRecoveryError(f"{label} canonical binding changed")
        _validate_evidence_repair_profile(
            self.stage,
            file_paths=set(self.file_paths),
            payload_reader=lambda relative: self.read_bytes(
                relative,
                label=f"{label} planning child profile",
            ),
        )
        _require_absent_evidence_paths(
            self.stage,
            self.required_absent,
            label=label,
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.pin.close()
        except Stage6InputPinError as exc:
            raise TerminalRecoveryError("Stage 6 evidence handle close failed") from exc


def _capture_stage6_evidence_handle(
    stage: Path,
    *,
    excluded: frozenset[str],
    required_absent: frozenset[str] = frozenset(),
    allow_reserved: bool = False,
) -> _Stage6EvidenceHandle:
    stage = _require_stage_root(stage)
    if not required_absent.issubset(excluded):
        raise TerminalRecoveryError("Stage 6 absent evidence paths drifted")
    _require_absent_evidence_paths(
        stage,
        required_absent,
        label="Stage 6 evidence capture",
    )
    directory_before = _validate_directory_rows(
        _capture_directory_graph(stage, excluded=excluded),
        allow_reserved=allow_reserved,
    )
    graph_before = _validate_graph_rows(
        _capture_graph(stage, excluded=excluded),
        allow_reserved=allow_reserved,
        require_mutable_prefix=True,
    )
    graph_paths = {str(row["path"]) for row in graph_before}
    if _directory_file_paths(directory_before) != graph_paths:
        raise TerminalRecoveryError(
            "Stage 6 evidence file and directory captures do not agree"
        )
    _validate_evidence_repair_profile(
        stage,
        file_paths=graph_paths,
    )
    try:
        pin = acquire_stage6_input_pin(
            tuple(
                (f"stage6-evidence:{relative}", stage / relative)
                for relative in sorted(graph_paths)
            )
        )
    except Stage6InputPinError as exc:
        raise TerminalRecoveryError("Stage 6 evidence strong pin failed") from exc
    try:
        files = _evidence_file_rows(pin, stage)
        reduced = [
            {
                "path": row["path"],
                "sha256": row["sha256"],
                "size_bytes": row["size_bytes"],
            }
            for row in files
        ]
        if reduced != graph_before:
            raise TerminalRecoveryError(
                "Stage 6 evidence bytes changed before strong pin"
            )
        directory_after = _validate_directory_rows(
            _capture_directory_graph(stage, excluded=excluded),
            allow_reserved=allow_reserved,
        )
        if directory_after != directory_before:
            raise TerminalRecoveryError(
                "Stage 6 evidence directory graph changed during pin"
            )
        _require_absent_evidence_paths(
            stage,
            required_absent,
            label="Stage 6 evidence capture",
        )
        graph_value = {"files": files, "directories": directory_after}
        binding = _validate_evidence_binding(
            {
                "schema_version": EVIDENCE_BINDING_SCHEMA,
                "graph_sha256": hashlib.sha256(
                    _canonical_json(graph_value)
                ).hexdigest(),
                **graph_value,
            },
            allow_reserved=allow_reserved,
        )
        handle = _Stage6EvidenceHandle(
            stage=stage,
            excluded=excluded,
            required_absent=required_absent,
            allow_reserved=allow_reserved,
            pin=pin,
            _binding=binding,
            graph_rows=graph_before,
            directory_rows=directory_after,
        )
        handle.require_current("Stage 6 evidence capture")
        return handle
    except BaseException as exc:
        try:
            pin.close()
        except BaseException as close_error:
            exc.add_note(f"suppressed evidence pin close failure: {close_error}")
        raise


def _capture_preterminal_evidence_handle(
    stage: Path,
    *,
    allow_receipt_commit: bool = False,
) -> _Stage6EvidenceHandle:
    if type(allow_receipt_commit) is not bool:
        raise TerminalRecoveryError("Stage 6 receipt-commit capture mode drifted")
    reserved = frozenset({RECEIPT_NAME, MANIFEST_NAME, *TERMINAL_ARTIFACT_NAMES})
    required_absent = frozenset({MANIFEST_NAME, *TERMINAL_ARTIFACT_NAMES})
    if not allow_receipt_commit:
        required_absent = frozenset({RECEIPT_NAME, *required_absent})
    return _capture_stage6_evidence_handle(
        stage,
        excluded=reserved,
        required_absent=required_absent,
    )


def _capture_terminal_evidence_handle(stage: Path) -> _Stage6EvidenceHandle:
    """Pin the exact complete terminal graph for one public verification."""

    return _capture_stage6_evidence_handle(
        stage,
        excluded=frozenset(),
        allow_reserved=True,
    )


class _VerifiedPlanningChildSemanticResult(dict[str, object]):
    __slots__ = ("_planning_child_recovery_capability",)

    def __init__(
        self,
        value: Mapping[str, object],
        capability: object,
    ) -> None:
        from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
            PlanningChildRecoveryCapability,
        )

        if type(capability) is not PlanningChildRecoveryCapability:
            raise TypeError(
                "verified planning child semantic capability is invalid"
            )
        super().__init__(value)
        self._planning_child_recovery_capability = capability


def _bind_planning_child_semantic_result(
    value: Mapping[str, object],
    *,
    capability: object,
) -> Mapping[str, object]:
    binding = _planning_child_recovery_semantic_binding(capability)
    result = {
        **dict(value),
        "planning_child_recovery": binding,
    }
    return _VerifiedPlanningChildSemanticResult(result, capability)


def _planning_child_recovery_semantic_binding(
    capability: object,
) -> dict[str, object]:
    from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
        PLANNING_CHILD_RECOVERY_EVIDENCE_SCHEMA,
        PlanningChildRecoveryCapability,
    )

    if (
        type(capability) is not PlanningChildRecoveryCapability
        or not _is_sha256(capability.capability_sha256)
        or not _is_sha256(capability.input_snapshot_sha256)
        or not _is_sha256(capability.parent_artifact_sha256)
        or not _is_sha256(capability.continuation_artifact_sha256)
        or not isinstance(capability.acceptance_binding, Mapping)
    ):
        raise TerminalRecoveryError(
            "planning child semantic result capability is missing or forged"
        )
    binding = capability.evidence_binding()
    return _validate_planning_child_recovery_semantic_binding(
        binding,
        expected_schema=PLANNING_CHILD_RECOVERY_EVIDENCE_SCHEMA,
    )


def _validate_planning_child_recovery_semantic_binding(
    value: object,
    *,
    expected_schema: str | None = None,
) -> dict[str, object]:
    if expected_schema is None:
        from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
            PLANNING_CHILD_RECOVERY_EVIDENCE_SCHEMA,
        )

        expected_schema = PLANNING_CHILD_RECOVERY_EVIDENCE_SCHEMA
    expected_fields = {
        "schema_version",
        "capability_sha256",
        "input_snapshot_sha256",
        "parent_artifact_sha256",
        "continuation_artifact_sha256",
        "acceptance_binding",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise TerminalRecoveryError(
            "planning child recovery semantic binding schema drifted"
        )
    result = dict(value)
    if (
        result.get("schema_version")
        != expected_schema
        or any(
            not _is_sha256(result.get(field))
            for field in (
                "capability_sha256",
                "input_snapshot_sha256",
                "parent_artifact_sha256",
                "continuation_artifact_sha256",
            )
        )
        or not isinstance(
            result.get("acceptance_binding"),
            Mapping,
        )
    ):
        raise TerminalRecoveryError(
            "planning child recovery semantic binding drifted"
        )
    result["acceptance_binding"] = json.loads(
        _canonical_json(
            result["acceptance_binding"]
        ).decode("utf-8")
    )
    return result


def _require_planning_child_recovery_binding(
    value: Mapping[str, object],
    capability: object,
) -> object:
    expected = value.get("planning_child_recovery")
    current = _planning_child_recovery_semantic_binding(capability)
    if not isinstance(expected, Mapping):
        raise TerminalRecoveryError(
            "planning child recovery semantic binding is missing"
        )
    if expected.get("capability_sha256") != current[
        "capability_sha256"
    ]:
        raise TerminalRecoveryError(
            "planning child recovery capability digest drifted"
        )
    if dict(expected) != current:
        raise TerminalRecoveryError(
            "planning child recovery semantic binding drifted"
        )
    return capability


def _semantic_planning_child_capability(
    value: object,
) -> object | None:
    if type(value) is not _VerifiedPlanningChildSemanticResult:
        return None
    capability = value._planning_child_recovery_capability
    return _require_planning_child_recovery_binding(
        value,
        capability,
    )


def _planning_child_resource_capability(
    *,
    evidence_handle: _Stage6EvidenceHandle,
    capability: object | None,
    receipt: Mapping[str, object] | None,
    label: str,
) -> object | None:
    graph = (
        evidence_handle.graph_snapshot()
        if receipt is None
        else receipt.get("preterminal_artifact_graph")
    )
    if not isinstance(graph, list):
        raise TerminalRecoveryError(
            f"{label} planning child graph is invalid"
        )
    child_rows = tuple(
        row
        for row in graph
        if isinstance(row, Mapping)
        and row.get("path") == PLANNING_CHILD_SOURCE_REPAIR_NAME
    )
    if not child_rows:
        if capability is not None:
            raise TerminalRecoveryError(
                f"{label} unexpected planning child capability"
            )
        return None
    if len(child_rows) != 1:
        raise TerminalRecoveryError(
            f"{label} planning child graph binding drifted"
        )
    from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
        PlanningChildRecoveryCapability,
    )

    if (
        type(capability) is not PlanningChildRecoveryCapability
        or capability.stage_root != evidence_handle.stage
        or child_rows[0].get("sha256")
        != capability.parent_artifact_sha256
    ):
        raise TerminalRecoveryError(
            f"{label} planning child amendment identity drifted"
        )
    continuation_rows = tuple(
        row
        for row in graph
        if isinstance(row, Mapping)
        and row.get("path") == PLANNING_CHILD_CONTINUATION_NAME
    )
    if (
        len(continuation_rows) != 1
        or continuation_rows[0].get("sha256")
        != capability.continuation_artifact_sha256
    ):
        raise TerminalRecoveryError(
            f"{label} planning child continuation identity drifted"
        )
    _planning_child_recovery_semantic_binding(capability)
    evidence_handle.require_current(
        f"{label} planning child recovery capability"
    )
    return capability


def _read_context_bytes(
    stage: Path,
    relative: str,
    *,
    label: str,
    evidence_handle: _Stage6EvidenceHandle | None,
) -> bytes:
    if evidence_handle is None:
        return _read_plain(stage, relative, label=label)
    if evidence_handle.stage != stage:
        raise TerminalRecoveryError(f"{label} evidence root drifted")
    return evidence_handle.read_bytes(relative, label=label)


def _context_file_paths(
    evidence_handle: _Stage6EvidenceHandle | None,
) -> set[str] | None:
    if evidence_handle is None:
        return None
    return set(evidence_handle.file_paths)


def _context_graph_rows(
    stage: Path,
    evidence_handle: _Stage6EvidenceHandle | None,
) -> list[dict[str, object]]:
    if evidence_handle is None:
        return _capture_graph(stage)
    if evidence_handle.stage != stage:
        raise TerminalRecoveryError("terminal evidence graph root drifted")
    return _validate_graph_rows(
        evidence_handle.graph_snapshot(),
        allow_reserved=True,
        require_mutable_prefix=True,
    )


def _context_preterminal_directory_rows(
    stage: Path,
    evidence_handle: _Stage6EvidenceHandle | None,
) -> list[dict[str, object]]:
    excluded = frozenset(
        {RECEIPT_NAME, MANIFEST_NAME, *TERMINAL_ARTIFACT_NAMES}
    )
    if evidence_handle is None:
        return _capture_directory_graph(stage, excluded=excluded)
    if evidence_handle.stage != stage:
        raise TerminalRecoveryError("terminal evidence directory root drifted")
    rows = evidence_handle.directory_snapshot()
    for row in rows:
        parent = str(row["path"])
        members = row["members"]
        assert isinstance(members, list)
        row["members"] = [
            member
            for member in members
            if (
                str(member["name"])
                if parent == "."
                else f"{parent}/{member['name']}"
            )
            not in excluded
        ]
    return _validate_directory_rows(rows)


def _record_hash(event: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(dict(event))).hexdigest()


def _validate_phase_payload(
    payload: bytes,
    *,
    immutable_bindings: Mapping[str, object],
    global_checkpoint: Mapping[str, object],
    require_preterminal: bool,
    receipt_identity: Mapping[str, object] | None = None,
    evidence_graph_sha256: object | None = None,
) -> tuple[dict[str, object], ...]:
    rows = _strict_jsonl(payload, label="Stage 6 phase state")
    states = tuple(row.get("state") for row in rows)
    allowed = (PRETERMINAL_STATES,) if require_preterminal else (
        PRETERMINAL_STATES,
        COMPLETE_STATES[:6],
        COMPLETE_STATES,
    )
    if states not in allowed:
        raise TerminalRecoveryError("Stage 6 phase prefix drifted")
    success_binding: dict[str, object] | None = None
    if require_preterminal:
        if receipt_identity is not None or evidence_graph_sha256 is not None:
            raise TerminalRecoveryError(
                "preterminal phase validation received terminal bindings"
            )
    else:
        receipt = _validate_identity(
            receipt_identity,
            label="terminal phase preterminal acceptance",
        )
        if not _is_sha256(evidence_graph_sha256):
            raise TerminalRecoveryError(
                "terminal phase evidence graph binding drifted"
            )
        success_binding = {
            "preterminal_acceptance_sha256": receipt["sha256"],
            "preterminal_evidence_graph_sha256": evidence_graph_sha256,
        }
    previous = ABSENT_AUTHORITY_SHA256
    current_binding_keys = {*immutable_bindings, "checkpoint_sha256"}
    for index, row in enumerate(rows):
        bindings = row.get("bindings")
        state = row.get("state")
        success_state = state in COMPLETE_STATES[-2:]
        historical_preflight = (
            index == 0
            and set(immutable_bindings) == _CURRENT_IMMUTABLE_FIELDS
            and isinstance(bindings, Mapping)
            and set(bindings) == {*_IMMUTABLE_FIELDS, "checkpoint_sha256"}
        )
        expected_binding_keys = (
            current_binding_keys | set(_SUCCESS_PHASE_BINDING_FIELDS)
            if success_state
            else (
                {*_IMMUTABLE_FIELDS, "checkpoint_sha256"}
                if historical_preflight
                else current_binding_keys
            )
        )
        event = {
            "state": state,
            "previous_record_hash": row.get("previous_record_hash"),
            "bindings": bindings,
        }
        if (
            set(row) != {*event, "record_hash"}
            or not isinstance(bindings, Mapping)
            or set(bindings) != expected_binding_keys
            or (
                historical_preflight
                and _validate_immutable_bindings(
                    {key: bindings.get(key) for key in _IMMUTABLE_FIELDS}
                )
                is None
            )
            or (
                not historical_preflight
                and any(
                    bindings.get(key) != value
                    for key, value in immutable_bindings.items()
                )
            )
            or not _is_sha256(bindings.get("checkpoint_sha256"))
            or (index >= 1 and bindings.get("checkpoint_sha256") != global_checkpoint["checkpoint_sha256"])
            or (
                success_state
                and (
                    success_binding is None
                    or any(
                        bindings.get(key) != value
                        for key, value in success_binding.items()
                    )
                )
            )
            or event["previous_record_hash"] != previous
            or row.get("record_hash") != _record_hash(event)
        ):
            raise TerminalRecoveryError("Stage 6 phase hash chain or binding drifted")
        previous = str(row["record_hash"])
    return rows


def _validate_receipt_payload(
    payload: bytes,
) -> tuple[dict[str, object], bytes, dict[str, object]]:
    receipt = _strict_json(payload, label="Stage 6 preterminal receipt")
    if set(receipt) != _RECEIPT_FIELDS or receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise TerminalRecoveryError("preterminal receipt schema or unknown field drifted")
    semantic = _validate_semantic_result(receipt.get("semantic_verification"))
    evidence_binding = _validate_evidence_binding(
        receipt.get("preterminal_evidence_binding")
    )
    if semantic["evidence_binding"] != evidence_binding:
        raise TerminalRecoveryError(
            "preterminal receipt semantic evidence binding drifted"
        )
    immutable = _validate_immutable_bindings(receipt.get("immutable_bindings"))
    global_checkpoint = _validate_global_checkpoint(
        receipt.get("global_checkpoint_identity")
    )
    if semantic["global_best_policy_state_sha256"] != global_checkpoint["policy_state_sha256"]:
        raise TerminalRecoveryError("preterminal receipt policy identity drifted")
    artifacts = _validate_artifact_rows(receipt.get("terminal_artifacts"))
    graph = _validate_graph_rows(receipt.get("preterminal_artifact_graph"))
    directory_graph = _validate_directory_rows(
        receipt.get("preterminal_directory_graph")
    )
    if _directory_file_paths(directory_graph) != {
        str(row["path"]) for row in graph
    }:
        raise TerminalRecoveryError(
            "preterminal file and directory membership graphs drifted"
        )
    evidence_files = evidence_binding["files"]
    evidence_directories = evidence_binding["directories"]
    assert isinstance(evidence_files, list)
    assert isinstance(evidence_directories, list)
    if (
        [
            {
                "path": row["path"],
                "sha256": row["sha256"],
                "size_bytes": row["size_bytes"],
            }
            for row in evidence_files
        ]
        != graph
        or evidence_directories != directory_graph
    ):
        raise TerminalRecoveryError(
            "preterminal receipt graph does not match semantic evidence binding"
        )
    resource_prefix = _validate_identity(
        receipt.get("resource_audit_prefix"), label="resource audit prefix"
    )
    phase_prefix = _validate_identity(
        receipt.get("phase_state_prefix"), label="phase state prefix"
    )
    graph_by_path = {str(row["path"]): row for row in graph}
    if (
        {key: graph_by_path[RESOURCE_NAME][key] for key in ("sha256", "size_bytes")}
        != resource_prefix
        or {key: graph_by_path[PHASE_NAME][key] for key in ("sha256", "size_bytes")}
        != phase_prefix
    ):
        raise TerminalRecoveryError("preterminal mutable prefix graph binding drifted")
    receipt["semantic_verification"] = semantic
    receipt["preterminal_evidence_binding"] = evidence_binding
    receipt["immutable_bindings"] = immutable
    receipt["global_checkpoint_identity"] = global_checkpoint
    receipt["terminal_artifacts"] = artifacts
    receipt["preterminal_artifact_graph"] = graph
    receipt["preterminal_directory_graph"] = directory_graph
    receipt["resource_audit_prefix"] = resource_prefix
    receipt["phase_state_prefix"] = phase_prefix
    return receipt, payload, _identity(payload)


def _load_receipt(
    stage: Path,
    *,
    evidence_handle: _Stage6EvidenceHandle | None = None,
) -> tuple[dict[str, object], bytes, dict[str, object]]:
    payload = _read_context_bytes(
        stage,
        RECEIPT_NAME,
        label="Stage 6 preterminal receipt",
        evidence_handle=evidence_handle,
    )
    return _validate_receipt_payload(payload)


def _validate_semantically_complete_resource_rows(
    rows: tuple[dict[str, object], ...],
    resource_payload: bytes,
    *,
    label: str,
    planning_child_capability: object | None = None,
) -> dict[str, object]:
    if type(resource_payload) is not bytes or not resource_payload:
        raise TerminalRecoveryError(f"{label} resource payload is invalid")
    try:
        validation = validate_resource_lifecycle_rows(
            rows,
            require_terminal=False,
        )
    except ResourceLifecycleError as exc:
        raise TerminalRecoveryError(
            f"{label} resource lifecycle replay failed"
        ) from exc
    if validation.get("terminal_evidence_present") is not False:
        raise TerminalRecoveryError(f"{label} contains terminal evidence")

    attempt_phases: dict[tuple[str, int], list[str]] = {}
    for row in rows:
        phase = row.get("phase")
        if phase not in {"pre", "post", "accepted"}:
            continue
        transaction_key = row.get("transaction_key")
        attempt = row.get("attempt")
        if not isinstance(transaction_key, str) or type(attempt) is not int:
            raise TerminalRecoveryError(f"{label} attempt identity drifted")
        attempt_phases.setdefault((transaction_key, attempt), []).append(str(phase))
    incomplete_attempts = {
        key: phases
        for key, phases in attempt_phases.items()
        if phases != ["pre", "post", "accepted"]
    }
    if planning_child_capability is not None:
        from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
            PlanningChildRecoveryCapability,
        )

        if type(planning_child_capability) is not PlanningChildRecoveryCapability:
            raise TerminalRecoveryError(
                f"{label} planning child capability is forged"
            )
        prefixes = planning_child_capability.acceptance_binding.get(
            "journal_prefixes"
        )
        binding = (
            prefixes.get(RESOURCE_NAME)
            if isinstance(prefixes, Mapping)
            else None
        )
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != RESOURCE_NAME
            or type(binding.get("size_bytes")) is not int
            or int(binding["size_bytes"]) <= 0
            or type(binding.get("line_count")) is not int
            or int(binding["line_count"]) <= 0
            or not _is_sha256(binding.get("sha256"))
            or int(binding["size_bytes"]) > len(resource_payload)
        ):
            raise TerminalRecoveryError(
                f"{label} planning child resource binding drifted"
            )
        bound_size = int(binding["size_bytes"])
        bound_payload = resource_payload[:bound_size]
        if (
            hashlib.sha256(bound_payload).hexdigest()
            != binding["sha256"]
            or bound_payload.count(b"\n") != binding["line_count"]
        ):
            raise TerminalRecoveryError(
                f"{label} planning child resource bound bytes drifted"
            )
        bound_rows = _strict_jsonl(
            bound_payload,
            label=f"{label} planning child bound resource prefix",
        )
        bound_attempt_phases: dict[tuple[str, int], list[str]] = {}
        for row in bound_rows:
            phase = row.get("phase")
            if phase not in {"pre", "post", "accepted"}:
                continue
            transaction_key = row.get("transaction_key")
            attempt = row.get("attempt")
            if (
                not isinstance(transaction_key, str)
                or type(attempt) is not int
            ):
                raise TerminalRecoveryError(
                    f"{label} planning child bound attempt drifted"
                )
            bound_attempt_phases.setdefault(
                (transaction_key, attempt),
                [],
            ).append(str(phase))
        allowed_incomplete = {
            key: phases
            for key, phases in bound_attempt_phases.items()
            if phases != ["pre", "post", "accepted"]
        }
        if any(
            allowed_incomplete.get(key) != phases
            for key, phases in incomplete_attempts.items()
        ):
            raise TerminalRecoveryError(
                f"{label} planning child incomplete attempt drifted"
            )
    elif incomplete_attempts:
        raise TerminalRecoveryError(
            f"{label} is not a semantically completed resource prefix"
        )
    return validation


def _validate_receipt_resource_prefix_and_suffix(
    receipt: Mapping[str, object],
    resource_payload: bytes,
    *,
    planning_child_capability: object | None,
) -> dict[str, object]:
    prefix_binding = receipt.get("resource_audit_prefix")
    if not isinstance(prefix_binding, Mapping):
        raise TerminalRecoveryError("receipt resource audit prefix drifted")
    prefix_size = int(prefix_binding["size_bytes"])
    if prefix_size > len(resource_payload):
        raise TerminalRecoveryError("resource audit prefix was truncated")
    semantic_payload = resource_payload[:prefix_size]
    suffix_payload = resource_payload[prefix_size:]
    if _identity(semantic_payload) != prefix_binding:
        raise TerminalRecoveryError("resource audit prefix drifted")

    semantic_rows = _strict_jsonl(
        semantic_payload,
        label="Stage 6 receipt semantic resource prefix",
    )
    suffix_rows = _strict_jsonl(
        suffix_payload,
        label="Stage 6 recovery resource suffix",
    )
    _validate_semantically_complete_resource_rows(
        semantic_rows,
        semantic_payload,
        label="Stage 6 receipt semantic resource prefix",
        planning_child_capability=planning_child_capability,
    )
    if any(row.get("phase") != "segment_start" for row in suffix_rows):
        raise TerminalRecoveryError(
            "receipt resource suffix is not pure recovery segment_start evidence"
        )

    full_rows = (*semantic_rows, *suffix_rows)
    validation = _validate_semantically_complete_resource_rows(
        full_rows,
        resource_payload,
        label="Stage 6 full recovery resource prefix",
        planning_child_capability=planning_child_capability,
    )
    seen_segment_ids = {
        str(row["segment_id"])
        for row in semantic_rows
        if row.get("phase") == "segment_start"
    }
    seen_root_pids = {
        int(row["root_pid"])
        for row in semantic_rows
        if row.get("phase") == "segment_start"
    }
    for row in suffix_rows:
        segment_id = str(row.get("segment_id"))
        root_pid = row.get("root_pid")
        first_sample = row.get("first_sample")
        if (
            segment_id in seen_segment_ids
            or type(root_pid) is not int
            or root_pid in seen_root_pids
            or not isinstance(first_sample, Mapping)
            or first_sample.get("rss_sample_count") != 1
        ):
            raise TerminalRecoveryError(
                "recovery segment must use a new PID, UUID, and reset counter"
            )
        seen_segment_ids.add(segment_id)
        seen_root_pids.add(root_pid)
    return {
        "semantic_rows": semantic_rows,
        "suffix_rows": suffix_rows,
        "full_rows": full_rows,
        "full_prefix_identity": _identity(resource_payload),
        "validation": validation,
    }


def _validate_resource_snapshot(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _RESOURCE_FIELDS:
        raise TerminalRecoveryError("terminal resource snapshot unknown field drifted")
    snapshot = dict(value)
    integer_fields = (
        "d_free_bytes",
        "rss_bytes",
        "peak_vram_bytes",
        "rss_root_pid",
        "rss_sample_count",
        "rss_latest_process_count",
        "rss_peak_process_count",
    )
    if (
        snapshot.get("rss_source")
        != "process_tree_lifecycle_peak_current_sum/v1"
        or any(
            type(snapshot.get(key)) is not int
            or int(snapshot[key]) < (1 if key.startswith("rss_") and key != "rss_bytes" else 0)
            for key in integer_fields
        )
        or not isinstance(snapshot.get("warnings"), list)
        or not all(isinstance(item, str) for item in snapshot["warnings"])
        or snapshot.get("hard_stops") != []
        or snapshot.get("passed") is not True
    ):
        raise TerminalRecoveryError("terminal resource snapshot drifted")
    return snapshot


def _validate_terminal_row(
    terminal: Mapping[str, object],
    *,
    receipt_identity: Mapping[str, object],
    resource_prefix: Mapping[str, object],
    resource_prefix_rows: tuple[dict[str, object], ...],
) -> None:
    if TERMINAL_RECEIPT_FIELD not in terminal:
        raise TerminalRecoveryError(
            "terminal v4 requires exact preterminal_acceptance receipt identity"
        )
    if set(terminal) != _TERMINAL_FIELDS:
        raise TerminalRecoveryError("terminal v4 schema or unknown field drifted")
    if (
        terminal.get("schema_version") != TERMINAL_SCHEMA
        or terminal.get("kind") != "terminal_lifecycle"
        or terminal.get("transaction_key") != "terminal:formal_backend"
        or terminal.get("phase") != "terminal"
        or terminal.get("accepted") is not True
        or terminal.get("measurement_boundary")
        != "after_finalization_payload_and_monitor_stop_before_success_commit/v4"
        or _validate_identity(
            terminal.get(TERMINAL_RECEIPT_FIELD),
            label="terminal preterminal_acceptance",
        )
        != receipt_identity
        or _validate_identity(
            terminal.get("prior_resource_audit"), label="terminal resource prefix"
        )
        != resource_prefix
    ):
        raise TerminalRecoveryError("terminal receipt or resource prefix binding drifted")
    resource = _validate_resource_snapshot(terminal.get("resource"))
    try:
        expected = build_bound_terminal_resource_evidence(
            resource_prefix_rows,
            terminal_resource=resource,
            preterminal_acceptance=receipt_identity,
        )
        validation = validate_resource_lifecycle_rows(
            (*resource_prefix_rows, dict(terminal)),
            require_terminal=True,
        )
    except ResourceLifecycleError as exc:
        raise TerminalRecoveryError(
            "terminal resource lifecycle prefix replay failed"
        ) from exc
    if dict(terminal) != expected or validation.get("terminal_evidence_present") is not True:
        raise TerminalRecoveryError(
            "terminal resource lifecycle replay does not exactly match terminal v4"
        )


def _validate_manifest(
    stage: Path,
    verifier: Callable[[Path], Mapping[str, object]] | None,
    *,
    evidence_handle: _Stage6EvidenceHandle | None,
) -> dict[str, object] | None:
    file_paths = _context_file_paths(evidence_handle)
    manifest_present = (
        MANIFEST_NAME in file_paths
        if file_paths is not None
        else os.path.lexists(stage / MANIFEST_NAME)
    )
    if not manifest_present:
        return None
    if verifier is None:
        raise TerminalRecoveryError("existing manifest requires a pure verifier callback")
    result = verifier(stage)
    expected = _validate_identity(result, label="manifest callback")
    payload = _read_context_bytes(
        stage,
        MANIFEST_NAME,
        label="Stage 6 terminal manifest",
        evidence_handle=evidence_handle,
    )
    if _identity(payload) != expected:
        raise TerminalRecoveryError("manifest callback identity drifted")
    return expected


def _validate_current_graph(
    stage: Path,
    receipt: Mapping[str, object],
    *,
    resource_payload: bytes,
    phase_payload: bytes,
    manifest_verifier: Callable[[Path], Mapping[str, object]] | None,
    evidence_handle: _Stage6EvidenceHandle | None,
) -> dict[str, object] | None:
    graph = receipt["preterminal_artifact_graph"]
    assert isinstance(graph, list)
    graph_by_path = {str(row["path"]): row for row in graph}
    current = _context_graph_rows(stage, evidence_handle)
    current_by_path = {str(row["path"]): row for row in current}
    allowed = {
        *graph_by_path,
        RECEIPT_NAME,
        MANIFEST_NAME,
        *TERMINAL_ARTIFACT_NAMES,
    }
    unknown = set(current_by_path) - allowed
    if unknown:
        raise TerminalRecoveryError(
            f"unknown terminal recovery artifact found: {sorted(unknown)[0]}"
        )
    missing = set(graph_by_path) - set(current_by_path)
    if missing:
        raise TerminalRecoveryError(
            f"preterminal artifact graph member is missing: {sorted(missing)[0]}"
        )
    for path, expected in graph_by_path.items():
        if path in {RESOURCE_NAME, PHASE_NAME}:
            continue
        if current_by_path[path] != expected:
            raise TerminalRecoveryError(f"preterminal artifact drifted: {path}")
    directory_graph = receipt["preterminal_directory_graph"]
    assert isinstance(directory_graph, list)
    current_directory_graph = _context_preterminal_directory_rows(
        stage,
        evidence_handle,
    )
    if current_directory_graph != directory_graph:
        raise TerminalRecoveryError("preterminal directory graph drifted")
    if _identity(resource_payload[: int(receipt["resource_audit_prefix"]["size_bytes"])]) != receipt[
        "resource_audit_prefix"
    ]:
        raise TerminalRecoveryError("resource audit prefix drifted")
    if _identity(phase_payload[: int(receipt["phase_state_prefix"]["size_bytes"])]) != receipt[
        "phase_state_prefix"
    ]:
        raise TerminalRecoveryError("phase state prefix drifted")
    terminal_rows = receipt["terminal_artifacts"]
    assert isinstance(terminal_rows, list)
    for row in terminal_rows:
        relative = str(row["path"])
        if relative not in current_by_path:
            continue
        payload = _read_context_bytes(
            stage,
            relative,
            label=f"terminal artifact {relative}",
            evidence_handle=evidence_handle,
        )
        expected_payload = str(row["utf8"]).encode("utf-8")
        if payload != expected_payload or _identity(payload) != {
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
        }:
            raise TerminalRecoveryError(f"terminal artifact drifted: {relative}")
    return _validate_manifest(
        stage,
        manifest_verifier,
        evidence_handle=evidence_handle,
    )


def _load_preterminal_context(
    stage: Path,
    *,
    evidence_handle: _Stage6EvidenceHandle | None = None,
    planning_child_capability: object | None = None,
) -> dict[str, object]:
    receipt, _, receipt_identity = _load_receipt(
        stage,
        evidence_handle=evidence_handle,
    )
    file_paths = _context_file_paths(evidence_handle)
    reserved_present = any(
        (
            relative in file_paths
            if file_paths is not None
            else os.path.lexists(stage / relative)
        )
        for relative in (MANIFEST_NAME, *TERMINAL_ARTIFACT_NAMES)
    )
    if reserved_present:
        raise TerminalRecoveryError(
            "receipt-only recovery contains a terminal artifact or manifest"
        )
    if evidence_handle is None:
        graph_rows = receipt.get("preterminal_artifact_graph")
        if (
            isinstance(graph_rows, list)
            and any(
                isinstance(row, Mapping)
                and row.get("path")
                == PLANNING_CHILD_SOURCE_REPAIR_NAME
                for row in graph_rows
            )
        ):
            raise TerminalRecoveryError(
                "planning child preterminal recovery requires pinned "
                "validated context"
            )
        bound_planning_child_capability = None
    else:
        bound_planning_child_capability = (
            _planning_child_resource_capability(
                evidence_handle=evidence_handle,
                capability=planning_child_capability,
                receipt=receipt,
                label="Stage 6 preterminal recovery",
            )
        )
    resource_payload = _read_context_bytes(
        stage,
        RESOURCE_NAME,
        label="Stage 6 resource audit",
        evidence_handle=evidence_handle,
    )
    resource_context = _validate_receipt_resource_prefix_and_suffix(
        receipt,
        resource_payload,
        planning_child_capability=bound_planning_child_capability,
    )
    phase_payload = _read_context_bytes(
        stage,
        PHASE_NAME,
        label="Stage 6 phase state",
        evidence_handle=evidence_handle,
    )
    phase_prefix = receipt["phase_state_prefix"]
    assert isinstance(phase_prefix, Mapping)
    if _identity(phase_payload) != phase_prefix:
        raise TerminalRecoveryError("receipt-only phase state prefix drifted")
    phase_rows = _validate_phase_payload(
        phase_payload,
        immutable_bindings=receipt["immutable_bindings"],
        global_checkpoint=receipt["global_checkpoint_identity"],
        require_preterminal=True,
    )
    _validate_current_graph(
        stage,
        receipt,
        resource_payload=resource_payload,
        phase_payload=phase_payload,
        manifest_verifier=None,
        evidence_handle=evidence_handle,
    )
    return {
        "receipt": receipt,
        "receipt_identity": receipt_identity,
        "resource_rows": resource_context["full_rows"],
        "resource_validation": resource_context["validation"],
        "phase_rows": phase_rows,
    }


def _load_terminal_context(
    stage: Path,
    *,
    manifest_verifier: Callable[[Path], Mapping[str, object]] | None,
    evidence_handle: _Stage6EvidenceHandle | None = None,
    planning_child_capability: object | None = None,
) -> dict[str, object]:
    receipt, _, receipt_identity = _load_receipt(
        stage,
        evidence_handle=evidence_handle,
    )
    if evidence_handle is None:
        graph_rows = receipt.get("preterminal_artifact_graph")
        if (
            isinstance(graph_rows, list)
            and any(
                isinstance(row, Mapping)
                and row.get("path")
                == PLANNING_CHILD_SOURCE_REPAIR_NAME
                for row in graph_rows
            )
        ):
            raise TerminalRecoveryError(
                "planning child terminal recovery requires pinned "
                "validated context"
            )
        bound_planning_child_capability = None
    else:
        bound_planning_child_capability = (
            _planning_child_resource_capability(
                evidence_handle=evidence_handle,
                capability=planning_child_capability,
                receipt=receipt,
                label="Stage 6 terminal recovery",
            )
        )
    resource_payload = _read_context_bytes(
        stage,
        RESOURCE_NAME,
        label="Stage 6 resource audit",
        evidence_handle=evidence_handle,
    )
    resource_rows = _strict_jsonl(resource_payload, label="Stage 6 resource audit")
    terminal_indexes = tuple(
        index
        for index, row in enumerate(resource_rows)
        if row.get("phase") == "terminal"
        or row.get("transaction_key") == "terminal:formal_backend"
    )
    if terminal_indexes != (len(resource_rows) - 1,):
        raise TerminalRecoveryError("terminal row is not the unique final resource row")
    terminal = resource_rows[-1]
    terminal_payload = _canonical_row(terminal)
    if len(terminal_payload) > len(resource_payload) or not resource_payload.endswith(
        terminal_payload
    ):
        raise TerminalRecoveryError("terminal resource append drifted")
    full_prefix_payload = resource_payload[: -len(terminal_payload)]
    resource_context = _validate_receipt_resource_prefix_and_suffix(
        receipt,
        full_prefix_payload,
        planning_child_capability=bound_planning_child_capability,
    )
    resource_prefix = resource_context["full_prefix_identity"]
    resource_prefix_rows = resource_context["full_rows"]
    assert isinstance(resource_prefix, Mapping)
    assert isinstance(resource_prefix_rows, tuple)
    _validate_terminal_row(
        terminal,
        receipt_identity=receipt_identity,
        resource_prefix=resource_prefix,
        resource_prefix_rows=resource_prefix_rows,
    )
    phase_payload = _read_context_bytes(
        stage,
        PHASE_NAME,
        label="Stage 6 phase state",
        evidence_handle=evidence_handle,
    )
    phase_prefix = receipt["phase_state_prefix"]
    assert isinstance(phase_prefix, Mapping)
    prefix_size = int(phase_prefix["size_bytes"])
    if _identity(phase_payload[:prefix_size]) != phase_prefix:
        raise TerminalRecoveryError("phase state receipt prefix drifted")
    phase_rows = _validate_phase_payload(
        phase_payload,
        immutable_bindings=receipt["immutable_bindings"],
        global_checkpoint=receipt["global_checkpoint_identity"],
        require_preterminal=False,
        receipt_identity=receipt_identity,
        evidence_graph_sha256=receipt["preterminal_evidence_binding"][
            "graph_sha256"
        ],
    )
    manifest_identity = _validate_current_graph(
        stage,
        receipt,
        resource_payload=resource_payload,
        phase_payload=phase_payload,
        manifest_verifier=manifest_verifier,
        evidence_handle=evidence_handle,
    )
    return {
        "receipt": receipt,
        "receipt_identity": receipt_identity,
        "terminal": terminal,
        "resource_rows": resource_prefix_rows,
        "phase_rows": phase_rows,
        "manifest_identity": manifest_identity,
    }


def _load_preterminal_context_for_planning_child(
    stage: Path,
    *,
    planning_child_recovery_capability: object | None,
) -> dict[str, object]:
    if planning_child_recovery_capability is None:
        return _load_preterminal_context(stage)
    with _capture_terminal_evidence_handle(stage) as evidence:
        return _load_preterminal_context(
            stage,
            evidence_handle=evidence,
            planning_child_capability=(
                planning_child_recovery_capability
            ),
        )


def _load_terminal_context_for_planning_child(
    stage: Path,
    *,
    manifest_verifier: Callable[[Path], Mapping[str, object]] | None,
    planning_child_recovery_capability: object | None,
) -> dict[str, object]:
    if planning_child_recovery_capability is None:
        return _load_terminal_context(
            stage,
            manifest_verifier=manifest_verifier,
        )
    with _capture_terminal_evidence_handle(stage) as evidence:
        return _load_terminal_context(
            stage,
            manifest_verifier=manifest_verifier,
            evidence_handle=evidence,
            planning_child_capability=(
                planning_child_recovery_capability
            ),
        )


@_guarded_formal_recovery_mutation
def write_stage6_preterminal_acceptance(
    *,
    stage_root: str | Path,
    semantic_result: Mapping[str, object],
    immutable_bindings: Mapping[str, object],
    global_checkpoint_identity: Mapping[str, object],
    terminal_artifacts: Mapping[str, bytes],
    execution_capability: object,
) -> dict[str, object]:
    """Persist the unique canonical receipt after an already-passed verifier.

    This function never calls a semantic verifier.  Its ``semantic_result`` is
    accepted only when it exactly matches the existing passed-result schema.
    """

    stage = _require_stage_root(stage_root)
    planning_child_capability = _semantic_planning_child_capability(
        semantic_result
    )
    semantic = _validate_semantic_result(semantic_result)
    immutable = _validate_immutable_bindings(immutable_bindings)
    global_checkpoint = _validate_global_checkpoint(global_checkpoint_identity)
    if semantic["global_best_policy_state_sha256"] != global_checkpoint["policy_state_sha256"]:
        raise TerminalRecoveryError("semantic/global checkpoint identity drifted")
    artifact_rows = _validate_terminal_artifacts(terminal_artifacts)
    reserved = {MANIFEST_NAME, *TERMINAL_ARTIFACT_NAMES}
    if any(os.path.lexists(stage / relative) for relative in reserved):
        raise TerminalRecoveryError(
            "preterminal success artifact or manifest already exists"
        )
    receipt_exists = os.path.lexists(stage / RECEIPT_NAME)
    with _capture_preterminal_evidence_handle(
        stage,
        allow_receipt_commit=True,
    ) as evidence:
        evidence_binding = evidence.binding
        if semantic["evidence_binding"] != evidence_binding:
            raise TerminalRecoveryError(
                "semantic evidence binding does not match the current preterminal graph"
            )
        graph_before = list(evidence.graph_rows)
        directory_graph_before = list(evidence.directory_rows)
        graph_paths = {str(row["path"]) for row in graph_before}
        bound_planning_child_capability = (
            _planning_child_resource_capability(
                evidence_handle=evidence,
                capability=planning_child_capability,
                receipt=None,
                label="Stage 6 preterminal receipt writer",
            )
        )
        if not {RESOURCE_NAME, PHASE_NAME}.issubset(graph_paths):
            raise TerminalRecoveryError(
                "preterminal mutable prefix artifact is missing"
            )
        resource_payload = evidence.read_bytes(
            RESOURCE_NAME,
            label="preterminal resource audit",
        )
        resource_rows = _strict_jsonl(
            resource_payload,
            label="preterminal resource audit",
        )
        if not resource_rows or any(
            row.get("phase") == "terminal"
            or row.get("transaction_key") == "terminal:formal_backend"
            for row in resource_rows
        ):
            raise TerminalRecoveryError(
                "preterminal resource prefix contains terminal evidence"
            )
        _validate_semantically_complete_resource_rows(
            resource_rows,
            resource_payload,
            label="preterminal resource prefix",
            planning_child_capability=bound_planning_child_capability,
        )
        phase_payload = evidence.read_bytes(
            PHASE_NAME,
            label="preterminal phase state",
        )
        _validate_phase_payload(
            phase_payload,
            immutable_bindings=immutable,
            global_checkpoint=global_checkpoint,
            require_preterminal=True,
        )
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "semantic_verification": semantic,
            "preterminal_evidence_binding": evidence_binding,
            "immutable_bindings": immutable,
            "global_checkpoint_identity": global_checkpoint,
            "terminal_artifacts": artifact_rows,
            "preterminal_artifact_graph": graph_before,
            "preterminal_directory_graph": directory_graph_before,
            "resource_audit_prefix": _identity(resource_payload),
            "phase_state_prefix": _identity(phase_payload),
        }
        receipt_payload = _canonical_json(receipt)
        evidence.require_current("preterminal evidence before receipt commit")
        if any(os.path.lexists(stage / relative) for relative in reserved):
            raise TerminalRecoveryError(
                "preterminal success artifact or manifest appeared before receipt commit"
            )
        if receipt_exists:
            existing = _read_plain(
                stage,
                RECEIPT_NAME,
                label="Stage 6 preterminal receipt",
            )
            _load_receipt(stage)
            if existing != receipt_payload:
                raise TerminalRecoveryError("existing preterminal receipt drifted")
            evidence.require_current(
                "preterminal evidence at existing receipt return"
            )
            return _identity(existing)
        _require_recovery_execution_capability(
            execution_capability,
            stage_root=stage,
            label="before preterminal acceptance write",
        )
        evidence.require_current("preterminal evidence immediately before receipt write")
        try:
            ArtifactStore(stage).write_bytes_exclusive(RECEIPT_NAME, receipt_payload)
        except (FileExistsError, OSError, ValueError) as exc:
            raise TerminalRecoveryError(
                "preterminal receipt exclusive write failed"
            ) from exc
        committed = _read_plain(
            stage,
            RECEIPT_NAME,
            label="Stage 6 preterminal receipt",
        )
        if committed != receipt_payload:
            raise TerminalRecoveryError("preterminal receipt commit drifted")
        evidence.require_current("preterminal evidence after receipt commit")
        if any(os.path.lexists(stage / relative) for relative in reserved):
            raise TerminalRecoveryError(
                "preterminal success artifact or manifest appeared at receipt commit"
            )
        _require_recovery_execution_capability(
            execution_capability,
            stage_root=stage,
            label="after preterminal acceptance write",
        )
        evidence.require_current(
            "preterminal evidence after receipt capability revalidation"
        )
        return _identity(committed)


@_guarded_formal_recovery_mutation
def append_stage6_recovery_resource_segment(
    *,
    stage_root: str | Path,
    first_sample: Mapping[str, object],
    execution_capability: object,
    planning_child_recovery_capability: object | None = None,
) -> dict[str, object]:
    """完整验证 receipt-only 上下文后，追加当前恢复进程的唯一 segment。"""

    stage = _require_stage_root(stage_root)
    before = _load_preterminal_context_for_planning_child(
        stage,
        planning_child_recovery_capability=(
            planning_child_recovery_capability
        ),
    )
    sample = _validate_resource_snapshot(first_sample)
    current_pid = os.getpid()
    if (
        sample["rss_root_pid"] != current_pid
        or sample["rss_sample_count"] != 1
    ):
        raise TerminalRecoveryError(
            "recovery segment must use the current PID and reset sample counter"
        )
    before_rows = before["resource_rows"]
    assert isinstance(before_rows, tuple)
    if any(
        row.get("phase") == "segment_start"
        and row.get("root_pid") == current_pid
        for row in before_rows
    ):
        raise TerminalRecoveryError(
            "recovery segment PID was already used by the resource ledger"
        )
    _require_recovery_execution_capability(
        execution_capability,
        stage_root=stage,
        label="before recovery resource segment append",
    )
    try:
        segment = append_resource_segment_start(
            stage / RESOURCE_NAME,
            first_sample=sample,
        )
    except ResourceLifecycleError as exc:
        raise TerminalRecoveryError(
            "recovery resource segment durable append failed"
        ) from exc
    after = _load_preterminal_context_for_planning_child(
        stage,
        planning_child_recovery_capability=(
            planning_child_recovery_capability
        ),
    )
    after_rows = after["resource_rows"]
    assert isinstance(after_rows, tuple)
    if (
        len(after_rows) != len(before_rows) + 1
        or after_rows[:-1] != before_rows
        or after_rows[-1] != segment
    ):
        raise TerminalRecoveryError("recovery resource segment replay drifted")
    _require_recovery_execution_capability(
        execution_capability,
        stage_root=stage,
        label="after recovery resource segment append",
    )
    return segment


@_guarded_formal_recovery_mutation
def append_stage6_recovery_resource_terminal(
    *,
    stage_root: str | Path,
    terminal_resource: Mapping[str, object],
    execution_capability: object,
    planning_child_recovery_capability: object | None = None,
) -> dict[str, object]:
    """在 receipt-only 完整前缀后追加精确 receipt-bound v4 terminal。"""

    stage = _require_stage_root(stage_root)
    before = _load_preterminal_context_for_planning_child(
        stage,
        planning_child_recovery_capability=(
            planning_child_recovery_capability
        ),
    )
    resource = _validate_resource_snapshot(terminal_resource)
    current_pid = os.getpid()
    if resource["rss_root_pid"] != current_pid:
        raise TerminalRecoveryError("terminal resource PID is not the current process")
    rows = before["resource_rows"]
    assert isinstance(rows, tuple)
    active_segment = next(
        (row for row in reversed(rows) if row.get("phase") == "segment_start"),
        None,
    )
    if active_segment is None or active_segment.get("root_pid") != current_pid:
        raise TerminalRecoveryError(
            "terminal recovery requires a current-process active segment"
        )
    receipt_identity = before["receipt_identity"]
    assert isinstance(receipt_identity, Mapping)
    _require_recovery_execution_capability(
        execution_capability,
        stage_root=stage,
        label="before recovery resource terminal append",
    )
    try:
        terminal = append_bound_resource_lifecycle_terminal(
            stage / RESOURCE_NAME,
            terminal_resource=resource,
            preterminal_acceptance=receipt_identity,
        )
    except ResourceLifecycleError as exc:
        raise TerminalRecoveryError(
            "bound terminal resource durable append failed"
        ) from exc
    after = _load_terminal_context_for_planning_child(
        stage,
        manifest_verifier=None,
        planning_child_recovery_capability=(
            planning_child_recovery_capability
        ),
    )
    if after["terminal"] != terminal:
        raise TerminalRecoveryError("bound terminal resource replay drifted")
    _require_recovery_execution_capability(
        execution_capability,
        stage_root=stage,
        label="after recovery resource terminal append",
    )
    return terminal


def detect_stage6_terminal_recovery(
    *,
    stage_root: str | Path,
    manifest_verifier: Callable[[Path], Mapping[str, object]] | None = None,
    evidence_handle: _Stage6EvidenceHandle | None = None,
    planning_child_recovery_capability: object | None = None,
) -> dict[str, object]:
    """Read-only earliest detector for absent, valid, or invalid terminal state."""

    try:
        stage_candidate = Path(stage_root).expanduser()
        if evidence_handle is not None and type(evidence_handle) is not _Stage6EvidenceHandle:
            raise TerminalRecoveryError("terminal recovery evidence handle is forged")
        if evidence_handle is None and not os.path.lexists(stage_candidate):
            return {
                "status": "no_terminal",
                "reason": "",
                "receipt_identity": None,
                "phase_states": (),
            }
        stage = _require_stage_root(stage_candidate)
        if (
            evidence_handle is None
            and planning_child_recovery_capability is not None
        ):
            with _capture_terminal_evidence_handle(stage) as evidence:
                return detect_stage6_terminal_recovery(
                    stage_root=stage,
                    manifest_verifier=manifest_verifier,
                    evidence_handle=evidence,
                    planning_child_recovery_capability=(
                        planning_child_recovery_capability
                    ),
                )
        if evidence_handle is not None and evidence_handle.stage != stage:
            raise TerminalRecoveryError("terminal recovery evidence root drifted")
        if (
            planning_child_recovery_capability is not None
            and evidence_handle is None
        ):
            raise TerminalRecoveryError(
                "planning child detector capability is not evidence-bound"
            )
        file_paths = _context_file_paths(evidence_handle)
        receipt_present = (
            RECEIPT_NAME in file_paths
            if file_paths is not None
            else os.path.lexists(stage / RECEIPT_NAME)
        )
        resource_present = (
            RESOURCE_NAME in file_paths
            if file_paths is not None
            else os.path.lexists(stage / RESOURCE_NAME)
        )
        if not resource_present:
            if receipt_present:
                raise TerminalRecoveryError(
                    "preterminal receipt exists without its resource ledger"
                )
            return {
                "status": "no_terminal",
                "reason": "",
                "receipt_identity": None,
                "phase_states": (),
            }
        resource_payload = _read_context_bytes(
            stage,
            RESOURCE_NAME,
            label="Stage 6 resource audit",
            evidence_handle=evidence_handle,
        )
        rows = _strict_jsonl(resource_payload, label="Stage 6 resource audit")
        terminal_indexes = tuple(
            index
            for index, row in enumerate(rows)
            if row.get("phase") == "terminal"
            or row.get("transaction_key") == "terminal:formal_backend"
        )
        if not receipt_present:
            try:
                validation = validate_resource_lifecycle_rows(
                    rows,
                    require_terminal=False,
                )
            except ResourceLifecycleError as exc:
                raise TerminalRecoveryError(
                    "receiptless resource ledger replay failed"
                ) from exc
            if terminal_indexes or validation.get("terminal_evidence_present") is True:
                raise TerminalRecoveryError(
                    "terminal resource evidence exists without a receipt"
                )
            return {
                "status": "no_terminal",
                "reason": "",
                "receipt_identity": None,
                "phase_states": (),
            }
        if not terminal_indexes:
            context = _load_preterminal_context(
                stage,
                evidence_handle=evidence_handle,
                planning_child_capability=(
                    planning_child_recovery_capability
                ),
            )
            phase_rows = context["phase_rows"]
            assert isinstance(phase_rows, tuple)
            return {
                "status": "valid_preterminal_recovery",
                "reason": "",
                "receipt_identity": context["receipt_identity"],
                "phase_states": tuple(str(row["state"]) for row in phase_rows),
            }
        context = _load_terminal_context(
            stage,
            manifest_verifier=manifest_verifier,
            evidence_handle=evidence_handle,
            planning_child_capability=(
                planning_child_recovery_capability
            ),
        )
        phase_rows = context["phase_rows"]
        assert isinstance(phase_rows, tuple)
        return {
            "status": "valid_terminal_recovery",
            "reason": "",
            "receipt_identity": context["receipt_identity"],
            "phase_states": tuple(str(row["state"]) for row in phase_rows),
        }
    except Exception as exc:
        return {
            "status": "invalid",
            "reason": str(exc) or exc.__class__.__name__,
            "receipt_identity": None,
            "phase_states": (),
        }


def _write_terminal_artifact(stage: Path, row: Mapping[str, object]) -> None:
    relative = str(row["path"])
    payload = str(row["utf8"]).encode("utf-8")
    path = stage / relative
    if os.path.lexists(path):
        current = _read_plain(stage, relative, label=f"terminal artifact {relative}")
        if current != payload:
            raise TerminalRecoveryError(f"terminal artifact drifted: {relative}")
        return
    try:
        ArtifactStore(stage).write_bytes_exclusive(relative, payload)
    except (FileExistsError, OSError, ValueError) as exc:
        raise TerminalRecoveryError(
            f"terminal artifact exclusive write failed: {relative}"
        ) from exc
    if _read_plain(stage, relative, label=f"terminal artifact {relative}") != payload:
        raise TerminalRecoveryError(f"terminal artifact commit drifted: {relative}")


def _append_success_phase(
    stage: Path,
    *,
    state: str,
    receipt: Mapping[str, object],
    receipt_identity: Mapping[str, object],
    phase_rows: tuple[dict[str, object], ...],
) -> None:
    immutable = receipt["immutable_bindings"]
    global_checkpoint = receipt["global_checkpoint_identity"]
    assert isinstance(immutable, Mapping)
    assert isinstance(global_checkpoint, Mapping)
    validated_receipt_identity = _validate_identity(
        receipt_identity,
        label="success phase preterminal acceptance",
    )
    evidence_binding = receipt.get("preterminal_evidence_binding")
    if not isinstance(evidence_binding, Mapping) or not _is_sha256(
        evidence_binding.get("graph_sha256")
    ):
        raise TerminalRecoveryError("success phase evidence binding drifted")
    bindings = {
        **dict(immutable),
        "checkpoint_sha256": global_checkpoint["checkpoint_sha256"],
        "preterminal_acceptance_sha256": validated_receipt_identity["sha256"],
        "preterminal_evidence_graph_sha256": evidence_binding["graph_sha256"],
    }
    previous = str(phase_rows[-1]["record_hash"])
    event = {
        "state": state,
        "previous_record_hash": previous,
        "bindings": bindings,
    }
    row = {**event, "record_hash": _record_hash(event)}
    try:
        DurableJsonl(stage / PHASE_NAME).append(row)
    except (DurableJsonlError, OSError) as exc:
        raise TerminalRecoveryError("terminal phase durable append failed") from exc


@_guarded_formal_recovery_mutation
def recover_stage6_terminal_commit(
    *,
    stage_root: str | Path,
    manifest_committer: Callable[[Path], Mapping[str, object]],
    execution_capability: object,
    planning_child_recovery_capability: object | None = None,
) -> dict[str, object]:
    """Idempotently publish receipt-fixed bytes after terminal evidence.

    ``manifest_committer`` is the only callback.  It must be workload-free and
    must write-or-verify ``manifest.json`` before returning its exact
    ``{sha256, size_bytes}`` identity.  No semantic callback is accepted.
    """

    if not callable(manifest_committer):
        raise TerminalRecoveryError("manifest committer must be a pure callback")
    stage = _require_stage_root(stage_root)
    detection = detect_stage6_terminal_recovery(
        stage_root=stage,
        manifest_verifier=(
            manifest_committer if os.path.lexists(stage / MANIFEST_NAME) else None
        ),
        planning_child_recovery_capability=(
            planning_child_recovery_capability
        ),
    )
    if detection["status"] != "valid_terminal_recovery":
        raise TerminalRecoveryError(
            f"terminal recovery prefix is {detection['status']}: {detection['reason']}"
        )
    context = _load_terminal_context_for_planning_child(
        stage,
        manifest_verifier=(
            manifest_committer if os.path.lexists(stage / MANIFEST_NAME) else None
        ),
        planning_child_recovery_capability=(
            planning_child_recovery_capability
        ),
    )
    receipt = context["receipt"]
    assert isinstance(receipt, Mapping)
    artifact_rows = receipt["terminal_artifacts"]
    assert isinstance(artifact_rows, list)
    for row in artifact_rows:
        _require_recovery_execution_capability(
            execution_capability,
            stage_root=stage,
            label=f"before terminal artifact write: {row['path']}",
        )
        _write_terminal_artifact(stage, row)
        _require_recovery_execution_capability(
            execution_capability,
            stage_root=stage,
            label=f"after terminal artifact write: {row['path']}",
        )
        _terminal_recovery_event(f"after_artifact:{row['path']}")

    while True:
        context = _load_terminal_context_for_planning_child(
            stage,
            manifest_verifier=(
                manifest_committer if os.path.lexists(stage / MANIFEST_NAME) else None
            ),
            planning_child_recovery_capability=(
                planning_child_recovery_capability
            ),
        )
        phase_rows = context["phase_rows"]
        assert isinstance(phase_rows, tuple)
        states = tuple(str(row["state"]) for row in phase_rows)
        if states == PRETERMINAL_STATES:
            state = "machine_passed"
        elif states == COMPLETE_STATES[:6]:
            state = "awaiting_independent_review"
        elif states == COMPLETE_STATES:
            break
        else:  # guarded by _load_terminal_context
            raise TerminalRecoveryError("terminal phase prefix drifted")
        _require_recovery_execution_capability(
            execution_capability,
            stage_root=stage,
            label=f"before terminal phase append: {state}",
        )
        _append_success_phase(
            stage,
            state=state,
            receipt=receipt,
            receipt_identity=context["receipt_identity"],
            phase_rows=phase_rows,
        )
        _require_recovery_execution_capability(
            execution_capability,
            stage_root=stage,
            label=f"after terminal phase append: {state}",
        )
        _load_terminal_context_for_planning_child(
            stage,
            manifest_verifier=None,
            planning_child_recovery_capability=(
                planning_child_recovery_capability
            ),
        )
        _terminal_recovery_event(f"after_phase:{state}")

    _require_recovery_execution_capability(
        execution_capability,
        stage_root=stage,
        label="before terminal manifest callback",
    )
    manifest_result = manifest_committer(stage)
    _require_recovery_execution_capability(
        execution_capability,
        stage_root=stage,
        label="after terminal manifest callback",
    )
    manifest_identity = _validate_identity(
        manifest_result, label="manifest committer"
    )
    manifest_payload = _read_plain(stage, MANIFEST_NAME, label="Stage 6 terminal manifest")
    if _identity(manifest_payload) != manifest_identity:
        raise TerminalRecoveryError("manifest committer returned a drifted identity")
    _terminal_recovery_event("after_manifest")
    final = _load_terminal_context_for_planning_child(
        stage,
        manifest_verifier=manifest_committer,
        planning_child_recovery_capability=(
            planning_child_recovery_capability
        ),
    )
    final_rows = final["phase_rows"]
    assert isinstance(final_rows, tuple)
    final_states = tuple(str(row["state"]) for row in final_rows)
    if final_states != COMPLETE_STATES:
        raise TerminalRecoveryError("terminal commit did not converge")
    return {
        "receipt_identity": final["receipt_identity"],
        "phase_states": final_states,
        "manifest_identity": final["manifest_identity"],
    }


__all__ = [
    "COMPLETE_STATES",
    "PRETERMINAL_STATES",
    "RECEIPT_NAME",
    "TERMINAL_ARTIFACT_NAMES",
    "TERMINAL_RECEIPT_FIELD",
    "TerminalRecoveryError",
    "append_stage6_recovery_resource_segment",
    "append_stage6_recovery_resource_terminal",
    "detect_stage6_terminal_recovery",
    "recover_stage6_terminal_commit",
    "write_stage6_preterminal_acceptance",
]
