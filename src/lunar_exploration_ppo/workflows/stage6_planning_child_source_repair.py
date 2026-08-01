"""Fail-closed source-repair bridge for an existing Stage 6 planning child."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.path_security import (
    PathSecurityError,
    lexical_absolute,
    require_plain_path,
    secure_read_bytes,
)
from lunar_exploration_ppo.utils.resource_lifecycle import (
    ResourceLifecycleError,
    validate_resource_lifecycle_ledger,
    validate_resource_lifecycle_rows,
)


PLANNING_CHILD_SOURCE_REPAIR_NAME: Final = (
    "planning-child-source-repair.json"
)
PLANNING_CHILD_SOURCE_REPAIR_SCHEMA: Final = (
    "stage6_planning_child_source_repair/v1"
)
PLANNING_CHILD_SOURCE_REPAIR_MODE: Final = (
    "same_run_exact_resume_after_reviewed_source_repair/v1"
)
PLANNING_CHILD_SOURCE_REPAIR_ACCEPTANCE_SCHEMA: Final = (
    "stage6_planning_child_source_repair_acceptance/v1"
)
LEGACY_PATH_PLANNER_RUNTIME_SCHEMA: Final = (
    "stage6_path_planner_legacy_runtime_source_set/v1"
)
LEGACY_PATH_PLANNER_DISTRIBUTION: Final = "path-planner"
LEGACY_PATH_PLANNER_FORBIDDEN_PREFIX: Final = "path_planner.v2"
ORIGIN_LINEAGE_SCHEMA: Final = "stage6_lineage_audit/v3"
FIRST_CHILD_UPDATE: Final = 75
LAST_ORIGIN_UPDATE: Final = 84
FIRST_REPAIRED_UPDATE: Final = 85
FINAL_CHILD_UPDATE: Final = 100
ABANDONED_ATTEMPT: Final = 1
NEXT_ATTEMPT: Final = 2

_REQUIRED_PREFIX_PATHS: Final = (
    "checkpoints/index.jsonl",
    "job-state.jsonl",
    "resource_audit.jsonl",
    "training_metrics.jsonl",
    "validation_metrics.jsonl",
)
_LEGACY_PATH_PLANNER_ROOT_FILES: Final = ("__init__.py", "platform.py")
_LEGACY_PATH_PLANNER_SOURCE_DIRECTORIES: Final = (
    "core",
    "postprocess",
    "regions",
    "search",
    "trajectory",
)
_LEGACY_PATH_PLANNER_SYMBOLS: Final = (
    "AStarPlanner",
    "Cell",
    "CostGrid",
    "GridSpec",
    "NeighborPolicy",
    "PlanRequest",
)
_EXCLUDED_PLANNER_SOURCE_PARTS: Final = ("__pycache__", "test", "tests")


class Stage6PlanningChildSourceRepairError(RuntimeError):
    """The planning-child source-repair contract is invalid or drifted."""


def _plain_planner_package_root(path: str | Path) -> Path:
    try:
        root = require_plain_path(
            path,
            leaf_kind="directory",
            label="legacy path_planner package root",
        )
    except (OSError, PathSecurityError) as exc:
        raise Stage6PlanningChildSourceRepairError(
            "legacy path_planner package root is unsafe or missing"
        ) from exc
    if root.name != "path_planner":
        raise Stage6PlanningChildSourceRepairError(
            "legacy path_planner package root drifted"
        )
    return root


def _is_included_planner_source(relative: Path) -> bool:
    if relative.suffix != ".py" or relative.is_absolute():
        return False
    if relative.name.startswith("test_"):
        return False
    if any(part in _EXCLUDED_PLANNER_SOURCE_PARTS for part in relative.parts):
        return False
    if relative.as_posix() in _LEGACY_PATH_PLANNER_ROOT_FILES:
        return True
    return (
        len(relative.parts) >= 2
        and relative.parts[0] in _LEGACY_PATH_PLANNER_SOURCE_DIRECTORIES
    )


def _legacy_planner_source_members(package_root: Path) -> tuple[Path, ...]:
    members: list[Path] = []
    for relative_text in _LEGACY_PATH_PLANNER_ROOT_FILES:
        relative = Path(relative_text)
        path = package_root / relative
        if not path.is_file():
            raise Stage6PlanningChildSourceRepairError(
                f"legacy path_planner membership missing {relative.as_posix()}"
            )
        members.append(path)
    for directory_name in _LEGACY_PATH_PLANNER_SOURCE_DIRECTORIES:
        source_root = package_root / directory_name
        try:
            plain_source_root = require_plain_path(
                source_root,
                base=package_root,
                leaf_kind="directory",
                label=f"legacy path_planner {directory_name} source root",
            )
        except (OSError, PathSecurityError) as exc:
            raise Stage6PlanningChildSourceRepairError(
                f"legacy path_planner membership missing {directory_name}"
            ) from exc
        for current, directory_names, file_names in os.walk(
            plain_source_root,
            followlinks=False,
        ):
            current_path = Path(current)
            directory_names[:] = sorted(
                name
                for name in directory_names
                if name not in _EXCLUDED_PLANNER_SOURCE_PARTS
                and name != "v2"
            )
            for file_name in sorted(file_names):
                candidate = current_path / file_name
                relative = candidate.relative_to(package_root)
                if _is_included_planner_source(relative):
                    members.append(candidate)
    ordered = tuple(
        sorted(
            {path.relative_to(package_root).as_posix(): path for path in members}.values(),
            key=lambda path: path.relative_to(package_root).as_posix(),
        )
    )
    if not ordered:
        raise Stage6PlanningChildSourceRepairError(
            "legacy path_planner membership is empty"
        )
    return ordered


def _validate_planner_runtime_identity_record(
    value: object,
) -> dict[str, object]:
    identity = _require_mapping(value, "legacy path_planner runtime identity")
    if (
        identity.get("schema_version")
        != LEGACY_PATH_PLANNER_RUNTIME_SCHEMA
        or identity.get("distribution_name")
        != LEGACY_PATH_PLANNER_DISTRIBUTION
        or identity.get("forbidden_loaded_prefix")
        != LEGACY_PATH_PLANNER_FORBIDDEN_PREFIX
    ):
        raise Stage6PlanningChildSourceRepairError(
            "legacy path_planner runtime identity schema drifted"
        )
    distribution_version = _require_string(
        identity.get("distribution_version"),
        "legacy path_planner distribution version",
    )
    package_root_text = _require_string(
        identity.get("package_root"),
        "legacy path_planner package root",
    )
    package_root = lexical_absolute(package_root_text)
    if package_root_text != package_root.as_posix():
        raise Stage6PlanningChildSourceRepairError(
            "legacy path_planner package root is not canonical"
        )
    path_values = identity.get("paths")
    if not isinstance(path_values, list) or not path_values:
        raise Stage6PlanningChildSourceRepairError(
            "legacy path_planner membership is missing"
        )
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, value_row in enumerate(path_values):
        row = _require_mapping(
            value_row,
            f"legacy path_planner source member {index}",
        )
        relative = _require_string(
            row.get("path"),
            f"legacy path_planner source member {index} path",
        )
        relative_path = Path(relative)
        if (
            relative != relative_path.as_posix()
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or not _is_included_planner_source(relative_path)
            or relative in seen
        ):
            raise Stage6PlanningChildSourceRepairError(
                "legacy path_planner membership contains an invalid path"
            )
        seen.add(relative)
        rows.append(
            {
                "path": relative,
                "size_bytes": _require_int(
                    row.get("size_bytes"),
                    f"legacy path_planner {relative} size",
                ),
                "sha256": _require_sha256(
                    row.get("sha256"),
                    f"legacy path_planner {relative} SHA",
                ),
            }
        )
    if [str(row["path"]) for row in rows] != sorted(seen):
        raise Stage6PlanningChildSourceRepairError(
            "legacy path_planner membership ordering drifted"
        )
    source_set_sha256 = _require_sha256(
        identity.get("source_set_sha256"),
        "legacy path_planner source-set SHA",
    )
    if source_set_sha256 != _canonical_sha256(rows):
        raise Stage6PlanningChildSourceRepairError(
            "legacy path_planner source-set bytes digest drifted"
        )
    resolved = _require_mapping(
        identity.get("resolved_symbols"),
        "legacy path_planner resolved symbols",
    )
    if set(resolved) != set(_LEGACY_PATH_PLANNER_SYMBOLS):
        raise Stage6PlanningChildSourceRepairError(
            "legacy path_planner resolved symbol set drifted"
        )
    resolved_symbols: dict[str, str] = {}
    for symbol in _LEGACY_PATH_PLANNER_SYMBOLS:
        relative = _require_string(
            resolved.get(symbol),
            f"legacy path_planner resolved symbol {symbol}",
        )
        if relative not in seen:
            raise Stage6PlanningChildSourceRepairError(
                f"legacy path_planner resolved symbol {symbol} escaped membership"
            )
        resolved_symbols[symbol] = relative
    return {
        "schema_version": LEGACY_PATH_PLANNER_RUNTIME_SCHEMA,
        "distribution_name": LEGACY_PATH_PLANNER_DISTRIBUTION,
        "distribution_version": distribution_version,
        "package_root": package_root.as_posix(),
        "source_set_sha256": source_set_sha256,
        "paths": rows,
        "resolved_symbols": resolved_symbols,
        "forbidden_loaded_prefix": LEGACY_PATH_PLANNER_FORBIDDEN_PREFIX,
    }


def build_legacy_path_planner_runtime_identity(
    *,
    package_root: str | Path,
    distribution_version: str,
    resolved_symbol_paths: Mapping[str, str | Path],
    loaded_module_names: Sequence[str],
) -> dict[str, object]:
    """Content-address the exact legacy planner package used by Stage 6."""

    forbidden = sorted(
        name
        for name in loaded_module_names
        if name == LEGACY_PATH_PLANNER_FORBIDDEN_PREFIX
        or name.startswith(f"{LEGACY_PATH_PLANNER_FORBIDDEN_PREFIX}.")
    )
    if forbidden:
        raise Stage6PlanningChildSourceRepairError(
            "legacy path_planner forbidden-v2 module is loaded"
        )
    version = _require_string(
        distribution_version,
        "legacy path_planner distribution version",
    )
    root = _plain_planner_package_root(package_root)
    members = _legacy_planner_source_members(root)
    rows: list[dict[str, object]] = []
    member_relatives: set[str] = set()
    for path in members:
        relative = path.relative_to(root).as_posix()
        _, payload = _secure_payload(
            path,
            label=f"legacy path_planner source {relative}",
            base=root,
        )
        rows.append(
            {
                "path": relative,
                "size_bytes": len(payload),
                "sha256": _sha256(payload),
            }
        )
        member_relatives.add(relative)
    if set(resolved_symbol_paths) != set(_LEGACY_PATH_PLANNER_SYMBOLS):
        raise Stage6PlanningChildSourceRepairError(
            "legacy path_planner resolved symbol set drifted"
        )
    resolved: dict[str, str] = {}
    for symbol in _LEGACY_PATH_PLANNER_SYMBOLS:
        symbol_path = lexical_absolute(resolved_symbol_paths[symbol])
        try:
            plain_symbol_path = require_plain_path(
                symbol_path,
                base=root,
                leaf_kind="file",
                label=f"legacy path_planner resolved symbol {symbol}",
            )
            relative = plain_symbol_path.relative_to(root).as_posix()
        except (OSError, PathSecurityError, ValueError) as exc:
            raise Stage6PlanningChildSourceRepairError(
                f"legacy path_planner resolved symbol {symbol} escaped package root"
            ) from exc
        if relative not in member_relatives:
            raise Stage6PlanningChildSourceRepairError(
                f"legacy path_planner resolved symbol {symbol} is outside source-set membership"
            )
        resolved[symbol] = relative
    return _validate_planner_runtime_identity_record(
        {
            "schema_version": LEGACY_PATH_PLANNER_RUNTIME_SCHEMA,
            "distribution_name": LEGACY_PATH_PLANNER_DISTRIBUTION,
            "distribution_version": version,
            "package_root": root.as_posix(),
            "source_set_sha256": _canonical_sha256(rows),
            "paths": rows,
            "resolved_symbols": resolved,
            "forbidden_loaded_prefix": LEGACY_PATH_PLANNER_FORBIDDEN_PREFIX,
        }
    )


def resolve_legacy_path_planner_runtime_identity() -> dict[str, object]:
    """Resolve the actual editable planner closure used by this interpreter."""

    try:
        import path_planner
        from lunar_exploration_ppo.integrations import path_planner_adapter

        package_file = inspect.getsourcefile(path_planner)
        if package_file is None:
            raise Stage6PlanningChildSourceRepairError(
                "legacy path_planner package root could not be resolved"
            )
        resolved_symbol_paths: dict[str, Path] = {}
        for symbol in _LEGACY_PATH_PLANNER_SYMBOLS:
            symbol_file = inspect.getsourcefile(
                getattr(path_planner_adapter, symbol)
            )
            if symbol_file is None:
                raise Stage6PlanningChildSourceRepairError(
                    f"legacy path_planner resolved symbol {symbol} has no source"
                )
            resolved_symbol_paths[symbol] = Path(symbol_file)
        distribution_version = importlib.metadata.version(
            LEGACY_PATH_PLANNER_DISTRIBUTION
        )
    except Stage6PlanningChildSourceRepairError:
        raise
    except (AttributeError, ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise Stage6PlanningChildSourceRepairError(
            "legacy path_planner runtime could not be resolved"
        ) from exc
    return build_legacy_path_planner_runtime_identity(
        package_root=Path(package_file).parent,
        distribution_version=distribution_version,
        resolved_symbol_paths=resolved_symbol_paths,
        loaded_module_names=tuple(sys.modules),
    )


def require_legacy_path_planner_runtime_identity_current(
    identity: Mapping[str, object],
    *,
    label: str,
    package_root: str | Path | None = None,
    distribution_version: str | None = None,
    resolved_symbol_paths: Mapping[str, str | Path] | None = None,
    loaded_module_names: Sequence[str] | None = None,
) -> None:
    """Fail closed when the live planner closure differs from its binding."""

    _require_string(label, "legacy path_planner currentness label")
    expected = _validate_planner_runtime_identity_record(identity)
    explicit = (
        package_root,
        distribution_version,
        resolved_symbol_paths,
        loaded_module_names,
    )
    if all(value is None for value in explicit):
        observed = resolve_legacy_path_planner_runtime_identity()
    elif any(value is None for value in explicit):
        raise Stage6PlanningChildSourceRepairError(
            "legacy path_planner explicit currentness inputs are incomplete"
        )
    else:
        observed = build_legacy_path_planner_runtime_identity(
            package_root=package_root,
            distribution_version=distribution_version,
            resolved_symbol_paths=resolved_symbol_paths,
            loaded_module_names=loaded_module_names,
        )
    if observed["package_root"] != expected["package_root"]:
        raise Stage6PlanningChildSourceRepairError(
            f"legacy path_planner package root drifted at {label}"
        )
    expected_paths = {
        str(row["path"]): row
        for row in expected["paths"]
        if isinstance(row, Mapping)
    }
    observed_paths = {
        str(row["path"]): row
        for row in observed["paths"]
        if isinstance(row, Mapping)
    }
    if set(observed_paths) != set(expected_paths):
        raise Stage6PlanningChildSourceRepairError(
            f"legacy path_planner membership drifted at {label}"
        )
    if observed_paths != expected_paths:
        raise Stage6PlanningChildSourceRepairError(
            f"legacy path_planner member bytes drifted at {label}"
        )
    if observed["resolved_symbols"] != expected["resolved_symbols"]:
        raise Stage6PlanningChildSourceRepairError(
            f"legacy path_planner resolved-symbol drifted at {label}"
        )
    if observed["distribution_version"] != expected["distribution_version"]:
        raise Stage6PlanningChildSourceRepairError(
            f"legacy path_planner distribution version drifted at {label}"
        )
    if observed["source_set_sha256"] != expected["source_set_sha256"]:
        raise Stage6PlanningChildSourceRepairError(
            f"legacy path_planner source-set bytes drifted at {label}"
        )


def _planner_runtime_immutable_bindings(
    identity: Mapping[str, object],
    *,
    label: str,
) -> dict[str, dict[str, object]]:
    runtime = _validate_planner_runtime_identity_record(identity)
    root = _plain_planner_package_root(
        _require_string(
            runtime.get("package_root"),
            "legacy path_planner package root",
        )
    )
    bindings: dict[str, dict[str, object]] = {}
    for value in runtime["paths"]:
        row = _require_mapping(value, "legacy path_planner source member")
        relative = _require_string(
            row.get("path"),
            "legacy path_planner source member path",
        )
        path, payload = _secure_payload(
            root / relative,
            label=f"{label} legacy path_planner source {relative}",
            base=root,
        )
        if (
            len(payload) != row.get("size_bytes")
            or _sha256(payload) != row.get("sha256")
        ):
            raise Stage6PlanningChildSourceRepairError(
                f"legacy path_planner member bytes drifted at {label}"
            )
        bindings[f"planner_runtime::{relative}"] = _file_binding(
            path,
            payload,
        )
    return bindings


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256(ArtifactStore.canonical_json_bytes(value))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _canonical_jsonl_row(value: object) -> bytes:
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


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Stage6PlanningChildSourceRepairError(
            f"{label} must be a JSON object"
        )
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Stage6PlanningChildSourceRepairError(
            f"{label} must be a non-empty string"
        )
    return value


def _require_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise Stage6PlanningChildSourceRepairError(
            f"{label} must be an integer"
        )
    return value


def _require_sha256(value: object, label: str) -> str:
    digest = _require_string(value, label)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise Stage6PlanningChildSourceRepairError(
            f"{label} must be a lowercase SHA-256"
        )
    return digest


def _secure_payload(
    path: str | Path,
    *,
    label: str,
    base: str | Path | None = None,
) -> tuple[Path, bytes]:
    lexical = lexical_absolute(path)
    try:
        result = secure_read_bytes(lexical, base=base, label=label)
    except (OSError, PathSecurityError) as exc:
        raise Stage6PlanningChildSourceRepairError(
            f"{label} could not be securely read"
        ) from exc
    return lexical, result.payload


def _json_mapping(payload: bytes, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage6PlanningChildSourceRepairError(
            f"{label} is not valid UTF-8 JSON"
        ) from exc
    return _require_mapping(value, label)


def _jsonl_rows(payload: bytes, label: str) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    try:
        lines = payload.splitlines(keepends=True)
        if b"".join(lines) != payload:
            raise ValueError("trailing JSONL bytes")
        for line_number, line in enumerate(lines, start=1):
            value = json.loads(
                line.decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
            if (
                not isinstance(value, Mapping)
                or _canonical_jsonl_row(value) != line
            ):
                raise ValueError(
                    f"noncanonical JSONL row {line_number}"
                )
            rows.append(value)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        raise Stage6PlanningChildSourceRepairError(
            f"{label} is not canonical UTF-8 JSONL"
        ) from exc
    return rows


def _file_binding(path: Path, payload: bytes) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "size_bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _prefix_binding(
    stage_root: Path,
    path: Path,
    payload: bytes,
) -> dict[str, object]:
    return {
        "path": path.relative_to(stage_root).as_posix(),
        "prefix_size_bytes": len(payload),
        "prefix_sha256": _sha256(payload),
        "line_count": payload.count(b"\n"),
    }


def _transaction_key(ordinal: int, seed: int, update: int) -> str:
    return f"{ordinal:04d}:{seed}:update:{update:03d}"


def _expected_updates() -> list[int]:
    return list(range(FIRST_CHILD_UPDATE, LAST_ORIGIN_UPDATE + 1))


def _validate_review(
    path: Path,
    payload: bytes,
    *,
    review_name: str,
) -> dict[str, object]:
    review = _json_mapping(payload, f"{review_name} review")
    counts = _require_mapping(
        review.get("finding_counts"),
        f"{review_name} review finding_counts",
    )
    critical = _require_int(
        counts.get("critical"),
        f"{review_name} review critical count",
    )
    important = _require_int(
        counts.get("important"),
        f"{review_name} review important count",
    )
    if review.get("verdict") != "PASS" or critical != 0 or important != 0:
        raise Stage6PlanningChildSourceRepairError(
            f"{review_name} review must PASS C0/I0"
        )
    return {
        **_file_binding(path, payload),
        "verdict": "PASS",
        "finding_counts": {
            "critical": critical,
            "important": important,
            "minor": _require_int(
                counts.get("minor"),
                f"{review_name} review minor count",
            ),
        },
    }


def _validate_current_authorization(
    *,
    formal_run_id: str,
    execution_identity: Mapping[str, object],
    verified_authorization: Mapping[str, object],
    immutable_bindings: Mapping[str, object],
    authorization_path: Path,
    authorization_payload: bytes,
) -> tuple[str, str]:
    identity_sha256 = _canonical_sha256(execution_identity)
    if verified_authorization.get("authorized") is not True:
        raise Stage6PlanningChildSourceRepairError(
            "current review authorization is not authorized"
        )
    if verified_authorization.get("formal_run_id") != formal_run_id:
        raise Stage6PlanningChildSourceRepairError(
            "current review authorization formal run id drifted"
        )
    authorization_sha256 = _sha256(authorization_payload)
    if (
        verified_authorization.get("authorization_file_sha256")
        != authorization_sha256
        or verified_authorization.get("authorization_file_size_bytes")
        != len(authorization_payload)
    ):
        raise Stage6PlanningChildSourceRepairError(
            "current review authorization file binding drifted"
        )
    if (
        verified_authorization.get("review_identity_sha256")
        != identity_sha256
    ):
        raise Stage6PlanningChildSourceRepairError(
            "current execution identity review binding drifted"
        )
    if immutable_bindings.get("formal_run_id") != formal_run_id:
        raise Stage6PlanningChildSourceRepairError(
            "current immutable formal run id drifted"
        )
    for key in ("source_set_sha256", "config_sha256"):
        if immutable_bindings.get(key) != execution_identity.get(key):
            raise Stage6PlanningChildSourceRepairError(
                f"current immutable {key} drifted"
            )
    if (
        immutable_bindings.get("authorization_file_sha256")
        != authorization_sha256
        or immutable_bindings.get("review_identity_sha256")
        != identity_sha256
    ):
        raise Stage6PlanningChildSourceRepairError(
            "current immutable authorization binding drifted"
        )
    return identity_sha256, authorization_sha256


def _validate_origin(
    *,
    stage_root: Path,
    formal_run_id: str,
    seed: int,
    planning_warm_start_path: Path,
    planning_warm_start_payload: bytes,
    lineage_path: Path,
    lineage_payload: bytes,
    config_payload: bytes,
) -> tuple[dict[str, object], Mapping[str, object]]:
    lineage = _json_mapping(lineage_payload, "origin lineage")
    if lineage.get("schema_version") != ORIGIN_LINEAGE_SCHEMA:
        raise Stage6PlanningChildSourceRepairError(
            "origin lineage schema drifted"
        )
    origin_identity = _require_mapping(
        lineage.get("execution_identity"),
        "origin execution identity",
    )
    origin_immutable = _require_mapping(
        lineage.get("immutable_bindings"),
        "origin immutable bindings",
    )
    origin_verified = _require_mapping(
        lineage.get("verified_review_authorization"),
        "origin verified review authorization",
    )
    if (
        origin_immutable.get("formal_run_id") != formal_run_id
        or origin_verified.get("formal_run_id") != formal_run_id
    ):
        raise Stage6PlanningChildSourceRepairError(
            "origin formal run id drifted"
        )
    config_sha256 = _sha256(config_payload)
    if (
        origin_identity.get("config_sha256") != config_sha256
        or origin_immutable.get("config_sha256") != config_sha256
    ):
        raise Stage6PlanningChildSourceRepairError(
            "origin config binding drifted"
        )
    origin_identity_sha256 = _canonical_sha256(origin_identity)
    if (
        origin_verified.get("review_identity_sha256")
        != origin_identity_sha256
        or origin_immutable.get("review_identity_sha256")
        != origin_identity_sha256
    ):
        raise Stage6PlanningChildSourceRepairError(
            "origin execution identity binding drifted"
        )

    warm_start = _json_mapping(
        planning_warm_start_payload,
        "planning warm-start",
    )
    lineage_warm = _require_mapping(
        lineage.get("planning_warm_start"),
        "origin planning warm-start binding",
    )
    if lineage_warm.get("artifact") != warm_start:
        raise Stage6PlanningChildSourceRepairError(
            "origin planning warm-start payload drifted"
        )
    if (
        lineage_warm.get("artifact_sha256")
        != _sha256(planning_warm_start_payload)
        or lineage_warm.get("artifact_size_bytes")
        != len(planning_warm_start_payload)
    ):
        raise Stage6PlanningChildSourceRepairError(
            "origin planning warm-start file binding drifted"
        )
    child = _require_mapping(
        warm_start.get("child"),
        "planning warm-start child",
    )
    if (
        child.get("run_id") != formal_run_id
        or child.get("seed") != seed
        or child.get("first_update") != FIRST_CHILD_UPDATE
        or child.get("final_update") != FINAL_CHILD_UPDATE
    ):
        raise Stage6PlanningChildSourceRepairError(
            "planning warm-start child binding drifted"
        )
    parent = _require_mapping(
        warm_start.get("parent"),
        "planning warm-start parent",
    )

    origin_authorization_path = planning_warm_start_path.parent / (
        "launch-authorization.json"
    )
    origin_authorization_path, origin_authorization_payload = _secure_payload(
        origin_authorization_path,
        label="origin launch authorization",
    )
    if (
        origin_verified.get("authorization_file_sha256")
        != _sha256(origin_authorization_payload)
        or origin_verified.get("authorization_file_size_bytes")
        != len(origin_authorization_payload)
        or origin_immutable.get("authorization_file_sha256")
        != _sha256(origin_authorization_payload)
    ):
        raise Stage6PlanningChildSourceRepairError(
            "origin launch authorization binding drifted"
        )

    return (
        {
            "lineage": {
                **_file_binding(lineage_path, lineage_payload),
                "schema_version": ORIGIN_LINEAGE_SCHEMA,
            },
            "launch_authorization": _file_binding(
                origin_authorization_path,
                origin_authorization_payload,
            ),
            "execution_identity": dict(origin_identity),
            "execution_identity_sha256": origin_identity_sha256,
            "immutable_bindings": dict(origin_immutable),
            "verified_review_authorization": dict(origin_verified),
            "planning_warm_start": {
                **_file_binding(
                    planning_warm_start_path,
                    planning_warm_start_payload,
                ),
                "artifact": dict(warm_start),
            },
            "parent": dict(parent),
        },
        warm_start,
    )


def _validate_accepted_prefix(
    *,
    stage_root: Path,
    seed: int,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, dict[str, object]],
]:
    prefix_payloads: dict[str, tuple[Path, bytes]] = {}
    for relative_path in _REQUIRED_PREFIX_PATHS:
        path, payload = _secure_payload(
            stage_root / relative_path,
            label=f"planning child journal {relative_path}",
            base=stage_root,
        )
        prefix_payloads[relative_path] = (path, payload)

    receipt_rows = _jsonl_rows(
        prefix_payloads["checkpoints/index.jsonl"][1],
        "checkpoint index",
    )
    expected_updates = _expected_updates()
    receipt_updates = [
        _require_int(row.get("update"), "checkpoint index update")
        for row in receipt_rows
        if row.get("seed") == seed
    ]
    if receipt_updates != expected_updates:
        raise Stage6PlanningChildSourceRepairError(
            "checkpoint index must contain exactly continuous updates 75..84"
        )
    receipt_by_update = {
        _require_int(row.get("update"), "checkpoint index update"): row
        for row in receipt_rows
        if row.get("seed") == seed
    }

    training_rows = _jsonl_rows(
        prefix_payloads["training_metrics.jsonl"][1],
        "training metrics",
    )
    training_updates = [
        _require_int(row.get("update"), "training metrics update")
        for row in training_rows
        if row.get("seed") == seed
    ]
    if training_updates != expected_updates:
        raise Stage6PlanningChildSourceRepairError(
            "training metrics must contain exactly continuous updates 75..84"
        )
    training_by_update = {
        _require_int(row.get("update"), "training metrics update"): row
        for row in training_rows
        if row.get("seed") == seed
    }

    job_rows = _jsonl_rows(
        prefix_payloads["job-state.jsonl"][1],
        "job state",
    )
    job_updates: list[int] = []
    for row in job_rows:
        state = row.get("state")
        marker = f"seed_{seed}_training_update_"
        if isinstance(state, str) and state.startswith(marker):
            try:
                job_updates.append(int(state.removeprefix(marker)))
            except ValueError as exc:
                raise Stage6PlanningChildSourceRepairError(
                    "job state training update is invalid"
                ) from exc
    if job_updates != expected_updates:
        raise Stage6PlanningChildSourceRepairError(
            "job state must contain exactly continuous updates 75..84"
        )

    resource_path, resource_payload = prefix_payloads[
        "resource_audit.jsonl"
    ]
    resource_rows = _jsonl_rows(resource_payload, "resource audit")
    u85_attempt1 = [
        row
        for row in resource_rows
        if row.get("seed") == seed
        and row.get("update") == FIRST_REPAIRED_UPDATE
        and row.get("attempt") == ABANDONED_ATTEMPT
    ]
    if (
        len(u85_attempt1) != 1
        or u85_attempt1[0].get("phase") != "pre"
        or u85_attempt1[0].get("accepted") is not False
    ):
        raise Stage6PlanningChildSourceRepairError(
            "U85 attempt1 must contain pre only"
        )
    try:
        resource_validation = validate_resource_lifecycle_ledger(
            resource_path,
            require_terminal=False,
        )
    except ResourceLifecycleError as exc:
        raise Stage6PlanningChildSourceRepairError(
            "resource audit lifecycle validation failed"
        ) from exc
    active_segment_index = _require_int(
        resource_validation.get("active_segment_index"),
        "resource validation active segment index",
    )
    accepted_resource_rows = [
        row
        for row in resource_rows
        if row.get("seed") == seed
        and row.get("phase") == "accepted"
        and row.get("accepted") is True
    ]
    resource_updates = [
        _require_int(row.get("update"), "resource accepted update")
        for row in accepted_resource_rows
    ]
    if resource_updates != expected_updates:
        raise Stage6PlanningChildSourceRepairError(
            "resource audit must contain exactly accepted updates 75..84"
        )
    resource_by_update = {
        _require_int(row.get("update"), "resource accepted update"): row
        for row in accepted_resource_rows
    }

    receipts: list[dict[str, object]] = []
    for ordinal, update in enumerate(expected_updates):
        expected_key = _transaction_key(ordinal, seed, update)
        receipt = receipt_by_update[update]
        training = training_by_update[update]
        resource = resource_by_update[update]
        for label, row in (
            ("checkpoint index", receipt),
            ("training metrics", training),
            ("resource audit", resource),
        ):
            if row.get("transaction_key") != expected_key:
                raise Stage6PlanningChildSourceRepairError(
                    f"{label} U{update} transaction key drifted"
                )
        if (
            training.get("checkpoint_sha256")
            != receipt.get("checkpoint_sha256")
            or training.get("policy_state_sha256")
            != receipt.get("policy_state_sha256")
        ):
            raise Stage6PlanningChildSourceRepairError(
                f"training metrics U{update} checkpoint binding drifted"
            )
        resource_checkpoint = _require_mapping(
            resource.get("checkpoint"),
            f"resource audit U{update} checkpoint",
        )
        for key in (
            "checkpoint_sha256",
            "complete_marker_sha256",
            "policy_state_sha256",
        ):
            if resource_checkpoint.get(key) != receipt.get(key):
                raise Stage6PlanningChildSourceRepairError(
                    f"resource audit U{update} {key} drifted"
                )
        receipts.append(dict(receipt))

    validation_rows = _jsonl_rows(
        prefix_payloads["validation_metrics.jsonl"][1],
        "validation metrics",
    )
    validation_updates = [
        row.get("update")
        for row in validation_rows
        if row.get("seed") == seed
    ]
    if validation_updates != [80]:
        raise Stage6PlanningChildSourceRepairError(
            "validation metrics must preserve the historical U80 record"
        )

    prefix_bindings = {
        relative_path: _prefix_binding(stage_root, path, payload)
        for relative_path, (path, payload) in prefix_payloads.items()
    }
    accepted_prefix = {
        "first_update": FIRST_CHILD_UPDATE,
        "last_update": LAST_ORIGIN_UPDATE,
        "receipt_count": len(receipts),
        "receipts": receipts,
        "validation_updates": [80],
        "resource_validation": dict(resource_validation),
    }
    resume_point = {
        "last_complete_update": LAST_ORIGIN_UPDATE,
        "next_update": FIRST_REPAIRED_UPDATE,
        "abandoned_attempt": ABANDONED_ATTEMPT,
        "next_attempt": NEXT_ATTEMPT,
        "abandoned_attempt_has_pre_only": True,
        "last_resource_segment_index": active_segment_index,
        "next_resource_segment_index": active_segment_index + 1,
        "next_transaction_key": _transaction_key(
            len(expected_updates),
            seed,
            FIRST_REPAIRED_UPDATE,
        ),
    }
    return accepted_prefix, resume_point, prefix_bindings


def _validate_u84_checkpoint(
    *,
    stage_root: Path,
    seed: int,
    accepted_prefix: Mapping[str, object],
    config_payload: bytes,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    checkpoint_root = (
        stage_root
        / "checkpoints"
        / f"seed-{seed}"
        / f"update-{LAST_ORIGIN_UPDATE:08d}"
    )
    members: dict[str, tuple[Path, bytes]] = {}
    for name in ("checkpoint.pt", "manifest.json", "complete.json"):
        members[name] = _secure_payload(
            checkpoint_root / name,
            label=f"U84 {name}",
            base=stage_root,
        )
    checkpoint_path, checkpoint_payload = members["checkpoint.pt"]
    manifest_path, manifest_payload = members["manifest.json"]
    complete_path, complete_payload = members["complete.json"]
    manifest = _json_mapping(manifest_payload, "U84 manifest")
    complete = _json_mapping(complete_payload, "U84 complete marker")
    checkpoint_sha256 = _sha256(checkpoint_payload)
    manifest_sha256 = _sha256(manifest_payload)
    complete_sha256 = _sha256(complete_payload)
    manifest_checkpoint = _require_mapping(
        manifest.get("checkpoint"),
        "U84 manifest checkpoint",
    )
    from lunar_exploration_ppo.ppo.checkpoint import (
        safe_load_checkpoint_payload,
    )

    try:
        checkpoint_record = safe_load_checkpoint_payload(
            checkpoint_payload
        )
    except Exception as exc:
        raise Stage6PlanningChildSourceRepairError(
            "U84 checkpoint payload could not be decoded"
        ) from exc
    if not isinstance(checkpoint_record, Mapping):
        raise Stage6PlanningChildSourceRepairError(
            "U84 checkpoint payload is not a mapping"
        )
    checkpoint_lineage = _require_mapping(
        checkpoint_record.get("lineage"),
        "U84 checkpoint lineage",
    )
    try:
        checkpoint_lineage_record = json.loads(
            ArtifactStore.canonical_json_bytes(
                checkpoint_lineage
            ).decode("utf-8")
        )
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage6PlanningChildSourceRepairError(
            "U84 checkpoint lineage is not canonical JSON data"
        ) from exc
    if (
        not isinstance(checkpoint_lineage_record, dict)
        or manifest.get("lineage_sha256")
        != _canonical_sha256(checkpoint_lineage_record)
    ):
        raise Stage6PlanningChildSourceRepairError(
            "U84 checkpoint lineage binding drifted"
        )
    if (
        manifest.get("update_step") != LAST_ORIGIN_UPDATE
        or manifest_checkpoint.get("sha256") != checkpoint_sha256
        or manifest_checkpoint.get("size_bytes") != len(checkpoint_payload)
        or manifest.get("config_sha256") != _sha256(config_payload)
    ):
        raise Stage6PlanningChildSourceRepairError(
            "U84 checkpoint manifest binding drifted"
        )
    if (
        complete.get("update_step") != LAST_ORIGIN_UPDATE
        or complete.get("checkpoint_sha256") != checkpoint_sha256
        or complete.get("manifest_sha256") != manifest_sha256
    ):
        raise Stage6PlanningChildSourceRepairError(
            "U84 complete marker binding drifted"
        )
    receipts = accepted_prefix.get("receipts")
    if not isinstance(receipts, Sequence) or not receipts:
        raise Stage6PlanningChildSourceRepairError(
            "accepted prefix receipts are missing"
        )
    u84_receipt = _require_mapping(receipts[-1], "U84 receipt")
    if (
        u84_receipt.get("update") != LAST_ORIGIN_UPDATE
        or u84_receipt.get("checkpoint_sha256") != checkpoint_sha256
        or u84_receipt.get("complete_marker_sha256") != complete_sha256
        or u84_receipt.get("policy_state_sha256")
        != manifest.get("policy_state_sha256")
    ):
        raise Stage6PlanningChildSourceRepairError(
            "U84 checkpoint bundle does not match the accepted receipt"
        )
    bindings = {
        "u84_checkpoint": _file_binding(checkpoint_path, checkpoint_payload),
        "u84_manifest": _file_binding(manifest_path, manifest_payload),
        "u84_complete": _file_binding(complete_path, complete_payload),
    }
    checkpoint = {
        "update": LAST_ORIGIN_UPDATE,
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "complete_sha256": complete_sha256,
        "policy_state_sha256": manifest.get("policy_state_sha256"),
        "config_sha256": manifest.get("config_sha256"),
        "lineage_sha256": manifest.get("lineage_sha256"),
        "lineage": checkpoint_lineage_record,
        "directory": checkpoint_root.as_posix(),
    }
    return checkpoint, bindings


def _validated_restart_tail_evidence(
    context: PlanningChildSourceRepairContext,
) -> tuple[
    dict[str, object],
    dict[str, Mapping[str, object]],
]:
    """Derive a restart boundary only from complete current-bound transactions."""

    from lunar_exploration_ppo.ppo.standard_training import (
        CheckpointReceiptIndex,
        StandardTrainingError,
        validate_standard_training_validation_rows,
        verify_journal_checkpoint_bindings,
    )
    from lunar_exploration_ppo.workflows.stage6 import (
        Stage6StateJournal,
        Stage6WorkflowError,
    )
    from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
        build_planning_child_transactions,
        parse_planning_effective_config_bytes,
    )

    payloads: dict[str, bytes] = {}
    prefixes = _require_mapping(
        context.artifact.get("append_only_prefixes"),
        "append-only prefixes",
    )
    for relative_path in _REQUIRED_PREFIX_PATHS:
        binding = _require_mapping(
            prefixes.get(relative_path),
            f"append-only prefix {relative_path}",
        )
        bound_relative = _require_string(
            binding.get("path"),
            f"append-only prefix {relative_path} path",
        )
        if bound_relative != relative_path:
            raise Stage6PlanningChildSourceRepairError(
                f"append-only prefix {relative_path} path drifted"
            )
        _, payload = _secure_payload(
            context.stage_root / relative_path,
            label=f"validated restart tail {relative_path}",
            base=context.stage_root,
        )
        context.verify_append_only_prefix_payload(
            relative_path,
            payload,
            label="validated restart tail",
            require_exact=False,
        )
        payloads[relative_path] = payload

    immutable_inputs = _require_mapping(
        context.artifact.get("immutable_inputs"),
        "immutable inputs",
    )
    config_binding = _require_mapping(
        immutable_inputs.get("config"),
        "immutable input config",
    )
    config_path = _require_string(
        config_binding.get("path"),
        "immutable input config path",
    )
    _, config_payload = _secure_payload(
        config_path,
        label="validated restart tail config",
    )
    try:
        effective = parse_planning_effective_config_bytes(config_payload)
        transactions = build_planning_child_transactions(
            effective.base_config
        )
        receipts = CheckpointReceiptIndex.verify_snapshot_bytes(
            payloads["checkpoints/index.jsonl"]
        )
        journal_rows = Stage6StateJournal.verify_snapshot_bytes(
            payloads["job-state.jsonl"]
        )
        completed = verify_journal_checkpoint_bindings(
            journal_rows,
            transactions,
            receipts,
            context.current_immutable_bindings,
            historical_immutable_bindings=(
                context.origin_immutable_bindings
                if not context.has_continuation
                else None
            ),
            current_binding_first_transaction_key=(
                context.next_transaction_key
                if not context.has_continuation
                else None
            ),
            immutable_binding_epochs=(
                context.journal_binding_epochs
                if context.has_continuation
                else None
            ),
        )
    except (
        Stage6WorkflowError,
        StandardTrainingError,
        TypeError,
        ValueError,
    ) as exc:
        raise Stage6PlanningChildSourceRepairError(
            "validated restart transaction journal/receipt drifted"
        ) from exc

    accepted_prefix = _require_mapping(
        context.artifact.get("accepted_prefix"),
        "accepted prefix",
    )
    frozen_receipt_count = _require_int(
        accepted_prefix.get("receipt_count"),
        "accepted prefix receipt count",
    )
    expected_completed = tuple(
        transaction.key for transaction in transactions[: len(completed)]
    )
    expected_journal_states = tuple(
        state
        for transaction in transactions[: len(completed)]
        for state in transaction.commit_states
    )
    if (
        completed != expected_completed
        or len(receipts) != len(completed)
        or len(completed) <= frozen_receipt_count
        or tuple(row.get("state") for row in journal_rows)
        != expected_journal_states
        or (
            len(completed) >= len(transactions)
            and not (
                context.has_continuation
                and len(completed) == len(transactions)
            )
        )
    ):
        raise Stage6PlanningChildSourceRepairError(
            "validated restart transactions are not a complete accepted prefix"
        )
    appended_transactions = tuple(
        transactions[frozen_receipt_count : len(completed)]
    )
    if (
        not appended_transactions
        or appended_transactions[0].update != FIRST_REPAIRED_UPDATE
        or appended_transactions[0].key != context.next_transaction_key
        or any(
            later.update != earlier.update + 1
            for earlier, later in zip(
                appended_transactions,
                appended_transactions[1:],
            )
        )
    ):
        raise Stage6PlanningChildSourceRepairError(
            "validated restart transaction order drifted"
        )

    training_rows = _jsonl_rows(
        payloads["training_metrics.jsonl"],
        "validated restart training metrics",
    )
    if len(training_rows) != len(completed):
        raise Stage6PlanningChildSourceRepairError(
            "validated restart training metrics are incomplete"
        )

    validation_rows = _jsonl_rows(
        payloads["validation_metrics.jsonl"],
        "validated restart validation metrics",
    )
    try:
        validate_standard_training_validation_rows(
            config=effective.base_config,
            transactions=transactions[: len(completed)],
            receipt_rows=receipts,
            training_rows=training_rows,
            validation_rows=validation_rows,
            planning_warm_start=True,
        )
    except StandardTrainingError as exc:
        raise Stage6PlanningChildSourceRepairError(
            "validated restart training/validation semantics drifted"
        ) from exc

    resource_rows = _jsonl_rows(
        payloads["resource_audit.jsonl"],
        "validated restart resource audit",
    )
    try:
        resource_validation = validate_resource_lifecycle_rows(
            resource_rows,
            require_terminal=False,
        )
    except ResourceLifecycleError as exc:
        raise Stage6PlanningChildSourceRepairError(
            "validated restart resource lifecycle drifted"
        ) from exc
    resource_binding = _require_mapping(
        prefixes.get("resource_audit.jsonl"),
        "append-only prefix resource_audit.jsonl",
    )
    frozen_resource_lines = _require_int(
        resource_binding.get("line_count"),
        "append-only prefix resource_audit.jsonl line count",
    )
    resource_tail = resource_rows[frozen_resource_lines:]
    update_rows = tuple(
        row for row in resource_tail if row.get("kind") == "update"
    )
    terminal_tail = (
        resource_validation.get("terminal_evidence_present") is True
    )
    terminal_tail_allowed = (
        context.has_continuation
        and len(completed) == len(transactions)
        and terminal_tail
    )
    if (
        not resource_tail
        or resource_tail[-1].get("phase")
        != ("terminal" if terminal_tail_allowed else "accepted")
        or (terminal_tail and not terminal_tail_allowed)
        or any(
            row.get("kind")
            not in {
                "resource_lifecycle_segment",
                "update",
                *(
                    {"terminal_lifecycle"}
                    if terminal_tail_allowed
                    else set()
                ),
            }
            for row in resource_tail
        )
        or len(update_rows) != 3 * len(appended_transactions)
    ):
        raise Stage6PlanningChildSourceRepairError(
            "validated restart resource tail is incomplete"
        )
    receipt_by_key = {
        str(receipt["transaction_key"]): receipt for receipt in receipts
    }
    for index, transaction in enumerate(appended_transactions):
        group = update_rows[index * 3 : index * 3 + 3]
        expected_attempt = (
            NEXT_ATTEMPT
            if transaction.update == FIRST_REPAIRED_UPDATE
            else 1
        )
        receipt = receipt_by_key[transaction.key]
        accepted_checkpoint = _require_mapping(
            group[2].get("checkpoint"),
            f"validated restart U{transaction.update} resource checkpoint",
        )
        if (
            tuple(row.get("phase") for row in group)
            != ("pre", "post", "accepted")
            or any(
                row.get("seed") != transaction.seed
                or row.get("update") != transaction.update
                or row.get("transaction_key") != transaction.key
                or row.get("attempt") != expected_attempt
                for row in group
            )
            or group[0].get("accepted") is not False
            or group[1].get("accepted") is not False
            or group[2].get("accepted") is not True
            or group[2].get("pre") != group[0].get("resource")
            or group[2].get("post") != group[1].get("resource")
            or any(
                accepted_checkpoint.get(key) != receipt[key]
                for key in (
                    "seed",
                    "update",
                    "transaction_key",
                    "checkpoint_sha256",
                    "complete_marker_sha256",
                    "policy_state_sha256",
                )
            )
        ):
            raise Stage6PlanningChildSourceRepairError(
                f"validated restart U{transaction.update} resource binding drifted"
            )

    from lunar_exploration_ppo.ppo.checkpoint import (
        CheckpointError,
        inspect_complete_checkpoint_snapshot,
    )
    from lunar_exploration_ppo.configs.stage6 import SafetyContract
    from lunar_exploration_ppo.workflows.stage6 import (
        _stage6_checkpoint_lineage_base,
        _stage6_planning_checkpoint_lineage_for_update,
    )
    from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
        Stage6PlanningWarmStartError,
        validate_planning_warm_start_artifact,
    )

    origin = _require_mapping(context.artifact.get("origin"), "origin")
    warm_binding = _require_mapping(
        origin.get("planning_warm_start"),
        "origin planning warm-start",
    )
    warm_artifact = _require_mapping(
        warm_binding.get("artifact"),
        "origin planning warm-start artifact",
    )
    warm_sha256 = _require_sha256(
        warm_binding.get("sha256"),
        "origin planning warm-start SHA",
    )
    expected_config_sha256 = _require_sha256(
        context.current_immutable_bindings.get("config_sha256"),
        "validated restart config SHA",
    )
    safety_contract = SafetyContract.from_stage6_config(
        effective.base_config
    )
    for transaction in appended_transactions[-1:]:
        receipt = receipt_by_key[transaction.key]
        checkpoint_root = (
            context.stage_root
            / "checkpoints"
            / f"seed-{transaction.seed}"
            / f"update-{transaction.update:08d}"
        )
        _, checkpoint_payload = _secure_payload(
            checkpoint_root / "checkpoint.pt",
            label=f"validated restart U{transaction.update} checkpoint",
            base=context.stage_root,
        )
        _, manifest_payload = _secure_payload(
            checkpoint_root / "manifest.json",
            label=f"validated restart U{transaction.update} manifest",
            base=context.stage_root,
        )
        _, complete_payload = _secure_payload(
            checkpoint_root / "complete.json",
            label=f"validated restart U{transaction.update} complete marker",
            base=context.stage_root,
        )
        try:
            warm_context = validate_planning_warm_start_artifact(
                warm_artifact,
                expected_child_run_id=context.formal_run_id,
                expected_child_config_bytes=config_payload,
            )
            current_checkpoint_lineage = _stage6_checkpoint_lineage_base(
                config=effective.base_config,
                bindings=context.current_immutable_bindings,
                safety_contract=safety_contract,
            )
            parent_current_checkpoint_lineage = (
                _stage6_checkpoint_lineage_base(
                    config=effective.base_config,
                    bindings=context.parent_current_immutable_bindings,
                    safety_contract=safety_contract,
                )
                if context.has_continuation
                else current_checkpoint_lineage
            )
            expected_lineage = (
                _stage6_planning_checkpoint_lineage_for_update(
                    update=transaction.update,
                    current_checkpoint_lineage=current_checkpoint_lineage,
                    historical_checkpoint_lineage=(
                        context.expected_historical_checkpoint_lineage(
                            LAST_ORIGIN_UPDATE
                        )
                    ),
                    planning_warm_start_context=warm_context,
                    planning_warm_start_sha256=warm_sha256,
                    planning_child_source_repair=context,
                    parent_current_checkpoint_lineage=(
                        parent_current_checkpoint_lineage
                    ),
                )
            )
            inspected = inspect_complete_checkpoint_snapshot(
                directory=checkpoint_root,
                update_step=transaction.update,
                schema_version=effective.base_config.checkpoint.schema_version,
                member_names=tuple(
                    sorted(
                        (
                            "checkpoint.pt",
                            "manifest.json",
                            "complete.json",
                        )
                    )
                ),
                checkpoint_bytes=checkpoint_payload,
                manifest_bytes=manifest_payload,
                complete_bytes=complete_payload,
                expected_config_sha256=expected_config_sha256,
                expected_lineage=expected_lineage,
                expected_safety_contract=safety_contract,
            )
        except (
            CheckpointError,
            Stage6PlanningWarmStartError,
            Stage6WorkflowError,
            TypeError,
            ValueError,
        ) as exc:
            raise Stage6PlanningChildSourceRepairError(
                f"validated restart U{transaction.update} checkpoint snapshot drifted"
            ) from exc
        if (
            inspected.update_step != transaction.update
            or inspected.checkpoint_sha256
            != receipt["checkpoint_sha256"]
            or inspected.manifest_sha256 != _sha256(manifest_payload)
            or inspected.policy_state_sha256
            != receipt["policy_state_sha256"]
            or _sha256(complete_payload)
            != receipt["complete_marker_sha256"]
        ):
            raise Stage6PlanningChildSourceRepairError(
                f"validated restart U{transaction.update} checkpoint binding drifted"
            )

    last_transaction = transactions[len(completed) - 1]
    next_transaction = (
        transactions[len(completed)]
        if len(completed) < len(transactions)
        else None
    )
    last_segment = _require_int(
        resource_validation.get("active_segment_index"),
        "validated restart active resource segment",
    )
    evidence = {
        "last_complete_update": last_transaction.update,
        "next_update": (
            next_transaction.update
            if next_transaction is not None
            else FINAL_CHILD_UPDATE + 1
        ),
        "next_attempt": 1,
        "next_transaction_key": (
            next_transaction.key
            if next_transaction is not None
            else _transaction_key(
                len(transactions),
                context.seed,
                FINAL_CHILD_UPDATE + 1,
            )
        ),
        "last_resource_segment_index": last_segment,
        "next_resource_segment_index": last_segment + 1,
    }
    startup_prefixes = {
        relative_path: MappingProxyType(
            {
                "size_bytes": len(payload),
                "sha256": _sha256(payload),
                "line_count": len(
                    _jsonl_rows(
                        payload,
                        f"validated restart startup {relative_path}",
                    )
                ),
            }
        )
        for relative_path, payload in payloads.items()
    }
    return evidence, startup_prefixes


@dataclass(frozen=True, slots=True)
class PlanningChildSourceRepairContext:
    artifact: Mapping[str, object]
    _canonical_artifact_payload: bytes
    artifact_path: Path | None
    artifact_sha256: str | None
    formal_run_id: str
    seed: int
    origin_lineage_sha256: str
    current_execution_identity_sha256: str
    planner_runtime_identity: Mapping[str, object]
    last_origin_update: int
    first_repaired_update: int
    next_attempt: int
    next_resource_segment_index: int
    stage_root: Path
    validated_restart_tail: Mapping[str, object] | None = None
    validated_restart_prefixes: (
        Mapping[str, Mapping[str, object]] | None
    ) = None
    continuation_artifact: Mapping[str, object] | None = None
    _canonical_continuation_payload: bytes | None = None
    continuation_artifact_path: Path | None = None
    continuation_artifact_sha256: str | None = None

    @property
    def origin_immutable_bindings(self) -> Mapping[str, object]:
        origin = _require_mapping(self.artifact.get("origin"), "origin")
        return MappingProxyType(
            dict(
                _require_mapping(
                    origin.get("immutable_bindings"),
                    "origin immutable bindings",
                )
            )
        )

    @property
    def origin_verified_review_authorization(self) -> Mapping[str, object]:
        origin = _require_mapping(self.artifact.get("origin"), "origin")
        return MappingProxyType(
            dict(
                _require_mapping(
                    origin.get("verified_review_authorization"),
                    "origin verified review authorization",
                )
            )
        )

    @property
    def current_immutable_bindings(self) -> Mapping[str, object]:
        source = (
            self.continuation_artifact
            if self.continuation_artifact is not None
            else self.artifact
        )
        current = _require_mapping(source.get("current"), "current")
        return MappingProxyType(
            dict(
                _require_mapping(
                    current.get("immutable_bindings"),
                    "current immutable bindings",
                )
            )
        )

    @property
    def parent_current_immutable_bindings(self) -> Mapping[str, object]:
        current = _require_mapping(
            self.artifact.get("current"),
            "parent current",
        )
        return MappingProxyType(
            dict(
                _require_mapping(
                    current.get("immutable_bindings"),
                    "parent current immutable bindings",
                )
            )
        )

    @property
    def parent_current_execution_identity_sha256(self) -> str:
        current = _require_mapping(
            self.artifact.get("current"),
            "parent current",
        )
        return _require_sha256(
            current.get("execution_identity_sha256"),
            "parent current execution identity SHA",
        )

    @property
    def has_continuation(self) -> bool:
        return self.continuation_artifact is not None

    @property
    def next_transaction_key(self) -> str:
        resume = _require_mapping(
            self.artifact.get("resume_point"),
            "resume point",
        )
        return _require_string(
            resume.get("next_transaction_key"),
            "resume next transaction key",
        )

    @property
    def effective_next_update(self) -> int:
        if self.validated_restart_tail is None:
            return self.first_repaired_update
        return _require_int(
            self.validated_restart_tail.get("next_update"),
            "validated restart next update",
        )

    @property
    def effective_next_attempt(self) -> int:
        if self.validated_restart_tail is None:
            return self.next_attempt
        return _require_int(
            self.validated_restart_tail.get("next_attempt"),
            "validated restart next attempt",
        )

    @property
    def effective_next_transaction_key(self) -> str:
        if self.validated_restart_tail is None:
            return self.next_transaction_key
        return _require_string(
            self.validated_restart_tail.get("next_transaction_key"),
            "validated restart next transaction key",
        )

    @property
    def effective_next_resource_segment_index(self) -> int:
        if self.validated_restart_tail is None:
            return self.next_resource_segment_index
        return _require_int(
            self.validated_restart_tail.get(
                "next_resource_segment_index"
            ),
            "validated restart next resource segment index",
        )

    @property
    def protected_checkpoint_updates(self) -> tuple[int, ...]:
        return (self.last_origin_update,)

    @property
    def acceptance_binding(self) -> dict[str, object]:
        origin = _require_mapping(self.artifact.get("origin"), "origin")
        warm = _require_mapping(
            origin.get("planning_warm_start"),
            "origin planning warm-start",
        )
        warm_artifact = _require_mapping(
            warm.get("artifact"),
            "origin planning warm-start artifact",
        )
        parent = _require_mapping(
            warm_artifact.get("parent"),
            "origin planning warm-start parent",
        )
        child = _require_mapping(
            warm_artifact.get("child"),
            "origin planning warm-start child",
        )
        artifact_sha256 = (
            self.artifact_sha256
            if self.artifact_sha256 is not None
            else _require_sha256(
                self.artifact.get("canonical_sha256"),
                "planning child source-repair canonical SHA",
            )
        )
        result: dict[str, object] = {
            "schema_version": (
                PLANNING_CHILD_SOURCE_REPAIR_ACCEPTANCE_SCHEMA
            ),
            "artifact_sha256": artifact_sha256,
            "formal_run_id": self.formal_run_id,
            "seed": self.seed,
            "origin_lineage_sha256": self.origin_lineage_sha256,
            "last_origin_update": self.last_origin_update,
            "first_repaired_update": self.first_repaired_update,
            "resume_attempt": self.next_attempt,
            "resume_resource_segment_index": (
                self.next_resource_segment_index
            ),
            "current_execution_identity_sha256": (
                self.parent_current_execution_identity_sha256
            ),
            "planner_runtime_source_set_sha256": _require_sha256(
                self.planner_runtime_identity.get("source_set_sha256"),
                "legacy path_planner source-set SHA",
            ),
            "planning_warm_start": {
                "schema_version": _require_string(
                    warm_artifact.get("schema_version"),
                    "origin planning warm-start schema",
                ),
                "artifact_sha256": _require_sha256(
                    warm.get("sha256"),
                    "origin planning warm-start SHA",
                ),
                "parent_run_id": _require_string(
                    parent.get("run_id"),
                    "origin planning warm-start parent run id",
                ),
                "parent_update": _require_int(
                    parent.get("update"),
                    "origin planning warm-start parent update",
                ),
                "child_run_id": _require_string(
                    child.get("run_id"),
                    "origin planning warm-start child run id",
                ),
                "child_first_update": _require_int(
                    child.get("first_update"),
                    "origin planning warm-start child first update",
                ),
                "child_final_update": _require_int(
                    child.get("final_update"),
                    "origin planning warm-start child final update",
                ),
            },
        }
        if self.continuation_artifact is not None:
            continuation = _require_mapping(
                self.continuation_artifact,
                "planning child continuation",
            )
            accepted_tail = _require_mapping(
                continuation.get("accepted_tail"),
                "planning child continuation accepted tail",
            )
            restart = _require_mapping(
                accepted_tail.get("restart_boundary"),
                "planning child continuation restart boundary",
            )
            current = _require_mapping(
                continuation.get("current"),
                "planning child continuation current",
            )
            if self.continuation_artifact_sha256 is None:
                continuation_sha256 = _require_sha256(
                    continuation.get("canonical_sha256"),
                    "planning child continuation canonical SHA",
                )
            else:
                continuation_sha256 = self.continuation_artifact_sha256
            result["continuation"] = {
                "schema_version": (
                    "stage6_planning_child_source_repair_"
                    "continuation_acceptance/v1"
                ),
                "artifact_sha256": continuation_sha256,
                "parent_artifact_sha256": artifact_sha256,
                "accepted_last_update": _require_int(
                    restart.get("last_complete_update"),
                    "continuation accepted last update",
                ),
                "first_continuation_update": _require_int(
                    restart.get("next_update"),
                    "continuation first update",
                ),
                "resume_attempt": _require_int(
                    restart.get("next_attempt"),
                    "continuation resume attempt",
                ),
                "resume_transaction_key": _require_string(
                    restart.get("next_transaction_key"),
                    "continuation resume transaction key",
                ),
                "resume_resource_segment_index": _require_int(
                    restart.get("next_resource_segment_index"),
                    "continuation resume resource segment index",
                ),
                "protected_checkpoint_updates": list(
                    self.protected_checkpoint_updates
                ),
                "current_execution_identity_sha256": _require_sha256(
                    current.get("execution_identity_sha256"),
                    "continuation current execution identity SHA",
                ),
            }
        return result

    def verify_origin_lineage(self, payload: bytes) -> None:
        origin = _require_mapping(self.artifact.get("origin"), "origin")
        lineage = _require_mapping(origin.get("lineage"), "origin lineage")
        if (
            len(payload) != lineage.get("size_bytes")
            or _sha256(payload) != self.origin_lineage_sha256
        ):
            raise Stage6PlanningChildSourceRepairError(
                "origin lineage bytes changed"
            )

    def verify_current_identity(
        self,
        *,
        execution_identity: Mapping[str, object],
        immutable_bindings: Mapping[str, object],
        verified_review_authorization: Mapping[str, object],
    ) -> None:
        source = (
            self.continuation_artifact
            if self.continuation_artifact is not None
            else self.artifact
        )
        current = _require_mapping(source.get("current"), "current")
        expected_identity_sha256 = _require_sha256(
            current.get("execution_identity_sha256"),
            "current execution identity SHA",
        )
        if (
            current.get("execution_identity") != execution_identity
            or current.get("immutable_bindings") != immutable_bindings
            or current.get("verified_review_authorization")
            != verified_review_authorization
            or _canonical_sha256(execution_identity)
            != expected_identity_sha256
        ):
            raise Stage6PlanningChildSourceRepairError(
                "current planning child source-repair identity drifted"
            )

    def require_current(self, label: str) -> None:
        if not isinstance(label, str) or not label.strip():
            raise Stage6PlanningChildSourceRepairError(
                "source-repair currentness label is invalid"
            )
        try:
            current_artifact_payload = ArtifactStore.canonical_json_bytes(
                dict(self.artifact)
            )
        except (TypeError, ValueError) as exc:
            raise Stage6PlanningChildSourceRepairError(
                f"planning child source-repair in-memory artifact drifted "
                f"at {label}"
            ) from exc
        if current_artifact_payload != self._canonical_artifact_payload:
            raise Stage6PlanningChildSourceRepairError(
                f"planning child source-repair in-memory artifact drifted "
                f"at {label}"
            )
        if self.artifact_path is not None:
            _, payload = _secure_payload(
                self.artifact_path,
                label=f"{label} amendment",
            )
            if (
                self.artifact_sha256 is None
                or _sha256(payload) != self.artifact_sha256
                or payload != self._canonical_artifact_payload
            ):
                raise Stage6PlanningChildSourceRepairError(
                    f"planning child source-repair artifact changed at {label}"
                )
        require_legacy_path_planner_runtime_identity_current(
            self.planner_runtime_identity,
            label=label,
        )
        immutable = _require_mapping(
            self.artifact.get("immutable_inputs"),
            "immutable inputs",
        )
        for input_label, value in immutable.items():
            binding = _require_mapping(
                value,
                f"immutable input {input_label}",
            )
            path = _require_string(
                binding.get("path"),
                f"immutable input {input_label} path",
            )
            _, payload = _secure_payload(
                path,
                label=f"{label} immutable input {input_label}",
            )
            if (
                len(payload) != binding.get("size_bytes")
                or _sha256(payload) != binding.get("sha256")
            ):
                raise Stage6PlanningChildSourceRepairError(
                    f"immutable input {input_label} changed at {label}"
                )
        if self.continuation_artifact is not None:
            if (
                self._canonical_continuation_payload is None
                or self.continuation_artifact_path is None
                or self.continuation_artifact_sha256 is None
            ):
                raise Stage6PlanningChildSourceRepairError(
                    "planning child continuation capability is incomplete"
                )
            try:
                in_memory_continuation = (
                    ArtifactStore.canonical_json_bytes(
                        dict(self.continuation_artifact)
                    )
                )
            except (TypeError, ValueError) as exc:
                raise Stage6PlanningChildSourceRepairError(
                    f"planning child continuation in-memory artifact "
                    f"drifted at {label}"
                ) from exc
            if (
                in_memory_continuation
                != self._canonical_continuation_payload
            ):
                raise Stage6PlanningChildSourceRepairError(
                    f"planning child continuation in-memory artifact "
                    f"drifted at {label}"
                )
            _, continuation_payload = _secure_payload(
                self.continuation_artifact_path,
                label=f"{label} continuation",
                base=self.stage_root,
            )
            if (
                continuation_payload
                != self._canonical_continuation_payload
                or _sha256(continuation_payload)
                != self.continuation_artifact_sha256
            ):
                raise Stage6PlanningChildSourceRepairError(
                    f"planning child continuation changed at {label}"
                )
            continuation_immutable = _require_mapping(
                self.continuation_artifact.get("immutable_inputs"),
                "planning child continuation immutable inputs",
            )
            for input_label, value in continuation_immutable.items():
                binding = _require_mapping(
                    value,
                    f"continuation immutable input {input_label}",
                )
                path = _require_string(
                    binding.get("path"),
                    f"continuation immutable input {input_label} path",
                )
                _, payload = _secure_payload(
                    path,
                    label=(
                        f"{label} continuation immutable input "
                        f"{input_label}"
                    ),
                )
                if (
                    len(payload) != binding.get("size_bytes")
                    or _sha256(payload) != binding.get("sha256")
                ):
                    raise Stage6PlanningChildSourceRepairError(
                        f"continuation immutable input {input_label} "
                        f"changed at {label}"
                    )

    def verify_append_only_prefixes(
        self,
        label: str,
        *,
        require_exact: bool = False,
    ) -> Mapping[str, int]:
        if not isinstance(label, str) or not label.strip():
            raise Stage6PlanningChildSourceRepairError(
                "append-only prefix verification label is invalid"
            )
        prefixes = _require_mapping(
            self.artifact.get("append_only_prefixes"),
            "append-only prefixes",
        )
        verified_line_counts: dict[str, int] = {}
        for prefix_label, value in prefixes.items():
            binding = _require_mapping(
                value,
                f"append-only prefix {prefix_label}",
            )
            relative = _require_string(
                binding.get("path"),
                f"append-only prefix {prefix_label} path",
            )
            _, payload = _secure_payload(
                self.stage_root / relative,
                label=f"{label} append-only prefix {prefix_label}",
                base=self.stage_root,
            )
            verified_line_counts[str(prefix_label)] = (
                self.verify_append_only_prefix_payload(
                    str(prefix_label),
                    payload,
                    label=label,
                    require_exact=(
                        require_exact
                        and self.validated_restart_tail is None
                    ),
                )
            )
        if self.validated_restart_tail is not None:
            if self.validated_restart_prefixes is None:
                raise Stage6PlanningChildSourceRepairError(
                    "validated restart prefix capability is missing"
                )
            for prefix_label, value in (
                self.validated_restart_prefixes.items()
            ):
                binding = _require_mapping(
                    value,
                    f"validated restart prefix {prefix_label}",
                )
                relative = _require_string(
                    _require_mapping(
                        prefixes.get(prefix_label),
                        f"append-only prefix {prefix_label}",
                    ).get("path"),
                    f"append-only prefix {prefix_label} path",
                )
                _, payload = _secure_payload(
                    self.stage_root / relative,
                    label=(
                        f"{label} validated restart prefix "
                        f"{prefix_label}"
                    ),
                    base=self.stage_root,
                )
                size_bytes = _require_int(
                    binding.get("size_bytes"),
                    f"validated restart prefix {prefix_label} size",
                )
                if len(payload) < size_bytes:
                    raise Stage6PlanningChildSourceRepairError(
                        f"validated restart prefix {prefix_label} "
                        f"was truncated at {label}"
                    )
                startup_payload = payload[:size_bytes]
                if _sha256(startup_payload) != binding.get("sha256"):
                    raise Stage6PlanningChildSourceRepairError(
                        f"validated restart prefix {prefix_label} "
                        f"changed at {label}"
                    )
                line_count = len(
                    _jsonl_rows(
                        startup_payload,
                        f"validated restart prefix {prefix_label}",
                    )
                )
                expected_line_count = _require_int(
                    binding.get("line_count"),
                    (
                        f"validated restart prefix {prefix_label} "
                        "line count"
                    ),
                )
                if line_count != expected_line_count:
                    raise Stage6PlanningChildSourceRepairError(
                        f"validated restart prefix {prefix_label} "
                        f"line count drifted at {label}"
                    )
        return MappingProxyType(verified_line_counts)

    def validate_restart_tail(
        self,
    ) -> PlanningChildSourceRepairContext:
        """Return a capability-bearing context for one fully accepted tail."""

        if self.validated_restart_tail is not None:
            raise Stage6PlanningChildSourceRepairError(
                "validated restart tail was already attached"
            )
        evidence, startup_prefixes = _validated_restart_tail_evidence(self)
        return replace(
            self,
            validated_restart_tail=MappingProxyType(evidence),
            validated_restart_prefixes=MappingProxyType(
                startup_prefixes
            ),
        )

    def verify_append_only_prefix_payload(
        self,
        prefix_label: str,
        payload: bytes,
        *,
        label: str,
        require_exact: bool = False,
    ) -> int:
        if (
            not isinstance(prefix_label, str)
            or not prefix_label
            or type(payload) is not bytes
            or not isinstance(label, str)
            or not label.strip()
            or type(require_exact) is not bool
        ):
            raise Stage6PlanningChildSourceRepairError(
                "append-only prefix snapshot verification input is invalid"
            )
        prefixes = _require_mapping(
            self.artifact.get("append_only_prefixes"),
            "append-only prefixes",
        )
        binding = _require_mapping(
            prefixes.get(prefix_label),
            f"append-only prefix {prefix_label}",
        )
        size_bytes = _require_int(
            binding.get("prefix_size_bytes"),
            f"append-only prefix {prefix_label} size",
        )
        if len(payload) < size_bytes:
            raise Stage6PlanningChildSourceRepairError(
                f"append-only prefix {prefix_label} was truncated at {label}"
            )
        if require_exact and len(payload) != size_bytes:
            raise Stage6PlanningChildSourceRepairError(
                f"append-only prefix {prefix_label} gained a tail at {label}"
            )
        prefix_payload = payload[:size_bytes]
        if _sha256(prefix_payload) != binding.get("prefix_sha256"):
            raise Stage6PlanningChildSourceRepairError(
                f"append-only prefix {prefix_label} changed at {label}"
            )
        line_count = len(
            _jsonl_rows(
                prefix_payload,
                f"append-only prefix {prefix_label}",
            )
        )
        expected_line_count = _require_int(
            binding.get("line_count"),
            f"append-only prefix {prefix_label} line count",
        )
        if line_count != expected_line_count:
            raise Stage6PlanningChildSourceRepairError(
                f"append-only prefix {prefix_label} line count drifted "
                f"at {label}"
            )
        return line_count

    def checkpoint_lineage_for_update(
        self,
        update: int,
    ) -> dict[str, object]:
        if (
            type(update) is not int
            or not self.first_repaired_update <= update <= FINAL_CHILD_UPDATE
        ):
            raise Stage6PlanningChildSourceRepairError(
                "repaired checkpoint update is outside U85..U100"
            )
        if self.artifact_sha256 is None:
            artifact_sha256 = _require_sha256(
                self.artifact.get("canonical_sha256"),
                "planning child source-repair canonical SHA",
            )
        else:
            artifact_sha256 = self.artifact_sha256
        parent_lineage = {
            "schema_version": PLANNING_CHILD_SOURCE_REPAIR_SCHEMA,
            "artifact_sha256": artifact_sha256,
            "origin_lineage_sha256": self.origin_lineage_sha256,
            "last_origin_update": self.last_origin_update,
            "first_repaired_update": self.first_repaired_update,
            "current_execution_identity_sha256": (
                self.parent_current_execution_identity_sha256
            ),
        }
        if (
            self.continuation_artifact is None
            or update == self.first_repaired_update
        ):
            return parent_lineage
        if update < self.first_repaired_update + 1:
            raise Stage6PlanningChildSourceRepairError(
                "continuation checkpoint update is outside U86..U100"
            )
        if self.continuation_artifact_sha256 is None:
            continuation_sha256 = _require_sha256(
                self.continuation_artifact.get("canonical_sha256"),
                "planning child continuation canonical SHA",
            )
        else:
            continuation_sha256 = self.continuation_artifact_sha256
        current = _require_mapping(
            self.continuation_artifact.get("current"),
            "planning child continuation current",
        )
        return {
            "schema_version": (
                "stage6_planning_child_source_repair_"
                "continuation_lineage/v1"
            ),
            "artifact_sha256": continuation_sha256,
            "parent": parent_lineage,
            "first_continuation_update": self.first_repaired_update + 1,
            "current_execution_identity_sha256": _require_sha256(
                current.get("execution_identity_sha256"),
                "continuation current execution identity SHA",
            ),
        }

    def checkpoint_immutable_bindings_for_update(
        self,
        update: int,
    ) -> Mapping[str, object]:
        if (
            type(update) is not int
            or not FIRST_CHILD_UPDATE <= update <= FINAL_CHILD_UPDATE
        ):
            raise Stage6PlanningChildSourceRepairError(
                "checkpoint immutable binding update is outside U75..U100"
            )
        if update <= self.last_origin_update:
            return self.origin_immutable_bindings
        if (
            self.continuation_artifact is not None
            and update == self.first_repaired_update
        ):
            return self.parent_current_immutable_bindings
        return self.current_immutable_bindings

    @property
    def journal_binding_epochs(
        self,
    ) -> tuple[tuple[str, Mapping[str, object]], ...]:
        origin = (
            _transaction_key(0, self.seed, FIRST_CHILD_UPDATE),
            self.origin_immutable_bindings,
        )
        parent = (
            self.next_transaction_key,
            self.parent_current_immutable_bindings,
        )
        if self.continuation_artifact is None:
            return (origin, parent)
        restart = _require_mapping(
            _require_mapping(
                self.continuation_artifact.get("accepted_tail"),
                "planning child continuation accepted tail",
            ).get("restart_boundary"),
            "planning child continuation restart boundary",
        )
        continuation = (
            _require_string(
                restart.get("next_transaction_key"),
                "planning child continuation next transaction key",
            ),
            self.current_immutable_bindings,
        )
        return (origin, parent, continuation)

    def expected_historical_checkpoint_lineage(
        self,
        update: int,
    ) -> dict[str, object]:
        if update != self.last_origin_update:
            raise Stage6PlanningChildSourceRepairError(
                "historical checkpoint update must be U84"
            )
        origin = _require_mapping(self.artifact.get("origin"), "origin")
        checkpoint = _require_mapping(
            origin.get("checkpoint"),
            "origin checkpoint",
        )
        lineage = _require_mapping(
            checkpoint.get("lineage"),
            "origin checkpoint lineage",
        )
        lineage_sha256 = _require_sha256(
            checkpoint.get("lineage_sha256"),
            "origin checkpoint lineage SHA",
        )
        if _canonical_sha256(lineage) != lineage_sha256:
            raise Stage6PlanningChildSourceRepairError(
                "origin checkpoint lineage binding drifted"
            )
        return dict(lineage)


@dataclass(frozen=True, slots=True)
class ValidatedPlanningChildParent:
    """Static, bytes-bound view of the immutable parent amendment."""

    formal_run_id: str
    seed: int
    stage_root: Path
    artifact_path: Path
    artifact_sha256: str
    canonical_sha256: str
    record: Mapping[str, object]
    origin_execution_identity: Mapping[str, object]
    origin_immutable_bindings: Mapping[str, object]
    current_execution_identity_sha256: str
    current_immutable_bindings: Mapping[str, object]
    immutable_inputs: Mapping[str, Mapping[str, object]]
    journal_prefixes: Mapping[str, Mapping[str, object]]
    input_pin_requests: tuple[tuple[str, Path], ...]


def _static_file_binding(
    value: object,
    *,
    label: str,
) -> dict[str, object]:
    binding = dict(_require_mapping(value, label))
    if set(binding) != {"path", "sha256", "size_bytes"}:
        raise Stage6PlanningChildSourceRepairError(
            f"{label} binding schema drifted"
        )
    path = lexical_absolute(
        _require_string(binding.get("path"), f"{label} path")
    )
    size_bytes = _require_int(
        binding.get("size_bytes"),
        f"{label} size",
    )
    if size_bytes < 0:
        raise Stage6PlanningChildSourceRepairError(
            f"{label} size drifted"
        )
    return {
        "path": path.as_posix(),
        "sha256": _require_sha256(
            binding.get("sha256"),
            f"{label} SHA",
        ),
        "size_bytes": size_bytes,
    }


def validate_planning_child_source_repair_bytes(
    payload: bytes,
    *,
    stage_root: str | Path,
    artifact_path: str | Path,
) -> ValidatedPlanningChildParent:
    """Validate only immutable parent bytes; never inspect mutable run tails."""

    if type(payload) is not bytes or not payload:
        raise Stage6PlanningChildSourceRepairError(
            "planning child parent payload is invalid"
        )
    root = lexical_absolute(stage_root)
    path = lexical_absolute(artifact_path)
    if path != root / PLANNING_CHILD_SOURCE_REPAIR_NAME:
        raise Stage6PlanningChildSourceRepairError(
            "planning child parent path is not canonical"
        )
    record = dict(
        _json_mapping(payload, "planning child source-repair parent")
    )
    if ArtifactStore.canonical_json_bytes(record) != payload:
        raise Stage6PlanningChildSourceRepairError(
            "planning child source-repair parent is not canonical JSON"
        )
    expected_fields = {
        "schema_version",
        "mode",
        "formal_run_id",
        "seed",
        "origin",
        "current",
        "accepted_prefix",
        "resume_point",
        "immutable_inputs",
        "append_only_prefixes",
        "review_evidence",
        "created_at_utc",
        "canonical_sha256",
    }
    stored_sha256 = _require_sha256(
        record.get("canonical_sha256"),
        "planning child parent canonical SHA",
    )
    detached = dict(record)
    del detached["canonical_sha256"]
    if (
        set(record) != expected_fields
        or record.get("schema_version")
        != PLANNING_CHILD_SOURCE_REPAIR_SCHEMA
        or record.get("mode") != PLANNING_CHILD_SOURCE_REPAIR_MODE
        or _canonical_sha256(detached) != stored_sha256
    ):
        raise Stage6PlanningChildSourceRepairError(
            "planning child parent schema or canonical binding drifted"
        )
    formal_run_id = _require_string(
        record.get("formal_run_id"),
        "planning child parent formal run id",
    )
    seed = _require_int(
        record.get("seed"),
        "planning child parent seed",
    )
    if root.parent.name != formal_run_id or seed <= 0:
        raise Stage6PlanningChildSourceRepairError(
            "planning child parent run identity drifted"
        )

    origin = _require_mapping(record.get("origin"), "parent origin")
    origin_identity = dict(
        _require_mapping(
            origin.get("execution_identity"),
            "parent origin execution identity",
        )
    )
    origin_immutable = dict(
        _require_mapping(
            origin.get("immutable_bindings"),
            "parent origin immutable bindings",
        )
    )
    current = _require_mapping(record.get("current"), "parent current")
    current_identity = dict(
        _require_mapping(
            current.get("execution_identity"),
            "parent current execution identity",
        )
    )
    current_identity_sha256 = _require_sha256(
        current.get("execution_identity_sha256"),
        "parent current execution identity SHA",
    )
    current_verified = _require_mapping(
        current.get("verified_review_authorization"),
        "parent current verified authorization",
    )
    current_immutable = dict(
        _require_mapping(
            current.get("immutable_bindings"),
            "parent current immutable bindings",
        )
    )
    launch_binding = _static_file_binding(
        current.get("launch_authorization"),
        label="parent current launch authorization",
    )
    if (
        _canonical_sha256(current_identity) != current_identity_sha256
        or current_verified.get("authorized") is not True
        or current_verified.get("formal_run_id") != formal_run_id
        or current_verified.get("review_identity_sha256")
        != current_identity_sha256
        or current_verified.get("authorization_file_sha256")
        != launch_binding["sha256"]
        or current_verified.get("authorization_file_size_bytes")
        != launch_binding["size_bytes"]
        or current_immutable.get("formal_run_id") != formal_run_id
        or current_immutable.get("review_identity_sha256")
        != current_identity_sha256
        or current_immutable.get("authorization_file_sha256")
        != launch_binding["sha256"]
    ):
        raise Stage6PlanningChildSourceRepairError(
            "planning child parent current binding drifted"
        )

    immutable_raw = _require_mapping(
        record.get("immutable_inputs"),
        "parent immutable inputs",
    )
    immutable_inputs = {
        str(label): MappingProxyType(
            _static_file_binding(
                value,
                label=f"parent immutable input {label}",
            )
        )
        for label, value in immutable_raw.items()
    }
    if (
        "current_review_authorization" not in immutable_inputs
        or dict(immutable_inputs["current_review_authorization"])
        != launch_binding
    ):
        raise Stage6PlanningChildSourceRepairError(
            "planning child parent authorization input drifted"
        )

    prefixes_raw = _require_mapping(
        record.get("append_only_prefixes"),
        "parent journal prefixes",
    )
    if set(prefixes_raw) != set(_REQUIRED_PREFIX_PATHS):
        raise Stage6PlanningChildSourceRepairError(
            "planning child parent journal prefix set drifted"
        )
    prefixes: dict[str, Mapping[str, object]] = {}
    for relative in _REQUIRED_PREFIX_PATHS:
        binding = dict(
            _require_mapping(
                prefixes_raw.get(relative),
                f"parent journal prefix {relative}",
            )
        )
        if (
            set(binding)
            != {
                "path",
                "prefix_sha256",
                "prefix_size_bytes",
                "line_count",
            }
            or binding.get("path") != relative
            or _require_int(
                binding.get("prefix_size_bytes"),
                f"parent journal prefix {relative} size",
            )
            <= 0
            or _require_int(
                binding.get("line_count"),
                f"parent journal prefix {relative} line count",
            )
            <= 0
        ):
            raise Stage6PlanningChildSourceRepairError(
                f"planning child parent journal prefix {relative} drifted"
            )
        _require_sha256(
            binding.get("prefix_sha256"),
            f"parent journal prefix {relative} SHA",
        )
        prefixes[relative] = MappingProxyType(binding)

    requests: list[tuple[str, Path]] = [
        ("planning-child-recovery:parent", path)
    ]
    requested_paths = {path}
    for label, binding in sorted(immutable_inputs.items()):
        input_path = lexical_absolute(
            _require_string(
                binding.get("path"),
                f"parent immutable input {label} path",
            )
        )
        if input_path not in requested_paths:
            requests.append(
                (
                    f"planning-child-recovery:parent-input:{label}",
                    input_path,
                )
            )
            requested_paths.add(input_path)

    return ValidatedPlanningChildParent(
        formal_run_id=formal_run_id,
        seed=seed,
        stage_root=root,
        artifact_path=path,
        artifact_sha256=_sha256(payload),
        canonical_sha256=stored_sha256,
        record=MappingProxyType(record),
        origin_execution_identity=MappingProxyType(origin_identity),
        origin_immutable_bindings=MappingProxyType(origin_immutable),
        current_execution_identity_sha256=current_identity_sha256,
        current_immutable_bindings=MappingProxyType(current_immutable),
        immutable_inputs=MappingProxyType(immutable_inputs),
        journal_prefixes=MappingProxyType(prefixes),
        input_pin_requests=tuple(requests),
    )


def build_planning_child_source_repair_artifact(
    *,
    stage_root: str | Path,
    formal_run_id: str,
    seed: int,
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
    current_review_authorization_path: str | Path,
    planning_warm_start_path: str | Path,
    spec_review_path: str | Path,
    quality_review_path: str | Path,
    implementation_report_path: str | Path,
    exact_replay_evidence_path: str | Path,
    created_at_utc: str,
) -> dict[str, object]:
    """Build, but do not publish, the reviewed same-child repair artifact."""

    root = lexical_absolute(stage_root)
    if type(seed) is not int or seed <= 0:
        raise Stage6PlanningChildSourceRepairError("seed is invalid")
    _require_string(formal_run_id, "formal run id")
    _require_string(created_at_utc, "created_at_utc")

    config_path, config_payload = _secure_payload(
        root / "config.json",
        label="child config",
        base=root,
    )
    lineage_path, lineage_payload = _secure_payload(
        root / "lineage_audit.json",
        label="origin lineage",
        base=root,
    )
    warm_path, warm_payload = _secure_payload(
        planning_warm_start_path,
        label="planning warm-start",
    )
    origin, _ = _validate_origin(
        stage_root=root,
        formal_run_id=formal_run_id,
        seed=seed,
        planning_warm_start_path=warm_path,
        planning_warm_start_payload=warm_payload,
        lineage_path=lineage_path,
        lineage_payload=lineage_payload,
        config_payload=config_payload,
    )

    authorization_path, authorization_payload = _secure_payload(
        current_review_authorization_path,
        label="current review authorization",
    )
    current_identity_sha256, _ = _validate_current_authorization(
        formal_run_id=formal_run_id,
        execution_identity=current_execution_identity,
        verified_authorization=current_verified_review_authorization,
        immutable_bindings=current_immutable_bindings,
        authorization_path=authorization_path,
        authorization_payload=authorization_payload,
    )
    planner_runtime_identity = (
        resolve_legacy_path_planner_runtime_identity()
    )
    require_legacy_path_planner_runtime_identity_current(
        planner_runtime_identity,
        label="planning child source-repair build",
    )

    spec_path, spec_payload = _secure_payload(
        spec_review_path,
        label="spec review",
    )
    quality_path, quality_payload = _secure_payload(
        quality_review_path,
        label="quality review",
    )
    report_path, report_payload = _secure_payload(
        implementation_report_path,
        label="implementation report",
    )
    replay_path, replay_payload = _secure_payload(
        exact_replay_evidence_path,
        label="exact replay evidence",
    )
    spec_review = _validate_review(
        spec_path,
        spec_payload,
        review_name="spec",
    )
    quality_review = _validate_review(
        quality_path,
        quality_payload,
        review_name="quality",
    )

    accepted_prefix, resume_point, prefix_bindings = (
        _validate_accepted_prefix(stage_root=root, seed=seed)
    )
    checkpoint, checkpoint_bindings = _validate_u84_checkpoint(
        stage_root=root,
        seed=seed,
        accepted_prefix=accepted_prefix,
        config_payload=config_payload,
    )
    origin["checkpoint"] = checkpoint

    immutable_inputs = {
        "config": _file_binding(config_path, config_payload),
        "origin_lineage": _file_binding(lineage_path, lineage_payload),
        "planning_warm_start": _file_binding(warm_path, warm_payload),
        "current_review_authorization": _file_binding(
            authorization_path,
            authorization_payload,
        ),
        "spec_review": _file_binding(spec_path, spec_payload),
        "quality_review": _file_binding(quality_path, quality_payload),
        "implementation_report": _file_binding(
            report_path,
            report_payload,
        ),
        "exact_replay_evidence": _file_binding(
            replay_path,
            replay_payload,
        ),
        **checkpoint_bindings,
        **_planner_runtime_immutable_bindings(
            planner_runtime_identity,
            label="planning child source-repair build",
        ),
    }
    artifact: dict[str, object] = {
        "schema_version": PLANNING_CHILD_SOURCE_REPAIR_SCHEMA,
        "mode": PLANNING_CHILD_SOURCE_REPAIR_MODE,
        "formal_run_id": formal_run_id,
        "seed": seed,
        "origin": origin,
        "current": {
            "execution_identity": dict(current_execution_identity),
            "execution_identity_sha256": current_identity_sha256,
            "planner_runtime_identity": planner_runtime_identity,
            "verified_review_authorization": dict(
                current_verified_review_authorization
            ),
            "immutable_bindings": dict(current_immutable_bindings),
            "launch_authorization": _file_binding(
                authorization_path,
                authorization_payload,
            ),
        },
        "accepted_prefix": accepted_prefix,
        "resume_point": resume_point,
        "immutable_inputs": immutable_inputs,
        "append_only_prefixes": prefix_bindings,
        "review_evidence": {
            "implementation_report": _file_binding(
                report_path,
                report_payload,
            ),
            "spec_review": spec_review,
            "quality_review": quality_review,
            "exact_replay_evidence": _file_binding(
                replay_path,
                replay_payload,
            ),
        },
        "created_at_utc": created_at_utc,
    }
    artifact["canonical_sha256"] = _canonical_sha256(artifact)
    return artifact


def validate_planning_child_source_repair_artifact(
    artifact: Mapping[str, object],
    *,
    stage_root: str | Path,
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
    require_exact_prefix: bool,
    artifact_path: str | Path | None = None,
    artifact_sha256: str | None = None,
) -> PlanningChildSourceRepairContext:
    """Validate a built or loaded amendment against live immutable inputs."""

    artifact_input = _require_mapping(
        artifact,
        "planning child source repair",
    )
    try:
        detached_payload = ArtifactStore.canonical_json_bytes(
            dict(artifact_input)
        )
        record = dict(
            _json_mapping(
                detached_payload,
                "planning child source repair",
            )
        )
    except (TypeError, ValueError) as exc:
        raise Stage6PlanningChildSourceRepairError(
            "planning child source-repair artifact is not canonical JSON"
        ) from exc
    stored_canonical_sha256 = _require_sha256(
        record.pop("canonical_sha256", None),
        "planning child source-repair canonical SHA",
    )
    if _canonical_sha256(record) != stored_canonical_sha256:
        raise Stage6PlanningChildSourceRepairError(
            "planning child source-repair canonical SHA drifted"
        )
    record["canonical_sha256"] = stored_canonical_sha256
    canonical_artifact_payload = ArtifactStore.canonical_json_bytes(record)
    if (
        record.get("schema_version")
        != PLANNING_CHILD_SOURCE_REPAIR_SCHEMA
        or record.get("mode") != PLANNING_CHILD_SOURCE_REPAIR_MODE
    ):
        raise Stage6PlanningChildSourceRepairError(
            "planning child source-repair schema or mode drifted"
        )
    formal_run_id = _require_string(
        record.get("formal_run_id"),
        "formal run id",
    )
    seed = _require_int(record.get("seed"), "seed")
    current = _require_mapping(record.get("current"), "current")
    planner_runtime_identity = (
        _validate_planner_runtime_identity_record(
            current.get("planner_runtime_identity")
        )
    )
    require_legacy_path_planner_runtime_identity_current(
        planner_runtime_identity,
        label="planning child source-repair validation",
    )
    current_identity_sha256 = _canonical_sha256(
        current_execution_identity
    )
    if (
        current.get("execution_identity") != current_execution_identity
        or current.get("execution_identity_sha256")
        != current_identity_sha256
        or current.get("verified_review_authorization")
        != current_verified_review_authorization
        or current.get("immutable_bindings") != current_immutable_bindings
    ):
        raise Stage6PlanningChildSourceRepairError(
            "current execution identity or authorization drifted"
        )
    origin = _require_mapping(record.get("origin"), "origin")
    lineage = _require_mapping(origin.get("lineage"), "origin lineage")
    accepted = _require_mapping(
        record.get("accepted_prefix"),
        "accepted prefix",
    )
    resume = _require_mapping(record.get("resume_point"), "resume point")
    context = PlanningChildSourceRepairContext(
        artifact=MappingProxyType(record),
        _canonical_artifact_payload=canonical_artifact_payload,
        artifact_path=(
            lexical_absolute(artifact_path)
            if artifact_path is not None
            else None
        ),
        artifact_sha256=(
            _require_sha256(artifact_sha256, "artifact file SHA")
            if artifact_sha256 is not None
            else None
        ),
        formal_run_id=formal_run_id,
        seed=seed,
        origin_lineage_sha256=_require_sha256(
            lineage.get("sha256"),
            "origin lineage SHA",
        ),
        current_execution_identity_sha256=current_identity_sha256,
        planner_runtime_identity=MappingProxyType(
            planner_runtime_identity
        ),
        last_origin_update=_require_int(
            accepted.get("last_update"),
            "accepted prefix last update",
        ),
        first_repaired_update=_require_int(
            resume.get("next_update"),
            "resume next update",
        ),
        next_attempt=_require_int(
            resume.get("next_attempt"),
            "resume next attempt",
        ),
        next_resource_segment_index=_require_int(
            resume.get("next_resource_segment_index"),
            "resume next resource segment index",
        ),
        stage_root=lexical_absolute(stage_root),
    )
    context.require_current("planning child source-repair validation")
    context.verify_append_only_prefixes(
        "planning child source-repair validation",
        require_exact=require_exact_prefix,
    )
    origin_lineage_path = _require_string(
        lineage.get("path"),
        "origin lineage path",
    )
    _, origin_lineage_payload = _secure_payload(
        origin_lineage_path,
        label="origin lineage validation",
        base=context.stage_root,
    )
    context.verify_origin_lineage(origin_lineage_payload)
    return context


def load_planning_child_source_repair_artifact(
    path: str | Path,
    *,
    stage_root: str | Path,
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
    require_exact_prefix: bool,
) -> tuple[PlanningChildSourceRepairContext, str]:
    """Securely load the canonical amendment and bind it to live inputs."""

    root = lexical_absolute(stage_root)
    artifact_path = lexical_absolute(path)
    expected_path = root / PLANNING_CHILD_SOURCE_REPAIR_NAME
    if artifact_path != expected_path:
        raise Stage6PlanningChildSourceRepairError(
            "planning child source-repair path is not canonical"
        )
    artifact_path, payload = _secure_payload(
        artifact_path,
        label="planning child source-repair artifact",
        base=root,
    )
    artifact = _json_mapping(
        payload,
        "planning child source-repair artifact",
    )
    if ArtifactStore.canonical_json_bytes(dict(artifact)) != payload:
        raise Stage6PlanningChildSourceRepairError(
            "planning child source-repair file is not canonical JSON"
        )
    artifact_sha256 = _sha256(payload)
    context = validate_planning_child_source_repair_artifact(
        artifact,
        stage_root=root,
        current_execution_identity=current_execution_identity,
        current_verified_review_authorization=(
            current_verified_review_authorization
        ),
        current_immutable_bindings=current_immutable_bindings,
        require_exact_prefix=require_exact_prefix,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
    )
    return context, artifact_sha256


def publish_planning_child_source_repair_artifact(
    artifact: Mapping[str, object],
    *,
    stage_root: str | Path,
    output_path: str | Path,
) -> Path:
    """Atomically and exclusively publish the canonical amendment once."""

    root = lexical_absolute(stage_root)
    output = lexical_absolute(output_path)
    expected = root / PLANNING_CHILD_SOURCE_REPAIR_NAME
    if output != expected:
        raise Stage6PlanningChildSourceRepairError(
            "planning child source-repair output must use the canonical stage root"
        )
    try:
        plain_root = require_plain_path(
            root,
            leaf_kind="directory",
            label="planning child source-repair stage root",
        )
        require_plain_path(
            output,
            base=plain_root,
            allow_missing=True,
            leaf_kind="file",
            label="planning child source-repair output",
        )
    except PathSecurityError as exc:
        raise Stage6PlanningChildSourceRepairError(
            "planning child source-repair output path is unsafe"
        ) from exc

    record = dict(_require_mapping(artifact, "planning child source repair"))
    stored_sha256 = _require_sha256(
        record.pop("canonical_sha256", None),
        "planning child source-repair canonical SHA",
    )
    if _canonical_sha256(record) != stored_sha256:
        raise Stage6PlanningChildSourceRepairError(
            "planning child source-repair canonical SHA drifted"
        )
    record["canonical_sha256"] = stored_sha256
    if (
        record.get("schema_version")
        != PLANNING_CHILD_SOURCE_REPAIR_SCHEMA
        or record.get("mode") != PLANNING_CHILD_SOURCE_REPAIR_MODE
    ):
        raise Stage6PlanningChildSourceRepairError(
            "planning child source-repair schema or mode drifted"
        )
    current = _require_mapping(record.get("current"), "current")
    require_legacy_path_planner_runtime_identity_current(
        _require_mapping(
            current.get("planner_runtime_identity"),
            "legacy path_planner runtime identity",
        ),
        label="planning child source-repair publication",
    )
    payload = ArtifactStore.canonical_json_bytes(record)
    published = ArtifactStore(plain_root).write_bytes_exclusive(
        PLANNING_CHILD_SOURCE_REPAIR_NAME,
        payload,
    )
    published_path, observed = _secure_payload(
        published,
        label="published planning child source-repair artifact",
        base=plain_root,
    )
    if published_path != output or observed != payload:
        raise Stage6PlanningChildSourceRepairError(
            "planning child source-repair publication verification failed"
        )
    return published_path


def planning_child_source_repair_input_pin_requests(
    path: str | Path,
) -> tuple[tuple[str, Path], ...]:
    """Return only the amendment and its fully immutable input paths."""

    artifact_path = lexical_absolute(path)
    if artifact_path.name != PLANNING_CHILD_SOURCE_REPAIR_NAME:
        raise Stage6PlanningChildSourceRepairError(
            "planning child source-repair input pin path is not canonical"
        )
    stage_root = artifact_path.parent
    artifact_path, payload = _secure_payload(
        artifact_path,
        label="planning child source-repair input pin artifact",
        base=stage_root,
    )
    artifact = _json_mapping(
        payload,
        "planning child source-repair input pin artifact",
    )
    if ArtifactStore.canonical_json_bytes(dict(artifact)) != payload:
        raise Stage6PlanningChildSourceRepairError(
            "planning child source-repair input pin artifact is not canonical"
        )
    record = dict(artifact)
    stored_sha256 = _require_sha256(
        record.pop("canonical_sha256", None),
        "planning child source-repair canonical SHA",
    )
    if _canonical_sha256(record) != stored_sha256:
        raise Stage6PlanningChildSourceRepairError(
            "planning child source-repair input pin canonical SHA drifted"
        )
    immutable = _require_mapping(
        artifact.get("immutable_inputs"),
        "planning child source-repair immutable inputs",
    )
    prefixes = _require_mapping(
        artifact.get("append_only_prefixes"),
        "planning child source-repair append-only prefixes",
    )
    mutable_paths = {
        lexical_absolute(stage_root / _require_string(
            _require_mapping(value, f"append-only prefix {label}").get(
                "path"
            ),
            f"append-only prefix {label} path",
        ))
        for label, value in prefixes.items()
    }
    current = _require_mapping(
        artifact.get("current"),
        "planning child source-repair current",
    )
    planner_runtime_identity = _validate_planner_runtime_identity_record(
        current.get("planner_runtime_identity")
    )
    require_legacy_path_planner_runtime_identity_current(
        planner_runtime_identity,
        label="planning child source-repair input pin",
    )
    requests: list[tuple[str, Path]] = [
        ("planning-child-source-repair:artifact", artifact_path)
    ]
    requested_paths = {artifact_path}
    planner_root = lexical_absolute(
        _require_string(
            planner_runtime_identity.get("package_root"),
            "legacy path_planner package root",
        )
    )
    for value in planner_runtime_identity["paths"]:
        row = _require_mapping(
            value,
            "legacy path_planner source member",
        )
        relative = _require_string(
            row.get("path"),
            "legacy path_planner source member path",
        )
        member_path = lexical_absolute(planner_root / relative)
        if member_path not in requested_paths:
            requests.append(
                (
                    f"planning-child-source-repair:planner-runtime:{relative}",
                    member_path,
                )
            )
            requested_paths.add(member_path)
    for label in sorted(immutable):
        binding = _require_mapping(
            immutable[label],
            f"immutable input {label}",
        )
        input_path = lexical_absolute(
            _require_string(
                binding.get("path"),
                f"immutable input {label} path",
            )
        )
        if input_path in mutable_paths:
            raise Stage6PlanningChildSourceRepairError(
                f"mutable journal {input_path.name} cannot be an immutable pin"
            )
        input_path, input_payload = _secure_payload(
            input_path,
            label=f"planning child source-repair immutable pin {label}",
        )
        if (
            len(input_payload) != binding.get("size_bytes")
            or _sha256(input_payload) != binding.get("sha256")
        ):
            raise Stage6PlanningChildSourceRepairError(
                f"planning child source-repair immutable pin {label} drifted"
            )
        if input_path not in requested_paths:
            requests.append(
                (
                    f"planning-child-source-repair:immutable:{label}",
                    input_path,
                )
            )
            requested_paths.add(input_path)
    return tuple(requests)


__all__ = [
    "LEGACY_PATH_PLANNER_RUNTIME_SCHEMA",
    "PLANNING_CHILD_SOURCE_REPAIR_MODE",
    "PLANNING_CHILD_SOURCE_REPAIR_NAME",
    "PLANNING_CHILD_SOURCE_REPAIR_SCHEMA",
    "PlanningChildSourceRepairContext",
    "Stage6PlanningChildSourceRepairError",
    "ValidatedPlanningChildParent",
    "build_legacy_path_planner_runtime_identity",
    "build_planning_child_source_repair_artifact",
    "load_planning_child_source_repair_artifact",
    "planning_child_source_repair_input_pin_requests",
    "publish_planning_child_source_repair_artifact",
    "require_legacy_path_planner_runtime_identity_current",
    "resolve_legacy_path_planner_runtime_identity",
    "validate_planning_child_source_repair_artifact",
    "validate_planning_child_source_repair_bytes",
]
