"""Stage 6 正式启动前的外部 review authorization 只读校验。"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Final

from lunar_exploration_ppo.utils.path_security import (
    PathSecurityError,
    lexical_absolute,
    require_plain_path,
    secure_read_bytes as _secure_read_bytes,
)
from lunar_exploration_ppo.workflows.stage6 import (
    STAGE5_COMMIT,
    STAGE6_STAGE_ID,
    Stage6WorkflowError,
    stage6_execution_identity,
    validate_stage6_formal_run_id,
)


AUTHORIZATION_SCHEMA: Final = "stage6_review_launch_authorization/v1"
VERIFIED_SCHEMA: Final = "stage6_verified_review_launch_authorization/v1"
REVIEW_SCHEMA: Final = "stage6_fresh_review/v1"
DIFF_FORMAT: Final = "git_diff_binary_full_index_tree_to_tree/v1"
SINGLE_SEED_SCOPE: Final = {
    "scope_kind": "single_seed_system_closure/v1",
    "seed": 20260716,
    "updates": 100,
}

_AUTHORIZATION_FIELDS: Final = frozenset(
    {
        "schema_version",
        "stage_id",
        "formal_run_id",
        "single_seed_scope",
        "authorized",
        "base_commit",
        "head_commit",
        "reviewed_execution_identity",
        "reviewed_execution_identity_sha256",
        "execution_identity",
        "reviewed_prospective_git_tree",
        "prospective_tree_sha256",
        "changed_path_set_sha256",
        "source_set_sha256",
        "config_sha256",
        "data_sha256",
        "environment_sha256",
        "frozen_diff",
        "spec_review",
        "quality_review",
    }
)
_SCOPE_FIELDS: Final = frozenset({"scope_kind", "seed", "updates"})
_ARTIFACT_FIELDS: Final = frozenset({"path", "sha256", "size_bytes"})
_DIFF_FIELDS: Final = frozenset(
    {"path", "sha256", "size_bytes", "format"}
)
_REVIEW_FIELDS: Final = frozenset(
    {
        "path",
        "sha256",
        "size_bytes",
        "review_kind",
        "stage_id",
        "formal_run_id",
        "reviewed_execution_identity_sha256",
        "reviewed_prospective_git_tree",
        "verdict",
        "finding_counts",
    }
)
_FINDING_FIELDS: Final = frozenset({"critical", "important", "minor"})
_REVIEW_DOCUMENT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "review_kind",
        "stage_id",
        "formal_run_id",
        "reviewed_execution_identity_sha256",
        "reviewed_prospective_git_tree",
        "verdict",
        "finding_counts",
        "findings",
    }
)
_REVIEW_FINDING_FIELDS: Final = frozenset(
    {"finding_id", "severity", "summary"}
)
_REVIEW_SEVERITIES: Final = ("critical", "important", "minor")
_EXECUTION_IDENTITY_FIELDS: Final = frozenset(
    {
        "schema_version",
        "base_commit",
        "head_commit",
        "prospective_git_tree",
        "prospective_tree_sha256",
        "changed_paths",
        "changed_path_set_sha256",
        "real_index_empty",
        "config_sha256",
        "source_set_sha256",
        "data_sha256",
        "catalog_sha256",
        "environment_identity",
        "environment_sha256",
        "source_identity",
        "data_identity",
    }
)
_REDUNDANT_IDENTITY_FIELDS: Final = (
    "prospective_tree_sha256",
    "changed_path_set_sha256",
    "source_set_sha256",
    "config_sha256",
    "data_sha256",
    "environment_sha256",
)


class Stage6ReviewAuthorizationError(RuntimeError):
    """外部审查授权不完整、不安全或与当前执行身份不一致。"""


@dataclass(frozen=True, slots=True)
class _PayloadSnapshot:
    path: Path
    stat_identity: tuple[int, int, int, int, int]
    link_count: int
    payload: bytes
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class Stage6ReviewEvidenceRecord:
    """Read-only path and byte identity for one fresh-review evidence member."""

    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _VerifiedAuthorizationCapture:
    record_payload: bytes
    members: tuple[_PayloadSnapshot, ...]


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        decoded = payload.decode("utf-8")
        value = json.loads(
            decoded,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise Stage6ReviewAuthorizationError(
            f"{label} is not canonical JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise Stage6ReviewAuthorizationError(
            f"{label} is not canonical JSON: top level must be an object"
        )
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_git_oid(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 40 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise Stage6ReviewAuthorizationError(f"{label} must be a JSON object")
    return value


def _require_exact_fields(
    value: object,
    expected: frozenset[str],
    *,
    label: str,
) -> dict[str, object]:
    mapping = _require_mapping(value, label=label)
    if frozenset(mapping) != expected:
        raise Stage6ReviewAuthorizationError(f"{label} field set drift")
    return mapping


def _review_package_drive_is_allowed(path: Path) -> bool:
    """生产策略 seam：review package 必须位于 D 盘。"""

    return path.drive.upper() == "D:"


def _review_package_root(authorization_path: Path) -> Path:
    root = authorization_path.parent
    if not _review_package_drive_is_allowed(root):
        raise Stage6ReviewAuthorizationError(
            "authorization must be located in a D drive review package"
        )
    try:
        require_plain_path(
            root,
            leaf_kind="directory",
            label="Stage 6 review package",
        )
    except PathSecurityError as exc:
        raise Stage6ReviewAuthorizationError(
            "review package contains a link/reparse point or is not plain"
        ) from exc
    return root


def _secure_payload_snapshot(
    path: Path,
    *,
    base: Path,
    label: str,
) -> _PayloadSnapshot:
    try:
        result = _secure_read_bytes(path, base=base, label=label)
    except (OSError, PathSecurityError) as exc:
        raise Stage6ReviewAuthorizationError(
            f"{label} secure single-link read failed"
        ) from exc
    if result.link_count != 1:
        raise Stage6ReviewAuthorizationError(
            f"{label} secure single-link read failed"
        )
    return _PayloadSnapshot(
        path=lexical_absolute(path),
        stat_identity=result.stat_identity,
        link_count=result.link_count,
        payload=result.payload,
        sha256=_sha256(result.payload),
        size_bytes=len(result.payload),
    )


def _secure_payload(path: Path, *, base: Path, label: str) -> bytes:
    return _secure_payload_snapshot(path, base=base, label=label).payload


def _parse_authorization(payload: bytes) -> dict[str, object]:
    value = _strict_json_object(payload, label="authorization")
    if _canonical_json_bytes(value) != payload:
        raise Stage6ReviewAuthorizationError(
            "authorization is not canonical JSON"
        )
    return _require_exact_fields(
        value,
        _AUTHORIZATION_FIELDS,
        label="authorization",
    )


def _safe_relative_reference(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or value != value.strip():
        raise Stage6ReviewAuthorizationError(
            f"{label} must be a safe relative reference"
        )
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        "\\" in value
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in posix.parts)
        or posix.as_posix() != value
    ):
        raise Stage6ReviewAuthorizationError(
            f"{label} must be a safe relative reference"
        )
    return posix


def _reference_path(
    root: Path,
    binding: Mapping[str, object],
    *,
    label: str,
) -> Path:
    relative = _safe_relative_reference(binding.get("path"), label=label)
    return lexical_absolute(root.joinpath(*relative.parts))


def _validate_artifact_binding(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, object]:
    binding = _require_exact_fields(value, fields, label=label)
    if not _is_sha256(binding["sha256"]):
        raise Stage6ReviewAuthorizationError(f"{label} SHA-256 is invalid")
    size = binding["size_bytes"]
    if type(size) is not int or size < 0:
        raise Stage6ReviewAuthorizationError(f"{label} size is invalid")
    _safe_relative_reference(binding["path"], label=label)
    return binding


def _read_bound_artifact(
    root: Path,
    binding: Mapping[str, object],
    *,
    label: str,
) -> _PayloadSnapshot:
    path = _reference_path(root, binding, label=label)
    snapshot = _secure_payload_snapshot(path, base=root, label=label)
    if snapshot.sha256 != binding["sha256"]:
        raise Stage6ReviewAuthorizationError(f"{label} SHA-256 drift")
    if snapshot.size_bytes != binding["size_bytes"]:
        raise Stage6ReviewAuthorizationError(f"{label} size drift")
    return snapshot


def _validate_review(
    value: object,
    *,
    label: str,
    expected_kind: str,
    authorization: Mapping[str, object],
) -> dict[str, object]:
    review = _validate_artifact_binding(
        value,
        fields=_REVIEW_FIELDS,
        label=label,
    )
    counts = _require_exact_fields(
        review["finding_counts"],
        _FINDING_FIELDS,
        label=f"{label} finding counts",
    )
    if any(type(counts[name]) is not int or counts[name] < 0 for name in counts):
        raise Stage6ReviewAuthorizationError(f"{label} finding counts are invalid")
    if review["review_kind"] != expected_kind:
        raise Stage6ReviewAuthorizationError(f"{label} kind drift")
    if review["stage_id"] != authorization["stage_id"]:
        raise Stage6ReviewAuthorizationError(f"{label} stage id drift")
    if review["formal_run_id"] != authorization["formal_run_id"]:
        raise Stage6ReviewAuthorizationError(f"{label} formal run id drift")
    if (
        review["reviewed_execution_identity_sha256"]
        != authorization["reviewed_execution_identity_sha256"]
    ):
        raise Stage6ReviewAuthorizationError(
            f"{label} execution identity binding drift"
        )
    if (
        review["reviewed_prospective_git_tree"]
        != authorization["reviewed_prospective_git_tree"]
    ):
        raise Stage6ReviewAuthorizationError(f"{label} reviewed tree binding drift")
    if review["verdict"] not in {"PASS", "FAIL"}:
        raise Stage6ReviewAuthorizationError(f"{label} verdict is invalid")
    return review


def _parse_review_document(payload: bytes, *, label: str) -> dict[str, object]:
    value = _strict_json_object(payload, label=label)
    if _canonical_json_bytes(value) != payload:
        raise Stage6ReviewAuthorizationError(f"{label} is not canonical JSON")
    if frozenset(value) != _REVIEW_DOCUMENT_FIELDS:
        raise Stage6ReviewAuthorizationError(f"{label} field set drift")
    if value["schema_version"] != REVIEW_SCHEMA:
        raise Stage6ReviewAuthorizationError(f"{label} schema drift")
    if value["review_kind"] not in {"spec_compliance", "quality_security"}:
        raise Stage6ReviewAuthorizationError(f"{label} kind drift")
    if value["verdict"] not in {"PASS", "FAIL"}:
        raise Stage6ReviewAuthorizationError(f"{label} verdict is invalid")
    counts = _require_exact_fields(
        value["finding_counts"],
        _FINDING_FIELDS,
        label=f"{label} finding counts",
    )
    if any(type(counts[name]) is not int or counts[name] < 0 for name in counts):
        raise Stage6ReviewAuthorizationError(f"{label} finding counts are invalid")
    findings = value["findings"]
    if not isinstance(findings, list):
        raise Stage6ReviewAuthorizationError(f"{label} findings must be a list")
    actual_counts = {name: 0 for name in _REVIEW_SEVERITIES}
    finding_ids: set[str] = set()
    for finding in findings:
        row = _require_exact_fields(
            finding,
            _REVIEW_FINDING_FIELDS,
            label=f"{label} finding",
        )
        finding_id = row["finding_id"]
        summary = row["summary"]
        severity = row["severity"]
        if (
            not isinstance(finding_id, str)
            or not finding_id
            or finding_id != finding_id.strip()
            or finding_id in finding_ids
            or not isinstance(summary, str)
            or not summary
            or summary != summary.strip()
            or not isinstance(severity, str)
            or severity not in actual_counts
        ):
            raise Stage6ReviewAuthorizationError(f"{label} finding is invalid")
        finding_ids.add(finding_id)
        actual_counts[str(severity)] += 1
    if actual_counts != counts:
        raise Stage6ReviewAuthorizationError(
            f"{label} finding counts do not match findings"
        )
    return value


def _validate_review_document_binding(
    payload: bytes,
    *,
    binding: Mapping[str, object],
    label: str,
) -> dict[str, object]:
    document = _parse_review_document(payload, label=label)
    semantic_fields = (
        "review_kind",
        "stage_id",
        "formal_run_id",
        "reviewed_execution_identity_sha256",
        "reviewed_prospective_git_tree",
        "verdict",
        "finding_counts",
    )
    for field in semantic_fields:
        if document[field] != binding[field]:
            if field == "verdict":
                raise Stage6ReviewAuthorizationError(
                    f"{label} verdict content/binding drift"
                )
            if field == "finding_counts":
                raise Stage6ReviewAuthorizationError(
                    f"{label} finding counts content/binding drift"
                )
            raise Stage6ReviewAuthorizationError(
                f"{label} content/binding drift for {field}"
            )
    counts = document["finding_counts"]
    if (
        document["verdict"] != "PASS"
        or counts["critical"] != 0
        or counts["important"] != 0
    ):
        raise Stage6ReviewAuthorizationError(
            "review must be PASS with zero Critical and Important findings"
        )
    return document


def _validate_authorization_semantics(
    authorization: dict[str, object],
    *,
    expected_run_id: str,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    if authorization["schema_version"] != AUTHORIZATION_SCHEMA:
        raise Stage6ReviewAuthorizationError("authorization schema drift")
    if authorization["stage_id"] != STAGE6_STAGE_ID:
        raise Stage6ReviewAuthorizationError("authorization stage id drift")
    if authorization["formal_run_id"] != expected_run_id:
        raise Stage6ReviewAuthorizationError("authorization formal run id drift")
    scope = _require_exact_fields(
        authorization["single_seed_scope"],
        _SCOPE_FIELDS,
        label="single-seed scope",
    )
    if (
        type(scope["seed"]) is not int
        or type(scope["updates"]) is not int
        or scope != SINGLE_SEED_SCOPE
    ):
        raise Stage6ReviewAuthorizationError("single-seed scope drift")
    if authorization["authorized"] is not True:
        raise Stage6ReviewAuthorizationError("authorization is not explicitly true")
    if (
        authorization["base_commit"] != STAGE5_COMMIT
        or authorization["head_commit"] != STAGE5_COMMIT
    ):
        raise Stage6ReviewAuthorizationError("Stage 5 commit binding drift")

    execution_identity = _validate_artifact_binding(
        authorization["execution_identity"],
        fields=_ARTIFACT_FIELDS,
        label="execution identity",
    )
    frozen_diff = _validate_artifact_binding(
        authorization["frozen_diff"],
        fields=_DIFF_FIELDS,
        label="frozen diff",
    )
    if frozen_diff["format"] != DIFF_FORMAT:
        raise Stage6ReviewAuthorizationError("frozen diff format drift")
    spec_review = _validate_review(
        authorization["spec_review"],
        label="spec review",
        expected_kind="spec_compliance",
        authorization=authorization,
    )
    quality_review = _validate_review(
        authorization["quality_review"],
        label="quality review",
        expected_kind="quality_security",
        authorization=authorization,
    )
    if not _is_sha256(authorization["reviewed_execution_identity_sha256"]):
        raise Stage6ReviewAuthorizationError(
            "reviewed execution identity SHA-256 is invalid"
        )
    if (
        authorization["reviewed_execution_identity_sha256"]
        != execution_identity["sha256"]
    ):
        raise Stage6ReviewAuthorizationError(
            "authorization identity binding drift"
        )
    for name in _REDUNDANT_IDENTITY_FIELDS:
        if not _is_sha256(authorization[name]):
            raise Stage6ReviewAuthorizationError(
                "authorization identity binding drift"
            )
    if not _is_git_oid(authorization["reviewed_prospective_git_tree"]):
        raise Stage6ReviewAuthorizationError(
            "authorization reviewed tree is invalid"
        )
    return execution_identity, frozen_diff, spec_review, quality_review


def _validate_reviewed_identity(
    authorization: Mapping[str, object],
    identity_payload: bytes,
) -> dict[str, object]:
    identity_file = _strict_json_object(
        identity_payload,
        label="execution identity",
    )
    if _canonical_json_bytes(identity_file) != identity_payload:
        raise Stage6ReviewAuthorizationError(
            "execution identity is not canonical JSON"
        )
    identity = _require_mapping(
        authorization["reviewed_execution_identity"],
        label="reviewed execution identity",
    )
    if identity_file != identity:
        raise Stage6ReviewAuthorizationError("execution identity object drift")
    if frozenset(identity) != _EXECUTION_IDENTITY_FIELDS:
        raise Stage6ReviewAuthorizationError(
            "reviewed execution identity field set drift"
        )
    if identity["schema_version"] != "stage6_execution_identity/v1":
        raise Stage6ReviewAuthorizationError("reviewed execution identity schema drift")
    if (
        identity["base_commit"] != STAGE5_COMMIT
        or identity["head_commit"] != STAGE5_COMMIT
        or identity["base_commit"] != authorization["base_commit"]
        or identity["head_commit"] != authorization["head_commit"]
    ):
        raise Stage6ReviewAuthorizationError("Stage 5 commit binding drift")
    if identity["real_index_empty"] is not True:
        raise Stage6ReviewAuthorizationError(
            "reviewed execution identity records a nonempty real index"
        )

    tree = identity["prospective_git_tree"]
    if not _is_git_oid(tree) or identity["prospective_tree_sha256"] != _sha256(
        tree.encode("ascii")
    ):
        raise Stage6ReviewAuthorizationError(
            "reviewed execution identity is internally inconsistent"
        )
    paths = identity["changed_paths"]
    if not isinstance(paths, list) or not paths:
        raise Stage6ReviewAuthorizationError(
            "reviewed changed path set is internally inconsistent"
        )
    for path in paths:
        if not isinstance(path, str):
            raise Stage6ReviewAuthorizationError(
                "reviewed changed path set is internally inconsistent"
            )
        try:
            _safe_relative_reference(path, label="changed path")
        except Stage6ReviewAuthorizationError as exc:
            raise Stage6ReviewAuthorizationError(
                "reviewed changed path set is internally inconsistent"
            ) from exc
    if (
        paths != sorted(set(paths))
        or identity["changed_path_set_sha256"] != _canonical_sha256(paths)
    ):
        raise Stage6ReviewAuthorizationError(
            "reviewed changed path set is internally inconsistent"
        )
    if any(
        not _is_sha256(identity[name]) for name in _REDUNDANT_IDENTITY_FIELDS
    ):
        raise Stage6ReviewAuthorizationError(
            "reviewed execution identity hash is invalid"
        )

    source_identity = _require_mapping(
        identity["source_identity"],
        label="reviewed source identity",
    )
    data_identity = _require_mapping(
        identity["data_identity"],
        label="reviewed data identity",
    )
    environment_identity = _require_mapping(
        identity["environment_identity"],
        label="reviewed environment identity",
    )
    if (
        source_identity.get("source_set_sha256") != identity["source_set_sha256"]
        or _canonical_sha256(data_identity) != identity["data_sha256"]
        or data_identity.get("catalog_sha256") != identity["catalog_sha256"]
        or _canonical_sha256(environment_identity)
        != identity["environment_sha256"]
    ):
        raise Stage6ReviewAuthorizationError(
            "reviewed execution identity is internally inconsistent"
        )
    if (
        authorization["reviewed_prospective_git_tree"]
        != identity["prospective_git_tree"]
        or any(
            authorization[name] != identity[name]
            for name in _REDUNDANT_IDENTITY_FIELDS
        )
    ):
        raise Stage6ReviewAuthorizationError(
            "authorization identity binding drift"
        )
    return identity


def _git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["LC_ALL"] = "C"
    environment["LANG"] = "C"
    return environment


def _run_git(repo: Path, arguments: list[str], *, label: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            capture_output=True,
            check=True,
            shell=False,
            env=_git_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Stage6ReviewAuthorizationError(f"{label} failed") from exc
    return completed.stdout


def _git_text(repo: Path, arguments: list[str], *, label: str) -> str:
    payload = _run_git(repo, arguments, label=label)
    try:
        return payload.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise Stage6ReviewAuthorizationError(f"{label} returned non-ASCII") from exc


def _require_empty_real_index(repo: Path) -> None:
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "diff",
                "--cached",
                "--quiet",
                "--exit-code",
                "--",
            ],
            capture_output=True,
            check=True,
            shell=False,
            env=_git_environment(),
        )
    except subprocess.CalledProcessError as exc:
        raise Stage6ReviewAuthorizationError(
            "real Git index must be empty"
        ) from exc
    except OSError as exc:
        raise Stage6ReviewAuthorizationError(
            "real Git index verification failed"
        ) from exc


def _require_stage5_git_state(repo: Path, reviewed_tree: str) -> None:
    head = _git_text(
        repo,
        ["rev-parse", "--verify", "HEAD^{commit}"],
        label="Stage 6 HEAD verification",
    )
    if head != STAGE5_COMMIT:
        raise Stage6ReviewAuthorizationError("Stage 5 commit binding drift")
    _run_git(
        repo,
        ["cat-file", "-e", f"{STAGE5_COMMIT}^{{commit}}"],
        label="Stage 5 commit object verification",
    )
    _run_git(
        repo,
        ["cat-file", "-e", f"{reviewed_tree}^{{tree}}"],
        label="reviewed tree object verification",
    )
    _require_empty_real_index(repo)


def _canonical_prospective_diff(
    repo: Path,
    *,
    base_commit: str,
    reviewed_tree: str,
) -> bytes:
    return _run_git(
        repo,
        [
            "diff",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--no-color",
            "--src-prefix=a/",
            "--dst-prefix=b/",
            base_commit,
            reviewed_tree,
            "--",
        ],
        label="canonical prospective diff",
    )


def _plain_repo_root(repo_root: str | Path) -> Path:
    lexical = lexical_absolute(repo_root)
    try:
        return require_plain_path(
            lexical,
            leaf_kind="directory",
            label="Stage 6 repository root",
        )
    except PathSecurityError as exc:
        raise Stage6ReviewAuthorizationError(
            "Stage 6 repository root is not a plain directory"
        ) from exc


def _capture_stage6_review_launch_authorization(
    *,
    authorization_path: str | Path,
    repo_root: str | Path,
    config_path: str | Path,
    expected_run_id: str,
    effective_config_bytes: bytes | None = None,
) -> _VerifiedAuthorizationCapture:
    """验证外部 fresh review 身份，并返回可供后续 lineage 绑定的记录。"""

    try:
        formal_run_id = validate_stage6_formal_run_id(expected_run_id)
    except (Stage6WorkflowError, TypeError) as exc:
        raise Stage6ReviewAuthorizationError(
            "expected formal run id is invalid"
        ) from exc

    authorization_file = lexical_absolute(authorization_path)
    package_root = _review_package_root(authorization_file)
    authorization_snapshot = _secure_payload_snapshot(
        authorization_file,
        base=package_root,
        label="authorization",
    )
    authorization_payload = authorization_snapshot.payload
    authorization = _parse_authorization(authorization_payload)
    (
        execution_identity_binding,
        frozen_diff_binding,
        spec_review_binding,
        quality_review_binding,
    ) = _validate_authorization_semantics(
        authorization,
        expected_run_id=formal_run_id,
    )

    references = (
        (execution_identity_binding, "execution identity"),
        (frozen_diff_binding, "frozen diff"),
        (spec_review_binding, "spec review"),
        (quality_review_binding, "quality review"),
    )
    reference_keys = [
        os.path.normcase(
            os.fspath(_reference_path(package_root, binding, label=label))
        )
        for binding, label in references
    ]
    if len(reference_keys) != len(set(reference_keys)):
        raise Stage6ReviewAuthorizationError(
            "authorization references must identify distinct files"
        )

    identity_snapshot = _read_bound_artifact(
        package_root,
        execution_identity_binding,
        label="execution identity",
    )
    frozen_diff_snapshot = _read_bound_artifact(
        package_root,
        frozen_diff_binding,
        label="frozen diff",
    )
    spec_review_snapshot = _read_bound_artifact(
        package_root,
        spec_review_binding,
        label="spec review",
    )
    quality_review_snapshot = _read_bound_artifact(
        package_root,
        quality_review_binding,
        label="quality review",
    )
    _validate_review_document_binding(
        spec_review_snapshot.payload,
        binding=spec_review_binding,
        label="spec review",
    )
    _validate_review_document_binding(
        quality_review_snapshot.payload,
        binding=quality_review_binding,
        label="quality review",
    )
    reviewed_identity = _validate_reviewed_identity(
        authorization,
        identity_snapshot.payload,
    )

    repo = _plain_repo_root(repo_root)
    reviewed_tree = str(reviewed_identity["prospective_git_tree"])
    _require_stage5_git_state(repo, reviewed_tree)
    try:
        identity_kwargs: dict[str, object] = {
            "repo_root": repo,
            "config_path": config_path,
        }
        if effective_config_bytes is not None:
            if type(effective_config_bytes) is not bytes or not effective_config_bytes:
                raise Stage6ReviewAuthorizationError(
                    "effective config bytes are invalid"
                )
            identity_kwargs["effective_config_bytes"] = effective_config_bytes
        current_identity = stage6_execution_identity(**identity_kwargs)
    except Exception as exc:
        raise Stage6ReviewAuthorizationError(
            "current Stage 6 execution identity recomputation failed"
        ) from exc
    if current_identity != reviewed_identity:
        raise Stage6ReviewAuthorizationError(
            "current execution identity differs from reviewed identity"
        )

    canonical_diff = _canonical_prospective_diff(
        repo,
        base_commit=str(authorization["base_commit"]),
        reviewed_tree=reviewed_tree,
    )
    if canonical_diff != frozen_diff_snapshot.payload:
        raise Stage6ReviewAuthorizationError(
            "frozen diff differs from canonical prospective diff"
        )
    _require_stage5_git_state(repo, reviewed_tree)

    record = {
        "schema_version": VERIFIED_SCHEMA,
        "stage_id": STAGE6_STAGE_ID,
        "formal_run_id": formal_run_id,
        "single_seed_scope": dict(SINGLE_SEED_SCOPE),
        "authorized": True,
        "base_commit": STAGE5_COMMIT,
        "authorization_file_sha256": _sha256(authorization_payload),
        "authorization_file_size_bytes": len(authorization_payload),
        "review_identity_sha256": authorization[
            "reviewed_execution_identity_sha256"
        ],
        "reviewed_prospective_git_tree": reviewed_tree,
        "prospective_tree_sha256": reviewed_identity["prospective_tree_sha256"],
        "changed_path_set_sha256": reviewed_identity[
            "changed_path_set_sha256"
        ],
        "source_set_sha256": reviewed_identity["source_set_sha256"],
        "config_sha256": reviewed_identity["config_sha256"],
        "data_sha256": reviewed_identity["data_sha256"],
        "environment_sha256": reviewed_identity["environment_sha256"],
        "frozen_diff_sha256": frozen_diff_binding["sha256"],
        "frozen_diff_size_bytes": frozen_diff_binding["size_bytes"],
        "spec_review_sha256": spec_review_binding["sha256"],
        "quality_review_sha256": quality_review_binding["sha256"],
    }
    return _VerifiedAuthorizationCapture(
        record_payload=_canonical_json_bytes(record),
        members=(
            authorization_snapshot,
            identity_snapshot,
            frozen_diff_snapshot,
            spec_review_snapshot,
            quality_review_snapshot,
        ),
    )


@dataclass(frozen=True, slots=True)
class Stage6ReviewAuthorizationHandle:
    """Immutable authority whose current evidence must be revalidated at use sites."""

    _authorization_path: Path
    _repo_root: Path
    _config_path: Path
    _effective_config_bytes: bytes | None
    _formal_run_id: str
    _record_payload: bytes
    _members: tuple[_PayloadSnapshot, ...]

    @property
    def evidence_paths(self) -> tuple[Path, ...]:
        """Return the exact five evidence members that a formal run must pin."""

        return tuple(member.path for member in self._members)

    @property
    def evidence_records(self) -> tuple[Stage6ReviewEvidenceRecord, ...]:
        """Return immutable path/hash/size records for all five evidence members."""

        return tuple(
            Stage6ReviewEvidenceRecord(
                path=member.path,
                sha256=member.sha256,
                size_bytes=member.size_bytes,
            )
            for member in self._members
        )

    @property
    def record(self) -> Mapping[str, object]:
        value = _strict_json_object(
            self._record_payload,
            label="verified review authorization record",
        )
        frozen = _freeze_json(value)
        if not isinstance(frozen, Mapping):
            raise Stage6ReviewAuthorizationError(
                "verified review authorization record is not a mapping"
            )
        return frozen

    def canonical_record(self) -> dict[str, object]:
        """Return a detached plain JSON object for lineage/checkpoint embedding."""

        value = _strict_json_object(
            self._record_payload,
            label="verified review authorization record",
        )
        if not isinstance(value, dict):
            raise Stage6ReviewAuthorizationError(
                "verified review authorization record is not a mapping"
            )
        return value

    def require_current(self, label: str) -> None:
        if not isinstance(label, str) or not label.strip():
            raise Stage6ReviewAuthorizationError(
                "review authorization revalidation label is invalid"
            )
        try:
            current = _capture_stage6_review_launch_authorization(
                authorization_path=self._authorization_path,
                repo_root=self._repo_root,
                config_path=self._config_path,
                expected_run_id=self._formal_run_id,
                effective_config_bytes=self._effective_config_bytes,
            )
        except Stage6ReviewAuthorizationError as exc:
            raise Stage6ReviewAuthorizationError(
                f"{label} revalidation failed: {exc}"
            ) from exc
        if current.record_payload != self._record_payload:
            raise Stage6ReviewAuthorizationError(f"{label} record changed")
        if len(current.members) != len(self._members):
            raise Stage6ReviewAuthorizationError(f"{label} member set changed")
        for expected, observed in zip(self._members, current.members, strict=True):
            if expected.path != observed.path:
                raise Stage6ReviewAuthorizationError(f"{label} member path changed")
            if expected.stat_identity != observed.stat_identity:
                raise Stage6ReviewAuthorizationError(
                    f"{label} member identity changed"
                )
            if expected.link_count != observed.link_count:
                raise Stage6ReviewAuthorizationError(
                    f"{label} member link count changed"
                )
            if expected.payload != observed.payload:
                raise Stage6ReviewAuthorizationError(f"{label} member bytes changed")
            if (
                expected.sha256 != observed.sha256
                or expected.size_bytes != observed.size_bytes
            ):
                raise Stage6ReviewAuthorizationError(
                    f"{label} member hash or size changed"
                )


def verify_stage6_review_launch_authorization(
    *,
    authorization_path: str | Path,
    repo_root: str | Path,
    config_path: str | Path,
    expected_run_id: str,
    effective_config_bytes: bytes | None = None,
) -> Stage6ReviewAuthorizationHandle:
    """Verify fresh reviews and return a continuously revalidatable handle."""

    capture = _capture_stage6_review_launch_authorization(
        authorization_path=authorization_path,
        repo_root=repo_root,
        config_path=config_path,
        expected_run_id=expected_run_id,
        effective_config_bytes=effective_config_bytes,
    )
    record = _strict_json_object(
        capture.record_payload,
        label="verified review authorization record",
    )
    return Stage6ReviewAuthorizationHandle(
        _authorization_path=lexical_absolute(authorization_path),
        _repo_root=lexical_absolute(repo_root),
        _config_path=lexical_absolute(config_path),
        _effective_config_bytes=effective_config_bytes,
        _formal_run_id=str(record["formal_run_id"]),
        _record_payload=capture.record_payload,
        _members=capture.members,
    )


__all__ = [
    "Stage6ReviewAuthorizationError",
    "Stage6ReviewAuthorizationHandle",
    "verify_stage6_review_launch_authorization",
]
