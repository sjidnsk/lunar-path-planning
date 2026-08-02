"""Stage 6 checkpoint retention；仅逐个删除已知文件。"""

from __future__ import annotations

# Security scope: this rejects testable same-host project-runner path swaps.
# It does not claim resistance to a privileged administrator or kernel attack.

import json
import os
import re
import stat
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from lunar_exploration_ppo.ppo.checkpoint import (
    CHECKPOINT_COMPLETE_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_LATEST_SCHEMA_VERSION,
    STAGE6_CHECKPOINT_COMPLETE_SCHEMA_VERSION,
    STAGE6_CHECKPOINT_LATEST_SCHEMA_VERSION,
    STAGE6_CHECKPOINT_SCHEMA_VERSION,
)
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.path_security import (
    DurableDirectoryIdentity,
    DurableFileIdentity,
    PathSecurityError,
    durable_file_identity,
    durable_rename,
    durable_rmdir,
    durable_unlink,
    is_link_or_reparse,
    lexical_absolute,
    path_identity,
    require_plain_path,
)


_UPDATE_RE = re.compile(r"update-(?P<update>[0-9]{8})")
_PENDING_UPDATE_RE = re.compile(
    r"\.pending-update-(?P<update>[0-9]{8})-(?P<transaction>[0-9a-f]{32})"
)
_RETENTION_TOMBSTONE_RE = re.compile(
    r"\.retention-tombstone-update-(?P<update>[0-9]{8})-"
    r"(?P<transaction>[0-9a-f]{32})"
)
_MEMBER_DELETE_ORDER = ("checkpoint.pt", "manifest.json", "complete.json")
_KNOWN_FILES = frozenset(_MEMBER_DELETE_ORDER)
_RECOVERY_NAME_PREFIXES = (".pending-", ".retention-tombstone-")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NamedMemberIdentity = tuple[str, DurableFileIdentity]


class CheckpointRetentionError(RuntimeError):
    """Retention 目标不完整、含未知成员或越界。"""


@dataclass(frozen=True, slots=True)
class CheckpointRetentionReceipt:
    kept_updates: tuple[int, ...]
    removed_updates: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _ValidatedUpdate:
    path: Path
    directory_identity: DurableDirectoryIdentity
    member_identities: tuple[_NamedMemberIdentity, ...]


@dataclass(frozen=True, slots=True)
class _ValidatedTransaction:
    path: Path
    directory_identity: DurableDirectoryIdentity
    member_identities: tuple[_NamedMemberIdentity, ...]


class CheckpointRetentionManager:
    def __init__(
        self,
        root: str | Path,
        *,
        schema_version: str = CHECKPOINT_SCHEMA_VERSION,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        unresolved = lexical_absolute(root)
        try:
            require_plain_path(
                unresolved,
                allow_missing=True,
                label="checkpoint retention root",
            )
            unresolved.mkdir(parents=True, exist_ok=True)
            self.root = require_plain_path(
                unresolved,
                leaf_kind="directory",
                label="checkpoint retention root",
            )
        except (OSError, PathSecurityError) as exc:
            raise CheckpointRetentionError(
                "checkpoint retention root is a link/reparse point"
            ) from exc
        self._unresolved_root = unresolved
        self._root_identity = path_identity(unresolved)
        self._fault_injector = fault_injector
        latest_schemas = {
            CHECKPOINT_SCHEMA_VERSION: CHECKPOINT_LATEST_SCHEMA_VERSION,
            STAGE6_CHECKPOINT_SCHEMA_VERSION: STAGE6_CHECKPOINT_LATEST_SCHEMA_VERSION,
        }
        if schema_version not in latest_schemas:
            raise CheckpointRetentionError("checkpoint retention schema is unsupported")
        self._latest_schema = latest_schemas[schema_version]
        if self._startup_recovery_required():
            self._validated_directories()

    def _require_root_current(self) -> None:
        try:
            resolved = require_plain_path(
                self._unresolved_root,
                leaf_kind="directory",
                label="checkpoint retention root",
            )
            current = path_identity(self._unresolved_root)
        except PathSecurityError as exc:
            raise CheckpointRetentionError(
                "checkpoint retention root is a link/reparse point"
            ) from exc
        if resolved != self.root or (
            current[0], current[1], current[2], current[4]
        ) != (
            self._root_identity[0],
            self._root_identity[1],
            self._root_identity[2],
            self._root_identity[4],
        ):
            raise CheckpointRetentionError(
                "checkpoint retention root identity changed"
            )

    def _startup_recovery_required(self) -> bool:
        self._require_root_current()
        return any(
            path.name.startswith(_RECOVERY_NAME_PREFIXES)
            for path in self.root.iterdir()
        )

    def apply(
        self,
        *,
        latest_update: int,
        periodic_updates: tuple[int, ...],
        best_update: int,
        periodic_keep_count: int,
        protected_updates: tuple[int, ...] = (),
    ) -> CheckpointRetentionReceipt:
        if periodic_keep_count != 5:
            raise CheckpointRetentionError("periodic keep count must remain 5")
        if (
            not isinstance(protected_updates, tuple)
            or any(
                type(update) is not int or update <= 0
                for update in protected_updates
            )
            or tuple(sorted(set(protected_updates))) != protected_updates
        ):
            raise CheckpointRetentionError("protected checkpoint set drifted")
        directories, latest = self._validated_directories()
        available = set(directories)
        if latest is not None and latest["update_step"] != latest_update:
            raise CheckpointRetentionError("checkpoint latest index drifted")
        if latest_update not in available or best_update not in available:
            raise CheckpointRetentionError("latest or best checkpoint is not complete")
        periodic = tuple(sorted(set(periodic_updates))[-periodic_keep_count:])
        if not set(periodic).issubset(available):
            raise CheckpointRetentionError("periodic checkpoint is not complete")
        if not set(protected_updates).issubset(available):
            raise CheckpointRetentionError("protected checkpoint is not complete")
        keep = {latest_update, best_update, *periodic, *protected_updates}
        remove = tuple(sorted(available - keep))
        for update in remove:
            self._remove_update(directories[update])
        return CheckpointRetentionReceipt(
            kept_updates=tuple(sorted(keep)),
            removed_updates=remove,
        )

    def _inject(self, phase: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(phase)

    def _remove_update(self, validated: _ValidatedUpdate) -> None:
        self._require_update_current(validated)

        # Preserve the established race-injection gates while the live update
        # still has its canonical path. Actual unlinks happen only after rename.
        for name in _MEMBER_DELETE_ORDER:
            self._inject(f"before_member_unlink:{name}")
            self._require_update_current(validated)
        self._inject("before_directory_rmdir")
        self._require_update_current(validated)

        tombstone = self._new_tombstone_path(validated.path.name)
        self._require_root_current()
        self._require_update_current(validated)
        try:
            durable_rename(
                validated.path,
                tombstone,
                expected_source_identity=validated.directory_identity,
                replace_existing=False,
            )
        except (OSError, PathSecurityError) as exc:
            raise CheckpointRetentionError(
                "checkpoint update tombstone rename failed closed"
            ) from exc

        self._inject("after_update_tombstone_rename")
        self._require_root_current()
        if _path_lexists(validated.path):
            raise CheckpointRetentionError(
                "checkpoint update remained after tombstone rename"
            )
        try:
            transaction = self._validate_transaction_directory(tombstone)
        except CheckpointRetentionError as exc:
            raise CheckpointRetentionError(
                "checkpoint update identity changed after tombstone rename"
            ) from exc
        if (
            transaction.directory_identity != validated.directory_identity
            or transaction.member_identities != validated.member_identities
        ):
            raise CheckpointRetentionError(
                "checkpoint update identity changed after tombstone rename"
            )
        self._remove_transaction(transaction)

    def _require_update_current(self, validated: _ValidatedUpdate) -> None:
        self._require_root_current()
        try:
            current = self._validate_update_directory(validated.path)
        except CheckpointRetentionError as exc:
            raise CheckpointRetentionError(
                "checkpoint update identity changed before retention"
            ) from exc
        if current != validated:
            raise CheckpointRetentionError(
                "checkpoint update identity changed before retention"
            )
        self._require_root_current()

    def _new_tombstone_path(self, directory_name: str) -> Path:
        tombstone = self.root / (
            f".retention-tombstone-{directory_name}-{uuid.uuid4().hex}"
        )
        self._require_root_current()
        try:
            resolved = require_plain_path(
                tombstone,
                base=self.root,
                allow_missing=True,
                label="checkpoint retention tombstone",
            )
        except PathSecurityError as exc:
            raise CheckpointRetentionError(
                "checkpoint retention tombstone escaped root"
            ) from exc
        if resolved != tombstone or _path_lexists(tombstone):
            raise CheckpointRetentionError(
                "checkpoint retention tombstone destination is not empty"
            )
        return tombstone

    def _remove_transaction(self, validated: _ValidatedTransaction) -> None:
        try:
            current = self._validate_transaction_directory(validated.path)
        except CheckpointRetentionError as exc:
            raise CheckpointRetentionError(
                "checkpoint transaction identity changed before recovery"
            ) from exc
        if current != validated:
            raise CheckpointRetentionError(
                "checkpoint transaction identity changed before recovery"
            )

        expected_by_name = dict(validated.member_identities)
        for name in _MEMBER_DELETE_ORDER:
            expected = expected_by_name.get(name)
            if expected is None:
                continue
            member = validated.path / name
            self._require_transaction_member_current(
                validated,
                member=member,
                expected=expected,
            )
            self._inject(f"before_transaction_member_unlink:{name}")
            self._require_transaction_member_current(
                validated,
                member=member,
                expected=expected,
            )
            try:
                durable_unlink(member, expected_identity=expected)
            except (OSError, PathSecurityError) as exc:
                raise CheckpointRetentionError(
                    "checkpoint transaction member unlink failed closed"
                ) from exc
            self._inject(f"after_transaction_member_unlink:{name}")
            self._require_root_current()
            if _path_lexists(member):
                raise CheckpointRetentionError(
                    "checkpoint transaction member remained after unlink"
                )

        self._inject("before_transaction_directory_rmdir")
        self._require_root_current()
        try:
            invalid_directory = (
                _is_link_or_reparse(validated.path)
                or _directory_identity(validated.path)
                != validated.directory_identity
                or any(validated.path.iterdir())
                or validated.path.resolve(strict=True).parent != self.root
            )
        except (OSError, CheckpointRetentionError) as exc:
            raise CheckpointRetentionError(
                "checkpoint transaction directory identity changed before rmdir"
            ) from exc
        if invalid_directory:
            raise CheckpointRetentionError(
                "checkpoint transaction directory identity changed before rmdir"
            )
        try:
            durable_rmdir(
                validated.path,
                expected_identity=validated.directory_identity,
            )
        except (OSError, PathSecurityError) as exc:
            raise CheckpointRetentionError(
                "checkpoint transaction directory rmdir failed closed"
            ) from exc
        if _path_lexists(validated.path):
            raise CheckpointRetentionError(
                "checkpoint transaction directory remained after rmdir"
            )

    def _require_transaction_member_current(
        self,
        validated: _ValidatedTransaction,
        *,
        member: Path,
        expected: DurableFileIdentity,
    ) -> None:
        self._require_root_current()
        try:
            self._require_contained_regular_file(member)
            if (
                _directory_identity(validated.path)
                != validated.directory_identity
                or _descriptor_file_identity(member) != (member.name, expected)
                or _file_identity(member) != (member.name, expected)
            ):
                raise CheckpointRetentionError(
                    "checkpoint transaction member identity changed before unlink"
                )
        except CheckpointRetentionError as exc:
            if "identity changed" in str(exc):
                raise
            raise CheckpointRetentionError(
                "checkpoint transaction member identity changed before unlink"
            ) from exc
        self._require_root_current()

    def verify_retained(
        self,
        *,
        latest_update: int,
        expected_updates: tuple[int, ...],
    ) -> CheckpointRetentionReceipt:
        if (
            type(latest_update) is not int
            or latest_update <= 0
            or not isinstance(expected_updates, tuple)
            or any(type(update) is not int or update <= 0 for update in expected_updates)
        ):
            raise CheckpointRetentionError("retained checkpoint expectation drifted")
        expected = tuple(sorted(set(expected_updates)))
        if expected != expected_updates or latest_update not in expected:
            raise CheckpointRetentionError("retained checkpoint expectation drifted")
        directories, latest = self._validated_directories()
        if (
            latest is None
            or latest.get("update_step") != latest_update
            or tuple(sorted(directories)) != expected
        ):
            raise CheckpointRetentionError("retained checkpoint set drifted")
        return CheckpointRetentionReceipt(
            kept_updates=expected,
            removed_updates=(),
        )

    def _validated_directories(
        self,
    ) -> tuple[dict[int, _ValidatedUpdate], dict[str, object] | None]:
        directories, latest, transactions = self._scan_root()
        for transaction in transactions:
            self._remove_transaction(transaction)
        if transactions:
            directories, latest, remaining = self._scan_root()
            if remaining:
                raise CheckpointRetentionError(
                    "checkpoint recovery transaction remained after cleanup"
                )
        return directories, latest

    def _scan_root(
        self,
    ) -> tuple[
        dict[int, _ValidatedUpdate],
        dict[str, object] | None,
        tuple[_ValidatedTransaction, ...],
    ]:
        self._require_root_current()
        if _is_link_or_reparse(self.root):
            raise CheckpointRetentionError("checkpoint retention root is a link/reparse point")
        directories: dict[int, _ValidatedUpdate] = {}
        latest: dict[str, object] | None = None
        transactions: list[_ValidatedTransaction] = []
        for path in sorted(self.root.iterdir(), key=lambda candidate: candidate.name):
            if _is_link_or_reparse(path):
                raise CheckpointRetentionError(
                    "checkpoint root member is a link/reparse point"
                )
            self._require_contained(path)
            if not path.is_dir():
                if path.name != "latest.json" or latest is not None:
                    raise CheckpointRetentionError("checkpoint root contains unknown member")
                latest = self._validate_latest(path)
                continue
            match = _UPDATE_RE.fullmatch(path.name)
            if match is not None:
                directories[int(match.group("update"))] = self._validate_update_directory(
                    path
                )
                continue
            if (
                _PENDING_UPDATE_RE.fullmatch(path.name) is not None
                or _RETENTION_TOMBSTONE_RE.fullmatch(path.name) is not None
            ):
                transactions.append(self._validate_transaction_directory(path))
                continue
            raise CheckpointRetentionError("checkpoint root contains unknown directory")
        if latest is not None:
            update = int(latest["update_step"])
            validated = directories.get(update)
            if validated is None or latest["directory"] != validated.path.name:
                raise CheckpointRetentionError("checkpoint latest directory drifted")
            complete = self._read_canonical_json(
                validated.path / "complete.json", "complete"
            )
            if (
                complete.get("manifest_sha256") != latest["manifest_sha256"]
                or complete.get("checkpoint_sha256") != latest["checkpoint_sha256"]
            ):
                raise CheckpointRetentionError("checkpoint latest hash binding drifted")
        return directories, latest, tuple(transactions)

    def _validate_update_directory(self, path: Path) -> _ValidatedUpdate:
        if _is_link_or_reparse(path) or not path.is_dir():
            raise CheckpointRetentionError(
                "checkpoint update directory is a link/reparse point"
            )
        self._require_contained(path)
        members = tuple(path.iterdir())
        if {member.name for member in members} != _KNOWN_FILES:
            raise CheckpointRetentionError(
                f"checkpoint update {path.name} contains unknown or incomplete members"
            )
        identities: list[_NamedMemberIdentity] = []
        for member in members:
            self._require_contained_regular_file(member)
            identities.append(_file_identity(member))
        return _ValidatedUpdate(
            path=path,
            directory_identity=_directory_identity(path),
            member_identities=tuple(sorted(identities)),
        )

    def _validate_transaction_directory(self, path: Path) -> _ValidatedTransaction:
        if _is_link_or_reparse(path) or not path.is_dir():
            raise CheckpointRetentionError(
                "checkpoint transaction directory is a link/reparse point"
            )
        self._require_contained(path)
        members = tuple(sorted(path.iterdir(), key=lambda member: member.name))
        if not {member.name for member in members}.issubset(_KNOWN_FILES):
            raise CheckpointRetentionError(
                f"checkpoint transaction {path.name} contains unknown members"
            )
        identities: list[_NamedMemberIdentity] = []
        for member in members:
            self._require_contained_regular_file(member)
            identities.append(_file_identity(member))
        return _ValidatedTransaction(
            path=path,
            directory_identity=_directory_identity(path),
            member_identities=tuple(sorted(identities)),
        )

    def _require_contained(self, path: Path) -> None:
        try:
            path.resolve(strict=True).relative_to(self.root)
        except (OSError, ValueError) as exc:
            raise CheckpointRetentionError(
                "checkpoint retention path escaped root"
            ) from exc

    def _require_contained_regular_file(self, path: Path) -> None:
        if _is_link_or_reparse(path) or not path.is_file():
            raise CheckpointRetentionError(
                "checkpoint member is a link/reparse point or non-file"
            )
        self._require_contained(path)

    def _validate_latest(self, path: Path) -> dict[str, object]:
        value = self._read_canonical_json(path, "latest")
        expected = {
            "schema_version",
            "update_step",
            "directory",
            "manifest_sha256",
            "checkpoint_sha256",
        }
        if (
            set(value) != expected
            or value.get("schema_version") != self._latest_schema
            or type(value.get("update_step")) is not int
            or int(value["update_step"]) <= 0
            or value.get("directory") != f"update-{int(value['update_step']):08d}"
            or any(
                not isinstance(value.get(name), str)
                or _SHA256_RE.fullmatch(str(value[name])) is None
                for name in ("manifest_sha256", "checkpoint_sha256")
            )
        ):
            raise CheckpointRetentionError("checkpoint latest index drifted")
        return value

    def _read_canonical_json(self, path: Path, label: str) -> dict[str, object]:
        try:
            self._require_root_current()
            payload = _secure_read_bytes(path)
            self._require_root_current()
            value = json.loads(payload.decode("utf-8"))
        except (OSError, PathSecurityError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointRetentionError(f"checkpoint {label} JSON drifted") from exc
        if (
            not isinstance(value, dict)
            or ArtifactStore.canonical_json_bytes(value) != payload
        ):
            raise CheckpointRetentionError(f"checkpoint {label} JSON drifted")
        return value


def verify_retained_checkpoint_snapshot(
    *,
    root_relative: str,
    schema_version: str,
    latest_update: int,
    expected_updates: tuple[int, ...],
    directory_snapshot: Sequence[Mapping[str, object]],
    read_bytes: Callable[..., bytes],
) -> CheckpointRetentionReceipt:
    """Verify retention from one bound file/directory snapshot only."""

    if (
        type(latest_update) is not int
        or latest_update <= 0
        or not isinstance(expected_updates, tuple)
        or any(type(update) is not int or update <= 0 for update in expected_updates)
    ):
        raise CheckpointRetentionError("retained checkpoint expectation drifted")
    expected = tuple(sorted(set(expected_updates)))
    if expected != expected_updates or latest_update not in expected:
        raise CheckpointRetentionError("retained checkpoint expectation drifted")
    root = _snapshot_relative_path(root_relative, label="checkpoint snapshot root")
    latest_schemas = {
        CHECKPOINT_SCHEMA_VERSION: CHECKPOINT_LATEST_SCHEMA_VERSION,
        STAGE6_CHECKPOINT_SCHEMA_VERSION: STAGE6_CHECKPOINT_LATEST_SCHEMA_VERSION,
    }
    complete_schemas = {
        CHECKPOINT_SCHEMA_VERSION: CHECKPOINT_COMPLETE_SCHEMA_VERSION,
        STAGE6_CHECKPOINT_SCHEMA_VERSION: STAGE6_CHECKPOINT_COMPLETE_SCHEMA_VERSION,
    }
    if schema_version not in latest_schemas:
        raise CheckpointRetentionError("checkpoint retention schema is unsupported")
    directories = _validated_snapshot_directories(directory_snapshot)
    expected_directory_names = {
        _update_directory_name(update) for update in expected
    }
    expected_paths = {
        root,
        *(f"{root}/{name}" for name in expected_directory_names),
    }
    subtree_paths = {
        path for path in directories if path == root or path.startswith(f"{root}/")
    }
    if subtree_paths != expected_paths:
        raise CheckpointRetentionError("retained checkpoint directory snapshot drifted")
    if directories[root] != tuple(
        sorted(
            {
                ("latest.json", "file"),
                *((name, "directory") for name in expected_directory_names),
            }
        )
    ):
        raise CheckpointRetentionError("retained checkpoint root membership drifted")
    expected_members = tuple(sorted((name, "file") for name in _KNOWN_FILES))
    for update in expected:
        directory = f"{root}/{_update_directory_name(update)}"
        if directories[directory] != expected_members:
            raise CheckpointRetentionError("retained checkpoint member snapshot drifted")

    latest_path = f"{root}/latest.json"
    latest = _snapshot_canonical_json(
        _bound_snapshot_read(read_bytes, latest_path, "checkpoint latest snapshot"),
        label="latest",
    )
    if (
        set(latest)
        != {
            "schema_version",
            "update_step",
            "directory",
            "manifest_sha256",
            "checkpoint_sha256",
        }
        or latest.get("schema_version") != latest_schemas[schema_version]
        or latest.get("update_step") != latest_update
        or latest.get("directory") != _update_directory_name(latest_update)
        or any(
            not isinstance(latest.get(name), str)
            or _SHA256_RE.fullmatch(str(latest[name])) is None
            for name in ("manifest_sha256", "checkpoint_sha256")
        )
    ):
        raise CheckpointRetentionError("checkpoint latest index drifted")
    complete_path = (
        f"{root}/{_update_directory_name(latest_update)}/complete.json"
    )
    complete = _snapshot_canonical_json(
        _bound_snapshot_read(
            read_bytes,
            complete_path,
            "checkpoint latest complete snapshot",
        ),
        label="complete",
    )
    if (
        set(complete)
        != {
            "schema_version",
            "update_step",
            "manifest_sha256",
            "checkpoint_sha256",
        }
        or complete.get("schema_version") != complete_schemas[schema_version]
        or complete.get("update_step") != latest_update
        or complete.get("manifest_sha256") != latest["manifest_sha256"]
        or complete.get("checkpoint_sha256") != latest["checkpoint_sha256"]
    ):
        raise CheckpointRetentionError("checkpoint latest hash binding drifted")
    return CheckpointRetentionReceipt(kept_updates=expected, removed_updates=())


def _snapshot_relative_path(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith(("/", "\\"))
        or "\\" in value
        or ":" in value
    ):
        raise CheckpointRetentionError(f"{label} drifted")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise CheckpointRetentionError(f"{label} drifted")
    return "/".join(parts)


def _validated_snapshot_directories(
    value: Sequence[Mapping[str, object]],
) -> dict[str, tuple[tuple[str, str], ...]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CheckpointRetentionError("checkpoint directory snapshot drifted")
    result: dict[str, tuple[tuple[str, str], ...]] = {}
    ordered_paths: list[str] = []
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "identity", "members"}:
            raise CheckpointRetentionError("checkpoint directory snapshot row drifted")
        path_value = raw.get("path")
        path = (
            "."
            if path_value == "."
            else _snapshot_relative_path(path_value, label="checkpoint snapshot directory")
        )
        identity = raw.get("identity")
        if (
            not isinstance(identity, Mapping)
            or set(identity)
            != {"device", "inode", "mode", "file_attributes", "link_count"}
            or any(type(item) is not int for item in identity.values())
        ):
            raise CheckpointRetentionError("checkpoint directory identity snapshot drifted")
        members_value = raw.get("members")
        if not isinstance(members_value, list):
            raise CheckpointRetentionError("checkpoint directory members snapshot drifted")
        members: list[tuple[str, str]] = []
        seen_names: set[str] = set()
        for member in members_value:
            if not isinstance(member, Mapping) or set(member) != {"name", "kind"}:
                raise CheckpointRetentionError("checkpoint directory member row drifted")
            name = member.get("name")
            kind = member.get("kind")
            if (
                not isinstance(name, str)
                or not name
                or name in {".", ".."}
                or "/" in name
                or "\\" in name
                or ":" in name
                or kind not in {"file", "directory"}
                or name in seen_names
            ):
                raise CheckpointRetentionError("checkpoint directory member drifted")
            seen_names.add(name)
            members.append((name, str(kind)))
        if members != sorted(members) or path in result:
            raise CheckpointRetentionError("checkpoint directory snapshot order drifted")
        ordered_paths.append(path)
        result[path] = tuple(members)
    if ordered_paths != sorted(ordered_paths):
        raise CheckpointRetentionError("checkpoint directory snapshot order drifted")
    return result


def _bound_snapshot_read(
    reader: Callable[..., bytes],
    relative: str,
    label: str,
) -> bytes:
    if not callable(reader):
        raise CheckpointRetentionError("checkpoint snapshot reader drifted")
    try:
        payload = reader(relative, label=label)
    except Exception as exc:
        raise CheckpointRetentionError(f"{label} failed") from exc
    if type(payload) is not bytes:
        raise CheckpointRetentionError(f"{label} bytes drifted")
    return payload


def _snapshot_canonical_json(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointRetentionError(f"checkpoint {label} JSON drifted") from exc
    if (
        not isinstance(value, dict)
        or ArtifactStore.canonical_json_bytes(value) != payload
    ):
        raise CheckpointRetentionError(f"checkpoint {label} JSON drifted")
    return value


def _update_directory_name(update: int) -> str:
    return f"update-{update:08d}"


def _is_link_or_reparse(path: Path) -> bool:
    return is_link_or_reparse(path)


def _directory_identity(path: Path) -> DurableDirectoryIdentity:
    try:
        value = path.lstat()
    except OSError as exc:
        raise CheckpointRetentionError(
            "checkpoint directory identity is unavailable"
        ) from exc
    if not stat.S_ISDIR(value.st_mode):
        raise CheckpointRetentionError(
            "checkpoint directory identity is unavailable"
        )
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(stat.S_IFMT(value.st_mode)),
        int(getattr(value, "st_file_attributes", 0)),
    )


def _file_identity(path: Path) -> _NamedMemberIdentity:
    try:
        identity = durable_file_identity(path)
    except (OSError, PathSecurityError) as exc:
        raise CheckpointRetentionError(
            "checkpoint member identity is unavailable"
        ) from exc
    return (path.name, identity)


def _descriptor_file_identity(path: Path) -> _NamedMemberIdentity:
    descriptor = -1
    try:
        flags = os.O_RDONLY
        for name in ("O_BINARY", "O_CLOEXEC", "O_NOINHERIT", "O_NOFOLLOW"):
            flags |= int(getattr(os, name, 0))
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CheckpointRetentionError(
                "checkpoint member descriptor is not a regular file"
            )
        durable_identity: DurableFileIdentity = (
            int(metadata.st_dev),
            int(metadata.st_ino),
            int(stat.S_IFMT(metadata.st_mode)),
            int(metadata.st_size),
            int(getattr(metadata, "st_file_attributes", 0)),
            int(metadata.st_nlink),
        )
        if durable_identity[5] != 1:
            raise CheckpointRetentionError(
                "checkpoint member descriptor hard link count drifted"
            )
        identity = (path.name, durable_identity)
    except CheckpointRetentionError:
        raise
    except OSError as exc:
        raise CheckpointRetentionError(
            "checkpoint member descriptor open failed closed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if _file_identity(path) != identity:
        raise CheckpointRetentionError(
            "checkpoint member identity changed after descriptor open"
        )
    return identity


def _secure_read_bytes(path: Path) -> bytes:
    descriptor = -1
    try:
        flags = os.O_RDONLY
        for name in ("O_BINARY", "O_CLOEXEC", "O_NOINHERIT", "O_NOFOLLOW"):
            flags |= int(getattr(os, name, 0))
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CheckpointRetentionError(
                "checkpoint JSON descriptor is not a regular file"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
        after = os.fstat(descriptor)
        durable_descriptor_identity: DurableFileIdentity = (
            int(after.st_dev),
            int(after.st_ino),
            int(stat.S_IFMT(after.st_mode)),
            int(after.st_size),
            int(getattr(after, "st_file_attributes", 0)),
            int(after.st_nlink),
        )
        if durable_descriptor_identity[5] != 1:
            raise CheckpointRetentionError(
                "checkpoint JSON hard link count drifted while read"
            )
        descriptor_identity = (path.name, durable_descriptor_identity)
        if (
            int(before.st_dev),
            int(before.st_ino),
            int(stat.S_IFMT(before.st_mode)),
            int(before.st_size),
            int(getattr(before, "st_file_attributes", 0)),
            int(before.st_nlink),
        ) != durable_descriptor_identity or len(payload) != durable_descriptor_identity[3]:
            raise CheckpointRetentionError(
                "checkpoint JSON identity changed while read"
            )
    except CheckpointRetentionError:
        raise
    except OSError as exc:
        raise CheckpointRetentionError(
            "checkpoint JSON descriptor read failed closed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if _file_identity(path) != descriptor_identity:
        raise CheckpointRetentionError(
            "checkpoint JSON identity changed while read"
        )
    return payload


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


__all__ = [
    "CheckpointRetentionError",
    "CheckpointRetentionManager",
    "CheckpointRetentionReceipt",
    "verify_retained_checkpoint_snapshot",
]
