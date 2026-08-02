"""Stage 1 artifact-set contracts, path safety, and manifest verification."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping

from pydantic import ValidationError

from lunar_exploration_ppo.configs.stage1 import Stage1Config


STAGE1_MACHINE_ARTIFACTS: Final = (
    "config.json",
    "summary.json",
    "routing.json",
    "report.md",
    "metrics.jsonl",
    "phase-state.jsonl",
)
STAGE1_FORBIDDEN_ARTIFACTS: Final = frozenset({"approval.json", "gate.json"})
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


class Stage1WorkflowError(RuntimeError):
    """Raised when a workflow cannot run without artifact drift."""


def _absolute_path(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _is_reparse(stat_result: os.stat_result) -> bool:
    return bool(getattr(stat_result, "st_file_attributes", 0) & 0x400)


def _stat_token(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_mode),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
    )


def _handle_stat_token(stat_result: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        *_stat_token(stat_result),
        int(stat_result.st_ctime_ns),
    )


def _require_no_reparse_components(path: Path, label: str) -> None:
    current = Path(path.anchor)
    parts = path.parts[1:] if path.anchor else path.parts
    for part in parts:
        current /= part
        try:
            current_stat = os.lstat(current)
        except OSError as exc:
            raise Stage1WorkflowError(f"{label} snapshot source is missing") from exc
        if stat.S_ISLNK(current_stat.st_mode) or _is_reparse(current_stat):
            raise Stage1WorkflowError(f"{label} snapshot path uses a symlink or reparse point")


def _require_within_root(path: Path, root: Path, label: str) -> None:
    normalized_path = os.path.normcase(os.fspath(path))
    normalized_root = os.path.normcase(os.fspath(root))
    try:
        common = os.path.commonpath((normalized_path, normalized_root))
    except ValueError as exc:
        raise Stage1WorkflowError(f"{label} snapshot path escapes its authority root") from exc
    if common != normalized_root:
        raise Stage1WorkflowError(f"{label} snapshot path escapes its authority root")


def _require_casefold_unique(names: tuple[str, ...], label: str) -> None:
    normalized = [name.casefold() for name in names]
    if len(normalized) != len(set(normalized)):
        raise Stage1WorkflowError(f"{label} contains case-normalized duplicate paths")


@dataclass(frozen=True, slots=True)
class FrozenFileSnapshot:
    """One regular file captured from a stable open handle."""

    path: Path
    payload: bytes
    sha256: str
    size_bytes: int
    _identity: tuple[int, int, int, int, int]

    @classmethod
    def capture(
        cls,
        path: str | Path,
        label: str = "file",
        *,
        authority_root: str | Path | None = None,
    ) -> "FrozenFileSnapshot":
        absolute = _absolute_path(path)
        if authority_root is not None:
            root = _absolute_path(authority_root)
            _require_within_root(absolute, root, label)
        _require_no_reparse_components(absolute, label)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(absolute, flags)
        except OSError as exc:
            raise Stage1WorkflowError(f"{label} snapshot source is missing or unsafe") from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or _is_reparse(before):
                raise Stage1WorkflowError(f"{label} snapshot source is not a regular file")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        payload = b"".join(chunks)
        before_token = _stat_token(before)
        if _handle_stat_token(before) != _handle_stat_token(after) or len(payload) != before.st_size:
            raise Stage1WorkflowError(f"{label} snapshot source changed during capture")
        try:
            path_stat = os.lstat(absolute)
        except OSError as exc:
            raise Stage1WorkflowError(f"{label} snapshot source changed during capture") from exc
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or stat.S_ISLNK(path_stat.st_mode)
            or _is_reparse(path_stat)
            or _stat_token(path_stat) != before_token
        ):
            raise Stage1WorkflowError(f"{label} snapshot source changed during capture")
        return cls(
            path=absolute,
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            _identity=before_token,
        )

    @property
    def lf_count(self) -> int:
        return self.payload.count(b"\n")

    @property
    def logical_line_count(self) -> int:
        return len(self.payload.splitlines())

    def binding(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "bytes": self.size_bytes,
            "lf_count": self.lf_count,
            "logical_line_count": self.logical_line_count,
        }

    def require_current(self, label: str = "file") -> None:
        current = type(self).capture(self.path, label)
        if current.payload != self.payload or current._identity != self._identity:
            raise Stage1WorkflowError(f"{label} snapshot drift detected")


def _directory_members(path: Path, label: str) -> tuple[tuple[tuple[str, str], ...], tuple[int, ...]]:
    _require_no_reparse_components(path, label)
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise Stage1WorkflowError(f"{label} snapshot directory is missing") from exc
    if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode) or _is_reparse(before):
        raise Stage1WorkflowError(f"{label} snapshot source is not a regular directory")
    members: list[tuple[str, str]] = []
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                entry_stat = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or _is_reparse(entry_stat):
                    kind = "reparse"
                elif stat.S_ISREG(entry_stat.st_mode):
                    kind = "regular_file"
                elif stat.S_ISDIR(entry_stat.st_mode):
                    kind = "directory"
                else:
                    kind = "non_regular"
                members.append((entry.name, kind))
    except OSError as exc:
        raise Stage1WorkflowError(f"{label} snapshot directory changed during capture") from exc
    after = os.lstat(path)
    if _stat_token(before) != _stat_token(after):
        raise Stage1WorkflowError(f"{label} snapshot directory changed during capture")
    ordered = tuple(sorted(members, key=lambda item: item[0]))
    _require_casefold_unique(tuple(name for name, _kind in ordered), label)
    return ordered, _stat_token(before)


@dataclass(frozen=True, slots=True)
class FrozenDirectorySnapshot:
    """An exact shallow directory member set plus frozen regular-file payloads."""

    path: Path
    members: tuple[tuple[str, str], ...]
    files: tuple[tuple[str, FrozenFileSnapshot], ...]
    _identity: tuple[int, ...]

    @classmethod
    def capture_exact_regular_files(
        cls,
        path: str | Path,
        expected_names: tuple[str, ...] | set[str] | frozenset[str],
        label: str = "directory",
    ) -> "FrozenDirectorySnapshot":
        return cls.capture_one_of_exact_regular_file_sets(
            path,
            (expected_names,),
            label,
        )

    @classmethod
    def capture_one_of_exact_regular_file_sets(
        cls,
        path: str | Path,
        expected_name_sets: tuple[
            tuple[str, ...] | set[str] | frozenset[str],
            ...,
        ],
        label: str = "directory",
    ) -> "FrozenDirectorySnapshot":
        """Capture one explicitly allowed exact shallow regular-file member set."""

        absolute = _absolute_path(path)
        if not expected_name_sets:
            raise Stage1WorkflowError(f"{label} has no allowed regular-file set")
        alternatives = tuple(tuple(sorted(names)) for names in expected_name_sets)
        for expected in alternatives:
            _require_casefold_unique(expected, label)
        if len(alternatives) != len(set(alternatives)):
            raise Stage1WorkflowError(f"{label} allowed regular-file sets are duplicated")
        members_before, identity = _directory_members(absolute, label)
        actual_names = tuple(name for name, _kind in members_before)
        if actual_names not in alternatives or any(kind != "regular_file" for _name, kind in members_before):
            raise Stage1WorkflowError(f"{label} does not contain the exact regular-file set")
        files = tuple(
            (
                name,
                FrozenFileSnapshot.capture(
                    absolute / name,
                    f"{label}/{name}",
                    authority_root=absolute,
                ),
            )
            for name in actual_names
        )
        members_after, identity_after = _directory_members(absolute, label)
        if members_after != members_before or identity_after != identity:
            raise Stage1WorkflowError(f"{label} snapshot directory changed during capture")
        for name, snapshot in files:
            snapshot.require_current(f"{label}/{name}")
        return cls(
            path=absolute,
            members=members_before,
            files=files,
            _identity=identity,
        )

    def file(self, name: str) -> FrozenFileSnapshot:
        for relative, snapshot in self.files:
            if relative == name:
                return snapshot
        raise Stage1WorkflowError(f"directory snapshot file is missing: {name}")

    def payloads(self) -> dict[str, bytes]:
        return {name: snapshot.payload for name, snapshot in self.files}

    def require_current(
        self,
        label: str = "directory",
        *,
        allowed_added_regular_files: tuple[str, ...] = (),
    ) -> None:
        additions = tuple(sorted(allowed_added_regular_files))
        _require_casefold_unique(additions, label)
        current_members, current_identity = _directory_members(self.path, label)
        expected_members = tuple(
            sorted(
                (*self.members, *((name, "regular_file") for name in additions)),
                key=lambda item: item[0],
            )
        )
        if current_members != expected_members:
            raise Stage1WorkflowError(f"{label} snapshot member set changed")
        if not additions and current_identity != self._identity:
            raise Stage1WorkflowError(f"{label} snapshot directory identity changed")
        for name, snapshot in self.files:
            snapshot.require_current(f"{label}/{name}")


def strict_json_object_from_bytes(payload: bytes, label: str) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8-sig"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise Stage1WorkflowError(f"{label} is invalid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise Stage1WorkflowError(f"{label} must be a JSON object")
    return value


def finite_tree(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def validate_run_id(run_id: str) -> None:
    basename = run_id.split(".", 1)[0].upper()
    if (
        _RUN_ID_PATTERN.fullmatch(run_id) is None
        or run_id.endswith((".", " "))
        or basename in _WINDOWS_RESERVED
    ):
        raise Stage1WorkflowError("run_id must be a Windows-safe non-reserved identifier")


def prepare_unique_run_root(run_root: Path) -> None:
    if run_root.exists():
        entries = tuple(run_root.iterdir())
        forbidden = sorted(path.name for path in entries if path.name in STAGE1_FORBIDDEN_ARTIFACTS)
        if forbidden:
            raise Stage1WorkflowError(
                f"forbidden approval artifacts already exist: {', '.join(forbidden)}"
            )
        if entries:
            raise Stage1WorkflowError(f"Stage 1 run root is non-empty: {run_root}")
        return
    run_root.mkdir(parents=True, exist_ok=False)


def read_object(path: Path, label: str) -> dict[str, object]:
    return strict_json_object_from_bytes(FrozenFileSnapshot.capture(path, label).payload, label)


def load_stage1_machine_config_bytes(
    payload_bytes: bytes,
) -> tuple[Stage1Config, dict[str, object]]:
    payload = strict_json_object_from_bytes(payload_bytes, "machine config")
    identity = payload.pop("execution_source_identity", None)
    if not isinstance(identity, dict):
        raise Stage1WorkflowError("machine config execution source identity is missing")
    try:
        config = Stage1Config.model_validate(payload)
    except ValidationError as exc:
        raise Stage1WorkflowError("machine config payload is invalid") from exc
    return config, identity


def load_stage1_machine_config(path: Path) -> tuple[Stage1Config, dict[str, object]]:
    snapshot = FrozenFileSnapshot.capture(path, "machine config")
    return load_stage1_machine_config_bytes(snapshot.payload)


def verify_manifest_entries_bytes(
    manifest_payload: bytes,
    machine_payloads: Mapping[str, bytes | FrozenFileSnapshot],
) -> None:
    manifest = strict_json_object_from_bytes(manifest_payload, "manifest")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or len(entries) != len(STAGE1_MACHINE_ARTIFACTS):
        raise Stage1WorkflowError("machine manifest is not exact")
    paths = [entry.get("path") for entry in entries if isinstance(entry, dict)]
    if len(paths) != len(entries) or set(paths) != set(STAGE1_MACHINE_ARTIFACTS):
        raise Stage1WorkflowError("machine manifest paths are not exact")
    _require_casefold_unique(tuple(str(path) for path in paths), "machine manifest")
    if set(machine_payloads) != set(STAGE1_MACHINE_ARTIFACTS):
        raise Stage1WorkflowError("frozen machine payload set is not exact")
    for entry in entries:
        assert isinstance(entry, dict)
        relative = entry.get("path")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise Stage1WorkflowError("machine manifest artifact path is unsafe")
        source = machine_payloads[relative]
        payload = source.payload if isinstance(source, FrozenFileSnapshot) else source
        if (
            entry.get("size_bytes") != len(payload)
            or entry.get("sha256") != hashlib.sha256(payload).hexdigest()
        ):
            raise Stage1WorkflowError(f"machine manifest artifact drift: {relative}")


def verify_manifest_entries(stage: Path) -> None:
    manifest = FrozenFileSnapshot.capture(stage / "manifest.json", "manifest")
    payloads = {
        name: FrozenFileSnapshot.capture(
            stage / name,
            f"machine artifact/{name}",
            authority_root=stage,
        )
        for name in STAGE1_MACHINE_ARTIFACTS
    }
    verify_manifest_entries_bytes(manifest.payload, payloads)
