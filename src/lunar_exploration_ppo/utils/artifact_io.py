from __future__ import annotations

import hashlib
import json
import os
import uuid
import warnings
from pathlib import Path
from typing import Any, Iterable

from lunar_exploration_ppo.utils.path_security import (
    DurableFileIdentity,
    DurableParentGuard,
    PathSecurityError,
    durable_fsync_directory,
    durable_makedirs,
    durable_publish_exclusive,
    durable_replace,
    durable_unlink,
    guarded_file_identity,
)


class ArtifactPathError(ValueError):
    """Raised when an artifact path violates the frozen path contract."""


class ArtifactStore:
    WARNING_PATH_CHARS = 180
    MAX_PATH_CHARS = 240

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self._validate_path(self.root)

    def resolve(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute():
            candidate = relative.resolve()
        else:
            candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ArtifactPathError(f"artifact path escapes output root: {candidate}") from exc
        self._validate_path(candidate)
        return candidate

    def write_json(self, relative_path: str | Path, value: Any) -> Path:
        return self.write_bytes(relative_path, self.canonical_json_bytes(value))

    def write_json_exclusive(self, relative_path: str | Path, value: Any) -> Path:
        return self.write_bytes_exclusive(relative_path, self.canonical_json_bytes(value))

    @staticmethod
    def canonical_json_bytes(value: Any) -> bytes:
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

    def write_bytes(self, relative_path: str | Path, payload: bytes) -> Path:
        destination = self.resolve(relative_path)
        self._atomic_write(destination, payload)
        return destination

    def write_bytes_exclusive(self, relative_path: str | Path, payload: bytes) -> Path:
        destination = self.resolve(relative_path)
        self._exclusive_write(destination, payload)
        return destination

    def write_checkpoint(self, relative_path: str | Path, payload: bytes) -> Path:
        return self.write_bytes(relative_path, payload)

    def append_jsonl(self, relative_path: str | Path, value: Any) -> Path:
        destination = self.resolve(relative_path)
        line = (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        self._make_parent(destination)
        created = not os.path.lexists(destination)
        with open(self._native_path(destination), "ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        if created:
            durable_fsync_directory(destination.parent)
        return destination

    def build_manifest(self, relative_paths: Iterable[str | Path]) -> dict[str, object]:
        artifacts: list[dict[str, object]] = []
        for relative_path in sorted((Path(path) for path in relative_paths), key=lambda path: path.as_posix()):
            path = self.resolve(relative_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            digest = hashlib.sha256()
            size_bytes = 0
            with open(self._native_path(path), "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size_bytes += len(chunk)
            artifacts.append(
                {
                    "path": path.relative_to(self.root).as_posix(),
                    "sha256": digest.hexdigest(),
                    "size_bytes": size_bytes,
                }
            )
        return {"schema_version": "sha256_manifest/v1", "artifacts": artifacts}

    def _atomic_write(self, destination: Path, payload: bytes) -> None:
        self._make_parent(destination)
        temporary = destination.parent / (
            f".{destination.name}.{uuid.uuid4().hex}.tmp"
        )
        temporary_identity: DurableFileIdentity | None = None
        try:
            temporary_identity = self._write_new_file(temporary, payload)
            durable_replace(
                temporary,
                destination,
                expected_source_identity=temporary_identity,
                replace_existing=True,
            )
            temporary_identity = None
        except BaseException as primary_error:
            self._cleanup_temporary(
                temporary,
                expected_identity=temporary_identity,
                primary_error=primary_error,
            )
            raise

    def _exclusive_write(self, destination: Path, payload: bytes) -> None:
        self._make_parent(destination)
        temporary = destination.parent / (
            f".{destination.name}.{uuid.uuid4().hex}.tmp"
        )
        temporary_identity: DurableFileIdentity | None = None
        try:
            temporary_identity = self._write_new_file(temporary, payload)
            durable_publish_exclusive(
                temporary,
                destination,
                expected_source_identity=temporary_identity,
            )
            temporary_identity = None
        except BaseException as primary_error:
            self._cleanup_temporary(
                temporary,
                expected_identity=temporary_identity,
                primary_error=primary_error,
            )
            raise

    def _write_new_file(
        self,
        path: Path,
        payload: bytes,
    ) -> DurableFileIdentity:
        identity: DurableFileIdentity | None = None
        try:
            with DurableParentGuard(path.parent) as parent_guard:
                descriptor = parent_guard.open_file(
                    path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                identity = guarded_file_identity(
                    descriptor,
                    path,
                    parent_guard=parent_guard,
                )
                primary_error: BaseException | None = None
                try:
                    self._write_all(
                        descriptor,
                        path,
                        payload,
                        parent_guard=parent_guard,
                    )
                    os.fsync(descriptor)
                    identity = guarded_file_identity(
                        descriptor,
                        path,
                        parent_guard=parent_guard,
                    )
                    if identity[3] != len(payload):
                        raise PathSecurityError(
                            "artifact temporary size changed while written"
                        )
                except BaseException as exc:
                    primary_error = exc
                    try:
                        identity = guarded_file_identity(
                            descriptor,
                            path,
                            parent_guard=parent_guard,
                        )
                    except BaseException as identity_error:
                        exc.add_note(
                            "suppressed failure while refreshing temporary identity: "
                            f"{identity_error}"
                        )
                    raise
                finally:
                    try:
                        os.close(descriptor)
                    except BaseException as close_error:
                        if primary_error is None:
                            raise
                        primary_error.add_note(
                            f"suppressed descriptor close failure: {close_error}"
                        )
        except BaseException as primary_error:
            self._cleanup_temporary(
                path,
                expected_identity=identity,
                primary_error=primary_error,
            )
            raise
        if identity is None:
            raise PathSecurityError("artifact temporary identity was not captured")
        return identity

    @staticmethod
    def _write_all(
        descriptor: int,
        path: Path,
        payload: bytes,
        *,
        parent_guard: DurableParentGuard,
    ) -> None:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("artifact descriptor write made no progress")
            written += count
            guarded_file_identity(
                descriptor,
                path,
                parent_guard=parent_guard,
            )

    @staticmethod
    def _cleanup_temporary(
        temporary: Path,
        *,
        expected_identity: DurableFileIdentity | None,
        primary_error: BaseException | None,
    ) -> None:
        if expected_identity is None or not os.path.lexists(temporary):
            return
        try:
            durable_unlink(temporary, expected_identity=expected_identity)
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                "suppressed identity-bound temporary cleanup failure: "
                f"{cleanup_error}"
            )

    def _make_parent(self, path: Path) -> None:
        durable_makedirs(path.parent)

    @classmethod
    def _validate_path(cls, path: Path) -> None:
        path_length = len(str(path))
        if path_length >= cls.MAX_PATH_CHARS:
            raise ArtifactPathError(f"artifact path length {path_length} reaches or exceeds 240 characters")
        if path_length > cls.WARNING_PATH_CHARS:
            warnings.warn(
                f"artifact path length {path_length} exceeds 180 characters",
                RuntimeWarning,
                stacklevel=3,
            )

    @staticmethod
    def _native_path(path: Path) -> Path:
        if os.name != "nt":
            return path
        raw = str(path.resolve())
        if raw.startswith("\\\\?\\"):
            return Path(raw)
        if raw.startswith("\\\\"):
            return Path("\\\\?\\UNC\\" + raw[2:])
        return Path("\\\\?\\" + raw)
