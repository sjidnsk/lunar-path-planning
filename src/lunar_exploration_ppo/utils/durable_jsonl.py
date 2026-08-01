"""Stage 6 可恢复 JSONL 与单写者租约基础设施。

安全边界是同一主机、同一 OS IPC 命名空间内的项目 runner，以及可测试的
leaf/parent 路径替换竞态。Linux 使用抽象 AF_UNIX guard、no-follow open 与
dir-fd 操作；Windows 使用全局命名 mutex，并在路径操作前后核对 descriptor/path
identity。这里不声称能够防御 root/管理员、恶意内核、隔离的 IPC namespace，
或可任意操纵本进程 handle/descriptor 的更高权限攻击者。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Callable

from lunar_exploration_ppo.utils.path_security import (
    DurableFileIdentity,
    DurableParentGuard,
    PathSecurityError,
    durable_fsync_directory,
    durable_makedirs,
    durable_publish_exclusive,
    durable_unlink,
    guarded_file_identity,
)

if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes
else:
    import fcntl


_PENDING_SCHEMA = "durable_jsonl_pending/v1"
_PENDING_KEYS = frozenset(
    {
        "prior_sha256",
        "prior_size",
        "row_base64",
        "row_sha256",
        "schema_version",
        "target_path",
    }
)


class DurableJsonlError(RuntimeError):
    """JSONL 事务状态无法被安全证明时抛出。"""


class RunLeaseError(RuntimeError):
    """run-level OS lease 无法安全获得或释放时抛出。"""


class _PathSafetyError(RuntimeError):
    pass


class _RunLeaseKernelGuard:
    def require_current(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _LinuxAbstractSocketGuard(_RunLeaseKernelGuard):
    def __init__(self, guard_socket: socket.socket, guard_name: str) -> None:
        self._socket: socket.socket | None = guard_socket
        self._guard_name = guard_name

    def require_current(self) -> None:
        guard_socket = self._socket
        try:
            current_name = (
                guard_socket.getsockname() if guard_socket is not None else None
            )
        except OSError as exc:
            raise RunLeaseError("run lease kernel guard is not current") from exc
        if (
            guard_socket is None
            or guard_socket.fileno() < 0
            or current_name != self._guard_name
        ):
            raise RunLeaseError("run lease kernel guard is not current")

    def close(self) -> None:
        guard_socket = self._socket
        if guard_socket is None:
            return
        self._socket = None
        guard_socket.close()


if os.name == "nt":

    class _WindowsNamedMutexGuard(_RunLeaseKernelGuard):
        def __init__(self, handle: int) -> None:
            self._handle: int | None = handle

        def require_current(self) -> None:
            handle = self._handle
            flags = wintypes.DWORD()
            if handle is None or not _GET_HANDLE_INFORMATION(
                handle,
                ctypes.byref(flags),
            ):
                raise RunLeaseError("run lease kernel guard is not current")

        def close(self) -> None:
            handle = self._handle
            if handle is None:
                return
            if not _CLOSE_HANDLE(handle):
                error_code = ctypes.get_last_error()
                raise RunLeaseError(
                    f"run lease kernel guard release failed ({error_code})"
                )
            self._handle = None


    _CREATE_MUTEX = ctypes.WinDLL("kernel32", use_last_error=True).CreateMutexW
    _CREATE_MUTEX.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
    _CREATE_MUTEX.restype = wintypes.HANDLE
    _CLOSE_HANDLE = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
    _CLOSE_HANDLE.argtypes = (wintypes.HANDLE,)
    _CLOSE_HANDLE.restype = wintypes.BOOL
    _GET_HANDLE_INFORMATION = ctypes.WinDLL(
        "kernel32", use_last_error=True
    ).GetHandleInformation
    _GET_HANDLE_INFORMATION.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    )
    _GET_HANDLE_INFORMATION.restype = wintypes.BOOL


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _require_canonical_jsonl_payload(payload: bytes) -> None:
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n"):
            raise DurableJsonlError(
                "durable JSONL has an unbound tail without a newline"
            )
        try:
            value = json.loads(
                line,
                parse_constant=_reject_json_constant,
            )
        except (TypeError, UnicodeError, ValueError) as exc:
            raise DurableJsonlError(
                f"durable JSONL row {line_number} is invalid JSON"
            ) from exc
        if _canonical_json(value) != line:
            raise DurableJsonlError(
                f"durable JSONL row {line_number} is not canonical"
            )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _target_path_id(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _run_lease_guard_digest(path: Path) -> str:
    return hashlib.sha256(os.fsencode(_target_path_id(path))).hexdigest()


def _acquire_run_lease_kernel_guard(path: Path) -> _RunLeaseKernelGuard:
    digest = _run_lease_guard_digest(path)
    if sys.platform.startswith("linux"):
        guard_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        guard_name = f"\0lunar-exploration-ppo-run-lease-{digest}"
        try:
            guard_socket.set_inheritable(False)
            guard_socket.bind(guard_name)
        except OSError as exc:
            guard_socket.close()
            raise RunLeaseError(
                "run lease kernel guard is already held or unavailable"
            ) from exc
        return _LinuxAbstractSocketGuard(guard_socket, guard_name)
    if os.name == "nt":
        mutex_name = f"Global\\LunarExplorationPpoRunLease-{digest}"
        ctypes.set_last_error(0)
        handle = _CREATE_MUTEX(None, False, mutex_name)
        error_code = ctypes.get_last_error()
        if not handle:
            raise RunLeaseError(
                f"run lease kernel guard is unavailable ({error_code})"
            )
        if error_code == 183:  # ERROR_ALREADY_EXISTS
            _CLOSE_HANDLE(handle)
            raise RunLeaseError("run lease kernel guard is already held")
        return _WindowsNamedMutexGuard(handle)
    raise RunLeaseError(
        "run lease kernel guard is unsupported on this platform; failing closed"
    )


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _require_plain_components(path: Path) -> None:
    parts = path.parts
    if not path.is_absolute() or not parts:
        raise _PathSafetyError("path is not absolute")
    current = Path(parts[0])
    missing_parent = False
    for index, part in enumerate(parts[1:], start=1):
        current = current / part
        exists = os.path.lexists(current)
        if missing_parent:
            if exists:
                raise _PathSafetyError("path chain is discontinuous")
            continue
        if not exists:
            missing_parent = True
            continue
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise _PathSafetyError("path metadata is unavailable") from exc
        if _is_link_or_reparse(metadata):
            raise _PathSafetyError("path contains a link or reparse point")
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise _PathSafetyError("path parent is not a directory")
        if index == len(parts) - 1 and not stat.S_ISREG(metadata.st_mode):
            raise _PathSafetyError("path leaf is not a regular file")


_DirectoryIdentity = tuple[int, int]
_FileIdentity = tuple[int, int, int, int, int]
_FileVersionMetadata = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class _CommittedVersionBinding:
    payload: bytes
    object_identity: _FileIdentity | None
    size_bytes: int
    sha256: str
    mtime_ns: int | None
    ctime_ns: int | None


@dataclass(frozen=True, slots=True)
class _RecoveryResult:
    recovered: bool
    committed_binding: _CommittedVersionBinding


def _metadata_identity(metadata: os.stat_result) -> _DirectoryIdentity:
    return int(metadata.st_dev), int(metadata.st_ino)


def _require_single_plain_file_metadata(metadata: os.stat_result) -> None:
    if not stat.S_ISREG(metadata.st_mode) or _is_link_or_reparse(metadata):
        raise _PathSafetyError("path leaf is not a plain regular file")
    if int(metadata.st_nlink) != 1:
        raise _PathSafetyError("path leaf hard link count must equal one")


def _file_identity(metadata: os.stat_result) -> _FileIdentity:
    _require_single_plain_file_metadata(metadata)
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(stat.S_IFMT(metadata.st_mode)),
        int(getattr(metadata, "st_file_attributes", 0)),
        int(metadata.st_nlink),
    )


def _file_version_metadata(metadata: os.stat_result) -> _FileVersionMetadata:
    return (
        int(metadata.st_size),
        int(getattr(metadata, "st_mtime_ns", 0)),
        int(getattr(metadata, "st_ctime_ns", 0)),
    )


def _plain_directory_chain(
    path: Path,
) -> tuple[tuple[Path, _DirectoryIdentity], ...]:
    parts = path.parts
    if not path.is_absolute() or not parts:
        raise _PathSafetyError("directory path is not absolute")
    current = Path(parts[0])
    entries: list[tuple[Path, _DirectoryIdentity]] = []
    for part in (None, *parts[1:]):
        if part is not None:
            current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise _PathSafetyError("directory path metadata is unavailable") from exc
        if _is_link_or_reparse(metadata):
            raise _PathSafetyError("directory path contains a link or reparse point")
        if not stat.S_ISDIR(metadata.st_mode):
            raise _PathSafetyError("directory path component is not a directory")
        entries.append((current, _metadata_identity(metadata)))
    return tuple(entries)


class _ParentPathGuard:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._chain = _plain_directory_chain(path)
        try:
            self._guard = DurableParentGuard(path)
        except (OSError, PathSecurityError) as exc:
            raise _PathSafetyError(
                "parent directory durability guard is unavailable"
            ) from exc
        try:
            self.revalidate()
        except BaseException:
            self.close()
            raise

    def revalidate(self) -> None:
        if _plain_directory_chain(self.path) != self._chain:
            raise _PathSafetyError("parent directory identity drifted")
        try:
            self._guard.revalidate()
        except (OSError, PathSecurityError) as exc:
            raise _PathSafetyError("parent durability guard drifted") from exc

    def close(self) -> None:
        try:
            self._guard.close()
        except (OSError, PathSecurityError) as exc:
            raise _PathSafetyError("parent durability guard close failed") from exc

    def fsync(self) -> None:
        try:
            self._guard.flush()
        except (OSError, PathSecurityError) as exc:
            raise _PathSafetyError("parent directory durability flush failed") from exc

    @property
    def descriptor(self) -> int | None:
        return self._guard.descriptor

    def open_file(self, path: Path, flags: int, mode: int = 0o600) -> int:
        try:
            return self._guard.open_file(path, flags, mode)
        except (OSError, PathSecurityError) as exc:
            raise _PathSafetyError("guarded leaf open failed") from exc

    def durable_file_identity(
        self,
        descriptor: int,
        path: Path,
    ) -> DurableFileIdentity:
        try:
            return guarded_file_identity(
                descriptor,
                path,
                parent_guard=self._guard,
            )
        except (OSError, PathSecurityError) as exc:
            raise _PathSafetyError("durable file identity is unavailable") from exc


def _close_parent_guard_preserving(parent_guard: _ParentPathGuard) -> None:
    primary_error = sys.exception()
    try:
        parent_guard.close()
    except BaseException as close_error:
        if primary_error is None:
            raise
        primary_error.add_note(
            f"suppressed parent guard close failure: {close_error}"
        )


def _close_descriptor_preserving(descriptor: int) -> None:
    primary_error = sys.exception()
    try:
        os.close(descriptor)
    except BaseException as close_error:
        if primary_error is None:
            raise
        primary_error.add_note(f"suppressed descriptor close failure: {close_error}")


def _require_descriptor_path_identity(descriptor: int, path: Path) -> _FileIdentity:
    try:
        descriptor_metadata = os.fstat(descriptor)
        path_metadata = path.lstat()
    except OSError as exc:
        raise _PathSafetyError("file descriptor path identity is unavailable") from exc
    _require_single_plain_file_metadata(descriptor_metadata)
    _require_single_plain_file_metadata(path_metadata)
    descriptor_identity = _file_identity(descriptor_metadata)
    if descriptor_identity != _file_identity(path_metadata):
        raise _PathSafetyError("file descriptor and path identity drifted")
    return descriptor_identity


def _open_plain_file(
    path: Path,
    flags: int,
    mode: int = 0o600,
    *,
    parent_guard: _ParentPathGuard,
) -> int:
    _require_plain_components(path)
    descriptor: int | None = None
    try:
        descriptor = parent_guard.open_file(path, flags, mode)
        _require_plain_components(path)
        _require_descriptor_path_identity(descriptor, path)
    except (OSError, _PathSafetyError) as exc:
        if descriptor is not None:
            _close_descriptor_preserving(descriptor)
        if isinstance(exc, _PathSafetyError):
            raise
        raise _PathSafetyError("plain file open failed") from exc
    return descriptor


def _plain_file_identity(
    path: Path,
    *,
    parent_guard: _ParentPathGuard,
) -> _FileIdentity:
    descriptor = _open_plain_file(
        path,
        os.O_RDONLY,
        parent_guard=parent_guard,
    )
    try:
        return _require_descriptor_path_identity(descriptor, path)
    finally:
        _close_descriptor_preserving(descriptor)


def _read_plain_version(
    path: Path,
    *,
    parent_guard: _ParentPathGuard,
) -> tuple[bytes, _FileIdentity, _FileVersionMetadata]:
    descriptor = _open_plain_file(
        path,
        os.O_RDONLY,
        parent_guard=parent_guard,
    )
    try:
        descriptor_before = os.fstat(descriptor)
        identity = _require_descriptor_path_identity(descriptor, path)
        version = _file_version_metadata(descriptor_before)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        descriptor_after = os.fstat(descriptor)
        final_identity = _require_descriptor_path_identity(descriptor, path)
        final_version = _file_version_metadata(descriptor_after)
        if (
            final_identity != identity
            or final_version != version
            or len(payload) != version[0]
        ):
            raise _PathSafetyError("file identity or content version changed while reading")
        return payload, identity, version
    finally:
        _close_descriptor_preserving(descriptor)


def _read_plain_bytes(
    path: Path,
    *,
    parent_guard: _ParentPathGuard,
) -> tuple[bytes, _FileIdentity]:
    payload, identity, _version = _read_plain_version(
        path,
        parent_guard=parent_guard,
    )
    return payload, identity


def _replace_plain_file(
    source: Path,
    destination: Path,
    *,
    expected_identity: _FileIdentity,
    parent_guard: _ParentPathGuard,
) -> None:
    if source.parent != parent_guard.path or destination.parent != parent_guard.path:
        raise _PathSafetyError("replace paths escaped the guarded parent")
    parent_guard.revalidate()
    if os.path.lexists(destination):
        raise _PathSafetyError("replace destination already exists")
    descriptor = _open_plain_file(
        source,
        os.O_RDONLY,
        parent_guard=parent_guard,
    )
    try:
        if _require_descriptor_path_identity(descriptor, source) != expected_identity:
            raise _PathSafetyError("replace source identity changed")
        durable_identity = parent_guard.durable_file_identity(descriptor, source)
        parent_guard.revalidate()
        descriptor_to_close = descriptor
        descriptor = -1
        _close_descriptor_preserving(descriptor_to_close)
        if (
            _plain_file_identity(source, parent_guard=parent_guard)
            != expected_identity
        ):
            raise _PathSafetyError("replace source identity changed")
        try:
            durable_publish_exclusive(
                source,
                destination,
                expected_source_identity=durable_identity,
            )
        except (OSError, PathSecurityError) as exc:
            raise _PathSafetyError("durable replace publication failed") from exc
        parent_guard.revalidate()
        if (
            _plain_file_identity(destination, parent_guard=parent_guard)
            != expected_identity
        ):
            raise _PathSafetyError("replace destination identity changed")
        if os.path.lexists(source):
            raise _PathSafetyError("replace source survived publication")
    finally:
        if descriptor >= 0:
            _close_descriptor_preserving(descriptor)


def _unlink_plain_file(
    path: Path,
    *,
    expected_identity: _FileIdentity,
    parent_guard: _ParentPathGuard,
) -> None:
    if path.parent != parent_guard.path:
        raise _PathSafetyError("unlink path escaped the guarded parent")
    parent_guard.revalidate()
    descriptor = _open_plain_file(
        path,
        os.O_RDONLY,
        parent_guard=parent_guard,
    )
    try:
        if _require_descriptor_path_identity(descriptor, path) != expected_identity:
            raise _PathSafetyError("unlink source identity changed")
        durable_identity = parent_guard.durable_file_identity(descriptor, path)
        parent_guard.revalidate()
        descriptor_to_close = descriptor
        descriptor = -1
        _close_descriptor_preserving(descriptor_to_close)
        if (
            _plain_file_identity(path, parent_guard=parent_guard)
            != expected_identity
        ):
            raise _PathSafetyError("unlink source identity changed")
        try:
            durable_unlink(path, expected_identity=durable_identity)
        except (OSError, PathSecurityError) as exc:
            raise _PathSafetyError("durable unlink failed") from exc
        parent_guard.revalidate()
        if os.path.lexists(path):
            raise _PathSafetyError("unlink source survived removal")
    finally:
        if descriptor >= 0:
            _close_descriptor_preserving(descriptor)


def _sha256_file(
    path: Path,
    *,
    parent_guard: _ParentPathGuard,
) -> tuple[int, str, _FileIdentity | None]:
    parent_guard.revalidate()
    digest = hashlib.sha256()
    size = 0
    if os.path.lexists(path):
        descriptor = _open_plain_file(
            path,
            os.O_RDONLY,
            parent_guard=parent_guard,
        )
        try:
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            identity = _require_descriptor_path_identity(descriptor, path)
        finally:
            _close_descriptor_preserving(descriptor)
        parent_guard.revalidate()
        return size, digest.hexdigest(), identity
    _require_plain_components(path)
    parent_guard.revalidate()
    return size, digest.hexdigest(), None


def _fsync_directory(path: Path) -> None:
    try:
        durable_fsync_directory(path)
    except (OSError, PathSecurityError) as exc:
        raise _PathSafetyError("directory durability flush failed") from exc


class DurableJsonl:
    """保留真实 append 语义的 JSONL 写入器。"""

    def __init__(
        self,
        path: str | Path,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.path = Path(os.path.abspath(Path(path).expanduser()))
        self.pending_path = Path(f"{self.path}.pending")
        self.pending_temp_path = Path(f"{self.pending_path}.tmp")
        self._fault_injector = fault_injector

    def recover(self) -> bool:
        self._require_safe_paths()
        if not os.path.lexists(self.path.parent):
            return False
        try:
            parent_guard = _ParentPathGuard(self.path.parent)
        except _PathSafetyError as exc:
            raise DurableJsonlError(
                "durable JSONL parent identity is unsafe or changed"
            ) from exc
        try:
            result = self._recover_with_parent_guard(parent_guard=parent_guard)
            return result.recovered
        except _PathSafetyError as exc:
            raise DurableJsonlError(
                "durable JSONL parent identity changed during recovery"
            ) from exc
        finally:
            _close_parent_guard_preserving(parent_guard)

    def recover_and_snapshot(self) -> bytes:
        """Recover and return one version-bound canonical committed snapshot."""

        self._require_safe_paths()
        if not os.path.lexists(self.path.parent):
            self._inject("after_recovery_before_snapshot")
            self._require_safe_paths()
            if any(
                os.path.lexists(candidate)
                for candidate in (
                    self.path.parent,
                    self.path,
                    self.pending_path,
                    self.pending_temp_path,
                )
            ):
                raise DurableJsonlError(
                    "durable JSONL missing parent changed before snapshot"
                )
            return b""
        try:
            parent_guard = _ParentPathGuard(self.path.parent)
        except _PathSafetyError as exc:
            raise DurableJsonlError(
                "durable JSONL parent identity is unsafe or changed"
            ) from exc
        try:
            recovery = self._recover_with_parent_guard(parent_guard=parent_guard)
            recovered_binding = recovery.committed_binding
            self._inject("after_recovery_before_snapshot")
            parent_guard.revalidate()
            snapshot_binding = self._committed_version_binding(
                parent_guard=parent_guard
            )
            if snapshot_binding != recovered_binding:
                raise DurableJsonlError(
                    "durable JSONL committed content version or identity changed"
                )
            return snapshot_binding.payload
        except _PathSafetyError as exc:
            raise DurableJsonlError(
                "durable JSONL parent or target identity changed during snapshot"
            ) from exc
        finally:
            _close_parent_guard_preserving(parent_guard)

    def _recover_with_parent_guard(
        self,
        *,
        parent_guard: _ParentPathGuard,
    ) -> _RecoveryResult:
        self._require_safe_paths()
        parent_guard.revalidate()
        self._recover_pending_temp(parent_guard=parent_guard)
        parent_guard.revalidate()
        if not os.path.lexists(self.pending_path):
            binding = self._committed_version_binding(
                parent_guard=parent_guard
            )
            return _RecoveryResult(
                recovered=False,
                committed_binding=binding,
            )
        pending, row, pending_identity = self._read_pending(
            parent_guard=parent_guard
        )
        parent_guard.revalidate()
        prior_size = pending["prior_size"]
        prefix_sha256, tail, target_identity = self._read_prefix_and_tail(
            prior_size,
            parent_guard=parent_guard,
        )
        parent_guard.revalidate()
        if prefix_sha256 != pending["prior_sha256"]:
            raise DurableJsonlError("durable JSONL historical prefix changed")
        self._inject("before_recovery_commit")
        parent_guard.revalidate()
        if tail == row:
            self._fsync_target(
                expected_identity=target_identity,
                parent_guard=parent_guard,
            )
        elif row.startswith(tail):
            if target_identity is not None:
                self._truncate_target(
                    prior_size,
                    expected_identity=target_identity,
                    expected_prefix_sha256=pending["prior_sha256"],
                    expected_tail=tail,
                    parent_guard=parent_guard,
                )
            self._append_row(
                row,
                expected_identity=target_identity,
                parent_guard=parent_guard,
            )
        else:
            raise DurableJsonlError(
                "durable JSONL has an unknown pending tail"
            )
        self._require_safe_paths()
        parent_guard.revalidate()
        _unlink_plain_file(
            self.pending_path,
            expected_identity=pending_identity,
            parent_guard=parent_guard,
        )
        parent_guard.revalidate()
        binding = self._committed_version_binding(
            parent_guard=parent_guard
        )
        return _RecoveryResult(
            recovered=True,
            committed_binding=binding,
        )

    def append(self, value: Any) -> Path:
        self._require_safe_paths()
        try:
            durable_makedirs(self.path.parent)
        except (OSError, PathSecurityError) as exc:
            raise DurableJsonlError(
                "durable JSONL path is unsafe (link/reparse or invalid parent)"
            ) from exc
        try:
            parent_guard = _ParentPathGuard(self.path.parent)
        except _PathSafetyError as exc:
            raise DurableJsonlError(
                "durable JSONL parent identity is unsafe or changed"
            ) from exc
        try:
            self._require_safe_paths()
            parent_guard.revalidate()
            if os.path.lexists(self.pending_path):
                raise DurableJsonlError(
                    "durable JSONL has an uncommitted pending append"
                )
            boundary_identity = self._require_committed_boundary(
                parent_guard=parent_guard
            )
            parent_guard.revalidate()
            row = _canonical_json(value)
            try:
                prior_size, prior_sha256, target_identity = _sha256_file(
                    self.path,
                    parent_guard=parent_guard,
                )
            except _PathSafetyError as exc:
                raise DurableJsonlError(
                    "durable JSONL target identity changed while hashing"
                ) from exc
            if boundary_identity != target_identity:
                raise DurableJsonlError(
                    "durable JSONL target identity changed before pending publication"
                )
            parent_guard.revalidate()
            pending = {
                "prior_sha256": prior_sha256,
                "prior_size": prior_size,
                "row_base64": base64.b64encode(row).decode("ascii"),
                "row_sha256": hashlib.sha256(row).hexdigest(),
                "schema_version": _PENDING_SCHEMA,
                "target_path": _target_path_id(self.path),
            }
            pending_identity = self._write_pending(
                _canonical_json(pending),
                parent_guard=parent_guard,
            )
            parent_guard.revalidate()
            self._inject("after_pending")
            try:
                parent_guard.revalidate()
            except _PathSafetyError as exc:
                raise DurableJsonlError(
                    "durable JSONL parent identity changed before append"
                ) from exc
            self._append_row(
                row,
                expected_identity=target_identity,
                parent_guard=parent_guard,
            )
            parent_guard.revalidate()
            self._inject("before_pending_remove")
            self._require_safe_paths()
            parent_guard.revalidate()
            _unlink_plain_file(
                self.pending_path,
                expected_identity=pending_identity,
                parent_guard=parent_guard,
            )
            return self.path
        except _PathSafetyError as exc:
            raise DurableJsonlError(
                "durable JSONL parent identity changed during append"
            ) from exc
        finally:
            _close_parent_guard_preserving(parent_guard)

    def _write_pending(
        self,
        payload: bytes,
        *,
        parent_guard: _ParentPathGuard,
    ) -> _FileIdentity:
        self._require_safe_paths()
        parent_guard.revalidate()
        if os.path.lexists(self.pending_temp_path):
            raise DurableJsonlError(
                "durable JSONL has an uncommitted pending temporary"
            )
        try:
            descriptor = _open_plain_file(
                self.pending_temp_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                parent_guard=parent_guard,
            )
            try:
                temp_identity = _require_descriptor_path_identity(
                    descriptor,
                    self.pending_temp_path,
                )
                offset = 0
                while offset < len(payload):
                    written = os.write(descriptor, payload[offset:])
                    if written <= 0:
                        raise DurableJsonlError(
                            "durable JSONL pending write was incomplete"
                        )
                    offset += written
                    _require_descriptor_path_identity(
                        descriptor,
                        self.pending_temp_path,
                    )
                os.fsync(descriptor)
                _require_descriptor_path_identity(
                    descriptor,
                    self.pending_temp_path,
                )
            finally:
                _close_descriptor_preserving(descriptor)
            parent_guard.revalidate()
            self._inject("before_pending_replace")
            parent_guard.revalidate()
            source_identity = _plain_file_identity(
                self.pending_temp_path,
                parent_guard=parent_guard,
            )
            if source_identity != temp_identity:
                raise DurableJsonlError(
                    "durable JSONL pending temporary identity changed"
                )
            _replace_plain_file(
                self.pending_temp_path,
                self.pending_path,
                expected_identity=temp_identity,
                parent_guard=parent_guard,
            )
            return temp_identity
        except DurableJsonlError:
            raise
        except (OSError, _PathSafetyError) as exc:
            raise DurableJsonlError(
                "durable JSONL pending open/write/replace identity changed"
            ) from exc

    def _recover_pending_temp(
        self,
        *,
        parent_guard: _ParentPathGuard,
    ) -> None:
        parent_guard.revalidate()
        if not os.path.lexists(self.pending_temp_path):
            return
        if os.path.lexists(self.pending_path):
            raise DurableJsonlError(
                "durable JSONL has both pending and pending temporary records"
            )
        try:
            temp_identity = _plain_file_identity(
                self.pending_temp_path,
                parent_guard=parent_guard,
            )
        except _PathSafetyError as exc:
            raise DurableJsonlError(
                "durable JSONL pending temporary identity is unsafe"
            ) from exc
        try:
            pending, row, read_identity = self._read_pending(
                self.pending_temp_path,
                parent_guard=parent_guard,
            )
        except DurableJsonlError:
            self._require_committed_boundary(parent_guard=parent_guard)
            parent_guard.revalidate()
            _unlink_plain_file(
                self.pending_temp_path,
                expected_identity=temp_identity,
                parent_guard=parent_guard,
            )
            return
        if read_identity != temp_identity:
            raise DurableJsonlError(
                "durable JSONL pending temporary identity changed while reading"
            )
        prefix_sha256, tail, _ = self._read_prefix_and_tail(
            pending["prior_size"],
            parent_guard=parent_guard,
        )
        if prefix_sha256 != pending["prior_sha256"] or tail:
            raise DurableJsonlError(
                "durable JSONL pending temporary does not match committed history"
            )
        if hashlib.sha256(row).hexdigest() != pending["row_sha256"]:
            raise DurableJsonlError("durable JSONL pending temporary row drifted")
        _replace_plain_file(
            self.pending_temp_path,
            self.pending_path,
            expected_identity=temp_identity,
            parent_guard=parent_guard,
        )

    def _inject(self, phase: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(phase)

    def _read_pending(
        self,
        source: Path | None = None,
        *,
        parent_guard: _ParentPathGuard,
    ) -> tuple[dict[str, Any], bytes, _FileIdentity]:
        self._require_safe_paths()
        parent_guard.revalidate()
        pending_source = self.pending_path if source is None else source
        try:
            payload, identity = _read_plain_bytes(
                pending_source,
                parent_guard=parent_guard,
            )
            pending = json.loads(
                payload,
                parse_constant=_reject_json_constant,
            )
            if not isinstance(pending, dict) or set(pending) != _PENDING_KEYS:
                raise ValueError("pending keys do not match the schema")
            if _canonical_json(pending) != payload:
                raise ValueError("pending record is not canonical JSON")
            if pending["schema_version"] != _PENDING_SCHEMA:
                raise ValueError("pending schema version is unsupported")
            if pending["target_path"] != _target_path_id(self.path):
                raise ValueError("pending target path does not match")
            if type(pending["prior_size"]) is not int or pending["prior_size"] < 0:
                raise ValueError("pending prior size is invalid")
            if not _is_sha256(pending["prior_sha256"]):
                raise ValueError("pending prior hash is invalid")
            if not _is_sha256(pending["row_sha256"]):
                raise ValueError("pending row hash is invalid")
            if not isinstance(pending["row_base64"], str):
                raise ValueError("pending row bytes are invalid")
            row = base64.b64decode(pending["row_base64"], validate=True)
            if hashlib.sha256(row).hexdigest() != pending["row_sha256"]:
                raise ValueError("pending row hash does not match")
            if not row.endswith(b"\n") or row.count(b"\n") != 1:
                raise ValueError("pending row is not a single JSONL line")
            row_value = json.loads(
                row,
                parse_constant=_reject_json_constant,
            )
            if _canonical_json(row_value) != row:
                raise ValueError("pending row is not canonical JSON")
            parent_guard.revalidate()
        except DurableJsonlError:
            raise
        except (
            KeyError,
            OSError,
            _PathSafetyError,
            TypeError,
            UnicodeError,
            ValueError,
        ) as exc:
            raise DurableJsonlError(
                "invalid durable JSONL pending record"
            ) from exc
        return pending, row, identity

    def _read_prefix_and_tail(
        self,
        prior_size: int,
        *,
        parent_guard: _ParentPathGuard,
    ) -> tuple[str, bytes, _FileIdentity | None]:
        self._require_safe_paths()
        parent_guard.revalidate()
        digest = hashlib.sha256()
        if not os.path.lexists(self.path):
            if prior_size != 0:
                raise DurableJsonlError(
                    "durable JSONL historical prefix is missing"
                )
            parent_guard.revalidate()
            return digest.hexdigest(), b"", None
        try:
            descriptor = _open_plain_file(
                self.path,
                os.O_RDONLY,
                parent_guard=parent_guard,
            )
        except _PathSafetyError as exc:
            raise DurableJsonlError(
                "durable JSONL target identity changed while reading recovery state"
            ) from exc
        try:
            identity = _require_descriptor_path_identity(descriptor, self.path)
            file_size = os.fstat(descriptor).st_size
            if file_size < prior_size:
                raise DurableJsonlError(
                    "durable JSONL historical prefix is shorter than pending state"
                )
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                remaining = prior_size
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise DurableJsonlError(
                            "durable JSONL historical prefix became unavailable"
                        )
                    digest.update(chunk)
                    remaining -= len(chunk)
                tail = stream.read()
            _require_descriptor_path_identity(descriptor, self.path)
            parent_guard.revalidate()
        except _PathSafetyError as exc:
            raise DurableJsonlError(
                "durable JSONL target identity changed while reading recovery state"
            ) from exc
        finally:
            _close_descriptor_preserving(descriptor)
        return digest.hexdigest(), tail, identity

    def _require_committed_boundary(
        self,
        *,
        parent_guard: _ParentPathGuard,
    ) -> _FileIdentity | None:
        self._require_safe_paths()
        parent_guard.revalidate()
        if not os.path.lexists(self.path):
            parent_guard.revalidate()
            return None
        try:
            payload, identity, _version = _read_plain_version(
                self.path,
                parent_guard=parent_guard,
            )
        except _PathSafetyError as exc:
            raise DurableJsonlError(
                "durable JSONL target identity changed while validating history"
            ) from exc
        _require_canonical_jsonl_payload(payload)
        parent_guard.revalidate()
        return identity

    def _require_snapshot_pending_absent(
        self,
        *,
        parent_guard: _ParentPathGuard,
    ) -> None:
        self._require_safe_paths()
        parent_guard.revalidate()
        if os.path.lexists(self.pending_path) or os.path.lexists(
            self.pending_temp_path
        ):
            raise DurableJsonlError(
                "durable JSONL pending state changed during committed snapshot"
            )
        parent_guard.revalidate()

    def _committed_version_binding(
        self,
        *,
        parent_guard: _ParentPathGuard,
    ) -> _CommittedVersionBinding:
        self._require_snapshot_pending_absent(parent_guard=parent_guard)
        if not os.path.lexists(self.path):
            _require_plain_components(self.path)
            parent_guard.revalidate()
            if os.path.lexists(self.path):
                raise DurableJsonlError(
                    "durable JSONL target changed while binding missing snapshot"
                )
            self._require_snapshot_pending_absent(parent_guard=parent_guard)
            return _CommittedVersionBinding(
                payload=b"",
                object_identity=None,
                size_bytes=0,
                sha256=hashlib.sha256(b"").hexdigest(),
                mtime_ns=None,
                ctime_ns=None,
            )
        try:
            payload, identity, version = _read_plain_version(
                self.path,
                parent_guard=parent_guard,
            )
        except _PathSafetyError as exc:
            raise DurableJsonlError(
                "durable JSONL target identity changed while binding snapshot"
            ) from exc
        _require_canonical_jsonl_payload(payload)
        self._require_snapshot_pending_absent(parent_guard=parent_guard)
        return _CommittedVersionBinding(
            payload=payload,
            object_identity=identity,
            size_bytes=version[0],
            sha256=hashlib.sha256(payload).hexdigest(),
            mtime_ns=version[1],
            ctime_ns=version[2],
        )

    def _require_safe_paths(self) -> None:
        try:
            _require_plain_components(self.path)
            _require_plain_components(self.pending_path)
            _require_plain_components(self.pending_temp_path)
        except _PathSafetyError as exc:
            raise DurableJsonlError(
                "durable JSONL path is unsafe (link/reparse or non-regular path)"
            ) from exc

    def _fsync_target(
        self,
        *,
        expected_identity: _FileIdentity | None,
        parent_guard: _ParentPathGuard,
    ) -> None:
        self._require_safe_paths()
        parent_guard.revalidate()
        if expected_identity is None:
            raise DurableJsonlError("durable JSONL target disappeared before fsync")
        try:
            descriptor = _open_plain_file(
                self.path,
                os.O_RDWR,
                parent_guard=parent_guard,
            )
        except _PathSafetyError as exc:
            raise DurableJsonlError(
                "durable JSONL target identity changed before fsync"
            ) from exc
        try:
            if (
                _require_descriptor_path_identity(descriptor, self.path)
                != expected_identity
            ):
                raise DurableJsonlError(
                    "durable JSONL target identity changed before fsync"
                )
            os.fsync(descriptor)
            _require_descriptor_path_identity(descriptor, self.path)
            parent_guard.revalidate()
        except _PathSafetyError as exc:
            raise DurableJsonlError(
                "durable JSONL target identity changed during fsync"
            ) from exc
        finally:
            _close_descriptor_preserving(descriptor)

    def _truncate_target(
        self,
        size: int,
        *,
        expected_identity: _FileIdentity,
        expected_prefix_sha256: str,
        expected_tail: bytes,
        parent_guard: _ParentPathGuard,
    ) -> None:
        parent_guard.revalidate()
        try:
            descriptor = _open_plain_file(
                self.path,
                os.O_RDWR,
                parent_guard=parent_guard,
            )
        except _PathSafetyError as exc:
            raise DurableJsonlError(
                "durable JSONL target identity changed before tail repair"
            ) from exc
        try:
            if (
                _require_descriptor_path_identity(descriptor, self.path)
                != expected_identity
            ):
                raise DurableJsonlError(
                    "durable JSONL target identity changed before tail repair"
                )
            self._inject("before_torn_tail_repair")
            parent_guard.revalidate()
            os.lseek(descriptor, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            remaining = size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise DurableJsonlError(
                        "durable JSONL prefix changed before tail repair"
                    )
                digest.update(chunk)
                remaining -= len(chunk)
            current_tail_chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                current_tail_chunks.append(chunk)
            if digest.hexdigest() != expected_prefix_sha256:
                raise DurableJsonlError(
                    "durable JSONL prefix changed before tail repair"
                )
            if b"".join(current_tail_chunks) != expected_tail:
                raise DurableJsonlError(
                    "durable JSONL torn tail changed before repair"
                )
            os.ftruncate(descriptor, size)
            _require_descriptor_path_identity(descriptor, self.path)
            os.fsync(descriptor)
            _require_descriptor_path_identity(descriptor, self.path)
            parent_guard.revalidate()
        except _PathSafetyError as exc:
            raise DurableJsonlError(
                "durable JSONL target identity changed during tail repair"
            ) from exc
        finally:
            _close_descriptor_preserving(descriptor)

    def _append_row(
        self,
        row: bytes,
        *,
        expected_identity: _FileIdentity | None = None,
        parent_guard: _ParentPathGuard,
    ) -> None:
        self._require_safe_paths()
        parent_guard.revalidate()
        split = max(1, len(row) // 2)
        flags = os.O_WRONLY | os.O_APPEND
        if expected_identity is None:
            flags |= os.O_CREAT | os.O_EXCL
        try:
            descriptor = _open_plain_file(
                self.path,
                flags,
                parent_guard=parent_guard,
            )
        except _PathSafetyError as exc:
            raise DurableJsonlError(
                "durable JSONL target identity changed before append"
            ) from exc
        try:
            opened_identity = _require_descriptor_path_identity(
                descriptor,
                self.path,
            )
            if expected_identity is not None and opened_identity != expected_identity:
                raise DurableJsonlError(
                    "durable JSONL target identity changed before append"
                )
            first_written = os.write(descriptor, row[:split])
            if first_written != split:
                raise DurableJsonlError(
                    "durable JSONL first append chunk was incomplete"
                )
            _require_descriptor_path_identity(descriptor, self.path)
            self._inject("mid_append")
            parent_guard.revalidate()
            second_written = os.write(descriptor, row[split:])
            if second_written != len(row) - split:
                raise DurableJsonlError(
                    "durable JSONL second append chunk was incomplete"
                )
            _require_descriptor_path_identity(descriptor, self.path)
            self._inject("after_append_write")
            parent_guard.revalidate()
            self._inject("after_append_flush")
            parent_guard.revalidate()
            os.fsync(descriptor)
            self._inject("after_append_fsync")
            parent_guard.revalidate()
            _require_descriptor_path_identity(descriptor, self.path)
            if expected_identity is None:
                parent_guard.fsync()
        except _PathSafetyError as exc:
            raise DurableJsonlError(
                "durable JSONL target identity changed during append"
            ) from exc
        finally:
            _close_descriptor_preserving(descriptor)


class RunLease:
    """使用持久锁文件承载、由 OS 自动释放的非阻塞单写者租约。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(os.path.abspath(Path(path).expanduser()))
        self._descriptor: int | None = None
        self._kernel_guard: _RunLeaseKernelGuard | None = None
        self._parent_guard: _ParentPathGuard | None = None

    def __enter__(self) -> RunLease:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, traceback
        try:
            self.release()
        except BaseException as release_error:
            if exc_value is None:
                raise
            exc_value.add_note(f"suppressed run lease release failure: {release_error}")

    def acquire(self) -> None:
        if (
            self._descriptor is not None
            or self._kernel_guard is not None
            or self._parent_guard is not None
        ):
            raise RunLeaseError("run lease is already held by this handle")
        self._require_safe_path()
        kernel_guard = _acquire_run_lease_kernel_guard(self.path)
        self._kernel_guard = kernel_guard
        try:
            durable_makedirs(self.path.parent)
        except (OSError, PathSecurityError) as exc:
            self._cleanup_failed_acquire(None, kernel_guard)
            raise RunLeaseError(
                "run lease path is unsafe (link/reparse or invalid parent)"
            ) from exc
        descriptor: int | None = None
        parent_guard: _ParentPathGuard | None = None
        try:
            self._require_safe_path()
            created = not os.path.lexists(self.path)
            parent_guard = _ParentPathGuard(self.path.parent)
            flags = os.O_RDWR | os.O_CREAT
            if created:
                flags |= os.O_EXCL
            descriptor = _open_plain_file(
                self.path,
                flags,
                parent_guard=parent_guard,
            )
            self._require_safe_path()
            metadata = os.fstat(descriptor)
            try:
                _require_single_plain_file_metadata(metadata)
            except _PathSafetyError as exc:
                raise RunLeaseError(
                    "run lease path hard link count must equal one"
                ) from exc
            try:
                _require_descriptor_path_identity(descriptor, self.path)
            except _PathSafetyError as exc:
                raise RunLeaseError("run lease path identity changed") from exc
            if metadata.st_size == 0:
                os.lseek(descriptor, 0, os.SEEK_SET)
                if os.write(descriptor, b"\0") != 1:
                    raise RunLeaseError("run lease sentinel write was incomplete")
                try:
                    _require_descriptor_path_identity(descriptor, self.path)
                except _PathSafetyError as exc:
                    raise RunLeaseError("run lease path identity changed") from exc
                os.fsync(descriptor)
                parent_guard.fsync()
            os.lseek(descriptor, 0, os.SEEK_SET)
            self._lock_nonblocking(descriptor)
            try:
                _require_descriptor_path_identity(descriptor, self.path)
            except _PathSafetyError as exc:
                raise RunLeaseError("run lease path identity changed") from exc
        except RunLeaseError:
            self._cleanup_failed_acquire(descriptor, kernel_guard, parent_guard)
            raise
        except (OSError, _PathSafetyError) as exc:
            self._cleanup_failed_acquire(descriptor, kernel_guard, parent_guard)
            raise RunLeaseError(
                "run lease is already held or unavailable"
            ) from exc
        self._descriptor = descriptor
        self._parent_guard = parent_guard

    def require_current(self) -> None:
        descriptor = self._descriptor
        kernel_guard = self._kernel_guard
        parent_guard = self._parent_guard
        if descriptor is None or kernel_guard is None or parent_guard is None:
            raise RunLeaseError("run lease handle is not currently acquired")
        try:
            self._require_safe_path()
            parent_guard.revalidate()
            _require_descriptor_path_identity(descriptor, self.path)
            kernel_guard.require_current()
            parent_guard.revalidate()
            _require_descriptor_path_identity(descriptor, self.path)
        except RunLeaseError:
            raise
        except (OSError, _PathSafetyError) as exc:
            raise RunLeaseError(
                "run lease descriptor, path, or parent identity is not current"
            ) from exc

    def _cleanup_failed_acquire(
        self,
        descriptor: int | None,
        kernel_guard: _RunLeaseKernelGuard,
        parent_guard: _ParentPathGuard | None = None,
    ) -> None:
        descriptor_error: OSError | None = None
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                descriptor_error = exc
        parent_error: _PathSafetyError | None = None
        if parent_guard is not None:
            try:
                parent_guard.close()
            except _PathSafetyError as exc:
                parent_error = exc
        try:
            kernel_guard.close()
        except RunLeaseError as exc:
            self._kernel_guard = kernel_guard
            raise RunLeaseError(
                "run lease failed-acquire guard release failed"
            ) from exc
        self._kernel_guard = None
        if descriptor_error is not None:
            raise RunLeaseError(
                "run lease failed-acquire descriptor release failed"
            ) from descriptor_error
        if parent_error is not None:
            raise RunLeaseError(
                "run lease failed-acquire parent guard release failed"
            ) from parent_error

    def release(self) -> None:
        descriptor = self._descriptor
        kernel_guard = self._kernel_guard
        parent_guard = self._parent_guard
        if descriptor is None and kernel_guard is None and parent_guard is None:
            return
        self._descriptor = None
        self._parent_guard = None
        release_error: OSError | RunLeaseError | None = None
        guard_closed = kernel_guard is None
        try:
            if descriptor is not None:
                if os.name == "nt":
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError as exc:
            release_error = exc
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    release_error = release_error or exc
            if parent_guard is not None:
                try:
                    parent_guard.close()
                except _PathSafetyError as exc:
                    release_error = release_error or RunLeaseError(
                        "run lease parent guard release failed"
                    )
            if kernel_guard is not None:
                try:
                    kernel_guard.close()
                except RunLeaseError as exc:
                    release_error = release_error or exc
                else:
                    guard_closed = True
        if guard_closed:
            self._kernel_guard = None
        if release_error is not None:
            raise RunLeaseError("run lease release failed") from release_error

    @staticmethod
    def _lock_nonblocking(descriptor: int) -> None:
        if os.name == "nt":
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _require_safe_path(self) -> None:
        try:
            _require_plain_components(self.path)
            if os.path.lexists(self.path):
                _require_single_plain_file_metadata(self.path.lstat())
        except _PathSafetyError as exc:
            raise RunLeaseError(
                "run lease path is unsafe (link/reparse, hard link, or non-regular path)"
            ) from exc


__all__ = [
    "DurableJsonl",
    "DurableJsonlError",
    "RunLease",
    "RunLeaseError",
]
