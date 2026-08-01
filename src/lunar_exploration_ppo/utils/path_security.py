"""Fail-closed path checks for Stage 6 runtime artifacts."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes


class PathSecurityError(RuntimeError):
    """A runtime artifact path crossed a link or containment boundary."""


@dataclass(frozen=True, slots=True)
class SecureReadResult:
    """Bytes and file identity captured from one stable, plain descriptor."""

    payload: bytes
    stat_identity: tuple[int, int, int, int, int]
    link_count: int

    @property
    def file_id(self) -> tuple[int, int]:
        """Return the filesystem identity shared by aliases of this file."""

        return self.stat_identity[:2]


def _stat_file_id(metadata: os.stat_result) -> tuple[int, int]:
    """Return the platform File ID/inode seam used by manifest alias checks."""

    return (int(metadata.st_dev), int(metadata.st_ino))


def _stat_link_count(metadata: os.stat_result) -> int:
    """Return the hard-link count seam used by fail-closed readers."""

    return int(metadata.st_nlink)


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    file_id = _stat_file_id(metadata)
    return (
        *file_id,
        int(metadata.st_mode),
        int(metadata.st_size),
        int(getattr(metadata, "st_file_attributes", 0)),
    )


def _stat_stability_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        *_stat_identity(metadata),
        _stat_link_count(metadata),
        int(getattr(metadata, "st_mtime_ns", 0)),
        int(getattr(metadata, "st_ctime_ns", 0)),
    )


def _is_metadata_reparse(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _require_single_plain_file_metadata(
    metadata: os.stat_result,
    *,
    label: str,
) -> None:
    if not stat.S_ISREG(metadata.st_mode) or _is_metadata_reparse(metadata):
        raise PathSecurityError(f"{label} is not a plain regular file")
    if _stat_link_count(metadata) != 1:
        raise PathSecurityError(f"{label} hard link count must equal one")


def _secure_read_event(
    event: str,
    path: Path,
    descriptor: int | None = None,
) -> None:
    """Private deterministic fault-injection boundary for security tests."""

    del event, path, descriptor


def _open_windows_no_follow(path: Path) -> int:
    """Open a Windows file handle without following a leaf reparse point."""

    import ctypes
    import msvcrt
    from ctypes import wintypes

    class _ByHandleFileInformation(ctypes.Structure):
        _fields_ = (
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        )

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    get_information = kernel32.GetFileInformationByHandle
    get_information.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_ByHandleFileInformation),
    )
    get_information.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    generic_read = 0x80000000
    share_read_delete = 0x00000001 | 0x00000004
    open_existing = 3
    file_attribute_normal = 0x00000080
    file_flag_open_reparse_point = 0x00200000
    invalid_handle = ctypes.c_void_p(-1).value
    handle = create_file(
        os.fspath(path),
        generic_read,
        share_read_delete,
        None,
        open_existing,
        file_attribute_normal | file_flag_open_reparse_point,
        None,
    )
    if handle == invalid_handle:
        raise OSError(ctypes.get_last_error(), "CreateFileW failed")

    transferred = False
    try:
        information = _ByHandleFileInformation()
        if not get_information(handle, ctypes.byref(information)):
            raise OSError(
                ctypes.get_last_error(),
                "GetFileInformationByHandle failed",
            )
        reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        if int(information.dwFileAttributes) & reparse_flag:
            raise PathSecurityError("opened Windows handle is a reparse point")
        descriptor = msvcrt.open_osfhandle(
            int(handle),
            os.O_RDONLY | getattr(os, "O_BINARY", 0),
        )
        transferred = True
        os.set_inheritable(descriptor, False)
        return descriptor
    finally:
        if not transferred:
            close_handle(handle)


def _open_no_follow(path: Path) -> int:
    if os.name == "nt":
        return _open_windows_no_follow(path)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise PathSecurityError("no-follow descriptor open is unavailable")
    flags = os.O_RDONLY | int(no_follow)
    flags |= int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    descriptor = os.open(path, flags)
    os.set_inheritable(descriptor, False)
    return descriptor


def secure_read_bytes(
    path: str | Path,
    *,
    base: str | Path | None = None,
    label: str = "artifact file",
) -> SecureReadResult:
    """Read one single-link file through a no-follow descriptor and rebind it."""

    lexical = lexical_absolute(path)
    trusted_base = lexical_absolute(base) if base is not None else None
    descriptor = -1
    try:
        resolved_before = require_plain_path(
            lexical,
            base=trusted_base,
            leaf_kind="file",
            label=label,
        )
        path_before = lexical.lstat()
        _require_single_plain_file_metadata(path_before, label=label)
        path_before_identity = _stat_identity(path_before)
        _secure_read_event("after_path_identity", lexical)

        descriptor = _open_no_follow(lexical)
        descriptor_before = os.fstat(descriptor)
        _require_single_plain_file_metadata(descriptor_before, label=label)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        _secure_read_event("after_descriptor_read", lexical, descriptor)
        descriptor_after = os.fstat(descriptor)
        _require_single_plain_file_metadata(descriptor_after, label=label)

        resolved_after = require_plain_path(
            lexical,
            base=trusted_base,
            leaf_kind="file",
            label=label,
        )
        path_after = lexical.lstat()
        _require_single_plain_file_metadata(path_after, label=label)
    except PathSecurityError:
        raise
    except OSError as exc:
        raise PathSecurityError(f"{label} secure descriptor read failed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    descriptor_identity = _stat_identity(descriptor_after)
    if (
        resolved_before != resolved_after
        or path_before_identity != descriptor_identity
        or descriptor_identity != _stat_identity(path_after)
        or _stat_stability_identity(descriptor_before)
        != _stat_stability_identity(descriptor_after)
        or len(payload) != descriptor_after.st_size
    ):
        raise PathSecurityError(f"{label} identity changed while read")
    return SecureReadResult(
        payload=payload,
        stat_identity=descriptor_identity,
        link_count=_stat_link_count(descriptor_after),
    )


def is_link_or_reparse(path: str | Path) -> bool:
    """Return whether the path itself is a symlink, junction, or reparse point."""

    try:
        metadata = Path(path).lstat()
    except OSError:
        return False
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def lexical_absolute(path: str | Path) -> Path:
    """Build an absolute path without resolving links or junctions."""

    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path.cwd() / value
    return Path(os.path.abspath(os.fspath(value)))


def require_plain_path(
    path: str | Path,
    *,
    base: str | Path | None = None,
    allow_missing: bool = False,
    leaf_kind: str | None = None,
    label: str = "artifact path",
) -> Path:
    """Validate every existing component before resolving a contained path.

    ``leaf_kind`` may be ``"file"`` or ``"directory"``.  Missing trailing
    components are accepted only when ``allow_missing`` is true.
    """

    if leaf_kind not in {None, "file", "directory"}:
        raise ValueError("leaf_kind must be file, directory, or None")
    lexical = lexical_absolute(path)
    lexical_base = lexical_absolute(base) if base is not None else None
    if lexical_base is not None:
        try:
            lexical.relative_to(lexical_base)
        except ValueError as exc:
            raise PathSecurityError(f"{label} escapes its trusted base") from exc

    parts = lexical.parts
    if not parts:
        raise PathSecurityError(f"{label} is empty")
    current = Path(parts[0])
    missing = not os.path.lexists(current)
    for index, part in enumerate(parts[1:], start=1):
        current = current / part
        exists = os.path.lexists(current)
        if missing:
            if exists:
                raise PathSecurityError(f"{label} has a discontinuous path chain")
            continue
        if not exists:
            missing = True
            continue
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise PathSecurityError(f"{label} metadata is unavailable") from exc
        if is_link_or_reparse(current):
            raise PathSecurityError(f"{label} contains a link or reparse point")
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise PathSecurityError(f"{label} parent is not a directory")

    exists = os.path.lexists(lexical)
    if not exists and not allow_missing:
        raise PathSecurityError(f"{label} is missing")
    if exists:
        try:
            leaf = lexical.lstat()
        except OSError as exc:
            raise PathSecurityError(f"{label} metadata is unavailable") from exc
        if is_link_or_reparse(lexical):
            raise PathSecurityError(f"{label} is a link or reparse point")
        if leaf_kind == "file" and not stat.S_ISREG(leaf.st_mode):
            raise PathSecurityError(f"{label} is not a regular file")
        if leaf_kind == "directory" and not stat.S_ISDIR(leaf.st_mode):
            raise PathSecurityError(f"{label} is not a directory")

    try:
        resolved = lexical.resolve(strict=exists)
        if lexical_base is not None:
            base_exists = os.path.lexists(lexical_base)
            resolved_base = lexical_base.resolve(strict=base_exists)
            resolved.relative_to(resolved_base)
    except (OSError, ValueError) as exc:
        raise PathSecurityError(f"{label} escaped during resolution") from exc
    return resolved


def path_identity(path: str | Path) -> tuple[int, int, int, int, int]:
    """Return a stable lstat identity for a previously validated path."""

    try:
        metadata = Path(path).lstat()
    except OSError as exc:
        raise PathSecurityError("artifact identity is unavailable") from exc
    return _stat_identity(metadata)


DurableDirectoryIdentity = tuple[int, int, int, int]
DurableFileIdentity = tuple[int, int, int, int, int, int]
_DirectoryIdentity = DurableDirectoryIdentity
_RegularFileIdentity = DurableFileIdentity


def _directory_identity(metadata: os.stat_result) -> _DirectoryIdentity:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(stat.S_IFMT(metadata.st_mode)),
        int(getattr(metadata, "st_file_attributes", 0)),
    )


def _regular_file_identity(metadata: os.stat_result) -> _RegularFileIdentity:
    _require_single_plain_file_metadata(metadata, label="writable artifact file")
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(stat.S_IFMT(metadata.st_mode)),
        int(metadata.st_size),
        int(getattr(metadata, "st_file_attributes", 0)),
        _stat_link_count(metadata),
    )


def _directory_chain(
    path: Path,
) -> tuple[tuple[Path, _DirectoryIdentity], ...]:
    parts = path.parts
    if not path.is_absolute() or not parts:
        raise PathSecurityError("durable directory path is not absolute")
    current = Path(parts[0])
    result: list[tuple[Path, _DirectoryIdentity]] = []
    for part in (None, *parts[1:]):
        if part is not None:
            current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise PathSecurityError(
                "durable directory chain metadata is unavailable"
            ) from exc
        if _is_metadata_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise PathSecurityError(
                "durable directory chain contains a link or non-directory"
            )
        result.append((current, _directory_identity(metadata)))
    return tuple(result)


if os.name == "nt":
    _DELETE_ACCESS = 0x00010000
    _FILE_READ_ATTRIBUTES = 0x00000080
    _GENERIC_READ = 0x80000000
    _GENERIC_WRITE = 0x40000000
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _CREATE_NEW = 1
    _CREATE_ALWAYS = 2
    _OPEN_EXISTING = 3
    _OPEN_ALWAYS = 4
    _TRUNCATE_EXISTING = 5
    _FILE_ATTRIBUTE_NORMAL = 0x00000080
    _FILE_ATTRIBUTE_DIRECTORY = 0x00000010
    _FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
    _FILE_FLAG_WRITE_THROUGH = 0x80000000
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _FILE_RENAME_INFO_CLASS = 3
    _FILE_DISPOSITION_INFO_CLASS = 4
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    class _WindowsHandleInformation(ctypes.Structure):
        _fields_ = (
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        )

    class _WindowsRenameInfoHeader(ctypes.Structure):
        _fields_ = (
            ("ReplaceIfExists", wintypes.DWORD),
            ("RootDirectory", wintypes.HANDLE),
            ("FileNameLength", wintypes.DWORD),
        )

    class _WindowsDispositionInfo(ctypes.Structure):
        _fields_ = (("DeleteFile", ctypes.c_ubyte),)

    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _CREATE_FILE = _KERNEL32.CreateFileW
    _CREATE_FILE.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    _CREATE_FILE.restype = wintypes.HANDLE
    _GET_FILE_INFORMATION = _KERNEL32.GetFileInformationByHandle
    _GET_FILE_INFORMATION.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_WindowsHandleInformation),
    )
    _GET_FILE_INFORMATION.restype = wintypes.BOOL
    _FLUSH_FILE_BUFFERS = _KERNEL32.FlushFileBuffers
    _FLUSH_FILE_BUFFERS.argtypes = (wintypes.HANDLE,)
    _FLUSH_FILE_BUFFERS.restype = wintypes.BOOL
    _SET_FILE_INFORMATION = _KERNEL32.SetFileInformationByHandle
    _SET_FILE_INFORMATION.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    _SET_FILE_INFORMATION.restype = wintypes.BOOL
    _CLOSE_WIN_HANDLE = _KERNEL32.CloseHandle
    _CLOSE_WIN_HANDLE.argtypes = (wintypes.HANDLE,)
    _CLOSE_WIN_HANDLE.restype = wintypes.BOOL


def _windows_native_path(path: Path) -> str:
    raw = os.path.abspath(os.fspath(path))
    if raw.startswith("\\\\?\\"):
        return raw
    if raw.startswith("\\\\"):
        return "\\\\?\\UNC\\" + raw[2:]
    return "\\\\?\\" + raw


def _windows_error(message: str) -> OSError:
    error_code = ctypes.get_last_error()
    return OSError(error_code, message)


def _windows_handle_information(handle: int) -> _WindowsHandleInformation:
    information = _WindowsHandleInformation()
    if not _GET_FILE_INFORMATION(handle, ctypes.byref(information)):
        raise _windows_error("GetFileInformationByHandle failed")
    return information


def _windows_handle_file_id(
    information: _WindowsHandleInformation,
) -> tuple[int, int]:
    index = (int(information.nFileIndexHigh) << 32) | int(
        information.nFileIndexLow
    )
    return int(information.dwVolumeSerialNumber), index


def _open_windows_directory_handle(path: Path, *, access: int) -> int:
    handle = _CREATE_FILE(
        _windows_native_path(path),
        access,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if handle == _INVALID_HANDLE_VALUE:
        raise _windows_error("CreateFileW directory open failed")
    handle_value = int(handle)
    try:
        information = _windows_handle_information(handle_value)
        attributes = int(information.dwFileAttributes)
        if not attributes & _FILE_ATTRIBUTE_DIRECTORY:
            raise PathSecurityError("Windows parent handle is not a directory")
        if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise PathSecurityError("Windows parent handle is a reparse point")
    except BaseException:
        _CLOSE_WIN_HANDLE(handle_value)
        raise
    return handle_value


def _open_windows_guarded_file(path: Path, flags: int) -> int:
    writable = bool(flags & (os.O_WRONLY | os.O_RDWR))
    if flags & os.O_RDWR:
        access = _GENERIC_READ | _GENERIC_WRITE
        crt_access = os.O_RDWR
    elif flags & os.O_WRONLY:
        access = _GENERIC_WRITE
        crt_access = os.O_WRONLY
    else:
        access = _GENERIC_READ
        crt_access = os.O_RDONLY
    create = bool(flags & os.O_CREAT)
    exclusive = bool(flags & os.O_EXCL)
    truncate = bool(flags & os.O_TRUNC)
    if create and exclusive:
        disposition = _CREATE_NEW
    elif create and truncate:
        disposition = _CREATE_ALWAYS
    elif create:
        disposition = _OPEN_ALWAYS
    elif truncate:
        disposition = _TRUNCATE_EXISTING
    else:
        disposition = _OPEN_EXISTING
    attributes = _FILE_ATTRIBUTE_NORMAL | _FILE_FLAG_OPEN_REPARSE_POINT
    if writable:
        attributes |= _FILE_FLAG_WRITE_THROUGH
    handle = _CREATE_FILE(
        _windows_native_path(path),
        access,
        _FILE_SHARE_READ,
        None,
        disposition,
        attributes,
        None,
    )
    if handle == _INVALID_HANDLE_VALUE:
        raise _windows_error("CreateFileW guarded file open failed")
    handle_value = int(handle)
    transferred = False
    try:
        information = _windows_handle_information(handle_value)
        file_attributes = int(information.dwFileAttributes)
        if file_attributes & (
            _FILE_ATTRIBUTE_DIRECTORY | _FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise PathSecurityError(
                "Windows writable leaf is not a plain regular file"
            )
        if int(information.nNumberOfLinks) != 1:
            raise PathSecurityError(
                "Windows writable leaf hard link count must equal one"
            )
        crt_flags = crt_access | int(getattr(os, "O_BINARY", 0))
        crt_flags |= int(getattr(os, "O_NOINHERIT", 0))
        if flags & os.O_APPEND:
            crt_flags |= os.O_APPEND
        descriptor = msvcrt.open_osfhandle(handle_value, crt_flags)
        transferred = True
        os.set_inheritable(descriptor, False)
        return descriptor
    finally:
        if not transferred:
            _CLOSE_WIN_HANDLE(handle_value)


def _namespace_event(
    event: str,
    source: Path,
    destination: Path | None,
) -> None:
    """Private deterministic fault boundary for namespace-race tests."""

    del event, source, destination


def _require_expected_file_identity(
    identity: object,
) -> DurableFileIdentity:
    if (
        not isinstance(identity, tuple)
        or len(identity) != 6
        or any(type(value) is not int for value in identity)
        or identity[2] != stat.S_IFREG
        or identity[5] != 1
    ):
        raise PathSecurityError("expected durable file identity is invalid")
    return identity


def _require_expected_directory_identity(
    identity: object,
) -> DurableDirectoryIdentity:
    if (
        not isinstance(identity, tuple)
        or len(identity) != 4
        or any(type(value) is not int for value in identity)
        or identity[2] != stat.S_IFDIR
    ):
        raise PathSecurityError("expected durable directory identity is invalid")
    return identity


def durable_file_identity(path: str | Path) -> DurableFileIdentity:
    """Capture the exact single-link regular-file identity expected by mutations."""

    leaf = lexical_absolute(path)
    try:
        return _regular_file_identity(leaf.lstat())
    except PathSecurityError:
        raise
    except OSError as exc:
        raise PathSecurityError("durable file identity is unavailable") from exc


def durable_directory_identity(path: str | Path) -> DurableDirectoryIdentity:
    """Capture a plain directory identity expected by a namespace mutation."""

    directory = lexical_absolute(path)
    try:
        metadata = directory.lstat()
    except OSError as exc:
        raise PathSecurityError("durable directory identity is unavailable") from exc
    if _is_metadata_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise PathSecurityError("durable directory is not plain")
    return _directory_identity(metadata)


def _windows_namespace_handle_identity(
    handle: int,
    path: Path,
    *,
    directory: bool,
) -> DurableFileIdentity | DurableDirectoryIdentity:
    information = _windows_handle_information(handle)
    attributes = int(information.dwFileAttributes)
    if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
        raise PathSecurityError("Windows namespace handle is a reparse point")
    if bool(attributes & _FILE_ATTRIBUTE_DIRECTORY) != directory:
        raise PathSecurityError("Windows namespace handle kind changed")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PathSecurityError("Windows namespace path identity is unavailable") from exc
    if _is_metadata_reparse(metadata):
        raise PathSecurityError("Windows namespace path became a reparse point")
    if int(metadata.st_ino) != _windows_handle_file_id(information)[1]:
        raise PathSecurityError("Windows namespace handle/path identity changed")
    if directory:
        if not stat.S_ISDIR(metadata.st_mode):
            raise PathSecurityError("Windows namespace directory kind changed")
        return _directory_identity(metadata)
    identity = _regular_file_identity(metadata)
    handle_size = (int(information.nFileSizeHigh) << 32) | int(
        information.nFileSizeLow
    )
    if handle_size != identity[3] or int(information.nNumberOfLinks) != 1:
        raise PathSecurityError("Windows namespace file identity changed")
    return identity


def _open_windows_namespace_handle(path: Path, *, directory: bool) -> int:
    flags = _FILE_FLAG_OPEN_REPARSE_POINT
    access = _DELETE_ACCESS | _FILE_READ_ATTRIBUTES
    share = _FILE_SHARE_READ
    if directory:
        flags |= _FILE_FLAG_BACKUP_SEMANTICS
        share = _FILE_SHARE_READ | _FILE_SHARE_WRITE
    else:
        flags |= _FILE_ATTRIBUTE_NORMAL
    handle = _CREATE_FILE(
        _windows_native_path(path),
        access,
        share,
        None,
        _OPEN_EXISTING,
        flags,
        None,
    )
    if handle == _INVALID_HANDLE_VALUE:
        raise _windows_error("CreateFileW namespace handle open failed")
    return int(handle)


def _close_windows_handle_preserving(
    handle: int,
    primary_error: BaseException | None,
) -> None:
    if _CLOSE_WIN_HANDLE(handle):
        return
    close_error = _windows_error("CloseHandle namespace operation failed")
    if primary_error is not None:
        primary_error.add_note(f"suppressed handle-close failure: {close_error}")
        return
    raise close_error


def _windows_rename_by_handle(
    handle: int,
    destination: Path,
    *,
    replace_existing: bool,
) -> None:
    destination_bytes = _windows_native_path(destination).encode("utf-16-le")
    file_name_offset = _WindowsRenameInfoHeader.FileNameLength.offset + ctypes.sizeof(
        wintypes.DWORD
    )
    buffer_size = max(
        ctypes.sizeof(_WindowsRenameInfoHeader),
        file_name_offset + len(destination_bytes) + ctypes.sizeof(wintypes.WCHAR),
    )
    buffer = ctypes.create_string_buffer(buffer_size)
    header = _WindowsRenameInfoHeader.from_buffer(buffer)
    header.ReplaceIfExists = int(replace_existing)
    header.RootDirectory = None
    header.FileNameLength = len(destination_bytes)
    ctypes.memmove(
        ctypes.addressof(buffer) + file_name_offset,
        destination_bytes,
        len(destination_bytes),
    )
    if not _SET_FILE_INFORMATION(
        handle,
        _FILE_RENAME_INFO_CLASS,
        ctypes.byref(buffer),
        buffer_size,
    ):
        error = _windows_error("SetFileInformationByHandle rename failed")
        if not replace_existing and getattr(error, "winerror", error.errno) in {
            80,
            183,
        }:
            raise FileExistsError(*error.args) from error
        raise error


def _windows_delete_by_handle(handle: int) -> None:
    disposition = _WindowsDispositionInfo(DeleteFile=1)
    if not _SET_FILE_INFORMATION(
        handle,
        _FILE_DISPOSITION_INFO_CLASS,
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
    ):
        raise _windows_error("SetFileInformationByHandle disposition failed")


class DurableParentGuard:
    """Hold one verified parent namespace stable for durable leaf operations."""

    def __init__(self, path: str | Path) -> None:
        self.path = lexical_absolute(path)
        self._chain = _directory_chain(self.path)
        self._descriptor: int | None = None
        self._windows_handles: list[int] = []
        self._windows_file_ids: list[tuple[int, int]] = []
        try:
            if os.name == "nt":
                for component, _identity in self._chain:
                    handle = _open_windows_directory_handle(
                        component,
                        access=_GENERIC_READ,
                    )
                    self._windows_handles.append(handle)
                    self._windows_file_ids.append(
                        _windows_handle_file_id(
                            _windows_handle_information(handle)
                        )
                    )
            else:
                no_follow = getattr(os, "O_NOFOLLOW", None)
                directory_flag = getattr(os, "O_DIRECTORY", None)
                if no_follow is None or directory_flag is None:
                    raise PathSecurityError(
                        "POSIX durable parent dir-fd primitives are unavailable"
                    )
                flags = os.O_RDONLY | int(directory_flag) | int(no_follow)
                flags |= int(getattr(os, "O_CLOEXEC", 0))
                descriptor = os.open(self.path, flags)
                os.set_inheritable(descriptor, False)
                self._descriptor = descriptor
            self.revalidate()
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> DurableParentGuard:
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
            exc_value.add_note(
                f"suppressed durable parent guard close failure: {close_error}"
            )

    @property
    def descriptor(self) -> int | None:
        return self._descriptor

    def revalidate(self) -> None:
        if _directory_chain(self.path) != self._chain:
            raise PathSecurityError("durable parent directory identity drifted")
        if os.name == "nt":
            if len(self._windows_handles) != len(self._chain):
                raise PathSecurityError("Windows parent guard is incomplete")
            for handle, expected_file_id in zip(
                self._windows_handles,
                self._windows_file_ids,
                strict=True,
            ):
                information = _windows_handle_information(handle)
                attributes = int(information.dwFileAttributes)
                if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
                    raise PathSecurityError(
                        "Windows guarded parent became a reparse point"
                    )
                if _windows_handle_file_id(information) != expected_file_id:
                    raise PathSecurityError(
                        "Windows guarded parent handle identity drifted"
                    )
            return
        descriptor = self._descriptor
        if descriptor is None:
            raise PathSecurityError("POSIX parent directory descriptor is closed")
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or _directory_identity(metadata) != self._chain[-1][1]
        ):
            raise PathSecurityError("POSIX parent dir-fd identity drifted")

    def open_file(self, path: str | Path, flags: int, mode: int = 0o600) -> int:
        leaf = lexical_absolute(path)
        if leaf.parent != self.path:
            raise PathSecurityError("guarded leaf escaped its parent directory")
        self.revalidate()
        if os.path.lexists(leaf):
            _require_single_plain_file_metadata(
                leaf.lstat(),
                label="guarded writable leaf",
            )
        descriptor = -1
        try:
            if os.name == "nt":
                descriptor = _open_windows_guarded_file(leaf, flags)
            else:
                no_follow = getattr(os, "O_NOFOLLOW", None)
                if no_follow is None or self._descriptor is None:
                    raise PathSecurityError(
                        "POSIX guarded leaf open primitive is unavailable"
                    )
                secure_flags = flags | int(no_follow)
                secure_flags |= int(getattr(os, "O_CLOEXEC", 0))
                secure_flags |= int(getattr(os, "O_BINARY", 0))
                descriptor = os.open(
                    leaf.name,
                    secure_flags,
                    mode,
                    dir_fd=self._descriptor,
                )
                os.set_inheritable(descriptor, False)
            descriptor_metadata = os.fstat(descriptor)
            path_metadata = leaf.lstat()
            descriptor_identity = _regular_file_identity(descriptor_metadata)
            path_identity_value = _regular_file_identity(path_metadata)
            if descriptor_identity[:3] != path_identity_value[:3]:
                raise PathSecurityError(
                    "guarded writable descriptor/path identity drifted"
                )
            self.revalidate()
            return descriptor
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            raise

    def flush(self) -> None:
        self.revalidate()
        if os.name == "nt":
            handle = _open_windows_directory_handle(
                self.path,
                access=_GENERIC_WRITE,
            )
            flush_error: BaseException | None = None
            try:
                if not _FLUSH_FILE_BUFFERS(handle):
                    raise _windows_error(
                        "FlushFileBuffers directory durability failed"
                    )
            except BaseException as exc:
                flush_error = exc
                raise
            finally:
                if not _CLOSE_WIN_HANDLE(handle):
                    close_error = _windows_error(
                        "CloseHandle directory flush failed"
                    )
                    if flush_error is not None:
                        flush_error.add_note(
                            f"suppressed directory flush close failure: {close_error}"
                        )
                    else:
                        raise close_error
        else:
            descriptor = self._descriptor
            if descriptor is None:
                raise PathSecurityError(
                    "POSIX parent directory descriptor is unavailable"
                )
            os.fsync(descriptor)
        self.revalidate()

    def close(self) -> None:
        descriptor = self._descriptor
        self._descriptor = None
        close_errors: list[OSError] = []
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                close_errors.append(exc)
        handles = self._windows_handles
        self._windows_handles = []
        self._windows_file_ids = []
        for handle in reversed(handles):
            if not _CLOSE_WIN_HANDLE(handle):
                close_errors.append(_windows_error("CloseHandle parent guard failed"))
        if close_errors:
            primary = close_errors[0]
            for suppressed in close_errors[1:]:
                primary.add_note(f"suppressed additional close failure: {suppressed}")
            raise primary


def guarded_file_identity(
    descriptor: int,
    path: str | Path,
    *,
    parent_guard: DurableParentGuard,
) -> _RegularFileIdentity:
    leaf = lexical_absolute(path)
    if leaf.parent != parent_guard.path:
        raise PathSecurityError("guarded identity leaf escaped its parent")
    parent_guard.revalidate()
    descriptor_identity = _regular_file_identity(os.fstat(descriptor))
    path_identity_value = _regular_file_identity(leaf.lstat())
    if descriptor_identity[:3] != path_identity_value[:3]:
        raise PathSecurityError("guarded file descriptor/path identity drifted")
    parent_guard.revalidate()
    return descriptor_identity


def _require_same_parent(source: Path, destination: Path) -> Path:
    source_parent = lexical_absolute(source).parent
    destination_parent = lexical_absolute(destination).parent
    if source_parent != destination_parent:
        raise PathSecurityError("durable namespace operation crossed parents")
    return source_parent


def durable_fsync_directory(path: str | Path) -> None:
    with DurableParentGuard(path) as guard:
        guard.flush()


def durable_replace(
    source: str | Path,
    destination: str | Path,
    *,
    expected_source_identity: DurableFileIdentity | DurableDirectoryIdentity,
    replace_existing: bool = True,
) -> None:
    source_path = lexical_absolute(source)
    destination_path = lexical_absolute(destination)
    if source_path == destination_path:
        raise PathSecurityError("durable namespace source equals destination")
    parent = _require_same_parent(source_path, destination_path)
    if isinstance(expected_source_identity, tuple) and len(
        expected_source_identity
    ) == 6:
        expected: DurableFileIdentity | DurableDirectoryIdentity = (
            _require_expected_file_identity(expected_source_identity)
        )
        source_is_directory = False
    elif isinstance(expected_source_identity, tuple) and len(
        expected_source_identity
    ) == 4:
        expected = _require_expected_directory_identity(expected_source_identity)
        source_is_directory = True
    else:
        raise PathSecurityError("expected durable source identity is invalid")
    with DurableParentGuard(parent) as guard:
        if os.name != "nt":
            raise PathSecurityError(
                "identity-bound POSIX namespace mutation is unsupported"
            )
        guard.revalidate()
        _namespace_event("before_source_handle_open", source_path, destination_path)
        handle = -1
        operation_error: BaseException | None = None
        try:
            handle = _open_windows_namespace_handle(
                source_path,
                directory=source_is_directory,
            )
            if (
                _windows_namespace_handle_identity(
                    handle,
                    source_path,
                    directory=source_is_directory,
                )
                != expected
            ):
                raise PathSecurityError("durable rename source identity changed")
            guard.revalidate()
            _namespace_event(
                "after_source_handle_bound",
                source_path,
                destination_path,
            )
            if (
                _windows_namespace_handle_identity(
                    handle,
                    source_path,
                    directory=source_is_directory,
                )
                != expected
            ):
                raise PathSecurityError("durable rename source identity changed")
            if os.path.lexists(destination_path):
                if not replace_existing:
                    raise FileExistsError(destination_path)
                destination_metadata = destination_path.lstat()
                if source_is_directory:
                    if _is_metadata_reparse(
                        destination_metadata
                    ) or not stat.S_ISDIR(destination_metadata.st_mode):
                        raise PathSecurityError(
                            "durable rename destination kind is unsafe"
                        )
                else:
                    _require_single_plain_file_metadata(
                        destination_metadata,
                        label="durable rename destination",
                    )
            _windows_rename_by_handle(
                handle,
                destination_path,
                replace_existing=replace_existing,
            )
            if os.path.lexists(source_path):
                raise PathSecurityError("durable rename source survived publication")
            if (
                _windows_namespace_handle_identity(
                    handle,
                    destination_path,
                    directory=source_is_directory,
                )
                != expected
            ):
                raise PathSecurityError(
                    "durable rename published destination identity changed"
                )
            guard.flush()
        except BaseException as exc:
            operation_error = exc
            raise
        finally:
            if handle >= 0:
                _close_windows_handle_preserving(handle, operation_error)


def durable_rename(
    source: str | Path,
    destination: str | Path,
    *,
    expected_source_identity: DurableFileIdentity | DurableDirectoryIdentity,
    replace_existing: bool = False,
) -> None:
    durable_replace(
        source,
        destination,
        expected_source_identity=expected_source_identity,
        replace_existing=replace_existing,
    )


def durable_publish_exclusive(
    source: str | Path,
    destination: str | Path,
    *,
    expected_source_identity: DurableFileIdentity,
) -> None:
    durable_replace(
        source,
        destination,
        expected_source_identity=expected_source_identity,
        replace_existing=False,
    )


def durable_unlink(
    path: str | Path,
    *,
    expected_identity: DurableFileIdentity,
) -> None:
    leaf = lexical_absolute(path)
    expected = _require_expected_file_identity(expected_identity)
    with DurableParentGuard(leaf.parent) as guard:
        if os.name != "nt":
            raise PathSecurityError(
                "identity-bound POSIX namespace mutation is unsupported"
            )
        _namespace_event("before_source_handle_open", leaf, None)
        handle = -1
        operation_error: BaseException | None = None
        try:
            handle = _open_windows_namespace_handle(leaf, directory=False)
            if (
                _windows_namespace_handle_identity(
                    handle,
                    leaf,
                    directory=False,
                )
                != expected
            ):
                raise PathSecurityError("durable unlink source identity changed")
            _namespace_event("after_source_handle_bound", leaf, None)
            if (
                _windows_namespace_handle_identity(
                    handle,
                    leaf,
                    directory=False,
                )
                != expected
            ):
                raise PathSecurityError("durable unlink source identity changed")
            _windows_delete_by_handle(handle)
        except BaseException as exc:
            operation_error = exc
            raise
        finally:
            if handle >= 0:
                _close_windows_handle_preserving(handle, operation_error)
        if os.path.lexists(leaf):
            raise PathSecurityError("durable unlink path was recreated or survived")
        guard.flush()


def durable_rmdir(
    path: str | Path,
    *,
    expected_identity: DurableDirectoryIdentity,
) -> None:
    directory = lexical_absolute(path)
    expected = _require_expected_directory_identity(expected_identity)
    with DurableParentGuard(directory.parent) as guard:
        if os.name != "nt":
            raise PathSecurityError(
                "identity-bound POSIX namespace mutation is unsupported"
            )
        _namespace_event("before_source_handle_open", directory, None)
        handle = -1
        operation_error: BaseException | None = None
        try:
            handle = _open_windows_namespace_handle(directory, directory=True)
            if (
                _windows_namespace_handle_identity(
                    handle,
                    directory,
                    directory=True,
                )
                != expected
            ):
                raise PathSecurityError("durable rmdir identity changed")
            _namespace_event("after_source_handle_bound", directory, None)
            if (
                _windows_namespace_handle_identity(
                    handle,
                    directory,
                    directory=True,
                )
                != expected
            ):
                raise PathSecurityError("durable rmdir identity changed")
            _windows_delete_by_handle(handle)
        except BaseException as exc:
            operation_error = exc
            raise
        finally:
            if handle >= 0:
                _close_windows_handle_preserving(handle, operation_error)
        if os.path.lexists(directory):
            raise PathSecurityError("durable rmdir path was recreated or survived")
        guard.flush()


def durable_mkdir(path: str | Path, *, mode: int = 0o777) -> None:
    directory = lexical_absolute(path)
    with DurableParentGuard(directory.parent) as guard:
        if os.name == "nt":
            os.mkdir(directory, mode)
        else:
            descriptor = guard.descriptor
            if descriptor is None:
                raise PathSecurityError("POSIX mkdir dir-fd is unavailable")
            os.mkdir(directory.name, mode, dir_fd=descriptor)
        metadata = directory.lstat()
        if _is_metadata_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise PathSecurityError("durable mkdir target is not a plain directory")
        guard.flush()


def durable_makedirs(path: str | Path, *, mode: int = 0o777) -> None:
    directory = lexical_absolute(path)
    missing: list[Path] = []
    current = directory
    while not os.path.lexists(current):
        missing.append(current)
        if current.parent == current:
            raise PathSecurityError("durable directory has no existing ancestor")
        current = current.parent
    require_plain_path(
        current,
        leaf_kind="directory",
        label="durable directory existing ancestor",
    )
    for candidate in reversed(missing):
        try:
            durable_mkdir(candidate, mode=mode)
        except FileExistsError:
            require_plain_path(
                candidate,
                leaf_kind="directory",
                label="concurrently created durable directory",
            )
    require_plain_path(
        directory,
        leaf_kind="directory",
        label="durable directory",
    )


__all__ = [
    "DurableDirectoryIdentity",
    "DurableFileIdentity",
    "DurableParentGuard",
    "PathSecurityError",
    "SecureReadResult",
    "durable_directory_identity",
    "durable_file_identity",
    "durable_fsync_directory",
    "durable_makedirs",
    "durable_mkdir",
    "durable_publish_exclusive",
    "durable_rename",
    "durable_replace",
    "durable_rmdir",
    "durable_unlink",
    "guarded_file_identity",
    "is_link_or_reparse",
    "lexical_absolute",
    "path_identity",
    "require_plain_path",
    "secure_read_bytes",
]
