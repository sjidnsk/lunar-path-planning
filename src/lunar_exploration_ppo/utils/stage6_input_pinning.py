"""Stage 6 正式输入的持久只读 pin。"""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from lunar_exploration_ppo.utils.path_security import (
    DurableFileIdentity,
    DurableParentGuard,
    PathSecurityError,
    guarded_file_identity,
    lexical_absolute,
)


class Stage6InputPinError(RuntimeError):
    """正式输入无法被强 pin 或在持锁期间发生漂移。"""


_STAGE6_INPUT_PIN_ISSUER = object()
_STAGE6_INPUT_PIN_LOCK = threading.RLock()
_STAGE6_ACTIVE_INPUT_PINS: dict[object, object] = {}


@dataclass(frozen=True, slots=True)
class Stage6PinnedInputRecord:
    """不含 OS handle、可供审计但不直接序列化 pin 对象的只读记录。"""

    labels: tuple[str, ...]
    path: Path
    sha256: str
    size_bytes: int
    identity: DurableFileIdentity


@dataclass(slots=True)
class _PinnedResource:
    labels: list[str]
    path: Path
    parent_guard: DurableParentGuard
    descriptor: int
    sha256: str
    size_bytes: int
    identity: DurableFileIdentity


def _descriptor_sha256(descriptor: int) -> tuple[str, int]:
    try:
        original_offset = os.lseek(descriptor, 0, os.SEEK_CUR)
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError as exc:
        raise Stage6InputPinError("pinned input descriptor is not seekable") from exc
    digest = hashlib.sha256()
    size_bytes = 0
    read_error: BaseException | None = None
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size_bytes += len(chunk)
    except BaseException as exc:
        read_error = exc
        raise
    finally:
        try:
            os.lseek(descriptor, original_offset, os.SEEK_SET)
        except OSError as reset_error:
            if read_error is not None:
                read_error.add_note(
                    f"suppressed pinned descriptor offset reset failure: {reset_error}"
                )
            else:
                raise Stage6InputPinError(
                    "pinned input descriptor offset reset failed"
                ) from reset_error
    return digest.hexdigest(), size_bytes


def _descriptor_bytes(descriptor: int) -> bytes:
    try:
        original_offset = os.lseek(descriptor, 0, os.SEEK_CUR)
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError as exc:
        raise Stage6InputPinError("pinned input descriptor is not seekable") from exc
    chunks: list[bytes] = []
    read_error: BaseException | None = None
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    except BaseException as exc:
        read_error = exc
        raise
    finally:
        try:
            os.lseek(descriptor, original_offset, os.SEEK_SET)
        except OSError as reset_error:
            if read_error is not None:
                read_error.add_note(
                    f"suppressed pinned descriptor offset reset failure: {reset_error}"
                )
            else:
                raise Stage6InputPinError(
                    "pinned input descriptor offset reset failed"
                ) from reset_error
    return b"".join(chunks)


def _close_resources(
    resources: list[_PinnedResource],
    guards: list[DurableParentGuard],
) -> None:
    errors: list[BaseException] = []
    for resource in reversed(resources):
        descriptor = resource.descriptor
        resource.descriptor = -1
        if descriptor < 0:
            continue
        try:
            os.close(descriptor)
        except BaseException as exc:
            errors.append(exc)
    for guard in reversed(guards):
        try:
            guard.close()
        except BaseException as exc:
            errors.append(exc)
    if errors:
        primary = errors[0]
        for suppressed in errors[1:]:
            primary.add_note(f"suppressed additional input pin close failure: {suppressed}")
        raise primary


class Stage6InputPin:
    """正式输入 pin 的生命周期句柄。"""

    __slots__ = ("_resources", "_guards", "_closed", "_registry_identity")

    def __init__(
        self,
        *,
        _issuer: object,
        resources: list[_PinnedResource],
        guards: list[DurableParentGuard],
    ) -> None:
        if type(self) is not Stage6InputPin or _issuer is not _STAGE6_INPUT_PIN_ISSUER:
            raise Stage6InputPinError(
                "Stage 6 input pin issuer is invalid; use acquisition"
            )
        self._resources = resources
        self._guards = guards
        self._closed = False
        self._registry_identity = object()
        with _STAGE6_INPUT_PIN_LOCK:
            _STAGE6_ACTIVE_INPUT_PINS[self] = self._registry_identity

    def require_active_issuance(self, label: str = "Stage 6 input pin") -> None:
        if not isinstance(label, str) or not label.strip():
            raise Stage6InputPinError("input pin issuance label is invalid")
        if type(self) is not Stage6InputPin:
            raise Stage6InputPinError(f"{label} input pin is forged")
        try:
            identity = self._registry_identity
            closed = self._closed
        except AttributeError as exc:
            raise Stage6InputPinError(f"{label} input pin is forged") from exc
        with _STAGE6_INPUT_PIN_LOCK:
            active_identity = _STAGE6_ACTIVE_INPUT_PINS.get(self)
        if closed is not False or active_identity is not identity:
            raise Stage6InputPinError(f"{label} input pin is closed or inactive")

    def __enter__(self) -> Stage6InputPin:
        self.require_active_issuance("Stage 6 input pin context entry")
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
            exc_value.add_note(f"suppressed Stage 6 input pin close failure: {close_error}")

    @property
    def records(self) -> tuple[Stage6PinnedInputRecord, ...]:
        self.require_active_issuance("Stage 6 input pin records")
        return tuple(
            Stage6PinnedInputRecord(
                labels=tuple(resource.labels),
                path=resource.path,
                sha256=resource.sha256,
                size_bytes=resource.size_bytes,
                identity=resource.identity,
            )
            for resource in self._resources
        )

    @property
    def paths(self) -> tuple[Path, ...]:
        self.require_active_issuance("Stage 6 input pin paths")
        return tuple(record.path for record in self.records)

    def read_bytes(self, path: str | Path, *, label: str) -> bytes:
        """Read exact bytes through the already-issued pinned descriptor."""

        if not isinstance(label, str) or not label.strip():
            raise Stage6InputPinError("pinned input read label is invalid")
        self.require_active_issuance(label)
        try:
            target = lexical_absolute(path)
        except (OSError, TypeError, ValueError) as exc:
            raise Stage6InputPinError(f"{label} pinned input path is invalid") from exc
        matches = [resource for resource in self._resources if resource.path == target]
        if len(matches) != 1:
            raise Stage6InputPinError(
                f"{label} path is not uniquely present in the active pin"
            )
        resource = matches[0]
        try:
            resource.parent_guard.revalidate()
            before = guarded_file_identity(
                resource.descriptor,
                resource.path,
                parent_guard=resource.parent_guard,
            )
            payload = _descriptor_bytes(resource.descriptor)
            after = guarded_file_identity(
                resource.descriptor,
                resource.path,
                parent_guard=resource.parent_guard,
            )
            resource.parent_guard.revalidate()
        except Stage6InputPinError:
            raise
        except (OSError, PathSecurityError) as exc:
            raise Stage6InputPinError(f"{label} pinned input read failed") from exc
        if (
            before != resource.identity
            or after != resource.identity
            or len(payload) != resource.size_bytes
            or hashlib.sha256(payload).hexdigest() != resource.sha256
        ):
            raise Stage6InputPinError(f"{label} pinned input bytes changed")
        return payload

    def require_current(self, label: str, *, rehash: bool = False) -> None:
        if not isinstance(label, str) or not label.strip():
            raise Stage6InputPinError("input pin revalidation label is invalid")
        self.require_active_issuance(label)
        try:
            for guard in self._guards:
                guard.revalidate()
            for resource in self._resources:
                current = guarded_file_identity(
                    resource.descriptor,
                    resource.path,
                    parent_guard=resource.parent_guard,
                )
                if current != resource.identity:
                    raise Stage6InputPinError(
                        f"{label} pinned input identity changed: {resource.path}"
                    )
                if rehash:
                    digest, size_bytes = _descriptor_sha256(resource.descriptor)
                    if (
                        digest != resource.sha256
                        or size_bytes != resource.size_bytes
                    ):
                        raise Stage6InputPinError(
                            f"{label} pinned input bytes changed: {resource.path}"
                        )
            for guard in self._guards:
                guard.revalidate()
        except Stage6InputPinError:
            raise
        except (OSError, PathSecurityError) as exc:
            raise Stage6InputPinError(
                f"{label} pinned input revalidation failed"
            ) from exc

    def close(self) -> None:
        if type(self) is not Stage6InputPin:
            raise Stage6InputPinError("Stage 6 input pin is forged")
        try:
            closed = self._closed
            identity = self._registry_identity
        except AttributeError as exc:
            raise Stage6InputPinError("Stage 6 input pin is forged") from exc
        if closed is True:
            return
        with _STAGE6_INPUT_PIN_LOCK:
            active_identity = _STAGE6_ACTIVE_INPUT_PINS.pop(self, None)
            self._closed = True
        if active_identity is not identity:
            raise Stage6InputPinError("Stage 6 input pin issuance is inactive")
        _close_resources(self._resources, self._guards)


def acquire_stage6_input_pin(
    inputs: Sequence[tuple[str, str | Path]],
) -> Stage6InputPin:
    """在 Windows 上以 read-only/no-write/no-delete sharing 持续 pin 输入。"""

    if os.name != "nt":
        raise Stage6InputPinError(
            "formal Stage 6 strong input pin is supported on Windows only"
        )
    if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
        raise Stage6InputPinError("input pin request must be a sequence")

    normalized: list[tuple[str, Path, str]] = []
    labels: set[str] = set()
    for item in inputs:
        if not isinstance(item, tuple) or len(item) != 2:
            raise Stage6InputPinError("input pin request row is invalid")
        label, path_value = item
        if (
            not isinstance(label, str)
            or not label
            or label != label.strip()
            or label in labels
        ):
            raise Stage6InputPinError("input pin label is invalid or duplicated")
        labels.add(label)
        try:
            path = lexical_absolute(path_value)
        except (OSError, TypeError, ValueError) as exc:
            raise Stage6InputPinError(f"input pin path is invalid: {label}") from exc
        normalized.append((label, path, os.path.normcase(os.fspath(path))))
    if not normalized:
        raise Stage6InputPinError("input pin request is empty")

    resources: list[_PinnedResource] = []
    resources_by_path: dict[str, _PinnedResource] = {}
    guards: list[DurableParentGuard] = []
    guards_by_parent: dict[str, DurableParentGuard] = {}
    pin: Stage6InputPin | None = None
    try:
        for label, path, path_key in normalized:
            existing = resources_by_path.get(path_key)
            if existing is not None:
                if existing.path != path:
                    raise Stage6InputPinError(
                        "case-normalized input aliases do not name one lexical path"
                    )
                existing.labels.append(label)
                continue
            parent_key = os.path.normcase(os.fspath(path.parent))
            guard = guards_by_parent.get(parent_key)
            if guard is None:
                guard = DurableParentGuard(path.parent)
                guards_by_parent[parent_key] = guard
                guards.append(guard)
            descriptor = guard.open_file(path, os.O_RDONLY)
            try:
                identity = guarded_file_identity(
                    descriptor,
                    path,
                    parent_guard=guard,
                )
                digest, size_bytes = _descriptor_sha256(descriptor)
                current = guarded_file_identity(
                    descriptor,
                    path,
                    parent_guard=guard,
                )
                if current != identity or size_bytes != identity[3]:
                    raise Stage6InputPinError(
                        f"input changed during pin acquisition: {label}"
                    )
            except BaseException as exc:
                try:
                    os.close(descriptor)
                except BaseException as close_error:
                    exc.add_note(
                        f"suppressed just-opened input close failure: {close_error}"
                    )
                raise
            resource = _PinnedResource(
                labels=[label],
                path=path,
                parent_guard=guard,
                descriptor=descriptor,
                sha256=digest,
                size_bytes=size_bytes,
                identity=identity,
            )
            resources.append(resource)
            resources_by_path[path_key] = resource
        pin = Stage6InputPin(
            _issuer=_STAGE6_INPUT_PIN_ISSUER,
            resources=resources,
            guards=guards,
        )
        pin.require_current("Stage 6 input pin acquisition complete")
        return pin
    except BaseException as exc:
        try:
            if pin is None:
                _close_resources(resources, guards)
            else:
                pin.close()
        except BaseException as close_error:
            exc.add_note(
                f"suppressed partial input pin cleanup failure: {close_error}"
            )
        if isinstance(exc, Stage6InputPinError):
            raise
        raise Stage6InputPinError(f"input pin acquisition failed: {exc}") from exc


__all__ = [
    "Stage6InputPin",
    "Stage6InputPinError",
    "Stage6PinnedInputRecord",
    "acquire_stage6_input_pin",
]
