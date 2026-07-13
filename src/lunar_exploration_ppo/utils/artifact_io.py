from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Iterable


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
        return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")

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
        self._make_parent(destination)
        line = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        with open(self._native_path(destination), "ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
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
        native_parent = self._native_path(destination.parent)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=native_parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._native_path(destination))
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def _exclusive_write(self, destination: Path, payload: bytes) -> None:
        self._make_parent(destination)
        native_parent = self._native_path(destination.parent)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=native_parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(self._native_path(temporary), self._native_path(destination))
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def _make_parent(self, path: Path) -> None:
        self._native_path(path.parent).mkdir(parents=True, exist_ok=True)

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
