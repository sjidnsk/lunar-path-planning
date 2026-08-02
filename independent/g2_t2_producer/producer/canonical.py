from __future__ import annotations

import hashlib
import io
import json
import math
import struct
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


_NPZ_ORDER = ("height_mm", "cell_class", "known", "confidence_ppm")
_NPZ_DTYPES = {
    "height_mm": np.dtype("<i4"),
    "cell_class": np.dtype("u1"),
    "known": np.dtype("u1"),
    "confidence_ppm": np.dtype("<u4"),
}


def _validate_canonical_value(value: Any, *, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value) or (
            value == 0.0 and math.copysign(1.0, value) < 0.0
        ):
            raise ValueError(f"non-canonical float at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_canonical_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, tuple):
        raise TypeError(f"tuple is not canonical JSON at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string key at {path}")
            _validate_canonical_value(item, path=f"{path}.{key}")
        return
    raise TypeError(f"unsupported canonical JSON value at {path}: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    _validate_canonical_value(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_loads(data: bytes | str) -> Any:
    text = data.decode("utf-8") if isinstance(data, bytes) else data

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-canonical float: {token}")

    value = json.loads(
        text,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )
    _validate_canonical_value(value)
    return value


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def canonical_npz_bytes(arrays: dict[str, Any], metadata: dict[str, Any]) -> bytes:
    if set(arrays) != set(_NPZ_ORDER):
        raise ValueError(f"terrain arrays must be exactly {list(_NPZ_ORDER)}")
    shapes: set[tuple[int, ...]] = set()
    encoded_arrays: dict[str, bytes] = {}
    for name in _NPZ_ORDER:
        array = np.asarray(arrays[name])
        expected_dtype = _NPZ_DTYPES[name]
        if array.dtype != expected_dtype:
            raise ValueError(f"{name} dtype must be {expected_dtype}")
        if array.ndim != 2:
            raise ValueError(f"{name} must be a 2-D array")
        shapes.add(array.shape)
        buffer = io.BytesIO()
        np.lib.format.write_array(
            buffer,
            np.ascontiguousarray(array),
            version=(2, 0),
            allow_pickle=False,
        )
        encoded_arrays[name] = buffer.getvalue()
    if len(shapes) != 1:
        raise ValueError("terrain array shapes differ")

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for filename, payload in [
            *((f"{name}.npy", encoded_arrays[name]) for name in _NPZ_ORDER),
            ("metadata.json", canonical_json_bytes(metadata)),
        ]:
            info = zipfile.ZipInfo(filename=filename, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.flag_bits = 0
            archive.writestr(info, payload)
    return output.getvalue()


def decode_canonical_npz(data: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_names = [*(f"{name}.npy" for name in _NPZ_ORDER), "metadata.json"]
    arrays: dict[str, Any] = {}
    with zipfile.ZipFile(io.BytesIO(data), mode="r") as archive:
        infos = archive.infolist()
        if [info.filename for info in infos] != expected_names:
            raise ValueError("non-canonical terrain archive member order")
        for info in infos:
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError("terrain archive is compressed")
            if info.date_time != (1980, 1, 1, 0, 0, 0):
                raise ValueError("terrain archive timestamp drift")
        for name in _NPZ_ORDER:
            with archive.open(f"{name}.npy", "r") as stream:
                array = np.lib.format.read_array(stream, allow_pickle=False)
            if array.dtype != _NPZ_DTYPES[name] or array.ndim != 2:
                raise ValueError(f"invalid {name} array contract")
            arrays[name] = array
        metadata = canonical_loads(archive.read("metadata.json"))
    return arrays, metadata


def derive_seed(
    root_seed: int,
    platform: str,
    artifact_kind: str,
    stratum: str,
    shard_index: int,
) -> int:
    digest = domain_hash(
        "g2-seed/v1",
        str(root_seed).encode("ascii"),
        platform.encode("utf-8"),
        artifact_kind.encode("utf-8"),
        stratum.encode("utf-8"),
        str(shard_index).encode("ascii"),
    )
    return int(digest[:16], 16)


def domain_hash(domain: str, *parts: bytes) -> str:
    payload = bytearray()
    for part in (domain.encode("utf-8"), *parts):
        payload.extend(struct.pack(">Q", len(part)))
        payload.extend(part)
    return hashlib.sha256(payload).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_index(root: Path, relative_paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative_path in sorted(relative_paths, key=lambda item: item.as_posix()):
        data = (root / relative_path).read_bytes()
        rows.append(
            {
                "byte_length": len(data),
                "relative_path": relative_path.as_posix(),
                "sha256": sha256_bytes(data),
            }
        )
    return rows


def content_index_root(rows: Iterable[Mapping[str, Any]]) -> str:
    normalized = sorted((dict(row) for row in rows), key=lambda row: row["relative_path"])
    return domain_hash("g2-payload-root/v1", canonical_json_bytes(normalized))
