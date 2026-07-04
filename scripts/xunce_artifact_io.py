from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable


def windows_safe_path(path: str | Path) -> str:
    raw = str(path)
    if os.name != "nt" or raw.startswith("\\\\?\\"):
        return raw
    absolute = str(Path(path).resolve())
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def ensure_parent(path: str | Path) -> None:
    parent = Path(path).parent
    os.makedirs(windows_safe_path(parent), exist_ok=True)


def make_dirs(path: str | Path) -> None:
    os.makedirs(windows_safe_path(path), exist_ok=True)


def path_is_file(path: str | Path) -> bool:
    return os.path.isfile(windows_safe_path(path))


def path_exists(path: str | Path) -> bool:
    return os.path.exists(windows_safe_path(path))


def read_text(path: str | Path, *, encoding: str = "utf-8-sig") -> str:
    with open(windows_safe_path(path), "r", encoding=encoding) as handle:
        return handle.read()


def read_bytes(path: str | Path) -> bytes:
    with open(windows_safe_path(path), "rb") as handle:
        return handle.read()


def write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    ensure_parent(path)
    with open(windows_safe_path(path), "w", encoding=encoding) as handle:
        handle.write(text)


def copy_file(src: str | Path, dst: str | Path) -> None:
    ensure_parent(dst)
    shutil.copy2(windows_safe_path(src), windows_safe_path(dst))


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(read_text(path))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def write_json(path: str | Path, payload: dict[str, Any], *, sort_keys: bool = True) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=sort_keys) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in read_text(path).splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], *, sort_keys: bool = True) -> None:
    write_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=sort_keys) + "\n" for row in rows))


def count_jsonl_rows(path: str | Path) -> int:
    if not path_is_file(path):
        return 0
    return sum(1 for line in read_text(path).splitlines() if line.strip())


def path_length(path: str | Path) -> int:
    return len(str(Path(path).resolve()))


def path_length_audit(paths: Iterable[str | Path], *, warn_at: int = 200, fail_at: int = 240) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    warning_count = 0
    failure_count = 0
    max_length = 0
    for path in paths:
        length = path_length(path)
        max_length = max(max_length, length)
        status = "ok"
        if length > fail_at:
            status = "fail"
            failure_count += 1
        elif length > warn_at:
            status = "warn"
            warning_count += 1
        rows.append({"path": str(path), "path_length": length, "status": status})
    return {
        "schema_version": "xunce-artifact-path-length-audit/v1",
        "warn_at": int(warn_at),
        "fail_at": int(fail_at),
        "max_path_length": int(max_length),
        "path_length_warning_count": int(warning_count),
        "path_length_failure_count": int(failure_count),
        "rows": rows,
    }
