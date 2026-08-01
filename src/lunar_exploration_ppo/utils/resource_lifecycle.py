"""Stage 6 跨进程资源生命周期 segment ledger。

本模块只处理 append-only resource audit 的进程 segment、segment-local RSS
时序与 terminal 聚合绑定；它不依赖 PPO、模型、checkpoint 或 Stage 6 backend。
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lunar_exploration_ppo.utils.durable_jsonl import (
    DurableJsonl,
    DurableJsonlError,
)


_SEGMENT_SCHEMA = "stage6_resource_lifecycle_segment/v1"
_TERMINAL_SCHEMA = "stage6_terminal_resource_evidence/v4"
_VALIDATION_SCHEMA = "stage6_resource_lifecycle_validation/v2"
_RSS_SOURCE = "process_tree_lifecycle_peak_current_sum/v1"
_TERMINAL_BOUNDARY = (
    "after_finalization_payload_and_monitor_stop_before_success_commit/v4"
)
_SEGMENT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "phase",
        "segment_id",
        "segment_index",
        "root_pid",
        "first_sample",
        "prior_resource_log",
    }
)


class ResourceLifecycleError(RuntimeError):
    """Resource lifecycle ledger 无法被完整证明时抛出。"""


@dataclass(slots=True)
class _SegmentState:
    segment_id: str
    segment_index: int
    root_pid: int
    local_first_sample_count: int
    local_final_sample_count: int
    local_peak_rss_bytes: int
    start_row_identity: dict[str, object]

    def summary(self) -> dict[str, object]:
        return {
            "segment_id": self.segment_id,
            "segment_index": self.segment_index,
            "root_pid": self.root_pid,
            "local_first_sample_count": self.local_first_sample_count,
            "local_final_sample_count": self.local_final_sample_count,
            "local_peak_rss_bytes": self.local_peak_rss_bytes,
            "start_row_identity": dict(self.start_row_identity),
        }


@dataclass(slots=True)
class _AttemptState:
    segment_id: str
    segment_index: int
    pre: dict[str, object] | None = None
    post: dict[str, object] | None = None
    accepted: bool = False


@dataclass(slots=True)
class _ValidatedLedger:
    result: dict[str, object]
    segments: list[_SegmentState]
    payload: bytes
    terminal_present: bool


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _canonical_row(value: object) -> bytes:
    try:
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
    except (TypeError, UnicodeError, ValueError) as exc:
        raise ResourceLifecycleError(
            "resource lifecycle row is not canonical JSON data"
        ) from exc


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _prefix_binding(payload: bytes) -> dict[str, object]:
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _validate_binding(value: object, *, label: str) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"sha256", "size_bytes"}
        or not _is_sha256(value.get("sha256"))
        or type(value.get("size_bytes")) is not int
        or int(value["size_bytes"]) < 0
    ):
        raise ResourceLifecycleError(f"{label} binding drifted")
    return dict(value)


def _validate_segment_id(value: object) -> str:
    if not isinstance(value, str):
        raise ResourceLifecycleError("resource segment id drifted")
    try:
        parsed = uuid.UUID(hex=value)
    except (AttributeError, ValueError) as exc:
        raise ResourceLifecycleError("resource segment id drifted") from exc
    if parsed.hex != value or parsed.version != 4:
        raise ResourceLifecycleError("resource segment id drifted")
    return value


def _validate_segment_index(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ResourceLifecycleError("resource segment index drifted")
    return value


def _resource_fields(
    value: object,
    *,
    label: str,
) -> tuple[int, int, int, bool]:
    if not isinstance(value, Mapping):
        raise ResourceLifecycleError(f"{label} resource snapshot drifted")
    root_pid = value.get("rss_root_pid")
    sample_count = value.get("rss_sample_count")
    rss_bytes = value.get("rss_bytes")
    passed = value.get("passed")
    if (
        value.get("rss_source") != _RSS_SOURCE
        or type(root_pid) is not int
        or root_pid <= 0
        or type(sample_count) is not int
        or sample_count <= 0
        or type(rss_bytes) is not int
        or rss_bytes < 0
        or type(passed) is not bool
    ):
        raise ResourceLifecycleError(f"{label} resource lifecycle fields drifted")
    return root_pid, sample_count, rss_bytes, passed


def _validate_segment_descriptor(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _SEGMENT_FIELDS:
        raise ResourceLifecycleError("resource segment start schema drifted")
    if (
        value.get("schema_version") != _SEGMENT_SCHEMA
        or value.get("kind") != "resource_lifecycle_segment"
        or value.get("phase") != "segment_start"
    ):
        raise ResourceLifecycleError("resource segment start schema drifted")
    segment_id = _validate_segment_id(value.get("segment_id"))
    segment_index = _validate_segment_index(value.get("segment_index"))
    root_pid = value.get("root_pid")
    if type(root_pid) is not int or root_pid <= 0:
        raise ResourceLifecycleError("resource segment root PID drifted")
    sample_root_pid, _, _, _ = _resource_fields(
        value.get("first_sample"),
        label="resource segment first",
    )
    if sample_root_pid != root_pid:
        raise ResourceLifecycleError("resource segment root PID drifted")
    _validate_binding(value.get("prior_resource_log"), label="resource segment prior")
    result = dict(value)
    result["segment_id"] = segment_id
    result["segment_index"] = segment_index
    result["root_pid"] = root_pid
    return result


def _ledger_path(path: str | Path) -> Path:
    try:
        return Path(os.path.abspath(Path(path).expanduser()))
    except (OSError, TypeError, ValueError) as exc:
        raise ResourceLifecycleError("resource lifecycle ledger path drifted") from exc


def _read_canonical_ledger(path: Path) -> tuple[bytes, list[dict[str, object]]]:
    try:
        payload = DurableJsonl(path).recover_and_snapshot()
    except (DurableJsonlError, OSError) as exc:
        raise ResourceLifecycleError(
            "resource lifecycle ledger committed snapshot failed"
        ) from exc
    if payload and not payload.endswith(b"\n"):
        raise ResourceLifecycleError(
            "resource lifecycle ledger has an incomplete tail"
        )
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        try:
            value = json.loads(line, parse_constant=_reject_json_constant)
        except (TypeError, UnicodeError, ValueError) as exc:
            raise ResourceLifecycleError(
                f"resource lifecycle row {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict) or _canonical_row(value) != line:
            raise ResourceLifecycleError(
                f"resource lifecycle row {line_number} is not canonical"
            )
        rows.append(value)
    return payload, rows


def _advance_segment(
    segment: _SegmentState,
    resource: object,
    *,
    label: str,
) -> dict[str, object]:
    root_pid, sample_count, rss_bytes, _ = _resource_fields(resource, label=label)
    if root_pid != segment.root_pid:
        raise ResourceLifecycleError("resource segment root PID drifted")
    if sample_count < segment.local_final_sample_count:
        raise ResourceLifecycleError("resource segment sample count rolled back")
    if rss_bytes < segment.local_peak_rss_bytes:
        raise ResourceLifecycleError("resource segment RSS peak rolled back")
    segment.local_final_sample_count = sample_count
    segment.local_peak_rss_bytes = rss_bytes
    return dict(resource)  # type: ignore[arg-type]


def _row_segment(
    row: Mapping[str, object],
    segments: Sequence[_SegmentState],
    *,
    require_latest: bool,
) -> _SegmentState:
    if not segments:
        raise ResourceLifecycleError("resource row precedes its segment start")
    segment_id = _validate_segment_id(row.get("segment_id"))
    segment_index = _validate_segment_index(row.get("segment_index"))
    if segment_index > len(segments):
        raise ResourceLifecycleError("resource row segment binding drifted")
    segment = segments[segment_index - 1]
    if segment_id != segment.segment_id:
        raise ResourceLifecycleError("resource row segment binding drifted")
    if require_latest and segment is not segments[-1]:
        raise ResourceLifecycleError(
            "resource sample row does not belong to the active segment"
        )
    return segment


def _attempt_key(row: Mapping[str, object]) -> tuple[str, int]:
    transaction_key = row.get("transaction_key")
    attempt = row.get("attempt")
    if (
        not isinstance(transaction_key, str)
        or not transaction_key
        or type(attempt) is not int
        or attempt <= 0
    ):
        raise ResourceLifecycleError("resource transaction identity drifted")
    return transaction_key, attempt


def _terminal_chain(
    segments: Sequence[_SegmentState],
    *,
    terminal_sample_count: int,
    terminal_peak_rss_bytes: int,
) -> list[dict[str, object]]:
    chain = [segment.summary() for segment in segments]
    chain[-1]["local_final_sample_count"] = terminal_sample_count
    chain[-1]["local_peak_rss_bytes"] = terminal_peak_rss_bytes
    return chain


def _expected_terminal(
    segments: Sequence[_SegmentState],
    *,
    prior_payload: bytes,
    terminal_resource: object,
) -> dict[str, object]:
    if not segments:
        raise ResourceLifecycleError("terminal resource segment is missing")
    active = segments[-1]
    root_pid, sample_count, rss_bytes, passed = _resource_fields(
        terminal_resource,
        label="terminal",
    )
    if root_pid != active.root_pid:
        raise ResourceLifecycleError("terminal must belong to the last segment")
    if sample_count <= active.local_final_sample_count:
        raise ResourceLifecycleError(
            "terminal resource sample must strictly advance"
        )
    if rss_bytes < active.local_peak_rss_bytes:
        raise ResourceLifecycleError("terminal resource RSS peak rolled back")
    if passed is not True:
        raise ResourceLifecycleError("terminal resource hard gate did not pass")
    chain = _terminal_chain(
        segments,
        terminal_sample_count=sample_count,
        terminal_peak_rss_bytes=rss_bytes,
    )
    run_peak = max(int(item["local_peak_rss_bytes"]) for item in chain)
    return {
        "schema_version": _TERMINAL_SCHEMA,
        "kind": "terminal_lifecycle",
        "transaction_key": "terminal:formal_backend",
        "phase": "terminal",
        "accepted": True,
        "measurement_boundary": _TERMINAL_BOUNDARY,
        "segment_id": active.segment_id,
        "segment_index": active.segment_index,
        "prior_resource_audit": _prefix_binding(prior_payload),
        "segment_chain": chain,
        "run_rss_lifecycle_peak_bytes": run_peak,
        "resource": dict(terminal_resource),  # type: ignore[arg-type]
    }


def _validate_rows(rows: Sequence[Mapping[str, object]]) -> _ValidatedLedger:
    segments: list[_SegmentState] = []
    attempts: dict[tuple[str, int], _AttemptState] = {}
    segment_ids: set[str] = set()
    prefix = bytearray()
    terminal_present = False

    for row_index, row_value in enumerate(rows):
        if not isinstance(row_value, Mapping):
            raise ResourceLifecycleError("resource lifecycle row drifted")
        row = dict(row_value)
        row_bytes = _canonical_row(row)
        phase = row.get("phase")
        if terminal_present:
            raise ResourceLifecycleError("row found after terminal resource evidence")

        if phase == "segment_start":
            segment = _validate_segment_descriptor(row)
            segment_id = str(segment["segment_id"])
            segment_index = int(segment["segment_index"])
            if segment_index != len(segments) + 1:
                raise ResourceLifecycleError("resource segment index chain has a gap")
            if segment_id in segment_ids:
                raise ResourceLifecycleError("resource segment id was reused")
            if segment["prior_resource_log"] != _prefix_binding(bytes(prefix)):
                raise ResourceLifecycleError(
                    "resource segment prior log binding drifted"
                )
            root_pid, sample_count, rss_bytes, _ = _resource_fields(
                segment["first_sample"],
                label="resource segment first",
            )
            start_identity = {
                "offset_bytes": len(prefix),
                "sha256": hashlib.sha256(row_bytes).hexdigest(),
                "size_bytes": len(row_bytes),
            }
            segments.append(
                _SegmentState(
                    segment_id=segment_id,
                    segment_index=segment_index,
                    root_pid=root_pid,
                    local_first_sample_count=sample_count,
                    local_final_sample_count=sample_count,
                    local_peak_rss_bytes=rss_bytes,
                    start_row_identity=start_identity,
                )
            )
            segment_ids.add(segment_id)
        elif phase in {"pre", "post", "accepted"}:
            active = _row_segment(
                row,
                segments,
                require_latest=phase in {"pre", "post"},
            )
            key = _attempt_key(row)
            state = attempts.get(key)
            if state is None:
                state = _AttemptState(active.segment_id, active.segment_index)
                attempts[key] = state
            if (
                state.segment_id != active.segment_id
                or state.segment_index != active.segment_index
            ):
                raise ResourceLifecycleError(
                    "resource transaction crossed a process segment"
                )
            if phase == "pre":
                if row.get("accepted") is not False or state.pre is not None:
                    raise ResourceLifecycleError("resource pre row drifted")
                state.pre = _advance_segment(
                    active,
                    row.get("resource"),
                    label="resource pre",
                )
            elif phase == "post":
                if (
                    row.get("accepted") is not False
                    or state.pre is None
                    or state.post is not None
                ):
                    raise ResourceLifecycleError("resource post row drifted")
                state.post = _advance_segment(
                    active,
                    row.get("resource"),
                    label="resource post",
                )
            else:
                if (
                    row.get("accepted") is not True
                    or state.accepted
                    or state.pre is None
                    or state.post is None
                ):
                    raise ResourceLifecycleError("resource acceptance row drifted")
                pre = row.get("pre")
                post = row.get("post")
                pre_root, pre_count, pre_peak, pre_passed = _resource_fields(
                    pre,
                    label="accepted pre",
                )
                post_root, post_count, post_peak, post_passed = _resource_fields(
                    post,
                    label="accepted post",
                )
                if (
                    pre_root != active.root_pid
                    or post_root != active.root_pid
                    or pre_count > post_count
                    or pre_peak > post_peak
                    or pre_passed is not True
                    or post_passed is not True
                    or dict(pre) != state.pre  # type: ignore[arg-type]
                    or dict(post) != state.post  # type: ignore[arg-type]
                    or post_count != active.local_final_sample_count
                    or post_peak != active.local_peak_rss_bytes
                ):
                    raise ResourceLifecycleError(
                        "resource acceptance segment binding drifted"
                    )
                state.accepted = True
        elif phase == "terminal":
            if row_index != len(rows) - 1:
                raise ResourceLifecycleError(
                    "row found after terminal resource evidence"
                )
            if row.get("schema_version") != _TERMINAL_SCHEMA:
                raise ResourceLifecycleError("terminal v4 schema drifted")
            if "preterminal_acceptance" not in row:
                raise ResourceLifecycleError(
                    "terminal v4 preterminal_acceptance receipt is missing"
                )
            receipt = _validate_binding(
                row.get("preterminal_acceptance"),
                label="terminal preterminal acceptance",
            )
            expected = _expected_terminal(
                segments,
                prior_payload=bytes(prefix),
                terminal_resource=row.get("resource"),
            )
            terminal_without_receipt = dict(row)
            del terminal_without_receipt["preterminal_acceptance"]
            if (
                set(row) != {*expected, "preterminal_acceptance"}
                or terminal_without_receipt != expected
                or row["preterminal_acceptance"] != receipt
            ):
                raise ResourceLifecycleError("terminal resource binding drifted")
            active = segments[-1]
            _, sample_count, rss_bytes, _ = _resource_fields(
                row["resource"],
                label="terminal",
            )
            active.local_final_sample_count = sample_count
            active.local_peak_rss_bytes = rss_bytes
            terminal_present = True
        else:
            raise ResourceLifecycleError("resource lifecycle phase drifted")
        prefix.extend(row_bytes)

    if not segments:
        raise ResourceLifecycleError("resource lifecycle segment is missing")
    chain = [segment.summary() for segment in segments]
    result = {
        "schema_version": _VALIDATION_SCHEMA,
        "passed": True,
        "segment_count": len(chain),
        "segment_chain": chain,
        "run_rss_lifecycle_peak_bytes": max(
            int(item["local_peak_rss_bytes"]) for item in chain
        ),
        "terminal_evidence_present": terminal_present,
        "active_segment_id": segments[-1].segment_id,
        "active_segment_index": segments[-1].segment_index,
        "active_root_pid": segments[-1].root_pid,
    }
    return _ValidatedLedger(
        result=result,
        segments=segments,
        payload=bytes(prefix),
        terminal_present=terminal_present,
    )


def _current_pid() -> int:
    return os.getpid()


def _new_segment_id() -> str:
    return uuid.uuid4().hex


def _append_exact(path: Path, *, prior_payload: bytes, row: Mapping[str, object]) -> None:
    row_bytes = _canonical_row(row)
    durable = DurableJsonl(path)
    try:
        durable.append(dict(row))
        committed = durable.recover_and_snapshot()
    except (DurableJsonlError, OSError) as exc:
        raise ResourceLifecycleError(
            "resource lifecycle durable append failed closed"
        ) from exc
    if committed != prior_payload + row_bytes:
        raise ResourceLifecycleError(
            "resource lifecycle append prefix or row identity drifted"
        )


def append_resource_segment_start(
    path: str | Path,
    *,
    first_sample: Mapping[str, object],
) -> dict[str, object]:
    """生成唯一 segment 并以 DurableJsonl 原子追加 start row。"""

    ledger_path = _ledger_path(path)
    prior_payload, rows = _read_canonical_ledger(ledger_path)
    prior: _ValidatedLedger | None = None
    if rows:
        prior = _validate_rows(rows)
        if prior.terminal_present:
            raise ResourceLifecycleError(
                "cannot append a resource segment after terminal evidence"
            )
    root_pid, _, _, _ = _resource_fields(
        first_sample,
        label="resource segment first",
    )
    if root_pid != _current_pid():
        raise ResourceLifecycleError(
            "resource segment root PID is not the current process"
        )
    segment_id = _new_segment_id()
    _validate_segment_id(segment_id)
    if prior is not None and any(
        item["segment_id"] == segment_id
        for item in prior.result["segment_chain"]  # type: ignore[union-attr]
    ):
        raise ResourceLifecycleError("resource segment id was reused")
    row = {
        "schema_version": _SEGMENT_SCHEMA,
        "kind": "resource_lifecycle_segment",
        "phase": "segment_start",
        "segment_id": segment_id,
        "segment_index": 1 if prior is None else len(prior.segments) + 1,
        "root_pid": root_pid,
        "first_sample": dict(first_sample),
        "prior_resource_log": _prefix_binding(prior_payload),
    }
    _append_exact(ledger_path, prior_payload=prior_payload, row=row)
    _, committed_rows = _read_canonical_ledger(ledger_path)
    _validate_rows(committed_rows)
    return row


def bind_resource_row_to_segment(
    row: Mapping[str, object],
    segment_start: Mapping[str, object],
) -> dict[str, object]:
    """纯函数：把一条 pre/post/accepted row 绑定到指定 segment。"""

    if not isinstance(row, Mapping):
        raise ResourceLifecycleError("resource row drifted")
    if "segment_id" in row or "segment_index" in row:
        raise ResourceLifecycleError("resource row already has a segment binding")
    segment = _validate_segment_descriptor(segment_start)
    phase = row.get("phase")
    if phase not in {"pre", "post", "accepted"}:
        raise ResourceLifecycleError("resource row phase cannot be segment-bound")
    _, first_count, first_peak, _ = _resource_fields(
        segment["first_sample"],
        label="resource segment first",
    )
    resources = (
        (row.get("resource"),)
        if phase in {"pre", "post"}
        else (row.get("pre"), row.get("post"))
    )
    lifecycle_values = [
        _resource_fields(resource, label=f"resource {phase}")
        for resource in resources
    ]
    if any(
        root_pid != segment["root_pid"]
        or sample_count < first_count
        or rss_bytes < first_peak
        for root_pid, sample_count, rss_bytes, _ in lifecycle_values
    ):
        raise ResourceLifecycleError("resource row segment binding drifted")
    if phase == "accepted":
        pre = lifecycle_values[0]
        post = lifecycle_values[1]
        if (
            pre[1] > post[1]
            or pre[2] > post[2]
            or pre[3] is not True
            or post[3] is not True
        ):
            raise ResourceLifecycleError("resource acceptance binding drifted")
    bound = {
        **dict(row),
        "segment_id": segment["segment_id"],
        "segment_index": segment["segment_index"],
    }
    _canonical_row(bound)
    return bound


def build_terminal_resource_evidence(
    rows: Sequence[Mapping[str, object]],
    *,
    terminal_resource: Mapping[str, object],
) -> dict[str, object]:
    """纯函数：由完整非 terminal prefix 构造 terminal evidence。"""

    validated = _validate_rows(rows)
    if validated.terminal_present:
        raise ResourceLifecycleError("terminal resource evidence already exists")
    return _expected_terminal(
        validated.segments,
        prior_payload=validated.payload,
        terminal_resource=terminal_resource,
    )


def build_bound_terminal_resource_evidence(
    rows: Sequence[Mapping[str, object]],
    *,
    terminal_resource: Mapping[str, object],
    preterminal_acceptance: Mapping[str, object],
) -> dict[str, object]:
    """构造只比 I2 terminal base 多出精确 receipt identity 的正式证据。"""

    terminal = build_terminal_resource_evidence(
        rows,
        terminal_resource=terminal_resource,
    )
    receipt = _validate_binding(
        preterminal_acceptance,
        label="terminal preterminal acceptance",
    )
    return {**terminal, "preterminal_acceptance": receipt}


def append_bound_resource_lifecycle_terminal(
    path: str | Path,
    *,
    terminal_resource: Mapping[str, object],
    preterminal_acceptance: Mapping[str, object],
) -> dict[str, object]:
    """原子追加 receipt-bound terminal，并以正式默认规则完整重放。"""

    ledger_path = _ledger_path(path)
    prior_payload, rows = _read_canonical_ledger(ledger_path)
    terminal = build_bound_terminal_resource_evidence(
        rows,
        terminal_resource=terminal_resource,
        preterminal_acceptance=preterminal_acceptance,
    )
    _append_exact(ledger_path, prior_payload=prior_payload, row=terminal)
    _, committed_rows = _read_canonical_ledger(ledger_path)
    validated = _validate_rows(committed_rows)
    if not validated.terminal_present:
        raise ResourceLifecycleError("terminal resource append was not committed")
    return terminal


def append_resource_lifecycle_terminal(
    path: str | Path,
    *,
    terminal_resource: Mapping[str, object],
    preterminal_acceptance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """构造、原子追加并重放验证唯一 terminal evidence。"""

    if preterminal_acceptance is None:
        raise ResourceLifecycleError(
            "formal terminal requires preterminal_acceptance receipt identity"
        )
    return append_bound_resource_lifecycle_terminal(
        path,
        terminal_resource=terminal_resource,
        preterminal_acceptance=preterminal_acceptance,
    )


def validate_resource_lifecycle_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    require_terminal: bool = True,
) -> dict[str, object]:
    """纯函数：重放 segment chain、同段单调性与 terminal 绑定。"""

    if type(require_terminal) is not bool:
        raise ResourceLifecycleError("terminal verification mode drifted")
    if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, Sequence):
        raise ResourceLifecycleError("resource lifecycle rows drifted")
    validated = _validate_rows(rows)
    if require_terminal and not validated.terminal_present:
        raise ResourceLifecycleError("terminal resource evidence is missing")
    return dict(validated.result)


def validate_resource_lifecycle_ledger(
    path: str | Path,
    *,
    require_terminal: bool = True,
) -> dict[str, object]:
    """读取 canonical ledger 后执行完整 lifecycle 重放校验。"""

    if type(require_terminal) is not bool:
        raise ResourceLifecycleError("terminal verification mode drifted")
    ledger_path = _ledger_path(path)
    _, rows = _read_canonical_ledger(ledger_path)
    return validate_resource_lifecycle_rows(rows, require_terminal=require_terminal)


__all__ = [
    "ResourceLifecycleError",
    "append_bound_resource_lifecycle_terminal",
    "append_resource_lifecycle_terminal",
    "append_resource_segment_start",
    "bind_resource_row_to_segment",
    "build_bound_terminal_resource_evidence",
    "build_terminal_resource_evidence",
    "validate_resource_lifecycle_ledger",
    "validate_resource_lifecycle_rows",
]
