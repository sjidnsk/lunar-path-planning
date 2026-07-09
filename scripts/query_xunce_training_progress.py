from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import xunce_artifact_io as artifact_io


SCHEMA_VERSION = "xunce-training-progress-query/v1"

STAGE26_8M_SUMMARY = "xunce-stage26-8m-summary.json"
STAGE26_8M_JOB_STATE = "xunce-stage26-8m-job-state.jsonl"

STAGE26_9_SUMMARY = "xunce-stage26-9-summary.json"
STAGE26_9_MATRIX = "xunce-stage26-9-long-horizon-matrix.jsonl"

STAGE26_10A_SUMMARY = "xunce-stage26-10a-summary.json"
STAGE26_10A_MATRIX = "xunce-stage26-10a-reward-weight-sweep-matrix.jsonl"

TRAINING_PROGRESS_EVENTS = "training-progress-events.jsonl"
TRAINING_PROGRESS_SUMMARY = "training-progress-summary.json"
STAGE26_8M_PHASE_COUNT = 5


def build_progress_snapshot(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    read_errors: list[dict[str, Any]] = []
    stage_kind = _detect_stage_kind(root)
    if stage_kind == "stage26_8m":
        snapshot = _stage26_8m_snapshot(root, read_errors)
    elif stage_kind == "stage26_9":
        snapshot = _stage26_9_snapshot(root, read_errors)
    elif stage_kind == "stage26_10a":
        snapshot = _stage26_10a_snapshot(root, read_errors)
    elif stage_kind == "training_progress":
        snapshot = _training_progress_snapshot(root, read_errors)
    else:
        snapshot = _base_snapshot(root, "unknown", read_errors)
        snapshot.update(
            {
                "progress_status": "unknown",
                "status": None,
                "next_required_change": None,
                "unit": "artifact",
                "current": 0,
                "total": 0,
                "percent": None,
                "counts": {},
                "next": {},
            }
        )

    if stage_kind != "training_progress":
        events = _read_jsonl_if_exists(root / TRAINING_PROGRESS_EVENTS, read_errors, "training_progress_events")
        summary = _read_json_if_exists(root / TRAINING_PROGRESS_SUMMARY, read_errors, "training_progress_summary")
        if events or summary:
            snapshot["telemetry"] = _telemetry_overlay(events, summary, root)
    return snapshot


def render_plain_progress(snapshot: dict[str, Any]) -> str:
    current = snapshot.get("current")
    total = snapshot.get("total")
    percent = snapshot.get("percent")
    progress = _plain_progress(current, total, percent, snapshot.get("unit"))
    lines = [
        f"stage_kind: {snapshot.get('stage_kind')}",
        f"progress_status: {snapshot.get('progress_status')}",
        f"summary_status: {snapshot.get('status')}",
        f"next_required_change: {snapshot.get('next_required_change')}",
        f"progress: {progress}",
    ]
    next_item = snapshot.get("next") if isinstance(snapshot.get("next"), dict) else {}
    if next_item:
        lines.append("next: " + json.dumps(next_item, ensure_ascii=False, sort_keys=True))
    counts = snapshot.get("counts") if isinstance(snapshot.get("counts"), dict) else {}
    if counts:
        lines.append("counts: " + json.dumps(counts, ensure_ascii=False, sort_keys=True))
    telemetry = snapshot.get("telemetry") if isinstance(snapshot.get("telemetry"), dict) else {}
    if telemetry:
        lines.append("last_event: " + json.dumps(telemetry.get("last_event", {}), ensure_ascii=False, sort_keys=True))
    read_errors = snapshot.get("read_errors") if isinstance(snapshot.get("read_errors"), list) else []
    if read_errors:
        lines.append("read_errors: " + json.dumps(read_errors, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines) + "\n"


def _plain_progress(current: Any, total: Any, percent: Any, unit: Any) -> str:
    current_int = _safe_int(current)
    total_int = _safe_int(total)
    unit_text = str(unit or "unit")
    if percent is None:
        return f"{current_int}/{total_int} {unit_text}"
    percent_float = float(percent)
    return f"{_progress_bar(percent_float)} {percent_float:.1f}% ({current_int}/{total_int} {unit_text})"


def _progress_bar(percent: float, *, width: int = 20) -> str:
    clamped = max(0.0, min(100.0, float(percent)))
    filled = int(round((clamped / 100.0) * width))
    filled = max(0, min(width, filled))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Query Xunce training progress artifacts without advancing a run.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--format", choices=("plain", "json"), default="plain")
    parser.add_argument("--strict", action="store_true", help="return non-zero when progress artifacts are unreadable")
    args = parser.parse_args(argv)

    snapshot = build_progress_snapshot(Path(args.output_root))
    if args.format == "json":
        print(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        sys.stdout.write(render_plain_progress(snapshot))
    return 2 if args.strict and snapshot.get("read_errors") else 0


def _base_snapshot(root: Path, stage_kind: str, read_errors: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "output_root": str(root),
        "stage_kind": stage_kind,
        "read_errors": read_errors,
    }


def _detect_stage_kind(root: Path) -> str:
    if artifact_io.path_is_file(root / STAGE26_8M_SUMMARY) or artifact_io.path_is_file(root / STAGE26_8M_JOB_STATE):
        return "stage26_8m"
    if artifact_io.path_is_file(root / STAGE26_9_SUMMARY) or artifact_io.path_is_file(root / STAGE26_9_MATRIX):
        return "stage26_9"
    if artifact_io.path_is_file(root / STAGE26_10A_SUMMARY) or artifact_io.path_is_file(root / STAGE26_10A_MATRIX):
        return "stage26_10a"
    if artifact_io.path_is_file(root / TRAINING_PROGRESS_EVENTS) or artifact_io.path_is_file(root / TRAINING_PROGRESS_SUMMARY):
        return "training_progress"
    return "unknown"


def _stage26_8m_snapshot(root: Path, read_errors: list[dict[str, Any]]) -> dict[str, Any]:
    summary_path = root / STAGE26_8M_SUMMARY
    state_path = root / STAGE26_8M_JOB_STATE
    summary = _read_json_if_exists(summary_path, read_errors, "stage26_8m_summary")
    rows = _read_jsonl_if_exists(state_path, read_errors, "stage26_8m_job_state")
    if rows:
        completed = _count_rows(rows, "complete")
        pending = _count_rows(rows, "pending")
        failed = _count_rows(rows, "failed")
        total = len(rows)
    else:
        completed = _safe_int(summary.get("completed_job_count")) * STAGE26_8M_PHASE_COUNT
        pending = _safe_int(summary.get("pending_job_count")) * STAGE26_8M_PHASE_COUNT
        failed = _safe_int(summary.get("failed_job_count")) * STAGE26_8M_PHASE_COUNT
        total = _safe_int(summary.get("job_count")) * STAGE26_8M_PHASE_COUNT
    current = completed
    active = _first_pending(rows, prefer_runnable=True)
    if not active:
        active = _summary_next_item(summary)
    snapshot = _base_snapshot(root, "stage26_8m", read_errors)
    snapshot.update(
        {
            "progress_status": _status_from_counts(current, total, pending, failed, read_errors),
            "status": summary.get("status"),
            "next_required_change": summary.get("next_required_change"),
            "unit": "phase",
            "current": current,
            "total": total,
            "percent": _percent(current, total),
            "counts": {
                "phase_count": total,
                "completed_phase_count": completed,
                "pending_phase_count": pending,
                "failed_phase_count": failed,
                "job_count": int(summary.get("job_count") or len({str(row.get("job_id")) for row in rows if row.get("job_id")})),
                "completed_job_count": int(summary.get("completed_job_count") or 0),
                "pending_job_count": int(summary.get("pending_job_count") or 0),
                "failed_job_count": int(summary.get("failed_job_count") or 0),
            },
            "next": active,
            "summary_path": str(summary_path),
            "state_path": str(state_path),
        }
    )
    return snapshot


def _stage26_9_snapshot(root: Path, read_errors: list[dict[str, Any]]) -> dict[str, Any]:
    summary_path = root / STAGE26_9_SUMMARY
    matrix_path = root / STAGE26_9_MATRIX
    summary = _read_json_if_exists(summary_path, read_errors, "stage26_9_summary")
    rows = _read_jsonl_if_exists(matrix_path, read_errors, "stage26_9_matrix")
    if rows:
        completed = _count_rows(rows, "complete")
        pending = _count_rows(rows, "pending")
        failed = _count_rows(rows, "failed")
        total = len(rows)
    else:
        completed = _safe_int(summary.get("completed_subjob_count"))
        pending = _safe_int(summary.get("pending_subjob_count"))
        failed = _safe_int(summary.get("failed_subjob_count"))
        total = _summary_total(summary, "completed_subjob_count", "pending_subjob_count", "failed_subjob_count")
    current = completed
    active = _first_pending(rows)
    snapshot = _base_snapshot(root, "stage26_9", read_errors)
    snapshot.update(
        {
            "progress_status": _status_from_counts(current, total, pending, failed, read_errors),
            "status": summary.get("status"),
            "next_required_change": summary.get("next_required_change"),
            "unit": "subjob",
            "current": current,
            "total": total,
            "percent": _percent(current, total),
            "counts": {
                "subjob_count": total,
                "completed_subjob_count": int(summary.get("completed_subjob_count") or completed),
                "pending_subjob_count": int(summary.get("pending_subjob_count") or pending),
                "failed_subjob_count": int(summary.get("failed_subjob_count") or failed),
            },
            "next": _stage26_9_next(active),
            "summary_path": str(summary_path),
            "matrix_path": str(matrix_path),
        }
    )
    return snapshot


def _stage26_10a_snapshot(root: Path, read_errors: list[dict[str, Any]]) -> dict[str, Any]:
    summary_path = root / STAGE26_10A_SUMMARY
    matrix_path = root / STAGE26_10A_MATRIX
    summary = _read_json_if_exists(summary_path, read_errors, "stage26_10a_summary")
    rows = _read_jsonl_if_exists(matrix_path, read_errors, "stage26_10a_matrix")
    top_ids = {str(item) for item in summary.get("top_k_profile_ids") or []}
    top_rows = [row for row in rows if str(row.get("profile_id")) in top_ids]
    offline_passed = sum(1 for row in rows if row.get("offline_status") == "passed")
    offline_failed = sum(1 for row in rows if row.get("offline_status") == "failed")
    topk_complete = sum(1 for row in top_rows if row.get("stage26_10_status") == "passed")
    topk_failed = sum(1 for row in top_rows if row.get("stage26_10_status") == "failed")
    topk_pending = sum(1 for row in top_rows if row.get("stage26_10_status") in {None, "", "pending"})
    profile_count = int(summary.get("profile_count") or len(rows))
    top_k = int(summary.get("top_k") or len(top_ids))
    if not rows:
        offline_passed = _safe_int(summary.get("offline_passed_count"))
        topk_pending = top_k
    total = profile_count + top_k
    current = offline_passed + offline_failed + topk_complete + topk_failed
    pending = max(0, total - current)
    failed = offline_failed + topk_failed
    active = _first_topk_pending(top_rows)
    snapshot = _base_snapshot(root, "stage26_10a", read_errors)
    snapshot.update(
        {
            "progress_status": _status_from_counts(current, total, pending, failed, read_errors),
            "status": summary.get("status"),
            "next_required_change": summary.get("next_required_change"),
            "unit": "profile_or_topk_eval",
            "current": current,
            "total": total,
            "percent": _percent(current, total),
            "counts": {
                "profile_count": profile_count,
                "offline_passed_count": offline_passed if rows else int(summary.get("offline_passed_count") or 0),
                "offline_failed_count": offline_failed,
                "top_k": top_k,
                "topk_completed_count": topk_complete,
                "topk_pending_count": topk_pending if top_rows else top_k,
                "topk_failed_count": topk_failed,
            },
            "next": _stage26_10a_next(active),
            "summary_path": str(summary_path),
            "matrix_path": str(matrix_path),
        }
    )
    return snapshot


def _training_progress_snapshot(root: Path, read_errors: list[dict[str, Any]]) -> dict[str, Any]:
    events_path = root / TRAINING_PROGRESS_EVENTS
    summary_path = root / TRAINING_PROGRESS_SUMMARY
    events = _read_jsonl_if_exists(events_path, read_errors, "training_progress_events")
    summary = _read_json_if_exists(summary_path, read_errors, "training_progress_summary")
    overlay = _telemetry_overlay(events, summary, root)
    snapshot = _base_snapshot(root, "training_progress", read_errors)
    snapshot.update(
        {
            "progress_status": overlay["progress_status"],
            "status": summary.get("status"),
            "next_required_change": None,
            "unit": "event",
            "current": overlay["current"],
            "total": overlay["total"],
            "percent": _percent(overlay["current"], overlay["total"]),
            "counts": {"event_count": len(events)},
            "next": {},
            "last_event": overlay["last_event"],
            "summary_path": str(summary_path),
            "events_path": str(events_path),
        }
    )
    return snapshot


def _telemetry_overlay(events: list[dict[str, Any]], summary: dict[str, Any], root: Path) -> dict[str, Any]:
    last_event = events[-1] if events else {}
    current = _safe_int(last_event.get("current"))
    total = _safe_int(last_event.get("total"))
    last_status = str(last_event.get("status") or summary.get("last_status") or summary.get("status") or "")
    if last_status in {"start", "progress"}:
        progress_status = "running"
    elif last_status == "failed":
        progress_status = "failed"
    elif last_status == "passed":
        progress_status = "complete"
    else:
        progress_status = "unknown"
    return {
        "progress_status": progress_status,
        "current": current,
        "total": total,
        "last_event": last_event,
        "summary": summary,
        "events_path": str(root / TRAINING_PROGRESS_EVENTS),
        "summary_path": str(root / TRAINING_PROGRESS_SUMMARY),
    }


def _read_json_if_exists(path: Path, read_errors: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if not artifact_io.path_is_file(path):
        return {}
    try:
        return artifact_io.read_json(path)
    except Exception as exc:  # noqa: BLE001 - query must survive partial writes.
        read_errors.append({"path": str(path), "reason": f"{label}_unreadable", "detail": str(exc)})
        return {}


def _read_jsonl_if_exists(path: Path, read_errors: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    if not artifact_io.path_is_file(path):
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = artifact_io.read_text(path).splitlines()
    except Exception as exc:  # noqa: BLE001
        read_errors.append({"path": str(path), "reason": f"{label}_unreadable", "detail": str(exc)})
        return []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            read_errors.append(
                {"path": str(path), "line": index, "reason": f"{label}_unreadable", "detail": str(exc)}
            )
            return []
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _status_from_counts(
    current: int,
    total: int,
    pending: int,
    failed: int,
    read_errors: list[dict[str, Any]],
) -> str:
    if read_errors and total == 0:
        return "unknown"
    if failed > 0:
        return "failed"
    if pending > 0:
        return "in_progress"
    if total > 0 and current >= total:
        return "complete"
    return "unknown"


def _percent(current: int, total: int) -> float | None:
    if total <= 0:
        return None
    return round((float(current) / float(total)) * 100.0, 3)


def _count_rows(rows: list[dict[str, Any]], status: str) -> int:
    return sum(1 for row in rows if row.get("status") == status)


def _summary_total(summary: dict[str, Any], completed_field: str, pending_field: str, failed_field: str) -> int:
    return _safe_int(summary.get(completed_field)) + _safe_int(summary.get(pending_field)) + _safe_int(summary.get(failed_field))


def _first_pending(rows: list[dict[str, Any]], *, prefer_runnable: bool = False) -> dict[str, Any]:
    pending = [row for row in rows if row.get("status") == "pending"]
    if prefer_runnable:
        runnable = [row for row in pending if row.get("resume_decision") == "run_next_pending_phase"]
        if runnable:
            return dict(runnable[0])
    return dict(pending[0]) if pending else {}


def _summary_next_item(summary: dict[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = {}
    if summary.get("next_job_id"):
        item["job_id"] = summary.get("next_job_id")
    if summary.get("next_phase"):
        item["phase"] = summary.get("next_phase")
    return item


def _stage26_9_next(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in ("horizon", "seed", "status", "sub_output_root", "stage26_8m_next_required_change")
        if key in row
    }


def _first_topk_pending(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if row.get("stage26_10_status") in {None, "", "pending"}:
            return dict(row)
    return {}


def _stage26_10a_next(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in ("profile_id", "stage26_10_status", "stage26_10_next_required_change")
        if key in row
    }


def _safe_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
