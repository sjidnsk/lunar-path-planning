import json
from pathlib import Path

import pytest

from scripts.query_xunce_training_progress import build_progress_snapshot, render_plain_progress


def test_stage26_8m_snapshot_uses_job_state_rows(tmp_path: Path) -> None:
    output_root = tmp_path / "s26_8m"
    output_root.mkdir()
    _write_json(
        output_root / "xunce-stage26-8m-summary.json",
        {
            "schema_version": "xunce-stage26-8m-summary/v1",
            "stage_id": "xunce-stage26-8m-generalized-resumable-training-pipeline",
            "status": "passed",
            "next_required_change": "continue_stage26_8m_jobs",
            "job_count": 1,
            "completed_job_count": 0,
            "pending_job_count": 1,
            "failed_job_count": 0,
            "next_job_id": "h16_s260801_sc3_cr16_er16_u_policy",
            "next_phase": "eval_pre",
        },
    )
    _write_jsonl(
        output_root / "xunce-stage26-8m-job-state.jsonl",
        [
            {"job_id": "h16_s260801_sc3_cr16_er16_u_policy", "phase": "collector", "status": "complete"},
            {"job_id": "h16_s260801_sc3_cr16_er16_u_policy", "phase": "update", "status": "complete"},
            {
                "job_id": "h16_s260801_sc3_cr16_er16_u_policy",
                "phase": "eval_pre",
                "status": "pending",
                "resume_decision": "run_next_pending_phase",
            },
            {"job_id": "h16_s260801_sc3_cr16_er16_u_policy", "phase": "eval_post", "status": "pending"},
            {"job_id": "h16_s260801_sc3_cr16_er16_u_policy", "phase": "aggregate", "status": "pending"},
        ],
    )

    snapshot = build_progress_snapshot(output_root)

    assert snapshot["schema_version"] == "xunce-training-progress-query/v1"
    assert snapshot["stage_kind"] == "stage26_8m"
    assert snapshot["progress_status"] == "in_progress"
    assert snapshot["unit"] == "phase"
    assert snapshot["current"] == 2
    assert snapshot["total"] == 5
    assert snapshot["percent"] == 40.0
    assert snapshot["next"]["job_id"] == "h16_s260801_sc3_cr16_er16_u_policy"
    assert snapshot["next"]["phase"] == "eval_pre"
    assert snapshot["counts"]["job_count"] == 1
    assert snapshot["counts"]["pending_job_count"] == 1
    assert snapshot["read_errors"] == []
    assert "stage26_8m" in render_plain_progress(snapshot)


def test_stage26_9_snapshot_summarizes_subjob_matrix(tmp_path: Path) -> None:
    output_root = tmp_path / "s26_9"
    output_root.mkdir()
    _write_json(
        output_root / "xunce-stage26-9-summary.json",
        {
            "schema_version": "xunce-stage26-9-summary/v1",
            "stage_id": "xunce-stage26-9-synthetic-terrain-long-horizon-efficiency-pilot",
            "status": "passed",
            "next_required_change": "continue_stage26_9_long_horizon_jobs",
            "completed_subjob_count": 1,
            "pending_subjob_count": 1,
            "failed_subjob_count": 0,
        },
    )
    _write_jsonl(
        output_root / "xunce-stage26-9-long-horizon-matrix.jsonl",
        [
            {"horizon": 20, "seed": 260801, "status": "complete", "completed_job_count": 1},
            {"horizon": 24, "seed": 260802, "status": "pending", "pending_job_count": 1},
        ],
    )

    snapshot = build_progress_snapshot(output_root)

    assert snapshot["stage_kind"] == "stage26_9"
    assert snapshot["unit"] == "subjob"
    assert snapshot["current"] == 1
    assert snapshot["total"] == 2
    assert snapshot["percent"] == 50.0
    assert snapshot["next"]["horizon"] == 24
    assert snapshot["next"]["seed"] == 260802


def test_stage26_9_snapshot_falls_back_to_summary_counts_when_matrix_is_missing(tmp_path: Path) -> None:
    output_root = tmp_path / "s26_9"
    output_root.mkdir()
    _write_json(
        output_root / "xunce-stage26-9-summary.json",
        {
            "schema_version": "xunce-stage26-9-summary/v1",
            "stage_id": "xunce-stage26-9-synthetic-terrain-long-horizon-efficiency-pilot",
            "status": "passed",
            "next_required_change": "continue_stage26_9_long_horizon_jobs",
            "completed_subjob_count": 2,
            "pending_subjob_count": 1,
            "failed_subjob_count": 0,
        },
    )

    snapshot = build_progress_snapshot(output_root)

    assert snapshot["stage_kind"] == "stage26_9"
    assert snapshot["progress_status"] == "in_progress"
    assert snapshot["current"] == 2
    assert snapshot["total"] == 3
    assert snapshot["percent"] == pytest.approx(66.667)


def test_stage26_10a_snapshot_tracks_offline_and_topk_work(tmp_path: Path) -> None:
    output_root = tmp_path / "s26_10a"
    output_root.mkdir()
    _write_json(
        output_root / "xunce-stage26-10a-summary.json",
        {
            "schema_version": "xunce-stage26-10a-terminal-aware-reward-weight-sweep-summary/v1",
            "stage_id": "xunce-stage26-10a-terminal-aware-reward-weight-sweep",
            "status": "passed",
            "next_required_change": "continue_stage26_10a_topk_bounded_eval",
            "profile_count": 3,
            "offline_passed_count": 3,
            "top_k": 2,
            "top_k_profile_ids": ["p1", "p2"],
        },
    )
    _write_jsonl(
        output_root / "xunce-stage26-10a-reward-weight-sweep-matrix.jsonl",
        [
            {"profile_id": "p1", "offline_status": "passed", "stage26_10_status": "passed"},
            {"profile_id": "p2", "offline_status": "passed", "stage26_10_status": "pending"},
            {"profile_id": "p3", "offline_status": "passed"},
        ],
    )

    snapshot = build_progress_snapshot(output_root)

    assert snapshot["stage_kind"] == "stage26_10a"
    assert snapshot["unit"] == "profile_or_topk_eval"
    assert snapshot["current"] == 4
    assert snapshot["total"] == 5
    assert snapshot["percent"] == 80.0
    assert snapshot["next"]["profile_id"] == "p2"
    assert snapshot["counts"]["topk_pending_count"] == 1


def test_telemetry_snapshot_handles_unfinished_event_stream(tmp_path: Path) -> None:
    output_root = tmp_path / "telemetry"
    output_root.mkdir()
    _write_jsonl(
        output_root / "training-progress-events.jsonl",
        [
            {
                "schema_version": "training-progress-event/v1",
                "run_id": "run-1",
                "stage": "ppo_update",
                "status": "start",
                "current": 0,
                "total": 4,
                "metrics": {},
            },
            {
                "schema_version": "training-progress-event/v1",
                "run_id": "run-1",
                "stage": "ppo_update",
                "status": "progress",
                "current": 2,
                "total": 4,
                "metrics": {"loss": 1.5},
            },
        ],
    )

    snapshot = build_progress_snapshot(output_root)

    assert snapshot["stage_kind"] == "training_progress"
    assert snapshot["progress_status"] == "running"
    assert snapshot["current"] == 2
    assert snapshot["total"] == 4
    assert snapshot["last_event"]["stage"] == "ppo_update"
    assert snapshot["last_event"]["metrics"]["loss"] == 1.5


def test_plain_renderer_shows_ascii_progress_bar() -> None:
    text = render_plain_progress(
        {
            "stage_kind": "stage26_9",
            "progress_status": "in_progress",
            "status": "passed",
            "next_required_change": "continue_stage26_9_long_horizon_jobs",
            "current": 1,
            "total": 2,
            "percent": 50.0,
            "unit": "subjob",
            "counts": {},
            "next": {},
            "read_errors": [],
        }
    )

    assert "progress: [##########----------] 50.0% (1/2 subjob)" in text


def test_query_reports_read_errors_instead_of_crashing(tmp_path: Path) -> None:
    output_root = tmp_path / "broken"
    output_root.mkdir()
    (output_root / "xunce-stage26-8m-summary.json").write_text("{bad json", encoding="utf-8")

    snapshot = build_progress_snapshot(output_root)

    assert snapshot["stage_kind"] == "stage26_8m"
    assert snapshot["progress_status"] == "unknown"
    assert snapshot["read_errors"]
    assert "unreadable" in snapshot["read_errors"][0]["reason"]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
