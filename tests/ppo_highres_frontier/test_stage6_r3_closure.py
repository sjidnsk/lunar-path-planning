from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from scripts.finalize_ppo_stage6_r3 import (
    EXPECTED_FINAL_KEYS,
    Stage6R3ClosureError,
    build_standard_replay_artifacts,
    capture_closure_input_manifest,
    publish_closure_artifacts,
    require_canonical_closure_paths,
    validate_final_summary_schedule,
    verify_checkpoint_member_bindings,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _binding(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def test_verify_checkpoint_member_bindings_rejects_member_drift(
    tmp_path: Path,
) -> None:
    bundle = (
        tmp_path
        / "checkpoints"
        / "seed-20260716"
        / "update-00000085"
    )
    bundle.mkdir(parents=True)
    checkpoint = bundle / "checkpoint.pt"
    manifest = bundle / "manifest.json"
    complete = bundle / "complete.json"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_bytes(b"manifest")
    complete.write_bytes(b"complete")
    latest = {
        "update": 85,
        "checkpoint_sha256": _binding(checkpoint)["sha256"],
        "manifest_sha256": _binding(manifest)["sha256"],
        "complete_marker_sha256": _binding(complete)["sha256"],
        "members": {
            "checkpoint.pt": _binding(checkpoint),
            "manifest.json": _binding(manifest),
            "complete.json": _binding(complete),
        },
    }

    assert verify_checkpoint_member_bindings(latest, tmp_path) == (
        checkpoint.resolve(),
        manifest.resolve(),
        complete.resolve(),
    )

    manifest.write_bytes(b"drifted")
    with pytest.raises(Stage6R3ClosureError, match="member"):
        verify_checkpoint_member_bindings(latest, tmp_path)


def test_require_canonical_closure_paths_rejects_substitution(
    tmp_path: Path,
) -> None:
    from scripts.finalize_ppo_stage6_r3 import (
        DEFAULT_EVAL_ROOT,
        DEFAULT_STAGE_ROOT,
        DEFAULT_STDERR_LOG,
    )

    require_canonical_closure_paths(
        DEFAULT_STAGE_ROOT,
        DEFAULT_EVAL_ROOT,
        DEFAULT_STDERR_LOG,
    )
    with pytest.raises(Stage6R3ClosureError, match="canonical input path"):
        require_canonical_closure_paths(
            tmp_path / "s6",
            DEFAULT_EVAL_ROOT,
            DEFAULT_STDERR_LOG,
        )


def test_capture_closure_input_manifest_detects_file_set_growth(
    tmp_path: Path,
) -> None:
    stage_root = tmp_path / "stage"
    eval_root = tmp_path / "eval"
    stage_root.mkdir()
    eval_root.mkdir()
    (stage_root / "a.json").write_bytes(b"{}")
    (eval_root / "b.json").write_bytes(b"{}")
    stderr_log = tmp_path / "stderr.log"
    stderr_log.write_bytes(b"")

    before = capture_closure_input_manifest(stage_root, eval_root, stderr_log)
    (eval_root / "new.json").write_bytes(b"{}")
    after = capture_closure_input_manifest(stage_root, eval_root, stderr_log)

    assert before != after
    assert [row["path"] for row in after] == sorted(
        [
            (stage_root / "a.json").resolve().as_posix(),
            (eval_root / "b.json").resolve().as_posix(),
            (eval_root / "new.json").resolve().as_posix(),
            stderr_log.resolve().as_posix(),
        ]
    )


def _final_summary() -> dict[str, object]:
    results: dict[str, object] = {}
    for key in EXPECTED_FINAL_KEYS:
        split, method = key.split(":", maxsplit=1)
        results[key] = {
            "schema_version": "stage6_r3_evaluation_result/v1",
            "split": split,
            "method": method,
            "episode_count": 64,
            "update": 80,
            "checkpoint_sha256": SHA_C,
            "policy_state_sha256": SHA_D,
            "config_sha256": SHA_B,
            "source_set_sha256": SHA_A,
            "trace_path": f"final/{split}-{method}-64.jsonl",
            "trace_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
            "trace_line_count": 64,
            "wall_seconds": 1.0,
            "result": {
                "metrics": {
                    "safety_violation_count": 0,
                    "invalid_action_count_mean": 0.0,
                    "planner_failure_count_mean": 0.0,
                },
                "bootstrap_audit": {},
                "fairness_audit": {},
            },
            "resources": [
                {
                    "boundary": "evaluation:post",
                    "snapshot": {},
                    "decision": {"hard_stops": [], "warnings": []},
                }
            ],
        }
    return {
        "schema_version": "stage6_r3_final_summary/v1",
        "completed_at_utc": "2026-08-01T00:00:00Z",
        "global_best": {
            "record": {"update": 80},
            "checkpoint_sha256": SHA_C,
            "policy_state_sha256": SHA_D,
            "source_set_sha256": SHA_A,
        },
        "evaluation_count": 10,
        "episode_count": 640,
        "source_set_sha256": SHA_A,
        "config_sha256": SHA_B,
        "results": results,
    }


def test_validate_final_summary_requires_exact_ten_group_schedule() -> None:
    summary = _final_summary()
    summary["results"] = dict(  # type: ignore[assignment]
        reversed(list(summary["results"].items()))  # type: ignore[union-attr]
    )

    assert validate_final_summary_schedule(
        summary,
        expected_source_set_sha256=SHA_A,
        expected_config_sha256=SHA_B,
        expected_checkpoint_sha256=SHA_C,
        expected_policy_state_sha256=SHA_D,
        expected_update=80,
    ) == EXPECTED_FINAL_KEYS

    del summary["results"][EXPECTED_FINAL_KEYS[-1]]  # type: ignore[index]
    with pytest.raises(Stage6R3ClosureError, match="schedule"):
        validate_final_summary_schedule(
            summary,
            expected_source_set_sha256=SHA_A,
            expected_config_sha256=SHA_B,
            expected_checkpoint_sha256=SHA_C,
            expected_policy_state_sha256=SHA_D,
            expected_update=80,
        )


def test_validate_final_summary_rejects_unaccepted_resource_snapshot() -> None:
    summary = _final_summary()
    first = summary["results"][EXPECTED_FINAL_KEYS[0]]  # type: ignore[index]
    first["resources"][0]["decision"]["hard_stops"] = ["rss"]  # type: ignore[index]
    first["resources"][0]["decision"]["passed"] = True  # type: ignore[index]

    with pytest.raises(Stage6R3ClosureError, match="resource"):
        validate_final_summary_schedule(
            summary,
            expected_source_set_sha256=SHA_A,
            expected_config_sha256=SHA_B,
            expected_checkpoint_sha256=SHA_C,
            expected_policy_state_sha256=SHA_D,
            expected_update=80,
        )


def test_validate_final_summary_rejects_nonzero_safety_metric() -> None:
    summary = _final_summary()
    first = summary["results"][EXPECTED_FINAL_KEYS[0]]  # type: ignore[index]
    first["result"]["metrics"]["safety_violation_count"] = 1  # type: ignore[index]

    with pytest.raises(Stage6R3ClosureError, match="safety"):
        validate_final_summary_schedule(
            summary,
            expected_source_set_sha256=SHA_A,
            expected_config_sha256=SHA_B,
            expected_checkpoint_sha256=SHA_C,
            expected_policy_state_sha256=SHA_D,
            expected_update=80,
        )


def test_build_standard_replay_artifacts_are_canonical_and_bound() -> None:
    trace = b'{"episode":0}\n'
    result = {
        "metrics": {"success_rate_under_fixed_step_budget": 1.0},
        "bootstrap_audit": {"resamples": 10},
        "fairness_audit": {"passed": True},
    }

    replay = build_standard_replay_artifacts(
        trace_bytes=trace,
        trace_name="test-ppo_policy-64.jsonl",
        split="test",
        method="ppo_policy",
        config_sha256=SHA_B,
        checkpoint_sha256=SHA_C,
        policy_state_sha256=SHA_D,
        result=result,
    )

    completion = json.loads(replay["summary_bytes"])
    commit = json.loads(replay["commit_bytes"])
    assert replay["summary_bytes"] == ArtifactStore.canonical_json_bytes(completion)
    assert replay["commit_bytes"] == ArtifactStore.canonical_json_bytes(commit)
    assert completion["episode_count"] == 64
    assert completion["result"]["episode_count"] == 64
    assert completion["trace"] == {
        "path": "test-ppo_policy-64.jsonl",
        "sha256": hashlib.sha256(trace).hexdigest(),
        "size_bytes": len(trace),
    }
    assert commit["summary"]["sha256"] == hashlib.sha256(
        replay["summary_bytes"]
    ).hexdigest()


def test_publish_closure_artifacts_is_write_once(tmp_path: Path) -> None:
    payload = {
        "schema_version": "stage6_r3_machine_acceptance/v1",
        "status": "passed",
        "closure_state": "machine_passed_awaiting_independent_review",
        "training": {
            "formal_run_id": "formal-run",
            "accepted_update_count": 26,
            "validation_updates": [80, 90, 100],
        },
        "final_evaluation": {
            "selected_update": 80,
            "evaluation_count": 10,
            "episode_count": 640,
            "metrics": {
                "test:ppo_policy": {
                    "success_rate_under_fixed_step_budget": 1.0,
                    "mean_final_coverage": 0.99,
                },
                "unseen:ppo_policy": {
                    "success_rate_under_fixed_step_budget": 1.0,
                    "mean_final_coverage": 0.98,
                },
            },
        },
    }

    paths = publish_closure_artifacts(tmp_path / "closure", payload)

    assert paths == (
        tmp_path / "closure" / "machine-acceptance.json",
        tmp_path / "closure" / "report.md",
    )
    assert json.loads(paths[0].read_text(encoding="utf-8")) == payload
    assert "machine_passed_awaiting_independent_review" in paths[1].read_text(
        encoding="utf-8"
    )
    with pytest.raises(FileExistsError):
        publish_closure_artifacts(tmp_path / "closure", payload)
