from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import math
import ast
import tomllib
import shutil
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import numpy as np
from pydantic import ValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE1_CONFIG = REPO_ROOT / "configs" / "ppo_highres_frontier_smoke_v1.json"
R1_REVIEW_REPORT = REPO_ROOT / ".superpowers" / "sdd" / "task-2-review-r1.md"
FOUNDATION_GATE = Path("D:/xunce/out/ppo_frontier/pfv1-20260710T1640-4ea1142/s0/gate.json")
FOUNDATION_ENVIRONMENT = Path(
    "D:/xunce/env-backups/ppo_frontier/20260710T111602/environment-lock-manifest.json"
)
FOUNDATION_BASE_COMMIT = "7378737a0d18ce6e4779e30605b557bd1ab25e6e"
CANONICAL_THREAD_ID = "019f49d3-8597-71a3-9908-e39febf4cbb3"
CANONICAL_USER_TURN_ID = "019f0000-0000-7000-8000-000000000001"
CANONICAL_REPLAY_USER_TURN_ID = "019f0000-0000-7000-8000-000000000002"
_CLEAN_REVIEW_ANCHOR_LINES = (
    "Specification compliance verdict: APPROVED",
    "Code quality verdict: APPROVED",
    "Final gate conclusion: READY_FOR_HUMAN_APPROVAL",
)
_CONTRADICTORY_REVIEW_ANCHOR_LINES = (
    "Specification compliance verdict: CHANGES_REQUIRED",
    "Code quality verdict: CHANGES_REQUIRED",
    "Final gate conclusion: NOT_READY_FOR_HUMAN_APPROVAL",
)


def _package_stats(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "review_package_sha256": hashlib.sha256(payload).hexdigest(),
        "review_package_bytes": len(payload),
        "review_package_lf_count": payload.count(b"\n"),
        "review_package_logical_line_count": len(payload.splitlines()),
    }


def _write_git_native_review_package(
    path: Path,
    *,
    reviewed_paths: set[str] | frozenset[str] | None = None,
) -> None:
    from lunar_exploration_ppo.workflows.stage1_source import STAGE1_REVIEWED_PATHS

    paths = sorted(STAGE1_REVIEWED_PATHS if reviewed_paths is None else reviewed_paths)
    temporary_root = Path("D:/xunce/tmp/ppo_frontier/git-index")
    temporary_root.mkdir(parents=True, exist_ok=True)
    index_path = temporary_root / f"stage1-test-{uuid.uuid4().hex}.index"
    lock_path = Path(str(index_path) + ".lock")
    environment = os.environ.copy()
    environment["GIT_INDEX_FILE"] = str(index_path)
    staged_before = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--cached", "--quiet"],
        check=False,
        capture_output=True,
    )
    assert staged_before.returncode == 0
    try:
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "read-tree", FOUNDATION_BASE_COMMIT],
            check=True,
            capture_output=True,
            env=environment,
        )
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "update-index", "--add", "--remove", "--", *paths],
            check=True,
            capture_output=True,
            env=environment,
        )
        package = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "diff",
                "--cached",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                "--no-renames",
                FOUNDATION_BASE_COMMIT,
                "--",
                *paths,
            ],
            check=True,
            capture_output=True,
            env=environment,
        ).stdout
        assert package.startswith(b"diff --git ")
        path.write_bytes(package)
    finally:
        if lock_path.is_file():
            lock_path.unlink()
        if index_path.is_file():
            index_path.unlink()
    staged_after = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--cached", "--quiet"],
        check=False,
        capture_output=True,
    )
    assert staged_after.returncode == 0


def _append_git_native_extra_path(package: Path) -> None:
    payload = b"unauthorized authority-chain path\n"
    object_id = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "hash-object", "--stdin"],
        input=payload,
        check=True,
        capture_output=True,
    ).stdout.decode("ascii").strip()
    package.write_bytes(
        package.read_bytes()
        + (
            "diff --git a/stage1-authority-extra.txt b/stage1-authority-extra.txt\n"
            "new file mode 100644\n"
            f"index {'0' * len(object_id)}..{object_id}\n"
            "--- /dev/null\n"
            "+++ b/stage1-authority-extra.txt\n"
            "@@ -0,0 +1 @@\n"
            "+unauthorized authority-chain path\n"
        ).encode("ascii")
    )


def _mutate_added_line_without_changing_package_stats(package: Path) -> None:
    lines = package.read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith(b"+") and not line.startswith(b"+++") and len(line.rstrip(b"\r\n")) > 1:
            offset = 1
            replacement = b"Z" if line[offset : offset + 1] != b"Z" else b"Y"
            lines[index] = line[:offset] + replacement + line[offset + 1 :]
            package.write_bytes(b"".join(lines))
            return
    raise AssertionError("generated review package has no mutable added line")


def _write_clean_review_report(
    path: Path,
    *,
    review_package: Path,
    reviewed_tree: str,
    overrides: dict[str, object] | None = None,
    body: str | None = None,
) -> None:
    front_matter: dict[str, object] = {
        "schema_version": "ppo_highres_frontier_stage1_review_report/v2",
        "spec_verdict": "APPROVED",
        "quality_verdict": "APPROVED",
        "critical_count": 0,
        "important_count": 0,
        "minor_count": 0,
        "final_gate_conclusion": "READY_FOR_HUMAN_APPROVAL",
        "foundation_base_commit": FOUNDATION_BASE_COMMIT,
        "reviewed_prospective_git_tree": reviewed_tree,
        **_package_stats(review_package),
    }
    front_matter.update(overrides or {})
    if body is None:
        body = "\n".join(
            (
                "# Independent Stage 1 review",
                "",
                *_CLEAN_REVIEW_ANCHOR_LINES,
                "",
                "All required checks completed cleanly.",
            )
        )
    lines = ["---", *(f"{key}: {value}" for key, value in front_matter.items()), "---", "", body]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")


def test_review_body_allows_historical_r4_tokens_with_single_current_anchors(tmp_path: Path) -> None:
    from lunar_exploration_ppo.workflows.stage1_review import _parse_review_report

    package = tmp_path / "package.diff"
    package.write_bytes(b"review-package\n")
    report = tmp_path / "review.md"
    body = "\n".join(
        (
            "# Independent Stage 1 review",
            "",
            *_CLEAN_REVIEW_ANCHOR_LINES,
            "",
            "## R4 closure history",
            "The old R4 report recorded Specification compliance verdict: CHANGES_REQUIRED "
            "and Final gate conclusion: NOT_READY_FOR_HUMAN_APPROVAL.",
        )
    )
    _write_clean_review_report(
        report,
        review_package=package,
        reviewed_tree=FOUNDATION_BASE_COMMIT,
        body=body,
    )

    _, parsed_body = _parse_review_report(report)

    assert "CHANGES_REQUIRED" in parsed_body
    assert "NOT_READY_FOR_HUMAN_APPROVAL" in parsed_body


@pytest.mark.parametrize("anchor_index", range(3))
@pytest.mark.parametrize("failure_mode", ("missing", "duplicate", "contradictory"))
def test_review_body_anchor_lines_fail_closed_when_missing_duplicated_or_contradictory(
    tmp_path: Path,
    anchor_index: int,
    failure_mode: str,
) -> None:
    from lunar_exploration_ppo.workflows.stage1_artifacts import Stage1WorkflowError
    from lunar_exploration_ppo.workflows.stage1_review import _parse_review_report

    anchors = list(_CLEAN_REVIEW_ANCHOR_LINES)
    if failure_mode == "missing":
        anchors.pop(anchor_index)
    elif failure_mode == "duplicate":
        anchors.insert(anchor_index, anchors[anchor_index])
    else:
        anchors[anchor_index] = _CONTRADICTORY_REVIEW_ANCHOR_LINES[anchor_index]
    package = tmp_path / "package.diff"
    package.write_bytes(b"review-package\n")
    report = tmp_path / "review.md"
    _write_clean_review_report(
        report,
        review_package=package,
        reviewed_tree=FOUNDATION_BASE_COMMIT,
        body="\n".join(("# Independent Stage 1 review", "", *anchors)),
    )

    with pytest.raises(Stage1WorkflowError, match="review report body"):
        _parse_review_report(report)


def _expected_reviewed_source_set_sha256() -> str:
    from lunar_exploration_ppo.workflows.stage1_source import STAGE1_REVIEWED_PATHS

    digest = hashlib.sha256()
    digest.update(b"ppo_highres_frontier_stage1_reviewed_source_set/v1\0")
    for relative in sorted(STAGE1_REVIEWED_PATHS):
        path_bytes = relative.encode("utf-8")
        payload = (REPO_ROOT / relative).read_bytes()
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _run_machine(tmp_path: Path, run_id: str):
    from lunar_exploration_ppo.workflows.stage1 import run_stage1_smoke_workflow

    return run_stage1_smoke_workflow(
        config_path=STAGE1_CONFIG,
        run_id=run_id,
        base_output_root=tmp_path / "runs",
    )


@pytest.fixture(scope="module")
def r4_machine_stage(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("stage1-r4-machine")
    return _run_machine(root, "r4-machine-template").stage_root


def _copy_machine_stage(template: Path, destination: Path) -> Path:
    stage = destination / template.parent.name / "s1"
    stage.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template, stage)
    return stage


def test_c1_review_rejects_non_git_package_before_writing_review(
    tmp_path: Path,
    r4_machine_stage: Path,
) -> None:
    from lunar_exploration_ppo.workflows.stage1 import record_stage1_independent_review
    from lunar_exploration_ppo.workflows.stage1_artifacts import Stage1WorkflowError
    from lunar_exploration_ppo.workflows.stage1_source import compute_stage1_prospective_tree

    stage = _copy_machine_stage(r4_machine_stage, tmp_path)
    package = tmp_path / "arbitrary-review-package.bin"
    package.write_bytes(b"not-a-git-native-review-package\n")
    report = tmp_path / "review.md"
    tree = compute_stage1_prospective_tree(REPO_ROOT, base_commit=FOUNDATION_BASE_COMMIT)
    _write_clean_review_report(report, review_package=package, reviewed_tree=tree)

    with pytest.raises(Stage1WorkflowError, match="package|Git|tree|patch"):
        record_stage1_independent_review(
            stage_root=stage,
            repo_root=REPO_ROOT,
            foundation_gate=FOUNDATION_GATE,
            foundation_environment_manifest=FOUNDATION_ENVIRONMENT,
            review_report=report,
            review_package=package,
        )
    assert not (stage / "review.json").exists()


@pytest.mark.parametrize("package_variant", ("missing_path", "extra_path", "same_stats_wrong_tree"))
def test_c1_review_package_must_rebuild_exact_32_path_reviewed_tree(
    tmp_path: Path,
    r4_machine_stage: Path,
    package_variant: str,
) -> None:
    from lunar_exploration_ppo.workflows.stage1 import record_stage1_independent_review
    from lunar_exploration_ppo.workflows.stage1_artifacts import Stage1WorkflowError
    from lunar_exploration_ppo.workflows.stage1_source import (
        STAGE1_REVIEWED_PATHS,
        compute_stage1_prospective_tree,
    )

    stage = _copy_machine_stage(r4_machine_stage, tmp_path / package_variant)
    package = tmp_path / f"{package_variant}.diff"
    if package_variant == "missing_path":
        _write_git_native_review_package(
            package,
            reviewed_paths=set(STAGE1_REVIEWED_PATHS) - {"src/lunar_exploration_ppo/workflows/stage1_gate.py"},
        )
    else:
        _write_git_native_review_package(package)
        if package_variant == "extra_path":
            _append_git_native_extra_path(package)
        else:
            before = _package_stats(package)
            _mutate_added_line_without_changing_package_stats(package)
            assert _package_stats(package)["review_package_sha256"] != before["review_package_sha256"]
            assert _package_stats(package)["review_package_bytes"] == before["review_package_bytes"]
            assert _package_stats(package)["review_package_lf_count"] == before["review_package_lf_count"]
            assert _package_stats(package)["review_package_logical_line_count"] == before[
                "review_package_logical_line_count"
            ]
    report = tmp_path / f"{package_variant}.md"
    tree = compute_stage1_prospective_tree(REPO_ROOT, base_commit=FOUNDATION_BASE_COMMIT)
    _write_clean_review_report(report, review_package=package, reviewed_tree=tree)

    with pytest.raises(Stage1WorkflowError, match="package|path|tree|patch|Git"):
        record_stage1_independent_review(
            stage_root=stage,
            repo_root=REPO_ROOT,
            foundation_gate=FOUNDATION_GATE,
            foundation_environment_manifest=FOUNDATION_ENVIRONMENT,
            review_report=report,
            review_package=package,
        )
    assert not (stage / "review.json").exists()


@pytest.mark.parametrize("extra_kind", ("file", "directory"))
def test_m1_review_rejects_noncanonical_machine_artifact_set_before_writing_review(
    tmp_path: Path,
    r4_machine_stage: Path,
    extra_kind: str,
) -> None:
    from lunar_exploration_ppo.workflows.stage1 import record_stage1_independent_review
    from lunar_exploration_ppo.workflows.stage1_artifacts import Stage1WorkflowError
    from lunar_exploration_ppo.workflows.stage1_source import compute_stage1_prospective_tree

    stage = _copy_machine_stage(r4_machine_stage, tmp_path / extra_kind)
    extra = stage / "unexpected-entry"
    extra.write_text("unexpected\n", encoding="utf-8") if extra_kind == "file" else extra.mkdir()
    package = tmp_path / f"{extra_kind}-package.diff"
    _write_git_native_review_package(package)
    report = tmp_path / f"{extra_kind}-review.md"
    tree = compute_stage1_prospective_tree(REPO_ROOT, base_commit=FOUNDATION_BASE_COMMIT)
    _write_clean_review_report(report, review_package=package, reviewed_tree=tree)

    with pytest.raises(Stage1WorkflowError, match="artifact set|unexpected|exact"):
        record_stage1_independent_review(
            stage_root=stage,
            repo_root=REPO_ROOT,
            foundation_gate=FOUNDATION_GATE,
            foundation_environment_manifest=FOUNDATION_ENVIRONMENT,
            review_report=report,
            review_package=package,
        )
    assert not (stage / "review.json").exists()


def test_i1_human_approval_rejects_incomplete_review_before_writing_approval(
    tmp_path: Path,
    r4_machine_stage: Path,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_gate import (
        Stage1GateError,
        record_stage1_human_approval,
    )
    from lunar_exploration_ppo.workflows.stage1_source import compute_stage1_prospective_tree

    stage = _copy_machine_stage(r4_machine_stage, tmp_path)
    tree = compute_stage1_prospective_tree(REPO_ROOT, base_commit=FOUNDATION_BASE_COMMIT)
    ArtifactStore(stage).write_json(
        "review.json",
        {
            "schema_version": "ppo_highres_frontier_stage1_independent_review/v2",
            "state": "awaiting_human_approval",
            "goal_id": "ppo-highres-frontier-map-exploration",
            "stage_id": "ppo_highres_frontier_stage1_smoke_environment/v1",
            "run_id": stage.parent.name,
            "reviewed_git_tree": tree,
            "approval_challenge": "a" * 64,
            "review_recorded_at_utc": "2026-07-11T00:00:00Z",
        },
    )

    with pytest.raises(Stage1GateError, match="context|review|verif"):
        record_stage1_human_approval(
            stage_root=stage,
            thread_id=CANONICAL_THREAD_ID,
            user_turn_id=CANONICAL_USER_TURN_ID,
            approval_text="\u6279\u51c6 Stage 1 Gate",
            user_turn_timestamp_utc="2026-07-11T00:00:01Z",
        )
    assert not (stage / "approval.json").exists()


def test_r4_failed_review_cannot_be_recorded_or_overridden_by_caller(
    tmp_path: Path,
    r4_machine_stage: Path,
) -> None:
    from lunar_exploration_ppo.workflows.stage1 import record_stage1_independent_review
    from lunar_exploration_ppo.workflows.stage1_artifacts import Stage1WorkflowError

    review_package = tmp_path / "review-package.diff"
    review_package.write_bytes(b"frozen-stage1-review-package\n")
    rejected_stage = _copy_machine_stage(r4_machine_stage, tmp_path / "rejected")
    with pytest.raises(Stage1WorkflowError, match="front matter|verdict|review report"):
        record_stage1_independent_review(
            stage_root=rejected_stage,
            repo_root=REPO_ROOT,
            foundation_gate=FOUNDATION_GATE,
            foundation_environment_manifest=FOUNDATION_ENVIRONMENT,
            review_report=REPO_ROOT / ".superpowers" / "sdd" / "task-2-review-r4.md",
            review_package=review_package,
        )
    assert not (rejected_stage / "review.json").exists()

    override_stage = _copy_machine_stage(r4_machine_stage, tmp_path / "override")
    with pytest.raises(TypeError):
        record_stage1_independent_review(
            stage_root=override_stage,
            repo_root=REPO_ROOT,
            foundation_gate=FOUNDATION_GATE,
            foundation_environment_manifest=FOUNDATION_ENVIRONMENT,
            review_report=REPO_ROOT / ".superpowers" / "sdd" / "task-2-review-r4.md",
            review_package=review_package,
            spec_verdict="approved",
            quality_verdict="approved",
            critical_count=0,
            important_count=0,
            minor_count=0,
        )


@pytest.mark.parametrize(
    ("override", "body"),
    [
        ({"review_package_sha256": "0" * 64}, "clean"),
        ({"review_package_bytes": 1}, "clean"),
        ({"review_package_lf_count": 99}, "clean"),
        ({"review_package_logical_line_count": 99}, "clean"),
        ({"reviewed_prospective_git_tree": "0" * 40}, "clean"),
        ({"foundation_base_commit": "0" * 40}, "clean"),
        ({"critical_count": 1}, "clean"),
        ({}, "Specification compliance verdict: CHANGES_REQUIRED"),
        ({}, "Final gate conclusion: NOT_READY_FOR_HUMAN_APPROVAL"),
    ],
)
def test_review_front_matter_and_body_drift_fail_closed(
    tmp_path: Path,
    r4_machine_stage: Path,
    override: dict[str, object],
    body: str,
) -> None:
    from lunar_exploration_ppo.workflows.stage1 import record_stage1_independent_review
    from lunar_exploration_ppo.workflows.stage1_artifacts import Stage1WorkflowError
    from lunar_exploration_ppo.workflows.stage1_source import compute_stage1_prospective_tree

    stage = _copy_machine_stage(r4_machine_stage, tmp_path)
    package = tmp_path / "package.diff"
    package.write_bytes(b"line-one\nline-two\n")
    report = tmp_path / "review.md"
    tree = compute_stage1_prospective_tree(REPO_ROOT, base_commit=FOUNDATION_BASE_COMMIT)
    _write_clean_review_report(
        report,
        review_package=package,
        reviewed_tree=tree,
        overrides=override,
        body=body,
    )

    with pytest.raises(Stage1WorkflowError, match="review|package|tree|base|finding|verdict|gate"):
        record_stage1_independent_review(
            stage_root=stage,
            repo_root=REPO_ROOT,
            foundation_gate=FOUNDATION_GATE,
            foundation_environment_manifest=FOUNDATION_ENVIRONMENT,
            review_report=report,
            review_package=package,
        )
    assert not (stage / "review.json").exists()


def test_clean_review_is_parsed_and_preserves_original_machine_attestation(
    tmp_path: Path,
    r4_machine_stage: Path,
) -> None:
    from lunar_exploration_ppo.workflows.stage1 import record_stage1_independent_review
    from lunar_exploration_ppo.workflows.stage1_source import compute_stage1_prospective_tree

    stage = _copy_machine_stage(r4_machine_stage, tmp_path)
    package = tmp_path / "package.diff"
    _write_git_native_review_package(package)
    report = tmp_path / "review.md"
    tree = compute_stage1_prospective_tree(REPO_ROOT, base_commit=FOUNDATION_BASE_COMMIT)
    _write_clean_review_report(report, review_package=package, reviewed_tree=tree)
    machine_bytes = {
        path.name: path.read_bytes()
        for path in stage.iterdir()
        if path.name != "review.json"
    }

    result = record_stage1_independent_review(
        stage_root=stage,
        repo_root=REPO_ROOT,
        foundation_gate=FOUNDATION_GATE,
        foundation_environment_manifest=FOUNDATION_ENVIRONMENT,
        review_report=report,
        review_package=package,
    )

    review = json.loads(result.review_path.read_text(encoding="utf-8"))
    assert review["spec_verdict"] == "APPROVED"
    assert review["quality_verdict"] == "APPROVED"
    assert review["issue_counts"] == {"critical": 0, "important": 0, "minor": 0}
    assert review["final_gate_conclusion"] == "READY_FOR_HUMAN_APPROVAL"
    assert review["state"] == "awaiting_human_approval"
    assert review["review_package"] == {
        "path": str(package.resolve()),
        "sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
        "bytes": len(package.read_bytes()),
        "lf_count": package.read_bytes().count(b"\n"),
        "logical_line_count": len(package.read_bytes().splitlines()),
    }
    assert len(review["approval_challenge"]) == 64
    assert review["review_recorded_at_utc"].endswith("Z")
    assert all((stage / name).read_bytes() == payload for name, payload in machine_bytes.items())


def test_r8_i1_public_review_evidence_wrapper_uses_handle_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.workflows.stage1  # noqa: F401
    from lunar_exploration_ppo.workflows.stage1_review import verify_stage1_review_evidence
    from lunar_exploration_ppo.workflows.stage1_source import compute_stage1_prospective_tree

    package = tmp_path / "public-wrapper-package.diff"
    _write_git_native_review_package(package)
    package_payload = package.read_bytes()
    reviewed_tree = compute_stage1_prospective_tree(
        REPO_ROOT,
        base_commit=FOUNDATION_BASE_COMMIT,
    )
    report = tmp_path / "public-wrapper-review.md"
    _write_clean_review_report(
        report,
        review_package=package,
        reviewed_tree=reviewed_tree,
    )
    report_payload = report.read_bytes()
    blocked = {
        os.path.normcase(os.fspath(report.resolve())),
        os.path.normcase(os.fspath(package.resolve())),
    }
    original_read_bytes = Path.read_bytes
    blocked_read_calls = 0

    def reject_authority_path_read(path: Path) -> bytes:
        nonlocal blocked_read_calls
        if os.path.normcase(os.fspath(path.resolve())) in blocked:
            blocked_read_calls += 1
            raise AssertionError("review authority Path.read_bytes must not be used")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_authority_path_read)

    front_matter, report_binding, package_binding = verify_stage1_review_evidence(
        repo_root=REPO_ROOT,
        foundation_base_commit=FOUNDATION_BASE_COMMIT,
        reviewed_git_tree=reviewed_tree,
        review_report=report,
        review_package=package,
    )

    assert blocked_read_calls == 0
    assert front_matter["final_gate_conclusion"] == "READY_FOR_HUMAN_APPROVAL"
    assert report_binding["sha256"] == hashlib.sha256(report_payload).hexdigest()
    assert package_binding["sha256"] == hashlib.sha256(package_payload).hexdigest()


def test_r8_i1_review_package_replay_compares_all_frozen_source_blobs(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    import lunar_exploration_ppo.workflows.stage1  # noqa: F401
    from lunar_exploration_ppo.workflows.stage1_artifacts import (
        FrozenFileSnapshot,
        Stage1WorkflowError,
    )
    from lunar_exploration_ppo.workflows.stage1_review import replay_stage1_review_package
    from lunar_exploration_ppo.workflows.stage1_source import (
        Stage1SourceSnapshot,
        compute_stage1_source_identity_from_snapshot,
    )

    package = tmp_path / "frozen-source-package.diff"
    _write_git_native_review_package(package)
    package_snapshot = FrozenFileSnapshot.capture(package, "review package")
    source_snapshot = Stage1SourceSnapshot.capture(
        REPO_ROOT,
        base_commit=FOUNDATION_BASE_COMMIT,
    )
    reviewed_tree = str(
        compute_stage1_source_identity_from_snapshot(source_snapshot)["prospective_git_tree"]
    )

    assert replay_stage1_review_package(
        repo_root=REPO_ROOT,
        foundation_base_commit=FOUNDATION_BASE_COMMIT,
        review_package=package_snapshot,
        expected_reviewed_git_tree=reviewed_tree,
        source_snapshot=source_snapshot,
    ) == reviewed_tree

    altered_relative = "src/lunar_exploration_ppo/workflows/stage1_review.py"
    original_file = source_snapshot.file(altered_relative)
    altered_payload = original_file.payload + b"\n# frozen-source-mismatch\n"
    altered_file = replace(
        original_file,
        payload=altered_payload,
        sha256=hashlib.sha256(altered_payload).hexdigest(),
        size_bytes=len(altered_payload),
    )
    altered_source = replace(
        source_snapshot,
        files=tuple(
            (relative, altered_file if relative == altered_relative else snapshot)
            for relative, snapshot in source_snapshot.files
        ),
    )

    with pytest.raises(Stage1WorkflowError, match="frozen|blob|payload|source"):
        replay_stage1_review_package(
            repo_root=REPO_ROOT,
            foundation_base_commit=FOUNDATION_BASE_COMMIT,
            review_package=package_snapshot,
            expected_reviewed_git_tree=reviewed_tree,
            source_snapshot=altered_source,
        )


def _make_r8_i1_isolated_stage1_source_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    extra_stage_file: bool = False,
) -> dict[str, object]:
    import lunar_exploration_ppo.workflows.stage1_source as source_module
    from lunar_exploration_ppo.workflows.stage1_source import STAGE1_REVIEWED_PATHS

    branch = "codex/ppo-highres-frontier-map-exploration"
    repo = tmp_path / "isolated-stage1-source-repo"

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()

    subprocess.run(
        ["git", "init", "--initial-branch", branch, str(repo)],
        check=True,
        capture_output=True,
    )
    git("config", "user.name", "Stage1 Snapshot Test")
    git("config", "user.email", "stage1-snapshot@example.invalid")
    git("config", "core.autocrlf", "false")
    (repo / "foundation.txt").write_bytes(b"foundation-base\n")
    package_init = repo / "src/lunar_exploration_ppo/__init__.py"
    package_init.parent.mkdir(parents=True, exist_ok=True)
    package_init.write_bytes(b'__version__ = "0.1.0"\n')
    git("add", "--", "foundation.txt", "src/lunar_exploration_ppo/__init__.py")
    git("commit", "-m", "foundation base")
    foundation_base_commit = git("rev-parse", "HEAD")

    for relative in sorted(STAGE1_REVIEWED_PATHS):
        source = repo / Path(relative)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"isolated reviewed source: {relative}\n".encode("utf-8"))
    if extra_stage_file:
        (repo / "unreviewed-stage1-extra.txt").write_bytes(b"not reviewed\n")
    git("add", "--all")
    git("commit", "-m", "feat: add smoke exploration environment")
    stage1_head = git("rev-parse", "HEAD")
    stage1_tree = git("rev-parse", "HEAD^{tree}")

    monkeypatch.setattr(source_module, "FOUNDATION_WORKTREE_ROOT", str(repo.resolve()))
    monkeypatch.setattr(source_module, "FOUNDATION_BRANCH", branch)
    monkeypatch.setattr(source_module, "FOUNDATION_GIT_DIR", str((repo / ".git").resolve()))
    monkeypatch.setattr(
        source_module,
        "FOUNDATION_GIT_COMMON_DIR",
        str((repo / ".git").resolve()),
    )
    package_import_path = (repo / "src/lunar_exploration_ppo/__init__.py").resolve()
    workflow_import_path = (
        repo / "src/lunar_exploration_ppo/workflows/stage1.py"
    ).resolve()
    original_resolve_imports = source_module._resolve_import_paths_without_reading

    def resolve_isolated_imports(
        received_repo: Path,
        *,
        package_import_path: str | Path | None = None,
        workflow_import_path: str | Path | None = None,
    ) -> tuple[Path, Path]:
        if Path(received_repo).resolve() == repo.resolve():
            return (
                Path(package_import_path).resolve()
                if package_import_path is not None
                else fixture_package_import,
                Path(workflow_import_path).resolve()
                if workflow_import_path is not None
                else fixture_workflow_import,
            )
        return original_resolve_imports(
            received_repo,
            package_import_path=package_import_path,
            workflow_import_path=workflow_import_path,
        )

    fixture_package_import = package_import_path
    fixture_workflow_import = workflow_import_path
    monkeypatch.setattr(
        source_module,
        "_resolve_import_paths_without_reading",
        resolve_isolated_imports,
    )
    return {
        "repo": repo.resolve(),
        "branch": branch,
        "foundation_base_commit": foundation_base_commit,
        "stage1_head": stage1_head,
        "stage1_tree": stage1_tree,
        "package_import_path": package_import_path,
        "workflow_import_path": workflow_import_path,
        "git": git,
    }


def test_r8_i1_gate_committed_source_snapshot_accepts_single_clean_stage1_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows.stage1_source import (
        ReviewedSourceError,
        Stage1SourceSnapshot,
        compute_stage1_source_identity_from_snapshot,
    )

    fixture = _make_r8_i1_isolated_stage1_source_repo(tmp_path, monkeypatch)
    common = {
        "base_commit": fixture["foundation_base_commit"],
        "package_import_path": fixture["package_import_path"],
        "workflow_import_path": fixture["workflow_import_path"],
    }
    with pytest.raises(ReviewedSourceError, match="Foundation base HEAD"):
        Stage1SourceSnapshot.capture(fixture["repo"], **common)

    snapshot = Stage1SourceSnapshot.capture(
        fixture["repo"],
        mode="gate_committed",
        **common,
    )

    assert snapshot.mode == "gate_committed"
    assert snapshot.git.head.decode("ascii").strip() == fixture["stage1_head"]
    assert snapshot.git.head_tree.decode("ascii").strip() == fixture["stage1_tree"]
    assert compute_stage1_source_identity_from_snapshot(snapshot)["prospective_git_tree"] == fixture[
        "stage1_tree"
    ]
    snapshot.require_current()


@pytest.mark.parametrize(
    ("drift_kind", "error"),
    (
        ("tree", "tree"),
        ("branch", "branch"),
        ("head", "HEAD|commit|parent"),
        ("index", "index"),
        ("status", "clean|status"),
    ),
)
def test_r8_i1_gate_committed_source_snapshot_rejects_wrong_repository_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift_kind: str,
    error: str,
) -> None:
    from lunar_exploration_ppo.workflows.stage1_source import (
        ReviewedSourceError,
        Stage1SourceSnapshot,
    )

    fixture = _make_r8_i1_isolated_stage1_source_repo(
        tmp_path,
        monkeypatch,
        extra_stage_file=drift_kind == "tree",
    )
    git = fixture["git"]
    repo = fixture["repo"]
    assert callable(git) and isinstance(repo, Path)
    if drift_kind == "branch":
        git("switch", "-c", "wrong-stage1-branch")
    elif drift_kind == "head":
        git("commit", "--allow-empty", "-m", "unauthorized extra commit")
    elif drift_kind == "index":
        source = repo / "src/lunar_exploration_ppo/workflows/stage1_gate.py"
        source.write_bytes(source.read_bytes() + b"staged drift\n")
        git("add", "--", "src/lunar_exploration_ppo/workflows/stage1_gate.py")
    elif drift_kind == "status":
        source = repo / "src/lunar_exploration_ppo/workflows/stage1_gate.py"
        source.write_bytes(source.read_bytes() + b"unstaged drift\n")

    with pytest.raises(ReviewedSourceError, match=error):
        Stage1SourceSnapshot.capture(
            repo,
            base_commit=fixture["foundation_base_commit"],
            package_import_path=fixture["package_import_path"],
            workflow_import_path=fixture["workflow_import_path"],
            mode="gate_committed",
        )


def test_machine_attests_full_32_path_source_identity_in_config_summary_report_and_manifest(
    r4_machine_stage: Path,
) -> None:
    from lunar_exploration_ppo.workflows.stage1_source import (
        STAGE1_REVIEWED_PATHS,
        compute_stage1_source_identity,
    )

    config = json.loads((r4_machine_stage / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((r4_machine_stage / "summary.json").read_text(encoding="utf-8"))
    report = (r4_machine_stage / "report.md").read_text(encoding="utf-8")
    manifest = json.loads((r4_machine_stage / "manifest.json").read_text(encoding="utf-8"))
    identity = summary["execution_source_identity"]

    assert identity == config["execution_source_identity"]
    assert identity == compute_stage1_source_identity(
        REPO_ROOT,
        base_commit=FOUNDATION_BASE_COMMIT,
    )
    assert identity["schema_version"] == "ppo_highres_frontier_stage1_execution_source_identity/v1"
    assert identity["reviewed_path_count"] == len(STAGE1_REVIEWED_PATHS) == 32
    assert "src/lunar_exploration_ppo/utils/artifact_io.py" in STAGE1_REVIEWED_PATHS
    assert identity["reviewed_source_set_sha256"] == _expected_reviewed_source_set_sha256()
    assert len(identity["prospective_git_tree"]) == 40
    expected_src = (REPO_ROOT / "src").resolve()
    assert Path(identity["repo_src_root"]) == expected_src
    assert Path(identity["package_import_path"]).is_relative_to(expected_src)
    assert Path(identity["workflow_import_path"]).is_relative_to(expected_src)
    assert identity["reviewed_source_set_sha256"] in report
    entries = {entry["path"]: entry for entry in manifest["artifacts"]}
    for name in ("config.json", "summary.json", "report.md"):
        payload = (r4_machine_stage / name).read_bytes()
        assert entries[name]["sha256"] == hashlib.sha256(payload).hexdigest()
        assert entries[name]["size_bytes"] == len(payload)


def test_machine_fails_if_source_identity_drifts_during_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows import stage1 as stage1_module
    from lunar_exploration_ppo.workflows.stage1_artifacts import Stage1WorkflowError
    from lunar_exploration_ppo.workflows.stage1_source import compute_stage1_source_identity

    identity_a = compute_stage1_source_identity(REPO_ROOT, base_commit=FOUNDATION_BASE_COMMIT)
    identity_b = dict(identity_a)
    identity_b["reviewed_source_set_sha256"] = "0" * 64
    identities = iter((identity_a, identity_b))
    monkeypatch.setattr(
        stage1_module,
        "compute_stage1_source_identity",
        lambda *args, **kwargs: next(identities),
        raising=False,
    )

    def fake_episode(env, *, episode_index, policy, metric_writer=None):
        if metric_writer is not None:
            metric_writer({"episode": episode_index, "finite": 1.0})
        return stage1_module.EpisodeResult(
            episode_index=episode_index,
            transition_count=1,
            trainable_transition_count=1,
            total_reward=1.0,
            final_coverage_rate=0.995,
            done_reason="success_done",
            reset_empty=False,
        )

    monkeypatch.setattr(stage1_module, "run_episode", fake_episode)
    with pytest.raises(Stage1WorkflowError, match="source identity drift"):
        stage1_module.run_stage1_smoke_workflow(
            config_path=STAGE1_CONFIG,
            run_id="r4-running-source-drift",
            base_output_root=tmp_path,
        )


def test_source_identity_rejects_package_import_outside_current_repo_src(tmp_path: Path) -> None:
    from lunar_exploration_ppo.workflows.stage1_source import (
        ReviewedSourceError,
        compute_stage1_source_identity,
    )

    outside_package = tmp_path / "foreign" / "lunar_exploration_ppo" / "__init__.py"
    outside_package.parent.mkdir(parents=True)
    outside_package.write_text("# foreign package\n", encoding="utf-8")
    with pytest.raises(ReviewedSourceError, match="package import|repo src"):
        compute_stage1_source_identity(
            REPO_ROOT,
            base_commit=FOUNDATION_BASE_COMMIT,
            package_import_path=outside_package,
        )


def test_review_rejects_machine_a_current_source_b_mismatch(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows import stage1_review as review_module
    from lunar_exploration_ppo.workflows.stage1_artifacts import Stage1WorkflowError
    from lunar_exploration_ppo.workflows.stage1_source import compute_stage1_source_identity

    stage = _copy_machine_stage(r4_machine_stage, tmp_path)
    identity_b = compute_stage1_source_identity(REPO_ROOT, base_commit=FOUNDATION_BASE_COMMIT)
    identity_b = dict(identity_b)
    identity_b["reviewed_source_set_sha256"] = "0" * 64
    monkeypatch.setattr(
        review_module,
        "compute_stage1_source_identity_from_snapshot",
        lambda snapshot: identity_b,
    )
    package = tmp_path / "package.diff"
    package.write_bytes(b"review-package\n")
    report = tmp_path / "review.md"
    _write_clean_review_report(
        report,
        review_package=package,
        reviewed_tree=str(identity_b["prospective_git_tree"]),
    )

    with pytest.raises(Stage1WorkflowError, match="machine.*source|source.*machine"):
        review_module.record_stage1_independent_review(
            stage_root=stage,
            repo_root=REPO_ROOT,
            foundation_gate=FOUNDATION_GATE,
            foundation_environment_manifest=FOUNDATION_ENVIRONMENT,
            review_report=report,
            review_package=package,
        )


def _reviewed_context_from_template(
    *,
    template: Path,
    destination: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from lunar_exploration_ppo.workflows.stage1 import record_stage1_independent_review
    from lunar_exploration_ppo.workflows.stage1_gate import (
        Stage1GateBindingVerifier,
        Stage1GateContext,
    )
    from lunar_exploration_ppo.workflows.stage1_source import compute_stage1_prospective_tree

    stage = _copy_machine_stage(template, destination)
    package = destination / "package.diff"
    _write_git_native_review_package(package)
    tree = compute_stage1_prospective_tree(REPO_ROOT, base_commit=FOUNDATION_BASE_COMMIT)
    report = destination / "review.md"
    _write_clean_review_report(report, review_package=package, reviewed_tree=tree)
    result = record_stage1_independent_review(
        stage_root=stage,
        repo_root=REPO_ROOT,
        foundation_gate=FOUNDATION_GATE,
        foundation_environment_manifest=FOUNDATION_ENVIRONMENT,
        review_report=report,
        review_package=package,
    )
    review = json.loads(result.review_path.read_text(encoding="utf-8"))
    _install_gate_committed_source_snapshot_fixture(monkeypatch)
    monkeypatch.setattr(
        Stage1GateBindingVerifier,
        "_verify_repository_identity",
        classmethod(lambda cls, repo_root: (REPO_ROOT.resolve(), tree)),
    )
    context = Stage1GateContext(
        repo_root=REPO_ROOT,
        stage_root=stage,
        foundation_gate=FOUNDATION_GATE,
        foundation_environment_manifest=FOUNDATION_ENVIRONMENT,
        run_id=stage.parent.name,
    )
    return stage, context, review


def _install_gate_committed_source_snapshot_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from lunar_exploration_ppo.workflows.stage1_source import Stage1SourceSnapshot

    original_capture = Stage1SourceSnapshot.capture.__func__
    review_snapshot = original_capture(
        Stage1SourceSnapshot,
        REPO_ROOT,
        base_commit=FOUNDATION_BASE_COMMIT,
    )
    committed_snapshot = replace(review_snapshot, mode="gate_committed")

    def capture_committed_view(
        cls,
        repo_root,
        *,
        base_commit,
        package_import_path=None,
        workflow_import_path=None,
        mode="review_dirty",
    ):
        if Path(repo_root).expanduser().resolve() == REPO_ROOT.resolve():
            assert mode == "gate_committed"
            assert base_commit == FOUNDATION_BASE_COMMIT
            assert package_import_path is None
            assert workflow_import_path is None
            return committed_snapshot
        return original_capture(
            cls,
            repo_root,
            base_commit=base_commit,
            package_import_path=package_import_path,
            workflow_import_path=workflow_import_path,
            mode=mode,
        )

    monkeypatch.setattr(
        Stage1SourceSnapshot,
        "capture",
        classmethod(capture_committed_view),
    )


def _review_recording_inputs(*, template: Path, destination: Path):
    from lunar_exploration_ppo.workflows.stage1_source import compute_stage1_prospective_tree

    stage = _copy_machine_stage(template, destination)
    package = destination / "mutation-package.diff"
    _write_git_native_review_package(package)
    tree = compute_stage1_prospective_tree(REPO_ROOT, base_commit=FOUNDATION_BASE_COMMIT)
    report = destination / "mutation-review.md"
    _write_clean_review_report(report, review_package=package, reviewed_tree=tree)
    return stage, report, package


def _record_review(stage: Path, report: Path, package: Path):
    from lunar_exploration_ppo.workflows.stage1 import record_stage1_independent_review

    return record_stage1_independent_review(
        stage_root=stage,
        repo_root=REPO_ROOT,
        foundation_gate=FOUNDATION_GATE,
        foundation_environment_manifest=FOUNDATION_ENVIRONMENT,
        review_report=report,
        review_package=package,
    )


def test_r8_i2_review_preexisting_payload_is_preserved(
    tmp_path: Path,
    r4_machine_stage: Path,
) -> None:
    from lunar_exploration_ppo.workflows.stage1_artifacts import Stage1WorkflowError

    stage, report, package = _review_recording_inputs(
        template=r4_machine_stage,
        destination=tmp_path,
    )
    review_path = stage / "review.json"
    foreign_payload = b'{"actor":"foreign-preexisting-review"}\n'
    review_path.write_bytes(foreign_payload)

    with pytest.raises(Stage1WorkflowError, match="existing review|artifact set"):
        _record_review(stage, report, package)

    assert review_path.read_bytes() == foreign_payload


def test_r8_i2_review_exclusive_publish_race_preserves_foreign_review(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, report, package = _review_recording_inputs(
        template=r4_machine_stage,
        destination=tmp_path,
    )
    review_path = stage / "review.json"
    foreign_payload = b'{"actor":"foreign-review-publish-racer"}\n'
    real_link = os.link
    publish_calls = 0

    def competing_link(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        nonlocal publish_calls
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path.name == "review.json":
            publish_calls += 1
            assert json.loads(source_path.read_bytes())["state"] == "awaiting_human_approval"
            assert not destination_path.exists()
            destination_path.write_bytes(foreign_payload)
        real_link(source, destination)

    monkeypatch.setattr(os, "link", competing_link)
    try:
        with pytest.raises(FileExistsError):
            _record_review(stage, report, package)
        assert publish_calls == 1
        assert review_path.read_bytes() == foreign_payload
        assert not [path for path in stage.iterdir() if path.name.startswith(".review.json.")]
    finally:
        if review_path.is_file():
            review_path.unlink()


def test_r9_i1_review_post_publish_failure_retains_owned_payload_and_retry_fails_closed(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_artifacts import Stage1WorkflowError

    class SentinelWriteError(RuntimeError):
        pass

    stage, report, package = _review_recording_inputs(
        template=r4_machine_stage,
        destination=tmp_path,
    )
    review_path = stage / "review.json"
    original_write_json = ArtifactStore.write_json_exclusive
    published_payloads: list[bytes] = []

    def write_then_raise(store, relative_path, value):
        written = original_write_json(store, relative_path, value)
        if Path(relative_path).as_posix() == "review.json":
            published_payloads.append(ArtifactStore.canonical_json_bytes(value))
            assert written.read_bytes() == published_payloads[-1]
            raise SentinelWriteError("sentinel review write-then-raise")
        return written

    monkeypatch.setattr(ArtifactStore, "write_json_exclusive", write_then_raise)
    try:
        with pytest.raises(SentinelWriteError, match="sentinel review write-then-raise"):
            _record_review(stage, report, package)
        assert len(published_payloads) == 1
        assert review_path.read_bytes() == published_payloads[0]

        with pytest.raises(Stage1WorkflowError, match="existing review|artifact set"):
            _record_review(stage, report, package)
        assert len(published_payloads) == 1
        assert review_path.read_bytes() == published_payloads[0]
    finally:
        if review_path.is_file():
            review_path.unlink()


def test_r9_i1_review_post_publish_failure_propagates_original_without_canonical_unlink(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore

    class SentinelWriteError(RuntimeError):
        pass

    stage, report, package = _review_recording_inputs(
        template=r4_machine_stage,
        destination=tmp_path,
    )
    review_path = (stage / "review.json").resolve()
    original_write_json = ArtifactStore.write_json_exclusive
    original_unlink = Path.unlink
    published_payloads: list[bytes] = []
    canonical_unlink_calls = 0

    def write_then_raise(store, relative_path, value):
        written = original_write_json(store, relative_path, value)
        if Path(relative_path).as_posix() == "review.json":
            published_payloads.append(ArtifactStore.canonical_json_bytes(value))
            raise SentinelWriteError("sentinel original review write failure")
        return written

    def reject_review_unlink(path, *args, **kwargs):
        nonlocal canonical_unlink_calls
        if path.resolve() == review_path:
            canonical_unlink_calls += 1
            raise PermissionError("sentinel review rollback unlink failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(ArtifactStore, "write_json_exclusive", write_then_raise)
    monkeypatch.setattr(Path, "unlink", reject_review_unlink)
    try:
        with pytest.raises(SentinelWriteError, match="original review write failure"):
            _record_review(stage, report, package)
        assert canonical_unlink_calls == 0
        assert len(published_payloads) == 1
        assert review_path.read_bytes() == published_payloads[0]
    finally:
        if review_path.is_file():
            original_unlink(review_path)


def test_toctou_mutation_report_after_snapshot_before_parse_leaves_no_review(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.workflows.stage1_review as review_module
    from lunar_exploration_ppo.workflows.stage1_artifacts import Stage1WorkflowError

    stage, report, package = _review_recording_inputs(
        template=r4_machine_stage,
        destination=tmp_path,
    )
    original_report = report.read_bytes()
    original_parse = review_module._parse_review_report
    mutated = False

    def mutate_then_parse(source):
        nonlocal mutated
        if not mutated:
            report.write_bytes(original_report + b"concurrent report mutation\n")
            mutated = True
        return original_parse(source)

    monkeypatch.setattr(review_module, "_parse_review_report", mutate_then_parse)
    try:
        with pytest.raises(Stage1WorkflowError, match="snapshot|drift|changed"):
            _record_review(stage, report, package)
        assert mutated
        assert not (stage / "review.json").exists()
    finally:
        report.write_bytes(original_report)
        leaked = stage / "review.json"
        if leaked.is_file():
            leaked.unlink()


def test_toctou_mutation_package_after_snapshot_before_git_replay_leaves_no_review(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.workflows.stage1_review as review_module
    from lunar_exploration_ppo.workflows.stage1_artifacts import Stage1WorkflowError

    stage, report, package = _review_recording_inputs(
        template=r4_machine_stage,
        destination=tmp_path,
    )
    original_package = package.read_bytes()
    original_replay = review_module.replay_stage1_review_package
    original_open = review_module.os.open
    opened_paths: list[Path] = []
    mutated = False

    def record_package_copy_open(path, flags, mode=0o777):
        opened_paths.append(Path(path))
        return original_open(path, flags, mode)

    def mutate_then_replay(**kwargs):
        nonlocal mutated
        if not mutated:
            package.write_bytes(original_package + b"\n")
            mutated = True
        return original_replay(**kwargs)

    monkeypatch.setattr(review_module, "replay_stage1_review_package", mutate_then_replay)
    monkeypatch.setattr(review_module.os, "open", record_package_copy_open)
    try:
        with pytest.raises(Stage1WorkflowError, match="snapshot|drift|changed"):
            _record_review(stage, report, package)
        assert mutated
        assert any(
            path.name.startswith("stage1-review-") and path.suffix == ".diff"
            for path in opened_paths
        )
        assert not (stage / "review.json").exists()
    finally:
        package.write_bytes(original_package)
        leaked = stage / "review.json"
        if leaked.is_file():
            leaked.unlink()


def test_r8_i2_review_post_write_foreign_replacement_preserves_foreign_review(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_artifacts import Stage1WorkflowError

    stage, report, package = _review_recording_inputs(
        template=r4_machine_stage,
        destination=tmp_path,
    )
    review_path = stage / "review.json"
    foreign_payload = b'{"actor":"foreign-review-after-write"}\n'
    original_write_json = ArtifactStore.write_json_exclusive
    mutated = False

    def mutate_before_write(store, relative_path, value):
        nonlocal mutated
        if Path(relative_path).as_posix() == "review.json" and not mutated:
            review_path = original_write_json(store, relative_path, value)
            review_path.write_bytes(foreign_payload)
            mutated = True
            return review_path
        return original_write_json(store, relative_path, value)

    monkeypatch.setattr(ArtifactStore, "write_json_exclusive", mutate_before_write)
    try:
        with pytest.raises(Stage1WorkflowError, match="authority|snapshot|record|bytes|drift|changed"):
            _record_review(stage, report, package)
        assert mutated
        assert review_path.read_bytes() == foreign_payload
    finally:
        if review_path.is_file():
            review_path.unlink()


def test_toctou_mutation_review_authority_during_initial_capture_leaves_no_review(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.workflows.stage1_review as review_module
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_artifacts import Stage1WorkflowError

    stage, report, package = _review_recording_inputs(
        template=r4_machine_stage,
        destination=tmp_path,
    )
    summary_path = stage / "summary.json"
    original_summary = summary_path.read_bytes()
    mutated_summary = json.loads(original_summary.decode("utf-8"))
    mutated_summary["scenario_hash"] = "0" * 64
    mutated_summary_bytes = ArtifactStore.canonical_json_bytes(mutated_summary)
    original_capture = review_module._capture_review_authority_snapshot
    capture_calls = 0
    mutated = False

    def mutate_before_initial_capture(**kwargs):
        nonlocal capture_calls, mutated
        capture_calls += 1
        if capture_calls == 1:
            summary_path.write_bytes(mutated_summary_bytes)
            mutated = True
        return original_capture(**kwargs)

    monkeypatch.setattr(
        review_module,
        "_capture_review_authority_snapshot",
        mutate_before_initial_capture,
    )
    try:
        with pytest.raises(
            Stage1WorkflowError,
            match="authority|manifest|summary|snapshot|drift|changed",
        ):
            _record_review(stage, report, package)
        assert mutated
        assert capture_calls == 1
        assert not (stage / "review.json").exists()
    finally:
        summary_path.write_bytes(original_summary)
        leaked = stage / "review.json"
        if leaked.is_file():
            leaked.unlink()


def test_toctou_aba_review_machine_inputs_returns_no_review(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.workflows.stage1_review as review_module
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_artifacts import Stage1WorkflowError

    stage, report, package = _review_recording_inputs(
        template=r4_machine_stage,
        destination=tmp_path,
    )
    summary_path = stage / "summary.json"
    manifest_path = stage / "manifest.json"
    original_summary = summary_path.read_bytes()
    original_manifest = manifest_path.read_bytes()
    invalid_summary = json.loads(original_summary.decode("utf-8"))
    invalid_summary["state"] = "machine_failed"
    invalid_summary_bytes = ArtifactStore.canonical_json_bytes(invalid_summary)
    invalid_manifest = json.loads(original_manifest.decode("utf-8"))
    for entry in invalid_manifest["artifacts"]:
        if entry["path"] == "summary.json":
            entry["size_bytes"] = len(invalid_summary_bytes)
            entry["sha256"] = hashlib.sha256(invalid_summary_bytes).hexdigest()
    invalid_manifest_bytes = ArtifactStore.canonical_json_bytes(invalid_manifest)
    original_capture = review_module._capture_review_authority_snapshot
    original_semantic_verify = review_module._verify_review_machine_snapshot
    capture_calls = 0
    semantic_calls = 0
    live_verifier_calls = 0

    def capture_b_then_restore_a(**kwargs):
        nonlocal capture_calls
        capture_calls += 1
        snapshot = original_capture(**kwargs)
        summary_path.write_bytes(invalid_summary_bytes)
        manifest_path.write_bytes(invalid_manifest_bytes)
        return snapshot

    def verify_frozen_b(*, snapshot, run_id):
        nonlocal semantic_calls
        semantic_calls += 1
        result = original_semantic_verify(snapshot=snapshot, run_id=run_id)
        assert result.summary["state"] == "machine_passed"
        return result

    def reject_live_machine_verifier(**kwargs):
        nonlocal live_verifier_calls
        live_verifier_calls += 1
        raise AssertionError("review correctness path called the legacy live verifier")

    monkeypatch.setattr(review_module, "_capture_review_authority_snapshot", capture_b_then_restore_a)
    monkeypatch.setattr(review_module, "_verify_review_machine_snapshot", verify_frozen_b)
    monkeypatch.setattr(review_module, "_verify_review_machine_inputs", reject_live_machine_verifier)
    try:
        with pytest.raises(Stage1WorkflowError, match="authority|snapshot|current|drift|changed"):
            _record_review(stage, report, package)
        assert capture_calls == 1
        assert semantic_calls == 1
        assert live_verifier_calls == 0
        assert summary_path.read_bytes() == invalid_summary_bytes
        assert manifest_path.read_bytes() == invalid_manifest_bytes
        assert not (stage / "review.json").exists()
    finally:
        summary_path.write_bytes(original_summary)
        manifest_path.write_bytes(original_manifest)
        leaked = stage / "review.json"
        if leaked.is_file():
            leaked.unlink()


def test_r8_i1_machine_semantics_are_pure_over_frozen_bytes(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.workflows.stage1_review as review_module
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_artifacts import Stage1WorkflowError

    stage, report, package = _review_recording_inputs(
        template=r4_machine_stage,
        destination=tmp_path,
    )
    snapshot = review_module._capture_review_authority_snapshot(
        stage=stage,
        repo_root=REPO_ROOT,
        foundation_gate=FOUNDATION_GATE,
        foundation_environment_manifest=FOUNDATION_ENVIRONMENT,
        review_report=report,
        review_package=package,
    )
    summary_path = stage / "summary.json"
    manifest_path = stage / "manifest.json"
    original_summary = summary_path.read_bytes()
    original_manifest = manifest_path.read_bytes()
    invalid_summary_object = json.loads(original_summary.decode("utf-8"))
    invalid_summary_object["state"] = "machine_failed"
    invalid_summary = ArtifactStore.canonical_json_bytes(invalid_summary_object)
    invalid_manifest_object = json.loads(original_manifest.decode("utf-8"))
    for entry in invalid_manifest_object["artifacts"]:
        if entry["path"] == "summary.json":
            entry["size_bytes"] = len(invalid_summary)
            entry["sha256"] = hashlib.sha256(invalid_summary).hexdigest()
    invalid_manifest = ArtifactStore.canonical_json_bytes(invalid_manifest_object)
    summary_path.write_bytes(invalid_summary)
    manifest_path.write_bytes(invalid_manifest)
    legacy_live_calls = 0

    def reject_legacy_live_helper(*args, **kwargs):
        nonlocal legacy_live_calls
        legacy_live_calls += 1
        raise AssertionError("frozen semantic verification attempted a live Path helper")

    for helper in (
        "read_object",
        "verify_manifest_entries",
        "load_stage1_machine_config",
        "compute_stage1_source_identity",
    ):
        monkeypatch.setattr(review_module, helper, reject_legacy_live_helper)
    try:
        verified = review_module._verify_review_machine_snapshot(
            snapshot=snapshot,
            run_id=stage.parent.name,
        )

        assert verified.summary["state"] == "machine_passed"
        assert legacy_live_calls == 0
        with pytest.raises(Stage1WorkflowError, match="snapshot|drift|changed"):
            review_module._require_review_authority_snapshot_current(snapshot)
    finally:
        summary_path.write_bytes(original_summary)
        manifest_path.write_bytes(original_manifest)


def test_r8_i1_repeated_aba_review_both_machine_verifiers_b_returns_no_review(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.workflows.stage1_review as review_module
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_artifacts import Stage1WorkflowError

    stage, report, package = _review_recording_inputs(
        template=r4_machine_stage,
        destination=tmp_path,
    )
    summary_path = stage / "summary.json"
    manifest_path = stage / "manifest.json"
    valid_summary_b = summary_path.read_bytes()
    valid_manifest_b = manifest_path.read_bytes()

    invalid_summary_a_object = json.loads(valid_summary_b.decode("utf-8"))
    invalid_summary_a_object["state"] = "machine_failed"
    invalid_summary_a = ArtifactStore.canonical_json_bytes(invalid_summary_a_object)
    invalid_manifest_a_object = json.loads(valid_manifest_b.decode("utf-8"))
    for entry in invalid_manifest_a_object["artifacts"]:
        if entry["path"] == "summary.json":
            entry["size_bytes"] = len(invalid_summary_a)
            entry["sha256"] = hashlib.sha256(invalid_summary_a).hexdigest()
    invalid_manifest_a = ArtifactStore.canonical_json_bytes(invalid_manifest_a_object)
    summary_path.write_bytes(invalid_summary_a)
    manifest_path.write_bytes(invalid_manifest_a)

    original_verify = review_module._verify_review_machine_inputs
    verify_calls = 0
    verified_b_results: list[object] = []

    def verify_b_then_restore_a(*, stage, repo_root, run_id):
        nonlocal verify_calls
        verify_calls += 1
        summary_path.write_bytes(valid_summary_b)
        manifest_path.write_bytes(valid_manifest_b)
        try:
            result = original_verify(stage=stage, repo_root=repo_root, run_id=run_id)
            verified_b_results.append(result)
            return result
        finally:
            summary_path.write_bytes(invalid_summary_a)
            manifest_path.write_bytes(invalid_manifest_a)

    monkeypatch.setattr(review_module, "_verify_review_machine_inputs", verify_b_then_restore_a)
    try:
        with pytest.raises(Stage1WorkflowError, match="machine|snapshot|authority|current|state"):
            _record_review(stage, report, package)
        assert verify_calls == 0
        assert not verified_b_results
        assert summary_path.read_bytes() == invalid_summary_a
        assert manifest_path.read_bytes() == invalid_manifest_a
        assert not (stage / "review.json").exists()
    finally:
        summary_path.write_bytes(valid_summary_b)
        manifest_path.write_bytes(valid_manifest_b)
        leaked = stage / "review.json"
        if leaked.is_file():
            leaked.unlink()


def test_review_package_snapshot_write_preserves_original_error_without_double_close(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.workflows.stage1_review as review_module

    class SnapshotWriteError(RuntimeError):
        pass

    class FailingPackageStream:
        def __init__(self, stream) -> None:
            self._stream = stream

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self._stream.close()
            return False

        def write(self, payload):
            raise SnapshotWriteError("sentinel package snapshot write failure")

    stage, report, package = _review_recording_inputs(
        template=r4_machine_stage,
        destination=tmp_path,
    )
    original_fdopen = review_module.os.fdopen

    def failing_fdopen(descriptor, mode):
        return FailingPackageStream(original_fdopen(descriptor, mode))

    monkeypatch.setattr(review_module.os, "fdopen", failing_fdopen)
    with pytest.raises(SnapshotWriteError, match="sentinel package snapshot write failure"):
        _record_review(stage, report, package)
    assert not (stage / "review.json").exists()


def test_r8_i2_approval_preexisting_payload_is_preserved(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows.stage1_gate import (
        Stage1GateError,
        record_stage1_human_approval,
    )

    stage, context, review = _reviewed_context_from_template(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    approval_path = stage / "approval.json"
    foreign_payload = b'{"actor":"foreign-preexisting-approval"}\n'
    approval_path.write_bytes(foreign_payload)
    review_time = datetime.fromisoformat(
        str(review["review_recorded_at_utc"]).replace("Z", "+00:00")
    )

    with pytest.raises(Stage1GateError, match="already exists|evidence already exists"):
        record_stage1_human_approval(
            context=context,
            thread_id=CANONICAL_THREAD_ID,
            user_turn_id=CANONICAL_USER_TURN_ID,
            approval_text="\u6279\u51c6 Stage 1 Gate",
            user_turn_timestamp_utc=(review_time + timedelta(seconds=1))
            .isoformat()
            .replace("+00:00", "Z"),
        )

    assert approval_path.read_bytes() == foreign_payload


def test_r8_i2_approval_exclusive_publish_race_preserves_foreign_approval(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows.stage1_gate import record_stage1_human_approval

    stage, context, review = _reviewed_context_from_template(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    approval_path = stage / "approval.json"
    foreign_payload = b'{"actor":"foreign-approval-publish-racer"}\n'
    real_link = os.link
    publish_calls = 0

    def competing_link(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        nonlocal publish_calls
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path.name == "approval.json":
            publish_calls += 1
            assert json.loads(source_path.read_bytes())["state"] == "approved"
            assert not destination_path.exists()
            destination_path.write_bytes(foreign_payload)
        real_link(source, destination)

    monkeypatch.setattr(os, "link", competing_link)
    review_time = datetime.fromisoformat(
        str(review["review_recorded_at_utc"]).replace("Z", "+00:00")
    )
    try:
        with pytest.raises(FileExistsError):
            record_stage1_human_approval(
                context=context,
                thread_id=CANONICAL_THREAD_ID,
                user_turn_id=CANONICAL_USER_TURN_ID,
                approval_text="\u6279\u51c6 Stage 1 Gate",
                user_turn_timestamp_utc=(review_time + timedelta(seconds=1))
                .isoformat()
                .replace("+00:00", "Z"),
            )
        assert publish_calls == 1
        assert approval_path.read_bytes() == foreign_payload
        assert not [path for path in stage.iterdir() if path.name.startswith(".approval.json.")]
    finally:
        if approval_path.is_file():
            approval_path.unlink()


def test_toctou_mutation_approval_after_first_compute_leaves_no_approval(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows.stage1_gate import (
        Stage1GateBindingVerifier,
        Stage1GateError,
        record_stage1_human_approval,
    )

    stage, context, review = _reviewed_context_from_template(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    package = Path(str(review["review_package"]["path"]))
    original_package = package.read_bytes()
    original_compute = Stage1GateBindingVerifier.compute.__func__
    compute_count = 0

    def mutate_after_first_compute(cls, received_context, *, target_state="awaiting_human_approval"):
        nonlocal compute_count
        compute_count += 1
        bindings = original_compute(cls, received_context, target_state=target_state)
        if compute_count == 1:
            package.write_bytes(original_package + b"\n")
        return bindings

    monkeypatch.setattr(
        Stage1GateBindingVerifier,
        "compute",
        classmethod(mutate_after_first_compute),
    )
    review_time = datetime.fromisoformat(str(review["review_recorded_at_utc"]).replace("Z", "+00:00"))
    try:
        with pytest.raises(Stage1GateError, match="review package|snapshot|drift|binding"):
            record_stage1_human_approval(
                context=context,
                thread_id=CANONICAL_THREAD_ID,
                user_turn_id=CANONICAL_USER_TURN_ID,
                approval_text="\u6279\u51c6 Stage 1 Gate",
                user_turn_timestamp_utc=(review_time + timedelta(seconds=1))
                .isoformat()
                .replace("+00:00", "Z"),
            )
        assert compute_count >= 2
        assert not (stage / "approval.json").exists()
    finally:
        package.write_bytes(original_package)
        leaked = stage / "approval.json"
        if leaked.is_file():
            leaked.unlink()


def test_r9_i1_approval_post_write_authority_drift_retains_committed_approval(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_gate import Stage1GateError, record_stage1_human_approval

    stage, context, review = _reviewed_context_from_template(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    package = Path(str(review["review_package"]["path"]))
    original_package = package.read_bytes()
    original_write_json = ArtifactStore.write_json_exclusive
    mutated = False
    published_payloads: list[bytes] = []

    def mutate_after_write(store, relative_path, value):
        nonlocal mutated
        written = original_write_json(store, relative_path, value)
        if Path(relative_path).as_posix() == "approval.json" and not mutated:
            published_payloads.append(ArtifactStore.canonical_json_bytes(value))
            package.write_bytes(original_package + b"\n")
            mutated = True
        return written

    monkeypatch.setattr(ArtifactStore, "write_json_exclusive", mutate_after_write)
    review_time = datetime.fromisoformat(str(review["review_recorded_at_utc"]).replace("Z", "+00:00"))
    try:
        with pytest.raises(Stage1GateError, match="review package|snapshot|drift|binding"):
            record_stage1_human_approval(
                context=context,
                thread_id=CANONICAL_THREAD_ID,
                user_turn_id=CANONICAL_USER_TURN_ID,
                approval_text="\u6279\u51c6 Stage 1 Gate",
                user_turn_timestamp_utc=(review_time + timedelta(seconds=1))
                .isoformat()
                .replace("+00:00", "Z"),
            )
        assert mutated
        assert len(published_payloads) == 1
        assert (stage / "approval.json").read_bytes() == published_payloads[0]
    finally:
        package.write_bytes(original_package)
        leaked = stage / "approval.json"
        if leaked.is_file():
            leaked.unlink()


def test_r9_i1_approval_post_publish_failure_retains_owned_payload_and_retry_fails_closed(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_gate import (
        Stage1GateError,
        record_stage1_human_approval,
    )

    class SentinelWriteError(RuntimeError):
        pass

    stage, context, review = _reviewed_context_from_template(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    approval_path = stage / "approval.json"
    original_write_json = ArtifactStore.write_json_exclusive
    published_payloads: list[bytes] = []

    def write_then_raise(store, relative_path, value):
        written = original_write_json(store, relative_path, value)
        if Path(relative_path).as_posix() == "approval.json":
            published_payloads.append(ArtifactStore.canonical_json_bytes(value))
            assert written.read_bytes() == published_payloads[-1]
            raise SentinelWriteError("sentinel approval write-then-raise")
        return written

    monkeypatch.setattr(ArtifactStore, "write_json_exclusive", write_then_raise)
    review_time = datetime.fromisoformat(
        str(review["review_recorded_at_utc"]).replace("Z", "+00:00")
    )
    invoke = lambda: record_stage1_human_approval(
        context=context,
        thread_id=CANONICAL_THREAD_ID,
        user_turn_id=CANONICAL_USER_TURN_ID,
        approval_text="\u6279\u51c6 Stage 1 Gate",
        user_turn_timestamp_utc=(review_time + timedelta(seconds=1))
        .isoformat()
        .replace("+00:00", "Z"),
    )
    try:
        with pytest.raises(SentinelWriteError, match="sentinel approval write-then-raise"):
            invoke()
        assert len(published_payloads) == 1
        assert approval_path.read_bytes() == published_payloads[0]

        with pytest.raises(Stage1GateError, match="already exists"):
            invoke()
        assert len(published_payloads) == 1
        assert approval_path.read_bytes() == published_payloads[0]
    finally:
        if approval_path.is_file():
            approval_path.unlink()


def test_r7_i2_approval_parses_and_hashes_one_immutable_review_read(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_gate import (
        Stage1GateBindingVerifier,
        Stage1GateError,
        record_stage1_human_approval,
    )

    stage, context, review = _reviewed_context_from_template(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    review_path = (stage / "review.json").resolve()
    approval_path = stage / "approval.json"
    review_a = review_path.read_bytes()
    review_b_object = json.loads(review_a.decode("utf-8"))
    review_b_object["approval_challenge"] = "f" * 64
    review_b = ArtifactStore.canonical_json_bytes(review_b_object)
    review_b_hash = hashlib.sha256(review_b).hexdigest()
    assert review_b_hash != hashlib.sha256(review_a).hexdigest()

    waiting_bindings = Stage1GateBindingVerifier.compute(
        context,
        target_state="awaiting_human_approval",
    ).model_copy(update={"review_hash": review_b_hash})
    original_read_bytes = Path.read_bytes
    review_read_count = 0

    def alternating_review_read(path):
        nonlocal review_read_count
        if path.resolve() == review_path:
            review_read_count += 1
            if review_read_count == 1:
                return review_a
            if review_read_count == 2:
                return review_b
            raise AssertionError("approval recorder read review.json more than twice")
        return original_read_bytes(path)

    def bindings_from_review_b(cls, received_context, *, target_state="awaiting_human_approval"):
        assert received_context == context
        if target_state == "awaiting_human_approval":
            return waiting_bindings
        if target_state == "approved":
            return waiting_bindings.model_copy(
                update={
                    "approval_state": "approved",
                    "approval_hash": hashlib.sha256(original_read_bytes(approval_path)).hexdigest(),
                }
            )
        raise AssertionError(f"unexpected target state: {target_state}")

    monkeypatch.setattr(Path, "read_bytes", alternating_review_read)
    monkeypatch.setattr(
        Stage1GateBindingVerifier,
        "compute",
        classmethod(bindings_from_review_b),
    )
    review_time = datetime.fromisoformat(
        str(review["review_recorded_at_utc"]).replace("Z", "+00:00")
    )
    try:
        with pytest.raises(Stage1GateError, match="review binding drifted"):
            record_stage1_human_approval(
                context=context,
                thread_id=CANONICAL_THREAD_ID,
                user_turn_id=CANONICAL_USER_TURN_ID,
                approval_text="\u6279\u51c6 Stage 1 Gate",
                user_turn_timestamp_utc=(review_time + timedelta(seconds=1))
                .isoformat()
                .replace("+00:00", "Z"),
            )
        assert review_read_count == 1
        assert not approval_path.exists()
    finally:
        if approval_path.is_file():
            approval_path.unlink()


def test_r9_i1_approval_post_publish_failure_propagates_original_without_canonical_unlink(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_gate import record_stage1_human_approval

    class SentinelWriteError(RuntimeError):
        pass

    stage, context, review = _reviewed_context_from_template(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    approval_path = (stage / "approval.json").resolve()
    original_write_json = ArtifactStore.write_json_exclusive
    original_unlink = Path.unlink
    published_payloads: list[bytes] = []
    canonical_unlink_calls = 0

    def write_then_raise(store, relative_path, value):
        written = original_write_json(store, relative_path, value)
        if Path(relative_path).as_posix() == "approval.json":
            published_payloads.append(ArtifactStore.canonical_json_bytes(value))
            raise SentinelWriteError("sentinel approval write failure")
        return written

    def reject_approval_unlink(path, *args, **kwargs):
        nonlocal canonical_unlink_calls
        if path.resolve() == approval_path:
            canonical_unlink_calls += 1
            raise PermissionError("sentinel approval rollback unlink failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(ArtifactStore, "write_json_exclusive", write_then_raise)
    monkeypatch.setattr(Path, "unlink", reject_approval_unlink)
    review_time = datetime.fromisoformat(
        str(review["review_recorded_at_utc"]).replace("Z", "+00:00")
    )
    try:
        with pytest.raises(SentinelWriteError, match="approval write failure"):
            record_stage1_human_approval(
                context=context,
                thread_id=CANONICAL_THREAD_ID,
                user_turn_id=CANONICAL_USER_TURN_ID,
                approval_text="\u6279\u51c6 Stage 1 Gate",
                user_turn_timestamp_utc=(review_time + timedelta(seconds=1))
                .isoformat()
                .replace("+00:00", "Z"),
            )
        assert canonical_unlink_calls == 0
        assert len(published_payloads) == 1
        assert approval_path.read_bytes() == published_payloads[0]
    finally:
        if approval_path.is_file():
            original_unlink(approval_path)


def test_r8_i2_approval_post_write_foreign_replacement_preserves_foreign_approval(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_gate import (
        Stage1GateBindingVerifier,
        record_stage1_human_approval,
    )

    class SentinelPostVerifyError(RuntimeError):
        pass

    stage, context, review = _reviewed_context_from_template(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    approval_path = stage / "approval.json"
    foreign_payload = ArtifactStore.canonical_json_bytes(
        {"actor": "another-process", "state": "foreign-approval"}
    )
    waiting_bindings = Stage1GateBindingVerifier.compute(
        context,
        target_state="awaiting_human_approval",
    )

    def replace_during_post_verify(
        cls,
        received_context,
        *,
        target_state="awaiting_human_approval",
    ):
        assert received_context == context
        if target_state == "awaiting_human_approval":
            return waiting_bindings
        if target_state == "approved":
            approval_path.write_bytes(foreign_payload)
            raise SentinelPostVerifyError("sentinel post-write verification failure")
        raise AssertionError(f"unexpected target state: {target_state}")

    monkeypatch.setattr(
        Stage1GateBindingVerifier,
        "compute",
        classmethod(replace_during_post_verify),
    )
    review_time = datetime.fromisoformat(
        str(review["review_recorded_at_utc"]).replace("Z", "+00:00")
    )
    try:
        with pytest.raises(
            SentinelPostVerifyError,
            match="sentinel post-write verification failure",
        ):
            record_stage1_human_approval(
                context=context,
                thread_id=CANONICAL_THREAD_ID,
                user_turn_id=CANONICAL_USER_TURN_ID,
                approval_text="\u6279\u51c6 Stage 1 Gate",
                user_turn_timestamp_utc=(review_time + timedelta(seconds=1))
                .isoformat()
                .replace("+00:00", "Z"),
            )
        assert approval_path.read_bytes() == foreign_payload
    finally:
        if approval_path.is_file():
            approval_path.unlink()


def test_r8_i2_approval_rollback_treats_missing_target_as_unowned(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_gate import record_stage1_human_approval

    class SentinelPreReplaceError(RuntimeError):
        pass

    stage, context, review = _reviewed_context_from_template(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    approval_path = stage / "approval.json"

    def fail_before_replace(store, relative_path, value):
        if Path(relative_path).as_posix() == "approval.json":
            raise SentinelPreReplaceError("sentinel pre-replace write failure")
        raise AssertionError(f"unexpected artifact write: {relative_path}")

    monkeypatch.setattr(ArtifactStore, "write_json_exclusive", fail_before_replace)
    review_time = datetime.fromisoformat(
        str(review["review_recorded_at_utc"]).replace("Z", "+00:00")
    )
    with pytest.raises(SentinelPreReplaceError, match="sentinel pre-replace write failure"):
        record_stage1_human_approval(
            context=context,
            thread_id=CANONICAL_THREAD_ID,
            user_turn_id=CANONICAL_USER_TURN_ID,
            approval_text="\u6279\u51c6 Stage 1 Gate",
            user_turn_timestamp_utc=(review_time + timedelta(seconds=1))
            .isoformat()
            .replace("+00:00", "Z"),
        )
    assert not approval_path.exists()


def test_i1_approval_recorder_requires_full_gate_verification_before_write(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows.stage1_gate import (
        Stage1GateBindingVerifier,
        Stage1GateError,
        record_stage1_human_approval,
    )

    stage, context, review = _reviewed_context_from_template(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    review_time = datetime.fromisoformat(
        str(review["review_recorded_at_utc"]).replace("Z", "+00:00")
    )

    def reject_full_verification(cls, received_context, *, target_state="awaiting_human_approval"):
        assert received_context == context
        assert target_state == "awaiting_human_approval"
        raise Stage1GateError("sentinel full gate verification")

    monkeypatch.setattr(
        Stage1GateBindingVerifier,
        "compute",
        classmethod(reject_full_verification),
    )
    with pytest.raises(Stage1GateError, match="sentinel full gate verification"):
        record_stage1_human_approval(
            context=context,
            thread_id=CANONICAL_THREAD_ID,
            user_turn_id=CANONICAL_USER_TURN_ID,
            approval_text="\u6279\u51c6 Stage 1 Gate",
            user_turn_timestamp_utc=(review_time + timedelta(seconds=1))
            .isoformat()
            .replace("+00:00", "Z"),
        )
    assert not (stage / "approval.json").exists()


def test_human_approval_v2_rejects_non_uuid_thread_id(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows.stage1_gate import (
        Stage1GateError,
        record_stage1_human_approval,
    )

    stage, context, review = _reviewed_context_from_template(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    review_time = datetime.fromisoformat(
        str(review["review_recorded_at_utc"]).replace("Z", "+00:00")
    )

    with pytest.raises(Stage1GateError, match="thread ID"):
        record_stage1_human_approval(
            context=context,
            stage_root=stage,
            thread_id="019f-stage1-thread",
            user_turn_id=CANONICAL_USER_TURN_ID,
            approval_text="\u6279\u51c6 Stage 1 Gate",
            user_turn_timestamp_utc=(review_time + timedelta(seconds=1))
            .isoformat()
            .replace("+00:00", "Z"),
        )
    assert not (stage / "approval.json").exists()


def test_human_approval_v2_binds_real_codex_user_turn_and_rejects_all_drift(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_gate import (
        Stage1GateError,
        Stage1GateRecord,
        record_stage1_human_approval,
    )

    stage, context, review = _reviewed_context_from_template(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    record = Stage1GateRecord.machine_passed(context)
    record = record.transition("awaiting_independent_review")
    record = record.transition("awaiting_human_approval")
    store = ArtifactStore(stage)
    old_v1 = {
        "schema_version": "ppo_highres_frontier_stage1_human_approval/v1",
        "run_id": context.run_id,
        "state": "approved",
        "reviewed_git_tree": review["reviewed_git_tree"],
    }
    store.write_json("approval.json", old_v1)
    with pytest.raises(Stage1GateError, match="approval"):
        record.transition("approved")
    (stage / "approval.json").unlink()

    exact_text = "\u6279\u51c6 Stage 1 Gate"
    review_time = datetime.fromisoformat(
        str(review["review_recorded_at_utc"]).replace("Z", "+00:00")
    )
    approval_time = (review_time + timedelta(seconds=1)).astimezone(timezone.utc)
    approval_time_text = approval_time.isoformat().replace("+00:00", "Z")
    with pytest.raises(Stage1GateError, match="exact approval text"):
        record_stage1_human_approval(
            context=context,
            stage_root=stage,
            thread_id=CANONICAL_THREAD_ID,
            user_turn_id=CANONICAL_USER_TURN_ID,
            approval_text=exact_text + "!",
            user_turn_timestamp_utc=approval_time_text,
        )
    assert not (stage / "approval.json").exists()

    approval_path = record_stage1_human_approval(
        context=context,
        stage_root=stage,
        thread_id=CANONICAL_THREAD_ID,
        user_turn_id=CANONICAL_USER_TURN_ID,
        approval_text=exact_text,
        user_turn_timestamp_utc=approval_time_text,
    )
    evidence = json.loads(approval_path.read_text(encoding="utf-8"))
    assert evidence["authority_kind"] == "codex_user_message/v1"
    assert evidence["actor"] == "user"
    assert evidence["thread_id"] == CANONICAL_THREAD_ID
    assert evidence["user_turn_id"] == CANONICAL_USER_TURN_ID
    assert evidence["approval_text"] == exact_text
    assert evidence["approval_text_utf8_sha256"] == hashlib.sha256(
        exact_text.encode("utf-8")
    ).hexdigest()
    assert evidence["approval_challenge"] == review["approval_challenge"]
    assert evidence["review_hash"] == hashlib.sha256(
        (stage / "review.json").read_bytes()
    ).hexdigest()

    invalid_evidence = [
        {key: value for key, value in evidence.items() if key != "actor"},
        evidence | {"extra": True},
        evidence | {"schema_version": "ppo_highres_frontier_stage1_human_approval/v1"},
        evidence | {"authority_kind": "local_file/v1"},
        evidence | {"actor": "assistant"},
        evidence | {"thread_id": ""},
        evidence | {"thread_id": "019f-stage1-thread"},
        evidence | {"user_turn_id": "not-a-turn-id"},
        evidence | {"approval_text": exact_text + "!"},
        evidence | {"approval_text_utf8_sha256": "0" * 64},
        evidence | {"goal_id": "different-goal"},
        evidence | {"stage_id": "different-stage"},
        evidence | {"run_id": "different-run"},
        evidence | {"reviewed_git_tree": "0" * len(review["reviewed_git_tree"])},
        evidence | {"review_hash": "0" * 64},
        evidence | {"approval_challenge": "0" * 64},
        evidence
        | {
            "approval_timestamp_utc": (
                review_time - timedelta(seconds=1)
            ).isoformat().replace("+00:00", "Z")
        },
    ]
    valid_bytes = approval_path.read_bytes()
    for invalid in invalid_evidence:
        approval_path.write_bytes(ArtifactStore.canonical_json_bytes(invalid))
        with pytest.raises(Stage1GateError, match="approval"):
            record.transition("approved")
    approval_path.write_bytes(valid_bytes)

    approved = record.transition("approved")
    assert approved.bindings.approval_state == "approved"
    assert approved.bindings.approval_hash == hashlib.sha256(valid_bytes).hexdigest()
    doc = inspect.getdoc(record_stage1_human_approval) or ""
    normalized_doc = " ".join(doc.lower().split())
    assert "platform signature" in normalized_doc
    assert "not cryptographically unforgeable" in normalized_doc


def test_review_controller_hashes_real_sources_and_records_prospective_tree(tmp_path: Path) -> None:
    from lunar_exploration_ppo.workflows.stage1 import record_stage1_independent_review
    from lunar_exploration_ppo.workflows.stage1_source import compute_stage1_prospective_tree

    workflow = _run_machine(tmp_path, "r1-review-binding")
    review_package = tmp_path / "review-package.diff"
    _write_git_native_review_package(review_package)
    review_report = tmp_path / "review-report.md"
    prospective_tree = compute_stage1_prospective_tree(
        REPO_ROOT,
        base_commit=FOUNDATION_BASE_COMMIT,
    )
    _write_clean_review_report(
        review_report,
        review_package=review_package,
        reviewed_tree=prospective_tree,
    )
    staged_before = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout

    result = record_stage1_independent_review(
        stage_root=workflow.stage_root,
        repo_root=REPO_ROOT,
        foundation_gate=FOUNDATION_GATE,
        foundation_environment_manifest=FOUNDATION_ENVIRONMENT,
        review_report=review_report,
        review_package=review_package,
    )

    review = json.loads(result.review_path.read_text(encoding="utf-8"))
    assert review["foundation_base_commit"] == FOUNDATION_BASE_COMMIT
    assert review["reviewed_git_tree"] != FOUNDATION_BASE_COMMIT
    assert len(review["reviewed_git_tree"]) == 40
    assert review["review_report"] == {
        "path": str(review_report.resolve()),
        "sha256": hashlib.sha256(review_report.read_bytes()).hexdigest(),
        "bytes": len(review_report.read_bytes()),
        "lf_count": review_report.read_bytes().count(b"\n"),
        "logical_line_count": len(review_report.read_bytes().splitlines()),
    }
    assert review["review_package"] == {
        "path": str(review_package.resolve()),
        "sha256": hashlib.sha256(review_package.read_bytes()).hexdigest(),
        "bytes": len(review_package.read_bytes()),
        "lf_count": review_package.read_bytes().count(b"\n"),
        "logical_line_count": len(review_package.read_bytes().splitlines()),
    }
    staged_after = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    assert staged_before == staged_after == ""


def test_stage2_authority_is_frozen_in_reviewed_stage1_config() -> None:
    from lunar_exploration_ppo.configs.stage1 import Stage1Config, load_stage1_config

    config = load_stage1_config(STAGE1_CONFIG)
    assert config.authorized_next_stage == "ppo_highres_frontier_stage2_observation_frontier/v1"
    payload = config.model_dump(mode="json")
    missing = dict(payload)
    missing.pop("authorized_next_stage")
    with pytest.raises(ValidationError):
        Stage1Config.model_validate(missing)
    with pytest.raises(ValidationError):
        Stage1Config.model_validate(
            payload | {"authorized_next_stage": "ppo_highres_frontier_stage2_policy_network/v1"}
        )


def test_stage1_gate_scenario_identity_binds_proxy_provenance_and_rejects_drift() -> None:
    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.workflows.stage1_gate import (
        Stage1GateBindingVerifier,
        Stage1GateError,
    )

    config = load_stage1_config(STAGE1_CONFIG)
    summary = {
        "scenario_id": "smoke-v1/procedural-rock-crater/v1",
        "scenario_hash": "1" * 64,
        "coverage_mask": {"sha256": "2" * 64},
        "proxy_morphology": {
            "proxy_generator_version": "procedural_lunar_rock_crater_proxy/v1",
            "density_profile": "medium",
            "generation_attempt": 0,
            "rock_count": 24,
            "crater_count": 4,
            "object_catalog_sha256": "3" * 64,
            "layer_hashes": {
                "height": "4" * 64,
                "hard_obstacle": "5" * 64,
                "slope": "6" * 64,
                "traversability": "7" * 64,
            },
        },
    }
    baseline = Stage1GateBindingVerifier._scenario_identity_hash(summary, config)
    catalog_drift = json.loads(json.dumps(summary))
    catalog_drift["proxy_morphology"]["object_catalog_sha256"] = "8" * 64
    assert Stage1GateBindingVerifier._scenario_identity_hash(catalog_drift, config) != baseline

    invalid_summaries = []
    missing = json.loads(json.dumps(summary))
    missing["proxy_morphology"].pop("layer_hashes")
    invalid_summaries.append(missing)
    extra = json.loads(json.dumps(summary))
    extra["proxy_morphology"]["catalog"] = []
    invalid_summaries.append(extra)
    invalid_hash = json.loads(json.dumps(summary))
    invalid_hash["proxy_morphology"]["layer_hashes"]["height"] = "short"
    invalid_summaries.append(invalid_hash)
    count_drift = json.loads(json.dumps(summary))
    count_drift["proxy_morphology"]["rock_count"] = 23
    invalid_summaries.append(count_drift)
    profile_drift = json.loads(json.dumps(summary))
    profile_drift["proxy_morphology"]["density_profile"] = "high"
    invalid_summaries.append(profile_drift)

    for invalid in invalid_summaries:
        with pytest.raises(Stage1GateError, match="proxy"):
            Stage1GateBindingVerifier._scenario_identity_hash(invalid, config)


def _reviewed_gate_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from lunar_exploration_ppo.workflows.stage1 import record_stage1_independent_review
    from lunar_exploration_ppo.workflows.stage1_gate import Stage1GateBindingVerifier, Stage1GateContext

    workflow = _run_machine(tmp_path, "r1-gate-binding")
    review_report = tmp_path / "review-report.md"
    review_package = tmp_path / "review-package.diff"
    _write_git_native_review_package(review_package)
    from lunar_exploration_ppo.workflows.stage1_source import compute_stage1_prospective_tree

    prospective_tree = compute_stage1_prospective_tree(
        REPO_ROOT,
        base_commit=FOUNDATION_BASE_COMMIT,
    )
    _write_clean_review_report(
        review_report,
        review_package=review_package,
        reviewed_tree=prospective_tree,
    )
    review_result = record_stage1_independent_review(
        stage_root=workflow.stage_root,
        repo_root=REPO_ROOT,
        foundation_gate=FOUNDATION_GATE,
        foundation_environment_manifest=FOUNDATION_ENVIRONMENT,
        review_report=review_report,
        review_package=review_package,
    )
    review = json.loads(review_result.review_path.read_text(encoding="utf-8"))
    _install_gate_committed_source_snapshot_fixture(monkeypatch)
    reviewed_tree = review["reviewed_git_tree"]
    repository_result = classmethod(lambda cls, repo_root: (REPO_ROOT.resolve(), reviewed_tree))
    monkeypatch.setattr(
        Stage1GateBindingVerifier,
        "_verify_clean_repository",
        repository_result,
        raising=False,
    )
    monkeypatch.setattr(
        Stage1GateBindingVerifier,
        "_verify_repository_identity",
        repository_result,
        raising=False,
    )
    context = Stage1GateContext(
        repo_root=REPO_ROOT,
        stage_root=workflow.stage_root,
        foundation_gate=FOUNDATION_GATE,
        foundation_environment_manifest=FOUNDATION_ENVIRONMENT,
        run_id="r1-gate-binding",
    )
    return workflow.stage_root, context, review_report, review_package, reviewed_tree


def test_toctou_mutation_gate_compute_after_verified_inputs_returns_no_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows.stage1_gate import Stage1GateBindingVerifier, Stage1GateError

    _, context, review_report, _, _ = _reviewed_gate_context(tmp_path, monkeypatch)
    original_report = review_report.read_bytes()
    original_scenario_hash = Stage1GateBindingVerifier._scenario_identity_hash.__func__
    mutated = False

    def mutate_after_verified_inputs(cls, summary, config):
        nonlocal mutated
        result = original_scenario_hash(cls, summary, config)
        if not mutated:
            review_report.write_bytes(original_report + b"concurrent gate mutation\n")
            mutated = True
        return result

    monkeypatch.setattr(
        Stage1GateBindingVerifier,
        "_scenario_identity_hash",
        classmethod(mutate_after_verified_inputs),
    )
    try:
        with pytest.raises(Stage1GateError, match="authority|snapshot|drift|changed"):
            Stage1GateBindingVerifier.compute(context)
        assert mutated
    finally:
        review_report.write_bytes(original_report)


def test_toctou_aba_gate_review_returns_no_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_gate import Stage1GateBindingVerifier, Stage1GateError
    from lunar_exploration_ppo.workflows.stage1_review import build_stage1_approval_challenge

    stage, context, review_report, _, _ = _reviewed_gate_context(tmp_path, monkeypatch)
    review_path = stage / "review.json"
    original_report = review_report.read_bytes()
    original_review = review_path.read_bytes()
    mutated_report = original_report + b"\nABA review report mutation\n"
    mutated_review = json.loads(original_review.decode("utf-8"))
    mutated_review["review_report"] = {
        "path": str(review_report.resolve()),
        "sha256": hashlib.sha256(mutated_report).hexdigest(),
        "bytes": len(mutated_report),
        "lf_count": mutated_report.count(b"\n"),
        "logical_line_count": len(mutated_report.splitlines()),
    }
    mutated_review["approval_challenge"] = build_stage1_approval_challenge(
        goal_id=str(mutated_review["goal_id"]),
        stage_id=str(mutated_review["stage_id"]),
        run_id=str(mutated_review["run_id"]),
        reviewed_git_tree=str(mutated_review["reviewed_git_tree"]),
        review_report_sha256=str(mutated_review["review_report"]["sha256"]),
        review_package_sha256=str(mutated_review["review_package"]["sha256"]),
        manifest_hash=str(mutated_review["manifest_hash"]),
    )
    mutated_review_bytes = ArtifactStore.canonical_json_bytes(mutated_review)
    original_capture = Stage1GateBindingVerifier._capture_authority_snapshot.__func__
    verified_b = False

    def capture_b_then_restore_a(cls, received_context, target_state):
        nonlocal verified_b
        review_report.write_bytes(mutated_report)
        review_path.write_bytes(mutated_review_bytes)
        try:
            snapshot = original_capture(cls, received_context, target_state)
            verified_b = (
                snapshot.stage.file("review.json").payload == mutated_review_bytes
                and snapshot.review_report.payload == mutated_report
            )
            return snapshot
        finally:
            review_report.write_bytes(original_report)
            review_path.write_bytes(original_review)

    monkeypatch.setattr(
        Stage1GateBindingVerifier,
        "_capture_authority_snapshot",
        classmethod(capture_b_then_restore_a),
    )
    try:
        with pytest.raises(Stage1GateError, match="authority|snapshot|binding|changed"):
            Stage1GateBindingVerifier.compute(context)
        assert verified_b
    finally:
        review_report.write_bytes(original_report)
        review_path.write_bytes(original_review)


def test_r8_i1_segmented_aba_gate_routing_phase_b_only_during_semantic_reads_returns_no_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_gate import (
        Stage1GateBindingVerifier,
        Stage1GateError,
    )
    from lunar_exploration_ppo.workflows.stage1_review import build_stage1_approval_challenge

    stage, context, _, _, _ = _reviewed_gate_context(tmp_path, monkeypatch)
    routing_path = (stage / "routing.json").resolve()
    phase_path = (stage / "phase-state.jsonl").resolve()
    manifest_path = (stage / "manifest.json").resolve()
    review_path = (stage / "review.json").resolve()

    valid_routing_b = routing_path.read_bytes()
    valid_phase_b = phase_path.read_bytes()
    valid_manifest_b = manifest_path.read_bytes()
    valid_review_b = review_path.read_bytes()

    invalid_routing_a_object = json.loads(valid_routing_b.decode("utf-8"))
    invalid_routing_a_object["route"] = "segmented_aba_invalid"
    invalid_routing_a = ArtifactStore.canonical_json_bytes(invalid_routing_a_object)

    invalid_phase_a_events = [
        json.loads(line) for line in valid_phase_b.decode("utf-8").splitlines()
    ]
    assert len(invalid_phase_a_events) == 1
    invalid_phase_a_events[0]["state"] = "machine_failed"
    invalid_phase_a = b"".join(
        ArtifactStore.canonical_json_bytes(event) for event in invalid_phase_a_events
    )

    invalid_manifest_a_object = json.loads(valid_manifest_b.decode("utf-8"))
    invalid_payloads = {
        "routing.json": invalid_routing_a,
        "phase-state.jsonl": invalid_phase_a,
    }
    for entry in invalid_manifest_a_object["artifacts"]:
        payload = invalid_payloads.get(entry["path"])
        if payload is not None:
            entry["size_bytes"] = len(payload)
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
    invalid_manifest_a = ArtifactStore.canonical_json_bytes(invalid_manifest_a_object)

    invalid_review_a_object = json.loads(valid_review_b.decode("utf-8"))
    invalid_review_a_object["manifest_hash"] = hashlib.sha256(invalid_manifest_a).hexdigest()
    invalid_review_a_object["approval_challenge"] = build_stage1_approval_challenge(
        goal_id=str(invalid_review_a_object["goal_id"]),
        stage_id=str(invalid_review_a_object["stage_id"]),
        run_id=str(invalid_review_a_object["run_id"]),
        reviewed_git_tree=str(invalid_review_a_object["reviewed_git_tree"]),
        review_report_sha256=str(invalid_review_a_object["review_report"]["sha256"]),
        review_package_sha256=str(invalid_review_a_object["review_package"]["sha256"]),
        manifest_hash=str(invalid_review_a_object["manifest_hash"]),
    )
    invalid_review_a = ArtifactStore.canonical_json_bytes(invalid_review_a_object)

    routing_path.write_bytes(invalid_routing_a)
    phase_path.write_bytes(invalid_phase_a)
    manifest_path.write_bytes(invalid_manifest_a)
    review_path.write_bytes(invalid_review_a)

    original_read_object = Stage1GateBindingVerifier._read_object
    original_verify_phase = Stage1GateBindingVerifier._verify_phase_state.__func__
    routing_live_calls = 0
    phase_live_calls = 0

    def routing_b_only(cls, path, label):
        nonlocal routing_live_calls
        if path.expanduser().resolve() != routing_path:
            return original_read_object(path, label)
        routing_live_calls += 1
        assert routing_path.read_bytes() == invalid_routing_a
        routing_path.write_bytes(valid_routing_b)
        try:
            return original_read_object(path, label)
        finally:
            routing_path.write_bytes(invalid_routing_a)

    def phase_b_only(cls, path, run_id):
        nonlocal phase_live_calls
        assert path.expanduser().resolve() == phase_path
        phase_live_calls += 1
        assert phase_path.read_bytes() == invalid_phase_a
        phase_path.write_bytes(valid_phase_b)
        try:
            return original_verify_phase(cls, path, run_id)
        finally:
            phase_path.write_bytes(invalid_phase_a)

    monkeypatch.setattr(Stage1GateBindingVerifier, "_read_object", classmethod(routing_b_only))
    monkeypatch.setattr(
        Stage1GateBindingVerifier,
        "_verify_phase_state",
        classmethod(phase_b_only),
    )
    try:
        with pytest.raises(Stage1GateError, match="routing|phase|snapshot|authority|state"):
            Stage1GateBindingVerifier.compute(context)
        assert routing_live_calls == 0
        assert phase_live_calls == 0
        assert routing_path.read_bytes() == invalid_routing_a
        assert phase_path.read_bytes() == invalid_phase_a
    finally:
        routing_path.write_bytes(valid_routing_b)
        phase_path.write_bytes(valid_phase_b)
        manifest_path.write_bytes(valid_manifest_b)
        review_path.write_bytes(valid_review_b)


def test_r8_i1_gate_valid_snapshot_semantics_never_call_legacy_live_read_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.workflows.stage1_gate as gate_module
    from lunar_exploration_ppo.workflows.stage1_gate import Stage1GateBindingVerifier

    _, context, _, _, reviewed_tree = _reviewed_gate_context(tmp_path, monkeypatch)
    legacy_calls: list[str] = []

    def reject_legacy_helper(*args, **kwargs):
        legacy_calls.append("called")
        raise AssertionError("gate snapshot semantics called a legacy live-read helper")

    for helper in (
        "_read_object",
        "_verify_phase_state",
        "_verify_manifest",
        "_verify_repository_identity",
        "_verify_clean_repository",
        "_verify_execution_source_identity",
        "_verify_foundation_sources",
        "_verify_environment_manifest",
        "_verify_stage_artifact_set",
        "_verify_review",
        "_verify_review_source",
        "_foundation_environment_gate_hash",
        "_authority_fingerprint",
    ):
        monkeypatch.setattr(
            Stage1GateBindingVerifier,
            helper,
            classmethod(reject_legacy_helper),
        )
    for helper in (
        "load_stage1_machine_config",
        "verify_stage1_review_evidence",
        "compute_stage1_reviewed_source_set_sha256",
        "resolve_stage1_import_identity",
    ):
        monkeypatch.setattr(gate_module, helper, reject_legacy_helper)

    bindings = Stage1GateBindingVerifier.compute(context)

    assert bindings.git_tree_hash == reviewed_tree
    assert legacy_calls == []


def test_r8_i1_gate_capture_valid_b_then_restore_a_is_rejected_by_current_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_gate import (
        Stage1GateBindingVerifier,
        Stage1GateError,
    )

    stage, context, _, _, _ = _reviewed_gate_context(tmp_path, monkeypatch)
    routing_path = (stage / "routing.json").resolve()
    routing_b = routing_path.read_bytes()
    routing_a_object = json.loads(routing_b.decode("utf-8"))
    routing_a_object["route"] = "capture_b_restore_a_invalid"
    routing_a = ArtifactStore.canonical_json_bytes(routing_a_object)
    original_capture = Stage1GateBindingVerifier._capture_authority_snapshot.__func__
    capture_calls = 0
    legacy_calls = 0

    def capture_b_then_restore_a(cls, received_context, target_state):
        nonlocal capture_calls
        capture_calls += 1
        snapshot = original_capture(cls, received_context, target_state)
        routing_path.write_bytes(routing_a)
        return snapshot

    def reject_legacy_helper(*args, **kwargs):
        nonlocal legacy_calls
        legacy_calls += 1
        raise AssertionError("gate current-check test reached a legacy live-read helper")

    monkeypatch.setattr(
        Stage1GateBindingVerifier,
        "_capture_authority_snapshot",
        classmethod(capture_b_then_restore_a),
    )
    for helper in ("_read_object", "_verify_phase_state", "_verify_manifest"):
        monkeypatch.setattr(
            Stage1GateBindingVerifier,
            helper,
            classmethod(reject_legacy_helper),
        )
    try:
        with pytest.raises(Stage1GateError, match="snapshot|current|drift|authority"):
            Stage1GateBindingVerifier.compute(context)
        assert capture_calls == 1
        assert legacy_calls == 0
        assert routing_path.read_bytes() == routing_a
    finally:
        routing_path.write_bytes(routing_b)


def test_r8_i1_gate_current_check_rejects_each_frozen_authority_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from lunar_exploration_ppo.workflows.stage1_gate import (
        Stage1GateBindingVerifier,
        Stage1GateError,
    )

    stage, context, review_report, review_package, _ = _reviewed_gate_context(
        tmp_path,
        monkeypatch,
    )
    original_capture = Stage1GateBindingVerifier._capture_authority_snapshot.__func__
    paths = {
        "phase": stage / "phase-state.jsonl",
        "report": review_report,
        "package": review_package,
    }

    for drift_kind in ("phase", "source", "foundation", "report", "package"):
        restore: tuple[Path, bytes] | None = None
        capture_calls = 0

        def capture_then_drift(cls, received_context, target_state):
            nonlocal capture_calls, restore
            capture_calls += 1
            snapshot = original_capture(cls, received_context, target_state)
            if drift_kind in paths:
                drift_path = paths[drift_kind]
                original_payload = drift_path.read_bytes()
                restore = (drift_path, original_payload)
                drift_path.write_bytes(original_payload + b"\n")
                return snapshot
            if drift_kind == "foundation":
                identity = snapshot.foundation_gate._identity
                stale_gate = replace(
                    snapshot.foundation_gate,
                    _identity=(*identity[:-1], identity[-1] + 1),
                )
                return replace(snapshot, foundation_gate=stale_gate)
            relative = "tests/ppo_highres_frontier/test_foundation.py"
            source_file = snapshot.source.file(relative)
            identity = source_file._identity
            stale_file = replace(
                source_file,
                _identity=(*identity[:-1], identity[-1] + 1),
            )
            stale_source = replace(
                snapshot.source,
                files=tuple(
                    (name, stale_file if name == relative else frozen)
                    for name, frozen in snapshot.source.files
                ),
            )
            return replace(snapshot, source=stale_source)

        monkeypatch.setattr(
            Stage1GateBindingVerifier,
            "_capture_authority_snapshot",
            classmethod(capture_then_drift),
        )
        try:
            with pytest.raises(Stage1GateError, match="snapshot|current|drift|authority"):
                Stage1GateBindingVerifier.compute(context)
            assert capture_calls == 1
        finally:
            if restore is not None:
                restore[0].write_bytes(restore[1])
            restore = None
            monkeypatch.setattr(
                Stage1GateBindingVerifier,
                "_capture_authority_snapshot",
                classmethod(original_capture),
            )


def test_r8_i1_gate_current_check_rejects_approval_drift(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows.stage1_gate import (
        Stage1GateBindingVerifier,
        Stage1GateError,
        record_stage1_human_approval,
    )

    stage, context, review = _reviewed_context_from_template(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    review_time = datetime.fromisoformat(
        str(review["review_recorded_at_utc"]).replace("Z", "+00:00")
    )
    approval_path = record_stage1_human_approval(
        context=context,
        thread_id=CANONICAL_THREAD_ID,
        user_turn_id=CANONICAL_USER_TURN_ID,
        approval_text="\u6279\u51c6 Stage 1 Gate",
        user_turn_timestamp_utc=(review_time + timedelta(seconds=1))
        .isoformat()
        .replace("+00:00", "Z"),
    )
    approval_b = approval_path.read_bytes()
    original_capture = Stage1GateBindingVerifier._capture_authority_snapshot.__func__
    capture_calls = 0

    def capture_b_then_drift_approval(cls, received_context, target_state):
        nonlocal capture_calls
        capture_calls += 1
        snapshot = original_capture(cls, received_context, target_state)
        approval_path.write_bytes(approval_b + b"\n")
        return snapshot

    monkeypatch.setattr(
        Stage1GateBindingVerifier,
        "_capture_authority_snapshot",
        classmethod(capture_b_then_drift_approval),
    )
    try:
        with pytest.raises(Stage1GateError, match="snapshot|current|drift|authority"):
            Stage1GateBindingVerifier.compute(context, target_state="approved")
        assert capture_calls == 1
    finally:
        approval_path.write_bytes(approval_b)


def test_r8_i1_gate_target_state_byte_graphs_are_exact_and_gate_is_current_only(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows.stage1_artifacts import (
        STAGE1_MACHINE_ARTIFACTS,
        Stage1WorkflowError,
    )
    from lunar_exploration_ppo.workflows.stage1_gate import (
        STAGE1_APPROVAL_ABSENT_HASH,
        Stage1GateBindingVerifier,
        record_stage1_human_approval,
    )

    stage, context, review = _reviewed_context_from_template(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    base = set(STAGE1_MACHINE_ARTIFACTS) | {"manifest.json", "review.json"}
    pre_approval_states = (
        "machine_passed",
        "awaiting_independent_review",
        "awaiting_human_approval",
    )

    for target_state in pre_approval_states:
        snapshot = Stage1GateBindingVerifier._capture_authority_snapshot(
            context,
            target_state,
        )
        assert {name for name, _kind in snapshot.stage.members} == base
    for target_state in ("approved", "next_stage"):
        with pytest.raises(Stage1WorkflowError, match="exact regular-file set"):
            Stage1GateBindingVerifier._capture_authority_snapshot(context, target_state)

    waiting = Stage1GateBindingVerifier.compute(
        context,
        target_state="awaiting_human_approval",
    )
    assert waiting.approval_state == "absent"
    assert waiting.approval_hash == STAGE1_APPROVAL_ABSENT_HASH

    review_time = datetime.fromisoformat(
        str(review["review_recorded_at_utc"]).replace("Z", "+00:00")
    )
    record_stage1_human_approval(
        context=context,
        thread_id=CANONICAL_THREAD_ID,
        user_turn_id=CANONICAL_USER_TURN_ID,
        approval_text="\u6279\u51c6 Stage 1 Gate",
        user_turn_timestamp_utc=(review_time + timedelta(seconds=1))
        .isoformat()
        .replace("+00:00", "Z"),
    )
    approved_members = base | {"approval.json"}
    for target_state in pre_approval_states:
        with pytest.raises(Stage1WorkflowError, match="exact regular-file set"):
            Stage1GateBindingVerifier._capture_authority_snapshot(context, target_state)
    for target_state in ("approved", "next_stage"):
        snapshot = Stage1GateBindingVerifier._capture_authority_snapshot(
            context,
            target_state,
        )
        assert {name for name, _kind in snapshot.stage.members} == approved_members

    approved = Stage1GateBindingVerifier.compute(context, target_state="approved")
    next_without_gate = Stage1GateBindingVerifier.compute(context, target_state="next_stage")
    assert approved.approval_state == next_without_gate.approval_state == "approved"
    assert approved == next_without_gate

    gate_path = stage / "gate.json"
    gate_a = b'{"synthetic_test_gate":"a"}\n'
    gate_b = b'{"synthetic_test_gate":"b"}\n'
    gate_path.write_bytes(gate_a)
    try:
        with pytest.raises(Stage1WorkflowError, match="exact regular-file set"):
            Stage1GateBindingVerifier._capture_authority_snapshot(context, "approved")
        gate_snapshot = Stage1GateBindingVerifier._capture_authority_snapshot(
            context,
            "next_stage",
        )
        assert {name for name, _kind in gate_snapshot.stage.members} == approved_members | {
            "gate.json"
        }
        assert Stage1GateBindingVerifier._compute_snapshot_bindings(
            context,
            gate_snapshot,
        ) == next_without_gate

        gate_path.write_bytes(gate_b)
        assert Stage1GateBindingVerifier._compute_snapshot_bindings(
            context,
            gate_snapshot,
        ) == next_without_gate
        with pytest.raises(Stage1WorkflowError, match="snapshot|drift|changed"):
            Stage1GateBindingVerifier._require_authority_snapshot_current(gate_snapshot)
    finally:
        if gate_path.is_file():
            gate_path.unlink()


def test_toctou_mutation_machine_passed_compute_before_proof_returns_no_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows.stage1_gate import (
        Stage1GateBindingVerifier,
        Stage1GateError,
        Stage1GateRecord,
    )

    _, context, _, review_package, _ = _reviewed_gate_context(tmp_path, monkeypatch)
    original_package = review_package.read_bytes()
    original_compute = Stage1GateBindingVerifier.compute.__func__
    compute_count = 0

    def mutate_after_first_compute(cls, received_context, *, target_state="awaiting_human_approval"):
        nonlocal compute_count
        compute_count += 1
        bindings = original_compute(cls, received_context, target_state=target_state)
        if compute_count == 1:
            review_package.write_bytes(original_package + b"\n")
        return bindings

    monkeypatch.setattr(
        Stage1GateBindingVerifier,
        "compute",
        classmethod(mutate_after_first_compute),
    )
    try:
        with pytest.raises(Stage1GateError, match="review package|snapshot|drift|binding"):
            Stage1GateRecord.machine_passed(context)
        assert compute_count >= 2
    finally:
        review_package.write_bytes(original_package)


def test_toctou_mutation_transition_compute_before_proof_returns_no_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows.stage1_gate import (
        Stage1GateBindingVerifier,
        Stage1GateError,
        Stage1GateRecord,
    )

    _, context, _, review_package, _ = _reviewed_gate_context(tmp_path, monkeypatch)
    record = Stage1GateRecord.machine_passed(context)
    original_package = review_package.read_bytes()
    original_compute = Stage1GateBindingVerifier.compute.__func__
    compute_count = 0

    def mutate_after_transition_compute(cls, received_context, *, target_state="awaiting_human_approval"):
        nonlocal compute_count
        compute_count += 1
        bindings = original_compute(cls, received_context, target_state=target_state)
        if compute_count == 1:
            review_package.write_bytes(original_package + b"\n")
        return bindings

    monkeypatch.setattr(
        Stage1GateBindingVerifier,
        "compute",
        classmethod(mutate_after_transition_compute),
    )
    try:
        with pytest.raises(Stage1GateError, match="review package|snapshot|drift|binding"):
            record.transition("awaiting_independent_review")
        assert compute_count >= 2
    finally:
        review_package.write_bytes(original_package)


def test_toctou_mutation_final_gate_serialization_revalidates_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows.stage1_gate import Stage1GateError, Stage1GateRecord

    _, context, review_report, _, _ = _reviewed_gate_context(tmp_path, monkeypatch)
    record = Stage1GateRecord.machine_passed(context)
    original_report = review_report.read_bytes()
    review_report.write_bytes(original_report + b"concurrent serialization mutation\n")
    try:
        with pytest.raises(Stage1GateError, match="review report|snapshot|drift|binding"):
            record.to_dict()
    finally:
        review_report.write_bytes(original_report)


def test_c1_gate_independently_replays_review_package_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_gate import Stage1GateBindingVerifier, Stage1GateError
    from lunar_exploration_ppo.workflows.stage1_review import build_stage1_approval_challenge
    from lunar_exploration_ppo.workflows.stage1_source import STAGE1_REVIEWED_PATHS

    stage, context, _, _, _ = _reviewed_gate_context(tmp_path, monkeypatch)
    incomplete_package = tmp_path / "gate-incomplete-package.diff"
    _write_git_native_review_package(
        incomplete_package,
        reviewed_paths=set(STAGE1_REVIEWED_PATHS) - {"src/lunar_exploration_ppo/workflows/stage1_review.py"},
    )
    payload = incomplete_package.read_bytes()
    review_path = stage / "review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["review_package"] = {
        "path": str(incomplete_package.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "lf_count": payload.count(b"\n"),
        "logical_line_count": len(payload.splitlines()),
    }
    review["approval_challenge"] = build_stage1_approval_challenge(
        goal_id=review["goal_id"],
        stage_id=review["stage_id"],
        run_id=review["run_id"],
        reviewed_git_tree=review["reviewed_git_tree"],
        review_report_sha256=review["review_report"]["sha256"],
        review_package_sha256=review["review_package"]["sha256"],
        manifest_hash=review["manifest_hash"],
    )
    review_path.write_bytes(ArtifactStore.canonical_json_bytes(review))

    with pytest.raises(Stage1GateError, match="package|path|tree|patch|Git"):
        Stage1GateBindingVerifier.compute(context)


def test_gate_binds_current_tree_and_rehashes_review_report_and_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lunar_exploration_ppo.workflows.stage1_gate import Stage1GateBindingVerifier, Stage1GateError

    _, context, review_report, review_package, reviewed_tree = _reviewed_gate_context(tmp_path, monkeypatch)
    bindings = Stage1GateBindingVerifier.compute(context)
    assert bindings.git_tree_hash == reviewed_tree
    assert bindings.authorized_next_stage == "ppo_highres_frontier_stage2_observation_frontier/v1"

    original_report = review_report.read_bytes()
    with review_report.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(Stage1GateError, match="review report"):
        Stage1GateBindingVerifier.compute(context)
    review_report.write_bytes(original_report)

    review_package.write_bytes(b"drifted-package\n")
    with pytest.raises(Stage1GateError, match="review package"):
        Stage1GateBindingVerifier.compute(context)


def test_gate_rejects_current_clean_tree_different_from_reviewed_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lunar_exploration_ppo.workflows.stage1_gate as gate_module
    from lunar_exploration_ppo.workflows.stage1_gate import Stage1GateBindingVerifier, Stage1GateError

    _, context, _, _, reviewed_tree = _reviewed_gate_context(tmp_path, monkeypatch)
    different_tree = ("0" if reviewed_tree[0] != "0" else "1") + reviewed_tree[1:]
    original_identity = gate_module.compute_stage1_source_identity_from_snapshot

    def mismatched_tree(snapshot):
        return original_identity(snapshot) | {"prospective_git_tree": different_tree}

    monkeypatch.setattr(
        gate_module,
        "compute_stage1_source_identity_from_snapshot",
        mismatched_tree,
    )
    with pytest.raises(Stage1GateError, match="source|reviewed|tree"):
        Stage1GateBindingVerifier.compute(context)


def test_stage1_gate_record_has_private_proven_state_machine_and_verified_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_gate import (
        Stage1GateError,
        Stage1GateRecord,
        record_stage1_human_approval,
    )

    stage, context, _, _, reviewed_tree = _reviewed_gate_context(tmp_path, monkeypatch)
    with pytest.raises(TypeError):
        Stage1GateRecord()
    record = Stage1GateRecord.machine_passed(context)
    with pytest.raises(Stage1GateError, match="transition"):
        record.transition("awaiting_human_approval")
    record = record.transition("awaiting_independent_review")
    record = record.transition("awaiting_human_approval")
    with pytest.raises(Stage1GateError, match="approval"):
        record.transition("approved")

    store = ArtifactStore(stage)
    approval = {
        "schema_version": "ppo_highres_frontier_stage1_human_approval/v1",
        "run_id": context.run_id,
        "state": "approved",
        "reviewed_git_tree": reviewed_tree,
    }
    invalid_approvals = (
        {key: value for key, value in approval.items() if key != "state"},
        approval | {"run_id": "different-run"},
        approval | {"state": "awaiting_human_approval"},
        approval | {"reviewed_git_tree": "0" * len(reviewed_tree)},
        approval | {"approval_source": "test-must-not-extend-exact-schema"},
    )
    for invalid in invalid_approvals:
        store.write_json("approval.json", invalid)
        with pytest.raises(Stage1GateError, match="approval"):
            record.transition("approved")

    (stage / "approval.json").unlink()
    review = json.loads((stage / "review.json").read_text(encoding="utf-8"))
    review_time = datetime.fromisoformat(
        str(review["review_recorded_at_utc"]).replace("Z", "+00:00")
    )
    approval_path = record_stage1_human_approval(
        context=context,
        stage_root=stage,
        thread_id=CANONICAL_THREAD_ID,
        user_turn_id=CANONICAL_REPLAY_USER_TURN_ID,
        approval_text="\u6279\u51c6 Stage 1 Gate",
        user_turn_timestamp_utc=(review_time + timedelta(seconds=1))
        .isoformat()
        .replace("+00:00", "Z"),
    )
    record = record.transition("approved")
    expected_approval_hash = hashlib.sha256(approval_path.read_bytes()).hexdigest()
    assert record.bindings.approval_state == "approved"
    assert record.bindings.approval_hash == expected_approval_hash

    approved_bytes = approval_path.read_bytes()
    approval_path.write_bytes(approved_bytes + b"\n")
    with pytest.raises(Stage1GateError, match="approval|binding drift"):
        record.transition("next_stage")
    approval_path.write_bytes(approved_bytes)

    record = record.transition("next_stage")
    assert record.state == "next_stage"
    assert record.history == (
        "machine_passed",
        "awaiting_independent_review",
        "awaiting_human_approval",
        "approved",
        "next_stage",
    )
    with pytest.raises(AttributeError):
        record.state = "approved"

    payload = record.to_dict()
    gate_path = record.write_verified()
    with pytest.raises(Stage1GateError, match="path|gate.json"):
        Stage1GateRecord.load_verified(payload)
    loaded = Stage1GateRecord.load_verified(gate_path)
    assert loaded.to_dict() == payload

    (stage / "unrelated-extra.txt").write_text("not allowed\n", encoding="utf-8")
    with pytest.raises(Stage1GateError, match="unexpected|artifact set"):
        Stage1GateRecord.load_verified(gate_path)


def _next_stage_gate_record(
    *,
    template: Path,
    destination: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from lunar_exploration_ppo.workflows.stage1_gate import (
        Stage1GateRecord,
        record_stage1_human_approval,
    )

    stage, context, review = _reviewed_context_from_template(
        template=template,
        destination=destination,
        monkeypatch=monkeypatch,
    )
    record = Stage1GateRecord.machine_passed(context)
    record = record.transition("awaiting_independent_review")
    record = record.transition("awaiting_human_approval")
    review_time = datetime.fromisoformat(
        str(review["review_recorded_at_utc"]).replace("Z", "+00:00")
    )
    record_stage1_human_approval(
        context=context,
        stage_root=stage,
        thread_id=CANONICAL_THREAD_ID,
        user_turn_id=CANONICAL_REPLAY_USER_TURN_ID,
        approval_text="\u6279\u51c6 Stage 1 Gate",
        user_turn_timestamp_utc=(review_time + timedelta(seconds=1))
        .isoformat()
        .replace("+00:00", "Z"),
    )
    record = record.transition("approved")
    return stage, context, record.transition("next_stage")


def test_r7_i3_verified_gate_writer_rejects_unproven_or_non_next_stage_before_write(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_gate import (
        Stage1GateError,
        Stage1GateRecord,
    )

    stage, context, _ = _reviewed_context_from_template(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    machine_record = Stage1GateRecord.machine_passed(context)
    writes: list[str] = []

    def reject_any_write(store, relative_path, value):
        writes.append(Path(relative_path).as_posix())
        raise AssertionError("verified gate writer reached ArtifactStore unexpectedly")

    monkeypatch.setattr(ArtifactStore, "write_json_exclusive", reject_any_write)
    for candidate in (object(), machine_record):
        with pytest.raises(Stage1GateError, match="record|next_stage|proven"):
            Stage1GateRecord.write_verified(candidate)
    assert writes == []
    assert not (stage / "gate.json").exists()


def test_r7_i3_verified_gate_writer_rejects_preexisting_gate_before_write(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_gate import Stage1GateError

    stage, _, record = _next_stage_gate_record(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    gate_path = stage / "gate.json"
    foreign_payload = b'{"actor":"foreign-preexisting"}\n'
    gate_path.write_bytes(foreign_payload)
    writes: list[str] = []

    def reject_any_write(store, relative_path, value):
        writes.append(Path(relative_path).as_posix())
        raise AssertionError("preexisting gate reached ArtifactStore unexpectedly")

    monkeypatch.setattr(ArtifactStore, "write_json_exclusive", reject_any_write)
    with pytest.raises(Stage1GateError, match="exists|preexisting|pre-existing"):
        record.write_verified()
    assert writes == []
    assert gate_path.read_bytes() == foreign_payload


def test_r7_i3_verified_gate_writer_revalidates_after_payload_freeze_before_write(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_gate import Stage1GateError, Stage1GateRecord

    stage, _, record = _next_stage_gate_record(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    gate_path = stage / "gate.json"
    review_report = tmp_path / "review.md"
    original_report = review_report.read_bytes()
    original_to_dict = Stage1GateRecord.to_dict
    original_write_json = ArtifactStore.write_json_exclusive
    to_dict_calls = 0
    write_calls = 0

    def drift_after_freeze(received_record):
        nonlocal to_dict_calls
        to_dict_calls += 1
        payload = original_to_dict(received_record)
        if to_dict_calls == 1:
            review_report.write_bytes(original_report + b"authority drift before gate write\n")
        return payload

    def count_write(store, relative_path, value):
        nonlocal write_calls
        write_calls += 1
        return original_write_json(store, relative_path, value)

    monkeypatch.setattr(Stage1GateRecord, "to_dict", drift_after_freeze)
    monkeypatch.setattr(ArtifactStore, "write_json_exclusive", count_write)
    try:
        with pytest.raises(Stage1GateError, match="review|authority|drift|binding"):
            record.write_verified()
        assert to_dict_calls >= 2
        assert write_calls == 0
        assert not gate_path.exists()
    finally:
        review_report.write_bytes(original_report)


def test_r9_i1_gate_post_publish_failure_retains_owned_payload_and_retry_fails_closed(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_gate import Stage1GateError

    class SentinelWriteError(RuntimeError):
        pass

    stage, _, record = _next_stage_gate_record(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    gate_path = stage / "gate.json"
    original_write_json = ArtifactStore.write_json_exclusive
    published_payloads: list[bytes] = []

    def write_then_raise(store, relative_path, value):
        written = original_write_json(store, relative_path, value)
        if Path(relative_path).as_posix() == "gate.json":
            published_payloads.append(ArtifactStore.canonical_json_bytes(value))
            assert written.read_bytes() == published_payloads[-1]
            raise SentinelWriteError("sentinel gate write failure after replace")
        return written

    monkeypatch.setattr(ArtifactStore, "write_json_exclusive", write_then_raise)
    with pytest.raises(SentinelWriteError, match="after replace"):
        record.write_verified()
    assert len(published_payloads) == 1
    assert gate_path.read_bytes() == published_payloads[0]

    with pytest.raises(Stage1GateError, match="already exists"):
        record.write_verified()
    assert len(published_payloads) == 1
    assert gate_path.read_bytes() == published_payloads[0]


def test_r7_i3_verified_gate_writer_rejects_unexpected_writer_path(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_gate import Stage1GateError

    stage, _, record = _next_stage_gate_record(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    gate_path = stage / "gate.json"
    original_write_json = ArtifactStore.write_json_exclusive
    expected_bytes = ArtifactStore.canonical_json_bytes(record.to_dict())

    def write_then_return_wrong_path(store, relative_path, value):
        original_write_json(store, relative_path, value)
        return stage / "not-gate.json"

    monkeypatch.setattr(ArtifactStore, "write_json_exclusive", write_then_return_wrong_path)
    with pytest.raises(Stage1GateError, match="path|target"):
        record.write_verified()
    assert gate_path.read_bytes() == expected_bytes


def test_r9_i1_verified_gate_writer_retains_gate_after_load_verified_failure(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows.stage1_gate import Stage1GateRecord

    class SentinelReplayError(RuntimeError):
        pass

    stage, _, record = _next_stage_gate_record(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    gate_path = stage / "gate.json"
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore

    expected_bytes = ArtifactStore.canonical_json_bytes(record.to_dict())

    def reject_replay(cls, received_gate_path):
        assert Path(received_gate_path).resolve() == gate_path.resolve()
        raise SentinelReplayError("sentinel verified replay failure")

    monkeypatch.setattr(Stage1GateRecord, "load_verified", classmethod(reject_replay))
    with pytest.raises(SentinelReplayError, match="verified replay"):
        record.write_verified()
    assert gate_path.read_bytes() == expected_bytes


@pytest.mark.parametrize("mismatch", ("state", "history", "bindings", "context", "payload"))
def test_r7_i3_verified_gate_writer_rejects_inexact_replay(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    from lunar_exploration_ppo.workflows.stage1_gate import Stage1GateError, Stage1GateRecord

    stage, context, record = _next_stage_gate_record(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    gate_path = stage / "gate.json"
    expected_payload = record.to_dict()
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore

    expected_bytes = ArtifactStore.canonical_json_bytes(expected_payload)

    class ReplayResult:
        def __init__(self):
            self.state = "approved" if mismatch == "state" else record.state
            self.history = record.history[:-1] if mismatch == "history" else record.history
            self.bindings = (
                record.bindings.model_copy(update={"approval_hash": "0" * 64})
                if mismatch == "bindings"
                else record.bindings
            )
            self.context = (
                context.model_copy(update={"run_id": "foreign-run"})
                if mismatch == "context"
                else context
            )

        def to_dict(self):
            if mismatch == "payload":
                return expected_payload | {"schema_version": "foreign-gate/v1"}
            return expected_payload

    replay = ReplayResult()

    def return_inexact_replay(cls, received_gate_path):
        assert Path(received_gate_path).resolve() == gate_path.resolve()
        return replay

    monkeypatch.setattr(
        Stage1GateRecord,
        "load_verified",
        classmethod(return_inexact_replay),
    )
    with pytest.raises(Stage1GateError, match="replay|exact|mismatch|verified"):
        record.write_verified()
    assert gate_path.read_bytes() == expected_bytes


def test_r9_i1_verified_gate_writer_retains_gate_after_post_revalidation_drift(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows.stage1_gate import Stage1GateError, Stage1GateRecord

    stage, _, record = _next_stage_gate_record(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    gate_path = stage / "gate.json"
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore

    expected_bytes = ArtifactStore.canonical_json_bytes(record.to_dict())
    review_report = tmp_path / "review.md"
    original_report = review_report.read_bytes()
    original_load_verified = Stage1GateRecord.load_verified.__func__
    original_to_dict = Stage1GateRecord.to_dict
    replay_holder: dict[str, object] = {}
    mutated = False

    def load_and_hold(cls, received_gate_path):
        replayed = original_load_verified(cls, received_gate_path)
        replay_holder["record"] = replayed
        return replayed

    def mutate_after_replay_serialization(received_record):
        nonlocal mutated
        payload = original_to_dict(received_record)
        if received_record is replay_holder.get("record") and not mutated:
            review_report.write_bytes(original_report + b"post-replay authority drift\n")
            mutated = True
        return payload

    monkeypatch.setattr(Stage1GateRecord, "load_verified", classmethod(load_and_hold))
    monkeypatch.setattr(Stage1GateRecord, "to_dict", mutate_after_replay_serialization)
    try:
        with pytest.raises(Stage1GateError, match="review|authority|drift|binding"):
            record.write_verified()
        assert mutated
        assert gate_path.read_bytes() == expected_bytes
    finally:
        review_report.write_bytes(original_report)


def test_r7_i3_verified_gate_writer_detects_disk_mismatch_and_preserves_foreign_payload(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_gate import Stage1GateError

    stage, _, record = _next_stage_gate_record(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    gate_path = stage / "gate.json"
    foreign_payload = b'{"actor":"foreign-after-write"}\n'
    original_write_json = ArtifactStore.write_json_exclusive

    def replace_with_foreign_after_write(store, relative_path, value):
        written = original_write_json(store, relative_path, value)
        written.write_bytes(foreign_payload)
        return written

    monkeypatch.setattr(ArtifactStore, "write_json_exclusive", replace_with_foreign_after_write)
    with pytest.raises(Stage1GateError, match="bytes|payload|changed"):
        record.write_verified()
    assert gate_path.read_bytes() == foreign_payload


@pytest.mark.parametrize("forbidden_operation", ("read", "unlink"))
def test_r9_i1_verified_gate_writer_propagates_original_without_touching_canonical(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
    forbidden_operation: str,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore

    class SentinelWriteError(RuntimeError):
        pass

    stage, _, record = _next_stage_gate_record(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    gate_path = (stage / "gate.json").resolve()
    original_write_json = ArtifactStore.write_json_exclusive
    original_read_bytes = Path.read_bytes
    original_unlink = Path.unlink
    post_publish_failure = False
    forbidden_calls = 0
    expected_bytes = ArtifactStore.canonical_json_bytes(record.to_dict())

    def write_then_raise(store, relative_path, value):
        nonlocal post_publish_failure
        written = original_write_json(store, relative_path, value)
        post_publish_failure = True
        raise SentinelWriteError("sentinel original gate write failure")

    def maybe_fail_read(path):
        nonlocal forbidden_calls
        if post_publish_failure and forbidden_operation == "read" and path.resolve() == gate_path:
            forbidden_calls += 1
            raise PermissionError("sentinel gate rollback read failure")
        return original_read_bytes(path)

    def maybe_fail_unlink(path, *args, **kwargs):
        nonlocal forbidden_calls
        if post_publish_failure and forbidden_operation == "unlink" and path.resolve() == gate_path:
            forbidden_calls += 1
            raise PermissionError("sentinel gate rollback unlink failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(ArtifactStore, "write_json_exclusive", write_then_raise)
    monkeypatch.setattr(Path, "read_bytes", maybe_fail_read)
    monkeypatch.setattr(Path, "unlink", maybe_fail_unlink)
    try:
        with pytest.raises(SentinelWriteError, match="original gate write failure"):
            record.write_verified()
        assert forbidden_calls == 0
        assert original_read_bytes(gate_path) == expected_bytes
    finally:
        if gate_path.is_file():
            original_unlink(gate_path)


def test_r7_i3_verified_gate_writer_writes_and_immediately_replays_exact_gate(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_gate import Stage1GateRecord

    stage, _, record = _next_stage_gate_record(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    expected_payload = record.to_dict()
    gate_path = record.write_verified()
    expected_bytes = ArtifactStore.canonical_json_bytes(expected_payload)
    assert gate_path.resolve() == (stage / "gate.json").resolve()
    assert gate_path.read_bytes() == expected_bytes
    replayed = Stage1GateRecord.load_verified(gate_path)
    assert replayed.state == record.state
    assert replayed.history == record.history
    assert replayed.bindings == record.bindings
    assert replayed.context == record.context
    assert replayed.to_dict() == expected_payload


def test_stage1_verified_gate_exclusive_publish_race_preserves_foreign_gate(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore

    stage, _, record = _next_stage_gate_record(
        template=r4_machine_stage,
        destination=tmp_path,
        monkeypatch=monkeypatch,
    )
    gate_path = stage / "gate.json"
    foreign = b'{"actor":"foreign-publish-racer"}\n'
    expected = ArtifactStore.canonical_json_bytes(record.to_dict())
    real_link = os.link
    gate_publish_calls = 0

    def competing_link(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        nonlocal gate_publish_calls
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path.name == "gate.json":
            gate_publish_calls += 1
            assert source_path.read_bytes() == expected
            assert not destination_path.exists()
            destination_path.write_bytes(foreign)
        real_link(source, destination)

    monkeypatch.setattr(os, "link", competing_link)

    with pytest.raises(FileExistsError):
        record.write_verified()

    assert gate_publish_calls == 1
    assert gate_path.read_bytes() == foreign
    assert not [path for path in stage.iterdir() if path.name.startswith(".gate.json.")]


@pytest.mark.parametrize("authority_kind", ("review", "approval", "gate"))
def test_r9_i1_exact_compare_then_unlink_race_never_deletes_foreign_authority(
    tmp_path: Path,
    r4_machine_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority_kind: str,
) -> None:
    import lunar_exploration_ppo.workflows.stage1_gate as gate_module
    import lunar_exploration_ppo.workflows.stage1_review as review_module
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.stage1_gate import record_stage1_human_approval

    class SentinelPostPublishError(RuntimeError):
        pass

    if authority_kind == "review":
        stage, report, package = _review_recording_inputs(
            template=r4_machine_stage,
            destination=tmp_path / authority_kind,
        )
        authority_path = (stage / "review.json").resolve()
        invoke = lambda: _record_review(stage, report, package)
        legacy_helper_present = hasattr(review_module, "_rollback_owned_review")
    elif authority_kind == "approval":
        stage, context, review = _reviewed_context_from_template(
            template=r4_machine_stage,
            destination=tmp_path / authority_kind,
            monkeypatch=monkeypatch,
        )
        authority_path = (stage / "approval.json").resolve()
        review_time = datetime.fromisoformat(
            str(review["review_recorded_at_utc"]).replace("Z", "+00:00")
        )
        invoke = lambda: record_stage1_human_approval(
            context=context,
            thread_id=CANONICAL_THREAD_ID,
            user_turn_id=CANONICAL_USER_TURN_ID,
            approval_text="\u6279\u51c6 Stage 1 Gate",
            user_turn_timestamp_utc=(review_time + timedelta(seconds=1))
            .isoformat()
            .replace("+00:00", "Z"),
        )
        legacy_helper_present = hasattr(gate_module, "_rollback_owned_approval")
    else:
        stage, _, record = _next_stage_gate_record(
            template=r4_machine_stage,
            destination=tmp_path / authority_kind,
            monkeypatch=monkeypatch,
        )
        authority_path = (stage / "gate.json").resolve()
        invoke = record.write_verified
        legacy_helper_present = hasattr(gate_module, "_rollback_owned_gate")

    target_name = authority_path.name
    original_write_json = ArtifactStore.write_json_exclusive
    original_unlink = Path.unlink
    owned_payloads: list[bytes] = []
    trace: dict[str, object] = {
        "canonical_unlink_calls": 0,
        "foreign_installed": False,
        "foreign_deleted": False,
    }

    def write_then_raise(store, relative_path, value):
        written = original_write_json(store, relative_path, value)
        if Path(relative_path).as_posix() == target_name:
            owned_payload = ArtifactStore.canonical_json_bytes(value)
            assert written.read_bytes() == owned_payload
            owned_payloads.append(owned_payload)
            raise SentinelPostPublishError(
                f"sentinel {authority_kind} failure after exclusive publish"
            )
        return written

    foreign_payload = ArtifactStore.canonical_json_bytes(
        {
            "actor": "foreign-r9-i1-exact-racer",
            "authority_kind": authority_kind,
        }
    )
    foreign_source = tmp_path / f"{authority_kind}-foreign-replacement.json"
    foreign_source.write_bytes(foreign_payload)
    foreign_inode = foreign_source.stat().st_ino

    def replace_foreign_at_old_unlink(path: Path, *args, **kwargs):
        if path.resolve() != authority_path:
            return original_unlink(path, *args, **kwargs)
        trace["canonical_unlink_calls"] = int(trace["canonical_unlink_calls"]) + 1
        assert owned_payloads
        assert path.read_bytes() == owned_payloads[0]
        os.replace(foreign_source, path)
        trace["foreign_installed"] = True
        trace["foreign_inode_before_unlink"] = path.stat().st_ino
        trace["foreign_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert trace["foreign_inode_before_unlink"] == foreign_inode
        assert path.read_bytes() == foreign_payload
        result = original_unlink(path, *args, **kwargs)
        trace["foreign_deleted"] = not os.path.lexists(path)
        return result

    monkeypatch.setattr(ArtifactStore, "write_json_exclusive", write_then_raise)
    monkeypatch.setattr(Path, "unlink", replace_foreign_at_old_unlink)
    try:
        with pytest.raises(
            SentinelPostPublishError,
            match=f"sentinel {authority_kind} failure after exclusive publish",
        ):
            invoke()

        assert bool(trace["foreign_installed"]) is legacy_helper_present
        assert not trace["foreign_deleted"], (
            "R9-I1 reproduced: foreign payload was installed by os.replace only after the "
            "owned-byte compare reached canonical unlink, then that foreign object was deleted; "
            f"authority={authority_kind}, foreign_inode={trace.get('foreign_inode_before_unlink')}, "
            f"foreign_sha256={trace.get('foreign_sha256')}, "
            f"canonical_exists_after={os.path.lexists(authority_path)}"
        )
        if trace["foreign_installed"]:
            assert authority_path.stat().st_ino == foreign_inode
            assert authority_path.read_bytes() == foreign_payload
        else:
            assert trace["canonical_unlink_calls"] == 0
            assert authority_path.read_bytes() == owned_payloads[0]
    finally:
        if authority_path.is_file():
            original_unlink(authority_path)
        if foreign_source.is_file():
            original_unlink(foreign_source)


def test_policy_callback_receives_only_observed_policy_observation(tmp_path: Path) -> None:
    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import LunarExplorationEnv
    from lunar_exploration_ppo.policy.observation import PolicyObservation
    from lunar_exploration_ppo.workflows.stage1 import run_episode, select_rule_action

    env = LunarExplorationEnv(load_stage1_config(STAGE1_CONFIG))
    received: list[PolicyObservation] = []

    def observed_only_policy(observation: PolicyObservation):
        received.append(observation)
        assert not hasattr(observation, "truth")
        assert not hasattr(observation, "coverable_mask")
        return select_rule_action(observation)

    episode = run_episode(env, episode_index=0, policy=observed_only_policy)
    assert episode.transition_count == len(received) > 0


def test_hidden_truth_official_masks_and_policy_arrays_are_readonly_and_env_isolated() -> None:
    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import LunarExplorationEnv

    config = load_stage1_config(STAGE1_CONFIG)
    first = LunarExplorationEnv(config)
    second = LunarExplorationEnv(config)
    observation = first.reset()

    assert not hasattr(first, "scenario")
    assert not hasattr(first, "coverage_masks")
    truth_layers = (
        first._scenario.truth.height,
        first._scenario.truth.hard_obstacle,
        first._scenario.truth.slope_deg,
        first._scenario.truth.traversability,
    )
    official_masks = (
        first._coverage_masks.safe_free_mask,
        first._coverage_masks.reachable_safe_mask,
        first._coverage_masks.coverable_mask,
    )
    assert all(not array.flags.writeable for array in (*truth_layers, *official_masks))
    assert all(not array.flags.writeable for array in observation.array_fields())
    assert not first.current_action_set.frontier_features.flags.writeable
    assert not first.current_action_set.candidate_mask.flags.writeable
    assert not any(
        __import__("numpy").shares_memory(left, right)
        for left, right in zip(
            official_masks,
            (
                second._coverage_masks.safe_free_mask,
                second._coverage_masks.reachable_safe_mask,
                second._coverage_masks.coverable_mask,
            ),
            strict=True,
        )
    )
    with pytest.raises(ValueError):
        first._coverage_masks.coverable_mask[0, 0] = True
    with pytest.raises(ValueError):
        observation.candidate_mask[0] = False


def _open_truth(width: int = 128, height: int = 128):
    from lunar_exploration_ppo.env.scenario import TruthMap
    from lunar_exploration_ppo.utils.geometry import GridGeometry

    geometry = GridGeometry(width, height, 0.5)
    return TruthMap(
        geometry=geometry,
        height=np.zeros(geometry.shape, dtype=float),
        hard_obstacle=np.zeros(geometry.shape, dtype=bool),
        slope_deg=np.zeros(geometry.shape, dtype=float),
        traversability=np.ones(geometry.shape, dtype=float),
        provenance={
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "physical_obstacle_cells_written": False,
        },
    )


@pytest.mark.parametrize("heading_deg", [0.0, 8.0, 38.0, 45.0])
def test_sensor_never_reveals_cell_center_beyond_exact_20m(heading_deg: float) -> None:
    from lunar_exploration_ppo.env.map_state import ObservedMapState
    from lunar_exploration_ppo.env.sensor_model import SensorPose, SensorUpdater
    from lunar_exploration_ppo.utils.geometry import CellXY

    truth = _open_truth()
    state = ObservedMapState.empty(truth.geometry)
    origin = truth.geometry.cell_to_world_center(CellXY(48, 48))
    delta = SensorUpdater(range_m=20.0, fov_deg=0.0).reveal(
        truth,
        state,
        (SensorPose(origin, math.radians(heading_deg), "endpoint_theta"),),
    )

    distances = [
        math.hypot(
            truth.geometry.cell_to_world_center(cell).x - origin.x,
            truth.geometry.cell_to_world_center(cell).y - origin.y,
        )
        for cell in delta.visible_cells
    ]
    assert max(distances) <= 20.0
    if heading_deg == 0.0:
        assert CellXY(88, 48) in delta.visible_cells
        assert CellXY(89, 48) not in delta.visible_cells


def test_sensor_range_uses_original_subcell_pose_and_stops_cleanly_at_map_edge() -> None:
    from lunar_exploration_ppo.env.map_state import ObservedMapState
    from lunar_exploration_ppo.env.sensor_model import SensorPose, SensorUpdater
    from lunar_exploration_ppo.utils.geometry import CellXY, WorldXY

    truth = _open_truth()
    updater = SensorUpdater(range_m=20.0, fov_deg=0.0)
    subcell_origin = WorldXY(20.01, 32.25)
    subcell = updater.reveal(
        truth,
        ObservedMapState.empty(truth.geometry),
        (SensorPose(subcell_origin, 0.0, "path_tangent"),),
    )
    assert CellXY(80, 64) not in subcell.visible_cells
    assert all(
        math.hypot(
            truth.geometry.cell_to_world_center(cell).x - subcell_origin.x,
            truth.geometry.cell_to_world_center(cell).y - subcell_origin.y,
        )
        <= 20.0
        for cell in subcell.visible_cells
    )

    edge_origin = truth.geometry.cell_to_world_center(CellXY(126, 126))
    edge = updater.reveal(
        truth,
        ObservedMapState.empty(truth.geometry),
        (SensorPose(edge_origin, math.pi / 4.0, "endpoint_theta"),),
    )
    assert edge.visible_cells == (CellXY(126, 126), CellXY(127, 127))


def _independent_coverable_oracle(
    truth,
    start,
    *,
    sensor_range_m: float,
    min_clearance_m: float,
    max_slope_deg: float = 30.0,
    traversability_threshold: float = 0.5,
):
    height, width = truth.geometry.shape
    resolution = truth.geometry.resolution_m
    free = (
        np.isfinite(truth.height)
        & np.isfinite(truth.slope_deg)
        & np.isfinite(truth.traversability)
        & ~truth.hard_obstacle
        & (truth.slope_deg <= max_slope_deg)
        & (truth.traversability >= traversability_threshold)
    )
    unsafe_y, unsafe_x = np.nonzero(~free)
    safe = np.zeros_like(free)
    for y in range(height):
        for x in range(width):
            if not free[y, x]:
                continue
            boundary_clearance = min(x + 0.5, width - x - 0.5, y + 0.5, height - y - 0.5) * resolution
            blocker_clearance = min(
                (math.hypot(x - bx, y - by) * resolution for by, bx in zip(unsafe_y, unsafe_x, strict=True)),
                default=math.inf,
            )
            safe[y, x] = min(boundary_clearance, blocker_clearance) >= min_clearance_m

    reachable = np.zeros_like(safe)
    if 0 <= start.x < width and 0 <= start.y < height and safe[start.y, start.x]:
        reachable[start.y, start.x] = True
        queue = deque([(start.x, start.y)])
        for_current = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1))
        while queue:
            x, y = queue.popleft()
            for dx, dy in for_current:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height) or reachable[ny, nx] or not safe[ny, nx]:
                    continue
                if dx and dy and (not safe[y, nx] or not safe[ny, x]):
                    continue
                reachable[ny, nx] = True
                queue.append((nx, ny))

    def independent_los(source_x: int, source_y: int, target_x: int, target_y: int) -> bool:
        steps = max(abs(target_x - source_x), abs(target_y - source_y)) * 32 + 1
        seen: set[tuple[int, int]] = set()
        for index in range(1, steps + 1):
            fraction = index / steps
            world_x = (source_x + 0.5 + fraction * (target_x - source_x)) * resolution
            world_y = (source_y + 0.5 + fraction * (target_y - source_y)) * resolution
            cell_x = min(width - 1, max(0, int(world_x / resolution)))
            cell_y = min(height - 1, max(0, int(world_y / resolution)))
            if (cell_x, cell_y) in seen:
                continue
            seen.add((cell_x, cell_y))
            if (cell_x, cell_y) == (target_x, target_y):
                return True
            if truth.hard_obstacle[cell_y, cell_x] or truth.slope_deg[cell_y, cell_x] > max_slope_deg:
                return False
        return True

    coverable = np.zeros_like(free)
    reachable_cells = [(x, y) for y, x in np.argwhere(reachable)]
    for target_y, target_x in np.argwhere(free):
        for source_x, source_y in reachable_cells:
            distance = math.hypot(target_x - source_x, target_y - source_y) * resolution
            if distance <= sensor_range_m and independent_los(source_x, source_y, int(target_x), int(target_y)):
                coverable[target_y, target_x] = True
                break
    return safe, reachable, coverable


def _manual_truth(
    obstacle: np.ndarray,
    *,
    slope: np.ndarray | None = None,
    traversability: np.ndarray | None = None,
):
    from lunar_exploration_ppo.env.scenario import TruthMap
    from lunar_exploration_ppo.utils.geometry import GridGeometry

    geometry = GridGeometry(obstacle.shape[1], obstacle.shape[0], 1.0)
    return TruthMap(
        geometry=geometry,
        height=np.zeros(obstacle.shape, dtype=float),
        hard_obstacle=np.asarray(obstacle, dtype=bool),
        slope_deg=np.zeros(obstacle.shape, dtype=float) if slope is None else slope,
        traversability=np.ones(obstacle.shape, dtype=float) if traversability is None else traversability,
        provenance={
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "physical_obstacle_cells_written": False,
        },
    )


def test_exact_coverable_matches_independent_bruteforce_oracle_per_cell() -> None:
    from lunar_exploration_ppo.env.coverage import compute_coverage_masks
    from lunar_exploration_ppo.utils.geometry import CellXY

    corner_obstacles = np.zeros((5, 5), dtype=bool)
    corner_obstacles[1, 2] = True
    corner_obstacles[2, 1] = True
    wall_obstacles = np.zeros((7, 7), dtype=bool)
    wall_obstacles[:, 3] = True
    mixed_obstacles = np.zeros((7, 7), dtype=bool)
    mixed_slope = np.zeros((7, 7), dtype=float)
    mixed_slope[3, 4] = 31.0
    mixed_traversability = np.ones((7, 7), dtype=float)
    mixed_traversability[2, 4] = 0.49
    fixtures = (
        (_manual_truth(corner_obstacles), CellXY(1, 1), 3.0),
        (_manual_truth(wall_obstacles), CellXY(1, 3), 5.0),
        (
            _manual_truth(
                mixed_obstacles,
                slope=mixed_slope,
                traversability=mixed_traversability,
            ),
            CellXY(2, 3),
            3.0,
        ),
    )
    for truth, start, sensor_range in fixtures:
        expected_safe, expected_reachable, expected_coverable = _independent_coverable_oracle(
            truth,
            start,
            sensor_range_m=sensor_range,
            min_clearance_m=0.5215874761,
        )
        actual = compute_coverage_masks(
            truth,
            start,
            sensor_range_m=sensor_range,
            min_clearance_m=0.5215874761,
            max_slope_deg=30.0,
            traversability_threshold=0.5,
        )
        assert np.array_equal(actual.safe_free_mask, expected_safe)
        assert np.array_equal(actual.reachable_safe_mask, expected_reachable)
        assert np.array_equal(actual.coverable_mask, expected_coverable)

    open_truth = _manual_truth(np.zeros((7, 7), dtype=bool))
    open_masks = compute_coverage_masks(
        open_truth,
        CellXY(3, 3),
        sensor_range_m=2.0,
        min_clearance_m=0.5215874761,
        max_slope_deg=30.0,
        traversability_threshold=0.5,
    )
    assert open_masks.coverable_mask[3, 1]
    assert open_masks.coverable_mask[3, 5]


def test_exact_coverable_zero_denominator_fails_with_stable_reason() -> None:
    from lunar_exploration_ppo.env.coverage import compute_coverage_masks
    from lunar_exploration_ppo.utils.geometry import CellXY

    truth = _manual_truth(np.ones((5, 5), dtype=bool))
    with pytest.raises(ValueError, match="coverable denominator is zero"):
        compute_coverage_masks(
            truth,
            CellXY(2, 2),
            sensor_range_m=20.0,
            min_clearance_m=0.5215874761,
            max_slope_deg=30.0,
            traversability_threshold=0.5,
        )


def _independent_observed_opportunity_count(state, pose, sensor_range_m: float) -> tuple[int, int]:
    safe = state.observed_safe_mask
    height, width = safe.shape
    component = np.zeros_like(safe)
    if safe[pose.cell.y, pose.cell.x]:
        component[pose.cell.y, pose.cell.x] = True
        queue = deque([(pose.cell.x, pose.cell.y)])
        directions = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1))
        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height) or component[ny, nx] or not safe[ny, nx]:
                    continue
                if dx and dy and (not safe[y, nx] or not safe[ny, x]):
                    continue
                component[ny, nx] = True
                queue.append((nx, ny))

    unknown = ~state.observed_mask
    resolution = state.geometry.resolution_m
    radius_cells = sensor_range_m / resolution

    def clear_observed_los(source_x: int, source_y: int, target_x: int, target_y: int) -> bool:
        steps = max(abs(target_x - source_x), abs(target_y - source_y)) * 16 + 1
        seen: set[tuple[int, int]] = set()
        for index in range(1, steps + 1):
            fraction = index / steps
            cell_x = min(width - 1, max(0, int(source_x + 0.5 + fraction * (target_x - source_x))))
            cell_y = min(height - 1, max(0, int(source_y + 0.5 + fraction * (target_y - source_y))))
            if (cell_x, cell_y) in seen:
                continue
            seen.add((cell_x, cell_y))
            if unknown[cell_y, cell_x]:
                return True
            if state.obstacle[cell_y, cell_x] or state.slope_deg[cell_y, cell_x] > 30.0:
                return False
        return False

    opportunities = 0
    for source_y, source_x in np.argwhere(component):
        min_x = max(0, math.floor(source_x - radius_cells))
        max_x = min(width - 1, math.ceil(source_x + radius_cells))
        min_y = max(0, math.floor(source_y - radius_cells))
        max_y = min(height - 1, math.ceil(source_y + radius_cells))
        local_y, local_x = np.nonzero(unknown[min_y : max_y + 1, min_x : max_x + 1])
        has_opportunity = False
        for target_y, target_x in zip(local_y + min_y, local_x + min_x, strict=True):
            if math.hypot(target_x - source_x, target_y - source_y) > radius_cells:
                continue
            if clear_observed_los(int(source_x), int(source_y), int(target_x), int(target_y)):
                has_opportunity = True
                break
        opportunities += int(has_opportunity)
    return opportunities, int(np.count_nonzero(component))


def test_production_no_candidate_terminal_has_zero_independent_observed_opportunities() -> None:
    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import LunarExplorationEnv
    from lunar_exploration_ppo.workflows.stage1 import select_rule_action

    env = LunarExplorationEnv(load_stage1_config(STAGE1_CONFIG))
    observation = env.reset()
    while not env.is_done:
        result = env.step(select_rule_action(observation))
        observation = result.observation
    if env.terminal_reason == "no_candidate_done":
        opportunity_count, component_size = _independent_observed_opportunity_count(
            env.observed_state,
            env.pose,
            env.config.sensor_range_m,
        )
        assert component_size > 0
        assert opportunity_count == 0
    else:
        assert env.terminal_reason == "success_done"


def test_real_frontier_generator_empty_with_observed_opportunity_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import LunarExplorationEnv
    from lunar_exploration_ppo.env.frontier import FrontierActionSet, FrontierGenerator

    config = load_stage1_config(STAGE1_CONFIG)
    generator = FrontierGenerator(top_m=config.frontier_top_m)
    env = LunarExplorationEnv(config, frontier_generator=generator)

    def forced_empty(observed_state, prior, pose):
        return FrontierActionSet(
            cells=(),
            frontier_features=np.zeros((config.frontier_top_m, 22), dtype=np.float32),
            candidate_mask=np.zeros((config.frontier_top_m,), dtype=bool),
        )

    monkeypatch.setattr(generator, "extract", forced_empty)
    with pytest.raises(RuntimeError, match="production frontier is empty.*opportunities remain"):
        env.reset()


def test_real_frontier_generator_zero_opportunity_is_legal_no_candidate_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import LunarExplorationEnv
    from lunar_exploration_ppo.env.frontier import FrontierGenerator
    from lunar_exploration_ppo.workflows.stage1 import select_rule_action

    config = load_stage1_config(STAGE1_CONFIG)
    generator = FrontierGenerator(top_m=config.frontier_top_m)
    production_extract = generator.extract
    call_count = 0

    def exhaust_unknown_before_second_extract(observed_state, prior, pose):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            observed_state.observed_mask[...] = True
        return production_extract(observed_state, prior, pose)

    monkeypatch.setattr(generator, "extract", exhaust_unknown_before_second_extract)
    env = LunarExplorationEnv(config, frontier_generator=generator)
    observation = env.reset()
    assert env.current_action_set.candidate_count > 0

    result = env.step(select_rule_action(observation))

    assert result.done is True
    assert result.reason == "no_candidate_done"
    assert result.terminal is True
    assert result.bootstrap_value == 0.0
    assert result.diagnostics.frontier.candidate_count == 0
    assert result.diagnostics.frontier.unknown_count == 0
    assert result.diagnostics.frontier.oracle_opportunity_count == 0


def test_observed_frontier_opportunity_audit_distinguishes_remaining_gain_from_exhaustion() -> None:
    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import LunarExplorationEnv
    from lunar_exploration_ppo.env.frontier_oracle import audit_frontier_opportunities

    env = LunarExplorationEnv(load_stage1_config(STAGE1_CONFIG))
    env.reset()
    initial = audit_frontier_opportunities(
        env.observed_state,
        env.pose,
        sensor_range_m=env.config.sensor_range_m,
    )
    assert initial.component_size > 0
    assert initial.unknown_count > 0
    assert initial.opportunity_count > 0

    env.observed_state.observed_mask[...] = True
    exhausted = audit_frontier_opportunities(
        env.observed_state,
        env.pose,
        sensor_range_m=env.config.sensor_range_m,
    )
    assert exhausted.unknown_count == 0
    assert exhausted.opportunity_count == 0


def test_no_candidate_diagnostics_record_independent_oracle_and_remaining_counts() -> None:
    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import LunarExplorationEnv
    from lunar_exploration_ppo.env.frontier import FrontierActionSet

    class EmptyGenerator:
        def extract(self, observed_state, prior, pose):
            return FrontierActionSet(
                cells=(),
                frontier_features=np.zeros((512, 22), dtype=np.float32),
                candidate_mask=np.zeros((512,), dtype=bool),
            )

    env = LunarExplorationEnv(
        load_stage1_config(STAGE1_CONFIG),
        frontier_generator=EmptyGenerator(),
    )
    observation = env.reset()
    diagnostics = env.last_reset_diagnostics.frontier
    assert not np.any(observation.candidate_mask)
    assert diagnostics.candidate_count == 0
    assert diagnostics.component_size > 0
    assert diagnostics.unknown_count > 0
    assert diagnostics.remaining_unobserved_coverable_count > 0
    assert diagnostics.oracle_opportunity_count > 0


def test_packaging_declares_planner_and_ast_confines_direct_external_imports() -> None:
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(metadata["project"]["dependencies"])
    assert "numpy>=1.26,<2.3" in dependencies
    assert "path-planner==0.1.0" in dependencies

    direct_importers: list[str] = []
    source_root = REPO_ROOT / "src" / "lunar_exploration_ppo"
    for source_file in source_root.rglob("*.py"):
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name == "path_planner" or alias.name.startswith("path_planner.") for alias in node.names):
                direct_importers.append(source_file.relative_to(source_root).as_posix())
            if isinstance(node, ast.ImportFrom) and node.module and (
                node.module == "path_planner" or node.module.startswith("path_planner.")
            ):
                direct_importers.append(source_file.relative_to(source_root).as_posix())
    assert sorted(set(direct_importers)) == ["integrations/path_planner_adapter.py"]
    integrations_init = (source_root / "integrations" / "__init__.py").read_text(encoding="utf-8")
    assert "import_module" not in integrations_init
    assert '".path_" + "planner_adapter"' not in integrations_init
