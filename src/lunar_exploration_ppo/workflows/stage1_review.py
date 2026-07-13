"""Stage 1 independent-review parsing, transition, and source binding."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict

from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.workflows.stage1_artifacts import (
    FrozenDirectorySnapshot,
    FrozenFileSnapshot,
    STAGE1_FORBIDDEN_ARTIFACTS,
    STAGE1_MACHINE_ARTIFACTS,
    Stage1WorkflowError,
    load_stage1_machine_config,
    load_stage1_machine_config_bytes,
    read_object,
    strict_json_object_from_bytes,
    verify_manifest_entries,
    verify_manifest_entries_bytes,
)
from lunar_exploration_ppo.workflows.stage1_source import (
    STAGE1_REVIEWED_PATHS,
    ReviewedSourceError,
    Stage1SourceSnapshot,
    compute_stage1_source_identity,
    compute_stage1_source_identity_from_snapshot,
)

if TYPE_CHECKING:
    from lunar_exploration_ppo.configs.stage1 import Stage1Config


_REVIEW_FRONT_MATTER_KEYS: Final = frozenset(
    {
        "schema_version",
        "spec_verdict",
        "quality_verdict",
        "critical_count",
        "important_count",
        "minor_count",
        "final_gate_conclusion",
        "foundation_base_commit",
        "reviewed_prospective_git_tree",
        "review_package_sha256",
        "review_package_bytes",
        "review_package_lf_count",
        "review_package_logical_line_count",
    }
)
_INTEGER_FRONT_MATTER_KEYS: Final = frozenset(
    {
        "critical_count",
        "important_count",
        "minor_count",
        "review_package_bytes",
        "review_package_lf_count",
        "review_package_logical_line_count",
    }
)
_REVIEW_BODY_ANCHORS: Final = (
    (
        "Specification compliance verdict:",
        "Specification compliance verdict: APPROVED",
    ),
    (
        "Code quality verdict:",
        "Code quality verdict: APPROVED",
    ),
    (
        "Final gate conclusion:",
        "Final gate conclusion: READY_FOR_HUMAN_APPROVAL",
    ),
)
_OID_PATTERN: Final = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")


class Stage1ReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    stage_root: Path
    review_path: Path


_ImmutableFileSnapshot = FrozenFileSnapshot


@dataclass(frozen=True, slots=True)
class _VerifiedReviewMachineInputs:
    summary: dict[str, object]
    config: Stage1Config
    machine_source_identity: dict[str, object]
    current_source_identity: dict[str, object]
    manifest_hash: str
    config_hash: str


@dataclass(frozen=True, slots=True)
class _ReviewAuthoritySnapshot:
    stage: FrozenDirectorySnapshot
    source: Stage1SourceSnapshot
    foundation_gate: _ImmutableFileSnapshot
    foundation_environment: _ImmutableFileSnapshot
    foundation_environment_files: tuple[tuple[str, _ImmutableFileSnapshot], ...]
    review_report: _ImmutableFileSnapshot
    review_package: _ImmutableFileSnapshot


def _require_review_authority_snapshot_current(
    snapshot: _ReviewAuthoritySnapshot,
    *,
    review_expected: bool = False,
) -> None:
    snapshot.stage.require_current(
        "Stage 1 machine authority",
        allowed_added_regular_files=("review.json",) if review_expected else (),
    )
    _require_snapshot_current(snapshot.foundation_gate, "Foundation gate")
    _require_snapshot_current(snapshot.foundation_environment, "Foundation environment manifest")
    for relative, environment_file in snapshot.foundation_environment_files:
        _require_snapshot_current(
            environment_file,
            f"Foundation environment file/{relative}",
        )
    _require_snapshot_current(snapshot.review_report, "review report")
    _require_snapshot_current(snapshot.review_package, "review package")
    try:
        snapshot.source.require_current()
    except ReviewedSourceError as exc:
        raise Stage1WorkflowError("reviewed source snapshot drift detected") from exc


def _capture_foundation_environment_files(
    manifest_snapshot: FrozenFileSnapshot,
) -> tuple[tuple[str, FrozenFileSnapshot], ...]:
    manifest = strict_json_object_from_bytes(
        manifest_snapshot.payload,
        "Foundation environment manifest",
    )
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise Stage1WorkflowError("Foundation environment manifest files are invalid")
    captured: list[tuple[str, FrozenFileSnapshot]] = []
    normalized: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"file", "size_bytes", "sha256"}:
            raise Stage1WorkflowError("Foundation environment manifest entry is invalid")
        declared = entry.get("file")
        if not isinstance(declared, str) or not declared or "\\" in declared:
            raise Stage1WorkflowError("Foundation environment manifest path is unsafe")
        relative = PurePosixPath(declared)
        if (
            relative.is_absolute()
            or relative.as_posix() != declared
            or any(part in {"", ".", ".."} or ":" in part for part in relative.parts)
        ):
            raise Stage1WorkflowError("Foundation environment manifest path is unsafe")
        folded = declared.casefold()
        if folded in normalized:
            raise Stage1WorkflowError(
                "Foundation environment manifest contains case-normalized duplicate paths"
            )
        normalized.add(folded)
        snapshot = FrozenFileSnapshot.capture(
            manifest_snapshot.path.parent / Path(*relative.parts),
            f"Foundation environment file/{declared}",
            authority_root=manifest_snapshot.path.parent,
        )
        if (
            entry.get("size_bytes") != snapshot.size_bytes
            or entry.get("sha256") != snapshot.sha256
        ):
            raise Stage1WorkflowError(f"Foundation environment file drift: {declared}")
        captured.append((declared, snapshot))
    return tuple(sorted(captured, key=lambda item: item[0]))


def capture_stage1_foundation_environment_files(
    manifest_snapshot: FrozenFileSnapshot,
) -> tuple[tuple[str, FrozenFileSnapshot], ...]:
    """Expose the shared frozen Foundation environment graph to the gate verifier."""

    return _capture_foundation_environment_files(manifest_snapshot)


def _capture_review_authority_snapshot(
    *,
    stage: Path,
    repo_root: str | Path,
    foundation_gate: str | Path,
    foundation_environment_manifest: str | Path,
    review_report: str | Path,
    review_package: str | Path,
) -> _ReviewAuthoritySnapshot:
    if os.path.lexists(stage / "review.json") or any(
        os.path.lexists(stage / name) for name in STAGE1_FORBIDDEN_ARTIFACTS
    ):
        raise Stage1WorkflowError("independent review found an existing review or forbidden artifact")
    expected_machine = set(STAGE1_MACHINE_ARTIFACTS) | {"manifest.json"}
    stage_snapshot = FrozenDirectorySnapshot.capture_exact_regular_files(
        stage,
        expected_machine,
        "Stage 1 machine authority",
    )
    config, _machine_source_identity = load_stage1_machine_config_bytes(
        stage_snapshot.file("config.json").payload
    )
    gate_snapshot = FrozenFileSnapshot.capture(foundation_gate, "Foundation gate")
    environment_snapshot = FrozenFileSnapshot.capture(
        foundation_environment_manifest,
        "Foundation environment manifest",
    )
    environment_files = _capture_foundation_environment_files(environment_snapshot)
    report_snapshot = FrozenFileSnapshot.capture(review_report, "review report")
    package_snapshot = FrozenFileSnapshot.capture(review_package, "review package")
    try:
        source_snapshot = Stage1SourceSnapshot.capture(
            repo_root,
            base_commit=config.foundation_base_commit,
        )
    except ReviewedSourceError as exc:
        raise Stage1WorkflowError("unable to freeze reviewed Stage 1 source identity") from exc
    snapshot = _ReviewAuthoritySnapshot(
        stage=stage_snapshot,
        source=source_snapshot,
        foundation_gate=gate_snapshot,
        foundation_environment=environment_snapshot,
        foundation_environment_files=environment_files,
        review_report=report_snapshot,
        review_package=package_snapshot,
    )
    _require_review_authority_snapshot_current(snapshot)
    return snapshot


def _verify_review_machine_snapshot(
    *,
    snapshot: _ReviewAuthoritySnapshot,
    run_id: str,
) -> _VerifiedReviewMachineInputs:
    stage = snapshot.stage
    summary = strict_json_object_from_bytes(stage.file("summary.json").payload, "summary")
    routing = strict_json_object_from_bytes(stage.file("routing.json").payload, "routing")
    if summary.get("state") != "machine_passed" or routing.get("route") != "awaiting_independent_review":
        raise Stage1WorkflowError("independent review requires machine_passed state")
    if summary.get("run_id") != run_id or routing.get("run_id") != run_id:
        raise Stage1WorkflowError("independent review run_id mismatch")
    verify_manifest_entries_bytes(
        stage.file("manifest.json").payload,
        {name: stage.file(name) for name in STAGE1_MACHINE_ARTIFACTS},
    )
    manifest_hash = stage.file("manifest.json").sha256
    config_snapshot = stage.file("config.json")
    config_hash = config_snapshot.sha256
    config, machine_source_identity = load_stage1_machine_config_bytes(config_snapshot.payload)
    if config.run_id != run_id:
        raise Stage1WorkflowError("independent review config run_id mismatch")
    if summary.get("config_hash") != config_hash:
        raise Stage1WorkflowError("independent review summary config hash mismatch")
    if summary.get("execution_source_identity") != machine_source_identity:
        raise Stage1WorkflowError("machine config and summary source identity mismatch")
    if snapshot.source.base_commit != config.foundation_base_commit:
        raise Stage1WorkflowError("reviewed source Foundation base commit mismatch")
    try:
        current_source_identity = compute_stage1_source_identity_from_snapshot(snapshot.source)
    except ReviewedSourceError as exc:
        raise Stage1WorkflowError("frozen reviewed source identity is invalid") from exc
    if current_source_identity != machine_source_identity:
        raise Stage1WorkflowError("machine execution source identity differs from reviewed source")
    return _VerifiedReviewMachineInputs(
        summary=summary,
        config=config,
        machine_source_identity=machine_source_identity,
        current_source_identity=current_source_identity,
        manifest_hash=manifest_hash,
        config_hash=config_hash,
    )


def _verify_review_machine_inputs(
    *,
    stage: Path,
    repo_root: str | Path,
    run_id: str,
) -> _VerifiedReviewMachineInputs:
    summary = read_object(stage / "summary.json", "summary")
    routing = read_object(stage / "routing.json", "routing")
    if summary.get("state") != "machine_passed" or routing.get("route") != "awaiting_independent_review":
        raise Stage1WorkflowError("independent review requires machine_passed state")
    if summary.get("run_id") != run_id or routing.get("run_id") != run_id:
        raise Stage1WorkflowError("independent review run_id mismatch")
    if (stage / "review.json").exists() or any(
        (stage / name).exists() for name in STAGE1_FORBIDDEN_ARTIFACTS
    ):
        raise Stage1WorkflowError("independent review found an existing review or forbidden artifact")
    _verify_machine_artifact_set(stage)

    verify_manifest_entries(stage)
    manifest_hash = hashlib.sha256((stage / "manifest.json").read_bytes()).hexdigest()
    config_path = stage / "config.json"
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    config, machine_source_identity = load_stage1_machine_config(config_path)
    if config.run_id != run_id:
        raise Stage1WorkflowError("independent review config run_id mismatch")
    if summary.get("config_hash") != config_hash:
        raise Stage1WorkflowError("independent review summary config hash mismatch")
    if summary.get("execution_source_identity") != machine_source_identity:
        raise Stage1WorkflowError("machine config and summary source identity mismatch")
    current_source_identity = compute_stage1_source_identity(
        repo_root,
        base_commit=config.foundation_base_commit,
    )
    if current_source_identity != machine_source_identity:
        raise Stage1WorkflowError("machine execution source identity differs from reviewed source")
    return _VerifiedReviewMachineInputs(
        summary=summary,
        config=config,
        machine_source_identity=machine_source_identity,
        current_source_identity=current_source_identity,
        manifest_hash=manifest_hash,
        config_hash=config_hash,
    )


def record_stage1_independent_review(
    *,
    stage_root: str | Path,
    repo_root: str | Path,
    foundation_gate: str | Path,
    foundation_environment_manifest: str | Path,
    review_report: str | Path,
    review_package: str | Path,
) -> Stage1ReviewResult:
    """Parse actual review evidence and append only a strictly bound review.json."""

    stage = Path(os.path.abspath(os.fspath(Path(stage_root).expanduser())))
    if stage.name != "s1":
        raise Stage1WorkflowError("independent review requires a canonical s1 stage root")
    run_id = stage.parent.name
    authority_snapshot = _capture_review_authority_snapshot(
        stage=stage,
        repo_root=repo_root,
        foundation_gate=foundation_gate,
        foundation_environment_manifest=foundation_environment_manifest,
        review_report=review_report,
        review_package=review_package,
    )
    machine_inputs = _verify_review_machine_snapshot(
        snapshot=authority_snapshot,
        run_id=run_id,
    )
    summary = machine_inputs.summary
    config = machine_inputs.config
    machine_source_identity = machine_inputs.machine_source_identity
    current_source_identity = machine_inputs.current_source_identity
    manifest_hash = machine_inputs.manifest_hash
    config_hash = machine_inputs.config_hash
    reviewed_git_tree = str(current_source_identity["prospective_git_tree"])
    front_matter, report_binding, package_binding = _verify_stage1_review_evidence_snapshots(
        repo_root=repo_root,
        foundation_base_commit=config.foundation_base_commit,
        reviewed_git_tree=reviewed_git_tree,
        report_snapshot=authority_snapshot.review_report,
        package_snapshot=authority_snapshot.review_package,
        source_snapshot=authority_snapshot.source,
    )

    approval_challenge = build_stage1_approval_challenge(
        goal_id=config.goal_id,
        stage_id=config.stage_id,
        run_id=run_id,
        reviewed_git_tree=reviewed_git_tree,
        review_report_sha256=str(report_binding["sha256"]),
        review_package_sha256=str(package_binding["sha256"]),
        manifest_hash=manifest_hash,
    )
    review = {
        "schema_version": "ppo_highres_frontier_stage1_independent_review/v2",
        "state": "awaiting_human_approval",
        "goal_id": config.goal_id,
        "stage_id": config.stage_id,
        "run_id": run_id,
        "spec_verdict": front_matter["spec_verdict"],
        "quality_verdict": front_matter["quality_verdict"],
        "issue_counts": {
            "critical": front_matter["critical_count"],
            "important": front_matter["important_count"],
            "minor": front_matter["minor_count"],
        },
        "final_gate_conclusion": front_matter["final_gate_conclusion"],
        "foundation_base_commit": config.foundation_base_commit,
        "reviewed_git_tree": reviewed_git_tree,
        "execution_source_identity": machine_source_identity,
        "execution_source_identity_hash": hashlib.sha256(
            ArtifactStore.canonical_json_bytes(machine_source_identity)
        ).hexdigest(),
        "review_report": report_binding,
        "review_package": package_binding,
        "config_hash": config_hash,
        "manifest_hash": manifest_hash,
        "scenario_hash": summary["scenario_hash"],
        "foundation_gate_hash": authority_snapshot.foundation_gate.sha256,
        "foundation_environment_hash": authority_snapshot.foundation_environment.sha256,
        "approval_challenge": approval_challenge,
        "review_recorded_at_utc": _utc_now(),
    }
    _require_review_authority_snapshot_current(authority_snapshot)
    review_path = stage / "review.json"
    expected_review_bytes = ArtifactStore.canonical_json_bytes(review)
    # Exclusive creation is the canonical commit point.  Any later failure keeps
    # the current shared authority in place so this append-only root fails closed.
    written_path = ArtifactStore(stage).write_json_exclusive("review.json", review)
    normalized_written = (
        Path(os.path.abspath(os.fspath(Path(written_path).expanduser())))
        if isinstance(written_path, (str, Path))
        else None
    )
    if normalized_written != review_path:
        raise Stage1WorkflowError("review writer returned an unexpected target path")
    _require_review_authority_snapshot_current(authority_snapshot, review_expected=True)
    written_snapshot = FrozenFileSnapshot.capture(review_path, "review authority")
    if written_snapshot.payload != expected_review_bytes:
        raise Stage1WorkflowError("review authority bytes changed after exclusive write")
    if strict_json_object_from_bytes(written_snapshot.payload, "review") != review:
        raise Stage1WorkflowError("review authority record changed after write")
    return Stage1ReviewResult(run_id=run_id, stage_root=stage, review_path=review_path)


def verify_stage1_review_evidence(
    *,
    repo_root: str | Path,
    foundation_base_commit: str,
    reviewed_git_tree: str,
    review_report: str | Path,
    review_package: str | Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Replay a Git-native review package and bind its strict review report."""

    report_snapshot = _capture_file_snapshot(Path(review_report), "review report")
    package_snapshot = _capture_file_snapshot(Path(review_package), "review package")
    try:
        source_snapshot = Stage1SourceSnapshot.capture(
            repo_root,
            base_commit=foundation_base_commit,
        )
    except ReviewedSourceError as exc:
        raise Stage1WorkflowError("unable to freeze reviewed Stage 1 source identity") from exc
    result = _verify_stage1_review_evidence_snapshots(
        repo_root=repo_root,
        foundation_base_commit=foundation_base_commit,
        reviewed_git_tree=reviewed_git_tree,
        report_snapshot=report_snapshot,
        package_snapshot=package_snapshot,
        source_snapshot=source_snapshot,
    )
    _require_snapshot_current(report_snapshot, "review report")
    _require_snapshot_current(package_snapshot, "review package")
    try:
        source_snapshot.require_current()
    except ReviewedSourceError as exc:
        raise Stage1WorkflowError("reviewed source snapshot drift detected") from exc
    return result


def _verify_stage1_review_evidence_snapshots(
    *,
    repo_root: str | Path,
    foundation_base_commit: str,
    reviewed_git_tree: str,
    report_snapshot: _ImmutableFileSnapshot,
    package_snapshot: _ImmutableFileSnapshot,
    source_snapshot: Stage1SourceSnapshot,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    _require_source_snapshot_context(
        source_snapshot,
        repo_root=repo_root,
        foundation_base_commit=foundation_base_commit,
    )
    report_binding = report_snapshot.binding()
    package_binding = package_snapshot.binding()
    front_matter, _body = _parse_review_report(report_snapshot)
    _verify_review_conclusions(front_matter)
    _verify_package_declarations(front_matter, package_binding)
    if front_matter["foundation_base_commit"] != foundation_base_commit:
        raise Stage1WorkflowError("review report Foundation base commit mismatch")
    if front_matter["reviewed_prospective_git_tree"] != reviewed_git_tree:
        raise Stage1WorkflowError("review report prospective Git tree mismatch")
    replay_stage1_review_package(
        repo_root=repo_root,
        foundation_base_commit=foundation_base_commit,
        review_package=package_snapshot,
        expected_reviewed_git_tree=reviewed_git_tree,
        source_snapshot=source_snapshot,
    )
    return front_matter, report_binding, package_binding


def verify_stage1_review_evidence_snapshots(
    *,
    repo_root: str | Path,
    foundation_base_commit: str,
    reviewed_git_tree: str,
    report_snapshot: _ImmutableFileSnapshot,
    package_snapshot: _ImmutableFileSnapshot,
    source_snapshot: Stage1SourceSnapshot,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Verify review evidence using only caller-owned frozen byte snapshots."""

    return _verify_stage1_review_evidence_snapshots(
        repo_root=repo_root,
        foundation_base_commit=foundation_base_commit,
        reviewed_git_tree=reviewed_git_tree,
        report_snapshot=report_snapshot,
        package_snapshot=package_snapshot,
        source_snapshot=source_snapshot,
    )


def _require_source_snapshot_context(
    snapshot: Stage1SourceSnapshot,
    *,
    repo_root: str | Path,
    foundation_base_commit: str,
) -> None:
    expected_repo = Path(os.path.abspath(os.fspath(Path(repo_root).expanduser())))
    if os.path.normcase(os.fspath(snapshot.repo_root)) != os.path.normcase(
        os.fspath(expected_repo)
    ):
        raise Stage1WorkflowError("frozen reviewed source repo root mismatch")
    if snapshot.base_commit != foundation_base_commit:
        raise Stage1WorkflowError("frozen reviewed source Foundation base commit mismatch")
    if len(snapshot.files) != len(STAGE1_REVIEWED_PATHS) or set(snapshot.payloads()) != set(
        STAGE1_REVIEWED_PATHS
    ):
        raise Stage1WorkflowError("frozen reviewed source path set is not exact")


def replay_stage1_review_package(
    *,
    repo_root: str | Path,
    foundation_base_commit: str,
    review_package: str | Path | _ImmutableFileSnapshot,
    expected_reviewed_git_tree: str,
    temporary_root: str | Path = "D:/xunce/tmp/ppo_frontier/git-index",
    source_snapshot: Stage1SourceSnapshot | None = None,
) -> str:
    """Apply the package to the Foundation tree in an isolated D-drive index."""

    repo = Path(repo_root).expanduser().resolve()
    captured_source = source_snapshot is None
    if source_snapshot is None:
        try:
            source_snapshot = Stage1SourceSnapshot.capture(
                repo,
                base_commit=foundation_base_commit,
            )
        except ReviewedSourceError as exc:
            raise Stage1WorkflowError("unable to freeze reviewed Stage 1 source identity") from exc
    _require_source_snapshot_context(
        source_snapshot,
        repo_root=repo,
        foundation_base_commit=foundation_base_commit,
    )
    captured_package = not isinstance(review_package, _ImmutableFileSnapshot)
    package_snapshot = (
        review_package
        if isinstance(review_package, _ImmutableFileSnapshot)
        else _capture_file_snapshot(Path(review_package), "review package")
    )
    temp_root = Path(temporary_root).expanduser().resolve()
    if temp_root.drive.upper() != "D:":
        raise Stage1WorkflowError("review package temporary Git index must be on D drive")
    if not repo.is_dir():
        raise Stage1WorkflowError("review package replay source is missing")
    if _OID_PATTERN.fullmatch(foundation_base_commit) is None:
        raise Stage1WorkflowError("review package Foundation base commit is invalid")
    if _OID_PATTERN.fullmatch(expected_reviewed_git_tree) is None:
        raise Stage1WorkflowError("review package expected tree is invalid")
    _require_empty_real_index(repo)
    temp_root.mkdir(parents=True, exist_ok=True)
    index_path = temp_root / f"stage1-review-{uuid.uuid4().hex}.index"
    lock_path = Path(str(index_path) + ".lock")
    package_copy = temp_root / f"stage1-review-{uuid.uuid4().hex}.diff"
    environment = os.environ.copy()
    environment["GIT_INDEX_FILE"] = str(index_path)
    replay_error: Stage1WorkflowError | None = None
    rebuilt_tree = ""
    try:
        descriptor = os.open(package_copy, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            stream = os.fdopen(descriptor, "wb")
        except Exception:
            os.close(descriptor)
            raise
        with stream:
            stream.write(package_snapshot.payload)
            stream.flush()
            os.fsync(stream.fileno())
        _git_review(repo, ["read-tree", foundation_base_commit], "temporary read-tree", environment)
        _git_review(
            repo,
            ["apply", "--cached", "--whitespace=nowarn", "--", str(package_copy)],
            "package apply",
            environment,
        )
        rebuilt_tree = _git_review(repo, ["write-tree"], "temporary write-tree", environment).strip()
        changed_paths = _git_review_bytes(
            repo,
            [
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                "-z",
                "--no-renames",
                foundation_base_commit,
                rebuilt_tree,
            ],
            "package path-set",
        )
        try:
            path_set = {
                item.decode("utf-8", errors="strict").replace("\\", "/")
                for item in changed_paths.split(b"\0")
                if item
            }
        except UnicodeDecodeError as exc:
            raise Stage1WorkflowError("review package path set is not strict UTF-8") from exc
        if path_set != set(STAGE1_REVIEWED_PATHS):
            raise Stage1WorkflowError(
                f"review package path set is not the exact {len(STAGE1_REVIEWED_PATHS)} reviewed paths"
            )
        if rebuilt_tree != expected_reviewed_git_tree:
            raise Stage1WorkflowError("review package rebuilt tree does not equal the reviewed tree")
        _verify_rebuilt_tree_matches_frozen_source(
            repo=repo,
            rebuilt_tree=rebuilt_tree,
            source_snapshot=source_snapshot,
        )
    except Stage1WorkflowError as exc:
        replay_error = exc
    finally:
        if lock_path.is_file():
            lock_path.unlink()
        if index_path.is_file():
            index_path.unlink()
        if package_copy.is_file():
            package_copy.unlink()
        try:
            _require_empty_real_index(repo)
        except Stage1WorkflowError as exc:
            replay_error = exc
    if replay_error is not None:
        raise replay_error
    if captured_package:
        _require_snapshot_current(package_snapshot, "review package")
    if captured_source:
        try:
            source_snapshot.require_current()
        except ReviewedSourceError as exc:
            raise Stage1WorkflowError("reviewed source snapshot drift detected") from exc
    return rebuilt_tree


def _verify_rebuilt_tree_matches_frozen_source(
    *,
    repo: Path,
    rebuilt_tree: str,
    source_snapshot: Stage1SourceSnapshot,
) -> None:
    records = _git_review_bytes(
        repo,
        [
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            rebuilt_tree,
            "--",
            *sorted(STAGE1_REVIEWED_PATHS),
        ],
        "rebuilt reviewed source tree",
    )
    entries: dict[str, tuple[bytes, bytes, str]] = {}
    for record in records.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, kind, raw_object_id = metadata.split(b" ", 2)
            relative = raw_path.decode("utf-8", errors="strict").replace("\\", "/")
            object_id = raw_object_id.decode("ascii", errors="strict")
        except (ValueError, UnicodeDecodeError) as exc:
            raise Stage1WorkflowError("rebuilt reviewed source tree listing is invalid") from exc
        if relative in entries:
            raise Stage1WorkflowError("rebuilt reviewed source tree contains duplicate paths")
        entries[relative] = (mode, kind, object_id)
    if set(entries) != set(STAGE1_REVIEWED_PATHS):
        raise Stage1WorkflowError("rebuilt reviewed source tree path set is not exact")
    for relative in sorted(STAGE1_REVIEWED_PATHS):
        mode, kind, object_id = entries[relative]
        if mode != b"100644" or kind != b"blob" or _OID_PATTERN.fullmatch(object_id) is None:
            raise Stage1WorkflowError(
                f"rebuilt reviewed source entry is not a regular 100644 blob: {relative}"
            )
        payload = _git_review_bytes(
            repo,
            ["cat-file", "blob", object_id],
            f"rebuilt reviewed source blob/{relative}",
        )
        if payload != source_snapshot.file(relative).payload:
            raise Stage1WorkflowError(
                f"review package blob differs from frozen reviewed source: {relative}"
            )


def _verify_machine_artifact_set(stage: Path) -> None:
    expected = set(STAGE1_MACHINE_ARTIFACTS) | {"manifest.json"}
    if {entry.name for entry in stage.iterdir()} != expected:
        raise Stage1WorkflowError("independent review requires the exact machine artifact set")


def _require_empty_real_index(repo: Path) -> None:
    completed = subprocess.run(
        ["git", "-C", str(repo), "diff", "--cached", "--quiet"],
        check=False,
        capture_output=True,
    )
    if completed.returncode == 1:
        raise Stage1WorkflowError("review package replay requires an empty real Git index")
    if completed.returncode != 0:
        raise Stage1WorkflowError("unable to verify the real Git index")


def _git_review(
    repo: Path,
    arguments: list[str],
    label: str,
    environment: dict[str, str],
) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    if completed.returncode != 0:
        raise Stage1WorkflowError(f"review package Git {label} failed")
    return completed.stdout


def _git_review_bytes(repo: Path, arguments: list[str], label: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise Stage1WorkflowError(f"review package Git {label} failed")
    return completed.stdout


def _parse_review_report(source: Path | _ImmutableFileSnapshot) -> tuple[dict[str, object], str]:
    try:
        snapshot = (
            source
            if isinstance(source, _ImmutableFileSnapshot)
            else _capture_file_snapshot(source, "review report")
        )
        payload = snapshot.payload
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Stage1WorkflowError("review report is not strict UTF-8") from exc
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise Stage1WorkflowError("review report strict front matter is missing")
    try:
        closing_index = lines.index("---", 1)
    except ValueError as exc:
        raise Stage1WorkflowError("review report strict front matter is not closed") from exc
    if "---" in lines[closing_index + 1 :]:
        raise Stage1WorkflowError("review report contains multiple front matter blocks")
    raw: dict[str, str] = {}
    for line in lines[1:closing_index]:
        if line.count(": ") != 1:
            raise Stage1WorkflowError("review report front matter line is invalid")
        key, value = line.split(": ", 1)
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key) or not value or key in raw:
            raise Stage1WorkflowError("review report front matter key is invalid or duplicated")
        raw[key] = value
    if set(raw) != _REVIEW_FRONT_MATTER_KEYS:
        raise Stage1WorkflowError("review report front matter schema is not exact")
    parsed: dict[str, object] = dict(raw)
    for key in _INTEGER_FRONT_MATTER_KEYS:
        value = raw[key]
        if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
            raise Stage1WorkflowError(f"review report {key} must be a nonnegative integer")
        parsed[key] = int(value)
    body_lines = lines[closing_index + 1 :]
    body = "\n".join(body_lines)
    for prefix, expected_line in _REVIEW_BODY_ANCHORS:
        anchored_lines = [line for line in body_lines if line.startswith(prefix)]
        if anchored_lines != [expected_line]:
            raise Stage1WorkflowError("review report body anchor is missing, duplicated, or contradictory")
    return parsed, body


def build_stage1_approval_challenge(
    *,
    goal_id: str,
    stage_id: str,
    run_id: str,
    reviewed_git_tree: str,
    review_report_sha256: str,
    review_package_sha256: str,
    manifest_hash: str,
) -> str:
    source = {
        "schema_version": "ppo_highres_frontier_stage1_approval_challenge/v1",
        "goal_id": goal_id,
        "stage_id": stage_id,
        "run_id": run_id,
        "reviewed_git_tree": reviewed_git_tree,
        "review_report_sha256": review_report_sha256,
        "review_package_sha256": review_package_sha256,
        "manifest_hash": manifest_hash,
    }
    return hashlib.sha256(ArtifactStore.canonical_json_bytes(source)).hexdigest()


def _verify_review_conclusions(front_matter: dict[str, object]) -> None:
    if front_matter["schema_version"] != "ppo_highres_frontier_stage1_review_report/v2":
        raise Stage1WorkflowError("review report schema version is invalid")
    if front_matter["spec_verdict"] != "APPROVED" or front_matter["quality_verdict"] != "APPROVED":
        raise Stage1WorkflowError("independent review verdicts must both be APPROVED")
    if front_matter["critical_count"] != 0 or front_matter["important_count"] != 0:
        raise Stage1WorkflowError("independent review has blocking findings")
    if front_matter["final_gate_conclusion"] != "READY_FOR_HUMAN_APPROVAL":
        raise Stage1WorkflowError("independent review gate is not ready for human approval")
    if _OID_PATTERN.fullmatch(str(front_matter["foundation_base_commit"])) is None:
        raise Stage1WorkflowError("review report Foundation base commit is invalid")
    if _OID_PATTERN.fullmatch(str(front_matter["reviewed_prospective_git_tree"])) is None:
        raise Stage1WorkflowError("review report prospective Git tree is invalid")


def _verify_package_declarations(
    front_matter: dict[str, object],
    package_binding: dict[str, object],
) -> None:
    expected = {
        "review_package_sha256": package_binding["sha256"],
        "review_package_bytes": package_binding["bytes"],
        "review_package_lf_count": package_binding["lf_count"],
        "review_package_logical_line_count": package_binding["logical_line_count"],
    }
    if any(front_matter[key] != value for key, value in expected.items()):
        raise Stage1WorkflowError("review package hash/bytes/line declarations drifted")
    if _SHA256_PATTERN.fullmatch(str(front_matter["review_package_sha256"])) is None:
        raise Stage1WorkflowError("review package SHA-256 declaration is invalid")


def _capture_file_snapshot(path: Path, label: str) -> _ImmutableFileSnapshot:
    return _ImmutableFileSnapshot.capture(path, label)


def _file_binding(path: Path) -> dict[str, object]:
    return _capture_file_snapshot(path, "file binding").binding()


def _require_snapshot_current(snapshot: _ImmutableFileSnapshot, label: str) -> None:
    snapshot.require_current(label)


def _frame_fingerprint_part(digest: object, label: str, kind: str, payload: bytes) -> None:
    for field in (label.encode("utf-8"), kind.encode("ascii"), payload):
        digest.update(len(field).to_bytes(8, "big"))
        digest.update(field)


def _path_fingerprint_parts(path: Path, label: str) -> list[tuple[str, str, bytes]]:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return [(label, "file", resolved.read_bytes())]
    if resolved.is_dir():
        return [(label, "dir", b"")]
    return [(label, "missing", b"")]


def _review_authority_fingerprint(
    *,
    stage: Path,
    repo_root: str | Path,
    foundation_gate: Path,
    foundation_environment_manifest: Path,
    review_report: Path,
    review_package: Path,
    base_commit: str,
    review_expected: bool,
) -> str:
    expected = set(STAGE1_MACHINE_ARTIFACTS) | {"manifest.json"}
    if review_expected:
        expected.add("review.json")
    actual = {entry.name for entry in stage.iterdir()}
    if actual != expected:
        raise Stage1WorkflowError("review authority artifact set changed")
    _require_empty_real_index(Path(repo_root).expanduser().resolve())
    source_identity = compute_stage1_source_identity(repo_root, base_commit=base_commit)
    parts: list[tuple[str, str, bytes]] = []
    for name in sorted(set(STAGE1_MACHINE_ARTIFACTS) | {"manifest.json"}):
        parts.extend(_path_fingerprint_parts(stage / name, f"stage/{name}"))
    for path, label in (
        (foundation_gate, "foundation/gate"),
        (foundation_environment_manifest, "foundation/environment-manifest"),
        (review_report, "review/report"),
        (review_package, "review/package"),
    ):
        parts.extend(_path_fingerprint_parts(path, label))
    try:
        environment = json.loads(foundation_environment_manifest.read_bytes().decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage1WorkflowError("Foundation environment manifest changed") from exc
    entries = environment.get("files") if isinstance(environment, dict) else None
    if not isinstance(entries, list):
        raise Stage1WorkflowError("Foundation environment manifest files changed")
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            raise Stage1WorkflowError("Foundation environment manifest entry changed")
        relative = Path(entry["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise Stage1WorkflowError("Foundation environment manifest path changed")
        parts.extend(
            _path_fingerprint_parts(
                foundation_environment_manifest.parent / relative,
                f"foundation/environment-file/{relative.as_posix()}",
            )
        )
    parts.append(
        (
            "repo/source-identity",
            "value",
            ArtifactStore.canonical_json_bytes(source_identity),
        )
    )
    digest = hashlib.sha256()
    digest.update(b"ppo-highres-frontier-stage1-review-authority/v1\0")
    for label, kind, payload in sorted(parts, key=lambda item: item[0]):
        _frame_fingerprint_part(digest, label, kind, payload)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
