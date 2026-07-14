"""Stage 3 CUDA machine workflow and frozen Stage 2 authority consumer."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch

from lunar_exploration_ppo.configs.stage2 import Stage2Config
from lunar_exploration_ppo.configs.stage3 import Stage3Config, load_stage3_config
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.workflows import stage2 as stage2_workflow
from lunar_exploration_ppo.workflows.stage1_artifacts import (
    FrozenFileSnapshot,
    Stage1WorkflowError,
    finite_tree,
    prepare_unique_run_root,
    strict_json_object_from_bytes,
    validate_run_id,
)
from lunar_exploration_ppo.workflows.stage3_gpu import (
    Stage3CudaAuditBundle,
    Stage3GpuGateError,
    evaluate_latency_gate,
    evaluate_microbatch_gate,
    MicrobatchTrial,
    run_stage3_cuda_audit,
)


STAGE3_ROOT_ARTIFACTS: Final = (
    "config.json",
    "summary.json",
    "routing.json",
    "manifest.json",
    "report.md",
    "metrics.jsonl",
    "phase-state.jsonl",
    "evidence",
)
STAGE3_MACHINE_ARTIFACTS: Final = (
    "config.json",
    "summary.json",
    "routing.json",
    "report.md",
    "metrics.jsonl",
    "phase-state.jsonl",
)
STAGE3_EVIDENCE_ARTIFACTS: Final = (
    "evidence/network_shape_audit.json",
    "evidence/logprob_recompute_audit.json",
    "evidence/mask_handling_audit.json",
    "evidence/cuda_latency_audit.json",
    "evidence/cuda_microbatch_audit.json",
    "evidence/sample_policy_outputs.npz",
    "evidence/stage2_authority_audit.json",
    "evidence/execution_identity_audit.json",
)
STAGE3_MANIFEST_BOUND_ARTIFACTS: Final = (
    *STAGE3_MACHINE_ARTIFACTS,
    *STAGE3_EVIDENCE_ARTIFACTS,
)
_STAGE2_APPROVAL_COMMIT: Final = "898911559ccc9ae8ef701b69a58a93b1d8a4d8d8"
_STAGE2_ROOT_ARTIFACTS: Final = {
    "config.json",
    "summary.json",
    "routing.json",
    "manifest.json",
    "report.md",
    "metrics.jsonl",
    "phase-state.jsonl",
    "evidence",
    "review.json",
    "approval.json",
    "gate.json",
}


class Stage3WorkflowError(RuntimeError):
    """Stage 3 cannot produce or trust machine evidence without drift."""


@dataclass(frozen=True, slots=True)
class Stage3WorkflowResult:
    run_id: str
    stage_root: Path
    summary: dict[str, object]
    manifest: dict[str, object]


@dataclass(frozen=True, slots=True)
class _Stage3MachinePayload:
    artifacts: dict[str, bytes]
    manifest: dict[str, object]
    summary: dict[str, object]


@dataclass(frozen=True, slots=True)
class FrozenStage2AuthorityHandle:
    """Verified Stage 2 identity plus every file snapshot that established it."""

    identity: dict[str, object]
    snapshots: tuple[tuple[str, FrozenFileSnapshot], ...]
    stage_root: Path
    root_artifacts: frozenset[str]
    evidence_artifacts: frozenset[str]

    def require_current(self, label: str = "Stage 2 authority") -> None:
        if (
            not self.stage_root.is_dir()
            or {path.name for path in self.stage_root.iterdir()} != self.root_artifacts
        ):
            raise Stage3WorkflowError(f"{label} root artifact set changed")
        evidence = self.stage_root / "evidence"
        if (
            not evidence.is_dir()
            or {path.name for path in evidence.iterdir()} != self.evidence_artifacts
        ):
            raise Stage3WorkflowError(f"{label} evidence artifact set changed")
        try:
            for snapshot_label, snapshot in self.snapshots:
                snapshot.require_current(f"{label} {snapshot_label}")
        except Stage1WorkflowError as exc:
            raise Stage3WorkflowError(f"{label} changed during verification") from exc


@dataclass(frozen=True, slots=True)
class Stage3MachineVerificationHandle:
    """Frozen machine artifacts and transitive Stage 2 authority."""

    stage_root: Path
    manifest_snapshot: FrozenFileSnapshot
    artifacts: tuple[tuple[str, FrozenFileSnapshot], ...]
    stage2_authority: FrozenStage2AuthorityHandle

    def artifact_snapshot(self, relative: str) -> FrozenFileSnapshot:
        for path, snapshot in self.artifacts:
            if path == relative:
                return snapshot
        raise Stage3WorkflowError(f"Stage 3 verified artifact is missing: {relative}")

    def require_current(self, label: str = "Stage 3 machine verification") -> None:
        try:
            self.stage2_authority.require_current(f"{label} Stage 2 authority")
        except Stage1WorkflowError as exc:
            raise Stage3WorkflowError(f"{label} Stage 2 authority changed") from exc
        try:
            self.manifest_snapshot.require_current(f"{label} manifest")
            for relative, snapshot in self.artifacts:
                snapshot.require_current(f"{label} artifact {relative}")
        except Stage1WorkflowError as exc:
            raise Stage3WorkflowError(f"{label} artifacts changed") from exc
        actual_root = {path.name for path in self.stage_root.iterdir()}
        if actual_root != set(STAGE3_ROOT_ARTIFACTS):
            raise Stage3WorkflowError(f"{label} root artifact set changed")
        evidence = self.stage_root / "evidence"
        expected_evidence = {
            Path(relative).name for relative in STAGE3_EVIDENCE_ARTIFACTS
        }
        if (
            not evidence.is_dir()
            or {path.name for path in evidence.iterdir()} != expected_evidence
        ):
            raise Stage3WorkflowError(f"{label} evidence artifact set changed")


def verify_frozen_git_source_identity(
    *,
    source_identity: object,
    repo_root: str | Path,
    commit: str,
) -> tuple[FrozenFileSnapshot, ...]:
    """Verify a recorded source-set against blobs in an immutable Git commit."""

    if not isinstance(source_identity, dict) or set(source_identity) != {
        "schema_version",
        "source_set_sha256",
        "paths",
    }:
        raise Stage3WorkflowError("frozen Stage 2 source identity schema drift")
    if source_identity.get("schema_version") != "stage2_reviewed_source_set/v1":
        raise Stage3WorkflowError("frozen Stage 2 source identity version drift")
    entries = source_identity.get("paths")
    if not isinstance(entries, list) or not entries:
        raise Stage3WorkflowError("frozen Stage 2 source path set is empty")
    repo = Path(repo_root).expanduser().resolve()
    _git_bytes(repo, ["cat-file", "-e", f"{commit}^{{commit}}"], "frozen commit")
    relative_paths: list[str] = []
    current_only_snapshots: list[FrozenFileSnapshot] = []
    digest = hashlib.sha256()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise Stage3WorkflowError("frozen Stage 2 source entry schema drift")
        relative = entry.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise Stage3WorkflowError("frozen Stage 2 source path is unsafe")
        relative_paths.append(relative)
        git_object = subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", f"{commit}:{relative}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if git_object.returncode == 0:
            payload = _git_bytes(
                repo,
                ["show", f"{commit}:{relative}"],
                f"frozen source {relative}",
            )
        else:
            try:
                snapshot = FrozenFileSnapshot.capture(
                    repo / relative,
                    f"recorded non-Git source {relative}",
                    authority_root=repo,
                )
            except Stage1WorkflowError as exc:
                raise Stage3WorkflowError(
                    "recorded non-Git Stage 2 source is missing or unsafe"
                ) from exc
            current_only_snapshots.append(snapshot)
            payload = snapshot.payload
        if (
            entry.get("size_bytes") != len(payload)
            or entry.get("sha256") != hashlib.sha256(payload).hexdigest()
        ):
            raise Stage3WorkflowError("frozen Stage 2 source blob drift")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    if relative_paths != sorted(set(relative_paths)):
        raise Stage3WorkflowError("frozen Stage 2 source path set is not exact")
    if source_identity.get("source_set_sha256") != digest.hexdigest():
        raise Stage3WorkflowError("frozen Stage 2 source-set hash drift")
    try:
        for snapshot in current_only_snapshots:
            snapshot.require_current("recorded non-Git Stage 2 source")
    except Stage1WorkflowError as exc:
        raise Stage3WorkflowError(
            "recorded non-Git Stage 2 source changed during verification"
        ) from exc
    return tuple(current_only_snapshots)


def verify_frozen_stage2_authority(
    *,
    gate_path: str | Path,
    repo_root: str | Path,
    expected_stage2_commit: str = _STAGE2_APPROVAL_COMMIT,
) -> FrozenStage2AuthorityHandle:
    """Consume a controller-persisted Stage 2 gate without issuing authority.

    Stage 2's original live verifier intentionally binds to its then-current source
    tree.  A later stage instead verifies the reviewed package and recorded source
    set against the immutable approved Stage 2 commit, while preserving all review,
    approval, manifest, challenge, and hash-chain checks.
    """

    repo = Path(repo_root).expanduser().resolve()
    gate_file = Path(gate_path).expanduser().resolve()
    stage = gate_file.parent
    if gate_file.name != "gate.json" or stage.name != "s2" or not stage.is_dir():
        raise Stage3WorkflowError("Stage 2 authority path must be canonical <run>/s2/gate.json")
    if {path.name for path in stage.iterdir()} != _STAGE2_ROOT_ARTIFACTS:
        raise Stage3WorkflowError("Stage 2 authority artifact set drift")
    if expected_stage2_commit != _STAGE2_APPROVAL_COMMIT:
        raise Stage3WorkflowError("Stage 2 approval commit drift")

    try:
        gate_snapshot = FrozenFileSnapshot.capture(
            gate_file,
            "Stage 2 gate",
            authority_root=stage,
        )
        review_snapshot = FrozenFileSnapshot.capture(
            stage / "review.json",
            "Stage 2 review",
            authority_root=stage,
        )
        approval_snapshot = FrozenFileSnapshot.capture(
            stage / "approval.json",
            "Stage 2 approval",
            authority_root=stage,
        )
        manifest_snapshot = FrozenFileSnapshot.capture(
            stage / "manifest.json",
            "Stage 2 manifest",
            authority_root=stage,
        )
        gate = strict_json_object_from_bytes(gate_snapshot.payload, "Stage 2 gate")
        review = strict_json_object_from_bytes(review_snapshot.payload, "Stage 2 review")
        approval = strict_json_object_from_bytes(
            approval_snapshot.payload,
            "Stage 2 approval",
        )
        manifest = strict_json_object_from_bytes(
            manifest_snapshot.payload,
            "Stage 2 manifest",
        )
    except Stage1WorkflowError as exc:
        raise Stage3WorkflowError("Stage 2 authority snapshot is invalid") from exc

    _require_canonical_json(gate_snapshot, gate, "Stage 2 gate")
    _require_canonical_json(review_snapshot, review, "Stage 2 review")
    _require_canonical_json(approval_snapshot, approval, "Stage 2 approval")
    _require_canonical_json(manifest_snapshot, manifest, "Stage 2 manifest")
    artifact_snapshots = _capture_manifest_artifact_snapshots(
        stage=stage,
        manifest=manifest,
        expected_paths=set(stage2_workflow.STAGE2_MANIFEST_BOUND_ARTIFACTS),
        label="Stage 2",
    )
    config_snapshot = artifact_snapshots["config.json"]
    try:
        machine_config_payload = strict_json_object_from_bytes(
            config_snapshot.payload,
            "Stage 2 config",
        )
    except Stage1WorkflowError as exc:
        raise Stage3WorkflowError("Stage 2 config snapshot is invalid") from exc

    recorded_source = machine_config_payload.pop("execution_source_identity", None)
    recorded_environment = machine_config_payload.pop(
        "execution_environment_identity",
        None,
    )
    try:
        machine_config = Stage2Config.model_validate(machine_config_payload)
        repository_config = Stage2Config.model_validate_json(
            _git_bytes(
                repo,
                ["show", f"{expected_stage2_commit}:configs/ppo_highres_frontier_stage2_v1.json"],
                "frozen Stage 2 config",
            ).decode("utf-8", errors="strict")
        )
    except Exception as exc:
        raise Stage3WorkflowError("frozen Stage 2 config binding drift") from exc
    if (
        machine_config.model_dump(exclude={"run_id"})
        != repository_config.model_dump(exclude={"run_id"})
        or machine_config.run_id != stage.parent.name
        or machine_config.authorized_next_stage
        != "ppo_highres_frontier_stage3_cross_attention_policy/v1"
    ):
        raise Stage3WorkflowError("frozen Stage 2 config binding drift")

    commit_tree = _git_text(
        repo,
        ["rev-parse", f"{expected_stage2_commit}^{{tree}}"],
        "Stage 2 commit tree",
    ).strip()
    commit_parent = _git_text(
        repo,
        ["rev-parse", f"{expected_stage2_commit}^"],
        "Stage 2 commit parent",
    ).strip()
    if commit_parent != machine_config.stage1_approval_commit:
        raise Stage3WorkflowError("Stage 2 commit parent is not the approved Stage 1 base")
    current_only_source_snapshots = verify_frozen_git_source_identity(
        source_identity=recorded_source,
        repo_root=repo,
        commit=expected_stage2_commit,
    )
    try:
        current_stage2_environment = stage2_workflow._environment_identity()
    except Exception as exc:
        raise Stage3WorkflowError("Stage 2 environment cannot be verified") from exc
    if recorded_environment != current_stage2_environment:
        raise Stage3WorkflowError("Stage 2 execution environment drift")

    summary = _strict_json_snapshot(
        artifact_snapshots["summary.json"],
        "Stage 2 summary",
    )
    routing = _strict_json_snapshot(
        artifact_snapshots["routing.json"],
        "Stage 2 routing",
    )
    acceptance = summary.get("acceptance")
    if (
        summary.get("state") != "machine_passed"
        or summary.get("run_id") != stage.parent.name
        or summary.get("stage_id") != machine_config.stage_id
        or summary.get("execution_source_identity") != recorded_source
        or summary.get("execution_environment_identity") != recorded_environment
        or summary.get("config_hash") != hashlib.sha256(config_snapshot.payload).hexdigest()
        or summary.get("checkpoint_state") != "stage2_no_checkpoint/v1"
        or not isinstance(acceptance, dict)
        or acceptance.get("all_18_stage2_items_passed") is not True
        or not isinstance(acceptance.get("items"), dict)
        or len(acceptance["items"]) != 18
        or not all(value is True for value in acceptance["items"].values())
        or routing.get("route") != "awaiting_independent_review"
        or routing.get("authorized_next_stage") != machine_config.authorized_next_stage
        or routing.get("next_stage_entered") is not False
    ):
        raise Stage3WorkflowError("Stage 2 machine acceptance binding drift")

    try:
        report_snapshot = stage2_workflow._capture_stage2_recorded_snapshot(
            review.get("review_report"),
            "Stage 2 review report",
        )
        package_snapshot = stage2_workflow._capture_stage2_recorded_snapshot(
            review.get("review_package"),
            "Stage 2 review package",
        )
        parsed = stage2_workflow._parse_stage2_review_report(report_snapshot)
        stage2_workflow._verify_stage2_review_conclusions(parsed)
    except (Stage1WorkflowError, stage2_workflow.Stage2WorkflowError) as exc:
        raise Stage3WorkflowError("Stage 2 independent review authority drift") from exc

    reviewed_base = str(parsed.get("reviewed_base_commit", ""))
    reviewed_tree = str(parsed.get("reviewed_prospective_git_tree", ""))
    reviewed_paths = parsed.get("reviewed_paths")
    if (
        reviewed_base != commit_parent
        or reviewed_tree != commit_tree
        or not isinstance(reviewed_paths, list)
    ):
        raise Stage3WorkflowError("Stage 2 reviewed Git commit binding drift")
    try:
        stage2_workflow.replay_stage2_review_package(
            repo_root=repo,
            base_commit=reviewed_base,
            review_package=package_snapshot.path,
            expected_prospective_git_tree=reviewed_tree,
            expected_reviewed_paths=reviewed_paths,
        )
    except stage2_workflow.Stage2WorkflowError as exc:
        raise Stage3WorkflowError("Stage 2 review package replay failed") from exc

    bindings: dict[str, object] = {
        "goal_id": machine_config.goal_id,
        "stage_id": machine_config.stage_id,
        "run_id": stage.parent.name,
        "reviewed_base_commit": reviewed_base,
        "reviewed_prospective_git_tree": reviewed_tree,
        "reviewed_paths": reviewed_paths,
        "reviewed_path_set_sha256": _hash_json(reviewed_paths),
        "review_package_sha256": package_snapshot.sha256,
        "review_package_bytes": package_snapshot.size_bytes,
        "review_package_lf_count": package_snapshot.lf_count,
        "review_package_logical_line_count": package_snapshot.logical_line_count,
        "source_set_sha256": str(recorded_source["source_set_sha256"]),
        "data_sha256": summary.get("catalog_sha256"),
        "config_sha256": config_snapshot.sha256,
        "environment_sha256": _hash_json(recorded_environment),
        "manifest_sha256": manifest_snapshot.sha256,
        "authorized_next_stage": machine_config.authorized_next_stage,
    }
    try:
        stage2_workflow._verify_stage2_review_report_bindings(parsed, bindings)
        challenge = stage2_workflow.build_stage2_approval_challenge(
            goal_id=str(bindings["goal_id"]),
            stage_id=str(bindings["stage_id"]),
            run_id=str(bindings["run_id"]),
            reviewed_prospective_git_tree=reviewed_tree,
            review_report_sha256=report_snapshot.sha256,
            review_package_sha256=package_snapshot.sha256,
            manifest_sha256=manifest_snapshot.sha256,
            config_sha256=config_snapshot.sha256,
            data_sha256=str(bindings["data_sha256"]),
            environment_sha256=str(bindings["environment_sha256"]),
            source_set_sha256=str(bindings["source_set_sha256"]),
            authorized_next_stage=str(bindings["authorized_next_stage"]),
        )
        review_recorded_at = str(review.get("review_recorded_at_utc", ""))
        expected_review = stage2_workflow._build_stage2_review_record(
            parsed=parsed,
            bindings=bindings,
            report_binding=report_snapshot.binding(),
            package_binding=package_snapshot.binding(),
            approval_challenge=challenge,
            review_recorded_at_utc=review_recorded_at,
        )
        if review != expected_review:
            raise Stage3WorkflowError("Stage 2 review record semantic drift")
        authority_bindings = {
            key: bindings[key]
            for key in (
                "goal_id",
                "stage_id",
                "run_id",
                "reviewed_base_commit",
                "reviewed_prospective_git_tree",
                "reviewed_path_set_sha256",
                "source_set_sha256",
                "data_sha256",
                "config_sha256",
                "environment_sha256",
                "manifest_sha256",
                "authorized_next_stage",
            )
        }
        authority_bindings.update(
            {
                "review_report_sha256": report_snapshot.sha256,
                "review_report_bytes": report_snapshot.size_bytes,
                "review_report_lf_count": report_snapshot.lf_count,
                "review_report_logical_line_count": report_snapshot.logical_line_count,
                "review_package_sha256": package_snapshot.sha256,
                "review_package_bytes": package_snapshot.size_bytes,
                "review_package_lf_count": package_snapshot.lf_count,
                "review_package_logical_line_count": package_snapshot.logical_line_count,
                "review_sha256": review_snapshot.sha256,
                "approval_challenge": challenge,
                "review_recorded_at_utc": review_recorded_at,
                "approval_sha256": stage2_workflow._ABSENT_AUTHORITY_HASH,
            }
        )
        stage2_workflow._validate_stage2_approval(
            approval,
            stage=stage,
            review_hash=review_snapshot.sha256,
            approval_challenge=challenge,
            review_recorded_at_utc=review_recorded_at,
            bindings=authority_bindings,
        )
        authority_bindings["approval_sha256"] = approval_snapshot.sha256
    except stage2_workflow.Stage2WorkflowError as exc:
        raise Stage3WorkflowError("Stage 2 approval authority drift") from exc

    if set(gate) != {
        "schema_version",
        "state",
        "authorized_next_stage",
        "run_id",
        "bindings",
        "history",
    }:
        raise Stage3WorkflowError("Stage 2 gate schema drift")
    if (
        gate.get("schema_version") != "ppo_highres_frontier_stage2_verified_gate/v1"
        or gate.get("state") != "next_stage"
        or gate.get("authorized_next_stage") != machine_config.authorized_next_stage
        or gate.get("run_id") != stage.parent.name
        or gate.get("bindings") != authority_bindings
        or gate.get("history")
        != stage2_workflow._build_stage2_gate_history(authority_bindings)
    ):
        raise Stage3WorkflowError("Stage 2 gate binding or history drift")

    identity: dict[str, object] = {
        "schema_version": "ppo_highres_frontier_stage3_stage2_authority_audit/v1",
        "verified": True,
        "stage2_commit": expected_stage2_commit,
        "stage2_commit_tree": commit_tree,
        "stage2_run_id": stage.parent.name,
        "gate_path": str(gate_file),
        "gate_sha256": gate_snapshot.sha256,
        "review_sha256": review_snapshot.sha256,
        "approval_sha256": approval_snapshot.sha256,
        "review_package_sha256": package_snapshot.sha256,
        "source_set_sha256": str(recorded_source["source_set_sha256"]),
        "authorized_stage": machine_config.authorized_next_stage,
        "authority_scope": (
            "external_controller_claim_integrity_not_local_identity_authentication/v1"
        ),
    }
    snapshots = (
        ("gate.json", gate_snapshot),
        ("review.json", review_snapshot),
        ("approval.json", approval_snapshot),
        ("manifest.json", manifest_snapshot),
        ("review report", report_snapshot),
        ("review package", package_snapshot),
        *tuple(
            (f"manifest artifact {relative}", snapshot)
            for relative, snapshot in sorted(artifact_snapshots.items())
        ),
        *tuple(
            (f"recorded non-Git source {index}", snapshot)
            for index, snapshot in enumerate(current_only_source_snapshots)
        ),
    )
    handle = FrozenStage2AuthorityHandle(
        identity=identity,
        snapshots=snapshots,
        stage_root=stage,
        root_artifacts=frozenset(_STAGE2_ROOT_ARTIFACTS),
        evidence_artifacts=frozenset(
            Path(relative).name
            for relative in stage2_workflow.STAGE2_MANIFEST_BOUND_ARTIFACTS
            if Path(relative).parent.name == "evidence"
        ),
    )
    handle.require_current()
    return handle


def stage3_source_identity(repo_root: str | Path) -> dict[str, object]:
    """Hash the exact Stage 3 implementation, tests, config, workflow, and specs."""

    repo = Path(repo_root).expanduser().resolve()
    paths: set[Path] = set((repo / "src/lunar_exploration_ppo").rglob("*.py"))
    paths.update((repo / "tests/ppo_highres_frontier").glob("test_stage3_*.py"))
    paths.update(
        {
            repo / "pyproject.toml",
            repo / "configs/ppo_highres_frontier_stage3_v1.json",
            repo / "scripts/run_ppo_highres_frontier_stage3.py",
            repo / ".github/workflows/platform-compatibility.yml",
            repo
            / "docs/superpowers/specs/2026-07-09-ppo-highres-frontier-map-exploration-design.md",
            repo
            / "docs/superpowers/plans/2026-07-10-ppo-highres-frontier-map-exploration.md",
            repo / "docs/ppo-highres-frontier-stage3.md",
        }
    )
    entries: list[dict[str, object]] = []
    digest = hashlib.sha256()
    try:
        ordered = sorted(paths, key=lambda path: path.relative_to(repo).as_posix())
    except ValueError as exc:
        raise Stage3WorkflowError("Stage 3 source path escaped the repository") from exc
    for path in ordered:
        if not path.is_file():
            raise Stage3WorkflowError(f"Stage 3 source binding path is missing: {path}")
        relative = path.relative_to(repo).as_posix()
        payload = path.read_bytes()
        entries.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return {
        "schema_version": "stage3_reviewed_source_set/v1",
        "source_set_sha256": digest.hexdigest(),
        "paths": entries,
    }


def stage3_environment_identity() -> dict[str, object]:
    """Capture stable runtime and physical CUDA identity without free-memory noise."""

    if not torch.cuda.is_available():
        raise Stage3WorkflowError("Stage 3 requires CUDA; CPU fallback is forbidden")
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    return {
        "schema_version": "ppo_highres_frontier_stage3_environment/v1",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": True,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_index": torch.cuda.current_device(),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "cuda_compute_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_total_memory_bytes": int(properties.total_memory),
        "compute_dtype": "float32",
        "amp_enabled": False,
    }


def stage3_git_identity(
    repo_root: str | Path,
    *,
    base_commit: str = _STAGE2_APPROVAL_COMMIT,
) -> dict[str, object]:
    """Build the prospective dirty tree in a D-drive temporary index."""

    repo = Path(repo_root).expanduser().resolve()
    _require_empty_real_index(repo)
    head = _git_text(repo, ["rev-parse", "HEAD"], "HEAD").strip()
    if head != base_commit:
        raise Stage3WorkflowError("Stage 3 HEAD is not the frozen Stage 2 approval commit")
    index_path = _temporary_index_path("stage3-prospective")
    environment = os.environ.copy()
    environment["GIT_INDEX_FILE"] = str(index_path)
    try:
        _git_text(repo, ["read-tree", base_commit], "prospective read-tree", environment)
        _git_text(repo, ["add", "-A", "--", "."], "prospective add", environment)
        tree = _git_text(repo, ["write-tree"], "prospective write-tree", environment).strip()
        output = _git_bytes(
            repo,
            [
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                "-z",
                "--no-renames",
                base_commit,
                tree,
            ],
            "prospective path set",
            environment,
        )
    finally:
        _remove_temporary_index(index_path)
    _require_empty_real_index(repo)
    try:
        paths = sorted(
            item.decode("utf-8", errors="strict").replace("\\", "/")
            for item in output.split(b"\0")
            if item
        )
    except UnicodeDecodeError as exc:
        raise Stage3WorkflowError("Stage 3 prospective path set is not UTF-8") from exc
    return {
        "schema_version": "ppo_highres_frontier_stage3_prospective_git_tree/v1",
        "head_commit": head,
        "base_commit": base_commit,
        "prospective_git_tree": tree,
        "changed_paths": paths,
        "changed_path_set_sha256": _hash_json(paths),
        "real_index_empty": True,
    }


def run_stage3_workflow(
    *,
    config_path: str | Path,
    run_id: str,
    stage2_gate_path: str | Path,
    base_output_root: str | Path | None = None,
) -> Stage3WorkflowResult:
    """Run Stage 3 machine checks and stop at awaiting independent review."""

    validate_run_id(run_id)
    repo = Path(__file__).resolve().parents[3]
    config_source = Path(config_path).expanduser().resolve()
    try:
        config_source_snapshot = FrozenFileSnapshot.capture(
            config_source,
            "Stage 3 repository config",
            authority_root=repo,
        )
        config = Stage3Config.model_validate_json(config_source_snapshot.payload)
    except (Stage1WorkflowError, ValueError) as exc:
        raise Stage3WorkflowError("Stage 3 repository config snapshot is invalid") from exc
    if config.run_id is not None:
        raise Stage3WorkflowError("repository Stage 3 config must not pre-bind a run ID")
    base_root = Path(base_output_root) if base_output_root is not None else Path(config.output_root)
    run_root = (base_root / run_id).expanduser().resolve()
    if run_root.drive.upper() != "D:":
        raise Stage3WorkflowError("Stage 3 runtime output must be on D drive")
    if run_root.exists():
        raise Stage3WorkflowError("Stage 3 run root already exists")

    source_identity = stage3_source_identity(repo)
    environment_identity = stage3_environment_identity()
    git_identity = stage3_git_identity(repo, base_commit=config.stage2_approval_commit)
    authority_handle = verify_frozen_stage2_authority(
        gate_path=stage2_gate_path,
        repo_root=repo,
        expected_stage2_commit=config.stage2_approval_commit,
    )
    stage2_authority = authority_handle.identity
    try:
        audit = run_stage3_cuda_audit(config)
    except Stage3GpuGateError as exc:
        raise Stage3WorkflowError("Stage 3 CUDA machine gate failed") from exc

    final_source = stage3_source_identity(repo)
    final_environment = stage3_environment_identity()
    final_git = stage3_git_identity(repo, base_commit=config.stage2_approval_commit)
    if (
        source_identity != final_source
        or environment_identity != final_environment
        or git_identity != final_git
    ):
        raise Stage3WorkflowError("source, environment, or Git drift during Stage 3")
    authority_handle.require_current("Stage 3 controller Stage 2 authority")
    try:
        config_source_snapshot.require_current("Stage 3 repository config")
    except Stage1WorkflowError as exc:
        raise Stage3WorkflowError("Stage 3 config changed during execution") from exc

    payload = _build_stage3_machine_payload(
        config=config,
        run_id=run_id,
        source_identity=source_identity,
        environment_identity=environment_identity,
        git_identity=git_identity,
        stage2_authority=stage2_authority,
        audit=audit,
    )
    prepare_unique_run_root(run_root)
    stage = run_root / "s3"
    stage.mkdir(exist_ok=False)
    store = ArtifactStore(stage)
    for relative in STAGE3_MANIFEST_BOUND_ARTIFACTS:
        store.write_bytes(relative, payload.artifacts[relative])
    store.write_json("manifest.json", payload.manifest)
    verification = verify_stage3_machine_run(
        stage_root=stage,
        repo_root=repo,
        stage2_gate_path=stage2_gate_path,
    )
    authority_handle.require_current("Stage 3 controller final Stage 2 authority")
    try:
        config_source_snapshot.require_current("Stage 3 controller final repository config")
    except Stage1WorkflowError as exc:
        raise Stage3WorkflowError("Stage 3 config changed before controller return") from exc
    verification.require_current("Stage 3 controller final verification")
    return Stage3WorkflowResult(
        run_id=run_id,
        stage_root=stage,
        summary=payload.summary,
        manifest=payload.manifest,
    )


def _build_stage3_machine_payload(
    *,
    config: Stage3Config,
    run_id: str,
    source_identity: dict[str, object],
    environment_identity: dict[str, object],
    git_identity: dict[str, object],
    stage2_authority: dict[str, object],
    audit: Stage3CudaAuditBundle,
) -> _Stage3MachinePayload:
    _validate_stage3_audit(config, audit)
    if (
        source_identity.get("schema_version") != "stage3_reviewed_source_set/v1"
        or environment_identity.get("schema_version")
        != "ppo_highres_frontier_stage3_environment/v1"
        or git_identity.get("schema_version")
        != "ppo_highres_frontier_stage3_prospective_git_tree/v1"
        or stage2_authority.get("verified") is not True
        or stage2_authority.get("stage2_commit") != config.stage2_approval_commit
        or stage2_authority.get("authorized_stage") != config.stage_id
    ):
        raise Stage3WorkflowError("Stage 3 execution binding is incomplete")
    config_payload = config.model_dump(mode="json")
    config_payload["run_id"] = run_id
    config_payload["execution_source_identity"] = source_identity
    config_payload["execution_environment_identity"] = environment_identity
    config_payload["execution_git_identity"] = git_identity
    config_payload["stage2_authority"] = stage2_authority
    config_bytes = ArtifactStore.canonical_json_bytes(config_payload)
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    latency = audit.cuda_latency_audit
    microbatch = audit.cuda_microbatch_audit
    acceptance_items = {
        "exact_architecture_contract": audit.network_shape_audit["passed"] is True,
        "fp32_cuda_forward": environment_identity["compute_dtype"] == "float32",
        "amp_disabled": environment_identity["amp_enabled"] is False,
        "outputs_and_gradients_finite": audit.mask_handling_audit[
            "all_outputs_and_gradients_finite"
        ]
        is True,
        "invalid_probability_exact_zero": audit.mask_handling_audit[
            "invalid_probability_exact_zero"
        ]
        is True,
        "invalid_gradient_exact_zero": audit.mask_handling_audit[
            "invalid_frontier_feature_gradient_exact_zero"
        ]
        is True,
        "selected_candidate_valid": audit.mask_handling_audit[
            "selected_candidate_valid"
        ]
        is True,
        "theta_half_open_interval": audit.mask_handling_audit[
            "selected_theta_in_half_open_interval"
        ]
        is True,
        "kappa_bounds": audit.mask_handling_audit["theta_kappa_in_bounds"] is True,
        "deterministic_action_bit_exact": audit.mask_handling_audit[
            "deterministic_action_bit_exact"
        ]
        is True,
        "logprob_factorization_exact": audit.logprob_recompute_audit[
            "factorization_exact"
        ]
        is True,
        "cpu_logprob_tolerance": max(
            audit.logprob_recompute_audit["cpu_max_abs_errors"].values()
        )
        <= config.distribution.cpu_logprob_tolerance,
        "cuda_logprob_tolerance": max(
            audit.logprob_recompute_audit["cuda_max_abs_errors"].values()
        )
        <= config.distribution.cuda_logprob_tolerance,
        "smoke_latency_gate": latency["smoke_p95_ms"]
        <= config.gpu.smoke_batch1_forward_p95_ms,
        "standard_latency_gate": latency["standard_p95_ms"]
        <= config.gpu.standard_batch1_forward_p95_ms,
        "microbatch_32_frozen": microbatch["selected_microbatch_size"]
        == config.gpu.frozen_microbatch_size,
    }
    if len(acceptance_items) != 16 or not all(acceptance_items.values()):
        raise Stage3WorkflowError("Stage 3 machine acceptance failed closed")
    summary: dict[str, object] = {
        "schema_version": "ppo_highres_frontier_stage3_summary/v1",
        "goal_id": config.goal_id,
        "stage_id": config.stage_id,
        "run_id": run_id,
        "state": "machine_passed",
        "config_hash": config_hash,
        "execution_source_set_sha256": source_identity["source_set_sha256"],
        "execution_environment_sha256": _hash_json(environment_identity),
        "prospective_git_tree": git_identity["prospective_git_tree"],
        "stage2_gate_sha256": stage2_authority["gate_sha256"],
        "network_architecture_version": config.architecture.network_architecture_version,
        "network_memory_mode": config.architecture.network_memory_mode,
        "cuda_device": environment_identity["cuda_device_name"],
        "smoke_forward_p95_ms": latency["smoke_p95_ms"],
        "standard_forward_p95_ms": latency["standard_p95_ms"],
        "selected_microbatch_size": microbatch["selected_microbatch_size"],
        "microbatch_warning": microbatch["warning"],
        "acceptance": {
            "policy": "network_forward_action_sampling_acceptance/v1",
            "all_16_stage3_items_passed": True,
            "items": acceptance_items,
        },
        "checkpoint_state": "stage3_no_checkpoint/v1",
        "claim_boundary": "forward_and_action_distribution_mechanics_only/v1",
    }
    if not finite_tree(summary):
        raise Stage3WorkflowError("Stage 3 summary contains non-finite values")
    routing = {
        "schema_version": "ppo_highres_frontier_stage3_routing/v1",
        "run_id": run_id,
        "route": "awaiting_independent_review",
        "authorized_next_stage": config.authorized_next_stage,
        "independent_review_required": True,
        "human_approval_required": True,
        "next_stage_entered": False,
    }
    report = (
        "# Stage 3 machine report\n\n"
        f"- Run ID: `{run_id}`\n"
        f"- CUDA device: `{environment_identity['cuda_device_name']}`\n"
        f"- Smoke batch-1 forward p95: {latency['smoke_p95_ms']:.6f} ms (limit 50 ms).\n"
        f"- Standard batch-1 forward p95: {latency['standard_p95_ms']:.6f} ms (limit 100 ms).\n"
        f"- Frozen safe Standard microbatch: {microbatch['selected_microbatch_size']}.\n"
        "- This stage verifies network forward and action-distribution mechanics only.\n"
        "- It establishes no PPO training result or task-performance advantage.\n"
        "- State: `machine_passed`; route: `awaiting_independent_review`.\n"
        "- No checkpoint, review, approval, gate, executor, or canary artifact was produced.\n"
    ).encode("utf-8")
    metrics = (
        {
            "kind": "network_shape",
            "passed": True,
            "parameter_count": audit.network_shape_audit["parameter_count"],
        },
        {
            "kind": "logprob_recompute",
            "cpu_max_abs_errors": audit.logprob_recompute_audit[
                "cpu_max_abs_errors"
            ],
            "cuda_max_abs_errors": audit.logprob_recompute_audit[
                "cuda_max_abs_errors"
            ],
        },
        {"kind": "mask_handling", **audit.mask_handling_audit},
        {
            "kind": "latency",
            "smoke_p95_ms": latency["smoke_p95_ms"],
            "standard_p95_ms": latency["standard_p95_ms"],
        },
        {
            "kind": "microbatch",
            "selected_microbatch_size": microbatch["selected_microbatch_size"],
            "warning": microbatch["warning"],
            "trials": microbatch["trials"],
        },
    )
    execution_identity = {
        "schema_version": "ppo_highres_frontier_stage3_execution_identity_audit/v1",
        "source": source_identity,
        "environment": environment_identity,
        "git": git_identity,
    }
    artifacts = {
        "config.json": config_bytes,
        "summary.json": ArtifactStore.canonical_json_bytes(summary),
        "routing.json": ArtifactStore.canonical_json_bytes(routing),
        "report.md": report,
        "metrics.jsonl": _canonical_jsonl_bytes(metrics),
        "phase-state.jsonl": _canonical_jsonl_bytes(
            (
                {
                    "state": "machine_passed",
                    "route": "awaiting_independent_review",
                    "run_id": run_id,
                },
            )
        ),
        "evidence/network_shape_audit.json": ArtifactStore.canonical_json_bytes(
            audit.network_shape_audit
        ),
        "evidence/logprob_recompute_audit.json": ArtifactStore.canonical_json_bytes(
            audit.logprob_recompute_audit
        ),
        "evidence/mask_handling_audit.json": ArtifactStore.canonical_json_bytes(
            audit.mask_handling_audit
        ),
        "evidence/cuda_latency_audit.json": ArtifactStore.canonical_json_bytes(
            audit.cuda_latency_audit
        ),
        "evidence/cuda_microbatch_audit.json": ArtifactStore.canonical_json_bytes(
            audit.cuda_microbatch_audit
        ),
        "evidence/sample_policy_outputs.npz": audit.sample_policy_outputs_npz,
        "evidence/stage2_authority_audit.json": ArtifactStore.canonical_json_bytes(
            stage2_authority
        ),
        "evidence/execution_identity_audit.json": ArtifactStore.canonical_json_bytes(
            execution_identity
        ),
    }
    if set(artifacts) != set(STAGE3_MANIFEST_BOUND_ARTIFACTS):
        raise Stage3WorkflowError("Stage 3 machine artifact path set drift")
    manifest = _manifest_for_artifacts(artifacts)
    return _Stage3MachinePayload(
        artifacts=artifacts,
        manifest=manifest,
        summary=summary,
    )


def verify_stage3_machine_run(
    *,
    stage_root: str | Path,
    repo_root: str | Path,
    stage2_gate_path: str | Path,
) -> Stage3MachineVerificationHandle:
    """Verify exact artifacts plus current source/config/environment/Git bindings."""

    stage = Path(stage_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    if stage.name != "s3" or not stage.is_dir():
        raise Stage3WorkflowError("Stage 3 root must be canonical <run>/s3")
    actual_root = {path.name for path in stage.iterdir()}
    if actual_root != set(STAGE3_ROOT_ARTIFACTS):
        raise Stage3WorkflowError("Stage 3 root artifact set drift")
    if any((stage / name).exists() for name in ("review.json", "approval.json", "gate.json")):
        raise Stage3WorkflowError("Stage 3 machine root contains forbidden authority artifact")
    evidence = stage / "evidence"
    expected_evidence_names = {
        Path(relative).name for relative in STAGE3_EVIDENCE_ARTIFACTS
    }
    if not evidence.is_dir() or {path.name for path in evidence.iterdir()} != expected_evidence_names:
        raise Stage3WorkflowError("Stage 3 evidence artifact set drift")

    manifest_snapshot = FrozenFileSnapshot.capture(
        stage / "manifest.json",
        "Stage 3 manifest",
        authority_root=stage,
    )
    manifest = strict_json_object_from_bytes(
        manifest_snapshot.payload,
        "Stage 3 manifest",
    )
    _require_canonical_json(manifest_snapshot, manifest, "Stage 3 manifest")
    artifact_snapshots = _capture_stage3_artifact_snapshots(
        stage=stage,
        manifest=manifest,
        expected_paths=set(STAGE3_MANIFEST_BOUND_ARTIFACTS),
    )
    config_snapshot = artifact_snapshots["config.json"]
    runtime_config = strict_json_object_from_bytes(
        config_snapshot.payload,
        "Stage 3 config",
    )
    source_identity = runtime_config.pop("execution_source_identity", None)
    environment_identity = runtime_config.pop("execution_environment_identity", None)
    git_identity = runtime_config.pop("execution_git_identity", None)
    stage2_authority = runtime_config.pop("stage2_authority", None)
    try:
        machine_config = Stage3Config.model_validate(runtime_config)
        repository_config = load_stage3_config(
            repo / "configs/ppo_highres_frontier_stage3_v1.json"
        )
    except Exception as exc:
        raise Stage3WorkflowError("Stage 3 config binding drift") from exc
    if (
        machine_config.model_dump(exclude={"run_id"})
        != repository_config.model_dump(exclude={"run_id"})
        or machine_config.run_id != stage.parent.name
    ):
        raise Stage3WorkflowError("Stage 3 config binding drift")
    current_source = stage3_source_identity(repo)
    current_environment = stage3_environment_identity()
    current_git = stage3_git_identity(
        repo,
        base_commit=machine_config.stage2_approval_commit,
    )
    current_authority = verify_frozen_stage2_authority(
        gate_path=stage2_gate_path,
        repo_root=repo,
        expected_stage2_commit=machine_config.stage2_approval_commit,
    )
    if (
        source_identity != current_source
        or environment_identity != current_environment
        or git_identity != current_git
        or stage2_authority != current_authority.identity
    ):
        raise Stage3WorkflowError("Stage 3 execution source/environment/Git/authority drift")
    audit = Stage3CudaAuditBundle(
        network_shape_audit=_strict_json_snapshot(
            artifact_snapshots["evidence/network_shape_audit.json"],
            "Stage 3 network shape audit",
        ),
        logprob_recompute_audit=_strict_json_snapshot(
            artifact_snapshots["evidence/logprob_recompute_audit.json"],
            "Stage 3 logprob audit",
        ),
        mask_handling_audit=_strict_json_snapshot(
            artifact_snapshots["evidence/mask_handling_audit.json"],
            "Stage 3 mask audit",
        ),
        cuda_latency_audit=_strict_json_snapshot(
            artifact_snapshots["evidence/cuda_latency_audit.json"],
            "Stage 3 latency audit",
        ),
        cuda_microbatch_audit=_strict_json_snapshot(
            artifact_snapshots["evidence/cuda_microbatch_audit.json"],
            "Stage 3 microbatch audit",
        ),
        sample_policy_outputs_npz=artifact_snapshots[
            "evidence/sample_policy_outputs.npz"
        ].payload,
    )
    if _strict_json_snapshot(
        artifact_snapshots["evidence/stage2_authority_audit.json"],
        "Stage 3 Stage 2 authority audit",
    ) != stage2_authority:
        raise Stage3WorkflowError("Stage 3 recorded Stage 2 authority drift")
    expected_execution = {
        "schema_version": "ppo_highres_frontier_stage3_execution_identity_audit/v1",
        "source": source_identity,
        "environment": environment_identity,
        "git": git_identity,
    }
    if _strict_json_snapshot(
        artifact_snapshots["evidence/execution_identity_audit.json"],
        "Stage 3 execution identity audit",
    ) != expected_execution:
        raise Stage3WorkflowError("Stage 3 execution identity evidence drift")
    expected = _build_stage3_machine_payload(
        config=machine_config,
        run_id=stage.parent.name,
        source_identity=source_identity,
        environment_identity=environment_identity,
        git_identity=git_identity,
        stage2_authority=stage2_authority,
        audit=audit,
    )
    for relative in STAGE3_MANIFEST_BOUND_ARTIFACTS:
        if artifact_snapshots[relative].payload != expected.artifacts[relative]:
            raise Stage3WorkflowError(f"Stage 3 deterministic artifact drift: {relative}")
    if manifest_snapshot.payload != ArtifactStore.canonical_json_bytes(expected.manifest):
        raise Stage3WorkflowError("Stage 3 manifest semantic drift")
    if (
        stage3_source_identity(repo) != current_source
        or stage3_environment_identity() != current_environment
        or stage3_git_identity(
            repo,
            base_commit=machine_config.stage2_approval_commit,
        )
        != current_git
    ):
        raise Stage3WorkflowError("Stage 3 execution identity changed during verification")
    verification = Stage3MachineVerificationHandle(
        stage_root=stage,
        manifest_snapshot=manifest_snapshot,
        artifacts=tuple(sorted(artifact_snapshots.items())),
        stage2_authority=current_authority,
    )
    verification.require_current()
    return verification


def _validate_stage3_audit(
    config: Stage3Config,
    audit: Stage3CudaAuditBundle,
) -> None:
    trees = (
        audit.network_shape_audit,
        audit.logprob_recompute_audit,
        audit.mask_handling_audit,
        audit.cuda_latency_audit,
        audit.cuda_microbatch_audit,
    )
    if any(not isinstance(tree, dict) or not finite_tree(tree) for tree in trees):
        raise Stage3WorkflowError("Stage 3 audit tree is invalid or non-finite")
    network = audit.network_shape_audit
    if (
        network.get("schema_version")
        != "ppo_highres_frontier_stage3_network_shape_audit/v1"
        or network.get("passed") is not True
        or network.get("all_parameters_fp32") is not True
        or any(network.get("forbidden_modules_present", {}).values())
    ):
        raise Stage3WorkflowError("Stage 3 network shape evidence failed")
    metadata = network.get("metadata")
    if not isinstance(metadata, dict):
        raise Stage3WorkflowError("Stage 3 network metadata is missing")
    for key, expected in (
        ("network_architecture_version", config.architecture.network_architecture_version),
        ("network_memory_mode", config.architecture.network_memory_mode),
        ("context_token_count", config.architecture.context_token_count),
        ("token_dim", config.architecture.token_dim),
        ("cross_attention_layers", config.architecture.cross_attention_layers),
        ("attention_heads", config.architecture.attention_heads),
        ("ffn_hidden_dim", config.architecture.ffn_hidden_dim),
        ("action_hidden_dim", config.architecture.action_hidden_dim),
        ("dropout", config.architecture.dropout),
    ):
        if metadata.get(key) != expected:
            raise Stage3WorkflowError("Stage 3 network metadata drift")
    logprob = audit.logprob_recompute_audit
    if (
        logprob.get("passed") is not True
        or logprob.get("factorization_exact") is not True
        or logprob.get("cpu_tolerance") != config.distribution.cpu_logprob_tolerance
        or logprob.get("cuda_tolerance") != config.distribution.cuda_logprob_tolerance
    ):
        raise Stage3WorkflowError("Stage 3 logprob evidence failed")
    for key, tolerance in (
        ("cpu_max_abs_errors", config.distribution.cpu_logprob_tolerance),
        ("cuda_max_abs_errors", config.distribution.cuda_logprob_tolerance),
    ):
        errors = logprob.get(key)
        if (
            not isinstance(errors, dict)
            or set(errors) != {"log_prob_frontier", "log_prob_theta", "log_prob_total"}
            or max(errors.values()) > tolerance
        ):
            raise Stage3WorkflowError("Stage 3 logprob tolerance evidence failed")
    mask = audit.mask_handling_audit
    required_mask_truths = (
        "passed",
        "invalid_probability_exact_zero",
        "invalid_frontier_feature_gradient_exact_zero",
        "deterministic_action_bit_exact",
        "selected_candidate_valid",
        "selected_theta_in_half_open_interval",
        "theta_kappa_in_bounds",
        "all_outputs_and_gradients_finite",
    )
    if any(mask.get(key) is not True for key in required_mask_truths):
        raise Stage3WorkflowError("Stage 3 mask evidence failed")
    _validate_stage3_latency_audit(config, audit.cuda_latency_audit)
    microbatch = audit.cuda_microbatch_audit
    raw_trials = microbatch.get("trials")
    if not isinstance(raw_trials, list):
        raise Stage3WorkflowError("Stage 3 microbatch trials are missing")
    try:
        trials = tuple(MicrobatchTrial(**trial) for trial in raw_trials)
        result = evaluate_microbatch_gate(
            trials,
            candidates=config.gpu.microbatch_candidates,
            warning_gib=config.gpu.memory_warning_gib,
            hard_stop_gib=config.gpu.memory_hard_stop_gib,
        )
    except (TypeError, Stage3GpuGateError) as exc:
        raise Stage3WorkflowError("Stage 3 microbatch evidence failed") from exc
    if (
        microbatch.get("amp_enabled") is not False
        or microbatch.get("compute_dtype") != "float32"
        or microbatch.get("representative_backward") is not True
        or microbatch.get("frozen_microbatch_size") != config.gpu.frozen_microbatch_size
        or microbatch.get("selected_microbatch_size")
        != config.gpu.frozen_microbatch_size
        or result.to_dict()
        != {
            key: microbatch[key]
            for key in (
                "selected_microbatch_size",
                "warning",
                "hard_gate_passed",
                "attempted_sizes",
                "trials",
            )
        }
    ):
        raise Stage3WorkflowError("Stage 3 frozen microbatch evidence failed")
    _validate_sample_outputs(audit.sample_policy_outputs_npz)


def _validate_stage3_latency_audit(
    config: Stage3Config,
    latency: dict[str, object],
) -> None:
    expected_fields = {
        "schema_version",
        "device",
        "synchronized_cuda_events",
        "batch_size",
        "warmup_iterations",
        "measure_iterations",
        "smoke_samples_ms",
        "standard_samples_ms",
        "smoke_peak_memory_gib",
        "standard_peak_memory_gib",
        "smoke_p95_ms",
        "standard_p95_ms",
        "smoke_limit_ms",
        "standard_limit_ms",
        "smoke_sample_count",
        "standard_sample_count",
        "gate_passed",
    }
    if set(latency) != expected_fields:
        raise Stage3WorkflowError("Stage 3 latency evidence schema failed")
    if (
        latency["schema_version"]
        != "ppo_highres_frontier_stage3_cuda_latency_audit/v1"
        or not isinstance(latency["device"], str)
        or not latency["device"]
        or latency["synchronized_cuda_events"] is not True
        or type(latency["batch_size"]) is not int
        or latency["batch_size"] != 1
        or type(latency["warmup_iterations"]) is not int
        or latency["warmup_iterations"] != config.gpu.timing_warmup_iterations
        or type(latency["measure_iterations"]) is not int
        or latency["measure_iterations"] != config.gpu.timing_measure_iterations
        or type(latency["gate_passed"]) is not bool
    ):
        raise Stage3WorkflowError("Stage 3 latency evidence schema failed")

    measure_iterations = config.gpu.timing_measure_iterations
    for key in ("smoke_samples_ms", "standard_samples_ms"):
        samples = latency[key]
        if (
            not isinstance(samples, list)
            or len(samples) != measure_iterations
            or any(
                type(sample) is not float
                or not math.isfinite(sample)
                or sample < 0.0
                for sample in samples
            )
        ):
            raise Stage3WorkflowError("Stage 3 latency sample evidence failed")

    for key in (
        "smoke_peak_memory_gib",
        "standard_peak_memory_gib",
        "smoke_p95_ms",
        "standard_p95_ms",
        "smoke_limit_ms",
        "standard_limit_ms",
    ):
        value = latency[key]
        if type(value) is not float or not math.isfinite(value) or value < 0.0:
            raise Stage3WorkflowError("Stage 3 latency numeric evidence failed")
    for key in ("smoke_sample_count", "standard_sample_count"):
        if type(latency[key]) is not int:
            raise Stage3WorkflowError("Stage 3 latency count evidence failed")

    try:
        recomputed = evaluate_latency_gate(
            smoke_samples_ms=latency["smoke_samples_ms"],
            standard_samples_ms=latency["standard_samples_ms"],
            smoke_limit_ms=config.gpu.smoke_batch1_forward_p95_ms,
            standard_limit_ms=config.gpu.standard_batch1_forward_p95_ms,
        )
    except Stage3GpuGateError as exc:
        raise Stage3WorkflowError("Stage 3 latency evidence failed") from exc
    recorded_gate = {
        key: latency[key]
        for key in (
            "smoke_p95_ms",
            "standard_p95_ms",
            "smoke_limit_ms",
            "standard_limit_ms",
            "smoke_sample_count",
            "standard_sample_count",
            "gate_passed",
        )
    }
    if recorded_gate != recomputed.to_dict():
        raise Stage3WorkflowError("Stage 3 latency gate recomputation drift")
    if any(
        latency[key] > config.gpu.memory_hard_stop_gib
        for key in ("smoke_peak_memory_gib", "standard_peak_memory_gib")
    ):
        raise Stage3WorkflowError("Stage 3 latency memory hard gate failed")


def _validate_sample_outputs(payload: bytes) -> None:
    expected = {
        "schema_version",
        "candidate_mask",
        "frontier_logits",
        "theta_mu",
        "theta_kappa",
        "value",
        "selected_frontier_index",
        "selected_theta",
        "log_prob_frontier",
        "log_prob_theta",
        "log_prob_total",
    }
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != expected:
                raise Stage3WorkflowError("Stage 3 sample output member set drift")
            if str(archive["schema_version"].item()) != (
                "ppo_highres_frontier_stage3_sample_policy_outputs/v1"
            ):
                raise Stage3WorkflowError("Stage 3 sample output schema drift")
            mask = np.asarray(archive["candidate_mask"])
            logits = np.asarray(archive["frontier_logits"])
            theta_mu = np.asarray(archive["theta_mu"])
            theta_kappa = np.asarray(archive["theta_kappa"])
            value = np.asarray(archive["value"])
            index = np.asarray(archive["selected_frontier_index"])
            selected_theta = np.asarray(archive["selected_theta"])
            log_frontier = np.asarray(archive["log_prob_frontier"])
            log_theta = np.asarray(archive["log_prob_theta"])
            log_total = np.asarray(archive["log_prob_total"])
    except Stage3WorkflowError:
        raise
    except Exception as exc:
        raise Stage3WorkflowError("Stage 3 sample output NPZ is invalid") from exc
    if (
        mask.dtype != np.bool_
        or logits.shape != mask.shape
        or theta_mu.shape != mask.shape
        or theta_kappa.shape != mask.shape
        or value.shape != (mask.shape[0],)
        or index.shape != (mask.shape[0],)
        or selected_theta.shape != (mask.shape[0],)
        or any(array.shape != (mask.shape[0],) for array in (log_frontier, log_theta, log_total))
        or any(
            array.dtype != np.float32 or not np.isfinite(array).all()
            for array in (
                logits,
                theta_mu,
                theta_kappa,
                value,
                selected_theta,
                log_frontier,
                log_theta,
                log_total,
            )
        )
        or index.dtype != np.int64
        or np.any(index < 0)
        or np.any(index >= mask.shape[1])
        or not np.all(mask[np.arange(mask.shape[0]), index])
        or np.any(selected_theta < -np.pi)
        or np.any(selected_theta >= np.pi)
        or np.any(theta_kappa < 1e-3)
        or np.any(theta_kappa > 20.0)
        or not np.array_equal(log_total, log_frontier + log_theta)
    ):
        raise Stage3WorkflowError("Stage 3 sample output numeric contract failed")


def _manifest_for_artifacts(artifacts: dict[str, bytes]) -> dict[str, object]:
    return {
        "schema_version": "sha256_manifest/v1",
        "artifacts": [
            {
                "path": path,
                "sha256": hashlib.sha256(artifacts[path]).hexdigest(),
                "size_bytes": len(artifacts[path]),
            }
            for path in sorted(artifacts)
        ],
    }


def _canonical_jsonl_bytes(values: tuple[dict[str, object], ...]) -> bytes:
    return b"".join(
        (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for value in values
    )


def _capture_stage3_artifact_snapshots(
    *,
    stage: Path,
    manifest: dict[str, object],
    expected_paths: set[str],
) -> dict[str, FrozenFileSnapshot]:
    return _capture_manifest_artifact_snapshots(
        stage=stage,
        manifest=manifest,
        expected_paths=expected_paths,
        label="Stage 3",
    )


def _capture_manifest_artifact_snapshots(
    *,
    stage: Path,
    manifest: dict[str, object],
    expected_paths: set[str],
    label: str,
) -> dict[str, FrozenFileSnapshot]:
    if set(manifest) != {"schema_version", "artifacts"} or manifest.get(
        "schema_version"
    ) != "sha256_manifest/v1":
        raise Stage3WorkflowError(f"{label} manifest schema drift")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        raise Stage3WorkflowError(f"{label} manifest entries are invalid")
    paths = [entry.get("path") for entry in entries if isinstance(entry, dict)]
    if paths != sorted(expected_paths) or len(paths) != len(entries):
        raise Stage3WorkflowError(f"{label} manifest path set drift")
    snapshots: dict[str, FrozenFileSnapshot] = {}
    try:
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {
                "path",
                "sha256",
                "size_bytes",
            }:
                raise Stage3WorkflowError(f"{label} manifest entry schema drift")
            relative_text = entry["path"]
            if not isinstance(relative_text, str):
                raise Stage3WorkflowError(f"{label} manifest path type drift")
            relative = Path(relative_text)
            if relative.is_absolute() or ".." in relative.parts:
                raise Stage3WorkflowError(f"{label} manifest path is unsafe")
            snapshot = FrozenFileSnapshot.capture(
                stage / relative,
                f"{label} artifact {relative_text}",
                authority_root=stage,
            )
            if (
                entry["size_bytes"] != snapshot.size_bytes
                or entry["sha256"] != snapshot.sha256
            ):
                raise Stage3WorkflowError(f"{label} artifact drift: {relative_text}")
            snapshots[relative_text] = snapshot
    except Stage1WorkflowError as exc:
        raise Stage3WorkflowError(f"{label} artifact snapshot is invalid") from exc
    return snapshots


def _strict_json_snapshot(
    snapshot: FrozenFileSnapshot,
    label: str,
) -> dict[str, object]:
    try:
        value = strict_json_object_from_bytes(snapshot.payload, label)
    except Stage1WorkflowError as exc:
        raise Stage3WorkflowError(f"{label} is invalid") from exc
    _require_canonical_json(snapshot, value, label)
    return value


def _require_canonical_json(
    snapshot: FrozenFileSnapshot,
    value: dict[str, object],
    label: str,
) -> None:
    if snapshot.payload != ArtifactStore.canonical_json_bytes(value):
        raise Stage3WorkflowError(f"{label} is not canonical JSON")


def _hash_json(value: object) -> str:
    return hashlib.sha256(ArtifactStore.canonical_json_bytes(value)).hexdigest()


def _temporary_index_path(label: str) -> Path:
    root = Path("D:/xunce/tmp/ppo_frontier/git-index").resolve()
    if root.drive.upper() != "D:":
        raise Stage3WorkflowError("Stage 3 temporary Git index must be on D drive")
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{label}-{uuid.uuid4().hex}.index"


def _remove_temporary_index(index_path: Path) -> None:
    lock_path = Path(str(index_path) + ".lock")
    if lock_path.is_file():
        lock_path.unlink()
    if index_path.is_file():
        index_path.unlink()


def _require_empty_real_index(repo: Path) -> None:
    completed = subprocess.run(
        ["git", "-C", str(repo), "diff", "--cached", "--quiet"],
        check=False,
    )
    if completed.returncode != 0:
        raise Stage3WorkflowError("real Git index is not empty")


def _git_text(
    repo: Path,
    arguments: list[str],
    label: str,
    environment: dict[str, str] | None = None,
    *,
    input_payload: bytes | None = None,
) -> str:
    return _git_bytes(
        repo,
        arguments,
        label,
        environment,
        input_payload=input_payload,
    ).decode("utf-8", errors="strict")


def _git_bytes(
    repo: Path,
    arguments: list[str],
    label: str,
    environment: dict[str, str] | None = None,
    *,
    input_payload: bytes | None = None,
) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        input=input_payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise Stage3WorkflowError(f"Stage 3 Git {label} failed: {detail}")
    return completed.stdout


__all__ = [
    "STAGE3_EVIDENCE_ARTIFACTS",
    "STAGE3_MANIFEST_BOUND_ARTIFACTS",
    "STAGE3_MACHINE_ARTIFACTS",
    "STAGE3_ROOT_ARTIFACTS",
    "Stage3WorkflowError",
    "Stage3WorkflowResult",
    "run_stage3_workflow",
    "stage3_environment_identity",
    "stage3_git_identity",
    "stage3_source_identity",
    "verify_frozen_git_source_identity",
    "verify_frozen_stage2_authority",
    "verify_stage3_machine_run",
]
