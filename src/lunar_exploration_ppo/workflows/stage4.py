"""Stage 4 machine workflow primitives and read-only Stage 3 authority consumer."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
import torch

from lunar_exploration_ppo.configs.stage1 import Stage1Config
from lunar_exploration_ppo.configs.stage3 import Stage3Config
from lunar_exploration_ppo.configs.stage4 import Stage4Config, load_stage4_config
from lunar_exploration_ppo.policy.cross_attention import CrossAttentionFrontierPolicy
from lunar_exploration_ppo.ppo.checkpoint import CheckpointManager
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.workflows import stage2 as stage2_workflow
from lunar_exploration_ppo.workflows import stage3 as stage3_workflow
from lunar_exploration_ppo.workflows.stage1_artifacts import (
    FrozenFileSnapshot,
    Stage1WorkflowError,
    finite_tree,
    prepare_unique_run_root,
    strict_json_object_from_bytes,
    validate_run_id,
)
from lunar_exploration_ppo.workflows.stage4_machine import (
    Stage4MachineAuditBundle,
    Stage4MachineError,
    deterministic_action_record,
    load_action_fixture_npz,
    run_stage4_smoke_acceptance,
)


STAGE3_COMMIT: Final = "4df7be92cd6517e77c648e890bf7d32c9fcd560b"
STAGE3_COMMIT_PARENT: Final = "898911559ccc9ae8ef701b69a58a93b1d8a4d8d8"
STAGE3_COMMIT_TREE: Final = "689876d5031eac53290397c534be3a155d017b10"
STAGE3_COMMIT_SUBJECT: Final = "feat: add cross-attention frontier policy"
STAGE3_GATE_SHA256: Final = (
    "6ea3dc5bc199dd9370fab4b1a5aac1ebab08d9e42f49c0d01b3aa2b1e9aae0f0"
)
STAGE3_APPROVAL_SHA256: Final = (
    "8e901e04d587110de681da302ad24025dfebed945ff646ffa1e85bc5f5ecb3f2"
)
STAGE3_REVIEW_SHA256: Final = (
    "62bf0869ba66f22bfec07464b7d17478d27fa618c3ae645039c98333262b2afc"
)
STAGE3_MANIFEST_SHA256: Final = (
    "2715e790d843ac9eb873b9123a569f5254e603a734f86aa9629ac8abce142a90"
)
STAGE3_CONFIG_SHA256: Final = (
    "89b1a4b860d6732143b3ff7ddca7f9b02573e268a1454f709ab3957602998947"
)
STAGE3_SOURCE_SET_SHA256: Final = (
    "26ef6026b40107347ea35b6d2545f47fee1297f7a181a5d4f517183609d9759c"
)
STAGE3_ENVIRONMENT_SHA256: Final = (
    "dac42c3bc7d28e4acb78264ab7b5955c89f5716b7a841eae93a4658704bdd7c0"
)
STAGE2_GATE_SHA256: Final = (
    "997203efa6bedd0d0e7d124f8b2362568983c1eac48a4b9e588c80d41b33dcb2"
)
ABSENT_AUTHORITY_SHA256: Final = (
    "3057252298292ae9a47b0879812581277456d1e9d9c0995178967da7c929236d"
)
STAGE3_RUN_ID: Final = "s3-task4-cifix-final-20260713T213739Z"
STAGE3_STAGE_ID: Final = "ppo_highres_frontier_stage3_cross_attention_policy/v1"
STAGE4_STAGE_ID: Final = "ppo_highres_frontier_stage4_rollout_ppo_update/v1"
GOAL_ID: Final = "ppo-highres-frontier-map-exploration"
STAGE3_REVIEWED_PATHS: Final = (
    ".github/workflows/platform-compatibility.yml",
    "configs/ppo_highres_frontier_stage3_v1.json",
    "docs/ppo-highres-frontier-stage3.md",
    "pyproject.toml",
    "scripts/run_ppo_highres_frontier_stage3.py",
    "src/lunar_exploration_ppo/configs/stage3.py",
    "src/lunar_exploration_ppo/policy/__init__.py",
    "src/lunar_exploration_ppo/policy/cross_attention.py",
    "src/lunar_exploration_ppo/workflows/stage3.py",
    "src/lunar_exploration_ppo/workflows/stage3_gpu.py",
    "tests/ppo_highres_frontier/test_stage2_workflow.py",
    "tests/ppo_highres_frontier/test_stage3_policy_cpu.py",
    "tests/ppo_highres_frontier/test_stage3_policy_cuda.py",
    "tests/ppo_highres_frontier/test_stage3_workflow.py",
)
STAGE3_REVIEWED_PATH_SET_SHA256: Final = (
    "dc75183d3be7993893fdbc4b169b00e3d24d27986a2c8e85442c2a8ba6a0754d"
)
STAGE4_ROOT_ARTIFACTS: Final = (
    "config.json",
    "summary.json",
    "routing.json",
    "manifest.json",
    "report.md",
    "metrics.jsonl",
    "phase-state.jsonl",
    "training_progress.jsonl",
    "evidence",
    "checkpoints",
)
STAGE4_MACHINE_ARTIFACTS: Final = (
    "config.json",
    "summary.json",
    "routing.json",
    "report.md",
    "metrics.jsonl",
    "phase-state.jsonl",
    "training_progress.jsonl",
)
STAGE4_EVIDENCE_ARTIFACTS: Final = (
    "evidence/smoke_update_audit.json",
    "evidence/rollout_snapshot_audit.json",
    "evidence/gae_audit.json",
    "evidence/ppo_loss_audit.json",
    "evidence/logprob_recompute_audit.json",
    "evidence/optimizer_step_audit.json",
    "evidence/checkpoint_load_audit.json",
    "evidence/training_resume_audit.json",
    "evidence/checkpoint_action_fixture.npz",
    "evidence/tiny_overfit_audit.json",
    "evidence/stage3_authority_audit.json",
    "evidence/execution_identity_audit.json",
)
STAGE4_CHECKPOINT_ARTIFACTS: Final = (
    "checkpoints/latest.json",
    *tuple(
        f"checkpoints/update-{step:08d}/{name}"
        for step in range(1, 4)
        for name in ("checkpoint.pt", "manifest.json", "complete.json")
    ),
)
STAGE4_MANIFEST_BOUND_ARTIFACTS: Final = (
    *STAGE4_MACHINE_ARTIFACTS,
    *STAGE4_EVIDENCE_ARTIFACTS,
    *STAGE4_CHECKPOINT_ARTIFACTS,
)
_STAGE4_UPDATE_AUDIT_KEYS: Final = frozenset(
    {
        "schema_version",
        "update_step",
        "config_sha256",
        "collection",
        "gae",
        "ppo",
        "buffer",
        "checkpoint",
    }
)
_STAGE4_COLLECTION_AUDIT_KEYS: Final = frozenset(
    {
        "trainable_transition_count",
        "per_env_trainable_counts",
        "diagnostic_reset_counts",
        "worker_pids",
        "worker_start_methods",
        "inference_pids",
        "inference_main_process_only",
        "inference_batch_count",
        "policy_device",
        "policy_state_sha256",
        "snapshot_count",
        "unique_snapshot_count",
        "terminal_transition_count",
        "old_logprob_factorization_exact",
    }
)
_STAGE3_ROOT_ARTIFACTS = frozenset(
    {
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
)
_REVIEW_KEYS = frozenset(
    {
        "schema_version",
        "state",
        "goal_id",
        "stage_id",
        "run_id",
        "reviewer_id",
        "reviewer_agent_id",
        "spec_verdict",
        "quality_verdict",
        "issue_counts",
        "final_conclusion",
        "reviewed_base_commit",
        "reviewed_prospective_git_tree",
        "reviewed_paths",
        "reviewed_path_set_sha256",
        "source_set_sha256",
        "config_sha256",
        "environment_sha256",
        "summary_sha256",
        "routing_sha256",
        "manifest_sha256",
        "stage2_gate_sha256",
        "checkpoint_state",
        "authorized_next_stage",
        "review_report",
        "review_package",
        "review_bindings",
        "review_controller",
        "review_recorder",
        "implementation_report",
        "real_git_index",
        "review_recorded_at_utc",
        "approval_challenge",
    }
)
_APPROVAL_KEYS = frozenset(
    {
        "actor",
        "approval_challenge",
        "approval_text",
        "approval_text_utf8_sha256",
        "authority_kind",
        "bindings",
        "commit_sha256",
        "controller_event_binding_sha256",
        "controller_event_id",
        "controller_observed_at_utc",
        "event_authenticity_source",
        "goal_id",
        "local_verification_scope",
        "review_sha256",
        "reviewed_prospective_git_tree",
        "run_id",
        "schema_version",
        "stage_id",
        "state",
        "thread_id",
    }
)
_CONTEXT_KEYS = frozenset(
    {
        "approval_challenge",
        "approval_controller",
        "approval_sha256",
        "authorized_next_stage",
        "checkpoint_state",
        "commit_parent",
        "commit_sha256",
        "commit_subject",
        "commit_tree",
        "config_sha256",
        "environment_sha256",
        "goal_id",
        "manifest_sha256",
        "review_bindings",
        "review_bindings_schema_version",
        "review_controller",
        "review_package",
        "review_recorded_at_utc",
        "review_recorder",
        "review_report",
        "review_sha256",
        "reviewed_base_commit",
        "reviewed_path_set_sha256",
        "reviewed_paths",
        "reviewed_prospective_git_tree",
        "routing_sha256",
        "run_id",
        "schema_version",
        "source_set_sha256",
        "stage2_gate_sha256",
        "stage_id",
        "summary_sha256",
        "thread_id",
    }
)


class Stage4WorkflowError(RuntimeError):
    """Stage 4 machine or inherited authority verification failed."""


@dataclass(frozen=True, slots=True)
class Stage4RuntimeBinding:
    payload: dict[str, object]
    config_bytes: bytes
    config_sha256: str


@dataclass(frozen=True, slots=True)
class Stage4WorkflowResult:
    run_id: str
    stage_root: Path
    summary: dict[str, object]
    manifest: dict[str, object]


@dataclass(frozen=True, slots=True)
class _Stage4MachinePayload:
    artifacts: dict[str, bytes]
    manifest: dict[str, object]
    summary: dict[str, object]


@dataclass(frozen=True, slots=True)
class FrozenStage3AuthorityHandle:
    identity: dict[str, object]
    snapshots: tuple[tuple[str, FrozenFileSnapshot], ...]
    stage2_authority: object | None
    stage_root: Path | None = None
    root_artifacts: frozenset[str] | None = None

    def require_current(self, label: str = "Stage 3 authority") -> None:
        try:
            if self.stage_root is not None and self.root_artifacts is not None:
                if not self.stage_root.is_dir() or {
                    path.name for path in self.stage_root.iterdir()
                } != self.root_artifacts:
                    raise Stage4WorkflowError(f"{label} artifact set changed")
            for name, snapshot in self.snapshots:
                snapshot.require_current(f"{label}/{name}")
            if self.stage2_authority is not None:
                self.stage2_authority.require_current(f"{label}/Stage 2 pass-through")
        except Stage4WorkflowError:
            raise
        except (Stage1WorkflowError, OSError) as exc:
            raise Stage4WorkflowError(f"{label} changed") from exc


@dataclass(frozen=True, slots=True)
class Stage4MachineVerificationHandle:
    """Frozen Stage 4 artifacts plus transitive read-only Stage 3 authority."""

    stage_root: Path
    manifest_snapshot: FrozenFileSnapshot
    artifacts: tuple[tuple[str, FrozenFileSnapshot], ...]
    stage3_authority: FrozenStage3AuthorityHandle

    def artifact_snapshot(self, relative: str) -> FrozenFileSnapshot:
        for path, snapshot in self.artifacts:
            if path == relative:
                return snapshot
        raise Stage4WorkflowError(f"Stage 4 verified artifact is missing: {relative}")

    def require_current(self, label: str = "Stage 4 machine verification") -> None:
        try:
            self.stage3_authority.require_current(f"{label} Stage 3 authority")
            self.manifest_snapshot.require_current(f"{label} manifest")
            for relative, snapshot in self.artifacts:
                snapshot.require_current(f"{label} artifact {relative}")
        except (Stage1WorkflowError, OSError) as exc:
            raise Stage4WorkflowError(f"{label} artifacts changed") from exc
        if (
            not self.stage_root.is_dir()
            or {path.name for path in self.stage_root.iterdir()}
            != set(STAGE4_ROOT_ARTIFACTS)
        ):
            raise Stage4WorkflowError(f"{label} root artifact set changed")
        evidence = self.stage_root / "evidence"
        if (
            not evidence.is_dir()
            or {path.name for path in evidence.iterdir()}
            != {Path(path).name for path in STAGE4_EVIDENCE_ARTIFACTS}
        ):
            raise Stage4WorkflowError(f"{label} evidence artifact set changed")
        if _checkpoint_relative_files(self.stage_root) != set(
            STAGE4_CHECKPOINT_ARTIFACTS
        ):
            raise Stage4WorkflowError(f"{label} checkpoint artifact set changed")


def verify_frozen_stage3_authority(
    *,
    gate_path: str | Path,
    repo_root: str | Path,
) -> FrozenStage3AuthorityHandle:
    """Strictly consume the externally persisted Stage 3 graph without issuing it."""

    repo = Path(repo_root).expanduser().resolve()
    gate_file = Path(gate_path).expanduser().resolve()
    stage = gate_file.parent
    if (
        gate_file.name != "gate.json"
        or stage.name != "s3"
        or stage.parent.name != STAGE3_RUN_ID
        or not stage.is_dir()
    ):
        raise Stage4WorkflowError(
            "Stage 3 authority path must be the frozen <run>/s3/gate.json"
        )
    if {path.name for path in stage.iterdir()} != _STAGE3_ROOT_ARTIFACTS:
        raise Stage4WorkflowError("Stage 3 authority artifact set drift")

    try:
        root_snapshots = {
            name: FrozenFileSnapshot.capture(
                stage / name,
                f"Stage 3 {name}",
                authority_root=stage,
            )
            for name in (
                "gate.json",
                "approval.json",
                "review.json",
                "manifest.json",
            )
        }
        gate = _strict_canonical(root_snapshots["gate.json"], "Stage 3 gate")
        approval = _strict_canonical(
            root_snapshots["approval.json"],
            "Stage 3 approval",
        )
        review = _strict_canonical(
            root_snapshots["review.json"],
            "Stage 3 review",
        )
        manifest = _strict_canonical(
            root_snapshots["manifest.json"],
            "Stage 3 manifest",
        )
    except Stage1WorkflowError as exc:
        raise Stage4WorkflowError("Stage 3 authority snapshot is invalid") from exc
    fixed_hashes = {
        "gate.json": STAGE3_GATE_SHA256,
        "approval.json": STAGE3_APPROVAL_SHA256,
        "review.json": STAGE3_REVIEW_SHA256,
        "manifest.json": STAGE3_MANIFEST_SHA256,
    }
    if any(
        root_snapshots[name].sha256 != expected
        for name, expected in fixed_hashes.items()
    ):
        raise Stage4WorkflowError("Stage 3 fixed authority SHA-256 drift")

    try:
        artifact_snapshots = stage3_workflow._capture_manifest_artifact_snapshots(
            stage=stage,
            manifest=manifest,
            expected_paths=set(stage3_workflow.STAGE3_MANIFEST_BOUND_ARTIFACTS),
            label="Stage 3",
        )
    except Exception as exc:
        raise Stage4WorkflowError("Stage 3 manifest graph drift") from exc
    config_snapshot = artifact_snapshots["config.json"]
    if config_snapshot.sha256 != STAGE3_CONFIG_SHA256:
        raise Stage4WorkflowError("Stage 3 machine config hash drift")
    machine_payload = _strict_canonical(config_snapshot, "Stage 3 config")
    recorded_source = machine_payload.pop("execution_source_identity", None)
    recorded_environment = machine_payload.pop("execution_environment_identity", None)
    recorded_git = machine_payload.pop("execution_git_identity", None)
    recorded_stage2 = machine_payload.pop("stage2_authority", None)
    try:
        machine_config = Stage3Config.model_validate(machine_payload)
        frozen_config = Stage3Config.model_validate_json(
            _git_bytes(
                repo,
                ["show", f"{STAGE3_COMMIT}:configs/ppo_highres_frontier_stage3_v1.json"],
            ).decode("utf-8", errors="strict")
        )
    except Exception as exc:
        raise Stage4WorkflowError("Stage 3 frozen config binding drift") from exc
    if (
        machine_config.model_dump(exclude={"run_id"})
        != frozen_config.model_dump(exclude={"run_id"})
        or machine_config.run_id != STAGE3_RUN_ID
        or machine_config.authorized_next_stage != STAGE4_STAGE_ID
    ):
        raise Stage4WorkflowError("Stage 3 machine config semantics drift")

    commit_parent = _git_text(repo, ["rev-parse", f"{STAGE3_COMMIT}^"])
    commit_tree = _git_text(repo, ["rev-parse", f"{STAGE3_COMMIT}^{{tree}}"])
    commit_subject = _git_text(
        repo,
        ["show", "-s", "--format=%s", STAGE3_COMMIT],
    )
    head = _git_text(repo, ["rev-parse", "HEAD"])
    changed_paths = tuple(
        value.decode("utf-8", errors="strict")
        for value in _git_bytes(
            repo,
            [
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                "-z",
                STAGE3_COMMIT,
            ],
        ).split(b"\0")
        if value
    )
    if (
        commit_parent != STAGE3_COMMIT_PARENT
        or commit_tree != STAGE3_COMMIT_TREE
        or commit_subject != STAGE3_COMMIT_SUBJECT
        or head != STAGE3_COMMIT
        or changed_paths != STAGE3_REVIEWED_PATHS
    ):
        raise Stage4WorkflowError("Stage 3 approved Git commit binding drift")
    cached = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--exit-code"],
        cwd=repo,
        check=False,
    )
    if cached.returncode != 0:
        raise Stage4WorkflowError("real Git index is not empty")
    if recorded_git != {
        "schema_version": "ppo_highres_frontier_stage3_prospective_git_tree/v1",
        "base_commit": STAGE3_COMMIT_PARENT,
        "head_commit": STAGE3_COMMIT_PARENT,
        "prospective_git_tree": STAGE3_COMMIT_TREE,
        "changed_paths": list(STAGE3_REVIEWED_PATHS),
        "changed_path_set_sha256": STAGE3_REVIEWED_PATH_SET_SHA256,
        "real_index_empty": True,
    }:
        raise Stage4WorkflowError("Stage 3 recorded Git identity drift")

    try:
        current_only = _verify_frozen_stage3_source_identity(
            source_identity=recorded_source,
            repo_root=repo,
            commit=STAGE3_COMMIT,
        )
        current_environment = stage3_workflow.stage3_environment_identity()
    except Exception as exc:
        raise Stage4WorkflowError("Stage 3 source or environment cannot be replayed") from exc
    if (
        not isinstance(recorded_source, dict)
        or recorded_source.get("source_set_sha256") != STAGE3_SOURCE_SET_SHA256
        or recorded_environment != current_environment
        or _canonical_sha256(recorded_environment) != STAGE3_ENVIRONMENT_SHA256
    ):
        raise Stage4WorkflowError("Stage 3 source or environment identity drift")

    summary = _strict_canonical(
        artifact_snapshots["summary.json"],
        "Stage 3 summary",
    )
    routing = _strict_canonical(
        artifact_snapshots["routing.json"],
        "Stage 3 routing",
    )
    acceptance = summary.get("acceptance")
    if (
        summary.get("state") != "machine_passed"
        or summary.get("run_id") != STAGE3_RUN_ID
        or summary.get("stage_id") != STAGE3_STAGE_ID
        or summary.get("prospective_git_tree") != STAGE3_COMMIT_TREE
        or summary.get("execution_source_set_sha256") != STAGE3_SOURCE_SET_SHA256
        or summary.get("execution_environment_sha256") != STAGE3_ENVIRONMENT_SHA256
        or summary.get("stage2_gate_sha256") != STAGE2_GATE_SHA256
        or not isinstance(acceptance, dict)
        or acceptance.get("all_16_stage3_items_passed") is not True
        or not isinstance(acceptance.get("items"), dict)
        or len(acceptance["items"]) != 16
        or not all(value is True for value in acceptance["items"].values())
        or routing.get("route") != "awaiting_independent_review"
        or routing.get("authorized_next_stage") != STAGE4_STAGE_ID
        or routing.get("next_stage_entered") is not False
    ):
        raise Stage4WorkflowError("Stage 3 machine acceptance binding drift")

    if not isinstance(recorded_stage2, dict):
        raise Stage4WorkflowError("Stage 3 Stage 2 pass-through is missing")
    try:
        stage2_handle = stage3_workflow.verify_frozen_stage2_authority(
            gate_path=str(recorded_stage2["gate_path"]),
            repo_root=repo,
        )
    except Exception as exc:
        raise Stage4WorkflowError("Stage 2 pass-through authority drift") from exc
    if (
        stage2_handle.identity != recorded_stage2
        or recorded_stage2.get("gate_sha256") != STAGE2_GATE_SHA256
        or _strict_canonical(
            artifact_snapshots["evidence/stage2_authority_audit.json"],
            "Stage 3 Stage 2 authority audit",
        )
        != recorded_stage2
    ):
        raise Stage4WorkflowError("Stage 3 recorded Stage 2 pass-through drift")

    external_snapshots = _validate_review_approval_gate(
        repo=repo,
        stage=stage,
        review=review,
        approval=approval,
        gate=gate,
        bindings_source_hash=STAGE3_SOURCE_SET_SHA256,
        config_snapshot=config_snapshot,
        manifest_snapshot=root_snapshots["manifest.json"],
        summary_snapshot=artifact_snapshots["summary.json"],
        routing_snapshot=artifact_snapshots["routing.json"],
    )
    identity = {
        "schema_version": "ppo_highres_frontier_stage4_stage3_authority_audit/v1",
        "verified": True,
        "stage3_commit": STAGE3_COMMIT,
        "stage3_commit_parent": STAGE3_COMMIT_PARENT,
        "stage3_commit_tree": STAGE3_COMMIT_TREE,
        "stage3_run_id": STAGE3_RUN_ID,
        "gate_path": str(gate_file),
        "gate_sha256": root_snapshots["gate.json"].sha256,
        "approval_sha256": root_snapshots["approval.json"].sha256,
        "review_sha256": root_snapshots["review.json"].sha256,
        "manifest_sha256": root_snapshots["manifest.json"].sha256,
        "source_set_sha256": STAGE3_SOURCE_SET_SHA256,
        "environment_sha256": STAGE3_ENVIRONMENT_SHA256,
        "stage2_gate_sha256": STAGE2_GATE_SHA256,
        "reviewed_prospective_git_tree": STAGE3_COMMIT_TREE,
        "reviewed_path_set_sha256": STAGE3_REVIEWED_PATH_SET_SHA256,
        "reviewed_path_count": len(STAGE3_REVIEWED_PATHS),
        "authorized_stage": STAGE4_STAGE_ID,
        "authority_scope": (
            "external_controller_claim_integrity_not_local_identity_authentication/v1"
        ),
    }
    all_snapshots = (
        *tuple((name, snapshot) for name, snapshot in sorted(root_snapshots.items())),
        *tuple(
            (f"manifest artifact {name}", snapshot)
            for name, snapshot in sorted(artifact_snapshots.items())
        ),
        *external_snapshots,
        *tuple(
            (f"recorded non-Git source {index}", snapshot)
            for index, snapshot in enumerate(current_only)
        ),
    )
    handle = FrozenStage3AuthorityHandle(
        identity=identity,
        snapshots=all_snapshots,
        stage2_authority=stage2_handle,
        stage_root=stage,
        root_artifacts=_STAGE3_ROOT_ARTIFACTS,
    )
    handle.require_current()
    return handle


def _validate_review_approval_gate(
    *,
    repo: Path,
    stage: Path,
    review: dict[str, object],
    approval: dict[str, object],
    gate: dict[str, object],
    bindings_source_hash: str,
    config_snapshot: FrozenFileSnapshot,
    manifest_snapshot: FrozenFileSnapshot,
    summary_snapshot: FrozenFileSnapshot,
    routing_snapshot: FrozenFileSnapshot,
) -> tuple[tuple[str, FrozenFileSnapshot], ...]:
    if set(review) != _REVIEW_KEYS:
        raise Stage4WorkflowError("Stage 3 review schema drift")
    declarations = {
        "schema_version": "ppo_highres_frontier_stage3_independent_review/v1",
        "state": "awaiting_human_approval",
        "goal_id": GOAL_ID,
        "stage_id": STAGE3_STAGE_ID,
        "run_id": STAGE3_RUN_ID,
        "spec_verdict": "PASS",
        "quality_verdict": "PASS",
        "issue_counts": {"critical": 0, "important": 0, "minor": 0},
        "final_conclusion": "READY_FOR_HUMAN_APPROVAL",
        "reviewed_base_commit": STAGE3_COMMIT_PARENT,
        "reviewed_prospective_git_tree": STAGE3_COMMIT_TREE,
        "reviewed_paths": list(STAGE3_REVIEWED_PATHS),
        "reviewed_path_set_sha256": STAGE3_REVIEWED_PATH_SET_SHA256,
        "source_set_sha256": bindings_source_hash,
        "config_sha256": config_snapshot.sha256,
        "environment_sha256": STAGE3_ENVIRONMENT_SHA256,
        "summary_sha256": summary_snapshot.sha256,
        "routing_sha256": routing_snapshot.sha256,
        "manifest_sha256": manifest_snapshot.sha256,
        "stage2_gate_sha256": STAGE2_GATE_SHA256,
        "checkpoint_state": "stage3_no_checkpoint/v1",
        "authorized_next_stage": STAGE4_STAGE_ID,
    }
    if any(review.get(name) != value for name, value in declarations.items()):
        raise Stage4WorkflowError("Stage 3 review declaration drift")
    _canonical_uuid(review.get("reviewer_agent_id"), "reviewer agent")
    _parse_utc(review.get("review_recorded_at_utc"), "review timestamp")

    snapshot_names = (
        "review_report",
        "review_package",
        "review_bindings",
        "review_controller",
        "review_recorder",
        "implementation_report",
    )
    snapshots = {
        name: _capture_recorded_binding(review.get(name), f"Stage 3 {name}")
        for name in snapshot_names
    }
    index_binding = review.get("real_git_index")
    if not isinstance(index_binding, dict) or set(index_binding) != {
        "path",
        "sha256",
        "bytes",
        "lf_count",
        "logical_line_count",
    }:
        raise Stage4WorkflowError("Stage 3 historical Git index binding schema drift")
    current_index = Path(
        _git_text(repo, ["rev-parse", "--git-path", "index"])
    ).expanduser().resolve()
    if Path(str(index_binding["path"])).expanduser().resolve() != current_index:
        raise Stage4WorkflowError("Stage 3 historical Git index path drift")
    bindings = _strict_canonical(
        snapshots["review_bindings"],
        "Stage 3 review bindings",
    )
    _validate_review_bindings(
        bindings=bindings,
        stage=stage,
        review=review,
        package=snapshots["review_package"],
    )
    challenge_material = {
        key: review[key]
        for key in (
            "goal_id",
            "stage_id",
            "run_id",
            "reviewer_agent_id",
            "reviewed_base_commit",
            "reviewed_prospective_git_tree",
            "reviewed_path_set_sha256",
            "source_set_sha256",
            "config_sha256",
            "environment_sha256",
            "manifest_sha256",
            "stage2_gate_sha256",
            "checkpoint_state",
            "authorized_next_stage",
            "review_report",
            "review_package",
            "review_bindings",
        )
    }
    challenge = _canonical_sha256(challenge_material)
    if review.get("approval_challenge") != challenge:
        raise Stage4WorkflowError("Stage 3 review approval challenge drift")
    try:
        stage2_workflow.replay_stage2_review_package(
            repo_root=repo,
            base_commit=STAGE3_COMMIT_PARENT,
            review_package=snapshots["review_package"].path,
            expected_prospective_git_tree=STAGE3_COMMIT_TREE,
            expected_reviewed_paths=list(STAGE3_REVIEWED_PATHS),
        )
    except Exception as exc:
        raise Stage4WorkflowError("Stage 3 review package replay failed") from exc

    approval_controller = _capture_recorded_binding(
        gate.get("bindings", {}).get("approval_controller")
        if isinstance(gate.get("bindings"), dict)
        else None,
        "Stage 3 approval controller",
    )
    approval_context = _build_context(
        review=review,
        bindings=bindings,
        approval_controller=approval_controller.binding(),
        approval_sha256=ABSENT_AUTHORITY_SHA256,
    )
    _validate_approval(approval, review=review, expected_context=approval_context)
    gate_context = _build_context(
        review=review,
        bindings=bindings,
        approval_controller=approval_controller.binding(),
        approval_sha256=STAGE3_APPROVAL_SHA256,
    )
    if set(gate) != {
        "schema_version",
        "state",
        "authorized_next_stage",
        "run_id",
        "commit_sha256",
        "bindings",
        "history",
    } or any(
        gate.get(key) != value
        for key, value in {
            "schema_version": "ppo_highres_frontier_stage3_verified_gate/v1",
            "state": "next_stage",
            "authorized_next_stage": STAGE4_STAGE_ID,
            "run_id": STAGE3_RUN_ID,
            "commit_sha256": STAGE3_COMMIT,
            "bindings": gate_context,
        }.items()
    ):
        raise Stage4WorkflowError("Stage 3 gate schema or context drift")
    _validate_stage3_gate_history(gate["history"], gate_context)
    return (
        *tuple((name, snapshot) for name, snapshot in sorted(snapshots.items())),
        ("approval_controller", approval_controller),
    )


def _validate_review_bindings(
    *,
    bindings: dict[str, object],
    stage: Path,
    review: dict[str, object],
    package: FrozenFileSnapshot,
) -> None:
    expected_keys = {
        "authorized_next_stage",
        "blocking_severities",
        "checkpoint_state",
        "config_sha256",
        "environment_sha256",
        "goal_id",
        "machine_stage_root",
        "manifest_sha256",
        "required_review_conclusions",
        "review_package_bytes",
        "review_package_lf_count",
        "review_package_logical_line_count",
        "review_package_path",
        "review_package_sha256",
        "reviewed_base_commit",
        "reviewed_path_set_sha256",
        "reviewed_paths",
        "reviewed_prospective_git_tree",
        "routing_sha256",
        "run_id",
        "schema_version",
        "source_set_sha256",
        "stage2_gate_sha256",
        "stage_id",
        "summary_sha256",
    }
    expected_values = {
        "schema_version": "ppo_highres_frontier_stage3_review_bindings/v1",
        "goal_id": GOAL_ID,
        "stage_id": STAGE3_STAGE_ID,
        "run_id": STAGE3_RUN_ID,
        "machine_stage_root": str(stage),
        "blocking_severities": ["Critical", "Important"],
        "required_review_conclusions": ["spec_compliance", "code_quality"],
        "reviewed_base_commit": STAGE3_COMMIT_PARENT,
        "reviewed_prospective_git_tree": STAGE3_COMMIT_TREE,
        "reviewed_paths": list(STAGE3_REVIEWED_PATHS),
        "reviewed_path_set_sha256": STAGE3_REVIEWED_PATH_SET_SHA256,
        "source_set_sha256": STAGE3_SOURCE_SET_SHA256,
        "config_sha256": review["config_sha256"],
        "environment_sha256": STAGE3_ENVIRONMENT_SHA256,
        "summary_sha256": review["summary_sha256"],
        "routing_sha256": review["routing_sha256"],
        "manifest_sha256": STAGE3_MANIFEST_SHA256,
        "stage2_gate_sha256": STAGE2_GATE_SHA256,
        "checkpoint_state": "stage3_no_checkpoint/v1",
        "authorized_next_stage": STAGE4_STAGE_ID,
        "review_package_path": str(package.path),
        "review_package_sha256": package.sha256,
        "review_package_bytes": package.size_bytes,
        "review_package_lf_count": package.lf_count,
        "review_package_logical_line_count": package.logical_line_count,
    }
    if set(bindings) != expected_keys or any(
        bindings.get(key) != value for key, value in expected_values.items()
    ):
        raise Stage4WorkflowError("Stage 3 review bindings drift")


def _build_context(
    *,
    review: dict[str, object],
    bindings: dict[str, object],
    approval_controller: dict[str, object],
    approval_sha256: str,
) -> dict[str, object]:
    context = {
        "schema_version": "ppo_highres_frontier_stage3_gate_context/v1",
        "goal_id": GOAL_ID,
        "thread_id": "019f49d3-8597-71a3-9908-e39febf4cbb3",
        "stage_id": STAGE3_STAGE_ID,
        "run_id": STAGE3_RUN_ID,
        "reviewed_base_commit": STAGE3_COMMIT_PARENT,
        "reviewed_prospective_git_tree": STAGE3_COMMIT_TREE,
        "reviewed_paths": list(STAGE3_REVIEWED_PATHS),
        "reviewed_path_set_sha256": STAGE3_REVIEWED_PATH_SET_SHA256,
        "source_set_sha256": STAGE3_SOURCE_SET_SHA256,
        "environment_sha256": STAGE3_ENVIRONMENT_SHA256,
        "config_sha256": review["config_sha256"],
        "summary_sha256": review["summary_sha256"],
        "routing_sha256": review["routing_sha256"],
        "manifest_sha256": STAGE3_MANIFEST_SHA256,
        "stage2_gate_sha256": STAGE2_GATE_SHA256,
        "review_sha256": STAGE3_REVIEW_SHA256,
        "review_recorded_at_utc": review["review_recorded_at_utc"],
        "approval_challenge": review["approval_challenge"],
        "review_report": review["review_report"],
        "review_package": review["review_package"],
        "review_bindings": review["review_bindings"],
        "review_controller": review["review_controller"],
        "review_recorder": review["review_recorder"],
        "approval_controller": approval_controller,
        "checkpoint_state": "stage3_no_checkpoint/v1",
        "commit_sha256": STAGE3_COMMIT,
        "commit_parent": STAGE3_COMMIT_PARENT,
        "commit_tree": STAGE3_COMMIT_TREE,
        "commit_subject": STAGE3_COMMIT_SUBJECT,
        "approval_sha256": approval_sha256,
        "authorized_next_stage": STAGE4_STAGE_ID,
        "review_bindings_schema_version": bindings["schema_version"],
    }
    if set(context) != _CONTEXT_KEYS:
        raise Stage4WorkflowError("internal Stage 3 context field drift")
    return context


def _validate_approval(
    approval: dict[str, object],
    *,
    review: dict[str, object],
    expected_context: dict[str, object],
) -> None:
    expected = {
        "schema_version": "ppo_highres_frontier_stage3_human_approval/v2",
        "state": "approved",
        "authority_kind": "externally_persisted_controller_user_approval_claim/v1",
        "actor": "user",
        "event_authenticity_source": (
            "external_controller_boundary_not_locally_authenticated/v1"
        ),
        "local_verification_scope": (
            "artifact_integrity_challenge_git_and_temporal_ordering_only/v1"
        ),
        "thread_id": "019f49d3-8597-71a3-9908-e39febf4cbb3",
        "approval_text": "批准 Stage 3 Gate",
        "approval_text_utf8_sha256": hashlib.sha256(
            "批准 Stage 3 Gate".encode("utf-8")
        ).hexdigest(),
        "goal_id": GOAL_ID,
        "stage_id": STAGE3_STAGE_ID,
        "run_id": STAGE3_RUN_ID,
        "reviewed_prospective_git_tree": STAGE3_COMMIT_TREE,
        "commit_sha256": STAGE3_COMMIT,
        "review_sha256": STAGE3_REVIEW_SHA256,
        "approval_challenge": review["approval_challenge"],
        "bindings": expected_context,
    }
    if set(approval) != _APPROVAL_KEYS or any(
        approval.get(key) != value for key, value in expected.items()
    ):
        raise Stage4WorkflowError("Stage 3 approval schema or declaration drift")
    _canonical_uuid(approval.get("thread_id"), "approval thread")
    _canonical_uuid(approval.get("controller_event_id"), "approval event")
    observed = _parse_utc(
        approval.get("controller_observed_at_utc"),
        "approval timestamp",
    )
    reviewed = _parse_utc(review.get("review_recorded_at_utc"), "review timestamp")
    if observed < reviewed:
        raise Stage4WorkflowError("Stage 3 approval predates review")
    event_source = {
        "schema_version": (
            "ppo_highres_frontier_stage3_external_controller_event_binding/v1"
        ),
        "context_binding_sha256": _canonical_sha256(approval["bindings"]),
        "thread_id": approval["thread_id"],
        "controller_event_id": approval["controller_event_id"],
        "approval_text": approval["approval_text"],
        "controller_observed_at_utc": approval["controller_observed_at_utc"],
        "review_sha256": approval["review_sha256"],
        "approval_challenge": approval["approval_challenge"],
        "commit_sha256": approval["commit_sha256"],
    }
    if approval.get("controller_event_binding_sha256") != _canonical_sha256(
        event_source
    ):
        raise Stage4WorkflowError("Stage 3 approval event binding drift")


def _validate_stage3_gate_history(
    history: object,
    bindings: Mapping[str, object],
) -> None:
    states = (
        "machine_passed",
        "awaiting_independent_review",
        "awaiting_human_approval",
        "approved",
        "next_stage",
    )
    if not isinstance(history, list) or len(history) != len(states):
        raise Stage4WorkflowError("Stage 3 gate history length drift")
    previous = ABSENT_AUTHORITY_SHA256
    for index, state in enumerate(states):
        event_bindings = json.loads(
            json.dumps(dict(bindings), ensure_ascii=False, sort_keys=True)
        )
        if state in {"machine_passed", "awaiting_independent_review"}:
            event_bindings["review_sha256"] = ABSENT_AUTHORITY_SHA256
        if state not in {"approved", "next_stage"}:
            event_bindings["approval_sha256"] = ABSENT_AUTHORITY_SHA256
            event_bindings["commit_sha256"] = ABSENT_AUTHORITY_SHA256
        event = {
            "state": state,
            "previous_record_hash": previous,
            "bindings": event_bindings,
        }
        expected = {**event, "record_hash": _canonical_sha256(event)}
        if history[index] != expected:
            raise Stage4WorkflowError("Stage 3 gate history hash-chain drift")
        previous = expected["record_hash"]


def _capture_recorded_binding(
    value: object,
    label: str,
) -> FrozenFileSnapshot:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "sha256",
        "bytes",
        "lf_count",
        "logical_line_count",
    }:
        raise Stage4WorkflowError(f"{label} binding schema drift")
    try:
        snapshot = FrozenFileSnapshot.capture(str(value["path"]), label)
    except Stage1WorkflowError as exc:
        raise Stage4WorkflowError(f"{label} is missing or unsafe") from exc
    if snapshot.binding() != value:
        raise Stage4WorkflowError(f"{label} binding drift")
    return snapshot


def _verify_frozen_stage3_source_identity(
    *,
    source_identity: object,
    repo_root: Path,
    commit: str,
) -> tuple[FrozenFileSnapshot, ...]:
    if (
        not isinstance(source_identity, dict)
        or set(source_identity) != {"schema_version", "source_set_sha256", "paths"}
        or source_identity.get("schema_version") != "stage3_reviewed_source_set/v1"
    ):
        raise Stage4WorkflowError("frozen Stage 3 source identity schema drift")
    entries = source_identity.get("paths")
    if not isinstance(entries, list) or not entries:
        raise Stage4WorkflowError("frozen Stage 3 source path set is empty")
    repo = repo_root.expanduser().resolve()
    _git_bytes(repo, ["cat-file", "-e", f"{commit}^{{commit}}"])
    relative_paths: list[str] = []
    current_only: list[FrozenFileSnapshot] = []
    digest = hashlib.sha256()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise Stage4WorkflowError("frozen Stage 3 source entry schema drift")
        relative = entry.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise Stage4WorkflowError("frozen Stage 3 source path is unsafe")
        relative_paths.append(relative)
        exists_in_commit = subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", f"{commit}:{relative}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if exists_in_commit.returncode == 0:
            payload = _git_bytes(repo, ["show", f"{commit}:{relative}"])
        else:
            try:
                snapshot = FrozenFileSnapshot.capture(
                    repo / relative,
                    f"recorded non-Git Stage 3 source {relative}",
                    authority_root=repo,
                )
            except Stage1WorkflowError as exc:
                raise Stage4WorkflowError(
                    "recorded non-Git Stage 3 source is missing or unsafe"
                ) from exc
            current_only.append(snapshot)
            payload = snapshot.payload
        if (
            entry.get("size_bytes") != len(payload)
            or entry.get("sha256") != hashlib.sha256(payload).hexdigest()
        ):
            raise Stage4WorkflowError("frozen Stage 3 source blob drift")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    if relative_paths != sorted(set(relative_paths)):
        raise Stage4WorkflowError("frozen Stage 3 source path set is not exact")
    if source_identity.get("source_set_sha256") != digest.hexdigest():
        raise Stage4WorkflowError("frozen Stage 3 source-set hash drift")
    try:
        for snapshot in current_only:
            snapshot.require_current("recorded non-Git Stage 3 source")
    except Stage1WorkflowError as exc:
        raise Stage4WorkflowError(
            "recorded non-Git Stage 3 source changed during verification"
        ) from exc
    return tuple(current_only)


def _build_stage4_runtime_binding(
    *,
    config: Stage4Config,
    run_id: str,
    source_identity: dict[str, object],
    environment_identity: dict[str, object],
    git_identity: dict[str, object],
    stage3_authority: dict[str, object],
    stage1_config_binding: dict[str, object],
) -> Stage4RuntimeBinding:
    validate_run_id(run_id)
    if config.run_id is not None:
        raise Stage4WorkflowError("repository Stage 4 config must not bind a run ID")
    if (
        source_identity.get("schema_version") != "stage4_reviewed_source_set/v1"
        or not isinstance(source_identity.get("paths"), list)
        or not source_identity["paths"]
        or environment_identity.get("schema_version")
        != "ppo_highres_frontier_stage4_environment/v1"
        or environment_identity.get("cuda_available") is not True
        or environment_identity.get("compute_dtype") != "float32"
        or environment_identity.get("amp_enabled") is not False
        or environment_identity.get("multiprocessing_start_method") != "spawn"
        or git_identity.get("schema_version")
        != "ppo_highres_frontier_stage4_prospective_git_tree/v1"
        or git_identity.get("head_commit") != STAGE3_COMMIT
        or git_identity.get("base_commit") != STAGE3_COMMIT
        or git_identity.get("real_index_empty") is not True
    ):
        raise Stage4WorkflowError("Stage 4 execution identity binding is incomplete")
    expected_authority = config.stage3_authority
    if (
        stage3_authority.get("verified") is not True
        or stage3_authority.get("stage3_commit") != expected_authority.commit_sha256
        or stage3_authority.get("gate_sha256") != expected_authority.gate_sha256
        or stage3_authority.get("approval_sha256")
        != expected_authority.approval_sha256
        or stage3_authority.get("review_sha256") != expected_authority.review_sha256
        or stage3_authority.get("manifest_sha256")
        != expected_authority.manifest_sha256
        or stage3_authority.get("source_set_sha256")
        != expected_authority.source_set_sha256
        or stage3_authority.get("environment_sha256")
        != expected_authority.environment_sha256
        or stage3_authority.get("stage2_gate_sha256")
        != expected_authority.stage2_gate_sha256
        or stage3_authority.get("reviewed_path_set_sha256")
        != expected_authority.reviewed_path_set_sha256
        or stage3_authority.get("authorized_stage") != config.stage_id
    ):
        raise Stage4WorkflowError("Stage 3 authority does not bind the Stage 4 config")
    if stage1_config_binding != {
        "path": stage1_config_binding.get("path"),
        "sha256": config.stage1_config_sha256,
    } or not isinstance(stage1_config_binding.get("path"), str):
        raise Stage4WorkflowError("Stage 1 Smoke config binding drift")
    payload = config.model_dump(mode="json")
    payload["run_id"] = run_id
    payload["execution_source_identity"] = source_identity
    payload["execution_environment_identity"] = environment_identity
    payload["execution_git_identity"] = git_identity
    payload["execution_stage3_authority"] = stage3_authority
    payload["stage1_config_binding"] = stage1_config_binding
    try:
        config_bytes = ArtifactStore.canonical_json_bytes(payload)
    except (TypeError, ValueError) as exc:
        raise Stage4WorkflowError("Stage 4 runtime config is not canonical") from exc
    return Stage4RuntimeBinding(
        payload=payload,
        config_bytes=config_bytes,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
    )


def _checkpoint_lineage(runtime: Stage4RuntimeBinding) -> dict[str, object]:
    source = runtime.payload["execution_source_identity"]
    git_identity = runtime.payload["execution_git_identity"]
    authority = runtime.payload["execution_stage3_authority"]
    assert isinstance(source, dict)
    assert isinstance(git_identity, dict)
    assert isinstance(authority, dict)
    return {
        "schema_version": "ppo_highres_frontier_stage4_checkpoint_lineage/v1",
        "stage3_commit": authority["stage3_commit"],
        "stage3_gate_sha256": authority["gate_sha256"],
        "stage3_approval_sha256": authority["approval_sha256"],
        "stage3_review_sha256": authority["review_sha256"],
        "stage4_run_id": runtime.payload["run_id"],
        "stage4_source_set_sha256": source["source_set_sha256"],
        "stage4_prospective_git_tree": git_identity["prospective_git_tree"],
        "stage4_config_sha256": runtime.config_sha256,
    }


def _validate_stage4_machine_audit(
    *,
    config: Stage4Config,
    runtime: Stage4RuntimeBinding,
    audit: Stage4MachineAuditBundle,
) -> None:
    if not isinstance(audit, Stage4MachineAuditBundle) or len(audit.updates) != 3:
        raise Stage4WorkflowError("Stage 4 machine acceptance update count drift")
    previous_after: str | None = None
    for expected_step, update in enumerate(audit.updates, start=1):
        if (
            not isinstance(update, dict)
            or set(update) != _STAGE4_UPDATE_AUDIT_KEYS
            or update.get("schema_version") != "stage4_smoke_update_audit/v1"
            or update.get("update_step") != expected_step
            or update.get("config_sha256") != runtime.config_sha256
            or not finite_tree(update)
        ):
            raise Stage4WorkflowError("Stage 4 machine acceptance update schema drift")
        collection = update.get("collection")
        gae = update.get("gae")
        ppo = update.get("ppo")
        buffer = update.get("buffer")
        checkpoint = update.get("checkpoint")
        if not all(
            isinstance(value, dict)
            for value in (collection, gae, ppo, buffer, checkpoint)
        ):
            raise Stage4WorkflowError("Stage 4 machine acceptance audit is incomplete")
        assert isinstance(collection, dict)
        assert isinstance(gae, dict)
        assert isinstance(ppo, dict)
        assert isinstance(buffer, dict)
        assert isinstance(checkpoint, dict)
        if set(collection) != _STAGE4_COLLECTION_AUDIT_KEYS:
            raise Stage4WorkflowError(
                "Stage 4 machine acceptance collection schema drift"
            )
        worker_pids = collection["worker_pids"]
        inference_pids = collection["inference_pids"]
        diagnostic_reset_counts = collection["diagnostic_reset_counts"]
        unique_snapshot_count = collection["unique_snapshot_count"]
        terminal_transition_count = collection["terminal_transition_count"]
        policy_device = collection["policy_device"]
        if (
            not _is_exact_int_list(
                collection["per_env_trainable_counts"],
                length=8,
                minimum=0,
            )
            or not _is_exact_int_list(
                diagnostic_reset_counts,
                length=8,
                minimum=0,
            )
            or not _is_exact_int_list(worker_pids, length=8, minimum=1)
            or len(set(worker_pids)) != 8
            or not _is_exact_int_list(inference_pids, length=1, minimum=1)
            or type(collection["inference_main_process_only"]) is not bool
            or type(collection["inference_batch_count"]) is not int
            or collection["inference_batch_count"] <= 0
            or not isinstance(policy_device, str)
            or not (
                policy_device == "cuda"
                or (
                    policy_device.startswith("cuda:")
                    and policy_device.removeprefix("cuda:").isdigit()
                )
            )
            or not _is_sha256(collection["policy_state_sha256"])
            or type(collection["snapshot_count"]) is not int
            or type(unique_snapshot_count) is not int
            or not 1 <= unique_snapshot_count <= collection["snapshot_count"]
            or type(terminal_transition_count) is not int
            or not 0 <= terminal_transition_count <= collection["snapshot_count"]
            or type(collection["old_logprob_factorization_exact"]) is not bool
        ):
            raise Stage4WorkflowError(
                "Stage 4 machine acceptance collection type drift"
            )
        if (
            type(collection["trainable_transition_count"]) is not int
            or collection["trainable_transition_count"] != 1024
            or collection["per_env_trainable_counts"] != [128] * 8
            or collection["worker_start_methods"] != ["spawn"] * 8
            or collection["inference_main_process_only"] is not True
            or collection["snapshot_count"] != 1024
            or collection["old_logprob_factorization_exact"] is not True
            or gae.get("finite") is not True
            or abs(float(gae.get("normalized_advantage_mean", math.inf))) > 1.0e-5
            or abs(float(gae.get("normalized_advantage_std", -math.inf)) - 1.0)
            > 1.0e-5
            or float(ppo.get("initial_ratio_max_abs_error", math.inf)) > 1.0e-5
            or float(ppo.get("grad_post_clip_norm_max", math.inf)) > 0.500001
            or float(ppo.get("parameter_change_l2", 0.0)) <= 0.0
            or int(ppo.get("optimizer_steps", 0)) <= 0
            or buffer != {"consumed": True, "cleared": True}
            or checkpoint.get("update_step") != expected_step
            or checkpoint.get("loaded_last_complete_step") != expected_step
            or checkpoint.get("deterministic_action_bit_exact") is not True
            or checkpoint.get("policy_state_sha256")
            != ppo.get("policy_state_sha256_after")
        ):
            raise Stage4WorkflowError("Stage 4 machine acceptance gate failed")
        finite_metrics = (
            "policy_loss",
            "value_loss",
            "frontier_entropy_mean",
            "approx_kl",
            "grad_pre_clip_norm_max",
            "grad_post_clip_norm_max",
            "parameter_change_l2",
        )
        if any(
            not isinstance(ppo.get(name), (int, float))
            or not math.isfinite(float(ppo[name]))
            for name in finite_metrics
        ):
            raise Stage4WorkflowError("Stage 4 PPO machine metric is non-finite")
        before = ppo.get("policy_state_sha256_before")
        after = ppo.get("policy_state_sha256_after")
        collection_hash = collection["policy_state_sha256"]
        if (
            not _is_sha256(before)
            or not _is_sha256(after)
            or before == after
            or collection_hash != before
            or (previous_after is not None and before != previous_after)
        ):
            raise Stage4WorkflowError("Stage 4 policy lineage chain drift")
        previous_after = str(after)
        approx_kl = float(ppo["approx_kl"])
        if (
            (ppo.get("early_stopped") is True and approx_kl <= config.ppo.target_kl)
            or (ppo.get("early_stopped") is False and approx_kl > config.ppo.target_kl)
        ):
            raise Stage4WorkflowError("Stage 4 KL early-stop audit drift")

    tiny = audit.tiny_overfit
    seeds = tiny.get("seeds") if isinstance(tiny, dict) else None
    if (
        not isinstance(tiny, dict)
        or tiny.get("schema_version") != "stage4_tiny_overfit_acceptance/v1"
        or tiny.get("context_count") != config.acceptance.tiny_overfit_contexts
        or tiny.get("passed") is not True
        or not isinstance(seeds, list)
        or len(seeds) != 3
        or tuple(seed.get("seed") for seed in seeds if isinstance(seed, dict))
        != config.acceptance.tiny_overfit_seeds
    ):
        raise Stage4WorkflowError("Stage 4 tiny-overfit machine acceptance drift")
    for seed in seeds:
        if (
            not isinstance(seed, dict)
            or seed.get("passed") is not True
            or int(seed.get("stop_update", 201))
            > config.acceptance.tiny_overfit_max_updates
            or float(seed.get("correct_frontier_probability_min", 0.0))
            < config.acceptance.tiny_frontier_probability_min
            or float(seed.get("theta_error_rad_max", math.inf))
            > config.acceptance.tiny_theta_error_rad_max
            or float(seed.get("value_mse_reduction", 0.0))
            < config.acceptance.tiny_value_mse_reduction_min
            or seed.get("ppo_update_count") != seed.get("stop_update")
            or seed.get("rollout_batch_count") != seed.get("stop_update")
        ):
            raise Stage4WorkflowError("Stage 4 tiny-overfit seed gate failed")
    resume = audit.training_resume
    resume_receipt = resume.get("resume_receipt") if isinstance(resume, dict) else None
    if (
        not isinstance(resume, dict)
        or set(resume)
        != {
            "schema_version",
            "fresh_trainer_initial_update_step",
            "resume_receipt",
            "normalization_state_applied",
            "scenario_sampler_state_applied",
            "vector_env_states_applied",
            "vector_env_state_count",
            "continued_from_current_state",
            "next_update_executed",
            "next_update_step",
            "collection_policy_state_sha256",
            "policy_state_sha256_before_next_update",
            "policy_state_sha256_after_next_update",
            "optimizer_state_sha256_after_next_update",
        }
        or resume.get("schema_version") != "stage4_training_resume_audit/v1"
        or resume.get("fresh_trainer_initial_update_step") != 0
        or resume.get("normalization_state_applied") is not True
        or resume.get("scenario_sampler_state_applied") is not True
        or resume.get("vector_env_states_applied") is not True
        or resume.get("vector_env_state_count") != 8
        or resume.get("continued_from_current_state") is not True
        or resume.get("next_update_executed") is not True
        or resume.get("next_update_step") != 2
        or not finite_tree(resume)
        or not isinstance(resume_receipt, dict)
        or set(resume_receipt)
        != {
            "schema_version",
            "update_step",
            "next_update_step",
            "checkpoint_sha256",
            "manifest_sha256",
            "policy_state_sha256",
            "optimizer_state_sha256",
            "normalization_state_sha256",
            "scenario_sampler_state_sha256",
            "vector_env_states_sha256",
        }
        or resume_receipt.get("schema_version") != "stage4_training_resume/v1"
        or resume_receipt.get("update_step") != 1
        or resume_receipt.get("next_update_step") != 2
    ):
        raise Stage4WorkflowError("Stage 4 training resume audit schema drift")
    first_checkpoint = audit.updates[0]["checkpoint"]
    second_collection = audit.updates[1]["collection"]
    second_ppo = audit.updates[1]["ppo"]
    assert isinstance(first_checkpoint, dict)
    assert isinstance(second_collection, dict)
    assert isinstance(second_ppo, dict)
    resume_hash_fields = (
        "checkpoint_sha256",
        "manifest_sha256",
        "policy_state_sha256",
        "optimizer_state_sha256",
        "normalization_state_sha256",
        "scenario_sampler_state_sha256",
        "vector_env_states_sha256",
    )
    if (
        any(not _is_sha256(resume_receipt.get(name)) for name in resume_hash_fields)
        or resume_receipt["checkpoint_sha256"]
        != first_checkpoint.get("checkpoint_sha256")
        or resume_receipt["manifest_sha256"]
        != first_checkpoint.get("manifest_sha256")
        or resume_receipt["policy_state_sha256"]
        != first_checkpoint.get("policy_state_sha256")
        or resume.get("collection_policy_state_sha256")
        != second_collection.get("policy_state_sha256")
        or resume.get("policy_state_sha256_before_next_update")
        != second_ppo.get("policy_state_sha256_before")
        or resume.get("policy_state_sha256_after_next_update")
        != second_ppo.get("policy_state_sha256_after")
        or resume_receipt["policy_state_sha256"]
        != resume.get("collection_policy_state_sha256")
        or not _is_sha256(resume.get("optimizer_state_sha256_after_next_update"))
    ):
        raise Stage4WorkflowError("Stage 4 training resume lineage drift")
    if not isinstance(audit.checkpoint_action_fixture_npz, bytes) or not (
        audit.checkpoint_action_fixture_npz
    ):
        raise Stage4WorkflowError("Stage 4 checkpoint action fixture is missing")


def _build_stage4_machine_payload(
    *,
    config: Stage4Config,
    runtime: Stage4RuntimeBinding,
    audit: Stage4MachineAuditBundle,
    checkpoint_artifacts: dict[str, bytes],
) -> _Stage4MachinePayload:
    _validate_stage4_machine_audit(config=config, runtime=runtime, audit=audit)
    if set(checkpoint_artifacts) != set(STAGE4_CHECKPOINT_ARTIFACTS) or any(
        not isinstance(value, bytes) or not value
        for value in checkpoint_artifacts.values()
    ):
        raise Stage4WorkflowError("Stage 4 checkpoint artifact path set drift")
    source = runtime.payload["execution_source_identity"]
    environment = runtime.payload["execution_environment_identity"]
    git_identity = runtime.payload["execution_git_identity"]
    authority = runtime.payload["execution_stage3_authority"]
    stage1_binding = runtime.payload["stage1_config_binding"]
    assert isinstance(source, dict)
    assert isinstance(environment, dict)
    assert isinstance(git_identity, dict)
    assert isinstance(authority, dict)
    assert isinstance(stage1_binding, dict)
    updates = list(audit.updates)
    latest = updates[-1]
    latest_checkpoint = latest["checkpoint"]
    assert isinstance(latest_checkpoint, dict)
    acceptance_items = {
        "three_consecutive_smoke_updates": len(updates) == 3,
        "fixed_8x128_trainable_layout": all(
            update["collection"]["per_env_trainable_counts"] == [128] * 8
            for update in updates
        ),
        "rollout_snapshot_count_1024": all(
            update["collection"]["snapshot_count"] == 1024
            for update in updates
        ),
        "gae_finite_normalized": all(update["gae"]["finite"] is True for update in updates),
        "initial_joint_ratio_matches_one": all(
            update["ppo"]["initial_ratio_max_abs_error"] <= 1.0e-5
            for update in updates
        ),
        "ppo_losses_entropy_and_kl_finite": True,
        "post_clip_grad_norm_bounded": all(
            update["ppo"]["grad_post_clip_norm_max"] <= 0.500001
            for update in updates
        ),
        "optimizer_changed_parameters": all(
            update["ppo"]["parameter_change_l2"] > 0.0 for update in updates
        ),
        "rollout_batches_consumed_and_cleared": all(
            update["buffer"] == {"consumed": True, "cleared": True}
            for update in updates
        ),
        "checkpoint_saved_and_last_complete_loaded": all(
            update["checkpoint"]["loaded_last_complete_step"]
            == update["update_step"]
            for update in updates
        ),
        "checkpoint_action_bit_exact": all(
            update["checkpoint"]["deterministic_action_bit_exact"] is True
            for update in updates
        ),
        "checkpoint_training_resume": (
            audit.training_resume["next_update_executed"] is True
            and audit.training_resume["continued_from_current_state"] is True
        ),
        "tiny_overfit_three_seeds": audit.tiny_overfit["passed"] is True,
        "cuda_fp32_no_amp": environment["compute_dtype"] == "float32"
        and environment["amp_enabled"] is False,
        "spawn_workers_main_process_inference": all(
            update["collection"]["inference_main_process_only"] is True
            for update in updates
        ),
    }
    if not all(acceptance_items.values()):
        raise Stage4WorkflowError("Stage 4 machine acceptance item failed")
    summary: dict[str, object] = {
        "schema_version": "ppo_highres_frontier_stage4_summary/v1",
        "goal_id": config.goal_id,
        "stage_id": config.stage_id,
        "run_id": runtime.payload["run_id"],
        "state": "machine_passed",
        "config_sha256": runtime.config_sha256,
        "execution_source_set_sha256": source["source_set_sha256"],
        "execution_environment_sha256": _canonical_sha256(environment),
        "prospective_git_tree": git_identity["prospective_git_tree"],
        "stage3_gate_sha256": authority["gate_sha256"],
        "stage3_approval_sha256": authority["approval_sha256"],
        "stage3_review_sha256": authority["review_sha256"],
        "stage1_config_sha256": stage1_binding["sha256"],
        "smoke_update_count": len(updates),
        "trainable_transitions_per_update": 1024,
        "latest_checkpoint_sha256": latest_checkpoint["checkpoint_sha256"],
        "latest_checkpoint_policy_state_sha256": latest_checkpoint[
            "policy_state_sha256"
        ],
        "tiny_overfit": {
            "context_count": audit.tiny_overfit["context_count"],
            "seed_count": len(audit.tiny_overfit["seeds"]),
            "passed": True,
        },
        "training_resume": {
            "resumed_from_update_step": audit.training_resume[
                "resume_receipt"
            ]["update_step"],
            "next_update_step": audit.training_resume["next_update_step"],
            "continued_from_current_state": audit.training_resume[
                "continued_from_current_state"
            ],
        },
        "acceptance": {
            "policy": "rollout_ppo_update_smoke_acceptance/v1",
            "passed": True,
            "items": acceptance_items,
        },
        "checkpoint_state": "stage4_machine_evidence_checkpoint_only/v1",
        "claim_boundary": "ppo_training_system_closure_no_task_advantage/v1",
    }
    if not finite_tree(summary):
        raise Stage4WorkflowError("Stage 4 summary contains non-finite data")
    routing = {
        "schema_version": "ppo_highres_frontier_stage4_routing/v1",
        "run_id": runtime.payload["run_id"],
        "route": "awaiting_independent_review",
        "authorized_next_stage": config.authorized_next_stage,
        "independent_review_required": True,
        "human_approval_required": True,
        "next_stage_entered": False,
    }
    report = (
        "# Stage 4 machine report\n\n"
        f"- Run ID: `{runtime.payload['run_id']}`\n"
        "- Completed 3 consecutive real Smoke PPO updates, each with 8 x 128 "
        "trainable transitions.\n"
        "- Completed the 4-context, 3-seed tiny-overfit acceptance.\n"
        "- Every update saved and loaded a complete checkpoint and reproduced "
        "the deterministic action bit-exactly.\n"
        "- A fresh trainer resumed update 1, applied normalizer/sampler/eight-worker "
        "state, and executed update 2 from restored current observations.\n"
        "- This proves PPO training-system closure only; it establishes no task "
        "performance advantage and publishes no checkpoint.\n"
        "- State: `machine_passed`; route: `awaiting_independent_review`.\n"
        "- No Stage 4 review, approval, gate, executor, or canary artifact was produced.\n"
    ).encode("utf-8")
    metrics = tuple(
        {
            "kind": "smoke_update",
            "update_step": update["update_step"],
            "trainable_transition_count": update["collection"][
                "trainable_transition_count"
            ],
            **update["ppo"],
        }
        for update in updates
    )
    progress = tuple(
        {
            "schema_version": "stage4_training_progress/v1",
            "update_id": update["update_step"],
            "total_updates": 3,
            "rollout_steps_collected": 1024,
            "rollout_batch_size": 1024,
            "ppo_epochs": config.ppo.ppo_epochs,
            "optimizer_steps": update["ppo"]["optimizer_steps"],
            "approx_kl": update["ppo"]["approx_kl"],
            "checkpoint_sha256": update["checkpoint"]["checkpoint_sha256"],
        }
        for update in updates
    )
    execution_identity = {
        "schema_version": "ppo_highres_frontier_stage4_execution_identity_audit/v1",
        "source": source,
        "environment": environment,
        "git": git_identity,
        "stage1_config": stage1_binding,
    }
    artifacts: dict[str, bytes] = {
        "config.json": runtime.config_bytes,
        "summary.json": ArtifactStore.canonical_json_bytes(summary),
        "routing.json": ArtifactStore.canonical_json_bytes(routing),
        "report.md": report,
        "metrics.jsonl": _canonical_jsonl_bytes(metrics),
        "phase-state.jsonl": _canonical_jsonl_bytes(
            (
                {
                    "state": "machine_passed",
                    "run_id": runtime.payload["run_id"],
                },
                {
                    "state": "awaiting_independent_review",
                    "run_id": runtime.payload["run_id"],
                },
            )
        ),
        "training_progress.jsonl": _canonical_jsonl_bytes(progress),
        "evidence/smoke_update_audit.json": ArtifactStore.canonical_json_bytes(
            {
                "schema_version": "stage4_three_smoke_updates/v1",
                "updates": updates,
            }
        ),
        "evidence/rollout_snapshot_audit.json": ArtifactStore.canonical_json_bytes(
            {
                "schema_version": "stage4_rollout_snapshot_audit/v1",
                "updates": [update["collection"] for update in updates],
            }
        ),
        "evidence/gae_audit.json": ArtifactStore.canonical_json_bytes(
            {
                "schema_version": "stage4_gae_audit/v1",
                "updates": [update["gae"] for update in updates],
            }
        ),
        "evidence/ppo_loss_audit.json": ArtifactStore.canonical_json_bytes(
            {
                "schema_version": "stage4_ppo_loss_audit/v1",
                "updates": [update["ppo"] for update in updates],
            }
        ),
        "evidence/logprob_recompute_audit.json": ArtifactStore.canonical_json_bytes(
            {
                "schema_version": "stage4_logprob_recompute_audit/v1",
                "old_factorization_exact": [
                    update["collection"]["old_logprob_factorization_exact"]
                    for update in updates
                ],
                "initial_ratio_max_abs_error": [
                    update["ppo"]["initial_ratio_max_abs_error"]
                    for update in updates
                ],
            }
        ),
        "evidence/optimizer_step_audit.json": ArtifactStore.canonical_json_bytes(
            {
                "schema_version": "stage4_optimizer_step_audit/v1",
                "updates": [
                    {
                        key: update["ppo"][key]
                        for key in (
                            "optimizer_steps",
                            "planned_optimizer_steps",
                            "grad_pre_clip_norm_max",
                            "grad_post_clip_norm_max",
                            "parameter_change_l2",
                            "policy_state_sha256_before",
                            "policy_state_sha256_after",
                        )
                    }
                    for update in updates
                ],
            }
        ),
        "evidence/checkpoint_load_audit.json": ArtifactStore.canonical_json_bytes(
            {
                "schema_version": "stage4_checkpoint_load_audit/v1",
                "updates": [update["checkpoint"] for update in updates],
            }
        ),
        "evidence/training_resume_audit.json": ArtifactStore.canonical_json_bytes(
            audit.training_resume
        ),
        "evidence/checkpoint_action_fixture.npz": (
            audit.checkpoint_action_fixture_npz
        ),
        "evidence/tiny_overfit_audit.json": ArtifactStore.canonical_json_bytes(
            audit.tiny_overfit
        ),
        "evidence/stage3_authority_audit.json": ArtifactStore.canonical_json_bytes(
            authority
        ),
        "evidence/execution_identity_audit.json": ArtifactStore.canonical_json_bytes(
            execution_identity
        ),
        **checkpoint_artifacts,
    }
    if set(artifacts) != set(STAGE4_MANIFEST_BOUND_ARTIFACTS):
        raise Stage4WorkflowError("Stage 4 machine artifact path set drift")
    manifest = _manifest_for_artifacts(artifacts)
    return _Stage4MachinePayload(
        artifacts=artifacts,
        manifest=manifest,
        summary=summary,
    )


def stage4_source_identity(repo_root: str | Path) -> dict[str, object]:
    """Hash the exact Stage 4 implementation, tests, configs, runner, and specs."""

    repo = Path(repo_root).expanduser().resolve()
    paths: set[Path] = set((repo / "src/lunar_exploration_ppo").rglob("*.py"))
    paths.update((repo / "tests/ppo_highres_frontier").glob("test_stage4_*.py"))
    paths.add(repo / "tests/ppo_highres_frontier/test_stage1_stage4_contract.py")
    paths.update(
        {
            repo / "pyproject.toml",
            repo / "configs/ppo_highres_frontier_smoke_v1.json",
            repo / "configs/ppo_highres_frontier_stage4_v1.json",
            repo / "scripts/run_ppo_highres_frontier_stage4.py",
            repo
            / "docs/superpowers/specs/2026-07-09-ppo-highres-frontier-map-exploration-design.md",
            repo
            / "docs/superpowers/plans/2026-07-10-ppo-highres-frontier-map-exploration.md",
            repo / "docs/ppo-highres-frontier-stage4.md",
        }
    )
    digest = hashlib.sha256()
    entries: list[dict[str, object]] = []
    try:
        ordered = sorted(paths, key=lambda path: path.relative_to(repo).as_posix())
    except ValueError as exc:
        raise Stage4WorkflowError("Stage 4 source path escaped the repository") from exc
    for path in ordered:
        if not path.is_file():
            raise Stage4WorkflowError(f"Stage 4 source binding path is missing: {path}")
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
        "schema_version": "stage4_reviewed_source_set/v1",
        "source_set_sha256": digest.hexdigest(),
        "paths": entries,
    }


def stage4_environment_identity() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise Stage4WorkflowError("Stage 4 requires CUDA; CPU fallback is forbidden")
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    return {
        "schema_version": "ppo_highres_frontier_stage4_environment/v1",
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
        "multiprocessing_start_method": "spawn",
    }


def stage4_git_identity(repo_root: str | Path) -> dict[str, object]:
    try:
        identity = stage3_workflow.stage3_git_identity(
            repo_root,
            base_commit=STAGE3_COMMIT,
        )
    except Exception as exc:
        raise Stage4WorkflowError("Stage 4 prospective Git identity failed") from exc
    identity["schema_version"] = (
        "ppo_highres_frontier_stage4_prospective_git_tree/v1"
    )
    return identity


def run_stage4_workflow(
    *,
    config_path: str | Path,
    run_id: str,
    stage3_gate_path: str | Path,
    base_output_root: str | Path | None = None,
) -> Stage4WorkflowResult:
    """Run real Stage 4 machine acceptance and stop at independent review."""

    validate_run_id(run_id)
    repo = Path(__file__).resolve().parents[3]
    config_path_resolved = Path(config_path).expanduser().resolve()
    try:
        config_snapshot = FrozenFileSnapshot.capture(
            config_path_resolved,
            "Stage 4 repository config",
            authority_root=repo,
        )
        config = Stage4Config.model_validate_json(config_snapshot.payload)
    except (Stage1WorkflowError, ValueError) as exc:
        raise Stage4WorkflowError("Stage 4 repository config snapshot is invalid") from exc
    stage1_path = (repo / config.stage1_config_path).resolve()
    try:
        stage1_snapshot = FrozenFileSnapshot.capture(
            stage1_path,
            "Stage 1 Smoke config",
            authority_root=repo,
        )
        stage1_config = Stage1Config.model_validate_json(stage1_snapshot.payload)
    except (Stage1WorkflowError, ValueError) as exc:
        raise Stage4WorkflowError("Stage 1 Smoke config snapshot is invalid") from exc
    if stage1_snapshot.sha256 != config.stage1_config_sha256:
        raise Stage4WorkflowError("Stage 1 Smoke config SHA-256 drift")
    base = Path(base_output_root) if base_output_root is not None else Path(config.output_root)
    run_root = (base / run_id).expanduser().resolve()
    if run_root.drive.upper() != "D:":
        raise Stage4WorkflowError("Stage 4 runtime output must be on D drive")
    if run_root.exists():
        raise Stage4WorkflowError("Stage 4 run root already exists")

    source_identity = stage4_source_identity(repo)
    environment_identity = stage4_environment_identity()
    git_identity = stage4_git_identity(repo)
    authority_handle = verify_frozen_stage3_authority(
        gate_path=stage3_gate_path,
        repo_root=repo,
    )
    stage1_binding = {
        "path": str(stage1_snapshot.path),
        "sha256": stage1_snapshot.sha256,
    }
    runtime = _build_stage4_runtime_binding(
        config=config,
        run_id=run_id,
        source_identity=source_identity,
        environment_identity=environment_identity,
        git_identity=git_identity,
        stage3_authority=authority_handle.identity,
        stage1_config_binding=stage1_binding,
    )
    prepare_unique_run_root(run_root)
    stage = run_root / "s4"
    stage.mkdir(exist_ok=False)
    (stage / "evidence").mkdir(exist_ok=False)
    checkpoint_root = stage / "checkpoints"
    checkpoint_root.mkdir(exist_ok=False)
    try:
        audit = run_stage4_smoke_acceptance(
            config=config,
            stage1_config=stage1_config,
            checkpoint_root=checkpoint_root,
            config_sha256=runtime.config_sha256,
            lineage=_checkpoint_lineage(runtime),
        )
    except Stage4MachineError as exc:
        raise Stage4WorkflowError("Stage 4 real machine acceptance failed") from exc
    checkpoint_artifacts = {
        relative: (stage / relative).read_bytes()
        for relative in STAGE4_CHECKPOINT_ARTIFACTS
    }
    if (
        stage4_source_identity(repo) != source_identity
        or stage4_environment_identity() != environment_identity
        or stage4_git_identity(repo) != git_identity
    ):
        raise Stage4WorkflowError("Stage 4 source, environment, or Git drifted")
    authority_handle.require_current("Stage 4 controller Stage 3 authority")
    try:
        config_snapshot.require_current("Stage 4 repository config")
        stage1_snapshot.require_current("Stage 4 Stage 1 config")
    except Stage1WorkflowError as exc:
        raise Stage4WorkflowError("Stage 4 config changed during execution") from exc
    payload = _build_stage4_machine_payload(
        config=config,
        runtime=runtime,
        audit=audit,
        checkpoint_artifacts=checkpoint_artifacts,
    )
    store = ArtifactStore(stage)
    for relative in STAGE4_MANIFEST_BOUND_ARTIFACTS:
        if relative not in STAGE4_CHECKPOINT_ARTIFACTS:
            store.write_bytes(relative, payload.artifacts[relative])
    store.write_json("manifest.json", payload.manifest)
    verification = verify_stage4_machine_run(
        stage_root=stage,
        repo_root=repo,
        stage3_gate_path=stage3_gate_path,
    )
    authority_handle.require_current("Stage 4 controller final Stage 3 authority")
    try:
        config_snapshot.require_current("Stage 4 controller final repository config")
        stage1_snapshot.require_current("Stage 4 controller final Stage 1 config")
    except Stage1WorkflowError as exc:
        raise Stage4WorkflowError("Stage 4 config changed before return") from exc
    verification.require_current("Stage 4 controller final verification")
    return Stage4WorkflowResult(
        run_id=run_id,
        stage_root=stage,
        summary=payload.summary,
        manifest=payload.manifest,
    )


def verify_stage4_machine_run(
    *,
    stage_root: str | Path,
    repo_root: str | Path,
    stage3_gate_path: str | Path,
) -> Stage4MachineVerificationHandle:
    stage = Path(stage_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    if stage.name != "s4" or not stage.is_dir():
        raise Stage4WorkflowError("Stage 4 root must be canonical <run>/s4")
    if {path.name for path in stage.iterdir()} != set(STAGE4_ROOT_ARTIFACTS):
        raise Stage4WorkflowError("Stage 4 root artifact set drift")
    if any((stage / name).exists() for name in ("review.json", "approval.json", "gate.json")):
        raise Stage4WorkflowError("Stage 4 machine root contains authority artifact")
    evidence = stage / "evidence"
    if (
        not evidence.is_dir()
        or {path.name for path in evidence.iterdir()}
        != {Path(path).name for path in STAGE4_EVIDENCE_ARTIFACTS}
    ):
        raise Stage4WorkflowError("Stage 4 evidence artifact set drift")
    if _checkpoint_relative_files(stage) != set(STAGE4_CHECKPOINT_ARTIFACTS):
        raise Stage4WorkflowError("Stage 4 checkpoint artifact set drift")

    try:
        manifest_snapshot = FrozenFileSnapshot.capture(
            stage / "manifest.json",
            "Stage 4 manifest",
            authority_root=stage,
        )
        manifest = _strict_canonical(manifest_snapshot, "Stage 4 manifest")
        artifact_snapshots = stage3_workflow._capture_manifest_artifact_snapshots(
            stage=stage,
            manifest=manifest,
            expected_paths=set(STAGE4_MANIFEST_BOUND_ARTIFACTS),
            label="Stage 4",
        )
    except Exception as exc:
        raise Stage4WorkflowError("Stage 4 manifest graph drift") from exc
    runtime_payload = _strict_canonical(
        artifact_snapshots["config.json"],
        "Stage 4 config",
    )
    source_identity = runtime_payload.pop("execution_source_identity", None)
    environment_identity = runtime_payload.pop("execution_environment_identity", None)
    git_identity = runtime_payload.pop("execution_git_identity", None)
    recorded_authority = runtime_payload.pop("execution_stage3_authority", None)
    stage1_binding = runtime_payload.pop("stage1_config_binding", None)
    try:
        machine_config = Stage4Config.model_validate(runtime_payload)
        repository_config = load_stage4_config(
            repo / "configs/ppo_highres_frontier_stage4_v1.json"
        )
    except Exception as exc:
        raise Stage4WorkflowError("Stage 4 config binding drift") from exc
    if (
        machine_config.model_dump(exclude={"run_id"})
        != repository_config.model_dump(exclude={"run_id"})
        or machine_config.run_id != stage.parent.name
        or not isinstance(source_identity, dict)
        or not isinstance(environment_identity, dict)
        or not isinstance(git_identity, dict)
        or not isinstance(recorded_authority, dict)
        or not isinstance(stage1_binding, dict)
    ):
        raise Stage4WorkflowError("Stage 4 runtime config semantics drift")
    stage1_path = (repo / machine_config.stage1_config_path).resolve()
    try:
        stage1_snapshot = FrozenFileSnapshot.capture(
            stage1_path,
            "Stage 4 verifier Stage 1 config",
            authority_root=repo,
        )
    except Stage1WorkflowError as exc:
        raise Stage4WorkflowError("Stage 1 config binding cannot be captured") from exc
    current_source = stage4_source_identity(repo)
    current_environment = stage4_environment_identity()
    current_git = stage4_git_identity(repo)
    authority_handle = verify_frozen_stage3_authority(
        gate_path=stage3_gate_path,
        repo_root=repo,
    )
    current_stage1_binding = {
        "path": str(stage1_snapshot.path),
        "sha256": stage1_snapshot.sha256,
    }
    if (
        source_identity != current_source
        or environment_identity != current_environment
        or git_identity != current_git
        or recorded_authority != authority_handle.identity
        or stage1_binding != current_stage1_binding
    ):
        raise Stage4WorkflowError("Stage 4 execution identity or authority drift")
    runtime = _build_stage4_runtime_binding(
        config=repository_config,
        run_id=stage.parent.name,
        source_identity=current_source,
        environment_identity=current_environment,
        git_identity=current_git,
        stage3_authority=authority_handle.identity,
        stage1_config_binding=current_stage1_binding,
    )
    if artifact_snapshots["config.json"].payload != runtime.config_bytes:
        raise Stage4WorkflowError("Stage 4 canonical runtime config drift")
    smoke = _strict_canonical(
        artifact_snapshots["evidence/smoke_update_audit.json"],
        "Stage 4 Smoke audit",
    )
    tiny = _strict_canonical(
        artifact_snapshots["evidence/tiny_overfit_audit.json"],
        "Stage 4 tiny-overfit audit",
    )
    training_resume = _strict_canonical(
        artifact_snapshots["evidence/training_resume_audit.json"],
        "Stage 4 training resume audit",
    )
    if set(smoke) != {"schema_version", "updates"} or smoke.get(
        "schema_version"
    ) != "stage4_three_smoke_updates/v1" or not isinstance(smoke.get("updates"), list):
        raise Stage4WorkflowError("Stage 4 Smoke audit schema drift")
    audit = Stage4MachineAuditBundle(
        updates=tuple(smoke["updates"]),
        tiny_overfit=tiny,
        checkpoint_action_fixture_npz=artifact_snapshots[
            "evidence/checkpoint_action_fixture.npz"
        ].payload,
        training_resume=training_resume,
    )
    checkpoint_artifacts = {
        relative: artifact_snapshots[relative].payload
        for relative in STAGE4_CHECKPOINT_ARTIFACTS
    }
    expected = _build_stage4_machine_payload(
        config=repository_config,
        runtime=runtime,
        audit=audit,
        checkpoint_artifacts=checkpoint_artifacts,
    )
    for relative in STAGE4_MANIFEST_BOUND_ARTIFACTS:
        if artifact_snapshots[relative].payload != expected.artifacts[relative]:
            raise Stage4WorkflowError(f"Stage 4 deterministic artifact drift: {relative}")
    if manifest_snapshot.payload != ArtifactStore.canonical_json_bytes(expected.manifest):
        raise Stage4WorkflowError("Stage 4 manifest semantic drift")

    policy = CrossAttentionFrontierPolicy().to(device="cuda", dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=3.0e-4,
        eps=1.0e-5,
        weight_decay=1.0e-4,
    )
    loaded = CheckpointManager(stage / "checkpoints").load_last_complete(
        policy=policy,
        optimizer=optimizer,
        expected_config_sha256=runtime.config_sha256,
        expected_lineage=_checkpoint_lineage(runtime),
    )
    observation, expected_action = load_action_fixture_npz(
        artifact_snapshots["evidence/checkpoint_action_fixture.npz"].payload
    )
    actual_action = deterministic_action_record(policy, observation, device="cuda")
    latest_checkpoint = audit.updates[-1]["checkpoint"]
    if (
        loaded.update_step != 3
        or actual_action != expected_action
        or not isinstance(latest_checkpoint, dict)
        or loaded.checkpoint_sha256 != latest_checkpoint.get("checkpoint_sha256")
        or loaded.manifest_sha256 != latest_checkpoint.get("manifest_sha256")
        or loaded.policy_state_sha256 != latest_checkpoint.get("policy_state_sha256")
    ):
        raise Stage4WorkflowError("Stage 4 checkpoint replay drift")
    if (
        stage4_source_identity(repo) != current_source
        or stage4_environment_identity() != current_environment
        or stage4_git_identity(repo) != current_git
    ):
        raise Stage4WorkflowError("Stage 4 identity changed during verification")
    authority_handle.require_current("Stage 4 verifier final Stage 3 authority")
    try:
        stage1_snapshot.require_current("Stage 4 verifier final Stage 1 config")
    except Stage1WorkflowError as exc:
        raise Stage4WorkflowError("Stage 1 config changed during verification") from exc
    verification = Stage4MachineVerificationHandle(
        stage_root=stage,
        manifest_snapshot=manifest_snapshot,
        artifacts=tuple(sorted(artifact_snapshots.items())),
        stage3_authority=authority_handle,
    )
    verification.require_current()
    return verification


def _checkpoint_relative_files(stage: Path) -> set[str]:
    root = stage / "checkpoints"
    if not root.is_dir():
        return set()
    return {
        path.relative_to(stage).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _manifest_for_artifacts(artifacts: Mapping[str, bytes]) -> dict[str, object]:
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


def _canonical_jsonl_bytes(values: Sequence[dict[str, object]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for value in values
    )


def _strict_canonical(
    snapshot: FrozenFileSnapshot,
    label: str,
) -> dict[str, object]:
    try:
        value = strict_json_object_from_bytes(snapshot.payload, label)
    except Stage1WorkflowError as exc:
        raise Stage4WorkflowError(f"{label} JSON is invalid") from exc
    if snapshot.payload != ArtifactStore.canonical_json_bytes(value):
        raise Stage4WorkflowError(f"{label} JSON is not canonical")
    return value


def _canonical_uuid(value: object, label: str) -> str:
    try:
        canonical = str(uuid.UUID(str(value)))
    except ValueError as exc:
        raise Stage4WorkflowError(f"Stage 3 {label} UUID is invalid") from exc
    if canonical != value:
        raise Stage4WorkflowError(f"Stage 3 {label} UUID is not canonical")
    return canonical


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise Stage4WorkflowError(f"Stage 3 {label} is not UTC Z form")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise Stage4WorkflowError(f"Stage 3 {label} is invalid") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise Stage4WorkflowError(f"Stage 3 {label} is not UTC")
    return parsed


def _git_bytes(repo: Path, arguments: Sequence[str]) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise Stage4WorkflowError(
            f"Git command failed: {' '.join(arguments)}: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return completed.stdout


def _git_text(repo: Path, arguments: Sequence[str]) -> str:
    return _git_bytes(repo, arguments).decode("utf-8", errors="strict").strip()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(ArtifactStore.canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_exact_int_list(
    value: object,
    *,
    length: int,
    minimum: int,
) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(type(item) is int and item >= minimum for item in value)
    )


__all__ = [
    "ABSENT_AUTHORITY_SHA256",
    "FrozenStage3AuthorityHandle",
    "STAGE3_APPROVAL_SHA256",
    "STAGE3_GATE_SHA256",
    "STAGE3_REVIEW_SHA256",
    "STAGE4_CHECKPOINT_ARTIFACTS",
    "STAGE4_EVIDENCE_ARTIFACTS",
    "STAGE4_MACHINE_ARTIFACTS",
    "STAGE4_MANIFEST_BOUND_ARTIFACTS",
    "STAGE4_ROOT_ARTIFACTS",
    "Stage4MachineAuditBundle",
    "Stage4MachineVerificationHandle",
    "Stage4RuntimeBinding",
    "Stage4WorkflowResult",
    "Stage4WorkflowError",
    "run_stage4_smoke_acceptance",
    "run_stage4_workflow",
    "stage4_environment_identity",
    "stage4_git_identity",
    "stage4_source_identity",
    "verify_frozen_stage3_authority",
    "verify_stage4_machine_run",
]
