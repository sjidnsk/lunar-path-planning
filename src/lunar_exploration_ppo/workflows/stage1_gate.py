"""Fail-closed Stage 1 approval bindings and verified gate persistence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lunar_exploration_ppo.configs.schema import (
    FOUNDATION_BRANCH,
    FOUNDATION_GIT_COMMON_DIR,
    FOUNDATION_GIT_DIR,
    FOUNDATION_WORKTREE_ROOT,
)
from lunar_exploration_ppo.configs.stage1 import Stage1Config
from lunar_exploration_ppo.env.terrain_proxy import scaled_count_bounds
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.workflows.gates import NO_CHECKPOINT_SENTINEL
from lunar_exploration_ppo.workflows.stage1_artifacts import (
    FrozenDirectorySnapshot,
    FrozenFileSnapshot,
    STAGE1_MACHINE_ARTIFACTS,
    Stage1WorkflowError,
    load_stage1_machine_config,
    load_stage1_machine_config_bytes,
    strict_json_object_from_bytes,
    verify_manifest_entries_bytes,
)
from lunar_exploration_ppo.workflows.stage1_review import (
    build_stage1_approval_challenge,
    capture_stage1_foundation_environment_files,
    verify_stage1_review_evidence,
    verify_stage1_review_evidence_snapshots,
)
from lunar_exploration_ppo.workflows.stage1_source import (
    ReviewedSourceError,
    Stage1SourceSnapshot,
    compute_stage1_reviewed_source_set_sha256,
    compute_stage1_source_identity_from_snapshot,
    resolve_stage1_import_identity,
)


STAGE1_GOAL_ID: Final = "ppo-highres-frontier-map-exploration"
STAGE1_STAGE_ID: Final = "ppo_highres_frontier_stage1_smoke_environment/v1"
STAGE1_AUTHORIZED_NEXT_STAGE: Final = "ppo_highres_frontier_stage2_observation_frontier/v1"
STAGE1_NO_CHECKPOINT_SENTINEL: Final = "stage1_no_checkpoint/v1"
FOUNDATION_AUTHORIZED_STAGE1: Final = "ppo_highres_frontier_stage1_smoke_environment/v1"
STAGE1_APPROVAL_ABSENT_HASH: Final = hashlib.sha256(
    b"ppo_highres_frontier_stage1_human_approval_absent/v1"
).hexdigest()
STAGE1_APPROVAL_TEXT: Final = "批准 Stage 1 Gate"
STAGE1_APPROVAL_TEXT_SHA256: Final = hashlib.sha256(
    STAGE1_APPROVAL_TEXT.encode("utf-8")
).hexdigest()
Stage1GateState = Literal[
    "machine_passed",
    "awaiting_independent_review",
    "awaiting_human_approval",
    "approved",
    "next_stage",
]
_STAGE1_STATE_SEQUENCE: Final[tuple[Stage1GateState, ...]] = (
    "machine_passed",
    "awaiting_independent_review",
    "awaiting_human_approval",
    "approved",
    "next_stage",
)


class Stage1GateError(RuntimeError):
    """Raised when a Stage 1 approval binding fails closed."""


class Stage1GateContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repo_root: Path
    stage_root: Path
    foundation_gate: Path
    foundation_environment_manifest: Path
    run_id: str = Field(min_length=1)


class Stage1GateBindings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    goal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    git_tree_hash: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_data_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    foundation_gate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    foundation_environment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_source_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_state: Literal["absent", "approved"]
    approval_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    authorized_next_stage: Literal["ppo_highres_frontier_stage2_observation_frontier/v1"]


@dataclass(frozen=True, slots=True)
class _VerifiedStage1Review:
    review_hash: str
    approval_challenge: str
    review_recorded_at_utc: str


@dataclass(frozen=True, slots=True)
class _GateAuthoritySnapshot:
    target_state: Stage1GateState
    stage: FrozenDirectorySnapshot
    source: Stage1SourceSnapshot
    foundation_gate: FrozenFileSnapshot
    foundation_environment: FrozenFileSnapshot
    foundation_environment_files: tuple[tuple[str, FrozenFileSnapshot], ...]
    review_report: FrozenFileSnapshot
    review_package: FrozenFileSnapshot


def record_stage1_human_approval(
    *,
    context: Stage1GateContext | None = None,
    stage_root: str | Path | None = None,
    thread_id: str,
    user_turn_id: str,
    approval_text: str,
    user_turn_timestamp_utc: str,
) -> Path:
    """Record audit evidence copied from a real Codex user turn.

    The controller is expected to obtain thread_id, user_turn_id, exact text, and
    timestamp from read_thread. This is strict audit binding, but without a platform
    signature verification API it is not cryptographically unforgeable.
    """

    if not isinstance(context, Stage1GateContext):
        raise Stage1GateError("Stage 1 approval requires a verified gate context")
    stage = context.stage_root.expanduser().resolve()
    if stage_root is not None and Path(stage_root).expanduser().resolve() != stage:
        raise Stage1GateError("Stage 1 approval stage root differs from verified context")
    if stage.name != "s1" or not stage.is_dir():
        raise Stage1GateError("Stage 1 approval requires a canonical s1 stage root")
    if approval_text != STAGE1_APPROVAL_TEXT:
        raise Stage1GateError("Stage 1 exact approval text is required")
    store = ArtifactStore(stage)
    approval_path = store.resolve("approval.json")
    if approval_path.exists() or (stage / "gate.json").exists():
        raise Stage1GateError("Stage 1 approval or gate evidence already exists")
    first_bindings = Stage1GateBindingVerifier.compute(context, target_state="awaiting_human_approval")
    pre_write_bindings = Stage1GateBindingVerifier.compute(context, target_state="awaiting_human_approval")
    if first_bindings != pre_write_bindings:
        raise Stage1GateError("Stage 1 approval binding snapshot drifted before write")
    review_path = stage / "review.json"
    try:
        review_bytes = review_path.read_bytes()
        review = json.loads(review_bytes.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage1GateError("Stage 1 approval review binding source is invalid") from exc
    if not isinstance(review, dict):
        raise Stage1GateError("Stage 1 approval review binding source must be a JSON object")
    review_hash = hashlib.sha256(review_bytes).hexdigest()
    if review_hash != pre_write_bindings.review_hash:
        raise Stage1GateError("Stage 1 approval review binding drifted after verification")
    if (
        review.get("schema_version") != "ppo_highres_frontier_stage1_independent_review/v2"
        or review.get("state") != "awaiting_human_approval"
        or review.get("run_id") != stage.parent.name
    ):
        raise Stage1GateError("Stage 1 approval requires a valid review record")
    evidence = {
        "schema_version": "ppo_highres_frontier_stage1_human_approval/v2",
        "state": "approved",
        "authority_kind": "codex_user_message/v1",
        "actor": "user",
        "thread_id": thread_id,
        "user_turn_id": user_turn_id,
        "approval_text": approval_text,
        "approval_text_utf8_sha256": hashlib.sha256(approval_text.encode("utf-8")).hexdigest(),
        "approval_timestamp_utc": user_turn_timestamp_utc,
        "goal_id": review.get("goal_id"),
        "stage_id": review.get("stage_id"),
        "run_id": review.get("run_id"),
        "reviewed_git_tree": review.get("reviewed_git_tree"),
        "review_hash": review_hash,
        "approval_challenge": review.get("approval_challenge"),
    }
    _validate_stage1_approval_evidence(
        evidence,
        run_id=str(review.get("run_id", "")),
        reviewed_tree=str(review.get("reviewed_git_tree", "")),
        review_hash=review_hash,
        approval_challenge=str(review.get("approval_challenge", "")),
        review_recorded_at_utc=str(review.get("review_recorded_at_utc", "")),
    )
    expected_approval_bytes = ArtifactStore.canonical_json_bytes(evidence)
    # Exclusive creation commits the append-only canonical authority.  Later
    # verification failures preserve whichever object currently occupies it.
    written_path = store.write_json_exclusive("approval.json", evidence)
    if written_path.resolve() != approval_path:
        raise Stage1GateError("Stage 1 approval writer returned an unexpected target path")
    post_write_bindings = Stage1GateBindingVerifier.compute(context, target_state="approved")
    if (
        pre_write_bindings.model_dump(exclude={"approval_state", "approval_hash"})
        != post_write_bindings.model_dump(exclude={"approval_state", "approval_hash"})
    ):
        raise Stage1GateError("Stage 1 approval binding drifted after write")
    written_approval_bytes = approval_path.read_bytes()
    if written_approval_bytes != expected_approval_bytes:
        raise Stage1GateError("Stage 1 approval evidence changed after exclusive write")
    written_approval_hash = hashlib.sha256(written_approval_bytes).hexdigest()
    if (
        post_write_bindings.approval_state != "approved"
        or post_write_bindings.approval_hash != written_approval_hash
    ):
        raise Stage1GateError("Stage 1 approval binding does not match written approval evidence")
    return approval_path


class Stage1GateBindingVerifier:
    @classmethod
    def compute(
        cls,
        context: Stage1GateContext,
        *,
        target_state: Stage1GateState = "awaiting_human_approval",
    ) -> Stage1GateBindings:
        if not isinstance(context, Stage1GateContext):
            raise Stage1GateError("Stage 1 gate requires a verified context")
        if target_state not in _STAGE1_STATE_SEQUENCE:
            raise Stage1GateError("Stage 1 gate target state is invalid")
        try:
            authority_snapshot = cls._capture_authority_snapshot(context, target_state)
            bindings = cls._compute_snapshot_bindings(context, authority_snapshot)
            cls._require_authority_snapshot_current(authority_snapshot)
        except Stage1GateError:
            raise
        except Stage1WorkflowError as exc:
            if "exact regular-file set" in str(exc):
                raise Stage1GateError("Stage 1 approval/state artifact set is not exact") from exc
            raise Stage1GateError("Stage 1 immutable authority snapshot verification failed") from exc
        except (OSError, ValueError, ValidationError, ReviewedSourceError) as exc:
            raise Stage1GateError("Stage 1 immutable authority snapshot verification failed") from exc
        return bindings

    @classmethod
    def _compute_snapshot_bindings(
        cls,
        context: Stage1GateContext,
        snapshot: _GateAuthoritySnapshot,
    ) -> Stage1GateBindings:
        stage = snapshot.stage
        config_snapshot = stage.file("config.json")
        config, machine_source_identity = load_stage1_machine_config_bytes(config_snapshot.payload)
        if config.goal_id != STAGE1_GOAL_ID or config.stage_id != STAGE1_STAGE_ID or config.run_id != context.run_id:
            raise Stage1GateError("Stage 1 config identity mismatch")
        config_hash = config_snapshot.sha256
        summary = strict_json_object_from_bytes(stage.file("summary.json").payload, "summary")
        routing = strict_json_object_from_bytes(stage.file("routing.json").payload, "routing")
        if (
            summary.get("state") != "machine_passed"
            or summary.get("run_id") != context.run_id
            or summary.get("goal_id") != STAGE1_GOAL_ID
            or summary.get("stage_id") != STAGE1_STAGE_ID
            or summary.get("config_hash") != config_hash
        ):
            raise Stage1GateError("Stage 1 summary state or config binding mismatch")
        if summary.get("execution_source_identity") != machine_source_identity:
            raise Stage1GateError("Stage 1 machine source identity declarations mismatch")
        if (
            routing.get("route") != "awaiting_independent_review"
            or routing.get("run_id") != context.run_id
            or routing.get("next_stage_entered") is not False
        ):
            raise Stage1GateError("Stage 1 routing state mismatch")
        cls._verify_phase_state_bytes(stage.file("phase-state.jsonl").payload, context.run_id)
        verify_manifest_entries_bytes(
            stage.file("manifest.json").payload,
            {name: stage.file(name) for name in STAGE1_MACHINE_ARTIFACTS},
        )
        manifest_hash = stage.file("manifest.json").sha256
        foundation_gate_hash, foundation_environment_hash = cls._verify_foundation_sources_snapshot(
            snapshot
        )
        if snapshot.source.base_commit != config.foundation_base_commit:
            raise Stage1GateError("Stage 1 reviewed source Foundation base commit mismatch")
        current_source_identity = compute_stage1_source_identity_from_snapshot(snapshot.source)
        execution_source_identity_hash, git_tree_hash = cls._verify_execution_source_identity_snapshot(
            machine_source_identity,
            current_source_identity,
        )
        verified_review = cls._verify_review_snapshot(
            snapshot,
            run_id=context.run_id,
            config_hash=config_hash,
            manifest_hash=manifest_hash,
            scenario_hash=str(summary.get("scenario_hash")),
            foundation_gate_hash=foundation_gate_hash,
            foundation_environment_hash=foundation_environment_hash,
            foundation_base_commit=config.foundation_base_commit,
            reviewed_git_tree=git_tree_hash,
            execution_source_identity=machine_source_identity,
            execution_source_identity_hash=execution_source_identity_hash,
        )
        approval_state, approval_hash = cls._verify_stage_artifact_snapshot(
            snapshot,
            context.run_id,
            git_tree_hash,
            review_hash=verified_review.review_hash,
            approval_challenge=verified_review.approval_challenge,
            review_recorded_at_utc=verified_review.review_recorded_at_utc,
            target_state=snapshot.target_state,
        )
        scenario_data_hash = cls._scenario_identity_hash(summary, config)
        return Stage1GateBindings(
            goal_hash=cls._identity_hash("goal", STAGE1_GOAL_ID),
            stage_hash=cls._identity_hash("stage", STAGE1_STAGE_ID),
            git_tree_hash=git_tree_hash,
            config_hash=config_hash,
            scenario_data_hash=scenario_data_hash,
            foundation_gate_hash=foundation_gate_hash,
            foundation_environment_hash=foundation_environment_hash,
            checkpoint_hash=hashlib.sha256(STAGE1_NO_CHECKPOINT_SENTINEL.encode("utf-8")).hexdigest(),
            review_hash=verified_review.review_hash,
            manifest_hash=manifest_hash,
            execution_source_identity_hash=execution_source_identity_hash,
            approval_state=approval_state,
            approval_hash=approval_hash,
            authorized_next_stage=config.authorized_next_stage,
        )

    @classmethod
    def _capture_authority_snapshot(
        cls,
        context: Stage1GateContext,
        target_state: Stage1GateState,
    ) -> _GateAuthoritySnapshot:
        repo_root = Path(os.path.abspath(os.fspath(context.repo_root.expanduser())))
        stage = Path(os.path.abspath(os.fspath(context.stage_root.expanduser())))
        if stage.name != "s1" or stage.parent.name != context.run_id:
            raise Stage1GateError("Stage 1 gate stage root or run_id is invalid")
        stage_snapshot = FrozenDirectorySnapshot.capture_one_of_exact_regular_file_sets(
            stage,
            cls._expected_stage_member_sets(target_state),
            "Stage 1 gate authority",
        )
        config, _machine_source_identity = load_stage1_machine_config_bytes(
            stage_snapshot.file("config.json").payload
        )
        source_snapshot = Stage1SourceSnapshot.capture(
            repo_root,
            base_commit=config.foundation_base_commit,
            mode="gate_committed",
        )
        foundation_gate = FrozenFileSnapshot.capture(context.foundation_gate, "Foundation gate")
        foundation_environment = FrozenFileSnapshot.capture(
            context.foundation_environment_manifest,
            "Foundation environment manifest",
        )
        foundation_environment_files = capture_stage1_foundation_environment_files(
            foundation_environment
        )
        review = strict_json_object_from_bytes(
            stage_snapshot.file("review.json").payload,
            "review",
        )
        review_report = cls._capture_review_binding_source(
            review.get("review_report"),
            "review report",
        )
        review_package = cls._capture_review_binding_source(
            review.get("review_package"),
            "review package",
        )
        snapshot = _GateAuthoritySnapshot(
            target_state=target_state,
            stage=stage_snapshot,
            source=source_snapshot,
            foundation_gate=foundation_gate,
            foundation_environment=foundation_environment,
            foundation_environment_files=foundation_environment_files,
            review_report=review_report,
            review_package=review_package,
        )
        cls._require_authority_snapshot_current(snapshot)
        return snapshot

    @classmethod
    def _expected_stage_member_sets(
        cls,
        target_state: Stage1GateState,
    ) -> tuple[frozenset[str], ...]:
        base = frozenset((*STAGE1_MACHINE_ARTIFACTS, "manifest.json", "review.json"))
        if target_state in {"machine_passed", "awaiting_independent_review", "awaiting_human_approval"}:
            return (base,)
        approved = base | {"approval.json"}
        if target_state == "approved":
            return (approved,)
        return (approved, approved | {"gate.json"})

    @staticmethod
    def _capture_review_binding_source(value: object, label: str) -> FrozenFileSnapshot:
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            raise Stage1GateError(f"Stage 1 {label} binding is invalid")
        return FrozenFileSnapshot.capture(value["path"], label)

    @staticmethod
    def _require_authority_snapshot_current(snapshot: _GateAuthoritySnapshot) -> None:
        snapshot.stage.require_current("Stage 1 gate authority")
        snapshot.foundation_gate.require_current("Foundation gate")
        snapshot.foundation_environment.require_current("Foundation environment manifest")
        for relative, environment_file in snapshot.foundation_environment_files:
            environment_file.require_current(f"Foundation environment file/{relative}")
        snapshot.review_report.require_current("review report")
        snapshot.review_package.require_current("review package")
        snapshot.source.require_current()

    @classmethod
    def _authority_fingerprint(
        cls,
        context: Stage1GateContext,
        target_state: Stage1GateState,
    ) -> str:
        """Bind every authority input around a complete gate verification pass."""
        digest = hashlib.sha256()
        digest.update(b"stage1-gate-authority-fingerprint/v1\0")
        cls._fingerprint_field(digest, "target-state", target_state.encode("utf-8"))
        cls._fingerprint_field(digest, "domain-version", STAGE1_STAGE_ID.encode("utf-8"))

        repo_root = context.repo_root.expanduser().resolve()
        cls._fingerprint_field(digest, "repository:path", repo_root.as_posix().encode("utf-8"))
        for label, arguments in (
            ("top-level", ["rev-parse", "--show-toplevel"]),
            ("branch", ["branch", "--show-current"]),
            ("head", ["rev-parse", "HEAD"]),
            ("tree", ["rev-parse", "HEAD^{tree}"]),
            ("porcelain-status", ["status", "--porcelain=v1", "-z"]),
            ("index-entries", ["ls-files", "--stage", "-z"]),
        ):
            cls._fingerprint_field(
                digest,
                f"repository:{label}",
                cls._run_git(repo_root, arguments, f"authority {label}").encode("utf-8"),
            )

        stage = context.stage_root.expanduser().resolve()
        cls._fingerprint_stage_tree(digest, stage)
        cls._fingerprint_file(digest, "foundation-gate", context.foundation_gate)
        environment_path = context.foundation_environment_manifest.expanduser().resolve()
        environment_bytes = cls._fingerprint_file(
            digest,
            "foundation-environment-manifest",
            environment_path,
        )
        cls._fingerprint_environment_files(digest, environment_path, environment_bytes)

        review_bytes = cls._fingerprint_file(digest, "stage-review", stage / "review.json")
        cls._fingerprint_review_sources(digest, review_bytes)
        return digest.hexdigest()

    @staticmethod
    def _fingerprint_field(digest: object, label: str, payload: bytes) -> None:
        label_bytes = label.encode("utf-8")
        digest.update(len(label_bytes).to_bytes(8, "big"))
        digest.update(label_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    @classmethod
    def _fingerprint_file(
        cls,
        digest: object,
        label: str,
        value: str | Path,
    ) -> bytes | None:
        path = Path(value).expanduser().resolve()
        cls._fingerprint_field(digest, f"{label}:path", path.as_posix().encode("utf-8"))
        if path.is_dir():
            cls._fingerprint_field(digest, f"{label}:kind", b"directory")
            return None
        if not path.is_file():
            cls._fingerprint_field(digest, f"{label}:kind", b"missing")
            return None
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise Stage1GateError(f"Stage 1 authority snapshot cannot read {label}") from exc
        cls._fingerprint_field(digest, f"{label}:kind", b"file")
        cls._fingerprint_field(digest, f"{label}:bytes", payload)
        return payload

    @classmethod
    def _fingerprint_stage_tree(cls, digest: object, stage: Path) -> None:
        cls._fingerprint_field(digest, "stage-root:path", stage.as_posix().encode("utf-8"))
        if not stage.is_dir():
            cls._fingerprint_field(digest, "stage-root:kind", b"missing")
            return
        cls._fingerprint_field(digest, "stage-root:kind", b"directory")
        try:
            entries = sorted(stage.rglob("*"), key=lambda path: path.relative_to(stage).as_posix())
        except OSError as exc:
            raise Stage1GateError("Stage 1 authority snapshot cannot enumerate stage root") from exc
        for entry in entries:
            relative = entry.relative_to(stage).as_posix()
            cls._fingerprint_field(digest, "stage-entry:path", relative.encode("utf-8"))
            if entry.is_dir():
                cls._fingerprint_field(digest, "stage-entry:kind", b"directory")
            elif entry.is_file():
                cls._fingerprint_field(digest, "stage-entry:kind", b"file")
                try:
                    cls._fingerprint_field(digest, "stage-entry:bytes", entry.read_bytes())
                except OSError as exc:
                    raise Stage1GateError("Stage 1 authority snapshot cannot read stage entry") from exc
            else:
                cls._fingerprint_field(digest, "stage-entry:kind", b"missing")

    @classmethod
    def _fingerprint_environment_files(
        cls,
        digest: object,
        environment_path: Path,
        environment_bytes: bytes | None,
    ) -> None:
        if environment_bytes is None:
            return
        try:
            manifest = json.loads(environment_bytes.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        entries = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(entries, list):
            return
        root = environment_path.parent.resolve()
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
                continue
            relative = Path(entry["file"])
            if relative.is_absolute() or ".." in relative.parts:
                raise Stage1GateError("Stage 1 authority snapshot environment file escapes lock root")
            source = (root / relative).resolve()
            try:
                source.relative_to(root)
            except ValueError as exc:
                raise Stage1GateError("Stage 1 authority snapshot environment file escapes lock root") from exc
            cls._fingerprint_file(digest, "foundation-environment-file", source)

    @classmethod
    def _fingerprint_review_sources(cls, digest: object, review_bytes: bytes | None) -> None:
        if review_bytes is None:
            return
        try:
            review = json.loads(review_bytes.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(review, dict):
            return
        for key in ("review_report", "review_package"):
            binding = review.get(key)
            if isinstance(binding, dict) and isinstance(binding.get("path"), str):
                cls._fingerprint_file(digest, f"{key}:source", binding["path"])

    @classmethod
    def _scenario_identity_hash(
        cls,
        summary: dict[str, object],
        config: Stage1Config,
    ) -> str:
        return cls._scenario_identity_hash_from_inputs(summary, config)

    @staticmethod
    def _scenario_identity_hash_from_inputs(
        summary: dict[str, object],
        config: Stage1Config,
    ) -> str:
        if not isinstance(summary, dict) or not isinstance(config, Stage1Config):
            raise Stage1GateError("Stage 1 scenario identity inputs are invalid")
        expected_scenario_id = "smoke-v1/procedural-rock-crater/v1"
        scenario_hash = summary.get("scenario_hash")
        coverage_mask = summary.get("coverage_mask")
        proxy = summary.get("proxy_morphology")
        if summary.get("scenario_id") != expected_scenario_id:
            raise Stage1GateError("Stage 1 scenario identity is invalid")
        if not isinstance(scenario_hash, str) or re.fullmatch(r"[0-9a-f]{64}", scenario_hash) is None:
            raise Stage1GateError("Stage 1 scenario hash is invalid")
        if not isinstance(coverage_mask, dict):
            raise Stage1GateError("Stage 1 coverage mask identity is invalid")
        coverage_hash = coverage_mask.get("sha256")
        if not isinstance(coverage_hash, str) or re.fullmatch(r"[0-9a-f]{64}", coverage_hash) is None:
            raise Stage1GateError("Stage 1 coverage mask hash is invalid")
        expected_proxy_keys = {
            "proxy_generator_version",
            "density_profile",
            "generation_attempt",
            "rock_count",
            "crater_count",
            "object_catalog_sha256",
            "layer_hashes",
        }
        if not isinstance(proxy, dict) or set(proxy) != expected_proxy_keys:
            raise Stage1GateError("Stage 1 proxy morphology schema is invalid")
        if proxy.get("proxy_generator_version") != config.proxy_generator_version:
            raise Stage1GateError("Stage 1 proxy generator version drifted")
        if proxy.get("density_profile") != config.proxy_density_profile:
            raise Stage1GateError("Stage 1 proxy density profile drifted")
        generation_attempt = proxy.get("generation_attempt")
        if (
            type(generation_attempt) is not int
            or generation_attempt < 0
            or generation_attempt >= config.proxy_max_scene_attempts
        ):
            raise Stage1GateError("Stage 1 proxy generation attempt is invalid")
        area_m2 = config.roi_width_m * config.roi_height_m
        for key, kind in (("rock_count", "rock"), ("crater_count", "crater")):
            count = proxy.get(key)
            low, high = scaled_count_bounds(config.proxy_density_profile, kind, area_m2)
            if type(count) is not int or not low <= count <= high:
                raise Stage1GateError(f"Stage 1 proxy {kind} count drifted")
        catalog_hash = proxy.get("object_catalog_sha256")
        if not isinstance(catalog_hash, str) or re.fullmatch(r"[0-9a-f]{64}", catalog_hash) is None:
            raise Stage1GateError("Stage 1 proxy catalog hash is invalid")
        layer_hashes = proxy.get("layer_hashes")
        if not isinstance(layer_hashes, dict) or set(layer_hashes) != {
            "height",
            "hard_obstacle",
            "slope",
            "traversability",
        }:
            raise Stage1GateError("Stage 1 proxy layer hash schema is invalid")
        if any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in layer_hashes.values()
        ):
            raise Stage1GateError("Stage 1 proxy layer hash is invalid")
        scenario_record = {
            "schema_version": "stage1_programmatic_scenario_data_identity/v1",
            "scenario_id": expected_scenario_id,
            "scenario_hash": scenario_hash,
            "coverage_mask": coverage_mask,
            "proxy_morphology": proxy,
            "synthetic_source_kind": config.synthetic_source_kind,
            "physical_obstacle_cells_written": config.physical_obstacle_cells_written,
            "value_prior_source": config.value_prior_source,
        }
        return hashlib.sha256(ArtifactStore.canonical_json_bytes(scenario_record)).hexdigest()

    @classmethod
    def _verify_repository_identity(cls, repo_root: str | Path) -> tuple[Path, str]:
        resolved = Path(repo_root).expanduser().resolve()
        if os.path.normcase(str(resolved)) != os.path.normcase(str(Path(FOUNDATION_WORKTREE_ROOT).resolve())):
            raise Stage1GateError("Stage 1 repository is not the frozen linked worktree")
        if not resolved.is_dir() or not (resolved / "pyproject.toml").is_file():
            raise Stage1GateError("Stage 1 repository root is invalid")
        top_level = cls._run_git(resolved, ["rev-parse", "--show-toplevel"], "top-level").strip()
        if os.path.normcase(str(Path(top_level).resolve())) != os.path.normcase(str(resolved)):
            raise Stage1GateError("Stage 1 repository top-level mismatch")
        branch = cls._run_git(resolved, ["branch", "--show-current"], "branch").strip()
        git_dir = cls._run_git(resolved, ["rev-parse", "--git-dir"], "git-dir").strip()
        git_common_dir = cls._run_git(resolved, ["rev-parse", "--git-common-dir"], "git-common-dir").strip()
        if branch != FOUNDATION_BRANCH:
            raise Stage1GateError("Stage 1 linked worktree branch mismatch")
        if cls._normalize_git_path(resolved, git_dir) != cls._normalize_contract_path(FOUNDATION_GIT_DIR):
            raise Stage1GateError("Stage 1 linked worktree git-dir mismatch")
        if cls._normalize_git_path(resolved, git_common_dir) != cls._normalize_contract_path(
            FOUNDATION_GIT_COMMON_DIR
        ):
            raise Stage1GateError("Stage 1 linked worktree common-dir mismatch")
        status = cls._run_git(resolved, ["status", "--short"], "status")
        if status.strip():
            raise Stage1GateError("Stage 1 approval requires a clean Git tree")
        tree_hash = cls._run_git(resolved, ["rev-parse", "HEAD^{tree}"], "tree").strip()
        if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", tree_hash) is None:
            raise Stage1GateError("Stage 1 Git tree hash is invalid")
        return resolved, tree_hash

    _verify_clean_repository = _verify_repository_identity

    @classmethod
    def _verify_execution_source_identity_snapshot(
        cls,
        identity: object,
        current_identity: dict[str, object],
    ) -> tuple[str, str]:
        expected_keys = {
            "schema_version",
            "prospective_git_tree",
            "reviewed_path_count",
            "reviewed_source_set_sha256",
            "repo_src_root",
            "package_import_path",
            "workflow_import_path",
        }
        if not isinstance(identity, dict) or set(identity) != expected_keys:
            raise Stage1GateError("Stage 1 execution source identity schema is invalid")
        if identity.get("schema_version") != "ppo_highres_frontier_stage1_execution_source_identity/v1":
            raise Stage1GateError("Stage 1 execution source identity version is invalid")
        if identity.get("reviewed_path_count") != 32 or identity != current_identity:
            raise Stage1GateError("Stage 1 frozen execution source identity drifted")
        git_tree_hash = identity.get("prospective_git_tree")
        if not isinstance(git_tree_hash, str) or re.fullmatch(
            r"(?:[0-9a-f]{40}|[0-9a-f]{64})",
            git_tree_hash,
        ) is None:
            raise Stage1GateError("Stage 1 frozen execution tree identity is invalid")
        return (
            hashlib.sha256(ArtifactStore.canonical_json_bytes(identity)).hexdigest(),
            git_tree_hash,
        )

    @classmethod
    def _verify_execution_source_identity(
        cls,
        repo_root: str | Path,
        identity: object,
        current_git_tree: str,
    ) -> str:
        expected_keys = {
            "schema_version",
            "prospective_git_tree",
            "reviewed_path_count",
            "reviewed_source_set_sha256",
            "repo_src_root",
            "package_import_path",
            "workflow_import_path",
        }
        if not isinstance(identity, dict) or set(identity) != expected_keys:
            raise Stage1GateError("Stage 1 execution source identity schema is invalid")
        if identity.get("schema_version") != "ppo_highres_frontier_stage1_execution_source_identity/v1":
            raise Stage1GateError("Stage 1 execution source identity version is invalid")
        if identity.get("reviewed_path_count") != 32:
            raise Stage1GateError("Stage 1 execution source path count is invalid")
        if identity.get("prospective_git_tree") != current_git_tree:
            raise Stage1GateError("machine tree does not equal reviewed tree and current tree")
        declared_digest = identity.get("reviewed_source_set_sha256")
        if (
            not isinstance(declared_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", declared_digest) is None
            or compute_stage1_reviewed_source_set_sha256(repo_root) != declared_digest
        ):
            raise Stage1GateError("Stage 1 execution source-set digest drifted")
        current_imports = resolve_stage1_import_identity(repo_root)
        if any(identity.get(key) != value for key, value in current_imports.items()):
            raise Stage1GateError("Stage 1 execution import source drifted")
        return hashlib.sha256(ArtifactStore.canonical_json_bytes(identity)).hexdigest()

    @staticmethod
    def _run_git(repo_root: Path, arguments: list[str], label: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise Stage1GateError(f"Stage 1 git {label} query failed")
        return completed.stdout

    @staticmethod
    def _normalize_git_path(repo_root: Path, value: str) -> str:
        path = Path(value)
        resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
        return os.path.normcase(str(resolved))

    @staticmethod
    def _normalize_contract_path(value: str) -> str:
        return os.path.normcase(str(Path(value).resolve()))

    @classmethod
    def _verify_manifest(cls, stage: Path, manifest_path: Path) -> str:
        manifest = cls._read_object(manifest_path, "manifest")
        entries = manifest.get("artifacts")
        if not isinstance(entries, list) or len(entries) != len(STAGE1_MACHINE_ARTIFACTS):
            raise Stage1GateError("Stage 1 manifest is not exact")
        paths = {entry.get("path") for entry in entries if isinstance(entry, dict)}
        if paths != set(STAGE1_MACHINE_ARTIFACTS):
            raise Stage1GateError("Stage 1 manifest paths drifted")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise Stage1GateError("Stage 1 manifest entry is invalid")
            artifact = (stage / entry["path"]).resolve()
            try:
                artifact.relative_to(stage)
            except ValueError as exc:
                raise Stage1GateError("Stage 1 manifest path escapes stage root") from exc
            if not artifact.is_file():
                raise Stage1GateError("Stage 1 manifest artifact is missing")
            payload = artifact.read_bytes()
            if entry.get("size_bytes") != len(payload) or entry.get("sha256") != hashlib.sha256(payload).hexdigest():
                raise Stage1GateError(f"Stage 1 manifest artifact drift: {entry['path']}")
        return hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    @classmethod
    def _verify_stage_artifact_snapshot(
        cls,
        snapshot: _GateAuthoritySnapshot,
        run_id: str,
        reviewed_tree: str,
        *,
        review_hash: str,
        approval_challenge: str,
        review_recorded_at_utc: str,
        target_state: Stage1GateState,
    ) -> tuple[Literal["absent", "approved"], str]:
        actual = frozenset(name for name, _kind in snapshot.stage.members)
        if actual not in cls._expected_stage_member_sets(target_state):
            raise Stage1GateError("Stage 1 stage artifact set contains unexpected entries")
        if any(Path(name).suffix.lower() in {".ckpt", ".pt", ".pth"} for name in actual):
            raise Stage1GateError("Stage 1 no-checkpoint sentinel was violated")
        approval_required = target_state in {"approved", "next_stage"}
        if not approval_required:
            return "absent", STAGE1_APPROVAL_ABSENT_HASH
        approval_snapshot = snapshot.stage.file("approval.json")
        approval = strict_json_object_from_bytes(approval_snapshot.payload, "approval")
        _validate_stage1_approval_evidence(
            approval,
            run_id=run_id,
            reviewed_tree=reviewed_tree,
            review_hash=review_hash,
            approval_challenge=approval_challenge,
            review_recorded_at_utc=review_recorded_at_utc,
        )
        return "approved", approval_snapshot.sha256

    @classmethod
    def _verify_stage_artifact_set(
        cls,
        stage: Path,
        run_id: str,
        reviewed_tree: str,
        *,
        review_hash: str,
        approval_challenge: str,
        review_recorded_at_utc: str,
        target_state: Stage1GateState,
    ) -> tuple[Literal["absent", "approved"], str]:
        base = set(STAGE1_MACHINE_ARTIFACTS) | {"manifest.json", "review.json"}
        approval_required = target_state in {"approved", "next_stage"}
        if target_state == "next_stage":
            allowed_sets = (base | {"approval.json"}, base | {"approval.json", "gate.json"})
        elif approval_required:
            allowed_sets = (base | {"approval.json"},)
        else:
            allowed_sets = (base,)
        actual = {entry.name for entry in stage.iterdir()}
        if approval_required and "approval.json" not in actual:
            raise Stage1GateError("Stage 1 human approval evidence is missing")
        if actual not in allowed_sets:
            raise Stage1GateError("Stage 1 stage artifact set contains unexpected entries")
        approval_path = stage / "approval.json"
        if not approval_required:
            return "absent", STAGE1_APPROVAL_ABSENT_HASH
        approval = cls._read_object(approval_path, "approval")
        _validate_stage1_approval_evidence(
            approval,
            run_id=run_id,
            reviewed_tree=reviewed_tree,
            review_hash=review_hash,
            approval_challenge=approval_challenge,
            review_recorded_at_utc=review_recorded_at_utc,
        )
        return "approved", hashlib.sha256(approval_path.read_bytes()).hexdigest()

    @classmethod
    def _verify_foundation_sources_snapshot(
        cls,
        snapshot: _GateAuthoritySnapshot,
    ) -> tuple[str, str]:
        gate = strict_json_object_from_bytes(snapshot.foundation_gate.payload, "Foundation gate")
        if gate.get("schema_version") != "foundation_gate_record/v1" or gate.get("state") != "next_stage":
            raise Stage1GateError("Foundation gate is not authorized for next_stage")
        bindings = gate.get("bindings")
        context = gate.get("context")
        if not isinstance(bindings, dict) or not isinstance(context, dict):
            raise Stage1GateError("Foundation gate bindings or context are invalid")
        if bindings.get("authorized_next_stage") != FOUNDATION_AUTHORIZED_STAGE1:
            raise Stage1GateError("Foundation gate does not authorize Stage 1")
        expected_checkpoint = hashlib.sha256(NO_CHECKPOINT_SENTINEL.encode("utf-8")).hexdigest()
        if bindings.get("checkpoint_hash") != expected_checkpoint:
            raise Stage1GateError("Foundation no-checkpoint binding drifted")
        declared_environment = Path(str(context.get("environment_manifest", ""))).expanduser()
        declared_environment = Path(os.path.abspath(os.fspath(declared_environment)))
        if os.path.normcase(os.fspath(declared_environment)) != os.path.normcase(
            os.fspath(snapshot.foundation_environment.path)
        ):
            raise Stage1GateError("Foundation environment lock path mismatch")
        cls._verify_environment_manifest_snapshot(snapshot)
        if bindings.get("environment_hash") != cls._foundation_environment_gate_hash_bytes(
            snapshot.foundation_environment
        ):
            raise Stage1GateError("Foundation environment lock gate hash mismatch")
        return snapshot.foundation_gate.sha256, snapshot.foundation_environment.sha256

    @staticmethod
    def _verify_environment_manifest_snapshot(snapshot: _GateAuthoritySnapshot) -> None:
        manifest = strict_json_object_from_bytes(
            snapshot.foundation_environment.payload,
            "Foundation environment manifest",
        )
        if manifest.get("schema_version") != "ppo_foundation_environment_lock_manifest/v1":
            raise Stage1GateError("Foundation environment manifest schema is invalid")
        entries = manifest.get("files")
        if not isinstance(entries, list) or not entries:
            raise Stage1GateError("Foundation environment manifest files are missing")
        captured = dict(snapshot.foundation_environment_files)
        if len(captured) != len(snapshot.foundation_environment_files):
            raise Stage1GateError("Foundation environment snapshot paths are duplicated")
        declared_names: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"file", "size_bytes", "sha256"}:
                raise Stage1GateError("Foundation environment manifest entry is invalid")
            relative = entry.get("file")
            if not isinstance(relative, str) or relative not in captured:
                raise Stage1GateError("Foundation environment snapshot file set is not exact")
            declared_names.append(relative)
            source = captured[relative]
            if entry.get("size_bytes") != source.size_bytes or entry.get("sha256") != source.sha256:
                raise Stage1GateError("Foundation environment file hash mismatch")
        if len(declared_names) != len(set(declared_names)) or set(declared_names) != set(captured):
            raise Stage1GateError("Foundation environment snapshot file set is not exact")

    @staticmethod
    def _foundation_environment_gate_hash_bytes(snapshot: FrozenFileSnapshot) -> str:
        digest = hashlib.sha256()
        digest.update(b"gate-environment-file/v1\0")
        digest.update(snapshot.path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(snapshot.payload)
        return digest.hexdigest()

    @classmethod
    def _verify_foundation_sources(cls, gate_path: Path, environment_path: Path) -> tuple[str, str]:
        gate = cls._read_object(gate_path.expanduser().resolve(), "Foundation gate")
        environment = environment_path.expanduser().resolve()
        if gate.get("schema_version") != "foundation_gate_record/v1" or gate.get("state") != "next_stage":
            raise Stage1GateError("Foundation gate is not authorized for next_stage")
        bindings = gate.get("bindings")
        context = gate.get("context")
        if not isinstance(bindings, dict) or not isinstance(context, dict):
            raise Stage1GateError("Foundation gate bindings or context are invalid")
        if bindings.get("authorized_next_stage") != FOUNDATION_AUTHORIZED_STAGE1:
            raise Stage1GateError("Foundation gate does not authorize Stage 1")
        expected_checkpoint = hashlib.sha256(NO_CHECKPOINT_SENTINEL.encode("utf-8")).hexdigest()
        if bindings.get("checkpoint_hash") != expected_checkpoint:
            raise Stage1GateError("Foundation no-checkpoint binding drifted")
        declared_environment = Path(str(context.get("environment_manifest", ""))).expanduser().resolve()
        if os.path.normcase(str(declared_environment)) != os.path.normcase(str(environment)):
            raise Stage1GateError("Foundation environment lock path mismatch")
        cls._verify_environment_manifest(environment)
        if bindings.get("environment_hash") != cls._foundation_environment_gate_hash(environment):
            raise Stage1GateError("Foundation environment lock gate hash mismatch")
        return hashlib.sha256(gate_path.expanduser().resolve().read_bytes()).hexdigest(), hashlib.sha256(
            environment.read_bytes()
        ).hexdigest()

    @classmethod
    def _verify_environment_manifest(cls, path: Path) -> None:
        manifest = cls._read_object(path, "Foundation environment manifest")
        if manifest.get("schema_version") != "ppo_foundation_environment_lock_manifest/v1":
            raise Stage1GateError("Foundation environment manifest schema is invalid")
        entries = manifest.get("files")
        if not isinstance(entries, list) or not entries:
            raise Stage1GateError("Foundation environment manifest files are missing")
        root = path.parent.resolve()
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
                raise Stage1GateError("Foundation environment manifest entry is invalid")
            relative = Path(entry["file"])
            if relative.is_absolute() or ".." in relative.parts:
                raise Stage1GateError("Foundation environment manifest path is unsafe")
            source = (root / relative).resolve()
            try:
                source.relative_to(root)
            except ValueError as exc:
                raise Stage1GateError("Foundation environment file escapes lock root") from exc
            if not source.is_file():
                raise Stage1GateError("Foundation environment file is missing")
            payload = source.read_bytes()
            if entry.get("size_bytes") != len(payload) or entry.get("sha256") != hashlib.sha256(payload).hexdigest():
                raise Stage1GateError("Foundation environment file hash mismatch")

    @classmethod
    def _verify_review_snapshot(
        cls,
        snapshot: _GateAuthoritySnapshot,
        *,
        run_id: str,
        config_hash: str,
        manifest_hash: str,
        scenario_hash: str,
        foundation_gate_hash: str,
        foundation_environment_hash: str,
        foundation_base_commit: str,
        reviewed_git_tree: str,
        execution_source_identity: dict[str, object],
        execution_source_identity_hash: str,
    ) -> _VerifiedStage1Review:
        review_snapshot = snapshot.stage.file("review.json")
        review = strict_json_object_from_bytes(review_snapshot.payload, "review")
        expected_keys = {
            "schema_version",
            "state",
            "goal_id",
            "stage_id",
            "run_id",
            "spec_verdict",
            "quality_verdict",
            "issue_counts",
            "final_gate_conclusion",
            "foundation_base_commit",
            "reviewed_git_tree",
            "execution_source_identity",
            "execution_source_identity_hash",
            "review_report",
            "review_package",
            "config_hash",
            "manifest_hash",
            "scenario_hash",
            "foundation_gate_hash",
            "foundation_environment_hash",
            "approval_challenge",
            "review_recorded_at_utc",
        }
        if set(review) != expected_keys:
            raise Stage1GateError("Stage 1 review schema is not exact")
        if review.get("reviewed_git_tree") != reviewed_git_tree:
            raise Stage1GateError("current Git tree does not equal the reviewed Git tree")
        if review.get("foundation_base_commit") != foundation_base_commit:
            raise Stage1GateError("reviewed Foundation base commit mismatch")
        required = {
            "schema_version": "ppo_highres_frontier_stage1_independent_review/v2",
            "state": "awaiting_human_approval",
            "goal_id": STAGE1_GOAL_ID,
            "stage_id": STAGE1_STAGE_ID,
            "run_id": run_id,
            "spec_verdict": "APPROVED",
            "quality_verdict": "APPROVED",
            "final_gate_conclusion": "READY_FOR_HUMAN_APPROVAL",
            "config_hash": config_hash,
            "manifest_hash": manifest_hash,
            "scenario_hash": scenario_hash,
            "foundation_gate_hash": foundation_gate_hash,
            "foundation_environment_hash": foundation_environment_hash,
            "execution_source_identity": execution_source_identity,
            "execution_source_identity_hash": execution_source_identity_hash,
        }
        if any(review.get(key) != value for key, value in required.items()):
            raise Stage1GateError("Stage 1 review declarations or cross-hashes mismatch")
        counts = review.get("issue_counts")
        if not isinstance(counts, dict) or counts.get("critical") != 0 or counts.get("important") != 0:
            raise Stage1GateError("Stage 1 review has blocking findings")
        if not isinstance(counts.get("minor"), int) or counts["minor"] < 0:
            raise Stage1GateError("Stage 1 review minor count is invalid")
        report_binding = cls._verify_review_source_snapshot(
            review.get("review_report"),
            "review report",
            snapshot.review_report,
        )
        package_binding = cls._verify_review_source_snapshot(
            review.get("review_package"),
            "review package",
            snapshot.review_package,
        )
        try:
            front_matter, replayed_report, replayed_package = verify_stage1_review_evidence_snapshots(
                repo_root=snapshot.source.repo_root,
                foundation_base_commit=foundation_base_commit,
                reviewed_git_tree=reviewed_git_tree,
                report_snapshot=snapshot.review_report,
                package_snapshot=snapshot.review_package,
                source_snapshot=snapshot.source,
            )
        except Stage1WorkflowError as exc:
            raise Stage1GateError("Stage 1 review report/package semantic replay failed") from exc
        if replayed_report != report_binding or replayed_package != package_binding:
            raise Stage1GateError("Stage 1 review report/package bindings drifted during replay")
        if (
            front_matter["spec_verdict"] != review["spec_verdict"]
            or front_matter["quality_verdict"] != review["quality_verdict"]
            or front_matter["critical_count"] != counts["critical"]
            or front_matter["important_count"] != counts["important"]
            or front_matter["minor_count"] != counts["minor"]
            or front_matter["final_gate_conclusion"] != review["final_gate_conclusion"]
        ):
            raise Stage1GateError("Stage 1 review report declarations differ from review record")
        expected_challenge = build_stage1_approval_challenge(
            goal_id=STAGE1_GOAL_ID,
            stage_id=STAGE1_STAGE_ID,
            run_id=run_id,
            reviewed_git_tree=reviewed_git_tree,
            review_report_sha256=str(report_binding["sha256"]),
            review_package_sha256=str(package_binding["sha256"]),
            manifest_hash=manifest_hash,
        )
        if review.get("approval_challenge") != expected_challenge:
            raise Stage1GateError("Stage 1 review approval challenge mismatch")
        recorded_at = str(review.get("review_recorded_at_utc", ""))
        _parse_utc_timestamp(recorded_at, "review")
        return _VerifiedStage1Review(
            review_hash=review_snapshot.sha256,
            approval_challenge=expected_challenge,
            review_recorded_at_utc=recorded_at,
        )

    @staticmethod
    def _verify_review_source_snapshot(
        value: object,
        label: str,
        source: FrozenFileSnapshot,
    ) -> dict[str, object]:
        if not isinstance(value, dict) or set(value) != {
            "path",
            "sha256",
            "bytes",
            "lf_count",
            "logical_line_count",
        }:
            raise Stage1GateError(f"Stage 1 {label} binding is invalid")
        if value != source.binding():
            raise Stage1GateError(f"Stage 1 {label} hash/bytes/lines mismatch")
        return value

    @classmethod
    def _verify_review(
        cls,
        path: Path,
        *,
        repo_root: Path,
        run_id: str,
        config_hash: str,
        manifest_hash: str,
        scenario_hash: str,
        foundation_gate_hash: str,
        foundation_environment_hash: str,
        foundation_base_commit: str,
        reviewed_git_tree: str,
        execution_source_identity: dict[str, object],
        execution_source_identity_hash: str,
    ) -> _VerifiedStage1Review:
        review = cls._read_object(path, "review")
        expected_keys = {
            "schema_version",
            "state",
            "goal_id",
            "stage_id",
            "run_id",
            "spec_verdict",
            "quality_verdict",
            "issue_counts",
            "final_gate_conclusion",
            "foundation_base_commit",
            "reviewed_git_tree",
            "execution_source_identity",
            "execution_source_identity_hash",
            "review_report",
            "review_package",
            "config_hash",
            "manifest_hash",
            "scenario_hash",
            "foundation_gate_hash",
            "foundation_environment_hash",
            "approval_challenge",
            "review_recorded_at_utc",
        }
        if set(review) != expected_keys:
            raise Stage1GateError("Stage 1 review schema is not exact")
        if review.get("reviewed_git_tree") != reviewed_git_tree:
            raise Stage1GateError("current Git tree does not equal the reviewed Git tree")
        if review.get("foundation_base_commit") != foundation_base_commit:
            raise Stage1GateError("reviewed Foundation base commit mismatch")
        required = {
            "schema_version": "ppo_highres_frontier_stage1_independent_review/v2",
            "state": "awaiting_human_approval",
            "goal_id": STAGE1_GOAL_ID,
            "stage_id": STAGE1_STAGE_ID,
            "run_id": run_id,
            "spec_verdict": "APPROVED",
            "quality_verdict": "APPROVED",
            "final_gate_conclusion": "READY_FOR_HUMAN_APPROVAL",
            "config_hash": config_hash,
            "manifest_hash": manifest_hash,
            "scenario_hash": scenario_hash,
            "foundation_gate_hash": foundation_gate_hash,
            "foundation_environment_hash": foundation_environment_hash,
            "execution_source_identity": execution_source_identity,
            "execution_source_identity_hash": execution_source_identity_hash,
        }
        if any(review.get(key) != value for key, value in required.items()):
            raise Stage1GateError("Stage 1 review declarations or cross-hashes mismatch")
        counts = review.get("issue_counts")
        if not isinstance(counts, dict) or counts.get("critical") != 0 or counts.get("important") != 0:
            raise Stage1GateError("Stage 1 review has blocking findings")
        if not isinstance(counts.get("minor"), int) or counts["minor"] < 0:
            raise Stage1GateError("Stage 1 review minor count is invalid")
        report_binding = cls._verify_review_source(review.get("review_report"), "review report")
        package_binding = cls._verify_review_source(review.get("review_package"), "review package")
        try:
            front_matter, replayed_report, replayed_package = verify_stage1_review_evidence(
                repo_root=repo_root,
                foundation_base_commit=foundation_base_commit,
                reviewed_git_tree=reviewed_git_tree,
                review_report=str(report_binding["path"]),
                review_package=str(package_binding["path"]),
            )
        except Stage1WorkflowError as exc:
            raise Stage1GateError("Stage 1 review report/package semantic replay failed") from exc
        if replayed_report != report_binding or replayed_package != package_binding:
            raise Stage1GateError("Stage 1 review report/package bindings drifted during replay")
        if (
            front_matter["spec_verdict"] != review["spec_verdict"]
            or front_matter["quality_verdict"] != review["quality_verdict"]
            or front_matter["critical_count"] != counts["critical"]
            or front_matter["important_count"] != counts["important"]
            or front_matter["minor_count"] != counts["minor"]
            or front_matter["final_gate_conclusion"] != review["final_gate_conclusion"]
        ):
            raise Stage1GateError("Stage 1 review report declarations differ from review record")
        expected_challenge = build_stage1_approval_challenge(
            goal_id=STAGE1_GOAL_ID,
            stage_id=STAGE1_STAGE_ID,
            run_id=run_id,
            reviewed_git_tree=reviewed_git_tree,
            review_report_sha256=str(report_binding["sha256"]),
            review_package_sha256=str(package_binding["sha256"]),
            manifest_hash=manifest_hash,
        )
        if review.get("approval_challenge") != expected_challenge:
            raise Stage1GateError("Stage 1 review approval challenge mismatch")
        recorded_at = str(review.get("review_recorded_at_utc", ""))
        _parse_utc_timestamp(recorded_at, "review")
        return _VerifiedStage1Review(
            review_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
            approval_challenge=expected_challenge,
            review_recorded_at_utc=recorded_at,
        )

    @staticmethod
    def _verify_review_source(value: object, label: str) -> dict[str, object]:
        if not isinstance(value, dict) or set(value) != {
            "path",
            "sha256",
            "bytes",
            "lf_count",
            "logical_line_count",
        }:
            raise Stage1GateError(f"Stage 1 {label} binding is invalid")
        source = Path(str(value.get("path", ""))).expanduser().resolve()
        expected_hash = value.get("sha256")
        if not source.is_file():
            raise Stage1GateError(f"Stage 1 {label} source is missing")
        payload = source.read_bytes()
        if (
            expected_hash != hashlib.sha256(payload).hexdigest()
            or value.get("bytes") != len(payload)
            or value.get("lf_count") != payload.count(b"\n")
            or value.get("logical_line_count") != len(payload.splitlines())
        ):
            raise Stage1GateError(f"Stage 1 {label} hash/bytes/lines mismatch")
        return value

    @classmethod
    def _verify_phase_state_bytes(cls, payload: bytes, run_id: str) -> None:
        try:
            lines = payload.decode("utf-8-sig").splitlines()
        except UnicodeDecodeError as exc:
            raise Stage1GateError("Stage 1 phase-state is not strict UTF-8") from exc
        if not lines:
            raise Stage1GateError("Stage 1 phase-state is missing")
        events = [
            strict_json_object_from_bytes(line.encode("utf-8"), "phase-state event")
            for line in lines
        ]
        if [event.get("state") for event in events] != ["machine_passed"] or any(
            event.get("run_id") != run_id for event in events
        ):
            raise Stage1GateError("Stage 1 phase-state sequence is invalid")

    @classmethod
    def _verify_phase_state(cls, path: Path, run_id: str) -> None:
        if not path.is_file():
            raise Stage1GateError("Stage 1 phase-state is missing")
        try:
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        except json.JSONDecodeError as exc:
            raise Stage1GateError("Stage 1 phase-state is invalid JSONL") from exc
        expected = ["machine_passed"]
        if [event.get("state") for event in events] != expected or any(
            event.get("run_id") != run_id for event in events
        ):
            raise Stage1GateError("Stage 1 phase-state sequence is invalid")

    @staticmethod
    def _foundation_environment_gate_hash(path: Path) -> str:
        digest = hashlib.sha256()
        digest.update(b"gate-environment-file/v1\0")
        digest.update(path.resolve().as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _identity_hash(kind: str, value: str) -> str:
        return hashlib.sha256(f"stage1-gate-{kind}-identity/v1\0{value}".encode("utf-8")).hexdigest()

    @staticmethod
    def _read_object(path: Path, label: str) -> dict[str, object]:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise Stage1GateError(f"{label} binding source is missing")
        try:
            value = json.loads(resolved.read_bytes().decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Stage1GateError(f"{label} is invalid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise Stage1GateError(f"{label} must be a JSON object")
        return value


def _validate_canonical_approval_uuid(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise Stage1GateError(f"Stage 1 approval {label} is invalid")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise Stage1GateError(f"Stage 1 approval {label} is invalid") from exc
    if str(parsed) != value:
        raise Stage1GateError(f"Stage 1 approval {label} is not canonical")


def _validate_stage1_approval_evidence(
    approval: dict[str, object],
    *,
    run_id: str,
    reviewed_tree: str,
    review_hash: str,
    approval_challenge: str,
    review_recorded_at_utc: str,
) -> None:
    expected_keys = {
        "schema_version",
        "state",
        "authority_kind",
        "actor",
        "thread_id",
        "user_turn_id",
        "approval_text",
        "approval_text_utf8_sha256",
        "approval_timestamp_utc",
        "goal_id",
        "stage_id",
        "run_id",
        "reviewed_git_tree",
        "review_hash",
        "approval_challenge",
    }
    if set(approval) != expected_keys:
        raise Stage1GateError("Stage 1 approval schema is not exact")
    expected = {
        "schema_version": "ppo_highres_frontier_stage1_human_approval/v2",
        "state": "approved",
        "authority_kind": "codex_user_message/v1",
        "actor": "user",
        "approval_text": STAGE1_APPROVAL_TEXT,
        "approval_text_utf8_sha256": STAGE1_APPROVAL_TEXT_SHA256,
        "goal_id": STAGE1_GOAL_ID,
        "stage_id": STAGE1_STAGE_ID,
        "run_id": run_id,
        "reviewed_git_tree": reviewed_tree,
        "review_hash": review_hash,
        "approval_challenge": approval_challenge,
    }
    if any(approval.get(key) != value for key, value in expected.items()):
        raise Stage1GateError("Stage 1 approval declarations are invalid")
    _validate_canonical_approval_uuid(approval.get("thread_id"), "thread ID")
    _validate_canonical_approval_uuid(approval.get("user_turn_id"), "user turn ID")
    review_time = _parse_utc_timestamp(review_recorded_at_utc, "review")
    approval_time = _parse_utc_timestamp(
        str(approval.get("approval_timestamp_utc", "")),
        "approval",
    )
    if approval_time < review_time:
        raise Stage1GateError("Stage 1 approval timestamp predates review")


def _parse_utc_timestamp(value: str, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise Stage1GateError(f"Stage 1 {label} timestamp must be UTC Z form")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise Stage1GateError(f"Stage 1 {label} timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise Stage1GateError(f"Stage 1 {label} timestamp is not UTC")
    return parsed


_STAGE1_GATE_TOKEN = object()
_STAGE1_GATE_GENESIS = hashlib.sha256(b"stage1_gate_record_genesis/v1").hexdigest()


@dataclass(frozen=True, slots=True)
class _Stage1TransitionProof:
    state: Stage1GateState
    previous_record_hash: str
    record_hash: str

    def to_dict(self) -> dict[str, str]:
        return {
            "state": self.state,
            "previous_record_hash": self.previous_record_hash,
            "record_hash": self.record_hash,
        }


class Stage1GateRecord:
    __slots__ = ("_state", "_proofs", "_bindings", "_context")

    def __init__(
        self,
        _token: object | None = None,
        *,
        state: Stage1GateState | None = None,
        proofs: tuple[_Stage1TransitionProof, ...] | None = None,
        bindings: Stage1GateBindings | None = None,
        context: Stage1GateContext | None = None,
    ) -> None:
        if _token is not _STAGE1_GATE_TOKEN:
            raise TypeError("Stage1GateRecord construction is private")
        if state is None or proofs is None or bindings is None or context is None:
            raise TypeError("Stage1GateRecord internal construction requires complete fields")
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_proofs", proofs)
        object.__setattr__(self, "_bindings", bindings)
        object.__setattr__(self, "_context", context)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("Stage1GateRecord is immutable")

    @property
    def state(self) -> Stage1GateState:
        return self._state

    @property
    def history(self) -> tuple[Stage1GateState, ...]:
        return tuple(proof.state for proof in self._proofs)

    @property
    def bindings(self) -> Stage1GateBindings:
        return self._bindings

    @property
    def context(self) -> Stage1GateContext:
        return self._context

    @classmethod
    def machine_passed(cls, context: Stage1GateContext) -> Stage1GateRecord:
        if not isinstance(context, Stage1GateContext):
            raise Stage1GateError("Stage1GateRecord requires a verified context")
        bindings = Stage1GateBindingVerifier.compute(context, target_state="machine_passed")
        if Stage1GateBindingVerifier.compute(context, target_state="machine_passed") != bindings:
            raise Stage1GateError("Stage 1 gate binding drift detected before machine proof")
        proof = cls._make_proof("machine_passed", _STAGE1_GATE_GENESIS, bindings, context)
        record = cls(
            _STAGE1_GATE_TOKEN,
            state="machine_passed",
            proofs=(proof,),
            bindings=bindings,
            context=context,
        )
        if Stage1GateBindingVerifier.compute(context, target_state="machine_passed") != bindings:
            raise Stage1GateError("Stage 1 gate binding drift detected after machine proof")
        return record

    def transition(self, target: Stage1GateState) -> Stage1GateRecord:
        next_index = len(self._proofs)
        expected = _STAGE1_STATE_SEQUENCE[next_index] if next_index < len(_STAGE1_STATE_SEQUENCE) else None
        if target != expected:
            raise Stage1GateError(f"invalid Stage 1 gate transition: {self.state} -> {target}")
        target_bindings = Stage1GateBindingVerifier.compute(self.context, target_state=target)
        if Stage1GateBindingVerifier.compute(self.context, target_state=target) != target_bindings:
            raise Stage1GateError("Stage 1 gate binding drift detected before transition proof")
        if self._base_binding_values(target_bindings) != self._base_binding_values(self.bindings):
            raise Stage1GateError("Stage 1 gate binding drift detected")
        if target == "approved":
            if self.bindings.approval_state != "absent" or target_bindings.approval_state != "approved":
                raise Stage1GateError("Stage 1 approval state transition is invalid")
        elif target == "next_stage":
            if target_bindings != self.bindings:
                raise Stage1GateError("Stage 1 approval binding drift detected")
        elif target_bindings != self.bindings:
            raise Stage1GateError("Stage 1 gate binding drift detected")
        proof = self._make_proof(
            target,
            self._proofs[-1].record_hash,
            target_bindings,
            self.context,
        )
        record = type(self)(
            _STAGE1_GATE_TOKEN,
            state=target,
            proofs=(*self._proofs, proof),
            bindings=target_bindings,
            context=self.context,
        )
        if Stage1GateBindingVerifier.compute(self.context, target_state=target) != target_bindings:
            raise Stage1GateError("Stage 1 gate binding drift detected after transition proof")
        return record

    def to_dict(self) -> dict[str, object]:
        if Stage1GateBindingVerifier.compute(self.context, target_state=self.state) != self.bindings:
            raise Stage1GateError("Stage 1 gate binding drift detected before serialization")
        return {
            "schema_version": "ppo_highres_frontier_stage1_gate_record/v1",
            "state": self.state,
            "history": [proof.to_dict() for proof in self._proofs],
            "bindings": self.bindings.model_dump(mode="json"),
            "context": self.context.model_dump(mode="json"),
        }

    def write_verified(self) -> Path:
        """Exclusively publish, replay, and post-verify an append-only final gate."""

        if type(self) is not Stage1GateRecord:
            raise Stage1GateError("Stage 1 verified gate writer requires a proven Stage1GateRecord")
        if self.state != "next_stage":
            raise Stage1GateError("Stage 1 verified gate writer requires a proven next_stage record")

        stage_root = self.context.stage_root.expanduser().resolve()
        store = ArtifactStore(stage_root)
        gate_path = store.resolve("gate.json")
        expected_gate_path = (stage_root / "gate.json").resolve()
        if os.path.normcase(str(gate_path)) != os.path.normcase(str(expected_gate_path)):
            raise Stage1GateError("Stage 1 verified gate target path mismatches its context")
        if gate_path.exists():
            raise Stage1GateError("Stage 1 gate.json already exists before verified write")

        expected_payload = self.to_dict()
        expected_bytes = ArtifactStore.canonical_json_bytes(expected_payload)
        pre_write_payload = self.to_dict()
        if (
            pre_write_payload != expected_payload
            or ArtifactStore.canonical_json_bytes(pre_write_payload) != expected_bytes
        ):
            raise Stage1GateError("Stage 1 gate payload drifted before verified write")
        if gate_path.exists():
            raise Stage1GateError("Stage 1 gate.json appeared before verified write")

        # Exclusive creation is the canonical commit point.  Once it may have
        # succeeded, no failure path mutates the shared canonical authority.
        written_path = store.write_json_exclusive("gate.json", expected_payload)
        if not isinstance(written_path, (str, Path)) or os.path.normcase(
            str(Path(written_path).expanduser().resolve())
        ) != os.path.normcase(str(gate_path)):
            raise Stage1GateError("Stage 1 gate writer returned an unexpected target path")
        if gate_path.read_bytes() != expected_bytes:
            raise Stage1GateError("Stage 1 gate bytes changed after atomic write")

        replayed = Stage1GateRecord.load_verified(gate_path)
        if replayed.state != self.state:
            raise Stage1GateError("Stage 1 verified gate replay state mismatch")
        if replayed.history != self.history:
            raise Stage1GateError("Stage 1 verified gate replay history mismatch")
        if replayed.bindings != self.bindings:
            raise Stage1GateError("Stage 1 verified gate replay bindings mismatch")
        if replayed.context != self.context:
            raise Stage1GateError("Stage 1 verified gate replay context mismatch")
        replayed_payload = replayed.to_dict()
        if (
            replayed_payload != expected_payload
            or ArtifactStore.canonical_json_bytes(replayed_payload) != expected_bytes
        ):
            raise Stage1GateError("Stage 1 verified gate replay payload mismatch")

        post_write_payload = self.to_dict()
        if (
            post_write_payload != expected_payload
            or ArtifactStore.canonical_json_bytes(post_write_payload) != expected_bytes
        ):
            raise Stage1GateError("Stage 1 gate authority drifted after verified replay")
        if gate_path.read_bytes() != expected_bytes:
            raise Stage1GateError("Stage 1 gate bytes changed after post-write verification")
        return gate_path

    @classmethod
    def load_verified(cls, gate_path: str | Path) -> Stage1GateRecord:
        if not isinstance(gate_path, (str, Path)):
            raise Stage1GateError("Stage1GateRecord loader requires an actual gate.json path")
        resolved_gate = Path(gate_path).expanduser().resolve()
        payload = Stage1GateBindingVerifier._read_object(resolved_gate, "gate.json")
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "state",
            "history",
            "bindings",
            "context",
        }:
            raise Stage1GateError("Stage1GateRecord loader schema is invalid")
        if payload.get("schema_version") != "ppo_highres_frontier_stage1_gate_record/v1":
            raise Stage1GateError("Stage1GateRecord schema version mismatch")
        try:
            context = Stage1GateContext.model_validate(payload["context"])
            bindings = Stage1GateBindings.model_validate(payload["bindings"])
        except ValidationError as exc:
            raise Stage1GateError("Stage1GateRecord field validation failed") from exc
        expected_gate = (context.stage_root.expanduser().resolve() / "gate.json").resolve()
        if os.path.normcase(str(resolved_gate)) != os.path.normcase(str(expected_gate)):
            raise Stage1GateError("Stage1GateRecord loader gate.json path mismatches its context")
        if payload.get("state") != "next_stage":
            raise Stage1GateError("Stage1GateRecord gate.json must represent next_stage")
        if Stage1GateBindingVerifier.compute(context, target_state="next_stage") != bindings:
            raise Stage1GateError("Stage1GateRecord binding drift detected")
        history = payload.get("history")
        if not isinstance(history, list) or len(history) != len(_STAGE1_STATE_SEQUENCE):
            raise Stage1GateError("Stage1GateRecord history is invalid")
        expected_states = _STAGE1_STATE_SEQUENCE[: len(history)]
        previous_hash = _STAGE1_GATE_GENESIS
        proofs: list[_Stage1TransitionProof] = []
        for index, (raw, expected_state) in enumerate(zip(history, expected_states, strict=True)):
            if not isinstance(raw, dict) or set(raw) != {"state", "previous_record_hash", "record_hash"}:
                raise Stage1GateError("Stage1GateRecord proof schema is invalid")
            if raw.get("state") != expected_state or raw.get("previous_record_hash") != previous_hash:
                raise Stage1GateError("Stage1GateRecord proof sequence is invalid")
            proof_bindings = cls._bindings_for_proof(expected_state, bindings)
            expected_proof = cls._make_proof(expected_state, previous_hash, proof_bindings, context)
            if raw.get("record_hash") != expected_proof.record_hash:
                raise Stage1GateError(f"Stage1GateRecord proof hash mismatch at index {index}")
            proofs.append(expected_proof)
            previous_hash = expected_proof.record_hash
        if payload.get("state") != expected_states[-1]:
            raise Stage1GateError("Stage1GateRecord final state mismatches history")
        return cls(
            _STAGE1_GATE_TOKEN,
            state=expected_states[-1],
            proofs=tuple(proofs),
            bindings=bindings,
            context=context,
        )

    @staticmethod
    def _base_binding_values(bindings: Stage1GateBindings) -> tuple[object, ...]:
        return tuple(
            getattr(bindings, name)
            for name in Stage1GateBindings.model_fields
            if name not in {"approval_state", "approval_hash"}
        )

    @staticmethod
    def _bindings_for_proof(
        state: Stage1GateState,
        final_bindings: Stage1GateBindings,
    ) -> Stage1GateBindings:
        if state in {"approved", "next_stage"}:
            return final_bindings
        payload = final_bindings.model_dump(mode="json")
        payload["approval_state"] = "absent"
        payload["approval_hash"] = STAGE1_APPROVAL_ABSENT_HASH
        return Stage1GateBindings.model_validate(payload)

    @staticmethod
    def _make_proof(
        state: Stage1GateState,
        previous_record_hash: str,
        bindings: Stage1GateBindings,
        context: Stage1GateContext,
    ) -> _Stage1TransitionProof:
        source = {
            "schema_version": "ppo_highres_frontier_stage1_gate_transition/v1",
            "state": state,
            "previous_record_hash": previous_record_hash,
            "bindings": bindings.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
        }
        record_hash = hashlib.sha256(ArtifactStore.canonical_json_bytes(source)).hexdigest()
        return _Stage1TransitionProof(state, previous_record_hash, record_hash)
