from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lunar_exploration_ppo.configs.schema import (
    FOUNDATION_BRANCH,
    FOUNDATION_GIT_COMMON_DIR,
    FOUNDATION_GIT_DIR,
    FOUNDATION_WORKTREE_ROOT,
    FoundationConfig,
    load_foundation_config,
)
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.workflows.foundation import (
    MACHINE_ARTIFACTS,
    FoundationPreflightError,
    validate_foundation_run_id,
)


FOUNDATION_GOAL_ID: Final = "ppo-highres-frontier-map-exploration"
FOUNDATION_STAGE_ID: Final = "foundation"
AUTHORIZED_NEXT_STAGE: Final = "ppo_highres_frontier_stage1_smoke_environment/v1"
NO_CHECKPOINT_SENTINEL: Final = "foundation_no_checkpoint/v1"
_RUNTIME_CONFIG_OVERRIDE_FIELDS: Final = frozenset({"run_id", "device"})

_FOUNDATION_REPOSITORY_MARKERS: Final = (
    Path("pyproject.toml"),
    Path("docs/superpowers/specs/2026-07-09-ppo-highres-frontier-map-exploration-design.md"),
    Path("docs/superpowers/plans/2026-07-10-ppo-highres-frontier-map-exploration.md"),
    Path("configs/ppo_highres_frontier_foundation_v1.json"),
)

GateState = Literal[
    "machine_passed",
    "awaiting_independent_review",
    "awaiting_human_approval",
    "approved",
    "next_stage",
]
_STATE_SEQUENCE: Final[tuple[GateState, ...]] = (
    "machine_passed",
    "awaiting_independent_review",
    "awaiting_human_approval",
    "approved",
    "next_stage",
)


class GateError(RuntimeError):
    """Raised when a gate binding or transition fails closed."""


class FoundationGateContext(BaseModel):
    """Minimal external locators for one canonical Foundation run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo_root: Literal["C:/Users/77634/.codex/worktrees/ca49/lunar-path-planning"]
    stage_root: Path
    environment_manifest: Path
    run_id: str = Field(min_length=1)


class GateBindings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    goal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    git_tree_hash: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    authorized_next_stage: Literal["ppo_highres_frontier_stage1_smoke_environment/v1"]


class GateBindingVerifier:
    """Derive all bindings from one authoritative Foundation context."""

    @classmethod
    def compute(cls, context: FoundationGateContext) -> GateBindings:
        if not isinstance(context, FoundationGateContext):
            raise GateError("Gate entry requires a FoundationGateContext")
        repo_root, git_tree_hash = cls._verify_foundation_repository(context.repo_root)
        stage = cls._verify_stage_root(context)
        config_path = stage / "config.json"
        summary_path = stage / "summary.json"
        routing_path = stage / "routing.json"
        manifest_path = stage / "manifest.json"
        review_path = stage / "review.json"

        config = cls._verify_config(repo_root, config_path, context.run_id)
        config_bytes = config_path.read_bytes()
        config_file_hash = hashlib.sha256(config_bytes).hexdigest()
        summary = cls._read_json_object("summary", summary_path)
        routing = cls._read_json_object("routing", routing_path)
        cls._verify_run_declarations(
            context.run_id,
            summary,
            routing,
            config_file_hash,
            config.device,
        )

        manifest_bytes = manifest_path.read_bytes() if manifest_path.is_file() else b""
        if not manifest_bytes:
            raise GateError(f"manifest binding source is missing: {manifest_path}")
        cls._verify_stage_manifest(stage, manifest_bytes, config_file_hash)
        manifest_file_hash = hashlib.sha256(manifest_bytes).hexdigest()

        environment_manifest = context.environment_manifest.expanduser().resolve()
        environment_bytes = cls._verify_environment_manifest(environment_manifest)
        environment_file_hash = hashlib.sha256(environment_bytes).hexdigest()
        cls._verify_review(
            review_path,
            run_id=context.run_id,
            config_hash=config_file_hash,
            manifest_hash=manifest_file_hash,
            environment_manifest_hash=environment_file_hash,
            repository_identity_hash=hashlib.sha256(
                ArtifactStore.canonical_json_bytes(config.repository_identity.model_dump(mode="json"))
            ).hexdigest(),
        )
        cls._verify_review_phase_state(stage / "phase-state.jsonl", context.run_id)

        verified_data_records: list[dict[str, Any]] = []
        for source in (config.verified_data_sources.dem, config.verified_data_sources.slope):
            physical_path = Path(source.path)
            size_bytes, sha256 = cls._stream_file_size_sha256(physical_path)
            if size_bytes != source.size_bytes:
                raise GateError(f"physical data size mismatch: {source.path}")
            if sha256 != source.sha256:
                raise GateError(f"physical data SHA-256 mismatch: {source.path}")
            verified_data_records.append(source.model_dump(mode="json"))
        verified_data = {
            "schema_version": "foundation_verified_physical_data/v1",
            "sources": verified_data_records,
        }
        data_hash = hashlib.sha256(ArtifactStore.canonical_json_bytes(verified_data)).hexdigest()
        return GateBindings(
            goal_hash=cls._hash_identity("goal", FOUNDATION_GOAL_ID),
            stage_hash=cls._hash_identity("stage", FOUNDATION_STAGE_ID),
            git_tree_hash=git_tree_hash,
            config_hash=cls._hash_file_source("config", config_path),
            data_hash=data_hash,
            environment_hash=cls._hash_file_source("environment", environment_manifest),
            checkpoint_hash=hashlib.sha256(NO_CHECKPOINT_SENTINEL.encode("utf-8")).hexdigest(),
            review_hash=cls._hash_file_source("review", review_path),
            manifest_hash=cls._hash_file_source("manifest", manifest_path),
            authorized_next_stage=AUTHORIZED_NEXT_STAGE,
        )

    @staticmethod
    def _hash_identity(kind: str, identity: str) -> str:
        return hashlib.sha256(f"gate-{kind}-identity/v1\0{identity}".encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_file_source(kind: str, path: Path) -> str:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise GateError(f"{kind} binding source is missing or not a file: {resolved}")
        digest = hashlib.sha256()
        digest.update(f"gate-{kind}-file/v1\0".encode("utf-8"))
        digest.update(resolved.as_posix().encode("utf-8"))
        digest.update(b"\0")
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _stream_file_size_sha256(path: Path) -> tuple[int, str]:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise GateError(f"physical data source is missing or not a file: {resolved}")
        digest = hashlib.sha256()
        size_bytes = 0
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size_bytes += len(chunk)
                digest.update(chunk)
        return size_bytes, digest.hexdigest()

    @classmethod
    def _verify_foundation_repository(cls, repo_root: str | Path) -> tuple[Path, str]:
        if str(repo_root).replace("\\", "/") != FOUNDATION_WORKTREE_ROOT:
            raise GateError("Foundation repository root does not match the frozen linked worktree")
        resolved = Path(repo_root).expanduser().resolve()
        if not resolved.is_dir():
            raise GateError(f"Foundation repository root is missing: {resolved}")
        for marker in _FOUNDATION_REPOSITORY_MARKERS:
            if not (resolved / marker).is_file():
                raise GateError(f"Foundation repository marker is missing: {marker.as_posix()}")

        top_level = cls._run_git(resolved, ["rev-parse", "--show-toplevel"], "top-level").strip()
        if os.path.normcase(str(Path(top_level).resolve())) != os.path.normcase(str(resolved)):
            raise GateError("Foundation repository top-level does not match repo_root")
        git_dir = cls._run_git(resolved, ["rev-parse", "--git-dir"], "git-dir").strip()
        git_common_dir = cls._run_git(resolved, ["rev-parse", "--git-common-dir"], "git-common-dir").strip()
        branch = cls._run_git(resolved, ["branch", "--show-current"], "branch").strip()
        if cls._normalize_git_path(resolved, git_dir) != cls._normalize_contract_path(FOUNDATION_GIT_DIR):
            raise GateError("Foundation Git dir does not match the frozen linked worktree")
        if cls._normalize_git_path(resolved, git_common_dir) != cls._normalize_contract_path(FOUNDATION_GIT_COMMON_DIR):
            raise GateError("Foundation Git common dir does not match the frozen repository")
        if branch != FOUNDATION_BRANCH:
            raise GateError("Foundation branch does not match the frozen contract")
        status = cls._run_git(
            resolved,
            ["status", "--porcelain=v1", "--untracked-files=all"],
            "status",
        )
        if status:
            raise GateError("Git binding requires a clean Foundation worktree")
        tree_oid = cls._run_git(resolved, ["rev-parse", "--verify", "HEAD^{tree}"], "HEAD tree").strip()
        if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", tree_oid) is None:
            raise GateError("Git HEAD tree binding returned an invalid object id")
        return resolved, tree_oid

    @staticmethod
    def _normalize_git_path(repo_root: Path, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            path = repo_root / path
        return os.path.normcase(str(path.resolve())).replace("\\", "/")

    @staticmethod
    def _normalize_contract_path(value: str) -> str:
        return os.path.normcase(str(Path(value).resolve())).replace("\\", "/")

    @staticmethod
    def _run_git(repo_root: Path, arguments: list[str], label: str) -> str:
        command = ["git", "-C", str(repo_root), *arguments]
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise GateError(f"Git {label} binding failed: {completed.stderr.strip()}")
        return completed.stdout

    @staticmethod
    def _verify_stage_root(context: FoundationGateContext) -> Path:
        try:
            validate_foundation_run_id(context.run_id)
        except FoundationPreflightError as exc:
            raise GateError(f"Foundation context run_id is invalid: {context.run_id}") from exc
        stage = context.stage_root.expanduser().resolve()
        if not stage.is_dir():
            raise GateError(f"Foundation stage root is missing: {stage}")
        if stage.name != "s0" or stage.parent.name != context.run_id:
            raise GateError("Foundation stage root must be canonical <base>/<run_id>/s0")
        return stage

    @staticmethod
    def _verify_config(repo_root: Path, config_path: Path, run_id: str) -> FoundationConfig:
        try:
            stage_config = load_foundation_config(config_path)
            repository_config = load_foundation_config(
                repo_root / "configs" / "ppo_highres_frontier_foundation_v1.json"
            )
        except Exception as exc:
            raise GateError(f"Foundation config validation failed: {exc}") from exc
        if stage_config.run_id != run_id:
            raise GateError("Foundation stage config run_id declaration mismatch")
        if stage_config.model_dump(exclude=_RUNTIME_CONFIG_OVERRIDE_FIELDS) != repository_config.model_dump(
            exclude=_RUNTIME_CONFIG_OVERRIDE_FIELDS
        ):
            raise GateError("Foundation stage config does not match the repository canonical config")
        return stage_config

    @classmethod
    def _verify_run_declarations(
        cls,
        run_id: str,
        summary: dict[str, Any],
        routing: dict[str, Any],
        config_hash: str,
        device: Literal["cpu", "cuda"],
    ) -> None:
        expected_summary = {
            "goal_id": FOUNDATION_GOAL_ID,
            "stage_id": FOUNDATION_STAGE_ID,
            "run_id": run_id,
            "state": "awaiting_human_approval",
            "device": device,
            "config_hash": config_hash,
            "config_hash_schema": "sha256_file_bytes/v1",
        }
        for field, expected in expected_summary.items():
            if summary.get(field) != expected:
                raise GateError(f"Foundation summary {field} declaration mismatch")
        if routing.get("run_id") != run_id:
            raise GateError("Foundation routing run_id declaration mismatch")
        if routing.get("route") != "awaiting_human_approval":
            raise GateError("Foundation routing route declaration mismatch")

    @classmethod
    def _verify_stage_manifest(cls, stage_root: Path, manifest_bytes: bytes, config_hash: str) -> None:
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GateError("Foundation manifest is not valid UTF-8 JSON") from exc
        if manifest.get("schema_version") != "sha256_manifest/v1" or not isinstance(manifest.get("artifacts"), list):
            raise GateError("Foundation manifest schema is invalid")
        artifact_paths = [
            entry.get("path") if isinstance(entry, dict) else None
            for entry in manifest["artifacts"]
        ]
        if any(not isinstance(path, str) for path in artifact_paths):
            raise GateError("Foundation manifest artifact entry is invalid")
        expected_paths = set(MACHINE_ARTIFACTS)
        if (
            len(artifact_paths) != len(MACHINE_ARTIFACTS)
            or len(set(artifact_paths)) != len(artifact_paths)
            or set(artifact_paths) != expected_paths
        ):
            raise GateError("Foundation manifest must contain exactly six unique machine artifacts")
        entries: dict[str, dict[str, Any]] = {}
        for entry in manifest["artifacts"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise GateError("Foundation manifest artifact entry is invalid")
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise GateError("Foundation manifest artifact path is unsafe")
            artifact = (stage_root / relative).resolve()
            try:
                artifact.relative_to(stage_root)
            except ValueError as exc:
                raise GateError("Foundation manifest artifact escapes stage root") from exc
            if not artifact.is_file():
                raise GateError(f"Foundation manifest artifact is missing: {relative.as_posix()}")
            payload = artifact.read_bytes()
            if entry.get("size_bytes") != len(payload) or entry.get("sha256") != hashlib.sha256(payload).hexdigest():
                raise GateError(f"Foundation manifest artifact hash mismatch: {relative.as_posix()}")
            entries[relative.as_posix()] = entry
        if entries.get("config.json", {}).get("sha256") != config_hash:
            raise GateError("Foundation manifest config.json hash mismatch")

    @classmethod
    def _verify_environment_manifest(cls, manifest_path: Path) -> bytes:
        if not manifest_path.is_file():
            raise GateError(f"environment manifest is missing: {manifest_path}")
        payload = manifest_path.read_bytes()
        manifest = cls._decode_json_object("environment manifest", payload)
        if manifest.get("schema_version") != "ppo_foundation_environment_lock_manifest/v1":
            raise GateError("environment manifest schema is invalid")
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise GateError("environment manifest files declaration is missing")
        for entry in files:
            if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
                raise GateError("environment manifest file declaration is invalid")
            relative = Path(entry["file"])
            if relative.is_absolute() or ".." in relative.parts:
                raise GateError("environment manifest file path is unsafe")
            source = (manifest_path.parent / relative).resolve()
            try:
                source.relative_to(manifest_path.parent)
            except ValueError as exc:
                raise GateError("environment manifest file escapes manifest root") from exc
            if not source.is_file():
                raise GateError(f"environment manifest file is missing: {relative.as_posix()}")
            source_bytes = source.read_bytes()
            if entry.get("size_bytes") != len(source_bytes) or entry.get("sha256") != hashlib.sha256(source_bytes).hexdigest():
                raise GateError(f"environment manifest file hash mismatch: {relative.as_posix()}")
        return payload

    @classmethod
    def _verify_review(
        cls,
        review_path: Path,
        *,
        run_id: str,
        config_hash: str,
        manifest_hash: str,
        environment_manifest_hash: str,
        repository_identity_hash: str,
    ) -> None:
        review = cls._read_json_object("review", review_path)
        required = {
            "schema_version": "foundation_independent_review/v2",
            "goal_id": FOUNDATION_GOAL_ID,
            "stage_id": FOUNDATION_STAGE_ID,
            "run_id": run_id,
            "spec_verdict": "approved",
            "quality_verdict": "approved",
            "repository_identity_hash": repository_identity_hash,
            "config_hash": config_hash,
            "manifest_hash": manifest_hash,
            "environment_manifest_hash": environment_manifest_hash,
        }
        if any(review.get(field) != expected for field, expected in required.items()):
            raise GateError("Foundation review declarations or cross-hashes mismatch")
        counts = review.get("issue_counts")
        if not isinstance(counts, dict) or counts.get("critical") != 0 or counts.get("important") != 0:
            raise GateError("Foundation review blocking issue counts are not zero")
        if not isinstance(counts.get("minor"), int) or counts["minor"] < 0:
            raise GateError("Foundation review minor issue count is invalid")
        for field in ("review_source_hash", "review_package_hash"):
            value = review.get(field)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise GateError(f"Foundation review {field} is invalid")
        if set(review) != set(required) | {"issue_counts", "review_source_hash", "review_package_hash"}:
            raise GateError("Foundation review schema contains missing or extra fields")

    @classmethod
    def _verify_review_phase_state(cls, path: Path, run_id: str) -> None:
        if not path.is_file():
            raise GateError("Foundation review phase-state is missing")
        try:
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        except json.JSONDecodeError as exc:
            raise GateError("Foundation review phase-state is invalid JSONL") from exc
        expected_states = ["machine_passed", "awaiting_independent_review", "awaiting_human_approval"]
        if [event.get("state") for event in events] != expected_states:
            raise GateError("Foundation review phase-state sequence is invalid")
        if any(event.get("run_id") != run_id for event in events):
            raise GateError("Foundation review phase-state run_id mismatch")

    @classmethod
    def _read_json_object(cls, label: str, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise GateError(f"{label} binding source is missing: {path}")
        return cls._decode_json_object(label, path.read_bytes())

    @staticmethod
    def _decode_json_object(label: str, payload: bytes) -> dict[str, Any]:
        try:
            value = json.loads(payload.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GateError(f"{label} is not valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise GateError(f"{label} must be a JSON object")
        return value


_GATE_RECORD_TOKEN = object()
_GENESIS_RECORD_HASH = hashlib.sha256(b"foundation_gate_record_genesis/v1").hexdigest()


@dataclass(frozen=True, slots=True)
class _TransitionProof:
    state: GateState
    previous_record_hash: str
    record_hash: str

    def to_dict(self) -> dict[str, str]:
        return {
            "state": self.state,
            "previous_record_hash": self.previous_record_hash,
            "record_hash": self.record_hash,
        }


class GateRecord:
    __slots__ = ("_state", "_proofs", "_bindings", "_context")

    def __init__(
        self,
        _token: object | None = None,
        *,
        state: GateState | None = None,
        proofs: tuple[_TransitionProof, ...] | None = None,
        bindings: GateBindings | None = None,
        context: FoundationGateContext | None = None,
        **_unsupported: object,
    ) -> None:
        if _token is not _GATE_RECORD_TOKEN:
            raise TypeError("GateRecord construction is private; use machine_passed or load_verified")
        if state is None or proofs is None or bindings is None or context is None:
            raise TypeError("GateRecord internal construction requires complete verified fields")
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_proofs", proofs)
        object.__setattr__(self, "_bindings", bindings)
        object.__setattr__(self, "_context", context)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("GateRecord is immutable")

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, _memo: dict[int, object]) -> Self:
        return self

    @property
    def state(self) -> GateState:
        return self._state

    @property
    def history(self) -> tuple[GateState, ...]:
        return tuple(proof.state for proof in self._proofs)

    @property
    def bindings(self) -> GateBindings:
        return self._bindings

    @property
    def context(self) -> FoundationGateContext:
        return self._context

    @classmethod
    def machine_passed(cls, context: FoundationGateContext) -> GateRecord:
        if not isinstance(context, FoundationGateContext):
            raise GateError("GateRecord.machine_passed requires a FoundationGateContext")
        bindings = GateBindingVerifier.compute(context)
        proof = cls._make_proof("machine_passed", _GENESIS_RECORD_HASH, bindings, context)
        return cls(
            _GATE_RECORD_TOKEN,
            state="machine_passed",
            proofs=(proof,),
            bindings=bindings,
            context=context,
        )

    def transition(self, target: GateState) -> GateRecord:
        current_bindings = GateBindingVerifier.compute(self.context)
        if current_bindings != self.bindings:
            drifted = [
                name
                for name in GateBindings.model_fields
                if getattr(current_bindings, name) != getattr(self.bindings, name)
            ]
            raise GateError(f"gate binding drift detected: {', '.join(drifted)}")
        next_index = len(self._proofs)
        expected = _STATE_SEQUENCE[next_index] if next_index < len(_STATE_SEQUENCE) else None
        if target != expected:
            raise GateError(f"invalid gate transition: {self.state} -> {target}")
        proof = self._make_proof(
            target,
            self._proofs[-1].record_hash,
            self.bindings,
            self.context,
        )
        return type(self)(
            _GATE_RECORD_TOKEN,
            state=target,
            proofs=(*self._proofs, proof),
            bindings=self.bindings,
            context=self.context,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "foundation_gate_record/v1",
            "state": self.state,
            "history": [proof.to_dict() for proof in self._proofs],
            "bindings": self.bindings.model_dump(mode="json"),
            "context": self.context.model_dump(mode="json"),
        }

    @classmethod
    def load_verified(cls, payload: dict[str, object]) -> GateRecord:
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "state",
            "history",
            "bindings",
            "context",
        }:
            raise GateError("GateRecord verified loader received an invalid schema")
        if payload.get("schema_version") != "foundation_gate_record/v1":
            raise GateError("GateRecord verified loader schema version mismatch")
        try:
            context = FoundationGateContext.model_validate(payload["context"])
            supplied_bindings = GateBindings.model_validate(payload["bindings"])
        except ValidationError as exc:
            raise GateError(f"GateRecord verified loader field validation failed: {exc}") from exc
        current_bindings = GateBindingVerifier.compute(context)
        if supplied_bindings != current_bindings:
            raise GateError("GateRecord verified loader binding drift detected")
        raw_history = payload.get("history")
        if not isinstance(raw_history, list) or not raw_history or len(raw_history) > len(_STATE_SEQUENCE):
            raise GateError("GateRecord verified loader history is invalid")
        expected_states = _STATE_SEQUENCE[: len(raw_history)]
        previous_hash = _GENESIS_RECORD_HASH
        for index, (raw_proof, expected_state) in enumerate(zip(raw_history, expected_states, strict=True)):
            if not isinstance(raw_proof, dict) or set(raw_proof) != {
                "state",
                "previous_record_hash",
                "record_hash",
            }:
                raise GateError("GateRecord verified loader history proof schema is invalid")
            if raw_proof.get("state") != expected_state or raw_proof.get("previous_record_hash") != previous_hash:
                raise GateError("GateRecord verified loader history sequence is invalid")
            expected_proof = cls._make_proof(expected_state, previous_hash, supplied_bindings, context)
            if raw_proof.get("record_hash") != expected_proof.record_hash:
                raise GateError(f"GateRecord verified loader history hash mismatch at index {index}")
            previous_hash = expected_proof.record_hash
        if payload.get("state") != expected_states[-1]:
            raise GateError("GateRecord verified loader final state mismatches history")

        replayed = cls.machine_passed(context)
        for state in expected_states[1:]:
            replayed = replayed.transition(state)
        if replayed.to_dict() != payload:
            raise GateError("GateRecord verified loader replay mismatch")
        return replayed

    @staticmethod
    def _make_proof(
        state: GateState,
        previous_record_hash: str,
        bindings: GateBindings,
        context: FoundationGateContext,
    ) -> _TransitionProof:
        payload = {
            "schema_version": "foundation_gate_transition_proof/v1",
            "state": state,
            "previous_record_hash": previous_record_hash,
            "bindings": bindings.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
        }
        record_hash = hashlib.sha256(ArtifactStore.canonical_json_bytes(payload)).hexdigest()
        return _TransitionProof(
            state=state,
            previous_record_hash=previous_record_hash,
            record_hash=record_hash,
        )
