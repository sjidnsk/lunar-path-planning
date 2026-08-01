"""Reviewed Stage 1 source identity from an isolated temporary Git index."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Mapping

from lunar_exploration_ppo.configs.schema import (
    FOUNDATION_BRANCH,
    FOUNDATION_GIT_COMMON_DIR,
    FOUNDATION_GIT_DIR,
    FOUNDATION_WORKTREE_ROOT,
)
from lunar_exploration_ppo.workflows.stage1_artifacts import (
    FrozenFileSnapshot,
    Stage1WorkflowError,
)


STAGE1_REVIEWED_PATHS: Final = frozenset(
    {
        "pyproject.toml",
        "configs/ppo_highres_frontier_smoke_v1.json",
        "docs/superpowers/plans/2026-07-10-ppo-highres-frontier-map-exploration.md",
        "docs/superpowers/specs/2026-07-10-rock-crater-proxy-fixture-design-addendum.md",
        "docs/superpowers/plans/2026-07-10-rock-crater-proxy-fixture.md",
        "scripts/run_ppo_stage1_smoke_env.py",
        "src/lunar_exploration_ppo/configs/stage1.py",
        "src/lunar_exploration_ppo/env/action_execution.py",
        "src/lunar_exploration_ppo/env/coverage.py",
        "src/lunar_exploration_ppo/env/env.py",
        "src/lunar_exploration_ppo/env/frontier.py",
        "src/lunar_exploration_ppo/env/frontier_oracle.py",
        "src/lunar_exploration_ppo/env/map_state.py",
        "src/lunar_exploration_ppo/env/reachability.py",
        "src/lunar_exploration_ppo/env/scenario.py",
        "src/lunar_exploration_ppo/env/sensor_model.py",
        "src/lunar_exploration_ppo/env/terrain_proxy.py",
        "src/lunar_exploration_ppo/integrations/__init__.py",
        "src/lunar_exploration_ppo/integrations/path_planner_adapter.py",
        "src/lunar_exploration_ppo/policy/observation.py",
        "src/lunar_exploration_ppo/utils/artifact_io.py",
        "src/lunar_exploration_ppo/utils/geometry.py",
        "src/lunar_exploration_ppo/workflows/stage1.py",
        "src/lunar_exploration_ppo/workflows/stage1_artifacts.py",
        "src/lunar_exploration_ppo/workflows/stage1_gate.py",
        "src/lunar_exploration_ppo/workflows/stage1_metrics.py",
        "src/lunar_exploration_ppo/workflows/stage1_review.py",
        "src/lunar_exploration_ppo/workflows/stage1_source.py",
        "tests/ppo_highres_frontier/test_foundation.py",
        "tests/ppo_highres_frontier/test_stage1_behavior_regression.py",
        "tests/ppo_highres_frontier/test_stage1_smoke_env.py",
        "tests/ppo_highres_frontier/test_stage1_terrain_proxy.py",
    }
)
Stage1SourceMode = Literal["review_dirty", "gate_committed"]


class ReviewedSourceError(RuntimeError):
    """Raised when reviewed source cannot be bound without touching the real index."""


@dataclass(frozen=True, slots=True)
class _FrozenGitSourceInputs:
    top_level: bytes
    branch: bytes
    head: bytes
    head_parents: bytes
    head_tree: bytes
    status: bytes
    index: bytes
    index_entries: bytes
    git_dir: bytes
    git_common_dir: bytes
    base_tree: bytes
    object_format: bytes


@dataclass(frozen=True, slots=True)
class Stage1SourceSnapshot:
    """Frozen reviewed source bytes plus the Git/import identity inputs that name them."""

    repo_root: Path
    base_commit: str
    mode: Stage1SourceMode
    files: tuple[tuple[str, FrozenFileSnapshot], ...]
    package_import: FrozenFileSnapshot
    workflow_import: FrozenFileSnapshot
    git: _FrozenGitSourceInputs

    @classmethod
    def capture(
        cls,
        repo_root: str | Path,
        *,
        base_commit: str,
        package_import_path: str | Path | None = None,
        workflow_import_path: str | Path | None = None,
        mode: Stage1SourceMode = "review_dirty",
    ) -> "Stage1SourceSnapshot":
        if mode not in {"review_dirty", "gate_committed"}:
            raise ReviewedSourceError("Stage 1 source snapshot mode is invalid")
        if len(STAGE1_REVIEWED_PATHS) != 32:
            raise ReviewedSourceError("Stage 1 reviewed source path count must remain 32")
        reviewed = tuple(sorted(STAGE1_REVIEWED_PATHS))
        _require_safe_reviewed_paths(reviewed)
        repo = Path(os.path.abspath(os.fspath(Path(repo_root).expanduser())))
        git_before = _capture_git_source_inputs(repo, base_commit)
        _verify_frozen_git_source_inputs(git_before, base_commit, repo, mode)
        files = tuple(
            (
                relative,
                FrozenFileSnapshot.capture(
                    repo / Path(relative),
                    f"reviewed source/{relative}",
                    authority_root=repo,
                ),
            )
            for relative in reviewed
        )
        file_map = dict(files)
        package_path, workflow_path = _resolve_import_paths_without_reading(
            repo,
            package_import_path=package_import_path,
            workflow_import_path=workflow_import_path,
        )
        package_snapshot = FrozenFileSnapshot.capture(
            package_path,
            "Stage 1 package import",
            authority_root=repo,
        )
        workflow_relative = workflow_path.relative_to(repo).as_posix()
        workflow_snapshot = file_map.get(workflow_relative)
        if workflow_snapshot is None:
            raise ReviewedSourceError("Stage 1 workflow import is outside the reviewed source set")
        if workflow_snapshot.path != workflow_path:
            raise ReviewedSourceError("Stage 1 workflow import path identity drifted")
        git_after = _capture_git_source_inputs(repo, base_commit)
        if git_after != git_before:
            raise ReviewedSourceError("Stage 1 Git source identity changed during capture")
        snapshot = cls(
            repo_root=repo,
            base_commit=base_commit,
            mode=mode,
            files=files,
            package_import=package_snapshot,
            workflow_import=workflow_snapshot,
            git=git_before,
        )
        if mode == "gate_committed":
            try:
                head_tree = snapshot.git.head_tree.strip().decode("ascii", errors="strict")
            except UnicodeDecodeError as exc:
                raise ReviewedSourceError("Stage 1 committed HEAD tree identity is invalid") from exc
            if _prospective_tree_from_frozen_inputs(snapshot) != head_tree:
                raise ReviewedSourceError(
                    "Stage 1 committed HEAD tree does not equal the reviewed prospective tree"
                )
        snapshot.require_current()
        return snapshot

    def payloads(self) -> dict[str, bytes]:
        return {relative: snapshot.payload for relative, snapshot in self.files}

    def file(self, relative: str) -> FrozenFileSnapshot:
        for candidate, snapshot in self.files:
            if candidate == relative:
                return snapshot
        raise ReviewedSourceError(f"Stage 1 reviewed source snapshot is missing: {relative}")

    def require_current(self) -> None:
        current_git = _capture_git_source_inputs(self.repo_root, self.base_commit)
        if current_git != self.git:
            raise ReviewedSourceError("Stage 1 Git source identity changed after capture")
        current_package, current_workflow = _resolve_import_paths_without_reading(self.repo_root)
        if current_package != self.package_import.path or current_workflow != self.workflow_import.path:
            raise ReviewedSourceError("Stage 1 import path identity changed after capture")
        for relative, snapshot in self.files:
            snapshot.require_current(f"reviewed source/{relative}")
        if self.package_import.path != self.workflow_import.path:
            self.package_import.require_current("Stage 1 package import")


def _require_safe_reviewed_paths(reviewed: tuple[str, ...]) -> None:
    normalized: set[str] = set()
    for relative in reviewed:
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or not relative
            or "\\" in relative
            or "." in relative_path.parts
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
        ):
            raise ReviewedSourceError("Stage 1 reviewed source path is unsafe")
        folded = relative.casefold()
        if folded in normalized:
            raise ReviewedSourceError("Stage 1 reviewed source paths contain a case duplicate")
        normalized.add(folded)


def _capture_git_source_inputs(repo: Path, base_commit: str) -> _FrozenGitSourceInputs:
    return _FrozenGitSourceInputs(
        top_level=_git_bytes(repo, ["rev-parse", "--show-toplevel"], "top-level"),
        branch=_git_bytes(repo, ["branch", "--show-current"], "branch"),
        head=_git_bytes(repo, ["rev-parse", "HEAD"], "HEAD"),
        head_parents=_git_bytes(repo, ["rev-list", "--parents", "-n", "1", "HEAD"], "HEAD parents"),
        head_tree=_git_bytes(repo, ["rev-parse", "HEAD^{tree}"], "HEAD tree"),
        status=_git_bytes(
            repo,
            ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
            "status",
        ),
        index=_git_bytes(repo, ["diff", "--cached", "--name-only", "-z"], "index"),
        index_entries=_git_bytes(repo, ["ls-files", "--stage", "-z"], "index entries"),
        git_dir=_git_bytes(repo, ["rev-parse", "--git-dir"], "git-dir"),
        git_common_dir=_git_bytes(
            repo,
            ["rev-parse", "--git-common-dir"],
            "git common-dir",
        ),
        base_tree=_git_bytes(
            repo,
            ["ls-tree", "-r", "-z", "--full-tree", base_commit],
            "Foundation tree",
        ),
        object_format=_git_bytes(repo, ["rev-parse", "--show-object-format"], "object format"),
    )


def _verify_frozen_git_source_inputs(
    inputs: _FrozenGitSourceInputs,
    base_commit: str,
    repo: Path,
    mode: Stage1SourceMode,
) -> None:
    try:
        top_level = inputs.top_level.strip().decode("utf-8", errors="strict")
        branch = inputs.branch.strip().decode("utf-8", errors="strict")
        head = inputs.head.strip().decode("ascii", errors="strict")
        head_parents = inputs.head_parents.strip().decode("ascii", errors="strict")
        head_tree = inputs.head_tree.strip().decode("ascii", errors="strict")
        git_dir = inputs.git_dir.strip().decode("utf-8", errors="strict")
        git_common_dir = inputs.git_common_dir.strip().decode("utf-8", errors="strict")
        object_format = inputs.object_format.strip().decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ReviewedSourceError("Stage 1 Git identity output is invalid") from exc
    normalized_repo = os.path.normcase(os.path.abspath(os.fspath(repo)))
    if normalized_repo != os.path.normcase(os.path.abspath(top_level)) or normalized_repo != os.path.normcase(
        os.path.abspath(FOUNDATION_WORKTREE_ROOT)
    ):
        raise ReviewedSourceError("Stage 1 Git top-level identity is invalid")
    if branch != FOUNDATION_BRANCH:
        raise ReviewedSourceError("Stage 1 Git branch identity is invalid")
    if len(head_tree) not in {40, 64} or any(character not in "0123456789abcdef" for character in head_tree):
        raise ReviewedSourceError("Stage 1 Git HEAD tree identity is invalid")
    if inputs.index:
        operation = "review" if mode == "review_dirty" else "gate"
        raise ReviewedSourceError(f"{operation} requires an empty real Git index")
    status_paths = _status_paths_from_payload(inputs.status)
    if mode == "review_dirty":
        if head != base_commit:
            raise ReviewedSourceError("review requires the frozen Foundation base HEAD")
        if status_paths != set(STAGE1_REVIEWED_PATHS):
            raise ReviewedSourceError("review requires the exact 32-path dirty source set")
    else:
        if inputs.status:
            raise ReviewedSourceError("gate requires a clean Git status")
        parent_tokens = head_parents.split()
        if parent_tokens != [head, base_commit]:
            raise ReviewedSourceError(
                "gate committed HEAD must be the single child of the Foundation base"
            )
    if _normalize_git_identity_path(repo, git_dir) != _normalize_git_identity_path(
        repo,
        FOUNDATION_GIT_DIR,
    ):
        raise ReviewedSourceError("Stage 1 Git dir identity is invalid")
    if _normalize_git_identity_path(repo, git_common_dir) != _normalize_git_identity_path(
        repo,
        FOUNDATION_GIT_COMMON_DIR,
    ):
        raise ReviewedSourceError("Stage 1 Git common-dir identity is invalid")
    if object_format not in {"sha1", "sha256"}:
        raise ReviewedSourceError("Stage 1 Git object format is unsupported")


def _normalize_git_identity_path(repo: Path, value: str) -> str:
    path = Path(value)
    absolute = path if path.is_absolute() else repo / path
    return os.path.normcase(os.path.abspath(os.fspath(absolute)))


def _resolve_import_paths_without_reading(
    repo: Path,
    *,
    package_import_path: str | Path | None = None,
    workflow_import_path: str | Path | None = None,
) -> tuple[Path, Path]:
    src_root = repo / "src"
    package_root = src_root / "lunar_exploration_ppo"
    if package_import_path is None:
        package_module = sys.modules.get("lunar_exploration_ppo")
        package_import_path = getattr(package_module, "__file__", None)
    if workflow_import_path is None:
        workflow_module = sys.modules.get("lunar_exploration_ppo.workflows.stage1")
        workflow_import_path = getattr(workflow_module, "__file__", None)
    if package_import_path is None or workflow_import_path is None:
        raise ReviewedSourceError("Stage 1 package/workflow import paths are unavailable")
    package_path = Path(os.path.abspath(os.fspath(Path(package_import_path).expanduser())))
    workflow_path = Path(os.path.abspath(os.fspath(Path(workflow_import_path).expanduser())))
    if package_path != package_root / "__init__.py":
        raise ReviewedSourceError("Stage 1 package import is not from current repo src")
    if workflow_path != package_root / "workflows" / "stage1.py":
        raise ReviewedSourceError("Stage 1 workflow import is not from current repo src")
    return package_path, workflow_path


def compute_stage1_reviewed_source_set_sha256_from_bytes(
    source_payloads: Mapping[str, bytes],
) -> str:
    if set(source_payloads) != set(STAGE1_REVIEWED_PATHS):
        raise ReviewedSourceError("Stage 1 frozen reviewed source path set must remain exact")
    digest = hashlib.sha256()
    digest.update(b"ppo_highres_frontier_stage1_reviewed_source_set/v1\0")
    for relative in sorted(STAGE1_REVIEWED_PATHS):
        path_bytes = relative.encode("utf-8")
        payload = source_payloads[relative]
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _git_object_id(kind: bytes, payload: bytes, algorithm: str) -> bytes:
    framed = kind + b" " + str(len(payload)).encode("ascii") + b"\0" + payload
    return hashlib.new(algorithm, framed).digest()


def _prospective_tree_from_frozen_inputs(snapshot: Stage1SourceSnapshot) -> str:
    object_format = snapshot.git.object_format.strip().decode("ascii")
    tree: dict[bytes, object] = {}

    def insert(path: tuple[bytes, ...], value: tuple[bytes, bytes]) -> None:
        if not path or any(not part for part in path):
            raise ReviewedSourceError("Foundation tree contains an unsafe path")
        node = tree
        for part in path[:-1]:
            existing = node.setdefault(part, {})
            if not isinstance(existing, dict):
                raise ReviewedSourceError("Foundation tree contains a file/directory conflict")
            node = existing
        if isinstance(node.get(path[-1]), dict):
            raise ReviewedSourceError("Foundation tree contains a file/directory conflict")
        node[path[-1]] = value

    for record in snapshot.git.base_tree.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_kind, object_id = metadata.split(b" ", 2)
            oid_bytes = bytes.fromhex(object_id.decode("ascii"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ReviewedSourceError("Foundation tree listing is invalid") from exc
        if object_kind not in {b"blob", b"commit"}:
            raise ReviewedSourceError("Foundation tree listing has an unsupported object kind")
        insert(tuple(raw_path.split(b"/")), (mode, oid_bytes))

    for relative, payload in snapshot.payloads().items():
        insert(
            tuple(part.encode("utf-8") for part in relative.split("/")),
            (b"100644", _git_object_id(b"blob", payload, object_format)),
        )

    def hash_tree(node: dict[bytes, object]) -> bytes:
        entries: list[tuple[bytes, bool, bytes, bytes]] = []
        for name, value in node.items():
            if isinstance(value, dict):
                entries.append((name, True, b"40000", hash_tree(value)))
            else:
                mode, object_id = value
                entries.append((name, False, mode, object_id))
        entries.sort(key=lambda item: item[0] + (b"/" if item[1] else b"\0"))
        body = b"".join(
            mode + b" " + name + b"\0" + object_id
            for name, _is_tree, mode, object_id in entries
        )
        return _git_object_id(b"tree", body, object_format)

    return hash_tree(tree).hex()


def compute_stage1_source_identity_from_snapshot(snapshot: Stage1SourceSnapshot) -> dict[str, object]:
    src_root = snapshot.repo_root / "src"
    return {
        "schema_version": "ppo_highres_frontier_stage1_execution_source_identity/v1",
        "prospective_git_tree": _prospective_tree_from_frozen_inputs(snapshot),
        "reviewed_path_count": len(STAGE1_REVIEWED_PATHS),
        "reviewed_source_set_sha256": compute_stage1_reviewed_source_set_sha256_from_bytes(
            snapshot.payloads()
        ),
        "repo_src_root": src_root.as_posix(),
        "package_import_path": snapshot.package_import.path.as_posix(),
        "workflow_import_path": snapshot.workflow_import.path.as_posix(),
    }


def compute_stage1_reviewed_source_set_sha256(repo_root: str | Path) -> str:
    """Hash all 32 reviewed paths with unambiguous path and byte-length framing."""

    if len(STAGE1_REVIEWED_PATHS) != 32:
        raise ReviewedSourceError("Stage 1 reviewed source path count must remain 32")
    repo = Path(repo_root).expanduser().resolve()
    digest = __import__("hashlib").sha256()
    digest.update(b"ppo_highres_frontier_stage1_reviewed_source_set/v1\0")
    for relative in sorted(STAGE1_REVIEWED_PATHS):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ReviewedSourceError("Stage 1 reviewed source path is unsafe")
        source = (repo / relative_path).resolve()
        try:
            source.relative_to(repo)
        except ValueError as exc:
            raise ReviewedSourceError("Stage 1 reviewed source path escapes repo") from exc
        if not source.is_file():
            raise ReviewedSourceError(f"Stage 1 reviewed source is missing: {relative}")
        path_bytes = relative.encode("utf-8")
        payload = source.read_bytes()
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def resolve_stage1_import_identity(
    repo_root: str | Path,
    *,
    package_import_path: str | Path | None = None,
    workflow_import_path: str | Path | None = None,
) -> dict[str, str]:
    """Verify the active package and workflow imports originate in this repo's src tree."""

    repo = Path(repo_root).expanduser().resolve()
    src_root = (repo / "src").resolve()
    package_root = (src_root / "lunar_exploration_ppo").resolve()
    if package_import_path is None:
        package_module = sys.modules.get("lunar_exploration_ppo")
        package_import_path = getattr(package_module, "__file__", None)
    if workflow_import_path is None:
        workflow_module = sys.modules.get("lunar_exploration_ppo.workflows.stage1")
        workflow_import_path = getattr(workflow_module, "__file__", None)
    if package_import_path is None or workflow_import_path is None:
        raise ReviewedSourceError("Stage 1 package/workflow import paths are unavailable")
    package_path = Path(package_import_path).expanduser().resolve()
    workflow_path = Path(workflow_import_path).expanduser().resolve()
    expected_package = package_root / "__init__.py"
    expected_workflow = package_root / "workflows" / "stage1.py"
    if package_path != expected_package or not package_path.is_file():
        raise ReviewedSourceError("Stage 1 package import is not from current repo src")
    if workflow_path != expected_workflow or not workflow_path.is_file():
        raise ReviewedSourceError("Stage 1 workflow import is not from current repo src")
    return {
        "repo_src_root": src_root.as_posix(),
        "package_import_path": package_path.as_posix(),
        "workflow_import_path": workflow_path.as_posix(),
    }


def compute_stage1_source_identity(
    repo_root: str | Path,
    *,
    base_commit: str,
    package_import_path: str | Path | None = None,
    workflow_import_path: str | Path | None = None,
) -> dict[str, object]:
    """Compute the complete execution-time identity for the dirty Foundation Stage 1 tree."""

    repo = Path(repo_root).expanduser().resolve()
    imports = resolve_stage1_import_identity(
        repo,
        package_import_path=package_import_path,
        workflow_import_path=workflow_import_path,
    )
    return {
        "schema_version": "ppo_highres_frontier_stage1_execution_source_identity/v1",
        "prospective_git_tree": compute_stage1_prospective_tree(
            repo,
            base_commit=base_commit,
        ),
        "reviewed_path_count": len(STAGE1_REVIEWED_PATHS),
        "reviewed_source_set_sha256": compute_stage1_reviewed_source_set_sha256(repo),
        **imports,
    }


def compute_stage1_prospective_tree(
    repo_root: str | Path,
    *,
    base_commit: str,
    temporary_root: str | Path = "D:/xunce/tmp/ppo_frontier/git-index",
) -> str:
    repo = Path(repo_root).expanduser().resolve()
    head = _git(repo, ["rev-parse", "HEAD"], "HEAD").strip()
    if head != base_commit:
        raise ReviewedSourceError("review requires the frozen Foundation base HEAD")
    staged = subprocess.run(
        ["git", "-C", str(repo), "diff", "--cached", "--quiet"],
        check=False,
        capture_output=True,
    )
    if staged.returncode == 1:
        raise ReviewedSourceError("review requires an empty real Git index")
    if staged.returncode != 0:
        raise ReviewedSourceError("unable to verify the real Git index")
    dirty_paths = _status_paths(repo)
    out_of_scope = sorted(dirty_paths - STAGE1_REVIEWED_PATHS)
    if out_of_scope:
        raise ReviewedSourceError(f"out-of-scope reviewed source paths: {', '.join(out_of_scope)}")
    if not dirty_paths:
        raise ReviewedSourceError("reviewed Stage 1 source set is empty")

    temp_root = Path(temporary_root).expanduser().resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    index_path = temp_root / f"stage1-{uuid.uuid4().hex}.index"
    lock_path = Path(str(index_path) + ".lock")
    environment = os.environ.copy()
    environment["GIT_INDEX_FILE"] = str(index_path)
    try:
        _git(repo, ["read-tree", base_commit], "temporary read-tree", environment=environment)
        _git(
            repo,
            ["update-index", "--add", "--remove", "--", *sorted(dirty_paths)],
            "temporary update-index",
            environment=environment,
        )
        tree = _git(repo, ["write-tree"], "temporary write-tree", environment=environment).strip()
    finally:
        if lock_path.is_file():
            lock_path.unlink()
        if index_path.is_file():
            index_path.unlink()
    if len(tree) not in (40, 64) or any(character not in "0123456789abcdef" for character in tree):
        raise ReviewedSourceError("prospective Git tree OID is invalid")
    return tree


def _status_paths(repo: Path) -> set[str]:
    payload = _git_bytes(
        repo,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        "status",
    )
    return _status_paths_from_payload(payload)


def _status_paths_from_payload(payload: bytes) -> set[str]:
    records = payload.split(b"\0")
    paths: set[str] = set()
    index = 0
    while index < len(records):
        record = records[index]
        if not record:
            index += 1
            continue
        if len(record) < 4:
            raise ReviewedSourceError("Git status record is invalid")
        status = record[:2].decode("ascii", errors="strict")
        paths.add(record[3:].decode("utf-8", errors="strict").replace("\\", "/"))
        if "R" in status or "C" in status:
            index += 1
            if index >= len(records) or not records[index]:
                raise ReviewedSourceError("Git rename status record is incomplete")
            paths.add(records[index].decode("utf-8", errors="strict").replace("\\", "/"))
        index += 1
    return paths


def _git(
    repo: Path,
    arguments: list[str],
    label: str,
    *,
    environment: dict[str, str] | None = None,
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
        raise ReviewedSourceError(f"Git {label} failed: {completed.stderr.strip()}")
    return completed.stdout


def _git_bytes(repo: Path, arguments: list[str], label: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ReviewedSourceError(f"Git {label} failed")
    return completed.stdout
