from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Iterable

DEFAULT_SUBMODULES = ("dev-platform-constraints", "model-explorer", "path-planner")


def git_snapshot(repo_root: Path, *, submodules: Iterable[str] = DEFAULT_SUBMODULES) -> dict[str, Any]:
    parent = git_repo_state(repo_root, repo_root=repo_root, display_path=".")
    module_states = {
        name: git_repo_state(repo_root / name, repo_root=repo_root, display_path=name)
        for name in submodules
    }
    return {
        "parent": parent,
        "submodules": module_states,
        "dirty": bool(parent.get("dirty")) or any(bool(state.get("dirty")) for state in module_states.values()),
    }


def git_repo_state(path: Path, *, repo_root: Path, display_path: str | None = None) -> dict[str, Any]:
    status_lines = _git_lines(path, "status", "--porcelain=v1", "--untracked-files=normal")
    ignored_lines = _git_lines(path, "status", "--porcelain=v1", "--ignored", "--untracked-files=normal")
    tracked_modified_count = sum(
        1 for line in status_lines if line and not line.startswith("?? ") and not line.startswith("!! ")
    )
    untracked_count = sum(1 for line in status_lines if line.startswith("?? "))
    ignored_untracked_count = sum(1 for line in ignored_lines if line.startswith("!! "))
    dirty = tracked_modified_count > 0 or untracked_count > 0
    return {
        "path": display_path if display_path is not None else _display_path(path, repo_root),
        "sha": _git_output(path, "rev-parse", "HEAD") or "unknown",
        "branch": _git_output(path, "branch", "--show-current"),
        "dirty": dirty,
        "tracked_modified_count": tracked_modified_count,
        "untracked_count": untracked_count,
        "ignored_untracked_count": ignored_untracked_count,
        "dirty_fingerprint": _dirty_fingerprint(path) if dirty else None,
    }


def git_snapshots_match(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    submodules: Iterable[str] = DEFAULT_SUBMODULES,
    allow_dirty_match: bool = False,
) -> bool:
    if not left or not right:
        return False
    if _repo_sha(left.get("parent")) != _repo_sha(right.get("parent")):
        return False
    left_modules = left.get("submodules") if isinstance(left.get("submodules"), dict) else {}
    right_modules = right.get("submodules") if isinstance(right.get("submodules"), dict) else {}
    for name in submodules:
        if _repo_sha(left_modules.get(name)) != _repo_sha(right_modules.get(name)):
            return False
    if _declares_cleanliness(left):
        if allow_dirty_match:
            return _snapshot_clean_or_matching_dirty(left, right, submodules=submodules)
        if not _snapshot_is_clean(left) or not _snapshot_is_clean(right):
            return False
    for name in submodules:
        if _declares_cleanliness(left_modules.get(name)):
            if allow_dirty_match:
                if not _repo_clean_or_matching_dirty(left_modules.get(name), right_modules.get(name)):
                    return False
            elif not _repo_is_clean(left_modules.get(name)) or not _repo_is_clean(right_modules.get(name)):
                return False
    return True


def inspect_source_git_provenance(
    payload: dict[str, Any],
    *,
    label: str,
    current_git: dict[str, Any],
    require_current_git_match: bool,
    reason_codes: list[str],
    submodules: Iterable[str] = DEFAULT_SUBMODULES,
    allow_dirty_current_git_match: bool = False,
    match_flag_keys: Iterable[str] = (
        "current_matches_sources",
        "current_matches_application",
        "current_matches_robustness",
        "current_matches_batch",
        "runs_match_batch",
    ),
) -> bool:
    """Validate a source summary's current git snapshot against the active checkout.

    A missing ``git_provenance.current`` is a hard failure when current matching is
    required. Legacy SHA-only snapshots remain comparable through
    ``git_snapshots_match``; they are different from a missing snapshot.
    """
    if not require_current_git_match:
        return True
    git = payload.get("git_provenance") if isinstance(payload.get("git_provenance"), dict) else {}
    source_current = git.get("current") if isinstance(git.get("current"), dict) else {}
    source_matches = True
    if not source_current:
        _append_reason(reason_codes, "current_git_provenance_missing")
        _append_reason(reason_codes, f"{label}_current_git_provenance_missing")
        source_matches = False
    elif not git_snapshots_match(
        source_current,
        current_git,
        submodules=submodules,
        allow_dirty_match=allow_dirty_current_git_match,
    ):
        _append_reason(reason_codes, "current_git_provenance_mismatch")
        _append_reason(reason_codes, f"{label}_current_git_provenance_mismatch")
        source_matches = False
    for key in match_flag_keys:
        if git.get(key) is False:
            _append_reason(reason_codes, "git_provenance_mismatch")
            _append_reason(reason_codes, f"{label}_git_provenance_mismatch")
            source_matches = False
    return source_matches


def public_git(payload: dict[str, Any]) -> dict[str, Any]:
    git = payload.get("git_provenance") if isinstance(payload.get("git_provenance"), dict) else {}
    return dict(git)


def _append_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _git_output(path: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _git_bytes(path: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        return b""
    return completed.stdout


def _git_lines(path: Path, *args: str) -> list[str]:
    output = _git_output(path, *args)
    if not output:
        return []
    return output.splitlines()


def _dirty_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for label, payload in (
        ("status", _git_bytes(path, "status", "--porcelain=v1", "--untracked-files=all")),
        ("diff", _git_bytes(path, "diff", "--binary", "HEAD", "--")),
        ("cached", _git_bytes(path, "diff", "--cached", "--binary", "--")),
    ):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    for rel in _git_bytes(path, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0"):
        if not rel:
            continue
        rel_text = rel.decode("utf-8", errors="surrogateescape")
        target = path / rel_text
        digest.update(b"untracked\0")
        digest.update(rel)
        digest.update(b"\0")
        if target.is_file():
            try:
                digest.update(hashlib.sha256(target.read_bytes()).hexdigest().encode("ascii"))
            except OSError:
                digest.update(b"<unreadable>")
        digest.update(b"\0")
    return digest.hexdigest()


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _repo_sha(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    sha = value.get("sha")
    return str(sha) if sha is not None else None


def _declares_cleanliness(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if "dirty" in value:
        return True
    parent = value.get("parent")
    if isinstance(parent, dict) and "dirty" in parent:
        return True
    modules = value.get("submodules")
    if isinstance(modules, dict):
        return any(isinstance(item, dict) and "dirty" in item for item in modules.values())
    return False


def _snapshot_is_clean(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("dirty") is True:
        return False
    if not _repo_is_clean(value.get("parent")):
        return False
    modules = value.get("submodules")
    if isinstance(modules, dict):
        return all(_repo_is_clean(item) for item in modules.values())
    return True


def _snapshot_clean_or_matching_dirty(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    submodules: Iterable[str],
) -> bool:
    if not _repo_clean_or_matching_dirty(left.get("parent"), right.get("parent")):
        return False
    left_modules = left.get("submodules") if isinstance(left.get("submodules"), dict) else {}
    right_modules = right.get("submodules") if isinstance(right.get("submodules"), dict) else {}
    return all(
        _repo_clean_or_matching_dirty(left_modules.get(name), right_modules.get(name))
        for name in submodules
    )


def _repo_clean_or_matching_dirty(left: Any, right: Any) -> bool:
    if not isinstance(left, dict):
        return True
    if _repo_is_clean(left) and _repo_is_clean(right):
        return True
    if not isinstance(right, dict):
        return False
    if left.get("dirty") is True and right.get("dirty") is True:
        return (
            int(left.get("tracked_modified_count", 0) or 0)
            == int(right.get("tracked_modified_count", 0) or 0)
            and int(left.get("untracked_count", 0) or 0)
            == int(right.get("untracked_count", 0) or 0)
            and left.get("dirty_fingerprint") is not None
            and left.get("dirty_fingerprint") == right.get("dirty_fingerprint")
        )
    return False


def _repo_is_clean(value: Any) -> bool:
    if not isinstance(value, dict):
        return True
    return (
        value.get("dirty") is not True
        and int(value.get("tracked_modified_count", 0) or 0) == 0
        and int(value.get("untracked_count", 0) or 0) == 0
    )
