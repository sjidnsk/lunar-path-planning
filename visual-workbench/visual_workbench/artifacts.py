from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


SUPPORTED_SCHEMA_VERSIONS = {
    "model-explorer-contract/v1",
    "path-planner-sidecar/v1",
    "path-planner-route/v1",
    "path-feedback-manifest/v1",
    "path-feedback-summary/v1",
    "model-explorer-experiment/v1",
}

TEXT_SUFFIXES = {".json", ".md", ".html", ".htm", ".txt", ".jsonl", ".csv"}
SCAN_SUFFIXES = TEXT_SUFFIXES | {".png"}


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _split_env_paths(raw: str | None) -> tuple[Path, ...]:
    if not raw:
        return ()
    parts = [part.strip() for part in raw.split(os.pathsep) if part.strip()]
    return tuple(Path(part) for part in parts)


class WorkbenchSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    repo_root: Path = Field(default_factory=default_repo_root)
    artifact_roots: tuple[Path, ...] | None = None
    python_executable: str = "python"

    def normalized_repo_root(self) -> Path:
        return self.repo_root.resolve()

    def resolved_artifact_roots(self) -> tuple[Path, ...]:
        if self.artifact_roots is not None:
            roots = self.artifact_roots
        else:
            repo = self.normalized_repo_root()
            env_roots = _split_env_paths(os.environ.get("VISUAL_WORKBENCH_ARTIFACT_ROOTS"))
            roots = (
                repo / "outputs",
                repo / "dev-platform-constraints" / "outputs",
                repo / "model-explorer" / "outputs",
                repo / "path-planner" / "outputs",
                *env_roots,
            )
        return tuple(root.resolve() for root in roots)


class ArtifactSummary(BaseModel):
    artifact_id: str
    name: str
    path: str
    relative_path: str
    kind: str
    schema_version: str | None = None
    status: str = "unknown"
    size_bytes: int
    modified_at: str


class ArtifactDetail(ArtifactSummary):
    summary: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class RawArtifact:
    content: bytes
    media_type: str
    filename: str


class ArtifactService:
    def __init__(self, settings: WorkbenchSettings):
        self.settings = settings

    def list_artifacts(self) -> list[ArtifactSummary]:
        artifacts: list[ArtifactSummary] = []
        for root in self.settings.resolved_artifact_roots():
            if not root.exists() or not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in SCAN_SUFFIXES:
                    continue
                try:
                    artifacts.append(self._summarize_path(path))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    artifacts.append(self._summarize_path(path, payload={}))
        artifacts.sort(key=lambda item: item.modified_at, reverse=True)
        return artifacts

    def get_artifact(self, artifact_id: str) -> ArtifactDetail:
        path = self._path_for_id(artifact_id)
        payload = self._read_json_payload(path) if path.suffix.lower() == ".json" else {}
        summary = self._summary_fields(payload)
        base = self._summarize_path(path, payload=payload)
        return ArtifactDetail(**base.model_dump(), summary=summary)

    def get_raw(self, artifact_id: str) -> RawArtifact:
        path = self._path_for_id(artifact_id)
        media_type = _media_type(path)
        return RawArtifact(content=path.read_bytes(), media_type=media_type, filename=path.name)

    def assert_allowed_path(self, path: str | Path) -> Path:
        resolved = Path(path).resolve()
        for root in self.settings.resolved_artifact_roots():
            if _is_relative_to(resolved, root):
                return resolved
        raise PermissionError(f"path is outside allowlisted artifact roots: {resolved}")

    def _path_for_id(self, artifact_id: str) -> Path:
        for artifact in self.list_artifacts():
            if artifact.artifact_id == artifact_id:
                return self.assert_allowed_path(artifact.path)
        raise FileNotFoundError(f"artifact not found: {artifact_id}")

    def _summarize_path(self, path: Path, payload: dict[str, Any] | None = None) -> ArtifactSummary:
        allowed = self.assert_allowed_path(path)
        payload = self._read_json_payload(allowed) if payload is None and allowed.suffix.lower() == ".json" else payload
        stat = allowed.stat()
        schema_version = _schema_version(payload or {})
        return ArtifactSummary(
            artifact_id=_artifact_id(allowed),
            name=allowed.name,
            path=str(allowed),
            relative_path=_relative_display_path(allowed, self.settings.normalized_repo_root()),
            kind=_kind_for_path(allowed),
            schema_version=schema_version,
            status=_status(payload or {}),
            size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        )

    def _read_json_payload(self, path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}

    def _summary_fields(self, payload: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "schema_version",
            "status",
            "reason_codes",
            "next_required_change",
            "scenario_count",
            "scenario_set",
            "diagnostic_profile",
            "top_k",
            "candidate_count",
            "reachable_count",
            "path_planning_failure_count",
            "replan_count",
            "selection_changed_rate",
            "open_grid_fallback_used",
            "reachable",
            "path_cost",
            "failure_reason",
            "grid",
            "outputs",
            "policy_ranking",
            "gate_summary",
            "per_group_winners",
            "failure_scenarios",
        )
        return {key: payload[key] for key in keys if key in payload}


def _artifact_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _schema_version(payload: dict[str, Any]) -> str | None:
    value = payload.get("schema_version")
    if isinstance(value, str):
        return value
    return None


def _status(payload: dict[str, Any]) -> str:
    for key in ("status", "readiness_status", "verdict"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    if payload.get("reachable") is True:
        return "reachable"
    if payload.get("reachable") is False:
        return "blocked"
    return "unknown"


def _kind_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".md":
        return "markdown"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix == ".png":
        return "image"
    return "text"


def _media_type(path: Path) -> str:
    if path.suffix.lower() == ".md":
        return "text/markdown; charset=utf-8"
    if path.suffix.lower() == ".json":
        return "application/json"
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _relative_display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
