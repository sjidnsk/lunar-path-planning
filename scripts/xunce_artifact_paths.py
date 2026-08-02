from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xunce_artifact_io import path_is_file, read_json, read_jsonl, write_json, write_jsonl


@dataclass(frozen=True)
class ArtifactName:
    key: str
    canonical: str
    legacy: tuple[str, ...] = ()

    def candidates(self) -> tuple[str, ...]:
        return (self.canonical, *self.legacy)


# The retained G1/G2/G3 chain uses only these short canonical artifact names.
MID_DUAL_CONFIG = ArtifactName("mid_dual_config", "config.json")
MID_DUAL_RESULTS = ArtifactName("mid_dual_results", "results.jsonl")
MID_DUAL_SUMMARY = ArtifactName("mid_dual_summary", "summary.json")
MID_DUAL_ROUTING = ArtifactName("mid_dual_routing", "routing.json")
MID_DUAL_MANIFEST = ArtifactName("mid_dual_manifest", "manifest.json")
MID_DUAL_PHASE_STATE = ArtifactName("mid_dual_phase_state", "phase-state.jsonl")
MID_DUAL_REPORT = ArtifactName("mid_dual_report", "report.md")


def resolve_artifact(root: str | Path, artifact: ArtifactName) -> tuple[Path | None, str | None]:
    base = Path(root)
    for index, name in enumerate(artifact.candidates()):
        path = base / name
        if path_is_file(path):
            return path, "canonical" if index == 0 else "legacy"
    return None, None


def artifact_path(root: str | Path, artifact: ArtifactName, *, canonical: bool = True) -> Path:
    return Path(root) / (artifact.canonical if canonical else artifact.legacy[0])


def missing_reason(artifact: ArtifactName) -> str:
    return f"missing_{artifact.legacy[0] if artifact.legacy else artifact.canonical}"


def read_json_artifact(root: str | Path, artifact: ArtifactName) -> tuple[dict[str, Any], str]:
    path, source = resolve_artifact(root, artifact)
    if path is None or source is None:
        raise FileNotFoundError(missing_reason(artifact))
    payload = read_json(path)
    payload.setdefault("artifact_source", source)
    return payload, source


def read_jsonl_artifact(root: str | Path, artifact: ArtifactName) -> tuple[list[dict[str, Any]], str]:
    path, source = resolve_artifact(root, artifact)
    if path is None or source is None:
        raise FileNotFoundError(missing_reason(artifact))
    return read_jsonl(path), source


def write_json_artifact(root: str | Path, artifact: ArtifactName, payload: dict[str, Any], *, dual_write: bool = True) -> None:
    canonical_payload = dict(payload)
    canonical_payload.setdefault("artifact_source", "canonical")
    write_json(artifact_path(root, artifact), canonical_payload)
    if dual_write:
        for legacy in artifact.legacy:
            legacy_payload = dict(payload)
            legacy_payload.setdefault("artifact_source", "legacy")
            write_json(Path(root) / legacy, legacy_payload)


def write_jsonl_artifact(root: str | Path, artifact: ArtifactName, rows: list[dict[str, Any]], *, dual_write: bool = True) -> None:
    write_jsonl(artifact_path(root, artifact), rows)
    if dual_write:
        for legacy in artifact.legacy:
            write_jsonl(Path(root) / legacy, rows)
