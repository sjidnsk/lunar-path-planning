from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any

import xunce_artifact_io as artifact_io


BOUNDARY_FIELDS = {
    "publishes_checkpoint": False,
    "replaces_default_policy": False,
    "connects_real_executor": False,
    "starts_online_canary": False,
}
GATE_ARTIFACT_NAMES = frozenset(
    {
        "config.json",
        "summary.json",
        "routing.json",
        "results.jsonl",
        "phase-state.jsonl",
        "review.json",
        "report.md",
        "manifest.json",
    }
)


def build_manifest_without_self_hash(output_root: Path) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    root = Path(output_root).resolve()
    safe_root = artifact_io.windows_safe_path(root)
    for directory, directory_names, file_names in os.walk(safe_root):
        directory_names.sort()
        for file_name in sorted(file_names):
            path = Path(directory) / file_name
            relative_path = Path(os.path.relpath(str(path), safe_root)).as_posix()
            if relative_path == "manifest.json":
                continue
            content = artifact_io.read_bytes(path)
            artifacts.append(
                {
                    "relative_path": relative_path,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size": len(content),
                }
            )
    artifacts.sort(key=lambda item: item["relative_path"])
    return {
        "schema_version": "xunce-path-v2-gate-manifest/v1",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def write_gate_artifacts(
    *,
    output_root: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    routing: dict[str, Any],
    rows: list[dict[str, Any]],
    phases: list[dict[str, Any]],
    review: dict[str, Any],
    report: str,
) -> dict[str, Any]:
    artifact_io.make_dirs(output_root)
    artifact_io.write_json(output_root / "config.json", config)
    artifact_io.write_json(output_root / "summary.json", summary)
    artifact_io.write_json(output_root / "routing.json", routing)
    artifact_io.write_jsonl(output_root / "results.jsonl", rows)
    artifact_io.write_jsonl(output_root / "phase-state.jsonl", phases)
    artifact_io.write_json(output_root / "review.json", review)
    artifact_io.write_text(output_root / "report.md", report)
    manifest = build_manifest_without_self_hash(output_root)
    artifact_io.write_json(output_root / "manifest.json", manifest)
    return manifest


def _validate_staged_gate_artifacts(staging_root: Path) -> dict[str, Any]:
    safe_root = artifact_io.windows_safe_path(Path(staging_root).resolve())
    entries = list(os.scandir(safe_root))
    if {entry.name for entry in entries} != GATE_ARTIFACT_NAMES or any(
        not entry.is_file(follow_symlinks=False) for entry in entries
    ):
        raise ValueError("staging root must contain exactly eight regular artifacts")
    stored_manifest = artifact_io.read_json(Path(staging_root) / "manifest.json")
    expected_manifest = build_manifest_without_self_hash(staging_root)
    if stored_manifest != expected_manifest:
        raise ValueError("staging manifest does not match artifact bytes")
    return stored_manifest


def write_gate_artifacts_atomically(
    *,
    output_root: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    routing: dict[str, Any],
    rows: list[dict[str, Any]],
    phases: list[dict[str, Any]],
    review: dict[str, Any],
    report: str,
) -> dict[str, Any]:
    target = Path(output_root).resolve()
    safe_target = artifact_io.windows_safe_path(target)
    if os.path.lexists(safe_target):
        raise RuntimeError("atomic gate artifact publish target already exists")
    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    artifact_io.make_dirs(staging)
    try:
        manifest = write_gate_artifacts(
            output_root=staging,
            config=config,
            summary=summary,
            routing=routing,
            rows=rows,
            phases=phases,
            review=review,
            report=report,
        )
    except Exception as exc:
        raise RuntimeError("atomic gate artifact staging write failed") from exc
    try:
        manifest = _validate_staged_gate_artifacts(staging)
    except Exception as exc:
        raise RuntimeError("atomic gate artifact staging validation failed") from exc
    if os.path.lexists(safe_target):
        raise RuntimeError("atomic gate artifact publish target appeared")
    try:
        os.rename(
            artifact_io.windows_safe_path(staging),
            safe_target,
        )
    except Exception as exc:
        raise RuntimeError("atomic gate artifact publish rename failed") from exc
    return manifest
