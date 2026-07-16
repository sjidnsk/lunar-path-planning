from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import xunce_artifact_io as artifact_io


BOUNDARY_FIELDS = {
    "publishes_checkpoint": False,
    "replaces_default_policy": False,
    "connects_real_executor": False,
    "starts_online_canary": False,
}


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
