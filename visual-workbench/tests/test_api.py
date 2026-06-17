from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from visual_workbench.api import create_app
from visual_workbench.artifacts import WorkbenchSettings


def test_api_exposes_health_artifacts_raw_and_rejects_run(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": "path-feedback-manifest/v1", "scenarios": []}), encoding="utf-8")
    client = TestClient(create_app(WorkbenchSettings(repo_root=tmp_path, artifact_roots=(root,))))

    assert client.get("/api/health").json()["status"] == "ok"

    artifacts = client.get("/api/artifacts").json()["artifacts"]
    assert artifacts[0]["schema_version"] == "path-feedback-manifest/v1"
    artifact_id = artifacts[0]["artifact_id"]

    assert client.get(f"/api/artifacts/{artifact_id}").json()["name"] == "manifest.json"
    assert "path-feedback-manifest/v1" in client.get(f"/api/artifacts/{artifact_id}/raw").text

    blocked = client.post("/api/commands/validate", json={"kind": "path-feedback", "action": "run", "manifest_path": str(manifest)})
    assert blocked.status_code == 400
    assert "run" in blocked.json()["detail"]
