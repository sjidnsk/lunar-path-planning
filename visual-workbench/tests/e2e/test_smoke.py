from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from visual_workbench.api import create_app
from visual_workbench.artifacts import WorkbenchSettings


def test_e2e_fixture_artifact_scan_and_project_status(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (tmp_path / "dev-platform-constraints").mkdir()
    (tmp_path / "model-explorer").mkdir()
    (tmp_path / "path-planner").mkdir()
    (outputs / "summary.json").write_text(
        json.dumps({"schema_version": "path-feedback-summary/v1", "status": "passed", "scenario_count": 1}),
        encoding="utf-8",
    )

    client = TestClient(create_app(WorkbenchSettings(repo_root=tmp_path, artifact_roots=(outputs,))))

    status = client.get("/api/project/status").json()
    artifacts = client.get("/api/artifacts").json()["artifacts"]

    assert status["subprojects"]["dev-platform-constraints"]["exists"] is True
    assert status["subprojects"]["model-explorer"]["exists"] is True
    assert status["subprojects"]["path-planner"]["exists"] is True
    assert artifacts[0]["schema_version"] == "path-feedback-summary/v1"
