from __future__ import annotations

import json
from pathlib import Path

import pytest

from visual_workbench.artifacts import ArtifactService, WorkbenchSettings


SUPPORTED_SCHEMAS = {
    "model-explorer-contract/v1",
    "path-planner-sidecar/v1",
    "path-planner-route/v1",
    "path-feedback-manifest/v1",
    "path-feedback-summary/v1",
    "model-explorer-experiment/v1",
}


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def artifact_root(tmp_path: Path) -> Path:
    root = tmp_path / "outputs"
    _write_json(root / "contract.json", {"schema_version": "model-explorer-contract/v1", "grid": {"width": 2, "height": 2}, "top_goals": []})
    _write_json(root / "sidecar.json", {"schema_version": "path-planner-sidecar/v1", "grid": {"width": 2, "height": 2}, "cost": [[1, 2], [3, 4]], "passable_mask": [[True, True], [False, True]]})
    _write_json(root / "route.json", {"schema_version": "path-planner-route/v1", "reachable": True, "geometric_path": [[0, 0], [1, 1]], "path_cost": 2.0})
    _write_json(root / "manifest.json", {"schema_version": "path-feedback-manifest/v1", "top_k": 3, "scenarios": [{"scenario_id": "s1"}]})
    _write_json(root / "summary.json", {"schema_version": "path-feedback-summary/v1", "status": "passed", "scenario_count": 1, "open_grid_fallback_used": False})
    _write_json(root / "experiment.json", {"schema_version": "model-explorer-experiment/v1", "outputs": {"root": "out"}, "scenarios": []})
    (root / "report.md").write_text("# Report\n\nArtifact report.", encoding="utf-8")
    return root


def test_indexes_supported_schemas_and_non_json_reports(artifact_root: Path) -> None:
    service = ArtifactService(WorkbenchSettings(repo_root=artifact_root.parent, artifact_roots=(artifact_root,)))

    artifacts = service.list_artifacts()
    schemas = {artifact.schema_version for artifact in artifacts}
    names = {artifact.name for artifact in artifacts}

    assert SUPPORTED_SCHEMAS.issubset(schemas)
    assert "report.md" in names
    assert any(artifact.kind == "markdown" for artifact in artifacts)


def test_checked_in_demo_fixtures_cover_supported_schemas() -> None:
    fixture_root = Path(__file__).resolve().parents[1] / "fixtures" / "demo-artifacts"
    service = ArtifactService(WorkbenchSettings(repo_root=fixture_root.parent, artifact_roots=(fixture_root,)))

    schemas = {artifact.schema_version for artifact in service.list_artifacts()}

    assert SUPPORTED_SCHEMAS.issubset(schemas)


def test_reads_summary_and_raw_content_by_artifact_id(artifact_root: Path) -> None:
    service = ArtifactService(WorkbenchSettings(repo_root=artifact_root.parent, artifact_roots=(artifact_root,)))
    route = next(artifact for artifact in service.list_artifacts() if artifact.schema_version == "path-planner-route/v1")

    detail = service.get_artifact(route.artifact_id)
    raw = service.get_raw(route.artifact_id)

    assert detail.summary["reachable"] is True
    assert detail.summary["path_cost"] == 2.0
    assert raw.media_type == "application/json"
    assert b"path-planner-route/v1" in raw.content


def test_rejects_paths_outside_allowlisted_roots(tmp_path: Path, artifact_root: Path) -> None:
    service = ArtifactService(WorkbenchSettings(repo_root=tmp_path, artifact_roots=(artifact_root,)))
    outside = tmp_path / "secret.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(PermissionError):
        service.assert_allowed_path(outside)
