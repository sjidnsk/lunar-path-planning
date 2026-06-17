from __future__ import annotations

import json
from pathlib import Path

import pytest

from visual_workbench.commands import CommandRequest, CommandService, CommandRejected
from visual_workbench.artifacts import WorkbenchSettings


def test_rejects_full_run_even_when_manifest_is_allowed(tmp_path: Path) -> None:
    manifest = tmp_path / "outputs" / "path-feedback.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"schema_version": "path-feedback-manifest/v1"}), encoding="utf-8")
    service = CommandService(WorkbenchSettings(repo_root=tmp_path, artifact_roots=(manifest.parent,)))

    with pytest.raises(CommandRejected, match="完整实验 run"):
        service.execute(CommandRequest(kind="path-feedback", action="run", manifest_path=str(manifest)))


def test_builds_whitelisted_path_feedback_validate_command(tmp_path: Path) -> None:
    manifest = tmp_path / "outputs" / "path-feedback.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"schema_version": "path-feedback-manifest/v1"}), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_runner(args: list[str], cwd: Path, env: dict[str, str]) -> tuple[int, str, str]:
        captured["args"] = args
        captured["cwd"] = cwd
        captured["env"] = env
        return 0, '{"status":"valid"}', ""

    service = CommandService(
        WorkbenchSettings(repo_root=tmp_path, artifact_roots=(manifest.parent,), python_executable="python"),
        runner=fake_runner,
    )

    result = service.execute(CommandRequest(kind="path-feedback", action="validate", manifest_path=str(manifest)))

    assert result.status == "completed"
    assert captured["args"] == ["python", "-m", "model_explorer", "path-feedback", "validate", str(manifest.resolve())]
    assert captured["cwd"] == tmp_path / "model-explorer"
    assert result.stdout_json == {"status": "valid"}


def test_rejects_manifest_outside_artifact_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "outputs"
    allowed.mkdir()
    manifest = tmp_path / "path-feedback.json"
    manifest.write_text("{}", encoding="utf-8")
    service = CommandService(WorkbenchSettings(repo_root=tmp_path, artifact_roots=(allowed,)))

    with pytest.raises(PermissionError):
        service.execute(CommandRequest(kind="experiment", action="dry-run", manifest_path=str(manifest)))
