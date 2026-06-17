from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from .artifacts import WorkbenchSettings, _is_relative_to


Runner = Callable[[list[str], Path, dict[str, str]], tuple[int, str, str]]


class CommandRejected(ValueError):
    pass


class CommandRequest(BaseModel):
    kind: str
    action: str
    manifest_path: str


class CommandResult(BaseModel):
    status: str
    kind: str
    action: str
    command: list[str]
    cwd: str
    return_code: int
    stdout: str
    stderr: str
    stdout_json: dict[str, Any] | None = None


class CommandService:
    def __init__(self, settings: WorkbenchSettings, runner: Runner | None = None):
        self.settings = settings
        self.runner = runner or _subprocess_runner

    def execute(self, request: CommandRequest) -> CommandResult:
        if request.action == "run":
            raise CommandRejected("禁止从可视化工作台触发完整实验 run；首版只允许 dry-run/validate。")
        if request.action not in {"dry-run", "validate"}:
            raise CommandRejected(f"unsupported command action: {request.action}")
        if request.kind not in {"path-feedback", "experiment"}:
            raise CommandRejected(f"unsupported command kind: {request.kind}")

        manifest = self._allowed_manifest(request.manifest_path)
        cwd = self.settings.normalized_repo_root() / "model-explorer"
        args = [
            self.settings.python_executable,
            "-m",
            "model_explorer",
            request.kind,
            request.action,
            str(manifest),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(cwd / "src")
        return_code, stdout, stderr = self.runner(args, cwd, env)
        stdout_json = _parse_json(stdout)
        return CommandResult(
            status="completed" if return_code == 0 else "failed",
            kind=request.kind,
            action=request.action,
            command=args,
            cwd=str(cwd),
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            stdout_json=stdout_json,
        )

    def _allowed_manifest(self, path: str | Path) -> Path:
        resolved = Path(path).resolve()
        roots = self.settings.resolved_artifact_roots()
        if not any(_is_relative_to(resolved, root) for root in roots):
            raise PermissionError(f"manifest path is outside allowlisted artifact roots: {resolved}")
        if not resolved.is_file():
            raise FileNotFoundError(f"manifest does not exist: {resolved}")
        return resolved


def _subprocess_runner(args: list[str], cwd: Path, env: dict[str, str]) -> tuple[int, str, str]:
    completed = subprocess.run(
        args,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    return completed.returncode, completed.stdout, completed.stderr


def _parse_json(stdout: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
