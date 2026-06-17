import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _load_batch_module(repo_root: Path):
    module_path = repo_root / "scripts" / "run_batch_path_feedback_validation.py"
    spec = importlib.util.spec_from_file_location("run_batch_path_feedback_validation", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_matrix(path: Path) -> Path:
    payload = {
        "schema_version": "path-feedback-batch-matrix/v1",
        "output_root": str(path.parent / "batch"),
        "runs": [
            {
                "run_id": "smoke",
                "scenario_set": "smoke",
                "diagnostic_profile": "baseline",
                "top_k": 3,
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_path_feedback_python_dry_run_does_not_create_output(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_path_feedback_validation.py",
            "--dry-run",
            "--scenario-set",
            "all",
            "--diagnostic-profile",
            "all",
            "--top-k",
            "3",
            "--output-root",
            str(output_root),
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Acceptance gate: semi-real-closed-loop" in completed.stdout
    assert "Python executable:" in completed.stdout
    assert "bash" not in completed.stdout.lower()
    assert not output_root.exists()


def test_batch_default_launcher_is_python_and_validate_only_needs_no_bash(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    matrix = _write_matrix(tmp_path / "matrix.json")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_batch_path_feedback_validation.py",
            "--matrix",
            str(matrix),
            "--dry-run",
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "command_launcher" in output
    assert "python" in output
    assert "run_path_feedback_validation.py" in output
    assert "bash" not in output.lower()


def test_batch_rejects_shell_single_run_script_on_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_batch_module(repo_root)
    matrix = _write_matrix(tmp_path / "matrix.json")
    shell_script = tmp_path / "fake_single_run.sh"
    shell_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    payload = module._load_matrix_json(matrix)

    monkeypatch.setattr(module.sys, "platform", "win32")

    with pytest.raises(module.MatrixError, match="bash_single_run_script_unsupported_on_windows"):
        module._build_batch_plan(
            payload,
            matrix_path=matrix,
            repo_root=repo_root,
            cli_output_root=None,
            single_run_script=shell_script,
        )
