import json
import subprocess
import sys
from pathlib import Path


def test_platform_validation_matrix_dry_run_does_not_create_output_root(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "matrix"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_platform_validation_matrix.py",
            "--profile",
            "windows-non-drake",
            "--dry-run",
            "--output-root",
            str(output_root),
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "[DRY RUN]" in completed.stdout
    assert "bootstrap_env.py" in completed.stdout
    assert "npm" in completed.stdout
    assert not output_root.exists()


def test_platform_validation_matrix_skip_node_summary_schema(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_platform_validation_matrix as matrix

    output_root = tmp_path / "matrix"

    def fake_run(_argv, **_kwargs):
        return subprocess.CompletedProcess(_argv, 0)

    monkeypatch.setattr(matrix.subprocess, "run", fake_run)

    code = matrix.main(
        [
            "--profile",
            "windows-non-drake",
            "--skip-bootstrap",
            "--skip-node",
            "--output-root",
            str(output_root),
        ]
    )

    assert code == 0
    summary = json.loads((output_root / "platform-validation-matrix-summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "platform-validation-matrix-summary/v1"
    assert summary["profile"] == "windows-non-drake"
    assert summary["bootstrap_mode"] == "skipped"
    assert summary["node_validation_executed"] is False
    assert summary["drake_validation_executed"] is False
    assert summary["final_status"] == "passed"
    assert summary["failure_reason"] == ""
    assert summary["command_results"]


def test_platform_validation_matrix_ubuntu_drake_missing_routes_to_exit_2(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_platform_validation_matrix as matrix

    output_root = tmp_path / "matrix"

    def fake_run(_argv, **_kwargs):
        return subprocess.CompletedProcess(_argv, 1)

    monkeypatch.setattr(matrix.subprocess, "run", fake_run)

    code = matrix.main(["--profile", "ubuntu-drake", "--output-root", str(output_root)])

    assert code == 2
    summary = json.loads((output_root / "platform-validation-matrix-summary.json").read_text(encoding="utf-8"))
    assert summary["profile"] == "ubuntu-drake"
    assert summary["drake_validation_executed"] is True
    assert summary["final_status"] == "failed"
    assert summary["failure_reason"] == "drake_runtime_missing"
