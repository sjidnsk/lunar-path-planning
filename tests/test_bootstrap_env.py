import subprocess
import sys
from pathlib import Path


def test_windows_bootstrap_dry_run_uses_d_drive_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/bootstrap_env.py",
            "--dry-run",
            "--platform",
            "windows",
            "--install-editable",
            "--with-training",
            "--with-visual-workbench",
            "--run-validation",
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "Platform: windows" in output
    assert "Conda environment target: -p D:\\conda_envs\\lunar-explorer" in output
    assert "Download root: D:\\CodexDownloads\\lunar-path-planning" in output
    assert "python=3.12" in output
    assert "path-planner" in output
    assert "model-explorer[training]" in output
    assert "dev-platform-constraints" in output
    assert "visual-workbench" in output
    assert "pytest" in output


def test_ubuntu_bootstrap_dry_run_uses_named_env_default() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/bootstrap_env.py",
            "--dry-run",
            "--platform",
            "ubuntu",
            "--run-validation",
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "Platform: ubuntu" in output
    assert "Conda environment target: -n lunar-explorer" in output
    assert "python=3.12" in output
    assert "pydrake" not in output.lower()
