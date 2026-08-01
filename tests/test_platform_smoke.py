import subprocess
import sys
from pathlib import Path


def test_platform_smoke_dry_run_uses_only_retained_parent_and_submodule_checks() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_platform_smoke.py",
            "--profile",
            "windows-non-drake",
            "--dry-run",
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "test_stage6_standard_config.py" in output
    assert "path-planner" in output
    assert "dev-platform-constraints" in output
    assert "model-explorer" not in output
    assert "visual-workbench" not in output
    assert "path_feedback" not in output
    assert "path-feedback" not in output
