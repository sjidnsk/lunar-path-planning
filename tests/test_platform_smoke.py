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


def test_platform_ci_and_setup_documentation_have_no_retired_bindings() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = (repo_root / ".github/workflows/platform-compatibility.yml").read_text(encoding="utf-8")
    setup = (repo_root / "docs/platform/windows-ubuntu-setup.md").read_text(encoding="utf-8")

    for text in (workflow, setup):
        assert "model-explorer" not in text
        assert "visual-workbench" not in text
        assert "path_feedback" not in text
        assert "path-feedback" not in text
        assert "--with-visual-workbench" not in text
        assert "test_path_feedback_windows_compat.py" not in text
    assert "actions/setup-node" not in workflow
    assert "test_stage6_standard_config.py" in workflow
