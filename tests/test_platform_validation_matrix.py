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
    assert "path-planner" in completed.stdout
    assert "model-explorer" not in completed.stdout
    assert "visual-workbench" not in completed.stdout
    assert "run_stage.py" not in completed.stdout
    assert not output_root.exists()


def test_platform_validation_matrix_uses_only_retained_submodule_checks() -> None:
    from scripts import run_platform_validation_matrix as matrix

    repo_root = Path(__file__).resolve().parents[1]
    commands = matrix._commands_for_profile(
        profile="windows-non-drake",
        repo_root=repo_root,
        skip_bootstrap=False,
        real_bootstrap=False,
    )
    rendered = "\n".join(
        f"{command['label']} {command['cwd']} {' '.join(command['argv'])}" for command in commands
    )

    assert "path_planner_non_drake_tests" in rendered
    assert "dev_platform_constraints_tests" in rendered
    assert "tests/test_platform_smoke.py" in rendered
    assert "model-explorer" not in rendered
    assert "visual-workbench" not in rendered
    assert "xunce_stage_15_18_tests" not in rendered


def test_platform_validation_matrix_summary_has_no_node_contract(tmp_path: Path, monkeypatch) -> None:
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
            "--output-root",
            str(output_root),
        ]
    )

    assert code == 0
    summary = json.loads((output_root / "platform-validation-matrix-summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "platform-validation-matrix-summary/v1"
    assert summary["profile"] == "windows-non-drake"
    assert summary["bootstrap_mode"] == "skipped"
    assert "node_validation_executed" not in summary
    assert summary["drake_validation_executed"] is False
    assert summary["final_status"] == "passed"
    assert summary["failure_reason"] == ""
    assert summary["command_results"]


def test_platform_matrix_contract_is_fresh_checkout_safe(tmp_path: Path) -> None:
    from scripts import bootstrap_env
    from scripts import run_platform_validation_matrix as matrix

    fresh_checkout = tmp_path / "fresh-checkout"
    (fresh_checkout / "path-planner").mkdir(parents=True)
    (fresh_checkout / "dev-platform-constraints").mkdir()
    commands = matrix._commands_for_profile(
        profile="windows-non-drake",
        repo_root=fresh_checkout,
        skip_bootstrap=False,
        real_bootstrap=False,
    )
    rendered = "\n".join(
        f"{command['label']} {command['cwd']} {' '.join(command['argv'])}" for command in commands
    )

    assert bootstrap_env.MODULES == ("path-planner", "dev-platform-constraints")
    assert "model-explorer" not in rendered
    assert "visual-workbench" not in rendered


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
