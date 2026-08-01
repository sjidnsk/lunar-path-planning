import ast
import re
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


def test_platform_ci_excludes_tests_that_access_retired_submodule_sources() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = (repo_root / ".github/workflows/platform-compatibility.yml").read_text(encoding="utf-8")
    ci_test_paths = {
        Path(match).as_posix()
        for match in re.findall(r"tests/[A-Za-z0-9_./-]+\.py", workflow)
    }
    retired_source_tests: set[str] = set()
    for relative in ci_test_paths:
        source = (repo_root / relative).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        imports_retired_module = any(
            (isinstance(node, ast.Import) and any(alias.name.startswith(("model_explorer", "visual_workbench")) for alias in node.names))
            or (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.startswith(("model_explorer", "visual_workbench"))
            )
            for node in ast.walk(tree)
        )
        accesses_retired_directory = any(
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and isinstance(node.right, ast.Constant)
            and node.right.value in {"model-explorer", "visual-workbench"}
            for node in ast.walk(tree)
        )
        if imports_retired_module or accesses_retired_directory:
            retired_source_tests.add(relative)

    assert "tests/test_xunce_high_fidelity_real_map_comparison.py" not in ci_test_paths
    assert "tests/test_xunce_mid_dual_g2_inputs.py" in ci_test_paths
    assert retired_source_tests == set()
