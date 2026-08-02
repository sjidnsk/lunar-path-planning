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
    assert "run_stage.py" not in output


def test_platform_ci_and_setup_documentation_have_no_retired_bindings() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = (repo_root / ".github/workflows/platform-compatibility.yml").read_text(encoding="utf-8")
    setup = (repo_root / "docs/platform/windows-ubuntu-setup.md").read_text(encoding="utf-8")

    for text in (workflow, setup):
        assert "model-explorer" not in text
        assert "visual-workbench" not in text
        assert "run_stage.py" not in text
    assert "actions/setup-node" not in workflow
    assert "test_stage6_standard_config.py" in workflow


def test_platform_ci_installs_local_path_planner_before_parent_wheel() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = (repo_root / ".github/workflows/platform-compatibility.yml").read_text(encoding="utf-8")

    path_planner_install = workflow.index("python -m pip install -e path-planner")
    parent_wheel_install = workflow.index("python -m pip install dist/*.whl")

    assert path_planner_install < parent_wheel_install


def test_platform_ci_excludes_identity_bound_foundation_gate_suite() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = (repo_root / ".github/workflows/platform-compatibility.yml").read_text(encoding="utf-8")

    assert "tests/ppo_highres_frontier/test_foundation.py" not in workflow
    assert 'resolve_device("cpu", cuda_available=False) == "cpu"' in workflow


def test_platform_profiles_exclude_only_explicit_external_evidence_replay() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = (repo_root / ".github/workflows/platform-compatibility.yml").read_text(
        encoding="utf-8"
    )
    stage6_source = (
        repo_root / "tests/ppo_highres_frontier/test_stage6_standard_config.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(stage6_source)

    external_evidence_tests = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            ast.unparse(decorator) == "pytest.mark.external_evidence"
            for decorator in node.decorator_list
        )
    }

    assert external_evidence_tests == {
        "test_stage5_gate_is_canonical_hash_chained_and_read_only",
        "test_stage5_gate_bytes_fail_closed_on_tamper",
        "test_stage5_gate_bytes_reject_noncanonical_json",
    }
    assert '-m "not external_evidence"' in workflow

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
    assert re.search(
        r"test_stage6_standard_config\.py -m [\"']not external_evidence[\"'] -q",
        output,
    )


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

    assert "tests/test_xunce_mid_dual_g2_inputs.py" in ci_test_paths
    assert retired_source_tests == set()
