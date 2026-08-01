import ast
from pathlib import Path


LEGACY_BASH_ALLOWED = {
    # Explicit .sh compatibility path for legacy Ubuntu wrappers. Windows
    # rejects this path before launch.
    "scripts/run_batch_path_feedback_validation.py",
}

PLATFORM_SUPPORTED_CHAIN = {
    "scripts/run_policy_gated_sequential_canary_rollout.py",
    "scripts/run_guarded_ppo_rollout_pilot.py",
    "scripts/run_iterative_ppo_mini_loop_stability.py",
    "scripts/run_quasi_real_guarded_ppo_stability_replay.py",
}


def _read_python_source(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _literal_command_list(node: ast.AST) -> list[str] | None:
    if not isinstance(node, ast.List):
        return None
    values: list[str] = []
    for item in node.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            return None
        values.append(item.value)
    return values


def test_no_platform_supported_python_script_invokes_bash_directly() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for path in sorted((repo_root / "scripts").glob("*.py")):
        relative = path.relative_to(repo_root).as_posix()
        if relative in LEGACY_BASH_ALLOWED:
            continue
        tree = ast.parse(_read_python_source(path), filename=relative)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "run":
                continue
            command = _literal_command_list(node.args[0]) if node.args else None
            if command and command[0] == "bash":
                offenders.append(relative)

    assert offenders == []


def test_platform_supported_chain_does_not_reference_shell_wrappers() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for relative in sorted(PLATFORM_SUPPORTED_CHAIN):
        text = _read_python_source(repo_root / relative)
        if '"bash"' in text or "'.sh'" in text or ".sh" in text:
            offenders.append(relative)

    assert offenders == []


def test_policy_training_readiness_review_is_called_as_python() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted((repo_root / "scripts").glob("*.py")):
        text = _read_python_source(path)
        if "run_policy_training_readiness_review.sh" in text:
            offenders.append(path.relative_to(repo_root).as_posix())

    assert offenders == []


def test_python_source_reader_strips_utf8_bom(tmp_path: Path) -> None:
    source = tmp_path / "bom_script.py"
    source.write_bytes(b"\xef\xbb\xbffrom __future__ import annotations\n")

    assert _read_python_source(source) == "from __future__ import annotations\n"
