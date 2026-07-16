from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


BOUNDARIES = {
    "publishes_checkpoint": False,
    "replaces_default_policy": False,
    "connects_real_executor": False,
    "starts_online_canary": False,
}
FOCUSED_TARGETS = [
    "tests/test_v2_contracts.py",
    "tests/test_v2_serialization.py",
    "tests/test_v2_terrain.py",
    "tests/test_v2_fine_safety_anchor.py",
    "tests/test_v2_profiles.py",
    "tests/test_v2_api.py",
]
CANONICAL_ARTIFACTS = {
    "config.json",
    "summary.json",
    "routing.json",
    "results.jsonl",
    "phase-state.jsonl",
    "review.json",
    "report.md",
    "manifest.json",
}


def _runner():
    try:
        return importlib.import_module("run_xunce_path_v2_gate_benchmark")
    except ModuleNotFoundError as exc:
        raise AssertionError("Gate 1 runner production module does not exist yet") from exc


def _testcase(nodeid: str, outcome: str = "passed", message: str = "") -> str:
    path, name = nodeid.split("::", 1)
    classname = path.removesuffix(".py").replace("/", ".")
    child = ""
    if outcome == "failed":
        child = '<failure message="failed">failed</failure>'
    elif outcome == "error":
        child = '<error message="error">error</error>'
    elif outcome == "skipped":
        child = f'<skipped message="{message}" />'
    return f'<testcase classname="{classname}" name="{name}">{child}</testcase>'


def _write_junit(path: Path, cases: list[tuple[str, str, str]]) -> None:
    body = "".join(_testcase(*case) for case in cases)
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<testsuite tests="{len(cases)}">{body}</testsuite>',
        encoding="utf-8",
    )


def _full_cases(*, extra_skip: bool = False, bad_skip_message: bool = False):
    cases = [
        (f"tests/test_legacy.py::test_pass_{index}", "passed", "")
        for index in range(156)
    ]
    cases.extend(
        (
            f"tests/test_optional.py::test_skip_{index}",
            "skipped",
            "optional dependency missing" if bad_skip_message and index == 0 else "pydrake unavailable",
        )
        for index in range(17)
    )
    cases.extend(
        (f"tests/test_v2_api.py::test_v2_{index}", "passed", "")
        for index in range(9)
    )
    if extra_skip:
        cases.append(("tests/test_optional.py::test_skip_extra", "skipped", "pydrake unavailable"))
    return cases


def _config(tmp_path: Path, *, boundaries=None) -> Path:
    payload = {
        "schema_version": "xunce-path-v2-gate1-contract/v1",
        "stage_id": "xunce-path-v2-gate1-contract",
        "python": "D:/conda_envs/lunar-explorer/python.exe",
        "expected_python_version": "3.12.13",
        "expected_git": {
            "branch": "codex/multiplatform-path-planner-v2",
            "base_commit": "b2a36d31f3802eb5a37fcfcf594f74a499aa719b",
            "nested_branch": "codex/multiplatform-path-planner-v2",
        },
        "temp_root": "D:/xunce/tmp/path_v2_g1",
        "focused": {
            "working_directory": "path-planner",
            "pythonpath": ["path-planner/src"],
            "pytest_targets": list(FOCUSED_TARGETS),
        },
        "full": {
            "working_directory": "path-planner",
            "pythonpath": ["path-planner/src"],
            "pytest_targets": ["tests"],
            "legacy_expected": {
                "passed": 156,
                "skipped": 17,
                "failures": 0,
                "errors": 0,
            },
            "allowed_skip_dependency": "pydrake",
        },
        "byte_repeat": {
            "repeat_count": 3,
            "python_hash_seeds": ["11", "29", "47"],
        },
        "boundaries": dict(BOUNDARIES if boundaries is None else boundaries),
        "pass_route": "implement_path_v2_wheel_provider",
    }
    path = tmp_path / "gate1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_full_junit_audit_separates_exact_legacy_counts_from_new_v2(tmp_path: Path) -> None:
    runner = _runner()
    junit = tmp_path / "full.xml"
    _write_junit(junit, _full_cases())

    audit = runner.audit_full_junit(
        junit,
        expected_legacy={"passed": 156, "skipped": 17, "failures": 0, "errors": 0},
        allowed_skip_dependency="pydrake",
    )

    assert audit["status"] == "passed"
    assert audit["legacy"] == {"passed": 156, "skipped": 17, "failures": 0, "errors": 0}
    assert audit["v2"]["passed"] == 9
    assert audit["total"]["passed"] == 165


@pytest.mark.parametrize(
    "cases",
    [
        _full_cases(extra_skip=True),
        _full_cases(bad_skip_message=True),
        _full_cases()[1:],
    ],
)
def test_full_junit_rejects_extra_non_pydrake_or_legacy_count_drift(
    tmp_path: Path,
    cases,
) -> None:
    runner = _runner()
    junit = tmp_path / "full.xml"
    _write_junit(junit, cases)

    audit = runner.audit_full_junit(
        junit,
        expected_legacy={"passed": 156, "skipped": 17, "failures": 0, "errors": 0},
        allowed_skip_dependency="pydrake",
    )

    assert audit["status"] == "failed"


def test_repeat_audit_requires_three_distinct_seeds_and_one_digest() -> None:
    runner = _runner()
    stable = [
        {"repeat": index, "python_hash_seed": seed, "digest": "a" * 64}
        for index, seed in enumerate(("11", "29", "47"), start=1)
    ]
    drifted = [*stable[:2], {**stable[2], "digest": "b" * 64}]

    assert runner.audit_repeat_digests(stable)["status"] == "passed"
    assert runner.audit_repeat_digests(drifted)["status"] == "failed"
    assert runner.audit_repeat_digests(stable[:2])["status"] == "failed"


@pytest.mark.parametrize("bad_value", [0, None, "false", True])
def test_boundaries_require_exact_boolean_false(bad_value) -> None:
    runner = _runner()
    boundaries = dict(BOUNDARIES)
    boundaries["publishes_checkpoint"] = bad_value

    assert runner.boundaries_match(BOUNDARIES) is True
    assert runner.boundaries_match(boundaries) is False


def test_output_root_inside_repo_is_rejected_before_creation() -> None:
    runner = _runner()
    output_root = REPO_ROOT / "never-create-gate1-output"

    with pytest.raises(ValueError, match="outside repo"):
        runner.validate_output_root(REPO_ROOT, output_root)
    assert not output_root.exists()


def test_stale_noncanonical_artifact_fails_closed_without_deletion(tmp_path: Path) -> None:
    runner = _runner()
    output_root = tmp_path / "out"
    output_root.mkdir()
    stale = output_root / "focused.junit.xml"
    stale.write_text("stale", encoding="utf-8")

    with pytest.raises(RuntimeError, match="stale"):
        runner.run_gate_benchmark(
            _config(tmp_path),
            output_root,
            REPO_ROOT,
            execute_tests=False,
        )
    assert stale.read_text(encoding="utf-8") == "stale"


def test_dry_run_writes_exact_eight_artifacts_and_manifest_hashes_seven(
    tmp_path: Path,
) -> None:
    runner = _runner()
    output_root = tmp_path / "out"

    summary = runner.run_gate_benchmark(
        _config(tmp_path),
        output_root,
        REPO_ROOT,
        execute_tests=False,
    )

    assert summary["status"] == "dry_run"
    assert {path.name for path in output_root.iterdir()} == CANONICAL_ARTIFACTS
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    entries = {entry["relative_path"]: entry for entry in manifest["artifacts"]}
    assert set(entries) == CANONICAL_ARTIFACTS - {"manifest.json"}
    for relative_path, entry in entries.items():
        content = (output_root / relative_path).read_bytes()
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()
        assert entry["size"] == len(content)


def test_registry_entry_has_exact_script_defaults_and_run_stage_placeholders() -> None:
    registry = json.loads(
        (REPO_ROOT / "configs" / "stage_registry.json").read_text(encoding="utf-8")
    )

    entry = registry["stages"]["xunce-path-v2-gate1-contract"]
    assert entry["script"] == "scripts/run_xunce_path_v2_gate_benchmark.py"
    assert entry["default_config"] == "configs/xunce_path_v2_gate1_contract_v1.json"
    assert entry["default_output_root"] == "D:/xunce/out/path_v2/g1"
    assert entry["args"] == [
        "--config",
        "{config}",
        "--output-root",
        "{output_root}",
        "--repo-root",
        "{repo_root}",
    ]


def test_dry_run_rejects_non_strict_boundary_config(tmp_path: Path) -> None:
    runner = _runner()
    boundaries = dict(BOUNDARIES)
    boundaries["starts_online_canary"] = 0

    summary = runner.run_gate_benchmark(
        _config(tmp_path, boundaries=boundaries),
        tmp_path / "out",
        REPO_ROOT,
        execute_tests=False,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "restore_gate1_safety_boundaries"
