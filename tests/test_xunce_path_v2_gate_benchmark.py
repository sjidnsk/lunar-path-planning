from __future__ import annotations

import hashlib
import importlib
import json
import sys
from copy import deepcopy
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
EXPECTED_BRANCH = "codex/multiplatform-path-planner-v2"
ORIGINAL_BASE_COMMIT = "b635740ee021258ef31811ec87c60add839fc5f9"
GATE_INPUT_COMMIT = "b2a36d31f3802eb5a37fcfcf594f74a499aa719b"
EXPECTED_PYTHON_VERSION = "3.12.13"
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
            "branch": EXPECTED_BRANCH,
            "base_commit": ORIGINAL_BASE_COMMIT,
            "gate_input_commit": GATE_INPUT_COMMIT,
            "nested_branch": EXPECTED_BRANCH,
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


FROZEN_CONFIG_TAMPERS = [
    pytest.param(("expected_git", "branch"), "codex/other", id="branch"),
    pytest.param(("expected_git", "nested_branch"), "codex/other", id="nested-branch"),
    pytest.param(("expected_git", "base_commit"), GATE_INPUT_COMMIT, id="original-base"),
    pytest.param(("expected_git", "gate_input_commit"), ORIGINAL_BASE_COMMIT, id="gate-input"),
    pytest.param(("python",), "D:/other/python.exe", id="python"),
    pytest.param(("expected_python_version",), "0.0.0", id="python-version"),
    pytest.param(("full", "legacy_expected", "passed"), 157, id="legacy-passed"),
    pytest.param(("full", "legacy_expected", "skipped"), 18, id="legacy-skipped"),
    pytest.param(("full", "legacy_expected", "failures"), 1, id="legacy-failures"),
    pytest.param(("full", "legacy_expected", "errors"), 1, id="legacy-errors"),
    pytest.param(("full", "allowed_skip_dependency"), "", id="empty-skip-dependency"),
    pytest.param(("full", "allowed_skip_dependency"), "numpy", id="other-skip-dependency"),
    pytest.param(("focused", "working_directory"), ".", id="focused-cwd"),
    pytest.param(("focused", "pythonpath"), ["src"], id="focused-pythonpath"),
    pytest.param(("full", "working_directory"), ".", id="full-cwd"),
    pytest.param(("full", "pythonpath"), ["src"], id="full-pythonpath"),
]


@pytest.mark.parametrize(("field_path", "tampered"), FROZEN_CONFIG_TAMPERS)
def test_frozen_config_rejects_each_tampered_field_before_side_effects(
    tmp_path: Path,
    monkeypatch,
    field_path,
    tampered,
) -> None:
    runner = _runner()
    config_path = _config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target = config
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = tampered
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_root = tmp_path / "out"

    def forbidden(*args, **kwargs):
        raise AssertionError("side effect occurred before frozen config rejection")

    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    monkeypatch.setattr(runner.gate_artifacts, "write_gate_artifacts", forbidden)

    with pytest.raises(ValueError, match="frozen"):
        runner.run_gate_benchmark(
            config_path,
            output_root,
            REPO_ROOT,
            execute_tests=False,
        )
    assert not output_root.exists()


def _runtime_audit() -> dict:
    path_planner_origin = str((REPO_ROOT / "path-planner" / "src" / "path_planner" / "__init__.py").resolve())
    ppo_origin = str((REPO_ROOT / "src" / "lunar_exploration_ppo" / "__init__.py").resolve())
    superproject_git = {
        "status": "passed",
        "head": "super-head",
        "branch": EXPECTED_BRANCH,
        "git_dir": "D:/git/worktrees/gate1",
        "git_common_dir": "D:/git",
        "clean_tree": True,
        "dirty_paths": [],
        "base_is_ancestor": True,
    }
    return {
        "status": "passed",
        "superproject_git": superproject_git,
        "gate_input_git": deepcopy(superproject_git),
        "nested_git": {
            "status": "passed",
            "head": "nested-head",
            "gitlink": "nested-head",
            "branch": EXPECTED_BRANCH,
            "clean_tree": True,
            "dirty_paths": [],
            "head_matches_gitlink": True,
        },
        "import_origins": {
            "status": "passed",
            "python": str(Path("D:/conda_envs/lunar-explorer/python.exe").resolve()),
            "python_version": EXPECTED_PYTHON_VERSION,
            "python_no_user_site": True,
            "pythonpath": f"{(REPO_ROOT / 'src').resolve()};{(REPO_ROOT / 'path-planner' / 'src').resolve()}",
            "path_planner": {"origin": path_planner_origin},
            "lunar_exploration_ppo": {"origin": ppo_origin},
            "path_planner_from_worktree": True,
            "ppo_from_worktree": True,
        },
        "python_version_matches": True,
        "original_base_is_ancestor": True,
        "gate_input_is_ancestor": True,
    }


def _postflight_with_mutation(mutation: str) -> dict:
    audit = _runtime_audit()
    if mutation == "dirty-superproject":
        audit["superproject_git"].update(status="failed", clean_tree=False, dirty_paths=[" M dirty.txt"])
        audit["status"] = "failed"
    elif mutation == "dirty-nested":
        audit["nested_git"].update(status="failed", clean_tree=False, dirty_paths=["?? dirty.txt"])
        audit["status"] = "failed"
    elif mutation == "super-head-drift":
        audit["superproject_git"]["head"] = "other-super-head"
        audit["gate_input_git"]["head"] = "other-super-head"
    elif mutation == "nested-gitlink-drift":
        audit["nested_git"].update(head="other-nested-head", gitlink="other-nested-head")
    elif mutation == "import-origin-drift":
        audit["import_origins"]["path_planner"]["origin"] = "D:/other/path_planner/__init__.py"
    else:
        raise AssertionError(f"unknown mutation {mutation}")
    return audit


def _install_green_formal_mocks(monkeypatch, runner, postflight: dict, events: list[str]) -> None:
    preflight = _runtime_audit()
    monkeypatch.setattr(runner, "_preflight", lambda config, repo_root: deepcopy(preflight))

    def fake_run_pytest(*, targets, **kwargs):
        suite = "focused" if tuple(targets) == tuple(FOCUSED_TARGETS) else "full"
        events.append(suite)
        return {
            "command": ["python", "-m", "pytest", *targets],
            "working_directory": "path-planner",
            "pythonpath": "path-planner/src",
            "returncode": 0,
            "stdout_tail": "green",
            "stderr_tail": "",
        }

    monkeypatch.setattr(runner, "_run_pytest", fake_run_pytest)
    monkeypatch.setattr(
        runner,
        "_audit_focused_junit",
        lambda path: {
            "status": "passed",
            "tests": 227,
            "passed": 227,
            "skipped": 0,
            "failures": 0,
            "errors": 0,
        },
    )
    monkeypatch.setattr(
        runner,
        "audit_full_junit",
        lambda *args, **kwargs: {
            "status": "passed",
            "total": {"passed": 383, "skipped": 17, "failures": 0, "errors": 0},
            "legacy": {"passed": 156, "skipped": 17, "failures": 0, "errors": 0},
            "v2": {"passed": 227, "skipped": 0, "failures": 0, "errors": 0},
            "legacy_skip_contract": True,
            "legacy_skip_messages": ["pydrake"] * 17,
        },
    )

    def fake_repeats(**kwargs):
        events.append("byte-repeat")
        return [
            {
                "suite": "byte-repeat",
                "check": "canonical_unknown_profile",
                "repeat": repeat,
                "python_hash_seed": seed,
                "returncode": 0,
                "digest": "a" * 64,
                "canonical_hex": "00",
                "root_has_plan_v2": False,
                "v1_route_schema": "path-planner-route/v1",
                "stderr_tail": "",
            }
            for repeat, seed in enumerate(("11", "29", "47"), start=1)
        ]

    monkeypatch.setattr(runner, "_run_byte_repeats", fake_repeats)

    def fake_postflight(config, repo_root):
        assert events == ["focused", "full", "byte-repeat"]
        events.append("postflight")
        return deepcopy(postflight)

    monkeypatch.setattr(runner, "_postflight", fake_postflight, raising=False)


@pytest.mark.parametrize(
    "mutation",
    [
        "dirty-superproject",
        "dirty-nested",
        "super-head-drift",
        "nested-gitlink-drift",
        "import-origin-drift",
    ],
)
def test_postflight_dirty_or_identity_drift_fails_runtime_isolation_with_evidence(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
) -> None:
    runner = _runner()
    events: list[str] = []
    _install_green_formal_mocks(
        monkeypatch,
        runner,
        _postflight_with_mutation(mutation),
        events,
    )
    output_root = tmp_path / "out"

    summary = runner.run_gate_benchmark(
        _config(tmp_path),
        output_root,
        REPO_ROOT,
        execute_tests=True,
    )

    assert events == ["focused", "full", "byte-repeat", "postflight"]
    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "restore_gate1_runtime_isolation"
    assert summary["checks"]["postflight"] is False
    assert "postflight" in summary
    rows = [json.loads(line) for line in (output_root / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(
        row.get("suite") == "boundary-review"
        and row.get("check") == "runtime_postflight"
        and row.get("status") == "failed"
        for row in rows
    )
    review = json.loads((output_root / "review.json").read_text(encoding="utf-8"))
    assert review["postflight"] == summary["postflight"]
    assert {path.name for path in output_root.iterdir()} == CANONICAL_ARTIFACTS


def test_green_formal_review_contains_complete_runtime_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    events: list[str] = []
    _install_green_formal_mocks(monkeypatch, runner, _runtime_audit(), events)
    output_root = tmp_path / "out"

    summary = runner.run_gate_benchmark(
        _config(tmp_path),
        output_root,
        REPO_ROOT,
        execute_tests=True,
    )

    assert summary["status"] == "passed"
    assert summary["checks"]["postflight"] is True
    review = json.loads((output_root / "review.json").read_text(encoding="utf-8"))
    assert {
        "preflight",
        "postflight",
        "execution",
        "focused_junit",
        "full_junit",
        "repeat_rows",
        "byte_repeat",
    } <= set(review)
    assert review["execution"]["attempt_root"].startswith("D:\\xunce\\tmp\\path_v2_g1")
    assert review["execution"]["isolation_env"]["PYTHONNOUSERSITE"] == "1"
    assert review["execution"]["focused_command_result"]["returncode"] == 0
    assert review["execution"]["full_command_result"]["returncode"] == 0
    assert len(review["repeat_rows"]) == 3


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
