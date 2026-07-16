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
GATE2_INPUT_COMMIT = "0cc3eb9728a77473dd436dc2b05e2df009a296c7"
GATE3_INPUT_COMMIT = "d6b6b93c7e2c148195cd907010f46bd88e97ce2b"
EXPECTED_PYTHON_VERSION = "3.12.13"
FOCUSED_TARGETS = [
    "tests/test_v2_contracts.py",
    "tests/test_v2_serialization.py",
    "tests/test_v2_terrain.py",
    "tests/test_v2_fine_safety_anchor.py",
    "tests/test_v2_profiles.py",
    "tests/test_v2_api.py",
]
GATE2_FOCUSED_TARGETS = [
    "tests/test_v2_benchmark.py",
    "tests/test_v2_wheel_provider.py",
    "tests/test_v2_route_validation.py",
    "tests/test_v2_geometry.py",
    "tests/test_v2_profiles.py",
    "tests/test_v2_wheel_contracts.py",
    "tests/test_v2_contracts.py",
    "tests/test_v2_api.py",
    "tests/test_v2_runtime.py",
    "tests/test_hybrid_astar.py",
    "tests/test_astar.py",
]
GATE3_FOCUSED_TARGETS = [
    "tests/test_v2_search.py",
    "tests/test_v2_hierarchy.py",
    "tests/test_v2_cache.py",
    "tests/test_v2_lazy_validation.py",
    "tests/test_v2_route_validation.py",
    "tests/test_v2_wheel_provider.py",
    "tests/test_v2_wheel_contracts.py",
    "tests/test_v2_api.py",
    "tests/test_v2_runtime.py",
    "tests/test_v2_serialization.py",
    "tests/test_v2_contracts.py",
    "tests/test_hybrid_astar.py",
    "tests/test_astar.py",
    "../tests/test_xunce_path_v2_gate_benchmark.py",
]
GATE3_CASES = [
    "fine_only",
    "multi_heuristic_only",
    "hierarchy_only",
    "lazy_validation_only",
    "lazy_validation_plus_cache",
    "full_v2",
]
GATE3_DISABLED_ACCELERATORS = [
    {"accelerator_id": "hierarchy", "reason": "not_integrated_into_provider"},
    {"accelerator_id": "lazy_validation", "reason": "not_integrated_into_provider"},
    {"accelerator_id": "multi_heuristic", "reason": "not_integrated_into_provider"},
    {"accelerator_id": "validation_cache", "reason": "not_integrated_into_provider"},
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


def _gate2_config(tmp_path: Path, *, boundaries=None) -> Path:
    payload = {
        "schema_version": "xunce-path-v2-gate2-wheel/v1",
        "stage_id": "xunce-path-v2-gate2-wheel",
        "python": "D:/conda_envs/lunar-explorer/python.exe",
        "expected_python_version": "3.12.13",
        "expected_git": {
            "branch": EXPECTED_BRANCH,
            "base_commit": ORIGINAL_BASE_COMMIT,
            "gate_input_commit": GATE2_INPUT_COMMIT,
            "nested_branch": EXPECTED_BRANCH,
        },
        "temp_root": "D:/xunce/tmp/path_v2_g2",
        "focused": {
            "working_directory": "path-planner",
            "pythonpath": ["path-planner/src"],
            "pytest_targets": list(GATE2_FOCUSED_TARGETS),
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
        "inputs": {
            "primitive_audit": "D:/xunce/inputs/path_v2/g2/independent_wheel_oracle_labels.jsonl",
            "exact_map_quality": "D:/xunce/inputs/path_v2/g2/independent_wheel_exact_map_optima.jsonl",
            "standard_episodes": "D:/xunce/inputs/path_v2/g2/standard_wheel_schedule.jsonl",
        },
        "baseline_evidence": {
            "schema_version": "xunce-path-v2-gate0-path-planner-junit/v1",
            "path": "D:/xunce/out/path_v2/g0/path_planner_baseline.junit.xml",
            "sha256": "90a02eb6af78805bfdf2fcce4828283cb3f4d77f30aae55519f533896a06e676",
            "test_count": 173,
        },
        "thresholds": {
            "min_primitive_independent_samples": 10000,
            "max_primitive_false_positives": 0,
            "min_primitive_recall": 0.98,
            "min_primitive_complete_l2_ratio": 1.0,
            "min_exact_map_independent_cases": 1,
            "min_exact_map_success_ratio": 1.0,
            "min_exact_map_resource_cost_ratio": 1.0,
            "max_exact_map_resource_cost_ratio": 1.10,
            "min_exact_map_complete_l2_ratio": 1.0,
            "min_standard_independent_episodes": 100,
            "min_standard_reachable_success_ratio": 0.99,
            "min_standard_complete_l2_ratio": 1.0,
            "max_standard_p95_runtime_ms": 250.0,
            "hard_timeout_ms": 2000.0,
            "max_hard_timeout_violations": 0,
        },
        "boundaries": dict(BOUNDARIES if boundaries is None else boundaries),
        "pass_route": "implement_path_v2_lazy_validation_and_cache",
    }
    path = tmp_path / "gate2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _gate3_config(tmp_path: Path, *, boundaries=None) -> Path:
    payload = {
        "schema_version": "xunce-path-v2-gate3-accelerators/v1",
        "stage_id": "xunce-path-v2-gate3-accelerators",
        "python": "D:/conda_envs/lunar-explorer/python.exe",
        "expected_python_version": "3.12.13",
        "expected_git": {
            "branch": EXPECTED_BRANCH,
            "base_commit": ORIGINAL_BASE_COMMIT,
            "gate_input_commit": GATE3_INPUT_COMMIT,
            "nested_branch": EXPECTED_BRANCH,
        },
        "formal_output_root": "D:/xunce/out/path_v2/g3",
        "temp_root": "D:/xunce/tmp/path_v2_g3",
        "focused": {
            "working_directory": "path-planner",
            "pythonpath": ["path-planner/src"],
            "pytest_targets": list(GATE3_FOCUSED_TARGETS),
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
        "baseline_evidence": {
            "schema_version": "xunce-path-v2-gate0-path-planner-junit/v1",
            "path": "D:/xunce/out/path_v2/g0/path_planner_baseline.junit.xml",
            "sha256": "90a02eb6af78805bfdf2fcce4828283cb3f4d77f30aae55519f533896a06e676",
            "test_count": 173,
        },
        "ablation": {
            "cases": list(GATE3_CASES),
            "worker_counts": [1, 4],
            "python_hash_seeds": [11, 29, 47],
            "repeat_count": 3,
        },
        "expected_disabled_accelerators": deepcopy(GATE3_DISABLED_ACCELERATORS),
        "boundaries": dict(BOUNDARIES if boundaries is None else boundaries),
        "pass_route": "implement_path_v2_legged_static_stability_oracle",
    }
    path = tmp_path / "gate3.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _gate3_probe_rows(*, digest: str = "a" * 64) -> list[dict]:
    rows = []
    for case_id in GATE3_CASES:
        for worker_count in (1, 4):
            for hash_seed in (11, 29, 47):
                for repeat in (1, 2, 3):
                    rows.append(
                        {
                            "case_id": case_id,
                            "worker_count": worker_count,
                            "python_hash_seed": hash_seed,
                            "repeat": repeat,
                            "status": "passed",
                            "decision_digest": digest,
                            "fine_only_digest": digest,
                            "safety_equivalent": True,
                            "authoritative_order_preserved": True,
                            "suggestion_non_authoritative": True,
                            "hierarchy_conservative": True,
                            "cache_l2_equivalent": True,
                            "l2_authority_preserved": True,
                            "fallback_isolated": True,
                            "fatal_reason": None,
                            "accelerator_used": False,
                            "runtime_disabled_accelerators": [],
                        }
                    )
    return rows


def _install_gate3_green_mocks(
    monkeypatch,
    runner,
    events: list[str],
    *,
    probe_rows: list[dict] | None = None,
) -> None:
    runtime = _runtime_audit()
    monkeypatch.setattr(
        runner,
        "_gate3_preflight",
        lambda config, repo_root: deepcopy(runtime),
    )

    def fake_run_pytest(*, targets, **kwargs):
        suite = "focused" if tuple(targets) == tuple(GATE3_FOCUSED_TARGETS) else "full"
        events.append(suite)
        return {
            "command": ["python", "-m", "pytest", *targets],
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
            "tests": 500,
            "passed": 500,
            "skipped": 0,
            "failures": 0,
            "errors": 0,
        },
    )
    monkeypatch.setattr(
        runner,
        "_audit_gate3_full_junit",
        lambda *args, **kwargs: {
            "schema_version": "xunce-path-v2-gate3-full-junit-audit/v1",
            "status": "passed",
            "total": {"passed": 836, "skipped": 17, "failures": 0, "errors": 0},
            "baseline": {"passed": 156, "skipped": 17, "failures": 0, "errors": 0},
            "v2": {"passed": 680, "skipped": 0, "failures": 0, "errors": 0},
            "baseline_not_reduced": True,
            "skip_contract": True,
            "v2_green": True,
        },
    )
    rows = deepcopy(_gate3_probe_rows() if probe_rows is None else probe_rows)
    audit = runner._audit_gate3_probe_rows(rows)

    def fake_probes(**kwargs):
        events.append("probes")
        return {
            **audit,
            "rows": deepcopy(rows),
            "commands": [
                {
                    "command": ["python", "-c", "<gate3-probe>"],
                    "environment": {
                        "PYTHONHASHSEED": "11",
                        "PATH_V2_GATE3_WORKER_COUNT": "1",
                        "PATH_V2_GATE3_REPEAT": "1",
                    },
                    "worker_count": 1,
                    "python_hash_seed": 11,
                    "repeat": 1,
                    "returncode": 0,
                    "stable_failure_reason": None,
                }
            ],
        }

    monkeypatch.setattr(runner, "_run_gate3_probes", fake_probes)

    def fake_postflight(config, repo_root):
        events.append("postflight")
        return deepcopy(runtime)

    monkeypatch.setattr(runner, "_gate3_postflight", fake_postflight)


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


def _gate2_loaded_datasets() -> dict[str, dict]:
    return {
        "primitive_audit": {
            "status": "loaded",
            "rows": [
                {
                    "schema_version": "path-planner-v2-primitive-audit-row/v1",
                    "row_id": "primitive-b",
                    "seed": 2,
                },
                {
                    "schema_version": "path-planner-v2-primitive-audit-row/v1",
                    "row_id": "primitive-a",
                    "seed": 1,
                },
            ],
            "summary": {
                "total_row_count": 10000,
                "formal_row_count": 10000,
                "excluded_row_count": 0,
                "false_positive_count": 0,
                "primitive_recall": 0.99,
                "provider_success_count": 9900,
                "complete_l2_ratio": 1.0,
                "runtime_p50_ms": 10.0,
                "runtime_p95_ms": 20.0,
                "runtime_p99_ms": 30.0,
                "timeout_count": 0,
                "hard_timeout_violation_count": 0,
                "reason_histogram": {},
            },
        },
        "exact_map_quality": {
            "status": "loaded",
            "rows": [
                {
                    "schema_version": "path-planner-v2-exact-map-quality-row/v1",
                    "row_id": "exact-a",
                    "seed": 3,
                }
            ],
            "summary": {
                "total_row_count": 1,
                "formal_row_count": 1,
                "excluded_row_count": 0,
                "provider_success_count": 1,
                "provider_success_ratio": 1.0,
                "complete_l2_ratio": 1.0,
                "resource_cost_ratios": [{"row_id": "exact-a", "ratio": 1.05}],
                "max_resource_cost_ratio": 1.05,
                "runtime_p50_ms": 12.0,
                "runtime_p95_ms": 12.0,
                "runtime_p99_ms": 12.0,
                "timeout_count": 0,
                "hard_timeout_violation_count": 0,
                "reason_histogram": {},
            },
        },
        "standard_episodes": {
            "status": "loaded",
            "rows": [
                {
                    "schema_version": "path-planner-v2-standard-episode-row/v1",
                    "episode_id": "standard-a",
                    "seed": 4,
                }
            ],
            "summary": {
                "total_episode_count": 100,
                "formal_episode_count": 100,
                "excluded_episode_count": 0,
                "formal_oracle_reachable_count": 100,
                "reachable_provider_success_count": 99,
                "reachable_query_success_ratio": 0.99,
                "complete_l2_ratio": 1.0,
                "runtime_p50_ms": 100.0,
                "runtime_p95_ms": 240.0,
                "runtime_p99_ms": 300.0,
                "timeout_count": 0,
                "hard_timeout_violation_count": 0,
                "reason_histogram": {},
            },
        },
    }


def _install_gate2_green_mocks(
    monkeypatch,
    runner,
    datasets: dict[str, dict],
    events: list[str],
) -> None:
    preflight = _runtime_audit()
    monkeypatch.setattr(
        runner,
        "_gate2_preflight",
        lambda config, repo_root: deepcopy(preflight),
    )

    def fake_run_pytest(*, targets, **kwargs):
        suite = "focused" if tuple(targets) == tuple(GATE2_FOCUSED_TARGETS) else "full"
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
            "tests": 316,
            "passed": 316,
            "skipped": 0,
            "failures": 0,
            "errors": 0,
        },
    )
    monkeypatch.setattr(
        runner,
        "_audit_gate2_full_junit",
        lambda *args, **kwargs: {
            "status": "passed",
            "total": {"passed": 672, "skipped": 17, "failures": 0, "errors": 0},
            "baseline": {"passed": 156, "skipped": 17, "failures": 0, "errors": 0},
            "baseline_not_reduced": True,
            "skip_contract": True,
            "skip_messages": ["pydrake"] * 17,
        },
    )

    def fake_load(*, dataset_name, **kwargs):
        events.append(dataset_name)
        return deepcopy(datasets[dataset_name])

    monkeypatch.setattr(runner, "_load_gate2_dataset", fake_load)

    def fake_postflight(config, repo_root):
        events.append("postflight")
        return deepcopy(preflight)

    monkeypatch.setattr(runner, "_gate2_postflight", fake_postflight)


@pytest.mark.parametrize(
    ("field_path", "tampered"),
    [
        pytest.param(("expected_git", "gate_input_commit"), GATE_INPUT_COMMIT, id="gate-input"),
        pytest.param(("inputs", "primitive_audit"), "D:/other.jsonl", id="oracle-input"),
        pytest.param(("baseline_evidence", "sha256"), "0" * 64, id="baseline-hash"),
        pytest.param(("thresholds", "min_primitive_independent_samples"), 9999, id="primitive-min"),
        pytest.param(("thresholds", "min_exact_map_resource_cost_ratio"), 0.99, id="exact-min-ratio"),
        pytest.param(("thresholds", "max_exact_map_resource_cost_ratio"), 1.11, id="exact-ratio"),
        pytest.param(("thresholds", "max_standard_p95_runtime_ms"), 251.0, id="standard-p95"),
        pytest.param(("pass_route",), "skip-gate3", id="pass-route"),
    ],
)
def test_gate2_frozen_config_rejects_tampering_before_side_effects(
    tmp_path: Path,
    monkeypatch,
    field_path,
    tampered,
) -> None:
    runner = _runner()
    config_path = _gate2_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target = config
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = tampered
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("side effect occurred before frozen Gate 2 config rejection")

    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    monkeypatch.setattr(runner.gate_artifacts, "write_gate_artifacts", forbidden)

    with pytest.raises(ValueError, match="frozen"):
        runner.run_gate_benchmark(
            config_path,
            tmp_path / "out",
            REPO_ROOT,
            execute_tests=False,
        )
    assert not (tmp_path / "out").exists()


def test_gate2_dry_run_writes_exact_artifacts_without_loading_formal_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run loaded formal evidence")

    monkeypatch.setattr(runner, "_load_gate2_dataset", forbidden, raising=False)
    output_root = tmp_path / "out"
    summary = runner.run_gate_benchmark(
        _gate2_config(tmp_path),
        output_root,
        REPO_ROOT,
        execute_tests=False,
    )

    assert summary["status"] == "dry_run"
    assert summary["next_required_change"] == "execute_gate2_wheel_evidence"
    assert {path.name for path in output_root.iterdir()} == CANONICAL_ARTIFACTS


def test_gate2_missing_inputs_block_in_exact_priority_after_code_checks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    datasets = {
        name: {"status": "missing", "rows": [], "summary": None}
        for name in _gate2_loaded_datasets()
    }
    events: list[str] = []
    _install_gate2_green_mocks(monkeypatch, runner, datasets, events)
    output_root = tmp_path / "out"

    summary = runner.run_gate_benchmark(
        _gate2_config(tmp_path),
        output_root,
        REPO_ROOT,
        execute_tests=True,
    )

    blockers = [
        "provide_independent_wheel_oracle_labels",
        "provide_independent_wheel_exact_map_optima",
        "provide_standard_wheel_schedule",
    ]
    assert events == [
        "focused",
        "full",
        "primitive_audit",
        "exact_map_quality",
        "standard_episodes",
        "postflight",
    ]
    assert summary["status"] == "blocked"
    assert summary["blocking_reasons"] == blockers
    assert summary["next_required_change"] == blockers[0]
    routing = json.loads((output_root / "routing.json").read_text(encoding="utf-8"))
    assert routing["route"] == blockers[0]
    assert routing["blocking_reasons"] == blockers
    assert {path.name for path in output_root.iterdir()} == CANONICAL_ARTIFACTS


@pytest.mark.parametrize(
    ("missing_dataset", "expected_route"),
    [
        ("exact_map_quality", "provide_independent_wheel_exact_map_optima"),
        ("standard_episodes", "provide_standard_wheel_schedule"),
    ],
)
def test_gate2_later_blocker_appears_only_after_prior_independent_evidence(
    tmp_path: Path,
    monkeypatch,
    missing_dataset: str,
    expected_route: str,
) -> None:
    runner = _runner()
    datasets = _gate2_loaded_datasets()
    datasets[missing_dataset] = {"status": "missing", "rows": [], "summary": None}
    events: list[str] = []
    _install_gate2_green_mocks(monkeypatch, runner, datasets, events)

    summary = runner.run_gate_benchmark(
        _gate2_config(tmp_path),
        tmp_path / "out",
        REPO_ROOT,
        execute_tests=True,
    )

    assert summary["status"] == "blocked"
    assert summary["next_required_change"] == expected_route
    assert expected_route in summary["blocking_reasons"]


@pytest.mark.parametrize("dataset_status", ["invalid", "non_independent"])
def test_gate2_invalid_or_self_labelled_oracle_rows_never_fake_formal_pass(
    tmp_path: Path,
    monkeypatch,
    dataset_status: str,
) -> None:
    runner = _runner()
    datasets = _gate2_loaded_datasets()
    datasets["primitive_audit"] = {
        "status": dataset_status,
        "rows": [],
        "summary": (
            None
            if dataset_status == "invalid"
            else {
                "total_row_count": 10000,
                "formal_row_count": 0,
                "excluded_row_count": 10000,
            }
        ),
    }
    events: list[str] = []
    _install_gate2_green_mocks(monkeypatch, runner, datasets, events)

    summary = runner.run_gate_benchmark(
        _gate2_config(tmp_path),
        tmp_path / "out",
        REPO_ROOT,
        execute_tests=True,
    )

    assert summary["status"] == "blocked"
    assert summary["next_required_change"] == "provide_independent_wheel_oracle_labels"
    assert summary["checks"]["primitive_formal_denominator"] is False


def test_gate2_sufficient_independent_evidence_with_metric_failure_is_failed_not_blocked(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    datasets = _gate2_loaded_datasets()
    datasets["primitive_audit"]["summary"]["false_positive_count"] = 1
    events: list[str] = []
    _install_gate2_green_mocks(monkeypatch, runner, datasets, events)

    summary = runner.run_gate_benchmark(
        _gate2_config(tmp_path),
        tmp_path / "out",
        REPO_ROOT,
        execute_tests=True,
    )

    assert summary["status"] == "failed"
    assert summary["blocking_reasons"] == []
    assert summary["next_required_change"] == "repair_wheel_oracle_false_positives"
    assert summary["checks"]["primitive_false_positives"] is False


@pytest.mark.parametrize(
    "ratio",
    [
        pytest.param(0.99, id="below-optimum"),
        pytest.param(1.100001, id="above-limit"),
        pytest.param(10**400, id="unrepresentable-huge-ratio"),
    ],
)
def test_gate2_exact_cost_ratio_outside_frozen_interval_fails(
    tmp_path: Path,
    monkeypatch,
    ratio,
) -> None:
    runner = _runner()
    datasets = _gate2_loaded_datasets()
    datasets["exact_map_quality"]["summary"]["resource_cost_ratios"] = [
        {"row_id": "exact-a", "ratio": ratio}
    ]
    datasets["exact_map_quality"]["summary"]["max_resource_cost_ratio"] = ratio
    events: list[str] = []
    _install_gate2_green_mocks(monkeypatch, runner, datasets, events)

    summary = runner.run_gate_benchmark(
        _gate2_config(tmp_path),
        tmp_path / "out",
        REPO_ROOT,
        execute_tests=True,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_wheel_exact_map_resource_cost"
    assert summary["checks"]["exact_map_resource_cost"] is False


@pytest.mark.parametrize("ratio", [1.0, 1.10])
def test_gate2_exact_cost_ratio_closed_interval_boundaries_pass(
    tmp_path: Path,
    monkeypatch,
    ratio: float,
) -> None:
    runner = _runner()
    datasets = _gate2_loaded_datasets()
    datasets["exact_map_quality"]["summary"]["resource_cost_ratios"] = [
        {"row_id": "exact-a", "ratio": ratio}
    ]
    datasets["exact_map_quality"]["summary"]["max_resource_cost_ratio"] = ratio
    events: list[str] = []
    _install_gate2_green_mocks(monkeypatch, runner, datasets, events)

    summary = runner.run_gate_benchmark(
        _gate2_config(tmp_path),
        tmp_path / "out",
        REPO_ROOT,
        execute_tests=True,
    )

    assert summary["status"] == "passed"
    assert summary["checks"]["exact_map_resource_cost"] is True


def test_gate2_all_sufficient_metrics_pass_with_stable_result_order_and_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    datasets = _gate2_loaded_datasets()
    events: list[str] = []
    _install_gate2_green_mocks(monkeypatch, runner, datasets, events)
    first_root = tmp_path / "first"

    first = runner.run_gate_benchmark(
        _gate2_config(tmp_path),
        first_root,
        REPO_ROOT,
        execute_tests=True,
    )

    assert first["status"] == "passed"
    assert first["next_required_change"] == "implement_path_v2_lazy_validation_and_cache"
    result_lines = (first_root / "results.jsonl").read_text(encoding="utf-8").splitlines()
    result_rows = [json.loads(line) for line in result_lines]
    assert result_rows == sorted(
        result_rows,
        key=lambda row: (
            str(row.get("suite", "")),
            int(row.get("seed", -1)),
            str(row.get("row_id", row.get("episode_id", row.get("check", "")))),
        ),
    )
    manifest = json.loads((first_root / "manifest.json").read_text(encoding="utf-8"))
    result_entry = next(
        entry for entry in manifest["artifacts"] if entry["relative_path"] == "results.jsonl"
    )
    assert result_entry["sha256"] == hashlib.sha256(
        (first_root / "results.jsonl").read_bytes()
    ).hexdigest()


def test_gate2_blocked_is_successful_cli_outcome_but_failed_is_not() -> None:
    runner = _runner()

    assert runner.status_exit_code("blocked") == 0
    assert runner.status_exit_code("passed") == 0
    assert runner.status_exit_code("dry_run") == 0
    assert runner.status_exit_code("failed") == 1


def test_gate2_registry_entry_has_frozen_defaults() -> None:
    registry = json.loads(
        (REPO_ROOT / "configs" / "stage_registry.json").read_text(encoding="utf-8")
    )

    entry = registry["stages"]["xunce-path-v2-gate2-wheel"]
    assert entry == {
        "script": "scripts/run_xunce_path_v2_gate_benchmark.py",
        "default_config": "configs/xunce_path_v2_gate2_wheel_v1.json",
        "default_output_root": "D:/xunce/out/path_v2/g2",
        "args": [
            "--config",
            "{config}",
            "--output-root",
            "{output_root}",
            "--repo-root",
            "{repo_root}",
        ],
    }


@pytest.mark.parametrize(
    ("dataset_name", "payloads", "formal_field", "expected_formal"),
    [
        (
            "primitive_audit",
            [
                {
                    "row_id": "primitive-b",
                    "seed": 2,
                    "expected_safe": False,
                    "expected_label_source": "provider-self-label/v1",
                    "expected_label_independent": False,
                    "provider_safe": False,
                    "provider_complete_l2": False,
                    "runtime_ms": 20.0,
                    "timed_out": False,
                    "reason_code": "provider_rejected",
                },
                {
                    "row_id": "primitive-a",
                    "seed": 1,
                    "expected_safe": True,
                    "expected_label_source": "independent-wheel-oracle/v1",
                    "expected_label_independent": True,
                    "provider_safe": True,
                    "provider_complete_l2": True,
                    "runtime_ms": 10.0,
                    "timed_out": False,
                    "reason_code": "safe",
                },
            ],
            "formal_row_count",
            1,
        ),
        (
            "exact_map_quality",
            [
                {
                    "row_id": "exact-a",
                    "seed": 3,
                    "optimum_resource_cost": 10.0,
                    "optimum_source": "independent-exact-solver/v1",
                    "optimum_independent": True,
                    "provider_success": True,
                    "provider_resource_cost": 10.5,
                    "provider_complete_l2": True,
                    "runtime_ms": 12.0,
                    "timed_out": False,
                    "reason_code": "goal_reached",
                }
            ],
            "formal_row_count",
            1,
        ),
        (
            "standard_episodes",
            [
                {
                    "episode_id": "standard-a",
                    "seed": 4,
                    "schedule_source": "independent-standard-wheel/v1",
                    "schedule_independent": True,
                    "oracle_reachable": True,
                    "provider_success": True,
                    "provider_complete_l2": True,
                    "runtime_ms": 25.0,
                    "timed_out": False,
                    "reason_code": "goal_reached",
                }
            ],
            "formal_episode_count",
            1,
        ),
    ],
)
def test_gate2_loader_uses_public_typed_api_and_preserves_independent_denominator(
    tmp_path: Path,
    dataset_name: str,
    payloads: list[dict],
    formal_field: str,
    expected_formal: int,
) -> None:
    runner = _runner()
    input_path = tmp_path / f"{dataset_name}.jsonl"
    input_path.write_text(
        "".join(json.dumps(payload, sort_keys=True) + "\n" for payload in payloads),
        encoding="utf-8",
    )

    loaded = runner._load_gate2_dataset(
        dataset_name=dataset_name,
        path=input_path,
        repo_root=REPO_ROOT,
    )

    assert loaded["status"] == "loaded"
    assert loaded["summary"][formal_field] == expected_formal
    assert loaded["input_sha256"] == hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert loaded["rows_sha256"] is not None
    assert [row["seed"] for row in loaded["rows"]] == sorted(
        row["seed"] for row in loaded["rows"]
    )


def test_gate2_loader_rejects_non_object_jsonl_row_instead_of_silently_dropping_it(
    tmp_path: Path,
) -> None:
    runner = _runner()
    input_path = tmp_path / "invalid.jsonl"
    input_path.write_text("[]\n", encoding="utf-8")

    loaded = runner._load_gate2_dataset(
        dataset_name="primitive_audit",
        path=input_path,
        repo_root=REPO_ROOT,
    )

    assert loaded["status"] == "invalid"
    assert loaded["error_type"] == "ValueError"
    assert loaded["rows"] == []
    assert loaded["input_sha256"] == hashlib.sha256(input_path.read_bytes()).hexdigest()


def test_gate2_loader_rejects_internal_blank_jsonl_row_instead_of_silently_dropping_it(
    tmp_path: Path,
) -> None:
    runner = _runner()
    valid = {
        "row_id": "primitive-a",
        "seed": 1,
        "expected_safe": True,
        "expected_label_source": "independent-wheel-oracle/v1",
        "expected_label_independent": True,
        "provider_safe": True,
        "provider_complete_l2": True,
        "runtime_ms": 10.0,
        "timed_out": False,
        "reason_code": "safe",
    }
    input_path = tmp_path / "blank-row.jsonl"
    input_path.write_text(
        json.dumps(valid) + "\n\n" + json.dumps({**valid, "row_id": "primitive-b"}) + "\n",
        encoding="utf-8",
    )

    loaded = runner._load_gate2_dataset(
        dataset_name="primitive_audit",
        path=input_path,
        repo_root=REPO_ROOT,
    )

    assert loaded["status"] == "invalid"
    assert loaded["error_type"] == "ValueError"
    assert loaded["rows"] == []
    assert loaded["input_sha256"] == hashlib.sha256(input_path.read_bytes()).hexdigest()


def test_gate2_loader_rejects_duplicate_json_key_that_could_override_independence(
    tmp_path: Path,
) -> None:
    runner = _runner()
    input_path = tmp_path / "duplicate-independence.jsonl"
    input_path.write_text(
        "{"
        '"row_id":"primitive-a",'
        '"seed":1,'
        '"expected_safe":true,'
        '"expected_label_source":"provider-self-label/v1",'
        '"expected_label_independent":false,'
        '"expected_label_independent":true,'
        '"provider_safe":true,'
        '"provider_complete_l2":true,'
        '"runtime_ms":10.0,'
        '"timed_out":false,'
        '"reason_code":"safe"'
        "}\n",
        encoding="utf-8",
    )

    loaded = runner._load_gate2_dataset(
        dataset_name="primitive_audit",
        path=input_path,
        repo_root=REPO_ROOT,
    )

    assert loaded["status"] == "invalid"
    assert loaded["error_type"] == "ValueError"
    assert loaded["rows"] == []
    assert loaded["input_sha256"] == hashlib.sha256(input_path.read_bytes()).hexdigest()


def test_gate2_loader_preserves_invalid_utf8_snapshot_hash(tmp_path: Path) -> None:
    runner = _runner()
    input_path = tmp_path / "invalid-utf8.jsonl"
    input_path.write_bytes(b"\xff\n")

    loaded = runner._load_gate2_dataset(
        dataset_name="primitive_audit",
        path=input_path,
        repo_root=REPO_ROOT,
    )

    assert loaded["status"] == "invalid"
    assert loaded["error_type"] == "UnicodeDecodeError"
    assert loaded["input_sha256"] == hashlib.sha256(input_path.read_bytes()).hexdigest()


def test_gate2_loader_classifies_deep_malformed_json_as_invalid_not_internal_error(
    tmp_path: Path,
) -> None:
    runner = _runner()
    input_path = tmp_path / "deep-malformed.jsonl"
    input_path.write_text("[" * 5000 + "0" + "]" * 5000 + "\n", encoding="utf-8")

    loaded = runner._load_gate2_dataset(
        dataset_name="primitive_audit",
        path=input_path,
        repo_root=REPO_ROOT,
    )

    assert loaded["status"] == "invalid"
    assert loaded["error_type"] in {"RecursionError", "ValueError"}
    assert loaded["input_sha256"] == hashlib.sha256(input_path.read_bytes()).hexdigest()


def test_gate2_loader_hashes_the_same_single_byte_snapshot_that_it_parses(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    payload = {
        "row_id": "primitive-a",
        "seed": 1,
        "expected_safe": True,
        "expected_label_source": "independent-wheel-oracle/v1",
        "expected_label_independent": True,
        "provider_safe": True,
        "provider_complete_l2": True,
        "runtime_ms": 10.0,
        "timed_out": False,
        "reason_code": "safe",
    }
    input_path = tmp_path / "single-snapshot.jsonl"
    content = (json.dumps(payload) + "\n").encode("utf-8")
    input_path.write_bytes(content)
    original_read_bytes = runner.artifact_io.read_bytes
    reads: list[Path] = []

    def counted_read_bytes(path):
        reads.append(Path(path))
        return original_read_bytes(path)

    monkeypatch.setattr(runner.artifact_io, "read_bytes", counted_read_bytes)
    loaded = runner._load_gate2_dataset(
        dataset_name="primitive_audit",
        path=input_path,
        repo_root=REPO_ROOT,
    )

    assert loaded["status"] == "loaded"
    assert reads == [input_path]
    assert loaded["input_sha256"] == hashlib.sha256(content).hexdigest()


def test_gate2_loader_classifies_huge_numeric_row_as_invalid_instead_of_crashing(
    tmp_path: Path,
) -> None:
    runner = _runner()
    payload = {
        "row_id": "primitive-huge-runtime",
        "seed": 1,
        "expected_safe": True,
        "expected_label_source": "independent-wheel-oracle/v1",
        "expected_label_independent": True,
        "provider_safe": True,
        "provider_complete_l2": True,
        "runtime_ms": 10**400,
        "timed_out": False,
        "reason_code": "safe",
    }
    input_path = tmp_path / "huge-number.jsonl"
    input_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    loaded = runner._load_gate2_dataset(
        dataset_name="primitive_audit",
        path=input_path,
        repo_root=REPO_ROOT,
    )

    assert loaded["status"] == "invalid"
    assert loaded["error_type"] in {"OverflowError", "ValueError"}
    assert loaded["rows"] == []


def test_gate2_internal_loader_error_is_code_failure_not_formal_blocker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    datasets = _gate2_loaded_datasets()
    datasets["primitive_audit"] = {
        "status": "internal_error",
        "rows": [],
        "summary": None,
        "error_type": "ImportError",
    }
    events: list[str] = []
    _install_gate2_green_mocks(monkeypatch, runner, datasets, events)

    summary = runner.run_gate_benchmark(
        _gate2_config(tmp_path),
        tmp_path / "out",
        REPO_ROOT,
        execute_tests=True,
    )

    assert summary["status"] == "failed"
    assert summary["blocking_reasons"] == []
    assert summary["next_required_change"] == "repair_gate2_benchmark_loader"
    assert summary["checks"]["benchmark_loader"] is False
    phases = [
        json.loads(line)
        for line in (tmp_path / "out" / "phase-state.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert next(row for row in phases if row["phase"] == "primitive-audit")["status"] == "failed"
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    primitive = next(row for row in rows if row["suite"] == "primitive-audit")
    assert primitive["reason"] == "repair_gate2_benchmark_loader"


def test_gate2_code_failure_leaves_formal_phases_not_run_not_blocked(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    events: list[str] = []
    _install_gate2_green_mocks(
        monkeypatch,
        runner,
        _gate2_loaded_datasets(),
        events,
    )
    monkeypatch.setattr(
        runner,
        "_audit_focused_junit",
        lambda path: {
            "status": "failed",
            "tests": 1,
            "passed": 0,
            "skipped": 0,
            "failures": 1,
            "errors": 0,
        },
    )

    summary = runner.run_gate_benchmark(
        _gate2_config(tmp_path),
        tmp_path / "out",
        REPO_ROOT,
        execute_tests=True,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "restore_gate2_focused_contracts"
    assert events == ["focused", "full", "postflight"]
    phases = [
        json.loads(line)
        for line in (tmp_path / "out" / "phase-state.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    formal = {
        row["phase"]: row["status"]
        for row in phases
        if row["phase"] in {"primitive-audit", "exact-map-quality", "standard-episodes"}
    }
    assert formal == {
        "primitive-audit": "not_run",
        "exact-map-quality": "not_run",
        "standard-episodes": "not_run",
    }


def test_gate2_full_audit_allows_new_tests_but_not_baseline_loss_or_new_skips(
    tmp_path: Path,
) -> None:
    runner = _runner()
    baseline = tmp_path / "baseline.xml"
    green = tmp_path / "green.xml"
    too_few = tmp_path / "too-few.xml"
    replacement = tmp_path / "replacement.xml"
    extra_skip = tmp_path / "extra-skip.xml"
    baseline_cases = [
        case for case in _full_cases() if not case[0].startswith("tests/test_v2_")
    ]
    green_cases = baseline_cases + [
        (f"tests/test_v2_new.py::test_added_{index}", "passed", "")
        for index in range(10)
    ]
    _write_junit(baseline, baseline_cases)
    _write_junit(green, green_cases)
    _write_junit(too_few, green_cases[1:])
    _write_junit(
        replacement,
        baseline_cases[1:]
        + [("tests/test_unrelated.py::test_replacement", "passed", "")],
    )
    _write_junit(
        extra_skip,
        green_cases + [("tests/test_v2_new.py::test_skip", "skipped", "pydrake unavailable")],
    )

    kwargs = {
        "expected_legacy": {"passed": 156, "skipped": 17, "failures": 0, "errors": 0},
        "allowed_skip_dependency": "pydrake",
        "baseline_evidence": {
            "schema_version": "test-baseline/v1",
            "path": baseline.as_posix(),
            "sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
            "test_count": len(baseline_cases),
        },
    }
    assert runner._audit_gate2_full_junit(green, **kwargs)["status"] == "passed"
    assert runner._audit_gate2_full_junit(too_few, **kwargs)["status"] == "failed"
    replacement_audit = runner._audit_gate2_full_junit(replacement, **kwargs)
    assert replacement_audit["status"] == "failed"
    assert replacement_audit["missing_baseline_nodeids"] == [
        "tests/test_legacy.py::test_pass_0"
    ]
    assert runner._audit_gate2_full_junit(extra_skip, **kwargs)["status"] == "failed"
    drifted_evidence = deepcopy(kwargs)
    drifted_evidence["baseline_evidence"]["sha256"] = "0" * 64
    drifted = runner._audit_gate2_full_junit(green, **drifted_evidence)
    assert drifted["status"] == "failed"
    assert drifted["baseline_evidence"]["status"] == "failed"


def test_checked_in_gate2_config_matches_frozen_test_contract(tmp_path: Path) -> None:
    checked_in = json.loads(
        (REPO_ROOT / "configs" / "xunce_path_v2_gate2_wheel_v1.json").read_text(
            encoding="utf-8"
        )
    )
    expected = json.loads(_gate2_config(tmp_path).read_text(encoding="utf-8"))

    assert checked_in == expected


@pytest.mark.parametrize(
    ("field_path", "tampered"),
    [
        pytest.param(("schema_version",), "xunce-path-v2-gate2-wheel/v1", id="schema"),
        pytest.param(("stage_id",), "xunce-path-v2-gate3-other", id="stage"),
        pytest.param(("expected_git", "gate_input_commit"), GATE2_INPUT_COMMIT, id="gate-input"),
        pytest.param(("formal_output_root",), "D:/xunce/out/path_v2/other", id="formal-root"),
        pytest.param(("temp_root",), "D:/xunce/tmp/path_v2_other", id="temp-root"),
        pytest.param(("focused", "pytest_targets"), list(GATE2_FOCUSED_TARGETS), id="focused"),
        pytest.param(("full", "legacy_expected", "passed"), 157, id="legacy"),
        pytest.param(("ablation", "cases"), list(reversed(GATE3_CASES)), id="cases"),
        pytest.param(("ablation", "worker_counts"), [4, 1], id="workers"),
        pytest.param(("ablation", "python_hash_seeds"), [11, 29, 48], id="seeds"),
        pytest.param(("ablation", "repeat_count"), 2, id="repeats"),
        pytest.param(
            ("expected_disabled_accelerators",),
            GATE3_DISABLED_ACCELERATORS[:-1],
            id="disabled-disclosure",
        ),
        pytest.param(("pass_route",), "release_default_policy", id="route"),
    ],
)
def test_gate3_frozen_config_rejects_tampering_before_side_effects(
    tmp_path: Path,
    monkeypatch,
    field_path,
    tampered,
) -> None:
    runner = _runner()
    config_path = _gate3_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target = config
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = tampered
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("Gate 3 side effect occurred before config rejection")

    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    monkeypatch.setattr(runner.gate_artifacts, "write_gate_artifacts", forbidden)
    with pytest.raises(ValueError, match="frozen"):
        runner.run_gate_benchmark(
            config_path,
            tmp_path / "out",
            REPO_ROOT,
            execute_tests=False,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_gate3_config_rejects_missing_or_extra_top_level_keys(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
) -> None:
    runner = _runner()
    config_path = _gate3_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        config.pop("ablation")
    else:
        config["unfrozen"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        runner.gate_artifacts,
        "write_gate_artifacts",
        lambda **kwargs: pytest.fail("artifacts written for malformed config"),
    )

    with pytest.raises(ValueError, match="top-level keys"):
        runner.run_gate_benchmark(
            config_path,
            tmp_path / "out",
            REPO_ROOT,
            execute_tests=False,
        )


def test_gate3_dry_run_dispatch_writes_exact_artifacts_without_tests_or_probes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()

    def forbidden(*args, **kwargs):
        raise AssertionError("Gate 3 dry run executed tests, probes, or runtime audit")

    monkeypatch.setattr(runner, "_run_pytest", forbidden)
    monkeypatch.setattr(runner, "_run_gate3_probes", forbidden, raising=False)
    monkeypatch.setattr(runner, "_gate3_preflight", forbidden, raising=False)
    output_root = tmp_path / "out"
    summary = runner.run_gate_benchmark(
        _gate3_config(tmp_path),
        output_root,
        REPO_ROOT,
        execute_tests=False,
    )

    assert summary["schema_version"] == "xunce-path-v2-gate3-accelerators/v1"
    assert summary["status"] == "dry_run"
    assert summary["next_required_change"] == "execute_gate3_accelerator_evidence"
    assert summary["blocking_reasons"] == []
    assert summary["disabled_accelerators"] == GATE3_DISABLED_ACCELERATORS
    assert summary["checks"]["boundaries_strict_false"] is True
    assert all(summary[field] is False for field in BOUNDARIES)
    assert {path.name for path in output_root.iterdir()} == CANONICAL_ARTIFACTS
    rows = [
        json.loads(line)
        for line in (output_root / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    ablation = [row for row in rows if row["suite"] == "ablation"]
    assert len(ablation) == 108
    assert [
        (row["case_id"], row["worker_count"], row["python_hash_seed"], row["repeat"])
        for row in ablation
    ] == [
        (case_id, workers, seed, repeat)
        for case_id in GATE3_CASES
        for workers in (1, 4)
        for seed in (11, 29, 47)
        for repeat in (1, 2, 3)
    ]
    assert all(row["status"] == "not_run" for row in ablation)
    assert {row["suite"] for row in rows} == {
        "test",
        "ablation",
        "fallback",
        "disclosure",
        "boundary",
    }
    report = (output_root / "report.md").read_text(encoding="utf-8")
    assert "# Path Planner v2 Gate 3 加速器证据" in report
    assert "v1 仍为默认，v2 仍为 opt-in" in report
    assert "当前组件均处于 disabled" in report
    assert "not_integrated_into_provider" in report
    assert "仅为 simulation proxy 组件证据" in report
    assert "synthetic terrain 仅是 proxy，不是 physical obstacle 数据" in report
    assert "不得标作 `physical_obstacle_cells`" in report
    assert "不声明 runtime speedup 或成功率提升" in report
    assert "不发布 checkpoint" in report
    assert "不替换 default policy" in report
    assert "不连接 executor" in report
    assert "不启动 canary" in report


def test_gate3_registry_entry_has_frozen_defaults() -> None:
    registry = json.loads(
        (REPO_ROOT / "configs" / "stage_registry.json").read_text(encoding="utf-8")
    )
    assert registry["stages"]["xunce-path-v2-gate3-accelerators"] == {
        "script": "scripts/run_xunce_path_v2_gate_benchmark.py",
        "default_config": "configs/xunce_path_v2_gate3_accelerators_v1.json",
        "default_output_root": "D:/xunce/out/path_v2/g3",
        "args": [
            "--config",
            "{config}",
            "--output-root",
            "{output_root}",
            "--repo-root",
            "{repo_root}",
        ],
    }


def test_checked_in_gate3_config_matches_frozen_test_contract(tmp_path: Path) -> None:
    checked_in = json.loads(
        (REPO_ROOT / "configs" / "xunce_path_v2_gate3_accelerators_v1.json").read_text(
            encoding="utf-8"
        )
    )
    expected = json.loads(_gate3_config(tmp_path).read_text(encoding="utf-8"))
    assert checked_in == expected


def test_gate3_probe_audit_requires_exact_matrix_order_and_one_digest() -> None:
    runner = _runner()
    rows = _gate3_probe_rows()

    audit = runner._audit_gate3_probe_rows(rows)

    assert audit["status"] == "passed"
    assert audit["row_count"] == 108
    assert audit["decision_digest"] == "a" * 64
    assert audit["matrix_complete"] is True
    assert audit["stable_row_order"] is True
    assert audit["one_decision_digest"] is True
    assert audit["safety_equivalence"] is True
    assert audit["suggestion_non_authority"] is True
    assert audit["hierarchy_conservatism"] is True
    assert audit["cache_l2_equivalence"] is True
    assert audit["fallback_isolation"] is True
    assert audit["provider_accelerator_unused"] is True

    wrong_order = deepcopy(rows)
    wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
    assert runner._audit_gate3_probe_rows(wrong_order)["status"] == "failed"

    digest_drift = deepcopy(rows)
    digest_drift[-1]["decision_digest"] = "b" * 64
    drift_audit = runner._audit_gate3_probe_rows(digest_drift)
    assert drift_audit["status"] == "failed"
    assert drift_audit["one_decision_digest"] is False


def test_gate3_optional_accelerator_failure_isolated_fallback_does_not_fail_gate() -> None:
    runner = _runner()
    rows = _gate3_probe_rows()
    hierarchy_row = next(row for row in rows if row["case_id"] == "hierarchy_only")
    hierarchy_row["runtime_disabled_accelerators"] = [
        {"accelerator_id": "hierarchy", "reason": "component_probe_failed"}
    ]

    audit = runner._audit_gate3_probe_rows(rows)

    assert audit["status"] == "passed"
    assert audit["fallback_isolation"] is True
    assert audit["runtime_fallback_count"] == 1


@pytest.mark.parametrize(
    "fatal_reason",
    [
        "planning_deadline_expired",
        "fine_anchor_failed",
        "l2_authority_malformed",
        "l2_rejected",
    ],
)
def test_gate3_timeout_fine_anchor_and_l2_failures_are_never_swallowed(
    fatal_reason: str,
) -> None:
    runner = _runner()
    rows = _gate3_probe_rows()
    rows[0].update(status="failed", fatal_reason=fatal_reason)

    audit = runner._audit_gate3_probe_rows(rows)

    assert audit["status"] == "failed"
    assert audit["fatal_authority_clean"] is False
    assert fatal_reason in audit["fatal_reasons"]


def test_gate3_runtime_fallback_cannot_disable_an_unrelated_accelerator() -> None:
    runner = _runner()
    rows = _gate3_probe_rows()
    rows[0]["runtime_disabled_accelerators"] = [
        {"accelerator_id": "hierarchy", "reason": "component_probe_failed"}
    ]

    audit = runner._audit_gate3_probe_rows(rows)

    assert audit["status"] == "failed"
    assert audit["fallback_isolation"] is False


def test_gate3_probe_audit_fails_closed_on_malformed_matrix_dimension() -> None:
    runner = _runner()
    rows = _gate3_probe_rows()
    rows[0]["worker_count"] = "1"

    audit = runner._audit_gate3_probe_rows(rows)

    assert audit["status"] == "failed"
    assert audit["matrix_complete"] is False
    assert audit["stable_row_order"] is False


def test_gate3_real_probe_is_isolated_and_exercises_component_safety_contracts(
    tmp_path: Path,
) -> None:
    runner = _runner()
    common_env = runner._common_env(REPO_ROOT, tmp_path / "probe")

    result = runner._run_gate3_probe_process(
        python=Path("D:/conda_envs/lunar-explorer/python.exe"),
        repo_root=REPO_ROOT,
        worker_count=4,
        hash_seed=11,
        repeat=1,
        common_env=common_env,
    )

    assert result["returncode"] == 0
    assert result["stable_failure_reason"] is None
    assert [row["case_id"] for row in result["rows"]] == GATE3_CASES
    for row in result["rows"]:
        assert row["status"] == "passed"
        assert row["safety_equivalent"] is True
        assert row["authoritative_order_preserved"] is True
        assert row["suggestion_non_authoritative"] is True
        assert row["hierarchy_conservative"] is True
        assert row["cache_l2_equivalent"] is True
        assert row["l2_authority_preserved"] is True
        assert row["fallback_isolated"] is True
        assert row["fatal_reason"] is None
        assert row["accelerator_used"] is False


@pytest.mark.parametrize(
    "successful_l2_calls",
    [
        pytest.param(1, id="fine-l2"),
        pytest.param(2, id="lazy-l2"),
        pytest.param(3, id="cache-l2"),
    ],
)
def test_gate3_real_probe_preserves_typed_l2_timeout_as_fatal_deadline(
    tmp_path: Path,
    monkeypatch,
    successful_l2_calls: int,
) -> None:
    runner = _runner()
    timeout_injection = f"""
import path_planner.v2.validation as _gate3_validation

_gate3_original_validate_route_l2 = _gate3_validation.validate_route_l2
_gate3_l2_call_count = 0


def _gate3_timeout_after_successes(*args, **kwargs):
    global _gate3_l2_call_count
    call_index = _gate3_l2_call_count
    _gate3_l2_call_count += 1
    if call_index < {successful_l2_calls}:
        return _gate3_original_validate_route_l2(*args, **kwargs)
    return _gate3_validation._timeout(
        _gate3_validation.WHEEL_ROUTE_VALIDATOR_ID_V2,
        0,
    )


_gate3_validation.validate_route_l2 = _gate3_timeout_after_successes
"""
    monkeypatch.setattr(
        runner,
        "GATE3_PROBE_CODE",
        timeout_injection + runner.GATE3_PROBE_CODE,
    )
    common_env = runner._common_env(REPO_ROOT, tmp_path / "typed-timeout-probe")

    result = runner._run_gate3_probe_process(
        python=Path("D:/conda_envs/lunar-explorer/python.exe"),
        repo_root=REPO_ROOT,
        worker_count=1,
        hash_seed=11,
        repeat=1,
        common_env=common_env,
    )

    assert result["returncode"] == 0
    assert result["stable_failure_reason"] is None
    assert {row["status"] for row in result["rows"]} == {"failed"}
    assert {row["fatal_reason"] for row in result["rows"]} == {
        "planning_deadline_expired"
    }
    assert all(
        not row["runtime_disabled_accelerators"] for row in result["rows"]
    )


@pytest.mark.parametrize(
    ("fault_mode", "expected_fatal_reason"),
    [
        pytest.param("wrong_type", "l2_authority_malformed", id="wrong-type"),
        pytest.param("exception", "l2_authority_malformed", id="exception"),
        pytest.param(
            "typed_timeout",
            "planning_deadline_expired",
            id="typed-timeout",
        ),
        pytest.param("typed_rejection", "l2_rejected", id="typed-rejection"),
        pytest.param(
            "malformed_then_timeout",
            "l2_authority_malformed",
            id="lazy-fatal-short-circuits-cache",
        ),
    ],
)
def test_gate3_real_probe_rechecks_fine_l2_before_lazy_or_cache_fallback(
    tmp_path: Path,
    monkeypatch,
    fault_mode: str,
    expected_fatal_reason: str,
) -> None:
    runner = _runner()
    fault_injection = f"""
import path_planner.v2.validation as _gate3_validation

_gate3_original_validate_route_l2 = _gate3_validation.validate_route_l2
_gate3_l2_call_count = 0
_gate3_fault_mode = {fault_mode!r}


def _gate3_fault_after_initial_fine(*args, **kwargs):
    global _gate3_l2_call_count
    call_index = _gate3_l2_call_count
    _gate3_l2_call_count += 1
    if call_index < 2:
        return _gate3_original_validate_route_l2(*args, **kwargs)
    if _gate3_fault_mode == "malformed_then_timeout":
        if call_index < 4:
            return object()
        return _gate3_validation._timeout(
            _gate3_validation.WHEEL_ROUTE_VALIDATOR_ID_V2,
            0,
        )
    if _gate3_fault_mode == "wrong_type":
        return object()
    if _gate3_fault_mode == "exception":
        raise RuntimeError("injected L2 authority failure")
    if _gate3_fault_mode == "typed_timeout":
        return _gate3_validation._timeout(
            _gate3_validation.WHEEL_ROUTE_VALIDATOR_ID_V2,
            0,
        )
    return _gate3_validation._result(
        _gate3_validation.WHEEL_ROUTE_VALIDATOR_ID_V2,
        "primitive_structure_mismatch",
    )


_gate3_validation.validate_route_l2 = _gate3_fault_after_initial_fine
"""
    monkeypatch.setattr(
        runner,
        "GATE3_PROBE_CODE",
        fault_injection + runner.GATE3_PROBE_CODE,
    )
    common_env = runner._common_env(REPO_ROOT, tmp_path / "l2-recheck-probe")

    result = runner._run_gate3_probe_process(
        python=Path("D:/conda_envs/lunar-explorer/python.exe"),
        repo_root=REPO_ROOT,
        worker_count=1,
        hash_seed=11,
        repeat=1,
        common_env=common_env,
    )

    assert result["returncode"] == 0
    assert result["stable_failure_reason"] is None
    assert {row["status"] for row in result["rows"]} == {"failed"}
    assert {row["fatal_reason"] for row in result["rows"]} == {
        expected_fatal_reason
    }
    assert all(
        not row["runtime_disabled_accelerators"] for row in result["rows"]
    )


@pytest.mark.parametrize(
    ("stage", "fault_call_index"),
    [
        pytest.param("lazy", 2, id="lazy"),
        pytest.param("cache", 3, id="cache"),
    ],
)
@pytest.mark.parametrize(
    ("fault_mode", "expected_fatal_reason"),
    [
        pytest.param("wrong_type", "l2_authority_malformed", id="wrong-type"),
        pytest.param("exception", "l2_authority_malformed", id="exception"),
        pytest.param(
            "typed_timeout",
            "planning_deadline_expired",
            id="typed-timeout",
        ),
        pytest.param("typed_rejection", "l2_rejected", id="typed-rejection"),
    ],
)
def test_gate3_real_probe_preserves_one_shot_l2_fault_source(
    tmp_path: Path,
    monkeypatch,
    stage: str,
    fault_call_index: int,
    fault_mode: str,
    expected_fatal_reason: str,
) -> None:
    runner = _runner()
    fault_injection = f"""
import path_planner.v2.validation as _gate3_validation

_gate3_original_validate_route_l2 = _gate3_validation.validate_route_l2
_gate3_l2_call_count = 0
_gate3_fault_call_index = {fault_call_index}
_gate3_fault_mode = {fault_mode!r}


def _gate3_one_shot_l2_fault(*args, **kwargs):
    global _gate3_l2_call_count
    call_index = _gate3_l2_call_count
    _gate3_l2_call_count += 1
    if call_index != _gate3_fault_call_index:
        return _gate3_original_validate_route_l2(*args, **kwargs)
    if _gate3_fault_mode == "wrong_type":
        return object()
    if _gate3_fault_mode == "exception":
        raise RuntimeError("injected one-shot L2 failure")
    if _gate3_fault_mode == "typed_timeout":
        return _gate3_validation._timeout(
            _gate3_validation.WHEEL_ROUTE_VALIDATOR_ID_V2,
            0,
        )
    return _gate3_validation._result(
        _gate3_validation.WHEEL_ROUTE_VALIDATOR_ID_V2,
        "primitive_structure_mismatch",
    )


_gate3_validation.validate_route_l2 = _gate3_one_shot_l2_fault
"""
    monkeypatch.setattr(
        runner,
        "GATE3_PROBE_CODE",
        fault_injection + runner.GATE3_PROBE_CODE,
    )
    common_env = runner._common_env(
        REPO_ROOT,
        tmp_path / f"one-shot-{stage}-{fault_mode}",
    )

    result = runner._run_gate3_probe_process(
        python=Path("D:/conda_envs/lunar-explorer/python.exe"),
        repo_root=REPO_ROOT,
        worker_count=1,
        hash_seed=11,
        repeat=1,
        common_env=common_env,
    )

    assert result["returncode"] == 0
    assert result["stable_failure_reason"] is None
    assert {row["status"] for row in result["rows"]} == {"failed"}
    assert {row["fatal_reason"] for row in result["rows"]} == {
        expected_fatal_reason
    }
    assert all(
        not row["runtime_disabled_accelerators"] for row in result["rows"]
    )


@pytest.mark.parametrize(
    ("fault_mode", "expected_fatal_reason"),
    [
        pytest.param(
            "timeout",
            "planning_deadline_expired",
            id="timeout",
        ),
        pytest.param(
            "exception",
            "l2_authority_malformed",
            id="exception",
        ),
    ],
)
def test_gate3_real_probe_classifies_initial_fine_l2_exceptions(
    tmp_path: Path,
    monkeypatch,
    fault_mode: str,
    expected_fatal_reason: str,
) -> None:
    runner = _runner()
    fault_injection = f"""
import path_planner.v2.validation as _gate3_validation

_gate3_original_validate_route_l2 = _gate3_validation.validate_route_l2
_gate3_l2_call_count = 0
_gate3_fault_mode = {fault_mode!r}


def _gate3_fail_initial_fine(*args, **kwargs):
    global _gate3_l2_call_count
    call_index = _gate3_l2_call_count
    _gate3_l2_call_count += 1
    if call_index == 0:
        return _gate3_original_validate_route_l2(*args, **kwargs)
    if _gate3_fault_mode == "timeout":
        raise TimeoutError("injected initial fine deadline")
    raise RuntimeError("injected initial fine L2 failure")


_gate3_validation.validate_route_l2 = _gate3_fail_initial_fine
"""
    monkeypatch.setattr(
        runner,
        "GATE3_PROBE_CODE",
        fault_injection + runner.GATE3_PROBE_CODE,
    )
    common_env = runner._common_env(REPO_ROOT, tmp_path / "initial-fine-probe")

    result = runner._run_gate3_probe_process(
        python=Path("D:/conda_envs/lunar-explorer/python.exe"),
        repo_root=REPO_ROOT,
        worker_count=1,
        hash_seed=11,
        repeat=1,
        common_env=common_env,
    )

    assert result["returncode"] == 0
    assert result["stable_failure_reason"] is None
    assert {row["status"] for row in result["rows"]} == {"failed"}
    assert {row["fatal_reason"] for row in result["rows"]} == {
        expected_fatal_reason
    }
    assert all(
        not row["runtime_disabled_accelerators"] for row in result["rows"]
    )


def test_gate3_real_probe_allows_component_fallback_after_exact_l2_recheck_passes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    fault_injection = """
import path_planner.v2.validation as _gate3_validation


def _gate3_component_failure(*args, **kwargs):
    raise RuntimeError("injected optional component failure")


_gate3_validation.validate_route = _gate3_component_failure
"""
    monkeypatch.setattr(
        runner,
        "GATE3_PROBE_CODE",
        fault_injection + runner.GATE3_PROBE_CODE,
    )
    common_env = runner._common_env(REPO_ROOT, tmp_path / "component-fallback-probe")

    result = runner._run_gate3_probe_process(
        python=Path("D:/conda_envs/lunar-explorer/python.exe"),
        repo_root=REPO_ROOT,
        worker_count=1,
        hash_seed=11,
        repeat=1,
        common_env=common_env,
    )

    assert result["returncode"] == 0
    assert result["stable_failure_reason"] is None
    assert {row["status"] for row in result["rows"]} == {"passed"}
    assert {row["fatal_reason"] for row in result["rows"]} == {None}
    disabled_by_case = {
        row["case_id"]: [
            item["accelerator_id"]
            for item in row["runtime_disabled_accelerators"]
        ]
        for row in result["rows"]
    }
    assert disabled_by_case == {
        "fine_only": [],
        "multi_heuristic_only": [],
        "hierarchy_only": [],
        "lazy_validation_only": ["lazy_validation"],
        "lazy_validation_plus_cache": [
            "lazy_validation",
            "validation_cache",
        ],
        "full_v2": ["lazy_validation", "validation_cache"],
    }


@pytest.mark.parametrize(
    "stage",
    [
        pytest.param("entry", id="entry-construction"),
        pytest.param("init", id="queue-init"),
        pytest.param("extend", id="queue-extend"),
    ],
)
@pytest.mark.parametrize(
    ("fault_type", "expected_fatal_reason"),
    [
        pytest.param(
            "TimeoutError",
            "planning_deadline_expired",
            id="deadline",
        ),
        pytest.param("RuntimeError", None, id="optional-component-failure"),
    ],
)
def test_gate3_real_probe_classifies_the_entire_search_probe_scope(
    tmp_path: Path,
    monkeypatch,
    stage: str,
    fault_type: str,
    expected_fatal_reason: str | None,
) -> None:
    runner = _runner()
    fault_injection = f"""
import path_planner.v2.search as _gate3_search

_gate3_search_stage = {stage!r}
_gate3_search_fault_type = {fault_type}


def _gate3_search_fault(*args, **kwargs):
    raise _gate3_search_fault_type("injected search probe failure")


if _gate3_search_stage == "entry":
    _gate3_search.SearchQueueEntryV2 = _gate3_search_fault
elif _gate3_search_stage == "init":
    _gate3_search.StableSearchQueueV2.__init__ = _gate3_search_fault
else:
    _gate3_search.StableSearchQueueV2.extend = _gate3_search_fault
"""
    monkeypatch.setattr(
        runner,
        "GATE3_PROBE_CODE",
        fault_injection + runner.GATE3_PROBE_CODE,
    )
    common_env = runner._common_env(
        REPO_ROOT,
        tmp_path / f"search-{stage}-{fault_type}",
    )

    result = runner._run_gate3_probe_process(
        python=Path("D:/conda_envs/lunar-explorer/python.exe"),
        repo_root=REPO_ROOT,
        worker_count=1,
        hash_seed=11,
        repeat=1,
        common_env=common_env,
    )

    assert result["returncode"] == 0
    assert result["stable_failure_reason"] is None
    if expected_fatal_reason is not None:
        assert {row["status"] for row in result["rows"]} == {"failed"}
        assert {row["fatal_reason"] for row in result["rows"]} == {
            expected_fatal_reason
        }
        assert all(
            not row["runtime_disabled_accelerators"]
            for row in result["rows"]
        )
    else:
        assert {row["status"] for row in result["rows"]} == {"passed"}
        assert {row["fatal_reason"] for row in result["rows"]} == {None}
        disabled_by_case = {
            row["case_id"]: [
                item["accelerator_id"]
                for item in row["runtime_disabled_accelerators"]
            ]
            for row in result["rows"]
        }
        assert disabled_by_case == {
            "fine_only": [],
            "multi_heuristic_only": ["multi_heuristic"],
            "hierarchy_only": [],
            "lazy_validation_only": [],
            "lazy_validation_plus_cache": [],
            "full_v2": ["multi_heuristic"],
        }


def test_gate3_initial_l2_fatal_precedes_later_hierarchy_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    fault_injection = """
import path_planner.v2.hierarchy as _gate3_hierarchy
import path_planner.v2.validation as _gate3_validation

_gate3_original_validate_route_l2 = _gate3_validation.validate_route_l2
_gate3_l2_call_count = 0


def _gate3_initial_l2_malformed(*args, **kwargs):
    global _gate3_l2_call_count
    call_index = _gate3_l2_call_count
    _gate3_l2_call_count += 1
    if call_index == 1:
        return object()
    return _gate3_original_validate_route_l2(*args, **kwargs)


def _gate3_later_hierarchy_timeout(cls, *args, **kwargs):
    raise TimeoutError("hierarchy must not replace initial L2 fatal")


_gate3_validation.validate_route_l2 = _gate3_initial_l2_malformed
_gate3_hierarchy.ConservativeHierarchyV2.build = classmethod(
    _gate3_later_hierarchy_timeout
)
"""
    monkeypatch.setattr(
        runner,
        "GATE3_PROBE_CODE",
        fault_injection + runner.GATE3_PROBE_CODE,
    )
    common_env = runner._common_env(
        REPO_ROOT,
        tmp_path / "initial-l2-before-hierarchy-timeout",
    )

    result = runner._run_gate3_probe_process(
        python=Path("D:/conda_envs/lunar-explorer/python.exe"),
        repo_root=REPO_ROOT,
        worker_count=1,
        hash_seed=11,
        repeat=1,
        common_env=common_env,
    )

    assert result["returncode"] == 0
    assert result["stable_failure_reason"] is None
    assert {row["status"] for row in result["rows"]} == {"failed"}
    assert {row["fatal_reason"] for row in result["rows"]} == {
        "l2_authority_malformed"
    }
    assert all(
        not row["runtime_disabled_accelerators"] for row in result["rows"]
    )


def test_gate3_initial_l2_fatal_short_circuits_search_probe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    fault_injection = """
import path_planner.v2.search as _gate3_search
import path_planner.v2.validation as _gate3_validation

_gate3_original_validate_route_l2 = _gate3_validation.validate_route_l2
_gate3_l2_call_count = 0


def _gate3_initial_l2_malformed(*args, **kwargs):
    global _gate3_l2_call_count
    call_index = _gate3_l2_call_count
    _gate3_l2_call_count += 1
    if call_index == 1:
        return object()
    return _gate3_original_validate_route_l2(*args, **kwargs)


def _gate3_search_must_not_run(*args, **kwargs):
    raise SystemExit("search ran after initial fatal")


_gate3_validation.validate_route_l2 = _gate3_initial_l2_malformed
_gate3_search.SearchQueueEntryV2 = _gate3_search_must_not_run
"""
    monkeypatch.setattr(
        runner,
        "GATE3_PROBE_CODE",
        fault_injection + runner.GATE3_PROBE_CODE,
    )
    common_env = runner._common_env(
        REPO_ROOT,
        tmp_path / "initial-l2-short-circuits-search",
    )

    result = runner._run_gate3_probe_process(
        python=Path("D:/conda_envs/lunar-explorer/python.exe"),
        repo_root=REPO_ROOT,
        worker_count=1,
        hash_seed=11,
        repeat=1,
        common_env=common_env,
    )

    assert result["returncode"] == 0
    assert result["stable_failure_reason"] is None
    assert {row["status"] for row in result["rows"]} == {"failed"}
    assert {row["fatal_reason"] for row in result["rows"]} == {
        "l2_authority_malformed"
    }


def test_gate3_search_deadline_short_circuits_all_later_optional_probes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    fault_injection = """
import path_planner.v2.hierarchy as _gate3_hierarchy
import path_planner.v2.search as _gate3_search
import path_planner.v2.validation as _gate3_validation

_gate3_original_validate_route_l2 = _gate3_validation.validate_route_l2
_gate3_l2_call_count = 0


def _gate3_l2_must_not_run_after_initial(*args, **kwargs):
    global _gate3_l2_call_count
    call_index = _gate3_l2_call_count
    _gate3_l2_call_count += 1
    if call_index < 2:
        return _gate3_original_validate_route_l2(*args, **kwargs)
    raise SystemExit("lazy or cache ran after search deadline")


def _gate3_search_timeout(*args, **kwargs):
    raise TimeoutError("injected search deadline")


def _gate3_hierarchy_must_not_run(cls, *args, **kwargs):
    raise SystemExit("hierarchy ran after search deadline")


_gate3_validation.validate_route_l2 = _gate3_l2_must_not_run_after_initial
_gate3_search.StableSearchQueueV2.__init__ = _gate3_search_timeout
_gate3_hierarchy.ConservativeHierarchyV2.build = classmethod(
    _gate3_hierarchy_must_not_run
)
"""
    monkeypatch.setattr(
        runner,
        "GATE3_PROBE_CODE",
        fault_injection + runner.GATE3_PROBE_CODE,
    )
    common_env = runner._common_env(
        REPO_ROOT,
        tmp_path / "search-deadline-short-circuits-later",
    )

    result = runner._run_gate3_probe_process(
        python=Path("D:/conda_envs/lunar-explorer/python.exe"),
        repo_root=REPO_ROOT,
        worker_count=1,
        hash_seed=11,
        repeat=1,
        common_env=common_env,
    )

    assert result["returncode"] == 0
    assert result["stable_failure_reason"] is None
    assert {row["status"] for row in result["rows"]} == {"failed"}
    assert {row["fatal_reason"] for row in result["rows"]} == {
        "planning_deadline_expired"
    }
    assert all(
        not row["runtime_disabled_accelerators"] for row in result["rows"]
    )


def test_gate3_real_probe_never_downgrades_hierarchy_timeout_to_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    timeout_injection = """
import path_planner.v2.hierarchy as _gate3_hierarchy


def _gate3_hierarchy_timeout(cls, *args, **kwargs):
    raise TimeoutError("injected hierarchy deadline")


_gate3_hierarchy.ConservativeHierarchyV2.build = classmethod(
    _gate3_hierarchy_timeout
)
"""
    monkeypatch.setattr(
        runner,
        "GATE3_PROBE_CODE",
        timeout_injection + runner.GATE3_PROBE_CODE,
    )
    common_env = runner._common_env(REPO_ROOT, tmp_path / "hierarchy-timeout-probe")

    result = runner._run_gate3_probe_process(
        python=Path("D:/conda_envs/lunar-explorer/python.exe"),
        repo_root=REPO_ROOT,
        worker_count=1,
        hash_seed=11,
        repeat=1,
        common_env=common_env,
    )

    assert result["returncode"] == 0
    assert result["stable_failure_reason"] is None
    assert {row["status"] for row in result["rows"]} == {"failed"}
    assert {row["fatal_reason"] for row in result["rows"]} == {
        "planning_deadline_expired"
    }
    assert all(
        not row["runtime_disabled_accelerators"] for row in result["rows"]
    )


def test_gate3_full_audit_preserves_every_gate0_nodeid_and_allows_only_new_v2_passes(
    tmp_path: Path,
) -> None:
    runner = _runner()
    baseline = tmp_path / "baseline.xml"
    current = tmp_path / "current.xml"
    missing = tmp_path / "missing.xml"
    baseline_cases = _full_cases()[:173]
    current_cases = baseline_cases + [
        (f"tests/test_v2_gate3.py::test_new_{index}", "passed", "")
        for index in range(20)
    ]
    _write_junit(baseline, baseline_cases)
    _write_junit(current, current_cases)
    _write_junit(missing, current_cases[1:])
    kwargs = {
        "expected_legacy": {"passed": 156, "skipped": 17, "failures": 0, "errors": 0},
        "allowed_skip_dependency": "pydrake",
        "baseline_evidence": {
            "schema_version": "test-baseline/v1",
            "path": baseline.as_posix(),
            "sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
            "test_count": 173,
        },
    }

    green = runner._audit_gate3_full_junit(current, **kwargs)
    drifted = runner._audit_gate3_full_junit(missing, **kwargs)

    assert green["schema_version"] == "xunce-path-v2-gate3-full-junit-audit/v1"
    assert green["status"] == "passed"
    assert green["baseline_not_reduced"] is True
    assert green["v2"]["skipped"] == 0
    assert drifted["status"] == "failed"
    assert drifted["missing_baseline_nodeids"]


def test_gate3_green_execution_passes_with_stable_rows_and_explicit_disclosure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    events: list[str] = []
    _install_gate3_green_mocks(monkeypatch, runner, events)
    output_root = tmp_path / "out"

    summary = runner.run_gate_benchmark(
        _gate3_config(tmp_path),
        output_root,
        REPO_ROOT,
        execute_tests=True,
    )

    assert events == ["focused", "full", "probes", "postflight"]
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "implement_path_v2_legged_static_stability_oracle"
    assert summary["blocking_reasons"] == []
    assert summary["disabled_accelerators"] == GATE3_DISABLED_ACCELERATORS
    assert summary["checks"]["provider_accelerator_unused"] is True
    assert summary["ablation"]["decision_digest"] == "a" * 64
    assert all(summary[field] is False for field in BOUNDARIES)

    rows = [
        json.loads(line)
        for line in (output_root / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    ablation = [row for row in rows if row["suite"] == "ablation"]
    assert [
        (
            row["case_id"],
            row["worker_count"],
            row["python_hash_seed"],
            row["repeat"],
        )
        for row in ablation
    ] == [
        (case_id, workers, seed, repeat)
        for case_id in GATE3_CASES
        for workers in (1, 4)
        for seed in (11, 29, 47)
        for repeat in (1, 2, 3)
    ]
    assert {row["suite"] for row in rows} == {
        "test",
        "ablation",
        "fallback",
        "disclosure",
        "boundary",
    }
    routing = json.loads((output_root / "routing.json").read_text(encoding="utf-8"))
    assert routing["route"] == summary["next_required_change"]
    assert routing["disabled_accelerators"] == GATE3_DISABLED_ACCELERATORS
    review = json.loads((output_root / "review.json").read_text(encoding="utf-8"))
    assert review["probe_audit"]["row_count"] == 108
    assert review["execution"]["probe_commands"][0]["environment"]["PYTHONHASHSEED"] == "11"
    assert "stderr" not in json.dumps(review["execution"]["probe_commands"])
    assert runner.status_exit_code(summary["status"]) == 0


def test_gate3_postflight_identity_drift_fails_with_stable_repair_route(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    events: list[str] = []
    _install_gate3_green_mocks(monkeypatch, runner, events)
    monkeypatch.setattr(
        runner,
        "_gate3_postflight",
        lambda config, repo_root: _postflight_with_mutation("nested-gitlink-drift"),
    )

    summary = runner.run_gate_benchmark(
        _gate3_config(tmp_path),
        tmp_path / "out",
        REPO_ROOT,
        execute_tests=True,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "restore_gate3_runtime_isolation"
    assert summary["checks"]["postflight"] is False
    assert runner.status_exit_code(summary["status"]) == 1


def test_gate3_fatal_l2_probe_failure_is_failed_not_optional_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    rows = _gate3_probe_rows()
    rows[0].update(status="failed", fatal_reason="l2_rejected")
    events: list[str] = []
    _install_gate3_green_mocks(monkeypatch, runner, events, probe_rows=rows)

    summary = runner.run_gate_benchmark(
        _gate3_config(tmp_path),
        tmp_path / "out",
        REPO_ROOT,
        execute_tests=True,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "restore_gate3_fine_l2_authority"
    assert summary["checks"]["fatal_authority_clean"] is False
    assert summary["blocking_reasons"] == []
    assert summary["disabled_accelerators"] == GATE3_DISABLED_ACCELERATORS


def test_gate3_dry_run_rejects_stale_noncanonical_artifact_without_deletion(
    tmp_path: Path,
) -> None:
    runner = _runner()
    output_root = tmp_path / "out"
    output_root.mkdir()
    stale = output_root / "stale.txt"
    stale.write_text("preserve", encoding="utf-8")

    with pytest.raises(RuntimeError, match="stale noncanonical"):
        runner.run_gate_benchmark(
            _gate3_config(tmp_path),
            output_root,
            REPO_ROOT,
            execute_tests=False,
        )
    assert stale.read_text(encoding="utf-8") == "preserve"
