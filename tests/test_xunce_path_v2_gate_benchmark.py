from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

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
GATE4_INPUT_COMMIT = "b7271935d0ad39a597df789e51aa62caa525eec6"
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
GATE4_FOCUSED_TARGETS = [
    "tests/test_v2_benchmark.py",
    "tests/test_v2_legged_oracle.py",
    "tests/test_v2_legged_provider.py",
    "tests/test_v2_route_validation.py",
    "tests/test_v2_profiles.py",
    "tests/test_v2_geometry.py",
    "tests/test_v2_search.py",
    "tests/test_v2_api.py",
    "tests/test_v2_runtime.py",
    "tests/test_v2_contracts.py",
    "tests/test_v2_serialization.py",
    "tests/test_v2_fine_safety_anchor.py",
    "tests/test_hybrid_astar.py",
    "tests/test_astar.py",
    "../tests/test_xunce_path_v2_gate_benchmark.py",
]
GATE4_INPUTS = {
    "primitive_audit": "D:/xunce/inputs/path_v2/g4/independent_legged_oracle_labels.jsonl",
    "exact_map_quality": "D:/xunce/inputs/path_v2/g4/independent_legged_exact_map_optima.jsonl",
    "standard_episodes": "D:/xunce/inputs/path_v2/g4/standard_legged_schedule.jsonl",
}
GATE4_THRESHOLDS = {
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
    "max_standard_unreachable_successes": 0,
    "min_standard_complete_l2_ratio": 1.0,
    "max_standard_p95_runtime_ms": 250.0,
    "hard_timeout_ms": 2000.0,
    "max_hard_timeout_violations": 0,
}
GATE4_CAPABILITY_DISCLOSURE = {
    "platform_kind": "legged",
    "platform_type": "legged_static_crawl",
    "capability": "simulation_proxy",
    "capability_revision": "simulation_proxy_static_crawl/v1",
    "primitive_capability": "simulation_proxy",
    "resource_proxy_id": "legged_static_crawl_relative_resource/v1",
    "route_validator_id": "path-planner-v2-legged-route-l2/v1",
    "dynamic_gait_claimed": False,
    "real_robot_stability_claimed": False,
}
GATE4_DATASET_CONTRACT = {
    "schema_version": "xunce-path-v2-gate4b-blocked-intake/v1",
    "accepts_formal_inputs": False,
    "future_stage_required": "xunce-path-v2-gate4c-legged-evidence",
    "formal_sources": {
        "primitive_audit": "independent-legged-oracle/v1",
        "exact_map_quality": "independent-legged-exact-solver/v1",
        "standard_episodes": "independent-standard-legged/v1",
    },
}
GATE4_TRUSTED_INPUT = {
    "approved_input_sha256": None,
    "approved_case_envelope_sha256": None,
    "approved_producer_id": None,
    "approved_producer_revision": None,
    "approval_id": None,
    "approved_provider_source_commit": "7d5c4dfc8e2a374855e6d8b962b69466c3353265",
    "approved_provider_build_id": None,
    "approved_collector_id": None,
    "approved_collector_revision": None,
    "approved_profile_id": "legged-static-crawl/v1",
    "approved_capability_revision": "simulation_proxy_static_crawl/v1",
    "approved_step_validator_id": "path-planner-v2-legged-static-stability/v1",
    "approved_route_validator_id": "path-planner-v2-legged-route-l2/v1",
}
GATE4_BLOCKERS = [
    "provide_independent_legged_oracle_labels",
    "provide_independent_legged_exact_map_optima",
    "provide_standard_legged_schedule",
]
GATE4_BENCHMARK_ROW_CLASSES = (
    "PrimitiveAuditRowV2",
    "ExactMapQualityRowV2",
    "StandardEpisodeRowV2",
)
GATE4_BENCHMARK_AGGREGATES = (
    "aggregate_primitive_audit_v2",
    "aggregate_exact_map_quality_v2",
    "aggregate_standard_episodes_v2",
)
GATE4_BENCHMARK_CONTRACT_AUDIT = {
    "schema_version": "xunce-path-v2-gate4-benchmark-contract-audit/v1",
    "status": "passed",
    "hard_timeout_ms": 2000.0,
    "typed_row_api": True,
}
GATE4_PHASES = [
    "preflight",
    "focused",
    "full",
    "primitive-audit",
    "exact-map-quality",
    "standard-episodes",
    "boundary-review",
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
        "probe_subprocess_timeout_s": 30.0,
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


def _gate4_payload(
    *,
    formal_output_root: str = "D:/xunce/out/path_v2/g4",
    temp_root: str = "D:/xunce/tmp/path_v2_g4",
    inputs: dict[str, str] | None = None,
    boundaries=None,
) -> dict:
    return {
        "schema_version": "xunce-path-v2-gate4-legged/v1",
        "stage_id": "xunce-path-v2-gate4-legged",
        "python": "D:/conda_envs/lunar-explorer/python.exe",
        "expected_python_version": "3.12.13",
        "expected_git": {
            "branch": EXPECTED_BRANCH,
            "base_commit": ORIGINAL_BASE_COMMIT,
            "gate_input_commit": GATE4_INPUT_COMMIT,
            "nested_branch": EXPECTED_BRANCH,
        },
        "formal_output_root": formal_output_root,
        "temp_root": temp_root,
        "focused": {
            "working_directory": "path-planner",
            "pythonpath": ["path-planner/src"],
            "pytest_targets": list(GATE4_FOCUSED_TARGETS),
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
        "inputs": dict(GATE4_INPUTS if inputs is None else inputs),
        "baseline_evidence": {
            "schema_version": "xunce-path-v2-gate0-path-planner-junit/v1",
            "path": "D:/xunce/out/path_v2/g0/path_planner_baseline.junit.xml",
            "sha256": "90a02eb6af78805bfdf2fcce4828283cb3f4d77f30aae55519f533896a06e676",
            "test_count": 173,
        },
        "thresholds": dict(GATE4_THRESHOLDS),
        "capability_disclosure": dict(GATE4_CAPABILITY_DISCLOSURE),
        "dataset_contract": deepcopy(GATE4_DATASET_CONTRACT),
        "trusted_inputs": {
            dataset: dict(GATE4_TRUSTED_INPUT) for dataset in GATE4_INPUTS
        },
        "boundaries": dict(BOUNDARIES if boundaries is None else boundaries),
        "pass_route": "implement_path_v2_lunar_ballistics_and_hopper_proxy_profile",
    }


def _gate4_config(
    tmp_path: Path,
    *,
    formal_output_root: str = "D:/xunce/out/path_v2/g4",
    temp_root: str = "D:/xunce/tmp/path_v2_g4",
    inputs: dict[str, str] | None = None,
    boundaries=None,
) -> Path:
    path = tmp_path / "gate4.json"
    path.write_text(
        json.dumps(
            _gate4_payload(
                formal_output_root=formal_output_root,
                temp_root=temp_root,
                inputs=inputs,
                boundaries=boundaries,
            )
        ),
        encoding="utf-8",
    )
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
        pytest.param(
            ("probe_subprocess_timeout_s",),
            0.0,
            id="probe-subprocess-timeout",
        ),
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
    stored_manifest = json.loads(
        (output_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert stored_manifest == runner.gate_artifacts.build_manifest_without_self_hash(
        output_root
    )
    assert list(output_root.parent.glob(f".{output_root.name}.staging-*")) == []
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


def test_gate3_dry_run_dispatch_calls_atomic_publisher_not_direct_writer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    output_root = tmp_path / "out"
    atomic_calls: list[dict] = []

    def fake_atomic_publisher(**kwargs):
        atomic_calls.append(kwargs)
        return {"artifact_count": 7, "artifacts": []}

    def forbidden_direct_writer(**kwargs):
        raise AssertionError("Gate 3 called the legacy direct artifact writer")

    monkeypatch.setattr(
        runner.gate_artifacts,
        "write_gate_artifacts_atomically",
        fake_atomic_publisher,
    )
    monkeypatch.setattr(
        runner.gate_artifacts,
        "write_gate_artifacts",
        forbidden_direct_writer,
    )

    summary = runner.run_gate_benchmark(
        _gate3_config(tmp_path),
        output_root,
        REPO_ROOT,
        execute_tests=False,
    )

    assert summary["status"] == "dry_run"
    assert len(atomic_calls) == 1
    assert atomic_calls[0]["output_root"] == output_root.resolve()


@pytest.mark.parametrize(
    "config_factory",
    [
        pytest.param(_config, id="gate1"),
        pytest.param(_gate2_config, id="gate2"),
    ],
)
def test_gate1_and_gate2_dry_runs_keep_legacy_direct_writer(
    tmp_path: Path,
    monkeypatch,
    config_factory,
) -> None:
    runner = _runner()
    output_root = tmp_path / "out"
    direct_calls: list[Path] = []
    original_direct_writer = runner.gate_artifacts.write_gate_artifacts

    def tracking_direct_writer(**kwargs):
        direct_calls.append(Path(kwargs["output_root"]))
        return original_direct_writer(**kwargs)

    def forbidden_atomic_publisher(**kwargs):
        raise AssertionError("Gate 1/2 called the Gate 3 atomic publisher")

    monkeypatch.setattr(
        runner.gate_artifacts,
        "write_gate_artifacts",
        tracking_direct_writer,
    )
    monkeypatch.setattr(
        runner.gate_artifacts,
        "write_gate_artifacts_atomically",
        forbidden_atomic_publisher,
    )

    summary = runner.run_gate_benchmark(
        config_factory(tmp_path),
        output_root,
        REPO_ROOT,
        execute_tests=False,
    )

    assert summary["status"] == "dry_run"
    assert [path.resolve() for path in direct_calls] == [output_root.resolve()]
    assert {path.name for path in output_root.iterdir()} == CANONICAL_ARTIFACTS


def test_gate3_dry_run_partial_writer_never_exposes_canonical_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    output_root = tmp_path / "out"

    def partial_writer(*, output_root, **kwargs):
        staging_root = Path(output_root)
        staging_root.mkdir(parents=True, exist_ok=True)
        (staging_root / "config.json").write_bytes(b"partial")
        raise OSError("injected Gate 3 partial writer")

    monkeypatch.setattr(
        runner.gate_artifacts,
        "write_gate_artifacts",
        partial_writer,
    )

    with pytest.raises(
        RuntimeError,
        match="atomic gate artifact staging write failed",
    ):
        runner.run_gate_benchmark(
            _gate3_config(tmp_path),
            output_root,
            REPO_ROOT,
            execute_tests=False,
        )

    assert not output_root.exists()
    staging_roots = list(
        output_root.parent.glob(f".{output_root.name}.staging-*")
    )
    assert len(staging_roots) == 1
    assert (staging_roots[0] / "config.json").read_bytes() == b"partial"


def test_gate3_dry_run_rename_failure_never_exposes_canonical_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    output_root = tmp_path / "out"
    rename_calls = 0

    def failed_rename(source, destination):
        nonlocal rename_calls
        rename_calls += 1
        raise OSError("injected Gate 3 rename failure")

    monkeypatch.setattr(runner.gate_artifacts.os, "rename", failed_rename)

    with pytest.raises(
        RuntimeError,
        match="atomic gate artifact publish rename failed",
    ):
        runner.run_gate_benchmark(
            _gate3_config(tmp_path),
            output_root,
            REPO_ROOT,
            execute_tests=False,
        )

    assert rename_calls == 1
    assert not output_root.exists()
    staging_roots = list(
        output_root.parent.glob(f".{output_root.name}.staging-*")
    )
    assert len(staging_roots) == 1
    assert {path.name for path in staging_roots[0].iterdir()} == CANONICAL_ARTIFACTS


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
    status, route, checks = runner._evaluate_gate3(
        preflight={"status": "passed"},
        postflight_ok=True,
        focused={"status": "passed"},
        full={"status": "passed"},
        probe_audit=audit,
        boundary_ok=True,
    )
    assert status == "failed"
    assert route == "restore_gate3_fine_l2_authority"
    assert checks["fatal_authority_clean"] is False


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


def test_gate3_real_probe_orchestrator_runs_complete_ordered_matrix(
    tmp_path: Path,
) -> None:
    runner = _runner()
    expected_command_keys = [
        (worker_count, hash_seed, repeat)
        for worker_count in (1, 4)
        for hash_seed in (11, 29, 47)
        for repeat in (1, 2, 3)
    ]
    expected_row_keys = [
        (case_id, worker_count, hash_seed, repeat)
        for case_id in GATE3_CASES
        for worker_count in (1, 4)
        for hash_seed in (11, 29, 47)
        for repeat in (1, 2, 3)
    ]

    result = runner._run_gate3_probes(
        python=Path("D:/conda_envs/lunar-explorer/python.exe"),
        repo_root=REPO_ROOT,
        common_env=runner._common_env(REPO_ROOT, tmp_path / "probe-matrix-real"),
        timeout_s=30.0,
    )

    assert len(result["commands"]) == 18
    assert [
        (
            command["worker_count"],
            command["python_hash_seed"],
            command["repeat"],
        )
        for command in result["commands"]
    ] == expected_command_keys
    assert all(command["returncode"] == 0 for command in result["commands"])
    assert all(
        command["stable_failure_reason"] is None
        for command in result["commands"]
    )
    assert len(result["rows"]) == 108
    assert [
        (
            row["case_id"],
            row["worker_count"],
            row["python_hash_seed"],
            row["repeat"],
        )
        for row in result["rows"]
    ] == expected_row_keys
    assert {row["status"] for row in result["rows"]} == {"passed"}
    assert len({row["decision_digest"] for row in result["rows"]}) == 1
    assert result["one_decision_digest"] is True
    assert result["fatal_reasons"] == []
    assert result["runtime_fallback_count"] == 0
    assert result["status"] == "passed"


def test_gate3_probe_subprocess_timeout_has_stable_failure_sentinel(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    observed: dict[str, object] = {}

    def timeout_run(command, **kwargs):
        observed["command"] = command
        observed["timeout"] = kwargs.get("timeout")
        raise runner.subprocess.TimeoutExpired(
            command,
            kwargs.get("timeout"),
            output="partial probe output",
            stderr="probe deadline exceeded",
        )

    monkeypatch.setattr(runner.subprocess, "run", timeout_run)

    result = runner._run_gate3_probe_process(
        python=Path("D:/conda_envs/lunar-explorer/python.exe"),
        repo_root=REPO_ROOT,
        worker_count=4,
        hash_seed=29,
        repeat=2,
        common_env=runner._common_env(REPO_ROOT, tmp_path / "probe-timeout"),
        timeout_s=30.0,
    )

    assert observed["timeout"] == 30.0
    assert result["returncode"] == 124
    assert result["stable_failure_reason"] == "probe_subprocess_timeout"
    assert result["rows"] == []
    assert result["worker_count"] == 4
    assert result["python_hash_seed"] == 29
    assert result["repeat"] == 2


@pytest.mark.parametrize(
    ("stable_failure_reason", "failure_returncode"),
    [
        pytest.param("probe_subprocess_timeout", 124, id="timeout"),
        pytest.param("probe_subprocess_failed", 17, id="subprocess-failed"),
        pytest.param("probe_output_invalid", 0, id="invalid-output"),
    ],
)
def test_gate3_probe_failure_fills_six_rows_and_routes_to_probe_repair(
    tmp_path: Path,
    monkeypatch,
    stable_failure_reason: str,
    failure_returncode: int,
) -> None:
    runner = _runner()
    failure_key = (4, 29, 2)
    calls: list[tuple[int, int, int]] = []
    child_row_counts: list[tuple[tuple[int, int, int], int]] = []

    def fake_probe_process(*, worker_count, hash_seed, repeat, **kwargs):
        key = (worker_count, hash_seed, repeat)
        calls.append(key)
        reason = stable_failure_reason if key == failure_key else None
        rows = (
            []
            if reason is not None
            else [
                {
                    "case_id": case_id,
                    "worker_count": worker_count,
                    "python_hash_seed": hash_seed,
                    "repeat": repeat,
                    "status": "passed",
                    "decision_digest": "a" * 64,
                    "fine_only_digest": "a" * 64,
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
                for case_id in GATE3_CASES
            ]
        )
        child_row_counts.append((key, len(rows)))
        return {
            "command": ["python", "-c", "<gate3-probe>"],
            "environment": {},
            "worker_count": worker_count,
            "python_hash_seed": hash_seed,
            "repeat": repeat,
            "returncode": failure_returncode if reason is not None else 0,
            "stable_failure_reason": reason,
            "rows": rows,
        }

    monkeypatch.setattr(
        runner,
        "_run_gate3_probe_process",
        fake_probe_process,
    )
    result = runner._run_gate3_probes(
        python=Path("D:/conda_envs/lunar-explorer/python.exe"),
        repo_root=REPO_ROOT,
        common_env=runner._common_env(REPO_ROOT, tmp_path / "probe-matrix"),
        timeout_s=30.0,
    )

    expected_calls = [
        (worker_count, hash_seed, repeat)
        for worker_count in (1, 4)
        for hash_seed in (11, 29, 47)
        for repeat in (1, 2, 3)
    ]
    assert calls == expected_calls
    assert child_row_counts == [
        (key, 0 if key == failure_key else 6) for key in expected_calls
    ]
    assert len(result["commands"]) == 18
    assert [
        (
            command["worker_count"],
            command["python_hash_seed"],
            command["repeat"],
        )
        for command in result["commands"]
    ] == expected_calls
    failure_commands = [
        command
        for command in result["commands"]
        if (
            command["worker_count"],
            command["python_hash_seed"],
            command["repeat"],
        )
        == failure_key
    ]
    assert len(failure_commands) == 1
    assert failure_commands[0]["returncode"] == failure_returncode
    assert failure_commands[0]["stable_failure_reason"] == stable_failure_reason
    assert all(
        command["returncode"] == 0
        and command["stable_failure_reason"] is None
        for command in result["commands"]
        if command is not failure_commands[0]
    )
    assert len(result["rows"]) == 108
    assert [
        (
            row["case_id"],
            row["worker_count"],
            row["python_hash_seed"],
            row["repeat"],
        )
        for row in result["rows"]
    ] == [
        (case_id, worker_count, hash_seed, repeat)
        for case_id in GATE3_CASES
        for worker_count in (1, 4)
        for hash_seed in (11, 29, 47)
        for repeat in (1, 2, 3)
    ]
    failure_rows = [
        row
        for row in result["rows"]
        if (
            row["worker_count"],
            row["python_hash_seed"],
            row["repeat"],
        )
        == failure_key
    ]
    assert len(failure_rows) == 6
    assert [row["case_id"] for row in failure_rows] == GATE3_CASES
    assert {row["status"] for row in failure_rows} == {"failed"}
    assert {row["fatal_reason"] for row in failure_rows} == {
        stable_failure_reason
    }
    assert all(not row["runtime_disabled_accelerators"] for row in failure_rows)
    assert result["status"] == "failed"
    assert result["matrix_complete"] is True
    assert result["stable_row_order"] is True
    assert result["runtime_fallback_count"] == 0
    assert result["fatal_reasons"] == [stable_failure_reason]

    status, route, checks = runner._evaluate_gate3(
        preflight={"status": "passed"},
        postflight_ok=True,
        focused={"status": "passed"},
        full={"status": "passed"},
        probe_audit=result,
        boundary_ok=True,
    )
    assert status == "failed"
    assert route == "repair_gate3_deterministic_component_probes"
    assert checks["fatal_authority_clean"] is False


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
    "fault_type",
    [
        pytest.param("RuntimeError", id="optional-constructor-failure"),
        pytest.param("TimeoutError", id="constructor-deadline"),
    ],
)
def test_gate3_cache_constructor_failure_is_classified_inside_child(
    tmp_path: Path,
    monkeypatch,
    fault_type: str,
) -> None:
    runner = _runner()
    fault_injection = f"""
import path_planner.v2.cache as _gate3_cache

_gate3_cache_fault_type = {fault_type}


def _gate3_cache_constructor_failure(*args, **kwargs):
    raise _gate3_cache_fault_type("injected cache constructor failure")


_gate3_cache.ValidationCacheV2 = _gate3_cache_constructor_failure
"""
    monkeypatch.setattr(
        runner,
        "GATE3_PROBE_CODE",
        fault_injection + runner.GATE3_PROBE_CODE,
    )
    common_env = runner._common_env(
        REPO_ROOT,
        tmp_path / f"cache-constructor-{fault_type}",
    )

    result = runner._run_gate3_probe_process(
        python=Path("D:/conda_envs/lunar-explorer/python.exe"),
        repo_root=REPO_ROOT,
        worker_count=1,
        hash_seed=11,
        repeat=1,
        common_env=common_env,
        timeout_s=30.0,
    )

    assert result["returncode"] == 0
    assert result["stable_failure_reason"] is None
    assert len(result["rows"]) == len(GATE3_CASES)
    if fault_type == "TimeoutError":
        assert {row["status"] for row in result["rows"]} == {"failed"}
        assert {row["fatal_reason"] for row in result["rows"]} == {
            "planning_deadline_expired"
        }
        assert all(
            not row["runtime_disabled_accelerators"]
            for row in result["rows"]
        )
    else:
        assert {row["status"] for row in result["rows"]} == {"passed"}
        assert {row["fatal_reason"] for row in result["rows"]} == {None}
        assert all(row["l2_authority_preserved"] for row in result["rows"])
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
            "lazy_validation_only": [],
            "lazy_validation_plus_cache": ["validation_cache"],
            "full_v2": ["validation_cache"],
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


@pytest.mark.parametrize(
    "root_kind",
    [
        pytest.param("empty", id="empty-directory"),
        pytest.param("old-pass", id="old-eight-artifact-pass"),
        pytest.param("noncanonical", id="noncanonical-file"),
        pytest.param("child-directory", id="child-directory"),
    ],
)
@pytest.mark.parametrize(
    "execute_tests",
    [False, True],
    ids=["dry-run", "execute"],
)
def test_gate3_rejects_every_existing_output_root_before_side_effects(
    tmp_path: Path,
    monkeypatch,
    root_kind: str,
    execute_tests: bool,
) -> None:
    runner = _runner()
    output_root = tmp_path / f"out-{root_kind}"
    if root_kind == "old-pass":
        runner.gate_artifacts.write_gate_artifacts(
            output_root=output_root,
            config={"old": "config"},
            summary={"status": "passed", "old": True},
            routing={"status": "passed", "route": "old-route"},
            rows=[{"status": "passed", "old": True}],
            phases=[{"phase": "old", "status": "completed"}],
            review={"status": "passed", "old": True},
            report="# old PASS\n",
        )
        assert {path.name for path in output_root.iterdir()} == CANONICAL_ARTIFACTS
    else:
        output_root.mkdir()
        if root_kind == "noncanonical":
            (output_root / "stale.bin").write_bytes(b"preserve-noncanonical")
        elif root_kind == "child-directory":
            child = output_root / "nested"
            child.mkdir()
            (child / "marker.bin").write_bytes(b"preserve-child")

    def snapshot() -> tuple[tuple[str, str, bytes | None], ...]:
        records = []
        for path in sorted(output_root.rglob("*"), key=lambda item: item.as_posix()):
            relative = path.relative_to(output_root).as_posix()
            if path.is_dir():
                records.append(("directory", relative, None))
            else:
                records.append(("file", relative, path.read_bytes()))
        return tuple(records)

    before = snapshot()
    forbidden_calls: list[str] = []

    def forbidden(name: str):
        def fail(*args, **kwargs):
            forbidden_calls.append(name)
            raise AssertionError(f"{name} ran before existing-root rejection")

        return fail

    monkeypatch.setattr(runner, "_gate3_preflight", forbidden("preflight"))
    monkeypatch.setattr(runner, "_run_pytest", forbidden("tests"))
    monkeypatch.setattr(runner, "_run_gate3_probes", forbidden("probes"))
    monkeypatch.setattr(
        runner.gate_artifacts,
        "write_gate_artifacts",
        forbidden("writer"),
    )

    with pytest.raises(RuntimeError, match="Gate 3 output_root already exists"):
        runner.run_gate_benchmark(
            _gate3_config(tmp_path),
            output_root,
            REPO_ROOT,
            execute_tests=execute_tests,
        )

    assert forbidden_calls == []
    assert output_root.is_dir()
    assert snapshot() == before


def _gate4_tampers() -> list:
    cases = [
        pytest.param(("stage_id",), "other", id="stage"),
        pytest.param(("python",), "D:/other/python.exe", id="python"),
        pytest.param(("expected_python_version",), "0.0.0", id="python-version"),
        pytest.param(("expected_git", "branch"), "codex/other", id="branch"),
        pytest.param(("expected_git", "base_commit"), GATE4_INPUT_COMMIT, id="base"),
        pytest.param(("expected_git", "gate_input_commit"), ORIGINAL_BASE_COMMIT, id="input-commit"),
        pytest.param(("expected_git", "nested_branch"), "codex/other", id="nested-branch"),
        pytest.param(("formal_output_root",), "D:/xunce/out/path_v2/g4-other", id="formal-root"),
        pytest.param(("temp_root",), "D:/xunce/tmp/path_v2_g4_other", id="temp-root"),
        pytest.param(("focused", "working_directory"), ".", id="focused-cwd"),
        pytest.param(("focused", "pythonpath"), ["src"], id="focused-pythonpath"),
        pytest.param(("focused", "pytest_targets"), GATE4_FOCUSED_TARGETS[:-1], id="focused-targets"),
        pytest.param(("full", "working_directory"), ".", id="full-cwd"),
        pytest.param(("full", "pythonpath"), ["src"], id="full-pythonpath"),
        pytest.param(("full", "pytest_targets"), ["tests", "extra"], id="full-targets"),
        pytest.param(("full", "legacy_expected", "passed"), 157, id="legacy-count"),
        pytest.param(("full", "allowed_skip_dependency"), "numpy", id="skip-dependency"),
        pytest.param(("baseline_evidence", "sha256"), "0" * 64, id="baseline-hash"),
        pytest.param(("pass_route",), "release_default_policy", id="pass-route"),
    ]
    cases.extend(
        pytest.param(("inputs", name), f"D:/wrong/{name}.jsonl", id=f"input-{name}")
        for name in GATE4_INPUTS
    )
    cases.extend(
        pytest.param(("thresholds", name), True, id=f"threshold-{name}")
        for name in GATE4_THRESHOLDS
    )
    cases.extend(
        pytest.param(
            ("thresholds", name),
            (
                value + 1
                if isinstance(value, int)
                else value - 0.01
                if value <= 1.10
                else value + 1.0
            ),
            id=f"threshold-numeric-drift-{name}",
        )
        for name, value in GATE4_THRESHOLDS.items()
    )
    cases.extend(
        pytest.param(
            ("capability_disclosure", name),
            (not value if isinstance(value, bool) else "tampered"),
            id=f"capability-{name}",
        )
        for name, value in GATE4_CAPABILITY_DISCLOSURE.items()
    )
    cases.extend(
        pytest.param(
            ("dataset_contract", "formal_sources", name),
            "self-labelled/v1",
            id=f"source-{name}",
        )
        for name in GATE4_INPUTS
    )
    cases.extend(
        [
            pytest.param(
                ("dataset_contract", "accepts_formal_inputs"), True, id="accepts-inputs"
            ),
            pytest.param(
                ("dataset_contract", "schema_version"), "other", id="intake-schema"
            ),
            pytest.param(
                ("dataset_contract", "future_stage_required"), "this-stage", id="future-stage"
            ),
        ]
    )
    cases.extend(
        pytest.param(
            ("trusted_inputs", dataset, name),
            "runtime-self-report" if value is None else "tampered",
            id=f"trust-{dataset}-{name}",
        )
        for dataset in GATE4_INPUTS
        for name, value in GATE4_TRUSTED_INPUT.items()
    )
    cases.extend(
        pytest.param(("boundaries", name), True, id=f"boundary-{name}")
        for name in BOUNDARIES
    )
    return cases


def _set_nested_value(payload: dict, field_path: tuple[str, ...], value) -> None:
    target = payload
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = value


def _install_gate4_synthetic_contract(monkeypatch, runner, tmp_path: Path):
    assert runner.GATE4_FORMAL_OUTPUT_ROOT == Path("D:/xunce/out/path_v2/g4")
    assert runner.GATE4_FORMAL_TEMP_ROOT == Path("D:/xunce/tmp/path_v2_g4")
    assert runner.GATE4_INPUTS == GATE4_INPUTS
    formal_root = (tmp_path / "formal-g4").resolve()
    temp_root = (tmp_path / "attempts").resolve()
    input_root = (tmp_path / "inputs").resolve()
    inputs = {
        "primitive_audit": (input_root / "primitive.jsonl").as_posix(),
        "exact_map_quality": (input_root / "exact.jsonl").as_posix(),
        "standard_episodes": (input_root / "standard.jsonl").as_posix(),
    }
    monkeypatch.setattr(runner, "GATE4_FORMAL_OUTPUT_ROOT", formal_root)
    monkeypatch.setattr(runner, "GATE4_FORMAL_TEMP_ROOT", temp_root)
    monkeypatch.setattr(runner, "GATE4_INPUTS", dict(inputs))
    config = _gate4_payload(
        formal_output_root=formal_root.as_posix(),
        temp_root=temp_root.as_posix(),
        inputs=inputs,
    )
    return config, formal_root, inputs


def _install_gate4_green_code_mocks(
    monkeypatch,
    runner,
    events: list[str],
    *,
    preflight_status: str = "passed",
    focused_status: str = "passed",
    full_status: str = "passed",
    contract_status: str = "passed",
    postflight_ok: bool = True,
) -> None:
    monkeypatch.setattr(
        runner,
        "_gate4_preflight",
        lambda config, repo_root: events.append("preflight")
        or {"status": preflight_status},
    )

    pytest_call_count = 0

    def fake_pytest(*, python, repo_root, targets, junit_path, basetemp, env):
        nonlocal pytest_call_count
        expected_targets = (
            tuple(GATE4_FOCUSED_TARGETS) if pytest_call_count == 0 else ("tests",)
        )
        assert tuple(targets) == expected_targets
        assert Path(python).resolve() == Path("D:/conda_envs/lunar-explorer/python.exe").resolve()
        assert Path(repo_root).resolve() == REPO_ROOT.resolve()
        assert Path(junit_path).parent == Path(basetemp).parent
        assert Path(junit_path).parent.name.startswith("attempt-")
        assert Path(basetemp).name == (
            "focused-basetemp" if pytest_call_count == 0 else "full-basetemp"
        )
        assert Path(junit_path).name == (
            "focused.junit.xml" if pytest_call_count == 0 else "full.junit.xml"
        )
        assert env["PYTHONNOUSERSITE"] == "1"
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        assert Path(env["PYTHONPATH"]).resolve() == (
            REPO_ROOT / "path-planner" / "src"
        ).resolve()
        assert Path(env["TEMP"]).resolve() == Path(junit_path).parent.resolve()
        assert env["TMP"] == env["TEMP"]
        assert Path(env["MPLCONFIGDIR"]).resolve() == (
            Path(junit_path).parent / "mpl"
        ).resolve()
        suite = "focused" if pytest_call_count == 0 else "full"
        pytest_call_count += 1
        events.append(suite)
        status = focused_status if suite == "focused" else full_status
        return {"command": ["pytest", *targets], "returncode": 0 if status == "passed" else 1}

    monkeypatch.setattr(runner, "_run_pytest", fake_pytest)
    monkeypatch.setattr(
        runner,
        "_audit_focused_junit",
        lambda path: {"status": focused_status, "passed": 1, "skipped": 0},
    )
    monkeypatch.setattr(
        runner,
        "_audit_gate4_full_junit",
        lambda *args, **kwargs: {
            "schema_version": "xunce-path-v2-gate4-full-junit-audit/v1",
            "status": full_status,
            "total": {"passed": 200, "skipped": 17, "failures": 0, "errors": 0},
            "baseline_not_reduced": full_status == "passed",
            "v2": {"passed": 27, "skipped": 0, "failures": 0, "errors": 0},
        },
    )
    monkeypatch.setattr(
        runner,
        "_audit_gate4_benchmark_contract",
        lambda repo_root: events.append("benchmark-contract")
        or {
            **GATE4_BENCHMARK_CONTRACT_AUDIT,
            "status": contract_status,
        },
    )
    monkeypatch.setattr(
        runner,
        "_gate4_postflight",
        lambda config, repo_root: events.append("postflight") or {"status": "passed"},
    )
    monkeypatch.setattr(
        runner,
        "_postflight_matches",
        lambda preflight, postflight: postflight_ok,
    )


@pytest.mark.parametrize(("field_path", "tampered"), _gate4_tampers())
def test_gate4_frozen_config_rejects_every_tamper_before_side_effects(
    tmp_path: Path,
    monkeypatch,
    field_path,
    tampered,
) -> None:
    runner = _runner()
    payload = _gate4_payload()
    _set_nested_value(payload, field_path, tampered)
    config_path = tmp_path / "tampered-gate4.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("Gate 4 side effect occurred before config rejection")

    monkeypatch.setattr(runner, "_gate4_preflight", forbidden)
    monkeypatch.setattr(runner, "_audit_gate4_benchmark_contract", forbidden)
    monkeypatch.setattr(runner, "_gate4_dataset_status", forbidden)
    monkeypatch.setattr(runner, "_assert_gate4_output_root_absent", forbidden)
    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    monkeypatch.setattr(runner.artifact_io, "make_dirs", forbidden)
    monkeypatch.setattr(runner.gate_artifacts, "write_gate_artifacts_atomically", forbidden)

    with pytest.raises(ValueError, match="frozen"):
        runner.run_gate_benchmark(
            config_path,
            tmp_path / "out",
            REPO_ROOT,
            execute_tests=False,
        )
    assert not (tmp_path / "out").exists()


def test_gate4_schema_tamper_is_rejected_without_requiring_frozen_wording(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    payload = _gate4_payload()
    payload["schema_version"] = "xunce-path-v2-gate4-legged/v2"
    config_path = tmp_path / "gate4-unknown-schema.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("unknown Gate 4 schema reached a side effect")

    monkeypatch.setattr(runner, "_gate4_preflight", forbidden)
    monkeypatch.setattr(runner, "_audit_gate4_benchmark_contract", forbidden)
    monkeypatch.setattr(runner, "_gate4_dataset_status", forbidden)
    monkeypatch.setattr(runner, "_assert_gate4_output_root_absent", forbidden)
    monkeypatch.setattr(runner.artifact_io, "make_dirs", forbidden)
    monkeypatch.setattr(runner.gate_artifacts, "write_gate_artifacts_atomically", forbidden)
    with pytest.raises(ValueError):
        runner.run_gate_benchmark(
            config_path, tmp_path / "out", REPO_ROOT, execute_tests=False
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("tampered_field", "tampered_value"),
    [
        pytest.param(
            "schema_version",
            "xunce-path-v2-gate4-legged/v2",
            id="stage-match",
        ),
        pytest.param(
            "stage_id",
            "xunce-path-v2-gate4-legged-other",
            id="schema-match",
        ),
    ],
)
def test_gate4_like_tampered_config_precedes_illegal_repo_output_root(
    tmp_path: Path,
    monkeypatch,
    tampered_field: str,
    tampered_value: str,
) -> None:
    runner = _runner()
    payload = _gate4_payload()
    payload[tampered_field] = tampered_value
    config_path = tmp_path / f"tampered-{tampered_field}.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("Gate 4 root handling ran before frozen config rejection")

    monkeypatch.setattr(runner, "_validate_gate4_output_mode", forbidden)
    monkeypatch.setattr(runner, "_assert_gate4_output_root_absent", forbidden)
    monkeypatch.setattr(runner, "validate_output_root", forbidden)
    monkeypatch.setattr(
        runner.gate_artifacts, "write_gate_artifacts_atomically", forbidden
    )

    with pytest.raises(ValueError, match="frozen"):
        runner.run_gate_benchmark(
            config_path,
            REPO_ROOT / "illegal-gate4-output",
            REPO_ROOT,
            execute_tests=False,
        )


def test_gate4_rejects_joint_runtime_attempt_to_approve_every_null_anchor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    payload = _gate4_payload()
    payload["dataset_contract"]["accepts_formal_inputs"] = True
    for dataset, anchors in payload["trusted_inputs"].items():
        for key, value in anchors.items():
            if value is None:
                anchors[key] = "a" * 64 if key.endswith("sha256") else f"approved-{dataset}-{key}"
    config_path = tmp_path / "gate4-fake-approval.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("self-approved Gate 4 input reached a side effect")

    monkeypatch.setattr(runner, "_gate4_preflight", forbidden)
    monkeypatch.setattr(runner, "_audit_gate4_benchmark_contract", forbidden)
    monkeypatch.setattr(runner, "_gate4_dataset_status", forbidden)
    monkeypatch.setattr(runner, "_assert_gate4_output_root_absent", forbidden)
    monkeypatch.setattr(runner.artifact_io, "make_dirs", forbidden)
    monkeypatch.setattr(runner.gate_artifacts, "write_gate_artifacts_atomically", forbidden)
    with pytest.raises(ValueError, match="frozen"):
        runner.run_gate_benchmark(
            config_path, tmp_path / "out", REPO_ROOT, execute_tests=False
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_gate4_config_rejects_missing_or_extra_top_level_key(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
) -> None:
    runner = _runner()
    payload = _gate4_payload()
    if mutation == "missing":
        payload.pop("dataset_contract")
    else:
        payload["runtime_trust_override"] = True
    config_path = tmp_path / "bad-keys.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        runner.gate_artifacts,
        "write_gate_artifacts_atomically",
        lambda **kwargs: pytest.fail("malformed Gate 4 config wrote artifacts"),
    )
    with pytest.raises(ValueError, match="top-level keys"):
        runner.run_gate_benchmark(config_path, tmp_path / "out", REPO_ROOT, execute_tests=False)


def test_checked_in_gate4_config_and_registry_match_exact_frozen_contract(tmp_path: Path) -> None:
    checked_in = json.loads(
        (REPO_ROOT / "configs" / "xunce_path_v2_gate4_legged_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert checked_in == _gate4_payload()
    registry = json.loads(
        (REPO_ROOT / "configs" / "stage_registry.json").read_text(encoding="utf-8")
    )
    assert registry["stages"]["xunce-path-v2-gate4-legged"] == {
        "runner": "scripts/run_xunce_path_v2_gate_benchmark.py",
        "default_config": "configs/xunce_path_v2_gate4_legged_v1.json",
        "default_output_root": "D:/xunce/out/path_v2/g4",
        "args": [
            "--config",
            "{config}",
            "--output-root",
            "{output_root}",
            "--repo-root",
            "{repo_root}",
        ],
    }


def test_gate4_dry_run_is_atomic_exact_eight_byte_stable_and_never_inspects_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()

    def forbidden(*args, **kwargs):
        raise AssertionError("Gate 4 dry run performed code audit or input inspection")

    monkeypatch.setattr(runner, "_gate4_preflight", forbidden)
    monkeypatch.setattr(runner, "_run_pytest", forbidden)
    monkeypatch.setattr(runner, "_audit_gate4_benchmark_contract", forbidden)
    monkeypatch.setattr(runner, "_gate4_dataset_status", forbidden)
    monkeypatch.setattr(runner.artifact_io, "path_is_file", forbidden)
    outputs = [tmp_path / "dry-a", tmp_path / "dry-b"]
    summaries = [
        runner.run_gate_benchmark(
            _gate4_config(tmp_path), output, REPO_ROOT, execute_tests=False
        )
        for output in outputs
    ]

    for summary, output in zip(summaries, outputs, strict=True):
        assert summary["status"] == "dry_run"
        assert summary["next_required_change"] == "execute_gate4_legged_evidence"
        assert summary["blocking_reasons"] == []
        assert summary["formal_metrics_status"] == "not_evaluated"
        assert summary["thresholds"] == GATE4_THRESHOLDS
        assert summary["capability_disclosure"] == GATE4_CAPABILITY_DISCLOSURE
        assert all(item["status"] == "not_run" for item in summary["datasets"].values())
        assert all(item["content_read"] is False for item in summary["datasets"].values())
        assert all(
            item["trusted_input"] == GATE4_TRUSTED_INPUT
            for item in summary["datasets"].values()
        )
        assert {path.name for path in output.iterdir()} == CANONICAL_ARTIFACTS
        routing = json.loads((output / "routing.json").read_text(encoding="utf-8"))
        review = json.loads((output / "review.json").read_text(encoding="utf-8"))
        assert routing["capability_disclosure"] == GATE4_CAPABILITY_DISCLOSURE
        assert review["capability_disclosure"] == GATE4_CAPABILITY_DISCLOSURE
        assert review["formal_metrics_status"] == "not_evaluated"
        phases = [
            json.loads(line)
            for line in (output / "phase-state.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert phases == [
            {"phase": phase, "status": "not_run"} for phase in GATE4_PHASES
        ]
        report = (output / "report.md").read_text(encoding="utf-8")
        assert "simulation_proxy_static_crawl/v1" in report
        assert "dynamic_gait_claimed=false" in report
        assert "real_robot_stability_claimed=false" in report
        assert "v1 仍为默认，v2 仍为 opt-in" in report
        assert "不发布 checkpoint" in report
        assert "不替换 default policy" in report
        assert "不连接 executor" in report
        assert "不启动 canary" in report
        assert list(output.parent.glob(f".{output.name}.staging-*")) == []
    assert {
        path.name: path.read_bytes() for path in outputs[0].iterdir()
    } == {path.name: path.read_bytes() for path in outputs[1].iterdir()}


@pytest.mark.parametrize(
    ("execute_tests", "use_formal_root"),
    [(False, True), (True, False)],
    ids=["dry-run-cannot-occupy-formal", "formal-requires-exact-root"],
)
def test_gate4_public_mode_rejects_wrong_root_before_existence_or_writer(
    tmp_path: Path,
    monkeypatch,
    execute_tests: bool,
    use_formal_root: bool,
) -> None:
    runner = _runner()

    def forbidden(*args, **kwargs):
        raise AssertionError("wrong Gate 4 mode/root reached a side effect")

    monkeypatch.setattr(runner, "_assert_gate4_output_root_absent", forbidden)
    monkeypatch.setattr(runner.gate_artifacts, "write_gate_artifacts_atomically", forbidden)
    output = Path("D:/xunce/out/path_v2/g4") if use_formal_root else tmp_path / "not-formal"
    with pytest.raises(ValueError, match="formal_output_root"):
        runner.run_gate_benchmark(
            _gate4_config(tmp_path), output, REPO_ROOT, execute_tests=execute_tests
        )
    assert not (tmp_path / "not-formal").exists()


def _windows_safe_lexical_test_path(path: Path) -> str:
    absolute = os.path.abspath(os.fspath(path))
    if os.name == "nt":
        if absolute.startswith("\\\\?\\"):
            return absolute
        if absolute.startswith("\\\\"):
            return "\\\\?\\UNC\\" + absolute[2:]
        return "\\\\?\\" + absolute
    return absolute


def test_gate4_public_fresh_root_checks_raw_lexical_path_even_after_mode_resolve(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    raw_output = tmp_path / "lexical-parent" / ".." / "dangling-gate4"
    expected_lexical = _windows_safe_lexical_test_path(raw_output)
    original_resolve = Path.resolve
    resolved_alias = tmp_path / "resolved-target-that-does-not-exist"
    lexists_paths: list[str] = []

    def tracked_resolve(path: Path, *args, **kwargs):
        if str(path) == str(raw_output):
            return resolved_alias
        return original_resolve(path, *args, **kwargs)

    def fake_lexists(path) -> bool:
        lexists_paths.append(str(path))
        assert str(path) == expected_lexical
        return True

    monkeypatch.setattr(Path, "resolve", tracked_resolve)
    monkeypatch.setattr(runner.os.path, "lexists", fake_lexists)
    monkeypatch.setattr(
        runner.gate_artifacts,
        "write_gate_artifacts_atomically",
        lambda **kwargs: pytest.fail("existing lexical Gate 4 root reached writer"),
    )

    with pytest.raises(RuntimeError, match="Gate 4 output_root already exists"):
        runner.run_gate_benchmark(
            _gate4_config(tmp_path), raw_output, REPO_ROOT, execute_tests=False
        )
    assert lexists_paths == [expected_lexical]


@pytest.mark.parametrize(
    "root_kind",
    ["empty", "file", "old-artifacts", "extra-child"],
)
def test_gate4_formal_root_must_be_fresh_before_any_code_work(
    tmp_path: Path,
    monkeypatch,
    root_kind: str,
) -> None:
    runner = _runner()
    config, formal_root, _ = _install_gate4_synthetic_contract(monkeypatch, runner, tmp_path)
    config_path = tmp_path / "existing-root-gate4.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    if root_kind == "empty":
        formal_root.mkdir()
    elif root_kind == "file":
        formal_root.write_bytes(b"preserve")
    elif root_kind == "old-artifacts":
        runner.gate_artifacts.write_gate_artifacts(
            output_root=formal_root,
            config={"old": "config"},
            summary={"status": "blocked", "old": True},
            routing={"status": "blocked", "route": "old-route"},
            rows=[{"status": "blocked", "old": True}],
            phases=[{"phase": "old", "status": "completed"}],
            review={"status": "blocked", "old": True},
            report="# old blocked snapshot\n",
        )
    else:
        child = formal_root / "extra-child"
        child.mkdir(parents=True)
        (child / "marker.bin").write_bytes(b"preserve-child")

    def snapshot() -> tuple[tuple[str, str, bytes | None], ...]:
        if formal_root.is_file():
            return (("file", ".", formal_root.read_bytes()),)
        records: list[tuple[str, str, bytes | None]] = [("directory", ".", None)]
        for path in sorted(formal_root.rglob("*"), key=lambda item: item.as_posix()):
            relative = path.relative_to(formal_root).as_posix()
            records.append(
                ("directory", relative, None)
                if path.is_dir()
                else ("file", relative, path.read_bytes())
            )
        return tuple(records)

    before = snapshot()

    def forbidden(*args, **kwargs):
        raise AssertionError("Gate 4 code work ran before fresh-root rejection")

    monkeypatch.setattr(runner, "_gate4_preflight", forbidden)
    monkeypatch.setattr(runner, "_run_pytest", forbidden)
    monkeypatch.setattr(runner.gate_artifacts, "write_gate_artifacts_atomically", forbidden)
    with pytest.raises(RuntimeError, match="Gate 4 output_root already exists"):
        runner.run_gate_benchmark(
            config_path=config_path,
            output_root=formal_root,
            repo_root=REPO_ROOT,
            execute_tests=True,
        )
    assert snapshot() == before


def _run_captured_gate4_formal(
    tmp_path: Path,
    monkeypatch,
    *,
    present: bool = False,
    present_datasets: set[str] | None = None,
    guard_input_reads: bool = False,
):
    runner = _runner()
    config, formal_root, inputs = _install_gate4_synthetic_contract(
        monkeypatch, runner, tmp_path
    )
    config_path = tmp_path / "synthetic-gate4.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    selected_datasets = (
        set(inputs) if present else set() if present_datasets is None else present_datasets
    )
    for index, (dataset, path_string) in enumerate(inputs.items()):
        if dataset in selected_datasets:
            path = Path(path_string)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\xffmalicious self-reported pass" + bytes([index]))
    if guard_input_reads:
        input_paths = {Path(path).resolve() for path in inputs.values()}

        def guard(method_name, original):
            def guarded(path, *args, **kwargs):
                if Path(path).resolve() in input_paths:
                    raise AssertionError(
                        f"Gate 4B used {method_name} on unapproved formal input"
                    )
                return original(path, *args, **kwargs)

            return guarded

        monkeypatch.setattr(
            runner.artifact_io,
            "read_bytes",
            guard("artifact_io.read_bytes", runner.artifact_io.read_bytes),
        )
        monkeypatch.setattr(
            runner.artifact_io,
            "read_json",
            guard("artifact_io.read_json", runner.artifact_io.read_json),
        )
        monkeypatch.setattr(
            runner.artifact_io,
            "read_jsonl",
            guard("artifact_io.read_jsonl", runner.artifact_io.read_jsonl),
        )
        monkeypatch.setattr(
            Path,
            "read_bytes",
            guard("Path.read_bytes", Path.read_bytes),
        )
        monkeypatch.setattr(
            Path,
            "read_text",
            guard("Path.read_text", Path.read_text),
        )
        monkeypatch.setattr(Path, "open", guard("Path.open", Path.open))
        original_open = open

        def guarded_open(file, *args, **kwargs):
            try:
                path = Path(file).resolve()
            except TypeError:
                return original_open(file, *args, **kwargs)
            if path in input_paths:
                raise AssertionError("Gate 4B used builtins.open on unapproved formal input")
            return original_open(file, *args, **kwargs)

        monkeypatch.setattr("builtins.open", guarded_open)
    events: list[str] = []
    original_is_file = runner.artifact_io.path_is_file
    input_names = {str(Path(path).resolve()): name for name, path in inputs.items()}

    def tracked_is_file(path):
        events.append(f"presence:{input_names[str(Path(path).resolve())]}")
        return original_is_file(path)

    monkeypatch.setattr(runner.artifact_io, "path_is_file", tracked_is_file)
    _install_gate4_green_code_mocks(monkeypatch, runner, events)
    captured: dict = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return {"schema_version": "xunce-path-v2-gate-manifest/v1", "artifact_count": 7}

    monkeypatch.setattr(runner.gate_artifacts, "write_gate_artifacts_atomically", capture)
    summary = runner.run_gate_benchmark(
        config_path=config_path,
        output_root=formal_root,
        repo_root=REPO_ROOT,
        execute_tests=True,
    )
    return runner, summary, captured, events, inputs


def _assert_gate4_blocked_artifacts(
    *,
    runner,
    summary: dict,
    captured: dict,
    inputs: dict[str, str],
    dataset_statuses: dict[str, str],
) -> None:
    assert summary["status"] == "blocked"
    assert summary["next_required_change"] == GATE4_BLOCKERS[0]
    assert summary["blocking_reasons"] == GATE4_BLOCKERS
    assert summary["formal_metrics_status"] == "not_evaluated"
    assert captured["review"]["formal_metrics_status"] == "not_evaluated"
    assert summary["benchmark_contract"] == GATE4_BENCHMARK_CONTRACT_AUDIT
    assert captured["review"]["benchmark_contract"] == GATE4_BENCHMARK_CONTRACT_AUDIT
    assert json.loads(json.dumps(summary["benchmark_contract"])) == (
        GATE4_BENCHMARK_CONTRACT_AUDIT
    )
    assert json.loads(json.dumps(captured["review"]["benchmark_contract"])) == (
        GATE4_BENCHMARK_CONTRACT_AUDIT
    )
    assert {
        name: summary["datasets"][name]["status"] for name in GATE4_INPUTS
    } == dataset_statuses
    assert all(
        summary["datasets"][name]["content_read"] is False for name in GATE4_INPUTS
    )
    formal_rows = [
        row for row in captured["rows"] if row.get("suite") == "formal-input"
    ]
    assert formal_rows == [
        {
            "suite": "formal-input",
            "dataset": dataset,
            "path": inputs[dataset],
            "status": dataset_statuses[dataset],
            "content_read": False,
            "reason": blocker,
        }
        for dataset, blocker in zip(GATE4_INPUTS, GATE4_BLOCKERS, strict=True)
    ]
    for row in captured["rows"]:
        assert row.get("suite") not in {"metric", "formal-metric", "formal_metric"}
        assert row.get("check") not in {"metric", "formal_metric", "formal-metric"}
        assert not any(key.startswith("actual_") for key in row)
    assert captured["routing"]["route"] == GATE4_BLOCKERS[0]
    assert captured["routing"]["blocking_reasons"] == GATE4_BLOCKERS
    assert captured["routing"]["route"] != (
        "implement_path_v2_lunar_ballistics_and_hopper_proxy_profile"
    )
    assert "pass_route" not in json.dumps(captured["routing"], sort_keys=True)
    report = captured["report"]
    assert "formal_metrics_status=not_evaluated" in report
    assert "actual_" not in report.lower()
    assert "=N/A" not in report.upper()
    assert ": N/A" not in report.upper()
    assert runner.status_exit_code(summary["status"]) == 0


def test_gate4_public_exact_formal_root_blocks_after_all_code_audits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner, summary, captured, events, inputs = _run_captured_gate4_formal(
        tmp_path, monkeypatch
    )
    assert events == [
        "preflight",
        "focused",
        "full",
        "benchmark-contract",
        "presence:primitive_audit",
        "presence:exact_map_quality",
        "presence:standard_episodes",
        "postflight",
    ]
    _assert_gate4_blocked_artifacts(
        runner=runner,
        summary=summary,
        captured=captured,
        inputs=inputs,
        dataset_statuses={name: "missing" for name in GATE4_INPUTS},
    )
    assert "implement_path_v2_lunar_ballistics_and_hopper_proxy_profile" not in json.dumps(
        {key: value for key, value in summary.items() if key != "config"}
    )
    assert [phase["phase"] for phase in captured["phases"]] == GATE4_PHASES


def test_gate4_present_unapproved_files_are_untrusted_and_never_read_or_parsed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()

    def forbidden(*args, **kwargs):
        raise AssertionError("Gate 4B read or parsed unapproved formal input")

    monkeypatch.setattr(runner, "_load_gate2_dataset", forbidden)
    runner, summary, captured, _, inputs = _run_captured_gate4_formal(
        tmp_path, monkeypatch, present=True, guard_input_reads=True
    )

    _assert_gate4_blocked_artifacts(
        runner=runner,
        summary=summary,
        captured=captured,
        inputs=inputs,
        dataset_statuses={name: "untrusted" for name in GATE4_INPUTS},
    )
    assert all(
        item["trusted_input"] == GATE4_TRUSTED_INPUT
        for item in summary["datasets"].values()
    )
    public = json.dumps(
        {
            "summary": summary,
            "routing": captured["routing"],
            "review": captured["review"],
            "rows": captured["rows"],
        },
        sort_keys=True,
    )
    for forbidden_key in (
        "actual_input_sha256",
        "actual_hash",
        "actual_rows",
        "metric_summary",
        '"metrics"',
        '"provenance"',
    ):
        assert forbidden_key not in public
    assert "passed" not in {
        row.get("status") for row in captured["rows"] if row.get("suite") == "formal-input"
    }


@pytest.mark.parametrize("present_dataset", list(GATE4_INPUTS))
def test_gate4_mixed_presence_keeps_all_blockers_and_never_reads_content(
    tmp_path: Path,
    monkeypatch,
    present_dataset: str,
) -> None:
    runner = _runner()
    monkeypatch.setattr(
        runner,
        "_load_gate2_dataset",
        lambda *args, **kwargs: pytest.fail("Gate 4B reused the Gate 2 parser"),
    )
    runner, summary, captured, _, inputs = _run_captured_gate4_formal(
        tmp_path,
        monkeypatch,
        present_datasets={present_dataset},
        guard_input_reads=True,
    )
    statuses = {
        name: "untrusted" if name == present_dataset else "missing"
        for name in GATE4_INPUTS
    }
    _assert_gate4_blocked_artifacts(
        runner=runner,
        summary=summary,
        captured=captured,
        inputs=inputs,
        dataset_statuses=statuses,
    )
    assert summary["next_required_change"] == GATE4_BLOCKERS[0]
    assert summary["blocking_reasons"] == GATE4_BLOCKERS
    assert runner.status_exit_code(summary["status"]) == 0


def test_gate4_benchmark_contract_failure_precedes_missing_input_blockers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    config, formal_root, _ = _install_gate4_synthetic_contract(monkeypatch, runner, tmp_path)
    events: list[str] = []
    _install_gate4_green_code_mocks(
        monkeypatch, runner, events, contract_status="failed"
    )
    monkeypatch.setattr(
        runner.artifact_io,
        "path_is_file",
        lambda path: pytest.fail("input presence checked after benchmark contract failure"),
    )
    captured: dict = {}
    monkeypatch.setattr(
        runner.gate_artifacts,
        "write_gate_artifacts_atomically",
        lambda **kwargs: captured.update(kwargs) or {},
    )
    summary = runner._run_gate4_benchmark(
        config=config, output_root=formal_root, repo_root=REPO_ROOT, execute_tests=True
    )
    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_gate4_benchmark_contract"
    assert summary["blocking_reasons"] == []
    assert all(item["status"] == "not_run" for item in summary["datasets"].values())
    assert runner.status_exit_code(summary["status"]) == 1


def _gate4_fake_benchmark_module(forbidden):
    values = {
        "__file__": str(
            REPO_ROOT / "path-planner/src/path_planner/v2/benchmark.py"
        ),
        "HARD_TIMEOUT_MS_V2": 2000.0,
    }
    values.update(
        {
            name: type(
                name,
                (),
                {
                    "__module__": "path_planner.v2.benchmark",
                    "__init__": forbidden,
                },
            )
            for name in GATE4_BENCHMARK_ROW_CLASSES
        }
    )
    values.update({name: forbidden for name in GATE4_BENCHMARK_AGGREGATES})
    return SimpleNamespace(**values)


def test_gate4_benchmark_contract_audit_accepts_exact_public_api_without_invocation(
    monkeypatch,
) -> None:
    runner = _runner()

    def forbidden(*args, **kwargs):
        raise AssertionError("Gate 4B audit executed a row, aggregate, provider, or planner")

    module = _gate4_fake_benchmark_module(forbidden)
    imports: list[str] = []

    def import_only_benchmark(name):
        imports.append(name)
        if name != "path_planner.v2.benchmark":
            raise AssertionError(f"Gate 4B audit imported forbidden module {name}")
        return module

    monkeypatch.setattr(runner.importlib, "import_module", import_only_benchmark)
    for name in ("plan_v2", "run_legged_provider", "run_search_a2", "run_final_9a2"):
        monkeypatch.setattr(runner, name, forbidden, raising=False)

    audit = runner._audit_gate4_benchmark_contract(REPO_ROOT)

    assert audit["status"] == "passed"
    assert imports == ["path_planner.v2.benchmark"]


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param("origin", id="module-origin"),
        pytest.param("timeout", id="hard-timeout"),
        *[
            pytest.param(f"row:{name}", id=f"row-{name}")
            for name in GATE4_BENCHMARK_ROW_CLASSES
        ],
        *[
            pytest.param(f"aggregate:{name}", id=f"aggregate-{name}")
            for name in GATE4_BENCHMARK_AGGREGATES
        ],
    ],
)
def test_gate4_benchmark_contract_audit_fails_closed_on_each_api_drift(
    monkeypatch,
    mutation: str,
) -> None:
    runner = _runner()

    def forbidden(*args, **kwargs):
        raise AssertionError("Gate 4B benchmark audit invoked typed evidence code")

    module = _gate4_fake_benchmark_module(forbidden)
    if mutation == "origin":
        module.__file__ = "D:/other/benchmark.py"
    elif mutation == "timeout":
        module.HARD_TIMEOUT_MS_V2 = 1999.0
    else:
        _, name = mutation.split(":", 1)
        setattr(module, name, None)
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: module)
    audit = runner._audit_gate4_benchmark_contract(REPO_ROOT)
    assert audit["status"] == "failed"
    assert audit["repair_route"] == "repair_gate4_benchmark_contract"


@pytest.mark.parametrize(
    ("failure", "route"),
    [
        ("preflight", "restore_gate4_runtime_isolation"),
        ("focused", "restore_gate4_focused_contracts"),
        ("full", "restore_path_planner_v1_regression"),
        ("postflight", "restore_gate4_runtime_isolation"),
    ],
)
def test_gate4_code_failures_take_precedence_over_external_blockers(
    tmp_path: Path,
    monkeypatch,
    failure: str,
    route: str,
) -> None:
    runner = _runner()
    config, formal_root, _ = _install_gate4_synthetic_contract(monkeypatch, runner, tmp_path)
    events: list[str] = []
    _install_gate4_green_code_mocks(
        monkeypatch,
        runner,
        events,
        preflight_status="failed" if failure == "preflight" else "passed",
        focused_status="failed" if failure == "focused" else "passed",
        full_status="failed" if failure == "full" else "passed",
        postflight_ok=failure != "postflight",
    )
    if failure != "postflight":
        monkeypatch.setattr(
            runner,
            "_audit_gate4_benchmark_contract",
            lambda repo_root: pytest.fail(
                f"benchmark audit ran after {failure} code failure"
            ),
        )
        monkeypatch.setattr(
            runner.artifact_io,
            "path_is_file",
            lambda path: pytest.fail(f"input presence checked after {failure} failure"),
        )
    else:
        original_is_file = runner.artifact_io.path_is_file
        input_names = {
            str(Path(path).resolve()): name for name, path in config["inputs"].items()
        }

        def tracked_is_file(path):
            events.append(f"presence:{input_names[str(Path(path).resolve())]}")
            return original_is_file(path)

        monkeypatch.setattr(runner.artifact_io, "path_is_file", tracked_is_file)
    captured: dict = {}
    monkeypatch.setattr(
        runner.gate_artifacts,
        "write_gate_artifacts_atomically",
        lambda **kwargs: captured.update(kwargs) or {},
    )
    summary = runner._run_gate4_benchmark(
        config=config, output_root=formal_root, repo_root=REPO_ROOT, execute_tests=True
    )
    assert summary["status"] == "failed"
    assert summary["next_required_change"] == route
    assert summary["blocking_reasons"] == []
    if failure != "postflight":
        assert all(item["status"] == "not_run" for item in summary["datasets"].values())
        assert "benchmark-contract" not in events
        assert not any(event.startswith("presence:") for event in events)
    if failure == "preflight":
        assert events == ["preflight", "postflight"]
    elif failure in {"focused", "full"}:
        assert events == ["preflight", "focused", "full", "postflight"]
    else:
        assert events == [
            "preflight",
            "focused",
            "full",
            "benchmark-contract",
            "presence:primitive_audit",
            "presence:exact_map_quality",
            "presence:standard_episodes",
            "postflight",
        ]


@pytest.mark.parametrize(
    ("failure_mode", "message"),
    [
        ("staging-write", "staging write failed"),
        ("staging-validation", "staging validation failed"),
        ("rename", "publish rename failed"),
    ],
)
def test_gate4_entry_preserves_failed_staging_and_never_exposes_canonical_root(
    tmp_path: Path,
    monkeypatch,
    failure_mode: str,
    message: str,
) -> None:
    runner = _runner()
    output_root = tmp_path / f"gate4-{failure_mode}"
    if failure_mode == "staging-write":
        def fail_after_partial_write(*, output_root, **kwargs):
            Path(output_root).mkdir(parents=True, exist_ok=True)
            (Path(output_root) / "config.json").write_bytes(b"partial")
            raise OSError("injected Gate 4 staging write failure")

        monkeypatch.setattr(
            runner.gate_artifacts, "write_gate_artifacts", fail_after_partial_write
        )
    elif failure_mode == "staging-validation":
        monkeypatch.setattr(
            runner.gate_artifacts,
            "_validate_staged_gate_artifacts",
            lambda path: (_ for _ in ()).throw(
                ValueError("injected Gate 4 staging validation failure")
            ),
        )
    else:
        monkeypatch.setattr(
            runner.gate_artifacts.os,
            "rename",
            lambda *args: (_ for _ in ()).throw(
                OSError("injected Gate 4 publish rename failure")
            ),
        )
    with pytest.raises(RuntimeError, match=message):
        runner.run_gate_benchmark(
            _gate4_config(tmp_path), output_root, REPO_ROOT, execute_tests=False
        )
    assert not output_root.exists()
    staging_roots = list(output_root.parent.glob(f".{output_root.name}.staging-*"))
    assert len(staging_roots) == 1
    assert staging_roots[0].is_dir()
    if failure_mode != "staging-write":
        assert {path.name for path in staging_roots[0].iterdir()} == CANONICAL_ARTIFACTS


def test_gate4_full_audit_preserves_gate0_nodeids_and_only_allows_new_v2_passes(
    tmp_path: Path,
) -> None:
    runner = _runner()
    baseline = tmp_path / "baseline.xml"
    current = tmp_path / "current.xml"
    missing = tmp_path / "missing.xml"
    baseline_cases = _full_cases()[:173]
    current_cases = baseline_cases + [
        (f"tests/test_v2_gate4.py::test_new_{index}", "passed", "")
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
    green = runner._audit_gate4_full_junit(current, **kwargs)
    drifted = runner._audit_gate4_full_junit(missing, **kwargs)
    assert green["schema_version"] == "xunce-path-v2-gate4-full-junit-audit/v1"
    assert green["status"] == "passed"
    assert green["baseline_not_reduced"] is True
    assert green["v2"]["skipped"] == 0
    assert drifted["status"] == "failed"
    assert drifted["missing_baseline_nodeids"]

    regressions = {
        "v2-failure": current_cases
        + [("tests/test_v2_gate4.py::test_failed", "failed", "")],
        "v2-error": current_cases
        + [("tests/test_v2_gate4.py::test_error", "error", "")],
        "v2-skip": current_cases
        + [("tests/test_v2_gate4.py::test_skip", "skipped", "pydrake unavailable")],
    }
    bad_legacy_skip = list(current_cases)
    bad_legacy_skip[156] = (
        bad_legacy_skip[156][0],
        "skipped",
        "numpy unavailable",
    )
    regressions["bad-legacy-skip"] = bad_legacy_skip
    for name, cases in regressions.items():
        path = tmp_path / f"{name}.xml"
        _write_junit(path, cases)
        audit = runner._audit_gate4_full_junit(path, **kwargs)
        assert audit["status"] == "failed", name


GATE5_CONFIG_PATH = REPO_ROOT / "configs" / "xunce_path_v2_gate5_hopper_v1.json"
GATE5_PROFILE_FIELDS = (
    "body_envelope_radius_m",
    "launch_reference_height_m",
    "arc_clearance_margin_m",
    "landing_footprint_radius_m",
    "stop_condition",
    "energy_model",
)
GATE5_BLOCKER = "freeze_hopper_simulation_proxy_profile_parameters"
GATE5_MANIFEST_METADATA = {
    "status": "blocked",
    "execution_class": "blocked_profile_freeze",
    "primary_blocker": GATE5_BLOCKER,
    "formal_evidence_eligible": False,
    "formal_row_count": 0,
    "parameter_set_id": None,
}


def _gate5_config_copy(tmp_path: Path) -> Path:
    payload = json.loads(GATE5_CONFIG_PATH.read_text(encoding="utf-8"))
    target = tmp_path / "gate5.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def test_gate5_checked_config_freezes_exact_null_profile_and_parameter_set() -> None:
    payload = json.loads(GATE5_CONFIG_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "xunce-path-v2-gate5-hopper/v1"
    assert payload["stage_id"] == "xunce-path-v2-gate5-hopper"
    assert payload["execution_class"] == "blocked_profile_freeze"
    assert payload["temp_root"] == "D:/xunce/tmp/path_v2_g5"
    assert payload["parameter_set_id"] is None
    assert tuple(payload["hopper_profile"]) == GATE5_PROFILE_FIELDS
    assert all(payload["hopper_profile"][name] is None for name in GATE5_PROFILE_FIELDS)
    assert payload["dataset_contract"]["accepts_formal_inputs"] is False
    assert payload["formal_evidence_eligible"] is False
    assert payload["boundaries"] == BOUNDARIES


def test_gate5_stage_registry_is_explicitly_blocked_and_ineligible() -> None:
    registry = json.loads(
        (REPO_ROOT / "configs" / "stage_registry.json").read_text(encoding="utf-8")
    )["stages"]["xunce-path-v2-gate5-hopper"]

    assert registry == {
        "runner": "scripts/run_xunce_path_v2_gate_benchmark.py",
        "default_config": "configs/xunce_path_v2_gate5_hopper_v1.json",
        "default_output_root": "D:/xunce/out/path_v2/g5",
        "execution_class": "blocked_profile_freeze",
        "formal_evidence_eligible": False,
        "args": [
            "--config",
            "{config}",
            "--output-root",
            "{output_root}",
            "--repo-root",
            "{repo_root}",
        ],
    }


@pytest.mark.parametrize("execute_tests", [False, True], ids=["dry-run", "controller"])
def test_gate5_profile_freeze_writes_exact_blocked_snapshot_without_pytest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execute_tests: bool,
) -> None:
    runner = _runner()
    output_root = tmp_path / "gate5"
    calls = []

    def forbidden_pytest(**kwargs):
        calls.append(kwargs)
        raise AssertionError("profile freeze must short-circuit pytest")

    monkeypatch.setattr(runner, "_run_pytest", forbidden_pytest)
    summary = runner.run_gate_benchmark(
        GATE5_CONFIG_PATH,
        output_root,
        REPO_ROOT,
        execute_tests=execute_tests,
    )

    assert calls == []
    assert {path.name for path in output_root.iterdir()} == CANONICAL_ARTIFACTS
    assert summary["status"] == "blocked"
    assert summary["execution_class"] == "blocked_profile_freeze"
    assert summary["primary_blocker"] == GATE5_BLOCKER
    assert summary["formal_metrics_status"] == "not_evaluated"
    assert summary["formal_evidence_eligible"] is False
    assert summary["formal_row_count"] == 0
    assert summary["parameter_set_id"] is None
    assert all(summary[name] is False for name in BOUNDARIES)
    assert (output_root / "results.jsonl").read_bytes() == b""
    for name in ("summary.json", "routing.json", "review.json", "manifest.json"):
        payload = json.loads((output_root / name).read_text(encoding="utf-8"))
        for key, value in GATE5_MANIFEST_METADATA.items():
            assert payload[key] == value
    report = (output_root / "report.md").read_text(encoding="utf-8")
    assert all(str(value).lower() in report for value in GATE5_MANIFEST_METADATA.values() if value is not None)
    phases = [json.loads(line) for line in (output_root / "phase-state.jsonl").read_text(encoding="utf-8").splitlines()]
    assert phases == [
        {"phase": "profile-freeze", "status": "blocked"},
        {"phase": "pytest", "status": "not_run_due_to_profile_freeze"},
    ]


def test_gate5_rejects_any_non_null_default_before_output(tmp_path: Path) -> None:
    runner = _runner()
    original = json.loads(GATE5_CONFIG_PATH.read_text(encoding="utf-8"))
    mutations = [("parameter_set_id", "fixture/v1")]
    mutations.extend((f"hopper_profile.{name}", 1.0) for name in GATE5_PROFILE_FIELDS)
    for index, (field, value) in enumerate(mutations):
        payload = deepcopy(original)
        if field.startswith("hopper_profile."):
            payload["hopper_profile"][field.split(".", 1)[1]] = value
        else:
            payload[field] = value
        config_path = tmp_path / f"bad-{index}.json"
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        output_root = tmp_path / f"out-{index}"
        with pytest.raises(ValueError, match="Gate 5 config contract"):
            runner.run_gate_benchmark(
                config_path, output_root, REPO_ROOT, execute_tests=False
            )
        assert not output_root.exists()


def test_gate5_pytest_envelope_constructor_is_exact(tmp_path: Path) -> None:
    runner = _runner()
    attempt_root = Path("D:/xunce/tmp/path_v2_g5/formal/attempt-test")

    invocation = runner._build_gate5_pytest_invocation(REPO_ROOT, attempt_root)

    assert invocation["cwd"] == REPO_ROOT / "path-planner"
    assert invocation["env"] == {
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": str(REPO_ROOT / "path-planner" / "src"),
        "TEMP": str(attempt_root),
        "TMP": str(attempt_root),
        "MPLCONFIGDIR": str(attempt_root / "mpl"),
    }
    assert invocation["argv"][:5] == [
        "D:/conda_envs/lunar-explorer/python.exe",
        "-m",
        "pytest",
        "-o",
        "addopts=",
    ]
    assert invocation["argv"][5:] == [
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(attempt_root / "basetemp"),
        "--junitxml",
        str(attempt_root / "gate5.junit.xml"),
    ]
