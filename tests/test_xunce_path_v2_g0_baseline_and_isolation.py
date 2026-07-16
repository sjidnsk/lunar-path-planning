from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


EXPECTED_BASE_COMMIT = "b635740ee021258ef31811ec87c60add839fc5f9"
BOUNDARY_FIELDS = {
    "publishes_checkpoint": False,
    "replaces_default_policy": False,
    "connects_real_executor": False,
    "starts_online_canary": False,
}
PPO_FAILURE_NODEIDS = (
    "tests/ppo_highres_frontier/test_stage1_smoke_env.py::"
    "test_frontier_recommended_theta_points_to_observed_unknown_side[single-side]",
    "tests/ppo_highres_frontier/test_stage1_smoke_env.py::"
    "test_frontier_recommended_theta_points_to_observed_unknown_side[diagonal]",
    "tests/ppo_highres_frontier/test_stage1_smoke_env.py::"
    "test_frontier_recommended_theta_points_to_observed_unknown_side[multi-neighbour-vector]",
    "tests/ppo_highres_frontier/test_stage1_smoke_env.py::"
    "test_frontier_recommended_theta_points_to_observed_unknown_side[symmetric-travel-fallback]",
    "tests/ppo_highres_frontier/test_stage1_smoke_env.py::"
    "test_env_and_workflow_rule_policy_share_one_conservative_selector",
    "tests/ppo_highres_frontier/test_stage1_smoke_env.py::"
    "test_post_reveal_endpoint_clearance_failure_is_severe_safety[hidden_rock]",
    "tests/ppo_highres_frontier/test_stage1_smoke_env.py::"
    "test_post_reveal_endpoint_clearance_failure_is_severe_safety[hidden_steep_terrain]",
    "tests/ppo_highres_frontier/test_stage1_smoke_env.py::"
    "test_post_reveal_distant_blocker_keeps_endpoint_safe",
    "tests/ppo_highres_frontier/test_stage1_smoke_env.py::"
    "test_conservative_smoke_contract_is_stable_across_python_hash_seeds",
    "tests/ppo_highres_frontier/test_stage1_smoke_env.py::"
    "test_ten_episode_workflow_writes_finite_append_only_machine_artifacts",
    "tests/ppo_highres_frontier/test_stage1_smoke_env.py::"
    "test_stage1_gate_binds_required_sources_without_writing_gate_artifact",
    "tests/ppo_highres_frontier/test_stage1_smoke_env.py::"
    "test_stage1_gate_fails_closed_on_missing_review_jump_state_and_artifact_drift",
    "tests/ppo_highres_frontier/test_stage1_smoke_env.py::"
    "test_stage1_gate_rejects_dirty_tree_and_has_no_force_path",
)


def _gate_modules():
    try:
        gate_artifacts = importlib.import_module("path_v2_gate_artifacts")
        runner = importlib.import_module("run_xunce_path_v2_g0_baseline_and_isolation")
    except ModuleNotFoundError as exc:
        raise AssertionError("Gate 0 production modules do not exist yet") from exc
    return gate_artifacts, runner


def _junit_case(nodeid: str, outcome: str = "passed") -> str:
    path, name = nodeid.split("::", 1)
    classname = path.removesuffix(".py").replace("/", ".")
    child = ""
    if outcome == "failed":
        child = '<failure message="inherited external baseline">failure</failure>'
    elif outcome == "error":
        child = '<error message="collection error">error</error>'
    elif outcome == "skipped":
        child = '<skipped message="optional dependency pydrake is unavailable" />'
    return f'<testcase classname="{classname}" name="{name}" time="0.001">{child}</testcase>'


def _write_junit(path: Path, cases: list[tuple[str, str]]) -> None:
    failures = sum(outcome == "failed" for _, outcome in cases)
    errors = sum(outcome == "error" for _, outcome in cases)
    skipped = sum(outcome == "skipped" for _, outcome in cases)
    body = "".join(_junit_case(nodeid, outcome) for nodeid, outcome in cases)
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<testsuites tests="{len(cases)}" failures="{failures}" errors="{errors}" skipped="{skipped}">'
        f'<testsuite name="pytest" tests="{len(cases)}" failures="{failures}" '
        f'errors="{errors}" skipped="{skipped}">{body}</testsuite></testsuites>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")


def _path_planner_cases() -> list[tuple[str, str]]:
    passed = [(f"tests/test_gate_fixture.py::test_pass_{index}", "passed") for index in range(156)]
    skipped = [(f"tests/test_gate_fixture.py::test_optional_{index}", "skipped") for index in range(17)]
    return passed + skipped


def _ppo_cases(extra_failure_nodeid: str | None = None) -> list[tuple[str, str]]:
    passed = [(f"tests/ppo_highres_frontier/test_stage1_smoke_env.py::test_pass_{index}", "passed") for index in range(43)]
    failed = [(nodeid, "failed") for nodeid in PPO_FAILURE_NODEIDS]
    if extra_failure_nodeid is not None:
        failed.append((extra_failure_nodeid, "failed"))
    return passed + failed


def _config(tmp_path: Path, extra_failure_nodeid: str | None = None) -> Path:
    output_root = tmp_path / "out"
    _write_junit(output_root / "path_planner_baseline.junit.xml", _path_planner_cases())
    _write_junit(output_root / "ppo_stage1_baseline.junit.xml", _ppo_cases(extra_failure_nodeid))
    known_failures = [
        {
            "nodeid": nodeid,
            "category": (
                "inherited_external_baseline_symptom"
                if index < 9
                else "frozen_foundation_identity_mismatch"
            ),
        }
        for index, nodeid in enumerate(PPO_FAILURE_NODEIDS)
    ]
    config = {
        "schema_version": "xunce-path-v2-g0-baseline-and-isolation/v1",
        "stage_id": "xunce-path-v2-g0-baseline-and-isolation",
        "python": str(Path(sys.executable).resolve()),
        "expected_git": {
            "branch": "codex/multiplatform-path-planner-v2",
            "base_commit": EXPECTED_BASE_COMMIT,
        },
        "temp_root": str(tmp_path / "runtime-temp"),
        "path_planner": {
            "working_directory": "path-planner",
            "pythonpath": ["path-planner/src"],
            "junit_xml": "path_planner_baseline.junit.xml",
            "expected": {"tests": 173, "passed": 156, "skipped": 17, "failures": 0, "errors": 0},
            "allowed_skip_dependency": "pydrake",
        },
        "ppo_stage1": {
            "working_directory": ".",
            "pythonpath": ["src", "path-planner/src"],
            "junit_xml": "ppo_stage1_baseline.junit.xml",
            "expected": {"tests": 56, "passed": 43, "skipped": 0, "failures": 13, "errors": 0},
            "known_failures": known_failures,
        },
        "boundaries": dict(BOUNDARY_FIELDS),
    }
    path = tmp_path / "gate0.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_gate0_accepts_green_v1_and_exact_external_allowlist(tmp_path: Path) -> None:
    _, runner = _gate_modules()
    summary = runner.run_gate0(
        config_path=_config(tmp_path),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
        execute_tests=False,
    )

    assert summary["status"] == "passed"
    assert summary["path_planner"] == {"passed": 156, "skipped": 17, "failed": 0, "errors": 0}
    assert summary["ppo_stage1"]["known_failure_count"] == 13
    assert summary["ppo_stage1"]["unexpected_failure_nodeids"] == []
    assert summary["replaces_default_policy"] is False
    assert (tmp_path / "out" / "manifest.json").is_file()


def test_gate0_rejects_one_extra_ppo_failure(tmp_path: Path) -> None:
    _, runner = _gate_modules()
    config_path = _config(tmp_path, extra_failure_nodeid="tests/example.py::test_extra")

    summary = runner.run_gate0(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
        execute_tests=False,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "restore_exact_ppo_stage1_external_baseline"
    assert summary["ppo_stage1"]["unexpected_failure_nodeids"] == ["tests/example.py::test_extra"]


def test_parse_junit_normalizes_pytest_classname_to_slash_nodeid(tmp_path: Path) -> None:
    _, runner = _gate_modules()
    junit = tmp_path / "nested.junit.xml"
    nodeid = PPO_FAILURE_NODEIDS[0]
    _write_junit(junit, [(nodeid, "failed"), ("tests/example.py::test_skip", "skipped")])

    parsed = runner.parse_junit(junit)

    assert parsed.tests == 2
    assert parsed.passed == 0
    assert parsed.failures == 1
    assert parsed.errors == 0
    assert parsed.skipped == 1
    assert parsed.failed_nodeids == (nodeid,)


def test_manifest_hashes_every_artifact_except_itself(tmp_path: Path) -> None:
    gate_artifacts, _ = _gate_modules()
    output_root = tmp_path / "artifacts"
    manifest = gate_artifacts.write_gate_artifacts(
        output_root=output_root,
        config={"schema_version": "config/v1"},
        summary={"status": "passed", **gate_artifacts.BOUNDARY_FIELDS},
        routing={"route": "gate0_passed"},
        rows=[{"suite": "path_planner", "status": "passed"}],
        phases=[{"phase": "baseline", "status": "completed"}],
        review={"status": "passed"},
        report="# Gate 0\n",
    )

    entries = {entry["relative_path"]: entry for entry in manifest["artifacts"]}
    assert "manifest.json" not in entries
    assert set(entries) == {
        "config.json",
        "phase-state.jsonl",
        "report.md",
        "results.jsonl",
        "review.json",
        "routing.json",
        "summary.json",
    }
    for relative_path, entry in entries.items():
        content = (output_root / relative_path).read_bytes()
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()
        assert entry["size"] == len(content)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def test_git_identity_requires_linked_clean_descendant_worktree(tmp_path: Path) -> None:
    _, runner = _gate_modules()
    primary = tmp_path / "primary"
    primary.mkdir()
    _git(primary, "init", "-b", "main")
    _git(primary, "config", "user.name", "Gate Test")
    _git(primary, "config", "user.email", "gate@example.invalid")
    tracked = primary / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    _git(primary, "add", "tracked.txt")
    _git(primary, "commit", "-m", "base")
    base_commit = _git(primary, "rev-parse", "HEAD")
    linked = tmp_path / "linked"
    _git(primary, "worktree", "add", "-b", "codex/gate0-test", str(linked))

    clean = runner.audit_git_identity(linked, "codex/gate0-test", base_commit)

    assert clean["status"] == "passed"
    assert clean["is_linked_worktree"] is True
    assert clean["merge_base"] == base_commit
    assert clean["clean_tree"] is True

    (linked / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    dirty = runner.audit_git_identity(linked, "codex/gate0-test", base_commit)
    assert dirty["status"] == "failed"
    assert dirty["clean_tree"] is False


def test_import_origin_audit_resolves_both_packages_from_explicit_worktree() -> None:
    _, runner = _gate_modules()
    audit = runner.audit_import_origins(Path(sys.executable), REPO_ROOT)

    assert audit["status"] == "passed"
    assert Path(audit["path_planner"]["origin"]).is_relative_to(REPO_ROOT / "path-planner" / "src")
    assert Path(audit["lunar_exploration_ppo"]["origin"]).is_relative_to(REPO_ROOT / "src")
    assert audit["python_no_user_site"] is True


def test_boundary_config_must_remain_fail_closed(tmp_path: Path) -> None:
    _, runner = _gate_modules()
    config_path = _config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["boundaries"]["publishes_checkpoint"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")

    summary = runner.run_gate0(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
        execute_tests=False,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "restore_gate0_safety_boundaries"
    assert summary["publishes_checkpoint"] is False


def test_boundary_config_rejects_numeric_zero_instead_of_boolean_false(tmp_path: Path) -> None:
    _, runner = _gate_modules()
    config_path = _config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["boundaries"]["publishes_checkpoint"] = 0
    config_path.write_text(json.dumps(config), encoding="utf-8")

    summary = runner.run_gate0(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
        execute_tests=False,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "restore_gate0_safety_boundaries"


def test_gate0_rejects_incorrect_known_failure_category(tmp_path: Path) -> None:
    _, runner = _gate_modules()
    config_path = _config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["ppo_stage1"]["known_failures"][0]["category"] = "frozen_foundation_identity_mismatch"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    summary = runner.run_gate0(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
        execute_tests=False,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "restore_exact_ppo_stage1_external_baseline"


def test_gate0_binds_each_known_failure_category_to_its_nodeid(tmp_path: Path) -> None:
    _, runner = _gate_modules()
    config_path = _config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = config["ppo_stage1"]["known_failures"]
    rows[0]["nodeid"], rows[9]["nodeid"] = rows[9]["nodeid"], rows[0]["nodeid"]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    summary = runner.run_gate0(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
        execute_tests=False,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "restore_exact_ppo_stage1_external_baseline"
