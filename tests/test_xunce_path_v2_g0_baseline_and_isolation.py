from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


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


ATOMIC_ARTIFACT_NAMES = {
    "config.json",
    "summary.json",
    "routing.json",
    "results.jsonl",
    "phase-state.jsonl",
    "review.json",
    "report.md",
    "manifest.json",
}


def _atomic_artifact_kwargs(output_root: Path) -> dict:
    return {
        "output_root": output_root,
        "config": {"schema_version": "config/v1"},
        "summary": {"status": "passed", **BOUNDARY_FIELDS},
        "routing": {"route": "gate-passed"},
        "rows": [{"suite": "gate", "status": "passed"}],
        "phases": [{"phase": "evidence", "status": "completed"}],
        "review": {"status": "passed"},
        "report": "# Gate evidence\n",
    }


def _atomic_staging_roots(output_root: Path) -> list[Path]:
    return sorted(
        output_root.parent.glob(f".{output_root.name}.staging-*"),
        key=lambda path: path.name,
    )


def test_atomic_gate_artifact_publish_succeeds_with_exact_valid_manifest(
    tmp_path: Path,
) -> None:
    gate_artifacts, _ = _gate_modules()
    output_root = tmp_path / "atomic-success"

    manifest = gate_artifacts.write_gate_artifacts_atomically(
        **_atomic_artifact_kwargs(output_root)
    )

    assert {path.name for path in output_root.iterdir()} == ATOMIC_ARTIFACT_NAMES
    assert all(path.is_file() for path in output_root.iterdir())
    stored_manifest = json.loads(
        (output_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest == stored_manifest
    assert stored_manifest == gate_artifacts.build_manifest_without_self_hash(
        output_root
    )
    assert stored_manifest["artifact_count"] == 7
    assert _atomic_staging_roots(output_root) == []


def test_atomic_gate_artifact_publish_creates_missing_parent_chain(
    tmp_path: Path,
) -> None:
    gate_artifacts, _ = _gate_modules()
    output_root = tmp_path / "missing" / "nested" / "atomic-success"
    assert not output_root.parent.exists()

    manifest = gate_artifacts.write_gate_artifacts_atomically(
        **_atomic_artifact_kwargs(output_root)
    )

    assert manifest == gate_artifacts.build_manifest_without_self_hash(output_root)
    assert {path.name for path in output_root.iterdir()} == ATOMIC_ARTIFACT_NAMES
    assert _atomic_staging_roots(output_root) == []


def test_atomic_gate_artifact_publish_never_reuses_colliding_staging_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gate_artifacts, _ = _gate_modules()
    output_root = tmp_path / "atomic-collision"
    colliding_root = output_root.parent / f".{output_root.name}.staging-collision"
    colliding_root.mkdir()
    sentinel = colliding_root / "sentinel.bin"
    sentinel.write_bytes(b"preserve-existing-staging")
    uuid_hexes = iter(("collision", "fresh"))

    class FixedUuid:
        def __init__(self, hex_value: str) -> None:
            self.hex = hex_value

    monkeypatch.setattr(
        gate_artifacts.uuid,
        "uuid4",
        lambda: FixedUuid(next(uuid_hexes)),
    )

    manifest = gate_artifacts.write_gate_artifacts_atomically(
        **_atomic_artifact_kwargs(output_root)
    )

    assert manifest == gate_artifacts.build_manifest_without_self_hash(output_root)
    assert {path.name for path in output_root.iterdir()} == ATOMIC_ARTIFACT_NAMES
    assert {path.name for path in colliding_root.iterdir()} == {"sentinel.bin"}
    assert sentinel.read_bytes() == b"preserve-existing-staging"
    assert _atomic_staging_roots(output_root) == [colliding_root]


def test_atomic_gate_artifact_publish_fails_stably_after_staging_collisions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gate_artifacts, _ = _gate_modules()
    output_root = tmp_path / "atomic-collision-exhausted"
    colliding_root = output_root.parent / f".{output_root.name}.staging-collision"
    colliding_root.mkdir()
    sentinel = colliding_root / "sentinel.bin"
    sentinel.write_bytes(b"preserve-all-collisions")
    uuid_call_count = 0

    class FixedUuid:
        hex = "collision"

    def colliding_uuid():
        nonlocal uuid_call_count
        uuid_call_count += 1
        return FixedUuid()

    monkeypatch.setattr(gate_artifacts.uuid, "uuid4", colliding_uuid)

    with pytest.raises(
        RuntimeError,
        match="atomic gate artifact staging reservation failed",
    ):
        gate_artifacts.write_gate_artifacts_atomically(
            **_atomic_artifact_kwargs(output_root)
        )

    assert uuid_call_count > 1
    assert not output_root.exists()
    assert {path.name for path in colliding_root.iterdir()} == {"sentinel.bin"}
    assert sentinel.read_bytes() == b"preserve-all-collisions"
    assert _atomic_staging_roots(output_root) == [colliding_root]


def test_atomic_gate_artifact_partial_write_keeps_unique_staging_and_no_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gate_artifacts, _ = _gate_modules()
    output_root = tmp_path / "atomic-partial"
    observed: list[Path] = []

    def partial_writer(*, output_root, **kwargs):
        staging_root = Path(output_root)
        observed.append(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)
        (staging_root / "config.json").write_bytes(b"partial")
        raise OSError("injected partial write")

    monkeypatch.setattr(gate_artifacts, "write_gate_artifacts", partial_writer)

    for _ in range(2):
        with pytest.raises(
            RuntimeError,
            match="atomic gate artifact staging write failed",
        ):
            gate_artifacts.write_gate_artifacts_atomically(
                **_atomic_artifact_kwargs(output_root)
            )

    assert not output_root.exists()
    assert len(observed) == 2
    assert len(set(observed)) == 2
    assert all(path.parent == output_root.parent for path in observed)
    assert _atomic_staging_roots(output_root) == sorted(
        observed,
        key=lambda path: path.name,
    )
    assert all((path / "config.json").read_bytes() == b"partial" for path in observed)


def test_atomic_gate_artifact_publish_rejects_invalid_staged_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gate_artifacts, _ = _gate_modules()
    output_root = tmp_path / "atomic-invalid-manifest"
    original_writer = gate_artifacts.write_gate_artifacts

    def invalid_manifest_writer(*, output_root, **kwargs):
        manifest = original_writer(output_root=output_root, **kwargs)
        (Path(output_root) / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "xunce-path-v2-gate-manifest/v1",
                    "artifact_count": 0,
                    "artifacts": [],
                }
            ),
            encoding="utf-8",
        )
        return manifest

    monkeypatch.setattr(
        gate_artifacts,
        "write_gate_artifacts",
        invalid_manifest_writer,
    )

    with pytest.raises(
        RuntimeError,
        match="atomic gate artifact staging validation failed",
    ):
        gate_artifacts.write_gate_artifacts_atomically(
            **_atomic_artifact_kwargs(output_root)
        )

    assert not output_root.exists()
    staging_roots = _atomic_staging_roots(output_root)
    assert len(staging_roots) == 1
    staging_root = staging_roots[0]
    assert {path.name for path in staging_root.iterdir()} == ATOMIC_ARTIFACT_NAMES
    assert json.loads(
        (staging_root / "manifest.json").read_text(encoding="utf-8")
    )["artifact_count"] == 0


def test_atomic_gate_artifact_publish_preserves_staging_when_rename_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gate_artifacts, _ = _gate_modules()
    output_root = tmp_path / "atomic-rename-failure"
    rename_calls: list[tuple[object, object]] = []

    def failed_rename(source, destination):
        rename_calls.append((source, destination))
        raise OSError("injected rename failure")

    monkeypatch.setattr(gate_artifacts.os, "rename", failed_rename)

    with pytest.raises(
        RuntimeError,
        match="atomic gate artifact publish rename failed",
    ):
        gate_artifacts.write_gate_artifacts_atomically(
            **_atomic_artifact_kwargs(output_root)
        )

    assert len(rename_calls) == 1
    assert not output_root.exists()
    staging_roots = _atomic_staging_roots(output_root)
    assert len(staging_roots) == 1
    assert {path.name for path in staging_roots[0].iterdir()} == ATOMIC_ARTIFACT_NAMES


def test_atomic_gate_artifact_publish_does_not_overwrite_racing_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gate_artifacts, _ = _gate_modules()
    output_root = tmp_path / "atomic-race"
    original_writer = gate_artifacts.write_gate_artifacts
    rename_called = False

    def racing_writer(*, output_root: Path, **kwargs):
        manifest = original_writer(output_root=output_root, **kwargs)
        race_root = tmp_path / "atomic-race"
        race_root.mkdir()
        (race_root / "racer.bin").write_bytes(b"racer-wins")
        return manifest

    def forbidden_rename(source, destination):
        nonlocal rename_called
        rename_called = True
        raise AssertionError("rename must not run after target appears")

    monkeypatch.setattr(gate_artifacts, "write_gate_artifacts", racing_writer)
    monkeypatch.setattr(gate_artifacts.os, "rename", forbidden_rename)

    with pytest.raises(
        RuntimeError,
        match="atomic gate artifact publish target appeared",
    ):
        gate_artifacts.write_gate_artifacts_atomically(
            **_atomic_artifact_kwargs(output_root)
        )

    assert rename_called is False
    assert {path.name for path in output_root.iterdir()} == {"racer.bin"}
    assert (output_root / "racer.bin").read_bytes() == b"racer-wins"
    staging_roots = _atomic_staging_roots(output_root)
    assert len(staging_roots) == 1
    assert {path.name for path in staging_roots[0].iterdir()} == ATOMIC_ARTIFACT_NAMES


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


GATE5_MANIFEST_METADATA = {
    "status": "blocked",
    "execution_class": "blocked_profile_freeze",
    "primary_blocker": "freeze_hopper_simulation_proxy_profile_parameters",
    "formal_evidence_eligible": False,
    "formal_row_count": 0,
    "parameter_set_id": None,
}


def test_manifest_optional_metadata_is_exact_and_hashes_remain_byte_derived(
    tmp_path: Path,
) -> None:
    gate_artifacts, _ = _gate_modules()
    kwargs = _atomic_artifact_kwargs(tmp_path / "gate5")
    manifest = gate_artifacts.write_gate_artifacts_atomically(
        **kwargs,
        manifest_metadata=GATE5_MANIFEST_METADATA,
    )

    assert {key: manifest[key] for key in GATE5_MANIFEST_METADATA} == GATE5_MANIFEST_METADATA
    assert manifest == gate_artifacts.build_manifest_without_self_hash(
        tmp_path / "gate5",
        manifest_metadata=GATE5_MANIFEST_METADATA,
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {key: value for key, value in GATE5_MANIFEST_METADATA.items() if key != "formal_row_count"},
        {**GATE5_MANIFEST_METADATA, "extra": False},
        {**GATE5_MANIFEST_METADATA, "formal_evidence_eligible": True},
    ],
    ids=["missing", "extra", "conflicting"],
)
def test_manifest_gate5_metadata_fails_closed(metadata: dict, tmp_path: Path) -> None:
    gate_artifacts, _ = _gate_modules()
    with pytest.raises(ValueError, match="manifest metadata contract"):
        gate_artifacts.write_gate_artifacts_atomically(
            **_atomic_artifact_kwargs(tmp_path / "invalid"),
            manifest_metadata=metadata,
        )


def test_manifest_existing_call_shape_and_bytes_remain_unchanged(tmp_path: Path) -> None:
    gate_artifacts, _ = _gate_modules()
    output_root = tmp_path / "legacy"
    manifest = gate_artifacts.write_gate_artifacts(
        **_atomic_artifact_kwargs(output_root)
    )

    assert set(manifest) == {"schema_version", "artifact_count", "artifacts"}
    assert manifest == gate_artifacts.build_manifest_without_self_hash(output_root)
