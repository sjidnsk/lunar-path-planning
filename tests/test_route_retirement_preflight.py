from pathlib import Path
import hashlib
import inspect
import json
import re
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_route_retirement_preflight import build_manifest, classify_path, load_policy, validate_graph


def test_protected_rules_win_over_retirement_patterns() -> None:
    policy = load_policy(Path("configs/route_retirement_policy_v1.json"))
    assert classify_path("scripts/xunce_artifact_io.py", policy).classification == "protected"
    assert classify_path("scripts/run_xunce_mid_dual_g3_closed_loop.py", policy).classification == "protected"
    assert classify_path("path-planner/src/path_planner/search/astar.py", policy).classification == "protected"
    assert classify_path("dev-platform-constraints/src/dev_platform_constraints/core/contracts.py", policy).classification == "protected"


def test_retired_and_manual_review_boundaries_are_distinct() -> None:
    policy = load_policy(Path("configs/route_retirement_policy_v1.json"))
    assert classify_path("scripts/run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke.py", policy).classification == "retire_candidate"
    assert classify_path("scripts/run_path_feedback_validation.py", policy).classification == "retire_candidate"
    assert classify_path("model-explorer", policy).classification == "manual_review"
    assert classify_path("visual-workbench", policy).classification == "manual_review"
    assert classify_path("configs/stage_registry.json", policy).classification == "manual_review"


def test_validates_the_frozen_knowledge_graph() -> None:
    summary = validate_graph(Path("D:/codex/project/lunar-path-planning/.ua/knowledge-graph.json"))
    assert (summary.analyzed_files, summary.nodes, summary.edges, summary.layers) == (252, 1480, 2710, 7)


def test_audit_module_has_no_delete_capability() -> None:
    import audit_route_retirement_preflight as module

    source = inspect.getsource(module).lower()
    for forbidden in ("unlink(", "rmdir(", "remove-item", "rm -rf", "git rm", "shutil.rmtree"):
        assert forbidden not in source


def test_actual_stage18_to_26_inventory_is_retired() -> None:
    import audit_route_retirement_preflight as module

    policy = load_policy(Path("configs/route_retirement_policy_v1.json"))
    stage_pattern = re.compile(r"(?:^|[/_-])stage(?:18|19|2[0-6])(?:[a-z0-9_-])", re.IGNORECASE)
    inventory = [path for path in module.list_tracked_paths(Path(".")) if stage_pattern.search(path)]
    assert inventory
    assert [
        (path, classify_path(path, policy).classification)
        for path in inventory
        if classify_path(path, policy).classification != "retire_candidate"
    ] == []


def test_multiplatform_planner_v3_inventory_is_protected() -> None:
    import audit_route_retirement_preflight as module

    policy = load_policy(Path("configs/route_retirement_policy_v1.json"))
    inventory = [
        path
        for path in module.list_tracked_paths(Path("."))
        if "multiplatform" in path.lower() and "planner" in path.lower() and "v3" in path.lower()
    ]
    assert inventory
    assert {classify_path(path, policy).classification for path in inventory} == {"protected"}


def test_retained_stage6_and_midterm_gates_are_never_retirement_candidates() -> None:
    import audit_route_retirement_preflight as module

    policy = load_policy(Path("configs/route_retirement_policy_v1.json"))
    inventory = [
        path
        for path in module.list_tracked_paths(Path("."))
        if "stage6" in path.lower() or "mid_dual" in path.lower() or path.startswith("independent/g2_t2_producer/")
    ]
    assert inventory
    assert [path for path in inventory if classify_path(path, policy).classification == "retire_candidate"] == []


@pytest.mark.parametrize("path", ["C:/repo/file.py", "../file.py", "scripts/../../file.py"])
def test_classification_rejects_paths_outside_the_repository(path: str) -> None:
    policy = load_policy(Path("configs/route_retirement_policy_v1.json"))
    with pytest.raises(ValueError, match="repository path must be relative"):
        classify_path(path, policy)


def test_bundle_verification_requires_the_expected_commit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import audit_route_retirement_preflight as module

    bundle_path = tmp_path / "fixture.bundle"
    bundle_path.write_bytes(b"valid bundle fixture")
    expected_hash = hashlib.sha256(bundle_path.read_bytes()).hexdigest()

    def fake_run(args: tuple[str, ...], cwd: Path, *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
        if tuple(args[:3]) == ("git", "bundle", "verify"):
            return subprocess.CompletedProcess(args, 0, b"", b"")
        if tuple(args[:3]) == ("git", "bundle", "list-heads"):
            return subprocess.CompletedProcess(args, 0, b"2222222222222222222222222222222222222222 HEAD\n", b"")
        raise AssertionError(args)

    monkeypatch.setattr(module, "_run", fake_run)
    audit = module._audit_backups(
        tmp_path,
        tmp_path,
        expected_hashes={"fixture.bundle": expected_hash},
        expected_commits={"fixture.bundle": "1111111111111111111111111111111111111111"},
    )
    assert audit["all_verified"] is False
    assert audit["records"][0]["contains_expected_commit"] is False
    assert audit["records"][0]["status"] == "missing_expected_commit"


def test_baseline_gitlink_mismatch_is_blocking(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import audit_route_retirement_preflight as module

    tree = b"160000 commit 2222222222222222222222222222222222222222\tcomponent\n"
    monkeypatch.setattr(
        module,
        "_run",
        lambda args, cwd, input_bytes=None: subprocess.CompletedProcess(args, 0, tree, b""),
    )
    audit = module._audit_baseline_gitlinks(
        tmp_path,
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        {"component": "1111111111111111111111111111111111111111"},
    )
    assert audit["all_match"] is False
    assert audit["records"][0]["status"] == "mismatch"


@pytest.mark.parametrize(
    ("tags", "stdout", "expected"),
    [
        (
            ["archive"],
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\trefs/tags/archive\n"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\trefs/tags/archive^{}\n",
            "pushed",
        ),
        (
            ["archive"],
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\trefs/tags/archive\n"
            "cccccccccccccccccccccccccccccccccccccccc\trefs/tags/archive^{}\n",
            "mismatch",
        ),
        (
            ["archive", "second"],
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\trefs/tags/archive\n",
            "partial",
        ),
        (["archive"], "", "not_pushed"),
    ],
)
def test_remote_tag_status_compares_peeled_commits(tags: list[str], stdout: str, expected: str) -> None:
    import audit_route_retirement_preflight as module

    assert module._classify_remote_tag_status(
        tags,
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        stdout.encode("ascii"),
    ) == expected


@pytest.mark.parametrize("permission_mode", ["exception", "stderr"])
def test_ignored_output_permission_failures_are_stable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    permission_mode: str,
) -> None:
    import audit_route_retirement_preflight as module

    if permission_mode == "exception":
        def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            raise PermissionError("denied")
    else:
        def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess([], 1, b"", b"fatal: Permission denied")

    monkeypatch.setattr(module, "_run", fake_run)
    assert module._audit_ignored_outputs(tmp_path) == {"status": "permission_denied", "paths": []}


def test_production_cli_rejects_noncanonical_backup_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import audit_route_retirement_preflight as module

    monkeypatch.setattr(module, "build_manifest", lambda *args: (_ for _ in ()).throw(AssertionError("must not build")))
    output_root = tmp_path / "out"
    with pytest.raises(ValueError, match="backup root"):
        module.main(
            [
                "--repo-root", ".",
                "--policy", "configs/route_retirement_policy_v1.json",
                "--knowledge-graph", "D:/codex/project/lunar-path-planning/.ua/knowledge-graph.json",
                "--backup-root", str(tmp_path / "wrong-backup"),
                "--output-root", str(output_root),
                "--baseline-commit", "6a4c2dd0352fd6c1918a5eef39c9783b9d3c5c65",
            ]
        )
    assert not output_root.exists()


def test_cli_writes_only_five_stable_artifacts_below_output_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import audit_route_retirement_preflight as module

    manifest = {
        "graph": {
            "git_commit_hash": "1682d7f9f755eb59c85e6d0ceaaa6f6d5d203dfd",
            "analyzed_files": 252,
            "nodes": 1480,
            "edges": 2710,
            "layers": 7,
            "is_ancestor_of_baseline": True,
        },
        "records": [
            {
                "path": "legacy.py",
                "classification": "retire_candidate",
                "matched_rules": ["legacy.py"],
                "tracked": True,
                "ignored": False,
                "blocking_references": [],
            }
        ],
        "blocking_reasons": [],
        "reference_audit": {"legacy.py": {"rg": [], "graph_edges": []}},
        "backups": {"all_verified": True, "records": []},
        "baseline_tree": {"all_match": True, "records": []},
        "ignored_outputs": {"status": "ok", "paths": []},
    }
    monkeypatch.setattr(module, "_validate_fixed_cli_inputs", lambda *args: None, raising=False)
    monkeypatch.setattr(module, "_resolve_commit", lambda *args: "6a4c2dd0352fd6c1918a5eef39c9783b9d3c5c65")
    monkeypatch.setattr(module, "build_manifest", lambda *args: json.loads(json.dumps(manifest)))
    monkeypatch.setattr(module, "_is_ancestor", lambda *args: True)
    monkeypatch.setattr(module, "_local_tags", lambda *args: ["archive"])
    monkeypatch.setattr(module, "_remote_tag_status", lambda *args: "not_pushed")
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    output_root = tmp_path / "out"
    argv = [
        "--repo-root", str(tmp_path),
        "--policy", str(tmp_path / "policy.json"),
        "--knowledge-graph", str(tmp_path / "graph.json"),
        "--backup-root", str(tmp_path / "backups"),
        "--output-root", str(output_root),
        "--baseline-commit", "6a4c2dd0352fd6c1918a5eef39c9783b9d3c5c65",
    ]
    assert module.main(argv) == 0
    first = {path.name: path.read_bytes() for path in output_root.iterdir()}
    assert sorted(first) == [
        "backup-manifest.json",
        "candidate-manifest.json",
        "reference-audit.json",
        "report.md",
        "summary.json",
    ]
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert module.main(argv) == 0
    assert {path.name: path.read_bytes() for path in output_root.iterdir()} == first


def test_build_manifest_binds_fixed_bundle_commits_and_baseline_gitlinks() -> None:
    manifest = build_manifest(
        Path("."),
        Path("configs/route_retirement_policy_v1.json"),
        Path("D:/codex/project/lunar-path-planning/.ua/knowledge-graph.json"),
        Path("D:/CodexDownloads/lunar-path-planning-backups/2026-08-01-6a4c2dd"),
    )
    assert manifest["baseline_tree"]["all_match"] is True
    assert manifest["backups"]["all_verified"] is True
    assert {
        record["name"]: (record["expected_commit"], record["contains_expected_commit"])
        for record in manifest["backups"]["records"]
    } == {
        "dev-platform-constraints.bundle": ("61e9fa8afd09db83632456bdcf181c222ee13513", True),
        "lunar-path-planning.bundle": ("6a4c2dd0352fd6c1918a5eef39c9783b9d3c5c65", True),
        "model-explorer.bundle": ("b547a997d94ad199c822136d1ae345e180b87ca7", True),
        "path-planner.bundle": ("2f6378d3c47da027c0d4146d94cab881b8f2a594", True),
        "visual-workbench.bundle": ("9acc4ce83fc2884221a9759fa00824692eab3315", True),
    }


def test_build_manifest_blocks_overall_backup_verification_on_gitlink_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import audit_route_retirement_preflight as module

    monkeypatch.setattr(module, "_audit_baseline_gitlinks", lambda *args: {"all_match": False, "records": []})
    monkeypatch.setattr(module, "list_tracked_paths", lambda *args: ())
    monkeypatch.setattr(module, "_audit_ignored_outputs", lambda *args: {"status": "ok", "paths": []})
    monkeypatch.setattr(module, "audit_references", lambda *args: {})
    monkeypatch.setattr(
        module,
        "_audit_backups",
        lambda *args: {"all_verified": True, "records": [], "hash_manifest_matches_expected": True},
    )
    manifest = build_manifest(
        Path("."),
        Path("configs/route_retirement_policy_v1.json"),
        Path("D:/codex/project/lunar-path-planning/.ua/knowledge-graph.json"),
        Path("D:/CodexDownloads/lunar-path-planning-backups/2026-08-01-6a4c2dd"),
    )
    assert manifest["backups"]["bundles_all_verified"] is True
    assert manifest["backups"]["all_verified"] is False
